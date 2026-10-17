"""OCR 工具：只负责识别文本和坐标，不做敏感判断，不修改原图。

引擎适配：
- rapid（默认）：rapidocr 3.x，轻量纯 pip；
- paddle：PaddleOCR 2.x/3.x，精度更高，需 py311。

两种引擎的返回形态不同，统一归一化为 OcrResult。
"""
from __future__ import annotations

import logging
import math
import threading
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from src.config import Settings
from src.schemas.ocr_result import OcrResult, OcrTextBlock

logger = logging.getLogger(__name__)


def _load_image(
    image_path: Optional[str], image_bytes: Optional[bytes]
) -> Tuple[Image.Image, str]:
    """优先使用字节流；只有路径时读文件。返回 (PIL 图, 来源标识)。"""
    if image_bytes:
        img = Image.open(BytesIO(image_bytes))
        img.load()
        return img, "bytes"
    if image_path:
        data = Path(image_path).read_bytes()
        img = Image.open(BytesIO(data))
        img.load()
        return img, image_path
    raise ValueError("image_path 与 image_bytes 不能同时为空")


def _quad_to_xyxy(quad) -> List[int]:
    """4 点坐标 [[x,y] x4] 或 ndarray 转为外接矩形 [x1, y1, x2, y2]。"""
    xs = [float(p[0]) for p in quad]
    ys = [float(p[1]) for p in quad]
    return [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]


def _quad_points(quad) -> List[List[int]]:
    """四边形顶点取整 [[x, y] x4]。"""
    return [[round(float(p[0])), round(float(p[1]))] for p in quad]


def _quad_angle(quad) -> float:
    """文字行方向角（度，[-180, 180)）。

    检测器顶点顺序为 [左上, 右上, 右下, 左下]，取上边与右边中
    较长的一条作为文字行方向。0=水平，90=从上往下竖排，
    180/-180=倒置，-90=从下往上竖排。
    """
    pts = [(float(p[0]), float(p[1])) for p in quad]
    if len(pts) < 4:
        return 0.0
    edges = (
        (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]),
        (pts[2][0] - pts[1][0], pts[2][1] - pts[1][1]),
    )
    dx, dy = max(edges, key=lambda e: e[0] ** 2 + e[1] ** 2)
    if dx == 0 and dy == 0:
        return 0.0
    angle = math.degrees(math.atan2(dy, dx))
    if angle >= 180:
        angle -= 360
    if angle < -180:
        angle += 360
    return angle


def _box_to_xyxy(box) -> List[int]:
    """已是 [x1, y1, x2, y2] 的框（如 PaddleOCR rec_boxes）直接取整。"""
    values = [int(round(float(v))) for v in list(box)[:4]]
    return values


class OcrTool:
    """OCR 工具门面：引擎可配置，输出统一为 OcrResult。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine_name = settings.ocr_engine
        self._engine = None
        # 方向矫正引擎（懒加载，缺失时自动降级）
        self._orientation_engine = None
        self._orientation_failed = False
        self._np = None
        # 批量模式下多个 case 线程共用同一实例，OCR 推理引擎
        # （onnxruntime 会话 + 预处理缓冲）非严格线程安全，串行化调用。
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 引擎

    def _get_rapid(self):
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def _get_paddle(self):
        if self._engine is None:
            from paddleocr import PaddleOCR

            try:  # 3.x 参数名
                self._engine = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwrapping=False,
                    use_textline_orientation=False,
                )
            except TypeError:  # 2.x 参数名
                self._engine = PaddleOCR(use_angle_cls=False, lang="ch")
        return self._engine

    def _run_rapid(self, image: Image.Image) -> List[Tuple[object, str, float]]:
        import numpy as np

        engine = self._get_rapid()
        arr = np.array(image.convert("RGB"))
        res = engine(arr)
        return self._parse_rapidocr(res)

    @staticmethod
    def _parse_rapidocr(res) -> List[Tuple[object, str, float]]:
        """兼容 rapidocr 3.x（RapidOCROutput 对象）与 2.x（tuple/list）。
        返回原始四边形（4x2），由调用方计算外接矩形与角度。"""
        if res is None:
            return []
        # 3.x：对象带 txts / scores / boxes（4x2 ndarray）
        txts = getattr(res, "txts", None)
        if txts is not None:
            boxes = getattr(res, "boxes", None)
            scores = getattr(res, "scores", None)
            boxes = list(boxes) if boxes is not None else []
            scores = list(scores) if scores is not None else []
            return [
                (boxes[i], str(txts[i]), float(scores[i]))
                for i in range(len(txts))
            ]
        # 2.x：可能包了一层 (result, elapse)
        if isinstance(res, tuple):
            res = res[0]
        if not res:
            return []
        out: List[Tuple[object, str, float]] = []
        for item in res:
            box, text, score = item[0], item[1], item[2]
            out.append((box, str(text), float(score)))
        return out

    def _run_paddle(self, image: Image.Image) -> List[Tuple[object, str, float]]:
        import numpy as np

        engine = self._get_paddle()
        arr = np.array(image.convert("RGB"))
        try:  # 3.x
            raw = engine.predict(arr)
        except AttributeError:  # 2.x
            raw = engine(arr, cls=False)
        return self._parse_paddleocr(raw)

    @staticmethod
    def _parse_paddleocr(raw) -> List[Tuple[object, str, float]]:
        """兼容 PaddleOCR 3.x（dict 列表）与 2.x（嵌套 list）。"""
        out: List[Tuple[object, str, float]] = []
        if not raw:
            return out
        first = raw[0]
        if isinstance(first, dict):  # 3.x：rec_texts / rec_scores / rec_boxes(xyxy)
            for page in raw:
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                boxes = page.get("rec_boxes")
                if boxes is None:
                    boxes = page.get("dt_polys")
                if boxes is None:
                    boxes = []
                for i, text in enumerate(texts):
                    quad = boxes[i]
                    quad_list = list(quad)
                    if len(quad_list) == 4 and hasattr(quad_list[0], "__len__"):
                        out.append((quad_list, str(text), float(scores[i])))  # dt_polys：4 点
                    else:  # rec_boxes：xyxy → 造 4 顶点矩形
                        x1, y1, x2, y2 = _box_to_xyxy(quad_list)
                        rect = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        out.append((rect, str(text), float(scores[i])))
        else:  # 2.x：[[quad, (text, score)], ...]
            for line in raw:
                for item in line:
                    quad, (text, score) = item[0], item[1]
                    out.append((list(quad), str(text), float(score)))
        return out

    # ------------------------------------------------------------------ 对外

    def extract(
        self, image_path: Optional[str] = None, image_bytes: Optional[bytes] = None
    ) -> OcrResult:
        """识别文字块与坐标。失败时抛异常，由节点落地为失败状态。

        开启 ocr_orientation_correct 时：先用 RapidOrientation 检测
        全局方向，非 0° 则转正后识别，**坐标保持转正图坐标系**，
        rotation 记录旋转度数、aligned_image 附带转正图字节——
        下游 OCR/VLM/脱敏/质检全部在转正图上进行，最后由 finalize
        节点统一逆旋转回原方向。未开启时行为与之前完全一致。
        """
        image, source = _load_image(image_path, image_bytes)

        # 共享引擎（RapidOCR / PaddleOCR）内部持有推理会话与可复用缓冲，
        # 批量模式多线程并发调用可能破坏状态，这里串行化引擎交互。
        with self._lock:
            rotation = 0
            ocr_image = image
            if self._settings.ocr_orientation_correct:
                rotation = self._detect_orientation(image)
                if rotation:
                    # RapidOrientation 返回值 = 需逆时针旋转的度数；
                    # PIL rotate(正角度) 即视觉逆时针
                    ocr_image = image.rotate(rotation, expand=True)
                    logger.info("方向矫正：原图需逆时针旋转 %d°，已转正后识别", rotation)

            if self._engine_name == "paddle":
                raw_items = self._run_paddle(ocr_image)
            else:
                raw_items = self._run_rapid(ocr_image)

        blocks: List[OcrTextBlock] = []
        for quad, text, score in raw_items:
            text = text.strip()
            if not text:
                continue
            blocks.append(
                OcrTextBlock(
                    block_id=f"b{len(blocks)}",
                    text=text,
                    bbox=_quad_to_xyxy(quad),
                    points=_quad_points(quad),
                    angle=round(_quad_angle(quad), 2),
                    confidence=min(max(float(score), 0.0), 1.0),
                )
            )

        aligned_bytes: Optional[bytes] = None
        if rotation:
            buf = BytesIO()
            ocr_image.save(buf, format="PNG")
            aligned_bytes = buf.getvalue()

        result = OcrResult(
            engine=self._engine_name,
            width=ocr_image.width,
            height=ocr_image.height,
            blocks=blocks,
            image_path=image_path,
            rotation=rotation,
            aligned_image=aligned_bytes,
        )
        logger.info(
            "OCR[%s] 识别 %d 个文字块（转正图坐标系 %dx%d），来源 %s",
            self._engine_name, len(blocks), result.width, result.height, source,
        )
        return result

    # ------------------------------------------------------------------ 方向矫正

    def _detect_orientation(self, image: Image.Image) -> int:
        """RapidOrientation 检测全局方向，返回需**逆时针**旋转的角度
        （0/90/180/270）。

        注意：OCR 栈对旋转文本鲁棒、四边形角度也被归一化，无法从
        识别结果反推转向——语义以 RapidOrientation 输出为准，已用
        特征点像素测试固化。

        依赖缺失或推理失败时降级为 0（原图直跑），不让可选增强变成故障点。
        """
        if self._orientation_failed:
            return 0
        if self._orientation_engine is None:
            try:
                import numpy as np
                from rapid_orientation import RapidOrientation

                self._orientation_engine = RapidOrientation()
                self._np = np
            except Exception as exc:
                logger.warning(
                    "rapid-orientation 不可用，跳过方向矫正（%s）；"
                    "安装：pip install rapid-orientation",
                    exc,
                )
                self._orientation_failed = True
                return 0
        try:
            arr = self._np.array(image.convert("RGB"))
            angle, _cost = self._orientation_engine(arr)
            # RapidOrientation 可能返回 int / float / numpy 标量，统一转角度。
            # 直接 int(str(angle)) 在 float（如 90.0）时会抛 ValueError，
            # 导致方向矫正被永久禁用（部署开启该功能时的隐患）。
            return int(round(float(angle))) % 360
        except Exception as exc:
            logger.warning("方向检测失败，跳过矫正：%s", exc)
            self._orientation_failed = True
            return 0
