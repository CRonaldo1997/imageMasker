"""脱敏工具：在原图上执行脱敏，必须保留原图。

两种模式：
- white_mask（默认）：白底覆盖后，在原 bbox 上重绘假数据。
  只替换敏感值本身，保留"姓名："等标签，业务含义不丢失；
  相同敏感值经 FakeDataMapper 映射到相同假数据（确定性、等长）；
- blur / mosaic：直接模糊/马赛克，不重绘（模糊区写宇无意义）。

坐标一律使用 OCR/VLM 传入的 bbox，不做修改。
不做二维码/条码检测与打码（已按需求移除：QR 内容通常为票据号/查验 URL，
按当前脱敏范围不属于敏感信息，且检测在实拍票据上耗时且常误检）。
"""
from __future__ import annotations

import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.config import PROJECT_ROOT, Settings
from src.fake_data import FakeDataMapper
from src.schemas.redaction_result import RedactedBlock, RedactionResult
from src.schemas.vlm_analysis import RedactionMethod, VlmAnalysis

logger = logging.getLogger(__name__)

_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
)

_BASE_FONT_SIZE = 48  # 基准字号，按 bbox 高度比例缩放

#: 扩展名 → PIL 编码格式。未知扩展名一律按 PNG 落盘。
_FORMAT_BY_SUFFIX: Dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
}

#: 无损/有损格式的重编码质量（JPEG/WEBP 生效）
_OUTPUT_QUALITY = 95


def format_for_suffix(suffix: str) -> str:
    """输出扩展名决定编码格式，保证「扩展名与内容格式一致」。"""
    return _FORMAT_BY_SUFFIX.get((suffix or "").lower(), "PNG")


def encode_image(img: "Image.Image", fmt: str) -> bytes:
    """按目标格式编码图片字节。

    历史实现只区分 .jpg 与「其它」：.tif/.bmp/.webp 会把 PNG 字节写进
    .tif 等文件，下游按扩展名处理时出现「文件头不是该格式」的问题
    （同踩坑 #15 的 .jpg 版本，只是输入扩展名更多）。
    """
    buf = BytesIO()
    fmt = (fmt or "PNG").upper()
    if fmt == "JPEG":
        img.convert("RGB").save(
            buf, format="JPEG", quality=_OUTPUT_QUALITY, optimize=True
        )
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", quality=_OUTPUT_QUALITY)
    elif fmt == "TIFF":
        img.save(buf, format="TIFF", compression="tiff_deflate")
    elif fmt == "BMP":
        img.save(buf, format="BMP")
    else:
        img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


class RedactionTool:
    """图像脱敏工具。输入原图字节与 VLM 分析，输出脱敏图与命中记录。"""

    def __init__(self, settings: Settings, mapper: Optional[FakeDataMapper] = None) -> None:
        self._settings = settings
        self._mapper = mapper or FakeDataMapper()
        self._font_cache: Dict[int, ImageFont.FreeTypeFont] = {}
        self._font_path: Optional[str] = self._detect_font_path()
        if not self._font_path:
            logger.warning("未找到中文字体，假数据重绘将退化为纯白底覆盖")

    # ------------------------------------------------------------------ 对外

    def apply(
        self,
        image: bytes,
        vlm_analysis: VlmAnalysis,
        output_path: Optional[str] = None,
    ) -> RedactionResult:
        """执行脱敏。原图字节不被修改；结果落盘到 output_path
        （批量模式由调用方指定以保持目录结构），未指定时用 output_dir。"""
        img = Image.open(BytesIO(image)).convert("RGB")
        targets: List[Tuple[List[int], str, str, str, str, List[List[int]], float]] = []
        # (bbox, method, block_id, 绘制文本, source, points, angle)

        for sb in vlm_analysis.sensitive_blocks:
            if not sb.bbox:
                logger.warning("敏感块 %s 缺少 bbox，无法定位，已跳过", sb.block_id)
                continue
            draw_text = self._build_draw_text(sb.text, sb.value, sb.kind, sb.method)
            targets.append(
                (
                    list(sb.bbox),
                    sb.method.value,
                    sb.block_id,
                    draw_text,
                    sb.source,
                    [list(p) for p in sb.points],
                    sb.angle,
                )
            )

        for bbox, method, block_id, draw_text, source, points, angle in targets:
            self._apply_one(img, bbox, method, draw_text, points, angle)

        records = [
            RedactedBlock(
                block_id=block_id,
                text=draw_text or "",
                bbox=bbox,
                points=points,
                angle=angle,
                method=method,
                source=source,
            )
            for bbox, method, block_id, draw_text, source, points, angle in targets
        ]

        buf = BytesIO()
        img.save(buf, format="PNG")
        redacted_bytes = buf.getvalue()
        saved_path, final_bytes = self._save(redacted_bytes, output_path)

        return RedactionResult(
            redacted_image=final_bytes,
            output_path=str(saved_path),
            image_format="PNG" if not output_path else Path(output_path).suffix.lstrip(".").upper(),
            width=img.width,
            height=img.height,
            redacted_blocks=records,
        )

    # ------------------------------------------------------------------ 假数据

    def _build_draw_text(self, text: str, value: str, kind: str, method) -> str:
        """生成 bbox 内要绘制的文本。

        有 value 且 method 为白底时：把文本中的敏感值替换为等长假数据，
        保留标签等业务上下文；否则返回空串（纯覆盖，不重绘）。
        """
        if method != RedactionMethod.WHITE_MASK:
            return ""
        if not value or value not in text:
            return ""
        if not self._settings.fake_data_enabled:
            return ""
        fake = self._mapper.map(value, kind or "generic")
        return text.replace(value, fake)

    # ------------------------------------------------------------------ 绘制

    def _apply_one(
        self,
        img: Image.Image,
        bbox: List[int],
        method: str,
        draw_text: str,
        points: Optional[List[List[int]]] = None,
        angle: float = 0.0,
    ) -> None:
        x1, y1, x2, y2 = self._padded(bbox, img.size)
        if x2 <= x1 or y2 <= y1:
            return
        if method == RedactionMethod.BLUR.value:
            region = img.crop((x1, y1, x2, y2)).filter(
                ImageFilter.GaussianBlur(radius=self._settings.blur_radius)
            )
            img.paste(region, (x1, y1))
        elif method == RedactionMethod.MOSAIC.value:
            factor = max(2, self._settings.mosaic_factor)
            w, h = x2 - x1, y2 - y1
            region = img.crop((x1, y1, x2, y2))
            small = region.resize((max(1, w // factor), max(1, h // factor)))
            img.paste(small.resize((w, h), Image.NEAREST), (x1, y1))
        else:  # white_mask（及未知方式）
            if points and len(points) >= 3:
                # 四边形精确覆盖：倾斜/竖排文字不误伤相邻行
                ImageDraw.Draw(img).polygon(
                    [tuple(p) for p in points], fill=self._settings.mask_color_rgb
                )
            else:
                ImageDraw.Draw(img).rectangle(
                    (x1, y1, x2, y2), fill=self._settings.mask_color_rgb
                )
            if draw_text:
                self._draw_text(
                    img, (x1, y1, x2, y2), draw_text, angle, points
                )

    def _draw_text(
        self,
        img: Image.Image,
        box: Tuple[int, int, int, int],
        text: str,
        angle: float = 0.0,
        points: Optional[List[List[int]]] = None,
    ) -> None:
        """在文字区域居中绘制假数据。

        文字方向跟随 OCR 检测器给出的行方向角 angle：
        0=水平；90=竖排（从上往下）；180=倒置。做法是渲染横排
        文字到透明层，旋转 -angle 后贴到区域中心——原图文字什么
        方向，假数据就什么方向。
        """
        if not self._font_path:
            return  # 无中文字体：保留纯白底覆盖
        draw = ImageDraw.Draw(img)
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1

        # 横排渲染的初始字号：按行高比例（竖排时行高=区域宽度）
        line_thickness = height if abs(angle) < 45 else width
        size = max(9, int(line_thickness * self._settings.fake_font_size_ratio))
        font = self._font_at(size)

        # 假数据沿行方向的长度不能超过区域长度
        line_length = width if abs(angle) < 45 else height
        while font.size > 9:
            text_bbox = draw.textbbox((0, 0), text, font=font)
            if text_bbox[2] - text_bbox[0] <= line_length:
                break
            font = self._font_at(font.size - 1)

        # 透明层要足够容纳旋转后的文字（按区域对角线开方）
        side = int((width**2 + height**2) ** 0.5) + 8
        layer = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer)
        layer_draw.text(
            (side / 2, side / 2),
            text,
            font=font,
            fill=self._settings.text_color_rgb + (255,),
            anchor="mm",
        )
        if abs(angle) >= 0.5:
            layer = layer.rotate(
                -angle, resample=Image.BICUBIC, center=(side / 2, side / 2)
            )
        # 贴到文字区域中心
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        img.paste(
            layer,
            (round(cx - side / 2), round(cy - side / 2)),
            layer,
        )

    # ------------------------------------------------------------------ 字体

    def _detect_font_path(self) -> Optional[str]:
        for candidate in _FONT_CANDIDATES:
            if Path(candidate).exists():
                return candidate
        return None

    def _font_at(self, size: int) -> ImageFont.FreeTypeFont:
        cached = self._font_cache.get(size)
        if cached is None:
            cached = ImageFont.truetype(self._font_path, size)
            self._font_cache[size] = cached
        return cached

    # ------------------------------------------------------------------ 其他

    def _padded(self, bbox: List[int], size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        pad = self._settings.mask_padding
        x1 = max(0, bbox[0] - pad)
        y1 = max(0, bbox[1] - pad)
        x2 = min(size[0], bbox[2] + pad)
        y2 = min(size[1], bbox[3] + pad)
        return x1, y1, x2, y2

    def _save(self, data: bytes, output_path: Optional[str] = None) -> Tuple[Path, bytes]:
        """保存脱敏图，返回 (路径, 最终字节)。

        编码格式按目标扩展名决定（.jpg→JPEG、.tif→TIFF、.webp→WEBP …），
        保证内容与扩展名一致；无 output_path 时用 output_dir + 时间戳（PNG）。
        """
        if output_path:
            path = Path(output_path)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            path.parent.mkdir(parents=True, exist_ok=True)
            fmt = format_for_suffix(path.suffix)
            if fmt != "PNG":  # PNG 与 redacted_image 的默认编码一致，无需重编码
                data = encode_image(Image.open(BytesIO(data)), fmt)
            path.write_bytes(data)
            return path, data
        out_dir = Path(self._settings.output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"redacted_{time.strftime('%Y%m%d_%H%M%S')}"
        path = out_dir / f"{stem}.png"
        counter = 1
        while path.exists():
            path = out_dir / f"{stem}_{counter}.png"
            counter += 1
        path.write_bytes(data)
        return path, data
