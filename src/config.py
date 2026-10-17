"""全局配置：环境变量优先，.env 兜底。

双环境切换：
    外网开发（ModelScope）与内网跑批（vLLM）使用不同的 .env 文件，
    通过环境变量 IMAGE_MASK_ENV_FILE 指定，默认读取项目根目录 .env：
        IMAGE_MASK_ENV_FILE=.env.production python main.py data/folder1
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 默认 .env；部署时可用 IMAGE_MASK_ENV_FILE 指向别的配置文件
_ENV_FILE = os.environ.get("IMAGE_MASK_ENV_FILE") or str(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """所有可调参数集中在这里，节点与工具只读配置。"""

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OCR：rapid（默认，轻量）| paddle（精度更高，需 py311）
    ocr_engine: str = "rapid"
    # OCR 引擎实例池上限。
    # 实测（20 核 CPU，4 张图并发）：instances=1 → 5.7s，=2 → 7.0s，=4 → 12~14.5s。
    # 即**并发 OCR 只会更慢**：onnxruntime 单会话已吃满全部核心，多会话互相争抢，
    # 加上 PIL/numpy 预处理受 GIL 串行化。因此默认 1（等价于旧的全局锁）。
    # 只有给每个会话显式限制线程数（或换成 GPU 推理）时，增大才有意义。
    # 每个实例约 200~400MB 内存。
    ocr_engine_instances: int = 1
    # 全局方向矫正（需 rapid-orientation）：非 0° 时转正后 OCR，
    # 整条管线在转正图上进行，finalize 节点最后逆旋转回原方向。
    # RapidOCR 对竖排文本本就鲁棒，开启可多检出少量块；
    # 语义（返回值=需逆时针旋转度数）已由特征点测试固化。
    ocr_orientation_correct: bool = False

    # VLM：OpenAI 兼容协议。
    # 外网开发：ModelScope（base_url=https://api-inference.modelscope.cn/v1）
    # 内网跑批：vLLM（base_url=http://服务器:8000/v1，api_key 任意非空即可）
    # 切换只改 base_url / model / api_key 三项，代码不动。
    vlm_model: str = "Qwen/Qwen3.8-27B"
    vlm_base_url: str = "https://api-inference.modelscope.cn/v1"
    vlm_api_key: str = ""
    vlm_temperature: float = 0.0
    vlm_enable_thinking: bool = False
    vlm_max_image_side: int = 1024
    vlm_max_tokens: int = 4096
    # 单次 VLM 调用超时（秒）。本地 vLLM 推理慢于云 API，
    # 27B + 双图 + 长 JSON 输出建议不低于 600
    vlm_timeout: int = 600
    # VLM 质检时「当前脱敏图」用的长边上限：质检只需判断遮罩位置与
    # 文字是否消失，不需要与原图同分辨率；vision token 按面积计，
    # 512 相对 1024 约省 3/4。0 或负数 = 跟随 VLM_MAX_IMAGE_SIDE。
    vlm_qa_mask_image_side: int = 512

    # 脱敏
    default_method: str = "white_mask"  # white_mask | blur | mosaic
    mask_padding: int = 3
    mask_color: str = "255,255,255"
    text_color: str = "0,0,0"
    fake_data_enabled: bool = True
    fake_font_size_ratio: float = 0.8
    blur_radius: int = 12
    mosaic_factor: int = 16
    output_dir: str = "outputs"

    # 质检
    qa_min_match_len: int = 4
    qa_min_digits: int = 8
    # 质检重 OCR 的长边上限：这一轮只做文字比对（残留/正则兜底），
    # 不产出坐标，因此先把长边压到该值再识别，实拍大图可省 1~2s。
    # 0 或负数 = 不缩放（blur/mosaic 想保留最高残留检出灵敏度时关掉）。
    qa_reocr_max_side: int = 1280
    # 质检并发：把确定性预检（重 OCR + 正则 + 几何误伤）放到后台线程，
    # 与 VLM 语义质检同时跑，省掉「先重 OCR 再调 VLM」的整段串行时间
    # （实测 ~1.4s/张）。代价是 VLM 不再带 deterministic_hints（仅供参考
    # 的上下文）；确定性结论仍然权威：发现硬残留就直接判失败。
    # false = 回到旧的串行 + 带 hints 行为。
    qa_precheck_background: bool = True

    @property
    def mask_color_rgb(self) -> Tuple[int, int, int]:
        parts = [int(x) for x in self.mask_color.split(",")]
        if len(parts) != 3:
            raise ValueError(f"MASK_COLOR 格式应为 r,g,b，当前为 {self.mask_color!r}")
        return (parts[0], parts[1], parts[2])

    @property
    def text_color_rgb(self) -> Tuple[int, int, int]:
        parts = [int(x) for x in self.text_color.split(",")]
        if len(parts) != 3:
            raise ValueError(f"TEXT_COLOR 格式应为 r,g,b，当前为 {self.text_color!r}")
        return (parts[0], parts[1], parts[2])


@lru_cache
def get_settings() -> Settings:
    return Settings()


def resolve_default_method() -> "RedactionMethod":
    """把 DEFAULT_METHOD / --method 解析成枚举，非法值回退 white_mask。

    配置里写错一个字母不应该让整条流水线在 vlm 节点抛 ValueError
    （那会退化成 failed，且错误只出现在 error 字段里，很难定位）。
    """
    from src.schemas.vlm_analysis import RedactionMethod

    raw = (get_settings().default_method or "").strip().lower()
    try:
        return RedactionMethod(raw)
    except ValueError:
        logger.warning(
            "DEFAULT_METHOD=%r 不是合法脱敏方式（white_mask/blur/mosaic），"
            "已回退 white_mask",
            raw,
        )
        return RedactionMethod.WHITE_MASK
