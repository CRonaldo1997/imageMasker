"""全局配置：环境变量优先，.env 兜底。

双环境切换：
    外网开发（ModelScope）与内网跑批（vLLM）使用不同的 .env 文件，
    通过环境变量 IMAGE_MASK_ENV_FILE 指定，默认读取项目根目录 .env：
        IMAGE_MASK_ENV_FILE=.env.production python main.py data/folder1
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict

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
    detect_qr_codes: bool = True

    # 质检
    qa_min_match_len: int = 4
    qa_min_digits: int = 8

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
