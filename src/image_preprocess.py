"""入口图像预处理：超大图片等比缩小，缩短 OCR/VLM 耗时。

为什么要做：
    OCR 与 VLM 的耗时基本随像素面积增长；实拍票据常见 10~30MB 的高分辨率图，
    缩到 5MB 以内通常不损失可识别性（VLM 侧本来就会把长边缩到
    VLM_MAX_IMAGE_SIDE=1024），但能显著减少解码、OCR 推理与 base64 传输开销。

设计约束（与 README「降级链无处不在」一致）：
1. 只缩小、不裁剪、不改宽高比；
2. **输出编码格式与输入一致**（.jpg 仍是 JPEG、.png 仍是 PNG），
   否则会违反「扩展名必须与内容格式一致」（踩坑 #15）；
3. 由调用方在入口统一执行，全链路（OCR/VLM/脱敏/质检/finalize）
   都基于缩小后的同一份字节，坐标天然自洽，不需要坐标反变换；
4. 任何异常（文件损坏、格式不支持、编码失败）一律返回原始字节并告警，
   预处理不能成为新的故障点。

入口还有一步 EXIF 方向归一化（normalize_exif_orientation）：
手机拍摄的票据带 Orientation 标记时先转正像素，否则输出图会横躺、
OCR 也会在侧倒页面上掉识别率。仅在 Orientation != 1 时才重编码。
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Optional, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

#: 默认阈值：超过 5MB 的图片进入缩小流程
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024

#: 每轮缩小的比例与最大轮数（0.85^8 ≈ 0.27，足够把体积降到 1/3 以下）
SCALE_STEP = 0.85
MAX_ROUNDS = 8

#: 长边下限：低于该值不再继续缩小，避免把票据文字缩到 OCR 认不出来
MIN_LONG_SIDE = 800

#: 重编码质量（仅对有损格式生效）
JPEG_QUALITY = 88

_EXIF_ORIENTATION_TAG = 0x0112


def maybe_shrink(
    image_bytes: bytes,
    max_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
    enabled: bool = True,
) -> Tuple[bytes, Optional[dict]]:
    """大于 max_bytes 时等比缩小并重编码，返回 (新字节, 缩小信息)。

    未缩小（未开启 / 未超阈值 / 读取失败）时返回 (原字节, None)，
    调用方据此判断是否需要记录日志。
    """
    if not enabled or max_bytes is None or max_bytes <= 0:
        return image_bytes, None
    if len(image_bytes) <= max_bytes:
        return image_bytes, None

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            fmt = (img.format or "PNG").upper()
            orig_size = img.size
            work = _to_rgb(img)
    except Exception as exc:  # 损坏文件/不支持格式：交给主流程自己报错
        logger.warning("大图缩小跳过（无法解码，%s: %s）", type(exc).__name__, exc)
        return image_bytes, None

    if fmt not in ("JPEG", "PNG", "BMP", "TIFF", "WEBP"):
        # 未知格式按 PNG 落盘，避免出现「扩展名与内容不一致」的中间产物
        fmt = "PNG"

    data = image_bytes
    shrunk_any = False
    for _ in range(MAX_ROUNDS):
        width, height = work.size
        long_side = max(width, height)
        next_long = round(long_side * SCALE_STEP)
        if next_long < MIN_LONG_SIDE:
            if shrunk_any:
                logger.warning(
                    "已到长边下限 %dpx，停止缩小（当前 %.1fMB，目标 %.1fMB）；"
                    "继续缩小会明显损害 OCR 识别率",
                    MIN_LONG_SIDE, len(data) / 1024 / 1024, max_bytes / 1024 / 1024,
                )
            break
        work = work.resize(
            (max(1, round(width * SCALE_STEP)), max(1, round(height * SCALE_STEP))),
            Image.LANCZOS,
        )
        try:
            data = _encode(work, fmt)
        except Exception as exc:
            logger.warning("大图缩小重编码失败（%s），放弃缩小", exc)
            return image_bytes, None
        shrunk_any = True
        if len(data) <= max_bytes:
            break

    if not shrunk_any:
        return image_bytes, None

    if len(data) >= len(image_bytes):  # 反变大（几乎不会发生）：保持原样
        return image_bytes, None

    info = {
        "original_bytes": len(image_bytes),
        "shrunk_bytes": len(data),
        "original_size": list(orig_size),
        "shrunk_size": list(work.size),
        "format": fmt,
    }
    logger.info(
        "大图缩小：%.1fMB → %.1fMB，%dx%d → %dx%d（%s）",
        len(image_bytes) / 1024 / 1024,
        len(data) / 1024 / 1024,
        orig_size[0], orig_size[1], work.size[0], work.size[1], fmt,
    )
    return data, info


def normalize_exif_orientation(image_bytes: bytes) -> Tuple[bytes, Optional[dict]]:
    """按 EXIF Orientation 把像素转正，返回 (新字节, 处理信息)。

    为什么需要：手机拍摄的票据常带 Orientation=6/8（像素横向存储、
    靠 EXIF 标记竖显）。后续所有节点（OCR/VLM/脱敏）都读原始像素，而
    输出图在重新编码时不含 EXIF，结果就是「原图竖着看、脱敏图横着躺」，
    OCR 在侧倒的页面上识别率也会明显下降。

    入口统一转正后：坐标、脱敏图、输出方向全部自洽，EXIF 也不再需要。

    降级（与 maybe_shrink 一致）：无 EXIF / Orientation=1 / 解码失败 /
    重编码失败时一律返回原字节，预处理不做新的故障点。
    """
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            # 读 EXIF 只需解析文件头，未触发整图解码，开销可忽略
            orientation = int(img.getexif().get(_EXIF_ORIENTATION_TAG, 1) or 1)
            if orientation in (0, 1):
                return image_bytes, None
            fmt = (img.format or "PNG").upper()
            orig_size = img.size
            upright = ImageOps.exif_transpose(img)
            if upright is None:  # 极端情况下拿不到转正结果，保持原样
                return image_bytes, None
            work = _to_rgb(upright)
            data = _encode(work, fmt, quality=95)
    except UnidentifiedImageError:
        # 不是图片（损坏/黑客文件）：主流程 OCR 会给出明确错误，这里不重复告警
        logger.debug("内容不是可解码图片，跳过 EXIF 方向矫正")
        return image_bytes, None
    except Exception as exc:
        logger.warning(
            "EXIF 方向矫正跳过（%s: %s），按原始像素处理",
            type(exc).__name__, exc,
        )
        return image_bytes, None

    info = {
        "orientation": orientation,
        "original_size": list(orig_size),
        "rotated_size": list(work.size),
        "format": fmt,
    }
    logger.info(
        "EXIF 方向矫正：Orientation=%d，%dx%d → %dx%d（%s）",
        orientation, orig_size[0], orig_size[1], work.size[0], work.size[1], fmt,
    )
    return data, info


def _to_rgb(img: Image.Image) -> Image.Image:
    """统一转 RGB；带透明通道的图合成到白底，避免透明区变黑。"""
    if img.mode == "RGB":
        return img.copy()
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGB", rgba.size, (255, 255, 255))
        canvas.paste(rgba, mask=rgba.split()[-1])
        return canvas
    return img.convert("RGB")


def _encode(img: Image.Image, fmt: str, quality: int = JPEG_QUALITY) -> bytes:
    """按原格式重编码，保证输出扩展名与内容一致。"""
    buf = BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    elif fmt == "WEBP":
        img.save(buf, format="WEBP", quality=quality)
    else:
        img.save(buf, format=fmt, optimize=True)
    return buf.getvalue()
