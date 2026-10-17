"""入口大图缩小：阈值判定、等比、格式一致、失败降级。"""
from __future__ import annotations

import os
from io import BytesIO

import pytest
from PIL import Image

from src.image_preprocess import (
    DEFAULT_MAX_IMAGE_BYTES,
    maybe_shrink,
    normalize_exif_orientation,
)


def _noise_image(width: int, height: int, fmt: str = "PNG", quality: int = 95) -> bytes:
    """生成难以压缩的随机噪声图，用于稳定构造"大文件"。"""
    img = Image.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    buf = BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(data)) as img:
        return img.size


def _magic(data: bytes) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    raise AssertionError("未知图片格式")


# ------------------------------------------------------------------ 不缩小


def test_small_image_untouched():
    """未超阈值：字节完全不变，不产生缩小信息。"""
    data = _noise_image(64, 64)
    out, info = maybe_shrink(data, DEFAULT_MAX_IMAGE_BYTES)
    assert out == data
    assert info is None


def test_disabled_returns_original():
    data = _noise_image(400, 400)
    out, info = maybe_shrink(data, max_bytes=1024, enabled=False)
    assert out == data
    assert info is None


def test_zero_threshold_means_disabled():
    data = _noise_image(400, 400)
    out, info = maybe_shrink(data, max_bytes=0)
    assert out == data
    assert info is None


# ------------------------------------------------------------------ 缩小


def test_shrinks_until_under_threshold():
    # JPEG 体积与像素面积近似线性，能稳定压到阈值以下
    data = _noise_image(1600, 1600, fmt="JPEG", quality=100)
    limit = 800_000
    assert len(data) > limit

    out, info = maybe_shrink(data, max_bytes=limit)

    assert len(out) <= limit
    assert info is not None
    assert info["original_bytes"] == len(data)
    assert info["shrunk_bytes"] == len(out)
    assert info["shrunk_bytes"] < info["original_bytes"]


def test_keeps_aspect_ratio():
    data = _noise_image(1200, 600)
    out, info = maybe_shrink(data, max_bytes=200_000)
    assert info is not None

    ow, oh = info["original_size"]
    nw, nh = info["shrunk_size"]
    assert abs(ow / oh - nw / nh) < 0.02     # 等比（允许取整误差）
    assert (nw, nh) == _size(out)            # 记录尺寸与实际一致
    assert nw < ow and nh < oh


def test_png_stays_png():
    """输出格式必须与输入一致，否则图与扩展名不符（踩坑 #15）。"""
    data = _noise_image(1600, 1600, fmt="PNG")
    out, info = maybe_shrink(data, max_bytes=1_000_000)
    assert info["format"] == "PNG"
    assert _magic(out) == "PNG"


def test_jpeg_stays_jpeg():
    data = _noise_image(1400, 1400, fmt="JPEG", quality=100)
    out, info = maybe_shrink(data, max_bytes=300_000)
    assert info["format"] == "JPEG"
    assert _magic(out) == "JPEG"


def test_rgba_composited_on_white():
    """带透明通道的图缩小时合成到白底，不应把透明区变黑。"""
    img = Image.frombytes("RGBA", (1600, 1600), os.urandom(1600 * 1600 * 4))
    buf = BytesIO()
    img.save(buf, format="PNG")
    out, info = maybe_shrink(buf.getvalue(), max_bytes=1_000_000)

    assert info is not None
    with Image.open(BytesIO(out)) as shrunk:
        assert shrunk.mode == "RGB"


def test_min_long_side_guard():
    """接近长边下限时停止继续缩小，避免把文字缩到认不出。"""
    data = _noise_image(950, 950)
    out, info = maybe_shrink(data, max_bytes=1)
    # 无法压到 1 字节，但必须收敛在长边下限之上且不无限循环
    assert info is not None
    assert max(info["shrunk_size"]) >= 800


# ------------------------------------------------------------------ 降级


def test_corrupted_bytes_pass_through():
    """超大但无法解码的内容：原样返回，交给主流程报错，预处理不抛异常。"""
    data = b"\x00" * 200_000
    out, info = maybe_shrink(data, max_bytes=1024)
    assert out == data
    assert info is None


def test_never_returns_bigger_data():
    """极端输入（已压缩到极限的小图但字节数超阈值）不应把文件变得更大。"""
    data = _noise_image(32, 32)
    padded = data + b"\x00" * 5000           # 尾巴脏数据，解码仍成功
    out, _info = maybe_shrink(padded, max_bytes=1024)
    assert len(out) <= len(padded)


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_info_fields(fmt):
    data = _noise_image(1200, 800, fmt=fmt)
    out, info = maybe_shrink(data, max_bytes=300_000)
    assert set(info) == {
        "original_bytes", "shrunk_bytes", "original_size", "shrunk_size", "format",
    }
    assert info["format"] == fmt
    assert info["original_size"] == [1200, 800]


# ------------------------------------------------------------------ EXIF 方向


def _jpeg_with_orientation(width: int, height: int, orientation: int) -> bytes:
    img = Image.new("RGB", (width, height), "white")
    exif = Image.Exif()
    exif[0x0112] = orientation
    buf = BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    return buf.getvalue()


def test_exif_orientation_is_normalized():
    """手机照片（Orientation=6，像素横向）必须转正，否则输出图横躺。"""
    data = _jpeg_with_orientation(40, 20, 6)

    out, info = normalize_exif_orientation(data)

    assert info is not None
    assert info["orientation"] == 6
    assert info["original_size"] == [40, 20]
    with Image.open(BytesIO(out)) as img:
        assert img.size == (20, 40)          # 像素已转正
        assert int(img.getexif().get(0x0112, 1) or 1) == 1  # 残留 EXIF 方向已清除
    assert _magic(out) == "JPEG"             # 格式与输入一致


def test_exif_absent_or_upright_returns_original_bytes():
    """无 EXIF / Orientation=1：字节完全不变，不做无谓重编码。"""
    png = _noise_image(64, 64)
    assert normalize_exif_orientation(png) == (png, None)

    upright = _jpeg_with_orientation(40, 20, 1)
    assert normalize_exif_orientation(upright) == (upright, None)


def test_exif_corrupted_bytes_pass_through():
    """无法解码时原样返回，预处理不抛异常。"""
    data = b"\x00" * 1000
    assert normalize_exif_orientation(data) == (data, None)
