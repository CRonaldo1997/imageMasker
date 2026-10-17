"""FakeDataMapper 测试：确定性、等长、拟真。"""
from __future__ import annotations

import re

import pytest

from src.fake_data import SURNAMES, FakeDataMapper, id_check_digit


@pytest.fixture
def mapper() -> FakeDataMapper:
    return FakeDataMapper()


# --------------------------------------------------------------------- 确定性


def test_same_value_same_fake(mapper):
    assert mapper.map("张三丰", "name") == mapper.map("张三丰", "name")
    assert mapper.map("13407982134", "mobile") == mapper.map("13407982134", "mobile")
    assert mapper.map("320123199001011234", "id_card") == mapper.map(
        "320123199001011234", "id_card"
    )


def test_different_values_different_fakes(mapper):
    assert mapper.map("张三丰", "name") != mapper.map("李四光", "name")


def test_consistency_across_instances():
    """跨实例一致：不同批次/进程映射结果相同（纯函数）。"""
    assert FakeDataMapper().map("王小明", "name") == FakeDataMapper().map("王小明", "name")


# --------------------------------------------------------------------- 等长


@pytest.mark.parametrize(
    "value,kind",
    [
        ("张三丰", "name"),
        ("李四", "name"),
        ("欧阳修文", "name"),
        ("13407982134", "mobile"),
        ("320123199001011234", "id_card"),
        ("YB2026101600123", "medical_card"),
        ("江苏省南京市鼓楼区中山路100号", "address"),
        ("上海市浦东新区张江高科技园区博云路2号", "address"),
    ],
)
def test_length_preserved(mapper, value, kind):
    fake = mapper.map(value, kind)
    assert len(fake) == len(value), f"{value!r} -> {fake!r}"


def test_fake_differs_from_original(mapper):
    for value, kind in [
        ("张三丰", "name"),
        ("13407982134", "mobile"),
        ("320123199001011234", "id_card"),
        ("江苏省南京市鼓楼区中山路100号", "address"),
    ]:
        assert mapper.map(value, kind) != value


# --------------------------------------------------------------------- 姓名


def test_name_is_realistic(mapper):
    for name in ["张三丰", "李四", "王芳", "欧阳娜娜"]:
        fake = mapper.map(name, "name")
        assert re.fullmatch(r"[\u4e00-\u9fff]+", fake)
        assert fake[0] in SURNAMES


def test_name_no_repeat_within_given(mapper):
    fake = mapper.map("张三丰", "name")
    assert len(set(fake)) == len(fake)  # 姓与名用字不重复


# --------------------------------------------------------------------- 手机号


@pytest.mark.parametrize("mobile", ["13407982134", "13800138000", "19912345678"])
def test_mobile_format(mapper, mobile):
    fake = mapper.map(mobile, "mobile")
    assert re.fullmatch(r"1[3-9]\d{9}", fake)


# --------------------------------------------------------------------- 身份证


def test_id_card_structure_and_check_digit(mapper):
    fake = mapper.map("320123199001011234", "id_card")
    assert re.fullmatch(r"\d{17}[\dXx]", fake)
    # 校验位合法
    assert id_check_digit(fake[:17]) == fake[17]
    # 地区码保留（公开信息），性别位保留
    assert fake[:6] == "320123"
    assert (int(fake[16]) % 2) == (int("320123199001011234"[16]) % 2)
    # 生日在 1940-2010 之间
    year = int(fake[6:10])
    assert 1940 <= year <= 2010


# --------------------------------------------------------------------- 地址


def test_address_keeps_structure_words(mapper):
    fake = mapper.map("江苏省南京市鼓楼区中山路100号", "address")
    for word in ("省", "市", "区", "路", "号"):
        assert word in fake, f"结构词 {word} 丢失：{fake!r}"
    # 省市区路名都被替换
    assert "江苏" not in fake and "南京" not in fake and "鼓楼" not in fake and "中山" not in fake


def test_address_digit_mapping(mapper):
    fake = mapper.map("江苏省南京市鼓楼区中山路100号", "address")
    digits = re.findall(r"\d+", fake)
    assert digits, "门牌号应保留数字结构"
    assert all(d[0] != "0" for d in digits)


# --------------------------------------------------------------------- 医保卡号 / generic


def test_medical_card_same_class_mapping(mapper):
    fake = mapper.map("YB2026101600123", "medical_card")
    assert len(fake) == 15
    assert fake[:2].isalpha() and fake[2:].isdigit()


def test_generic_keeps_char_classes(mapper):
    fake = mapper.map("AB-1234", "generic")
    assert len(fake) == 7
    assert fake[2] == "-"  # 结构字符保留
    assert fake[:2].isalpha() and fake[3:].isdigit()


def test_unknown_kind_falls_back_to_generic(mapper):
    assert mapper.map("some-secret-42", "weird_kind") == mapper.map("some-secret-42", "generic")


# --------------------------------------------------------------------- 校验位


def test_id_check_digit_known_values():
    # 常见测试身份证号的校验位
    assert id_check_digit("11010519491231002") == "X"
    assert id_check_digit("44052418800101001") == "4"
