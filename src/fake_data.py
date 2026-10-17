"""假数据映射：相同敏感值 → 相同假值，映射前后字数一致。

设计要点：
- 确定性：映射是纯函数（sha256(值) 做种子），同一敏感值在任何
  图片、任何批次中都映射到同一假数据，保证跨票据业务关联可用；
- 等长：所有生成器保证 len(假值) == len(原值)，版面不塌；
- 拟真：假姓名用真实姓氏+常用名用字；假手机号号段合法；
  假身份证号符合 GB 11643-1999 校验位；地址保留行政结构词。

kind 取值：name / mobile / id_card / medical_card / address / generic
"""
from __future__ import annotations

import hashlib
import random
import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------- 词池

SURNAMES: Tuple[str, ...] = tuple(
    "李王张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧程曹袁邓许傅沈曾彭吕"
    "苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史"
    "顾侯邵孟龙万段雷钱汤尹黎易常武乔贺赖龚文"
)

GIVEN_NAME_CHARS: Tuple[str, ...] = tuple(
    "伟芳娜敏静丽强磊军洋勇艳杰娟涛明超秀兰英华慧巧美玉萍红娥玲芬燕彬鹏飞鑫波斌宇浩凯"
    "健俊帆晨曦阳雪琳云颖欣怡璇嘉雨恩雅诗悦若梓文然舟山川野原禾墨青森火炎淼淼铭锐志忠"
    "义礼智信孝友德春秋冬东南西"
)

PROVINCE_NAMES: Tuple[str, ...] = (
    "北京", "上海", "天津", "重庆", "河北", "山西", "辽宁", "吉林", "黑龙江", "江苏",
    "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "海南",
    "四川", "贵州", "云南", "陕西", "甘肃", "青海", "台湾", "内蒙古", "广西", "西藏",
    "宁夏", "新疆", "香港", "澳门",
)

CITY_NAMES: Tuple[str, ...] = (
    "南京", "苏州", "无锡", "常州", "徐州", "南通", "扬州", "镇江", "杭州", "宁波",
    "温州", "嘉兴", "湖州", "绍兴", "金华", "合肥", "芜湖", "福州", "厦门", "泉州",
    "南昌", "赣州", "济南", "青岛", "烟台", "潍坊", "郑州", "洛阳", "开封", "武汉",
    "宜昌", "襄阳", "长沙", "株洲", "湘潭", "广州", "深圳", "珠海", "汕头", "佛山",
    "东莞", "中山", "南宁", "桂林", "柳州", "海口", "三亚", "成都", "绵阳", "自贡",
    "贵阳", "六盘水", "昆明", "曲靖", "西安", "宝鸡", "咸阳", "兰州", "天水", "西宁",
    "银川", "乌鲁木齐", "哈尔滨", "齐齐哈尔", "长春", "吉林", "沈阳", "大连", "石家庄",
    "唐山", "保定", "太原", "大同",
)

DISTRICT_NAMES: Tuple[str, ...] = (
    "玄武", "秦淮", "鼓楼", "建邺", "栖霞", "雨花台", "江宁", "六合", "浦口", "溧水",
    "高淳", "滨江", "西湖", "上城", "拱墅", "余杭", "萧山", "钱塘", "临平", "富阳",
    "临安", "包河", "蜀山", "庐阳", "瑶海", "雨湖", "岳塘", "天心", "岳麓", "开福",
    "芙蓉", "雨花", "天元", "荷塘", "石峰", "芦淞", "河东", "河西", "城东", "城西",
)

ROAD_NAMES: Tuple[str, ...] = (
    "中山", "人民", "解放", "建设", "文化", "新华", "北京", "南京", "上海", "广州",
    "成都", "武汉", "长江", "黄河", "青年", "复兴", "和平", "幸福", "科技", "创新",
    "学院", "大学", "医院", "健康", "迎宾", "滨江", "湖滨", "望江", "抚河", "八一",
    "长风",
)

_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECK_CODES = "10X98765432"

_DIGIT_RE = re.compile(r"\d+")


def id_check_digit(body17: str) -> str:
    """GB 11643-1999 身份证校验位。"""
    total = sum(int(d) * w for d, w in zip(body17, _ID_WEIGHTS))
    return _ID_CHECK_CODES[total % 11]


class FakeDataMapper:
    """确定性假数据映射器。实例可共享（同值同假）。"""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], str] = {}

    # ---------------------------------------------------------------- 对外

    def map(self, value: str, kind: str) -> str:
        """把敏感值映射为等长假数据。kind 未知时走通用同类映射。"""
        if not value:
            return value
        key = (kind, value)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        generator = getattr(self, f"_map_{kind}", None) or self._map_generic
        try:
            fake = generator(value)
        except Exception:
            fake = self._map_generic(value)
        # 等长是硬约束：任何生成器失控都退化为同类映射
        if len(fake) != len(value):
            fake = self._map_generic(value)
        self._cache[key] = fake
        return fake

    # ---------------------------------------------------------------- 内部

    @staticmethod
    def _seed(value: str) -> int:
        digest = hashlib.sha256(value.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big")

    @staticmethod
    def _pick_same_length(pool: Tuple[str, ...], length: int, rng: random.Random) -> str:
        """从词池中选等长候选；没有则生成等长随机汉字兜底。"""
        same = [p for p in pool if len(p) == length]
        if same:
            return rng.choice(same)
        return "".join(rng.choice(GIVEN_NAME_CHARS) for _ in range(length))

    # ---------------- 各类型生成器 ----------------

    def _map_name(self, value: str) -> str:
        """姓名：等长。姓 1 字 + 名 n-1 字（用字不重复）。"""
        rng = random.Random(self._seed(value))
        n = len(value)
        if n < 2:
            return self._map_generic(value)
        surname = rng.choice(SURNAMES)
        if n - 1 <= len(GIVEN_NAME_CHARS):
            given = "".join(rng.sample(GIVEN_NAME_CHARS, n - 1))
        else:  # 超长姓名，允许重复
            given = "".join(rng.choice(GIVEN_NAME_CHARS) for _ in range(n - 1))
        return surname + given

    def _map_mobile(self, value: str) -> str:
        """手机号：11 位、1 开头、第二位 3-9（号段合法）。"""
        rng = random.Random(self._seed(value))
        digits = re.sub(r"\D", "", value)
        if len(digits) == 11:
            return "1" + str(rng.randrange(3, 10)) + "".join(
                str(rng.randrange(10)) for _ in range(9)
            )
        return self._map_generic(value)  # 非标准长度，同类映射兜底

    def _map_id_card(self, value: str) -> str:
        """身份证号：保留地区码（公开信息）与性别位，生日合法，校验位正确。"""
        rng = random.Random(self._seed(value))
        digits = re.sub(r"\D", "", value)
        if len(digits) != 18:
            return self._map_generic(value)
        area = digits[:6]
        year = rng.randrange(1940, 2011)
        month = rng.randrange(1, 13)
        day = rng.randrange(1, 29)
        gender = digits[16]  # 第 17 位奇男偶女，保留以维持业务含义
        # 结构：6 地区码 + 8 生日 + 2 随机 + 1 性别位 = 17 位
        body = (
            area
            + f"{year:04d}{month:02d}{day:02d}"
            + rng.choice("0123456789")
            + rng.choice("0123456789")
            + gender
        )
        return body + id_check_digit(body)

    def _map_address(self, value: str) -> str:
        """详细地址：等长替换省/市/区/路名与门牌数字，保留行政结构词。"""
        rng = random.Random(self._seed(value))
        text = value

        def sub_province(m: "re.Match[str]") -> str:
            return self._pick_same_length(PROVINCE_NAMES, len(m.group(1)), rng) + "省"

        def sub_city(m: "re.Match[str]") -> str:
            # 正则吃掉了前缀"省"，替换时加回
            return "省" + self._pick_same_length(CITY_NAMES, len(m.group(1)), rng) + "市"

        def sub_district(m: "re.Match[str]") -> str:
            # 正则吃掉了前缀"市"，替换时加回
            return "市" + self._pick_same_length(DISTRICT_NAMES, len(m.group(1)), rng) + m.group(2)

        def sub_road(m: "re.Match[str]") -> str:
            # 正则吃掉了前缀"区/县"，替换时加回
            return m.group(0)[0] + self._pick_same_length(
                ROAD_NAMES, len(m.group(1)), rng
            ) + m.group(2)

        def sub_digits(m: "re.Match[str]") -> str:
            n = len(m.group(0))
            return str(rng.randrange(1, 10)) + "".join(
                str(rng.randrange(10)) for _ in range(n - 1)
            )

        text = re.sub(r"^(.+?)省", sub_province, text, count=1)
        text = re.sub(r"省(.+?)市", sub_city, text, count=1)
        text = re.sub(r"市(.+?)([区县])", sub_district, text, count=1)
        text = re.sub(r"[区县](.+?)([路街巷])", sub_road, text, count=1)
        text = _DIGIT_RE.sub(sub_digits, text)
        return text

    def _map_medical_card(self, value: str) -> str:
        """医保卡号：等长同类映射（数字→数字，字母→字母）。"""
        return self._map_generic(value)

    def _map_generic(self, value: str) -> str:
        """通用等长同类映射：数字→数字、字母→字母、汉字→常用字。"""
        rng = random.Random(self._seed(value) ^ 0x9E3779B9)
        out: List[str] = []
        for ch in value:
            if ch.isdigit():
                out.append(str(rng.randrange(10)))
            elif ch.isascii() and ch.isalpha():
                base = ord("A") if ch.isupper() else ord("a")
                out.append(chr(base + rng.randrange(26)))
            elif "\u4e00" <= ch <= "\u9fff":
                out.append(rng.choice(GIVEN_NAME_CHARS))
            else:
                out.append(ch)  # 横线、空格等结构字符原样保留
        return "".join(out)
