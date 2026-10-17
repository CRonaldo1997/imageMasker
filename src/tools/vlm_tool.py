"""VLM 工具：判断哪些 OCR 文字块属于敏感信息；兼做质检语义判断。

输入约定：
- OCR 结果以 JSON 数组传入（block_id / text / bbox / confidence），
  VLM 对结构化 JSON 的处理远优于管道分隔文本；
- 首次判断只给原图；回退判断额外给质检意见与当前 mask 图；
- 输出走 langchain 结构化输出（with_structured_output，固定 ToolStrategy /
  method="function_calling"）：用工具调用强制模型提交结构化结果，规避
  ModelScope 兼容端点原生 json_schema 返回数组的 bug；极端兼容问题
  时再回退 JSON 解析。

坐标策略：VLM 输出 bbox，但与 OCR 坐标校验（IoU 过低视为
幻觉坐标，以 OCR 为准）。
"""
from __future__ import annotations

import base64
import json
import logging
import re
import threading
from io import BytesIO
from typing import List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from PIL import Image

from src.config import Settings
from src.schemas.ocr_result import OcrResult
from src.schemas.qa_verdict import QaVerdict
from src.schemas.redaction_result import RedactionResult
from src.schemas.vlm_analysis import (
    RedactionMethod,
    SensitiveBlock,
    SensitiveKind,
    VlmAnalysis,
)

logger = logging.getLogger(__name__)

#: 透传发送时使用的 MIME（键为 PIL format 名）
_MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}

# 结构化输出重试次数：ToolStrategy 仍可能产出坏字段（类型不符 / 缺必填 /
# 枚举越界）导致 ValidationError，这里最多尝试 3 次，全部失败才抛
# StructuredOutputError，由节点决定如何降级（而不是让整条流水线崩掉）。
STRUCTURED_OUTPUT_ATTEMPTS = 3

_RETRY_HINT = (
    "上一次输出未通过结构化 schema 校验。请严格按工具参数的 JSON schema 重新提交："
    "字段类型必须匹配，必填字段不能缺失，枚举值只能取 schema 允许的取值，"
    "不要输出任何多余字段或包裹数组。"
)


class StructuredOutputError(RuntimeError):
    """LLM 多次重试后仍无法产出符合 schema 的结构化输出。"""


def _looks_like_param_rejection(exc: Exception) -> bool:
    """错误是否「长得像 extra_body 参数被服务端拒绝」。

    超时/限流/5xx 等瞬时故障返回 False——那些情况不该动思考模式参数，
    否则一次网络抖动就可能把关闭思考的候选写法耗尽，之后整批都开着思考跑。
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _PARAM_REJECTION_MARKERS)


SYSTEM_PROMPT = """你是医疗票据隐私脱敏专家。给你一张医疗票据图片，以及 OCR 识别出的文字块（JSON 数组，含 block_id / text / bbox），请判断哪些文字块包含患者隐私信息、需要脱敏。

需要脱敏的敏感信息（仅以下五类）：
1. 姓名：患者/交款人姓名。医院名称、科室、医务人员姓名不脱敏；
2. 身份证号；
3. 手机号；
4. 社保号/医保卡号；
5. 能精确找到人的详细住址：精确到具体建筑/楼层/房间的住址。仅"南京市"这类城市或省级别地名不脱敏。

不需要脱敏的普通信息：医院名称、科室、医务人员姓名、就诊/打印日期、检查项目名称、金额、票据号码、流水号、页码、版式文字、固定电话号码、城市/省级别地名。

输出要求：
1. 通过 block_id 引用文字块，不要自创块；
2. 每个敏感块给出 kind（name/mobile/id_card/medical_card/address 之一）；
3. value 必须是块文本中敏感值的精确子串：块文本"姓名：张三"时 value 为"张三"；
   块文本"地址：江苏省南京市鼓楼区中山路100号"时 value 为"江苏省南京市鼓楼区中山路100号"；
4. bbox 直接复制该块在输入 JSON 中的坐标，不要自己估算；
5. 不需要判断脱敏方式（method 保持默认即可，实际方式由系统配置决定）；
6. 给出简短判断理由；
7. 保守判断：拿不准是否敏感时，倾向脱敏；
8. 直接给出结论，不要输出任何思考/推理过程。"""

QA_SYSTEM_PROMPT = """你是医疗票据脱敏质检专家。给你原图、当前脱敏图（mask 图）、OCR 文字块（JSON 数组）以及当前被判为敏感的块列表（JSON），请判断脱敏是否合格。

合格标准：
1. 所有敏感信息在 mask 图中均已不可识别（已被假数据替换或覆盖）；
2. 没有误伤：非敏感信息（医院、科室、日期、项目、金额等）未被遮挡。

重要说明：
- 输入中会给出每个敏感块已替换成的假数据文本。mask 图中出现这些
  假数据是预期结果，不算残留、不算遮挡；
- 只有原敏感值（如真实姓名、真实证件号）在 mask 图中仍可识别时，
  才判残留；
- 脱敏采用"保留标签、替换值"策略（如"姓名：张三"→"姓名：李四"），
  标签文字仍在图中是正常的，不要判为误伤。

输出要求：
- status：passed / failed；
- reason：简要结论；
- feedback：未通过时给出可执行的修正意见（指明块 ID）；
- residual_block_ids：mask 图中仍可识别的敏感块 ID；
- damaged_block_ids：被误遮挡的非敏感块 ID；
- revised_block_ids：你复判后认为应当脱敏的块 ID 全集（供下一轮修正使用）。

直接给出结论，不要输出任何思考/推理过程。
"""


#: 关闭 Qwen3 思考模式的 extra_body 候选写法。
#: 不同 vLLM / 兼容端点版本接受的字段名不同，按顺序试探，第一个被服务端
#: 接受的即固定下来；全部被拒才放弃（并打 ERROR，绝不静默开思考）。

#: 只有「长得像参数被拒」的错误才触发换写法：超时/限流/5xx 等瞬时错误
#: 不能顺手把思考模式关掉的能力耗尽（那会静默退化成开着思考跑批）。
_PARAM_REJECTION_MARKERS = (
    "chat_template_kwargs",
    "enable_thinking",
    "extra_body",
    "unexpected keyword",
    "unsupported",
    "unrecognized",
    "additional properties",
    "400",
    "422",
    "bad request",
    "unprocessable",
    "invalid",
)
_THINKING_OFF_PAYLOADS: Tuple[dict, ...] = (
    {"chat_template_kwargs": {"enable_thinking": False}},  # vLLM 主流写法
    {"enable_thinking": False},                            # 部分版本/兼容端点
    {"thinking": {"type": "disabled"}},                    # 另一种常见写法
)


class VlmTool:
    """VLM 工具：敏感判断 + 质检语义判断。模型走 OpenAI 兼容协议。

    思考模式：本项目的两次调用（敏感判断 + 语义质检）共用同一个模型实例，
    因此共用同一份 extra_body——关闭思考模式对两条路径同时生效，不存在
    「一条关一条没关」。若服务端不接受当前写法，会自动换下一种候选写法；
    全部被拒时打 ERROR 日志（部署时 grep 这行即可确认），不会静默开着思考。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._thinking_off = not settings.vlm_enable_thinking
        # 候选写法副本：被拒绝一个就弹出一个，全部弹出即放弃
        self._payload_candidates: List[dict] = (
            [dict(p) for p in _THINKING_OFF_PAYLOADS] if self._thinking_off else []
        )
        self._extra_body: dict = (
            dict(self._payload_candidates[0]) if self._payload_candidates else {}
        )
        self._extra_body_disabled = False
        self._model = self._build_model()
        # 批量模式多线程共用同一实例：降级会重建客户端并更换 extra_body，
        # 用锁串行化这段共享状态变更，避免并发下反复重建/读到半更新状态。
        self._degrade_lock = threading.Lock()
        self._thinking_state_logged = False

    def _build_model(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=self._settings.vlm_model,
            api_key=self._settings.vlm_api_key or "EMPTY",  # vLLM 不校验 key，占位即可
            base_url=self._settings.vlm_base_url,
            temperature=self._settings.vlm_temperature,
            # 多文字块的结构化 JSON（26 块 x 字段）可能超过 2048，放宽到 4096
            max_tokens=self._settings.vlm_max_tokens,
            request_timeout=self._settings.vlm_timeout,
            extra_body=self._extra_body or None,
        )

    # ------------------------------------------------------------------ 敏感判断

    def analyze(
        self,
        image: bytes,
        ocr_result: OcrResult,
        previous_redaction: Optional[RedactionResult] = None,
        feedback: Optional[str] = None,
    ) -> VlmAnalysis:
        """判断敏感块。首次执行 feedback 为空；回退执行时按质检意见修正。"""
        messages = self._build_analyze_messages(
            image, ocr_result, previous_redaction, feedback
        )
        analysis = self._invoke_structured(messages, VlmAnalysis)
        self._sanitize(analysis, ocr_result)
        analysis.model = self._settings.vlm_model
        return analysis

    # ------------------------------------------------------------------ 质检判断

    def qa_verify(
        self,
        original_image: bytes,
        redacted_image: bytes,
        vlm_analysis: VlmAnalysis,
        ocr_result: Optional[OcrResult] = None,
        feedback: Optional[str] = None,
        deterministic_hints: Optional[dict] = None,
        redaction_result: Optional[RedactionResult] = None,
    ) -> QaVerdict:
        """VLM 质检：比对原图、mask 图与 OCR/敏感块 JSON。

        redaction_result：必须提供——mask 图里对应块已替换为假数据，
        需告知 VLM 各块的绘制文本，否则它看到文字就误判"仍清晰可见"。
        deterministic_hints：本地确定性预检结果，作为辅助上下文。
        """
        messages = self._build_qa_messages(
            original_image, redacted_image, vlm_analysis, ocr_result,
            feedback, deterministic_hints, redaction_result,
        )
        verdict = self._invoke_structured(messages, QaVerdict)
        return verdict

    # ------------------------------------------------------------------ 消息构建

    @staticmethod
    def _ocr_to_json(ocr_result: OcrResult) -> str:
        """给 VLM 的 OCR 块 JSON：只含锚定三要素（ID / 文本 / 坐标）。
        置信度是 OCR 引擎的自我评估，对语义判断没有决策价值，不传入。"""
        blocks = [
            {
                "block_id": b.block_id,
                "text": b.text,
                "bbox": list(b.bbox),
            }
            for b in ocr_result.blocks
        ]
        return json.dumps(blocks, ensure_ascii=False, indent=1)

    @staticmethod
    def _sensitive_to_json(analysis: VlmAnalysis) -> str:
        blocks = [
            {
                "block_id": sb.block_id,
                "text": sb.text,
                "value": sb.value,
                "kind": sb.kind,
                "bbox": list(sb.bbox),
            }
            for sb in analysis.sensitive_blocks
        ]
        return json.dumps(blocks, ensure_ascii=False, indent=1)

    def _build_analyze_messages(
        self,
        image: bytes,
        ocr_result: OcrResult,
        previous_redaction: Optional[RedactionResult],
        feedback: Optional[str],
    ) -> List:
        ocr_json = self._ocr_to_json(ocr_result)
        user_text = f"OCR 文字块（JSON）：\n{ocr_json}"

        content = [
            {"type": "image_url", "image_url": {"url": self._image_url(image)}},
        ]
        if feedback:
            # 回退判断：给质检意见 + 当前 mask 图，让 VLM 看到脱敏现状
            content.append({"type": "text", "text": user_text})
            previous = (
                "、".join(rb.block_id for rb in previous_redaction.redacted_blocks)
                if previous_redaction
                else "无"
            )
            revised = ""
            if previous_redaction:
                revised = (
                    "当前 mask 图示意（各块脱敏情况）：\n"
                    + json.dumps(
                        [
                            {"block_id": rb.block_id, "drawn_text": rb.text}
                            for rb in previous_redaction.redacted_blocks
                        ],
                        ensure_ascii=False,
                        indent=1,
                    )
                )
            content.append(
                {
                    "type": "text",
                    "text": (
                        "\n\n【上一轮质检未通过，反馈如下】\n"
                        f"{feedback}\n\n"
                        f"上一轮已脱敏块：{previous}\n"
                        f"{revised}\n"
                        "下图是当前脱敏图，请对照原图与反馈修正："
                        "反馈中指出的残留块必须纳入敏感列表；"
                        "被误伤的非敏感块应移出敏感列表。"
                    ),
                }
            )
            # 质检失败必然有上一轮的脱敏图；但回退消息也可能被单独构造
            # （如上游未产出 redaction_result 时），这里显式判空，避免
            # AttributeError 把“重判”变成整图 failed。
            if previous_redaction is not None and previous_redaction.redacted_image:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_url(
                                previous_redaction.redacted_image,
                                self._qa_mask_max_side(),
                            )
                        },
                    }
                )
        else:
            content.append(
                {"type": "text", "text": user_text + "\n\n请判断哪些文字块需要脱敏。"}
            )

        return [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=content),
        ]

    def _build_qa_messages(
        self,
        original_image: bytes,
        redacted_image: bytes,
        vlm_analysis: VlmAnalysis,
        ocr_result: Optional[OcrResult],
        feedback: Optional[str],
        deterministic_hints: Optional[dict],
        redaction_result: Optional[RedactionResult] = None,
    ) -> List:
        parts = ["下图是原图。"]
        parts.append(f"当前被判为敏感的块（JSON）：\n{self._sensitive_to_json(vlm_analysis)}")
        if redaction_result is not None:
            # 关键：告知各块已替换成的假数据（绘制文本）。
            # 否则 VLM 看到 mask 图里有文字就误判"仍清晰可见"。
            drawn = {
                rb.block_id: rb.text
                for rb in redaction_result.redacted_blocks
                if rb.text
            }
            if drawn:
                parts.append(
                    "脱敏图中上述各块已替换为以下假数据（看到这些文本是预期的，"
                    "不属于残留）：\n"
                    + json.dumps(drawn, ensure_ascii=False, indent=1)
                )
        if ocr_result is not None:
            parts.append(f"OCR 文字块（JSON）：\n{self._ocr_to_json(ocr_result)}")
        if deterministic_hints:
            parts.append(
                "本地确定性预检结果（供参考）：\n"
                + json.dumps(deterministic_hints, ensure_ascii=False, indent=1)
            )
        if feedback:
            parts.append(f"历史质检意见：{feedback}")
        parts.append("请判断当前脱敏图是否合格。")

        return [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {"url": self._image_url(original_image)},
                    },
                    {"type": "text", "text": "\n\n".join(parts)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": self._image_url(
                                redacted_image, self._qa_mask_max_side()
                            )
                        },
                    },
                ]
            ),
        ]

    def _image_url(self, image: bytes, max_side: Optional[int] = None) -> str:
        payload, mime = self._prepare_image(image, max_side)
        return f"data:{mime};base64,{payload}"

    def _qa_mask_max_side(self) -> int:
        """质检用「脱敏图」的长边上限（0/负数表示跟随原图上限）。"""
        side = self._settings.vlm_qa_mask_image_side
        return side if side and side > 0 else self._settings.vlm_max_image_side

    def _prepare_image(
        self, image: bytes, max_side: Optional[int] = None
    ) -> Tuple[str, str]:
        """返回 (base64 载荷, MIME)。

        - 长边超过阈值时缩放：vision token 按像素面积计，是 VLM 输入
          成本的大头，判断依据是块 ID，缩放不影响结论；
        - 未超阈值、且是 JPEG/PNG/WEBP、且无 EXIF 方向标记时**原样透传**：
          省掉一次解码+重编码，也避免把 JPEG 膨胀成 PNG 再 base64；
        - 其它情况按 PNG 重编码——顺带剥掉 EXIF 方向，防止服务端按
          EXIF 旋转后与我们的遮罩坐标不一致。
        """
        if max_side is None:
            max_side = self._settings.vlm_max_image_side
        try:
            with Image.open(BytesIO(image)) as img:
                fmt = (img.format or "").upper()
                orientation = int(img.getexif().get(0x0112, 1) or 1)
                if (
                    max(img.size) <= max_side
                    and orientation in (0, 1)
                    and fmt in _MIME_BY_FORMAT
                ):
                    # 只解析了文件头，没有解码像素
                    return base64.b64encode(image).decode(), _MIME_BY_FORMAT[fmt]
                work = img.convert("RGB")
        except Exception as exc:
            logger.warning("VLM 输入图像解码失败（%s），按原始字节发送", exc)
            return base64.b64encode(image).decode(), "image/png"

        if max(work.size) > max_side:
            ratio = max_side / max(work.size)
            work = work.resize(
                (max(1, round(work.width * ratio)), max(1, round(work.height * ratio))),
                Image.LANCZOS,
            )
        buf = BytesIO()
        work.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode(), "image/png"

    # ------------------------------------------------------------------ 调用

    def _invoke_structured(self, messages: List, schema):
        """带重试的结构化输出。

        策略固定 ToolStrategy（method="function_calling"）：把 Pydantic 模型
        包装成一个 tool，强制模型以工具调用方式提交结果，规避 ModelScope
        兼容端点原生 json_schema 返回数组的 bug。
        （注：本环境 langchain-openai 版本仅接受 "function_calling" /
         "json_mode"，ToolStrategy 在此版本的方法名即 function_calling。）

        ToolStrategy 仍可能产出坏字段（类型不符 / 缺必填 / 枚举越界），因此：
        1. 最多重试 STRUCTURED_OUTPUT_ATTEMPTS 次，每次追加纠正提示；
        2. 模型把对象包成 list 的怪癖直接解包，不浪费一次重试；
        3. extra_body 不被服务端接受时自动换下一种关闭思考的写法（最多 3 种）；
        4. 重试耗尽后走一次 JSON 文本兜底；
        5. 仍失败才抛 StructuredOutputError，由节点降级而非让程序崩溃。
        """
        from pydantic import ValidationError

        self._log_thinking_state_once()
        last_exc: Optional[Exception] = None
        current = messages
        for attempt in range(1, STRUCTURED_OUTPUT_ATTEMPTS + 1):
            try:
                return self._invoke_tool_strategy(current, schema)
            except ValidationError as exc:
                last_exc = exc
                unwrapped = self._unwrap_list(exc, schema)
                if unwrapped is not None:
                    return unwrapped
                logger.warning(
                    "结构化输出校验失败（第 %d/%d 次）：%s",
                    attempt, STRUCTURED_OUTPUT_ATTEMPTS, exc,
                )
            except Exception as exc:
                last_exc = exc
                self._degrade_extra_body(exc)
                logger.warning(
                    "结构化输出调用失败（第 %d/%d 次）：%s",
                    attempt, STRUCTURED_OUTPUT_ATTEMPTS, exc,
                )
            if attempt < STRUCTURED_OUTPUT_ATTEMPTS:
                current = [*messages, HumanMessage(content=_RETRY_HINT)]

        try:
            return self._invoke_json_fallback(messages, schema)
        except Exception as exc:
            raise StructuredOutputError(
                f"{schema.__name__} 连续 {STRUCTURED_OUTPUT_ATTEMPTS} 次结构化输出失败，"
                f"JSON 兜底亦失败：{last_exc or exc}"
            ) from (last_exc or exc)

    def _invoke_tool_strategy(self, messages: List, schema):
        return self._model.with_structured_output(
            schema, method="function_calling"
        ).invoke(messages)

    @staticmethod
    def _unwrap_list(exc, schema):
        """Qwen3 系偶发把单个对象包成 list 返回：[{...}]，直接解包。"""
        from pydantic import ValidationError

        for error in exc.errors():
            value = error.get("input")
            if isinstance(value, list) and value:
                try:
                    return schema.model_validate(value[0])
                except ValidationError:
                    continue
        return None

    def _degrade_extra_body(self, exc: Exception) -> None:
        """当前 extra_body 不被服务端接受：换下一种关闭思考的写法，全部试完才放弃。

        只在错误「长得像参数被拒」时动作：超时/限流/5xx 等瞬时故障不该
        把关闭思考的候选写法耗尽（那等于静默开着思考跑完剩下的图）。
        每次更换/放弃都打 ERROR，部署时 grep 这行即可确认到底关没关掉。
        """
        if not _looks_like_param_rejection(exc):
            logger.warning(
                "VLM 调用失败但与关闭思考的参数无关（%s: %s），保留当前参数重试",
                type(exc).__name__, exc,
            )
            return
        with self._degrade_lock:
            # 无需关闭思考（显式开启），或已经放弃过 → 不再动作
            if not self._thinking_off or self._extra_body_disabled:
                return
            current = self._extra_body
            if current in self._payload_candidates:
                self._payload_candidates.remove(current)
            if self._payload_candidates:
                self._extra_body = self._payload_candidates.pop(0)
                logger.error(
                    "关闭思考模式的参数 %s 不被服务端接受（%s）；改试 %s",
                    current, exc, self._extra_body,
                )
            else:
                self._extra_body = {}
                self._extra_body_disabled = True
                logger.error(
                    "关闭思考模式的所有参数写法均被服务端拒绝（最后一次：%s）；"
                    "已放弃下发该参数——Qwen3 将输出思考 token：结构化输出不稳定、"
                    "单张耗时显著上升。请升级 vLLM，或设 VLM_ENABLE_THINKING=true "
                    "并在 prompt 层约束（不推荐）。",
                    exc,
                )
            self._model = self._build_model()
            self._thinking_state_logged = False

    def _log_thinking_state_once(self) -> None:
        """首次调用前把思考模式的实际状态打出来，便于部署后核对。"""
        if self._thinking_state_logged:
            return
        self._thinking_state_logged = True
        if not self._thinking_off:
            logger.warning(
                "VLM_ENABLE_THINKING=true：思考模式处于开启状态（显式配置），"
                "结构化输出可能不稳定、单张更慢"
            )
        elif self._extra_body:
            logger.info("VLM 思考模式：已关闭（extra_body=%s）", self._extra_body)
        else:
            logger.error("VLM 思考模式：未能关闭（参数均被服务端拒绝）")

    def _invoke_json_fallback(self, messages: List, schema):
        resp = self._model.invoke(messages)
        text = resp.content if isinstance(resp.content, str) else str(resp.content)
        data = self._extract_json(text)
        return schema.model_validate(data)

    @staticmethod
    def _extract_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        candidate = match.group(1).strip() if match else text.strip()

        # 定位首个 JSON 值（对象或数组）的起点
        starts = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
        if not starts:
            # 错误信息里不内嵌模型输出原文（含票据文本），只给长度便于排查
            raise ValueError(f"VLM 输出中未找到 JSON（输出长度 {len(text)}）")

        # raw_decode 只解析第一个完整 JSON 值并忽略其后的多余内容，
        # 兼容模型输出多个对象拼接（{...}{...}）或数组带额外文本等情况。
        try:
            data, _ = json.JSONDecoder().raw_decode(candidate[min(starts):])
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM 输出 JSON 解析失败（输出长度 {len(text)}）：{exc}") from exc

        if isinstance(data, list):  # 模型偶发把对象包成 list
            if not data:
                raise ValueError("VLM 输出的 JSON 数组为空")
            data = data[0]
        if not isinstance(data, dict):
            raise ValueError(f"VLM 输出的 JSON 不是对象（输出长度 {len(text)}）")
        return data

    # ------------------------------------------------------------------ 校验

    @staticmethod
    def _sanitize(analysis: VlmAnalysis, ocr_result: OcrResult) -> None:
        """校验 VLM 输出：块 ID 必须存在；bbox 与 OCR 坐标 IoU 过低
        时以 OCR 为准（防坐标幻觉）；value 必须是 text 子串；
        角度与四边形顶点一律从 OCR 拷贝（文字方向以检测器为准）。"""
        bbox_map = {b.block_id: list(b.bbox) for b in ocr_result.blocks}
        points_map = {b.block_id: [list(p) for p in b.points] for b in ocr_result.blocks}
        angle_map = {b.block_id: b.angle for b in ocr_result.blocks}
        text_map = {b.block_id: b.text for b in ocr_result.blocks}
        valid_kinds = {k.value for k in SensitiveKind}
        cleaned: List[SensitiveBlock] = []
        for sb in analysis.sensitive_blocks:
            if sb.block_id not in bbox_map:
                logger.warning("VLM 引用了不存在的块 %s，已丢弃", sb.block_id)
                continue
            ocr_bbox = bbox_map[sb.block_id]
            if not sb.bbox or len(sb.bbox) != 4:
                if sb.bbox:
                    logger.warning(
                        "块 %s 的 bbox 长度异常（%s），以 OCR 坐标为准",
                        sb.block_id, sb.bbox,
                    )
                sb.bbox = ocr_bbox
            elif _iou(sb.bbox, ocr_bbox) < 0.3:
                logger.warning(
                    "块 %s 的 bbox 与 OCR 坐标 IoU 过低（%s vs %s），以 OCR 为准",
                    sb.block_id, sb.bbox, ocr_bbox,
                )
                sb.bbox = ocr_bbox
            sb.points = points_map.get(sb.block_id) or []
            sb.angle = angle_map.get(sb.block_id, 0.0)
            if sb.method not in RedactionMethod.__members__.values():
                sb.method = RedactionMethod.WHITE_MASK
            if sb.kind and sb.kind not in valid_kinds:
                logger.warning("块 %s 的 kind=%s 不合法，按 generic 处理", sb.block_id, sb.kind)
                sb.kind = "generic"
            if sb.value and sb.value not in text_map.get(sb.block_id, ""):
                # 不打印 value 原文：脱敏工具的日志不应落入任何敏感信息
                logger.warning(
                    "块 %s 的 value 不是块文本子串（长度 %d），已清空（退化为纯覆盖）",
                    sb.block_id,
                    len(sb.value),
                )
                sb.value = ""
            cleaned.append(sb)
        analysis.sensitive_blocks = cleaned


def _iou(a: List[int], b: List[int]) -> float:
    """两个 AABB 的 IoU。"""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0
