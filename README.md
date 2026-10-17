# imageMask — 医疗票据图片隐私脱敏工具

基于 **LangGraph + OCR + VLM** 的确定性脱敏流水线，支持批量跑批（tqdm 进度）、断点续传、大图等比缩小、文字方向矫正、假数据等长替换。

> **本文档面向后续接手维护/开发的编程 AI**：先读「架构与核心机制」和「踩坑记录」再动手；
> 改任何节点前先看「约束与禁止项」。

---

## 1. 项目定位

对医疗票据/检查单图片自动脱敏患者隐私，核心链路：

```
OCR 识别文字块与坐标 → VLM 判断敏感块 → 假数据替换/遮罩 → 质检（闭环） → 通过/回退/兜底
```

- 流程控制全部交给 **LangGraph StateGraph**（确定性流水线，**不用 ReAct**）；
- LLM 只出现在 VLM（敏感判断）与 QA（语义质检）两个节点内部；
- OCR、脱敏、正则规则、重 OCR 质检都是**确定性代码**，可单测、可复现。

当前脱敏范围（5 类，封闭集合）：**姓名、身份证号、手机号、社保号/医保卡号、能精确找到人的详细住址**（精确到建筑/楼层/房间；城市/省级地名不脱敏）。固定电话明确不脱敏。医院、科室、医务人员姓名、日期、项目、金额、票据号、流水号等一律不脱敏。

匹配方式：**身份证号、手机号走正则规则通道**（格式唯一性强）；姓名、社保号、详细住址无稳定格式，由 VLM 结合上下文判断。

---

## 2. 架构

### 2.1 状态图

```
START → ocr → vlm → redact → qa → route_after_qa
                                    ├─ end       → finalize → END
                                    ├─ retry_vlm → vlm          （带质检反馈+mask图重判）
                                    └─ fallback  → fallback → finalize → END
```

- `route_after_qa`：未开质检 → end；**状态已 failed（重试轮里上游失败）→ end**；无报告 → end；passed → end；重试耗尽 → fallback；否则 retry_vlm。**每个分支都有出口**（见踩坑 #16）。
- `finalize`：开启方向矫正时，把脱敏图逆旋转回原方向（见 3.5）。
- `recursion_limit = 3*(max_retries+1)+6`，由 `compute_recursion_limit()` 计算，**禁止硬编码省略**。

### 2.2 分工原则（改代码前必须理解）

| 组件 | 职责 | 明确不做 |
|---|---|---|
| OCR（RapidOCR） | 文字块、四边形坐标、行方向角 | 不做敏感判断 |
| 规则通道（正则） | 身份证号/手机号的确定性兜底（固话已按需求移除，见坑 #12） | 不判姓名/社保号/地址 |
| VLM | 语义判断：哪些块敏感（输出 JSON） | **不产坐标、不生成脱敏图、不选脱敏方式**（方式由 `DEFAULT_METHOD`/`--method` 统一决定） |
| 假数据映射器 | 确定性等长假值（纯函数） | 不依赖 VLM、不查表 |
| 脱敏工具（PIL） | 白底覆盖 + 假数据重绘 | 不修改 OCR 坐标；不做二维码检测/打码 |
| 质检 | 确定性预检 + VLM 语义质检 | 不无限回退 |

**坐标来源永远是 OCR**：VLM 输出 `bbox` 但 sanitize 时与 OCR 坐标做 IoU 校验（<0.3 视为幻觉，以 OCR 为准）；`points`/`angle` 一律从 OCR 拷贝。

---

## 3. 核心机制详解

### 3.1 假数据映射（`src/fake_data.py`）

- **确定性**：`seed = sha256(敏感值)` 驱动 `random.Random`，同一敏感值在任何图片/批次/进程中映射到同一假值（纯函数，无需存储）。
- **等长是硬约束**：每个生成器保证 `len(假值)==len(原值)`；任何生成器失控都退化为 `_map_generic` 逐字符同类映射。
- 各类型算法：
  - 姓名：姓氏池 + 名用字池（`rng.sample` 不重复）；
  - 手机号：`1` + `3~9` + 9 位随机（号段合法）；
  - 身份证号：保留前 6 位地区码（公开信息）与第 17 位性别位，生日 1940~2010 合法，**第 18 位按 GB 11643-1999 校验位计算**（假号格式完全合法）；
  - 社保号/医保卡号/通用：逐字符同类映射（数字→数字、字母→字母、汉字→常用字、结构字符保留）；
  - 地址：五步顺序正则替换省/市/区/路名（词池等长选择）+ 门牌号等长数字映射，**保留全部行政结构词**（省市区路号）。
- **替换粒度**：只替换块文本中的敏感值子串（`value`），保留"姓名："等标签——`text.replace(value, fake)`。这就是 VLM 必须输出 `value`（块文本精确子串）的原因。

### 3.2 VLM 交互（`src/tools/vlm_tool.py`）

- 输入：**原图（或转正图）+ OCR 块 JSON**（`block_id/text/bbox` 三要素，刻意不含 confidence）。
- 结构化输出：`with_structured_output(Pydantic Schema)`；三层容错：
  1. Qwen3 偶发把对象包成 list（`[{...}]`）→ 捕获 `ValidationError` 从 `error.input` 解包；
  2. 模型不支持 function calling → 普通 invoke + `_extract_json`（支持 ```json 围栏/裸 JSON/顶层 list）；
  3. `_sanitize` 语义校验：block_id 必须存在、bbox IoU 校验、value 必须是块文本子串（否则清空退化为纯覆盖）、kind 必须在枚举内。
- 回退判断（QA 未通过）：输入追加**质检意见 + 当前 mask 图（第二张图）**。
- QA 质检（`qa_verify`）：输入原图 + mask 图 + OCR JSON + 敏感块 JSON + **各块已替换的假数据（绘制文本，必须传 `redaction_result`，见坑 #13）** + 本地预检结果 + 历史意见 → `QaVerdict`。
- 关闭 Qwen3 思考模式：`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`（不关则结构化输出不稳定）。
- **思考模式覆盖两条调用路径**：敏感判断与语义质检共用同一个模型实例、同一份 extra_body，不存在"一条关一条没关"。服务端不接受当前写法时自动换下一种候选写法（共 3 种：`chat_template_kwargs` / 顶层 `enable_thinking` / `thinking.type`），全部被拒才放弃并打 ERROR；瞬时错误（超时/限流/5xx）不触发换写法，避免一次网络抖动就静默开着思考跑完整批。部署后核对：`grep "思考模式" logs/run_*.log` 应出现 `已关闭（extra_body=...）`。
- `VLM_MAX_TOKENS=4096`：26 块的结构化 JSON 会超 2048，**不要调小**。

### 3.3 质检（`src/tools/qa_tool.py`，两层）

1. **确定性预检**（零成本）：重 OCR 残留 + 正则兜底 + 几何误伤。发现硬残留直接 failed，不花 VLM 调用。
   - 残留判定 = 归一化包含 + 长数字段完整出现（**刻意不用相似度**，见坑 #3）；
   - 正则兜底会剔除已绘制的假数据（假号格式合法，见坑 #4）；
   - 误伤判定 = 非敏感块**中心点**是否被遮罩多边形覆盖（不用外接矩形重叠比，竖排会误报）。
2. **VLM 语义质检**：抓确定性通道查不了的语义漏判（如漏掉的姓名），输出 `QaVerdict` 含 `revised_block_ids` 供回退修正。

**为什么质检要再跑一次 OCR，不能复用首轮结果**：首轮 OCR 描述的是**脱敏前**的原图，质检要回答的是「**输出图**里的敏感信息还在不在」——这是两张不同的图。复用首轮结果只能证明"我们画了框"，证明不了"字没了"。只有重 OCR 才能发现的失效模式：

- 遮罩框偏移/偏小（OCR bbox 误差、四边形顶点异常）导致原值仍在框外可读；
- blur/mosaic 强度不足，原文字仍可辨认；
- 假数据重绘字号过大出框，压住相邻的非敏感文字（配合几何误伤检查）；
- 首轮 OCR 自己漏识别的文字（首轮没读到 ≠ 不存在）。

误伤检查用的是**首轮** OCR 的 bbox + 遮罩几何，本来就不需要重 OCR；重 OCR 只服务于「残留 + 正则兜底」两项。想省这 ~1.4s 的正确方向是「只重识别遮罩区域」而不是「复用首轮结果」——前者仍然在读输出图。另外这 ~1.4s 现在已被 `QA_PRECHECK_BACKGROUND` 藏进 VLM 调用里，不再计入单张耗时（见 3.8）。

### 3.4 批量与断点续传（`src/batch.py`）

- case = 输入文件夹的直接子目录；图片发现用 `glob(**/*, recursive=True)`；
- 输出**完全镜像**输入相对路径与文件名，仅顶层目录加 `_masked`（`folder1 → folder1_masked`）；
- 启动时先扫出图片总数并打印 `共发现 N 张图片`，随后用 **tqdm 进度条**展示实时进度（跳过的图也计数，保证走得满）；进度条只在 TTY 下显示，`nohup`/重定向到文件时自动关闭，避免刷日志；
- 断点：每处理完一张图立即原子写 `batch_progress.json`（临时文件 + `os.replace`），passed 的图重启后跳过，failed/max_retry 重试；`--no-resume` 强制全量；
- **续跑会校验产物**：进度记录 passed 但输出文件已不存在（被删/移走）时重新处理，不会静默跳过——避免"报告说成功、目录里没文件"；
- **输出目录不允许落在输入目录内**（`--output` 指到里面直接报错）：否则脱敏结果会写回原图位置，违反"原图必须保留"，下次运行还会把输出目录当成新 case 扫进去；
- **Ctrl+C 只取消排队中的 case**：并行模式下中断不会把剩余 case 全跑完（那要等几小时），在跑的那几个（≤ `--workers`）收尾后退出，进度不丢；
- 图结束状态只认 `passed/failed/max_retry`，出现中间态一律按 `failed` 记（防止"卡住"被算成成功）；
- 并行粒度是 **case**（case 内图片串行），避免打爆 VLM API；进度写盘有 `threading.Lock` 保护；
- 汇总报告 `batch_report.json`（含每图状态/输出/QA 结论/耗时）。

### 3.5 方向矫正（`OCR_ORIENTATION_CORRECT=true`）

- RapidOrientation 检测全局角度 → **返回值 = 需逆时针旋转的度数**（已用特征点像素测试固化，见坑 #6）；
- 转正后跑 OCR，**整条管线（OCR/VLM/脱敏/QA）都在转正图坐标系**，`ocr_node` 把状态中的 `original_image` 替换为转正图、记录 `rotation`；
- `finalize_node` 是唯一逆旋转的地方（`rotate(-rotation)`），输出与原图方向一致；
- 好处：假数据在水平图中渲染后随图旋转，**方向天然与原文字一致**；VLM 看正向文字；
- 引擎缺失自动降级为原图直跑（不阻断）。

---

### 3.6 大图预处理（`src/image_preprocess.py`）

- 入口统一做：**超过阈值（默认 5MB）的图片等比缩小后重编码**，再进入 OCR/VLM/脱敏/质检/finalize。全链路共用缩小后的同一份字节，**坐标天然自洽，无需反变换**；
- 只缩不裁、保持宽高比；**输出编码格式与输入一致**（JPEG 仍 JPEG、PNG 仍 PNG），不破坏「扩展名与内容格式一致」（坑 #15）；
- 缩小策略：按 `0.85` 逐轮缩放 + 重编码，直到体积达标或长边触到 `MIN_LONG_SIDE=800`（再缩会明显掉 OCR 识别率），最多 8 轮；
- 降级：无法解码 / 重编码失败 / 结果反而更大时**一律返回原始字节**并告警，预处理不能成为新的故障点；
- 开关：`--max-image-mb <MB>` 调阈值（`0` 或负数 = 关闭），`--no-shrink` 直接关闭；命中缩小的张数计入报告 `statistics.shrunk_images` 与每图 `shrunk` 字段。

### 3.7 EXIF 方向归一化（`src/image_preprocess.normalize_exif_orientation`）

- 手机拍的票据常带 `Orientation=6/8`（像素横向存储、靠 EXIF 竖显）。后续节点只读像素，输出图又不带 EXIF，不处理就会「原图竖着看、脱敏图横着躺」，OCR 在侧倒页面上也会掉识别率；
- 入口在缩小之前统一 `exif_transpose` 转正（只解析文件头判断，Orientation=1 时**原字节不变**，不重编码）；
- 与 3.5 的全局方向矫正互补：EXIF 是元数据级的 90° 修正，RapidOrientation 处理「拍歪但没写 EXIF」的情况。

### 3.8 单张耗时构成与调优入口

本地实测（900×560 票据，20 核 CPU，RapidOCR；VLM 两项为 27B vLLM 估值）：

| 阶段 | 耗时 | 说明 |
|---|---|---|
| OCR 首轮 | ~1.4s | 检测在固定 736 长边下进行，**与输入分辨率基本无关** |
| VLM 敏感判断 | 2~4s | 1 张图 + OCR JSON → 结构化 JSON |
| 脱敏 | <10ms | PIL |
| 质检重 OCR | ~1.4s | 只做文字比对，已跳过方向检测 + 按 `QA_REOCR_MAX_SIDE` 缩小 |
| VLM 质检 | 3~6s | 原图 + 脱敏图 + JSON → 结论 |

结论：**两次 VLM 调用占 ~70%，两轮 OCR 占 ~25%**。

**已实现的并发优化（`QA_PRECHECK_BACKGROUND=true`，默认开）**：确定性预检（重 OCR + 正则 + 几何误伤）放后台线程，与 VLM 语义质检同时跑——两者互不依赖，于是「先重 OCR 再调 VLM」的整段串行时间被藏进 VLM 调用里。实测 `verify` 从 `重OCR + VLM` 变为 `max(重OCR, VLM)`，**单张省 ~1.2~1.5s**。代价：VLM 不再带 `deterministic_hints`（那只是"供参考"上下文）；确定性结论仍然权威——发现硬残留就直接判 failed，无论 VLM 说什么。设 `false` 回到旧的「串行 + 带 hints」行为。

**实测无效、已避开的弯路**：
- 并发 OCR（多引擎实例）：20 核 CPU 上 4 张图并发，1 实例 5.7s / 2 实例 7.0s / 4 实例 12~14.5s。onnxruntime 单会话已吃满全部核心，多会话互相争抢，加上 PIL/numpy 预处理受 GIL 串行化。`OCR_ENGINE_INSTANCES` 默认 1（等价于旧的全局锁），只有限制每会话线程数或换 GPU 推理时调大才有意义。
- 多进程：同理，CPU 已饱和，拆进程不会变快，只会让断点续传/进度条复杂化。
- asyncio：单张图内各阶段是严格数据依赖（OCR→VLM→脱敏→质检），没有可并发的东西；跨图的 I/O 重叠已由 case 级线程池承担，事件循环不会额外增加并发度。

**剩下的调优优先级**：

1. **先确认思考模式真的关掉了**：日志出现 `带 extra_body 的调用失败…降级去掉 extra_body` 说明 vLLM 不接受 `chat_template_kwargs`，Qwen3 会输出大量推理 token，两次调用各慢数秒。升级 vLLM 或改用其支持的关闭方式。
2. `VLM_QA_MASK_IMAGE_SIDE=512`（默认）：质检只需判断遮敏位置与文字是否消失，vision token 约为 1024 的 1/4。
3. `VLM_MAX_IMAGE_SIDE` 1024→768：两张图 token 都降 ~44%；先用几张真实票据对比 QA 通过率再定。
4. `QA_REOCR_MAX_SIDE=1280`：重 OCR 先缩小，实测省 ~0.2s/张。
5. 跑批 `--workers` 建议 3~4：OCR 阶段 CPU 已饱和（串行），再加并发只会让 vLLM 排队、单张更慢。
6. 想大幅提速只能减 VLM 调用次数（每张固定 2 次）。质检的确定性层零成本且必跑；VLM 语义质检抓的是确定性通道查不到的漏判（如漏掉的姓名），**不建议默认关闭**。

---

## 4. 数据契约

### 4.1 `RedactionState`（`src/graph_state.py`）

节点**只返回需要更新的字段**（dict merge），禁止整体替换 state。字段：`image_path / original_image / rotation / output_path / ocr_result / vlm_analysis / redaction_result / qa_report / qa_enabled / retry_count / max_retries / feedback / status / error`。

失败约定：节点捕获异常后返回 `{"status": "failed", "error": "[节点] 原因"}`（不裸抛、不吞）；后续节点检测到 `status=="failed"` 短路返回 `{}`，图自然走到 END。

### 4.2 关键 schema（`src/schemas/`）

- `OcrTextBlock`：`block_id / text / bbox(外接矩形) / points(四边形顶点) / angle(行方向角) / confidence`；
- `SensitiveBlock`：`block_id / text / bbox / points / angle / value(精确子串) / kind / method / source / reason`；
- `RedactedBlock`：命中记录（审计用，**不记敏感原文**，text 记假数据）；
- `QaReport` / `QaVerdict`：质检结论（status/reason/feedback/残留/误伤/revised）。

---

## 5. 目录结构与扩展指南

```
main.py                  CLI：单图 / 批量（文件夹自动触发）/ --make-sample
requirements.txt         核心依赖 + Ubuntu 系统依赖说明
.env / .env.example     外网开发配置（ModelScope）
.env.production.example 内网跑批模板（vLLM，含启动命令）
src/
├── config.py            Settings；双环境：IMAGE_MASK_ENV_FILE 指定配置文件
├── graph.py             图编排 + route_after_qa + recursion_limit 公式
├── graph_state.py       RedactionState
├── sensitive_rules.py   正则规则通道（与 VLM 取并集，按 block_id 去重 VLM 优先）
├── fake_data.py         FakeDataMapper（确定性等长假数据）
├── image_preprocess.py  入口大图等比缩小（>5MB，可 --no-shrink 关闭）
├── schemas/             5 个 Pydantic 模型
├── tools/               4 个工具 + 懒加载单例门面（tests 可向 _cache 注入假工具）
└── nodes/               6 个节点 + common.py（fail/already_failed/require 辅助）
tests/                   167 个单测 + conftest（隔离 .env）+ e2e_local/remote
```

**常见扩展点**：

| 需求 | 做法 |
|---|---|
| 新增敏感类型 | `SensitiveKind` 加枚举 → VLM system prompt 加描述 → `fake_data.py` 加 `_map_xxx` 生成器（保证等长）→ 测试 |
| 新增脱敏方式 | `RedactionMethod` 加枚举 → `redaction_tool._apply_one` 加分支 |
| 新增节点 | `nodes/xxx_node.py` → `nodes/__init__.py` 导出 → `graph.py` 注册 + 连线 → **同步改 `compute_recursion_limit`** |
| 换 OCR 引擎 | `OCR_ENGINE=paddle`（已适配 2.x/3.x）；或仿照 `OcrTool` 加适配分支 |
| 换 VLM | 只改 `VLM_BASE_URL/VLM_MODEL/VLM_API_KEY` 三项 |
| 票据类型识别/人工复核/审计日志 | 往 StateGraph 加节点（**不要改回 ReAct**） |

---

## 6. 约束与禁止项（来自项目规格 + 实践）

- 禁止把 OCR/VLM/脱敏/质检改成 ReAct 工具；
- 禁止节点直接修改整个 state（只返回增量字段）；
- 禁止在节点内硬编码循环（重试由图控制）；
- 禁止删除 `retry_count`/`max_retries`/`recursion_limit`；
- 禁止让 VLM 直接输出脱敏图片或自行估算坐标（坐标以 OCR 为准）；
- 禁止质检不通过时无限回退（`max_retries` 默认 3，耗尽走 fallback 保留现状并标记 `max_retry`）；
- 禁止 `qa_enabled=False` 时仍执行质检逻辑；
- 禁止输出文件加 `_masked` 等后缀（目录镜像，顶层目录已带标识）；
- 禁止审计日志记录敏感原文（记块 ID + 假数据 + 脱敏方式）。

---

## 7. 踩坑记录（改动相关代码前必读）

1. **Windows 路径分隔符**：进度 key 必须 `relative_to(...).as_posix()`，否则 Linux 生成的进度 Windows 读不到。
2. **numpy 布尔求值**：`getattr(res, "boxes", []) or []` 对 ndarray 会抛 "truth value is ambiguous"——必须显式 `is None` 判断。
3. **残留检测不能用相似度**：票据号与医保号常共享日期前缀，`SequenceMatcher ratio 0.727` 会误报。用包含匹配 + 长数字段（≥8 位）完整出现。
4. **假数据会被正则兜底误报**：假身份证/假手机号格式合法，QA 正则扫描前必须剔除已绘制的假数据文本（`_looks_like_drawn_fake`）。
5. **Qwen3 结构化输出怪癖**：偶发把单个对象包成 list 返回；在 `_invoke_structured` 捕获 `ValidationError` 解包，别当成模型故障。
6. **OCR 输出无法反推图像转向**：检测器对旋转文本鲁棒，且四边形角度被栈内归一化（顺/逆时针转正后都报 ≈0°）——方向语义只能以 RapidOrientation 输出为准（返回值 = 需**逆时针**旋转度数），已用特征点像素测试固化，**不要依据 OCR 块数/置信度/角度判断转向**。
7. **PIL 旋转语义**：`img.rotate(-90, expand=True)` 与 `cv2.ROTATE_90_CLOCKWISE` 像素一致；PIL 正角度 = 视觉逆时针。转正/还原必须互逆。
8. **测试会被 .env 污染**：pydantic-settings 默认读项目 `.env`；`tests/conftest.py` 用 `IMAGE_MASK_ENV_FILE=os.devnull` 隔离，新测试不要绕过它。
9. **VLM 的 `value` 必须是块文本精确子串**：这是保标签替换（`姓名：张三 → 姓名：孙礼`）的前提；不是子串会被 sanitize 清空退化为纯覆盖。
10. **降级链无处不在**：无中文字体 → 纯白底；方向引擎缺失 → 原图直跑；structured output 失败 → JSON 解析；QA 的 VLM 失败 → failed + 明确原因。新增外部依赖时保持同样的降级思路。
11. **opencv 在无头 Linux** 需要 `libgl1`；中文字体缺失时假数据重绘静默退化为色块（有 warning）——部署检查这两个。
12. **规则通道只收格式唯一性强的类型**（真实票据教训）：固定电话正则 `0\d{2,3}-?\d{7,8}` 与票据号码/流水号同构（票据号 `0137126549` 被误判为固话并替换）。固定电话经需求确认**不脱敏**，正则已移除；规则通道现只留身份证、手机号。新增规则类型前先问：这个格式会不会被非敏感数字串撞到？
13. **VLM 质检必须知道假数据的存在**（真实票据教训）：QA 输入若只给"哪些块敏感"而不给"这些块已替换成什么"，VLM 看到 mask 图里的假数据文字会误判"仍清晰可见"，导致无谓重试直至 max_retry。`qa_verify` 的 `redaction_result` 参数必须传，QA prompt 中已明确"看到假数据是预期的，不算残留"。
14. **vLLM 的 extra_body 兼容性**（部署教训）：`chat_template_kwargs`（关闭思考模式）不被旧版 vLLM 接受时会直接 400，导致整个跑批全灭。`VlmTool` 已内置自动降级：带 extra_body 调用失败 → 去掉重试一次（日志 warning）。`tests/e2e_local.py` 等测试替身同步 `qa_verify`/`analyze` 签名时勿漏 `redaction_result`。
15. **输出扩展名必须与内容格式一致**（部署教训）：`.jpg` 输出早前直接写 PNG 字节（文件头是 PNG），下游按扩展名处理的工具会出问题。`RedactionTool._save` 与 `finalize_node` 现统一走 `format_for_suffix` + `encode_image`（JPEG/WEBP/BMP/TIFF/PNG 全覆盖），新增扩展名只改那张表。
16. **重试轮里上游失败会空转**：`qa_node` 在 `status=="failed"` 时短路返回 `{}`，`qa_report` 仍是上一轮的 `failed`。`route_after_qa` 只看报告会判 `retry_vlm`，而 vlm/redact/qa 全部短路后路由不变 → 空转至 `recursion_limit`，报 `GraphRecursionError` 并掩盖真实错误。路由第一件事必须判 `status=="failed"`。
17. **`--method` 曾经对 VLM 块无效**：只有规则通道读 `DEFAULT_METHOD`，VLM 块永远用 prompt 里写死的 `white_mask`，`--method blur` 只影响身份证/手机号块。现由 `vlm_node` 统一覆盖 `SensitiveBlock.method`（`resolve_default_method()` 负责非法值回退），**VLM 只判断“哪些块敏感”，不决定“怎么脱”**。

---

## 8. 配置与部署

### 8.1 双环境

| 环境 | 配置文件 | VLM |
|---|---|---|
| 外网开发 | `.env`（默认） | ModelScope `Qwen/Qwen3.8-27B` |
| 内网跑批 | `.env.production` | vLLM `qwen3.8-27b` |

切换方式（代码零改动）：
```bash
IMAGE_MASK_ENV_FILE=.env.production python main.py /data/folder1 --workers 4
```

vLLM 启动参考（`.env.production.example` 内有完整版）：
```bash
vllm serve /models/qwen3.8-27b --served-model-name qwen3.8-27b \
     --port 8000 --max-model-len 16384 --limit-mm-per-prompt image=4
```

Ubuntu 系统依赖：`sudo apt install -y libgl1 libglib2.0-0 fonts-noto-cjk`，Python 3.11。

### 8.2 常用命令

```bash
# 单图
python main.py data/ticket.png
# 批量（folder1 → folder1_masked，断点续传默认开启）
python main.py data/folder1 --workers 4
# 强制全量 / 跳过已有输出 / 关质检 / 指定脱敏方式
python main.py data/folder1 --no-resume | --skip-existing | --no-qa | --method blur
# 大图预处理（默认 >5MB 等比缩小）：关闭 / 改阈值
python main.py data/folder1 --no-shrink | --max-image-mb 10
# 生成示例票据
python main.py --make-sample
```

退出码：单图 0=passed；批量 0=全部 passed，1=存在 failed/max_retry。

**完整 Ubuntu 服务器部署 + 跑批 + 续跑 + 排障手册：[`docs/ubuntu-runbook.md`](docs/ubuntu-runbook.md)**

---

## 9. 测试与验证入口

```bash
python -m pytest tests/ -q                 # 167 个单测（无需 VLM/网络，秒级）；
                                           # 未装 rapid-orientation 时 2 个方向测试 skip
python tests/e2e_local.py                  # 本地全链路（启发式假 VLM + 真实 OCR/脱敏/质检）
python tests/e2e_remote.py                 # 真实 VLM 全链路（需 .env 配好 API Key，会消耗配额）
```

`requirements.txt` 锁死全部确切版本（= 上述测试跑通的那套环境，Python 3.11）。
注意 `rapidocr` 自己不依赖 `onnxruntime`（它的推理引擎），`requirements.txt`
里显式锁了——漏掉它全新环境会在第一张图报 `ModuleNotFoundError`。

测试文件职责：`test_graph`（流程/路由/失败传播）、`test_tools`（脱敏像素/质检判定/假数据不误报）、`test_fake_data`（映射确定性/等长/合法性）、`test_batch`（glob/镜像/断点/并行/大图缩小）、`test_preprocess`（缩小阈值/等比/格式一致/降级）、`test_rotation`（四边形角度/误伤）、`test_orientation`（转正坐标系/finalize 往返/方向语义，夹具图用 `make_sample` 在 tmp 目录自生成，不依赖 data/）。`fakes.py` 提供全套测试替身；新工具签名变化时先同步 fakes。

**改完必跑**：`pytest tests/ -q` 全绿 + 至少一次 `e2e_local`；涉及 VLM/方向/假数据的改动再跑 `e2e_remote`。

---

## 10. 已验证的能力基线（回归参考）

**合成测试图**：
- 横版/竖版票据均正确处理（竖版：检测 90° 逆时针 → 转正管线 → 逆旋转还原，VLM 目验水平正向）；
- 假数据等长、同值同假、格式合法（身份证校验位正确）；
- 质检闭环：VLM 语义质检 + 确定性预检；重试受控；
- 批量：目录镜像、断点续传（强杀进程后重跑跳过已完成）、case 级并行；
- 双环境：ModelScope（外网）/ vLLM（内网）配置切换验证通过。

**真实票据（实拍，含红章/透视变形/光照不均）**：

| 票据 | 结果 |
|---|---|
| 四川省医疗住院收费票据 | 交款人姓名→假名、医保编号→等长假号；票据号码/收费员/日期/金额完整保留；零泄露；QA passed 零重试 |
| 广东省医疗住院收费票据 | 姓名→假名；该票医保编号字段为空，VLM 正确未动；其余非敏感内容完整；QA passed 零重试 |

真实票据验证固化的行为（回归时不应改变）：
- 票据代码/号码/校验码/流水号/病历号/金额/日期/章上文字一律不脱敏；
- OCR 对实拍图有少量误字（如“陈无咎→陈无餐”），假数据按 OCR 文本等长替换，不影响安全（整块覆盖重绘，原值不可见）；
- QA 能正确处理“假数据已在图中”与“标签保留”策略，无误判重试。

**已知限制**：
- **不做二维码/条码检测与打码**（按需求移除）：QR 内容通常是票据号/查验 URL，按当前 5 类脱敏范围均不脱敏；OpenCV 检测在实拍票据上耗时且常因透视变形漏检，性价比低。若将来需要打码，应由 VLM 圈定区域或换鲁棒检测库，不要恢复旧的 `QRCodeDetector` 路径。
