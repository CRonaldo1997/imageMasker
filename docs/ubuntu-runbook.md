# imageMask Ubuntu 服务器操作手册

> 面向：在内网 Ubuntu 跑批机上部署本工具、使用 172.16.84.211 上的 vLLM（Qwen3.8-27B）做医疗票据批量脱敏。
> 目标：照着命令从上往下敲即可完成一次完整跑批，且中途断了能接着跑。

---

## 0. 速查卡（已经部署过，直接看这里）

```bash
cd /opt/imageMask && source .venv/bin/activate
export IMAGE_MASK_ENV_FILE=/opt/imageMask/.env.production

python main.py data/testCase --workers 3        # 跑批（断了重跑这行即可续上）
tail -f run.log                                 # 看进度
jq '.summary, .statistics' data/testCase_masked/batch_report.json   # 看统计
```

---

## 1. 一次性环境准备

### 1.1 系统依赖

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 fonts-noto-cjk python3.11 python3.11-venv jq

# 说明：
# libgl1 / libglib2.0-0  → opencv 运行时必需，缺了 import cv2 直接失败
# fonts-noto-cjk         → 假数据重绘中文字体，缺了敏感值会被画成纯色块（有 warning）
# jq                     → 可选，查看 batch_report.json 方便
```

### 1.2 代码与虚拟环境

```bash
sudo mkdir -p /opt/imageMask && sudo chown $USER:$USER /opt/imageMask
# 上传代码（任选其一）
#   rsync -av --exclude .venv --exclude data ./imageMask/ user@server:/opt/imageMask/
#   git clone <repo> /opt/imageMask

cd /opt/imageMask
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 方向矫正：.env.production 里 OCR_ORIENTATION_CORRECT=true，必须装这个，
# 否则会静默降级为原图直跑（只有一条 warning，竖版票据识别率下降）
pip install rapid-orientation==0.0.11
```

> 内网无外网也能装：`rapidocr` 的 ONNX 模型打在 wheel 里，不需要运行时联网下载；用内网镜像即可 `pip install -r requirements.txt -i http://<内网pypi>/simple --trusted-host <host>`。
> `requirements.txt` 全部锁了确切版本（就是跑过 167 个单测 + `e2e_local` 的那套环境）；升级任何包之前先在本地跑一遍 `pytest tests/ -q` 和 `e2e_local`。
> 无 GPU 的机器也能跑本工具（OCR 走 CPU onnxruntime），只要求能访问 vLLM 的 HTTP 接口。

### 1.3 目录约定（很重要）

```
/data/testCase/            ← 输入根目录
├── case1/                 ← case = 输入根目录的【直接子目录】，一个 case 可以再嵌套多层
│   ├── 0.png
│   └── 123/L4iX.jpg
├── case2/
└── case3/

/data/testCase_masked/     ← 输出（自动生成，默认 = 输入同级 + _masked）
├── case1/0.png            ← 目录结构和文件名完全镜像，不加任何后缀
├── case1/123/L4iX.jpg
├── batch_progress.json    ← 断点进度（续跑依据，别手动改）
└── batch_report.json      ← 汇总报告（统计口径都在这）
```

- 图片扩展名支持：png / jpg / jpeg / bmp / tif / tiff / webp（大小写不敏感）。
- **case 层级决定并行度**：一张图不是 case，一个目录才是。所有图平铺在根目录 = 0 个 case = 什么都不跑。
- 输入路径建议放 `/data` 这类稳定位置，**换路径会导致断点失效**（见第 6 节）。

---

## 2. 配置

### 2.1 配置文件

仓库里已提供 `/.env.production`（内网版，已按 172.16.84.211 填好）。核心内容：

```ini
VLM_BASE_URL=http://172.16.84.211:8000/v1
VLM_MODEL=/home/lenovo/Qwen3.8-27B     # 与 vLLM 的服务名严格一致
VLM_API_KEY=EMPTY                      # vLLM 未设 --api-key 时占位即可
VLM_ENABLE_THINKING=false              # 关思考模式，结构化输出才稳
VLM_MAX_TOKENS=4096                    # 26+ 块 JSON 会超 2048，勿调小
VLM_TIMEOUT=600                        # 本地 27B 双图推理慢

OCR_ENGINE=rapid
OCR_ORIENTATION_CORRECT=true           # 需已装 rapid-orientation
DEFAULT_METHOD=white_mask              # white_mask（假数据重绘）| blur | mosaic
FAKE_DATA_ENABLED=true                 # 等长假数据替换
```

### 2.2 生效方式

```bash
export IMAGE_MASK_ENV_FILE=/opt/imageMask/.env.production
```

三条铁律：

1. 必须在**启动 python 之前** export（配置在 import 时读取，中途设置无效）。
2. 设了它之后**完全不再读 `.env`**，两个文件不会合并。
3. 用**绝对路径**（相对路径按当前工作目录解析，换个目录跑就静默失效回落到默认 `.env`，即外网 ModelScope）。

**验证配置真的生效了**（每次部署/改配置后都做一次）：

```bash
python -c "from src.config import get_settings as g; s=g(); \
print(s.vlm_base_url, s.vlm_model, s.vlm_timeout)"
# 期望：http://172.16.84.211:8000/v1 /home/lenovo/Qwen3.8-27B 600
# 若打印 modelscope 地址 → IMAGE_MASK_ENV_FILE 没传进去
```

---

## 3. vLLM 服务端（172.16.84.211）

### 3.1 启动命令（含必要补丁）

```bash
vllm serve /home/lenovo/Qwen3.8-27B \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.93 \
  --max-num-seqs 256 \
  --limit-mm-per-prompt image=4 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data
```

与原命令的唯一差异是 **`--limit-mm-per-prompt image=4`，这一项必须有**：质检节点一次传「原图 + 脱敏图」两张图，vLLM 默认每 prompt 只允许 1 张，不加会 400 报错导致全部图片 failed。

### 3.2 服务自检

```bash
# 在跑批机上执行
curl -s http://172.16.84.211:8000/v1/models | jq -r '.data[].id'
# 输出必须与配置里的 VLM_MODEL 完全一致：/home/lenovo/Qwen3.8-27B
```

### 3.3 关于工具调用解析器

`--enable-auto-tool-choice --tool-call-parser qwen3_coder` 是本工具 `with_structured_output(method="function_calling")` 所依赖的。若日志频繁出现：

```
结构化输出调用失败（第 1/3 次）... 
```

且最后降级到 `JSON 兜底` 才成功，说明工具调用格式没对上。按顺序排查：
1. 确认启动命令里两项都在（重启 vLLM 才生效）；
2. 把 `qwen3_coder` 换成 `hermes`（Qwen 系通用解析器）重启再试；
3. 都失败也不致命——工具会自动用 JSON 文本兜底，只是多耗 token、稳定性略降。

---

## 4. 冒烟验证（部署后必做，约 3 分钟）

```bash
cd /opt/imageMask && source .venv/bin/activate
export IMAGE_MASK_ENV_FILE=/opt/imageMask/.env.production

# ① 单元测试：不需要网络和 VLM，秒级，验证代码/依赖完整
#    装了 rapid-orientation 应为 167 passed；
#    没装则 2 个方向矫正测试 skip（165 passed, 2 skipped）——出现 skip 就先补装
python -m pytest tests/ -q

# ② 生成示例票据 + 单图全链路（会真的调 vLLM）
python main.py --make-sample
python main.py data/sample_bill.png
# 期望日志：状态: passed；退出码 echo $? → 0
# 单图输出在 outputs/redacted_<时间戳>.png

# ③ 真实票据端到端（可选，更贴近生产）
python tests/e2e_remote.py
```

任一步失败**不要开始跑批**，先按第 8 节排查。

**VLM 调用量预算**（API 有配额时先算这个）：每张图固定 **2 次**调用（敏感判断 1 + 语义质检 1）；质检不通过时每次重试再 +2。上面这套冒烟约 6~10 次调用。排障时可给单图加 `--no-qa` 降到 1 次/张（但会连确定性质检一起关，仅限排障）。

---

## 5. 跑批

### 5.1 前台先小规模试跑

```bash
python main.py /data/testCase --workers 3
```

- 启动会打印 `共发现 N 张图片（M 个 case，并行 3）`，随后显示 tqdm 进度条：

  ```
  脱敏进度: 42%|████▏     | 8/19 [02:31<03:28, 18.9s/img]
  ```

  跳过的图也计入进度，所以进度条一定走得满。进度条只在交互式终端显示，`nohup ... > run.log` 时自动关闭（日志里看每张图的 `处理 ...` 行即可）。
- `--workers` 是 **case 级并行**（case 内图片串行，OCR 有锁）。**不要按 CPU 核数给**，建议 `min(case 数, 4)`；vLLM 是 TP=4 单实例，并发过高只会让每张图变慢、更易超时。
- 输出默认 `/data/testCase_masked`；要换位置用 `--output /data/out`。

### 5.2 正式跑批（断 SSH 也不中断）

```bash
cd /opt/imageMask && source .venv/bin/activate
export IMAGE_MASK_ENV_FILE=/opt/imageMask/.env.production

mkdir -p logs
nohup python -u main.py /data/testCase --workers 3 \
      > logs/run_$(date +%Y%m%d_%H%M).log 2>&1 &
echo $! > batch.pid

tail -f logs/run_*.log
```

- `-u` 关缓冲：不加的话日志会在缓冲区里憋很久才落地，看起来像卡死。
- 结束判断：日志出现 `================ 批量处理汇总 ================`，或 `ps -p $(cat batch.pid)` 无输出。

### 5.3 常用参数

| 参数 | 用途 |
|---|---|
| `--workers N` | case 级并行数，建议 3~4 |
| `--no-qa` | 关质检（只跑 OCR+VLM 判断+脱敏），排障/赶时间用 |
| `--method blur` / `mosaic` | 换脱敏方式（默认 white_mask 假数据重绘） |
| `--max-retries 3` | 质检不通过的最大重试次数（默认 3，耗尽走 fallback 并标 `max_retry`） |
| `--skip-existing` | 输出文件已存在就跳过（不看断点进度） |
| `--no-resume` | **忽略断点，全量重跑**（慎用） |
| `--output DIR` | 指定输出根目录（会改变断点归属） |
| `--max-image-mb 5` | 超过该大小（MB）的图先等比缩小再处理；`0`/负数 = 关闭。默认 5 |
| `--no-shrink` | 关闭大图缩小，按原始分辨率处理 |

### 5.4 退出码

| 退出码 | 含义 |
|---|---|
| `0` | 全部图片 passed（或全部跳过） |
| `1` | 存在 failed / max_retry，需查看报告 |

---

## 6. 中断后续跑（不需要从头开始）

**默认就开启断点续传**，中断后把**同一条命令再敲一遍**即可：

```bash
python -u main.py /data/testCase --workers 3 >> logs/run_resume.log 2>&1
```

工作机制：

- 每张图处理完立刻原子写 `/data/testCase_masked/batch_progress.json`（临时文件 + `os.replace`，断电/强杀不会写坏）。
- 重启时读进度：`passed` 的图直接跳过（日志：`断点跳过（已完成）：...`）；`failed` / `max_retry` / 无记录的图重新处理。
- 启动日志会有 `检测到断点进度：N 张图片已完成，将跳过`，`batch_report.json` 里 `resumed: true`、`resumed_skipped: N`。

**续跑失效的三种原因（务必避免）：**

1. 命令里加了 `--no-resume` → 忽略进度，全量重跑；
2. 输入根路径变了（比如从 `/data/testCase` 换成 `/data/testcase` 或软链接路径）→ 进度里存的是绝对路径字符串，对不上就整批忽略；
3. 换了 `--output` 目录 → 新目录没有进度文件，等于从零开始。

**中断后的行为保证**（都有测试固化：`pytest tests/test_batch.py -k "kill_mid or ctrl_c"`）：

| 场景 | 行为 |
|---|---|
| `kill -9` / 断电 | 进度文件是「临时文件 + `os.replace`」原子写，最多丢正在处理的那 1 张；报告没写也不要紧，续跑会重写 |
| Ctrl+C（并行模式） | 只取消**排队中**的 case，在跑的那几个收尾后立刻退出（不会把剩下几百个 case 跑完）；进度不丢 |
| 进度说 passed 但输出文件被删 | 续跑时重新处理该图，不会静默跳过 |
| `--output` 指到输入目录里面 | 直接报错退出（防止把脱敏结果写回原图位置） |
| 跑到一半磁盘写满 | 该 case 记 `error`，批次继续跑完并照常出报告；报告写不成功时只打错误日志，不丢统计 |

> 续跑完成后 `summary` 里可能出现 `skipped`（整个 case 的图都是断点跳过的），这是正常状态，不是失败。

> 想彻底重来：用 `--no-resume`，**不要**手动删 `batch_progress.json`（删了也行，但 `--no-resume` 语义更清晰、不会误伤别的状态）。

---

## 7. 查看结果与统计

### 7.1 终端汇总（跑完自动打印）

```
================ 批量处理汇总 ================
输入：/data/testCase
输出：/data/testCase_masked
耗时：156.9s，case 状态分布：{'max_retry': 1, 'passed': 2}
---------------- 统计指标 ----------------
处理 case 数：3
累计处理图片：16 张
平均每个 case 处理时长：123.49s
平均每个 case 图像数：5.33 张
平均每张图处理时长：23.15s
报告：/data/testCase_masked/batch_report.json
需关注：case case1 状态 max_retry
```

### 7.2 从 JSON 取指标

```bash
cd /data/testCase_masked

jq '.summary' batch_report.json
jq '.statistics' batch_report.json
jq '.duration_sec, .resumed, .resumed_skipped' batch_report.json
```

| 字段 | 含义 |
|---|---|
| `duration_sec`（顶层） | 整批 wall-clock 耗时 |
| `summary` | case 级状态分布 |
| `statistics.cases_processed` | 实际处理的 case 数 |
| `statistics.images_processed` | 实际处理的图片数（不含断点跳过） |
| `statistics.shrunk_images` | 其中因超过阈值被等比缩小的张数 |
| `statistics.avg_case_duration_sec` | case 耗时之和 ÷ case 数 |
| `statistics.avg_images_per_case` | 图片数 ÷ case 数 |
| `statistics.avg_image_duration_sec` | 图片耗时之和 ÷ 图片数 |

> 两个易误读点：① 顶层 `duration_sec` ≠ Σ `case.duration_sec`（case 之间并行）；② 续跑那一轮 `duration_sec` 明显偏小，不能拿来和全量轮比较性能。

### 7.3 定位问题图片

```bash
# 所有非 passed 的图片
jq -r '.cases[] | .case as $c | .images[] | select(.status != "passed")
       | [$c, .status, .image, (.qa_reason // .error // "")] | @tsv' batch_report.json

# 按耗时降序（找慢图）
jq -r '.cases[] | .images[] | [.duration_sec, .status, .image] | @tsv' batch_report.json \
  | sort -rn | head -20
```

状态含义：

| status | 含义 | 处理 |
|---|---|---|
| `passed` | 质检通过 | 无需处理 |
| `max_retry` | 重试 `--max-retries` 次仍不通过，已 fallback 保留当前脱敏结果 | 看 `qa_reason`（残留块 / 误伤块 ID） |
| `failed` | 节点异常（OCR/VLM 报错、文件损坏等） | 看 `error` |
| `skipped` | 断点或 `--skip-existing` 跳过 | 正常 |

重跑某一个 case 时，把整批命令再执行一次即可（passed 的会跳过，只重跑失败的）。

---

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `model not found` / `The model ... does not exist` | `VLM_MODEL` 与 vLLM 服务名不一致 | `curl /v1/models` 核对；未设 `--served-model-name` 时必须写模型路径 |
| `At most 1 image(s) may be provided` / 图片全 failed | vLLM 缺 `--limit-mm-per-prompt image=4` | 补参数后重启 vLLM |
| 日志打到 modelscope 地址、报 401 | `IMAGE_MASK_ENV_FILE` 没生效，回落到 `.env` | 用第 2.2 节自检命令确认；export 后再启动 |
| 大量 `结构化输出调用失败` 后靠 JSON 兜底 | tool-call parser 不匹配 | 见 3.3；换 `hermes` 或核对 vLLM 版本 |
| `带 extra_body 的调用失败…降级去掉 extra_body` | 旧版 vLLM 不认 `chat_template_kwargs` | 自动降级，无需处理；也可设 `VLM_ENABLE_THINKING=true` 绕开 |
| 调用超时 / 结构化输出被截断 | 27B 双图推理慢 | `VLM_TIMEOUT=900`；确认 `VLM_MAX_TOKENS=4096`、vLLM `max-model-len` 足够 |
| 假数据位置是纯色块、日志有 `未找到中文字体` | 缺中文字体 | `sudo apt install fonts-noto-cjk` 后重跑 |
| `import cv2` 报 `libGL.so.1` | 缺 opencv 系统库 | `sudo apt install libgl1 libglib2.0-0`，或换 `opencv-python-headless` |
| 竖版图方向乱 / 日志无方向矫正记录 | `OCR_ORIENTATION_CORRECT=true` 但 rapid-orientation 未装 | `pip install rapid-orientation`（未装会静默降级为原图直跑） |
| `下发现 0 个 case` | 图片直接放在输入根目录，根目录下没有子目录 | 按 1.3 节整理成 `root/caseX/*.jpg` |
| 断点没生效、又跑了一遍全部 | 路径/输出目录变了，或加了 `--no-resume` | 见第 6 节 |
| 日志 `已到长边下限 800px，停止缩小` | 图体积大但已缩到保护下限 | 正常保护（再缩会掉 OCR 识别率）；想更激进就调大 `--max-image-mb`，想死守体积才改动代码里的 `MIN_LONG_SIDE` |
| 想确认大图是否被处理 | — | 日志有 `大图缩小：12.4MB → 4.1MB，4000x3000 → 2048x1536`；或看报告的 `statistics.shrunk_images` |
| 某个 case 状态 `error` | case 级异常（如权限、磁盘满） | 看日志 `case xxx 处理异常` 的堆栈 |
| 跑批中途机器重启 | — | 直接重跑原命令，passed 自动跳过 |

日志里快速定位异常：

```bash
grep -nE "ERROR|Traceback|异常|失败" logs/run_*.log | head -50
grep -c "断点跳过" logs/run_*.log        # 续跑生效的证据
```

---

## 9. 日常操作清单

**每次跑批前**
1. `source .venv/bin/activate`，`export IMAGE_MASK_ENV_FILE=/opt/imageMask/.env.production`
2. `curl -s http://172.16.84.211:8000/v1/models` 确认 vLLM 在线
3. 确认输入目录里是 `case/图片` 结构
4. `df -h` 看磁盘（输出图与输入同量级）

**跑批中**
5. `tail -f logs/run_*.log`，看是否出现连续 failed / 超时
6. 中断了 → 原命令重跑，先确认日志出现 `检测到断点进度`

**跑批后**
7. 读 `batch_report.json` 的 `summary` / `statistics`
8. `jq` 列出非 passed 的图，看 `qa_reason` 判断是漏脱敏还是误伤
9. 抽查 2~3 张输出图（`data/testCase_masked/...`）肉眼确认

---

## 10. 附：单图模式

```bash
# 单张图片：输出到 outputs/redacted_<时间戳>.png
python main.py /data/one/ticket.jpg

# 关质检只看脱敏效果（快）
python main.py /data/one/ticket.jpg --no-qa

# 换脱敏方式
python main.py /data/one/ticket.jpg --method blur
```

单图退出码：`0` = passed，`1` = 未通过。

---

## 11. 附：性能排查（单张 ~12s 是正常的，想更快看这里）

单张耗时构成（本地实测 + 27B 估值）：OCR 首轮 ~1.4s + VLM 判断 2~4s + 脱敏 <0.01s + 质检重 OCR ~1.4s + VLM 质检 3~6s ≈ **VLM 占 70%，OCR 占 25%**。

**已实现的并发**：确定性预检（重 OCR + 正则 + 几何误伤）默认放后台线程，与 VLM 语义质检同时跑，`verify` 从 `重OCR + VLM` 变成 `max(重OCR, VLM)`，单张省 ~1.2~1.5s（`QA_PRECHECK_BACKGROUND=false` 可关）。
**实测无效的弯路**（别再试）：并发 OCR——20 核 CPU 上 4 张并发，1 实例 5.7s vs 4 实例 12~14.5s，onnxruntime 单会话已吃满全部核心，多会话互相争抢；多进程同理，只增加断点/进度协调复杂度；asyncio 也没有可并发部分（单张图内是严格数据依赖，跨图 I/O 重叠已由 case 级线程池承担）。

按顺序排查：

```bash
# ① 思考模式是否真的关掉（最重要）
grep "思考模式" logs/run_*.log
#   期望：VLM 思考模式：已关闭（extra_body={'chat_template_kwargs': {'enable_thinking': False}}）
#   若出现「不被服务端接受…改试 …」→ 该 vLLM 不认这种写法，工具已自动换写法，看下一行确认
#   若出现「所有参数写法均被服务端拒绝」→ 思考模式没关掉：结构化输出不稳、单张慢数秒，
#     升级 vLLM（或临时设 VLM_ENABLE_THINKING=true 并接受代价，不推荐）

# ② 两次 VLM 调用各花多久、prompt/generation token 多少（看 vLLM 日志或 /metrics）
curl -s http://172.16.84.211:8000/metrics | grep -E "ttft|generation_tokens|prompt_tokens"

# ③ 是不是并发互相排队：同样数据分别用 --workers 1 和 3 跑 10 张，比较 avg_image_duration_sec
```

可调项（改 `.env.production` 后重启进程生效）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `QA_PRECHECK_BACKGROUND` | true | 质检预检与 VLM 并发跑，单张省 ~1.2~1.5s；false = 串行且给 VLM 带预检 hints |
| `VLM_QA_MASK_IMAGE_SIDE` | 512 | 质检「脱敏图」长边，vision token 约为 1024 的 1/4；0 = 跟随 `VLM_MAX_IMAGE_SIDE` |
| `VLM_MAX_IMAGE_SIDE` | 1024 | 降到 768 可再省 ~44% 图像 token，先用几张真实票据对比 QA 通过率 |
| `QA_REOCR_MAX_SIDE` | 1280 | 质检重 OCR 先缩小，实测省 ~0.2s/张；0 = 不缩放 |
| `OCR_ENGINE_INSTANCES` | 1 | OCR 引擎实例数。CPU 上并发只会更慢，保持 1；只有限制每会话线程数或换 GPU 推理时才调大 |
| `--workers` | 3 | OCR 阶段 CPU 已饱和（串行），超过 4 只会让 vLLM 排队、单张更慢 |

注意：`--no-qa` 会连确定性质检一起关（残留/正则/误伤全部不查），只用于排障，**不要用于正式跑批**。

---

## 12. 附：开机自启（可选）

`/etc/systemd/system/imagemask-batch.service`：

```ini
[Unit]
Description=imageMask batch redaction
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/opt/imageMask
Environment=IMAGE_MASK_ENV_FILE=/opt/imageMask/.env.production
Environment=PYTHONIOENCODING=utf-8
# logs 目录不存在时 systemd 打不开 append 目标，服务会直接起不来
ExecStartPre=/bin/mkdir -p /opt/imageMask/logs
ExecStart=/opt/imageMask/.venv/bin/python -u main.py /data/testCase --workers 3
StandardOutput=append:/opt/imageMask/logs/systemd.log
StandardError=append:/opt/imageMask/logs/systemd.log
Restart=no

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl start imagemask-batch
sudo systemctl status imagemask-batch
journalctl -u imagemask-batch -f
```

`Restart=no` 是刻意的：本工具靠断点续传而非自动重启，避免失败时反复打 VLM。
