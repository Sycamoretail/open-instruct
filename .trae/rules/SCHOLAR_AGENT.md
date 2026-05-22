# Scholar Search Agent —— 项目整理手册

本文件汇总了 **把 Qwen3-8B 训练成 scholar / search agent** 这条主线上所有涉及的
代码、脚本、数据格式与已知修复点，便于其他 Agent 或协作者快速接管。

整体链路（按时间顺序）：

```
原始 scholar parquet          Teacher(GPT-5.x)                RL rollout
(data/scholar_en/*.parquet)   经 tools 调用                    经 tools 调用
        │                           │                              │
        ▼                           ▼                              ▼
prepare_scholar_data.py    distill_sft_trajectories.py     grpo_fast.py
→ RLVR parquet             → SFT trajectories (v1)        → 策略更新
(data/scholar_en_rlvr/)    (data/scholar/*.jsonl)
        │                           │
        │                  distill_sft_trajectories_v2.py
        │                  → SFT trajectories (v2, prompt-based)
        │                  (data/scholar/sft_distill_v4.jsonl)
        │                           │
        │                  filter_sft_format.py
        │                  → 格式验证 & 过滤
        │                           │                              │
        └──── SFT 训练（open_instruct/finetune.py）──────┘         │
                         │                                          │
                         └─── 初始化 RL 训练 ───────────────────────┘
                                         │
                                         ▼
                             DRB / DR-Tulu / DeepScholar Bench 评测
                             (scripts/eval/dr_tulu)
```

---

## 0. 环境准备

```bash
export PATH="/home/tiger/.local/bin:$PATH"
cd /mlx_devbox/users/luoyunze/playground/open-instruct
uv sync                        # 主环境
uv sync --extra dr-tulu        # 评测依赖（dr_tulu_agent_eval.sh 自动执行）
```

关键外部依赖：

- **内部 tools 包**：`tools/`（`ScholarSearchTool` / `GeneralSearchTool` /
  `Fetch` / `_reset_internal_state` / `_INTERNAL_STATE`）
- **Azure OpenAI** (ByteDance gpt_openapi)：SFT distill 的 teacher
  - endpoint: `https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi`
  - API version: `2024-03-01-preview`
  - 支持多个 AK（逗号分隔，进程内按 AK cache client）
- **内部 HTTP judge**：`https://ivavmlgq.fn.bytedance.net`
  （DRB / scholar_evaluator 评测用，走 `sys-proxy-rd-relay.byted.org:8118` 代理）

Lint / 格式化：`make style && make quality`；测试：`uv run pytest`。

---

## 1. 数据准备

### 1.1 原始数据

`data/scholar_en/` 下 4 份任务 parquet，原始 schema `{data_source, prompt,
extra_info}`，其中 `extra_info.reward_model.ground_truth` 是 JSON 字符串，
内含 `verifiable_meta = {paper_title, synthesis_answer, published_date, query, ...}`。

| 文件 | 任务类型 | 行数 | paper_title | published_date |
|------|----------|------|-------------|----------------|
| `search_data.2023.parquet` | search | 3326 | 3326 | 3326 |
| `understanding_data.2023.parquet` | understanding | 2049 | 2049 | 0 |
| `understanding_data.2023_new.parquet` | understanding | 2364 | 2364 | 0 |
| `write_section_data.2023.parquet` | write_section | 1666 | 1666 | 1666 |
| `write_survey_data.2023.parquet` | write_survey | 1628 | 1628 | 1628 |

### 1.2 转成 RLVR 格式

[prepare_scholar_data.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/prepare_scholar_data.py) 会：

1. 丢掉原始 system prompt，只保留 user query。
2. 注入一个新的 system prompt（定义 `<think>/<call_tool>/<answer>/<cite>` 协议 +
   工具声明），由 `build_system_prompt(tool_backend)` 产生，`tool_backend` =
   `"internal"`（线上 tools）或 `"mock"`（离线假 search）。
3. 把 `verifiable_meta.paper_title → curr_title` 和
   `verifiable_meta.published_date → last_time` 写到每行的 `env_config` 列，
   面向 4 个 scholar env：`scholar_search / general_search / fetch /
   mock_scholar_search`（这是让 RL 运行时拿到 per-sample 过滤字段的关键，见 §6.2）。

输出列：`{messages, ground_truth, dataset, env_config}`。

```bash
uv run python scripts/data/scholar/prepare_scholar_data.py \
    --input_dir data/scholar_en \
    --output_dir data/scholar_en_rlvr \
    --tool_backend internal
```

- `--print_prompt` 可打印出 system prompt 不写文件。
- 若要换 backend，改 `--tool_backend mock` 并重新生成一份。

**注意**：RL 启动脚本默认通过 `.prep.<backend>.done` flag 幂等跳过重跑；如需
强制重生成，删掉 `data/scholar_en_rlvr/.prep.internal.done`。

---

## 2. SFT 数据蒸馏（GPT-5.x Teacher）

### 2.1 设计（v1，function-calling）

[distill_sft_trajectories.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/distill_sft_trajectories.py) —— 让 Azure GPT-5.x teacher 在 function-calling 协议下调用
`ScholarSearchTool / GeneralSearchTool / Fetch`，跑多轮 tool-use 直到给出
final answer；然后把整条轨迹**翻译**成 `<think>/<call_tool>/<tool_output>/<answer>/<cite>`
XML 协议，格式与 RL 阶段一致。

每条输出：

```jsonl
{"messages": [...], "dataset": "scholar_search", "ground_truth": "{...}",
 "_meta": {
    "example_id": "...", "teacher_model": "gpt-5-2025-08-07",
    "data_type": "search",
    "reward": 0.67, "reward_breakdown": {...}, "reward_threshold": 0.3,
    ...
 }}
```

非 assistant 轮 tokenize 后自动 mask 为 -100，可直接喂 `finetune.py` 的
`sft_tulu_tokenize_and_truncate_v1`。

### 2.2 关键参数

| 参数 | 说明 |
|------|------|
| `--input` | 1 个或多个 `.jsonl` / `.parquet`（RLVR 格式） |
| `--output-jsonl` | 追加写；进程重启自动从 `_meta.example_id` 续跑 |
| `--num-examples` | 仅取前 N 条做快速试 |
| `--ak-list` | 逗号分隔多个 Azure AK（会随机挑一个打） |
| `--model-list` | 与 AK 对齐的模型名，例 `gpt-5-2025-08-07` |
| `--max-completion-tokens` | 默认 8000（GPT-5 会把 reasoning tokens 也计入！） |
| `--reasoning-effort` | `low/medium/high`，默认 high |
| `--max-steps` | agent loop 最多工具轮次，默认 8 |
| `--tool-backend` | `internal`（调真 tools）或 `mock` |
| `--last-time` | 全局 fallback；**per-sample `last_time` 会从输入 parquet 自动取出**（§6.1） |
| `--min-tool-calls` | 轨迹必须 ≥N 次 tool 调用才保留，默认 2 |
| `--require-answer` / `--no-require-answer` | 是否必须有 `<answer>` |
| `--min-reward` | 全局 reward 阈值，调 `scholar_evaluator` 打分后过滤 |
| `--min-reward-per-type` | 按 data_type 分别设阈值，例 `search=0.3,understanding=0.5,write_section=0.4,write_survey=0.4`，优先级高于 `--min-reward` |
| `--always-score` | 强制打分但不做阈值过滤，只记录到 `_meta.reward` |
| `--score-timeout-s` | 单条评分超时，默认 180s |
| `--max-concurrency` | 线程池并发，默认 4，建议生产跑调到 16+ |
| `--overwrite` | 清空已有 output 再写 |

### 2.3 典型命令

**Pilot 测试（20 条，不过滤 reward）**
```bash
uv run python scripts/data/scholar/distill_sft_trajectories.py \
    --input data/scholar_en_rlvr/search_data.2023.parquet \
    --output-jsonl data/scholar/sft_distill_pilot.jsonl \
    --ak-list "$AK1,$AK2" --model-list "gpt-5-2025-08-07" \
    --num-examples 20 --max-concurrency 4
```

**全量蒸馏（打分但不过滤，离线再筛 —— 推荐）**
```bash
uv run python scripts/data/scholar/distill_sft_trajectories.py \
    --input data/scholar_en_rlvr/search_data.2023.parquet \
            data/scholar_en_rlvr/understanding_data.2023.parquet \
            data/scholar_en_rlvr/understanding_data.2023_new.parquet \
            data/scholar_en_rlvr/write_section_data.2023.parquet \
            data/scholar_en_rlvr/write_survey_data.2023.parquet \
    --output-jsonl data/scholar/sft_distill_score.jsonl \
    --ak-list "$AK1,$AK2,$AK3,$AK4" \
    --model-list "gpt-5-2025-08-07,gpt-5-2025-08-07,gpt-5-2025-08-07,gpt-5-2025-08-07" \
    --max-concurrency 16 --always-score
```

**在线过滤（per-type 阈值）**
```bash
    ... \
    --min-reward-per-type "search=0.3,understanding=0.5,write_section=0.4,write_survey=0.4"
```

### 2.4 续跑 & 并发

- 输出文件以 append 模式写；重启脚本会 `load_done_ids()` 扫描已有 `_meta.example_id`，只跑剩余样本。
- 并发仅限线程池并发（受 Azure 单 AK QPS 限制）；提升吞吐两条路：
  1. `--max-concurrency` 上调；
  2. 增加更多 AK，`--ak-list "$AK1,$AK2,$AK3,..."`。
- 被丢弃的样本**不会被记进 done 列表**（设计如此：允许修完过滤器重跑）。

### 2.5 一键脚本

[scripts/data/scholar/distill.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/distill.sh)
封装了上面 3 个 profile，自动按 AK 数量广播 `--model-list`、自动校验输入文件存在：

```bash
# Profile=score（默认）：5 份 RLVR parquet，--always-score，不过滤
AK_LIST="$AK1,$AK2,$AK3,$AK4" ./scripts/data/scholar/distill.sh

# Profile=pilot：单文件 20 条
PROFILE=pilot AK_LIST="$AK1" ./scripts/data/scholar/distill.sh

# Profile=filter：5 份 parquet + per-data_type 阈值
PROFILE=filter AK_LIST="$AK1,$AK2" ./scripts/data/scholar/distill.sh
```

可调 env：`PROFILE / MODEL_LIST / RLVR_DIR / OUTPUT_JSONL / NUM_EXAMPLES /
MAX_CONCURRENCY / MAX_STEPS / MAX_COMPLETION_TOKENS / REASONING_EFFORT /
TOOL_BACKEND / MIN_REWARD_PER_TYPE / OVERWRITE`。

### 2.6 v2 蒸馏脚本（prompt-based，推荐）

[distill_sft_trajectories_v2.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/distill_sft_trajectories_v2.py) —— 基于 prompt 的蒸馏方案，解决 v1 function-calling 方案丢失 teacher reasoning 的问题。

**v1 vs v2 对比**：

| 维度 | v1 (function-calling) | v2 (prompt-based) |
|------|----------------------|-------------------|
| 工具调用方式 | `tools=[...]` 参数 + function-calling 协议 | 工具描述写在 system prompt 里，无 `tools` 参数 |
| teacher reasoning | 隐藏在 hidden reasoning tokens 中，`content` 常为空 | `<think>/<call_tool>/<answer>` 直接在 `content` 中输出 |
| 每轮是否有显式推理 | 仅 7% 样本有 initial think | 每轮天然有显式 reasoning |
| `max_completion_tokens` | 默认 8000 | 默认 16000（reasoning 现在在 content 里） |
| `reasoning_effort` | 默认 high | 默认 medium |
| 格式校验 | 无 | 内置 5-check 格式验证（与 RL 一致） |

**v2 工作原理**：

1. 不传 `tools=[...]`，改为在 system prompt 中完整描述工具协议和 `<think>/<call_tool>/<answer>` 格式
2. Teacher 直接在 `content` 字段中输出 XML 标签
3. 每次工具调用前后都有 `<think>` 推理过程
4. 典型输出 pattern：`T+C -> T+C -> T+C -> ... -> T+A`（Think+CallTool 交替，最后 Think+Answer）

**输出数据质量**（`sft_distill_v4.jsonl`）：

- 136 samples
- 100% format pass（通过全部 5 项格式检查）
- 100% 有 initial think plan（开头即有 `<think>` 规划）
- 100% 有 think after each tool response（每个 tool 返回后都有 `<think>` 分析）
- 平均每 sample 7.7 个 think 块

**与 v1 数据对比**：

| 指标 | v1 | v2 |
|------|----|----|
| initial think 占比 | 7% | 100% |
| think after tool 占比 | 4% | 100% |
| 平均 thinks/sample | 1.3 | 7.7 |

**典型命令**：
```bash
uv run python scripts/data/scholar/distill_sft_trajectories_v2.py \
    --input data/scholar_en_rlvr/search_data.2023.parquet \
    --output-jsonl data/scholar/sft_distill_v4.jsonl \
    --ak-list "$AK1,$AK2" --model-list "gpt-5-2025-08-07" \
    --max-concurrency 8
```

### 2.7 格式过滤脚本

[filter_sft_format.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/filter_sft_format.py) —— 对 SFT 数据进行格式合规性验证，使用与 RL 训练阶段 `_protocol_format_reward` 相同的 5 项检查（见 §4.4）：

1. `has_call_tool` —— 至少有一次 `<call_tool>` 调用
2. `has_single_answer` —— 恰好有一个 `<answer>` 块
3. `has_cite_inside_answer` —— `<answer>` 内含 `<cite>`
4. `tags_balanced` —— 所有 XML 标签配对闭合
5. `answer_not_nested` —— `<answer>` 不在 `<think>` 内部

过滤不通过的样本会被丢弃或标记，确保 SFT 数据与 RL reward 格式完全一致。

```bash
uv run python scripts/data/scholar/filter_sft_format.py \
    --input data/scholar/sft_distill_v4.jsonl \
    --output data/scholar/sft_distill_v4_filtered.jsonl
```

---

## 3. SFT 训练

入口：[open_instruct/finetune.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/finetune.py)（配合 `sft_tulu_tokenize_and_truncate_v1`，
非 assistant 轮 token 自动 mask 为 -100）。

蒸馏产物 `data/scholar/sft_distill_*.jsonl` 的字段 `{messages, dataset,
ground_truth, _meta}` 与 Tulu SFT 直接兼容；`_meta` 列会自动被丢弃，
`reward` 在 SFT 阶段**不参与梯度**，只是数据筛选副产品。

**推荐数据**：`data/scholar/sft_distill_v4.jsonl`（v2 prompt-based 蒸馏，136 条，100% 格式通过，每轮都有显式 reasoning）。

**推荐 MAX_SEQ_LENGTH**：20480–24576（v2 数据因为 reasoning 在 content 中，
序列长度比 v1 更长；20480 能覆盖 95%+ 样本，24576 可完整保留所有样本）。

固化的 8 GPU 启动脚本：
[scripts/train/scholar/sft_qwen3_8b.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/train/scholar/sft_qwen3_8b.sh)

```bash
# 默认：Qwen3-8B + sft_distill_v4.jsonl，8 GPU + Deepspeed Stage3
./scripts/train/scholar/sft_qwen3_8b.sh

# 用别的 base / 别的数据 / 别的输出目录
BASE_MODEL_NAME_OR_PATH=Qwen/Qwen3-0.6B \
SFT_DATA_PATH=data/scholar/sft_distill_v4.jsonl \
MAX_SEQ_LENGTH=20480 \
NUM_PROCESSES=1 \
OUTPUT_DIR=outputs/qwen3_06b_scholar_smoke \
    ./scripts/train/scholar/sft_qwen3_8b.sh
```

可调 env 列表（脚本头部注释里有完整说明）：`BASE_MODEL_NAME_OR_PATH /
SFT_DATA_PATH / OUTPUT_DIR / NUM_PROCESSES / MAX_SEQ_LENGTH /
LEARNING_RATE / NUM_TRAIN_EPOCHS / GRADIENT_ACCUMULATION_STEPS /
WANDB_MODE`。`MAX_SEQ_LENGTH` 推荐 20480–24576（v2 数据），RL 端
`--pack_length` 建议同步调大。

> 多机：直接参考 `scripts/train/debug/oc_sft_multinode.sh` 的 launcher
> 头改 `--num_machines / --machine_rank` 即可。

---

## 4. RL（GRPO）训练

### 4.1 入口

[open_instruct/grpo_fast.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/grpo_fast.py)

### 4.2 启动脚本

| 场景 | 脚本 |
|------|------|
| 本地 8 GPU（推荐日常开发） | [qwen3_8b_scholar_search_agent_local.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/train/debug/tools/qwen3_8b_scholar_search_agent_local.sh) |
| Beaker / 容器 | [qwen3_8b_scholar_search_agent.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/train/debug/tools/qwen3_8b_scholar_search_agent.sh) |

两者都支持两阶段训练：

- **Phase1**：`SCHOLAR_REWARD_MODE=mock SCHOLAR_TOOL_BACKEND=mock`，只奖励
  协议合规性（`<think>/<call_tool>/<answer>/<cite>`），用于把新模型引导进正确
  的输出格式。实现见 [scholar_verifier.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/scholar_verifier.py) 的 mock 分支（`SCHOLAR_MOCK_REWARD=1`）。
- **Phase2**：`SCHOLAR_REWARD_MODE=real SCHOLAR_TOOL_BACKEND=internal`，接
  `scholar_evaluator.evaluate_scholar_score_verifier`，奖励 nugget coverage /
  reference coverage / citation precision / relevance rate。可通过
  `INIT_FROM_MOCK=1` 从 Phase1 的 latest checkpoint 起跑。

### 4.3 本地典型命令

```bash
export MODEL_NAME_OR_PATH=/path/to/Qwen3-8B
SCHOLAR_REWARD_MODE=mock SCHOLAR_TOOL_BACKEND=mock \
    ./scripts/train/debug/tools/qwen3_8b_scholar_search_agent_local.sh    # Phase1

INIT_FROM_MOCK=1 \
SCHOLAR_REWARD_MODE=real SCHOLAR_TOOL_BACKEND=internal \
    ./scripts/train/debug/tools/qwen3_8b_scholar_search_agent_local.sh    # Phase2
```

脚本内部默认参数节选：

- `--tools`：`internal` 下 = `scholar_search general_search fetch`；`mock`
  下 = `mock_scholar_search`
- `--tool_call_names`：`ScholarSearch GeneralSearch Fetch`（这是 policy 输出
  `<call_tool name="...">` 中的名字，必须和 `scholar_tools.py` 的 `call_name`
  一致）
- `--tool_parser_type dr_tulu`（见 `open_instruct/environments/tools/parsers.py`）
- 分布式：8 GPU，`--deepspeed_stage 3`，`--vllm_num_engines 2 --vllm_tensor_parallel_size 1`（本地脚本）或 8/1（beaker）
- Checkpoint：每 `CHECKPOINT_STATE_FREQ=20` 步保存训练 state，每
  `SAVE_FREQ=50` 步保存完整模型，保留最近 2 份

### 4.4 Reward 实现

- [scholar_verifier.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/scholar_verifier.py)
  —— `VerifierFunction`，挂在 4 个 `data_source` 名下：
  `scholar_search / scholar_understanding / scholar_write_section /
  scholar_write_survey`；决定走 mock 还是调
  `scholar_evaluator.evaluate_scholar_score_verifier`。
- [scholar_evaluator.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/scholar_evaluator.py)
  —— 真实评分函数，按 `data_type` 分 4 路计算 sub-score 并加权；同时被 distill
  脚本复用来过滤 SFT 数据。

**`_protocol_format_reward` 的 5 项格式检查**：

| # | 检查项 | 说明 |
|---|--------|------|
| 1 | `has_call_tool` | 输出中至少出现一次 `<call_tool>` 调用 |
| 2 | `has_single_answer` | 恰好有且仅有一个 `<answer>...</answer>` 块 |
| 3 | `has_cite_inside_answer` | `<answer>` 块内部包含 `<cite>` 引用 |
| 4 | `tags_balanced` | 所有 XML 标签（`<think>`/`<call_tool>`/`<answer>`）正确配对闭合 |
| 5 | `answer_not_nested` | `<answer>` 不出现在 `<think>` 内部（防止嵌套） |

格式得分 = `passed_checks / 5`。例如通过 4 项得 0.8，全部通过得 1.0。

> `answer_not_nested` 是新增检查项，用于解决 v1 蒸馏数据中 `<answer>` 被错误嵌套在 `<think>` 内的问题（见 §8 坑位）。

---

## 5. 服务部署与推理采样

评测前先把训练好的 checkpoint 起成 OpenAI 兼容服务，再用统一的多轮 tool-use
循环把题目跑出 trajectories；trajectories 既能喂 DRB（RACE+FACT），也能喂
DR-Tulu 评测矩阵。

### 5.1 起 vLLM 服务

[scripts/serve/scholar/serve_vllm.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/serve/scholar/serve_vllm.sh)
（默认 `127.0.0.1:30001`，model 名用 `--served-model-name` 自定义）

```bash
# 起 SFT checkpoint，单卡
MODEL_NAME_OR_PATH=outputs/qwen3_8b_scholar_sft \
SERVED_MODEL_NAME=scholar-8b-sft \
GPU_VISIBLE=0 \
    ./scripts/serve/scholar/serve_vllm.sh

# 起 RL checkpoint，2 卡 TP
MODEL_NAME_OR_PATH=outputs/qwen3_8b_scholar_search_agent_en_real/step_300_checkpoint \
SERVED_MODEL_NAME=scholar-8b-rl-300 \
TENSOR_PARALLEL_SIZE=2 GPU_VISIBLE=0,1 PORT=30002 \
    ./scripts/serve/scholar/serve_vllm.sh
```

可调 env：`MODEL_NAME_OR_PATH / SERVED_MODEL_NAME / HOST / PORT /
TENSOR_PARALLEL_SIZE / GPU_MEMORY_UTIL / MAX_MODEL_LEN / DTYPE / API_KEY /
GPU_VISIBLE / EXTRA_VLLM_ARGS`。

### 5.2 采样 trajectories（get_response）

[scripts/data/scholar/get_response.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/get_response.sh)
封装了 [generate_responses.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/generate_responses.py)，默认按训练时一致的
`<think>/<call_tool>/<answer>/<cite>` XML loop 跑（`--agent-mode internal`），
工具走 `tools.build_default_registry("internal")`，跨样本由
`reset_internal_state()` 复位 reference id：

```bash
# DRB（默认 dataset）
MODEL=scholar-8b-sft API_BASE=http://127.0.0.1:30001/v1 \
    ./scripts/data/scholar/get_response.sh deep_research_bench

# SimpleQA，agent_mode=none（单轮）
MODEL=scholar-8b-sft AGENT_MODE=none NUM_EXAMPLES=50 \
    ./scripts/data/scholar/get_response.sh simpleqa
```

输出默认写到 `eval_output/scholar/${MODEL}_${DATASET}.jsonl`，每条带
`full_traces.tool_calls`，可直接喂 DRB FACT 评测。

可调 env：`DATASET / MODEL / API_BASE / API_KEY / OUTPUT_FILE / AGENT_MODE
/ MAX_TURNS / MAX_TOKENS / TEMPERATURE / TOP_P / NUM_EXAMPLES / SUBSET /
MAX_CONCURRENCY / OVERWRITE`。

---

## 6. 评测

### 6.1 DR-Tulu 评测矩阵（SimpleQA / BrowseComp / HealthBench / ResearchQA / ShortForm / GeneticDiseases）

入口：[dr_tulu_agent_eval.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu_agent_eval.sh)（包一层 uv sync + dispatch）
→ [evaluate.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluate.py) 按任务名派发到
`scripts/eval/dr_tulu/evaluation/<task>/`。

```bash
./scripts/eval/dr_tulu_agent_eval.sh simpleqa /abs/path/to/simpleqa.jsonl
GRADER_MODEL=gpt-4.1-mini-2025-04-14 \
    ./scripts/eval/dr_tulu_agent_eval.sh researchqa /abs/path/to/results.jsonl
```

常用环境变量：

- `GRADER_MODEL`（默认 `gpt-4.1-2025-04-14`）
- `RUN_MODE`（默认 `auto_reason_search`）
- `EVAL_SAVE_PATH`、`TASK_TYPE`、`DEBUG_EVAL`

### 6.2 Deep Research Bench (DRB) —— RACE + FACT

入口：[scripts/eval/scholar/run_drb_eval.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/scholar/run_drb_eval.sh)
（包了 [run_eval.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/run_eval.py)，按 backend 处理 env 透传）

```bash
# OpenAI 兼容（默认）
LLM_API_BASE=https://api.openai.com/v1 LLM_API_KEY=$OPENAI_API_KEY \
RACE_MODEL=gpt-4.1-2025-04-14 FACT_MODEL=gpt-4.1-mini-2025-04-14 \
    ./scripts/eval/scholar/run_drb_eval.sh \
    eval_output/scholar/scholar-8b-sft_deep_research_bench.jsonl scholar_8b_sft

# 内部 HTTP judge（与 scholar_evaluator 同一条链路）
LLM_BACKEND=internal_http RACE_MODEL=gpt-5-2025-08-07 \
    ./scripts/eval/scholar/run_drb_eval.sh /abs/drb.jsonl my_run

# Gemini
LLM_BACKEND=gemini GEMINI_API_KEY=$GEMINI \
RACE_MODEL=gemini-2.5-pro FACT_MODEL=gemini-2.5-flash \
    ./scripts/eval/scholar/run_drb_eval.sh /abs/drb.jsonl my_run
```

可调 env：`INPUT_FILE / TASK_NAME / OUTPUT_DIR / MAX_WORKERS / LIMIT /
ONLY_EN / ONLY_ZH / SKIP_RACE / SKIP_FACT / SKIP_CLEANING / FORCE /
LLM_BACKEND / LLM_API_BASE / LLM_API_KEY / RACE_MODEL / FACT_MODEL /
CLEAN_MODEL`。

输出：`OUTPUT_DIR/race/<task>/race_result.txt`（每维度均分）+
`OUTPUT_DIR/fact/<task>/fact_result.txt`（citation 准确率）。

**RACE JSON fallback（方案 A+B，已集成）**：
- 方案 B：在 `SCORE_PROMPT_EN` / `SCORE_PROMPT_ZH` 里加入 7 条 `CRITICAL OUTPUT
  RULES`，强约束 judge 直接输出纯 JSON。
- 方案 A：`run_eval.py` 内置 7 级解析兜底，Method 7 是 Markdown-prose
  fallback（支持中英维度名别名），解决 judge 偶尔返回带 `**` / 标题头的 prose
  输出。

相关辅助：

- [debug_race_single.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/debug_race_single.py)
  单条调试 RACE 解析链路
- [ping_judge_and_teacher.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/ping_judge_and_teacher.py)
  探活 internal judge + Azure GPT-5 连通性
- [drb_formatter.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/drb_formatter.py) /
  [run_benchmark_scraped.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/run_benchmark_scraped.sh)

### 6.3 DeepScholar Bench 评测

DeepScholar Bench 是面向学术深度研究场景的综合评测，包含 6 个维度指标：

| # | 指标 | 说明 |
|---|------|------|
| 1 | `reference_coverage` | 引用覆盖率：模型输出引用的参考文献占 ground truth 参考文献的比例 |
| 2 | `cite_p` (citation_precision) | 引用精确度：模型 `<cite>` 标注是否准确对应所引内容 |
| 3 | `nugget_coverage` | 信息点覆盖率：ground truth 中的关键信息点被模型输出覆盖的比例 |
| 4 | `organization` | 组织结构：输出的逻辑连贯性、分段合理性 |
| 5 | `claim_coverage` | 论点覆盖率：核心论点/观点的覆盖程度 |
| 6 | `coverage_relevance_rate` | 覆盖相关性：被覆盖内容的相关性比率 |

**LLM backend 支持**：DeepScholar Bench 评测支持 `internal_http` 后端（与 scholar_evaluator / DRB 共用同一条内部 HTTP judge 链路），无需额外 API key 即可运行评测。

```bash
# 使用内部 HTTP judge
LLM_BACKEND=internal_http \
    ./scripts/eval/scholar/run_deepscholar_bench.sh \
    eval_output/scholar/scholar-8b-sft_deep_research_bench.jsonl scholar_8b_sft

# 使用 OpenAI 兼容接口
LLM_BACKEND=openai LLM_API_BASE=https://api.openai.com/v1 LLM_API_KEY=$OPENAI_API_KEY \
    ./scripts/eval/scholar/run_deepscholar_bench.sh \
    eval_output/scholar/scholar-8b-sft_deep_research_bench.jsonl scholar_8b_sft
```

### 6.4 推荐评测全流程

```bash
# 1) 起服务
SERVED_MODEL_NAME=scholar-8b-sft \
MODEL_NAME_OR_PATH=outputs/qwen3_8b_scholar_sft \
GPU_VISIBLE=0 ./scripts/serve/scholar/serve_vllm.sh &

# 2) 跑题（多轮 tool-use trajectories）
MODEL=scholar-8b-sft ./scripts/data/scholar/get_response.sh deep_research_bench

# 3) 评 DRB
LLM_BACKEND=internal_http RACE_MODEL=gpt-5-2025-08-07 \
    ./scripts/eval/scholar/run_drb_eval.sh \
    eval_output/scholar/scholar-8b-sft_deep_research_bench.jsonl scholar_8b_sft

# 4) 评 DeepScholar Bench
LLM_BACKEND=internal_http \
    ./scripts/eval/scholar/run_deepscholar_bench.sh \
    eval_output/scholar/scholar-8b-sft_deep_research_bench.jsonl scholar_8b_sft

# 5) （可选）评 DR-Tulu 矩阵
./scripts/eval/dr_tulu_agent_eval.sh simpleqa \
    eval_output/scholar/scholar-8b-sft_simpleqa.jsonl
```

---

## 7. 关键修复（Per-sample `last_time` / `curr_title` 贯通）

背景：`tools/__init__.py` 用一个模块级 `_INTERNAL_STATE` 存 `last_time` /
`curr_title` / citation 计数，`scholar_search` / `general_search` 从中读
出使用。老版本里只有一个全局常量（distill 端硬编码、RL 端完全没调），导致
所有样本共用同一个 `last_time` 且 `curr_title=None`，search 结果会把目标
paper 自己召回 —— **数据泄漏**。

### 7.1 distill 侧修复

[distill_sft_trajectories.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/distill_sft_trajectories.py#L843-L887)

新增 `_extract_sample_tool_context(ground_truth_str, default_last_time)`：
- `last_time` ← `verifiable_meta["published_date"]`（为空时 fallback 到 `--last-time`）
- `curr_title` ← `verifiable_meta["paper_title"]`

`distill_one` 改成每条样本**独立计算** context 再传给 `run_teacher_trace`。

### 7.2 RL 侧修复

这里要修两处，因为 `last_time` / `curr_title` 的数据流是：

```
parquet 的 env_config 列  (prepare_scholar_data.py 写入)
    └── dataset.map _normalize_env_config_row  (dataset_transformation.py)
        └── sample["env_config"]
            └── data_loader._merge_env_config → EnvConfigEntry.kwargs
                └── vllm_utils: actor.reset.remote(**entry.kwargs)
                    └── _ScholarToolBase.reset(**kwargs)
                        └── tools._reset_internal_state(last_time, curr_title)
                            └── tools._INTERNAL_STATE 被更新
                                └── scholar_search / general_search 读取它
```

**7.2.1 [scholar_tools.py::_ScholarToolBase.reset](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/environments/tools/scholar_tools.py#L77-L112)**

原来 `_ScholarToolBase` 没覆盖 `reset()`，继承的默认实现扔掉 `**kwargs`。
修复：覆写一个 `async def reset(self, **kwargs)`，从 kwargs 里读
`last_time`/`curr_title`，`importlib.import_module("tools")` 后调用
`_reset_internal_state(...)`。其他未知 kwargs 忽略（保持 Tool 协议兼容）。

**7.2.2 [prepare_scholar_data.py::_build_env_config](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/prepare_scholar_data.py#L142-L231)**

RLVR parquet 老版本只有 `{messages, ground_truth, dataset}`，完全没有
`env_config` 列 —— 即便 reset 接住 kwargs 也无源可接。

修复：新增 `_build_env_config(ground_truth_str)`，把 `verifiable_meta.
paper_title / published_date` 打包进 4 个 scholar env name 的 kwargs：

```python
{
  "env_configs": [
    {"env_name": "scholar_search",       "curr_title": "...", "last_time": "..."},
    {"env_name": "general_search",       "curr_title": "...", "last_time": "..."},
    {"env_name": "fetch",                "curr_title": "...", "last_time": "..."},
    {"env_name": "mock_scholar_search",  "curr_title": "...", "last_time": "..."},
  ]
}
```

`convert_file` 把它作为 `env_config` 列写进每行 parquet。
`_merge_env_config` 会按 `env_name` 把 kwargs 合进 base，交给对应
pool 的 actor 在 `reset()` 时使用。

**7.2.3 端到端验证**

复现命令（不依赖 ray/torch，纯 dataclass 级仿真）：

```bash
uv run python -c "
import asyncio, pandas as pd
from open_instruct.data_types import EnvConfig, EnvConfigEntry
from open_instruct.environments.tools.scholar_tools import ScholarSearchTool

def _merge(base, s):
    merged = dict(base.env_configs)
    for e in s.get('env_configs', []):
        name = e['env_name']; b = merged.get(name)
        extra = {k:v for k,v in e.items() if k not in ('env_name','is_text_env')}
        kwargs = {**(b.kwargs if b else {}), **extra}
        merged[name] = EnvConfigEntry(env_name=name, is_text_env=False, kwargs=kwargs)
    return EnvConfig(max_steps=base.max_steps, env_configs=merged)

df = pd.read_parquet('data/scholar_en_rlvr/search_data.2023.parquet')
sec = {'env_configs':[dict(x) for x in df.iloc[0]['env_config']['env_configs']]}
base = EnvConfig(max_steps=10, env_configs={'scholar_search':EnvConfigEntry('scholar_search',False,{})})
m = _merge(base, sec)
async def r():
    await ScholarSearchTool().reset(**m.env_configs['scholar_search'].kwargs)
    import tools
    print('last_time =', tools._INTERNAL_STATE['last_time'])
    print('curr_title=', tools._INTERNAL_STATE['curr_title'][:60])
asyncio.run(r())
"
```

期望看到：`last_time = '2024-07-01T00:00:00Z'`、`curr_title = '6G: The...'`、`page_idx = 0`。

---

## 8. 代码地图

### 8.1 数据生成

- [scripts/data/scholar/prepare_scholar_data.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/prepare_scholar_data.py)
  原始 parquet → RLVR parquet（含 `env_config` 列）
- [scripts/data/scholar/distill_sft_trajectories.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/distill_sft_trajectories.py)
  GPT-5.x teacher 蒸馏 SFT trajectories（v1，function-calling），支持 per-type reward 过滤
- [scripts/data/scholar/distill_sft_trajectories_v2.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/distill_sft_trajectories_v2.py)
  GPT-5.x teacher 蒸馏 SFT trajectories（v2，prompt-based），显式 reasoning
- [scripts/data/scholar/filter_sft_format.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/data/scholar/filter_sft_format.py)
  SFT 数据格式验证 & 过滤（5-check，与 RL 一致）

### 8.2 训练

- [open_instruct/finetune.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/finetune.py) —— 通用 SFT
- [open_instruct/grpo_fast.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/grpo_fast.py) —— GRPO RL
- [scripts/train/debug/tools/qwen3_8b_scholar_search_agent_local.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/train/debug/tools/qwen3_8b_scholar_search_agent_local.sh)
- [scripts/train/debug/tools/qwen3_8b_scholar_search_agent.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/train/debug/tools/qwen3_8b_scholar_search_agent.sh)

### 8.3 环境 & 工具适配层

- [open_instruct/environments/tools/scholar_tools.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/environments/tools/scholar_tools.py)
  `_ScholarToolBase` / `ScholarSearchTool` / `GeneralSearchTool` / `FetchTool` /
  `MockScholarSearchTool` + 对应 Config
- [open_instruct/environments/tools/utils.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/environments/tools/utils.py)
  `Tool` 基类 & `reset()` 协议
- [open_instruct/environments/tools/parsers.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/environments/tools/parsers.py)
  `dr_tulu` XML tool parser
- [open_instruct/data_loader.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/data_loader.py#L664-L683)
  `_merge_env_config`（per-sample 覆盖 base）
- [open_instruct/vllm_utils.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/vllm_utils.py#L890-L929)
  `actor.reset.remote(**entry.kwargs)` 的实际调用点
- [tools/__init__.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/tools/__init__.py)
  `_INTERNAL_STATE` + `_reset_internal_state` + `build_default_registry`

### 8.4 Reward / Evaluator

- [open_instruct/scholar_verifier.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/scholar_verifier.py) —— RL 时的 `VerifierFunction`
- [open_instruct/scholar_evaluator.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/open_instruct/scholar_evaluator.py) —— 真实打分 + distill 复用

### 8.5 评测

- [scripts/eval/dr_tulu_agent_eval.sh](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu_agent_eval.sh)
- [scripts/eval/dr_tulu/evaluate.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluate.py)
- [scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/run_eval.py](file:///mlx_devbox/users/luoyunze/playground/open-instruct/scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/run_eval.py)

---

## 9. 常见坑位速查

| 症状 | 原因 | 处理 |
|------|------|------|
| scholar search 召回目标 paper 本身 | `curr_title` 没生效 | 检查 parquet 是否有 `env_config` 列（§7.2.2）；检查 `_ScholarToolBase.reset()` 是否被覆盖（§7.2.1） |
| `last_time` 全部等于 distill 的 `--last-time` 默认值 | distill 走了老版本硬编码 | 确保脚本包含 `_extract_sample_tool_context`（§7.1） |
| Azure 报 `max_completion_tokens` 不够 | GPT-5 reasoning tokens 也计数 | 调大 `--max-completion-tokens`（v1 默认 8000，v2 默认 16000） |
| RACE "Failed to extract JSON" | judge 偶尔输出 Markdown prose | 已内置 Method 7 兜底（§6.2），若仍失败先用 `debug_race_single.py` 复现 |
| RL rollout 每条都复用同一个 `page_idx` | `_reset_internal_state` 未被调用 | 确认 RL 侧修复已生效（§7.2.1） |
| distill 续跑重复处理 | `_meta.example_id` 字段缺失或重复 | 查 `load_done_ids()` 逻辑；不要手动改 jsonl 行 |
| SFT 数据 `<answer>` 嵌套在 `<think>` 内 | v1 distill：GPT content 为空时（reasoning 隐藏在 hidden tokens），`_render_assistant_xml` 会把 `<answer>` 错误嵌套进 `<think>` | 使用 v2 prompt-based 脚本（§2.6）；或用 `filter_sft_format.py`（§2.7）过滤掉不合规样本 |
| RL `citation_precision` 偏低（~0.1） | 1) `page_idx` 在 truncation 后跳变，导致 cite id 对不上；2) `FORCE_FINAL_ANSWER_PROMPT` 中使用空 cite 格式 `<cite id="N"></cite>` | 1) `tools/__init__.py` 改为连续 `page_idx`（`_truncate_xml_documents` 重编号）；2) `vllm_utils.py` 中 `FORCE_FINAL_ANSWER_PROMPT` 改用 wrapping cite 格式 `<cite id="N">claim text</cite>` |
| 评测输出 `<think>>` 和嵌套标签 | Qwen3 thinking mode 冲突：模型内置 think 机制与 prompt 中的 `<think>` 标签互相干扰 | `generate_responses.py` 中设置 `enable_thinking: False`；增加 `_sanitize_thinking_artifacts` 后处理清理残留 |
| Cite 格式不一致（空 cite vs wrapping cite） | v1 使用空 `<cite id="N"></cite>`，不含 claim text | 全链路统一为 wrapping 格式 `<cite id="N">claim text</cite>`：SFT system prompt / RL FORCE_FINAL_ANSWER_PROMPT / Eval generate_responses.py 均已对齐 |

---

## 10. 变更一览（累计）

**新增脚本**：
- **distill_sft_trajectories_v2.py**：prompt-based 蒸馏（v2），不使用 function-calling，teacher 直接在 content 中输出 `<think>/<call_tool>/<answer>` XML，每轮保留显式 reasoning
- **filter_sft_format.py**：SFT 数据格式过滤脚本，使用与 RL 相同的 5-check 验证

**scholar_evaluator.py**：
- `_protocol_format_reward` 新增第 5 项检查 `answer_not_nested`：检测 `<answer>` 是否被嵌套在 `<think>` 内部
- 格式得分改为 `passed_checks / 5`

**vllm_utils.py**：
- `FORCE_FINAL_ANSWER_PROMPT` 中 cite 格式从空 `<cite id="N"></cite>` 改为 wrapping `<cite id="N">claim text</cite>`

**generate_responses.py**：
- system prompt 和 force-answer-prompt 中 cite 格式统一为 wrapping 格式
- 新增 `enable_thinking: False` 配置，关闭 Qwen3 内置 thinking mode 避免标签冲突
- 新增 `_sanitize_thinking_artifacts` 后处理函数，清理模型输出中的 thinking 残留（如 `<think>>`）

**tools/__init__.py**：
- 修复 `page_idx` 在 truncation 后跳变问题：`_truncate_xml_documents` 实现连续重编号
- 默认 `top_k=5`（之前无默认值）
- 新增 `_truncate_xml_documents` 辅助函数

**scholar_tools.py**：
- `str(out)[:8192]` 截断（防止单次 tool 输出过长）
- `_ScholarToolBase` 覆盖 `reset(**kwargs)`，显式调用 `tools._reset_internal_state`

**distill_sft_trajectories.py（v1）**：
- 修复 `_render_assistant_xml` 嵌套 bug（content 为空时 `<answer>` 嵌套进 `<think>`）
- scholar_evaluator 集成、per-type 阈值、`--always-score`、async scoring
- `_extract_sample_tool_context`、续跑优化

**prepare_scholar_data.py**：
- 新增 `_build_env_config` + 4 个 env 的 kwargs 写入

**DRB run_eval.py**：
- `SCORE_PROMPT_EN` / `SCORE_PROMPT_ZH` 各加 7 条 CRITICAL OUTPUT RULES（方案 B）
- 保留 Method 7 Markdown 兜底（方案 A）

**ping_judge_and_teacher.py**：新增（或恢复）探活脚本。

**data/scholar_en_rlvr/**：全部 5 份 parquet 已按新逻辑重新生成，带 `env_config` 列。

所有变更 `uv run ruff check` 通过。
