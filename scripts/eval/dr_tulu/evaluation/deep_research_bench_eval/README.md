## Introduction

We evaluate on Deep Research Bench (DRB) to acquire results for two metrics:
1. **RACE** (Reference-based and Adaptive Criteria-driven Evaluation framework with Dynamic Weighting) — article quality
2. **FACT** (Factual Abundance and Citation Trustworthiness) — citation verification

---

## Self-Contained Evaluation (Recommended)

The self-contained script `run_eval.py` bundles all evaluation code and data. **No external repo needed.**

### Prerequisites

```bash
pip install tqdm huggingface_hub requests
```

### Judge LLM Setup

Default mode uses an OpenAI-compatible endpoint, similar to the prompt -> response -> JSON parsing flow used by `scholar_evaluator.py`.

```bash
export DRB_JUDGE_API_BASE="http://127.0.0.1:30001/v1"
export DRB_RACE_MODEL="rl-research/DR-Tulu-8B"
export DRB_FACT_MODEL="rl-research/DR-Tulu-8B"
# Optional if your endpoint requires auth:
export DRB_JUDGE_API_KEY="your_api_key"
```

If you want to use the exact same internal HTTP style as `scholar_evaluator.py`, switch to the `internal_http` backend:

```bash
export DRB_JUDGE_BACKEND=internal_http
export DRB_INTERNAL_JUDGE_URL="https://ivavmlgq.fn.bytedance.net"
export DRB_INTERNAL_PROXY="http://sys-proxy-rd-relay.byted.org:8118"
```

Gemini remains available as an optional backend:

```bash
pip install google-genai
export DRB_JUDGE_BACKEND=gemini
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### Quick Start

```bash
# Full pipeline: format conversion + RACE + FACT
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model

# Explicit OpenAI-compatible judge config
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model \
    --llm_backend openai_compatible \
    --llm_api_base http://127.0.0.1:30001/v1 \
    --race_model rl-research/DR-Tulu-8B \
    --fact_model rl-research/DR-Tulu-8B

# Use the same internal HTTP judge style as scholar_evaluator.py
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model \
    --llm_backend internal_http

# RACE only:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --skip_fact

# FACT only:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --skip_race

# Test with limit:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --limit 2

# English only / Chinese only:
python evaluation/deep_research_bench_eval/run_eval.py \
    --input_file eval_output/auto_search_sft/deep_research_bench.jsonl \
    --task_name my_model --only_en
```

### Via the unified evaluate.py

```bash
python scripts/evaluate.py deep_research_bench eval_output/auto_search_sft/deep_research_bench.jsonl
```

### Output Structure

```
<output_dir>/
├── raw_data/<task_name>.jsonl          # Formatted articles
├── cleaned_data/<task_name>.jsonl      # Cleaned articles (citations removed)
├── race/<task_name>/
│   ├── raw_results.jsonl               # Per-item RACE scores
│   └── race_result.txt                 # Aggregated RACE metrics
└── fact/<task_name>/
    ├── scraped.jsonl                   # Articles with scraped citation content
    ├── validated.jsonl                 # Validated citations
    └── fact_result.txt                 # Aggregated FACT metrics
```

### Evaluation Data

The following data files are hosted at [`rl-research/dr-tulu-eval-data`](https://huggingface.co/datasets/rl-research/dr-tulu-eval-data) and **auto-downloaded on first run**:
- `query.jsonl` — 100 evaluation queries (50 EN + 50 ZH)
- `criteria.jsonl` — Task-specific evaluation criteria with weights
- `reference.jsonl` — Reference articles for comparison (from the original DRB repo)

---

## Legacy Method (External Repo)

If you prefer using the [original DRB repository](https://github.com/Ayanami0730/deep_research_bench):

### 1. Generate the DRB output from our system
Add `deep_research_bench` to the task in the scripts, e.g., `agent/scripts/auto_search.sh`.

### 2. Format conversion
```bash
python drb_formatter.py \
    --input_file_path /path/to/drb-ablation-s2.jsonl \
    --task_name drb-ablation-s2 \
    --drb_repo_path /path/to/deep_research_bench
```

### 3. Set up the external repo

```bash
git clone https://github.com/Ayanami0730/deep_research_bench
cd deep_research_bench
conda create -n drb python=3.9
conda activate drb
pip install -r requirements.txt
```

🚨 **Crucial**: The legacy external-repo flow still assumes Gemini unless you patch that repo separately.

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
export JINA_API_KEY="your_jina_api_key_here"
```

### 4. Run evaluation
Copy `run_benchmark_scraped.sh` to the DRB repo root, edit `TARGET_MODELS`, and run:
```bash
bash run_benchmark_scraped.sh
```
Results: RACE in `output_<task>.log`, FACT in `results/fact/<task>/fact_result.txt`.
