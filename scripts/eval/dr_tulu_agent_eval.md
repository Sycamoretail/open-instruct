# DR-Tulu Evaluation

This repository now keeps only the DR-Tulu evaluation code, copied into:

- `scripts/eval/dr_tulu/evaluate.py`
- `scripts/eval/dr_tulu/evaluation/`
- `scripts/eval/dr_tulu/dataset_utils/`

The rest of the downloaded `dr-tulu` project is not needed for evaluation.

## What is included

- Unified evaluator entrypoint:
  - `scripts/eval/dr_tulu/evaluate.py`
- Benchmark evaluators and helper samplers:
  - `scripts/eval/dr_tulu/evaluation/`
- Minimal dataset loaders that some evaluators depend on:
  - `scripts/eval/dr_tulu/dataset_utils/`

## Install dependencies

```bash
cd /mlx_devbox/users/luoyunze/playground/open-instruct
uv sync --extra dr-tulu
```

The `dr-tulu` extra now means evaluation-only dependencies, not the full `dr-agent` package.

## Usage

Run the Python entrypoint directly:

```bash
./.venv/bin/python scripts/eval/dr_tulu/evaluate.py simpleqa /abs/path/to/simpleqa.jsonl
```

Or use the small wrapper:

```bash
scripts/eval/dr_tulu_agent_eval.sh simpleqa /abs/path/to/simpleqa.jsonl
```

Examples:

```bash
scripts/eval/dr_tulu_agent_eval.sh healthbench /abs/path/to/healthbench.jsonl

GRADER_MODEL=gpt-4.1-mini-2025-04-14 \
scripts/eval/dr_tulu_agent_eval.sh researchqa /abs/path/to/researchqa.jsonl
```

## Common overrides

```bash
EVAL_SAVE_PATH=/abs/path/to/task_eval_results.json \
GRADER_MODEL=gpt-4.1-2025-04-14 \
RUN_MODE=auto_reason_search \
scripts/eval/dr_tulu_agent_eval.sh genetic_diseases_qa /abs/path/to/genetic.jsonl
```

## Notes

- `sqa_cs_v2` still needs extra dependencies such as `astabench` and `inspect_ai`.
- `deep_research_bench` still needs its own grader-side dependencies such as `google-genai`.
- The copied evaluation tree intentionally preserves upstream layout to minimize code changes.
