"""单条 RACE 调用 debug 工具：复用 cleaned_data/第一条数据，构造一次真实
score prompt，调一次 internal_http judge，把 raw response 和
extract_json_from_markdown 的解析结果都打出来。

用法::

    DRB_JUDGE_BACKEND=internal_http ./.venv/bin/python \\
        scripts/eval/dr_tulu/evaluation/deep_research_bench_eval/debug_race_single.py \\
        --task_name dr_tulu_8b_eval \\
        --output_dir eval_output/dr_tulu/drb_eval_dr_tulu_8b_eval

这样不必等整轮 RACE 跑完，就能知道 judge 到底吐了什么、
``Failed to extract JSON`` 是哪一类失败。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 直接复用 run_eval.py 里的辅助
from scripts.eval.dr_tulu.evaluation.deep_research_bench_eval import run_eval  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task_name", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument(
        "--llm_backend",
        default=os.environ.get("DRB_JUDGE_BACKEND", "internal_http"),
    )
    p.add_argument("--case_idx", type=int, default=0, help="Which case in cleaned_data to probe.")
    args = p.parse_args()

    # 加载 criteria / reference / query
    run_eval.CRITERIA_FILE, run_eval.REFERENCE_FILE, run_eval.QUERY_FILE = (
        run_eval._ensure_data_files()
    )

    cleaned_path = Path(args.output_dir) / "cleaned_data" / f"{args.task_name}.jsonl"
    if not cleaned_path.exists():
        print(f"cleaned_data not found: {cleaned_path}")
        sys.exit(1)

    cleaned = run_eval.load_jsonl(str(cleaned_path))
    target = cleaned[args.case_idx]
    prompt = target.get("prompt") or target.get("problem") or ""
    target_article = target.get("article", "")
    print(f"== probe case_idx={args.case_idx} prompt[:80]={prompt[:80]!r}")

    criteria_list = run_eval.load_jsonl(str(run_eval.CRITERIA_FILE))
    criteria_map = {item["prompt"]: item for item in criteria_list}
    reference_list = run_eval.load_jsonl(str(run_eval.REFERENCE_FILE))
    reference_map = {item["prompt"]: item for item in reference_list}

    if prompt not in criteria_map:
        print("Prompt not in criteria_map. First 3 criteria prompts:")
        for c in criteria_list[:3]:
            print("  -", c["prompt"][:80])
        sys.exit(1)
    if prompt not in reference_map:
        print("Prompt not in reference_map.")
        sys.exit(1)

    criteria_list_str = run_eval.format_criteria_list(criteria_map[prompt])
    reference_article = reference_map[prompt].get("article", "")

    # 语言从 query.jsonl 查
    query_data = run_eval.load_jsonl(str(run_eval.QUERY_FILE))
    language = "en"
    for q in query_data:
        if q.get("prompt") == prompt:
            language = q.get("language", "en")
            break
    print(f"   language={language} target_article_len={len(target_article)}")

    score_prompt = (
        run_eval.SCORE_PROMPT_ZH if language == "zh" else run_eval.SCORE_PROMPT_EN
    )
    user_prompt = score_prompt.format(
        task_prompt=prompt,
        article_1=target_article,
        article_2=reference_article,
        criteria_list=criteria_list_str,
    )

    # 调一次 judge
    llm_client = run_eval.JudgeLLMClient(backend=args.llm_backend)
    print("\n== calling judge (this may take 60-180s for a long article)...")
    raw = llm_client.generate(user_prompt=user_prompt, system_prompt="")
    print("\n== RAW RESPONSE (first 2000 chars) ==")
    print(raw[:2000])
    print(f"\n... (truncated, total len={len(raw)})")

    extracted = run_eval.extract_json_from_markdown(raw)
    print("\n== extract_json_from_markdown result ==")
    if extracted is None:
        print("  None  <-- this is the 'Failed to extract JSON' case")
    else:
        print("  first 400 chars:", extracted[:400])
        try:
            parsed = json.loads(extracted)
            print("  parsed ok. keys:", list(parsed.keys()))
            expected = [
                "comprehensiveness",
                "insight",
                "instruction_following",
                "readability",
            ]
            missing = [k for k in expected if k not in parsed]
            if missing:
                print("  MISSING DIMENSIONS:", missing)
            else:
                print("  all 4 dimensions present -- RACE would have succeeded.")
        except json.JSONDecodeError as e:
            print("  JSON parse error:", e)


if __name__ == "__main__":
    main()
