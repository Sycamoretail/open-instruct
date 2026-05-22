"""Filter SFT data using the same format validation logic as RL training.

Applies _protocol_format_reward checks from scholar_evaluator.py:
1. has_call_tool: at least one <call_tool ...>...</call_tool>
2. has_single_answer: exactly one <answer>...</answer> block
3. has_cite_inside_answer: <cite id="...">text</cite> inside the answer
4. tags_balanced: all XML tags properly balanced

Only keeps samples where ALL checks pass (format_score == 1.0).
"""

import json
import re
import sys

# Same regexes as in scholar_evaluator.py
_CALL_TOOL_RE = re.compile(
    r"<call_tool\s+([^>]*name\s*=\s*\"[^\"]+\"[^>]*)>(.*?)</call_tool>", re.DOTALL | re.IGNORECASE
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CITE_RE = re.compile(r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.*?)</cite>", re.DOTALL | re.IGNORECASE)
_CITE_CONTENT_RE = re.compile(r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.+?)</cite>", re.DOTALL | re.IGNORECASE)


def protocol_format_check(text: str) -> tuple[float, dict[str, bool]]:
    """Replica of _protocol_format_reward from scholar_evaluator.py."""
    checks: dict[str, bool] = {}

    checks["has_call_tool"] = bool(_CALL_TOOL_RE.search(text))
    answer_matches = _ANSWER_RE.findall(text)
    checks["has_single_answer"] = len(answer_matches) == 1

    if checks["has_single_answer"]:
        answer_body = answer_matches[0]
        checks["has_cite_inside_answer"] = bool(
            _CITE_CONTENT_RE.search(answer_body) or _CITE_RE.search(answer_body)
        )
    else:
        checks["has_cite_inside_answer"] = False

    # Tag balance
    call_balanced = text.count("<call_tool") == text.count("</call_tool>")
    answer_balanced = text.count("<answer>") == text.count("</answer>")
    think_balanced = text.count("<think>") == text.count("</think>")
    truncated = (
        not text.rstrip().endswith("</answer>")
        and text.count("<think>") > text.count("</think>")
    )
    checks["tags_balanced"] = call_balanced and answer_balanced and (think_balanced or truncated)

    # Check nesting: <answer> must NOT be inside <think>...</think>
    # i.e. the last <answer> must appear AFTER the last </think>
    last_think_close = text.rfind("</think>")
    last_answer_open = text.rfind("<answer>")
    checks["answer_not_nested"] = (
        last_answer_open > last_think_close if last_think_close >= 0 else True
    )

    score = sum(1 for v in checks.values() if v) / max(1, len(checks))
    return score, checks


def get_assistant_text(messages: list[dict]) -> str:
    """Concatenate all assistant message contents (excluding system prompt)."""
    parts = []
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "assistant":
            parts.append(m["content"])
    return "\n".join(parts)


def main():
    input_path = "data/scholar/sft_distill_v2.jsonl"
    output_path = "data/scholar/sft_distill_v2_filtered.jsonl"

    data = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    print(f"Total samples: {len(data)}")

    # Step 0: Auto-fix nested <answer> inside <think>
    _NESTED_RE = re.compile(r'^<think>\s*\n(<answer>.*</answer>)\s*\n?</think>\s*$', re.DOTALL)
    fixed_count = 0
    for item in data:
        for msg in item["messages"]:
            if msg["role"] != "assistant":
                continue
            m = _NESTED_RE.match(msg["content"])
            if m:
                msg["content"] = f"<think>\n(planning complete)\n</think>\n{m.group(1)}"
                fixed_count += 1
    if fixed_count:
        print(f"Auto-fixed {fixed_count} nested <answer> turns")

    # Stats
    pass_count = 0
    fail_reasons: dict[str, int] = {
        "has_call_tool": 0,
        "has_single_answer": 0,
        "has_cite_inside_answer": 0,
        "tags_balanced": 0,
        "answer_not_nested": 0,
    }
    filtered = []
    failed_examples: list[dict] = []

    for item in data:
        text = get_assistant_text(item["messages"])
        score, checks = protocol_format_check(text)

        if score == 1.0:
            pass_count += 1
            filtered.append(item)
        else:
            for key, passed in checks.items():
                if not passed:
                    fail_reasons[key] += 1
            if len(failed_examples) < 5:
                failed_examples.append({
                    "id": item.get("example_id", item.get("id", "?")),
                    "checks": checks,
                    "answer_count": len(_ANSWER_RE.findall(text)),
                    "text_snippet_tail": text[-300:],
                })

    print(f"\nPassed format check: {pass_count}/{len(data)} ({100*pass_count/len(data):.1f}%)")
    print(f"Failed: {len(data) - pass_count}")
    print(f"\nFailure breakdown (a sample can fail multiple checks):")
    for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Write filtered data
    with open(output_path, "w") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\nFiltered data written to: {output_path} ({pass_count} samples)")

    # Show failed examples
    if failed_examples:
        print("\n=== Failed Examples ===")
        for i, ex in enumerate(failed_examples):
            print(f"\n--- Failed #{i+1} ---")
            print(f"  ID: {ex['id']}")
            print(f"  Checks: {ex['checks']}")
            print(f"  Answer count: {ex['answer_count']}")
            print(f"  Tail 200 chars: {repr(ex['text_snippet_tail'][:200])}")


if __name__ == "__main__":
    main()
