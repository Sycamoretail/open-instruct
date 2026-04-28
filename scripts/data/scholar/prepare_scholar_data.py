"""Convert ``./scholar/*.parquet`` to open-instruct RLVR format.

Changes versus the previous revision:
    * Drop the system prompt that ships with the raw parquet.
    * Keep only the user's ``query`` turn.
    * Prepend a fresh system prompt that specifies the deep-research
      protocol from the paper figure (``think`` / ``tool`` / ``answer`` /
      ``cite``) and advertises the company-internal tools (or their mock
      fallbacks) registered in ``open_instruct.environments.tools``.

Output columns: ``{messages, ground_truth, dataset}`` -- exactly what
``open_instruct/dataset_transformation.py::rlvr_tokenize_v3`` expects.

Usage::

    python3 scripts/data/scholar/prepare_scholar_data.py \\
        --input_dir scholar \\
        --output_dir scholar_rlvr \\
        --tool_backend internal   # or "mock" for the offline fake
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# System prompt (matches the action space from the figure in the user msg).
# ---------------------------------------------------------------------------


_INTERNAL_TOOLS_SPEC = """\
You have access to the following search-related tools:

- name="ScholarSearch": Search the internal scholar corpus. Inner text is a
  natural-language academic query (Chinese or English). Returns an XML
  block of <document reference_id="<|superscript|>:N"> entries with
  title / abstract / author / url / publish_time.
- name="GeneralSearch": General-purpose web search. Inner text is a single
  query. Returns the same <document>-block XML.
- name="Fetch": Open a URL (web page or PDF) returned by a previous search
  call and extract its content. Inner text is the URL; set the
  description attribute to state what you want to extract."""


_MOCK_TOOLS_SPEC = """\
You have access to one search tool:

- name="ScholarSearch": Search a mock scholar corpus. Inner text is a
  natural-language academic query. Returns an XML block of
  <document reference_id="scholar:N"> entries."""


_SYSTEM_PROMPT_TEMPLATE = """\
You are a deep-research assistant. You answer the user's question by
thinking in natural language, calling external search tools when needed,
and finally producing a cited answer.

You operate with the following action space: {{think, tool, answer, cite}}.
Every action MUST be wrapped in the exact XML tags shown below:

  1. <think>...</think>
     Private chain-of-thought. Use this to plan which tools to call,
     compare evidence, and decide when to answer.

  2. <call_tool name="TOOL_NAME" attr1="..." attr2="...">QUERY</call_tool>
     Invoke one of the available search-related tools. The tool is chosen
     by setting the `name` attribute. Tool-specific arguments are passed
     as additional XML attributes; the tool's primary input is the inner
     text of the tag. Example:
       <call_tool name="ScholarSearch" top_k="5">retrieval augmented generation</call_tool>

  3. <answer>...</answer>
     Produce the final response and stop. Exactly one <answer> block may
     appear; everything after </answer> is ignored.

  4. <cite id="SOURCE_ID"></cite>
     Used INSIDE <answer> to wrap claims in citation tags that point to
     the supporting source. SOURCE_ID must be the numeric part of a
     document's reference_id returned by a previous tool call. Prefer
     localized citations (individual documents) over broad ones.

{tools_spec}

Output format rules:
  - Start every turn with exactly one <think>...</think> block.
  - Follow it with EITHER a single <call_tool>...</call_tool> OR a single
    <answer>...</answer>. Never both in one turn.
  - After a <call_tool>, stop; the environment will reply with a
    <tool_response>...</tool_response> message. Use it to plan the next
    action.
  - Inside <answer>, back every factual claim with <cite id="N"></cite>
    where N is a reference_id returned by one of your tool calls. Do not
    invent reference IDs.
"""


def build_system_prompt(tool_backend: str) -> str:
    tools_spec = _MOCK_TOOLS_SPEC if tool_backend == "mock" else _INTERNAL_TOOLS_SPEC
    return _SYSTEM_PROMPT_TEMPLATE.format(tools_spec=tools_spec)


# ---------------------------------------------------------------------------
# Row transforms.
# ---------------------------------------------------------------------------


def _extract_user_query(prompt: list[dict[str, Any]]) -> str:
    """Drop system messages; keep only the user question.

    Raw scholar parquet packs the task-specific query into a single user
    turn (role=="user"). If there are multiple user turns we concatenate
    their content with blank lines.
    """
    chunks: list[str] = []
    for msg in prompt or []:
        role = (msg.get("role") or "").lower()
        content = msg.get("content") or ""
        if role == "user" and content.strip():
            chunks.append(content.strip())
    return "\n\n".join(chunks).strip()


def _extract_ground_truth(extra_info: dict[str, Any]) -> str:
    rm = (extra_info or {}).get("reward_model") or {}
    gt = rm.get("ground_truth")
    if gt is None:
        return "{}"
    if isinstance(gt, (dict, list)):
        return json.dumps(gt, ensure_ascii=False)
    return str(gt)


def convert_file(src: str, dst: str, system_prompt: str) -> None:
    table = pq.read_table(src)
    rows = table.to_pylist()

    out_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        dataset = row.get("data_source") or "scholar_search"
        user_query = _extract_user_query(row.get("prompt") or [])
        if not user_query:
            skipped += 1
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]
        out_rows.append(
            {
                "messages": messages,
                "ground_truth": _extract_ground_truth(row.get("extra_info") or {}),
                "dataset": dataset,
            }
        )

    out_table = pa.Table.from_pylist(out_rows)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    pq.write_table(out_table, dst)
    print(f"[scholar-prep] {src} -> {dst}  rows={len(out_rows)} skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", default="scholar")
    parser.add_argument("--output_dir", default="scholar_rlvr")
    parser.add_argument(
        "--tool_backend",
        default="internal",
        choices=["internal", "mock"],
        help="Which tools the system prompt should advertise. Should match --tools in the training script.",
    )
    parser.add_argument(
        "--print_prompt",
        action="store_true",
        help="Print the assembled system prompt and exit without writing files.",
    )
    args = parser.parse_args()

    system_prompt = build_system_prompt(args.tool_backend)
    if args.print_prompt:
        print(textwrap.dedent(system_prompt))
        return

    for name in sorted(os.listdir(args.input_dir)):
        if not name.endswith(".parquet"):
            continue
        src = os.path.join(args.input_dir, name)
        dst = os.path.join(args.output_dir, name)
        convert_file(src, dst, system_prompt)


if __name__ == "__main__":
    main()
