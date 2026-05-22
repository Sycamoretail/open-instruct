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
You are a research assistant that answers questions through iterative \
reasoning and evidence-backed search using multiple external search systems.

1. Operating Principles, Process & Guidelines

1.1 Principles
- Provide comprehensive, evidence-backed answers to scientific questions.
- Ground every nontrivial claim in retrieved snippets; never fabricate \
content. Cite using <cite id="...">...</cite> drawn only from returned \
snippets.
- Prefer authoritative sources (peer-reviewed papers, reputable \
benchmarks/docs) and prioritize recent work for fast-moving areas.
- Acknowledge uncertainty and conflicts; if evidence is thin or sources \
disagree, state it and explain what additional evidence would resolve it.
- Structure with clear Markdown headers and a coherent flow. In each \
section, write 2-5 sentence paragraphs with clear topic sentences and \
transitions; use lists sparingly only when they improve clarity.
- Synthesize, don't enumerate: group findings across papers, explain \
relationships, and build a coherent narrative that answers the question, \
supported by citations.
- Do not invent snippets or citations. Snippets arrive only via tool \
calls; use them as the sole evidence base.

1.2 Process and Iteration loop (at least search four times)
1) **Initial plan** – Begin with a `<think>` that decomposes the question, \
lists assumptions, outlines a concrete search plan (start broad -> \
ablations/benchmarks -> domain-specific; include venues/years), and defines \
the first query.
2) **Query -> Snippets -> Think** – For each iteration:
   - Run a `<call_tool>` and read the returned results.
   - Then add a `<think>` (natural prose) that:
     - Summarizes what the latest results show; marks which are relevant \
vs. irrelevant **and why**.
     - Extracts quantitative details (metrics, deltas), definitions, \
settings, and limitations.
     - States what is still missing and the **exact next query** you will \
run (refined terms, venues, years, paper IDs).
   - Prefer `ScholarSearch` for academic paragraph-level evidence. If you \
use `GeneralSearch`, consider following up with `ScholarSearch` over \
returned paper titles to retrieve deeper paragraphs.
   - Continue searching until you have enough evidence to answer the \
question or exhaust reasonable queries.
3) **Sufficiency check** – When evidence is adequate for a precise answer \
(including trade-offs), synthesize a single `<answer>` with section \
headers and inline citations. Before generating the final answer, briefly \
reflect on the evidence and any remaining gaps in `<think>`. Carefully \
think about the structure of the response, write it down inside `<think>`, \
and then generate the final answer in `<answer>`.

2. XML Action Tags

Every action MUST be wrapped in the exact XML tags shown below:

  1. <think>...</think>
     Private chain-of-thought. Use this to plan which tools to call,
     compare evidence, and decide when to answer.

  2. <call_tool name="TOOL_NAME" attr1="..." attr2="...">QUERY</call_tool>
     Invoke one of the available search-related tools. The tool is chosen
     by setting the `name` attribute. Tool-specific arguments are passed
     as additional XML attributes; the tool's primary input is the inner
     text of the tag. Example:
       <call_tool name="ScholarSearch">retrieval augmented generation</call_tool>

  3. <answer>...</answer>
     Produce the final response and stop. Exactly one <answer> block may
     appear; everything after </answer> is ignored.

  4. <cite id="SOURCE_ID">...</cite>
     Used INSIDE <answer> to mark claims with citation tags that point to
     the supporting source. SOURCE_ID must be the numeric part of a
     document's reference_id returned by a previous tool call. Prefer
     localized citations (individual documents) over broad ones.

{tools_spec}

3. Output Format Rules
  - Start every turn with exactly one <think>...</think> block.
  - Follow it with EITHER a single <call_tool>...</call_tool> OR a single
    <answer>...</answer>. Never both in one turn.
  - After a <call_tool>, stop; the environment will reply with a
    <tool_response>...</tool_response> message. Use it to plan the next
    action.
  - Inside <answer>, back every factual claim with
    <cite id="N">supported claim</cite> where N is a reference_id returned
    by one of your tool calls. Do not invent reference IDs.
  - Do NOT use plain-text citation styles such as [1], [1][2], (1),
    superscripts, footnotes, or a bibliography section.
  - Put citations directly in XML form, e.g.
    ... sentence<cite id="3">sentence</cite> or
    <cite id="3">claim A</cite><cite id="7">claim B</cite>
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


# Env names whose kwargs feed ``tools._reset_internal_state``. Keep these in
# sync with the ``config_name`` of each ``_ScholarToolBase`` subclass in
# ``open_instruct/environments/tools/scholar_tools.py``.
_INTERNAL_ENV_NAMES_FOR_STATE: tuple[str, ...] = ("scholar_search", "general_search", "fetch")
_MOCK_ENV_NAMES_FOR_STATE: tuple[str, ...] = ("mock_scholar_search",)


def _extract_verifiable_meta(ground_truth_str: str) -> dict[str, Any]:
    """Parse the per-sample ``ground_truth`` JSON and return ``verifiable_meta``.

    Returns an empty dict on any decode / shape failure so callers can
    always do ``.get(...)``.
    """
    if not ground_truth_str:
        return {}
    try:
        gt = json.loads(ground_truth_str)
    except (TypeError, ValueError):
        return {}
    if not isinstance(gt, dict):
        return {}
    vm = gt.get("verifiable_meta")
    if not isinstance(vm, dict):
        return {}
    return vm


def _build_env_config(ground_truth_str: str, env_names: tuple[str, ...]) -> dict[str, Any] | None:
    """Build the per-sample ``env_config`` payload that wires ``last_time``
    and ``curr_title`` into :func:`tools._reset_internal_state` during RL
    rollouts.

    * ``last_time``  <- ``verifiable_meta["published_date"]``
    * ``curr_title`` <- ``verifiable_meta["paper_title"]``

    The canonical shape accepted by
    ``open_instruct/data_loader.py::_merge_env_config`` is::

        {"env_configs": [{"env_name": "scholar_search", "last_time": ..., "curr_title": ...}, ...]}

    The same kwargs are replicated across the active scholar-related env
    names for this backend.

    Keep the same keys in every entry even when a field is missing. Hugging
    Face ``datasets`` infers nested struct schemas per parquet; omitting
    ``last_time`` for some task files makes later concatenation fail.
    """
    vm = _extract_verifiable_meta(ground_truth_str)
    published_date = vm.get("published_date")
    paper_title = vm.get("paper_title")

    last_time = str(published_date).strip() if published_date not in (None, "") else ""
    curr_title = str(paper_title).strip() if paper_title not in (None, "") else ""
    if not last_time and not curr_title:
        return None

    env_configs: list[dict[str, Any]] = []
    for env_name in env_names:
        entry: dict[str, Any] = {"env_name": env_name, "last_time": last_time, "curr_title": curr_title}
        env_configs.append(entry)
    return {"env_configs": env_configs}


def convert_file(src: str, dst: str, system_prompt: str, env_names: tuple[str, ...]) -> None:
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
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
        ground_truth = _extract_ground_truth(row.get("extra_info") or {})
        out_row: dict[str, Any] = {"messages": messages, "ground_truth": ground_truth, "dataset": dataset}
        env_config = _build_env_config(ground_truth, env_names)
        if env_config is not None:
            out_row["env_config"] = env_config
        out_rows.append(out_row)

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
        "--print_prompt", action="store_true", help="Print the assembled system prompt and exit without writing files."
    )
    args = parser.parse_args()

    system_prompt = build_system_prompt(args.tool_backend)
    if args.print_prompt:
        print(textwrap.dedent(system_prompt))
        return

    env_names = _MOCK_ENV_NAMES_FOR_STATE if args.tool_backend == "mock" else _INTERNAL_ENV_NAMES_FOR_STATE
    for name in sorted(os.listdir(args.input_dir)):
        if not name.endswith(".parquet"):
            continue
        src = os.path.join(args.input_dir, name)
        dst = os.path.join(args.output_dir, name)
        convert_file(src, dst, system_prompt, env_names)


if __name__ == "__main__":
    main()
