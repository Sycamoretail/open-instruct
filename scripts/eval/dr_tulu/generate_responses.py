#!/usr/bin/env python3
"""
Generate model responses for DR-Tulu evaluation datasets using an OpenAI-compatible endpoint.

Two modes:

1. ``--agent-mode none``
   Plain single-turn chat completion. ``full_traces`` is left as ``{}``.
   Suitable for chat-style evals that don't need tool traces.

2. ``--agent-mode internal`` (default)
   Multi-turn XML agent loop, matching the training protocol used in
   ``scripts/data/scholar/prepare_scholar_data.py``:

       <think>...</think>
       <call_tool name="ScholarSearch" top_k="5">...</call_tool>
       <answer>...<cite id="...">...</cite>...</answer>

   Tools are executed by ``tools.build_default_registry("internal")``
   (ScholarSearch / GeneralSearch / Fetch). After each tool call, its
   XML output is wrapped in ``<tool_response>...</tool_response>`` and
   appended to the conversation as a ``user`` turn. The loop stops when
   the model emits ``<answer>`` or hits ``--max-turns``.

   The resulting JSONL contains a ``full_traces.tool_calls`` field in
   the exact shape DRB's ``format_drb_data`` expects, so FACT evaluation
   works end-to-end.

Examples:
    python scripts/eval/dr_tulu/generate_responses.py \\
        --dataset deep_research_bench \\
        --model dr-tulu-8b \\
        --api-base http://127.0.0.1:30001/v1 \\
        --output-file eval_output/dr_tulu/deep_research_bench_dr_tulu_8b.jsonl

    python scripts/eval/dr_tulu/generate_responses.py \\
        --dataset simpleqa --agent-mode none \\
        --model my-model \\
        --api-base http://127.0.0.1:30001/v1 \\
        --num-examples 50 \\
        --output-file eval_output/dr_tulu/simpleqa_my_model_50.jsonl
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[2]
for candidate in (SCRIPT_ROOT, REPO_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import tools as internal_tools  # noqa: E402
from dataset_utils.load_dataset import SUPPORTED_TASKS, load_dataset  # noqa: E402

from open_instruct import logger_utils  # noqa: E402

logger = logger_utils.setup_logger(__name__)


# ---------------------------------------------------------------------------
# XML protocol (matches training-time system prompt in prepare_scholar_data.py).
# ---------------------------------------------------------------------------

CALL_TOOL_RE = re.compile(
    r"<call_tool([^>]*)>(.*?)</call_tool>",
    re.DOTALL,
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

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
  - Do NOT use plain-text citation styles such as [1], [1][2], (1),
    superscripts, footnotes, or a bibliography section.
  - Put citations directly in XML form after the supported claim, e.g.
    ... sentence.<cite id="3"></cite> or
    ... sentence.<cite id="3"></cite><cite id="7"></cite>
"""


def build_default_system_prompt() -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(tools_spec=_INTERNAL_TOOLS_SPEC)


# ---------------------------------------------------------------------------
# Tool registry (lazy; only loaded when agent-mode == internal).
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, Any] = {}
_REGISTRY_LOCK = threading.Lock()


def get_internal_registry(tool_backend: str = "internal"):
    """Return a lazily-initialised tool registry for the given backend.

    Backends: ``mock`` / ``internal`` / ``real`` (mirrors the RL-side
    ``tools.build_default_registry``).
    """
    if tool_backend not in _REGISTRY:
        with _REGISTRY_LOCK:
            if tool_backend not in _REGISTRY:
                _REGISTRY[tool_backend] = internal_tools.build_default_registry(tool_backend)
    return _REGISTRY[tool_backend]


def reset_internal_state() -> None:
    """Reset reference-id counters so each rollout starts from 0."""
    internal_tools._reset_internal_state()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Parse internal tool XML output into DRB-friendly records.
# ---------------------------------------------------------------------------


_DOC_RE = re.compile(r"<document\b[^>]*>(.*?)</document>", re.DOTALL)
_DOC_REFID_RE = re.compile(r'reference_id\s*=\s*"([^"]+)"')
_DOC_FIELD_RE = {
    "title": re.compile(r"<title>(.*?)</title>", re.DOTALL),
    "snippet": re.compile(r"<snippet>(.*?)</snippet>", re.DOTALL),
    "abstract": re.compile(r"<abstract>(.*?)</abstract>", re.DOTALL),
    "url": re.compile(r"<url>(.*?)</url>", re.DOTALL),
    "author": re.compile(r"<author>(.*?)</author>", re.DOTALL),
    "publish_time": re.compile(r"<publish_time>(.*?)</publish_time>", re.DOTALL),
}


def _strip(s: str | None) -> str:
    return (s or "").strip()


def parse_internal_tool_output(raw: str) -> list[dict[str, str]]:
    """Parse internal <document reference_id="..."> blocks into Title/URL/Snippet.

    DRB's ``parse_search_results`` expects plain-text blocks of the form
    ``Title: ...\\nURL: ...\\nSnippet: ...``. We pre-format internal XML
    into that shape (plus keep reference_id available via the tool-call
    record) so the FACT pipeline can consume the traces directly.
    """
    records: list[dict[str, str]] = []
    for match in _DOC_RE.finditer(raw or ""):
        header_match = _DOC_REFID_RE.search(match.group(0))
        reference_id = _strip(header_match.group(1) if header_match else "")
        body = match.group(1)
        title = ""
        for key in ("title",):
            m = _DOC_FIELD_RE[key].search(body)
            if m:
                title = _strip(m.group(1))
                break
        url = ""
        for key in ("url",):
            m = _DOC_FIELD_RE[key].search(body)
            if m:
                url = _strip(m.group(1))
                break
        snippet = ""
        for key in ("snippet", "abstract"):
            m = _DOC_FIELD_RE[key].search(body)
            if m:
                snippet = _strip(m.group(1))
                break
        if not (title or url or snippet):
            continue
        records.append(
            {
                "reference_id": reference_id,
                "Title": title,
                "URL": url,
                "Snippet": snippet,
            }
        )
    return records


def records_to_drb_block(records: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for rec in records:
        chunks.append(
            f"Title: {rec.get('Title', '')}\n"
            f"URL: {rec.get('URL', '')}\n"
            f"Snippet: {rec.get('Snippet', '')}"
        )
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model responses for DR-Tulu evaluation datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(
            set(SUPPORTED_TASKS.keys())
            | {"browsecomp", "simpleqa", "healthbench", "dsqa", "webshaper"}
        ),
        help="Dataset name to run generation on.",
    )
    parser.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="Path to save JSONL generation results.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name served by the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Base URL of the OpenAI-compatible endpoint, e.g. http://127.0.0.1:30001/v1",
    )
    parser.add_argument("--api-key", default=None, help="Optional API key for the endpoint.")
    parser.add_argument(
        "--num-examples",
        default=None,
        help="Number of examples to run, or one of: ablation, final_run, final_run_100.",
    )
    parser.add_argument("--subset", default=None, help="Dataset subset, mainly used for healthbench.")
    parser.add_argument("--local-path", default=None, help="Optional local dataset path override.")
    parser.add_argument(
        "--system-prompt",
        default=None,
        help=(
            "Optional system prompt. "
            "Defaults to the internal agent prompt when --agent-mode=internal, "
            "empty when --agent-mode=none."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096, help="Maximum output tokens per LLM call.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Optional top-p sampling parameter.")
    parser.add_argument("--timeout-s", type=int, default=3600, help="Request timeout in seconds.")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Concurrent generation requests.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per LLM call on failure.")
    parser.add_argument(
        "--max-tool-response-chars",
        type=int,
        default=12000,
        help=(
            "Maximum characters of each raw tool response fed back to the model. "
            "Use <=0 to disable truncation. Full parsed traces are still saved."
        ),
    )
    parser.add_argument(
        "--skip-additional-instructions",
        action="store_true",
        help="Do not append dataset-provided additional instructions to the user prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--agent-mode",
        choices=["internal", "none"],
        default="internal",
        help="Agent mode: 'internal' runs the XML tool-use loop (default); 'none' is single-turn chat.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=6,
        help="Max (think+tool_or_answer) turns for agent-mode=internal.",
    )
    parser.add_argument(
        "--force-answer-after-max-turns",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When the internal agent exhausts --max-turns without <answer>, "
            "append a no-more-tools instruction and make one final LLM call. "
            "The final call is never executed as a tool call."
        ),
    )
    parser.add_argument(
        "--force-answer-prompt",
        default=(
            "You have reached the maximum number of allowed tool calls. "
            "Do not call any tools again. Based only on the evidence already returned above, "
            "write the final answer now. Your response MUST contain exactly one "
            "<think>...</think> block followed by exactly one <answer>...</answer> block. "
            "Inside <answer>, cite supporting evidence with <cite id=\"SOURCE_ID\"></cite> "
            "using only source IDs that appeared in the previous tool responses. "
            "Do NOT use bracket citations like [1], [1][2], superscripts, footnotes, or a References section. "
            "Every factual sentence in <answer> must end with one or more empty citation tags such as "
            "<cite id=\"12\"></cite> or <cite id=\"12\"></cite><cite id=\"18\"></cite>. "
            "The citation tag itself must stay exactly in XML form; do not replace it with plain-text numbers. "
            "If the evidence is incomplete, state the uncertainty instead of searching again."
        ),
        help="User message appended before the forced final-answer call.",
    )
    parser.add_argument(
        "--stop-after-answer",
        action="store_true",
        default=True,
        help="Truncate final_response at the first </answer>.",
    )
    return parser.parse_args()


def parse_num_examples(value: str | None) -> int | str | None:
    if value is None:
        return None
    if value in {"ablation", "final_run", "final_run_100"}:
        return value
    return int(value)


# ---------------------------------------------------------------------------
# LLM helpers.
# ---------------------------------------------------------------------------


def build_messages(
    example: dict[str, Any],
    system_prompt: str,
    skip_additional_instructions: bool,
) -> list[dict[str, str]]:
    user_prompt = example["problem"]
    extra = example.get("additional_instructions")
    if extra and not skip_additional_instructions:
        user_prompt = f"{user_prompt}\n\n{extra}".strip()

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def extract_response_text(response_json: dict[str, Any]) -> tuple[str, str]:
    """Return (content_text, finish_reason) from a chat/completions response."""
    choices = response_json.get("choices") or []
    if not choices:
        raise ValueError(f"Missing choices in response: {response_json}")
    choice = choices[0]
    finish_reason = str(choice.get("finish_reason") or "")
    message = choice.get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content, finish_reason
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_chunks.append(item.get("text", ""))
        text = "".join(text_chunks).strip()
        if text:
            return text, finish_reason
    raise ValueError(f"Unsupported message content in response: {response_json}")


def call_llm(
    *,
    api_base: str,
    api_key: str | None,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    timeout_s: int,
    stop: list[str] | None = None,
) -> tuple[str, str]:
    """Call the OpenAI-compatible endpoint and return (content, finish_reason)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if stop:
        payload["stop"] = stop

    response = requests.post(
        f"{api_base}/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout_s,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:2000] if response.text else ""
        approx_chars = sum(len(m.get("content", "")) for m in messages)
        raise requests.HTTPError(
            f"{exc}; response_body={body!r}; "
            f"num_messages={len(messages)} approx_prompt_chars={approx_chars} "
            f"max_tokens={max_tokens}"
        ) from exc
    return extract_response_text(response.json())


# ---------------------------------------------------------------------------
# Agent loop.
# ---------------------------------------------------------------------------


def _extract_first_call_tool(text: str) -> tuple[str, str] | None:
    match = CALL_TOOL_RE.search(text)
    if match is None:
        return None
    return match.group(0), text[: match.end()]


def _extract_answer(text: str) -> str | None:
    match = ANSWER_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def run_internal_agent(
    *,
    example: dict[str, Any],
    system_prompt: str,
    api_base: str,
    api_key: str | None,
    model: str,
    temperature: float,
    top_p: float | None,
    max_tokens: int,
    timeout_s: int,
    max_retries: int,
    max_turns: int,
    max_tool_response_chars: int,
    force_answer_after_max_turns: bool,
    force_answer_prompt: str,
    skip_additional_instructions: bool,
    tool_backend: str = "internal",
) -> dict[str, Any]:
    """Drive the <call_tool>/<answer> loop using the internal tool registry."""

    registry = get_internal_registry(tool_backend)
    reset_internal_state()

    messages = build_messages(example, system_prompt, skip_additional_instructions)

    tool_calls: list[dict[str, Any]] = []
    finish_reasons: list[str] = []
    final_response: str = ""

    def call_with_retry() -> tuple[str, str]:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                return call_llm(
                    api_base=api_base,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    stop=None,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "LLM call failed attempt=%s/%s error=%s", attempt, max_retries, exc
                )
        raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")

    last_finish_reason = ""
    truncated = False
    forced_answer_attempted = False
    max_turns_exhausted = False
    for _turn in range(max_turns):
        assistant_text, finish_reason = call_with_retry()
        last_finish_reason = finish_reason
        finish_reasons.append(finish_reason)

        answer = _extract_answer(assistant_text)
        tool_match = _extract_first_call_tool(assistant_text)

        if answer is not None and (tool_match is None or assistant_text.index("<answer>") < tool_match[1].rfind("<call_tool")):
            final_response = assistant_text
            messages.append({"role": "assistant", "content": assistant_text})
            break

        if tool_match is None:
            # No closed <call_tool> and no <answer>. Check whether the model
            # started a tool call that got truncated so we surface it clearly
            # in the trace rather than silently returning an empty tool_calls
            # list.
            has_open_call = "<call_tool" in assistant_text and "</call_tool>" not in assistant_text
            has_open_think = "<think>" in assistant_text and "</think>" not in assistant_text
            has_open_answer = "<answer>" in assistant_text and "</answer>" not in assistant_text
            if has_open_call or has_open_think or has_open_answer or finish_reason == "length":
                truncated = True
                tool_calls.append(
                    {
                        "call_id": f"call_truncated_{uuid.uuid4().hex[:8]}",
                        "raw_call": "",
                        "ok": False,
                        "truncated": True,
                        "finish_reason": finish_reason,
                        "output": "",
                        "raw_output": {"tool_outputs": [], "raw_text": ""},
                        "generated_text": assistant_text,
                        "diagnostic": {
                            "has_open_call_tool": has_open_call,
                            "has_open_think": has_open_think,
                            "has_open_answer": has_open_answer,
                            "tail": assistant_text[-200:],
                        },
                    }
                )
                logger.warning(
                    "Generation truncated example=%s finish_reason=%s open_call=%s open_think=%s open_answer=%s tail=%r",
                    example.get("id", example.get("example_id", "?")),
                    finish_reason,
                    has_open_call,
                    has_open_think,
                    has_open_answer,
                    assistant_text[-120:],
                )
            final_response = assistant_text
            messages.append({"role": "assistant", "content": assistant_text})
            break

        raw_call_block, assistant_prefix = tool_match
        messages.append({"role": "assistant", "content": assistant_prefix})

        call_id = f"call_{uuid.uuid4().hex[:12]}"
        try:
            ok, tool_raw = registry.execute(raw_call_block)
        except Exception as exc:  # noqa: BLE001
            ok, tool_raw = False, f"[tool error] {type(exc).__name__}: {exc}"

        tool_records = parse_internal_tool_output(tool_raw) if ok else []
        drb_text = records_to_drb_block(tool_records) if tool_records else ""

        # DRB FACT maps citations via "{call_id}-{i}" keys. Rewrite the
        # reference_ids in the <tool_response> that goes back to the model,
        # so when the model emits <cite id="..."> it matches the traces
        # dictionary built by ``format_drb_data``.
        rewritten_raw = tool_raw
        for idx, rec in enumerate(tool_records):
            orig = rec.get("reference_id") or ""
            if not orig:
                continue
            new_id = f"{call_id}-{idx}"
            rewritten_raw = rewritten_raw.replace(orig, new_id)

        tool_outputs = [
            {
                "reference_id": f"{call_id}-{idx}",
                "title": rec.get("Title", ""),
                "url": rec.get("URL", ""),
                "snippet": rec.get("Snippet", ""),
                "output": (
                    f"Title: {rec.get('Title', '')}\n"
                    f"URL: {rec.get('URL', '')}\n"
                    f"Snippet: {rec.get('Snippet', '')}"
                ),
            }
            for idx, rec in enumerate(tool_records)
        ]

        tool_calls.append(
            {
                "call_id": call_id,
                "raw_call": raw_call_block,
                "ok": ok,
                "finish_reason": finish_reason,
                "truncated": False,
                "output": drb_text,
                "raw_output": {
                    "tool_outputs": tool_outputs,
                    "raw_text": tool_raw if isinstance(tool_raw, str) else str(tool_raw),
                },
                "generated_text": assistant_prefix,
            }
        )

        model_tool_raw = rewritten_raw
        if max_tool_response_chars > 0 and len(model_tool_raw) > max_tool_response_chars:
            omitted = len(model_tool_raw) - max_tool_response_chars
            model_tool_raw = (
                model_tool_raw[:max_tool_response_chars]
                + f"\n\n[tool response truncated: omitted {omitted} characters]"
            )
        tool_response = f"<tool_response>\n{model_tool_raw}\n</tool_response>"
        messages.append({"role": "user", "content": tool_response})

    if not final_response:
        max_turns_exhausted = True

    if (
        not final_response
        and force_answer_after_max_turns
        and force_answer_prompt.strip()
    ):
        forced_answer_attempted = True
        messages.append({"role": "user", "content": force_answer_prompt.strip()})
        assistant_text, finish_reason = call_with_retry()
        last_finish_reason = finish_reason
        finish_reasons.append(finish_reason)
        final_response = assistant_text
        messages.append({"role": "assistant", "content": assistant_text})

    if not final_response:
        final_response = messages[-1]["content"] if messages and messages[-1]["role"] == "assistant" else ""

    return {
        "final_response": final_response,
        "tool_calls": tool_calls,
        "num_turns": len(tool_calls) + (1 if _extract_answer(final_response) else 0),
        "finish_reasons": finish_reasons,
        "last_finish_reason": last_finish_reason,
        "truncated": truncated,
        "max_turns_exhausted": max_turns_exhausted,
        "forced_answer_attempted": forced_answer_attempted,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------


def stable_example_id(example: dict[str, Any]) -> str:
    value = example.get("id")
    if value is not None:
        return str(value)
    return hashlib.md5(example["problem"].encode()).hexdigest()


def load_processed_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()

    processed_ids: set[str] = set()
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("generation_error"):
                continue
            example_id = obj.get("example_id")
            if example_id is not None:
                processed_ids.add(str(example_id))
    return processed_ids


def main() -> None:
    args = parse_args()

    api_base = args.api_base or ""
    if not api_base:
        raise ValueError("--api-base is required.")
    api_base = api_base.rstrip("/")
    if not api_base.endswith("/v1"):
        api_base = f"{api_base}/v1"

    if args.output_file.exists() and args.overwrite:
        args.output_file.unlink()
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.system_prompt is None:
        system_prompt = build_default_system_prompt() if args.agent_mode == "internal" else ""
    else:
        system_prompt = args.system_prompt

    processed_ids = load_processed_ids(args.output_file)
    if processed_ids:
        logger.info(
            "Resuming generation: found %s existing examples in %s",
            len(processed_ids),
            args.output_file,
        )

    dataset_config = {
        "name": args.dataset,
        "num_examples": parse_num_examples(args.num_examples),
        "subset": args.subset,
        "local_path": args.local_path,
    }
    examples = load_dataset(dataset_config)
    pending_examples = [
        example for example in examples if stable_example_id(example) not in processed_ids
    ]

    logger.info(
        "Loaded %s examples for dataset=%s, pending=%s, agent_mode=%s",
        len(examples),
        args.dataset,
        len(pending_examples),
        args.agent_mode,
    )

    if args.agent_mode == "internal":
        get_internal_registry()

    file_lock = threading.Lock()
    pbar = tqdm(total=len(pending_examples), desc=f"Generate {args.dataset}")

    def generate_one(example: dict[str, Any]) -> dict[str, Any]:
        example_id = stable_example_id(example)
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                if args.agent_mode == "internal":
                    agent_result = run_internal_agent(
                        example=example,
                        system_prompt=system_prompt,
                        api_base=api_base,
                        api_key=args.api_key,
                        model=args.model,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout_s,
                        max_retries=args.max_retries,
                        max_turns=args.max_turns,
                        max_tool_response_chars=args.max_tool_response_chars,
                        force_answer_after_max_turns=args.force_answer_after_max_turns,
                        force_answer_prompt=args.force_answer_prompt,
                        skip_additional_instructions=args.skip_additional_instructions,
                    )
                    result = {
                        "example_id": example_id,
                        "problem": example["problem"],
                        "final_response": agent_result["final_response"],
                        "full_traces": {"tool_calls": agent_result["tool_calls"]},
                        "num_turns": agent_result["num_turns"],
                        "finish_reasons": agent_result["finish_reasons"],
                        "last_finish_reason": agent_result["last_finish_reason"],
                        "truncated": agent_result["truncated"],
                        "max_turns_exhausted": agent_result["max_turns_exhausted"],
                        "forced_answer_attempted": agent_result["forced_answer_attempted"],
                        "original_data": example,
                        "model": args.model,
                    }
                else:
                    messages = build_messages(
                        example,
                        system_prompt=system_prompt,
                        skip_additional_instructions=args.skip_additional_instructions,
                    )
                    response_text, finish_reason = call_llm(
                        api_base=api_base,
                        api_key=args.api_key,
                        model=args.model,
                        messages=messages,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_tokens=args.max_tokens,
                        timeout_s=args.timeout_s,
                    )
                    result = {
                        "example_id": example_id,
                        "problem": example["problem"],
                        "final_response": response_text,
                        "full_traces": {},
                        "finish_reasons": [finish_reason],
                        "last_finish_reason": finish_reason,
                        "truncated": finish_reason == "length",
                        "original_data": example,
                        "model": args.model,
                    }

                with file_lock, args.output_file.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                return result
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Generation failed for example_id=%s attempt=%s/%s error=%s",
                    example_id,
                    attempt,
                    args.max_retries,
                    exc,
                )
        result = {
            "example_id": example_id,
            "problem": example["problem"],
            "final_response": "",
            "full_traces": {},
            "finish_reasons": [],
            "last_finish_reason": "error",
            "truncated": False,
            "generation_error": str(last_error),
            "original_data": example,
            "model": args.model,
        }
        with file_lock, args.output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    failures = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.max_concurrency)
    ) as executor:
        futures = [executor.submit(generate_one, example) for example in pending_examples]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures += 1
                logger.error("Generation failed: %s", exc)
            finally:
                pbar.update(1)
    pbar.close()

    logger.info(
        "Generation finished. output=%s total=%s completed=%s failures=%s",
        args.output_file,
        len(examples),
        len(pending_examples) - failures,
        failures,
    )


if __name__ == "__main__":
    main()
