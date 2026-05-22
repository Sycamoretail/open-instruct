"""Distill multi-turn tool-use SFT trajectories from a GPT teacher (prompt-based).

Unlike v1 (``distill_sft_trajectories.py``) which uses native function-calling
and loses the teacher's reasoning (hidden in reasoning tokens), this script uses
a **prompt-based** approach: tools are described in the system prompt and the
teacher outputs ``<think>``/``<call_tool>`` tags directly in plain text. This
ensures every turn has explicit reasoning that transfers to the student model.

Pipeline
========

* Teacher side: GPT-5.x (or GPT-4o) via Azure OpenAI (ByteDance gpt_openapi).
  **No** ``tools=[...]`` parameter is passed; instead, the system prompt
  describes the XML tool protocol and available tools.
* The teacher is instructed to output:
      <think>reasoning...</think>
      <call_tool name="...">query</call_tool>
  or:
      <think>sufficiency check...</think>
      <answer>...</answer>
* The script parses ``<call_tool>`` from the text, executes the real tool,
  and returns results as ``<tool_response>...</tool_response>`` in a user turn.
* Output: identical ``{"messages": [...]}`` format to v1, ready for SFT.

Example::

    python scripts/data/scholar/distill_sft_trajectories_v2.py \\
        --input data/scholar_en_rlvr/search_data.2023.parquet \\
                data/scholar_en_rlvr/understanding_data.2023.parquet \\
        --output-jsonl data/scholar/sft_distill_v3.jsonl \\
        --ak-list "$AK1,$AK2" \\
        --model-list "gpt-5-2025-08-07" \\
        --max-steps 8 \\
        --max-completion-tokens 16000 \\
        --num-examples 50
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import openai
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_instruct import logger_utils, scholar_evaluator  # noqa: E402, I001
from scripts.data.scholar.prepare_scholar_data import build_system_prompt  # noqa: E402, I001
from tools.linkreader_tool import Fetch  # noqa: E402, I001
from tools.scholar_search_tool import ScholarSearchTool  # noqa: E402, I001
from tools.search_tool import GeneralSearchTool  # noqa: E402, I001

logger = logger_utils.setup_logger(__name__)


# ---------------------------------------------------------------------------
# Azure OpenAI config.
# ---------------------------------------------------------------------------

DEFAULT_AZURE_ENDPOINT = "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
DEFAULT_AZURE_API_VERSION = "2024-03-01-preview"

_CLIENT_CACHE: dict[str, openai.AzureOpenAI] = {}
_CLIENT_CACHE_LOCK = Lock()


def _get_or_create_client(ak: str, endpoint: str, api_version: str, timeout_s: float) -> openai.AzureOpenAI:
    key = f"{endpoint}::{api_version}::{ak}"
    cli = _CLIENT_CACHE.get(key)
    if cli is not None:
        return cli
    with _CLIENT_CACHE_LOCK:
        cli = _CLIENT_CACHE.get(key)
        if cli is None:
            cli = openai.AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, api_key=ak, timeout=timeout_s)
            _CLIENT_CACHE[key] = cli
    return cli


# ---------------------------------------------------------------------------
# Tool singletons.
# ---------------------------------------------------------------------------

_TOOLS_LOCK = Lock()
_TOOLS: dict[str, Any] = {}


def _get_tools() -> tuple[Any, Any, Any]:
    if _TOOLS:
        return _TOOLS["search"], _TOOLS["linkreader"], _TOOLS["scholar"]
    with _TOOLS_LOCK:
        if not _TOOLS:
            _TOOLS["search"] = GeneralSearchTool(return_prompt_type="xml")
            _TOOLS["linkreader"] = Fetch(return_prompt_type="xml")
            _TOOLS["scholar"] = ScholarSearchTool(return_prompt_type="xml")
    return _TOOLS["search"], _TOOLS["linkreader"], _TOOLS["scholar"]


# ---------------------------------------------------------------------------
# Teacher call (no tools parameter - prompt-based).
# ---------------------------------------------------------------------------


def _call_teacher(
    *,
    ak: str,
    model_name: str,
    endpoint: str,
    api_version: str,
    messages: list[dict[str, Any]],
    max_completion_tokens: int,
    reasoning_effort: str,
    timeout_s: float,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Call Azure OpenAI WITHOUT tools param. Returns the raw response dict or None."""
    client = _get_or_create_client(ak, endpoint, api_version, timeout_s)
    params: dict[str, Any] = dict(
        stream=False,
        model=model_name,
        reasoning_effort=reasoning_effort,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    # NO tools parameter — teacher must output XML tags in plain text.

    for attempt in range(max_attempts):
        try:
            resp = client.chat.completions.create(**params).model_dump()
            return resp
        except Exception as exc:  # noqa: BLE001
            backoff = min(10.0, (2**attempt) + random.random())
            logger.warning(
                "teacher call failed attempt=%s/%s err=%s; sleep %.1fs", attempt + 1, max_attempts, exc, backoff
            )
            time.sleep(backoff)
    return None


# ---------------------------------------------------------------------------
# Parse <call_tool> from teacher text output.
# ---------------------------------------------------------------------------

_CALL_TOOL_RE = re.compile(
    r"<call_tool\s+([^>]*name\s*=\s*\"([^\"]+)\"[^>]*)>(.*?)</call_tool>", re.DOTALL | re.IGNORECASE
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# For Fetch: extract description attr
_DESC_RE = re.compile(r'description\s*=\s*"([^"]*)"', re.IGNORECASE)


def _parse_call_tool(text: str) -> list[dict[str, Any]]:
    """Parse all <call_tool> invocations from teacher output text.

    Returns list of {name, query, description (optional)}.
    """
    results = []
    for m in _CALL_TOOL_RE.finditer(text):
        attrs_str = m.group(1)
        tool_name = m.group(2)
        inner_text = m.group(3).strip()

        desc_m = _DESC_RE.search(attrs_str)
        desc = desc_m.group(1) if desc_m else ""

        results.append({"name": tool_name, "query": inner_text, "description": desc})
    return results


# ---------------------------------------------------------------------------
# Tool dispatch.
# ---------------------------------------------------------------------------


def _run_tool(
    name: str, query: str, description: str, *, page_idx: int, last_time: str | None, curr_title: str | None
) -> tuple[str, int]:
    """Execute one tool call; returns (content_text, new_page_idx)."""
    search, linkreader, scholar = _get_tools()
    # Normalize tool names
    normalized = name.lower().replace("_", "").replace("-", "")

    try:
        if normalized in ("fetch",):
            url = query.strip()
            results = linkreader(fetch_request_list=[{"url": url, "snippet": {"query": description}}], page_idx=page_idx)
        elif normalized in ("generalsearch", "searchquery"):
            results = search(
                is_cot=True,
                search_request_list=[{"query": query}],
                page_idx=page_idx,
                last_time=last_time,
                curr_title=curr_title,
            )
        elif normalized in ("scholarsearch", "scholarsearch"):
            last_time_date = last_time.split("T")[0] if last_time else None
            results = scholar(
                search_request_list=[{"query": query}], page_idx=page_idx, last_time=last_time_date, curr_title=curr_title
            )
        else:
            return f"Unknown tool name: {name}", page_idx

        content = results.get("content", "") or ""
        new_page_idx = results.get("page_idx", page_idx)
        if not content:
            content = f"{name} returns empty content"
        return content, new_page_idx
    except Exception as exc:  # noqa: BLE001
        logger.debug("tool %s failed: %s", name, exc)
        return f"Tool error: {exc}; reflect and try again", page_idx + 1


# ---------------------------------------------------------------------------
# Prompt-based agent loop.
# ---------------------------------------------------------------------------


def _pick_ak_model(ak_list: list[str], model_list: list[str]) -> tuple[str, str]:
    if not ak_list or not model_list:
        raise ValueError("--ak-list and --model-list must both be non-empty.")
    n = min(len(ak_list), len(model_list))
    idx = random.randint(0, n - 1)
    return ak_list[idx], model_list[idx]


def run_teacher_trace(
    *,
    user_prompt: str,
    system_prompt: str,
    ak_list: list[str],
    model_list: list[str],
    endpoint: str,
    api_version: str,
    max_completion_tokens: int,
    reasoning_effort: str,
    timeout_s: float,
    max_attempts: int,
    max_steps: int,
    last_time: str | None,
    curr_title: str | None,
    max_tool_response_chars: int = 8192,
) -> dict[str, Any]:
    """Run prompt-based teacher until it produces <answer> or hits max_steps.

    Unlike v1, the teacher outputs <think>/<call_tool>/<answer> in plain text.
    The script parses tool calls, executes them, and feeds results back.

    Returns dict with messages (already in RL XML format), final_text, stats.
    """
    ak, model_name = _pick_ak_model(ak_list, model_list)

    # Build the conversation in the target SFT format directly.
    # messages = the actual API call messages (system + user + assistant + user(tool_response) ...)
    api_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # The output SFT messages (what we'll save)
    sft_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    step = 0
    tool_call_num = 0
    page_idx = 0
    final_text = ""
    empty_retry = 0

    while step < max_steps:
        resp = _call_teacher(
            ak=ak,
            model_name=model_name,
            endpoint=endpoint,
            api_version=api_version,
            messages=api_messages,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
        )
        if resp is None:
            step += 1
            continue

        choice = resp["choices"][0]
        content = (choice["message"].get("content") or "").strip()
        finish_reason = choice.get("finish_reason")

        if not content:
            if finish_reason == "length":
                logger.warning(
                    "teacher ran out of budget (finish_reason=length, step=%s); "
                    "increase --max-completion-tokens or lower --reasoning-effort.",
                    step,
                )
                break
            empty_retry += 1
            if empty_retry > 3:
                break
            step += 1
            continue

        empty_retry = 0

        # Check if teacher produced <answer> -> done
        if _ANSWER_RE.search(content):
            # This is the final turn
            api_messages.append({"role": "assistant", "content": content})
            sft_messages.append({"role": "assistant", "content": content})
            final_text = content
            break

        # Check for <call_tool> invocations
        tool_calls = _parse_call_tool(content)
        if tool_calls:
            # Save assistant message as-is (contains <think> + <call_tool>)
            api_messages.append({"role": "assistant", "content": content})
            sft_messages.append({"role": "assistant", "content": content})

            # Execute each tool call and collect results
            tool_response_parts: list[str] = []
            for tc in tool_calls:
                tool_content, page_idx = _run_tool(
                    tc["name"], tc["query"], tc["description"],
                    page_idx=page_idx, last_time=last_time, curr_title=curr_title,
                )
                tool_call_num += 1

                # Build the search-like header for tool response
                search_query = tc["query"]
                tool_response_parts.append(
                    f'<search query="{search_query}" search_time="{last_time or ""}">'
                    f"{tool_content}</search>"
                    if "search" in tc["name"].lower()
                    else tool_content
                )
                logger.debug("step=%s tool=%s query=%s", step, tc["name"], tc["query"][:60])

            # Merge and truncate tool responses
            merged_response = "\n".join(tool_response_parts)
            if max_tool_response_chars > 0 and len(merged_response) > max_tool_response_chars:
                merged_response = merged_response[:max_tool_response_chars]

            # Feed tool response back as user message
            tool_response_msg = f"<tool_response>\n{merged_response}\n</tool_response>"
            api_messages.append({"role": "user", "content": tool_response_msg})
            sft_messages.append({"role": "user", "content": tool_response_msg})

            step += 1
            continue

        # Teacher output has neither <answer> nor <call_tool>
        # Treat as think-only; append and let teacher continue
        api_messages.append({"role": "assistant", "content": content})
        sft_messages.append({"role": "assistant", "content": content})
        # Nudge teacher to take action
        nudge = (
            "Please proceed: output a <think>...</think> block followed by either "
            "a <call_tool>...</call_tool> to search, or <answer>...</answer> if you "
            "have sufficient evidence."
        )
        api_messages.append({"role": "user", "content": nudge})
        sft_messages.append({"role": "user", "content": nudge})
        step += 1

    # If we haven't gotten a final answer, force one
    if not final_text and step >= max_steps:
        force_msg = (
            "Step limit reached. Based on gathered info, output a final "
            "<think>...</think> with a sufficiency reflection, then "
            "<answer>...</answer> now. Do not call tools again."
        )
        api_messages.append({"role": "user", "content": force_msg})
        sft_messages.append({"role": "user", "content": force_msg})

        forced = _call_teacher(
            ak=ak,
            model_name=model_name,
            endpoint=endpoint,
            api_version=api_version,
            messages=api_messages,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            max_attempts=max(4, max_attempts // 2),
        )
        if forced is not None:
            last_content = (forced["choices"][0]["message"].get("content") or "").strip()
            if last_content:
                api_messages.append({"role": "assistant", "content": last_content})
                sft_messages.append({"role": "assistant", "content": last_content})
                final_text = last_content

    return {
        "sft_messages": sft_messages,
        "api_messages": api_messages,
        "final_text": final_text,
        "tool_call_num": tool_call_num,
        "steps": step,
        "ak_tail": ak[-4:],
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# Input loading (reused from v1).
# ---------------------------------------------------------------------------

_VALID_DATA_TYPES = {"search", "understanding", "write_section", "write_survey"}


def _extract_query(row: dict[str, Any]) -> tuple[str, str]:
    raw_id = row.get("id") or row.get("example_id") or row.get("uid")
    example_id = str(raw_id) if raw_id is not None else ""

    if isinstance(row.get("messages"), list):
        for msg in row["messages"]:
            if msg.get("role") == "user" and msg.get("content"):
                query = str(msg["content"]).strip()
                if not example_id:
                    example_id = hashlib.md5(query.encode("utf-8")).hexdigest()
                return example_id, query

    for key in ("question", "problem", "query", "prompt"):
        if row.get(key):
            query = str(row[key]).strip()
            if not example_id:
                example_id = hashlib.md5(query.encode("utf-8")).hexdigest()
            return example_id, query

    raise ValueError(f"Could not locate user query in row: keys={list(row.keys())}")


def _iter_jsonl_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skip malformed line in %s: %s", path, e)


def _iter_parquet_rows(path: Path):
    table = pq.read_table(path)
    yield from table.to_pylist()


def _iter_input_rows(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from _iter_parquet_rows(path)
    elif suffix in (".jsonl", ".json"):
        yield from _iter_jsonl_rows(path)
    else:
        raise ValueError(f"Unsupported input extension {suffix!r} for {path}. Use .jsonl or .parquet.")


def _extract_ground_truth(row: dict[str, Any]) -> str:
    gt = row.get("ground_truth", "")
    if isinstance(gt, (dict, list)):
        return json.dumps(gt, ensure_ascii=False)
    return str(gt or "")


def _extract_dataset_label(row: dict[str, Any], fallback: str) -> str:
    return str(row.get("dataset") or fallback)


def load_queries(inputs: list[Path], num_examples: int | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
        for row in _iter_input_rows(path):
            try:
                eid, q = _extract_query(row)
            except ValueError as e:
                logger.warning("Skip row in %s: %s", path, e)
                continue
            out.append(
                {
                    "example_id": eid,
                    "query": q,
                    "ground_truth": _extract_ground_truth(row),
                    "dataset": _extract_dataset_label(row, fallback=path.stem),
                }
            )
            if num_examples is not None and len(out) >= num_examples:
                return out
    return out


def load_done_ids(output_jsonl: Path) -> set[str]:
    if not output_jsonl.exists():
        return set()
    done: set[str] = set()
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            eid = obj.get("example_id")
            if eid is not None:
                done.add(str(eid))
    return done


# ---------------------------------------------------------------------------
# Filtering and scoring.
# ---------------------------------------------------------------------------


def _parse_ground_truth(gt: str) -> dict[str, Any] | None:
    if not gt:
        return None
    try:
        obj = json.loads(gt)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if "data_type" not in obj or "verifiable_meta" not in obj:
        return None
    return obj


def _build_tool_results_str(sft_messages: list[dict[str, Any]]) -> str:
    """Extract all tool_response content for scholar_evaluator."""
    parts: list[str] = []
    for msg in sft_messages:
        if msg.get("role") == "user":
            c = msg.get("content", "")
            if "<tool_response>" in c:
                parts.append(c)
    return "\n".join(parts)


def _extract_candidate_text(sft_messages: list[dict[str, Any]]) -> str:
    """All assistant turn content concatenated."""
    out: list[str] = []
    for m in sft_messages:
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
    return "\n".join(out)


def parse_per_type_thresholds(spec: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Bad --min-reward-per-type entry {chunk!r}; expected 'name=value'.")
        name, val = chunk.split("=", 1)
        name = name.strip()
        if name not in _VALID_DATA_TYPES:
            raise ValueError(f"Unknown data_type {name!r}; valid: {sorted(_VALID_DATA_TYPES)}.")
        try:
            out[name] = float(val.strip())
        except ValueError as e:
            raise ValueError(f"Bad float in --min-reward-per-type entry {chunk!r}: {e}") from e
    return out


def resolve_min_reward(data_type: str | None, per_type: dict[str, float], global_min: float | None) -> float | None:
    if data_type and data_type in per_type:
        return per_type[data_type]
    return global_min


def score_with_scholar_evaluator(
    sft_messages: list[dict[str, Any]], ground_truth_str: str, timeout_s: float
) -> dict[str, Any] | None:
    gt = _parse_ground_truth(ground_truth_str)
    if gt is None:
        logger.debug("skip scoring: ground_truth is not a parseable scholar spec")
        return None

    candidate = _extract_candidate_text(sft_messages)
    tool_results_str = _build_tool_results_str(sft_messages)

    async def _run() -> dict[str, Any]:
        res_tuple = await scholar_evaluator.evaluate_scholar_score_verifier(candidate, gt, tool_results_str)
        if isinstance(res_tuple, tuple) and len(res_tuple) == 2:
            rd, _ok = res_tuple
            return rd if isinstance(rd, dict) else {}
        return {}

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout=timeout_s))
    except Exception as exc:  # noqa: BLE001
        logger.warning("scholar_evaluator scoring failed: %s", exc)
        logger.debug("%s", traceback.format_exc())
        return None


def passes_filters(
    *,
    num_ok_tool_calls: int,
    has_answer: bool,
    min_tool_calls: int,
    require_answer: bool,
    reward: float | None,
    min_reward: float | None,
) -> tuple[bool, str]:
    if num_ok_tool_calls < min_tool_calls:
        return False, f"tool_calls<{min_tool_calls}"
    if require_answer and not has_answer:
        return False, "no_answer"
    if min_reward is not None:
        if reward is None:
            return False, "reward_unavailable"
        if reward < min_reward:
            return False, f"reward<{min_reward:.3f}"
    return True, ""


def _extract_sample_tool_context(
    ground_truth_str: str, default_last_time: str | None
) -> tuple[str | None, str | None]:
    """Extract last_time / curr_title from sample's ground_truth JSON."""
    gt = _parse_ground_truth(ground_truth_str)
    if gt is None:
        return default_last_time, None
    vm = gt.get("verifiable_meta") or {}
    if not isinstance(vm, dict):
        return default_last_time, None

    published = vm.get("published_date")
    last_time = str(published).strip() if published else None
    if not last_time:
        last_time = default_last_time

    paper_title = vm.get("paper_title")
    curr_title = str(paper_title).strip() if paper_title else None
    if curr_title == "":
        curr_title = None

    return last_time, curr_title


# ---------------------------------------------------------------------------
# Format validation (same checks as RL).
# ---------------------------------------------------------------------------

_FORMAT_CALL_TOOL_RE = re.compile(
    r"<call_tool\s+([^>]*name\s*=\s*\"[^\"]+\"[^>]*)>(.*?)</call_tool>", re.DOTALL | re.IGNORECASE
)
_CITE_CONTENT_RE = re.compile(r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.+?)</cite>", re.DOTALL | re.IGNORECASE)
_CITE_RE = re.compile(r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.*?)</cite>", re.DOTALL | re.IGNORECASE)


def _protocol_format_check(sft_messages: list[dict[str, Any]]) -> tuple[float, dict[str, bool]]:
    """Same 5-check format validation as RL's _protocol_format_reward."""
    text = "\n".join(m.get("content", "") for m in sft_messages if m.get("role") == "assistant")

    checks: dict[str, bool] = {}
    checks["has_call_tool"] = bool(_FORMAT_CALL_TOOL_RE.search(text))
    answer_matches = _ANSWER_RE.findall(text)
    checks["has_single_answer"] = len(answer_matches) == 1

    if answer_matches:
        answer_body = answer_matches[-1]
        checks["has_cite_inside_answer"] = bool(_CITE_CONTENT_RE.search(answer_body) or _CITE_RE.search(answer_body))
    else:
        checks["has_cite_inside_answer"] = False

    call_balanced = text.count("<call_tool") == text.count("</call_tool>")
    answer_balanced = text.count("<answer>") == text.count("</answer>")
    think_balanced = text.count("<think>") == text.count("</think>")
    truncated = not text.rstrip().endswith("</answer>") and text.count("<think>") > text.count("</think>")
    checks["tags_balanced"] = call_balanced and answer_balanced and (think_balanced or truncated)

    last_think_close = text.rfind("</think>")
    last_answer_open = text.rfind("<answer>")
    checks["answer_not_nested"] = (
        last_answer_open > last_think_close if last_think_close >= 0 else True
    )

    score = sum(1 for v in checks.values() if v) / max(1, len(checks))
    return score, checks


# ---------------------------------------------------------------------------
# Single example distillation worker.
# ---------------------------------------------------------------------------


def distill_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    system_prompt = build_system_prompt(args.tool_backend)

    sample_last_time, sample_curr_title = _extract_sample_tool_context(
        item.get("ground_truth") or "", default_last_time=args.last_time
    )

    try:
        result = run_teacher_trace(
            user_prompt=item["query"],
            system_prompt=system_prompt,
            ak_list=args.ak_list_parsed,
            model_list=args.model_list_parsed,
            endpoint=args.azure_endpoint,
            api_version=args.azure_api_version,
            max_completion_tokens=args.max_completion_tokens,
            reasoning_effort=args.reasoning_effort,
            timeout_s=args.timeout_s,
            max_attempts=args.max_retries,
            max_steps=args.max_steps,
            last_time=sample_last_time,
            curr_title=sample_curr_title,
            max_tool_response_chars=args.max_tool_response_chars,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("teacher trace failed example=%s: %s", item["example_id"], exc)
        logger.debug("%s", traceback.format_exc())
        return None

    sft_messages = result["sft_messages"]
    final_text = result["final_text"] or ""
    has_answer = bool(_ANSWER_RE.search(final_text))
    num_tool_calls = result["tool_call_num"]

    # Format validation
    format_score, format_checks = _protocol_format_check(sft_messages)
    if format_score < 1.0:
        logger.info(
            "format check failed example=%s score=%.2f checks=%s",
            item["example_id"], format_score, format_checks,
        )
        if not args.keep_format_failures:
            return None

    # Reward scoring
    gt_obj = _parse_ground_truth(item.get("ground_truth") or "")
    data_type = gt_obj.get("data_type") if gt_obj else None

    effective_min_reward = resolve_min_reward(
        data_type=data_type, per_type=args.min_reward_per_type_parsed, global_min=args.min_reward
    )

    reward_result: dict[str, Any] | None = None
    reward_score: float | None = None
    need_score = effective_min_reward is not None or args.always_score
    if need_score:
        reward_result = score_with_scholar_evaluator(
            sft_messages=sft_messages,
            ground_truth_str=item.get("ground_truth") or "",
            timeout_s=args.score_timeout_s,
        )
        if reward_result is not None:
            try:
                reward_score = float(reward_result.get("score"))
            except (TypeError, ValueError):
                reward_score = None

    keep, reason = passes_filters(
        num_ok_tool_calls=num_tool_calls,
        has_answer=has_answer,
        min_tool_calls=args.min_tool_calls,
        require_answer=args.require_answer,
        reward=reward_score,
        min_reward=effective_min_reward,
    )
    if not keep:
        logger.info(
            "drop example=%s data_type=%s reason=%s tool_calls=%s steps=%s reward=%s",
            item["example_id"],
            data_type or "unknown",
            reason,
            num_tool_calls,
            result["steps"],
            f"{reward_score:.4f}" if reward_score is not None else "N/A",
        )
        return None

    meta: dict[str, Any] = {
        "num_tool_calls": num_tool_calls,
        "steps": result["steps"],
        "teacher_model": result["model_name"],
        "ak_tail": result["ak_tail"],
        "tool_backend": args.tool_backend,
        "distill_version": "v2_prompt_based",
        "format_score": format_score,
    }
    if data_type is not None:
        meta["data_type"] = data_type
    if reward_score is not None:
        meta["reward"] = reward_score
    if effective_min_reward is not None:
        meta["reward_threshold"] = effective_min_reward
    if reward_result is not None:
        meta["reward_breakdown"] = {
            k: v
            for k, v in reward_result.items()
            if k not in {"candidate", "oracle_answer", "judge_response", "format_reward_checks"}
        }

    return {
        "example_id": item["example_id"],
        "messages": sft_messages,
        "dataset": item["dataset"] or "scholar_distill",
        "ground_truth": item["ground_truth"],
        "_meta": meta,
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill multi-turn tool-use SFT trajectories (prompt-based, v2)."
    )
    parser.add_argument(
        "--input", type=Path, nargs="+", required=True, help="One or more input files (.jsonl or .parquet)."
    )
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--dataset-label", default="scholar_distill")

    # Teacher (Azure OpenAI)
    parser.add_argument("--ak-list", required=True, help="Comma-separated list of Azure API keys.")
    parser.add_argument(
        "--model-list",
        default="gpt-5-2025-08-07",
        help="Comma-separated list of model names (aligned with --ak-list).",
    )
    parser.add_argument("--azure-endpoint", default=DEFAULT_AZURE_ENDPOINT)
    parser.add_argument("--azure-api-version", default=DEFAULT_AZURE_API_VERSION)
    parser.add_argument(
        "--max-completion-tokens", type=int, default=16000,
        help="Passed to Azure as max_completion_tokens. Higher than v1 since "
        "teacher now outputs reasoning in content (not hidden reasoning tokens).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="GPT reasoning_effort parameter. 'medium' recommended for prompt-based "
        "since reasoning is now explicit in content.",
    )
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=8)

    # Agent loop
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument(
        "--max-tool-response-chars", type=int, default=8192,
        help="Max chars per tool response (consistent with RL training).",
    )
    parser.add_argument(
        "--tool-backend",
        choices=["mock", "internal", "real"],
        default="internal",
        help="Which tool backend to use. 'real' aliases 'internal'.",
    )
    parser.add_argument(
        "--last-time", default="2026-05-01T00:00:00", help="ISO timestamp passed to search tools."
    )

    # Filtering
    parser.add_argument("--min-tool-calls", type=int, default=2)
    parser.add_argument("--require-answer", action="store_true", default=True)
    parser.add_argument("--no-require-answer", dest="require_answer", action="store_false")
    parser.add_argument("--keep-format-failures", action="store_true", default=False,
                        help="Keep samples that fail format checks (for debugging).")
    parser.add_argument(
        "--min-reward", type=float, default=None,
        help="Global reward threshold.",
    )
    parser.add_argument(
        "--min-reward-per-type", default=None,
        help="Per-data_type reward thresholds, e.g. 'search=0.3,understanding=0.5'.",
    )
    parser.add_argument("--always-score", action="store_true")
    parser.add_argument("--score-timeout-s", type=float, default=180.0)

    # Concurrency / resume
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.ak_list_parsed = _parse_csv_list(args.ak_list)
    args.model_list_parsed = _parse_csv_list(args.model_list)
    if not args.ak_list_parsed:
        raise ValueError("--ak-list must contain at least one AK.")
    if not args.model_list_parsed:
        raise ValueError("--model-list must contain at least one model name.")

    args.min_reward_per_type_parsed = parse_per_type_thresholds(args.min_reward_per_type)
    if args.min_reward_per_type_parsed:
        logger.info("Per-data_type reward thresholds: %s", args.min_reward_per_type_parsed)
    if args.min_reward is not None:
        logger.info("Global reward fallback threshold: %.4f", args.min_reward)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and args.output_jsonl.exists():
        args.output_jsonl.unlink()

    items = load_queries(args.input, args.num_examples)
    logger.info("Loaded %s queries from %s file(s)", len(items), len(args.input))

    done = load_done_ids(args.output_jsonl)
    if done:
        logger.info("Resuming: %s example_ids already in %s", len(done), args.output_jsonl)
    todo = [it for it in items if it["example_id"] not in done]
    logger.info("To distill: %s queries", len(todo))

    kept = 0
    dropped = 0
    started = time.time()

    with (
        args.output_jsonl.open("a", encoding="utf-8") as fout,
        ThreadPoolExecutor(max_workers=args.max_concurrency) as pool,
    ):
        futures = {pool.submit(distill_one, it, args): it["example_id"] for it in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            rec = fut.result()
            if rec is None:
                dropped += 1
            else:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                kept += 1
            if i % 10 == 0 or i == len(futures):
                elapsed = time.time() - started
                logger.info("progress=%s/%s kept=%s dropped=%s elapsed=%.1fs", i, len(futures), kept, dropped, elapsed)

    logger.info("Done. kept=%s dropped=%s output=%s", kept, dropped, args.output_jsonl)


if __name__ == "__main__":
    main()
