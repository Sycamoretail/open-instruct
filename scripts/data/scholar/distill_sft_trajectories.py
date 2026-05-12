"""Distill multi-turn tool-use SFT trajectories from a GPT-5.x teacher.

Pipeline
========

* Teacher side: GPT-5.x via Azure OpenAI (ByteDance gpt_openapi), using the
  **native function-calling** protocol (``tools=[...]`` schemas). Multiple AKs
  and model names can be supplied; we pick one at random per example.
* Tools: ``ScholarSearchTool`` / ``GeneralSearchTool`` / ``Fetch`` from
  ``tools/`` (same tool backend the RL stage uses).
* Output: each trajectory is **translated into the XML tool-use protocol**
  used by our RL training pipeline and written as

      {"messages": [...], "dataset": str, "ground_truth": str, "_meta": {...}}

  so ``open_instruct/finetune.py`` (with ``sft_tulu_tokenize_and_truncate_v1``)
  can consume it directly. Non-assistant turns are auto-masked to -100.

Example::

    python scripts/data/scholar/distill_sft_trajectories.py \\
        --input data/scholar_en_rlvr/search_data.2023.parquet \\
                data/scholar_en_rlvr/understanding_data.2023.parquet \\
        --output-jsonl data/scholar/sft_distill.jsonl \\
        --ak-list "$AK1,$AK2" \\
        --model-list "gpt-5-2025-08-07,gpt-5-2025-08-07" \\
        --max-steps 8 \\
        --max-completion-tokens 8000 \\
        --num-examples 50 \\
        --min-tool-calls 2 \\
        --require-answer
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
# Azure OpenAI (ByteDance gpt_openapi) config.
# ---------------------------------------------------------------------------

DEFAULT_AZURE_ENDPOINT = "https://search.bytedance.net/gpt/openapi/online/v2/crawl/openai/deployments/gpt_openapi"
DEFAULT_AZURE_API_VERSION = "2024-03-01-preview"

_CLIENT_CACHE: dict[str, openai.AzureOpenAI] = {}
_CLIENT_CACHE_LOCK = Lock()


def _get_or_create_client(ak: str, endpoint: str, api_version: str, timeout_s: float) -> openai.AzureOpenAI:
    """按 AK 在进程内复用一个 AzureOpenAI client。"""
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
# 工具单例（与 RL 侧共享实例，token 预算和限流一致）。
# ---------------------------------------------------------------------------

_TOOLS_LOCK = Lock()
_TOOLS: dict[str, Any] = {}


def _get_tools() -> tuple[Any, Any, Any]:
    """返回 ``(search, linkreader, scholar)``，懒加载。"""
    if _TOOLS:
        return _TOOLS["search"], _TOOLS["linkreader"], _TOOLS["scholar"]
    with _TOOLS_LOCK:
        if not _TOOLS:
            _TOOLS["search"] = GeneralSearchTool(return_prompt_type="xml")
            _TOOLS["linkreader"] = Fetch(return_prompt_type="xml")
            _TOOLS["scholar"] = ScholarSearchTool(return_prompt_type="xml")
    return _TOOLS["search"], _TOOLS["linkreader"], _TOOLS["scholar"]


# ---------------------------------------------------------------------------
# Teacher 侧的 function-calling schema。
# 工具名有意与 RL 侧 XML 工具名一一对应：
#   scholar_search  -> ScholarSearch
#   search_query    -> GeneralSearch
#   fetch           -> Fetch
# ---------------------------------------------------------------------------

TEACHER_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_query",
            "description": ("A web search tool that returns several related pages based on a search query."),
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "short search query <=5 keywords"}},
                "required": ["q"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch",
            "description": (
                "Read a given URL and return summary based on query. "
                "Use full URLs (http...). For relative links, expand with base first."
            ),
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "query": {"type": "string"}},
                "required": ["url", "query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scholar_search",
            "description": ("An academic search tool that returns several related pages based on a search query."),
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string", "description": "academic search query"}},
                "required": ["q"],
                "additionalProperties": False,
            },
        },
    },
]

TEACHER_TOOL_TO_XML_NAME: dict[str, str] = {
    "scholar_search": "ScholarSearch",
    "search_query": "GeneralSearch",
    "fetch": "Fetch",
}


# ---------------------------------------------------------------------------
# Teacher 调用（带指数退避）。
# ---------------------------------------------------------------------------


def _call_teacher(
    *,
    ak: str,
    model_name: str,
    endpoint: str,
    api_version: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    max_completion_tokens: int,
    reasoning_effort: str,
    timeout_s: float,
    max_attempts: int,
) -> dict[str, Any] | None:
    """Call Azure OpenAI with exponential backoff. Returns the raw response dict or None."""
    client = _get_or_create_client(ak, endpoint, api_version, timeout_s)
    params: dict[str, Any] = dict(
        stream=False,
        model=model_name,
        reasoning_effort=reasoning_effort,
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    if tools:
        params["tools"] = tools

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
# Tool dispatch.
# ---------------------------------------------------------------------------


def _run_tool(
    name: str, args: dict[str, Any], *, page_idx: int, last_time: str | None, curr_title: str | None
) -> tuple[str, int]:
    """执行一次工具调用；返回 ``(content_text, new_page_idx)``。"""
    search, linkreader, scholar = _get_tools()
    try:
        if name == "fetch":
            url = str(args.get("url", "")).strip()
            query = str(args.get("query", "")).strip()
            results = linkreader(fetch_request_list=[{"url": url, "snippet": {"query": query}}], page_idx=page_idx)
        elif name == "search_query":
            q = str(args.get("q", "")).strip()
            results = search(
                is_cot=True,
                search_request_list=[{"query": q}],
                page_idx=page_idx,
                last_time=last_time,
                curr_title=curr_title,
            )
        elif name == "scholar_search":
            q = str(args.get("q", "")).strip()
            last_time_date = last_time.split("T")[0] if last_time else None
            results = scholar(
                search_request_list=[{"query": q}], page_idx=page_idx, last_time=last_time_date, curr_title=curr_title
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
# Agent loop（teacher 用 function-calling，顺带记录整条 trace）。
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
) -> dict[str, Any]:
    """跑 teacher 直到它给出最终 content，或触达 max_steps。

    返回一个 dict，包含原始 OpenAI trace 和步数统计。
    """
    ak, model_name = _pick_ak_model(ak_list, model_list)

    messages: list[dict[str, Any]] = [
        {"role": "developer", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    trace: list[dict[str, Any]] = list(messages)

    step = 0
    tool_call_num = 0
    empty_retry = 0
    page_idx = 0
    continue_decoding = True
    final_text = ""

    while continue_decoding and step < max_steps:
        resp = _call_teacher(
            ak=ak,
            model_name=model_name,
            endpoint=endpoint,
            api_version=api_version,
            messages=trace,
            tools=TEACHER_TOOL_SCHEMAS,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
        )
        if resp is None:
            step += 1
            continue

        choice = resp["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason")
        trace.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            for tc in tool_calls:
                fn = tc["function"]
                tool_name = fn["name"]
                raw_args = fn.get("arguments", "")
                if isinstance(raw_args, str):
                    try:
                        tool_args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        tool_args = {}
                else:
                    tool_args = raw_args or {}

                content, page_idx = _run_tool(
                    tool_name, tool_args, page_idx=page_idx, last_time=last_time, curr_title=curr_title
                )
                trace.append({"tool_call_id": tc["id"], "role": "tool", "name": tool_name, "content": content})
                tool_call_num += 1
                logger.debug("step=%s tool=%s args=%s", step, tool_name, tool_args)
            step += 1
            continue

        # 没有 tool_calls：要么是最终答案，要么是空。
        content = (msg.get("content") or "").strip()
        if not content:
            # GPT-5 的 reasoning tokens 也占 max_completion_tokens；如果 finish_reason
            # 是 length，说明预算被 reasoning 吃光，retry 也无用，直接放弃这条。
            if finish_reason == "length":
                logger.warning(
                    "teacher ran out of budget (finish_reason=length, step=%s); "
                    "increase --max-completion-tokens or lower --reasoning-effort.",
                    step,
                )
                trace.pop()
                break
            # 否则（真实空回复），少量重试。
            trace.pop()
            empty_retry += 1
            if empty_retry <= 3:
                step += 1
                continue

        final_text = content
        continue_decoding = False
        step += 1

    # 仍未给出 final_text -> 强制收束一次（不带 tools）。
    if continue_decoding and step >= max_steps:
        force_msg = {
            "role": "user",
            "content": (
                "Step limit reached. Based on gathered info, output only the final "
                "<answer>...</answer> now. Do not call tools again."
            ),
        }
        forced_trace = trace + [force_msg]
        forced = _call_teacher(
            ak=ak,
            model_name=model_name,
            endpoint=endpoint,
            api_version=api_version,
            messages=forced_trace,
            tools=None,
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            timeout_s=timeout_s,
            max_attempts=max(4, max_attempts // 2),
        )
        if forced is not None:
            last = forced["choices"][0]["message"]
            trace.append(force_msg)
            trace.append(last)
            final_text = (last.get("content") or "").strip()

    return {
        "trace": trace,
        "final_text": final_text,
        "tool_call_num": tool_call_num,
        "steps": step,
        "ak_tail": ak[-4:],
        "model_name": model_name,
    }


# ---------------------------------------------------------------------------
# 把 OpenAI function-calling trace 翻译成 RL 侧 XML messages。
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _tool_args_to_inner_text(tool_name: str, args: dict[str, Any]) -> tuple[str, str]:
    """返回 ``(call_tool 标签的 inner text, attr 后缀)``。

    - ``ScholarSearch`` / ``GeneralSearch``：把 query 放 inner text。
    - ``Fetch``：URL 放 inner text，query 作为 ``description`` 属性。
    """
    xml_name = TEACHER_TOOL_TO_XML_NAME.get(tool_name, tool_name)
    if xml_name == "Fetch":
        url = str(args.get("url", "")).strip()
        query = str(args.get("query", "")).strip()
        safe_desc = query.replace('"', "'").replace("\n", " ")
        return url, f' description="{safe_desc}"' if safe_desc else ""
    q = str(args.get("q", "")).strip()
    return q, ""


def _render_assistant_xml(assistant_msg: dict[str, Any]) -> str:
    """把一条 OpenAI assistant 消息（可能带 tool_calls）渲染成 RL 侧 XML。"""
    content = (assistant_msg.get("content") or "").strip()
    tool_calls = assistant_msg.get("tool_calls") or []

    # 优先复用 teacher 自己写的 <think>；没有就把 plain content 当 think body。
    think_text = ""
    if content:
        m = _THINK_RE.search(content)
        if m:
            think_text = m.group(1).strip()
            content_wo_think = _THINK_RE.sub("", content, count=1).strip()
        else:
            think_text = content
            content_wo_think = ""
    else:
        content_wo_think = ""

    parts: list[str] = []
    if think_text:
        parts.append(f"<think>\n{think_text}\n</think>")

    if tool_calls:
        for tc in tool_calls:
            fn = tc.get("function") or {}
            tool_name = fn.get("name", "")
            raw_args = fn.get("arguments", "")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args or {}
            inner, attrs = _tool_args_to_inner_text(tool_name, args)
            xml_name = TEACHER_TOOL_TO_XML_NAME.get(tool_name, tool_name)
            parts.append(f'<call_tool name="{xml_name}"{attrs}>{inner}</call_tool>')
    elif content_wo_think:
        m = _ANSWER_RE.search(content_wo_think)
        if m:
            parts.append(f"<answer>{m.group(1).strip()}</answer>")
        else:
            parts.append(f"<answer>{content_wo_think.strip()}</answer>")
    elif not think_text:
        parts.append("<think>\n(no content)\n</think>")

    return "\n".join(parts)


def translate_trace_to_xml_messages(trace: list[dict[str, Any]], scholar_system_prompt: str) -> list[dict[str, Any]]:
    """把 OpenAI tool-use trace 翻译成 RL 侧 ``messages``。

    * system：替换成 scholar RL system prompt。
    * user：保留（原始问题）。
    * assistant (带 tool_calls)：``<think>...</think>\\n<call_tool ...>...</call_tool>``。
    * tool：折叠进单个 user turn ``<tool_response>...\\n</tool_response>``。
    * assistant (最终)：``<think>...</think>\\n<answer>...</answer>``。
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": scholar_system_prompt}]

    pending_tool_responses: list[str] = []

    def _flush_tool_responses() -> None:
        if not pending_tool_responses:
            return
        merged = "\n".join(pending_tool_responses)
        out.append({"role": "user", "content": f"<tool_response>\n{merged}\n</tool_response>"})
        pending_tool_responses.clear()

    for msg in trace:
        role = msg.get("role")
        if role == "system" or role == "developer":
            continue
        if role == "user":
            _flush_tool_responses()
            out.append({"role": "user", "content": str(msg.get("content", "")).strip()})
            continue
        if role == "tool":
            pending_tool_responses.append(str(msg.get("content", "")).strip())
            continue
        if role == "assistant":
            _flush_tool_responses()
            xml = _render_assistant_xml(msg)
            out.append({"role": "assistant", "content": xml})
            continue
    _flush_tool_responses()
    return out


# ---------------------------------------------------------------------------
# 输入加载。
# ---------------------------------------------------------------------------


def _extract_query(row: dict[str, Any]) -> tuple[str, str]:
    """返回 ``(example_id, user_query)``。"""
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
    """返回 ``[{example_id, query, ground_truth, dataset}, ...]``。"""
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
# 过滤。
# ---------------------------------------------------------------------------


def _build_tool_results_str(trace: list[dict[str, Any]]) -> str:
    """把 OpenAI trace 里所有 role==tool 的 content 拼接起来。

    给 ``scholar_evaluator`` 用，它会从里头正则提取 ``<document ...>`` 片段。
    """
    parts: list[str] = []
    for msg in trace:
        if msg.get("role") == "tool":
            c = msg.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c)
    return "\n".join(parts)


def _extract_candidate_text(xml_messages: list[dict[str, Any]]) -> str:
    """候选回答文本 = 所有 assistant turn 的拼接（含 <think>/<call_tool>/<answer>/<cite>）。

    scholar_evaluator 内部会用 ``clean_rich_media_reference`` 去掉 ``<cite>`` 标签并
    抽 citation id，所以我们只要把完整 assistant 文本给它即可。
    """
    out: list[str] = []
    for m in xml_messages:
        if m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                out.append(c)
    return "\n".join(out)


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


_VALID_DATA_TYPES = {"search", "understanding", "write_section", "write_survey"}


def parse_per_type_thresholds(spec: str | None) -> dict[str, float]:
    """解析 ``--min-reward-per-type`` 参数。

    支持形如 ``"search=0.3,understanding=0.5,write_section=0.4,write_survey=0.4"``
    的字符串；允许只指定其中几个 data_type。
    """
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
    """优先用按 data_type 指定的阈值，没指定再回退到全局。"""
    if data_type and data_type in per_type:
        return per_type[data_type]
    return global_min


def score_with_scholar_evaluator(
    xml_messages: list[dict[str, Any]], trace: list[dict[str, Any]], ground_truth_str: str, timeout_s: float
) -> dict[str, Any] | None:
    """用 RL 侧的 scholar_evaluator 给一条 trajectory 打分。

    返回 evaluator 的 ``result_dict``（含 ``score`` 等），失败返回 ``None``。
    """
    gt = _parse_ground_truth(ground_truth_str)
    if gt is None:
        logger.debug("skip scoring: ground_truth is not a parseable scholar spec")
        return None

    candidate = _extract_candidate_text(xml_messages)
    tool_results_str = _build_tool_results_str(trace)

    async def _run() -> dict[str, Any]:
        res_tuple = await scholar_evaluator.evaluate_scholar_score_verifier(candidate, gt, tool_results_str)
        # evaluator 返回 (result_dict, ok_bool)。
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
    """从单条样本的 ground_truth JSON 里取 ``last_time`` / ``curr_title``。

    这两个字段是 ScholarSearch / GeneralSearch 的关键参数：
    * ``last_time``  <- ``verifiable_meta["published_date"]``（用于过滤时效）
    * ``curr_title`` <- ``verifiable_meta["paper_title"]``（用于避免检索到目标论文本身）

    如果样本里拿不到 published_date，就回退到 ``default_last_time``（CLI 全局值）。
    """
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
# 单条 example 的蒸馏 worker。
# ---------------------------------------------------------------------------


def distill_one(item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    system_prompt = build_system_prompt(args.tool_backend)

    # 从本条样本的 ground_truth 里取 per-sample 的 last_time / curr_title。
    # 这两个字段是 scholar/general search 的关键参数；写死会导致检索结果偏离真值。
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
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("teacher trace failed example=%s: %s", item["example_id"], exc)
        logger.debug("%s", traceback.format_exc())
        return None

    xml_messages = translate_trace_to_xml_messages(result["trace"], scholar_system_prompt=system_prompt)

    final_text = result["final_text"] or ""
    has_answer = bool(_ANSWER_RE.search(final_text)) or any(
        _ANSWER_RE.search(m.get("content", "") or "") for m in xml_messages if m.get("role") == "assistant"
    )
    num_tool_calls = result["tool_call_num"]

    # 先取 data_type，用于按任务选阈值。
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
            xml_messages=xml_messages,
            trace=result["trace"],
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
            "drop example=%s data_type=%s reason=%s tool_calls=%s steps=%s reward=%s threshold=%s",
            item["example_id"],
            data_type or "unknown",
            reason,
            num_tool_calls,
            result["steps"],
            f"{reward_score:.4f}" if reward_score is not None else "N/A",
            f"{effective_min_reward:.4f}" if effective_min_reward is not None else "None",
        )
        return None

    meta: dict[str, Any] = {
        "num_tool_calls": num_tool_calls,
        "steps": result["steps"],
        "teacher_model": result["model_name"],
        "ak_tail": result["ak_tail"],
        "tool_backend": args.tool_backend,
    }
    if data_type is not None:
        meta["data_type"] = data_type
    if reward_score is not None:
        meta["reward"] = reward_score
    if effective_min_reward is not None:
        meta["reward_threshold"] = effective_min_reward
    if reward_result is not None:
        # 记录子分数便于事后分析，去掉 candidate/judge_response 等大字段。
        meta["reward_breakdown"] = {
            k: v
            for k, v in reward_result.items()
            if k not in {"candidate", "oracle_answer", "judge_response", "format_reward_checks"}
        }

    return {
        "example_id": item["example_id"],
        "messages": xml_messages,
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
        description="Distill multi-turn tool-use SFT trajectories from a GPT-5.x teacher."
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
        "--max-completion-tokens", type=int, default=8000, help="Passed to Azure as max_completion_tokens."
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="high",
        help="GPT-5 reasoning_effort parameter.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=8)

    # Agent loop
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--tool-backend",
        choices=["mock", "internal", "real"],
        default="internal",
        help="Which RL-side tool backend to use when calling real tools. "
        "'real' aliases 'internal'. 'mock' is for offline debug.",
    )
    parser.add_argument(
        "--last-time", default="2024-01-01T00:00:00", help="ISO timestamp passed to search tools as 'last_time'."
    )

    # Filtering
    parser.add_argument("--min-tool-calls", type=int, default=2)
    parser.add_argument("--require-answer", action="store_true", default=True)
    parser.add_argument("--no-require-answer", dest="require_answer", action="store_false")
    parser.add_argument(
        "--min-reward",
        type=float,
        default=None,
        help="Global reward threshold: drop trajectories whose "
        "scholar_evaluator score < this value. Used as fallback when a "
        "data_type does not have a per-type threshold. Requires the parquet "
        "row to carry a valid scholar `ground_truth` JSON "
        "(data_type + verifiable_meta).",
    )
    parser.add_argument(
        "--min-reward-per-type",
        default=None,
        help="Per-data_type reward thresholds, e.g. "
        "'search=0.3,understanding=0.5,write_section=0.4,write_survey=0.4'. "
        "Takes precedence over --min-reward for the data_types listed. "
        "Valid keys: search / understanding / write_section / write_survey.",
    )
    parser.add_argument(
        "--always-score",
        action="store_true",
        help="Always compute the scholar_evaluator reward and record it in "
        "`_meta.reward`, even when no reward threshold is set (no filtering, "
        "just bookkeeping).",
    )
    parser.add_argument(
        "--score-timeout-s",
        type=float,
        default=180.0,
        help="Per-trajectory scoring timeout (evaluator may fan out several LLM judge calls).",
    )

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

    if args.tool_backend == "mock":
        logger.warning(
            "--tool-backend=mock selected: teacher will still be called with real "
            "function schemas but downstream tools will return fake content."
        )

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
