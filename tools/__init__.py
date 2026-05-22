"""Tool registry + internal/external tool adapters.

本模块承担了原 ``cullm/tools.py`` 的所有职责（ReAct XML 协议 + 工具注册表
+ fake 后端），同时新增了**公司内部工具适配层**，把
``cullm.tools.scholar_search_tool.ScholarSearchTool`` /
``cullm.tools.search_tool.GeneralSearchTool`` /
``cullm.tools.linkreader_tool.Fetch`` 这三个内部实现包装成和 fake 后端完全
兼容的 ``registry.execute`` 接口。

暴露的 backend：

* ``"mock"``      -- 纯离线的 fake scholar search（老流程）。
* ``"internal"``  -- 公司内部三件套（ScholarSearch / GeneralSearch / Fetch），
                     需要机器可以访问 ``gpt.bytedance.net`` 等内网 endpoint。
* ``"real"``      -- 预留；当前指向 internal。
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple
import inspect

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ReAct / XML protocol tokens and regexes.
# Keep these local so the tools package is self-contained and does not rely on
# a partially ported `tools.config`.
# ---------------------------------------------------------------------------

TOOL_CALL_OPEN = "<call_tool"
TOOL_CALL_CLOSE = "</call_tool>"
CALL_TOOL_OPEN_PREFIX = TOOL_CALL_OPEN
CALL_TOOL_CLOSE = TOOL_CALL_CLOSE

TOOL_RESP_OPEN = "<tool_response>"
TOOL_RESP_CLOSE = "</tool_response>"

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

CALL_TOOL_RE = re.compile(
    re.escape(TOOL_CALL_OPEN) + r"([^>]*)>(.*?)" + re.escape(TOOL_CALL_CLOSE),
    re.DOTALL,
)

CITE_RE = re.compile(
    r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.*?)</cite>",
    re.DOTALL,
)


def _shorten_text(text: str, limit: int = 240) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _runtime_rank() -> str:
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            return value
    return "na"

# Legacy alias: older callers (notebooks, tests) still import ``TOOL_CALL_RE``
# from ``cullm.tools``. We now point it at the new ``<call_tool>`` regex.
TOOL_CALL_RE = CALL_TOOL_RE

# ---------------------------------------------------------------------------
# <call_tool ...>query</call_tool> attribute parser.
#
# The policy is expected to emit calls in the form
#
#     <call_tool name="ScholarSearch" top_k="5">retrieval augmented generation</call_tool>
#
# Here we:
#
# * pull out the attribute block (everything between ``<call_tool`` and
#   the first ``>``) with a regex that tolerates whitespace / newlines;
# * parse attributes via :func:`_parse_call_tool_attrs`, which handles
#   both double- and single-quoted values;
# * treat the inner text as the primary query, merged under the key
#   ``query`` unless the tool's schema explicitly overrides it (Fetch
#   uses ``url`` instead -- we detect this in :meth:`ToolRegistry.execute`).
# ---------------------------------------------------------------------------

_ATTR_RE = re.compile(
    r"""([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""",
    re.DOTALL,
)


def _parse_call_tool_attrs(attr_block: str) -> Dict[str, str]:
    """Return ``{attr_name -> value}`` extracted from an attribute block."""
    attrs: Dict[str, str] = {}
    for m in _ATTR_RE.finditer(attr_block or ""):
        key = m.group(1)
        val = m.group(2) if m.group(2) is not None else (m.group(3) or "")
        attrs[key] = val
    return attrs


def _coerce_arg(value: str) -> Any:
    """Best-effort type coercion for attribute string values.

    Tools declare their arg types in their JSON-schema, but the wire format
    only carries strings. We do the obvious int/float/bool promotions so
    callers don't have to cast manually.
    """
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    # int
    try:
        if value.strip().lstrip("-").isdigit():
            return int(value)
    except Exception:
        pass
    # float
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
    except Exception:
        pass
    return value


def parse_call_tool_block(raw: str) -> Tuple[Optional[str], Dict[str, Any], str]:
    """Parse a raw ``<call_tool ...>query</call_tool>`` string.

    Returns ``(tool_name, kwargs, query_text)``. If the block is
    malformed, ``tool_name`` is ``None`` and the caller should treat the
    rollout step as a parse error.
    """
    match = CALL_TOOL_RE.search(raw)
    if match is None:
        return None, {}, ""
    attr_block, inner = match.group(1), match.group(2)
    attrs = _parse_call_tool_attrs(attr_block)

    # ``name`` selects the tool; everything else becomes a kwarg.
    tool_name = attrs.pop("name", None) or attrs.pop("tool", None)
    kwargs: Dict[str, Any] = {k: _coerce_arg(v) for k, v in attrs.items()}
    query_text = (inner or "").strip()
    return tool_name, kwargs, query_text


# ---------------------------------------------------------------------------
# Fake scholar search backend（保持与旧 tools.py 完全一致）。
# ---------------------------------------------------------------------------


def _deterministic_seed(query: str, salt: str = "") -> int:
    h = hashlib.sha1(f"{salt}::{query}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _fake_scholar_documents(query: str, top_k: int) -> List[Dict[str, str]]:
    seed = _deterministic_seed(query)
    docs: List[Dict[str, str]] = []
    for i in range(1, max(1, top_k) + 1):
        docs.append(
            {
                "reference_id": f"scholar:{i}",
                "title": f"[fake#{seed % 9999:04d}-{i}] A study on {query}",
                "snippet": (
                    f"This paper (stub id={i}) discusses aspects of '{query}'. "
                    "Replace this fake backend with the real scholar search "
                    "adapter to obtain grounded results."
                ),
            }
        )
    return docs


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_scholar_xml(query: str, docs: List[Dict[str, str]]) -> str:
    parts: List[str] = [f'<search query="{_xml_escape(query)}">']
    for d in docs:
        parts.append(f'  <document reference_id="{d["reference_id"]}">')
        parts.append(f"    <title>{_xml_escape(d['title'])}</title>")
        parts.append(f"    <snippet>{_xml_escape(d['snippet'])}</snippet>")
        parts.append("  </document>")
    parts.append("</search>")
    return "\n".join(parts)


def fake_scholar_search(query: str, top_k: int = 3) -> str:
    top_k = max(1, min(int(top_k or 3), 10))
    docs = _fake_scholar_documents(query, top_k)
    return _render_scholar_xml(query, docs)


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


class ToolRegistry:
    """``name -> (schema, fn)`` 映射，被 trainer 直接消费。"""

    def __init__(self) -> None:
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        fn: Callable[..., str],
        query_key: str = "query",
    ) -> None:
        """Register a tool.

        Args:
            name: the string the policy puts in ``<call_tool name="...">``.
            description: one-paragraph summary rendered into the prompt.
            parameters: JSON-schema-like dict describing accepted attributes.
                The ``query_key`` entry (default ``"query"``) is the one
                that will be fed from the inner text of ``<call_tool>``.
            fn: the Python callable. Keyword-only arguments are matched by
                attribute name; the primary input arrives via ``query_key``.
            query_key: which kwarg receives the inner text. For Fetch this
                is ``"url"``; for the search tools it is ``"query"``.
        """
        self._tools[name] = {
            "description": description,
            "parameters": parameters,
            "fn": fn,
            "query_key": query_key,
        }

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def render_schema(self) -> str:
        """Render a human-readable schema block for the system prompt.

        The rendering describes **how attributes map to kwargs** so the
        policy knows which attributes to set on ``<call_tool>``. Each tool's
        ``parameters`` schema names the accepted attributes explicitly.
        """
        lines: List[str] = []
        for name, meta in self._tools.items():
            params = meta["parameters"] or {}
            properties = params.get("properties", {}) if isinstance(params, dict) else {}
            required = set(params.get("required", [])) if isinstance(params, dict) else set()
            query_key = meta.get("query_key", "query")

            attr_lines: List[str] = []
            for attr_name, attr_schema in properties.items():
                if attr_name == query_key:
                    continue
                type_str = attr_schema.get("type", "string") if isinstance(attr_schema, dict) else "string"
                desc = attr_schema.get("description", "") if isinstance(attr_schema, dict) else ""
                req_mark = " (required)" if attr_name in required else ""
                attr_lines.append(
                    f"      - {attr_name}: {type_str}{req_mark} -- {desc}"
                )

            attrs_block = "\n".join(attr_lines) if attr_lines else "      (no extra attributes)"
            lines.append(
                f"- {name}: {meta['description']}\n"
                f"    Inner text ({query_key}): "
                f"{properties.get(query_key, {}).get('description', 'primary input') if isinstance(properties.get(query_key, {}), dict) else 'primary input'}\n"
                f"    Attributes:\n{attrs_block}"
            )
        return "\n".join(lines)

    def execute(self, raw_call_tool_block: str) -> Tuple[bool, str]:
        """Parse a raw ``<call_tool ...>query</call_tool>`` block and invoke the tool.

        Contract:

        * Input is the *entire* ``<call_tool ...>...</call_tool>`` substring
          captured by :data:`cullm.config.CALL_TOOL_RE`.
        * Output is ``(ok, response_text)``. ``response_text`` is always a
          plain string (possibly an ``[error]`` message) capped at 4096
          chars so rollouts can't blow past the context window.
        """
        rank = _runtime_rank()
        started = time.perf_counter()
        logger.info(
            "[tool_registry][rank=%s] execute_start raw_block=%s",
            rank,
            _shorten_text(raw_call_tool_block, limit=320),
        )
        tool_name, kwargs, query_text = parse_call_tool_block(raw_call_tool_block)
        if tool_name is None:
            logger.info(
                "[tool_registry][rank=%s] execute_parse_error elapsed=%.3fs",
                rank,
                time.perf_counter() - started,
            )
            return False, "[parse error] could not extract <call_tool> block"
        if tool_name not in self._tools:
            logger.info(
                "[tool_registry][rank=%s] execute_unknown_tool tool=%s elapsed=%.3fs",
                rank,
                tool_name,
                time.perf_counter() - started,
            )
            return (
                False,
                f"[unknown tool] {tool_name!r}; available: {self.names()}",
            )

        meta = self._tools[tool_name]
        query_key = meta.get("query_key", "query")
        # Merge inner text into kwargs under the tool's primary query key
        # unless the policy already provided it explicitly via an attribute
        # (rare, but supported).
        if query_text and query_key not in kwargs:
            kwargs[query_key] = query_text

        logger.info(
            "[tool_registry][rank=%s] execute_dispatch tool=%s query_key=%s kwargs=%s",
            rank,
            tool_name,
            query_key,
            _shorten_text(json.dumps(kwargs, ensure_ascii=False, default=str), limit=320),
        )
        try:
            fn = meta["fn"]
            sig = inspect.signature(fn)
            params = sig.parameters

            # 若函数声明了 **kwargs，则无需过滤
            has_var_kwargs = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )

            if has_var_kwargs:
                filtered_kwargs = kwargs
            else:
                # 仅保留函数显式接受的关键字参数
                accepted_names = {
                    name
                    for name, p in params.items()
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                filtered_kwargs = {
                    k: v
                    for k, v in kwargs.items()
                    if k in accepted_names
                }

            out = fn(**filtered_kwargs)
            out_str = str(out)[:4096]
            logger.info(
                "[tool_registry][rank=%s] execute_done tool=%s ok=true elapsed=%.3fs "
                "chars=%s documents=%s",
                rank,
                tool_name,
                time.perf_counter() - started,
                len(out_str),
                out_str.count("<document "),
            )
            return True, out_str
        except Exception as e:  # noqa: BLE001
            logger.info(
                "[tool_registry][rank=%s] execute_done tool=%s ok=false elapsed=%.3fs "
                "error=%s: %s",
                rank,
                tool_name,
                time.perf_counter() - started,
                type(e).__name__,
                e,
            )
            return (
                False,
                f"[{tool_name} error] {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()[:500]}",
            )


# ---------------------------------------------------------------------------
# Internal tool adapters（公司内部工具 -> 统一的字符串返回接口）。
#
# 关键约定：
#
# * 适配层只暴露简洁的 kwargs（query / queries / url / description / top_k）。
# * 内部工具仍然需要 page_idx / map 字典等状态，这里用模块级缓存给它们
#   提供默认值，避免把复杂的内部 state 暴露给模型。
# * 所有适配器都返回 ``str``；evaluator 会从字符串里提取 ``<document …>`` 或
#   ``<|superscript|>:N`` 这样的引用标记。
# ---------------------------------------------------------------------------


# ------- Lazy singletons ----------------------------------------------------


_SCHOLAR_TOOL = None
_GENERAL_TOOL = None
_FETCH_TOOL = None

# 工具之间共享的 page_idx / url 映射，跨一次 rollout 内递增。
# trainer 每次调用 execute 时会累积 page_idx，保证引用号在同一回合内不重复。
_INTERNAL_STATE: Dict[str, Any] = {
    "page_idx": 0,
    "refid_url_map": {},
    "url_tosurl_map": {},
    "tosurl_url_map": {},
    "no_display_url_map": {},
    "url_no_display_map": {},
    "last_time": None,
    "curr_title": None,
}


def _reset_internal_state(
    last_time: Optional[str] = None,
    curr_title: Optional[str] = None,
) -> None:
    """由 trainer 在每个 rollout/sample 开始时调用，让引用号从 0 重新计数。

    Args:
        last_time: 传入 ScholarSearch / GeneralSearch 的 ``last_time`` 字段，
            通常取自该 sample 的 ``verifiable_meta["published_date"]``。
        curr_title: 传入 ScholarSearch / GeneralSearch 的 ``curr_title`` 字段，
            通常取自该 sample 的 ``verifiable_meta["paper_title"]``。
    """
    _INTERNAL_STATE["page_idx"] = 0
    _INTERNAL_STATE["refid_url_map"].clear()
    _INTERNAL_STATE["url_tosurl_map"].clear()
    _INTERNAL_STATE["tosurl_url_map"].clear()
    _INTERNAL_STATE["no_display_url_map"].clear()
    _INTERNAL_STATE["url_no_display_map"].clear()
    _INTERNAL_STATE["last_time"] = last_time
    _INTERNAL_STATE["curr_title"] = curr_title


def _get_scholar_tool():
    global _SCHOLAR_TOOL
    if _SCHOLAR_TOOL is None:
        from .scholar_search_tool import ScholarSearchTool
        _SCHOLAR_TOOL = ScholarSearchTool(return_prompt_type="xml")
    return _SCHOLAR_TOOL


def _get_general_tool():
    global _GENERAL_TOOL
    if _GENERAL_TOOL is None:
        from .search_tool import GeneralSearchTool
        _GENERAL_TOOL = GeneralSearchTool(return_prompt_type="xml")
    return _GENERAL_TOOL


def _get_fetch_tool():
    global _FETCH_TOOL
    if _FETCH_TOOL is None:
        from .linkreader_tool import Fetch
        _FETCH_TOOL = Fetch(return_prompt_type="xml")
    return _FETCH_TOOL


# ------- Public adapter functions（会被 ToolRegistry 注册） ----------------


def _normalize_queries(queries: Any) -> List[str]:
    """把 str / list / None 统一成 list[str]。"""
    if queries is None:
        return []
    if isinstance(queries, str):
        return [queries]
    if isinstance(queries, (list, tuple)):
        return [str(q) for q in queries if q]
    return [str(queries)]


def _normalize_urls(url: Any) -> List[str]:
    if url is None:
        return []
    if isinstance(url, str):
        return [url]
    if isinstance(url, (list, tuple)):
        return [str(u) for u in url if u]
    return [str(url)]


_DEFAULT_SCHOLAR_TOP_K = int(os.environ.get("SCHOLAR_SEARCH_TOP_K", "5"))


def _truncate_xml_documents(xml_content: str, top_k: int) -> str:
    """只保留 XML 中前 top_k 个 <document> 块，减少 token 开销。"""
    if top_k <= 0:
        return xml_content
    # 按 <document 分割，保留前 top_k 个
    parts = re.split(r"(?=<document\s)", xml_content)
    # parts[0] 是 <document 之前的内容（如 <search ...>）
    header = parts[0]
    doc_parts = parts[1:]
    if len(doc_parts) <= top_k:
        return xml_content
    # 保留前 top_k 个文档 + 闭合标签
    kept = header + "".join(doc_parts[:top_k])
    # 确保有闭合的 </search> 或其他结尾标签
    if "</search>" not in kept and "</search>" in xml_content:
        kept = kept.rstrip() + "\n</search>"
    return kept


def scholar_search_adapter(query: str, top_k: int = _DEFAULT_SCHOLAR_TOP_K) -> str:
    """``ScholarSearchTool`` 的最小包装。

    Args:
        query: 学术搜索 query。
        top_k: 每个 query 返回的文档数。可通过环境变量 SCHOLAR_SEARCH_TOP_K
            设置默认值（默认 5）。模型也可在 call_tool 属性中指定。
    """
    queries = _normalize_queries(query)
    if not queries:
        return "[ScholarSearch] empty query"
    search_request_list = [{"query": q} for q in queries]

    tool = _get_scholar_tool()
    prev_page_idx = _INTERNAL_STATE["page_idx"]
    out = tool(
        search_request_list=search_request_list,
        page_idx=prev_page_idx,
        last_time=_INTERNAL_STATE["last_time"],
        curr_title=_INTERNAL_STATE["curr_title"],
        refid_url_map=_INTERNAL_STATE["refid_url_map"],
        url_tosurl_map=_INTERNAL_STATE["url_tosurl_map"],
        tosurl_url_map=_INTERNAL_STATE["tosurl_url_map"],
        no_display_url_map=_INTERNAL_STATE["no_display_url_map"],
        url_no_display_map=_INTERNAL_STATE["url_no_display_map"],
    )
    _INTERNAL_STATE["page_idx"] = out.get("page_idx", _INTERNAL_STATE["page_idx"])
    content = out.get("content", "")
    if not content:
        return "[ScholarSearch] empty response; check logid=" + str(
            out.get("metric", {}).get("log_id", "")
        )
    # 截断到 top_k 个文档，并修正 page_idx 使 ID 连续
    top_k = max(1, min(int(top_k or _DEFAULT_SCHOLAR_TOP_K), 20))
    content = _truncate_xml_documents(content, top_k)
    # 用截断后实际保留的文档数更新 page_idx，避免 ID 跳跃
    actual_docs = len(re.findall(r"<document\s", content))
    _INTERNAL_STATE["page_idx"] = prev_page_idx + actual_docs
    return content


_DEFAULT_GENERAL_TOP_K = int(os.environ.get("GENERAL_SEARCH_TOP_K", "5"))


def general_search_adapter(query: Any, top_k: int = _DEFAULT_GENERAL_TOP_K, is_cot: bool = True) -> str:
    """``GeneralSearchTool`` 的最小包装。支持 ``query`` 为 str 或 list。"""
    queries = _normalize_queries(query)
    if not queries:
        return "[GeneralSearch] empty query"
    search_request_list = [{"query": q} for q in queries]

    tool = _get_general_tool()
    prev_page_idx = _INTERNAL_STATE["page_idx"]
    out = tool(
        is_cot=bool(is_cot),
        search_request_list=search_request_list,
        page_idx=prev_page_idx,
        last_time=_INTERNAL_STATE["last_time"],
        curr_title=_INTERNAL_STATE["curr_title"],
        refid_url_map=_INTERNAL_STATE["refid_url_map"],
        url_tosurl_map=_INTERNAL_STATE["url_tosurl_map"],
        tosurl_url_map=_INTERNAL_STATE["tosurl_url_map"],
        no_display_url_map=_INTERNAL_STATE["no_display_url_map"],
        url_no_display_map=_INTERNAL_STATE["url_no_display_map"],
    )
    _INTERNAL_STATE["page_idx"] = out.get("page_idx", _INTERNAL_STATE["page_idx"])
    content = out.get("content", "")
    if not content:
        return "[GeneralSearch] empty response; check logid=" + str(
            out.get("metric", {}).get("log_id", "")
        )
    # 截断到 top_k 个文档，并修正 page_idx 使 ID 连续
    top_k = max(1, min(int(top_k or _DEFAULT_GENERAL_TOP_K), 20))
    content = _truncate_xml_documents(content, top_k)
    # 用截断后实际保留的文档数更新 page_idx，避免 ID 跳跃
    actual_docs = len(re.findall(r"<document\s", content))
    _INTERNAL_STATE["page_idx"] = prev_page_idx + actual_docs
    return content


def fetch_adapter(url: Any, description: str = "") -> str:
    """``Fetch`` (LinkReader) 的最小包装。``url`` 支持 str 或 list。"""
    urls = _normalize_urls(url)
    if not urls:
        return "[Fetch] empty url"
    fetch_request_list = [{"url": u, "snippet": {}} for u in urls]

    tool = _get_fetch_tool()
    out = _call_fetch(tool, fetch_request_list)
    content = out.get("content", "") if isinstance(out, dict) else ""
    if isinstance(out, dict):
        _INTERNAL_STATE["page_idx"] = out.get("page_idx", _INTERNAL_STATE["page_idx"])
    if not content:
        logid = ""
        if isinstance(out, dict):
            logid = str(out.get("metric", {}).get("log_id", ""))
        return "[Fetch] empty response; check logid=" + logid
    return content


def _call_fetch(tool, fetch_request_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fetch.__call__ 的签名在不同版本略有差异，这里做兼容处理。"""
    for req in fetch_request_list:
        req.setdefault("snippet", {})
    try:
        return tool(
            fetch_request_list=fetch_request_list,
            page_idx=_INTERNAL_STATE["page_idx"],
        )
    except TypeError:
        # 兜底：直接用底层 get_response + adapt_to_text。
        raw, metric = tool.get_response(fetch_request_list)
        if raw is None:
            return {"content": "", "metric": metric, "page_idx": _INTERNAL_STATE["page_idx"]}
        content, new_page = tool.adapt_to_text(raw, _INTERNAL_STATE["page_idx"])
        return {"content": content, "metric": metric, "page_idx": new_page}


# ---------------------------------------------------------------------------
# Schemas（写进 system prompt，供模型组装 JSON 调用）。
# ---------------------------------------------------------------------------


_SCHOLAR_DESCRIPTION = (
    "Search the internal scholar corpus. The inner text of <call_tool> is a "
    "natural-language scholar query. Returns an XML snippet containing "
    "<document reference_id=\"<|superscript|>:N\"> blocks with title/abstract/"
    "author/url. In your final <answer>, cite returned documents with "
    "<cite id=\"N\"></cite> where N is the numeric reference_id."
)

_GENERAL_SEARCH_DESCRIPTION = (
    "General-purpose web search tool. The inner text is a natural-language "
    "query. Returns an XML snippet of <document reference_id=\"<|superscript|>:N\"> "
    "blocks. Prefer ScholarSearch for academic questions."
)

_FETCH_DESCRIPTION = (
    "Fetch the full text of a URL (web page or PDF). The inner text of "
    "<call_tool> must be the target URL; the ``description`` attribute "
    "tells the tool what to extract. Only call Fetch on URLs that appeared "
    "in a previous tool response."
)


# ---------------------------------------------------------------------------
# Default registry.
# ---------------------------------------------------------------------------


def _register_fake(reg: "ToolRegistry") -> None:
    reg.register(
        name="scholar_search",
        description=(
            "Search the scholar corpus for papers relevant to a natural-language "
            "query. The inner text of <call_tool> is the query. Returns an XML "
            "snippet of <document reference_id=\"scholar:N\">...</document> "
            "blocks. Cite documents in the final answer using "
            "<cite id=\"N\"></cite> where N is the numeric reference_id."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language scholar search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of documents to return (1-10).",
                },
            },
            "required": ["query"],
        },
        fn=fake_scholar_search,
        query_key="query",
    )


def _register_internal(reg: "ToolRegistry") -> None:
    reg.register(
        name="ScholarSearch",
        description=_SCHOLAR_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language scholar query (Chinese or English).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Unused by the internal tool; kept for API compat.",
                },
            },
            "required": ["query"],
        },
        fn=scholar_search_adapter,
        query_key="query",
    )
    reg.register(
        name="GeneralSearch",
        description=_GENERAL_SEARCH_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A single web search query.",
                },
                "is_cot": {
                    "type": "boolean",
                    "description": "Whether this call is inside a CoT rollout (affects top-k).",
                },
            },
            "required": ["query"],
        },
        fn=general_search_adapter,
        query_key="query",
    )
    reg.register(
        name="Fetch",
        description=_FETCH_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Target URL (or list of URLs) to fetch.",
                },
                "description": {
                    "type": "string",
                    "description": "What information you want to extract from the page.",
                },
            },
            "required": ["url"],
        },
        fn=fetch_adapter,
        query_key="url",
    )


def build_default_registry(backend: str = "internal") -> ToolRegistry:
    """构建默认工具注册表。

    Args:
        backend:
            * ``"mock"``     -- 纯离线 fake scholar search（走 ``fake_scholar_search``）。
            * ``"internal"`` -- 公司内部三件套 ScholarSearch / GeneralSearch / Fetch。
            * ``"real"``     -- 别名，当前等价于 internal。
    """
    reg = ToolRegistry()
    backend = (backend or "internal").lower()
    if backend == "mock":
        _register_fake(reg)
    elif backend in ("internal", "real"):
        try:
            _register_internal(reg)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to initialise internal tools (%s: %s); falling back to mock.",
                type(e).__name__,
                e,
            )
            _register_fake(reg)
    else:
        raise ValueError(f"Unknown tool backend: {backend!r}")
    return reg


# ---------------------------------------------------------------------------
# Re-exports.
# ---------------------------------------------------------------------------


__all__ = [
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
    "TOOL_RESP_OPEN",
    "TOOL_RESP_CLOSE",
    "ANSWER_CLOSE",
    "CALL_TOOL_OPEN_PREFIX",
    "CALL_TOOL_CLOSE",
    "CALL_TOOL_RE",
    "CITE_RE",
    "TOOL_CALL_RE",
    "ToolRegistry",
    "build_default_registry",
    "fake_scholar_search",
    "scholar_search_adapter",
    "general_search_adapter",
    "fetch_adapter",
    "parse_call_tool_block",
    "_reset_internal_state",
]
