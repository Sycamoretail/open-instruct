"""Open-instruct Tool wrappers around the company-internal ``tools`` package.

The internal package (``tools/__init__.py``) already exposes:

* ``build_default_registry(backend=...)`` -- assembles a :class:`tools.ToolRegistry`
  where tools are addressed by name (``ScholarSearch`` / ``GeneralSearch`` /
  ``Fetch`` for ``backend="internal"``; ``scholar_search`` for ``backend="mock"``).
* ``_reset_internal_state(...)`` -- resets ref-id counters between rollouts.

What open-instruct needs from each tool is a ``Tool`` subclass plus a
``BaseEnvConfig`` entry in ``TOOL_REGISTRY``. We generate one adapter class
per internal tool; each adapter:

1. forwards ``step()`` to the underlying callable in the internal registry,
2. reports a stable ``config_name`` so users can reference it from the CLI
   via ``--tools scholar_search`` / ``--tools general_search`` /
   ``--tools fetch``,
3. declares ``call_name`` matching the ``name="..."`` attribute that the
   policy is expected to emit inside ``<call_tool>``.

Paired with the ``dr_tulu`` tool parser (see
``open_instruct/environments/tools/parsers.py::DRTuluToolParser``), the
protocol on the wire is:

    <call_tool name="ScholarSearch">retrieval augmented generation</call_tool>

and the env will write back ``<tool_response>...</tool_response>`` messages
into the conversation.
"""

from __future__ import annotations

import asyncio
import importlib
import time
from dataclasses import dataclass
from typing import Any, ClassVar

from open_instruct import logger_utils
from open_instruct.environments.base import BaseEnvConfig, EnvCall, StepResult
from open_instruct.environments.tools.utils import Tool, coerce_args, get_openai_tool_definitions, log_env_call

logger = logger_utils.setup_logger(__name__)


_REGISTRY_CACHE: dict[str, Any] = {}


def _get_registry(backend: str):
    """Build (once) and return the ``tools.ToolRegistry`` for the backend."""
    if backend not in _REGISTRY_CACHE:
        tools_module = importlib.import_module("tools")
        _REGISTRY_CACHE[backend] = tools_module.build_default_registry(backend)
    return _REGISTRY_CACHE[backend]


def _call_internal(backend: str, tool_name: str, kwargs: dict[str, Any]) -> tuple[bool, str]:
    """Invoke a named tool inside ``tools.ToolRegistry`` and return (ok, str)."""
    reg = _get_registry(backend)
    meta = reg._tools.get(tool_name)  # noqa: SLF001  (internal access is fine here)
    if meta is None:
        return False, f"[unknown tool] {tool_name!r}; available: {reg.names()}"
    try:
        out = meta["fn"](**kwargs)
        return True, str(out)[:8192]
    except Exception as e:
        return False, f"[{tool_name} error] {type(e).__name__}: {e}"


class _ScholarToolBase(Tool):
    """Shared machinery: run the blocking internal call on a background thread."""

    backend: ClassVar[str] = "internal"
    internal_name: ClassVar[str] = ""

    async def reset(self, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        """Per-rollout reset that wires per-sample ``last_time`` / ``curr_title``.

        The internal ``tools`` package keeps a module-level ``_INTERNAL_STATE``
        that scholar / general search read from to:
          * filter results by ``last_time`` (a cutoff publication date), and
          * avoid leaking the target paper back via ``curr_title``.

        The training pipeline delivers these fields through the canonical
        per-sample ``env_config`` channel:

            sample["env_config"] -> _merge_env_config ->
            EnvConfigEntry.kwargs -> actor.reset.remote(**kwargs)

        So any kwargs (e.g. ``last_time``, ``curr_title``) forwarded into
        ``reset`` should update the internal state for this rollout.

        Kwargs that are not understood are simply ignored; this keeps the
        reset contract compatible with the default ``Tool.reset``.
        """
        tools_module = importlib.import_module("tools")
        reset_fn = getattr(tools_module, "_reset_internal_state", None)
        if reset_fn is not None:
            last_time = kwargs.get("last_time")
            curr_title = kwargs.get("curr_title")
            try:
                reset_fn(last_time=last_time, curr_title=curr_title)
            except Exception as e:
                logger.warning(
                    "scholar_tools._reset_internal_state failed (last_time=%r, curr_title=%r): %s",
                    last_time,
                    curr_title,
                    e,
                )
        return StepResult(result=""), [get_openai_tool_definitions(self)]

    async def step(self, call: EnvCall) -> StepResult:
        args = coerce_args(self.parameters, call.args)
        start = time.time()

        loop = asyncio.get_event_loop()
        ok, content = await loop.run_in_executor(None, _call_internal, self.backend, self.internal_name, args)
        metadata = {"error": "" if ok else content, "runtime": time.time() - start}
        result = StepResult(result=content if ok else "", metadata=metadata)
        log_env_call(self.call_name, str(args)[:400], result)
        return result


# ---------------------------------------------------------------------------
# Internal-backend tools (ScholarSearch / GeneralSearch / Fetch).
# ---------------------------------------------------------------------------


class ScholarSearchTool(_ScholarToolBase):
    config_name = "scholar_search"
    internal_name = "ScholarSearch"
    call_name = "ScholarSearch"
    description = (
        "Academic search over the internal scholar corpus. "
        "Inner text is a natural-language query. Returns an XML snippet of "
        '<document reference_id="<|superscript|>:N"> blocks.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Scholar search query (CN or EN)."},
            "top_k": {"type": "integer", "description": "Number of docs to return."},
        },
        "required": ["query"],
    }

    def __init__(self, call_name: str | None = None) -> None:
        if call_name:
            self.call_name = call_name


class GeneralSearchTool(_ScholarToolBase):
    config_name = "general_search"
    internal_name = "GeneralSearch"
    call_name = "GeneralSearch"
    description = (
        "General-purpose web search. Inner text is a query. Returns an XML "
        'snippet of <document reference_id="<|superscript|>:N"> blocks.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "A single web search query."},
            "is_cot": {"type": "boolean", "description": "Inside CoT rollout."},
        },
        "required": ["query"],
    }

    def __init__(self, call_name: str | None = None) -> None:
        if call_name:
            self.call_name = call_name


class FetchTool(_ScholarToolBase):
    config_name = "fetch"
    internal_name = "Fetch"
    call_name = "Fetch"
    description = (
        "Fetch the full text of a URL (web page or PDF). Inner text of "
        "<call_tool> is the URL; use the description attribute to indicate "
        "what to extract."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Target URL to fetch."},
            "description": {"type": "string", "description": "What to extract."},
        },
        "required": ["url"],
    }

    def __init__(self, call_name: str | None = None) -> None:
        if call_name:
            self.call_name = call_name


# ---------------------------------------------------------------------------
# Mock-backend tool (deterministic fake responses, no network required).
# ---------------------------------------------------------------------------


class MockScholarSearchTool(_ScholarToolBase):
    config_name = "mock_scholar_search"
    backend = "mock"
    internal_name = "scholar_search"
    call_name = "ScholarSearch"
    description = (
        "Offline fake scholar search (deterministic, no network). "
        "Useful for verifying the <call_tool>/<answer>/<cite> protocol "
        "before switching to the real ScholarSearch backend."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Scholar search query."},
            "top_k": {"type": "integer", "description": "Docs to return (1-10)."},
        },
        "required": ["query"],
    }

    def __init__(self, call_name: str | None = None) -> None:
        if call_name:
            self.call_name = call_name


# ---------------------------------------------------------------------------
# Config dataclasses (open-instruct reads these via TOOL_REGISTRY).
# ---------------------------------------------------------------------------


@dataclass
class ScholarSearchToolConfig(BaseEnvConfig):
    tool_class: ClassVar[type[Tool]] = ScholarSearchTool


@dataclass
class GeneralSearchToolConfig(BaseEnvConfig):
    tool_class: ClassVar[type[Tool]] = GeneralSearchTool


@dataclass
class FetchToolConfig(BaseEnvConfig):
    tool_class: ClassVar[type[Tool]] = FetchTool


@dataclass
class MockScholarSearchToolConfig(BaseEnvConfig):
    tool_class: ClassVar[type[Tool]] = MockScholarSearchTool


SCHOLAR_TOOL_CONFIGS: list[type[BaseEnvConfig]] = [
    ScholarSearchToolConfig,
    GeneralSearchToolConfig,
    FetchToolConfig,
    MockScholarSearchToolConfig,
]
