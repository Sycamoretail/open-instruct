"""Protocol tokens + regexes used by :mod:`tools` (ported from ``cullm.config``).

The search-agent speaks the ReAct-style XML dialect described in the paper
figure:

* ``<think>...</think>``                    -- internal planning.
* ``<call_tool name="ScholarSearch">query</call_tool>`` -- tool invocation.
* ``<tool_response>...</tool_response>``    -- result injected back by the env.
* ``<answer>...</answer>``                  -- final response.
* ``<cite id="12"></cite>``                 -- citations inside <answer>.

Only the bits of the original ``cullm.config`` that :mod:`tools` actually
imports are reproduced here so the package is self-contained inside the
open-instruct checkout.
"""

from __future__ import annotations

import re

# -- Raw protocol tokens ----------------------------------------------------

TOOL_CALL_OPEN = "<call_tool"
TOOL_CALL_CLOSE = "</call_tool>"

TOOL_RESP_OPEN = "<tool_response>"
TOOL_RESP_CLOSE = "</tool_response>"

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

CALL_TOOL_OPEN_PREFIX = "<call_tool"

# -- Regexes ----------------------------------------------------------------

# Matches ``<call_tool ...attrs...>inner text</call_tool>`` with attributes in
# a single capture group so ``tools.__init__._parse_call_tool_attrs`` can run
# over group(1). ``re.DOTALL`` is important so multi-line queries work.
CALL_TOOL_RE = re.compile(
    re.escape("<call_tool") + r"([^>]*)>(.*?)" + re.escape("</call_tool>"),
    re.DOTALL,
)

# Matches ``<cite id="12"></cite>`` or ``<cite id="12">optional text</cite>``.
CITE_RE = re.compile(
    r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.*?)</cite>",
    re.DOTALL,
)
