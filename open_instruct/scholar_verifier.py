"""Scholar search-agent verifier.

Bridges the project-level ``evaluator.py`` (research-quality judge that
scores candidate answers against per-task ``verifiable_meta`` ground
truths) into the open-instruct GRPO reward pipeline.

The verifier is registered under four dataset names that match the
``data_source`` values packed in ``scholar/*.parquet``:

* ``scholar_search``          -> ``data_type = "search"``
* ``scholar_understanding``   -> ``data_type = "understanding"``
* ``scholar_write_section``   -> ``data_type = "write_section"``
* ``scholar_write_survey``    -> ``data_type = "write_survey"``

Each sample's ground truth is a JSON string of the form::

    {"verifiable_meta": {...}, "data_type": "search"}

The raw tool outputs captured during the rollout are forwarded via
``rollout_state["tool_output"]`` so ``evaluator`` can parse the
``<document>...</document>`` snippets that the model cited.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import re
import sys
import traceback
from typing import Any

from open_instruct import logger_utils
from open_instruct.ground_truth_utils import (
    VerificationResult,
    VerifierConfig,
    VerifierFunction,
)

logger = logger_utils.setup_logger(__name__)


# Make evaluator.py importable - it lives at the repo root next to this file.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import evaluator as scholar_evaluator  # noqa: E402  (path mangling above)
except Exception as e:  # pragma: no cover - best effort import
    logger.warning(f"Failed to import top-level evaluator.py: {e}")
    scholar_evaluator = None


# ---------------------------------------------------------------------------
# Mock mode: deterministic format-only reward.
#
# Use this while bootstrapping a fresh policy that doesn't yet emit the right
# XML protocol. Enable by exporting ``SCHOLAR_MOCK_REWARD=1`` before launching
# training. The verifier will then bypass ``evaluator.py`` entirely and score
# purely based on whether the model output obeys the protocol:
#
#   1. contains a <call_tool> with a ``name="..."`` attribute
#   2. contains exactly one <answer>...</answer> block
#   3. contains at least one <cite id="..."></cite> inside the answer
#   4. does not leave open/unclosed tags
#
# Each satisfied criterion contributes +0.25, capped at 1.0.
# ---------------------------------------------------------------------------


_MOCK_ENV_VAR = "SCHOLAR_MOCK_REWARD"


_CALL_TOOL_RE = re.compile(
    r"<call_tool\s+([^>]*name\s*=\s*\"[^\"]+\"[^>]*)>(.*?)</call_tool>",
    re.DOTALL,
)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_CITE_RE = re.compile(r"<cite\s+id\s*=\s*\"([^\"]+)\"\s*>(.*?)</cite>", re.DOTALL)


def _mock_format_reward(prediction: str) -> tuple[float, dict[str, bool]]:
    text = prediction or ""
    checks: dict[str, bool] = {}

    checks["has_call_tool"] = bool(_CALL_TOOL_RE.search(text))
    answer_matches = _ANSWER_RE.findall(text)
    checks["has_single_answer"] = len(answer_matches) == 1

    if checks["has_single_answer"]:
        checks["has_cite_inside_answer"] = bool(_CITE_RE.search(answer_matches[0]))
    else:
        checks["has_cite_inside_answer"] = False

    # Balanced tags: naive check that opening and closing counts match for
    # the protocol tokens we care about.
    balanced = (
        text.count("<call_tool") == text.count("</call_tool>")
        and text.count("<answer>") == text.count("</answer>")
        and text.count("<think>") == text.count("</think>")
    )
    checks["tags_balanced"] = balanced

    score = sum(1 for v in checks.values() if v) / max(1, len(checks))
    return score, checks


def _mock_mode_enabled() -> bool:
    return os.environ.get(_MOCK_ENV_VAR, "").lower() in {"1", "true", "yes", "on"}


_DATA_TYPE_BY_DATASET = {
    "scholar_search": "search",
    "scholar_understanding": "understanding",
    "scholar_write_section": "write_section",
    "scholar_write_survey": "write_survey",
}


@dataclasses.dataclass
class ScholarVerifierConfig(VerifierConfig):
    """No extra knobs today - kept for symmetry with other verifiers."""


def _coerce_ground_truth(label: Any, default_data_type: str) -> dict:
    """Normalize the ground truth into the dict that evaluator.py expects."""
    if isinstance(label, dict):
        gt = dict(label)
    elif isinstance(label, str):
        try:
            gt = json.loads(label)
        except json.JSONDecodeError:
            logger.warning("ScholarVerifier: ground truth is not JSON, wrapping raw text.")
            gt = {"verifiable_meta": {"text": label}}
    else:
        gt = {"verifiable_meta": {}}

    gt.setdefault("data_type", default_data_type)
    gt.setdefault("verifiable_meta", {})
    return gt


def _collect_tool_results_str(rollout_state: dict | None) -> str:
    """Flatten tool outputs produced during the rollout.

    ``rollout_state`` is the dict populated by
    :func:`open_instruct.vllm_utils` and carries either a single
    ``tool_output`` string or a list under ``tool_outputs``.
    """
    if not rollout_state:
        return ""
    if isinstance(rollout_state.get("tool_output"), str):
        return rollout_state["tool_output"]
    outputs = rollout_state.get("tool_outputs")
    if isinstance(outputs, (list, tuple)):
        return "\n".join(str(o) for o in outputs if o)
    return ""


class _ScholarVerifierMixin:
    """Shared async implementation - subclasses only set the dataset name."""

    dataset_name: str = ""

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        assert self.dataset_name, "dataset_name must be set in subclass"
        super().__init__(self.dataset_name, verifier_config=verifier_config, weight=1.0)  # type: ignore[misc]

    @classmethod
    def get_config_class(cls) -> type:
        return ScholarVerifierConfig

    async def async_call(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        # Mock mode: score the response purely on protocol compliance so the
        # policy can first learn to emit well-formed <call_tool>/<answer>/<cite>
        # before we pay for an evaluator-LLM round trip.
        if _mock_mode_enabled():
            score, checks = _mock_format_reward(prediction)
            return VerificationResult(
                score=float(score),
                reasoning=json.dumps({"mock": True, **checks}, ensure_ascii=False),
            )

        if scholar_evaluator is None:
            return VerificationResult(score=0.0, reasoning="evaluator import failed")

        data_type = _DATA_TYPE_BY_DATASET[self.dataset_name]
        ground_truth = _coerce_ground_truth(label, data_type)
        tool_results_str = _collect_tool_results_str(rollout_state)

        try:
            result_dict, ok = await scholar_evaluator.evaluate_scholar_score_verifier(
                candidate=prediction or "",
                ground_truth=ground_truth,
                tool_results_str=tool_results_str,
            )
        except Exception as e:
            logger.error(f"ScholarVerifier[{self.dataset_name}] evaluator error: {e}\n{traceback.format_exc()}")
            return VerificationResult(score=0.0, reasoning=f"evaluator error: {e}")

        if not ok or not isinstance(result_dict, dict):
            return VerificationResult(score=0.0, reasoning="evaluator returned invalid result")

        try:
            score = float(result_dict.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        reasoning_bits = {k: v for k, v in result_dict.items() if k not in {"candidate", "judge_response"}}
        return VerificationResult(score=score, reasoning=json.dumps(reasoning_bits, ensure_ascii=False, default=str))

    def __call__(
        self,
        tokenized_prediction: list[int],
        prediction: str,
        label: Any,
        query: str | None = None,
        rollout_state: dict | None = None,
    ) -> VerificationResult:
        """Synchronous fallback - run the async path on a fresh loop."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "ScholarVerifier must be invoked via async_call from an async context."
            )
        except RuntimeError:
            pass
        return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query, rollout_state))


class ScholarSearchVerifier(_ScholarVerifierMixin, VerifierFunction):
    dataset_name = "scholar_search"


class ScholarUnderstandingVerifier(_ScholarVerifierMixin, VerifierFunction):
    dataset_name = "scholar_understanding"


class ScholarWriteSectionVerifier(_ScholarVerifierMixin, VerifierFunction):
    dataset_name = "scholar_write_section"


class ScholarWriteSurveyVerifier(_ScholarVerifierMixin, VerifierFunction):
    dataset_name = "scholar_write_survey"
