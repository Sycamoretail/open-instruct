import unittest
from unittest.mock import AsyncMock, patch

from open_instruct import scholar_evaluator


class TestScholarRewardCandidateGate(unittest.TestCase):
    def test_rejects_tool_trace_without_final_answer(self):
        candidate = '<call_tool name="ScholarSearch">foo</call_tool><tool_output><search>bar</search></tool_output>'
        self.assertEqual(
            scholar_evaluator._invalid_reward_candidate_reason(candidate), "tool_trace_without_final_answer"
        )

    def test_allows_tool_trace_when_final_answer_exists(self):
        candidate = (
            '<call_tool name="ScholarSearch">foo</call_tool><tool_output>bar</tool_output>'
            '<answer>Done.<cite id="1"></cite></answer>'
        )
        self.assertIsNone(scholar_evaluator._invalid_reward_candidate_reason(candidate))

    def test_rejects_malformed_tool_trace_without_final_answer(self):
        candidate = '<call_tool name="ScholarSearch">foo<tool_output>bar</tool_output>'
        self.assertEqual(
            scholar_evaluator._invalid_reward_candidate_reason(candidate), "malformed_tool_trace_without_final_answer"
        )


class TestScholarRewardVerifierGate(unittest.IsolatedAsyncioTestCase):
    async def test_search_reward_short_circuits_invalid_tool_trace(self):
        candidate = '<call_tool name="ScholarSearch">foo</call_tool><tool_output>bar</tool_output>'
        ground_truth = {
            "data_type": "search",
            "verifiable_meta": {
                "query": "test query",
                "paper_title": "Paper",
                "section_title": "Related Work",
                "text": "Ground truth text",
                "citations": ["Ref A"],
            },
        }

        with (
            patch.object(
                scholar_evaluator, "relevance_rate_evaluate", AsyncMock(side_effect=AssertionError("should not run"))
            ),
            patch.object(
                scholar_evaluator,
                "reference_coverage_evaluate",
                AsyncMock(side_effect=AssertionError("should not run")),
            ),
            patch.object(
                scholar_evaluator,
                "citation_precision_evaluate",
                AsyncMock(side_effect=AssertionError("should not run")),
            ),
        ):
            result_dict, ok = await scholar_evaluator.evaluate_scholar_score_verifier(candidate, ground_truth, "")

        self.assertTrue(ok)
        self.assertEqual(result_dict["score"], 0.0)
        self.assertEqual(result_dict["citation_len"], 0)
        self.assertEqual(result_dict["invalid_candidate_reason"], "tool_trace_without_final_answer")
        self.assertEqual(result_dict["format_reward"], 0.0)
        self.assertEqual(result_dict["format_reward_weight"], 0.0)
