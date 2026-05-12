# debug-rl-missing-answer-reward

Status: [OPEN]

## Symptom

RL rollout can start, but many samples appear to have no reward and no final `<answer>` in the printed trajectory.

## Hypotheses

1. The forced-answer branch is not triggered because `saw_answer` is incorrectly true, or `rollout.step_count` does not reach `max_steps`.
2. The forced-answer prompt is appended as plain text instead of chat/user-formatted text, so Qwen does not treat it as a final user instruction.
3. Stop sequences still include tool-call stop strings, so the forced-answer call is cut off after another `<call_tool>` before `<answer>`.
4. Reward extraction expects `<answer>...</answer>` but the forced output is outside that format or masked/truncated before reward computation.
5. Printed trajectory omits the mask-0 force prompt or final tokens, making the sample look answer-less even when tokens exist.

## Evidence Plan

Inspect the latest trajectory/debug logs and reward logs before changing business logic.

## Evidence

- `outputs/qwen3_8b_scholar_sft_rl_real/train.log` shows multiple zero-reward trajectories whose model output is only `<call_tool ...>` and has no `<answer>`.
- Some bad trajectories use invalid tool names such as `InternalTool` and `InternalNote`; these are not in the configured tools `ScholarSearch`, `GeneralSearch`, `Fetch`.
- `DRTuluToolParser` ignores unknown tool names, so `process_request()` sees `tool_calls=[]` and exits as if the model chose to stop naturally.
- The run log references `vllm_utils.py:1496` for `num_gpus`, while the current file has that line at `1610`, so the inspected run likely predates the latest force-answer patch or is from a stale actor/process.

## Current Conclusion

The max-step force-answer branch is not sufficient. Rollout also needs to force a final answer when the model emits a raw `<call_tool>` block that cannot be dispatched to any active tool, and forced final-answer calls should not inherit tool stop sequences.

## Fix Applied

- `open_instruct/vllm_utils.py` now detects raw `<call_tool>...</call_tool>` output even when the parser rejects it as an unknown/invalid tool name.
- If a raw tool call exists but no active tool can be dispatched and no `<answer>` is present, rollout appends the final-answer prompt and makes one extra no-tool generation call with `force_answer_reason="invalid_tool_call"`.
- The forced final-answer call removes `</call_tool>` from stop strings so it is not cut off by another tool-call close tag.
- The forced prompt is encoded through the tokenizer chat template when available, falling back to raw tokenization.

## Validation

- `python -m py_compile open_instruct/vllm_utils.py` passed.
- `uv run ruff check open_instruct/vllm_utils.py` passed.
- `git diff --check -- open_instruct/vllm_utils.py debug-rl-missing-answer-reward.md` passed.

## Post-Restart Evidence

- `.dbg/rl-rollout-stall-local.ndjson` now contains `starting forced final-answer completion request` for both `force_answer_reason="max_steps_exhausted"` and `force_answer_reason="invalid_tool_call"`.
- There are no matching `finished forced final-answer completion request` events in the same log.
- Affected base requests such as `train_0_6727` and `train_0_6553` also do not show `vllm_utils.finalize_completed_request -> put result into queue`.
- `train.log` still shows evaluator input as raw tool calls like `<call_tool name="InternalTool">...` with no `<answer>` and reward components all zero:
  - citation/relevance/reference scores are `0.0`
  - `scholar_reward_debug ... final_score=0.1` from format reward only

## Updated Conclusion

The forced-answer branch is being entered, but the forced completion is not reaching a successful completion event. The reward path is therefore still seeing the pre-force raw tool-call output. The next step should be instrumentation around the forced completion API call itself, including success/error/timeout reporting, before changing more rollout logic.

## New Instrumentation

- Wrapped the forced final-answer `actor.client.completions.create(...)` call in `asyncio.wait_for(..., timeout=actor.tool_call_timeout)`.
- Added debug events for:
  - `forced final-answer completion timed out`
  - `forced final-answer completion failed`
  - enriched `finished forced final-answer completion request`
- Each event now records:
  - `force_answer_reason`
  - `elapsed_s`
  - `timeout_s` for timeout events
  - `error_type` and `error` for failure events
  - `current_prompt_len`, `current_max_tokens`, and `stop`
