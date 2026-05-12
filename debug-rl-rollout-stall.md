# debug-rl-rollout-stall

Status: [OPEN]

## Symptom

RL training waits indefinitely at `DataPreparationActor.get_data` for `step=0` with `current_prepared_step=-1`.

## Hypotheses

1. Prompt dispatch stalls before vLLM actors receive requests from `param_prompt_Q`.
2. vLLM generation request starts but does not return, e.g. long generation, context issue, or engine stall.
3. Generation returns a tool call, but a tool actor step blocks or times out.
4. Tool loop finishes, but reward computation blocks before result is put into `inference_results_Q`.
5. Completion is produced, but finalization or queue put into `inference_results_Q` is not reached.

## Instrumentation Plan

Add temporary debug logs around queue receive, vLLM request, tool step, reward computation, completion finalization, and result queue put.

## Instrumentation Added

- `open_instruct/vllm_utils.py`: prompt queue receive, subrequest scheduling, vLLM request start/end, tool step start/end/error/timeout, reward start/end, result queue put.
- `open_instruct/data_loader.py`: accumulate loop start, wait for result, result received.

## Validation

- `python -m py_compile open_instruct/vllm_utils.py open_instruct/data_loader.py` passed.
- IDE diagnostics only show pre-existing unresolved optional dependency warnings.

## Evidence So Far

- `train.log` shows `DataPreparationActor` reached `accumulate_inference_batches` for step 0.
- It is waiting for `result 1/16 from inference_results_Q`.
- No `Got result` appears after that point.
- The debug HTTP collector only contains a manual health event, so Ray actor HTTP reporting is not reaching this sandbox debug server.

## Current Conclusion

The stall is before any inference result is queued. Existing logs do not yet distinguish prompt dispatch, vLLM generation, tool step, or reward.

## 2026-05-11 Recheck

- `.dbg/trae-debug-log-rl-rollout-stall.ndjson` exists but has 0 events.
- `outputs/qwen3_8b_scholar_sft_rl_real_debug/train.log` did not enter rollout; it failed config validation because `num_unique_prompts_rollout * num_samples_per_prompt_rollout = 2 * 2 = 4`, but at least 6 are required.
- `outputs/qwen3_8b_scholar_sft_rl_real/train.log` is the latest actual stuck run and still waits at `Waiting for result 1/16 from inference_results_Q`.

## 2026-05-11 Local File Instrumentation

- New evidence: with `3 * 2` rollout config, the debug run enters rollout and waits at `Waiting for result 1/3 from inference_results_Q`.
- HTTP debug server still receives 0 events.
- Updated `_debug_report` in `open_instruct/vllm_utils.py` and `open_instruct/data_loader.py` to append every event to `.dbg/rl-rollout-stall-local.ndjson` before attempting HTTP reporting.
- `python -m py_compile open_instruct/vllm_utils.py open_instruct/data_loader.py` passed.

## 2026-05-11 Evidence From Local File

- `.dbg/rl-rollout-stall-local.ndjson` contains 148 events.
- `prompt_queue.get` succeeds; many prompt requests are received.
- `add_request` schedules subrequests successfully.
- `process_request` starts for subrequests.
- No event reaches `starting vLLM completion request`.
- No event reaches `starting tool step`.
- Therefore the stall is before generation and before actual tool invocation, inside `_acquire_and_reset_pools(...)`.
- Added finer instrumentation around `pool.acquire.remote()`, `target_actor.reset.remote(...)`, and `target_actor.get_response_role.remote()`.
- `python -m py_compile open_instruct/vllm_utils.py` passed.

## 2026-05-11 Second Restart Check

- Latest `.dbg/rl-rollout-stall-local.ndjson` still contains only the pre-pool events: prompt received, subrequest scheduled, process_request started.
- No `starting pool acquire` event is present, despite source containing that instrumentation.
- `train.log` confirms the run still waits at `Waiting for result 1/3 from inference_results_Q`.
- Hypothesis refined: `_debug_report` may be blocking after writing the first `process_request` event due to the attempted HTTP report to the debug server.
- Updated `_debug_report` in both `open_instruct/vllm_utils.py` and `open_instruct/data_loader.py` to local-file-only logging. No HTTP request is attempted now.
- `python -m py_compile open_instruct/vllm_utils.py open_instruct/data_loader.py` passed.

## 2026-05-11 Third Restart Check

- Latest run uses the local-file-only `_debug_report`; line numbers in `train.log` match the updated source.
- `.dbg/rl-rollout-stall-local.ndjson` still stops after `started processing subrequest`; no pool/vLLM/tool/reward events appear.
- Added raw file instrumentation `.dbg/rl-rollout-stall-raw.log` immediately after process-start, after unknown target computation, and before/after `_acquire_and_reset_pools(...)`.
- `python -m py_compile open_instruct/vllm_utils.py` passed.

## 2026-05-11 Raw Log Root Cause

- `.dbg/rl-rollout-stall-raw.log` contains 96 lines for 48 subrequests.
- Every subrequest reaches `after-process-start before-unknown-targets`.
- Every subrequest reaches `after-unknown-targets count=3`.
- No subrequest reaches `before-acquire-reset`.
- Root cause: sample `env_config` uses registry names (`scholar_search`, `general_search`, `fetch`) while configured tool pools use call names (`ScholarSearch`, `GeneralSearch`, `Fetch`).
- The old `_merge_env_config` kept both sets of keys, so `unknown_targets = set(env_config.env_configs) - configured_tools` contained the 3 lower-case registry names and raised before generation.
- Fixed `_merge_env_config` to resolve sample env names to existing base keys using a case/underscore-insensitive match.
- Added regression test for `scholar_search -> ScholarSearch`, `general_search -> GeneralSearch`, and `fetch -> Fetch`.
- `uv run pytest open_instruct/test_data_loader_env_config.py` passed.
- `python -m py_compile open_instruct/data_loader.py open_instruct/vllm_utils.py` passed.
