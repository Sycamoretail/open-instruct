# Debug Session: rollout-step0-stall

Status: [OPEN]

## Symptom

- RL run stalls with `DataPreparationActor.get_data` repeatedly logging `still waiting for step=0, current_prepared_step=-1`.
- Latest evidence already shows requests can pass acquire/reset and reach `starting tool step`, so the visible symptom is not enough to identify the actual blocked stage.

## Scope

- Focus on the latest run only.
- No business-logic changes in this phase; collect runtime evidence first.

## Hypotheses

1. A request enters `starting tool step` and the tool coroutine never returns, so no later completion/result events are emitted.
2. A request finishes the tool step but hangs before or during the next vLLM completion call, so rollout never finalizes.
3. A request finishes generation but stalls in reward aggregation or result packaging, so `put result into queue` never happens.
4. The run reaches forced final-answer handling, but the forced-answer completion still blocks before success/error/timeout logs are emitted.

## Evidence To Collect

- For latest `base_request_id`s, whether each request has:
  - `finished tool step`
  - subsequent `starting/finished vLLM completion request`
  - `starting/finished forced final-answer completion request`
  - `finished reward computation`
  - `put result into queue`
- Whether a small set of requests all stop at the same stage.

## Evidence Collected

- Latest stalled requests do progress far beyond `after-acquire-reset`; e.g. `train_0_7101_*` and `train_0_7197_*` repeatedly emit:
  - `starting/finished vLLM completion request`
  - `starting/finished tool step`
  - then `starting forced final-answer completion request` at `step_count=10`
- The blocking failure is now reproduced with concrete runtime evidence:
  - `forced final-answer completion failed`
  - `error_type=BadRequestError`
  - request validation error mentions prompt entries like `'input_ids'` and `'attention_mask'`
- This points to malformed prompt token construction for the forced-answer turn, not a hanging tool step or reward queue issue.

## Minimal Fix

- In `process_force_answer_prompt_tokens()`, treat `apply_chat_template(..., tokenize=True)` as possibly returning a BatchEncoding/dict-like object.
- Extract `input_ids` explicitly and only then convert to a Python list, preventing mapping keys such as `input_ids` / `attention_mask` from being appended into `current_prompt`.

## Notes

- Existing related session: `debug-rl-missing-answer-reward.md`
- This session specifically narrows the new `step=0/current_prepared_step=-1` stall to an internal blocked phase in the latest run.
