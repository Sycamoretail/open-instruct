# Debug Session: citation-len-zero

Status: [OPEN]

## Symptom

- Many latest trajectories finish rollout and reward computation, but `scholar_reward_debug` still shows `citation_len=0`.
- The main question is whether this is caused by generation behavior, citation-format extraction, or reward-side input mismatch.

## Scope

- Focus on the latest successful runs after the forced-answer 400 fix.
- Do not modify business logic in this phase; gather evidence first.

## Hypotheses

1. The model often produces `<answer>` without any `<cite id="..."></cite>` tags, so `citation_len=0` is an accurate signal rather than an evaluator bug.
2. The model includes citations, but the tag format deviates from the evaluator expectation, so citations are present semantically but not counted.
3. The forced-answer branch completes, but the final answer omits source IDs from previous tool outputs, causing missing citations even when tool evidence exists.
4. The reward pipeline sometimes evaluates an intermediate / malformed response instead of the final answer, so `citation_len=0` reflects an upstream packaging issue.

## Evidence To Collect

- Sample latest `scholar_reward_debug` lines with `citation_len=0`.
- Match them to trajectory / rollout debug lines for the same requests.
- Inspect evaluator-side citation extraction expectations in reward code.
- Check whether zero-citation cases cluster around forced-answer completions or happen broadly.

## Evidence Collected

- Evaluator behavior is strict and explicit:
  - `clean_rich_media_reference(..., return_citation_ids=True)` only counts citations from:
    - `<answer>...</answer>` containing `<cite id="N">...</cite>`
    - or the older `<RichMediaReference>...superscript:N...</RichMediaReference>` format
  - `citation_len` is set to `len(gen_references)` after matching those extracted citation IDs against parsed tool documents.
- A first bucket of `citation_len=0` comes from malformed/non-final reward inputs:
  - old and some intermediate samples show reward seeing raw `<call_tool ...>` / `<tool_output>` instead of a final `<answer>`
  - example log shows `CitationPrecisionEvaluator Error` followed by raw `<call_tool name="InternalTool">...`
- A second bucket appears in later successful runs:
  - final answers exist, but citations are rendered as bare bracket references like `[11][12][1][6]`
  - these are human-readable references but are **not** counted by the current evaluator, because they are not `<cite id="..."></cite>`
- Latest successful training continues to progress through steps 13-21, so the current issue is not the old step-0 stall anymore.

## Current Assessment

- `citation_len=0` is not primarily caused by a bug inside the numeric counting itself.
- The dominant causes are:
  1. reward sometimes evaluates non-final tool text in some trajectories
  2. even when final answers exist, the model frequently uses `[N]` style references instead of the required XML `<cite id="N"></cite>` tags

## Fix Applied

- Added a pre-reward gate in `open_instruct/scholar_evaluator.py`.
- If a candidate still looks like an intermediate tool trajectory:
  - contains tool-trace markers such as `<call_tool ...>` / `<tool_output ...>`
  - and does not contain exactly one final `<answer>...</answer>` block
  - then scholar reward short-circuits immediately
- For those invalid trajectories:
  - skip content evaluators entirely
  - assign `score = 0.0`
  - record `invalid_candidate_reason`
  - disable format-reward blending so malformed tool traces no longer receive residual reward
- Added unit coverage in `open_instruct/test_scholar_evaluator.py` to verify:
  - invalid tool traces are rejected
  - valid answers are not rejected merely because prior tool traces exist
  - invalid candidates skip async evaluator calls

## Notes

- Related sessions:
  - `debug-rl-missing-answer-reward.md`
  - `debug-rollout-step0-stall.md`
