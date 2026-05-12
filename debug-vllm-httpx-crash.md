# [OPEN] vllm-httpx-crash

## Context
- Symptom: training exited during health check with `ray.exceptions.UnserializableException`, while the nested stack showed `ray::LLMRayActor.check_background_threads()` entering `httpx/_transports/default.py` and `httpcore/_async/connection_pool.py`.
- Goal: recover the original truncated `httpx/httpcore` exception from Ray worker logs and determine whether the failure was connection refusal, timeout, protocol reset, or an upstream EngineCore crash.

## Hypotheses
1. `LLMRayActor` failed to connect to the local vLLM OpenAI server on `127.0.0.1:<port>` because the server had already died.
2. The local vLLM server was alive, but the request hung and failed with `ReadTimeout` or a related `httpcore` timeout.
3. `EngineCore` hit an internal error around weight loading/sync first, and the later `httpx` exception is only a downstream symptom.
4. Ray truncated the transport exception during serialization, but the original stack is still present in `worker-*.err/out` or `python-core-worker-*.log`.

## Evidence Plan
- Inspect `/tmp/ray/session_latest/logs` for `worker-*.err`, `worker-*.out`, and `python-core-worker-*.log`.
- Search for `httpx`, `httpcore`, `ConnectError`, `ReadTimeout`, `RemoteProtocolError`, `Connection refused`, `Connection reset`, and `EngineCore`.
- Correlate timestamps around `2026-04-30 02:20:11` to find the first fatal event before the shutdown cascade.

## Status
- In progress: collecting worker log evidence.

## Evidence Collected
- `wandb/run-20260429_185500-fih38y8a/files/output.log` around lines `231183-231240` shows the truncated Ray exception expanded to:
  - `httpcore.ReadError`
  - then `httpx.ReadError`
  - then `openai.APIConnectionError: Connection error.`
- The failing request originates from `open_instruct/vllm_utils.py:1004` inside `process_request`, specifically `await actor.client.completions.create(...)`.
- Immediately before shutdown, the same log shows repeated `EngineCore` warnings at `04-30 02:20:11 [layerwise.py:230] ... Failed to load weights`, ending with `LogitsProcessor: Failed to load weights`, followed by the health-check-triggered shutdown.

## Hypothesis Status
1. `ConnectError/ConnectionRefused`: not supported by current evidence.
2. `ReadTimeout`: not supported by current evidence.
3. `EngineCore` failed first and HTTP was downstream: plausible, but not yet proven.
4. Ray truncated the transport exception but worker-level stack preserved it: confirmed.

## Current Best Explanation
- The immediate fatal exception is a read-path transport failure while the actor was calling its local OpenAI-compatible vLLM server:
  - `httpcore.ReadError` -> `httpx.ReadError` -> `openai.APIConnectionError`.
- This indicates the connection was established enough to enter the response-read phase, but the peer closed or broke the stream before a valid HTTP response completed.
