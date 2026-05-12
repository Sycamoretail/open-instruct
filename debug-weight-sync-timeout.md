[OPEN] weight-sync-timeout

# Debug Session

- Session ID: `weight-sync-timeout`
- Symptom: training exits with `RuntimeError: Weight sync timed out after ... - vLLM engines may be stuck`
- Goal: determine whether the repeated timeout is caused by slow-but-finite weight sync, rollout draining backlog, or a systematic stall in vLLM / actor coordination

# Hypotheses

1. The timeout is primarily caused by rollout requests still running when weight sync starts, so `update_weights()` spends most of the time draining active tasks.
2. The 6-learner / 2-vLLM configuration increases concurrent in-flight generation enough that the fixed timeout is frequently exceeded even without a hard deadlock.
3. One or more vLLM engines are slower than the others during `sleep -> update_weights -> wake_up`, causing tail latency to dominate the sync wall time.
4. The actor stop/resume handshake (`should_stop`) is delayed, so the main thread times out waiting for actors to pause even though engine-side weight transfer later completes.
5. Query mix or response length has become heavy enough that weight sync collides with long-running generations, making timeouts workload-dependent rather than purely infrastructure-related.

# Evidence Plan

- Inspect current code paths for timeout, actor pause/resume, and engine update drain behavior.
- Inspect latest training / Ray logs around timeout windows to compare query activity, engine behavior, and final sync durations.
- Determine whether the failure pattern is "always one slow engine", "all engines busy draining", or "main thread timeout threshold too tight".
