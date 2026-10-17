# DeepSeek v4 Flash Baseline — Manual Test Results

**Date**: 2026-05-11
**Model**: deepseek-v4-flash
**Provider**: DeepSeek via LiteLLM
**Environment**: local simulation runner
**Config**: configZ_major_v2

## Summary

| # | Test | Result | Key Finding |
|---|------|--------|-------------|
| 1 | 2-life-step smoke test | PASS | Completed 6/6 simulation steps; verified DeepSeek execution after stripping `on_token` |
| 2 | 4-life-step baseline pair | PASS | Both runs completed 12/12 simulation steps with low retry counts |
| 3 | 8-life-step stress test | PASS | Completed 24/24 simulation steps; checkpoint growth became the next bottleneck |

## Critical Findings for Implementation

### 1. MUST strip `on_token` before LiteLLM serialization

The DeepSeek via LiteLLM path ran successfully after the streaming `on_token`
callback was stripped before calling LiteLLM. This avoids passing a Python
callback object into LiteLLM/API serialization.

### 2. Retry summaries must de-duplicate mirrored event-bus retry events

Retry counts should use canonical `type = retry` records and ignore mirrored
event-bus retry events. The corrected summary logic counts canonical retry
records and computes retry rate against unique agent-step decision requests
when available, with a fallback for older logs.

### 3. Prey spawn logging must use the current reward schema

The current hunting reward is HP gain. Prey spawn logs and observations should
use `reward_hp` and `num_agents_to_kill`, not legacy `meat_units` / `nutrition`
fields.

### 4. Raw checkpoint growth is the next bottleneck

The 8-life-step stress test reached 647M of local run data. Before many-seed
experiments, checkpoint growth should be addressed with checkpoint slimming.

---

## Protocol

Baseline command shape:

```bash
uv run python main.py run \
  --config_dir configZ_major_v2 \
  --no_db \
  --config.llm.provider deepseek \
  --config.llm.chat_model deepseek-v4-flash \
  --config.llm.two_stage_model false \
  --config.llm.max_retries 2 \
  --config.llm.async_config.max_concurrent_calls 3 \
  --config.llm.async_config.call_timeout_seconds 120 \
  --config.world.max_life_steps 4
```

With `communication_and_sharing_steps = 2`, each life step expands to three
simulation steps:

```text
life_steps * (2 social steps + 1 production step) = simulation steps
```

Raw checkpoint directories under `data/` are intentionally not committed. They
are large run artifacts and are already ignored by `.gitignore`; this document
keeps the reproducible command shape, run IDs, and compact post-hoc metrics.

---

## Detailed Test Results

### Test 1: 2-Life-Step Smoke Test

The first DeepSeek smoke test completed all 6 simulation steps and confirmed
that the local `deepseek-v4-flash` path was usable after stripping the streaming
`on_token` callback before LiteLLM serialization.

| Run ID | Completed | Final Population | Data Size | Retries | Retry Rate | Timeouts | Stale Actions | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `0511-052122` | 6/6 | 8 | 25M | 0 | 0.00 | 0 | 6 | Smoke test only; verified end-to-end DeepSeek execution. |

**Action counts**:

| Run ID | Communicate | Hunt | Collect | Allocate | Reproduce | Rob | DoNothing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0511-052122` | 29 | 9 | 7 | 1 | 0 | 0 | 2 |

**Result**: PASS

---

### Test 2: 4-Life-Step Baseline Pair

Both runs completed all 12 simulation steps with file-only checkpoints.

| Run ID | Completed | Final Population | Data Size | Decisions | Retries | Retry Rate | Timeouts | Stale Actions | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0511-144201` | 12/12 | 10 | 103M | 102 | 2 | 0.02 | 0 | 17 | Clean completion; retries were one invalid target and one JSON syntax failure. |
| `0511-151214` | 12/12 | 13 | 106M | 102 | 2 | 0.02 | 1 | 12 | Completed despite one timeout and one non-fatal prey spawn logging bug. |

The retry counts above use the corrected summary logic that counts canonical
`type = retry` records, ignores mirrored event-bus retry events, and computes
retry rate against unique agent-step decision requests when available, with a
fallback for older logs.

**Action counts**:

| Run ID | Communicate | Hunt | Collect | Allocate | Reproduce | Rob | DoNothing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0511-144201` | 62 | 19 | 13 | 6 | 2 | 0 | 0 |
| `0511-151214` | 56 | 20 | 6 | 9 | 5 | 2 | 4 |

**Final moral-type counts**:

| Run ID | Universal | Reciprocal | Kin-Focused | Reproductive Selfish |
|---|---:|---:|---:|---:|
| `0511-144201` | 3 | 3 | 2 | 2 |
| `0511-151214` | 4 | 3 | 3 | 3 |

**Final agent states**:

These states are the post-Phase-3 final states inferred from the last saved
Phase-2 checkpoint. The runner emits the final population after Phase 3, while
file checkpoints are written during Phase 2 action application.

`0511-144201`:

```text
agent_1   universal_group_focused_moral    age 14 hp 18
agent_2   universal_group_focused_moral    age 14 hp 32
agent_3   reciprocal_group_focused_moral   age 14 hp 10
agent_4   reciprocal_group_focused_moral   age 14 hp 11
agent_5   kin_focused_moral                age 14 hp 18
agent_6   kin_focused_moral                age 14 hp 24
agent_7   reproductive_selfish             age 14 hp 11
agent_8   reproductive_selfish             age 14 hp 27
agent_3_1 reciprocal_group_focused_moral   age 3  hp 8
agent_1_1 universal_group_focused_moral    age 1  hp 2
```

`0511-151214`:

```text
agent_1   universal_group_focused_moral    age 14 hp 8
agent_2   universal_group_focused_moral    age 14 hp 29
agent_3   reciprocal_group_focused_moral   age 14 hp 27
agent_4   reciprocal_group_focused_moral   age 14 hp 14
agent_5   kin_focused_moral                age 14 hp 16
agent_6   kin_focused_moral                age 14 hp 11
agent_7   reproductive_selfish             age 14 hp 16
agent_8   reproductive_selfish             age 14 hp 9
agent_1_1 universal_group_focused_moral    age 2  hp 8
agent_8_1 reproductive_selfish             age 2  hp 6
agent_2_1 universal_group_focused_moral    age 1  hp 2
agent_3_1 reciprocal_group_focused_moral   age 1  hp 2
agent_6_1 kin_focused_moral                age 1  hp 2
```

**Result**: PASS

---

### Test 3: 8-Life-Step Stress Test

The 8-life-step stress test completed all 24 simulation steps using the same
DeepSeek settings, with `max_life_steps = 8`.

| Run ID | Completed | Final Population | Data Size | Decisions | Retries | Retry Rate | Timeouts | Stale Actions | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `0511-155902` | 24/24 | 14 | 647M | 257 | 3 | 0.012 | 0 | 54 | Clean stress-test completion; checkpoint volume becomes the next bottleneck. |

**Action counts**:

| Run ID | Communicate | Hunt | Collect | Allocate | Reproduce | Rob | DoNothing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0511-155902` | 132 | 29 | 45 | 23 | 7 | 6 | 15 |

**Final moral-type counts**:

| Run ID | Universal | Reciprocal | Kin-Focused | Reproductive Selfish |
|---|---:|---:|---:|---:|
| `0511-155902` | 5 | 2 | 3 | 4 |

**Retry breakdown**:

| Run ID | Validation JSON | Validation Contextual | Root Causes |
|---|---:|---:|---|
| `0511-155902` | 1 | 2 | `json_syntax`: 1; `state_hallucination`: 2 |

**Result**: PASS

---

## Interpretation and Scope

These runs show that the `deepseek-v4-flash` baseline is usable at the systems
level for 2, 4, and 8 life-step runs.

Do not treat these runs as final scientific evidence. Their purpose is to lock
down the baseline protocol before adding new agent architecture changes. They
are systems baselines, not final evidence for the new agent architecture.

The 8-life-step run also shows that raw checkpoint growth becomes a practical
bottleneck. Before many-seed experiments, checkpoint growth / checkpoint
slimming should be addressed.

## Current Status

```text
1. `deepseek-v4-flash` can run 2, 4, and 8 life-step baselines end to end.
2. The retry summary now avoids double counting mirrored event-bus retry events.
3. The prey spawn logging bug caused by legacy `meat_units` / `nutrition` fields is fixed.
4. Raw checkpoints grow quickly, so checkpoint slimming should happen before many-seed experiments.
5. These runs are systems baselines, not final evidence for the new agent architecture.
```
