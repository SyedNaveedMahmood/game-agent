# Qwen3.8-27B remediation and viability evidence

This folder contains the clean source snapshot under source/social-evol-sim-snapshot/ and auditable results from the local Qwen3.8-27B Q4_K_M evaluation. The source snapshot is based on `MoralAgentSim/social-evol-sim` commit `4863257c941c17ecf7556a1b2947c91400df8a47` with the working-tree remediation changes applied. It excludes the virtual environment, llama.cpp binaries, model weights, and bulky raw data.

## State at pause

The user requested an immediate pause during longitudinal environment seed `20261017`, round 30. Rounds 1?29 are committed. Round 30 is uncommitted: nine decision artifacts are complete, one first-stage request is saved without a response, and no round-30 after-state or commit exists. Seeds `20261019` and `20261023` were never started. The report labels this as an interim hold, not a completed 30-round result.

The compact result set includes the main report, 29 per-round result files, the round-30 before-state/RNG and partial decision ledgers, metrics, audits, regression log, and blinded pass-one/pass-two human-review CSVs. The unblinding key is intentionally not in this GitHub folder.

## Full raw evidence

All raw prompts, responses, checkpoints, telemetry, logs, and historical evidence are in `QWEN38_REMEDIATION_V2_EVIDENCE_PAUSED_20261017.zip`, delivered separately because its 232,455,203-byte size exceeds GitHub's single-file upload limit. Its SHA-256 is recorded in `results/archive_manifest_paused_20261017.json` and the accompanying `.sha256` file. The archive contains the directory `qwen38_remediation_v2_20261016/`; extract its contents into the `data/` directory of the source snapshot to restore its expected location.

## Verification

`results/tests/regression_paused_push_20261017.log`: 119 passed, 2 existing Pydantic deprecation warnings. No external LLM API was used. Human moral ratings remain pending.
