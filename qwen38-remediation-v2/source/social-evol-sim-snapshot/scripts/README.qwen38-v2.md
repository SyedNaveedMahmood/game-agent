# Corrected Qwen viability suite

Reuse the existing `.venv` and `.local` model/llama.cpp installation. From the repository root, launch a persistent supervisor:

```powershell
$root = (Get-Location).Path
Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File', (Join-Path $root 'scripts/supervise-qwen38-v2.ps1')) -WorkingDirectory $root -WindowStyle Hidden
```

The supervisor starts/reuses only its owned local server, records worker PIDs, and resumes successful decision caches and committed rounds. Run the same command again after a disconnect; a mutex prevents duplicate supervisors. The selected 64K/Q8 configuration was measured on this RTX 3090 before selection. Runtime defaults are preserved in `manifest.json` and `code_manifest.json`.

```powershell
Get-Content data/qwen38_remediation_v2_20261016/progress.json -Raw
Get-Content data/qwen38_remediation_v2_20261016/heartbeat.json -Raw
Get-Content reports/QWEN38_REMEDIATION_AND_VIABILITY_V2.md
.\.venv\Scripts\python.exe -m pytest scr/tests -q
```

Protocol: A has 100 initial paired decisions, with a frozen sign-instability extension rule; B is six actual rounds; C has 24 synthetic C1/C2 history cases and four cases sourced from B; native low is paired with 20 exact OFF baseline cases; longitudinal testing is three seeds × 30 rounds. Both native conditions retain the original two-stage reflection. Deep-memory auditing does not repair memories or add retries. The full 240-round run is excluded.

The finalizer audits coverage and checkpoint/transfer/prompt consistency, runs regression tests, and produces `QWEN38_REMEDIATION_V2_BUNDLE.zip` plus its SHA256 sidecar. Every packaged file is hashed and the entire ZIP is read and checked. The primary report distinguishes completed, failed, blocked, and pending human judgments. Historical artifacts and initial engineering failure traces remain preserved.
