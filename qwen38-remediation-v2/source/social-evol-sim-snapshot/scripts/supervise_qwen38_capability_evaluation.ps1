$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$evalDir = Join-Path $root 'data/qwen38_capability_evaluation_20261016'
$telemetry = Join-Path $evalDir 'telemetry'
$python = Join-Path $root '.venv/Scripts/python.exe'
$runner = Join-Path $root 'scripts/qwen38_capability_evaluation.py'
$finalizer = Join-Path $root 'scripts/finalize_qwen38_capability_evaluation.py'
New-Item -ItemType Directory -Force -Path $telemetry | Out-Null
$env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
$env:DO_NOT_TRACK = '1'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$backoff = @(5, 10, 20, 40, 60)
$lifecycle = Join-Path $telemetry 'supervisor_lifecycle.jsonl'

function Write-SupervisorEvent([hashtable]$record) {
    $record['at'] = [DateTime]::UtcNow.ToString('o')
    ($record | ConvertTo-Json -Compress -Depth 12) | Add-Content -LiteralPath $lifecycle -Encoding utf8
}

function Get-EvaluationCounts {
    $statePath = Join-Path $evalDir 'progress.json'
    if (-not (Test-Path -LiteralPath $statePath)) { return $null }
    try { $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json }
    catch { return $null }
    $aDone = @($state.A.PSObject.Properties | Where-Object { $_.Value.status -in @('complete', 'blocked') }).Count
    $bDone = @($state.B.rounds.PSObject.Properties | Where-Object { $_.Value.status -like 'complete*' -or $_.Value.status -like 'blocked*' }).Count
    $cDone = @($state.C.PSObject.Properties | Where-Object { $_.Value.status -in @('complete', 'blocked') }).Count
    return @{ state = $state; A = $aDone; B = $bDone; C = $cDone; all_resolved = ($aDone -eq 40 -and $bDone -eq 6 -and $cDone -eq 8) }
}

function Finalize-ResolvedEvaluation($counts) {
    Write-SupervisorEvent @{ event = 'offline_finalization_start'; A = $counts.A; B = $counts.B; C = $counts.C }
    $finalize = Start-Process -FilePath $python -ArgumentList @($finalizer) -WorkingDirectory $root `
        -RedirectStandardOutput (Join-Path $telemetry 'offline_finalizer.stdout.log') `
        -RedirectStandardError (Join-Path $telemetry 'offline_finalizer.stderr.log') -WindowStyle Hidden -PassThru -Wait
    $finalize.Refresh()
    Write-SupervisorEvent @{ event = 'offline_finalization_exit'; exit_code = $finalize.ExitCode }
    return $finalize.ExitCode -eq 0
}

$initialCounts = Get-EvaluationCounts
if ($initialCounts -and $initialCounts.all_resolved) {
    if ($initialCounts.state.status -notin @('complete', 'complete_with_blockers', 'complete_with_integrity_warnings', 'blocked_with_unresolved_units')) {
        if (-not (Finalize-ResolvedEvaluation $initialCounts)) { exit 1 }
    }
    exit 0
}

for ($attempt = 1; $attempt -le 6; $attempt++) {
    $stdout = Join-Path $telemetry "runner_attempt_$attempt.stdout.log"
    $stderr = Join-Path $telemetry "runner_attempt_$attempt.stderr.log"
    Write-SupervisorEvent @{ event = 'runner_start'; attempt = $attempt }
    $proc = Start-Process -FilePath $python -ArgumentList @($runner) -WorkingDirectory $root `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
    Set-Content -LiteralPath (Join-Path $evalDir 'runner.pid') -Value $proc.Id -Encoding ascii
    $proc.Refresh()
    $exitCode = $proc.ExitCode
    Write-SupervisorEvent @{ event = 'runner_exit'; attempt = $attempt; pid = $proc.Id; exit_code = $exitCode }
    $counts = Get-EvaluationCounts
    if ($counts -and $counts.all_resolved) {
        if ($counts.state.status -notin @('complete', 'complete_with_blockers', 'complete_with_integrity_warnings', 'blocked_with_unresolved_units')) {
            if (-not (Finalize-ResolvedEvaluation $counts)) { exit 1 }
        }
        exit 0
    }
    if ($exitCode -eq 0) { exit 0 }
    if ($attempt -lt 6) { Start-Sleep -Seconds $backoff[$attempt - 1] }
}

$statePath = Join-Path $evalDir 'progress.json'
$state = if (Test-Path -LiteralPath $statePath) { Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json } else { $null }
$blocker = @{
    status = 'blocked_after_five_process_recoveries'
    at = [DateTime]::UtcNow().ToString('o')
    attempts = 6
    process_recoveries = 5
    phase = if ($state) { $state.phase } else { 'runner_startup' }
    current_unit = if ($state) { $state.current_unit } else { $null }
    last_error = if ($state) { $state.last_error } else { 'The evaluation process exited unsuccessfully five times.' }
    progress_path = $statePath
    supervisor_log = $lifecycle
}
$blocker | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $evalDir 'supervisor_blocker.json') -Encoding utf8
Write-SupervisorEvent @{ event = 'blocked'; attempt = 5 }
exit 1
