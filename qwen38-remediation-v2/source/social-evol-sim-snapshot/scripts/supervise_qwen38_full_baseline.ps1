$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$runDir = Join-Path $root 'data/qwen38_full_baseline_20261016'
$telemetry = Join-Path $runDir 'telemetry'
$python = Join-Path $root '.venv/Scripts/python.exe'
$runner = Join-Path $PSScriptRoot 'run_qwen38_full_baseline.py'
$lifecycle = Join-Path $telemetry 'supervisor_lifecycle.jsonl'
New-Item -ItemType Directory -Force -Path $telemetry | Out-Null
$env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
$env:DO_NOT_TRACK = '1'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
$backoff = @(5, 10, 20, 40, 60)

function Write-SupervisorEvent([hashtable]$record) {
    $record['at'] = [DateTime]::UtcNow.ToString('o')
    ($record | ConvertTo-Json -Compress -Depth 12) | Add-Content -LiteralPath $lifecycle -Encoding utf8
}

function Read-RunState {
    $path = Join-Path $runDir 'progress.json'
    if (-not (Test-Path -LiteralPath $path)) { return $null }
    try { return Get-Content -Raw -LiteralPath $path | ConvertFrom-Json }
    catch { return $null }
}

for ($attempt = 1; $attempt -le 6; $attempt++) {
    $state = Read-RunState
    if ($state -and $state.status -eq 'complete') {
        Write-SupervisorEvent @{ event = 'already_complete'; ticks = $state.completed_ticks }
        exit 0
    }
    if ($state -and $state.status -eq 'blocked') {
        Write-SupervisorEvent @{ event = 'persistent_blocker'; tick = $state.current_tick; blocker = $state.blocker }
        exit 2
    }

    $stdout = Join-Path $telemetry "runner_attempt_$attempt.stdout.log"
    $stderr = Join-Path $telemetry "runner_attempt_$attempt.stderr.log"
    Write-SupervisorEvent @{ event = 'runner_start'; attempt = $attempt }
    try {
        $proc = Start-Process -FilePath $python -ArgumentList @($runner) -WorkingDirectory $root `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru -Wait
        $proc.Refresh()
        $exitCode = $proc.ExitCode
        Set-Content -LiteralPath (Join-Path $runDir 'runner.pid') -Value $proc.Id -Encoding ascii
        Write-SupervisorEvent @{ event = 'runner_exit'; attempt = $attempt; pid = $proc.Id; exit_code = $exitCode }
    }
    catch {
        $exitCode = 1
        Write-SupervisorEvent @{ event = 'runner_launch_error'; attempt = $attempt; error = $_.Exception.Message }
    }

    $state = Read-RunState
    if ($state -and $state.status -eq 'complete') { exit 0 }
    if ($state -and $state.status -eq 'blocked') { exit 2 }
    if ($exitCode -eq 0) { exit 0 }
    if ($attempt -lt 6) { Start-Sleep -Seconds $backoff[$attempt - 1] }
}

$state = Read-RunState
$blocker = @{
    status = 'blocked_after_five_process_recoveries'
    at = [DateTime]::UtcNow.ToString('o')
    attempts = 6
    process_recoveries = 5
    current_tick = if ($state) { $state.current_tick } else { $null }
    last_error = if ($state) { $state.last_error } else { 'The run process exited unsuccessfully six times.' }
    progress_path = Join-Path $runDir 'progress.json'
    supervisor_log = $lifecycle
}
$blocker | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $runDir 'supervisor_blocker.json') -Encoding utf8
Write-SupervisorEvent @{ event = 'blocked_after_process_recoveries'; attempt = 6 }
exit 1
