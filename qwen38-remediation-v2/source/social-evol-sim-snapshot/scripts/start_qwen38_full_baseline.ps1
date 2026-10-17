$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$runDir = Join-Path $root 'data/qwen38_full_baseline_20261016'
$telemetry = Join-Path $runDir 'telemetry'
$supervisor = Join-Path $PSScriptRoot 'supervise_qwen38_full_baseline.ps1'
New-Item -ItemType Directory -Force -Path $telemetry | Out-Null

$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'powershell.exe' -and $_.CommandLine -and $_.CommandLine.Contains('-File ' + $supervisor)
}
if ($active) {
    $active | Select-Object ProcessId, Name, CommandLine | Format-List
    Write-Output 'The full-baseline supervisor is already running.'
    exit 0
}

$stdout = Join-Path $telemetry 'supervisor.stdout.log'
$stderr = Join-Path $telemetry 'supervisor.stderr.log'
$proc = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $supervisor) `
    -WorkingDirectory $root -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $runDir 'supervisor.pid') -Value $proc.Id -Encoding ascii
$launch = @{
    launched_at = [DateTime]::UtcNow.ToString('o')
    supervisor_pid = $proc.Id
    supervisor_script = $supervisor
    runner_script = Join-Path $PSScriptRoot 'run_qwen38_full_baseline.py'
    report = Join-Path $root 'reports/QWEN38_CAPABILITY_EVALUATION.md'
    progress = Join-Path $runDir 'progress.json'
    heartbeat = Join-Path $runDir 'heartbeat.json'
}
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runDir 'launch.json') -Encoding utf8
Write-Output "Started hidden full-baseline supervisor PID $($proc.Id)."
Write-Output "Progress: $(Join-Path $runDir 'heartbeat.json')"
Write-Output "Report: $(Join-Path $root 'reports/QWEN38_CAPABILITY_EVALUATION.md')"
