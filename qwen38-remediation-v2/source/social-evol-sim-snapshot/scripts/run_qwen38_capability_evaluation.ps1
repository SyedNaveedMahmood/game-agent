$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$evalDir = Join-Path $root 'data/qwen38_capability_evaluation_20261016'
$telemetry = Join-Path $evalDir 'telemetry'
$supervisor = Join-Path $PSScriptRoot 'supervise_qwen38_capability_evaluation.ps1'
New-Item -ItemType Directory -Force -Path $telemetry | Out-Null

$runnerProcess = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -and $_.CommandLine.Contains('scripts/qwen38_capability_evaluation.py')
}
$supervisorProcess = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'powershell.exe' -and $_.CommandLine -and $_.CommandLine.Contains('-File ' + $supervisor)
}
if ($runnerProcess -or $supervisorProcess) {
    @($runnerProcess) + @($supervisorProcess) | Where-Object { $_ } | Select-Object ProcessId, Name, CommandLine | Format-List
    Write-Output 'An evaluation supervisor or runner is already active; the journaled run will continue.'
    exit 0
}

$supervisorOut = Join-Path $telemetry 'supervisor.stdout.log'
$supervisorErr = Join-Path $telemetry 'supervisor.stderr.log'
$proc = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $supervisor) `
    -WorkingDirectory $root -RedirectStandardOutput $supervisorOut `
    -RedirectStandardError $supervisorErr -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $evalDir 'supervisor.pid') -Value $proc.Id -Encoding ascii
$launch = @{
    launched_at = [DateTime]::UtcNow.ToString('o')
    supervisor_pid = $proc.Id
    supervisor_script = $supervisor
    report = Join-Path $root 'reports/QWEN38_CAPABILITY_EVALUATION.md'
    heartbeat = Join-Path $evalDir 'heartbeat.json'
}
$launch | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $evalDir 'launch.json') -Encoding utf8
Write-Output "Started hidden resumable evaluation supervisor PID $($proc.Id)."
Write-Output "Progress: $(Join-Path $evalDir 'heartbeat.json')"
Write-Output "Report: $(Join-Path $root 'reports/QWEN38_CAPABILITY_EVALUATION.md')"
