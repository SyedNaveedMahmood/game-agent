$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$runDir = Join-Path $root 'data/qwen38_full_baseline_20261016'
$telemetry = Join-Path $runDir 'telemetry'
$watcher = Join-Path $PSScriptRoot 'watch_qwen38_full_baseline_finalizer.ps1'
New-Item -ItemType Directory -Force -Path $telemetry | Out-Null
$active = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'powershell.exe' -and $_.CommandLine -and $_.CommandLine.Contains('-File ' + $watcher)
}
if ($active) {
    Write-Output "Finalizer watcher already active (PID $($active.ProcessId))."
    exit 0
}
$stdout = Join-Path $telemetry 'finalizer_watcher.stdout.log'
$stderr = Join-Path $telemetry 'finalizer_watcher.stderr.log'
$proc = Start-Process -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$watcher) `
    -WorkingDirectory $root -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
    -WindowStyle Hidden -PassThru
Set-Content -LiteralPath (Join-Path $runDir 'finalizer_watcher.pid') -Value $proc.Id -Encoding ascii
Write-Output "Started finalizer watcher PID $($proc.Id). It will audit and update the report when the run stops."
