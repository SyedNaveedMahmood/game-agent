$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$runDir = Join-Path $root 'data/qwen38_full_baseline_20261016'
$telemetry = Join-Path $runDir 'telemetry'
$python = Join-Path $root '.venv/Scripts/python.exe'
$finalizer = Join-Path $PSScriptRoot 'finalize_qwen38_full_baseline.py'
$log = Join-Path $telemetry 'finalizer_watcher.jsonl'

while ($true) {
    $state = $null
    $statePath = Join-Path $runDir 'progress.json'
    if (Test-Path -LiteralPath $statePath) {
        try { $state = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json } catch { }
    }
    $supervisorBlocker = Test-Path -LiteralPath (Join-Path $runDir 'supervisor_blocker.json')
    if (($state -and $state.status -in @('complete','blocked')) -or $supervisorBlocker) {
        $record = @{ at = [DateTime]::UtcNow.ToString('o'); event = 'finalizer_start'; status = if ($state) { $state.status } else { 'supervisor_blocked' } }
        ($record | ConvertTo-Json -Compress) | Add-Content -LiteralPath $log -Encoding utf8
        $proc = Start-Process -FilePath $python -ArgumentList @($finalizer) -WorkingDirectory $root `
            -RedirectStandardOutput (Join-Path $telemetry 'finalizer.stdout.log') `
            -RedirectStandardError (Join-Path $telemetry 'finalizer.stderr.log') -WindowStyle Hidden -PassThru -Wait
        $proc.Refresh()
        $done = @{ at = [DateTime]::UtcNow.ToString('o'); event = 'finalizer_exit'; exit_code = $proc.ExitCode }
        ($done | ConvertTo-Json -Compress) | Add-Content -LiteralPath $log -Encoding utf8
        exit $proc.ExitCode
    }
    Start-Sleep -Seconds 60
}
