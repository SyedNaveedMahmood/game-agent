param([string]$EvidenceDirectory='data/qwen38_remediation_v2_20261016')
$ErrorActionPreference='Continue'
$root=Split-Path $PSScriptRoot -Parent
$evidence=Join-Path $root $EvidenceDirectory
$telemetry=Join-Path $evidence 'telemetry'
New-Item -ItemType Directory -Force -Path $telemetry | Out-Null
$mutex=New-Object Threading.Mutex($false,'Local\Qwen38RemediationV2Supervisor')
if(-not $mutex.WaitOne(0)){Write-Output 'An existing V2 supervisor owns the evaluation.';exit 0}
function Read-State([string]$Name){try{return Get-Content -LiteralPath (Join-Path $evidence $Name) -Raw -Encoding UTF8 | ConvertFrom-Json}catch{return $null}}
function Event([hashtable]$Record){$Record.at=[DateTime]::UtcNow.ToString('o');($Record|ConvertTo-Json -Compress -Depth 12)|Add-Content -LiteralPath (Join-Path $telemetry 'supervisor.jsonl') -Encoding UTF8}
try{
    @{pid=$PID;started_at=[DateTime]::UtcNow.ToString('o');script=$PSCommandPath}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $evidence 'supervisor.json') -Encoding UTF8
    $env:PYTHONUTF8='1';$env:PYTHONUNBUFFERED='1';$env:LITELLM_LOCAL_MODEL_COST_MAP='True';$env:HF_HUB_OFFLINE='1';$env:DO_NOT_TRACK='1'
    $runner=Join-Path $root 'scripts/qwen38_remediation_v2.py'
    $python=Join-Path $root '.venv/Scripts/python.exe'
    for($attempt=0;$attempt -le 5;$attempt++){
        $state=Read-State 'progress.json';$verified=Read-State 'archive_verification.json'
        if($state.status -in @('complete','complete_with_documented_blockers') -and $verified.verified){Event @{event='already_complete'};exit 0}
        $stamp=[DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
        $proc=Start-Process -FilePath $python -ArgumentList @($runner) -WorkingDirectory $root -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $telemetry "runner-$stamp.stdout.log") `
            -RedirectStandardError (Join-Path $telemetry "runner-$stamp.stderr.log")
        @{pid=$proc.Id;attempt=$attempt;started_at=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $evidence 'runner_process.json') -Encoding UTF8
        Event @{event='runner_started';pid=$proc.Id;recovery_attempt=$attempt}
        while(-not $proc.HasExited){
            $state=Read-State 'progress.json';$heartbeat=Read-State 'heartbeat.json'
            $health=$null;try{$health=Invoke-RestMethod 'http://127.0.0.1:8080/health' -TimeoutSec 5}catch{}
            Event @{event='supervisor_heartbeat';runner_pid=$proc.Id;phase=$state.phase;unit=$state.current_unit;runner_status=$state.status;worker_heartbeat=$heartbeat.at;server_health=$health.status}
            # The worker's thread records token progress and owns server recovery.
            # A live worker doing slow inference is never killed for request age.
            Start-Sleep -Seconds 15
            $proc.Refresh()
        }
        $proc.WaitForExit();Event @{event='runner_exited';pid=$proc.Id;exit_code=$proc.ExitCode;recovery_attempt=$attempt}
        $verified=Read-State 'archive_verification.json';$state=Read-State 'progress.json'
        if($proc.ExitCode -eq 0 -and $state.status -in @('complete','complete_with_documented_blockers') -and $verified.verified){Event @{event='suite_finished_and_archive_verified'};exit 0}
        if($attempt -lt 5){Start-Sleep -Seconds ([math]::Min(60,5*[math]::Pow(2,$attempt)))}
    }
    $state=Read-State 'progress.json'
    @{at=[DateTime]::UtcNow.ToString('o');status='persistent_supervisor_blocker';recoveries=5;last_error=$state.last_error;traceback=$state.last_traceback}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $evidence 'supervisor_blocker.json') -Encoding UTF8
    Event @{event='persistent_blocker';error=$state.last_error}
    & $python $runner --seal-supervisor-blocker
    exit $LASTEXITCODE
}finally{$mutex.ReleaseMutex();$mutex.Dispose()}
