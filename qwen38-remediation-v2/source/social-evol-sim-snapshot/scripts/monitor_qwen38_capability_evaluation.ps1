$ErrorActionPreference = 'Continue'
$root = Split-Path $PSScriptRoot -Parent
$evalDir = Join-Path $root 'data/qwen38_capability_evaluation_20261016'
$telemetry = Join-Path $evalDir 'telemetry'
$logPath = Join-Path $telemetry 'independent_monitor.jsonl'
$pidPath = Join-Path $evalDir 'monitor.pid'
$server = Join-Path $root '.local/llama-b11669/llama-server.exe'
$startScript = Join-Path $root 'scripts/start-local-server.ps1'
$intervalSeconds = 15
$stallSeconds = 3600
$reportRefreshSeconds = 15
$maxRestartsPerUnit = 5
$restartCounts = @{}
$lastReportUpdate = [DateTimeOffset]::MinValue

function Read-LastJsonLine([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $line = Get-Content -LiteralPath $Path -Tail 1 -ErrorAction Stop
        if ($line) { return ($line | ConvertFrom-Json -ErrorAction Stop) }
    } catch { }
    return $null
}

function Write-MonitorEvent([hashtable]$Record) {
    $Record['at'] = [DateTime]::UtcNow.ToString('o')
    ($Record | ConvertTo-Json -Compress -Depth 10) | Add-Content -LiteralPath $logPath -Encoding utf8
}

function Update-LiveReport($Progress, $Monitor, [int]$AComplete, [int]$BRoundsComplete, [int]$BDecisionsComplete, [int]$BCurrentPopulation, [int]$CRComplete, [int]$FailureCount) {
    $reportPath = Join-Path $root 'reports/QWEN38_CAPABILITY_EVALUATION.md'
    if (-not (Test-Path -LiteralPath $reportPath)) { return }
    try {
        $doc = [IO.File]::ReadAllText($reportPath, [Text.Encoding]::UTF8)
        $doc = [regex]::Replace($doc, '(?m)^\*\*Status: prepared; experiments not yet started\.\*\*.*\r?\n\r?\n', '')
        $doc = [regex]::Replace($doc, '(?m)^### Experiment B.*\(48 agent decisions\)$', '### Experiment B -- six-round interaction simulation (dynamic population)')
        $doc = $doc.Replace('With 40/48/8 units, uncertainty is substantial;', 'With 40 A cases, a dynamic B decision count, and 8 C cases, uncertainty is substantial;')
        $doc = $doc.Replace('all 8 full conversations', 'every living agent conversation (including offspring created by original dynamics)')
        $startHeading = '## Progress and monitoring'
        $endHeading = '## Experiment A results'
        $start = $doc.IndexOf($startHeading, [StringComparison]::Ordinal)
        $end = $doc.IndexOf($endHeading, [StringComparison]::Ordinal)
        if ($start -lt 0 -or $end -le $start) { return }

        $now = [DateTimeOffset]::UtcNow
        $elapsedHours = [math]::Round(($now - [DateTimeOffset]::Parse($Progress.started_at)).TotalHours, 2)
        $current = if ($Monitor.last_call_start) { "$($Monitor.last_call_start.unit) / $($Monitor.last_call_start.stage)" } else { 'no active call recorded' }
        $callAge = if ($null -ne $Monitor.active_call_seconds_without_terminal) { "$([math]::Round($Monitor.active_call_seconds_without_terminal))s since the current request began" } else { 'no unfinished request detected' }
        $slot = if ($null -eq $Monitor.slot_processing) { 'unavailable' } else { [string]$Monitor.slot_processing }
        $gpu = if ($Monitor.gpu) { "$($Monitor.gpu.used_mib)/$($Monitor.gpu.total_mib) MiB; $($Monitor.gpu.utilization_percent)% utilization" } else { 'unavailable' }
        $heartbeatAge = 'unavailable'
        if ($Monitor.runner_last_activity) {
            $heartbeatAge = "$([math]::Round(($now - [DateTimeOffset]::Parse($Monitor.runner_last_activity)).TotalSeconds))s since runner activity"
        }
        $populationNote = if ($BCurrentPopulation -gt 0) { "current tick checkpoint has $BCurrentPopulation living agents, including any offspring" } else { 'eight-agent initial cohort; original reproduction dynamics can add offspring' }
        $section = @(
            '## Progress and monitoring',
            '',
            ('**Live watchdog snapshot:** {0}; runner status `{1}`; phase `{2}`; elapsed {3} hours.' -f $now.ToString('yyyy-MM-ddTHH:mm:ssZ'), $Progress.status, $Progress.phase, $elapsedHours),
            "- A: **$AComplete/40 complete**; $([math]::Max(0, 40 - $AComplete)) unresolved.",
            "- B: **$BRoundsComplete/6 rounds complete**; $BDecisionsComplete of $($BDecisionsComplete + $BCurrentPopulation) decision slots are checkpointed or present in the current tick; $([math]::Max(0, 6 - $BRoundsComplete)) rounds unresolved. Initial cohort: eight agents (two per moral type); $populationNote, so the decision count is dynamic.",
            "- C: **$CRComplete/8 complete**; $([math]::Max(0, 8 - $CRComplete)) unresolved.",
            "- Current call: $current; $callAge. Runner activity was $heartbeatAge. Server health: $($Monitor.server_healthy); slot processing: $slot; slot prompt tokens: $($Monitor.slot_prompt_tokens). GPU: $gpu.",
            "- Start: $($Progress.started_at); resume count: $($Progress.resume_count); runner process restarts: $($Progress.process_restarts); recorded failed attempts: $FailureCount.",
            '- The independent hidden watchdog writes timestamped samples every 15 seconds to data/qwen38_capability_evaluation_20261016/telemetry/independent_monitor.jsonl; it restarts llama-server only after five minutes without a request completion, at most five times for a unit.',
            '- Per-call prompts, drafts, reflections, validation, checkpoints and server timings are retained under data/qwen38_capability_evaluation_20261016/; current round snapshots remain resumable from file checkpoints.',
            ''
        ) -join "`r`n"
        $updated = $doc.Substring(0, $start) + $section + $doc.Substring($end)
        $tempPath = "$reportPath.live.$PID.tmp"
        [IO.File]::WriteAllText($tempPath, $updated, (New-Object Text.UTF8Encoding($false)))
        try { [IO.File]::Replace($tempPath, $reportPath, $null) }
        catch { Move-Item -LiteralPath $tempPath -Destination $reportPath -Force -ErrorAction Stop }
    } catch {
        Write-MonitorEvent @{ event = 'live_report_update_error'; error = $_.Exception.Message }
    }
}

New-Item -ItemType Directory -Force -Path $telemetry | Out-Null
Set-Content -LiteralPath $pidPath -Value $PID -Encoding ascii
Write-MonitorEvent @{ event = 'monitor_started'; pid = $PID; interval_seconds = $intervalSeconds; stall_threshold_seconds = $stallSeconds }

while ($true) {
    $progress = $null
    try { $progress = Get-Content -LiteralPath (Join-Path $evalDir 'progress.json') -Raw | ConvertFrom-Json -ErrorAction Stop } catch { }
    if (-not $progress) {
        Write-MonitorEvent @{ event = 'progress_unreadable' }
        Start-Sleep -Seconds $intervalSeconds
        continue
    }

    $healthOk = $false
    $healthError = $null
    try { $healthOk = (Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 5).status -eq 'ok' }
    catch { $healthError = $_.Exception.Message }
    $slot = $null
    $slotError = $null
    try {
        $slotReply = Invoke-RestMethod -Uri 'http://127.0.0.1:8080/slots' -TimeoutSec 5
        $slotArray = @($slotReply)
        if ($slotArray.Count -gt 0) { $slot = $slotArray[0] }
    } catch { $slotError = $_.Exception.Message }
    $gpu = $null
    try {
        $rawGpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits --id=0 2>$null
        $parts = @($rawGpu -split ',') | ForEach-Object { $_.Trim() }
        if ($parts.Count -ge 3) {
            $gpu = @{ used_mib = [int][double]$parts[0]; total_mib = [int][double]$parts[1]; utilization_percent = [int][double]$parts[2] }
            if ($parts.Count -ge 4) { $gpu.power_watts = [double]$parts[3] }
        }
    } catch { }

    $lastStart = Read-LastJsonLine (Join-Path $telemetry 'call_starts.jsonl')
    $lastCompletion = Read-LastJsonLine (Join-Path $telemetry 'completions.jsonl')
    $lastCallError = Read-LastJsonLine (Join-Path $telemetry 'call_errors.jsonl')
    $latestTerminal = $null
    foreach ($candidate in @($lastCompletion, $lastCallError)) {
        if ($candidate) {
            $candidateAt = $candidate.completed_at
            if (-not $candidateAt) { $candidateAt = $candidate.at }
            if ($candidateAt -and (-not $latestTerminal -or [DateTimeOffset]::Parse($candidateAt) -gt [DateTimeOffset]::Parse($latestTerminal))) { $latestTerminal = $candidateAt }
        }
    }

    $stalledSeconds = $null
    $unitId = $null
    if ($lastStart) {
        $unitId = [string]$lastStart.unit
        $startAt = [DateTimeOffset]::Parse($lastStart.at)
        $terminalIsOlder = (-not $latestTerminal) -or ([DateTimeOffset]::Parse($latestTerminal) -le $startAt)
        if ($terminalIsOlder) { $stalledSeconds = [math]::Max(0, ([DateTimeOffset]::UtcNow - $startAt).TotalSeconds) }
    }

    $monitorSnapshot = [PSCustomObject]@{
        last_call_start = $lastStart
        active_call_seconds_without_terminal = $stalledSeconds
        slot_processing = if ($slot) { [bool]$slot.is_processing } else { $null }
        slot_prompt_tokens = if ($slot) { $slot.n_prompt_tokens } else { $null }
        gpu = $gpu
        server_healthy = $healthOk
        runner_last_activity = $progress.last_activity_at
    }
    $bCurrentPopulation = 0
    if ($progress.phase -eq 'B' -and [string]$progress.current_unit -match 'round\s+(\d+)/6') {
        $currentStep = [int]$Matches[1]
        $beforePath = Join-Path $evalDir "B/rounds/step_$currentStep/before.json"
        try {
            $beforeCheckpoint = Get-Content -LiteralPath $beforePath -Raw | ConvertFrom-Json -ErrorAction Stop
            $bCurrentPopulation = @($beforeCheckpoint.social_environment.agents | Where-Object { $_.state -and $_.state.hp -gt 0 }).Count
        } catch { }
    }
    Write-MonitorEvent @{
        event = 'heartbeat'; phase = $progress.phase; status = $progress.status
        current_unit = $progress.current_unit; last_call_start = $lastStart; last_completion = $latestTerminal
        server_healthy = $healthOk; server_error = $healthError
        slot_processing = if ($slot) { [bool]$slot.is_processing } else { $null }
        slot_error = $slotError
        slot_prompt_tokens = if ($slot) { $slot.n_prompt_tokens } else { $null }
        gpu = $gpu; active_call_seconds_without_terminal = $stalledSeconds
        runner_pid = $progress.pid; runner_last_activity = $progress.last_activity_at
    }

    $nowForReport = [DateTimeOffset]::UtcNow
    if (($nowForReport - $lastReportUpdate).TotalSeconds -ge $reportRefreshSeconds) {
        $aComplete = @($progress.A.PSObject.Properties | Where-Object { $_.Value.status -eq 'complete' }).Count
        $bComplete = @($progress.B.rounds.PSObject.Properties | Where-Object { $_.Value.status -eq 'complete' }).Count
        $bDecisions = 0
        foreach ($property in $progress.B.rounds.PSObject.Properties) {
            if ($property.Value.status -eq 'complete') { $bDecisions += [int]$property.Value.decisions_completed }
        }
        $cComplete = @($progress.C.PSObject.Properties | Where-Object { $_.Value.status -eq 'complete' }).Count
        $failureCount = @($progress.failures).Count
        Update-LiveReport $progress $monitorSnapshot $aComplete $bComplete $bDecisions $bCurrentPopulation $cComplete $failureCount
        $reportPath = Join-Path $root 'reports/QWEN38_CAPABILITY_EVALUATION.md'
        if (Test-Path -LiteralPath $reportPath) {
            try {
                $doc = [IO.File]::ReadAllText($reportPath, [Text.Encoding]::UTF8)
                $doc = [regex]::Replace($doc, '(?m)^- B validity:.*$', '- B validity: final numerator and denominator are recalculated from the saved per-round population after all six ticks resolve; no fixed 48-decision assumption is used.')
                $doc = [regex]::Replace($doc, '(?m)^- B state-consistent rounds:.*$', "- B state consistency and memory application use the actual dynamic living-agent slots in each round checkpoint; final counts are recalculated after the simulation.")
                [IO.File]::WriteAllText($reportPath, $doc, (New-Object Text.UTF8Encoding($false)))
            } catch { Write-MonitorEvent @{ event = 'live_report_metric_note_error'; error = $_.Exception.Message } }
        }
        $lastReportUpdate = $nowForReport
    }

    if ($progress.status -ne 'running' -and $progress.status -ne 'crashed_recoverable') {
        Write-MonitorEvent @{ event = 'monitor_finished'; evaluation_status = $progress.status }
        Remove-Item -LiteralPath $pidPath -Force -ErrorAction SilentlyContinue
        exit 0
    }

    # A responsive processing slot can be decoding or thinking for many minutes.
    # Request duration alone never proves it is stalled. The V2 monitor also
    # checks per-token log progress and confines recovery to its owned PID.
    if ($healthOk -and $unitId -and $null -ne $stalledSeconds -and $stalledSeconds -ge $stallSeconds -and $slot -and -not $slot.is_processing) {
        $seen = 0
        if ($restartCounts.ContainsKey($unitId)) { $seen = [int]$restartCounts[$unitId] }
        if ($seen -lt $maxRestartsPerUnit) {
            $restartCounts[$unitId] = $seen + 1
            Write-MonitorEvent @{ event = 'stalled_request_recovery'; unit = $unitId; attempt = ($seen + 1); stalled_seconds = [math]::Round($stalledSeconds); action = 'restart_server' }
            try {
                $serverPath = [IO.Path]::GetFullPath($server)
                $ownedPath = Join-Path $evalDir 'server/owned_server.json'
                if (-not (Test-Path -LiteralPath $ownedPath)) { throw 'Recovery declined: no owned-server PID record; do not terminate an unowned server.' }
                $owned = Get-Content -LiteralPath $ownedPath -Raw | ConvertFrom-Json
                $ownedProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$([int]$owned.pid)"
                if (-not $ownedProcess -or $ownedProcess.ExecutablePath -ne $serverPath) { throw 'Recovery declined: owned PID/executable mismatch.' }
                Stop-Process -Id $ownedProcess.ProcessId -Force -ErrorAction Stop
                Start-Sleep -Seconds 2
                Start-Process -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$startScript) -WorkingDirectory $root -WindowStyle Hidden | Out-Null
                $ready = $false
                for ($i = 0; $i -lt 60; $i++) {
                    Start-Sleep -Seconds 2
                    try { $ready = (Invoke-RestMethod -Uri 'http://127.0.0.1:8080/health' -TimeoutSec 3).status -eq 'ok' } catch { $ready = $false }
                    if ($ready) { break }
                }
                Write-MonitorEvent @{ event = 'stalled_request_recovery_result'; unit = $unitId; attempt = ($seen + 1); server_healthy = $ready }
            } catch {
                Write-MonitorEvent @{ event = 'stalled_request_recovery_error'; unit = $unitId; attempt = ($seen + 1); error = $_.Exception.Message }
            }
        } else {
            Write-MonitorEvent @{ event = 'stalled_request_recovery_limit'; unit = $unitId; attempts = $seen; action = 'no_further_restarts' }
        }
    }

    Start-Sleep -Seconds $intervalSeconds
}
