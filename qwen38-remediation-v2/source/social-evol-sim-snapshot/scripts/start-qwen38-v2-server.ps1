param(
    [int]$ContextSize = 65536,
    [string]$EvidenceDirectory = 'data/qwen38_remediation_v2_20261016'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$evidence = [IO.Path]::GetFullPath((Join-Path $root $EvidenceDirectory))
New-Item -ItemType Directory -Force -Path (Join-Path $evidence 'server') | Out-Null
$listener=Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if($listener){
    $ownedPath=Join-Path $evidence 'server/owned_server.json'
    $owned=$null;if(Test-Path -LiteralPath $ownedPath){$owned=Get-Content -LiteralPath $ownedPath -Raw|ConvertFrom-Json}
    if($owned -and $listener.OwningProcess -eq $owned.pid){Write-Output "Owned llama-server PID $($owned.pid) is already listening.";exit 0}
    throw 'Port 8080 is already owned by another process; this script will not replace it.'
}
$server = Join-Path $root '.local/llama-b11669/llama-server.exe'
$model = Join-Path $root '.local/models/Qwen3.8-27B-Q4_K_M.gguf'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$serverLog = Join-Path $evidence "server/llama-$stamp.log"
# Per-request reasoning_effort=none disables native thinking; low enables it.
# Both conditions use the unchanged simulator's two explicit response stages.
$arguments = @('--model', $model, '--alias', 'Qwen3.8-27B-Q4_K_M',
    '--host', '127.0.0.1', '--port', '8080', '--ctx-size', "$ContextSize", '--parallel', '1',
    '--n-gpu-layers', 'all', '--device', 'CUDA0', '--split-mode', 'none', '--fit', 'off',
    '--flash-attn', 'on', '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0',
    '--no-mmproj', '--jinja', '--reasoning', 'auto', '--reasoning-format', 'deepseek',
    '--reasoning-effort', 'low', '--reasoning-budget', '2048', '--no-context-shift',
    '--timeout', '3600', '--metrics', '--offline', '--log-verbosity', '4', '--log-file', $serverLog)
$proc = Start-Process -FilePath $server -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $evidence "server/stdout-$stamp.log") `
    -RedirectStandardError (Join-Path $evidence "server/stderr-$stamp.log")
@{pid=$proc.Id;started_at=[DateTime]::UtcNow.ToString('o');executable=$server;arguments=$arguments;log=$serverLog;context=$ContextSize} |
    ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidence 'server/owned_server.json') -Encoding utf8
Write-Output "Started owned llama-server PID $($proc.Id); context $ContextSize; log $serverLog"
