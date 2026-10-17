$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$server = Join-Path $root '.local/llama-b11669/llama-server.exe'
$model = Join-Path $root '.local/models/Qwen3.8-27B-Q4_K_M.gguf'
if (-not (Test-Path -LiteralPath $server)) { throw 'Run scripts/setup-local.ps1 first' }
if (-not (Test-Path -LiteralPath $model)) { throw 'Model is missing; run scripts/setup-local.ps1' }

# Disable implicit resizing/offloading: fail visibly if this allocation does not fit.
# No projector is loaded. Model reasoning is separate from the paper's reflection.
& $server --model $model --alias Qwen3.8-27B-Q4_K_M `
    --host 127.0.0.1 --port 8080 --ctx-size 32768 --parallel 1 `
    --n-gpu-layers all --device CUDA0 --split-mode none --fit off `
    --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 `
    --no-mmproj --jinja --reasoning off --timeout 3600 --metrics --offline `
    --log-verbosity 4 `
    --log-file (Join-Path $root '.local/llama-server.log')
exit $LASTEXITCODE
