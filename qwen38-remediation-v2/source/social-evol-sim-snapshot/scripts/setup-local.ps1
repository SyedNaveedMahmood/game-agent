param([switch]$SkipPython)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$runtime = Join-Path $root '.local'
$bin = Join-Path $runtime 'llama-b11669'
$models = Join-Path $runtime 'models'
New-Item -ItemType Directory -Force $bin, $models | Out-Null

# Official CUDA 12.4 binaries include Ampere support; no CUDA toolkit needed.
foreach ($asset in @('llama-b11669-bin-win-cuda-12.4-x64.zip', 'cudart-llama-bin-win-cuda-12.4-x64.zip')) {
    $archive = Join-Path $runtime $asset
    & curl.exe --fail --location --retry 5 --continue-at - --output $archive "https://github.com/ggml-org/llama.cpp/releases/download/b11669/$asset"
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $asset" }
    Expand-Archive -LiteralPath $archive -DestinationPath $bin -Force
}

$model = Join-Path $models 'Qwen3.8-27B-Q4_K_M.gguf'
& curl.exe --fail --location --retry 5 --continue-at - --output $model 'https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF/resolve/71bc7b627595dc8a91039addd9c791ae548d6747/Qwen3.8-27B-Q4_K_M.gguf'
if ($LASTEXITCODE -ne 0) { throw 'Model download failed' }
if ((Get-FileHash -LiteralPath $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne 'c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747') {
    throw 'Model SHA256 mismatch'
}

if (-not $SkipPython) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Creating Python environment failed' }
    $python = Join-Path $root '.venv/Scripts/python.exe'
    $dependencies = & $python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))"
    & $python -m pip install @dependencies
    if ($LASTEXITCODE -ne 0) { throw 'Installing project dependencies failed' }
}
Write-Host 'Local setup complete. Run scripts/start-local-server.ps1 next.'
