# setup.ps1 - bootstrap the docgraph dev environment on Windows.
#
# Creates .venv next to this script, installs docgraph (editable), and drops
# a `docgraph.bat` shim into the user's ~/.local/bin so the CLI is on PATH.
#
# torch is installed from PyTorch's per-CUDA index. The `+cuXY` wheels bundle
# their own CUDA + cuDNN runtime, so GPU works without a separate CUDA
# Toolkit install. CPU is the universal fallback.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File <repo>\setup.ps1
#   .\setup.ps1                            # if execution policy allows; auto-detects CUDA
#   .\setup.ps1 -Recreate                  # wipe .venv and start fresh
#   .\setup.ps1 -Python python3.11
#   .\setup.ps1 -NoShim                    # skip the ~/.local/bin shim
#   .\setup.ps1 -CudaVersion cpu           # force CPU-only (skip the ~1.9 GB CUDA wheel)
#   .\setup.ps1 -CudaVersion cu124         # pin to a specific CUDA wheel
#   .\setup.ps1 -PredownloadModels         # warm the HF cache with default embed + rerank
#
# Supported -CudaVersion values mirror what https://download.pytorch.org/whl/
# publishes: cu118, cu121, cu124, cu126, cu128, cu129, cu130, cpu. The set
# moves with PyTorch; if a new index appears, extend ValidateSet below.

[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$Recreate,
    [switch]$NoShim,
    [string]$ShimDir = (Join-Path $HOME ".local\bin"),
    [ValidateSet("cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cpu", "auto")]
    [string]$CudaVersion = "auto",
    [switch]$PredownloadModels
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvCli = Join-Path $VenvDir "Scripts\docgraph.exe"

# When the user doesn't pick a torch wheel explicitly, probe nvidia-smi.
# An NVIDIA GPU + the WDDM driver that ships with CUDA 13.x → default to
# cu130. Anything older or no GPU → cpu. The detection is best-effort;
# users can always override with -CudaVersion.
function Resolve-CudaVersion {
    $smi = & nvidia-smi.exe --query-gpu=driver_version --format=csv,noheader 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($smi)) {
        return "cpu"
    }
    # nvidia-smi includes the bundled "CUDA Version: X.Y" line in default output.
    # `--query-gpu=driver_version` gives us just the WDDM driver. Driver ≥ 570
    # supports CUDA 13.x; pre-570 needs an older CUDA wheel.
    $driver = ($smi -split "`n")[0].Trim()
    $major = [int]($driver -split "\.")[0]
    if ($major -ge 570) {
        return "cu130"
    } elseif ($major -ge 525) {
        return "cu124"
    }
    return "cpu"
}

if ($CudaVersion -eq "auto") {
    $CudaVersion = Resolve-CudaVersion
    Write-Host "Auto-detected torch wheel: $CudaVersion (override with -CudaVersion)"
}

Write-Host "docgraph root : $Root"
Write-Host "venv          : $VenvDir"
Write-Host "torch wheel   : $CudaVersion"

if ($Recreate -and (Test-Path $VenvDir)) {
    Write-Host "Removing existing .venv ..."
    Remove-Item -Recurse -Force $VenvDir
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating venv with '$Python' ..."
    & $Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
} else {
    Write-Host "Reusing existing venv."
}

Write-Host "Upgrading pip ..."
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }

# Install torch from PyTorch's per-CUDA index before the editable docgraph
# install. pyproject.toml declares `torch>=2.4` without an index URL, so the
# default install would pull a CPU wheel from PyPI even on a CUDA machine.
# Pre-seeding torch from the right index makes the later `pip install -e .`
# satisfy the dep against what's already installed.
$TorchIndex = "https://download.pytorch.org/whl/$CudaVersion"
Write-Host "Installing torch from $TorchIndex ..."
& $VenvPython -m pip install --index-url $TorchIndex torch
if ($LASTEXITCODE -ne 0) { throw "torch install failed (exit $LASTEXITCODE)" }

Write-Host "Installing docgraph (editable) ..."
& $VenvPython -m pip install -e $Root
if ($LASTEXITCODE -ne 0) { throw "docgraph install failed (exit $LASTEXITCODE)" }

# Quick sanity check — log device / dtype / version so the install record
# itself surfaces a misconfigured CUDA runtime before the host ever runs.
$check = & $VenvPython -c @"
import torch
print('torch', torch.__version__,
      '| cuda_available=', torch.cuda.is_available(),
      '| device_count=', torch.cuda.device_count())
"@
Write-Host "  $check"

# Optional: warm the HuggingFace cache with the default embed + rerank
# models. First `docgraph host` becomes 2-3 s faster afterwards. Off by
# default to keep the install lean.
if ($PredownloadModels) {
    Write-Host "Predownloading default embed + rerank models ..."
    & $VenvPython -c @"
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')
CrossEncoder('jinaai/jina-reranker-v1-tiny-en', device='cpu')
print('  cached BAAI/bge-small-en-v1.5 + jinaai/jina-reranker-v1-tiny-en')
"@
}

if (-not $NoShim) {
    if (-not (Test-Path $ShimDir)) {
        New-Item -ItemType Directory -Path $ShimDir | Out-Null
    }
    $ShimPath = Join-Path $ShimDir "docgraph.bat"
    $ShimBody = @"
@echo off
REM docgraph CLI shim - installed by setup.ps1 from $Root.
"$VenvCli" %*
"@
    Set-Content -Path $ShimPath -Value $ShimBody -Encoding ASCII
    Write-Host "Installed shim : $ShimPath"
    if (-not (($env:PATH -split ";") -contains $ShimDir)) {
        Write-Host "  note: $ShimDir is not on PATH - add it to use 'docgraph' anywhere."
    }
}

Write-Host ""
Write-Host "Done."
Write-Host "  CLI         : $VenvCli"
Write-Host "  Repo shim   : $(Join-Path $Root 'docgraph.bat')"
& $VenvCli --help | Select-Object -First 3
Write-Host ""
Write-Host "Next steps:"
Write-Host "  docgraph host                          # serve cwd as a single root"
Write-Host "  docgraph host --root /a --root /b      # multi-root host"
Write-Host "  docgraph host --gpu --embed-torch-compile  # CUDA + torch.compile"
if (-not $env:HF_TOKEN) {
    Write-Host ""
    Write-Host "Tip: setting `$env:HF_TOKEN` (any read token from"
    Write-Host "  https://huggingface.co/settings/tokens) raises HF Hub rate"
    Write-Host "  limits + silences the 'unauthenticated requests' warning."
}
