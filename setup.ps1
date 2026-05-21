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
#   .\setup.ps1                 # if execution policy allows
#   .\setup.ps1 -Recreate       # wipe .venv and start fresh
#   .\setup.ps1 -Python python3.11
#   .\setup.ps1 -NoShim         # skip the ~/.local/bin shim
#   .\setup.ps1 -CudaVersion cpu       # CPU-only install (no CUDA wheel)
#   .\setup.ps1 -CudaVersion cu124     # NVIDIA + CUDA 12.4 wheel
#   .\setup.ps1 -CudaVersion cu130     # NVIDIA + CUDA 13.x wheel (default on Windows)

[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$Recreate,
    [switch]$NoShim,
    [string]$ShimDir = (Join-Path $HOME ".local\bin"),
    [ValidateSet("cu130", "cu124", "cpu")]
    [string]$CudaVersion = "cu130"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvCli = Join-Path $VenvDir "Scripts\docgraph.exe"

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
