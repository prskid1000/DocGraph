# setup.ps1 - bootstrap the docgraph dev environment on Windows.
#
# Creates .venv next to this script, installs docgraph (editable), and drops
# a `docgraph.bat` shim into the user's ~/.local/bin so the CLI is on PATH.
#
# Usage (from anywhere):
#   powershell -ExecutionPolicy Bypass -File <repo>\setup.ps1
#   .\setup.ps1                 # if execution policy allows
#   .\setup.ps1 -Recreate       # wipe .venv and start fresh
#   .\setup.ps1 -Python python3.11
#   .\setup.ps1 -NoShim         # skip the ~/.local/bin shim
#   .\setup.ps1 -Gpu none       # skip GPU ORT install (default: directml on Windows)
#   .\setup.ps1 -Gpu cuda       # use onnxruntime-gpu (NVIDIA + CUDA toolkit required)

[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$Recreate,
    [switch]$NoShim,
    [string]$ShimDir = (Join-Path $HOME ".local\bin"),
    [ValidateSet("directml", "cuda", "none")]
    [string]$Gpu = "directml"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvCli = Join-Path $VenvDir "Scripts\docgraph.exe"

Write-Host "docgraph root : $Root"
Write-Host "venv          : $VenvDir"

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

Write-Host "Installing docgraph (editable) ..."
& $VenvPython -m pip install -e $Root
if ($LASTEXITCODE -ne 0) { throw "docgraph install failed (exit $LASTEXITCODE)" }

# GPU runtime - pyproject.toml only declares the base `onnxruntime` (CPU);
# GPU is opt-in per docgraph's design. Swap to the requested variant here.
# The directml/gpu wheels ship the CPU provider too, so we uninstall the
# base package first to avoid a conflicting double-install.
if ($Gpu -ne "none") {
    $gpuPkg = if ($Gpu -eq "directml") { "onnxruntime-directml" } else { "onnxruntime-gpu" }
    Write-Host "Installing GPU runtime ($gpuPkg) ..."
    & $VenvPython -m pip uninstall -y onnxruntime 2>&1 | Out-Null
    & $VenvPython -m pip install $gpuPkg
    if ($LASTEXITCODE -ne 0) { throw "$gpuPkg install failed (exit $LASTEXITCODE)" }
    $check = & $VenvPython -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))"
    Write-Host "  ORT providers: $check"
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
