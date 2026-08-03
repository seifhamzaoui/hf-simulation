<#
Builds a fully self-contained, portable distributable of this project:
  dist\SimulationSuite\
    SimulationSuite.exe   <- tiny launcher (see launcher_stub.py)
    runtime\              <- a whole, ordinary CPython install (copied
                             from the base install this venv points at),
                             with the venv's own site-packages (ngsolve,
                             netgen, numpy, scipy, matplotlib, cupy,
                             nvidia CUDA wheels, ezdxf, ...) laid on top
    app\                  <- this whole project directory (minus .venv/
                             .git/__pycache__/build_exe/dist), so
                             simulation_ui.py and everything it launches
                             as a subprocess (transformer_geometry.py,
                             simulation_ngsolve*.py, ...) are right next
                             to it, unmodified.

Why copy a whole Python install instead of having PyInstaller freeze
everything: NGSolve/Netgen's native 3D viewer, Tcl/Tk bundling, and
runtime plugin loading are known to be difficult for PyInstaller's
dependency analysis to capture correctly. Shipping a real, ordinary
python.exe (just relocated) sidesteps that entirely -- every subprocess
call in simulation_ui.py already just uses sys.executable, which will
correctly point at THIS bundled interpreter once launched through it, no
code changes needed there.

Known limitations (see the caveats printed at the end):
  - Assumes the target machine's Windows version provides the Universal
    C Runtime (ucrtbase.dll etc, part of Windows 10+ since a 2015-era
    update) -- NOT bundled here. Most modern Windows 10/11 machines
    already have it; genuinely ancient/unpatched Windows might not.
  - GPU stages need a real NVIDIA GPU + driver on the target machine --
    the CUDA *libraries* are bundled (nvidia/cupy), but a driver is a
    kernel-mode component this can't ship.
  - Only tested by relocating this SAME machine's install to a new path,
    not on a genuinely separate "nothing pre-installed" machine.

-CpuOnly: skips cupy/cupy_backends/cupyx and every nvidia_* CUDA wheel
(~2GB of the ~3.5GB full bundle) when overlaying site-packages. The GPU
Stages / GPU R-L Ratio tabs in simulation_ui.py will show their existing
"not available" behavior (import fails) instead of running -- everything
else (Config Editor, Geometry Builder, NGSolve CPU Stages, AC Litz sweep,
CPU R/L Ratio, HF Model) is unaffected, since none of those import cupy.
#>

param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$DistDir    = Join-Path $ProjectDir "dist\SimulationSuite"
$RuntimeDir = Join-Path $DistDir "runtime"
$AppDir     = Join-Path $DistDir "app"
$BasePython = "C:\Program Files\Python314"
$VenvSitePackages = Join-Path $ProjectDir ".venv\Lib\site-packages"

function Invoke-Robocopy($src, $dst, $extraArgs) {
    $args = @($src, $dst) + $extraArgs + @("/NFL", "/NDL", "/NJH", "/NJS", "/NP", "/R:1", "/W:1")
    & robocopy @args | Out-Null
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed ($LASTEXITCODE) copying $src -> $dst"
    }
}

Write-Host "=== 1/4: copying base Python install (excluding its own site-packages) ==="
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null
Invoke-Robocopy $BasePython $RuntimeDir @("/E", "/XD", (Join-Path $BasePython "Lib\site-packages"))

$RuntimeSitePackages = Join-Path $RuntimeDir "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $RuntimeSitePackages | Out-Null
if ($CpuOnly) {
    Write-Host "=== 2/4: overlaying venv site-packages (CPU-only -- skipping cupy/nvidia CUDA wheels) ==="
    $gpuDirNames = Get-ChildItem $VenvSitePackages -Directory | Where-Object { $_.Name -like "nvidia*" -or $_.Name -like "cupy*" } | Select-Object -ExpandProperty Name
    $xdArgs = $gpuDirNames | ForEach-Object { Join-Path $VenvSitePackages $_ }
    Invoke-Robocopy $VenvSitePackages $RuntimeSitePackages (@("/E", "/XD") + $xdArgs)
} else {
    Write-Host "=== 2/4: overlaying venv site-packages (ngsolve, netgen, numpy, scipy, matplotlib, cupy, nvidia, ezdxf, ...) ==="
    Invoke-Robocopy $VenvSitePackages $RuntimeSitePackages @("/E")
}

Write-Host "=== 3/4: copying project files into app\ ==="
New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
Invoke-Robocopy $ProjectDir $AppDir @(
    "/E", "/XD",
    (Join-Path $ProjectDir ".venv"),
    (Join-Path $ProjectDir ".git"),
    (Join-Path $ProjectDir "__pycache__"),
    (Join-Path $ProjectDir "build_exe"),
    (Join-Path $ProjectDir "dist"),
    "/XF", "*.pyc"
)

Write-Host "=== 4/4: building launcher exe ==="
$BuildWork = Join-Path $PSScriptRoot "pyinstaller_work"
& "$ProjectDir\.venv\Scripts\python.exe" -m PyInstaller `
    --onefile --noconsole --name SimulationSuite `
    --distpath $DistDir --workpath (Join-Path $BuildWork "build") `
    --specpath $BuildWork `
    (Join-Path $PSScriptRoot "launcher_stub.py")

Write-Host ""
Write-Host "=== Done: $DistDir ==="
Write-Host "Launch by double-clicking $DistDir\SimulationSuite.exe"
Write-Host ""
Write-Host "Caveats:"
Write-Host "  - Needs the target machine's Windows Universal C Runtime (Windows 10+ has it)."
Write-Host "  - GPU stages need a real NVIDIA GPU + driver on the target machine."
Write-Host "  - Verified by relocation on THIS machine, not yet on a separate clean machine."
