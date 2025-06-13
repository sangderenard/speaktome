<#
===============================================================================
NAME:      build-wheelhouse.ps1
PURPOSE:   Fully-automated Git-LFS wheelhouse builder for Python 3.12
           – PyTorch (CPU + CUDA 11.8), Poetry, and poetry-core –
           for Windows 64-bit and manylinux 2.28 64-bit.

USAGE:     1.  Adjust $RepoRemote (git remote URL) if you want auto-push.
           2.  Put a manifest file `wheel_manifest.json` next to this script.
               (A default sample manifest is generated on first run.)
           3.  Run:  powershell -ExecutionPolicy Bypass -File build-wheelhouse.ps1
           4.  After wheels download, script commits to Git LFS and (optionally)
               pushes to the remote.

DEPENDENCIES:
           • PowerShell 7+
           • git  +  git-lfs
           • Python 3.12 with pip ≥ 23.3
           • Internet access for the initial download
===============================================================================
#>

param(
    [string]$WheelhouseDir = ".\torch-wheelhouse",
    [string]$ManifestPath  = ".\wheel_manifest.json",
    [string]$RepoRemote    = "git@github.com:YourOrg/torch-wheelhouse.git",
    [switch]$SkipPush
)

# ─── Prereq sanity ──────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"
if (-not (Get-Command git -ErrorAction SilentlyContinue))      { throw "git not on PATH" }
if (-not (Get-Command python -ErrorAction SilentlyContinue))   { throw "python not on PATH" }
if (-not (Get-Command pip -ErrorAction SilentlyContinue))      { throw "pip not on PATH" }

git lfs install | Out-Null

# ─── Helper: ensure directory layout exists ────────────────────────────────
function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) { New-Item -ItemType Directory -Path $Path | Out-Null }
}

# ─── Step 0: create default manifest if absent ─────────────────────────────
if (-not (Test-Path $ManifestPath)) {
    $defaultManifest = @'
{
  "python_tag": "cp312",
  "packages": [
    { "name": "torch",       "version": "2.7.1", "flavours": ["cpu", "cu118"] },
    { "name": "poetry",      "version": "1.8.2" },
    { "name": "poetry-core", "version": "1.9.0" }
  ]
}
'@
    $defaultManifest | Set-Content $ManifestPath -Encoding UTF8
    Write-Host "📝  Created sample manifest at $ManifestPath . Edit as needed." -ForegroundColor Yellow
    exit
}

# ─── Load manifest ─────────────────────────────────────────────────────────
$manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$PyTag    = $manifest.python_tag  # cp312

# ─── Initialise / open wheelhouse repo ─────────────────────────────────────
Ensure-Dir $WheelhouseDir
Set-Location  $WheelhouseDir
if (-not (Test-Path ".git")) {
    git init | Out-Null
    git lfs track "*.whl"
    git add .gitattributes
}

Ensure-Dir "wheels/linux/$PyTag"
Ensure-Dir "wheels/windows/$PyTag"
Ensure-Dir "scripts"

# ─── MAIN LOOP: download wheels from manifest ──────────────────────────────
foreach ($pkg in $manifest.packages) {
    $name    = $pkg.name
    $version = $pkg.version
    $flavours = if ($pkg.PSObject.Properties.Name -contains 'flavours') {
        $pkg.flavours
    } else { @('') }   # empty string → no flavour suffix

    Write-Host "╔══════════════════════════════════════════════════════════════════════╗"
    Write-Host "║  📦 $name  $version" -ForegroundColor Cyan
    Write-Host "╚══════════════════════════════════════════════════════════════════════╝"

    foreach ($flavour in $flavours) {
        $suffix = if ($flavour) { "+$flavour" } else { "" }

        # Skip CUDA flavour for non-torch packages
        if (($name -ne "torch") -and ($flavour -ne "")) { continue }

        Write-Host " →  Windows x64   $suffix"
        pip download "$name==$version$suffix" `
            --python-version 312 `
            --only-binary=:all: `
            --platform win_amd64 `
            --dest "wheels/windows/$PyTag" | Out-Null

        Write-Host " →  Manylinux x64  $suffix"
        pip download "$name==$version$suffix" `
            --python-version 312 `
            --only-binary=:all: `
            --platform manylinux_2_28_x86_64 `
            --dest "wheels/linux/$PyTag" | Out-Null
    }
}

# ─── SHA256 summary for supply-chain integrity ─────────────────────────────
Get-ChildItem -Recurse -Filter *.whl |
  Get-FileHash -Algorithm SHA256 |
  ForEach-Object { "$($_.Hash)  $($_.Path -replace '\\','/')" } |
  Set-Content -Path SHA256SUMS.txt

# ─── Generate / overwrite install script (PowerShell) ──────────────────────
@"
<#
Install PyPI packages from local wheelhouse.
Usage examples:
  .\scripts\install_torch.ps1 -Package torch -Version 2.7.1 -Flavour cpu
  .\scripts\install_torch.ps1 -Package poetry -Version 1.8.2
#>
param(
  [string] \$Package  = "torch",
  [string] \$Version  = "2.7.1",
  [string] \$Flavour  = "cpu"    # ignored for non-torch
)
\$PyTag = "$PyTag"
\$Root  = Split-Path -Parent \$MyInvocation.MyCommand.Definition
if (\$IsWindows) {
    \$Find = Join-Path \$Root "..\\wheels\\windows\\\$PyTag"
} else {
    \$Find = Join-Path \$Root "../wheels/linux/\$PyTag"
}
if (\$Package -ne "torch") { \$Flavour = "" }
\$suffix = if (\$Flavour) { "+\$Flavour" } else { "" }
pip install --no-index --find-links=\$Find "\$Package==\$Version\$suffix"
"@ | Set-Content scripts\install_torch.ps1 -Encoding UTF8

# ─── Commit changes ────────────────────────────────────────────────────────
git add wheels scripts SHA256SUMS.txt
git commit -m "Update wheelhouse ($(Get-Date -Format u))" | Out-Null

# ─── Optional push ─────────────────────────────────────────────────────────
if (-not $SkipPush) {
    if (-not (git remote get-url origin 2>$null)) {
        git remote add origin $RepoRemote
    }
    git push --set-upstream origin main
}

Write-Host "`n✅  Wheelhouse build finished." -ForegroundColor Green
