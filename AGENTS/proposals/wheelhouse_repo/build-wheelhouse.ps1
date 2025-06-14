param(
  [string]$WheelhouseDir = "wheelhouse",
  [string]$Requirements = "requirements.txt"
)
$ErrorActionPreference = "Stop"
$PyTag = "cp312"
$linuxDir = Join-Path $WheelhouseDir "wheels/linux/$PyTag"
$winDir   = Join-Path $WheelhouseDir "wheels/windows/$PyTag"
New-Item -ItemType Directory -Path $linuxDir -Force | Out-Null
New-Item -ItemType Directory -Path $winDir -Force | Out-Null
Get-Content $Requirements | ForEach-Object {
  if (-not $_ -or $_.StartsWith('#')) { return }
  Write-Host "Downloading $_" -ForegroundColor Cyan
  pip download $_ `
    --python-version 312 `
    --only-binary=:all: `
    --platform win_amd64 `
    --dest $winDir
  pip download $_ `
    --python-version 312 `
    --only-binary=:all: `
    --platform manylinux_2_28_x86_64 `
    --dest $linuxDir
}
Get-ChildItem -Recurse -Filter *.whl -Path $WheelhouseDir |
  Get-FileHash -Algorithm SHA256 |
  ForEach-Object { "$($_.Hash)  $($_.Path -replace '\\','/')" } |
  Set-Content -Path (Join-Path $WheelhouseDir 'SHA256SUMS.txt')
Write-Host "Wheelhouse written to $WheelhouseDir" -ForegroundColor Green
