# Apply TensorPrinting and TimeSync changes
param(
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

Set-Location $RepoRoot

$patch = Join-Path $PSScriptRoot 'tensor_timesync_changes.patch'

# Apply patch without .editorconfig changes
git apply --binary $patch

# Remove old directories from before renaming
if (Test-Path 'tensor printing') {
    Remove-Item 'tensor printing' -Recurse -Force
}
if (Test-Path 'time_sync') {
    Remove-Item 'time_sync' -Recurse -Force
}

# Stage files for commit
git add -A

Write-Host 'Patch applied. Review changes with `git status`.'
