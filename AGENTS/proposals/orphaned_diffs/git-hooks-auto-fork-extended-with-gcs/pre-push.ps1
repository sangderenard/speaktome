# PowerShell pre-push hook
Write-Output "[HOOK] Pre-push validation starting..."
git submodule update --init --recursive

$dirty = git submodule foreach --quiet 'git diff --quiet || echo $name'
if ($dirty) {
    Write-Output "[HOOK] Dirty submodules detected:"
    Write-Output $dirty
    exit 1
}

# Optional commit forking for audit trail
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$hash = git rev-parse HEAD
$branch = "agent/auto-$timestamp"
git branch $branch $hash
git push origin $branch
Write-Output "[HOOK] Forked current commit to $branch for historical audit."

# Optional GCS sync
$gcsSyncScript = ".git/hooks/sync-gcs.ps1"
if (Test-Path $gcsSyncScript) {
    Write-Output "[HOOK] Running GCS sync script..."
    & $gcsSyncScript
}

Write-Output "[HOOK] Ready to push."

if (-not $?) {
    Write-Host "Push failed. Creating forked branch..."
    $suffix = "$(Get-Date -Format yyyyMMdd)-$(git rev-parse --short HEAD)"
    $forkBranch = "auto/fork/$suffix"
    git checkout -b $forkBranch
    git push -u origin $forkBranch
    Write-Host "Changes pushed to forked branch: $forkBranch"
}
