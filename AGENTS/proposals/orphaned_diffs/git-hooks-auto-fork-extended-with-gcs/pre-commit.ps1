# PowerShell pre-commit hook
Write-Output "[HOOK] Checking submodule state before commit..."
git submodule update --init --recursive

$dirty = git submodule foreach --quiet 'git diff --quiet || echo $name'
if ($dirty) {
    Write-Output "[HOOK] Error: The following submodules have uncommitted changes:"
    Write-Output $dirty
    exit 1
}

git diff --quiet wheelhouse || git add wheelhouse
git diff --quiet wheelhouse/lfsavoider || git add wheelhouse/lfsavoider
Write-Output "[HOOK] Submodule state clean. Proceeding."

if (-not $?) {
    Write-Host "Commit failed. Creating forked branch..."
    $suffix = "$(Get-Date -Format yyyyMMdd)-$(git rev-parse --short HEAD)"
    $forkBranch = "auto/fork/$suffix"
    git checkout -b $forkBranch
    git push -u origin $forkBranch
    Write-Host "Changes pushed to forked branch: $forkBranch"
}
