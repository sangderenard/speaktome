# PowerShell script to add files larger than a size threshold to .gitignore.
# Usage: .\gitignore_large_files.ps1 [-SizeMB 100] [-GitignorePath .gitignore]
param(
    [int]$SizeMB = 50,
    [string]$GitignorePath = '.gitignore'
)

$SizeBytes = $SizeMB * 1MB
if (-not (Test-Path $GitignorePath)) {
    New-Item -ItemType File -Path $GitignorePath | Out-Null
}

Get-ChildItem -Path . -Recurse -File -Force |
    Where-Object { $_.FullName -notmatch '\\.git\\' -and $_.Length -gt $SizeBytes } |
    ForEach-Object {
        $relativeSlashed = ($_.FullName.Substring((Get-Location).Path.Length + 1)) -replace '\\', '/'
        if (-not (Select-String -Path $GitignorePath -SimpleMatch -Quiet -Pattern "$relativeSlashed")) {
            Add-Content -Path $GitignorePath -Value $relativeSlashed
            Write-Host "Added $relative to $GitignorePath"
        }
    }
