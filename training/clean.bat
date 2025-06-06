@echo off
setlocal enabledelayedexpansion

echo [1/6] Removing .gitignore in current folder...
del .gitignore 2>nul

echo [2/6] Running PowerShell script: gitignore_large_files.ps1
powershell -ExecutionPolicy Bypass -File gitignore_large_files.ps1

echo [3/6] Checking for large (>100MB) files in git history...
for /f "tokens=*" %%f in ('git rev-list --objects --all') do (
    for /f "tokens=1,* delims= " %%a in ("%%f") do (
        set SHA=%%a
        set FILE=%%b
        if exist "!FILE!" (
            for %%s in ("!FILE!") do (
                set SIZE=%%~zs
                if !SIZE! GTR 104857600 (
                    echo Deleting large file: !FILE! (!SIZE! bytes)
                    git rm --cached "!FILE!"
                    del /f /q "!FILE!"
                )
            )
        )
    )
)

echo [4/6] Committing cleanup...
git add -A
git commit -m "Remove oversized files and refresh .gitignore state"

echo [5/6] Rebuilding Git LFS tracking (if applicable)...
git lfs install
git lfs track "*.bin"
git add .gitattributes
git commit -m "Track binary files with Git LFS"

echo [6/6] Ready for push. Run 'git push origin main' when you're confident.

echo Done. Please review commit state and confirm.
pause
