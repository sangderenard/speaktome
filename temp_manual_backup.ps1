# 1) Make a dated backup folder inside the project
$stamp = Get-Date -UFormat '%Y%m%d_%H%M%S'
$backupDir = "BACKUPS\preMerge_$stamp"
New-Item -ItemType Directory -Path $backupDir | Out-Null

# 2) COPY the exact files Git complained about
#    (adjust the list if you see more paths)
Copy-Item AGENTS\experience_reports\1749852048_TTICKET_Pytest_Env_Setup_Failure.md        "$backupDir\" -Force
Copy-Item AGENTS\experience_reports\1749852753_DOC_PoetryCore_Powershell_Check.md          "$backupDir\" -Force
Copy-Item AGENTS\experience_reports\1749853535_DOC_Minimal_Wheelhouse_Builder.md           "$backupDir\" -Force
Copy-Item AGENTS\experience_reports\archive\1749846307_DOC_Lookahead_Demo_Header_Fix.md    "$backupDir\" -Force
Copy-Item AGENTS\experience_reports\archive\1749846629_TTICKET_Pytest_Run_Failure.md       "$backupDir\" -Force
Copy-Item AGENTS\experience_reports\archive\1749847156_AUDIT_Pytest_Env_Root_Detection.md  "$backupDir\" -Force
Copy-Item AGENTS\experience_reports\archive\1749847163_DOC_Revert_Header_Validator_Fix.md  "$backupDir\" -Force
Copy-Item AGENTS\proposals\wheelhouse_repo\*                                                "$backupDir\" -Recurse -Force

# 3) Add & commit the backup folder
git add $backupDir
git commit -m "Backup of untracked files before merge on $stamp"
