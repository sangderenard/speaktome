# PowerShell automated reinstall and demo run for SpeakToMe
param(
    [switch]$interactive
)

$ErrorActionPreference = 'Stop'

# Reinstall environment
powershell -ExecutionPolicy Bypass -File reinstall_env.ps1 -Yes

# First demo run (non-interactive)
& ./run.ps1 -s "Automation demo" -m 5 -auto_expand 2 -safe_mode

# Second demo run with visualization
& ./run.ps1 -s "Visualization demo" -m 5 -final_viz -safe_mode

# Optional interactive run
if ($interactive) {
    & ./run.ps1 -s "Interactive session" -human_control
}
