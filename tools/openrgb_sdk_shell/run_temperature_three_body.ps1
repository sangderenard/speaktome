$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
& (Join-Path $PSScriptRoot 'ensure_openrgb_server.ps1')

py -3.11 .\openrgb_dt_three_body.py `
    --device-name 'ASRock B650M Pro RS WiFi' `
    --zone 'Addressable Header 3/Audio' `
    --position-count 8 `
    --fps 60 `
    --frames 0 `
    --backend eager `
    --center-contribution 0.125 `
    --cpu-temperature `
    --window `
    --apply
