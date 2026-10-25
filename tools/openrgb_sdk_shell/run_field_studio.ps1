param(
    [switch] $CpuTemperature,
    [switch] $FanRpm,
    [string] $FanSensor = '',
    [Nullable[double]] $Fps = $null
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$scopeSource = Join-Path $PSScriptRoot 'native_audio_scope\openrgb_audio_scope.cpp'
$scopeDll = Join-Path $PSScriptRoot 'native_audio_scope\build\Release\openrgb_audio_scope.dll'
$hidSource = Join-Path $PSScriptRoot 'native_asrock_hid\asrock_polychrome_hid.cpp'
$hidDll = Join-Path $PSScriptRoot 'native_asrock_hid\build\Release\asrock_polychrome_hid.dll'
if (-not (Test-Path -LiteralPath $scopeDll) -or
    (Get-Item -LiteralPath $scopeSource).LastWriteTimeUtc -gt (Get-Item -LiteralPath $scopeDll).LastWriteTimeUtc) {
    & (Join-Path $PSScriptRoot 'build_audio_scope.ps1')
}
if (-not (Test-Path -LiteralPath $hidDll) -or
    (Get-Item -LiteralPath $hidSource).LastWriteTimeUtc -gt (Get-Item -LiteralPath $hidDll).LastWriteTimeUtc) {
    & (Join-Path $PSScriptRoot 'build_asrock_hid.ps1')
}

$openRgbProcesses = @(Get-Process -Name 'OpenRGB' -ErrorAction SilentlyContinue)
if ($openRgbProcesses.Count -gt 0) {
    $openRgbProcesses | Stop-Process
    $openRgbProcesses | Wait-Process -ErrorAction SilentlyContinue
}

$arguments = @(
    '-3.11', '.\openrgb_field_studio.py',
    '--config', '.\openrgb_field_studio_config.json',
    '--frames', '0',
    '--apply'
)
if ($null -ne $Fps) {
    if ($Fps -le 0.0 -or $Fps -gt 1000.0) {
        throw '-Fps must be greater than 0 and no greater than 1000.'
    }
    $arguments += @('--fps', $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture))
}
if ($CpuTemperature) {
    $arguments += '--cpu-temperature'
}
if ($FanRpm) {
    $arguments += '--fan-rpm'
    if ($FanSensor) {
        $arguments += @('--fan-sensor', $FanSensor)
    }
}
& py @arguments
