$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'native_asrock_hid'
$build = Join-Path $source 'build'

cmake -S $source -B $build -A x64
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed for native ASRock HID transport.' }
cmake --build $build --config Release
if ($LASTEXITCODE -ne 0) { throw 'Native ASRock HID transport build failed.' }
