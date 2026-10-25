$ErrorActionPreference = 'Stop'
$source = Join-Path $PSScriptRoot 'native_audio_scope'
$build = Join-Path $source 'build'

cmake -S $source -B $build -A x64
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed for native audio scope.' }
cmake --build $build --config Release
if ($LASTEXITCODE -ne 0) { throw 'Native audio scope build failed.' }
