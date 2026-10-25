param(
    [string] $OpenRgbExecutable = 'C:\Users\alber\AppData\Local\Temp\OpenRGB-1.0-81bbe18\OpenRGB Windows 64-bit\OpenRGB.exe',
    [string] $OpenRgbConfig = 'C:\Users\alber\AppData\Local\Temp\OpenRGB-1.0-total-coordination-20261003',
    [string] $ServerHost = '127.0.0.1',
    [ValidateRange(1, 65535)]
    [int] $ServerPort = 6742,
    [ValidateRange(1, 120)]
    [int] $StartupTimeoutSeconds = 30
)

$ErrorActionPreference = 'Stop'

function Test-OpenRgbSdkSocket {
    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $pending = $client.BeginConnect($ServerHost, $ServerPort, $null, $null)
        if (-not $pending.AsyncWaitHandle.WaitOne(250)) {
            return $false
        }
        $client.EndConnect($pending)
        return $client.Connected
    }
    catch {
        return $false
    }
    finally {
        $client.Dispose()
    }
}

if (Test-OpenRgbSdkSocket) {
    return
}

$existing = @(Get-Process -Name 'OpenRGB' -ErrorAction SilentlyContinue)
if ($existing.Count -gt 0) {
    throw "OpenRGB is already running but its SDK server is unavailable at ${ServerHost}:${ServerPort}. Close that instance so the reviewed server configuration can start."
}
if (-not (Test-Path -LiteralPath $OpenRgbExecutable -PathType Leaf)) {
    throw "Reviewed OpenRGB executable not found: $OpenRgbExecutable"
}
if (-not (Test-Path -LiteralPath $OpenRgbConfig -PathType Container)) {
    throw "Reviewed OpenRGB configuration not found: $OpenRgbConfig"
}

$arguments = @(
    '--config', $OpenRgbConfig,
    '--noautoconnect',
    '--server',
    '--server-host', $ServerHost,
    '--server-port', $ServerPort.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
$process = Start-Process -FilePath $OpenRgbExecutable -ArgumentList $arguments -WindowStyle Hidden -PassThru
$deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
while ([DateTime]::UtcNow -lt $deadline) {
    if ($process.HasExited) {
        throw "OpenRGB exited before its SDK server became available (exit code $($process.ExitCode))."
    }
    if (Test-OpenRgbSdkSocket) {
        Write-Host "Started OpenRGB SDK server at ${ServerHost}:${ServerPort}."
        return
    }
    Start-Sleep -Milliseconds 200
}

throw "OpenRGB did not expose its SDK server at ${ServerHost}:${ServerPort} within ${StartupTimeoutSeconds} seconds."
