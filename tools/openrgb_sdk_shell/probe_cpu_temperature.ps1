param(
    [string] $AssemblyPath = 'C:\Users\alber\AppData\Local\Programs\NiceHash Miner\app_3.0.9.2\LibreHardwareMonitorLib.dll'
)

$ErrorActionPreference = 'Stop'

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$administrator = $principal.IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $administrator) {
    throw 'CPU sensor access requires an elevated Windows PowerShell process.'
}
if (-not (Test-Path -LiteralPath $AssemblyPath -PathType Leaf)) {
    throw "LibreHardwareMonitor assembly not found: $AssemblyPath"
}

Add-Type -Path $AssemblyPath
$computer = [LibreHardwareMonitor.Hardware.Computer]::new()
$computer.IsCpuEnabled = $true
try {
    $computer.Open()
    $rows = foreach ($hardware in $computer.Hardware) {
        $hardware.Update()
        foreach ($sensor in $hardware.Sensors) {
            if ($sensor.SensorType -eq [LibreHardwareMonitor.Hardware.SensorType]::Temperature) {
                [pscustomobject]@{
                    Hardware = [string] $hardware.Name
                    Identifier = [string] $sensor.Identifier
                    Sensor = [string] $sensor.Name
                    Celsius = [single] $sensor.Value
                }
            }
        }
        foreach ($subHardware in $hardware.SubHardware) {
            $subHardware.Update()
            foreach ($sensor in $subHardware.Sensors) {
                if ($sensor.SensorType -eq [LibreHardwareMonitor.Hardware.SensorType]::Temperature) {
                    [pscustomobject]@{
                        Hardware = [string] $subHardware.Name
                        Identifier = [string] $sensor.Identifier
                        Sensor = [string] $sensor.Name
                        Celsius = [single] $sensor.Value
                    }
                }
            }
        }
    }
    if (-not $rows) {
        throw 'LibreHardwareMonitor exposed no CPU temperature sensors.'
    }
    $rows | Format-Table -AutoSize
}
finally {
    $computer.Close()
}
