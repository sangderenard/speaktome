param(
    [Parameter(Mandatory = $true)]
    [string] $AssemblyPath,

    [string] $SensorIdentifier = ''
)

$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
Add-Type -Path $AssemblyPath
$computer = [LibreHardwareMonitor.Hardware.Computer]::new()
$computer.IsCpuEnabled = $true
$computer.IsMotherboardEnabled = $true
$computer.IsControllerEnabled = $true
$computer.IsGpuEnabled = $true

function Find-FanSensor {
    param([object] $Hardware)
    $Hardware.Update()
    foreach ($sensor in $Hardware.Sensors) {
        if ($sensor.SensorType -ne [LibreHardwareMonitor.Hardware.SensorType]::Fan) { continue }
        if ($SensorIdentifier -and [string] $sensor.Identifier -ne $SensorIdentifier) { continue }
        return $sensor
    }
    foreach ($child in $Hardware.SubHardware) {
        $match = Find-FanSensor -Hardware $child
        if ($null -ne $match) { return $match }
    }
    return $null
}

try {
    $computer.Open()
    while ([Console]::In.ReadLine() -ne $null) {
        $match = $null
        foreach ($hardware in $computer.Hardware) {
            $match = Find-FanSensor -Hardware $hardware
            if ($null -ne $match) { break }
        }
        if ($null -eq $match) {
            $wanted = if ($SensorIdentifier) { $SensorIdentifier } else { 'first available fan' }
            [Console]::Out.WriteLine("ERROR|fan sensor not found: $wanted")
        }
        elseif ($null -eq $match.Value -or [single] $match.Value -le 0) {
            [Console]::Out.WriteLine(
                'ERROR|fan sensor returned no usable RPM; run the parent program as Administrator'
            )
        }
        else {
            $value = ([single] $match.Value).ToString('R', $culture)
            [Console]::Out.WriteLine("$value|$($match.Identifier)|$($match.Name)")
        }
        [Console]::Out.Flush()
    }
}
catch {
    [Console]::Out.WriteLine("ERROR|$($_.Exception.Message)")
    [Console]::Out.Flush()
}
finally {
    $computer.Close()
}
