param(
    [Parameter(Mandatory = $true)]
    [string] $AssemblyPath,

    [string] $SensorIdentifier = '/amdcpu/0/temperature/2'
)

$ErrorActionPreference = 'Stop'
$culture = [Globalization.CultureInfo]::InvariantCulture
Add-Type -Path $AssemblyPath
$computer = [LibreHardwareMonitor.Hardware.Computer]::new()
$computer.IsCpuEnabled = $true

try {
    $computer.Open()
    while ([Console]::In.ReadLine() -ne $null) {
        $match = $null
        foreach ($hardware in $computer.Hardware) {
            $hardware.Update()
            foreach ($sensor in $hardware.Sensors) {
                if ([string] $sensor.Identifier -eq $SensorIdentifier) {
                    $match = $sensor
                    break
                }
            }
            if ($null -ne $match) { break }
        }
        if ($null -eq $match) {
            [Console]::Out.WriteLine("ERROR|sensor not found: $SensorIdentifier")
        }
        elseif ($null -eq $match.Value -or [single] $match.Value -le 0) {
            [Console]::Out.WriteLine(
                "ERROR|sensor returned no usable value; run the parent program as Administrator"
            )
        }
        else {
            [Console]::Out.WriteLine(([single] $match.Value).ToString('R', $culture))
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
