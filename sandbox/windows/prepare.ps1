<#
.SYNOPSIS
Prepares the Windows Sandbox folder (once per machine, or when the game package moves).

.DESCRIPTION
Creates <Boot>\package (the game package as hard links, nothing copied, gameos.xvd left out),
<Boot>\shims (boot.cmd, the DLL folder, the GameInput redistributable), <Boot>\logs, <Boot>\crash,
and <Boot>\boot.wsb with this machine's paths and the network disabled. It only reads the package.
Nothing is started: run.ps1 builds the DLLs and starts the sandbox.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File sandbox\windows\prepare.ps1 -Package "G:\Games\Title" -Boot "G:\Work\boot" -GameInputMsi "G:\Downloads\GameInputRedist.msi"
#>
param(
    # The extracted game package (MicrosoftGame.config, the data files, the package DLLs).
    [Parameter(Mandatory = $true)] [string] $Package,
    # The folder the sandbox maps. Must be on the same NTFS volume as the package (hard links).
    [Parameter(Mandatory = $true)] [string] $Boot,
    # Optional: an executable to use instead of the package's own (same file name in the package).
    [string] $Exe,
    # The title's executable name, as decant-launch starts it (default: the first Executable of MicrosoftGame.config).
    [string] $ExeName,
    # Microsoft's GameInput redistributable (NuGet Microsoft.GameInput, file redist\GameInputRedist.msi).
    [string] $GameInputMsi,
    # Memory given to the sandbox, in MiB.
    [int] $MemoryMB = 20480
)
$ErrorActionPreference = 'Stop'

$Package = (Resolve-Path -LiteralPath $Package).Path.TrimEnd('\')
New-Item -ItemType Directory -Force $Boot | Out-Null
$Boot = (Resolve-Path -LiteralPath $Boot).Path.TrimEnd('\')
if ((Split-Path -Qualifier $Package) -ne (Split-Path -Qualifier $Boot)) {
    throw "The package ($Package) and the boot folder ($Boot) must be on the same drive: hard links do not cross volumes."
}
if ($Boot.StartsWith($Package + '\', [StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "Note: $Boot is inside the package folder; it is skipped when the package is linked."
}
if (-not (Test-Path -LiteralPath (Join-Path $Package 'MicrosoftGame.config'))) {
    Write-Warning "No MicrosoftGame.config in ${Package}: is this the extracted package?"
}
if (-not (Test-Path "$env:windir\System32\WindowsSandbox.exe")) {
    Write-Warning 'Windows Sandbox is not installed. In an administrator PowerShell: Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All, then reboot.'
}

# 1. The package as hard links (read-only in the sandbox). gameos.xvd is never used.
$pkgOut = Join-Path $Boot 'package'
New-Item -ItemType Directory -Force $pkgOut | Out-Null
$skip = @('gameos.xvd')
$linked = 0
Get-ChildItem -LiteralPath $Package -Recurse -File -Force | Where-Object {
    $skip -notcontains $_.Name -and -not $_.FullName.StartsWith($Boot + '\', [StringComparison]::OrdinalIgnoreCase)
} | ForEach-Object {
    $rel = $_.FullName.Substring($Package.Length + 1)
    $target = Join-Path $pkgOut $rel
    New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null
    if (-not (Test-Path -LiteralPath $target)) {
        New-Item -ItemType HardLink -Path $target -Target $_.FullName | Out-Null
    }
    $linked++
}
if (-not $ExeName) {
    $cfg = Join-Path $Package 'MicrosoftGame.config'
    if (Test-Path -LiteralPath $cfg) {
        $node = ([xml](Get-Content -LiteralPath $cfg -Raw)).SelectSingleNode('//ExecutableList/Executable')
        if ($node) { $ExeName = $node.GetAttribute('Name') }
    }
    if (-not $ExeName) { throw 'No executable name: pass -ExeName.' }
}
if ($Exe) {
    Copy-Item -LiteralPath $Exe -Destination (Join-Path $pkgOut $ExeName) -Force
}
$exePath = Join-Path $pkgOut $ExeName
if (-not (Test-Path -LiteralPath $exePath)) { throw "$ExeName is not in the package; pass -Exe or -ExeName." }
$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $exePath).Hash
Write-Host "package: $linked files linked, $ExeName SHA-256 $hash"

# 2. shims: boot.cmd, the DLL folder (filled by run.ps1), the GameInput redistributable.
$shims = Join-Path $Boot 'shims'
New-Item -ItemType Directory -Force (Join-Path $shims 'dll'), (Join-Path $shims 'redist'), (Join-Path $Boot 'logs'), (Join-Path $Boot 'crash') | Out-Null
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'boot.cmd') -Destination $shims -Force
"set `"EXE=$ExeName`"" | Set-Content -Encoding ascii (Join-Path $shims 'boot_env.cmd')
if ($GameInputMsi) {
    $sig = Get-AuthenticodeSignature -LiteralPath $GameInputMsi
    if ($sig.Status -ne 'Valid' -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw "$GameInputMsi is not validly signed by Microsoft ($($sig.Status))."
    }
    Copy-Item -LiteralPath $GameInputMsi -Destination (Join-Path $shims 'redist\GameInputRedist.msi') -Force
    Write-Host 'GameInput redistributable: copied (signed by Microsoft)'
} elseif (-not (Test-Path (Join-Path $shims 'redist\GameInputRedist.msi'))) {
    Write-Warning 'No GameInputRedist.msi: GameInputCreate will fail in the sandbox. Get it from the NuGet package Microsoft.GameInput (redist\GameInputRedist.msi) and pass -GameInputMsi.'
}

# 3. boot.wsb: no network, vGPU, read-only package and shims, writable logs and crash.
$wsb = @"
<Configuration>
  <Networking>Disable</Networking>
  <vGPU>Enable</vGPU>
  <ClipboardRedirection>Disable</ClipboardRedirection>
  <PrinterRedirection>Disable</PrinterRedirection>
  <AudioInput>Disable</AudioInput>
  <VideoInput>Disable</VideoInput>
  <MemoryInMB>$MemoryMB</MemoryInMB>
  <MappedFolders>
    <MappedFolder><HostFolder>$pkgOut</HostFolder><SandboxFolder>C:\boot\package</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$shims</HostFolder><SandboxFolder>C:\boot\shims</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$(Join-Path $Boot 'logs')</HostFolder><SandboxFolder>C:\boot\logs</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
    <MappedFolder><HostFolder>$(Join-Path $Boot 'crash')</HostFolder><SandboxFolder>C:\boot\crash</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
  </MappedFolders>
  <LogonCommand>
    <Command>C:\boot\shims\boot.cmd</Command>
  </LogonCommand>
</Configuration>
"@
[IO.File]::WriteAllText((Join-Path $Boot 'boot.wsb'), $wsb, (New-Object Text.UTF8Encoding $false))
Write-Host "Ready: $Boot. Next: sandbox\windows\run.ps1 -Boot `"$Boot`""
