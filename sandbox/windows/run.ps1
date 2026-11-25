<#
.SYNOPSIS
One test run: build this checkout, copy the DLLs into the sandbox folder, start Windows Sandbox.

.DESCRIPTION
Builds the repository this script belongs to (build.cmd in an x64 Visual Studio environment, plus
decant-launch), copies the DLLs and decant-launch.exe into <Boot>\shims, archives the previous logs, starts
<Boot>\boot.wsb and, when the sandbox window is closed, prints the end of the run log. The title
only ever runs inside the sandbox, whose network boot.wsb disables. prepare.ps1 must have run once.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File sandbox\windows\run.ps1 -Boot "G:\Work\boot"
#>
param(
    # The folder prepare.ps1 made.
    [Parameter(Mandatory = $true)] [string] $Boot,
    # Where the build goes (outside the repository).
    [string] $BuildDir = (Join-Path (Split-Path $Boot) 'build_sandbox'),
    # Build and copy, but do not start the sandbox.
    [switch] $NoLaunch
)
$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Boot = (Resolve-Path -LiteralPath $Boot).Path
$wsb = Join-Path $Boot 'boot.wsb'

if (-not (Select-String -LiteralPath $wsb -SimpleMatch '<Networking>Disable</Networking>' -Quiet)) {
    throw "$wsb does not disable the network: nothing is started."
}
if (Get-Process -Name WindowsSandbox, WindowsSandboxRemoteSession -ErrorAction SilentlyContinue) {
    throw 'A Windows Sandbox is already open: close it first.'
}

# 1. Build.
$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'Visual Studio with the C++ x64 tools was not found.' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
New-Item -ItemType Directory -Force $BuildDir | Out-Null
$commit = (git -C $repo rev-parse --short HEAD)
Write-Host "Building $repo at $commit into $BuildDir"
# vcvars64 calls vswhere by name; native tools write progress to stderr, which must not stop the script.
$env:PATH = "$env:PATH;$(Split-Path $vswhere)"
$ErrorActionPreference = 'Continue'
cmd /d /c "call `"$vcvars`" >nul 2>nul && set `"DECANT_BUILD=$BuildDir`" && call `"$repo\build.cmd`"" 2>&1 | ForEach-Object { "$_" }
$code = $LASTEXITCODE
if ($code) { throw "build.cmd failed ($code)." }
Push-Location $repo
cargo build --release --locked --target-dir (Join-Path $BuildDir 'cargo') -p decant-launch --bin decant-launch 2>&1 | ForEach-Object { "$_" }
$code = $LASTEXITCODE
Pop-Location
$ErrorActionPreference = 'Stop'
if ($code) { throw "Building decant-launch failed ($code)." }

# 2. Copy the DLLs and decant-launch.
# Every DLL build.cmd leaves in the build folder: the list grows (XboxBvhBuilder.dll is decant_bvh.dll under the
# package name, so the console BVH builder is replaced; decant_xma2.dll), a fixed list misses new ones.
$dlls = @(Get-ChildItem -LiteralPath $BuildDir -Filter *.dll -File)
foreach ($d in 'xmem.dll', 'd3d12_xs.dll', 'xgameruntime.dll', 'XboxBvhBuilder.dll') {
    if (-not ($dlls | Where-Object Name -eq $d)) { throw "$d is missing from the build." }
}
foreach ($d in $dlls) { Copy-Item -LiteralPath $d.FullName (Join-Path $Boot 'shims\dll') -Force }
Copy-Item (Join-Path $BuildDir 'cargo\release\decant-launch.exe') (Join-Path $Boot 'shims') -Force
Write-Host "$($dlls.Count) DLLs ($(($dlls | ForEach-Object Name) -join ', ')) and decant-launch.exe copied"
if ($NoLaunch) { return }

# 3. Archive the previous logs, then start the sandbox.
$stamp = Get-Date -Format 'yyyy-MM-dd_HHmm'
$archive = Join-Path $Boot "archives\$stamp"
foreach ($f in 'logs', 'crash') {
    $from = Join-Path $Boot $f
    New-Item -ItemType Directory -Force (Join-Path $archive $f) | Out-Null
    Get-ChildItem -LiteralPath $from -Force | Move-Item -Destination (Join-Path $archive $f)
}
"$commit, run of $stamp" | Set-Content -Encoding utf8 (Join-Path $Boot 'logs\commit.txt')
Write-Host 'Starting Windows Sandbox (no network). Leave it open until the title exits (10 minutes at most), then close it.'
Start-Process "$env:windir\System32\WindowsSandbox.exe" -ArgumentList "`"$wsb`""
Start-Sleep -Seconds 10
while (Get-Process -Name WindowsSandboxRemoteSession, WindowsSandboxServer -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 5 }

# 4. Result.
Write-Host "`n=== Result ($commit) ==="
$console = Join-Path $Boot 'logs\decant_launch_console.txt'
if (Test-Path $console) { Get-Content -LiteralPath $console -Tail 15 } else { Write-Host 'No decant_launch_console.txt: see logs\checks.txt (network guard, GameInput).' }
Write-Host "Full logs: $(Join-Path $Boot 'logs') (keep them on this machine and quote only the line where the boot stops)."
