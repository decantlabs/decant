<#
.SYNOPSIS
  Checks the tools a Decant development machine needs and offers to install the missing ones.

.DESCRIPTION
  Each tool is checked first. A missing one is installed with winget only after you answer yes, or
  without asking when -Yes is given. -Check only reports and installs nothing. Nothing is downloaded
  from anywhere else, and no game file, SDK or runtime outside this list is fetched.
  Optional components (FidelityFX, DirectStorage, the public GDK, the translator sources) are
  listed with where to get them; they are never installed automatically.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File setup.ps1 -Check
  powershell -ExecutionPolicy Bypass -File setup.ps1
  powershell -ExecutionPolicy Bypass -File setup.ps1 -Yes
#>
param(
    [switch]$Check,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$results = New-Object System.Collections.Generic.List[object]

function Get-Tool([string]$name, [string[]]$extraPaths) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in $extraPaths) {
        if ($p -and (Test-Path $p)) { return $p }
    }
    return $null
}

function Get-FirstLine([string]$exe, [string[]]$arguments) {
    try {
        $out = & $exe @arguments 2>&1 | Select-Object -First 1
        return "$out".Trim()
    } catch {
        return ''
    }
}

function Test-VisualCpp {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) { return $null }
    $path = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($path) { return "$path".Trim() }
    return $null
}

function Test-SpirvVal {
    $candidates = @()
    if ($env:SPIRV_TOOLS) { $candidates += (Join-Path $env:SPIRV_TOOLS 'spirv-val.exe'); $candidates += (Join-Path $env:SPIRV_TOOLS 'bin\spirv-val.exe') }
    if ($env:VULKAN_SDK) { $candidates += (Join-Path $env:VULKAN_SDK 'Bin\spirv-val.exe') }
    $exe = Get-Tool 'spirv-val' $candidates
    if (-not $exe) { return $null }
    $line = Get-FirstLine $exe @('--version')
    if ($line -match 'v(\d{4})\.') {
        if ([int]$Matches[1] -lt 2026) { return "$exe ($line) - TOO OLD: 2026.x needed, 2025.1 rejects every mesh shader" }
    }
    return "$exe ($line)"
}

# Name, test, winget id, extra winget arguments, required
$tools = @(
    @{ Name = 'Git'; Test = { Get-Tool 'git' @() }; Id = 'Git.Git'; Args = @(); Required = $true },
    @{ Name = 'GitHub CLI (gh)'; Test = { Get-Tool 'gh' @() }; Id = 'GitHub.cli'; Args = @(); Required = $false },
    @{ Name = 'Visual Studio C++ x64 tools + Windows SDK'; Test = { Test-VisualCpp }; Id = 'Microsoft.VisualStudio.2022.Community';
       Args = @('--override', '--wait --passive --add Microsoft.VisualStudio.Workload.NativeDesktop --includeRecommended'); Required = $true },
    @{ Name = 'Rust (rustup)'; Test = { Get-Tool 'rustup' @((Join-Path $HOME '.cargo\bin\rustup.exe')) }; Id = 'Rustlang.Rustup'; Args = @(); Required = $true },
    @{ Name = 'LLVM (clang-cl, ld.lld)'; Test = { Get-Tool 'clang-cl' @((Join-Path $env:ProgramFiles 'LLVM\bin\clang-cl.exe')) }; Id = 'LLVM.LLVM'; Args = @(); Required = $false },
    @{ Name = 'CMake'; Test = { Get-Tool 'cmake' @((Join-Path $env:ProgramFiles 'CMake\bin\cmake.exe'), (Join-Path "$(Test-VisualCpp)" 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe')) }; Id = 'Kitware.CMake'; Args = @(); Required = $false },
    @{ Name = 'Ninja'; Test = { Get-Tool 'ninja' @((Join-Path "$(Test-VisualCpp)" 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe')) }; Id = 'Ninja-build.Ninja'; Args = @(); Required = $false },
    @{ Name = 'Vulkan SDK (validation layers)'; Test = { if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) { $env:VULKAN_SDK } else { $null } }; Id = 'KhronosGroup.VulkanSDK'; Args = @(); Required = $false },
    @{ Name = 'SPIRV-Tools spirv-val 2026.x'; Test = { Test-SpirvVal }; Id = 'KhronosGroup.VulkanSDK'; Args = @(); Required = $false }
)

$winget = Get-Command winget -ErrorAction SilentlyContinue
$installed = @{}

Write-Host ''
Write-Host 'Decant - development machine check' -ForegroundColor Cyan
Write-Host ''

foreach ($tool in $tools) {
    $found = & $tool.Test
    if (-not $found -and -not $Check) {
        $where = if ($tool.Required) { 'required' } else { 'needed for the translator, GPU tests or CI checks' }
        if (-not $winget) {
            Write-Host ("  {0}: missing ({1}). winget is not available: install it by hand." -f $tool.Name, $where) -ForegroundColor Yellow
        } elseif ($installed.ContainsKey($tool.Id)) {
            Write-Host ("  {0}: installed with {1} above; open a new terminal to see it." -f $tool.Name, $tool.Id)
        } else {
            $answer = 'y'
            if (-not $Yes) {
                $answer = Read-Host ("  {0} is missing ({1}). Install it with winget ({2})? [y/N]" -f $tool.Name, $where, $tool.Id)
            }
            if ($answer -match '^(y|yes|o|oui)$') {
                $arguments = @('install', '--id', $tool.Id, '-e', '--accept-package-agreements', '--accept-source-agreements') + $tool.Args
                & winget @arguments
                $installed[$tool.Id] = $true
                $found = & $tool.Test
                if (-not $found) { $found = $null; $tool.Note = 'installed: open a new terminal and run setup.ps1 -Check' }
            }
        }
    }
    $status = if ($found -and "$found" -notmatch 'TOO OLD') { 'ok' } elseif ($found) { 'old' } else { 'missing' }
    $results.Add([pscustomobject]@{ Tool = $tool.Name; Status = $status; Required = $tool.Required; Detail = $(if ($found) { "$found" } elseif ($tool.Note) { $tool.Note } else { "winget install --id $($tool.Id) -e" }) })
}

# The Rust toolchain is pinned in rust-toolchain.toml (with clippy and rustfmt).
$rustupExe = Get-Tool 'rustup' @((Join-Path $HOME '.cargo\bin\rustup.exe'))
if ($rustupExe) {
    $workspace = $Root
    Push-Location $workspace
    try {
        $active = (& $rustupExe show active-toolchain 2>$null | Select-Object -First 1)
        if (-not "$active" -or "$active" -match 'not installed') {
            $active = $null
        }
        if (-not $active -and -not $Check) {
            $answer = 'y'
            if (-not $Yes) { $answer = Read-Host '  Install the pinned Rust toolchain of this repository (rustup toolchain install)? [y/N]' }
            if ($answer -match '^(y|yes|o|oui)$') {
                & $rustupExe toolchain install
                $active = (& $rustupExe show active-toolchain 2>$null | Select-Object -First 1)
            }
        }
        $results.Add([pscustomobject]@{ Tool = 'Pinned Rust toolchain (clippy, rustfmt)'; Status = $(if ($active) { 'ok' } else { 'missing' }); Required = $true; Detail = $(if ($active) { "$active" } else { 'rustup toolchain install' }) })
    } finally {
        Pop-Location
    }
}

Write-Host ''
$results | Format-Table -AutoSize -Wrap

Write-Host 'Optional, installed by hand only when a task needs them:' -ForegroundColor Cyan
Write-Host '  - FidelityFX SDK (AMD, MIT headers): build.cmd "<FidelityFX root>" for the upscaling test'
Write-Host '  - DirectStorage 1.3 (NuGet Microsoft.Direct3D.DirectStorage): dstorage.dll beside a title for PC DirectStorage (loaded at run time). The GDeflate folder of microsoft/DirectStorage in DECANT_GDEFLATE_ROOT adds the GDeflate decoder when build.cmd builds the DirectStorage layer'
Write-Host '  - Public Microsoft GDK: https://github.com/microsoft/GDK (never the console GDKX)'
Write-Host '  - Linux track: Rust target x86_64-pc-windows-gnu (make -C src/gdk), Docker for the air-gapped sandbox (sandbox/README.md)'
Write-Host '  Licence: LICENSE (MIT). Credits: CREDITS.md'
Write-Host ''

$missing = @($results | Where-Object { $_.Required -and $_.Status -ne 'ok' })
if ($missing.Count -gt 0) {
    Write-Host ("{0} required tool(s) missing. Run setup.ps1 without -Check to install them." -f $missing.Count) -ForegroundColor Yellow
    exit 1
}
Write-Host 'Required tools present. Next: build.cmd, then cargo build --release --locked.' -ForegroundColor Green
exit 0
