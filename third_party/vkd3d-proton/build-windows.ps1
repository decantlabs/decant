# Builds the x64 Windows d3d12.dll and d3d12core.dll of vkd3d-proton with vkd3d-decant-shader.patch applied.
# vkd3d-proton is LGPL-2.1, see README.md in this folder. Nothing built here is committed to the repository.
#
# Usage:  powershell -ExecutionPolicy Bypass -File third_party\vkd3d-proton\build-windows.ps1 [-Work <dir>]
# Needs:  git, Python 3 (the py launcher), the Vulkan SDK (glslangValidator) and network access for the first run.
# Output: <Work>\out\d3d12.dll and <Work>\out\d3d12core.dll, with their SHA-256 printed at the end.

param(
    [string]$Work = (Join-Path $env:USERPROFILE 'repos\vkd3d-build'),
    [string]$VulkanSdk = $(if ($env:VULKAN_SDK) { $env:VULKAN_SDK } else { 'C:\VulkanSDK\1.4.363.0' })
)

$ErrorActionPreference = 'Stop'

# Pinned inputs
$VkdCommit  = 'cae8353ed7959c448ac976342052de1ec2b81715'
$VkdUrl     = 'https://github.com/HansKristian-Work/vkd3d-proton'
$WinLibsTag = '16.2.0posix-23.1.3-14.0.0-ucrt-r3'
$WinLibsZip = 'winlibs-x86_64-posix-seh-gcc-16.2.0-mingw-w64ucrt-14.0.0-r3.zip'
$WinLibsUrl = "https://github.com/brechtsanders/winlibs_mingw/releases/download/$WinLibsTag/$WinLibsZip"
$WinLibsSha = '3c69305081e4a647144339a936244fcf7d3d9e558319e1ad7be2cd2886d4b6a1'
$MesonVer   = '1.12.1'
$NinjaVer   = '1.13.2'

$Patch = Join-Path $PSScriptRoot 'vkd3d-decant-shader.patch'
if (-not (Test-Path $Patch)) { throw "Missing $Patch" }
if (-not (Test-Path (Join-Path $VulkanSdk 'Bin\glslangValidator.exe'))) { throw "glslangValidator not found in $VulkanSdk\Bin" }

function Run([string]$exe, [string[]]$arguments) {
    & $exe @arguments
    if ($LASTEXITCODE -ne 0) { throw "$exe $($arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

New-Item -ItemType Directory -Force $Work, "$Work\dl", "$Work\out" | Out-Null

# 1. mingw-w64 toolchain (gcc, widl, windres) from WinLibs, hash checked
$mingw = Join-Path $Work 'toolchain\mingw64'
if (-not (Test-Path "$mingw\bin\widl.exe")) {
    $zip = Join-Path $Work "dl\$WinLibsZip"
    if (-not (Test-Path $zip)) { Invoke-WebRequest $WinLibsUrl -OutFile $zip }
    $got = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($got -ne $WinLibsSha) { throw "SHA-256 mismatch for $WinLibsZip : $got" }
    Expand-Archive $zip (Join-Path $Work 'toolchain') -Force
}

# 2. meson and ninja in a venv
$venv = Join-Path $Work 'venv'
if (-not (Test-Path "$venv\Scripts\meson.exe")) {
    Run 'py' @('-m', 'venv', $venv)
    Run "$venv\Scripts\python.exe" @('-m', 'pip', 'install', "meson==$MesonVer", "ninja==$NinjaVer")
}

# 3. fresh checkout at the pinned commit with submodules, then the patch
$src = Join-Path $Work 'src'
$build = Join-Path $Work 'build64'
if (Test-Path $src)   { Remove-Item -Recurse -Force $src }
if (Test-Path $build) { Remove-Item -Recurse -Force $build }
Run 'git' @('-c', 'core.longpaths=true', 'clone', '--recursive', $VkdUrl, $src)
Run 'git' @('-C', $src, 'config', 'core.longpaths', 'true')
Run 'git' @('-C', $src, 'checkout', $VkdCommit)
Run 'git' @('-C', $src, '-c', 'core.longpaths=true', 'submodule', 'update', '--init', '--recursive')
Run 'git' @('-C', $src, 'apply', $Patch)

# 4. build: x64, release, stripped, as package-release.sh does, but native instead of cross
$env:PATH = "$mingw\bin;$VulkanSdk\Bin;$venv\Scripts;$env:PATH"
Push-Location $src
try {
    Run "$venv\Scripts\meson.exe" @('setup', '--buildtype', 'release', '--strip', '-Denable_tests=false', $build)
    Run "$venv\Scripts\ninja.exe" @('-C', $build)
} finally { Pop-Location }

# 5. collect
Copy-Item "$build\libs\d3d12\d3d12.dll", "$build\libs\d3d12core\d3d12core.dll" "$Work\out" -Force
$bytes = [IO.File]::ReadAllBytes("$Work\out\d3d12core.dll")
if (-not [Text.Encoding]::ASCII.GetString($bytes).Contains('VKD3D_DECANT_SHADER_DIR')) { throw 'Patch is not in d3d12core.dll' }
Get-FileHash "$Work\out\d3d12.dll", "$Work\out\d3d12core.dll" -Algorithm SHA256 | ForEach-Object { "$($_.Hash.ToLower())  $(Split-Path $_.Path -Leaf)" }
