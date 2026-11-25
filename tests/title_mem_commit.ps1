# Commit check of title memory (d3d12_xs title_mem). The pool heaps must give their commit back while
# the process lives and must not outlive it. It measures the commit of the process it starts (committed
# private and mapped pages, which the test itself reads with VirtualQuery and prints), never the free
# commit of the system (a game run or any other process changes that meanwhile). The process counters
# of Windows do not show the commit of sections, so the number comes from the test. A normal run of
# d3d12xs_test prints its commit around the arena commit and XMemFree, it passes when XMemFree gave it
# back within 128 MiB (the pool and the device stay while the process lives). A second run with
# DECANT_TEST_HOLD=1 (it waits after its checks and prints HOLDING pid and its commit) must hold steady, then it is killed and must be gone (the
# system took its commit charge back with the process). Run it before every push that touches
# title_mem.rs or xmem.
#
# Usage: powershell -File tests\title_mem_commit.ps1 [-Release] [-NoBuild] [-LimitMiB 128]
# Needs a D3D12 12_0 GPU. A machine without one skips the title memory checks, then the commit check
# passes without having tested anything, and the script says so.
param(
    [switch]$Release,
    [switch]$NoBuild,
    [int]$LimitMiB = 128
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$workspace = $root
$profileDir = if ($Release) { 'release' } else { 'debug' }

if (-not $NoBuild) {
    Push-Location $workspace
    try {
        $flags = @('build', '--locked', '-p', 'decant-d3d12x', '-p', 'decant-xmem', '--bins')
        if ($Release) { $flags += '--release' }
        & cargo @flags
        if ($LASTEXITCODE -ne 0) { throw 'cargo build failed' }
    } finally {
        Pop-Location
    }
}

$bin = Join-Path $workspace "target\$profileDir"
$test = Join-Path $bin 'd3d12xs_test.exe'
$dll = Join-Path $bin 'd3d12_xs.dll'
foreach ($p in @($test, $dll, (Join-Path $bin 'xmem.dll'))) {
    if (-not (Test-Path $p)) { throw "missing $p" }
}

function Get-ProcessCommitMiB($p) {
    try { $p.Refresh(); if ($p.HasExited) { return $null }; [int]($p.PrivateMemorySize64 / 1MB) } catch { $null }
}

$env:DECANT_D3D12X_TITLE_MEM = '1'
$env:DECANT_XMEM_PHYSICAL_MIB = '256'
Remove-Item Env:DECANT_TEST_HOLD -ErrorAction SilentlyContinue

$failed = $false
function Check($ok, $text) {
    if ($ok) { Write-Host "ok    $text" } else { Write-Host "FAIL  $text"; $script:failed = $true }
}

# Starts the test and watches that process every 250 ms until it exits, or until $StopOn shows in its
# output. Returns the process, the samples of its private memory in MiB (an outside view, it does not
# show sections) and the output file.
function Start-Sampled($hold, $stopOn) {
    $log = Join-Path ([IO.Path]::GetTempPath()) "title_mem_commit_${PID}_$([guid]::NewGuid().ToString('N')).txt"
    if ($hold) { $env:DECANT_TEST_HOLD = '1' }
    $proc = Start-Process -FilePath $test -ArgumentList "`"$dll`"" -RedirectStandardOutput $log -NoNewWindow -PassThru
    $null = $proc.Handle  # keeps the handle, so ExitCode is readable after the exit
    Remove-Item Env:DECANT_TEST_HOLD -ErrorAction SilentlyContinue
    $peak = 0
    $samples = @()
    for ($i = 0; $i -lt 480; $i++) {
        $c = Get-ProcessCommitMiB $proc
        if ($null -ne $c) { $samples += $c; if ($c -gt $peak) { $peak = $c } }
        if ($proc.HasExited) { break }
        if ($stopOn -and (Test-Path $log) -and (Select-String -Path $log -Pattern $stopOn -Quiet)) { break }
        Start-Sleep -Milliseconds 250
    }
    [pscustomobject]@{ Proc = $proc; Peak = $peak; Samples = $samples; Log = $log }
}

# 1. A normal run: the process reports its own commit charge before and after the checks.
$run = Start-Sampled $false $null
if (-not $run.Proc.HasExited) { Stop-Process -Id $run.Proc.Id -Force; Check $false 'd3d12xs_test ends on its own' }
$null = $run.Proc.Handle
$run.Proc.WaitForExit()
$exit = $run.Proc.ExitCode
$out = Get-Content $run.Log
$out | Select-String 'process commit|skip  title|FAIL' | ForEach-Object { Write-Host "  $_" }
Check ($exit -eq 0) "d3d12xs_test exits with 0 (exit $exit)"
$skipped = [bool]($out | Select-String 'skip  title memory|skip  no D3D12|skip  no xmem')
$arena = $out | Select-String 'process commit around the arena: (\d+) MiB before, (\d+) MiB after XMemFree'
if ($arena) {
    $b = [int]$arena.Matches[0].Groups[1].Value
    $a = [int]$arena.Matches[0].Groups[2].Value
    Check (($a - $b) -le $LimitMiB) "XMemFree gave the arena commit back within $LimitMiB MiB ($b MiB before the arena, $a MiB after)"
} elseif (-not $skipped) {
    Check $false 'the arena commit line is in the output'
}
Remove-Item $run.Log -ErrorAction SilentlyContinue

# 2. A run that is held and then killed.
$held = Start-Sampled $true 'HOLDING pid'
$proc = $held.Proc
$holding = (-not $proc.HasExited) -and (Select-String -Path $held.Log -Pattern 'HOLDING pid' -Quiet)
if ($holding) {
    $hold = Select-String -Path $held.Log -Pattern 'HOLDING pid \d+ commit (\d+) MiB'
    $m2 = Select-String -Path $held.Log -Pattern 'process commit after the title memory checks: (\d+) MiB'
    if ($hold -and $m2) {
        $h = [int]$hold.Matches[0].Groups[1].Value
        $e = [int]$m2.Matches[0].Groups[1].Value
        Write-Host "commit of the held process: $h MiB ($e MiB right after the checks)"
        Check ([Math]::Abs($h - $e) -le $LimitMiB) "the held process keeps its commit steady within $LimitMiB MiB"
    } else {
        Check $false 'the commit lines are in the output of the held run'
    }
} else {
    Write-Host 'the run did not reach HOLDING (it skipped or failed), nothing to kill'
    $skipped = $true
}
if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
$proc.WaitForExit()
Start-Sleep -Seconds 1
Check ($null -eq (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) 'the killed process is gone, the system took its commit charge back'
Remove-Item $held.Log -ErrorAction SilentlyContinue

if ($skipped) { Write-Host 'note  the title memory checks were skipped on this machine, the commit check tested nothing' }
if ($failed) { Write-Host 'title_mem_commit: FAILED'; exit 1 }
Write-Host 'title_mem_commit: passed'
