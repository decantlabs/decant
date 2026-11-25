@echo off
rem Runs INSIDE Windows Sandbox only (the LogonCommand of boot.wsb, written by prepare.ps1).
rem C:\boot\package and C:\boot\shims are mapped read-only; C:\boot\logs and C:\boot\crash are the only
rem writable mappings. The title is never started if the sandbox has any network.
if /i not "%USERNAME%"=="WDAGUtilityAccount" (echo Not in Windows Sandbox: refusing to run. & exit /b 3)
set "EXE=title.exe"
if exist C:\boot\shims\boot_env.cmd call C:\boot\shims\boot_env.cmd
set "CHK=C:\boot\logs\checks.txt"

rem The title's crash handler writes t:\<exe>-<date>.crash.log and a minidump: T: is the crash folder.
subst T: C:\boot\crash
echo [T:] > "%CHK%"
subst >> "%CHK%" 2>&1
echo test> T:\t_write_test.txt && (echo T: writable >> "%CHK%") || (echo T: NOT writable >> "%CHK%")

rem Network guard: nothing answers a ping, no name resolves, no default route. Otherwise stop here.
echo [network] >> "%CHK%"
ipconfig >> "%CHK%" 2>&1
ping -n 2 -w 1000 1.1.1.1 > "%TEMP%\ping.txt" 2>&1
type "%TEMP%\ping.txt" >> "%CHK%"
findstr /c:"TTL=" "%TEMP%\ping.txt" >nul && (echo NETWORK REACHABLE: title not started >> "%CHK%" & exit /b 4)
nslookup example.org > "%TEMP%\dns.txt" 2>&1
findstr /b /c:"Name:" /c:"Nom :" /c:"Nom:" "%TEMP%\dns.txt" >nul && (echo DNS RESOLVES: title not started >> "%CHK%" & type "%TEMP%\dns.txt" >> "%CHK%" & exit /b 4)
route print -4 0.0.0.0 | findstr /c:"0.0.0.0          0.0.0.0" >nul && (echo DEFAULT ROUTE PRESENT: title not started >> "%CHK%" & exit /b 4)
echo no network: ping fails, DNS fails, no default route >> "%CHK%"

rem GameInputCreate fails (after about 5 s) without the GameInput service, and the title breaks into the
rem debugger. Microsoft's redistributable (NuGet Microsoft.GameInput, redist\GameInputRedist.msi) installs offline.
echo [GameInput] >> "%CHK%"
if exist C:\boot\shims\redist\GameInputRedist.msi (
  start "" /wait msiexec /i C:\boot\shims\redist\GameInputRedist.msi /quiet /norestart /l*v C:\boot\logs\gameinput_install.log
  echo msiexec exit %ERRORLEVEL% >> "%CHK%"
) else (
  echo GameInputRedist.msi missing: GameInputCreate will fail >> "%CHK%"
)
sc query GameInputSvc >> "%CHK%" 2>&1

rem Only read when D3D12 goes through vkd3d-proton; Windows D3D12 ignores it.
set "VKD3D_CONFIG=descriptor_heap"
set "DECANT_LOG=1"
set "DECANT_LOG_PATH=C:\boot\logs\decant_session.log"
rem decant-launch maps the Xbox game drive G: to the overlay itself for the length of the run.
C:\boot\shims\decant-launch.exe all --package C:\boot\package --overlay C:\boot\overlay --shims C:\boot\shims\dll --exe "%EXE%" --log C:\boot\logs\decant_launch.log --timeout 600 > C:\boot\logs\decant_launch_console.txt 2>&1
echo exit code %ERRORLEVEL% >> C:\boot\logs\decant_launch_console.txt
