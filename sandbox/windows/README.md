# Windows Sandbox test loop

On Windows, the title only ever runs inside Windows Sandbox, a throwaway VM. The sandbox's network is
switched off by its configuration, and a guard checks again before the start. The sandbox sees four
folders of the host:

| Host folder | In the sandbox | Access |
|---|---|---|
| `<Boot>\package` | `C:\boot\package` | read only: the title package as hard links (nothing copied, `gameos.xvd` left out) |
| `<Boot>\shims` | `C:\boot\shims` | read only: `boot.cmd`, `boot_env.cmd`, `decant-launch.exe`, `dll\` (every DLL `build.cmd` produces, `XboxBvhBuilder.dll` and `decant_xma2.dll` included), `redist\GameInputRedist.msi` |
| `<Boot>\logs` | `C:\boot\logs` | writable: `checks.txt`, `decant_launch_console.txt`, `decant_launch.log`, `decant_session.log`, `commit.txt` |
| `<Boot>\crash` | `C:\boot\crash` = `T:` | writable: the title's own crash log and minidump |

`boot.cmd` runs at sandbox logon. It does the following, in order:
1. Refuses to run outside the sandbox.
2. Maps `T:`.
3. Stops unless nothing answers a ping, no name resolves and there is no default route.
4. Installs Microsoft's GameInput redistributable. Without it `GameInputCreate` fails after about 5 s and the title breaks into the debugger.
5. Runs `decant-launch all` with a 600 s limit. decant-launch stages the package into `C:\boot\overlay`, maps `G:` to it (the Xbox game drive of the title), and logs every DLL, debug line and exception.

## Once per machine

1. **Windows 11 Pro / Enterprise with Windows Sandbox.** In an administrator PowerShell, then reboot:
   `Enable-WindowsOptionalFeature -Online -FeatureName Containers-DisposableClientVM -All`
2. **Build tools:** Visual Studio with the C++ x64 tools, and the Rust toolchain of `rust-toolchain.toml`.
3. **GameInput redistributable:** download the NuGet package `Microsoft.GameInput` from nuget.org (4.1.315 is known to work). It is a zip file. Take `redist/GameInputRedist.msi` out of it. prepare.ps1 checks that it is signed by Microsoft.
4. **Prepare.** `-Boot` must be on the same NTFS drive as the package, because of the hard links:
   ```powershell
   powershell -ExecutionPolicy Bypass -File sandbox\windows\prepare.ps1 -Package "<extracted package>" -Boot "<work folder>\boot" -GameInputMsi "<path>\GameInputRedist.msi"
   ```
   `-Exe <file>` uses another executable than the package's own, `-ExeName` changes the name decant-launch starts (default the name from `MicrosoftGame.config`). Run prepare.ps1 again when the package moves.

## Each test run

```powershell
powershell -ExecutionPolicy Bypass -File sandbox\windows\run.ps1 -Boot "<work folder>\boot"
```

run.ps1 does the following:
1. Builds this checkout (`build.cmd`, then `decant-launch`) into `<work folder>\build_sandbox`.
2. Copies every DLL of the build (not a fixed list) and `decant-launch.exe` into `shims`.
3. Archives the previous `logs` and `crash` under `<Boot>\archives\<date>`.
4. Starts the sandbox and waits until you close it.
5. Prints the end of `decant_launch_console.txt`.

Leave the sandbox open until the console log shows `exit code` (10 minutes at most). `-NoLaunch` builds and copies only.

Then read where the run stopped in `decant_launch.log`. Quote only the one or two lines that show it when you report it, and keep the logs on your machine. Fix the cause, run the checks from `CLAUDE.md`, and run again.

## Setting this up on another machine

* **Do not edit the scripts for your paths.** Every machine specific path is a parameter. Ask the user three things:
  * where the extracted title package is (`-Package`).
  * which work folder to use on the same drive (`-Boot`, outside the repository).
  * where the GameInput MSI is (`-GameInputMsi`).

  Then run prepare.ps1 once and run.ps1 for each test.
* `boot.wsb` is generated with absolute host paths. Never commit it, prepare.ps1 writes it into `-Boot`.
* **Errors from prepare.ps1:**
  * Package and `-Boot` on different drives: pick a `-Boot` on the package's drive. Hard links cannot cross volumes, and the sandbox cannot follow symbolic links to host paths.
  * No `MicrosoftGame.config`: `-Package` is not the extracted package root.
  * `WindowsSandbox.exe` missing: the feature is not enabled, or the PC has not rebooted since.
* **If nothing appears in `logs` after a minute,** look at `logs\checks.txt`:
  * the GameInput install takes about 2 minutes, and the title's logs start after it.
  * `NETWORK REACHABLE` / `DNS RESOLVES` / `DEFAULT ROUTE PRESENT` means the guard refused to start the title.
  * if `boot.wsb` was edited by hand, keep `<Networking>Disable</Networking>`.
* **Never start the executable on the host,** not even to test a fix. On Linux, use `sandbox/` (Docker and Wine, no network) instead of this folder.
