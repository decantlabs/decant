# decant-gameinput

`GameInput.dll`, a shim that gives a title one always connected gamepad which a script or a tool drives, and `decant-pad`, the small tool that writes to the script. It lets a title that reads its pad through GameInput run with no controller, and be played by a test.

The title imports `GameInputCreate` by ordinal 1 of `GameInput.dll` (the real one exports `GameInputCreate` as ordinal 1, `DllCanUnloadNow` and `DllGetClassObject`). This DLL exports the same three names and the same ordinal for `GameInputCreate`. A static import is looked up in the title folder before System32, so staging the DLL there is enough (see below for what is not verified).

## What the DLL answers

The first GameInput interface set, in the vtable order of the public `GameInput.h`: `IGameInput` (IID `11be2a7e-4254-445a-9c09-ffc40f006918`), `IGameInputDevice` and `IGameInputReading`.

* One device, a connected Xbox style gamepad, always present. `RegisterDeviceCallback` calls the callback before it returns for a blocking enumeration and from another thread for an asynchronous one.
* `GetCurrentReading` for the gamepad, controller or UI navigation kinds returns a reading. `GetGamepadState` and `GetUiNavigationState` give the simulated state. Timestamps are microseconds and only go up. Other kinds give `GAMEINPUT_E_READING_NOT_FOUND`.
* Reading callbacks are called from the polling thread when the state changes. `UnregisterCallback` waits for a running call, and a callback may unregister itself.
* The counts of axes, buttons, switches, keys and touches are 0. The states of other device kinds answer false. Methods nobody wrote answer `E_NOTIMPL`. Each is logged once, with its name, so the log shows what a title really calls.
* A `QueryInterface` for an interface it does not know is refused with `E_NOINTERFACE` and logged once with the IID. That is how to see which interface version a title asks for.
* Rumble is accepted and goes nowhere.

Logging goes through `decant-log` (`DECANT_LOG`, `DECANT_LOG_PATH`). In steady state only changes of the pad are logged.

## The control file

`DECANT_PAD_SCRIPT=<path>` names a text file. The DLL looks at it every 16 ms and plays new complete lines in order, so the file works as a script that exists before the start and as a live channel that a tool appends to at any time. If the file becomes shorter or disappears, the pad goes back to neutral and the file is played again from its first line (`decant-pad clear` does this). Without the variable the pad stays neutral and no thread runs.

One command per line. Case does not matter. `#` starts a comment.

```text
A down                 press A and keep it pressed
A up                   release A
press A [ms]           A down, wait, A up (100 ms when no time is given)
hold Start 1000        the same with the time required
wait 500               hold back the following lines for 500 ms
stick left 0.0 1.0     left stick x y, from -1 to 1, y positive is up
stick right -1 0
trigger right 0.5      from 0 to 1
release                every button, stick and trigger back to neutral
```

Buttons: `A B X Y`, `Up Down Left Right` (D pad, also `DpadUp` and so on), `Menu` (`Start`), `View` (`Back`), `LB`, `RB`, `LS` (`L3`), `RS` (`R3`). In `Up down` the first word is the button and the second the action.

Commands run in order. `wait`, and the wait inside `press` and `hold`, holds back every later line, so two `press A` lines are two separate presses one after the other. A wait counts from when it was due, not from when the file was polled, so a script does not drift. A line that does not parse is logged (the first 20) and skipped.

## Timed inputs

`DECANT_KEYS=<second>:<key>,...` presses buttons, moves sticks and triggers at fixed times since the process started, for
example `DECANT_KEYS=12:A,20:Menu*1000,25:L:0:1`. It works alone or together with the control file. The format is in
`docs/DEBUG_TOOLS.md` and in the `keys` module.

## decant-pad

```text
decant-pad [--file PATH] press A [ms]
decant-pad [--file PATH] hold Start 1000
decant-pad [--file PATH] stick left x y
decant-pad [--file PATH] trigger right v
decant-pad [--file PATH] A down | A up | wait 500 | release
decant-pad [--file PATH] clear
```

The file is `--file` or `DECANT_PAD_SCRIPT`. Every command is checked with the same parser before it is appended (exit code 2 and a message otherwise).

## Layout of the code

* `pad`: the state, the language, `Machine` (the timeline) and `Feed` (reads appended lines). No Windows code.
* `runtime`: the live state, the strictly increasing clock and the polling thread.
* `com`: the three objects, their vtables and the exports. `build.rs` pins `GameInputCreate` to ordinal 1.

## Open points

* It is checked by the tests of this crate and not yet against a title. The IID a title queries is unknown until a run logs it. Only the first interface set is answered. A title built for a later set would get `E_NOINTERFACE` and a log line with its IID.
* The vtable orders and the `GameInputDeviceInfo` offsets follow the public header from memory and are checked only by the tests of this crate, not by a title.
* Real controllers are not read. The pad is the script only.
