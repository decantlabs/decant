# Debugging switches for a title run

Switches that show what a title does while it runs under our layer. Each one is an environment
variable, off by default, and costs nothing when it is off. Set them in the shell that starts
`decant-launch`: the launch tool passes its environment on to the title.

| Switch | What it shows | Where it runs | Output |
|---|---|---|---|
| `DECANT_CALLS=1` | every shim answer that was "not implemented", a stub value or an error | inside the title, in every Decant DLL | `decant_calls.txt` and the run log |
| `DECANT_WATCH=<where>[:len],...` | every write of a title value: old and new value, the thread, its stack | the launch tool (debug registers) | the run log |
| `DECANT_WATCH_READ=<where>[:len],...` | the same for every read and write | the launch tool | the run log |
| `DECANT_STACKS=<seconds>,...` | the stack of every thread at those times, with names and waits | the launch tool | `decant_stacks.txt` |
| `DECANT_WAITS=<seconds>` | threads that sat in the same wait longer than that, with who created the handle | the launch tool, with a line from the DLLs | the run log and `decant_waits.txt` |
| `DECANT_KEYS=<second>:<key>,...` | pad input at fixed times, with or without window focus | inside the title, in `GameInput.dll` | the session log |
| `DECANT_CAPTURE=<dir>` | the presented frames as PNG, `DECANT_CAPTURE_TARGETS=1` also every render target drawn | inside the title, in `d3d12_xs.dll` | PNG files in `<dir>` |
| `DECANT_D3D12X_KIND_CHECK=1` | at every draw, dispatch and indirect execution, each table offset a shader reads whose slot holds another kind of view (a buffer read where a constant buffer sits, a texture where a buffer is wanted, a slot never written or with zero title words) | inside the title, in `d3d12_xs.dll` | the session log, lines `kind check:` |
| `DECANT_D3D12X_ZERO_SLOT_NULL` (on by default, `0` turns it off) | not a diagnostic: a table offset a bound shader declares whose title words are all zero (no tag) gets a null view of the declared kind in the host slot (a constant buffer: a view of the zeroed fallback buffer), once until the slot is tagged | inside the title, in `d3d12_xs.dll` | the session log, lines `zero slot null:` (first 8, then a total every 1000) |
| `DECANT_D3D12X_PRESENT_OVER` (on by default, `0` turns it off) | not a diagnostic: composes the second `PresentX` plane (UI/overlay) over plane 0 with premultiplied alpha blending when `plane_count >= 2` | inside the title, in `d3d12_xs.dll` | `present over: second presentation plane pass ready` and `present: second plane composed over plane 0` |
| `DECANT_D3D12X_BARRIER_MERGE` (on by default, `0` turns it off) | not a diagnostic: drops consecutive per-list global UAV barriers (slot 146 and slot 26 `pResource == NULL`) when no draw, dispatch, indirect call, or UAV clear ran since the previous global UAV barrier | inside the title, in `d3d12_xs.dll` | `uav barrier merge: active` and periodic `uav barrier merge: R recorded / S skipped` |
| `DECANT_D3D12X_INDIRECT_CLAMP` (off by default, `1` turns it on; `DECANT_D3D12X_INDIRECT_CLAMP_SCOPED=1` for scoped pre-barriers) | not a diagnostic: clamps single-action `ExecuteIndirect` arguments on the GPU into per-list scratch with a saturating `2^27` work budget and bounded 256-try CAS loop to prevent GPU hangs from stale/refused compute passes | inside the title, in `d3d12_xs.dll` | `indirect clamp: enabled` and `indirect clamp: active ... first clamped ExecuteIndirect` |
| `DECANT_D3D12X_ALLOC_SWAP` (on by default, `0` turns it off) | not a diagnostic: swaps a busy PC command allocator for a spare or new one when the title calls `Reset` while recordings are still live or in flight on the GPU (preventing debug-layer ID 541 device removal), and logs in-flight list resubmits (ID 553) | inside the title, in `d3d12_xs.dll` | `alloc pool: N allocator resets with recordings still held or on the GPU` and `alloc pool: a command list is submitted while its previous submit is still on the GPU` |
| `DECANT_GENTLE=1` (or `decant-launch run --gentle`) | nothing, it lowers the load of the title on the desktop: CPU priority and GPU scheduling priority below normal | the launch tool | one `DECANT_GENTLE:` line in the run log |

Where the code lives:

* **In the launch tool** (crate `decant-launch`, module `tools`): `decant-launch` runs the title as its
  debugger. It sees every thread from its first instruction and every exception before the title
  does, and it can stop a thread and read it without running code inside the title. Switches that
  need the title's threads live here. `run.rs` calls them through one hook point, `tools::Tools`.
* **In the Decant DLLs**: switches that must act inside the title (count calls, press keys,
  capture frames) live in the DLLs, with the shared parts in `decant-log`.

## DECANT_CALLS: unanswered questions

`DECANT_CALLS=1` counts every call a Decant DLL could not really answer: an export or slot that
returns `E_NOTIMPL` or `ERROR_CALL_NOT_IMPLEMENTED`, a made up value, or an error for a question the
shim cannot serve. Each distinct question is one entry with its call count, the time of its first
call and its first caller outside the shim, as `module+rva`. The title's first unanswered question
is often the reason it waits.

The report goes to `decant_calls.txt` in the title folder (the title's current folder), or to the file
named by `DECANT_CALLS_PATH`. The DLLs write it themselves once a second while new entries appear and
every 30 seconds otherwise. `decant-launch` writes it once more after the process ended and puts its
first lines into the run log:

```
DECANT_CALLS process 8904 at exit: 2 distinct unanswered questions
first call  count     answer           shim and question  <-  first caller
    1.204s       3  not implemented  d3d12_xs.dll!console device slot 55 (derived pipeline)  <-  title.exe+0x123450
    1.380s     812  error            xgameruntime.dll!QueryApiImpl(api {...}, riid {...}) -> 0x80040154  <-  title.exe+0x2340
most called:
        812  xgameruntime.dll!QueryApiImpl(...) (error)
```

Read it top down: the lines are in the order of the first call.

How it works: all DLLs of a process count into one named shared memory block,
`Local\decant-calls-<pid>` (2048 entries of 256 bytes, atomics only). The launch tool creates the
block before the title runs and keeps it open, so it can still read it after the title ended.
Without the launch tool (a plain start, Wine), the DLLs create it themselves and only their own report
is written.

Adding a stub path: call `decant_log::calls::not_implemented("what")`, `stub("what")` or
`error("what")`, or `record_with(kind, || format!(...))` when the text needs formatting (it is only
built when the switch is on). Name the question so that two different questions get two entries,
for example `console device slot 55` or `QueryApiImpl(api ..., riid ...)`. The paths covered now:
the unknown and refused slots of `d3d12_xs` (device, command list, queue, DXGI), missing PC
`d3d12.dll` exports, the unknown slots and the refused `QueryApiImpl` tables of `xgameruntime`,
the unwritten GameInput methods, the open points of `xg_xs`, AcpHal commands, DirectStorage console
slots and codecs, and the unimplemented `xmem` exports. DLLs made by `decant-stub` log through
`OutputDebugString` only.

## DECANT_WATCH and DECANT_WATCH_READ: who changes a value

`DECANT_WATCH` takes a comma separated list of up to four places, each with an optional length:

* `exe+0x1234` or a bare `0x1234`: an offset in the title executable (a bare number below 4 GiB is
  always read as such an offset),
* `d3d12_xs.dll+0x10`: an offset in another module, watched from the moment that module is loaded,
* `0x7ff6a0001000`: a plain address (for example one the log printed),
* `:1`, `:2`, `:4` or `:8` after it: the width of the value. Without it the widest the address
  alignment allows is used, at most 8. An address must be aligned to its width.

Example: `set DECANT_WATCH=0x5a3c120:4,exe+0x5a3c128` before `decant-launch run ...`.

The launch tool puts the addresses into the debug registers of every title thread, also of each thread
created later. The processor stops a thread right after it wrote one of them. The launch tool logs the
old and the new value, the thread (id and name) and its stack, then lets the thread go on. The
title does not notice. A write that leaves the value as it was is counted but not logged.

```
   0.003 DECANT_WATCH: #0 exe+0x6fc00 at 0x7ff60b03fc00, 8 bytes, on writes, now 0x0
   0.019 DECANT_WATCH: #0 exe+0x6fc00 0x0 -> 0x1 by thread 18248 "probe writer": tools_probe.exe+0x883d < tools_probe.exe+0x30c9 < ... < kernel32.dll+0x2dda7 (kernel32.dll!BaseThreadInitThunk+0x17) < ntdll.dll+0xc9c6c (ntdll.dll!RtlUserThreadStart+0x2c)
   ...
   0.019 DECANT_WATCH: #0 exe+0x6fc00: 3 hits, 3 changes, last value 0x3
```

The first frame is the instruction after the one that wrote (a data breakpoint stops after the
access). The other frames are return addresses, walked with the unwind tables of each module
(`.pdata`), so the stack is complete as long as every module on it has them. A frame whose function
is an export also gets its name in brackets.

`DECANT_WATCH_READ` takes the same list and stops on reads as well as writes, every access is logged.
Both share the four debug registers. Each watch logs at most 2000 lines, then only counts, and the
counts are logged at the exit. A value written thousands of times per second slows the title down,
since every stop is a round trip through the debugger.

## DECANT_STACKS: every thread at given times

`DECANT_STACKS=5,30,60` takes a picture of all title threads 5, 30 and 60 seconds after the start
(fractions such as `0.5` work). The launch tool suspends every thread, reads its registers, walks its
stack with the unwind tables, then resumes them all. The pictures go to `decant_stacks.txt` in the
title folder, one block per time, and the run log gets one line per picture.

```
--- at 1.062 s: 2 threads, 2 waiting
thread 23788 "probe waiter": waits in NtWaitForSingleObject < WaitForSingleObjectEx: on 0xbc Event (auto reset, not set)
    ntdll.dll+0x166464 (ntdll.dll!NtWaitForSingleObject+0x14)
    KernelBase.dll+0x1427f (KernelBase.dll!WaitForSingleObjectEx+0xaf)
    tools_probe.exe+0x5189
    ...
    ntdll.dll+0xc9c6c (ntdll.dll!RtlUserThreadStart+0x2c)
thread 23580 "main": waits in NtWaitForSingleObject < WaitForSingleObjectEx: on 0xb8 Thread 23788
```

* The thread name is the one the title set with `SetThreadDescription`, or with the older
  `RaiseException(0x406D1388)` convention.
* A thread whose top frame is a wait system call (`NtWaitForSingleObject`,
  `NtWaitForMultipleObjects`, `NtWaitForAlertByThreadId` for SRW locks, condition variables and
  `WaitOnAddress`, `NtDelayExecution` for a sleep, `NtRemoveIoCompletion`, the thread pool's
  `NtWaitForWorkViaWorkerFactory`, message waits) is shown as waiting, with the wait functions on
  top of its stack.
* For a handle wait, the handles are named with their kind, name and state: an event (manual or
  auto reset, set or not), a semaphore (count of maximum), a mutex (held or free, its owner thread),
  a thread or process (its id). Windows does not keep the handle argument of a thread that sits in
  a system call, so the launch tool looks for it where the wait function keeps it to retry after an
  alert: the non volatile registers and the stack of the system frames. Every waitable handle found
  there is listed. Usually that is exactly the one waited for, sometimes one more.

## DECANT_WAITS: threads that wait too long

`DECANT_WAITS=10` reports every thread that sat in the same wait for more than 10 seconds. The launch tool
looks at all threads about once a second (four times per threshold for a threshold below four
seconds), as `DECANT_STACKS` does. A thread counts as in the same wait while its wait and its whole
stack stay the same, so a thread that waits again and again with a timeout in one loop counts too.
Each such wait is reported once in the run log, with the first title frames below the wait, and
with its whole stack in `decant_waits.txt`. At the exit the waits that were still going on at the last
look are listed again.

```
   1.463 DECANT_WAITS: thread 11696 "probe waiter" has waited 1.0 s in NtWaitForSingleObject < WaitForSingleObjectEx: on 0xbc Event (auto reset, not set), created by tools_probe.exe: probe event, called from KERNEL32.DLL+0x2dda7, from tools_probe.exe+0x51e5 < tools_probe.exe+0x5b72 < ...
   1.462 DECANT_WAITS: thread 3532 "main" has waited 1.0 s in NtWaitForSingleObject < WaitForSingleObjectEx: on 0x74 Thread 11696, from tools_probe.exe+0x23981 < ...
```

The handles are found as described for `DECANT_STACKS`, so a wait can list one handle too many (a
value that sat in a register and happens to be a waitable handle as well).

Who created a handle: a Decant DLL that creates an event a title thread may wait on calls
`decant_log::handles::created(handle, "what")`. With `DECANT_WAITS` set this sends one debug line,
`[decant-handle] 0xbc d3d12_xs.dll: fence cell wake event, called from title.exe+0x...`, and the
launcher puts it next to the handle. The events of the fence cells and the CPU writes of
`d3d12_xs`, the DirectStorage queue completion event and the XApu events report so. Handles the
title creates itself show their kind and state only.

## DECANT_KEYS: pad input on a schedule

`DECANT_KEYS` presses pad buttons at fixed times since the title process started, so a run can get
through a menu with nobody at the controls. The input goes through our `GameInput.dll` straight
into the title's readings, so the window does not need the focus.

```
set DECANT_KEYS=12:A,14:Down,14.5:A,20:Menu*1000,25:L:0:1,27:L:0:0,30:RT:1,31:release
```

| Entry | Meaning |
|---|---|
| `12:A` | press A at 12 s, for 100 ms |
| `20:Menu*1000` | hold Menu for 1000 ms |
| `25:L:0:1` | left stick at x 0, y 1 (up), it stays until the next `L` entry |
| `26:R:-1:0` | right stick fully left |
| `30:LT:1`, `30:RT:0.5` | a trigger, from 0 to 1, it stays until changed |
| `31:release` | every button, stick and trigger back to neutral |

Buttons: `A B X Y`, `Menu` (`Start`), `View` (`Back`), `Up Down Left Right` (the D pad), `LB RB`,
`LS RS` (stick clicks). Times may have fractions. Each input is written to the session log
(`gameinput: DECANT_KEYS at 12.000 s: press A`) with the pad state after it. `DECANT_PAD_SCRIPT` can be
used at the same time: buttons of both are pressed, and a stick or trigger the schedule moved
wins over the script's.

The schedule starts when the title creates its GameInput object, and an entry whose time already
passed then is played at once. A title that reads its input another way (raw keyboard messages,
XInput) does not see these inputs: our `GameInput.dll` has a gamepad only.

## DECANT_CAPTURE: the frames as pictures

`DECANT_CAPTURE=<dir>` writes the frames the title presents as PNG files into `<dir>` (made when
missing). `DECANT_CAPTURE_FRAMES` picks them by number, the first present being frame 1: a list such
as `1,2,10,600`, or `pow2` for 1, 2, 4, 8, 16 and so on, which is the default.

* `frame<n>-presented.png`: at `PresentX`, the resource of plane 0 is copied on the title's queue
  into a readback buffer, the queue is drained, and the picture is written. A title that installed
  its own present hook (`DecantD3D12X_SetPresentHook`) is not captured.
* `DECANT_CAPTURE_TARGETS=1` also writes every render target drawn into during a picked frame, as
  `frame<n>-draw<k>-<w>x<h>.png`. The copy is put into the title's own command list just before
  the list changes state after drawing into its render target 0: before a resource barrier, before
  new render targets are bound, and before the list is closed. At that point the target is still a
  render target, so the copy's barriers are correct. `k` is the number of the frame's last draw into
  that target, counted from the first draw after the previous present. The pictures are written at
  the next present, after its queue is drained. At most 400 pictures and 2 GiB of readback memory
  per frame.

Formats written: 8 bit RGBA and BGRA (UNORM, SRGB, TYPELESS), R10G10B10A2, R16G16B16A16 (float and
UNORM), R32G32B32A32 float, R11G11B10 float, and the one and two channel 8, 16 and 32 bit formats.
Float values are clamped to 0..1 and sRGB encoded, so a picture of a linear HDR target looks as it
would on screen without tone mapping. Other formats, multisampled targets and depth buffers are
skipped and named once in the session log (`capture: format ... is not written`).

A picked frame costs a drained queue and a copy per target, so it stutters. Frames that are not
picked cost one atomic load per draw, barrier, bind and close.

Limits: the targets are taken from command lists recorded between two presents. Lists recorded for
a frame but run after its present, or lists run on another queue that is not finished when the
present's queue is drained, give pictures of an older state.

## DECANT_D3D12X_KIND_CHECK: a shader reads a view of the wrong kind

`DECANT_D3D12X_KIND_CHECK=1` compares, at every draw, dispatch and indirect execution after the table
sync, what each shader of the bound pipeline reads at each offset of its tables with what the host
slot holds. A shader reads a view at the offset its own binding table names (a constant buffer, a
buffer SRV or UAV, a texture SRV or UAV with its dimension), and the title packs each table for that
shader. A slot of another kind is read as a view of the wrong type, which on a GPU is undefined: on
the AMD Linux driver a ring timeout. vkd3d-proton reports it only in its descriptor QA build, which
does not exist on Windows. This gives the same signal on both systems. It needs the mirror
(`DECANT_D3D12X_DIRECT=1` or `DECANT_D3D12X_DXIL_OVERRIDE`), as the tables are found through it, and it turns
the record of the views of `DECANT_D3D12X_PRESENT_PROBE` on (without that probe's draw descriptions).

What a shader declares is kept per pipeline when the pipeline is made:

* a direct pipeline: from the console container of each stage (binding table and `PSV0` resources).
* a DXIL override: from `table_views` in its `<hash>.json`, which `decant_shader_dxil_cache` writes
  (`{"dword":9,"slot":5,"class":"srv","kind":"raw_buffer"}`). An override built before that field is
  not checked, and the log says so once (`has no table_views`). Rebuild it to check it.

One line per mismatch, for the first 4 of each (pipeline, root parameter, offset), and a line with the
total every 1000:

```
kind check: list 0x.. pipeline 0x.. CS 73191475f2669087 (override) param 1 table base host index 183460 console index 183460 offset 5: wants SRV buffer, found constant buffer (cbv); title words [0x7f3a.., 0x0, 0x0, 0x0]; mismatch 1 of this slot
```

It names the list, the PC pipeline pointer, the shader hash and stage, `direct` or `override`, the
root parameter, the host and console index of the table, the offset (slot) in it, the kind wanted
and the kind found, and the title's raw words of the slot (all zero: the title never wrote it, the
table sync leaves the slot as it was). A slot the layer never wrote reads `never written`. A table that
the layer moved is looked at by its host index and shows no title words. Off, the cost is one atomic
load per call.

## DECANT_GENTLE: a title that leaves the desktop usable

`DECANT_GENTLE=1` (or `--gentle` on `decant-launch run` and `all`) sets, for every process the launch tool starts
for the title (relaunches included), the CPU priority class to below normal and the GPU scheduling
priority class to below normal (`D3DKMTSetProcessSchedulingPriorityClass`, gdi32.dll). The kernel
refuses the GPU step with `STATUS_INVALID_PARAMETER` until the process has opened its adapter, so the
launcher tries again twice a second and logs `DECANT_GENTLE: GPU scheduling priority below normal, <n> s
after the start` when it takes. A refused step is logged and never fails the launch. The memory
priority is left alone on purpose, since nothing shows that a low memory priority does not make the
title's own loading thrash. Boot and frame times are slower in this mode, so name it in any report that
quotes times.

## DECANT_D3D12X_ZERO_SLOT_NULL: a zero descriptor reads as nothing

On the console a table slot of zero words reads as nothing, the shader guards its loop with a count from a
constant buffer. The table sync copies tagged slots only, so the host slot of a zero slot keeps an older view
and the shader reads it as the kind it declares. Before the sync of each draw the layer writes, for each offset the bound pipeline
declares (the declared kinds are kept per pipeline whether or not `DECANT_D3D12X_KIND_CHECK` is on), a null view of
that kind into the host slot: a constant buffer a view of the zeroed fallback buffer, a buffer SRV or UAV a
typed one element null view, a texture SRV or UAV a null view of the declared dimension. Once for each slot
until the slot is tagged or copied over. No null view for a typed view whose shape the declaration does not say,
nor for a multisampled or cube UAV. `grep "zero slot null:" <session log>`.
