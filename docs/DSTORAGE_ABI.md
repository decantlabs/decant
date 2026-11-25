# DirectStorage console ABI

On Xbox a title loads its assets through DirectStorage, with console interfaces (`IDStorageFactoryX`,
`IDStorageQueueX` and so on) taken from `DStorageGetFactory`. `dstorage_xs.dll` (the Rust crate `crates/decant-dstorage`) provides
that entry point on PC, over PC DirectStorage when it is installed or a CPU fallback otherwise. The
same implementation is also built as `xcrdapi.dll` (the companion crate `decant-dstorage-xcrdapi`),
since a title may import `DStorageGetFactory` under either name.

A console title calls these objects through raw vtables: it knows each interface by its **IID** and
each method by its **slot**. Neither is in the public documentation. Rather than hard coding a guess,
`dstorage_xs.dll` reads them from a **spec file**, and the queue engine never sees the console
layout. Reordering a method or adding an IID only changes the spec.

## What is known, and from where

* **Interface names, methods and inheritance** come from the public GDK reference on learn.microsoft.com (the factory, file, queue and status array interfaces and `DStorageGetFactory`). The X1 and X2 interfaces extend X.
* **Slot order, slot counts, IIDs and the queue and request structures** are not public. They were measured on retail executables and system DLLs and live in the spec files, which are the reference: see `crates/decant-dstorage/abi/`. The public reference lists methods alphabetically, so it gives no order.
* **PC IIDs and PC order** come from the public PC header `dstorage.h` (Microsoft.Direct3D.DirectStorage 1.3.0) and are in `pc_hypothesis.abi`.
* The console request differs from the PC request in its field order, and the queue description differs in its fields. `DecantConsoleRead` in `crates/decant-dstorage/src/decant.rs` reads the console request.

The current reference page of `DStorageGetFactory` uses `IDStorageFactoryX2`, older titles may ask for `IDStorageFactoryX1` or `IDStorageFactoryX`.

## Shipped specs (`crates/decant-dstorage/abi/`)

* `console_hypothesis.abi`: **template**. The console interface names from the public reference, with the hypothesis order: methods shared with the PC header keep its order, console only methods follow at the end of each table. Every IID is `unknown`, so as shipped it refuses every console IID.
* `pc_hypothesis.abi`: the public PC interfaces, PC IIDs and PC order.
* `other_title.abi`: IIDs and order confirmed at run time on a second retail title. If an executable asks for the same IIDs, this spec can be used as is.
* `retail_title.abi`: read from a retail executable (its hash is in the header), with comments on what was measured and what is still a hypothesis. `decant-launch stage` copies it as `decant_storage_abi.txt` into the overlay when shims are staged and the title folder has no such file.

None is loaded by default: without a spec, every console IID gets `E_NOINTERFACE`, logged with its GUID.

## Choosing a spec

`dstorage_xs.dll` reads its spec once, at the first `DStorageGetFactory` call with an IID other than
the project's own: the path in `DECANT_DSTORAGE_ABI`, otherwise `decant_storage_abi.txt` next to the DLL. A
spec that does not parse is rejected as a whole and logged with its line number.

## Format

```
abi 1                                   # first directive
queue_desc_capacity_offset 8            # byte offset of the UINT16 capacity in the queue description

interface factory IDStorageFactoryX     # role: factory, file, queue or status; then a label
  status template                       # free word, logged (template, lead, verified, …)
  iid {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx} IDStorageFactoryX   # one line per IID this vtable answers
  iid unknown IDStorageFactoryX1        # template: no GUID yet, nothing can reach it
  method 3 factory.create_queue CreateQueue   # slot, engine handler or "?", optional name
  method 6 ? SetDebugFlags              # "?": logged, returns E_NOTIMPL
end
```

Slots 0 to 2 are `IUnknown` and implicit. Engine handlers, each usable only in its role:

| Role | Handlers |
|---|---|
| factory | `factory.create_queue`, `factory.open_file`, `factory.create_status_array`, `factory.set_cpu_affinity`, `factory.set_debug_flags`, `factory.set_staging_buffer_size` |
| queue | `queue.enqueue_request`, `queue.enqueue_status`, `queue.enqueue_signal`, `queue.enqueue_set_event`, `queue.submit` |
| status | `status.is_complete`, `status.get_hresult` |

A handler states that the console method at that slot has the signature of the public PC method of
the same name. The three tuning handlers (`SetCpuAffinity(u64)`, `SetDebugFlags(u32)` without a result,
`SetStagingBufferSize(u32)`) log their value and accept it: this engine has no affinity setting, debug
layer or staging buffer to size. `queue.enqueue_request` reads the console request layout of `DecantConsoleRead`
(`DecantConsoleRead` in `decant.rs`).

## Filling the spec from the executable

```
audit_xbox_imports --dstorage-probe <title.exe> --out <folder>
```

(`crates/decant-audit-imports`.) It reads the PE as data and reports:

* the imports from `dstorage_xs`, `xcrdapi` or `dstorage`.
* each call to `DStorageGetFactory`, direct (`call [rip+disp32]`) or through a linker thunk.
* the GUID loaded into `rcx` (the IID argument) just before each call.
* the indirect calls `call [reg+disp]` that follow, as vtable slot `disp / 8`.

It writes `dstorage_probe.md` and `decant_storage_abi.draft.abi`, a factory interface with the IIDs found
and every slot seen as `?`. It is a byte scan, not a disassembly: slots are only seen when the calls
are inlined after `DStorageGetFactory`, and a stray byte pattern can look like a call. Check each
candidate on the executable, then copy the IIDs into `console_hypothesis.abi` (or correct its order)
and map the slots to handlers.

## Queue behaviour

`Submit` hands the recorded operations to the queue's own worker thread and returns at once.
Completion is reported through status arrays, events and fences. Each queue has its own worker, so
several queues read in parallel, and operations of one queue complete in the order they were
enqueued. Closing a queue lets the operation in flight finish, cancels the rest (status slots get
`ERROR_CANCELLED`, events are not set) and returns only once nothing is pending.

## Tests

Both are test programs of the crate, and both take the DLL to test as their first argument
(`dstoragetest <dstorage_xs.dll>`), so they run against any build of the DLL. `cargo test -p
decant-dstorage` runs them against the DLL cargo built.

* `dstoragetest`: reads, zlib, GDeflate, statuses, events, fences, the BCPack error, and on the CPU
  backend asynchronous submission, parallel queues, status order and closing during a read (a named
  pipe stands in for a slow disk). It makes its compressed data with libdeflate, so it needs the
  GDeflate checkout at build time and prints `SKIP` without it. BCPack requests of every mode are
  covered by the engine's own tests (`engine_tests.rs`), with streams from the `decant-bcpack` test encoder.
* `dstorage_abi_test`: `none`, `spec a`, `spec b` (every method moved, no engine change), `bad`,
  `parse <spec>`, `template <console_hypothesis.abi>` (base IIDs filled with test GUIDs, the factory's
  tuning slots called, then a read
  end to end through the console vtables only), `retail <retail_title.abi>` (the call sequence of that spec, ending in a request and submit) and `refused <spec>`.
* The parser of the spec format has unit tests on small invented specs, with the exact error text
  (`line N: reason`).

## Building, and what stays external

`build.cmd` builds `dstorage_xs.dll`, `xcrdapi.dll`, `dstoragetest.exe` and `dstorage_abi_test.exe` with
`cargo build --release --locked`, with the static C runtime. Nothing is skipped when an external piece is
missing, because none is needed to build:

* **PC DirectStorage** (`dstorage.dll` of the `Microsoft.Direct3D.DirectStorage` package) is loaded at
  run time. `DECANT_DSTORAGE_PC_DLL` names the DLL to use, or `cpu` to force the CPU backend. Without it the
  layer looks for `dstorage.dll` in the application folder and falls back to the CPU backend.
* **GDeflate** decoding on the CPU path uses libdeflate with the NVIDIA additions. Set
  `DECANT_GDEFLATE_ROOT` to the `GDeflate` folder of a checkout of microsoft/DirectStorage (with the
  `3rdparty/libdeflate` submodule) and `build.rs` compiles and links it. Without it, GDeflate requests on
  the CPU path answer `E_NOTIMPL`, and zlib still works (it is decoded by the `flate2` crate).
* **BCPack** (BC1, BC3, BC4, BC5, BC6H, BC7) is decoded in process by the `decant-bcpack` crate, which
  `dstorage_xs.dll` links. BC2 has no packed form and answers `E_NOTIMPL`.
* **Physical page destinations** (option bit 21): the destination field is an array of 64 bit frame
  numbers, as `XMemAllocatePhysicalPages` gives them, one per 64 KiB, and the low 16 bits of the field at
  0x10 are where the bytes start in the first page. The engine reads the frames the destination covers at
  enqueue, stages the bytes (read, then zlib or BCPack decoded), and `xmem.dll` writes them through its
  own view of the pool (`DecantXmemWritePhysicalPages`): a page the title maps only later gets them all the
  same, the pages become dirty for the GPU mirror, and the views that show them now are reported to
  `d3d12_xs.dll`. A retail title reads every chunk of its chunked files this way (its request builder is in the executable): options 0x200000, plus zlib or a BCPack mode from the chunk type.

## Not handled yet

* `xcrdapi.dll` exports only `DStorageGetFactory`, which is all the title we know imports from it. If an executable imports another name from `XCRDAPI.dll`, it will not load: check `audit_xbox_imports` first.
* Request option bits (`console_options` in `decant.rs`): bit 0 zlib, bits 1 to 4 BCPack mode, bits 5 to 20
  swizzle mode, bit 21 destination in physical pages, bit 22 source in memory, bit 23 source in physical
  pages. The engine refuses a swizzle mode, bits 22 and 23, bits from 24 up, BCPack modes above 7 and zlib
  together with BCPack, each named in the log and the calls report (`E_NOTIMPL` at enqueue). What the console does with
  a swizzle mode is not established, and no run has shown one yet.
* `RetrieveErrorRecord`, `Query` and the other slots the specs mark `?`: logged, `E_NOTIMPL`.
* On console DirectStorage decompresses: a zlib compressed container header that reached a retail title raw
  crashed it. The engine decompresses zlib (bit 0) itself.
