# decant-acphal

`AcpHal.dll`, the audio control processor library of the console, in Rust (specification in `docs/CONSOLE_LIBS.md`, section 2).

It gives a title the APU memory allocator (memory the audio processor addresses with 32 bit APU addresses), the SHAPE context arrays and the `IAcpHal` object that sends commands to the audio control processor and returns messages.

## Exports

All ten are exported by name, with the prototypes of the specification: `ApuCreateHeap`, `ApuAlloc`, `ApuFree`, `ApuMapVirtualAddress`, `ApuMapApuAddress`, `ApuIsVirtualAddressValid`, `ApuHeapGetState`, `AcpHalAllocateShapeContexts`, `AcpHalReleaseShapeContexts` and `AcpHalCreate`. The specification gives no ordinals, so there is no `build.rs` and no `.def` file: rustc exports the `no_mangle` functions by name on the MSVC and the gnu targets. `build.cmd` builds the DLL with the static C runtime like the other shims.

## Layout of the code

* `abi`: result codes, constants, `APU_HEAP_STATE`, `SHAPE_CONTEXTS` (0x60 bytes), `ACP_MESSAGE` (28 bytes), message and command types.
* `heap`: the heaps. Two public heaps plus the internal one, each a zeroed host allocation aligned to 64 KiB with a best fit free list that merges neighbours. `Heaps` is plain Rust, the exports use one process wide instance behind a lock. The heaps appear with the default sizes at the first `ApuAlloc` or shape allocation, and the blocks the library takes for itself make a fresh heap 1 report 3 allocations.
* `shape`: the six context arrays (five allocations in heap 2, XMA and PCM sharing one), their release, the service thread that counts audio frames of 128 samples from the elapsed time and runs the XMA engine every 2 ms, and the engine state the commands change.
* `xma`: the XMA engine. It decodes the enabled XMA contexts into their output rings with the pure Rust decoder `decant-xma`: it moves the write block, clears "output writable" when the ring is full, advances the read position in bits, clears an input's valid bit when it is used up, switches between the two inputs, plays loops (loop start and end in bits, loop count, 255 for ever, the subframe kept at the loop end and the subframes skipped after the jump), skips the start subframes, and sets error status 4 with the error bit when both inputs ran out (and clears them when the title gives new input). Command 5 and 6 enable (and clear the error field), 7 and 8 disable, 0x12 moves the read block and makes the ring writable. The word layout is in the module documentation.
* `monitor`: plays what the XMA engine writes on the default Windows device (WASAPI, shared mode, 48 kHz stereo), each context converted to 48 kHz and mixed. It is off unless `DECANT_AUDIO_RAW_MONITOR=1`. The first read of that variable is logged with its value, and a monitor thread that cannot start is logged with the reason. `cargo run -p decant-acphal --example xma_tone` is a listening check: a synthetic stream through a context, two seconds of two tones.
* `stats`: `DECANT_AUDIO_STATS=1` counts per XMA context the blocks written and taken, the times the ring was full, the times the decoder had nothing to write and the frames the monitor missed, and per device the periods and frames mixed. One line a second goes to the log with a timestamp, and a voice that stops gets a last line. The counters do not change what the engine or the monitor does.
* `hal`: the `IAcpHal` object, a C++ style object with a table of 12 functions. `SubmitCommand` writes the command to the next slot of the command ring in heap 1, runs it at once and frees the slot. Messages go to a bounded ring and are dropped when it is full.
* `exports`: the ten exports.
* Logging goes through `decant-log` (`DECANT_LOG`, `DECANT_LOG_PATH`). Every failing call and every command type that is not implemented is logged.

## Open points

The specification leaves these open or marks them as not specified. They stay as stand ins:

* The register layouts of the SRC and DMA contexts. Those arrays are allocated and zeroed, nothing converts or mixes them. The XMA layout is the one titles write (see `xma`). Command 0xE (update a context from a staging copy) and the overlap buffer address (word 8) are not used: a context decodes from a fresh decoder when it is enabled. Flowgraph operations 2 (run a SRC context) and 0x10 (mix buffer to a DMA context) are counted and do nothing. Operation 1 clears a mix buffer of 128 frames. The operation is read from the low 8 bits of the command word, the specification only says "the low bits".
* Command types in range 1 to 0x1B that the specification does not list (among them 3 and 4) return `E_NOTIMPL` and are logged. The words after the index of command 0x13 are not specified, only the index is kept. Command 0xE records the value and the staging address and copies nothing.
* The index checks of the XMA commands (below 0x200) and the DMA command (below 0x1000) are ours, they give `E_INVALIDARG`.
* The messages "SRC blocked", "DMA blocked" and "error" are never produced, because nothing in the specification says when.
* "Audio frame start" messages come every 128 samples at 48 kHz (2.667 ms) from `Connect` on, once the title registered them with command 2: word 0 is the mask of message kinds, a command 2 without data registers every kind. They are made when the title reads its messages, a late reader gets the last 8 it missed. Only frame start messages follow the mask, the other kinds are always queued.
* Command 2 may come without data (the title sends it so right after `Connect`, id 0xF0F0F0F0) and is acknowledged. The first command of every type is logged with its id, frame and first data words.
* `HAL_E_QUEUE_FULL` is returned when all command slots are in use. Commands run synchronously, so it cannot happen with one caller. `SHAPE_E_BUSY` is defined and not used.
* An alignment above 64 KiB aligns the APU address, the host address then has the alignment of the heap base only.
* The blocks the library takes for itself can be freed with `ApuFree` like any other allocation.

## Tests

`cargo test -p decant-acphal` runs the XMA engine on synthetic streams (ring filling and positions, the end of the input and its recovery, loops), the unit tests of the heap (best fit, merging, alignment, zeroing, mappings, limits), the life cycle of the heap exports in the order a title calls them, the implicit creation, the SHAPE allocation (every limit, the failure cleanup, the service thread), the `IAcpHal` table driven as a C client does (connect and disconnect rules, the limit of 4 connections, every command, the rings, the messages) and the load of the built DLL.
