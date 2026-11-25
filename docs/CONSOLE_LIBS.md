# Console system libraries

This page describes the console system libraries that a title imports and Windows does not provide, and where our implementation of each one lives. It is an overview for developers. The code is the reference for layouts, constant values and result codes, and each section names the source files to read.

Libraries covered: `xg_xs`, `AcpHal`, `xapu`, `chad`, the physical page entry points of `xmem`, and the one export we provide under the name `XCRDAPI`.

## Contents

* [Sources and conventions](#sources-and-conventions)
* [1. `xg_xs`: texture and resource layout](#1-xg_xs-texture-and-resource-layout)
* [2. `AcpHal`: audio control processor and SHAPE](#2-acphal-audio-control-processor-and-shape)
* [3. `xapu`: Opus decoding and sample rate conversion](#3-xapu-opus-decoding-and-sample-rate-conversion)
* [4. `chad`: XDSP convolution and FFT](#4-chad-xdsp-convolution-and-fft)
* [5. `xmem`: physical page entry points](#5-xmem-physical-page-entry-points)
* [6. `XCRDAPI`](#6-xcrdapi)
* [7. Open points](#7-open-points)

## Sources and conventions

* Public D3D12 and DXGI headers: resource descriptions, formats and flags.
* The AMD addrlib GFX10 tables as shipped in Mesa (MIT), in `crates/decant-texture/src/tiling/tables.rs`.
* Our own code in the crates named below, and the public audio summaries in [AUDIO_INPUT.md](AUDIO_INPUT.md).
* Behaviour of the console libraries that is not public was measured, and is recorded in the code and its tests.

The calling convention everywhere is Windows x64 (`extern "system"`). Each library crate exports its functions by name. The import list of the executable is not in the repository: `audit_xbox_imports --shims` writes it outside. Run the audit first and implement at least every name it lists.

## 1. `xg_xs`: texture and resource layout

Implemented in `crates/decant-xg` (the DLL and its ABI wrappers) on `crates/decant-texture/src/xg` (the computation). Open points are kept as `E_NOTIMPL`, see the crate README.

`xg_xs.dll` is the Series X|S version of the XG library. Given a texture description it says which GFX10 swizzle mode to use, how large the resource is, where each mip level and array slice lives, and the byte offset of any element. A title uses it to allocate texture memory, to place data it streams itself and to tile and detile on the CPU. Apart from DCC and BVH it is pure computation on its inputs plus a little global state, so it is deterministic and testable without a GPU.

Exports our crate implements:

* Library state: `XGInitializeLibrary`, `XGTerminateLibrary`, `XGSetHardwareVersion`, `XGEnableExclusiveLocking`, `XGEnterExclusiveLock`, `XGLeaveExclusiveLock`
* Swizzle mode choice: `XGComputeOptimalSwizzleMode`, `XGComputeOptimalDepthStencilSwizzleMode`, `XGValidateDepthStencilSwizzleMode`, `XGSuggestSwizzleMode`
* Layout of a whole resource: `XGComputeBufferLayout`, `XGComputeTexture1DLayout`, `XGComputeTexture2DLayout`, `XGComputeTexture3DLayout`
* Texture computer objects: `XGCreateTexture1DComputer`, `XGCreateTexture2DComputer`, `XGCreateTexture3DComputer`, `XGCreateTextureComputer`, `XGCreateTextureComputerFromDescriptor`, `XGValidateDescriptor`
* Tile shape: `XGComputeTileShape`
* Stand ins: `XGEncodeDCC`, `XGDecodeDCC`, `XGCreateBVHComputer`

Where the details live:

* `crates/decant-xg/src/abi.rs`: the description and layout structures as the DLL reads and writes them.
* `crates/decant-xg/src/exports.rs`, `state.rs`, `computer.rs`: the exports, the library state and the texture computer object with its method table.
* `crates/decant-texture/src/xg/modes.rs` and `formats.rs`: swizzle modes, bind and misc flags, formats, element and block sizes.
* `crates/decant-texture/src/xg/swizzle.rs`: the swizzle mode suggestion.
* `crates/decant-texture/src/xg/layout.rs`, `resource_tiling.rs`, `tile_shape.rs`: layout of linear and tiled surfaces, 1D, 2D, 3D and MSAA, the 64 KiB tiling and the tile shape.
* `crates/decant-texture/src/xg/d3d12.rs`: conversion of a `D3D12_RESOURCE_DESC` to an XG description.
* `crates/decant-texture/src/xg/descriptor.rs`: the GPU image descriptor read by the descriptor functions.
* `crates/decant-texture/src/xg/equations.rs` and `equations_data.rs`: the address equations per swizzle mode, element size and sample count, held as data. The data file is the reference for the equations.

The tests in `crates/decant-xg/tests` and in `decant-texture` cover the exports and the layout computation with many input and output vectors. They are the output of our own layout code and are not compared with a console DLL.

One finding worth keeping: a title was seen to import `XGSuggestSwizzleMode`, which is in no console export list we know. It takes a resource description, flags and a pointer for the answer. We export it, and it gives the same mode `XGCreateTextureComputer` picks for layout 0.

## 2. `AcpHal`: audio control processor and SHAPE

Implemented in `crates/decant-acphal` (the DLL `AcpHal.dll`). Open points are kept as `E_NOTIMPL` or logged stand ins, see the crate README.

`AcpHal.dll` gives a title direct access to the audio hardware: an allocator for memory the audio processor can address with 32 bit APU addresses, the SHAPE context arrays (XMA decoders, sample rate converters, DMA), and an `IAcpHal` object that sends commands to the audio control processor and returns messages. Most titles reach it only through XAudio2, which our XAudio2 layer replaces. A title that imports it directly needs the SHAPE engine. [AUDIO_INPUT.md](AUDIO_INPUT.md) covers the audio layers.

Exports our crate implements: `ApuCreateHeap`, `ApuAlloc`, `ApuFree`, `ApuMapVirtualAddress`, `ApuMapApuAddress`, `ApuIsVirtualAddressValid`, `ApuHeapGetState`, `AcpHalAllocateShapeContexts`, `AcpHalReleaseShapeContexts`, `AcpHalCreate`.

Where the details live:

* `crates/decant-acphal/src/abi.rs`: result codes, constants and structures.
* `crates/decant-acphal/src/exports.rs`: the exports.
* `crates/decant-acphal/src/heap.rs` and `heap/`: the APU heaps.
* `crates/decant-acphal/src/shape.rs`: the SHAPE context arrays and the service thread that counts audio frames.
* `crates/decant-acphal/src/hal.rs`: the `IAcpHal` object, a C++ style object whose first field points at a table of functions.
* `crates/decant-acphal/src/xma.rs`, `monitor.rs`, `priority.rs`: the decoder side and the supporting pieces.

## 3. `xapu`: Opus decoding and sample rate conversion

Implemented in `crates/decant-xapu` (the DLL `xapu.dll`). Open points are kept as `E_NOTIMPL` or logged stand ins, see the crate README.

`xapu.dll` (Series X|S) drives the XAPU: Opus decoding, decoding followed by conversion, and PCM sample rate conversion with pitch. The calling sequence is the public one: connect, activate a stream, send one command per packet and collect its result before the next, deactivate, disconnect. Streams are 48 kHz, mono or stereo, with 16 byte aligned pointers.

Exports our crate implements: `XApuConnect`, `XApuDisconnect`, `XApuEnqueueCommand`, `XApuDequeueResult`, `XApuGetResultCount`, `XApuGetUtilization`.

Where the details live:

* `crates/decant-xapu/src/abi.rs`: constants, result codes and binary layouts.
* `crates/decant-xapu/src/exports.rs`: the exports.
* `crates/decant-xapu/src/engine.rs`: connections, streams and the command queue.
* `crates/decant-xapu/src/converter.rs` and `converter/`: the pitch converting sample rate converter.

## 4. `chad`: XDSP convolution and FFT

Implemented in `crates/decant-chad` (the DLL `chad.dll`). Open points are kept as `E_NOTIMPL` or logged stand ins, see the crate README.

`chad.dll` exposes the XDSP audio DSP: partitioned convolution for reverb, and forward and inverse FFT streams. A client is connected over a block of memory, streams are activated on it, and commands are submitted per block. Our engine does the work synchronously, so a command is complete when it is submitted. Convolution uses uniformly partitioned overlap save. The polar order of the FFT and the console envelope storage are inferred, not confirmed.

Exports our crate implements: `XDspConnect`, `XDspConnectWithMaximumStreamLimit`, `XDspDisconnect`, `XDspActivate`, `XDspSubmitCommand`, `XDspSubmitCommandWithEnvelope`, `XDspDeactivate`.

Where the details live:

* `crates/decant-chad/src/abi.rs`: constants, result codes and structures.
* `crates/decant-chad/src/exports.rs`: the exports.
* `crates/decant-chad/src/engine.rs`: clients, contexts, argument checks and the DSP work.
* `crates/decant-chad/src/fft.rs`: the FFT.

## 5. `xmem`: physical page entry points

Implemented in `crates/decant-xmem`. This section covers the physical page functions. The crate exports the full set a title imports from `xmem.dll`.

Exports for physical pages: `XMemAllocatePhysicalPages`, `XMemFreePhysicalPages`, `XMemMapPhysicalPages`, `XMemMapPhysicalPagesScatter`, `XMemVirtualAlloc`, `XMemFree`.

Concepts:

* Pages travel as an array of page frame numbers, one entry per page, never as a start and a count. A page is 64 KiB. Failure is `FALSE` or null with the reason in `GetLastError`.
* Allocation flags choose 64 KiB pages or large (2 MiB) pages. The Windows part of the allocation type of `XMemVirtualAlloc` is masked to the bits Windows knows, and the console only bits are logged and dropped. The console protection values are mapped to `PAGE_*` values.
* `XMemVirtualAlloc` reserves a placeholder range and allocates nothing. `XMemMapPhysicalPages` and `XMemMapPhysicalPagesScatter` map frames into it, and one page can appear at several addresses. Frame 0 or a null array unmaps a page back to the reserved state, and a mapped page is replaced by a new mapping.
* Physical pages are backed by a pagefile backed section, 1 GiB by default, and `DECANT_XMEM_PHYSICAL_MIB` changes the size. Pages that are unmapped and mapped again keep their old bytes. `XMemFree` of the reservation base ends the whole reservation.
* `XMemTransferMemory` logs its first raw arguments and fails with `ERROR_CALL_NOT_IMPLEMENTED` until its signature is known.

Where the details live: `crates/decant-xmem/src/exports.rs` (the entry points and their argument checks), `pages.rs` and `regions.rs` (the page pool and the reserved regions), `engine.rs`, `track.rs` and `journal.rs`. The tests in `crates/decant-xmem/tests` cover the behaviour.

## 6. `XCRDAPI`

The console `xcrdapi.dll` is a large system DLL for installed content and storage management. Titles reach most of it only through the system. Our `xcrdapi.dll` (the crate `crates/decant-dstorage-xcrdapi`) is built from the same code as `dstorage_xs.dll` and exports one function:

* `DStorageGetFactory`, the DirectStorage factory entry point, described in [DSTORAGE_ABI.md](DSTORAGE_ABI.md).

The export is made at ordinal 1 by `crates/decant-dstorage-xcrdapi/build.rs` and `src/lib.rs`. If the audit shows that a title imports another name from `XCRDAPI.dll`, the title will not load, so check `audit_xbox_imports` first. A name that turns out to be needed should get a logging stand in that fails with `HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED)`, unless the title is seen to depend on it.

## 7. Open points

1. `XGSetHardwareVersion`: which version a title sets decides the equation set. Log it.
2. Depth plus stencil formats in `xg_xs` (two planes) are not specified and return `E_NOTIMPL`.
3. `GetTexelCoordinate` and `GetTextureViewDescriptor` of the texture computer return `E_NOTIMPL`.
4. DCC and BVH functions of `xg_xs` need a separate specification if a title imports them.
5. The SHAPE context register layouts of `AcpHal`.
6. The unit of the `xmem` frame numbers, and the console meaning of one allocation type bit that we drop.
7. The test vectors of `xg_xs` are not yet compared with a console.
