# D3D12.X layer (`d3d12_xs.dll`)

The layer turns the Scarlett D3D12.X entry points into calls to a PC Direct3D 12. The PC runtime is whatever `d3d12.dll` resolves to: vkd3d-proton (in a Proton or Wine prefix, or app local on Windows) or the one of Windows. The source is the crate `crates/decant-d3d12x`. `exports.rs` has the entry points, `proxy.rs` and the `proxy/` folder the console ABI proxies, `abi.rs` the method order table, `frame.rs` the frame events, `present.rs` the built in present and `rts0.rs` the DXBC and root signature container helpers. The test program is `d3d12xs_test` (`src/bin/d3d12xs_test.rs`), run by CI after `build.cmd` and by `cargo test -p decant-d3d12x`.

Related: [D3D12X_DIFFERENCES.md](D3D12X_DIFFERENCES.md) (published differences), [LAUNCH_ARCHITECTURE.md](LAUNCH_ARCHITECTURE.md) (where the layer sits), [DEBUG_TOOLS.md](DEBUG_TOOLS.md) (the switches that show what a title does).

Only public material is used: public D3D12 header layouts, the public documentation of the GDK, and what console DLLs of a retail package show when they are disassembled. Slot numbers, offsets, flag values and GUIDs below are facts measured that way.

## State of each entry point

| Entry point | State | What it does |
|---|---|---|
| `D3D12SerializeRootSignature`, `D3D12SerializeVersionedRootSignature` | **done** | Clear the root signature flag bits the PC runtime does not define (anything above `0xFFF`), log each distinct value once, then call the PC serializer. Without this the PC serializer refuses the console flags with `E_INVALIDARG` |
| `D3D12CreateRootSignatureDeserializer`, `D3D12CreateVersionedRootSignatureDeserializer` | **done** | When the RTS0 part of the container carries such bits: clean a copy, rewrite the DXBC checksum, hand the copy to the PC deserializer. Otherwise the call goes through untouched |
| `D3D12GetDebugInterface` | **done** (forwarder) | Forwarded to the PC `d3d12.dll` |
| `D3D12XboxCreateDevice`, `D3D12CreateDevice` | **partial** | Both create a PC device (feature level 12_0 minimum) and return it inside the **console device proxy**, whatever IID was asked, because the caller is console code. A retail title imports both from `d3d12_xs`. `DECANT_D3D12X_PROXY=0` returns the bare PC device instead, for comparisons. The adapter argument is ignored and logged, and the parameters structure of `D3D12XboxCreateDevice` is logged by address only. The PC adapter is the high performance one that DXGI lists first, `DECANT_D3D12X_ADAPTER=<index or name part>` overrides it (`adapter_choice.rs`). `D3D12CreateDevice` with a null output only runs the PC capability check |
| `IID_ID3D12Device` | **done** (data export) | The public PC GUID, exported as data. The console DLL exports the PC IIDs this way and titles import them, and a missing data import stops an executable from loading at all. The other console IIDs are listed in `crates/decant-d3d12x/src/abi.rs`, and only this one is exported until the import table of an executable asks for more |
| Console ABI proxies: device, command list, queue, descriptor heap | **partial** | `proxy.rs`, order as data in `abi.rs` (see below). Relayed slots swap `this` for the PC object and jump to the PC method, slots whose signature or return convention differs are Rust methods, unknown slots log their number and return `E_NOTIMPL`. Lists come from device slots 12 and 73, queues from 8 and 48, heaps from 14, all proxied. Proxied objects are unwrapped before they reach the PC (`ExecuteCommandLists`, `ExecuteBundle`, `SetDescriptorHeaps`, `MakeResident`, `Evict`) |
| `QueryInterface(IDXGIDevice)` on the device | **done** | The console merges the DXGI device into the D3D12 device, and a retail title asks for it right after creation. Relayed to the PC device when it implements it, otherwise answered by a DXGI device object (`dxgi.rs`): private data relayed to the PC device, `GetAdapter` and `GetParent` give the PC adapter with the LUID of the device and a headless output, `SetGPUThreadPriority` kept without effect, `CreateSurface` and `QueryResourceResidency` logged and `E_NOTIMPL` |
| Frame events: device 78 `SetFrameIntervalX`, 79 `ScheduleFrameEventX`, 80 `WaitFrameEventX` | **emulated** | A frame clock (`frame.rs`). 78 sets the period (the interval in microseconds, the periodicity is ignored) and restarts the clock, 79 sets an offset per event type and for type 0 answers `STARTED_CLOCK` (`0x10000000`) once, then `S_FALSE` while already scheduled. 80 **really waits** for the next deadline of that type, and for type 0 yields a token from a tracked pool of recently issued tokens that grows by one per frame. The period is 60 Hz until 78 is called. A wait that times out returns `S_FALSE` for every event type. For type 0, flags bit `0x2` still issues a new token and advances the frame on that timeout, and a passed in token must be one this layer issued and must not come with `0x2` (`E_INVALIDARG`). Without these waits the game thread of a console engine waits for a frame start that never comes, and the watchdog of the engine ended a retail title after about 90 s. The same title stopped at device slot 80, type 0, timeout 0, flags 2 and did not accept `DXGI_ERROR_WAIT_TIMEOUT` there. |
| Placed textures (device slots 44 and 133) | **detiled** | The committed PC texture is filled from title memory by a compute pass (`detile.rs`, `shaders/detile.hlsl`, layout from `decant_texture::xg` in `detile_plan.rs`). The pass runs before the lists of each `ExecuteCommandLists` (direct queues) for textures made or possibly rewritten since: creation, `Unmap` of a placed mapping and DirectStorage completions (`DecantD3D12XNoteWrite`). It covers every 2D one sample texture the GPU does not write and the XG layout models: all swizzle modes with an equation, BC1 to BC7, RGBA8 and the other 1, 2, 4, 8 and 16 byte formats, mips, tail and array slices. Not caught: plain CPU writes after the first detile, GPU copies into the memory, 3D and thick modes, MSAA. Switch `DECANT_D3D12X_DETILE=0` |
| `PresentX` (console queue slot 30) | **built in present, or hook** | `PresentX(plane count, planes, parameters)`. With a function installed through the export `DecantD3D12X_SetPresentHook(hook)` the call reaches it with the PC queue and the three arguments untouched. Without a hook the built in present (`present.rs`) copies the first resource of plane 0 into a DXGI swap chain on the top level window of the title, on the title's queue, and presents it without waiting for vblank, because the frame clock paces the title. The frame token (the first 8 bytes of plane 0, 0 is valid) is read and handed to the frame clock. The call then returns `S_OK`, since a console present does not fail for the title. `DECANT_D3D12X_PRESENT=0` turns the built in present off, and the call is then only accepted. `DECANT_CAPTURE` writes the presented frames as pictures |
| `D3D12XboxSetProcessDebugFlags` | **stub** | Logged, returns `S_OK`, no effect (the console DLL also answers `S_OK`) |
| `D3D12ConfigureVirtualMemoryX` | **done** | One argument, a 32 byte description whose layout is in `crates/decant-d3d12x/src/abi.rs`. `E_INVALIDARG` for a flag other than bit 0 or for more than 0x21 pages, otherwise both outputs are 0 and the result is `0x10000000` |
| `DXGIXPresentArray`, `DXGIXSetVLineNotification`, `DXGIXSetFrameNotification`, `DXGIXGetFrameStatistics` | **as on console** | Logged, `DXGI_ERROR_UNSUPPORTED` (`0x887A0004`). The console DLL answers that from all four without reading its arguments (seen on its disassembly), and the arguments are not read here either |
| Resources, fences, allocators | **PC objects** | Returned as the PC device creates them, not proxied. Their console order is not measured. Fence slot 12 returns the address of the completed value, which a retail title polls with `MONITORX` and `MWAITX`: each fence asked for it gets a memory cell that a watcher thread keeps equal to the PC value (`fence_cell.rs`) |
| Pipelines (device slots 10, 11, 60, 62, 86) | **made on first bind** | Creation answers at once with an object of ours that holds a copy of the description (`lazy_pso.rs`). A slot 86 stream with a mesh shader becomes our mesh description (the graphics one plus the amplification and the mesh shader, `mesh_pso.rs`), made at its first bind through `ID3D12Device2::CreatePipelineState` (`proxy/mesh_pipeline.rs`). A pipeline the PC runtime refuses is refused by name at that bind and counted in `DECANT_CALLS`. The mesh shader tier of the PC device is logged at device creation, and mesh dispatches (list 129 and 100 in the newer layout, 127 in the older) go to `ID3D12GraphicsCommandList6::DispatchMesh`. The PC pipeline, with its DXIL overrides, is made when a list first binds it (`SetPipelineState`, `Reset`, `CreateCommandList`). One the PC runtime refuses leaves out the work of its list, as a stand in does. A retail title creates more than 50,000 pipelines while it boots and its boot flow waited minutes for the driver to compile them all, which is why creation is lazy. Switch `DECANT_D3D12X_LAZY_PSO=0` |
| Shaders of a pipeline | **direct path, off by default** | With `DECANT_D3D12X_DIRECT=1` the console shaders of a pipeline go to `decant-shader` (`direct_source.rs`), the HLSL to DXC (`decant-dxc`) on worker threads (`compile_service.rs`), and the DXIL to a disk cache (`direct_cache.rs`, folder `DECANT_D3D12X_DIRECT_CACHE`, default under `%LOCALAPPDATA%\Decant\direct-dxil`). A pipeline the translator refuses is made from the title's own shaders, which take the DXIL override files of `DECANT_D3D12X_DXIL_OVERRIDE=<dir>` (`dxil_override.rs`, named by content hash). `DECANT_D3D12X_FRAME_WAIT_MS` bounds the time one frame waits for pipelines that are still compiling. `DECANT_D3D12X_PIPELINE_MANIFEST=<file>` records one line per pipeline description, and `decant_shader_warm` fills the cache ahead of time from it |
| Root views (list slots 41 to 46) | **title memory, upload memory, placed buffers** | One function (`proxy/root_views.rs`) finds the PC address of a title address: upload memory of list slot 148, the window buffer of title memory (`title_mem.rs`) or a buffer placed at the address. For a root SRV or UAV every page up to the end of the reservation of the title is given to the GPU, since the shader reads at offsets it takes from the data (a retail title reads root SRVs `t0` and `t4` of space 3 so), and `title_mem::sync` makes the bytes current at each submission. An address no PC resource holds gets the zeroed fallback buffer (`fallback.rs`), is counted in `DECANT_CALLS`, and its 64 KiB page is logged with what Windows knows of its memory. Root views are counted by kind and source, with a summary every 4096. The root signature rewrites keep the root views of the title at their index, register and space (test in `rts0.rs`). Title memory is a switch, `DECANT_D3D12X_TITLE_MEM=1` |
| Pipeline cache loads (device slots 57 and 67) | **a miss, as the title expects** | `DeserializeGraphicsPipelineStateX` and its compute twin `DeserializeComputePipelineStateX` take `(cache entry, riid, out)`. The entries are the serialized pipelines of the console driver, so each load answers `D3D12_ERROR_DRIVER_VERSION_MISMATCH` with a cleared result (`proxy/pso_cache.rs`). A retail title tries 25,450 graphics and 2,689 compute entries while it boots. On a failure its graphics loop goes on with the next entry, and its compute lookup creates the pipeline through slot 62. Only the device lost codes are special to it. Not counted in `DECANT_CALLS` |
| Page mappings (list queue copies) | **done** | A title registers a pool of 64 KiB pages and tells its queue which pool pages its GPU shall see at which addresses (`page_maps.rs`). Here the memory of the title and the view of the GPU are one thing, so the layer asks `xmem.dll` to show the pages at the destination (`DecantXmemShowPages`) |

Every call that is not fully handled writes a line starting with `d3d12_xs:` to `OutputDebugString` (visible with `WINEDEBUG=+debugstr`, or in `decant_launch.log` on Windows) and to the Decant session log (`DECANT_LOG_PATH`).

## Switches

All are environment variables read once. The diagnostic ones are in [DEBUG_TOOLS.md](DEBUG_TOOLS.md), the header of each module lists its own.

| Switch | Effect |
|---|---|
| `DECANT_D3D12X_PROXY=0` | Hand out the bare PC device |
| `DECANT_D3D12X_ADAPTER=<n or text>` | Choose the PC adapter |
| `DECANT_D3D12X_DETILE=0` | Do not detile placed textures |
| `DECANT_D3D12X_LAZY_PSO=0` | Create pipelines at creation instead of first bind |
| `DECANT_D3D12X_PRESENT=0` | No built in present |
| `DECANT_D3D12X_DIRECT=1` | The direct shader path |
| `DECANT_D3D12X_TITLE_MEM=1` | Title memory seen by the GPU (`title_mem.rs`) |
| `DECANT_D3D12X_DXIL_OVERRIDE=<dir>` | Replacement DXIL by content hash |
| `DECANT_D3D12X_DEBUG=1`, `DECANT_D3D12X_DRED=0` | The D3D12 debug layer, and DRED off (`diagnose.rs`, on by default) |
| `DECANT_D3D12X_KIND_CHECK=1`, `DECANT_D3D12X_ZERO_SLOT_NULL=0` | See DEBUG_TOOLS.md |

## What was checked

* **DXBC checksum.** `rts0::dxbc_checksum` gives the same 16 bytes as the PC runtime writes (test), and as the 52 containers of the console module `XboxBvhBuilder.dll` of a package carry (checked locally, no package file is in this repository).
* **The console flags.** The 52 DXIL containers of that module all carry an RTS0 version 2 root signature with flags `0x03000000`. Checked locally on the D3D12 of Windows:
  * the plain PC **deserializer accepts** all 52 as they are (with a correct checksum, the unknown flag bits are not refused).
  * deserializing each one and **serializing the result again fails 52 out of 52** on the plain PC serializer (`E_INVALIDARG`), and **succeeds 52 out of 52** through `d3d12_xs`.

  So the serializer is the one that needed the change. The deserializer path also cleans the flags, so the description handed back to the caller only has public bits, and the same blob is fine for vkd3d-proton (not yet run under Proton).
* **Builds.** `build.cmd` (cargo, MSVC target) and `src/gdk/Makefile` (cargo, `x86_64-pc-windows-gnu`) produce the same exports.

## Assumptions to validate

1. The `0x03000000` root signature flags have no effect the PC can reproduce, so dropping them is safe. Their console meaning is unknown.
2. The PC device at feature level 12_0 or above on the chosen adapter is the right base. The console GPU is 12_2 class, and a weaker PC GPU fails the call instead of getting a lesser device.
3. The deserializers parse the source during creation (true on Windows: the test frees the cleaned copy before reading the description back). To check on vkd3d-proton.
4. 52 containers against the 53 compute pipelines one public source mentions: not explained yet (a variant created twice?).

## Console method order (measured)

`abi.rs` holds the order as data. It was measured on the console compiled modules of a package with `d3d12x_vtprobe` (crate `decant-audit-imports`). From the typed C++ exports of `XboxBvhBuilder.dll` (`ID3D12Device11*`, `ID3D12GraphicsCommandList4/6/8*`), a forward data flow follows the object pointers to each `call [vtable+N]`. On a console upscaler module (C exports), a call is kept when an argument is a known interface ID, which tells the method family. Each slot below was also checked by hand on the disassembly (argument count and kinds).

| Object | Console slot | PC slot / method | Status |
|---|---|---|---|
| device | 14 | 14 `CreateDescriptorHeap` | **measured** (`riid` in r8) |
| device | 16 | 16 `CreateRootSignature` | **measured** (two DLLs, exact PC arguments) |
| device | 27, 28, 29 | 27 `CreateCommittedResource`, 28 `CreateHeap`, 29 `CreatePlacedResource` | **measured** (`riid` position) |
| device | 41 | 41 `CreateCommandSignature` | **measured** (`riid` in r9) |
| device | 11 | 11 `CreateComputePipelineState` | never called by the package |
| device | 62 | none (PC 62 is `CreateStateObject`) | **measured**: `(desc, 0, 0, riid=PipelineState, out)`, two DLLs. Handled as `CreateComputePipelineState` |
| device | 67 | none (PC 67 is `CreateProtectedResourceSession1`) | **measured**: `(desc, riid=PipelineState, out)`, a console pipeline description, callers fall back to 62 |
| device | 76 | 27 shape (PC 76 is `CreateCommittedResource3`) | **measured**: `riid` is the 7th argument |
| device | 73, 78, 79, 80 | none | not called by the package |
| list | 14, 25, 26 | same | **measured** (`Dispatch(1,1,1)`, pipeline, UAV barrier) |
| list | 33 / 37 / 39 / 43 / 45 | 29 / 33 / 35 / 39 / 41 | **measured** by argument shape: root signature, 32 bit constant, 32 bit constants, root SRV, root UAV |
| list | 53, 61 | 49 `ClearUnorderedAccessViewUint`, 59 `ExecuteIndirect` | not called on a typed list here |
| list | 80 | none | **measured**: 5 arguments `(n, 1, 1, 0x8020)`, identity open |
| list | 85, 141 | none | not observed on a typed list |
| list | 146, 148 | none | **measured**, console only |
| list fields | `+0x18` cursor, `+0x28` and `+0x30` limits | none | **measured** (`mov rax,[rbx+18h]; cmp rax,[rbx+30h]` after slot 39) |
| queue | 30 | none | not called by the package |

On slot 11 against 62: the two callers of the package create their compute pipelines through 62 (with two extra parameters), never through 11. Both statements can hold, the executable of a title shows which ones it uses.

Measure again with `d3d12x_vtprobe <dll>... --pc-header <path to the Windows SDK um/d3d12.h> --out <folder>`. It writes `vtprobe.md` and `vtprobe.json`, and `--dump <dll> <rva>` prints the instructions before a call site. Run it on the executable of a title when it is available: it is the witness for every slot marked as not called.

## Measured on a retail title

Facts measured on the calls of a retail title.

| Object | Console slot | Measured |
|---|---|---|
| export | `D3D12ConfigureVirtualMemoryX` | One argument, a 32 byte description (layout in `abi.rs`). Implemented in `exports.rs` |
| device | 133 | `CreatePlacedResourceX(this, address, const D3D12_RESOURCE_DESC*, 0xFFFFFFFF, NULL, riid, out)`: a resource placed at a GPU address, called after the descriptor heaps are set up |
| device | 127 | Creates the display targets, called with `ALLOW_DISPLAY`: two 3840x2160 targets, each 32 MiB taken in 4 MiB pages from the allocation hook of the title with display target attributes |
| device | 108 | `GetResourceAllocationInfo` in the newer layout, with 25 in the older (`alloc_info.rs`) |
| device | 70, 71 | Views at a GPU address |
| device | 61 | `CreateSamplerX` |
| device | 25, 26 | Struct return: the result pointer in `rcx`, `this` in `rdx` |
| fence | 12 | The address of the completed value cell of the fence |
| list | 72 to 96 | The command lists of the title use the newer layout: each slot from 72 to 96 is two lower (72 becomes 70, 96 becomes 94) |
| command signature | indirect argument type 105 | `DispatchX`, forwarded as `Dispatch` in `CreateCommandSignature` |
| descriptor heaps | none | The memory of the shader visible CBV, SRV and UAV heap (1,245,184 descriptors) comes from the exported `__XMemAllocationHooks__` of the executable (object type 4, size = count × increment), and `GetCPUDescriptorHandleForHeapStart` returns that address |

The index of which slot is called with which arguments comes from disassembly of the executable. New slot shapes go into this table as they are measured.

## Order table changes measured on other titles

Facts (slot numbers, argument shapes, GUIDs, sizes) measured on another retail title. The executable of the title at hand stays the witness, and `d3d12x_vtprobe` reads it.

| Object | Console slots | Behaviour in `abi.rs` |
|---|---|---|
| device | 44 to 61 | Console methods: 44 resource placed at an address, 48 queue creation (proxied, the first 16 bytes of the description equal the PC description), 55 derived pipeline (refused, logged), 60 graphics pipeline (PC `CreateGraphicsPipelineState`, extensions logged and ignored), 61 extended sampler, the others unknown with their hints |
| device | 14, 25, 26, 34, 35 | 14 wraps the heap, 25 and 26 take the result pointer in `rcx` and `this` in `rdx`, 34 and 35 unwrap proxied objects |
| device | 73 | Command list creation from `{type at 0, node mask at 8}` |
| device | 78, 79, 80 | Frame events, emulated |
| list | 27 | `ExecuteBundle` |
| list | 28 to 31 | `BeginQuery`, `EndQuery`, `ResolveQueryData`, `SetPredication` (PC 52 to 55) |
| list | 53 to 61 | 53 to 55 are PC + 4, **56 to 59 are in PC order without shift (`ExecuteIndirect` = 59)**, 60 to 63 unknown (memory copy and fills) |
| list | 67, 68, 83 | 67 texture cache flush (nothing to do on PC), 68 `OMSetDepthBounds` (PC 62), 83 `CopyTextureRegion` with an 8th argument |
| queue | 23 | `Signal(fence, value, flags)` goes to the PC `Signal`, flags logged |
| queue | 30 | `PresentX` with three arguments |
| heap | 9, 10 | Proxied heap, the start handles come back in `rax` |

### Still to do

* **Return conventions elsewhere.** The console returns 8 byte structures in `rax` and passes the pointer of a larger result first (`this` in `rdx`). Only device 25, 26 and heap 9, 10 are measured. Device 43 `GetAdapterLuid`, heap 8 and queue 18 `GetDesc` (and `GetDesc` on resources, which are not proxied) return structures by value too and are still relayed in the PC convention. To check on the executable.
* **`GetDevice` on a list, queue or heap** returns the PC device, whose vtable is not the console one. Not observed as a problem yet.
* **`CheckFeatureSupport`**: some console structures are 64 bytes against 60 on PC, relayed as is.
* **`GetResourceAllocationInfo`**: the PC runtime crashed on some console texture descriptions in a retail title, so the console device answers with the size and alignment of the console layout, as `xg_xs.dll` computes them (`alloc_info.rs`).
* **Command lists are created open** (slots 12 and 73), as the PC does. A retail engine expects a closed list and calls `Reset` first, which crashed the NVIDIA driver. Closing the list at creation is to be decided on the executable.
* **Unified memory**: device 44 (placement at an address) and resources on physical pages the PC GPU cannot see removed the device in a retail title. The memory work of the layer (`title_mem.rs`) addresses this.
* **Other data exports**: the console DLL exports more IIDs than the PC list (see `crates/decant-d3d12x/src/abi.rs`), and functions such as `D3D12ConfigureVirtualMemoryX`. Add what `audit_xbox_imports` shows the executable imports.
* **Queue 16** `GetTimestampFrequency`: the console answers a constant 99,750,000 Hz, and the PC frequency is kept on purpose, since the timestamps come from the PC GPU.
* **Device 81** (frame statistics) and the other console slots: unknown, logged.

## Presentation sequence (public)

From the public Xbox GDK samples of Microsoft (`Samples/IntroGraphics/SimpleDeviceAndSwapChain` and `Samples/Graphics/FramePacing`, MIT): a console title has no DXGI swap chain. It sets the frame interval on the device (`SetFrameIntervalX`), schedules the frame origin event (`ScheduleFrameEventX`), and each frame waits for it (`WaitFrameEventX`, which yields a `D3D12XBOX_FRAME_PIPELINE_TOKEN`), records and submits, then calls `PresentX` on the command queue with the back buffer planes and that token. On PC that becomes a swap chain owned by the layer, the emulated frame clock behind `WaitFrameEventX` (device 78 to 80), and in the present a copy of the back buffer of the plane into the swap chain image followed by `Present`. The plane and parameter structures are not public, so the hook receives them as pointers.
