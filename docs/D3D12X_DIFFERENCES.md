# D3D12.X and PC Direct3D 12: published differences

This catalogue lists the differences between Direct3D 12 on Xbox (D3D12.X, headers `d3d12_xs.h` for Xbox Series X|S and `d3d12_x.h` for Xbox One) and Direct3D 12 on PC that public material documents. Every entry names its source and the version of that source.

The catalogue is a map of what is publicly known, so that work on `d3d12_xs.dll` can tell documented behaviour apart from guesses.

## Method

* **Public material only.** Microsoft's own MIT licensed code on GitHub, public pages on Microsoft Learn, Microsoft blogs, Xbox Wire, published conference slides and talks, and interviews in which Microsoft engineers are quoted.
* **Excluded on purpose.**
  * Pages that Microsoft Learn marks as requiring the Xbox developer agreement. Most of the console graphics section is in this group, including the page on differences from PC.
  * Pages that `docs.xbox.com` serves although Learn still marks them as private. Their status conflicts, so none is cited here.
  * Microsoft's public index of NDA page titles. A title is not evidence of behaviour.
  * Leaked SDKs or headers, or anything copied from them.
  * Forum reposts.
  * No console files and no output of this project's tools were used.
* **Versions.** Code is cited by commit SHA and line range, with the GDK edition the sample states. Learn pages are cited with their `ms.date`. Blogs, slides and talks are cited with their publication date. Web pages were read on 20 November 2026.
* **Kind.** *Stated* means the source says it in words (text, readme or code comment). *Code* means public code shows it by calling the API that way. *Implied* means it follows from the source but is not said.

Microsoft's console samples show console only identifiers in public code. That is how most entries below are sourced. The samples show how an identifier is used, not its full contract.

## Sources

| Key | Source | Version |
|---|---|---|
| GS | [microsoft/Xbox-GDK-Samples](https://github.com/microsoft/Xbox-GDK-Samples), MIT | commit `b1e77b6ac10f95b07296025fda171ca2ddc42dbe` (6 November 2026). Each sample states its minimum GDK edition |
| TK | [microsoft/DirectXTK12](https://github.com/microsoft/DirectXTK12), MIT | commit `f3555e3014a53c81a6202b786dc4d6509b95a936` (7 November 2026) |
| TX | [microsoft/DirectXTex](https://github.com/microsoft/DirectXTex), MIT | commit `6e88c1c52e977cfd9c30671240f7995cba6d4968` (16 November 2026) |
| AS | [microsoft/Xbox-ATG-Samples](https://github.com/microsoft/Xbox-ATG-Samples), MIT, archived, Xbox One XDK era | commit `ff629a6e4876e38fe8761ead116a0e53f634ca67` (13 November 2023) |
| L1 | [GDK introduction](https://learn.microsoft.com/en-us/gaming/gdk/docs/gdk-dev/intro/introduction) | `ms.date` 2022-08-01, updated 2026-07-07 |
| L2 | [Preprocessor definitions](https://learn.microsoft.com/en-us/gaming/gdk/docs/tools/tools-console/visualstudio/preprocessor-definitions) | `ms.date` 2026-11-13 |
| L3 | [GDKX layout](https://learn.microsoft.com/en-us/gaming/gdk/docs/gdk-dev/console-dev/overviews/gdkx-new-layout) | `ms.date` 2026-09-28 |
| L4 | [PIX for Windows](https://learn.microsoft.com/en-us/gaming/gdk/docs/tools/tools-pc/pix-for-windows) | `ms.date` 2026-10-16 |
| L5 | [DirectStorage on Xbox Series X\|S](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview) | `ms.date` 2026-11-13 |
| L6 | [Game streaming latency compensation](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/common/game-streaming/game-streaming-latency-compensation-deep-dive) | `ms.date` 2026-03-25, partly deprecated since GDK 2604 |
| B1 | [Announcing DirectX 12 Ultimate](https://devblogs.microsoft.com/directx/announcing-directx-12-ultimate/) | 19 March 2020 |
| B2 | [DirectStorage is coming to PC](https://devblogs.microsoft.com/directx/directstorage-is-coming-to-pc/) | 1 September 2020 |
| B3 | [DirectStorage 1.1 coming soon](https://devblogs.microsoft.com/directx/directstorage-1-1-coming-soon/) | 13 October 2022 |
| B4 | [Raising the bar with Direct3D](https://blogs.windows.com/windowsdeveloper/2013/10/14/raising-the-bar-with-direct3d/) | 14 October 2013 |
| W1 | [Xbox Series X tech](https://news.xbox.com/en-us/2020/03/16/xbox-series-x-tech/) | 16 March 2020 |
| W2 | [A closer look at Xbox Velocity Architecture](https://news.xbox.com/en-us/2020/07/14/a-closer-look-at-xbox-velocity-architecture/) | 14 July 2020 |
| W3 | [Xbox Series S product page](https://www.xbox.com/en-us/consoles/xbox-series-s) | read 20 November 2026 |
| W4 | [Xbox Series X\|S and RDNA 2](https://news.xbox.com/en-us/2020/10/28/a-closer-look-at-how-xbox-series-xs-integrates-full-amd-rdna-2-architecture/) | 28 October 2020 |
| HC | [Xbox Series X SoC, Hot Chips 32](https://hc32.hotchips.org/assets/program/conference/day1/HotChips2020_GPU_Microsoft_Jeff_Andrews_v2.pdf), Jeff Andrews and Mark Grossman (Microsoft) | 17 August 2020 |
| DF1 | [Inside Xbox Series X: the full specs](https://www.digitalfoundry.net/articles/digitalfoundry-2020-inside-xbox-series-x-full-specs), Digital Foundry quoting Andrew Goossen (Microsoft) | 16 March 2020 |
| DF2 | [Project Scorpio tech revealed](https://www.digitalfoundry.net/articles/digitalfoundry-2017-project-scorpio-tech-revealed), Digital Foundry quoting Andrew Goossen | 6 April 2017 |
| E1 | [Ray tracing open worlds](https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-RayTracingOpenWorlds-NetzelCosta.pptx), Aleksander Netzel and Tiago Costa (Epic Games), SIGGRAPH Advances | August 2022 |
| E2 | [Lumen performance guide](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-performance-guide-for-unreal-engine), Epic Games | Unreal Engine 5.9 documentation |
| F1 | [Metro Exodus on PlayStation 5 and Xbox Series X\|S](https://www.4a-games.com.mt/4a-dna/everything-technical-about-metro-exodus-upgrade-for-playstation-5-and-xbox-series-x-s), 4A Games | 17 June 2021 |

## Catalogue

### Platform and build

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D1 | On Xbox the GDK supports only D3D12.x. D3D11.x is not supported | D3D11 and D3D12 | "On Xbox, Microsoft Game Development Kit (GDK) supports only D3D12.x. D3D11.x is not supported." | L1 | Stated |
| D2 | Console titles target the Xbox extensions for Direct3D 12, and D3D12.x differs only a little between Xbox One ERA and the GDK | stock D3D12 | "you'll also need to target the Xbox extensions for Direct3D 12 (D3D12.x)". "There are only minor differences in D3D12.x for Xbox One ERA and Microsoft Game Development Kit (GDK)." | L1 | Stated |
| D3 | Separate headers and libraries: `d3d12_xs.h` and `d3d12_xs.lib` on Series X\|S, `d3d12_x.h` and `d3d12_x.lib` on Xbox One, plus `xg_xs.lib` and `xmem.lib` | `d3d12.h`, `d3d12.lib` | header choice under `_GAMING_XBOX_SCARLETT`, library list per platform | [GS pch][gs-pch], L3 | Stated |
| D4 | The Scarlett build defines `_GAMING_XBOX`, `_GAMING_XBOX_SCARLETT` and `WINAPI_FAMILY=WINAPI_FAMILY_GAMES` | `_GAMING_DESKTOP`, desktop API family | compiler switches listed per platform | L2 | Stated |
| D5 | A binary linked for one console generation fails device creation on the other with `D3D12_ERROR_DRIVER_VERSION_MISMATCH` | one runtime for all GPUs | error strings for both directions | [GS device][gs-dev-mismatch] (GDK 241000) | Stated |
| D6 | Interface requests use `IID_GRAPHICS_PPV_ARGS`, and `dxguid.lib` is not linked on Xbox | `IID_PPV_ARGS`, `dxguid.lib` | fallback macro and conditional pragma | [TK helpers][tk-helpers-iid], [TK helpers][tk-helpers-dxguid] | Code |

### Device creation and lifetime

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D7 | The device is created with `D3D12XboxCreateDevice` and a `D3D12XBOX_CREATE_DEVICE_PARAMETERS` structure instead of a minimum feature level | `D3D12CreateDevice(adapter, level, ...)` | samples pass a null adapter. The readme lists this as one of the few setup differences | [GS device][gs-dev-create], [GS readme][gs-sdasc-readme] | Stated |
| D8 | The title sizes engine resources itself: graphics command ring, graphics and compute scratch memory. Flags turn off geometry shader, tessellation and DXR allocations to save title memory. Amplification shaders must be enabled explicitly on Series X\|S since the October 2022 GDK | sized by the driver | parameter fields and their comments | [GS params][gs-sdasc-params] (GDK 220300) | Stated |
| D9 | Debug and instrumented drivers are chosen with `ProcessDebugFlags` at device creation (`D3D12_PROCESS_DEBUG_FLAG_DEBUG_LAYER_ENABLED`, `D3D12XBOX_PROCESS_DEBUG_FLAG_INSTRUMENTED`) | `ID3D12Debug::EnableDebugLayer` before creation | debug and profile build branches | [GS device][gs-dev-debug] | Code |
| D10 | `GetGpuHardwareConfigurationX` reports the console hardware version and GPU frequency | DXGI adapter description | values read and shown | [GS SystemInfo][gs-sysinfo-hw] | Code |
| D11 | Titles do not handle device removal | `DXGI_ERROR_DEVICE_REMOVED` and reset handling | "Xbox apps do not need to handle DXGI_ERROR_DEVICE_REMOVED or DXGI_ERROR_DEVICE_RESET." | [GS device][gs-dev-removed] | Stated |
| D12 | The title suspends and resumes the GPU explicitly (`SuspendX`, `ResumeX`) and registers its frame events again after resume | no equivalent | "Xbox apps need to explicitly suspend the GPU" | [GS suspend][gs-sdasc-suspend] | Stated |

### Presentation and frame pacing

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D13 | The GDK path has no DXGI swap chain. Back buffers are committed textures with `D3D12_HEAP_FLAG_ALLOW_DISPLAY`, sized from the device type (1080p Xbox One, 1440p Series S, 2160p Series X) | `IDXGISwapChain` buffers | back buffer creation | [GS buffers][gs-dev-backbuffers] | Code |
| D14 | Present is a command queue method, `ID3D12CommandQueue::PresentX`, taking display planes (`D3D12XBOX_PRESENT_PLANE_PARAMETERS`) | `IDXGISwapChain::Present` | "Present the backbuffer using the PresentX API" | [GS present][gs-dev-present], L6 | Stated |
| D15 | Frame pacing is explicit: `WaitFrameEventX` returns a frame pipeline token for `PresentX`, `SetFrameIntervalX` sets 24, 30, 40, 60 or 120 Hz intervals, `ScheduleFrameEventX` moves the origin event | waitable swap chain, sync interval | "games using Microsoft Game Development Kit (GDK) call `ID3D12Device::WaitFrameEventX` and receive a `D3D12XBOX_FRAME_PIPELINE_TOKEN`" | [GS wait][gs-dev-wait], [GS pacing][gs-fp-interval], L6 | Stated |
| D16 | `D3D12XBOX_PRESENT_PARAMETERS::ImmediateThresholdPercent` allows a late frame to flip immediately while scanout is within that share of the screen | tearing flag and sync interval 0 | comment on the field | [GS pacing][gs-fp-threshold] (GDK 241000) | Stated |
| D17 | Frame statistics are per frame token through `GetFrameStatisticsX`. The driver gives none for the first frame | `IDXGISwapChain::GetFrameStatistics` | statistics loop and its comment | [GS pacing][gs-fp-stats] | Stated |
| D18 | `PresentX` can be called on a compute queue | present belongs to the swap chain | compute queue present in the sample | [GS pacing][gs-fp-threshold] | Code |
| D19 | One present can carry several planes, for example HDR10 plus an SDR plane for GameDVR | one swap chain per window | two plane present | [GS HDR10][gs-hdr-planes] | Code |
| D20 | `IDXGIOutput::GetDesc` is not supported. Display modes, variable refresh and 24, 40 or 120 Hz support come from `GetDisplayModeListX` | `GetDesc`, `GetDisplayModeList` | "on Xbox, the IDXGIOutput::GetDesc method is not supported" | [GS SystemInfo][gs-sysinfo-modes], [GS 120 Hz][gs-120hz-modes] (GDK 220300) | Stated |
| D21 | Xbox One XDK only, not the GDK: the swap chain is created from the device instead of the queue, and `DXGIXPresentArray` presents several swap chains, with `DXGIXSetVLineNotification` for scanline events | DXGI from the queue | "Xbox One uses device here, not the command queue!" | [AS swap chain][as-esram-swapchain], [AS present][as-esram-present], [AS vline][as-async-vline] | Stated |

### Memory and addressing

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D22 | Titles allocate graphics memory themselves with `XMemAlloc`, whose attributes are built with `MAKE_XALLOC_ATTRIBUTES` from an allocator id, a memory type, a page size and an alignment | heaps and committed resources | texture loaders use `XALLOC_MEMTYPE_GRAPHICS_WRITECOMBINE_GPU_READONLY`, `XALLOC_PAGESIZE_64KB`, `XALLOC_ALIGNMENT_64K`. The bit layout of the attributes is not public | [TK loader][tk-xdds-attr], [TX][tx-attr] | Code |
| D23 | A resource can be placed at a raw GPU virtual address with `CreatePlacedResourceX`, and the public loader passes the CPU pointer from `XMemAlloc` as that address | `CreatePlacedResource(heap, offset, ...)` | `CreatePlacedResourceX(reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(grfxMemory), ...)` | [TK loader][tk-xdds-placed], [TX][tx-placed] | Code. That the two addresses are equal for this memory is implied |
| D24 | Memory is unified between CPU and GPU, and Series X splits it into 10 GB at 560 GB/s and 6 GB at 336 GB/s. Series S has 8 GB at 224 GB/s and 2 GB at 56 GB/s | discrete VRAM plus system memory | "The UMA-based alising trick used here". Goossen: "Memory performance is asymmetrical, it's not something we could have done with the PC" | [GS UMA][gs-fbc-uma], W1, W3, HC slide 5, DF1 | Stated |
| D25 | Titles can reserve virtual memory and map physical pages themselves (`XMemVirtualAlloc`, `XMemAllocatePhysicalPages`, `XMemMapPhysicalPages`) | `ID3D12Heap`, reserved resources | page allocator of the sample | [GS pages][gs-esram-pages] | Code |
| D26 | Page mappings are copied to the GPU on the GPU timeline with `RegisterPagePoolX`, `CopyPageMappingsX` and `CopyPageMappingsBatchX`, for 64 KB aliasing of transient resources | `UpdateTileMappings` on reserved resources | "copy CPU page table entries to the GPU TLB on the GPU timeline" | [GS readme][gs-esram-readme] | Stated |
| D27 | ESRAM exists only on Xbox One and One S. Xbox One X and Series X\|S have none | not applicable | "Xbox One X & Xbox Series X\|S don't have ESRAM" | [GS readme][gs-esram-readme] | Stated |

### Resource creation and layouts

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D28 | Extended creation calls take a `D3D12XBOX_RESOURCE_DESC` with delta colour compression options, and metadata planes can be placed at separate addresses (`CreateCommittedResourceX`, `CreateComponentPlacedResourceX1`) | driver managed compression | FSR 3 sample | [GS FSR3][gs-fsr3-dcc] | Code. Semantics implied |
| D29 | Texture memory layouts are console swizzle modes computed with the XG library (`xg_xs.h`) and passed in the layout field. Xbox One tile modes and Series swizzle modes are not interchangeable | `D3D12_TEXTURE_LAYOUT_UNKNOWN` or standard swizzle | loaders reject the other generation's layout | [TK loader][tk-xdds-tile], [TX][tx-tile] | Stated |
| D30 | Console resource flags control compression and indirect arguments: `D3D12XBOX_RESOURCE_FLAG_ALLOW_DCC`, `..._FORCE_TEXTURE_COMPATIBILITY`, `..._DENY_COMPRESSION_DATA`, `..._ALLOW_INDIRECT_BUFFER` with `D3D12XBOX_HEAP_FLAG_ALLOW_INDIRECT_BUFFERS` | no flags, state `INDIRECT_ARGUMENT` only | back buffer and depth flags, indirect argument buffer | [GS flags][gs-dev-dcc], [GS indirect][gs-ei-indirect] | Stated for DCC and indirect, code for the rest |
| D31 | New copy targets start in `COPY_DEST` on Xbox | `COMMON` | `c_initialCopyTargetState` | [TK helpers][tk-helpers-state] | Stated |
| D32 | Texture data pitch alignment is 1024 bytes | 256 bytes (`D3D12_TEXTURE_DATA_PITCH_ALIGNMENT`) | "Round up the srcPitch to multiples of 1024" | [TK screen grab][tk-screengrab-pitch] | Stated |
| D33 | Xbox only formats exist, such as `D16_UNORM_S8_UINT`, `R10G10B10_7E3_A2_FLOAT`, `R10G10B10_SNORM_A2_UNORM` and `R4G4_UNORM`, and Series X can render, blend and display a 32 bit 999E5 format | not available | loader format tables, slide 27 | [TK formats][tk-loader-formats], HC | Code, stated |

### Queues and command lists

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D34 | `CreateCommandQueueX` lets the title choose the hardware engine or pipe and the queue index | `CreateCommandQueue` with type and priority | compute queue creation | [GS queue][gs-fp-queue] | Code |
| D35 | Caches and pipeline stages are flushed explicitly with `FlushPipelineX`. Textures placed with `CreatePlacedResourceX` need a cache flush before use | resource barriers | header comment and transient allocator | [TK flush][tk-xdds-flush], [GS flush][gs-esram-flush] | Stated |
| D36 | `KickoffX` submits recorded work early to avoid a full frame of latency | no equivalent | "avoid forcing a full frame of latency" | [GS kickoff][gs-xcloud-kickoff] | Stated |
| D37 | Timestamps and cross queue waits go through memory: `Write64BitValueBottomOfPipeX`, `Wait64BitValueX` | timestamp queries, fences | timers and post processing pipeline | [GS timers][gs-timers], [GS waits][gs-ppp-wait] | Code |
| D38 | The console GPU runs two virtualised command streams with different DX API levels, one for the title and one for the system | not applicable | slide 27 | HC | Stated |
| D39 | Xbox One XDK only: DMA queues with hardware LZ compression (`D3D12XBOX_COMMAND_LIST_TYPE_DMA`, `LZDecompressMemoryX`) | copy queue, no LZ | DMA decompression sample | [AS DMA][as-dma] | Code |

### Shaders and pipeline state

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D40 | Shaders are compiled offline with an Xbox build of DXC (`DxCompiler_xs.dll`). FXC is not supported for Xbox, and only the precompiled console code is needed at runtime | DXIL, compiled by the driver at pipeline creation | "the legacy fxc compiler is no longer supported for Xbox", `__XBOX_STRIP_DXIL` | [GS compiler][gs-hlsl-compiler], [TK shaders][tk-compileshaders] | Stated |
| D41 | If a root signature does not match the one a shader was compiled with, the shader is recompiled when the pipeline state loads | root signature checked at creation | "Root signatures must match definition in each effect, or shaders will be recompiled on Xbox when PSO loads" | [TK root signatures][tk-rootsig] | Stated |
| D42 | Shader authors steer console compilation with macros such as `__XBOX_PRECOMPILE_VS_PS`, `__XBOX_ENABLE_WAVE32` and `__XBOX_PER_THREAD_SCRATCH_SIZE_LIMIT_IN_BYTES` | wave size chosen by the driver | sample shader headers | [GS macros][gs-dynlod-macros] | Code |
| D43 | With the ACO back end, a pipeline state is serialised offline on the PC (`AcoOfflineSerializePipelineStateXEx`) and loaded with `DeserializeGraphicsPipelineStateX`. The asset is tied to the GDK version. The PC side `d3d12_xs.dll` cannot deserialise Xbox root signatures | `ID3D12PipelineLibrary` | readme of the HLSL compile sample | [GS readme][gs-hlsl-readme] (ACO needs the October 2026 GDK) | Stated |
| D44 | `D3D12XBOX_ROOT_SIGNATURE_FLAG_RAYTRACING` cannot be written in HLSL. The sample deserialises the root signature, adds the flag and serialises it again | no such flag | comment and code | [GS DXR][gs-dxr-rootsig] (GDK 220300) | Stated |
| D45 | Xbox One only: geometry shaders on chip (`D3D12XBOX_PIPELINE_STATE_FLAG_GS_ONCHIP_TS_OFFCHIP`) and a native quad topology (`D3D12XBOX_PRIMITIVE_TOPOLOGY_TYPE_QUAD`) | not available | point sprite sample | [GS quads][gs-ps-quad] | Stated |

### Ray tracing

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D46 | The hardware accelerates ray box and ray triangle tests. BVH traversal runs in shaders, in parallel with other shader work | traversal owned by the driver | "Shader can run in parallel for BVH traversal" (slides 25 and 26). Epic: traversal is a compute shader on console | HC, E1 | Stated |
| D47 | Bottom level acceleration structures can be built offline on console. On PC a runtime build is the only option | runtime build | Goossen: "support for offline BVH construction and optimisation". Epic: runtime build is the "only option available on PC" | DF1, E1 | Stated |
| D48 | Studios work with the console acceleration structure format and traversal code directly | opaque format | 4A: formats "changing with almost every SDK release" (PlayStation 5 and Xbox together). Epic: "with access to console traversal code" | F1, E1 | Stated |
| D49 | Console traversal can be limited to a number of iterations | no such limit | "On consoles, you can limit the number of BVH traversal iterations" | E2 | Stated |
| D50 | DXR allocations can be turned off at device creation (`DisableDXR`) | not applicable | device parameter | [GS params][gs-sdasc-params] | Code |

### DirectX 12 Ultimate features

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D51 | DXR, variable rate shading, mesh shaders and sampler feedback form one feature set shared by PC and Xbox Series X, in hardware on Series X and S | DirectX 12 Ultimate | "an unprecedented alignment between PC and Xbox Series X" | B1, W4 | Stated |
| D52 | The Series X slide lists shading rate inputs per draw, vertex, primitive, screen tile and anti aliasing level | per draw, per primitive and screen space image | slide 17 | HC | Inputs stated. That PC lacks two of them is implied |
| D53 | Sampler feedback streaming adds a residency map, a request map and a residency map filter in hardware, about 2.5 times effective memory and I/O on average | sampler feedback without a published filter | slides 18 and 24, Velocity post | HC, W2 | Hardware stated. PC difference implied |
| D54 | The same meshlet sample builds for PC and Series X\|S, not for Xbox One | mesh shaders | sample readme | [GS meshlets][gs-meshlet-readme] | Implied shared API |

### Storage

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D55 | DirectStorage on Series X\|S is its own API (`IDStorageFactoryX`) with hardware BCPack and DEFLATE decompression and swizzle, aimed at 50,000 requests per second within 10 % of one core, with at least 2.0 GB/s over 250 ms | PC DirectStorage decompresses GDeflate on the GPU | "The new minimum performance guarantee is 2.0 GB/s over a 250 ms window for raw data." | L5, B3 | Stated |
| D56 | Series X reads 2.4 GB/s raw and 4.8 GB/s compressed, and DirectStorage was originally designed for this architecture | PC port of DirectStorage | Velocity post and DirectX blog | W2, B2 | Stated |
| D57 | Textures can be stored in the console swizzle layout and unshuffled on the GPU after hardware decompression | GDeflate stream | sample readme | [GS shuffle][gs-dstorage-readme] (GDK 240600) | Stated |

### Debugging and tools

| # | Difference | PC Direct3D 12 | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D58 | Validation is controlled with `SetDebugFlagsX`, `SetDebugErrorFilterX` and per resource `SetValidationFlagsX` | `ID3D12Debug`, `ID3D12InfoQueue` | comments on validation waits and false positives | [GS debug][gs-fp-debug], [GS filter][gs-ppp-filter], [GS UMA][gs-fbc-uma] | Stated |
| D59 | PIX for Xbox ships with the GDK. PIX for Windows is a separate tool for PC builds | PIX for Windows | "If you are an Xbox developer, use the version of PIX that is included with the GDK" | L4, [PIX download](https://devblogs.microsoft.com/pix/download/) | Stated |

### Xbox One context

These entries describe the previous console generation. They are listed because older sources mix them up with D3D12.X on Series X|S.

| # | Difference | PC | Evidence | Source | Kind |
|---|---|---|---|---|---|
| D60 | The Xbox One launch API was Direct3D 11.x, a superset of Direct3D 11.2 with lower runtime overhead | Direct3D 11.2 | Windows developer blog | B4 | Stated |
| D61 | "Fast semantics" is a Direct3D 11.X device option. With it the title rotates swap chain buffers itself | DXGI rotates buffers | XDK sample | [AS fast semantics][as-fast-semantics] | Stated |
| D62 | On Xbox One, D3D12 is built into the GPU command processor | runtime and driver on the CPU | Goossen in DF2 | DF2 | Stated |

## Claims without a public primary source

Often repeated online, but not found in any public primary source while preparing this catalogue:

* "Fast semantics" as a D3D12.X feature. Public sources show it for Direct3D 11.X only (D61).
* The 32 MB size and bandwidth of the Xbox One ESRAM from Microsoft's own documents. The Hot Chips 25 slides are no longer online, and the GDK pages on ESRAM require the developer agreement. Digital Foundry's 2013 interview with Microsoft engineers is the nearest public source.
* A default wave size on Series X\|S. Public code only shows the `__XBOX_ENABLE_WAVE32` opt in (D42).
* Differences in descriptor heaps or queue priorities beyond the entries above.
* "25 TFLOPS equivalent" ray tracing performance. Hot Chips gives 380 G/s ray box and 95 G/s ray triangle peak rates only.
* That console games never see shader compilation stutter because nothing is compiled at runtime (see D41).
* "GPU optimal" and "standard" memory as Microsoft's wording. The primary Microsoft wording is "GPU optimized memory" (W2). The other terms come from Digital Foundry's article.

## Consequences for `d3d12_xs.dll`

Only what follows from the catalogue:

* A replacement has to provide device creation from `D3D12XBOX_CREATE_DEVICE_PARAMETERS` (D7, D8) and presentation through queue methods and frame events (D13 to D17). That is a different shape from DXGI, not a renamed export.
* The `DXGIX` exports match the Xbox One XDK, not the GDK (D21). Titles of the GDK present with `PresentX` (D14).
* Forwarding root signature serialisation to PC D3D12 is unsupported by public evidence, and Microsoft's own readme points the other way (D43).
* Resources at raw addresses (D23), explicit flushes (D35), page mapping on the GPU timeline (D26) and console texture layouts (D29) have no direct PC counterpart. Translating them needs more than forwarding calls.

[gs-pch]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/pch.h#L38-L50
[gs-dev-mismatch]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L104-L113
[gs-dev-create]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L63-L103
[gs-dev-debug]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L73-L79
[gs-dev-removed]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L401
[gs-dev-backbuffers]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L243-L292
[gs-dev-present]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L386-L399
[gs-dev-wait]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L441-L474
[gs-dev-dcc]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleTriangle/DeviceResources.cpp#L254-L260
[gs-sdasc-readme]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleDeviceAndSwapChain/readme.md
[gs-sdasc-params]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleDeviceAndSwapChain/SimpleDeviceAndSwapChain.cpp#L253-L297
[gs-sdasc-suspend]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/IntroGraphics/SimpleDeviceAndSwapChain/SimpleDeviceAndSwapChain.cpp#L229-L243
[gs-sysinfo-hw]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/System/SystemInfo/SystemInfo.cpp#L699-L718
[gs-sysinfo-modes]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/System/SystemInfo/SystemInfo.cpp#L745-L788
[gs-120hz-modes]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/Simple120Hz/Simple120Hz.cpp#L561-L572
[gs-fp-interval]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/FramePacing/FramePacing.cpp#L2036-L2046
[gs-fp-threshold]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/FramePacing/FramePacing.cpp#L2096-L2112
[gs-fp-stats]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/FramePacing/FramePacing.cpp#L1305-L1366
[gs-fp-queue]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/FramePacing/FramePacing.cpp#L1739-L1746
[gs-fp-debug]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/FramePacing/FramePacing.cpp#L1822-L1827
[gs-hdr-planes]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/HDR10/DeviceResources.cpp#L423-L437
[gs-fbc-uma]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/FastBlockCompress/FBC_GPU.cpp#L377-L380
[gs-esram-pages]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/AdvancedESRAM/PageAllocator.cpp#L197-L232
[gs-esram-readme]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/AdvancedESRAM/readme.md
[gs-esram-flush]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/AdvancedESRAM/TransientAllocator.cpp#L108-L114
[gs-fsr3-dcc]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/AMDFidelityFX_FSR3FrameInterpolation/FSR3FrameInterpolation.cpp#L2535-L2563
[gs-ei-indirect]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/ExecuteIndirect/ExecuteIndirect.cpp#L1338-L1346
[gs-xcloud-kickoff]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/xCloud/CustomResolution/CustomResolution.cpp#L492-L494
[gs-timers]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Kits/ATGTK/PerformanceTimersXbox.cpp#L395-L404
[gs-ppp-wait]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/PipelinedPostprocess/PipelinedPostprocess.cpp#L292-L400
[gs-ppp-filter]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/PipelinedPostprocess/PipelinedPostprocess.cpp#L688-L696
[gs-hlsl-compiler]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/HlslCompile/MyHlslCompiler/MyHlslCompiler.cpp#L152-L216
[gs-hlsl-readme]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/HlslCompile/readme.md
[gs-dynlod-macros]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/DynamicLOD/Shared.h#L10-L19
[gs-dxr-rootsig]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/DXRTriangle/DXRTriangle/DXRTriangle.cpp#L350-L359
[gs-ps-quad]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/PointSprites/PointSprites.cpp#L477-L585
[gs-meshlet-readme]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/SimpleMeshlet/readme.md
[gs-dstorage-readme]: https://github.com/microsoft/Xbox-GDK-Samples/blob/b1e77b6ac10f95b07296025fda171ca2ddc42dbe/Samples/Graphics/DirectStorageTextureShuffling/readme.md
[tk-helpers-iid]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Inc/DirectXHelpers.h#L135-L137
[tk-helpers-dxguid]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Inc/DirectXHelpers.h#L33-L35
[tk-helpers-state]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Inc/DirectXHelpers.h#L141-L149
[tk-xdds-attr]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Src/XboxDDSTextureLoader.cpp#L35-L41
[tk-xdds-placed]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Src/XboxDDSTextureLoader.cpp#L200-L206
[tk-xdds-tile]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Src/XboxDDSTextureLoader.cpp#L292-L303
[tk-xdds-flush]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Inc/XboxDDSTextureLoader.h#L72-L90
[tk-screengrab-pitch]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Src/ScreenGrab.cpp#L283-L290
[tk-loader-formats]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Src/LoaderHelpers.h#L103-L120
[tk-compileshaders]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Src/Shaders/CompileShaders.cmd#L36-L53
[tk-rootsig]: https://github.com/microsoft/DirectXTK12/blob/f3555e3014a53c81a6202b786dc4d6509b95a936/Src/Shaders/RootSig.fxh#L6
[tx-attr]: https://github.com/microsoft/DirectXTex/blob/6e88c1c52e977cfd9c30671240f7995cba6d4968/Auxiliary/DirectXTexXboxD3D12X.cpp#L28-L39
[tx-placed]: https://github.com/microsoft/DirectXTex/blob/6e88c1c52e977cfd9c30671240f7995cba6d4968/Auxiliary/DirectXTexXboxD3D12X.cpp#L85-L91
[tx-tile]: https://github.com/microsoft/DirectXTex/blob/6e88c1c52e977cfd9c30671240f7995cba6d4968/Auxiliary/DirectXTexXbox.h#L24-L58
[as-esram-swapchain]: https://github.com/microsoft/Xbox-ATG-Samples/blob/ff629a6e4876e38fe8761ead116a0e53f634ca67/XDKSamples/Graphics/AdvancedESRAM12/DeviceResources.cpp#L239-L246
[as-esram-present]: https://github.com/microsoft/Xbox-ATG-Samples/blob/ff629a6e4876e38fe8761ead116a0e53f634ca67/XDKSamples/Graphics/AdvancedESRAM12/DeviceResources.cpp#L402-L420
[as-async-vline]: https://github.com/microsoft/Xbox-ATG-Samples/blob/ff629a6e4876e38fe8761ead116a0e53f634ca67/XDKSamples/Graphics/AsyncPresent/AsyncPresent.cpp#L57
[as-dma]: https://github.com/microsoft/Xbox-ATG-Samples/blob/ff629a6e4876e38fe8761ead116a0e53f634ca67/XDKSamples/Graphics/SimpleDmaDecompression/SimpleDmaDecompression12/SimpleDmaDecompression12.cpp#L736-L787
[as-fast-semantics]: https://github.com/microsoft/Xbox-ATG-Samples/blob/ff629a6e4876e38fe8761ead116a0e53f634ca67/XDKSamples/IntroGraphics/SimpleDeviceAndSwapChain/SimpleDeviceAndSwapChain.cpp#L123-L217
