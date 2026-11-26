# Zero copy `OpenExistingHeapFromAddress` evaluation for `title_mem`

This note documents the hardware evaluation of `ID3D12Device3::OpenExistingHeapFromAddress` as a candidate replacement for the `title_mem` mirror pools in `crates/decant-d3d12x/src/title_mem.rs`.

## Context

In the 180 second loading trace, `title_mem:copy_runs` copied 711,920 MiB (695 GiB) across 12,000 syncs (35.48 s of CPU copy time) and Peak Private Commit reached 25.27 GiB (16.70 GiB Working Set). `title_mem` currently copies dirty 64 KiB pages from `decant-xmem` read only views into 64 MiB `CreateHeap` mirror pools and maps them into 64 GiB `CreateReservedResource` windows with `UpdateTileMappings`.

## Measured hardware results

Measured with a hardware `ID3D12Device3` test harness on Windows 11 with an NVIDIA GeForce RTX 3090:

* `VirtualAlloc(MEM_RESERVE | MEM_COMMIT)` at 64 KiB and 2 MiB: `OpenExistingHeapFromAddress` succeeded (`SizeInBytes` matched, `D3D12_HEAP_TYPE_CUSTOM`, `D3D12_CPU_PAGE_PROPERTY_WRITE_BACK`, `D3D12_MEMORY_POOL_L0`, `Alignment=65536`, `Flags=1057`).
* `CreateFileMappingW(SEC_RESERVE)` + `MapViewOfFile(FILE_MAP_WRITE)` partially committed (64 KiB committed out of 2 MiB): `OpenExistingHeapFromAddress` failed with `0x80070057` (`E_INVALIDARG`).
* `CreateFileMappingW(SEC_RESERVE)` + `MapViewOfFile(FILE_MAP_WRITE)` fully committed (2 MiB): `OpenExistingHeapFromAddress` succeeded.
* `CreateFileMappingW(SEC_RESERVE)` + `MapViewOfFile(FILE_MAP_READ)` fully committed (2 MiB): `OpenExistingHeapFromAddress` failed with `0x80070057` (`E_INVALIDARG`).
* `CreatePlacedResource` with `D3D12_RESOURCE_FLAG_NONE` on the opened heap: failed with `0x80070057` (`E_INVALIDARG`).
* `CreatePlacedResource` with `D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER` on the opened heap: succeeded.
* GPU read coherency without `copy_runs`: CPU writes of `0xcafebabe12345678` and then `0xdeadbeef99990001` to the backing view were read back identically by `CopyBufferRegion` without `Map`, `Unmap`, or `copy_runs`.
* `UpdateTileMappings` from a `CreateReservedResource` window buffer onto the opened heap: failed and removed the device with `0x887A0001` (`DXGI_ERROR_INVALID_CALL`).
* `VirtualProtect(..., PAGE_READONLY)` on committed `VirtualAlloc` memory while an `OpenExistingHeapFromAddress` heap was live: failed with `0x80070057` (`E_INVALIDARG`).

## Driver and platform considerations

* **WDDM pinning**: `OpenExistingHeapFromAddress` locks the backing virtual pages in WDDM for GPU DMA, which prevents `VirtualProtect(..., PAGE_READONLY)` from succeeding and keeps the full allocation committed for the lifetime of the `ID3D12Heap`.
* **No tiled window support**: D3D12 forbids `UpdateTileMappings` onto heaps created by `OpenExistingHeapFromAddress`. `title_mem` relies on `CreateReservedResource` windows plus `UpdateTileMappings` to present scattered 64 KiB physical pages and aliased mappings as contiguous GPU address ranges.
* **Placed resource restrictions**: Placed buffers on an `OpenExistingHeapFromAddress` heap require `D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER` on this driver and cannot freely use `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` or arbitrary console resource layouts.
* **Windows vs Wine**: On Linux under Wine and `vkd3d-proton`, `OpenExistingHeapFromAddress` maps onto `VK_EXT_external_memory_host`, which has host pointer alignment constraints and likewise does not support sparse tile binding onto host memory heaps.

## Recommendation

**No Go** for replacing `title_mem` mirror pools (`CreateHeap` + `CreateReservedResource` + `UpdateTileMappings`) with `OpenExistingHeapFromAddress`.

1. `UpdateTileMappings` onto `OpenExistingHeapFromAddress` heaps causes device removal (`DXGI_ERROR_INVALID_CALL`), so scattered physical pages and aliased views cannot be stitched into contiguous GPU windows.
2. Partially committed `SEC_RESERVE` sections fail `OpenExistingHeapFromAddress` with `E_INVALIDARG`, and live heaps block `VirtualProtect(..., PAGE_READONLY)`, which would break `decant-xmem` write tracking.
3. To cut the 695 GiB `copy_runs` volume and 25.27 GiB Peak Private Commit, keep the `CreateReservedResource` mirror architecture and focus on reducing redundant dirty page faults, coalescing sync batches, and avoiding unnecessary page copies on static or GPU only allocations.
