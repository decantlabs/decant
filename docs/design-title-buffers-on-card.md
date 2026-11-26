# Title buffer residency on GPU VRAM (`title_mem` L1 vs L0 policy)

This design note defines the VRAM budget, hysteresis eviction rule, host RAM protection for 32 GiB and smaller machines, buffer residency classification, and Wave 20 R1 selector interaction for `title_mem` and the shared local VRAM owners in `crates/decant-d3d12x`.

## Context and measured baseline

On console hardware, CPU and GPU share unified memory at one address space. In Decant (`8efa0d2`), `crates/decant-d3d12x/src/title_mem.rs` mirrors title memory pages managed by `crates/decant-xmem` into 16 MiB D3D12 pool heaps (`POOL_HEAP_BYTES = 16 MiB`, `256` tiles of `64 KiB`) and maps them into 64 GiB `CreateReservedResource` windows with `UpdateTileMappings`. Wave 20 R1 (`d78d0a2`) added `MirrorPool::L1` (`D3D12_HEAP_TYPE_DEFAULT`) backed by a 64 MiB `D3D12_HEAP_TYPE_UPLOAD` `StagingRing` (`STAGING_BYTES = 64 MiB`, `8` command allocators) and a static device creation switch `DECANT_D3D12X_MIRROR_POOL` (`l0`, `l1`, or auto when `DedicatedVideoMemory >= 16 GiB`).

The design is grounded in two measured datasets:

* **Windows 24 GiB Wave 20 run (`8efa0d2`, Wave 20 R1 active in L1 mode)**: Windows 11, discrete 24 GiB NVIDIA GPU (`DedicatedVideoMemory = 24,326 MiB`, `IDXGIAdapter3::QueryVideoMemoryInfo` local `Budget = 23,558 MiB`), 32 GiB physical host RAM, `265.36 s` traced wall time (`3,907` presents, `16,146` traced `title_mem:sync` calls and `16,200` logged syncs, `metrics_wave20.json`, `trace.json`, and launch log).
  * Local VRAM peaked at `VramLocalMiB = 22,040.6 MiB` (`93.56%` of the `23,558 MiB` DXGI budget and `90.61%` of `24,326 MiB` dedicated VRAM), leaving `1,517.4 MiB` below budget. One `0x8007000E` (`E_OUTOFMEMORY`) occurred during late compute pipeline creation (`refused pipeline 37, hr=0x8007000E, compute, slot 62`).
  * At peak (`t = 252.4 s`), `OwnerLocalAccountedMiB = 21,371.5 MiB` broke down into `OwnerTitleMirrorMiB = 6,704.0 MiB` (`419` L1 heaps of `16 MiB`, up to `107,264` tiles), `OwnerViewHeapsMiB = 5,056.0 MiB` (holding `OwnerViewTexturesMiB = 4,844.8 MiB` across more than `41,566` created view textures), and `9,611.5 MiB` of other accounted local owners (committed render targets, depth targets, committed stand in resources for `CreatePlacedResourceX` textures, `DetileStaging` at its `256 MiB` cap, GPU visible descriptor heaps, and `3 MiB` of `ServiceBuffer`). The remaining `669.2 MiB` was unaccounted local usage (DXGI swapchain backbuffers at `3840x2160` RGBA8, driver page tables, command buffers, shader ISA, and PSO scratch).
  * Moving `title_mem` heaps from L0 to L1 lowered `WorkingSetMiB` max from `24.82 GiB` (pre R1 L0 baseline) to `18.97 GiB` (`19,423.0 MiB`), with `AvailRamMiB` staying at or above `6,379.8 MiB`. However, `PrivateCommitMiB` peaked at `30,290.1 MiB` (`29.58 GiB`) on 32 GiB host RAM because `xmem` section commits, compiler workers, and WDDM commit backing for L1 heaps all charge system commit.
  * `VramNonLocalMiB` peaked at `772.5 MiB` (`OwnerUploadMiB` max `106.1 MiB`, mean `50.9 MiB`, plus the untracked `64 MiB` `StagingRing`, `8 MiB` page table, and detile parameter buffers).
  * Across `16,200` syncs, `title_mem` copied `2,517,663` pages (`157,353.9 MiB` or `153.67 GiB`), a `23.47x` average rewrite factor over the `6,704 MiB` mirror ceiling. Total `title_mem:sync` time was `53.94 s` (`37.04 s` in `take_dirty`, `5.49 s` in `copy_runs`, `3.20 s` in `note_tiles`, `0.70 s` in `upload`, `0.25 s` in `update_tiles`).
* **Linux 16 GiB reference run**: Public speed fork family on a 16 GiB AMD GPU under RADV, `vkd3d-proton`, and Wine, with 31 GiB host RAM over 14 minutes of world play.
  * Keeping copies of title buffers on the GPU card (with CPU side copy hashing in that fork) reduced GPU frame time from `79 ms` to `66 ms`, confirming that on card residency matters for GPU bound rendering.
  * However, an eviction loop near `88%` to `90%` VRAM released `25 GiB` and re created `5.5 GiB` over 14 minutes while textures arrived `340` to `450` frames late. Lifting the low memory limits reduced the oldest overdue texture from `340` to `450` frames down to `8` frames.

### Where the measurement stops

The Wave 20 trace (`8efa0d2`) records aggregate per sync durations, 1 Hz memory owner totals, logged `place_buffer` and `XMemMapPhysicalPages` calls, and 100 sync copy totals. It does not record per virtual address GPU shader read counts or per page reuse histograms. Buffer residency classes below are derived from observed placement sizes, `detile` texture registrations, `upload_mem` chunk lifetimes, and the three measured sync volume phases. Exact per page GPU read heat counters are not yet instrumented and are noted explicitly where they enter promotion heuristics.

### Root causes in the `8efa0d2` implementation

Four concrete defects in `8efa0d2` cause VRAM exhaustion on 24 GiB and immediate oversubscription on 16 GiB cards:

1. **Unbounded all or nothing L1 placement (`crates/decant-d3d12x/src/title_mem.rs:550-574`)**: `parse_mirror_pool` checks `DedicatedVideoMemory >= 16 GiB` once at device creation, ignores `QueryVideoMemoryInfo().Budget`, places every mirror page into permanent L1 heaps, and never demotes or releases a heap until device destruction. On a 16 GiB card (`Budget` near `14.3` to `15.4 GiB`), placing `6.7 GiB` of mirror heaps in L1 on top of `~10 GiB` of render targets and textures guarantees oversubscription.
2. **Duplicate VRAM storage of tiled texture sources (`detile.rs` + `title_mem.rs`)**: During world loading (`t = 30..90 s`), `OwnerTitleMirrorMiB` grew by `+4,688 MiB` (`1,712` to `6,400 MiB`) in lockstep with `OwnerViewHeapsMiB` (`+4,592 MiB`, `144` to `4,736 MiB`). Every streamed console texture is copied into an L1 mirror heap, read once by `detile.hlsl` into a separate L1 PC texture in `view_heap` or `CommittedSampled`, and then kept permanently in L1 inside `title_mem`.
3. **Monotonic mixed class tile allocation (`crates/decant-d3d12x/src/title_mirror.rs:171-192`)**: `give_tile` hands out tiles monotonically (`0, 1, 2, ...`) in first write order. One GPU written buffer page or pinned page inside a 16 MiB heap prevents that entire 16 MiB `ID3D12Heap` from ever being reclaimed.
4. **Uncoordinated, zero hysteresis view texture eviction (`crates/decant-d3d12x/src/views_at.rs:193-217,331-436` and `proxy.rs:2930-3017`)**: `views_at::cap_from` ignores `budget` whenever `dedicated != 0` and subtracts only a `2 GiB` reserve (`22,278 MiB` cap on a 24 GiB card). `ViewTextures::collect` triggers and stops at the exact same threshold (`held_before > cap` to `held_after <= cap`), drops idle guards to `8` presents and `0 ms` when `held > 2 * cap`, and does not coordinate with live `QueryVideoMemoryInfo().CurrentUsage` or `title_mem` L1 heaps. Meanwhile `collect_budget` (`views_at.rs:441`) is dead code.

---

## 1. Driver budget driven VRAM budget rule (8, 12, 16, and 24 GiB cards)

All local VRAM residency decisions are governed by one shared budget controller queried from `IDXGIAdapter3::QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut info)`. The controller reads live `B = info.Budget` and `U_local = info.CurrentUsage` at 1 Hz (reusing the existing `IDXGIAdapter3` handle from `mem_owners.rs:620` / `mem_sample.rs:50`) and before allocating any new `16 MiB` `title_mem` L1 heap or `64 MiB` `view_heap`. If `QueryVideoMemoryInfo` fails, `B` falls back to `floor(0.90 * DedicatedVideoMemory)`, or `0` (strict L0 only) if both are unknown.

### Core formulas

Let `B` be the driver reported local VRAM budget in MiB. Let `U_pending_retire` be the exact sum of bytes in local `title_mem` heaps and `view_heap`s already unmapped and queued in `retire::release` awaiting GPU fence completion. Define effective local usage:

```text
U_eff(t) = max(0, info.CurrentUsage(t) - U_pending_retire(t))
```

Define the non overlapping reserves and watermarks from `B`:

* **Unallocated driver, swapchain, and PSO burst reserve (`R_drv`)**:
  `R_drv(B) = max(1536 MiB, floor(0.16 * B))`
  This VRAM stays unallocated below `Budget` so swapchain resizes, driver internal scratch, PSO compilation (preventing the Wave 20 `0x8007000E` failure at `1,517 MiB` headroom), and `detile` staging bursts never exhaust driver memory.
* **Global high water mark (`HWM_total`) and low water mark (`LWM_total`)**:
  `HWM_total(B) = B - R_drv(B)`
  `Delta_H(B)   = max(1024 MiB, floor(0.10 * B))`
  `LWM_total(B) = HWM_total(B) - Delta_H(B)`
  On every card with `B >= 10,240 MiB`, `HWM_total` sits at `84.0%` of `Budget` (strictly below the `88%` to `90%` RADV/vkd3d-proton thrash band) and `LWM_total` sits at `74.0%` of `Budget`.
* **Protected render target, depth, and mandatory GPU floor (`R_rt`)**:
  `R_rt(B) = clamp(floor(0.25 * B), 2048 MiB, 4096 MiB)`
  This floor is reserved exclusively for render targets (`CommittedRenderTarget`), depth/stencil buffers (`CommittedDepth`), swapchain targets, `detile` staging (`256 MiB` cap), GPU descriptor heaps, and `ServiceBuffer`. Even at boot when `CurrentUsage` is small (`616 MiB` at `t = 1 s` in Wave 20), `title_mem` computes non mirror demand as `U_non_mirror(t) = max(U_eff(t) - U_mirror_L1(t), R_rt(B))` so mirror heaps can never pre fill space reserved for render targets created later.
* **Hard L1 mirror ceiling (`M_cap`) and mirror hysteresis band (`M_hwm`, `M_lwm`)**:
  * `B < 10,240 MiB` (8 GiB class): `M_cap = 0 MiB` (`0` heaps, L0 only)
  * `10,240 MiB <= B < 14,336 MiB` (12 GiB class): `M_cap = 1,024 MiB` (`64` heaps of `16 MiB`)
  * `14,336 MiB <= B < 20,480 MiB` (16 GiB class): `M_cap = 2,048 MiB` (`128` heaps of `16 MiB`)
  * `B >= 20,480 MiB` (24 GiB class): `M_cap = 4,096 MiB` (`256` heaps of `16 MiB`)
  * Live mirror high and low watermarks:
    `M_hwm(t) = min(M_cap(B), round_down_16(max(0, HWM_total(B) - U_non_mirror(t))))`
    `M_lwm(t) = round_down_16(floor(0.75 * M_hwm(t)))` (when `M_hwm > 0`, maintaining at least `256 MiB` or `16` heaps of deadband at full `M_cap`)
* **Coordinated view texture cache target (`V_cap`) and low watermark (`V_lwm`)**:
  `V_cap` replaces the fixed `dedicated - 2 GiB` rule in `views_at.rs:193-203` with tiered targets (`1,536`, `2,048`, `3,072`, and `5,120 MiB` for 8, 12, 16, and 24 GiB classes) bounded dynamically by `HWM_total(B) - U_non_view(t)`, with `V_lwm = round_down_64(floor(0.75 * V_cap))`.

### Exact budget numbers by GPU memory tier

Because `B = info.Budget` varies slightly by OS, compositor, and WDDM/RADV reservation, the table states exact numbers for nominal `Budget` (`~93.8%` of physical on 8 to 16 GiB, a lower Linux RADV/compositor 16 GiB budget at `14,336 MiB`, and the exact Wave 20 measured 24 GiB budget `23,558 MiB`). At runtime, the formulas above compute exact MiB values directly from live `info.Budget`.

| Card tier | Dedicated (MiB) | Driver `Budget` `B` (MiB) | Driver/burst reserve `R_drv` (MiB) | Global `HWM_total` (MiB / % of `B`) | Global `LWM_total` (MiB / % of `B`) | Deadband `Delta_H` (MiB) | Protected RT/DS floor `R_rt` (MiB) | L1 mirror `M_cap` (MiB / 16 MiB heaps) | L1 mirror `M_lwm` (MiB) | View cache `V_cap` / `V_lwm` (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 GiB (nominal) | 8,192 | 7,680 | 1,536 (20.0%) | 6,144 (80.0%) | 5,120 (66.7%) | 1,024 | 2,048 | 0 (0) | 0 | 1,536 / 1,152 |
| 12 GiB (nominal) | 12,288 | 11,520 | 1,843 (16.0%) | 9,677 (84.0%) | 8,525 (74.0%) | 1,152 | 2,880 | 1,024 (64) | 768 | 2,048 / 1,536 |
| 16 GiB (nominal) | 16,384 | 15,360 | 2,457 (16.0%) | 12,903 (84.0%) | 11,367 (74.0%) | 1,536 | 3,840 | 2,048 (128) | 1,536 | 3,072 / 2,304 |
| 16 GiB (RADV tight `B`) | 16,384 | 14,336 | 2,293 (16.0%) | 12,043 (84.0%) | 10,610 (74.0%) | 1,433 | 3,584 | 2,048 (128) | 1,536 | 3,072 / 2,304 |
| 24 GiB (Wave 20 measured) | 24,326 | 23,558 | 3,769 (16.0%) | 19,789 (84.0%) | 17,434 (74.0%) | 2,355 | 4,096 | 4,096 (256) | 3,072 | 5,120 / 3,840 |

Verification against the Wave 20 24 GiB peak (`U_non_mirror = 15,336.6 MiB` = `9,611.5 MiB` other local + `669.2 MiB` unaccounted + `5,056.0 MiB` view heaps):

* Even at full `M_cap = 4,096 MiB`, peak local usage is `15,336.6 + 4,096.0 = 19,432.6 MiB` (`82.5%` of `23,558 MiB` budget), staying strictly below `HWM_total = 19,789 MiB` and leaving `4,125.4 MiB` of unallocated headroom (`2.7x` the `1,517.4 MiB` headroom at which Wave 20 hit `0x8007000E`).
* Once `P0` detiled texture sources (estimated `~4.5 GiB` of Wave 20's `6,704 MiB` mirror) stay in transient L0 rather than L1, active L1 mirror residency drops near `~2,200 MiB`, putting peak local VRAM around `~17,537 MiB` (`74.4%` of `Budget`, `~6,021 MiB` headroom).

---

## 2. Hysteresis eviction rule and non oscillation proof

Decant's `CreateReservedResource` window (`title_mem.rs:1105-1121`) maps each 64 KiB window tile onto any `16 MiB` `ID3D12Heap` via `UpdateTileMappings` without changing the GPU virtual address seen by shaders or placed buffers. This enables **VA preserving L1 to L0 demotion** for title buffers instead of destructive eviction.

### Priority classes (0 = evict first, 4 = never evict)

Pool heaps are partitioned by `(MirrorPool, TileClass)` so high priority tiles never pin low priority 16 MiB heaps (eliminating the `2,497 textures in 0 heaps freed` fragmentation observed in Wave 20 `view_heap` at log line `L2229`):

* **Class `P0` (Disposable read once texture sources, L0 only, discard to `Target::Null`)**: Pages belonging solely to `detile::Registry` textures whose detile flush fence has completed and that do not overlap any `place_buffer` or active buffer view. Never placed in L1. Recycled to `Target::Null` once detiled (the authoritative bytes remain in `xmem` read only views and re materialize on demand if rewritten or translated as a buffer).
* **Class `P1` (Hot CPU write / streaming rings, L0 only, immediate demotion if ever in L1)**: Pages marked hot by `xmem::track` (`HOT_FAULTS = 3` within `HOT_WINDOW = 250 ms`), pages dirtied in `>= 2` of the last `8` presents, or pages carrying active `IO_HOLD` / `active_writes` kernel I/O pins (video planes, per frame dynamic staging). Never promoted to L1. If a static `P2` tile turns hot, it demotes to L0 on its next sync copy with zero extra copy cost.
* **Class `P2` (Write clean static read only buffers, L1 eligible, primary L1 demotion candidate)**: CPU authored static vertex, index, meshlet, constant, and LUT pages that are not GPU written and have had `0` CPU writes for at least `T_stable = 3,000 ms` and `60` presents. Demotion copies clean bytes from `xmem`'s read only section view to an L0 heap on CPU threads (zero GPU readback), remaps all window and placed buffer tiles via `UpdateTileMappings`, waits for retirement, and frees emptied 16 MiB L1 heaps. Also includes idle unreferenced `views_at` textures (`!view_shadow::referenced().contains(&res)`), which are evicted only **after** eligible `title_mem` `P2` L1 heaps because destroying a view texture costs a future re detile whereas demoting a `P2` buffer tile to L0 is lossless.
* **Class `P3` (GPU written buffers, L1 strongly preferred, emergency GPU migration only)**: Buffers created by `title_mem::place_buffer` (`CreatePlacedResourceX` buffers) and UAV/acceleration structure spans (`give_span`). Because GPU writes do not flow back to `xmem` (`title_mem.rs:35`), `P3` tiles are **never discarded to `Target::Null`**. Under emergency pressure only (`U_eff > B - 512 MiB` after all `P0..P2` are exhausted and tile idle `>= 600` presents), a `P3` heap migrates via GPU `CopyBufferRegion` (L1 to L0) followed by `UpdateTileMappings` and fence wait before releasing the L1 heap.
* **Class `P4` (Pinned non evictable)**: Swapchain backbuffers, render targets, depth/stencil targets, GPU descriptor heaps, `ServiceBuffer`, active `detile` staging, view textures referenced by any live descriptor slot (`view_shadow::referenced()`), and any heap with an incomplete GPU retirement fence or age below `T_min_res`.

### Two state hysteresis controller and temporal guards

Let controller state be `S in {NORMAL, EVICTING}`:

1. **Transition `NORMAL -> EVICTING`**: Fires when `U_eff(t) > HWM_total(B)` OR `U_mir(t) > M_hwm(t)`.
2. **Transition `EVICTING -> NORMAL`**: Fires only when **both** `U_eff(t) <= LWM_total(B)` AND `U_mir(t) <= M_lwm(t)`. If a full eviction sweep over eligible `P0..P2` heaps reclaims `0` heaps (all remaining VRAM is `P3`/`P4`), the controller enters a quiet backoff for `60` presents so it cannot spin.
3. **Asymmetric promotion admission gate**: L0 to L1 promotion of `P2` tiles is allowed **only** when all four conditions hold simultaneously:
   * `S == NORMAL`.
   * `U_eff(t) + batch_bytes <= LWM_total(B)` AND `U_mir(t) + batch_bytes <= M_lwm(t)` (strictly at or below **low water marks**, never inside the `(LWM, HWM]` deadband).
   * The global post eviction freeze window `T_freeze` (`300` presents AND `5,000 ms`) has elapsed since the last `EVICTING -> NORMAL` transition.
   * Promotion rate is capped at `R_prom <= 64 MiB/s` (`<= 16 MiB` per present).
4. **Per tile / per heap minimum residency age (`T_min_res`)**: No L1 tile or `view_heap` texture may be demoted or evicted until `age_presents >= 120` AND `age_ms >= 5,000 ms` (except `P1` hot reclassification, which moves cleanly to L0 on the next write, or critical OOM guard `U_eff > B - 768 MiB` which may demote `P2` to L0 after GPU fence completion and `age_ms >= 2,000 ms` while preserving GPU VAs). For `views_at` texture destruction, `MIN_IDLE_PRESENTS = 180` AND `MIN_IDLE_MS = 15,000 ms` remain mandatory (removing the current `views_at.rs:337-342` `severely_over` bypass that collapses idle protection to `8` presents and `0 ms`).
5. **Per page exponential re promotion cooldown (`T_cool`)**: Each time page `p` is demoted from L1 to L0, its demotion counter `k_p` increments by `1`. Page `p` cannot re enter L1 until it stays write clean and satisfies:
   `delta_presents >= 300 * 2^(min(k_p - 1, 4))` AND `delta_ms >= 10,000 * 2^(min(k_p - 1, 4))`
   giving backoff steps of `10 s`, `20 s`, `40 s`, `80 s`, and `160 s` (capped at `160 s` and `4,800` presents).
6. **Whole heap evacuation and bounded step**: Each eviction step processes at most `256 MiB` (`16` heaps of `16 MiB`) and waits `8` presents before the next step. If a target `L1-P2` heap has `<= 16` live tiles remaining, those tiles compact into another `L1-P2` heap or demote to L0 so the whole `16 MiB` `ID3D12Heap` reaches `0` live tiles and returns to D3D12.

### Mathematical proof of non oscillation

Define the state space `(S, U_eff, U_mir)`. We prove that the controller cannot enter the single threshold evict/re upload oscillation observed near `88%` to `90%` VRAM on the 16 GiB Linux reference run (`25 GiB` released, `5.5 GiB` re created in 14 minutes).

* **Invariant 1 (Disjoint action regions with positive energy gap)**:
  For every supported card (`B >= 7,680 MiB`), `HWM_total(B) - LWM_total(B) = Delta_H(B) >= 1,024 MiB`, and for every L1 enabled card (`M_cap >= 1,024 MiB`), `M_hwm - M_lwm >= 256 MiB`. Let `A_evict = {(S, U_eff, U_mir) : U_eff > HWM_total or U_mir > M_hwm}` and `A_promote = {(S, U_eff, U_mir) : S = NORMAL and U_eff <= LWM_total and U_mir <= M_lwm}`. Then `A_evict` and `A_promote` are strictly disjoint and separated by an L1 distance of at least `min(Delta_H, M_hwm - M_lwm) >= 256 MiB` (`>= 1,024 MiB` globally, `1,536 MiB` on a 16 GiB card).
* **Invariant 2 (Zero self induced limit cycles in steady state)**:
  Assume non mirror title allocations `U_non_mirror` are constant (or non increasing) after an eviction cycle completes at time `t_0` with `U_eff(t_0) <= LWM_total` and `U_mir(t_0) <= M_lwm`. Every subsequent promotion step `b > 0` enforces the precondition `U_eff + b <= LWM_total` and `U_mir + b <= M_lwm`. By induction over all promotion steps, `U_eff(t) <= LWM_total <= HWM_total - 1,024 MiB < HWM_total` and `U_mir(t) <= M_lwm < M_hwm` for all `t >= t_0`. Therefore L1 promotions alone can **never** re enter `A_evict`. In steady state, the system undergoes at most one eviction pass and zero oscillations for all future time.
* **Invariant 3 (Bounded worst case churn under adversarial external sawtooth demand)**:
  Suppose external title allocations (`U_non_mirror`) oscillate adversarially by `>= Delta_H` (`>= 1,536 MiB` on a 16 GiB card) to force repeated `NORMAL -> EVICTING -> NORMAL` transitions. Any single page `p` that is promoted and demoted `k` times must spend at least `T_min_res = 5 s` resident in L1 and at least `T_cool(j) = 10 * 2^(min(j - 1, 4)) s` cooled down in L0 on round trip `j`. Cumulative elapsed time after `k` round trips satisfies:
  `sum_{j=1..k} (5 + 10 * 2^(min(j - 1, 4))) <= D`
  For a `14` minute (`D = 840 s`) session matching the Linux 16 GiB reference run, round trips `1` through `5` require `15 + 25 + 45 + 85 + 165 = 335 s`, and each subsequent round trip requires `165 s` (`500 s`, `665 s`, `830 s`). Thus any individual page can bounce **at most `8` times in 14 minutes**, regardless of workload pattern. Moreover, `U_eff` subtracts `U_pending_retire`, so GPU retirement latency cannot trigger duplicate eviction waves on adjacent frames.
* **Invariant 4 (Decoupling of buffer demotion from texture streaming latency)**:
  Because `P2` buffer demotion remaps window tiles from L1 to L0 while keeping the GPU virtual address live and valid, buffer demotion introduces **0 frames of stall or re creation delay** for shaders or textures. Because `P2` L1 buffer demotion runs ahead of `views_at` texture destruction and `views_at` preserves its `180` present and `15,000 ms` idle guard plus `view_shadow::referenced()` pins, actively drawn textures cannot be prematurely evicted and pushed `340` to `450` frames behind.

---

## 3. Behavior when host RAM is 32 GiB or less

Wave 20 measured the dual constraint directly on a 32 GiB Windows host:

* Pre R1 (all L0 mirror): `WorkingSetMiB` peaked at `24.82 GiB`, consuming `78%` of 32 GiB physical RAM for one process because every mirrored page was resident twice in host RAM (once in `xmem` backing sections and once in pinned L0 D3D12 heaps).
* Wave 20 R1 (all L1 mirror): `WorkingSetMiB` dropped to `18.97 GiB` (`AvailRamMiB` min `6.23 GiB`), yet `PrivateCommitMiB` still reached `29.58 GiB` (`30,290.1 MiB`) while `VramLocalMiB` hit `22,040.6 MiB` and failed with `0x8007000E`.

Blindly demoting `4` to `6 GiB` of L1 mirror heaps into permanent L0 heaps on a `<= 32 GiB` host (especially 8, 12, or 16 GiB GPU pairings) would merely trade VRAM OOM for host pagefile thrashing inside `title_mem:copy_runs` and `xmem`. The third path that avoids **both** VRAM OOM and host swap thrashing is **discarding clean `P0` and cold `P2` mirror tiles to `Target::Null`**, because `xmem` already holds the authoritative CPU copy in its backing sections.

### Explicit `TotalPhys <= 32 GiB` policy

Query `GlobalMemoryStatusEx` (`ullTotalPhys`, `ullAvailPhys`, `ullAvailPageFile`, already wired in `mem_owners.rs:564-573`) alongside `QueryVideoMemoryInfo`. Whenever `ullTotalPhys <= 33,554 MiB` (capturing 16 GiB, 31 GiB Linux, and 32 GiB Windows machines):

1. **Bounded L0 mirror pool ceiling (`L0_cap`)**:
   * `TotalPhys <= 16 GiB`: `L0_cap = 1,024 MiB` (`64` heaps of `16 MiB`).
   * `16 GiB < TotalPhys <= 32 GiB`: `L0_cap = 2,048 MiB` (`128` heaps of `16 MiB`) while `AvailPhys >= 4,096 MiB` and `AvailPageFile >= 4,096 MiB`.
   * Dynamic RAM pressure throttle: if `AvailPhys < 4,096 MiB` or `AvailPageFile < 4,096 MiB`, shrink `L0_cap` to `1,024 MiB`. If `AvailPhys < 2,560 MiB` or `AvailPageFile < 2,048 MiB`, shrink `L0_cap` to `512 MiB` and halt all L0 heap growth.
   * (For reference, on hosts with `TotalPhys > 32 GiB`, `L0_cap = min(6,144 MiB, floor(0.20 * TotalPhys))`.)
2. **Mandatory transient `P0` discard (saves `~4.5` to `5.0 GiB` in BOTH VRAM and host RAM)**:
   * Tiled texture source pages (`P0`) live only in a bounded transient L0 sub pool (`<= 512 MiB`, `<= 256 MiB` when `AvailPhys < 4,096 MiB`).
   * Immediately after `detile::flush` completes and its GPU fence retires (and no `place_buffer` or buffer view covers those pages), unmap their window tiles to `Target::Null`, mark the page `MirrorState::Evicted` (distinct from `Untouched`), and return the tile to the `L0-P0` free list. When an `L0-P0` heap empties beyond `1` warm heap, release the `ID3D12Heap` to D3D12 so both `WorkingSetMiB` and `PrivateCommitMiB` drop immediately.
   * If a title later writes those pages or calls `title_mem::translate` on them as a buffer, `Evicted` pages re materialize cleanly from `xmem`'s read only section view before submission.
3. **Cold `P2` L0 discard under host RAM pressure (instead of paging)**:
   * When L1 VRAM pressure requires demoting a `P2` tile AND host RAM is already at `U_mirror_L0 >= L0_cap` or `AvailPhys < 4,096 MiB`, do **not** allocate a new L0 heap. First drain all `P0` tiles to `Target::Null`. If `U_mirror_L0` is still above `0.75 * L0_cap`, discard cold clean `P2` L0 tiles (idle `>= 300` presents and `>= 10,000 ms`, never GPU written) to `Target::Null` and release emptied `16 MiB` L0 heaps.
   * `P3` GPU written tiles are never discarded to `Target::Null`. They keep reserved slots in L1 (or emergency L0 within `L0_cap`).
4. **Auxiliary staging and zero page savings on `<= 32 GiB` hosts**:
   * Size `title_mem::StagingRing` to `32 MiB` (instead of `64 MiB`) when `TotalPhys <= 32 GiB` and `AvailPhys < 6,144 MiB` (still accommodating `512` pages per sync chunk).
   * Trim idle `detile` staging (`detile.rs:753-767`) from `256 MiB` down to `64 MiB` whenever `AvailPhys < 4,096 MiB` or `U_eff > HWM_total`.
   * Never back evicted `views_at` textures with system RAM copies. Destroy them through `retire::release` and rely on `xmem` for on demand re detiling if ever requested again.
   * Avoid allocating physical tiles in `give_span` (`title_mem.rs:1674`) for untouched all zero pages unless the span is a UAV or `place_buffer` target (`Target::Null` already reads as zeros under D3D12).
   * Enforce a `4 GiB` commit floor before creating new L0/L1 mirror heaps (tighter than `xmem`'s `HEADROOM_BYTES = 1 GiB` in `crates/decant-xmem/src/pages.rs:44`) so mirror growth cannot starve `xmem::commit_page` or DXC worker threads.

Projected impact on the Wave 20 32 GiB Windows host: discarding `~4.5 GiB` of post detile `P0` mirror pages reduces `OwnerTitleMirrorMiB` from `6,704 MiB` to `~2.2 GiB`, bringing `PrivateCommitMiB` down from `29.58 GiB` to `~25.0 GiB`, keeping `WorkingSetMiB` near `~18.5` to `19.5 GiB`, and lowering `VramLocalMiB` from `22,041 MiB` (`93.6%`) to `~17.5 GiB` (`74.4%`).

---

## 4. Buffer residency classification by size and write frequency (Wave 20 trace)

### Break even bandwidth model

For a buffer or `64 KiB` tile of size `S` written by the CPU at frequency `f_w` and read by the GPU at frequency `f_r`, let `R = f_r / f_w` be GPU reads per CPU write:

* **In L0 (`WRITE_BACK` system memory)**: Each CPU write pays one CPU `memcpy` (`CopyPool` measured `157,354 MiB / 5.492 s = 28.0 GiB/s` aggregate across `2..=8` workers in Wave 20), `0` `CopyBufferRegion` commands, `0` L1 VRAM, and `R * S` bytes read over PCIe.
* **In L1 (`DEFAULT` VRAM via `StagingRing`)**: Each CPU write pays one CPU `memcpy` into `StagingRing`, **one PCIe write** (`1 * S` via `CopyBufferRegion` on `map_queue` + queue fence wait), `S` bytes of L1 VRAM, and `R * S` bytes read at local VRAM bandwidth (`20x` to `40x` PCIe bandwidth).

Therefore:

* **`R <= 1` is strictly worse in L1 than in L0**: Moving bytes to L1 transfers `1 * S` over PCIe (via `CopyBufferRegion`), the exact same PCIe byte volume as reading directly from L0 once, while adding staging ring pressure, command list overhead, and permanent L1 VRAM consumption.
* **`R >= 2` wins in L1**: Each subsequent GPU read hits local VRAM instead of PCIe. The larger `R * S`, the higher the GPU frame time reduction (matching the `79 -> 66 ms` GPU improvement reported for on card buffer residency in the public speed fork).

### Classification from the Wave 20 trace (`8efa0d2`)

| Buffer class | Size range (Wave 20 observed) | CPU write frequency `f_w` (Wave 20 observed) | GPU read reuse `R` | Target residency | Wave 20 trace evidence |
|---|---|---|---|---|---|
| A. Persistent GPU written placed buffers (`place_buffer`, UAV/AS spans) | `115 KiB` to `174 MiB` (`186.6 MiB` in first 8 logged buffers: `174.00 MiB` at `0x181cc2a0000`, `10.49 MiB` at `0xef370000`, `900 KiB`, `450 KiB`, `281 KiB`, `240 KiB`, `124 KiB`, `115 KiB`) | Near zero after initialization (GPU writes every frame) | Very high (`R >> 10`, plus GPU write authority) | **On card (L1, Class `P3`)** | `title_mem::place_buffer` logs (`L107..L1071`) plus repeated root view binds inside `0x181cc2a0000`. Must stay in L1 (never discarded) because GPU writes do not return to `xmem`. |
| B. Write once / static read many buffers (vertex, index, meshlet, instance, static SRV/CBV tables) | `>= 64 KiB` contiguous runs (`185 MiB` early physical maps: `47`, `100`, `10`, `1`, `27 MiB`, `on_demand` commits `64 KiB` to `4 MiB`, static slice of `6,236 MiB` pool and `902.3 MiB` scatter maps) | `1` burst write at load, then `0` writes for `>= 3 s` (`syncs 15800..16200` dropped to `590` pages / `36 MiB` per 100 syncs and `+304 MiB` mirror growth across `140 s`) | High (`R >> 2`, read every frame across draws) | **On card (L1, Class `P2`)**, promoted after `3 s` + `60` presents clean | In `t = 125..265 s`, `copy_runs` fell below `4 ms` per 100 syncs. Static buffers deliver the bulk of on card GPU read speedup once separated from `P0` textures. |
| C. Sub 64 KiB static data sharing a hot 64 KiB page | `< 64 KiB` mixed inside an active `64 KiB` page | Page level `f_w >= 1` write per `3 s` | Mixed | **Host RAM (L0, Class `P1`)** | D3D12 tile granularity is `64 KiB` (`title_mirror::PAGE`). Promoting a mixed page drags hot CPU writes through `StagingRing`. Keep in L0 until the full `64 KiB` page is write clean for `3 s`. |
| D. Hot CPU rewritten streaming and video planes | `64 KiB` to `~36 MiB` active working set (including `8,294,400` byte `3840x2160` R8 video planes at `0x17c9e7c0000` and `11` to `12` textures / `36 MiB` per detile flush) | Very high: `>= 3` faults in `250 ms` (`HOT_FAULTS`) or rewritten every frame | `R approx 1` (read once per frame by `detile.hlsl` or single draw) | **Host RAM (L0, Class `P1`)** | **Largest copy volume in Wave 20**: syncs `4700..13700` copied `2,021,872` pages (`126,367.0 MiB` = `123.41 GiB`, **`80.3%` of all copied bytes**) at `222.2` pages/sync (peak `53,756` pages / `3,359 MiB` in 100 syncs at sync `5600`) with `0` tile range updates in `87` of `91` batches. L1 wasted `123.41 GiB` of staging copies for `R = 1`. |
| E. One shot tiled texture source pages (`detile` inputs) | `4 KiB` (`16x4` LUTs) to `12.0 MiB` (`512x512x32` arrays, `11.2 MiB` `4096x4096` BC textures), **`~4.5` to `5.0 GiB` aggregate** | Written once via CPU or DirectStorage during world load (`t = 30..90 s`), then `0` writes | `R = 1` (read once by `detile.hlsl` into PC texture in `view_heap` or `CommittedSampled`) | **Transient L0 (`<= 512 MiB`), then discard to `Target::Null` (Class `P0`)** | During `t = 30..90 s`, `OwnerTitleMirrorMiB` grew `+4,688 MiB` alongside `OwnerViewHeapsMiB` (`+4,592 MiB`) across `187,705` detiled textures (`41,566+` view textures) and `49,517` tile range updates. L1 duplicated `~4.5` to `5.0 GiB` of VRAM for zero GPU reuse. |
| F. Per command allocator transient upload memory (`AllocateUploadMemory`) | `16 B` to `1 MiB` chunks (`FIRST_CHUNK_BYTES = 64 KiB`, `CHUNK_BYTES = 1 MiB` in `upload_mem.rs:24-26`) | Written once per command list recording, freed on allocator reset (`5,000+` deferred releases) | `R approx 1` (single submission lifetime) | **Host RAM (`D3D12_HEAP_TYPE_UPLOAD`, unchanged)** | `OwnerUploadMiB` averaged `50.9 MiB` and peaked at `106.1 MiB` (`70.8 MiB` at end). Already optimal in upload memory. |

### Comparison with CPU buffer copy hashing in the public speed fork

The public speed fork maintains on card buffer copies and hashes bytes on the CPU to detect changes. In Decant:

* **Full buffer hashing is unnecessary for static/clean pages (`Category B`)**: `decant-xmem` write tracking (`PAGE_READONLY` faults in `crates/decant-xmem/src/track.rs`, plus guard rest coalescing) already guarantees `0` bytes hashed and `0` bytes copied for clean pages.
* **Targeted `4 KiB` sub page compare applies only to held `P1` pages (`Category D`)**: While a hot or I/O held page remains unprotected during `HOT_HOLD` (`250 ms`) or `IO_HOLD` (`2,000 ms`), `take_dirty` returns it on every sync (producing the `123.41 GiB` copy volume in syncs `4700..13700`). Comparing `4 KiB` sub blocks against the L0 mirror before copying skips unchanged sub pages on held pages without touching clean pages.

---

## 5. Interaction with Wave 20 R1 (`d78d0a2`) and `DECANT_D3D12X_MIRROR_POOL`

Wave 20 R1 (`d78d0a2`) introduced the exact DMA upload infrastructure needed for L1 heaps (`StagingRing`, `UploadRun`, `CopyBufferRegion` on `map_queue`, and `Heap::ptr == 0` for L1 heaps in `crates/decant-d3d12x/src/title_mem.rs:538-742`). The budget and tiering design builds directly on R1 with minimal structural changes:

### Required extensions to Wave 20 R1 (`d78d0a2`)

1. **Per heap pool and class tag instead of a single device wide `State::pool`**:
   * Replace the single `use_staging = self.pool == MirrorPool::L1` check in `State::plan_runs` (`title_mem.rs:1349`) with a per heap check `self.heaps[heap].pool == MirrorPool::L1`.
   * Dirty pages targeting an L0 heap copy directly via `CopyPool` to `heap.ptr` (zero `StagingRing` traffic). Dirty pages targeting an L1 heap use R1's `StagingRing` + `CopyBufferRegion` unchanged.
2. **Class segregated tile allocation and reclamation in `title_mirror.rs`**:
   * Replace monotonic `next_tile` (`title_mirror.rs:160,177`) with per class free lists (`L0-P0`, `L0-P1`, `L0-P2`, `L1-P2`, `L1-P3`) and a 3 state page table (`Untouched`, `Resident(tile)`, `Evicted`).
   * Add `Mirror::remap_tile(page, new_tile)` and `Mirror::evict_to_null(page)` emitting `Target::Heap(new_tile)` or `Target::Null` into `Mirror::pending`, reusing the existing `drain_updates` and `UpdateTileMappings` pipeline in `title_mem.rs:1298-1322`.
   * Record `place_buffer` (`title_mem.rs:1795`) tile ranges so remapping a `P2`/`P3` page also updates any placed reserved buffer referencing it (or marks `place_buffer` pages directly as `P3` pinned in `L1-P3` heaps).
3. **Accounting and live budget fixes**:
   * Track `StagingRing` (`64 MiB`, or `32 MiB` on RAM constrained hosts) in `mem_owners` (`Kind::UploadChunk`, `Segment::System`).
   * Charge `Kind::MirrorTile` in `title_mirror.rs:181-186` to the tile's actual `Segment` (`Local` vs `System`), matching `TitleMirrorHeap` in `title_mem.rs:1094`.
   * Query live `IDXGIAdapter3::QueryVideoMemoryInfo` (`Budget` and `CurrentUsage`) inside `title_mem` and `views_at` via a shared budget snapshot rather than static `dedicated_vram()` (`title_mem.rs:567`) and `cap_from` (`views_at.rs:193`).

### Exact `DECANT_D3D12X_MIRROR_POOL` selector switch semantics

`parse_mirror_pool` (`title_mem.rs:555-565`) currently accepts `"l0"`, `"l1"`, or unset (auto: `dedicated_vram >= 16 GiB`). The switch retains full backward compatibility with refined behavior:

| `DECANT_D3D12X_MIRROR_POOL` value | Mode | Behavior on 8 / 12 / 16 / 24 GiB cards | Interaction with R1 `StagingRing` and safety rules |
|---|---|---|---|
| Unset or `auto` (default) | **Budget driven hybrid L0 + L1** | Queries live `QueryVideoMemoryInfo().Budget`. 8 GiB (`B < 10,240 MiB`): `M_cap = 0`, pure L0. 12 GiB: `M_cap = 1,024 MiB`. 16 GiB: `M_cap = 2,048 MiB`. 24 GiB: `M_cap = 4,096 MiB`. `P0` stays transient L0 then discards to `Null`, `P1` stays in L0, `P3` starts in L1, `P2` promotes to L1 when `U_eff <= LWM_total`. | Allocates `StagingRing` only when `M_cap > 0` (`64 MiB`, or `32 MiB` when `TotalPhys <= 32 GiB` and `AvailPhys < 4,096 MiB`). Full VRAM hysteresis (`HWM_total / LWM_total`) and `L0_cap` host RAM cap active. |
| `l0` | **Strict L0 only** | Forces `M_cap = 0` on all cards (including 16 and 24 GiB). No L1 mirror heaps are created. | Does **not** allocate `StagingRing` (`0 MiB` upload ring). Still enables `P0` post detile discard and `L0_cap` host RAM cap on `<= 32 GiB` machines so `l0` avoids the pre R1 `24.82 GiB` working set bloat. |
| `l1` | **Prefer L1 with safety guardrails** | Enables L1 mirror even on 8 GiB cards and admits `P2` directly to L1 on first write (skipping the `3 s` stability delay, matching R1's eager L1 placement). | Allocates `StagingRing`. Still keeps `P0` disposable, demotes `P1` hot pages to L0, and enforces the global `HWM_total / LWM_total` hysteresis safety valve so an explicit `l1` override cannot trigger `0x8007000E` OOM. |

Optional diagnostic overrides follow existing `DECANT_D3D12X_*_MIB` naming:

* `DECANT_D3D12X_MIRROR_L1_MIB=<n>`: Overrides `M_cap` directly in MiB (`0` disables L1, useful for A/B sweeps or reproducing R1's unbounded L1 when set to a high diagnostic cap).
* `DECANT_D3D12X_MIRROR_L0_MIB=<n>`: Overrides `L0_cap` in MiB for testing `<= 32 GiB` host RAM policies on larger workstations.
* `DECANT_D3D12X_VIEW_TEX_MIB=<n>` and `DECANT_D3D12X_VIEW_TEX_BUDGET_MIB=<n>`: Remain honored as upper bounds on `V_cap`, while always respecting the shared `HWM_total / LWM_total` hysteresis state machine and `MIN_IDLE_PRESENTS` / `MIN_IDLE_MS` guards.

## Recommended implementation sequence

1. **P0 transient L0 separation and post detile discard** (`title_mem.rs`, `title_mirror.rs`, `detile.rs`): Stops duplicating `~4.5` to `5.0 GiB` of tiled texture sources in L1 VRAM and host RAM. Immediate largest reduction in both `VramLocalMiB` and `PrivateCommitMiB`.
2. **Shared live `QueryVideoMemoryInfo` + `GlobalMemoryStatusEx` budget controller and per heap L0/L1 routing** (`title_mem.rs`, `views_at.rs`, `view_heap.rs`, `mem_owners.rs`): Replaces static `dedicated_vram >= 16 GiB` and `dedicated - 2 GiB` with the exact 8/12/16/24 GiB rules, `U_eff` retirement accounting, and `auto` hybrid mode.
3. **Hysteresis L1 to L0 demotion, `P2` promotion gate, and `P1` L0 routing** (`title_mem.rs`, `title_mirror.rs`, `crates/decant-xmem/src/track.rs`): Enforces `HWM_total / LWM_total`, `T_min_res`, `T_freeze`, exponential `T_cool`, and routes hot `xmem` pages directly to L0 (eliminating `123.41 GiB` of redundant `StagingRing` copies in video/streaming phases).
