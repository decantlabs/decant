# Descriptor Tracking and Comparison Design Note

This note evaluates whether descriptor optimization techniques reported in the public speed fork (which cut frame time from 280 ms to 124 ms by byte-comparing descriptor records on worker threads, skipping empty slots, and deferring resolution to the draws that use them) apply to decant.

The short answer is no. Decant intercepts view creation and descriptor copy calls directly and stamps 24-byte tags into title heap memory rather than scanning raw descriptor heaps. Active CPU time spent in decant's descriptor tracking and descriptor copy paths is under 0.02 ms/frame. The real descriptor-related costs in decant are eager view-texture creation and quadratic texture eviction (`views_at:create_srv_at` and `views_at:trim`), draw-time mutex contention inside `list_mirror::readable`, and exclusive lock contention in `title_mem::translate` during buffer view creation.

## 1. Measured Per-Frame Cost

### Measurement Environment and Commits

* Machine: Windows host, 24 GiB dedicated VRAM (24,326 MiB reported by DXGI), 32 GiB host RAM.
* Primary traced baseline: commit `8efa0d2`, 300 s capture, 3,907 frames, 265,360.6 ms traced wall time, mean 67.860 ms/frame.
* Supporting per-thread trace, nested sub-span breakdown, runtime counters, and PDB-symbolized 20 ms stack-sample profile: commit `ca3fcd8` on the same machine (300 s capture, 4,210 frames, 227,968.0 ms traced wall time, warm shader cache). `git diff 8efa0d2..ca3fcd8` confirms zero byte changes in any descriptor, view, or paging module (`slot_tags.rs`, `title_heap.rs`, `view_heap.rs`, `views_at.rs`, `view_shadow.rs`, `view_window.rs`, `cbv_copies.rs`, `zero_null.rs`, `user_mirror.rs`, `proxy.rs`, and `proxy/*` descriptor paths).

### Instrumented Descriptor and View Spans

Only two `decant_trace` spans exist across decant's descriptor and view subsystems at commit `8efa0d2`:

1. `views_at:create_srv_at` (`crates/decant-d3d12x/src/proxy.rs:2808`), covering texture creation misses inside `view_texture`.
2. `views_at:trim` (`crates/decant-d3d12x/src/proxy.rs:2936`), called from `proxy.rs:2894` and strictly nested inside `views_at:create_srv_at` (14 of 14 calls).

| Metric | Commit `8efa0d2` (3,907 frames) | Commit `ca3fcd8` (4,210 frames) |
| --- | --- | --- |
| `views_at:create_srv_at` total | 3,638.076 ms (45,447 calls) | 3,653.177 ms (45,758 calls) |
| `views_at:create_srv_at` per frame (all frames) | 0.931 ms/frame (1.4% of 67.860 ms) | 0.868 ms/frame |
| `views_at:create_srv_at` in loading (30 to 120 s) | 3,320.899 ms / 553 frames = 6.005 ms/frame | 3,538.377 ms / 586 frames = 6.038 ms/frame |
| `views_at:create_srv_at` outside loading | 317.177 ms / 3,354 frames (< 0.1 ms/frame) | 114.800 ms / 3,624 frames (< 0.04 ms/frame) |
| `views_at:create_srv_at` max single call | 587.703 ms | 677.269 ms |
| Nested `views_at:trim` total | not split in wave summary | 1,275.536 ms (14 calls, max 677.214 ms) |
| Nested `views_at:trim` per frame (all / loading) | not split in wave summary | 0.303 ms/frame (2.177 ms/frame in loading) |
| Exclusive non-trim `create_srv_at` (all / loading) | not split in wave summary | 0.565 ms/frame (3.862 ms/frame in loading) |

Per-thread distribution of `views_at:create_srv_at` (and its nested `views_at:trim`) from the full `ca3fcd8` trace:

| Thread | `create_srv_at` total | Calls | Per frame (all) | Per frame (30 to 120 s) | Nested `trim` total (calls, max) |
| --- | --- | --- | --- | --- | --- |
| Title main thread (`tid=1`) | 2,602.029 ms (71.2%) | 28,422 | 0.618 ms/frame | 4.324 ms/frame | 752.390 ms (5 calls, max 677.214 ms) |
| Streaming worker (`tid=78`) | 1,048.238 ms (28.7%) | 17,322 | 0.249 ms/frame | 1.714 ms/frame | 523.138 ms (8 calls, max 268.752 ms) |
| Five other workers (`tid=33, 40, 94, 27, 29`) | 2.910 ms (0.1%) | 14 | 0.001 ms/frame | 0.000 ms/frame | 0.008 ms (1 call) |
| Queue submit and present thread (`tid=34`) | 0.000 ms (0.0%) | 0 | 0.000 ms/frame | 0.000 ms/frame | 0.000 ms (0 calls) |
| Total (all threads) | 3,653.177 ms (100%) | 45,758 | 0.868 ms/frame | 6.038 ms/frame | 1,275.536 ms (14 calls, max 677.214 ms) |

The workload is strongly bursty. Only 185 of 4,211 frames (4.4%) recorded any `views_at:create_srv_at` event (all-frame p50 = 0.000 ms, p95 = 0.000 ms, p99 = 14.779 ms, max = 689.030 ms). Among the 185 active frames, p50 is 4.865 ms and p95 is 69.137 ms.

For context at commit `8efa0d2`, dominant non-descriptor spans in the same run were `direct_pso:compile` (1,516,192 ms off-thread, 388.1 ms/frame), `title_mem:sync` (53,941 ms on the submission thread, 13.81 ms/frame, including 37,038 ms in `take_dirty`), `detile:flush` (7,388 ms, 1.89 ms/frame), and `present:show` (5,345 ms, 1.37 ms/frame).

### Uninstrumented Sub-Paths and Empirical Bounds

All descriptor tracking, tagging, heap paging, buffer-window creation, and descriptor copy paths lack `decant_trace::span` instrumentation at commit `8efa0d2`:

* Device view creation hooks: `dev_create_cbv`, `dev_create_srv`, `dev_create_uav`, `dev_create_rtv`, `dev_create_dsv`, `dev_create_sampler`, the `dev_create_view_at` texture-reuse and buffer paths, and `view_window::prepare` (over 8,192 reserved buffer windows created).
* Descriptor copy hooks: `dev_copy_descriptors` (`copy_title_tags`, `copy_shadow_rows`, `note_copies`) and `dev_copy_descriptors_simple`.
* Slot tagging, heap paging, and shadow tracking: `title_heap::{translate_cpu, tag_slot, copy_tags, table, table_copies, written, flush, move_table}`, `slot_tags::{made, overwritten, state, taken}`, `view_shadow::{view_made, owner_made, copied, referenced}`, `cbv_copies::{recorded, conflicts, submitted}`, and `zero_null::{due, mark, forget}`.
* Draw-time resolution: `list_mirror::{prepare, sync_tables, null_zero_slots, write_null, copied_cbv, move_table, restore_moved}` and `user_mirror::*`.

The PDB-symbolized 20 ms stack sampler from the 300 s `ca3fcd8` run (1,762,926 total samples across 11 windows, 59,222 active samples outside `ntdll` and `win32u`) bounds these uninstrumented paths directly:

* Active CPU in uninstrumented descriptor tracking and copies is near zero. Zero descriptor tracking or copy functions appear in any window's top 20 active symbols. Across the entire 300 s capture, only 3 of 59,222 active samples (0.005%, roughly 60 ms total or under 0.02 ms/frame) landed inside active descriptor tracking or copy code: 1 sample in `view_shadow::copied` via `title_heap::flush`, 1 sample in `title_heap::written` via `list_mirror::prepare`, and 1 sample in `list_set_compute_root_table`.
* Runtime heap counters explain why active CPU stays low. The shader-visible view heap has 1,245,184 console slots (38 MiB of title memory) paged onto 1,000,000 host slots (240 host pages of 4,096 slots plus a 16,384-slot moved-table ring). By present 4,096, the pager had written 249 console pages (218 table pages, 31 direct-index pages) with 64 page evictions and 0 refusals. `slot_tags` recorded 15,810 `Sync::Keep` hits, 5,618 `Sync::Again` CBV copies, 79 lost tags, and 0 `move_table` hits. `zero_slot_null` recorded 0 writes.
* Hidden lock-wait stalls on uninstrumented descriptor paths are much larger than active CPU. First, 122 stack samples on the title main thread blocked inside `list_mirror::prepare` calling `sync_tables` calling `readable` calling `title_mem::writable` calling `xmem::Engine::writable` (81 samples or 8.1% of main-thread wall time in the 30 to 60 s window, plus 41 samples or 4.2% in the 60 to 90 s window), representing roughly 3.6 s of main-thread stalls (about 6.1 ms/frame during 30 to 90 s loading, or 0.86 ms/frame averaged over 4,210 frames).
* Second, buffer view VA translation (`dev_create_cbv`, `dev_create_view_at` buffer fallback, and `view_window::prepare`) shares `title_mem::STATE` and `title_mem::DRAIN_LOCK` with non-descriptor callers (root views, vertex and index buffers, and `title_mem:sync`). At commit `8efa0d2`, `title_mem::STATE` accumulated 3,704.466 ms across 415,312 locks (0.948 ms/frame) and `title_mem::DRAIN_LOCK` accumulated 3,386.686 ms across 1,360 locks (0.867 ms/frame) across all threads. In the per-thread `ca3fcd8` trace, streaming worker `tid=78` (which runs zero `title_mem:sync`, `present:show`, or PSO compilation work) incurred 188,398 `STATE` waits totaling 700.4 ms (100% outside `views_at:create_srv_at`) plus 206 `DRAIN_LOCK` waits totaling 293.5 ms (0.236 ms/frame off-thread). Title main thread `tid=1` incurred 72,390 `STATE` waits (1,597.9 ms) and 391 `DRAIN_LOCK` waits (790.7 ms), split between descriptor VA translation and non-descriptor callers.

Putting instrumented spans and uninstrumented lock waits together, decant spends roughly 1.5 to 2.0 ms/frame on the title main thread and roughly 2.0 to 3.0 ms/frame across all threads on descriptor and view-at-VA work averaged over 300 s (peaking around 12 to 15 ms/frame during 30 to 120 s asset loading, and falling below 0.1 ms/frame during steady-state frames). Active CPU for descriptor tracking and descriptor copies accounts for less than 0.02 ms/frame of that total.

## 2. Three Largest Costs

### 1. `CreateSrvAt` View-Texture Creation and Quadratic `trim_view_textures` Eviction

* Files and lines:
  * `crates/decant-d3d12x/src/proxy.rs:2808-2907` (`view_texture`, span `views_at:create_srv_at`)
  * `crates/decant-d3d12x/src/view_heap.rs:334-411` (`view_heap::place`), `crates/decant-d3d12x/src/view_heap.rs:414-422` (`retire_resource`), and `crates/decant-d3d12x/src/view_heap.rs:426-439` (`released`)
  * `crates/decant-d3d12x/src/detile.rs:1308-1328` (`detile::register`) and `crates/decant-d3d12x/src/detile.rs:1363-1370` (`detile::forget`)
  * `crates/decant-d3d12x/src/proxy.rs:2894, 2930-3017` (`trim_view_textures`, span `views_at:trim`)
  * `crates/decant-d3d12x/src/views_at.rs:63-65` (`VaTable::forget`) and `crates/decant-d3d12x/src/views_at.rs:426-431` (per-texture `retire_resource` loop)
  * `crates/decant-d3d12x/src/view_shadow.rs:254-267` (`view_shadow::referenced`)
* Measured cost: 3,638.076 ms across 45,447 calls at `8efa0d2` (0.931 ms/frame overall, 6.005 ms/frame during loading, max 587.703 ms). At `ca3fcd8`, 3,653.177 ms total, split into 2,377.641 ms of texture creation (`CreatePlacedResource` / `CreateCommittedResource`, `detile::register`, and lock traffic) plus 1,275.536 ms across 14 `views_at:trim` passes (max 677.214 ms on the title main thread).
* Root cause of spikes: When `trim_view_textures` evicts `K` textures (the runtime log recorded passes freeing 10,462, 3,435, and 2,850 textures while roughly 27,500 view textures were live), `proxy.rs:2972-2974` calls `views_at::forget_resource(e.res)` once per texture. `VaTable::forget` (`views_at.rs:63-65`) runs `self.by_start.retain(|_, e| e.host_res != host_res)`, scanning the entire `BTreeMap` in `O(N)` per evicted texture under `VA_TABLE`'s mutex. A single 10,462-texture trim visits roughly 288 million `BTreeMap` entries. In the same pass, `proxy.rs:2973`, `views_at.rs:427-431`, and `proxy.rs:2998-3000` acquire `detile::REGISTRY`, `view_heap::ALLOCS`, and `view_heap::HEAPS` once per evicted texture, after `view_shadow::referenced` (`proxy.rs:2959`, `view_shadow.rs:254-267`) scans roughly 2.28 million shadow owner slots across live heaps.

### 2. Draw-Time `sync_tables` Contending on the Global `xmem` Mutex Inside `readable`

* Files and lines:
  * `crates/decant-d3d12x/src/proxy/list_mirror.rs:546-606` (`sync_tables`, `Sync::Again` guard at `proxy/list_mirror.rs:570`)
  * `crates/decant-d3d12x/src/proxy/list_mirror.rs:611-615` (`readable`)
  * `crates/decant-d3d12x/src/title_mem.rs:1829-1832` (`title_mem::writable`)
  * `crates/decant-xmem/src/engine.rs:2524-2535` (`Engine::writable`, acquiring `self.lock()`)
* Measured cost: uninstrumented by `decant_trace`, exposed by stack sampling at 122 samples on the title main thread during 30 to 90 s loading (roughly 3.6 s total, about 6.1 ms/frame during loading, 0.86 ms/frame averaged over all frames).
* Root cause: On every `Sync::Again((va, size))` hit (5,618 occurrences in the logged run), `proxy/list_mirror.rs:614` evaluates `crate::title_mem::writable(va, size) || (in_heap(va) && in_heap(va + size.max(1) - 1))`. Because `title_mem::writable` is evaluated first, every check acquires `xmem`'s global engine mutex even when the constant buffer lives inside title heap memory (as all logged unaligned constant buffers do, where the right-hand `in_heap` predicate succeeds locally via `title_heap::translate_cpu`). Whenever the submission thread holds the `xmem` mutex inside `title_mem:take_dirty`, the title main thread stalls mid-draw.

### 3. Exclusive `title_mem::STATE` and `DRAIN_LOCK` Waits on Buffer View VA Translation and Window Creation

* Files and lines:
  * `crates/decant-d3d12x/src/proxy.rs:1172-1174, 1180-1226` (`pc_gpu_address`, `dev_create_cbv`)
  * `crates/decant-d3d12x/src/proxy.rs:2637-2656, 2698` (`dev_create_view_at` buffer fallback and `view_window::prepare` call)
  * `crates/decant-d3d12x/src/view_window.rs:73-168` (`view_window::prepare`, calling `title_mem::tiles_of` at `view_window.rs:127, 149` and `title_mem::place_buffer` at `view_window.rs:147`)
  * `crates/decant-d3d12x/src/title_mem.rs:1482-1553` (`with_state`), `crates/decant-d3d12x/src/title_mem.rs:1636-1638, 1667-1683, 1700-1708` (`translate`, `translate_span`, `span_of_with`), `crates/decant-d3d12x/src/title_mem.rs:1795-1825` (`place_buffer`), and `crates/decant-d3d12x/src/title_mem.rs:2040-2043` (`tiles_of`)
  * `crates/decant-d3d12x/src/title_mirror.rs:195-211, 412-426` (`give_span` and `wants_use`)
* Measured cost: uninstrumented as a descriptor span, visible in lock traces. At `8efa0d2`, `title_mem::STATE` totaled 3,704.5 ms across 415,312 locks and `title_mem::DRAIN_LOCK` totaled 3,386.7 ms across 1,360 locks across all callers. On streaming worker `tid=78` (pure view creation, zero sync/present/PSO work), 188,398 `STATE` waits cost 700.4 ms and 206 `DRAIN_LOCK` waits cost 293.5 ms (0.236 ms/frame off-thread). Title main thread `tid=1` added 1,597.9 ms of `STATE` waits and 790.7 ms of `DRAIN_LOCK` waits across descriptor and non-descriptor callers.
* Root cause: Unlike `title_mem::lookup` (`title_mem.rs:1642-1644, 1703-1705`), which takes a shared read lock via `with_state_read` (`title_mem.rs:1557-1572`), `title_mem::translate` (`title_mem.rs:1707`) and `title_mem::tiles_of` (`title_mem.rs:2042`) unconditionally acquire exclusive `STATE.write()` (and `DRAIN_LOCK` on any journal change). Inside `translate_span`, `give_span` (`title_mirror.rs:195-211`) also allocates a temporary `Vec` on every call even when every page in the span already has a mirror tile. Meanwhile, `view_window::prepare` created over 8,192 distant reserved buffers, calling `tiles_of` twice and `place_buffer` once per window creation (and `tiles_of` once on every cache hit).

Other descriptor tracking paths inspected (`CopyDescriptorsSimple` issued one slot at a time in `proxy/list_mirror.rs:594-600`, per-slot tag and shadow loops in `proxy.rs:1612-1638`, `null_zero_slots` called even when `hosts` is empty at `proxy/list_mirror.rs:549-551`, and `(min_lo, max_hi)` dirty-span coalescing per 4,096-slot page in `title_heap.rs:689-694`) remain below 0.02 ms/frame in current traces and do not enter the top three costs.

## 3. Evaluation of Public Speed Fork Techniques

| Public speed fork technique | Rating | Summary |
| --- | --- | --- |
| Byte-comparing descriptor records on worker threads | **does not transfer** | Decant uses API-level 24-byte slot tags and never scans raw descriptor heaps. Active tracking CPU is under 0.02 ms/frame. |
| Skipping empty slots | **already present** | `table_copies` skips all untagged slots, `slot_tags::sync` skips unchanged held views, and `zero_null` writes only pipeline-declared null offsets once. |
| Deferring resolution to the draws that use them | **already present** (eager view-at-VA creation **does not transfer**) | Table copies, unaligned CBV materialization, and paged-heap flushes already wait until draw or dispatch. Direct heap indexing (`ResourceDescriptorHeap[i]`) prevents deferring `CreateSrvAt` texture or buffer-window creation. |

### Byte-Comparing Descriptor Records on Worker Threads: **does not transfer**

The public speed fork reduced a 280 ms/frame bottleneck because its architecture scanned raw console descriptor memory to discover changes made by title code. Decant does not scan descriptor heaps for diffs. Every D3D12X view creation hook (`dev_create_cbv`, `dev_create_srv`, `dev_create_uav`, `dev_create_view_at`) writes a 24-byte tag `[TAG, word, TAG ^ word]` into the title's heap slot (`crates/decant-d3d12x/src/title_heap.rs:154-186`) carrying the host slot handle and a 16-bit per-slot generation counter stored in 64-way sharded maps (`crates/decant-d3d12x/src/slot_tags.rs:127-132, 160-161, 188-227`). API copies propagate those tags (`title_heap::copy_tags`, `title_heap.rs:193-212`), and title memory copies carry the 24-byte tag automatically alongside the console descriptor bytes. At draw time, `title_heap::table_copies` (`title_heap.rs:291-306`) reads at most `TABLE_WINDOW = 64` slots per newly bound table handle and checks the 24-byte magic/XOR word rather than byte-comparing the 38 MiB (1,245,184-slot) heap.

Because total active CPU across all descriptor tracking, tagging, and copying is 3 of 59,222 active samples (under 0.02 ms/frame), dispatching descriptor checks to worker threads cannot recover meaningful frame time. Worker offloading would add inter-thread synchronization overhead larger than the work itself and would break the strict per-command-list draw ordering required by `cbv_copies` (`crates/decant-d3d12x/src/cbv_copies.rs`) and `move_table` (`crates/decant-d3d12x/src/proxy/list_mirror.rs:576-604`).

### Skipping Empty Slots: **already present**

Decant already skips empty and unchanged slots across every active descriptor path:

* `title_heap::table_copies` (`crates/decant-d3d12x/src/title_heap.rs:165-167, 291-306`) validates `words[0] == TAG && words[2] == TAG ^ words[1] && words[1] != 0` and immediately ignores every empty, zeroed, or untagged slot in the 64-slot window.
* `slot_tags::sync` (`crates/decant-d3d12x/src/slot_tags.rs:108-116`) returns `Sync::Keep` when a table slot already holds the tagged view (skipping 15,810 slots by present 4,096), and `sync_tables` skips in-place views where `dst == src` (`crates/decant-d3d12x/src/proxy/list_mirror.rs:589-591`).
* `zero_null` (`crates/decant-d3d12x/src/zero_null.rs:167-228`, `crates/decant-d3d12x/src/proxy/list_mirror.rs:421-495`) checks only the exact table offsets declared by the bound pipeline's shaders and tracks written null slots in `ONCE` so each slot is written at most once until overwritten (0 writes occurred in the measured run).

One minor over-copy remains in the paged view heap: `Pager::dirty` (`crates/decant-d3d12x/src/title_heap.rs:689-694`) coalesces dirty slots within each 4,096-slot host page into a single `(min_lo, max_hi)` bounding interval, so `take_flush` (`title_heap.rs:700-709`) can copy untouched interior slots between two sparse writes on the same page. In practice, `title_heap::flush` and `view_shadow::copied` combined accounted for 1 of 59,222 active samples, so tracking per-slot bitmasks inside pages would not measurably change frame time.

### Deferring Resolution to the Draws That Use Them: **already present**

Descriptor table resolution in decant is already deferred until draw or dispatch time:

* Binding a table on a command list only appends the handle to `Shadow::unsynced` (`crates/decant-d3d12x/src/user_mirror.rs:252-260`). Redundant table binds with no intervening draw incur no tag scanning or descriptor copying.
* `list_mirror::prepare` (`crates/decant-d3d12x/src/proxy/list_mirror.rs:99-110`) runs immediately before draw or dispatch calls. It drains `take_unsynced` (`user_mirror.rs:280-282`), executes `sync_tables` (`proxy/list_mirror.rs:546-606`) to copy tagged views and materialize unaligned constant buffers (`copied_cbv`) or moved tables (`move_table`), and flushes paged host heap dirty ranges (`heap_pages::flush`).

The only eager resource materialization paths that still execute during descriptor creation are `CreateSrvAt` texture creation (`view_texture`, `crates/decant-d3d12x/src/proxy.rs:2769-2907`) and distant buffer window creation (`view_window::prepare`, `crates/decant-d3d12x/src/view_window.rs:73-168`). Deferring those two paths to draw time **does not transfer** cleanly to decant because the title uses Shader Model 6.6 direct heap indexing (`CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED`, evidenced by 31 index-only pages written in the view-heap pager log). Shaders compute heap slot indices dynamically on the GPU (`ResourceDescriptorHeap[i]`), so the CPU command list recorder cannot know which directly indexed heap slots a draw will sample. Those descriptors must hold valid host views before any draw or dispatch can index them.

## 4. Ranked Changes

None of the public speed fork's 156 ms/frame descriptor-comparison speedup exists in decant. Rebuilding decant's descriptor layer around worker-thread heap scanning would regress performance and complexity.

However, the trace and stack profile identify three targeted, low-risk changes inside decant's existing descriptor and view-at-VA paths that remove real measured main-thread stalls and lock contention:

1. **Check `in_heap` before `title_mem::writable` in `list_mirror::readable`** (`crates/decant-d3d12x/src/proxy/list_mirror.rs:611-615`).
   * Change: Short-circuit `(in_heap(va) && in_heap(va + size.max(1) - 1)) || crate::title_mem::writable(va, size)` so heap-resident constant buffers skip `xmem::Engine::writable` (`crates/decant-xmem/src/engine.rs:2524-2535`).
   * Expected gain: Removes roughly 3.6 s of title main-thread stalls during 30 to 90 s loading (122 stack samples, about 6.1 ms/frame during loading and 0.86 ms/frame across the 300 s run) whenever `title_mem:take_dirty` holds the `xmem` mutex.
   * Risk: Very low. Both predicates are pure read-only queries with identical boolean disjunction semantics.

2. **Batch `O(K * N)` teardown inside `trim_view_textures`** (`crates/decant-d3d12x/src/proxy.rs:2972-3002`, `crates/decant-d3d12x/src/views_at.rs:63-65, 426-431`, `crates/decant-d3d12x/src/detile.rs:1363-1370`, `crates/decant-d3d12x/src/view_heap.rs:414-439`).
   * Change: Replace per-texture `VaTable::forget` (`self.by_start.retain(...)` in `views_at.rs:64`) with either a reverse `host_res -> va` map or a single-pass batch `forget_resources(&HashSet<usize>)`, and batch `detile::forget`, `view_heap::retire_resource`, and `view_heap::released` so each registry or pool lock is taken once per trim pass rather than once per evicted texture.
   * Expected gain: Cuts `views_at:trim` from 1,275.5 ms across 14 passes (max 677.2 ms single-call stall on the title main thread at `ca3fcd8`, 587.7 ms max at `8efa0d2`) down to single-digit milliseconds per pass (over 95% reduction in trim time, saving about 2.1 ms/frame during 30 to 120 s loading and removing half-second main-thread hitches).
   * Risk: Low. Mechanical batching preserves exact eviction order, `retire::release` GPU fencing, and `REUSE_PRESENTS` heap reclamation, and stays orthogonal to view-texture memory budget policies.

3. **Add a shared read-lock fast path to `title_mem::translate` and `title_mem::tiles_of`** (`crates/decant-d3d12x/src/title_mem.rs:1636-1708, 2040-2043`, `crates/decant-d3d12x/src/title_mirror.rs:195-211, 412-426`).
   * Change: Mirror `title_mem::lookup`'s existing `with_state_read` fast path (`title_mem.rs:1703-1705`) for `translate` and `tiles_of`. Under `STATE.read()`, verify `!mirror.wants_use(va, len.max(ADDRESS_REACH))` and verify without allocating that every mapped page in the reach span already has a mirror tile, returning the translated span (or cached tile list) directly. Fall back to exclusive `with_state` only when the journal sequence changed, on-demand pages must be requested, or a page needs its first mirror tile. Optionally add `decant_trace::span` markers on `sync_tables` and `view_window::prepare` to keep future regressions visible in traces.
   * Expected gain: Eliminates exclusive `STATE.write()` serialization and per-call `Vec` allocation in `give_span` across concurrent view creation on streaming workers and the title main thread, recovering a large share of the 700.4 ms `STATE` wait on streaming worker `tid=78` (188,398 locks) and reducing writer contention on the title main thread's 1,597.9 ms `STATE` wait (estimated 0.3 to 0.8 ms/frame overall, higher during loading).
   * Risk: Low to moderate. Must fall back to exclusive `with_state` whenever `should_skip_drain` is false, `wants_use` is true, or any page in the span has `NO_TILE`, and must preserve `pending_in` freshness reporting. Existing unit tests in `title_mem.rs` already validate `STATE.read()` concurrency and journal-sequence skip logic.
