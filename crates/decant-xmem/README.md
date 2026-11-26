# Physical page tracking

The engine separates its address space state from GPU dirty tracking. The dirty mutex owns the tracker, a writable address index and the pool view metadata needed to query commitment. `take_dirty` and tracked write faults use this mutex without acquiring the address space mutex. The exception handler tries a tracked write before the general commitment path, and retries the tracked write path after commitment when another thread commits and protects the page in between. Tracked write faults skip the `VirtualQuery` commitment check once the tracker already knows the pool page is committed, and `prepare_write_until` commits pages under the address space mutex before acquiring the dirty mutex.

Mapping changes acquire the address space mutex before the dirty mutex. They keep both through Windows mapping calls, alias updates and protection changes. Dirty tracking never acquires the address space mutex while holding its mutex. Address space inspection, including the full commitment walk in `footprint`, does not acquire the dirty mutex.

Every writable alias of a physical page shares its dirty state. Tracker transitions and Windows protection changes remain serialized. Read only aliases keep their protection. Hot page holds, kernel write holds, cached commitment and adjacent protection run merging keep their existing rules.

Issue #10 reported 781 ms waiting for the engine mutex and 53 ms inspecting pages per 200 calls during streaming. Those figures describe the original report. This change has no streaming workload measurement. It removes the address space mutex dependency from dirty takes and tracked write faults. Mapping transactions still serialize with dirty tracking, so they can still cause waits.

The contention regression test holds the address space mutex while another thread takes dirty pages, writes through a protected alias and takes the page again. All three operations must finish before the mutex is released. The stress test races writes through two aliases with repeated mapping and unmapping of a separate page and capped dirty takes. Existing tracker tests cover holds, cached commitment and protection run merging.

The synchronization design was reviewed against [QEMU memory synchronization documentation](https://www.qemu.org/docs/master/devel/memory.html). No external source code was copied.
