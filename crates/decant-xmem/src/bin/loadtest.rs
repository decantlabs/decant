//! `loadtest <xmem.dll>`: the xmem checks that were once the C program `loadtest.c`, with the same lines,
//! in the same order, and the same exit codes (0 all passed, 1 a check failed, 2 bad arguments
//! or a DLL that does not load). `gpu_diag` runs it from the build folder.

#[cfg(windows)]
#[path = "../testapi.rs"]
mod testapi;

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    let args = decant_testkit::args();
    if args.len() != 1 {
        return std::process::ExitCode::from(2);
    }
    let Ok(lib) = decant_testkit::Library::open(&args[0]) else {
        return std::process::ExitCode::from(2);
    };
    let mut checks = decant_testkit::Checks::new(62, "OK", "ECHEC");
    run::test_xmem(&lib, &mut checks);
    lib.free();
    std::process::ExitCode::from(u8::from(checks.failures() != 0))
}

#[cfg(not(windows))]
fn main() {
    eprintln!("loadtest runs on Windows only");
}

#[cfg(windows)]
mod run {
    use std::ffi::c_void;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU32, Ordering};

    use decant_testkit::{Checks, Library};

    use crate::testapi::{AllocFn, CURRENT_PROCESS, FreeFn, GRAPHICS, PAGE_64K, XMem, align, call_alloc, call_free, fill, poke, present};

    const MEM_COMMIT: u32 = 0x1000;

    /// Callers known so far pass stack buffers of 64 and 128 bytes: nothing may be written past them.
    fn caller_buffers(x: &XMem, c: &mut Checks) {
        let mut info = [0xCCu8; 64 + 32];
        let mut statistics = [0xCCu8; 128 + 32];
        let ok_query = x.query(CURRENT_PROCESS, info.as_ptr().cast(), &mut info);
        let ok_stats = x.stats(0x2000_0001, &mut statistics);
        let intact = info[64..].iter().all(|&b| b == 0xCC) && statistics[32..].iter().all(|&b| b == 0xCC);
        c.check(ok_query && ok_stats && intact, "XMemVirtualQuery/Statistics : rien d'ecrit apres 64/32 octets");
    }

    /// Nothing guarantees that the caller frees with the attributes it allocated with.
    fn mismatched_free(x: &XMem, c: &mut Checks) {
        let (before, _, _) = x.counters();
        let heap = x.alloc(1000, align(5));
        let gfx = x.alloc(300000, PAGE_64K | GRAPHICS);
        let page = x.alloc(4096, align(12) | GRAPHICS);
        c.check(!page.is_null() && (page as usize & 0xFFF) == 0, "XMemAlloc memoire graphique, alignement 2^12");
        x.free(heap, PAGE_64K | GRAPHICS);
        x.free(gfx, 0);
        x.free(page, align(5));
        let (current, _, _) = x.counters();
        c.check(!heap.is_null() && !gfx.is_null() && current == before, "XMemFree avec d'autres attributs qu'a l'allocation");
    }

    fn expected_alignment(log2: u32, page_size: u64, kind: u64) -> usize {
        let mut alignment = if log2 > 4 { 1usize << log2 } else { 16 };
        if kind != 0 || page_size != 0 {
            let page = if page_size != 0 { 0x10000 } else { 0x1000 };
            alignment = alignment.max(page);
        }
        alignment
    }

    /// Ticket 05: boundary sizes x alignments x types, everything freed, the same peak twice.
    fn boundaries(x: &XMem, c: &mut Checks) {
        const SIZES: [usize; 12] = [1, 15, 16, 17, 4095, 4096, 4097, 65535, 65536, 65537, 0xFFFFF, 0x100000];
        const LOGS: [u32; 6] = [0, 4, 5, 8, 12, 16];
        const COUNT: u64 = 2 * 2 * LOGS.len() as u64 * SIZES.len() as u64;
        let (base, _, base_count) = x.counters();
        let mut first_peak = 0;
        let mut aligned = true;
        for pass in 0..2 {
            let mut total = 0u64;
            let mut blocks = Vec::new();
            for kind in [0, GRAPHICS] {
                for page in [0, PAGE_64K] {
                    for log2 in LOGS {
                        for size in SIZES {
                            let expected = expected_alignment(log2, page, kind);
                            let attributes = kind | page | align(u64::from(log2));
                            let block = x.alloc(size, attributes);
                            aligned &= !block.is_null() && (block as usize & (expected - 1)) == 0;
                            if !block.is_null() {
                                poke(block, 0, 0x5A); // an access violation here is a failure
                                poke(block, size - 1, 0xA5);
                            }
                            blocks.push((block, attributes));
                            total += size as u64;
                        }
                    }
                }
            }
            let (current, peak, _) = x.counters();
            if pass == 0 {
                c.check(aligned, "boucle limites : 288 blocs alignes et ecrits aux deux bouts");
                c.check(current == base + total, "boucle limites : octets en cours = somme des tailles");
                first_peak = peak;
            }
            for (block, attributes) in blocks {
                x.free(block, attributes);
            }
        }
        let (current, peak, count) = x.counters();
        c.check(current == base && count == base_count + 2 * COUNT, "boucle limites : zero allocation vivante, compte exact");
        c.check(peak == first_peak, "boucle limites : pic identique sur deux passes");

        let huge = x.alloc(usize::MAX - 8, align(16));
        c.check(huge.is_null(), "XMemAlloc refuse une taille qui deborde");
    }

    static HOOK_ALLOCS: AtomicU32 = AtomicU32::new(0);
    static HOOK_FREES: AtomicU32 = AtomicU32::new(0);
    static DEFAULTS: OnceLock<(AllocFn, FreeFn)> = OnceLock::new();

    extern "system" fn counting_alloc(size: usize, attributes: u64) -> *mut c_void {
        HOOK_ALLOCS.fetch_add(1, Ordering::SeqCst);
        DEFAULTS.get().map_or(std::ptr::null_mut(), |d| call_alloc(d.0, size, attributes))
    }

    extern "system" fn counting_free(address: *mut c_void, attributes: u64) {
        HOOK_FREES.fetch_add(1, Ordering::SeqCst);
        if let Some(d) = DEFAULTS.get() {
            call_free(d.1, address, attributes);
        }
    }

    fn hooks(x: &XMem, c: &mut Checks) {
        let (Some(alloc), Some(free), true) = (x.default_alloc, x.default_free, x.has_hooks()) else {
            c.check(false, "XMemSetAllocationHooks et les fonctions par defaut sont exportees");
            return;
        };
        let _ = DEFAULTS.set((alloc, free));
        let (before, _, _) = x.counters();
        x.set_hooks(Some(counting_alloc), Some(counting_free));
        x.free(x.alloc(64, 0), 0);
        x.set_hooks(None, None);
        x.free(x.alloc(64, 0), 0);
        let (current, _, _) = x.counters();
        c.check(
            HOOK_ALLOCS.load(Ordering::SeqCst) == 1 && HOOK_FREES.load(Ordering::SeqCst) == 1 && current == before,
            "XMemSetAllocationHooks : crochets appeles, (NULL, NULL) retablit",
        );
    }

    fn churn(x: &XMem, failures: &AtomicU32) {
        for i in 0..2000usize {
            let size = 16 + (i * 37 % 5000);
            let attributes = if i & 3 == 3 { GRAPHICS } else { align(4 + (i % 9) as u64) };
            let p = x.alloc(size, attributes);
            if p.is_null() {
                failures.fetch_add(1, Ordering::SeqCst);
                continue;
            }
            poke(p, 0, i as u8);
            poke(p, size - 1, i as u8);
            x.free(p, attributes);
        }
    }

    /// Ticket 05, concurrency: four threads allocate and free at the same time.
    fn concurrency(x: &XMem, c: &mut Checks) {
        let failures = AtomicU32::new(0);
        let (before, _, before_count) = x.counters();
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| churn(x, &failures));
            }
        });
        let (current, _, count) = x.counters();
        c.check(
            failures.load(Ordering::SeqCst) == 0 && current == before && count == before_count + 8000,
            "4 threads x 2000 allocations : compteurs revenus a zero",
        );
    }

    /// Ticket #82, the race test: eight threads free the same 128 graphics blocks
    /// at the same instant, 300 rounds. Without the lock around `XMemFreeDefault`, two threads pass
    /// the header check before one clears it, count the block twice (the total no longer comes back
    /// to its base) or touch a page already returned (the process crashes). With the lock, each
    /// block is freed and counted exactly once.
    fn double_free_race(x: &XMem, c: &mut Checks) {
        const BLOCKS: usize = 128;
        const THREADS: usize = 8;
        const ROUNDS: usize = 300;
        let (base, _, _) = x.counters();
        let mut bad = false;
        for _ in 0..ROUNDS {
            // Pages, so a double release shows in the counter without corrupting the heap.
            let blocks: Vec<usize> = (0..BLOCKS).map(|_| x.alloc(4096, GRAPHICS) as usize).collect();
            let go = std::sync::atomic::AtomicBool::new(false);
            std::thread::scope(|s| {
                for _ in 0..THREADS {
                    s.spawn(|| {
                        while !go.load(Ordering::Acquire) {
                            std::hint::spin_loop();
                        }
                        for &block in &blocks {
                            x.free(block as *mut u8, GRAPHICS);
                        }
                    });
                }
                go.store(true, Ordering::Release); // all of them start freeing at the same instant
            });
            let (now, _, _) = x.counters();
            if now != base {
                bad = true; // a block was counted more than once
                break;
            }
        }
        c.check(!bad, "double liberation concurrente : chaque bloc libere et decompte une seule fois");
    }

    pub fn test_xmem(lib: &Library, c: &mut Checks) {
        println!("xmem.dll :");
        let all = present(lib).iter().all(|&p| p);
        c.check(all, "les quatre fonctions de la console sont exportees");
        let Some(x) = XMem::load(lib).filter(|x| all && x.has_counters()) else {
            return;
        };

        let a = x.alloc(1000, 5 << 24);
        c.check(!a.is_null() && (a as usize & 31) == 0, "XMemAlloc(1000, 32-byte alignment) aligne sur 32 octets");
        if !a.is_null() {
            fill(a, 0xAB, 1000);
        }
        // 64 KiB large pages (bits 14-15 = 1) and graphics memory (bits 29+ = 1).
        let b = x.alloc(300000, (1 << 14) | (1 << 29));
        c.check(!b.is_null() && (b as usize & 0xFFFF) == 0, "XMemAlloc memoire graphique alignee sur 64 Ko");
        if !b.is_null() {
            poke(b, 0, 1);
            poke(b, 299999, 2);
        }
        let (current, _, count) = x.counters();
        c.check(current == 301000 && count == 2, "compteurs : 301000 octets en cours, 2 allocations");

        let mut info = [0xFFu8; 64];
        let ok = x.query(CURRENT_PROCESS, b.cast(), &mut info);
        let state = u32::from_le_bytes([info[0x20], info[0x21], info[0x22], info[0x23]]);
        c.check(ok && state & MEM_COMMIT != 0 && info[0x24] & 0x6 != 0 && info[0x30] & 3 != 0, "XMemVirtualQuery : memoire validee, lisible par le processeur");
        let ok = x.query(CURRENT_PROCESS, 0x10 as *const c_void, &mut info);
        c.check(!ok || info[0x30] & 3 == 0, "XMemVirtualQuery : adresse invalide refusee");

        let mut statistics = [0xFFu8; 128];
        let ok = x.stats(0x2000_0001, &mut statistics);
        let word = |i: usize| u64::from_le_bytes(statistics[i * 8..i * 8 + 8].try_into().unwrap_or_default());
        c.check(ok && word(1) > 0 && word(1) < word(0), "XMemGetWorkingSetStatistics : usage coherent");

        x.free(a, 5 << 24);
        x.free(b, (1 << 14) | (1 << 29));
        x.free(std::ptr::null_mut(), 0);
        let (current, peak, _) = x.counters();
        c.check(current == 0 && peak == 301000, "XMemFree : tout est libere, pic conserve");

        caller_buffers(&x, c);
        mismatched_free(&x, c);
        boundaries(&x, c);
        hooks(&x, c);
        concurrency(&x, c);
        double_free_race(&x, c);
    }
}
