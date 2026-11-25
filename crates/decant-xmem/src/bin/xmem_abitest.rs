//! `xmem_abitest <xmem.dll>`: calls XMem* exactly the way the shipped companion DLLs do, the port
//! of the C program `xmem_abitest.c` it replaced. Unlike `loadtest` it needs no Decant specific export,
//! so it pins the console ABI for any xmem.dll implementation. The two caller patterns:
//!   BVH builder: `XMemVirtualQuery(GetCurrentProcess(), addr, buf64)` returns BOOL, then checks
//!       `buf[0x30] & 3`, `*(u32*)(buf + 0x20) & MEM_COMMIT` and `buf[0x24] & 6`
//!   upscaler: `XMemGetWorkingSetStatistics(0x20000001, buf128)` returns BOOL, then reads
//!       `*(u64*)(buf + 8)`
//! Plus the generic allocator contract. It also prints `PROCESS_MEMORY_COUNTERS_EX`, so a Wine
//! specific zero can be told apart from a shim bug. Exit code: the number of failures, 2 when the
//! arguments are wrong or the DLL does not load.

#[cfg(windows)]
#[path = "../testapi.rs"]
mod testapi;

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    let args = decant_testkit::args();
    if args.len() != 1 {
        return std::process::ExitCode::from(2);
    }
    let lib = match decant_testkit::Library::open(&args[0]) {
        Ok(lib) => lib,
        Err(error) => {
            println!("load failed {error}");
            return std::process::ExitCode::from(2);
        }
    };
    std::process::ExitCode::from(run::abitest(&lib).min(255) as u8)
}

#[cfg(not(windows))]
fn main() {
    eprintln!("xmem_abitest runs on Windows only");
}

#[cfg(windows)]
mod run {
    use std::ffi::c_void;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU32, Ordering};

    use decant_testkit::{Checks, Library};

    use crate::testapi::{
        AllocFn, CURRENT_PROCESS, UNKNOWN_SIGNATURES, XMem, call_alloc, foreign_page, host_counters, peek, poke, protection_of, read_file_into, release_foreign_page, release_pages,
    };

    const MEM_COMMIT: u32 = 0x1000;

    static HOOK_CALLS: AtomicU32 = AtomicU32::new(0);
    static DEFAULT_ALLOC: OnceLock<AllocFn> = OnceLock::new();

    extern "system" fn counting_alloc(size: usize, attributes: u64) -> *mut c_void {
        HOOK_CALLS.fetch_add(1, Ordering::SeqCst);
        DEFAULT_ALLOC.get().map_or(std::ptr::null_mut(), |&f| call_alloc(f, size, attributes))
    }

    /// The exports titles import beyond the four above (an executable importing them must load).
    /// `XMemVirtualAlloc` follows `VirtualAlloc` with the console type bits, `XMemTransferMemory`
    /// has no known signature and fails without effect. The four physical page exports are
    /// checked in `physical_pages`.
    fn entry_points(x: &XMem, c: &mut Checks) {
        const MEM_COMMIT: u32 = 0x1000;
        const MEM_RESERVE: u32 = 0x2000;
        const PAGE_READWRITE: u32 = 0x04;
        const CONSOLE_TYPE: u32 = 0x2000_0000;
        const ERROR_CALL_NOT_IMPLEMENTED: u32 = 120;
        c.check(x.has_virtual_alloc() && (0..UNKNOWN_SIGNATURES.len()).all(|i| x.has_unknown(i)) && x.has_physical(), "XMemVirtualAlloc, XMemTransferMemory and the 4 physical page exports present");
        c.check(x.tools_mem_available() == Some(false), "XMemIsToolsMemAvailable present and FALSE");
        if !x.has_virtual_alloc() {
            return;
        }
        // A commit with a console page bit is pool memory of the engine, and a console protection becomes
        // read write for committed pages. XMemFree gives it back.
        let page = x.virtual_alloc(std::ptr::null_mut(), 0x1_0000, CONSOLE_TYPE | MEM_COMMIT | MEM_RESERVE, 0x1);
        c.check(!page.is_null() && protection_of(page) == PAGE_READWRITE, "XMemVirtualAlloc: console page bit, console protection fixed");
        if !page.is_null() {
            poke(page, 0, 1);
            poke(page, 0xFFFF, 2);
            x.free(page, 0);
            c.check(matches!(protection_of(page), 0 | 0x1), "XMemFree releases a commit with a console page bit (free memory reads as no access)");
        }
        // A reservation without commit stays uncommitted.
        let reserved = x.virtual_alloc(std::ptr::null_mut(), 0x1_0000, MEM_RESERVE, 0x1);
        c.check(!reserved.is_null() && protection_of(reserved) == 0, "XMemVirtualAlloc: bare reservation stays uncommitted");
        if !reserved.is_null() {
            x.free(reserved, 0);
        }
        let home = x.virtual_alloc(std::ptr::null_mut(), 24usize << 30, 0x2040_2000, 0x21);
        c.check(!home.is_null() && protection_of(home) == 0, "XMemVirtualAlloc: 24 GiB console home reserved without committing");
        if !home.is_null() {
            x.free(home, 0);
        }
        const ARENA_SIZE: usize = 0x4000_0000;
        let arena = x.virtual_alloc(std::ptr::null_mut(), ARENA_SIZE, 0x2000_3000, 0x183);
        let writable = !arena.is_null() && protection_of(arena) == PAGE_READWRITE;
        c.check(writable, "XMemVirtualAlloc: 0x183 becomes PAGE_READWRITE for the committed arena");
        if writable {
            poke(arena, 0, 1);
            poke(arena, ARENA_SIZE - 1, 2);
            c.check(peek(arena, 0) == 1 && peek(arena, ARENA_SIZE - 1) == 2, "committed arena is writable at both ends");
            c.check((arena as usize).is_multiple_of(0x20_0000) && arena as usize + ARENA_SIZE <= 16 << 30, &format!("the arena lies below 16 GiB on 2 MiB ({arena:p})"));
        }
        if !arena.is_null() {
            x.free(arena, 0);
        }
        let commit = x.virtual_alloc(std::ptr::null_mut(), 0x1_0000, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        c.check(!commit.is_null() && protection_of(commit) == PAGE_READWRITE, "XMemVirtualAlloc: a PC call behaves as VirtualAlloc");
        if !commit.is_null() {
            release_pages(commit);
        }
        let impossible = x.virtual_alloc(std::ptr::null_mut(), usize::MAX, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        let error = x.last_error();
        c.check(impossible.is_null() && error != 0, "XMemVirtualAlloc: impossible size returns NULL with a Win32 error");
        // XMemTransferMemory logs its raw arguments and fails with ERROR_CALL_NOT_IMPLEMENTED.
        let failing = (0..UNKNOWN_SIGNATURES.len()).all(|i| x.call_unknown(i, [1, 2, 3, 4]) == Some((0, ERROR_CALL_NOT_IMPLEMENTED)));
        c.check(failing, "XMemTransferMemory returns 0 with ERROR_CALL_NOT_IMPLEMENTED");
    }

    /// The physical page exports, called as the specification gives them (`docs/CONSOLE_LIBS.md`,
    /// 5.1): pages are 64 KiB, frame numbers count 4 KiB units (page `i` of the section is frame
    /// `(i + 1) * 16`, 0 means none). `XMemVirtualAlloc` with `0x20406000` only reserves the range
    /// the frames are mapped into. This process starts the engine, so the first page is frame 16.
    fn physical_pages(x: &XMem, c: &mut Checks) {
        const P: usize = 0x1_0000;
        const MAP_LATER: u32 = 0x2040_6000;
        const RESERVE_PLACEHOLDER: u32 = 0x42000;
        const PAGES_64K: u64 = 0x2040_0000;
        const PAGES_LARGE: u64 = 0x2000_0000;
        const READWRITE: u32 = 0x04;
        const READONLY: u32 = 0x02;
        const INVALID_PARAMETER: u32 = 87;
        const NOACCESS: u32 = 998;
        const INVALID_ADDRESS: u32 = 487;
        if !x.has_virtual_alloc() || !x.has_physical() {
            return;
        }
        let mapped = |address: *const u8| protection_of(address) == READWRITE;

        // 0x20406000 reserves, it allocates and maps nothing.
        let area = x.virtual_alloc(std::ptr::null_mut(), 8 * P, MAP_LATER, READWRITE);
        c.check(!area.is_null() && (area as usize).is_multiple_of(P), "XMemVirtualAlloc 0x20406000: a 64 KiB aligned range is reserved");
        if area.is_null() {
            return;
        }
        let at = |page: usize| area.wrapping_add(page * P);
        c.check((0..8).all(|i| !mapped(at(i))), "the reservation holds no mapped page");
        let impossible = x.virtual_alloc(std::ptr::null_mut(), usize::MAX, MAP_LATER, READWRITE);
        c.check(impossible.is_null(), "an impossible reservation returns NULL");
        let small = x.virtual_alloc(std::ptr::null_mut(), 0x1234, MAP_LATER, READWRITE);
        c.check(!small.is_null() && x.map_pages(small, 1, Some(&[16 * 99])).0.is_null(), "a reservation of a few bytes covers one 64 KiB page, and a frame never allocated is refused");
        x.free(small, 0);

        // Allocate pages: frame numbers in 4 KiB units, 16 apart for consecutive 64 KiB pages.
        let (ok, count, frames, _) = x.allocate_pages(PAGES_64K, 3);
        c.check(ok && count == 3 && frames == [16, 32, 48], "XMemAllocatePhysicalPages: 3 pages are frames 16, 32 and 48");
        if !ok {
            x.free(area, 0);
            return;
        }

        // Map them contiguously and write through the mapping.
        let (result, _) = x.map_pages(at(1), 3, Some(&frames));
        c.check(result == at(1) && (1..4).all(|i| mapped(at(i))) && !mapped(at(0)) && !mapped(at(4)), "XMemMapPhysicalPages: 3 frames mapped from the second page of the range, the address is returned");
        poke(at(1), 0, 0x11);
        poke(at(2), P - 1, 0x22);
        poke(at(3), 7, 0x33);

        // The same page at a second address.
        let (second, _) = x.map_pages(at(6), 1, Some(&frames[..1]));
        c.check(second == at(6) && peek(at(6), 0) == 0x11, "frame 16 mapped at a second address shows the same bytes");
        poke(at(6), 100, 0x5A);
        c.check(peek(at(1), 100) == 0x5A, "a write at the second address shows at the first");

        // An unaligned address maps at its 64 KiB boundary and is returned as passed.
        let odd = at(7).wrapping_add(0x1234);
        let (unaligned, _) = x.map_pages(odd, 1, Some(&frames[2..]));
        c.check(unaligned == odd && mapped(at(7)) && peek(at(7), 7) == 0x33, "an unaligned address is rounded down to 64 KiB and returned as passed");

        // Scatter: each frame at its own address, in any order.
        let (ok, _, pair, _) = x.allocate_pages(PAGES_64K, 2);
        c.check(ok && pair == [64, 80], "two more pages are frames 64 and 80");
        let (scattered, _) = x.map_scatter(Some(&[at(5), at(0)]), 2, Some(&pair));
        poke(at(5), 0, 0x44);
        poke(at(0), 0, 0x55);
        c.check(scattered && mapped(at(5)) && mapped(at(0)), "XMemMapPhysicalPagesScatter: two frames at two addresses");
        let (again, _) = x.map_pages(at(4), 1, Some(&pair[..1]));
        c.check(!again.is_null() && peek(at(4), 0) == 0x44, "each frame is at the address it was given");

        // Mapping over a mapped page replaces it and leaves the neighbours.
        let (replaced, _) = x.map_pages(at(2), 1, Some(&pair[1..]));
        c.check(!replaced.is_null() && peek(at(2), 0) == 0x55 && peek(at(1), 0) == 0x11 && peek(at(3), 7) == 0x33, "a mapped page is replaced, its neighbours stay");

        // The 0 sentinel and the null array unmap.
        let (zeroed, _) = x.map_pages(at(5), 1, Some(&[0]));
        c.check(zeroed == at(5) && !mapped(at(5)) && mapped(at(4)), "an entry of 0 unmaps its page and returns to the reserved state");
        let (nulled, _) = x.map_pages(at(3), 2, None);
        c.check(nulled == at(3) && !mapped(at(3)) && !mapped(at(4)) && mapped(at(2)), "a null array unmaps the range");
        let (scatter_zero, _) = x.map_scatter(Some(&[at(6), at(0)]), 2, Some(&[0, pair[0]]));
        c.check(scatter_zero && !mapped(at(6)) && mapped(at(0)), "scatter: 0 unmaps, another entry still maps");
        let (scatter_null, _) = x.map_scatter(Some(&[at(0), at(2)]), 2, None);
        c.check(scatter_null && !mapped(at(0)) && !mapped(at(2)), "scatter with a null array unmaps");
        let (remap, _) = x.map_pages(at(2), 1, Some(&frames[1..2]));
        c.check(!remap.is_null() && peek(at(2), P - 1) == 0x22, "an unmapped page keeps its bytes and maps again");

        // Invalid arguments.
        let (r, e) = x.map_pages(std::ptr::null_mut(), 1, Some(&frames));
        c.check(r.is_null() && e == INVALID_PARAMETER, "map: a null address is ERROR_INVALID_PARAMETER");
        let (r, _) = x.map_pages(at(5), 0, Some(&frames));
        c.check(r == at(5), "map: a count of 0 returns the address unchanged");
        let (r, e) = x.map_pages(at(5), 1, Some(&[16 * 99]));
        c.check(r.is_null() && e == INVALID_PARAMETER && !mapped(at(5)), "map: a frame never allocated is refused");
        let (r, e) = x.map_pages(at(5), 1, Some(&[17]));
        c.check(r.is_null() && e == INVALID_PARAMETER, "map: a frame that is no multiple of 16 is refused");
        let (r, e) = x.map_pages(at(7), 3, Some(&frames));
        c.check(r.is_null() && e == INVALID_ADDRESS && mapped(at(7)), "map: a range past the reservation is refused and changes nothing");
        let outside = foreign_page();
        let (r, e) = x.map_pages(outside, 1, Some(&frames[..1]));
        c.check(r.is_null() && e == INVALID_ADDRESS, "map: an address the title did not reserve is refused");
        release_foreign_page(outside);
        let (r, e) = x.map_scatter(None, 1, Some(&frames));
        c.check(!r && e == INVALID_PARAMETER, "scatter: a null address array is ERROR_INVALID_PARAMETER");
        let (r, e) = x.map_scatter(Some(&[at(5), at(7)]), 2, Some(&[frames[0], 16 * 99]));
        c.check(!r && e == INVALID_PARAMETER && !mapped(at(5)), "scatter: a bad frame maps nothing");
        let (r, _) = x.map_scatter(Some(&[at(5)]), 0, None);
        c.check(r, "scatter: a count of 0 succeeds");

        // Allocation arguments.
        c.check(x.allocate_pages_null_count(PAGES_64K) == (false, INVALID_PARAMETER), "allocate: a null count is ERROR_INVALID_PARAMETER");
        c.check(x.allocate_pages_null_array(PAGES_64K, 2) == (false, 0, INVALID_PARAMETER), "allocate: a count with a null array fails and the count becomes 0");
        let (ok, left, _, e) = x.allocate_pages(PAGES_LARGE, 5);
        c.check(!ok && left == 0 && e == INVALID_PARAMETER, "allocate: large pages need a multiple of 32, the count becomes 0");
        let (ok, left, _, _) = x.allocate_pages(PAGES_64K, 0);
        c.check(ok && left == 0, "allocate: a count of 0 succeeds");
        let (ok, left, large, _) = x.allocate_pages(PAGES_LARGE, 32);
        c.check(ok && left == 32 && large.iter().all(|f| f % 16 == 0 && *f != 0), "allocate: 32 large pages succeed and keep their count");
        let (ok, left, _, _) = x.allocate_pages(PAGES_64K, 1 << 20);
        c.check(!ok && left == 0, "allocate: more pages than the section has fails and the count becomes 0");

        // Free: a mapped page stays readable until it is unmapped, then comes back.
        c.check(x.free_pages(0, None).0, "free: a count of 0 succeeds");
        c.check(x.free_pages(1, None) == (false, NOACCESS), "free: a null array is ERROR_NOACCESS");
        c.check(x.free_pages(2, Some(&[frames[0], frames[0]])) == (false, INVALID_PARAMETER), "free: a frame listed twice is refused");
        c.check(x.free_pages(2, Some(&[frames[0], 16 * 99])) == (false, INVALID_PARAMETER), "free: a frame never allocated is refused");
        x.map_pages(at(6), 1, Some(&frames[..1]));
        let (freed, _) = x.free_pages(1, Some(&frames[..1]));
        c.check(freed && peek(at(1), 100) == 0x5A && peek(at(6), 100) == 0x5A, "free: page 16 freed while mapped is still readable through its views");
        c.check(x.free_pages(1, Some(&frames[..1])) == (false, INVALID_PARAMETER), "free: a second free of the same frame is refused");
        let (r, _) = x.map_pages(at(5), 1, Some(&frames[..1]));
        c.check(r.is_null(), "a frame the title freed is not mapped again");
        c.check(!x.free_pages(2, Some(&[0, frames[1]])).0, "free: the frame 0 is not a page");
        x.map_pages(at(1), 1, None);
        x.map_pages(at(6), 1, None);
        let (ok, _, reused, _) = x.allocate_pages(PAGES_64K, 1);
        c.check(ok && reused == [16], "free: once unmapped everywhere the page is allocated again as frame 16");
        let all: Vec<u64> = [frames.as_slice(), pair.as_slice(), reused.as_slice(), large.as_slice()].concat();
        let chunk_check = x.free_pages(all.len(), Some(&all));
        c.check(!chunk_check.0 && chunk_check.1 == INVALID_PARAMETER, "free: a list with a frame already freed is refused (frame 16 is listed twice)");
        let live: Vec<u64> = [&frames[1..], pair.as_slice(), reused.as_slice(), large.as_slice()].concat();
        c.check(x.free_pages(live.len(), Some(&live)).0, "free: the pages still held are freed in one call");

        // Reserved with a read only protection, the mappings are read only.
        let ro = x.virtual_alloc(std::ptr::null_mut(), P, MAP_LATER, READONLY);
        let (ok, _, one, _) = x.allocate_pages(PAGES_64K, 1);
        let (r, _) = x.map_pages(ro, 1, Some(&one));
        c.check(ok && r == ro && protection_of(ro) == READONLY, "a mapping takes the protection of its reservation");
        x.free(ro, 0);
        x.free_pages(1, Some(&one));

        // XMemFree ends the whole reservation, whatever its attributes (0x8002 is not special).
        let (ok, _, held, _) = x.allocate_pages(PAGES_64K, 2);
        let (mapped_ok, _) = x.map_scatter(Some(&[at(3), at(5)]), 2, Some(&held));
        c.check(ok && mapped_ok && mapped(at(3)), "two pages mapped before the reservation is freed");
        x.free(area, 0x8002);
        let back = x.virtual_alloc(area.cast(), 8 * P, MAP_LATER, READWRITE);
        c.check(back == area, "XMemFree with 0x8002 released the whole reservation, views included: the address can be reserved again");
        c.check(!mapped(back), "the new reservation holds no page");
        c.check(x.free_pages(2, Some(&held)).0, "the pages of the released views are still the title's to free");
        x.free(back, 0);
        let placeholder = x.virtual_alloc(std::ptr::null_mut(), 2 * P, RESERVE_PLACEHOLDER, READWRITE);
        let (ok, _, one, _) = x.allocate_pages(PAGES_64K, 1);
        let (r, _) = x.map_pages(placeholder, 1, Some(&one));
        c.check(ok && !placeholder.is_null() && r == placeholder, "0x42000 also reserves a range that frames map into");
        x.free(placeholder, 0);
        x.free_pages(1, Some(&one));
    }

    /// The title asks for far more physical memory than the first pool of 1 GiB (it asked for 100000
    /// than 6 GiB in one call). The pool grows: 1.25 GiB are allocated, mapped over one
    /// placeholder, written at both ends and at the page where the first 1 GiB ends, then unmapped
    /// and freed. Only the touched pages use memory.
    fn beyond_the_first_pool(x: &XMem, c: &mut Checks) {
        const P: usize = 0x1_0000;
        const PAGES: usize = 20 * 1024;
        const FIRST_POOL_PAGES: usize = 16 * 1024;
        const RESERVE_PLACEHOLDER: u32 = 0x42000;
        const PAGES_64K: u64 = 0x2040_0000;
        if !x.has_virtual_alloc() || !x.has_physical() {
            return;
        }
        let area = x.virtual_alloc(std::ptr::null_mut(), PAGES * P, RESERVE_PLACEHOLDER, 0x04);
        c.check(!area.is_null(), "0x42000 reserves a placeholder of 1.25 GiB");
        if area.is_null() {
            return;
        }
        let (ok, count, frames, error) = x.allocate_pages(PAGES_64K, PAGES);
        c.check(ok && count == PAGES, &format!("XMemAllocatePhysicalPages: 20480 pages, more than the first pool of 16384, are allocated (error {error})"));
        if ok {
            let (mapped, error) = x.map_pages(area, PAGES, Some(&frames));
            c.check(mapped == area, &format!("  ... all 20480 pages map into the placeholder (error {error})"));
            if mapped == area {
                let marks = [(0, 0xB1), (FIRST_POOL_PAGES - 1, 0xB2), (FIRST_POOL_PAGES, 0xB3), (PAGES - 1, 0xB4)];
                for &(page, value) in &marks {
                    poke(area.wrapping_add(page * P), if page % 2 == 0 { 0 } else { P - 1 }, value);
                }
                c.check(marks.iter().all(|&(page, value)| peek(area.wrapping_add(page * P), if page % 2 == 0 { 0 } else { P - 1 }) == value), "  ... both ends and the page where the first pool ends are writable and read back");
                let (unmapped, _) = x.map_pages(area, PAGES, None);
                c.check(!unmapped.is_null() && protection_of(area) == 0, "  ... unmapping gives the placeholder back");
            }
            let (freed, error) = x.free_pages(PAGES, Some(&frames));
            c.check(freed, &format!("XMemFreePhysicalPages: the 20480 pages are freed (error {error})"));
            let (again, _, _, _) = x.allocate_pages(PAGES_64K, PAGES);
            c.check(again, "  ... and they can be allocated again");
        }
        x.free(area, 0);
    }

    /// The title's sequence: a plain `MEM_RESERVE` of 0x190000000 bytes (`0x20402000`, protection
    /// 0x23) is a target for physical pages, as on the console. 100000 pages are allocated, mapped
    /// into the start of it, written at both ends, unmapped and freed. A commit inside such a
    /// range works, and `XMemFree` gives the whole reservation back.
    fn plain_reserve_takes_pages(x: &XMem, c: &mut Checks) {
        const P: usize = 0x1_0000;
        const WHOLE: usize = 0x1_9000_0000;
        const PAGES: usize = 100000;
        const PLAIN_RESERVE: u32 = 0x2040_2000;
        const PLAIN_COMMIT: u32 = 0x2040_3000;
        const PAGES_64K: u64 = 0x2040_0000;
        if !x.has_virtual_alloc() || !x.has_physical() {
            return;
        }
        let area = x.virtual_alloc(std::ptr::null_mut(), WHOLE, PLAIN_RESERVE, 0x23);
        c.check(!area.is_null() && (area as usize).is_multiple_of(P) && protection_of(area) == 0, &format!("XMemVirtualAlloc 0x20402000, 0x23: 0x190000000 bytes are reserved, nothing committed ({area:p}, protection {}, error {})", protection_of(area), x.last_error()));
        if area.is_null() {
            return;
        }
        let (ok, count, frames, error) = x.allocate_pages(PAGES_64K, PAGES);
        c.check(ok && count == PAGES, &format!("XMemAllocatePhysicalPages: 100000 pages (error {error})"));
        if ok {
            let (mapped, error) = x.map_pages(area, PAGES, Some(&frames));
            c.check(mapped == area, &format!("XMemMapPhysicalPages: the pages map into the start of the plain reservation (error {error})"));
            if mapped == area {
                poke(area, 0, 0xC1);
                poke(area, PAGES * P - 1, 0xC2);
                c.check(peek(area, 0) == 0xC1 && peek(area, PAGES * P - 1) == 0xC2, "  ... both ends are writable and read back");
                let (unmapped, _) = x.map_pages(area, PAGES, None);
                c.check(!unmapped.is_null() && protection_of(area) == 0, "  ... unmapping gives the reservation back");
            }
            let (freed, _) = x.free_pages(PAGES, Some(&frames));
            c.check(freed, "XMemFreePhysicalPages: the pages are freed");
        }
        let tail = area.wrapping_add(WHOLE - 4 * P);
        let committed = x.virtual_alloc(tail.cast(), 2 * P, PLAIN_COMMIT, 0x04);
        if !committed.is_null() {
            poke(committed, 0, 3);
            poke(committed, 2 * P - 1, 4);
        }
        c.check(committed == tail && peek(committed, 2 * P - 1) == 4, "a commit inside the plain reservation works");
        x.free(area, 0);
        c.check(x.virtual_alloc(area.cast(), WHOLE, PLAIN_RESERVE, 0x04) == area, "XMemFree releases the whole reservation, its address can be reserved again");
        x.free(area, 0);
    }

    /// The exports `d3d12_xs.dll` uses to follow the pages from the GPU. With tracking on, a mapped
    /// page is dirty once, protected after `DecantXmemGpuTakeDirty`, and a write to it is seen again. A
    /// file read into a clean page with this program's own imported `ReadFile` succeeds, because the
    /// guard of `xmem.dll` opens the page first. A query reports the title's protection.
    fn gpu_exports(x: &XMem, c: &mut Checks) {
        const P: usize = 0x1_0000;
        if !x.has_gpu_exports() {
            return;
        }
        let placeholder = x.virtual_alloc(std::ptr::null_mut(), 4 * P, 0x42000, 0x04);
        let (ok, _, frames, _) = x.allocate_pages(0x2040_0000, 2);
        let (mapped, _) = x.map_pages(placeholder, 2, Some(&frames));
        if ok && frames[1] == frames[0] + 16 && mapped == placeholder {
            let page = (frames[0] / 16 - 1) as u32;
            x.gpu_track(true);
            let first = x.gpu_take_dirty(1024);
            c.check(first.contains(&page) && first.contains(&(page + 1)), "DecantXmemGpuTakeDirty: the mapped pages are dirty at first");
            c.check(x.gpu_take_dirty(1024).is_empty(), "DecantXmemGpuTakeDirty: each page is returned once");
            let mut info = [0u8; 64];
            let queried = x.query(CURRENT_PROCESS, placeholder.cast(), &mut info);
            let protect = u32::from_le_bytes([info[0x24], info[0x25], info[0x26], info[0x27]]);
            c.check(queried && protect == 0x04, "XMemVirtualQuery: a clean page still reports the title's read write protection");
            c.check(x.gpu_writable(placeholder, 2 * P) && !x.gpu_writable(placeholder, 3 * P), "DecantXmemGpuWritable: the mapped pages are, a placeholder is not");
            poke(placeholder, 0, 0x42);
            c.check(x.gpu_take_dirty(1024) == [page], "a write to a clean page is seen, and only that page comes back");
            // The second page is clean again after this take. A file read into it needs the guard.
            let path = std::env::temp_dir().join(format!("xmem_abitest_{}.bin", std::process::id()));
            let wrote = std::fs::write(&path, [0x7Eu8; 4096]).is_ok();
            let second = placeholder.wrapping_add(P);
            let read = read_file_into(&path, second, 4096);
            let _ = std::fs::remove_file(&path);
            c.check(wrote && read == Some(4096) && peek(second, 0) == 0x7E && peek(second, 4095) == 0x7E, "ReadFile into a clean page succeeds through the guard");
            x.prepare_write(placeholder.wrapping_add(P), 8);
            c.check(x.gpu_take_dirty(1024).contains(&(page + 1)), "the page of a read comes back as dirty");
            x.gpu_track(false);
        }
        x.free(placeholder, 0);
        x.free_pages(2, Some(&frames));
    }

    /// The sequence of a retail title: `0x42000` reserves a placeholder, then
    /// `XMemVirtualAlloc(base + offset, 0xa00000, 0x20406000, 0x27)` names a range of it. Without
    /// `MEM_COMMIT` the range stays a placeholder of exactly that size, 10 MiB of physical pages are
    /// mapped into it, written at both ends, unmapped and freed, at the base and in the middle, twice.
    /// With `MEM_COMMIT` (`0x20407000`) the range is committed memory instead, which `XMemFree` gives back.
    fn placed_in_placeholder(x: &XMem, c: &mut Checks) {
        const P: usize = 0x1_0000;
        const LEN: usize = 0xa0_0000;
        const WHOLE: usize = 0xb90_0000;
        const RESERVE_PLACEHOLDER: u32 = 0x42000;
        const CONSOLE_TYPE: u32 = 0x2040_6000;
        const CONSOLE_COMMIT: u32 = 0x2040_7000;
        const CONSOLE_PROTECT: u32 = 0x27;
        const PAGES_64K: u64 = 0x2040_0000;
        if !x.has_virtual_alloc() || !x.has_physical() {
            return;
        }
        let base = x.virtual_alloc(std::ptr::null_mut(), WHOLE, RESERVE_PLACEHOLDER, CONSOLE_PROTECT);
        c.check(!base.is_null(), "0x42000 reserves a placeholder of 0xb900000 bytes");
        if base.is_null() {
            return;
        }
        for (name, offset) in [("at the base", 0), ("in the middle", 0x35 * P)] {
            for round in 1..=2 {
                let at = base.wrapping_add(offset);
                let got = x.virtual_alloc(at.cast(), LEN, CONSOLE_TYPE, CONSOLE_PROTECT);
                c.check(got == at && protection_of(got) == 0, &format!("0xa00000 bytes {name}, 0x20406000 and protect 0x27 (round {round}): the placeholder is kept, nothing committed"));
                if got != at {
                    break;
                }
                let (ok, _, frames, _) = x.allocate_pages(PAGES_64K, LEN / P);
                let (mapped, error) = x.map_pages(got, LEN / P, Some(&frames));
                c.check(ok && mapped == got, &format!("  ... {} physical pages map into it (error {})", LEN / P, if mapped == got { 0 } else { error }));
                if mapped == got {
                    poke(got, 0, 0xA1);
                    poke(got, LEN - 1, 0xA2);
                    c.check(peek(got, 0) == 0xA1 && peek(got, LEN - 1) == 0xA2, "  ... both ends are writable and read back");
                    let (unmapped, _) = x.map_pages(got, LEN / P, None);
                    c.check(!unmapped.is_null() && protection_of(got) == 0, "  ... unmapping gives the placeholder back");
                }
                if ok {
                    x.free_pages(LEN / P, Some(&frames));
                }
            }
        }
        let committed = x.virtual_alloc(base.wrapping_add(LEN).cast(), 2 * P, CONSOLE_COMMIT, CONSOLE_PROTECT);
        if !committed.is_null() {
            poke(committed, 0, 7);
            poke(committed, 2 * P - 1, 8);
        }
        c.check(committed == base.wrapping_add(LEN) && peek(committed, 2 * P - 1) == 8 && protection_of(committed) == 0x04, "0x20407000 commits the range, writable at both ends");
        x.free(committed, 0);
        c.check(protection_of(committed) == 0, "  ... XMemFree gives it back to the reservation");
        // The pages of a commit go back to the pool, and a commit that gets them again reads zeros.
        let again = x.virtual_alloc(base.wrapping_add(LEN).cast(), 2 * P, CONSOLE_COMMIT, CONSOLE_PROTECT);
        c.check(again == committed && peek(again, 0) == 0 && peek(again, 2 * P - 1) == 0, "  ... a second commit of the range reads zeros");
        x.free(again, 0);
        let outside = x.virtual_alloc(base.wrapping_add(WHOLE - P).cast(), 2 * P, CONSOLE_TYPE, CONSOLE_PROTECT);
        c.check(outside.is_null() && x.last_error() == 487, "a range that sticks out of the placeholder fails with ERROR_INVALID_ADDRESS");
        x.free(base, 0);
        let back = x.virtual_alloc(base.cast(), WHOLE, RESERVE_PLACEHOLDER, CONSOLE_PROTECT);
        c.check(back == base, "XMemFree of the reservation released all of it");
        x.free(back, 0);
    }

    pub fn abitest(lib: &Library) -> u32 {
        let mut c = Checks::new(66, "OK", "FAIL");
        let xmem = XMem::load(lib);
        c.check(xmem.is_some(), "4 imported XMem exports present");
        let Some(x) = xmem else {
            return 1;
        };

        let heap = x.alloc(1000, 5 << 24);
        let gfx = x.alloc(300000, (1 << 14) | (1 << 29));
        let big = x.alloc(4096, (12 << 24) | (1 << 29)); // 4 KiB graphics, 2^12 alignment
        c.check(!heap.is_null() && heap as usize & 31 == 0, "heap, 2^5 alignment");
        c.check(!gfx.is_null() && gfx as usize & 0xFFFF == 0, "graphics, 64K page size -> 64K aligned");
        c.check(!big.is_null() && big as usize & 0xFFF == 0, "graphics, 2^12 alignment");

        // XboxBvhBuilder pattern: 64 byte zeroed stack buffer.
        let mut buf = [0u8; 64];
        let r = x.query(CURRENT_PROCESS, gfx.cast(), &mut buf);
        let state = u32::from_le_bytes([buf[0x20], buf[0x21], buf[0x22], buf[0x23]]);
        c.check(
            r && buf[0x30] & 3 != 0 && state & MEM_COMMIT != 0 && buf[0x24] & 6 != 0,
            "XMemVirtualQuery(GetCurrentProcess(), p, buf64) like XboxBvhBuilder",
        );
        let mut canary = [0xCCu8; 64];
        let r = x.query(CURRENT_PROCESS, 0x10 as *const c_void, &mut canary);
        c.check(!r || canary[0x30] & 3 == 0, "XMemVirtualQuery rejects unmapped address");

        // FSR upscaler pattern: 0x80 byte stack buffer, selector 0x20000001.
        let mut st = [0xEEu8; 128];
        let r = x.stats(0x2000_0001, &mut st);
        let word = |i: usize| u64::from_le_bytes(st[i * 8..i * 8 + 8].try_into().unwrap_or_default());
        println!("    ret={} st[0]={} st[1]={}", i32::from(r), word(0), word(1));
        c.check(r, "XMemGetWorkingSetStatistics(0x20000001, buf128) returns TRUE");
        c.check(r && word(1) != 0xEEEE_EEEE_EEEE_EEEE, "  ... and writes the u64 at +8 that the caller reads");

        if let Some((private, working_set, pagefile)) = host_counters() {
            println!("    host: PrivateUsage={private} WorkingSet={working_set} PagefileUsage={pagefile}");
        }

        // Free with *different* attributes than at allocation (callers are not guaranteed to match).
        x.free(heap, 5 << 24);
        x.free(gfx, (1 << 14) | (1 << 29));
        x.free(big, 0);
        x.free(std::ptr::null_mut(), 0);
        c.check(true, "frees (incl. mismatched attributes) did not crash");

        // Frees that must be ignored, not crash: a second free of a graphics allocation (its pages
        // are already released), a pointer at the start of a page (no header in front of it), and a
        // second free of a heap allocation.
        let twice = x.alloc(300000, (1 << 14) | (1 << 29));
        x.free(twice, 0);
        x.free(twice, 0);
        let page = foreign_page();
        x.free(page, 0);
        release_foreign_page(page);
        let little = x.alloc(32, 0);
        x.free(little, 0);
        x.free(little, 0);
        c.check(true, "double frees and a foreign page-start pointer are ignored");

        // Sizes that cannot be served return NULL instead of wrapping around.
        c.check(x.alloc(usize::MAX, 0).is_null(), "XMemAlloc(SIZE_MAX) returns NULL");
        c.check(x.alloc(usize::MAX - 63, 31 << 24).is_null(), "size plus a 2^31 alignment that overflows returns NULL");

        // Every alignment the 5 bit field can ask for up to 2^20 is honoured, heap and pages.
        let mut aligned = true;
        for log2 in 5..=20u64 {
            for gfx_type in 0..=1u64 {
                let q = x.alloc(100, (log2 << 24) | (gfx_type << 29));
                aligned &= !q.is_null() && q as usize & ((1usize << log2) - 1) == 0;
                if !q.is_null() {
                    poke(q, 0, 1);
                    poke(q, 99, 2);
                }
                x.free(q, 0);
            }
        }
        c.check(aligned, "alignments 2^5 to 2^20, heap and graphics, are honoured and writable");

        // XMemSetAllocationHooks(alloc, NULL): free must fall back to the default, not to NULL.
        if let (true, Some(default)) = (x.has_hooks(), x.default_alloc) {
            let _ = DEFAULT_ALLOC.set(default);
            x.set_hooks(Some(counting_alloc), None);
            let p = x.alloc(64, 0);
            x.free(p, 0);
            x.set_hooks(None, None);
            x.free(x.alloc(64, 0), 0);
            c.check(HOOK_CALLS.load(Ordering::SeqCst) == 1, "XMemSetAllocationHooks(alloc, NULL) keeps a working free");
        }
        entry_points(&x, &mut c);
        physical_pages(&x, &mut c);
        placed_in_placeholder(&x, &mut c);
        gpu_exports(&x, &mut c);
        beyond_the_first_pool(&x, &mut c);
        plain_reserve_takes_pages(&x, &mut c);
        let fails = c.failures();
        println!("{} ({fails} failure(s))", if fails != 0 { "ABITEST FAIL" } else { "ABITEST PASS" });
        fails
    }
}
