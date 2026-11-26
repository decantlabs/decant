//! `xmem.dll`: the console allocator of Decant, on Windows virtual memory and the process heap.
//! Replaces the C source `xmem.c` (deleted), with the same exports, calling convention and
//! behaviour. The rules (attribute fields, alignment, header) are plain functions here. The
//! exports and every Windows call are in `exports`, the only module with `unsafe` code.
//!
//! The placeholder and physical page model lives in `pages` (frame numbers, page allocator),
//! `regions` (placeholders and views) and `engine` (the Windows calls), see the README.
//!
//! An allocation is preceded by a 24 byte header (raw address, requested size, magic, origin) so
//! that `XMemFree` finds what to release whatever attributes the caller passes.

pub mod journal;
pub mod pages;
pub mod regions;
pub mod smallbuf;
pub mod track;

#[cfg(all(windows, target_arch = "x86_64"))]
mod cpu_emu_veh;
#[cfg(windows)]
mod engine;
#[cfg(windows)]
mod exports;
#[cfg(windows)]
mod io_guard;
#[cfg(windows)]
mod memsample;

/// Page size field, bits 14 and 15 of the attributes.
pub fn page_size_field(attributes: u64) -> u32 {
    ((attributes >> 14) & 0x3) as u32
}

/// log2 alignment field, bits 24 to 28.
pub fn alignment_field(attributes: u64) -> u32 {
    ((attributes >> 24) & 0x1F) as u32
}

/// Memory type field, bits 29 to 32. Not zero means graphics memory.
pub fn memory_type_field(attributes: u64) -> u32 {
    ((attributes >> 29) & 0xF) as u32
}

/// "XMEM", the header magic of a live allocation.
pub const HEADER_MAGIC: u32 = 0x584D_454D;

/// Size of the header in front of every allocation.
pub const HEADER_BYTES: usize = 24;

/// The alignment an allocation gets: at least 16, the requested power of two above 2^4, and the
/// page size for graphics or large page memory.
pub fn alignment_of(attributes: u64) -> usize {
    const PAGE_SIZES: [usize; 4] = [0x1000, 0x10000, 0x40_0000, 0x1000];
    let log2 = alignment_field(attributes);
    let mut alignment = if log2 > 4 { 1usize << log2 } else { 16 };
    if memory_type_field(attributes) != 0 || page_size_field(attributes) != 0 {
        alignment = alignment.max(PAGE_SIZES[page_size_field(attributes) as usize]);
    }
    alignment
}

/// Whether the block comes from `VirtualAlloc` (graphics, large pages or 1 MiB and more) rather
/// than from the process heap.
pub fn uses_pages(size: usize, attributes: u64) -> bool {
    memory_type_field(attributes) != 0 || page_size_field(attributes) != 0 || size >= 0x10_0000
}

/// Bytes to reserve for `size` user bytes, or `None` when the sum wraps around.
pub fn total_bytes(size: usize, alignment: usize) -> Option<usize> {
    let total = size.wrapping_add(alignment).wrapping_add(HEADER_BYTES);
    (total >= size).then_some(total)
}

/// The user address inside a block that starts at `raw`: aligned, with room for the header.
pub fn user_address(raw: usize, alignment: usize) -> usize {
    (raw + HEADER_BYTES + alignment - 1) & !(alignment - 1)
}

/// The allocation type bits of `VirtualAlloc` on PC: `MEM_COMMIT`, `MEM_RESERVE`, `MEM_RESET`,
/// `MEM_RESET_UNDO` and `MEM_TOP_DOWN`. Any other bit of a console caller is dropped.
pub const PC_ALLOCATION_TYPES: u32 = 0x1000 | 0x2000 | 0x8_0000 | 0x100_0000 | 0x10_0000;

const MEM_COMMIT: u32 = 0x1000;
const PAGE_NOACCESS: u32 = 0x01;
const PAGE_READWRITE: u32 = 0x04;

/// Whether the low byte of `protect` is a protection `VirtualAlloc` accepts for committed pages:
/// one of the read, write copy and execute forms, not `PAGE_NOACCESS` and not a console value.
pub fn usable_protection(protect: u32) -> bool {
    matches!(protect & 0xFF, 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80)
}

const MEM_RESERVE: u32 = 0x2000;
/// `MEM_RESERVE_PLACEHOLDER`: the reservation is a placeholder (`0x42000` with `MEM_RESERVE`).
pub const MEM_RESERVE_PLACEHOLDER: u32 = 0x4_0000;
/// The console bit 0x4000 that comes with `MEM_RESERVE` in `0x20406000`. The specification
/// reads it as a reservation that the title maps frames into afterwards. Windows names the same
/// value `MEM_REPLACE_PLACEHOLDER`, which only `MapViewOfFile3` takes, so it is dropped here.
pub const MAP_LATER: u32 = 0x4000;
/// The two console bits of 64 KiB pages (`0x2040_0000`).
pub const PAGES_64K: u32 = 0x2040_0000;
/// Console bit of large (2 MiB) pages, with no 64 KiB bit.
pub const PAGES_LARGE: u32 = 0x2000_0000;

/// Which of the two models an `XMemVirtualAlloc` call asks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VirtualRoute {
    /// Any reservation without `MEM_COMMIT` (`0x20406000`, `0x20402000`, `0x42000`, a plain
    /// `MEM_RESERVE`): on the console it is a valid target for `XMemMapPhysicalPages`, so it is a
    /// placeholder that the title maps physical pages into later. Nothing is allocated.
    Placeholder,
    /// A commit with a console page bit (`0x20003000`, `0x20403000`, `0x20401000`): arenas of
    /// 0.5 to 2.2 GiB in a retail title, which hold vertex and index data the GPU is given by address. The engine reserves
    /// the range and commits it with pool pages, so that the GPU mirror can show its pages.
    Pool,
    /// Anything else: `VirtualAlloc` with the PC bits.
    Plain,
}

pub fn virtual_route(allocation_type: u32) -> VirtualRoute {
    if allocation_type & MEM_RESERVE != 0 && allocation_type & MEM_COMMIT == 0 {
        VirtualRoute::Placeholder
    } else if allocation_type & MEM_COMMIT != 0 && allocation_type & PAGES_LARGE != 0 {
        VirtualRoute::Pool
    } else {
        VirtualRoute::Plain
    }
}

/// What placing memory inside an existing placeholder does: `(commit, protection)`. Only a type with
/// `MEM_COMMIT` commits, and then the placeholder is replaced by private memory. Without it (like
/// `0x20406000`) the placeholder is cut to the range and kept as one, because the title maps physical
/// pages into it afterwards. The Windows protection is the console one when `VirtualAlloc2` accepts
/// it, else `PAGE_READWRITE`.
pub fn placement_plan(allocation_type: u32, protect: u32) -> (bool, u32) {
    (allocation_type & MEM_COMMIT != 0, if usable_protection(protect) { protect } else { PAGE_READWRITE })
}

/// The granularity the console bits of `allocationType` give: 64 KiB, 2 MiB for large pages alone, else 4 KiB.
pub fn granularity_of(allocation_type: u32) -> usize {
    if allocation_type & PAGES_64K == PAGES_64K {
        0x1_0000
    } else if allocation_type & PAGES_LARGE != 0 {
        0x20_0000
    } else {
        0x1000
    }
}

/// `[address, address + size)` widened to `granularity`: the address rounded down and the end up.
/// `None` when the end wraps around or the size is 0.
pub fn round_range(address: usize, size: usize, granularity: usize) -> Option<(usize, usize)> {
    let start = address & !(granularity - 1);
    let end = address.checked_add(size)?.checked_next_multiple_of(granularity)?;
    (size != 0).then_some((start, end - start))
}

/// What `XMemVirtualAlloc` hands to `VirtualAlloc` for a console call.
#[derive(Debug, PartialEq, Eq)]
pub struct VirtualAllocPlan {
    pub allocation_type: u32,
    pub protect: u32,
    /// Console type bits were removed (to log).
    pub type_changed: bool,
    /// A console protection was replaced by `PAGE_READWRITE` (to log).
    pub protect_replaced: bool,
}

/// Keeps only PC type bits. A protection that cannot apply to committed pages becomes
/// `PAGE_READWRITE`, and one that cannot apply to a bare reservation becomes `PAGE_NOACCESS`.
pub fn virtual_alloc_plan(allocation_type: u32, protect: u32) -> VirtualAllocPlan {
    let pc_type = allocation_type & PC_ALLOCATION_TYPES;
    let mut plan = VirtualAllocPlan { allocation_type: pc_type, protect, type_changed: pc_type != allocation_type, protect_replaced: false };
    if !usable_protection(protect) {
        if pc_type & MEM_COMMIT != 0 {
            plan.protect = PAGE_READWRITE;
            plan.protect_replaced = true;
        } else {
            plan.protect = PAGE_NOACCESS;
        }
    }
    plan
}

/// The most fixed addresses one arena commit probes before it gives up on the low window. Each
/// probe is a system call on the title's allocating thread, and a small request in a full window
/// would otherwise try every 64 KiB base below 16 GiB (262,143 of them) before falling back.
pub const PROBE_LIMIT: usize = 256;

/// Addresses to try, in order, for a fixed-address reservation of `size` bytes below
/// `highest_end`, aligned on `align` (at least 64 KiB, as `VirtualAlloc2` needs): starting
/// just under `highest_end` and stepping down to `align`, at most `PROBE_LIMIT` of them. When the
/// window holds more aligned bases than that, the step grows to a multiple of `align` so the
/// probes still span the whole window (a gap narrower than the step can then be missed: the
/// caller falls back to any address, as it did before there was a probe). Pure and platform
/// independent so the choice can be tested with a fake allocator; `engine::Api` calls each
/// address in turn with the real fixed-address reserve.
pub fn probe_addresses(size: usize, highest_end: usize, align: usize) -> Vec<usize> {
    let align = align.max(0x1_0000);
    if size == 0 || size > highest_end {
        return Vec::new();
    }
    let top = (highest_end - size) & !(align - 1);
    let step = (top / align).div_ceil(PROBE_LIMIT).max(1) * align;
    let mut at = top;
    let mut out = Vec::new();
    while at >= align && out.len() < PROBE_LIMIT {
        out.push(at);
        if at < step {
            break;
        }
        at -= step;
    }
    out
}

/// Runs `probe_addresses` against `attempt` (the engine's fixed-address reserve in production,
/// a fake in tests), stopping at the first address `attempt` accepts. `attempt` must leave no
/// partial state behind when it refuses an address (the real `VirtualAlloc2` call does not).
pub fn probe_reserve(size: usize, highest_end: usize, align: usize, mut attempt: impl FnMut(usize) -> bool) -> Option<usize> {
    probe_addresses(size, highest_end, align).into_iter().find(|&at| attempt(at))
}

/// Which attempt placed a reservation `Engine::reserve` made without an address: above the low
/// window the arenas need (first choice now), or anywhere (the only, previous behaviour, kept as
/// the fallback).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlainPlacement {
    Above,
    Anywhere,
}

/// Runs `try_above` then, only when it is refused, `try_anywhere`: the order an address-less
/// reservation without a console commit bit now uses (the title's own plain windows, like its two
/// 24 GiB "anywhere" reservations). Under Wine those used to land low and fill the window
/// `reserve_below` needs for the arenas; asking them to go above `GPU_LOW_END` first keeps that
/// window free, matching the layout such reservations get on Windows. `try_above`'s error is
/// discarded (the caller logs it), the same as `commit_region`'s own probe fallback does for its
/// skipped attempts.
pub fn reserve_plain<T, E>(mut try_above: impl FnMut() -> Result<T, E>, mut try_anywhere: impl FnMut() -> Result<T, E>) -> Result<(T, PlainPlacement), E> {
    match try_above() {
        Ok(at) => Ok((at, PlainPlacement::Above)),
        Err(_) => try_anywhere().map(|at| (at, PlainPlacement::Anywhere)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GRAPHICS: u64 = 1 << 29;
    const PAGE_64K: u64 = 1 << 14;

    #[test]
    fn alignment_follows_the_fields() {
        assert_eq!(alignment_of(0), 16);
        assert_eq!(alignment_of(4 << 24), 16);
        assert_eq!(alignment_of(5 << 24), 32);
        assert_eq!(alignment_of(GRAPHICS), 0x1000);
        assert_eq!(alignment_of(PAGE_64K), 0x10000);
        assert_eq!(alignment_of(2 << 14), 0x40_0000);
        assert_eq!(alignment_of(3 << 14), 0x1000);
        assert_eq!(alignment_of(GRAPHICS | (16 << 24)), 0x10000);
        assert_eq!(alignment_of(31 << 24), 1 << 31);
    }

    #[test]
    fn page_backed_blocks() {
        assert!(!uses_pages(0xF_FFFF, 0));
        assert!(uses_pages(0x10_0000, 0));
        assert!(uses_pages(1, GRAPHICS));
        assert!(uses_pages(1, PAGE_64K));
    }

    #[test]
    fn sizes_that_wrap_are_refused() {
        assert_eq!(total_bytes(1000, 32), Some(1056));
        assert_eq!(total_bytes(usize::MAX, 16), None);
        assert_eq!(total_bytes(usize::MAX - 63, 1 << 31), None);
    }

    #[test]
    fn the_user_address_leaves_room_for_the_header() {
        assert_eq!(user_address(0x1000, 16), 0x1020);
        assert_eq!(user_address(0x1008, 32), 0x1020);
        assert_eq!(user_address(0x10000, 0x10000), 0x20000);
    }

    #[test]
    fn the_console_requests_pick_their_route() {
        assert_eq!(virtual_route(0x2040_6000), VirtualRoute::Placeholder);
        assert_eq!(virtual_route(0x42000), VirtualRoute::Placeholder);
        assert_eq!(virtual_route(0x2040_3000), VirtualRoute::Pool, "64 KiB pages, reserve and commit");
        assert_eq!(virtual_route(0x2000_3000), VirtualRoute::Pool, "large pages, the arenas");
        assert_eq!(virtual_route(0x2040_1000), VirtualRoute::Pool, "64 KiB pages, commit alone");
        assert_eq!(virtual_route(0x2040_2000), VirtualRoute::Placeholder, "the title's plain 64 KiB reserve");
        assert_eq!(virtual_route(0x2000), VirtualRoute::Placeholder, "a plain MEM_RESERVE");
        assert_eq!(virtual_route(0x2000_2000), VirtualRoute::Placeholder, "a large page reserve without commit");
        assert_eq!(virtual_route(0x3000), VirtualRoute::Plain, "reserve and commit");
        assert_eq!(virtual_route(0x2040_4000), VirtualRoute::Plain, "no MEM_RESERVE, no MEM_COMMIT");
        assert_eq!(virtual_route(0x1000), VirtualRoute::Plain, "a PC commit");
        assert_eq!(virtual_route(0x40000), VirtualRoute::Plain, "a placeholder needs MEM_RESERVE");
        assert_eq!(virtual_route(0x8_0000), VirtualRoute::Plain, "MEM_RESET");
        assert_eq!(PAGES_64K | MAP_LATER | MEM_RESERVE, 0x2040_6000);
    }

    #[test]
    fn placement_commits_only_with_the_commit_bit() {
        assert_eq!(placement_plan(0x2040_6000, 0x27), (false, 0x04), "the measured call keeps its placeholder");
        assert_eq!(placement_plan(0x2040_6000, 0x02), (false, 0x02));
        assert_eq!(placement_plan(0x2040_7000, 0x01), (true, 0x04), "MEM_COMMIT");
        assert_eq!(placement_plan(0x1000, 0x40), (true, 0x40));
    }

    #[test]
    fn granularity_follows_the_console_bits() {
        assert_eq!(granularity_of(0x2040_3000), 0x1_0000);
        assert_eq!(granularity_of(0x2040_6000), 0x1_0000);
        assert_eq!(granularity_of(0x2000_3000), 0x20_0000);
        assert_eq!(granularity_of(0x3000), 0x1000);
    }

    #[test]
    fn a_range_is_widened_to_the_granularity() {
        assert_eq!(round_range(0x1_2345, 0x10, 0x1_0000), Some((0x1_0000, 0x1_0000)));
        assert_eq!(round_range(0x1_FFF0, 0x20, 0x1_0000), Some((0x1_0000, 0x2_0000)));
        assert_eq!(round_range(0, 0, 0x1_0000), None);
        assert_eq!(round_range(usize::MAX - 5, 100, 0x1_0000), None);
    }

    #[test]
    fn console_type_bits_are_dropped() {
        let plan = virtual_alloc_plan(0x2000_0000 | 0x1000 | 0x2000, 0x04);
        assert_eq!(plan, VirtualAllocPlan { allocation_type: 0x3000, protect: 0x04, type_changed: true, protect_replaced: false });
        assert!(!virtual_alloc_plan(0x3000, 0x04).type_changed);
    }

    #[test]
    fn probe_addresses_step_down_from_just_under_the_window() {
        // 1 MiB arena, window 16 MiB, 1 MiB alignment: candidates at 15, 14, ..., 1 MiB.
        let addrs = probe_addresses(1 << 20, 16 << 20, 1 << 20);
        assert_eq!(addrs, (1..=15).rev().map(|n| n << 20).collect::<Vec<_>>());
    }

    #[test]
    fn probe_addresses_is_empty_when_the_arena_does_not_fit_the_window() {
        assert_eq!(probe_addresses(32 << 20, 16 << 20, 0x1_0000), Vec::<usize>::new(), "bigger than the whole window");
        assert_eq!(probe_addresses(0, 16 << 20, 0x1_0000), Vec::<usize>::new(), "zero size");
    }

    #[test]
    fn probe_addresses_are_bounded_and_still_span_the_window() {
        // A 64 KiB request below 16 GiB has 262,143 aligned bases: only PROBE_LIMIT are tried,
        // evenly spaced from just under the window down towards its start.
        let addrs = probe_addresses(0x1_0000, 16 << 30, 0x1_0000);
        assert_eq!(addrs.len(), PROBE_LIMIT);
        assert_eq!(addrs[0], (16 << 30) - 0x1_0000);
        assert!(addrs.windows(2).all(|w| w[0] - w[1] == addrs[0] - addrs[1]), "one step throughout");
        assert!(addrs.iter().all(|a| a % 0x1_0000 == 0 && *a >= 0x1_0000));
        assert!(*addrs.last().unwrap() < 1 << 27, "the last probe is near the start of the window");
        // A window with fewer bases than the limit is walked base by base, as before.
        assert_eq!(probe_addresses(1 << 20, 16 << 20, 1 << 20).len(), 15);
    }

    #[test]
    fn probe_addresses_rounds_the_alignment_up_to_64_kib() {
        // align below 64 KiB is still rounded to 64 KiB steps.
        let addrs = probe_addresses(0x1000, 0x30000, 0x1000);
        assert_eq!(addrs, vec![0x20000, 0x10000]);
    }

    #[test]
    fn probe_reserve_stops_at_the_first_address_the_fake_allocator_accepts() {
        let mut tried = Vec::new();
        let taken = [15usize << 20, 14 << 20]; // the fake pretends these are already in use
        let got = probe_reserve(1 << 20, 16 << 20, 1 << 20, |at| {
            tried.push(at);
            !taken.contains(&at)
        });
        assert_eq!(got, Some(13 << 20), "the first address not in `taken`");
        assert_eq!(tried, vec![15 << 20, 14 << 20, 13 << 20], "probed in descending order, stopped on success");
    }

    #[test]
    fn probe_reserve_is_none_when_the_fake_allocator_refuses_every_address() {
        let mut calls = 0;
        let got = probe_reserve(1 << 20, 4 << 20, 1 << 20, |_| {
            calls += 1;
            false
        });
        assert_eq!(got, None);
        assert_eq!(calls, 3, "every candidate below 4 MiB for a 1 MiB arena: 3, 2, 1 MiB");
    }

    #[test]
    fn reserve_plain_asks_above_first_and_does_not_touch_anywhere_when_it_works() {
        let mut anywhere_calls = 0;
        let got = reserve_plain::<_, &str>(|| Ok(0x4_0000_0000usize), || {
            anywhere_calls += 1;
            Ok(0x1000)
        });
        assert_eq!(got, Ok((0x4_0000_0000, PlainPlacement::Above)));
        assert_eq!(anywhere_calls, 0, "the fallback is not tried when the first attempt succeeds");
    }

    #[test]
    fn reserve_plain_falls_back_to_anywhere_when_above_is_refused() {
        let got = reserve_plain(|| Err::<usize, _>("no room above"), || Ok::<usize, &str>(0x1000));
        assert_eq!(got, Ok((0x1000, PlainPlacement::Anywhere)));
    }

    #[test]
    fn reserve_plain_fails_when_both_attempts_are_refused() {
        let got = reserve_plain(|| Err::<usize, _>("above refused"), || Err::<usize, _>("anywhere refused"));
        assert_eq!(got, Err("anywhere refused"), "the anywhere attempt's error is the one reported");
    }

    #[test]
    fn an_unusable_protection_depends_on_the_commit_bit() {
        let committed = virtual_alloc_plan(0x3000, 0x1);
        assert_eq!((committed.protect, committed.protect_replaced), (0x04, true));
        let reserved = virtual_alloc_plan(0x2000, 0x1);
        assert_eq!((reserved.protect, reserved.protect_replaced), (0x01, false));
        // The low byte decides, so modifier flags on a good protection keep it.
        assert_eq!(virtual_alloc_plan(0x3000, 0x104).protect, 0x104);
        assert!(usable_protection(0x10) && usable_protection(0x80));
        assert!(!usable_protection(0) && !usable_protection(0x1) && !usable_protection(0x100));
    }
}
