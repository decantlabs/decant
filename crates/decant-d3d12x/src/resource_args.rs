//! Resource pointers inside the D3D12 structures a console title passes to a relayed call.
//!
//! A placed resource is handed to the title as a console proxy (see `proxy`), so a structure that
//! names it (a resource barrier, a texture copy location) must reach the PC runtime with the PC
//! object in that field. The title's own structure is never written: the caller copies it, and
//! these functions rewrite the copy. Layouts are the public d3d12.h ones; `proxy` checks the two
//! sizes against the `windows` crate at compile time for Windows builds.
//!
//! Pointers are handled as `u64` so the functions are plain byte code and can be unit tested on any
//! host. The `unwrap` argument maps a proxy to its PC object and any other pointer to itself.

/// `D3D12_RESOURCE_BARRIER`: `Type` (u32) at 0, `Flags` (u32) at 4, the union at 8 (24 bytes).
pub const BARRIER_BYTES: usize = 32;
/// `D3D12_TEXTURE_COPY_LOCATION`: `pResource` (pointer) at 0, `Type` at 8, the union at 16 (32 bytes).
pub const COPY_LOCATION_BYTES: usize = 48;

const BARRIER_TYPE_TRANSITION: u32 = 0;
const BARRIER_TYPE_ALIASING: u32 = 1;
const BARRIER_TYPE_UAV: u32 = 2;

/// The union of a barrier starts here.
const BARRIER_UNION: usize = 8;

/// Replaces the pointer stored at `at` with `unwrap(pointer)`.
#[inline]
fn rewrite(bytes: &mut [u8], at: usize, unwrap: &impl Fn(u64) -> u64) {
    let orig = u64::from_le_bytes([
        bytes[at],
        bytes[at + 1],
        bytes[at + 2],
        bytes[at + 3],
        bytes[at + 4],
        bytes[at + 5],
        bytes[at + 6],
        bytes[at + 7],
    ]);
    let mapped = unwrap(orig);
    if mapped != orig {
        bytes[at..at + 8].copy_from_slice(&mapped.to_le_bytes());
    }
}

/// Unwraps the resource pointers of one `D3D12_RESOURCE_BARRIER` in place. A transition and a UAV
/// barrier hold one resource (`Transition.pResource`, `UAV.pResource`, both at the start of the
/// union), an aliasing barrier holds two (`pResourceBefore`, `pResourceAfter`). Any other type has
/// no resource and is left as it is.
#[inline]
pub fn unwrap_barrier(barrier: &mut [u8; BARRIER_BYTES], unwrap: impl Fn(u64) -> u64) {
    match u32::from_le_bytes([barrier[0], barrier[1], barrier[2], barrier[3]]) {
        BARRIER_TYPE_TRANSITION | BARRIER_TYPE_UAV => rewrite(barrier, BARRIER_UNION, &unwrap),
        BARRIER_TYPE_ALIASING => {
            rewrite(barrier, BARRIER_UNION, &unwrap);
            rewrite(barrier, BARRIER_UNION + 8, &unwrap);
        }
        _ => {}
    }
}

/// Unwraps the resource of one `D3D12_TEXTURE_COPY_LOCATION` in place (`pResource` at 0).
#[inline]
pub fn unwrap_copy_location(location: &mut [u8; COPY_LOCATION_BYTES], unwrap: impl Fn(u64) -> u64) {
    rewrite(location, 0, &unwrap);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put_u64(bytes: &mut [u8], at: usize, value: u64) {
        bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn get_u64(bytes: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
    }

    /// A stand-in for a proxy table: 0x1000 is a proxy whose PC object is 0x2000, anything else is
    /// not one of ours.
    fn unwrap_table(p: u64) -> u64 {
        if p == 0x1000 { 0x2000 } else { p }
    }

    fn barrier(ty: u32, first: u64, second: u64) -> [u8; BARRIER_BYTES] {
        let mut b = [0u8; BARRIER_BYTES];
        b[0..4].copy_from_slice(&ty.to_le_bytes());
        b[4..8].copy_from_slice(&0x55u32.to_le_bytes());
        put_u64(&mut b, 8, first);
        put_u64(&mut b, 16, second);
        b
    }

    #[test]
    fn a_transition_barrier_gets_its_resource_unwrapped_and_nothing_else_changes() {
        let mut b = barrier(BARRIER_TYPE_TRANSITION, 0x1000, 0);
        // Subresource at 16, StateBefore at 20, StateAfter at 24: the second word of the union
        b[16..20].copy_from_slice(&7u32.to_le_bytes());
        b[20..24].copy_from_slice(&0x4u32.to_le_bytes());
        b[24..28].copy_from_slice(&0x8u32.to_le_bytes());
        let before = b;
        unwrap_barrier(&mut b, unwrap_table);
        assert_eq!(get_u64(&b, 8), 0x2000);
        assert_eq!(&b[0..8], &before[0..8], "type and flags");
        assert_eq!(&b[16..], &before[16..], "subresource and states");
    }

    #[test]
    fn a_uav_barrier_gets_its_resource_unwrapped() {
        let mut b = barrier(BARRIER_TYPE_UAV, 0x1000, 0);
        unwrap_barrier(&mut b, unwrap_table);
        assert_eq!(get_u64(&b, 8), 0x2000);
    }

    #[test]
    fn an_aliasing_barrier_gets_both_resources_unwrapped() {
        let mut b = barrier(BARRIER_TYPE_ALIASING, 0x1000, 0x3000);
        unwrap_barrier(&mut b, unwrap_table);
        assert_eq!(get_u64(&b, 8), 0x2000);
        assert_eq!(
            get_u64(&b, 16),
            0x3000,
            "a PC object or a foreign pointer stays as it is"
        );
    }

    #[test]
    fn a_barrier_without_a_resource_type_is_left_alone() {
        let mut b = barrier(7, 0x1000, 0x1000);
        let before = b;
        unwrap_barrier(&mut b, unwrap_table);
        assert_eq!(b, before);
    }

    #[test]
    fn a_copy_location_gets_its_resource_unwrapped_and_the_rest_is_kept() {
        let mut c = [0u8; COPY_LOCATION_BYTES];
        put_u64(&mut c, 0, 0x1000);
        c[8..12].copy_from_slice(&1u32.to_le_bytes()); // Type: SUBRESOURCE_INDEX
        put_u64(&mut c, 16, 0x0000_0000_0042_0000);
        unwrap_copy_location(&mut c, unwrap_table);
        assert_eq!(get_u64(&c, 0), 0x2000);
        assert_eq!(
            &c[8..],
            &{
                let mut rest = [0u8; COPY_LOCATION_BYTES - 8];
                rest[0..4].copy_from_slice(&1u32.to_le_bytes());
                put_u64(&mut rest, 8, 0x0000_0000_0042_0000);
                rest
            }[..]
        );
    }

    #[test]
    fn a_null_resource_stays_null() {
        let mut c = [0u8; COPY_LOCATION_BYTES];
        unwrap_copy_location(&mut c, unwrap_table);
        assert_eq!(get_u64(&c, 0), 0);
    }
}
