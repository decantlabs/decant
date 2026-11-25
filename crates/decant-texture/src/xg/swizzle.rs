//! The swizzle mode the console library suggests for a texture (`docs/CONSOLE_LIBS.md` 1.5).

use super::formats::bits_per_pixel;

/// `XG_BIND_FLAG` values the rules look at.
pub const BIND_RENDER_TARGET: u32 = 0x20;
pub const BIND_DEPTH_STENCIL: u32 = 0x40;
pub const BIND_UNORDERED_ACCESS: u32 = 0x80;
/// The "64 KiB D layout" misc flag.
pub const MISC_64K_D: u32 = 0x800_0000;

pub const MODE_LINEAR: u32 = 0;
pub const MODE_4K_S: u32 = 5;
pub const MODE_64K_D: u32 = 10;
pub const MODE_4K_S_X: u32 = 21;
pub const MODE_64K_Z_X: u32 = 24;
pub const MODE_64K_R_X: u32 = 27;

/// The input of [`compute_optimal_swizzle_mode`], with the types of the console prototype.
#[derive(Clone, Copy, Debug)]
pub struct SwizzleQuery {
    /// `XG_RESOURCE_DIMENSION`: 1 buffer, 2 1D, 3 2D, 4 3D.
    pub dimension: u32,
    pub format: u32,
    pub width: u64,
    pub height: u32,
    pub depth_or_array: u32,
    pub sample_count: u32,
    pub bind_flags: u32,
    pub misc_flags: u32,
}

/// "Small": the format has a bit size and `bits × width × height × k`, rounded down to a multiple of 8, is at most 0x80000
/// (bits, not bytes).
fn small(q: &SwizzleQuery, k: u32) -> bool {
    let bpp = u128::from(bits_per_pixel(q.format));
    if bpp == 0 {
        return false;
    }
    let bits = bpp * u128::from(q.width) * u128::from(q.height) * u128::from(k);
    bits / 8 * 8 <= 0x80000
}

/// The suggested mode, rules in the order of 1.5 (the default 22000 behaviour).
pub fn compute_optimal_swizzle_mode(q: &SwizzleQuery) -> u32 {
    if q.dimension == 1 {
        return MODE_LINEAR;
    }
    if q.bind_flags & BIND_DEPTH_STENCIL != 0 {
        return MODE_64K_Z_X;
    }
    if q.dimension != 3 && q.dimension != 4 {
        return MODE_LINEAR;
    }
    let rt = q.bind_flags & BIND_RENDER_TARGET != 0;
    let uav = q.bind_flags & BIND_UNORDERED_ACCESS != 0;
    let d27 = q.misc_flags & MISC_64K_D != 0;
    if q.dimension == 3 {
        if d27 {
            MODE_64K_D
        } else if rt {
            MODE_64K_R_X
        } else if !uav {
            MODE_4K_S
        } else if q.sample_count == 1 && small(q, 1) {
            MODE_4K_S_X
        } else {
            MODE_64K_R_X
        }
    } else if d27 {
        MODE_64K_D
    } else if !rt && !uav {
        MODE_4K_S
    } else if !rt && small(q, q.depth_or_array) {
        MODE_4K_S_X
    } else {
        MODE_64K_R_X
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_compares_bits() {
        let q = SwizzleQuery { dimension: 3, format: 28, width: 128, height: 128, depth_or_array: 1, sample_count: 1, bind_flags: 0x88, misc_flags: 0 };
        assert_eq!(compute_optimal_swizzle_mode(&q), 21);
        assert_eq!(compute_optimal_swizzle_mode(&SwizzleQuery { width: 129, ..q }), 27);
        assert_eq!(compute_optimal_swizzle_mode(&SwizzleQuery { format: 0, ..q }), 27);
    }

    #[test]
    fn depth_precedes_the_dimension() {
        let q = SwizzleQuery { dimension: 2, format: 28, width: 4096, height: 1, depth_or_array: 1, sample_count: 1, bind_flags: 0x40, misc_flags: 0 };
        assert_eq!(compute_optimal_swizzle_mode(&q), 24);
        assert_eq!(compute_optimal_swizzle_mode(&SwizzleQuery { dimension: 1, ..q }), 0);
    }
}
