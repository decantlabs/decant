//! `XGComputeTileShape`: the shape of one 64 KiB tile (`docs/CONSOLE_LIBS.md`, section 1).

use super::XgError;
use super::formats::{Format, classify};

/// Three counts, as `XG_TILE_SHAPE`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TileShape {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

/// What the function writes: the shape in texels and in elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileShapes {
    pub texels: TileShape,
    pub elements: TileShape,
}

fn shape(width: u32, height: u32, depth: u32) -> TileShape {
    TileShape { width, height, depth }
}

/// The tile shape of a resource dimension (1 buffer, 2 1D, 3 2D, 4 3D), format, element size and sample count.
pub fn compute_tile_shape(dimension: u32, format: u32, bytes_per_element: u32, samples: u32) -> Result<TileShapes, XgError> {
    if dimension != 3 && samples != 1 {
        return Err(XgError::InvalidArg);
    }
    if format > 192 {
        return Err(XgError::InvalidArg);
    }
    match dimension {
        1 => {
            if format != 0 || bytes_per_element > 1 {
                return Err(XgError::InvalidArg);
            }
            Ok(TileShapes { texels: shape(0, 0, 0), elements: shape(65536, 1, 1) })
        }
        2 | 4 => Err(XgError::NotImpl),
        3 => {
            let (bpe, block) = if format == 0 {
                (bytes_per_element, None)
            } else {
                match classify(format) {
                    Format::Element { bpe, block } => (bpe, Some(block)),
                    Format::NotModelled => return Err(XgError::Fail),
                    Format::Unknown => return Err(XgError::InvalidArg),
                }
            };
            let (mut w, mut h) = match bpe {
                1 => (256, 256),
                2 => (256, 128),
                4 => (128, 128),
                8 => (128, 64),
                16 => (64, 64),
                _ => return Err(if format == 0 { XgError::InvalidArg } else { XgError::Fail }),
            };
            match samples {
                1 => {}
                2 => h /= 2,
                4 => {
                    w /= 2;
                    h /= 2;
                }
                8 => {
                    w /= 2;
                    h /= 4;
                }
                _ => return Err(XgError::InvalidArg),
            }
            let texels = match block {
                Some(b) => shape(w * b, h * b, 1),
                None => shape(0, 0, 0),
            };
            Ok(TileShapes { texels, elements: shape(w, h, 1) })
        }
        _ => Err(XgError::InvalidArg),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_few_shapes() {
        let s = compute_tile_shape(3, 71, 0, 1).unwrap();
        assert_eq!((s.texels, s.elements), (shape(512, 256, 1), shape(128, 64, 1)));
        assert_eq!(compute_tile_shape(3, 28, 0, 16), Err(XgError::InvalidArg));
        assert_eq!(compute_tile_shape(3, 0, 3, 1), Err(XgError::InvalidArg));
        assert_eq!(compute_tile_shape(3, 193, 0, 1), Err(XgError::InvalidArg));
        assert_eq!(compute_tile_shape(3, 45, 0, 1), Err(XgError::Fail));
        assert_eq!(compute_tile_shape(2, 28, 0, 2), Err(XgError::InvalidArg));
    }
}
