//! Finding frames by bit position in a buffer of whole packets of one stream.
//!
//! Positions count bits from the start of the buffer, packet headers included, as the console's XMA contexts do. A frame
//! that does not fit its packet goes on behind the header of the next packet of its stream. The packet header tells how many
//! packets of other streams lie in between.

use crate::{HEADER_BITS, PACKET_BITS, PACKET_BYTES};

/// The fields of a packet header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PacketHeader {
    /// Frames that begin in the packet.
    pub frames: u32,
    /// Bits at the start of the payload that end the frame of the previous packet.
    pub first_frame_bit: u32,
    /// Packets of other streams that follow before the next packet of this stream.
    pub skip: u32,
}

/// The header of packet `index` of `data`, if `data` holds it.
pub fn packet_header(data: &[u8], index: usize) -> Option<PacketHeader> {
    let at = index.checked_mul(PACKET_BYTES)?;
    let raw = data.get(at..at.checked_add(4)?)?;
    let word = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    Some(PacketHeader {
        frames: word >> 26,
        first_frame_bit: (word >> 11) & 0x7FFF,
        skip: word & 0xFF,
    })
}

/// One frame with its bits gathered out of its packets.
#[derive(Clone, Debug)]
pub struct Frame {
    /// The frame from its length field on, most significant bit first.
    pub data: Vec<u8>,
    /// Its length in bits.
    pub bits: usize,
    /// Where it starts in the buffer.
    pub start: usize,
    /// Where the next frame starts, `None` when the buffer holds no further frame.
    pub next: Option<usize>,
}

/// Width of the length field at the start of a frame.
const LENGTH_BITS: usize = 15;
/// A length field of all ones marks the rest of the packet as padding.
const PADDING: usize = 0x7FFF;

/// The packet of this stream after packet `index`, if the buffer has it.
fn following_packet(data: &[u8], index: usize) -> Option<usize> {
    let next = index + 1 + packet_header(data, index)?.skip as usize;
    (next * PACKET_BYTES < data.len()).then_some(next)
}

/// Where the payload of `packet` begins its first own frame.
fn first_start(data: &[u8], packet: usize) -> Option<usize> {
    Some(packet * PACKET_BITS + HEADER_BITS + packet_header(data, packet)?.first_frame_bit as usize)
}

/// Walks the bits of one stream. At the end of a packet it jumps over the headers and packets of other streams.
struct StreamBits<'a> {
    data: &'a [u8],
    pos: usize,
}

impl StreamBits<'_> {
    fn bit(&mut self) -> Option<bool> {
        if self.pos.is_multiple_of(PACKET_BITS) {
            let last = (self.pos / PACKET_BITS).checked_sub(1)?;
            self.pos = following_packet(self.data, last)? * PACKET_BITS + HEADER_BITS;
        }
        let byte = *self.data.get(self.pos / 8)?;
        let bit = byte >> (7 - self.pos % 8) & 1 == 1;
        self.pos += 1;
        Some(bit)
    }

    fn number(&mut self, width: usize) -> Option<usize> {
        let mut value = 0;
        for _ in 0..width {
            value = value << 1 | usize::from(self.bit()?);
        }
        Some(value)
    }
}

/// Whether a length field read as `length` starts a real frame.
fn is_frame_length(length: usize) -> bool {
    length != PADDING && length > LENGTH_BITS
}

/// Moves `pos` forward to a position where a frame starts: past padding, past packets that hold no frame of their own, past
/// the end of a packet. `None` when the buffer ends first.
fn settle(data: &[u8], mut pos: usize) -> Option<usize> {
    let packets = data.len() / PACKET_BYTES;
    for _ in 0..=packets {
        let packet = pos / PACKET_BITS;
        if packet >= packets {
            return None;
        }
        let offset = pos % PACKET_BITS;
        if offset >= HEADER_BITS && offset + LENGTH_BITS <= PACKET_BITS {
            let length = StreamBits { data, pos }.number(LENGTH_BITS)?;
            if is_frame_length(length) {
                return Some(pos);
            }
        }
        let next = following_packet(data, packet)?;
        pos = first_start(data, next)?;
        if packet_header(data, next)?.first_frame_bit as usize >= PACKET_BITS - HEADER_BITS {
            // The whole packet continues one frame, so look at the packet after the next.
            pos = (next + 1) * PACKET_BITS - 1;
        }
    }
    None
}

/// Where the first frame of a buffer starts.
pub fn first_frame(data: &[u8]) -> Option<usize> {
    settle(data, first_start(data, 0)?)
}

/// The frame that starts at bit `start`. `None` when there is none, because of padding or because the buffer ends inside it.
pub fn read_frame(data: &[u8], start: usize) -> Option<Frame> {
    let mut frame = Frame {
        data: Vec::new(),
        bits: 0,
        start: 0,
        next: None,
    };
    read_frame_into(data, start, &mut frame).then_some(frame)
}

/// Reads the frame that starts at bit `start` into `frame`, reusing `frame.data`.
/// Returns `false` when there is no complete frame at `start`.
pub fn read_frame_into(data: &[u8], start: usize, frame: &mut Frame) -> bool {
    let mut walker = StreamBits { data, pos: start };
    let Some(bits) = walker.number(LENGTH_BITS) else {
        return false;
    };
    if !is_frame_length(bits) {
        return false;
    }
    let bytes = bits.div_ceil(8);
    frame.data.clear();
    frame.data.resize(bytes, 0);
    walker.pos = start;
    let mut copied = 0usize;
    while copied < bits {
        if walker.pos.is_multiple_of(PACKET_BITS) {
            let Some(last) = (walker.pos / PACKET_BITS).checked_sub(1) else {
                return false;
            };
            let Some(next_pkt) = following_packet(data, last) else {
                return false;
            };
            walker.pos = next_pkt * PACKET_BITS + HEADER_BITS;
        }
        let pkt_rem = PACKET_BITS - (walker.pos % PACKET_BITS);
        let chunk = (bits - copied).min(pkt_rem);
        let mut done = 0usize;
        if copied.is_multiple_of(8) && chunk >= 8 {
            let nbytes = chunk / 8;
            let src_byte = walker.pos / 8;
            let bit_off = walker.pos % 8;
            if bit_off == 0 {
                if src_byte + nbytes > data.len() {
                    return false;
                }
                frame.data[copied / 8..copied / 8 + nbytes]
                    .copy_from_slice(&data[src_byte..src_byte + nbytes]);
                done = nbytes * 8;
            } else if src_byte + nbytes < data.len() {
                let dst = &mut frame.data[copied / 8..copied / 8 + nbytes];
                let src = &data[src_byte..src_byte + nbytes + 1];
                let rshift = 8 - bit_off;
                for (d, w) in dst.iter_mut().zip(src.windows(2)) {
                    *d = (w[0] << bit_off) | (w[1] >> rshift);
                }
                done = nbytes * 8;
            }
        }
        walker.pos += done;
        copied += done;
        while done < chunk {
            let Some(bit) = walker.bit() else {
                return false;
            };
            if bit {
                frame.data[copied / 8] |= 0x80 >> (copied % 8);
            }
            copied += 1;
            done += 1;
        }
    }
    let end = walker.pos;
    let more_in_packet = frame.data[(bits - 1) / 8] >> (7 - (bits - 1) % 8) & 1 == 1;
    let last_packet = (end - 1) / PACKET_BITS;
    let crossed = last_packet != start / PACKET_BITS;
    let next = if more_in_packet || crossed {
        // The next frame follows right behind this one.
        settle(data, end)
    } else {
        // This was the last frame of its packet, so the next one starts in the following packet of the stream.
        following_packet(data, last_packet)
            .and_then(|packet| settle(data, first_start(data, packet)?))
    };
    frame.bits = bits;
    frame.start = start;
    frame.next = next;
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::pack;

    /// Packs frames of the given lengths with an empty body. A length of 0 pads to the next packet.
    fn packets(lens: &[usize]) -> Vec<u8> {
        let frames: Vec<Option<Vec<bool>>> = lens
            .iter()
            .map(|&len| {
                (len != 0).then(|| {
                    (0..len)
                        .map(|b| b < 15 && len >> (14 - b) & 1 == 1)
                        .collect()
                })
            })
            .collect();
        pack(&frames)
    }

    fn walk(data: &[u8]) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        let mut at = first_frame(data);
        while let Some(pos) = at {
            let frame = read_frame(data, pos).expect("a frame where the walk says");
            out.push((frame.start, frame.bits));
            at = frame.next;
        }
        out
    }

    #[test]
    fn headers_are_read() {
        let data = packets(&[100, 200]);
        assert_eq!(
            packet_header(&data, 0),
            Some(PacketHeader {
                frames: 2,
                first_frame_bit: 0,
                skip: 0
            })
        );
        assert_eq!(packet_header(&data, 1), None);
    }

    #[test]
    fn header_fields_sit_where_the_layout_says() {
        let mut data = vec![0u8; PACKET_BYTES];
        let word: u32 = 37 << 26 | 12345 << 11 | 0b101 << 8 | 200;
        data[..4].copy_from_slice(&word.to_be_bytes());
        assert_eq!(
            packet_header(&data, 0),
            Some(PacketHeader {
                frames: 37,
                first_frame_bit: 12345,
                skip: 200
            })
        );
    }

    #[test]
    fn frames_are_walked_across_packets() {
        let data = packets(&[5000; 7]);
        let walked = walk(&data);
        assert_eq!(walked.len(), 7);
        assert_eq!(walked[0], (32, 5000));
        assert_eq!(walked[1].0, 5032);
        assert_eq!(walked[3].0, 15032);
        // The fourth frame ends 15032 + 5000 - 16384 = 3648 bits into the second payload.
        assert_eq!(walked[4].0, PACKET_BITS + 32 + 3648);
        assert_eq!(packet_header(&data, 1).unwrap().first_frame_bit, 3648);
    }

    #[test]
    fn padding_moves_to_the_next_packet() {
        let data = packets(&[8000, 8000, 0, 300]);
        assert_eq!(
            walk(&data),
            [(32, 8000), (8032, 8000), (PACKET_BITS + 32, 300)]
        );
    }

    #[test]
    fn the_walk_ends_with_the_buffer() {
        let data = packets(&[10000, 10000]);
        let walked = walk(&data);
        assert_eq!(walked.len(), 2);
        assert_eq!(read_frame(&data, walked[1].0).unwrap().next, None);
        assert!(first_frame(&[]).is_none());
    }

    #[test]
    fn packets_of_other_streams_are_skipped() {
        // Packet 0 holds a frame that crosses into the next packet of the stream, which is packet 2. Packet 1 belongs to
        // another stream.
        let mut data = vec![0xAAu8; PACKET_BYTES * 3];
        let payload = PACKET_BITS - HEADER_BITS;
        let len = payload + 100;
        // Packet 0: one frame, which starts at the payload start and goes on in packet 2. Its skip field is 1.
        let h0: u32 = 1 << 26 | 1;
        data[..4].copy_from_slice(&h0.to_be_bytes());
        let mut bits = vec![false; len];
        for (i, b) in bits.iter_mut().take(15).enumerate() {
            *b = len >> (14 - i) & 1 == 1;
        }
        for (i, &b) in bits.iter().enumerate() {
            let pos = if i < payload {
                HEADER_BITS + i
            } else {
                2 * PACKET_BITS + HEADER_BITS + (i - payload)
            };
            let mask = 0x80u8 >> (pos % 8);
            if b {
                data[pos / 8] |= mask;
            } else {
                data[pos / 8] &= !mask;
            }
        }
        // Packet 2: the frame ends 100 bits into the payload.
        let h2: u32 = 100 << 11;
        data[2 * PACKET_BYTES..2 * PACKET_BYTES + 4].copy_from_slice(&h2.to_be_bytes());
        let frame = read_frame(&data, 32).unwrap();
        assert_eq!(frame.bits, len);
        assert_eq!(first_frame(&data), Some(32));
    }

    #[test]
    fn a_frame_cut_by_the_buffer_is_none() {
        // A frame of 20000 bits crosses into the second packet. Without that packet it cannot be read.
        let data = packets(&[20000]);
        assert_eq!(data.len(), 2 * PACKET_BYTES);
        assert!(read_frame(&data, 32).is_some());
        assert!(read_frame(&data[..PACKET_BYTES], 32).is_none());
        assert!(read_frame(&data, 0).is_none());
    }

    #[test]
    fn read_frame_into_reuses_buffer_and_matches_read_frame() {
        let data = packets(&[5003, 5007, 17001, 400]);
        let mut reused = Frame {
            data: Vec::new(),
            bits: 0,
            start: 0,
            next: None,
        };
        let mut at = first_frame(&data);
        while let Some(pos) = at {
            let expected = read_frame(&data, pos).unwrap();
            assert!(read_frame_into(&data, pos, &mut reused));
            assert_eq!(reused.data, expected.data);
            assert_eq!(reused.bits, expected.bits);
            assert_eq!(reused.start, expected.start);
            assert_eq!(reused.next, expected.next);
            at = reused.next;
        }
    }
}
