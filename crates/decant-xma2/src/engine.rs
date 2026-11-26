//! The decoding behind the C entry points, in safe Rust: one `decant-xma` decoder per XMA stream.
//!
//! A data block is a run of 2048 byte packets. Each packet belongs to one stream. The packet
//! header of a stream tells how many packets of other streams follow it, so a packet goes to the
//! first stream whose skip count is used up. The packets of a stream are gathered in a buffer of
//! its own (with the skip count cleared, since the packets are next to each other there) and its
//! frames are decoded as soon as they are complete. The streams hold two channels each, and the
//! output interleaves all channels, stream by stream.

use decant_xma::{Decoder, Frame, PACKET_BITS, PACKET_BYTES, first_frame, read_frame_into};

use crate::Format;

/// Why a format cannot be decoded.
#[derive(Debug, PartialEq, Eq)]
pub enum Unsupported {
    /// Only packets of 2048 bytes can be decoded. Carries the block size of the format.
    BlockSize(u16),
    /// `NumStreams` does not match the channel count (one stream per two channels).
    StreamCount { streams: u16, channels: u16 },
}

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Unsupported::BlockSize(size) => write!(f, "block size {size} is not 2048"),
            Unsupported::StreamCount { streams, channels } => {
                write!(f, "{streams} streams for {channels} channels")
            }
        }
    }
}

/// One XMA stream of one or two channels.
struct Stream {
    decoder: Decoder,
    channels: usize,
    /// The packets of this stream that are not consumed yet, next to each other.
    buf: Vec<u8>,
    /// Start bit of the last frame decoded.
    last: Option<usize>,
    /// Start bit of the next frame, when known.
    pos: Option<usize>,
    /// Reusable frame buffer to avoid per-frame heap allocations.
    frame: Frame,
    /// Decoded samples of this stream, interleaved, not yet merged into the output.
    out: Vec<i16>,
    /// Read cursor into `out`.
    out_pos: usize,
    /// Packets of other streams still to come before the next packet of this one.
    skip: u32,
    /// Number of the packet of the whole block that came last to this stream (0 for none yet).
    last_packet: u64,
}

impl Stream {
    fn new(channels: usize, rate: u32) -> Stream {
        Stream {
            decoder: Decoder::new(channels, rate),
            channels,
            buf: Vec::new(),
            last: None,
            pos: None,
            frame: Frame {
                data: Vec::new(),
                bits: 0,
                start: 0,
                next: None,
            },
            out: Vec::new(),
            out_pos: 0,
            skip: 0,
            last_packet: 0,
        }
    }

    fn add_packet(&mut self, packet: &[u8]) {
        let at = self.buf.len();
        self.buf.extend_from_slice(packet);
        // The skip count is the low byte of the header. Packets are adjacent in this buffer.
        self.buf[at + 3] = 0;
        self.skip = u32::from(packet[3]);
        self.decode_ready();
    }

    /// Decodes every frame that is complete in the buffer.
    fn decode_ready(&mut self) {
        loop {
            if self.pos.is_none() {
                self.pos = match self.last {
                    Some(last) => {
                        if read_frame_into(&self.buf, last, &mut self.frame) {
                            self.frame.next
                        } else {
                            None
                        }
                    }
                    None => first_frame(&self.buf),
                };
            }
            let Some(pos) = self.pos else {
                break;
            };
            if !read_frame_into(&self.buf, pos, &mut self.frame) {
                break;
            }
            // A frame that does not decode leaves silence of the same length and the stream goes on.
            let _ = self.decoder.decode(&self.frame, &mut self.out);
            self.last = Some(self.frame.start);
            self.pos = self.frame.next;
        }
        // Packets before the last frame are not read again.
        if let Some(last) = self.last {
            let drop = last / PACKET_BITS;
            if drop > 0 {
                self.buf.drain(..drop * PACKET_BYTES);
                self.last = Some(last - drop * PACKET_BITS);
                self.pos = self.pos.map(|p| p - drop * PACKET_BITS);
            }
        }
    }

    fn frames(&self) -> usize {
        (self.out.len() - self.out_pos) / self.channels
    }

    fn consume(&mut self, samples: usize) {
        self.out_pos += samples;
        if self.out_pos >= self.out.len() {
            self.out.clear();
            self.out_pos = 0;
        } else if self.out_pos >= 4096 {
            self.out.copy_within(self.out_pos.., 0);
            self.out.truncate(self.out.len() - self.out_pos);
            self.out_pos = 0;
        }
    }

    /// Ends the stream: the held half frame is appended and the stream starts afresh.
    fn finish(&mut self) {
        self.decoder.flush(&mut self.out);
        self.buf.clear();
        self.last = None;
        self.pos = None;
        self.skip = 0;
        self.last_packet = 0;
    }
}

/// The decoder of one open `XMA2WAVEFORMATEX`.
pub struct Engine {
    channels: usize,
    streams: Vec<Stream>,
    /// Interleaved samples of all channels, merged and not yet handed out.
    pending: Vec<i16>,
    /// Read cursor into `pending`.
    pending_pos: usize,
    /// The stream was ended and no packet came since, so a further flush does nothing.
    flushed: bool,
    /// Packets fed since the stream began.
    packets: u64,
}

impl Engine {
    /// An engine for `format`, or why it cannot be decoded.
    pub fn new(format: &Format) -> Result<Engine, Unsupported> {
        if format.block_align != 0 && usize::from(format.block_align) != PACKET_BYTES {
            return Err(Unsupported::BlockSize(format.block_align));
        }
        let wanted = format.channels.div_ceil(2);
        if format.streams != 0 && format.streams != wanted {
            return Err(Unsupported::StreamCount {
                streams: format.streams,
                channels: format.channels,
            });
        }
        let streams = (0..wanted)
            .map(|i| Stream::new(usize::from((format.channels - i * 2).min(2)), format.rate))
            .collect();
        Ok(Engine {
            channels: usize::from(format.channels),
            streams,
            pending: Vec::new(),
            pending_pos: 0,
            flushed: false,
            packets: 0,
        })
    }

    /// Frames decoded but not yet handed out.
    pub fn pending_frames(&self) -> usize {
        (self.pending.len() - self.pending_pos) / self.channels
    }

    /// Feeds whole packets. A rest shorter than one packet is ignored.
    pub fn push(&mut self, block: &[u8]) {
        self.flushed &= block.len() < PACKET_BYTES;
        for packet in block.as_chunks::<PACKET_BYTES>().0 {
            // The stream with no packets to wait for, else the one that comes soonest. Of equal
            // streams (the skip count of the last packet of a stream says nothing) the one that
            // had its turn longest ago.
            let at = (0..self.streams.len())
                .min_by_key(|&i| (self.streams[i].skip, self.streams[i].last_packet))
                .unwrap_or(0);
            self.packets += 1;
            self.streams[at].last_packet = self.packets;
            for (i, s) in self.streams.iter_mut().enumerate() {
                if i != at {
                    s.skip = s.skip.saturating_sub(1);
                }
            }
            self.streams[at].add_packet(packet);
        }
        self.merge();
    }

    /// Ends the stream. The streams are padded with silence to the same length. Calling it again
    /// before more packets come does nothing.
    pub fn flush(&mut self) {
        if std::mem::replace(&mut self.flushed, true) {
            return;
        }
        self.packets = 0;
        for s in &mut self.streams {
            s.finish();
        }
        let longest = self.streams.iter().map(Stream::frames).max().unwrap_or(0);
        for s in &mut self.streams {
            s.out.resize(longest * s.channels, 0);
        }
        self.merge();
    }

    /// Hands out at most `max_frames` frames into `pcm`. Returns the frames written.
    pub fn take(&mut self, pcm: &mut [i16], max_frames: usize) -> usize {
        let give = self
            .pending_frames()
            .min(max_frames)
            .min(pcm.len() / self.channels);
        let samples = give * self.channels;
        let end = self.pending_pos + samples;
        pcm[..samples].copy_from_slice(&self.pending[self.pending_pos..end]);
        self.pending_pos = end;
        if self.pending_pos >= self.pending.len() {
            self.pending.clear();
            self.pending_pos = 0;
        } else if self.pending_pos >= 8192 {
            self.pending.copy_within(self.pending_pos.., 0);
            self.pending.truncate(self.pending.len() - self.pending_pos);
            self.pending_pos = 0;
        }
        give
    }

    /// Interleaves the frames every stream has ready into `pending`.
    fn merge(&mut self) {
        if let [only] = self.streams.as_mut_slice() {
            if self.pending_pos == self.pending.len() && self.pending_pos > 0 {
                self.pending.clear();
                self.pending_pos = 0;
            }
            self.pending.extend_from_slice(&only.out[only.out_pos..]);
            only.out.clear();
            only.out_pos = 0;
            return;
        }
        let ready = self.streams.iter().map(Stream::frames).min().unwrap_or(0);
        if ready == 0 {
            return;
        }
        self.pending.reserve(ready * self.channels);
        for frame in 0..ready {
            for s in &self.streams {
                let start = s.out_pos + frame * s.channels;
                self.pending
                    .extend_from_slice(&s.out[start..start + s.channels]);
            }
        }
        for s in &mut self.streams {
            s.consume(ready * s.channels);
        }
    }
}
