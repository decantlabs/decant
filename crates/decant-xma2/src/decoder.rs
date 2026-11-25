//! The decoder behind the `DecantXma2Decoder` table: `Open`, `Decode` and `Close` over FFmpeg's xma2
//! decoder, as `decant_xma2.c` did them.
#![allow(unsafe_code)]

use std::ffi::{c_int, c_void};

use crate::ffmpeg::{
    AV_CODEC_ID_XMA2, AV_INPUT_BUFFER_PADDING_SIZE, AV_SAMPLE_FMT_FLT, AV_SAMPLE_FMT_FLTP, AV_SAMPLE_FMT_S16, AV_SAMPLE_FMT_S16P, AVERROR_EAGAIN,
    AVERROR_EOF, AVERROR_INVALIDDATA, AVCodecContext, AVFrame, AVPacket, Ff, ffmpeg,
};
use crate::{FORMAT_HEADER, Format, FormatError, to_short};

/// One open decoder.
pub struct Context {
    ff: &'static Ff,
    codec: *mut AVCodecContext,
    packet: *mut AVPacket,
    frame: *mut AVFrame,
    block_align: u32,
    channels: u32,
    /// One packet with the zeroed padding FFmpeg may read past its end.
    scratch: Vec<u8>,
    /// Interleaved 16 bit samples decoded but not yet handed out.
    pending: Vec<i16>,
}

impl Context {
    /// Frames decoded but not yet handed out.
    pub fn pending_frames(&self) -> usize {
        self.pending.len() / self.channels as usize
    }

    /// Frees what FFmpeg allocated. A null member is skipped.
    fn release(&mut self) {
        // SAFETY: each pointer is null or was allocated by the matching FFmpeg function, and the
        // free functions reset it to null.
        unsafe {
            if !self.frame.is_null() {
                (self.ff.av_frame_free)(&mut self.frame);
            }
            if !self.packet.is_null() {
                (self.ff.av_packet_free)(&mut self.packet);
            }
            if !self.codec.is_null() {
                (self.ff.avcodec_free_context)(&mut self.codec);
            }
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        self.release();
    }
}

/// `Close`.
///
/// # Safety
/// `context` is null or a pointer `open` returned and that was not closed yet.
pub unsafe extern "C" fn close(context: *mut c_void) {
    if !context.is_null() {
        // SAFETY: the pointer came from `Box::into_raw` in `open` and is closed once.
        drop(unsafe { Box::from_raw(context.cast::<Context>()) });
    }
}

/// `Open`. `format`: a `WAVEFORMATEX` with `wFormatTag` 0x166 followed by its `cbSize` extra
/// bytes (the XMA2 part).
///
/// # Safety
/// `format` is null or points to `format_bytes` readable bytes.
pub unsafe extern "C" fn open(format: *const c_void, format_bytes: u32) -> *mut c_void {
    if format.is_null() || (format_bytes as usize) < FORMAT_HEADER {
        return std::ptr::null_mut();
    }
    let Some(ff) = ffmpeg() else {
        return std::ptr::null_mut();
    };
    // SAFETY: the caller gives `format_bytes` readable bytes at `format`.
    let f = unsafe { std::slice::from_raw_parts(format.cast::<u8>(), format_bytes as usize) };
    let parsed = match Format::parse(f) {
        Ok(parsed) => parsed,
        Err(FormatError::TooShort) => return std::ptr::null_mut(),
        Err(FormatError::NotXma2) => {
            crate::ffmpeg::log("[decant_xma2] not an XMA2 format\r\n");
            return std::ptr::null_mut();
        }
    };
    // SAFETY: FFmpeg calls with the signatures of the loaded version. Every pointer is checked
    // before use, and `Context` frees what was allocated on every early return.
    unsafe {
        let xma2 = (ff.avcodec_find_decoder)(AV_CODEC_ID_XMA2);
        if xma2.is_null() {
            return std::ptr::null_mut();
        }
        let mut c = Box::new(Context {
            ff,
            codec: (ff.avcodec_alloc_context3)(xma2),
            packet: (ff.av_packet_alloc)(),
            frame: (ff.av_frame_alloc)(),
            block_align: 0,
            channels: 0,
            scratch: Vec::new(),
            pending: Vec::new(),
        });
        if c.codec.is_null() || c.packet.is_null() || c.frame.is_null() {
            return std::ptr::null_mut();
        }
        // As FFmpeg's own WAV reader sets them up: the extra bytes after WAVEFORMATEX are the decoder's extradata.
        let extra = usize::from(parsed.extra);
        ff.set_sample_rate(c.codec, parsed.rate as c_int);
        ff.set_block_align(c.codec, if parsed.block_align != 0 { c_int::from(parsed.block_align) } else { 2048 });
        ff.set_default_layout(c.codec, c_int::from(parsed.channels));
        let extradata = (ff.av_mallocz)(extra + AV_INPUT_BUFFER_PADDING_SIZE).cast::<u8>();
        if extradata.is_null() {
            return std::ptr::null_mut();
        }
        std::ptr::copy_nonoverlapping(f.as_ptr().add(FORMAT_HEADER), extradata, extra);
        ff.set_extradata(c.codec, extradata, extra as c_int);
        if (ff.avcodec_open2)(c.codec, xma2, std::ptr::null_mut()) < 0 {
            crate::ffmpeg::log("[decant_xma2] avcodec_open2 failed for XMA2\r\n");
            return std::ptr::null_mut();
        }
        c.block_align = ff.block_align(c.codec) as u32;
        c.channels = u32::from(parsed.channels);
        c.scratch = vec![0u8; c.block_align as usize + AV_INPUT_BUFFER_PADDING_SIZE];
        Box::into_raw(c).cast()
    }
}

/// Writes `n` frames of `frame_channels`, in sample format `format`, as interleaved 16 bit samples
/// of `channels` channels into `out` (`n * channels` zeroed samples). Channels the frame does not
/// have stay zero, an unknown sample format gives silence.
///
/// # Safety
/// `planes` is the `extended_data` of a frame of `format`: `frame_channels` planes of `n`
/// samples, or one plane of `n * frame_channels` interleaved samples for the packed formats.
unsafe fn convert(out: &mut [i16], format: c_int, n: usize, frame_channels: usize, channels: usize, planes: *const *const u8) {
    // The channels both sides have. The packed formats are indexed by this count, as the C code did.
    let ch = channels.min(frame_channels);
    for i in 0..n {
        for k in 0..ch {
            // SAFETY: per the contract, plane `k` (or plane 0 for the packed formats) holds the sample read.
            let v = unsafe {
                match format {
                    AV_SAMPLE_FMT_FLTP => planes.add(k).read().cast::<f32>().add(i).read_unaligned(),
                    AV_SAMPLE_FMT_FLT => planes.read().cast::<f32>().add(i * ch + k).read_unaligned(),
                    AV_SAMPLE_FMT_S16P => {
                        out[i * channels + k] = planes.add(k).read().cast::<i16>().add(i).read_unaligned();
                        continue;
                    }
                    AV_SAMPLE_FMT_S16 => {
                        out[i * channels + k] = planes.read().cast::<i16>().add(i * ch + k).read_unaligned();
                        continue;
                    }
                    _ => 0.0,
                }
            };
            out[i * channels + k] = to_short(v);
        }
    }
}

/// Moves every frame the decoder has ready into the pending samples. 0, or a negative FFmpeg
/// error (or -1 when memory runs out).
unsafe fn drain(c: &mut Context) -> c_int {
    let ff = c.ff;
    loop {
        // SAFETY: the codec and frame are live (checked in `open`).
        let r = unsafe { (ff.avcodec_receive_frame)(c.codec, c.frame) };
        if r == AVERROR_EAGAIN || r == AVERROR_EOF {
            return 0;
        }
        if r < 0 {
            return r;
        }
        // SAFETY: the frame was just filled by `avcodec_receive_frame`.
        let (n, format, frame_channels, planes) = unsafe {
            (ff.frame_nb_samples(c.frame).max(0) as usize, ff.frame_format(c.frame), ff.frame_channels(c.frame).max(0) as usize, ff.frame_planes(c.frame))
        };
        let channels = c.channels as usize;
        let start = c.pending.len();
        if c.pending.try_reserve(n * channels).is_err() {
            // SAFETY: the frame is live.
            unsafe { (ff.av_frame_unref)(c.frame) };
            return -1;
        }
        c.pending.resize(start + n * channels, 0);
        // SAFETY: the frame holds `n` samples in the planes of its format.
        unsafe { convert(&mut c.pending[start..], format, n, frame_channels, channels, planes) };
        // SAFETY: the frame is live.
        unsafe { (ff.av_frame_unref)(c.frame) };
    }
}

/// `Decode`: decodes `block` (whole XMA2 packets of `block_align` bytes, 2048 in practice) and
/// hands out at most `max_frames`. What does not fit is kept for the next call. A call with
/// `block_bytes` 0 only hands out what is kept. A null block ends the stream: the decoder is
/// flushed, and further null calls hand out the rest.
///
/// # Safety
/// `context` is null or from `open`. `block` is null or `block_bytes` readable bytes. `pcm`
/// holds `max_frames * channels` samples when `max_frames` is not 0. `frames_out` is null or
/// writable.
pub unsafe extern "C" fn decode(context: *mut c_void, block: *const u8, block_bytes: u32, pcm: *mut i16, max_frames: u32, frames_out: *mut u32) -> i32 {
    if context.is_null() || frames_out.is_null() || (max_frames != 0 && pcm.is_null()) {
        return -1;
    }
    // SAFETY: a non null context is a live `Context` from `open`, and `frames_out` is writable.
    let (c, frames_out) = unsafe { (&mut *context.cast::<Context>(), &mut *frames_out) };
    *frames_out = 0;
    let ff = c.ff;
    let flush = block.is_null();
    if flush {
        // AVERROR_EOF once already flushed: nothing more to do.
        // SAFETY: a null packet is FFmpeg's flush request.
        unsafe { (ff.avcodec_send_packet)(c.codec, std::ptr::null()) };
    }
    let step = c.block_align as usize;
    let mut at = 0usize;
    while !block.is_null() && step != 0 && at + step <= block_bytes as usize {
        // A copy with zeroed padding behind it: the decoder may read a little past the packet.
        // SAFETY: the caller gives `block_bytes` readable bytes and `at + step` is within them.
        unsafe { std::ptr::copy_nonoverlapping(block.add(at), c.scratch.as_mut_ptr(), step) };
        // SAFETY: the codec and packet are live. The packet points at `scratch`, which outlives the
        // call, and is cleared again before anything else can free it.
        unsafe {
            ff.set_packet(c.packet, c.scratch.as_mut_ptr(), step as c_int);
            let mut r = (ff.avcodec_send_packet)(c.codec, c.packet);
            if r == AVERROR_EAGAIN {
                if drain(c) < 0 {
                    return -1;
                }
                r = (ff.avcodec_send_packet)(c.codec, c.packet);
            }
            ff.set_packet(c.packet, std::ptr::null_mut(), 0);
            if r < 0 && r != AVERROR_INVALIDDATA {
                return -1;
            }
            if drain(c) < 0 {
                return -1;
            }
        }
        at += step;
    }
    // SAFETY: the codec and frame are live.
    if flush && unsafe { drain(c) } < 0 {
        return -1;
    }
    let channels = c.channels as usize;
    let give = c.pending_frames().min(max_frames as usize);
    if give > 0 {
        // SAFETY: the caller's buffer holds `max_frames` frames and `give <= max_frames`.
        unsafe { std::ptr::copy_nonoverlapping(c.pending.as_ptr(), pcm, give * channels) };
        c.pending.drain(..give * channels);
    }
    *frames_out = give as u32;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes_of(data: &[Vec<u8>]) -> Vec<*const u8> {
        data.iter().map(|d| d.as_ptr()).collect()
    }

    fn bytes_f32(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bytes_i16(v: &[i16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn planar_float_is_interleaved_and_rounded() {
        let data = [bytes_f32(&[0.5, -0.5]), bytes_f32(&[1.0, -1.0])];
        let planes = planes_of(&data);
        let mut out = [0i16; 4];
        // SAFETY: two planes of two samples each.
        unsafe { convert(&mut out, AV_SAMPLE_FMT_FLTP, 2, 2, 2, planes.as_ptr()) };
        assert_eq!(out, [16384, 32767, -16384, -32768]);
    }

    #[test]
    fn packed_float_is_read_with_the_shared_channel_count() {
        let data = [bytes_f32(&[0.5, 0.25, -0.5, -0.25])];
        let planes = planes_of(&data);
        let mut out = [0i16; 4];
        // SAFETY: one packed plane of two frames of two channels.
        unsafe { convert(&mut out, AV_SAMPLE_FMT_FLT, 2, 2, 2, planes.as_ptr()) };
        assert_eq!(out, [16384, 8192, -16384, -8192]);
    }

    #[test]
    fn sixteen_bit_formats_are_copied() {
        let planar = [bytes_i16(&[1, 2]), bytes_i16(&[3, 4])];
        let planes = planes_of(&planar);
        let mut out = [0i16; 4];
        // SAFETY: two planes of two samples each.
        unsafe { convert(&mut out, AV_SAMPLE_FMT_S16P, 2, 2, 2, planes.as_ptr()) };
        assert_eq!(out, [1, 3, 2, 4]);
        let packed = [bytes_i16(&[1, 3, 2, 4])];
        let planes = planes_of(&packed);
        let mut out = [0i16; 4];
        // SAFETY: one packed plane of two frames of two channels.
        unsafe { convert(&mut out, AV_SAMPLE_FMT_S16, 2, 2, 2, planes.as_ptr()) };
        assert_eq!(out, [1, 3, 2, 4]);
    }

    #[test]
    fn missing_channels_and_unknown_formats_stay_silent() {
        // A mono frame into a stereo buffer: the second channel is left at zero.
        let data = [bytes_f32(&[0.5, -0.5])];
        let planes = planes_of(&data);
        let mut out = [0i16; 4];
        // SAFETY: one plane of two samples.
        unsafe { convert(&mut out, AV_SAMPLE_FMT_FLTP, 2, 1, 2, planes.as_ptr()) };
        assert_eq!(out, [16384, 0, -16384, 0]);
        let mut out = [5i16; 4];
        // SAFETY: an unknown format reads nothing.
        unsafe { convert(&mut out, 99, 2, 2, 2, planes.as_ptr()) };
        assert_eq!(out, [0, 0, 0, 0]);
    }

    #[test]
    fn the_entry_points_refuse_bad_arguments_without_ffmpeg() {
        let mut frames = 5u32;
        // SAFETY: null pointers are the refusals under test.
        unsafe {
            assert_eq!(decode(std::ptr::null_mut(), std::ptr::null(), 0, std::ptr::null_mut(), 0, &mut frames), -1);
            assert_eq!(frames, 5);
            assert!(open(std::ptr::null(), 100).is_null());
            assert!(open([0u8; 4].as_ptr().cast(), 4).is_null());
            close(std::ptr::null_mut());
        }
    }
}
