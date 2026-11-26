//! Merging of the destination ranges of several writes, so that one write pin
//! (`DecantXmemBeginWrite` .. `DecantXmemFinishWrite`) and one `DecantD3D12XNoteWrite` cover a
//! whole run of adjacent writes instead of one per request.
//!
//! Only ranges that touch or overlap are merged, so a merged span covers exactly the bytes its
//! members write and nothing else: no byte is pinned or reported that no member writes, and every
//! byte a member writes lies inside the span that is pinned while it is written.

/// The longest merged span: pins and notes take a `u32` length.
pub const MAX_SPAN_BYTES: usize = 64 * 1024 * 1024;

/// A run of touching or overlapping destination ranges.
#[derive(Debug, PartialEq, Eq)]
pub struct Span {
    /// First byte address.
    pub start: usize,
    /// Bytes from `start`.
    pub len: u32,
    /// Indexes (into the input) of the ranges inside the span, in input order.
    pub members: Vec<usize>,
}

/// Merges `(address, length)` ranges that touch or overlap into spans, in address order. Empty
/// ranges belong to no span.
pub fn merge(ranges: &[(usize, u32)]) -> Vec<Span> {
    let mut order: Vec<usize> = (0..ranges.len()).filter(|&i| ranges[i].1 > 0).collect();
    order.sort_unstable_by_key(|&i| (ranges[i].0, i));
    let mut spans: Vec<Span> = Vec::new();
    let mut end = 0usize;
    for index in order {
        let (start, len) = ranges[index];
        let range_end = start.saturating_add(len as usize);
        if let Some(span) = spans.last_mut()
            && start <= end
            && range_end.max(end) - span.start <= MAX_SPAN_BYTES
        {
            end = end.max(range_end);
            span.len = (end - span.start) as u32;
            span.members.push(index);
            continue;
        }
        spans.push(Span { start, len, members: vec![index] });
        end = range_end;
    }
    for span in &mut spans {
        span.members.sort_unstable();
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_and_overlapping_ranges_merge_into_one_span() {
        let spans = merge(&[(0x2000, 0x1000), (0x1000, 0x1000), (0x3000, 0x800), (0x1800, 0x2000)]);
        assert_eq!(spans, vec![Span { start: 0x1000, len: 0x2800, members: vec![0, 1, 2, 3] }]);
    }

    #[test]
    fn a_gap_keeps_ranges_apart_so_no_unwritten_byte_is_covered() {
        let spans = merge(&[(0x1000, 0x1000), (0x2001, 0x10), (0x5000, 0x10)]);
        assert_eq!(
            spans,
            vec![
                Span { start: 0x1000, len: 0x1000, members: vec![0] },
                Span { start: 0x2001, len: 0x10, members: vec![1] },
                Span { start: 0x5000, len: 0x10, members: vec![2] },
            ]
        );
    }

    #[test]
    fn a_range_inside_another_does_not_shrink_the_span() {
        let spans = merge(&[(0x1000, 0x4000), (0x1800, 0x100), (0x5000, 0x10)]);
        assert_eq!(spans, vec![Span { start: 0x1000, len: 0x4010, members: vec![0, 1, 2] }]);
    }

    #[test]
    fn empty_ranges_are_ignored() {
        assert!(merge(&[(0x1000, 0)]).is_empty());
        let spans = merge(&[(0x1000, 0), (0x1000, 8)]);
        assert_eq!(spans, vec![Span { start: 0x1000, len: 8, members: vec![1] }]);
    }

    #[test]
    fn spans_are_capped_in_length() {
        let chunk = (MAX_SPAN_BYTES / 2) as u32;
        let base = 0x1_0000usize;
        let spans = merge(&[(base, chunk), (base + chunk as usize, chunk), (base + 2 * chunk as usize, chunk)]);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].len as usize, MAX_SPAN_BYTES);
        assert_eq!(spans[0].members, vec![0, 1]);
        assert_eq!(spans[1].members, vec![2]);
    }

    #[test]
    fn every_member_lies_inside_its_span_and_spans_never_cover_unwritten_bytes() {
        // A pseudo random mix of ranges: the union of the members equals the union of the spans.
        let mut state = 0x1234_5678u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state >> 8
        };
        let ranges: Vec<(usize, u32)> = (0..200).map(|_| (0x10_0000 + (next() % 0x4000) as usize, next() % 0x300)).collect();
        let spans = merge(&ranges);
        let mut covered = vec![false; 0x6000];
        for &(start, len) in &ranges {
            for byte in start..start + len as usize {
                covered[byte - 0x10_0000] = true;
            }
        }
        let mut spanned = vec![false; 0x6000];
        for span in &spans {
            for byte in span.start..span.start + span.len as usize {
                spanned[byte - 0x10_0000] = true;
            }
            for &m in &span.members {
                let (start, len) = ranges[m];
                assert!(start >= span.start && start + len as usize <= span.start + span.len as usize);
            }
        }
        assert_eq!(covered, spanned);
        let members: usize = spans.iter().map(|s| s.members.len()).sum();
        assert_eq!(members, ranges.iter().filter(|r| r.1 > 0).count());
    }
}
