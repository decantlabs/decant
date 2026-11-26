//! A growable buffer that keeps its first `N` elements inline, on the stack, and moves to the heap
//! only when more arrive. The page writes of the engine (`Engine::write_frames`) are mostly a few
//! pages, so their working lists never allocate.

/// `N` elements inline, a `Vec` beyond that.
pub enum SmallBuf<T: Copy + Default, const N: usize> {
    Inline { len: usize, buf: [T; N] },
    Heap(Vec<T>),
}

impl<T: Copy + Default, const N: usize> SmallBuf<T, N> {
    pub fn new() -> Self {
        Self::Inline { len: 0, buf: [T::default(); N] }
    }

    /// An empty buffer that has room for `capacity` elements: inline when they fit, else on the heap.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= N { Self::new() } else { Self::Heap(Vec::with_capacity(capacity)) }
    }

    pub fn push(&mut self, value: T) {
        match self {
            Self::Inline { len, buf } if *len < N => {
                buf[*len] = value;
                *len += 1;
            }
            Self::Inline { len, buf } => {
                let mut heap = Vec::with_capacity(N * 2);
                heap.extend_from_slice(&buf[..*len]);
                heap.push(value);
                *self = Self::Heap(heap);
            }
            Self::Heap(v) => v.push(value),
        }
    }

    pub fn as_slice(&self) -> &[T] {
        match self {
            Self::Inline { len, buf } => &buf[..*len],
            Self::Heap(v) => v.as_slice(),
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match self {
            Self::Inline { len, buf } => &mut buf[..*len],
            Self::Heap(v) => v.as_mut_slice(),
        }
    }

    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keeps the first `len` elements.
    pub fn truncate(&mut self, new_len: usize) {
        match self {
            Self::Inline { len, .. } => *len = (*len).min(new_len),
            Self::Heap(v) => v.truncate(new_len),
        }
    }

    /// Whether the elements live on the stack.
    pub fn is_inline(&self) -> bool {
        matches!(self, Self::Inline { .. })
    }
}

impl<T: Copy + Default, const N: usize> Default for SmallBuf<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + Default, const N: usize> FromIterator<T> for SmallBuf<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut out = Self::with_capacity(iter.size_hint().0);
        for value in iter {
            out.push(value);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stays_inline_up_to_n_and_keeps_order_on_the_heap() {
        let mut b: SmallBuf<u32, 4> = SmallBuf::new();
        assert!(b.is_empty());
        for k in 0..4 {
            b.push(k);
        }
        assert!(b.is_inline());
        assert_eq!(b.as_slice(), [0, 1, 2, 3]);
        b.push(4);
        assert!(!b.is_inline());
        assert_eq!(b.as_slice(), [0, 1, 2, 3, 4]);
        b.as_mut_slice().reverse();
        assert_eq!(b.as_slice(), [4, 3, 2, 1, 0]);
        b.truncate(2);
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn collects_inline_when_small_and_on_the_heap_when_large() {
        let small: SmallBuf<u32, 8> = (0..8).collect();
        assert!(small.is_inline());
        let large: SmallBuf<u32, 8> = (0..9).collect();
        assert!(!large.is_inline());
        assert_eq!(large.as_slice(), (0..9).collect::<Vec<u32>>());
    }
}
