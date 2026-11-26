//! Which view a slot tag of the title's heap memory means.
//!
//! A tag (`title_heap::tag_words`) names the host slot of a view. The title makes the views of a draw in
//! a few slots of a small heap of its own, copies them (with the tags) into a new table with its own
//! memory copies, and makes the next draw's views in the same few slots. So the host slot of an older
//! tag holds a newer view, while on the console the older table keeps the descriptor it was given. The
//! layer looks at a fixed window of slots per table (`title_heap::TABLE_WINDOW`), which covers tables of
//! earlier draws too (measured in the title's menus: the tables of two text draws are 49 slots apart), and
//! must not read such an older tag as the newer view.
//!
//! So each view call that writes a tag counts up a number of its host slot, and the tag carries it in the
//! 16 bits above the 48 address bits of a user mode handle. A tag whose number is not the slot's is old.
//! What a table slot took the last time it was looked at is kept too: an old tag that the table slot
//! already took is [`State::Held`], one it never took is [`State::Lost`]. The slot of a held tag keeps
//! what it took, as the console's table does ([`sync`]).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

const SHIFT: u32 = 48;
const HOST: u64 = (1 << SHIFT) - 1;
const SHARDS: usize = 64;
const SHARD_MASK: usize = SHARDS - 1;

#[derive(Default)]
struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5) ^ u64::from(b);
            self.0 = self.0.wrapping_mul(0x517C_C1B7_2722_0A95);
        }
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0 ^ (i as u64)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0 ^ i).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

#[inline]
fn shard_of(slot: usize) -> usize {
    // Host descriptor handles are strided by descriptor increment (typically 32+ bytes),
    // so mix into the upper bits to distribute uniformly across all shards.
    ((slot.wrapping_mul(0x9E37_79B9_7F4A_7C15) as u64) >> (64 - 6)) as usize & SHARD_MASK
}

/// The host slot a tag word names.
pub fn host_of(word: u64) -> usize {
    (word & HOST) as usize
}

/// What a table slot took from a tag: the tag word, the list and the present it was taken in, and for a
/// constant buffer whose copy the slot got the title's range (address, bytes) and the recording of the
/// list that owns the copy (`cbv_copies::recording`; 0 when the slot itself did not get it, as the table
/// moved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Took {
    pub word: u64,
    pub list: usize,
    pub present: u64,
    pub buffer: Option<(u64, u32)>,
    pub recording: u64,
}

/// What a tag in a table slot means now.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    /// The source slot still holds the view the tag was written for.
    Current,
    /// The source slot holds a newer view, and the table slot took the tag's own one before.
    Held(Took),
    /// The source slot holds a newer view, and the table slot never took the tag's own one.
    Lost,
}

/// What a table slot gets for a draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sync {
    /// The view in the source slot.
    View,
    /// A copy of the title's constant buffer range the source slot holds a view of.
    Buffer((u64, u32)),
    /// A copy of the range it took before: the list that owns that copy was reset since.
    Again((u64, u32)),
    /// Nothing, it holds the view already.
    Keep,
}

/// What a table slot gets whose tag is in `state`, when its source slot holds a view of the constant
/// buffer range `now` (`None`: any other view). `live` tells whether the copy a slot took still exists.
pub fn sync(state: State, now: Option<(u64, u32)>, live: impl Fn(&Took) -> bool) -> Sync {
    match state {
        State::Current | State::Lost => now.map_or(Sync::View, Sync::Buffer),
        State::Held(t) => match t.buffer {
            Some(range) if !live(&t) => Sync::Again(range),
            _ => Sync::Keep,
        },
    }
}

/// The numbers of the source slots and what the table slots took. Pure, for the tests.
#[derive(Default)]
pub struct Tags {
    numbers: FastMap<usize, u16>,
    took: FastMap<usize, Took>,
}

impl Tags {
    /// A view was made in host slot `host`: the word of its tag.
    pub fn made(&mut self, host: usize) -> u64 {
        let n = self.numbers.entry(host).or_insert(0);
        // 0 is a tag without a number
        *n = n.checked_add(1).unwrap_or(1);
        (host as u64 & HOST) | u64::from(*n) << SHIFT
    }

    /// Host slot `host` was written without a tag of its own (a copy through the API): its older tags
    /// no longer name what it holds.
    pub fn overwritten(&mut self, host: usize) {
        if let Some(n) = self.numbers.get_mut(&host) {
            *n = n.checked_add(1).unwrap_or(1);
        }
    }

    /// What tag `word` in table slot `dst` (a host slot) means.
    pub fn state(&self, dst: usize, word: u64) -> State {
        let number = (word >> SHIFT) as u16;
        if number == 0 || self.numbers.get(&host_of(word)).is_none_or(|&n| n == number) {
            return State::Current;
        }
        match self.took.get(&dst) {
            Some(t) if t.word == word => State::Held(*t),
            _ => State::Lost,
        }
    }

    /// Table slot `dst` (a host slot) took what `took` says.
    pub fn taken(&mut self, dst: usize, took: Took) {
        self.took.insert(dst, took);
    }
}

static NUMBERS: [RwLock<Option<FastMap<usize, u16>>>; SHARDS] = [const { RwLock::new(None) }; SHARDS];
static TOOK: [RwLock<Option<FastMap<usize, Took>>>; SHARDS] = [const { RwLock::new(None) }; SHARDS];

#[inline]
fn with_numbers_write<R>(slot: usize, f: impl FnOnce(&mut FastMap<usize, u16>) -> R) -> R {
    let mut guard = NUMBERS[shard_of(slot)].write().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(Default::default))
}

#[inline]
fn with_numbers_read<R>(slot: usize, f: impl FnOnce(Option<&FastMap<usize, u16>>) -> R) -> R {
    let guard = NUMBERS[shard_of(slot)].read().unwrap_or_else(|e| e.into_inner());
    f(guard.as_ref())
}

#[inline]
fn with_took_write<R>(dst: usize, f: impl FnOnce(&mut FastMap<usize, Took>) -> R) -> R {
    let mut guard = TOOK[shard_of(dst)].write().unwrap_or_else(|e| e.into_inner());
    f(guard.get_or_insert_with(Default::default))
}

#[inline]
fn with_took_read<R>(dst: usize, f: impl FnOnce(Option<&FastMap<usize, Took>>) -> R) -> R {
    let guard = TOOK[shard_of(dst)].read().unwrap_or_else(|e| e.into_inner());
    f(guard.as_ref())
}

/// A view was made in host slot `host`: the word of its tag.
pub fn made(host: usize) -> u64 {
    with_numbers_write(host, |numbers| {
        let n = numbers.entry(host).or_insert(0);
        // 0 is a tag without a number
        *n = n.checked_add(1).unwrap_or(1);
        (host as u64 & HOST) | u64::from(*n) << SHIFT
    })
}

/// The `n` host slots from `host`, `increment` bytes each, were written by a copy through the API.
pub fn overwritten(host: usize, n: u32, increment: u32) {
    let inc = increment as usize;
    for i in 0..n as usize {
        let slot = host + i * inc;
        with_numbers_write(slot, |numbers| {
            if let Some(num) = numbers.get_mut(&slot) {
                *num = num.checked_add(1).unwrap_or(1);
            }
        });
    }
}

/// What tag `word` in table slot `dst` means.
pub fn state(dst: usize, word: u64) -> State {
    let number = (word >> SHIFT) as u16;
    if number == 0 {
        return State::Current;
    }
    let src = host_of(word);
    let current = with_numbers_read(src, |numbers| {
        numbers.is_none_or(|map| map.get(&src).is_none_or(|&n| n == number))
    });
    if current {
        return State::Current;
    }
    with_took_read(dst, |took| match took.and_then(|map| map.get(&dst)) {
        Some(t) if t.word == word => State::Held(*t),
        _ => State::Lost,
    })
}

/// Table slot `dst` took what `took` says.
pub fn taken(dst: usize, took: Took) {
    with_took_write(dst, |map| {
        map.insert(dst, took);
    });
}

/// How often the table syncs met an old tag, for the log.
pub struct Counts {
    /// Held tags whose slot an earlier draw of the same list, in the same present, reads: it kept its view.
    pub kept: AtomicU64,
    /// Held tags taken by another list or in an earlier present.
    pub elsewhere: AtomicU64,
    /// Constant buffer copies of held tags made again, as the list that owned them was reset.
    pub again_copied: AtomicU64,
    /// Lost tags.
    pub lost: AtomicU64,
    /// Draws that read a table again, without setting it, that was moved for the draw before.
    pub again: AtomicU64,
}

pub static COUNTS: Counts = Counts { kept: AtomicU64::new(0), elsewhere: AtomicU64::new(0), again_copied: AtomicU64::new(0), lost: AtomicU64::new(0), again: AtomicU64::new(0) };

/// Counts one up.
pub fn count(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: usize = 0x70E5_0291;
    const T: usize = 0x7F44_CA33_DBE0;

    fn took(word: u64) -> Took {
        Took { word, list: 1, present: 5, buffer: None, recording: 0 }
    }

    #[test]
    fn a_tag_names_its_host_slot_whatever_its_number() {
        let mut t = Tags::default();
        let (a, b) = (t.made(S), t.made(S));
        assert_ne!(a, b);
        assert_eq!((host_of(a), host_of(b)), (S, S));
        assert_eq!(host_of(t.made(T)), T);
    }

    #[test]
    fn a_tag_is_current_until_its_source_slot_takes_another_view() {
        let mut t = Tags::default();
        let a = t.made(S);
        assert_eq!(t.state(T, a), State::Current);
        t.taken(T, took(a));
        assert_eq!(t.state(T, a), State::Current);
        let b = t.made(S);
        assert_eq!(t.state(T, a), State::Held(took(a)));
        assert_eq!(t.state(T - 0x620, b), State::Current);
        assert_eq!(t.state(T + 32, a), State::Lost, "another slot never took it");
        t.taken(T, took(b));
        assert_eq!(t.state(T, a), State::Lost, "the slot holds the newer view now");
    }

    #[test]
    fn a_copy_through_the_api_makes_the_older_tags_of_its_slots_old() {
        let mut t = Tags::default();
        let a = t.made(S);
        t.overwritten(S + 32);
        assert_eq!(t.state(T, a), State::Current);
        t.overwritten(S);
        assert_eq!(t.state(T, a), State::Lost);
        assert!(!t.numbers.contains_key(&(S + 32)), "a slot no tag names is not kept");
    }

    #[test]
    fn a_tag_without_a_number_and_one_of_an_unknown_slot_are_current() {
        let mut t = Tags::default();
        t.made(S);
        assert_eq!(t.state(T, S as u64), State::Current);
        assert_eq!(t.state(T, 0x1234 | 7 << SHIFT), State::Current);
    }

    /// Two text draws as the title records them: the tables 0x620 bytes apart, both buffers through the
    /// same source slot, the first on 256 bytes (a plain view), the second off it.
    #[test]
    fn an_earlier_draws_slot_keeps_its_view_when_a_later_window_reaches_it() {
        let mut t = Tags::default();
        let first = t.made(S);
        assert_eq!(sync(t.state(T, first), None, |_| true), Sync::View);
        t.taken(T, took(first));
        let second = t.made(S);
        let now = Some((0x10_0127_D5E0, 1536));
        assert_eq!(sync(t.state(T - 0x620, second), now, |_| true), Sync::Buffer((0x10_0127_D5E0, 1536)), "the draw's own slot");
        assert_eq!(sync(t.state(T, first), now, |_| true), Sync::Keep, "the slot of the draw before");
        assert_eq!(sync(t.state(T, first), None, |_| false), Sync::Keep, "a plain view needs no copy that could go");
    }

    #[test]
    fn a_held_constant_buffer_copy_is_made_again_once_its_list_was_reset() {
        let mut t = Tags::default();
        let word = t.made(S);
        let range = (0x10_0127_D5E0, 1536);
        t.taken(T, Took { buffer: Some(range), recording: 4, ..took(word) });
        t.made(S);
        let other = Some((0x10_0200_0020, 64));
        assert_eq!(sync(t.state(T, word), other, |k| k.recording == 4), Sync::Keep);
        assert_eq!(sync(t.state(T, word), other, |k| k.recording == 5), Sync::Again(range), "its own range, not the newer view's");
    }

    #[test]
    fn a_lost_tag_takes_what_its_source_slot_holds() {
        let mut t = Tags::default();
        let word = t.made(S);
        t.made(S);
        assert_eq!(sync(t.state(T, word), None, |_| true), Sync::View);
        assert_eq!(sync(t.state(T, word), Some((0x20, 16)), |_| true), Sync::Buffer((0x20, 16)));
    }

    #[test]
    fn the_number_never_becomes_zero() {
        let mut t = Tags::default();
        assert!((0..=u32::from(u16::MAX) + 1).all(|_| t.made(S) >> SHIFT != 0));
    }

    #[test]
    fn global_sharded_tags_handle_concurrent_strided_slots() {
        let base_src = 0x70E5_1000_0000usize;
        let base_dst = 0x7F44_1000_0000usize;
        std::thread::scope(|scope| {
            for t in 0..4usize {
                scope.spawn(move || {
                    for i in 0..64usize {
                        let src = base_src + (t * 64 + i) * 32;
                        let dst = base_dst + (t * 64 + i) * 32;
                        let w1 = made(src);
                        assert_eq!(state(dst, w1), State::Current);
                        let tk = Took { word: w1, list: t + 1, present: 1, buffer: None, recording: 0 };
                        taken(dst, tk);
                        let w2 = made(src);
                        assert_ne!(w1, w2);
                        assert_eq!(state(dst, w1), State::Held(tk));
                        assert_eq!(state(dst, w2), State::Current);
                        overwritten(src, 1, 32);
                        assert_eq!(state(dst, w2), State::Lost);
                    }
                });
            }
        });
    }
}
