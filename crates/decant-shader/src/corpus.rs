//! Access to a shader corpus for the census and for local experiments: the title's shader
//! file or folder of containers (whole containers) or the `shaders_gpu.pack` the extractor writes (code only).
//!
//! The corpus is read on several threads. A [`Selection`] narrows it by name, hash, stage and a
//! seeded random sample before any program is translated.
//!
//! Game data never enters the repository. The tests of this crate use programs built with
//! `crate::asm`, and this module is only used by the command line tool.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::api::{Stage, StageProgram};
use crate::container;

/// One program of a corpus.
pub struct Entry {
    /// Position in the input or pack.
    pub index: usize,
    pub name: String,
    /// FNV-1a of the container, 0 for a pack entry (the pack has no containers).
    pub hash: u64,
    pub program: Result<StageProgram, crate::Refusal>,
}

impl Entry {
    /// Names the program in a stored report: the container hash and the name, or the position
    /// and the name for a pack (it has no hash). Program names alone repeat.
    pub fn id(&self) -> String {
        if self.hash == 0 {
            format!("#{:05} {}", self.index, self.name)
        } else {
            format!("{:016x} {}", self.hash, self.name)
        }
    }
}

/// Which programs of a corpus are looked at. Every filter that is set must match.
#[derive(Clone, Debug, Default)]
pub struct Selection {
    /// Name pattern, case insensitive. `*` and `?` are wildcards. Without a wildcard the pattern
    /// matches anywhere in the name.
    pub name: Option<String>,
    /// Container hashes (container inputs only).
    pub hashes: Option<HashSet<u64>>,
    /// A stage. A container that does not decode has no stage and always passes.
    pub stage: Option<Stage>,
    /// `(count, seed)`: a random sample of that many programs of those the other filters keep.
    /// The same seed gives the same sample.
    pub sample: Option<(usize, u64)>,
}

impl Selection {
    fn name_ok(&self, name: &str) -> bool {
        self.name.as_deref().is_none_or(|p| glob(p, name))
    }

    fn hash_ok(&self, hash: u64) -> bool {
        self.hashes.as_ref().is_none_or(|h| h.contains(&hash))
    }

    fn stage_ok(&self, program: &Result<StageProgram, crate::Refusal>) -> bool {
        match (self.stage, program) {
            (Some(want), Ok(p)) => p.stage == want,
            _ => true,
        }
    }
}

/// Case insensitive match of `pattern` against `text`, `*` any run, `?` one character.
/// A pattern without wildcards is a substring test.
fn glob(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let t: Vec<char> = text.to_lowercase().chars().collect();
    if !p.iter().any(|c| matches!(c, '*' | '?')) {
        return t.windows(p.len().max(1)).any(|w| w == p.as_slice()) || p.is_empty();
    }
    let (mut pi, mut ti) = (0, 0);
    let (mut star, mut mark) = (None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|c| *c == '*')
}

/// Reads a hash list: one hash per line as hex digits (an optional `0x`), anything after the
/// first blank and everything after `#` is ignored.
pub fn parse_hashes(text: &str) -> HashSet<u64> {
    text.lines()
        .filter_map(|l| {
            let tok = l.split('#').next()?.split_whitespace().next()?;
            u64::from_str_radix(tok.trim_start_matches("0x"), 16).ok()
        })
        .collect()
}

/// Calls `visit(acc, i)` for `i` in `0..n` on `threads` threads (0 means every core). Each thread
/// owns one value made by `init`. The values come back in no particular order.
pub fn parallel<T: Send>(
    n: usize,
    threads: usize,
    init: &(impl Fn() -> T + Sync),
    visit: impl Fn(&mut T, usize) + Sync,
) -> Vec<T> {
    let next = AtomicUsize::new(0);
    std::thread::scope(|s| {
        let workers: Vec<_> = (0..thread_count(threads).min(n.max(1)))
            .map(|_| {
                s.spawn(|| {
                    let mut acc = init();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break acc;
                        }
                        visit(&mut acc, i);
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .map(|w| w.join().expect("census worker"))
            .collect()
    })
}

fn thread_count(threads: usize) -> usize {
    if threads == 0 {
        std::thread::available_parallelism().map_or(1, |n| n.get())
    } else {
        threads
    }
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Where the programs come from.
enum Source {
    Files(Vec<decant_extract_shaders::Shader>),
    Pack(Vec<Mutex<Option<Entry>>>),
}

impl Source {
    fn len(&self) -> usize {
        match self {
            Source::Files(shaders) => shaders.len(),
            Source::Pack(v) => v.len(),
        }
    }

    fn name(&self, i: usize) -> String {
        match self {
            Source::Files(shaders) => shaders[i].name.clone(),
            Source::Pack(v) => v[i]
                .lock()
                .unwrap()
                .as_ref()
                .map(|e| e.name.clone())
                .unwrap_or_default(),
        }
    }

    /// The entry at `i`, after the cheap filters. `None` when the selection leaves it out.
    fn entry(&self, i: usize, sel: &Selection) -> Option<Entry> {
        let e = match self {
            Source::Files(shaders) => {
                let item = &shaders[i];
                let blob = item.container();
                let hash = container::fnv1a64(blob);
                if !sel.hash_ok(hash) {
                    return None;
                }
                Entry {
                    index: i,
                    name: item.name.clone(),
                    hash,
                    program: container::program(blob, &item.name),
                }
            }
            Source::Pack(v) => {
                let e = v[i].lock().unwrap().take()?;
                if !sel.hash_ok(e.hash) {
                    return None;
                }
                e
            }
        };
        sel.stage_ok(&e.program).then_some(e)
    }
}

/// Runs `visit` for the selected programs on `threads` threads (0 means every core). Each thread
/// owns one value made by `init`, and the values come back in no particular order: merge them.
/// Returns the number of shaders in the corpus and the values.
fn run<T: Send>(
    source: &Source,
    sel: &Selection,
    threads: usize,
    init: &(impl Fn() -> T + Sync),
    visit: &(impl Fn(&mut T, Entry) + Sync),
) -> (usize, Vec<T>) {
    let n = source.len();
    let pick = |wanted: Option<&HashSet<usize>>| {
        parallel(n, threads, init, |acc, i| {
            if wanted.is_some_and(|w| !w.contains(&i)) {
                return;
            }
            if wanted.is_none() && !sel.name_ok(&source.name(i)) {
                return;
            }
            if let Some(e) = source.entry(i, sel) {
                visit(acc, e);
            }
        })
    };
    let Some((count, seed)) = sel.sample else {
        return (n, pick(None));
    };
    // The sample is drawn from what the filters keep: a first pass lists those, ordered by a
    // hash of the position and the seed. Pack entries are consumed by `entry`, so a pack is
    // sampled from the names only.
    let mut kept: Vec<usize> = match source {
        Source::Pack(_) => (0..n).filter(|&i| sel.name_ok(&source.name(i))).collect(),
        Source::Files(_) => parallel(n, threads, &Vec::new, |v: &mut Vec<usize>, i| {
            if sel.name_ok(&source.name(i)) && source.entry(i, sel).is_some() {
                v.push(i);
            }
        })
        .into_iter()
        .flatten()
        .collect(),
    };
    kept.sort_by_key(|&i| (mix(seed ^ mix(i as u64)), i));
    kept.truncate(count);
    let wanted: HashSet<usize> = kept.into_iter().collect();
    (n, pick(Some(&wanted)))
}

/// Raw console containers from files (the layer dump of a shader that the input does not
/// have). Positions count on from `first_index`. The name is the file name without extension.
pub fn bin_entries(files: &[std::path::PathBuf], first_index: usize) -> Result<Vec<Entry>, String> {
    files
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let blob = std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?;
            let name = f
                .file_stem()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            Ok(Entry {
                index: first_index + i,
                hash: container::fnv1a64(&blob),
                program: container::program(&blob, &name),
                name,
            })
        })
        .collect()
}

/// The selected programs of a file or folder of shader containers. See [`run`] for the threading.
pub fn par_corpus<T: Send>(
    path: &Path,
    sel: &Selection,
    threads: usize,
    init: impl Fn() -> T + Sync,
    visit: impl Fn(&mut T, Entry) + Sync,
) -> Result<(usize, Vec<T>), String> {
    let source = Source::Files(decant_extract_shaders::find_shaders(path)?);
    Ok(run(&source, sel, threads, &init, &visit))
}

/// The selected records of a pack. Signatures and resource lists are empty.
pub fn par_pack<T: Send>(
    path: &Path,
    sel: &Selection,
    threads: usize,
    init: impl Fn() -> T + Sync,
    visit: impl Fn(&mut T, Entry) + Sync,
) -> Result<(usize, Vec<T>), String> {
    let (header, records) = decant_extract_shaders::pack::read(path, |_, _, _| true)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if !header.is_current() {
        return Err(format!("{}: not an XSGC version 1 pack", path.display()));
    }
    let entries = records
        .into_iter()
        .map(|r| {
            let name_len = usize::from(u16::from_le_bytes([r.bytes[0], r.bytes[1]]));
            let words = r.bytes[2 + name_len + 21..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            let program = Stage::from_pack_code(r.stage)
                .map(|stage| StageProgram {
                    name: r.name.clone(),
                    ..StageProgram::new(stage, words)
                })
                .ok_or_else(|| {
                    crate::Refusal::new(crate::refusal::RefusalKind::Container("stage"), "")
                });
            Mutex::new(Some(Entry {
                index: r.index,
                name: r.name,
                hash: 0,
                program,
            }))
        })
        .collect();
    let source = Source::Pack(entries);
    let (_, out) = run(&source, sel, threads, &init, &visit);
    Ok((header.count as usize, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches() {
        assert!(glob("blit", "VS_Blit_x"));
        assert!(glob("vs_*x", "VS_Blit_x"));
        assert!(!glob("vs_*y", "VS_Blit_x"));
        assert!(glob("?s_b*", "VS_Blit_x"));
        assert!(!glob("ps_*", "VS_Blit_x"));
    }

    #[test]
    fn hash_list_text() {
        let h = parse_hashes("# c\n28d887aa04ffa15a  ui vs\n0xB40D9CA24FE3D23D\n\nzz\n");
        assert_eq!(h.len(), 2);
        assert!(h.contains(&0xb40d_9ca2_4fe3_d23d));
    }

    #[test]
    fn sample_is_stable_and_seeded() {
        let key = |seed: u64| {
            let mut v: Vec<usize> = (0..100).collect();
            v.sort_by_key(|&i| (mix(seed ^ mix(i as u64)), i));
            v.truncate(5);
            v
        };
        assert_eq!(key(1), key(1));
        assert_ne!(key(1), key(2));
    }

    #[test]
    fn raw_containers_get_positions_and_hashes() {
        let d = tempfile::tempdir().unwrap();
        let f = d.path().join("one.bin");
        std::fs::write(&f, b"not a container").unwrap();
        let e = bin_entries(&[f], 100).unwrap();
        assert_eq!(e.len(), 1);
        assert_eq!((e[0].index, e[0].name.as_str()), (100, "one"));
        assert_eq!(e[0].hash, container::fnv1a64(b"not a container"));
        assert!(e[0].program.is_err());
    }
}
