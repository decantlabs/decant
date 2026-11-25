//! The disk side of the local XGameSave table (`local_gamesave`): the containers of one provider
//! (one user and one service configuration) kept in a host folder.
//!
//! Layout under the provider folder:
//!
//! * One folder per container, named by the container name made file safe (cut to [`READABLE`]
//!   characters), a `~`, and the FNV-1a hash of the exact name. Names that differ only in case or
//!   in characters the host forbids therefore get folders of their own. The exact name is kept in
//!   the manifest.
//! * Inside, the blob files `d<generation>_<slot>` and one `manifest`. The manifest is the single
//!   source of truth: it names the container, its display name and, per blob, the blob name, the
//!   size and the file that holds the bytes. A file the manifest does not list does not exist.
//! * An update picks the next generation, writes the new blobs as fresh files (the unchanged
//!   ones keep their files, nothing is rewritten in place), writes the whole new manifest as
//!   `manifest.next`, flushes it, and commits by renaming it over `manifest`. Only afterwards the
//!   files of the old generation are removed. A process killed at any point leaves either the
//!   old manifest (plus unlisted files) or the new one, never a mixture. Unlisted files are swept
//!   the next time the container is looked at.
//! * A deletion removes the manifest, which is the commit, and then the rest of the folder. A
//!   folder without a manifest is not a container and is swept by the next enumeration.
//!
//! Every operation holds one process wide lock, so none sees another's half made generation.
//! Offline there is no cloud copy, so nothing ever needs a sync.

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

/// `E_GS_INVALID_CONTAINER_NAME`.
pub(crate) const INVALID_CONTAINER_NAME: i32 = 0x8083_0001_u32 as i32;
/// `E_GS_NO_ACCESS`.
pub(crate) const NO_ACCESS: i32 = 0x8083_0002_u32 as i32;
/// `E_GS_OUT_OF_LOCAL_STORAGE`.
pub(crate) const OUT_OF_LOCAL_STORAGE: i32 = 0x8083_0003_u32 as i32;
/// `E_GS_UPDATE_TOO_BIG`.
pub(crate) const UPDATE_TOO_BIG: i32 = 0x8083_0005_u32 as i32;
/// `E_GS_QUOTA_EXCEEDED`.
pub(crate) const QUOTA_EXCEEDED: i32 = 0x8083_0006_u32 as i32;
/// `E_GS_PROVIDED_BUFFER_TOO_SMALL`.
pub(crate) const BUFFER_TOO_SMALL: i32 = 0x8083_0007_u32 as i32;
/// `E_GS_BLOB_NOT_FOUND`.
pub(crate) const BLOB_NOT_FOUND: i32 = 0x8083_0008_u32 as i32;

/// The quota of one user and title, as documented for the console.
pub(crate) const QUOTA: u64 = 256 << 20;
/// What one update may write at most, as documented.
pub(crate) const MAX_UPDATE: u64 = 16 << 20;
/// The longest container or blob name taken.
pub(crate) const MAX_NAME: usize = 256;
/// The readable part of a container folder name, at most.
const READABLE: usize = 48;
const MANIFEST: &str = "manifest";
const MANIFEST_NEXT: &str = "manifest.next";
/// The first line of every manifest. A manifest that starts otherwise is not one of ours.
const MAGIC: &str = "decant_gs 2";

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// `text` with every character but ASCII letters, digits, `-` and `_` replaced by `_`, cut to
/// `max` characters. Never empty.
pub(crate) fn safe_name(text: &str, max: usize) -> String {
    let kept = |c: char| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' };
    let mut out: String = text.chars().take(max).map(kept).collect();
    if out.is_empty() {
        out.push('_');
    }
    out
}

fn fnv1a(text: &str) -> u64 {
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = BASIS;
    for byte in text.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Hex digits of the UTF-8 bytes, so any text fits on one manifest line.
fn to_hex(text: &str) -> String {
    use std::fmt::Write as _;
    text.bytes().fold(String::new(), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

fn from_hex(digits: &str) -> Option<String> {
    let raw = digits.as_bytes();
    if !raw.len().is_multiple_of(2) {
        return None;
    }
    let nibble = |c: u8| (c as char).to_digit(16).map(|d| d as u8);
    let bytes: Option<Vec<u8>> = raw.chunks(2).map(|pair| Some(nibble(pair[0])? << 4 | nibble(pair[1])?)).collect();
    String::from_utf8(bytes?).ok()
}

/// A container or blob name the table takes: 1 to [`MAX_NAME`] bytes.
pub(crate) fn valid_name(name: &str) -> bool {
    (1..=MAX_NAME).contains(&name.len())
}

/// One blob of a manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Entry {
    name: String,
    size: u64,
    /// The file in the container folder that holds the bytes.
    file: String,
}

/// What a container's manifest says.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Manifest {
    name: String,
    display: String,
    generation: u64,
    /// By blob name.
    blobs: Vec<Entry>,
}

/// Blob files are named `d<generation>_<slot>`. Anything else in a manifest is damage, and
/// refusing it keeps a forged manifest from pointing outside the container folder.
fn plain_file_name(file: &str) -> bool {
    file.strip_prefix('d')
        .and_then(|rest| rest.split_once('_'))
        .is_some_and(|(generation, slot)| !generation.is_empty() && !slot.is_empty() && generation.bytes().chain(slot.bytes()).all(|b| b.is_ascii_digit()))
}

impl Manifest {
    fn render(&self) -> String {
        let mut text = format!("{MAGIC}\nname {}\ndisplay {}\ngeneration {}\n", to_hex(&self.name), to_hex(&self.display), self.generation);
        for blob in &self.blobs {
            text.push_str(&format!("blob {} {} {}\n", to_hex(&blob.name), blob.size, blob.file));
        }
        text
    }

    fn parse(text: &str) -> Option<Manifest> {
        let mut lines = text.lines();
        if lines.next()? != MAGIC {
            return None;
        }
        let mut field = |key: &str| lines.next()?.strip_prefix(key)?.strip_prefix(' ').map(str::to_string);
        let name = from_hex(&field("name")?)?;
        let display = from_hex(&field("display")?)?;
        let generation = field("generation")?.parse().ok()?;
        let mut blobs = Vec::new();
        for line in lines {
            let mut parts = line.strip_prefix("blob ")?.split(' ');
            let (blob, size, file) = (parts.next()?, parts.next()?, parts.next()?);
            if parts.next().is_some() || !plain_file_name(file) {
                return None;
            }
            blobs.push(Entry { name: from_hex(blob)?, size: size.parse().ok()?, file: file.to_string() });
        }
        Some(Manifest { name, display, generation, blobs })
    }

    fn total(&self) -> u64 {
        self.blobs.iter().map(|b| b.size).sum()
    }
}

/// A container as found on disk.
struct Found {
    dir: PathBuf,
    manifest: Manifest,
}

/// What the table reports of a container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ContainerSummary {
    pub(crate) name: String,
    pub(crate) display: String,
    pub(crate) blob_count: u32,
    pub(crate) total_size: u64,
    /// Seconds since 1970 of the commit, 0 when unknown.
    pub(crate) modified: i64,
}

/// A blob change of an update: new bytes, or `None` to delete the blob.
pub(crate) type Change = (String, Option<Vec<u8>>);

/// The fault points of the tests: each step of a commit or a deletion may stop the operation as
/// a killed process would, leaving the disk as it is.
fn step() -> std::io::Result<()> {
    #[cfg(test)]
    if tests::stop_here() {
        return Err(std::io::Error::other("stopped by the test"));
    }
    Ok(())
}

fn io_error(e: &std::io::Error) -> i32 {
    if e.kind() == std::io::ErrorKind::StorageFull { OUT_OF_LOCAL_STORAGE } else { NO_ACCESS }
}

fn write_flushed(path: &Path, data: &[u8]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    file.write_all(data)?;
    file.sync_all()
}

fn modified_secs(path: &Path) -> i64 {
    let time = std::fs::metadata(path).and_then(|m| m.modified()).ok();
    time.and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_secs() as i64)
}

/// Reads the committed state of the container folder `dir` and removes every file the manifest
/// does not list (what a killed update or deletion left). `None` when there is no valid manifest.
fn load(dir: &Path) -> std::io::Result<Option<Found>> {
    let manifest = match std::fs::read_to_string(dir.join(MANIFEST)) {
        Ok(text) => Manifest::parse(&text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    if let Some(manifest) = &manifest {
        for entry in std::fs::read_dir(dir)?.flatten() {
            let file = entry.file_name();
            let listed = file == MANIFEST || manifest.blobs.iter().any(|b| file.to_str() == Some(b.file.as_str()));
            if !listed {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    Ok(manifest.map(|manifest| Found { dir: dir.to_path_buf(), manifest }))
}

/// The containers of one provider.
#[derive(Debug)]
pub(crate) struct Store {
    root: PathBuf,
}

impl Store {
    /// The store in `root`, made when missing.
    pub(crate) fn open(root: PathBuf) -> Result<Store, i32> {
        std::fs::create_dir_all(&root).map_err(|e| io_error(&e))?;
        Ok(Store { root })
    }

    pub(crate) fn root(&self) -> &Path {
        &self.root
    }

    fn folder(&self, name: &str) -> PathBuf {
        self.root.join(format!("{}~{:016x}", safe_name(name, READABLE), fnv1a(name)))
    }

    /// Container `name` as committed. A folder that holds another name with the same hash is not
    /// this container.
    fn find(&self, name: &str) -> Result<Option<Found>, i32> {
        let found = load(&self.folder(name)).map_err(|e| io_error(&e))?;
        Ok(found.filter(|f| f.manifest.name == name))
    }

    /// Every committed container, by name. Folders without a manifest are removed on the way.
    fn scan(&self) -> Vec<Found> {
        let mut found = Vec::new();
        for entry in std::fs::read_dir(&self.root).into_iter().flatten().flatten() {
            let path = entry.path();
            if !path.is_dir() || !entry.file_name().to_string_lossy().contains('~') {
                continue;
            }
            match load(&path) {
                Ok(Some(f)) => found.push(f),
                Ok(None) => {
                    let _ = std::fs::remove_dir_all(&path);
                }
                Err(_) => {}
            }
        }
        found.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
        found
    }

    fn summarize(found: &Found) -> ContainerSummary {
        let m = &found.manifest;
        ContainerSummary {
            name: m.name.clone(),
            display: m.display.clone(),
            blob_count: m.blobs.len() as u32,
            total_size: m.total(),
            modified: modified_secs(&found.dir.join(MANIFEST)),
        }
    }

    /// Every container, by name.
    pub(crate) fn containers(&self) -> Vec<ContainerSummary> {
        let _lock = lock();
        self.scan().iter().map(Self::summarize).collect()
    }

    /// Container `name` when it is committed.
    pub(crate) fn container(&self, name: &str) -> Result<Option<ContainerSummary>, i32> {
        let _lock = lock();
        Ok(self.find(name)?.as_ref().map(Self::summarize))
    }

    /// The blobs of container `name` with their sizes, by name. None for a missing container.
    pub(crate) fn blobs(&self, name: &str) -> Result<Vec<(String, u64)>, i32> {
        let _lock = lock();
        let found = self.find(name)?;
        Ok(found.map(|f| f.manifest.blobs.into_iter().map(|b| (b.name, b.size)).collect()).unwrap_or_default())
    }

    /// The bytes all containers hold.
    pub(crate) fn used(&self) -> u64 {
        let _lock = lock();
        self.scan().iter().map(|f| f.manifest.total()).sum()
    }

    /// The quota left.
    pub(crate) fn remaining(&self) -> u64 {
        QUOTA.saturating_sub(self.used())
    }

    /// The named blobs of container `name` in the order asked, or all of them by name for `None`.
    /// `BLOB_NOT_FOUND` for a missing container or blob.
    pub(crate) fn read(&self, name: &str, wanted: Option<&[String]>) -> Result<Vec<(String, Vec<u8>)>, i32> {
        let _lock = lock();
        let found = self.find(name)?.ok_or(BLOB_NOT_FOUND)?;
        let blobs = &found.manifest.blobs;
        let picked: Vec<&Entry> = match wanted {
            None => blobs.iter().collect(),
            Some(names) => names.iter().map(|n| blobs.iter().find(|b| b.name == *n).ok_or(BLOB_NOT_FOUND)).collect::<Result<_, _>>()?,
        };
        let mut out = Vec::with_capacity(picked.len());
        for blob in picked {
            let data = std::fs::read(found.dir.join(&blob.file)).map_err(|e| io_error(&e))?;
            out.push((blob.name.clone(), data));
        }
        Ok(out)
    }

    /// Applies one update to container `name` as one new committed generation: the blob changes
    /// in order, the display name when given (else the current one is kept).
    pub(crate) fn submit(&self, name: &str, display: Option<&str>, changes: &[Change]) -> Result<(), i32> {
        let _lock = lock();
        let dir = self.folder(name);
        let before = load(&dir).map_err(|e| io_error(&e))?;
        if before.as_ref().is_some_and(|f| f.manifest.name != name) {
            // Another container's name hashes alike: never mix the two.
            return Err(NO_ACCESS);
        }
        let before = before.map(|f| f.manifest);

        // The blobs after the update, by name: entries that keep their file, and new bytes.
        let mut kept: Vec<Entry> = before.as_ref().map(|m| m.blobs.clone()).unwrap_or_default();
        let mut fresh: Vec<(&str, &[u8])> = Vec::new();
        for (blob, data) in changes {
            kept.retain(|e| e.name != *blob);
            fresh.retain(|(n, _)| *n != blob.as_str());
            if let Some(data) = data {
                fresh.push((blob, data));
            }
        }
        let written: u64 = fresh.iter().map(|(_, d)| d.len() as u64).sum();
        if written > MAX_UPDATE {
            return Err(UPDATE_TOO_BIG);
        }
        let others: u64 = self.scan().iter().filter(|f| f.manifest.name != name).map(|f| f.manifest.total()).sum();
        if others + written + kept.iter().map(|e| e.size).sum::<u64>() > QUOTA {
            return Err(QUOTA_EXCEEDED);
        }

        let generation = before.as_ref().map_or(0, |m| m.generation) + 1;
        let commit = || -> std::io::Result<()> {
            std::fs::create_dir_all(&dir)?;
            step()?;
            let mut blobs = kept.clone();
            for (slot, (blob, data)) in fresh.iter().enumerate() {
                let file = format!("d{generation}_{slot}");
                write_flushed(&dir.join(&file), data)?;
                blobs.push(Entry { name: blob.to_string(), size: data.len() as u64, file });
                step()?;
            }
            blobs.sort_by(|a, b| a.name.cmp(&b.name));
            let display = display.map(str::to_string).or_else(|| before.as_ref().map(|m| m.display.clone())).unwrap_or_default();
            let next = Manifest { name: name.to_string(), display, generation, blobs };
            write_flushed(&dir.join(MANIFEST_NEXT), next.render().as_bytes())?;
            step()?;
            // The commit: one rename over the old manifest.
            std::fs::rename(dir.join(MANIFEST_NEXT), dir.join(MANIFEST))?;
            step()?;
            // The files only the old generation listed are garbage now.
            load(&dir).map(drop)
        };
        // A failed step leaves the disk as it is: the next look sweeps what is unlisted.
        commit().map_err(|e| io_error(&e))
    }

    /// Deletes container `name`. A missing container is no error.
    pub(crate) fn delete(&self, name: &str) -> Result<(), i32> {
        let _lock = lock();
        if self.find(name)?.is_none() {
            return Ok(());
        }
        let dir = self.folder(name);
        let run = || -> std::io::Result<()> {
            step()?;
            // Without its manifest the container is gone, whatever else is left in the folder.
            std::fs::remove_file(dir.join(MANIFEST))?;
            step()?;
            std::fs::remove_dir_all(&dir)
        };
        run().map_err(|e| io_error(&e))
    }
}

/// The bytes of every file under `dir`, for the quota of the save files folder.
pub(crate) fn folder_bytes(dir: &Path) -> u64 {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| match entry.file_type() {
            Ok(t) if t.is_dir() => folder_bytes(&entry.path()),
            Ok(_) => entry.metadata().map_or(0, |m| m.len()),
            Err(_) => 0,
        })
        .sum()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        /// The step at which the running operation stops (1 is the first), none for no stop.
        static STOP_AT: Cell<Option<u32>> = const { Cell::new(None) };
        static STEPS: Cell<u32> = const { Cell::new(0) };
    }

    pub(crate) fn stop_here() -> bool {
        let n = STEPS.with(|s| {
            s.set(s.get() + 1);
            s.get()
        });
        STOP_AT.with(|s| s.get() == Some(n))
    }

    /// Runs `f` with a stop at step `at`. Gives the result and whether the stop was reached.
    fn stopped_at<R>(at: u32, f: impl FnOnce() -> R) -> (R, bool) {
        STEPS.with(|s| s.set(0));
        STOP_AT.with(|s| s.set(Some(at)));
        let r = f();
        STOP_AT.with(|s| s.set(None));
        (r, STEPS.with(Cell::get) >= at)
    }

    /// A fresh folder for one test.
    pub(crate) fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("decant-xgs-store-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    type Snapshot = Option<(String, Vec<(String, String)>)>;

    /// The container "Slot" as a reader sees it: display name and blobs as text.
    fn snapshot(store: &Store) -> Snapshot {
        let summary = store.container("Slot").unwrap()?;
        let blobs = store.read("Slot", None).unwrap();
        Some((summary.display, blobs.into_iter().map(|(n, d)| (n, String::from_utf8(d).unwrap())).collect()))
    }

    fn expect(display: &str, blobs: &[(&str, &str)]) -> Snapshot {
        Some((display.to_string(), blobs.iter().map(|(n, d)| (n.to_string(), d.to_string())).collect()))
    }

    fn put(name: &str, data: &str) -> Change {
        (name.to_string(), Some(data.as_bytes().to_vec()))
    }

    fn drop_blob(name: &str) -> Change {
        (name.to_string(), None)
    }

    fn files_in(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        names
    }

    #[test]
    fn a_manifest_survives_a_round_trip_and_damage_is_refused() {
        let m = Manifest {
            name: "a/B c".into(),
            display: "Chapter \u{e9}".into(),
            generation: 7,
            blobs: vec![Entry { name: "x y".into(), size: 3, file: "d7_0".into() }, Entry { name: String::new(), size: 0, file: "d3_1".into() }],
        };
        assert_eq!(Manifest::parse(&m.render()), Some(m.clone()));
        assert_eq!(Manifest::parse(&m.render().replace(MAGIC, "other 1")), None);
        assert_eq!(Manifest::parse(&format!("{MAGIC}\nname 61\n")), None);
        assert_eq!(Manifest::parse(&m.render().replace("d7_0", "..")), None, "a blob file outside the naming scheme");
        assert_eq!(Manifest::parse(&m.render().replace("name 612f", "name 61zz")), None);
        assert_eq!(safe_name("a/b.C~d", 48), "a_b_C_d");
        assert_eq!(safe_name("", 48), "_");
        assert_eq!(safe_name("abcdef", 3), "abc");
        assert!(valid_name("x") && !valid_name("") && !valid_name(&"a".repeat(257)));
    }

    #[test]
    fn an_update_keeps_unchanged_blobs_and_the_display_name() {
        let store = Store::open(scratch("keep")).unwrap();
        assert_eq!(snapshot(&store), None);
        store.submit("Slot", Some("one"), &[put("a", "A"), put("b", "B")]).unwrap();
        store.submit("Slot", None, &[put("b", "BB"), put("c", "C"), drop_blob("a")]).unwrap();
        assert_eq!(snapshot(&store), expect("one", &[("b", "BB"), ("c", "C")]));
        assert_eq!(files_in(&store.folder("Slot")).len(), 3, "the manifest and the two live blobs only");
        assert_eq!(store.read("Slot", Some(&["c".into(), "b".into()])).unwrap(), [("c".into(), b"C".to_vec()), ("b".into(), b"BB".to_vec())]);
        assert_eq!(store.read("Slot", Some(&["a".into()])), Err(BLOB_NOT_FOUND));
        assert_eq!(store.read("Other", None), Err(BLOB_NOT_FOUND));
        assert_eq!((store.used(), store.remaining()), (3, QUOTA - 3));
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn a_stop_at_any_step_of_an_update_leaves_the_old_or_the_new_container() {
        let old = expect("one", &[("a", "old a"), ("b", "old b")]);
        let new = expect("two", &[("a", "new a"), ("c", "new c")]);
        let (mut old_seen, mut new_seen) = (0, 0);
        for at in 1.. {
            let store = Store::open(scratch(&format!("stop{at}"))).unwrap();
            store.submit("Slot", Some("one"), &[put("a", "old a"), put("b", "old b")]).unwrap();
            let (result, reached) = stopped_at(at, || store.submit("Slot", Some("two"), &[put("a", "new a"), drop_blob("b"), put("c", "new c")]));
            assert_eq!(result.is_err(), reached, "step {at}");
            // What a restarted process reads.
            let after = snapshot(&Store::open(store.root().to_path_buf()).unwrap());
            if after == old {
                old_seen += 1;
            } else {
                assert_eq!(after, new, "stopped at step {at}");
                new_seen += 1;
            }
            // The next update works and leaves nothing but its own files.
            store.submit("Slot", None, &[put("d", "d")]).unwrap();
            assert_eq!(files_in(&store.folder("Slot")).len(), 1 + store.blobs("Slot").unwrap().len());
            let _ = std::fs::remove_dir_all(store.root());
            if !reached {
                break;
            }
        }
        assert!(old_seen >= 3 && new_seen >= 1, "{old_seen} {new_seen}");
    }

    #[test]
    fn a_stop_during_a_deletion_leaves_the_container_or_nothing() {
        for at in 1..=3 {
            let store = Store::open(scratch(&format!("del{at}"))).unwrap();
            store.submit("Slot", Some("one"), &[put("a", "A")]).unwrap();
            let (result, reached) = stopped_at(at, || store.delete("Slot"));
            assert_eq!(result.is_err(), reached);
            let after = snapshot(&store);
            assert!(after.is_none() || after == expect("one", &[("a", "A")]), "step {at}: {after:?}");
            assert!(store.containers().len() <= 1);
            assert!(after.is_some() || !store.folder("Slot").exists(), "an enumeration sweeps a folder without a manifest");
            let _ = std::fs::remove_dir_all(store.root());
        }
    }

    #[test]
    fn files_nobody_listed_are_never_read_and_get_swept() {
        let store = Store::open(scratch("unlisted")).unwrap();
        store.submit("Slot", Some("one"), &[put("a", "A")]).unwrap();
        let dir = store.folder("Slot");
        // What a kill in the middle of an update leaves: a part written blob and a part written
        // next manifest.
        std::fs::write(dir.join("d2_0"), b"half").unwrap();
        std::fs::write(dir.join(MANIFEST_NEXT), b"decant_gs 2\nname 53").unwrap();
        assert_eq!(snapshot(&store), expect("one", &[("a", "A")]));
        assert_eq!(files_in(&dir).len(), 2, "the stray files were swept");
        // Damage from outside: a manifest nobody can parse is no container.
        std::fs::write(dir.join(MANIFEST), b"garbage").unwrap();
        assert_eq!(snapshot(&store), None);
        assert!(store.containers().is_empty() && !dir.exists());
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn names_differing_in_case_or_forbidden_characters_stay_apart() {
        let store = Store::open(scratch("names")).unwrap();
        let names = ["Save", "save", "SAVE/CON.x", "a:b*c?"];
        for name in names {
            store.submit(name, Some(name), &[put("CON", name), put("con", "lower")]).unwrap();
        }
        let listed: Vec<String> = store.containers().into_iter().map(|c| c.name).collect();
        assert_eq!(listed, ["SAVE/CON.x", "Save", "a:b*c?", "save"]);
        for name in names {
            assert_eq!(store.read(name, Some(&["CON".into()])).unwrap(), [("CON".into(), name.as_bytes().to_vec())]);
        }
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn the_update_size_is_enforced_and_a_refused_update_writes_nothing() {
        let store = Store::open(scratch("limits")).unwrap();
        let big = vec![0u8; MAX_UPDATE as usize + 1];
        assert_eq!(store.submit("Slot", None, &[("a".into(), Some(big))]), Err(UPDATE_TOO_BIG));
        assert_eq!(store.container("Slot").unwrap(), None);
        assert!(!store.folder("Slot").exists());
        let _ = std::fs::remove_dir_all(store.root());
    }

    #[test]
    fn the_folder_size_counts_nested_files() {
        let dir = scratch("bytes");
        std::fs::create_dir_all(dir.join("in")).unwrap();
        std::fs::write(dir.join("one"), [0u8; 5]).unwrap();
        std::fs::write(dir.join("in").join("two"), [0u8; 7]).unwrap();
        assert_eq!(folder_bytes(&dir), 12);
        assert_eq!(folder_bytes(&dir.join("missing")), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
