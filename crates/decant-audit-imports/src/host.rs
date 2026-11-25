//! What this PC can provide, without running any file of the game. DLLs in System32 are read
//! from disk as data. Only API sets (`api-ms-win-*`, `ext-ms-*`) are resolved by the Windows
//! loader, from System32 only, as the Python auditor does.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::pe::{Func, Pe};

/// Answers whether a DLL provides a function.
pub enum Checker {
    Exports(HashSet<String>, HashSet<u64>),
    #[cfg(windows)]
    ApiSet(windows::Module),
}

impl Checker {
    pub fn has(&self, func: &Func) -> bool {
        match (self, func) {
            (Checker::Exports(names, _), Func::Name(n)) => names.contains(n),
            (Checker::Exports(_, ordinals), Func::Ordinal(o)) => ordinals.contains(&u64::from(*o)),
            #[cfg(windows)]
            (Checker::ApiSet(module), f) => module.has(f),
        }
    }
}

/// Exported names and ordinals of a DLL.
type Exports = (HashSet<String>, HashSet<u64>);

pub struct Host {
    system32: PathBuf,
    files: HashMap<String, Option<Exports>>,
    #[cfg(windows)]
    sets: HashMap<String, Option<windows::Module>>,
}

/// Where a DLL comes from, and how to check its functions.
pub struct Lookup {
    pub origin: &'static str,
    pub checker: Option<Checker>,
}

fn safe_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['\\', '/', ':']) && name.chars().all(|c| !c.is_control())
}

fn is_api_set(name: &str) -> bool {
    let low = name.to_lowercase();
    low.starts_with("api-ms-") || low.starts_with("ext-ms-")
}

impl Host {
    pub fn new() -> Host {
        let root = std::env::var_os("SystemRoot").map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from);
        Host {
            system32: root.join("System32"),
            files: HashMap::new(),
            #[cfg(windows)]
            sets: HashMap::new(),
        }
    }

    pub fn lookup(&mut self, dll: &str) -> Lookup {
        let absent = Lookup { origin: "absente", checker: None };
        if !safe_name(dll) {
            return absent;
        }
        let key = dll.to_lowercase();
        if is_api_set(dll) {
            #[cfg(windows)]
            {
                let module = self.sets.entry(key).or_insert_with(|| windows::Module::load_from_system32(dll));
                return match module {
                    Some(m) => Lookup { origin: "PC (ensemble d'API)", checker: Some(Checker::ApiSet(*m)) },
                    None => absent,
                };
            }
            #[cfg(not(windows))]
            return absent;
        }
        let path = self.system32.join(dll);
        let exports = self.files.entry(key).or_insert_with(|| path.is_file().then(|| Pe::open(&path).ok().map(|pe| pe.exports())).flatten());
        match exports {
            Some((names, ordinals)) => Lookup { origin: "PC", checker: Some(Checker::Exports(names.clone(), ordinals.clone())) },
            None => absent,
        }
    }

    /// The readable form of a decorated C++ name, when Windows can undecorate it.
    pub fn demangle(&self, name: &str) -> Option<String> {
        if !name.starts_with('?') {
            return None;
        }
        #[cfg(windows)]
        return windows::undecorate(name).filter(|text| text != name);
        #[cfg(not(windows))]
        None
    }
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows {
    //! The three Windows calls the auditor needs. Each takes owned, zero terminated buffers built
    //! here, so the pointers stay valid for the whole call.

    use windows_sys::Win32::Foundation::HMODULE;
    use windows_sys::Win32::System::Diagnostics::Debug::UnDecorateSymbolName;
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW};

    use crate::pe::Func;

    /// A loaded system module. It is never freed, as in the Python version: the process is short.
    #[derive(Clone, Copy)]
    pub struct Module(HMODULE);

    impl Module {
        pub fn load_from_system32(name: &str) -> Option<Module> {
            let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
            // SAFETY: `wide` is a zero terminated UTF 16 string that outlives the call, and the
            // flag restricts the search to System32.
            let handle = unsafe { LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32) };
            (!handle.is_null()).then_some(Module(handle))
        }

        pub fn has(&self, func: &Func) -> bool {
            match func {
                Func::Name(name) => {
                    let latin1: Vec<u8> = name.chars().map(|c| u8::try_from(u32::from(c)).unwrap_or(b'?')).chain(std::iter::once(0)).collect();
                    // SAFETY: the module handle came from LoadLibraryExW and is never freed, and
                    // `latin1` is a zero terminated string that outlives the call.
                    unsafe { GetProcAddress(self.0, latin1.as_ptr()) }.is_some()
                }
                // SAFETY: an ordinal is passed as a pointer value below 0x10000, which
                // GetProcAddress documents as an ordinal, not a string.
                Func::Ordinal(o) => unsafe { GetProcAddress(self.0, usize::from(*o) as *const u8) }.is_some(),
            }
        }
    }

    pub fn undecorate(name: &str) -> Option<String> {
        let input: Vec<u8> = name.chars().map(|c| u8::try_from(u32::from(c)).unwrap_or(b'?')).chain(std::iter::once(0)).collect();
        let mut out = vec![0u8; 2048];
        // SAFETY: `input` is zero terminated and `out` holds the 2048 bytes the call is told about.
        let written = unsafe { UnDecorateSymbolName(input.as_ptr(), out.as_mut_ptr(), out.len() as u32, 0) };
        if written == 0 {
            return None;
        }
        let end = out.iter().position(|&b| b == 0).unwrap_or(out.len());
        Some(out[..end].iter().map(|&b| char::from(b)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_screened_before_any_lookup() {
        assert!(!safe_name(r"..\evil.dll"));
        assert!(!safe_name("c:x.dll"));
        assert!(safe_name("kernel32.dll"));
        assert!(is_api_set("API-MS-WIN-core-file-l1-1-0.dll"));
    }

    #[cfg(windows)]
    #[test]
    fn windows_resolves_api_sets_and_undecorates_names() {
        let mut host = Host::new();
        let set = host.lookup("api-ms-win-core-processthreads-l1-1-0.dll");
        assert_eq!(set.origin, "PC (ensemble d'API)");
        assert!(set.checker.as_ref().unwrap().has(&Func::Name("GetCurrentProcessId".into())));
        assert!(!set.checker.as_ref().unwrap().has(&Func::Name("NoSuchFunction".into())));
        let file = host.lookup("kernel32.dll");
        assert_eq!(file.origin, "PC");
        assert_eq!(host.lookup("no_such_library.dll").origin, "absente");
        assert_eq!(host.demangle("?foo@@YAXXZ").as_deref(), Some("void __cdecl foo(void)"));
        assert_eq!(host.demangle("plain"), None);
    }
}
