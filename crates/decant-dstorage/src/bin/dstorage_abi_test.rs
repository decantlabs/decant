//! Test of the spec driven console ABI of `dstorage_xs.dll`, the former `tests/dstorage_abi_test.cpp`.
//! Synthetic files and synthetic IIDs only. The first argument is the DLL, as for the other tests.
//! One process per spec, since the DLL reads its spec once:
//!
//! ```text
//! dstorage_abi_test <dll> none          no spec: console IIDs refused, the project factory still works
//! dstorage_abi_test <dll> spec a|b      the same scenario through two specs with different slot orders
//! dstorage_abi_test <dll> bad           a malformed spec is rejected: console IIDs refused
//! dstorage_abi_test <dll> parse <file>  a shipped spec loads and its factory IID gives a factory
//! dstorage_abi_test <dll> template <f>  the console template, once its base IIDs are filled, runs the scenario
//! dstorage_abi_test <dll> refused <f>   a spec (e.g. the unfilled template) refuses the test factory IID
//! ```
//!
//! Exit codes: 0 all passed, 1 a check failed, 2 bad arguments or a DLL that does not load.

#![allow(unsafe_code)]

#[cfg(windows)]
#[path = "../testapi.rs"]
mod testapi;

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    let args = decant_testkit::args();
    let Some(dll) = args.first().and_then(|path| testapi::Dll::load(path)) else {
        return std::process::ExitCode::from(2);
    };
    std::process::ExitCode::from(run::run(&dll, &args[1..]))
}

#[cfg(not(windows))]
fn main() {
    eprintln!("dstorage_abi_test runs on Windows only");
}

#[cfg(windows)]
mod run {
    use std::ffi::{OsStr, OsString, c_void};
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    use dstorage_xs::decant::*;
    use dstorage_xs::spec::parse_guid;

    use crate::check;
    use crate::testapi::{Dll, slot};

    const FACTORY: GUID = GUID::from_u128(0x7e57a000_0000_4000_8000_000000000001);
    const FILE: GUID = GUID::from_u128(0x7e57a000_0000_4000_8000_000000000002);
    const QUEUE: GUID = GUID::from_u128(0x7e57a000_0000_4000_8000_000000000003);
    const STATUS: GUID = GUID::from_u128(0x7e57a000_0000_4000_8000_000000000004);
    const UNKNOWN_IID: GUID = GUID::from_u128(0x7e57a000_0000_4000_8000_000000000009);

    type Query = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT;

    fn release_of(object: *mut c_void) -> u32 {
        // SAFETY: the callers pass a COM object of the DLL.
        unsafe { release(object) }
    }

    /// `QueryInterface` through slot 0.
    fn query(object: *mut c_void, iid: &GUID, out: &mut *mut c_void) -> HRESULT {
        // SAFETY: slot 0 of any COM object is `QueryInterface`.
        unsafe { slot::<Query>(object, 0)(object, iid, out) }
    }

    struct Slots {
        create_queue: usize,
        open_file: usize,
        create_status: usize,
        unimplemented: usize,
        request: usize,
        status: usize,
        submit: usize,
        is_complete: usize,
        get_hresult: usize,
    }

    /// Spec "a" follows the PC order, spec "b" moves every method, which must need no engine change.
    const SPEC_A: &str = "abi 1\n\
        interface factory testfactory\n status test\n\
        \x20iid {7e57a000-0000-4000-8000-000000000001}\n\
        \x20method 3 factory.create_queue\n method 4 factory.open_file\n\
        \x20method 5 factory.create_status_array\n method 6 ? SomethingElse\nend\n\
        interface file testfile\n iid {7e57a000-0000-4000-8000-000000000002}\nend\n\
        interface queue testqueue\n iid {7e57a000-0000-4000-8000-000000000003}\n\
        \x20method 3 queue.enqueue_request\n method 4 queue.enqueue_status\n\
        \x20method 6 queue.submit\nend\n\
        interface status teststatus\n iid {7e57a000-0000-4000-8000-000000000004}\n\
        \x20method 3 status.is_complete\n method 4 status.get_hresult\nend\n";
    const SLOTS_A: Slots = Slots { create_queue: 3, open_file: 4, create_status: 5, unimplemented: 6, request: 3, status: 4, submit: 6, is_complete: 3, get_hresult: 4 };
    const SPEC_B: &str = "abi 1\n\
        # every method moved, and a gap left in the factory\n\
        interface factory testfactory\n\
        \x20iid {7e57a000-0000-4000-8000-000000000001}\n\
        \x20method 7 factory.create_queue\n method 3 factory.open_file\n\
        \x20method 9 factory.create_status_array\n method 4 ?\nend\n\
        interface file testfile\n iid {7e57a000-0000-4000-8000-000000000002}\nend\n\
        interface queue testqueue\n iid {7e57a000-0000-4000-8000-000000000003}\n\
        \x20method 10 queue.enqueue_request\n method 5 queue.enqueue_status\n\
        \x20method 3 queue.submit\nend\n\
        interface status teststatus\n iid {7e57a000-0000-4000-8000-000000000004}\n\
        \x20method 6 status.get_hresult\n method 5 status.is_complete\nend\n";
    const SLOTS_B: Slots = Slots { create_queue: 7, open_file: 3, create_status: 9, unimplemented: 4, request: 10, status: 5, submit: 3, is_complete: 5, get_hresult: 6 };
    /// A handler on the wrong role: the whole spec must be refused.
    const SPEC_BAD: &str = "abi 1\n\
        interface factory testfactory\n iid {7e57a000-0000-4000-8000-000000000001}\n\
        \x20method 3 queue.submit\nend\n";

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(name)
    }

    fn set_spec_path(path: &OsStr) {
        // SAFETY: the program is single threaded when it sets its environment.
        unsafe { std::env::set_var("DECANT_DSTORAGE_ABI", path) };
    }

    fn use_spec(text: &[u8], name: &str) -> bool {
        let path = temp_path(name);
        if std::fs::write(&path, text).is_err() {
            return false;
        }
        set_spec_path(path.as_os_str());
        true
    }

    fn scenario(dll: &Dll, s: &Slots) -> u8 {
        let data: Vec<u8> = (0..4096usize).map(|i| (i * 7 + 3) as u8).collect();
        let path = temp_path("decant_dstorage_abi_data.bin");
        check!(std::fs::write(&path, &data).is_ok());
        let wide_path: Vec<u16> = crate::testapi::wide(path.as_os_str()).into_iter().chain(std::iter::once(0)).collect();

        let (hr, factory) = dll.get_factory(&FACTORY);
        check!(hr == S_OK && !factory.is_null());
        let factory = factory.take();
        let mut other = std::ptr::null_mut();
        check!(query(factory, &UNKNOWN_IID, &mut other) == E_NOINTERFACE && other.is_null());
        check!(query(factory, &IID_IUNKNOWN, &mut other) == S_OK && other == factory);
        release_of(other);
        // SAFETY: the vtable of the console factory is `s.unimplemented` slots long at least, an extra argument is ignored.
        check!(unsafe { slot::<unsafe extern "system" fn(*mut c_void) -> HRESULT>(factory, s.unimplemented)(factory) } == E_NOTIMPL);

        let mut file: *mut c_void = std::ptr::null_mut();
        let mut queue: *mut c_void = std::ptr::null_mut();
        let mut status: *mut c_void = std::ptr::null_mut();
        type OpenFile = unsafe extern "system" fn(*mut c_void, *const u16, *const GUID, *mut *mut c_void) -> HRESULT;
        // SAFETY: the slot holds the engine handler of that name, whose signature this is.
        let open_file = unsafe { slot::<OpenFile>(factory, s.open_file) };
        // SAFETY: as above.
        check!(unsafe { open_file(factory, wide_path.as_ptr(), &QUEUE, &mut file) } == E_NOINTERFACE && file.is_null());
        // SAFETY: as above.
        check!(unsafe { open_file(factory, wide_path.as_ptr(), &FILE, &mut file) } == S_OK && !file.is_null());
        let mut desc = [0u8; 32];
        desc[8..10].copy_from_slice(&16u16.to_le_bytes());
        type CreateQueue = unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT;
        // SAFETY: the slot holds `factory.create_queue`.
        check!(unsafe { slot::<CreateQueue>(factory, s.create_queue)(factory, desc.as_ptr(), &QUEUE, &mut queue) } == S_OK && !queue.is_null());
        type CreateStatus = unsafe extern "system" fn(*mut c_void, u32, *const u8, *const GUID, *mut *mut c_void) -> HRESULT;
        // SAFETY: the slot holds `factory.create_status_array`.
        check!(unsafe { slot::<CreateStatus>(factory, s.create_status)(factory, 2, c"test".as_ptr().cast(), &STATUS, &mut status) } == S_OK && !status.is_null());

        let mut out = vec![0xCDu8; 1024];
        let mut r = DecantConsoleRead::zeroed();
        r.destination = out.as_mut_ptr().cast();
        r.destination_size = out.len() as u32;
        r.file = file; // the console file object, as a title would pass it
        r.offset = 512;
        r.source_size = out.len() as u32;
        // SAFETY: the slots hold `queue.enqueue_request`, `queue.enqueue_status` and `queue.submit`.
        unsafe {
            check!(slot::<unsafe extern "system" fn(*mut c_void, *const DecantConsoleRead) -> HRESULT>(queue, s.request)(queue, &r) == S_OK);
            check!(slot::<unsafe extern "system" fn(*mut c_void, *mut c_void, u32) -> HRESULT>(queue, s.status)(queue, status, 0) == S_OK);
            check!(slot::<unsafe extern "system" fn(*mut c_void) -> HRESULT>(queue, s.submit)(queue) == S_OK);
        }
        // Submit is asynchronous: wait for the status slot through the console vtable.
        // SAFETY: the slots hold `status.is_complete` and `status.get_hresult`.
        let is_complete = unsafe { slot::<unsafe extern "system" fn(*mut c_void, u32) -> bool>(status, s.is_complete) };
        let end = Instant::now() + Duration::from_secs(10);
        // SAFETY: as above.
        while !unsafe { is_complete(status, 0) } && Instant::now() < end {
            std::thread::sleep(Duration::from_millis(1));
        }
        // SAFETY: as above.
        check!(unsafe { is_complete(status, 0) });
        // SAFETY: as above.
        check!(unsafe { slot::<unsafe extern "system" fn(*mut c_void, u32) -> HRESULT>(status, s.get_hresult)(status, 0) } == S_OK);
        check!(out[..] == data[512..512 + 1024]);

        let mut bad = r;
        bad.file = factory; // not a file object of this layer
        // SAFETY: the slot holds `queue.enqueue_request`.
        check!(unsafe { slot::<unsafe extern "system" fn(*mut c_void, *const DecantConsoleRead) -> HRESULT>(queue, s.request)(queue, &bad) } == E_INVALIDARG);

        release_of(status);
        release_of(queue);
        release_of(file);
        check!(release_of(factory) == 0);
        let _ = std::fs::remove_file(&path);
        0
    }

    /// The IID of the first `iid` line of the factory interface of a spec file, as the C++ test's
    /// `sscanf` loop found it: `IIDFromString` wants the braces.
    fn factory_iid_of(path: &OsStr) -> Option<GUID> {
        let text = std::fs::read(path).ok()?;
        let mut in_factory = false;
        for line in String::from_utf8_lossy(&text).lines() {
            let mut words = line.split_whitespace();
            let (Some(a), b) = (words.next(), words.next().unwrap_or("")) else { continue };
            if a == "interface" {
                in_factory = b == "factory";
            } else if in_factory
                && a == "iid"
                && b.starts_with('{')
                && let Some(guid) = parse_guid(b)
            {
                return Some(GUID::from_u128(guid.0));
            }
        }
        None
    }

    /// Fills a template spec as one would once the executable gives the IIDs: each `iid unknown
    /// <name>` of the base interfaces gets a test GUID. The X1/X2 lines stay unknown.
    fn fill_template(path: &OsStr) -> Option<Vec<u8>> {
        let mut text = std::fs::read(path).ok()?;
        const FILLS: [(&str, &str); 4] = [
            ("iid unknown IDStorageFactoryX", "iid {7e57a000-0000-4000-8000-000000000001} IDStorageFactoryX"),
            ("iid unknown IDStorageFileX", "iid {7e57a000-0000-4000-8000-000000000002} IDStorageFileX"),
            ("iid unknown IDStorageQueueX", "iid {7e57a000-0000-4000-8000-000000000003} IDStorageQueueX"),
            ("iid unknown IDStorageStatusArrayX", "iid {7e57a000-0000-4000-8000-000000000004} IDStorageStatusArrayX"),
        ];
        for (from, to) in FILLS {
            let from = from.as_bytes();
            let mut at = 0;
            loop {
                let found = text.get(at..)?.windows(from.len()).position(|w| w == from)? + at;
                if matches!(text.get(found + from.len()), Some(b'\r' | b'\n' | b' ')) {
                    at = found;
                    break; // the base interface, not X1/X2
                }
                at = found + from.len();
            }
            text.splice(at..at + from.len(), to.bytes());
        }
        Some(text)
    }

    pub fn run(dll: &Dll, args: &[OsString]) -> u8 {
        let mode = args.first().map(|m| m.to_string_lossy().into_owned()).unwrap_or_default();
        let second = args.get(1).map(|a| a.to_string_lossy().into_owned());
        match (mode.as_str(), args.get(1)) {
            ("none", _) => {
                set_spec_path(temp_path("decant_no_such_spec.abi").as_os_str());
                let (hr, factory) = dll.get_factory(&FACTORY);
                check!(hr == E_NOINTERFACE && factory.is_null());
                let (hr, project) = dll.get_factory(&IID_DECANT_FACTORY);
                check!(hr == S_OK && !project.is_null());
            }
            ("spec", Some(which)) if which.to_str() == Some("a") => {
                check!(use_spec(SPEC_A.as_bytes(), "decant_abi_spec_a.abi"));
                let failed = scenario(dll, &SLOTS_A);
                if failed != 0 {
                    return failed;
                }
            }
            ("spec", Some(which)) if which.to_str() == Some("b") => {
                check!(use_spec(SPEC_B.as_bytes(), "decant_abi_spec_b.abi"));
                let failed = scenario(dll, &SLOTS_B);
                if failed != 0 {
                    return failed;
                }
            }
            ("bad", _) => {
                check!(use_spec(SPEC_BAD.as_bytes(), "decant_abi_spec_bad.abi"));
                let (hr, factory) = dll.get_factory(&FACTORY);
                check!(hr == E_NOINTERFACE && factory.is_null());
            }
            ("template", Some(path)) => {
                // As shipped, the template has no IID: every console IID is refused.
                set_spec_path(path);
                let Some(filled) = fill_template(path) else { return fail(line!(), "fillTemplate(argv[2], filled)") };
                check!(use_spec(&filled, "decant_abi_template_filled.abi"));
                // The factory's tuning slots (6 SetCpuAffinity, 7 SetDebugFlags, 8 SetStagingBufferSize) are
                // accepted and logged.
                let (hr, factory) = dll.get_factory(&FACTORY);
                check!(hr == S_OK && !factory.is_null());
                let factory = factory.take();
                // SAFETY: the template maps slots 6 to 8 to the three tuning handlers, with these signatures.
                unsafe {
                    check!(slot::<unsafe extern "system" fn(*mut c_void, u64) -> HRESULT>(factory, 6)(factory, 1) == S_OK);
                    slot::<unsafe extern "system" fn(*mut c_void, u32)>(factory, 7)(factory, 0);
                    check!(slot::<unsafe extern "system" fn(*mut c_void, u32) -> HRESULT>(factory, 8)(factory, 1 << 20) == S_OK);
                }
                check!(release_of(factory) == 0);
                // Filled with the IIDs, the hypothesis order runs the whole scenario through the console vtables
                // (slot 9 of the factory is not identified: E_NOTIMPL).
                let failed = scenario(dll, &Slots { create_queue: 3, open_file: 4, create_status: 5, unimplemented: 9, request: 3, status: 4, submit: 6, is_complete: 3, get_hresult: 4 });
                if failed != 0 {
                    return failed;
                }
            }
            ("refused", Some(path)) => {
                set_spec_path(path);
                let (hr, factory) = dll.get_factory(&FACTORY);
                check!(hr == E_NOINTERFACE && factory.is_null());
            }
            ("parse", Some(path)) => {
                let Some(iid) = factory_iid_of(path) else { return fail(line!(), "factoryIidOf(argv[2], iid)") };
                set_spec_path(path);
                let (hr, factory) = dll.get_factory(&iid);
                check!(hr == S_OK && !factory.is_null());
                let factory = factory.take();
                let mut other = std::ptr::null_mut();
                check!(query(factory, &iid, &mut other) == S_OK);
                release_of(other);
                check!(release_of(factory) == 0);
            }
            ("retail", Some(path)) => {
                // What a retail title does: DStorageGetFactory with its IID, slot 8 with 0x2000000 (a
                // negative result traps it), then slot 10 with a 0x20 byte description and the queue IID
                // {3c02cb0c-c050-4195-99a1-a1ce94ba33c5} (a negative result traps it too), then its first queue
                // calls: slot 3 with one request and slot 8 (Close, no argument, result ignored).
                // Slot 9 stays unidentified: E_NOTIMPL, no crash.
                let Some(iid) = factory_iid_of(path) else { return fail(line!(), "factoryIidOf(argv[2], iid)") };
                set_spec_path(path);
                let (hr, factory) = dll.get_factory(&iid);
                check!(hr == S_OK && !factory.is_null());
                let factory = factory.take();
                let queue_iid = GUID::from_u128(0x3c02cb0c_c050_4195_99a1_a1ce94ba33c5);
                let file_iid = GUID::from_u128(0x43704f9f_27c6_4a9f_8815_d16527894df4);
                let data: Vec<u8> = (0..4096usize).map(|i| (i * 5 + 1) as u8).collect();
                let data_path = temp_path("decant_dstorage_retail_data.bin");
                check!(std::fs::write(&data_path, &data).is_ok());
                let wide_path: Vec<u16> = crate::testapi::wide(data_path.as_os_str()).into_iter().chain(std::iter::once(0)).collect();
                let mut desc = [0u8; 0x20];
                desc[8..10].copy_from_slice(&16u16.to_le_bytes());
                desc[0x18] = 1;
                let mut queue: *mut c_void = std::ptr::null_mut();
                let mut file: *mut c_void = std::ptr::null_mut();
                let mut out = vec![0xCDu8; 1024];
                let mut r = DecantConsoleRead::zeroed();
                r.destination = out.as_mut_ptr().cast();
                r.destination_size = out.len() as u32;
                r.offset = 256;
                r.source_size = out.len() as u32;
                // SAFETY: retail_title.abi maps slot 8 to SetStagingBufferSize(u32), slot 10 to factory.create_queue,
                // slot 4 to factory.open_file, queue slot 3 to queue.enqueue_request, 6 to queue.submit, and
                // leaves factory slot 9 and queue slot 8 unmapped.
                unsafe {
                    check!(slot::<unsafe extern "system" fn(*mut c_void, u32) -> HRESULT>(factory, 8)(factory, 0x200_0000) == S_OK);
                    check!(slot::<unsafe extern "system" fn(*mut c_void) -> HRESULT>(factory, 9)(factory) == E_NOTIMPL);
                    type CreateQueue = unsafe extern "system" fn(*mut c_void, *const u8, *const GUID, *mut *mut c_void) -> HRESULT;
                    // The IID of the other title is not this one: refused.
                    check!(slot::<CreateQueue>(factory, 10)(factory, desc.as_ptr(), &QUEUE, &mut queue) == E_NOINTERFACE && queue.is_null());
                    check!(slot::<CreateQueue>(factory, 10)(factory, desc.as_ptr(), &queue_iid, &mut queue) == S_OK && !queue.is_null());
                    type OpenFile = unsafe extern "system" fn(*mut c_void, *const u16, *const GUID, *mut *mut c_void) -> HRESULT;
                    check!(slot::<OpenFile>(factory, 4)(factory, wide_path.as_ptr(), &file_iid, &mut file) == S_OK && !file.is_null());
                    r.file = file;
                    check!(slot::<unsafe extern "system" fn(*mut c_void, *const DecantConsoleRead) -> HRESULT>(queue, 3)(queue, &r) == S_OK);
                    check!(slot::<unsafe extern "system" fn(*mut c_void) -> HRESULT>(queue, 6)(queue) == S_OK);
                    // Close is unmapped: logged, the result is ignored by the title.
                    slot::<unsafe extern "system" fn(*mut c_void) -> HRESULT>(queue, 8)(queue);
                }
                release_of(queue);
                release_of(file);
                let _ = std::fs::remove_file(&data_path);
                check!(release_of(factory) == 0);
            }
            _ => {
                eprintln!("usage: dstorage_abi_test none | spec a | spec b | bad | parse <spec> | retail <spec> | template <spec> | refused <spec>");
                return 2;
            }
        }
        println!("OK dstorage_abi_test {mode} {}", second.unwrap_or_default());
        0
    }

    fn fail(line: u32, what: &str) -> u8 {
        eprintln!("FAIL line {line}: {what}");
        1
    }
}
