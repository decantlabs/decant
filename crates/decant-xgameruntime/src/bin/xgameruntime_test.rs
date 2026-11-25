//! `xgameruntime_test <our xgameruntime.dll> offline|online|preload|direct`: the checks of the
//! former `tests/xgameruntime_test.cpp`, with the same lines and exit codes (the exit code is the
//! number of failed checks, 2 for bad arguments).
//!
//! * `offline`: no PC runtime (`DECANT_XGR_REAL` points nowhere). The 7 exports sit at the PC
//!   ordinals, `InitializeApiImplEx2` accepts the initialisation, `QueryApiImpl` serves the local
//!   XAsync / XTaskQueue / XThread table for `({073B7DCB-...}, {073B7DCB-...})` (a manual queue
//!   is created, fed and dispatched through it) and the local XUser table for
//!   `({01ACD177-...}, {01ACD177-...})` (`XUserAddAsync` with the default-user option completes
//!   with the offline user, signed in, and a token request fails) and the local offline
//!   XNetworking table for `({37E56907-...}, {37E56907-...})` and `({37E56907-...}, {BF2346B2-...})`
//!   (one object; the connectivity hint says no network, the UDP port is 3074) and the local
//!   XStore table for `({0DD112AC-...}, {B09D803C-...})`, refuses every
//!   other couple (the store without a known interface, the other XUser interfaces, an unknown table) with
//!   `REGDB_E_CLASSNOTREG`, names a known console table in the log, and checks that the
//!   interposer imports no network DLL (Winsock, WinHTTP, WinINet, IP helper, DNS).
//! * `online`: built with the public GDK (`DECANT_GDK_ROOT` set when Cargo built it): the GDK's own
//!   loader (`XGameRuntimeInitialize`) picks our DLL when it is already loaded, which forwards to
//!   the PC runtime. Then a task queue is created, a callback submitted and waited for. The log
//!   shows which api_id GUIDs the PC GDK thunks request, to compare with the ones a console title
//!   uses. `preload` loads our DLL first, the way a launcher would. Without the GDK both print
//!   SKIP and pass, as the C test did.
//! * `direct` (new, needs no GDK): calls the same `QueryApiImpl` couples through our DLL and
//!   straight on the PC runtime in System32, and checks that the answers are the same and that
//!   the log recorded the forwarding. Prints SKIP when there is no PC runtime.

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    std::process::ExitCode::from(run::main())
}

#[cfg(not(windows))]
fn main() {
    eprintln!("xgameruntime_test runs on Windows only");
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod run {
    use std::ffi::{OsStr, c_void};

    use decant_testkit::Library;
    use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
    use windows_sys::core::GUID;

    const LOG: &str = "xgameruntime_test.log";
    const REGDB_E_CLASSNOTREG: i32 = 0x8004_0154_u32 as i32;

    type QueryFn = unsafe extern "system" fn(*const GUID, *const GUID, *mut *mut c_void) -> i32;
    type InitFn = unsafe extern "system" fn(usize, usize, usize, usize) -> usize;

    const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> GUID {
        GUID { data1, data2, data3, data4 }
    }

    /// The XAsync / XTaskQueue / XThread table the PC GDK thunks ask for.
    const TABLE: GUID = guid(0x073B7DCB, 0x1FCF, 0x4030, [0x94, 0xBE, 0xE3, 0xC9, 0xEB, 0x62, 0x34, 0x28]);

    struct Test {
        fails: u32,
    }

    impl Test {
        fn check(&mut self, ok: bool, what: &str) {
            println!("{} {what}", if ok { "PASS" } else { "FAIL" });
            if !ok {
                self.fails += 1;
            }
        }
    }

    fn read_log() -> String {
        std::fs::read(LOG).map(|bytes| String::from_utf8_lossy(&bytes).into_owned()).unwrap_or_default()
    }

    pub fn main() -> u8 {
        let args = decant_testkit::args();
        if args.len() < 2 {
            eprintln!("usage: xgameruntime_test <our xgameruntime.dll> offline|online|direct");
            return 2;
        }
        let _ = std::fs::remove_file(LOG);
        // SAFETY: the program is still single threaded, nothing reads the environment meanwhile.
        unsafe { std::env::set_var("DECANT_XGR_LOG", LOG) };
        let mut test = Test { fails: 0 };
        let dll = args[0].as_os_str();
        let status = match args[1].to_string_lossy().as_ref() {
            "offline" => offline(&mut test, dll),
            "direct" => direct(&mut test, dll),
            #[cfg(decant_with_gdk)]
            "online" => online::run(&mut test, dll, false),
            #[cfg(decant_with_gdk)]
            "preload" => online::run(&mut test, dll, true),
            #[cfg(not(decant_with_gdk))]
            "online" | "preload" => {
                println!("SKIP: built without DECANT_GDK_ROOT");
                0
            }
            _ => {
                eprintln!("unknown mode");
                return 2;
            }
        };
        status.min(255) as u8
    }

    fn offline(test: &mut Test, dll: &OsStr) -> u32 {
        // SAFETY: the program is still single threaded.
        unsafe { std::env::set_var("DECANT_XGR_REAL", "Z:\\nowhere\\xgameruntime.dll") };
        let lib = Library::open(dll);
        test.check(lib.is_ok(), "the interposer loads");
        let Ok(lib) = lib else { return 1 };
        let names = ["DllCanUnloadNow", "InitializeApiImpl", "InitializeApiImplEx", "InitializeApiImplEx2", "QueryApiImpl", "UninitializeApiImpl", "XErrorReport"];
        // DllCanUnloadNow is a PRIVATE COM export and keeps a linker chosen ordinal (it is never
        // imported by ordinal), the other six match the PC runtime's ordinals 2 to 7.
        let mut ordinals = !lib.address(names[0]).is_null();
        for (i, name) in names.iter().enumerate().skip(1) {
            let by_name = lib.address(name);
            // SAFETY: the module is loaded. An ordinal is passed as the low word of the name pointer.
            let by_ordinal = unsafe { GetProcAddress(lib.handle(), (i + 1) as *const u8) }.map_or(std::ptr::null(), |f| f as *const c_void);
            if by_name.is_null() || by_name != by_ordinal {
                println!("  ordinal {} is not {name}", i + 1);
                ordinals = false;
            }
        }
        test.check(ordinals, "the exports sit at the PC runtime's ordinals 2-7, DllCanUnloadNow by name");

        // SAFETY: the exports have these signatures. `InitializeApiImplEx2` takes at least four
        // integers, and the interposer returns without a PC runtime before it reads more.
        let init: Option<InitFn> = unsafe { lib.get("InitializeApiImplEx2") };
        // SAFETY: `InitFn` is called with four integers, the interposer reads them as such.
        let init_result = init.map(|f| unsafe { f(1, 2, 3, 4) } as u32 as i32);
        test.check(init_result == Some(0), "without a PC runtime, InitializeApiImplEx2 accepts the initialisation (S_OK)");
        // SAFETY: `QueryFn` is the export's signature.
        let query: Option<QueryFn> = unsafe { lib.get("QueryApiImpl") };
        let unknown = guid(0, 0, 0, [0xC0, 0, 0, 0, 0, 0, 0, 0x46]);
        let mut out = std::ptr::dangling_mut::<c_void>(); // address 1, as in the C test
        // SAFETY: both GUID pointers and `out` are valid for the call.
        let hr = query.map_or(0, |f| unsafe { f(&TABLE, &unknown, &mut out) });
        test.check(hr == REGDB_E_CLASSNOTREG, "without a PC runtime, QueryApiImpl refuses with REGDB_E_CLASSNOTREG");
        let mut log = read_log();
        test.check(log.contains("{073B7DCB-1FCF-4030-94BE-E3C9EB623428} = XAsync / XTaskQueue / XThread"), "the log names the known console table");
        test.check(log.contains("PC runtime not loaded"), "the log says the PC runtime is missing");
        test.check(
            log.contains("riid {00000000-0000-0000-C000-000000000046} = version not listed for this family"),
            "an unlisted riid of a known family is reported as such",
        );
        // The couple (api_id, riid) selects the table.
        let net = guid(0x37E56907, 0x2F10, 0x41E8, [0xB7, 0x2F, 0x36, 0xED, 0xB1, 0x85, 0x33, 0x1A]);
        let net_iid = guid(0xbf2346b2, 0x39af, 0x4658, [0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3]);
        let sys = guid(0xe349bd1a, 0xfc20, 0x4e40, [0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f]);
        let sys_partial = guid(0x67ce4bfc, 0x1111, 0x2222, [0, 0, 0, 0, 0, 0, 0x7a, 0x85]);
        let user = guid(0x01acd177, 0x91f9, 0x4763, [0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0]);
        let store = guid(0x0dd112ac, 0x7c24, 0x448c, [0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c]);
        let store_context = guid(0xb09d803c, 0x2414, 0x4a05, [0x82, 0xc6, 0x66, 0xdf, 0xdc, 0x9e, 0x9a, 0x44]);
        let user_store = guid(0x079415e3, 0x6727, 0x437f, [0x8e, 0x9d, 0x8f, 0x8f, 0x9b, 0x24, 0x39, 0xf7]);
        let mut table = std::ptr::null_mut::<c_void>();
        let (mut table_hr, mut user_hr, mut user_table) = (-1, -1, std::ptr::null_mut::<c_void>());
        let unknown_table = guid(0x12345678, 0x9abc, 0xdef0, [1, 2, 3, 4, 5, 6, 7, 8]);
        let mut refused = Vec::new();
        let (mut net_hr, mut net_table, mut net_hr2, mut net_table2) = (-1, std::ptr::null_mut::<c_void>(), -1, std::ptr::null_mut::<c_void>());
        let (mut store_hr, mut store_table) = (-1, std::ptr::null_mut::<c_void>());
        if let Some(f) = query {
            // SAFETY: as above.
            unsafe {
                table_hr = f(&TABLE, &TABLE, &mut table);
                f(&sys, &sys_partial, &mut out);
                user_hr = f(&user, &user, &mut user_table);
                net_hr = f(&net, &net_iid, &mut net_table);
                net_hr2 = f(&net, &net, &mut net_table2);
                store_hr = f(&store, &store_context, &mut store_table);
                for (api, riid) in [(&store, &store), (&user, &user_store), (&unknown_table, &unknown_table)] {
                    let mut o = std::ptr::dangling_mut::<c_void>();
                    let hr = f(api, riid, &mut o);
                    refused.push(hr == REGDB_E_CLASSNOTREG && o.is_null());
                }
            }
        }
        test.check(table_hr == 0 && !table.is_null(), "without a PC runtime, QueryApiImpl serves the local XAsync table (S_OK, an object)");
        test.check(table.is_null() || local_queue_round_trip(table), "the local table runs a manual task queue through its slots (this first)");
        test.check(user_hr == 0 && !user_table.is_null(), "without a PC runtime, QueryApiImpl serves the local XUser table (S_OK, an object)");
        test.check(user_table.is_null() || table.is_null() || local_user_round_trip(user_table, table), "XUserAddAsync (default user) completes with a signed-in user; a token request fails");
        test.check(net_hr == 0 && net_hr2 == 0 && !net_table.is_null() && net_table == net_table2, "without a PC runtime, QueryApiImpl serves the local XNetworking table under both riids (S_OK, one object)");
        test.check(net_table.is_null() || local_networking_offline(net_table), "the XNetworking connectivity hint says no network (level None) and the UDP port is 3074");
        test.check(store_hr == 0 && !store_table.is_null(), "without a PC runtime, QueryApiImpl serves the local XStore table for its context interface (S_OK, an object)");
        test.check(refused.len() == 3 && refused.iter().all(|r| *r), "the store table without a known interface, the XUser store interface and an unknown table are still refused with a null object");
        let net_dlls = network_imports(&lib);
        if !net_dlls.is_empty() {
            println!("  network imports: {}", net_dlls.join(", "));
        }
        test.check(net_dlls.is_empty(), "the interposer imports no network DLL (ws2_32, wsock32, winhttp, wininet, iphlpapi, dnsapi, mswsock)");
        log = read_log();
        test.check(
            log.contains(
                "{073B7DCB-1FCF-4030-94BE-E3C9EB623428} = XAsync / XTaskQueue / XThread (32 slots) [hypothesis], riid {073B7DCB-1FCF-4030-94BE-E3C9EB623428} = riid = api_id",
            ),
            "the couple (api_id, api_id) names the XAsync table version",
        );
        test.check(log.contains("local XAsync/XTaskQueue/XThread table (no PC runtime)"), "the log says the local table was served");
        test.check(log.contains("riid {01ACD177-91F9-4763-A38E-CCBB55CE32E0} = riid = api_id (up to 38 slots depending on the version)) -> 0x00000000") && log.contains("local XUser table (no PC runtime)"), "the log says the local XUser table was served");
        test.check(log.contains("XUserAddAsync(options 0x1, async ") && log.contains("-> local offline user 'Player1' (no PC runtime)"), "the log shows XUserAddAsync and the offline user");
        test.check(log.contains("local XUser table: slot 7 XUserAddAsync(options 1, async_block 0x") && log.contains("local XAsync/XTaskQueue/XThread table: slot 12 XTaskQueueCreate(work 0, completion 0, out 0x"), "the calls on the local tables are logged with their arguments");
        test.check(
            log.contains("= XNetworking [hypothesis], riid {BF2346B2-39AF-4658-B5EA-44713C7E83B3} = riid {bf2346b2-...}"),
            "the couple (XNetworking, second riid) is recognised",
        );
        test.check(
            log.contains("riid {BF2346B2-39AF-4658-B5EA-44713C7E83B3} = riid {bf2346b2-...}) -> 0x00000000") && log.contains("local XNetworking table (no PC runtime)"),
            "the log says the local XNetworking table was served",
        );
        test.check(
            log.contains("local XNetworking table: slot 15 XNetworkingGetConnectivityHint(hint 0x") && log.contains("XNetworkingGetConnectivityHint -> offline: level 1 (None)"),
            "the log shows the connectivity hint call and its offline answer",
        );
        test.check(
            log.contains("= XSystem [hypothesis], riid {67CE4BFC-1111-2222-0000-000000007A85} = riid 67ce4bfc-...-7a85 (partial GUID), matched on the first field only"),
            "a riid known by its first field only is reported as a partial match",
        );
        test.fails
    }

    static RAN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    unsafe extern "system" fn on_work(_ctx: *mut c_void, _canceled: bool) {
        RAN.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Creates a manual queue through slot 12, submits a callback (18), dispatches it (16) and
    /// closes the handle (17), each slot called with the object first, as the GDK thunks do.
    fn local_queue_round_trip(object: *mut c_void) -> bool {
        type Create = unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> i32;
        type Submit = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *mut c_void, unsafe extern "system" fn(*mut c_void, bool)) -> i32;
        type Dispatch = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32) -> bool;
        type Close = unsafe extern "system" fn(*mut c_void, *mut c_void);
        // SAFETY: `object` is the interposer's local table, a vtable of 32 slots with these
        // signatures at 12, 16, 17 and 18.
        unsafe {
            let vtbl = *(object as *const *const usize);
            let create = std::mem::transmute::<usize, Create>(*vtbl.add(12));
            let dispatch = std::mem::transmute::<usize, Dispatch>(*vtbl.add(16));
            let close = std::mem::transmute::<usize, Close>(*vtbl.add(17));
            let submit = std::mem::transmute::<usize, Submit>(*vtbl.add(18));
            let mut queue = std::ptr::null_mut();
            if create(object, 0, 0, &mut queue) != 0 || queue.is_null() {
                return false;
            }
            let before = RAN.load(std::sync::atomic::Ordering::SeqCst);
            let ok = submit(object, queue, 0, std::ptr::null_mut(), on_work) == 0
                && RAN.load(std::sync::atomic::Ordering::SeqCst) == before
                && dispatch(object, queue, 0, 0)
                && RAN.load(std::sync::atomic::Ordering::SeqCst) == before + 1;
            close(object, queue);
            ok
        }
    }

    /// Through the slots of the local XUser table, `this` first: `XUserAddAsync` (7) with
    /// `AddDefaultUserSilently`, waited for with `XAsyncGetStatus` of the XAsync table, then
    /// `XUserAddResult` (8), `XUserGetLocalId` (9), `XUserGetState` (14), a token request (23)
    /// that fails with `0x80072EFD`, and `XUserCloseHandle` (4).
    fn local_user_round_trip(object: *mut c_void, xasync: *mut c_void) -> bool {
        #[repr(C)]
        struct Block {
            queue: *mut c_void,
            context: *mut c_void,
            callback: usize,
            internal: [u8; 32],
        }
        type AddAsync = unsafe extern "system" fn(*mut c_void, u32, *mut Block) -> i32;
        type AddResult = unsafe extern "system" fn(*mut c_void, *mut Block, *mut *mut c_void) -> i32;
        type GetU64 = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u64) -> i32;
        type GetU32 = unsafe extern "system" fn(*mut c_void, *mut c_void, *mut u32) -> i32;
        type Token = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, *const u8, *const u8, usize, *const c_void, usize, *const c_void, *mut Block) -> i32;
        type Close = unsafe extern "system" fn(*mut c_void, *mut c_void);
        const E_CANNOT_CONNECT: i32 = 0x8007_2EFD_u32 as i32;
        // SAFETY: `object` is the interposer's local XUser table, a vtable of 50 slots with these
        // signatures; the XAsync table's slot 3 is XAsyncGetStatus(this, block, wait).
        unsafe {
            let vtbl = *(object as *const *const usize);
            let add_async = std::mem::transmute::<usize, AddAsync>(*vtbl.add(7));
            let add_result = std::mem::transmute::<usize, AddResult>(*vtbl.add(8));
            let get_local_id = std::mem::transmute::<usize, GetU64>(*vtbl.add(9));
            let get_state = std::mem::transmute::<usize, GetU32>(*vtbl.add(14));
            let token = std::mem::transmute::<usize, Token>(*vtbl.add(23));
            let close = std::mem::transmute::<usize, Close>(*vtbl.add(4));
            type GetStatus = unsafe extern "system" fn(*mut c_void, *mut Block, bool) -> i32;
            let get_status = std::mem::transmute::<usize, GetStatus>(*(*(xasync as *const *const usize)).add(3));

            let mut b: Block = std::mem::zeroed();
            if add_async(object, 1, &mut b) != 0 || get_status(xasync, &mut b, true) != 0 {
                return false;
            }
            let mut user = std::ptr::null_mut();
            if add_result(object, &mut b, &mut user) != 0 || user.is_null() {
                return false;
            }
            let (mut id, mut state) = (0u64, 9u32);
            let ok = get_local_id(object, user, &mut id) == 0 && id != 0 && get_state(object, user, &mut state) == 0 && state == 0;
            let mut b: Block = std::mem::zeroed();
            let n = std::ptr::null();
            let token_ok = token(object, user, 0, c"GET".as_ptr().cast(), c"https://example.invalid/".as_ptr().cast(), 0, n, 0, n, &mut b) == 0
                && get_status(xasync, &mut b, true) == E_CANNOT_CONNECT;
            close(object, user);
            ok && token_ok
        }
    }

    /// Through the slots of the local XNetworking table, `this` first:
    /// `XNetworkingGetConnectivityHint` (15) answers level 1 (`None`), never internet access, and
    /// `XNetworkingQueryPreferredLocalUdpMultiplayerPort` (3) answers 3074.
    fn local_networking_offline(object: *mut c_void) -> bool {
        #[repr(C)]
        struct Hint {
            level: u32,
            cost: u32,
            iana: u32,
            initialized: bool,
            approaching: bool,
            over: bool,
            roaming: bool,
        }
        type GetHint = unsafe extern "system" fn(*mut c_void, *mut Hint) -> i32;
        type GetPort = unsafe extern "system" fn(*mut c_void, *mut u16) -> i32;
        // SAFETY: `object` is the interposer's local XNetworking table, a vtable of 24 slots with
        // these signatures at 3 and 15.
        unsafe {
            let vtbl = *(object as *const *const usize);
            let get_hint = std::mem::transmute::<usize, GetHint>(*vtbl.add(15));
            let get_port = std::mem::transmute::<usize, GetPort>(*vtbl.add(3));
            let mut hint = Hint { level: 3, cost: 0, iana: 0, initialized: false, approaching: true, over: true, roaming: true };
            let mut port = 0u16;
            let hint_ok = get_hint(object, &mut hint) == 0
                && hint.level == 1
                && hint.cost == 1
                && hint.iana == 6
                && hint.initialized
                && !hint.approaching
                && !hint.over
                && !hint.roaming;
            if !hint_ok {
                println!("  connectivity hint: level {}, cost {}, interface {}, initialized {}", hint.level, hint.cost, hint.iana, hint.initialized);
            }
            hint_ok && get_port(object, &mut port) == 0 && port == 3074
        }
    }

    /// The network DLLs among the DLLs the loaded interposer imports, normal and delay-loaded,
    /// read from its import directories.
    fn network_imports(lib: &Library) -> Vec<String> {
        const NETWORK: [&str; 7] = ["ws2_32.dll", "wsock32.dll", "winhttp.dll", "wininet.dll", "iphlpapi.dll", "dnsapi.dll", "mswsock.dll"];
        let base = lib.handle() as *const u8;
        let mut names = Vec::new();
        // SAFETY: `base` is the mapped image of a loaded PE32+ DLL: the DOS header's e_lfanew
        // (at 0x3C) leads to the NT headers, whose optional header (at +24) holds the data
        // directories at +112 (import at index 1, delay import at index 13), each an RVA and a
        // size into the image.
        unsafe {
            let nt = base.add((base.add(0x3C) as *const u32).read_unaligned() as usize);
            let dirs = nt.add(24 + 112);
            let dir = |i: usize| ((dirs.add(i * 8) as *const u32).read_unaligned() as usize, (dirs.add(i * 8 + 4) as *const u32).read_unaligned() as usize);
            let name_at = |rva: usize| std::ffi::CStr::from_ptr(base.add(rva).cast()).to_string_lossy().to_ascii_lowercase();
            // IMAGE_IMPORT_DESCRIPTOR: 20 bytes, Name RVA at +12; a zero entry ends the list.
            let (rva, size) = dir(1);
            if rva != 0 && size != 0 {
                let mut d = base.add(rva);
                while (d.add(12) as *const u32).read_unaligned() != 0 {
                    names.push(name_at((d.add(12) as *const u32).read_unaligned() as usize));
                    d = d.add(20);
                }
            }
            // IMAGE_DELAYLOAD_DESCRIPTOR: 32 bytes, DllNameRVA at +4; a zero entry ends the list.
            let (rva, size) = dir(13);
            if rva != 0 && size != 0 {
                let mut d = base.add(rva);
                while (d.add(4) as *const u32).read_unaligned() != 0 {
                    names.push(name_at((d.add(4) as *const u32).read_unaligned() as usize));
                    d = d.add(32);
                }
            }
        }
        if names.is_empty() {
            return vec!["(no import read: the check itself failed)".to_string()];
        }
        names.retain(|n| NETWORK.contains(&n.as_str()));
        names
    }

    /// The PC runtime in the system folder, when there is one.
    fn pc_runtime() -> Option<std::path::PathBuf> {
        use std::os::windows::ffi::OsStringExt;
        use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;
        let mut dir = [0u16; 260];
        // SAFETY: `dir` has room for the 260 units passed.
        let n = unsafe { GetSystemDirectoryW(dir.as_mut_ptr(), dir.len() as u32) } as usize;
        let path = std::path::PathBuf::from(std::ffi::OsString::from_wide(&dir[..n.min(dir.len())])).join("xgameruntime.dll");
        (n != 0 && path.is_file()).then_some(path)
    }

    fn direct(test: &mut Test, dll: &OsStr) -> u32 {
        let Some(real_path) = pc_runtime() else {
            println!("SKIP: no PC runtime in the system folder");
            return 0;
        };
        // SAFETY: the program is still single threaded.
        unsafe { std::env::remove_var("DECANT_XGR_REAL") };
        let real = Library::open_altered(real_path.as_os_str());
        test.check(real.is_ok(), "the PC runtime loads");
        let ours = Library::open(dll);
        test.check(ours.is_ok(), "the interposer loads");
        let (Ok(real), Ok(ours)) = (real, ours) else { return test.fails };
        // SAFETY: `QueryFn` is the signature of both exports.
        let (real_query, our_query): (Option<QueryFn>, Option<QueryFn>) = unsafe { (real.get("QueryApiImpl"), ours.get("QueryApiImpl")) };
        let (Some(real_query), Some(our_query)) = (real_query, our_query) else {
            test.check(false, "both runtimes export QueryApiImpl");
            return test.fails;
        };
        let unknown = guid(0, 0, 0, [0xC0, 0, 0, 0, 0, 0, 0, 0x46]);
        let net = guid(0x37E56907, 0x2F10, 0x41E8, [0xB7, 0x2F, 0x36, 0xED, 0xB1, 0x85, 0x33, 0x1A]);
        let couples = [(&TABLE, &TABLE), (&TABLE, &unknown), (&net, &net)];
        let mut same = true;
        let mut async_objects = (std::ptr::null_mut::<c_void>(), std::ptr::null_mut::<c_void>());
        for (api, riid) in couples {
            let (mut real_out, mut our_out) = (std::ptr::null_mut::<c_void>(), std::ptr::null_mut::<c_void>());
            // SAFETY: the GUID pointers and the out locations are valid for the calls.
            let (real_hr, our_hr) = unsafe { (real_query(api, riid, &mut real_out), our_query(api, riid, &mut our_out)) };
            println!("  QueryApiImpl {:08X} -> PC runtime 0x{:08x}, interposer 0x{:08x}", api.data1, real_hr as u32, our_hr as u32);
            // A PC object reaches the title wrapped in the logging proxy (`runtime_proxy`), so only
            // whether there is one can match.
            same &= real_hr == our_hr && real_out.is_null() == our_out.is_null();
            if std::ptr::eq(api, &TABLE) && std::ptr::eq(riid, &TABLE) {
                async_objects = (real_out, our_out);
            }
        }
        test.check(same, "through the interposer, QueryApiImpl answers what the PC runtime answers");
        if let (real, proxy) = async_objects
            && !real.is_null()
            && !proxy.is_null()
        {
            // Slot 3, `XAsyncGetStatus`, on a block no runtime started: called directly and through the proxy.
            type GetStatus = unsafe extern "system" fn(*mut c_void, *mut [usize; 7], bool) -> i32;
            let status = |object: *mut c_void| {
                let mut block = [0usize; 7];
                // SAFETY: both objects start with a vtable whose slot 3 is `XAsyncGetStatus`.
                unsafe { std::mem::transmute::<usize, GetStatus>(*(*(object as *const *const usize)).add(3))(object, &mut block, false) }
            };
            let (direct, proxied) = (status(real), status(proxy));
            println!("  XAsyncGetStatus of an idle block -> PC runtime 0x{:08x}, through the proxy 0x{:08x}", direct as u32, proxied as u32);
            test.check(direct == proxied, "a call through the logging proxy answers what the PC runtime answers");
        }
        let log = read_log();
        test.check(log.contains("forwarding to "), "the log says where the calls are forwarded");
        test.check(log.contains("QueryApiImpl(api {073B7DCB-1FCF-4030-94BE-E3C9EB623428}") && !log.contains("(no PC runtime)"), "the log recorded the calls with the PC runtime's answers");
        test.check(log.contains("slot 3 XAsyncGetStatus -> "), "the proxy logged the slot call");
        test.fails
    }

    #[cfg(decant_with_gdk)]
    mod online {
        use std::ffi::c_void;
        use std::ffi::OsStr;

        use decant_testkit::Library;
        use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, MAX_PATH, WAIT_OBJECT_0};
        use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
        use windows_sys::Win32::System::ProcessStatus::K32EnumProcessModules;
        use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentProcess, SetEvent, WaitForSingleObject};

        use super::{Test, read_log};

        // The public GDK's own entry points, from its `xgameruntime.lib` (linked by build.rs).
        unsafe extern "C" {
            fn XGameRuntimeInitialize() -> i32;
            fn XGameRuntimeUninitialize();
            fn XTaskQueueCreate(work_mode: i32, completion_mode: i32, queue: *mut *mut c_void) -> i32;
            fn XTaskQueueSubmitCallback(queue: *mut c_void, port: i32, context: *mut c_void, callback: unsafe extern "system" fn(*mut c_void, bool)) -> i32;
            fn XTaskQueueCloseHandle(queue: *mut c_void);
        }

        /// `XTaskQueueDispatchMode::ThreadPool` and `XTaskQueuePort::Work`.
        const THREAD_POOL: i32 = 1;
        const PORT_WORK: i32 = 0;

        unsafe extern "system" fn on_work(context: *mut c_void, _canceled: bool) {
            // SAFETY: `context` is the event handle `run` passed.
            unsafe { SetEvent(context as HANDLE) };
        }

        /// Lists every module named xgameruntime.dll in the process.
        fn list_runtimes() {
            let mut modules = [std::ptr::null_mut::<c_void>(); 1024];
            let mut need = 0u32;
            // SAFETY: the buffer size passed is the real one.
            if unsafe { K32EnumProcessModules(GetCurrentProcess(), modules.as_mut_ptr(), size_of_val(&modules) as u32, &mut need) } == 0 {
                return;
            }
            for module in modules.iter().take((need as usize / size_of::<*mut c_void>()).min(1024)) {
                let mut path = [0u16; MAX_PATH as usize];
                // SAFETY: the buffer size passed is the real one.
                let n = unsafe { GetModuleFileNameW(*module, path.as_mut_ptr(), MAX_PATH) } as usize;
                let text = String::from_utf16_lossy(&path[..n]).to_lowercase();
                if n != 0 && text.contains("xgameruntime.dll") {
                    println!("  loaded: {text}");
                }
            }
        }

        /// `preload`: load our DLL first, the way a launcher would, before the GDK loader runs.
        pub fn run(test: &mut Test, dll: &OsStr, preload: bool) -> u32 {
            let _ours = if preload {
                let lib = Library::open(dll);
                test.check(lib.is_ok(), "preload of the interposer");
                lib.ok()
            } else {
                None
            };
            // SAFETY: no precondition, the GDK loads the runtime.
            let hr = unsafe { XGameRuntimeInitialize() };
            println!("XGameRuntimeInitialize -> 0x{:08x}", hr as u32);
            test.check(hr >= 0, "the game runtime initializes");
            list_runtimes();
            let picked = read_log().contains("InitializeApiImpl");
            println!("INFO the GDK loader {} the interposer", if picked { "went through" } else { "did NOT go through" });
            if preload {
                test.check(picked, "with a preload, the runtime calls go through the interposer");
            }
            if hr < 0 {
                return test.fails;
            }
            let mut queue = std::ptr::null_mut::<c_void>();
            // SAFETY: `queue` is a writable handle location.
            let hr = unsafe { XTaskQueueCreate(THREAD_POOL, THREAD_POOL, &mut queue) };
            test.check(hr >= 0, "XTaskQueueCreate");
            // SAFETY: a manual reset event, closed below.
            let done = unsafe { CreateEventW(std::ptr::null(), 1, 0, std::ptr::null()) };
            if hr >= 0 {
                // SAFETY: `queue` was just created and `on_work` matches the callback type.
                let hr = unsafe { XTaskQueueSubmitCallback(queue, PORT_WORK, done, on_work) };
                test.check(hr >= 0, "XTaskQueueSubmitCallback");
                // SAFETY: `done` is a valid event.
                test.check(unsafe { WaitForSingleObject(done, 5000) } == WAIT_OBJECT_0, "the submitted work ran");
                // SAFETY: `queue` is the handle created above, closed once.
                unsafe { XTaskQueueCloseHandle(queue) };
            }
            // SAFETY: `done` is closed once.
            unsafe { CloseHandle(done) };
            // SAFETY: the runtime was initialized above.
            unsafe { XGameRuntimeUninitialize() };
            print!("--- interposer log ---\n{}", read_log());
            if preload {
                test.check(read_log().contains("QueryApiImpl(api "), "the log recorded the api tables requested");
            }
            test.fails
        }
    }
}
