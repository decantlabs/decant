//! Known xgameruntime API tables, as data. The Rust form of the former `xgr_tables.h`.
//!
//! A GDK title reaches the game runtime through `xgameruntime.dll`: it calls one of the
//! `InitializeApiImpl*` exports once, then `QueryApiImpl(api_id, riid, out)` to get COM style
//! tables whose slots lead to the flat functions (`XAsync*`, `XTaskQueue*`, `XThread*`...).
//!
//! The `(api_id, riid)` COUPLE selects the table: one `api_id` serves several riids (versions of
//! the same family) with different slots. So each entry below is a couple. A family is named as
//! soon as its `api_id` is known, and the log says whether the riid is a listed version.
//!
//! Sources, all facts measured on other titles, no code taken:
//! * One title on another GDK edition: the XAsync / XTaskQueue / XThread table and the slots of
//!   `{073B7DCB-...}`.
//! * The other families, read on the client vtables of the console `xgameruntime.dll` and on
//!   titles' call sites, as listed in a cross check report. Where that report abbreviates a
//!   riid, only its first field is matched (`Match::IidData1`) and the log says so. The full
//!   value is to be read in the title's executable.
//!
//! Everything here is a hypothesis to confirm on the title executable: the interposer logs
//! every couple the title asks for. Nothing here comes from the console SDK.

/// A GUID with the layout of the Windows one, so a pointer from the title can be read as it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Guid {
    pub data1: u32,
    pub data2: u16,
    pub data3: u16,
    pub data4: [u8; 8],
}

/// How an entry matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Match {
    /// `api` and `iid` both equal.
    IidExact,
    /// `api` equal, riid known only by its first field (partial GUID).
    IidData1,
    /// `api` known only by its first field, riid not known.
    ApiData1,
}

#[derive(Debug)]
pub struct KnownTable {
    pub api: Guid,
    /// `Match::IidData1` and `Match::ApiData1`: only `data1` is meaningful.
    pub iid: Guid,
    pub kind: Match,
    /// Family, as logged.
    pub name: &'static str,
    /// Which version, or the partial GUID as written in the report.
    pub iid_note: &'static str,
    pub source: &'static str,
    /// Slot names, index = slot number, `None` = unknown. Slots 0 to 2 are the IUnknown like head.
    pub slots: &'static [Option<&'static str>],
}

/// How well a `(api_id, riid)` couple is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Only the family (`api_id`) is known.
    Family = 0,
    /// The riid matches a listed version by its first field only (partial GUID).
    Partial = 1,
    /// The couple is listed.
    Listed = 2,
}

const fn guid(data1: u32, data2: u16, data3: u16, data4: [u8; 8]) -> Guid {
    Guid { data1, data2, data3, data4 }
}

/// A GUID known by its first field only.
const fn data1(data1: u32) -> Guid {
    guid(data1, 0, 0, [0; 8])
}

/// The known table for the couple `(api, riid)`, with the level of the match. `riid` is `None`
/// when it is unreadable. `None` when the `api_id` is not known at all.
pub fn known(api: &Guid, riid: Option<&Guid>) -> Option<(&'static KnownTable, Level)> {
    let mut family = None;
    let mut partial = None;
    for table in KNOWN_TABLES {
        let api_ok = if table.kind == Match::ApiData1 { api.data1 == table.api.data1 } else { *api == table.api };
        if !api_ok {
            continue;
        }
        family.get_or_insert(table);
        let Some(riid) = riid else { continue };
        if table.kind == Match::IidExact && *riid == table.iid {
            return Some((table, Level::Listed));
        }
        if partial.is_none() && table.kind != Match::IidExact && table.iid.data1 != 0 && riid.data1 == table.iid.data1 {
            partial = Some(table);
        }
    }
    partial.map(|t| (t, Level::Partial)).or(family.map(|t| (t, Level::Family)))
}

/// Slots 27 to 30 as the public PC GDK's `xgameruntime.lib` thunks call them (the local table of
/// `local_table` follows them): 28 `XThreadVerifyNotTimeSensitive`, 29
/// `XThreadAssertNotTimeSensitive`, 30 `XThreadIsTimeSensitive`. One title puts
/// `XThreadIsTimeSensitive` at 28, another at 30, as the PC thunks do.
static SLOTS_073B7DCB: [Option<&str>; 32] = [
    None,
    None,
    None,
    Some("XAsyncGetStatus"),                                                          // 3
    Some("XAsyncGetResultSize"),                                                      // 4
    Some("XAsyncCancel"),                                                             // 5
    Some("XAsyncRun"),                                                                // 6
    Some("XAsyncBegin"),                                                              // 7
    None,                                                                             // 8
    Some("XAsyncSchedule"),                                                           // 9
    Some("XAsyncComplete"),                                                           // 10
    Some("XAsyncGetResult"),                                                          // 11
    Some("XTaskQueueCreate"),                                                         // 12
    Some("XTaskQueueCreateComposite"),                                                // 13
    Some("XTaskQueueGetPort"),                                                        // 14
    Some("XTaskQueueDuplicateHandle"),                                                // 15
    Some("XTaskQueueDispatch"),                                                       // 16
    Some("XTaskQueueCloseHandle"),                                                    // 17
    Some("XTaskQueueSubmitCallback"),                                                 // 18
    Some("XTaskQueueSubmitDelayedCallback"),                                          // 19
    Some("XTaskQueueRegisterWaiter"),                                                 // 20 (not served by other titles)
    Some("XTaskQueueUnregisterWaiter"),                                               // 21 (not served by other titles)
    Some("XTaskQueueTerminate"),                                                      // 22
    Some("XTaskQueueRegisterMonitor"),                                                // 23
    Some("XTaskQueueUnregisterMonitor"),                                              // 24
    Some("XTaskQueueGetCurrentProcessTaskQueue"),                                     // 25
    Some("XTaskQueueSetCurrentProcessTaskQueue"),                                     // 26
    Some("XThreadSetTimeSensitive"),                                                  // 27
    Some("XThreadVerifyNotTimeSensitive"),                                            // 28
    Some("XThreadAssertNotTimeSensitive"),                                            // 29
    Some("XThreadIsTimeSensitive"),                                                   // 30
    None,                                                                             // 31
];

static SLOTS_XERROR: [Option<&str>; 6] = [None, None, None, Some("XErrorReport"), Some("XErrorSetCallback"), Some("XErrorSetOptions")];
static SLOTS_FEATURE: [Option<&str>; 4] = [None, None, None, Some("XGameRuntimeIsFeatureAvailable")];
static SLOTS_XSYSTEM: [Option<&str>; 6] =
    [None, None, None, Some("XSystemGetConsoleId"), Some("XSystemGetXboxLiveSandboxId"), Some("XSystemGetAppSpecificDeviceId")];
static SLOTS_XGAME: [Option<&str>; 5] = [None, None, None, Some("XGameGetXboxTitleId"), Some("XLaunchNewGame")];
static SLOTS_GAMEEVENT: [Option<&str>; 4] = [None, None, None, Some("XGameEventWrite")];

const OTHER_EDITION: &str = "hypothesis: another title and GDK edition; confirm on the title exe";
const OTHER_TITLES: &str = "hypothesis: read on other titles; confirm on the title exe";
const MEASURED: &str = "measured: asked for by a retail title (run log, 24 Nov 2026)";

// api_id GUIDs, shared by several entries.
const API_ASYNC: Guid = guid(0x073B7DCB, 0x1FCF, 0x4030, [0x94, 0xBE, 0xE3, 0xC9, 0xEB, 0x62, 0x34, 0x28]);
const API_NETWORKING: Guid = guid(0x37E56907, 0x2F10, 0x41E8, [0xB7, 0x2F, 0x36, 0xED, 0xB1, 0x85, 0x33, 0x1A]);
const API_ERROR: Guid = guid(0x8ca467f7, 0x22e8, 0x4096, [0x84, 0x56, 0xbb, 0x8a, 0xa1, 0x3f, 0x79, 0xd8]);
const API_FEATURE: Guid = guid(0x8836fe87, 0xedb9, 0x4fe3, [0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40]);
const API_DISPLAY: Guid = guid(0x03f0fe74, 0xfdd9, 0x4e5c, [0xb6, 0x30, 0xf9, 0x33, 0x9c, 0x47, 0xac, 0xc5]);
const API_SYSTEM: Guid = guid(0xe349bd1a, 0xfc20, 0x4e40, [0xb9, 0x9c, 0x41, 0x78, 0xcc, 0x6b, 0x40, 0x9f]);
const API_ANALYTICS: Guid = guid(0xb884675d, 0xb738, 0x4a9c, [0x81, 0x5d, 0x9a, 0x9a, 0x1e, 0x0c, 0x6c, 0x9b]);
const API_GAME: Guid = guid(0x973a344e, 0x24bf, 0x4d0f, [0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29]);
const API_LAUNCHURI: Guid = guid(0x1b339674, 0x328d, 0x4283, [0xa2, 0x00, 0x31, 0x71, 0xf1, 0x8d, 0x36, 0x39]);
const API_INVITE: Guid = guid(0x0651aae2, 0x4012, 0x4077, [0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c]);
const API_GAMEEVENT: Guid = guid(0xbbfbdcc7, 0xbfe7, 0x409b, [0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d]);
const API_PLS: Guid = guid(0xf4faf4d4, 0x2d04, 0x4fce, [0xb3, 0xe0, 0x47, 0x4a, 0x71, 0x3a, 0x3e, 0x84]);
const API_USER: Guid = guid(0x01acd177, 0x91f9, 0x4763, [0xa3, 0x8e, 0xcc, 0xbb, 0x55, 0xce, 0x32, 0xe0]);
const API_USERDEVICE: Guid = guid(0x7d824997, 0x10dc, 0x45ab, [0x86, 0xb7, 0x27, 0x37, 0x76, 0x7c, 0x0b, 0xf1]);
const API_GAMESAVE: Guid = guid(0x704c3f58, 0xe629, 0x4cc2, [0xb1, 0x97, 0x30, 0x51, 0x1b, 0x99, 0x6f, 0xe2]);
const API_STORE: Guid = guid(0x0dd112ac, 0x7c24, 0x448c, [0xb9, 0x2b, 0x39, 0x60, 0xfb, 0x5b, 0xd3, 0x0c]);
const API_PACKAGE: Guid = guid(0xaf406016, 0xe850, 0x4aa8, [0xa8, 0x8d, 0x2f, 0x3d, 0xcb, 0x9d, 0xac, 0x7e]);
const API_GAMEUI: Guid = guid(0xdfcd4649, 0x4ff8, 0x4043, [0xba, 0x07, 0x35, 0xd6, 0x07, 0xdf, 0x98, 0xb0]);
const API_STREAMING: Guid = guid(0x0a2192aa, 0xb2d5, 0x4d58, [0x83, 0xbe, 0x38, 0x3b, 0x6d, 0x80, 0x79, 0x9e]);
const API_ACCESSIBILITY: Guid = guid(0x3e241977, 0x6237, 0x41e9, [0x85, 0x59, 0x10, 0x2c, 0x2d, 0x99, 0x83, 0xf1]);

const fn entry(api: Guid, iid: Guid, kind: Match, name: &'static str, iid_note: &'static str, source: &'static str, slots: &'static [Option<&'static str>]) -> KnownTable {
    KnownTable { api, iid, kind, name, iid_note, source, slots }
}

pub static KNOWN_TABLES: &[KnownTable] = &[
    entry(API_ASYNC, API_ASYNC, Match::IidExact, "XAsync / XTaskQueue / XThread (32 slots)", "riid = api_id", OTHER_EDITION, &SLOTS_073B7DCB),
    entry(
        API_NETWORKING,
        API_NETWORKING,
        Match::IidExact,
        "XNetworking",
        "riid = api_id (our former \"second table\": slots 3-7, 15-17 known by signature only)",
        OTHER_TITLES,
        &[],
    ),
    entry(
        API_NETWORKING,
        guid(0xbf2346b2, 0x39af, 0x4658, [0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3]),
        Match::IidExact,
        "XNetworking",
        "riid {bf2346b2-...}",
        OTHER_TITLES,
        &[],
    ),
    entry(API_ERROR, API_ERROR, Match::IidExact, "XError", "riid = api_id", OTHER_TITLES, &SLOTS_XERROR),
    entry(API_FEATURE, API_FEATURE, Match::IidExact, "XGameRuntimeFeature", "riid = api_id", OTHER_TITLES, &SLOTS_FEATURE),
    entry(
        API_DISPLAY,
        guid(0x35f07670, 0x706e, 0x4bfb, [0x94, 0x76, 0x09, 0x07, 0x98, 0xc5, 0xeb, 0xf3]),
        Match::IidExact,
        "XDisplay",
        "riid {35f07670-...} (slots 3, 4)",
        OTHER_TITLES,
        &[],
    ),
    entry(API_SYSTEM, API_SYSTEM, Match::IidExact, "XSystem", "riid = api_id", OTHER_TITLES, &SLOTS_XSYSTEM),
    entry(API_SYSTEM, guid(0x1861cf2e, 0xe18b, 0x4834, [0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf]), Match::IidExact, "XSystem", "riid {1861cf2e-...}", MEASURED, &SLOTS_XSYSTEM),
    entry(API_SYSTEM, data1(0x67ce4bfc), Match::IidData1, "XSystem", "riid 67ce4bfc-...-7a85 (partial GUID)", OTHER_TITLES, &SLOTS_XSYSTEM),
    entry(API_SYSTEM, data1(0x6fd71f09), Match::IidData1, "XSystem", "riid 6fd71f09-...-f852 (partial GUID)", OTHER_TITLES, &SLOTS_XSYSTEM),
    entry(API_ANALYTICS, API_ANALYTICS, Match::IidExact, "XSystemAnalytics", "riid = api_id (slot 3)", OTHER_TITLES, &[]),
    entry(API_GAME, data1(0x50849859), Match::IidData1, "XGame", "riid 50849859-...-f852 (partial GUID)", OTHER_TITLES, &SLOTS_XGAME),
    entry(API_GAME, guid(0x2549f142, 0x6419, 0x4a06, [0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34]), Match::IidExact, "XGame", "riid {2549f142-...}", MEASURED, &SLOTS_XGAME),
    entry(API_LAUNCHURI, API_LAUNCHURI, Match::IidExact, "XLaunchUri", "riid = api_id (slots 3-5)", OTHER_TITLES, &[]),
    entry(API_INVITE, API_INVITE, Match::IidExact, "XGameInvite", "riid = api_id (slots 3, 4)", OTHER_TITLES, &[]),
    entry(API_GAMEEVENT, API_GAMEEVENT, Match::IidExact, "XGameEvent", "riid = api_id", OTHER_TITLES, &SLOTS_GAMEEVENT),
    entry(
        API_PLS,
        guid(0xd29411df, 0x0794, 0x4553, [0x8b, 0x27, 0x95, 0xfc, 0x02, 0xd0, 0xf7, 0x5d]),
        Match::IidExact,
        "XPersistentLocalStorage",
        "riid {d29411df-...} (slots 3-7)",
        OTHER_TITLES,
        &[],
    ),
    entry(API_USER, API_USER, Match::IidExact, "XUser", "riid = api_id (up to 38 slots depending on the version)", OTHER_TITLES, &[]),
    entry(API_USER, data1(0xeb9bf948), Match::IidData1, "XUser", "riid eb9bf948-...-c4c0 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_USER, data1(0x1bf2f8c5), Match::IidData1, "XUser", "riid 1bf2f8c5-...-1161 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_USER, data1(0xcef4fac0), Match::IidData1, "XUser", "riid cef4fac0-...-5b74, gamertag (partial GUID)", OTHER_TITLES, &[]),
    entry(API_USERDEVICE, API_USERDEVICE, Match::IidExact, "XUserDevice", "riid = api_id (slots 3-8)", OTHER_TITLES, &[]),
    entry(API_USERDEVICE, data1(0x0cc6a956), Match::IidData1, "XUserDevice", "riid 0cc6a956-...-ebc8, slots 3-10 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_GAMESAVE, API_GAMESAVE, Match::IidExact, "XGameSave", "riid = api_id", OTHER_TITLES, &[]),
    entry(API_GAMESAVE, data1(0x704c3f58), Match::IidData1, "XGameSave", "riid 704c3f58-...-6ee2 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_GAMESAVE, guid(0x1bfff3af, 0xf14a, 0x40a3, [0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9]), Match::IidExact, "XGameSave", "riid {1bfff3af-...}", MEASURED, &[]),
    entry(API_STORE, data1(0xde3dbdd4), Match::IidData1, "XStore", "riid de3dbdd4-...-d06a (partial GUID)", OTHER_TITLES, &[]),
    entry(API_STORE, data1(0x2d42fea5), Match::IidData1, "XStore", "riid 2d42fea5-...-ae5d (partial GUID)", OTHER_TITLES, &[]),
    entry(API_STORE, data1(0x60b09f4e), Match::IidData1, "XStore", "riid 60b09f4e-...-30e1 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_PACKAGE, guid(0x0b9d556e, 0xab70, 0x43f4, [0xba, 0xa0, 0x99, 0x99, 0xac, 0x4c, 0xb0, 0x40]), Match::IidExact, "XPackage", "riid {0b9d556e-...}", MEASURED, &[]),
    entry(API_PACKAGE, data1(0xbe3f3e9f), Match::IidData1, "XPackage", "riid be3f3e9f-...-59a5 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_PACKAGE, data1(0xf92d8712), Match::IidData1, "XPackage", "riid f92d8712-...-eb42 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_PACKAGE, data1(0x3720de07), Match::IidData1, "XPackage", "riid 3720de07-...-be55 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_GAMEUI, data1(0x6eeaa73e), Match::IidData1, "XGameUi", "riid 6eeaa73e-...-50a1 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_GAMEUI, data1(0x72c4e70f), Match::IidData1, "XGameUi", "riid 72c4e70f-...-4fe5 (partial GUID)", OTHER_TITLES, &[]),
    entry(API_GAMEUI, data1(0xfc0b9bd0), Match::IidData1, "XGameUi", "riid fc0b9bd0-...-4cb6 (partial GUID)", OTHER_TITLES, &[]),
    entry(
        API_STREAMING,
        guid(0x5f5e5169, 0x746c, 0x4001, [0xad, 0x1c, 0xda, 0x72, 0x8d, 0x01, 0xc9, 0xeb]),
        Match::IidExact,
        "streaming",
        "riid {5f5e5169-...} (Minecraft: slots 3, 6, 7, 8, 12, 13, 17)",
        OTHER_TITLES,
        &[],
    ),
    entry(
        API_ACCESSIBILITY,
        guid(0xd722b373, 0x8c4d, 0x4692, [0x9e, 0x51, 0xd6, 0xad, 0x9b, 0x37, 0xaa, 0x7d]),
        Match::IidExact,
        "accessibility",
        "riid {d722b373-...}",
        OTHER_TITLES,
        &[],
    ),
    entry(data1(0x186d5592), data1(0), Match::ApiData1, "broadcast / capture", "api 186d5592-...-647a (partial GUID), riid not known", OTHER_TITLES, &[]),
    entry(
        data1(0xa4f1aee2),
        data1(0x3a949778),
        Match::ApiData1,
        "broadcast / capture",
        "api a4f1aee2-...-ac27, riid 3a949778-...-6baa (partial GUIDs)",
        OTHER_TITLES,
        &[],
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    const UNLISTED: Guid = guid(0, 0, 0, [0xC0, 0, 0, 0, 0, 0, 0, 0x46]);

    #[test]
    fn the_table_has_the_c_entries() {
        assert_eq!(KNOWN_TABLES.len(), 40);
        assert_eq!(SLOTS_073B7DCB.len(), 32);
    }

    #[test]
    fn a_listed_couple_is_level_listed() {
        let (t, level) = known(&API_ASYNC, Some(&API_ASYNC)).unwrap();
        assert_eq!((t.name, level), ("XAsync / XTaskQueue / XThread (32 slots)", Level::Listed));
        let second = guid(0xbf2346b2, 0x39af, 0x4658, [0xb5, 0xea, 0x44, 0x71, 0x3c, 0x7e, 0x83, 0xb3]);
        let (t, level) = known(&API_NETWORKING, Some(&second)).unwrap();
        assert_eq!((t.iid_note, level), ("riid {bf2346b2-...}", Level::Listed));
    }

    #[test]
    fn the_couple_selects_the_table() {
        // The same api_id with its own riid is the first XNetworking entry, not the second.
        let (t, _) = known(&API_NETWORKING, Some(&API_NETWORKING)).unwrap();
        assert!(t.iid_note.starts_with("riid = api_id"));
    }

    #[test]
    fn a_riid_known_by_its_first_field_is_partial() {
        let riid = guid(0x67ce4bfc, 0x1111, 0x2222, [0, 0, 0, 0, 0, 0, 0x7a, 0x85]);
        let (t, level) = known(&API_SYSTEM, Some(&riid)).unwrap();
        assert_eq!((t.iid_note, level), ("riid 67ce4bfc-...-7a85 (partial GUID)", Level::Partial));
    }

    #[test]
    fn an_unlisted_riid_gives_the_family_and_an_unreadable_one_too() {
        assert_eq!(known(&API_ASYNC, Some(&UNLISTED)).map(|(t, l)| (t.name, l)), Some(("XAsync / XTaskQueue / XThread (32 slots)", Level::Family)));
        assert_eq!(known(&API_ASYNC, None).map(|(_, l)| l), Some(Level::Family));
    }

    #[test]
    fn the_interfaces_the_title_asks_for_are_listed() {
        for (api, riid, name) in [
            (API_PACKAGE, guid(0x0b9d556e, 0xab70, 0x43f4, [0xba, 0xa0, 0x99, 0x99, 0xac, 0x4c, 0xb0, 0x40]), "XPackage"),
            (API_SYSTEM, guid(0x1861cf2e, 0xe18b, 0x4834, [0xa9, 0xf5, 0xb4, 0xa4, 0xe6, 0xef, 0xb4, 0xcf]), "XSystem"),
            (API_GAME, guid(0x2549f142, 0x6419, 0x4a06, [0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34]), "XGame"),
            (API_GAMESAVE, guid(0x1bfff3af, 0xf14a, 0x40a3, [0x8e, 0x35, 0x9a, 0xda, 0x90, 0x65, 0x93, 0xf9]), "XGameSave"),
        ] {
            assert_eq!(known(&api, Some(&riid)).map(|(t, l)| (t.name, l)), Some((name, Level::Listed)));
        }
    }

    #[test]
    fn an_unknown_api_is_not_known() {
        assert!(known(&UNLISTED, Some(&UNLISTED)).is_none());
    }

    #[test]
    fn an_api_known_by_its_first_field_matches_on_it() {
        let api = guid(0x186d5592, 1, 2, [3; 8]);
        assert_eq!(known(&api, Some(&UNLISTED)).map(|(t, l)| (t.name, l)), Some(("broadcast / capture", Level::Family)));
        let api = guid(0xa4f1aee2, 1, 2, [3; 8]);
        let riid = guid(0x3a949778, 5, 6, [7; 8]);
        assert_eq!(known(&api, Some(&riid)).map(|(_, l)| l), Some(Level::Partial));
    }
}
