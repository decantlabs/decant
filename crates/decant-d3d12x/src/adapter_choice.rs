//! Which PC adapter the device of the title is made on. The title passes no adapter (or a console
//! object we ignore), and `D3D12CreateDevice(NULL)` then takes the Windows default, which on a hybrid
//! laptop is the integrated GPU (its driver faulted at 3 s in the first run on an RTX 4060 laptop,
//! issue #195). [`pick`] asks DXGI for the high performance adapter instead.
//!
//! `DECANT_D3D12X_ADAPTER=<text>` overrides it: a number is an index into the list the DXGI factory
//! gives (high performance first), anything else picks the first adapter whose description contains
//! the text, case ignored. A value that matches nothing logs the list and falls back to the default
//! choice. When DXGI cannot list adapters (old Windows, Wine) the caller keeps passing NULL.

/// The facts about one adapter that the choice looks at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdapterInfo {
    pub description: String,
    pub vendor_id: u32,
    pub dedicated_video_memory: u64,
    pub software: bool,
}

/// What an override value asks for.
enum Request<'a> {
    Index(usize),
    Name(&'a str),
}

/// Reads the override setting. Blank or missing means no request.
fn request(setting: Option<&str>) -> Option<Request<'_>> {
    let text = setting?.trim();
    if text.is_empty() {
        None
    } else if let Ok(index) = text.parse::<usize>() {
        Some(Request::Index(index))
    } else {
        Some(Request::Name(text))
    }
}

/// Where in `list` a request points, if anywhere.
fn resolve(list: &[AdapterInfo], req: &Request<'_>) -> Option<usize> {
    match req {
        Request::Index(i) => (*i < list.len()).then_some(*i),
        Request::Name(name) => {
            let wanted = name.to_lowercase();
            list.iter().position(|a| a.description.to_lowercase().contains(&wanted))
        }
    }
}

/// The index in `list` (factory order) of the adapter to use. An override that resolves wins,
/// otherwise it is the first hardware adapter. `None` means pass NULL, the Windows default.
pub fn choose(list: &[AdapterInfo], over_ride: Option<&str>) -> Option<usize> {
    let requested = request(over_ride).and_then(|r| resolve(list, &r));
    requested.or_else(|| list.iter().position(|a| !a.software))
}

/// Whether an override is set that points at no adapter of `list`.
pub fn override_misses(list: &[AdapterInfo], over_ride: Option<&str>) -> bool {
    request(over_ride).is_some_and(|r| resolve(list, &r).is_none())
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod com {
    use super::{AdapterInfo, choose, override_misses};
    use crate::log::xs_log;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IDXGIAdapter1, IDXGIFactory6,
    };
    use windows::core::Interface;

    /// The adapter to hand to the PC `D3D12CreateDevice`: a reference the caller releases with
    /// [`release`], or null for the Windows default. The choice is logged once per process.
    pub fn pick() -> *mut core::ffi::c_void {
        static LOGGED: std::sync::Once = std::sync::Once::new();
        // SAFETY: plain DXGI calls with valid arguments.
        let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory6>() }) else {
            LOGGED.call_once(|| xs_log!("adapter: no IDXGIFactory6, the PC default adapter is used"));
            return core::ptr::null_mut();
        };
        let mut adapters: Vec<IDXGIAdapter1> = Vec::new();
        let mut list: Vec<AdapterInfo> = Vec::new();
        let mut luids = Vec::new();
        // SAFETY: plain DXGI calls on live interfaces.
        while let Ok(adapter) = unsafe { factory.EnumAdapterByGpuPreference::<IDXGIAdapter1>(adapters.len() as u32, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) } {
            let Ok(desc) = (unsafe { adapter.GetDesc1() }) else { break };
            let len = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
            list.push(AdapterInfo {
                description: String::from_utf16_lossy(&desc.Description[..len]),
                vendor_id: desc.VendorId,
                dedicated_video_memory: desc.DedicatedVideoMemory as u64,
                software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
            });
            luids.push(desc.AdapterLuid);
            adapters.push(adapter);
        }
        let over_ride = std::env::var("DECANT_D3D12X_ADAPTER").ok();
        let chosen = choose(&list, over_ride.as_deref());
        LOGGED.call_once(|| {
            if override_misses(&list, over_ride.as_deref()) {
                xs_log!("adapter: DECANT_D3D12X_ADAPTER={:?} matches no adapter, falling back to the default choice", over_ride.as_deref().unwrap_or(""));
                for (i, a) in list.iter().enumerate() {
                    xs_log!("adapter: [{i}] {} (vendor 0x{:04X}, {} MiB){}", a.description, a.vendor_id, a.dedicated_video_memory >> 20, if a.software { ", software" } else { "" });
                }
            }
            match chosen {
                Some(i) => {
                    let a = &list[i];
                    xs_log!(
                        "adapter: chose [{i}] {} (vendor 0x{:04X}, {} MiB dedicated, LUID {:08X}:{:08X}) of {}",
                        a.description,
                        a.vendor_id,
                        a.dedicated_video_memory >> 20,
                        luids[i].HighPart as u32,
                        luids[i].LowPart,
                        list.len()
                    );
                }
                None => xs_log!("adapter: no usable hardware adapter listed ({}), the PC default adapter is used", list.len()),
            }
        });
        match chosen {
            Some(i) => adapters.swap_remove(i).into_raw(),
            None => core::ptr::null_mut(),
        }
    }

    /// Releases what [`pick`] returned.
    ///
    /// # Safety
    /// `adapter` is null or the pointer [`pick`] returned, not released before.
    pub unsafe fn release(adapter: *mut core::ffi::c_void) {
        if !adapter.is_null() {
            // SAFETY: by the contract above, `adapter` owns one reference.
            drop(unsafe { IDXGIAdapter1::from_raw(adapter) });
        }
    }
}

#[cfg(windows)]
pub use com::{pick, release};

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(name: &str, software: bool) -> AdapterInfo {
        AdapterInfo { description: name.to_string(), vendor_id: 0, dedicated_video_memory: 0, software }
    }

    fn laptop() -> Vec<AdapterInfo> {
        vec![adapter("Discrete Example GPU 9000", false), adapter("Integrated Example Graphics", false), adapter("Software Render Driver", true)]
    }

    #[test]
    fn without_an_override_the_first_hardware_adapter_wins() {
        assert_eq!(choose(&laptop(), None), Some(0));
        let software_first = vec![adapter("Software Render Driver", true), adapter("Integrated Example Graphics", false)];
        assert_eq!(choose(&software_first, None), Some(1));
    }

    #[test]
    fn no_hardware_adapter_means_null() {
        assert_eq!(choose(&[], None), None);
        assert_eq!(choose(&[adapter("Software Render Driver", true)], Some("nothing")), None);
    }

    #[test]
    fn a_text_override_matches_a_substring_ignoring_case() {
        assert_eq!(choose(&laptop(), Some("integrated")), Some(1));
        assert_eq!(choose(&laptop(), Some("  EXAMPLE  ")), Some(0));
        assert_eq!(choose(&laptop(), Some("render")), Some(2));
    }

    #[test]
    fn a_number_override_is_an_index_even_for_software_adapters() {
        assert_eq!(choose(&laptop(), Some("1")), Some(1));
        assert_eq!(choose(&laptop(), Some("2")), Some(2));
    }

    #[test]
    fn an_override_that_names_nothing_falls_back_and_is_reported() {
        for text in ["radeon", "3", "99"] {
            assert_eq!(choose(&laptop(), Some(text)), Some(0), "{text}");
            assert!(override_misses(&laptop(), Some(text)), "{text}");
        }
    }

    #[test]
    fn blank_or_matching_overrides_are_not_misses() {
        for text in [None, Some(""), Some("   "), Some("integrated"), Some("0")] {
            assert!(!override_misses(&laptop(), text), "{text:?}");
        }
    }
}
