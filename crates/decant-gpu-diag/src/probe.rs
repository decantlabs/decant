//! What this PC is: adapters from DXGI with their D3D12 ray tracing tier and driver version, the
//! Windows version, the CPU model and the installed memory. Only public model names and version
//! numbers are read; LUIDs, subsystem ids and anything else unique to a machine are not.

#[cfg(not(windows))]
use crate::record::{Adapter, System};

#[cfg(windows)]
pub use self::windows::{adapters, system};

#[cfg(not(windows))]
pub fn adapters(_selected: usize) -> Vec<Adapter> {
    Vec::new()
}

#[cfg(not(windows))]
pub fn system() -> System {
    System { windows: "not Windows".to_owned(), cpu: "unknown".to_owned(), ram_gb: 0 }
}

/// `a.b.c.d` from the packed driver version DXGI reports.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn driver_version(packed: i64) -> String {
    let p = packed as u64;
    format!("{}.{}.{}.{}", p >> 48, (p >> 32) & 0xFFFF, (p >> 16) & 0xFFFF, p & 0xFFFF)
}

/// The name Windows gives itself. The registry still says "Windows 10" on Windows 11, whose
/// builds start at 22000.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn windows_name(product: &str, display: &str, build: u32, revision: u32) -> String {
    let product = if build >= 22000 { product.replacen("Windows 10", "Windows 11", 1) } else { product.to_owned() };
    let display = if display.is_empty() { String::new() } else { format!(" {display}") };
    format!("{product}{display} build {build}.{revision}")
}

/// Installed memory in whole gigabytes, as a spec sheet gives it.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn round_gb(bytes: u64) -> u64 {
    (bytes + (1 << 29)) >> 30
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod windows {
    use windows::Win32::Foundation::ERROR_SUCCESS;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::{D3D12_FEATURE_D3D12_OPTIONS5, D3D12_FEATURE_DATA_D3D12_OPTIONS5, D3D12CreateDevice, ID3D12Device};
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIAdapter1, IDXGIDevice, IDXGIFactory1};
    use windows::Win32::System::Registry::{HKEY_LOCAL_MACHINE, RRF_RT_REG_DWORD, RRF_RT_REG_SZ, RegGetValueW};
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows::core::{HSTRING, Interface};

    use super::{driver_version, round_gb, windows_name};
    use crate::record::{Adapter, System};

    /// Every adapter DXGI lists, the first being the default one.
    pub fn adapters(selected: usize) -> Vec<Adapter> {
        // SAFETY: a plain factory creation; the interface is released when dropped.
        let Ok(factory) = (unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }) else { return Vec::new() };
        let mut out = Vec::new();
        // SAFETY: EnumAdapters1 fails past the last adapter, which ends the loop.
        while let Ok(adapter) = unsafe { factory.EnumAdapters1(out.len() as u32) } {
            let index = out.len();
            if let Some(a) = describe(&adapter, index == 0, index == selected) {
                out.push(a);
            } else {
                break;
            }
        }
        out
    }

    fn describe(adapter: &IDXGIAdapter1, default: bool, selected: bool) -> Option<Adapter> {
        // SAFETY: GetDesc1 fills a structure it owns and returns it by value.
        let desc = unsafe { adapter.GetDesc1() }.ok()?;
        let end = desc.Description.iter().position(|&c| c == 0).unwrap_or(desc.Description.len());
        // SAFETY: the interface id is a constant; the call only reads it.
        let driver = unsafe { adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }.ok().map(driver_version);
        Some(Adapter {
            name: String::from_utf16_lossy(&desc.Description[..end]).trim().to_owned(),
            vendor_id: desc.VendorId,
            device_id: desc.DeviceId,
            driver,
            raytracing_tier: raytracing_tier(adapter),
            software: desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
            default,
            selected,
        })
    }

    fn raytracing_tier(adapter: &IDXGIAdapter1) -> Option<u32> {
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: `device` receives the created interface and releases it when dropped.
        unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }.ok()?;
        let device = device?;
        let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: `options` is the structure this feature expects, with its size.
        unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, (&raw mut options).cast(), size_of_val(&options) as u32) }.ok()?;
        Some(options.RaytracingTier.0 as u32)
    }

    fn registry_text(key: &str, value: &str) -> Option<String> {
        let (key, value) = (HSTRING::from(key), HSTRING::from(value));
        let mut buffer = vec![0u16; 512];
        let mut size = (buffer.len() * 2) as u32;
        // SAFETY: `buffer` holds `size` bytes and both names are zero terminated HSTRINGs.
        let status = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, &key, &value, RRF_RT_REG_SZ, None, Some(buffer.as_mut_ptr().cast()), Some(&mut size)) };
        if status != ERROR_SUCCESS {
            return None;
        }
        let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        Some(String::from_utf16_lossy(&buffer[..end]).trim().to_owned())
    }

    fn registry_number(key: &str, value: &str) -> Option<u32> {
        let (key, value) = (HSTRING::from(key), HSTRING::from(value));
        let mut number = 0u32;
        let mut size = 4u32;
        // SAFETY: `number` holds the four bytes a DWORD needs.
        let status = unsafe { RegGetValueW(HKEY_LOCAL_MACHINE, &key, &value, RRF_RT_REG_DWORD, None, Some((&raw mut number).cast()), Some(&mut size)) };
        (status == ERROR_SUCCESS).then_some(number)
    }

    pub fn system() -> System {
        const VERSION: &str = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion";
        let build = registry_text(VERSION, "CurrentBuildNumber").and_then(|b| b.parse().ok()).unwrap_or(0);
        let windows = windows_name(
            &registry_text(VERSION, "ProductName").unwrap_or_else(|| "Windows".to_owned()),
            &registry_text(VERSION, "DisplayVersion").unwrap_or_default(),
            build,
            registry_number(VERSION, "UBR").unwrap_or(0),
        );
        let cpu = registry_text(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0", "ProcessorNameString").unwrap_or_else(|| "unknown".to_owned());
        let mut memory = MEMORYSTATUSEX { dwLength: size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
        // SAFETY: `memory` has its length set as the call requires.
        let ram_gb = if unsafe { GlobalMemoryStatusEx(&mut memory) }.is_ok() { round_gb(memory.ullTotalPhys) } else { 0 };
        System { windows, cpu, ram_gb }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every Windows has the software adapter (WARP), so the list is never empty, and exactly
        /// one adapter is the default.
        #[test]
        fn dxgi_lists_a_default_and_a_software_adapter() {
            let list = adapters(0);
            assert!(list.iter().any(|a| a.software), "{list:?}");
            assert_eq!(list.iter().filter(|a| a.default).count(), 1);
            assert!(list[0].default && list[0].selected);
            assert!(list.iter().all(|a| !a.name.is_empty()));
        }

        #[test]
        fn the_system_reads_as_windows_with_memory() {
            let s = system();
            assert!(s.windows.starts_with("Windows"), "{}", s.windows);
            assert!(s.ram_gb > 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn versions_and_sizes_are_worded_as_spec_sheets_do() {
        assert_eq!(driver_version((32 << 48) | (16 << 16) | 1742), "32.0.16.1742");
        assert_eq!(windows_name("Windows 10 Pro", "24H2", 26300, 1000), "Windows 11 Pro 24H2 build 26300.1000");
        assert_eq!(windows_name("Windows 10 Pro", "", 19045, 1), "Windows 10 Pro build 19045.1");
        assert_eq!(round_gb(34_157_002_752), 32);
        assert_eq!(round_gb(0), 0);
    }
}
