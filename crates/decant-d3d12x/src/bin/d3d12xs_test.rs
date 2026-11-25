//! `d3d12xs_test <d3d12_xs.dll>`: checks the root signature wrappers and the device entry point of
//! `d3d12_xs.dll`, the port of the former `tests/d3d12xs_test.c`.
//!
//! 1. The DXBC checksum code (`rts0`) reproduces the checksum the PC runtime writes.
//! 2. Serializing a root signature whose flags carry the console bits 0x03000000 succeeds through
//!    d3d12_xs and yields a blob with only the public bits.
//! 3. A serialized container patched to carry those bits (checksum left stale, as a raw console
//!    container would be for the PC) deserializes through d3d12_xs, both entry points, and reports
//!    the public flags and the right parameter count.
//! 4. `D3D12XboxCreateDevice` returns the console ABI proxy, driven only through console slots:
//!    relayed CreateRootSignature (16), unknown slots answer `E_NOTIMPL`, a proxied queue (8) and
//!    list (12) with the list cursor held at its limit, the list run on the GPU through the queue
//!    (10, 14 and a fence), PresentX (30) accepted without a hook. Then the console only slots:
//!    queue (48) and list (73) creation from console parameters, Signal with flags (queue 23), the
//!    descriptor heap (14) whose start handles come back in rax (9, 10) and that SetDescriptorHeaps
//!    (list 32) unwraps, GetResourceAllocationInfo and GetCustomHeapProperties with the result
//!    pointer first (25, 26), QueryInterface(IDXGIDevice), D3D12CreateDevice, and the emulated
//!    frame events (78, 79, 80: real waits at the requested cadence). Skipped (not failed) when the
//!    machine has no D3D12 12_0 device, as on hosted CI runners.
//! 5. Exports a console title imports: `IID_ID3D12Device` as data, `D3D12CreateDevice`, and the
//!    four `DXGIX*` entry points answering `DXGI_ERROR_UNSUPPORTED`.
//!
//! 6. Title memory (with `DECANT_D3D12X_TITLE_MEM=1`): pages that `xmem.dll` maps (found next to
//!    `d3d12_xs.dll`) are written by the CPU and read by a compute shader through the address
//!    `title_mem` gives, after `title_mem::sync`. An alias, a remap, an unmap, a write after a sync, a
//!    span over scattered frames and the detile pass are checked. The commit charge of the process is printed before
//!    and after, and `DECANT_TEST_HOLD=1` waits at the end (`tests/title_mem_commit.ps1`). Skipped
//!    without `xmem.dll` or a GPU.
//!
//! 7. The console slots the title calls while it boots, on a device created as `ID3D12Device11`
//!    (`src/bin/run/pipeline_slots.rs`, the file of module `run::pipeline_slots`), with the
//!    `DECANT_CALLS` table read back.
//! 8. With `DECANT_D3D12X_TITLE_MEM=1`: a root SRV `t0` of space 3 at a title address in an xmem arena is
//!    read by a shader through the proxy, and one at an address no PC resource holds reads the zero
//!    buffer and is counted (`src/bin/run/root_views.rs`).
//!
//! 8. `d3d12xs_test <d3d12_xs.dll> direct` runs the device case of the direct shader path instead
//!    (`direct_device_case`): a pair of synthetic console shaders is translated, compiled and drawn
//!    through the console proxies, and the colour is read back. It needs `DECANT_D3D12X_DIRECT=1`,
//!    which changes the root signatures the layer makes, so it is a run of its own.
//!
//! The plain PC runtime's own answers to the console flags are printed for information. Exit code:
//! 0 when every check passed, 1 on a failure, 2 when the arguments are wrong.

/// The container code of the DLL, compiled into this program too: linking the library itself would
/// clash with the `d3d12.dll` imports of the `windows` crate, as it exports the same names.
#[cfg(windows)]
#[path = "../rts0.rs"]
#[allow(dead_code)]
mod rts0;

/// The log, title memory and the GPU copy of the DLL, compiled into this program too for the same
/// reason: the title memory case writes through an `xmem.dll` view and reads it on the GPU.
#[cfg(windows)]
#[path = "../log.rs"]
#[allow(dead_code)]
mod log;
#[cfg(windows)]
#[path = "../mem_owners.rs"]
#[allow(dead_code)]
mod mem_owners;
#[cfg(windows)]
#[path = "../title_mem.rs"]
mod title_mem;
#[cfg(windows)]
#[path = "../title_mirror.rs"]
mod title_mirror;
#[cfg(windows)]
#[path = "../gpu_copy.rs"]
mod gpu_copy;
#[cfg(windows)]
#[path = "../detile_plan.rs"]
#[allow(dead_code)]
mod detile_plan;
#[cfg(windows)]
#[path = "../detile.rs"]
mod detile;
/// The removal check of the DLL's `diagnose` is not needed here: the checks run on their own device.
#[cfg(windows)]
mod diagnose {
    pub fn removed_after(_what: impl FnOnce() -> String) {}
}

/// Synthetic console shaders and the device case of the direct shader path.
#[cfg(windows)]
#[path = "../direct_fixture.rs"]
mod direct_fixture;
#[cfg(windows)]
#[path = "../direct_device_case.rs"]
#[allow(unsafe_code)]
mod direct_device_case;

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    let args = decant_testkit::args();
    let code = match args.as_slice() {
        [dll] => run::main(dll),
        [dll, case] if case == "direct" => direct_device_case::main(dll),
        _ => {
            println!("usage: d3d12xs_test <d3d12_xs.dll> [direct]");
            2
        }
    };
    std::process::ExitCode::from(code)
}

#[cfg(not(windows))]
fn main() {
    eprintln!("d3d12xs_test runs on Windows only");
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod run {
    use std::ffi::{OsStr, c_void};
    use std::time::{Duration, Instant};

    use decant_testkit::Library;
    use windows::Win32::Foundation::LUID;
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, ID3DBlob};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_UNSUPPORTED, IDXGIDevice};
    use windows::core::{GUID, Interface};

    use crate::rts0::{dxbc_checksum, flags_offset, rd32, wr32};

    type SerializeVersioned = unsafe extern "system" fn(*const D3D12_VERSIONED_ROOT_SIGNATURE_DESC, *mut *mut c_void, *mut *mut c_void) -> i32;
    type Deserializer = unsafe extern "system" fn(*const c_void, usize, *const GUID, *mut *mut c_void) -> i32;
    type XboxCreate = unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32;
    type Create = unsafe extern "system" fn(*mut c_void, u32, *const GUID, *mut *mut c_void) -> i32;
    type Release = unsafe extern "system" fn(*mut c_void) -> u32;
    type Query = unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> i32;
    type NoArgs = unsafe extern "system" fn(*mut c_void) -> i32;

    const E_NOTIMPL: u32 = 0x8000_4001;
    const E_INVALIDARG: u32 = 0x8007_0057;
    const INFINITE: u32 = u32::MAX;

    struct Test {
        failures: u32,
    }

    /// One check line, `ok    ` or `FAIL  ` and the text, as the C `CHECK` macro printed it.
    macro_rules! check {
        ($t:expr, $cond:expr, $($arg:tt)*) => {{
            let ok = $cond;
            let text = format!($($arg)*);
            $t.line(ok, &text);
        }};
    }

    mod pipeline_slots;
    mod root_views;

    impl Test {
        fn line(&mut self, ok: bool, text: &str) {
            if ok {
                println!("ok    {text}");
            } else {
                println!("FAIL  {text}");
                self.failures += 1;
            }
        }
    }

    /// The function in slot `slot` of the vtable of `object`.
    ///
    /// # Safety
    /// `object` is a live COM object, and `F` is the function pointer type of that slot.
    unsafe fn vt<F: Copy>(object: *mut c_void, slot: usize) -> F {
        const { assert!(size_of::<F>() == size_of::<usize>()) };
        // SAFETY: by the contract above.
        unsafe {
            let table = *(object as *const *const usize);
            std::mem::transmute_copy::<usize, F>(&*table.add(slot))
        }
    }

    /// `IUnknown::Release` through slot 2.
    fn release(object: *mut c_void) -> u32 {
        // SAFETY: callers pass a live object they own a reference to.
        unsafe { vt::<Release>(object, 2)(object) }
    }

    fn hr(value: i32) -> u32 {
        value as u32
    }

    fn wait_fence(fence: &ID3D12Fence, value: u64) {
        for _ in 0..500 {
            // SAFETY: a live fence.
            if unsafe { fence.GetCompletedValue() } >= value {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// A pointer to an interface, as the raw pointer the console slots take.
    fn raw<T: Interface>(object: &T) -> *mut c_void {
        object.as_raw()
    }

    /// Console only device, list and queue slots (order measured on the console DLL).
    fn console_slots(t: &mut Test, dev: *mut c_void, alloc: *mut c_void, fence: Option<&ID3D12Fence>) {
        // SAFETY: `dev` is the console device proxy and every slot is called with the argument
        // shapes of the console order (docs/D3D12X_LAYER.md). The objects are live until released
        // at the end.
        unsafe {
            // Queue 48: console description, its first 16 bytes the PC one.
            let qdesc = [0u8; 0x18];
            let mut q: *mut c_void = std::ptr::null_mut();
            let r = vt::<unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32>(dev, 48)(
                dev,
                qdesc.as_ptr().cast(),
                &ID3D12CommandQueue::IID,
                &mut q,
            );
            check!(t, r >= 0 && !q.is_null(), "device slot 48 creates a proxied queue from a 0x18-byte console description (hr=0x{:08x})", hr(r));

            // Queue 48 with console flags 2 and 3 (compute queues of a title): the unknown bit is dropped.
            for flags in [2u32, 3] {
                let mut qdesc = [0u32; 6];
                qdesc[0] = D3D12_COMMAND_LIST_TYPE_COMPUTE.0 as u32;
                qdesc[2] = flags;
                let mut q: *mut c_void = std::ptr::null_mut();
                let r = vt::<unsafe extern "system" fn(*mut c_void, *const c_void, *const GUID, *mut *mut c_void) -> i32>(dev, 48)(
                    dev,
                    qdesc.as_ptr().cast(),
                    &ID3D12CommandQueue::IID,
                    &mut q,
                );
                check!(t, r >= 0 && !q.is_null(), "device slot 48 creates a compute queue with console flags 0x{flags:X} (hr=0x{:08x})", hr(r));
            }

            // List 73: parameters {type at 0, node mask at 8}.
            let params = [D3D12_COMMAND_LIST_TYPE_DIRECT.0 as u32, 0, 0, 0];
            let mut list: *mut c_void = std::ptr::null_mut();
            ID3D12CommandAllocator::from_raw_borrowed(&alloc).expect("allocator").Reset().ok();
            let r = vt::<unsafe extern "system" fn(*mut c_void, *const c_void, *mut c_void, *mut c_void, *const GUID, *mut *mut c_void) -> i32>(dev, 73)(
                dev,
                params.as_ptr().cast(),
                alloc,
                std::ptr::null_mut(),
                &ID3D12GraphicsCommandList::IID,
                &mut list,
            );
            check!(t, r >= 0 && !list.is_null(), "device slot 73 creates a proxied list from console parameters (hr=0x{:08x})", hr(r));

            // Heap 14, start handles in rax (9, 10).
            let hd = D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 4,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            };
            let mut heap: *mut c_void = std::ptr::null_mut();
            let r = vt::<unsafe extern "system" fn(*mut c_void, *const D3D12_DESCRIPTOR_HEAP_DESC, *const GUID, *mut *mut c_void) -> i32>(dev, 14)(
                dev,
                &hd,
                &ID3D12DescriptorHeap::IID,
                &mut heap,
            );
            check!(t, r >= 0 && !heap.is_null(), "device slot 14 creates a proxied descriptor heap (hr=0x{:08x})", hr(r));
            if !heap.is_null() {
                let cpu = vt::<unsafe extern "system" fn(*mut c_void) -> u64>(heap, 9)(heap);
                let gpu = vt::<unsafe extern "system" fn(*mut c_void) -> u64>(heap, 10)(heap);
                check!(t, cpu != 0 && gpu != 0, "heap slots 9 and 10 return the start handles in rax (cpu {cpu:016X}, gpu {gpu:016X})");
                let mut same: *mut c_void = std::ptr::null_mut();
                let r = vt::<Query>(heap, 0)(heap, &ID3D12DescriptorHeap::IID, &mut same);
                check!(t, r >= 0 && same == heap, "QueryInterface on the heap proxy answers with the proxy");
                if !same.is_null() {
                    release(same);
                }
            }
            if !list.is_null() && !heap.is_null() {
                // Unwrapped for the PC.
                vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(list, 32)(list, 1, &heap);
            }

            // 25 and 26: result pointer first, `this` second.
            let rd = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 65536,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut info = D3D12_RESOURCE_ALLOCATION_INFO { SizeInBytes: 0, Alignment: 0 };
            let info_ptr: *mut c_void = (&raw mut info).cast();
            let r = vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, *const D3D12_RESOURCE_DESC) -> *mut c_void>(dev, 25)(info_ptr, dev, 0, 1, &rd);
            check!(
                t,
                r == info_ptr && info.SizeInBytes == 65536 && info.Alignment == 65536,
                "device slot 25 with the console convention: {} bytes, alignment {}",
                info.SizeInBytes,
                info.Alignment
            );
            let mut props: D3D12_HEAP_PROPERTIES = std::mem::zeroed();
            std::ptr::write_bytes(&raw mut props, 0xCC, 1);
            let props_ptr: *mut c_void = (&raw mut props).cast();
            let r = vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, u32, i32) -> *mut c_void>(dev, 26)(props_ptr, dev, 0, D3D12_HEAP_TYPE_UPLOAD.0);
            check!(
                t,
                r == props_ptr && props.Type == D3D12_HEAP_TYPE_CUSTOM && props.CPUPageProperty != D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                "device slot 26 with the console convention: custom heap, CPU page property {}",
                props.CPUPageProperty.0
            );

            // List run on the console queue, signalled through queue 23 (with flags).
            if !list.is_null() {
                let r = vt::<NoArgs>(list, 9)(list);
                check!(t, r >= 0, "list from slot 73 closes after SetDescriptorHeaps with a proxied heap (hr=0x{:08x})", hr(r));
            }
            if let (false, false, Some(fence)) = (q.is_null(), list.is_null(), fence) {
                vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(q, 10)(q, 1, &list);
                let r = vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, u64, u64) -> i32>(q, 23)(q, raw(fence), 2, 0);
                wait_fence(fence, 2);
                check!(t, r >= 0 && fence.GetCompletedValue() >= 2, "queue slot 23 Signal(fence, 2, flags) reaches the fence");
            }

            // 44: CreatePlacedResourceX backed by a sanitized committed resource on PC.
            let mut res: *mut c_void = std::ptr::null_mut();
            let r = vt::<unsafe extern "system" fn(*mut c_void, u64, *const D3D12_RESOURCE_DESC, u32, *const c_void, *const GUID, *mut *mut c_void) -> i32>(dev, 44)(
                dev,
                0x1000_0000,
                &rd,
                0,
                std::ptr::null(),
                &ID3D12Resource::IID,
                &mut res,
            );
            check!(t, r >= 0 && !res.is_null(), "device slot 44 creates a committed backing resource for a placed VA (hr=0x{:08x})", hr(r));
            release_if(res);

            // 133: newer-layout CreatePlacedResourceX with initial D3D12_BARRIER_LAYOUT (UNDEFINED 0xFFFFFFFF) and console flags 0x20004.
            let mut rd133 = rd;
            rd133.Flags = D3D12_RESOURCE_FLAGS(0x20004);
            let mut res133: *mut c_void = std::ptr::null_mut();
            let r = vt::<unsafe extern "system" fn(*mut c_void, u64, *const D3D12_RESOURCE_DESC, u32, *const c_void, *const GUID, *mut *mut c_void) -> i32>(dev, 133)(
                dev,
                0xA53F_0000,
                &rd133,
                0xFFFF_FFFF,
                std::ptr::null(),
                &ID3D12Resource::IID,
                &mut res133,
            );
            check!(t, r >= 0 && !res133.is_null(), "device slot 133 creates a committed backing resource with barrier layout UNDEFINED and console flags (hr=0x{:08x})", hr(r));
            release_if(res133);

            // 127: newer-layout CreateCommittedResource3 with initial barrier layout and ALLOW_DISPLAY (0x8) stripped.
            let hp = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
            let mut res127: *mut c_void = std::ptr::null_mut();
            let r = vt::<unsafe extern "system" fn(*mut c_void, *const D3D12_HEAP_PROPERTIES, u32, *const D3D12_RESOURCE_DESC, u32, *const c_void, *mut c_void, u32, *const u32, *const GUID, *mut *mut c_void) -> i32>(dev, 127)(
                dev,
                &hp,
                0x8,
                &rd,
                0xFFFF_FFFF,
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
                &ID3D12Resource::IID,
                &mut res127,
            );
            check!(t, r >= 0 && !res127.is_null(), "device slot 127 creates a committed resource with ALLOW_DISPLAY stripped and barrier layout UNDEFINED (hr=0x{:08x})", hr(r));
            release_if(res127);

            // IDXGIDevice.
            let mut dxgi: *mut c_void = std::ptr::null_mut();
            let r = vt::<Query>(dev, 0)(dev, &IDXGIDevice::IID, &mut dxgi);
            check!(t, r >= 0 && !dxgi.is_null(), "the console device answers QueryInterface(IDXGIDevice) (hr=0x{:08x})", hr(r));
            if !dxgi.is_null() {
                let dxgi = IDXGIDevice::from_raw(dxgi);
                let adapter = dxgi.GetAdapter();
                check!(
                    t,
                    adapter.is_ok(),
                    "IDXGIDevice::GetAdapter gives the device's adapter (hr=0x{:08x})",
                    adapter.as_ref().map_or_else(|e| e.code().0 as u32, |_| 0)
                );
                if let Ok(adapter) = adapter {
                    let mut luid = LUID::default();
                    // The PC device behind the proxy (+0x08), explicit aggregate return.
                    let pc_dev = *(dev as *const *mut c_void).add(1);
                    vt::<unsafe extern "system" fn(*mut c_void, *mut LUID) -> *mut LUID>(pc_dev, 43)(pc_dev, &mut luid);
                    let same_luid = adapter.GetDesc().is_ok_and(|desc| desc.AdapterLuid.LowPart == luid.LowPart && desc.AdapterLuid.HighPart == luid.HighPart);
                    check!(t, same_luid, "the adapter has the PC device's LUID");
                }
                let mut back: *mut c_void = std::ptr::null_mut();
                let r = vt::<Query>(raw(&dxgi), 0)(raw(&dxgi), &ID3D12Device::IID, &mut back);
                check!(t, r >= 0 && back == dev, "IDXGIDevice::QueryInterface(ID3D12Device) gives the console device proxy back");
                if !back.is_null() {
                    release(back);
                }
            }

            // Frame events: 60 Hz, three waits of about 50 ms, tokens one apart, a zero timeout
            // before the next one.
            type Interval = unsafe extern "system" fn(*mut c_void, *mut c_void, u32, u32, u32) -> i32;
            type Schedule = unsafe extern "system" fn(*mut c_void, u32, u32, *const c_void, u32) -> i32;
            type Wait = unsafe extern "system" fn(*mut c_void, u32, u32, *const c_void, u32, *mut u64) -> i32;
            let r = vt::<Interval>(dev, 78)(dev, std::ptr::null_mut(), 16667, 1, 0);
            check!(t, r == 0, "device slot 78 SetFrameIntervalX(16667 us) (hr=0x{:08x})", hr(r));
            let r = vt::<Schedule>(dev, 79)(dev, 0, 0, std::ptr::null(), 0);
            check!(t, r == 0x1000_0000 || r == 1, "device slot 79 ScheduleFrameEventX(type 0): STARTED_CLOCK or S_FALSE (hr=0x{:08x})", hr(r));
            let mut tokens = [0u64; 4];
            let mut waits = 0i32;
            let wait = vt::<Wait>(dev, 80);
            waits |= wait(dev, 0, INFINITE, std::ptr::null(), 0, &mut tokens[0]);
            let t0 = Instant::now();
            for token in &mut tokens[1..] {
                waits |= wait(dev, 0, INFINITE, std::ptr::null(), 0, token);
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            check!(t, waits == 0 && ms > 40.0 && ms < 120.0, "device slot 80 WaitFrameEventX waits for each frame: 3 frames in {ms:.1} ms at 60 Hz");
            check!(
                t,
                tokens[0] != 0 && tokens[1] == tokens[0] + 1 && tokens[2] == tokens[1] + 1 && tokens[3] == tokens[2] + 1,
                "frame tokens follow each other ({}, {}, {}, {})",
                tokens[0],
                tokens[1],
                tokens[2],
                tokens[3]
            );
            let mut none = 0u64;
            let r = wait(dev, 0, 0, std::ptr::null(), 0, &mut none);
            check!(t, r == 1 && none == 0, "a zero timeout before the next frame returns S_FALSE, no token without flag 0x2 (hr=0x{:08x})", hr(r));
            let mut advanced = 0u64;
            let r = wait(dev, 0, 0, std::ptr::null(), 0x2, &mut advanced);
            check!(t, r == 1 && advanced != 0 && advanced > tokens[3], "a zero timeout with flags 0x2 still issues a new token and advances the frame (hr=0x{:08x}, token {advanced})", hr(r));
            let r = wait(dev, 0, INFINITE, std::ptr::null(), 0x2, &mut advanced);
            check!(t, hr(r) == E_INVALIDARG, "a passed-in token together with flag 0x2 is refused (hr=0x{:08x})", hr(r));
            let mut bogus = 0xDEAD_u64;
            let r = wait(dev, 0, INFINITE, std::ptr::null(), 0, &mut bogus);
            check!(t, hr(r) == E_INVALIDARG, "a passed-in token this layer never issued is refused (hr=0x{:08x})", hr(r));
            let r = wait(dev, 99, INFINITE, std::ptr::null(), 0, &mut none);
            check!(t, hr(r) == E_INVALIDARG, "an event type out of range is refused (hr=0x{:08x})", hr(r));

            if !heap.is_null() {
                release(heap);
            }
            if !list.is_null() {
                release(list);
            }
            if !q.is_null() {
                release(q);
            }
        }
    }

    /// The root signature of the test: four 32 bit constants and a UAV.
    struct Sample {
        /// The parameters `desc` points to.
        _params: Box<[D3D12_ROOT_PARAMETER1; 2]>,
        desc: D3D12_VERSIONED_ROOT_SIGNATURE_DESC,
    }

    fn sample_desc(flags: u32) -> Sample {
        let params = Box::new([
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 } },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: 0, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ]);
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 2,
                    pParameters: params.as_ptr(),
                    NumStaticSamplers: 0,
                    pStaticSamplers: std::ptr::null(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAGS(flags as i32),
                },
            },
        };
        Sample { _params: params, desc }
    }

    /// The blob behind a pointer the serializer returned, `None` for null.
    ///
    /// # Safety
    /// `blob` is null or an `ID3DBlob` whose reference this takes over.
    unsafe fn blob_of(blob: *mut c_void) -> Option<ID3DBlob> {
        // SAFETY: by the contract above.
        (!blob.is_null()).then(|| unsafe { ID3DBlob::from_raw(blob) })
    }

    /// The bytes of a blob.
    fn bytes(blob: &ID3DBlob) -> &[u8] {
        // SAFETY: the blob owns `GetBufferSize` bytes at `GetBufferPointer` for its lifetime.
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()) }
    }

    pub fn main(xs_path: &OsStr) -> u8 {
        let xs = Library::open(xs_path);
        let pc = Library::open(OsStr::new("d3d12.dll"));
        let (Ok(xs), Ok(pc)) = (xs, pc) else {
            println!("FAIL  cannot load {} or d3d12.dll", xs_path.to_string_lossy());
            return 1;
        };
        // SAFETY: each type is the signature of the export it names.
        let (pc_ser, pc_deser, xs_ser, xs_deser_v, xs_deser, xs_create) = unsafe {
            (
                pc.get::<SerializeVersioned>("D3D12SerializeVersionedRootSignature"),
                pc.get::<Deserializer>("D3D12CreateVersionedRootSignatureDeserializer"),
                xs.get::<SerializeVersioned>("D3D12SerializeVersionedRootSignature"),
                xs.get::<Deserializer>("D3D12CreateVersionedRootSignatureDeserializer"),
                xs.get::<Deserializer>("D3D12CreateRootSignatureDeserializer"),
                xs.get::<XboxCreate>("D3D12XboxCreateDevice"),
            )
        };
        let (Some(pc_ser), Some(pc_deser), Some(xs_ser), Some(xs_deser_v), Some(xs_deser), Some(xs_create)) =
            (pc_ser, pc_deser, xs_ser, xs_deser_v, xs_deser, xs_create)
        else {
            println!("FAIL  missing export");
            return 1;
        };
        let mut t = Test { failures: 0 };

        // 5. Exports a console title imports.
        let iid_data = xs.address("IID_ID3D12Device");
        // SAFETY: the export is a GUID when present.
        check!(t, !iid_data.is_null() && unsafe { *iid_data.cast::<GUID>() } == ID3D12Device::IID, "IID_ID3D12Device is exported as data with the public PC value");
        // SAFETY: the type is the signature of the export.
        let xs_create_pc = unsafe { xs.get::<Create>("D3D12CreateDevice") };
        check!(t, xs_create_pc.is_some(), "D3D12CreateDevice is exported");
        for name in ["DXGIXPresentArray", "DXGIXSetVLineNotification", "DXGIXSetFrameNotification", "DXGIXGetFrameStatistics"] {
            type AnyArgs = unsafe extern "system" fn(u64, u64, u64, u64) -> i32;
            // SAFETY: the DXGIX entry points ignore their arguments, as the console ones do.
            let result = unsafe { xs.get::<AnyArgs>(name) }.map_or(0x8000_4005_u32 as i32, |f| unsafe { f(1, 2, 3, 4) });
            check!(t, result == DXGI_ERROR_UNSUPPORTED.0, "{name} returns DXGI_ERROR_UNSUPPORTED (hr=0x{:08x})", hr(result));
        }

        {
            type Configure = unsafe extern "system" fn(*mut [u64; 4]) -> i32;
            // The title's description: flags 0, 4 page-table pages, preserve 0; outputs set to be cleared.
            let mut desc = [4u64 << 32, 0, u64::MAX, u64::MAX];
            // SAFETY: the entry point reads and writes the 32-byte description passed here.
            let result = unsafe { xs.get::<Configure>("D3D12ConfigureVirtualMemoryX") }.map_or(0x8000_4005_u32 as i32, |f| unsafe { f(&mut desc) });
            check!(t, result == 0x1000_0000 && desc[2] == 0 && desc[3] == 0, "D3D12ConfigureVirtualMemoryX returns 0x10000000 and clears both outputs (hr=0x{:08x})", hr(result));
        }

        const PUB: u32 = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT.0 as u32;
        const CONSOLE: u32 = 0x0300_0000;

        // 1. Checksum.
        let d = sample_desc(PUB);
        let mut reference: *mut c_void = std::ptr::null_mut();
        let mut err: *mut c_void = std::ptr::null_mut();
        // SAFETY: `d` is a valid description that outlives the call, the outputs are writable.
        let r = unsafe { pc_ser(&d.desc, &mut reference, &mut err) };
        if r < 0 {
            println!("FAIL  PC serializer, hr=0x{:08x}", hr(r));
            return 1;
        }
        // SAFETY: the serializer returned blobs that this program owns.
        let (reference, _) = unsafe { (blob_of(reference).expect("blob"), blob_of(err)) };
        let rb = bytes(&reference);
        let rsize = rb.len();
        let sum = dxbc_checksum(rb, rsize);
        check!(t, (0..4).all(|i| rd32(rb, 4 + 4 * i) == sum[i]), "DXBC checksum matches the PC runtime's ({rsize}-byte container)");
        check!(t, flags_offset(rb).is_some_and(|at| rd32(rb, at) == PUB), "RTS0 flags found at the expected place");

        // Information: the plain PC runtime facing the console bits.
        let dc = sample_desc(PUB | CONSOLE);
        let (mut b, mut e): (*mut c_void, *mut c_void) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: as above.
        let r = unsafe { pc_ser(&dc.desc, &mut b, &mut e) };
        println!("info  plain PC serializer with flags 0x03000001: hr=0x{:08x}", hr(r));
        // SAFETY: the blobs belong to this program.
        drop(unsafe { (blob_of(b), blob_of(e)) });

        // 2. Serialize through d3d12_xs.
        let (mut b, mut e): (*mut c_void, *mut c_void) = (std::ptr::null_mut(), std::ptr::null_mut());
        // SAFETY: as above.
        let r = unsafe { xs_ser(&dc.desc, &mut b, &mut e) };
        check!(t, r >= 0, "d3d12_xs serializes a root signature with flags 0x03000001 (hr=0x{:08x})", hr(r));
        // SAFETY: the blobs belong to this program.
        let (b, _e) = unsafe { (blob_of(b), blob_of(e)) };
        if let Some(b) = b {
            let p = bytes(&b);
            check!(t, flags_offset(p).is_some_and(|at| rd32(p, at) == PUB), "the serialized blob carries the public flags only");
        }

        // 3. Deserialize a container carrying the console bits, stale checksum.
        let mut patched = rb.to_vec();
        let at = flags_offset(&patched).expect("flags");
        wr32(&mut patched, at, PUB | CONSOLE);
        let mut vd: *mut c_void = std::ptr::null_mut();
        let iid_versioned = ID3D12VersionedRootSignatureDeserializer::IID;
        // SAFETY: `patched` is a readable buffer of the stated size, `vd` is writable.
        let r = unsafe { pc_deser(patched.as_ptr().cast(), rsize, &iid_versioned, &mut vd) };
        println!("info  plain PC deserializer on the patched container: hr=0x{:08x}", hr(r));
        release_if(vd);
        {
            // Same, with a correct checksum: does the PC runtime reject the flag bits themselves?
            let mut fixed = patched.clone();
            let s = dxbc_checksum(&fixed, rsize);
            for (i, word) in s.iter().enumerate() {
                wr32(&mut fixed, 4 + 4 * i, *word);
            }
            vd = std::ptr::null_mut();
            // SAFETY: as above.
            let r = unsafe { pc_deser(fixed.as_ptr().cast(), rsize, &iid_versioned, &mut vd) };
            println!("info  plain PC deserializer, console flags with a valid checksum: hr=0x{:08x}", hr(r));
            release_if(vd);
        }

        vd = std::ptr::null_mut();
        // SAFETY: as above.
        let r = unsafe { xs_deser_v(patched.as_ptr().cast(), rsize, &iid_versioned, &mut vd) };
        check!(t, r >= 0 && !vd.is_null(), "d3d12_xs versioned deserializer accepts it (hr=0x{:08x})", hr(r));
        if !vd.is_null() {
            // SAFETY: the deserializer returned an `ID3D12VersionedRootSignatureDeserializer`.
            let deser = unsafe { ID3D12VersionedRootSignatureDeserializer::from_raw(vd) };
            // SAFETY: the description lives as long as the deserializer.
            let out = unsafe { deser.GetUnconvertedRootSignatureDesc().as_ref() };
            // SAFETY: the version is 1.1, so `Desc_1_1` is the active member.
            let (flags, count) = out.map_or((0, 0), |o| unsafe { (o.Anonymous.Desc_1_1.Flags.0 as u32, o.Anonymous.Desc_1_1.NumParameters) });
            check!(t, out.is_some() && flags == PUB && count == 2, "deserialized: flags 0x{flags:x}, {count} parameters");
        }
        let mut rd: *mut c_void = std::ptr::null_mut();
        // SAFETY: as above.
        let r = unsafe { xs_deser(patched.as_ptr().cast(), rsize, &ID3D12RootSignatureDeserializer::IID, &mut rd) };
        check!(t, r >= 0 && !rd.is_null(), "d3d12_xs v1.0 deserializer accepts it (hr=0x{:08x})", hr(r));
        if !rd.is_null() {
            // SAFETY: the deserializer returned an `ID3D12RootSignatureDeserializer`.
            let deser = unsafe { ID3D12RootSignatureDeserializer::from_raw(rd) };
            // SAFETY: the description lives as long as the deserializer.
            let out = unsafe { deser.GetRootSignatureDesc().as_ref() };
            let flags = out.map_or(0, |o| o.Flags.0 as u32);
            check!(t, out.is_some_and(|o| o.Flags.0 as u32 == PUB && o.NumParameters == 2), "deserialized v1.0 view: flags 0x{flags:x}");
        }
        check!(t, rd32(&patched, at) == (PUB | CONSOLE), "the caller's buffer is left untouched");
        drop(reference);

        // 4. Console ABI proxies, driven only through console vtable slots.
        let mut dev: *mut c_void = std::ptr::null_mut();
        // SAFETY: the export takes (adapter, parameters, riid, device).
        let r = unsafe { xs_create(std::ptr::null_mut(), std::ptr::null(), &ID3D12Device::IID, &mut dev) };
        if r < 0 {
            println!("skip  no D3D12 12_0 device on this machine (hr=0x{:08x}): device checks skipped", hr(r));
        } else {
            device_checks(&mut t, dev, xs_create, xs_create_pc, pc_ser);
        }

        // 6. Title memory: pages that xmem maps are read by the GPU through a heap on their section.
        title_memory_checks(&mut t, xs_path);

        // 7. The console slots the title calls while it boots, on a device created as ID3D12Device11.
        // Last: that device switches the layer to the newer console layout for the process.
        pipeline_slots::console_pipeline_slots(&mut t, xs_create);

        // 8. Root views at title addresses read by a shader (with DECANT_D3D12X_TITLE_MEM=1).
        root_views::root_view_reads(&mut t, xs_path, xs_create);

        if t.failures != 0 {
            println!("d3d12xs_test: {} failure(s)", t.failures);
        } else {
            println!("d3d12xs_test: all checks passed");
        }
        u8::from(t.failures != 0)
    }

    type VirtualAlloc = unsafe extern "system" fn(*mut c_void, usize, u32, u32) -> *mut c_void;
    type XFree = unsafe extern "system" fn(*mut c_void, u64);
    type PhysAlloc = unsafe extern "system" fn(u64, *mut usize, *mut u64) -> i32;
    type PhysMap = unsafe extern "system" fn(*mut c_void, usize, *const u64) -> *mut c_void;
    type PhysFree = unsafe extern "system" fn(usize, *const u64) -> i32;

    /// The `xmem.dll` that sits next to `d3d12_xs.dll`, or in the folder above it (the `deps` folder
    /// of `cargo test`).
    fn xmem_path(xs_path: &OsStr) -> Option<std::path::PathBuf> {
        let dir = std::path::Path::new(xs_path).parent()?;
        [dir.join("xmem.dll"), dir.parent()?.join("xmem.dll")].into_iter().find(|p| p.is_file())
    }



    /// The commit this process holds, in MiB: committed private and mapped pages found by
    /// `VirtualQuery` (images excluded). A leaked section would add to it. The process counters
    /// (`PrivateUsage`) do not show the commit of sections, and the free commit of the system changes
    /// with every other process (a game run), so neither is used. Printed before and after the title
    /// memory checks.
    fn process_commit_mib() -> u64 {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_IMAGE, MEMORY_BASIC_INFORMATION, VirtualQuery};
        let (mut at, mut total) = (0usize, 0u64);
        loop {
            // SAFETY: MEMORY_BASIC_INFORMATION is plain data.
            let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is a valid output buffer of the size passed.
            if unsafe { VirtualQuery(at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) } == 0 {
                break;
            }
            if info.State == MEM_COMMIT && info.Type != MEM_IMAGE {
                total += info.RegionSize as u64;
            }
            match (info.BaseAddress as usize).checked_add(info.RegionSize) {
                Some(next) if next > at => at = next,
                _ => break,
            }
        }
        total >> 20
    }

    /// Makes the GPU mirror current for `queue`, as the proxy does before each submission.
    fn sync(queue: &ID3D12CommandQueue) {
        // SAFETY: the queue is live.
        unsafe { crate::title_mem::sync(queue.as_raw()) };
    }

    /// `words` words of title memory at `va` as the GPU reads them through the window, after a sync.
    fn gpu_read(gpu: &mut crate::gpu_copy::Gpu, va: u64, words: u32) -> Option<Vec<u32>> {
        sync(&gpu.queue.clone());
        let span = crate::title_mem::translate(va, u64::from(words) * 4)?;
        gpu.copy(span.gpu, None, words)
    }

    /// Writes words through views that `xmem.dll` mapped, and reads them from the GPU through the window of
    /// `title_mem`: the first picture of a page, an alias, a remap, an unmap, a write after a sync and a span
    /// over scattered pages, and an arena commit with a console page bit that reaches the GPU once it is given
    /// an address in it. Prints the commit charge of the process before and after (a leak would show). Skipped without
    /// `xmem.dll` or without a GPU.
    fn title_memory_checks(t: &mut Test, xs_path: &OsStr) {
        const PAGE: usize = 0x1_0000;
        if !std::env::var_os("DECANT_D3D12X_TITLE_MEM").is_some_and(|v| v == "1") {
            println!("skip  title memory checks: DECANT_D3D12X_TITLE_MEM=1 runs them");
            return;
        }
        let Some(path) = xmem_path(xs_path) else {
            println!("skip  no xmem.dll next to {}: title memory checks skipped", xs_path.to_string_lossy());
            return;
        };
        let Ok(xmem) = Library::open(path.as_os_str()) else {
            println!("FAIL  cannot load {}", path.display());
            t.failures += 1;
            return;
        };
        println!("process commit before the title memory checks: {} MiB", process_commit_mib());
        let Some(mut gpu) = crate::gpu_copy::Gpu::new() else {
            println!("skip  no D3D12 12_0 device on this machine: title memory checks skipped");
            return;
        };
        // SAFETY: each type is the signature of the export it names.
        let (alloc, free, phys_alloc, phys_map, phys_free) = unsafe {
            (
                xmem.get::<VirtualAlloc>("XMemVirtualAlloc"),
                xmem.get::<XFree>("XMemFree"),
                xmem.get::<PhysAlloc>("XMemAllocatePhysicalPages"),
                xmem.get::<PhysMap>("XMemMapPhysicalPages"),
                xmem.get::<PhysFree>("XMemFreePhysicalPages"),
            )
        };
        let (Some(alloc), Some(free), Some(phys_alloc), Some(phys_map), Some(phys_free)) = (alloc, free, phys_alloc, phys_map, phys_free) else {
            println!("FAIL  xmem.dll lacks an XMem export");
            t.failures += 1;
            return;
        };
        // SAFETY: the calls follow the console prototypes of the exports. Views are written inside the pages
        // mapped into them. `device` is the live device of `gpu`.
        unsafe {
            crate::title_mem::device_created(gpu.device.as_raw());
            // 0x42000 reserves a placeholder that frames map into, protection read write.
            let base = alloc(std::ptr::null_mut(), 8 * PAGE, 0x42000, 0x04).cast::<u8>();
            let mut count = 4usize;
            let mut frames = [0u64; 4];
            let got = phys_alloc(0x2040_0000, &mut count, frames.as_mut_ptr());
            check!(t, !base.is_null() && got != 0 && count == 4 && frames[1] == frames[0] + 16, "xmem gives a placeholder and four pages");
            if base.is_null() || got == 0 {
                return;
            }
            let at = |page: usize| base.add(page * PAGE).cast::<u32>();
            let map = |page: usize, frame: Option<&u64>| phys_map(base.add(page * PAGE).cast(), 1, frame.map_or(std::ptr::null(), |f| f as *const u64));
            let ok = map(0, Some(&frames[0])) == base.cast() && map(1, Some(&frames[1])) == base.add(PAGE).cast();
            check!(t, ok, "XMemMapPhysicalPages maps two pages");
            for i in 0..64 {
                at(0).add(i).write_volatile(0xC000_0000 + i as u32);
                at(1).add(i).write_volatile(0xD000_0000 + i as u32);
            }
            let va = |page: usize, off: usize| base as u64 + (page * PAGE + off) as u64;
            // 1. The pages the CPU wrote are read by the GPU with no copy command of ours.
            let seen = gpu_read(&mut gpu, va(0, 0), 64);
            check!(t, seen.is_some_and(|w| w.iter().enumerate().all(|(i, &x)| x == 0xC000_0000 + i as u32)), "the GPU reads the first page the CPU wrote");
            let seen = gpu_read(&mut gpu, va(1, 0), 64);
            check!(t, seen.is_some_and(|w| w.iter().enumerate().all(|(i, &x)| x == 0xD000_0000 + i as u32)), "the GPU reads the second page");
            let seen = gpu_read(&mut gpu, va(1, 0) - 8, 4);
            check!(t, seen.is_some_and(|w| w[2] == 0xD000_0000 && w[3] == 0xD000_0001), "a span over two pages is one run on the GPU");
            let (first, second) = (crate::title_mem::translate(va(0, 0), 4), crate::title_mem::translate(va(1, 0), 4));
            check!(t, first.zip(second).is_some_and(|(a, b)| a.resource == b.resource && b.gpu == a.gpu + PAGE as u64), "both pages are in one slot, a page apart");
            check!(t, crate::title_mem::translate(va(2, 0), 4).is_none(), "an unmapped page has no GPU address");
            check!(t, crate::title_mem::translate(0x1000, 4).is_none(), "an address outside xmem has none either");
            // 2. One page at two addresses: a write through one is read through the other.
            check!(t, map(3, Some(&frames[0])) == base.add(3 * PAGE).cast(), "the first page is mapped again at page 3");
            at(3).add(5).write_volatile(0xA11A_5000);
            let seen = gpu_read(&mut gpu, va(0, 0), 8);
            check!(t, seen.is_some_and(|w| w[5] == 0xA11A_5000 && w[4] == 0xC000_0004), "a write through the alias is read through the first address");
            let seen = gpu_read(&mut gpu, va(3, 0), 8);
            check!(t, seen.is_some_and(|w| w[5] == 0xA11A_5000), "and through the alias itself");
            // 3. A page remapped to another frame shows the other frame.
            let remap = map(1, Some(&frames[2]));
            check!(t, remap == base.add(PAGE).cast(), "page 1 is mapped to the third frame");
            at(1).add(1).write_volatile(0xE000_0001);
            let seen = gpu_read(&mut gpu, va(1, 0), 4);
            check!(t, seen.is_some_and(|w| w[1] == 0xE000_0001 && w[0] == 0), "a remapped page shows the new frame");
            // 4. An unmapped page reads as zeros.
            let gone = crate::title_mem::translate(va(1, 0), 4).map(|s| s.gpu);
            check!(t, map(1, None) == base.add(PAGE).cast(), "page 1 is unmapped");
            sync(&gpu.queue.clone());
            let seen = gone.and_then(|g| gpu.copy(g, None, 4));
            check!(t, seen.is_some_and(|w| w == [0, 0, 0, 0]), "an unmapped page reads as zeros");
            // 5. A write after a sync reaches the GPU at the next one, with no fence on the CPU side.
            for round in 0..5u32 {
                at(0).add(9).write_volatile(0x5EED_0000 + round);
                let seen = gpu_read(&mut gpu, va(0, 0), 16);
                check!(t, seen.is_some_and(|w| w[9] == 0x5EED_0000 + round), "round {round}: a write after a sync is read after the next one");
            }
            // 6. A span over two pages whose frames are not consecutive reads both.
            let scattered = map(4, Some(&frames[0])) == base.add(4 * PAGE).cast() && map(5, Some(&frames[3])) == base.add(5 * PAGE).cast();
            check!(t, scattered, "pages 4 and 5 map frames that are not consecutive");
            at(5).write_volatile(0xB0B0_0005);
            let seen = gpu_read(&mut gpu, va(5, 0) - 8, 4);
            check!(t, seen.is_some_and(|w| w[0] == 0 && w[1] == 0 && w[2] == 0xB0B0_0005), "a span over scattered frames reads the second one");
            let seen = gpu_read(&mut gpu, va(4, 0), 8);
            check!(t, seen.is_some_and(|w| w[5] == 0xA11A_5000), "and the first one at its own address");
            // 7. The detile pass reads title memory through the window.
            let detile_base = alloc(std::ptr::null_mut(), 2 * PAGE, 0x42000, 0x04).cast::<u8>();
            let ok = !detile_base.is_null() && phys_map(detile_base.cast(), 2, frames.as_ptr()) == detile_base.cast();
            check!(t, ok, "two pages for the detile checks");
            if ok {
                detile_checks(t, &mut gpu, detile_base);
            }
            // 8. A commit with a console page bit (a title's arenas, 0x20003000 with protection 0x183): pool
            // pages below 16 GiB, which reach the GPU once it is given an address in them.
            const ARENA: usize = 4 << 20;
            let before = process_commit_mib();
            let arena = alloc(std::ptr::null_mut(), ARENA, 0x2000_3000, 0x183).cast::<u8>();
            check!(t, !arena.is_null() && arena as usize + ARENA <= 16 << 30, "an arena commit lies below 16 GiB");
            if !arena.is_null() {
                let data = arena.add(0x2_2400).cast::<u32>();
                let va = arena as u64 + 0x2_2400;
                for i in 0..16 {
                    data.add(i).write_volatile(0xA7E0_0000 + i as u32);
                }
                check!(t, crate::title_mem::lookup(va, 64).is_none(), "an arena page the GPU was not given has no GPU address yet");
                check!(t, crate::title_mem::translate(va, 64).is_some(), "given to the GPU, it has one");
                let seen = gpu_read(&mut gpu, va, 16);
                check!(t, seen.is_some_and(|w| w.iter().enumerate().all(|(i, &x)| x == 0xA7E0_0000 + i as u32)), "the GPU reads what the title wrote into the arena");
                data.add(3).write_volatile(0x5EED_A7E0);
                let seen = gpu_read(&mut gpu, va, 16);
                check!(t, seen.is_some_and(|w| w[3] == 0x5EED_A7E0 && w[4] == 0xA7E0_0004), "a write after the sync reaches the GPU at the next one");
                free(arena.cast(), 0);
                check!(t, crate::title_mem::lookup(va, 4).is_none(), "XMemFree takes the arena out of the window");
            }
            println!("process commit around the arena: {before} MiB before, {} MiB after XMemFree", process_commit_mib());
            free(detile_base.cast(), 0);
            free(base.cast(), 0);
            phys_free(4, frames.as_ptr());
        }
        println!("process commit after the title memory checks: {} MiB", process_commit_mib());
        if std::env::var_os("DECANT_TEST_HOLD").is_some() {
            use std::io::Write;
            println!("HOLDING pid {} commit {} MiB", std::process::id(), process_commit_mib());
            let _ = std::io::stdout().flush();
            std::thread::sleep(Duration::from_secs(600));
        }
    }

    /// A tiled texture laid out by `decant_texture::xg` in title memory (an `xmem.dll` view) is detiled by
    /// the compute pass into a PC texture, and the rows read back are the linear texels. One RGBA8
    /// texture with mips and array slices, one BC1 texture, both in the default layout of a sampled texture.
    ///
    /// # Safety
    /// `base` is a view of `xmem.dll` with the first two pages (64 KiB each) mapped, and `gpu`'s device is the
    /// one `title_mem` was told about.
    unsafe fn detile_checks(t: &mut Test, gpu: &mut crate::gpu_copy::Gpu, base: *mut u8) {
        use decant_texture::xg::compute_layout;
        use decant_texture::xg::d3d12::{D3d12ResourceDesc, texture_desc_from_d3d12};
        use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
        fn word(mip: u32, x: u32, y: u32, z: u32, k: u32) -> u32 {
            ((mip << 28) ^ (z << 24) ^ (y << 12) ^ x).wrapping_mul(0x9E37_79B1) ^ k.wrapping_mul(0x85EB_CA6B)
        }
        let device = gpu.device.clone();
        // SAFETY: plain D3D12 calls on objects this function makes, and writes inside the mapped pages of `base`.
        unsafe {
            let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).expect("a queue");
            let Some(mut detiler) = crate::detile::Detiler::new(&device) else {
                println!("FAIL  the detile pipeline cannot be created");
                t.failures += 1;
                return;
            };
            for (format, name, offset, mips, slices) in [(28u32, "RGBA8", 0usize, 3u16, 2u16), (71, "BC1", 0x1_0000, 4, 1)] {
                let mut raw = [0u8; 0x38];
                raw[0..4].copy_from_slice(&3u32.to_le_bytes());
                raw[0x10..0x18].copy_from_slice(&64u64.to_le_bytes());
                raw[0x18..0x1C].copy_from_slice(&64u32.to_le_bytes());
                raw[0x1C..0x1E].copy_from_slice(&slices.to_le_bytes());
                raw[0x1E..0x20].copy_from_slice(&mips.to_le_bytes());
                raw[0x20..0x24].copy_from_slice(&format.to_le_bytes());
                raw[0x24..0x28].copy_from_slice(&1u32.to_le_bytes());
                let plan = crate::detile_plan::plan(&raw, false).expect("a plan");
                let layout = compute_layout(&texture_desc_from_d3d12(&D3d12ResourceDesc::from_bytes(&raw)).expect("a description"), false).expect("a layout");
                let bpe = plan.bytes_per_element as usize;
                std::ptr::write_bytes(base.add(offset), 0, plan.size as usize);
                for (mip, l) in layout.plane0().levels.iter().enumerate() {
                    for z in 0..u32::from(slices) {
                        for y in 0..l.height_elements {
                            for x in 0..l.width_elements {
                                let at = layout.element_offset(mip as u32, u64::from(x), y, z, 0).expect("an offset") as usize;
                                for k in 0..bpe / 4 {
                                    base.add(offset + at + k * 4).cast::<u32>().write_unaligned(word(mip as u32, x, y, z, k as u32));
                                }
                            }
                        }
                    }
                }
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 64,
                    Height: 64,
                    DepthOrArraySize: slices,
                    MipLevels: mips,
                    Format: DXGI_FORMAT(format as i32),
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                let mut texture: Option<ID3D12Resource> = None;
                device.CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut texture).expect("a texture");
                let texture = texture.expect("a texture");
                let Some(span) = crate::title_mem::translate(base as u64 + offset as u64, plan.size) else {
                    println!("FAIL  {name}: no GPU address for the texture memory");
                    t.failures += 1;
                    continue;
                };
                let jobs = [crate::detile::Job { texture: texture.clone(), src_gpu: span.gpu, plan: &plan, states: &[0] }];
                sync(&queue);
                let done = detiler.run(&queue, &jobs, true);
                check!(t, done == [true], "{name}: the detile pass is submitted");
                // Read every subresource back through a row major buffer.
                let n = u32::from(mips) * u32::from(slices);
                let mut fps = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); n as usize];
                let mut total = 0u64;
                device.GetCopyableFootprints(&desc, 0, n, 0, Some(fps.as_mut_ptr()), None, None, Some(&mut total));
                let mut readback: Option<ID3D12Resource> = None;
                let rprops = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                device.CreateCommittedResource(&rprops, D3D12_HEAP_FLAG_NONE, &crate::gpu_copy::buffer_desc(total, D3D12_RESOURCE_FLAG_NONE), D3D12_RESOURCE_STATE_COPY_DEST, None, &mut readback).expect("a readback buffer");
                let readback = readback.expect("a readback buffer");
                let copied = gpu.submit(|list| {
                    for (i, fp) in fps.iter().enumerate() {
                        let dst = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&readback),
                            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: *fp },
                        };
                        let src = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&texture),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: i as u32 },
                        };
                        list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                    }
                });
                check!(t, copied, "{name}: the texture is read back");
                let mut mapped = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut mapped)).expect("a mapped readback buffer");
                let mut wrong = 0u32;
                for (i, fp) in fps.iter().enumerate() {
                    let (mip, z) = (i as u32 % u32::from(mips), i as u32 / u32::from(mips));
                    let l = &plan.mips[mip as usize];
                    for y in 0..l.height_elements {
                        for x in 0..l.width_elements {
                            for k in 0..bpe / 4 {
                                let at = fp.Offset as usize + y as usize * fp.Footprint.RowPitch as usize + x as usize * bpe + k * 4;
                                if mapped.cast::<u8>().add(at).cast::<u32>().read_unaligned() != word(mip, x, y, z, k as u32) {
                                    wrong += 1;
                                }
                            }
                        }
                    }
                }
                readback.Unmap(0, None);
                check!(t, wrong == 0, "{name}: the detiled texels are the linear ones ({wrong} words differ, {n} subresources)");
            }
        }
    }

    fn release_if(object: *mut c_void) {
        if !object.is_null() {
            release(object);
        }
    }

    fn device_checks(t: &mut Test, dev: *mut c_void, xs_create: XboxCreate, xs_create_pc: Option<Create>, pc_ser: SerializeVersioned) {
        // SAFETY: `dev` is the console device proxy, every slot is called with the argument shapes
        // of the console order, and each object is released once.
        unsafe {
            check!(t, !dev.is_null(), "D3D12XboxCreateDevice returns a device for IID_ID3D12Device");
            let console_iid = GUID::from_values(0x12345678, 0x9abc, 0xdef0, [1, 2, 3, 4, 5, 6, 7, 8]);
            let mut dev2: *mut c_void = std::ptr::null_mut();
            let r = xs_create(std::ptr::null_mut(), std::ptr::null(), &console_iid, &mut dev2);
            check!(t, r >= 0 && !dev2.is_null(), "a console-only IID also gets the console device proxy (hr=0x{:08x})", hr(r));
            release_if(dev2);

            // Device slot 16 = CreateRootSignature, PC order (measured).
            let rsd = sample_desc(D3D12_ROOT_SIGNATURE_FLAG_NONE.0 as u32);
            let mut rsblob: *mut c_void = std::ptr::null_mut();
            pc_ser(&rsd.desc, &mut rsblob, std::ptr::null_mut());
            let rsblob = blob_of(rsblob);
            let mut rs: *mut c_void = std::ptr::null_mut();
            type CreateRs = unsafe extern "system" fn(*mut c_void, u32, *const c_void, usize, *const GUID, *mut *mut c_void) -> i32;
            let r = match &rsblob {
                Some(blob) => vt::<CreateRs>(dev, 16)(dev, 0, blob.GetBufferPointer(), blob.GetBufferSize(), &ID3D12RootSignature::IID, &mut rs),
                None => 0x8000_4005_u32 as i32,
            };
            check!(t, r >= 0 && !rs.is_null(), "device slot 16 relays CreateRootSignature (hr=0x{:08x})", hr(r));
            release_if(rs);
            drop(rsblob);

            let r = vt::<NoArgs>(dev, 64)(dev);
            check!(t, hr(r) == E_NOTIMPL, "an unknown console device slot (64) returns E_NOTIMPL (hr=0x{:08x})", hr(r));

            // Queue (device slot 8), allocator (9), list (12).
            let qd = D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, Priority: 0, Flags: D3D12_COMMAND_QUEUE_FLAG_NONE, NodeMask: 0 };
            let (mut q, mut alloc, mut list, mut fence): (*mut c_void, *mut c_void, *mut c_void, *mut c_void) =
                (std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut());
            type CreateQueue = unsafe extern "system" fn(*mut c_void, *const D3D12_COMMAND_QUEUE_DESC, *const GUID, *mut *mut c_void) -> i32;
            type CreateAlloc = unsafe extern "system" fn(*mut c_void, i32, *const GUID, *mut *mut c_void) -> i32;
            type CreateList = unsafe extern "system" fn(*mut c_void, u32, i32, *mut c_void, *mut c_void, *const GUID, *mut *mut c_void) -> i32;
            type CreateFence = unsafe extern "system" fn(*mut c_void, u64, i32, *const GUID, *mut *mut c_void) -> i32;
            let r = vt::<CreateQueue>(dev, 8)(dev, &qd, &ID3D12CommandQueue::IID, &mut q);
            check!(t, r >= 0 && !q.is_null(), "device slot 8 creates a (proxied) queue (hr=0x{:08x})", hr(r));
            let r = vt::<CreateAlloc>(dev, 9)(dev, D3D12_COMMAND_LIST_TYPE_DIRECT.0, &ID3D12CommandAllocator::IID, &mut alloc);
            check!(t, r >= 0 && !alloc.is_null(), "device slot 9 creates an allocator (hr=0x{:08x})", hr(r));
            let r = if alloc.is_null() {
                0x8000_4005_u32 as i32
            } else {
                vt::<CreateList>(dev, 12)(dev, 0, D3D12_COMMAND_LIST_TYPE_DIRECT.0, alloc, std::ptr::null_mut(), &ID3D12GraphicsCommandList::IID, &mut list)
            };
            check!(t, r >= 0 && !list.is_null(), "device slot 12 creates a (proxied) command list (hr=0x{:08x})", hr(r));
            // A DMA list: the console's own IID, a copy list. It has its own small vtable: Close at 9,
            // S_OK from the methods past the copies.
            {
                let dma_iid = GUID::from_u128(0x30bc77b9_8379_4b25_95de_b8588c42275d);
                let mut copy_alloc: *mut c_void = std::ptr::null_mut();
                let mut dma: *mut c_void = std::ptr::null_mut();
                let r = vt::<CreateAlloc>(dev, 9)(dev, D3D12_COMMAND_LIST_TYPE_COPY.0, &ID3D12CommandAllocator::IID, &mut copy_alloc);
                let r = if r >= 0 { vt::<CreateList>(dev, 12)(dev, 0, D3D12_COMMAND_LIST_TYPE_COPY.0, copy_alloc, std::ptr::null_mut(), &dma_iid, &mut dma) } else { r };
                check!(t, r >= 0 && !dma.is_null(), "device slot 12 creates a DMA list from its own IID (hr=0x{:08x})", hr(r));
                if !dma.is_null() {
                    let r = vt::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(dma, 20)(dma, 1, 2);
                    check!(t, r == 0, "a DMA list method past the copies returns S_OK (hr=0x{:08x})", hr(r));
                    let r = vt::<NoArgs>(dma, 9)(dma);
                    check!(t, r >= 0, "DMA list slot 9 Close reaches the PC list (hr=0x{:08x})", hr(r));
                    release(dma);
                }
                release_if(copy_alloc);
            }
            if !list.is_null() {
                let f = list as *const *const c_void;
                check!(t, !(*f.add(3)).is_null() && *f.add(3) == *f.add(5) && *f.add(3) == *f.add(6), "list cursor +0x18 == limits +0x28/+0x30: every command goes through the vtable");
                let mut same: *mut c_void = std::ptr::null_mut();
                let r = vt::<Query>(list, 0)(list, &ID3D12GraphicsCommandList::IID, &mut same);
                check!(t, r >= 0 && same == list, "QueryInterface on the list proxy answers with the proxy");
                release_if(same);
                // Console 32 = PC SetDescriptorHeaps.
                vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(list, 32)(list, 0, std::ptr::null());
                let r = vt::<NoArgs>(list, 66)(list);
                check!(t, hr(r) == E_NOTIMPL, "a console-only list slot (66) returns E_NOTIMPL");
                // Newer-layout list slot 70 (OMSetDepthBounds, a void method): forwarded to PC when the
                // newer layout is forced (DECANT_D3D12X_LAYOUT=late), logged as an unknown slot otherwise.
                vt::<unsafe extern "system" fn(*mut c_void, f32, f32)>(list, 70)(list, 0.0, 1.0);
                // Close.
                let r = vt::<NoArgs>(list, 9)(list);
                check!(t, r >= 0, "list slot 9 Close (hr=0x{:08x})", hr(r));
                // Allocator proxy: slot 9 (memory callbacks), slot 8 Reset, then list slot 10 Reset
                // with the proxy, which must reach the PC list as the PC allocator.
                if !alloc.is_null() {
                    extern "system" fn dummy_alloc(_size: usize, _align: usize, _ctx: usize) -> usize {
                        0
                    }
                    let n = vt::<unsafe extern "system" fn(*mut c_void) -> u32>(alloc, 1)(alloc);
                    check!(t, n == 2, "allocator slot 1 AddRef returns 2 ({n})");
                    let n = release(alloc);
                    check!(t, n == 1, "allocator slot 2 Release returns 1 ({n})");
                    let mut same: *mut c_void = std::ptr::null_mut();
                    let r = vt::<Query>(alloc, 0)(alloc, &ID3D12CommandAllocator::IID, &mut same);
                    check!(t, r >= 0 && same == alloc, "QueryInterface on the allocator proxy answers with the proxy");
                    if !same.is_null() {
                        let n = release(same);
                        check!(t, n == 1, "the QueryInterface reference releases back to 1 ({n})");
                    }
                    let r = vt::<unsafe extern "system" fn(*mut c_void, usize, usize) -> i32>(alloc, 9)(alloc, dummy_alloc as *const () as usize, 0x2000);
                    check!(t, r == 0, "allocator slot 9 (memory callbacks) returns S_OK (hr=0x{:08x})", hr(r));
                    // List slot 148 AllocateUploadMemory (the list was created with this allocator): memory
                    // the title can write, aligned, one block after the other, freed by the allocator Reset.
                    type Alloc = unsafe extern "system" fn(*mut c_void, usize, usize) -> *mut u8;
                    let p1 = vt::<Alloc>(list, 148)(list, 100, 64);
                    let p2 = vt::<Alloc>(list, 148)(list, 16, 16);
                    check!(t, !p1.is_null() && !p2.is_null(), "list slot 148 AllocateUploadMemory returns memory");
                    if !p1.is_null() && !p2.is_null() {
                        check!(t, (p1 as usize).is_multiple_of(64), "the block honours the alignment 64");
                        check!(t, p2 as usize >= p1 as usize + 100, "the second block follows the first");
                        p1.write_bytes(0xAB, 100);
                        check!(t, p1.add(99).read() == 0xAB, "the memory is writable");
                    }
                    let r = vt::<NoArgs>(alloc, 8)(alloc);
                    check!(t, r >= 0, "allocator slot 8 Reset relays to the PC allocator (hr=0x{:08x})", hr(r));
                    let r = vt::<NoArgs>(alloc, 10)(alloc);
                    check!(t, hr(r) == E_NOTIMPL, "allocator slot 10 returns E_NOTIMPL (hr=0x{:08x})", hr(r));
                    let r = vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> i32>(list, 10)(list, alloc, std::ptr::null_mut());
                    check!(t, r >= 0, "list slot 10 Reset with the allocator proxy (hr=0x{:08x})", hr(r));
                    let r = vt::<NoArgs>(list, 9)(list);
                    check!(t, r >= 0, "list slot 9 Close after the Reset (hr=0x{:08x})", hr(r));
                }
            }
            let r = vt::<CreateFence>(dev, 36)(dev, 0, D3D12_FENCE_FLAG_NONE.0, &ID3D12Fence::IID, &mut fence);
            check!(t, r >= 0 && !fence.is_null(), "device slot 36 creates a fence (hr=0x{:08x})", hr(r));
            let fence_if = (!fence.is_null()).then(|| ID3D12Fence::from_raw(fence));
            if let (false, false, Some(fence)) = (q.is_null(), list.is_null(), &fence_if) {
                vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void)>(q, 10)(q, 1, &list);
                // Newer-layout queue slot 35 (ExecuteCommandListsX(count, lists, value, flags)) runs the
                // closed list again when the newer layout is forced, and is logged as unknown otherwise.
                vt::<unsafe extern "system" fn(*mut c_void, u32, *const *mut c_void, u64, u32)>(q, 35)(q, 1, &list, 0, 0);
                let r = vt::<unsafe extern "system" fn(*mut c_void, *mut c_void, u64) -> i32>(q, 14)(q, raw(fence), 1);
                let mut done = 0u64;
                for _ in 0..500 {
                    if done >= 1 {
                        break;
                    }
                    done = fence.GetCompletedValue();
                    if done < 1 {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                check!(t, r >= 0 && done >= 1, "queue slot 10 runs the proxied list on the GPU, slot 14 signals the fence (value {done})");
                let r = vt::<unsafe extern "system" fn(*mut c_void, u32, *const c_void, *const c_void) -> i32>(q, 30)(q, 1, std::ptr::null(), std::ptr::null());
                check!(t, hr(r) == 0, "queue slot 30 (PresentX?) without a present hook is accepted with S_OK");
            }
            console_slots(t, dev, alloc, fence_if.as_ref());
            drop(fence_if);
            release_if(list);
            if !alloc.is_null() {
                let left = release(alloc);
                check!(t, left == 0, "the allocator proxy releases to zero ({left})");
            }
            release_if(q);
            let mut dev3: *mut c_void = std::ptr::null_mut();
            let r = match xs_create_pc {
                Some(create) => create(std::ptr::null_mut(), D3D_FEATURE_LEVEL_11_0.0 as u32, &ID3D12Device::IID, &mut dev3),
                None => 0x8000_4005_u32 as i32,
            };
            let same_table = !dev3.is_null() && *(dev3 as *const *const c_void) == *(dev as *const *const c_void);
            check!(t, r >= 0 && same_table, "D3D12CreateDevice returns the console device proxy (hr=0x{:08x})", hr(r));
            release_if(dev3);
            let left = release(dev);
            check!(t, left == 0, "the device proxy releases to zero ({left})");
        }
    }
}
