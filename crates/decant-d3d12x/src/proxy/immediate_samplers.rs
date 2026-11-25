//! The PC side of the immediate sampler registry (`sampler_registry`): writing the samplers into
//! the tail of the sampler heaps, and a sampler heap of the layer for lists the title gives none.

use super::*;

/// Writes one sampler with the PC device's CreateSampler (slot 22).
fn write(device: usize, desc: &[u8; 0x34], cpu: usize) {
    // SAFETY: `device` is the live PC device that made the heap `cpu` points into, CreateSampler
    // takes a D3D12_SAMPLER_DESC and a CPU handle.
    unsafe { method::<CreateSamplerPc>(device as Object, 22)(device as Object, desc.as_ptr().cast(), cpu) };
}

/// The PC device of the first heap made under the direct path, for the layer's own sampler heap.
static DEVICE: AtomicUsize = AtomicUsize::new(0);

/// A sampler heap `h` of device `pc` was made with `host` descriptors for the title's `title`.
pub(super) fn heap_made(pc: Object, h: Object, title: u32, host: u32, increment: u32, cpu: usize) {
    DEVICE.store(pc as usize, Ordering::Relaxed);
    if host != crate::sampler_registry::HEAP_SIZE || title > crate::sampler_registry::HEAP_BASE {
        static SHOWN: AtomicI32 = AtomicI32::new(0);
        if log_due(&SHOWN) {
            xs_log!(
                "direct shaders: sampler heap {} of {title} descriptors has no room for immediate samplers after {} (host {host})",
                p(h.cast()),
                crate::sampler_registry::HEAP_BASE
            );
        }
        crate::sampler_registry::heap_created(h as usize, None, write);
        return;
    }
    crate::sampler_registry::heap_created(h as usize, Some(crate::sampler_registry::Tail { device: pc as usize, cpu_start: cpu, increment }), write);
}

/// The layer's own shader visible sampler heap, made on first need.
static OWN_HEAP: OnceLock<usize> = OnceLock::new();

fn own_heap() -> Option<Object> {
    let heap = *OWN_HEAP.get_or_init(|| {
        let device = DEVICE.load(Ordering::Relaxed) as Object;
        if device.is_null() {
            return 0;
        }
        // D3D12_DESCRIPTOR_HEAP_DESC: sampler type 1, HEAP_SIZE descriptors, shader visible.
        let desc: [u32; 4] = [1, crate::sampler_registry::HEAP_SIZE, 1, 0];
        let mut h: Object = std::ptr::null_mut();
        // SAFETY: `device` is the live PC device, the GUID is ID3D12DescriptorHeap's.
        let hr = unsafe { method::<CreateHeap>(device, 14)(device, desc.as_ptr().cast(), &IID_DESCRIPTOR_HEAP, &mut h) };
        if hr < 0 || h.is_null() {
            xs_log!("direct shaders: the layer's sampler heap was refused, hr=0x{:08X}", hr as u32);
            return 0;
        }
        // SAFETY: `h` is the heap just made.
        unsafe {
            let increment = method::<GetIncSize>(device, 15)(device, 1);
            let mut cpu = 0u64;
            method::<HeapStart>(h, 9)(h, &mut cpu);
            crate::mem_owners::add_charge(crate::mem_owners::Charge {
                kind: crate::mem_owners::Kind::DescriptorHeap,
                segment: crate::mem_owners::Segment::Local,
                bytes: crate::sampler_registry::HEAP_SIZE as u64 * increment as u64,
            });
            heap_made(device, h, 0, crate::sampler_registry::HEAP_SIZE, increment, cpu as usize);
        }
        xs_log!("direct shaders: the layer's sampler heap {} is bound on lists the title gives none", p(h.cast()));
        h as usize
    });
    (heap != 0).then_some(heap as Object)
}

/// `ID3D12DescriptorHeap` (public d3d12.h).
const IID_DESCRIPTOR_HEAP: GUID = GUID { data1: 0x8efb_471d, data2: 0x616c, data3: 0x4f49, data4: [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51] };

/// The heaps to bind for `heaps` (PC objects): under the direct path, a list given view heaps and
/// no sampler heap gets the layer's sampler heap too, so translated shaders find their immediate
/// samplers. Unchanged otherwise.
pub(super) fn with_sampler_heap(heaps: Vec<Object>) -> Vec<Object> {
    if !crate::compile_service::direct_enabled() || heaps.is_empty() || heaps.len() >= 2 {
        return heaps;
    }
    if heaps.iter().any(|h| crate::sampler_registry::is_sampler_heap(*h as usize)) {
        return heaps;
    }
    let mut out = heaps;
    out.extend(own_heap());
    out
}
