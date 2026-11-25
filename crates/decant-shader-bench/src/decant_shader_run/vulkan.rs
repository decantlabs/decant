//! Every Vulkan call of `decant_shader_run`, through `ash`, in the order of the former C++ bench.
//!
//! The objects are never destroyed: the process ends right after the image is written, as the
//! C++ did, and the loader stays loaded until then.
#![allow(unsafe_code)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::{CStr, c_void};
use std::io::Write;
use std::mem::ManuallyDrop;
use std::ptr;

use ash::vk;

use decant_shader_bench::layout::{self, LayoutEntry};
use decant_shader_bench::run_args::RunArgs;
use decant_shader_bench::{Exit, Vertex, cfmt, floats, out, pattern};

const SIZE: u32 = 64;
const FORMAT: vk::Format = vk::Format::R32G32B32A32_SFLOAT;
const TEXTURE_SIZE: u32 = 4;
const STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(
    vk::ShaderStageFlags::VERTEX.as_raw() | vk::ShaderStageFlags::FRAGMENT.as_raw(),
);

#[cfg(windows)]
const LIBRARY: &str = "vulkan-1.dll";
#[cfg(not(windows))]
const LIBRARY: &str = "libvulkan.so.1";

/// The functions the C++ loaded with `vkGetInstanceProcAddr` right after creating the instance.
/// A missing one ended it with exit 3.
const INSTANCE_FUNCTIONS: [&CStr; 42] = [
    c"vkEnumeratePhysicalDevices",
    c"vkGetPhysicalDeviceProperties",
    c"vkGetPhysicalDeviceQueueFamilyProperties",
    c"vkGetPhysicalDeviceMemoryProperties",
    c"vkCreateDevice",
    c"vkGetDeviceQueue",
    c"vkCreateImage",
    c"vkGetImageMemoryRequirements",
    c"vkAllocateMemory",
    c"vkBindImageMemory",
    c"vkCreateImageView",
    c"vkCreateRenderPass",
    c"vkCreateFramebuffer",
    c"vkCreateShaderModule",
    c"vkCreateDescriptorSetLayout",
    c"vkCreatePipelineLayout",
    c"vkCreateGraphicsPipelines",
    c"vkCreateBuffer",
    c"vkGetBufferMemoryRequirements",
    c"vkBindBufferMemory",
    c"vkMapMemory",
    c"vkUnmapMemory",
    c"vkCreateDescriptorPool",
    c"vkAllocateDescriptorSets",
    c"vkUpdateDescriptorSets",
    c"vkCreateCommandPool",
    c"vkAllocateCommandBuffers",
    c"vkBeginCommandBuffer",
    c"vkCmdBeginRenderPass",
    c"vkCmdBindPipeline",
    c"vkCmdBindVertexBuffers",
    c"vkCmdBindDescriptorSets",
    c"vkCmdPushConstants",
    c"vkCmdDraw",
    c"vkCmdEndRenderPass",
    c"vkCmdCopyImageToBuffer",
    c"vkEndCommandBuffer",
    c"vkQueueSubmit",
    c"vkQueueWaitIdle",
    c"vkCreateSampler",
    c"vkCmdCopyBufferToImage",
    c"vkCmdPipelineBarrier",
];

fn check<T>(result: Result<T, vk::Result>, what: &str) -> Result<T, Exit> {
    result.map_err(|e| Exit::new(4, format!("echec {what} : {}", e.as_raw())))
}

fn require(entry: &ash::Entry, instance: vk::Instance, name: &CStr) -> Result<(), Exit> {
    // SAFETY: `instance` is a live instance of this loader and `name` is NUL terminated.
    let function = unsafe { entry.get_instance_proc_addr(instance, name.as_ptr()) };
    match function {
        Some(_) => Ok(()),
        None => Err(Exit::new(
            3,
            format!("fonction Vulkan absente : {}", name.to_string_lossy()),
        )),
    }
}

#[derive(Clone, Copy, Default)]
struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

/// The device and what the helpers need to allocate on it.
struct Gpu {
    device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
}

impl Gpu {
    fn memory_type(&self, bits: u32, wanted: vk::MemoryPropertyFlags) -> Result<u32, Exit> {
        let types = &self.memory_properties.memory_types
            [..self.memory_properties.memory_type_count as usize];
        types
            .iter()
            .enumerate()
            .find(|(i, t)| bits & (1 << i) != 0 && t.property_flags.contains(wanted))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| Exit::new(4, "type de memoire introuvable"))
    }

    /// A host visible, coherent buffer, filled with `data` when given.
    fn host_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        data: Option<&[u8]>,
    ) -> Result<HostBuffer, Exit> {
        let d = &self.device;
        let info = vk::BufferCreateInfo::default().size(size).usage(usage);
        // SAFETY: valid create info, no allocator.
        let buffer = check(unsafe { d.create_buffer(&info, None) }, "vkCreateBuffer")?;
        // SAFETY: `buffer` was just created on this device.
        let requirements = unsafe { d.get_buffer_memory_requirements(buffer) };
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(self.memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )?);
        // SAFETY: the size and type come from the buffer's own requirements.
        let memory = check(
            unsafe { d.allocate_memory(&allocate, None) },
            "vkAllocateMemory",
        )?;
        // SAFETY: fresh memory, large enough for the buffer, bound once at offset 0.
        check(
            unsafe { d.bind_buffer_memory(buffer, memory, 0) },
            "vkBindBufferMemory",
        )?;
        if let Some(data) = data {
            // SAFETY: host visible memory of at least `size` bytes, mapped once and unmapped below.
            let mapped = check(
                unsafe { d.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) },
                "vkMapMemory",
            )?;
            // SAFETY: the mapping covers `size` bytes and the caller gives at least that many.
            unsafe {
                ptr::copy_nonoverlapping(data.as_ptr(), mapped.cast::<u8>(), size as usize);
                d.unmap_memory(memory);
            }
        }
        Ok(HostBuffer { buffer, memory })
    }

    /// A device local image with its memory bound.
    fn image(&self, info: &vk::ImageCreateInfo, what: [&str; 3]) -> Result<vk::Image, Exit> {
        let d = &self.device;
        // SAFETY: valid create info, no allocator.
        let image = check(unsafe { d.create_image(info, None) }, what[0])?;
        // SAFETY: `image` was just created on this device.
        let requirements = unsafe { d.get_image_memory_requirements(image) };
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(self.memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?);
        // SAFETY: the size and type come from the image's own requirements.
        let memory = check(unsafe { d.allocate_memory(&allocate, None) }, what[1])?;
        // SAFETY: fresh memory, large enough for the image, bound once at offset 0.
        check(unsafe { d.bind_image_memory(image, memory, 0) }, what[2])?;
        Ok(image)
    }

    fn view(
        &self,
        image: vk::Image,
        kind: vk::ImageViewType,
        format: vk::Format,
        layers: u32,
        what: &str,
    ) -> Result<vk::ImageView, Exit> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(kind)
            .format(format)
            .subresource_range(color_range(layers));
        // SAFETY: `image` is a live image of this device with at least `layers` layers.
        check(unsafe { self.device.create_image_view(&info, None) }, what)
    }

    /// Nearest neighbour, clamped to the edges.
    fn sampler(&self, what: &str) -> Result<vk::Sampler, Exit> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: valid create info, no allocator.
        check(unsafe { self.device.create_sampler(&info, None) }, what)
    }

    fn write(&self, write: vk::WriteDescriptorSet) {
        // SAFETY: the write names a live set and points at infos that outlive the call.
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }
}

fn color_range(layers: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: layers,
    }
}

fn image_2d(
    format: vk::Format,
    extent: vk::Extent3D,
    layers: u32,
    usage: vk::ImageUsageFlags,
) -> vk::ImageCreateInfo<'static> {
    vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(usage)
}

fn barrier(image: vk::Image, range: vk::ImageSubresourceRange) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(range)
}

/// An image of `--layout` waiting for its texels.
struct Upload {
    staging: HostBuffer,
    image: vk::Image,
    extent: vk::Extent3D,
    layers: u32,
}

/// The resources of `--layout`, made once per kind and shape.
struct LayoutResources<'a> {
    gpu: &'a Gpu,
    run: &'a RunArgs,
    views: HashMap<String, vk::ImageView>,
    texture2d: vk::Image,
    texture3d: vk::Image,
    uploads: Vec<Upload>,
    storage_images: Vec<vk::Image>,
}

impl LayoutResources<'_> {
    fn layout_image(
        &self,
        kind: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
        layers: u32,
        usage: vk::ImageUsageFlags,
        flags: vk::ImageCreateFlags,
    ) -> Result<vk::Image, Exit> {
        let info = image_2d(format, extent, layers, usage)
            .image_type(kind)
            .flags(flags);
        self.gpu.image(
            &info,
            [
                "vkCreateImage(layout)",
                "vkAllocateMemory(layout)",
                "vkBindImageMemory(layout)",
            ],
        )
    }

    fn layout_view(
        &self,
        image: vk::Image,
        kind: vk::ImageViewType,
        format: vk::Format,
        layers: u32,
    ) -> Result<vk::ImageView, Exit> {
        self.gpu
            .view(image, kind, format, layers, "vkCreateImageView(layout)")
    }

    fn upload(
        &mut self,
        data: &[u8],
        image: vk::Image,
        extent: vk::Extent3D,
        layers: u32,
    ) -> Result<(), Exit> {
        let staging = self.gpu.host_buffer(
            data.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some(data),
        )?;
        self.uploads.push(Upload {
            staging,
            image,
            extent,
            layers,
        });
        Ok(())
    }

    /// Sampled images: the 4 x 4 RGBA8 pattern, 6 layers (2D, 2D array, cube) and 4 x 4 x 4 in 3D,
    /// or the `--texture-file` image for a 2D sampler that is not an array.
    fn sampled_view(&mut self, e: &LayoutEntry) -> Result<vk::ImageView, Exit> {
        let key = e.view_key();
        if let Some(&view) = self.views.get(&key) {
            return Ok(view);
        }
        let format = vk::Format::R8G8B8A8_UNORM;
        let sampled = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let view = if e.dim == "3d" {
            if self.texture3d == vk::Image::null() {
                let extent = vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 4,
                };
                self.texture3d = self.layout_image(
                    vk::ImageType::TYPE_3D,
                    format,
                    extent,
                    1,
                    sampled,
                    vk::ImageCreateFlags::empty(),
                )?;
                self.upload(&pattern(64), self.texture3d, extent, 1)?;
            }
            self.layout_view(self.texture3d, vk::ImageViewType::TYPE_3D, format, 1)?
        } else if !self.run.real_texture.is_empty() && e.dim == "2d" && e.arrayed == 0 {
            let (w, h) = (self.run.real_w, self.run.real_h);
            let size = u64::from(w) * u64::from(h) * 4;
            if (self.run.real_texture.len() as u64) < size {
                return Err(Exit::new(4, "image trop petite"));
            }
            let extent = vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            };
            let real = self.layout_image(
                vk::ImageType::TYPE_2D,
                format,
                extent,
                1,
                sampled,
                vk::ImageCreateFlags::empty(),
            )?;
            let texels = &self.run.real_texture[..size as usize];
            self.upload(texels, real, extent, 1)?;
            self.layout_view(real, vk::ImageViewType::TYPE_2D, format, 1)?
        } else {
            if self.texture2d == vk::Image::null() {
                let extent = vk::Extent3D {
                    width: 4,
                    height: 4,
                    depth: 1,
                };
                self.texture2d = self.layout_image(
                    vk::ImageType::TYPE_2D,
                    format,
                    extent,
                    6,
                    sampled,
                    vk::ImageCreateFlags::CUBE_COMPATIBLE,
                )?;
                self.upload(&pattern(16 * 6), self.texture2d, extent, 6)?;
            }
            let arrayed = e.arrayed != 0;
            if e.dim == "cube" {
                let kind = if arrayed {
                    vk::ImageViewType::CUBE_ARRAY
                } else {
                    vk::ImageViewType::CUBE
                };
                self.layout_view(self.texture2d, kind, format, 6)?
            } else {
                let kind = if arrayed {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                };
                self.layout_view(self.texture2d, kind, format, if arrayed { 6 } else { 1 })?
            }
        };
        self.views.insert(key, view);
        Ok(view)
    }

    /// Storage images: 16 x 16 (2 layers) or 8 x 8 x 8, in the requested format.
    fn storage_view(&mut self, e: &LayoutEntry) -> Result<vk::ImageView, Exit> {
        let format = layout::storage_format(e.format);
        let key = e.storage_key();
        if let Some(&view) = self.views.get(&key) {
            return Ok(view);
        }
        let usage = vk::ImageUsageFlags::STORAGE;
        let view = if e.dim == "3d" {
            let extent = vk::Extent3D {
                width: 8,
                height: 8,
                depth: 8,
            };
            let image = self.layout_image(
                vk::ImageType::TYPE_3D,
                format,
                extent,
                1,
                usage,
                vk::ImageCreateFlags::empty(),
            )?;
            self.storage_images.push(image);
            self.layout_view(image, vk::ImageViewType::TYPE_3D, format, 1)?
        } else {
            let extent = vk::Extent3D {
                width: 16,
                height: 16,
                depth: 1,
            };
            let image = self.layout_image(
                vk::ImageType::TYPE_2D,
                format,
                extent,
                2,
                usage,
                vk::ImageCreateFlags::empty(),
            )?;
            self.storage_images.push(image);
            let arrayed = e.arrayed != 0;
            let kind = if arrayed {
                vk::ImageViewType::TYPE_2D_ARRAY
            } else {
                vk::ImageViewType::TYPE_2D
            };
            self.layout_view(image, kind, format, if arrayed { 2 } else { 1 })?
        };
        self.views.insert(key, view);
        Ok(view)
    }
}

/// Builds the pipeline, draws the triangle, writes the image and prints the two pixels.
/// Returns the exit code (0, or 2 when the image could not be written).
pub fn render(
    run: &RunArgs,
    layout_entries: &[LayoutEntry],
    vs_code: &[u8],
    ps_code: &[u8],
) -> Result<i32, Exit> {
    // SAFETY: loading the system Vulkan loader runs its initialisation, which is what the bench is for.
    // It is never unloaded (ManuallyDrop): the device must not outlive it.
    let entry = ManuallyDrop::new(
        unsafe { ash::Entry::load_from(LIBRARY) }
            .map_err(|_| Exit::new(3, format!("{LIBRARY} introuvable")))?,
    );
    let app = vk::ApplicationInfo::default()
        .application_name(c"decant_shader_run")
        .api_version(vk::API_VERSION_1_3);
    let instance_info = vk::InstanceCreateInfo::default().application_info(&app);
    // SAFETY: valid create info, no allocator.
    let instance = check(
        unsafe { entry.create_instance(&instance_info, None) },
        "vkCreateInstance",
    )?;
    for name in INSTANCE_FUNCTIONS {
        require(&entry, instance.handle(), name)?;
    }

    // SAFETY: `instance` is live, the call returns every physical device.
    let devices = check(
        unsafe { instance.enumerate_physical_devices() },
        "vkEnumeratePhysicalDevices",
    )?;
    let mut chosen: Option<(vk::PhysicalDevice, vk::PhysicalDeviceProperties)> = None;
    for &candidate in &devices {
        // SAFETY: `candidate` comes from this instance.
        let current = unsafe { instance.get_physical_device_properties(candidate) };
        if chosen.is_none() || current.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            chosen = Some((candidate, current));
        }
    }
    let Some((gpu, properties)) = chosen else {
        return Err(Exit::new(3, "aucune carte Vulkan"));
    };
    let name = properties
        .device_name_as_c_str()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    out(&format!("carte : {name}\n"));

    // SAFETY: `gpu` comes from this instance.
    let families = unsafe { instance.get_physical_device_queue_family_properties(gpu) };
    let family = families
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .unwrap_or(families.len()) as u32;

    let priority = [1.0f32];
    let queue_info = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priority)];
    // Every feature the card offers (Vulkan 1.0 to 1.3, barycentrics): depending on the module,
    // translated shaders ask for 64-bit integers, half floats, subgroup operations, descriptor
    // indexing and so on.
    require(&entry, instance.handle(), c"vkGetPhysicalDeviceFeatures2")?;
    require(
        &entry,
        instance.handle(),
        c"vkEnumerateDeviceExtensionProperties",
    )?;
    let mut barycentric = vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR::default();
    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features11 = vk::PhysicalDeviceVulkan11Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default();
    // The chain is reached through these pointers only, from here on.
    let (p_bary, p13, p12, p11, pf) = (
        &raw mut barycentric,
        &raw mut features13,
        &raw mut features12,
        &raw mut features11,
        &raw mut features,
    );
    // Review note: only the structures of the version the card announces are chained.
    let vk12 = properties.api_version >= vk::API_VERSION_1_2;
    let vk13 = properties.api_version >= vk::API_VERSION_1_3;
    let mut extensions: Vec<*const std::ffi::c_char> = Vec::new();
    // SAFETY: the five structures live until the device is created below, each `p_next` points at
    // the next one or is null, and the tail pointer names the last `p_next` of the chain.
    unsafe {
        (*pf).p_next = p11.cast::<c_void>();
        (*p11).p_next = if vk12 {
            p12.cast::<c_void>()
        } else {
            ptr::null_mut()
        };
        (*p12).p_next = if vk13 {
            p13.cast::<c_void>()
        } else {
            ptr::null_mut()
        };
        let chain_tail: *mut *mut c_void = if vk13 {
            &raw mut (*p13).p_next
        } else if vk12 {
            &raw mut (*p12).p_next
        } else {
            &raw mut (*p11).p_next
        };
        let available = instance
            .enumerate_device_extension_properties(gpu)
            .unwrap_or_default();
        for extension in &available {
            if extension.extension_name_as_c_str()
                == Ok(ash::khr::fragment_shader_barycentric::NAME)
            {
                extensions.push(ash::khr::fragment_shader_barycentric::NAME.as_ptr());
                *chain_tail = p_bary.cast::<c_void>();
            }
        }
        instance.get_physical_device_features2(gpu, &mut *pf);
        // Translated vertex shaders declare gl_ClipDistance.
        (*pf).features.shader_clip_distance = vk::TRUE;
        if !run.robust {
            // Without robustness, an out of bounds read is no longer replaced by zero: it may fault.
            (*pf).features.robust_buffer_access = vk::FALSE;
            (*p13).robust_image_access = vk::FALSE;
        }
    }
    out(&format!(
        "robustesse : {}\n",
        if run.robust { "oui" } else { "non" }
    ));
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queue_info)
        .enabled_extension_names(&extensions);
    device_info.p_next = pf.cast_const().cast::<c_void>();
    // SAFETY: the create info and its feature chain are valid and alive for the call.
    let device = check(
        unsafe { instance.create_device(gpu, &device_info, None) },
        "vkCreateDevice",
    )?;
    // SAFETY: queue 0 of the family the device was created with.
    let queue = unsafe { device.get_device_queue(family, 0) };
    // SAFETY: `gpu` comes from this instance.
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(gpu) };
    let g = Gpu {
        device,
        memory_properties,
    };
    let d = &g.device;

    let extent = vk::Extent3D {
        width: SIZE,
        height: SIZE,
        depth: 1,
    };
    let image_info = image_2d(
        FORMAT,
        extent,
        1,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let image = g.image(
        &image_info,
        [
            "vkCreateImage",
            "vkAllocateMemory(image)",
            "vkBindImageMemory",
        ],
    )?;
    let view = g.view(
        image,
        vk::ImageViewType::TYPE_2D,
        FORMAT,
        1,
        "vkCreateImageView",
    )?;

    let attachment = [vk::AttachmentDescription::default()
        .format(FORMAT)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
    let reference = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpass = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&reference)];
    // The copy to the readback buffer must wait for the colour writes (and the transition to
    // finalLayout): the implicit external dependency orders nothing for a copy after the pass.
    let to_copy = [vk::SubpassDependency::default()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
    let pass_info = vk::RenderPassCreateInfo::default()
        .attachments(&attachment)
        .subpasses(&subpass)
        .dependencies(&to_copy);
    // SAFETY: valid create info, no allocator.
    let pass = check(
        unsafe { d.create_render_pass(&pass_info, None) },
        "vkCreateRenderPass",
    )?;
    let views = [view];
    let framebuffer_info = vk::FramebufferCreateInfo::default()
        .render_pass(pass)
        .attachments(&views)
        .width(SIZE)
        .height(SIZE)
        .layers(1);
    // SAFETY: the pass and view are live and compatible.
    let framebuffer = check(
        unsafe { d.create_framebuffer(&framebuffer_info, None) },
        "vkCreateFramebuffer",
    )?;

    // The SPIR-V is given as read, size in bytes included, as the C++ did: no check, no padding
    // seen by the driver (the copy into words only aligns it).
    let module = |code: &[u8], what: &str| -> Result<vk::ShaderModule, Exit> {
        let mut words = vec![0u32; code.len().div_ceil(4)];
        // SAFETY: `words` holds at least `code.len()` bytes.
        unsafe {
            ptr::copy_nonoverlapping(code.as_ptr(), words.as_mut_ptr().cast::<u8>(), code.len())
        };
        let info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: words.as_ptr(),
            ..Default::default()
        };
        // SAFETY: `p_code` points at `code_size` readable bytes, aligned to 4.
        check(unsafe { d.create_shader_module(&info, None) }, what)
    };
    let vs = module(vs_code, "vkCreateShaderModule(vs)")?;
    let ps = module(ps_code, "vkCreateShaderModule(ps)")?;

    let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = run
        .bound
        .iter()
        .map(|entry| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(entry.binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(STAGES)
        })
        .collect();
    if run.texture_binding >= 0 {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(run.texture_binding as u32)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);
        bindings.push(binding);
        bindings.push(
            binding
                .binding(run.sampler_binding as u32)
                .descriptor_type(vk::DescriptorType::SAMPLER),
        );
    }
    // Descriptor sets: set 0 receives --buffer and --texture, --layout may add others.
    let set_count = layout_entries.iter().map(|e| e.set + 1).fold(1, u32::max);
    let mut set_bindings: Vec<Vec<vk::DescriptorSetLayoutBinding>> =
        vec![Vec::new(); set_count as usize];
    set_bindings[0] = bindings;
    for e in layout_entries {
        set_bindings[e.set as usize].push(
            vk::DescriptorSetLayoutBinding::default()
                .binding(e.binding)
                .descriptor_type(layout::descriptor_type(e)?)
                .descriptor_count(e.count)
                .stage_flags(STAGES),
        );
    }
    let mut set_layouts = Vec::with_capacity(set_count as usize);
    for list in &set_bindings {
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(list);
        // SAFETY: valid create info, no allocator.
        set_layouts.push(check(
            unsafe { d.create_descriptor_set_layout(&info, None) },
            "vkCreateDescriptorSetLayout",
        )?);
    }
    // Push constants: 32 words (128 bytes), filled by --push.
    let push = [vk::PushConstantRange {
        stage_flags: STAGES,
        offset: 0,
        size: 128,
    }];
    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push);
    // SAFETY: valid create info with live set layouts.
    let layout = check(
        unsafe { d.create_pipeline_layout(&layout_info, None) },
        "vkCreatePipelineLayout",
    )?;

    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vs)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(ps)
            .name(c"main"),
    ];
    let vertex_binding = [vk::VertexInputBindingDescription {
        binding: 0,
        stride: Vertex::SIZE,
        input_rate: vk::VertexInputRate::VERTEX,
    }];
    let attributes = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: Vertex::POSITION,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32A32_SFLOAT,
            offset: Vertex::COLOR,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: Vertex::UV,
        },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&vertex_binding)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: SIZE as f32,
        height: SIZE as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: SIZE,
            height: SIZE,
        },
    };
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewport)
        .scissors(&scissors);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachment);
    let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .color_blend_state(&blend)
        .layout(layout)
        .render_pass(pass)];
    // --pipeline-cache: seed a VkPipelineCache from the file (when it exists), pass it to the
    // creation, then write it back. The driver really compiles the pipeline (a stronger check than
    // spirv-val alone) and the cache is kept between runs.
    let pipeline_cache = match &run.pipeline_cache_path {
        Some(path) => {
            let seed = std::fs::read(path).unwrap_or_default();
            let cache_info = vk::PipelineCacheCreateInfo::default().initial_data(&seed);
            // SAFETY: `seed` outlives the call.
            check(unsafe { d.create_pipeline_cache(&cache_info, None) }, "vkCreatePipelineCache")?
        }
        None => vk::PipelineCache::null(),
    };
    // SAFETY: every state the create info points at is alive for the call.
    let pipelines = unsafe { d.create_graphics_pipelines(pipeline_cache, &pipeline_info, None) };
    let pipeline = check(pipelines.map_err(|(_, e)| e), "vkCreateGraphicsPipelines")?[0];
    if pipeline_cache != vk::PipelineCache::null() {
        // SAFETY: the cache was created on this device and is no longer used by any creation.
        if let (Ok(data), Some(path)) =
            (unsafe { d.get_pipeline_cache_data(pipeline_cache) }, &run.pipeline_cache_path)
            && !data.is_empty()
        {
            let _ = std::fs::write(path, &data);
        }
        // SAFETY: as above.
        unsafe { d.destroy_pipeline_cache(pipeline_cache, None) };
    }

    // Triangle covering the whole screen, a different colour and coordinate per vertex.
    let vertices = [
        Vertex {
            position: [-1.0, -1.0, 0.0],
            color: [1.00, 0.00, 0.25, 1.00],
            uv: [0.0, 0.0],
        },
        Vertex {
            position: [3.0, -1.0, 0.0],
            color: [0.00, 1.00, 0.50, 0.50],
            uv: [2.0, 0.0],
        },
        Vertex {
            position: [-1.0, 3.0, 0.0],
            color: [0.50, 0.25, 1.00, 0.75],
            uv: [0.0, 2.0],
        },
    ];
    let vertex_bytes = Vertex::bytes(&vertices);
    let vertex_buffer = g.host_buffer(
        vertex_bytes.len() as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        Some(&vertex_bytes),
    )?;

    // Bench texture: 4 x 4 RGBA8 texels, all different, read nearest and clamped to the edges.
    let mut texture = vk::Image::null();
    let mut texture_view = vk::ImageView::null();
    let mut sampler = vk::Sampler::null();
    let mut staging = HostBuffer::default();
    if run.texture_binding >= 0 {
        let texels = pattern(TEXTURE_SIZE * TEXTURE_SIZE);
        staging = g.host_buffer(
            texels.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            Some(&texels),
        )?;
        let info = image_2d(
            vk::Format::R8G8B8A8_UNORM,
            vk::Extent3D {
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
                depth: 1,
            },
            1,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        );
        texture = g.image(
            &info,
            [
                "vkCreateImage(texture)",
                "vkAllocateMemory(texture)",
                "vkBindImageMemory(texture)",
            ],
        )?;
        texture_view = g.view(
            texture,
            vk::ImageViewType::TYPE_2D,
            vk::Format::R8G8B8A8_UNORM,
            1,
            "vkCreateImageView(texture)",
        )?;
        sampler = g.sampler("vkCreateSampler")?;
    }

    let mut sets: Vec<vk::DescriptorSet> = Vec::new();
    if set_bindings.iter().any(|list| !list.is_empty()) {
        let mut totals: BTreeMap<i32, u32> = BTreeMap::new();
        for b in set_bindings.iter().flatten() {
            *totals.entry(b.descriptor_type.as_raw()).or_default() += b.descriptor_count;
        }
        let pool_sizes: Vec<vk::DescriptorPoolSize> = totals
            .iter()
            .map(|(&ty, &count)| vk::DescriptorPoolSize {
                ty: vk::DescriptorType::from_raw(ty),
                descriptor_count: count,
            })
            .collect();
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(set_count)
            .pool_sizes(&pool_sizes);
        // SAFETY: valid create info, no allocator.
        let pool = check(
            unsafe { d.create_descriptor_pool(&pool_info, None) },
            "vkCreateDescriptorPool",
        )?;
        let allocate = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&set_layouts);
        // SAFETY: the pool was sized for these layouts.
        sets = check(
            unsafe { d.allocate_descriptor_sets(&allocate) },
            "vkAllocateDescriptorSets",
        )?;
        let set = sets[0];
        for entry in &run.bound {
            let buffer = g.host_buffer(
                entry.data.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                Some(&entry.data),
            )?;
            let info = [vk::DescriptorBufferInfo {
                buffer: buffer.buffer,
                offset: 0,
                range: vk::WHOLE_SIZE,
            }];
            g.write(
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(entry.binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&info),
            );
        }
        if run.texture_binding >= 0 {
            let image_descriptor = [vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: texture_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            }];
            let sampler_descriptor = [vk::DescriptorImageInfo {
                sampler,
                image_view: vk::ImageView::null(),
                image_layout: vk::ImageLayout::UNDEFINED,
            }];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(run.texture_binding as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&image_descriptor),
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(run.sampler_binding as u32)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&sampler_descriptor),
            ];
            // SAFETY: both writes name the live set and infos that outlive the call.
            unsafe { d.update_descriptor_sets(&writes, &[]) };
        }
    }

    // Resources of --layout, made once per kind and bound to every element of the arrays:
    //   buffers: 64 KiB of zeros (storage and uniform), sampled images: the 4 x 4 RGBA8 pattern,
    //   storage images in the requested format, sampler: nearest, clamped to the edges.
    let mut resources = LayoutResources {
        gpu: &g,
        run,
        views: HashMap::new(),
        texture2d: vk::Image::null(),
        texture3d: vk::Image::null(),
        uploads: Vec::new(),
        storage_images: Vec::new(),
    };
    if !layout_entries.is_empty() {
        let zeros = vec![0u8; 65536];
        let buffer = g.host_buffer(
            zeros.len() as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
            Some(&zeros),
        )?;
        // Memory read by address (bda_pagetable): every 16 KiB page of the console address space
        // points to one page of zeros (32 KiB, a read may spill into the next). 1 M entries cover
        // 16 GiB, beyond, robustness reads 0 ("missing page") and the fault buffer drops the write.
        let mut pagetable = HostBuffer::default();
        let mut faults = HostBuffer::default();
        if layout_entries.iter().any(|e| e.kind == "bda_pagetable") {
            require(&entry, instance.handle(), c"vkGetBufferDeviceAddress")?;
            let info = vk::BufferCreateInfo::default().size(32768).usage(
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            );
            // SAFETY: valid create info, no allocator.
            let zero_buffer = check(
                unsafe { d.create_buffer(&info, None) },
                "vkCreateBuffer(zero page)",
            )?;
            // SAFETY: `zero_buffer` was just created on this device.
            let requirements = unsafe { d.get_buffer_memory_requirements(zero_buffer) };
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let allocate = vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(g.memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )?)
                .push_next(&mut flags);
            // SAFETY: the size and type come from the buffer's own requirements.
            let zero_memory = check(
                unsafe { d.allocate_memory(&allocate, None) },
                "vkAllocateMemory(zero page)",
            )?;
            // SAFETY: fresh memory bound once at offset 0.
            check(
                unsafe { d.bind_buffer_memory(zero_buffer, zero_memory, 0) },
                "vkBindBufferMemory(zero page)",
            )?;
            // SAFETY: host visible memory of at least 32 KiB, mapped once and unmapped below.
            let mapped = check(
                unsafe { d.map_memory(zero_memory, 0, info.size, vk::MemoryMapFlags::empty()) },
                "vkMapMemory(zero page)",
            )?;
            // SAFETY: the mapping covers `info.size` bytes.
            unsafe {
                ptr::write_bytes(mapped.cast::<u8>(), 0, info.size as usize);
                d.unmap_memory(zero_memory);
            }
            let address = vk::BufferDeviceAddressInfo::default().buffer(zero_buffer);
            // SAFETY: the buffer has the device address usage and memory allocated with the flag.
            let base = unsafe { d.get_buffer_device_address(&address) };
            let entries = 1usize << 20;
            let table: Vec<u8> = std::iter::repeat_n(base.to_ne_bytes(), entries)
                .flatten()
                .collect();
            pagetable = g.host_buffer(
                table.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                Some(&table),
            )?;
            let fault_bits = vec![0u8; 1 << 20];
            faults = g.host_buffer(
                fault_bits.len() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                Some(&fault_bits),
            )?;
            out(&format!(
                "table de pages : {entries} entrees -> page de zeros 0x{base:x}\n"
            ));
        }
        let layout_sampler = g.sampler("vkCreateSampler(layout)")?;
        for e in layout_entries {
            let ty = layout::descriptor_type(e)?;
            let write = vk::WriteDescriptorSet::default()
                .dst_set(sets[e.set as usize])
                .dst_binding(e.binding)
                .descriptor_type(ty);
            let n = e.count as usize;
            if ty == vk::DescriptorType::STORAGE_BUFFER || ty == vk::DescriptorType::UNIFORM_BUFFER
            {
                let chosen = if e.kind == "bda_pagetable" {
                    pagetable.buffer
                } else if e.kind == "fault_buffer" && faults.buffer != vk::Buffer::null() {
                    faults.buffer
                } else {
                    buffer.buffer
                };
                let infos = vec![
                    vk::DescriptorBufferInfo {
                        buffer: chosen,
                        offset: 0,
                        range: vk::WHOLE_SIZE
                    };
                    n
                ];
                g.write(write.buffer_info(&infos));
            } else if ty == vk::DescriptorType::SAMPLER {
                let infos = vec![
                    vk::DescriptorImageInfo {
                        sampler: layout_sampler,
                        image_view: vk::ImageView::null(),
                        image_layout: vk::ImageLayout::UNDEFINED
                    };
                    n
                ];
                g.write(write.image_info(&infos));
            } else if ty == vk::DescriptorType::STORAGE_IMAGE {
                let view = resources.storage_view(e)?;
                let infos = vec![
                    vk::DescriptorImageInfo {
                        sampler: vk::Sampler::null(),
                        image_view: view,
                        image_layout: vk::ImageLayout::GENERAL
                    };
                    n
                ];
                g.write(write.image_info(&infos));
            } else {
                let view = resources.sampled_view(e)?;
                let infos = vec![
                    vk::DescriptorImageInfo {
                        sampler: layout_sampler,
                        image_view: view,
                        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                    };
                    n
                ];
                g.write(write.image_info(&infos));
            }
        }
    }

    let pixels_size = u64::from(SIZE * SIZE * 4) * 4;
    let readback = g.host_buffer(pixels_size, vk::BufferUsageFlags::TRANSFER_DST, None)?;

    let pool_info = vk::CommandPoolCreateInfo::default().queue_family_index(family);
    // SAFETY: valid create info, no allocator.
    let command_pool = check(
        unsafe { d.create_command_pool(&pool_info, None) },
        "vkCreateCommandPool",
    )?;
    let command_allocate = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    // SAFETY: the pool is live.
    let commands = check(
        unsafe { d.allocate_command_buffers(&command_allocate) },
        "vkAllocateCommandBuffers",
    )?[0];
    let begin = vk::CommandBufferBeginInfo::default();
    // SAFETY: a fresh primary command buffer.
    check(
        unsafe { d.begin_command_buffer(commands, &begin) },
        "vkBeginCommandBuffer",
    )?;

    let pipeline_barrier = |src: vk::PipelineStageFlags,
                            dst: vk::PipelineStageFlags,
                            barrier: vk::ImageMemoryBarrier| {
        // SAFETY: recording into the command buffer begun above, the barrier names a live image.
        unsafe {
            d.cmd_pipeline_barrier(
                commands,
                src,
                dst,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
    };
    let copy_to_image =
        |source: vk::Buffer, image: vk::Image, extent: vk::Extent3D, layers: u32| {
            let copy = vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: layers,
                })
                .image_extent(extent);
            // SAFETY: recording, the staging buffer holds the texels of the whole extent and layers.
            unsafe {
                d.cmd_copy_buffer_to_image(
                    commands,
                    source,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[copy],
                )
            };
        };
    let upload_image = |source: vk::Buffer,
                        image: vk::Image,
                        extent: vk::Extent3D,
                        layers: u32,
                        readers: vk::PipelineStageFlags| {
        let to_copy = barrier(image, color_range(layers))
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        pipeline_barrier(
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            to_copy,
        );
        copy_to_image(source, image, extent, layers);
        let to_read = to_copy
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        pipeline_barrier(vk::PipelineStageFlags::TRANSFER, readers, to_read);
    };
    if texture != vk::Image::null() {
        let extent = vk::Extent3D {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth: 1,
        };
        upload_image(
            staging.buffer,
            texture,
            extent,
            1,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
    }
    let shaders = vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER;
    for upload in &resources.uploads {
        upload_image(
            upload.staging.buffer,
            upload.image,
            upload.extent,
            upload.layers,
            shaders,
        );
    }
    for &storage in &resources.storage_images {
        let general = barrier(storage, color_range(vk::REMAINING_ARRAY_LAYERS))
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::GENERAL);
        pipeline_barrier(vk::PipelineStageFlags::TOP_OF_PIPE, shaders, general);
    }
    // Impossible value: a pixel that was not drawn shows.
    let clear = [vk::ClearValue {
        color: vk::ClearColorValue { float32: [-1.0; 4] },
    }];
    let pass_begin = vk::RenderPassBeginInfo::default()
        .render_pass(pass)
        .framebuffer(framebuffer)
        .render_area(scissor)
        .clear_values(&clear);
    let push_bytes: Vec<u8> = run.user_data.iter().flat_map(|w| w.to_ne_bytes()).collect();
    let region = vk::BufferImageCopy::default()
        .image_subresource(vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        })
        .image_extent(extent);
    // Makes the copy visible to the host read (coherent memory: no invalidation to do).
    let to_host = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .buffer(readback.buffer)
        .size(vk::WHOLE_SIZE);
    // SAFETY: recording into the command buffer begun above, with objects of this device that
    // stay alive until the queue is idle.
    unsafe {
        d.cmd_begin_render_pass(commands, &pass_begin, vk::SubpassContents::INLINE);
        d.cmd_bind_pipeline(commands, vk::PipelineBindPoint::GRAPHICS, pipeline);
        d.cmd_bind_vertex_buffers(commands, 0, &[vertex_buffer.buffer], &[0]);
        if !sets.is_empty() {
            d.cmd_bind_descriptor_sets(
                commands,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &sets,
                &[],
            );
        }
        d.cmd_push_constants(commands, layout, STAGES, 0, &push_bytes);
        d.cmd_draw(commands, 3, 1, 0, 0);
        d.cmd_end_render_pass(commands);
        d.cmd_copy_image_to_buffer(
            commands,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.buffer,
            &[region],
        );
        d.cmd_pipeline_barrier(
            commands,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[to_host],
            &[],
        );
    }
    // SAFETY: the command buffer is recording.
    check(
        unsafe { d.end_command_buffer(commands) },
        "vkEndCommandBuffer",
    )?;
    let command_buffers = [commands];
    let submit = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
    // SAFETY: a complete command buffer on the queue of its pool's family, no fence.
    check(
        unsafe { d.queue_submit(queue, &submit, vk::Fence::null()) },
        "vkQueueSubmit",
    )?;
    // SAFETY: the queue is live.
    check(unsafe { d.queue_wait_idle(queue) }, "vkQueueWaitIdle")?;

    // SAFETY: host visible readback memory of `pixels_size` bytes, written by the finished copy.
    let mapped = check(
        unsafe { d.map_memory(readback.memory, 0, pixels_size, vk::MemoryMapFlags::empty()) },
        "vkMapMemory(lecture)",
    )?;
    // SAFETY: the mapping covers `pixels_size` bytes, 4 byte aligned, and stays mapped while read.
    let bytes = unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), pixels_size as usize) };
    let written = std::fs::File::create(&run.output)
        .and_then(|mut file| file.write_all(bytes))
        .is_ok();
    let pixel = |x: usize, y: usize| -> Vec<f32> {
        let at = (y * SIZE as usize + x) * 16;
        floats(&bytes[at..at + 16])
    };
    out(&format!(
        "pixel (0,0)   : {}\n",
        cfmt::fixed4_rgba(&pixel(0, 0))
    ));
    out(&format!(
        "pixel (32,32) : {}\n",
        cfmt::fixed4_rgba(&pixel(32, 32))
    ));
    // SAFETY: the memory is mapped, `bytes` is not used past this point.
    unsafe { d.unmap_memory(readback.memory) };
    Ok(if written { 0 } else { 2 })
}
