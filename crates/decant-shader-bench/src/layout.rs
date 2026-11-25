//! The `--layout` file of `decant_shader_run`, written by `decant_shader_gallery.py` from the SPIR-V module:
//! one line per resource,
//! `<set> <binding> <kind> <count> <dim> <arrayed 0|1> <texel type> <SPIR-V format> <ms> <depth>`.
//! The bench makes one resource of each kind (nothing comes from the game) and binds it to every
//! element. Exit code 5 marks a resource the bench cannot make (the shader counts as skipped).

use ash::vk;

use crate::Exit;
use crate::cfmt::Fields;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LayoutEntry {
    pub set: u32,
    pub binding: u32,
    pub count: u32,
    pub kind: String,
    pub dim: String,
    pub stype: String,
    pub arrayed: i32,
    pub format: i32,
    pub ms: i32,
    pub depth: i32,
}

impl LayoutEntry {
    pub fn is_image(&self) -> bool {
        matches!(
            self.kind.as_str(),
            "sampled_image" | "combined_image" | "storage_image"
        )
    }

    /// Key of the sampled view shared by every entry of the same shape.
    pub fn view_key(&self) -> String {
        format!("{}{}", self.dim, if self.arrayed != 0 { "[]" } else { "" })
    }

    /// Key of the storage view shared by every entry of the same shape and format.
    pub fn storage_key(&self) -> String {
        format!(
            "storage {}{}{}",
            self.dim,
            if self.arrayed != 0 { "[]" } else { "" },
            self.format
        )
    }
}

/// Parses the entries as the C++ `while (in >> ...)` loop did: it stops at the first line that
/// does not read completely. A count of 0 (array of unknown size) becomes 64.
pub fn parse(bytes: &[u8]) -> Vec<LayoutEntry> {
    let mut fields = Fields::new(bytes);
    let mut entries = Vec::new();
    let mut next = || -> Option<LayoutEntry> {
        let mut e = LayoutEntry {
            set: fields.unsigned()?,
            binding: fields.unsigned()?,
            kind: fields.word()?,
            count: fields.unsigned()?,
            dim: fields.word()?,
            arrayed: fields.signed()?,
            stype: fields.word()?,
            format: fields.signed()?,
            ms: fields.signed()?,
            depth: fields.signed()?,
        };
        if e.count == 0 {
            e.count = 64;
        }
        Some(e)
    };
    while let Some(e) = next() {
        entries.push(e);
    }
    entries
}

/// Vulkan format of a storage image for its SPIR-V image format, `UNDEFINED` when the bench
/// does not make it.
pub fn storage_format(spirv: i32) -> vk::Format {
    match spirv {
        0 | 1 => vk::Format::R32G32B32A32_SFLOAT,
        2 => vk::Format::R16G16B16A16_SFLOAT,
        3 => vk::Format::R32_SFLOAT,
        4 => vk::Format::R8G8B8A8_UNORM,
        24 => vk::Format::R32_SINT,
        33 => vk::Format::R32_UINT,
        _ => vk::Format::UNDEFINED,
    }
}

fn unsupported(e: &LayoutEntry, why: &str) -> Exit {
    Exit::new(
        5,
        format!(
            "non gere : {} au jeu {} binding {} ({why})",
            e.kind, e.set, e.binding
        ),
    )
}

/// Descriptor type of an entry, or the exit 5 of an unknown kind.
pub fn descriptor_type(e: &LayoutEntry) -> Result<vk::DescriptorType, Exit> {
    Ok(match e.kind.as_str() {
        "storage_buffer" | "bda_pagetable" | "fault_buffer" => vk::DescriptorType::STORAGE_BUFFER,
        "uniform_buffer" => vk::DescriptorType::UNIFORM_BUFFER,
        "sampler" => vk::DescriptorType::SAMPLER,
        "sampled_image" => vk::DescriptorType::SAMPLED_IMAGE,
        "combined_image" => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        "storage_image" => vk::DescriptorType::STORAGE_IMAGE,
        _ => return Err(unsupported(e, "genre")),
    })
}

/// The checks the C++ ran on every entry before touching the GPU, in the same order, so the
/// first refused entry and its reason are the same.
pub fn check(entries: &[LayoutEntry], robust: bool) -> Result<(), Exit> {
    for e in entries {
        descriptor_type(e)?;
        let image = e.is_image();
        if image && e.dim != "2d" && e.dim != "3d" && e.dim != "cube" {
            return Err(unsupported(e, &e.dim));
        }
        // Review of #45: a float view for an isampler or usampler, a single sample for a
        // sampler2DMS or a sampler without comparison for a sampler2DShadow are invalid uses.
        let sampled = e.kind == "sampled_image" || e.kind == "combined_image";
        if sampled && e.stype != "float" {
            return Err(unsupported(e, "texels entiers"));
        }
        // Storage image: the texel type must follow the format (R32_SINT for int, R32_UINT for uint).
        let mismatch = match e.stype.as_str() {
            "sint" => e.format != 24,
            "uint" => e.format != 33,
            _ => e.format == 24 || e.format == 33,
        };
        if e.kind == "storage_image" && mismatch {
            return Err(unsupported(e, "type des texels et format"));
        }
        if image && e.ms != 0 {
            return Err(unsupported(e, "multi-echantillons"));
        }
        // The bench page table is short: without robustness, a page past it would read out of bounds.
        if e.kind == "bda_pagetable" && !robust {
            return Err(unsupported(e, "table de pages sans robustesse"));
        }
        if image && e.depth == 1 {
            return Err(unsupported(e, "comparaison de profondeur"));
        }
        if e.kind == "storage_image"
            && (e.dim == "cube" || storage_format(e.format) == vk::Format::UNDEFINED)
        {
            return Err(unsupported(e, "format d'image"));
        }
    }
    Ok(())
}
