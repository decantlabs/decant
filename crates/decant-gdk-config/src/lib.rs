//! Reads a GDK `MicrosoftGame.config` and checks the headers of a staged title executable, as
//! the former C++ reader `src/gdk/gdk_config_reader.cpp` did, with the same results and error texts.
//!
//! The config reader extracts Identity (Name, Publisher, Version), TitleId, StoreId, MSAAppId,
//! GameOsVersion and the ExecutableList. Its XML tokenizer covers the subset the schema uses:
//! elements, attributes, text, comments, `<?xml?>` declarations, self closing tags and the five
//! predefined entities. No DTDs, CDATA or namespaces.
//!
//! The input is read as bytes like the C++ reader. Field values are returned as strings, with
//! bytes that are not UTF-8 replaced.

mod pe;
mod xml;

pub use pe::{check_pe_image, check_pe_image_from_memory};

use std::path::Path;

/// One `Executable` entry of the `ExecutableList`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Executable {
    pub name: String,
    /// "Scarlett", "XboxOne", "PC", or empty.
    pub target_device_family: String,
    pub is_dev_only: bool,
}

/// The fields Decant and `launch_prep.sh` need from `MicrosoftGame.config`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GdkConfig {
    /// "0" or "1".
    pub config_version: String,
    pub name: String,
    pub publisher: String,
    /// Identity Version, a quad, "1.0.0.0" when absent.
    pub version: String,
    pub title_id: String,
    pub store_id: String,
    pub msa_app_id: String,
    /// Console only.
    pub game_os_version: String,
    pub executables: Vec<Executable>,
    /// The first entry that is not dev only, else the first entry.
    pub primary_executable: String,
    /// `<PersistentLocalStorage><SizeMB>`: the persistent local storage quota in MiB, when given.
    pub persistent_local_storage_mb: Option<u64>,
}

/// Parses `MicrosoftGame.config` from a file.
pub fn read_gdk_config(path: &Path) -> Result<GdkConfig, String> {
    let data = std::fs::read(path).map_err(|_| format!("cannot open {}", path.display()))?;
    read_gdk_config_from_bytes(&data)
}

/// Parses `MicrosoftGame.config` from its bytes in memory.
pub fn read_gdk_config_from_bytes(xml: &[u8]) -> Result<GdkConfig, String> {
    let root = xml::parse(xml)?;
    if root.name != b"Game" {
        return Err(format!("root element is <{}>, expected <Game>", text(&root.name)));
    }
    let mut out = GdkConfig { config_version: root.attr(b"configVersion").map(text).unwrap_or_default(), ..GdkConfig::default() };
    if let Some(identity) = root.child(b"Identity") {
        let field = |key: &[u8]| identity.attr(key).map(text).unwrap_or_default();
        (out.name, out.publisher, out.version) = (field(b"Name"), field(b"Publisher"), field(b"Version"));
    }
    if out.version.is_empty() {
        out.version = "1.0.0.0".to_owned(); // schema default
    }
    if !is_version_quad(out.version.as_bytes()) {
        return Err(format!("Identity Version is not a quad: {}", out.version));
    }
    if out.name.is_empty() || out.publisher.is_empty() {
        return Err("Identity Name and Publisher are required".to_owned());
    }
    let child_text = |name: &[u8]| root.child(name).map(|n| text(&n.text)).unwrap_or_default();
    out.title_id = child_text(b"TitleId");
    out.store_id = child_text(b"StoreId");
    out.msa_app_id = child_text(b"MSAAppId");
    out.game_os_version = child_text(b"GameOsVersion");
    out.persistent_local_storage_mb = root.child(b"PersistentLocalStorage").and_then(|n| n.child(b"SizeMB")).and_then(|n| text(&n.text).trim().parse().ok());

    if let Some(list) = root.child(b"ExecutableList") {
        for c in list.children.iter().filter(|c| c.name == b"Executable") {
            let Some(name) = c.attr(b"Name").filter(|n| !n.is_empty()) else { continue };
            out.executables.push(Executable {
                name: text(name),
                target_device_family: c.attr(b"TargetDeviceFamily").map(text).unwrap_or_default(),
                is_dev_only: c.attr(b"IsDevOnly").is_some_and(|v| v == b"true" || v == b"1"),
            });
        }
    }
    let primary = out.executables.iter().find(|e| !e.is_dev_only).or(out.executables.first());
    out.primary_executable = primary.map(|e| e.name.clone()).unwrap_or_default();
    if out.executables.is_empty() {
        return Err("ExecutableList has no Executable entries".to_owned());
    }
    Ok(out)
}

/// A `<TitleId>` value as a number: hexadecimal, with or without `0x`.
pub fn title_id_number(text: &str) -> Option<u32> {
    let t = text.trim();
    let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
    u32::from_str_radix(t, 16).ok()
}

/// The folder Decant keeps a title's data in: `%LOCALAPPDATA%\Decant\<title id as eight hex
/// digits>`. `None` without `LOCALAPPDATA`.
pub fn title_data_folder(title_id: u32) -> Option<std::path::PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty())?;
    Some(std::path::PathBuf::from(base).join("Decant").join(format!("{title_id:08X}")))
}

/// The host folder of a title's persistent local storage: `pls` in [`title_data_folder`].
pub fn persistent_local_storage_folder(title_id: u32) -> Option<std::path::PathBuf> {
    title_data_folder(title_id).map(|f| f.join("pls"))
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Four runs of ASCII digits separated by dots.
fn is_version_quad(v: &[u8]) -> bool {
    let parts: Vec<&[u8]> = v.split(|&c| c == b'.').collect();
    parts.len() == 4 && parts.iter().all(|p| !p.is_empty() && p.iter().all(u8::is_ascii_digit))
}

#[cfg(test)]
mod tests;
