//! What `decant_shader_identify.py` holds once imported, with its pack positions and metadata read on
//! first use, and the steps `identify` and `adapt` share.

use std::sync::OnceLock;
use std::path::Path;

use crate::adapt::{self, Layout};
use crate::identify::{self, Resource, Table};
use crate::py::{self, Failure};
use crate::shaders::{self, Archive, Config};

pub struct Session {
    pub config: Config,
    pub archive: Archive,
    positions: OnceLock<Vec<u64>>,
    metadata: OnceLock<Vec<serde_json::Value>>,
}

impl Session {
    /// Opens the archive named by `DECANT_SHADERS_RPF`, as importing the Python module does.
    pub fn open(config: Config) -> Result<Session, Failure> {
        let archive = Archive::open(&shaders::archive_path()?)?;
        Ok(Session { config, archive, positions: OnceLock::new(), metadata: OnceLock::new() })
    }

    pub fn positions(&self) -> Result<&[u64], Failure> {
        if let Some(p) = self.positions.get() {
            return Ok(p);
        }
        let positions = shaders::pack_positions(&self.config)?;
        Ok(self.positions.get_or_init(|| positions))
    }

    fn metadata(&self) -> Result<&[serde_json::Value], Failure> {
        if let Some(m) = self.metadata.get() {
            return Ok(m);
        }
        let rows = shaders::metadata(&self.config)?;
        Ok(self.metadata.get_or_init(|| rows))
    }

    fn metadata_field(&self, pack_id: i64, key: &str) -> Result<&serde_json::Value, Failure> {
        py::index(self.metadata()?, pack_id)?.get(key).ok_or_else(|| Failure::key(py::repr_str(key)))
    }

    /// `heap_flags_of(pack_id)`, as its `repr()`.
    pub fn heap_flags(&self, pack_id: i64) -> Result<String, Failure> {
        Ok(py::repr_json(self.metadata_field(pack_id, "heap_flags")?))
    }

    pub fn name(&self, pack_id: i64) -> Result<String, Failure> {
        Ok(self.archive.item_of(self.positions()?, pack_id)?.name.clone())
    }

    /// The container of pack entry `pack_id`.
    pub fn container(&self, pack_id: i64) -> Result<Vec<u8>, Failure> {
        let item = self.archive.item_of(self.positions()?, pack_id)?;
        self.archive.container(item)
    }

    /// `table(data, pack_id)`: the binding tables of the shader's FSSM block.
    pub fn table(&self, pack_id: i64) -> Result<Table, Failure> {
        let blob = self.container(pack_id)?;
        let fs = decant_extract_shaders::container::gpu_code(&blob).ok_or_else(|| Failure::error("TypeError", "cannot convert 'NoneType' object to bytes"))?;
        identify::table(&fs)
    }

    /// `identify(data, pack_id, lines)`.
    pub fn identify(&self, pack_id: i64, lines: &[String]) -> Result<Vec<Resource>, Failure> {
        let table = self.table(pack_id)?;
        self.heap_flags(pack_id)?;
        identify::identify(&table, pack_id, lines)
    }

    /// The `main()` of `decant_shader_adapt.py` up to writing the module: translates the shader into
    /// `work` (removed afterwards, even on failure), identifies its resources and rewrites it.
    pub fn adapt(&self, pack_id: i64, layout: &Layout, work: &Path) -> Result<(Vec<u8>, Vec<String>), Failure> {
        let result = self.adapt_in(pack_id, layout, work);
        let _ = std::fs::remove_dir_all(work);
        result
    }

    fn adapt_in(&self, pack_id: i64, layout: &Layout, work: &Path) -> Result<(Vec<u8>, Vec<String>), Failure> {
        let lines = identify::sources(&self.config, pack_id, work)?.ok_or_else(|| Failure::Exit("traduction impossible".into()))?;
        let resources = self.identify(pack_id, &lines)?;
        let push = match lines.iter().find(|l| l.starts_with("push ")) {
            Some(line) => Some(py::split(line)[1..].iter().map(|f| py::int(f)).collect::<Result<Vec<i64>, Failure>>()?),
            None => None,
        };
        let stage = self.metadata_field(pack_id, "stage")?.clone();
        let path = work.join("0.spv");
        let module = std::fs::read(&path).map_err(|e| Failure::io(&e, &path))?;
        adapt::adapt(&module, &resources, push.as_deref(), layout, stage == "pixel")
    }
}
