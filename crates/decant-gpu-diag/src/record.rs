//! The record the runner shows. It holds only the fields docs/GPU_TESTING.md allows: component,
//! version, public hardware and driver, capability, elapsed milliseconds, result and an error
//! category. Free text from the system goes through the scrubber before it lands here.

use serde::Serialize;

use crate::sanitize::Scrubber;

/// The three native tests, the entire hardware scope.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Component {
    Xmem,
    Dxr,
    FidelityFx,
}

impl Component {
    pub const ALL: [Component; 3] = [Component::Xmem, Component::Dxr, Component::FidelityFx];

    pub fn name(self) -> &'static str {
        match self {
            Component::Xmem => "xmem",
            Component::Dxr => "dxr",
            Component::FidelityFx => "fidelityfx",
        }
    }

    /// The test executable in the build folder.
    pub fn executable(self) -> &'static str {
        match self {
            Component::Xmem => "loadtest.exe",
            Component::Dxr => "decant_bvh_test.exe",
            Component::FidelityFx => "ffxtest.exe",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Pass,
    Fail,
    Timeout,
    Unavailable,
}

/// Why a test did not pass, from a fixed vocabulary. The test's own output is never read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Category {
    /// The test ended with this exit code.
    ExitCode(i32),
    /// The test ended without an exit code.
    Ended,
    /// The deadline passed and the process tree was stopped.
    Deadline,
    /// The test executable is not in the build folder.
    MissingExecutable,
    /// An input the test needs, such as `xmem.dll`, is not in the build folder.
    MissingInput,
    /// No FidelityFX backend was given, or the given one does not exist.
    MissingBackend,
    /// The default adapter reports no ray tracing tier.
    NoRaytracing,
    /// The test could not be started or contained.
    LaunchFailed,
}

impl Category {
    pub fn text(self) -> String {
        match self {
            Category::ExitCode(code) => format!("exit code {code}"),
            Category::Ended => "ended without exit code".to_owned(),
            Category::Deadline => "deadline reached, process tree stopped".to_owned(),
            Category::MissingExecutable => "test executable not built".to_owned(),
            Category::MissingInput => "test input not built".to_owned(),
            Category::MissingBackend => "no FidelityFX backend given".to_owned(),
            Category::NoRaytracing => "default adapter without ray tracing".to_owned(),
            Category::LaunchFailed => "test could not be started".to_owned(),
        }
    }
}

/// One adapter as DXGI lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Adapter {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// User mode driver version, `a.b.c.d`.
    pub driver: Option<String>,
    /// D3D12 ray tracing tier as the API reports it (`10` is tier 1.0), `None` when no D3D12
    /// device could be created on the adapter.
    pub raytracing_tier: Option<u32>,
    pub software: bool,
    /// DXGI lists the default adapter first; the tests run on it.
    pub default: bool,
    pub selected: bool,
}

impl Adapter {
    pub fn raytracing(&self) -> String {
        match self.raytracing_tier {
            None => "no D3D12 device".to_owned(),
            Some(0) => "not supported".to_owned(),
            Some(t) => format!("tier {}.{} (value {t})", t / 10, t % 10),
        }
    }

    fn roles(&self) -> String {
        let roles: Vec<&str> = [(self.default, "default"), (self.selected, "selected"), (self.software, "software")].into_iter().filter(|r| r.0).map(|r| r.1).collect();
        if roles.is_empty() { "other".to_owned() } else { roles.join(", ") }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct System {
    pub windows: String,
    pub cpu: String,
    pub ram_gb: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestResult {
    pub component: Component,
    pub outcome: Outcome,
    /// `None` when the test did not run.
    pub elapsed_ms: Option<u64>,
    pub category: Option<Category>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Record {
    pub tool: String,
    pub system: System,
    pub adapters: Vec<Adapter>,
    pub tests: Vec<TestResult>,
}

/// The JSON form: exactly these keys, in this order.
#[derive(Serialize)]
struct AdapterView {
    name: String,
    vendor_id: String,
    device_id: String,
    driver: String,
    raytracing: String,
    roles: String,
}

#[derive(Serialize)]
struct TestView {
    component: &'static str,
    result: Outcome,
    elapsed_ms: Option<u64>,
    error_category: Option<String>,
}

#[derive(Serialize)]
struct RecordView {
    tool: String,
    windows: String,
    cpu: String,
    ram_gb: u64,
    adapters: Vec<AdapterView>,
    tests: Vec<TestView>,
}

impl Record {
    /// Runs every free text field through `scrub`. Numbers and fixed words need no scrubbing.
    pub fn scrubbed(mut self, scrub: &Scrubber) -> Record {
        self.tool = scrub.clean(&self.tool);
        self.system.windows = scrub.clean(&self.system.windows);
        self.system.cpu = scrub.clean(&self.system.cpu);
        for a in &mut self.adapters {
            a.name = scrub.clean(&a.name);
            a.driver = a.driver.as_deref().map(|d| scrub.clean(d));
        }
        self
    }

    fn view(&self) -> RecordView {
        RecordView {
            tool: self.tool.clone(),
            windows: self.system.windows.clone(),
            cpu: self.system.cpu.clone(),
            ram_gb: self.system.ram_gb,
            adapters: self
                .adapters
                .iter()
                .map(|a| AdapterView {
                    name: a.name.clone(),
                    vendor_id: format!("0x{:04X}", a.vendor_id),
                    device_id: format!("0x{:04X}", a.device_id),
                    driver: a.driver.clone().unwrap_or_else(|| "unknown".to_owned()),
                    raytracing: a.raytracing(),
                    roles: a.roles(),
                })
                .collect(),
            tests: self.tests.iter().map(|t| TestView { component: t.component.name(), result: t.outcome, elapsed_ms: t.elapsed_ms, error_category: t.category.map(Category::text) }).collect(),
        }
    }

    pub fn json(&self) -> String {
        serde_json::to_string_pretty(&self.view()).expect("the record serializes")
    }

    /// The record in the order of the GPU form fields.
    pub fn markdown(&self) -> String {
        let v = self.view();
        let mut out = String::new();
        let mut field = |label: &str, body: String| out.push_str(&format!("**{label}**\n{body}\n\n"));
        let adapters: Vec<String> = v.adapters.iter().map(|a| format!("* {} ({}, vendor {}, device {}), ray tracing {}", a.name, a.roles, a.vendor_id, a.device_id, a.raytracing)).collect();
        field("GPU model", adapters.join("\n"));
        let drivers: Vec<String> = v.adapters.iter().map(|a| format!("* {}: {}", a.name, a.driver)).collect();
        field("Driver version", drivers.join("\n"));
        field("Windows version", v.windows.clone());
        field("CPU model", v.cpu.clone());
        field("RAM", format!("{} GB", v.ram_gb));
        field("Tool version or commit", v.tool.clone());
        field("Component", v.tests.iter().map(|t| t.component).collect::<Vec<_>>().join(", "));
        let results: Vec<String> = v
            .tests
            .iter()
            .map(|t| {
                let elapsed = t.elapsed_ms.map_or("not run".to_owned(), |ms| format!("{ms} ms"));
                let result = serde_json::to_value(t.result).ok().and_then(|r| r.as_str().map(str::to_owned)).unwrap_or_default();
                format!("* {}: {result}, {elapsed}{}", t.component, t.error_category.as_ref().map_or(String::new(), |c| format!(", {c}")))
            })
            .collect();
        field("Result and exit code", results.join("\n"));
        field("Sanitized logs", "Error categories only, listed above. The tests' own output was not read or kept.".to_owned());
        field(
            "Reproduction steps",
            "1. Build the native subset with `build.cmd`.\n2. Run `gpu_diag --build <build folder> [--backend <FidelityFX backend DLL>]`.\n3. Each test runs separately with a 45 s deadline on its process tree.".to_owned(),
        );
        out.trim_end().to_owned()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample() -> Record {
        Record {
            tool: "gpu_diag 0.4.0, commit 1a2b3c4".to_owned(),
            system: System { windows: "Windows 11 Pro 24H2 build 26300.1000".to_owned(), cpu: "Invented CPU 9000".to_owned(), ram_gb: 32 },
            adapters: vec![
                Adapter { name: "Invented GPU 100".to_owned(), vendor_id: 0x10DE, device_id: 0x2204, driver: Some("32.0.16.1742".to_owned()), raytracing_tier: Some(12), software: false, default: true, selected: true },
                Adapter { name: "Microsoft Basic Render Driver".to_owned(), vendor_id: 0x1414, device_id: 0x8C, driver: None, raytracing_tier: Some(0), software: true, default: false, selected: false },
            ],
            tests: vec![
                TestResult { component: Component::Xmem, outcome: Outcome::Pass, elapsed_ms: Some(88), category: None },
                TestResult { component: Component::Dxr, outcome: Outcome::Timeout, elapsed_ms: Some(45_000), category: Some(Category::Deadline) },
                TestResult { component: Component::FidelityFx, outcome: Outcome::Unavailable, elapsed_ms: None, category: Some(Category::MissingBackend) },
            ],
        }
    }

    #[test]
    fn json_has_exactly_the_allowlisted_keys() {
        let value: serde_json::Value = serde_json::from_str(&sample().json()).unwrap();
        let keys = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(keys(&value), ["tool", "windows", "cpu", "ram_gb", "adapters", "tests"]);
        assert_eq!(keys(&value["adapters"][0]), ["name", "vendor_id", "device_id", "driver", "raytracing", "roles"]);
        assert_eq!(keys(&value["tests"][0]), ["component", "result", "elapsed_ms", "error_category"]);
        assert_eq!(value["tests"][1]["result"], "timeout");
        assert_eq!(value["tests"][2]["result"], "unavailable");
        assert_eq!(value["adapters"][1]["roles"], "software");
    }

    #[test]
    fn markdown_follows_the_gpu_form_and_keeps_the_four_results_apart() {
        let text = sample().markdown();
        let labels = ["GPU model", "Driver version", "Windows version", "CPU model", "RAM", "Tool version or commit", "Component", "Result and exit code", "Sanitized logs", "Reproduction steps"];
        let positions: Vec<usize> = labels.iter().map(|l| text.find(&format!("**{l}**")).unwrap_or_else(|| panic!("{l} missing"))).collect();
        assert!(positions.is_sorted(), "fields out of form order");
        assert!(text.contains("* xmem: pass, 88 ms"));
        assert!(text.contains("* dxr: timeout, 45000 ms, deadline reached"));
        assert!(text.contains("* fidelityfx: unavailable, not run, no FidelityFX backend given"));
        assert!(text.contains("ray tracing tier 1.2 (value 12)"));
    }

    #[test]
    fn fail_and_unavailable_categories_stay_distinct() {
        let all = [Category::ExitCode(3), Category::Ended, Category::Deadline, Category::MissingExecutable, Category::MissingInput, Category::MissingBackend, Category::NoRaytracing, Category::LaunchFailed];
        let mut texts: Vec<String> = all.iter().map(|c| c.text()).collect();
        texts.sort();
        texts.dedup();
        assert_eq!(texts.len(), all.len());
    }
}
