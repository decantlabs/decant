//! Shader precompilation for `decant-launch`: discovers shader archives and per title pipeline
//! manifests, invokes `decant_shader_warm` with `--progress`, renders a live console progress bar
//! and optionally shows a lightweight native Win32 progress window.

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::io::{BufRead, BufReader, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Environment variable that enables the direct shader path in `d3d12_xs.dll`.
pub const DIRECT_ENV: &str = "DECANT_D3D12X_DIRECT";
/// Environment variable that overrides the direct shader disk cache folder.
pub const DIRECT_CACHE_ENV: &str = "DECANT_D3D12X_DIRECT_CACHE";
/// Environment variable that sets the per title pipeline manifest path.
pub const PIPELINE_MANIFEST_ENV: &str = "DECANT_D3D12X_PIPELINE_MANIFEST";
/// Optional override pointing directly to the `decant_shader_warm` executable.
pub const SHADER_WARM_ENV: &str = "DECANT_SHADER_WARM";

const MAX_DISCOVERY_DEPTH: usize = 16;
const SHADER_EXTENSIONS: &[&str] = &["dxbc", "fssm", "cso", "shd", "shader", "shaders"];

/// Inputs for [`precompile`].
#[derive(Clone, Debug, Default)]
pub struct PrecompileConfig {
    pub package: Option<PathBuf>,
    pub overlay: Option<PathBuf>,
    pub cache: Option<PathBuf>,
    pub manifest: Option<PathBuf>,
    pub threads: Option<usize>,
    pub ui: bool,
}

/// Outcome of a precompile step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PrecompileOutcome {
    /// No manifest and no shader archives were found, so nothing was run.
    Skipped,
    /// `decant_shader_warm` ran and completed with optional final progress counters.
    Warmed(Option<Progress>),
}

/// Structured progress reported by `decant_shader_warm --progress`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Progress {
    pub done: usize,
    pub total: usize,
    pub ready: u64,
    pub compiled: u64,
    pub cache_hits: u64,
    pub failed: u64,
}

/// Parses one `PROGRESS <done>/<total> ready=<n> compiled=<n> cache_hits=<n> failed=<n>` line.
pub fn parse_progress_line(line: &str) -> Option<Progress> {
    let mut parts = line.split_whitespace();
    if parts.next()? != "PROGRESS" {
        return None;
    }
    let (done_str, total_str) = parts.next()?.split_once('/')?;
    let done = done_str.parse::<usize>().ok()?;
    let total = total_str.parse::<usize>().ok()?;
    let mut progress = Progress { done, total, ready: 0, compiled: 0, cache_hits: 0, failed: 0 };
    for token in parts {
        let Some((key, value)) = token.split_once('=') else { continue };
        let Ok(n) = value.parse::<u64>() else { continue };
        match key {
            "ready" => progress.ready = n,
            "compiled" => progress.compiled = n,
            "cache_hits" => progress.cache_hits = n,
            "failed" => progress.failed = n,
            _ => {}
        }
    }
    Some(progress)
}

/// Formats one console progress bar line for `p`.
pub fn format_console_bar(p: &Progress, width: usize) -> String {
    let width = width.max(10);
    let pct = (p.done.min(p.total) * 100).checked_div(p.total).unwrap_or(100);
    let filled = (p.done.min(p.total) * width).checked_div(p.total).unwrap_or(width);
    let mut bar = String::with_capacity(width);
    for i in 0..width {
        if i < filled {
            bar.push('=');
        } else if i == filled && p.done < p.total {
            bar.push('>');
        } else {
            bar.push(' ');
        }
    }
    format!("[{bar}] {pct:>3}% ({}/{}) ready={} compiled={} hits={} failed={}", p.done, p.total, p.ready, p.compiled, p.cache_hits, p.failed)
}

/// Resolves `%LOCALAPPDATA%\Decant\<title-id>\manifest.txt` from `title_folder\MicrosoftGame.config`.
pub fn per_title_manifest_path_with(title_folder: &Path, local_appdata: Option<&OsStr>) -> Option<PathBuf> {
    let base = local_appdata.filter(|v| !v.is_empty())?;
    let config = decant_gdk_config::read_gdk_config(&title_folder.join("MicrosoftGame.config")).ok()?;
    let title_id = decant_gdk_config::title_id_number(&config.title_id)?;
    Some(PathBuf::from(base).join("Decant").join(format!("{title_id:08X}")).join("manifest.txt"))
}

/// Resolves the per title manifest path using the process `LOCALAPPDATA` environment variable.
pub fn per_title_manifest_path(title_folder: &Path) -> Option<PathBuf> {
    per_title_manifest_path_with(title_folder, std::env::var_os("LOCALAPPDATA").as_deref())
}

fn env_value(name: &str, extra_env: &[(String, String)], lookup: &impl Fn(&str) -> Option<String>) -> Option<String> {
    if let Some((_, v)) = extra_env.iter().rev().find(|(k, _)| k.eq_ignore_ascii_case(name)) {
        return Some(v.clone());
    }
    lookup(name)
}

/// Returns `(PIPELINE_MANIFEST_ENV, path)` when `DECANT_D3D12X_DIRECT` is active,
/// `DECANT_D3D12X_PIPELINE_MANIFEST` is unset or empty, and `title_folder` has a valid title id.
pub fn default_pipeline_manifest_env_with(
    title_folder: &Path,
    extra_env: &[(String, String)],
    lookup: impl Fn(&str) -> Option<String>,
    local_appdata: Option<&OsStr>,
) -> Option<(String, String)> {
    let direct_on = env_value(DIRECT_ENV, extra_env, &lookup).is_some_and(|v| !v.is_empty() && v != "0");
    if !direct_on {
        return None;
    }
    let manifest_set = env_value(PIPELINE_MANIFEST_ENV, extra_env, &lookup).is_some_and(|v| !v.trim().is_empty());
    if manifest_set {
        return None;
    }
    let path = per_title_manifest_path_with(title_folder, local_appdata)?;
    Some((PIPELINE_MANIFEST_ENV.to_owned(), path.to_string_lossy().into_owned()))
}

/// Convenience wrapper around [`default_pipeline_manifest_env_with`] using `LOCALAPPDATA`.
pub fn default_pipeline_manifest_env(
    title_folder: &Path,
    extra_env: &[(String, String)],
    lookup: impl Fn(&str) -> Option<String>,
) -> Option<(String, String)> {
    default_pipeline_manifest_env_with(title_folder, extra_env, lookup, std::env::var_os("LOCALAPPDATA").as_deref())
}

/// Computes the default direct shader cache folder from explicit environment values.
pub fn default_cache_dir_with(
    windows: bool,
    local_appdata: Option<&OsStr>,
    user_profile: Option<&OsStr>,
    xdg_cache: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Option<PathBuf> {
    let non_empty = |v: Option<&OsStr>| v.filter(|s| !s.is_empty()).map(PathBuf::from);
    let base = if windows {
        non_empty(local_appdata).or_else(|| non_empty(user_profile).map(|p| p.join("AppData").join("Local")))
    } else {
        non_empty(xdg_cache).or_else(|| non_empty(home).map(|h| h.join(".cache")))
    };
    base.map(|b| b.join("Decant").join("direct-dxil"))
}

/// Resolves the cache folder for `decant_shader_warm` from an explicit flag or `DECANT_D3D12X_DIRECT_CACHE`.
pub fn resolve_cache_dir(explicit: Option<&Path>, env_override: Option<&OsStr>) -> Option<PathBuf> {
    if let Some(path) = explicit {
        return Some(path.to_path_buf());
    }
    let default = default_cache_dir_with(
        cfg!(windows),
        std::env::var_os("LOCALAPPDATA").as_deref(),
        std::env::var_os("USERPROFILE").as_deref(),
        std::env::var_os("XDG_CACHE_HOME").as_deref(),
        std::env::var_os("HOME").as_deref(),
    );
    let env_path = env_override.filter(|v| !v.is_empty()).map(PathBuf::from);
    match env_path.as_deref() {
        Some(p) if p == Path::new("0") => default,
        Some(p) if p.is_absolute() => env_path,
        Some(p) => default.map(|base| base.join(p)),
        None => default,
    }
}

/// Discovers shader archives and shader folders under `root`.
///
/// Directories whose name contains `shader` are returned directly so `decant_shader_warm` scans
/// them without overflowing command line limits. Files elsewhere whose name contains `shader` or
/// whose extension is a known shader container extension are returned individually.
pub fn discover_shader_archives(root: &Path) -> Vec<PathBuf> {
    let mut found = BTreeSet::new();
    walk_archives(root, 0, &mut found);
    found.into_iter().collect()
}

fn walk_archives(dir: &Path, depth: usize, found: &mut BTreeSet<PathBuf>) {
    if depth > MAX_DISCOVERY_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        let Ok(meta) = std::fs::metadata(&path) else { continue };
        if meta.is_dir() {
            if name.contains("shader") {
                found.insert(path);
            } else {
                walk_archives(&path, depth + 1, found);
            }
        } else if meta.is_file() && is_shader_archive_name(&name, &path) {
            found.insert(path);
        }
    }
}

fn is_shader_archive_name(lower_name: &str, path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
    if let Some(ext) = &ext {
        if matches!(ext.as_str(), "dll" | "exe" | "pdb" | "log" | "txt" | "xml" | "config" | "json" | "toml" | "ini") {
            return false;
        }
        if SHADER_EXTENSIONS.contains(&ext.as_str()) {
            return true;
        }
    }
    lower_name.contains("shader")
}

/// Locates `decant_shader_warm` from `DECANT_SHADER_WARM`, next to `decant-launch`, or on `PATH`.
pub fn find_shader_warm() -> Option<PathBuf> {
    let exe_name = if cfg!(windows) { "decant_shader_warm.exe" } else { "decant_shader_warm" };
    if let Some(override_path) = std::env::var_os(SHADER_WARM_ENV).filter(|v| !v.is_empty()) {
        let p = PathBuf::from(override_path);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(current) = std::env::current_exe()
        && let Some(dir) = current.parent()
    {
        let sibling = dir.join(exe_name);
        if sibling.is_file() {
            return Some(sibling);
        }
        if dir.file_name().is_some_and(|n| n == "deps")
            && let Some(parent) = dir.parent()
        {
            let target_sibling = parent.join(exe_name);
            if target_sibling.is_file() {
                return Some(target_sibling);
            }
        }
    }
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let candidate = dir.join(exe_name);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Resolves the manifest path and shader archives from `cfg` without running `decant_shader_warm`.
pub fn resolve_inputs(cfg: &PrecompileConfig) -> Result<(Option<PathBuf>, Vec<PathBuf>), String> {
    let manifest = match &cfg.manifest {
        Some(path) => {
            if !path.is_file() {
                return Err(format!("pipeline manifest not found: {}", path.display()));
            }
            Some(path.clone())
        }
        None => {
            let candidates = [cfg.package.as_deref(), cfg.overlay.as_deref()];
            candidates.into_iter().flatten().find_map(per_title_manifest_path).filter(|p| p.is_file())
        }
    };

    let mut archives = BTreeSet::new();
    for root in [cfg.package.as_deref(), cfg.overlay.as_deref()].into_iter().flatten() {
        for item in discover_shader_archives(root) {
            archives.insert(item);
        }
    }
    Ok((manifest, archives.into_iter().collect()))
}

/// Runs shader precompilation according to `cfg`.
pub fn precompile(cfg: &PrecompileConfig) -> Result<PrecompileOutcome, String> {
    if cfg.package.is_none() && cfg.overlay.is_none() && cfg.manifest.is_none() {
        return Err("pass --package, --overlay or --manifest".to_owned());
    }
    let (manifest, archives) = resolve_inputs(cfg)?;
    if manifest.is_none() && archives.is_empty() {
        println!("decant-launch: no pipeline manifest or shader archives found, skipping precompile");
        return Ok(PrecompileOutcome::Skipped);
    }
    let cache = resolve_cache_dir(cfg.cache.as_deref(), std::env::var_os(DIRECT_CACHE_ENV).as_deref())
        .ok_or_else(|| "cannot determine shader cache folder (pass --cache or set LOCALAPPDATA)".to_owned())?;
    std::fs::create_dir_all(&cache).map_err(|e| format!("cannot create cache folder {}: {e}", cache.display()))?;

    let warm_exe = find_shader_warm()
        .ok_or_else(|| "decant_shader_warm not found beside decant-launch or on PATH (build with cargo build -p decant-d3d12x --bin decant_shader_warm)".to_owned())?;

    let mut cmd = Command::new(&warm_exe);
    cmd.arg("--cache").arg(&cache).arg("--progress");
    if let Some(m) = &manifest {
        cmd.arg("--manifest").arg(m);
    }
    for a in &archives {
        cmd.arg("--shaders").arg(a);
    }
    if let Some(threads) = cfg.threads {
        cmd.arg("--threads").arg(threads.to_string());
    }
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("failed to start {}: {e}", warm_exe.display()))?;
    let stdout = child.stdout.take().ok_or_else(|| "failed to capture decant_shader_warm stdout".to_owned())?;
    let stderr = child.stderr.take().ok_or_else(|| "failed to capture decant_shader_warm stderr".to_owned())?;

    let stderr_thread = std::thread::spawn(move || -> Vec<String> {
        BufReader::new(stderr).lines().map_while(Result::ok).collect()
    });

    let ui_window = if cfg.ui { ui::ProgressWindow::spawn("Decant Shader Precompile") } else { None };
    let is_tty = std::io::stdout().is_terminal();
    let mut last_progress: Option<Progress> = None;

    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if let Some(progress) = parse_progress_line(&line) {
            let bar = format_console_bar(&progress, 28);
            if is_tty {
                print!("\r{bar}");
                let _ = std::io::stdout().flush();
            } else {
                println!("{bar}");
            }
            if let Some(win) = &ui_window {
                win.update(&progress);
            }
            last_progress = Some(progress);
        }
    }
    if is_tty && last_progress.is_some() {
        println!();
    }

    let status = child.wait().map_err(|e| format!("waiting for decant_shader_warm failed: {e}"))?;
    let stderr_lines = stderr_thread.join().unwrap_or_default();
    if let Some(win) = ui_window {
        win.close();
    }

    if !status.success() {
        let tail: Vec<&str> = stderr_lines.iter().rev().take(5).map(String::as_str).collect();
        let detail = if tail.is_empty() {
            format!("exit status {status}")
        } else {
            tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
        };
        return Err(format!("decant_shader_warm failed ({detail})"));
    }
    Ok(PrecompileOutcome::Warmed(last_progress))
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod ui {
    use std::sync::{mpsc, Arc, Mutex};
    use std::thread::JoinHandle;

    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows_sys::Win32::Graphics::Gdi::{GetStockObject, UpdateWindow, COLOR_WINDOW, DEFAULT_GUI_FONT, HBRUSH};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::Controls::{InitCommonControlsEx, ICC_PROGRESS_CLASS, INITCOMMONCONTROLSEX, PBM_SETPOS, PBM_SETRANGE32, PBS_SMOOTH, PROGRESS_CLASSW};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetMessageW, GetSystemMetrics, GetWindowLongPtrW, PostMessageW, PostQuitMessage,
        RegisterClassExW, SendMessageW, SetWindowLongPtrW, SetWindowTextW, ShowWindow, TranslateMessage, CW_USEDEFAULT, GWLP_USERDATA, MSG,
        SM_CXSCREEN, SM_CYSCREEN, SW_SHOW, WM_APP, WM_CLOSE, WM_DESTROY, WM_SETFONT, WNDCLASSEXW, WS_CAPTION, WS_CHILD, WS_OVERLAPPED, WS_SYSMENU,
        WS_VISIBLE,
    };

    const SS_LEFT: u32 = 0;

    use super::Progress;

    const WM_APP_UPDATE: u32 = WM_APP + 1;
    const WM_APP_CLOSE: u32 = WM_APP + 2;

    struct SharedState {
        label_hwnd: isize,
        bar_hwnd: isize,
        progress: Option<Progress>,
    }

    pub struct ProgressWindow {
        hwnd: isize,
        state: Arc<Mutex<SharedState>>,
        thread: Option<JoinHandle<()>>,
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    unsafe extern "system" fn wnd_proc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        match msg {
            WM_APP_UPDATE => {
                let ptr = unsafe { GetWindowLongPtrW(hwnd, GWLP_USERDATA) } as *const Mutex<SharedState>;
                if !ptr.is_null() {
                    let state_mutex = unsafe { &*ptr };
                    if let Ok(guard) = state_mutex.lock()
                        && let Some(p) = &guard.progress
                    {
                        let pct = (p.done.min(p.total) * 100).checked_div(p.total).unwrap_or(100);
                        let text = wide(&format!(
                            "Precompiling shaders: {} / {} ({pct}%)  [ready: {}, hits: {}, compiled: {}]",
                            p.done, p.total, p.ready, p.cache_hits, p.compiled
                        ));
                        unsafe {
                            SetWindowTextW(guard.label_hwnd as HWND, text.as_ptr());
                            SendMessageW(guard.bar_hwnd as HWND, PBM_SETRANGE32, 0, p.total.max(1) as LPARAM);
                            SendMessageW(guard.bar_hwnd as HWND, PBM_SETPOS, p.done.min(p.total), 0);
                        }
                    }
                }
                0
            }
            WM_APP_CLOSE => {
                unsafe { DestroyWindow(hwnd) };
                0
            }
            WM_CLOSE => 0,
            WM_DESTROY => {
                unsafe { PostQuitMessage(0) };
                0
            }
            _ => unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) },
        }
    }

    impl ProgressWindow {
        pub fn spawn(title: &str) -> Option<Self> {
            let state = Arc::new(Mutex::new(SharedState { label_hwnd: 0, bar_hwnd: 0, progress: None }));
            let thread_state = Arc::clone(&state);
            let title = title.to_owned();
            let (tx, rx) = mpsc::sync_channel::<isize>(1);

            let thread = std::thread::spawn(move || {
                let icc = INITCOMMONCONTROLSEX { dwSize: size_of::<INITCOMMONCONTROLSEX>() as u32, dwICC: ICC_PROGRESS_CLASS };
                unsafe { InitCommonControlsEx(&icc) };

                let hinstance = unsafe { GetModuleHandleW(std::ptr::null()) };
                let class_name = wide("DecantPrecompileProgressClass");
                let wc = WNDCLASSEXW {
                    cbSize: size_of::<WNDCLASSEXW>() as u32,
                    style: 0,
                    lpfnWndProc: Some(wnd_proc),
                    cbClsExtra: 0,
                    cbWndExtra: 0,
                    hInstance: hinstance,
                    hIcon: std::ptr::null_mut(),
                    hCursor: std::ptr::null_mut(),
                    hbrBackground: (COLOR_WINDOW + 1) as HBRUSH,
                    lpszMenuName: std::ptr::null(),
                    lpszClassName: class_name.as_ptr(),
                    hIconSm: std::ptr::null_mut(),
                };
                unsafe { RegisterClassExW(&wc) };

                let win_w = 480;
                let win_h = 130;
                let screen_w = unsafe { GetSystemMetrics(SM_CXSCREEN) };
                let screen_h = unsafe { GetSystemMetrics(SM_CYSCREEN) };
                let x = if screen_w > win_w { (screen_w - win_w) / 2 } else { CW_USEDEFAULT };
                let y = if screen_h > win_h { (screen_h - win_h) / 2 } else { CW_USEDEFAULT };

                let title_w = wide(&title);
                let hwnd = unsafe {
                    CreateWindowExW(
                        0,
                        class_name.as_ptr(),
                        title_w.as_ptr(),
                        WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
                        x,
                        y,
                        win_w,
                        win_h,
                        std::ptr::null_mut(),
                        std::ptr::null_mut(),
                        hinstance,
                        std::ptr::null(),
                    )
                };
                if hwnd.is_null() {
                    let _ = tx.send(0);
                    return;
                }

                let static_class = wide("STATIC");
                let initial_text = wide("Preparing shader precompilation...");
                let label_hwnd = unsafe {
                    CreateWindowExW(
                        0,
                        static_class.as_ptr(),
                        initial_text.as_ptr(),
                        WS_CHILD | WS_VISIBLE | SS_LEFT,
                        16,
                        16,
                        432,
                        24,
                        hwnd,
                        std::ptr::null_mut(),
                        hinstance,
                        std::ptr::null(),
                    )
                };

                let bar_hwnd = unsafe {
                    CreateWindowExW(
                        0,
                        PROGRESS_CLASSW,
                        std::ptr::null(),
                        WS_CHILD | WS_VISIBLE | PBS_SMOOTH,
                        16,
                        48,
                        432,
                        24,
                        hwnd,
                        std::ptr::null_mut(),
                        hinstance,
                        std::ptr::null(),
                    )
                };

                let font = unsafe { GetStockObject(DEFAULT_GUI_FONT) };
                if !font.is_null() {
                    unsafe {
                        SendMessageW(label_hwnd, WM_SETFONT, font as WPARAM, 1);
                    }
                }

                if let Ok(mut guard) = thread_state.lock() {
                    guard.label_hwnd = label_hwnd as isize;
                    guard.bar_hwnd = bar_hwnd as isize;
                }
                let raw_state = Arc::into_raw(Arc::clone(&thread_state));
                unsafe { SetWindowLongPtrW(hwnd, GWLP_USERDATA, raw_state as isize) };

                unsafe {
                    ShowWindow(hwnd, SW_SHOW);
                    UpdateWindow(hwnd);
                }
                let _ = tx.send(hwnd as isize);

                let mut msg: MSG = unsafe { std::mem::zeroed() };
                while unsafe { GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) } > 0 {
                    unsafe {
                        TranslateMessage(&msg);
                        DispatchMessageW(&msg);
                    }
                }

                unsafe {
                    SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
                    drop(Arc::from_raw(raw_state));
                }
            });

            let hwnd = rx.recv().ok().filter(|&h| h != 0)?;
            Some(Self { hwnd, state, thread: Some(thread) })
        }

        pub fn update(&self, progress: &Progress) {
            if let Ok(mut guard) = self.state.lock() {
                guard.progress = Some(progress.clone());
            }
            unsafe { PostMessageW(self.hwnd as HWND, WM_APP_UPDATE, 0, 0) };
        }

        pub fn close(mut self) {
            self.shutdown();
        }

        fn shutdown(&mut self) {
            if self.hwnd != 0 {
                unsafe { PostMessageW(self.hwnd as HWND, WM_APP_CLOSE, 0, 0) };
                self.hwnd = 0;
            }
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
        }
    }

    impl Drop for ProgressWindow {
        fn drop(&mut self) {
            self.shutdown();
        }
    }
}

#[cfg(not(windows))]
mod ui {
    use super::Progress;

    pub struct ProgressWindow;

    impl ProgressWindow {
        pub fn spawn(_title: &str) -> Option<Self> {
            None
        }
        pub fn update(&self, _progress: &Progress) {}
        pub fn close(self) {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_config(dir: &Path, title_id: &str) {
        let xml = format!(
            r#"<Game configVersion="1">
  <Identity Name="TestTitle" Publisher="CN=Test" Version="1.0.0.0" />
  <TitleId>{title_id}</TitleId>
  <ExecutableList>
    <Executable Name="TestTitle.exe" TargetDeviceFamily="Scarlett" />
  </ExecutableList>
</Game>"#
        );
        std::fs::write(dir.join("MicrosoftGame.config"), xml).unwrap();
    }

    #[test]
    fn manifest_path_resolves_from_title_id() {
        let temp = tempfile::tempdir().unwrap();
        let pkg = temp.path().join("pkg");
        std::fs::create_dir_all(&pkg).unwrap();
        write_config(&pkg, "12AB34CD");

        let local = temp.path().join("localappdata");
        let got = per_title_manifest_path_with(&pkg, Some(local.as_os_str())).unwrap();
        assert_eq!(got, local.join("Decant").join("12AB34CD").join("manifest.txt"));
    }

    #[test]
    fn default_manifest_env_requires_direct_on_and_unset_manifest() {
        let temp = tempfile::tempdir().unwrap();
        let pkg = temp.path().join("pkg");
        std::fs::create_dir_all(&pkg).unwrap();
        write_config(&pkg, "0x00ff0011");
        let local = temp.path().join("local");

        assert_eq!(default_pipeline_manifest_env_with(&pkg, &[], |_| None, Some(local.as_os_str())), None);

        let direct_off = [(DIRECT_ENV.to_owned(), "0".to_owned())];
        assert_eq!(default_pipeline_manifest_env_with(&pkg, &direct_off, |_| None, Some(local.as_os_str())), None);

        let direct_on = [(DIRECT_ENV.to_owned(), "1".to_owned())];
        let got = default_pipeline_manifest_env_with(&pkg, &direct_on, |_| None, Some(local.as_os_str())).unwrap();
        assert_eq!(got.0, PIPELINE_MANIFEST_ENV);
        assert_eq!(PathBuf::from(got.1), local.join("Decant").join("00FF0011").join("manifest.txt"));

        let already_set = [(DIRECT_ENV.to_owned(), "1".to_owned()), (PIPELINE_MANIFEST_ENV.to_owned(), "custom.txt".to_owned())];
        assert_eq!(default_pipeline_manifest_env_with(&pkg, &already_set, |_| None, Some(local.as_os_str())), None);
    }

    #[test]
    fn archive_discovery_finds_shader_dirs_and_files_and_skips_non_shaders() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        std::fs::create_dir_all(root.join("Shaders").join("nested")).unwrap();
        std::fs::create_dir_all(root.join("Content")).unwrap();
        std::fs::write(root.join("Shaders").join("nested").join("a.bin"), b"dxbc").unwrap();
        std::fs::write(root.join("Content").join("title_shaders.pack"), b"pack").unwrap();
        std::fs::write(root.join("Content").join("effect.dxbc"), b"dxbc").unwrap();
        std::fs::write(root.join("Content").join("textures.pak"), b"tex").unwrap();
        std::fs::write(root.join("d3d12_xs.dll"), b"dll").unwrap();

        let found = discover_shader_archives(root);
        assert_eq!(
            found,
            vec![
                root.join("Content").join("effect.dxbc"),
                root.join("Content").join("title_shaders.pack"),
                root.join("Shaders"),
            ]
        );
    }

    #[test]
    fn progress_line_parses_and_formats_console_bar() {
        let line = "PROGRESS 15/30 ready=14 compiled=4 cache_hits=10 failed=1";
        let parsed = parse_progress_line(line).unwrap();
        assert_eq!(parsed, Progress { done: 15, total: 30, ready: 14, compiled: 4, cache_hits: 10, failed: 1 });
        assert_eq!(parse_progress_line("decant_shader_warm: something else"), None);

        let bar = format_console_bar(&parsed, 10);
        assert!(bar.contains(" 50% (15/30)"), "{bar}");
        assert!(bar.contains("ready=14 compiled=4 hits=10 failed=1"), "{bar}");
    }

    #[test]
    fn precompile_fast_paths_when_no_manifest_or_archives_exist() {
        let temp = tempfile::tempdir().unwrap();
        let pkg = temp.path().join("empty_pkg");
        std::fs::create_dir_all(&pkg).unwrap();
        let outcome = precompile(&PrecompileConfig {
            package: Some(pkg),
            overlay: None,
            cache: Some(temp.path().join("cache")),
            manifest: None,
            threads: None,
            ui: false,
        })
        .unwrap();
        assert_eq!(outcome, PrecompileOutcome::Skipped);
    }
}
