//! `launch_prep`, the port of `src/gdk/launch_prep.sh`: stage and (eventually) launch a Scarlett
//! GDK title on Linux through Ferestre's patched Proton runtime and the Decant and GDK shims.
//! Every step, variable, message and exit code is the script's. Where it differs, the item says
//! so.

use std::io::Write;

use crate::ctx::{Ctx, Exit, Res, Sink, cat_or_question, glob_names, is_dir, is_executable, is_file};
use crate::sandbox;

/// The usage text: lines 2 to 19 of the script header, which it printed (with lines 20 and 21,
/// a blank and the start of the next comment), without the comment marks.
pub const USAGE: &str = "\
launch_prep — stage and (eventually) launch a Scarlett GDK title on
Linux through Ferestre's patched Proton runtime + the Decant/GDK shims.

  launch_prep runtime   download + verify the pinned Ferestre runtime
  launch_prep stubs     build the shim DLLs (xmem, d3d12_xs), the Wine
                        tests and the gdk_config_reader CLI
  launch_prep prefix    create the Proton prefix (vkd3d-proton installed)
  launch_prep stage     build the install dir: config, DLLs, shims, data
  launch_prep test      run the companion-DLL smoke test (air-gapped
                        container by default; host only with the same
                        DECANT_HOST_LAUNCH=1 opt-out as launch)
  launch_prep all       runtime + stubs + prefix + stage + test
  launch_prep launch    run the staged title exe: inside the air-gapped
                        sandbox container by default; on the host only
                        with DECANT_HOST_LAUNCH=1 (loud warning)
  launch_prep status    print every component's state; exits non-zero
                        only after the full report
  launch_prep env       print the launch environment (for copy/paste)
";

/// Pinned Ferestre runtime release: the digest of ferestre-runtime-11.0.20261107.1.tar.xz,
/// confirmed out of band. Its .minisig verifies against Ferestre's release signing key (the
/// public key embedded in crates/ferestre-core/src/signing.rs of icex/ferestre), and the archive
/// was then hashed locally. The release's own SHA256SUMS is not consulted: it ships from the same
/// source as the archive. Bumping RUNTIME_TAG requires redoing this and updating the pin.
/// `runtime` fails closed on any mismatch.
pub const RUNTIME_SHA256_PIN: &str = "b5af9be3a87e025eba5c941e11dac1e51c42923270b123eb06e7c9ff1016f83d";

/// Shims copied app local from the stubs folder.
pub const SHIM_DLLS: [&str; 2] = ["xmem.dll", "d3d12_xs.dll"];

/// Metadata copied into the install dir. Everything else at the top level of the payload
/// (archives, data folders...) is symlinked, not copied.
pub const META_ITEMS: [&str; 2] = ["MicrosoftGame.Config", "appxmanifest.xml"];

/// WINEDLLOVERRIDES:
/// * xmem, d3d12_xs: native (shims, app local)
/// * d3d12, d3d12core: native (vkd3d-proton, installed into the prefix by Proton)
/// * dxgi: native (DXVK dxgi, Proton default)
/// * msvc runtime: native,builtin (the package's own copies first)
/// * xaudio2_9: builtin (Wine/FAudio, console ordinal layout UNVERIFIED)
/// * xgameruntime, gameinput: builtin (Ferestre implementations)
pub const DLL_OVERRIDES: &str = "xmem,d3d12_xs=n;d3d12,d3d12core,dxgi=n;vcruntime140,vcruntime140_1,msvcp140,msvcp140_1,msvcp140_2,msvcp140_atomic_wait,msvcp140_codecvt_ids,concrt140,vccorlib140,vcomp140=n,b;xaudio2_9=b;xgameruntime,gameinput=b";

/// The default WINEDEBUG of a launch: GDK titles log through OutputDebugString.
const LAUNCH_WINEDEBUG: &str = "+debugstr,err+module,+loaddll,+xgameruntime";

/// The commands of the smoke test inside the container, as the script wrote them.
pub const CONTAINER_SMOKE: &str = "
            mkdir -p /tmp/smoke && cd /tmp/smoke &&
            cp /data/game/*.dll . 2>/dev/null; cp /data/stubs/*.dll . 2>/dev/null;
            cp /tmp/gdk_smoketest.exe . &&
            exec wine64 ./gdk_smoketest.exe";

/// The repository checkout this program was built from: the default of DECANT_ROOT, as the
/// script's own folder was.
pub fn default_root() -> String {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.ancestors().nth(2).unwrap_or(manifest);
    root.to_string_lossy().replace('\\', "/")
}

/// Every variable of the script's config block, read once at the start as the script does.
#[derive(Debug, Clone)]
pub struct Config {
    pub root: String,
    pub gdk_dir: String,
    pub sandbox_dir: String,
    pub src_files: String,
    pub runtime_tag: String,
    pub runtime_name: String,
    pub runtime_base: String,
    pub runtime_dir: String,
    pub prefix: String,
    pub game_dir: String,
    pub exe_name: String,
    pub stubs_dir: String,
    pub log_dir: String,
    pub fake_steam: String,
    pub container: String,
    pub config_reader: String,
    pub runtime_sha256: String,
    pub runtime_minisign_pub: String,
    pub runtime_url: String,
}

impl Config {
    pub fn from(ctx: &Ctx) -> Self {
        let root = ctx.var_or("DECANT_ROOT", &default_root());
        let src_default = ctx.var_or("DECANT_PACKAGE_DIR", &format!("{root}/extracted/package"));
        let runtime_tag = ctx.var_or("RUNTIME_TAG", "runtime-11.0.20261107.1");
        let runtime_name = ctx.var_or("RUNTIME_NAME", "ferestre-runtime-11.0.20261107.1");
        let runtime_base = ctx.var_or("RUNTIME_BASE", &format!("{root}/runtime"));
        Self {
            gdk_dir: ctx.var_or("GDK_DIR", &format!("{root}/src/gdk")),
            sandbox_dir: ctx.var_or("SANDBOX_DIR", &format!("{root}/sandbox")),
            src_files: ctx.var_or("SRC_FILES", &src_default),
            runtime_dir: ctx.var_or("RUNTIME_DIR", &format!("{runtime_base}/{runtime_name}")),
            prefix: ctx.var_or("PREFIX", &format!("{runtime_base}/prefix")),
            game_dir: ctx.var_or("GAME_DIR", &format!("{root}/install/title")),
            exe_name: ctx.var_or("DECANT_EXE_NAME", ""),
            stubs_dir: ctx.var_or("STUBS_DIR", &format!("{root}/build/gdk")),
            log_dir: ctx.var_or("LOG_DIR", &format!("{root}/logs")),
            fake_steam: ctx.var_or("FAKE_STEAM", &format!("{runtime_base}/fake-steam")),
            container: ctx.var_or("CONTAINER_NAME", "decant-sandbox"),
            config_reader: ctx.var_or("CONFIG_READER", ""),
            runtime_sha256: ctx.var_or("RUNTIME_SHA256", RUNTIME_SHA256_PIN),
            runtime_minisign_pub: ctx.var_or("RUNTIME_MINISIGN_PUB", ""),
            runtime_url: format!("https://github.com/icex/ferestre/releases/download/{runtime_tag}"),
            runtime_tag,
            runtime_name,
            runtime_base,
            root,
        }
    }

    fn proton(&self) -> String {
        format!("{}/proton", self.runtime_dir)
    }

    fn archive(&self) -> String {
        format!("{}/{}.tar.xz", self.runtime_base, self.runtime_name)
    }

    fn system_reg(&self) -> String {
        format!("{}/pfx/system.reg", self.prefix)
    }
}

/// The lower case hex SHA-256 of a file, as `sha256sum` prints it.
pub fn sha256_file(path: &str) -> std::io::Result<String> {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    let mut file = std::fs::File::open(path)?;
    let mut buffer = vec![0u8; 1 << 16];
    loop {
        let n = std::io::Read::read(&mut file, &mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The launch driver, one run.
pub struct Prep<'a> {
    pub ctx: &'a mut Ctx,
    pub cfg: Config,
    /// The scratch folder of the run (`mktemp -d`), removed when the run ends.
    work_dir: Option<tempfile::TempDir>,
}

impl<'a> Prep<'a> {
    pub fn new(ctx: &'a mut Ctx) -> Self {
        let cfg = Config::from(ctx);
        Self { ctx, cfg, work_dir: None }
    }

    /// `f 2>/dev/null`: runs `f` with this program's own stderr hidden.
    fn quietly<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        let saved = std::mem::replace(&mut self.ctx.err, Box::new(std::io::sink()));
        let result = f(self);
        self.ctx.err = saved;
        result
    }

    fn prog(&self) -> String {
        self.ctx.prog.clone()
    }

    fn host_launch(&self) -> bool {
        self.ctx.var_or("DECANT_HOST_LAUNCH", "0") == "1"
    }

    /// `make_workdir`: `mktemp -d "${TMPDIR:-/tmp}/decant-prep.XXXXXX"`.
    fn make_workdir(&mut self) -> Res<String> {
        if self.work_dir.is_none() {
            let base = self.ctx.var_or("TMPDIR", "/tmp");
            let made = tempfile::Builder::new().prefix("decant-prep.").rand_bytes(6).tempdir_in(&base);
            match made {
                Ok(dir) => self.work_dir = Some(dir),
                Err(e) => {
                    let msg =
                        format!("mktemp: failed to create directory via template '{base}/decant-prep.XXXXXX': {e}");
                    let _ = writeln!(self.ctx.err, "{msg}");
                    return Err(Exit(1));
                }
            }
        }
        Ok(self.work_dir.as_ref().map(|d| d.path().to_string_lossy().replace('\\', "/")).unwrap_or_default())
    }

    /// `find_reader`: an external reader only when CONFIG_READER names one. Without it the
    /// reader is built in (the gdk-config crate), so it cannot be missing. The script looked
    /// for the gdk_config_reader binary in STUBS_DIR instead.
    fn external_reader(&mut self) -> Res<Option<String>> {
        let reader = self.cfg.config_reader.clone();
        if reader.is_empty() {
            return Ok(None);
        }
        if !is_executable(std::path::Path::new(&reader)) {
            return Err(self.ctx.die(&format!("CONFIG_READER={reader} is not executable")));
        }
        Ok(Some(reader))
    }

    /// `exe_name`: DECANT_EXE_NAME, else the primary executable of the payload's config (the
    /// first ExecutableList entry without IsDevOnly). `None` when it cannot be determined, with
    /// the reader's message on stderr unless `quiet` (the script's `2>/dev/null`).
    pub fn exe_name(&mut self, quiet: bool) -> Res<Option<String>> {
        if !self.cfg.exe_name.is_empty() {
            return Ok(Some(self.cfg.exe_name.clone()));
        }
        let config = format!("{}/MicrosoftGame.Config", self.cfg.src_files);
        if let Some(reader) = self.external_reader()? {
            let cmd = self.ctx.command(&reader).args(["--exec", config.as_str()]).stdout(Sink::Capture);
            let cmd = if quiet { cmd.stderr(Sink::Null) } else { cmd };
            let out = self.ctx.exec(&cmd);
            let name = out.stdout.trim_end_matches('\n').to_owned();
            return Ok((out.code == 0).then_some(name));
        }
        let message = match decant_gdk_config::read_gdk_config(std::path::Path::new(&config)) {
            Ok(info) => match info.executables.iter().find(|e| !e.is_dev_only) {
                Some(exe) => return Ok(Some(exe.name.clone())),
                None => format!(
                    "gdk_config_reader: {config}: every ExecutableList entry is IsDevOnly; nothing to launch (set DECANT_EXE_NAME to override)"
                ),
            },
            Err(e) => format!("gdk_config_reader: {e}"),
        };
        if !quiet {
            let _ = writeln!(self.ctx.err, "{message}");
        }
        Ok(None)
    }

    /// `pe_preflight`: DOS and PE headers, AMD64 machine, a normal GUI or CUI subsystem. It reads
    /// headers only and cannot see encryption.
    fn pe_preflight(&mut self, exe: &str) -> Res {
        let reader = self.external_reader()?;
        self.ctx.say(&format!("PE pre-flight: {exe}"));
        let ok = match reader {
            Some(reader) => {
                let cmd = self.ctx.command(&reader).args(["--pe-check", exe]);
                self.ctx.exec(&cmd).code == 0
            }
            None => match decant_gdk_config::check_pe_image(std::path::Path::new(exe)) {
                Ok(()) => {
                    self.ctx.echo(&format!("ok: {exe} is a plain x86-64 PE (GUI/CUI subsystem)"));
                    true
                }
                Err(e) => {
                    let _ = self.ctx.out.flush();
                    let _ = writeln!(self.ctx.err, "{e}");
                    false
                }
            },
        };
        if ok { Ok(()) } else { Err(self.ctx.die("executable failed the PE pre-flight check")) }
    }

    /// `verify_runtime_archive`: against the repository pin. Fails closed: with neither anchor
    /// set it stops with instructions instead of trusting anything the release ships.
    pub fn verify_runtime_archive(&mut self, archive: &str) -> Res {
        let pinned = self.cfg.runtime_sha256.clone();
        if !pinned.is_empty() {
            if self.ctx.dry_run && !is_file(archive) {
                let _ = writeln!(self.ctx.err, "[dry-run] sha256sum {archive} (pinned {pinned})");
                return Ok(());
            }
            let got = match sha256_file(archive) {
                Ok(got) => got,
                Err(e) => {
                    let _ = writeln!(self.ctx.err, "sha256sum: {archive}: {e}");
                    return Err(self.ctx.die(&format!("cannot hash {archive}")));
                }
            };
            if got != pinned {
                return Err(self.ctx.die(&format!("runtime archive hash mismatch: got {got}, pinned {pinned}")));
            }
            self.ctx.say(&format!("runtime archive hash OK ({got})"));
            return Ok(());
        }
        let key = self.cfg.runtime_minisign_pub.clone();
        if !key.is_empty() {
            if !self.ctx.has("minisign") {
                return Err(self.ctx.die(&format!("minisign is required to verify the runtime against {key}")));
            }
            let sig = format!("{archive}.minisig");
            if !is_file(&sig) {
                let base = std::path::Path::new(archive).file_name().map(|n| n.to_string_lossy().into_owned());
                let url = format!("{}/{}.minisig", self.cfg.runtime_url, base.unwrap_or_default());
                let cmd = self.ctx.command("curl").args(["-fSL", "-o", sig.as_str(), url.as_str()]);
                if self.ctx.exec(&cmd).code != 0 {
                    return Err(self.ctx.die(&format!("minisig download failed: {}", self.cfg.runtime_url)));
                }
            }
            let cmd = self.ctx.command("minisign").args(["-Vm", archive, "-p", key.as_str()]);
            if self.ctx.exec(&cmd).code != 0 {
                return Err(self.ctx.die(&format!("minisign verification failed for {archive}")));
            }
            self.ctx.say(&format!("runtime minisig OK (key: {key})"));
            return Ok(());
        }
        Err(self.ctx.die(&format!(
            "no pinned trust anchor for {}; refusing to extract. Verify the release digest out-of-band, then either:
    - set RUNTIME_SHA256=<digest> (and commit it as RUNTIME_SHA256_PIN in this script), or
    - point RUNTIME_MINISIGN_PUB at the release's minisign public key (minisign required).",
            self.cfg.runtime_name
        )))
    }

    /// `launch_env`: the environment shared by `test` and the host launch. Mirrors
    /// ferestre/scripts/launch-gdk.sh (Steam variables dropped, GameInput on).
    fn launch_env(&mut self) {
        for name in ["LD_PRELOAD", "WINEPREFIX", "WINEDLLPATH", "WINELOADER", "WINESERVER"] {
            self.ctx.unset(name);
        }
        let c = &self.cfg;
        let fixed = [
            ("STEAM_COMPAT_DATA_PATH", c.prefix.clone()),
            ("STEAM_COMPAT_CLIENT_INSTALL_PATH", c.fake_steam.clone()),
            ("WINEDLLOVERRIDES", DLL_OVERRIDES.to_owned()),
            ("WINE_GAMEINPUT", "1".to_owned()),
        ];
        for (name, value) in fixed {
            self.ctx.export(name, &value);
        }
        for (name, default) in [("VKD3D_CONFIG", "dxr"), ("VKD3D_FEATURE_LEVEL", "12_2"), ("PROTON_LOG", "1")] {
            let value = self.ctx.var_or(name, default);
            self.ctx.export(name, &value);
        }
        let log_dir = self.cfg.log_dir.clone();
        self.ctx.export("PROTON_LOG_DIR", &log_dir);
        let debug = self.ctx.var_or("WINEDEBUG", LAUNCH_WINEDEBUG);
        self.ctx.export("WINEDEBUG", &debug);
        let decant = self.ctx.var_or("DECANT_LOG", "1");
        self.ctx.export("DECANT_LOG", &decant);
        let game_dir = self.cfg.game_dir.clone();
        self.ctx.export("DECANT_INSTALL_DIR", &game_dir);
        if let Some(xuid) = self.ctx.var("XGR_XUID").filter(|v| !v.is_empty()) {
            self.ctx.export("XGR_XUID", &xuid);
        }
    }

    pub fn step_runtime(&mut self) -> Res {
        let proton = self.cfg.proton();
        if is_executable(std::path::Path::new(&proton)) {
            let version = cat_or_question(&format!("{}/version", self.cfg.runtime_dir));
            self.ctx.say(&format!("runtime present: {} ({version})", self.cfg.runtime_dir));
            return Ok(());
        }
        let base = self.cfg.runtime_base.clone();
        self.ctx.mkdir_p(&base)?;
        let archive = self.cfg.archive();
        if !is_file(&archive) {
            self.ctx.say(&format!("downloading {} (~300 MB)", self.cfg.runtime_name));
            let url = format!("{}/{}.tar.xz", self.cfg.runtime_url, self.cfg.runtime_name);
            let cmd = self.ctx.command("curl").args(["-fSL", "-o", archive.as_str(), url.as_str()]);
            if self.ctx.exec(&cmd).code != 0 {
                return Err(self.ctx.die(&format!("download failed: {}", self.cfg.runtime_url)));
            }
        }
        self.verify_runtime_archive(&archive)?;
        let cmd = self.ctx.command("tar").args(["-C", base.as_str(), "-xJf", archive.as_str()]);
        self.ctx.exec_ok(&cmd)?;
        if !self.ctx.dry_run && !is_executable(std::path::Path::new(&proton)) {
            return Err(self.ctx.die(&format!("runtime archive did not contain {}/proton", self.cfg.runtime_name)));
        }
        Ok(())
    }

    pub fn step_stubs(&mut self) -> Res {
        if !self.ctx.has("x86_64-w64-mingw32-gcc") {
            return Err(self.ctx.die("need mingw-w64 (x86_64-w64-mingw32-gcc)"));
        }
        let out = format!("OUT={}", self.cfg.stubs_dir);
        let cmd = self.ctx.command("make").args([
            "-C",
            self.cfg.gdk_dir.as_str(),
            out.as_str(),
            "all",
            "tests",
        ]);
        self.ctx.exec_ok(&cmd.stdout(Sink::Null))?;
        let dlls: Vec<String> = glob_names(&self.cfg.stubs_dir).into_iter().filter(|n| n.ends_with(".dll")).collect();
        let dlls = if dlls.is_empty() { "*.dll".to_owned() } else { dlls.join(" ") };
        self.ctx.say(&format!("shims built: {dlls}"));
        let reader = match self.external_reader() {
            Ok(Some(reader)) => reader,
            Ok(None) => "built in (gdk-config)".to_owned(),
            Err(_) => "MISSING".to_owned(),
        };
        self.ctx.say(&format!("config reader: {reader}"));
        Ok(())
    }

    pub fn step_prefix(&mut self) -> Res {
        let prog = self.prog();
        if !is_executable(std::path::Path::new(&self.cfg.proton())) {
            return Err(self.ctx.die(&format!("no runtime; run: {prog} runtime")));
        }
        for dir in [self.cfg.prefix.clone(), self.cfg.fake_steam.clone(), self.cfg.log_dir.clone()] {
            self.ctx.mkdir_p(&dir)?;
        }
        self.launch_env();
        let log = format!("{}/prefix-init.log", self.cfg.log_dir);
        let proton = self.cfg.proton();
        if !is_file(&self.cfg.system_reg()) {
            self.ctx.say(&format!("creating Proton prefix at {}", self.cfg.prefix));
            let cmd = self.ctx.command(&proton).env("DISPLAY", "").args(["run", "wineboot", "-u"]);
            let code = self.ctx.exec(&cmd.stdout(Sink::File(log.clone(), false)).stderr(Sink::Stdout)).code;
            if code != 0 {
                self.ctx.warn(&format!("wineboot returned {code} (see {log})"));
            }
        }
        if !is_file(&format!("{}/pfx/drive_c/windows/system32/d3d12.dll", self.cfg.prefix)) {
            self.ctx.warn("vkd3d-proton d3d12.dll not in prefix yet");
        }
        // xgameruntime registers Gaming Services itself on InitializeApiImpl. Doing it ahead of
        // time makes XSystemGetRuntimeInfo non zero from the first call.
        let cmd = self.ctx.command(&proton).args(["run", "regsvr32", "/s", "xgameruntime.dll"]);
        if self.ctx.exec(&cmd.stdout(Sink::File(log, true)).stderr(Sink::Stdout)).code != 0 {
            self.ctx.warn("regsvr32 xgameruntime failed");
        }
        self.ctx.say(&format!("prefix ready: {}/pfx", self.cfg.prefix));
        Ok(())
    }

    pub fn step_stage(&mut self) -> Res {
        let prog = self.prog();
        let (src, game, stubs) = (self.cfg.src_files.clone(), self.cfg.game_dir.clone(), self.cfg.stubs_dir.clone());
        if !is_file(&format!("{src}/MicrosoftGame.Config")) {
            return Err(self.ctx.die(&format!("no package payload at {src} (set DECANT_PACKAGE_DIR)")));
        }
        if !is_file(&format!("{stubs}/xmem.dll")) {
            return Err(self.ctx.die(&format!("shims not built; run: {prog} stubs")));
        }
        self.ctx.mkdir_p(&game)?;
        for item in META_ITEMS {
            if std::path::Path::new(&format!("{src}/{item}")).exists() {
                self.ctx.copy_force(&format!("{src}/{item}"), &format!("{game}/{item}"))?;
            } else {
                self.ctx.warn(&format!("missing payload file: {item}"));
            }
        }
        // Companion DLLs: every top level *.[dD][lL][lL] of the payload, copied app local, so no
        // package specific file name lives here.
        let mut copied = 0;
        for name in glob_names(&src) {
            if name.to_ascii_lowercase().ends_with(".dll") && is_file(&format!("{src}/{name}")) {
                self.ctx.copy_force(&format!("{src}/{name}"), &format!("{game}/{name}"))?;
                copied += 1;
            }
        }
        self.ctx.say(&format!("copied {copied} companion DLL(s) app-local"));
        for shim in SHIM_DLLS {
            self.ctx.copy_force(&format!("{stubs}/{shim}"), &format!("{game}/{shim}"))?;
        }
        for name in glob_names(&src) {
            let link = format!("{game}/{name}");
            if !self.ctx.exists(&link) {
                self.ctx.symlink(&format!("{src}/{name}"), &link)?;
            }
        }
        let Some(exe) = self.exe_name(false)? else {
            return Err(self.ctx.die(&format!(
                "cannot read the executable name from {src}/MicrosoftGame.Config; set DECANT_EXE_NAME"
            )));
        };
        if !std::path::Path::new(&format!("{game}/{exe}")).exists() {
            let text = format!("Place the title executable here as {exe}, then run: {prog} launch\n");
            self.ctx.write_file(&format!("{game}/{exe}.PLACEHOLDER.txt"), &text)?;
        }
        self.ctx.say(&format!("staged {game} (exe expected: {exe})"));
        Ok(())
    }

    pub fn step_test(&mut self) -> Res {
        let prog = self.prog();
        if !is_file(&self.cfg.system_reg()) {
            return Err(self.ctx.die(&format!("no prefix; run: {prog} prefix")));
        }
        if !is_file(&format!("{}/gdk_smoketest.exe", self.cfg.stubs_dir)) {
            return Err(self.ctx.die(&format!("tests not built; run: {prog} stubs")));
        }
        if !self.host_launch() {
            if !self.ctx.has("docker") {
                return Err(self.ctx.die("the smoke test loads package DLLs; default is the air-gapped container (docker), or DECANT_HOST_LAUNCH=1 to run it on this host — against the project rules"));
            }
            return self.container_smoke();
        }
        self.ctx.warn("DECANT_HOST_LAUNCH=1: running package DLLs on this host, with network access");
        self.ctx.warn(
            "this contradicts the project rules (see src/gdk/README.md); the air-gapped container is the default",
        );
        self.host_smoke()
    }

    /// The sandbox, started and checked as `sandbox/run_airgapped.sh` did, as a child script:
    /// what it exports does not reach the steps after it.
    fn sandbox_up(&mut self) -> Res {
        let compose = format!("{}/docker-compose.yml", self.cfg.sandbox_dir);
        if !is_file(&compose) {
            return Err(self.ctx.die(&format!("sandbox runner missing under {}", self.cfg.sandbox_dir)));
        }
        let game = self.ctx.var_or("DECANT_PACKAGE_DIR", &self.cfg.src_files.clone());
        self.ctx.export("DECANT_PACKAGE_DIR", &game);
        let saved = self.ctx.save_env();
        let dir = sandbox::absolute(&self.cfg.sandbox_dir);
        let result = sandbox::run_airgapped(self.ctx, &dir);
        self.ctx.restore_env(saved);
        result
    }

    /// `grep -aE 'RESULT|SUMMARY' log | tr -d '\r'`, then the verdict on `SUMMARY failures=0`.
    fn smoke_verdict(&mut self, log: &str) -> Res {
        let text = std::fs::read(log).map(|b| String::from_utf8_lossy(&b).into_owned()).unwrap_or_default();
        for line in text.split_inclusive('\n') {
            if line.contains("RESULT") || line.contains("SUMMARY") {
                let _ = write!(self.ctx.out, "{}", line.replace('\r', ""));
            }
        }
        if text.contains("SUMMARY failures=0") {
            Ok(())
        } else {
            Err(self.ctx.die(&format!("smoke test reported failures (see {log})")))
        }
    }

    /// The smoke test inside the air-gapped container: the same isolation as `launch`, with the
    /// smoke test and the package and shim DLLs copied into the container's own /tmp.
    fn container_smoke(&mut self) -> Res {
        self.sandbox_up()?;
        let log_dir = self.cfg.log_dir.clone();
        self.ctx.mkdir_p(&log_dir)?;
        let log = format!("{log_dir}/smoketest.log");
        let exe = format!("{}/gdk_smoketest.exe", self.cfg.stubs_dir);
        let target = format!("{}:/tmp/gdk_smoketest.exe", self.cfg.container);
        let cmd = self.ctx.command("docker").args(["cp", exe.as_str(), target.as_str()]);
        if self.ctx.exec(&cmd).code != 0 {
            return Err(self.ctx.die("could not copy the smoke test into the sandbox container"));
        }
        self.ctx.say(&format!("smoke test inside the air-gapped container (log: {log})"));
        let overrides = format!("WINEDLLOVERRIDES={DLL_OVERRIDES}");
        let debug = format!("WINEDEBUG={}", self.ctx.var_or("WINEDEBUG", "err+module"));
        let container = self.cfg.container.clone();
        let cmd = self.ctx.command("docker").args(["exec", "-e", overrides.as_str(), "-e", debug.as_str()]);
        let cmd = cmd.args([container.as_str(), "bash", "-c", CONTAINER_SMOKE]);
        self.ctx.exec(&cmd.stdout(Sink::File(log.clone(), false)).stderr(Sink::Stdout));
        self.smoke_verdict(&log)
    }

    /// The smoke test on the host: loader bring up only, the explicit opt out.
    fn host_smoke(&mut self) -> Res {
        self.launch_env();
        let log_dir = self.cfg.log_dir.clone();
        self.ctx.mkdir_p(&log_dir)?;
        let work = format!("{}/smoke", self.make_workdir()?);
        self.ctx.mkdir_p(&work)?;
        // `cp "$GAME_DIR"/*.dll "$STUBS_DIR/gdk_smoketest.exe" "$work/"`: everything that can be
        // copied is, and any miss (no DLL at all included) gives the warning.
        let game = self.cfg.game_dir.clone();
        let mut sources: Vec<String> =
            glob_names(&game).into_iter().filter(|n| n.ends_with(".dll")).map(|n| format!("{game}/{n}")).collect();
        let missed_glob = sources.is_empty();
        sources.push(format!("{}/gdk_smoketest.exe", self.cfg.stubs_dir));
        let mut failed = missed_glob;
        for source in sources {
            let name = source.rsplit('/').next().unwrap_or_default().to_owned();
            if !is_file(&source) || self.quiet_copy(&source, &format!("{work}/{name}")).is_err() {
                failed = true;
            }
        }
        if failed {
            self.ctx.warn("no staged DLLs to copy");
        }
        let log = format!("{log_dir}/smoketest.log");
        self.ctx.say(&format!("smoke test in {work} (log: {log})"));
        // Proton's `run` goes through its steam.exe wrapper, which drops a console program's
        // stdout: drive the same runtime's wine directly on the prefix Proton prepared.
        let wine = format!("{}/files/bin/wine", self.cfg.runtime_dir);
        let pfx = format!("{}/pfx", self.cfg.prefix);
        let cmd = self.ctx.command(&wine).cwd(&work).env("DISPLAY", "").env("WINEPREFIX", &pfx);
        let cmd = cmd.env("WINEDEBUG", "err+module").arg("gdk_smoketest.exe");
        self.ctx.exec(&cmd.stdout(Sink::File(log.clone(), false)).stderr(Sink::Stdout));
        self.smoke_verdict(&log)
    }

    /// `cp ... 2>/dev/null`: a copy whose error is not printed.
    fn quiet_copy(&mut self, src: &str, dst: &str) -> Res {
        if self.ctx.dry_run {
            return self.ctx.copy_force(src, dst);
        }
        std::fs::copy(src, dst).map(|_| ()).map_err(|_| Exit(1))
    }

    /// The host launch, the explicit opt out (DECANT_HOST_LAUNCH=1): runs the package's own
    /// code on this machine WITH network access, against the project rule.
    fn host_launch_run(&mut self, exe_path: &str) -> Res {
        let prog = self.prog();
        self.ctx.warn("DECANT_HOST_LAUNCH=1: running untrusted package code on this host, with network access");
        self.ctx.warn(
            "this contradicts the project rules (see src/gdk/README.md); the air-gapped container is the default",
        );
        if !is_file(&self.cfg.system_reg()) {
            return Err(self.ctx.die(&format!("no prefix; run: {prog} prefix")));
        }
        if !is_executable(std::path::Path::new(&self.cfg.proton())) {
            return Err(self.ctx.die(&format!("no runtime; run: {prog} runtime")));
        }
        self.launch_env();
        let log_dir = self.cfg.log_dir.clone();
        self.ctx.mkdir_p(&log_dir)?;
        let log = format!("{log_dir}/launch-{}.log", self.ctx.stamp());
        self.ctx.say(&format!("launching {exe_path} on the host (log: {log})"));
        // An encrypted (XVC) executable cannot run this way, and nothing here decrypts one.
        let cmd = self.ctx.command(&self.cfg.proton()).args(["waitforexitandrun", exe_path]);
        self.ctx.exec_ok(&cmd.stdout(Sink::Tee(log)).stderr(Sink::Stdout)).map(|_| ())
    }

    /// The default launch: the package's code runs only inside the air-gapped container
    /// (network_mode none, read-only package mount, isolation verified before anything runs).
    fn container_launch(&mut self, exe: &str, exe_path: &str) -> Res {
        let compose = format!("{}/docker-compose.yml", self.cfg.sandbox_dir);
        if !is_file(&compose) {
            return Err(self.ctx.die(&format!("sandbox runner missing under {}", self.cfg.sandbox_dir)));
        }
        if !self.ctx.has("docker") {
            return Err(self.ctx.die("docker is required: launch runs package code only inside the network-less container (host opt-out: DECANT_HOST_LAUNCH=1, which runs it here — against the project rules)"));
        }
        self.sandbox_up()?;
        let log_dir = self.cfg.log_dir.clone();
        self.ctx.mkdir_p(&log_dir)?;
        let log = format!("{log_dir}/launch-{}.log", self.ctx.stamp());
        // The package mount is read only, so the executable goes into the container's own
        // ephemeral file system.
        let target = format!("{}:/tmp/{exe}", self.cfg.container);
        let cmd = self.ctx.command("docker").args(["cp", exe_path, target.as_str()]);
        if self.ctx.exec(&cmd).code != 0 {
            return Err(self.ctx.die("could not copy the executable into the sandbox container"));
        }
        self.ctx.say(&format!("launching {exe} inside the air-gapped container (log: {log})"));
        self.ctx.warn("container note: distribution Wine with its own prefix (sandbox/prefix); Ferestre's GDK runtime is wired for the host path only so far");
        let mut args = vec![
            "exec".to_owned(),
            "-e".to_owned(),
            format!("WINEDLLOVERRIDES={DLL_OVERRIDES}"),
            "-e".to_owned(),
            "WINE_GAMEINPUT=1".to_owned(),
            "-e".to_owned(),
            format!("WINEDEBUG={}", self.ctx.var_or("WINEDEBUG", LAUNCH_WINEDEBUG)),
            "-e".to_owned(),
            format!("DECANT_LOG={}", self.ctx.var_or("DECANT_LOG", "1")),
        ];
        // `${XGR_XUID:+-e XGR_XUID=$XGR_XUID}`, unquoted: split on blanks as the shell splits it.
        if let Some(xuid) = self.ctx.var("XGR_XUID").filter(|v| !v.is_empty()) {
            args.push("-e".to_owned());
            args.extend(format!("XGR_XUID={xuid}").split_whitespace().map(str::to_owned));
        }
        args.extend([self.cfg.container.clone(), "bash".to_owned(), "-c".to_owned()]);
        args.push(format!("cd /data/game && exec wine64 /tmp/{exe}"));
        let cmd = self.ctx.command("docker").args(args);
        let rc = self.ctx.exec(&cmd.stdout(Sink::Tee(log.clone())).stderr(Sink::Stdout)).code;
        if rc == 0 {
            self.ctx.say(&format!("container launch finished (log: {log})"));
        } else {
            self.ctx.warn(&format!("container launch exited {rc} (log: {log})"));
        }
        self.ctx.say(&format!("stop the sandbox: docker compose -f {}/docker-compose.yml down", self.cfg.sandbox_dir));
        if rc == 0 { Ok(()) } else { Err(Exit(rc)) }
    }

    pub fn step_launch(&mut self) -> Res {
        let Some(exe) = self.exe_name(false)? else {
            return Err(self.ctx.die(&format!(
                "cannot read the executable name from {}/MicrosoftGame.Config; set DECANT_EXE_NAME",
                self.cfg.src_files
            )));
        };
        let exe_path = self.ctx.var_or("DECANT_EXE", &format!("{}/{exe}", self.cfg.game_dir));
        if !is_file(&exe_path) {
            return Err(self.ctx.die(&format!("title executable not present: {exe_path} (placeholder only)")));
        }
        self.pe_preflight(&exe_path)?;
        if self.host_launch() { self.host_launch_run(&exe_path) } else { self.container_launch(&exe, &exe_path) }
    }

    /// Full report first, exit code last: a missing component does not stop the other checks.
    pub fn step_status(&mut self) -> Res {
        let prog = self.prog();
        let mut problems = false;
        let mut rows: Vec<(String, String)> = Vec::new();
        let row = |rows: &mut Vec<(String, String)>, name: &str, value: String| rows.push((name.to_owned(), value));

        // The script ran find_reader in a subshell with its errors hidden.
        match self.quietly(Self::external_reader) {
            Ok(Some(path)) => row(&mut rows, "config reader", path),
            Ok(None) => row(&mut rows, "config reader", "built in (gdk-config)".to_owned()),
            Err(_) => {
                row(&mut rows, "config reader", "MISSING (CONFIG_READER is not executable)".to_owned());
                problems = true;
            }
        }

        let archive = self.cfg.archive();
        if is_file(&archive) {
            let mut state = "present, not verified (no pinned digest)";
            if !self.cfg.runtime_sha256.is_empty() {
                // `sha256sum -c` reads the digest in either case.
                let ok = sha256_file(&archive).is_ok_and(|got| got.eq_ignore_ascii_case(&self.cfg.runtime_sha256));
                state = if ok { "present, hash OK" } else { "present, HASH MISMATCH" };
            }
            problems |= state == "present, HASH MISMATCH";
            row(&mut rows, "runtime archive", format!("{archive} ({state})"));
        } else {
            row(&mut rows, "runtime archive", format!("MISSING (run: {prog} runtime)"));
            problems = true;
        }

        if is_executable(std::path::Path::new(&self.cfg.proton())) {
            let version = cat_or_question(&format!("{}/version", self.cfg.runtime_dir));
            row(&mut rows, "runtime", format!("{} ({version})", self.cfg.runtime_dir));
        } else {
            row(&mut rows, "runtime", format!("MISSING (run: {prog} runtime)"));
            problems = true;
        }

        let mut shims = Vec::new();
        for shim in SHIM_DLLS {
            if is_file(&format!("{}/{shim}", self.cfg.stubs_dir)) {
                shims.push(shim.to_owned());
            } else {
                shims.push(format!("{shim}(MISSING)"));
                problems = true;
            }
        }
        row(&mut rows, "shims", shims.join(" "));

        if is_file(&self.cfg.system_reg()) {
            row(&mut rows, "prefix", format!("{}/pfx", self.cfg.prefix));
        } else {
            row(&mut rows, "prefix", format!("MISSING (run: {prog} prefix)"));
            problems = true;
        }

        let game = self.cfg.game_dir.clone();
        if is_dir(&game) {
            row(&mut rows, "game dir", format!("{game} ({} entries)", glob_names(&game).len()));
        } else {
            row(&mut rows, "game dir", format!("MISSING (run: {prog} stage)"));
            problems = true;
        }

        if is_file(&format!("{game}/MicrosoftGame.Config")) {
            row(&mut rows, "staged config", "MicrosoftGame.Config".to_owned());
        } else {
            row(&mut rows, "staged config", format!("MISSING (run: {prog} stage)"));
            problems = true;
        }

        let exe = self.quietly(|p| p.exe_name(true)).unwrap_or(None).filter(|e| !e.is_empty());
        match exe {
            Some(exe) if is_file(&format!("{game}/{exe}")) => row(&mut rows, "executable", format!("{game}/{exe}")),
            Some(exe) => {
                row(&mut rows, "executable", format!("MISSING: {game}/{exe} (place the title executable)"));
                problems = true;
            }
            None => {
                row(
                    &mut rows,
                    "executable",
                    "MISSING: no primary executable (config unreadable; set DECANT_EXE_NAME)".to_owned(),
                );
                problems = true;
            }
        }

        let host = self.host_launch();
        if host {
            row(
                &mut rows,
                "launch mode",
                "HOST (DECANT_HOST_LAUNCH=1: package code on this host, with network)".to_owned(),
            );
        } else {
            row(
                &mut rows,
                "launch mode",
                "air-gapped container (default; host opt-out: DECANT_HOST_LAUNCH=1)".to_owned(),
            );
        }
        for (name, value) in rows {
            let _ = writeln!(self.ctx.out, "{name:<22} {value}");
        }
        if host {
            self.ctx.warn("host launch enabled; the air-gapped container is the project default");
        }
        if problems {
            return Err(self.ctx.die("status: missing components above"));
        }
        self.ctx.say("status: all components present");
        Ok(())
    }

    /// `env | grep -E '^(STEAM_COMPAT|...|DECANT_)' | sort`
    pub fn step_env(&mut self) -> Res {
        self.launch_env();
        let prefixes =
            ["STEAM_COMPAT", "WINEDLLOVERRIDES", "WINE_GAMEINPUT", "VKD3D_", "PROTON_", "WINEDEBUG", "DECANT_"];
        let mut lines: Vec<String> = self
            .ctx
            .environment()
            .into_iter()
            .map(|(k, v)| format!("{k}={v}"))
            .filter(|line| prefixes.iter().any(|p| line.starts_with(p)))
            .collect();
        lines.sort();
        for line in lines {
            self.ctx.echo(&line);
        }
        Ok(())
    }

    /// One command of the script's `case`.
    pub fn dispatch(&mut self, command: &str) -> Res {
        match command {
            "runtime" => self.step_runtime(),
            "stubs" => self.step_stubs(),
            "prefix" => self.step_prefix(),
            "stage" => self.step_stage(),
            "test" => self.step_test(),
            "all" => {
                self.step_runtime()?;
                self.step_stubs()?;
                self.step_prefix()?;
                self.step_stage()?;
                self.step_test()
            }
            "launch" => self.step_launch(),
            "status" => self.step_status(),
            "env" => self.step_env(),
            _ => {
                let _ = write!(self.ctx.out, "{USAGE}");
                Err(Exit(2))
            }
        }
    }
}

/// The whole program: `launch_prep [--dry-run] [command]`, `status` by default. Answers the
/// exit code.
pub fn main(ctx: &mut Ctx, args: &[String]) -> i32 {
    let mut args = args.iter().map(String::as_str).peekable();
    if args.peek() == Some(&"--dry-run") {
        ctx.dry_run = true;
        args.next();
    }
    let command = args.next().unwrap_or("status");
    let result = Prep::new(ctx).dispatch(command);
    let _ = ctx.out.flush();
    let _ = ctx.err.flush();
    match result {
        Ok(()) => 0,
        Err(Exit(code)) => code,
    }
}
