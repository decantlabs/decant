//! What the shell gave the scripts for free: the environment (with what a step exports or unsets
//! for the commands after it), running external programs with redirections, `command -v`, the
//! file tests and the file operations, and a dry run mode that prints the commands and file
//! changes instead of making them.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};

/// How a run stops: the exit code the script would have exited with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exit(pub i32);

pub type Res<T = ()> = Result<T, Exit>;

/// Where one output stream of a command goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sink {
    Inherit,
    /// `>/dev/null`
    Null,
    /// `>file` or `>>file` (append).
    File(String, bool),
    /// stdout only: kept and answered as text, as `$(...)` does.
    Capture,
    /// stdout only: `| tee file`, to the console and the file.
    Tee(String),
    /// stderr only: `2>&1`, wherever stdout goes.
    Stdout,
}

/// One external command, as the script would run it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmd {
    pub program: String,
    pub args: Vec<String>,
    /// The variables exported or unset (None) before it, then its own `VAR=value` prefixes.
    pub env: Vec<(String, Option<String>)>,
    /// The variables of `env` that the command line itself sets, for the dry run listing.
    pub own_env: Vec<String>,
    pub cwd: Option<String>,
    pub stdout: Sink,
    pub stderr: Sink,
}

impl Cmd {
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    pub fn args<S: Into<String>>(mut self, args: impl IntoIterator<Item = S>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// `VAR=value command`.
    pub fn env(mut self, name: &str, value: &str) -> Self {
        self.env.push((name.to_owned(), Some(value.to_owned())));
        self.own_env.push(name.to_owned());
        self
    }

    pub fn cwd(mut self, dir: &str) -> Self {
        self.cwd = Some(dir.to_owned());
        self
    }

    pub fn stdout(mut self, sink: Sink) -> Self {
        self.stdout = sink;
        self
    }

    pub fn stderr(mut self, sink: Sink) -> Self {
        self.stderr = sink;
        self
    }

    /// The command as a shell line, for the dry run.
    pub fn shell_line(&self) -> String {
        let mut line = String::new();
        for name in &self.own_env {
            if let Some((_, Some(value))) = self.env.iter().rev().find(|(n, _)| n == name) {
                line.push_str(&format!("{name}={} ", quote(value)));
            }
        }
        line.push_str(&quote(&self.program));
        for arg in &self.args {
            line.push(' ');
            line.push_str(&quote(arg));
        }
        match &self.stdout {
            Sink::Null => line.push_str(" >/dev/null"),
            Sink::File(path, false) => line.push_str(&format!(" >{}", quote(path))),
            Sink::File(path, true) => line.push_str(&format!(" >>{}", quote(path))),
            _ => {}
        }
        match &self.stderr {
            Sink::Null => line.push_str(" 2>/dev/null"),
            Sink::Stdout => line.push_str(" 2>&1"),
            Sink::File(path, _) => line.push_str(&format!(" 2>{}", quote(path))),
            _ => {}
        }
        if let Sink::Tee(path) = &self.stdout {
            line.push_str(&format!(" | tee {}", quote(path)));
        }
        if let Some(dir) = &self.cwd {
            line = format!("(cd {} && {line})", quote(dir));
        }
        line
    }
}

/// `value` quoted for a POSIX shell when it needs it.
pub fn quote(value: &str) -> String {
    if !value.is_empty() && value.bytes().all(|b| b.is_ascii_alphanumeric() || b"%+,-./:=@_^".contains(&b)) {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', r"'\''"))
    }
}

/// The result of a command: its exit code (`128 + signal` when killed) and the captured stdout.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Outcome {
    pub code: i32,
    pub stdout: String,
}

/// Runs external programs. The real one spawns them, the tests record them.
pub trait Runner {
    /// Runs `cmd`. For `Sink::Tee`, `echo` gets the lines that also go to the file.
    fn run(&mut self, cmd: &Cmd, echo: &mut dyn Write) -> Outcome;
    /// `command -v program`.
    fn find(&self, program: &str) -> bool;
}

/// Spawns the programs, inheriting this process's environment.
pub struct System;

impl Runner for System {
    fn run(&mut self, cmd: &Cmd, echo: &mut dyn Write) -> Outcome {
        match spawn(cmd, echo) {
            Ok(outcome) => outcome,
            // The shell's own message, where the command's stderr was going.
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                let what = if cmd.program.contains('/') { "No such file or directory" } else { "command not found" };
                report(cmd, &format!("{}: {what}", cmd.program));
                Outcome { code: 127, stdout: String::new() }
            }
            Err(e) => {
                report(cmd, &format!("{}: {e}", cmd.program));
                Outcome { code: 126, stdout: String::new() }
            }
        }
    }

    fn find(&self, program: &str) -> bool {
        find_in_path(program, std::env::var_os("PATH").as_deref())
    }
}

/// `command -v` for a program name or path.
pub fn find_in_path(program: &str, path: Option<&std::ffi::OsStr>) -> bool {
    if program.contains('/') {
        return is_executable(Path::new(program));
    }
    path.is_some_and(|path| std::env::split_paths(path).any(|dir| is_executable(&dir.join(program))))
}

/// A line on the stderr of `cmd`, wherever that goes, for a command that could not start.
fn report(cmd: &Cmd, line: &str) {
    let target = match (&cmd.stderr, &cmd.stdout) {
        (Sink::Null, _) | (Sink::Stdout, Sink::Null) => return,
        (Sink::File(path, append), _) | (Sink::Stdout, Sink::File(path, append)) => Some((path, *append)),
        _ => None,
    };
    match target.and_then(|(path, append)| open_sink(path, append).ok()) {
        Some(mut file) => {
            let _ = writeln!(file, "{line}");
        }
        None => eprintln!("{line}"),
    }
}

fn open_sink(path: &str, append: bool) -> io::Result<fs::File> {
    fs::OpenOptions::new().write(true).create(true).append(append).truncate(!append).open(path)
}

fn spawn(cmd: &Cmd, echo: &mut dyn Write) -> io::Result<Outcome> {
    let mut command = Command::new(&cmd.program);
    command.args(&cmd.args);
    for (name, value) in &cmd.env {
        match value {
            Some(value) => command.env(name, value),
            None => command.env_remove(name),
        };
    }
    if let Some(dir) = &cmd.cwd {
        command.current_dir(dir);
    }
    // The stdout end, and a second handle on it for `2>&1`.
    let mut tee = None;
    let (stdout, merged): (Stdio, Option<Stdio>) = match &cmd.stdout {
        Sink::Inherit | Sink::Stdout => (Stdio::inherit(), Some(Stdio::from(io::stdout()))),
        Sink::Null => (Stdio::null(), Some(Stdio::null())),
        Sink::File(path, append) => {
            let file = open_sink(path, *append)?;
            let second = file.try_clone()?;
            (file.into(), Some(second.into()))
        }
        Sink::Capture => (Stdio::piped(), None),
        Sink::Tee(path) => {
            let (reader, writer) = io::pipe()?;
            let second = writer.try_clone()?;
            tee = Some((reader, open_sink(path, false)?));
            (writer.into(), Some(second.into()))
        }
    };
    let stderr = match &cmd.stderr {
        Sink::Inherit | Sink::Capture | Sink::Tee(_) => Stdio::inherit(),
        Sink::Null => Stdio::null(),
        Sink::File(path, append) => open_sink(path, *append)?.into(),
        Sink::Stdout => merged.unwrap_or_else(Stdio::inherit),
    };
    command.stdout(stdout).stderr(stderr);
    let mut child = command.spawn()?;
    // The pipe ends of the parent go with `command`, so that the reader sees the end.
    drop(command);
    let mut captured = Vec::new();
    if let Some((mut reader, mut file)) = tee {
        let mut buffer = [0u8; 8192];
        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            echo.write_all(&buffer[..n])?;
            echo.flush()?;
            file.write_all(&buffer[..n])?;
        }
    } else if let Some(mut out) = child.stdout.take() {
        out.read_to_end(&mut captured)?;
    }
    let status = child.wait()?;
    Ok(Outcome { code: exit_code(status), stdout: String::from_utf8_lossy(&captured).into_owned() })
}

fn exit_code(status: std::process::ExitStatus) -> i32 {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return 128 + signal;
        }
    }
    status.code().unwrap_or(1)
}

/// `[ -x path ]`: present and executable (on Windows: present, or present with `.exe`).
pub fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).is_ok_and(|m| m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        path.is_file() || path.with_extension("exe").is_file()
    }
}

/// `[ -f path ]`: a regular file, following links.
pub fn is_file(path: &str) -> bool {
    Path::new(path).is_file()
}

/// `[ -d path ]`
pub fn is_dir(path: &str) -> bool {
    Path::new(path).is_dir()
}

/// `[ -e path ] || [ -L path ]`: anything there, a dangling link included.
pub fn is_there(path: &str) -> bool {
    fs::symlink_metadata(path).is_ok()
}

/// The names of a folder that `*` matches: no hidden ones, sorted as the C locale sorts them.
pub fn glob_names(dir: &str) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .map(|entries| entries.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    names.retain(|n| !n.starts_with('.'));
    names.sort();
    names
}

/// `$(cat file 2>/dev/null || echo '?')`: the content without its trailing newlines.
pub fn cat_or_question(path: &str) -> String {
    match fs::read(path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).trim_end_matches('\n').to_owned(),
        Err(_) => "?".to_owned(),
    }
}

/// Everything a step needs: the program name for messages, the environment, the runner and the
/// console streams.
pub struct Ctx {
    /// `$0` of the script: how the program was started.
    pub prog: String,
    /// The environment the process started with.
    base: BTreeMap<String, String>,
    /// What steps exported (Some) or unset (None) since.
    overlay: BTreeMap<String, Option<String>>,
    pub runner: Box<dyn Runner>,
    pub out: Box<dyn Write>,
    pub err: Box<dyn Write>,
    /// Print commands and file changes instead of making them.
    pub dry_run: bool,
    /// Paths a dry run would have created, so that later tests see them.
    planned: BTreeSet<String>,
    /// A fixed `date +%Y%m%d-%H%M%S`, for tests.
    pub stamp: Option<String>,
}

impl Ctx {
    pub fn new(prog: String, base: BTreeMap<String, String>, runner: Box<dyn Runner>) -> Self {
        Self {
            prog,
            base,
            overlay: BTreeMap::new(),
            runner,
            out: Box::new(io::stdout()),
            err: Box::new(io::stderr()),
            dry_run: false,
            planned: BTreeSet::new(),
            stamp: None,
        }
    }

    /// The process environment of this program, as the scripts saw it.
    pub fn from_process(runner: Box<dyn Runner>) -> Self {
        let prog = std::env::args().next().unwrap_or_else(|| "launch_prep".to_owned());
        let base =
            std::env::vars_os().filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?))).collect();
        let mut ctx = Self::new(prog, base, runner);
        ctx.dry_run = ctx.var("DECANT_DRY_RUN").as_deref() == Some("1");
        ctx
    }

    /// `$NAME`, unset when not set.
    pub fn var(&self, name: &str) -> Option<String> {
        match self.overlay.get(name) {
            Some(value) => value.clone(),
            None => self.base.get(name).cloned(),
        }
    }

    /// `${NAME:-default}`: the default when unset or empty.
    pub fn var_or(&self, name: &str, default: &str) -> String {
        self.var(name).filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_owned())
    }

    /// `export NAME=value`
    pub fn export(&mut self, name: &str, value: &str) {
        self.overlay.insert(name.to_owned(), Some(value.to_owned()));
    }

    /// `unset NAME`
    pub fn unset(&mut self, name: &str) {
        self.overlay.insert(name.to_owned(), None);
    }

    /// The overlay, to put back after a step that ran as a child script with its own exports.
    pub fn save_env(&self) -> BTreeMap<String, Option<String>> {
        self.overlay.clone()
    }

    pub fn restore_env(&mut self, saved: BTreeMap<String, Option<String>>) {
        self.overlay = saved;
    }

    /// The exported environment, sorted by name, as `env` lists it.
    pub fn environment(&self) -> Vec<(String, String)> {
        let mut all = self.base.clone();
        for (name, value) in &self.overlay {
            match value {
                Some(value) => all.insert(name.clone(), value.clone()),
                None => all.remove(name),
            };
        }
        all.into_iter().collect()
    }

    /// A command with the current exports.
    pub fn command(&self, program: &str) -> Cmd {
        Cmd {
            program: program.to_owned(),
            args: Vec::new(),
            env: self.overlay.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            own_env: Vec::new(),
            cwd: None,
            stdout: Sink::Inherit,
            stderr: Sink::Inherit,
        }
    }

    /// Runs `cmd` and answers its outcome, whatever the exit code.
    pub fn exec(&mut self, cmd: &Cmd) -> Outcome {
        let _ = self.out.flush();
        if self.dry_run {
            let _ = writeln!(self.err, "[dry-run] {}", cmd.shell_line());
            return Outcome::default();
        }
        self.runner.run(cmd, &mut *self.out)
    }

    /// Runs `cmd` as a plain line of a `set -e` script: a failure ends the run with its code.
    pub fn exec_ok(&mut self, cmd: &Cmd) -> Res<Outcome> {
        let outcome = self.exec(cmd);
        if outcome.code == 0 { Ok(outcome) } else { Err(Exit(outcome.code)) }
    }

    /// `command -v program >/dev/null`
    pub fn has(&self, program: &str) -> bool {
        self.runner.find(program)
    }

    pub fn say(&mut self, msg: &str) {
        let _ = writeln!(self.out, "\x1b[1m:: {msg}\x1b[0m");
        let _ = self.out.flush();
    }

    pub fn warn(&mut self, msg: &str) {
        let _ = self.out.flush();
        let _ = writeln!(self.err, "\x1b[33m** {msg}\x1b[0m");
    }

    /// Prints the message and answers the exit code 1, for `return Err(ctx.die(...))`.
    pub fn die(&mut self, msg: &str) -> Exit {
        let _ = self.out.flush();
        let _ = writeln!(self.err, "\x1b[31m!! {msg}\x1b[0m");
        Exit(1)
    }

    /// `echo text`
    pub fn echo(&mut self, text: &str) {
        let _ = writeln!(self.out, "{text}");
    }

    /// An error of a file operation, reported as the coreutils tool would and ending the run
    /// with its exit code 1.
    fn fs_error(&mut self, what: String, e: io::Error) -> Exit {
        let _ = self.out.flush();
        let _ = writeln!(self.err, "{what}: {e}");
        Exit(1)
    }

    fn dry(&mut self, line: String, creates: &str) -> bool {
        if self.dry_run {
            let _ = writeln!(self.err, "[dry-run] {line}");
            self.planned.insert(creates.to_owned());
        }
        self.dry_run
    }

    /// A file test that also sees what a dry run would have created.
    pub fn exists(&self, path: &str) -> bool {
        is_there(path) || self.planned.contains(path)
    }

    /// `mkdir -p dir`
    pub fn mkdir_p(&mut self, dir: &str) -> Res {
        if self.dry(format!("mkdir -p {}", quote(dir)), dir) {
            return Ok(());
        }
        fs::create_dir_all(dir).map_err(|e| self.fs_error(format!("mkdir: cannot create directory '{dir}'"), e))
    }

    /// `cp -f src dst`: through an existing destination, or replacing it when it cannot be
    /// opened for writing.
    pub fn copy_force(&mut self, src: &str, dst: &str) -> Res {
        if self.dry(format!("cp -f {} {}", quote(src), quote(dst)), dst) {
            return Ok(());
        }
        if !Path::new(src).exists() {
            let e = io::Error::new(io::ErrorKind::NotFound, "No such file or directory");
            return Err(self.fs_error(format!("cp: cannot stat '{src}'"), e));
        }
        let copied = fs::copy(src, dst).or_else(|_| {
            let _ = fs::remove_file(dst);
            fs::copy(src, dst)
        });
        copied.map(|_| ()).map_err(|e| self.fs_error(format!("cp: cannot create regular file '{dst}'"), e))
    }

    /// `ln -sfn target link`
    pub fn symlink(&mut self, target: &str, link: &str) -> Res {
        if self.dry(format!("ln -sfn {} {}", quote(target), quote(link)), link) {
            return Ok(());
        }
        if fs::symlink_metadata(link).is_ok_and(|m| !m.is_dir()) {
            let _ = fs::remove_file(link);
        }
        make_symlink(target, link).map_err(|e| self.fs_error(format!("ln: failed to create symbolic link '{link}'"), e))
    }

    /// `printf ... >file`
    pub fn write_file(&mut self, path: &str, text: &str) -> Res {
        if self.dry(format!("printf %s {} >{}", quote(text), quote(path)), path) {
            return Ok(());
        }
        fs::write(path, text).map_err(|e| self.fs_error(format!("{}: {path}", self.prog), e))
    }

    /// `date +%Y%m%d-%H%M%S`, local time through `date` itself.
    pub fn stamp(&mut self) -> String {
        if let Some(stamp) = &self.stamp {
            return stamp.clone();
        }
        let cmd = self.command("date").arg("+%Y%m%d-%H%M%S").stdout(Sink::Capture);
        let was_dry = std::mem::replace(&mut self.dry_run, false);
        let out = self.runner.run(&cmd, &mut io::sink());
        self.dry_run = was_dry;
        out.stdout.trim_end_matches('\n').to_owned()
    }
}

#[cfg(unix)]
fn make_symlink(target: &str, link: &str) -> io::Result<()> {
    std::os::unix::fs::symlink(target, link)
}

#[cfg(windows)]
fn make_symlink(target: &str, link: &str) -> io::Result<()> {
    // The launch driver is a Linux tool. On Windows (tests, dry runs) a link needs developer mode.
    let base = Path::new(link).parent().unwrap_or(Path::new("."));
    if base.join(target).is_dir() || Path::new(target).is_dir() {
        std::os::windows::fs::symlink_dir(target, link)
    } else {
        std::os::windows::fs::symlink_file(target, link)
    }
}

#[cfg(not(any(unix, windows)))]
fn make_symlink(_target: &str, _link: &str) -> io::Result<()> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "symbolic links"))
}
