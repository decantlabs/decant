//! Runs one test executable under a deadline that covers its whole process tree, timed with a
//! monotonic clock. The test's standard streams go nowhere: its output is never read or kept.

use std::ffi::OsString;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use crate::record::{Category, Outcome};

/// The deadline docs/GPU_TESTING.md sets for each test.
pub const DEADLINE: Duration = Duration::from_secs(45);
/// How often a running test is checked.
const POLL: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Run {
    pub outcome: Outcome,
    pub elapsed_ms: Option<u64>,
    pub category: Option<Category>,
}

impl Run {
    pub fn unavailable(category: Category) -> Run {
        Run { outcome: Outcome::Unavailable, elapsed_ms: None, category: Some(category) }
    }
}

/// Runs `program` with `args` and stops its whole process tree at `deadline`.
pub fn run(program: &Path, args: &[OsString], deadline: Duration) -> Run {
    if !program.is_file() {
        return Run::unavailable(Category::MissingExecutable);
    }
    let mut command = Command::new(program);
    command.args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    let start = Instant::now();
    let Ok((mut child, tree)) = tree::spawn(&mut command) else {
        return Run { outcome: Outcome::Fail, elapsed_ms: None, category: Some(Category::LaunchFailed) };
    };
    let ended = wait(&mut child, start, deadline);
    let elapsed_ms = Some(start.elapsed().as_millis() as u64);
    match ended {
        Some(status) => {
            tree.close();
            match status.code() {
                Some(0) => Run { outcome: Outcome::Pass, elapsed_ms, category: None },
                Some(code) => Run { outcome: Outcome::Fail, elapsed_ms, category: Some(Category::ExitCode(code)) },
                None => Run { outcome: Outcome::Fail, elapsed_ms, category: Some(Category::Ended) },
            }
        }
        None => {
            tree.stop(&mut child);
            let _ = child.wait();
            Run { outcome: Outcome::Timeout, elapsed_ms, category: Some(Category::Deadline) }
        }
    }
}

/// The exit status, or `None` once `deadline` has passed since `start`.
fn wait(child: &mut Child, start: Instant, deadline: Duration) -> Option<std::process::ExitStatus> {
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) if start.elapsed() < deadline => std::thread::sleep(POLL),
            _ => return None,
        }
    }
}

/// On Windows the test starts suspended, joins a job object that kills every process in it when
/// the job closes, and only then resumes, so no child of the test can escape the deadline.
#[cfg(windows)]
#[allow(unsafe_code)]
pub(crate) mod tree {
    use std::os::windows::io::AsRawHandle;
    use std::os::windows::process::CommandExt;
    use std::process::{Child, Command};

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
        TerminateJobObject,
    };
    use windows::Win32::System::Threading::{OpenThread, ResumeThread, THREAD_SUSPEND_RESUME};

    const CREATE_SUSPENDED: u32 = 0x4;

    /// A job object holding one test and everything it starts.
    pub struct Tree(HANDLE);

    // SAFETY: a job handle may be used and closed from any thread.
    unsafe impl Send for Tree {}

    pub fn spawn(command: &mut Command) -> std::io::Result<(Child, Tree)> {
        let mut child = command.creation_flags(CREATE_SUSPENDED).spawn()?;
        match Tree::new().and_then(|tree| tree.adopt(&child).map(|()| tree)) {
            Ok(tree) => Ok((child, tree)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                Err(e)
            }
        }
    }

    impl Tree {
        fn new() -> std::io::Result<Tree> {
            // SAFETY: no attributes and no name; the handle is owned by the returned Tree.
            let job = Tree(unsafe { CreateJobObjectW(None, None) }?);
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `limits` is the structure this information class expects, with its size.
            unsafe { SetInformationJobObject(job.0, JobObjectExtendedLimitInformation, (&raw const limits).cast(), size_of_val(&limits) as u32) }?;
            Ok(job)
        }

        /// Puts the suspended `child` into the job, then resumes its threads.
        fn adopt(&self, child: &Child) -> std::io::Result<()> {
            let process = HANDLE(child.as_raw_handle());
            // SAFETY: both handles are open: the job is owned by self, the process by `child`.
            unsafe { AssignProcessToJobObject(self.0, process) }?;
            resume_threads(child.id())
        }

        /// Closes the job, which also ends anything the test left running.
        pub fn close(self) {}

        /// Ends every process in the job, then the child itself if it somehow is not in it.
        pub fn stop(&self, child: &mut Child) {
            // SAFETY: the job handle is open for the lifetime of self.
            let _ = unsafe { TerminateJobObject(self.0, 1) };
            let _ = child.kill();
        }

        /// Processes still running in the job.
        #[cfg(test)]
        pub fn active(&self) -> u32 {
            use windows::Win32::System::JobObjects::{JOBOBJECT_BASIC_ACCOUNTING_INFORMATION, JobObjectBasicAccountingInformation, QueryInformationJobObject};
            let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
            // SAFETY: `info` is the structure this information class expects, with its size.
            let ok = unsafe { QueryInformationJobObject(Some(self.0), JobObjectBasicAccountingInformation, (&raw mut info).cast(), size_of_val(&info) as u32, None) };
            if ok.is_ok() { info.ActiveProcesses } else { 0 }
        }
    }

    impl Drop for Tree {
        fn drop(&mut self) {
            // SAFETY: the handle came from CreateJobObjectW and is closed once, here. The job's
            // kill on close limit ends whatever is still in it.
            let _ = unsafe { CloseHandle(self.0) };
        }
    }

    fn resume_threads(pid: u32) -> std::io::Result<()> {
        // SAFETY: a thread snapshot of the whole system; the handle is closed below.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) }?;
        let mut entry = THREADENTRY32 { dwSize: size_of::<THREADENTRY32>() as u32, ..Default::default() };
        let mut resumed = 0;
        // SAFETY: `entry` has its size set as the snapshot functions require.
        let mut more = unsafe { Thread32First(snapshot, &mut entry) }.is_ok();
        while more {
            if entry.th32OwnerProcessID == pid {
                // SAFETY: the thread id comes from the snapshot; the handle is closed right after.
                if let Ok(thread) = unsafe { OpenThread(THREAD_SUSPEND_RESUME, false, entry.th32ThreadID) } {
                    // SAFETY: `thread` is open with suspend and resume rights.
                    unsafe {
                        ResumeThread(thread);
                        let _ = CloseHandle(thread);
                    }
                    resumed += 1;
                }
            }
            // SAFETY: as for Thread32First.
            more = unsafe { Thread32Next(snapshot, &mut entry) }.is_ok();
        }
        // SAFETY: the snapshot handle is open and closed once.
        let _ = unsafe { CloseHandle(snapshot) };
        if resumed == 0 { Err(std::io::Error::other("no thread of the test process to resume")) } else { Ok(()) }
    }
}

/// Elsewhere the test runs as a plain child; only the tests of this crate use that path.
#[cfg(not(windows))]
pub(crate) mod tree {
    use std::process::{Child, Command};

    pub struct Tree;

    pub fn spawn(command: &mut Command) -> std::io::Result<(Child, Tree)> {
        Ok((command.spawn()?, Tree))
    }

    impl Tree {
        pub fn close(self) {}

        pub fn stop(&self, child: &mut Child) {
            let _ = child.kill();
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    fn cmd(script: &str) -> (std::path::PathBuf, Vec<OsString>) {
        let shell = std::path::PathBuf::from(std::env::var_os("ComSpec").unwrap_or_else(|| r"C:\Windows\System32\cmd.exe".into()));
        (shell, vec!["/d".into(), "/c".into(), script.into()])
    }

    #[test]
    fn exit_codes_give_pass_and_fail() {
        let (shell, args) = cmd("exit 0");
        let run = run(&shell, &args, DEADLINE);
        assert_eq!((run.outcome, run.category), (Outcome::Pass, None));
        assert!(run.elapsed_ms.is_some());
        let (shell, args) = cmd("exit 3");
        assert_eq!(super::run(&shell, &args, DEADLINE).category, Some(Category::ExitCode(3)));
    }

    #[test]
    fn a_missing_executable_is_unavailable_not_failed() {
        let run = run(Path::new(r"Z:\no\such\loadtest.exe"), &[], DEADLINE);
        assert_eq!(run, Run::unavailable(Category::MissingExecutable));
    }

    #[test]
    fn the_deadline_stops_a_hanging_test_and_its_children() {
        let (shell, args) = cmd("start /b ping -n 60 127.0.0.1 >nul & ping -n 60 127.0.0.1 >nul");
        let started = Instant::now();
        let run = run(&shell, &args, Duration::from_millis(1500));
        assert_eq!((run.outcome, run.category), (Outcome::Timeout, Some(Category::Deadline)));
        let ms = run.elapsed_ms.unwrap();
        assert!((1500..10_000).contains(&ms), "elapsed {ms} ms");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn stopping_the_tree_ends_grandchildren_too() {
        let (shell, args) = cmd("start /b ping -n 60 127.0.0.1 >nul & ping -n 60 127.0.0.1 >nul");
        let mut command = Command::new(shell);
        command.args(args).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
        let (mut child, tree) = tree::spawn(&mut command).unwrap();
        let start = Instant::now();
        while tree.active() < 3 && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(POLL);
        }
        assert!(tree.active() >= 3, "cmd and its two pings should be in the job, found {}", tree.active());
        tree.stop(&mut child);
        let _ = child.wait();
        let start = Instant::now();
        while tree.active() > 0 && start.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(POLL);
        }
        assert_eq!(tree.active(), 0);
    }
}
