//! External programs (the DecantShader probe, `spirv-val`) run as separate processes with a deadline,
//! as `subprocess.run(..., timeout=...)` runs them.

use std::io::{self, Read};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

/// How a program ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ended {
    /// Its exit code, as Python's `returncode` gives it.
    Exit(i64),
    /// It ran past the deadline and was killed.
    Timeout,
}

/// The result of [`run`].
pub struct Output {
    pub ended: Ended,
    /// Empty unless the output was captured.
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

/// `returncode`: on Windows the exit code as an unsigned 32 bit value, elsewhere the code, or the
/// negated signal number when a signal ended the program.
pub fn return_code(status: ExitStatus) -> i64 {
    #[cfg(windows)]
    {
        i64::from(status.code().unwrap_or(0) as u32)
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status.code().map(i64::from).or_else(|| status.signal().map(|s| -i64::from(s))).unwrap_or(0)
    }
    #[cfg(not(any(windows, unix)))]
    {
        i64::from(status.code().unwrap_or(0))
    }
}

fn reader(pipe: Option<impl Read + Send + 'static>) -> Option<thread::JoinHandle<Vec<u8>>> {
    pipe.map(|mut p| {
        thread::spawn(move || {
            let mut out = Vec::new();
            let _ = p.read_to_end(&mut out);
            out
        })
    })
}

fn wait(child: &mut Child, timeout: Duration) -> io::Result<Ended> {
    let deadline = Instant::now() + timeout;
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(Ended::Exit(return_code(status)));
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            child.wait()?;
            return Ok(Ended::Timeout);
        }
        thread::sleep(Duration::from_millis(5));
    }
}

/// Runs `command` until it ends or `timeout` passes. With `capture`, its output is collected,
/// otherwise it goes nowhere.
pub fn run(command: &mut Command, timeout: Duration, capture: bool) -> io::Result<Output> {
    let pipe = || if capture { Stdio::piped() } else { Stdio::null() };
    let mut child = command.stdout(pipe()).stderr(pipe()).spawn()?;
    let (out, err) = (reader(child.stdout.take()), reader(child.stderr.take()));
    let ended = wait(&mut child, timeout)?;
    let collect = |h: Option<thread::JoinHandle<Vec<u8>>>| h.map(|h| h.join().unwrap_or_default()).unwrap_or_default();
    Ok(Output { ended, stdout: collect(out), stderr: collect(err) })
}

/// Runs `command` until it ends or `timeout` passes, with at most `max_mb` megabytes of memory for
/// the process on Windows (none when `max_mb` is 0, and no ceiling elsewhere). Its standard output
/// goes to `stdout` when given, its error output nowhere. Some modules make the compiler of the
/// probe or spirv-diff ask for tens of gigabytes: the ceiling ends such a run instead of the machine.
pub fn run_limited(command: &mut Command, timeout: Duration, max_mb: u64, stdout: Option<std::fs::File>) -> io::Result<Ended> {
    let out = stdout.map_or_else(Stdio::null, Stdio::from);
    let mut child = command.stdin(Stdio::null()).stdout(out).stderr(Stdio::null()).spawn()?;
    let job = ceiling::Job::new(max_mb).filter(|job| job.adopt(&child));
    let ended = wait(&mut child, timeout);
    drop(job);
    ended
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod ceiling {
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOB_OBJECT_LIMIT_PROCESS_MEMORY, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JobObjectExtendedLimitInformation, SetInformationJobObject,
    };

    /// A job object that caps the memory of each process in it and ends them when it is closed.
    pub struct Job(HANDLE);

    impl Job {
        /// `None` without a ceiling, or when Windows refuses the job: the program then runs without one.
        pub fn new(max_mb: u64) -> Option<Job> {
            if max_mb == 0 {
                return None;
            }
            // SAFETY: no attributes and no name; the handle is owned by the returned Job.
            let job = Job(unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) });
            if job.0.is_null() {
                return None;
            }
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            limits.ProcessMemoryLimit = usize::try_from(max_mb << 20).unwrap_or(usize::MAX);
            // SAFETY: `limits` is the structure this information class expects, with its size.
            let set = unsafe { SetInformationJobObject(job.0, JobObjectExtendedLimitInformation, (&raw const limits).cast(), size_of_val(&limits) as u32) };
            (set != 0).then_some(job)
        }

        /// Puts `child` into the job. False when Windows refuses.
        pub fn adopt(&self, child: &Child) -> bool {
            // SAFETY: both handles are open: the job is owned by self, the process by `child`.
            unsafe { AssignProcessToJobObject(self.0, child.as_raw_handle()) != 0 }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: the handle came from CreateJobObjectW and is closed once, here.
            unsafe { CloseHandle(self.0) };
        }
    }
}

#[cfg(not(windows))]
mod ceiling {
    /// No memory ceiling outside Windows.
    pub struct Job;

    impl Job {
        pub fn new(_max_mb: u64) -> Option<Job> {
            None
        }

        pub fn adopt(&self, _child: &std::process::Child) -> bool {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shell(script: &str) -> Command {
        if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/d", "/c", script]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", script]);
            c
        }
    }

    #[test]
    fn a_limited_run_gives_the_exit_code_or_stops_at_the_deadline() {
        assert_eq!(run_limited(&mut shell("exit 3"), Duration::from_secs(30), 512, None).unwrap(), Ended::Exit(3));
        let sleep = if cfg!(windows) { "ping -n 30 127.0.0.1 >nul" } else { "sleep 30" };
        let start = Instant::now();
        assert_eq!(run_limited(&mut shell(sleep), Duration::from_millis(500), 512, None).unwrap(), Ended::Timeout);
        assert!(start.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn a_limited_run_writes_its_output_to_the_given_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let file = std::fs::File::create(&path).unwrap();
        assert_eq!(run_limited(&mut shell("echo hello"), Duration::from_secs(30), 0, Some(file)).unwrap(), Ended::Exit(0));
        assert!(std::fs::read_to_string(&path).unwrap().starts_with("hello"));
    }
}
