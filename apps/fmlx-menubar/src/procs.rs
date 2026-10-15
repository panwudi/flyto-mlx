// Process helpers. Liveness is always judged by PID (kill(pid, 0)), never by
// the port: a stopping server closes its socket within a second but can keep
// tens of GB of memory for another minute while it unloads and flushes the
// SSD cache. Starting a new server in that window makes the new one count the
// old one as external memory and pin its ceiling low.

use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

pub fn alive(pid: i32) -> bool {
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 only checks existence and permission.
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

pub fn signal(pid: i32, sig: i32) {
    if pid > 0 {
        // SAFETY: plain kill(2) on a pid we found ourselves.
        unsafe {
            libc::kill(pid, sig);
        }
    }
}

/// PID listening on the TCP port, if any.
pub fn port_owner(port: u16) -> Option<i32> {
    let out = Command::new("/usr/sbin/lsof")
        .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-t"])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.trim().parse().ok())
}

/// Every `python -m omlx.cli serve --base-path <base>` process, alive or
/// shutting down. Only argv[0] that is a python interpreter counts, so a
/// shell whose command line merely mentions the pattern is not matched.
pub fn serve_pids(base: &Path) -> Vec<i32> {
    let Ok(out) = Command::new("/bin/ps").args(["-axww", "-o", "pid=,args="]).output() else {
        return Vec::new();
    };
    let base = base.to_string_lossy();
    let me = std::process::id() as i32;
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let line = line.trim_start();
            let (pid, args) = line.split_once(' ')?;
            let pid: i32 = pid.parse().ok()?;
            is_serve_cmdline(args, &base).then_some(pid).filter(|p| *p != me)
        })
        .collect()
}

fn is_serve_cmdline(args: &str, base: &str) -> bool {
    let mut parts = args.split_whitespace();
    let argv0 = parts.next().unwrap_or("");
    let exe = argv0.rsplit('/').next().unwrap_or("");
    if !exe.starts_with("python") {
        return false;
    }
    let rest: Vec<&str> = parts.collect();
    let serve = rest.windows(3).any(|w| w == ["-m", "omlx.cli", "serve"]);
    let same_base = rest.windows(2).any(|w| {
        w[0] == "--base-path" && w[1].trim_end_matches('/') == base.trim_end_matches('/')
    });
    serve && same_base
}

/// Wait until every pid is gone or the deadline passes. Returns the pids
/// still alive.
pub fn wait_gone(pids: &[i32], timeout: Duration) -> Vec<i32> {
    let deadline = Instant::now() + timeout;
    loop {
        let left: Vec<i32> = pids.iter().copied().filter(|p| alive(*p)).collect();
        if left.is_empty() || Instant::now() >= deadline {
            return left;
        }
        thread::sleep(Duration::from_millis(200));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_only_python_serve_for_the_same_base() {
        let base = "/Users/a/.fmlx";
        assert!(is_serve_cmdline(
            "/Users/a/Code/omlx/.venv/bin/python -m omlx.cli serve --base-path /Users/a/.fmlx --port 8000",
            base
        ));
        assert!(is_serve_cmdline(
            "python3.12 -m omlx.cli serve --port 8000 --base-path /Users/a/.fmlx/",
            base
        ));
        assert!(!is_serve_cmdline(
            "/bin/bash -c pgrep -f omlx.cli serve --base-path /Users/a/.fmlx",
            base
        ));
        assert!(!is_serve_cmdline(
            "/x/python -m omlx.cli serve --base-path /tmp/other --port 8001",
            base
        ));
    }

    #[test]
    fn own_pid_is_alive_and_bogus_is_not() {
        assert!(alive(std::process::id() as i32));
        assert!(!alive(0));
    }
}
