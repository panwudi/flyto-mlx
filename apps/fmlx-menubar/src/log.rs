// Append-only event log for the menubar itself:
// ~/Library/Application Support/oMLX/logs/menubar.log

use std::fs::{self, OpenOptions};
use std::io::Write;

use crate::config;

fn timestamp() -> String {
    // SAFETY: localtime_r / strftime on stack buffers we own.
    unsafe {
        let t = libc::time(std::ptr::null_mut());
        let mut tm: libc::tm = std::mem::zeroed();
        libc::localtime_r(&t, &mut tm);
        let mut buf = [0u8; 32];
        let n = libc::strftime(buf.as_mut_ptr().cast(), buf.len(), c"%Y-%m-%d %H:%M:%S".as_ptr(), &tm);
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }
}

pub fn line(msg: &str) {
    let path = config::menubar_log_path();
    if let Some(dir) = path.parent() {
        let _ = fs::create_dir_all(dir);
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{} {msg}", timestamp());
    }
    eprintln!("[fmlx-menubar] {msg}");
}
