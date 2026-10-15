// Small modal dialogs through osascript, so the menubar needs no window
// code. Every call blocks: run them off the event-loop thread.

use std::process::Command;

use crate::text;

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn run(script: &str) -> Option<String> {
    let out = Command::new("/usr/bin/osascript").args(["-e", script]).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

pub fn alert(msg: &str) {
    let script = format!(
        "display dialog {} with title \"Flyto MLX\" buttons {{{}}} default button 1 with icon note",
        quote(msg),
        quote(&text::ok())
    );
    run(&script);
}

/// True when the user pressed OK.
pub fn confirm(msg: &str) -> bool {
    let script = format!(
        "display dialog {} with title \"Flyto MLX\" buttons {{{}, {}}} default button 2 cancel button 1 with icon caution",
        quote(msg),
        quote(&text::cancel()),
        quote(&text::ok())
    );
    run(&script).is_some()
}

/// The text entered, or None on cancel.
pub fn prompt(msg: &str, default: &str) -> Option<String> {
    let script = format!(
        "text returned of (display dialog {} with title \"Flyto MLX\" default answer {} buttons {{{}, {}}} default button 2 cancel button 1)",
        quote(msg),
        quote(default),
        quote(&text::cancel()),
        quote(&text::ok())
    );
    run(&script)
}

pub fn notify(msg: &str) {
    let script = format!("display notification {} with title \"Flyto MLX\"", quote(msg));
    run(&script);
}

#[cfg(test)]
mod tests {
    #[test]
    fn quote_escapes() {
        assert_eq!(super::quote(r#"a "b" \c"#), r#""a \"b\" \\c""#);
    }
}
