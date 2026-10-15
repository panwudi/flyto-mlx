// Where the server lives and how to reach it.
//
// Same resolution as the Swift shell it replaces, so an installed machine keeps
// working without migration:
//   base path: OMLX_BASE_PATH, then the one-line bootstrap file
//              ~/Library/Application Support/oMLX/base-path, then ~/.fmlx
//   port, api key: <base>/settings.json (server.port, auth.api_key), then the
//              OMLX_PORT / OMLX_API_KEY env overrides
//   python: OMLX_PYTHON_OVERRIDE, then ~/Code/omlx/.venv/bin/python

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde_json::Value;

pub const DEFAULT_PORT: u16 = 8000;

#[derive(Clone, Debug)]
pub struct Config {
    pub base_path: PathBuf,
    pub port: u16,
    pub api_key: Option<String>,
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

/// ~/Library/Application Support/oMLX: logs and the base-path bootstrap file.
pub fn app_support_dir() -> PathBuf {
    let dir = home().join("Library/Application Support/oMLX");
    let _ = fs::create_dir_all(&dir);
    dir
}

pub fn server_log_path() -> PathBuf {
    app_support_dir().join("logs/server.log")
}

pub fn menubar_log_path() -> PathBuf {
    app_support_dir().join("logs/menubar.log")
}

fn expand_tilde(raw: &str) -> PathBuf {
    match raw.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None if raw == "~" => home(),
        None => PathBuf::from(raw),
    }
}

pub fn base_path() -> PathBuf {
    if let Ok(v) = std::env::var("OMLX_BASE_PATH") {
        if !v.trim().is_empty() {
            return expand_tilde(v.trim());
        }
    }
    if let Ok(raw) = fs::read_to_string(app_support_dir().join("base-path")) {
        let t = raw.trim();
        if !t.is_empty() {
            return expand_tilde(t);
        }
    }
    home().join(".fmlx")
}

pub fn settings_path(base: &Path) -> PathBuf {
    base.join("settings.json")
}

pub fn load() -> Config {
    let base = base_path();
    let settings: Value = fs::read_to_string(settings_path(&base))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);

    let mut port = settings
        .pointer("/server/port")
        .and_then(Value::as_u64)
        .and_then(|p| u16::try_from(p).ok())
        .unwrap_or(DEFAULT_PORT);
    let mut api_key = settings
        .pointer("/auth/api_key")
        .and_then(Value::as_str)
        .filter(|k| !k.is_empty())
        .map(str::to_owned);

    if let Some(p) = std::env::var("OMLX_PORT").ok().and_then(|v| v.parse().ok()) {
        port = p;
    }
    if let Ok(k) = std::env::var("OMLX_API_KEY") {
        if !k.is_empty() {
            api_key = Some(k);
        }
    }
    Config { base_path: base, port, api_key }
}

/// Write server.port into settings.json, keeping every other key. Only call
/// this while the server is stopped: a running server owns the file.
pub fn save_port(base: &Path, port: u16) -> io::Result<()> {
    let path = settings_path(base);
    let mut settings: Value = fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| serde_json::json!({ "version": "1.0" }));
    if !settings.is_object() {
        settings = serde_json::json!({ "version": "1.0" });
    }
    let root = settings.as_object_mut().expect("object");
    let server = root
        .entry("server")
        .or_insert_with(|| serde_json::json!({}));
    if !server.is_object() {
        *server = serde_json::json!({});
    }
    server
        .as_object_mut()
        .expect("object")
        .insert("port".into(), Value::from(port));

    fs::create_dir_all(base)?;
    let tmp = path.with_extension("json.menubar-tmp");
    let mut text = serde_json::to_string_pretty(&settings).map_err(io::Error::other)?;
    text.push('\n');
    fs::write(&tmp, text)?;
    fs::rename(&tmp, &path)
}

pub fn python() -> PathBuf {
    if let Ok(v) = std::env::var("OMLX_PYTHON_OVERRIDE") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    home().join("Code/omlx/.venv/bin/python")
}

/// PATH for the server: ours plus the Homebrew prefixes, so ffmpeg and
/// friends resolve when the app was launched from Finder.
pub fn server_path_env() -> String {
    let mut path = std::env::var("PATH").unwrap_or_default();
    for prefix in ["/usr/local/bin", "/opt/homebrew/sbin", "/opt/homebrew/bin"] {
        if !path.split(':').any(|p| p == prefix) {
            path = if path.is_empty() { prefix.to_owned() } else { format!("{prefix}:{path}") };
        }
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_port_keeps_other_keys() {
        let dir = std::env::temp_dir().join(format!("fmlx-menubar-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            settings_path(&dir),
            r#"{"version":"1.0","server":{"host":"0.0.0.0","port":8000},"auth":{"api_key":"k"},"cache":{"x":1}}"#,
        )
        .unwrap();
        save_port(&dir, 8123).unwrap();
        let v: Value = serde_json::from_str(&fs::read_to_string(settings_path(&dir)).unwrap()).unwrap();
        assert_eq!(v.pointer("/server/port").unwrap(), 8123);
        assert_eq!(v.pointer("/server/host").unwrap(), "0.0.0.0");
        assert_eq!(v.pointer("/auth/api_key").unwrap(), "k");
        assert_eq!(v.pointer("/cache/x").unwrap(), 1);
        fs::remove_dir_all(&dir).unwrap();
    }
}
