// Menu wording in Chinese or English, picked once from the macOS language
// list (first entry zh-* means Chinese).

use std::process::Command;
use std::sync::OnceLock;

use crate::api::Totals;
use crate::supervisor::{Failure, Phase, Snapshot};

static ZH: OnceLock<bool> = OnceLock::new();

pub fn zh() -> bool {
    *ZH.get_or_init(|| {
        if let Ok(v) = std::env::var("FMLX_MENUBAR_LANG") {
            return v.starts_with("zh");
        }
        Command::new("/usr/bin/defaults")
            .args(["read", "-g", "AppleLanguages"])
            .output()
            .ok()
            .and_then(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .split('"')
                    .nth(1)
                    .map(|first| first.starts_with("zh"))
            })
            .unwrap_or(false)
    })
}

fn pick(zh_text: &str, en_text: &str) -> String {
    if zh() { zh_text.to_owned() } else { en_text.to_owned() }
}

pub fn start() -> String { pick("启动服务", "Start Server") }
pub fn stop() -> String { pick("停止服务", "Stop Server") }
pub fn restart() -> String { pick("重启服务", "Restart Server") }
pub fn force_restart() -> String { pick("强制重启 (立即结束进程)", "Force Restart (kill now)") }
pub fn admin() -> String { pick("打开管理页", "Open Admin Panel") }
pub fn chat() -> String { pick("打开对话", "Open Chat") }
pub fn open_log() -> String { pick("查看服务日志", "Show Server Log") }
pub fn set_port() -> String { pick("修改端口…", "Set Port…") }
pub fn launch_at_login() -> String { pick("开机自动启动", "Launch at Login") }
pub fn about() -> String { pick("关于 Flyto MLX", "About Flyto MLX") }
pub fn quit() -> String { pick("退出 Flyto MLX", "Quit Flyto MLX") }
pub fn stats() -> String { pick("服务数据", "Serving Stats") }

pub fn header(s: &Snapshot) -> String {
    let port = s.port;
    match &s.phase {
        Phase::Stopped => pick("● 已停止", "● Stopped"),
        Phase::Starting => pick(&format!("● 正在启动… (端口 {port})"), &format!("● Starting… (port {port})")),
        Phase::Running => {
            let pid = s.pid.unwrap_or(0);
            pick(&format!("● 运行中 · 端口 {port} · pid {pid}"), &format!("● Running · port {port} · pid {pid}"))
        }
        Phase::Unresponsive => pick("● 无响应 (可能正忙, 可重启)", "● Not responding (busy? restart available)"),
        Phase::Stopping { killing: false } => {
            pick("● 正在停止… (卸载模型, 缓存写盘)", "● Stopping… (unloading, flushing cache)")
        }
        Phase::Stopping { killing: true } => pick("● 正在强制结束…", "● Killing…"),
        Phase::WaitingForOld(pids) => pick(
            &format!("● 等旧进程 {pids:?} 退出后启动…"),
            &format!("● Waiting for old server {pids:?} to exit…"),
        ),
        Phase::Backoff { attempt, secs } => pick(
            &format!("● 服务意外退出, {secs} 秒后重启 (第 {attempt} 次)"),
            &format!("● Server exited, restarting in {secs}s (attempt {attempt})"),
        ),
        Phase::Failed(f) => format!("● {}", failure(f)),
    }
}

pub fn failure(f: &Failure) -> String {
    match f {
        Failure::PortTaken { port, pid } => {
            let who = pid.map(|p| format!("pid {p}")).unwrap_or_else(|| "?".into());
            pick(
                &format!("端口 {port} 被其他程序占用 ({who}), 可以改端口"),
                &format!("Port {port} is used by another program ({who}); change the port"),
            )
        }
        Failure::OldServerStuck(pids) => pick(
            &format!("旧服务进程 {pids:?} 一直没退出, 可以强制重启"),
            &format!("Old server {pids:?} did not exit; use Force Restart"),
        ),
        Failure::PythonMissing(p) => pick(&format!("找不到 Python: {p}"), &format!("Python not found: {p}")),
        Failure::SpawnFailed(e) => pick(&format!("启动失败: {e}"), &format!("Start failed: {e}")),
        Failure::CrashLoop => pick(
            "服务连续退出 3 次, 已停止自动重启, 请看服务日志",
            "Server exited 3 times; auto-restart stopped, see the server log",
        ),
        Failure::WontExit(pid) => pick(
            &format!("进程 {pid} 结束不掉"),
            &format!("Process {pid} would not exit"),
        ),
    }
}

pub fn adopted_note() -> String {
    pick("已接管正在运行的服务", "Attached to the server that was already running")
}

pub fn models_line(models: &[String]) -> String {
    if models.is_empty() {
        pick("模型: 未加载", "Models: none loaded")
    } else {
        pick(&format!("模型: {}", models.join(", ")), &format!("Models: {}", models.join(", ")))
    }
}

pub fn requests_line(active: u64, waiting: u64) -> String {
    pick(
        &format!("请求: 处理中 {active} · 排队 {waiting}"),
        &format!("Requests: {active} running · {waiting} waiting"),
    )
}

pub fn memory_line(used: u64, max: u64) -> String {
    let gb = |b: u64| b as f64 / 1e9;
    pick(
        &format!("模型内存: {:.1} / {:.1} GB", gb(used), gb(max)),
        &format!("Model memory: {:.1} / {:.1} GB", gb(used), gb(max)),
    )
}

pub fn stats_unavailable(has_key: bool, running: bool) -> String {
    if !running {
        pick("服务未运行", "Server is off")
    } else if !has_key {
        pick("settings.json 里没有 API key, 无法读取", "No API key in settings.json")
    } else {
        pick("正在读取…", "Loading…")
    }
}

pub fn stats_lines(session: Option<&Totals>, alltime: Option<&Totals>) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(s) = session {
        out.push(pick("本次启动以来", "This session"));
        out.push(row(&pick("处理 token", "Tokens processed"), compact(s.prompt_tokens)));
        out.push(row(&pick("缓存命中 token", "Cached tokens"), compact(s.cached_tokens)));
        out.push(row(&pick("缓存命中率", "Cache efficiency"), percent(s.cache_efficiency)));
        out.push(row(&pick("平均预填速度", "Avg prefill speed"), tps(s.avg_prefill_tps)));
        out.push(row(&pick("平均出字速度", "Avg generation speed"), tps(s.avg_generation_tps)));
    }
    if let Some(a) = alltime {
        out.push(pick("累计", "All time"));
        out.push(row(&pick("处理 token", "Tokens processed"), compact(a.prompt_tokens)));
        out.push(row(&pick("缓存命中 token", "Cached tokens"), compact(a.cached_tokens)));
        out.push(row(&pick("缓存命中率", "Cache efficiency"), percent(a.cache_efficiency)));
        out.push(row(&pick("请求数", "Requests"), compact(a.requests)));
    }
    out
}

fn row(label: &str, value: String) -> String {
    format!("    {label}: {value}")
}

fn compact(v: Option<u64>) -> String {
    match v {
        None => "-".into(),
        Some(n) if n >= 1_000_000_000 => format!("{:.1}B", n as f64 / 1e9),
        Some(n) if n >= 1_000_000 => format!("{:.1}M", n as f64 / 1e6),
        Some(n) if n >= 1_000 => format!("{:.1}K", n as f64 / 1e3),
        Some(n) => n.to_string(),
    }
}

fn percent(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}%")).unwrap_or_else(|| "-".into())
}

fn tps(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1} tok/s")).unwrap_or_else(|| "-".into())
}

pub fn confirm_busy(action: &str, active: u64) -> String {
    pick(
        &format!("现在有 {active} 个请求正在处理, {action}会中断它们. 确定吗?"),
        &format!("{active} request(s) are running and will be cut off if you {action}. Continue?"),
    )
}

pub fn action_restart() -> String { pick("重启", "restart") }
pub fn action_stop() -> String { pick("停止", "stop") }
pub fn action_quit() -> String { pick("退出", "quit") }
pub fn ok() -> String { pick("确定", "OK") }
pub fn cancel() -> String { pick("取消", "Cancel") }
pub fn port_prompt(current: u16) -> String {
    pick(
        &format!("新的服务端口 (现在是 {current}). 修改后服务会重启."),
        &format!("New server port (currently {current}). The server restarts to apply it."),
    )
}
pub fn port_invalid() -> String { pick("端口要是 1 到 65535 之间的整数.", "Enter a whole number from 1 to 65535.") }
pub fn about_body(version: &str) -> String {
    pick(
        &format!("Flyto MLX 菜单栏 {version}\n负责启动和看护本机的 fmlx 服务."),
        &format!("Flyto MLX menubar {version}\nStarts and supervises the local fmlx server."),
    )
}
pub fn login_item_failed(e: &str) -> String {
    pick(&format!("开机自启设置失败: {e}"), &format!("Could not change Launch at Login: {e}"))
}
