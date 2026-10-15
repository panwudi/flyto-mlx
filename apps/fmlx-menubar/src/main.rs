// Flyto MLX menubar: a status-bar menu that starts, adopts and supervises the
// local `omlx.cli serve` process. Replaces the Swift shell in apps/omlx-mac.
//
// Threads: the tao event loop owns the status item and menu (AppKit wants the
// main thread); the supervisor thread owns the server process and sends menu
// snapshots back through the event-loop proxy; dialogs run on short-lived
// threads because osascript blocks.

mod api;
mod config;
mod dialogs;
mod icons;
mod log;
mod login_item;
mod procs;
mod supervisor;
mod text;

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Sender};
use std::thread;
use std::time::Duration;

use tao::event::{Event, StartCause};
use tao::event_loop::{ControlFlow, EventLoopBuilder, EventLoopProxy};
use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS, EventLoopWindowTargetExtMacOS};
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{TrayIcon, TrayIconBuilder};

use supervisor::{Cmd, Out, Phase, Snapshot, Supervisor, QUIT_REQUESTED, SUPERVISOR_DONE};

enum UserEvent {
    Out(Out),
    Menu(MenuId),
    CheckVisible { recreated: bool },
    /// Header text to show right after a click, before the supervisor's
    /// next snapshot arrives.
    Pending(String),
}

struct Ui {
    tray: TrayIcon,
    header: MenuItem,
    models: MenuItem,
    requests: MenuItem,
    memory: MenuItem,
    start: MenuItem,
    stop: MenuItem,
    restart: MenuItem,
    force: MenuItem,
    stats: Submenu,
    stats_text: Vec<String>,
    admin: MenuItem,
    chat: MenuItem,
    log: MenuItem,
    port: MenuItem,
    login: CheckMenuItem,
    about: MenuItem,
    quit: MenuItem,
    serving: Option<bool>,
}

fn disabled(text: &str) -> MenuItem {
    MenuItem::new(text, false, None)
}

impl Ui {
    fn new() -> Ui {
        let header = disabled("● …");
        let models = disabled(&text::models_line(&[]));
        let requests = disabled("-");
        let memory = disabled("-");
        let start = MenuItem::new(text::start(), false, None);
        let stop = MenuItem::new(text::stop(), false, None);
        let restart = MenuItem::new(text::restart(), false, None);
        let force = MenuItem::new(text::force_restart(), false, None);
        let stats = Submenu::new(text::stats(), true);
        let admin = MenuItem::new(text::admin(), false, None);
        let chat = MenuItem::new(text::chat(), false, None);
        let log = MenuItem::new(text::open_log(), true, None);
        let port = MenuItem::new(text::set_port(), true, None);
        let login = CheckMenuItem::new(text::launch_at_login(), true, login_item::enabled(), None);
        let about = MenuItem::new(text::about(), true, None);
        let quit = MenuItem::new(text::quit(), true, None);

        let menu = Menu::new();
        let sep = PredefinedMenuItem::separator;
        menu.append_items(&[
            &header,
            &models,
            &requests,
            &memory,
            &sep(),
            &start,
            &stop,
            &restart,
            &force,
            &sep(),
            &stats,
            &sep(),
            &admin,
            &chat,
            &log,
            &port,
            &login,
            &sep(),
            &about,
            &quit,
        ])
        .expect("menu builds");

        let tray = TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_icon_templated(icons::menubar(false))
            .with_tooltip("Flyto MLX")
            .build()
            .expect("status item");

        Ui {
            tray,
            header,
            models,
            requests,
            memory,
            start,
            stop,
            restart,
            force,
            stats,
            stats_text: Vec::new(),
            admin,
            chat,
            log,
            port,
            login,
            about,
            quit,
            serving: None,
        }
    }

    fn update(&mut self, s: &Snapshot) {
        let mut header = text::header(s);
        if s.adopted && s.phase == Phase::Running {
            header = format!("{header} ({})", text::adopted_note());
        }
        self.header.set_text(header);

        let running = s.phase == Phase::Running;
        let live_like = matches!(s.phase, Phase::Starting | Phase::Running | Phase::Unresponsive);
        let stopped_like = matches!(s.phase, Phase::Stopped | Phase::Failed(_));
        self.start.set_enabled(stopped_like);
        self.stop.set_enabled(live_like || matches!(s.phase, Phase::Backoff { .. } | Phase::WaitingForOld(_)));
        self.restart.set_enabled(live_like || matches!(s.phase, Phase::Failed(_) | Phase::Backoff { .. }));
        self.force.set_enabled(!matches!(s.phase, Phase::Stopped));
        self.admin.set_enabled(running);
        self.chat.set_enabled(running);

        match &s.live {
            Some(l) if running => {
                self.models.set_text(text::models_line(&l.models));
                self.requests.set_text(text::requests_line(l.active, l.waiting));
                self.memory.set_text(text::memory_line(l.memory_used, l.memory_max));
            }
            _ => {
                self.models.set_text(text::models_line(&[]));
                self.requests.set_text("-");
                self.memory.set_text("-");
            }
        }

        let mut lines = text::stats_lines(s.session.as_ref(), s.alltime.as_ref());
        if !running || lines.is_empty() {
            lines = vec![text::stats_unavailable(s.has_key, running)];
        }
        if lines != self.stats_text {
            while self.stats.remove_at(0).is_some() {}
            for line in &lines {
                let _ = self.stats.append(&disabled(line));
            }
            self.stats_text = lines;
        }

        let serving = matches!(s.phase, Phase::Running | Phase::Unresponsive);
        if self.serving != Some(serving) {
            let _ = self.tray.set_icon_templated(Some(icons::menubar(serving)));
            self.serving = Some(serving);
        }
    }

    fn is_hidden(&self) -> bool {
        use objc2::MainThreadMarker;
        use objc2_app_kit::NSWindowOcclusionState;
        let Some(mtm) = MainThreadMarker::new() else { return false };
        let Some(item) = self.tray.ns_status_item() else { return true };
        let Some(window) = item.button(mtm).and_then(|b| b.window()) else { return true };
        let visible = item.isVisible()
            && window.isVisible()
            && window.occlusionState().contains(NSWindowOcclusionState::Visible);
        !visible
    }
}

fn open_url(port: u16, path: &str) {
    let _ = Command::new("/usr/bin/open").arg(format!("http://127.0.0.1:{port}{path}")).spawn();
}

fn busy_count(last: &Option<Snapshot>) -> u64 {
    last.as_ref()
        .and_then(|s| s.live.as_ref())
        .map(|l| l.active + l.waiting)
        .unwrap_or(0)
}

/// Show `pending` in the header and send `cmd`; when requests are in flight,
/// ask first (the header changes only after the user confirms).
fn confirm_then(
    tx: &Sender<Cmd>,
    proxy: &EventLoopProxy<UserEvent>,
    busy: u64,
    action: String,
    pending: String,
    cmd: Cmd,
) {
    if busy == 0 {
        let _ = proxy.send_event(UserEvent::Pending(pending));
        let _ = tx.send(cmd);
        return;
    }
    let tx = tx.clone();
    let proxy = proxy.clone();
    thread::spawn(move || {
        if dialogs::confirm(&text::confirm_busy(&action, busy)) {
            let _ = proxy.send_event(UserEvent::Pending(pending));
            let _ = tx.send(cmd);
        }
    });
}

fn handle_menu(
    id: &MenuId,
    ui: &mut Ui,
    tx: &Sender<Cmd>,
    proxy: &EventLoopProxy<UserEvent>,
    last: &Option<Snapshot>,
) {
    let port = last.as_ref().map(|s| s.port).unwrap_or(config::DEFAULT_PORT);
    let busy = busy_count(last);
    if id == ui.start.id() {
        ui.header.set_text(text::header(&Snapshot {
            phase: Phase::Starting,
            pid: None,
            adopted: false,
            port,
            has_key: false,
            live: None,
            session: None,
            alltime: None,
        }));
        let _ = tx.send(Cmd::Start);
    } else if id == ui.stop.id() {
        confirm_then(tx, proxy, busy, text::action_stop(), text::pending_stop(), Cmd::Stop);
    } else if id == ui.restart.id() {
        confirm_then(tx, proxy, busy, text::action_restart(), text::pending_restart(), Cmd::Restart);
    } else if id == ui.force.id() {
        confirm_then(tx, proxy, busy, text::action_restart(), text::pending_force(), Cmd::ForceRestart);
    } else if id == ui.admin.id() {
        open_url(port, "/admin");
    } else if id == ui.chat.id() {
        open_url(port, "/admin/chat");
    } else if id == ui.log.id() {
        let _ = Command::new("/usr/bin/open").args(["-a", "Console"]).arg(config::server_log_path()).spawn();
    } else if id == ui.port.id() {
        let tx = tx.clone();
        let proxy = proxy.clone();
        thread::spawn(move || {
            let Some(raw) = dialogs::prompt(&text::port_prompt(port), &port.to_string()) else { return };
            match raw.trim().parse::<u16>() {
                Ok(p) if p > 0 => {
                    if p != port {
                        let _ = proxy.send_event(UserEvent::Pending(text::pending_port(p)));
                        let _ = tx.send(Cmd::SetPort(p));
                    }
                }
                _ => dialogs::alert(&text::port_invalid()),
            }
        });
    } else if id == ui.login.id() {
        // The check mark already flipped; make the system match it.
        let want = ui.login.is_checked();
        if let Err(e) = login_item::set(want) {
            ui.login.set_checked(login_item::enabled());
            thread::spawn(move || dialogs::alert(&text::login_item_failed(&e)));
        }
    } else if id == ui.about.id() {
        thread::spawn(|| dialogs::alert(&text::about_body(env!("CARGO_PKG_VERSION"))));
    } else if id == ui.quit.id() {
        confirm_then(tx, proxy, busy, text::action_quit(), text::pending_quit(), Cmd::Quit);
    }
}

/// First launch of this app on a machine: register the login item once.
/// Afterwards the menu check box (or System Settings) owns it.
fn register_login_item_once() {
    // A bare dev binary is not an app bundle; never register it.
    let in_bundle = std::env::current_exe()
        .map(|p| p.to_string_lossy().contains(".app/Contents/MacOS/"))
        .unwrap_or(false);
    if !in_bundle || std::env::var_os("FMLX_MENUBAR_NO_LOGIN_ITEM").is_some() {
        return;
    }
    let marker = config::app_support_dir().join("menubar-login-item-initialised");
    if marker.exists() {
        return;
    }
    if login_item::enabled() || login_item::set(true).is_ok() {
        let _ = fs::write(marker, "1\n");
    }
}

extern "C" fn on_signal(_: libc::c_int) {
    QUIT_REQUESTED.store(true, Ordering::SeqCst);
}

fn install_signal_handlers() {
    for sig in [libc::SIGTERM, libc::SIGINT, libc::SIGHUP] {
        // SAFETY: the handler only stores to an atomic.
        unsafe {
            libc::signal(sig, on_signal as *const () as libc::sighandler_t);
        }
    }
}

/// `fmlx-menubar --render-app-icon <dir>` writes the PNG sizes iconutil
/// needs; scripts/build.sh turns them into AppIcon.icns.
fn render_app_icon(dir: PathBuf) {
    fs::create_dir_all(&dir).expect("iconset dir");
    for (name, size) in [
        ("icon_16x16.png", 16),
        ("icon_16x16@2x.png", 32),
        ("icon_32x32.png", 32),
        ("icon_32x32@2x.png", 64),
        ("icon_128x128.png", 128),
        ("icon_128x128@2x.png", 256),
        ("icon_256x256.png", 256),
        ("icon_256x256@2x.png", 512),
        ("icon_512x512.png", 512),
        ("icon_512x512@2x.png", 1024),
    ] {
        fs::write(dir.join(name), icons::app_icon_png(size)).expect("write png");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == "--render-app-icon" {
        render_app_icon(PathBuf::from(&args[2]));
        return;
    }

    install_signal_handlers();
    let cfg = config::load();
    log::line(&format!(
        "menubar {} starting: base {} port {}",
        env!("CARGO_PKG_VERSION"),
        cfg.base_path.display(),
        cfg.port
    ));

    let mut event_loop = EventLoopBuilder::<UserEvent>::with_user_event().build();
    // Regular first, Accessory once the status item exists: setting
    // Accessory up front (or LSUIElement) can leave the item hidden on Tahoe.
    event_loop.set_activation_policy(ActivationPolicy::Regular);

    let menu_proxy = event_loop.create_proxy();
    MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
        let _ = menu_proxy.send_event(UserEvent::Menu(e.id));
    }));

    let (tx, rx) = mpsc::channel::<Cmd>();
    let sup_proxy = event_loop.create_proxy();
    let sup = Supervisor::new(
        cfg,
        Box::new(move |o| {
            let _ = sup_proxy.send_event(UserEvent::Out(o));
        }),
    );
    thread::spawn(move || sup.run(rx));

    let proxy: EventLoopProxy<UserEvent> = event_loop.create_proxy();
    let mut ui: Option<Ui> = None;
    let mut last: Option<Snapshot> = None;

    event_loop.run(move |event, target, control_flow| {
        *control_flow = ControlFlow::Wait;
        match event {
            Event::NewEvents(StartCause::Init) => {
                let mut fresh = Ui::new();
                if let Some(s) = &last {
                    fresh.update(s);
                }
                ui = Some(fresh);
                target.set_activation_policy_at_runtime(ActivationPolicy::Accessory);
                register_login_item_once();
                let p = proxy.clone();
                thread::spawn(move || {
                    thread::sleep(Duration::from_secs(3));
                    let _ = p.send_event(UserEvent::CheckVisible { recreated: false });
                });
            }
            Event::UserEvent(UserEvent::Out(out)) => match out {
                Out::Snapshot(s) => {
                    if let Some(u) = ui.as_mut() {
                        u.update(&s);
                    }
                    last = Some(s);
                }
                Out::Notice(msg) => {
                    thread::spawn(move || dialogs::notify(&msg));
                }
                Out::Exit => {
                    log::line("exiting");
                    *control_flow = ControlFlow::Exit;
                }
            },
            Event::UserEvent(UserEvent::Menu(id)) => {
                if let Some(u) = ui.as_mut() {
                    handle_menu(&id, u, &tx, &proxy, &last);
                }
            }
            Event::UserEvent(UserEvent::CheckVisible { recreated }) => {
                let hidden = ui.as_ref().is_some_and(Ui::is_hidden);
                if !hidden {
                    return;
                }
                if recreated {
                    // No dialog: a locked screen or sleeping display reads as
                    // hidden too, and m5max usually runs unattended.
                    log::line("status item still reads as hidden after re-creating it");
                } else {
                    log::line("status item hidden, re-creating it");
                    ui = None;
                    let mut fresh = Ui::new();
                    if let Some(s) = &last {
                        fresh.update(s);
                    }
                    ui = Some(fresh);
                    let p = proxy.clone();
                    thread::spawn(move || {
                        thread::sleep(Duration::from_secs(1));
                        let _ = p.send_event(UserEvent::CheckVisible { recreated: true });
                    });
                }
            }
            Event::UserEvent(UserEvent::Pending(header)) => {
                if let Some(u) = ui.as_mut() {
                    u.header.set_text(header);
                }
            }
            Event::LoopDestroyed => {
                // Reached via NSApp terminate (osascript "quit app", logout)
                // as well as our own exit. Stop the server first unless the
                // supervisor already did; bounded so logout is not held up
                // forever (a server that got SIGTERM finishes on its own, and
                // the next launch adopts or waits for it).
                if !SUPERVISOR_DONE.load(Ordering::SeqCst) {
                    log::line("terminate requested, stopping the server first");
                    let _ = tx.send(Cmd::Quit);
                    let deadline = std::time::Instant::now() + Duration::from_secs(60);
                    while !SUPERVISOR_DONE.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                        thread::sleep(Duration::from_millis(200));
                    }
                }
                log::line("event loop destroyed");
            }
            _ => {}
        }
    });
}
