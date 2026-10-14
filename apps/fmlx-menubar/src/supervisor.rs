// Server lifecycle, on its own thread so the menu never blocks.
//
//   start: port held by a healthy fmlx      -> adopt it (Running, adopted)
//          port held by something else      -> Failed
//          an older serve for this base path
//          is still exiting                  -> WaitingForOld, then spawn
//          otherwise                         -> spawn (Starting -> Running)
//   stop:  SIGTERM, wait for the PID to be gone (up to STOP_GRACE, the server
//          unloads models and flushes the SSD cache first), then SIGKILL
//   crash: the PID vanished without us asking -> restart after 5/10/20 s,
//          at most 3 times until the server has been up for a minute
//
// An adopted server is watched by PID and health only (it is not our child),
// but stop / restart / quit treat it exactly like one we spawned.

use std::fs::{self, OpenOptions};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

use crate::api::{Client, Live, Totals};
use crate::config::{self, Config};
use crate::log;
use crate::procs;

const STOP_GRACE: Duration = Duration::from_secs(120);
const KILL_GRACE: Duration = Duration::from_secs(10);
const OLD_SERVER_WAIT: Duration = Duration::from_secs(180);
const HEALTH_EVERY: Duration = Duration::from_secs(2);
const STATS_EVERY: Duration = Duration::from_secs(3);
const ALLTIME_EVERY_N_STATS: u32 = 10;
const UNHEALTHY_AFTER: u32 = 3;
const MAX_AUTO_RESTARTS: u32 = 3;
const STABLE_AFTER: Duration = Duration::from_secs(60);

/// Set from the SIGTERM / SIGINT / SIGHUP handler.
pub static QUIT_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Set once the supervisor has stopped the server and left its loop.
pub static SUPERVISOR_DONE: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, PartialEq)]
pub enum Phase {
    Stopped,
    Starting,
    Running,
    Unresponsive,
    Stopping { killing: bool },
    WaitingForOld(Vec<i32>),
    Backoff { attempt: u32, secs: u64 },
    Failed(Failure),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Failure {
    PortTaken { port: u16, pid: Option<i32> },
    OldServerStuck(Vec<i32>),
    PythonMissing(String),
    SpawnFailed(String),
    CrashLoop,
    WontExit(i32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub phase: Phase,
    pub pid: Option<i32>,
    pub adopted: bool,
    pub port: u16,
    pub has_key: bool,
    pub live: Option<Live>,
    pub session: Option<Totals>,
    pub alltime: Option<Totals>,
}

#[derive(Debug)]
pub enum Cmd {
    Start,
    Stop,
    Restart,
    ForceRestart,
    SetPort(u16),
    Quit,
}

pub enum Out {
    Snapshot(Snapshot),
    Notice(String),
    Exit,
}

enum After {
    Nothing,
    Start,
    SetPort(u16),
    Quit,
}

pub struct Supervisor {
    cfg: Config,
    client: Client,
    emit: Box<dyn Fn(Out) + Send>,

    phase: Phase,
    child: Option<Child>,
    pid: Option<i32>,
    adopted: bool,

    after: After,
    stop_deadline: Instant,
    old_deadline: Instant,
    restart_at: Option<Instant>,
    restarts: u32,
    running_since: Option<Instant>,
    health_failures: u32,
    next_health: Instant,
    next_stats: Instant,
    stats_ticks: u32,

    live: Option<Live>,
    session: Option<Totals>,
    alltime: Option<Totals>,
    last_sent: Option<Snapshot>,
    done: bool,
}

impl Supervisor {
    pub fn new(cfg: Config, emit: Box<dyn Fn(Out) + Send>) -> Self {
        let now = Instant::now();
        let client = Client::new(cfg.port, cfg.api_key.clone());
        Supervisor {
            cfg,
            client,
            emit,
            phase: Phase::Stopped,
            child: None,
            pid: None,
            adopted: false,
            after: After::Nothing,
            stop_deadline: now,
            old_deadline: now,
            restart_at: None,
            restarts: 0,
            running_since: None,
            health_failures: 0,
            next_health: now,
            next_stats: now,
            stats_ticks: 0,
            live: None,
            session: None,
            alltime: None,
            last_sent: None,
            done: false,
        }
    }

    pub fn run(mut self, rx: Receiver<Cmd>) {
        self.start();
        self.publish();
        let mut quitting = false;
        while !self.done {
            match rx.recv_timeout(Duration::from_millis(300)) {
                Ok(cmd) => {
                    log::line(&format!("command {cmd:?} in {:?}", self.phase));
                    self.handle(cmd);
                    self.publish();
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => {
                    if !quitting {
                        quitting = true;
                        self.handle(Cmd::Quit);
                    }
                }
            }
            if QUIT_REQUESTED.load(Ordering::SeqCst) && !quitting {
                quitting = true;
                log::line("signal received, stopping the server before exit");
                self.handle(Cmd::Quit);
            }
            self.tick();
            self.publish();
        }
        SUPERVISOR_DONE.store(true, Ordering::SeqCst);
        (self.emit)(Out::Exit);
    }

    fn handle(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Start => {
                if matches!(self.phase, Phase::Stopped | Phase::Failed(_)) {
                    self.restarts = 0;
                    self.start();
                }
            }
            Cmd::Stop => match self.phase {
                Phase::Backoff { .. } | Phase::WaitingForOld(_) => {
                    self.restart_at = None;
                    self.phase = Phase::Stopped;
                }
                Phase::Starting | Phase::Running | Phase::Unresponsive => {
                    self.begin_stop(After::Nothing)
                }
                _ => {}
            },
            Cmd::Restart => match self.phase {
                Phase::Starting | Phase::Running | Phase::Unresponsive => {
                    self.restarts = 0;
                    self.begin_stop(After::Start)
                }
                Phase::Stopped | Phase::Failed(_) | Phase::Backoff { .. } => {
                    self.restarts = 0;
                    self.start()
                }
                Phase::Stopping { .. } => self.after = After::Start,
                Phase::WaitingForOld(_) => {}
            },
            Cmd::ForceRestart => {
                self.force_kill_all();
                self.restarts = 0;
                self.start();
            }
            Cmd::SetPort(port) => {
                if port == self.cfg.port {
                    return;
                }
                match self.phase {
                    Phase::Starting | Phase::Running | Phase::Unresponsive => {
                        self.begin_stop(After::SetPort(port))
                    }
                    Phase::Stopping { .. } => self.after = After::SetPort(port),
                    _ => {
                        self.restart_at = None;
                        if self.apply_port(port) {
                            self.start();
                        }
                    }
                }
            }
            Cmd::Quit => match self.phase {
                Phase::Starting | Phase::Running | Phase::Unresponsive => {
                    self.begin_stop(After::Quit)
                }
                Phase::Stopping { .. } => self.after = After::Quit,
                _ => self.done = true,
            },
        }
    }

    // ---- start ---------------------------------------------------------

    fn start(&mut self) {
        self.restart_at = None;
        self.health_failures = 0;
        let port = self.cfg.port;

        if let Some(owner) = procs::port_owner(port) {
            let ours = procs::serve_pids(&self.cfg.base_path).contains(&owner);
            if self.client.health() {
                log::line(&format!("adopting running server pid {owner} on :{port}"));
                self.adopt(owner, true);
            } else if ours {
                log::line(&format!("adopting starting server pid {owner} on :{port}"));
                self.adopt(owner, false);
            } else {
                self.fail(Failure::PortTaken { port, pid: Some(owner) });
            }
            return;
        }

        let old: Vec<i32> = procs::serve_pids(&self.cfg.base_path)
            .into_iter()
            .filter(|p| procs::alive(*p))
            .collect();
        if !old.is_empty() {
            log::line(&format!("waiting for previous server {old:?} to exit"));
            self.old_deadline = Instant::now() + OLD_SERVER_WAIT;
            self.phase = Phase::WaitingForOld(old);
            return;
        }
        self.spawn();
    }

    fn adopt(&mut self, pid: i32, healthy: bool) {
        self.child = None;
        self.pid = Some(pid);
        self.adopted = true;
        if healthy {
            self.phase = Phase::Running;
            self.running_since = Some(Instant::now());
        } else {
            self.phase = Phase::Starting;
        }
        self.next_health = Instant::now();
        self.next_stats = Instant::now();
    }

    fn spawn(&mut self) {
        let python = config::python();
        if !python.exists() {
            self.fail(Failure::PythonMissing(python.display().to_string()));
            return;
        }
        let log_path = config::server_log_path();
        if let Some(dir) = log_path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let _ = fs::create_dir_all(&self.cfg.base_path);
        let (out, err) = match OpenOptions::new().create(true).append(true).open(&log_path) {
            Ok(f) => match f.try_clone() {
                Ok(g) => (Stdio::from(f), Stdio::from(g)),
                Err(_) => (Stdio::null(), Stdio::null()),
            },
            Err(_) => (Stdio::null(), Stdio::null()),
        };
        let spawned = Command::new(&python)
            .args(["-m", "omlx.cli", "serve", "--base-path"])
            .arg(&self.cfg.base_path)
            .args(["--port", &self.cfg.port.to_string()])
            .env("PATH", config::server_path_env())
            .current_dir(config::home())
            .stdin(Stdio::null())
            .stdout(out)
            .stderr(err)
            .spawn();
        match spawned {
            Ok(child) => {
                let pid = child.id() as i32;
                log::line(&format!("spawned server pid {pid} on :{}", self.cfg.port));
                self.child = Some(child);
                self.pid = Some(pid);
                self.adopted = false;
                self.phase = Phase::Starting;
                self.next_health = Instant::now() + Duration::from_secs(1);
            }
            Err(e) => self.fail(Failure::SpawnFailed(e.to_string())),
        }
    }

    // ---- stop ----------------------------------------------------------

    fn begin_stop(&mut self, after: After) {
        self.after = after;
        self.restart_at = None;
        match self.pid.filter(|p| procs::alive(*p)) {
            Some(pid) => {
                log::line(&format!("SIGTERM server pid {pid}"));
                procs::signal(pid, libc::SIGTERM);
                self.stop_deadline = Instant::now() + STOP_GRACE;
                self.phase = Phase::Stopping { killing: false };
            }
            None => self.finish_stop(),
        }
    }

    fn finish_stop(&mut self) {
        if let Some(mut c) = self.child.take() {
            let _ = c.wait();
        }
        self.pid = None;
        self.adopted = false;
        self.running_since = None;
        self.clear_stats();
        match std::mem::replace(&mut self.after, After::Nothing) {
            After::Nothing => self.phase = Phase::Stopped,
            After::Start => self.start(),
            After::SetPort(port) => {
                self.phase = Phase::Stopped;
                if self.apply_port(port) {
                    self.start();
                }
            }
            After::Quit => {
                self.phase = Phase::Stopped;
                self.done = true;
            }
        }
    }

    fn force_kill_all(&mut self) {
        let mut pids = procs::serve_pids(&self.cfg.base_path);
        if let Some(p) = self.pid {
            if !pids.contains(&p) {
                pids.push(p);
            }
        }
        // Only kill what is ours: a foreign process on the port is not.
        log::line(&format!("SIGKILL {pids:?}"));
        for p in &pids {
            procs::signal(*p, libc::SIGKILL);
        }
        if let Some(mut c) = self.child.take() {
            let _ = c.wait();
        }
        let left = procs::wait_gone(&pids, KILL_GRACE);
        if !left.is_empty() {
            log::line(&format!("still alive after SIGKILL: {left:?}"));
        }
        self.pid = None;
        self.adopted = false;
        self.running_since = None;
        self.after = After::Nothing;
        self.clear_stats();
        self.phase = Phase::Stopped;
    }

    fn apply_port(&mut self, port: u16) -> bool {
        match config::save_port(&self.cfg.base_path, port) {
            Ok(()) => {
                self.cfg.port = port;
                self.client = Client::new(port, self.cfg.api_key.clone());
                true
            }
            Err(e) => {
                (self.emit)(Out::Notice(format!("settings.json: {e}")));
                false
            }
        }
    }

    // ---- monitor -------------------------------------------------------

    fn process_gone(&mut self) -> bool {
        if let Some(c) = self.child.as_mut() {
            if let Ok(Some(status)) = c.try_wait() {
                log::line(&format!("server exited: {status}"));
                self.child = None;
            }
        }
        self.pid.is_none_or(|p| !procs::alive(p))
    }

    fn tick(&mut self) {
        let now = Instant::now();
        match self.phase.clone() {
            Phase::Stopping { killing } => {
                if self.process_gone() {
                    self.finish_stop();
                } else if now >= self.stop_deadline {
                    let pid = self.pid.unwrap_or(0);
                    if !killing {
                        log::line(&format!("pid {pid} still alive after grace, SIGKILL"));
                        procs::signal(pid, libc::SIGKILL);
                        self.stop_deadline = now + KILL_GRACE;
                        self.phase = Phase::Stopping { killing: true };
                    } else {
                        self.after = After::Nothing;
                        self.fail(Failure::WontExit(pid));
                    }
                }
            }
            Phase::WaitingForOld(pids) => {
                let left: Vec<i32> = pids.into_iter().filter(|p| procs::alive(*p)).collect();
                if left.is_empty() {
                    self.start();
                } else if now >= self.old_deadline {
                    self.fail(Failure::OldServerStuck(left));
                } else {
                    self.phase = Phase::WaitingForOld(left);
                }
            }
            Phase::Backoff { .. } => {
                if self.restart_at.is_some_and(|t| now >= t) {
                    self.start();
                }
            }
            Phase::Starting | Phase::Running | Phase::Unresponsive => {
                if self.process_gone() {
                    self.on_unexpected_exit();
                    return;
                }
                if now >= self.next_health {
                    self.next_health = now + HEALTH_EVERY;
                    self.check_health(now);
                }
                if self.phase == Phase::Running && now >= self.next_stats {
                    self.next_stats = now + STATS_EVERY;
                    self.refresh_stats();
                }
                if self.running_since.is_some_and(|t| now.duration_since(t) >= STABLE_AFTER) {
                    self.restarts = 0;
                }
            }
            Phase::Stopped | Phase::Failed(_) => {}
        }
    }

    fn check_health(&mut self, now: Instant) {
        if self.client.health() {
            self.health_failures = 0;
            if self.phase != Phase::Running {
                log::line("server healthy");
                self.phase = Phase::Running;
                self.next_stats = now;
                if self.running_since.is_none() {
                    self.running_since = Some(now);
                }
            }
        } else {
            self.health_failures += 1;
            if self.phase == Phase::Running && self.health_failures >= UNHEALTHY_AFTER {
                log::line("server unresponsive");
                self.phase = Phase::Unresponsive;
            }
        }
    }

    fn refresh_stats(&mut self) {
        if !self.client.has_key() {
            return;
        }
        if let Some((session, live)) = self.client.session() {
            self.session = Some(session);
            self.live = Some(live);
        }
        if self.stats_ticks % ALLTIME_EVERY_N_STATS == 0 {
            if let Some(all) = self.client.alltime() {
                self.alltime = Some(all);
            }
        }
        self.stats_ticks = self.stats_ticks.wrapping_add(1);
    }

    fn on_unexpected_exit(&mut self) {
        log::line(&format!("server pid {:?} went away unexpectedly", self.pid));
        self.pid = None;
        self.adopted = false;
        self.running_since = None;
        self.clear_stats();
        if self.restarts >= MAX_AUTO_RESTARTS {
            self.fail(Failure::CrashLoop);
            return;
        }
        self.restarts += 1;
        let secs = 5u64 << (self.restarts - 1);
        self.restart_at = Some(Instant::now() + Duration::from_secs(secs));
        self.phase = Phase::Backoff { attempt: self.restarts, secs };
    }

    fn fail(&mut self, f: Failure) {
        log::line(&format!("failed: {f:?}"));
        self.phase = Phase::Failed(f);
    }

    fn clear_stats(&mut self) {
        self.live = None;
        self.session = None;
        self.stats_ticks = 0;
    }

    fn publish(&mut self) {
        let snap = Snapshot {
            phase: self.phase.clone(),
            pid: self.pid,
            adopted: self.adopted,
            port: self.cfg.port,
            has_key: self.client.has_key(),
            live: self.live.clone(),
            session: self.session.clone(),
            alltime: self.alltime.clone(),
        };
        if self.last_sent.as_ref() != Some(&snap) {
            self.last_sent = Some(snap.clone());
            (self.emit)(Out::Snapshot(snap));
        }
    }
}
