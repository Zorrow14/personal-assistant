//! Supervises the Python backend, the `jarvis-backend` sidecar (`jarvis --serve` frozen by
//! PyInstaller): start it, wait for /health, show the panel, restart it once if it crashes,
//! and stop it when the app quits.
//!
//! Stopping is cooperative first: the sidecar runs with `--managed`, so a `shutdown` line on
//! its stdin (or EOF, which the OS delivers if this app dies without asking) makes it shut
//! down like Ctrl-C and let the PyInstaller bootloader clean up its temp files. Only if it's
//! still running after `STOP_GRACE` is it killed.

use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use tauri::AppHandle;
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

use crate::ui::{self, Status};

const SIDECAR: &str = "jarvis-backend";
const HEALTH_POLL: Duration = Duration::from_millis(300);
/// The sidecar forces its own exit 8 s after a shutdown request; this leaves room for that.
const STOP_GRACE: Duration = Duration::from_secs(12);
/// At most one automatic restart in this window; a second crash is reported instead.
const AUTO_RESTART_WINDOW: Duration = Duration::from_secs(10 * 60);
const LOG_TAIL_LINES: usize = 30;

pub struct Config {
    pub port: u16,
    pub wake: bool,
    pub home: PathBuf,
    pub startup_timeout: Duration,
    pub log_path: PathBuf,
    /// Where the settings live, for error hints.
    pub settings_path: PathBuf,
}

#[derive(Default)]
struct State {
    /// Bumped on every launch; events from an older child are ignored.
    generation: u64,
    child: Option<CommandChild>,
    running: bool,
    healthy: bool,
    /// We asked it to stop, so its exit is not a crash.
    stopping: bool,
    /// A restart (manual or automatic) is in progress.
    restarting: bool,
    /// The app is quitting: start nothing new.
    quitting: bool,
    last_auto_restart: Option<Instant>,
    tail: VecDeque<String>,
    log: Option<File>,
}

pub struct Backend {
    config: Config,
    state: Mutex<State>,
}

enum AfterExit {
    Expected,
    Restart,
    Fail { title: String, detail: String },
}

impl Backend {
    pub fn new(config: Config) -> Arc<Self> {
        if let Some(dir) = config.log_path.parent() {
            let _ = fs::create_dir_all(dir);
        }
        let log = File::create(&config.log_path).ok();
        Arc::new(Self { config, state: Mutex::new(State { log, ..State::default() }) })
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Record a line in the log file and the error tail (and the dev console).
    pub fn note(&self, line: &str) {
        let mut state = self.state();
        Self::record(&mut state, line);
    }

    fn record(state: &mut State, line: &str) {
        if cfg!(debug_assertions) {
            eprintln!("[backend] {line}");
        }
        if let Some(log) = state.log.as_mut() {
            let _ = writeln!(log, "{line}");
        }
        state.tail.push_back(line.to_string());
        while state.tail.len() > LOG_TAIL_LINES {
            state.tail.pop_front();
        }
    }

    fn tail(&self) -> String {
        self.state().tail.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    /// Start the backend in the background (app start-up).
    pub fn start(self: &Arc<Self>, app: &AppHandle) {
        let (this, app) = (Arc::clone(self), app.clone());
        thread::spawn(move || this.launch(&app, Status::Starting));
    }

    /// Stop the backend and start it again ("Restart backend" in the tray).
    pub fn restart(self: &Arc<Self>, app: &AppHandle) {
        {
            let mut state = self.state();
            if state.restarting {
                return;
            }
            state.restarting = true;
        }
        let (this, app) = (Arc::clone(self), app.clone());
        thread::spawn(move || {
            ui::show_status(&app, Status::Restarting);
            this.stop();
            this.state().last_auto_restart = None; // a manual restart refills the budget
            this.launch(&app, Status::Restarting);
            this.state().restarting = false;
        });
    }

    /// Spawn the sidecar and wait until it's healthy (or fails). Blocks the calling thread.
    fn launch(self: &Arc<Self>, app: &AppHandle, status: Status) {
        let port = self.config.port;
        let generation = {
            let mut state = self.state();
            state.generation += 1;
            state.running = false;
            state.healthy = false;
            state.stopping = false;
            Self::record(&mut state, &format!("--- starting the backend on 127.0.0.1:{port} ---"));
            state.generation
        };
        ui::show_status(app, status);

        if port_in_use(port) {
            self.note(&format!("127.0.0.1:{port} is already in use"));
            ui::show_error(
                app,
                "Jarvis couldn't start",
                &format!(
                    "Port {port} is already in use, perhaps by `jarvis --serve` running in a \
                     terminal. Close that, or change \"port\" in {}, then choose Restart \
                     backend from the tray icon.",
                    self.config.settings_path.display()
                ),
                "",
            );
            return;
        }

        let mut args = vec!["--managed"];
        if self.config.wake {
            args.push("--wake");
        }
        let spawned = app.shell().sidecar(SIDECAR).and_then(|command| {
            command
                .args(args)
                .current_dir(&self.config.home)
                .env("JARVIS_UI_HOST", "127.0.0.1")
                .env("JARVIS_UI_PORT", port.to_string())
                .env("PYTHONUNBUFFERED", "1")
                .spawn()
        });
        let (events, child) = match spawned {
            Ok(pair) => pair,
            Err(err) => {
                self.note(&format!("can't start the backend: {err}"));
                ui::show_error(
                    app,
                    "Jarvis couldn't start",
                    &format!("The backend program couldn't be started: {err}"),
                    &self.tail(),
                );
                return;
            }
        };
        {
            let mut state = self.state();
            if state.quitting {
                // Started just as the app quit: ask it to stop. Its stdin closes when this
                // app exits, which stops it too, so no orphan is left either way.
                let mut child = child;
                let _ = child.write(b"shutdown\n");
                return;
            }
            Self::record(&mut state, &format!("backend pid {}", child.pid()));
            state.child = Some(child);
            state.running = true;
        }
        let (this, pump_app) = (Arc::clone(self), app.clone());
        thread::spawn(move || this.pump(&pump_app, generation, events));

        let deadline = Instant::now() + self.config.startup_timeout;
        loop {
            {
                let state = self.state();
                if state.generation != generation || !state.running {
                    return; // it exited (the pump reports that) or was replaced
                }
            }
            if health_ok(port) {
                {
                    let mut state = self.state();
                    if state.generation != generation || !state.running {
                        return;
                    }
                    state.healthy = true;
                    Self::record(&mut state, "--- backend healthy; showing the panel ---");
                }
                ui::show_panel(app, port);
                return;
            }
            if Instant::now() >= deadline {
                let secs = self.config.startup_timeout.as_secs();
                self.note(&format!("no healthy /health after {secs} s; stopping it"));
                self.stop();
                ui::show_error(
                    app,
                    "Jarvis didn't finish starting",
                    &format!(
                        "The backend didn't answer on 127.0.0.1:{port} within {secs} s. \
                         Its log is below; raise \"startup_timeout_secs\" in {} if it's \
                         just slow.",
                        self.config.settings_path.display()
                    ),
                    &self.tail(),
                );
                return;
            }
            thread::sleep(HEALTH_POLL);
        }
    }

    /// Forward the sidecar's output to the log until it exits, then decide what to do.
    fn pump(
        self: &Arc<Self>,
        app: &AppHandle,
        generation: u64,
        mut events: tauri::async_runtime::Receiver<CommandEvent>,
    ) {
        let mut code = None;
        while let Some(event) = events.blocking_recv() {
            match event {
                CommandEvent::Stdout(bytes) | CommandEvent::Stderr(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    self.note(text.trim_end_matches(['\r', '\n']));
                }
                CommandEvent::Error(err) => self.note(&format!("backend output error: {err}")),
                CommandEvent::Terminated(payload) => {
                    code = payload.code;
                    break;
                }
                _ => {}
            }
        }
        self.on_exit(app, generation, code);
    }

    fn on_exit(self: &Arc<Self>, app: &AppHandle, generation: u64, code: Option<i32>) {
        let exit = code.map_or_else(|| "no exit code".to_string(), |c| format!("exit code {c}"));
        let after = {
            let mut state = self.state();
            if state.generation != generation {
                return;
            }
            let was_healthy = state.healthy;
            state.running = false;
            state.healthy = false;
            state.child = None;
            Self::record(&mut state, &format!("--- backend exited ({exit}) ---"));
            let recently = state.last_auto_restart.is_some_and(|t| t.elapsed() < AUTO_RESTART_WINDOW);
            if state.stopping {
                AfterExit::Expected
            } else if was_healthy && !recently {
                state.last_auto_restart = Some(Instant::now());
                AfterExit::Restart
            } else if was_healthy {
                AfterExit::Fail {
                    title: "Jarvis stopped".into(),
                    detail: format!(
                        "The backend crashed again ({exit}) soon after an automatic restart. \
                         Choose Restart backend from the tray icon to try again."
                    ),
                }
            } else {
                AfterExit::Fail {
                    title: "Jarvis couldn't start".into(),
                    detail: format!(
                        "The backend exited ({exit}) before it was ready. It runs in {} and \
                         reads .env there; set \"home\" in {} to use another folder.",
                        self.config.home.display(),
                        self.config.settings_path.display()
                    ),
                }
            }
        };
        match after {
            AfterExit::Expected => {}
            AfterExit::Restart => {
                self.note("restarting the backend once after an unexpected exit");
                self.launch(app, Status::Restarting);
            }
            AfterExit::Fail { title, detail } => ui::show_error(app, &title, &detail, &self.tail()),
        }
    }

    /// The app is quitting: stop the backend and never start another. Blocks.
    pub fn shutdown(&self) {
        self.state().quitting = true;
        self.stop();
    }

    /// Ask the backend to shut down, wait for it, and kill it if it overstays. Blocks.
    pub fn stop(&self) {
        let generation = {
            let mut state = self.state();
            if !state.running {
                return;
            }
            state.stopping = true;
            if let Some(child) = state.child.as_mut() {
                if let Err(err) = child.write(b"shutdown\n") {
                    Self::record(&mut state, &format!("can't ask the backend to stop: {err}"));
                }
            }
            state.generation
        };
        let deadline = Instant::now() + STOP_GRACE;
        while Instant::now() < deadline {
            {
                let state = self.state();
                if state.generation != generation || !state.running {
                    return;
                }
            }
            thread::sleep(Duration::from_millis(50));
        }
        let child = {
            let mut state = self.state();
            Self::record(&mut state, "the backend didn't stop in time; killing it");
            state.child.take()
        };
        if let Some(child) = child {
            let _ = child.kill();
        }
    }
}

/// Is something already listening on 127.0.0.1:<port>?
fn port_in_use(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

/// GET /health on the loopback panel; true once it answers 200 {"status":"ok"}.
fn health_ok(port: u16) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
    let request =
        format!("GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = String::new();
    let _ = stream.take(16 * 1024).read_to_string(&mut response);
    is_healthy_response(&response)
}

fn is_healthy_response(response: &str) -> bool {
    let status_ok = response.lines().next().is_some_and(|line| line.split(' ').nth(1) == Some("200"));
    let body = response.split("\r\n\r\n").nth(1).unwrap_or("");
    status_ok && body.replace(' ', "").contains(r#""status":"ok""#)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn recognises_a_healthy_response() {
        let ok = "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\n\r\n{\"status\":\"ok\"}";
        assert!(is_healthy_response(ok));
        assert!(!is_healthy_response("HTTP/1.1 403 Forbidden\r\n\r\n{\"status\":\"ok\"}"));
        assert!(!is_healthy_response("HTTP/1.1 200 OK\r\n\r\n{\"status\":\"starting\"}"));
        assert!(!is_healthy_response(""));
    }

    #[test]
    fn health_check_talks_http_to_the_port() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let n = conn.read(&mut buf).unwrap();
            let request = String::from_utf8_lossy(&buf[..n]).to_string();
            let body = r#"{"status":"ok"}"#;
            write!(conn, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
            request
        });
        assert!(health_ok(port));
        let request = server.join().unwrap();
        assert!(request.starts_with("GET /health HTTP/1.1\r\n"));
        assert!(request.contains(&format!("Host: 127.0.0.1:{port}\r\n")));
    }

    #[test]
    fn a_free_port_is_not_in_use() {
        let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        assert!(!port_in_use(port)); // the listener was dropped, so nothing listens now
    }
}
