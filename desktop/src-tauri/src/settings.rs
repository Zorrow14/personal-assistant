//! Desktop settings: `desktop.json` in the app's config directory, overridable by env vars.
//!
//! ```json
//! { "port": 8000, "hotkey": "Ctrl+Alt+J", "wake": false, "home": null, "startup_timeout_secs": 180 }
//! ```
//!
//! Env overrides (handy with `tauri dev`): JARVIS_DESKTOP_PORT, JARVIS_DESKTOP_HOTKEY,
//! JARVIS_DESKTOP_WAKE, JARVIS_HOME.

use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const FILE_NAME: &str = "desktop.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DesktopSettings {
    /// Loopback port of the backend's panel; the backend gets it as JARVIS_UI_PORT.
    pub port: u16,
    /// Global shortcut that shows and focuses the window. Empty disables it.
    pub hotkey: String,
    /// Start the backend with `--wake`: always listening for the wake word.
    pub wake: bool,
    /// The Jarvis home: the backend's working directory, holding `.env`, `.jarvis/` and
    /// voices. Unset: the repo in dev builds, the app's data directory when installed.
    pub home: Option<PathBuf>,
    /// How long to wait for the backend's /health before reporting a failed start.
    pub startup_timeout_secs: u64,
}

impl Default for DesktopSettings {
    fn default() -> Self {
        Self {
            port: 8000,
            hotkey: "Ctrl+Alt+J".into(),
            wake: false,
            home: None,
            startup_timeout_secs: 180,
        }
    }
}

/// Settings plus where they came from and anything wrong with them (already corrected).
pub struct Loaded {
    pub settings: DesktopSettings,
    pub path: PathBuf,
    pub problems: Vec<String>,
}

/// Read `<config_dir>/desktop.json` (writing the defaults there on first run), then apply
/// env overrides. Never fails: problems are reported and the defaults are used instead.
pub fn load(config_dir: &Path) -> Loaded {
    let path = config_dir.join(FILE_NAME);
    let mut problems = Vec::new();
    let mut settings = match fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str(&text).unwrap_or_else(|err| {
            problems.push(format!("{}: {err}; using the defaults", path.display()));
            DesktopSettings::default()
        }),
        Err(_) => {
            let defaults = DesktopSettings::default();
            let _ = fs::create_dir_all(config_dir);
            if let Ok(text) = serde_json::to_string_pretty(&defaults) {
                let _ = fs::write(&path, text + "\n");
            }
            defaults
        }
    };
    apply_env(&mut settings, |name| std::env::var(name).ok(), &mut problems);
    if settings.port == 0 {
        problems.push("port 0 is not allowed; using 8000".into());
        settings.port = DesktopSettings::default().port;
    }
    Loaded { settings, path, problems }
}

fn apply_env(
    settings: &mut DesktopSettings,
    var: impl Fn(&str) -> Option<String>,
    problems: &mut Vec<String>,
) {
    if let Some(value) = var("JARVIS_DESKTOP_PORT") {
        match value.trim().parse() {
            Ok(port) => settings.port = port,
            Err(_) => problems.push(format!("JARVIS_DESKTOP_PORT={value:?} is not a port")),
        }
    }
    if let Some(value) = var("JARVIS_DESKTOP_HOTKEY") {
        settings.hotkey = value;
    }
    if let Some(value) = var("JARVIS_DESKTOP_WAKE") {
        settings.wake = matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on");
    }
    if let Some(value) = var("JARVIS_HOME").filter(|v| !v.trim().is_empty()) {
        settings.home = Some(PathBuf::from(value));
    }
}

/// Where the backend runs: the configured home, else the repo (dev) or `data_dir` (installed).
pub fn resolve_home(configured: Option<&Path>, data_dir: &Path) -> PathBuf {
    if let Some(home) = configured {
        return home.to_path_buf();
    }
    if cfg!(debug_assertions) {
        // desktop/src-tauri -> the repo root, where the developer's .env already lives.
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        if let Some(repo) = manifest.parent().and_then(Path::parent) {
            return repo.to_path_buf();
        }
    }
    data_dir.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> =
            pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name| map.get(name).cloned()
    }

    #[test]
    fn partial_file_fills_in_defaults() {
        let parsed: DesktopSettings = serde_json::from_str(r#"{"port": 8765}"#).unwrap();
        assert_eq!(parsed.port, 8765);
        assert_eq!(parsed.hotkey, "Ctrl+Alt+J");
        assert!(!parsed.wake);
    }

    #[test]
    fn env_overrides_win() {
        let mut settings = DesktopSettings::default();
        let mut problems = Vec::new();
        apply_env(
            &mut settings,
            env(&[
                ("JARVIS_DESKTOP_PORT", "8123"),
                ("JARVIS_DESKTOP_HOTKEY", "Ctrl+Shift+Space"),
                ("JARVIS_DESKTOP_WAKE", "true"),
                ("JARVIS_HOME", r"D:\jarvis"),
            ]),
            &mut problems,
        );
        assert!(problems.is_empty());
        assert_eq!(settings.port, 8123);
        assert_eq!(settings.hotkey, "Ctrl+Shift+Space");
        assert!(settings.wake);
        assert_eq!(settings.home, Some(PathBuf::from(r"D:\jarvis")));
    }

    #[test]
    fn bad_port_is_reported_and_ignored() {
        let mut settings = DesktopSettings::default();
        let mut problems = Vec::new();
        apply_env(&mut settings, env(&[("JARVIS_DESKTOP_PORT", "http")]), &mut problems);
        assert_eq!(settings.port, 8000);
        assert_eq!(problems.len(), 1);
    }

    #[test]
    fn first_run_writes_defaults_and_bad_json_falls_back() {
        let dir = std::env::temp_dir().join(format!("jarvis-desktop-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let first = load(&dir);
        assert!(first.path.is_file());
        assert_eq!(first.settings.port, DesktopSettings::default().port);

        fs::write(&first.path, "{ not json").unwrap();
        let broken = load(&dir);
        assert_eq!(broken.settings.port, DesktopSettings::default().port);
        assert!(broken.problems.iter().any(|p| p.contains("using the defaults")));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn configured_home_wins() {
        let home = resolve_home(Some(Path::new(r"C:\j")), Path::new(r"C:\data"));
        assert_eq!(home, PathBuf::from(r"C:\j"));
    }
}
