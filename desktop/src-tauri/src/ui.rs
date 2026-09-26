//! The main window: the splash page (starting / restarting / error) and the live panel.

use tauri::{AppHandle, Manager, WebviewWindow};
use url::Url;

pub const MAIN: &str = "main";

/// The splash page's URL, and the hint it shows under the status.
pub struct Splash {
    pub url: Url,
    pub hint: String,
}

pub enum Status {
    Starting,
    Restarting,
}

fn window(app: &AppHandle) -> Option<WebviewWindow> {
    app.get_webview_window(MAIN)
}

/// The panel's address. Always loopback: the backend refuses anything else.
pub fn panel_url(port: u16) -> Url {
    Url::parse(&format!("http://127.0.0.1:{port}/")).expect("valid loopback URL")
}

/// Pages the window may show: the bundled splash, and the panel on 127.0.0.1:<port>.
pub fn navigation_allowed(url: &Url, splash: &Url, port: u16) -> bool {
    let same_origin = |a: &Url, b: &Url| a.origin() == b.origin();
    same_origin(url, splash) || same_origin(url, &panel_url(port))
}

fn show_splash(app: &AppHandle, pairs: &[(&str, &str)]) {
    let (Some(window), Some(splash)) = (window(app), app.try_state::<Splash>()) else {
        return;
    };
    let mut fragment = url::form_urlencoded::Serializer::new(String::new());
    for (key, value) in pairs {
        fragment.append_pair(key, value);
    }
    fragment.append_pair("hint", &splash.hint);
    let mut url = splash.url.clone();
    url.set_fragment(Some(&fragment.finish()));
    if let Err(err) = window.navigate(url) {
        eprintln!("jarvis-desktop: can't show the splash page: {err}");
    }
}

pub fn show_status(app: &AppHandle, status: Status) {
    let state = match status {
        Status::Starting => "starting",
        Status::Restarting => "restarting",
    };
    show_splash(app, &[("state", state)]);
}

pub fn show_error(app: &AppHandle, title: &str, detail: &str, log: &str) {
    show_splash(app, &[("state", "error"), ("title", title), ("detail", detail), ("log", log)]);
    summon(app); // a failure shouldn't sit unseen in the tray
}

pub fn show_panel(app: &AppHandle, port: u16) {
    if let Some(window) = window(app) {
        if let Err(err) = window.navigate(panel_url(port)) {
            eprintln!("jarvis-desktop: can't open the panel: {err}");
        }
    }
}

/// Show, un-minimize and focus the window (the hotkey, a second launch, an error).
pub fn summon(app: &AppHandle) {
    if let Some(window) = window(app) {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}

/// Tray left-click / "Show/Hide Jarvis": hide it if it's up, else bring it up.
pub fn toggle(app: &AppHandle) {
    if let Some(window) = window(app) {
        let up = window.is_visible().unwrap_or(false) && !window.is_minimized().unwrap_or(false);
        if up {
            let _ = window.hide();
        } else {
            summon(app);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_splash_and_this_panel_are_allowed() {
        let splash = Url::parse("http://tauri.localhost/index.html").unwrap();
        let ok = |s: &str| navigation_allowed(&Url::parse(s).unwrap(), &splash, 8000);
        assert!(ok("http://tauri.localhost/index.html#state=error"));
        assert!(ok("http://127.0.0.1:8000/"));
        assert!(!ok("http://127.0.0.1:9000/"));
        assert!(!ok("http://localhost:8000/"));
        assert!(!ok("https://example.com/"));
    }
}
