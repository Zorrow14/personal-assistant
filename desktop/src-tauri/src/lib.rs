//! Jarvis desktop shell: the existing local panel in a native window, a tray icon, a global
//! "summon" hotkey, and the Python backend run as a managed sidecar. Nothing here talks to the
//! network except 127.0.0.1:<port>, the backend's loopback-only panel.

mod backend;
mod settings;
mod ui;

use std::sync::Arc;
use std::time::Duration;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::webview::Color;
use tauri::{App, AppHandle, Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};
use tauri_plugin_window_state::StateFlags;

use backend::Backend;

pub fn run() {
    tauri::Builder::default()
        // First, so a second launch focuses this window instead of starting another backend
        // that would fight over the port.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| ui::summon(app)))
        .plugin(tauri_plugin_shell::init())
        .plugin(
            // Remember size and position, but always start visible (never "hidden in tray").
            tauri_plugin_window_state::Builder::new()
                .with_state_flags(StateFlags::all() - StateFlags::VISIBLE)
                .build(),
        )
        .setup(|app| {
            setup(app)?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Jarvis desktop app")
        .run(|app, event| match event {
            // Closing the window only hides it, so this is a programmatic exit (Quit).
            RunEvent::ExitRequested { code: None, api, .. } => api.prevent_exit(),
            RunEvent::Exit => shutdown(app),
            _ => {}
        });
}

fn setup(app: &mut App) -> Result<(), Box<dyn std::error::Error>> {
    let paths = app.path();
    let loaded = settings::load(&paths.app_config_dir()?);
    let config = &loaded.settings;
    let home = settings::resolve_home(config.home.as_deref(), &paths.app_data_dir()?);
    let _ = std::fs::create_dir_all(&home);

    let backend = Backend::new(backend::Config {
        port: config.port,
        wake: config.wake,
        home: home.clone(),
        startup_timeout: Duration::from_secs(config.startup_timeout_secs.max(5)),
        log_path: paths.app_log_dir()?.join("backend.log"),
        settings_path: loaded.path.clone(),
    });
    backend.note(&format!("settings: {} ({config:?})", loaded.path.display()));
    backend.note(&format!("jarvis home: {}", home.display()));
    for problem in &loaded.problems {
        backend.note(&format!("settings problem: {problem}"));
    }

    let port = config.port;
    let splash_url = splash_url(app)?;
    let window = WebviewWindowBuilder::new(app, ui::MAIN, WebviewUrl::App("index.html".into()))
        .title("Jarvis")
        .inner_size(1080.0, 720.0)
        .min_inner_size(420.0, 560.0)
        .center()
        .background_color(Color(5, 7, 10, 255))
        .on_navigation({
            let splash_url = splash_url.clone();
            move |url| {
                let allowed = ui::navigation_allowed(url, &splash_url, port)
                    || url.scheme() == "tauri"; // the splash on non-Windows platforms
                if !allowed {
                    eprintln!("jarvis-desktop: blocked navigation to {url}");
                }
                allowed
            }
        })
        .build()?;
    // Closing the window hides it to the tray; Quit (tray menu) really exits.
    let hider = window.clone();
    window.on_window_event(move |event| {
        if let WindowEvent::CloseRequested { api, .. } = event {
            api.prevent_close();
            let _ = hider.hide();
        }
    });

    let hotkey = register_hotkey(app, &config.hotkey, &backend);
    let hint = match &hotkey {
        Some(keys) => format!("Closing the window keeps Jarvis in the tray; {keys} brings it back."),
        None => "Closing the window keeps Jarvis in the tray; click the tray icon to bring it back.".into(),
    };
    app.manage(ui::Splash { url: splash_url, hint });
    create_tray(app, Arc::clone(&backend))?;

    app.manage(Arc::clone(&backend));
    backend.start(app.handle());
    Ok(())
}

/// The bundled splash page's URL: the CLI's dev server under `tauri dev`, otherwise the app's
/// own asset origin (http://tauri.localhost on Windows).
fn splash_url(app: &App) -> Result<tauri::Url, url::ParseError> {
    #[cfg(dev)]
    if let Some(dev) = app.config().build.dev_url.as_ref() {
        return dev.join("index.html");
    }
    let _ = app;
    tauri::Url::parse("http://tauri.localhost/index.html")
}

/// Register the summon hotkey. Returns it for display, or None if disabled or unavailable
/// (another app may own it); the tray still works either way.
fn register_hotkey(app: &mut App, keys: &str, backend: &Backend) -> Option<String> {
    let keys = keys.trim();
    if keys.is_empty() {
        return None;
    }
    let shortcut: Shortcut = match keys.parse() {
        Ok(shortcut) => shortcut,
        Err(err) => {
            backend.note(&format!("hotkey {keys:?} is not valid ({err}); none registered"));
            return None;
        }
    };
    let plugin = tauri_plugin_global_shortcut::Builder::new()
        .with_handler(|app, _shortcut, event| {
            if event.state() == ShortcutState::Pressed {
                ui::summon(app);
            }
        })
        .build();
    if let Err(err) = app.handle().plugin(plugin) {
        backend.note(&format!("global shortcuts unavailable: {err}"));
        return None;
    }
    match app.global_shortcut().register(shortcut) {
        Ok(()) => {
            backend.note(&format!("hotkey {keys} registered"));
            Some(keys.to_string())
        }
        Err(err) => {
            backend.note(&format!("hotkey {keys} couldn't be registered: {err}"));
            None
        }
    }
}

fn create_tray(app: &mut App, backend: Arc<Backend>) -> tauri::Result<()> {
    let toggle = MenuItem::with_id(app, "toggle", "Show/Hide Jarvis", true, None::<&str>)?;
    let restart = MenuItem::with_id(app, "restart", "Restart backend", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Jarvis", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&toggle, &restart, &separator, &quit])?;

    let mut tray = TrayIconBuilder::with_id("jarvis")
        .tooltip("Jarvis")
        .menu(&menu)
        .show_menu_on_left_click(false) // left-click toggles; right-click opens the menu
        .on_menu_event(move |app, event| match event.id.as_ref() {
            "toggle" => ui::toggle(app),
            "restart" => backend.restart(app),
            "quit" => {
                if let Some(window) = app.get_webview_window(ui::MAIN) {
                    let _ = window.hide(); // feels instant while the backend shuts down
                }
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                ui::toggle(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        tray = tray.icon(icon.clone());
    }
    tray.build(app)?;
    Ok(())
}

/// On exit: release the hotkey and stop the backend, so no Python process outlives the app.
fn shutdown(app: &AppHandle) {
    let _ = app.global_shortcut().unregister_all();
    if let Some(backend) = app.try_state::<Arc<Backend>>() {
        backend.note("--- app exiting: stopping the backend ---");
        backend.shutdown();
    }
}
