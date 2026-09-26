# Jarvis desktop app

A native desktop shell for Jarvis, built with [Tauri v2](https://v2.tauri.app/). It
wraps the same local panel that `jarvis --serve` serves:

- **A native window** showing the panel. The UI isn't reimplemented: the window
  loads `http://127.0.0.1:<port>/` from the backend.
- **The backend as a managed child process.** Jarvis's Python backend
  (`jarvis --serve`, frozen by PyInstaller into one `.exe`) is bundled as a Tauri
  *sidecar*. The app starts it, waits for `GET /health`, then shows the panel. It
  restarts the backend once if it crashes, and stops it when you quit.
- **A tray icon.** Left-click shows or hides the window. Right-click opens a
  menu: *Show/Hide Jarvis*, *Restart backend*, *Quit Jarvis*. Closing the window
  hides it to the tray; only *Quit* exits.
- **A global hotkey**, <kbd>Ctrl</kbd>+<kbd>Alt</kbd>+<kbd>J</kbd> by default, which
  shows and focuses the window from any app. This is separate from the
  "Hey Jarvis" voice wake word, which still runs in the backend (see `wake` below).

Everything stays on this computer. The backend binds 127.0.0.1 only, as it
always has. The window may only show the bundled start-up page and the panel's
own origin; any other navigation is blocked. Neither page gets Tauri API access.

The terminal modes (`python -m jarvis.cli`, `--voice`, `--wake`, `--serve`) are
unchanged; the app is just another way to run Jarvis.

## Prerequisites (Windows)

| What | Why | Get it |
|---|---|---|
| Rust (stable, MSVC toolchain) | compiles the shell | <https://rustup.rs> (`rustup-init.exe`, defaults) |
| Microsoft C++ Build Tools + Windows SDK | the MSVC linker Rust uses | Visual Studio Build Tools, "Desktop development with C++" |
| WebView2 runtime | renders the window | preinstalled on Windows 10/11 |
| Node.js 18+ | runs the Tauri CLI (installed locally by `npm install`) | <https://nodejs.org> |
| uv + the Jarvis dev setup | builds the backend with PyInstaller | `uv sync` in the repo root, see the main README |

PyInstaller doesn't need a separate install: it lives in the repo's `packaging`
dependency group, and the build script runs it with `uv run --group packaging`.
The NSIS installer tooling is downloaded by `tauri build` on first use.

If `uv` isn't on your PATH, set `UV` to its full path before building, for example
`$env:UV = "$env:APPDATA\Python\Python314\Scripts\uv.exe"`.

## Run it in development

```powershell
cd desktop
npm install          # once: the Tauri CLI
npm run dev          # = tauri dev
```

`tauri dev` first runs `npm run sidecar:if-missing`, which builds the backend exe
only if `src-tauri/binaries/` doesn't have one yet. That first build takes a few
minutes. After you change Python code, rebuild the sidecar with `npm run sidecar`;
`tauri dev` doesn't do it for you.

In a dev build the backend runs in **the repo root**, so it uses your existing
`.env`, `.jarvis/` and `voices/`. The backend's output is echoed to the terminal
as `[backend] ...` lines.

## Build the installer

```powershell
cd desktop
npm run build        # = tauri build: rebuilds the sidecar, then the app and installer
```

The output is `src-tauri/target/release/bundle/nsis/Jarvis_0.1.0_x64-setup.exe`.
It installs per user, so no admin prompt is needed, and adds a Start-menu
shortcut. The installer includes the sidecar.

To build only the backend exe: `npm run sidecar`. This runs
`packaging/jarvis-backend.spec` and copies `packaging/dist/jarvis-backend.exe` to
`src-tauri/binaries/jarvis-backend-<target-triple>.exe`, the name Tauri's
`externalBin` requires.

**Not signed.** The installer and app are unsigned, which is fine for personal
use; Windows SmartScreen may warn on first run ("More info" → "Run anyway").
TODO(distribution): code-signing, an auto-updater (tauri-plugin-updater) and
store packaging are separate, later steps.

## Where Jarvis keeps its data (the "Jarvis home")

Nothing personal is baked into the binary. The backend runs with its working
directory set to the **Jarvis home** and reads everything from there, exactly
like running `python -m jarvis.cli --serve` in that folder:

- `.env` (your API key, vault path and every `JARVIS_*` setting)
- `.jarvis/` (memory index, embedding-model cache, reminders, metrics)
- Piper voices (`JARVIS_TTS_VOICE`, relative to the home unless absolute)

The home is:

- **Dev builds (`tauri dev`):** the repo root.
- **The installed app:** `"home"` from `desktop.json` (below). If unset, it's
  `%APPDATA%\com.zorrow.jarvis\`. To use your existing setup, point `"home"` at
  the repo, or copy `.env`, `.jarvis\` and `voices\` into that folder.

The Whisper model downloads to the Hugging Face cache (`~\.cache\huggingface`) on
first use, as in the terminal modes.

## Settings: port, hotkey, wake word, home

On first run the app writes `%APPDATA%\com.zorrow.jarvis\desktop.json`:

```json
{
  "port": 8000,
  "hotkey": "Ctrl+Alt+J",
  "wake": false,
  "home": null,
  "startup_timeout_secs": 180
}
```

| Key | Meaning |
|---|---|
| `port` | The panel's port on 127.0.0.1. The app passes it to the backend as `JARVIS_UI_PORT`, so it overrides the value in `.env`. If the port is already taken, for example by `jarvis --serve` in a terminal, the app says so instead of starting. |
| `hotkey` | The summon shortcut: modifiers `Ctrl`, `Alt`, `Shift`, `Super` plus a key, e.g. `"Ctrl+Shift+Space"`, `"Alt+F12"`. `""` disables it. If another app already owns the combination, Jarvis logs that and relies on the tray. |
| `wake` | `true` starts the backend with `--wake`: always listening for "Hey Jarvis", like `--serve --wake`. The default `false` opens the mic only while you use Talk. |
| `home` | The Jarvis home (above). Use forward slashes or `\\` in JSON: `"C:/Users/you/jarvis-personal-assistant"`. |
| `startup_timeout_secs` | How long to wait for `/health` before reporting a failed start. The first start of the installed app unpacks the backend and loads models, so it can take a while. |

Quit and reopen the app after editing. For one run, the environment variables
`JARVIS_DESKTOP_PORT`, `JARVIS_DESKTOP_HOTKEY`, `JARVIS_DESKTOP_WAKE` and
`JARVIS_HOME` override the file (handy with `tauri dev`).

## Replacing the icon

The icon is a placeholder (the panel's idle orb): `icons-src/app-icon.png`,
1024×1024. To replace it, drop in your own square PNG and regenerate every size:

```powershell
cd desktop
npx tauri icon path\to\your-icon.png   # writes src-tauri/icons/*
```

The window, taskbar, tray and installer all use it.

## Troubleshooting

- **The backend's log** is `%LOCALAPPDATA%\com.zorrow.jarvis\logs\backend.log`.
  It's rewritten on each app start. When the backend fails, its last lines are
  also shown in the window.
- **"Port 8000 is already in use"**: another Jarvis, often `--serve` in a
  terminal, has it. Stop that, or change `port`.
- **"Jarvis couldn't start … exited before it was ready"**: usually config. The
  log shows the same `FAIL` line the terminal would, e.g. a missing `.env` or
  `JARVIS_VAULT_PATH`. Check that the home is where your `.env` is.
- **Actions that need your OK are declined.** The terminal modes ask `y/N` before
  side-effect tools, such as writing a note or saving a reminder. The app has no
  terminal, so those prompts are declined automatically: the safe default. The
  panel shows the call as "declined". To let such tools run without asking, set
  `JARVIS_CONFIRM_SIDE_EFFECTS=false` in `.env`; understand what that allows first.
  TODO(phase-6+): approve side-effect actions in the panel itself.
- **Nothing happens on the hotkey**: see `backend.log` for "hotkey … couldn't be
  registered" (another app owns it), and pick another in `desktop.json`.

## How it fits together

```
desktop/
├── package.json            Tauri CLI + scripts (dev, build, sidecar)
├── scripts/build-sidecar.mjs  PyInstaller -> src-tauri/binaries/jarvis-backend-<triple>.exe
├── splash/                 the start-up / error page (bundled; the panel itself is not)
├── icons-src/app-icon.png  placeholder icon source
└── src-tauri/
    ├── tauri.conf.json     app id, bundle (NSIS, externalBin), CSP
    ├── capabilities/       deliberately empty: no page gets Tauri APIs
    └── src/
        ├── lib.rs          window, tray, hotkey, plugins, exit handling
        ├── backend.rs      sidecar supervisor: spawn, /health, restart, stop
        ├── settings.rs     desktop.json + env overrides
        └── ui.rs           splash/panel navigation, show/hide/summon
packaging/
├── jarvis-backend.spec     PyInstaller spec (one-file backend)
└── hooks/                  PyInstaller hook overrides
src/jarvis/sidecar.py       the backend entrypoint: `--serve` + a stdin control channel
```

**Starting.** The app spawns `jarvis-backend --managed` in the home, with
`JARVIS_UI_HOST=127.0.0.1` and `JARVIS_UI_PORT=<port>`. It polls `/health` every
0.3 s, showing "Starting Jarvis…", and then navigates the window to the panel.

**Stopping.** `--managed` makes the backend treat its stdin as a control channel.
On *Quit*, the app writes `shutdown`, and the backend stops the way Ctrl-C stops
`--serve`; the PyInstaller bootloader then deletes its temp folder. If the app
itself dies, even when killed from Task Manager, the OS closes that pipe, and the
backend sees EOF and shuts down the same way. As a last resort, a backend still
running 12 s after *Quit* is killed.

**Crashes.** If the backend dies after it was healthy, the app logs it and
restarts it once. A second crash within 10 minutes shows an error with the log's
tail. *Restart backend* in the tray always tries again.
