# PyInstaller spec: the Jarvis backend (`python -m jarvis.sidecar`) as ONE self-contained
# executable, used by the desktop app (desktop/) as its Tauri sidecar.
#
# Build it (from the repo root; desktop/scripts/build-sidecar.mjs runs this and copies the
# result to desktop/src-tauri/binaries/jarvis-backend-<target-triple>.exe):
#
#     uv run --group packaging pyinstaller packaging/jarvis-backend.spec --noconfirm \
#         --distpath packaging/dist --workpath packaging/build
#
# What is baked in: Python, Jarvis's code, the panel page (jarvis/server/static), and the
# model files that ship INSIDE Python packages - openWakeWord's pretrained wake words,
# faster-whisper's VAD model and piper's espeak-ng phoneme data.
#
# What is NOT baked in (resolved at runtime from the Jarvis home, i.e. the working directory
# the desktop app starts the sidecar in): `.env` and the API key, the vault, `.jarvis/`
# (memory index, fastembed cache, reminders, metrics) and Piper voices (`JARVIS_TTS_VOICE`).
# The Whisper model downloads to the Hugging Face cache (~/.cache/huggingface) on first use.

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH is defined by PyInstaller
ENTRY = ROOT / "src" / "jarvis" / "sidecar.py"

hiddenimports = [
    # Tools are discovered with pkgutil at runtime and voice backends import lazily, so
    # nothing imports them statically: include every Jarvis module explicitly.
    *collect_submodules("jarvis"),
    # chromadb names its components by dotted string (settings), so static analysis misses them.
    *collect_submodules("chromadb", filter=lambda name: ".test" not in name),
    # plyer picks the platform implementation by name at call time.
    *collect_submodules("plyer.platforms.win"),
    "pyttsx3.drivers",
    "pyttsx3.drivers.sapi5",
]

datas = [
    *collect_data_files("jarvis"),  # the panel: jarvis/server/static/index.html
    *collect_data_files("chromadb"),
    *collect_data_files("openwakeword"),  # resources/models/*.onnx (hey_jarvis, ...)
    *collect_data_files("faster_whisper"),  # assets/silero_vad_v6.onnx
    *collect_data_files("piper"),  # espeak-ng-data
    *collect_data_files("fastembed"),
]

binaries = [
    *collect_dynamic_libs("ctranslate2"),
    *collect_dynamic_libs("piper"),
    *collect_dynamic_libs("chromadb_rust_bindings"),
]

a = Analysis(  # noqa: F821
    [str(ENTRY)],
    pathex=[str(ROOT / "src")],
    hookspath=[str(ROOT / "packaging" / "hooks")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[
        # Dev tooling that happens to live in the same venv.
        "pytest",
        "_pytest",
        "PyInstaller",
        "ruff",
        # GUI toolkits Jarvis never uses.
        "tkinter",
        "matplotlib",
        "IPython",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="jarvis-backend",
    # A console program, so stdout/stderr exist for the app's log. The desktop app starts
    # it without a window (CREATE_NO_WINDOW), so no console ever appears.
    console=True,
    upx=False,  # UPX corrupts some of the native libraries (onnxruntime, ctranslate2)
    # Each start unpacks ~0.5 GB into a fresh _MEI folder, deleted on a clean exit. A folder
    # of Jarvis's own (Windows expands the variable) lets jarvis.sidecar clear the ones a
    # hard kill leaves behind without touching other programs' temp files.
    runtime_tmpdir=r"%LOCALAPPDATA%\Jarvis\backend-runtime",
    debug=False,
    strip=False,
)
