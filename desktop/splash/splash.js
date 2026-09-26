// Renders the state the desktop shell puts in the URL fragment. No Tauri APIs: this page
// needs no IPC, so the app grants it none.
"use strict";

const DEFAULTS = {
  starting: ["Starting Jarvis…", "Loading the assistant. The first start can take a minute."],
  restarting: ["Restarting Jarvis…", "The backend stopped unexpectedly; starting it again."],
  error: ["Jarvis couldn't start", ""],
};

function render() {
  const params = new URLSearchParams(location.hash.slice(1));
  const state = DEFAULTS[params.get("state")] ? params.get("state") : "starting";
  const [title, detail] = DEFAULTS[state];
  document.body.className = state;
  document.getElementById("title").textContent = params.get("title") || title;
  document.getElementById("detail").textContent = params.get("detail") || detail;
  const log = document.getElementById("log");
  log.textContent = params.get("log") || "";
  log.classList.toggle("filled", log.textContent !== "");
  log.scrollTop = log.scrollHeight;
  const hint = params.get("hint");
  if (hint) document.getElementById("hint").textContent = hint;
  document.title = state === "error" ? "Jarvis — not running" : "Jarvis";
}

window.addEventListener("hashchange", render);
render();
