// Build the Python backend into one executable with PyInstaller and put it where Tauri's
// `externalBin` expects it: src-tauri/binaries/jarvis-backend-<target-triple>(.exe).
//
//   node scripts/build-sidecar.mjs               always rebuild (what `tauri build` runs)
//   node scripts/build-sidecar.mjs --if-missing  only if there's no sidecar yet (`tauri dev`)
//
// Needs uv (https://docs.astral.sh/uv/) and Rust. Set UV=<path to uv> if uv isn't on PATH.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const desktop = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repo = resolve(desktop, "..");
const ext = process.platform === "win32" ? ".exe" : "";

function targetTriple() {
  if (process.env.TAURI_ENV_TARGET_TRIPLE) return process.env.TAURI_ENV_TARGET_TRIPLE;
  return execFileSync("rustc", ["--print", "host-tuple"], { encoding: "utf8" }).trim();
}

// uv is often installed where PATH doesn't reach (pip --user, the standalone installer).
function findUv() {
  if (process.env.UV) return process.env.UV;
  try {
    execFileSync("uv", ["--version"], { stdio: "ignore" });
    return "uv";
  } catch {}
  const candidates = [
    join(homedir(), ".local", "bin", `uv${ext}`),
    join(homedir(), ".cargo", "bin", `uv${ext}`),
  ];
  const pyUser = process.env.APPDATA && join(process.env.APPDATA, "Python");
  if (pyUser && existsSync(pyUser)) {
    for (const dir of readdirSync(pyUser)) candidates.push(join(pyUser, dir, "Scripts", `uv${ext}`));
  }
  const found = candidates.find((path) => existsSync(path));
  if (!found) throw new Error("uv not found: install it, or set UV to its full path");
  return found;
}

const triple = targetTriple();
const target = join(desktop, "src-tauri", "binaries", `jarvis-backend-${triple}${ext}`);

if (process.argv.includes("--if-missing") && existsSync(target)) {
  console.log(`sidecar: using existing ${target}`);
  process.exit(0);
}

console.log("sidecar: building the backend with PyInstaller (takes a few minutes)...");
execFileSync(
  findUv(),
  [
    "run", "--group", "packaging",
    "pyinstaller", "packaging/jarvis-backend.spec", "--noconfirm",
    "--distpath", "packaging/dist", "--workpath", "packaging/build",
  ],
  { cwd: repo, stdio: "inherit" },
);

const built = join(repo, "packaging", "dist", `jarvis-backend${ext}`);
mkdirSync(dirname(target), { recursive: true });
copyFileSync(built, target);
const mb = (statSync(target).size / 1024 / 1024).toFixed(0);
console.log(`sidecar: ${target} (${mb} MB)`);
