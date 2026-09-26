// No console window alongside the app in release builds. DO NOT REMOVE!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    jarvis_desktop_lib::run()
}
