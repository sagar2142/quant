// Neutron desktop shell — MASTER_PLAN §13.6.
//
// Wraps the *identical* React build the browser serves. That is the whole point
// of the staged approach: the console is built and validated as a web app
// during paper trading, and gains a native shell only when live capital makes
// the browser's limitations matter.
//
// The six things a browser tab cannot do for this job:
//
//   1. Browser chrome eats ~80px of vertical space; §12.9 budgets every pixel
//      of an 836px workspace.
//   2. Ctrl+W closes the console mid-session, with no confirmation.
//   3. Ctrl+T / Ctrl+F / "/" collide with the keyboard-first scheme in §12.8.
//   4. A global hotkey for the kill switch works even when unfocused.
//   5. Native OS notifications and a tray icon.
//   6. Confirm-on-quit.
//
// ~5MB using the system webview. Electron would ship 100MB+ of Chromium for
// the same six properties.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Manager, WindowEvent};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut};

/// Ctrl+Alt+K, reachable whether or not the window has focus.
///
/// Deliberately awkward: three keys, none of them adjacent. A single-key panic
/// button gets pressed by a cat.
const KILL_SHORTCUT: Shortcut =
    Shortcut::new(Some(Modifiers::CONTROL.union(Modifiers::ALT)), Code::KeyK);

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(|app| {
            let handle = app.handle().clone();

            // The global hotkey does not itself halt trading. It raises the
            // window and focuses the kill control, so the typed confirmation
            // still happens (§12.8). A hotkey that halts directly is a hotkey
            // that halts by accident.
            app.global_shortcut().on_shortcut(KILL_SHORTCUT, move |_app, _sc, _event| {
                if let Some(window) = handle.get_webview_window("main") {
                    let _ = window.show();
                    let _ = window.set_focus();
                    let _ = window.emit("focus-kill-switch", ());
                }
            })?;

            Ok(())
        })
        .on_window_event(|window, event| {
            // Confirm on quit. Closing the console does not stop trading — the
            // engine is a separate daemon (§13.6) — but an operator who closes
            // it by accident has lost their view of a live book.
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.emit("confirm-close", ());
            }
        })
        .run(tauri::generate_context!())
        .expect("failed to start the Neutron console");
}
