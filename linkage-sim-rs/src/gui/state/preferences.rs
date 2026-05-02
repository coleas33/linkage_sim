//! User preferences store — display settings that follow the user, not the
//! mechanism.
//!
//! `MechanismJson` carries everything mechanism-specific (geometry, joints,
//! sensors, analysis config). This module handles the orthogonal concern:
//! display preferences that should NOT change when you load a colleague's
//! mechanism — unit choice, grid visibility, accent toggles, "skip welcome"
//! status, etc.
//!
//! Storage:
//! - Native: `~/.linkage-sim/preferences.json`
//! - WASM:   localStorage key `linkage_user_prefs`
//!
//! The save path is auto-created on first write; missing / corrupt files
//! return defaults silently (preferences are best-effort UX, not load-bearing
//! behaviour).
//!
//! Serialization model: a single flat `UserPreferences` struct with serde
//! derives. Old fields stay backward-compatible via `#[serde(default)]` so
//! adding a new pref doesn't invalidate existing files.

use serde::{Deserialize, Serialize};

use super::{DisplayUnits, GridSettings};

/// Persisted user preferences. Default values match the in-memory defaults
/// so a missing/corrupt prefs file produces the same UX as a first launch.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Display unit preferences (length: m/mm, angle: rad/deg).
    #[serde(default)]
    pub display_units: DisplayUnits,
    /// Grid visibility, snap behaviour, spacing (m).
    #[serde(default)]
    pub grid: GridSettings,
    /// Show body / link dimension labels on the canvas.
    #[serde(default = "default_true")]
    pub show_dimensions: bool,
    /// Show body / joint ID labels on the canvas.
    #[serde(default = "default_true")]
    pub show_labels: bool,
    /// Overlay loop equations on the canvas (View → Show equations).
    #[serde(default)]
    pub show_equation_overlay: bool,
    /// Show force-element arrows on the canvas.
    #[serde(default = "default_true")]
    pub show_forces: bool,
    /// Show the bottom plot panel.
    #[serde(default = "default_true")]
    pub show_plots: bool,
    /// Show the parametric study side panel.
    #[serde(default)]
    pub show_parametric: bool,
    /// Show the developer debug overlay.
    #[serde(default)]
    pub show_debug_overlay: bool,
    /// User has dismissed the welcome dialog at least once.
    #[serde(default)]
    pub dismiss_welcome: bool,
    /// Easter egg toggle.
    #[serde(default)]
    pub nathan_mode: bool,
}

fn default_true() -> bool {
    true
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            display_units: DisplayUnits::default(),
            grid: GridSettings::default(),
            show_dimensions: true,
            show_labels: true,
            show_equation_overlay: false,
            show_forces: true,
            show_plots: true,
            show_parametric: false,
            show_debug_overlay: false,
            dismiss_welcome: false,
            nathan_mode: false,
        }
    }
}

impl UserPreferences {
    /// Load from persistent storage. Returns defaults on any failure
    /// (missing file, corrupt JSON, no localStorage, etc.) since
    /// preferences are best-effort UX.
    pub fn load() -> Self {
        load_impl()
    }

    /// Save to persistent storage. Failures are logged at warn level but not
    /// surfaced to the caller — prefs are best-effort, an I/O blip shouldn't
    /// disrupt the user's actual work.
    pub fn save(&self) {
        save_impl(self);
    }
}

// ── Native: read/write `~/.linkage-sim/preferences.json` ─────────────────

#[cfg(feature = "native")]
fn prefs_path() -> Option<std::path::PathBuf> {
    dirs::home_dir().map(|h| h.join(".linkage-sim").join("preferences.json"))
}

#[cfg(feature = "native")]
fn load_impl() -> UserPreferences {
    let Some(path) = prefs_path() else {
        return UserPreferences::default();
    };
    let Ok(json) = std::fs::read_to_string(&path) else {
        return UserPreferences::default();
    };
    serde_json::from_str(&json).unwrap_or_default()
}

#[cfg(feature = "native")]
fn save_impl(prefs: &UserPreferences) {
    let Some(path) = prefs_path() else { return };
    if let Some(parent) = path.parent() {
        if std::fs::create_dir_all(parent).is_err() {
            log::warn!("Failed to create preferences dir: {:?}", parent);
            return;
        }
    }
    match serde_json::to_string_pretty(prefs) {
        Ok(s) => {
            if let Err(e) = std::fs::write(&path, s) {
                log::warn!("Failed to write preferences {:?}: {}", path, e);
            }
        }
        Err(e) => log::warn!("Failed to serialize preferences: {}", e),
    }
}

// ── WASM: read/write localStorage `linkage_user_prefs` ───────────────────

#[cfg(target_arch = "wasm32")]
const PREFS_KEY: &str = "linkage_user_prefs";

#[cfg(target_arch = "wasm32")]
fn load_impl() -> UserPreferences {
    let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten())
    else {
        return UserPreferences::default();
    };
    match storage.get_item(PREFS_KEY) {
        Ok(Some(s)) => serde_json::from_str(&s).unwrap_or_default(),
        _ => UserPreferences::default(),
    }
}

#[cfg(target_arch = "wasm32")]
fn save_impl(prefs: &UserPreferences) {
    if let Some(storage) =
        web_sys::window().and_then(|w| w.local_storage().ok().flatten())
    {
        if let Ok(s) = serde_json::to_string(prefs) {
            let _ = storage.set_item(PREFS_KEY, &s);
        }
    }
}

// ── Fallback (no native, not wasm32) — no-op so `--no-default-features`
//    on a native host links cleanly. ───────────────────────────────────────

#[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
fn load_impl() -> UserPreferences {
    UserPreferences::default()
}

#[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
fn save_impl(_prefs: &UserPreferences) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn user_preferences_round_trip_through_json() {
        let mut prefs = UserPreferences::default();
        prefs.show_dimensions = false;
        prefs.show_equation_overlay = true;
        prefs.dismiss_welcome = true;
        prefs.nathan_mode = true;
        prefs.grid.snap_enabled = false;
        prefs.grid.spacing_m = 0.005;
        prefs.display_units.length = super::super::LengthUnit::Meters;
        prefs.display_units.angle = super::super::AngleUnit::Radians;

        let s = serde_json::to_string(&prefs).expect("serialize");
        let restored: UserPreferences = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(prefs, restored);
    }

    #[test]
    fn user_preferences_load_handles_corrupt_json() {
        // The serde_json::from_str path returns Err for malformed JSON;
        // unwrap_or_default falls back. Verify the defaults shape.
        let restored: UserPreferences =
            serde_json::from_str("not json").unwrap_or_default();
        assert_eq!(restored, UserPreferences::default());
    }

    #[test]
    fn user_preferences_load_handles_missing_fields() {
        // Backward compatibility: a JSON with only some fields should
        // default the rest. This is what an older preferences file looks
        // like after we add new prefs to the struct.
        let partial = r#"{ "show_dimensions": false }"#;
        let restored: UserPreferences =
            serde_json::from_str(partial).expect("partial JSON should load");
        assert!(!restored.show_dimensions, "explicit field should win");
        assert!(restored.show_labels, "missing field should default to true");
        assert!(!restored.dismiss_welcome, "missing bool defaults to false");
    }

    #[test]
    fn appstate_apply_then_snapshot_round_trip() {
        // Verifies AppState::apply_user_preferences and current_user_preferences
        // round-trip lossly: apply a custom UserPreferences, then read it
        // back via current_user_preferences, and check equality.
        use crate::gui::state::AppState;

        let mut state = AppState::default();
        let mut prefs = UserPreferences::default();
        prefs.show_dimensions = false;
        prefs.show_labels = false;
        prefs.show_forces = false;
        prefs.show_equation_overlay = true;
        prefs.dismiss_welcome = true;
        prefs.nathan_mode = true;
        prefs.grid.snap_enabled = false;
        prefs.grid.show_grid = false;
        prefs.grid.spacing_m = 0.005;
        prefs.display_units.length = super::super::LengthUnit::Meters;
        prefs.display_units.angle = super::super::AngleUnit::Radians;

        state.apply_user_preferences(&prefs);
        let snap = state.current_user_preferences();
        assert_eq!(snap, prefs);
    }

    #[test]
    fn tick_save_user_prefs_skips_when_unchanged() {
        // The tick should be cheap when nothing changed — nothing to assert
        // about disk writes since they go to ~/.linkage-sim or localStorage,
        // but we can verify the in-memory snapshot equality short-circuit
        // by calling tick twice and confirming last_saved_prefs is stable.
        use crate::gui::state::AppState;

        let mut state = AppState::default();
        // After Default::default(), last_saved_prefs == current_user_preferences()
        // (the constructor primes it). First tick should be a no-op.
        let before = state.last_saved_prefs.clone();
        state.tick_save_user_prefs();
        assert_eq!(state.last_saved_prefs, before, "no-op tick should not change snapshot");
    }
}
