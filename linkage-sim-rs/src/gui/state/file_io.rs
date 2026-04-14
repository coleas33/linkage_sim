//! File I/O operations: save, load, autosave, recent files, share via URL.

use std::path::Path;

use crate::core::driver::DriverMeta;
use crate::forces::elements::ForceElement;
use crate::io::{load_mechanism_unbuilt, mechanism_to_json};
use super::{AppState, LoadCaseManager};
use super::blueprint_ops::detect_driver_joint_id;

// ── Share via URL helpers (compress + base64 encode) ─────────────────────────

/// Compress a JSON string with deflate and then base64url-encode it.
/// This produces a URL-safe string suitable for `?m=` parameter.
pub fn encode_mechanism_for_url(json_str: &str) -> String {
    use base64::Engine;
    use flate2::write::DeflateEncoder;
    use flate2::Compression;
    use std::io::Write;

    let mut encoder = DeflateEncoder::new(Vec::new(), Compression::best());
    encoder.write_all(json_str.as_bytes()).unwrap_or_default();
    let compressed = encoder.finish().unwrap_or_default();
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&compressed)
}

/// Decode a URL-safe base64 string back to the mechanism JSON.
/// Reverses `encode_mechanism_for_url`: base64-decode then deflate-decompress.
pub fn decode_mechanism_from_url(encoded: &str) -> Result<String, String> {
    use base64::Engine;
    use flate2::read::DeflateDecoder;
    use std::io::Read;

    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded)
        .map_err(|e| format!("Base64 decode error: {}", e))?;
    let mut decoder = DeflateDecoder::new(&bytes[..]);
    let mut json_str = String::new();
    decoder
        .read_to_string(&mut json_str)
        .map_err(|e| format!("Deflate decompress error: {}", e))?;
    Ok(json_str)
}

impl AppState {
    /// Generate a share URL encoding the current mechanism.
    ///
    /// Returns the full URL string, or an error if no mechanism is loaded.
    pub fn generate_share_url(&self) -> Result<String, String> {
        // Include the driver angle in the JSON so the mechanism loads at the
        // same configuration the user was viewing (not just angle 0).
        let mut json_str = self.serialize_to_json_string()?;
        // Inject driver_angle into the JSON before encoding.
        if let Ok(mut val) = serde_json::from_str::<serde_json::Value>(&json_str) {
            val["_driver_angle"] = serde_json::Value::from(self.driver_angle);
            if self.sweep_range_enabled {
                val["_sweep_range_enabled"] = serde_json::Value::from(true);
                val["_sweep_angle_min"] = serde_json::Value::from(self.sweep_angle_min_deg);
                val["_sweep_angle_max"] = serde_json::Value::from(self.sweep_angle_max_deg);
            }
            json_str = serde_json::to_string(&val).unwrap_or(json_str);
        }
        let encoded = encode_mechanism_for_url(&json_str);
        Ok(format!("https://linkage.colesorkness.com/?m={}", encoded))
    }

    /// Serialize the current mechanism to a JSON file at the given path.
    ///
    /// Load cases are included in the saved JSON so they persist across
    /// save/load cycles.
    ///
    /// Returns `Err` with a human-readable message on any failure.
    pub fn save_to_file(&mut self, path: &Path) -> Result<(), String> {
        self.write_json_to(path)?;

        self.last_save_path = Some(path.to_path_buf());
        self.dirty = false;
        #[cfg(not(target_arch = "wasm32"))]
        self.add_recent_file(path);
        self.status_message = Some(format!("Saved: {}", path.display()));
        self.status_message_time = 3.0;
        Ok(())
    }

    /// Perform periodic autosave to a temp file alongside the last save path.
    ///
    /// Called from the update loop with accumulated dt. Saves every 30 seconds
    /// if there are unsaved changes and a mechanism is loaded.
    #[cfg(feature = "native")]
    pub fn tick_autosave(&mut self, dt: f64) {
        const AUTOSAVE_INTERVAL: f64 = 30.0;

        if !self.dirty || self.mechanism.is_none() {
            return;
        }

        self.autosave_timer += dt;
        if self.autosave_timer < AUTOSAVE_INTERVAL {
            return;
        }
        self.autosave_timer = 0.0;

        let autosave_path = self.autosave_path();
        if let Some(path) = autosave_path {
            if let Err(e) = self.write_json_to(&path) {
                log::warn!("Autosave failed: {}", e);
            } else {
                log::debug!("Autosaved to {:?}", path);
            }
        }
    }

    /// Compute the autosave file path (sibling to last save, or temp dir).
    #[cfg(feature = "native")]
    pub(crate) fn autosave_path(&self) -> Option<std::path::PathBuf> {
        if let Some(ref save_path) = self.last_save_path {
            let mut p = save_path.clone();
            let stem = p.file_stem()?.to_string_lossy().to_string();
            p.set_file_name(format!(".{}.autosave.json", stem));
            Some(p)
        } else {
            let mut p = std::env::temp_dir();
            p.push("linkage_simulator_autosave.json");
            Some(p)
        }
    }

    /// Serialize the current mechanism to a pretty-printed JSON string.
    ///
    /// Includes load cases, mounting angle, and blueprint point masses.
    /// Used by both native file writes and WASM localStorage autosave.
    fn serialize_to_json_string(&self) -> Result<String, String> {
        let mech = self
            .mechanism
            .as_ref()
            .ok_or_else(|| "No mechanism loaded".to_string())?;
        let mut json_struct = mechanism_to_json(mech).map_err(|e| e.to_string())?;
        json_struct.load_cases = self.load_cases.cases.clone();
        json_struct.mounting_angle = self.mounting_angle;
        // Preserve blueprint point masses (baked into mass/CG/Izz at build time).
        if let Some(ref bp) = self.blueprint {
            for (body_id, bp_body) in &bp.bodies {
                if let Some(json_body) = json_struct.bodies.get_mut(body_id) {
                    json_body.point_masses = bp_body.point_masses.clone();
                }
            }
        }
        serde_json::to_string_pretty(&json_struct).map_err(|e| e.to_string())
    }

    /// Write mechanism JSON to an arbitrary path (doesn't clear dirty flag).
    fn write_json_to(&self, path: &Path) -> Result<(), String> {
        let json = self.serialize_to_json_string()?;
        std::fs::write(path, json).map_err(|e| format!("Failed to write: {}", e))?;
        Ok(())
    }

    // ── Recent files (native only — no filesystem on WASM) ────────────

    /// Add a path to the recent files list (deduplicates, keeps max 5).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_recent_file(&mut self, path: &Path) {
        let canonical = path.to_path_buf();
        self.recent_files.retain(|p| p != &canonical);
        self.recent_files.insert(0, canonical);
        self.recent_files.truncate(5);
        self.save_recent_files();
    }

    /// Path to the recent files JSON in the user's temp directory.
    #[cfg(not(target_arch = "wasm32"))]
    fn recent_files_path() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push("linkage_simulator_recent.json");
        p
    }

    /// Load recent files list from disk (returns empty Vec on any failure).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn load_recent_files() -> Vec<std::path::PathBuf> {
        let path = Self::recent_files_path();
        let Ok(json) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        serde_json::from_str(&json).unwrap_or_default()
    }

    /// Check for an autosave file in the temp directory on startup.
    /// Returns the path if a recoverable file exists (< 1 hour old).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn check_autosave_recovery() -> Option<std::path::PathBuf> {
        let mut p = std::env::temp_dir();
        p.push("linkage_simulator_autosave.json");
        if !p.exists() {
            return None;
        }
        // Only offer recovery for recent autosaves (< 1 hour).
        if let Ok(metadata) = std::fs::metadata(&p) {
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    if elapsed.as_secs() < 3600 {
                        return Some(p);
                    }
                }
            }
        }
        None
    }

    /// Save recent files list to disk.
    #[cfg(not(target_arch = "wasm32"))]
    fn save_recent_files(&self) {
        let path = Self::recent_files_path();
        if let Ok(json) = serde_json::to_string(&self.recent_files) {
            let _ = std::fs::write(&path, json);
        }
    }

    /// Load a mechanism from a JSON string, solve at t=0, and update all state.
    ///
    /// After loading, the driver is restored from the JSON. If the file has no
    /// driver, the mechanism is loaded without one (the user can assign one via
    /// the Driver panel).
    ///
    /// This is the shared implementation used by both `load_from_file` (native)
    /// and WASM autosave recovery.
    ///
    /// Returns `Err` with a human-readable message on any failure.
    pub fn load_from_json_str(&mut self, json_str: &str) -> Result<(), String> {
        // Parse JSON and store as blueprint before building
        let json_struct: crate::io::MechanismJson =
            serde_json::from_str(json_str).map_err(|e| e.to_string())?;
        self.blueprint = Some(json_struct);

        let mut mech =
            load_mechanism_unbuilt(json_str).map_err(|e| e.to_string())?;
        mech.build().map_err(|e| e.to_string())?;

        // Extract driver parameters before we move mech into self.
        let (driver_omega, driver_theta_0) = if let Some(driver) = mech.drivers().first() {
            match driver.meta() {
                Some(DriverMeta::ConstantSpeed { omega, theta_0 }) => (*omega, *theta_0),
                Some(DriverMeta::Expression { .. }) => (2.0 * std::f64::consts::PI, 0.0),
                Some(DriverMeta::LinearLength { .. })
                | Some(DriverMeta::CosineStroke { .. }) => (2.0 * std::f64::consts::PI, 0.0),
                None => (2.0 * std::f64::consts::PI, 0.0),
            }
        } else {
            (2.0 * std::f64::consts::PI, 0.0)
        };

        // Detect the driven joint ID.
        let driver_joint_id = detect_driver_joint_id(&mech);

        // Check for embedded driver angle and sweep range (from share URL).
        let shared_json: Option<serde_json::Value> =
            serde_json::from_str::<serde_json::Value>(json_str).ok();
        let shared_angle: Option<f64> = shared_json
            .as_ref()
            .and_then(|v| v.get("_driver_angle").and_then(|a| a.as_f64()));
        if let Some(ref val) = shared_json {
            if val.get("_sweep_range_enabled").and_then(|v| v.as_bool()).unwrap_or(false) {
                self.sweep_range_enabled = true;
                if let Some(min) = val.get("_sweep_angle_min").and_then(|v| v.as_f64()) {
                    self.sweep_angle_min_deg = min;
                }
                if let Some(max) = val.get("_sweep_angle_max").and_then(|v| v.as_f64()) {
                    self.sweep_angle_max_deg = max;
                }
            }
        }

        // Solve at the shared angle (if present) or t=0.
        let target_angle = shared_angle.unwrap_or(driver_theta_0);
        let target_t = (target_angle - driver_theta_0) / driver_omega;

        let q0 = mech.state().make_q();
        let mut converged = self.solve_and_update(&mech, &q0, target_t, 1e-10, 50, Some(q0.clone()));

        if !converged {
            // Try solving at several angles and continuing to t=0.
            use crate::solver::kinematics::solve_position;
            let try_angles = [
                std::f64::consts::FRAC_PI_4,
                std::f64::consts::FRAC_PI_2,
                std::f64::consts::PI,
                -std::f64::consts::FRAC_PI_4,
            ];
            for &start_angle in &try_angles {
                let t_start = (start_angle - driver_theta_0) / driver_omega;
                let q_zero = mech.state().make_q();
                if let Ok(result) = solve_position(&mech, &q_zero, t_start, 1e-10, 100) {
                    if result.converged {
                        // Continuation: step from start_angle to target angle
                        let steps = 20;
                        let mut q_cont = result.q;
                        let mut cont_ok = true;
                        for i in 1..=steps {
                            let frac = i as f64 / steps as f64;
                            let t = t_start + (target_t - t_start) * frac;
                            match solve_position(&mech, &q_cont, t, 1e-10, 100) {
                                Ok(r) if r.converged => q_cont = r.q,
                                _ => { cont_ok = false; break; }
                            }
                        }
                        if cont_ok {
                            converged = self.solve_and_update(
                                &mech, &q_cont, target_t, 1e-10, 50, Some(q_cont.clone()),
                            );
                            if converged { break; }
                        }
                    }
                }
            }
        }

        self.driver_omega = driver_omega;
        self.driver_theta_0 = driver_theta_0;
        self.driver_angle = target_angle;
        self.q_at_zero = self.q.clone();
        self.driver_joint_id = driver_joint_id;
        // Initialize stroke range from the first LinearActuator force element
        // (if any) so the stroke sweep UI has sensible defaults.
        self.sweep_stroke_min = 0.0;
        self.sweep_stroke_max = 0.0;
        for force in mech.forces() {
            if let ForceElement::LinearActuator(act) = force {
                if act.stroke_min > 0.0 || act.stroke_max > 0.0 {
                    self.sweep_stroke_min = act.stroke_min;
                    self.sweep_stroke_max = act.stroke_max;
                    break;
                }
            }
        }

        self.mechanism = Some(mech);
        self.current_sample = None;
        self.selected = None;

        // Restore mounting angle from the blueprint.
        if let Some(ref bp) = self.blueprint {
            self.mounting_angle = bp.mounting_angle;
        }

        // Restore load cases from the blueprint, or create a default one
        if let Some(ref bp) = self.blueprint {
            if !bp.load_cases.is_empty() {
                self.load_cases = LoadCaseManager {
                    cases: bp.load_cases.clone(),
                    active_index: 0,
                };
            } else if let Some(ref joint_id) = self.driver_joint_id {
                self.load_cases = LoadCaseManager::new_default(
                    joint_id,
                    self.driver_omega,
                    self.driver_theta_0,
                );
            } else {
                self.load_cases = LoadCaseManager::default();
            }
        } else if let Some(ref joint_id) = self.driver_joint_id {
            self.load_cases =
                LoadCaseManager::new_default(joint_id, self.driver_omega, self.driver_theta_0);
        } else {
            self.load_cases = LoadCaseManager::default();
        }

        self.playing = false;
        self.animation_direction = 1.0;
        self.pending_driver_reassignment = None;
        self.undo_history.clear();
        self.auto_grid_spacing();
        self.compute_forces(0.0);
        self.update_grashof();
        self.compute_sweep();
        self.compute_validation();
        self.dirty = false;
        self.autosave_timer = 0.0;

        Ok(())
    }

    /// Load a mechanism from a JSON file, solve at t=0, and update all state.
    ///
    /// After loading, the driver is restored from the JSON. If the file has no
    /// driver, the mechanism is loaded without one (the user can assign one via
    /// the Driver panel).
    ///
    /// Returns `Err` with a human-readable message on any failure.
    pub fn load_from_file(&mut self, path: &Path) -> Result<(), String> {
        let json_str =
            std::fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

        self.load_from_json_str(&json_str)?;

        self.last_save_path = Some(path.to_path_buf());
        #[cfg(not(target_arch = "wasm32"))]
        self.add_recent_file(path);

        Ok(())
    }

    // ── WASM autosave (localStorage) ─────────────────────────────────────

    /// Perform periodic autosave to localStorage on WASM.
    ///
    /// Called from the update loop with accumulated dt. Saves every 30 seconds
    /// if there are unsaved changes and a mechanism is loaded.
    #[cfg(target_arch = "wasm32")]
    pub fn tick_autosave(&mut self, dt: f64) {
        const AUTOSAVE_INTERVAL: f64 = 30.0;

        if !self.dirty || self.mechanism.is_none() {
            return;
        }

        self.autosave_timer += dt;
        if self.autosave_timer < AUTOSAVE_INTERVAL {
            return;
        }
        self.autosave_timer = 0.0;

        match self.serialize_to_json_string() {
            Ok(json) => {
                Self::wasm_save_autosave(&json);
                log::debug!("WASM autosaved to localStorage");
            }
            Err(e) => {
                log::warn!("WASM autosave serialization failed: {}", e);
            }
        }
    }

    /// Save a JSON string to localStorage under the autosave key.
    #[cfg(target_arch = "wasm32")]
    fn wasm_save_autosave(json_str: &str) {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(storage)) = window.local_storage() {
                let _ = storage.set_item("linkage_autosave", json_str);
            }
        }
    }

    /// Load the autosave JSON string from localStorage, if present.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn wasm_load_autosave() -> Option<String> {
        let window = web_sys::window()?;
        let storage = window.local_storage().ok()??;
        storage.get_item("linkage_autosave").ok()?
    }

    /// Remove the autosave entry from localStorage.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn wasm_clear_autosave() {
        if let Some(window) = web_sys::window() {
            if let Ok(Some(storage)) = window.local_storage() {
                let _ = storage.remove_item("linkage_autosave");
            }
        }
    }

    /// Check localStorage for a recoverable autosave on WASM startup.
    ///
    /// Returns `true` if a non-empty autosave string is present.
    /// Unlike native, there's no filesystem timestamp, so any existing
    /// autosave is considered recoverable.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn check_wasm_autosave_recovery() -> bool {
        Self::wasm_load_autosave()
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn share_url_encode_decode_roundtrip() {
        let json = r#"{"schema_version":"1.0.0","bodies":{},"joints":{},"drivers":{}}"#;
        let encoded = encode_mechanism_for_url(json);
        // Encoded should be URL-safe (no +, /, or = padding with URL_SAFE_NO_PAD).
        assert!(!encoded.contains('+'), "encoded contains '+': not URL-safe");
        assert!(!encoded.contains('/'), "encoded contains '/': not URL-safe");
        let decoded = decode_mechanism_from_url(&encoded).expect("decode failed");
        assert_eq!(decoded, json);
    }

    #[test]
    fn share_url_compressed_is_shorter_than_json() {
        // A realistic-ish JSON payload.
        let json = r#"{"schema_version":"1.0.0","bodies":{"ground":{"attachment_points":{"A":[0.0,0.0],"B":[0.1,0.0]},"mass":0.0,"cg_local":[0.0,0.0],"izz_cg":0.0}},"joints":{},"drivers":{},"load_cases":[],"forces":[],"mounting_angle":0.0,"linear_drivers":[]}"#;
        let encoded = encode_mechanism_for_url(json);
        // Deflate + base64 should be shorter than raw base64 of the JSON.
        assert!(
            encoded.len() < json.len(),
            "encoded ({}) not shorter than raw JSON ({})",
            encoded.len(),
            json.len()
        );
    }

    #[test]
    fn decode_invalid_base64_returns_error() {
        let result = decode_mechanism_from_url("!!!not_valid_base64!!!");
        assert!(result.is_err());
    }

    #[test]
    fn decode_valid_base64_but_not_deflate_returns_error() {
        use base64::Engine;
        let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b"not compressed data");
        let result = decode_mechanism_from_url(&encoded);
        assert!(result.is_err());
    }
}
