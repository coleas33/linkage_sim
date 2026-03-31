//! Mechanism template management: save, load, delete named mechanism blueprints.
//!
//! On native: templates are persisted as individual JSON files under
//! `~/.linkage-sim/templates/`.
//!
//! On WASM: templates are persisted in localStorage with the prefix
//! `linkage_template_`.

use super::AppState;

impl AppState {
    /// Save the current blueprint as a named template.
    ///
    /// Serializes the blueprint to JSON, stores it in `saved_templates`, and
    /// persists to disk (native) or localStorage (WASM).
    pub fn save_as_template(&mut self, name: &str) {
        let Some(ref bp) = self.blueprint else { return };
        let Ok(json) = serde_json::to_string_pretty(bp) else { return };

        // Replace existing template with the same name, or append.
        if let Some(pos) = self.saved_templates.iter().position(|(n, _)| n == name) {
            self.saved_templates[pos].1 = json.clone();
        } else {
            self.saved_templates.push((name.to_string(), json.clone()));
        }

        self.persist_template(name, &json);

        self.status_message = Some(format!("Template saved: {}", name));
        self.status_message_time = 3.0;
    }

    /// Load a template by index, replacing the current mechanism.
    ///
    /// Delegates to `load_from_json_str` which handles deserialization,
    /// solving, driver detection, and all state resets.
    pub fn load_template(&mut self, index: usize) {
        let Some((name, json)) = self.saved_templates.get(index).cloned() else {
            return;
        };

        if let Err(e) = self.load_from_json_str(&json) {
            self.error_log.push(format!("Failed to load template '{}': {}", name, e));
            self.show_error_panel = true;
            return;
        }

        // Templates are not associated with a file path.
        self.last_save_path = None;
        self.pending_fit_to_view = true;

        self.status_message = Some(format!("Template loaded: {}", name));
        self.status_message_time = 3.0;
    }

    /// Delete a template by index, removing it from memory and persistent storage.
    pub fn delete_template(&mut self, index: usize) {
        if index >= self.saved_templates.len() {
            return;
        }
        let (name, _) = self.saved_templates.remove(index);
        self.remove_persisted_template(&name);

        self.status_message = Some(format!("Template deleted: {}", name));
        self.status_message_time = 3.0;
    }

    /// Load all saved templates from persistent storage into `saved_templates`.
    ///
    /// Called once at startup.
    pub(crate) fn load_saved_templates() -> Vec<(String, String)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Self::load_templates_native()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::load_templates_wasm()
        }
    }

    // ── Native persistence ──────────────────────────────────────────────────

    /// Directory for template files: `~/.linkage-sim/templates/`.
    #[cfg(not(target_arch = "wasm32"))]
    fn templates_dir() -> Option<std::path::PathBuf> {
        dirs::home_dir().map(|h| h.join(".linkage-sim").join("templates"))
    }

    /// Persist a single template to disk as `<name>.json`.
    #[cfg(not(target_arch = "wasm32"))]
    fn persist_template(&self, name: &str, json: &str) {
        let Some(dir) = Self::templates_dir() else { return };
        if std::fs::create_dir_all(&dir).is_err() {
            log::warn!("Failed to create templates directory: {:?}", dir);
            return;
        }
        let path = dir.join(format!("{}.json", sanitize_filename(name)));
        if let Err(e) = std::fs::write(&path, json) {
            log::warn!("Failed to write template '{}': {}", name, e);
        }
    }

    /// Remove a template file from disk.
    #[cfg(not(target_arch = "wasm32"))]
    fn remove_persisted_template(&self, name: &str) {
        let Some(dir) = Self::templates_dir() else { return };
        let path = dir.join(format!("{}.json", sanitize_filename(name)));
        let _ = std::fs::remove_file(&path);
    }

    /// Load all `.json` files from the templates directory.
    #[cfg(not(target_arch = "wasm32"))]
    fn load_templates_native() -> Vec<(String, String)> {
        let Some(dir) = Self::templates_dir() else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut templates = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(json) = std::fs::read_to_string(&path) {
                    // Derive name from filename (strip .json extension).
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unnamed")
                        .to_string();
                    templates.push((name, json));
                }
            }
        }
        templates.sort_by(|a, b| a.0.cmp(&b.0));
        templates
    }

    // ── WASM persistence ────────────────────────────────────────────────────

    #[cfg(target_arch = "wasm32")]
    fn persist_template(&self, name: &str, json: &str) {
        if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let key = format!("linkage_template_{}", name);
            let _ = storage.set_item(&key, json);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_persisted_template(&self, name: &str) {
        if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let key = format!("linkage_template_{}", name);
            let _ = storage.remove_item(&key);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn load_templates_wasm() -> Vec<(String, String)> {
        let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) else {
            return Vec::new();
        };
        let Ok(len) = storage.length() else {
            return Vec::new();
        };
        let mut templates = Vec::new();
        let prefix = "linkage_template_";
        for i in 0..len {
            if let Ok(Some(key)) = storage.key(i) {
                if let Some(name) = key.strip_prefix(prefix) {
                    if let Ok(Some(json)) = storage.get_item(&key) {
                        templates.push((name.to_string(), json));
                    }
                }
            }
        }
        templates.sort_by(|a, b| a.0.cmp(&b.0));
        templates
    }
}

/// Sanitize a template name for use as a filename.
///
/// Replaces characters that are invalid in file paths with underscores.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ' ' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create an AppState with an empty saved_templates list,
    /// ignoring any templates persisted on disk from previous runs.
    fn test_state() -> AppState {
        let mut state = AppState::default();
        state.saved_templates.clear();
        state
    }

    #[test]
    fn sanitize_filename_strips_special_chars() {
        assert_eq!(sanitize_filename("My Mech/v2"), "My Mech_v2");
        assert_eq!(sanitize_filename("test<>:"), "test___");
        assert_eq!(sanitize_filename("normal-name_1"), "normal-name_1");
    }

    #[test]
    fn save_as_template_replaces_existing() {
        let mut state = test_state();
        state.save_as_template("test");
        assert_eq!(state.saved_templates.len(), 1);

        // Saving with the same name should replace, not duplicate.
        state.save_as_template("test");
        assert_eq!(state.saved_templates.len(), 1);
    }

    #[test]
    fn save_and_load_template_roundtrip() {
        let mut state = test_state();
        // Save current (empty) blueprint as a template.
        state.save_as_template("roundtrip");
        assert_eq!(state.saved_templates.len(), 1);

        let original_bp_json = state.saved_templates[0].1.clone();

        // Load the template back.
        state.load_template(0);
        // The blueprint should match what was saved.
        let loaded_json = serde_json::to_string_pretty(state.blueprint.as_ref().unwrap()).unwrap();
        assert_eq!(loaded_json, original_bp_json);
    }

    #[test]
    fn delete_template_removes_entry() {
        let mut state = test_state();
        state.save_as_template("a");
        state.save_as_template("b");
        assert_eq!(state.saved_templates.len(), 2);

        state.delete_template(0);
        assert_eq!(state.saved_templates.len(), 1);
        assert_eq!(state.saved_templates[0].0, "b");
    }

    #[test]
    fn delete_template_out_of_bounds_is_noop() {
        let mut state = test_state();
        state.save_as_template("only");
        state.delete_template(5); // out of bounds
        assert_eq!(state.saved_templates.len(), 1);
    }

    #[test]
    fn load_template_invalid_index_is_noop() {
        let mut state = test_state();
        // No templates exist; loading index 0 should not panic.
        state.load_template(0);
        assert!(state.status_message.is_none());
    }
}
