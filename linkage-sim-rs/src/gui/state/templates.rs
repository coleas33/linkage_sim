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
        #[cfg(feature = "native")]
        {
            Self::load_templates_native()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::load_templates_wasm()
        }
        #[cfg(not(any(feature = "native", target_arch = "wasm32")))]
        {
            Vec::new()
        }
    }

    // ── Native persistence ──────────────────────────────────────────────────
    //
    // Gated by `feature = "native"` (not `not(target_arch = "wasm32")`)
    // because this code path uses the `dirs` crate which is only pulled
    // in when the feature is enabled. The two are normally synonymous on
    // desktop, but a `--no-default-features` native build (e.g. CI sanity
    // checks) needs them to differ.

    /// Directory for template files: `~/.linkage-sim/templates/`.
    #[cfg(feature = "native")]
    fn templates_dir() -> Option<std::path::PathBuf> {
        dirs::home_dir().map(|h| h.join(".linkage-sim").join("templates"))
    }

    /// Persist a single template to disk as `<name>.json`.
    #[cfg(feature = "native")]
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
    #[cfg(feature = "native")]
    fn remove_persisted_template(&self, name: &str) {
        let Some(dir) = Self::templates_dir() else { return };
        let path = dir.join(format!("{}.json", sanitize_filename(name)));
        let _ = std::fs::remove_file(&path);
    }

    /// Load all `.json` files from the templates directory.
    #[cfg(feature = "native")]
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

    // ── No-op fallback (no `native` feature, not wasm32) ────────────────────
    //
    // Kept callable from the ungated `save_as_template` / `delete_template`
    // entry points so a `--no-default-features` native build still links.
    // Templates simply don't persist — the in-memory `saved_templates`
    // vector still works for the lifetime of the session.

    #[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
    fn persist_template(&self, _name: &str, _json: &str) {}

    #[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
    fn remove_persisted_template(&self, _name: &str) {}

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

// ── Custom samples (user-promoted templates shown in the Samples dropdown) ──

impl AppState {
    /// Save the current blueprint as a named custom sample.
    ///
    /// Serializes the blueprint to JSON, stores it in `custom_samples`, and
    /// persists to disk (native) or localStorage (WASM).
    pub fn save_as_custom_sample(&mut self, name: &str) {
        let Some(ref bp) = self.blueprint else { return };
        let Ok(json) = serde_json::to_string_pretty(bp) else { return };

        // Replace existing sample with the same name, or append.
        if let Some(pos) = self.custom_samples.iter().position(|(n, _)| n == name) {
            self.custom_samples[pos].1 = json.clone();
        } else {
            self.custom_samples.push((name.to_string(), json.clone()));
        }

        self.persist_custom_sample(name, &json);

        self.status_message = Some(format!("Sample saved: {}", name));
        self.status_message_time = 3.0;
    }

    /// Load a custom sample by index, replacing the current mechanism.
    pub fn load_custom_sample(&mut self, index: usize) {
        let Some((name, json)) = self.custom_samples.get(index).cloned() else {
            return;
        };

        if let Err(e) = self.load_from_json_str(&json) {
            self.error_log.push(format!("Failed to load custom sample '{}': {}", name, e));
            self.show_error_panel = true;
            return;
        }

        self.last_save_path = None;
        self.pending_fit_to_view = true;

        self.status_message = Some(format!("Sample loaded: {}", name));
        self.status_message_time = 3.0;
    }

    /// Delete a custom sample by index, removing it from memory and persistent storage.
    pub fn delete_custom_sample(&mut self, index: usize) {
        if index >= self.custom_samples.len() {
            return;
        }
        let (name, _) = self.custom_samples.remove(index);
        self.remove_persisted_custom_sample(&name);

        self.status_message = Some(format!("Sample deleted: {}", name));
        self.status_message_time = 3.0;
    }

    /// Load all saved custom samples from persistent storage.
    ///
    /// Called once at startup.
    pub(crate) fn load_saved_custom_samples() -> Vec<(String, String)> {
        #[cfg(feature = "native")]
        {
            Self::load_custom_samples_native()
        }
        #[cfg(target_arch = "wasm32")]
        {
            Self::load_custom_samples_wasm()
        }
        #[cfg(not(any(feature = "native", target_arch = "wasm32")))]
        {
            Vec::new()
        }
    }

    // ── Native persistence (custom samples) ────────────────────────────

    /// Directory for custom sample files: `~/.linkage-sim/custom_samples/`.
    #[cfg(feature = "native")]
    fn custom_samples_dir() -> Option<std::path::PathBuf> {
        dirs::home_dir().map(|h| h.join(".linkage-sim").join("custom_samples"))
    }

    /// Persist a single custom sample to disk as `<name>.json`.
    #[cfg(feature = "native")]
    fn persist_custom_sample(&self, name: &str, json: &str) {
        let Some(dir) = Self::custom_samples_dir() else { return };
        if std::fs::create_dir_all(&dir).is_err() {
            log::warn!("Failed to create custom_samples directory: {:?}", dir);
            return;
        }
        let path = dir.join(format!("{}.json", sanitize_filename(name)));
        if let Err(e) = std::fs::write(&path, json) {
            log::warn!("Failed to write custom sample '{}': {}", name, e);
        }
    }

    /// Remove a custom sample file from disk.
    #[cfg(feature = "native")]
    fn remove_persisted_custom_sample(&self, name: &str) {
        let Some(dir) = Self::custom_samples_dir() else { return };
        let path = dir.join(format!("{}.json", sanitize_filename(name)));
        let _ = std::fs::remove_file(&path);
    }

    /// Load all `.json` files from the custom_samples directory.
    #[cfg(feature = "native")]
    fn load_custom_samples_native() -> Vec<(String, String)> {
        let Some(dir) = Self::custom_samples_dir() else {
            return Vec::new();
        };
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut samples = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(json) = std::fs::read_to_string(&path) {
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unnamed")
                        .to_string();
                    samples.push((name, json));
                }
            }
        }
        samples.sort_by(|a, b| a.0.cmp(&b.0));
        samples
    }

    // ── WASM persistence (custom samples) ──────────────────────────────

    #[cfg(target_arch = "wasm32")]
    fn persist_custom_sample(&self, name: &str, json: &str) {
        if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let key = format!("linkage_csample_{}", name);
            let _ = storage.set_item(&key, json);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn remove_persisted_custom_sample(&self, name: &str) {
        if let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) {
            let key = format!("linkage_csample_{}", name);
            let _ = storage.remove_item(&key);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn load_custom_samples_wasm() -> Vec<(String, String)> {
        let Some(storage) = web_sys::window().and_then(|w| w.local_storage().ok().flatten()) else {
            return Vec::new();
        };
        let Ok(len) = storage.length() else {
            return Vec::new();
        };
        let mut samples = Vec::new();
        let prefix = "linkage_csample_";
        for i in 0..len {
            if let Ok(Some(key)) = storage.key(i) {
                if let Some(name) = key.strip_prefix(prefix) {
                    if let Ok(Some(json)) = storage.get_item(&key) {
                        samples.push((name.to_string(), json));
                    }
                }
            }
        }
        samples.sort_by(|a, b| a.0.cmp(&b.0));
        samples
    }

    // ── No-op fallback (custom samples) ─────────────────────────────────────

    #[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
    fn persist_custom_sample(&self, _name: &str, _json: &str) {}

    #[cfg(all(not(feature = "native"), not(target_arch = "wasm32")))]
    fn remove_persisted_custom_sample(&self, _name: &str) {}
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

    /// Create an AppState with empty saved_templates and custom_samples lists,
    /// ignoring any entries persisted on disk from previous runs.
    fn test_state() -> AppState {
        let mut state = AppState::default();
        state.saved_templates.clear();
        state.custom_samples.clear();
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

    // ── Custom sample tests ─────────────────────────────────────────

    #[test]
    fn save_as_custom_sample_replaces_existing() {
        let mut state = test_state();
        state.save_as_custom_sample("test");
        assert_eq!(state.custom_samples.len(), 1);

        // Saving with the same name should replace, not duplicate.
        state.save_as_custom_sample("test");
        assert_eq!(state.custom_samples.len(), 1);
    }

    #[test]
    fn save_and_load_custom_sample_roundtrip() {
        let mut state = test_state();
        // Save current (empty) blueprint as a custom sample.
        state.save_as_custom_sample("roundtrip");
        assert_eq!(state.custom_samples.len(), 1);

        let original_bp_json = state.custom_samples[0].1.clone();

        // Load the custom sample back.
        state.load_custom_sample(0);
        // The blueprint should match what was saved.
        let loaded_json = serde_json::to_string_pretty(state.blueprint.as_ref().unwrap()).unwrap();
        assert_eq!(loaded_json, original_bp_json);
    }

    #[test]
    fn delete_custom_sample_removes_entry() {
        let mut state = test_state();
        state.save_as_custom_sample("a");
        state.save_as_custom_sample("b");
        assert_eq!(state.custom_samples.len(), 2);

        state.delete_custom_sample(0);
        assert_eq!(state.custom_samples.len(), 1);
        assert_eq!(state.custom_samples[0].0, "b");
    }

    #[test]
    fn delete_custom_sample_out_of_bounds_is_noop() {
        let mut state = test_state();
        state.save_as_custom_sample("only");
        state.delete_custom_sample(5); // out of bounds
        assert_eq!(state.custom_samples.len(), 1);
    }

    #[test]
    fn load_custom_sample_invalid_index_is_noop() {
        let mut state = test_state();
        // No custom samples exist; loading index 0 should not panic.
        state.load_custom_sample(0);
        assert!(state.status_message.is_none());
    }
}
