// ── Load cases ────────────────────────────────────────────────────────────────

use crate::io::LoadCaseJson;

/// A named driver configuration on the same mechanism geometry.
///
/// Engineers use load cases to compare different operating conditions
/// (which joint is driven, speed, direction) without rebuilding the mechanism.
///
/// Type alias for `LoadCaseJson` — the canonical definition lives in
/// `io::schema` so load cases can be serialized/deserialized
/// alongside the mechanism JSON.
pub type LoadCase = LoadCaseJson;

/// Manages multiple load cases for the current mechanism.
#[derive(Debug, Clone)]
pub struct LoadCaseManager {
    pub cases: Vec<LoadCase>,
    pub active_index: usize,
}

impl Default for LoadCaseManager {
    fn default() -> Self {
        Self {
            cases: Vec::new(),
            active_index: 0,
        }
    }
}

impl LoadCaseManager {
    /// Create a manager with a single default load case from the current driver settings.
    pub fn new_default(driver_joint_id: &str, omega: f64, theta_0: f64) -> Self {
        Self {
            cases: vec![LoadCase {
                name: "Default".to_string(),
                driver_joint_id: driver_joint_id.to_string(),
                omega,
                theta_0,
            }],
            active_index: 0,
        }
    }

    /// Add a new load case by copying the current driver settings.
    ///
    /// Returns the index of the newly added case.
    pub fn add_case(&mut self, driver_joint_id: &str, omega: f64, theta_0: f64) -> usize {
        let n = self.cases.len() + 1;
        self.cases.push(LoadCase {
            name: format!("Case {}", n),
            driver_joint_id: driver_joint_id.to_string(),
            omega,
            theta_0,
        });
        self.cases.len() - 1
    }

    /// Remove the load case at the given index.
    ///
    /// Returns false (no-op) if there is only one case remaining.
    pub fn remove_case(&mut self, index: usize) -> bool {
        if self.cases.len() <= 1 || index >= self.cases.len() {
            return false;
        }
        self.cases.remove(index);
        // Adjust active_index if it's out of bounds or was pointing at the removed case
        if self.active_index >= self.cases.len() {
            self.active_index = self.cases.len() - 1;
        }
        true
    }
}
