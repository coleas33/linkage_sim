//! Undo/redo infrastructure using mechanism state snapshots.
//!
//! Uses the memento pattern: before each undoable action, the current mechanism
//! state is serialized to a JSON snapshot and pushed onto the undo stack.
//! Undo/redo swap snapshots between undo and redo stacks.

use std::collections::VecDeque;

/// Snapshot of the mechanism state for undo/redo.
///
/// Captures only the "document" state that the user would want to undo:
/// the mechanism itself plus driver parameters. Does NOT include view
/// transform, selection, or animation state.
#[derive(Clone, Debug)]
pub struct MechanismSnapshot {
    /// Serialized mechanism JSON (compact representation).
    pub mechanism_json: String,
    /// Driver angle at time of snapshot.
    pub driver_angle: f64,
    /// Driver angular velocity (rad/s).
    pub driver_omega: f64,
    /// Driver initial angle (rad) at t=0.
    pub driver_theta_0: f64,
    /// Which joint is currently driven.
    pub driver_joint_id: Option<String>,
    /// Generalized coordinates at snapshot time, used as initial guess on restore.
    pub q: Vec<f64>,
}

/// Manages undo/redo stacks of mechanism snapshots.
pub struct UndoHistory {
    /// Stack of past states (most recent at back).
    undo_stack: VecDeque<MechanismSnapshot>,
    /// Stack of undone states (for redo).
    redo_stack: VecDeque<MechanismSnapshot>,
    /// Maximum number of undo levels.
    max_levels: usize,
}

impl UndoHistory {
    /// Create a new UndoHistory with the given maximum number of undo levels.
    pub fn new(max_levels: usize) -> Self {
        Self {
            undo_stack: VecDeque::new(),
            redo_stack: VecDeque::new(),
            max_levels,
        }
    }

    /// Push a snapshot onto the undo stack (called BEFORE each undoable action).
    ///
    /// Clears the redo stack, since a new action invalidates any undone states.
    /// If the undo stack exceeds `max_levels`, the oldest entry is dropped.
    pub fn push(&mut self, snapshot: MechanismSnapshot) {
        self.redo_stack.clear();
        self.undo_stack.push_back(snapshot);
        if self.undo_stack.len() > self.max_levels {
            self.undo_stack.pop_front();
        }
    }

    /// Undo the last action: push `current` onto the redo stack and pop from undo.
    ///
    /// Returns the snapshot to restore, or `None` if there is nothing to undo.
    pub fn undo(&mut self, current: MechanismSnapshot) -> Option<MechanismSnapshot> {
        let previous = self.undo_stack.pop_back()?;
        self.redo_stack.push_back(current);
        Some(previous)
    }

    /// Redo the last undone action: push `current` onto the undo stack and pop from redo.
    ///
    /// Returns the snapshot to restore, or `None` if there is nothing to redo.
    pub fn redo(&mut self, current: MechanismSnapshot) -> Option<MechanismSnapshot> {
        let next = self.redo_stack.pop_back()?;
        self.undo_stack.push_back(current);
        Some(next)
    }

    /// Returns true if there is at least one state to undo.
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Returns true if there is at least one state to redo.
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Clear all undo/redo history (e.g., on new mechanism load).
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_snapshot(label: &str) -> MechanismSnapshot {
        MechanismSnapshot {
            mechanism_json: label.to_string(),
            driver_angle: 0.0,
            driver_omega: 1.0,
            driver_theta_0: 0.0,
            driver_joint_id: None,
            q: Vec::new(),
        }
    }

    #[test]
    fn new_history_is_empty() {
        let history = UndoHistory::new(50);
        assert!(!history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn push_enables_undo() {
        let mut history = UndoHistory::new(50);
        history.push(make_snapshot("state_0"));
        assert!(history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn undo_returns_previous_state() {
        let mut history = UndoHistory::new(50);
        history.push(make_snapshot("state_0"));

        let restored = history.undo(make_snapshot("state_1")).unwrap();
        assert_eq!(restored.mechanism_json, "state_0");
        assert!(!history.can_undo());
        assert!(history.can_redo());
    }

    #[test]
    fn redo_returns_undone_state() {
        let mut history = UndoHistory::new(50);
        history.push(make_snapshot("state_0"));
        history.undo(make_snapshot("state_1")).unwrap();

        let restored = history.redo(make_snapshot("state_0")).unwrap();
        assert_eq!(restored.mechanism_json, "state_1");
        assert!(history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn undo_on_empty_returns_none() {
        let mut history = UndoHistory::new(50);
        assert!(history.undo(make_snapshot("current")).is_none());
    }

    #[test]
    fn redo_on_empty_returns_none() {
        let mut history = UndoHistory::new(50);
        assert!(history.redo(make_snapshot("current")).is_none());
    }

    #[test]
    fn push_clears_redo_stack() {
        let mut history = UndoHistory::new(50);
        history.push(make_snapshot("state_0"));
        history.push(make_snapshot("state_1"));
        // Undo once so redo has something
        history.undo(make_snapshot("state_2")).unwrap();
        assert!(history.can_redo());

        // New push should clear redo
        history.push(make_snapshot("state_3"));
        assert!(!history.can_redo());
    }

    #[test]
    fn clear_empties_both_stacks() {
        let mut history = UndoHistory::new(50);
        history.push(make_snapshot("state_0"));
        history.push(make_snapshot("state_1"));
        history.undo(make_snapshot("state_2")).unwrap();

        history.clear();
        assert!(!history.can_undo());
        assert!(!history.can_redo());
    }

    #[test]
    fn max_levels_drops_oldest() {
        let mut history = UndoHistory::new(3);
        history.push(make_snapshot("state_0"));
        history.push(make_snapshot("state_1"));
        history.push(make_snapshot("state_2"));
        // At capacity (3). Push one more should drop state_0.
        history.push(make_snapshot("state_3"));

        // We should be able to undo 3 times (state_3, state_2, state_1)
        let r1 = history.undo(make_snapshot("state_4")).unwrap();
        assert_eq!(r1.mechanism_json, "state_3");

        let r2 = history.undo(make_snapshot("state_3")).unwrap();
        assert_eq!(r2.mechanism_json, "state_2");

        let r3 = history.undo(make_snapshot("state_2")).unwrap();
        assert_eq!(r3.mechanism_json, "state_1");

        // state_0 was dropped, so this should return None
        assert!(history.undo(make_snapshot("state_1")).is_none());
    }

    #[test]
    fn multiple_undo_redo_cycle() {
        let mut history = UndoHistory::new(50);
        history.push(make_snapshot("state_0"));
        history.push(make_snapshot("state_1"));
        history.push(make_snapshot("state_2"));

        // Current state is conceptually "state_3"
        let r = history.undo(make_snapshot("state_3")).unwrap();
        assert_eq!(r.mechanism_json, "state_2");

        let r = history.undo(make_snapshot("state_2")).unwrap();
        assert_eq!(r.mechanism_json, "state_1");

        // Redo back
        let r = history.redo(make_snapshot("state_1")).unwrap();
        assert_eq!(r.mechanism_json, "state_2");

        let r = history.redo(make_snapshot("state_2")).unwrap();
        assert_eq!(r.mechanism_json, "state_3");

        // No more redo
        assert!(!history.can_redo());
    }

    #[test]
    fn snapshot_preserves_driver_fields() {
        let snapshot = MechanismSnapshot {
            mechanism_json: "{}".to_string(),
            driver_angle: 1.5,
            driver_omega: 3.14,
            driver_theta_0: 0.5,
            driver_joint_id: Some("J1".to_string()),
            q: vec![1.0, 2.0, 3.0],
        };

        let mut history = UndoHistory::new(50);
        history.push(snapshot);

        let current = make_snapshot("current");
        let restored = history.undo(current).unwrap();

        assert_eq!(restored.driver_angle, 1.5);
        assert_eq!(restored.driver_omega, 3.14);
        assert_eq!(restored.driver_theta_0, 0.5);
        assert_eq!(restored.driver_joint_id, Some("J1".to_string()));
    }
}
