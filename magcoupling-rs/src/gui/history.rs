//! Undo and redo of input changes (spec M4 "Session": "undo/redo of input changes").
//!
//! [`History`] keeps snapshots of the state (the panel's [`crate::gui::session::Design`]: the
//! inputs and the sizing state). The panel shows it the state once per frame with
//! [`History::observe`]; a change becomes one undo step when the edit has settled (no pointer
//! button down, no key held down, no text field of the design focused), so a drag, a typed
//! value or a part name typed letter by letter is one step, each arrow-key nudge (a key
//! pressed and released) is its own, and a held arrow key's whole auto-repeat run is one
//! (decision M41-14). Reset all and loading a design or a share link are edits like any
//! other, so they can be undone; a design the host opens as the session's start (a share link
//! at start-up) is not an edit: the panel starts a new history there.

use std::collections::VecDeque;

/// Undo levels kept; the oldest is dropped past this.
pub const MAX_UNDO_LEVELS: usize = 100;

/// Undo and redo stacks of state snapshots.
#[derive(Clone, Debug)]
pub struct History<T> {
    /// The state as of the last settled change: what the next undo step starts from.
    committed: T,
    /// Earlier settled states, oldest first.
    undo: VecDeque<T>,
    /// Undone states, the next redo last.
    redo: Vec<T>,
    max_levels: usize,
}

impl<T: Clone + PartialEq> History<T> {
    /// A history starting at `initial`, with nothing to undo, keeping [`MAX_UNDO_LEVELS`].
    pub fn new(initial: T) -> Self {
        Self::with_levels(initial, MAX_UNDO_LEVELS)
    }

    /// A history keeping at most `max_levels` undo steps (at least one).
    pub fn with_levels(initial: T, max_levels: usize) -> Self {
        Self {
            committed: initial,
            undo: VecDeque::new(),
            redo: Vec::new(),
            max_levels: max_levels.max(1),
        }
    }

    /// Records `current` as one undo step if it differs from the last settled state and the
    /// edit has `settled`. Returns whether it recorded a step.
    pub fn observe(&mut self, current: &T, settled: bool) -> bool {
        if settled && *current != self.committed {
            self.commit(current);
            true
        } else {
            false
        }
    }

    /// The state before the last change: an edit not yet settled counts as the last change.
    /// `None` when there is nothing to undo.
    pub fn undo(&mut self, current: &T) -> Option<T> {
        if *current != self.committed {
            self.commit(current);
        }
        let previous = self.undo.pop_back()?;
        let undone = std::mem::replace(&mut self.committed, previous.clone());
        self.redo.push(undone);
        Some(previous)
    }

    /// The state the last undo left. `None` when nothing was undone, or a change was made since
    /// (that change is recorded and the redo steps are dropped, as after any edit).
    pub fn redo(&mut self, current: &T) -> Option<T> {
        if *current != self.committed {
            self.commit(current);
            return None;
        }
        let next = self.redo.pop()?;
        let left = std::mem::replace(&mut self.committed, next.clone());
        self.undo.push_back(left);
        Some(next)
    }

    /// Whether [`History::undo`] would give a state for `current`.
    pub fn can_undo(&self, current: &T) -> bool {
        !self.undo.is_empty() || *current != self.committed
    }

    /// Whether [`History::redo`] would give a state for `current`.
    pub fn can_redo(&self, current: &T) -> bool {
        !self.redo.is_empty() && *current == self.committed
    }

    /// The number of undo steps recorded (an unsettled edit not included).
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    fn commit(&mut self, current: &T) {
        let previous = std::mem::replace(&mut self.committed, current.clone());
        self.undo.push_back(previous);
        while self.undo.len() > self.max_levels {
            self.undo.pop_front();
        }
        self.redo.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_settled_change_is_one_step_and_undo_and_redo_walk_it() {
        let mut history = History::new(0);
        assert!(!history.can_undo(&0));
        assert!(history.observe(&1, true));
        assert!(history.observe(&2, true));
        assert_eq!(history.undo_len(), 2);
        assert_eq!(history.undo(&2), Some(1));
        assert_eq!(history.undo(&1), Some(0));
        assert_eq!(history.undo(&0), None);
        assert!(history.can_redo(&0));
        assert_eq!(history.redo(&0), Some(1));
        assert_eq!(history.redo(&1), Some(2));
        assert_eq!(history.redo(&2), None);
    }

    #[test]
    fn an_unsettled_edit_is_coalesced_into_one_step() {
        // A drag: many frames with the pointer down, then the release.
        let mut history = History::new(0);
        for value in 1..=5 {
            assert!(!history.observe(&value, false));
        }
        assert_eq!(history.undo_len(), 0);
        assert!(history.observe(&5, true));
        assert_eq!(history.undo_len(), 1);
        assert_eq!(history.undo(&5), Some(0));
    }

    #[test]
    fn an_unchanged_state_records_nothing() {
        let mut history = History::new(7);
        for _ in 0..3 {
            assert!(!history.observe(&7, true));
        }
        // A drag that ends where it started is no step either.
        assert!(!history.observe(&8, false));
        assert!(!history.observe(&7, true));
        assert_eq!(history.undo_len(), 0);
    }

    #[test]
    fn undo_during_an_unsettled_edit_reverts_that_edit() {
        let mut history = History::new(0);
        history.observe(&1, true);
        assert!(!history.observe(&9, false));
        assert!(history.can_undo(&9));
        assert_eq!(history.undo(&9), Some(1));
        assert_eq!(history.redo(&1), Some(9));
    }

    #[test]
    fn a_new_change_drops_the_redo_steps() {
        let mut history = History::new(0);
        history.observe(&1, true);
        assert_eq!(history.undo(&1), Some(0));
        assert!(history.observe(&5, true));
        assert!(!history.can_redo(&5));
        assert_eq!(history.redo(&5), None);
        assert_eq!(history.undo(&5), Some(0));
    }

    #[test]
    fn redo_after_an_unrecorded_change_records_it_and_gives_nothing() {
        let mut history = History::new(0);
        history.observe(&1, true);
        assert_eq!(history.undo(&1), Some(0));
        assert!(!history.can_redo(&3));
        assert_eq!(history.redo(&3), None);
        assert_eq!(history.undo(&3), Some(0));
    }

    #[test]
    fn the_oldest_step_is_dropped_past_the_limit() {
        let mut history = History::with_levels(0, 3);
        for value in 1..=5 {
            history.observe(&value, true);
        }
        assert_eq!(history.undo_len(), 3);
        assert_eq!(history.undo(&5), Some(4));
        assert_eq!(history.undo(&4), Some(3));
        assert_eq!(history.undo(&3), Some(2));
        assert_eq!(history.undo(&2), None);
        assert_eq!(History::with_levels(0, 0).max_levels, 1);
        assert_eq!(History::new(0).max_levels, MAX_UNDO_LEVELS);
    }
}
