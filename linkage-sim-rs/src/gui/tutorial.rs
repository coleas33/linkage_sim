//! Interactive step-by-step tutorial overlay for the linkage simulator.
//!
//! Currently provides one tutorial: "Build a 4-Bar Linkage", which walks the
//! user through placing ground pivots, drawing links, setting a driver, and
//! running the animation.

use eframe::egui;

use super::state::AppState;

// ── Tutorial data structures ─────────────────────────────────────────────────

/// A single step in a tutorial sequence.
pub struct TutorialStep {
    /// Short title shown in bold at the top of the overlay.
    pub title: &'static str,
    /// Multi-line description with instructions for the user.
    pub description: &'static str,
    /// Returns `true` when the user has completed this step's objective.
    /// The "Next" button is always available (so users can skip), but it
    /// highlights green when this returns `true`.
    pub is_complete: fn(&AppState) -> bool,
}

/// Tracks the active tutorial state. When `active` is false, no overlay is
/// drawn and no tutorial logic runs.
pub struct TutorialState {
    /// Whether a tutorial is currently active.
    pub active: bool,
    /// Current step index (0-based).
    pub step: usize,
    /// The ordered list of tutorial steps.
    pub steps: Vec<TutorialStep>,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self {
            active: false,
            step: 0,
            steps: Vec::new(),
        }
    }
}

impl TutorialState {
    /// Create the "Build a 4-Bar" tutorial sequence.
    pub fn new_fourbar() -> Self {
        Self {
            active: true,
            step: 0,
            steps: fourbar_steps(),
        }
    }

    /// Advance to the next step if one exists. If already on the last step,
    /// deactivate the tutorial.
    pub fn advance(&mut self) {
        if self.step + 1 < self.steps.len() {
            self.step += 1;
        } else {
            self.active = false;
        }
    }

    /// Go back to the previous step (clamped at 0).
    pub fn go_back(&mut self) {
        self.step = self.step.saturating_sub(1);
    }

    /// Close the tutorial without completing it.
    pub fn close(&mut self) {
        self.active = false;
    }

    /// Returns `true` if the current step's completion condition is met.
    pub fn current_step_complete(&self, state: &AppState) -> bool {
        self.steps
            .get(self.step)
            .map(|s| (s.is_complete)(state))
            .unwrap_or(false)
    }
}

// ── 4-Bar tutorial steps ─────────────────────────────────────────────────────

fn ground_attachment_count(state: &AppState) -> usize {
    state
        .blueprint
        .as_ref()
        .and_then(|bp| bp.bodies.get("ground"))
        .map(|g| g.attachment_points.len())
        .unwrap_or(0)
}

fn moving_body_count(state: &AppState) -> usize {
    state
        .blueprint
        .as_ref()
        .map(|bp| bp.bodies.keys().filter(|k| *k != "ground").count())
        .unwrap_or(0)
}

fn joint_count(state: &AppState) -> usize {
    state
        .blueprint
        .as_ref()
        .map(|bp| bp.joints.len())
        .unwrap_or(0)
}

fn fourbar_steps() -> Vec<TutorialStep> {
    vec![
        TutorialStep {
            title: "Welcome",
            description: "This tutorial will guide you through building a \
                4-bar linkage mechanism from scratch.\n\n\
                Click Next to begin.",
            is_complete: |_| true,
        },
        TutorialStep {
            title: "Place Ground Pivots",
            description: "1. Click '+ Ground' in the toolbar\n\
                2. Click on the canvas to place a ground pivot\n\
                3. Place a second ground pivot about 40mm to the right",
            is_complete: |state| ground_attachment_count(state) >= 2,
        },
        TutorialStep {
            title: "Draw the Crank",
            description: "1. Select 'Draw Link' from the toolbar\n\
                2. Click on the first ground pivot\n\
                3. Drag to create a short crank (~20mm)",
            is_complete: |state| moving_body_count(state) >= 1,
        },
        TutorialStep {
            title: "Draw the Coupler",
            description: "1. Click on the crank's free end\n\
                2. Drag to create the coupler link (~40mm)",
            is_complete: |state| moving_body_count(state) >= 2,
        },
        TutorialStep {
            title: "Draw the Rocker",
            description: "1. Click on the coupler's free end\n\
                2. Drag to the second ground pivot\n\
                3. This closes the kinematic loop",
            is_complete: |state| moving_body_count(state) >= 3 && joint_count(state) >= 4,
        },
        TutorialStep {
            title: "Set the Driver",
            description: "1. Right-click on the first ground joint (J1)\n\
                2. Select 'Set as Driver' from the context menu",
            is_complete: |state| state.driver_joint_id.is_some(),
        },
        TutorialStep {
            title: "Play!",
            description: "Click Play to animate your mechanism.\n\
                Drag the crank angle slider to explore different positions.\n\n\
                Congratulations! You built a 4-bar linkage.",
            is_complete: |_| true,
        },
    ]
}

// ── Overlay rendering ────────────────────────────────────────────────────────

/// Draw the tutorial overlay panel when a tutorial is active.
///
/// This is a floating `egui::Window` anchored near the top-right corner.
pub fn draw_tutorial_overlay(ctx: &egui::Context, state: &mut AppState) {
    if !state.tutorial.active {
        return;
    }

    let total = state.tutorial.steps.len();
    let current = state.tutorial.step;

    // Check completion *before* borrowing tutorial mutably for the UI.
    let step_complete = state.tutorial.current_step_complete(state);

    let (title, description) = state
        .tutorial
        .steps
        .get(current)
        .map(|s| (s.title, s.description))
        .unwrap_or(("", ""));

    let mut should_close = false;
    let mut should_advance = false;
    let mut should_go_back = false;

    egui::Window::new("Tutorial")
        .collapsible(false)
        .resizable(false)
        .default_width(320.0)
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 50.0))
        .show(ctx, |ui| {
            // Step counter
            ui.label(
                egui::RichText::new(format!("Step {} of {}", current + 1, total))
                    .small()
                    .color(egui::Color32::LIGHT_GRAY),
            );

            ui.add_space(4.0);

            // Title
            ui.label(egui::RichText::new(title).strong().size(16.0));

            ui.add_space(6.0);

            // Description
            ui.label(description);

            ui.add_space(10.0);

            // Completion indicator
            if step_complete {
                ui.label(
                    egui::RichText::new("Step complete!")
                        .color(egui::Color32::from_rgb(100, 200, 100))
                        .italics(),
                );
                ui.add_space(4.0);
            }

            // Navigation buttons
            ui.horizontal(|ui| {
                if current > 0 && ui.button("Back").clicked() {
                    should_go_back = true;
                }

                let next_text = if current + 1 >= total {
                    "Finish"
                } else {
                    "Next"
                };

                let next_button = if step_complete {
                    egui::Button::new(
                        egui::RichText::new(next_text)
                            .color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(40, 120, 80))
                } else {
                    egui::Button::new(next_text)
                };

                if ui.add(next_button).clicked() {
                    should_advance = true;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button(
                            egui::RichText::new("Close Tutorial")
                                .color(egui::Color32::LIGHT_GRAY),
                        )
                        .clicked()
                    {
                        should_close = true;
                    }
                });
            });
        });

    // Apply deferred actions (avoids mutable borrow conflicts).
    if should_close {
        state.tutorial.close();
    } else if should_advance {
        state.tutorial.advance();
    } else if should_go_back {
        state.tutorial.go_back();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tutorial_is_inactive() {
        let tutorial = TutorialState::default();
        assert!(!tutorial.active);
        assert_eq!(tutorial.step, 0);
        assert!(tutorial.steps.is_empty());
    }

    #[test]
    fn new_fourbar_starts_active_at_step_zero() {
        let tutorial = TutorialState::new_fourbar();
        assert!(tutorial.active);
        assert_eq!(tutorial.step, 0);
        assert_eq!(tutorial.steps.len(), 7);
    }

    #[test]
    fn advance_increments_step() {
        let mut tutorial = TutorialState::new_fourbar();
        assert_eq!(tutorial.step, 0);
        tutorial.advance();
        assert_eq!(tutorial.step, 1);
        assert!(tutorial.active);
    }

    #[test]
    fn advance_past_last_step_deactivates() {
        let mut tutorial = TutorialState::new_fourbar();
        let total = tutorial.steps.len();
        for _ in 0..total {
            tutorial.advance();
        }
        assert!(!tutorial.active);
    }

    #[test]
    fn go_back_decrements_step() {
        let mut tutorial = TutorialState::new_fourbar();
        tutorial.advance();
        tutorial.advance();
        assert_eq!(tutorial.step, 2);
        tutorial.go_back();
        assert_eq!(tutorial.step, 1);
    }

    #[test]
    fn go_back_clamps_at_zero() {
        let mut tutorial = TutorialState::new_fourbar();
        tutorial.go_back();
        assert_eq!(tutorial.step, 0);
    }

    #[test]
    fn close_deactivates() {
        let mut tutorial = TutorialState::new_fourbar();
        assert!(tutorial.active);
        tutorial.close();
        assert!(!tutorial.active);
    }

    #[test]
    fn welcome_step_is_always_complete() {
        let state = AppState::default();
        let tutorial = TutorialState::new_fourbar();
        let is_complete = (tutorial.steps[0].is_complete)(&state);
        assert!(is_complete, "Welcome step should always be completable");
    }

    #[test]
    fn ground_pivot_step_detects_two_pivots() {
        let mut state = AppState::default();
        // Default state has empty ground with no attachment points.
        let is_complete = (TutorialState::new_fourbar().steps[1].is_complete)(&state);
        assert!(
            !is_complete,
            "Should not be complete with no ground pivots"
        );

        // Add two attachment points to ground.
        if let Some(ref mut bp) = state.blueprint {
            if let Some(ground) = bp.bodies.get_mut("ground") {
                ground
                    .attachment_points
                    .insert("A".to_string(), [0.0, 0.0]);
                ground
                    .attachment_points
                    .insert("B".to_string(), [0.04, 0.0]);
            }
        }
        let is_complete = (TutorialState::new_fourbar().steps[1].is_complete)(&state);
        assert!(is_complete, "Should be complete with 2 ground pivots");
    }

    #[test]
    fn driver_step_detects_driver_set() {
        let mut state = AppState::default();
        let steps = fourbar_steps();
        assert!(
            !(steps[5].is_complete)(&state),
            "Should not be complete without a driver"
        );
        state.driver_joint_id = Some("J1".to_string());
        assert!(
            (steps[5].is_complete)(&state),
            "Should be complete with a driver"
        );
    }

    #[test]
    fn current_step_complete_delegates_correctly() {
        let state = AppState::default();
        let tutorial = TutorialState::new_fourbar();
        // Step 0 (Welcome) is always complete.
        assert!(tutorial.current_step_complete(&state));
    }

    #[test]
    fn step_titles_are_unique() {
        let steps = fourbar_steps();
        let titles: Vec<&str> = steps.iter().map(|s| s.title).collect();
        for (i, title) in titles.iter().enumerate() {
            for (j, other) in titles.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        title, other,
                        "Duplicate step title: '{}' at indices {} and {}",
                        title, i, j
                    );
                }
            }
        }
    }

    #[test]
    fn moving_body_count_excludes_ground() {
        let state = AppState::default();
        // Default state has only the "ground" body.
        assert_eq!(moving_body_count(&state), 0);
    }
}
