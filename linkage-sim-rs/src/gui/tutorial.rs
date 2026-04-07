//! Interactive step-by-step tutorial overlay for the linkage simulator.
//!
//! Currently provides one tutorial: "Build a 4-Bar Linkage", which walks the
//! user through placing ground pivots, drawing links, setting a driver, and
//! running the animation.

use eframe::egui;

use crate::forces::elements::ForceElement;

use super::samples::SampleMechanism;
use super::state::{AppState, MotionProfile};

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

    /// Create the "Actuator Sizing" tutorial sequence.
    pub fn new_actuator_sizing() -> Self {
        Self {
            active: true,
            step: 0,
            steps: actuator_sizing_steps(),
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
                2. Click on the canvas to place the first ground pivot\n\
                3. Click to the RIGHT to place the second pivot\n\
                   (Grid labels show distance in mm \u{2014} aim for ~40mm apart)\n\
                4. Positions can be adjusted later with arrow keys or the Ground Pivots panel",
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

// ── Actuator sizing tutorial steps ───────────────────────────────────────────

fn actuator_sizing_steps() -> Vec<TutorialStep> {
    vec![
        TutorialStep {
            title: "Actuator Sizing Tutorial",
            description: "This tutorial teaches you how to size a linear actuator \
                for a mechanism.\n\n\
                You'll learn to:\n\
                - Set up an output load (force zone)\n\
                - Read required actuator force, speed, and power\n\
                - Check safety margins\n\
                - Compare motion profiles\n\n\
                Click Next to begin.",
            is_complete: |_| true,
        },
        TutorialStep {
            title: "Step 1: Load the Mechanism",
            description: "Load the 'Custom 6-Bar Press' from the Samples dropdown.\n\n\
                This is a 6-bar linkage with a linear actuator and force zone \
                already configured.",
            is_complete: |state| {
                state.current_sample == Some(SampleMechanism::Custom6Bar)
            },
        },
        TutorialStep {
            title: "Step 2: Set Actuator Force to Zero",
            description: "In the sidebar, expand 'Force Elements' and find the \
                Linear Actuator.\n\n\
                Set its force to 0 N.\n\n\
                This tells the solver to COMPUTE the required force instead of \
                applying a fixed value.",
            is_complete: |state| {
                state.blueprint.as_ref().map_or(false, |bp| {
                    bp.forces.iter().any(|f| {
                        if let ForceElement::LinearActuator(a) = f {
                            a.force.abs() < 1.0
                        } else {
                            false
                        }
                    })
                })
            },
        },
        TutorialStep {
            title: "Step 3: Position the Force Zone",
            description: "The force zone represents your output load.\n\n\
                Check the Diagnostics section \u{2014} if it shows '0% overlap', \
                the zone doesn't cover the output link.\n\n\
                Drag the crank angle slider to find where the output link passes \
                through the zone. Adjust the zone position if needed.\n\n\
                The zone should show green 'X N applied' in Diagnostics.",
            // Manual step \u{2014} always completable (force zone overlap is hard
            // to check without evaluating the mechanism at the current angle).
            is_complete: |_| true,
        },
        TutorialStep {
            title: "Step 4: Read the Actuator Force Plot",
            description: "Click the 'Actuator Force' tab in the bottom plot panel.\n\n\
                This shows the required actuator force (N) at each crank angle.\n\n\
                - Red solid line = statics (quasi-static)\n\
                - Blue dashed line = with inertia (includes acceleration)\n\n\
                The peak force is what your actuator must handle.",
            is_complete: |_| true,
        },
        TutorialStep {
            title: "Step 5: Check Speed and Power",
            description: "Click the 'Actuator Speed' tab to see extension rate (mm/s).\n\n\
                Click 'Actuator Power' tab to see required power (W).\n\n\
                These determine your motor/pump sizing:\n\
                - Peak speed \u{2192} actuator speed rating\n\
                - Peak power \u{2192} motor power rating\n\
                - RMS power \u{2192} continuous duty rating",
            is_complete: |_| true,
        },
        TutorialStep {
            title: "Step 6: Enter Rated Force",
            description: "In the Actuator Force plot, find the 'Rated Force' input \
                field.\n\n\
                Enter your actuator's maximum rated force (N).\n\n\
                The plot will show:\n\
                - Green dashed lines at +/- rated force\n\
                - Green dots = under 50% capacity\n\
                - Yellow dots = 50\u{2013}80% capacity\n\
                - Red dots = over 80% (danger zone)\n\n\
                The Health Report shows peak utilization %.",
            is_complete: |state| state.actuator_rated_force > 0.0,
        },
        TutorialStep {
            title: "Step 7: Check the Health Report",
            description: "In the sidebar, expand 'Mechanism Health'.\n\n\
                Look for:\n\
                - Actuator Stroke: total travel (mm)\n\
                - Peak Actuator Force: statics vs with inertia\n\
                - RMS Force and Power\n\
                - Utilization %\n\
                - Angles exceeding 80% capacity\n\n\
                This is your actuator sizing summary.",
            is_complete: |_| true,
        },
        TutorialStep {
            title: "Step 8: Try a Motion Profile",
            description: "In the Driver section, change Motion Profile to \
                'Trapezoidal'.\n\n\
                Adjust the accel/decel fractions.\n\n\
                Watch the Inverse Dynamics plot \u{2014} the green 'Profile Torque' \
                line shows how acceleration phases increase the required effort.\n\n\
                For high-speed mechanisms, this can double the peak force!",
            is_complete: |state| {
                matches!(state.motion_profile, MotionProfile::Trapezoidal { .. })
            },
        },
        TutorialStep {
            title: "Step 9: Export Results",
            description: "File > Export CSV to get force/speed/power data for every \
                angle.\n\n\
                File > Export HTML Report for a professional summary with \
                interactive plots.\n\n\
                The CSV includes columns:\n\
                - actuator_force_N\n\
                - actuator_force_id_N (with inertia)\n\
                - actuator_speed_m_s\n\
                - actuator_power_W",
            is_complete: |_| true,
        },
        TutorialStep {
            title: "Tutorial Complete!",
            description: "You've learned to size a linear actuator:\n\n\
                1. Set actuator force to 0 (compute mode)\n\
                2. Position the output load (force zone)\n\
                3. Read force, speed, power from plots\n\
                4. Check safety margins with rated force\n\
                5. Compare motion profiles\n\
                6. Export results\n\n\
                Tip: Use the Parametric Study to sweep a design parameter and \
                see how it affects actuator force.",
            is_complete: |_| true,
        },
    ]
}

// ── Overlay rendering ────────────────────────────────────────────────────────

/// Draw the tutorial overlay panel when a tutorial is active.
///
/// This is a floating `egui::Window` anchored near the top-right corner.
/// The overlay features a progress bar, styled title, color-coded completion
/// indicator, and accent-colored navigation buttons.
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

    let accent = egui::Color32::from_rgb(80, 160, 255);
    let bar_bg = egui::Color32::from_rgb(60, 60, 70);

    egui::Window::new("Tutorial")
        .collapsible(false)
        .resizable(false)
        .min_width(350.0)
        .default_width(370.0)
        .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-10.0, 50.0))
        .show(ctx, |ui| {
            // ── Progress bar ─────────────────────────────────────────
            let progress = (current as f32 + 1.0) / total as f32;
            let bar_response = ui.allocate_rect(
                egui::Rect::from_min_size(ui.cursor().min, egui::vec2(ui.available_width(), 4.0)),
                egui::Sense::hover(),
            );
            ui.painter()
                .rect_filled(bar_response.rect, 2.0, bar_bg);
            let filled_rect = egui::Rect::from_min_size(
                bar_response.rect.min,
                egui::vec2(bar_response.rect.width() * progress, 4.0),
            );
            ui.painter().rect_filled(filled_rect, 2.0, accent);

            ui.add_space(6.0);

            // ── Step counter ─────────────────────────────────────────
            ui.label(
                egui::RichText::new(format!("Step {} of {}", current + 1, total))
                    .small()
                    .color(egui::Color32::LIGHT_GRAY),
            );

            ui.add_space(4.0);

            // ── Title ────────────────────────────────────────────────
            let title_text = if step_complete {
                egui::RichText::new(format!("\u{2705} {}", title))
                    .strong()
                    .size(16.0)
            } else {
                egui::RichText::new(title).strong().size(16.0)
            };
            ui.label(title_text);

            ui.add_space(6.0);

            // ── Description ──────────────────────────────────────────
            ui.label(description);

            ui.add_space(10.0);

            // ── Completion indicator ─────────────────────────────────
            if step_complete {
                ui.label(
                    egui::RichText::new("Step complete!")
                        .color(egui::Color32::from_rgb(100, 200, 100))
                        .italics(),
                );
                ui.add_space(4.0);
            }

            // ── Navigation buttons ───────────────────────────────────
            ui.horizontal(|ui| {
                // Back button: subtle styling
                if current > 0 {
                    let back_btn = egui::Button::new(
                        egui::RichText::new("Back")
                            .color(egui::Color32::LIGHT_GRAY),
                    )
                    .fill(egui::Color32::TRANSPARENT);
                    if ui.add(back_btn).clicked() {
                        should_go_back = true;
                    }
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
                    .fill(accent)
                } else {
                    egui::Button::new(next_text)
                };

                if ui.add(next_button).clicked() {
                    should_advance = true;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .small_button(
                            egui::RichText::new("Close")
                                .small()
                                .color(egui::Color32::GRAY),
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

    // ── Actuator sizing tutorial tests ───────────────────────────────

    #[test]
    fn new_actuator_sizing_starts_active_at_step_zero() {
        let tutorial = TutorialState::new_actuator_sizing();
        assert!(tutorial.active);
        assert_eq!(tutorial.step, 0);
        // 9 numbered steps + welcome intro + completion summary = 11
        assert_eq!(tutorial.steps.len(), 11);
    }

    #[test]
    fn actuator_sizing_welcome_is_always_complete() {
        let state = AppState::default();
        let tutorial = TutorialState::new_actuator_sizing();
        assert!(
            (tutorial.steps[0].is_complete)(&state),
            "Welcome step should always be completable"
        );
    }

    #[test]
    fn actuator_sizing_step_titles_are_unique() {
        let steps = actuator_sizing_steps();
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
    fn actuator_sizing_advance_through_all_steps() {
        let mut tutorial = TutorialState::new_actuator_sizing();
        let total = tutorial.steps.len();
        for i in 0..total {
            assert!(tutorial.active, "Should still be active at step {}", i);
            assert_eq!(tutorial.step, i);
            tutorial.advance();
        }
        assert!(!tutorial.active, "Should deactivate after the last step");
    }

    #[test]
    fn actuator_sizing_load_mechanism_step_detects_custom_6bar() {
        let mut state = AppState::default();
        let steps = actuator_sizing_steps();
        // Step 1: "Load the Mechanism"
        assert!(
            !(steps[1].is_complete)(&state),
            "Should not be complete without loading the sample"
        );
        state.current_sample = Some(SampleMechanism::Custom6Bar);
        assert!(
            (steps[1].is_complete)(&state),
            "Should be complete with Custom6Bar loaded"
        );
    }

    #[test]
    fn actuator_sizing_rated_force_step_detects_nonzero() {
        let mut state = AppState::default();
        let steps = actuator_sizing_steps();
        // Step 6: "Enter Rated Force"
        assert!(
            !(steps[6].is_complete)(&state),
            "Should not be complete with zero rated force"
        );
        state.actuator_rated_force = 500.0;
        assert!(
            (steps[6].is_complete)(&state),
            "Should be complete with nonzero rated force"
        );
    }

    #[test]
    fn actuator_sizing_motion_profile_step_detects_trapezoidal() {
        let mut state = AppState::default();
        let steps = actuator_sizing_steps();
        // Step 8: "Try a Motion Profile"
        assert!(
            !(steps[8].is_complete)(&state),
            "Should not be complete with default (ConstantSpeed) profile"
        );
        state.motion_profile = MotionProfile::Trapezoidal {
            accel_fraction: 0.25,
            decel_fraction: 0.25,
        };
        assert!(
            (steps[8].is_complete)(&state),
            "Should be complete with Trapezoidal profile"
        );
    }

    #[test]
    fn actuator_sizing_no_titles_overlap_with_fourbar() {
        let fourbar = fourbar_steps();
        let actuator = actuator_sizing_steps();
        let fourbar_titles: Vec<&str> = fourbar.iter().map(|s| s.title).collect();
        let actuator_titles: Vec<&str> = actuator.iter().map(|s| s.title).collect();
        for title in &actuator_titles {
            assert!(
                !fourbar_titles.contains(title),
                "Actuator sizing title '{}' collides with a 4-bar tutorial title",
                title
            );
        }
    }
}
