//! Profile editor for `TrajectoryProfile` + inline `h(t)` preview plot.
//!
//! Used inside the "Profile" collapsing header of the trajectory input panel.
//! Edits the active `TrajectoryProfile` carried inside `SweepMode::Trajectory`
//! and renders a small inline `egui_plot` preview so the shape is visible
//! while the user tunes parameters.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::gui::state::{AppState, MotionProfile};
use crate::gui::sweep::SweepMode;

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    let SweepMode::Trajectory {
        profile, n_samples, ..
    } = &mut state.sweep_mode
    else {
        ui.label("(only visible in Trajectory mode)");
        return;
    };

    // Shape dropdown.
    let mut shape_label = match profile.shape {
        MotionProfile::ConstantSpeed => "ConstantSpeed",
        MotionProfile::Trapezoidal { .. } => "Trapezoidal",
    };
    egui::ComboBox::from_label("Shape")
        .selected_text(shape_label)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut shape_label, "ConstantSpeed", "ConstantSpeed");
            ui.selectable_value(&mut shape_label, "Trapezoidal", "Trapezoidal");
        });
    profile.shape = match shape_label {
        "ConstantSpeed" => MotionProfile::ConstantSpeed,
        "Trapezoidal" => match profile.shape {
            MotionProfile::Trapezoidal { .. } => profile.shape,
            _ => MotionProfile::Trapezoidal {
                accel_fraction: 0.2,
                decel_fraction: 0.2,
            },
        },
        _ => profile.shape,
    };

    // Start / end values.
    ui.horizontal(|ui| {
        ui.label("Start:");
        ui.add(egui::DragValue::new(&mut profile.start_value).speed(0.001));
        ui.label("End:");
        ui.add(egui::DragValue::new(&mut profile.end_value).speed(0.001));
    });

    // Duration. Clamp away from zero to keep `evaluate` well-defined.
    ui.horizontal(|ui| {
        ui.label("Duration (s):");
        ui.add(
            egui::DragValue::new(&mut profile.duration)
                .speed(0.01)
                .range(0.001..=1000.0),
        );
    });

    // Trapezoidal-only fields.
    if let MotionProfile::Trapezoidal {
        accel_fraction,
        decel_fraction,
    } = &mut profile.shape
    {
        ui.horizontal(|ui| {
            ui.label("Accel fraction:");
            ui.add(
                egui::DragValue::new(accel_fraction)
                    .speed(0.01)
                    .range(0.05..=0.45),
            );
            ui.label("Decel fraction:");
            ui.add(
                egui::DragValue::new(decel_fraction)
                    .speed(0.01)
                    .range(0.05..=0.45),
            );
        });
    }

    // Sample count for the trajectory solve.
    ui.horizontal(|ui| {
        ui.label("Samples:");
        ui.add(egui::DragValue::new(n_samples).range(10usize..=2000));
    });

    // Inline h(t) preview. Clone the profile so we don't carry a mutable
    // borrow into the plot closure.
    let profile_clone = profile.clone();
    Plot::new("traj_profile_preview")
        .height(80.0)
        .show_axes([false, false])
        .show(ui, |plot_ui| {
            let n = 100;
            let pts: PlotPoints = (0..=n)
                .map(|i| {
                    let t = profile_clone.duration * (i as f64) / (n as f64);
                    let (h, _, _) = profile_clone.evaluate(t);
                    [t, h]
                })
                .collect();
            plot_ui.line(Line::new("h(t)", pts));
        });
}
