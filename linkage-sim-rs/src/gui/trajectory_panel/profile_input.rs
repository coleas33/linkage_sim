//! Trajectory editor for `SweepMode::Trajectory`: analytic profile or
//! user-defined keyframe table, plus inline `h(t)` preview plot.
//!
//! Used inside the "Profile" collapsing header of the trajectory input panel.
//! Edits the active `Trajectory` carried inside `SweepMode::Trajectory` and
//! renders a small inline `egui_plot` preview so the shape is visible while
//! the user tunes parameters.
//!
//! The top-level dropdown selects between "Profile" (closed-form motion
//! profile: ConstantSpeed / Trapezoidal / SCurve) and "Keyframes" (linear
//! interpolation between user-supplied (t, h) waypoints, with optional CSV
//! import). Switching variants substitutes a sensible default for the new
//! variant; the previous variant's parameters are not preserved.

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

use crate::gui::state::{
    AppState, KeyframeTrajectory, MotionProfile, Trajectory, TrajectoryProfile,
};
use crate::gui::sweep::SweepMode;

pub fn draw(state: &mut AppState, ui: &mut egui::Ui) {
    let SweepMode::Trajectory {
        trajectory,
        n_samples,
        ..
    } = &mut state.sweep_mode
    else {
        ui.label("(only visible in Trajectory mode)");
        return;
    };

    // ── Top-level kind selector: Profile vs Keyframes ────────────────────
    let current_kind = match trajectory {
        Trajectory::Profile(_) => "Profile",
        Trajectory::KeyframeTable(_) => "Keyframes",
    };
    let mut selected_kind = current_kind;
    egui::ComboBox::from_label("Trajectory kind")
        .selected_text(selected_kind)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut selected_kind, "Profile", "Profile");
            ui.selectable_value(&mut selected_kind, "Keyframes", "Keyframes");
        });

    // Switching kinds substitutes a sensible default for the new variant.
    if selected_kind != current_kind {
        *trajectory = match selected_kind {
            "Profile" => Trajectory::Profile(TrajectoryProfile {
                shape: MotionProfile::ConstantSpeed,
                start_value: 0.0,
                end_value: 1.0,
                duration: 1.0,
            }),
            "Keyframes" => Trajectory::KeyframeTable(KeyframeTrajectory::new(vec![
                (0.0, 0.0),
                (1.0, 1.0),
            ])),
            _ => trajectory.clone(),
        };
    }

    // Variant-specific UI.
    match trajectory {
        Trajectory::Profile(p) => draw_profile_editor(p, ui),
        Trajectory::KeyframeTable(kt) => draw_keyframe_editor(kt, ui),
    }

    // Sample count for the trajectory solve (shared across variants).
    ui.horizontal(|ui| {
        ui.label("Samples:");
        ui.add(egui::DragValue::new(n_samples).range(10usize..=2000));
    });

    // Inline h(t) preview. Clone so we don't carry a mutable borrow into
    // the plot closure.
    let traj_clone = trajectory.clone();
    let preview_dur = traj_clone.duration().max(0.001);
    Plot::new("traj_profile_preview")
        .height(80.0)
        .show_axes([false, false])
        .show(ui, |plot_ui| {
            let n = 100;
            let pts: PlotPoints = (0..=n)
                .map(|i| {
                    let t = preview_dur * (i as f64) / (n as f64);
                    let (h, _, _) = traj_clone.evaluate(t);
                    [t, h]
                })
                .collect();
            plot_ui.line(Line::new("h(t)", pts));
            // Overlay waypoint markers when in keyframe mode for visual
            // confirmation that the table matches what's expected.
            if let Trajectory::KeyframeTable(kt) = &traj_clone {
                let marker_pts: PlotPoints =
                    kt.waypoints.iter().map(|(t, h)| [*t, *h]).collect();
                plot_ui.points(
                    egui_plot::Points::new("waypoints", marker_pts)
                        .radius(4.0)
                        .color(egui::Color32::from_rgb(255, 200, 80)),
                );
            }
        });
}

/// Analytic motion profile editor: shape dropdown + start/end/duration UI
/// + (Trapezoidal-only) accel/decel fractions.
fn draw_profile_editor(profile: &mut TrajectoryProfile, ui: &mut egui::Ui) {
    let mut shape_label = match profile.shape {
        MotionProfile::ConstantSpeed => "ConstantSpeed",
        MotionProfile::Trapezoidal { .. } => "Trapezoidal",
        MotionProfile::SCurve { .. } => "SCurve",
    };
    egui::ComboBox::from_label("Shape")
        .selected_text(shape_label)
        .show_ui(ui, |ui| {
            ui.selectable_value(&mut shape_label, "ConstantSpeed", "ConstantSpeed");
            ui.selectable_value(&mut shape_label, "Trapezoidal", "Trapezoidal");
            ui.selectable_value(&mut shape_label, "SCurve", "SCurve");
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
        "SCurve" => match profile.shape {
            MotionProfile::SCurve { .. } => profile.shape,
            _ => MotionProfile::SCurve { jerk_fraction: 0.2 },
        },
        _ => profile.shape,
    };

    ui.horizontal(|ui| {
        ui.label("Start:");
        ui.add(egui::DragValue::new(&mut profile.start_value).speed(0.001));
        ui.label("End:");
        ui.add(egui::DragValue::new(&mut profile.end_value).speed(0.001));
    });

    ui.horizontal(|ui| {
        ui.label("Duration (s):");
        ui.add(
            egui::DragValue::new(&mut profile.duration)
                .speed(0.01)
                .range(0.001..=1000.0),
        );
    });

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
}

/// Keyframe waypoint table editor: row-per-waypoint with inline drag-edit
/// of `t` and `h`, remove buttons, an Add button, an explicit Sort-by-t
/// button, and an "Import CSV..." button (native only).
fn draw_keyframe_editor(kt: &mut KeyframeTrajectory, ui: &mut egui::Ui) {
    ui.label("Waypoints (t, h):");

    let mut to_remove: Option<usize> = None;
    for (i, (t, h)) in kt.waypoints.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.label(format!("[{}]", i));
            ui.label("t:");
            ui.add(
                egui::DragValue::new(t)
                    .speed(0.01)
                    .range(0.0..=1000.0),
            );
            ui.label("h:");
            ui.add(egui::DragValue::new(h).speed(0.001));
            // Use a plain ASCII "X" rather than the unicode multiplication
            // sign so the button renders cleanly across platforms / fonts.
            if ui
                .button("X")
                .on_hover_text("Remove waypoint")
                .clicked()
            {
                to_remove = Some(i);
            }
        });
    }
    if let Some(i) = to_remove {
        // Keep at least 2 waypoints so the trajectory has non-zero duration
        // and `evaluate` falls into the bracketing-pair branch.
        if kt.waypoints.len() > 2 {
            kt.waypoints.remove(i);
        }
    }

    ui.horizontal(|ui| {
        if ui.button("+ Add waypoint").clicked() {
            // Append at the end with sensible defaults (extend the last
            // segment by 0.5 s; carry the last h forward).
            let last_t = kt.waypoints.last().map(|w| w.0).unwrap_or(0.0);
            let last_h = kt.waypoints.last().map(|w| w.1).unwrap_or(0.0);
            kt.waypoints.push((last_t + 0.5, last_h));
        }
        if ui
            .button("Sort by t")
            .on_hover_text("Re-sort waypoints by ascending t")
            .clicked()
        {
            kt.waypoints
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Native-only CSV import. WASM lacks a synchronous file picker; the
        // button is hidden in the browser build for now.
        #[cfg(feature = "native")]
        {
            if ui
                .button("Import CSV...")
                .on_hover_text(
                    "Load a 2-column CSV (t_seconds, target_value) as keyframes.",
                )
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .add_filter("CSV", &["csv"])
                    .pick_file()
                {
                    match parse_keyframes_csv(&path) {
                        Ok(new_kt) => *kt = new_kt,
                        Err(e) => log::warn!("CSV import failed: {}", e),
                    }
                }
            }
        }
    });
}

/// Parse a 2-column CSV `t_seconds, target_value` into a `KeyframeTrajectory`.
///
/// Behaviour:
///   - Blank lines and lines starting with `#` are skipped.
///   - The first row is treated as a header if its first cell fails to parse
///     as a float (e.g. `t_seconds`); otherwise it's a data row.
///   - Subsequent rows must have two numeric cells; non-numeric content
///     produces an error with the offending line number (1-based).
///   - At least one valid waypoint is required.
///
/// Waypoints are sorted by `t` ascending via `KeyframeTrajectory::new`.
#[allow(dead_code)] // used from the native-feature button path
fn parse_keyframes_csv(path: &std::path::Path) -> Result<KeyframeTrajectory, String> {
    use std::io::BufRead;
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let reader = std::io::BufReader::new(file);
    let mut waypoints = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut parts = trimmed.split(',');
        let t_str = parts
            .next()
            .ok_or_else(|| format!("line {}: missing t", line_no + 1))?;
        let h_str = parts
            .next()
            .ok_or_else(|| format!("line {}: missing h", line_no + 1))?;
        let t: f64 = match t_str.trim().parse() {
            Ok(v) => v,
            Err(_) if line_no == 0 => continue, // header row
            Err(e) => return Err(format!("line {}: t parse error: {}", line_no + 1, e)),
        };
        let h: f64 = h_str.trim().parse().map_err(
            |e: std::num::ParseFloatError| {
                format!("line {}: h parse error: {}", line_no + 1, e)
            },
        )?;
        waypoints.push((t, h));
    }
    if waypoints.is_empty() {
        return Err("No valid waypoints found".to_string());
    }
    Ok(KeyframeTrajectory::new(waypoints))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_keyframes_csv_simple() {
        use std::io::Write;
        let tmp = std::env::temp_dir().join("test_kt_simple.csv");
        let mut f = std::fs::File::create(&tmp).unwrap();
        writeln!(f, "t_seconds,target_value").unwrap();
        writeln!(f, "0.0, 0.0").unwrap();
        writeln!(f, "0.5, 1.5").unwrap();
        writeln!(f, "1.0, 2.0").unwrap();
        drop(f);
        let kt = parse_keyframes_csv(&tmp).unwrap();
        assert_eq!(kt.waypoints.len(), 3);
        assert!((kt.duration() - 1.0).abs() < 1e-12);
        let _ = std::fs::remove_file(tmp);
    }
}
