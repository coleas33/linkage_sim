//! TEMPORARY audit repro tests for BL-010 / BL-017. Not for commit.
//! (BL-011 repro promoted to src/gui/sweep/mod.rs tests.)
//! Run: cargo test --test braindump_repro -- --nocapture

use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::AppState;

#[test]
fn bl010_label_vs_plot_divergence() {
    // BL-010: canvas actuator label (force_render.rs:240-259) vs actuator
    // force plot (plot_panel/actuator.rs:47-54 + mod.rs:648-661).
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);

    // As shipped, la.force = 2225 N (samples/fourbar.rs:671): the canvas label
    // shows the STORED value "2225 N" verbatim (force_render.rs:266) while the
    // plot shows the computed statics curve from sweep.actuator_forces.
    if let Some(bp) = state.blueprint.as_ref() {
        for f in &bp.forces {
            if let linkage_sim_rs::forces::elements::ForceElement::LinearActuator(la) = f {
                println!("as-shipped label shows stored la.force = {} N", la.force);
            }
        }
    }
    state.compute_sweep();
    if let Some(sweep) = state.sweep_data.as_ref() {
        if let Some(forces) = sweep.actuator_forces.as_ref() {
            // Plot value at the pose the mechanism loads at (driver angle 0):
            println!(
                "plot ('Statics' series) at 0 deg shows {:.1} N -- label vs plot: 2225 vs {:.1}",
                forces[0], forces[0]
            );
        }
    }
    let sweep = state.sweep_data.as_ref().expect("sweep computed");
    let forces = sweep.actuator_forces.as_ref().expect("actuator forces");
    let angles = &sweep.angles_deg;
    assert_eq!(angles.len(), forces.len());

    // Replicate the PLOT path: finite filter + Tukey fence (10*IQR), exactly
    // as plot_panel/mod.rs::filter_actuator_outliers does.
    let pairs: Vec<(f64, f64)> = angles
        .iter()
        .zip(forces.iter())
        .filter(|&(_, &f)| f.is_finite())
        .map(|(&a, &f)| (a, f))
        .collect();
    let mut ys: Vec<f64> = pairs.iter().map(|(_, y)| *y).collect();
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ys.len();
    let q1 = ys[n / 4];
    let q3 = ys[3 * n / 4];
    let iqr = (q3 - q1).max(1.0);
    let (lo, hi) = (q1 - 10.0 * iqr, q3 + 10.0 * iqr);
    println!("Tukey fence: [{:.1}, {:.1}] N over {} finite samples", lo, hi, n);

    // Samples the LABEL would show but the PLOT silently drops:
    let mut dropped = Vec::new();
    for &(a, f) in &pairs {
        if f < lo || f > hi {
            dropped.push((a, f));
        }
    }
    println!("{} samples dropped from plot but shown by label:", dropped.len());
    for (a, f) in dropped.iter().take(12) {
        // What the plotted curve shows near this x: nearest surviving sample.
        let nearest_kept = pairs
            .iter()
            .filter(|(_, y)| *y >= lo && *y <= hi)
            .min_by(|(x1, _), (x2, _)| {
                (x1 - a).abs().partial_cmp(&(x2 - a).abs()).unwrap()
            })
            .unwrap();
        println!(
            "  angle {:6.1} deg: label shows {:9.1} N, plot curve nearby shows {:9.1} N (at {:.1} deg)",
            a, f, nearest_kept.1, nearest_kept.0
        );
    }
}

/// Side discovery while reproducing BL-010: following the actuator-sizing
/// tutorial (tutorial.rs:204 "Step 2: Set Actuator Force to Zero") on the
/// ChebyshevLambdaActuator sample makes the very first sweep panic in any
/// debug build: the pass-2 driver lambda fails to collapse at some pose and
/// the data-dependent `debug_assert!` at solver/reactions.rs:666 fires.
#[test]
#[should_panic(expected = "Failed validation")]
fn bl010_side_sizing_mode_sweep_panics_debug_assert() {
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);
    if let Some(bp) = state.blueprint.as_mut() {
        for f in &mut bp.forces {
            if let linkage_sim_rs::forces::elements::ForceElement::LinearActuator(la) = f {
                la.force = 0.0; // sizing mode, per the tutorial
            }
        }
    }
    state.rebuild();
    state.compute_sweep(); // panics at reactions.rs:666 in debug builds
}

#[test]
fn bl010_label_index_lookup_wrong_for_seam_range_sweep() {
    // BL-010 second divergence: the label's nearest-index lookup applies
    // rem_euclid(360) to the driver angle (force_render.rs:247) but
    // sweep.angles_deg for a seam-crossing range sweep (UI allows 200..=365,
    // input_panel.rs:146-154) contains raw values above 360.
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ChebyshevLambdaActuator);
    state.sweep_range_enabled = true;
    state.sweep_angle_min_deg = 200.0;
    state.sweep_angle_max_deg = 365.0;
    state.compute_sweep();
    let sweep = state.sweep_data.as_ref().expect("sweep computed");
    let forces = sweep.actuator_forces.as_ref().expect("actuator forces");
    let angles = &sweep.angles_deg;
    println!(
        "range sweep angles: {:.1}..{:.1} ({} samples)",
        angles.first().unwrap(),
        angles.last().unwrap(),
        angles.len()
    );

    // User scrubs to 365 deg (e.g. clicking the plot at x=365; plot click
    // path sets driver_angle = clicked_x - offset, plot_panel/mod.rs:335-337).
    let driver_angle_rad = 365.0_f64.to_radians() - state.driver_display_offset;
    println!("driver_display_offset = {:.3} deg", state.driver_display_offset.to_degrees());

    // Replicate the label lookup from force_render.rs:247-257 verbatim.
    let current_deg = driver_angle_rad.to_degrees().rem_euclid(360.0);
    let idx = angles
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - current_deg)
                .abs()
                .partial_cmp(&(*b - current_deg).abs())
                .unwrap()
        })
        .map(|(i, _)| i)
        .unwrap();
    let label_force = forces[idx];

    // The sample actually at the scrubbed pose (365 deg):
    let true_idx = angles
        .iter()
        .position(|a| (a - 365.0).abs() < 0.5)
        .expect("365 deg sample exists in the sweep");
    let true_force = forces[true_idx];

    println!(
        "label lookup: current_deg={:.1} -> idx {} (angle {:.1} deg, force {:.1} N)",
        current_deg, idx, angles[idx], label_force
    );
    println!(
        "actual pose sample: idx {} (angle {:.1} deg, force {:.1} N)",
        true_idx, angles[true_idx], true_force
    );
    assert_ne!(idx, true_idx, "label picked the correct sample; divergence refuted");
}
