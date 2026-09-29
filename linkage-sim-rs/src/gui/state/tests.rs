    use super::*;
    use super::blueprint_ops::{joint_body_ids, seed_q_by_body_id};
    use crate::gui::test_support::sorted_link_ids;
    use crate::io::JointJson;

    #[test]
    fn default_state_has_empty_mechanism() {
        let state = AppState::default();
        // Default state starts with an empty mechanism (ground only).
        assert!(state.has_mechanism());
        assert!(state.current_sample.is_none());
        assert!(state.blueprint.is_some());
    }

    #[test]
    fn load_sample_solves_initial() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.has_mechanism());
        assert!(
            state.solver_status.converged,
            "Initial solve did not converge, residual = {}",
            state.solver_status.residual_norm
        );
        assert!(!state.q.is_empty());
    }

    #[test]
    fn solve_at_angle_updates_state() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let q_initial = state.q.clone();

        state.solve_at_angle(0.5);

        assert!(
            state.solver_status.converged,
            "solve_at_angle(0.5) did not converge, residual = {}",
            state.solver_status.residual_norm
        );
        // q should have changed from the initial position
        assert_ne!(state.q, q_initial, "q did not change after solve_at_angle");
    }

    #[test]
    fn step_animation_advances_angle() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.playing = true;
        state.animation_speed_deg_per_sec = 180.0;

        let angle_before = state.driver_angle;
        let still_playing = state.step_animation(1.0 / 60.0);
        assert!(still_playing);
        assert!(state.driver_angle > angle_before);
    }

    #[test]
    fn driver_display_offset_zero_for_sample_builders() {
        // Sample-built bodies place attachment points at local (0,0) and
        // (len, 0), so the A→B direction lies on +X and the offset is 0.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(
            state.driver_display_offset.abs() < 1e-9,
            "sample FourBar should have zero display offset, got {}",
            state.driver_display_offset
        );
    }

    #[test]
    fn driver_display_offset_picks_up_rotated_crank() {
        // Move the crank's far attachment point so the A→B vector sits
        // at a known angle (+30°) in local coords. The display offset
        // should track to match. This simulates what a DXF-imported
        // crank looks like when the sketch was drawn at an angle.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // FourBar's crank has attachment points A (grounded) and B.
        // Place B at (len * cos(30°), len * sin(30°)) keeping the
        // length at 2.0 (the sample default).
        let new_bx = 2.0_f64 * 30f64.to_radians().cos();
        let new_by = 2.0_f64 * 30f64.to_radians().sin();
        if let Some(bp) = state.blueprint.as_mut() {
            if let Some(crank) = bp.bodies.get_mut("crank") {
                crank.attachment_points.insert(
                    "B".to_string(),
                    [new_bx, new_by],
                );
            }
        }
        state.rebuild();
        assert!(
            (state.driver_display_offset - 30f64.to_radians()).abs() < 1e-6,
            "offset should be ~30° for a crank rotated 30° in local frame, got {:.6} rad",
            state.driver_display_offset
        );
    }

    #[test]
    fn convert_actuator_to_linear_driver_switches_mode() {
        // ParallelogramActuator has a LinearActuator force element
        // and a revolute driver. After conversion, the mechanism
        // should have a LinearDriver, no revolute driver, and the
        // GUI dispatch should report Linear mode.
        use crate::forces::elements::ForceElement;
        use crate::gui::state::DriverKind;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        assert!(matches!(state.driver_kind, DriverKind::Revolute { .. }));

        // Find the LinearActuator force index.
        let act_index = state
            .mechanism
            .as_ref()
            .unwrap()
            .forces()
            .iter()
            .position(|f| matches!(f, ForceElement::LinearActuator(_)))
            .expect("sample should have a linear actuator");

        state.convert_actuator_to_linear_driver(act_index);

        assert!(
            matches!(state.driver_kind, DriverKind::Linear { .. }),
            "post-convert driver_kind should be Linear"
        );
        let mech = state.mechanism.as_ref().unwrap();
        assert_eq!(mech.n_drivers(), 0, "revolute driver should be removed");
        assert_eq!(
            mech.n_linear_drivers(),
            1,
            "exactly one linear driver should be added"
        );
        // length_0 should be the actuator's pre-conversion length, so
        // driver_stroke is set to that value (no pose jump).
        assert!(
            state.driver_stroke() > 0.0,
            "driver_stroke should be initialised to the current actuator length"
        );
    }

    #[test]
    fn step_animation_advances_at_slow_speed() {
        // At 0.5 deg/s (the slider's new floor), 600 frames at 1/60 s should
        // advance the crank by about 5 deg. Regression test for the user
        // report that "anything less than 15 deg/s doesn't move".
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.playing = true;
        state.animation_speed_deg_per_sec = 0.5;

        let angle_before = state.driver_angle;
        for _ in 0..600 {
            state.step_animation(1.0 / 60.0);
        }
        let advanced_deg = (state.driver_angle - angle_before).to_degrees();
        assert!(
            (advanced_deg - 5.0).abs() < 0.5,
            "at 0.5 deg/s after 10 s expected ~5 deg of advance, got {:.3}",
            advanced_deg
        );
    }

    #[test]
    fn set_constant_speed_driver_rejects_tiny_omega() {
        // Typing 0 in the RPM DragValue used to freeze the animation: the
        // driver closure captured omega=0, so f(t) = theta_0 forever and
        // solve_at_angle divided by zero. The UI entry point now clamps
        // to the minimum magnitude.
        use crate::core::driver::MIN_DRIVER_OMEGA_ABS;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.set_constant_speed_driver(0.0, 0.0);
        assert!(
            state.driver_omega().abs() >= MIN_DRIVER_OMEGA_ABS - 1e-12,
            "expected driver_omega clamped to >= {}, got {}",
            MIN_DRIVER_OMEGA_ABS,
            state.driver_omega()
        );

        // Negative sign must be preserved.
        state.set_constant_speed_driver(-1e-6, 0.0);
        assert!(
            state.driver_omega() < 0.0,
            "negative-sign omega should remain negative after clamp"
        );

        // And after the clamp, solve_at_angle must actually advance the
        // driver angle — no stuck animation.
        state.solve_at_angle(std::f64::consts::FRAC_PI_6);
        assert!(state.solver_status.converged);
        assert!(
            (state.driver_angle - std::f64::consts::FRAC_PI_6).abs() < 1e-6,
            "driver_angle should have reached the requested angle after clamp"
        );
    }

    #[test]
    fn step_animation_noop_when_paused() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.playing = false;

        let angle_before = state.driver_angle;
        assert!(!state.step_animation(1.0 / 60.0));
        assert_eq!(state.driver_angle, angle_before);
    }

    #[test]
    fn reassign_driver_changes_joint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));

        state.reassign_driver("J4");
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));
        assert!(state.solver_status.converged);
        assert!(!state.playing);
    }

    // ── Undo / Redo integration tests ──────────────────────────────────

    #[test]
    fn take_snapshot_returns_some_for_default_state() {
        let state = AppState::default();
        // Default state has an empty mechanism, so snapshots should work.
        assert!(state.take_snapshot().is_some());
    }

    #[test]
    fn take_snapshot_returns_some_with_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.take_snapshot().is_some());
    }

    #[test]
    fn snapshot_roundtrip_preserves_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let snapshot = state.take_snapshot().unwrap();
        let body_count = state.mechanism.as_ref().unwrap().bodies().len();
        let joint_count = state.mechanism.as_ref().unwrap().joints().len();

        // Clobber state then restore
        state.load_sample(SampleMechanism::SliderCrank);
        state.restore_snapshot(&snapshot);

        let mech = state.mechanism.as_ref().unwrap();
        assert_eq!(mech.bodies().len(), body_count);
        assert_eq!(mech.joints().len(), joint_count);
        assert!(state.solver_status.converged);
    }

    #[test]
    fn snapshot_roundtrip_preserves_driver_fields() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.set_driver_omega(5.0);
        state.set_driver_theta_0(1.0);
        state.driver_angle = 2.0;
        state.driver_joint_id = Some("J1".to_string());

        let snapshot = state.take_snapshot().unwrap();
        state.restore_snapshot(&snapshot);

        assert_eq!(state.driver_omega(), 5.0);
        assert_eq!(state.driver_theta_0(), 1.0);
        assert_eq!(state.driver_angle, 2.0);
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));
    }

    #[test]
    fn load_sample_clears_undo_history() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.push_undo();
        assert!(state.can_undo());

        state.load_sample(SampleMechanism::SliderCrank);
        assert!(!state.can_undo());
        assert!(!state.can_redo());
    }

    #[test]
    fn push_undo_works_on_default_empty_mechanism() {
        let mut state = AppState::default();
        state.push_undo();
        // Default state has an empty mechanism, so undo should work.
        assert!(state.can_undo());
    }

    #[test]
    fn undo_noop_without_mechanism() {
        let mut state = AppState::default();
        state.undo();
        assert!(!state.can_undo());
    }

    #[test]
    fn undo_after_driver_reassignment_restores_previous_driver() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));

        state.reassign_driver("J4");
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));
        assert!(state.can_undo());

        state.undo();
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));
        assert!(state.solver_status.converged);
    }

    #[test]
    fn redo_after_undo_restores_forward_state() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.reassign_driver("J4");
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));

        state.undo();
        assert_eq!(state.driver_joint_id, Some("J1".to_string()));
        assert!(state.can_redo());

        state.redo();
        assert_eq!(state.driver_joint_id, Some("J4".to_string()));
        assert!(!state.can_redo());
    }

    #[test]
    fn new_push_clears_redo_stack() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.reassign_driver("J4");
        state.undo();
        assert!(state.can_redo());

        // A new undoable action should clear redo
        state.push_undo();
        assert!(!state.can_redo());
    }

    #[test]
    fn world_to_screen_roundtrip() {
        let view = ViewTransform::default();
        let wx = 0.019;
        let wy = 0.015;

        let [sx, sy] = view.world_to_screen(wx, wy);
        let [wx2, wy2] = view.screen_to_world(sx, sy);

        assert!(
            (wx - wx2).abs() < 1e-6,
            "X roundtrip error: {} vs {}",
            wx,
            wx2
        );
        assert!(
            (wy - wy2).abs() < 1e-6,
            "Y roundtrip error: {} vs {}",
            wy,
            wy2
        );
    }

    // ── Sweep tests ──────────────────────────────────────────────────────

    #[test]
    fn compute_sweep_produces_data_for_fourbar() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // load_sample calls compute_sweep internally
        let sweep = state.sweep_data.as_ref().expect("sweep_data should be Some after load_sample");

        // FourBar is a Grashof crank-rocker -- full 361 points (0-360 inclusive).
        assert_eq!(
            sweep.angles_deg.len(),
            361,
            "Expected 361 sweep points, got {}",
            sweep.angles_deg.len()
        );

        // Body angles should exist for all 3 moving bodies.
        assert_eq!(sweep.body_angles.len(), 3);
        for (body_id, angles) in &sweep.body_angles {
            assert_eq!(
                angles.len(),
                361,
                "Body '{}' has {} angle entries, expected 361",
                body_id,
                angles.len()
            );
        }
    }

    #[test]
    fn compute_sweep_coupler_traces_non_empty() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let sweep = state.sweep_data.as_ref().unwrap();

        assert!(
            !sweep.coupler_traces.is_empty(),
            "coupler_traces should not be empty"
        );
        for (key, trace) in &sweep.coupler_traces {
            assert!(
                !trace.is_empty(),
                "trace for '{}' should not be empty",
                key
            );
        }
    }

    #[test]
    fn mass_change_does_not_alter_coupler_traces() {
        // Reproduces user report: load parallelogram, change mass on coupler,
        // coupler trace should NOT change (mass doesn't affect kinematics).
        // Test at multiple driver angles since the parallelogram is a change-point
        // mechanism and the solver might be sensitive to initial guesses.
        let test_angles: &[f64] = &[0.0, 0.5, 1.0, std::f64::consts::PI, 3.0, 5.0];

        for &angle in test_angles {
            let mut state = AppState::default();
            state.load_sample(SampleMechanism::Parallelogram);

            // Move the driver to a non-zero angle (simulates user dragging the slider).
            if angle != 0.0 {
                state.solve_at_angle(angle);
            }

            // Recompute sweep at this angle.
            state.compute_sweep();
            let sweep_before = state.sweep_data.as_ref().expect("sweep should exist");
            let traces_before: std::collections::HashMap<String, Vec<[f64; 2]>> =
                sweep_before.coupler_traces.clone();
            assert!(!traces_before.is_empty(), "should have coupler traces");

            // Change mass on the coupler (exactly what the user does).
            state.set_body_mass("coupler", 5.0);
            // Trigger sweep recomputation (in the GUI this happens after debounce).
            state.compute_sweep();

            let sweep_after = state.sweep_data.as_ref()
                .expect("sweep should exist after mass change");

            // Coupler trace positions must be identical — mass is irrelevant to kinematics.
            assert_eq!(
                traces_before.len(),
                sweep_after.coupler_traces.len(),
                "angle={:.2}: number of coupler traces changed",
                angle,
            );
            for (key, trace_before) in &traces_before {
                let trace_after = sweep_after
                    .coupler_traces
                    .get(key)
                    .unwrap_or_else(|| panic!("missing trace key '{}' after mass change", key));
                assert_eq!(
                    trace_before.len(),
                    trace_after.len(),
                    "angle={:.2}: trace '{}' has different length",
                    angle,
                    key
                );
                let mut max_delta = 0.0_f64;
                let mut worst_step = 0;
                let mut divergences = Vec::new();
                for (i, (pt_before, pt_after)) in
                    trace_before.iter().zip(trace_after.iter()).enumerate()
                {
                    let dx = (pt_before[0] - pt_after[0]).abs();
                    let dy = (pt_before[1] - pt_after[1]).abs();
                    let d = dx.max(dy);
                    if d > max_delta {
                        max_delta = d;
                        worst_step = i;
                    }
                    if d > 1e-6 {
                        divergences.push(format!(
                            "  step {}: before=({:.6}, {:.6}), after=({:.6}, {:.6}), delta=({:.2e}, {:.2e})",
                            i, pt_before[0], pt_before[1], pt_after[0], pt_after[1], dx, dy
                        ));
                    }
                }
                if !divergences.is_empty() {
                    panic!(
                        "angle={:.2}: trace '{}' diverged significantly ({} steps > 1e-6, max_delta={:.2e} at step {})\n{}",
                        angle, key, divergences.len(), max_delta, worst_step,
                        divergences.join("\n")
                    );
                }
            }
        }
    }

    #[test]
    fn compute_sweep_fourbar_has_transmission_angle() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let sweep = state.sweep_data.as_ref().unwrap();

        let ta = sweep
            .transmission_angles
            .as_ref()
            .expect("FourBar should have transmission angles");
        assert_eq!(ta.len(), 361);
        // All transmission angles should be in (0, 180).
        for &angle in ta {
            assert!(
                angle > 0.0 && angle < 180.0,
                "transmission angle {} out of range",
                angle
            );
        }
    }

    #[test]
    fn compute_sweep_partial_for_non_crank() {
        // Double-rocker cannot complete full 360 rotation. The sweep should
        // still return 361 angle entries (one per degree) so data vectors
        // stay aligned across all channels, but some angles will have NaN
        // values where the position solver couldn't find a configuration.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::DoubleRocker);
        let sweep = state.sweep_data.as_ref().unwrap();

        // Should have full 361 points (NaN-padded for unreachable angles).
        assert_eq!(
            sweep.angles_deg.len(),
            361,
            "Sweep should have 361 angle entries (NaN-padded for unreachable angles)"
        );

        // At least some body angles should be NaN (unreachable config), and
        // at least some should be finite (reachable config), proving it's
        // a non-Grashof mechanism handled gracefully.
        let (_body_id, angles) = sweep.body_angles.iter().next().expect("has body angles");
        let n_nan = angles.iter().filter(|a| a.is_nan()).count();
        let n_finite = angles.iter().filter(|a| a.is_finite()).count();
        assert!(
            n_nan > 0,
            "Double-rocker should have unreachable angles (NaN), got {} NaN out of {}",
            n_nan, angles.len()
        );
        assert!(
            n_finite > 0,
            "Double-rocker should have some reachable angles, got {} finite out of {}",
            n_finite, angles.len()
        );
    }

    // ── Blueprint + rebuild tests ────────────────────────────────────────

    #[test]
    fn load_sample_populates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.blueprint.is_some(), "blueprint should be Some after load_sample");
        assert!(state.mechanism.is_some());
        assert!(state.solver_status.converged);
    }

    #[test]
    fn blueprint_bodies_match_mechanism_bodies() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let bp = state.blueprint.as_ref().unwrap();
        let mech = state.mechanism.as_ref().unwrap();
        assert_eq!(
            bp.bodies.len(),
            mech.bodies().len(),
            "Blueprint and mechanism should have the same number of bodies"
        );
        for body_id in mech.bodies().keys() {
            assert!(
                bp.bodies.contains_key(body_id),
                "Blueprint should contain body '{}'",
                body_id
            );
        }
    }

    #[test]
    fn rebuild_produces_valid_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Manually trigger rebuild
        state.rebuild();

        assert!(state.mechanism.is_some());
        assert!(
            state.solver_status.converged,
            "rebuild() should produce a converged mechanism, residual = {}",
            state.solver_status.residual_norm
        );
    }

    #[test]
    fn set_body_mass_updates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.set_body_mass("crank", 1.5);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert!(
            (crank.mass - 1.5).abs() < f64::EPSILON,
            "Blueprint mass should be 1.5, got {}",
            crank.mass
        );
        // Mechanism should still be valid after rebuild
        assert!(state.mechanism.is_some());
    }

    #[test]
    fn set_body_izz_updates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        state.set_body_izz("crank", 0.05);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert!(
            (crank.izz_cg - 0.05).abs() < f64::EPSILON,
            "Blueprint izz_cg should be 0.05, got {}",
            crank.izz_cg
        );
    }

    #[test]
    fn move_attachment_point_updates_blueprint_and_rebuilds() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Read original position of an attachment point
        let bp = state.blueprint.as_ref().unwrap();
        let orig = bp.bodies.get("crank").unwrap().attachment_points.get("B").unwrap();
        let orig_x = orig[0];
        let orig_y = orig[1];

        // Move it slightly
        let new_x = orig_x + 0.001;
        let new_y = orig_y + 0.001;
        state.move_attachment_point("crank", "B", new_x, new_y);

        // Check blueprint was updated
        let bp = state.blueprint.as_ref().unwrap();
        let moved = bp.bodies.get("crank").unwrap().attachment_points.get("B").unwrap();
        assert!(
            (moved[0] - new_x).abs() < f64::EPSILON,
            "Blueprint point X should be {}, got {}",
            new_x,
            moved[0]
        );
        assert!(
            (moved[1] - new_y).abs() < f64::EPSILON,
            "Blueprint point Y should be {}, got {}",
            new_y,
            moved[1]
        );

        // Mechanism should still be present (rebuild happened)
        assert!(state.mechanism.is_some());
    }

    #[test]
    fn load_from_file_populates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Save to file, then reload
        let path = std::env::temp_dir().join("linkage_test_blueprint.json");
        state.save_to_file(&path).expect("save_to_file failed");

        let mut state2 = AppState::default();
        state2.load_from_file(&path).expect("load_from_file failed");

        assert!(
            state2.blueprint.is_some(),
            "blueprint should be Some after load_from_file"
        );
        assert!(state2.mechanism.is_some());

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn edit_nonexistent_body_is_noop() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Edit a body that doesn't exist -- should not crash
        state.set_body_mass("nonexistent", 999.0);

        // Mechanism should still be valid (rebuild ran but nothing changed)
        assert!(state.mechanism.is_some());
        assert!(state.solver_status.converged);
    }

    #[test]
    fn edit_nonexistent_point_is_noop() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Edit a point that doesn't exist on the body -- should not crash
        state.move_attachment_point("crank", "nonexistent", 0.0, 0.0);

        // The rebuild still happens but the blueprint wasn't changed
        assert!(state.mechanism.is_some());
        assert!(state.solver_status.converged);
    }

    #[test]
    fn rebuild_all_samples_have_blueprints() {
        for sample in SampleMechanism::all() {
            let mut state = AppState::default();
            state.load_sample(*sample);
            assert!(
                state.blueprint.is_some(),
                "Sample {:?} should have a blueprint after load_sample",
                sample
            );
            assert!(
                state.mechanism.is_some(),
                "Sample {:?} should have a mechanism after load_sample",
                sample
            );
        }
    }

    // ── Create / delete tests ───────────────────────────────────────────

    #[test]
    fn add_ground_pivot_updates_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let n_before = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();
        state.add_ground_pivot("O5", 5.0, 0.0);
        let n_after = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();
        assert_eq!(n_after, n_before + 1);
        // The new point should exist with the right coordinates.
        let pt = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .get("O5")
            .unwrap();
        assert!((pt[0] - 5.0).abs() < f64::EPSILON);
        assert!((pt[1] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn add_ground_pivot_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_before = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();

        state.add_ground_pivot("P99", 1.0, 2.0);
        assert!(state.can_undo());

        state.undo();
        let n_after = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();
        assert_eq!(n_after, n_before);
    }

    #[test]
    fn update_ground_pivot_position_rebuilds_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Find an existing ground pivot name.
        let pivot_name = {
            let ground = state
                .blueprint
                .as_ref()
                .unwrap()
                .bodies
                .get("ground")
                .unwrap();
            ground
                .attachment_points
                .keys()
                .next()
                .unwrap()
                .clone()
        };

        // Move it.
        state.update_ground_pivot_position(&pivot_name, 1.0, 1.0);

        // Verify blueprint updated.
        let pt = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .get(&pivot_name)
            .unwrap();
        assert!((pt[0] - 1.0).abs() < f64::EPSILON);
        assert!((pt[1] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn update_ground_pivot_position_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Find an existing ground pivot name and record its position.
        let (pivot_name, original_pos) = {
            let ground = state
                .blueprint
                .as_ref()
                .unwrap()
                .bodies
                .get("ground")
                .unwrap();
            let (name, pos) = ground
                .attachment_points
                .iter()
                .next()
                .unwrap();
            (name.clone(), *pos)
        };

        state.update_ground_pivot_position(&pivot_name, 99.0, 99.0);
        assert!(state.can_undo());

        state.undo();
        let pt = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .get(&pivot_name)
            .unwrap();
        assert!((pt[0] - original_pos[0]).abs() < f64::EPSILON);
        assert!((pt[1] - original_pos[1]).abs() < f64::EPSILON);
    }

    #[test]
    fn update_ground_pivot_position_noop_for_missing_point() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let n_before = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();

        // Try to move a nonexistent pivot.
        state.update_ground_pivot_position("NONEXISTENT", 1.0, 1.0);

        // Count should not change (no new point inserted).
        let n_after = state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .get("ground")
            .unwrap()
            .attachment_points
            .len();
        assert_eq!(n_before, n_after);
    }

    #[test]
    fn add_body_with_points_creates_new_body_in_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_bodies_before = state.blueprint.as_ref().unwrap().bodies.len();

        let points = vec![
            ("X".to_string(), [0.0, 0.0]),
            ("Y".to_string(), [0.05, 0.0]),
        ];
        let body_id = state.add_body_with_points(&points);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies.len(), n_bodies_before + 1);
        assert!(bp.bodies.contains_key(&body_id));
        let body = bp.bodies.get(&body_id).unwrap();
        assert_eq!(body.attachment_points.len(), 2);
        assert!(body.attachment_points.contains_key("X"));
        assert!(body.attachment_points.contains_key("Y"));
    }

    #[test]
    fn remove_body_cascades_to_joints() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let joints_before = state.blueprint.as_ref().unwrap().joints.len();

        state.remove_body("coupler");

        let bp = state.blueprint.as_ref().unwrap();
        assert!(!bp.bodies.contains_key("coupler"));
        // Joints connected to coupler should be removed.
        assert!(
            bp.joints.len() < joints_before,
            "Expected fewer joints after removing coupler, got {} (was {})",
            bp.joints.len(),
            joints_before
        );
        // No remaining joint should reference "coupler".
        for (_id, joint) in &bp.joints {
            let (bi, bj) = joint_body_ids(joint);
            assert_ne!(bi, "coupler", "Joint still references removed body");
            assert_ne!(bj, "coupler", "Joint still references removed body");
        }
    }

    #[test]
    fn remove_body_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_bodies = state.blueprint.as_ref().unwrap().bodies.len();
        let n_joints = state.blueprint.as_ref().unwrap().joints.len();

        state.remove_body("crank");
        assert!(state.can_undo());

        state.undo();
        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies.len(), n_bodies);
        assert_eq!(bp.joints.len(), n_joints);
    }

    #[test]
    fn add_revolute_joint_creates_joint_in_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_joints_before = state.blueprint.as_ref().unwrap().joints.len();

        // Add a new body first so we have a valid target.
        let points = vec![
            ("E1".to_string(), [0.0, 0.0]),
            ("E2".to_string(), [0.01, 0.0]),
        ];
        let extra_id = state.add_body_with_points(&points);

        // Add joint between ground and the new body.
        state.add_revolute_joint("ground", "O2", &extra_id, "E1");

        let bp = state.blueprint.as_ref().unwrap();
        // +1 from the new joint (add_body_with_points doesn't add joints).
        assert_eq!(bp.joints.len(), n_joints_before + 1);
    }

    #[test]
    fn remove_joint_removes_from_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_joints_before = state.blueprint.as_ref().unwrap().joints.len();

        // Get the first joint ID.
        let joint_id = state
            .blueprint
            .as_ref()
            .unwrap()
            .joints
            .keys()
            .next()
            .unwrap()
            .clone();
        state.remove_joint(&joint_id);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.joints.len(), n_joints_before - 1);
        assert!(!bp.joints.contains_key(&joint_id));
    }

    #[test]
    fn remove_joint_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let n_joints = state.blueprint.as_ref().unwrap().joints.len();

        let joint_id = state
            .blueprint
            .as_ref()
            .unwrap()
            .joints
            .keys()
            .next()
            .unwrap()
            .clone();
        state.remove_joint(&joint_id);
        assert!(state.can_undo());

        state.undo();
        assert_eq!(state.blueprint.as_ref().unwrap().joints.len(), n_joints);
    }

    #[test]
    fn next_body_id_generates_unique_ids() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let id1 = state.next_body_id();
        let points = vec![
            ("A".to_string(), [0.0, 0.0]),
            ("B".to_string(), [0.01, 0.0]),
        ];
        state.add_body_with_points(&points);

        let id2 = state.next_body_id();
        assert_ne!(id1, id2, "Second ID should differ from first");
        assert!(
            !state.blueprint.as_ref().unwrap().bodies.contains_key(&id2),
            "Generated ID should not already exist in blueprint"
        );
    }

    #[test]
    fn next_ground_pivot_name_generates_unique_names() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let name1 = state.next_ground_pivot_name();
        state.add_ground_pivot(&name1, 1.0, 0.0);

        let name2 = state.next_ground_pivot_name();
        assert_ne!(name1, name2);
    }

    // ── Validation tests ────────────────────────────────────────────────

    #[test]
    fn validation_no_warnings_for_valid_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // FourBar has driver, correct DOF, no disconnected bodies.
        state.compute_validation();

        assert!(
            state.validation_warnings.dof_warning.is_none(),
            "FourBar should have DOF=0, got: {:?}",
            state.validation_warnings.dof_warning
        );
        assert!(
            !state.validation_warnings.missing_driver,
            "FourBar should have a driver"
        );
        assert!(
            state.validation_warnings.disconnected_bodies.is_empty(),
            "FourBar should have no disconnected bodies"
        );
    }

    #[test]
    fn validation_disconnected_body_detected() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Add a body with no joints.
        let points = vec![
            ("F1".to_string(), [0.0, 0.5]),
            ("F2".to_string(), [0.01, 0.5]),
        ];
        let floating_id = state.add_body_with_points(&points);
        state.compute_validation();

        assert!(
            state.validation_warnings.disconnected_bodies.contains(&floating_id),
            "Should detect '{}' as disconnected",
            floating_id,
        );
    }

    #[test]
    fn validation_dof_warning_after_removing_joint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Remove a joint -- DOF should no longer be 0.
        let joint_id = state
            .blueprint
            .as_ref()
            .unwrap()
            .joints
            .keys()
            .next()
            .unwrap()
            .clone();
        state.remove_joint(&joint_id);

        // Validation is computed during rebuild.
        assert!(
            state.validation_warnings.dof_warning.is_some(),
            "Should have DOF warning after removing a joint"
        );
    }

    #[test]
    fn remove_body_cascades_to_drivers() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // The driver references ground + crank. Removing crank should
        // also remove the driver.
        let drivers_before = state.blueprint.as_ref().unwrap().drivers.len();
        assert!(drivers_before > 0, "FourBar should have a driver");

        state.remove_body("crank");

        let bp = state.blueprint.as_ref().unwrap();
        assert!(
            bp.drivers.is_empty(),
            "Drivers referencing removed body should be cascaded"
        );
    }

    #[test]
    fn remove_body_cascades_to_force_elements() {
        // ParallelogramActuator has a LinearActuator force element
        // attached to `crank`. Removing `crank` must also strip the
        // actuator from the blueprint — otherwise the rebuild tries to
        // resolve the actuator's attachment point on a missing body and
        // freezes the GUI.
        use crate::forces::elements::ForceElement;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);

        let forces_before = state.blueprint.as_ref().unwrap().forces.len();
        let had_actuator_on_crank = state
            .blueprint
            .as_ref()
            .unwrap()
            .forces
            .iter()
            .any(|f| {
                matches!(f, ForceElement::LinearActuator(_))
                    && f.attached_body_ids().contains(&"crank")
            });
        assert!(had_actuator_on_crank, "sample should start with a crank-attached actuator");
        assert!(forces_before > 0);

        state.remove_body("crank");

        let bp = state.blueprint.as_ref().unwrap();
        for f in &bp.forces {
            assert!(
                !f.attached_body_ids().contains(&"crank"),
                "force element still references deleted body: {:?}",
                f
            );
        }
    }

    // ── Grid snap tests ──────────────────────────────────────────────────

    #[test]
    fn snap_to_grid_rounds_to_nearest() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 1.0,
        };
        assert_eq!(grid.snap(2.3), 2.0);
        assert_eq!(grid.snap(2.7), 3.0);
        assert_eq!(grid.snap(-0.4), 0.0);
        assert_eq!(grid.snap(0.5), 1.0); // half-away-from-zero: 0.5 rounds to 1
        assert_eq!(grid.snap(1.5), 2.0); // half-away-from-zero: 1.5 rounds to 2
    }

    #[test]
    fn snap_disabled_passes_through() {
        let grid = GridSettings {
            snap_enabled: false,
            show_grid: true,
            spacing_m: 1.0,
        };
        assert_eq!(grid.snap(2.3), 2.3);
        assert_eq!(grid.snap(-7.777), -7.777);
    }

    #[test]
    fn snap_zero_spacing_passes_through() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 0.0,
        };
        assert_eq!(grid.snap(2.3), 2.3);
    }

    #[test]
    fn snap_point_snaps_both_axes() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 0.5,
        };
        let (sx, sy) = grid.snap_point(1.3, -0.2);
        assert!((sx - 1.5).abs() < 1e-12);
        assert!((sy - 0.0).abs() < 1e-12);
    }

    #[test]
    fn snap_fine_spacing() {
        let grid = GridSettings {
            snap_enabled: true,
            show_grid: true,
            spacing_m: 0.005,
        };
        // 0.0123 is closest to 0.010 (2.46 grid units, rounds to 2)
        // Actually 0.0123 / 0.005 = 2.46, rounds to 2 -> 0.010
        let result = grid.snap(0.0123);
        assert!((result - 0.010).abs() < 1e-12, "got {}", result);
    }

    #[test]
    fn auto_grid_spacing_fourbar_small_scale() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // FourBar has ground=0.038m. Expect a fine grid spacing.
        assert!(
            state.grid.spacing_m >= 0.001 && state.grid.spacing_m <= 0.01,
            "FourBar grid spacing should be 1-10 mm, got {} m",
            state.grid.spacing_m
        );
    }

    #[test]
    fn auto_grid_spacing_crank_rocker_large_scale() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        // CrankRocker has d=4m ground. Expect a coarser grid.
        assert!(
            state.grid.spacing_m >= 0.1 && state.grid.spacing_m <= 2.0,
            "CrankRocker grid spacing should be 0.1-2.0 m, got {} m",
            state.grid.spacing_m
        );
    }

    // ── Load case tests ───────────────────────────────────────────────

    #[test]
    fn default_load_case_created_on_sample_load() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        assert_eq!(state.load_cases.cases.len(), 1);
        assert_eq!(state.load_cases.cases[0].name, "Default");
        assert_eq!(state.load_cases.active_index, 0);
    }

    #[test]
    fn add_load_case_copies_current() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        state.add_load_case();
        assert_eq!(state.load_cases.cases.len(), 2);
        assert_eq!(state.load_cases.cases[1].name, "Case 2");
        // New case should have same driver params as current
        assert_eq!(
            state.load_cases.cases[1].omega,
            state.driver_omega(),
        );
    }

    #[test]
    fn remove_load_case_prevents_last_removal() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        assert_eq!(state.load_cases.cases.len(), 1);
        // Should not be able to remove the last case
        state.remove_active_load_case();
        assert_eq!(state.load_cases.cases.len(), 1);
    }

    #[test]
    fn remove_load_case_works_with_multiple() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        state.add_load_case();
        assert_eq!(state.load_cases.cases.len(), 2);
        state.remove_active_load_case();
        assert_eq!(state.load_cases.cases.len(), 1);
    }

    #[test]
    fn switch_load_case_changes_omega() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);

        let original_omega = state.driver_omega();
        let new_omega = original_omega * 2.0;

        // Add a case with different omega
        state.load_cases.cases.push(LoadCase {
            name: "Fast".to_string(),
            driver_joint_id: state.driver_joint_id.clone().unwrap(),
            omega: new_omega,
            theta_0: 0.0,
        });

        state.apply_load_case(1);
        assert_eq!(state.driver_omega(), new_omega);
        assert_eq!(state.load_cases.active_index, 1);
    }

    #[test]
    fn switch_load_case_changes_driver_joint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);

        // Add a case targeting a different joint (J4 is the rocker pivot)
        state.load_cases.cases.push(LoadCase {
            name: "Rocker Drive".to_string(),
            driver_joint_id: "J4".to_string(),
            omega: 2.0 * PI,
            theta_0: 0.0,
        });

        state.apply_load_case(1);
        // The pending_driver_reassignment should be set for the next frame
        assert_eq!(
            state.pending_driver_reassignment,
            Some("J4".to_string()),
        );
    }

    #[test]
    fn load_case_manager_new_default() {
        let mgr = LoadCaseManager::new_default("J1", 2.0 * PI, 0.5);
        assert_eq!(mgr.cases.len(), 1);
        assert_eq!(mgr.cases[0].name, "Default");
        assert_eq!(mgr.cases[0].driver_joint_id, "J1");
        assert_eq!(mgr.cases[0].omega, 2.0 * PI);
        assert_eq!(mgr.cases[0].theta_0, 0.5);
        assert_eq!(mgr.active_index, 0);
    }

    #[test]
    fn load_case_manager_add_case_returns_index() {
        let mut mgr = LoadCaseManager::new_default("J1", PI, 0.0);
        let idx = mgr.add_case("J2", 2.0 * PI, 1.0);
        assert_eq!(idx, 1);
        assert_eq!(mgr.cases.len(), 2);
        assert_eq!(mgr.cases[1].driver_joint_id, "J2");
    }

    #[test]
    fn load_case_manager_remove_adjusts_active_index() {
        let mut mgr = LoadCaseManager::new_default("J1", PI, 0.0);
        mgr.add_case("J2", 2.0 * PI, 0.0);
        mgr.add_case("J3", 3.0 * PI, 0.0);
        mgr.active_index = 2; // point to last

        mgr.remove_case(2);
        // active_index should be clamped to the last valid index
        assert_eq!(mgr.active_index, 1);
        assert_eq!(mgr.cases.len(), 2);
    }

    #[test]
    fn load_case_manager_remove_single_case_is_noop() {
        let mut mgr = LoadCaseManager::new_default("J1", PI, 0.0);
        assert!(!mgr.remove_case(0));
        assert_eq!(mgr.cases.len(), 1);
    }

    #[test]
    fn sync_active_load_case_updates_params() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        assert_eq!(state.load_cases.cases[0].omega, 2.0 * PI);

        state.set_driver_omega(10.0);
        state.sync_active_load_case();
        assert_eq!(state.load_cases.cases[0].omega, 10.0);
    }

    #[test]
    fn load_cases_persist_through_save_load() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_load_case();
        state.load_cases.cases[1].name = "High Speed".to_string();
        state.load_cases.cases[1].omega = 10.0 * PI;
        assert_eq!(state.load_cases.cases.len(), 2);

        let path = std::env::temp_dir().join("linkage_test_load_cases.json");
        state.save_to_file(&path).expect("save_to_file failed");

        let mut state2 = AppState::default();
        state2.load_from_file(&path).expect("load_from_file failed");

        assert_eq!(state2.load_cases.cases.len(), 2);
        assert_eq!(state2.load_cases.cases[0].name, "Default");
        assert_eq!(state2.load_cases.cases[1].name, "High Speed");
        assert!(
            (state2.load_cases.cases[1].omega - 10.0 * PI).abs() < 1e-10,
            "omega should be preserved, got {}",
            state2.load_cases.cases[1].omega,
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn reassign_driver_resets_load_cases() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_load_case();
        assert_eq!(state.load_cases.cases.len(), 2);

        state.reassign_driver("J4");
        // After reassignment, load cases should be reset to a single default
        assert_eq!(state.load_cases.cases.len(), 1);
        assert_eq!(state.load_cases.cases[0].name, "Default");
        assert_eq!(state.load_cases.cases[0].driver_joint_id, "J4");
    }

    #[test]
    fn load_sample_no_driver_has_empty_load_cases() {
        // Build a state, check that samples without a driver don't crash
        let state = AppState::default();
        assert!(state.load_cases.cases.is_empty());
    }

    #[test]
    fn apply_load_case_out_of_bounds_is_noop() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::CrankRocker);
        let omega_before = state.driver_omega();
        state.apply_load_case(999);
        assert_eq!(state.driver_omega(), omega_before);
    }

    // ── next_attachment_point_name tests ─────────────────────────────────

    #[test]
    fn next_attachment_point_name_skips_existing() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // FourBar sample uses body names "crank", "coupler", "rocker" with points A/B
        let name = state.next_attachment_point_name("crank");
        assert_eq!(name, "C");
    }

    #[test]
    fn next_attachment_point_name_empty_body() {
        let mut state = AppState::default();
        let bp = state.blueprint.as_mut().unwrap();
        bp.bodies.insert("empty".to_string(), BodyJson {
            attachment_points: HashMap::new(),
            mass: 1.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.01,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        });
        let name = state.next_attachment_point_name("empty");
        assert_eq!(name, "A");
    }

    #[test]
    fn next_attachment_point_name_overflow_past_z() {
        let mut state = AppState::default();
        let bp = state.blueprint.as_mut().unwrap();
        let mut pts = HashMap::new();
        for c in b'A'..=b'Z' {
            pts.insert(String::from(c as char), [0.0, 0.0]);
        }
        bp.bodies.insert("full".to_string(), BodyJson {
            attachment_points: pts,
            mass: 1.0,
            cg_local: [0.0, 0.0],
            izz_cg: 0.01,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        });
        let name = state.next_attachment_point_name("full");
        assert_eq!(name, "AA");
    }

    // ── world_to_body_local tests ─────────────────────────────────────────

    #[test]
    fn world_to_body_local_identity_pose() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let [lx, ly] = state.world_to_body_local("ground", 0.05, 0.03);
        assert!((lx - 0.05).abs() < 1e-10);
        assert!((ly - 0.03).abs() < 1e-10);
    }

    // ── Raw helper tests ─────────────────────────────────────────────────

    #[test]
    fn add_ground_pivot_raw_mutates_blueprint() {
        let mut state = AppState::default();
        state.add_ground_pivot_raw("P1", 0.1, 0.2);
        let bp = state.blueprint.as_ref().unwrap();
        let ground = bp.bodies.get("ground").unwrap();
        assert!(ground.attachment_points.contains_key("P1"));
        let pt = ground.attachment_points.get("P1").unwrap();
        assert!((pt[0] - 0.1).abs() < 1e-15);
        assert!((pt[1] - 0.2).abs() < 1e-15);
        assert!(!state.can_undo());
    }

    #[test]
    fn add_revolute_joint_raw_mutates_blueprint() {
        let mut state = AppState::default();
        state.add_ground_pivot_raw("O", 0.0, 0.0);
        let bp = state.blueprint.as_mut().unwrap();
        let mut pts = HashMap::new();
        pts.insert("A".to_string(), [0.0, 0.0]);
        pts.insert("B".to_string(), [0.1, 0.0]);
        bp.bodies.insert("link".to_string(), BodyJson {
            attachment_points: pts,
            mass: 1.0,
            cg_local: [0.05, 0.0],
            izz_cg: 0.01,
            mount_points: HashMap::new(),
            coupler_points: HashMap::new(),
            point_masses: Vec::new(),
            label: None,
            geometry: None,
        });
        state.add_revolute_joint_raw("ground", "O", "link", "A");
        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.joints.len(), 1);
        assert!(!state.can_undo());
    }

    #[test]
    fn add_body_with_points_raw_first_point_is_local_origin() {
        let mut state = AppState::default();
        let points = vec![
            ("A".to_string(), [0.05, 0.03]),
            ("B".to_string(), [0.15, 0.03]),
            ("C".to_string(), [0.10, 0.08]),
        ];
        let new_body_id = state.next_body_id();
        state.add_body_with_points_raw(&new_body_id, &points);
        let bp = state.blueprint.as_ref().unwrap();
        let body = bp.bodies.values()
            .find(|b| b.attachment_points.contains_key("A") && b.attachment_points.contains_key("C"))
            .expect("body with A, B, C");
        let a = body.attachment_points.get("A").unwrap();
        assert!((a[0]).abs() < 1e-15);
        assert!((a[1]).abs() < 1e-15);
        let b = body.attachment_points.get("B").unwrap();
        assert!((b[0] - 0.10).abs() < 1e-15);
        assert!((b[1] - 0.00).abs() < 1e-15);
        let c = body.attachment_points.get("C").unwrap();
        assert!((c[0] - 0.05).abs() < 1e-15);
        assert!((c[1] - 0.05).abs() < 1e-15);
        let expected_cg_x = (0.0 + 0.10 + 0.05) / 3.0;
        let expected_cg_y = (0.0 + 0.0 + 0.05) / 3.0;
        assert!((body.cg_local[0] - expected_cg_x).abs() < 1e-10);
        assert!((body.cg_local[1] - expected_cg_y).abs() < 1e-10);
        assert!(!state.can_undo());
    }

    #[test]
    fn add_attachment_point_local_raw_adds_to_existing_body() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.add_attachment_point_local_raw("crank", "C", 0.005, 0.003);
        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert!(crank.attachment_points.contains_key("C"));
        let c = crank.attachment_points.get("C").unwrap();
        assert!((c[0] - 0.005).abs() < 1e-15);
        assert!((c[1] - 0.003).abs() < 1e-15);
    }

    // ── add_attachment_point_to_body / remove_attachment_point tests ──────────

    #[test]
    fn add_attachment_point_to_body_converts_world_to_local() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Ground is at (0,0,0), so world = local for ground
        state.add_attachment_point_to_body("ground", "PX", 0.05, 0.02);
        let bp = state.blueprint.as_ref().unwrap();
        let ground = bp.bodies.get("ground").unwrap();
        let px = ground.attachment_points.get("PX").unwrap();
        assert!((px[0] - 0.05).abs() < 1e-10);
        assert!((px[1] - 0.02).abs() < 1e-10);
        assert!(state.can_undo());
    }

    #[test]
    fn remove_attachment_point_cascades_to_joints() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Crank has points A and B. J1 connects ground:O2 to crank:A.
        // Removing crank:A should also remove J1.
        let bp = state.blueprint.as_ref().unwrap();
        let joint_count_before = bp.joints.len();
        state.remove_attachment_point("crank", "A");
        let bp = state.blueprint.as_ref().unwrap();
        assert!(!bp.bodies.get("crank").unwrap().attachment_points.contains_key("A"));
        assert!(bp.joints.len() < joint_count_before);
        assert!(state.can_undo());
    }

    // ── Integration tests: multi-pivot body editing workflows ─────────────────

    #[test]
    fn create_ternary_body_via_add_body_with_points() {
        let mut state = AppState::default();
        let points = vec![
            ("A".to_string(), [0.0, 0.0]),
            ("B".to_string(), [0.1, 0.0]),
            ("C".to_string(), [0.05, 0.08]),
        ];
        state.add_body_with_points(&points);
        let bp = state.blueprint.as_ref().unwrap();
        // Should have ground + the new body
        assert_eq!(bp.bodies.len(), 2);
        let body = bp.bodies.iter()
            .find(|(id, _)| *id != "ground")
            .unwrap().1;
        assert_eq!(body.attachment_points.len(), 3);
        assert!(state.can_undo());
    }

    #[test]
    fn add_pivot_then_joint_creates_ternary_mechanism() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Add a third pivot to the coupler body
        // (FourBar coupler has attachment points B and C)
        state.add_attachment_point_to_body("coupler", "P", 0.02, 0.01);
        let bp = state.blueprint.as_ref().unwrap();
        let coupler = bp.bodies.get("coupler").unwrap();
        assert_eq!(coupler.attachment_points.len(), 3); // B, C, P
    }

    #[test]
    fn remove_attachment_point_with_min_two_points_survives() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Crank has A and B. Removing B should leave just A.
        state.remove_attachment_point("crank", "B");
        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        assert_eq!(crank.attachment_points.len(), 1);
        assert!(crank.attachment_points.contains_key("A"));
    }

    #[test]
    fn compound_draw_link_is_single_undo_step() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp_before = state.blueprint.as_ref().unwrap().clone();
        let bodies_before = bp_before.bodies.len();
        let joints_before = bp_before.joints.len();

        // Simulate a compound Draw Link operation
        state.push_undo();
        state.add_ground_pivot_raw("PX", 0.1, 0.1);
        let new_body_id = state.next_body_id();
        let points = vec![
            ("A".to_string(), [0.1, 0.1]),
            ("B".to_string(), [0.2, 0.1]),
        ];
        state.add_body_with_points_raw(&new_body_id, &points);
        state.add_revolute_joint_raw("ground", "PX", &new_body_id, "A");
        state.rebuild();

        // Verify operation added bodies/joints
        let bp_after = state.blueprint.as_ref().unwrap();
        assert!(bp_after.bodies.len() > bodies_before);
        assert!(bp_after.joints.len() > joints_before);

        // One undo should restore the entire previous state
        assert!(state.can_undo());
        state.undo();
        let bp_undone = state.blueprint.as_ref().unwrap();
        assert_eq!(bp_undone.bodies.len(), bodies_before);
        assert_eq!(bp_undone.joints.len(), joints_before);
    }

    #[test]
    fn dirty_flag_set_on_push_undo() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(!state.dirty, "freshly loaded sample should not be dirty");
        state.push_undo();
        assert!(state.dirty, "push_undo should set dirty flag");
    }

    #[test]
    fn show_shortcuts_defaults_false() {
        let state = AppState::default();
        assert!(!state.show_shortcuts);
    }

    #[test]
    fn show_dimensions_defaults_true() {
        let state = AppState::default();
        assert!(state.show_dimensions);
    }

    #[test]
    fn show_labels_defaults_true() {
        let state = AppState::default();
        assert!(state.show_labels);
    }

    #[cfg(feature = "native")]
    #[test]
    fn autosave_path_with_no_save_uses_temp_dir() {
        let state = AppState::default();
        let path = state.autosave_path();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(p.to_string_lossy().contains("linkage_simulator_autosave"));
    }

    #[cfg(feature = "native")]
    #[test]
    fn autosave_path_with_save_path_creates_sibling() {
        let mut state = AppState::default();
        state.last_save_path = Some(std::path::PathBuf::from("/tmp/my_mechanism.json"));
        let path = state.autosave_path();
        assert!(path.is_some());
        let p = path.unwrap();
        assert!(
            p.to_string_lossy().contains(".my_mechanism.autosave.json"),
            "Expected sibling autosave path, got {:?}",
            p
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn save_to_file_clears_dirty_flag() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.push_undo(); // sets dirty=true
        assert!(state.dirty);

        let tmp = std::env::temp_dir().join("linkage_test_save.json");
        state.save_to_file(&tmp).expect("save should succeed");
        assert!(!state.dirty, "save_to_file should clear dirty flag");
        assert_eq!(state.last_save_path.as_deref(), Some(tmp.as_path()));

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[cfg(feature = "native")]
    #[test]
    fn load_from_file_clears_dirty_flag() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Save first
        let tmp = std::env::temp_dir().join("linkage_test_load.json");
        state.save_to_file(&tmp).expect("save should succeed");

        // Dirty it, then reload
        state.push_undo();
        assert!(state.dirty);

        state.load_from_file(&tmp).expect("load should succeed");
        assert!(!state.dirty, "load_from_file should clear dirty flag");
        assert_eq!(state.last_save_path.as_deref(), Some(tmp.as_path()));

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn add_point_mass_modifies_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Find a non-ground body
        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        assert!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.is_empty(),
            "should start with no point masses"
        );

        state.add_point_mass(&body_id, 0.5, [0.01, 0.0]);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies[&body_id].point_masses.len(), 1);
        assert!((bp.bodies[&body_id].point_masses[0].mass - 0.5).abs() < 1e-10);
    }

    #[test]
    fn remove_point_mass_modifies_blueprint() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        let weight_id = state.add_point_mass(&body_id, 0.5, [0.01, 0.0]).expect("add should succeed");
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.len(),
            1
        );

        assert!(state.remove_point_mass_by_id(&body_id, &weight_id));
        assert!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.is_empty(),
            "point mass should be removed"
        );
    }

    #[test]
    fn add_point_mass_is_undoable() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        state.add_point_mass(&body_id, 0.5, [0.01, 0.0]);
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.len(),
            1
        );

        state.undo();
        assert!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.is_empty(),
            "undo should remove the point mass"
        );
    }

    #[test]
    fn point_mass_affects_built_mechanism_mass() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        let mass_before = state.mechanism.as_ref().unwrap().bodies()[&body_id].mass;

        state.add_point_mass(&body_id, 2.0, [0.01, 0.0]);

        let mass_after = state.mechanism.as_ref().unwrap().bodies()[&body_id].mass;
        assert!(
            (mass_after - mass_before - 2.0).abs() < 1e-10,
            "built mechanism mass should increase by 2.0 kg, was {} now {}",
            mass_before,
            mass_after
        );
    }

    #[cfg(feature = "native")]
    #[test]
    fn point_mass_persists_through_save_load() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        state.add_point_mass(&body_id, 1.5, [0.02, -0.01]);

        // Save
        let tmp = std::env::temp_dir().join("linkage_test_pm.json");
        state.save_to_file(&tmp).expect("save should succeed");

        // Reload into fresh state
        let mut state2 = AppState::default();
        state2.load_from_file(&tmp).expect("load should succeed");

        let bp2 = state2.blueprint.as_ref().unwrap();
        assert_eq!(
            bp2.bodies[&body_id].point_masses.len(),
            1,
            "point mass should persist through save/load"
        );
        assert!((bp2.bodies[&body_id].point_masses[0].mass - 1.5).abs() < 1e-10);

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    // ── BL-023: save / autosave / share URL must not double-count point masses ──

    /// Mass, CG and Izz of one body (either the built composite or the blueprint base).
    #[derive(Debug, Clone, Copy)]
    struct MassProps {
        mass: f64,
        cg: [f64; 2],
        izz: f64,
    }

    /// Composite (point-mass-inclusive) mass properties of every body in the
    /// built mechanism, keyed by body id.
    fn built_mass_props(state: &AppState) -> std::collections::BTreeMap<String, MassProps> {
        state
            .mechanism
            .as_ref()
            .expect("mechanism should be built")
            .bodies()
            .iter()
            .map(|(id, b)| {
                (
                    id.clone(),
                    MassProps { mass: b.mass, cg: [b.cg_local.x, b.cg_local.y], izz: b.izz_cg },
                )
            })
            .collect()
    }

    /// Base (point-mass-exclusive) mass properties of every body in the blueprint.
    fn blueprint_mass_props(state: &AppState) -> std::collections::BTreeMap<String, MassProps> {
        state
            .blueprint
            .as_ref()
            .expect("blueprint should exist")
            .bodies
            .iter()
            .map(|(id, b)| (id.clone(), MassProps { mass: b.mass, cg: b.cg_local, izz: b.izz_cg }))
            .collect()
    }

    fn assert_mass_props_match(
        what: &str,
        expected: &std::collections::BTreeMap<String, MassProps>,
        actual: &std::collections::BTreeMap<String, MassProps>,
    ) {
        const TOL: f64 = 1e-12;
        assert_eq!(
            expected.keys().collect::<Vec<_>>(),
            actual.keys().collect::<Vec<_>>(),
            "{what}: body id sets differ"
        );
        for (id, e) in expected {
            let a = &actual[id];
            assert!((e.mass - a.mass).abs() < TOL, "{what}: body '{id}' mass {} -> {}", e.mass, a.mass);
            assert!((e.cg[0] - a.cg[0]).abs() < TOL, "{what}: body '{id}' cg.x {} -> {}", e.cg[0], a.cg[0]);
            assert!((e.cg[1] - a.cg[1]).abs() < TOL, "{what}: body '{id}' cg.y {} -> {}", e.cg[1], a.cg[1]);
            assert!((e.izz - a.izz).abs() < TOL, "{what}: body '{id}' izz {} -> {}", e.izz, a.izz);
        }
    }

    /// Four-bar with off-CG point masses: two on one body (so accumulation
    /// order matters) and one on another. Returns the state and the id of
    /// the body carrying two point masses.
    fn four_bar_with_point_masses() -> (AppState, String) {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let ids = sorted_link_ids(&state);

        let base = blueprint_mass_props(&state);
        state.add_point_mass(&ids[0], 50.0, [0.03, 0.02]);
        state.add_point_mass(&ids[0], 2.5, [-0.01, 0.04]);
        state.add_point_mass(&ids[1], 7.0, [0.05, -0.02]);

        // Precondition (guards against a vacuous test): the point masses are
        // applied exactly once to the live mechanism, and the blueprint base
        // is untouched.
        let built = built_mass_props(&state);
        assert!(
            (built[&ids[0]].mass - (base[&ids[0]].mass + 52.5)).abs() < 1e-12,
            "setup: composite mass should be base + 52.5"
        );
        assert!((built[&ids[1]].mass - (base[&ids[1]].mass + 7.0)).abs() < 1e-12);
        assert_mass_props_match("setup: blueprint base", &base, &blueprint_mass_props(&state));

        (state, ids[0].clone())
    }

    /// Assert that `dst` (loaded from `src`'s serialization) reproduces the
    /// same built composite properties, the same blueprint base properties
    /// and the same point-mass lists as `src`.
    fn assert_round_trip_preserves_mass(what: &str, src: &AppState, dst: &AppState) {
        assert_mass_props_match(&format!("{what}: built"), &built_mass_props(src), &built_mass_props(dst));
        assert_mass_props_match(
            &format!("{what}: blueprint base"),
            &blueprint_mass_props(src),
            &blueprint_mass_props(dst),
        );
        let src_bp = src.blueprint.as_ref().unwrap();
        let dst_bp = dst.blueprint.as_ref().unwrap();
        for (id, sb) in &src_bp.bodies {
            let db = &dst_bp.bodies[id];
            assert_eq!(sb.point_masses.len(), db.point_masses.len(), "{what}: body '{id}' point-mass count");
            for (s, d) in sb.point_masses.iter().zip(&db.point_masses) {
                assert_eq!(s.id, d.id, "{what}: body '{id}' point-mass id");
                assert_eq!(s.label, d.label, "{what}: body '{id}' point-mass label");
                assert!((s.mass - d.mass).abs() < 1e-12, "{what}: body '{id}' point-mass mass");
                assert!((s.local_pos[0] - d.local_pos[0]).abs() < 1e-12, "{what}: body '{id}' point-mass x");
                assert!((s.local_pos[1] - d.local_pos[1]).abs() < 1e-12, "{what}: body '{id}' point-mass y");
            }
        }
    }

    #[test]
    fn point_masses_not_double_counted_by_serialize_load_bl023() {
        let (src, heavy_body) = four_bar_with_point_masses();

        let json = src.serialize_to_json_string().expect("serialize should succeed");
        let mut dst = AppState::default();
        dst.load_from_json_str(&json).expect("load should succeed");
        assert_round_trip_preserves_mass("save/load", &src, &dst);

        // A second generation (autosave of a just-loaded file) must be stable too.
        let json2 = dst.serialize_to_json_string().expect("re-serialize should succeed");
        let mut dst2 = AppState::default();
        dst2.load_from_json_str(&json2).expect("re-load should succeed");
        assert_round_trip_preserves_mass("second save/load", &src, &dst2);

        // Point masses removed after a reload still leave the true base mass.
        let mut dst3 = dst2;
        let base_mass = blueprint_mass_props(&src)[&heavy_body].mass;
        assert!(dst3.remove_point_mass_by_id(&heavy_body, "W2"));
        assert!(dst3.remove_point_mass_by_id(&heavy_body, "W1"));
        let m = dst3.mechanism.as_ref().unwrap().bodies()[&heavy_body].mass;
        assert!((m - base_mass).abs() < 1e-12, "base mass after removing point masses: {m} vs {base_mass}");
    }

    #[test]
    fn point_masses_not_double_counted_by_share_url_bl023() {
        let (src, _) = four_bar_with_point_masses();

        let url = src.generate_share_url().expect("generate_share_url failed");
        let encoded = url.split("?m=").nth(1).expect("share URL missing ?m=");
        let json = super::file_io::decode_mechanism_from_url(encoded).expect("decode failed");

        let mut dst = AppState::default();
        dst.load_from_json_str(&json).expect("load should succeed");
        assert_round_trip_preserves_mass("share URL", &src, &dst);
    }

    #[cfg(feature = "native")]
    #[test]
    fn point_masses_not_double_counted_by_file_save_load_bl023() {
        let (mut src, _) = four_bar_with_point_masses();

        let tmp = std::env::temp_dir().join("linkage_test_pm_bl023.json");
        src.save_to_file(&tmp).expect("save should succeed");
        let mut dst = AppState::default();
        let loaded = dst.load_from_file(&tmp);
        let _ = std::fs::remove_file(&tmp);
        loaded.expect("load should succeed");

        assert_round_trip_preserves_mass("file save/load", &src, &dst);
    }

    // ── Payload weights: point-mass identity (ids, labels, loader validation) ──

    /// Ids of the point masses on `body_id`, in list order.
    fn point_mass_ids(state: &AppState, body_id: &str) -> Vec<String> {
        state.blueprint.as_ref().unwrap().bodies[body_id]
            .point_masses
            .iter()
            .map(|pm| pm.id.clone())
            .collect()
    }

    #[test]
    fn saved_file_carries_point_mass_ids() {
        let (src, heavy) = four_bar_with_point_masses();
        let links = sorted_link_ids(&src);
        assert_eq!(point_mass_ids(&src, &heavy), ["W1", "W2"], "add_point_mass assigns W<n>");
        assert_eq!(point_mass_ids(&src, &links[1]), ["W3"]);

        let v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        assert_eq!(v["bodies"][heavy.as_str()]["point_masses"][1]["id"], "W2");
        assert_eq!(v["bodies"][links[1].as_str()]["point_masses"][0]["id"], "W3");
    }

    #[test]
    fn old_file_without_point_mass_ids_gets_ids_on_load() {
        let (src, heavy) = four_bar_with_point_masses();
        let links = sorted_link_ids(&src);
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        for body in v["bodies"].as_object_mut().unwrap().values_mut() {
            if let Some(pms) = body.get_mut("point_masses").and_then(|p| p.as_array_mut()) {
                for pm in pms {
                    let pm = pm.as_object_mut().unwrap();
                    pm.remove("id");
                    pm.remove("label");
                }
            }
        }

        let mut dst = AppState::default();
        dst.load_from_json_str(&v.to_string()).expect("old file should load");
        // Bodies sorted by id, masses in list order.
        assert_eq!(point_mass_ids(&dst, &heavy), ["W1", "W2"]);
        assert_eq!(point_mass_ids(&dst, &links[1]), ["W3"]);
        // Masses and positions intact and applied exactly once.
        assert_round_trip_preserves_mass("old file", &src, &dst);
        assert!(dst.error_log.is_empty(), "a valid file reports nothing: {:?}", dst.error_log);
    }

    #[test]
    fn point_mass_ids_and_labels_round_trip_through_save_and_share() {
        let (mut src, heavy) = four_bar_with_point_masses();
        src.blueprint.as_mut().unwrap().bodies.get_mut(&heavy).unwrap().point_masses[1].label =
            Some("Robot torso".to_string());

        let mut dst = AppState::default();
        dst.load_from_json_str(&src.serialize_to_json_string().unwrap()).unwrap();
        assert_round_trip_preserves_mass("save/load with label", &src, &dst);

        let url = src.generate_share_url().expect("generate_share_url failed");
        let encoded = url.split("?m=").nth(1).expect("share URL missing ?m=");
        let json = super::file_io::decode_mechanism_from_url(encoded).expect("decode failed");
        let mut dst2 = AppState::default();
        dst2.load_from_json_str(&json).unwrap();
        assert_round_trip_preserves_mass("share URL with label", &src, &dst2);
    }

    #[test]
    fn load_skips_and_reports_invalid_point_masses() {
        let (src, body, other) = four_bar_with_one_point_mass();
        let base = blueprint_mass_props(&src);
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        v["bodies"][GROUND_ID]["point_masses"] =
            serde_json::json!([{"id": "on_ground", "mass": 5.0, "local_pos": [0.0, 0.0]}]);
        v["bodies"][other.as_str()]["point_masses"] = serde_json::json!([
            {"id": "neg", "mass": -3.0, "local_pos": [0.01, 0.0]},
            {"mass": 0.0, "local_pos": [0.02, 0.0]}
        ]);

        let mut dst = AppState::default();
        dst.load_from_json_str(&v.to_string()).expect("a file with bad weights still loads");

        // Skipped: ground stays massless and `other` keeps its base mass; the
        // valid 2 kg weight on `body` is still applied.
        let built = built_mass_props(&dst);
        assert_eq!(built[GROUND_ID].mass, 0.0);
        assert!((built[&other].mass - base[&other].mass).abs() < 1e-12, "{}", built[&other].mass);
        assert!((built[&body].mass - (base[&body].mass + 2.0)).abs() < 1e-12);

        // Reported once each, naming the weight and body.
        assert_eq!(dst.error_log.len(), 3, "{:?}", dst.error_log);
        assert!(dst.error_log.iter().any(|w| w.contains("'on_ground'")), "{:?}", dst.error_log);
        assert!(dst.error_log.iter().any(|w| w.contains("'neg'")), "{:?}", dst.error_log);
        assert!(dst.show_error_panel, "load warnings must be visible");

        // Kept in the blueprint (a save writes them back unchanged) and
        // addressable: the blank id got the smallest unused W<n>.
        assert_eq!(point_mass_ids(&dst, GROUND_ID), ["on_ground"]);
        assert_eq!(point_mass_ids(&dst, &other), ["neg", "W2"]);
    }

    // ── Weight editing: id-addressed, one undo step per edit (BL-025) ──────
    //
    // Undo snapshots carry the editable weight list (ids, labels, base mass),
    // so "restores prior state" is asserted on both the built composite
    // properties (what the physics sees) and the blueprint weight lists.

    /// Four-bar with one 2 kg weight "W1" at (0.03, 0.02) on the first
    /// (sorted) non-ground body. Returns the state, that body id, and a
    /// second non-ground body id.
    fn four_bar_with_one_point_mass() -> (AppState, String, String) {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let ids = sorted_link_ids(&state);
        assert_eq!(state.add_point_mass(&ids[0], 2.0, [0.03, 0.02]).as_deref(), Some("W1"));
        (state, ids[0].clone(), ids[1].clone())
    }

    /// Every blueprint body's weight list (ids, labels, masses, positions).
    fn blueprint_weights(state: &AppState) -> std::collections::BTreeMap<String, Vec<crate::io::PointMassJson>> {
        state
            .blueprint
            .as_ref()
            .unwrap()
            .bodies
            .iter()
            .map(|(id, b)| (id.clone(), b.point_masses.clone()))
            .collect()
    }

    /// Run `edit` and assert it recorded exactly one undo entry, that it
    /// really changed the built composite mass properties (guards against a
    /// vacuous test), and that a single `undo()` restores the prior composite
    /// mass properties, the prior weight lists and the prior undo depth.
    fn assert_edit_is_one_undoable_step(
        what: &str,
        state: &mut AppState,
        edit: impl FnOnce(&mut AppState),
    ) {
        let depth_before = state.undo_history.undo_count();
        let props_before = built_mass_props(state);
        let weights_before = blueprint_weights(state);

        edit(state);

        assert_eq!(
            state.undo_history.undo_count(),
            depth_before + 1,
            "{what}: expected exactly one new undo entry"
        );
        let props_after = built_mass_props(state);
        let changed = props_before.iter().any(|(id, b)| {
            let a = &props_after[id];
            (b.mass - a.mass).abs() > 1e-9
                || (b.cg[0] - a.cg[0]).abs() > 1e-9
                || (b.cg[1] - a.cg[1]).abs() > 1e-9
                || (b.izz - a.izz).abs() > 1e-9
        });
        assert!(changed, "{what}: edit did not change the built mass properties");

        state.undo();
        assert_eq!(
            state.undo_history.undo_count(),
            depth_before,
            "{what}: one undo should consume exactly the one entry"
        );
        assert_mass_props_match(&format!("{what}: after undo"), &props_before, &built_mass_props(state));
        assert_eq!(blueprint_weights(state), weights_before, "{what}: undo must restore the weight lists");
    }

    #[test]
    fn point_mass_numeric_mass_edit_is_one_undo_step_bl025() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        assert_edit_is_one_undoable_step("mass edit", &mut state, |s| {
            assert!(s.set_point_mass_mass(&body, "W1", 5.0));
        });
    }

    #[test]
    fn point_mass_numeric_position_edit_is_one_undo_step_bl025() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        assert_edit_is_one_undoable_step("position edit", &mut state, |s| {
            assert!(s.move_point_mass(&body, "W1", &body, [-0.04, 0.05]));
        });
    }

    #[test]
    fn point_mass_reposition_is_one_undo_step_bl025() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        // Mirrors the canvas Reposition click: world point -> body-local -> move.
        assert_edit_is_one_undoable_step("reposition", &mut state, |s| {
            let [lx, ly] = s.world_to_body_local(&body, 0.07, 0.06);
            assert!(s.move_point_mass(&body, "W1", &body, [lx, ly]));
        });
    }

    #[test]
    fn point_mass_move_to_link_is_one_undo_step_bl025() {
        let (mut state, from_body, to_body) = four_bar_with_one_point_mass();
        assert_edit_is_one_undoable_step("move to link", &mut state, |s| {
            assert!(s.move_point_mass(&from_body, "W1", &to_body, [0.04, -0.01]));

            // The weight left the old body and landed, unchanged, on the new one.
            let bp = s.blueprint.as_ref().unwrap();
            assert!(bp.bodies[&from_body].point_masses.is_empty(), "old body should lose the point mass");
            let moved = &bp.bodies[&to_body].point_masses;
            assert_eq!(moved.len(), 1, "new body should gain exactly one point mass");
            assert_eq!(moved[0].id, "W1", "the weight keeps its id");
            assert!((moved[0].mass - 2.0).abs() < 1e-12);
            assert!((moved[0].local_pos[0] - 0.04).abs() < 1e-12);
            assert!((moved[0].local_pos[1] + 0.01).abs() < 1e-12);
        });
    }

    #[test]
    fn remove_point_mass_by_id_is_one_undo_step() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        assert_edit_is_one_undoable_step("remove", &mut state, |s| {
            assert!(s.remove_point_mass_by_id(&body, "W1"));
            assert!(s.find_point_mass(&body, "W1").is_none());
        });
    }

    #[test]
    fn add_point_mass_is_one_undo_step() {
        let (mut state, _, other) = four_bar_with_one_point_mass();
        assert_edit_is_one_undoable_step("add", &mut state, |s| {
            assert_eq!(s.add_point_mass(&other, 3.0, [0.02, 0.01]).as_deref(), Some("W2"));
        });
    }

    #[test]
    fn last_point_mass_kg_defaults_to_one_kilogram() {
        assert_eq!(AppState::default().last_point_mass_kg, 1.0);
    }

    #[test]
    fn add_point_mass_returns_the_new_id_and_remembers_the_mass() {
        let (mut state, body, other) = four_bar_with_one_point_mass();
        assert_eq!(state.last_point_mass_kg, 2.0, "the fixture's add sets the last mass");

        assert_eq!(state.add_point_mass(&other, 7.5, [0.01, 0.0]).as_deref(), Some("W2"));
        assert_eq!(state.last_point_mass_kg, 7.5);
        assert_eq!(state.add_point_mass(&body, 0.25, [0.0, 0.01]).as_deref(), Some("W3"));
        assert_eq!(
            state.find_point_mass(&body, "W3"),
            Some(&crate::io::PointMassJson {
                id: "W3".to_string(),
                label: None,
                mass: 0.25,
                local_pos: [0.0, 0.01],
            })
        );

        // A freed number is handed out again (smallest unused).
        assert!(state.remove_point_mass_by_id(&other, "W2"));
        assert_eq!(state.add_point_mass(&other, 1.0, [0.0, 0.0]).as_deref(), Some("W2"));
    }

    #[test]
    fn add_point_mass_rejects_invalid_input_without_undo_entry() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        state.last_point_mass_kg = 4.0;
        let depth = state.undo_history.undo_count();
        let weights = blueprint_weights(&state);

        let b = body.as_str();
        for (what, target, mass, pos) in [
            ("ground", GROUND_ID, 1.0, [0.0, 0.0]),
            ("missing body", "no_such_body", 1.0, [0.0, 0.0]),
            ("zero mass", b, 0.0, [0.0, 0.0]),
            ("negative mass", b, -1.0, [0.0, 0.0]),
            ("NaN mass", b, f64::NAN, [0.0, 0.0]),
            ("infinite mass", b, f64::INFINITY, [0.0, 0.0]),
            ("NaN position", b, 1.0, [f64::NAN, 0.0]),
            ("infinite position", b, 1.0, [0.0, f64::INFINITY]),
        ] {
            assert_eq!(state.add_point_mass(target, mass, pos), None, "{what}");
        }

        assert_eq!(state.undo_history.undo_count(), depth, "rejected adds must not push undo entries");
        assert_eq!(blueprint_weights(&state), weights, "rejected adds must not change the model");
        assert_eq!(state.last_point_mass_kg, 4.0, "a rejected add must not change the last mass");
    }

    #[test]
    fn move_point_mass_on_the_same_body_repositions_in_place() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        assert_eq!(state.add_point_mass(&body, 3.0, [0.05, 0.0]).as_deref(), Some("W2"));
        assert!(state.set_point_mass_label(&body, "W1", Some("Torso".to_string())));

        assert!(state.move_point_mass(&body, "W1", &body, [-0.02, 0.04]));

        let pms = &state.blueprint.as_ref().unwrap().bodies[&body].point_masses;
        assert_eq!(pms.len(), 2);
        assert_eq!(
            pms[0],
            crate::io::PointMassJson {
                id: "W1".to_string(),
                label: Some("Torso".to_string()),
                mass: 2.0,
                local_pos: [-0.02, 0.04],
            },
            "same slot, same id/label/mass, new position"
        );
        assert_eq!(pms[1].id, "W2");
    }

    #[test]
    fn move_point_mass_to_another_body_keeps_id_label_and_mass() {
        let (mut state, body, other) = four_bar_with_one_point_mass();
        assert!(state.set_point_mass_label(&body, "W1", Some("Torso".to_string())));

        assert!(state.move_point_mass(&body, "W1", &other, [0.01, -0.02]));

        assert!(state.find_point_mass(&body, "W1").is_none());
        assert_eq!(
            state.find_point_mass(&other, "W1"),
            Some(&crate::io::PointMassJson {
                id: "W1".to_string(),
                label: Some("Torso".to_string()),
                mass: 2.0,
                local_pos: [0.01, -0.02],
            })
        );
    }

    #[test]
    fn set_point_mass_label_is_one_undo_step_and_trims_or_clears() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        let label = |s: &AppState| s.find_point_mass(&body, "W1").expect("W1 must survive").label.clone();
        let depth = state.undo_history.undo_count();

        assert!(state.set_point_mass_label(&body, "W1", Some("  Robot torso ".to_string())));
        assert_eq!(label(&state).as_deref(), Some("Robot torso"));
        assert_eq!(state.undo_history.undo_count(), depth + 1);

        assert!(state.set_point_mass_label(&body, "W1", Some("   ".to_string())));
        assert_eq!(label(&state), None, "a blank label clears it");
        assert_eq!(state.undo_history.undo_count(), depth + 2);

        state.undo();
        assert_eq!(label(&state).as_deref(), Some("Robot torso"), "undo restores the label");
        state.undo();
        assert_eq!(label(&state), None);
        assert_eq!(state.undo_history.undo_count(), depth);
    }

    #[test]
    fn undo_and_redo_restore_the_editable_weight_list() {
        let (mut state, body, other) = four_bar_with_one_point_mass();
        assert!(state.set_point_mass_label(&body, "W1", Some("Torso".to_string())));
        let weights_before = blueprint_weights(&state);
        let base = blueprint_mass_props(&state);
        let built_before = built_mass_props(&state);

        assert!(state.move_point_mass(&body, "W1", &other, [0.04, -0.01]));
        let weights_after = blueprint_weights(&state);
        let built_after = built_mass_props(&state);

        state.undo();
        assert_eq!(blueprint_weights(&state), weights_before, "undo puts W1 (id, label) back on its body");
        assert_mass_props_match("undo: blueprint keeps base mass", &base, &blueprint_mass_props(&state));
        assert_mass_props_match("undo: built composite", &built_before, &built_mass_props(&state));

        state.redo();
        assert_eq!(blueprint_weights(&state), weights_after, "redo re-applies the move");
        assert_mass_props_match("redo: blueprint keeps base mass", &base, &blueprint_mass_props(&state));
        assert_mass_props_match("redo: built composite", &built_after, &built_mass_props(&state));
    }

    #[test]
    fn point_mass_invalid_targets_create_no_undo_entry_bl025() {
        let (mut state, body, other) = four_bar_with_one_point_mass();
        let depth = state.undo_history.undo_count();
        let props = built_mass_props(&state);
        let weights = blueprint_weights(&state);

        let origin = [0.0, 0.0];
        assert!(!state.move_point_mass(&body, "W9", &body, origin), "unknown id");
        assert!(!state.move_point_mass(&body, "", &body, origin), "blank id");
        assert!(!state.move_point_mass(&other, "W1", &other, origin), "weight is on another body");
        assert!(!state.move_point_mass("no_such_body", "W1", &body, origin), "missing source body");
        assert!(!state.move_point_mass(&body, "W1", GROUND_ID, origin), "ground target");
        assert!(!state.move_point_mass(&body, "W1", "no_such_body", origin), "missing target body");
        assert!(!state.move_point_mass(&body, "W1", &body, [f64::NAN, 0.0]), "NaN position");
        assert!(!state.move_point_mass(&body, "W1", &other, [0.0, f64::INFINITY]), "infinite position");
        assert!(!state.set_point_mass_mass(&body, "W1", 0.0), "zero mass");
        assert!(!state.set_point_mass_mass(&body, "W1", -2.0), "negative mass");
        assert!(!state.set_point_mass_mass(&body, "W1", f64::NAN), "NaN mass");
        assert!(!state.set_point_mass_mass(&body, "W9", 3.0), "unknown id");
        assert!(!state.set_point_mass_label(&body, "W9", Some("x".to_string())), "unknown id");
        assert!(!state.remove_point_mass_by_id(&body, "W9"), "unknown id");
        assert!(!state.remove_point_mass_by_id(&body, ""), "blank id");
        assert!(!state.remove_point_mass_by_id(&other, "W1"), "weight is on another body");
        assert!(!state.remove_point_mass_by_id("no_such_body", "W1"), "missing body");

        assert_eq!(state.undo_history.undo_count(), depth, "invalid targets must not push undo entries");
        assert_mass_props_match("invalid targets leave the model untouched", &props, &built_mass_props(&state));
        assert_eq!(blueprint_weights(&state), weights, "invalid targets must not drop or change the weight");
    }

    #[test]
    fn no_op_weight_edits_succeed_without_undo_entry() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        let depth = state.undo_history.undo_count();

        assert!(state.move_point_mass(&body, "W1", &body, [0.03, 0.02]), "same position");
        assert!(state.set_point_mass_mass(&body, "W1", 2.0), "same mass");
        assert!(state.set_point_mass_label(&body, "W1", None), "same (absent) label");
        assert!(state.set_point_mass_label(&body, "W1", Some("  ".to_string())), "blank = absent");

        assert_eq!(state.undo_history.undo_count(), depth, "no-op edits must not push undo entries");
    }

    #[test]
    fn weights_skipped_by_the_loader_can_be_repaired() {
        let (src, body, _) = four_bar_with_one_point_mass();
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        v["bodies"][GROUND_ID]["point_masses"] =
            serde_json::json!([{"id": "on_ground", "mass": 5.0, "local_pos": [0.0, 0.0]}]);
        v["bodies"][body.as_str()]["point_masses"][0]["mass"] = serde_json::json!(0.0);
        let mut state = AppState::default();
        state.load_from_json_str(&v.to_string()).unwrap();
        let base = blueprint_mass_props(&state)[&body].mass;
        let built = |s: &AppState| built_mass_props(s)[&body].mass;
        assert!((built(&state) - base).abs() < 1e-12, "both weights start skipped");

        // A skipped zero-mass weight stays addressable; a valid mass applies it.
        assert!(state.set_point_mass_mass(&body, "W1", 2.0));
        assert!((built(&state) - (base + 2.0)).abs() < 1e-12);

        // A weight on ground can be moved onto a link, where it applies.
        assert!(state.move_point_mass(GROUND_ID, "on_ground", &body, [0.0, 0.0]));
        assert!((built(&state) - (base + 7.0)).abs() < 1e-12);
        assert!(state.find_point_mass(GROUND_ID, "on_ground").is_none());
    }

    // ── BL-024: base-mass edits must keep point masses in the live mechanism ──
    //
    // `set_body_mass` / `set_body_izz` edit the blueprint BASE values and patch
    // the live mechanism without a rebuild. The live body must still equal what
    // a fresh build of the blueprint produces (base + point masses).

    /// Composite mass properties of a mechanism freshly built from the
    /// blueprint (the rebuild path), keyed by body id.
    fn fresh_built_mass_props(state: &AppState) -> std::collections::BTreeMap<String, MassProps> {
        let bp = state.blueprint.as_ref().expect("blueprint should exist");
        let mut mech = crate::io::load_mechanism_unbuilt_from_json(bp).expect("blueprint should load");
        mech.build().expect("mechanism should build");
        mech.bodies()
            .iter()
            .map(|(id, b)| {
                (id.clone(), MassProps { mass: b.mass, cg: [b.cg_local.x, b.cg_local.y], izz: b.izz_cg })
            })
            .collect()
    }

    /// Gravity generalized force of the LIVE mechanism at the current pose
    /// (no rebuild), plus the y-coordinate index of `body_id` in q.
    fn live_gravity_q(state: &AppState, body_id: &str) -> (nalgebra::DVector<f64>, usize) {
        let mech = state.mechanism.as_ref().expect("mechanism should be built");
        let g = mech
            .forces()
            .iter()
            .find_map(|f| match f {
                crate::forces::elements::ForceElement::Gravity(g) => Some(g),
                _ => None,
            })
            .expect("live mechanism should carry a gravity element");
        let q_vec = crate::forces::elements::evaluate_gravity(g, mech.state(), mech.bodies(), &state.q);
        let y_idx = mech.state().get_index(body_id).expect("body index").y_idx();
        (q_vec, y_idx)
    }

    #[test]
    fn set_body_mass_keeps_point_masses_in_live_mechanism_bl024() {
        let (mut state, body, _) = four_bar_with_one_point_mass(); // 2 kg point mass
        assert_eq!(state.mounting_angle, 0.0, "setup: gravity must point along -y");
        let old_base = state.blueprint.as_ref().unwrap().bodies[&body].mass;
        let (q_before, y_idx) = live_gravity_q(&state, &body);

        let new_base = old_base + 3.0;
        state.set_body_mass(&body, new_base);

        // Blueprint keeps the BASE mass; the live body is base + point mass.
        assert_eq!(state.blueprint.as_ref().unwrap().bodies[&body].mass, new_base);
        let live = built_mass_props(&state);
        assert!(
            (live[&body].mass - (new_base + 2.0)).abs() < 1e-12,
            "live composite mass should be new base + 2 kg point mass, got {}",
            live[&body].mass
        );

        // Composite CG and Izz depend on the base mass too: live == fresh rebuild.
        assert_mass_props_match("live vs fresh build after set_body_mass", &fresh_built_mass_props(&state), &live);

        // Gravity Q (no explicit rebuild) reflects the new composite mass.
        let (q_after, _) = live_gravity_q(&state, &body);
        let expected_fy = -state.gravity_magnitude * (new_base + 2.0);
        assert!(
            (q_after[y_idx] - expected_fy).abs() < 1e-9,
            "gravity Q_y should be -g * (new base + point masses) = {}, got {}",
            expected_fy,
            q_after[y_idx]
        );
        assert!(
            (q_after[y_idx] - q_before[y_idx]).abs() > 1.0,
            "setup: the edit must actually change the gravity load"
        );
    }

    #[test]
    fn set_body_mass_to_zero_leaves_only_point_mass_in_live_mechanism_bl024() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        state.set_body_mass(&body, 0.0);

        let live = built_mass_props(&state);
        assert!(
            (live[&body].mass - 2.0).abs() < 1e-12,
            "with zero base mass the live body carries just the 2 kg point mass, got {}",
            live[&body].mass
        );
        assert_mass_props_match("live vs fresh build at zero base mass", &fresh_built_mass_props(&state), &live);
    }

    #[test]
    fn set_body_izz_keeps_point_masses_in_live_mechanism_bl024() {
        let (mut state, body, _) = four_bar_with_one_point_mass();
        let mass_before = built_mass_props(&state)[&body].mass;
        let cg_before = built_mass_props(&state)[&body].cg;
        let old_base_izz = state.blueprint.as_ref().unwrap().bodies[&body].izz_cg;

        let new_base_izz = old_base_izz + 0.05;
        state.set_body_izz(&body, new_base_izz);

        assert_eq!(state.blueprint.as_ref().unwrap().bodies[&body].izz_cg, new_base_izz);
        let live = built_mass_props(&state);
        // Composite Izz keeps the point-mass parallel-axis term: live == fresh rebuild.
        assert_mass_props_match("live vs fresh build after set_body_izz", &fresh_built_mass_props(&state), &live);
        assert!(
            live[&body].izz > new_base_izz + 1e-6,
            "composite Izz {} must exceed base Izz {} by the point-mass contribution",
            live[&body].izz,
            new_base_izz
        );
        // Izz does not move mass or CG.
        assert!((live[&body].mass - mass_before).abs() < 1e-12);
        assert!((live[&body].cg[0] - cg_before[0]).abs() < 1e-12);
        assert!((live[&body].cg[1] - cg_before[1]).abs() < 1e-12);
    }

    /// The no-rebuild mass sync must apply the loader's skip rules too: a
    /// weight the loader rejected (here a negative mass, kept in the
    /// blueprint) stays out of the live body after a base-mass edit.
    #[test]
    fn set_body_mass_skips_point_masses_the_loader_rejects() {
        let (src, body, _) = four_bar_with_one_point_mass(); // valid 2 kg "W1"
        let mut v: serde_json::Value =
            serde_json::from_str(&src.serialize_to_json_string().unwrap()).unwrap();
        v["bodies"][body.as_str()]["point_masses"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!({"id": "neg", "mass": -3.0, "local_pos": [0.05, 0.0]}));
        let mut state = AppState::default();
        state.load_from_json_str(&v.to_string()).expect("a file with a bad weight still loads");
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body].point_masses.len(),
            2,
            "setup: the rejected weight stays in the blueprint"
        );

        let new_base = state.blueprint.as_ref().unwrap().bodies[&body].mass + 1.0;
        state.set_body_mass(&body, new_base);

        let live = built_mass_props(&state);
        assert!(
            (live[&body].mass - (new_base + 2.0)).abs() < 1e-12,
            "live mass should be new base + the valid 2 kg weight only, got {}",
            live[&body].mass
        );
        assert_mass_props_match("live vs fresh build with a rejected weight", &fresh_built_mass_props(&state), &live);
    }

    /// The parametric BodyMass / BodyIzz sweep varies the BASE value on a
    /// blueprint clone and rebuilds through the loader, so point masses stay
    /// on top of the swept value at every step (documented on `SweepParameter`).
    #[test]
    fn parametric_body_mass_sweep_varies_base_and_keeps_point_masses_bl024() {
        let (state, body, _) = four_bar_with_one_point_mass();
        for swept in [0.4, 1.0, 7.5] {
            let mut bp = state.blueprint.as_ref().unwrap().clone();
            let mut omega = state.driver_omega();
            assert!(AppState::set_parameter_on_blueprint(
                &mut bp,
                &SweepParameter::BodyMass(body.clone()),
                swept,
                &mut omega,
            ));
            let mut mech = crate::io::load_mechanism_unbuilt_from_json(&bp).unwrap();
            mech.build().unwrap();
            assert!(
                (mech.bodies()[&body].mass - (swept + 2.0)).abs() < 1e-12,
                "swept base {swept} kg + 2 kg point mass, got {}",
                mech.bodies()[&body].mass
            );
        }
    }

    #[test]
    fn new_empty_mechanism_resets_state() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(state.current_sample.is_some());

        // Add a recent file so we can verify it persists
        state.recent_files.push(std::path::PathBuf::from("/fake/file.json"));
        state.display_units.length = LengthUnit::Meters;

        state.new_empty_mechanism();

        // Mechanism should be ground-only
        assert!(state.current_sample.is_none());
        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.bodies.len(), 1, "should have only ground body");
        assert!(bp.bodies.contains_key(GROUND_ID));
        assert!(bp.joints.is_empty());

        // Preferences should persist
        assert!(
            state.recent_files.iter().any(|p| p.to_string_lossy().contains("fake")),
            "recent_files should contain the fake entry we added"
        );
        assert_eq!(state.display_units.length, LengthUnit::Meters);
    }

    #[test]
    fn add_prismatic_joint_creates_joint_in_blueprint() {
        let mut state = AppState::default();
        // Build a minimal mechanism with two bodies + ground pivots.
        state.add_ground_pivot("PX", 0.0, 0.0);
        let body_id = state.next_body_id();
        state.push_undo();
        state.add_body_with_points_raw(&body_id, &[("A".into(), [0.0, 0.0]), ("B".into(), [0.1, 0.0])]);
        state.rebuild();
        let bp = state.blueprint.as_ref().unwrap();
        let body_point = bp.bodies[&body_id]
            .attachment_points.keys().next().unwrap().clone();

        let joints_before = state.blueprint.as_ref().unwrap().joints.len();
        state.add_prismatic_joint(GROUND_ID, "PX", &body_id, &body_point);

        let bp = state.blueprint.as_ref().unwrap();
        assert_eq!(bp.joints.len(), joints_before + 1);
        // Find the new joint and verify it's prismatic
        let new_joint = bp.joints.values().last().unwrap();
        match new_joint {
            JointJson::Prismatic { axis_local_i, .. } => {
                // Axis should be a unit vector
                let len = (axis_local_i[0].powi(2) + axis_local_i[1].powi(2)).sqrt();
                assert!(
                    (len - 1.0).abs() < 1e-6 || len < 1e-12,
                    "axis should be normalized or zero, got length {}",
                    len
                );
            }
            _ => panic!("Expected Prismatic joint, got {:?}", new_joint),
        }
    }

    #[test]
    fn add_fixed_joint_creates_joint_in_blueprint() {
        let mut state = AppState::default();
        state.add_ground_pivot("PX", 0.0, 0.0);
        let body_id = state.next_body_id();
        state.push_undo();
        state.add_body_with_points_raw(&body_id, &[("A".into(), [0.0, 0.0]), ("B".into(), [0.1, 0.0])]);
        state.rebuild();
        let bp = state.blueprint.as_ref().unwrap();
        let body_point = bp.bodies[&body_id]
            .attachment_points.keys().next().unwrap().clone();

        state.add_fixed_joint(GROUND_ID, "PX", &body_id, &body_point);

        let bp = state.blueprint.as_ref().unwrap();
        let has_fixed = bp.joints.values().any(|j| matches!(j, JointJson::Fixed { .. }));
        assert!(has_fixed, "Should have a Fixed joint in the blueprint");
    }

    #[test]
    fn add_prismatic_joint_is_undoable() {
        let mut state = AppState::default();
        state.add_ground_pivot("PX", 0.0, 0.0);
        let body_id = state.next_body_id();
        state.push_undo();
        state.add_body_with_points_raw(&body_id, &[("A".into(), [0.0, 0.0]), ("B".into(), [0.1, 0.0])]);
        state.rebuild();
        let bp = state.blueprint.as_ref().unwrap();
        let body_point = bp.bodies[&body_id]
            .attachment_points.keys().next().unwrap().clone();

        let joints_before = state.blueprint.as_ref().unwrap().joints.len();
        state.add_prismatic_joint(GROUND_ID, "PX", &body_id, &body_point);
        assert_eq!(state.blueprint.as_ref().unwrap().joints.len(), joints_before + 1);

        state.undo();
        assert_eq!(
            state.blueprint.as_ref().unwrap().joints.len(),
            joints_before,
            "undo should remove the prismatic joint"
        );
    }

    // ── Parametric study tests ────────────────────────────────────────────

    #[test]
    fn available_parameters_includes_body_mass_and_driver_omega() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        let params = state.available_parameters();
        assert!(!params.is_empty());
        assert!(
            params.iter().any(|p| matches!(p, SweepParameter::DriverOmega)),
            "should include DriverOmega"
        );
        assert!(
            params.iter().any(|p| matches!(p, SweepParameter::BodyMass(_))),
            "should include at least one BodyMass"
        );
    }

    #[test]
    fn run_parametric_study_produces_results() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Use PeakKineticEnergy — always non-zero for a mechanism with mass
        state.parametric_config = ParametricStudyConfig {
            parameter: SweepParameter::DriverOmega,
            min_value: 1.0,
            max_value: 10.0,
            num_steps: 3,
            metric: ParametricMetric::PeakKineticEnergy,
        };

        state.run_parametric_study();

        let result = state.parametric_result.as_ref().expect("should have results");
        assert_eq!(result.parameter_values.len(), 3);
        assert_eq!(result.metric_values.len(), 3);
        assert!(
            result.metric_values.iter().all(|v| v.is_finite()),
            "all metric values should be finite, got {:?}",
            result.metric_values
        );
    }

    #[test]
    fn parametric_study_body_mass_sweep() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Find a non-ground body
        let body_id = {
            let bp = state.blueprint.as_ref().unwrap();
            bp.bodies.keys().find(|k| k.as_str() != GROUND_ID).unwrap().clone()
        };

        // Sweep mass — just verify it produces finite results at each step
        state.parametric_config = ParametricStudyConfig {
            parameter: SweepParameter::BodyMass(body_id),
            min_value: 0.5,
            max_value: 5.0,
            num_steps: 3,
            metric: ParametricMetric::PeakKineticEnergy,
        };

        state.run_parametric_study();

        let result = state.parametric_result.as_ref().expect("should have results");
        assert_eq!(result.parameter_values.len(), 3);
        assert!(
            result.metric_values.iter().all(|v| v.is_finite()),
            "all metric values should be finite, got {:?}",
            result.metric_values
        );
    }

    #[test]
    fn parametric_metric_extract_peak_ke() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.compute_sweep();
        let sweep = state.sweep_data.as_ref().expect("should have sweep data");

        let ke = ParametricMetric::PeakKineticEnergy.extract(sweep);
        assert!(ke >= 0.0, "peak KE should be non-negative, got {}", ke);
    }

    // ── Counterbalance tests ──────────────────────────────────────────────

    #[test]
    fn counterbalance_study_produces_results() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Use "crank" body point A and ground point O2
        state.counterbalance_config = CounterbalanceConfig {
            body_a: GROUND_ID.to_string(),
            point_a: "O2".to_string(),
            body_b: "crank".to_string(),
            point_b: "B".to_string(),
            k_min: 10.0,
            k_max: 100.0,
            k_steps: 3,
            free_length_min: 0.01,
            free_length_max: 0.05,
            free_length_steps: 2,
        };

        state.run_counterbalance_study();

        let result = state.counterbalance_result.as_ref().expect("should have results");
        assert_eq!(result.k_values.len(), 3);
        assert_eq!(result.fl_values.len(), 2);
        assert_eq!(result.grid.len(), 3);
        assert_eq!(result.grid[0].len(), 2);
        assert!(!result.angles_deg.is_empty(), "should have sweep angles");
        assert!(!result.baseline_torques.is_empty(), "should have baseline torques");
    }

    // ── Error panel / simulation error surfacing tests ────────────────────

    #[test]
    fn run_simulation_no_blueprint_no_panic() {
        let mut state = AppState::default();
        state.blueprint = None;
        state.run_simulation(1.0);
        // No blueprint means early return with no errors logged.
        assert!(state.error_log.is_empty());
    }

    #[test]
    fn run_simulation_corrupted_mechanism_surfaces_error() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        if let Some(ref mut bp) = state.blueprint {
            bp.bodies.clear();
        }
        state.run_simulation(1.0);
        assert!(
            !state.error_log.is_empty(),
            "Simulation with corrupted mechanism should surface an error"
        );
        assert!(state.show_error_panel, "Error panel should auto-show on failure");
    }

    #[test]
    fn run_simulation_valid_mechanism_succeeds() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.run_simulation(1.0);
        if state.simulation.is_some() {
            assert!(state.error_log.is_empty());
        } else {
            assert!(!state.error_log.is_empty());
        }
    }

    #[test]
    fn rebuild_marks_sweep_dirty() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.sweep_dirty = false; // reset for test
        state.rebuild();
        assert!(state.sweep_dirty, "rebuild() should mark sweep as dirty");
    }

    #[test]
    fn load_sample_produces_sweep_data() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        assert!(
            state.sweep_data.is_some(),
            "load_sample should produce sweep data immediately"
        );
        let sweep = state.sweep_data.as_ref().unwrap();
        assert!(
            !sweep.angles_deg.is_empty(),
            "Sweep should have angle data"
        );
        assert!(
            !sweep.joint_reaction_magnitudes.is_empty(),
            "Sweep should have joint reaction data"
        );
    }

    #[test]
    fn gravity_magnitude_zero_disables_gravity_force() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.gravity_magnitude = 0.0;
        state.sync_gravity();
        let mech = state.mechanism.as_ref().unwrap();
        assert!(
            !mech.forces().iter().any(|f| matches!(f, ForceElement::Gravity(_))),
            "Gravity should be removed when magnitude is 0"
        );
    }

    #[test]
    fn gravity_magnitude_nondefault_updates_element() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.gravity_magnitude = 3.71;
        state.sync_gravity();
        let mech = state.mechanism.as_ref().unwrap();
        let grav = mech.forces().iter().find(|f| matches!(f, ForceElement::Gravity(_)));
        assert!(grav.is_some(), "Gravity element should exist");
        if let Some(ForceElement::Gravity(g)) = grav {
            assert!((g.g_vector[1] - (-3.71)).abs() < 1e-10);
        }
    }

    #[test]
    fn set_link_length_maintains_direction() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        let pa = crank.attachment_points.get("A").unwrap();
        let pb = crank.attachment_points.get("B").unwrap();
        let dx = pb[0] - pa[0];
        let dy = pb[1] - pa[1];
        let orig_len = (dx * dx + dy * dy).sqrt();
        let orig_angle = dy.atan2(dx);

        let new_len = orig_len * 1.5;
        state.set_link_length("crank", "A", "B", new_len);

        let bp = state.blueprint.as_ref().unwrap();
        let crank = bp.bodies.get("crank").unwrap();
        let pa2 = crank.attachment_points.get("A").unwrap();
        let pb2 = crank.attachment_points.get("B").unwrap();
        let dx2 = pb2[0] - pa2[0];
        let dy2 = pb2[1] - pa2[1];
        let actual_len = (dx2 * dx2 + dy2 * dy2).sqrt();
        let actual_angle = dy2.atan2(dx2);

        assert!(
            (actual_len - new_len).abs() < 1e-10,
            "Length should be {}, got {}", new_len, actual_len
        );
        assert!(
            (actual_angle - orig_angle).abs() < 1e-10,
            "Direction should be preserved: expected {}, got {}", orig_angle, actual_angle
        );
    }

    #[test]
    fn set_link_length_point_a_stays_fixed() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        let bp = state.blueprint.as_ref().unwrap();
        let pa_before = *bp.bodies.get("crank").unwrap()
            .attachment_points.get("A").unwrap();

        state.set_link_length("crank", "A", "B", 0.05);

        let bp = state.blueprint.as_ref().unwrap();
        let pa_after = bp.bodies.get("crank").unwrap()
            .attachment_points.get("A").unwrap();

        assert!((pa_after[0] - pa_before[0]).abs() < 1e-12);
        assert!((pa_after[1] - pa_before[1]).abs() < 1e-12);
    }

    #[test]
    fn integration_gravity_affects_driver_torque() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Give bodies non-zero mass so gravity has an observable effect.
        state.set_body_mass("crank", 0.5);
        state.set_body_mass("coupler", 1.0);
        state.set_body_mass("rocker", 0.5);

        // With gravity
        state.gravity_magnitude = 9.81;
        state.sync_gravity();
        state.compute_sweep();
        let sweep_grav = state.sweep_data.as_ref().unwrap();
        let torques_grav: Vec<f64> = sweep_grav
            .driver_torques
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .collect();

        // Without gravity
        state.gravity_magnitude = 0.0;
        state.sync_gravity();
        state.compute_sweep();
        let sweep_no_grav = state.sweep_data.as_ref().unwrap();
        let torques_no_grav: Vec<f64> = sweep_no_grav
            .driver_torques
            .as_ref()
            .unwrap()
            .iter()
            .copied()
            .collect();

        assert_eq!(torques_grav.len(), torques_no_grav.len());
        let differs = torques_grav
            .iter()
            .zip(torques_no_grav.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            differs,
            "Driver torque should change when gravity is toggled"
        );
    }

    #[test]
    fn integration_force_add_uses_context() {
        use crate::gui::force_toolbar::resolve_target_bodies;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // No selection — should return None
        state.selected = None;
        let (sel, _) = resolve_target_bodies(&state);
        assert!(sel.is_none());

        // Select crank — should resolve
        state.selected = Some(SelectedEntity::Body("crank".to_string()));
        let (sel, conn) = resolve_target_bodies(&state);
        assert_eq!(sel, Some("crank".to_string()));
        assert!(conn.is_some());
    }

    #[test]
    fn all_samples_have_sweep_data_after_load() {
        for sample in SampleMechanism::all() {
            let mut state = AppState::default();
            state.load_sample(*sample);
            assert!(
                state.sweep_data.is_some(),
                "Sample {:?} should have sweep data after load",
                sample
            );
        }
    }

    #[test]
    fn mounting_angle_rotates_gravity_vector() {
        let mut state = AppState::default();
        state.load_sample(crate::gui::samples::SampleMechanism::FourBar);
        state.gravity_magnitude = 9.81;
        state.mounting_angle = std::f64::consts::FRAC_PI_2; // 90°
        state.rebuild();

        let mech = state.mechanism.as_ref().unwrap();
        let gravity_forces: Vec<_> = mech.forces().iter()
            .filter_map(|f| match f {
                crate::forces::elements::ForceElement::Gravity(g) => Some(g),
                _ => None,
            })
            .collect();

        assert_eq!(gravity_forces.len(), 1);
        let g = gravity_forces[0];
        // At 90° mount, gravity should point in -x: (-9.81, ~0)
        assert!((g.g_vector[0] - (-9.81)).abs() < 1e-6,
            "g_x should be -9.81, got {}", g.g_vector[0]);
        assert!(g.g_vector[1].abs() < 1e-6,
            "g_y should be ~0, got {}", g.g_vector[1]);
    }

    #[test]
    fn mounting_angle_round_trips_through_json() {
        let mut state = AppState::default();
        state.load_sample(crate::gui::samples::SampleMechanism::FourBar);
        state.mounting_angle = 0.5; // ~28.6 degrees

        // Sync to blueprint via rebuild
        state.rebuild();

        // Verify blueprint has the angle
        let bp = state.blueprint.as_ref().unwrap();
        assert!((bp.mounting_angle - 0.5).abs() < 1e-10,
            "blueprint should have mounting_angle 0.5, got {}", bp.mounting_angle);

        // Serialize and deserialize
        let json = serde_json::to_string(bp).unwrap();
        assert!(json.contains("mounting_angle"),
            "JSON should contain mounting_angle field");

        let reloaded: crate::io::MechanismJson = serde_json::from_str(&json).unwrap();
        assert!((reloaded.mounting_angle - 0.5).abs() < 1e-10,
            "round-trip should preserve mounting_angle, got {}", reloaded.mounting_angle);
    }

    #[test]
    fn mounting_angle_defaults_to_zero_for_old_files() {
        // Simulate loading an old file that has no mounting_angle field.
        let json = r#"{
            "schema_version": "1.0.0",
            "bodies": {
                "ground": {
                    "attachment_points": {},
                    "mass": 0.0,
                    "cg_local": [0.0, 0.0],
                    "izz_cg": 0.0
                }
            },
            "joints": {}
        }"#;
        let parsed: crate::io::MechanismJson = serde_json::from_str(json).unwrap();
        assert!((parsed.mounting_angle - 0.0).abs() < 1e-10,
            "missing mounting_angle should default to 0.0, got {}", parsed.mounting_angle);
    }

    #[test]
    fn flip_assembly_branch_lands_on_alternate_config() {
        // 4-bar crank rockers have two valid assembly branches. Flipping
        // across the ground line must converge and produce a q that
        // differs from the original by more than numerical noise.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        // Move off the symmetric start so the two branches are distinct.
        state.solve_at_angle(45f64.to_radians());
        assert!(state.solver_status.converged, "setup must converge");
        let q_before = state.q.clone();

        state.flip_assembly_branch();

        assert!(state.solver_status.converged, "flip must converge");
        let diff = (&state.q - &q_before).norm();
        assert!(diff > 1e-3, "flip must produce a distinctly different q, got drift={}", diff);
    }

    #[test]
    fn flip_assembly_branch_noops_without_mechanism() {
        // A freshly built AppState has a tiny ground-only mechanism with
        // no non-ground bodies to reflect. The call should leave state
        // essentially unchanged and not panic.
        let mut state = AppState::default();
        let q_before = state.q.clone();
        state.flip_assembly_branch();
        // q is either unchanged OR equal to a re-solved version of the
        // same ground-only config; either way norm difference is ~0.
        let diff = (&state.q - &q_before).norm();
        assert!(diff < 1e-9, "flip on ground-only must not move anything, got drift={}", diff);
    }

    // ── Motion ribbon smoke test ────────────────────────────────────────
    #[test]
    fn motion_ribbon_toggle_does_not_panic_with_no_sweep_data() {
        // Toggling the ribbon on without a completed trajectory sweep
        // must be a no-op: the renderer short-circuits when sweep_data
        // is missing or u_values is empty. This is a regression guard
        // for the canvas drawing path under construction-time states.
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.show_motion_ribbon = true;
        state.motion_ribbon_n_ghosts = 8;
        // Ribbon is gated on Trajectory mode; we don't enter it in this
        // test — just verify the field flips and defaults are sensible.
        assert!(state.show_motion_ribbon);
        assert!(state.sweep_data.is_some()); // sample sets up an Angle sweep
    }

    #[test]
    fn motion_ribbon_pose_resolve_loop_completes_for_small_trajectory() {
        // Asserts the per-ghost solve loop converges across N samples
        // for a canonical 4-bar with a feasible Angle target trajectory.
        // This is the same loop the canvas runs each frame, so it
        // exercises the cost-per-frame path.
        use crate::gui::sweep::SweepMode;
        use crate::gui::state::{Trajectory, TrajectoryProfile};
        use crate::solver::inverse_kinematics::{ControlTarget, Severity};
        use crate::solver::kinematics::solve_position;

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);

        // Switch into a feasible Trajectory mode covering a small angle range.
        let target = ControlTarget::angle("crank");
        state.sweep_mode = SweepMode::Trajectory {
            target,
            trajectory: Trajectory::Profile(TrajectoryProfile {
                shape: crate::gui::state::MotionProfile::ConstantSpeed,
                start_value: 0.5,
                end_value: 1.5,
                duration: 1.0,
            }),
            severity: Severity::Analysis,
            n_samples: 10,
        };
        state.compute_sweep();

        let sweep_data = state.sweep_data.as_ref().expect("sweep should populate");
        let u_values = sweep_data
            .u_values
            .as_ref()
            .expect("trajectory sweep populates u_values");
        assert_eq!(u_values.len(), 10);

        // Run the same loop the renderer would: 10 ghosts, warm-started.
        let mech = state.mechanism.as_ref().unwrap();
        let nominal_rate = state.driver_omega();
        let u_0 = state.driver_theta_0();
        let mut q_seed = state.last_good_q.clone();
        let mut converged_count = 0;
        for &u_k in u_values.iter() {
            let t_mech = (u_k - u_0) / nominal_rate;
            if let Ok(res) = solve_position(mech, &q_seed, t_mech, 1e-10, 50) {
                if res.converged {
                    q_seed = res.q;
                    converged_count += 1;
                }
            }
        }
        // We don't require all to converge (some samples on the
        // trajectory may straddle a singularity), but the bulk should.
        assert!(
            converged_count >= 8,
            "expected >=8/10 ghost solves to converge, got {}",
            converged_count
        );
    }

    // ── Trajectory playback tick ───────────────────────────────────────
    #[test]
    fn trajectory_playback_advances_t_by_dt_times_speed() {
        use crate::gui::sweep::SweepMode;
        use crate::gui::state::{Trajectory, TrajectoryProfile};
        use crate::solver::inverse_kinematics::{ControlTarget, Severity};

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.sweep_mode = SweepMode::Trajectory {
            target: ControlTarget::angle("crank"),
            trajectory: Trajectory::Profile(TrajectoryProfile {
                shape: crate::gui::state::MotionProfile::ConstantSpeed,
                start_value: 0.5,
                end_value: 1.5,
                duration: 2.0,
            }),
            severity: Severity::Analysis,
            n_samples: 50,
        };
        state.compute_sweep();
        state.trajectory_playback_active = true;
        state.trajectory_playback_t = 0.0;
        state.trajectory_playback_speed = 1.0;
        state.trajectory_playback_loop = true;

        let stepped = state.step_trajectory_playback(0.1);
        assert!(stepped, "step should report active");
        assert!(
            (state.trajectory_playback_t - 0.1).abs() < 1e-12,
            "t should advance by dt*speed = 0.1, got {}",
            state.trajectory_playback_t
        );

        // Half-speed.
        state.trajectory_playback_speed = 0.5;
        state.step_trajectory_playback(0.2);
        assert!(
            (state.trajectory_playback_t - 0.2).abs() < 1e-12,
            "t should be 0.1 + 0.2*0.5 = 0.2, got {}",
            state.trajectory_playback_t
        );

        // Plot cursor should track playback time.
        assert_eq!(state.last_trajectory_scrub_t, Some(0.2));
    }

    #[test]
    fn trajectory_playback_loops_at_duration_when_loop_enabled() {
        use crate::gui::sweep::SweepMode;
        use crate::gui::state::{Trajectory, TrajectoryProfile};
        use crate::solver::inverse_kinematics::{ControlTarget, Severity};

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.sweep_mode = SweepMode::Trajectory {
            target: ControlTarget::angle("crank"),
            trajectory: Trajectory::Profile(TrajectoryProfile {
                shape: crate::gui::state::MotionProfile::ConstantSpeed,
                start_value: 0.5,
                end_value: 1.5,
                duration: 1.0,
            }),
            severity: Severity::Analysis,
            n_samples: 20,
        };
        state.compute_sweep();
        state.trajectory_playback_active = true;
        state.trajectory_playback_t = 0.95;
        state.trajectory_playback_speed = 1.0;
        state.trajectory_playback_loop = true;

        // dt=0.1 pushes past the 1.0 duration: should wrap to 0.05.
        state.step_trajectory_playback(0.1);
        assert!(state.trajectory_playback_active, "loop mode keeps playing");
        assert!(
            state.trajectory_playback_t < 1.0,
            "looped t should wrap below duration, got {}",
            state.trajectory_playback_t
        );
        assert!(
            (state.trajectory_playback_t - 0.05).abs() < 1e-9,
            "expected wrap to 0.05, got {}",
            state.trajectory_playback_t
        );
    }

    #[test]
    fn trajectory_playback_stops_at_duration_when_loop_disabled() {
        use crate::gui::sweep::SweepMode;
        use crate::gui::state::{Trajectory, TrajectoryProfile};
        use crate::solver::inverse_kinematics::{ControlTarget, Severity};

        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.sweep_mode = SweepMode::Trajectory {
            target: ControlTarget::angle("crank"),
            trajectory: Trajectory::Profile(TrajectoryProfile {
                shape: crate::gui::state::MotionProfile::ConstantSpeed,
                start_value: 0.5,
                end_value: 1.5,
                duration: 1.0,
            }),
            severity: Severity::Analysis,
            n_samples: 20,
        };
        state.compute_sweep();
        state.trajectory_playback_active = true;
        state.trajectory_playback_t = 0.95;
        state.trajectory_playback_speed = 1.0;
        state.trajectory_playback_loop = false;

        state.step_trajectory_playback(0.1);
        assert!(
            !state.trajectory_playback_active,
            "non-loop playback stops at end"
        );
        assert!(
            (state.trajectory_playback_t - 1.0).abs() < 1e-12,
            "non-loop playback clamps to duration, got {}",
            state.trajectory_playback_t
        );
    }

    #[test]
    fn trajectory_playback_inactive_returns_false_immediately() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::FourBar);
        state.trajectory_playback_active = false;
        let t_before = state.trajectory_playback_t;
        let stepped = state.step_trajectory_playback(0.1);
        assert!(!stepped);
        assert_eq!(state.trajectory_playback_t, t_before);
    }

    // ── BL-022: warm start across a topology-changing rebuild ────────────

    /// Seeding the expanded (cylinder + rod) mechanism from the direct one
    /// keeps every surviving body's pose by id and starts new bodies at the
    /// origin; it falls back to zeros without a usable previous state.
    #[test]
    fn seed_q_by_body_id_keeps_surviving_poses_bl022() {
        let mut state = AppState::default();
        state.load_sample(SampleMechanism::ParallelogramActuator);
        let prev = state.mechanism.as_ref().unwrap();
        let prev_q = state.last_good_q.clone();
        assert_eq!(prev_q.len(), prev.state().n_coords());

        let mut expanded =
            crate::io::load_mechanism_unbuilt_from_json(state.blueprint.as_ref().unwrap()).unwrap();
        expanded.build().unwrap();
        assert!(expanded.state().n_coords() > prev_q.len(), "fixture must add compound bodies");

        let seeded = seed_q_by_body_id(Some(prev), &prev_q, &expanded);
        assert_eq!(seeded.len(), expanded.state().n_coords());
        for id in prev.state().body_ids() {
            assert_eq!(
                expanded.state().get_pose(&id, &seeded),
                prev.state().get_pose(&id, &prev_q),
                "surviving body {id} must keep its pose"
            );
        }
        for id in ["force_0_cyl", "force_0_rod"] {
            assert_eq!(expanded.state().get_pose(id, &seeded), (0.0, 0.0, 0.0));
        }

        let zeros = expanded.state().make_q();
        assert_eq!(seed_q_by_body_id(None, &prev_q, &expanded), zeros);
        let wrong_len = DVector::zeros(prev_q.len() + 1);
        assert_eq!(seed_q_by_body_id(Some(prev), &wrong_len, &expanded), zeros);
    }

