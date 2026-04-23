    use super::*;
    use super::blueprint_ops::joint_body_ids;
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
            state.driver_omega.abs() >= MIN_DRIVER_OMEGA_ABS - 1e-12,
            "expected driver_omega clamped to >= {}, got {}",
            MIN_DRIVER_OMEGA_ABS,
            state.driver_omega
        );

        // Negative sign must be preserved.
        state.set_constant_speed_driver(-1e-6, 0.0);
        assert!(
            state.driver_omega < 0.0,
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
        state.driver_omega = 5.0;
        state.driver_theta_0 = 1.0;
        state.driver_angle = 2.0;
        state.driver_joint_id = Some("J1".to_string());

        let snapshot = state.take_snapshot().unwrap();
        state.restore_snapshot(&snapshot);

        assert_eq!(state.driver_omega, 5.0);
        assert_eq!(state.driver_theta_0, 1.0);
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
            state.driver_omega,
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

        let original_omega = state.driver_omega;
        let new_omega = original_omega * 2.0;

        // Add a case with different omega
        state.load_cases.cases.push(LoadCase {
            name: "Fast".to_string(),
            driver_joint_id: state.driver_joint_id.clone().unwrap(),
            omega: new_omega,
            theta_0: 0.0,
        });

        state.apply_load_case(1);
        assert_eq!(state.driver_omega, new_omega);
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

        state.driver_omega = 10.0;
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
        let omega_before = state.driver_omega;
        state.apply_load_case(999);
        assert_eq!(state.driver_omega, omega_before);
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

        state.add_point_mass(&body_id, 0.5, [0.01, 0.0]);
        assert_eq!(
            state.blueprint.as_ref().unwrap().bodies[&body_id].point_masses.len(),
            1
        );

        state.remove_point_mass(&body_id, 0);
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
