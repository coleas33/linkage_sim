# Verified: linkage-gui-state
Worktree HEAD 1048962. All line numbers re-read in C:/Users/Cole/source/repos/lsim-audit/linkage-sim-rs/src/gui.

## LGS-01: Drag-and-drop file import only reads `file.bytes`, which native eframe never fills
- Verdict: PARTLY (the code reads as described; the native-egui behaviour is not provable from the worktree)
- Checked: mod.rs:103-110 reads only `file.bytes`; `git grep DroppedFile` and `git grep "\.path\b"` in src/gui find no read of `DroppedFile::path` (only entry.path() in templates.rs). The egui-winit source is not in ~/.cargo/registry, so "native fills path, not bytes" cannot be proven here.
- Safe to apply without judgment: no (behaviour change, needs native check)

## LGS-02: Autosave recovery deletes the autosave file even when loading it failed
- Verdict: CONFIRMED
- Checked: mod.rs:1022-1029 (native: log::error then unconditional remove_file) and 1065-1073 (wasm: log then wasm_clear_autosave). No status_message or error_log on Err.
- Safe to apply without judgment: no (changes recovery behaviour)

## LGS-03: Shortcut help lists Ctrl+S and Ctrl+Shift+S on the web build, where no handler exists
- Verdict: PARTLY. Holds: rows at mod.rs:1166-1167 are in an unconditional array; the only `Key::S` handlers (1259, 1276) are `#[cfg(feature = "native")]`; their failures only reach log::error! (1262, 1270, 1283). Does not hold: "unlike the menu save" - menu_bar.rs:203-213 and 226-228 also only log::error!, so the menu save behaves the same (see LGS-10).
- Checked: git grep "Key::S" (2 hits); mod.rs 1164-1168, 1255-1285; menu_bar.rs 198-230.
- Safe to apply without judgment: no (UI behaviour on web)

## LGS-04: The driver-angle to time mapping is written out five times in scope, and one copy lacks the zero-omega guard
- Verdict: CONFIRMED
- Checked: guarded copies at state/mod.rs:1197, 1341, blueprint_ops.rs:346, 562, undo_ops.rs:47 (snapshot fields); unguarded at mod.rs:1300, file_io.rs:408, 423. The comment at mod.rs:1191-1193 claims file_io uses the same guarded pattern, which is not so. Out-of-scope copy sweep/mod.rs:1520 also unguarded.
- Safe to apply without judgment: no (refactor across sites, unguarded sites would change at omega 0)

## LGS-05: Same block of code copied within state
- Verdict: CONFIRMED
- Checked: (a) state/mod.rs:1080-1092 vs blueprint_ops.rs:386-396 identical (third copy file_io.rs:483-493); (b) blueprint_ops.rs:1345-1351 vs 1393-1397 identical; (c) blueprint_ops.rs:578-583 vs mod.rs:1396-1401 same g_vector formula.
- Safe to apply without judgment: no (refactor)

## LGS-06: Doc comment of `solve_at_angle` is split
- Verdict: CONFIRMED
- Checked: state/mod.rs:1118-1119 hold "Solve the position problem for the given driver angle (radians)." + "///", followed by "Recompute `driver_display_offset`..." for recompute_driver_display_offset (1127); mod.rs:1182 begins "Uses `last_good_q`..." directly above solve_at_angle (1186).
- Safe to apply without judgment: yes (doc comments only)
- Exact change: file linkage-sim-rs/src/gui/state/mod.rs. Delete lines 1118-1119 (`    /// Solve the position problem for the given driver angle (radians).` and `    ///`). Then, immediately above the current line 1182 (`    /// Uses `last_good_q` as the initial guess for Newton-Raphson.`), insert those two lines in the same order, so the block reads: `/// Solve the position problem for the given driver angle (radians).` / `///` / `/// Uses `last_good_q` ...`.

## LGS-07: Menu "Parametric Study [WIP]" is a permanently disabled checkbox
- Verdict: CONFIRMED
- Checked: menu_bar.rs:743-744 add_enabled(false, ...); `git grep show_parametric` over the whole worktree: menu_bar.rs:743, mod.rs:817, state/mod.rs:297,648,772,791, preferences.rs:58,85 only; no other writer than the preferences load (default false).
- Safe to apply without judgment: no (owner decision)

## LGS-08: Grid "Spacing" setting and `auto_grid_spacing` overwritten on first canvas frame
- Verdict: CONFIRMED
- Checked: canvas/mod.rs:43 assigns spacing_m unconditionally every frame; menu_bar.rs:808-820 DragValue range 0.001..=100.0 with tooltip "Set to 0 for auto-spacing"; grid.rs zoom_spacing returns a clean step from CLEAN_SPACINGS; auto_grid_spacing called at mod.rs:1109, file_io.rs:531; state/mod.rs:765 `grid: self.grid` in preferences and tick_save_user_prefs writes when `current != last_saved_prefs`.
- Safe to apply without judgment: no (owner decision)

## LGS-09: Eight-plus copies of the "generate text, download, report outcome" menu block
- Verdict: CONFIRMED
- Checked: menu_bar.rs download_text calls at 253, 325, 357, 398, 426, 465, 494, 533, 567 and download_bytes at 602, 645; read 310-345 shows the Ok/Err shape with error_log.push + show_error_panel = true.
- Safe to apply without judgment: no (refactor)

## LGS-10: Native open/save/recover failures are logged but never shown to the user
- Verdict: CONFIRMED
- Checked: menu_bar.rs:139-141, 191-193, 203-205, 211-213, 226-228; mod.rs:70-72, 1024-1026, 1068-1070, 1261-1263, 1269-1271, 1282-1284 all log::error! only. save_to_file (file_io.rs:85-95) sets a status only on success.
- Safe to apply without judgment: no

## LGS-11: Native autosave is never cleaned up, sibling autosaves never offered, New brings the recovery prompt back
- Verdict: CONFIRMED (by code reading, not run)
- Checked: file_io.rs:127-138 autosave_path (sibling once last_save_path set, else temp); check_autosave_recovery (252-269) only looks at temp_dir/linkage_simulator_autosave.json; remove_file in src/gui only in export/csv.rs, export/download.rs (tests) and mod.rs:1028,1032; save_to_file deletes nothing; new_empty_mechanism (mod.rs:939-972) does `AppState::default()` (mod.rs:686 runs check_autosave_recovery) and clears only the wasm autosave.
- Safe to apply without judgment: no

## LGS-12: A failed load leaves the app half-updated
- Verdict: CONFIRMED
- Checked: file_io.rs:302-358 assign sweep_mode, trajectory_severity, sensor_config, motion_profile, actuator_rated_force, simulation_duration, parametric_config and `self.blueprint = Some(json_struct)` before the fallible load_mechanism_unbuilt / build `?` at 360-362.
- Safe to apply without judgment: no

## LGS-13: Dead branches and copies inside `load_from_json_str`
- Verdict: CONFIRMED
- Checked: file_io.rs:365-375 four arms (Expression, LinearLength|CosineStroke, None, and the no-driver else) all give (2PI, 0); `self.blueprint` is assigned once (358) and never cleared before 455 and 505, so the `else` at 478-480 and 520-525 are unreachable.
- Safe to apply without judgment: no (the else-branch removal needs a restructure that cannot be compile-checked here)
- Exact change (collapsing the identical arms is behaviour-neutral if wanted alone): file_io.rs lines 368-371, replace the three arms `Some(DriverMeta::Expression { .. }) => (2.0 * std::f64::consts::PI, 0.0),` / `Some(DriverMeta::LinearLength { .. }) | Some(DriverMeta::CosineStroke { .. }) => (2.0 * std::f64::consts::PI, 0.0),` / `None => (2.0 * std::f64::consts::PI, 0.0),` with `_ => (2.0 * std::f64::consts::PI, 0.0),`.

## LGS-14: Template and custom-sample tests write into the developer's real ~/.linkage-sim/
- Verdict: CONFIRMED
- Checked: templates.rs:96-112 persist_template writes dirs::home_dir()/.linkage-sim/templates/<name>.json under `#[cfg(feature = "native")]` (default feature, Cargo.toml `default = ["native"]`); save_as_template (16-31) calls it unconditionally; test_state (406-411) only clears the in-memory lists; tests at 421-465 save "test", "roundtrip", "a", "b", "only"; delete_template(5) is a no-op so "only" stays; no cfg(test) gate on persistence.
- Safe to apply without judgment: no

## LGS-15: Templates and custom samples are two full copies of the same persistence and CRUD code
- Verdict: CONFIRMED
- Checked: templates.rs fn list (16-182 vs 212-382) is the same set of functions per list; mod.rs:1091-1139 the two name dialogs use template_name_buf / custom_sample_name_buf; sanitize_filename (388-398) maps every non-alphanumeric to `_` so "A:B" and "A?B" collide.
- Safe to apply without judgment: no

## LGS-16: Undo/redo restore skips refreshes that rebuild() and load do
- Verdict: CONFIRMED
- Checked: undo_ops.rs:35-92 ends compute_forces, update_grashof, compute_validation, mark_sweep_dirty with no recompute_driver_display_offset and no stroke-range extraction; `git grep recompute_driver_display_offset` hits are blueprint_ops.rs:403, driver_ops.rs:50, file_io.rs:537, mod.rs:1114 (not undo_ops). Linear stroke is `ld.length_0` (undo_ops.rs:70), prior-stroke kept in rebuild (blueprint_ops.rs:310-313) and load (file_io.rs:458-463).
- Safe to apply without judgment: no

## LGS-17: Removing a body leaves linear drivers (and force point refs) that point at it
- Verdict: CONFIRMED (linear driver part); the force-point-ref sentence for remove_attachment_point was not checked
- Checked: entity_crud.rs:277-303 retains joints, drivers, forces only; remove_attachment_point (243-252) retains joints only; no linear_drivers retain anywhere in gui/state (grep); core/mechanism.rs add_linear_driver calls validate_body_exists on both bodies and io/from_json.rs propagates the error, so rebuild fails with a dangling body.
- Safe to apply without judgment: no

## LGS-18: The expression-driver editor applies on every keystroke and again on focus loss
- Verdict: CONFIRMED
- Checked: input_panel.rs:720-765 each box sets changed on `lost_focus() || changed()`, then set_expression_driver (driver_ops.rs:128-154) calls push_undo() then rebuild(); undo.rs push (no equality check) pops the oldest past max 50 (UndoHistory::new(50) at state/mod.rs:607).
- Safe to apply without judgment: no

## LGS-19: Dead methods and variables in state and the input panel
- Verdict: CONFIRMED
- Checked: whole-worktree `git grep -w`: update_prismatic_axis only at its definition (blueprint_ops.rs:1135); move_attachment_point only tests.rs:655,718 (plus a prose mention in docs/superpowers/plans/2026-03-20-gui-bugfix-ux-rework.md:504); sync_active_load_case only tests.rs:1430; input_panel.rs:673 `current_theta_0` read only by `let _ = current_theta_0;` at 694.
- Safe to apply without judgment: yes for update_prismatic_axis and the dead local (unused on every target and feature, no test, script or doc uses them); no for the other two methods
- Exact change: (1) linkage-sim-rs/src/gui/state/blueprint_ops.rs lines 1132-1144: delete the doc comment, the fn `update_prismatic_axis` (1135-1143) and the blank line after it. (2) linkage-sim-rs/src/gui/input_panel.rs: delete line 673 `        let current_theta_0 = state.driver_theta_0();` and line 694 `            let _ = current_theta_0;`.

## LGS-20: Repeated blocks and per-frame work in `update()`
- Verdict: CONFIRMED (minor: the two demo blocks differ by one line, the advance copy at mod.rs:282 also sets animation_speed_deg_per_sec = 90.0)
- Checked: mod.rs:262-285 vs 882-902 same sequence; 83-85 apply_nathan_mode per frame and theme.rs apply_nathan_mode starts from Visuals::dark() while restore_normal_visuals sets cad_dark_visuals(); 88 window_title per frame; 103 dropped_files clone; `git grep -c "status_message_time = "` gives 3+19+8+15+1+4+6 = 56; mod.rs:404 tooltip says "+ Link" while the button is "Draw Link" (mod.rs:346).
- Safe to apply without judgment: no

## Seen while checking
- (none)