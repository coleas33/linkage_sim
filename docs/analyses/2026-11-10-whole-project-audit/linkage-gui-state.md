# Audit findings: linkage-gui-state
Scope read: gui/mod.rs, menu_bar.rs, input_panel.rs, theme.rs, error_panel.rs, undo.rs and gui/state/{mod,blueprint_ops,file_io,driver_ops,entity_crud,templates,undo_ops,solver_helpers,trajectory_ops,preferences,grid,simulation}.rs plus the first 330 lines of types.rs

## LGS-01: Drag-and-drop file import only reads `file.bytes`, which native eframe never fills
- Kind: broken
- Where: linkage-sim-rs/src/gui/mod.rs:103-105 (loop), :905 (welcome text "Or drag & drop a JSON file to open"), :128/:161/:193/:213 (status texts "(drag & drop)")
- Evidence: `for file in &dropped_files { if let Some(bytes) = &file.bytes {` ... is the only use; `git grep "file.path\|\.path"` finds no branch that reads `DroppedFile::path`. The comment at :99 says "works on native + WASM".
- Why it matters: egui's native (winit) backend delivers dropped files with `path` set and `bytes` None, so on the desktop build a dropped JSON/DXF/CSV/image is silently ignored while the welcome screen invites the user to drop a file.
- Suggested action: when `bytes` is None and `path` is Some (native only), read the file with `std::fs::read` and feed the same dispatch.
- Fix class: needs-judgment
- Confidence: likely (egui-winit source is not in the local cargo registry; reading the egui `DroppedFile` docs or dropping a file on the native build settles it)

## LGS-02: Autosave recovery deletes the autosave file even when loading it failed
- Kind: broken
- Where: linkage-sim-rs/src/gui/mod.rs:1022-1029 (native), :1065-1073 (wasm)
- Evidence: `if let Err(e) = self.state.load_from_file(&path) { log::error!(...) }  let _ = std::fs::remove_file(&path);` and on wasm `if let Err(e) = ...load_from_json_str(&json_str) { log::error!(..) }  AppState::wasm_clear_autosave();`
- Why it matters: if the recovered file is corrupt or fails to parse, the user sees nothing (only a log line, no status message or error_log entry) and the only copy of the unsaved work is deleted.
- Suggested action: delete/clear only on Ok; on Err push to `error_log`/`status_message` and keep the file.
- Fix class: needs-judgment
- Confidence: certain

## LGS-03: Shortcut help lists Ctrl+S and Ctrl+Shift+S on the web build, where no handler exists
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/mod.rs:1166-1167 (help rows), :1258 and :1275 (handlers gated `#[cfg(feature = "native")]`)
- Evidence: `("Ctrl+S", "Save (quick save to last path)"), ("Ctrl+Shift+S", "Save As (choose new path)")` are in the unconditional `shortcuts` array; `git grep "Key::S"` over src/gui finds only the two native-gated handlers.
- Why it matters: web users are told a save shortcut exists; pressing it does nothing. Also a failed quick save only reaches `log::error!` (:1262, :1270, :1283), never the status bar, unlike the menu save.
- Suggested action: filter the two rows with `cfg!(feature = "native")` (or wire a web download) and report save failures through `status_message`.
- Fix class: needs-judgment
- Confidence: certain for the missing handler; likely for the menu path reporting differently (menu_bar.rs not yet compared)

## LGS-04: The driver-angle to time mapping is written out five times in scope, and one copy lacks the zero-omega guard the others have
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/state/mod.rs:1197-1201 (solve_at_angle), :1300 (flip_assembly_branch, unguarded), :1341-1345 (solve_at_stroke); gui/state/blueprint_ops.rs:345-350 (rebuild), :561-566 (recompute_dynamics); gui/state/undo_ops.rs:47-51 (restore_snapshot), gui/state/file_io.rs:408 and :423 (load, no guard at all)
- Evidence: four copies read `let t = if omega.abs() > f64::EPSILON { (x - self.driver_theta_0()) / omega } else { 0.0 };`; flip_assembly_branch reads `let t = (self.driver_angle - self.driver_theta_0()) / self.driver_omega();`. The comment at mod.rs:1191-1195 says "Other solver callsites ... use the same pattern".
- Why it matters: repeated logic (owner flags repetition); the unguarded copy gives t = inf/NaN if omega is 0 (the clamp to MIN_DRIVER_OMEGA_ABS makes this unlikely, so it is defence in depth that one site skipped). Out-of-scope copies also exist: gui/sweep/mod.rs:283-288,1520; property_panel/equations.rs:164; export/raster.rs:127,424; samples/mod.rs:352; samples/special.rs:1059.
- Suggested action: one `AppState::time_at_driver_value(&self, value: f64) -> f64` helper used by all sites.
- Fix class: needs-judgment
- Confidence: certain

## LGS-05: Same block of code copied within state: stroke-range extraction, trajectory/sweep q seed, gravity element build
- Kind: inconsistent
- Where: (a) gui/state/mod.rs:1080-1092 (load_sample) and gui/state/blueprint_ops.rs:386-396 (rebuild); (b) blueprint_ops.rs:1345-1351 and :1393-1397 (compute_sweep); (c) blueprint_ops.rs:578-583 (sync_gravity) and gui/state/mod.rs:1396-1402 (run_simulation)
- Evidence: (a) both: `self.sweep_stroke_min = 0.0; self.sweep_stroke_max = 0.0; for force in ...forces() { if let ForceElement::LinearActuator(act) = force { if act.stroke_min > 0.0 || act.stroke_max > 0.0 { ... break; } } }`. (b) both: `if self.q_at_zero.len() == self.last_good_q.len() && self.q_at_zero.len() > 0 { self.q_at_zero.clone() } else { self.last_good_q.clone() }`. (c) both build `GravityElement { g_vector: [-g * theta.sin(), -g * theta.cos()] }`.
- Why it matters: a later change to the gravity vector rule or the stroke-range rule has to be made in two places and can drift (the gravity vector is physics-relevant).
- Suggested action: three small helpers (`refresh_stroke_range_from(mech)`, `sweep_seed_q()`, `gravity_element(g, theta)`).
- Fix class: needs-judgment
- Confidence: certain

## LGS-06: Doc comment of `solve_at_angle` is split: its first lines sit on `recompute_driver_display_offset`
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/state/mod.rs:1118-1120 and :1182-1185
- Evidence: line 1118 "/// Solve the position problem for the given driver angle (radians)." is followed by "///" and then "Recompute `driver_display_offset`..." for `recompute_driver_display_offset`; the continuation "Uses `last_good_q` as the initial guess..." sits on `solve_at_angle` at :1182 after the intervening functions.
- Why it matters: rustdoc for both functions is wrong (one has a stray first line, the other lacks its summary).
- Suggested action: move line 1118-1119 to just above :1182.
- Fix class: mechanical-safe
- Confidence: certain

## LGS-07: Menu "Parametric Study [WIP]" is a permanently disabled checkbox, so the whole parametric/counterbalance panel is unreachable
- Kind: not needed
- Where: linkage-sim-rs/src/gui/menu_bar.rs:743-744 (disabled checkbox); gui/mod.rs:817-829 (panel only drawn when `show_parametric`); gui/state/mod.rs:296-308,648-670 (show_parametric, parametric_config, saved_parametric_sweeps, saved_parametric_counter, counterbalance_*); gui/state/preferences.rs:58,85 (persisted pref); gui/state/parametric.rs and driver_ops.rs counterbalance runner; gui/parametric_panel.rs
- Evidence: `ui.add_enabled(false, egui::Checkbox::new(&mut state.show_parametric, "Parametric Study [WIP]")).on_hover_text("Parametric study panel (coming soon)")`. `git grep show_parametric` finds no other writer than preferences load (default false) and this disabled checkbox.
- Why it matters: a finished-looking feature (panel, sweeps, counterbalance study, tests) cannot be opened by any user; only a hand-edited preferences file can turn it on. Its state fields and the per-frame pref compare carry it for nothing.
- Suggested action: owner decides: enable the menu entry, or remove the panel and its state.
- Fix class: owner-decision
- Confidence: certain that nothing enables it; whether the owner wants it back is theirs to say

## LGS-08: The Grid "Spacing" setting and `auto_grid_spacing` are overwritten on the first canvas frame; zoom also rewrites the preferences file
- Kind: broken
- Where: linkage-sim-rs/src/gui/canvas/mod.rs:43 (`state.grid.spacing_m = state.grid.zoom_spacing(...)` every frame); gui/menu_bar.rs:804-821 (Spacing DragValue, hover text "Set to 0 for auto-spacing based on zoom level"); gui/state/blueprint_ops.rs:798-837 (`auto_grid_spacing`, called from mod.rs:1109 and file_io.rs:531); gui/state/mod.rs:762-777 and :802-808 (grid is part of the persisted preferences)
- Evidence: the canvas assigns `spacing_m` unconditionally each frame; the menu DragValue has `.range(0.001..=100.0)` so 0 cannot be entered, yet the tooltip says "Set to 0 for auto-spacing". `grid.spacing_m` is inside `UserPreferences` (`grid: self.grid`), and `tick_save_user_prefs` writes when `current != last_saved_prefs`.
- Why it matters: typing a grid spacing, and the sample/file-load auto spacing, have no lasting effect; the tooltip describes a mode that does not exist; each zoom step that changes the clean spacing writes the preferences sidecar to disk/localStorage from the render loop.
- Suggested action: owner picks: drop the manual Spacing control and `auto_grid_spacing` (zoom spacing is the only mode), or let the canvas honour a user value; keep `spacing_m` out of persisted prefs either way.
- Fix class: owner-decision
- Confidence: certain for the overwrite (canvas/mod.rs:43 has no condition); the disk-write rate is likely (depends on how often the clean step changes)

## LGS-09: Eight-plus copies of the "generate text, download, report outcome" menu block
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/menu_bar.rs:310-345, 346-377, 386-417, 418-446, 447-485, 486-514, 515-553, 554-584 (and the png/gif pair :589-665 with `download_bytes`, wasm JSON download :236-272)
- Evidence: each is `match export::generate_X(..) { Ok(s) => { let outcome = export::download::download_text("name", "mime", &s, FileFilter { label, extensions: &[..] }); apply_download_outcome(state, outcome); } Err(e) => { state.error_log.push(format!("X failed: {}", e)); state.show_error_panel = true; } }`
- Why it matters: about 300 of the file's 874 `draw_menu_bar` lines (the longest-function candidate in leads.md) are the same shape; the error path is hand-copied each time (the `error_log.push + show_error_panel = true` pair also appears in mod.rs and state files).
- Suggested action: a helper `export_text(state, result, filename, mime, label, ext)` and an `AppState::report_error(msg)`; table-drive the menu entries.
- Fix class: needs-judgment
- Confidence: certain

## LGS-10: Native open/save/recover failures are logged but never shown to the user
- Kind: broken
- Where: linkage-sim-rs/src/gui/menu_bar.rs:139-141 (Open JSON), :191-193 (Recent Files), :203-205 and :211-213 (Save), :226-228 (Save As); gui/mod.rs:1261-1263, :1269-1271, :1282-1284 (Ctrl+S, Ctrl+Shift+S), :70-72 (shared URL load), :1024-1026, :1068-1070 (recovery)
- Evidence: `if let Err(e) = state.save_to_file(&path) { log::error!("Save failed: {}", e); }` with no `status_message` or `error_log` entry; the sibling paths in the same menu (DXF import :164-168, share URL :119-123, every export) do report to the status bar or error panel.
- Why it matters: a failed save (read-only folder, disk full) looks like a successful one; the window title's `*` stays but nothing says why.
- Suggested action: route these errors through `status_message`/`error_log` the way the DXF and export paths do (one shared helper).
- Fix class: needs-judgment
- Confidence: certain for these call sites; save_to_file itself may set a status on success (file_io.rs not yet read)

## LGS-11: Native autosave is never cleaned up, sibling autosaves are never offered back, and New brings the recovery prompt back
- Kind: broken
- Where: linkage-sim-rs/src/gui/state/file_io.rs:101-138 (tick_autosave, autosave_path), :249-269 (check_autosave_recovery); gui/state/mod.rs:939-972 (new_empty_mechanism), :686 (recovery_path set in Default)
- Evidence: `autosave_path()` returns `.<stem>.autosave.json` next to the saved file once `last_save_path` is set, but `check_autosave_recovery()` only looks at `temp_dir()/linkage_simulator_autosave.json`. `git grep remove_file` over src finds deletes only in the recovery prompt (mod.rs:1028,1032); `save_to_file` (file_io.rs:85-95) and `new_empty_mechanism` delete nothing on native. `new_empty_mechanism` does `let fresh = AppState::default()` (which runs `check_autosave_recovery()`) and then `*self = fresh`, clearing only the WASM autosave (mod.rs:954-958).
- Why it matters: (1) a crash while editing a saved file leaves `.name.autosave.json` that is never offered for recovery and never removed (hidden files pile up beside the user's files); (2) a temp autosave written before the first save survives that save, and the next launch within an hour offers stale work; (3) File > New (or Ctrl+N) after 30 s of unsaved work makes the "Recover Unsaved Work?" window pop up again in the same session.
- Suggested action: delete the autosave in `save_to_file` and `new_empty_mechanism`; make recovery look at the same path function (or both locations).
- Fix class: needs-judgment
- Confidence: likely for (3) and (2) (read from code, not run); certain for (1)

## LGS-12: A failed load leaves the app half-updated (blueprint and settings replaced, mechanism not)
- Kind: broken
- Where: linkage-sim-rs/src/gui/state/file_io.rs:302-358 (state written), :360-362 (fallible `load_mechanism_unbuilt` / `mech.build()?`)
- Evidence: `self.sweep_mode = mode; self.trajectory_severity = ...; self.sensor_config = ...; self.motion_profile = mp; ...; self.blueprint = Some(json_struct);` all run before `let mut mech = load_mechanism_unbuilt(json_str).map_err(..)?; mech.build().map_err(..)?;`.
- Why it matters: a file that parses as `MechanismJson` but fails to build (bad joint reference, dependent constraints) returns Err yet already replaced `blueprint`, sweep mode, sensors, motion profile and rated force; the old `mechanism` and canvas stay, the next edit calls `rebuild()` from the broken blueprint and the user's working mechanism is lost. Same path serves share links, drops, recent files and recovery.
- Suggested action: build and validate the mechanism first (or snapshot and restore the touched fields on Err), then commit state.
- Fix class: needs-judgment
- Confidence: likely (read from code; a test loading a JSON that builds badly and checking `blueprint` would settle it)

## LGS-13: Dead branches and copies inside `load_from_json_str`
- Kind: not needed
- Where: linkage-sim-rs/src/gui/state/file_io.rs:365-375 (driver tuple), :455-480 (driver_kind), :483-493 (stroke range, third copy, see LGS-05), :500-525 (load cases)
- Evidence: `match driver.meta() { Some(ConstantSpeed{..}) => (*omega,*theta_0), Some(Expression{..}) => (2PI,0), Some(LinearLength{..}) | Some(CosineStroke{..}) => (2PI,0), None => (2PI,0) }` has four arms returning the same value. `self.blueprint = Some(json_struct)` at :358 means every later `if let Some(ref bp) = self.blueprint { .. } else { .. }` (:455, :500-525) has an unreachable else (`DriverKind::None`, a second `LoadCaseManager::new_default` copy).
- Why it matters: about 30 lines that can never run; the duplicate default-load-case branch repeats the one inside.
- Suggested action: collapse the match arms; read `json_struct`'s fields before moving it into `self.blueprint` and drop the else branches.
- Fix class: mechanical-safe
- Confidence: certain

## LGS-14: Template and custom-sample tests write into the developer's real `~/.linkage-sim/` and leave files behind
- Kind: broken
- Where: linkage-sim-rs/src/gui/state/templates.rs:404-411 (`test_state`), :420-465 (template tests), :477-520 (custom-sample tests); persistence at :95-120, :281-306; also every `AppState::default()` in tests (state/mod.rs:695-698, :735 `UserPreferences::load()`, :682 `load_recent_files`)
- Evidence: `test_state()` only runs `state.saved_templates.clear(); state.custom_samples.clear();` ("ignoring any entries persisted on disk"), then `state.save_as_template("test"|"roundtrip"|"a"|"b"|"only")` calls `persist_template`, which does `std::fs::write(dirs::home_dir()/.linkage-sim/templates/<name>.json)`; `custom_samples` does the same under `custom_samples/`. `delete_template(0)` removes only "a"; `delete_template(5)` (out of range) removes nothing, so "only", "b", "test", "roundtrip" stay on disk.
- Why it matters: every `cargo test` adds templates and custom samples named test, roundtrip, b and only to the owner's real File > Load Template / My Samples menus, and overwrites or deletes a real template that happens to share one of those names ("a", "b", "test"). Tests that call `AppState::default()` also read the real preferences file (display units, grid snap, dismiss_welcome), so results can depend on the machine.
- Suggested action: give the persistence functions a base directory the tests can point at a temp dir (or gate persistence off under `cfg(test)`), and make `AppState::default()` not read user files in tests.
- Fix class: needs-judgment
- Confidence: certain for the writes (the code paths are unconditional on native); the files on this machine can be listed with `dir ~/.linkage-sim/templates`

## LGS-15: Templates and custom samples are two full copies of the same persistence and CRUD code
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/state/templates.rs:12-203 (templates) versus :207-383 (custom samples); menu_bar.rs:56-71 and mod.rs:1082-1151 (the two name dialogs)
- Evidence: `save_as_template`/`save_as_custom_sample`, `load_template`/`load_custom_sample`, `delete_*`, `load_saved_*`, `*_dir`, `persist_*`, `remove_persisted_*`, `load_*_native`, `load_*_wasm` differ only in the list field, the directory name (`templates` vs `custom_samples`), the localStorage prefix (`linkage_template_` vs `linkage_csample_`) and the status wording; the two "Save as ..." dialogs in `update` are line-for-line the same with `template_name_buf`/`custom_sample_name_buf`.
- Why it matters: about 300 duplicated lines; a fix to one (for example the filename sanitizing, where two names such as "A:B" and "A?B" map to one file and the reloaded name differs from the saved one) must be repeated.
- Suggested action: one `NamedStore { dir, prefix }` type with save/load/delete, used for both lists; one `name_dialog()` helper.
- Fix class: needs-judgment
- Confidence: certain

## LGS-16: Undo/redo restore skips refreshes that rebuild() and load do, and reconstructs `driver_kind` a third way
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/state/undo_ops.rs:35-92 (restore_snapshot) versus gui/state/blueprint_ops.rs:301-340,400-404 (rebuild) and file_io.rs:455-480 (load)
- Evidence: rebuild and load end with `recompute_driver_display_offset()` (the field doc at state/mod.rs:170 says "Recomputed on every rebuild") and re-extract `sweep_stroke_min/max`; `restore_snapshot` ends `compute_forces; update_grashof; compute_validation; mark_sweep_dirty` only. Linear stroke is `ld.length_0` here, "keep a finite prior stroke" in rebuild, "keep prior if nonzero" in load.
- Why it matters: after undoing a driver-body edit or a driver reassignment, `driver_display_offset` (used by the slider, plot axes and the animation bounds) and the stroke range stay at the pre-undo mechanism's values until some later rebuild; three paths for one job give three behaviours.
- Suggested action: have restore_snapshot call the same tail as rebuild (offset, stroke range) and share one `detect_driver_kind(blueprint, prior)` helper.
- Fix class: needs-judgment
- Confidence: likely (offset staleness read from code; an undo test on a driver-body move would settle it)


## LGS-17: Removing a body leaves linear drivers (and force point refs) that point at it
- Kind: broken
- Where: linkage-sim-rs/src/gui/state/entity_crud.rs:277-303 (remove_body), :243-252 (remove_attachment_point)
- Evidence: `remove_body` clears `bp.joints`, `bp.drivers` and `bp.forces` that touch the body but never `bp.linear_drivers`, although `LinearDriverJson` carries `body_a`/`body_b` (driver_ops.rs:274-282). `git grep linear_drivers` in gui/state shows no cascade anywhere in entity_crud.rs. The comment in remove_body explains that a dangling force "would freeze the solver at the next rebuild"; the same holds for a dangling linear driver. `remove_attachment_point` cascades only joints, not linear drivers or force `point_*` references to that point.
- Why it matters: deleting a body that hosts an actuator-converted linear driver makes `rebuild()` fail (`log::warn!("Blueprint rebuild failed")`, solver status FAIL) while the old mechanism stays on screen; the user must undo.
- Suggested action: retain-filter `linear_drivers` by body in remove_body, and by (body, point) in remove_attachment_point; add a test beside the existing remove_body tests.
- Fix class: needs-judgment
- Confidence: likely (read from code; a test that converts an actuator, then removes its body and checks `solver_status` would settle it)

## LGS-18: The expression-driver editor applies (undo snapshot + rebuild) on every keystroke and again on focus loss
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/input_panel.rs:720-765 (`if response.lost_focus() || response.changed() { changed = true; }`), gui/state/driver_ops.rs:128-154 (`set_expression_driver`: `push_undo()` then `rebuild()`)
- Evidence: each of the three text boxes sets `changed` on `changed()` and on `lost_focus()`; when parsing succeeds the code calls `state.set_expression_driver(...)` which calls `self.push_undo()` (full mechanism JSON snapshot) and `self.rebuild()`. Pressing Enter or Tab after typing fires a second identical apply.
- Why it matters: typing `2*pi*t` makes about 6 undo entries and 6 rebuilds; the undo stack is capped at 50 (`UndoHistory::new(50)`), so editing an expression a few times evicts the user's real edit history, and the redundant apply on focus loss repeats an identical snapshot.
- Suggested action: apply on `lost_focus()` only (or debounce), and skip the push when the three strings equal the stored driver.
- Fix class: needs-judgment
- Confidence: certain for the call chain; the undo flood follows from `push_undo` having no equality check (undo_ops.rs:97-102)

## LGS-19: Dead methods and variables in state and the input panel
- Kind: not needed
- Where: linkage-sim-rs/src/gui/state/blueprint_ops.rs:1132-1143 (`update_prismatic_axis`); gui/state/blueprint_ops.rs:409-425 (`move_attachment_point`); gui/state/driver_ops.rs:351-367 (`sync_active_load_case`); gui/input_panel.rs:673 and :694 (`current_theta_0`)
- Evidence: `git grep -w update_prismatic_axis` finds only its definition (no test either); `move_attachment_point` is called only from state/tests.rs:655,718; `sync_active_load_case` only from state/tests.rs:1424-1430 (its doc says "Called when the user changes driver settings", but nothing in src/ calls it); input_panel.rs:673 `let current_theta_0 = state.driver_theta_0();` is used only by `let _ = current_theta_0;` at :694.
- Why it matters: untested-in-production API surface that suggests behaviour (load case staying in sync with driver edits) that does not happen: `set_constant_speed_driver` (the omega editor path) never updates the active load case, so a later `apply_load_case` can restore stale omega.
- Suggested action: delete `update_prismatic_axis` and the dead local; decide whether `sync_active_load_case` should be wired into the omega editor or removed with its test, and whether `move_attachment_point` stays as a test-only helper.
- Fix class: needs-judgment (the two tested methods); mechanical-safe (`update_prismatic_axis`, the dead local)
- Confidence: certain for unused; likely for the stale-omega consequence (load case sync read from code only)

## LGS-20: Repeated blocks and per-frame work in `update()` (leads.md: 1130-line function)
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/mod.rs:262-285 and :882-902 (demo start/advance: the same ten assignments); :83-85 (`apply_nathan_mode` builds `Visuals::dark()` and calls `set_visuals` every frame while Nathan Mode is on); :88 (`window_title` formats a String every frame before `send_title` compares it); :103 (`dropped_files.clone()` every frame); `status_message = Some(..); status_message_time = N` written out 56 times in src/gui (mod.rs 15, menu_bar.rs 8, templates.rs 6, state/mod.rs 4, dxf_import.rs 19, canvas/interaction.rs 3, file_io.rs 1)
- Evidence: demo: `load_sample(sample); driver_angle = 0.0; solve_at_angle(0.0); last_good_q = q.clone(); q_at_zero = q.clone(); playing = true; loop_mode = true; animation_direction = 1.0; pending_fit_to_view = true` appears at both sites. `apply_nathan_mode` also starts from `Visuals::dark()` rather than `cad_dark_visuals()`, so Nathan Mode on drops the CAD panel colours that `restore_normal_visuals` puts back.
- Why it matters: small per-frame allocations while idle repaint (egui repaints on input anyway), and the 56-site status pattern is the clearest repetition in the part; a `set_status(msg, secs)` helper would also let the toast logic live in one place. Also the toolbar's "+ Body" tooltip (mod.rs:404) says "Draw links with + Link" while the button is labelled "Draw Link".
- Suggested action: `AppState::set_status`, one `start_demo_sample()`, apply Nathan visuals only on change of the flag; fix the tooltip wording.
- Fix class: needs-judgment
- Confidence: certain for the copies; the per-frame cost is small (likely not measurable)

## Leads settled
- blueprint_ops.rs `update_prismatic_axis` (line 1135): real. Defined once, no caller in src, tests, web or docs (LGS-19).
- 19 duplicated lines gui/theme.rs vs magcoupling-rs/src/app/theme.rs:13-43: not real. magcoupling's theme.rs header says it is a deliberate copy and a test compares the two function bodies.
- `update` at gui/mod.rs:81, 1130 lines: real as a size problem; the repeated blocks inside it are LGS-02, LGS-15, LGS-20 (no further defect from length alone).
- `draw_menu_bar` at gui/menu_bar.rs:27, 874 lines: real; about a third is the repeated export block (LGS-09), plus the dead Parametric entry (LGS-07).
- `load_from_json_str` at gui/state/file_io.rs:290, 264 lines: real; defects found are LGS-12 and LGS-13.
- `draw_input_panel` at gui/input_panel.rs:10, 253 lines: not a defect by itself; most of it is collapsing-header glue that delegates to helpers (the real issues in the file are LGS-18 and LGS-19).
- `gui/state/tests.rs` 3633 lines: not a defect (a test file); I only searched it, did not read it.
- Baseline-noted `MAGCOUPLER_PATH` never used (native): not dead; it is used by `MAGCOUPLER_URL` on wasm32 (menu_bar.rs:22-25) and by a test.

## Not read
- linkage-sim-rs/src/gui/state/display_units.rs, load_cases.rs, parametric.rs, view_transform.rs (grep-checked for unused public functions only)
- linkage-sim-rs/src/gui/test_support.rs
- linkage-sim-rs/src/gui/state/tests.rs (searched, not read); the in-file test modules of mod.rs, menu_bar.rs, undo.rs, file_io.rs and templates.rs were skimmed, not checked assertion by assertion

