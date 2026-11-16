# Audit findings: linkage-gui-canvas-panels
Scope read: canvas/ (mod, rendering/*, interaction in part), eq_rendering, plot_panel/actuator, pending_edits header, profile_input csv path; rest skimmed by search (see Not read)

## LGC-01: Two dead functions with misleading names (driver_lambda_units, draw_actuator_stroke_annotation)
- Kind: not needed
- Where: linkage-sim-rs/src/gui/eq_rendering.rs:65 ; linkage-sim-rs/src/gui/plot_panel/actuator.rs:186-234
- Evidence: `git grep driver_lambda_units draw_actuator_stroke_annotation` over the whole worktree returns only the two definitions (no cfg, test, doc or JS user). The actuator one is named "stroke annotation" but prints peak/RMS force, and has a stray blank line between its doc comment and `fn` (a removed attribute). `eq_rendering.rs:64` doc points at `lambda_units_at_row`, which exists nowhere in the worktree.
- Why it matters: dead code with a name that disagrees with what it does; the peak/RMS text is never shown on the actuator plot.
- Suggested action: delete both (or wire the annotation into draw_actuator_force if the owner wants the text).
- Fix class: mechanical-safe
- Confidence: certain (searched whole worktree, both targets)

## LGC-02: Stale #[allow(dead_code)] attributes on live items
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/trajectory_panel/profile_input.rs:365-366 ; linkage-sim-rs/src/gui/canvas/hit_testing.rs:20
- Evidence: profile_input.rs:365 `#[allow(dead_code)] // used from the native-feature button path` sits on `parse_keyframes_csv`, which is called at profile_input.rs:274 under the same `native` feature, so the allow hides nothing and the comment says it is used. hit_testing.rs:20 `#[allow(dead_code)] pub struct SegmentHit` is built and returned at hit_testing.rs:63-97.
- Why it matters: blanket allows hide real future dead-code warnings in these files.
- Suggested action: remove both attributes; build once natively and for wasm32 to see if any field warns, then drop only the unread field.
- Fix class: needs-judgment (cannot build here; fields of SegmentHit may be unread)
- Confidence: likely; a native + wasm cargo check without the attribute settles it

## LGC-03: Coupler-trace dashes re-implement draw_dashed_line, and the dashed force-zone rectangle is written twice
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/canvas/rendering/mod.rs:119-141 ; linkage-sim-rs/src/gui/canvas/rendering/force_render.rs:380-398 ; linkage-sim-rs/src/gui/canvas/interaction.rs:1053-1078
- Evidence: mod.rs:122-140 hand-codes dash stepping (`let dash_len = ...; while t < seg_len {... painter.line_segment`) with the same loop as `draw_dashed_line` (primitives.rs:285-307). force_render.rs and interaction.rs both compute min/max corners tl/tr/br/bl and call draw_dashed_line four times with identical dash=6, gap=4, FORCE_ZONE_COLOR.
- Why it matters: three copies of the same drawing code (owner flags repetition); the trace loop also restarts the dash phase on every sweep segment, so a dense sweep gets dotted instead of dashed and issues many tiny line_segment calls every frame.
- Suggested action: add `draw_dashed_rect(painter, rect, stroke, dash, gap)` in primitives.rs and use it in both zone sites; draw each coupler trace as one polyline via a dashed-polyline helper (egui has Shape::dashed_line) built once rather than per segment.
- Fix class: needs-judgment (visual output of traces changes slightly)
- Confidence: certain for the duplication

## LGC-04: Arrow drawing duplicated; the "one arrowhead" helper is bypassed
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/canvas/rendering/primitives.rs:127-161 (draw_external_force_arrow), 390-418 (draw_force_arrow), 449-494 (draw_axis_component), 167-211 (draw_torque_arc hand-rolled head); linkage-sim-rs/src/gui/canvas/rendering/mod.rs:716-731 (gravity indicator head)
- Evidence: primitives.rs:354-362 says draw_arrowhead is "The one arrowhead of every straight canvas arrow". draw_force_arrow and draw_external_force_arrow repeat the same mag/px_len clamp/unit-vector/label steps and differ only in colour and tail-vs-tip anchoring. mod.rs:716-731 draws the gravity arrowhead with two raw line_segment calls; draw_torque_arc:204-211 builds its own head with trig.
- Why it matters: head size/angle differ between arrows, and a change to arrow style needs five edits.
- Suggested action: one `draw_scaled_force_arrow(painter, origin, f, color, anchor)` for the three force arrows; use draw_arrow/draw_arrowhead for the gravity indicator.
- Fix class: needs-judgment
- Confidence: certain

## LGC-05: Whole SweepData cloned every frame to shift two vectors
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/plot_panel/mod.rs:265-268, 384-396
- Evidence: `let display_sweep = sweep_in_display_frame(sweep, offset_deg);` runs every frame draw_plot_panel runs; the function does `sweep.clone()` (every curve, coupler traces, joint reaction series, weight breakdown) then adds `offset_deg` to `angles_deg` and `toggle_angles` only. The doc comment calls the full clone "cheap enough for a per-frame panel redraw".
- Why it matters: the plot panel repaints continuously during playback; a 360-point sweep with many series is copied about 60 times a second to change two Vec<f64>.
- Suggested action: leave SweepData alone and add the offset in sweep_x_to_display (it already is the single x-mapping point), or cache the shifted copy keyed on (sweep version, offset).
- Fix class: needs-judgment
- Confidence: certain for the clone; the cost size is unmeasured

## LGC-06: Fourteen copies of the plot trailer (cursor line, toggle markers, range markers, click) and the per-plot Plot builder
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/plot_panel/actuator.rs (3), coupler.rs (3), dynamics.rs (3), mechanics.rs (4), weights.rs (1); also the Plot::new chain `.allow_zoom(true).allow_drag(true).x_axis_label(..).y_axis_label(..).legend(..Corner::LeftTop).height(ui.available_height().max(50.0))` in each
- Evidence: `git grep -c "from_rgba_premultiplied(255, 255, 255, 100)"` gives 3+3+3+4+1 = 14 identical `plot_ui.vline(VLine::new("cursor", current_driver_display).color(...).width(1.0)); draw_toggle_markers(..); draw_range_boundary_markers(..); clicked_x = detect_plot_click(plot_ui);` blocks. The nathan-mode recolour `if nathan_mode { to_grayscale(Color32::from_rgb(100,200,255)) } else {..}` is hand-written 4 more times (actuator.rs:84, :104, :366; dynamics.rs:130) although draw_angle_series_with_range already does it internally.
- Why it matters: owner flags repetition; any change to the cursor style or click rule needs 14 edits, and the four nathan blocks can drift from the helper.
- Suggested action: add `finish_sweep_plot(plot_ui, sweep, units, cursor_x) -> Option<f64>` and a `sweep_plot(ui, id, y_label, sweep, units)` builder in plot_panel/mod.rs; give draw_angle_series_with_range a dashed-overlay sibling so callers stop recolouring by hand.
- Fix class: needs-judgment (touches every plot; gui smoke needed)
- Confidence: certain

## LGC-07: Two tabs both labelled "Actuator Force" in stroke mode
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/plot_panel/mod.rs:109-125 and 185-193
- Evidence: `let torque_tab_label = if sweep.sweep_mode.is_stroke() { "Actuator Force" } else { "Driver Torque" };` (tab DriverTorque) and a separate tab `PlotTab::ActuatorForce` labelled "Actuator Force". The two have different hover texts and different data (driver_torques as force, versus the LinearActuator element force).
- Why it matters: with a stroke sweep that also has a LinearActuator element, two identically named tabs show different curves; the user cannot tell which is which.
- Suggested action: rename one (e.g. "Required Force" for the driver-effort tab, or "Actuator Element Force"), or hide one.
- Fix class: owner-decision (wording)
- Confidence: likely (needs a stroke-mode mechanism with a LinearActuator element to see both enabled; both `has_dt` and `has_af` conditions can be true together)

## LGC-08: Reference lines span a hard-coded 0..360 x range, wrong for stroke mode and offset-shifted angles
- Kind: broken
- Where: linkage-sim-rs/src/gui/plot_panel/mechanics.rs:102, 134-147 (transmission thresholds and zone text at x_max*0.5), 232-239 (MA = 1 line)
- Evidence: `let x_max = units.angle(2.0 * std::f64::consts::PI); let ideal_low: PlotPoints = [[0.0, 40.0], [x_max, 40.0]]`; the MA unity line does the same. The data x values go through sweep_x_to_display, and draw_plot_panel shifts angles by driver_display_offset (mod.rs:265-268), and in stroke mode x is millimetres.
- Why it matters: with a non-zero crank display offset the threshold and MA=1 lines stop at 360 while the curve extends past it; in stroke mode (MA is computed for any sweep) the line is drawn 0..360 over an x axis that is in mm, and the zone labels sit at x=180 on it.
- Suggested action: use HLine::new for the reference lines (infinite in x) and anchor the zone text to the plot bounds or the data x-range.
- Fix class: needs-judgment
- Confidence: likely (the arithmetic is as quoted; whether MA exists in stroke sweeps is not checked)

## LGC-09: Doc comments attached to the wrong function, plus stale in-body comments
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/plot_panel/mod.rs:547-551 (coupler-trace doc on draw_toggle_markers); mechanics.rs:185-188 (driver-torque doc on draw_mechanical_advantage); coupler.rs:86-88 (body-angles doc on draw_coupler_velocity); actuator.rs:29-34 (comment about a rated-force input that "we render in the caller instead" inside a function that has no such code); mod.rs:97,107 etc. say tabs are "only shown if data exists" but they are only disabled
- Evidence: e.g. mechanics.rs:185 `/// Plot driver torque (N*m) vs driver angle.` followed by a blank line and `pub(super) fn draw_mechanical_advantage`. Same blank-line-after-doc pattern at 12 sites (leftovers of removed attributes).
- Why it matters: readers and rustdoc get the wrong description; the blank line also detaches the doc from the item.
- Suggested action: fix each doc to describe its function and delete the blank lines.
- Fix class: mechanical-safe
- Confidence: certain

## LGC-10: Force editor repeats one DragValue block about 50 times in a 1,100-line function
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/property_panel/force_editor.rs:61-1170 (draw_force_element_details); the same shape of code in property_panel/mod.rs:282-380 (seven geometry slider blocks)
- Evidence: every field is `let mut v = e.field; ui.horizontal(|ui| { ui.label(..); if ui.add(egui::DragValue::new(&mut v).speed(..).range(0.0..=f64::MAX).suffix(..)).on_hover_text(..).changed() { let mut updated = e.clone(); updated.field = v; *pending = Some(PendingPropertyEdit::UpdateForce { index, force: ForceElement::X(updated) }); } });` (e.g. BearingFriction lines 492-621 is six copies; GasSpring, JointLimit, Motor, LinearActuator the same). Each Linear* arm also repeats the two draw_point_picker calls with the same clone-and-set closure.
- Why it matters: the owner flags repetition; adding a field means copy-pasting 20 lines and the sign/range/units conventions drift (see LGC-11).
- Suggested action: a `drag_field(ui, label, &mut f64, speed, range, suffix, tip) -> bool` helper and a `commit(updated: ForceElement)` closure per arm; same for the geometry sliders.
- Fix class: needs-judgment (large mechanical refactor; gui tests exist for some arms)
- Confidence: certain

## LGC-11: Length units: three different conventions inside one property panel
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/property_panel/mod.rs:200-207 (link length uses display units), 303-373 (geometry sliders hard-coded "(mm)", `geo.width * 1e3`), 422-441 (mount points hard-coded " m"), 622-629 (ground pivots use display units); property_panel/force_editor.rs (" m", " rad" suffixes throughout); parametric_panel.rs:47 (unit_suffix always SI); parametric_panel.rs:404-405 (counterbalance plot "Driver angle (deg)")
- Evidence: the same panel shows `Slider ... .suffix(units.length_suffix())` for link length but `.text("Width (mm)")` for geometry width and `.suffix(" m")` for mount point x/y, regardless of the DisplayUnits setting the user chose.
- Why it matters: with inches selected the Link Editor mixes inches, mm and m on adjacent rows.
- Suggested action: route these through state.display_units like the link-length and ground-pivot rows already do.
- Fix class: owner-decision (some fields may be intentionally SI)
- Confidence: certain for the inconsistency; intent unknown

## LGC-12: Link label edit has no undo step, no dirty flag, and forces a sweep recompute per keystroke
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/property_panel/pending_edits.rs:239-251 (UpdateLabel); trigger at property_panel/mod.rs:122-128
- Evidence: `PendingPropertyEdit::UpdateLabel` edits blueprint and mechanism then calls `state.mark_sweep_dirty()` only; the sibling edits (set_body_mass blueprint_ops.rs:507-508, edit_geometry pending_edits.rs:98-99) call `push_undo()`, which also sets `dirty` (undo_ops.rs:97-101). The text box fires on every `.changed()` (each keystroke). A label has no effect on the sweep.
- Why it matters: a rename cannot be undone, does not mark the file changed (web autosave skips it, cf. BL-045 for force edits), and each keystroke schedules a pointless full sweep recompute.
- Suggested action: push one undo step per committed edit (on lost_focus) and drop the mark_sweep_dirty.
- Fix class: needs-judgment
- Confidence: likely (label is not read by the solver or sweep; not verified for every sweep metric)

## LGC-13: Parametric panel: five LinearActuator fields have no current-value lookup, and a dead comment
- Kind: broken
- Where: linkage-sim-rs/src/gui/parametric_panel.rs:447-505 (get_force_field_value), 34-40, 93; linkage-sim-rs/src/gui/state/blueprint_ops.rs:234-238 (force_sweepable_fields); state/parametric.rs:44-54
- Evidence: force_sweepable_fields offers for LinearActuator `stroke_min, stroke_max, end_stop_stiffness, end_stop_damping, end_stop_restitution`; get_force_field_value's LinearActuator arm only knows `force` and `speed_limit`, so for the five others `get_current_value` returns the `unwrap_or(1.0)` default and the panel sets the range to 0.5..1.5 whatever the real value (stroke_min = 0.0 gives 0.5..1.5 m). unit_suffix and requires_positive also have no entry for them (suffix "", not positive-only). Line 93 comment "Recompute metric values from cached sweeps if available" sits on code that does nothing, so changing the metric dropdown leaves the plot on the old metric until Run Study.
- Why it matters: a wrong default range for stroke-limit studies, no units on screen, and a dropdown that appears to change a plot but does not.
- Suggested action: add the five fields to get_force_field_value, unit_suffix and requires_positive (one table used by all three would remove the drift); delete or implement the line-93 comment.
- Fix class: mechanical-safe for the lookup; needs-judgment for the dropdown behaviour
- Confidence: certain for the missing lookup (read both lists)

## LGC-14: Health and Diagnostics sections show the same facts twice with different thresholds, colours and wording
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/property_panel/health.rs:41-44,62-71,124-129 vs diagnostics.rs:105-148,191-209,220-233
- Evidence: Grashof class is drawn in both (health: "Crank-Rocker (Grashof)", colour (80,200,120); diagnostics: "Crank-Rocker", colour (100,200,100)); Jacobian conditioning is drawn in both with the same 1e4 / 1e8 cut-offs but different text ("Well-conditioned" / "Conditioning: κ = ..") and different colours; driver torque stats in both ("Peak x / RMS y" vs "Torque: min to max", "RMS"). Peak/RMS actuator force is recomputed by compute_envelope in health.rs:347-352 and 378-382 each frame (up to eleven O(n) passes per frame over sweep data while the section is open, which is default).
- Why it matters: two places to update when a threshold changes, and the user sees one fact in two colours.
- Suggested action: one set of status helpers (label, colour) shared by both, and compute the envelope summary once per sweep (store it on SweepData).
- Fix class: needs-judgment
- Confidence: certain for the duplication

## LGC-15: Actuator-sizing tutorial names menu items and a message that do not exist
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/tutorial.rs:225-226, 292-295
- Evidence: Step 3 says Diagnostics "shows '0% overlap'"; the Diagnostics code (property_panel/diagnostics.rs:69-85) prints "Force zone #N: 0 N applied (body 'x' outside zone)" or "...N applied to 'x' (in zone)", with no percentage. Step 9 says "File > Export CSV" and "File > Export HTML Report"; menu_bar.rs:313 is "Export Sweep CSV..." and menu_bar.rs:518 "Generate Report (HTML)..." (and 557 "Export Animation (HTML)...").
- Why it matters: a user following the tutorial looks for text and menu entries that are not on screen.
- Suggested action: reword the three strings to the real labels; add a test that greps the menu labels the tutorial quotes (cheap guard).
- Fix class: mechanical-safe
- Confidence: certain

## LGC-16: Health panel prints body-frame angles (and metres as degrees) while the plots show display-frame values
- Kind: broken
- Where: linkage-sim-rs/src/gui/property_panel/health.rs:180-183 (toggle angles), 444-488 (utilization, "Exceeding 80%")
- Evidence: toggle angles are formatted `"{:.0}\u{00b0}"` straight from `sweep.toggle_angles`, which has_toggles_in_range's own comment (health.rs:191) says are body-frame, while draw_plot_panel shifts them by driver_display_offset (plot_panel/mod.rs:384-395) and the plot markers use that. draw_actuator_utilization_indicator formats `exceeding.first()/last()` of `sweep.angles_deg` as degrees, but in stroke mode angles_deg holds metres (plot_panel/mod.rs:262-264, 409-411), and the code comment says "contiguous angle ranges" while it prints only first-to-last of all samples over 80% (a gap between two separate ranges is hidden).
- Why it matters: with a crank display offset the Health row says a toggle at 120 deg while the plot marks it at 120 + offset; in a stroke sweep the row reads "0 deg to 0 deg".
- Suggested action: convert through the same x-mapping as the plots (display offset, stroke metres to mm and units) and list each contiguous range, or label it as "first to last".
- Fix class: needs-judgment
- Confidence: likely (follows from the quoted comments; not run)

## LGC-17: Force toolbar "Joint Torques" pairs the selected link with an arbitrary neighbour
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/force_toolbar.rs:172-186 (resolve_target_bodies), 40-77
- Evidence: `mech.joints().iter().find_map(...)` returns the first joint in joint order that touches the selected body; Motor, Torsion Spring, Rotary Damper, Bearing Friction and Joint Limit are then created between the selected link and that one body. A coupler or ground link with three joints silently gets the first one; the menu shows no partner name (only "Target: {body}").
- Why it matters: the user cannot choose or even see which joint the element is placed on.
- Suggested action: show the partner in the label ("Target: crank <-> ground") or offer a picker when more than one neighbour exists.
- Fix class: owner-decision
- Confidence: likely

## LGC-18: Motion ribbon runs up to 20 Newton position solves every frame
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/canvas/rendering/mod.rs:60-70 (call), 1542-1591 (draw_motion_ribbon)
- Evidence: render_mechanism calls `draw_motion_ribbon` each frame whenever `show_motion_ribbon && sweep_mode.is_trajectory()`; the function loops `for i in 0..n` (n = motion_ribbon_n_ghosts, slider up to 20, trajectory_panel/mod.rs:118) and calls `solve_position(mech, &q_seed, t_mech, 1e-10, 50)` for each ghost, then re-derives the bar outlines. Inputs (sweep u_values, driver rate, last_good_q) are unchanged between frames of a static view; `last_good_q` also changes during playback, so the ghost solves depend on the live pose seed.
- Why it matters: up to 20 constrained-position solves plus drawing on every repaint, including idle repaints from hover; the ghosts do not change unless the sweep, n, or the mechanism changes.
- Suggested action: cache the ghost q list on AppState keyed by (sweep version, n, mechanism generation) and draw from the cache; seed from the sweep's own q samples rather than last_good_q if they are stored.
- Fix class: needs-judgment
- Confidence: certain for per-frame solves

## LGC-19: Link-bar drawing and per-element screen-point boilerplate copied in several places
- Kind: inefficient
- Where: linkage-sim-rs/src/gui/canvas/rendering/mod.rs:216-236 (draw_link_bar closure) and 1642-1663 (draw_bar closure in draw_mechanism_ghost, comment says "Reuse the bar geometry" but it is a copy); 162-175 and 1625-1635 (sorted attachment points to screen points, again in the ghost); force_render.rs:112-355 (the same "body_point_global -> world_to_screen -> Pos2::new" four times for the two-point elements, and "get_pose x2 -> screen -> mid" five times for the rotary elements); force_render.rs:269-301 (min and max stroke tick blocks identical)
- Evidence: e.g. force_render.rs:269-284 and 286-301 differ only by `stroke_min` vs `stroke_max`; the draw_link_bar and draw_bar closures compute the same normal, corners and PathShape::convex_polygon.
- Why it matters: the owner flags repetition; a bar-style change must be made twice or the ghosts diverge from the live links.
- Suggested action: `link_bar_polygon(a, b) -> [Pos2;4]`, `attachment_screen_points(body, ...)`, `element_endpoints_screen(...)` and `draw_stroke_tick(frac)` helpers in rendering/primitives.rs.
- Fix class: needs-judgment
- Confidence: certain

## LGC-20: Grid labels and hover tooltips ignore display units; Y grid labels lose the sub-mm precision the X labels have
- Kind: inconsistent
- Where: linkage-sim-rs/src/gui/canvas/rendering/mod.rs:1431-1437 vs 1462-1465 (grid labels); 1304-1307 (force-zone tooltip "mm"); 1511-1523 (body tooltip "mm"); also stray `let _ = shown_tooltip; // suppress unused warning` at 1317
- Evidence: the X loop does `let label = if spacing < 0.001 { format!("{:.1}", mm) } else { format!("{:.0}", mm) };` but the Y loop always uses `format!("{:.0}", mm)`; with zoom spacing 0.1 mm the Y major lines (every 5th, 0.5 mm) all read "0" or "1". Both loops hard-code millimetres although the toolbar has an inch setting (state.display_units) used for dimension labels at mod.rs:288-291.
- Why it matters: at high zoom the vertical axis labels are wrong/duplicated; with inches selected the canvas mixes inch dimension labels and mm grid and tooltip labels.
- Suggested action: share one label formatter between the two loops and run it through display_units.
- Fix class: needs-judgment
- Confidence: certain for the Y/X precision difference

## Leads settled
- eq_rendering.rs `driver_lambda_units` (line 65): real, dead (LGC-01); its doc also points at `lambda_units_at_row`, which does not exist anywhere in the worktree.
- plot_panel/actuator.rs `draw_actuator_stroke_annotation` (line 191): real, dead (LGC-01).
- Longest functions (force_editor draw_force_element_details 1108 lines; draw_property_panel 560; render_mechanism 537; handle_interaction 494; draw_plot_panel 324; draw_diagnostics_section 273; render_overlays 273; draw_force_elements 258; draw_parametric_panel 228; trajectory render 224; target_picker draw 212): the first, second and fourth show real repetition (LGC-10, LGC-19); the rest are long but not wrong. Length alone is not reported.
- Known and not repeated: BL-044 covers the stubbed draw-geometry tool (handle_draw_body_geometry, DrawBodyGeometryState, EnterDrawGeometryMode, and the `let _ = handle_draw_body_geometry;` line at interaction.rs:510). BL-045 covers force-element edits skipping undo. BL-010 covers the tutorial actuator step panic.
- `PendingPropertyEdit::RenameMountPoint` has no producer anywhere (the allow(dead_code) at pending_edits.rs:14 says so); it is the same family as BL-044 but not named there. Left as a note for the decider; it is wired to a working state method, so keep-or-remove is owner-decision.

## Not read
- canvas/interaction.rs lines 340-end (read lines 1-340, 500-515, 1040-1160 only), canvas/context_menu.rs, canvas/alignment.rs, canvas/colors.rs, canvas/hit_testing.rs (grep only), canvas/rendering/weights.rs, canvas/rendering/force_render.rs lines 410-end
- plot_panel/trajectory.rs, plot_panel/weights.rs, plot_panel/coupler.rs lines 130-end
- property_panel/equations.rs, property_panel/weight_editor.rs, property_panel/undo_panel.rs, property_panel/diagnostics.rs lines 300-end, property_panel/force_editor.rs lines 200-489 and 640-end (structure and two arms read), pending_edits.rs lines 320-end, property_panel/mod.rs lines 640-end
- trajectory_panel/profile_input.rs (csv path only), trajectory_panel/target_picker.rs
- canvas/mod.rs tests (lines 420-end)
