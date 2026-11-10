# Verified: linkage-gui-canvas-panels
Worktree HEAD 1048962. Paths are under C:/Users/Cole/source/repos/lsim-audit/linkage-sim-rs/src/gui.

## LGC-01: Two dead functions with misleading names (driver_lambda_units, draw_actuator_stroke_annotation)
- Verdict: CONFIRMED
- Checked: whole-worktree `git grep -e driver_lambda_units -e draw_actuator_stroke_annotation -e lambda_units_at_row` gives only eq_rendering.rs:64 (doc mention), :65 (def) and plot_panel/actuator.rs:191 (def); no cfg, test, doc or JS user. The actuator fn prints Peak/RMS force; a blank line sits between its doc (186-189) and `fn` (191); `lambda_units_at_row` exists nowhere.
- Safe to apply without judgment: yes (no caller on any target or feature, nothing in tests, scripts or docs uses either)
- Exact change: (1) eq_rendering.rs: delete lines 60-71 (the blank line, the four doc lines 61-64 and `pub fn driver_lambda_units` through its closing `}` at 71); the impl's closing `}` at 72 stays. (2) plot_panel/actuator.rs: delete lines 186-235 (doc comment, blank line, `pub(super) fn draw_actuator_stroke_annotation` through its closing `}` at 234, and the blank at 235); and at line 4 change `use egui_plot::{HLine, Line, Plot, PlotPoint, PlotPoints, Points, Text as PlotText, VLine};` to `use egui_plot::{HLine, Line, Plot, PlotPoints, Points, VLine};` (PlotPoint and PlotText are used only inside the deleted fn, lines 230; PlotPoint hits at 80 and 359 are PlotPoints).

## LGC-02: Stale #[allow(dead_code)] attributes on live items
- Verdict: PARTLY. Holds: trajectory_panel/profile_input.rs:365 allow on `parse_keyframes_csv` (cfg native, line 364) is redundant: it is called at :274 inside a `#[cfg(feature = "native")]` block (261) and in tests (386). Not proven: canvas/hit_testing.rs:20 allow on SegmentHit; the struct is built and returned (97-99) but whether every field (world_pos, screen_pos, point_a_name, point_b_name) is read cannot be shown without a build, so the allow may still be hiding unread-field warnings.
- Checked: git grep SegmentHit / parse_keyframes_csv over the worktree; hit_testing.rs 16-100; profile_input.rs 258-276, 362-368.
- Safe to apply without judgment: yes for the profile_input attribute only (the function is used under the same cfg), no for SegmentHit
- Exact change: trajectory_panel/profile_input.rs line 365: delete `#[allow(dead_code)] // used from the native-feature button path`.

## LGC-03: Coupler-trace dashes re-implement draw_dashed_line, and the dashed force-zone rectangle is written twice
- Verdict: CONFIRMED
- Checked: rendering/mod.rs:122-141 hand-codes dash stepping per sweep segment (phase restarts each segment); primitives.rs:285-307 draw_dashed_line; force_render.rs:380-398 and canvas/interaction.rs:1053-1078 both build tl/tr/br/bl and call draw_dashed_line four times with dash 6.0, gap 4.0 (stroke widths differ slightly between the two sites).
- Safe to apply without judgment: no (refactor, trace look changes)

## LGC-04: Arrow drawing duplicated; the "one arrowhead" helper is bypassed
- Verdict: PARTLY. Holds: draw_force_arrow (primitives.rs:390-418) and draw_external_force_arrow (127-161) repeat the same mag, clamp, unit-vector and label steps; draw_torque_arc (198-211, head 5 px, 0.5 rad) and the gravity indicator (rendering/mod.rs:716-731, 4 px right-angle head) hand-roll heads. Does not hold: the helper is not bypassed by the force arrows or draw_axis_component, they all call draw_arrowhead with ARROW_HEAD_LEN_PX (lines 151, 408, 483), so their heads are identical; only two sites differ.
- Checked: primitives.rs 125-162, 198-212, 352-364, 388-420, grep of draw_arrowhead; mod.rs 710-733.
- Safe to apply without judgment: no

## LGC-05: Whole SweepData cloned every frame to shift two vectors
- Verdict: CONFIRMED
- Checked: plot_panel/mod.rs:265-268 calls sweep_in_display_frame each draw; 384-396 `sweep.clone()` (also cloned on the stroke / zero-offset path at 385-387), then shifts angles_deg and toggle_angles; the doc comment at 382 says the full clone is cheap enough. Cost size unmeasured.
- Safe to apply without judgment: no

## LGC-06: Fourteen copies of the plot trailer and the per-plot Plot builder
- Verdict: CONFIRMED
- Checked: `git grep -c "from_rgba_premultiplied(255, 255, 255, 100)"` in plot_panel: actuator 3, coupler 3, dynamics 3, mechanics 4, weights 1 = 14; draw_range_boundary_markers( and detect_plot_click( each appear 15 times (includes the definition or one helper use); nathan recolours by hand at actuator.rs:84, 104, 366 and dynamics.rs:130 (actuator.rs:145 is another closure).
- Safe to apply without judgment: no

## LGC-07: Two tabs both labelled "Actuator Force" in stroke mode
- Verdict: CONFIRMED
- Checked: plot_panel/mod.rs:107-125 (PlotTab::DriverTorque labelled "Actuator Force" in stroke mode, enabled by `driver_torques.is_some()`) and 185-193 (PlotTab::ActuatorForce "Actuator Force", enabled by actuator_forces non-empty); hover texts differ. Both enabled together depends on data, plausible.
- Safe to apply without judgment: no (owner wording)

## LGC-08: Reference lines span a hard-coded 0..360 x range
- Verdict: CONFIRMED
- Checked: mechanics.rs:102 `x_max = units.angle(2*PI)`, 134-135 threshold lines from 0.0 to x_max, 150 text at x_max*0.5, 232-233 MA = 1 line 0.0..x_max; curve x goes through sweep_x_to_display and the plot_panel/mod.rs:265-268 display offset; sweep/mod.rs:636-652 computes mechanical_advantage in the shared sweep path with no stroke-mode exclusion (transmission angle in stroke mode not checked).
- Safe to apply without judgment: no

## LGC-09: Doc comments attached to the wrong function, plus stale in-body comments
- Verdict: CONFIRMED
- Checked: plot_panel/mod.rs:547-551 coupler-trace doc above draw_toggle_markers (552); mechanics.rs:185-189 driver-torque doc above draw_mechanical_advantage; coupler.rs:86-90 body-angles doc above draw_coupler_velocity; actuator.rs:29-34 stale comment; mod.rs:107 "Only show ... if data exists" over add_enabled_ui. A scan for a `///` line, blank line, then fn/pub finds exactly 12 sites (actuator 190, 241, 306; coupler 89, 163, 236; dynamics 75, 182; mechanics 88, 188, 262; mod 551).
- Safe to apply without judgment: no (deleting only the blank lines would attach the wrong docs to the wrong functions; each doc needs rewording)

## LGC-10: Force editor repeats one DragValue block about 50 times in a 1,100-line function
- Verdict: CONFIRMED
- Checked: force_editor.rs has 50 `DragValue::new` uses and 1514 lines; draw_force_element_details starts at 61; BearingFriction block (492-621) shows the clone-and-set pattern; property_panel/mod.rs has 9 Slider::new, geometry blocks from 282.
- Safe to apply without judgment: no

## LGC-11: Length units: three different conventions inside one property panel
- Verdict: CONFIRMED
- Checked: property_panel/mod.rs:204 length_suffix; 306, 321, 336, 352, 366 hard-coded "(mm)"; 425, 441 `.suffix(" m")`; 629, 643, 676 length_suffix; parametric_panel.rs:47 uses parameter.unit_suffix() (state/parametric.rs:39-57, SI only); parametric_panel.rs counterbalance plot label "Driver angle (deg)". Intent unknown.
- Safe to apply without judgment: no

## LGC-12: Link label edit has no undo step, no dirty flag, and forces a sweep recompute per keystroke
- Verdict: PARTLY. Holds: pending_edits.rs:239-251 UpdateLabel edits blueprint and mechanism and calls only mark_sweep_dirty (no push_undo, so no undo entry and dirty stays false); property_panel/mod.rs:123 fires on every `.changed()`. Does not hold: "A label has no effect on the sweep": analysis/gravity_breakdown.rs:130 builds weight-breakdown names from `body.label`, so a sweep recompute after a rename has a visible use; dropping mark_sweep_dirty would leave stale names.
- Checked: pending_edits.rs 236-254; property_panel/mod.rs 116-132; git grep "\.label" in gui/sweep and analysis.
- Safe to apply without judgment: no

## LGC-13: Parametric panel: five LinearActuator fields have no current-value lookup, and a dead comment
- Verdict: PARTLY. Holds: blueprint_ops.rs:234-238 offers stroke_min, stroke_max, end_stop_stiffness, end_stop_damping, end_stop_restitution; parametric_panel.rs:498-502 LinearActuator arm only has "force" and "speed_limit"; get_current_value falls back to 1.0 (parametric_panel.rs:441, range 0.5..1.5 at 36-40); state/parametric.rs unit_suffix (39-57) has no entry for the five (end_stop_stiffness does not match "stiffness"); line 93 comment sits on no code. Not verified: the claim that changing the metric dropdown leaves the plot on the old metric (the plot code was not read).
- Checked: the three listings above, parametric_panel.rs 30-100 and 440-505.
- Safe to apply without judgment: no (adds behaviour)

## LGC-14: Health and Diagnostics sections show the same facts twice with different thresholds, colours and wording
- Verdict: CONFIRMED
- Checked: health.rs draw_grashof_indicator ("Crank-Rocker (Grashof)", COLOR_OK = (80,200,120)) vs diagnostics.rs:105-119 ("Crank-Rocker", (100,200,100)); conditioning in health.rs:297-303 ("Well-conditioned", 1e4 / 1e8) vs diagnostics.rs:199-208 ("Conditioning:", same cut-offs, other colours); torque "Peak / RMS" (health 235) vs "Torque: min to max" and "RMS" (diagnostics 229-232); health.rs has 10 compute_envelope( call sites.
- Safe to apply without judgment: no

## LGC-15: Actuator-sizing tutorial names menu items and a message that do not exist
- Verdict: CONFIRMED
- Checked: tutorial.rs:225 '0% overlap', 292 "File > Export CSV", 294 "File > Export HTML Report"; menu_bar.rs:313 "Export Sweep CSV...", 518 "Generate Report (HTML)...", 557 "Export Animation (HTML)..." all inside the File menu (menu_button("File") at 34, next top-level menu at 694); diagnostics.rs:73,81 print "Force zone #{}: 0 N applied ..." / "... N applied to ... (in zone)", no percentage. Whole-worktree `git grep` for the three strings hits only tutorial.rs, so no test, script or doc quotes them.
- Safe to apply without judgment: yes (string text in a tutorial, nothing matches on it)
- Exact change: tutorial.rs line 225 `if it shows '0% overlap', \` -> `if it shows 'Force zone #N: 0 N applied', \`; line 292 `File > Export CSV to get` -> `File > Export Sweep CSV... to get`; line 294 `File > Export HTML Report for` -> `File > Generate Report (HTML)... for`.

## LGC-16: Health panel prints body-frame angles (and metres as degrees) while the plots show display-frame values
- Verdict: CONFIRMED
- Checked: health.rs:180-184 formats sweep.toggle_angles with `{:.0}°`; has_toggles_in_range comment 191 says body-frame; plot_panel/mod.rs:384-395 shifts them by the display offset for the plots. health.rs:473-488 formats first and last of all samples over 80% as degrees from angles_deg (metres in stroke mode per plot_panel/mod.rs:262-264) and the comment says "contiguous angle ranges".
- Safe to apply without judgment: no

## LGC-17: Force toolbar "Joint Torques" pairs the selected link with an arbitrary neighbour
- Verdict: CONFIRMED
- Checked: force_toolbar.rs:172-186 find_map returns the first joint touching the selected body; the dropdown (40-77) creates Motor, TorsionSpring, RotaryDamper, BearingFriction, JointLimit between selected and that body; the label (27) shows only `Target: {body}`.
- Safe to apply without judgment: no (owner decision)

## LGC-18: Motion ribbon runs up to 20 Newton position solves every frame
- Verdict: CONFIRMED
- Checked: rendering/mod.rs:64-70 calls draw_motion_ribbon each frame in trajectory mode; 1542-1591 loops n ghosts calling solve_position(..., 1e-10, 50) seeded from state.last_good_q; slider range 2..=20 at trajectory_panel/mod.rs:118; no cache.
- Safe to apply without judgment: no

## LGC-19: Link-bar drawing and per-element screen-point boilerplate copied in several places
- Verdict: CONFIRMED
- Checked: rendering/mod.rs:216-236 draw_link_bar vs 1642-1663 draw_bar (same geometry; the ghost comment says "Reuse the bar geometry"; only the stroke width and Vec vs array differ); attachment-point sort and screen map at 1625-1635 duplicates the 162-175 pattern (not opened); force_render.rs:269-301 min and max stroke tick blocks differ only by stroke_min vs stroke_max.
- Safe to apply without judgment: no

## LGC-20: Grid labels and hover tooltips ignore display units; Y grid labels lose sub-mm precision
- Verdict: CONFIRMED
- Checked: rendering/mod.rs:1431-1437 X labels use `{:.1}` when spacing < 0.001, 1462-1464 Y labels always `{:.0}`; both multiply by 1000 (mm); 1303-1307 force-zone tooltip "mm"; 1510-1523 body tooltip "mm"; 1317 `let _ = shown_tooltip; // suppress unused warning`.
- Safe to apply without judgment: no

## Seen while checking
- (none)