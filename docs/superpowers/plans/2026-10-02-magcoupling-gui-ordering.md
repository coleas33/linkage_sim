# Magcoupling GUI: Smarter Ordering of Inputs and Outputs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer the user's "Is there a way to sort the inputs and outputs more intelligently?" in the magcoupling panel: the inputs ordered by design workflow, the requirements first (the workbook's package groups one toggle away), with their rarely changed rows under Advanced and a filter box, the results table grouped by physics chain with the headline first, check badges (each group heading with its worst check) and a failing-checks filter, and a click that traces an input to the explained results it drives or a result to the inputs it reads, with design files, share links and exports byte-for-byte unchanged.

**Architecture:** Everything is view state and lookup tables in `magcoupling-rs/src/gui/`; the engine is untouched. `inputs.rs` gains a second arrangement of the same `InputEntry` values (`WORKFLOW`, `InputCatalogue::workflow`, `groups_in(InputOrder)`, `filter_inputs`), and `panel.rs` draws the order the session chose (`MagcouplingPanel::input_order`, never saved), an Advanced sub-heading per workflow group and the filtered view. `dashboard.rs` lists every design check (`CHECKS`) with a level for each text it gives (`verdict_level`, extended), and a new `result_groups.rs` arranges `table_entries` into the headline, the explorer's A-3 chains (`engine::explain::scope::SCOPE`) widened by a prefix table (`CHAIN_PREFIXES`) with two chains of their own (adhesive, mass), and the other results by package; `results_table.rs` turns a search, an order, the failing filter and the open groups into one list of equal-height `Line`s, so `show_rows` still draws only what is on screen, and badges each group heading with the worst level of its checks. A new `trace.rs` asks the equation registry for an input's `downstream` results or a result's `upstream_inputs` (only at the two click sites, never per frame), and the panel adds those paths to the frame's `TermColors` (`with_marked`), so the existing `Readouts::mark` frames them wherever they are drawn (dashboard, table, callouts, input rows).

**Tech Stack:** Rust 2024 (rustc 1.89); egui 0.32.3 and egui_plot 0.33.0 (no new dependency); headless egui tests (`egui::Context::run` with injected input, painted shapes and texts inspected); Playwright MCP for the browser check.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-order/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, section M4 "Layout" ("**Left — inputs**, generated from metadata, grouped as the package groups them (`coupling`, `metal`, `calibration`, `materials`, `temperature`, `clamps`). A **Key design** group on top ..."; "**Results table**: every computed value with label, unit, cell; searchable; CSV and JSON export"), which this plan amends (decision O-1; Task 6 writes the amendment), and the user's request recorded in `docs/ai/04-memory.yaml` ("NEXT (user request 2026-10-02)": the five candidate approaches, "Keep the workbook grouping available as an alternate view (schema and share links unchanged)").

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-order`, branch `magcoupling/ordering`, created by Task 0 from `main` at `746f73e` with LF line endings and a worktree-scoped `core.autocrlf=false`. Every command uses absolute paths into it. Nothing is pushed.

**Process (the user's lean process):** seven tasks (0 to 6). Tasks 1 to 6 give the exact code and text, so the implementers transcribe; every implementation and every per-task review runs on `sonnet`; there is no pre-flight scan, so every block below quotes `main` at `746f73e` (or the file as the earlier tasks leave it) exactly, and was parsed back out of this file and replayed literally (Verification record); the whole-branch review after Task 6 runs on the session model.

## Decisions to confirm

**Confirmed (user, 2026-10-02): the recommended option on all of O-1 to O-8.** No task stops on a decision.

**Not yet confirmed.** These UX choices arose while turning the request into code; no approved decision settles them. The plan implements the recommended option of each (the code and docs cite the ids). A critic's review of the first version changed the recommendations of O-2, O-4, O-6 and O-8 (and the empty-state text of O-7); each earlier recommendation is now an alternative, and the Self-review record at the end lists every change. Task 0 Step 8 asks the user and records the answers; if the user picks another option, the task that implements it stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| O-1 | Which input order is the default, and does the choice persist? | **The workflow order by default**, "Workbook groups" one click away on a toggle row above the filter box; the choice is panel state for the session (never in a design file or a share link, back to the workflow order on reload) | (b) the workbook order by default, the workflow order opt-in; (c) remember the choice across reloads (the panel's first persisted UI preference) | 1, 2 |
| O-2 | The workflow groups, their names and order | **The eight groups of the table below, in the order a designer works, the spec first:** Requirements and operating conditions (the required pull-out torque, the operating and minimum temperatures, the space claim, the drive and gearbox, the duty and the slip events: what a designer fixes before choosing magnets), Magnets and rings, Gap and clearances, Housing and retainers, Shaft, key and clamps, Materials, Thermal and demagnetization, Calibration and model; sections with headings inside; related inputs share a section (the adhesive with its bondlines, recommended bondline, shear modulus and fatigue data in Materials; the optional adapter's geometry with its joint screws; the fields stored from a 3D run beside the 3D reference torques in Calibration and model, each section labelled "3D results: ..."); the Key design group stays on top and its inputs stay in their groups too (as M41-13) | (b) the order before this revision: Magnets and rings first and Operating conditions sixth, the space claim under Housing and retainers, the adhesive's inputs in three groups, the adapter joint under Shaft, key and clamps, the stored 3D fields under Thermal and demagnetization; (c) split Thermal and demagnetization into two groups; (d) four coarser groups (Geometry, Materials, Duty, Model) | 1 |
| O-3 | Badge levels of the 23 checks off the dashboard | **The check-level table below:** red where a limit is exceeded, amber where the workbook asks for a look or a test (an unknown magnet rating, a CHECK screen, a hardened washer, the vent port, the daily-cycle screen), no badge where the check does not apply; "Nominal only: hot test" is green, as the temperature verdict's "Confirm ... by test" already is; the clamp table's per-size checks and the sweeps' status texts rate candidates, not the design, and get none | (b) badges for the dashboard's 7 checks only; (c) "Nominal only: hot test" amber | 3, 4 |
| O-4 | Which rows go under Advanced (closed by default) | **27 rows in 6 sections:** the bedding clearances, the optional adapter and its joint, the clamp and screw factors (both clamp factors, the nut factor, the thread engagement, the stripping safety factor), the screw classes, the slip-loss end factor and the model constants (both end-effect coefficients, the harmonic set, both vacuum permeabilities, the calibration's alpha and original coefficient). The clamp's two flagged assumptions (the shaft-to-bore friction and the preload share) stay in the Clamp section, and the 14 fields stored from a 3D run stay in view, labelled "(refresh after a 3D run)": both scale results directly, and the stored fields go stale with every geometry change | (b) the 40 rows in 8 sections before this revision (the friction and the preload share in the screw model, the stored 3D fields under Thermal and demagnetization's Advanced heading); (c) the stored 3D fields under Advanced, with these labels; (d) no Advanced heading; (e) Advanced chosen by metadata (every assumption and every Rust-only input) | 1, 2 |
| O-5 | How the inputs filter shows its matches | **The matching rows alone,** each run under "Group / Section" (the group's label alone for its own section), with "N of 172 inputs"; it matches label, path or workbook cell like the results search (one shared helper); the Key design group and the groups give way while it holds more than blanks; a leaf term focusing a row clears it | (b) keep the groups, open, and hide the rows that do not match; (c) match the help text too | 2 |
| O-6 | How the results table groups its rows, and what it shows first | **By physics chain by default:** "Headline" (the dashboard's 17 rows, in its order) open; then eight chain groups, closed: the six A-3 chains (Torque, Temperature, Demagnetization, Slip heating, Clamps, Geometry, in the explorer scope's order), each also taking the results of its nested groups that the scope leaves out (`CHAIN_PREFIXES`, the table below: the cold demagnetization limits and check, the slip temperatures and slip life, the magnet life's peak, the metal design's clearances, gaps and reserves), with Adhesive (the adhesive and mismatch screens) after Slip heating and Mass last; then "Other results" by package (in schema order; a package left empty is dropped), closed; a result listed in two places sits in the first; a heading opens and closes its group and shows the worst level of its checks, so a closed group with a failing check shows red; a search opens every group it matches, and a heading click while it holds text is ignored (its hover text says to clear the search); "Engine order" brings back the flat table; the exports keep the engine order | (b) the chains open by default; (c) the headline as `HEADLINE`'s 15 rows; (d) the engine order by default; (e) the grouping before this revision: the scope's lists alone, so 883 of the 1086 rows (81%) fell under "Other results", among them all 85 temperature design rows and the cup ring check, red at the defaults, in a closed group with no sign of it | 3, 4 |
| O-7 | What the failing filter shows | **A "Failing checks only" checkbox:** the red checks, then the amber ones, each colour in schema order, flat (no headings), counted in "N of 1086 results"; every check row carries the dashboard's badge before its value in every view; with no failing check the table says "No check fails or asks for a look.", and with a search no failing check matches, "No result matches the search." | (b) the red ones only; (c) "failing first": every check, red, amber, then green | 4 |
| O-8 | How tracing is triggered and drawn | **A click:** on an input row's label it traces the input: the explained results downstream of it are framed (the registry explains 392 of the 1086 results, so an input whose results have no equation record, such as the drive torque, reaches none, and its banner says so); on a result readout (which still opens the Equation panel) it traces the inputs the result reads, except that a click on a result an input's trace marks keeps that trace (reading the equations of the traced rows is the table's main gesture) and a click on a result without an equation record keeps the trace (nothing is known of its inputs); it lasts until Clear trace, a second click on the label, or a click that replaces it; frames in the selection colour (the colour of the frame a leaf term draws) in the explorer's mark style, under the equation's own term colours; group headings count the rows traced; "Traced only" in the table, for an input's trace only (a result's marks inputs), goes off with the trace | (b) hover (it already means "show this equation's direct terms"; transient, and a frame per hover); (c) a Trace button on every row; (d) the rule before this revision: every readout click replaces the trace (a result without a record ends it) and "Traced only" stays ticked after Clear trace, so reading a traced row collapses the filtered table to that row | 5 |

### The proposed input grouping (O-2, O-4)

Every one of the 172 inputs sits in exactly one section (Task 1's coverage test fails, naming the input, when an input is added to the engine without one). The Key design inputs (face gap, pole count, both magnet parts, axial length, operating temperature, back iron, cup wall, conductance, measured drag) also sit on top, as today.

| Group | Section | Advanced | Inputs, in order (label, `path`; A = model assumption, R = Rust-only) |
|---|---|---|---|
| Requirements and operating conditions | (the group's own, no heading) |  | Required minimum service pull-out torque `metal.required_min_Nm`; Operating magnet temperature `coupling.op_temp_C`; Minimum magnet temperature `metal.min_temp_C`; Symmetric torque variation allowance `metal.variation` (A) |
| Requirements and operating conditions | Space claim |  | Maximum rotating coupling diameter `metal.max_diameter_mm`; Maximum overall axial length `metal.max_overall_axial_mm`; Maximum large-diameter axial region `metal.max_large_dia_axial_mm` |
| Requirements and operating conditions | Drive and gearbox |  | Torque the coupling must carry for driving (at the wheel) `coupling.drive_torque_Nm`; Safety factor wanted on the driving torque `coupling.drive_safety_factor`; Gearbox ratio `coupling.gear_ratio`; Gearbox efficiency `coupling.gear_efficiency`; Gearbox input torque rating `coupling.gearbox_input_rating_Nm` |
| Requirements and operating conditions | Duty and life |  | Wheel-side (inner) rotor maximum speed `temperature.duty.wheel_rotor_rpm`; Hot-day ambient temperature `temperature.duty.hot_ambient_C`; Coupling rise above ambient while driving (no slip) `temperature.duty.driving_rise_C` (A); Operating hours over the system life `temperature.duty.life_hours`; Service life `temperature.adhesive_life.service_years`; Daily temperature swing at the coupling `temperature.adhesive_life.daily_swing_C` |
| Requirements and operating conditions | Slip events |  | Relative slip speed `metal.slip_rpm`; Slip duration per event `metal.slip_event_s` (A); Life events `metal.life_events`; Slip fault trip time (unbroken slip) `temperature.duty.fault_trip_s` |
| Magnets and rings | (the group's own, no heading) |  | Number of poles per ring `coupling.npole`; Inner magnet part `coupling.magnets.part_inner`; Outer magnet part `coupling.magnets.part_outer`; Axial magnet length, both rings `coupling.magnets.axial_length_mm` (R); Geometry type `coupling.faceted`; Inner magnet back apothem, including bondline `coupling.inner_back_apothem_mm` |
| Magnets and rings | Grades (manual dimensions) |  | Inner magnet grade (manual dimensions) `coupling.magnets.grade_inner` (R); Outer magnet grade (manual dimensions) `coupling.magnets.grade_outer` (R) |
| Magnets and rings | Manual inner blocks |  | Manual inner length (axial) `coupling.magnets.manual_inner_length_mm`; Manual inner width (tangential) `coupling.magnets.manual_inner_width_mm`; Manual inner thickness (radial) `coupling.magnets.manual_inner_thickness_mm`; Manual inner Br at 20 °C `coupling.magnets.manual_inner_br_T` |
| Magnets and rings | Manual outer blocks |  | Manual outer length (axial) `coupling.magnets.manual_outer_length_mm`; Manual outer width (tangential) `coupling.magnets.manual_outer_width_mm`; Manual outer thickness (radial) `coupling.magnets.manual_outer_thickness_mm`; Manual outer Br at 20 °C `coupling.magnets.manual_outer_br_T` |
| Gap and clearances | (the group's own, no heading) |  | Candidate flat-face magnetic gap `metal.face_gap_mm`; Inner rotating retaining sleeve thickness `metal.sleeve_mm`; Outer rotating keeper liner thickness `metal.liner_mm`; Minimum desired residual running clearance `metal.residual_target_mm` |
| Gap and clearances | Running-clearance allowances |  | Relative shaft radial displacement allowance `metal.shaft_displacement_mm`; Combined assembled runout allowance `metal.runout_mm`; Additional load deflection / tilt allowance `metal.deflection_mm`; Differential thermal movement allowance `metal.thermal_mm`; Sleeve fit / thickness / form allowance `metal.sleeve_form_mm`; Magnet position / size allowance `metal.magnet_position_mm` |
| Gap and clearances | Bedding clearances | yes | Inner sleeve minimum bedding clearance `metal.sleeve_bedding_mm`; Outer liner minimum bedding clearance `metal.liner_bedding_mm` |
| Housing and retainers | (the group's own, no heading) |  | Back iron `coupling.backiron`; Minimum outer return ring wall `metal.cup_wall_corner_mm`; Steel inner hub axial length `metal.hub_length_mm`; Cup cavity axial depth `metal.cup_depth_mm`; Integral steel rear web thickness `metal.web_mm`; Integral steel boss axial length `metal.boss_length_mm`; Shaft boss outside diameter `metal.boss_od_mm` |
| Housing and retainers | Front cap |  | Front cap axial addition `metal.cap_axial_mm`; Threaded cap OD `metal.cap_od_mm`; Cap thread nominal diameter `metal.cap_thread_dia_mm`; Cap thread engagement length `metal.cap_thread_engagement_mm` |
| Housing and retainers | Endplates and retainers |  | Inner front endplate thickness `metal.front_endplate_mm`; Inner rear endplate thickness `metal.rear_endplate_mm`; Nominal retainer axial span `metal.retainer_span_mm`; Rear endplate screw clearance diameter `metal.rear_endplate_hole_mm` |
| Housing and retainers | Optional adapter and its joint | yes | Optional adapter flange diameter `metal.adapter_flange_dia_mm`; Optional adapter flange thickness `metal.adapter_flange_mm`; Optional adapter pilot diameter `metal.adapter_pilot_dia_mm`; Optional adapter pilot length `metal.adapter_pilot_mm`; Optional adapter boss extension `metal.adapter_boss_mm`; Optional joint extra hardware allowance `metal.adapter_hardware_g`; Adapter joint screws `clamps.joint_screws`; Bolt circle diameter `clamps.joint_bolt_circle_mm`; Friction coefficient, nickel plate on aluminium `clamps.joint_friction` |
| Shaft, key and clamps | (the group's own, no heading) |  | Keyed bore diameter `coupling.bore_mm`; Keyway depth in the hub `coupling.keyway_depth_mm`; Key width `clamps.key_width_mm`; Key contact height in the hub `clamps.key_contact_mm` |
| Shaft, key and clamps | Clamp |  | Clamp type `clamps.clamp_type`; Safety factor, clamp alone `clamps.safety_factor`; Screw class `clamps.screw_class`; Aluminium `clamps.alloy`; Boss outside diameter `clamps.boss_od_mm`; Clamp length, free end to relief cut `clamps.clamp_length_mm`; Friction coefficient, shaft to bore `clamps.friction` (A); Preload as a share of proof load `clamps.preload_fraction` (A) |
| Shaft, key and clamps | Clamp geometry |  | Slit width `clamps.slit_mm`; Minimum ligament, bore to screw hole `clamps.ligament_mm`; Minimum wall outside the screw hole `clamps.wall_out_mm`; Minimum head-side jaw (grip) `clamps.grip_min_mm`; Axial margin, clamp end to counterbore edge `clamps.axial_margin_mm`; Relief cut width `clamps.relief_mm`; Hinge left under the relief cut `clamps.hinge_mm` |
| Shaft, key and clamps | Clamp and screw factors | yes | Clamp factor, one-piece `clamps.factor_one_piece`; Clamp factor, two-piece `clamps.factor_two_piece`; Nut factor `clamps.nut_factor`; Thread engagement in aluminium (× screw diameter) `clamps.engagement_x_d`; Safety factor on thread stripping `clamps.strip_sf` |
| Materials | (the group's own, no heading) |  | Back iron material (hub, cup and boss) `materials.parts.back_iron` (R); Sleeve and liner material `materials.parts.sleeve_liner` (R); Cap and housing material `materials.parts.cap_housing` (R) |
| Materials | Back-iron steel |  | Design flux density for the back-iron check `materials.steel.bsat_T` (A); Electrical conductivity `materials.steel.conductivity_S_m`; Incremental relative permeability (with the magnet bias) `materials.steel.mu_r_incremental`; Specific heat `materials.steel.specific_heat_J_kgK`; Expansion coefficient `materials.steel.cte_per_C`; Elastic modulus `materials.steel.modulus_GPa`; Density `materials.steel.density_g_cm3` |
| Materials | Magnet properties |  | NdFeB expansion in the bond plane `temperature.mismatch.ndfeb_cte_per_C`; NdFeB elastic modulus `temperature.mismatch.ndfeb_modulus_GPa`; NdFeB conductivity `temperature.slip_loss.sigma_ndfeb_S_m`; Specific heat, NdFeB `temperature.thermal.c_ndfeb` |
| Materials | Adhesive and bondlines |  | Selected adhesive (code) `temperature.adhesive.selected`; Inner magnet back bondline `metal.bond_inner_mm`; Outer magnet back bondline `metal.bond_outer_mm`; Recommended bondline `temperature.mismatch.recommended_bondline_mm`; Adhesive shear modulus `temperature.mismatch.adhesive_shear_modulus_GPa`; Share of lap-shear strength retained at the peak temperature `temperature.adhesive_life.hot_strength_retained`; Fatigue endurance at 10^8+ cycles, share of static strength `temperature.adhesive_life.fatigue_endurance` |
| Materials | Other part properties |  | 316L conductivity `temperature.slip_loss.sigma_316_S_m`; Specific heat, 316L `temperature.thermal.c_316`; Specific heat, aluminium `temperature.thermal.c_aluminium`; Plating thickness per surface `materials.nickel.thickness_mm` |
| Materials | Densities and mass allowance |  | Steel density (4140) `metal.steel_density_g_mm3`; Sleeve density `metal.sleeve_density_g_mm3`; Aluminium cap / adapter density `metal.al_density_g_mm3`; Keys / screws / lock tab mass allowance `metal.hardware_g` |
| Materials | Screw classes | yes | Class 12.9 proof stress `materials.screws.proof_12_9_MPa`; Class 10.9 proof stress `materials.screws.proof_10_9_MPa`; Stainless A4-70 yield stress `materials.screws.yield_A4_70_MPa` |
| Thermal and demagnetization | (the group's own, no heading) |  | Thermal conductance to the housing and shafts `temperature.thermal.conductance_W_K` (A); Measured mean slip drag torque `metal.measured_drag_Nm`; High-case multiplier on the estimate `temperature.slip_loss.high_multiplier` |
| Thermal and demagnetization | Demagnetization |  | Coercivity for the demagnetization check `temperature.demag.coercivity_source` (R); Intrinsic coercivity Hcj at 20 °C (grade minimum) `temperature.demag.hcj20_kA_m`; Hcj temperature coefficient (effective, 20–150 °C) `temperature.demag.beta_hcj_per_C` (A); Knee field as a fraction of Hcj `temperature.demag.knee_fraction` (A); Design margin below the onset `temperature.demag.design_margin_C` (A) |
| Thermal and demagnetization | Slip-loss model | yes | End factor for thin shells and the cap `temperature.slip_loss.end_factor` |
| Calibration and model | (the group's own, no heading) |  | Measured pull-out torque `calibration.measured_torque_Nm`; Assumed test magnet temperature `calibration.test_temp_C`; Reported spacing `calibration.spacing_mm`; Gap definition `calibration.gap_definition`; Total installed magnets `calibration.total_magnets` |
| Calibration and model | Prototype magnets |  | Prototype inner hub apothem `calibration.apothem_mm`; Prototype magnet axial length `calibration.magnet_length_mm`; Prototype magnet tangential width `calibration.magnet_width_mm`; Prototype magnet radial thickness `calibration.magnet_thickness_mm`; Prototype remanence at 20 °C `calibration.br_T` |
| Calibration and model | 3D results: reference torques |  | 3D result at 1.0 mm corner gap `calibration.fea_torque1_Nm`; 3D result at 1.5 mm corner gap `calibration.fea_torque2_Nm` |
| Calibration and model | 3D results: stored reverse fields (refresh after a 3D run) |  | 3D worst reverse field, rings aligned `temperature.demag.h_rev_aligned_kA_m`; 3D worst reverse field at pull-out `temperature.demag.h_rev_pullout_kA_m`; 3D worst reverse field, like poles facing `temperature.demag.h_rev_likepole_kA_m`; 3D worst reverse field, single ring on its carrier `temperature.demag.h_rev_single_ring_kA_m` |
| Calibration and model | 3D results: stored slip-loss fields (refresh after a 3D run) |  | Opposite-ring field at hub steel (fundamental) `temperature.slip_loss.b_hub_T`; Opposite-ring field at cup steel (fundamental) `temperature.slip_loss.b_cup_T`; Opposite-ring field at the inner sleeve (fundamental) `temperature.slip_loss.b_sleeve_T`; Opposite-ring field at the outer liner (fundamental) `temperature.slip_loss.b_liner_T`; Cap-face end field, ∫Bz² r² dA `temperature.slip_loss.cap_integral_T2m4`; Rear-web end field, ∫B² dA `temperature.slip_loss.web_integral_T2m2`; Alternating radial field inside the blocks `temperature.slip_loss.b_magnet_T`; Opposite-ring field at an aluminium hub (fundamental, free space) `temperature.slip_loss.b_hub_free_T` (R); Opposite-ring field at an aluminium cup (fundamental, free space) `temperature.slip_loss.b_cup_free_T` (R); Rear-web end field of an aluminium web, ∫B² dA (free space) `temperature.slip_loss.web_integral_free_T2m2` (R) |
| Calibration and model | Model constants | yes | End-effect coefficient `coupling.c_end` (A); Highest odd harmonic summed `coupling.max_harmonic` (AR); Vacuum permeability `coupling.mu0`; Reversible Br temperature coefficient `calibration.alpha_br_per_C` (A); Original end-effect coefficient `calibration.c_end` (A); Original calibration coefficient `calibration.f_cal_original` (A); Vacuum permeability `calibration.mu0` |

### The results groups (O-6)

| Group | Rows | What it holds |
|---|---|---|
| Headline (open) | 17 | the dashboard's rows (`DASHBOARD`): the 15 headline numbers, the space claim, the end-effect check |
| Torque | 49 | the scope's torque chain (harmonic shear stresses, 2D torque, end and calibration factors, pull-outs, the metal design's torques), less the headline's |
| Temperature | 38 | the temperature chain (Br at temperature, the summary's limits and margins, ratings, adhesive and mismatch limits) and the magnet life's peak (`temperature.magnet_life.`) |
| Demagnetization | 27 | the demagnetization chain's onsets, limits and margins not in Temperature, and the rest of `temperature.demag.` (the coercivities used, the cold onsets, limits and check, each ring's limits) |
| Slip heating | 66 | slip losses, drag, the thermal network's times and rotations, and the rest of `temperature.summary.` (the steady slip temperatures, the peak with the fault, the slip rotations), `temperature.duty.`, `temperature.thermal.` and `temperature.slip_life.` |
| Adhesive | 30 | `temperature.adhesive.`, `temperature.adhesive_life.` and `temperature.mismatch.`: the bond stresses, the fatigue and daily screens, the mismatch reading (no scope chain lists them) |
| Clamps | 51 | preload, tightening, capacity and safety factors, including every row of the clamp table's chain columns |
| Geometry | 20 | gaps, clearances, stack lengths, overshoots, and ten of the metal design's: the running clearance, the corner gap and clearance, the assembled face gap, the sleeve-to-liner clearance, the allowed displacement, the cup body OD, the three reserves |
| Mass | 5 | `mass.`: the magnets, the cup, the hub, the boss, the added inertia (the total is on the headline) |
| Other results: Calibration, Coupling model, Retainers, Metal design, Materials, Shaft clamps, Material warnings, Housing, Gap sweep, Pole sweep | 21, 66, 9, 12, 20, 152, 6, 3, 338, 156 (783 rows, 289 outside the two sweeps) | every other result under its package, in schema order (the Temperature design and Mass packages are left empty and dropped); each heading badges the worst of its checks (Coupling model red at the defaults) |

### The check levels (O-3)

The dashboard's seven checks keep their levels. The 23 others (Task 3 extends `verdict_level`; texts as the engine writes them):

| Check | Green | Amber | Red | No badge |
|---|---|---|---|---|
| `calibration.end_effect_check` | `OK` | | `End-effect model out of range` | |
| `model.inner_flat_check`, `model.outer_flat_check` | `OK, ...` | | `TOO NARROW: ...` | `n/a (arcs)` |
| `model.verdict` | `Nominal only: hot test` | | `Below hot minimum` | |
| `model.cup_ring_check`, `model.hub_check` | `Thickness OK` | | `Too thin` | `No back iron` |
| `model.inner_temp_check`, `model.outer_temp_check` | `OK` | `unknown` (a manual magnet without a grade has no rating) | `OVER the magnet rating` | |
| `temperature.summary.torque_hot_day_note`, `temperature.magnet_life.torque_hot_day_check` | `Meets it nominally (no variation allowance)` | | `Below it` | |
| `temperature.demag.cold_check` | `OK` | | `Below the cold demagnetization limit` | `n/a (coercivity rises as the magnet cools)` |
| `temperature.adhesive.fatigue_screen` | `OK: Nx margin` | `CHECK` | | |
| `temperature.mismatch.reading` | `Below the lap-shear strength` | | `Above the lap-shear strength at the block ends` | |
| `temperature.adhesive_life.hot_fatigue_screen` | `OK` | `CHECK: get hot fatigue data` | | |
| `temperature.adhesive_life.daily_screen` | `Below the fatigue endurance` | `Above the fatigue endurance: qualify by thermal cycling` | | |
| `clamps.head_check` | `OK` | `Use a hardened washer` | | empty (no screw fits) |
| `clamps.vent_port` | `Yes: ...` | `No: key too large` | | empty (no screw fits) |
| `warnings.*` (6 rules) | | a caution's text | a warning's text | empty (the rule does not fire) |

At the defaults four checks fail, all red: `model.cup_ring_check`, `metal.hot_min_check`, `metal.clearance_check`, `materials.cup_wall_check`.

## Global Constraints

Every task's requirements implicitly include this section.

- The engine and the data are untouched: no file under `magcoupling-rs/src/engine/`, `magcoupling-rs/tests/`, `magcoupling-rs/Cargo.*` or `reference/` is edited; no dependency is added. Every engine test keeps Task 0's count.
- Byte compatibility (the user's request): `session::Design`, design files, share links and both results exports are unchanged. The input order, the filter texts, the results order, the open groups, the failing and traced filters and the trace live in `MagcouplingPanel` and `ResultsTable` only. The exports keep the engine's schema order (`results_csv`, `results_json` are not edited).
- `linkage-sim-rs` is not edited; its calculator window hosts the same panel, and its tests keep Task 0's counts (the embed still works).
- Formatting: `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --check` prints nothing after every task (every block below is already in rustfmt's layout). Never run `cargo fmt` in `linkage-sim-rs` (not rustfmt-clean at `746f73e`).
- Lints: magcoupling-rs clippy with warnings as errors stays clean (gates 5, 8 and 9: `--all-targets`, `--all-targets --features app`, wasm32 `--features gui --lib` and the web binary).
- UI text is ASCII (`->`, never an arrow glyph): `every_text_the_panel_shows_has_glyphs_in_the_default_fonts` checks every new text.
- Blocks: they quote the files with LF line endings, as git stores them (this machine's system gitconfig sets `core.autocrlf=true`; Task 0 makes the worktree LF). "Create `path`:" writes a new file with the block's text and a final newline. "In `path`, replace: ... with: ..." is one exact replacement: the old block occurs exactly once in the file at that point, as whole lines. "In `path`, replace (part of one line): ... with: ..." replaces text inside one long line (the README's and `03-structure.yaml`'s one-line table rows and entries), also occurring exactly once. Apply a step's blocks in the order given. If an old block is not found, stop and escalate; never improvise a match.
- No test in this plan reads a source file; a test that did would normalize CRLF (`.replace("\r\n", "\n")`).
- Gate runs (Task 0 and every task before its commit): `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- Windows paths: the worktree path is short on purpose (a build under a deep directory fails with `LNK1104`, MAX_PATH).
- Commits: subjects `feat(magcoupling-rs): ...` (Tasks 1 to 5) and `docs(magcoupling): ...` (Task 6). Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that writes the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The blocks write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the plan's writer); a `sonnet` implementer writes its own model's name there. Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`; the user's rule): Tasks 1 to 5 are intermediate commits on `magcoupling/ordering`; Task 6 updates `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` and the spec's M4 layout, and commits this plan; the branch is reviewed as a whole after Task 6. Read `docs/ai/*.yaml` first (Task 0).
- Model tiers (CLAUDE.md section 5): every task gives the exact code, so the implementations and the per-task reviews run on `sonnet` (`model: 'sonnet'`); Task 0 is commands and reporting (`sonnet`). A `sonnet` attempt that ends `blocked`, leaves a check red or is rejected in review escalates every retry to the session model. The whole-branch review after Task 6 runs on the session model (omit `model`, comment `// session model: final whole-branch review`). No task is physics: the engine is untouched, and the check levels (O-3) are a UX mapping of texts the engine already writes.

## Review Focus

The conditions the request implies but does not name that are most likely to bite a user, each pinned by tests in the task that owns the code:

1. **An input added to the engine later.** Expected: it cannot silently vanish from the workflow order. `every_input_is_in_exactly_one_workflow_section` (Task 1) fails and names every input without a section, every listed path that is no input, and every input listed twice.
2. **The Equation panel pointing at a row the user cannot see:** a leaf term naming an input under a closed Advanced heading, or one the filter hides. Expected: the group and its Advanced heading open, the filter clears, the row is framed and scrolled to. Test: `a_focused_advanced_input_opens_its_group_and_heading_and_clears_the_filter` (Task 2).
3. **A check text the badges do not know** (the engine gains a branch or a new check). Expected: no check loses its badge silently. `every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed` fails for a new `*_check`, `*_screen`, `*verdict`, `*reading` or warning result not in `CHECKS`, and `every_verdict_of_every_other_check_is_classified` reaches every branch of every check through a design, but for two texts no design reaches, which it checks as written: the vent port's "No: key too large" (no bore of the clamp model takes an M6 key) and the ferromagnetic sleeve warning (no sleeve of the material library is ferromagnetic); all six warning rules are covered (Task 3).
4. **The failing filter with a search typed.** Expected: only the failing checks the search matches, the red first, and when the search matches none of them, "No result matches the search.", not "No check fails or asks for a look.". Tests: `the_engine_order_and_the_failing_filter_are_flat` (Task 4: the search narrows the failing rows to the two amber rating checks), `the_failing_filter_shows_exactly_the_failing_checks_with_their_badges` (Task 4: exactly the failing labels, four red and two amber badges in the table, and the empty state with a search). With no failing check at all the table says "No check fails or asks for a look." with a count of 0; that text is glyph-checked, but no test draws it (no simple design passes every check).
5. **Clickable labels.** A click meant for a group heading or a slider that lands on a row's label now starts a trace (headings shift while a group animates open). Expected: only the label traces; a second click on it, Clear trace, or a click on a result outside an input's trace ends or replaces the trace; a trace changes no input and makes no undo step. Tests: `clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click` and `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group` (Task 5); the glyph test opens the groups bottom up so no click lands on a label (Task 2).
6. **Reading the results an input drives.** With an input traced and "Traced only" ticked, the user clicks the traced rows to read their equations. Expected: the equation opens and the input's trace stays (the filtered table keeps its rows); a click on a result without an equation record keeps the trace; Clear trace also unticks the filter, so a later click never filters the table unasked. Tests: `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows`, `an_input_whose_results_have_no_equation_record_traces_none_and_says_so` (Task 5).
7. **An input that reaches no explained result.** The registry explains 392 of the 1086 results; 27 inputs (the drive torque, the adhesive's, the adapter's) drive none of them. Expected: the banner says so instead of "drives 0 explained results", and "Traced only" says the trace marks nothing instead of blaming a search. Tests: `an_input_no_explained_result_reads_traces_none_and_says_so` (pins the 392), `an_input_whose_results_have_no_equation_record_traces_none_and_says_so` (Task 5).
8. **A failing check in a closed group.** The cup ring check is red at the defaults and sits in the closed "Coupling model" group. Expected: the group's heading shows the worst level of its checks. Test: `a_group_heading_opens_its_rows_and_the_engine_order_lists_them_without_headings` (Task 4: red on the headline and the coupling model, none on the mass).
9. **A heading clicked during a search.** A search opens every group, so the click would flip a group's state out of sight. Expected: the click is ignored (the hover text says to clear the search), and the group is as the user left it once the search is cleared. Test: `a_heading_click_while_searching_leaves_the_group_as_it_was` (Task 4).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-order/`):

| File | Responsibility | Task |
|---|---|---|
| `magcoupling-rs/src/gui/result_groups.rs` | the results table's groups: `ResultGroup`, `result_groups` (built once), `groups_of`, `CHAIN_LABELS` (the scope's six chains, adhesive, mass), `CHAIN_PREFIXES`, `PACKAGE_LABELS`, `HEADLINE_LABEL`, `OTHER_RESULTS`, `row_template`, `package_of`, `package_label` | 3 |
| `magcoupling-rs/src/gui/trace.rs` | `Trace` (`of`, `marks`, `count_in`, `banner`), `TraceKind`, `TRACING`, `CLEAR_TRACE` | 5 |

Modified:

| File | Change | Task |
|---|---|---|
| `magcoupling-rs/src/gui/inputs.rs` | `InputOrder`, `ADVANCED_HEADING`, `WorkflowSection`, `WorkflowGroup`, `WORKFLOW`; `InputSection.prefix` renamed `id`, `InputSection.advanced`; `InputCatalogue.workflow`, `groups_in`, `section_of` (Task 1); `InputEntry.haystack`, `FILTER_HINT`, `SectionMatches`, `filter_inputs` (Task 2) | 1, 2 |
| `magcoupling-rs/src/gui/format.rs` | `search_haystack`, `search_needle` | 2 |
| `magcoupling-rs/src/gui/panel.rs` | the `section.id` rename (Task 1); `input_order`, `input_filter`, the toggle row, the filter box, `section_ui`, `filtered_inputs_ui`, the Advanced heading and the focus opening it (`section_of`) (Task 2); `trace`, the marks, the banner, the traced counts (`count_in`), the label click, the readout click that keeps an input's trace (Task 5); tests in each | 1, 2, 4, 5 |
| `magcoupling-rs/src/gui/results_table.rs` | the search helpers (Task 2); `TableEntry.check`, `entry_index`, `ResultOrder`, `Line`, `table_lines`, the order toggle, the failing filter, the group headings with the worst check level (`HeadingLine`, ignored while searching: `CLEAR_TO_CLOSE`), the badges, the empty states (Task 4); the trace argument, "Traced only" for an input's trace (`NOTHING_TRACED`), the traced counts (Task 5) | 2, 4, 5 |
| `magcoupling-rs/src/gui/dashboard.rs` | `Level` ordered by severity; `CHECKS`, the 23 checks' levels in `verdict_level`, `check_level`, `failing_checks`; `badge` made `pub(crate)` | 3 |
| `magcoupling-rs/src/gui/typeset.rs` | `TermColors::with_marked` | 5 |
| `magcoupling-rs/src/gui/input_ui.rs` | the clickable label, `RowOutput.label_clicked`, `TRACE_HINT` | 5 |
| `magcoupling-rs/src/gui/mod.rs` | `pub mod result_groups;` (Task 3), `pub mod trace;` (Task 5) | 3, 5 |
| `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`, `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` | docs; the spec's M4 layout amendment | 6 |
| `docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md` | this plan | 6 |

Order: the input catalogue's data first (a mapping a reviewer can judge alone), then the inputs side that draws it, then the check levels and the results groups (logic), then the table that draws them, then tracing (it touches both sides), then the docs and the browser check.

## Verification record

This revision was replayed before it was handed over, as the first version was; nothing below is carried over from the first version's replay. The code was developed in a scratch git repository holding an LF export of `746f73e` (`git -c core.autocrlf=false archive 746f73e`) and one red and one green commit per task (the red commit is the task's tests, and its `mod.rs` line, on the previous task's code). This file's blocks were merged from that history block by block: a block whose diff the revision left alone kept its text, a block whose old or new text changed took the new text, and a change next to a block widened it or, away from every block, became a block of its own at its place in file order; each merged run was checked to turn its pre-state into its post-state. The blocks were then **parsed back out of this file and applied literally, task by task, to a fresh LF export of `746f73e`** (a scratch git repository with one commit per task, at `W:/replay`: `W:` is a `subst` drive for `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/order-plan`, for MAX_PATH): Step 1's blocks, Step 2's command, Step 3's blocks, then each step's commands as written with the worktree path replaced by the scratch path. After the replay only prose changed in this file (the expected outputs, this record and the Self-review record); the blocks are identical, checked by parsing both versions. The replayed `magcoupling-rs/src` equals the development tree's file for file, the docs are the Task 6 blocks applied, and the verification tree (`C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/order-plan/verify`) was refreshed from the replayed tree (the PNGs under `docs/chebyshev_lambda` aside, which the gate's linkage tests rewrite and the steps restore).

- **Task 0:** the baseline counts of Step 5 (magcoupling with `app`: library `449 passed`, every integration binary as listed; linkage library `955 passed`, every binary as listed); the gate passed (`1159 passed`, the differential data current, `GATE PASS`, no `SKIP gate`); `cargo fmt --check` clean.
- **Red steps failed as stated** (the exact error lines are in each Step 2), and **green steps passed** with the counts below. After every task `cargo fmt --check` printed nothing and the gate passed (`exit=0`, `1159 passed`, `GATE PASS`, no `SKIP gate`), so magcoupling clippy with warnings as errors (native all targets, with `app`, wasm32 `gui` and the web binary), the wasm32 checks, the workbook-parity guard with its negative controls, the linkage tests (the calculator window's included) and the oracle stayed green throughout. One fault of the machine, not of the plan: the first run of Task 5's gate stopped with `link.exe` exit code `0xc0000142` (a Windows DLL-initialization failure) while linking a linkage-sim-rs test binary, before any of its tests ran. Task 5 was then replayed again from Task 4's commit, with its zero-reach panel test extended (Self-review record, item 2), and that run is the one recorded here: its gate passed.

| After task | magcoupling `--features app --lib` | Red step (Step 2) | Green step (Step 4) |
|---|---|---|---|
| 0 (base) | 449 | — | — |
| 1 | 451 | 23 errors | 13 passed |
| 2 | 456 | 34 errors | 13 passed |
| 3 | 463 | 46 errors, 1 warning | 18 passed |
| 4 | 469 | 54 errors | 19 passed |
| 5 | 480 | 48 errors, 1 warning | 14 passed |
| 6 | 480 | — (`yaml ok` with the system Python, as Step 4 runs it; `git diff --check` clean) | — |

- **Task 6 Step 6, on the replayed tree** (a git worktree at its final commit): `build_web.sh` through both guards (`linkage-web builds without workbook-parity`, `magcoupling-web builds without workbook-parity`, both `complete!`): `linkage-web_bg.wasm` 12,113,610 bytes, `magcoupling-web_bg.wasm` 5,104,467 bytes. Served with `python -m http.server` and opened with Playwright at 1400 x 900: `/magcoupling/` showed "By workflow" selected, "Workbook groups", the filter box, the Key design group and the eight workflow groups closed, "Requirements and operating conditions" first (opened: its own rows, then "Space claim", "Drive and gearbox", "Duty and life"); Calibration and model opened onto its rows, "Prototype magnets", the three "3D results: ..." sections (the stored fields in view) and its closed Advanced heading. The Results table tab showed "By physics chain", "Engine order", "Failing checks only", "Traced only" (disabled), "1086 of 1086 results", "Headline (17)" open with a red badge after its heading and red and green badges on its check rows, the eight chain headings ("Torque (49)" to "Mass (5)", green badges after Temperature and Adhesive) and "Other results" with the package headings, "Coupling model (66)" with a red badge. A click on the Key design's face-gap label showed "Tracing Candidate flat-face magnetic gap: drives 146 explained results", framed the dashboard rows and table rows the gap drives (not the governing temperature limit or its margin, which read the stored 3D fields) and counted them per heading ("Headline (17, 13 traced)"); with "Traced only" ticked ("146 of 1086 results") a click on the pull-out's table row opened its equation and kept the trace and the 146 rows; Clear trace unticked and disabled the filter ("1086 of 1086 results"). `/?tool=magcoupling` showed the same inputs side in the linkage app's calculator window, and the same click traced there. Both pages logged 0 errors and 0 warnings. The server's Python process was stopped by its PID.

- **Mutation checks** (the replayed tree, each mutation restored after; every one failed the named tests): the nine of the first version, re-run on the revised code (a path dropped from `WORKFLOW`; the filter not cleared for a focused row, or the Advanced heading not opened for it; `Too thin` amber instead of red; the failing filter ignoring the search; a search not opening the groups; `with_marked` overwriting the equation's colours; the trace's source not marked; a clicked result not traced), and fourteen for this revision: a heading click toggling during a search (`a_heading_click_while_searching_leaves_the_group_as_it_was`); no badge after a heading (`a_group_heading_opens_its_rows_and_the_engine_order_lists_them_without_headings`); a click on a result inside an input's trace replacing it, and a click on a result without a record ending it (`reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows`, both); the trace filter left on after the trace (`an_input_whose_results_have_no_equation_record_traces_none_and_says_so`); the zero-reach banner dropped (that test and `an_input_no_explained_result_reads_traces_none_and_says_so`); `NOTHING_TRACED` dropped from the empty state (the panel test); "No check fails" shown for a search (`the_failing_filter_shows_exactly_the_failing_checks_with_their_badges`); the prefixes placing nothing (`every_chain_prefix_places_a_result_the_scope_leaves_out`, `the_headline_comes_first_then_the_chains_then_the_other_results`); "Other results" drawn once per package (`the_lines_group_the_rows_with_only_the_headline_open_at_first`); the inputs' haystack without the cell (`the_filter_matches_label_path_and_cell_by_section_in_the_order_shown`); `section_of` reading the workbook order (`the_workflow_groups_run_in_design_order_with_the_advanced_rows_last`, `a_focused_advanced_input_opens_its_group_and_heading_and_clears_the_filter`); the failing checks sorted green-first (`the_failing_checks_are_the_red_then_the_amber`); `count_in` counting every path (`a_trace_counts_the_paths_it_marks`, `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group`). Not mutated: the idle-frame test (it pins the absence of a write-back, and the revision added no writing code).

Not exercised by the replay: the native app by hand (the headless tests drive the same panel code), the `gui-smoke` workflow as a workflow (its `/magcoupling/` checks read the share link, sizing, design file, geometry view and explorer, which this plan does not change; the browser check above covers the new views), and the user's answers to the Decisions to confirm.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 8 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: `main` at `746f73e`, or a later commit that leaves the files this plan's blocks touch unchanged (Step 2).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-order` on the new branch `magcoupling/ordering` with LF line endings, a green baseline with its test counts, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 main
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/ordering C:/Users/Cole/source/repos/lsim-mag-order main
git -C C:/Users/Cole/source/repos/lsim-mag-order config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-order rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-order status --short
```

Expected: the `log` line is `746f73e docs(ai): next item, smarter ordering of the magcoupling inputs and outputs (user request)` or a later commit; `worktree add` prints `Preparing worktree (new branch 'magcoupling/ordering')`; then `magcoupling/ordering`; no status lines (the main checkout's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md` is not in the worktree).

- [ ] **Step 2: Check that the blocks' files are still those of `746f73e`**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order diff --stat 746f73e HEAD -- magcoupling-rs docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md
```

Expected: no output. If any file is listed, stop and escalate: the blocks that touch it must be re-derived first.

- [ ] **Step 3: Check the line endings**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order config --get core.autocrlf; head -c 3000 C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/src/gui/inputs.rs | tr -cd '\r' | wc -c
```

Expected: `false` and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-order rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-order reset -q --hard` and check again.

- [ ] **Step 4: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-order/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` (the item this plan resolves: "NEXT (user request 2026-10-02)"); `magcoupling-rs/README.md` ("The panel", the test table); the spec's M4 section; and the code this plan changes: `magcoupling-rs/src/gui/inputs.rs` (`InputCatalogue::new`, `KEY_DESIGN`, `SECTION_LABELS`), `input_ui.rs` (`input_row`), `panel.rs` (`ui`, `inputs_ui`, `design_inputs_ui`, `input_row_ui`, `centre_ui` and the test `Harness`), `results_table.rs` (`table_entries`, `search`, `ResultsTable::ui`, `row_ui`), `dashboard.rs` (`DASHBOARD`, `Level`, `verdict_level`, `badge`), `readouts.rs` (`Readouts::mark`), `typeset.rs` (`TermColors`), `explorer.rs` (`Explorer::marks`, `focus`, `term_label`) and `engine/explain/registry.rs` (`downstream`, `upstream_inputs`, `is_leaf_input`), `engine/explain/scope.rs` (`SCOPE`); and this plan's Decisions to confirm.

- [ ] **Step 5: Run both crates' tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features app 2>&1 | grep -E "Running|test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/Cargo.toml --all 2>&1 | grep -E "Running|test result"
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
```

Expected: magcoupling with `app`: the library `test result: ok. 449 passed`; the binaries `magcoupling_app` and `magcoupling_web` 0; `tests\assumptions.rs` 8; `deviations` 54; `differential` 19; `explain` 12 (2 ignored); `grades` 11; `material_library` 4; `material_links` 11; `parity` 4; `python_schema` 7; `robustness` 12; `schema` 7; `sizing` 31; `static_data` 8; doc-tests 1 (1 ignored). Then the linkage library `test result: ok. 955 passed`; the binaries `linkage_gui` and `linkage_web` 0; `tests\actuator_force_label.rs` 4; `braindump_repro` 1; `compound_actuator_rebuild` 5; `compound_force_integration` 9; `dxf_import_test` 1; `force_zone_tests` 8; `geometry_tests` 18; `golden_fixtures` 11; `gravity_breakdown_reference` 2; `mount_point_integration` 1; `parallelogram_actuator_sample` 4; `property_tests` 8; `singular_behavior` 18; doc-tests 0. If a count differs but every binary is `ok`, record the actual counts and read every later count as an offset from them.

- [ ] **Step 6: Check the oracle Python and the toolchain**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
wasm-bindgen --version
rustc --version
```

Expected: the path, then `wasm-bindgen 0.2.114`, then `rustc 1.89.0 (29483883e 2025-08-04)`. If the Python is missing, stop and escalate: every gate run uses it.

- [ ] **Step 7: Run the gate**

Run:

```bash
mkdir -p C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task0.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are `1159 passed in ...s`, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0`; `cargo fmt --check` prints nothing.

- [ ] **Step 8: Decisions**

The controller asks the user the plan's **Decisions to confirm** (O-1 to O-8, with the three tables under them) before Task 1 and records the answers in the execution notes. Every task implements the recommended option and names the decision where it implements it; if the user picks another option, the task that implements it stops and escalates instead of improvising.

---

### Task 1: The workflow arrangement of the inputs

**Model:** `sonnet` (the plan gives the exact code: transcription plus running commands; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/src/gui/inputs.rs` (module doc; after `SECTION_LABELS`: `InputOrder`, `ADVANCED_HEADING`, `WorkflowSection`, `WorkflowGroup`, `WORKFLOW`; `InputSection`, `InputGroup`, `InputCatalogue`; `InputCatalogue::new`, `groups_in`, `section_of`; tests)
- Modify: `magcoupling-rs/src/gui/panel.rs` (`design_inputs_ui`: `section.prefix` is now `section.id`)

**Interfaces:**
- Consumes: `input_rows(&DesignInputs::default())` (engine metadata, one `InputRow` per input).
- Produces: `pub enum InputOrder { Workflow (default), Workbook }` with `ALL` and `const fn label(self) -> &'static str` ("By workflow", "Workbook groups"); `pub const ADVANCED_HEADING: &str = "Advanced"`; `pub struct WorkflowSection { id, label, advanced: bool, paths: &'static [&'static str] }`; `pub struct WorkflowGroup { id, label, sections: &'static [WorkflowSection] }`; `pub const WORKFLOW: [WorkflowGroup; 8]`; `InputSection { id: String, label, advanced: bool, entries }` (the field `prefix` is renamed `id`); `InputCatalogue { key_design, groups, workflow: Vec<InputGroup> }`, `fn groups_in(&self, order: InputOrder) -> &[InputGroup]` and `fn section_of(&self, order: InputOrder, path: &str) -> Option<(&InputGroup, &InputSection)>` (the one place that finds an input's group and section: the tests here, the focus in Task 2). A section whose `id` equals its group's `name` is drawn without a heading (unchanged rule).

- [ ] **Step 1: Write the failing tests**

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
        for group in &catalogue.groups {
            for section in &group.sections {
                assert!(
                    section.prefix.starts_with(&group.name),
                    "{}",
                    section.prefix
                );
                assert!(!section.entries.is_empty());
                for entry in &section.entries {
                    assert_eq!(entry.path.rsplit_once('.').unwrap().0, section.prefix);
                }
            }
        }
    }
````

with:

````rust
        for group in &catalogue.groups {
            for section in &group.sections {
                assert!(section.id.starts_with(&group.name), "{}", section.id);
                assert!(!section.entries.is_empty());
                assert!(!section.advanced, "no workbook section is advanced");
                for entry in &section.entries {
                    assert_eq!(entry.path.rsplit_once('.').unwrap().0, section.id);
                }
            }
        }
        assert_eq!(
            catalogue.groups_in(InputOrder::Workbook),
            &catalogue.groups[..]
        );
    }

    #[test]
    fn every_input_is_in_exactly_one_workflow_section() {
        // Fails when an input is added to the engine without a place in WORKFLOW, or a path
        // there names no input: the message lists each one.
        let schema: Vec<String> = input_rows(&DesignInputs::default())
            .into_iter()
            .map(|r| r.path)
            .collect();
        let listed: Vec<&str> = WORKFLOW
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.paths.iter().copied())
            .collect();
        let missing: Vec<&str> = schema
            .iter()
            .map(String::as_str)
            .filter(|path| !listed.contains(path))
            .collect();
        assert!(
            missing.is_empty(),
            "inputs without a workflow section: {missing:?}"
        );
        let unknown: Vec<&str> = listed
            .iter()
            .copied()
            .filter(|path| !schema.iter().any(|s| s == path))
            .collect();
        assert!(
            unknown.is_empty(),
            "workflow paths that are no input: {unknown:?}"
        );
        let mut twice: Vec<&str> = listed
            .iter()
            .copied()
            .filter(|path| listed.iter().filter(|p| *p == path).count() > 1)
            .collect();
        twice.dedup();
        assert!(
            twice.is_empty(),
            "inputs in two workflow sections: {twice:?}"
        );
        assert_eq!(listed.len(), schema.len());
        // The catalogue's workflow arrangement is WORKFLOW, entry for entry, and each entry is
        // the workbook arrangement's.
        let catalogue = InputCatalogue::new();
        let arranged: Vec<&str> = catalogue
            .groups_in(InputOrder::Workflow)
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.entries.iter())
            .map(|e| e.path.as_str())
            .collect();
        assert_eq!(arranged, listed);
        for entry in catalogue
            .workflow
            .iter()
            .flat_map(|g| g.sections.iter())
            .flat_map(|s| s.entries.iter())
        {
            assert_eq!(Some(entry), catalogue.entry(&entry.path));
        }
    }

    #[test]
    fn the_workflow_groups_run_in_design_order_with_the_advanced_rows_last() {
        let labels: Vec<&str> = WORKFLOW.iter().map(|g| g.label).collect();
        assert_eq!(
            labels,
            [
                "Requirements and operating conditions",
                "Magnets and rings",
                "Gap and clearances",
                "Housing and retainers",
                "Shaft, key and clamps",
                "Materials",
                "Thermal and demagnetization",
                "Calibration and model",
            ]
        );
        let mut ids: Vec<&str> = Vec::new();
        for group in &WORKFLOW {
            // The first section is the group's own (drawn without a heading), never advanced.
            let first = &group.sections[0];
            assert_eq!(first.id, group.id);
            assert!(!first.advanced, "{}", group.id);
            let mut seen_advanced = false;
            for section in group.sections {
                assert!(!section.paths.is_empty(), "{}", section.id);
                if section.id != group.id {
                    assert!(
                        section.id.starts_with(&format!("{}.", group.id)),
                        "{}",
                        section.id
                    );
                }
                // The Advanced heading closes the group: no ordinary section after it.
                assert!(!seen_advanced || section.advanced, "{}", section.id);
                seen_advanced |= section.advanced;
                assert!(section.label.is_ascii(), "{}", section.label);
                ids.push(section.id);
            }
            assert!(group.label.is_ascii(), "{}", group.label);
        }
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "section ids are unique");
        // Every Key design input is in an ordinary section, so its group shows it.
        let catalogue = InputCatalogue::new();
        let section = |path: &str| {
            catalogue
                .section_of(InputOrder::Workflow, path)
                .unwrap_or_else(|| panic!("{path} is in no workflow section"))
                .1
        };
        for path in KEY_DESIGN {
            assert!(!section(path).advanced, "{path}");
        }
        // The rows a designer rarely changes are advanced (decision O-4): fit and screw
        // factors, physical and model constants, the optional adapter and its joint. The
        // clamp's two assumptions and the fields stored from a 3D run stay in view: they scale
        // the clamp capacity, and the stored fields must be refreshed after a geometry change.
        let advanced = |path: &str| section(path).advanced;
        for path in [
            "coupling.mu0",
            "calibration.mu0",
            "coupling.max_harmonic",
            "clamps.nut_factor",
            "clamps.joint_friction",
        ] {
            assert!(advanced(path), "{path}");
        }
        for path in [
            "clamps.friction",
            "clamps.preload_fraction",
            "temperature.demag.h_rev_pullout_kA_m",
            "temperature.slip_loss.b_hub_free_T",
        ] {
            assert!(!advanced(path), "{path}");
        }
        let rows = |advanced: bool| {
            WORKFLOW
                .iter()
                .flat_map(|g| g.sections.iter())
                .filter(|s| s.advanced == advanced)
                .map(|s| s.paths.len())
                .sum::<usize>()
        };
        assert_eq!(rows(true), 27, "decision O-4: 27 advanced rows");
        // Related inputs share a section (decision O-2): the adhesive and its bondlines, the
        // optional adapter and its joint; the 3D run's stored results sit in one group.
        for related in [
            &[
                "temperature.adhesive.selected",
                "metal.bond_inner_mm",
                "temperature.mismatch.recommended_bondline_mm",
                "temperature.mismatch.adhesive_shear_modulus_GPa",
                "temperature.adhesive_life.fatigue_endurance",
            ][..],
            &[
                "metal.adapter_flange_dia_mm",
                "clamps.joint_screws",
                "clamps.joint_friction",
            ],
        ] {
            for path in related {
                assert_eq!(section(path).id, section(related[0]).id, "{path}");
            }
        }
        for path in [
            "calibration.fea_torque1_Nm",
            "temperature.demag.h_rev_aligned_kA_m",
            "temperature.slip_loss.b_magnet_T",
        ] {
            let (group, _) = catalogue.section_of(InputOrder::Workflow, path).unwrap();
            assert_eq!(group.name, "calibration", "{path}");
        }
        assert_eq!(
            catalogue.section_of(InputOrder::Workflow, "no.such.input"),
            None
        );
        assert!(!advanced("metal.face_gap_mm"));
        assert_eq!(InputOrder::default(), InputOrder::Workflow);
        assert_eq!(ADVANCED_HEADING, "Advanced");
    }
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
            .iter()
            .flat_map(|g| {
                std::iter::once(g.name.as_str()).chain(g.sections.iter().map(|s| s.prefix.as_str()))
            })
            .collect();
````

with:

````rust
            .iter()
            .flat_map(|g| {
                std::iter::once(g.name.as_str()).chain(g.sections.iter().map(|s| s.id.as_str()))
            })
            .collect();
````

- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib gui::inputs 2>&1 | grep -E "^error" | sort | uniq
```

Expected: FAIL to compile, with these lines:

```
error: could not compile `magcoupling-rs` (lib test) due to 23 previous errors
error[E0425]: cannot find value `ADVANCED_HEADING` in this scope
error[E0425]: cannot find value `WORKFLOW` in this scope
error[E0433]: failed to resolve: use of undeclared type `InputOrder`
error[E0599]: no method named `groups_in` found for struct `gui::inputs::InputCatalogue` in the current scope
error[E0599]: no method named `section_of` found for struct `gui::inputs::InputCatalogue` in the current scope
error[E0609]: no field `advanced` on type `&InputSection`
error[E0609]: no field `id` on type `&InputSection`
error[E0609]: no field `workflow` on type `gui::inputs::InputCatalogue`
```

- [ ] **Step 3: Add the workflow arrangement**

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
//!
//! [`InputCatalogue`] is built once from the engine's input metadata: the Key design group
//! ([`KEY_DESIGN`]), then every input in its package group (`coupling`, `metal`,
//! `calibration`, `materials`, `temperature`, `clamps`), each group split into sections by
//! its nested input groups (`coupling.magnets`, `temperature.demag`, ...). Every input is in
//! exactly one section; the Key design inputs are also in their group (decision M41-13), and
//! both rows edit the same value.

use std::sync::OnceLock;
````

with:

````rust
//!
//! [`InputCatalogue`] is built once from the engine's input metadata: the Key design group
//! ([`KEY_DESIGN`]), then every input arranged two ways ([`InputOrder`]). The workflow order
//! (the default, decision O-1) follows [`WORKFLOW`]: the groups in the order a designer works,
//! each with its rarely changed rows under a collapsed Advanced heading. The workbook order puts
//! every input in its package group (`coupling`, `metal`, `calibration`, `materials`,
//! `temperature`, `clamps`), each group split into sections by its nested input groups
//! (`coupling.magnets`, `temperature.demag`, ...). In either order every input is in exactly
//! one section; the Key design inputs are also in their group (decision M41-13), and both rows
//! edit the same value.

use std::sync::OnceLock;
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
];

/// Where an optional input starts when the user enters a value (decision M41-12): the result
/// it overrides, unrounded. The axial length override enters at the inner ring's length in
````

with:

````rust
];

/// How the left side orders the design inputs (decision O-1): by design workflow (the default)
/// or as the workbook's package groups. A per-session choice of view: no design file or share
/// link holds it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputOrder {
    #[default]
    Workflow,
    Workbook,
}

impl InputOrder {
    /// Both orders, in toggle order.
    pub const ALL: [InputOrder; 2] = [InputOrder::Workflow, InputOrder::Workbook];

    /// The toggle's text.
    pub const fn label(self) -> &'static str {
        match self {
            InputOrder::Workflow => "By workflow",
            InputOrder::Workbook => "Workbook groups",
        }
    }
}

/// The heading the advanced sections of a workflow group sit under, closed by default.
pub const ADVANCED_HEADING: &str = "Advanced";

/// One section of a workflow group ([`WORKFLOW`]).
#[derive(Clone, Copy, Debug)]
pub struct WorkflowSection {
    /// `<group>.<section>`; the group's id for its first section, drawn without a heading.
    pub id: &'static str,
    pub label: &'static str,
    /// Under the group's [`ADVANCED_HEADING`]: rows a designer rarely changes (fit and screw
    /// factors, physical and model constants, the optional adapter).
    pub advanced: bool,
    /// Its inputs, in the order a designer sets them.
    pub paths: &'static [&'static str],
}

/// One workflow group: an id, a heading and its sections, the advanced ones last.
#[derive(Clone, Copy, Debug)]
pub struct WorkflowGroup {
    pub id: &'static str,
    pub label: &'static str,
    pub sections: &'static [WorkflowSection],
}

/// The workflow groups, in the order a designer works (decision O-2): the requirements and
/// operating conditions first (the spec fixed before a magnet is chosen: the torque, the
/// temperatures, the space claim, the drive and the duty), then the magnets and their rings, the
/// gap, the housing around them, the shaft and its clamps, the materials (the adhesive and its
/// bondlines among them), the thermal and demagnetization inputs, then the calibration, the
/// results stored from a 3D run and the model's constants. Every input sits in exactly one
/// section (a test lists any that does not); the Key design inputs also sit on top, as in the
/// workbook order.
pub const WORKFLOW: [WorkflowGroup; 8] = [
    WorkflowGroup {
        id: "operating",
        label: "Requirements and operating conditions",
        sections: &[
            WorkflowSection {
                id: "operating",
                label: "Requirements and operating conditions",
                advanced: false,
                paths: &[
                    "metal.required_min_Nm",
                    "coupling.op_temp_C",
                    "metal.min_temp_C",
                    "metal.variation",
                ],
            },
            WorkflowSection {
                id: "operating.envelope",
                label: "Space claim",
                advanced: false,
                paths: &[
                    "metal.max_diameter_mm",
                    "metal.max_overall_axial_mm",
                    "metal.max_large_dia_axial_mm",
                ],
            },
            WorkflowSection {
                id: "operating.drive",
                label: "Drive and gearbox",
                advanced: false,
                paths: &[
                    "coupling.drive_torque_Nm",
                    "coupling.drive_safety_factor",
                    "coupling.gear_ratio",
                    "coupling.gear_efficiency",
                    "coupling.gearbox_input_rating_Nm",
                ],
            },
            WorkflowSection {
                id: "operating.duty",
                label: "Duty and life",
                advanced: false,
                paths: &[
                    "temperature.duty.wheel_rotor_rpm",
                    "temperature.duty.hot_ambient_C",
                    "temperature.duty.driving_rise_C",
                    "temperature.duty.life_hours",
                    "temperature.adhesive_life.service_years",
                    "temperature.adhesive_life.daily_swing_C",
                ],
            },
            WorkflowSection {
                id: "operating.slip",
                label: "Slip events",
                advanced: false,
                paths: &[
                    "metal.slip_rpm",
                    "metal.slip_event_s",
                    "metal.life_events",
                    "temperature.duty.fault_trip_s",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "magnets",
        label: "Magnets and rings",
        sections: &[
            WorkflowSection {
                id: "magnets",
                label: "Magnets and rings",
                advanced: false,
                paths: &[
                    "coupling.npole",
                    "coupling.magnets.part_inner",
                    "coupling.magnets.part_outer",
                    "coupling.magnets.axial_length_mm",
                    "coupling.faceted",
                    "coupling.inner_back_apothem_mm",
                ],
            },
            WorkflowSection {
                id: "magnets.grades",
                label: "Grades (manual dimensions)",
                advanced: false,
                paths: &[
                    "coupling.magnets.grade_inner",
                    "coupling.magnets.grade_outer",
                ],
            },
            WorkflowSection {
                id: "magnets.manual_inner",
                label: "Manual inner blocks",
                advanced: false,
                paths: &[
                    "coupling.magnets.manual_inner_length_mm",
                    "coupling.magnets.manual_inner_width_mm",
                    "coupling.magnets.manual_inner_thickness_mm",
                    "coupling.magnets.manual_inner_br_T",
                ],
            },
            WorkflowSection {
                id: "magnets.manual_outer",
                label: "Manual outer blocks",
                advanced: false,
                paths: &[
                    "coupling.magnets.manual_outer_length_mm",
                    "coupling.magnets.manual_outer_width_mm",
                    "coupling.magnets.manual_outer_thickness_mm",
                    "coupling.magnets.manual_outer_br_T",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "gap",
        label: "Gap and clearances",
        sections: &[
            WorkflowSection {
                id: "gap",
                label: "Gap and clearances",
                advanced: false,
                paths: &[
                    "metal.face_gap_mm",
                    "metal.sleeve_mm",
                    "metal.liner_mm",
                    "metal.residual_target_mm",
                ],
            },
            WorkflowSection {
                id: "gap.allowances",
                label: "Running-clearance allowances",
                advanced: false,
                paths: &[
                    "metal.shaft_displacement_mm",
                    "metal.runout_mm",
                    "metal.deflection_mm",
                    "metal.thermal_mm",
                    "metal.sleeve_form_mm",
                    "metal.magnet_position_mm",
                ],
            },
            WorkflowSection {
                id: "gap.bedding",
                label: "Bedding clearances",
                advanced: true,
                paths: &["metal.sleeve_bedding_mm", "metal.liner_bedding_mm"],
            },
        ],
    },
    WorkflowGroup {
        id: "housing",
        label: "Housing and retainers",
        sections: &[
            WorkflowSection {
                id: "housing",
                label: "Housing and retainers",
                advanced: false,
                paths: &[
                    "coupling.backiron",
                    "metal.cup_wall_corner_mm",
                    "metal.hub_length_mm",
                    "metal.cup_depth_mm",
                    "metal.web_mm",
                    "metal.boss_length_mm",
                    "metal.boss_od_mm",
                ],
            },
            WorkflowSection {
                id: "housing.cap",
                label: "Front cap",
                advanced: false,
                paths: &[
                    "metal.cap_axial_mm",
                    "metal.cap_od_mm",
                    "metal.cap_thread_dia_mm",
                    "metal.cap_thread_engagement_mm",
                ],
            },
            WorkflowSection {
                id: "housing.retainers",
                label: "Endplates and retainers",
                advanced: false,
                paths: &[
                    "metal.front_endplate_mm",
                    "metal.rear_endplate_mm",
                    "metal.retainer_span_mm",
                    "metal.rear_endplate_hole_mm",
                ],
            },
            WorkflowSection {
                id: "housing.adapter",
                label: "Optional adapter and its joint",
                advanced: true,
                paths: &[
                    "metal.adapter_flange_dia_mm",
                    "metal.adapter_flange_mm",
                    "metal.adapter_pilot_dia_mm",
                    "metal.adapter_pilot_mm",
                    "metal.adapter_boss_mm",
                    "metal.adapter_hardware_g",
                    "clamps.joint_screws",
                    "clamps.joint_bolt_circle_mm",
                    "clamps.joint_friction",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "shaft",
        label: "Shaft, key and clamps",
        sections: &[
            WorkflowSection {
                id: "shaft",
                label: "Shaft, key and clamps",
                advanced: false,
                paths: &[
                    "coupling.bore_mm",
                    "coupling.keyway_depth_mm",
                    "clamps.key_width_mm",
                    "clamps.key_contact_mm",
                ],
            },
            WorkflowSection {
                id: "shaft.clamp",
                label: "Clamp",
                advanced: false,
                paths: &[
                    "clamps.clamp_type",
                    "clamps.safety_factor",
                    "clamps.screw_class",
                    "clamps.alloy",
                    "clamps.boss_od_mm",
                    "clamps.clamp_length_mm",
                    "clamps.friction",
                    "clamps.preload_fraction",
                ],
            },
            WorkflowSection {
                id: "shaft.clamp_geometry",
                label: "Clamp geometry",
                advanced: false,
                paths: &[
                    "clamps.slit_mm",
                    "clamps.ligament_mm",
                    "clamps.wall_out_mm",
                    "clamps.grip_min_mm",
                    "clamps.axial_margin_mm",
                    "clamps.relief_mm",
                    "clamps.hinge_mm",
                ],
            },
            WorkflowSection {
                id: "shaft.screw_model",
                label: "Clamp and screw factors",
                advanced: true,
                paths: &[
                    "clamps.factor_one_piece",
                    "clamps.factor_two_piece",
                    "clamps.nut_factor",
                    "clamps.engagement_x_d",
                    "clamps.strip_sf",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "materials",
        label: "Materials",
        sections: &[
            WorkflowSection {
                id: "materials",
                label: "Materials",
                advanced: false,
                paths: &[
                    "materials.parts.back_iron",
                    "materials.parts.sleeve_liner",
                    "materials.parts.cap_housing",
                ],
            },
            WorkflowSection {
                id: "materials.steel",
                label: "Back-iron steel",
                advanced: false,
                paths: &[
                    "materials.steel.bsat_T",
                    "materials.steel.conductivity_S_m",
                    "materials.steel.mu_r_incremental",
                    "materials.steel.specific_heat_J_kgK",
                    "materials.steel.cte_per_C",
                    "materials.steel.modulus_GPa",
                    "materials.steel.density_g_cm3",
                ],
            },
            WorkflowSection {
                id: "materials.magnet",
                label: "Magnet properties",
                advanced: false,
                paths: &[
                    "temperature.mismatch.ndfeb_cte_per_C",
                    "temperature.mismatch.ndfeb_modulus_GPa",
                    "temperature.slip_loss.sigma_ndfeb_S_m",
                    "temperature.thermal.c_ndfeb",
                ],
            },
            WorkflowSection {
                id: "materials.adhesive",
                label: "Adhesive and bondlines",
                advanced: false,
                paths: &[
                    "temperature.adhesive.selected",
                    "metal.bond_inner_mm",
                    "metal.bond_outer_mm",
                    "temperature.mismatch.recommended_bondline_mm",
                    "temperature.mismatch.adhesive_shear_modulus_GPa",
                    "temperature.adhesive_life.hot_strength_retained",
                    "temperature.adhesive_life.fatigue_endurance",
                ],
            },
            WorkflowSection {
                id: "materials.other",
                label: "Other part properties",
                advanced: false,
                paths: &[
                    "temperature.slip_loss.sigma_316_S_m",
                    "temperature.thermal.c_316",
                    "temperature.thermal.c_aluminium",
                    "materials.nickel.thickness_mm",
                ],
            },
            WorkflowSection {
                id: "materials.mass",
                label: "Densities and mass allowance",
                advanced: false,
                paths: &[
                    "metal.steel_density_g_mm3",
                    "metal.sleeve_density_g_mm3",
                    "metal.al_density_g_mm3",
                    "metal.hardware_g",
                ],
            },
            WorkflowSection {
                id: "materials.screws",
                label: "Screw classes",
                advanced: true,
                paths: &[
                    "materials.screws.proof_12_9_MPa",
                    "materials.screws.proof_10_9_MPa",
                    "materials.screws.yield_A4_70_MPa",
                ],
            },
        ],
    },
    WorkflowGroup {
        id: "thermal",
        label: "Thermal and demagnetization",
        sections: &[
            WorkflowSection {
                id: "thermal",
                label: "Thermal and demagnetization",
                advanced: false,
                paths: &[
                    "temperature.thermal.conductance_W_K",
                    "metal.measured_drag_Nm",
                    "temperature.slip_loss.high_multiplier",
                ],
            },
            WorkflowSection {
                id: "thermal.demag",
                label: "Demagnetization",
                advanced: false,
                paths: &[
                    "temperature.demag.coercivity_source",
                    "temperature.demag.hcj20_kA_m",
                    "temperature.demag.beta_hcj_per_C",
                    "temperature.demag.knee_fraction",
                    "temperature.demag.design_margin_C",
                ],
            },
            WorkflowSection {
                id: "thermal.slip_model",
                label: "Slip-loss model",
                advanced: true,
                paths: &["temperature.slip_loss.end_factor"],
            },
        ],
    },
    WorkflowGroup {
        id: "calibration",
        label: "Calibration and model",
        sections: &[
            WorkflowSection {
                id: "calibration",
                label: "Calibration and model",
                advanced: false,
                paths: &[
                    "calibration.measured_torque_Nm",
                    "calibration.test_temp_C",
                    "calibration.spacing_mm",
                    "calibration.gap_definition",
                    "calibration.total_magnets",
                ],
            },
            WorkflowSection {
                id: "calibration.prototype",
                label: "Prototype magnets",
                advanced: false,
                paths: &[
                    "calibration.apothem_mm",
                    "calibration.magnet_length_mm",
                    "calibration.magnet_width_mm",
                    "calibration.magnet_thickness_mm",
                    "calibration.br_T",
                ],
            },
            WorkflowSection {
                id: "calibration.fea",
                label: "3D results: reference torques",
                advanced: false,
                paths: &["calibration.fea_torque1_Nm", "calibration.fea_torque2_Nm"],
            },
            WorkflowSection {
                id: "calibration.stored_demag",
                label: "3D results: stored reverse fields (refresh after a 3D run)",
                advanced: false,
                paths: &[
                    "temperature.demag.h_rev_aligned_kA_m",
                    "temperature.demag.h_rev_pullout_kA_m",
                    "temperature.demag.h_rev_likepole_kA_m",
                    "temperature.demag.h_rev_single_ring_kA_m",
                ],
            },
            WorkflowSection {
                id: "calibration.stored_slip",
                label: "3D results: stored slip-loss fields (refresh after a 3D run)",
                advanced: false,
                paths: &[
                    "temperature.slip_loss.b_hub_T",
                    "temperature.slip_loss.b_cup_T",
                    "temperature.slip_loss.b_sleeve_T",
                    "temperature.slip_loss.b_liner_T",
                    "temperature.slip_loss.cap_integral_T2m4",
                    "temperature.slip_loss.web_integral_T2m2",
                    "temperature.slip_loss.b_magnet_T",
                    "temperature.slip_loss.b_hub_free_T",
                    "temperature.slip_loss.b_cup_free_T",
                    "temperature.slip_loss.web_integral_free_T2m2",
                ],
            },
            WorkflowSection {
                id: "calibration.model",
                label: "Model constants",
                advanced: true,
                paths: &[
                    "coupling.c_end",
                    "coupling.max_harmonic",
                    "coupling.mu0",
                    "calibration.alpha_br_per_C",
                    "calibration.c_end",
                    "calibration.f_cal_original",
                    "calibration.mu0",
                ],
            },
        ],
    },
];

/// Where an optional input starts when the user enters a value (decision M41-12): the result
/// it overrides, unrounded. The axial length override enters at the inner ring's length in
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
}

/// A run of inputs under one heading: a group's own fields, or one of its nested groups.
#[derive(Clone, Debug, PartialEq)]
pub struct InputSection {
    /// The path prefix (`coupling`, `coupling.magnets`).
    pub prefix: String,
    pub label: &'static str,
    pub entries: Vec<InputEntry>,
}

/// A package group (`coupling`, ...) and its sections, in schema order.
#[derive(Clone, Debug, PartialEq)]
pub struct InputGroup {
````

with:

````rust
}

/// A run of inputs under one heading: in the workbook order a group's own fields or one of its
/// nested groups, in the workflow order a [`WorkflowSection`].
#[derive(Clone, Debug, PartialEq)]
pub struct InputSection {
    /// The workbook order's path prefix (`coupling`, `coupling.magnets`), or the workflow
    /// section's id (`magnets`, `magnets.grades`). A section whose id is its group's name is
    /// drawn without a heading.
    pub id: String,
    pub label: &'static str,
    /// Under the group's [`ADVANCED_HEADING`] (workflow order only).
    pub advanced: bool,
    pub entries: Vec<InputEntry>,
}

/// A group of the left side and its sections: a package group (`coupling`, ...) in schema
/// order, or a workflow group ([`WORKFLOW`]).
#[derive(Clone, Debug, PartialEq)]
pub struct InputGroup {
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
    /// [`KEY_DESIGN`], in order.
    pub key_design: Vec<InputEntry>,
    /// Every input, by package group and section, in schema order.
    pub groups: Vec<InputGroup>,
}
````

with:

````rust
    /// [`KEY_DESIGN`], in order.
    pub key_design: Vec<InputEntry>,
    /// Every input, by package group and section, in schema order (the workbook order).
    pub groups: Vec<InputGroup>,
    /// Every input, by workflow group and section ([`WORKFLOW`], the workflow order).
    pub workflow: Vec<InputGroup>,
}
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
            }
            let group = groups.last_mut().expect("pushed above");
            if group.sections.last().is_none_or(|s| s.prefix != prefix) {
                group.sections.push(InputSection {
                    prefix: prefix.to_owned(),
                    label: section_label(prefix).unwrap_or("Inputs"),
                    entries: Vec::new(),
                });
````

with:

````rust
            }
            let group = groups.last_mut().expect("pushed above");
            if group.sections.last().is_none_or(|s| s.id != prefix) {
                group.sections.push(InputSection {
                    id: prefix.to_owned(),
                    label: section_label(prefix).unwrap_or("Inputs"),
                    advanced: false,
                    entries: Vec::new(),
                });
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
            section.entries.push(entry(row));
        }
        Self { key_design, groups }
    }
````

with:

````rust
            section.entries.push(entry(row));
        }
        let workflow = WORKFLOW
            .iter()
            .map(|group| InputGroup {
                name: group.id.to_owned(),
                label: group.label,
                sections: group
                    .sections
                    .iter()
                    .map(|section| InputSection {
                        id: section.id.to_owned(),
                        label: section.label,
                        advanced: section.advanced,
                        entries: section
                            .paths
                            .iter()
                            .map(|&path| {
                                let row = rows.iter().find(|row| row.path == path);
                                entry(row.unwrap_or_else(|| panic!("WORKFLOW: no input {path}")))
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect();
        Self {
            key_design,
            groups,
            workflow,
        }
    }

    /// The groups of `order`: the workflow groups or the package groups.
    pub fn groups_in(&self, order: InputOrder) -> &[InputGroup] {
        match order {
            InputOrder::Workflow => &self.workflow,
            InputOrder::Workbook => &self.groups,
        }
    }

    /// The group and the section of `order` that hold the input at `path`; `None` for a path
    /// that is no input.
    pub fn section_of(
        &self,
        order: InputOrder,
        path: &str,
    ) -> Option<(&InputGroup, &InputSection)> {
        self.groups_in(order).iter().find_map(|group| {
            group
                .sections
                .iter()
                .find(|section| section.entries.iter().any(|e| e.path == path))
                .map(|section| (group, section))
        })
    }
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                        .show(ui, |ui| {
                            for section in &group.sections {
                                if section.prefix != group.name {
                                    ui.add_space(4.0);
                                    ui.strong(section.label);
````

with:

````rust
                        .show(ui, |ui| {
                            for section in &group.sections {
                                if section.id != group.name {
                                    ui.add_space(4.0);
                                    ui.strong(section.label);
````

- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib gui::inputs 2>&1 | grep -E "^test |test result"
```

Expected: 13 lines `test gui::inputs::tests::<name> ... ok` (in any order), among them `every_input_is_in_exactly_one_workflow_section`, `the_workflow_groups_run_in_design_order_with_the_advanced_rows_last`, `every_input_is_in_exactly_one_section_in_schema_order` and `every_section_has_a_heading_and_every_heading_a_section`, then `test result: ok. 13 passed; 0 failed`.

- [ ] **Step 5: Run the checks and the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --check
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED"
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task1.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task1.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
```

Expected: `cargo fmt --check` prints nothing; `test result: ok. 451 passed; 0 failed` (Task 0's 449 and the two new tests); `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order add magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-order commit -q -F - <<'EOF'
feat(magcoupling-rs): the inputs' workflow arrangement, every input in one section (O-1, O-2, O-4)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-order status --short
```

Expected: no status lines.

---

### Task 2: The inputs side: order toggle, Advanced headings, filter box

**Model:** `sonnet` (exact code; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/src/gui/format.rs` (`search_haystack`, `search_needle` after `with_unit`; a test)
- Modify: `magcoupling-rs/src/gui/results_table.rs` (imports; `table_entries` and `search` use the shared helpers)
- Modify: `magcoupling-rs/src/gui/inputs.rs` (import; `InputEntry.haystack`, built once in `InputCatalogue::new`; `FILTER_HINT`, `SectionMatches`, `filter_inputs` after `impl Default for InputCatalogue`; a test)
- Modify: `magcoupling-rs/src/gui/panel.rs` (imports; fields `input_order`, `input_filter`; `design_inputs_ui` rewritten (the focused row's group through `section_of`), `section_ui`, `filtered_inputs_ui`; tests: `idle_frames_change_no_input` (the workbook order, then the workflow order's filtered view of every row), `every_group_opens_and_draws_a_row_for_each_of_its_inputs`, three new tests, the glyph test, the grade-picker test)

**Interfaces:**
- Consumes (Task 1): `InputOrder`, `ADVANCED_HEADING`, `InputCatalogue::groups_in`, `InputCatalogue::section_of`, `InputCatalogue::workflow`, `InputSection { id, advanced, .. }`, `WORKFLOW`.
- Produces: `pub(crate) fn search_haystack(label: &str, path: &str, cell: Option<&str>) -> String` and `pub(crate) fn search_needle(query: &str) -> String` (in `gui::format`); `pub const FILTER_HINT: &str = "Filter inputs by label, path or cell"`; `pub struct SectionMatches<'a> { group: &'a InputGroup, section: &'a InputSection, entries: Vec<&'a InputEntry> }` with `fn heading(&self) -> String` ("Group / Section", or the group's label for its own section); `pub fn filter_inputs<'a>(groups: &'a [InputGroup], query: &str) -> Vec<SectionMatches<'a>>` (it matches each entry's `haystack`, built once like the results table's); panel fields `input_order: InputOrder` and `input_filter: String` (tests set them directly); the drawn texts "N of 172 inputs" and the run headings.

- [ ] **Step 1: Write the failing tests**

In `magcoupling-rs/src/gui/format.rs`, replace:

````rust
    fn num(x: f64) -> String {
        format_value(&Value::Num(x))
    }
````

with:

````rust
    fn num(x: f64) -> String {
        format_value(&Value::Num(x))
    }

    #[test]
    fn a_search_reads_label_path_and_cell_in_lowercase_and_a_needle_drops_the_blanks() {
        assert_eq!(
            search_haystack(
                "Gearbox ratio",
                "coupling.gear_ratio",
                Some("Calculator!C45")
            ),
            "gearbox ratio\ncoupling.gear_ratio\ncalculator!c45"
        );
        // A Rust-only input or result has no cell.
        assert_eq!(
            search_haystack("Highest odd harmonic summed", "coupling.max_harmonic", None),
            "highest odd harmonic summed\ncoupling.max_harmonic\n"
        );
        assert_eq!(search_needle("  GEARBOX Ratio "), "gearbox ratio");
        assert_eq!(search_needle("   "), "");
    }
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
        assert!(npole.contains("Slider 4 to 40, step 2"), "{npole}");
    }
}
````

with:

````rust
        assert!(npole.contains("Slider 4 to 40, step 2"), "{npole}");
    }

    #[test]
    fn the_filter_matches_label_path_and_cell_by_section_in_the_order_shown() {
        let catalogue = InputCatalogue::new();
        let paths = |order: InputOrder, query: &str| -> Vec<String> {
            filter_inputs(catalogue.groups_in(order), query)
                .iter()
                .flat_map(|m| m.entries.iter().map(|e| e.path.clone()))
                .collect()
        };
        let gearbox = [
            "coupling.gear_ratio",
            "coupling.gear_efficiency",
            "coupling.gearbox_input_rating_Nm",
        ];
        // By label (and path), ignoring case and the surrounding blanks, in either order.
        assert_eq!(paths(InputOrder::Workflow, "  GEARBOX "), gearbox);
        assert_eq!(paths(InputOrder::Workbook, "gearbox"), gearbox);
        // By workbook cell, and by path.
        assert_eq!(
            paths(InputOrder::Workflow, "calculator!c45"),
            ["coupling.gear_ratio"]
        );
        assert_eq!(
            paths(InputOrder::Workflow, "clamps.key"),
            ["clamps.key_width_mm", "clamps.key_contact_mm"]
        );
        // A Rust-only input (no cell) by its path.
        assert_eq!(
            paths(InputOrder::Workflow, "max_harmonic"),
            ["coupling.max_harmonic"]
        );
        // Each run is headed by its group, and its section unless it is the group's own.
        let workflow = filter_inputs(catalogue.groups_in(InputOrder::Workflow), "gearbox");
        assert_eq!(workflow.len(), 1);
        assert_eq!(
            workflow[0].heading(),
            "Requirements and operating conditions / Drive and gearbox"
        );
        let workbook = filter_inputs(catalogue.groups_in(InputOrder::Workbook), "gearbox");
        assert_eq!(workbook[0].heading(), "Coupling");
        // A blank filter matches every input; a filter nothing contains, none.
        for order in InputOrder::ALL {
            assert_eq!(paths(order, "").len(), catalogue.all().count());
            assert_eq!(paths(order, "   ").len(), catalogue.all().count());
        }
        assert!(
            filter_inputs(catalogue.groups_in(InputOrder::Workflow), "no such input").is_empty()
        );
    }
}
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        // group open, on a screen tall enough to draw every row).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 12000.0));
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        harness.panel.open_design(design.clone());
        // Bottom up: opening a group moves only the groups below it, so no click lands on a
        // row that an opening group above has just moved there.
````

with:

````rust
        // group open, on a screen tall enough to draw every row), in the workbook order and in
        // the workflow order's filtered view (every row, the advanced ones included).
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 20000.0));
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        harness.panel.open_design(design.clone());
        // The workbook order: every row is in a group, none under an Advanced heading.
        harness.panel.input_order = InputOrder::Workbook;
        // Bottom up: opening a group moves only the groups below it, so no click lands on a
        // row that an opening group above has just moved there.
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        );
        assert!(!harness.panel.history.can_undo(&design));
    }
````

with:

````rust
        );
        assert!(!harness.panel.history.can_undo(&design));
        // The workflow order, filtered by "." (in every path): all 172 rows, both vacuum
        // permeabilities (under Calibration and model's Advanced heading when unfiltered).
        harness.panel.input_order = InputOrder::Workflow;
        harness.panel.input_filter = ".".to_owned();
        let mut output = harness.frame(Vec::new());
        for _ in 0..15 {
            output = harness.frame(Vec::new());
        }
        let total = InputCatalogue::get().all().count();
        assert_eq!(count(&output, &format!("{total} of {total} inputs")), 1);
        assert_eq!(count(&output, "Vacuum permeability"), 2);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), 0);
    }
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    fn every_group_opens_and_draws_a_row_for_each_of_its_inputs() {
        let catalogue = InputCatalogue::get();
        for group in &catalogue.groups {
            // Tall enough for the longest group (temperature) to fit without scrolling.
            let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
            harness.click_text(group.label);
            // The header opens over a few frames (its animation).
            let mut output = harness.frame(Vec::new());
            for _ in 0..10 {
                output = harness.frame(Vec::new());
            }
            let texts = drawn_texts(&output);
            for entry in group.sections.iter().flat_map(|s| s.entries.iter()) {
                assert!(
                    texts.iter().any(|t| t == entry.meta.label),
                    "{}: missing {:?}",
                    group.name,
                    entry.meta.label
                );
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
    }
````

with:

````rust
    fn every_group_opens_and_draws_a_row_for_each_of_its_inputs() {
        let catalogue = InputCatalogue::get();
        for order in InputOrder::ALL {
            for group in catalogue.groups_in(order) {
                // Tall enough for the longest group (temperature) to fit without scrolling.
                let mut harness = Harness::on_screen(egui::vec2(1280.0, 6000.0));
                harness.panel.input_order = order;
                harness.click_text(group.label);
                // The header opens over a few frames (its animation).
                let mut output = harness.frame(Vec::new());
                for _ in 0..10 {
                    output = harness.frame(Vec::new());
                }
                if group.sections.iter().any(|s| s.advanced) {
                    // The advanced sections stay closed until their heading is clicked.
                    let texts = drawn_texts(&output);
                    for section in group.sections.iter().filter(|s| s.advanced) {
                        assert!(
                            !texts.iter().any(|t| t == section.label),
                            "{}: {:?} open by default",
                            group.name,
                            section.label
                        );
                    }
                    harness.click_text(ADVANCED_HEADING);
                    for _ in 0..10 {
                        output = harness.frame(Vec::new());
                    }
                }
                let texts = drawn_texts(&output);
                for section in &group.sections {
                    if section.id != group.name {
                        assert!(
                            texts.iter().any(|t| t == section.label),
                            "{}: no heading {:?}",
                            group.name,
                            section.label
                        );
                    }
                    for entry in &section.entries {
                        assert!(
                            texts.iter().any(|t| t == entry.meta.label),
                            "{}: missing {:?}",
                            group.name,
                            entry.meta.label
                        );
                    }
                }
                assert_eq!(harness.panel.inputs(), &DesignInputs::default());
            }
        }
    }

    #[test]
    fn the_order_toggle_switches_the_view_and_changes_no_input() {
        // Decision O-1: the workflow order by default, the workbook's package groups one click
        // away; a view of this session only, so the design, its share link and the undo history
        // stay as they are.
        let mut harness = Harness::new();
        let link = harness.panel.share_link();
        let workflow: Vec<&str> = InputCatalogue::get()
            .workflow
            .iter()
            .map(|g| g.label)
            .collect();
        let workbook = [
            "Coupling",
            "Metal design",
            "Temperature design",
            "Shaft clamps",
        ];
        let drawn = |output: &egui::FullOutput, labels: &[&str]| {
            labels.iter().all(|label| count(output, label) == 1)
        };
        let none = |output: &egui::FullOutput, labels: &[&str]| {
            labels.iter().all(|label| count(output, label) == 0)
        };
        assert_eq!(harness.panel.input_order, InputOrder::Workflow);
        let output = harness.frame(Vec::new());
        assert!(drawn(&output, &workflow) && none(&output, &workbook));
        harness.click_text(InputOrder::Workbook.label());
        assert_eq!(harness.panel.input_order, InputOrder::Workbook);
        let output = harness.frame(Vec::new());
        assert!(drawn(&output, &workbook) && none(&output, &workflow[..4]));
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.share_link(), link);
        assert!(!harness.panel.history.can_undo(&Design::default()));
        harness.click_text(InputOrder::Workflow.label());
        let output = harness.frame(Vec::new());
        assert!(drawn(&output, &workflow) && none(&output, &workbook));
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.share_link(), link);
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design() {
        let mut harness = Harness::new();
        let total = InputCatalogue::get().all().count();
        harness.click_text(FILTER_HINT);
        harness.frame(vec![egui::Event::Text("gearbox".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("3 of {total} inputs")), 1);
        assert_eq!(
            count(
                &output,
                "Requirements and operating conditions / Drive and gearbox"
            ),
            1
        );
        for label in [
            "Gearbox ratio",
            "Gearbox efficiency",
            "Gearbox input torque rating",
        ] {
            assert_eq!(count(&output, label), 1, "{label}");
        }
        // The Key design group and the other groups give way to the matches.
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 0);
        assert_eq!(count(&output, "Magnets and rings"), 0);
        // A matched row is a row like any other: an arrow key nudges its value.
        let ratio = InputCatalogue::get().entry("coupling.gear_ratio").unwrap();
        let before = harness.number("coupling.gear_ratio");
        let slider = harness.panel.design_widgets[0];
        harness.ctx.memory_mut(|m| m.request_focus(slider));
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        let step = ratio.meta.range.unwrap().step;
        assert!((harness.number("coupling.gear_ratio") - (before + step)).abs() < 1e-9);
        // Typing in the filter is no edit of the design: the nudge is the one undo step.
        harness.frame(Vec::new());
        assert_eq!(harness.panel.history.undo_len(), 1);
        // The workbook order heads the run by its package group.
        harness.click_text(InputOrder::Workbook.label());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Coupling"), 1);
        assert_eq!(count(&output, &format!("3 of {total} inputs")), 1);
        // No match says so; a blank filter brings the groups back.
        harness.panel.input_filter = "no such input".to_owned();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("0 of {total} inputs")), 1);
        harness.panel.input_filter = "   ".to_owned();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, KEY_DESIGN_HEADING), 1);
    }

    #[test]
    fn a_focused_advanced_input_opens_its_group_and_heading_and_clears_the_filter() {
        // The vacuum permeability is no assumption and sits under Calibration and model's
        // Advanced heading: a leaf term naming it opens both, past a filter it does not match.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.panel.input_filter = "gearbox".to_owned();
        harness.frame(Vec::new());
        harness.panel.explorer.focus_input("coupling.mu0");
        let mut output = harness.frame(Vec::new());
        for _ in 0..10 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.input_filter, "");
        assert_eq!(harness.panel.inputs_view, InputsView::Design);
        let label = InputCatalogue::get()
            .entry("coupling.mu0")
            .unwrap()
            .meta
            .label;
        let rows: Vec<egui::Rect> = text_rects(&output, label)
            .into_iter()
            .filter(|r| r.left() < INPUTS_WIDTH)
            .collect();
        assert_eq!(
            rows.len(),
            2,
            "both mu0 rows, under the open Advanced heading"
        );
        let focus = focus_rects(&output);
        assert!(
            rows.iter()
                .any(|r| focus.iter().any(|f| f.contains_rect(*r))),
            "{rows:?} {focus:?}"
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                .filter_map(|c| crate::gui::dashboard::hover_text(c.path)),
        );
        for group in &InputCatalogue::get().groups {
            harness.click_text(group.label);
        }
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        // The Assumptions view: every rationale and source.
        harness.click_text(InputsView::Assumptions.label());
````

with:

````rust
                .filter_map(|c| crate::gui::dashboard::hover_text(c.path)),
        );
        // Every group open, in both orders; every workflow heading; the filtered view's runs.
        for order in [InputOrder::Workflow, InputOrder::Workbook] {
            harness.click_text(order.label());
            // Bottom up: opening a group moves only the groups below it.
            for group in InputCatalogue::get().groups_in(order).iter().rev() {
                harness.click_text(group.label);
            }
            for _ in 0..10 {
                harness.frame(Vec::new());
            }
            texts.extend(drawn_texts(&harness.frame(Vec::new())));
        }
        for group in &crate::gui::inputs::WORKFLOW {
            texts.push(group.label.to_owned());
            texts.extend(group.sections.iter().map(|s| s.label.to_owned()));
        }
        texts.push(ADVANCED_HEADING.to_owned());
        texts.push(FILTER_HINT.to_owned());
        harness.panel.input_filter = "a".to_owned();
        texts.extend(drawn_texts(&harness.frame(Vec::new())));
        harness.panel.input_filter.clear();
        // The Assumptions view: every rationale and source.
        harness.click_text(InputsView::Assumptions.label());
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        // The outer ring starts on a grade of its own, so the blank choice is drawn once.
        harness.panel.inputs.coupling.magnets.grade_outer = "Y30".to_owned();
        harness.click_text("Coupling");
        for _ in 0..10 {
            harness.frame(Vec::new());
````

with:

````rust
        // The outer ring starts on a grade of its own, so the blank choice is drawn once.
        harness.panel.inputs.coupling.magnets.grade_outer = "Y30".to_owned();
        // Both grade rows are in the Magnets and rings group.
        harness.click_text("Magnets and rings");
        for _ in 0..10 {
            harness.frame(Vec::new());
````

- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib 2>&1 | grep -E "^error" | sort | uniq
```

Expected: FAIL to compile, with these lines:

```
error: could not compile `magcoupling-rs` (lib test) due to 34 previous errors
error[E0425]: cannot find function `filter_inputs` in this scope
error[E0425]: cannot find function `search_haystack` in this scope
error[E0425]: cannot find function `search_needle` in this scope
error[E0425]: cannot find value `ADVANCED_HEADING` in this scope
error[E0425]: cannot find value `FILTER_HINT` in this scope
error[E0433]: failed to resolve: use of undeclared type `InputOrder`
error[E0609]: no field `input_filter` on type `gui::panel::MagcouplingPanel`
error[E0609]: no field `input_order` on type `gui::panel::MagcouplingPanel`
```

- [ ] **Step 3: Add the search helpers, the filter and the inputs side**

In `magcoupling-rs/src/gui/format.rs`, replace:

````rust
        format!("{text} {unit}")
    }
}
````

with:

````rust
        format!("{text} {unit}")
    }
}

/// What a search box matches an input or a result by: its label, path and workbook cell, one
/// per line, lowercase. The results search and the inputs filter share it, so both match alike.
pub(crate) fn search_haystack(label: &str, path: &str, cell: Option<&str>) -> String {
    format!("{label}\n{path}\n{}", cell.unwrap_or("")).to_lowercase()
}

/// A search box's text as it is matched against a [`search_haystack`]: lowercase, without the
/// surrounding blanks. An empty needle matches everything.
pub(crate) fn search_needle(query: &str) -> String {
    query.trim().to_lowercase()
}
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{ResultInfo, hover_text, result_info};
use crate::gui::format::{format_value, non_finite_text, with_unit};
use crate::gui::readouts::Readouts;
use crate::gui::session::{Design, design_json, json_value};
````

with:

````rust
use crate::gui::corrections::{CorrectionIndex, marker_text};
use crate::gui::dashboard::{ResultInfo, hover_text, result_info};
use crate::gui::format::{
    format_value, non_finite_text, search_haystack, search_needle, with_unit,
};
use crate::gui::readouts::Readouts;
use crate::gui::session::{Design, design_json, json_value};
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
            .map(|row| {
                let info = result_info(&row.path).expect("every result has its info");
                let haystack = format!(
                    "{}\n{}\n{}",
                    info.meta.label,
                    row.path,
                    info.cell.as_deref().unwrap_or("")
                )
                .to_lowercase();
                TableEntry {
                    marker: marker_text(CorrectionIndex::get().marks(info.cell.as_deref())),
````

with:

````rust
            .map(|row| {
                let info = result_info(&row.path).expect("every result has its info");
                let haystack = search_haystack(info.meta.label, &row.path, info.cell.as_deref());
                TableEntry {
                    marker: marker_text(CorrectionIndex::get().marks(info.cell.as_deref())),
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// surrounding blanks; every row for a blank query.
pub fn search(entries: &[TableEntry], query: &str) -> Vec<usize> {
    let needle = query.trim().to_lowercase();
    (0..entries.len())
        .filter(|&i| needle.is_empty() || entries[i].haystack.contains(&needle))
````

with:

````rust
/// surrounding blanks; every row for a blank query.
pub fn search(entries: &[TableEntry], query: &str) -> Vec<usize> {
    let needle = search_needle(query);
    (0..entries.len())
        .filter(|&i| needle.is_empty() || entries[i].haystack.contains(&needle))
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
use crate::engine::library;
use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value, input_rows};

/// The Key design group, in order (spec M4 "Layout": face gap, pole count, magnet part, axial
````

with:

````rust
use crate::engine::library;
use crate::engine::meta::{FieldType, InputMeta, SliderRange, Value, input_rows};
use crate::gui::format::{search_haystack, search_needle};

/// The Key design group, in order (spec M4 "Layout": face gap, pole count, magnet part, axial
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
    pub meta: &'static InputMeta,
    pub default: Value,
}
````

with:

````rust
    pub meta: &'static InputMeta,
    pub default: Value,
    /// Label, path and cell, lowercase: what the inputs filter matches ([`search_haystack`],
    /// built once, as the results table's).
    haystack: String,
}
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
            meta: row.meta,
            default: row.value.clone(),
        };
        let key_design = KEY_DESIGN
````

with:

````rust
            meta: row.meta,
            default: row.value.clone(),
            haystack: search_haystack(row.meta.label, &row.path, row.meta.cell),
        };
        let key_design = KEY_DESIGN
````

In `magcoupling-rs/src/gui/inputs.rs`, replace:

````rust
}

/// Decimal places of a slider step: the fewest that write the step exactly (0.01 → 2,
/// 0.005 → 3, 2 → 0, 1e-13 → 13). Slider values are rounded to them (decision M41-1), so a
````

with:

````rust
}

/// The filter box's hint: it matches as the results search does (decision O-5).
pub const FILTER_HINT: &str = "Filter inputs by label, path or cell";

/// The inputs of one section that a filter matches.
#[derive(Clone, Debug, PartialEq)]
pub struct SectionMatches<'a> {
    pub group: &'a InputGroup,
    pub section: &'a InputSection,
    pub entries: Vec<&'a InputEntry>,
}

impl SectionMatches<'_> {
    /// The run's heading: the group's label, then the section's unless it is the group's own.
    pub fn heading(&self) -> String {
        if self.section.id == self.group.name {
            self.group.label.to_owned()
        } else {
            format!("{} / {}", self.group.label, self.section.label)
        }
    }
}

/// The inputs of `groups` whose label, path or workbook cell contains `query`, ignoring case
/// and the surrounding blanks (as the results search), by section in the order of `groups`;
/// a section without a match is left out. Every input of `groups` for a blank query.
pub fn filter_inputs<'a>(groups: &'a [InputGroup], query: &str) -> Vec<SectionMatches<'a>> {
    let needle = search_needle(query);
    let mut matches = Vec::new();
    for group in groups {
        for section in &group.sections {
            let entries: Vec<&InputEntry> = section
                .entries
                .iter()
                .filter(|e| needle.is_empty() || e.haystack.contains(&needle))
                .collect();
            if !entries.is_empty() {
                matches.push(SectionMatches {
                    group,
                    section,
                    entries,
                });
            }
        }
    }
    matches
}

/// Decimal places of a slider step: the fewest that write the step exactly (0.01 → 2,
/// 0.005 → 3, 2 → 0, 1e-13 → 13). Slider values are rounded to them (decision M41-1), so a
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::explorer::{EQUATION_PANEL, Explorer, FOCUS_WIDTH, PANEL_HEIGHT, explorer_ui};
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, input_row, slider};
use crate::gui::inputs::{InputCatalogue, InputEntry, KEY_DESIGN, optional_seed};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
````

with:

````rust
use crate::gui::dashboard::{Level, dashboard_ui, end_effect_banner};
use crate::gui::explorer::{EQUATION_PANEL, Explorer, FOCUS_WIDTH, PANEL_HEIGHT, explorer_ui};
use crate::gui::format::search_needle;
use crate::gui::geometry_view::geometry_ui;
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, input_row, slider};
use crate::gui::inputs::{
    ADVANCED_HEADING, FILTER_HINT, InputCatalogue, InputEntry, InputGroup, InputOrder,
    InputSection, KEY_DESIGN, filter_inputs, optional_seed,
};
use crate::gui::plots::{PlotKind, plot_ui};
use crate::gui::readouts::{Readouts, registry};
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    /// What the inputs side shows.
    inputs_view: InputsView,
    /// The assumptions banner drawn in the last frame: the header sizes from the last frame,
    /// so a change asks for one more frame.
````

with:

````rust
    /// What the inputs side shows.
    inputs_view: InputsView,
    /// How the design inputs are ordered (decision O-1): a view of this session only, never in
    /// a design file or a share link.
    input_order: InputOrder,
    /// The inputs filter's text: while it holds more than blanks, the design inputs view shows
    /// the matching inputs alone.
    input_filter: String,
    /// The assumptions banner drawn in the last frame: the header sizes from the last frame,
    /// so a change asks for one more frame.
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            centre: CentreView::Geometry,
            inputs_view: InputsView::Design,
            banner: None,
            results_table: ResultsTable::default(),
````

with:

````rust
            centre: CentreView::Geometry,
            inputs_view: InputsView::Design,
            input_order: InputOrder::default(),
            input_filter: String::new(),
            banner: None,
            results_table: ResultsTable::default(),
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    }

    /// The design inputs: the Key design group, then every input by package group.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        // A leaf term clicked in the Equation panel: its group opens so its row can scroll
        // into view (the Key design group for a Key design input).
        let focus = self
            .explorer
            .focus()
            .filter(|f| f.scroll)
            .map(|f| f.path.clone());
        let open_key_design = focus
            .as_deref()
            .is_some_and(|path| KEY_DESIGN.contains(&path));
        let open_group = focus
            .as_deref()
            .filter(|_| !open_key_design)
            .and_then(|path| {
                catalogue
                    .groups
                    .iter()
                    .find(|g| {
                        g.sections
                            .iter()
                            .any(|s| s.entries.iter().any(|e| e.path == path))
                    })
                    .map(|g| g.name.as_str())
            });
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
````

with:

````rust
    }

    /// The design inputs: the order toggle and the filter box, then the Key design group and
    /// every input by group in the order chosen (decision O-1), each workflow group's advanced
    /// sections under its closed Advanced heading; while the filter holds more than blanks, the
    /// matching inputs alone.
    fn design_inputs_ui(&mut self, ui: &mut egui::Ui, readouts: &Readouts) {
        let catalogue = InputCatalogue::get();
        // A leaf term clicked in the Equation panel: its group (and its Advanced heading) opens
        // so its row can scroll into view (the Key design group for a Key design input), and
        // the filter is cleared so the row is drawn.
        let focus = self
            .explorer
            .focus()
            .filter(|f| f.scroll)
            .map(|f| f.path.clone());
        if focus.is_some() {
            self.input_filter.clear();
        }
        ui.horizontal(|ui| {
            for order in InputOrder::ALL {
                ui.selectable_value(&mut self.input_order, order, order.label());
            }
        });
        ui.add(
            egui::TextEdit::singleline(&mut self.input_filter)
                .hint_text(FILTER_HINT)
                .desired_width(f32::INFINITY),
        );
        let groups = catalogue.groups_in(self.input_order);
        if !search_needle(&self.input_filter).is_empty() {
            self.filtered_inputs_ui(ui, groups, readouts);
            return;
        }
        let open_key_design = focus
            .as_deref()
            .is_some_and(|path| KEY_DESIGN.contains(&path));
        // The group and the section of the focused row (outside the Key design group).
        let focused = focus
            .as_deref()
            .filter(|_| !open_key_design)
            .and_then(|path| catalogue.section_of(self.input_order, path))
            .map(|(group, section)| (group.name.as_str(), section.advanced));
        // The workbook order keeps its groups' ids (and so their open state) from before the
        // workflow order existed.
        let salt = match self.input_order {
            InputOrder::Workflow => "workflow",
            InputOrder::Workbook => "group",
        };
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_scroll")
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                        }
                    });
                for group in &catalogue.groups {
                    egui::CollapsingHeader::new(group.label)
                        .id_salt(("group", &group.name))
                        .default_open(false)
                        .open((open_group == Some(group.name.as_str())).then_some(true))
                        .show(ui, |ui| {
                            for section in &group.sections {
                                if section.id != group.name {
                                    ui.add_space(4.0);
                                    ui.strong(section.label);
                                }
                                for entry in &section.entries {
                                    let widget = self.input_row_ui(ui, entry, readouts);
                                    self.design_widgets.push(widget);
                                }
                            }
                        });
                }
            });
````

with:

````rust
                        }
                    });
                for group in groups {
                    let (open_group, open_advanced) = match focused {
                        Some((name, advanced)) if name == group.name => (true, advanced),
                        _ => (false, false),
                    };
                    egui::CollapsingHeader::new(group.label)
                        .id_salt((salt, &group.name))
                        .default_open(false)
                        .open(open_group.then_some(true))
                        .show(ui, |ui| {
                            for section in group.sections.iter().filter(|s| !s.advanced) {
                                self.section_ui(ui, group, section, readouts);
                            }
                            if group.sections.iter().any(|s| s.advanced) {
                                egui::CollapsingHeader::new(ADVANCED_HEADING)
                                    .id_salt(("advanced", &group.name))
                                    .default_open(false)
                                    .open(open_advanced.then_some(true))
                                    .show(ui, |ui| {
                                        for section in group.sections.iter().filter(|s| s.advanced)
                                        {
                                            self.section_ui(ui, group, section, readouts);
                                        }
                                    });
                            }
                        });
                }
            });
    }

    /// One section of a group: its heading (none for the group's own section), then its rows.
    fn section_ui(
        &mut self,
        ui: &mut egui::Ui,
        group: &InputGroup,
        section: &'static InputSection,
        readouts: &Readouts,
    ) {
        if section.id != group.name {
            ui.add_space(4.0);
            ui.strong(section.label);
        }
        for entry in &section.entries {
            let widget = self.input_row_ui(ui, entry, readouts);
            self.design_widgets.push(widget);
        }
    }

    /// The inputs the filter matches, by section in the order chosen, each run under its
    /// group and section, with their count.
    fn filtered_inputs_ui(
        &mut self,
        ui: &mut egui::Ui,
        groups: &'static [InputGroup],
        readouts: &Readouts,
    ) {
        let matches = filter_inputs(groups, &self.input_filter);
        let shown: usize = matches.iter().map(|m| m.entries.len()).sum();
        let total = InputCatalogue::get().all().count();
        ui.weak(format!("{shown} of {total} inputs"));
        egui::ScrollArea::vertical()
            .id_salt("magcoupling_inputs_filter_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for run in &matches {
                    ui.add_space(4.0);
                    ui.strong(run.heading());
                    for &entry in &run.entries {
                        let widget = self.input_row_ui(ui, entry, readouts);
                        self.design_widgets.push(widget);
                    }
                }
            });
````

- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib -- search_reads filter_matches order_toggle filter_box focused_advanced every_group_opens idle_frames has_glyphs grade_pickers leaf_term_opens results_table_lists 2>&1 | grep -E "^test |test result"
```

Expected: these 13 tests `ok` (in any order): `gui::format::tests::a_search_reads_label_path_and_cell_in_lowercase_and_a_needle_drops_the_blanks`, `gui::inputs::tests::the_filter_matches_label_path_and_cell_by_section_in_the_order_shown`, `gui::pickers::tests::the_part_and_grade_pickers_are_text_inputs_of_the_catalogue`, `gui::pickers::tests::every_picker_text_has_glyphs_in_the_default_fonts`, and in `gui::panel::tests`: `the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design`, `the_order_toggle_switches_the_view_and_changes_no_input`, `a_focused_advanced_input_opens_its_group_and_heading_and_clears_the_filter`, `the_results_table_lists_results_and_filters_by_the_search`, `a_leaf_term_opens_its_group_and_highlights_its_input_row`, `the_grade_pickers_set_a_grade_by_its_id_or_none_for_either_ring`, `idle_frames_change_no_input`, `every_group_opens_and_draws_a_row_for_each_of_its_inputs`, `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`; then `test result: ok. 13 passed; 0 failed`.

- [ ] **Step 5: Run the checks and the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --check
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED"
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task2.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
```

Expected: `cargo fmt --check` prints nothing; `test result: ok. 456 passed; 0 failed`; `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order add magcoupling-rs/src/gui/format.rs magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/inputs.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-order commit -q -F - <<'EOF'
feat(magcoupling-rs): the inputs by workflow or workbook group, Advanced headings, a filter box (O-1, O-4, O-5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-order status --short
```

Expected: no status lines.

---

### Task 3: Every check's level and the results groups

**Model:** `sonnet` (exact code; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/src/gui/dashboard.rs` (`Level` ordered by severity; `CHECKS` before `verdict_level`; its new arms; `check_level`, `failing_checks` after it; `badge` made `pub(crate)`; three tests)
- Create: `magcoupling-rs/src/gui/result_groups.rs` (tests first, then the module above them)
- Modify: `magcoupling-rs/src/gui/mod.rs` (`pub mod result_groups;`)

**Interfaces:**
- Consumes: `DASHBOARD`, `severity_level`, `WARNING_RULES` (dashboard); `table_entries() -> &'static [TableEntry]` (results_table, unchanged here); `engine::explain::scope::SCOPE` (each `Chain { id, status, paths }`; a table's every row written `clamps.table[].<column>`).
- Produces: `Level` derives `PartialOrd, Ord` (green < amber < red: the worst of several is their maximum); `pub const CHECKS: [&str; 30]` (schema order); `pub fn check_level(results: &DesignResults, path: &str) -> Option<Level>`; `pub fn failing_checks(results: &DesignResults) -> Vec<(&'static str, Level)>` (red first, each colour in `CHECKS` order); `pub(crate) fn badge(ui: &mut egui::Ui, level: Option<Level>)`; `pub struct ResultGroup { id: &'static str, label: &'static str, other: bool, rows: Vec<usize> }` (rows index `table_entries()`; a chain and a package may share an `id`: `other` tells them apart); `pub fn result_groups() -> &'static [ResultGroup]` (index 0 is the headline); `pub const OTHER_RESULTS: &str = "Other results"`; `pub const HEADLINE_LABEL: &str = "Headline"`; `pub const CHAIN_LABELS: [(&str, &str); 8]` (the scope's six chains in its order, `adhesive` after `slip_heating`, `mass` last); `pub const CHAIN_PREFIXES: [(&str, &str); 19]` (prefix, chain id: where a result the scope leaves out goes, first match wins); `PACKAGE_LABELS`, `row_template`, `package_of`, `package_label`, `groups_of`.

- [ ] **Step 1: Write the failing tests**

In `magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
        assert_eq!(verdict_level("metal.hot_min_check", "OK"), None);
        assert_eq!(verdict_level("model.pullout_Nm", "OK"), None);
    }
````

with:

````rust
        assert_eq!(verdict_level("metal.hot_min_check", "OK"), None);
        assert_eq!(verdict_level("model.pullout_Nm", "OK"), None);
    }

    /// Manual magnets of the ferrite grade Y30 (positive beta: the cold side is checked).
    fn ferrite(inputs: &mut DesignInputs) {
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        inputs.coupling.magnets.grade_inner = "Y30".to_owned();
        inputs.coupling.magnets.grade_outer = "Y30".to_owned();
    }

    #[test]
    fn every_verdict_of_every_other_check_is_classified() {
        // Decision O-3: the checks off the dashboard, each from a design that reaches the
        // branch (the dashboard's own are in the test above); the two texts no design reaches
        // are checked as written below.
        use Level::{Bad, Caution, Good};
        let default = DesignInputs::default;
        let cases: Vec<(&str, DesignInputs, Level)> = vec![
            ("calibration.end_effect_check", default(), Good),
            (
                "calibration.end_effect_check",
                design(|i| {
                    i.calibration.c_end = 0.5;
                    i.calibration.magnet_length_mm = 2.0;
                }),
                Bad,
            ),
            ("model.inner_flat_check", default(), Good),
            (
                "model.inner_flat_check",
                design(|i| i.coupling.npole = 40),
                Bad,
            ),
            ("model.outer_flat_check", default(), Good),
            (
                "model.outer_flat_check",
                design(|i| i.coupling.npole = 40),
                Bad,
            ),
            ("model.verdict", default(), Good),
            (
                "model.verdict",
                design(|i| i.metal.required_min_Nm = 10.0),
                Bad,
            ),
            ("model.cup_ring_check", default(), Bad),
            (
                "model.cup_ring_check",
                design(|i| i.metal.cup_wall_corner_mm = 6.0),
                Good,
            ),
            ("model.hub_check", default(), Good),
            (
                "model.hub_check",
                design(|i| i.coupling.bore_mm = 17.0),
                Bad,
            ),
            ("model.inner_temp_check", default(), Good),
            (
                "model.inner_temp_check",
                design(|i| i.coupling.op_temp_C = 200.0),
                Bad,
            ),
            ("model.outer_temp_check", default(), Good),
            (
                "model.outer_temp_check",
                design(|i| i.coupling.op_temp_C = 200.0),
                Bad,
            ),
            ("temperature.summary.torque_hot_day_note", default(), Good),
            (
                "temperature.summary.torque_hot_day_note",
                design(|i| i.metal.required_min_Nm = 10.0),
                Bad,
            ),
            (
                "temperature.magnet_life.torque_hot_day_check",
                default(),
                Good,
            ),
            (
                "temperature.magnet_life.torque_hot_day_check",
                design(|i| i.metal.required_min_Nm = 10.0),
                Bad,
            ),
            ("temperature.demag.cold_check", design(ferrite), Bad),
            (
                "temperature.demag.cold_check",
                design(|i| {
                    ferrite(i);
                    i.metal.min_temp_C = 20.0;
                    i.temperature.demag.h_rev_aligned_kA_m = 10.0;
                    i.temperature.demag.h_rev_pullout_kA_m = 10.0;
                    i.temperature.demag.h_rev_likepole_kA_m = 10.0;
                    i.temperature.demag.h_rev_single_ring_kA_m = 10.0;
                }),
                Good,
            ),
            ("temperature.adhesive.fatigue_screen", default(), Good),
            (
                "temperature.adhesive.fatigue_screen",
                design(|i| i.temperature.adhesive_life.fatigue_endurance = 0.05),
                Caution,
            ),
            ("temperature.mismatch.reading", default(), Good),
            (
                "temperature.mismatch.reading",
                design(|i| i.materials.steel.cte_per_C = 3e-5),
                Bad,
            ),
            (
                "temperature.adhesive_life.hot_fatigue_screen",
                default(),
                Good,
            ),
            (
                "temperature.adhesive_life.hot_fatigue_screen",
                design(|i| i.temperature.adhesive_life.hot_strength_retained = 0.05),
                Caution,
            ),
            ("temperature.adhesive_life.daily_screen", default(), Good),
            (
                "temperature.adhesive_life.daily_screen",
                design(|i| i.temperature.adhesive_life.daily_swing_C = 300.0),
                Caution,
            ),
            ("clamps.head_check", default(), Good),
            ("clamps.head_check", design(|i| i.clamps.alloy = 2), Caution),
            ("clamps.vent_port", default(), Good),
            (
                "warnings.non_ferromagnetic_back_iron",
                design(|i| i.materials.parts.back_iron = 7),
                Bad,
            ),
            (
                "warnings.cte_mismatch_with_magnets",
                design(|i| i.materials.parts.back_iron = 7),
                Caution,
            ),
            (
                "warnings.low_saturation",
                design(|i| i.materials.parts.back_iron = 3),
                Caution,
            ),
            (
                "warnings.high_conductivity_sleeve_or_liner",
                design(|i| i.temperature.slip_loss.sigma_316_S_m = 5e6),
                Caution,
            ),
            (
                "warnings.uncoated_low_alloy_steel",
                design(|i| i.materials.nickel.thickness_mm = 0.0),
                Caution,
            ),
        ];
        for (check, inputs, want) in cases {
            let results = compute_all(&inputs);
            let Some(Value::Text(text)) = results.get(check) else {
                panic!("{check} is text")
            };
            assert_eq!(verdict_level(check, &text), Some(want), "{check}: {text:?}");
            assert_eq!(check_level(&results, check), Some(want), "{check}");
        }
        // The key of an M6 or larger screw misses the vent port: no design of the clamp
        // model's bore reaches it, so its text (clamps.rs) is checked as written.
        assert_eq!(
            verdict_level("clamps.vent_port", "No: key too large"),
            Some(Caution)
        );
        // No sleeve or liner of the material library is ferromagnetic, so no design fires that
        // warning: its text (warnings.rs) is checked as written, red as a warning.
        let ferromagnetic = WARNING_RULES
            .iter()
            .find(|rule| rule.id == "ferromagnetic_sleeve_or_liner")
            .unwrap();
        assert_eq!(
            verdict_level("warnings.ferromagnetic_sleeve_or_liner", ferromagnetic.text),
            Some(Bad)
        );
        // A check that does not apply gives no badge: arcs have no flats, no back iron needs
        // no wall, NdFeB's coercivity rises as it cools, no screw fits (no head, no key), and
        // a warning that does not fire is empty.
        let none = |inputs: DesignInputs, checks: &[&str]| {
            let results = compute_all(&inputs);
            for check in checks {
                let Some(Value::Text(text)) = results.get(check) else {
                    panic!("{check} is text")
                };
                assert_eq!(verdict_level(check, &text), None, "{check}: {text:?}");
                assert_eq!(check_level(&results, check), None, "{check}");
            }
        };
        none(
            design(|i| i.coupling.faceted = 0),
            &["model.inner_flat_check", "model.outer_flat_check"],
        );
        none(
            design(|i| i.coupling.backiron = 0),
            &["model.cup_ring_check", "model.hub_check"],
        );
        none(default(), &["temperature.demag.cold_check"]);
        none(
            design(|i| {
                i.clamps.boss_od_mm = 12.0;
                i.clamps.clamp_length_mm = 3.0;
            }),
            &["clamps.head_check", "clamps.vent_port"],
        );
        let warnings: Vec<&str> = CHECKS
            .iter()
            .copied()
            .filter(|c| c.starts_with("warnings."))
            .collect();
        none(default(), &warnings);
        // A manual magnet without a grade has no rating to check against: amber.
        let manual = compute_all(&design(|i| {
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
        }));
        assert_eq!(manual.model.inner_temp_check, "unknown");
        assert_eq!(
            check_level(&manual, "model.inner_temp_check"),
            Some(Caution)
        );
        // No arm reads a text of another check, or of a path that is no check.
        assert_eq!(verdict_level("model.hub_check", "OK"), None);
        assert_eq!(verdict_level("warnings.no_such_rule", "text"), None);
        assert_eq!(check_level(&manual, "model.pullout_Nm"), None);
    }

    #[test]
    fn every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed() {
        let results = compute_all(&DesignInputs::default());
        let rows = result_rows(&results);
        // In schema order, each a text.
        let mut last = 0;
        for check in CHECKS {
            let index = rows
                .iter()
                .position(|r| r.path == check)
                .unwrap_or_else(|| panic!("CHECKS: no result {check}"));
            assert!(index >= last, "{check} out of schema order");
            last = index;
            assert!(matches!(rows[index].value, Value::Text(_)), "{check}");
        }
        // A new check (a text named *_check, *_screen, *verdict or *reading, or a warning)
        // fails here until it is listed and classified. The clamp table's and the sweeps' rows
        // rate candidates, not the design.
        for row in &rows {
            let name = row.path.rsplit('.').next().unwrap();
            let named = name.ends_with("_check")
                || name.ends_with("_screen")
                || name.ends_with("verdict")
                || name.ends_with("reading")
                || row.path.starts_with("warnings.");
            let per_row = row.path.contains('[');
            if named && !per_row && matches!(row.value, Value::Text(_)) {
                assert!(
                    CHECKS.contains(&row.path.as_str()),
                    "{} is not in CHECKS",
                    row.path
                );
            }
        }
        // Every warning rule is a check.
        for rule in WARNING_RULES {
            assert!(
                CHECKS.contains(&format!("warnings.{}", rule.id).as_str()),
                "{}",
                rule.id
            );
        }
    }

    #[test]
    fn the_failing_checks_are_the_red_then_the_amber() {
        use Level::{Bad, Caution, Good};
        // The levels order by severity: the worst of several is their maximum.
        assert!(Good < Caution && Caution < Bad);
        assert_eq!([Caution, Bad, Good].into_iter().max(), Some(Bad));
        // The defaults fail four checks, all red.
        let defaults = compute_all(&DesignInputs::default());
        assert_eq!(
            failing_checks(&defaults),
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
            ]
        );
        // Manual magnets add two amber rating checks, after the red.
        let manual = compute_all(&design(|i| {
            i.coupling.magnets.part_inner.clear();
            i.coupling.magnets.part_outer.clear();
        }));
        assert_eq!(
            failing_checks(&manual),
            [
                ("model.cup_ring_check", Bad),
                ("metal.hot_min_check", Bad),
                ("metal.clearance_check", Bad),
                ("materials.cup_wall_check", Bad),
                ("model.inner_temp_check", Caution),
                ("model.outer_temp_check", Caution),
            ]
        );
        // Exactly the checks whose level is red or amber, each once.
        for inputs in [DesignInputs::default(), short_magnets(), design(ferrite)] {
            let results = compute_all(&inputs);
            let failing: Vec<&str> = failing_checks(&results).iter().map(|(p, _)| *p).collect();
            let want: Vec<&str> = CHECKS
                .iter()
                .copied()
                .filter(|c| matches!(check_level(&results, c), Some(Bad | Caution)))
                .collect();
            let mut sorted = failing.clone();
            sorted.sort_unstable();
            let mut want_sorted = want.clone();
            want_sorted.sort_unstable();
            assert_eq!(sorted, want_sorted);
        }
    }
````

Create `magcoupling-rs/src/gui/result_groups.rs`:

````rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The group holding the row of `path`.
    fn group_of(path: &str) -> &'static str {
        let entries = table_entries();
        result_groups()
            .iter()
            .find(|g| g.rows.iter().any(|&i| entries[i].path == path))
            .unwrap_or_else(|| panic!("{path} in no group"))
            .id
    }

    #[test]
    fn every_result_is_in_exactly_one_group() {
        let entries = table_entries();
        let mut rows: Vec<usize> = result_groups()
            .iter()
            .flat_map(|g| g.rows.iter().copied())
            .collect();
        assert_eq!(rows.len(), entries.len(), "no row twice, none left out");
        rows.sort_unstable();
        assert_eq!(rows, (0..entries.len()).collect::<Vec<_>>());
        assert!(result_groups().iter().all(|g| !g.rows.is_empty()));
    }

    #[test]
    fn the_headline_comes_first_then_the_chains_then_the_other_results() {
        let groups = result_groups();
        let ids: Vec<&str> = groups.iter().map(|g| g.id).collect();
        assert_eq!(
            ids[..9],
            [
                "headline",
                "torque",
                "temperature",
                "demagnetization",
                "slip_heating",
                "adhesive",
                "clamps",
                "geometry",
                "mass"
            ]
        );
        assert!(groups[..9].iter().all(|g| !g.other));
        assert!(groups[9..].iter().all(|g| g.other));
        // A group is known by its id and its section.
        let mut keys: Vec<(bool, &str)> = groups.iter().map(|g| (g.other, g.id)).collect();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), groups.len());
        // The headline is the dashboard's rows, in its order.
        let entries = table_entries();
        let headline: Vec<&str> = groups[0]
            .rows
            .iter()
            .map(|&i| entries[i].path.as_str())
            .collect();
        let dashboard: Vec<&str> = DASHBOARD.iter().map(|(path, _)| *path).collect();
        assert_eq!(headline, dashboard);
        // A path in two places sits in the first: the pull-out on the dashboard and in two
        // chains, an onset in the temperature and the demagnetization chains.
        assert_eq!(group_of("model.pullout_Nm"), "headline");
        assert_eq!(group_of("model.f_end"), "torque");
        assert_eq!(
            group_of("temperature.summary.onset_aligned_C"),
            "temperature"
        );
        assert_eq!(
            group_of("temperature.demag.onset_pullout_C"),
            "demagnetization"
        );
        // A table template places every row of the table (the clamp table has five).
        for row in 0..5 {
            assert_eq!(
                group_of(&format!("clamps.table[{row}].preload_N")),
                "clamps"
            );
        }
        assert_eq!(group_of("clamps.table[0].size"), "clamps");
        // The results of a chain's nested groups that the scope leaves out join the chain.
        assert_eq!(group_of("temperature.demag.cold_check"), "demagnetization");
        assert_eq!(
            group_of("temperature.demag.inner_magnet_limit_C"),
            "demagnetization"
        );
        assert_eq!(
            group_of("temperature.summary.peak_with_fault_C"),
            "slip_heating"
        );
        assert_eq!(
            group_of("temperature.thermal.temp_at_fault_C"),
            "slip_heating"
        );
        assert_eq!(group_of("temperature.magnet_life.peak_C"), "temperature");
        assert_eq!(group_of("temperature.mismatch.reading"), "adhesive");
        assert_eq!(
            group_of("temperature.adhesive_life.daily_screen"),
            "adhesive"
        );
        assert_eq!(group_of("mass.magnets_g"), "mass");
        assert_eq!(group_of("metal.corner_gap_mm"), "geometry");
        // So no temperature design result is left among the other results.
        assert!(!groups.iter().any(|g| g.other && g.id == "temperature"));
        // The rest by package, in schema order.
        assert_eq!(group_of("model.verdict"), "model");
        assert_eq!(group_of("gap_sweep[3].f_end"), "gap_sweep");
        assert_eq!(group_of("retainers.retainers_g"), "retainers");
    }

    #[test]
    fn every_chain_path_names_results_and_every_package_has_a_heading() {
        let entries = table_entries();
        for chain in SCOPE {
            for path in chain.paths {
                assert!(
                    entries.iter().any(|e| row_template(&e.path) == *path),
                    "{}: {path} names no result",
                    chain.id
                );
            }
        }
        // Every scope entry but the dashboard's is a chain group, in the scope's order.
        let scope: Vec<&str> = SCOPE
            .iter()
            .map(|c| c.id)
            .filter(|id| *id != "dashboard")
            .collect();
        let chains: Vec<&str> = CHAIN_LABELS
            .iter()
            .map(|(id, _)| *id)
            .filter(|id| SCOPE.iter().any(|c| c.id == *id))
            .collect();
        assert_eq!(scope, chains);
        for entry in entries {
            let package = package_of(&entry.path);
            assert!(package_label(package).is_some(), "no heading for {package}");
        }
        assert_eq!(package_of("gap_sweep[12].tau_Pa"), "gap_sweep");
        assert_eq!(package_of("model.f_end"), "model");
        assert_eq!(
            row_template("clamps.table[3].preload_N"),
            "clamps.table[].preload_N"
        );
        assert_eq!(row_template("model.f_end"), "model.f_end");
    }

    #[test]
    fn every_chain_prefix_places_a_result_the_scope_leaves_out() {
        let entries = table_entries();
        let groups = result_groups();
        // A result the dashboard or a chain of the scope lists is placed by that list.
        let listed = |path: &str| {
            let template = row_template(path);
            DASHBOARD.iter().any(|(p, _)| *p == template)
                || SCOPE.iter().any(|c| c.paths.contains(&template.as_str()))
        };
        for (prefix, chain) in CHAIN_PREFIXES {
            let group = groups
                .iter()
                .find(|g| !g.other && g.id == chain)
                .unwrap_or_else(|| panic!("{prefix}: no chain group {chain}"));
            let placed = group
                .rows
                .iter()
                .map(|&i| entries[i].path.as_str())
                .filter(|path| !listed(path))
                .filter(|path| {
                    CHAIN_PREFIXES
                        .iter()
                        .find(|(p, _)| path.starts_with(p))
                        .is_some_and(|(p, _)| *p == prefix)
                })
                .count();
            assert!(placed > 0, "{prefix} places no result in {chain}");
        }
        // A chain of no scope entry is filled by its prefixes alone, and has rows.
        for (id, _) in CHAIN_LABELS {
            if SCOPE.iter().all(|c| c.id != id) {
                assert!(CHAIN_PREFIXES.iter().any(|(_, c)| *c == id), "{id}");
                assert!(groups.iter().any(|g| !g.other && g.id == id), "{id}");
            }
        }
    }
}
````

In `magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub mod plots;
pub mod readouts;
pub mod results_table;
pub mod session;
````

with:

````rust
pub mod plots;
pub mod readouts;
pub mod result_groups;
pub mod results_table;
pub mod session;
````

- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib 2>&1 | grep -E "^error" | sort | uniq
```

Expected: FAIL to compile, with these lines:

```
error: could not compile `magcoupling-rs` (lib test) due to 46 previous errors; 1 warning emitted
error[E0277]: the trait bound `gui::dashboard::Level: Ord` is not satisfied
error[E0369]: binary operation `<` cannot be applied to type `gui::dashboard::Level`
error[E0425]: cannot find function `check_level` in this scope
error[E0425]: cannot find function `failing_checks` in this scope
error[E0425]: cannot find function `package_label` in this scope
error[E0425]: cannot find function `package_of` in this scope
error[E0425]: cannot find function `result_groups` in this scope
error[E0425]: cannot find function `row_template` in this scope
error[E0425]: cannot find function `table_entries` in this scope
error[E0425]: cannot find value `CHAIN_LABELS` in this scope
error[E0425]: cannot find value `CHAIN_PREFIXES` in this scope
error[E0425]: cannot find value `CHECKS` in this scope
error[E0425]: cannot find value `DASHBOARD` in this scope
error[E0425]: cannot find value `SCOPE` in this scope
```

- [ ] **Step 3: Add the check levels and the groups**

In `magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// Green: the check passes.
````

with:

````rust
use crate::{DesignInputs, DesignResults, compute_all};

/// A badge colour, from a check verdict. Ordered by severity (green, amber, red): the worst of
/// several levels is their maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    /// Green: the check passes.
````

In `magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
pub const END_EFFECT_BANNER: &str = END_EFFECT_OUT_OF_RANGE;

/// The badge level of a check's verdict text; `None` (no badge) for a path that is no check, a
/// verdict that says the check does not apply (the cup wall's "No back iron"), or a text the
/// check does not produce.
pub fn verdict_level(path: &str, text: &str) -> Option<Level> {
    use Level::{Bad, Caution, Good};
    match (path, text) {
        ("metal.hot_min_check", "Estimate covers hot min") => Some(Good),
        ("metal.hot_min_check", "Below hot minimum") => Some(Bad),
````

with:

````rust
pub const END_EFFECT_BANNER: &str = END_EFFECT_OUT_OF_RANGE;

/// Every design-level check of the results, in schema order (decision O-3): the dashboard's
/// verdicts, the coupling model's, the temperature design's and the clamp's checks and
/// screens, the material warnings and the space claim. Each gives a badge level
/// ([`verdict_level`]): the results table shows it, and its failing filter lists the checks
/// that fail ([`failing_checks`]). The clamp table's per-size checks and the sweeps' status
/// texts rate candidates, not the design, so they are left out.
pub const CHECKS: [&str; 30] = [
    "calibration.end_effect_check",
    "model.inner_flat_check",
    "model.outer_flat_check",
    "model.end_effect_check",
    "model.verdict",
    "model.cup_ring_check",
    "model.hub_check",
    "model.inner_temp_check",
    "model.outer_temp_check",
    "metal.hot_min_check",
    "metal.clearance_check",
    "materials.cup_wall_check",
    "temperature.summary.torque_hot_day_note",
    TEMPERATURE_VERDICT,
    "temperature.demag.cold_check",
    "temperature.adhesive.fatigue_screen",
    "temperature.mismatch.reading",
    "temperature.magnet_life.torque_hot_day_check",
    "temperature.adhesive_life.hot_fatigue_screen",
    "temperature.adhesive_life.daily_screen",
    "clamps.recommended",
    "clamps.head_check",
    "clamps.vent_port",
    "warnings.non_ferromagnetic_back_iron",
    "warnings.ferromagnetic_sleeve_or_liner",
    "warnings.high_conductivity_sleeve_or_liner",
    "warnings.low_saturation",
    "warnings.uncoated_low_alloy_steel",
    "warnings.cte_mismatch_with_magnets",
    "housing.space_claim_check",
];

/// The badge level of a check's verdict text; `None` (no badge) for a path that is no check, a
/// verdict that says the check does not apply (the cup wall's "No back iron"), or a text the
/// check does not produce. A warning (`warnings.<rule>`) that fires has its severity's level.
pub fn verdict_level(path: &str, text: &str) -> Option<Level> {
    use Level::{Bad, Caution, Good};
    const FLATS: [&str; 2] = ["model.inner_flat_check", "model.outer_flat_check"];
    const HOT_DAY: [&str; 2] = [
        "temperature.summary.torque_hot_day_note",
        "temperature.magnet_life.torque_hot_day_check",
    ];
    match (path, text) {
        ("calibration.end_effect_check", "OK") => Some(Good),
        ("calibration.end_effect_check", END_EFFECT_OUT_OF_RANGE) => Some(Bad),
        (p, t) if FLATS.contains(&p) && t.starts_with("OK, ") => Some(Good),
        (p, t) if FLATS.contains(&p) && t.starts_with("TOO NARROW: ") => Some(Bad),
        // Arcs have no flats: the check does not apply.
        (p, "n/a (arcs)") if FLATS.contains(&p) => None,
        // The nominal pull-out covers the floor; the workbook asks for a hot test to confirm
        // it, as the temperature verdict asks for its tests (both green).
        ("model.verdict", "Nominal only: hot test") => Some(Good),
        ("model.verdict", "Below hot minimum") => Some(Bad),
        ("model.cup_ring_check" | "model.hub_check", "Thickness OK") => Some(Good),
        ("model.cup_ring_check" | "model.hub_check", "Too thin") => Some(Bad),
        // E9: without back iron no magnetic rule sizes the steel.
        ("model.cup_ring_check" | "model.hub_check", "No back iron") => None,
        ("model.inner_temp_check" | "model.outer_temp_check", "OK") => Some(Good),
        ("model.inner_temp_check" | "model.outer_temp_check", "OVER the magnet rating") => {
            Some(Bad)
        }
        // A manual magnet without a grade has no rating to check against: a look, as an
        // unknown space claim.
        ("model.inner_temp_check" | "model.outer_temp_check", "unknown") => Some(Caution),
        (p, "Meets it nominally (no variation allowance)") if HOT_DAY.contains(&p) => Some(Good),
        (p, "Below it") if HOT_DAY.contains(&p) => Some(Bad),
        ("temperature.demag.cold_check", "OK") => Some(Good),
        ("temperature.demag.cold_check", "Below the cold demagnetization limit") => Some(Bad),
        // NdFeB's coercivity rises as it cools: the cold check does not apply.
        ("temperature.demag.cold_check", t) if t.starts_with("n/a") => None,
        ("temperature.adhesive.fatigue_screen", t) if t.starts_with("OK: ") => Some(Good),
        ("temperature.adhesive.fatigue_screen", "CHECK") => Some(Caution),
        ("temperature.mismatch.reading", "Below the lap-shear strength") => Some(Good),
        ("temperature.mismatch.reading", "Above the lap-shear strength at the block ends") => {
            Some(Bad)
        }
        ("temperature.adhesive_life.hot_fatigue_screen", "OK") => Some(Good),
        ("temperature.adhesive_life.hot_fatigue_screen", "CHECK: get hot fatigue data") => {
            Some(Caution)
        }
        ("temperature.adhesive_life.daily_screen", "Below the fatigue endurance") => Some(Good),
        (
            "temperature.adhesive_life.daily_screen",
            "Above the fatigue endurance: qualify by thermal cycling",
        ) => Some(Caution),
        // The clamp's head and key checks read empty when no screw fits (the recommended
        // screw's "None:" is the red one).
        ("clamps.head_check", "OK") => Some(Good),
        ("clamps.head_check", "Use a hardened washer") => Some(Caution),
        ("clamps.vent_port", t) if t.starts_with("Yes: ") => Some(Good),
        ("clamps.vent_port", t) if t.starts_with("No: ") => Some(Caution),
        (p, t) if !t.is_empty() && p.starts_with("warnings.") => WARNING_RULES
            .iter()
            .find(|rule| p.strip_prefix("warnings.") == Some(rule.id))
            .map(|rule| severity_level(rule.severity)),
        ("metal.hot_min_check", "Estimate covers hot min") => Some(Good),
        ("metal.hot_min_check", "Below hot minimum") => Some(Bad),
````

In `magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
        _ => None,
    }
}
````

with:

````rust
        _ => None,
    }
}

/// The badge level of the check at `path` for `results`: [`verdict_level`] of its text; `None`
/// for a path that is no check or holds no text, or a verdict that gives no badge.
pub fn check_level(results: &DesignResults, path: &str) -> Option<Level> {
    match results.get(path)? {
        Value::Text(text) => verdict_level(path, &text),
        _ => None,
    }
}

/// The checks of `results` that fail (red) or ask for a look (amber), the red first, each in
/// [`CHECKS`] order: what the results table's failing filter shows.
pub fn failing_checks(results: &DesignResults) -> Vec<(&'static str, Level)> {
    let mut failing: Vec<(&'static str, Level)> = CHECKS
        .iter()
        .filter_map(|&path| match check_level(results, path) {
            Some(level @ (Level::Bad | Level::Caution)) => Some((path, level)),
            _ => None,
        })
        .collect();
    // A stable sort, the most severe first: the red, then the amber, each in CHECKS order.
    failing.sort_by_key(|&(_, level)| std::cmp::Reverse(level));
    failing
}
````

In `magcoupling-rs/src/gui/dashboard.rs`, replace:

````rust
}

/// A badge: a filled circle in the level's colour, or an empty cell.
fn badge(ui: &mut egui::Ui, level: Option<Level>) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    if let Some(level) = level {
````

with:

````rust
}

/// A badge: a filled circle in the level's colour, or an empty cell (the results table's check
/// rows draw it too).
pub(crate) fn badge(ui: &mut egui::Ui, level: Option<Level>) {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
    if let Some(level) = level {
````

In `magcoupling-rs/src/gui/result_groups.rs`, replace:

````rust
#[cfg(test)]
mod tests {
    use super::*;
````

with:

````rust
//! The groups of the results table (decision O-6): the headline first (the dashboard's rows,
//! [`DASHBOARD`]), then the physics chains: the A-3 chains of the explorer's scope ([`SCOPE`]:
//! torque, temperature, demagnetization, slip heating, clamps, geometry) and two of results the
//! scope leaves out (adhesive, mass), each chain also taking the results of its nested groups
//! ([`CHAIN_PREFIXES`]: the cold demagnetization limits, the slip temperatures, the adhesive and
//! bond screens); then every other result under its package group, the "Other results". Every
//! result is in exactly one group: a path on the dashboard and in a chain, or in two chains, sits
//! in the first group that lists it.
//!
//! The groups index [`table_entries`], built once; the CSV and JSON exports keep the engine's
//! order.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::engine::explain::scope::SCOPE;
use crate::gui::dashboard::DASHBOARD;
use crate::gui::results_table::{TableEntry, table_entries};

/// The headline group's heading.
pub const HEADLINE_LABEL: &str = "Headline";

/// The heading over the package groups of the results no chain lists.
pub const OTHER_RESULTS: &str = "Other results";

/// The chain groups, in the order shown, by id, with their headings: the chains of the
/// explorer's scope in its order (its `dashboard` entry is the headline's), with the adhesive
/// after slip heating and the mass last. `adhesive` and `mass` are no chain of the scope: only
/// [`CHAIN_PREFIXES`] fills them.
pub const CHAIN_LABELS: [(&str, &str); 8] = [
    ("torque", "Torque"),
    ("temperature", "Temperature"),
    ("demagnetization", "Demagnetization"),
    ("slip_heating", "Slip heating"),
    ("adhesive", "Adhesive"),
    ("clamps", "Clamps"),
    ("geometry", "Geometry"),
    ("mass", "Mass"),
];

/// The chain each result the scope leaves out goes to, by path prefix, before the other results
/// (decision O-6): a nested group of a package (`temperature.demag.`), the stem of two
/// (`temperature.adhesive` takes `temperature.adhesive.` and `temperature.adhesive_life.`), or
/// one result's whole path (`metal.corner_gap_mm`). A result goes to the chain of the first
/// entry its path starts with, after the chain's own rows, in schema order; a test checks that
/// every entry places a result.
pub const CHAIN_PREFIXES: [(&str, &str); 19] = [
    ("temperature.demag.", "demagnetization"),
    // The summary rows the temperature chain leaves out: the steady temperatures while slipping,
    // the peak with the slip fault, the slip rotations and the average slip heating over life.
    ("temperature.summary.", "slip_heating"),
    ("temperature.magnet_life.", "temperature"),
    ("temperature.thermal.", "slip_heating"),
    ("temperature.slip_life.", "slip_heating"),
    ("temperature.duty.", "slip_heating"),
    ("temperature.adhesive", "adhesive"),
    ("temperature.mismatch.", "adhesive"),
    ("mass.", "mass"),
    // The clearances, gaps, diameters and reserves of the metal design.
    ("metal.running_clearance_mm", "geometry"),
    ("metal.corner_gap_mm", "geometry"),
    ("metal.assembled_face_gap_mm", "geometry"),
    ("metal.corner_clearance_mm", "geometry"),
    ("metal.nominal_sleeve_liner_mm", "geometry"),
    ("metal.allowed_radial_disp_mm", "geometry"),
    ("metal.cup_body_od_mm", "geometry"),
    ("metal.diameter_reserve_mm", "geometry"),
    ("metal.large_dia_reserve_mm", "geometry"),
    ("metal.axial_reserve_mm", "geometry"),
];

/// The heading of each package group of the other results, by the first segment of the path
/// (a table's index dropped: `gap_sweep[3].f_end` is in `gap_sweep`), in schema order.
pub const PACKAGE_LABELS: [(&str, &str); 12] = [
    ("calibration", "Calibration"),
    ("model", "Coupling model"),
    ("mass", "Mass"),
    ("retainers", "Retainers"),
    ("metal", "Metal design"),
    ("materials", "Materials"),
    ("temperature", "Temperature design"),
    ("clamps", "Shaft clamps"),
    ("warnings", "Material warnings"),
    ("housing", "Housing"),
    ("gap_sweep", "Gap sweep"),
    ("pole_sweep", "Pole sweep"),
];

/// One group of the results table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultGroup {
    /// `headline`, a chain's id, or a package's (`calibration`, `gap_sweep`). A chain and a
    /// package may share an id (`temperature`, `clamps`): `other` tells them apart.
    pub id: &'static str,
    pub label: &'static str,
    /// Under the "Other results" heading: a package group.
    pub other: bool,
    /// Its rows, as indices into [`table_entries`], in the order shown.
    pub rows: Vec<usize>,
}

/// A path with each table index emptied, as the explorer's scope writes a table's every row:
/// `clamps.table[3].preload_N` gives `clamps.table[].preload_N`; a path without an index is
/// unchanged.
pub fn row_template(path: &str) -> String {
    let mut template = String::with_capacity(path.len());
    let mut in_index = false;
    for c in path.chars() {
        match c {
            '[' => {
                in_index = true;
                template.push(c);
            }
            ']' => {
                in_index = false;
                template.push(c);
            }
            _ if in_index => {}
            _ => template.push(c),
        }
    }
    template
}

/// The package of a path: its first segment, a table's index dropped.
pub fn package_of(path: &str) -> &str {
    let first = path.split('.').next().unwrap_or(path);
    first.split('[').next().unwrap_or(first)
}

/// The heading of a package; `None` if [`PACKAGE_LABELS`] lacks it (a test checks none does).
pub fn package_label(package: &str) -> Option<&'static str> {
    PACKAGE_LABELS
        .iter()
        .find(|(id, _)| *id == package)
        .map(|(_, label)| *label)
}

/// The groups of `entries`: the headline, the chains (each its scope rows, then the rows its
/// [`CHAIN_PREFIXES`] place), then the other results by package; a group left empty (every row
/// in an earlier group) is dropped.
pub fn groups_of(entries: &[TableEntry]) -> Vec<ResultGroup> {
    let mut by_template: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, entry) in entries.iter().enumerate() {
        by_template
            .entry(row_template(&entry.path))
            .or_default()
            .push(index);
    }
    let mut placed = vec![false; entries.len()];
    // The rows of `paths` (a table's template gives each of its rows) not placed yet, in order.
    let mut take = |paths: &mut dyn Iterator<Item = &str>| -> Vec<usize> {
        let mut rows = Vec::new();
        for path in paths {
            for &index in by_template.get(path).map_or(&[][..], Vec::as_slice) {
                if !placed[index] {
                    placed[index] = true;
                    rows.push(index);
                }
            }
        }
        rows
    };
    let mut groups = vec![ResultGroup {
        id: "headline",
        label: HEADLINE_LABEL,
        other: false,
        rows: take(&mut DASHBOARD.iter().map(|(path, _)| *path)),
    }];
    for (id, label) in CHAIN_LABELS {
        // A chain of the scope starts with its rows; the adhesive and the mass start empty.
        let rows = match SCOPE.iter().find(|chain| chain.id == id) {
            Some(chain) => take(&mut chain.paths.iter().copied()),
            None => Vec::new(),
        };
        groups.push(ResultGroup {
            id,
            label,
            other: false,
            rows,
        });
    }
    for (index, entry) in entries.iter().enumerate() {
        if placed[index] {
            continue;
        }
        let Some(&(_, chain)) = CHAIN_PREFIXES
            .iter()
            .find(|(prefix, _)| entry.path.starts_with(prefix))
        else {
            continue;
        };
        let group = groups
            .iter_mut()
            .find(|group| !group.other && group.id == chain)
            .unwrap_or_else(|| panic!("CHAIN_PREFIXES: no chain {chain} in CHAIN_LABELS"));
        group.rows.push(index);
        placed[index] = true;
    }
    for (id, label) in PACKAGE_LABELS {
        let rows: Vec<usize> = (0..entries.len())
            .filter(|&index| !placed[index] && package_of(&entries[index].path) == id)
            .collect();
        groups.push(ResultGroup {
            id,
            label,
            other: true,
            rows,
        });
    }
    groups.retain(|group| !group.rows.is_empty());
    groups
}

/// The groups of the results table, built once (the layout of the results does not depend on
/// the inputs).
pub fn result_groups() -> &'static [ResultGroup] {
    static GROUPS: OnceLock<Vec<ResultGroup>> = OnceLock::new();
    GROUPS.get_or_init(|| groups_of(table_entries()))
}

#[cfg(test)]
mod tests {
    use super::*;
````

- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib -- gui::dashboard gui::result_groups 2>&1 | grep -E "^test |test result"
```

Expected: 18 tests `ok`: the 14 of `gui::dashboard::tests` (the new `every_verdict_of_every_other_check_is_classified`, `every_check_is_a_text_result_and_every_result_named_as_a_check_is_listed`, `the_failing_checks_are_the_red_then_the_amber` and the 11 there before, `every_verdict_each_check_gives_is_classified` among them) and the 4 of `gui::result_groups::tests` (`every_result_is_in_exactly_one_group`, `the_headline_comes_first_then_the_chains_then_the_other_results`, `every_chain_path_names_results_and_every_package_has_a_heading`, `every_chain_prefix_places_a_result_the_scope_leaves_out`); then `test result: ok. 18 passed; 0 failed`.

- [ ] **Step 5: Run the checks and the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --check
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED"
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task3.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
```

Expected: `cargo fmt --check` prints nothing; `test result: ok. 463 passed; 0 failed`; `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order add magcoupling-rs/src/gui/dashboard.rs magcoupling-rs/src/gui/result_groups.rs magcoupling-rs/src/gui/mod.rs
git -C C:/Users/Cole/source/repos/lsim-mag-order commit -q -F - <<'EOF'
feat(magcoupling-rs): a level for every design check, and the results groups (O-3, O-6)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-order status --short
```

Expected: no status lines.

---

### Task 4: The results table by group, with badges and the failing filter

**Model:** `sonnet` (exact code; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/src/gui/results_table.rs` (module doc; imports; `FAILING_ONLY`, `NOTHING_FAILS`, `NO_RESULT`, `CLEAR_TO_CLOSE`, `OTHER_INDENT`; `TableEntry.check`; `entry_index`, `ResultOrder`, `Line`, `table_lines` before `search`; `ResultsTable` fields, `opens_by_default`, `is_open` (with `searching` and `is_open_while`, worked out once a frame), `ui` rewritten; `HeadingLine`, `group_heading_ui` with the badge of the group's worst check; `row_ui` with the badge; three tests)
- Modify: `magcoupling-rs/src/gui/panel.rs` (tests: `the_results_table_lists_results_and_filters_by_the_search`, three new tests (`a_group_heading_opens_its_rows_and_the_engine_order_lists_them_without_headings` with the heading badges, `a_heading_click_while_searching_leaves_the_group_as_it_was`, `the_failing_filter_shows_exactly_the_failing_checks_with_their_badges` with the empty state), the glyph test)

**Interfaces:**
- Consumes (Task 3): `CHECKS`, `Level`, `badge`, `check_level`, `failing_checks` (dashboard); `result_groups`, `OTHER_RESULTS`, `ResultGroup { label, other, rows, .. }` (result_groups); (Task 2) `search_needle`.
- Produces: `TableEntry.check: bool`; `pub fn entry_index(path: &str) -> Option<usize>`; `pub enum ResultOrder { Grouped (default), Engine }` with `ALL`, `label` ("By physics chain", "Engine order"); `pub enum Line { OtherResults, Group { group: usize, shown: usize, open: bool }, Row(usize) }`; `pub fn table_lines(matches: &[usize], order: ResultOrder, failing: Option<&[usize]>, open: &dyn Fn(usize) -> bool) -> Vec<Line>`; `ResultsTable::opens_by_default(group: usize) -> bool` (the headline only) and `ResultsTable::is_open(&self, group: usize) -> bool`; `pub const FAILING_ONLY: &str = "Failing checks only"`; `pub const CLEAR_TO_CLOSE: &str` (a heading's hover text while searching, when a click on it is ignored); a group heading's text `"{label} ({shown})"` followed by the badge of the worst level of the group's checks. `ResultsTable::ui(&mut self, ui, results, readouts)` keeps its signature (Task 5 adds the trace).

- [ ] **Step 1: Write the failing tests**

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
        let claim = rows
            .iter()
            .find(|r| r["path"] == "housing.space_claim_check")
            .unwrap();
        assert_eq!(claim["cell"], Json::Null);
    }
}
````

with:

````rust
        let claim = rows
            .iter()
            .find(|r| r["path"] == "housing.space_claim_check")
            .unwrap();
        assert_eq!(claim["cell"], Json::Null);
    }

    /// The row indices of `lines`, headings left out.
    fn rows_of(lines: &[Line]) -> Vec<usize> {
        lines
            .iter()
            .filter_map(|line| match line {
                Line::Row(index) => Some(*index),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn the_lines_group_the_rows_with_only_the_headline_open_at_first() {
        let entries = table_entries();
        let groups = result_groups();
        let all: Vec<usize> = (0..entries.len()).collect();
        let lines = table_lines(
            &all,
            ResultOrder::Grouped,
            None,
            &ResultsTable::opens_by_default,
        );
        // The headline's heading and its rows, then every other group's closed heading, the
        // package groups under "Other results".
        let headline = groups[0].rows.len();
        assert_eq!(
            lines[0],
            Line::Group {
                group: 0,
                shown: headline,
                open: true
            }
        );
        assert_eq!(rows_of(&lines[1..=headline]), groups[0].rows);
        let mut want = Vec::new();
        for (group, entry) in groups.iter().enumerate().skip(1) {
            if entry.other && !want.contains(&Line::OtherResults) {
                want.push(Line::OtherResults);
            }
            want.push(Line::Group {
                group,
                shown: entry.rows.len(),
                open: false,
            });
        }
        assert_eq!(lines[headline + 1..], want[..]);
        // Every group open: every row once.
        let lines = table_lines(&all, ResultOrder::Grouped, None, &|_| true);
        let mut rows = rows_of(&lines);
        assert_eq!(rows.len(), entries.len());
        rows.sort_unstable();
        assert_eq!(rows, all);
    }

    #[test]
    fn a_search_shows_only_the_groups_it_matches() {
        let entries = table_entries();
        let groups = result_groups();
        let pullout = entry_index("model.pullout_Nm").unwrap();
        assert_eq!(entries[pullout].path, "model.pullout_Nm");
        let matches = search(entries, "calculator!c93");
        let lines = table_lines(&matches, ResultOrder::Grouped, None, &|_| true);
        assert_eq!(
            lines,
            [
                Line::Group {
                    group: 0,
                    shown: 1,
                    open: true
                },
                Line::Row(pullout)
            ]
        );
        // A sweep row's columns: the Other results heading, the gap sweep's heading, the rows.
        let matches = search(entries, "gap_sweep[3].");
        let lines = table_lines(&matches, ResultOrder::Grouped, None, &|_| true);
        let gap_sweep = groups.iter().position(|g| g.id == "gap_sweep").unwrap();
        assert_eq!(lines[0], Line::OtherResults);
        assert_eq!(
            lines[1],
            Line::Group {
                group: gap_sweep,
                shown: matches.len(),
                open: true
            }
        );
        assert_eq!(rows_of(&lines), matches);
        assert!(table_lines(&[], ResultOrder::Grouped, None, &|_| true).is_empty());
        // The search opens every group it matches, whatever the user toggled.
        let mut table = ResultsTable::default();
        assert!(table.is_open(0) && !table.is_open(1));
        table.toggled.insert(0);
        assert!(!table.is_open(0));
        table.query = " f_end ".to_owned();
        assert!(table.is_open(0) && table.is_open(1));
        assert!(!ResultsTable::opens_by_default(1));
    }

    #[test]
    fn the_engine_order_and_the_failing_filter_are_flat() {
        let entries = table_entries();
        let all: Vec<usize> = (0..entries.len()).collect();
        let lines = table_lines(&all, ResultOrder::Engine, None, &|_| false);
        assert_eq!(rows_of(&lines), all);
        assert_eq!(lines.len(), all.len(), "no headings");
        // The failing filter: exactly the failing checks, the red first, in either order.
        let mut inputs = DesignInputs::default();
        inputs.coupling.magnets.part_inner.clear();
        inputs.coupling.magnets.part_outer.clear();
        let results = compute_all(&inputs);
        let failing: Vec<usize> = failing_checks(&results)
            .iter()
            .map(|(path, _)| entry_index(path).unwrap())
            .collect();
        assert_eq!(failing.len(), 6);
        for order in ResultOrder::ALL {
            let lines = table_lines(&all, order, Some(&failing), &|_| true);
            assert_eq!(
                lines,
                failing.iter().map(|&i| Line::Row(i)).collect::<Vec<_>>()
            );
        }
        // And only those the search matches: the two amber rating checks.
        let matches = search(entries, "temperature check");
        let lines = table_lines(&matches, ResultOrder::Grouped, Some(&failing), &|_| true);
        assert_eq!(
            lines,
            [
                Line::Row(entry_index("model.inner_temp_check").unwrap()),
                Line::Row(entry_index("model.outer_temp_check").unwrap())
            ]
        );
        // Each check's row knows it is one.
        for check in CHECKS {
            assert!(entries[entry_index(check).unwrap()].check, "{check}");
        }
        assert!(!entries[entry_index("model.pullout_Nm").unwrap()].check);
        assert_eq!(entry_index("no.such.result"), None);
    }
}
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    #[test]
    fn the_results_table_lists_results_and_filters_by_the_search() {
        let entries = crate::gui::results_table::table_entries();
        let cell = |index: usize| entries[index].info.cell.clone().unwrap();
        let (first, last) = (cell(0), cell(entries.len() - 1));
        let mut harness = Harness::new();
        let output = harness.click_text(CentreView::Results.label());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // The first rows are on screen (they show their cells), the last is far below.
        assert_eq!(count(&output, &first), 1, "{first}");
        assert_eq!(count(&output, &last), 0, "{last}");
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
````

with:

````rust
    #[test]
    fn the_results_table_lists_results_and_filters_by_the_search() {
        let entries = crate::gui::results_table::table_entries();
        let cell = |index: usize| entries[index].info.cell.clone().unwrap();
        let (first, last) = (cell(0), cell(entries.len() - 1));
        let mut harness = Harness::new();
        let output = harness.click_text(CentreView::Results.label());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // The headline's rows are on screen (the pull-out shows its cell); the first and the
        // last rows of the schema are in closed groups.
        assert_eq!(count(&output, "Calculator!C93"), 1);
        assert_eq!(count(&output, &first), 0, "{first}");
        assert_eq!(count(&output, &last), 0, "{last}");
        harness.click_text(crate::gui::results_table::SEARCH_HINT);
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    #[test]
    fn a_narrow_window_shows_each_row_s_label_and_value_without_scrolling() {
````

with:

````rust
    #[test]
    fn a_group_heading_opens_its_rows_and_the_engine_order_lists_them_without_headings() {
        use crate::gui::result_groups::{OTHER_RESULTS, result_groups};
        use crate::gui::results_table::ResultOrder;
        let groups = result_groups();
        let heading = |g: usize| format!("{} ({})", groups[g].label, groups[g].rows.len());
        // Tall enough to draw the headline and the whole torque chain.
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        let output = harness.frame(Vec::new());
        for g in 0..groups.len() {
            assert_eq!(count(&output, &heading(g)), 1, "{}", heading(g));
        }
        assert_eq!(count(&output, OTHER_RESULTS), 1);
        // A heading shows the worst level of its group's checks after its text: the defaults
        // fail the hot minimum on the headline and the cup ring check in the closed coupling
        // model group, so both draw a red badge; the mass has no check and draws none.
        let visuals = harness.ctx.style().visuals.clone();
        let badges_after = |output: &egui::FullOutput, text: &str, level: Level| {
            let text = text_rect(output, text).unwrap();
            crate::gui::test_support::flat_shapes(output)
                .into_iter()
                .filter(|shape| {
                    matches!(shape, egui::Shape::Circle(c)
                        if c.fill == level.color(&visuals)
                            && c.center.x > text.right()
                            && c.center.x < text.right() + 30.0
                            && (c.center.y - text.center().y).abs() < text.height())
                })
                .count()
        };
        let model = groups
            .iter()
            .position(|g| g.other && g.id == "model")
            .unwrap();
        let mass = groups
            .iter()
            .position(|g| !g.other && g.id == "mass")
            .unwrap();
        assert_eq!(badges_after(&output, &heading(0), Level::Bad), 1);
        assert_eq!(badges_after(&output, &heading(model), Level::Bad), 1);
        for level in [Level::Good, Level::Caution, Level::Bad] {
            assert_eq!(badges_after(&output, &heading(mass), level), 0);
        }
        // The torque chain starts closed: its end-effect factor (Calculator!C92) is not drawn
        // until its heading is clicked, and is gone again after a second click.
        assert_eq!(groups[1].id, "torque");
        assert_eq!(count(&output, "Calculator!C92"), 0);
        harness.click_text(&heading(1));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 1);
        harness.click_text(&heading(1));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 0);
        // The engine order: the schema's first row first, no headings.
        harness.click_text(ResultOrder::Engine.label());
        let output = harness.frame(Vec::new());
        let first = crate::gui::results_table::table_entries()[0]
            .info
            .cell
            .clone()
            .unwrap();
        assert_eq!(count(&output, &first), 1);
        assert_eq!(count(&output, &heading(0)), 0);
        assert_eq!(count(&output, OTHER_RESULTS), 0);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn a_heading_click_while_searching_leaves_the_group_as_it_was() {
        use crate::gui::result_groups::result_groups;
        use crate::gui::results_table::{CLEAR_TO_CLOSE, SEARCH_HINT, search, table_entries};
        use crate::gui::test_support::select_all;
        // A search opens every group it matches, so a click on a heading would change nothing
        // on screen: it is ignored, and once the search is cleared the torque chain is closed,
        // as it started (the click did not toggle it behind the search).
        let entries = table_entries();
        let torque = &result_groups()[1];
        assert_eq!(torque.id, "torque");
        let matches = search(entries, "f_end");
        let shown = torque.rows.iter().filter(|i| matches.contains(i)).count();
        let heading = format!("{} ({shown})", torque.label);
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        tooltips_at_once(&harness);
        harness.click_text(CentreView::Results.label());
        harness.click_text(SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("f_end".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 1, "the search opens it");
        let output = harness.click_text(&heading);
        let at = text_rect(&output, &heading).unwrap().center();
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, "Calculator!C92"), 1, "still open");
        // Its hover text says why (egui shows no tooltip until the pointer moves after a click).
        harness.frame_after(
            0.5,
            vec![egui::Event::PointerMoved(at + egui::vec2(2.0, 0.0))],
        );
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, CLEAR_TO_CLOSE), 1);
        // Clear the search: the torque chain is closed again.
        harness.click_text("f_end");
        harness.frame(select_all());
        harness.frame(key_tap(egui::Key::Backspace));
        let output = harness.frame(Vec::new());
        let total = entries.len();
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        assert_eq!(count(&output, &format!("{} (49)", torque.label)), 1);
        assert_eq!(count(&output, "Calculator!C92"), 0);
    }

    #[test]
    fn the_failing_filter_shows_exactly_the_failing_checks_with_their_badges() {
        use crate::gui::dashboard::{CHECKS, failing_checks};
        use crate::gui::results_table::{
            FAILING_ONLY, NO_RESULT, NOTHING_FAILS, SEARCH_HINT, table_entries,
        };
        // Manual magnets: the defaults' four red checks and two amber rating checks.
        let mut harness = Harness::new();
        harness.panel.inputs.coupling.magnets.part_inner.clear();
        harness.panel.inputs.coupling.magnets.part_outer.clear();
        harness.click_text(CentreView::Results.label());
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        let failing = failing_checks(harness.panel.results());
        assert_eq!(failing.len(), 6);
        let total = table_entries().len();
        assert_eq!(count(&output, &format!("6 of {total} results")), 1);
        // In the table, between the inputs and the dashboard: each failing check's label, and
        // no passing check's (two checks may share a label: "Verdict").
        let centre =
            |r: &egui::Rect| r.left() > INPUTS_WIDTH && r.right() < SCREEN.x - DASHBOARD_WIDTH;
        let label = |path: &str| result_info(path).unwrap().meta.label;
        for check in CHECKS {
            let drawn = text_rects(&output, label(check))
                .iter()
                .filter(|r| centre(r))
                .count();
            let want = failing
                .iter()
                .filter(|(path, _)| label(path) == label(check))
                .count();
            assert_eq!(drawn, want, "{check}");
        }
        // Four red badges and two amber in the table, in the dashboard's colours: every badge
        // drawn but the dashboard's own.
        let visuals = harness.ctx.style().visuals.clone();
        let lines = crate::gui::dashboard::dashboard_lines(harness.panel.results());
        let table_badges = |level: Level| {
            let drawn = crate::gui::test_support::flat_shapes(&output)
                .into_iter()
                .filter(|shape| matches!(shape, egui::Shape::Circle(c) if c.fill == level.color(&visuals)))
                .count();
            drawn
                - lines
                    .iter()
                    .filter(|line| line.level == Some(level))
                    .count()
        };
        assert_eq!(
            (table_badges(Level::Bad), table_badges(Level::Caution)),
            (4, 2)
        );
        // Off again: the groups come back.
        harness.click_text(FAILING_ONLY);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        assert_eq!(count(&output, crate::gui::result_groups::OTHER_RESULTS), 1);
        // On, with a search no failing check matches: the table says the search matches
        // nothing, not that no check fails.
        harness.click_text(FAILING_ONLY);
        harness.click_text(SEARCH_HINT);
        harness.frame(vec![egui::Event::Text("gearbox".to_owned())]);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("0 of {total} results")), 1);
        assert_eq!(count(&output, NO_RESULT), 1);
        assert_eq!(count(&output, NOTHING_FAILS), 0);
    }

    #[test]
    fn a_narrow_window_shows_each_row_s_label_and_value_without_scrolling() {
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        texts.push(ADVANCED_HEADING.to_owned());
        texts.push(FILTER_HINT.to_owned());
````

with:

````rust
        texts.push(ADVANCED_HEADING.to_owned());
        texts.push(FILTER_HINT.to_owned());
        texts.push(crate::gui::results_table::NOTHING_FAILS.to_owned());
        texts.push(crate::gui::results_table::NO_RESULT.to_owned());
        texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());
````

- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib 2>&1 | grep -E "^error" | sort | uniq
```

Expected: FAIL to compile, with these lines:

```
error: could not compile `magcoupling-rs` (lib test) due to 54 previous errors
error[E0412]: cannot find type `Line` in this scope
error[E0425]: cannot find function `entry_index` in this scope
error[E0425]: cannot find function `failing_checks` in this scope
error[E0425]: cannot find function `result_groups` in this scope
error[E0425]: cannot find function `table_lines` in this scope
error[E0425]: cannot find value `CHECKS` in this scope
error[E0425]: cannot find value `CLEAR_TO_CLOSE` in module `crate::gui::results_table`
error[E0425]: cannot find value `NOTHING_FAILS` in module `crate::gui::results_table`
error[E0425]: cannot find value `NO_RESULT` in module `crate::gui::results_table`
error[E0432]: unresolved import `crate::gui::results_table::CLEAR_TO_CLOSE`
error[E0432]: unresolved import `crate::gui::results_table::ResultOrder`
error[E0432]: unresolved imports `crate::gui::results_table::FAILING_ONLY`, `crate::gui::results_table::NO_RESULT`, `crate::gui::results_table::NOTHING_FAILS`
error[E0433]: failed to resolve: use of undeclared type `Line`
error[E0433]: failed to resolve: use of undeclared type `ResultOrder`
error[E0599]: no function or associated item named `opens_by_default` found for struct `results_table::ResultsTable` in the current scope
error[E0599]: no method named `is_open` found for struct `results_table::ResultsTable` in the current scope
error[E0609]: no field `toggled` on type `results_table::ResultsTable`
```

- [ ] **Step 3: Draw the table by group, with badges and the failing filter**

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
//! Every result, scalars and table rows, in the Python schema order ([`result_rows`]). The
//! layout of the results does not depend on the inputs, so the rows (path, metadata, cell,
//! marker, search text) are built once; a frame reads only the values of the rows on screen.
//! The exports write every result at full precision: CSV for spreadsheets, JSON with the
//! design that produced it. JSON has no infinity or NaN, so both write a non-finite number as
//! `+inf`, `-inf` or `NaN` (decision M41-15).
````

with:

````rust
//! Every result, scalars and table rows. The layout of the results does not depend on the
//! inputs, so the rows (path, metadata, cell, marker, search text) are built once in the Python
//! schema order ([`result_rows`]); a frame reads only the values of the rows on screen. The
//! table shows them by group (decision O-6, [`crate::gui::result_groups`]: the headline open,
//! every other group a heading that opens it) or in the engine's order ([`ResultOrder`]); a
//! check's row carries the dashboard's badge, and the failing filter lists the failing checks
//! alone, the red first (decision O-7). The exports write every result in schema order at full
//! precision: CSV for spreadsheets, JSON with the design that produced it. JSON has no infinity
//! or NaN, so both write a non-finite number as `+inf`, `-inf` or `NaN` (decision M41-15).
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
use std::sync::OnceLock;

use serde_json::{Map, Value as Json};
````

with:

````rust
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use serde_json::{Map, Value as Json};
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
use crate::gui::dashboard::{ResultInfo, hover_text, result_info};
````

with:

````rust
use crate::gui::dashboard::{
    CHECKS, Level, ResultInfo, badge, check_level, failing_checks, hover_text, result_info,
};
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
use crate::gui::readouts::Readouts;
use crate::gui::session::{Design, design_json, json_value};
````

with:

````rust
use crate::gui::readouts::Readouts;
use crate::gui::result_groups::{OTHER_RESULTS, result_groups};
use crate::gui::session::{Design, design_json, json_value};
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";
````

with:

````rust
/// The search box's hint.
pub const SEARCH_HINT: &str = "Search label, path or cell";

/// The failing filter's checkbox (decision O-7).
pub const FAILING_ONLY: &str = "Failing checks only";

/// What the table says when the failing filter finds no check to show.
pub const NOTHING_FAILS: &str = "No check fails or asks for a look.";

/// What the table says when the search matches no result.
pub const NO_RESULT: &str = "No result matches the search.";

/// A group heading's hover text while the search holds more than blanks: every group is open
/// then, and a click on a heading does nothing.
pub const CLEAR_TO_CLOSE: &str = "Clear the search to close a group";

/// How far a package group's heading sits in under the "Other results" heading [points].
pub const OTHER_INDENT: f32 = 12.0;
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
    /// The corrections' marker text, empty for none.
    pub marker: String,
    /// Label, path and cell, lowercase: what the search matches.
    haystack: String,
}
````

with:

````rust
    /// The corrections' marker text, empty for none.
    pub marker: String,
    /// One of the design's checks ([`CHECKS`]): its row carries a badge.
    pub check: bool,
    /// Label, path and cell, lowercase: what the search matches.
    haystack: String,
}
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
                TableEntry {
                    marker: marker_text(CorrectionIndex::get().marks(info.cell.as_deref())),
                    path: row.path,
````

with:

````rust
                TableEntry {
                    marker: marker_text(CorrectionIndex::get().marks(info.cell.as_deref())),
                    check: CHECKS.contains(&row.path.as_str()),
                    path: row.path,
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// The indices of the rows whose label, path or cell contains `query`, ignoring case and the
/// surrounding blanks; every row for a blank query.
````

with:

````rust
/// The index of the row of the result at `path` in [`table_entries`].
pub fn entry_index(path: &str) -> Option<usize> {
    static INDEX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    INDEX
        .get_or_init(|| {
            table_entries()
                .iter()
                .enumerate()
                .map(|(index, entry)| (entry.path.as_str(), index))
                .collect()
        })
        .get(path)
        .copied()
}

/// How the table orders its rows (decision O-6).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ResultOrder {
    /// By group: the headline, the physics chains, then the other results by package.
    #[default]
    Grouped,
    /// The engine's (Python's schema) order, as the exports write it.
    Engine,
}

impl ResultOrder {
    /// Both orders, in toggle order.
    pub const ALL: [ResultOrder; 2] = [ResultOrder::Grouped, ResultOrder::Engine];

    /// The toggle's text.
    pub const fn label(self) -> &'static str {
        match self {
            ResultOrder::Grouped => "By physics chain",
            ResultOrder::Engine => "Engine order",
        }
    }
}

/// One line the table draws, all of one height.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Line {
    /// The heading over the package groups.
    OtherResults,
    /// A group's heading: the group (an index into [`result_groups`]), the rows of it the
    /// filters let through, and whether it is open (its rows follow).
    Group {
        group: usize,
        shown: usize,
        open: bool,
    },
    /// A result's row (an index into [`table_entries`]).
    Row(usize),
}

/// The lines of the table: the rows of `matches` (the search's, ascending). With `failing`
/// (the failing filter on: the failing checks' rows, the red first) those of them the search
/// matches, flat; else in `order`: flat in the engine's, or by group, each group with a match
/// under its heading (the package groups under "Other results"), its rows when `open` says so.
pub fn table_lines(
    matches: &[usize],
    order: ResultOrder,
    failing: Option<&[usize]>,
    open: &dyn Fn(usize) -> bool,
) -> Vec<Line> {
    let mut admitted = vec![false; table_entries().len()];
    for &index in matches {
        admitted[index] = true;
    }
    if let Some(failing) = failing {
        return failing
            .iter()
            .copied()
            .filter(|&index| admitted[index])
            .map(Line::Row)
            .collect();
    }
    match order {
        ResultOrder::Engine => matches.iter().copied().map(Line::Row).collect(),
        ResultOrder::Grouped => {
            let mut lines = Vec::new();
            // The "Other results" heading, before the first package group with a match.
            let mut other_started = false;
            for (group, entry) in result_groups().iter().enumerate() {
                let rows: Vec<usize> = entry
                    .rows
                    .iter()
                    .copied()
                    .filter(|&index| admitted[index])
                    .collect();
                if rows.is_empty() {
                    continue;
                }
                if entry.other && !other_started {
                    lines.push(Line::OtherResults);
                    other_started = true;
                }
                let open = open(group);
                lines.push(Line::Group {
                    group,
                    shown: rows.len(),
                    open,
                });
                if open {
                    lines.extend(rows.into_iter().map(Line::Row));
                }
            }
            lines
        }
    }
}

/// The indices of the rows whose label, path or cell contains `query`, ignoring case and the
/// surrounding blanks; every row for a blank query.
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// The table's state: the search text and the rows it matches.
#[derive(Clone, Debug, Default)]
pub struct ResultsTable {
    query: String,
    /// The rows matching `matched_query`; `None` until the first frame.
    matches: Option<Vec<usize>>,
    matched_query: String,
}
````

with:

````rust
/// The table's state: the search text and the rows it matches, the order, the failing filter
/// and the groups opened or closed.
#[derive(Clone, Debug, Default)]
pub struct ResultsTable {
    query: String,
    /// The rows matching `matched_query`; `None` until the first frame.
    matches: Option<Vec<usize>>,
    matched_query: String,
    order: ResultOrder,
    /// The failing checks alone (decision O-7).
    failing_only: bool,
    /// The groups (indices into [`result_groups`]) the user opened or closed: each starts as
    /// [`ResultsTable::opens_by_default`] says.
    toggled: BTreeSet<usize>,
}
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
    /// The search text.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Draws the table: the search box and the export buttons, then the rows on screen (the
    /// end-effect banner is the centre region's, over every view: decision M42-1), each row a
    /// readout (`readouts`). Returns an export asked for.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        results: &DesignResults,
        readouts: &mut Readouts,
    ) -> Option<TableAction> {
        let entries = table_entries();
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.query)
                    .hint_text(SEARCH_HINT)
                    .desired_width(260.0),
            );
            if self.matches.is_none() || self.query != self.matched_query {
                self.matches = Some(search(entries, &self.query));
                self.matched_query = self.query.clone();
            }
            let shown = self.matches.as_ref().map_or(0, Vec::len);
            ui.weak(format!("{shown} of {} results", entries.len()));
            if ui.button(EXPORT_CSV).clicked() {
                action = Some(TableAction::ExportCsv);
            }
            if ui.button(EXPORT_JSON).clicked() {
                action = Some(TableAction::ExportJson);
            }
        });
        ui.separator();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        let widths = column_widths(ui.available_width(), ui.spacing().item_spacing.x);
        egui::ScrollArea::both()
            .id_salt("magcoupling_results_scroll")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, matches.len(), |ui, range| {
                for &index in &matches[range] {
                    row_ui(ui, &entries[index], results, row_height, widths, readouts);
                }
            });
        action
    }
}
````

with:

````rust
    /// The search text.
    pub fn query(&self) -> &str {
        &self.query
    }

    /// Whether a group starts open: the headline does, every other group starts closed.
    pub fn opens_by_default(group: usize) -> bool {
        group == 0
    }

    /// Whether a group is open: as it starts, unless the user toggled it, and every group while
    /// the search holds more than blanks (its matches are what the user is after).
    pub fn is_open(&self, group: usize) -> bool {
        self.is_open_while(group, self.searching())
    }

    /// Whether the search holds more than blanks.
    fn searching(&self) -> bool {
        !search_needle(&self.query).is_empty()
    }

    /// [`ResultsTable::is_open`] given [`ResultsTable::searching`], which a frame works out
    /// once for every group.
    fn is_open_while(&self, group: usize, searching: bool) -> bool {
        let default = Self::opens_by_default(group);
        let toggled = self.toggled.contains(&group);
        let chosen = if toggled { !default } else { default };
        chosen || searching
    }

    /// Draws the table: the search box, the order toggle, the failing filter, the count and the
    /// export buttons, then the lines on screen (the end-effect banner is the centre region's,
    /// over every view: decision M42-1), each row a readout (`readouts`), each group heading a
    /// button that opens or closes it (not while the search holds more than blanks: every group
    /// is open then), with the worst level of its checks. Returns an export asked for.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        results: &DesignResults,
        readouts: &mut Readouts,
    ) -> Option<TableAction> {
        let entries = table_entries();
        let mut action = None;
        // The failing checks' rows, the red first, when the filter is on (after its checkbox).
        let mut failing: Option<Vec<usize>> = None;
        ui.horizontal_wrapped(|ui| {
            ui.add(
                egui::TextEdit::singleline(&mut self.query)
                    .hint_text(SEARCH_HINT)
                    .desired_width(260.0),
            );
            for order in ResultOrder::ALL {
                ui.selectable_value(&mut self.order, order, order.label());
            }
            ui.checkbox(&mut self.failing_only, FAILING_ONLY)
                .on_hover_text(
                    "The checks that fail (red) or ask for a look (amber), the red first",
                );
            if self.matches.is_none() || self.query != self.matched_query {
                self.matches = Some(search(entries, &self.query));
                self.matched_query = self.query.clone();
            }
            let matches = self.matches.as_deref().unwrap_or(&[]);
            failing = self.failing_only.then(|| {
                failing_checks(results)
                    .iter()
                    .filter_map(|(path, _)| entry_index(path))
                    .collect()
            });
            let shown = match &failing {
                Some(rows) => rows
                    .iter()
                    .filter(|index| matches.binary_search(index).is_ok())
                    .count(),
                None => matches.len(),
            };
            ui.weak(format!("{shown} of {} results", entries.len()));
            if ui.button(EXPORT_CSV).clicked() {
                action = Some(TableAction::ExportCsv);
            }
            if ui.button(EXPORT_JSON).clicked() {
                action = Some(TableAction::ExportJson);
            }
        });
        ui.separator();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        let searching = self.searching();
        let lines = table_lines(matches, self.order, failing.as_deref(), &|group| {
            self.is_open_while(group, searching)
        });
        if lines.is_empty() {
            // Nothing fails only if no search narrows the failing checks.
            ui.weak(match (self.failing_only, searching) {
                (true, false) => NOTHING_FAILS,
                _ => NO_RESULT,
            });
        }
        let row_height = ui.text_style_height(&egui::TextStyle::Body) + 4.0;
        let spacing = ui.spacing().item_spacing.x;
        let widths = column_widths(ui.available_width(), spacing);
        let row_width = widths.iter().sum::<f32>() + 3.0 * spacing;
        let mut clicked = None;
        egui::ScrollArea::both()
            .id_salt("magcoupling_results_scroll")
            .auto_shrink([false, false])
            .show_rows(ui, row_height, lines.len(), |ui, range| {
                for line in &lines[range] {
                    match *line {
                        Line::OtherResults => {
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(row_width, row_height),
                                egui::Sense::hover(),
                            );
                            ui.painter().text(
                                rect.left_center(),
                                egui::Align2::LEFT_CENTER,
                                OTHER_RESULTS,
                                egui::TextStyle::Body.resolve(ui.style()),
                                ui.visuals().strong_text_color(),
                            );
                        }
                        Line::Group { group, shown, open } => {
                            let heading = &result_groups()[group];
                            let text = format!("{} ({shown})", heading.label);
                            // The worst level of the group's checks: a closed group shows that
                            // one of them fails.
                            let level = heading
                                .rows
                                .iter()
                                .map(|&index| &entries[index])
                                .filter(|entry| entry.check)
                                .filter_map(|entry| check_level(results, &entry.path))
                                .max();
                            let line = HeadingLine {
                                text: &text,
                                open,
                                level,
                                indent: if heading.other { OTHER_INDENT } else { 0.0 },
                                searching,
                            };
                            if group_heading_ui(ui, &line, egui::vec2(row_width, row_height)) {
                                clicked = Some(group);
                            }
                        }
                        Line::Row(index) => {
                            let entry = &entries[index];
                            let level = entry
                                .check
                                .then(|| check_level(results, &entry.path))
                                .flatten();
                            row_ui(ui, entry, results, level, row_height, widths, readouts);
                        }
                    }
                }
            });
        // A click while searching would change nothing on screen: it is ignored, so a group is
        // as the user left it once the search is cleared.
        if let Some(group) = clicked
            && !searching
            && !self.toggled.remove(&group)
        {
            self.toggled.insert(group);
        }
        action
    }
}

/// What a group's heading line shows.
struct HeadingLine<'a> {
    /// The group's label and its count.
    text: &'a str,
    open: bool,
    /// The worst level of the group's checks, a badge after the text; `None` for no check.
    level: Option<Level>,
    /// Points in from the left (a package group's, under "Other results").
    indent: f32,
    /// The search holds more than blanks: every group is open, and a click does nothing.
    searching: bool,
}

/// A group's heading, `size` points: the open or closed triangle of egui's collapsing header,
/// then the text and the badge of `line`. Returns whether it was clicked.
fn group_heading_ui(ui: &mut egui::Ui, line: &HeadingLine, size: egui::Vec2) -> bool {
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    // The triangle where egui's collapsing header puts it: its inner icon square, centred in
    // the indent.
    let indent_width = ui.spacing().indent;
    let (mut icon, _) = ui.spacing().icon_rectangles(rect);
    icon.set_center(egui::pos2(
        rect.left() + line.indent + indent_width / 2.0,
        rect.center().y,
    ));
    let openness = if line.open { 1.0 } else { 0.0 };
    egui::collapsing_header::paint_default_icon(
        ui,
        openness,
        &response.clone().with_new_rect(icon),
    );
    let text = ui.painter().text(
        egui::pos2(rect.left() + line.indent + indent_width, rect.center().y),
        egui::Align2::LEFT_CENTER,
        line.text,
        egui::TextStyle::Body.resolve(ui.style()),
        ui.visuals().strong_text_color(),
    );
    if line.level.is_some() {
        let at = egui::pos2(
            text.right() + ui.spacing().item_spacing.x,
            rect.center().y - 6.0,
        );
        // In a child Ui: the badge takes no room from the table's lines, all of one height.
        let mut badge_ui = ui.new_child(
            egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(at, egui::vec2(12.0, 12.0))),
        );
        badge(&mut badge_ui, line.level);
    }
    response
        .on_hover_text(if line.searching {
            CLEAR_TO_CLOSE
        } else if line.open {
            "Close the group"
        } else {
            "Open the group"
        })
        .clicked()
}
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// One table row: label, value with unit, workbook cell, marker, in columns `widths` wide
/// ([`column_widths`]; the path is in the hover text, which is built only while the row is
/// hovered). The whole row is a readout: hover it for its equation, click it to open it.
fn row_ui(
    ui: &mut egui::Ui,
    entry: &TableEntry,
    results: &DesignResults,
    height: f32,
    widths: [f32; 4],
    readouts: &mut Readouts,
) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
        // Left-aligned columns of fixed width (add_sized would centre the text).
        let cell = |ui: &mut egui::Ui, width: f32, text: &str| {
            let layout = egui::Layout::left_to_right(egui::Align::Center);
            ui.allocate_ui_with_layout(egui::vec2(width, height), layout, |ui| {
                ui.set_min_width(width);
                ui.add(egui::Label::new(text).truncate());
            });
        };
        let [label, number, workbook, marker] = widths;
        cell(ui, label, entry.info.meta.label);
        cell(
            ui,
            number,
            &with_unit(format_value(&value), entry.info.meta.unit),
        );
        cell(
            ui,
            workbook,
            entry.info.cell.as_deref().unwrap_or("Rust-only"),
        );
        cell(ui, marker, &entry.marker);
    });
````

with:

````rust
/// One table row: label, value with unit (after the badge of a check's `level`), workbook
/// cell, marker, in columns `widths` wide ([`column_widths`]; the path is in the hover text,
/// which is built only while the row is hovered). The whole row is a readout: hover it for its
/// equation, click it to open it.
fn row_ui(
    ui: &mut egui::Ui,
    entry: &TableEntry,
    results: &DesignResults,
    level: Option<Level>,
    height: f32,
    widths: [f32; 4],
    readouts: &mut Readouts,
) {
    let value = results.get(&entry.path).unwrap_or(Value::None);
    let row = ui.horizontal(|ui| {
        // Left-aligned columns of fixed width (add_sized would centre the text).
        let cell = |ui: &mut egui::Ui, width: f32, text: &str, level: Option<Level>| {
            let layout = egui::Layout::left_to_right(egui::Align::Center);
            ui.allocate_ui_with_layout(egui::vec2(width, height), layout, |ui| {
                ui.set_min_width(width);
                if level.is_some() {
                    badge(ui, level);
                }
                ui.add(egui::Label::new(text).truncate());
            });
        };
        let [label, number, workbook, marker] = widths;
        cell(ui, label, entry.info.meta.label, None);
        cell(
            ui,
            number,
            &with_unit(format_value(&value), entry.info.meta.unit),
            level,
        );
        cell(
            ui,
            workbook,
            entry.info.cell.as_deref().unwrap_or("Rust-only"),
            None,
        );
        cell(ui, marker, &entry.marker, None);
    });
````

- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib -- gui::results_table results_table_lists group_heading heading_click failing_filter narrow_window has_glyphs geometry_view_is_the_default 2>&1 | grep -E "^test |test result"
```

Expected: 19 tests `ok`: the 11 of `gui::results_table::tests` (the new `the_lines_group_the_rows_with_only_the_headline_open_at_first`, `a_search_shows_only_the_groups_it_matches`, `the_engine_order_and_the_failing_filter_are_flat` and the 8 there before), `gui::pickers::tests::every_picker_text_has_glyphs_in_the_default_fonts`, and in `gui::panel::tests`: `the_results_table_lists_results_and_filters_by_the_search`, `a_group_heading_opens_its_rows_and_the_engine_order_lists_them_without_headings`, `a_heading_click_while_searching_leaves_the_group_as_it_was`, `the_failing_filter_shows_exactly_the_failing_checks_with_their_badges`, `a_narrow_window_shows_each_row_s_label_and_value_without_scrolling`, `the_geometry_view_is_the_default_and_follows_the_design_shown`, `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`; then `test result: ok. 19 passed; 0 failed`.

- [ ] **Step 5: Run the checks and the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --check
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED"
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task4.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task4.log
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
```

Expected: `cargo fmt --check` prints nothing; `test result: ok. 469 passed; 0 failed`; `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order add magcoupling-rs/src/gui/results_table.rs magcoupling-rs/src/gui/panel.rs
git -C C:/Users/Cole/source/repos/lsim-mag-order commit -q -F - <<'EOF'
feat(magcoupling-rs): the results table by physics chain, check badges, the failing filter (O-6, O-7)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-order status --short
```

Expected: no status lines.

---

### Task 5: Tracing an input to its results and a result to its inputs

**Model:** `sonnet` (exact code; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/src/gui/typeset.rs` (`TermColors::with_marked` before `get`; a test)
- Create: `magcoupling-rs/src/gui/trace.rs` (tests first, then the module above them)
- Modify: `magcoupling-rs/src/gui/mod.rs` (`pub mod trace;`)
- Modify: `magcoupling-rs/src/gui/panel.rs` (import; field `trace`; the marks in `ui`; the trace set from a clicked readout, unless an input's trace marks it or it has no equation record; the banner in `inputs_ui`; the traced counts on group headings (`Trace::count_in`); the label click in `input_row_ui`; the trace passed to the table; five tests and the glyph test)
- Modify: `magcoupling-rs/src/gui/input_ui.rs` (`TRACE_HINT`; `RowOutput.label_clicked`; the label senses clicks)
- Modify: `magcoupling-rs/src/gui/results_table.rs` (imports; `TRACED_ONLY`, `NOTHING_TRACED`; field `traced_only`; `ui` takes `trace: Option<&Trace>` and unticks the filter without an input's trace; the checkbox, enabled for an input's trace; the traced rows worked out once a frame; the empty state; the traced counts; `shown`, `traced`)

**Interfaces:**
- Consumes: `registry()` (readouts): `is_leaf_input(&str) -> bool`, `downstream(&str) -> BTreeSet<String>`, `upstream_inputs(&str) -> Option<&BTreeSet<String>>`, `equation_for`; `term_label(&str) -> &str` (explorer); `Readouts::mark` (frames any path its `TermColors` colours); (Task 4) `ResultsTable::ui`, `Line`, the heading text.
- Produces: `TermColors::with_marked<'a>(self, paths: impl IntoIterator<Item = &'a str>, color: Color32) -> Self` (adds only uncoloured paths); `pub struct Trace { source: String, kind: TraceKind, paths: BTreeSet<String> }` with `fn of(path: &str) -> Option<Trace>`, `fn marks(&self, path: &str) -> bool` (the source too), `fn count_in<'a>(&self, paths: impl IntoIterator<Item = &'a str>) -> usize`, `fn banner(&self) -> String` (an input that reaches no explained result says so); `pub enum TraceKind { Input, Result }`; `pub const TRACING: &str = "Tracing"`, `pub const CLEAR_TRACE: &str = "Clear trace"`; `RowOutput.label_clicked: bool`; `pub const TRACE_HINT: &str`; `ResultsTable::ui(&mut self, ui, results, trace: Option<&Trace>, readouts)`; `pub const TRACED_ONLY: &str = "Traced only"` (for an input's trace only), `pub const NOTHING_TRACED: &str`; headings `"{label} ({shown}, {traced} traced)"` while a trace frames rows in a group; the panel field `trace: Option<Trace>` (tests read and set it).

- [ ] **Step 1: Write the failing tests**

In `magcoupling-rs/src/gui/typeset.rs`, replace:

````rust
        assert_eq!(f_end_color, Some(TERM_PALETTE[1]));
    }
````

with:

````rust
        assert_eq!(f_end_color, Some(TERM_PALETTE[1]));
    }

    #[test]
    fn marked_paths_take_the_colour_given_under_the_equation_s_own() {
        let eq = registry().equation_for("model.pullout_Nm").unwrap();
        let trace = Color32::from_rgb(1, 2, 3);
        let colors = TermColors::of(eq).with_marked(["model.f_end", "metal.face_gap_mm"], trace);
        // A term of the equation keeps its colour; a path it does not show takes the trace's.
        assert_eq!(colors.get("model.f_end"), Some(TERM_PALETTE[1]));
        assert_eq!(colors.get("metal.face_gap_mm"), Some(trace));
        let marked = TermColors::none().with_marked(["coupling.npole"], trace);
        assert_eq!(marked.get("coupling.npole"), Some(trace));
        assert!(!marked.is_empty());
        assert!(
            TermColors::none()
                .with_marked(std::iter::empty(), trace)
                .is_empty()
        );
    }
````

Create `magcoupling-rs/src/gui/trace.rs`:

````rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::results_table::table_entries;

    #[test]
    fn an_input_traces_the_results_its_value_flows_into() {
        let trace = Trace::of("metal.face_gap_mm").expect("an input");
        assert_eq!(trace.kind, TraceKind::Input);
        assert_eq!(trace.source, "metal.face_gap_mm");
        assert_eq!(trace.paths, registry().downstream("metal.face_gap_mm"));
        // The pull-out reads the gap; the clamp's preload does not.
        assert!(trace.marks("model.pullout_Nm"));
        assert!(!trace.marks("clamps.joint_preload_N"));
        // The source is marked too: its own row is framed.
        assert!(trace.marks("metal.face_gap_mm"));
        assert_eq!(
            trace.banner(),
            format!(
                "{TRACING} Candidate flat-face magnetic gap: drives {} explained results",
                trace.paths.len()
            )
        );
    }

    #[test]
    fn a_result_traces_the_inputs_it_reads() {
        let trace = Trace::of("model.pullout_Nm").expect("an explained result");
        assert_eq!(trace.kind, TraceKind::Result);
        let inputs = registry().upstream_inputs("model.pullout_Nm").unwrap();
        assert_eq!(&trace.paths, inputs);
        assert!(trace.marks("metal.face_gap_mm"));
        assert!(trace.marks("model.pullout_Nm"));
        assert!(!trace.marks("clamps.friction"));
        // Every traced path is an input.
        assert!(trace.paths.iter().all(|p| registry().is_leaf_input(p)));
        assert_eq!(
            trace.banner(),
            format!(
                "{TRACING} Pull-out torque at operating temperature: reads {} inputs",
                inputs.len()
            )
        );
        // Both directions agree: the gap drives the pull-out, which reads the gap.
        assert!(
            Trace::of("metal.face_gap_mm")
                .unwrap()
                .marks("model.pullout_Nm")
        );
    }

    #[test]
    fn an_input_no_explained_result_reads_traces_none_and_says_so() {
        // The drive torque sets the required floor and the verdict, which have no equation
        // record: its trace reaches no result, and its banner says why.
        let trace = Trace::of("coupling.drive_torque_Nm").expect("an input");
        assert_eq!(trace.kind, TraceKind::Input);
        assert!(trace.paths.is_empty());
        assert_eq!(
            trace.banner(),
            format!(
                "{TRACING} Torque the coupling must carry for driving (at the wheel): no \
                 explained result reads it (its results have no equation record)"
            )
        );
        // Only the explained results are traced: those with an equation record.
        let explained = table_entries()
            .iter()
            .filter(|e| registry().equation_for(&e.path).is_some())
            .count();
        assert_eq!((explained, table_entries().len()), (392, 1086));
        // Every input is a leaf of the registry: a click on any input's label traces it.
        for entry in crate::gui::inputs::InputCatalogue::get().all() {
            let trace = Trace::of(&entry.path).unwrap_or_else(|| panic!("{}", entry.path));
            assert_eq!(trace.kind, TraceKind::Input, "{}", entry.path);
        }
    }

    #[test]
    fn a_trace_counts_the_paths_it_marks() {
        let trace = Trace::of("metal.face_gap_mm").unwrap();
        // Its source, a result it drives; not a result it does not, nor an unknown path.
        assert_eq!(
            trace.count_in([
                "metal.face_gap_mm",
                "model.pullout_Nm",
                "clamps.joint_preload_N",
                "no.such.path"
            ]),
            2
        );
        assert_eq!(trace.count_in(std::iter::empty()), 0);
    }

    #[test]
    fn a_result_without_an_equation_record_and_an_unknown_path_trace_nothing() {
        let cell_only = table_entries()
            .iter()
            .find(|e| registry().equation_for(&e.path).is_none())
            .expect("a result without a record");
        assert_eq!(Trace::of(&cell_only.path), None, "{}", cell_only.path);
        assert_eq!(Trace::of("no.such.path"), None);
    }
}
````

In `magcoupling-rs/src/gui/mod.rs`, replace:

````rust
pub(crate) mod test_support;
pub mod typeset;
````

with:

````rust
pub(crate) mod test_support;
pub mod trace;
pub mod typeset;
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    #[test]
    fn a_narrow_window_shows_each_row_s_label_and_value_without_scrolling() {
````

with:

````rust
    /// The mark rects of the trace (the selection colour) painted in a frame.
    fn trace_marks(harness: &Harness, output: &egui::FullOutput) -> Vec<egui::Rect> {
        mark_rects(
            output,
            Some(harness.ctx.style().visuals.selection.stroke.color),
        )
    }

    #[test]
    fn clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click() {
        use crate::gui::dashboard::DASHBOARD;
        use crate::gui::trace::TraceKind;
        let mut harness = Harness::new();
        let gap = InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label;
        harness.click_text(gap);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Input, FACE_GAP)
        );
        assert_eq!(trace.paths, registry().downstream(FACE_GAP));
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner()), 1);
        let marks = trace_marks(&harness, &output);
        // On the dashboard (right of its first label's badge): one frame per row the gap
        // drives, none on the others.
        let pullout_label = result_info("model.pullout_Nm").unwrap().meta.label;
        let dashboard_left = text_rect(&output, pullout_label).unwrap().left() - 30.0;
        let on_dashboard = marks
            .iter()
            .filter(|m| m.center().x > dashboard_left)
            .count();
        let driven = DASHBOARD
            .iter()
            .filter(|(path, _)| trace.marks(path))
            .count();
        assert!(driven > 0);
        assert_eq!(on_dashboard, driven);
        // On the inputs side: the gap's own Key design row alone.
        let row = text_rects(&output, gap)
            .into_iter()
            .find(|r| r.left() < INPUTS_WIDTH)
            .unwrap();
        let on_inputs: Vec<&egui::Rect> = marks
            .iter()
            .filter(|m| m.center().x < INPUTS_WIDTH)
            .collect();
        assert_eq!(on_inputs.len(), 1);
        assert!(on_inputs[0].contains_rect(row));
        // A second click on the label ends the trace and its frames.
        harness.click_text(gap);
        assert_eq!(harness.panel.trace, None);
        let output = harness.frame(Vec::new());
        assert!(trace_marks(&harness, &output).is_empty());
        assert_eq!(harness.panel.design(), Design::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group() {
        use crate::gui::trace::{CLEAR_TRACE, TraceKind};
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(&displayed_pullout(&DesignInputs::default()));
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(
            (trace.kind, trace.source.as_str()),
            (TraceKind::Result, "model.pullout_Nm")
        );
        assert_eq!(
            &trace.paths,
            registry().upstream_inputs("model.pullout_Nm").unwrap()
        );
        assert!(harness.panel.explorer.open, "the click still opens it");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner()), 1);
        // Each Key design row is framed exactly when the pull-out reads its input.
        let marks = trace_marks(&harness, &output);
        for entry in &InputCatalogue::get().key_design {
            let row = text_rects(&output, entry.meta.label)
                .into_iter()
                .find(|r| r.left() < INPUTS_WIDTH)
                .unwrap();
            let framed = marks.iter().any(|m| m.contains_rect(row));
            assert_eq!(framed, trace.marks(&entry.path), "{}", entry.path);
        }
        // Each closed group says how many of its rows the trace frames.
        for group in &InputCatalogue::get().workflow {
            let traced = group
                .sections
                .iter()
                .flat_map(|s| s.entries.iter())
                .filter(|e| trace.marks(&e.path))
                .count();
            let title = if traced > 0 {
                format!("{} ({traced} traced)", group.label)
            } else {
                group.label.to_owned()
            };
            assert_eq!(count(&output, &title), 1, "{title}");
        }
        // Clear ends it.
        harness.click_text(CLEAR_TRACE);
        assert_eq!(harness.panel.trace, None);
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_traced_filter_shows_the_results_the_trace_marks() {
        use crate::gui::result_groups::result_groups;
        use crate::gui::results_table::{TRACED_ONLY, table_entries};
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::new();
        harness.click_text(CentreView::Results.label());
        // Without a trace the filter cannot be ticked.
        harness.click_text(TRACED_ONLY);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        // Trace the face gap and show only the results it drives.
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
        harness.click_text(TRACED_ONLY);
        let output = harness.frame(Vec::new());
        let trace = harness.panel.trace.clone().unwrap();
        let traced = entries.iter().filter(|e| trace.marks(&e.path)).count();
        assert!(traced > 0 && traced < total);
        assert_eq!(count(&output, &format!("{traced} of {total} results")), 1);
        // Each heading counts the rows the trace marks in it.
        let headline = result_groups()[0]
            .rows
            .iter()
            .filter(|&&i| trace.marks(&entries[i].path))
            .count();
        assert_eq!(
            count(
                &output,
                &format!("Headline ({headline}, {headline} traced)")
            ),
            1
        );
        // Without the trace the filter lets every row through again.
        harness.panel.trace = None;
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
    }

    #[test]
    fn an_input_whose_results_have_no_equation_record_traces_none_and_says_so() {
        use crate::gui::results_table::{NO_RESULT, NOTHING_TRACED, TRACED_ONLY, table_entries};
        // The drive torque sets the required floor and the verdict, which have no equation
        // record: the trace reaches no result, the banner says why, and "Traced only" says the
        // trace marks nothing rather than that the search matches nothing.
        let total = table_entries().len();
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(InputCatalogue::get().workflow[0].label);
        for _ in 0..10 {
            harness.frame(Vec::new());
        }
        let drive = "coupling.drive_torque_Nm";
        harness.click_text(InputCatalogue::get().entry(drive).unwrap().meta.label);
        let trace = harness.panel.trace.clone().expect("a trace");
        assert_eq!(trace.source, drive);
        assert!(trace.paths.is_empty());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &trace.banner()), 1);
        assert!(trace.banner().contains("no explained result reads it"));
        harness.click_text(CentreView::Results.label());
        harness.click_text(TRACED_ONLY);
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &format!("0 of {total} results")), 1);
        assert_eq!(count(&output, NOTHING_TRACED), 1);
        assert_eq!(count(&output, NO_RESULT), 0);
        // Clear trace turns the filter off with it: every row again, and a later trace does not
        // filter the table until the filter is ticked again.
        harness.click_text(crate::gui::trace::CLEAR_TRACE);
        let output = harness.frame(Vec::new());
        assert_eq!(harness.panel.trace, None);
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
        let output = harness.frame(Vec::new());
        assert!(harness.panel.trace.is_some());
        assert_eq!(count(&output, &format!("{total} of {total} results")), 1);
    }

    #[test]
    fn reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows() {
        use crate::gui::results_table::{ResultOrder, TRACED_ONLY, table_entries};
        use crate::gui::trace::TraceKind;
        // An input traced and "Traced only" ticked: a click on a traced row (the table's main
        // gesture, to read its equation) opens its equation and keeps the input's trace, so the
        // table keeps its rows; a click on a result without an equation record keeps it too.
        let entries = table_entries();
        let total = entries.len();
        let mut harness = Harness::on_screen(egui::vec2(1280.0, 3000.0));
        harness.click_text(CentreView::Results.label());
        harness.click_text(InputCatalogue::get().entry(FACE_GAP).unwrap().meta.label);
        harness.click_text(TRACED_ONLY);
        let trace = harness.panel.trace.clone().unwrap();
        let traced = entries.iter().filter(|e| trace.marks(&e.path)).count();
        let shown = format!("{traced} of {total} results");
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, &shown), 1);
        // The pull-out's row in the table (between the inputs and the dashboard).
        let centre =
            |r: &egui::Rect| r.left() > INPUTS_WIDTH && r.right() < 1280.0 - DASHBOARD_WIDTH;
        let pullout = result_info("model.pullout_Nm").unwrap().meta.label;
        let row = text_rects(&output, pullout)
            .into_iter()
            .find(centre)
            .expect("the pull-out's row");
        assert!(trace.marks("model.pullout_Nm"));
        harness.click(row.center());
        let output = harness.frame(Vec::new());
        assert!(harness.panel.explorer.open, "the click opens its equation");
        assert_eq!(
            harness.panel.trace.as_ref(),
            Some(&trace),
            "the input's trace stays"
        );
        assert_eq!(count(&output, &shown), 1, "the table keeps its rows");
        // Every row again, in the engine's order: a result without an equation record keeps
        // the trace (its inputs are unknown).
        harness.click_text(TRACED_ONLY);
        harness.click_text(ResultOrder::Engine.label());
        let output = harness.frame(Vec::new());
        let recordless = entries
            .iter()
            .find(|e| registry().equation_for(&e.path).is_none())
            .unwrap();
        let row = text_rects(&output, recordless.info.meta.label)
            .into_iter()
            .find(centre)
            .unwrap_or_else(|| panic!("{} on screen", recordless.path));
        harness.click(row.center());
        assert_eq!(harness.panel.trace.as_ref(), Some(&trace));
        // A result outside the trace, with a record (on the dashboard): its trace replaces the
        // input's.
        let outside = crate::gui::dashboard::dashboard_lines(harness.panel.results())
            .into_iter()
            .find(|line| registry().equation_for(line.path).is_some() && !trace.marks(line.path))
            .expect("a dashboard result the face gap does not drive");
        let output = harness.frame(Vec::new());
        // The rightmost: the dashboard's.
        let value = text_rects(&output, &outside.value)
            .into_iter()
            .max_by(|a, b| a.left().total_cmp(&b.left()))
            .unwrap();
        harness.click(value.center());
        let replaced = harness.panel.trace.clone().unwrap();
        assert_eq!(
            (replaced.kind, replaced.source.as_str()),
            (TraceKind::Result, outside.path)
        );
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn a_narrow_window_shows_each_row_s_label_and_value_without_scrolling() {
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        texts.push(crate::gui::results_table::NOTHING_FAILS.to_owned());
        texts.push(crate::gui::results_table::NO_RESULT.to_owned());
        texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());
````

with:

````rust
        texts.push(crate::gui::results_table::NOTHING_FAILS.to_owned());
        texts.push(crate::gui::results_table::NO_RESULT.to_owned());
        texts.push(crate::gui::results_table::TRACED_ONLY.to_owned());
        texts.push(crate::gui::input_ui::TRACE_HINT.to_owned());
        texts.push(crate::gui::trace::CLEAR_TRACE.to_owned());
        texts.push(crate::gui::results_table::NOTHING_TRACED.to_owned());
        for path in [FACE_GAP, "model.pullout_Nm", "coupling.drive_torque_Nm"] {
            texts.push(crate::gui::trace::Trace::of(path).unwrap().banner());
        }
        texts.push(crate::gui::results_table::CLEAR_TO_CLOSE.to_owned());
````

- [ ] **Step 2: Run the tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib 2>&1 | grep -E "^error" | sort | uniq
```

Expected: FAIL to compile, with these lines:

```
error: could not compile `magcoupling-rs` (lib test) due to 48 previous errors; 1 warning emitted
error[E0425]: cannot find function `registry` in this scope
error[E0425]: cannot find value `CLEAR_TRACE` in module `crate::gui::trace`
error[E0425]: cannot find value `NOTHING_TRACED` in module `crate::gui::results_table`
error[E0425]: cannot find value `TRACED_ONLY` in module `crate::gui::results_table`
error[E0425]: cannot find value `TRACE_HINT` in module `crate::gui::input_ui`
error[E0425]: cannot find value `TRACING` in this scope
error[E0432]: unresolved import `crate::gui::results_table::TRACED_ONLY`
error[E0432]: unresolved import `crate::gui::trace::TraceKind`
error[E0432]: unresolved imports `crate::gui::results_table::NOTHING_TRACED`, `crate::gui::results_table::TRACED_ONLY`
error[E0432]: unresolved imports `crate::gui::trace::CLEAR_TRACE`, `crate::gui::trace::TraceKind`
error[E0433]: failed to resolve: could not find `Trace` in `trace`
error[E0433]: failed to resolve: use of undeclared type `TraceKind`
error[E0433]: failed to resolve: use of undeclared type `Trace`
error[E0599]: no method named `with_marked` found for struct `typeset::TermColors` in the current scope
error[E0609]: no field `trace` on type `gui::panel::MagcouplingPanel`
```

- [ ] **Step 3: Add the trace and draw it**

In `magcoupling-rs/src/gui/typeset.rs`, replace:

````rust
    /// The colour of a path or template, if the equation shows it.
    pub fn get(&self, path: &str) -> Option<Color32> {
````

with:

````rust
    /// Adds each of `paths` that has no colour yet, in `color`: the marks of a trace (decision
    /// O-8), under the equation's own term colours.
    pub fn with_marked<'a>(
        mut self,
        paths: impl IntoIterator<Item = &'a str>,
        color: Color32,
    ) -> Self {
        for path in paths {
            self.colors.entry(path.to_owned()).or_insert(color);
        }
        self
    }

    /// The colour of a path or template, if the equation shows it.
    pub fn get(&self, path: &str) -> Option<Color32> {
````

In `magcoupling-rs/src/gui/trace.rs`, replace:

````rust
#[cfg(test)]
mod tests {
    use super::*;
````

with:

````rust
//! Tracing between the inputs and the results (decision O-8): a click on an input's label
//! traces the results its value flows into (the registry's `downstream`), a click on a result
//! (any readout: a dashboard row, a table row, a callout) traces the inputs it reads (its
//! `upstream_inputs`). The panel frames every traced path where it is drawn, as the Equation
//! panel frames an equation's terms, in the selection colour (the colour of the frame a leaf
//! term draws around its input row), under the equation's own term colours. The registry knows
//! the dependencies of the results it explains (the A-3 scope: 392 of the 1086 results), so a
//! result without an equation record traces nothing, and an input whose results have none (the
//! drive torque, the adhesive's inputs) reaches no result: its banner says so.

use std::collections::BTreeSet;

use crate::gui::explorer::term_label;
use crate::gui::readouts::registry;

/// The start of the trace's banner over the inputs side.
pub const TRACING: &str = "Tracing";

/// The banner's button that ends the trace.
pub const CLEAR_TRACE: &str = "Clear trace";

/// What is traced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceKind {
    /// An input: the trace marks the explained results downstream of it.
    Input,
    /// A result with an equation record: the trace marks the inputs upstream of it.
    Result,
}

/// A trace: its source and the paths it reaches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Trace {
    pub source: String,
    pub kind: TraceKind,
    /// An input's explained results downstream, or a result's inputs upstream (transitively).
    pub paths: BTreeSet<String>,
}

impl Trace {
    /// The trace of `path`: an input's downstream results, an explained result's upstream
    /// inputs; `None` for a result without an equation record or a path that is neither.
    pub fn of(path: &str) -> Option<Trace> {
        let registry = registry();
        if registry.is_leaf_input(path) {
            return Some(Trace {
                source: path.to_owned(),
                kind: TraceKind::Input,
                paths: registry.downstream(path),
            });
        }
        registry.upstream_inputs(path).map(|inputs| Trace {
            source: path.to_owned(),
            kind: TraceKind::Result,
            paths: inputs.clone(),
        })
    }

    /// Whether the trace marks `path`: its source, or a path it reaches.
    pub fn marks(&self, path: &str) -> bool {
        self.source == path || self.paths.contains(path)
    }

    /// How many of `paths` the trace marks: what a group heading counts.
    pub fn count_in<'a>(&self, paths: impl IntoIterator<Item = &'a str>) -> usize {
        paths.into_iter().filter(|path| self.marks(path)).count()
    }

    /// The banner over the inputs side: what is traced and how far it reaches.
    pub fn banner(&self) -> String {
        let label = term_label(&self.source);
        let count = self.paths.len();
        match self.kind {
            TraceKind::Input if count == 0 => format!(
                "{TRACING} {label}: no explained result reads it (its results have no equation record)"
            ),
            TraceKind::Input => format!("{TRACING} {label}: drives {count} explained results"),
            TraceKind::Result => format!("{TRACING} {label}: reads {count} inputs"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
````

In `magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
/// The text of a blank optional input.
pub const BLANK_TEXT: &str = "blank";
````

with:

````rust
/// The text of a blank optional input.
pub const BLANK_TEXT: &str = "blank";

/// The line the label's hover text ends with: a click on the label traces the input
/// (decision O-8).
pub const TRACE_HINT: &str = "Click the label to trace the explained results it drives";
````

In `magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
    /// The row's main widget (the slider, drop-down, checkbox or text field).
    pub widget: egui::Response,
}
````

with:

````rust
    /// The row's main widget (the slider, drop-down, checkbox or text field).
    pub widget: egui::Response,
    /// The label was clicked: trace the input (decision O-8).
    pub label_clicked: bool,
}
````

In `magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
    let changed = *current != entry.default;
    let mut edit = None;
    ui.push_id(&entry.path, |ui| {
        ui.horizontal(|ui| {
            let dot = if changed { CHANGED_DOT } else { " " };
            ui.colored_label(ui.visuals().selection.stroke.color, dot)
                .on_hover_text("Changed from the default");
            ui.label(meta.label).on_hover_text(&tooltip);
````

with:

````rust
    let changed = *current != entry.default;
    let mut edit = None;
    let mut label_clicked = false;
    ui.push_id(&entry.path, |ui| {
        ui.horizontal(|ui| {
            let dot = if changed { CHANGED_DOT } else { " " };
            ui.colored_label(ui.visuals().selection.stroke.color, dot)
                .on_hover_text("Changed from the default");
            label_clicked = ui
                .add(egui::Label::new(meta.label).sense(egui::Sense::click()))
                .on_hover_text(format!("{tooltip}\n{TRACE_HINT}"))
                .clicked();
````

In `magcoupling-rs/src/gui/input_ui.rs`, replace:

````rust
        RowOutput { edit, widget }
    })
    .inner
````

with:

````rust
        RowOutput {
            edit,
            widget,
            label_clicked,
        }
    })
    .inner
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
use crate::gui::session::{Design, design_json, json_value};
````

with:

````rust
use crate::gui::session::{Design, design_json, json_value};
use crate::gui::trace::{Trace, TraceKind};
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
/// The failing filter's checkbox (decision O-7).
pub const FAILING_ONLY: &str = "Failing checks only";
````

with:

````rust
/// The failing filter's checkbox (decision O-7).
pub const FAILING_ONLY: &str = "Failing checks only";

/// The trace filter's checkbox (decision O-8): the rows an input's trace marks alone.
pub const TRACED_ONLY: &str = "Traced only";

/// What the table says when the trace filter leaves no row: the trace reaches only results with
/// an equation record, and an input may reach none of them.
pub const NOTHING_TRACED: &str =
    "The trace marks no result shown here (only results with an equation record are traced).";
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
    /// The failing checks alone (decision O-7).
    failing_only: bool,
````

with:

````rust
    /// The failing checks alone (decision O-7).
    failing_only: bool,
    /// The rows the trace marks alone, while an input is traced (decision O-8).
    traced_only: bool,
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
    /// Draws the table: the search box, the order toggle, the failing filter, the count and the
    /// export buttons, then the lines on screen (the end-effect banner is the centre region's,
    /// over every view: decision M42-1), each row a readout (`readouts`), each group heading a
    /// button that opens or closes it (not while the search holds more than blanks: every group
    /// is open then), with the worst level of its checks. Returns an export asked for.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        results: &DesignResults,
        readouts: &mut Readouts,
    ) -> Option<TableAction> {
````

with:

````rust
    /// Draws the table: the search box, the order toggle, the failing and trace filters, the
    /// count and the export buttons, then the lines on screen (the end-effect banner is the
    /// centre region's, over every view: decision M42-1), each row a readout (`readouts`, which
    /// frame the rows `trace` marks), each group heading a button that opens or closes it (not
    /// while the search holds more than blanks: every group is open then), with the worst level
    /// of its checks and the rows the trace marks in it. Returns an export asked for.
    pub fn ui(
        &mut self,
        ui: &mut egui::Ui,
        results: &DesignResults,
        trace: Option<&Trace>,
        readouts: &mut Readouts,
    ) -> Option<TableAction> {
        // The trace filter keeps the results an input's trace marks: it goes off with the trace,
        // and a result's trace (which marks inputs) cannot turn it on.
        let input_traced = trace.is_some_and(|t| t.kind == TraceKind::Input);
        if !input_traced {
            self.traced_only = false;
        }
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
        // The failing checks' rows, the red first, when the filter is on (after its checkbox).
        let mut failing: Option<Vec<usize>> = None;
        ui.horizontal_wrapped(|ui| {
            ui.add(
````

with:

````rust
        // The failing checks' rows, the red first, when the filter is on (after its checkbox).
        let mut failing: Option<Vec<usize>> = None;
        // The search's rows the trace marks, when the trace filter is on (after its checkbox).
        let mut traced: Option<Vec<usize>> = None;
        ui.horizontal_wrapped(|ui| {
            ui.add(
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
            ui.checkbox(&mut self.failing_only, FAILING_ONLY)
                .on_hover_text(
                    "The checks that fail (red) or ask for a look (amber), the red first",
                );
            if self.matches.is_none() || self.query != self.matched_query {
                self.matches = Some(search(entries, &self.query));
                self.matched_query = self.query.clone();
            }
            let matches = self.matches.as_deref().unwrap_or(&[]);
````

with:

````rust
            ui.checkbox(&mut self.failing_only, FAILING_ONLY)
                .on_hover_text(
                    "The checks that fail (red) or ask for a look (amber), the red first",
                );
            ui.add_enabled(
                input_traced,
                egui::Checkbox::new(&mut self.traced_only, TRACED_ONLY),
            )
            .on_hover_text("The results an input's trace marks: click the input's label");
            if self.matches.is_none() || self.query != self.matched_query {
                self.matches = Some(search(entries, &self.query));
                self.matched_query = self.query.clone();
            }
            traced = self.traced(trace);
            let matches = self.shown(&traced);
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
        ui.separator();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        let searching = self.searching();
        let lines = table_lines(matches, self.order, failing.as_deref(), &|group| {
            self.is_open_while(group, searching)
        });
        if lines.is_empty() {
            // Nothing fails only if no search narrows the failing checks.
            ui.weak(match (self.failing_only, searching) {
                (true, false) => NOTHING_FAILS,
````

with:

````rust
        ui.separator();
        let matches = self.shown(&traced);
        let searching = self.searching();
        let lines = table_lines(matches, self.order, failing.as_deref(), &|group| {
            self.is_open_while(group, searching)
        });
        if lines.is_empty() {
            // The trace filter first (an input's trace may reach no result); nothing fails only
            // if no search narrows the failing checks.
            ui.weak(match (self.traced_only, self.failing_only, searching) {
                (true, _, _) => NOTHING_TRACED,
                (false, true, false) => NOTHING_FAILS,
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
                        Line::Group { group, shown, open } => {
                            let heading = &result_groups()[group];
                            let text = format!("{} ({shown})", heading.label);
````

with:

````rust
                        Line::Group { group, shown, open } => {
                            let heading = &result_groups()[group];
                            // A trace counts the rows it frames in each group, as on the
                            // inputs side.
                            let traced = trace.map_or(0, |trace| {
                                trace.count_in(
                                    heading
                                        .rows
                                        .iter()
                                        .map(|&index| entries[index].path.as_str()),
                                )
                            });
                            let text = if traced > 0 {
                                format!("{} ({shown}, {traced} traced)", heading.label)
                            } else {
                                format!("{} ({shown})", heading.label)
                            };
````

In `magcoupling-rs/src/gui/results_table.rs`, replace:

````rust
        // A click while searching would change nothing on screen: it is ignored, so a group is
        // as the user left it once the search is cleared.
        if let Some(group) = clicked
            && !searching
            && !self.toggled.remove(&group)
        {
            self.toggled.insert(group);
        }
        action
    }
}
````

with:

````rust
        // A click while searching would change nothing on screen: it is ignored, so a group is
        // as the user left it once the search is cleared.
        if let Some(group) = clicked
            && !searching
            && !self.toggled.remove(&group)
        {
            self.toggled.insert(group);
        }
        action
    }

    /// The rows the search and the trace filter let through: `traced` while the filter is on,
    /// else every row the search matches (the failing filter cuts them further).
    fn shown<'a>(&'a self, traced: &'a Option<Vec<usize>>) -> &'a [usize] {
        match traced {
            Some(rows) => rows,
            None => self.matches.as_deref().unwrap_or(&[]),
        }
    }

    /// The search's rows the trace marks, while the trace filter is on (an input is traced);
    /// `None` while it is off: every row the search matches is shown.
    fn traced(&self, trace: Option<&Trace>) -> Option<Vec<usize>> {
        let trace = trace.filter(|_| self.traced_only)?;
        let entries = table_entries();
        let matches = self.matches.as_deref().unwrap_or(&[]);
        Some(
            matches
                .iter()
                .copied()
                .filter(|&index| trace.marks(&entries[index].path))
                .collect(),
        )
    }
}
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
use crate::gui::sizing::{
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::{DesignInputs, DesignResults, compute_all};
````

with:

````rust
use crate::gui::sizing::{
    SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, TARGET_RANGE_INPUT,
    variable_label,
};
use crate::gui::trace::{CLEAR_TRACE, Trace, TraceKind};
use crate::{DesignInputs, DesignResults, compute_all};
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    /// The Equation panel: the equation open, its trail, the readout hovered in the last frame
    /// (this frame marks its equation's terms) and the input row a leaf term highlights.
    explorer: Explorer,
}
````

with:

````rust
    /// The Equation panel: the equation open, its trail, the readout hovered in the last frame
    /// (this frame marks its equation's terms) and the input row a leaf term highlights.
    explorer: Explorer,
    /// The input or result traced (decision O-8): a click on an input's label or on a result
    /// sets it; every path it reaches is framed in the selection colour.
    trace: Option<Trace>,
}
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
            explorer: Explorer::default(),
        }
    }
````

with:

````rust
            explorer: Explorer::default(),
            trace: None,
        }
    }
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let mut readouts = Readouts::new(self.explorer.marks(&self.inputs, &self.results));
````

with:

````rust
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let mut marks = self.explorer.marks(&self.inputs, &self.results);
        if let Some(trace) = &self.trace {
            let reached = trace.paths.iter().map(String::as_str);
            marks = marks.with_marked(
                reached.chain([trace.source.as_str()]),
                ui.visuals().selection.stroke.color,
            );
        }
        let mut readouts = Readouts::new(marks);
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        self.explorer.end_frame(ui.ctx(), readouts.finish());
        // One undo step per settled edit.
````

with:

````rust
        let events = readouts.finish();
        // A result clicked opens in the Equation panel and traces its inputs, unless an input's
        // trace marks it: reading the equations of the results an input drives keeps that trace
        // (and the table's "Traced only" rows). A result without an equation record keeps the
        // trace too: nothing is known of its inputs.
        if let Some(path) = &events.clicked {
            let inside = self
                .trace
                .as_ref()
                .is_some_and(|t| t.kind == TraceKind::Input && t.paths.contains(path));
            if !inside && let Some(trace) = Trace::of(path) {
                self.trace = Some(trace);
            }
        }
        self.explorer.end_frame(ui.ctx(), events);
        // One undo step per settled edit.
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        ui.horizontal(|ui| {
            for view in InputsView::ALL {
                ui.selectable_value(&mut self.inputs_view, view, view.label());
            }
        });
        ui.separator();
        match self.inputs_view {
````

with:

````rust
        ui.horizontal(|ui| {
            for view in InputsView::ALL {
                ui.selectable_value(&mut self.inputs_view, view, view.label());
            }
        });
        if let Some(banner) = self.trace.as_ref().map(Trace::banner) {
            ui.horizontal_wrapped(|ui| {
                ui.colored_label(ui.visuals().selection.stroke.color, banner);
                if ui.small_button(CLEAR_TRACE).clicked() {
                    self.trace = None;
                }
            });
        }
        ui.separator();
        match self.inputs_view {
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                    egui::CollapsingHeader::new(group.label)
                        .id_salt((salt, &group.name))
````

with:

````rust
                    // A trace counts the rows it frames in each group, closed or open.
                    let traced = self.trace.as_ref().map_or(0, |trace| {
                        trace.count_in(
                            group
                                .sections
                                .iter()
                                .flat_map(|s| s.entries.iter())
                                .map(|e| e.path.as_str()),
                        )
                    });
                    let title = if traced > 0 {
                        format!("{} ({traced} traced)", group.label)
                    } else {
                        group.label.to_owned()
                    };
                    egui::CollapsingHeader::new(title)
                        .id_salt((salt, &group.name))
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
        let output = row.inner;
        if locked {
            ui.weak(SIZED_NOTE);
            return output.widget.id;
        }
````

with:

````rust
        let output = row.inner;
        if output.label_clicked {
            // A second click on the source's label ends the trace.
            let same = self.trace.as_ref().is_some_and(|t| t.source == entry.path);
            self.trace = if same { None } else { Trace::of(&entry.path) };
            ui.ctx().request_repaint();
        }
        if locked {
            ui.weak(SIZED_NOTE);
            return output.widget.id;
        }
````

In `magcoupling-rs/src/gui/panel.rs`, replace:

````rust
                let action = self.results_table.ui(ui, &self.results, readouts);
````

with:

````rust
                let action =
                    self.results_table
                        .ui(ui, &self.results, self.trace.as_ref(), readouts);
````

- [ ] **Step 4: Run the tests to see them pass**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features gui --lib -- gui::trace marked_paths clicking_an_input clicking_a_result traced_filter no_equation_record reading_a_traced has_glyphs clicking_a_value_opens 2>&1 | grep -E "^test |test result"
```

Expected: 14 tests `ok`: `gui::typeset::tests::marked_paths_take_the_colour_given_under_the_equation_s_own`, the 5 of `gui::trace::tests` (`an_input_traces_the_results_its_value_flows_into`, `a_result_traces_the_inputs_it_reads`, `an_input_no_explained_result_reads_traces_none_and_says_so`, `a_trace_counts_the_paths_it_marks`, `a_result_without_an_equation_record_and_an_unknown_path_trace_nothing`), `gui::pickers::tests::every_picker_text_has_glyphs_in_the_default_fonts`, and in `gui::panel::tests`: `clicking_an_input_s_label_frames_the_results_it_drives_until_a_second_click`, `clicking_a_result_frames_the_inputs_it_reads_and_counts_them_by_group`, `the_traced_filter_shows_the_results_the_trace_marks`, `an_input_whose_results_have_no_equation_record_traces_none_and_says_so`, `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows`, `clicking_a_value_opens_it_and_a_term_drills_in_and_the_breadcrumb_returns`, `every_text_the_panel_shows_has_glyphs_in_the_default_fonts`; then `test result: ok. 14 passed; 0 failed`.

- [ ] **Step 5: Run the checks and the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --check
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-order/magcoupling-rs/Cargo.toml --features app --lib 2>&1 | grep -E "test result|FAILED"
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task5.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task5.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task5.log
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
```

Expected: `cargo fmt --check` prints nothing; `test result: ok. 480 passed; 0 failed`; `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 6: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order add magcoupling-rs/src/gui/typeset.rs magcoupling-rs/src/gui/trace.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/input_ui.rs magcoupling-rs/src/gui/results_table.rs
git -C C:/Users/Cole/source/repos/lsim-mag-order commit -q -F - <<'EOF'
feat(magcoupling-rs): trace an input to the results it drives and a result to its inputs (O-8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-order status --short
```

Expected: no status lines.

---

### Task 6: Docs, the plan, and the browser check

**Model:** `sonnet` (doc edits given verbatim, commands and a browser check; CLAUDE.md section 5). Escalate a retry to the session model.

**Files:**
- Modify: `magcoupling-rs/README.md` (status paragraph; the module table's `src/gui/` row; "The panel": Inputs and Results table; the test table's panel row and inputs/dashboard/corrections/results_table row)
- Modify: `docs/ai/03-structure.yaml` (the `gui:` entry), `docs/ai/04-memory.yaml` (the NEXT item becomes the open confirmation), `docs/ai/05-update-tracker.md` (a new top entry)
- Modify: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (M4 layout amendment)
- Create: `docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md` (this plan, copied)

**Interfaces:**
- Consumes: every name the earlier tasks produced (the docs cite them).
- Produces: docs only.

- [ ] **Step 1: The magcoupling README**

In `magcoupling-rs/README.md`, replace:

````markdown
pickers and the material warnings (decisions M43-1 to M43-15). Next: M3 (live 3D fields).
````

with:

````markdown
pickers and the material warnings (decisions M43-1 to M43-15). The ordering of the inputs and the
results complete (plan `docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md`): the
inputs by design workflow (the workbook's package groups one toggle away) with a filter box, the
results by physics chain with check badges and a failing filter, and tracing between an input and
the results it drives (decisions O-1 to O-8). Next: M3 (live 3D fields).
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
`inputs.rs` (`InputCatalogue`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`)
````

with:

````markdown
`inputs.rs` (`InputCatalogue` with `groups_in` and `section_of`, `InputOrder`, `WORKFLOW`, `ADVANCED_HEADING`, `filter_inputs`, `FILTER_HINT`, `KEY_DESIGN`, `SECTION_LABELS`, `OPTIONAL_SEEDS`, `step_decimals`, `text_hint`, `input_tooltip`)
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
`input_ui.rs` (`input_row`, `slider`: one input row of any type)
````

with:

````markdown
`input_ui.rs` (`input_row`, `slider`: one input row of any type; a click on its label traces the input)
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
`dashboard.rs` (`DASHBOARD`, `verdict_level`,
````

with:

````markdown
`dashboard.rs` (`DASHBOARD`, `CHECKS`, `verdict_level`, `check_level`, `failing_checks`,
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
`results_table.rs` (`table_entries`, `search`, `results_csv`, `results_json`, `column_widths`)
````

with:

````markdown
`results_table.rs` (`table_entries`, `entry_index`, `search`, `ResultOrder`, `Line`, `table_lines`, `results_csv`, `results_json`, `column_widths`), `result_groups.rs` (`result_groups`: the headline, the eight chains, the other results by package; `CHAIN_LABELS`, `CHAIN_PREFIXES`, `PACKAGE_LABELS`), `trace.rs` (`Trace`: an input's downstream results, a result's upstream inputs; `count_in`)
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
`format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included)
````

with:

````markdown
`format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match)
````

In `magcoupling-rs/README.md`, replace:

````markdown
- **Inputs** (`inputs.rs`, `input_ui.rs`): every input, generated from its metadata and grouped as
  the package groups them, each nested group under its heading; the Key design group on top
````

with:

````markdown
- **Inputs** (`inputs.rs`, `input_ui.rs`): every input, generated from its metadata, in the
  workflow order by default (decision O-1, `WORKFLOW`, O-2: Requirements and operating conditions
  first, the spec a designer fixes before choosing magnets (the torque, the temperatures, the
  space claim, the drive and the duty); then Magnets and rings, Gap and clearances, Housing and
  retainers, Shaft, key and clamps, Materials (the adhesive and its bondlines in one section),
  Thermal and demagnetization, and Calibration and model (the fields stored from a 3D run beside
  the 3D reference torques, in view: they must be refreshed after a geometry change); each group's
  rarely changed rows (O-4: the bedding clearances, the optional adapter and its joint, the clamp
  and screw factors, the screw classes, the slip-loss end factor, the model constants) under a
  closed Advanced heading) or, one toggle away ("Workbook groups"), grouped as the package groups
  them, each nested group under its heading; the order is a view of the session, in no design file
  or share link. A filter box matches label, path or workbook cell as the results search does and
  shows the matching rows alone, under their group and section (O-5). The Key design group sits on
  top
````

In `magcoupling-rs/README.md`, replace:

````markdown
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help, path, workbook cell, slider range and default. A tab
  row above them switches to the **Assumptions** view (decision M43-7).
````

with:

````markdown
  grade or not). Each row shows a dot when changed from the default, a
  reset button, and a tooltip with help, path, workbook cell, slider range and default. A tab
  row above them switches to the **Assumptions** view (decision M43-7). A click on a row's label
  traces the input (decision O-8, `trace.rs`): every explained result its value flows into (the
  registry's `downstream`; the registry explains 392 of the 1086 results, so an input such as the
  drive torque reaches none, and its banner says so) is framed wherever it is drawn, in the
  selection colour, under a banner with a Clear trace button; a click on a result (it still opens
  the Equation panel) traces the inputs it reads (`upstream_inputs`), unless an input's trace
  marks it or it has no equation record: the trace then stays. Each group heading counts the rows
  the trace frames.
````

In `magcoupling-rs/README.md`, replace:

````markdown
- **Results table** (`results_table.rs`): every result with label, value, unit, workbook cell and
  marker; search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
````

with:

````markdown
- **Results table** (`results_table.rs`): every result with label, value, unit, workbook cell and
  marker; by physics chain by default (decision O-6, `result_groups.rs`: the headline, that is the
  dashboard's 17 rows, open; then the closed chain groups: the A-3 chains torque, temperature,
  demagnetization, slip heating, clamps and geometry, each with the results of its nested groups
  that the explorer's scope leaves out (`CHAIN_PREFIXES`: the cold demagnetization limits, the
  slip temperatures, the metal design's clearances and reserves), and the adhesive and mass
  chains; then "Other results" by package, each result in the first group that lists it; a heading
  opens its group and shows the worst level of its checks, so a closed group shows a failing
  check; a search opens every group it matches, and a heading click does nothing until the search
  is cleared) or in the engine's order; each design check's value carries the dashboard's badge
  (`CHECKS`, the dashboard's verdicts and the other 23 checks, screens and warnings), and "Failing
  checks only" lists the red then the amber ones alone (O-7); "Traced only" keeps the rows an
  input's trace frames; search by label, path or cell; CSV (`path,label,value,unit,cell`) and JSON (with the
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
a ~930 px window shows a row's label (filling its 120-point column) and value |
````

with:

````markdown
a ~930 px window shows a row's label (filling its 120-point column) and value; the ordering (O-1 to O-8): every group of both orders opens and draws its rows and headings, the Advanced sections closed until their heading is clicked; the order toggle changes no input, design, share link or undo history; the filter box shows the matches alone (count, run headings; a matched row nudges; no undo step for typing; the workbook order's headings; none; blank); a focused advanced input opens its group and heading past a filter; the results by group (every heading drawn, a heading opens and closes its rows, the engine order without headings); the failing filter shows exactly the failing checks with four red and two amber badges; a click on an input's label frames exactly the dashboard rows it drives and its own row until a second click; a click on a result frames exactly the Key design rows it reads and counts the traced rows per group; the traced filter (disabled without a trace, the traced count, the headline's traced heading); a heading shows the worst level of its group's checks (the headline and the closed coupling model red, the mass none); a heading click while searching leaves its group as it was, its hover text saying why; the failing filter with a search no failing check matches says no result matches; an input whose results have no equation record traces none, says so, and the traced filter then says the trace marks nothing (Clear trace turns the filter off); reading a traced result keeps the input's trace and its traced rows, a result without a record keeps it, a result outside it replaces it; idle frames in the workflow order's filtered view change nothing |
````

In `magcoupling-rs/README.md`, replace (part of one line):

````markdown
the label column flexes down to its minimum; the hover text carries the correction marks |
````

with:

````markdown
the label column flexes down to its minimum; the hover text carries the correction marks. The ordering: every input in exactly one workflow section (a test lists any input without one), the groups in design order with the advanced sections last and every Key design input in an ordinary one; the filter matches label, path and cell by section in either order; every one of the 30 checks is a text result in schema order and every result named as a check is listed; every verdict of every other check is classified (a design per branch; the does-not-apply texts give no badge); the failing checks are the red then the amber; every result in exactly one group, the headline first, then the eight chains, then the other results, a path in two places in the first, every package with a heading, every chain prefix placing a result the scope leaves out; the levels order by severity; the lines with only the headline open, a search shows only the groups it matches, the engine order and the failing filter are flat; an input traces the results downstream of it, a result the inputs upstream, a result without a record nothing, an input no explained result reads none (it says so; 392 of the 1086 results are explained, every input traceable); a trace counts the paths it marks; marked paths take the trace's colour under the equation's own |
````

- [ ] **Step 2: The project memory**

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
TermColors (of, with_members)
````

with:

````yaml
TermColors (of, with_members, with_marked)
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
inputs.rs (InputCatalogue::get, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS, step_decimals, outside_range, text_hint, input_tooltip)
````

with:

````yaml
inputs.rs (InputCatalogue::get with groups (the workbook order) and workflow, groups_in(InputOrder), section_of(InputOrder, path); InputOrder (Workflow, the default; Workbook), WORKFLOW (8 groups of WorkflowSection, the requirements first, advanced sections last), ADVANCED_HEADING, filter_inputs -> SectionMatches (InputEntry's haystack, built once), FILTER_HINT, KEY_DESIGN, SECTION_LABELS, OPTIONAL_SEEDS, step_decimals, outside_range, text_hint, input_tooltip)
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
input_ui.rs (input_row, slider, RowEdit)
````

with:

````yaml
input_ui.rs (input_row -> RowOutput {edit, widget, label_clicked}, slider, RowEdit, TRACE_HINT)
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
dashboard.rs (DASHBOARD, Level, verdict_level,
````

with:

````yaml
dashboard.rs (DASHBOARD, Level (ordered by severity), CHECKS (30 design checks), verdict_level, check_level, failing_checks, badge,
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
results_table.rs (table_entries, search, exact_number, results_csv, results_json, ResultsTable, row_tooltip, column_widths)
````

with:

````yaml
results_table.rs (table_entries (TableEntry.check), entry_index, search, ResultOrder (Grouped, the default; Engine), Line, table_lines, exact_number, results_csv, results_json, ResultsTable (is_open, opens_by_default, ui with the trace; group headings with the worst check level, ignored while searching), FAILING_ONLY, TRACED_ONLY, NOTHING_FAILS, NOTHING_TRACED, NO_RESULT, CLEAR_TO_CLOSE, row_tooltip, column_widths), result_groups.rs (result_groups -> ResultGroup {id, label, other, rows}: the headline, CHAIN_LABELS (the scope's six chains, adhesive, mass), CHAIN_PREFIXES, PACKAGE_LABELS; row_template, package_of, groups_of), trace.rs (Trace::of -> {source, kind: TraceKind, paths}, marks, count_in, banner; TRACING, CLEAR_TRACE)
````

In `docs/ai/03-structure.yaml`, replace (part of one line):

````yaml
format.rs (format_value, with_unit, non_finite_text)
````

with:

````yaml
format.rs (format_value, with_unit, non_finite_text, search_haystack, search_needle)
````

In `docs/ai/04-memory.yaml`, replace:

````yaml
  - "NEXT (user request 2026-10-02): sort the magcoupling GUI inputs and outputs more intelligently. Today inputs follow the workbook package groups (coupling, metal, calibration, materials, temperature, clamps; gui/inputs.rs SECTION_LABELS) with the Key design group on top, and the results table follows engine order with a text search. Candidate approaches to plan: (1) inputs grouped by design workflow / physical subsystem (magnets and rings, gap and clearances, housing and retainers, shaft and clamps, materials, operating conditions, thermal), with Advanced/workbook-only rows collapsed; (2) results grouped by the A-3 chains (torque, temperature, demagnetization, slip heating, clamps, geometry) with the headline first; (3) a failing-checks-first filter and verdict badges in the table; (4) select an input to see the results it drives (registry used_by / downstream) and vice versa; (5) a filter box on the inputs side like the results search. Keep the workbook grouping available as an alternate view (schema and share links unchanged)."
````

with:

````yaml
  - "Magcoupling GUI ordering (plan docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md, branch magcoupling/ordering): built with the recommended option of each decision O-1 to O-8 (workflow input groups with the requirements and operating conditions first and Advanced only for the rarely changed rows, the workflow order as default with the workbook groups one toggle away, results by physics chain (the scope's chains widened by CHAIN_PREFIXES, plus adhesive and mass) with only the headline open and each heading showing its worst check, levels for the 23 checks off the dashboard, the failing filter flat with the red first, click to trace in the selection colour, kept while reading the results an input drives). Tracing reaches only the 392 explained results of 1086. Before merge the user confirms the plan's Decisions table and looks at the two views in the browser."
````

In `docs/ai/05-update-tracker.md`, replace:

````markdown
Reverse chronological (newest at top).

---
````

with:

````markdown
Reverse chronological (newest at top).

---

## 2026-10-02 — Magcoupling GUI: smarter ordering of the inputs and the results (branch magcoupling/ordering)
- Inputs (decision O-1, O-2): the workflow order is the default, `inputs::WORKFLOW`:
  Requirements and operating conditions, Magnets and rings, Gap and clearances, Housing and
  retainers, Shaft, key and clamps, Materials, Thermal and demagnetization, Calibration and model,
  every input in exactly one section (a coverage test lists any input added without one); the
  adhesive and its bondlines share a section, the optional adapter and its joint too, and the
  fields stored from a 3D run sit in view beside the 3D reference torques; the clamp and screw
  factors, the screw classes, the optional adapter and the model constants (27 rows, O-4) sit
  under each group's closed Advanced heading. "Workbook groups"
  brings back the package groups; the order is panel state only (design files and share links are
  byte-for-byte unchanged: a test compares the design and the link across the toggle). A filter
  box (O-5) matches label, path or cell as the results search (`format::search_haystack`, shared)
  and shows the matching rows alone; a leaf term focusing a row clears it.
- Results (O-6, O-7): the table groups the results: the headline (the dashboard's 17 rows, open),
  eight chains (the six A-3 chains, each with the results of its nested groups the scope leaves
  out, `CHAIN_PREFIXES`, then adhesive and mass), then "Other results" by package
  (`result_groups.rs`), each result in exactly one group (the first that lists it); each heading
  shows the worst level of its checks, and a heading click while searching is ignored; "Engine
  order" brings back the flat table, and both exports keep the engine order. `dashboard::CHECKS`
  lists the 30 design checks (the dashboard's 7 and 23 more, with levels in `verdict_level`);
  their rows carry the dashboard's badge, and "Failing checks only" lists the red then the amber
  ones.
- Tracing (O-8, `trace.rs`): a click on an input's label frames every explained result it drives
  (the registry's `downstream`: 392 of the 1086 results have an equation record; an input that
  reaches none says so), a click on a result frames the inputs it reads (`upstream_inputs`) unless
  an input's trace marks it or it has no record (the trace then stays), in the selection colour
  through `Readouts::mark` (`TermColors::with_marked`), with a banner, a Clear trace button,
  per-group traced counts and a "Traced only" table filter for an input's trace.
- The linkage app's calculator window is unchanged (its tests pass). Docs: the magcoupling README,
  03-structure, 04-memory, the spec's M4 layout amendment.
````

- [ ] **Step 3: The spec's amendment**

In `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, replace:

````markdown
  A **Key design** group on top: face gap, pole count, magnet part, axial length,
  operating temperature, back iron, cup wall, conductance, measured drag (about a
  dozen).
````

with:

````markdown
  A **Key design** group on top: face gap, pole count, magnet part, axial length,
  operating temperature, back iron, cup wall, conductance, measured drag (about a
  dozen). *Amended 2026-10-02 (plan
  `docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md`, decisions O-1 to
  O-8):* the inputs are ordered by design workflow by default, the requirements and
  operating conditions first, with the package groups one toggle away, a filter box
  and each group's rarely changed rows under Advanced; the results table groups the
  results by physics chain, badges the checks (each group heading with its worst
  check) and filters the failing ones; a click traces an input to the explained
  results it drives and a result to the inputs it reads.
````

- [ ] **Step 4: Copy this plan into the repository and check the docs**

Run:

```bash
cp C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/order-plan/plan.md C:/Users/Cole/source/repos/lsim-mag-order/docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md
python -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ('C:/Users/Cole/source/repos/lsim-mag-order/docs/ai/03-structure.yaml', 'C:/Users/Cole/source/repos/lsim-mag-order/docs/ai/04-memory.yaml')]; print('yaml ok')"
git -C C:/Users/Cole/source/repos/lsim-mag-order diff --check
```

Expected: `yaml ok` (the system Python has PyYAML; the oracle's venv does not); `git diff --check` prints nothing. (If the scratchpad copy is gone, the controller hands the implementer the plan file to copy.)

- [ ] **Step 5: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -1 C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task6.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-order checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`, `GATE PASS`, `0`.

- [ ] **Step 6: Build both web bundles and look at the calculator in a browser**

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs && bash scripts/build_web.sh 2>&1 | grep -E "parity guard|complete!"
```

Expected: both `parity guard: ... builds without workbook-parity` lines and both `complete!` lines. Then serve it in the background (`cd C:/Users/Cole/source/repos/lsim-mag-order/linkage-sim-rs/web && python -m http.server 8080`, noting the python process's PID), load the Playwright tools through ToolSearch (`select:mcp__plugin_playwright_playwright__browser_navigate,mcp__plugin_playwright_playwright__browser_resize,mcp__plugin_playwright_playwright__browser_run_code_unsafe,mcp__plugin_playwright_playwright__browser_take_screenshot,mcp__plugin_playwright_playwright__browser_console_messages,mcp__plugin_playwright_playwright__browser_close`), resize to 1400 x 900 and open `http://localhost:8080/magcoupling/`. Expected, after about 3 seconds: the inputs side shows "By workflow" selected, "Workbook groups", the filter box and the Key design group, then the eight workflow groups closed, "Requirements and operating conditions" first; the Results table tab shows "By physics chain" selected, "Engine order", "Failing checks only", "Traced only" (disabled), "1086 of 1086 results", "Headline (17)" open with its rows and badges (a red badge after its heading), the eight chain headings (Torque to Mass) and "Other results" with its package headings ("Coupling model (66)" with a red badge); a click on the Key design's "Candidate flat-face magnetic gap" label shows the banner "Tracing Candidate flat-face magnetic gap: drives ... explained results" and blue frames on the dashboard rows it drives; `browser_console_messages` at level `error` reports 0 errors. Then open `http://localhost:8080/?tool=magcoupling`: the linkage app's calculator window shows the same inputs side, 0 errors. Stop the server and kill its python process by PID (it can outlive the shell that started it). The bundles are gitignored: `git -C C:/Users/Cole/source/repos/lsim-mag-order status --short` lists only this task's docs.

- [ ] **Step 7: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-order add magcoupling-rs/README.md docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md docs/superpowers/plans/2026-10-02-magcoupling-gui-ordering.md
git -C C:/Users/Cole/source/repos/lsim-mag-order commit -q -F - <<'EOF'
docs(magcoupling): the ordering of the inputs and the results; the plan (O-1 to O-8)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-order status --short
```

Expected: no status lines. The branch is then reviewed as a whole on the session model; nothing is pushed or merged before the user confirms the Decisions to confirm and looks at the two views.

---

## Self-review record

This revision answers a critic's review of the first version (the review found no blocking defect: every block applied, coverage and the registry were exact, design files, share links and exports were unchanged, and no idle frame wrote back). Every finding below was fixed, the non-blocking ones included. For each: what changed and where, how it is pinned, and where this revision departs from the critic's literal fix and why. The Decisions table is still unconfirmed, so where a finding changed a recommendation (O-2, O-4, O-6, O-8), the earlier recommendation is now an alternative in that row.

1. **Tracing reaches only the explained results (high).** Task 5: `Trace::banner` gains the arm `TraceKind::Input if count == 0` ("no explained result reads it (its results have no equation record)"; the critic's "yet" dropped: nothing schedules more records); `TRACE_HINT` reads "Click the label to trace the explained results it drives"; `NOTHING_TRACED` and the three-way empty state `(traced_only, failing_only, searching)`; the glyph test draws `NOTHING_TRACED` and the drive torque's banner. Tests: `an_input_no_explained_result_reads_traces_none_and_says_so` (the banner; 392 of the 1086 results have an equation record; every one of the 172 inputs is a registry leaf, so `Trace::of` is `Some` for each) and the panel test `an_input_whose_results_have_no_equation_record_traces_none_and_says_so` (open Requirements and operating conditions, click the drive torque's label, the banner, "Traced only" draws `NOTHING_TRACED` and not `NO_RESULT`, "0 of 1086 results"). Docs: O-8, the README, the tracker and 04-memory say 392 of 1086. Beyond the fix: Task 4's own empty state already tells a search from no failing check (`(true, false) => NOTHING_FAILS`), so the wrong "No check fails" never ships even between Task 4 and Task 5, and `the_failing_filter_shows_exactly_the_failing_checks_with_their_badges` draws that state (a search no failing check matches says `NO_RESULT`).
2. **A readout click replaced the trace (O-8 amendment).** Task 5, panel `ui`: a click on a result that an input's trace marks only opens its equation, and a click on a result without an equation record keeps the trace (`if !inside && let Some(trace) = Trace::of(path)`). Results table: the trace filter goes off whenever no input is traced (`if !input_traced { self.traced_only = false; }`, which covers the critic's `trace.is_none()` and also a result's trace, whose marks are inputs), and its checkbox is enabled for an input's trace only. Test: `reading_a_traced_result_keeps_the_input_s_trace_and_its_traced_rows` (trace the face gap, tick "Traced only", click the pull-out's table row: the Equation panel opens, the trace and the count stay; untick, Engine order, click a row without a record: the trace stays; click a dashboard result the gap does not drive: a result trace replaces it). The zero-reach panel test checks that Clear trace unticks the filter: a later input trace shows every row until it is ticked again. O-8 records the rule; the old rule is alternative (d).
3. **A heading click during a search flipped a hidden state.** Task 4: the click is ignored while the search holds text; the heading's hover text is then `CLEAR_TO_CLOSE` ("Clear the search to close a group", a constant so the glyph test draws it); `searching` is worked out once a frame and passed in a small `HeadingLine` struct (with the text, the open state, the badge level and the indent: eight loose arguments otherwise, past clippy's limit of seven). Test: `a_heading_click_while_searching_leaves_the_group_as_it_was` (type "f_end", click the Torque heading, the chain stays open and the hover text shows, clear the search: "Torque (49)" closed, Calculator!C92 not drawn). A mutation that toggles during a search fails it.
4. **No idle-frame test of the workflow order.** Task 2, `idle_frames_change_no_input`: the screen is 20000 points tall and, after the workbook order's checks, the workflow order filtered by "." draws all 172 rows ("172 of 172 inputs", both vacuum permeabilities) for 15 idle frames with the design, `last_error` and the undo stack unchanged, as the critic wrote it plus the count and `last_error`.
5. **81% of the results under "Other results" (O-6).** Task 3: `CHAIN_PREFIXES` places, after the scope's lists and before the packages, each unplaced result by the first prefix its path starts with; `CHAIN_LABELS` has eight chains (Adhesive after Slip heating, Mass last; both no scope chain, filled by prefixes alone). Departures from the critic's table, each following the finding's own text: `temperature.summary.` goes to Slip heating, not Temperature (the finding says the steady and fault temperatures skip Slip heating; the five summary rows the temperature chain leaves out are exactly the slip ones), and ten whole paths put the metal design's clearances, gaps, cup body OD and reserves in Geometry (the finding names them as misplaced; the critic's nine prefixes did not reach them): 19 entries. The other results fall from 883 rows to 783, and from 389 to 289 outside the two sweeps; most of what remains is per-candidate data (the sweeps' 494 rows, and 130 of the 152 Shaft clamps rows are the clamp table's per-size columns). The Temperature design and Mass packages are left empty and dropped. Task 4: each heading draws the worst level of its group's checks after its text (`Level` now derives `PartialOrd, Ord` in severity order, Task 3, and `failing_checks` sorts by `Reverse(level)` instead of an ad hoc key), drawn in a child `Ui` so the equal-height lines of `show_rows` stay equal; "Coupling model (66)" is red at the defaults. Tests: `every_chain_prefix_places_a_result_the_scope_leaves_out` (every entry places a result no list placed; each extra chain has rows), new asserts in `the_headline_comes_first_then_the_chains_then_the_other_results` (nine groups before the others; the cold check, a ring limit, the fault peak, the mismatch reading, a daily screen, a mass and the corner gap in their chains; no Temperature design package), the scope test now compares the scope's ids with the scope chains among `CHAIN_LABELS`, and `a_group_heading_opens_its_rows_and_the_engine_order_lists_them_without_headings` checks the red badges after the headline's and the coupling model's headings and none after the mass's.
6. **The requirements came sixth (O-2).** Task 1: the first group is "Requirements and operating conditions" (the operating group, its id kept), with the space claim moved in as its second section (id `operating.envelope`, so every section id still starts with its group's); the order test, the README, the tracker and 03-structure follow.
7. **Important rows under Advanced (O-4).** Task 1: the shaft-to-bore friction and the preload share close the Clamp section; the two stored 3D sections are no longer advanced and are labelled "(refresh after a 3D run)". The count is 27 Advanced rows in 6 sections, not the critic's 23: 16 rows leave Advanced (the 2 clamp rows, 4 reverse fields and 10 slip-loss fields; the critic counted 17), and finding 9 moves the 3 adapter-joint rows into the Advanced adapter section, so 40 - 16 + 3 = 27. The test pins 27 and checks the four moved paths are not advanced and the joint friction is. A visible side effect, seen in the browser check: with the stored slip-loss fields in view, opening Calibration and model widens the inputs side (to about 410 points) to fit their long labels ("Opposite-ring field at an aluminium hub (fundamental, free space)"); the side widens to its labels already (the Requirements group's drive-torque label does too), and alternative (c) of O-4 (the stored fields under Advanced with the new labels) avoids it.
8. **Related inputs split.** Task 1: one Materials section "Adhesive and bondlines" (the selected adhesive, both bondlines, the recommended bondline, the shear modulus, the hot strength retained, the fatigue endurance), so the former Bondlines and Thermal/Adhesive sections are gone and "Magnet and adhesive properties" becomes "Magnet properties"; the adapter's geometry and its joint screws form one Advanced section, "Optional adapter and its joint"; the stored 3D sections sit after the 3D reference torques in Calibration and model. Sections do not nest, so "under a '3D results' label" is three adjacent sections labelled "3D results: reference torques", "3D results: stored reverse fields (refresh after a 3D run)" and "3D results: stored slip-loss fields (refresh after a 3D run)". The workflow test checks that the adhesive's inputs share a section, that the adapter and its joint share one, and that the 3D results sit in the calibration group.
9. **The test claim of Review Focus 3.** Task 3: two designs fire the remaining reachable rules (`temperature.slip_loss.sigma_316_S_m = 5e6` the high-conductivity caution, `materials.nickel.thickness_mm = 0` the uncoated-steel caution). No design fires the ferromagnetic sleeve warning: `engine/api.rs` takes it from the sleeve material's `ferromagnetic` flag and none of the library's four sleeves (316L, Ti-6Al-4V, Inconel 625, PEEK) has it, so its rule text is checked as written (red), as the vent port's "No: key too large" already was. Review Focus 3 now says exactly this.
10. **Per-frame work.** Task 5: the traced rows are worked out once a frame, inside the header row after the search refresh, and kept in an outer `Option` as `failing` is (the critic's "before `horizontal_wrapped`" would read the previous frame's matches on the frame the query changes); a `shown` helper gives the rows for both the count and the lines. Task 4: `searching` once a frame (`is_open_while`), and an `other_started` flag instead of scanning the lines. `with_marked` is kept as it is: it runs only while a trace is on, adds at most 221 entries (the largest downstream set, the inner magnet part's 220 results, and its source) to the term-colour map the explorer already rebuilds every frame, and avoiding it would need a borrowed `TermColors` or a cache keyed by both the trace and the hovered equation, more machinery than a few hundred short strings a frame warrant.
11. **Repetition.** `Trace::count_in` serves both traced counts; `InputEntry.haystack` is built once in `InputCatalogue::new` and `filter_inputs` matches it; `InputCatalogue::section_of(order, path)` replaces the three hand-written searches (the workflow test's two, the focus in Task 2); the XOR in `is_open` is written out (`let chosen = if toggled { !default } else { default }; chosen || searching`).
12. **The verification record.** The whole plan was replayed again (Tasks 0 to 6, every gate), `verify/` was refreshed from that replay, and the record above describes this replay only; the README block and the verification tree now agree.
13. **No change needed** (docs, model tiers, the green "Nominal only: hot test"): unchanged.

Re-run for this revision: every task's steps and gates, the browser check and the mutation checks (the Verification record). Not re-run: the native app by hand and the `gui-smoke` workflow as a workflow (as before). The blocks of Tasks 1 to 5 that no finding touched kept their text; the merge rewrote only the blocks whose code changed, and a fresh block where a change fell outside every block.
