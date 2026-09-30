# Magcoupling Addendum A-2: Parameters and Sizing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the Addendum A engine parameters and sizing to the A-1 engine: every A3 assumption as a flagged engine parameter with its workbook default, rationale and source (the harmonic set up to 11 through one general E7 peak search), "assumptions modified" and reset, A1 inverse sizing on one free variable, the housing autofit suggestion, the space claim and an axial housing that follows the length override (decision A2-8, option B), and the four A-1 carry-overs, while the defaults stay workbook-exact.

**Architecture:** Every new input is Rust-only with a default that reproduces the ported behaviour bit for bit (the harmonic set `coupling.max_harmonic` = 5, the axial length override `coupling.magnets.axial_length_mm` = none), so `Deviations::NONE` parity (1,149 checks), the differential data and every registry probe never move, and no correction is added (the deviation registry is not edited). The E7 search becomes one root isolation of dT/dx in cos² x for any odd set (decision 29). Three small Addendum modules sit beside the ported ones: `assumptions.rs` (the A3 panel's data and engine support), `sizing.rs` (inverse sizing by repeated `compute_all`: a coarse scan refined at the first crossing, at each peak and at each validity edge) and `housing.rs` (the space claim, a Rust-only result group, and the axial housing: with the length override set, the hub length, the cup cavity depth and the retainer span follow the rings they bound, so the stacks the claim reads follow the magnets); the autofit's wall suggestion is one number beside the wall check it comes from.

**Tech Stack:** Rust 2024 edition, std only for the library (wasm32-unknown-unknown clean); `serde_json` (feature `float_roundtrip`) as the only dev-dependency; the vendored Python 3.12 oracle `reference/magcoupling-py/` run with `C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe` (only `--check`: no data file changes); Git Bash for every command.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-a2/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, Addendum A: A1 (self-sizing geometry: inverse sizing, housing autofit, space claim), A3 (assumptions panel: the engine side) and "Addendum testing" (the Inverse sizing, Housing autofit and Assumptions items). A2 (equation explorer, with the dependency-graph traceability test of the assumptions), A4 (teaching notes) and the GUI are OUT of scope: plan A-3 and M4 follow. So is the E4 pole-sweep bondline mismatch that `docs/ai/04-memory.yaml` carries under "Addendum A3" (the block-fit term keeps the workbook's literal 0.05 mm, so at `bond_inner` other than 0.05 the two terms of the max() disagree): the A3 panel lists no bondline assumption, and changing E4 is a registered deviation that needs the user's approval; Task 11 relabels that memory line with its owner (plan A-3 or a deviation proposal) and its trigger. Authorities read with it: the approved verification report `C:/Users/Cole/source/repos/lsim-mag-a2/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (all 31 decisions option A; for this plan decisions 27 (the cup wall stays an input, autofit suggests), 28 (autofit covers classes D and R; the M41 cap thread and the 22 against 25 mm boss OD are M4 design questions), 29 (one general harmonic search with a root-pair guard; `peak_angle_is_the_e7_gate` compared within 1e-12) and section 6.5) and its data file `.../docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`; the A-1 plan `C:/Users/Cole/source/repos/linkage_simulation/docs/superpowers/plans/2026-09-30-magcoupling-addendum-a1-data-physics.md` (decisions A1-A13 confirmed; its carry-overs in `docs/ai/04-memory.yaml`); the user's decision that a negative pull-out (f_end <= 0) is flagged, not corrected; `magcoupling-rs/README.md` (the translation rules and the deviation mechanics are binding).

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-a2`, branch `magcoupling/addendum-a2`, created by Task 0 from the final head of `magcoupling/addendum-a1`. Every command uses absolute paths into it. Nothing is pushed.

## Decisions to confirm

These questions arose while turning the spec, decisions 27-29 and the A-1 carry-overs into code; none of the approved decisions settles them.

**Confirmed (user, 2026-09-30): the recommended option on A2-1 to A2-7 and A2-9; A2-8 takes option B (below).** This plan implements the recommended option of every row except A2-8, whose option B Task 7b implements. Task 0 records these answers and asks nothing again; if the user later changes an answer, the task that implements it stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here, except A2-8) | Alternatives | Tasks |
|---|---|---|---|---|
| A2-1 | The harmonic set: which harmonics, and which sums read it | **The odd harmonics 1, 3, ... up to a selectable highest one** (the Rust-only selector `coupling.max_harmonic`, codes 1, 3, 5, 7, 9, 11, default 5 = the workbook's 1, 3, 5), summed by the Calculator's pull-out, its two circuit sums (C95, C96), every sweep row **and the Calibration prototype**, so the bench correction C9 = measured / model compares like with like. The cells of 1, 3 and 5 keep their terms; a harmonic left out reads 0 shear stress; 7, 9 and 11 add Rust-only cells (the sweeps' column U includes them). A code outside the choices gives NaN sums (decision D3) | (b) any subset (a bit mask): no physical reason to skip a harmonic, and the peak search would need gaps; (c) the Calculator only, the Calibration keeps 1, 3, 5: C9 would correct a different model than the one it is applied to | 1, 2 |
| A2-2 | "Axial magnet length (default)" as the free variable, when every default ring is a library part whose length is fixed | **A Rust-only override `coupling.magnets.axial_length_mm` (blank by default) that sets both rings' length and keeps the rest of each ring** (part or manual cross-section, grade, Br, rating: blocks cut or stacked to length); the M4 Key design group's "axial length" can drive it too | (b) size the manual lengths only and refuse (`Err`) when a ring is a library part: the default design could not use the default free variable; (c) convert library rings to custom dimensions of their grade inside the solver: the arcs' Br changes 1.42 → 1.41 T (E19), the calibration factor switches, and a round trip from a library design fails | 5, 6 |
| A2-3 | The measured calibration factor (Calculator C42) under a length override | **Unchanged: it keys on the part names, as the workbook does**, so B842SH rings with no back iron keep the bench correction at any length and a sized length passes 12.7 mm without a step in torque | (b) the bench correction also requires no override (or the prototype's 12.7 mm): a torque step at the prototype length, and sizing the no-back-iron default for its own torque would not return 12.7 mm | 5 |
| A2-4 | What "meets the target" means for inverse sizing | **A value counts (`sizing::is_valid`, one predicate) when its blocks fit (faceted blocks their flats, the Calculator's C52/C59 comparison, now one function; arcs do not overlap at the magnet mid-radius: width at most 2π(apothem of the ring's inner surface + thickness/2)/N, the inner back apothem for the inner ring and the outer face apothem for the outer, the fill C66/C67 before its min(1, ...)), the keyway leaves hub wall (Calculator C53, the wall past the keyway, > 0), the end-effect factor is in range (f_end > 0, audit M9) and the hot-low torque is finite; it meets when it counts and that torque is at least the target.** The space claim is not a condition (spec A1 shows exceeding it as a red callout); a solution may exceed it and the housing results say by how much | (b) the torque alone: on the default hub the answer for 2.5 N·m would be 12 poles, whose inner blocks overlap (the inner flat is 5.44 mm for a 6.35 mm block); (b') the flat check and the end effect alone, with no arc or hub rule: 12 arcs of 6.35 mm on a 6.14 mm mid-radius pitch would count (the fill's min(1, ...) prices them as if they fitted, so 2.3 N·m would return 12 poles), and a 16 mm bore with a 2.5 mm keyway would be sized to 9.77 mm for 0.5 N·m, where the keyway breaks through the hub (C53 = −0.78 mm); (c) also inside the space claim: turns the spec's callout into a constraint | 6 |
| A2-5 | C45's slider (−0.008 to −0.001) cannot reach ferrite's +0.0035 with the coercivity source at 0 (A-1 carry-over: "widen or split it") | **Neither: keep the workbook's NdFeB slider; a positive β is typed** (a slider range does not reject typed values; M4's value box takes typed entry for every input), C45's help says so, and a test pins E20's cold side for a typed +0.0035. A graded magnet gets its β from its grade (source 1) and needs neither | (b) widen the range to +0.005: `gen_differential.py` samples every random and range-end case from the range, so the differential data regenerate, against "the M2 parity and differential tests are unchanged" (and Python applies |β|: a positive C45 has no workbook meaning); (c) split C45 into two inputs (negative and positive β): one quantity, two inputs | 9 |
| A2-6 | E18 uses Alliance's 68.9 GPa for 6061; the A5 library's 6061 record carries Kaiser's 68.3 GPa (A-1 decision A8: "unify when the user decides") | **The library's 6061 engine modulus (and expansion) read E18's constants, 68.9 GPa and 23.6e-6 /°C; Kaiser's 68.3 GPa stays the sourced reference.** E18's approved probe values stay; the non-default "6061-T6 aluminium" back-iron pick now equals the default no-back-iron design cell for cell (it differed in C104, C105, C201) | (b) E18 reads the library's 68.3: moves E18's approved numbers (M2-basis C104 20.76 → 20.75 MPa, workbook basis 73.87 → 73.74 MPa); (c) keep both (the status quo) | 8 |
| A2-7 | A-1 decision A4 kept one α(Br) (Calibration C22) and the NdFeB density for every magnet and carried "per-ring alpha and density from the grade" to A-2 | **A ring in the grade mode (manual dimensions with a grade picked) takes its grade's α(Br) and density**: its Br at temperature, every torque at another temperature (∝ Br_i(T)·Br_o(T)), its own E20 onsets (its reverse fields scale with its own Br: the stored 3D fields do not separate the opposing ring's share, M3's live fields would), the magnets' mass and the bond block's mass. Library parts (all sintered NdFeB, whose grade values equal C22's default and the NdFeB density) and gradeless magnets keep C22 and the NdFeB density, so C22 stays a live assumption and the NONE-mode differential data stay exact. The magnets' heat capacity keeps the NdFeB specific heat (the grade table has none) | (b) keep A-1's single α (the M4 GUI presets C22 from the grade): mixed rings (NdFeB with ferrite or SmCo) scale with one wrong coefficient and ferrite rings weigh as NdFeB; (c) the grade's α for every ring with a grade, library parts included: C22 would change nothing in the shipped build (every library part has a grade) unless gated as a new registered deviation | 10 |
| A2-8 | A sized axial length can outgrow the cup cavity depth, the retainer span and the hub length (class N inputs, no rule: decision 28). With no rule, no dimension the space claim reads moves with the axial length (the axial stack C134 and the large-diameter stack C137 are sums of class N inputs), so a length-sized design reads "Inside the space claim" at any length up to 50.8 mm: the default design sized to 9.9 N·m gets 50.62 mm magnets in the 15.5 mm cup (stack 31.8 of 35) | Recommended, **not chosen**: no engine rule in A-2, an M4 design question beside the M41 cap thread and the 22 against 25 mm boss OD (decision 28), with Task 7's `a_length_sized_design_reads_inside_the_space_claim_at_any_length` pinning it | **(b), chosen by the user and implemented by Task 7b: the axial housing grows with the magnets.** With the Rust-only override `coupling.magnets.axial_length_mm` set, each class N dimension a ring bounds is its input plus that ring's length change from the ring's own length (its part's, or its manual length): the hub length C123 (it prices the hub, Calculator C112, on whose flats the inner ring sits) with the inner ring; the cup cavity depth C124 (the pocket of the outer ring, and the only term of C134 and C137 that bounds a magnet) with the outer ring; the retainer span C172 (one length for the sleeve over the inner ring and the liner inside the outer ring, C46, echoed as C45) with the longer ring, which it must cover. The cap C133, the web C125, the boss C126, the endplates and the cap thread bound no magnet and stay as typed. The margin the user typed is kept (0.3, 2.8 and 1.8 mm at the defaults), and no dimension ends shorter than its ring, the physical minimum (the workbook has no axial clearance input to add, and report 6.5 finds two identities for the cup depth, so neither is a rule): the floor binds only where an input is already shorter than its ring, and there even the rings' own length raises it. Blank, the three inputs are used untouched: every existing result is bit-identical. Every result that reads them follows (the masses, the heat capacity, both stacks and reserves, the hybrid length, the space claim; C45 shows the span in effect), and `housing.hub_length_mm`, `cup_depth_mm` and `retainer_span_mm` report the values in effect for M4 to draw. Only the override drives it; a manual length or another part moves no housing dimension, as in the workbook. The default design sized by length to 2.5 N·m: 13.77 mm magnets, stacks 32.87 of 35 and 19.87 of 20 mm, inside; to 9.9 N·m: 50.62 mm, "Exceeds the space claim: overall length 34.72 mm over, large-diameter bay 36.72 mm over". (The draft's (b), an overshoot check per ring with no growth, is superseded by the user's choice.) | 5, 7, 7b, 11 |
| A2-9 | Where the sizing mode, the target and the free variable live | **Engine function arguments only** (`sizing::solve(inputs, variable, target_Nm)`); the mode switch and its persistence are M4's GUI state | (b) Rust-only inputs for mode, target and variable, so design files record them now: M4 has not designed the file format | 6, 11 |

Settled here without a decision (each follows from the spec or an approved decision and is stated where it is implemented): the coarse scan has 64 cells, refined at the first crossing (bisection), at each peak (a golden-section search where three valid samples rise and then do not rise) and at each validity edge (bisection), each to 1e-9 mm (a round trip needs 1e-6); a torque hump whose rise and fall both lie inside one cell is the stated limit (a unit test pins it); "not reachable" reports the best valid value the search evaluated (the samples, the validity edges and the refined peaks), which is the spec's "best value achieved inside the range" up to that limit and, at a flat peak, the torque's floating-point resolution; the space claim reads the workbook's reserves (C136, C138, C139: exceeded exactly when below 0) and names each exceeded axis with its overshoot to 0.01 mm and at least 0.01 mm (never "0.00 mm over"), an axis that is not a number reading "unknown" beside the exceeded ones; the autofit's wall suggestion is the wall check's own `ceiling(t_bi, 0.1)` (so 1.9 mm reads 19 × 0.1 as Excel's CEILING gives it) and "n/a" exactly when the check reads "No back iron"; the demagnetization block's C43 shows the governing ring's coefficient; the end-effect flag is strict at 0 and NaN is out of range.

## Global Constraints

Every task's requirements implicitly include this section.

- Crate `magcoupling-rs/` stays a sibling of `linkage-sim-rs/`: "No Cargo workspace conversion." The library is pure std with no `[dependencies]`; `cargo check --target wasm32-unknown-unknown --lib` stays clean. The `workbook-parity` feature is reachable only through the self dev-dependency.
- "One pure entry point `compute_all(&DesignInputs) -> Results`: no I/O, no global state, milliseconds per call." Its signature does not change; inverse sizing calls it, it never calls sizing.
- "Text verdicts are reproduced character for character." "Selector integer codes match the workbook." "Units as the package: mm, N·m, °C, T, kA/m, MPa, W, J, rpm, g."
- Addendum A: "At default inputs and default assumptions every result stays workbook-exact, so the M2 parity and differential tests are unchanged." After every task: `Deviations::NONE` parity (1,149 checks: numbers 1e-9 relative, 1e-12 absolute, text exact), every differential file and every registry test pass unchanged, and `all_corrections_together_give_the_reviewed_headline` (pull-out 2.688 N·m, limit 93.06 °C, clamp M4 x 14) is never edited. This plan adds no correction: `src/engine/deviations.rs` is not edited.
- Every new input is Rust-only (`param_rust_only`): no workbook cell, never passed to Python, and its default reproduces the ported behaviour bit for bit. Every new result is Rust-only (`out_rust_only`) unless it is a workbook cell (none is).
- `reference/magcoupling-py/` is not edited (its engine never; this plan changes no tool either); `gen_differential.py --check` must report the data current after every task. The Addendum A report and data file are evidence: never edited.
- Porting and translation rules (architecture sections 7 and 8, README): operand order never changed; `py_min`/`py_max`; never `+`, `-` or `*` on an `i64` from an input; a selector code outside its choices gives NaN or `"#N/A"`, never another row (decision D3); `x.powi(2)` is `x * x` exactly, which the per-ring products rely on for bit-identity.
- Every gate run: `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate 7/7` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them). Gate 5 is `cargo clippy --all-targets -- -D warnings` on `magcoupling-rs`.
- `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml` before every commit. Every edit block below is already in rustfmt's layout (it was produced by formatting), so each `cargo fmt` step should change nothing; run it anyway, where it appears.
- The "replace" blocks quote the A-1 end state as the A-1 plan's replay produced it. If a block's text is not found (A-1's executors changed a line the plan quotes), stop and escalate; do not improvise a match. The likely mismatch points are the blocks in `magcoupling-rs/README.md`, `docs/ai/*.yaml` and `src/lib.rs` (they quote whole table rows or paragraphs as context); the Rust source blocks were produced by rustfmt and are stable. Files in the worktree may carry CRLF line endings (`core.autocrlf`): match the text, not the line endings; git normalizes them on commit.
- Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that wrote the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The commit blocks below write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the session model); a `sonnet` task replaces `Claude Opus 5.5` with its own model's name. Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`): each task updates `magcoupling-rs/README.md` where it changes layout, tests or behaviour; Task 11 updates `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` and `05-update-tracker.md`. Read `docs/ai/*.yaml` before starting (Task 0).
- Model tiers (CLAUDE.md section 5): each task's **Model:** line. Tasks 0, 3 to 9 (7b included) and 11 give exact code or data and run on `sonnet` (set `model: 'sonnet'`); Tasks 1, 2 and 10 are physics (the E7 search, the harmonic sums, temperature scaling across modules) and run on the session model (omit `model` on purpose, comment `// session model: physics`), as do their physics reviews. A `sonnet` attempt that ends `blocked`, leaves the gate red or is rejected in review is retried on the session model. The final whole-branch review runs on the session model.
- Equality edges (architecture section 7 step 8): each new threshold comparison gets a unit test at exact equality that asserts the equality first (`the_end_effect_check_is_strict_at_zero`, `the_space_claim_is_exceeded_exactly_when_a_dimension_exceeds_it_per_axis`, `the_wall_suggestion_is_the_number_the_advice_quotes`, `arcs_fit_exactly_at_a_pitch_share_of_one`, `the_hub_wall_counts_only_while_positive`, `the_space_claim_trips_exactly_where_a_derived_stack_crosses_its_claim`, `a_housing_dimension_never_ends_shorter_than_its_ring`).
- Commit subjects: `feat(magcoupling-rs): ...` for features, `fix(magcoupling-rs): ...` for the one data unification (Task 8), `docs(magcoupling-rs): ...` for Task 9 (a help text and a test pinning existing behaviour, no number changes) and `docs(magcoupling): ...` for Task 11.

## Review Focus

Six input classes the spec implies but no parity or differential case holds (every new input is Rust-only, so the differential data only ever hold its default). Each names the tests that pin it and its task.

1. **A sizing target met only where the model is out of range, or never met by a layout that fits** (a tiny target at short lengths with a large end-effect coefficient; a pole count whose blocks overlap on the default hub, as flats or as arcs; a keyway that breaks through the hub; blocks too wide for any pole count). Expected: the smallest value that counts (f_end > 0, the blocks fit, the keyway leaves hub wall); "not reachable" with the best valid value, or none. Tests: `short_lengths_where_f_end_is_not_positive_never_count`, `the_default_design_sized_to_its_requirement`, `overlapping_arcs_never_count`, `a_hub_the_keyway_breaks_through_never_counts`, `nothing_valid_in_the_range_reports_no_best_value`, and the equality edges `arcs_fit_exactly_at_a_pitch_share_of_one`, `the_hub_wall_counts_only_while_positive` (Task 6).
2. **Torque that is not monotone in the free variable** (the ring radius on the default design: 2.5 N·m is met from 11.7 to 15.7 mm and again from 29.8 to 30 mm; with no back iron and a 0.5 mm gap, 2.0695 N·m is met only inside one grid cell near 12.11 mm, between two samples that miss; a target just below the default design's true peak, 2.5951122 N·m, is met by no grid value; with a 24 mm bore and a 4 mm keyway, 2.46 N·m is met just above the validity edge at 16.05 mm and missed at the next sample). Expected: the smallest meeting value, never a later bracket and never a false "not reachable"; "not reachable" reports the refined peak. The stated limit: a hump whose rise and fall both lie inside one cell. Tests: `the_smaller_of_two_meeting_intervals_is_returned`, `a_meeting_interval_inside_one_cell_is_found`, `a_target_just_below_the_true_peak_is_solved`, `a_target_met_only_where_the_keyway_starts_to_leave_wall_is_found`, `a_target_beyond_the_range_is_not_reachable`, and the search's unit tests in `sizing.rs` (each refinement, both directions of a validity edge, a plateau, stepping, the bound on a peak search, the stated limit) (Task 6).
3. **A value no slider allows, from a struct literal, a design file or a share link** (an odd pole count as the sizing base; a harmonic code of 4; a NaN length override or space claim; a NaN input or a non-positive target handed to sizing). Expected: even poles only; NaN, never another harmonic set; `validate()` names the path; the space claim reads unknown, not inside (and names any exceeded axis beside it); sizing returns `Err`, never panics. Tests: `poles_stay_even`, `invalid_targets_and_inputs_fail_loudly` (Task 6), `an_invalid_harmonic_set_is_nan_not_another_set` (Task 2), `compute_all_never_panics_on_non_finite_struct_literals` (Task 5), `several_axes_are_named_in_order_and_nan_is_unknown` (Task 7), `sizing_never_panics_on_extreme_designs` (Task 6).
4. **A derived dimension exactly at its space claim, just past it, or several axes over at once; a length-sized design** (claim = dimension − 0.001 mm; the default design sized by length to 2.5 or 9.9 N·m; an override that carries a stack across its claim). Expected: at the claim is inside; each exceeded axis is named in order with its overshoot, at least 0.01 mm; a length-sized design grows its housing (decision A2-8, option B): inside at 2.5 N·m, over by 34.72 mm (length) and 36.72 mm (bay) at 9.9 N·m, as the hand sums of C134 and C137 give; each stack trips the claim exactly where it crosses it (found to the last bit: a typed 13.9 mm override gives a bay stack of 20.000000000000004 mm, "0.01 mm over"). Tests: `the_space_claim_is_exceeded_exactly_when_a_dimension_exceeds_it_per_axis`, `several_axes_are_named_in_order_and_nan_is_unknown`, `a_tiny_overshoot_reads_at_least_a_hundredth` (Task 7), `a_design_sized_by_length_grows_its_housing_and_can_exceed_the_space_claim`, `the_space_claim_trips_exactly_where_a_derived_stack_crosses_its_claim` (Task 7b).
5. **An axial length override over housing inputs that do or do not cover their rings** (rings of different own lengths, either ring the longer; an override at the rings' own length, far below it or far above it; a hub typed shorter than its magnets; the ranges' extremes, 50.8 mm manual rings set to 2 mm over 3 mm inputs; a NaN override). Expected: blank, the three inputs untouched (no floor either); at the rings' own length nothing moves where each dimension covers its ring; the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring by its change; never shorter than the ring (the floor raises an input already shorter than its ring as soon as the override is set, and never replaces a NaN input); the masses and the heat capacity follow; NaN reads unknown, never inside. Tests: `with_the_override_blank_the_axial_housing_is_the_inputs`, `setting_the_override_to_the_rings_own_length_changes_nothing`, `a_short_override_shrinks_the_axial_housing`, `each_dimension_follows_the_ring_it_bounds`, `a_housing_dimension_never_ends_shorter_than_its_ring`, `the_housing_masses_follow_the_dimensions_in_effect`, `a_nan_override_leaves_the_axial_housing_unknown`, and Task 5's `a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass` (rewritten: the stack and the span grow) (Task 7b).
6. **Rings of different materials in the grade mode** (an NdFeB inner ring with a ferrite outer ring; a Y30 ring beside a library part). Expected: each ring's own α(Br) and density; torques at temperature with both rings' coefficients; each ring's E20 onsets with its own; library parts unchanged. Tests: `a_grade_ring_takes_its_grade_alpha_and_density`, `each_ring_is_checked_with_its_own_coefficient`, `e20_checks_both_rings_each_side_from_the_weaker` (updated), `a_grade_ring_scales_with_its_own_alpha_and_weighs_at_its_density` (Task 10).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/`):

| File | Responsibility | Task |
|---|---|---|
| `src/engine/assumptions.rs` | Addendum A3: `ASSUMPTIONS` (the spec's 14 rows over the 15 flagged inputs: label, paths, rationale, source), `states`, `modified`, `any_modified`, `reset_to_workbook_defaults` | 4 |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing: `FreeVariable`, `is_valid`, `solve` (the search, unit-tested with plain functions), `SizingOutcome`, `SizingError` | 6 |
| `src/engine/housing.rs` | Addendum A1 space claim (`HousingResults`, the Rust-only `housing` group) and the autofit classes in its docs; the axial housing that follows the length override (`AxialHousing`, `axial_housing`, decision A2-8) and its three dimensions in effect | 7, 7b |
| `tests/assumptions.rs` | The registry against the metadata; modified and reset | 4 |
| `tests/sizing.rs` | The length override, inverse sizing, the space claim and the axial housing that follows the override, end to end | 5, 6, 7, 7b |

Modified: `src/engine/{model,calibration,sweeps,api,materials,material_library,metal_design,temperature,grades,mod}.rs`, `src/lib.rs`, `tests/{schema,robustness,material_library,material_links,grades}.rs`, `tests/data/input_schema.json` (blessed, Tasks 2, 5, 7b, 9), `README.md`, `docs/ai/{02-system,03-structure,04-memory}.yaml`, `docs/ai/05-update-tracker.md`. No differential, parity, golden or Python file changes (they are the proof the defaults did not move).

Order: the peak search alone first (so its bit-identity on 1, 3, 5 is isolated), then the harmonic set; the end-effect flag before the assumptions text that names it and the sizing rule that uses it; the length override before sizing; sizing before the space claim test that sizes a design; the space claim before the axial housing that moves it (Task 7b, the user's choice on A2-8, a task of its own so the forward space claim is proven first and the growth rule carries its own red test and gate); then the three carry-overs and the docs.

## Verification record

Every edit, script and command of Tasks 0 to 11 and 7b was replayed (last after the amendment for the user's decisions of 2026-09-30, recorded at the end of this plan with the critique's revision), task by task and in this order (7b between 7 and 8), on a fresh copy of the A-1 end state (the tree the A-1 plan's replay produced: `magcoupling-rs`, `reference/magcoupling-py`, `docs/ai`, `docs/analyses` and the specs, with line endings normalized to LF as git stores them), not in the worktree. Each see-it-fail step failed with the error it names, each later step passed, and after every task `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo check --target wasm32-unknown-unknown --lib`, `cargo fmt --check` and `gen_differential.py --check` (with the oracle interpreter) passed. The gate script itself was not run in the replay (the copy has no `linkage-sim-rs`); its four magcoupling gates were run one by one. The oracle's own `pytest` (1159 passed at A-1) is unaffected: this plan changes no Python file. The replayed tree equals, file for file, the development tree the edit blocks were taken from (git-ignored Python bytecode caches aside), except `docs/ai/05-update-tracker.md`, whose entry Task 11's script writes with the day's date.

Bit for bit: every result of the default design, of the workbook defaults and of all 3,391 differential input sets, with the corrections off and on, was compared as bit patterns with the A-1 end state (the results this plan adds left out), on the development tree, which the replay proved equal to the replayed one. With the corrections off nothing moves. With them on, 383 input sets differ, in E7-active values only and from Task 1 alone (the general root search against the closed form): headline values by at most 7e-16 relative, a near-zero harmonic term by 1.1e-12 relative, no text anywhere. Tasks 2 to 11 and 7b move no bit of any existing result in either mode. The dump was taken again after the critique's revision of Tasks 6, 7 and 11 (the refined search, `pitch_share` factored out of the fill, the space claim's quoting) and equals the earlier one bit for bit. It was taken again after the amendment (Task 7b's code, and the rebuilt Tasks 8 to 11), in a debug and in a release build, each with and without the four results Task 7 adds to `housing`, and each equals its pre-amendment dump bit for bit (6,786 case hashes; a set of 65 hashes in which the debug build differs from the release build is the same before and after): the axial housing acts only through the override, which no dump case sets. Task 7b's own tests show what the override does: at the rings' own length every result keeps its bits, and a design sized by length now reads its overshoot.

Test counts after each task (unit tests / `assumptions` / `deviations` / `differential` / `grades` / `material_library` / `material_links` / `parity` / `python_schema` / `robustness` / `schema` / `sizing` / `static_data` / doc-tests):

| After | Counts |
|---|---|
| Task 0 (baseline) | 107 / – / 54 / 19 / 9 / 4 / 11 / 4 / 7 / 9 / 7 / – / 8 / 1 (+1 ignored) |
| Task 1 | unit 112 |
| Task 2 | unit 118, robustness 10 |
| Task 3 | unit 121 |
| Task 4 | assumptions 8 |
| Task 5 | unit 122, sizing 1 |
| Task 6 | unit 133, robustness 11, sizing 17 |
| Task 7 | unit 134, sizing 23 |
| Task 7b | sizing 31 |
| Task 8 | unchanged |
| Task 9 | robustness 12 |
| Task 10 | unit 138, grades 10 |
| Task 11 | 138 / 8 / 54 / 19 / 10 / 4 / 11 / 4 / 7 / 12 / 7 / 31 / 8 / 1 (+1 ignored) |

Cost, measured on the development machine: `compute_all` at the defaults with every correction on takes about 17 µs in a release build and 55 to 65 µs in a debug build (A-1: about 8.5 and 26 µs; the E7 root search runs about 23 times per call); one `sizing::solve` on the default design takes 3 to 139 `compute_all` calls: 19 for magnets per ring (3 at 1 N·m, met at 8 poles), 37 to 65 for the axial length, 33 to 139 for the ring radius (its validity edge near 9.77 mm adds about 29 and its peak near 13.6 mm about 45); that is about 0.1 to 2.7 ms in a release build and 0.35 to 10 ms in a debug build (measured at the targets 1, 2.5 and 10 N·m). A torque noisy enough to show a peak at every other sample would cost about 1,500 calls (65 samples and 32 peak searches of about 45), about 26 ms release.

Task 7b's cost, measured the same way before and after it (release build, three rounds): `compute_all` at the defaults 15.6 to 16.1 µs before and after (the override blank adds no work); with the override set, 15.1 to 15.2 µs before and 16.1 to 16.8 µs after (a second magnet lookup for the rings' own lengths); one length solve on the default design at 2.5 N·m, 0.81 to 0.85 ms before and 0.83 ms after.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` for Steps 1-5 (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 6 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: branch `magcoupling/addendum-a1` at its final head (the A-1 plan's Task 14 commit, `docs(magcoupling): Addendum A-1 status, invariants and open items`, reviewed and gate-green).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-a2` on the new branch `magcoupling/addendum-a2`, a confirmed green baseline, and the controller's record of the user's answers to the Decisions to confirm (already given: see Step 6).

- [ ] **Step 1: Create the worktree**

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 magcoupling/addendum-a1
git -C C:/Users/Cole/source/repos/linkage_simulation worktree add -b magcoupling/addendum-a2 C:/Users/Cole/source/repos/lsim-mag-a2 magcoupling/addendum-a1
git -C C:/Users/Cole/source/repos/lsim-mag-a2 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short
```

Expected: the `log` line is A-1's final commit, `docs(magcoupling): Addendum A-1 status, invariants and open items` (if it is not, A-1 has not finished: stop and escalate); `worktree add` prints `Preparing worktree (new branch 'magcoupling/addendum-a2')`; then `magcoupling/addendum-a2`; no status lines.

- [ ] **Step 2: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`; `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` whole (the Deviations and Translation rules sections are binding); the spec's Addendum A (`C:/Users/Cole/source/repos/lsim-mag-a2/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, sections A1, A3 and "Addendum testing"); the verification report `C:/Users/Cole/source/repos/lsim-mag-a2/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (sections 6.5, 6.6 and 8, decisions 27 to 29); the A-1 plan's Decisions to confirm (`C:/Users/Cole/source/repos/linkage_simulation/docs/superpowers/plans/2026-09-30-magcoupling-addendum-a1-data-physics.md`, A1 to A13) and this plan's Decisions to confirm.

- [ ] **Step 3: Run the crate's tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: in order: unit tests `107 passed`; `tests\deviations.rs` 54 passed; `tests\differential.rs` 19 passed; `tests\grades.rs` 9 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 9 passed; `tests\schema.rs` 7 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. These are the counts of the tree this plan was proven against (A-1's replay); if a count differs but every binary is `ok`, record the actual counts, continue, and read every later task's counts as offsets from them.

- [ ] **Step 4: Check the oracle Python**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
cd C:/Users/Cole/source/repos/lsim-mag-a2/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `ls` prints the path; then `differential data is current (...)`. If `ls` fails, stop and escalate: every gate run of this plan uses that interpreter.

- [ ] **Step 5: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task0.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Decisions**

The user answered this plan's **Decisions to confirm** on 2026-09-30: the recommended option on A2-1 to A2-7 and A2-9, and option B on A2-8 (the axial housing grows with the magnets: Task 7b). The controller records these answers in the plan's execution notes and does not ask again. Every task implements its decision's confirmed option and names the decision in its intro; if the user later changes an answer, the task that implements it stops and escalates instead of improvising.

---

### Task 1: One general E7 peak search for any odd harmonic set (decision 29)

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5: the E7 physics). Physics reviewer: report section 6.5 ("Harmonic set"), decision 29, audit row E7, and the memory note that the three E7 cancellation tests and the fill-0.4 test must keep passing.

Decision 29 A: "one general search with a Sturm or Q' root-pair guard, and `peak_angle_is_the_e7_gate` changed to compare within
1e-12". dT/dx = cos x · Q(u) with u = cos² x, Q a polynomial of degree (n_max − 1)/2 (at most 5 for harmonic 11). The search
isolates every root of Q in [0, 1] by recursion on derivatives: the derivative's roots split [0, 1] into pieces where Q is monotone,
so each piece holds at most one root and bisection pins it. That is the report's Q' guard applied at every level, with no scan grid,
so the report's failure mode (a root pair inside one scan cell, 2 flips in 100,000 cases) cannot occur, and no quadratic formula, so a
vanishing top amplitude cannot cancel digits. The closed form moves into the tests as the reference the new search must agree with
within 1e-12 rad on 3,000 triples (the report measured 5.0e-16). This task changes no harmonic set: every caller still passes the
workbook's three amplitudes, so with E7 off nothing runs and with E7 on the None/Some decision is the closed form's; the E7 registry
tests (`e7_leaves_every_default_cell_bit_for_bit`, `each_probe_shows_its_correction`, `e7_finds_the_peak_at_a_fill_of_exactly_0_4`)
and `all_corrections_together_give_the_reviewed_headline` pass unedited. Where E7 moves the peak (off-default designs only) the new
root can differ from the closed form's in the last bits: over the 3,391 differential input sets with every correction on, 383 differ,
headline values by at most 7e-16 relative and a near-zero harmonic term by 1.1e-12 relative, all inside the 1e-9 parity rule and with
no text change (the report measured 7.4e-13 for its prototype; decision 29 accepts it). Every default design, and every case with the
corrections off, stays bit-identical (the plan's replay compared every result's bits; Tasks 2 to 11 move none).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs` (`ODD_HARMONICS`, `peak_off_half_pitch` on a slice, `stationary_polynomial`, `roots_in_unit_interval`, `peak_angle` on a slice; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs` (`peak_angle(&amps, dev)`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (E7 row)

**Interfaces:**
- Consumes: `model::peak_off_half_pitch(a: [f64; 3]) -> Option<f64>` (the cancellation-free closed form for 1, 3, 5), `model::peak_angle(a: [f64; 3], dev: Deviations) -> Option<f64>`, `model::tau_at(a: f64, n: u32, x: f64) -> f64`.
- Produces:
  - `pub const ODD_HARMONICS: [u32; 6] = [1, 3, 5, 7, 9, 11]` (the harmonics the search handles; `HARMONICS: [u32; 3] = [1, 3, 5]` stays the workbook set, pinned to Python by `tests/static_data.rs`);
  - `pub fn peak_off_half_pitch(a: &[f64]) -> Option<f64>`: `a[i]` is the amplitude of harmonic `ODD_HARMONICS[i]` (1 to 6 entries); `None` exactly when half a pitch is the maximum (same 1e-12 relative rule as before);
  - private `fn stationary_polynomial(a: &[f64]) -> Vec<f64>` (Q(u), u = cos² x) and `fn roots_in_unit_interval(c: &[f64]) -> Vec<f64>` (every real root in [0, 1], ascending);
  - `pub(crate) fn peak_angle(a: &[f64], dev: Deviations) -> Option<f64>` (callers pass `&[..]`; Task 2 passes the harmonic set's slice).

- [ ] **Step 1: Write the failing tests**

The existing E7 unit tests keep their assertions; their calls pass slices, the triple generator of `peak_off_half_pitch_finds_the_brute_force_maximum` moves into `random_triple` (same seed, same draws) and `torque_at` / `grid_maximum` take any number of harmonics. New: the closed-form reference `closed_form_peak` with the 3,000-triple agreement test, a 3,000-spectrum brute-force test up to harmonic 11, and three tests of `roots_in_unit_interval` (a quintic's five roots, a root pair inside one scan cell, the edges).

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust

    #[test]
    fn half_pitch_stays_the_peak_when_the_third_harmonic_is_small() {
        assert_eq!(peak_off_half_pitch([1.0, -0.011, 0.001]), None); // default design: A3/A1 = -0.011
        assert_eq!(peak_off_half_pitch([1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn a_large_third_harmonic_moves_the_peak_and_raises_it() {
        let a = [1.0, 0.2, 0.0]; // A1 < 9 A3: half a pitch is a local minimum
        let x = peak_off_half_pitch(a).expect("the peak moves");
        let t = |x: f64| a[0] * x.sin() + a[1] * (3.0 * x).sin() + a[2] * (5.0 * x).sin();
        assert!(x > 0.0 && x < std::f64::consts::FRAC_PI_2);
        assert!(t(x) > t(std::f64::consts::FRAC_PI_2));
```

with:

```rust

    #[test]
    fn half_pitch_stays_the_peak_when_the_third_harmonic_is_small() {
        assert_eq!(peak_off_half_pitch(&[1.0, -0.011, 0.001]), None); // default design: A3/A1 = -0.011
        assert_eq!(peak_off_half_pitch(&[1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn a_large_third_harmonic_moves_the_peak_and_raises_it() {
        let a = [1.0, 0.2, 0.0]; // A1 < 9 A3: half a pitch is a local minimum
        let x = peak_off_half_pitch(&a).expect("the peak moves");
        let t = |x: f64| a[0] * x.sin() + a[1] * (3.0 * x).sin() + a[2] * (5.0 * x).sin();
        assert!(x > 0.0 && x < std::f64::consts::FRAC_PI_2);
        assert!(t(x) > t(std::f64::consts::FRAC_PI_2));
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        assert!(slope.abs() < 1e-9, "{slope}");
    }

    /// T(x) = a1 sin x + a3 sin 3x + a5 sin 5x, the curve [`peak_off_half_pitch`] maximizes.
    fn torque_at(a: [f64; 3], x: f64) -> f64 {
        a[0] * x.sin() + a[1] * (3.0 * x).sin() + a[2] * (5.0 * x).sin()
    }

    /// Brute-force maximum of `torque_at` on [0, π/2]: a grid, then a ternary search
    /// around the best grid point. Never above the true maximum.
    fn grid_maximum(a: [f64; 3], points: usize) -> f64 {
        let step = std::f64::consts::FRAC_PI_2 / (points - 1) as f64;
        let (best_i, best) = (0..points)
            .map(|i| (i, torque_at(a, i as f64 * step)))
```

with:

```rust
        assert!(slope.abs() < 1e-9, "{slope}");
    }

    /// T(x) = Σ a_i sin((2i + 1) x), the curve [`peak_off_half_pitch`] maximizes.
    fn torque_at(a: &[f64], x: f64) -> f64 {
        a.iter()
            .zip(ODD_HARMONICS)
            .fold(0.0, |acc, (&an, n)| acc + an * (f64::from(n) * x).sin())
    }

    /// Brute-force maximum of `torque_at` on [0, π/2]: a grid, then a ternary search
    /// around the best grid point. Never above the true maximum.
    fn grid_maximum(a: &[f64], points: usize) -> f64 {
        let step = std::f64::consts::FRAC_PI_2 / (points - 1) as f64;
        let (best_i, best) = (0..points)
            .map(|i| (i, torque_at(a, i as f64 * step)))
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        best.max(torque_at(a, (lo + hi) / 2.0))
    }

    #[test]
    fn peak_angle_is_the_e7_gate() {
        // The one gate every harmonic sum shares: E7 off keeps half a pitch even where it
        // is a local minimum; E7 on is exactly peak_off_half_pitch.
        let a = [1.0, 0.2, 0.0];
        assert_eq!(peak_angle(a, Deviations::NONE), None);
        let on = peak_angle(a, Deviations::only(DeviationId::E7));
        assert!(on.is_some());
        assert_eq!(on, peak_off_half_pitch(a));
        assert_eq!(peak_angle(a, Deviations::ALL), on);
        assert_eq!(tau_at(2.0, 3, on.unwrap()), 2.0 * (3.0 * on.unwrap()).sin());
    }
```

with:

```rust
        best.max(torque_at(a, (lo + hi) / 2.0))
    }

    /// A uniform source in [0, 1) (splitmix64), the property tests' generator.
    fn uniform_source(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed;
        move || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// A random amplitude triple (a1, a3, a5), case `i` of three bands: the general case,
    /// a fifth harmonic in the band |a5/a1| <= 1.5e-13 where the textbook quadratic cancels
    /// (a ring at a fill of exactly 0.4 or 0.8 gives sin(5 fill pi/2) ~ 1e-16), and a5 = 0.
    /// a1 in [0.5, 1.5], a3 in +-a1/2 (A1 < 9 A3 moves the peak), so half a pitch stays
    /// positive and the maximum is a stationary point in (0, pi/2].
    fn random_triple(uniform: &mut impl FnMut() -> f64, i: usize) -> [f64; 3] {
        let a1 = 0.5 + uniform();
        let a3 = (uniform() - 0.5) * a1;
        let sign = if uniform() < 0.5 { -1.0 } else { 1.0 };
        let a5 = match i % 3 {
            0 => (uniform() - 0.5) * 0.6 * a1,
            1 => sign * a1 * 10f64.powf(-20.0 + 7.0 * uniform()) * 1.5,
            _ => 0.0,
        };
        [a1, a3, a5]
    }

    #[test]
    fn peak_angle_is_the_e7_gate() {
        // The one gate every harmonic sum shares: E7 off keeps half a pitch even where it
        // is a local minimum; E7 on is exactly peak_off_half_pitch, which agrees with the
        // workbook set's closed form within 1e-12 rad (decision 29 A).
        let a = [1.0, 0.2, 0.0];
        assert_eq!(peak_angle(&a, Deviations::NONE), None);
        let on = peak_angle(&a, Deviations::only(DeviationId::E7));
        assert!(on.is_some());
        assert_eq!(on, peak_off_half_pitch(&a));
        let closed = closed_form_peak(a).expect("the closed form moves the peak too");
        assert!((on.unwrap() - closed).abs() <= 1e-12, "{on:?} vs {closed}");
        assert_eq!(peak_angle(&a, Deviations::ALL), on);
        assert_eq!(tau_at(2.0, 3, on.unwrap()), 2.0 * (3.0 * on.unwrap()).sin());
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    fn a_vanishing_fifth_harmonic_does_not_lose_the_peak() {
        // Final review, E7: with a5 -> 0 the quadratic's small root must tend to the linear
        // root -qc/qb (the a5 = 0 answer), not cancel to a wrong angle or to None.
        let x0 = peak_off_half_pitch([1.0, 0.2, 0.0]).expect("the peak moves");
        for a5 in [1e-20, -1e-20, 1e-16, -1e-16, 1e-14, 1.5e-13, -1.5e-13] {
            let x = peak_off_half_pitch([1.0, 0.2, a5]).unwrap_or_else(|| panic!("{a5}: None"));
            assert!((x - x0).abs() < 1e-9, "{a5}: {x} vs {x0}");
        }
    }

    #[test]
    fn peak_off_half_pitch_finds_the_brute_force_maximum() {
        // Property test over random amplitude triples: the general case, a fifth harmonic
        // in the band |a5/a1| <= 1.5e-13 where the textbook quadratic cancels (a ring at a
        // fill of exactly 0.4 or 0.8 gives sin(5 fill pi/2) ~ 1e-16), and a5 = 0.
        // a1 in [0.5, 1.5], a3 in +-a1/2 (A1 < 9 A3 moves the peak), so half a pitch
        // stays positive and the maximum is a stationary point in (0, pi/2].
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut uniform = move || {
            // splitmix64
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut misses = Vec::new();
        for i in 0..3000 {
            let a1 = 0.5 + uniform();
            let a3 = (uniform() - 0.5) * a1;
            let sign = if uniform() < 0.5 { -1.0 } else { 1.0 };
            let a5 = match i % 3 {
                0 => (uniform() - 0.5) * 0.6 * a1,
                1 => sign * a1 * 10f64.powf(-20.0 + 7.0 * uniform()) * 1.5,
                _ => 0.0,
            };
            let a = [a1, a3, a5];
            let got = match peak_off_half_pitch(a) {
                Some(x) => {
                    assert!(
                        (0.0..=std::f64::consts::FRAC_PI_2).contains(&x),
                        "{a:?}: {x}"
                    );
                    torque_at(a, x)
                }
                None => a1 - a3 + a5,
            };
            let want = grid_maximum(a, 1001);
            if got < want - 4e-12 * (a1.abs() + a3.abs() + a5.abs()) {
                misses.push(format!("{a:?}: {got} < {want}"));
            }
        }
```

with:

```rust
    fn a_vanishing_fifth_harmonic_does_not_lose_the_peak() {
        // Final review, E7: with a5 -> 0 the quadratic's small root must tend to the linear
        // root -qc/qb (the a5 = 0 answer), not cancel to a wrong angle or to None.
        let x0 = peak_off_half_pitch(&[1.0, 0.2, 0.0]).expect("the peak moves");
        for a5 in [1e-20, -1e-20, 1e-16, -1e-16, 1e-14, 1.5e-13, -1.5e-13] {
            let x = peak_off_half_pitch(&[1.0, 0.2, a5]).unwrap_or_else(|| panic!("{a5}: None"));
            assert!((x - x0).abs() < 1e-9, "{a5}: {x} vs {x0}");
        }
    }

    #[test]
    fn peak_off_half_pitch_finds_the_brute_force_maximum() {
        // Property test over random amplitude triples (`random_triple`: the general case, a
        // vanishing fifth harmonic, and a5 = 0).
        let mut uniform = uniform_source(0x9E37_79B9_7F4A_7C15);
        let mut misses = Vec::new();
        for i in 0..3000 {
            let a = random_triple(&mut uniform, i);
            let got = match peak_off_half_pitch(&a) {
                Some(x) => {
                    assert!(
                        (0.0..=std::f64::consts::FRAC_PI_2).contains(&x),
                        "{a:?}: {x}"
                    );
                    torque_at(&a, x)
                }
                None => a[0] - a[1] + a[2],
            };
            let want = grid_maximum(&a, 1001);
            if got < want - 4e-12 * (a[0].abs() + a[1].abs() + a[2].abs()) {
                misses.push(format!("{a:?}: {got} < {want}"));
            }
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            "{} misses, e.g. {:?}",
            misses.len(),
            &misses[..misses.len().min(5)]
        );
    }
```

with:

```rust
            "{} misses, e.g. {:?}",
            misses.len(),
            &misses[..misses.len().min(5)]
        );
    }

    /// The closed form E7 used for the workbook set 1, 3, 5 before the harmonic set became a
    /// parameter (kept as the reference decision 29 A compares with): the real roots in
    /// u = cos² x of 80 a5 u² + (12 a3 − 100 a5) u + (a1 − 9 a3 + 25 a5), taken
    /// cancellation-free, filtered as `peak_off_half_pitch` filters.
    fn closed_form_peak(a: [f64; 3]) -> Option<f64> {
        let [a1, a3, a5] = a;
        let torque = |x: f64| a1 * x.sin() + a3 * (3.0 * x).sin() + a5 * (5.0 * x).sin();
        let half_pitch = a1 - a3 + a5;
        let (qa, qb, qc) = (80.0 * a5, 12.0 * a3 - 100.0 * a5, a1 - 9.0 * a3 + 25.0 * a5);
        let roots: Vec<f64> = if qa != 0.0 {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                Vec::new()
            } else {
                let q = -0.5 * (qb + disc.sqrt().copysign(qb));
                if q == 0.0 {
                    vec![0.0]
                } else {
                    vec![q / qa, qc / q]
                }
            }
        } else if qb != 0.0 {
            vec![-qc / qb]
        } else {
            Vec::new()
        };
        roots
            .into_iter()
            .filter(|u| (0.0..=1.0).contains(u))
            .map(|u| u.sqrt().acos())
            .map(|x| (x, torque(x)))
            .filter(|&(_, t)| t - half_pitch > 1e-12 * half_pitch.abs())
            .max_by(|p, q| p.1.total_cmp(&q.1))
            .map(|(x, _)| x)
    }

    #[test]
    fn the_general_search_agrees_with_the_closed_form_on_the_workbook_set() {
        // Decision 29 A: one general search replaces the closed form for 1, 3, 5. On 3,000
        // random triples of the three bands both give None in the same cases and otherwise
        // the same angle within 1e-12 rad.
        let mut uniform = uniform_source(0x2545_F491_4F6C_DD1D);
        let mut disagreements = Vec::new();
        for i in 0..3000 {
            let a = random_triple(&mut uniform, i);
            let (general, closed) = (peak_off_half_pitch(&a), closed_form_peak(a));
            let agree = match (general, closed) {
                (Some(x), Some(y)) => (x - y).abs() <= 1e-12,
                (None, None) => true,
                _ => false,
            };
            if !agree {
                disagreements.push(format!("{a:?}: {general:?} vs {closed:?}"));
            }
        }
        assert!(
            disagreements.is_empty(),
            "{} disagreements, e.g. {:?}",
            disagreements.len(),
            &disagreements[..disagreements.len().min(5)]
        );
    }

    #[test]
    fn the_general_search_finds_the_brute_force_maximum_up_to_harmonic_11() {
        // Decision 29 A: 3,000 random spectra of 1 to 6 odd harmonics (up to 11). a1 in
        // [0.5, 1.5]; harmonic n in +-a1/n, so half a pitch is often a local minimum and the
        // curve can have several peaks; every fourth spectrum of two or more harmonics has a
        // vanishing top harmonic (|a/a1| <= 1.5e-13). The search is never below the brute
        // force by more than 4e-12 of Σ|a|.
        let mut uniform = uniform_source(0xD1B5_4A32_D192_ED03);
        let mut misses = Vec::new();
        for i in 0..3000 {
            let count = 1 + i % 6;
            let a1 = 0.5 + uniform();
            let mut a = vec![a1];
            for &n in &ODD_HARMONICS[1..count] {
                a.push((uniform() - 0.5) * 2.0 * a1 / f64::from(n));
            }
            if count > 1 && (i / 6) % 4 == 0 {
                let sign = if uniform() < 0.5 { -1.0 } else { 1.0 };
                a[count - 1] = sign * a1 * 10f64.powf(-20.0 + 7.0 * uniform()) * 1.5;
            }
            let got = match peak_off_half_pitch(&a) {
                Some(x) => {
                    assert!(
                        (0.0..=std::f64::consts::FRAC_PI_2).contains(&x),
                        "{a:?}: {x}"
                    );
                    torque_at(&a, x)
                }
                None => torque_at(&a, std::f64::consts::FRAC_PI_2),
            };
            let want = grid_maximum(&a, 2001);
            let scale: f64 = a.iter().map(|x| x.abs()).sum();
            if got < want - 4e-12 * scale {
                misses.push(format!("{a:?}: {got} < {want}"));
            }
        }
        assert!(
            misses.is_empty(),
            "{} misses, e.g. {:?}",
            misses.len(),
            &misses[..misses.len().min(5)]
        );
    }

    #[test]
    fn roots_in_unit_interval_finds_every_root_of_a_quintic() {
        // cos(11x) / cos(x) = P_11(u) with u = cos² x has five roots in (0, 1), at
        // x = (2k + 1) π / 22 for k = 0 .. 4.
        let p11 = [-11.0, 220.0, -1232.0, 2816.0, -2816.0, 1024.0];
        let roots = roots_in_unit_interval(&p11);
        let mut want: Vec<f64> = (0..5)
            .map(|k| (f64::from(2 * k + 1) * PI / 22.0).cos().powi(2))
            .collect();
        want.sort_by(f64::total_cmp);
        assert_eq!(roots.len(), 5, "{roots:?}");
        for (got, want) in roots.iter().zip(&want) {
            assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        }
    }

    #[test]
    fn roots_in_unit_interval_finds_a_close_pair_a_scan_would_miss() {
        // Decision 29's failure mode of a sampled scan: two roots inside one cell of a
        // 1,024-cell scan, with the same sign at both cell ends. The roots of the derivative
        // split the interval first, so each piece holds one root.
        let q = [0.5002 * 0.5006, -(0.5002 + 0.5006), 1.0]; // (u - 0.5002)(u - 0.5006)
        let value = |u: f64| q[0] + q[1] * u + q[2] * u * u;
        let (cell_lo, cell_hi) = (512.0 / 1024.0, 513.0 / 1024.0);
        assert!(
            value(cell_lo) > 0.0 && value(cell_hi) > 0.0,
            "same sign at the cell ends"
        );
        let roots = roots_in_unit_interval(&q);
        assert_eq!(roots.len(), 2, "{roots:?}");
        assert!((roots[0] - 0.5002).abs() < 1e-12, "{}", roots[0]);
        assert!((roots[1] - 0.5006).abs() < 1e-12, "{}", roots[1]);
    }

    #[test]
    fn roots_in_unit_interval_handles_the_edges() {
        let none = Vec::<f64>::new();
        assert_eq!(roots_in_unit_interval(&[]), none);
        assert_eq!(roots_in_unit_interval(&[1.0]), none, "a constant");
        assert_eq!(
            roots_in_unit_interval(&[0.0, 0.0]),
            none,
            "zero: no isolated root"
        );
        assert_eq!(roots_in_unit_interval(&[-0.25, 1.0]), [0.25]);
        assert_eq!(roots_in_unit_interval(&[0.0, 1.0]), [0.0], "a root at 0");
        assert_eq!(roots_in_unit_interval(&[-1.0, 1.0]), [1.0], "a root at 1");
        assert_eq!(
            roots_in_unit_interval(&[2.0, -1.0]),
            none,
            "root at 2: outside"
        );
        assert_eq!(
            roots_in_unit_interval(&[1.0, 0.0, 1.0]),
            none,
            "no real root"
        );
        assert_eq!(
            roots_in_unit_interval(&[0.25, -1.0, 1.0]),
            [0.5],
            "double root"
        );
        assert_eq!(
            roots_in_unit_interval(&[-0.25, 1.0, 0.0, 0.0]),
            [0.25],
            "trailing zeros dropped"
        );
    }
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib peak 2>&1 | grep -E "^error" | head -4
```

Expected: compile errors, first ``error[E0425]: cannot find value `ODD_HARMONICS` in this scope`` and ``error[E0308]: mismatched types`` (the slice calls against the array signature); further down (without `head`), ``error[E0425]: cannot find function `roots_in_unit_interval` in this scope``.

- [ ] **Step 3: Replace the closed form with the general search**

`stationary_polynomial` builds Q(u) = Σ n a_n P_n(u) from the recurrence P_{n+2} = 2 (2u − 1) P_n − P_{n−2} (exact integer coefficients); for 1, 3, 5 it is the closed form's quadratic. `roots_in_unit_interval` solves a linear polynomial directly and otherwise recurses on the derivative. The half-pitch torque and the torque fold keep the closed form's operand order ((a1 − a3) + a5; a left fold from 0), so the 1e-12 filter sees the same numbers.

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| E4 | Pole sweep!C6:C11 | a_i = MAX(w/(2 tan(pi/N)) + 0.05, bore/2 + key + 2.5) | + inner bondline in the wall term; the 6-pole row reads "outside OD envelope" | E4 |
| E5 | Temperature design!C121 (feeds C125, C130 and the thermal rows) | 1.035e-5 T²·m² (free-space field) | 4.14e-5 T²·m² (doubled at the steel surface); total slip loss 2.751 W; fields3d part is M3 | E5 |
| E6 | Calculator!C63 | OD/2 − block-back apothem | OD/2 − (block back + outer bondline): 2.723 mm | E6 |
| E7 | Calculator!C76, C82, C88 → C89-C96; sweeps N, Q, T, U; Calibration!C40-C42 | every harmonic at half a pole pitch | the maximum of the harmonic torque-angle curve (closed form); same at defaults; 6 poles 0.861 → 0.911 N·m | E7 |
| E8 | Calculator!C9, C57, C111; Metal design!C175 | flat-block corner geometry in arc mode | the corner radius C55 (face radius for arcs), round pocket; arc-mode pull-out 2.99 → 2.65 N·m | E8 |
| E9 | Calculator!C111, C113; Materials!C22 | steel cup and boss, back-iron wall advice even with no back iron | aluminium cup and boss, "No back iron"; total mass 156.9 → 96.8 g at backiron = 0 | E9 |
| E10 | Calculator!C103 → C104-C106; Materials!C20-C22 | (Br_i + Br_o)/2 · (t_i + t_o)/(t_i + t_o + g) | (Br_i t_i + Br_o t_o)/(t_i + t_o + g); same for identical rings | E10 |
```

with:

```markdown
| E4 | Pole sweep!C6:C11 | a_i = MAX(w/(2 tan(pi/N)) + 0.05, bore/2 + key + 2.5) | + inner bondline in the wall term; the 6-pole row reads "outside OD envelope" | E4 |
| E5 | Temperature design!C121 (feeds C125, C130 and the thermal rows) | 1.035e-5 T²·m² (free-space field) | 4.14e-5 T²·m² (doubled at the steel surface); total slip loss 2.751 W; fields3d part is M3 | E5 |
| E6 | Calculator!C63 | OD/2 − block-back apothem | OD/2 − (block back + outer bondline): 2.723 mm | E6 |
| E7 | Calculator!C76, C82, C88 → C89-C96; sweeps N, Q, T, U; Calibration!C40-C42 | every harmonic at half a pole pitch | the maximum of the harmonic torque-angle curve (`model::peak_off_half_pitch`: every root of dT/dx in cos² x, for any odd harmonic set up to 11, with no closed form and no scan grid; Addendum A decision 29); same at defaults; 6 poles 0.861 → 0.911 N·m | E7 |
| E8 | Calculator!C9, C57, C111; Metal design!C175 | flat-block corner geometry in arc mode | the corner radius C55 (face radius for arcs), round pocket; arc-mode pull-out 2.99 → 2.65 N·m | E8 |
| E9 | Calculator!C111, C113; Materials!C22 | steel cup and boss, back-iron wall advice even with no back iron | aluminium cup and boss, "No back iron"; total mass 156.9 → 96.8 g at backiron = 0 | E9 |
| E10 | Calculator!C103 → C104-C106; Materials!C20-C22 | (Br_i + Br_o)/2 · (t_i + t_o)/(t_i + t_o + g) | (Br_i t_i + Br_o t_o)/(t_i + t_o + g); same for identical rings | E10 |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust

    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    let amps = [amp_n(1), amp_n(3), amp_n(5)];
    let peak = peak_angle(amps, dev);
    let tau_n = |a: f64, n: u32| match peak {
        Some(x) => tau_at(a, n, x),
        None => a * (f64::from(n) * PI / 2.0).sin(), // the workbook expression, bit for bit
```

with:

```rust

    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    let amps = [amp_n(1), amp_n(3), amp_n(5)];
    let peak = peak_angle(&amps, dev);
    let tau_n = |a: f64, n: u32| match peak {
        Some(x) => tau_at(a, n, x),
        None => a * (f64::from(n) * PI / 2.0).sin(), // the workbook expression, bit for bit
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
//! remanence, via `resolve_magnets`, and the manual Br defaults C17 and C27),
//! E6 (Calculator!C63, the ring wall at the flats without the outer
//! bondline), E7 (pull-out at the maximum over angle,
//! [`peak_off_half_pitch`]; `peak_angle`, the one E7 gate, and `tau_at`, shared
//! with the sweeps and Calibration through `at_pull_out` or directly), E8
//! (arc mode: Calculator!C9 from the corner radius C55, the round pocket in
//! C111), E9 (no back iron: aluminium cup and boss in C111 and C113, as the
```

with:

```rust
//! remanence, via `resolve_magnets`, and the manual Br defaults C17 and C27),
//! E6 (Calculator!C63, the ring wall at the flats without the outer
//! bondline), E7 (pull-out at the maximum over angle,
//! [`peak_off_half_pitch`], one search for any odd harmonic set up to 11, decision 29 A;
//! `peak_angle`, the one E7 gate, and `tau_at`, shared
//! with the sweeps and Calibration through `at_pull_out` or directly), E8
//! (arc mode: Calculator!C9 from the corner radius C55, the round pocket in
//! C111), E9 (no back iron: aluminium cup and boss in C111 and C113, as the
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust

/// The odd space harmonics the model sums (the workbook's set).
pub const HARMONICS: [u32; 3] = [1, 3, 5];

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";
```

with:

```rust

/// The odd space harmonics the model sums (the workbook's set).
pub const HARMONICS: [u32; 3] = [1, 3, 5];

/// The odd space harmonics the E7 peak search handles, 1 to 11 (Addendum A3: "selectable
/// up to 11"); amplitude `a[i]` of [`peak_off_half_pitch`] belongs to `ODD_HARMONICS[i]`.
pub const ODD_HARMONICS: [u32; 6] = [1, 3, 5, 7, 9, 11];

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust

/// E7: the electrical angle of the true pull-out, when half a pole pitch is not it.
///
/// Torque against electrical angle x is T(x) = a1 sin x + a3 sin 3x + a5 sin 5x,
/// `a` the per-harmonic amplitudes. Half a pitch (x = π/2) is always a stationary
/// point, and the workbook evaluates every harmonic there. The other stationary
/// points solve dT/dx = 0; with c = cos x,
/// dT/dx = c·[(a1 − 9 a3 + 25 a5) + (12 a3 − 100 a5) c² + 80 a5 c⁴],
/// a quadratic in u = c². T is symmetric about π/2 (odd harmonics), so
/// x in [0, π/2] suffices. Returns `None` when half a pitch is the maximum, so
/// callers keep the workbook expression and default outputs stay bit-identical;
/// otherwise the angle whose torque exceeds the half-pitch torque by more than
/// 1e-12 relative. Valid for the harmonic set [1, 3, 5] only (`HARMONICS`); the
/// Addendum A plan generalizes it with the harmonic parameter (decision D7).
pub fn peak_off_half_pitch(a: [f64; 3]) -> Option<f64> {
    let [a1, a3, a5] = a;
    let torque = |x: f64| a1 * x.sin() + a3 * (3.0 * x).sin() + a5 * (5.0 * x).sin();
    let half_pitch = a1 - a3 + a5; // sin(π/2) = 1, sin(3π/2) = −1, sin(5π/2) = 1
    let (qa, qb, qc) = (80.0 * a5, 12.0 * a3 - 100.0 * a5, a1 - 9.0 * a3 + 25.0 * a5);
    let roots: Vec<f64> = if qa != 0.0 {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < 0.0 {
            Vec::new()
        } else {
            // Cancellation-free roots: q adds -qb and the root of the same sign, so a
            // vanishing a5 (|qa| << |qb|, e.g. a fill of exactly 0.4) keeps the small root,
            // qc / q -> -qc / qb, where (-qb ± √disc) / (2 qa) lost every digit.
            let q = -0.5 * (qb + disc.sqrt().copysign(qb));
            if q == 0.0 {
                vec![0.0] // qb = 0 and disc = 0, so qc = 0: the double root u = 0
            } else {
                vec![q / qa, qc / q]
            }
        }
    } else if qb != 0.0 {
        vec![-qc / qb]
    } else {
        Vec::new()
    };
    roots
        .into_iter()
        .filter(|u| (0.0..=1.0).contains(u))
        .map(|u| u.sqrt().acos())
        .map(|x| (x, torque(x)))
        .filter(|&(_, t)| t - half_pitch > 1e-12 * half_pitch.abs())
```

with:

```rust

/// E7: the electrical angle of the true pull-out, when half a pole pitch is not it.
///
/// Torque against electrical angle x is T(x) = Σ a_i sin((2i + 1) x): `a[i]` is the
/// amplitude of the odd harmonic 2i + 1 ([`ODD_HARMONICS`], at most six). Half a pitch
/// (x = π/2) is always a stationary point, and the workbook evaluates every harmonic there.
/// The other stationary points solve dT/dx = Σ n a_n cos(n x) = cos x · Q(u) = 0 with
/// u = cos² x, where Q is a polynomial of degree `a.len() − 1` ([`stationary_polynomial`]).
/// T is symmetric about π/2 (odd harmonics), so x in [0, π/2] (u in [0, 1]) suffices.
/// [`roots_in_unit_interval`] finds every root of Q there with no closed form, so a vanishing
/// top amplitude (a ring at a fill of exactly 0.4 zeroes the fifth) cannot cancel digits, and
/// with no scan grid, so no pair of roots can hide inside a cell (decision 29 A). Returns
/// `None` when half a pitch is the maximum, so callers keep the workbook expression and
/// default outputs stay bit-identical; otherwise the angle whose torque exceeds the half-pitch
/// torque by more than 1e-12 relative.
pub fn peak_off_half_pitch(a: &[f64]) -> Option<f64> {
    let torque = |x: f64| {
        a.iter()
            .zip(ODD_HARMONICS)
            .fold(0.0, |acc, (&an, n)| acc + an * (f64::from(n) * x).sin())
    };
    // sin(nπ/2) = 1, −1, 1, ... for n = 1, 3, 5, ...: a1 − a3 + a5 − ...
    let half_pitch = a.iter().enumerate().fold(
        0.0,
        |acc, (i, &an)| if i % 2 == 0 { acc + an } else { acc - an },
    );
    roots_in_unit_interval(&stationary_polynomial(a))
        .into_iter()
        .map(|u| u.sqrt().acos())
        .map(|x| (x, torque(x)))
        .filter(|&(_, t)| t - half_pitch > 1e-12 * half_pitch.abs())
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        .map(|(x, _)| x)
}

/// The E7 gate every harmonic sum shares (the pull-out, the two circuit sums
/// C95 and C96, the sweep rows, Calibration C40-C42): the angle of the true
/// pull-out when E7 is on and half a pitch is not the maximum of the curve with
/// amplitudes `a` ([`peak_off_half_pitch`]). `None` means: keep the workbook's
/// half-pitch expression, bit for bit.
pub(crate) fn peak_angle(a: [f64; 3], dev: Deviations) -> Option<f64> {
    if dev.is_on(DeviationId::E7) {
        peak_off_half_pitch(a)
    } else {
```

with:

```rust
        .map(|(x, _)| x)
}

/// Q(u), the coefficients (Q = q[0] + q[1] u + ...) of dT/dx / cos x for
/// T(x) = Σ a_i sin((2i + 1) x): Q = Σ n a_n P_n(u), where cos(n x) = cos x · P_n(cos² x)
/// and P_{n+2} = 2 (2u − 1) P_n − P_{n−2}, P_1 = P_{−1} = 1 (so P_3 = 4u − 3,
/// P_5 = 16u² − 20u + 5). For 1, 3, 5 that is (a1 − 9 a3 + 25 a5) + (12 a3 − 100 a5) u + 80 a5 u².
fn stationary_polynomial(a: &[f64]) -> Vec<f64> {
    let mut q = vec![0.0; a.len()];
    let (mut p_before, mut p) = (vec![1.0], vec![1.0]); // P_{n−2} and P_n, from n = 1
    for (&an, n) in a.iter().zip(ODD_HARMONICS) {
        for (qk, &pk) in q.iter_mut().zip(&p) {
            *qk += f64::from(n) * an * pk;
        }
        let mut next = vec![0.0; p.len() + 1]; // P_{n+2} = 4u P_n − 2 P_n − P_{n−2}
        for (k, &pk) in p.iter().enumerate() {
            next[k + 1] += 4.0 * pk;
            next[k] -= 2.0 * pk;
        }
        for (k, &pk) in p_before.iter().enumerate() {
            next[k] -= pk;
        }
        p_before = std::mem::replace(&mut p, next);
    }
    q
}

/// Every real root in [0, 1] of the polynomial c[0] + c[1] u + c[2] u² + ..., ascending.
///
/// A linear polynomial's root is −c[0] / c[1]. Above degree 1, the roots of the derivative
/// (found the same way) split [0, 1] into intervals on which the polynomial is monotone, so
/// each interval holds at most one root and bisection finds it to the last bit: no root pair
/// can hide between samples, the failure mode of a sampled scan (decision 29 A's Q' guard,
/// applied at every level).
/// Trailing zero coefficients are dropped; a constant (zero included) has no isolated root.
/// A double root is found only where the polynomial is exactly 0 there; that is an inflection
/// of the torque curve, never its peak.
fn roots_in_unit_interval(c: &[f64]) -> Vec<f64> {
    let degree = match c.iter().rposition(|&ck| ck != 0.0) {
        Some(d) if d > 0 => d,
        _ => return Vec::new(),
    };
    let c = &c[..=degree];
    if degree == 1 {
        let root = -c[0] / c[1];
        return if (0.0..=1.0).contains(&root) {
            vec![root]
        } else {
            Vec::new()
        };
    }
    let value = |u: f64| c.iter().rev().fold(0.0, |acc, &ck| acc * u + ck);
    let derivative: Vec<f64> = c
        .iter()
        .enumerate()
        .skip(1)
        .map(|(k, &ck)| k as f64 * ck)
        .collect();
    let mut breaks = vec![0.0];
    breaks.extend(
        roots_in_unit_interval(&derivative)
            .into_iter()
            .filter(|&u| u > 0.0 && u < 1.0),
    );
    breaks.push(1.0);
    let mut roots = Vec::new();
    for w in breaks.windows(2) {
        let (mut lo, mut hi) = (w[0], w[1]);
        let (f_lo, f_hi) = (value(lo), value(hi));
        if f_lo == 0.0 {
            roots.push(lo);
            continue;
        }
        // A root at `hi` is found as the next interval's `lo` (or at 1 below).
        if f_hi == 0.0 || (f_lo < 0.0) == (f_hi < 0.0) {
            continue;
        }
        let lo_negative = f_lo < 0.0;
        let mut root = None;
        for _ in 0..128 {
            let mid = lo + (hi - lo) / 2.0;
            if mid <= lo || mid >= hi {
                break; // lo and hi are adjacent doubles
            }
            let f_mid = value(mid);
            if f_mid == 0.0 {
                root = Some(mid);
                break;
            }
            if (f_mid < 0.0) == lo_negative {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        roots.push(root.unwrap_or(if value(lo).abs() <= value(hi).abs() {
            lo
        } else {
            hi
        }));
    }
    if value(1.0) == 0.0 {
        roots.push(1.0);
    }
    roots
}
/// The E7 gate every harmonic sum shares (the pull-out, the two circuit sums
/// C95 and C96, the sweep rows, Calibration C40-C42): the angle of the true
/// pull-out when E7 is on and half a pitch is not the maximum of the curve with
/// amplitudes `a` ([`peak_off_half_pitch`]). `None` means: keep the workbook's
/// half-pitch expression, bit for bit.
pub(crate) fn peak_angle(a: &[f64], dev: Deviations) -> Option<f64> {
    if dev.is_on(DeviationId::E7) {
        peak_off_half_pitch(a)
    } else {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
) -> [Harmonic; 3] {
    let amplitude = |x: &Harmonic| x.bi * x.bo / (2.0 * mu0) * x.s(backiron);
    let mut h_pull = h;
    if let Some(x) = peak_angle(h.map(|hn| amplitude(&hn)), dev) {
        for hn in h_pull.iter_mut() {
            hn.tau = tau_at(amplitude(hn), hn.n, x);
        }
```

with:

```rust
) -> [Harmonic; 3] {
    let amplitude = |x: &Harmonic| x.bi * x.bo / (2.0 * mu0) * x.s(backiron);
    let mut h_pull = h;
    if let Some(x) = peak_angle(&h.map(|hn| amplitude(&hn)), dev) {
        for hn in h_pull.iter_mut() {
            hn.tau = tau_at(amplitude(hn), hn.n, x);
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    // E7: each circuit at the maximum of its own torque-angle curve.
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients = h.map(|x| x.bi * x.bo * s(&x));
        match peak_angle(coefficients, dev) {
            Some(x) => h
                .iter()
                .zip(coefficients)
```

with:

```rust
    // E7: each circuit at the maximum of its own torque-angle curve.
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients = h.map(|x| x.bi * x.bo * s(&x));
        match peak_angle(&coefficients, dev) {
            Some(x) => h
                .iter()
                .zip(coefficients)
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib peak 2>&1 | grep "test result"
```

Expected: `test result: ok. 6 passed` (the unit tests whose names contain `peak`, the three E7 cancellation tests among them).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib the_general_search 2>&1 | grep "test result"
```

Expected: `test result: ok. 2 passed` (the closed-form agreement and the harmonic-11 brute force).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib roots_in_unit_interval 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test deviations e7 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed` (`e7_pull_out_over_angle_matches_the_report`, `e7_finds_the_peak_at_a_fill_of_exactly_0_4` and `e7_leaves_every_default_cell_bit_for_bit`; E7's probes run in `each_probe_shows_its_correction`, in the full run below).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `112 passed` (107 + 5 new), the others as in Task 0.

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task1.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task1.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/calibration.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): one general E7 peak search for any odd harmonic set (decision 29)

dT/dx = cos x Q(cos^2 x): every root of Q in [0, 1] by recursion on derivatives
(the Q' root-pair guard at every level), no closed form and no scan grid.
The 1, 3, 5 closed form stays in the tests as the reference (agreement within
1e-12 rad on 3,000 triples); a brute-force test covers up to harmonic 11.
Every caller still passes the workbook set: defaults bit-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 2: The harmonic set as a Rust-only assumption (`coupling.max_harmonic`, up to 11)

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; the torque model's harmonic sums). Physics reviewer: spec A3 ("harmonics included (workbook 1, 3, 5; selectable up to 11)"), report section 6.5 ("Calibration's own harmonic formula ... moves by 2.3e-16 to 3.5e-16 if it is routed through `shear_stress`"), and this plan's decision A2-1.

Spec A3: "harmonics included (workbook 1, 3, 5; selectable up to 11)", and "assumptions that are hard-coded in the Python engine
today (for example the harmonic list ...) become engine parameters ... with defaults equal to the workbook's". The set is always
contiguous (1, 3, ... up to the chosen harmonic, decision A2-1), so a selector of the highest harmonic states it exactly, and
`peak_off_half_pitch` takes the set's amplitudes in order. The Calculator's pull-out, its two circuit sums (C95, C96), every sweep row
and the Calibration prototype sum the same set (decision A2-1: the one-point correction must compare like with like). The per-harmonic
cells of 1, 3 and 5 keep their terms (wave number, amplitudes, geometry factors do not depend on the set); a harmonic left out reads 0
shear stress, so the total is always the sum of the cells, and 7 to 11 add Rust-only cells (the sweeps' column U includes them: a
table has no Rust-only columns). The Calibration keeps its own operand order (report 6.5: routing it through `shear_stress` would move
C40 to C42 by 2.3e-16 to 3.5e-16); its `(t1 + t3 + t5)` is the same left fold `harmonic_sum` does. At the default (5) every sum runs
over the same three terms in the same order: parity, the differential data and every registry test are untouched. A code outside the
choices, set on the struct, gives NaN sums (decision D3), never another set.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs` (`WORKBOOK_MAX_HARMONIC`, `MAX_HARMONIC_CHOICES`, `harmonic_count`, `harmonic_sum`, `harmonic_slot`; the input; `tau7_Pa` to `tau11_Pa`; `shear_stress` over six harmonics; `at_pull_out` on a slice; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs` (`compute(c, max_harmonic, dev)`; Rust-only τ7 to τ11; test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs` (`SweepContext::max_harmonic`; test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs` (pass the set to the Calibration and the sweeps)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/schema.rs` (an assumption may be a selector)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs` (an invalid set end to end)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/data/input_schema.json` (the new Rust-only input)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (model row; invalid inputs)

**Interfaces:**
- Consumes: Task 1's `ODD_HARMONICS`, `peak_angle(a: &[f64], dev)`, `peak_off_half_pitch(a: &[f64])`; `model::shear_stress(..) -> [Harmonic; 3]`; `model::at_pull_out(h: [Harmonic; 3], backiron, mu0, dev) -> [Harmonic; 3]`; `calibration::compute(c: &CalibrationInputs, dev: Deviations)`.
- Produces:
  - input `coupling.max_harmonic: i64 = WORKBOOK_MAX_HARMONIC` (Rust-only selector, `.assumption()`; choices `MAX_HARMONIC_CHOICES`, codes 1, 3, 5, 7, 9, 11);
  - `pub const WORKBOOK_MAX_HARMONIC: i64 = 5`, `pub const MAX_HARMONIC_CHOICES: [(i64, &str); 6]`;
  - `pub fn harmonic_count(max_harmonic: i64) -> Option<usize>` (1 to 6, `None` outside the choices);
  - `pub(crate) fn harmonic_sum(count: Option<usize>, terms: impl Iterator<Item = f64>) -> f64` (left fold from 0; NaN for `None`) and `pub(crate) fn harmonic_slot(count: Option<usize>, terms: &[f64], i: usize) -> f64` (term, 0 when left out, NaN for `None`);
  - `shear_stress(..) -> [Harmonic; 6]` (every harmonic of `ODD_HARMONICS`); `at_pull_out(h: &[Harmonic], ..) -> Vec<Harmonic>`;
  - Rust-only results `model.tau7_Pa`, `tau9_Pa`, `tau11_Pa` and `calibration.tau7_Pa`, `tau9_Pa`, `tau11_Pa`;
  - `calibration::compute(c: &CalibrationInputs, max_harmonic: i64, dev: Deviations) -> CalibrationResults`; `sweeps::SweepContext::max_harmonic: i64`.

- [ ] **Step 1: Write the failing tests**

Unit tests in `model.rs` (the choices map, adding and dropping terms, NaN for an invalid code, E7 on an 11-harmonic curve against the brute-force maximum from Task 1's `grid_maximum`), `calibration.rs` and `sweeps.rs` (the set reaches the prototype and every row); `tests/robustness.rs` checks an invalid code end to end and `validate()`; `tests/schema.rs` lets an assumption be a selector.

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_design_reproduces_the_bench_correction() {
```

with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::WORKBOOK_MAX_HARMONIC;

    #[test]
    fn default_design_reproduces_the_bench_correction() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
            br_T: 1.29,
            ..CalibrationInputs::default()
        };
        let r = compute(&workbook, Deviations::NONE);
        assert!((r.model_torque_Nm - 1.6044531397852).abs() < 1e-12);
        assert!((r.f_cal_updated - 1.06578369763353).abs() < 1e-12);
        assert_eq!(r.poles_per_ring, 10.0);
```

with:

```rust
            br_T: 1.29,
            ..CalibrationInputs::default()
        };
        let r = compute(&workbook, WORKBOOK_MAX_HARMONIC, Deviations::NONE);
        assert!((r.model_torque_Nm - 1.6044531397852).abs() < 1e-12);
        assert!((r.f_cal_updated - 1.06578369763353).abs() < 1e-12);
        assert_eq!(r.poles_per_ring, 10.0);
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
    }

    #[test]
    fn corner_definition_uses_the_spacing_as_the_corner_gap() {
        let c = CalibrationInputs {
            gap_definition: 0,
            ..CalibrationInputs::default()
        };
        let r = compute(&c, Deviations::ALL);
        assert_eq!(r.corner_gap_mm, c.spacing_mm);
        assert!(
            r.flat_gap_mm > c.spacing_mm,
```

with:

```rust
    }

    #[test]
    fn the_harmonic_set_reaches_the_prototype_model() {
        // Addendum A3: the prototype's model sums the same harmonics as the Calculator, so the
        // one-point correction compares like with like. A left-out harmonic reads 0.
        let c = CalibrationInputs::default();
        let (one, workbook, eleven) = (
            compute(&c, 1, Deviations::NONE),
            compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::NONE),
            compute(&c, 11, Deviations::NONE),
        );
        assert_eq!((one.tau3_Pa, one.tau5_Pa, one.tau7_Pa), (0.0, 0.0, 0.0));
        assert_eq!(one.tau1_Pa, workbook.tau1_Pa);
        assert_eq!(
            (workbook.tau7_Pa, workbook.tau9_Pa, workbook.tau11_Pa),
            (0.0, 0.0, 0.0)
        );
        assert!(eleven.tau7_Pa != 0.0 && eleven.tau11_Pa != 0.0);
        let sum = [
            eleven.tau1_Pa,
            eleven.tau3_Pa,
            eleven.tau5_Pa,
            eleven.tau7_Pa,
            eleven.tau9_Pa,
            eleven.tau11_Pa,
        ]
        .iter()
        .fold(0.0, |acc, t| acc + t);
        assert_eq!(
            eleven.torque_2d_Nm,
            sum * 2.0 * PI * (eleven.gap_radius_mm / 1000.0).powi(2) * (c.magnet_length_mm / 1000.0)
        );
        assert!(eleven.model_torque_Nm != workbook.model_torque_Nm);
        // A code outside the choices: NaN, never another set (decision D3).
        let invalid = compute(&c, 4, Deviations::NONE);
        assert!(invalid.model_torque_Nm.is_nan() && invalid.tau1_Pa.is_nan());
    }

    #[test]
    fn corner_definition_uses_the_spacing_as_the_corner_gap() {
        let c = CalibrationInputs {
            gap_definition: 0,
            ..CalibrationInputs::default()
        };
        let r = compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::ALL);
        assert_eq!(r.corner_gap_mm, c.spacing_mm);
        assert!(
            r.flat_gap_mm > c.spacing_mm,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
                spacing_mm: spacing,
                ..CalibrationInputs::default()
            };
            let r = compute(&c, Deviations::ALL);
            match (inside, r.fea_interp_Nm, r.fea_interp_error) {
                (true, NumOrText::Num(_), NumOrText::Num(_)) => {}
                (false, NumOrText::Text(OUTSIDE_RANGE), NumOrText::Text(NOT_APPLICABLE)) => {}
```

with:

```rust
                spacing_mm: spacing,
                ..CalibrationInputs::default()
            };
            let r = compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::ALL);
            match (inside, r.fea_interp_Nm, r.fea_interp_error) {
                (true, NumOrText::Num(_), NumOrText::Num(_)) => {}
                (false, NumOrText::Text(OUTSIDE_RANGE), NumOrText::Text(NOT_APPLICABLE)) => {}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
                spacing_mm: spacing,
                ..CalibrationInputs::default()
            };
            compute(&c, Deviations::ALL).fea_interp_Nm
        };
        assert_eq!(at(1.0), NumOrText::Num(2.06));
        assert_eq!(at(1.5), NumOrText::Num(1.7));
```

with:

```rust
                spacing_mm: spacing,
                ..CalibrationInputs::default()
            };
            compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::ALL).fea_interp_Nm
        };
        assert_eq!(at(1.0), NumOrText::Num(2.06));
        assert_eq!(at(1.5), NumOrText::Num(1.7));
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
            total_magnets: 40,
            ..CalibrationInputs::default()
        };
        let r = compute(&c, Deviations::ALL);
        assert_eq!((r.fill_inner, r.fill_outer), (1.0, 1.0));
    }
}
```

with:

```rust
            total_magnets: 40,
            ..CalibrationInputs::default()
        };
        let r = compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::ALL);
        assert_eq!((r.fill_inner, r.fill_outer), (1.0, 1.0));
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    }

    #[test]
    fn calibration_factor_needs_every_prototype_condition() {
        let f = |backiron, npole, poles| {
            select_calibration_factor(backiron, npole, "B842SH", "B842SH", poles, 1.0658, 0.95)
```

with:

```rust
    }

    #[test]
    fn harmonic_count_maps_each_choice_and_nothing_else() {
        for (i, &(code, _)) in MAX_HARMONIC_CHOICES.iter().enumerate() {
            assert_eq!(code, i64::from(ODD_HARMONICS[i]));
            assert_eq!(harmonic_count(code), Some(i + 1), "{code}");
        }
        assert_eq!(harmonic_count(WORKBOOK_MAX_HARMONIC), Some(HARMONICS.len()));
        assert_eq!(CouplingInputs::default().max_harmonic, WORKBOOK_MAX_HARMONIC);
        for code in [0, 2, 4, 6, 12, 13, -1, i64::MIN, i64::MAX] {
            assert_eq!(harmonic_count(code), None, "{code}");
        }
    }

    #[test]
    fn the_harmonic_set_adds_or_drops_terms() {
        // Addendum A3: the model sums 1, 3, ... up to the chosen harmonic. The terms of 1, 3
        // and 5 (wave number, amplitudes, geometry factors) do not depend on the set; a left-out
        // harmonic's shear stress reads 0, so the total is always the sum of the six cells.
        let with = |max_harmonic| {
            at(&CouplingInputs {
                max_harmonic,
                ..CouplingInputs::default()
            })
        };
        let (one, workbook, eleven) = (with(1), with(5), with(11));
        assert_eq!(workbook, at(&CouplingInputs::default()));
        assert_eq!((one.tau3_Pa, one.tau5_Pa), (0.0, 0.0));
        assert_eq!(one.tau_Pa, one.tau1_Pa);
        assert_eq!(
            (workbook.tau7_Pa, workbook.tau9_Pa, workbook.tau11_Pa),
            (0.0, 0.0, 0.0)
        );
        for r in [&one, &eleven] {
            assert_eq!(
                (r.k3, r.b_i5, r.s5_iron, r.s1_free),
                (workbook.k3, workbook.b_i5, workbook.s5_iron, workbook.s1_free)
            );
        }
        assert!(eleven.tau7_Pa != 0.0 && eleven.tau9_Pa != 0.0 && eleven.tau11_Pa != 0.0);
        let cells = [
            eleven.tau1_Pa,
            eleven.tau3_Pa,
            eleven.tau5_Pa,
            eleven.tau7_Pa,
            eleven.tau9_Pa,
            eleven.tau11_Pa,
        ];
        assert_eq!(eleven.tau_Pa, cells.iter().fold(0.0, |acc, t| acc + t));
        assert!(close(
            eleven.pullout_Nm / workbook.pullout_Nm,
            eleven.tau_Pa / workbook.tau_Pa
        ));
        // The steel circuit sum (C95) is the pull-out's circuit here (both factors 0.95).
        for r in [&one, &workbook, &eleven] {
            assert!(close(r.pullout_iron_Nm, r.pullout_Nm), "{}", r.tau_Pa);
        }
    }

    #[test]
    fn an_invalid_harmonic_set_gives_nan_not_another_set() {
        // Decision D3: a code outside the choices, set on the struct, never selects another set.
        let r = at(&CouplingInputs {
            max_harmonic: 4,
            ..CouplingInputs::default()
        });
        for (what, x) in [
            ("tau", r.tau_Pa),
            ("tau1", r.tau1_Pa),
            ("tau5", r.tau5_Pa),
            ("tau11", r.tau11_Pa),
            ("pull-out", r.pullout_Nm),
            ("steel circuit", r.pullout_iron_Nm),
            ("free-space circuit", r.pullout_noiron_Nm),
        ] {
            assert!(x.is_nan(), "{what}: {x}");
        }
        assert!(r.k1.is_finite() && r.b_i1.is_finite() && r.s1_iron.is_finite());
    }

    #[test]
    fn e7_finds_the_peak_of_an_eleven_harmonic_curve() {
        // Decision 29 A through the model: 6 poles in free space (the layout of E7's probe),
        // every harmonic up to 11. E7's pull-out is the maximum of the model's own curve built
        // from the half-pitch terms (tau_n = a_n sin(n pi/2), so a_n = +-tau_n), and the
        // free-space circuit sum (C96) finds the same peak.
        let ci = CouplingInputs {
            npole: 6,
            backiron: 0,
            max_harmonic: 11,
            ..CouplingInputs::default()
        };
        let off = at(&ci);
        let on = at_with(&ci, 0.05, Deviations::only(DeviationId::E7));
        let half_pitch = [
            off.tau1_Pa,
            off.tau3_Pa,
            off.tau5_Pa,
            off.tau7_Pa,
            off.tau9_Pa,
            off.tau11_Pa,
        ];
        let a: Vec<f64> = half_pitch
            .iter()
            .enumerate()
            .map(|(i, &t)| if i % 2 == 0 { t } else { -t })
            .collect();
        let peak = grid_maximum(&a, 20001);
        assert!(on.tau_Pa > off.tau_Pa, "{} vs {}", on.tau_Pa, off.tau_Pa);
        assert!(close(on.tau_Pa, peak), "{} vs {peak}", on.tau_Pa);
        assert!(close(on.pullout_noiron_Nm, on.pullout_Nm));
    }

    #[test]
    fn calibration_factor_needs_every_prototype_condition() {
        let f = |backiron, npole, poles| {
            select_calibration_factor(backiron, npole, "B842SH", "B842SH", poles, 1.0658, 0.95)
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The default design's sweep context (`api::compute` at the workbook defaults).
    fn ctx() -> SweepContext {
```

with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::model::WORKBOOK_MAX_HARMONIC;

    /// The default design's sweep context (`api::compute` at the workbook defaults).
    fn ctx() -> SweepContext {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
            gear_eff: 0.95,
            required_floor_Nm: 2.5,
            max_diameter_mm: 43.0,
        }
    }

    #[test]
```

with:

```rust
            gear_eff: 0.95,
            required_floor_Nm: 2.5,
            max_diameter_mm: 43.0,
            max_harmonic: WORKBOOK_MAX_HARMONIC,
        }
    }

    #[test]
    fn the_harmonic_set_reaches_every_row() {
        // Addendum A3: a row sums the harmonics the Calculator sums; a left-out one reads 0,
        // and a code outside the choices gives NaN (decision D3).
        let run = |max_harmonic| {
            let c = SweepContext {
                max_harmonic,
                ..ctx()
            };
            row(&c, 1.25, 10, 10.15, 1.25, 0.95, Deviations::NONE)
        };
        let (one, workbook, eleven) = (run(1), run(WORKBOOK_MAX_HARMONIC), run(11));
        assert_eq!((one.tau3_Pa, one.tau5_Pa), (0.0, 0.0));
        assert_eq!(one.tau_Pa, one.tau1_Pa);
        assert_eq!(
            workbook.tau_Pa,
            0.0 + workbook.tau1_Pa + workbook.tau3_Pa + workbook.tau5_Pa
        );
        assert!(eleven.tau_Pa != workbook.tau_Pa);
        assert_eq!(
            (eleven.tau1_Pa, eleven.s3, eleven.k5),
            (workbook.tau1_Pa, workbook.s3, workbook.k5)
        );
        let invalid = run(4);
        assert!(invalid.tau_Pa.is_nan() && invalid.pullout_op_Nm.is_nan());
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs`, replace:

```rust
}

#[test]
fn an_invalid_coercivity_source_uses_the_inputs() {
    // A Rust-only selector set on the struct (bypassing set()): any code but 1 means the Hcj
    // and beta inputs for both rings, as the catch-all else of a two-way IF (Global
```

with:

```rust
}

#[test]
fn an_invalid_harmonic_set_is_nan_not_another_set() {
    // A Rust-only selector set on the struct (bypassing set()): the Calculator, the
    // Calibration prototype and every sweep row read NaN, never another harmonic set, with
    // the corrections off and on; validate() names the path.
    for dev in [Deviations::NONE, Deviations::ALL] {
        let mut inputs = DesignInputs::defaults_with(dev);
        inputs.coupling.max_harmonic = 4;
        let res = compute_all_with(&inputs, dev);
        assert!(res.model.pullout_Nm.is_nan() && res.metal.torque_hot_low_Nm.is_nan());
        assert!(res.calibration.model_torque_Nm.is_nan());
        assert!(res.gap_sweep.iter().all(|r| r.tau_Pa.is_nan()));
        assert!(res.pole_sweep.iter().all(|r| r.pullout_op_Nm.is_nan()));
        let errors = inputs.validate().expect_err("an invalid code");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].path, "coupling.max_harmonic");
        assert_eq!(errors[0].kind, SetErrorKind::NotAChoice { code: 4 });
    }
}

#[test]
fn an_invalid_coercivity_source_uses_the_inputs() {
    // A Rust-only selector set on the struct (bypassing set()): any code but 1 means the Hcj
    // and beta inputs for both rings, as the catch-all else of a two-way IF (Global
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/schema.rs`, replace:

```rust
}

#[test]
fn assumptions_are_numeric_inputs_with_sliders() {
    for row in input_rows(&DesignInputs::default())
        .iter()
        .filter(|r| r.meta.assumption)
    {
        assert!(
            matches!(row.meta.ty, FieldType::F64 | FieldType::I64) && row.meta.range.is_some(),
            "{}: an assumption is a numeric input with a range",
            row.path
        );
    }
```

with:

```rust
}

#[test]
fn assumptions_are_numeric_inputs_with_sliders_or_selectors() {
    // Addendum A3: every assumption but the harmonic set is a number with a slider; the
    // harmonic set (`coupling.max_harmonic`) is a selector of odd harmonics.
    for row in input_rows(&DesignInputs::default())
        .iter()
        .filter(|r| r.meta.assumption)
    {
        let slider =
            matches!(row.meta.ty, FieldType::F64 | FieldType::I64) && row.meta.range.is_some();
        let selector = row.meta.ty == FieldType::I64 && !row.meta.choices.is_empty();
        assert!(
            slider || selector,
            "{}: an assumption is a numeric input with a range, or a selector",
            row.path
        );
    }
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib harmonic 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors such as ``error[E0432]: unresolved import `crate::engine::model::WORKBOOK_MAX_HARMONIC` `` and ``error[E0560]: struct `CouplingInputs` has no field named `max_harmonic` ``.

- [ ] **Step 3: Add the input, the helpers and the set's sums**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
be inf or NaN. Call `DesignInputs::validate()` where inputs enter: it returns
every offending path, in schema order, with the reason `set` would give. The
Rust-only selectors follow the same rule: a material code outside its choices
gives NaN properties and the name `"#N/A"`, never another material.

## Differences from the workbook
```

with:

```markdown
be inf or NaN. Call `DesignInputs::validate()` where inputs enter: it returns
every offending path, in schema order, with the reason `set` would give. The
Rust-only selectors follow the same rule: a material code outside its choices
gives NaN properties and the name `"#N/A"`, never another material, and a harmonic
set outside its choices (`coupling.max_harmonic`) gives NaN torques, never another set.

## Differences from the workbook
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6), `ModelResults` (73 cells, plus the Rust-only `inner_grade` and `outer_grade`), the fixed harmonic set `HARMONICS` (1, 3, 5), and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
```

with:

```markdown
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade` and `tau7_Pa` to `tau11_Pa`), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
    ti.slip_loss.sigma_316_S_m = parts.sleeve_liner.props.sigma_S_m;
    ti.thermal.c_316 = parts.sleeve_liner.props.cp_J_kgK;
    ti.thermal.c_aluminium = parts.cap.props.cp_J_kgK;
    let cal = calibration::compute(cal_in, dev);
    let f_cal = model::select_calibration_factor(
        ci.backiron,
        ci.npole,
```

with:

```rust
    ti.slip_loss.sigma_316_S_m = parts.sleeve_liner.props.sigma_S_m;
    ti.thermal.c_316 = parts.sleeve_liner.props.cp_J_kgK;
    ti.thermal.c_aluminium = parts.cap.props.cp_J_kgK;
    let cal = calibration::compute(cal_in, ci.max_harmonic, dev);
    let f_cal = model::select_calibration_factor(
        ci.backiron,
        ci.npole,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        gear_eff: ci.gear_efficiency,
        required_floor_Nm: m.required_floor_Nm,
        max_diameter_mm: md.max_diameter_mm,
    };
    let gap = sweeps::gap_sweep(&ctx, ci.npole, ci.inner_back_apothem_mm, f_cal, dev);
    let pole = sweeps::pole_sweep(
```

with:

```rust
        gear_eff: ci.gear_efficiency,
        required_floor_Nm: m.required_floor_Nm,
        max_diameter_mm: md.max_diameter_mm,
        max_harmonic: ci.max_harmonic,
    };
    let gap = sweeps::gap_sweep(&ctx, ci.npole, ci.inner_back_apothem_mm, f_cal, dev);
    let pole = sweeps::pole_sweep(
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
//!   follows the corrected N42SH remanence, 1.30 T), and so is E7 (the τ_n
//!   harmonic sum, C40-C42, at the maximum over angle), see
//!   [`crate::engine::deviations::REGISTRY`].

use std::f64::consts::PI;

use super::compat::py_min;
use super::constants::MU0;
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, param, results};
use super::model::{br_factor, corner_radius, peak_angle, tau_at};

inputs! {
    /// Prototype inputs (Calibration!C5:C25, C47:C48).
```

with:

```rust
//!   follows the corrected N42SH remanence, 1.30 T), and so is E7 (the τ_n
//!   harmonic sum, C40-C42, at the maximum over angle), see
//!   [`crate::engine::deviations::REGISTRY`].
//! - Addendum A3: the prototype sums the Calculator's harmonic set
//!   (`coupling.max_harmonic`, passed in), with the Rust-only τ7 to τ11.

use std::f64::consts::PI;

use super::compat::py_min;
use super::constants::MU0;
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, out_rust_only, param, results};
use super::model::{
    ODD_HARMONICS, br_factor, corner_radius, harmonic_count, harmonic_slot, harmonic_sum,
    peak_angle, tau_at,
};

inputs! {
    /// Prototype inputs (Calibration!C5:C25, C47:C48).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
            tau1_Pa: f64 => out("Pa", "Shear stress, harmonic 1", "", "Calibration!C40"),
            tau3_Pa: f64 => out("Pa", "Shear stress, harmonic 3", "", "Calibration!C41"),
            tau5_Pa: f64 => out("Pa", "Shear stress, harmonic 5", "", "Calibration!C42"),
            torque_2d_Nm: f64 => out("N·m", "2D torque before end and calibration factors", "", "Calibration!C43"),
            original_model_Nm: f64 => out("N·m", "Original model pull-out torque",
                "Same value as model_torque_Nm.", "Calibration!C44"),
```

with:

```rust
            tau1_Pa: f64 => out("Pa", "Shear stress, harmonic 1", "", "Calibration!C40"),
            tau3_Pa: f64 => out("Pa", "Shear stress, harmonic 3", "", "Calibration!C41"),
            tau5_Pa: f64 => out("Pa", "Shear stress, harmonic 5", "", "Calibration!C42"),
            tau7_Pa: f64 => out_rust_only("Pa", "Shear stress, harmonic 7",
                "Addendum A3: summed when the highest harmonic (coupling.max_harmonic) is 7 or more; 0 otherwise."),
            tau9_Pa: f64 => out_rust_only("Pa", "Shear stress, harmonic 9",
                "Summed when the highest harmonic is 9 or more; 0 otherwise."),
            tau11_Pa: f64 => out_rust_only("Pa", "Shear stress, harmonic 11",
                "Summed when the highest harmonic is 11; 0 otherwise."),
            torque_2d_Nm: f64 => out("N·m", "2D torque before end and calibration factors", "", "Calibration!C43"),
            original_model_Nm: f64 => out("N·m", "Original model pull-out torque",
                "Same value as model_torque_Nm.", "Calibration!C44"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
pub const NOT_APPLICABLE: &str = "n.a.";

/// The Calibration sheet. Line by line the Python `calibration.compute`, with
/// E7 (the τ_n harmonic sum at the maximum over angle) when `dev` has it on.
pub fn compute(c: &CalibrationInputs, dev: Deviations) -> CalibrationResults {
    let poles = c.total_magnets as f64 / 2.0;
    let r_face = c.apothem_mm + c.magnet_thickness_mm;
    let r_corner = corner_radius(r_face, c.magnet_width_mm);
```

with:

```rust
pub const NOT_APPLICABLE: &str = "n.a.";

/// The Calibration sheet. Line by line the Python `calibration.compute`, with
/// E7 (the τ_n harmonic sum at the maximum over angle) when `dev` has it on, over
/// the Calculator's harmonic set `max_harmonic` (`coupling.max_harmonic`, Addendum A3).
pub fn compute(c: &CalibrationInputs, max_harmonic: i64, dev: Deviations) -> CalibrationResults {
    let poles = c.total_magnets as f64 / 2.0;
    let r_face = c.apothem_mm + c.magnet_thickness_mm;
    let r_corner = corner_radius(r_face, c.magnet_width_mm);
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
            / 2.0
    };

    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    let amps = [amp_n(1), amp_n(3), amp_n(5)];
    let peak = peak_angle(&amps, dev);
    let tau_n = |a: f64, n: u32| match peak {
        Some(x) => tau_at(a, n, x),
        None => a * (f64::from(n) * PI / 2.0).sin(), // the workbook expression, bit for bit
    };
    let (t1, t3, t5) = (tau_n(amps[0], 1), tau_n(amps[1], 3), tau_n(amps[2], 5));
    let t2d = (t1 + t3 + t5) * 2.0 * PI * (r_g / 1000.0).powi(2) * (c.magnet_length_mm / 1000.0);
    let model = t2d * f_end * c.f_cal_original;
    let (interp, interp_err) = if (1.0..=1.5).contains(&corner_gap) {
        let interp =
```

with:

```rust
            / 2.0
    };

    // Addendum A3: the Calculator's harmonic set, so the one-point correction compares like
    // with like (the workbook's 1, 3, 5 by default). E7: every harmonic at the true pull-out
    // angle when half a pitch is not the maximum.
    let count = harmonic_count(max_harmonic);
    let amps: Vec<f64> = ODD_HARMONICS[..count.unwrap_or(0)]
        .iter()
        .map(|&n| amp_n(n))
        .collect();
    let peak = peak_angle(&amps, dev);
    let tau_n = |a: f64, n: u32| match peak {
        Some(x) => tau_at(a, n, x),
        None => a * (f64::from(n) * PI / 2.0).sin(), // the workbook expression, bit for bit
    };
    let taus: Vec<f64> = amps
        .iter()
        .zip(ODD_HARMONICS)
        .map(|(&a, n)| tau_n(a, n))
        .collect();
    let t = |i: usize| harmonic_slot(count, &taus, i);
    // (t1 + t3 + t5) in the workbook: the same left fold.
    let t2d = harmonic_sum(count, taus.iter().copied())
        * 2.0
        * PI
        * (r_g / 1000.0).powi(2)
        * (c.magnet_length_mm / 1000.0);
    let model = t2d * f_end * c.f_cal_original;
    let (interp, interp_err) = if (1.0..=1.5).contains(&corner_gap) {
        let interp =
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
        br_test_T: br_t,
        pole_pitch_mm: tau_p,
        f_end,
        tau1_Pa: t1,
        tau3_Pa: t3,
        tau5_Pa: t5,
        torque_2d_Nm: t2d,
        original_model_Nm: model,
        fea_interp_Nm: interp,
```

with:

```rust
        br_test_T: br_t,
        pole_pitch_mm: tau_p,
        f_end,
        tau1_Pa: t(0),
        tau3_Pa: t(1),
        tau5_Pa: t(2),
        tau7_Pa: t(3),
        tau9_Pa: t(4),
        tau11_Pa: t(5),
        torque_2d_Nm: t2d,
        original_model_Nm: model,
        fea_interp_Nm: interp,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
        .fold(0.0, |acc, t| acc + t);
        assert_eq!(
            eleven.torque_2d_Nm,
            sum * 2.0 * PI * (eleven.gap_radius_mm / 1000.0).powi(2) * (c.magnet_length_mm / 1000.0)
        );
        assert!(eleven.model_torque_Nm != workbook.model_torque_Nm);
        // A code outside the choices: NaN, never another set (decision D3).
```

with:

```rust
        .fold(0.0, |acc, t| acc + t);
        assert_eq!(
            eleven.torque_2d_Nm,
            sum * 2.0
                * PI
                * (eleven.gap_radius_mm / 1000.0).powi(2)
                * (c.magnet_length_mm / 1000.0)
        );
        assert!(eleven.model_torque_Nm != workbook.model_torque_Nm);
        // A code outside the choices: NaN, never another set (decision D3).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
//! `corner_radius` (√(r_face² + (w/2)²)) and `br_factor` (1 + α (T − 20 °C)).
//! `mass_estimate` (Calculator rows 110-115) is ported with `MassResults`.
//!
//! The harmonic set is the workbook's fixed 1, 3, 5 ([`HARMONICS`]); selecting
//! more harmonics belongs to the Addendum A engine plan.
//!
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied are E3 (the N42SH
```

with:

```rust
//! `corner_radius` (√(r_face² + (w/2)²)) and `br_factor` (1 + α (T − 20 °C)).
//! `mass_estimate` (Calculator rows 110-115) is ported with `MassResults`.
//!
//! The harmonic set is the Rust-only assumption `coupling.max_harmonic` (Addendum A3):
//! the odd harmonics 1, 3, ... up to 11 ([`ODD_HARMONICS`], [`harmonic_count`]), the
//! workbook's 1, 3, 5 ([`HARMONICS`]) by default. The Calculator, the sweeps and the
//! Calibration prototype sum the same set.
//!
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied are E3 (the N42SH
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// The odd space harmonics the E7 peak search handles, 1 to 11 (Addendum A3: "selectable
/// up to 11"); amplitude `a[i]` of [`peak_off_half_pitch`] belongs to `ODD_HARMONICS[i]`.
pub const ODD_HARMONICS: [u32; 6] = [1, 3, 5, 7, 9, 11];

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";
```

with:

```rust
/// The odd space harmonics the E7 peak search handles, 1 to 11 (Addendum A3: "selectable
/// up to 11"); amplitude `a[i]` of [`peak_off_half_pitch`] belongs to `ODD_HARMONICS[i]`.
pub const ODD_HARMONICS: [u32; 6] = [1, 3, 5, 7, 9, 11];

/// The workbook's highest harmonic: the default of `coupling.max_harmonic`, the set [`HARMONICS`].
pub const WORKBOOK_MAX_HARMONIC: i64 = 5;

/// The choices of `coupling.max_harmonic` (Addendum A3): the highest odd harmonic summed.
pub const MAX_HARMONIC_CHOICES: [(i64, &str); 6] = [
    (1, "1"),
    (3, "1, 3"),
    (5, "1, 3, 5 (workbook)"),
    (7, "1, 3, 5, 7"),
    (9, "1, 3, 5, 7, 9"),
    (11, "1, 3, 5, 7, 9, 11"),
];

/// How many harmonics of [`ODD_HARMONICS`] the model sums for a `max_harmonic` code (3 for
/// the workbook's 5); `None` for a code outside [`MAX_HARMONIC_CHOICES`] (decision D3: every
/// harmonic sum is then NaN, never another set).
pub fn harmonic_count(max_harmonic: i64) -> Option<usize> {
    MAX_HARMONIC_CHOICES
        .iter()
        .position(|&(code, _)| code == max_harmonic)
        .map(|i| i + 1)
}

/// Σ `terms` of the harmonic set, a left fold from 0 as Python's `sum()`; NaN when the set
/// is invalid (`count` is `None`, [`harmonic_count`]).
pub(crate) fn harmonic_sum(count: Option<usize>, terms: impl Iterator<Item = f64>) -> f64 {
    match count {
        Some(_) => terms.fold(0.0, |acc, t| acc + t),
        None => f64::NAN,
    }
}

/// What the per-harmonic cell of `ODD_HARMONICS[i]` shows: `terms[i]` when the set sums
/// that harmonic, 0 when the set leaves it out, NaN when the set is invalid.
pub(crate) fn harmonic_slot(count: Option<usize>, terms: &[f64], i: usize) -> f64 {
    match count {
        Some(_) => terms.get(i).copied().unwrap_or(0.0),
        None => f64::NAN,
    }
}

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            c_end: f64 = 0.15 => param("-", "End-effect coefficient",
                "f_end = 1 − c_end · pole pitch / L.", "Calculator!C41")
                .range(0.0, 0.5, 0.005)
                .assumption(),
            mu0: f64 = MU0 => param("T·m/A", "Vacuum permeability", "", "Calculator!C43")
                .range(1.2566e-6, 1.2567e-6, 1e-11),
```

with:

```rust
            c_end: f64 = 0.15 => param("-", "End-effect coefficient",
                "f_end = 1 − c_end · pole pitch / L.", "Calculator!C41")
                .range(0.0, 0.5, 0.005)
                .assumption(),
            max_harmonic: i64 = WORKBOOK_MAX_HARMONIC => param_rust_only("-", "Highest odd harmonic summed",
                "Addendum A3 assumption. The torque model sums the odd space harmonics 1, 3, ... up to this one: the pull-out, the two circuit sums (C95, C96), every sweep row and the Calibration prototype, so the measured correction compares like with like. Workbook: 1, 3, 5. The cells of harmonics 1, 3 and 5 keep their terms; a harmonic left out reads 0 shear stress, and harmonics 7 to 11 add the Rust-only tau7_Pa, tau9_Pa and tau11_Pa.")
                .choices(&MAX_HARMONIC_CHOICES)
                .assumption(),
            mu0: f64 = MU0 => param("T·m/A", "Vacuum permeability", "", "Calculator!C43")
                .range(1.2566e-6, 1.2567e-6, 1e-11),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            s5_free: f64 => out("-", "Harmonic 5 geometry factor without back iron",
                "", "Calculator!C87"),
            tau5_Pa: f64 => out("Pa", "Harmonic 5 shear stress", "", "Calculator!C88"),
            tau_Pa: f64 => out("Pa", "Total magnetic shear stress at pull-out",
                "PM-PM couplings typically 100–250 kPa.", "Calculator!C89"),
            area_lever_m3: f64 => out("m³", "Gap area × lever arm (2π R_g² L)",
```

with:

```rust
            s5_free: f64 => out("-", "Harmonic 5 geometry factor without back iron",
                "", "Calculator!C87"),
            tau5_Pa: f64 => out("Pa", "Harmonic 5 shear stress", "", "Calculator!C88"),
            tau7_Pa: f64 => out_rust_only("Pa", "Harmonic 7 shear stress",
                "Addendum A3: summed when the highest harmonic (coupling.max_harmonic) is 7 or more; 0 otherwise."),
            tau9_Pa: f64 => out_rust_only("Pa", "Harmonic 9 shear stress",
                "Summed when the highest harmonic is 9 or more; 0 otherwise."),
            tau11_Pa: f64 => out_rust_only("Pa", "Harmonic 11 shear stress",
                "Summed when the highest harmonic is 11; 0 otherwise."),
            tau_Pa: f64 => out("Pa", "Total magnetic shear stress at pull-out",
                "PM-PM couplings typically 100–250 kPa.", "Calculator!C89"),
            area_lever_m3: f64 => out("m³", "Gap area × lever arm (2π R_g² L)",
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    1.0 + alpha_br * (temp_C - 20.0)
}

/// Per-harmonic pull-out shear stress [Pa] and its parts, for HARMONICS.
#[allow(clippy::too_many_arguments)] // Python signature
pub fn shear_stress(
    br_i: f64,
```

with:

```rust
    1.0 + alpha_br * (temp_C - 20.0)
}

/// Per-harmonic pull-out shear stress [Pa] and its parts, for every harmonic of
/// [`ODD_HARMONICS`] (Python: for `HARMONICS`); the model sums the first
/// [`harmonic_count`] of them (Addendum A3).
#[allow(clippy::too_many_arguments)] // Python signature
pub fn shear_stress(
    br_i: f64,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    g_mm: f64,
    backiron: i64,
    mu0: f64,
) -> [Harmonic; 3] {
    HARMONICS.map(|n| {
        let nf = f64::from(n);
        let k = nf * (npole as f64 / 2.0) / (r_g_mm / 1000.0);
        let bi = harmonic_amplitude(br_i, n, fill_i);
```

with:

```rust
    g_mm: f64,
    backiron: i64,
    mu0: f64,
) -> [Harmonic; 6] {
    ODD_HARMONICS.map(|n| {
        let nf = f64::from(n);
        let k = nf * (npole as f64 / 2.0) / (r_g_mm / 1000.0);
        let bi = harmonic_amplitude(br_i, n, fill_i);
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// half a pitch is not the maximum ([`peak_angle`] on the amplitudes
/// B_in,n·B_on,n/(2μ0)·S_n of the circuit `backiron` selects). Returns `h`
/// unchanged, bit for bit, when E7 is off or half a pitch is the maximum.
/// Shared by [`compute`] and the sweep rows.
pub(crate) fn at_pull_out(
    h: [Harmonic; 3],
    backiron: i64,
    mu0: f64,
    dev: Deviations,
) -> [Harmonic; 3] {
    let amplitude = |x: &Harmonic| x.bi * x.bo / (2.0 * mu0) * x.s(backiron);
    let mut h_pull = h;
    if let Some(x) = peak_angle(&h.map(|hn| amplitude(&hn)), dev) {
        for hn in h_pull.iter_mut() {
            hn.tau = tau_at(amplitude(hn), hn.n, x);
        }
```

with:

```rust
/// half a pitch is not the maximum ([`peak_angle`] on the amplitudes
/// B_in,n·B_on,n/(2μ0)·S_n of the circuit `backiron` selects). Returns `h`
/// unchanged, bit for bit, when E7 is off or half a pitch is the maximum.
/// `h` is the harmonic set summed (Addendum A3). Shared by [`compute`] and the sweep rows.
pub(crate) fn at_pull_out(
    h: &[Harmonic],
    backiron: i64,
    mu0: f64,
    dev: Deviations,
) -> Vec<Harmonic> {
    let amplitude = |x: &Harmonic| x.bi * x.bo / (2.0 * mu0) * x.s(backiron);
    let mut h_pull = h.to_vec();
    let amplitudes: Vec<f64> = h.iter().map(amplitude).collect();
    if let Some(x) = peak_angle(&amplitudes, dev) {
        for hn in h_pull.iter_mut() {
            hn.tau = tau_at(amplitude(hn), hn.n, x);
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        ci.backiron,
        ci.mu0,
    );
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    // `h` (the half-pitch terms) stays for the per-circuit sums below.
    let h_pull = at_pull_out(h, ci.backiron, ci.mu0, dev);
    let tau = h_pull.iter().fold(0.0, |acc, x| acc + x.tau); // Python sum(): left fold from 0
    let AL = 2.0 * PI * (R_g / 1000.0).powi(2) * (L / 1000.0);
    let T2D = tau * AL;
    let f_end = 1.0 - ci.c_end * tau_p / L;
```

with:

```rust
        ci.backiron,
        ci.mu0,
    );
    // Addendum A3: the harmonics summed, 1, 3, ... up to coupling.max_harmonic.
    let count = harmonic_count(ci.max_harmonic);
    let used = &h[..count.unwrap_or(0)];
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    // `used` (the half-pitch terms) stays for the per-circuit sums below.
    let h_pull = at_pull_out(used, ci.backiron, ci.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let tau = harmonic_sum(count, taus.iter().copied()); // Python sum(): left fold from 0
    let AL = 2.0 * PI * (R_g / 1000.0).powi(2) * (L / 1000.0);
    let T2D = tau * AL;
    let f_end = 1.0 - ci.c_end * tau_p / L;
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    // sum(bi * bo * S * sin(n pi/2) for n) / (2 mu0) * ...: note S inside the product, /(2 mu0) after the sum
    // E7: each circuit at the maximum of its own torque-angle curve.
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients = h.map(|x| x.bi * x.bo * s(&x));
        match peak_angle(&coefficients, dev) {
            Some(x) => h
                .iter()
                .zip(coefficients)
                .fold(0.0, |acc, (hn, c)| acc + tau_at(c, hn.n, x)),
            None => h.iter().fold(0.0, |acc, x| {
                acc + x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin()
            }),
        }
    };
    let T_iron = circuit(|x| x.s_iron) / (2.0 * ci.mu0) * AL * f_end * f_cal_original;
```

with:

```rust
    // sum(bi * bo * S * sin(n pi/2) for n) / (2 mu0) * ...: note S inside the product, /(2 mu0) after the sum
    // E7: each circuit at the maximum of its own torque-angle curve.
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients: Vec<f64> = used.iter().map(|x| x.bi * x.bo * s(x)).collect();
        match peak_angle(&coefficients, dev) {
            Some(x) => harmonic_sum(
                count,
                used.iter()
                    .zip(&coefficients)
                    .map(|(hn, &c)| tau_at(c, hn.n, x)),
            ),
            None => harmonic_sum(
                count,
                used.iter()
                    .map(|x| x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin()),
            ),
        }
    };
    let T_iron = circuit(|x| x.s_iron) / (2.0 * ci.mu0) * AL * f_end * f_cal_original;
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        };
        text.to_owned()
    };
    let [h1, h3, h5] = h_pull;

    ModelResults {
        inner_length_mm: mi.length_mm,
```

with:

```rust
        };
        text.to_owned()
    };
    let [h1, h3, h5, ..] = h;
    let tau_n = |i: usize| harmonic_slot(count, &taus, i);

    ModelResults {
        inner_length_mm: mi.length_mm,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        b_o1: h1.bo,
        s1_iron: h1.s_iron,
        s1_free: h1.s_free,
        tau1_Pa: h1.tau,
        k3: h3.k,
        b_i3: h3.bi,
        b_o3: h3.bo,
        s3_iron: h3.s_iron,
        s3_free: h3.s_free,
        tau3_Pa: h3.tau,
        k5: h5.k,
        b_i5: h5.bi,
        b_o5: h5.bo,
        s5_iron: h5.s_iron,
        s5_free: h5.s_free,
        tau5_Pa: h5.tau,
        tau_Pa: tau,
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
```

with:

```rust
        b_o1: h1.bo,
        s1_iron: h1.s_iron,
        s1_free: h1.s_free,
        tau1_Pa: tau_n(0),
        k3: h3.k,
        b_i3: h3.bi,
        b_o3: h3.bo,
        s3_iron: h3.s_iron,
        s3_free: h3.s_free,
        tau3_Pa: tau_n(1),
        k5: h5.k,
        b_i5: h5.bi,
        b_o5: h5.bo,
        s5_iron: h5.s_iron,
        s5_free: h5.s_free,
        tau5_Pa: tau_n(2),
        tau7_Pa: tau_n(3),
        tau9_Pa: tau_n(4),
        tau11_Pa: tau_n(5),
        tau_Pa: tau,
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            assert_eq!(harmonic_count(code), Some(i + 1), "{code}");
        }
        assert_eq!(harmonic_count(WORKBOOK_MAX_HARMONIC), Some(HARMONICS.len()));
        assert_eq!(CouplingInputs::default().max_harmonic, WORKBOOK_MAX_HARMONIC);
        for code in [0, 2, 4, 6, 12, 13, -1, i64::MIN, i64::MAX] {
            assert_eq!(harmonic_count(code), None, "{code}");
        }
```

with:

```rust
            assert_eq!(harmonic_count(code), Some(i + 1), "{code}");
        }
        assert_eq!(harmonic_count(WORKBOOK_MAX_HARMONIC), Some(HARMONICS.len()));
        assert_eq!(
            CouplingInputs::default().max_harmonic,
            WORKBOOK_MAX_HARMONIC
        );
        for code in [0, 2, 4, 6, 12, 13, -1, i64::MIN, i64::MAX] {
            assert_eq!(harmonic_count(code), None, "{code}");
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        for r in [&one, &eleven] {
            assert_eq!(
                (r.k3, r.b_i5, r.s5_iron, r.s1_free),
                (workbook.k3, workbook.b_i5, workbook.s5_iron, workbook.s1_free)
            );
        }
        assert!(eleven.tau7_Pa != 0.0 && eleven.tau9_Pa != 0.0 && eleven.tau11_Pa != 0.0);
```

with:

```rust
        for r in [&one, &eleven] {
            assert_eq!(
                (r.k3, r.b_i5, r.s5_iron, r.s1_free),
                (
                    workbook.k3,
                    workbook.b_i5,
                    workbook.s5_iron,
                    workbook.s1_free
                )
            );
        }
        assert!(eleven.tau7_Pa != 0.0 && eleven.tau9_Pa != 0.0 && eleven.tau11_Pa != 0.0);
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
//! [`crate::engine::deviations::REGISTRY`]): E4 is applied (Pole sweep C6:C11,
//! the keyed-bore wall adds the inner bondline), and so is E7 (columns N, Q, T
//! and U, and through them V to AA: pull-out at the maximum over angle, through
//! the model's `at_pull_out`).

use std::f64::consts::PI;

use super::compat::{py_max, py_min};
use super::deviations::{DeviationId, Deviations};
use super::meta::{col, rows};
use super::model::{at_pull_out, corner_radius, shear_stress};

/// Corner gaps of the gap sweep [mm] (rows 6-18).
pub const GAP_SWEEP_CORNER_GAPS_MM: [f64; 13] = [
```

with:

```rust
//! [`crate::engine::deviations::REGISTRY`]): E4 is applied (Pole sweep C6:C11,
//! the keyed-bore wall adds the inner bondline), and so is E7 (columns N, Q, T
//! and U, and through them V to AA: pull-out at the maximum over angle, through
//! the model's `at_pull_out`). Addendum A3: a row sums the Calculator's harmonic
//! set (`coupling.max_harmonic`); column U includes harmonics 7 to 11 when the set
//! does, which have no column of their own.

use std::f64::consts::PI;

use super::compat::{py_max, py_min};
use super::deviations::{DeviationId, Deviations};
use super::meta::{col, rows};
use super::model::{
    at_pull_out, corner_radius, harmonic_count, harmonic_slot, harmonic_sum, shear_stress,
};

/// Corner gaps of the gap sweep [mm] (rows 6-18).
pub const GAP_SWEEP_CORNER_GAPS_MM: [f64; 13] = [
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
    pub gear_eff: f64,
    pub required_floor_Nm: f64,
    pub max_diameter_mm: f64,
}

/// One sweep row (Python `_row`), columns C-AA with the workbook's status priority.
```

with:

```rust
    pub gear_eff: f64,
    pub required_floor_Nm: f64,
    pub max_diameter_mm: f64,
    /// The Calculator's highest harmonic (`coupling.max_harmonic`, Addendum A3).
    pub max_harmonic: i64,
}

/// One sweep row (Python `_row`), columns C-AA with the workbook's status priority.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
        ctx.mu0,
    );
    let s = h.map(|x| x.s(ctx.backiron));
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum (as the model).
    let h_pull = at_pull_out(h, ctx.backiron, ctx.mu0, dev);
    let U = h_pull.iter().fold(0.0, |acc, x| acc + x.tau);
    let V = U * 2.0 * PI * (H / 1000.0).powi(2) * (ctx.L / 1000.0);
    let W = 1.0 - ctx.c_end * I / ctx.L;
    let X = V * W * factor;
```

with:

```rust
        ctx.mu0,
    );
    let s = h.map(|x| x.s(ctx.backiron));
    // Addendum A3: the Calculator's harmonic set. E7: every harmonic at the true pull-out
    // angle when half a pitch is not the maximum (as the model).
    let count = harmonic_count(ctx.max_harmonic);
    let h_pull = at_pull_out(&h[..count.unwrap_or(0)], ctx.backiron, ctx.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let U = harmonic_sum(count, taus.iter().copied());
    let V = U * 2.0 * PI * (H / 1000.0).powi(2) * (ctx.L / 1000.0);
    let W = 1.0 - ctx.c_end * I / ctx.L;
    let X = V * W * factor;
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
    } else {
        "nominal: test needed"
    };
    let [h1, h3, h5] = h_pull;
    SweepRow {
        variable,
        inner_apothem_mm: a_i,
```

with:

```rust
    } else {
        "nominal: test needed"
    };
    let [h1, h3, h5, ..] = h;
    SweepRow {
        variable,
        inner_apothem_mm: a_i,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
        fill_outer: K,
        k1: h1.k,
        s1: s[0],
        tau1_Pa: h1.tau,
        k3: h3.k,
        s3: s[1],
        tau3_Pa: h3.tau,
        k5: h5.k,
        s5: s[2],
        tau5_Pa: h5.tau,
        tau_Pa: U,
        torque_2d_Nm: V,
        f_end: W,
```

with:

```rust
        fill_outer: K,
        k1: h1.k,
        s1: s[0],
        tau1_Pa: harmonic_slot(count, &taus, 0),
        k3: h3.k,
        s3: s[1],
        tau3_Pa: harmonic_slot(count, &taus, 1),
        k5: h5.k,
        s5: s[2],
        tau5_Pa: harmonic_slot(count, &taus, 2),
        tau_Pa: U,
        torque_2d_Nm: V,
        f_end: W,
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

- [ ] **Step 4: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test schema
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib harmonic 2>&1 | grep "test result"
```

Expected: the bless run passes (it adds the `coupling.max_harmonic` row, `"rust_only": true`, `"assumption": true`, to `tests/data/input_schema.json`); then `test result: ok. 10 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib harmonic 2>&1 | grep "test result"
```

Expected: `test result: ok. 10 passed` (the six new tests and four older ones whose names contain `harmonic`).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `118 passed`, `tests\robustness.rs` 10 passed, the others as in Task 1 (parity, the differential tests and the registry tests pass unedited: the default set sums the same terms in the same order).

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a2/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task2.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/calibration.rs magcoupling-rs/src/engine/sweeps.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/tests/schema.rs magcoupling-rs/tests/robustness.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): the harmonic set as a Rust-only assumption (coupling.max_harmonic, up to 11)

Spec A3: odd harmonics 1, 3, ... up to 11, the workbook's 1, 3, 5 by default.
The pull-out, both circuit sums, every sweep row and the Calibration prototype
sum the same set; a left-out harmonic reads 0, 7 to 11 add Rust-only cells.
A code outside the choices gives NaN (decision D3). Defaults bit-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 3: The end-effect validity flag (audit M9: f_end <= 0)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

The A-1 carry-over in `docs/ai/04-memory.yaml`: "M9 f_end <= 0 is a GUI flag, so expose an explicit engine validity flag for
it". The user decided on 2026-09-30 that a negative pull-out (f_end = 1 − c_end · pole pitch / L <= 0, audit M9) "is handled by a GUI
flag ('end-effect model out of range', affected numbers greyed); no physics change". The engine states it once, as a function and a
text result on the two sheets that compute f_end; the sweep rows keep their workbook columns (a table has no Rust-only column), and
the GUI flags a row through the same `end_effect_in_range` on its `f_end`. The flag is strict at 0 (a factor of exactly 0 gives a zero
pull-out, as meaningless as a negative one) and NaN is out of range. No existing number changes.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs` (`END_EFFECT_OUT_OF_RANGE`, `end_effect_in_range`, `end_effect_check`; Rust-only `model.end_effect_check`; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs` (Rust-only `calibration.end_effect_check`; test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs` (module doc: how a row is flagged)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (model and calibration rows)

**Interfaces:**
- Consumes: `ModelResults::f_end` (Calculator C92), `CalibrationResults::f_end` (Calibration C38).
- Produces:
  - `pub const END_EFFECT_OUT_OF_RANGE: &str = "End-effect model out of range"`;
  - `pub fn end_effect_in_range(f_end: f64) -> bool` (`f_end > 0.0`; NaN is out of range) and `pub fn end_effect_check(f_end: f64) -> &'static str` (`"OK"` or the constant);
  - Rust-only results `model.end_effect_check: String` and `calibration.end_effect_check: String`.
  Task 6 (inverse sizing) uses `end_effect_in_range` in its "meets the target" rule.

- [ ] **Step 1: Write the failing tests**

The equality edge (architecture section 7 step 8) on the pure function; 2 mm manual blocks with c_end = 0.5 (both inside their sliders) on the Calculator; a 2 mm prototype on the Calibration.

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
    }

    #[test]
    fn fill_factor_is_capped_at_one() {
        let c = CalibrationInputs {
            magnet_width_mm: 25.4,
```

with:

```rust
    }

    #[test]
    fn a_short_prototype_flags_the_end_effect_model() {
        // Audit M9 on the prototype: the same end-effect factor, the same flag.
        let r = compute(
            &CalibrationInputs::default(),
            WORKBOOK_MAX_HARMONIC,
            Deviations::NONE,
        );
        assert_eq!(r.end_effect_check, "OK");
        let c = CalibrationInputs {
            magnet_length_mm: 2.0,
            c_end: 0.5,
            ..CalibrationInputs::default()
        };
        let r = compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::NONE);
        assert!(r.f_end < 0.0, "{}", r.f_end);
        assert_eq!(r.end_effect_check, crate::engine::model::END_EFFECT_OUT_OF_RANGE);
    }

    #[test]
    fn fill_factor_is_capped_at_one() {
        let c = CalibrationInputs {
            magnet_width_mm: 25.4,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    }

    #[test]
    fn temperature_check_is_inclusive_at_the_rating() {
        let mut ci = CouplingInputs {
            op_temp_C: 150.0,
```

with:

```rust
    }

    #[test]
    fn the_end_effect_check_is_strict_at_zero() {
        // Audit M9, the user's decision: a flag, no physics change. Architecture section 7
        // step 8: the comparison `f_end > 0` at exact equality, then either side of it.
        assert_eq!(end_effect_check(0.0), END_EFFECT_OUT_OF_RANGE);
        assert_eq!(end_effect_check(-0.0), END_EFFECT_OUT_OF_RANGE);
        assert_eq!(end_effect_check(f64::MIN_POSITIVE), "OK");
        assert_eq!(end_effect_check(-1e-300), END_EFFECT_OUT_OF_RANGE);
        assert_eq!(end_effect_check(f64::NAN), END_EFFECT_OUT_OF_RANGE);
        assert!(end_effect_in_range(1.0) && !end_effect_in_range(f64::NEG_INFINITY));
    }

    #[test]
    fn short_magnets_flag_the_end_effect_model() {
        // Audit M9: f_end = 1 - c_end * pole pitch / L turns negative below L = c_end * pole
        // pitch (1.32 mm at the defaults), and the pull-out with it. 2 mm manual blocks with
        // c_end = 0.5, both inside their sliders, reach it.
        let r = at(&CouplingInputs::default());
        assert_eq!((r.f_end > 0.0, r.end_effect_check.as_str()), (true, "OK"));
        let mut ci = CouplingInputs {
            c_end: 0.5,
            ..CouplingInputs::default()
        };
        ci.magnets.part_inner = String::new();
        ci.magnets.part_outer = String::new();
        ci.magnets.manual_inner_length_mm = 2.0;
        ci.magnets.manual_outer_length_mm = 2.0;
        let r = at(&ci);
        assert!(r.f_end < 0.0 && r.pullout_Nm < 0.0, "{} {}", r.f_end, r.pullout_Nm);
        assert_eq!(r.end_effect_check, END_EFFECT_OUT_OF_RANGE);
    }

    #[test]
    fn temperature_check_is_inclusive_at_the_rating() {
        let mut ci = CouplingInputs {
            op_temp_C: 150.0,
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib end_effect 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors: ``error[E0425]: cannot find value `END_EFFECT_OUT_OF_RANGE` in module `crate::engine::model` `` and ``error[E0425]: cannot find function `end_effect_check` in this scope``.

- [ ] **Step 3: Add the flag**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade` and `tau7_Pa` to `tau11_Pa`), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
```

with:

```markdown
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa` and `end_effect_check`) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, out_rust_only, param, results};
use super::model::{
    ODD_HARMONICS, br_factor, corner_radius, harmonic_count, harmonic_slot, harmonic_sum,
    peak_angle, tau_at,
};

inputs! {
```

with:

```rust
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, out_rust_only, param, results};
use super::model::{
    ODD_HARMONICS, br_factor, corner_radius, end_effect_check, harmonic_count, harmonic_slot,
    harmonic_sum, peak_angle, tau_at,
};

inputs! {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
            br_test_T: f64 => out("T", "Br at assumed test temperature", "", "Calibration!C36"),
            pole_pitch_mm: f64 => out("mm", "Pole pitch", "", "Calibration!C37"),
            f_end: f64 => out("-", "End-effect factor", "", "Calibration!C38"),
            tau1_Pa: f64 => out("Pa", "Shear stress, harmonic 1", "", "Calibration!C40"),
            tau3_Pa: f64 => out("Pa", "Shear stress, harmonic 3", "", "Calibration!C41"),
            tau5_Pa: f64 => out("Pa", "Shear stress, harmonic 5", "", "Calibration!C42"),
```

with:

```rust
            br_test_T: f64 => out("T", "Br at assumed test temperature", "", "Calibration!C36"),
            pole_pitch_mm: f64 => out("mm", "Pole pitch", "", "Calibration!C37"),
            f_end: f64 => out("-", "End-effect factor", "", "Calibration!C38"),
            end_effect_check: String => out_rust_only("", "End-effect model check",
                "Audit M9 on the prototype: 'End-effect model out of range' when f_end is 0 or below; 'OK' otherwise."),
            tau1_Pa: f64 => out("Pa", "Shear stress, harmonic 1", "", "Calibration!C40"),
            tau3_Pa: f64 => out("Pa", "Shear stress, harmonic 3", "", "Calibration!C41"),
            tau5_Pa: f64 => out("Pa", "Shear stress, harmonic 5", "", "Calibration!C42"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
        br_test_T: br_t,
        pole_pitch_mm: tau_p,
        f_end,
        tau1_Pa: t(0),
        tau3_Pa: t(1),
        tau5_Pa: t(2),
```

with:

```rust
        br_test_T: br_t,
        pole_pitch_mm: tau_p,
        f_end,
        end_effect_check: end_effect_check(f_end).to_owned(),
        tau1_Pa: t(0),
        tau3_Pa: t(1),
        tau5_Pa: t(2),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
        };
        let r = compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::NONE);
        assert!(r.f_end < 0.0, "{}", r.f_end);
        assert_eq!(r.end_effect_check, crate::engine::model::END_EFFECT_OUT_OF_RANGE);
    }

    #[test]
```

with:

```rust
        };
        let r = compute(&c, WORKBOOK_MAX_HARMONIC, Deviations::NONE);
        assert!(r.f_end < 0.0, "{}", r.f_end);
        assert_eq!(
            r.end_effect_check,
            crate::engine::model::END_EFFECT_OUT_OF_RANGE
        );
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";

inputs! {
    /// Magnet parts and the manual fallbacks (Calculator!C11:C27).
```

with:

```rust

/// Text of the maximum-temperature cells when the magnet is not a library part.
pub const NOT_IN_LIBRARY: &str = "n/a";

/// Text of the end-effect checks (`model.end_effect_check`, `calibration.end_effect_check`)
/// when f_end = 1 − c_end · pole pitch / L is 0 or below. Audit M9: the empirical form turns
/// negative for short magnets, and the pull-out with it; the user's decision (2026-09-30) is
/// a flag, no physics change (the GUI greys the numbers computed from the pull-out).
pub const END_EFFECT_OUT_OF_RANGE: &str = "End-effect model out of range";

/// Whether an end-effect factor is inside the model's range (positive; NaN is not). The
/// sweep rows' `f_end` column and inverse sizing use it too.
pub fn end_effect_in_range(f_end: f64) -> bool {
    f_end > 0.0
}

/// "OK" or [`END_EFFECT_OUT_OF_RANGE`], by [`end_effect_in_range`].
pub fn end_effect_check(f_end: f64) -> &'static str {
    if end_effect_in_range(f_end) {
        "OK"
    } else {
        END_EFFECT_OUT_OF_RANGE
    }
}

inputs! {
    /// Magnet parts and the manual fallbacks (Calculator!C11:C27).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            torque_2d_Nm: f64 => out("N·m", "2D pull-out torque (infinite length)",
                "", "Calculator!C91"),
            f_end: f64 => out("-", "End-effect factor", "", "Calculator!C92"),
            pullout_Nm: f64 => out("N·m", "Pull-out torque at operating temperature",
                "Analytical estimate, not a guaranteed minimum.", "Calculator!C93"),
            pullout_20C_Nm: f64 => out("N·m", "Pull-out torque at 20 °C", "", "Calculator!C94"),
```

with:

```rust
            torque_2d_Nm: f64 => out("N·m", "2D pull-out torque (infinite length)",
                "", "Calculator!C91"),
            f_end: f64 => out("-", "End-effect factor", "", "Calculator!C92"),
            end_effect_check: String => out_rust_only("", "End-effect model check",
                "Audit M9 (the user's decision: a flag, no physics change): 'End-effect model out of range' when f_end is 0 or below, which makes the pull-out and every number computed from it meaningless; 'OK' otherwise."),
            pullout_Nm: f64 => out("N·m", "Pull-out torque at operating temperature",
                "Analytical estimate, not a guaranteed minimum.", "Calculator!C93"),
            pullout_20C_Nm: f64 => out("N·m", "Pull-out torque at 20 °C", "", "Calculator!C94"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
        f_end,
        pullout_Nm: T_pull,
        pullout_20C_Nm: T_pull20,
        pullout_iron_Nm: T_iron,
```

with:

```rust
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
        f_end,
        end_effect_check: end_effect_check(f_end).to_owned(),
        pullout_Nm: T_pull,
        pullout_20C_Nm: T_pull20,
        pullout_iron_Nm: T_iron,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        ci.magnets.manual_inner_length_mm = 2.0;
        ci.magnets.manual_outer_length_mm = 2.0;
        let r = at(&ci);
        assert!(r.f_end < 0.0 && r.pullout_Nm < 0.0, "{} {}", r.f_end, r.pullout_Nm);
        assert_eq!(r.end_effect_check, END_EFFECT_OUT_OF_RANGE);
    }
```

with:

```rust
        ci.magnets.manual_inner_length_mm = 2.0;
        ci.magnets.manual_outer_length_mm = 2.0;
        let r = at(&ci);
        assert!(
            r.f_end < 0.0 && r.pullout_Nm < 0.0,
            "{} {}",
            r.f_end,
            r.pullout_Nm
        );
        assert_eq!(r.end_effect_check, END_EFFECT_OUT_OF_RANGE);
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
//! and U, and through them V to AA: pull-out at the maximum over angle, through
//! the model's `at_pull_out`). Addendum A3: a row sums the Calculator's harmonic
//! set (`coupling.max_harmonic`); column U includes harmonics 7 to 11 when the set
//! does, which have no column of their own.

use std::f64::consts::PI;
```

with:

```rust
//! and U, and through them V to AA: pull-out at the maximum over angle, through
//! the model's `at_pull_out`). Addendum A3: a row sums the Calculator's harmonic
//! set (`coupling.max_harmonic`); column U includes harmonics 7 to 11 when the set
//! does, which have no column of their own. A row whose `f_end` (column W) is 0 or
//! below is outside the end-effect model (audit M9): the GUI flags it with
//! `model::end_effect_in_range`, as the Calculator's `end_effect_check` does.

use std::f64::consts::PI;
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib end_effect 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `121 passed`, the others as in Task 2 (the new results are Rust-only: the metadata, order and differential tests skip them).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task3.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/calibration.rs magcoupling-rs/src/engine/sweeps.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): end-effect validity flag (audit M9, f_end <= 0)

The user's decision: a negative pull-out is flagged, not corrected. The engine
states the rule once (end_effect_in_range, strict at 0, NaN out of range) and
reports it as the Rust-only model.end_effect_check and calibration.end_effect_check;
sweep rows are flagged through the same function on their f_end.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 4: The assumptions registry (spec A3): rationale, source, "assumptions modified", reset to workbook defaults

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. The rationale and source texts are given verbatim; do not reword them.

Spec A3: "Each assumption shows its value, unit, rationale and source", the v1 set, engine parameters with workbook defaults, an
"assumptions modified" banner "whenever any assumption differs from its workbook default, beside a 'reset to workbook defaults'
button". Report 6.5 settles three details this task follows: every A3 item but the harmonic set was already an input with a cell and
14 carry `.assumption()` (the knee fraction is Temperature design C46, not hard-coded); the end-effect coefficient is two inputs, so one
row maps to both; the clamp friction assumption is `clamps.friction` (C18), not the adapter joint's C66. Value, unit, label and slider
stay in the input metadata (one source); the registry adds the rationale and the source, and a test requires each source to cite the
workbook cell of every input its row sets. "Workbook default" is `DesignInputs::default()`: no approved correction changes an
assumption's default (a test compares with `defaults_with(Deviations::NONE)`), and the Rust-only harmonic set's is 5 (1, 3, 5).

**The traceability test is plan A-3's.** The spec's "changing each assumption changes every dependent result and no independent one,
using the equation registry's dependency graph" needs the A2 equation registry, which plan A-3 builds. This task adds the smoke
version: each assumption moves at least one result at the default design. It documents the override the A-1 plan recorded (decisions
A2, A9; `docs/ai/04-memory.yaml`): with E20 and the coercivity source at 1, the Hcj coefficient (C45) changes nothing for a library
part, and moves results with the source at 0. (The back-iron design flux density's override, a 1018 back iron, is off the default
design, so it moves results here.)

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/assumptions.rs` (`Assumption`, `ASSUMPTIONS` (14 rows), `AssumptionState`, `states`, `modified`, `any_modified`, `reset_to_workbook_defaults`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/assumptions.rs` (the registry against the metadata, and the engine support)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/mod.rs` (`pub mod assumptions;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (layout and tests rows)

**Interfaces:**
- Consumes: the 15 inputs flagged `.assumption()` (14 from M2, `coupling.max_harmonic` from Task 2); `InputSet::get/set`; `meta::input_rows`; `DesignInputs::default()`; `DesignInputs::defaults_with(Deviations::NONE)` (test-only); Task 3's `end_effect_check` (named in the end-effect rationale).
- Produces (module `engine::assumptions`):
  - `pub struct Assumption { pub id: &'static str, pub label: &'static str, pub paths: &'static [&'static str], pub rationale: &'static str, pub source: &'static str }` (`Clone, Copy, Debug, PartialEq, Eq`);
  - `pub const ASSUMPTIONS: [Assumption; 14]`, ids in the spec's order: `harmonics`, `end_effect` (paths `coupling.c_end`, `calibration.c_end`), `calibration_factor`, `production_variation`, `br_temperature_coefficient`, `hcj_temperature_coefficient`, `knee_fraction`, `demag_margin`, `backiron_design_flux_density`, `thermal_conductance`, `driving_rise`, `slip_event_duration`, `clamp_friction` (`clamps.friction`), `preload_fraction`;
  - `pub struct AssumptionState { pub assumption: &'static Assumption, pub values: Vec<Value>, pub defaults: Vec<Value>, pub unit: &'static str, pub modified: bool }`;
  - `pub fn states(inputs: &DesignInputs) -> Vec<AssumptionState>`, `pub fn modified(inputs: &DesignInputs) -> Vec<&'static Assumption>`, `pub fn any_modified(inputs: &DesignInputs) -> bool`, `pub fn reset_to_workbook_defaults(inputs: &mut DesignInputs)`.

- [ ] **Step 1: Write the failing tests**

`other_value` picks, for each assumption input, the slider end farther from its default (or another choice), so every change stays inside the slider.

Create `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/assumptions.rs` with exactly this content:

```rust
//! Addendum A3: the assumptions registry (`src/engine/assumptions.rs`) against the input
//! metadata, and its engine support ("assumptions modified", reset to workbook defaults).
//!
//! The spec's traceability test ("changing each assumption changes every dependent result
//! and no independent one, using the equation registry's dependency graph") needs the A2
//! equation registry and belongs to plan A-3. `each_assumption_moves_a_result_at_the_default_design`
//! is the smoke version: each assumption moves at least one result, with the documented
//! override.

use std::collections::BTreeSet;

use magcoupling::engine::api::{DesignInputs, compute_all};
use magcoupling::engine::assumptions::{
    ASSUMPTIONS, any_modified, modified, reset_to_workbook_defaults, states,
};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputMeta, InputSet, Value, input_rows, result_rows};

/// The metadata of the input at `path`.
fn meta(path: &str) -> &'static InputMeta {
    input_rows(&DesignInputs::default())
        .into_iter()
        .find(|r| r.path == path)
        .unwrap_or_else(|| panic!("{path}: not an input"))
        .meta
}

/// A value of the input at `path` other than its default, inside its slider or among its
/// choices: the range end farther from the default, or the first other choice.
fn other_value(path: &str) -> Value {
    let m = meta(path);
    let default = DesignInputs::default().get(path).expect("an input");
    if !m.choices.is_empty() {
        let code = m
            .choices
            .iter()
            .map(|&(c, _)| c)
            .find(|&c| Value::Int(c) != default)
            .expect("a second choice");
        return Value::Int(code);
    }
    let r = m.range.expect("a numeric assumption has a slider");
    let x = match default {
        Value::Num(x) => x,
        other => panic!("{path}: {other:?}"),
    };
    Value::Num(if (r.max - x).abs() >= (x - r.min).abs() {
        r.max
    } else {
        r.min
    })
}

#[test]
fn every_flagged_input_is_in_exactly_one_row() {
    let flagged: BTreeSet<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| r.meta.assumption)
        .map(|r| r.path)
        .collect();
    let mut listed = BTreeSet::new();
    for a in &ASSUMPTIONS {
        assert!(!a.paths.is_empty(), "{}", a.id);
        for &path in a.paths {
            assert!(listed.insert(path.to_owned()), "{path} is in two rows");
            assert!(meta(path).assumption, "{path} is not flagged .assumption()");
        }
    }
    assert_eq!(listed, flagged);
}

#[test]
fn the_rows_are_the_spec_v1_set() {
    // Spec A3, in its order: harmonics included, end-effect coefficient, calibration factor,
    // production variation, Br and Hcj temperature coefficients, demag knee fraction, demag
    // margin, back-iron design flux density, thermal conductance, driving rise, slip-event
    // duration, clamp friction coefficient, preload fraction of proof load.
    let ids: Vec<&str> = ASSUMPTIONS.iter().map(|a| a.id).collect();
    assert_eq!(
        ids,
        [
            "harmonics",
            "end_effect",
            "calibration_factor",
            "production_variation",
            "br_temperature_coefficient",
            "hcj_temperature_coefficient",
            "knee_fraction",
            "demag_margin",
            "backiron_design_flux_density",
            "thermal_conductance",
            "driving_rise",
            "slip_event_duration",
            "clamp_friction",
            "preload_fraction",
        ]
    );
    // The end-effect coefficient is the Calculator's and the Calibration's (report 6.5);
    // the clamp friction is the shaft-to-bore input, not the adapter joint's C66.
    let paths = |id: &str| ASSUMPTIONS.iter().find(|a| a.id == id).unwrap().paths;
    assert_eq!(paths("end_effect"), ["coupling.c_end", "calibration.c_end"]);
    assert_eq!(paths("clamp_friction"), ["clamps.friction"]);
    assert_eq!(paths("harmonics"), ["coupling.max_harmonic"]);
}

#[test]
fn every_row_has_a_label_rationale_source_and_one_unit() {
    let mut ids = BTreeSet::new();
    for a in &ASSUMPTIONS {
        assert!(ids.insert(a.id), "{}: duplicate id", a.id);
        for (what, text) in [
            ("label", a.label),
            ("rationale", a.rationale),
            ("source", a.source),
        ] {
            assert!(!text.trim().is_empty(), "{}: empty {what}", a.id);
        }
        let unit = meta(a.paths[0]).unit;
        for &path in a.paths {
            let m = meta(path);
            assert_eq!(m.unit, unit, "{}: {path} has another unit", a.id);
            // The source cites the workbook cell of every workbook input it sets; a Rust-only
            // input (the harmonic set) has none and cites the spec.
            match m.cell {
                Some(cell) => assert!(a.source.contains(cell), "{}: source omits {cell}", a.id),
                None => assert!(a.source.contains("Addendum A3"), "{}", a.id),
            }
        }
    }
}

#[test]
fn the_workbook_defaults_are_the_shipped_defaults() {
    // No approved correction changes an assumption's default (E1, E3 and E5 correct other
    // inputs), so "reset to workbook defaults" is `DesignInputs::default()` for every row; the
    // Rust-only harmonic set's workbook value is 1, 3, 5.
    let shipped = DesignInputs::default();
    let workbook = DesignInputs::defaults_with(Deviations::NONE);
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            assert_eq!(shipped.get(path), workbook.get(path), "{path}");
        }
    }
    assert_eq!(shipped.get("coupling.max_harmonic"), Some(Value::Int(5)));
}

#[test]
fn nothing_is_modified_at_the_defaults() {
    let inputs = DesignInputs::default();
    assert!(!any_modified(&inputs));
    assert!(modified(&inputs).is_empty());
    let panel = states(&inputs);
    assert_eq!(panel.len(), ASSUMPTIONS.len());
    for (s, a) in panel.iter().zip(&ASSUMPTIONS) {
        assert_eq!(s.assumption, a);
        assert!(!s.modified, "{}", a.id);
        assert_eq!(s.values, s.defaults, "{}", a.id);
        assert_eq!(s.unit, meta(a.paths[0]).unit);
        assert_eq!(s.values.len(), a.paths.len());
    }
}

#[test]
fn changing_one_assumption_flags_exactly_its_row() {
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            let mut inputs = DesignInputs::default();
            inputs.set(path, other_value(path)).unwrap();
            let ids: Vec<&str> = modified(&inputs).iter().map(|m| m.id).collect();
            assert_eq!(ids, [a.id], "{path}");
            assert!(any_modified(&inputs), "{path}");
            let s = states(&inputs)
                .into_iter()
                .find(|s| s.assumption.id == a.id)
                .unwrap();
            assert!(s.modified && s.values != s.defaults, "{path}");
        }
    }
    // A design input is not an assumption: the banner stays off.
    let mut inputs = DesignInputs::default();
    inputs.coupling.npole = 12;
    inputs.metal.face_gap_mm = 1.2;
    assert!(!any_modified(&inputs));
}

#[test]
fn reset_restores_every_assumption_and_keeps_the_design_inputs() {
    let mut design = DesignInputs::default();
    design.coupling.npole = 12;
    design.metal.face_gap_mm = 1.2;
    design.coupling.magnets.part_inner = "B842".to_owned();
    design.materials.parts.back_iron = 2;
    let mut inputs = design.clone();
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            inputs.set(path, other_value(path)).unwrap();
        }
    }
    assert_eq!(modified(&inputs).len(), ASSUMPTIONS.len());
    reset_to_workbook_defaults(&mut inputs);
    assert!(!any_modified(&inputs));
    assert_eq!(inputs, design, "only the assumptions change");
}

#[test]
fn each_assumption_moves_a_result_at_the_default_design() {
    // The smoke version of A3's traceability test (the dependency-graph version is plan A-3's).
    // Documented override (A-1 plan decisions A2 and A9): with correction E20 and the coercivity
    // source at 1 (the default), the Hcj temperature coefficient (C45) acts only for a magnet
    // without a grade, and every library part has one; with the source at 0 it moves results.
    let values = |inputs: &DesignInputs| -> Vec<Value> {
        result_rows(&compute_all(inputs))
            .into_iter()
            .map(|r| r.value)
            .collect()
    };
    let base = values(&DesignInputs::default());
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            let mut inputs = DesignInputs::default();
            inputs.set(path, other_value(path)).unwrap();
            let moved = values(&inputs) != base;
            if path == "temperature.demag.beta_hcj_per_C" {
                assert!(!moved, "{path}: the grade's beta governs (E20)");
                inputs.temperature.demag.coercivity_source = 0;
                let mut source_0 = DesignInputs::default();
                source_0.temperature.demag.coercivity_source = 0;
                assert!(values(&inputs) != values(&source_0), "{path} with source 0");
            } else {
                assert!(moved, "{path} moves no result");
            }
        }
    }
    // Every assumption is numeric (tests/schema.rs); the harmonic set is the one selector.
    for a in &ASSUMPTIONS {
        for &path in a.paths {
            let m = meta(path);
            assert!(matches!(m.ty, FieldType::F64 | FieldType::I64), "{path}");
        }
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test assumptions 2>&1 | grep -E "^error" | head -3
```

Expected: ``error[E0432]: unresolved import `magcoupling::engine::assumptions` ``.

- [ ] **Step 3: Create the registry**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

with:

```markdown
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `tests/grades.rs` | The grade table equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (engine literals bit for bit; the N42SH engine beta is the workbook's); every library part resolves to a grade; a part's workbook Br and Tmax equal its grade's except the registered differences; sintered NdFeB alpha and density equal the engine constants; only ferrite has a positive beta; every part cites its vendor page for coating and magnetization. |
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/grades.rs` | The grade table equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (engine literals bit for bit; the N42SH engine beta is the workbook's); every library part resolves to a grade; a part's workbook Br and Tmax equal its grade's except the registered differences; sintered NdFeB alpha and density equal the engine constants; only ferrite has a positive beta; every part cites its vendor page for coating and magnetization. |
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

Create `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/assumptions.rs` with exactly this content:

```rust
//! Addendum A3: the model assumptions, apart from the design inputs.
//!
//! Every assumption of the spec's v1 set is an input flagged `.assumption()` where it is
//! declared, and its value, unit, label, slider and workbook cell live there, once. This
//! module adds what the assumptions panel shows beside them, a rationale and a source, and
//! the engine support the spec names: which assumptions differ from their workbook default
//! (the "assumptions modified" banner, [`any_modified`], [`modified`]) and the reset
//! ([`reset_to_workbook_defaults`]). The end-effect coefficient is two inputs, the
//! Calculator's and the Calibration's, which the differential data vary independently
//! (report 6.5), so a row lists its input paths.
//!
//! "Workbook default" is [`DesignInputs::default`]: no approved correction changes an
//! assumption's default (E1, E3 and E5 correct other inputs; `tests/assumptions.rs`
//! checks it), and the Rust-only harmonic set defaults to the workbook's 1, 3, 5.
//!
//! Two assumptions have documented overrides (A-1 plan decisions A2 and A9): with
//! correction E20 and the coercivity source at 1, the Hcj temperature coefficient acts only
//! for a magnet without a grade; a library back iron with its own design flux density
//! (1018) replaces the back-iron design flux density in the wall check.
//!
//! The spec's traceability test (each assumption changes every dependent result and no
//! independent one, by the equation registry's dependency graph) needs the A2 equation
//! registry and belongs to plan A-3.

use super::api::DesignInputs;
use super::meta::{InputSet, Value, input_rows};

/// One assumption of the Addendum A3 panel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Assumption {
    /// Stable id (the GUI's key).
    pub id: &'static str,
    /// The panel's label.
    pub label: &'static str,
    /// The input paths it sets, each flagged `.assumption()`, all in one unit.
    pub paths: &'static [&'static str],
    /// Why the value is what it is.
    pub rationale: &'static str,
    /// Where it comes from: the workbook cell of each input and the report that discusses it.
    pub source: &'static str,
}

/// The spec's v1 set, in the spec's order.
pub const ASSUMPTIONS: [Assumption; 14] = [
    Assumption {
        id: "harmonics",
        label: "Harmonics included",
        paths: &["coupling.max_harmonic"],
        rationale: "Each ring's alternating magnetization is a square wave whose odd space harmonics 1, 3, 5, ... add to the torque; the workbook sums 1, 3 and 5. A harmonic's wave number grows with its order, so its field decays faster across the gap: higher harmonics add little at the default gap and more at small gaps or low fill.",
        source: "Spec Addendum A3 (workbook 1, 3, 5; selectable up to 11); workbook Calculator rows 71 to 88; M2 decision D7 and Addendum A decision 29 (the peak search for any set).",
    },
    Assumption {
        id: "end_effect",
        label: "End-effect coefficient",
        paths: &["coupling.c_end", "calibration.c_end"],
        rationale: "Empirical factor for the finite axial length, f_end = 1 − c_end · pole pitch / L, on the Calculator and on the Calibration prototype (each keeps its own input). Within 1 % of 3D for magnets longer than about 3 mm; at or below L = c_end · pole pitch the factor is 0 or negative and the end-effect model is out of range (end_effect_check).",
        source: "Workbook Calculator!C41 and Calibration!C23 (0.15); M1 audit M8 (error band against 3D) and M9 (negative for short magnets).",
    },
    Assumption {
        id: "calibration_factor",
        label: "Calibration factor",
        paths: &["calibration.f_cal_original"],
        rationale: "The model's calibration coefficient for every design but the measured prototype: the bench correction (Calibration C9) applies only to the prototype's own rings, pole count and circuit (Calculator C42). The audit found the 2D model within a few per cent of exact 2D sections.",
        source: "Workbook Calibration!C24 (0.95); M1 audit M4 to M6.",
    },
    Assumption {
        id: "production_variation",
        label: "Production variation",
        paths: &["metal.variation"],
        rationale: "Symmetric allowance on the pull-out for magnet, gap and assembly scatter: the hot low torque is the pull-out at the operating temperature times (1 − variation), the cold high torque the cold pull-out times (1 + variation). An engineering allowance, not measured.",
        source: "Workbook Metal design!C18 (±15 %).",
    },
    Assumption {
        id: "br_temperature_coefficient",
        label: "Br temperature coefficient",
        paths: &["calibration.alpha_br_per_C"],
        rationale: "Reversible remanence coefficient: Br(T) = Br(20 °C) · (1 + α (T − 20 °C)), and torque scales with Br², on every sheet. −0.12 %/°C is the sintered NdFeB value of every library part's grade.",
        source: "Workbook Calibration!C22 (−0.0012 /°C); Addendum A grade table (K&J, sintered NdFeB).",
    },
    Assumption {
        id: "hcj_temperature_coefficient",
        label: "Hcj temperature coefficient",
        paths: &["temperature.demag.beta_hcj_per_C"],
        rationale: "Effective coefficient of the intrinsic coercivity over 20 to 150 °C, for the knee Hk(T) = knee · Hcj(20 °C) · (1 + β (T − 20 °C)). With correction E20 a magnet with a grade (every library part) uses its grade's β unless the coercivity source is set to the inputs; this value acts for a magnet without a grade or with that source.",
        source: "Workbook Temperature design!C45 (−0.50 %/°C, N42SH); Addendum A decisions 18 (Arnold's −0.55 %/°C is the reference, not the default) and 19 (E20).",
    },
    Assumption {
        id: "knee_fraction",
        label: "Demagnetization knee fraction",
        paths: &["temperature.demag.knee_fraction"],
        rationale: "The knee of the demagnetization curve as a fraction of Hcj: the reverse field at which irreversible loss starts. The onsets are calibrated to the magnet's rating; the audit found that calibration uses 9.83 °C of the 10 °C design margin, so the margin left for model-form uncertainty is thin.",
        source: "Workbook Temperature design!C46 (0.9); M1 audit rulings (onset calibration) and M13.",
    },
    Assumption {
        id: "demag_margin",
        label: "Demagnetization margin",
        paths: &["temperature.demag.design_margin_C"],
        rationale: "Margin kept from the skipping onset (like poles facing, the largest reverse field): the magnet design limit is that onset minus this margin, and with E20 a positive-beta magnet's cold limit is its cold onset plus it.",
        source: "Workbook Temperature design!C51 (10 °C); M1 audit rulings (onset calibration).",
    },
    Assumption {
        id: "backiron_design_flux_density",
        label: "Back-iron design flux density",
        paths: &["materials.steel.bsat_T"],
        rationale: "The flux density the back-iron wall is sized to, t = B_gap · pole pitch / (π · B): a design limit below saturation for annealed 4140 (1018 about 1.7 T, pre-hardened stock about 1.4 T). A library back iron with its own design value (1018) replaces it in the wall check.",
        source: "Workbook Materials!C13 (1.5 T); Addendum A decision 20 and the A-1 plan's decision A9; M1 audit M1 (the wall formula reads 12 to 16 % thin).",
    },
    Assumption {
        id: "thermal_conductance",
        label: "Thermal conductance",
        paths: &["temperature.thermal.conductance_W_K"],
        rationale: "One conductance from the rotating coupling to the housing and both shafts; it sets the steady slip rise and the thermal time constant. A placeholder, not measured.",
        source: "Workbook Temperature design!C142 (0.3 W/K); M1 audit placeholder inputs (−2.48 °C of steady high-case magnet temperature per +10 %).",
    },
    Assumption {
        id: "driving_rise",
        label: "Driving temperature rise",
        paths: &["temperature.duty.driving_rise_C"],
        rationale: "The coupling's rise above ambient while driving without slip (housing air, sun, gearbox heat). It sets the hot-day starting temperature, the most influential thermal input. A placeholder, not measured.",
        source: "Workbook Temperature design!C36 (10 °C); M1 audit placeholder inputs.",
    },
    Assumption {
        id: "slip_event_duration",
        label: "Slip-event duration",
        paths: &["metal.slip_event_s"],
        rationale: "How long one slip event lasts; it sets the life slip rotations, the heat per event and the slip duty. Illustrative: replace it with the recorded value.",
        source: "Workbook Metal design!C87 (0.1 s); M1 audit placeholder inputs.",
    },
    Assumption {
        id: "clamp_friction",
        label: "Clamp friction coefficient",
        paths: &["clamps.friction"],
        rationale: "Friction between the shaft and the clamp bore in the clamp capacity µ · preload · d · factor: degreased; 0.10 if the bore could be oily. The adapter joint's friction (Shaft clamps C66) is a design input, not this assumption.",
        source: "Workbook Shaft clamps!C18 (0.15); Addendum A report section 6.5 (which input the assumption names).",
    },
    Assumption {
        id: "preload_fraction",
        label: "Preload fraction of proof load",
        paths: &["clamps.preload_fraction"],
        rationale: "Screw preload as a share of the proof load (bolted-joint practice), capped by thread stripping in the aluminium.",
        source: "Workbook Shaft clamps!C29 (75 %); spec M1 (clamps).",
    },
];

/// One assumption as the panel shows it for a set of inputs.
#[derive(Clone, Debug, PartialEq)]
pub struct AssumptionState {
    pub assumption: &'static Assumption,
    /// The current value of each path, in `paths` order.
    pub values: Vec<Value>,
    /// The workbook default of each path, in `paths` order.
    pub defaults: Vec<Value>,
    /// The unit of every path.
    pub unit: &'static str,
    /// Whether any path differs from its workbook default (the panel's changed dot).
    pub modified: bool,
}

/// The value at an assumption path (every path is an input: `tests/assumptions.rs`).
fn value_at(inputs: &DesignInputs, path: &str) -> Value {
    inputs
        .get(path)
        .expect("an assumption path is an input (tests/assumptions.rs)")
}

/// The panel for `inputs`: every assumption with its values, workbook defaults and unit, in
/// [`ASSUMPTIONS`] order.
pub fn states(inputs: &DesignInputs) -> Vec<AssumptionState> {
    let defaults = DesignInputs::default();
    let rows = input_rows(inputs);
    ASSUMPTIONS
        .iter()
        .map(|a| {
            let values: Vec<Value> = a.paths.iter().map(|p| value_at(inputs, p)).collect();
            let workbook: Vec<Value> = a.paths.iter().map(|p| value_at(&defaults, p)).collect();
            let unit = rows
                .iter()
                .find(|r| r.path == a.paths[0])
                .map_or("", |r| r.meta.unit);
            AssumptionState {
                assumption: a,
                modified: values != workbook,
                values,
                defaults: workbook,
                unit,
            }
        })
        .collect()
}

/// The assumptions that differ from their workbook default, in [`ASSUMPTIONS`] order.
pub fn modified(inputs: &DesignInputs) -> Vec<&'static Assumption> {
    let defaults = DesignInputs::default();
    ASSUMPTIONS
        .iter()
        .filter(|a| {
            a.paths
                .iter()
                .any(|p| value_at(inputs, p) != value_at(&defaults, p))
        })
        .collect()
}

/// Whether any assumption differs from its workbook default (the spec's "assumptions
/// modified" banner).
pub fn any_modified(inputs: &DesignInputs) -> bool {
    !modified(inputs).is_empty()
}

/// Puts every assumption back to its workbook default and leaves every design input as it
/// is (the spec's "reset to workbook defaults" button).
pub fn reset_to_workbook_defaults(inputs: &mut DesignInputs) {
    let defaults = DesignInputs::default();
    for a in &ASSUMPTIONS {
        for path in a.paths {
            inputs
                .set(path, value_at(&defaults, path))
                .expect("a default is a valid value of its own input");
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/mod.rs`, replace:

```rust
//! - [`deviations`]: the registry of approved corrections to the workbook.

pub mod api;
pub mod calibration;
pub mod clamps;
pub mod compat;
```

with:

```rust
//! - [`deviations`]: the registry of approved corrections to the workbook.

pub mod api;
pub mod assumptions;
pub mod calibration;
pub mod clamps;
pub mod compat;
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test assumptions 2>&1 | grep "test result"
```

Expected: `test result: ok. 8 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: the new `tests\assumptions.rs` 8 passed, the others as in Task 3.

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task4.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task4.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/assumptions.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/tests/assumptions.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): assumptions registry (spec A3: rationale, source, modified, reset)

The spec's 14 rows over the 15 inputs flagged .assumption() (the end-effect
coefficient is two inputs; report 6.5). Value, unit and slider stay in the
metadata; each source cites its inputs' workbook cells. modified/any_modified
drive the banner, reset_to_workbook_defaults keeps the design inputs. The
dependency-graph traceability test is plan A-3's.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 5: The axial length override (`coupling.magnets.axial_length_mm`, Rust-only)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

Spec A1 names "axial magnet length (default)" as inverse sizing's free variable, and the M4 Key design group lists "magnet part, axial
length" side by side; the engine had no axial length that acts with a library part (the manual lengths apply only when the part is not
in the library, and every default ring is one). Decision A2-2 adds this override. Its interactions, stated in its help and pinned by
tests: it overrides manual lengths too; the calibration factor still follows the part names as the workbook's C42 does (decision A2-3:
no step in torque as a sized length passes the prototype's 12.7 mm); the magnets' mass and every length-dependent result follow; the
retainer span, hub length and cup depth stay as typed in this task (Metal design C172, C123, C124, class N in report 6.5: decision 28
gives them no rule), and Task 7b makes them follow the override (Decisions to confirm A2-8, option B, the user's choice).
`None` is the default: parity and the differential data are unchanged.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs` (the input; `resolve_magnets` applies it to both rings; test)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs` (the override end to end (Task 6 and Task 7 add to it))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs` (a non-finite override on the struct)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/data/input_schema.json` (the new Rust-only input)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (model row; tests row)

**Interfaces:**
- Consumes: `model::MagnetInputs`, `model::resolve_magnets(m: &MagnetInputs, dev) -> (ResolvedMagnet, ResolvedMagnet)`.
- Produces: input `coupling.magnets.axial_length_mm: Option<f64> = None` (Rust-only, slider 2.0 to 50.8 mm, step 0.01, the manual lengths' range). `None` keeps each ring's part or manual length; `Some(L)` sets both rings' `length_mm` to L after the part, grade or manual resolution, so width, thickness, Br, rating and grade are the ring's own. Task 6 sizes this input.

- [ ] **Step 1: Write the failing tests**

A unit test in `model.rs` (both rings take the length and keep the rest; torque scales with L − c_end · pole pitch; manual lengths are overridden; `None` changes nothing), the first test of the new `tests/sizing.rs` (the measured calibration factor stays, the magnets' mass follows, the housing inputs do not; Task 7b rewrites that last assertion), and a fifth non-finite struct literal in `tests/robustness.rs`.

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    }

    #[test]
    fn e3_reaches_library_parts_but_never_a_typed_manual_br() {
        let mut m = MagnetInputs {
            part_inner: "BX082SH".to_owned(),
```

with:

```rust
    }

    #[test]
    fn the_axial_length_override_sets_both_rings_and_keeps_the_rest() {
        // Addendum A1: blocks of the parts' cross-section, grade, Br and rating, cut or stacked
        // to one axial length. Torque ~ L * f_end = L - c_end * pole pitch (the pitch does not
        // depend on L).
        let base = at(&CouplingInputs::default());
        let mut ci = CouplingInputs::default();
        ci.magnets.axial_length_mm = Some(20.0);
        let r = at(&ci);
        assert_eq!(
            (r.inner_length_mm, r.outer_length_mm, r.active_length_mm),
            (20.0, 20.0, 20.0)
        );
        assert_eq!(
            (r.inner_width_mm, r.inner_thickness_mm, r.outer_width_mm),
            (
                base.inner_width_mm,
                base.inner_thickness_mm,
                base.outer_width_mm
            )
        );
        assert_eq!(
            (r.inner_br_T, r.inner_tmax_C),
            (base.inner_br_T, base.inner_tmax_C)
        );
        assert_eq!((r.inner_grade.as_str(), r.f_cal), ("N42SH", base.f_cal));
        assert_eq!(r.pole_pitch_mm, base.pole_pitch_mm);
        let excess = |l: f64| l - CouplingInputs::default().c_end * base.pole_pitch_mm;
        assert!(close(
            r.pullout_Nm / base.pullout_Nm,
            excess(20.0) / excess(12.7)
        ));
        // It overrides manual lengths too, and a blank override changes nothing.
        let mut manual = CouplingInputs::default();
        manual.magnets.part_inner = String::new();
        manual.magnets.manual_inner_length_mm = 30.0;
        manual.magnets.axial_length_mm = Some(15.0);
        let r = at(&manual);
        assert_eq!((r.inner_length_mm, r.outer_length_mm), (15.0, 15.0));
        let mut blank = CouplingInputs::default();
        blank.magnets.axial_length_mm = None;
        assert_eq!(at(&blank), base);
    }

    #[test]
    fn e3_reaches_library_parts_but_never_a_typed_manual_br() {
        let mut m = MagnetInputs {
            part_inner: "BX082SH".to_owned(),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs`, replace:

```rust
    // a share link can hold one. compute_all must not panic on it (gate 4 is a debug build,
    // overflow checks on), with the corrections off or on; validate() names the one path.
    type Put = fn(&mut DesignInputs, f64);
    let cases: [(&str, Put); 4] = [
        ("metal.face_gap_mm", |i, x| i.metal.face_gap_mm = x),
        ("clamps.boss_od_mm", |i, x| i.clamps.boss_od_mm = x),
        ("temperature.thermal.conductance_W_K", |i, x| {
```

with:

```rust
    // a share link can hold one. compute_all must not panic on it (gate 4 is a debug build,
    // overflow checks on), with the corrections off or on; validate() names the one path.
    type Put = fn(&mut DesignInputs, f64);
    let cases: [(&str, Put); 5] = [
        ("metal.face_gap_mm", |i, x| i.metal.face_gap_mm = x),
        ("clamps.boss_od_mm", |i, x| i.clamps.boss_od_mm = x),
        ("temperature.thermal.conductance_W_K", |i, x| {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs`, replace:

```rust
        }),
        ("metal.measured_drag_Nm", |i, x| {
            i.metal.measured_drag_Nm = Some(x)
        }),
    ];
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
```

with:

```rust
        }),
        ("metal.measured_drag_Nm", |i, x| {
            i.metal.measured_drag_Nm = Some(x)
        }),
        ("coupling.magnets.axial_length_mm", |i, x| {
            i.coupling.magnets.axial_length_mm = Some(x)
        }),
    ];
    for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
```

Create `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs` with exactly this content:

```rust
//! Addendum A1: the axial length override (Task 5), inverse sizing (Task 6) and the space
//! claim (Task 7), end to end through `compute_all`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;

#[test]
fn a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass() {
    // Decision A2-3: the measured correction keys on the part names, as the workbook's C42
    // does, so a length override keeps it for the prototype's rings with no back iron (no
    // step in torque as a sized length passes the prototype's 12.7 mm). The magnets' mass
    // follows the length; the housing inputs do not (decision 28: no rule sizes them).
    let mut inputs = DesignInputs::default();
    inputs.coupling.backiron = 0;
    let at_part = compute_all(&inputs);
    inputs.coupling.magnets.axial_length_mm = Some(20.0);
    let long = compute_all(&inputs);
    assert_eq!(at_part.model.f_cal, at_part.calibration.f_cal_updated);
    assert_eq!(long.model.f_cal, at_part.model.f_cal);
    assert!((long.mass.magnets_g / at_part.mass.magnets_g - 20.0 / 12.7).abs() < 1e-12);
    assert_eq!(long.metal.axial_stack_mm, at_part.metal.axial_stack_mm);
    assert_eq!(
        long.retainers.retainer_span_mm,
        at_part.retainers.retainer_span_mm
    );
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib axial_length 2>&1 | grep -E "^error" | head -2
```

Expected: ``error[E0609]: no field `axial_length_mm` on type `MagnetInputs` `` (or the struct-literal variant).

- [ ] **Step 3: Add the input**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa` and `end_effect_check`) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
```

with:

```markdown
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa` and `end_effect_check`) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass, not the housing inputs. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                "Addendum A6: a grade of the grade table, by exact name (e.g. N42SH, Y30). Used only when the inner part is not in the library: the manual dimensions with the grade's Br at 20 °C and maximum temperature. Blank = the manual Br and no rating."),
            grade_outer: String = "" => param_rust_only("-", "Outer magnet grade (manual dimensions)",
                "As the inner grade, for the outer ring."),
        }
    }
}
```

with:

```rust
                "Addendum A6: a grade of the grade table, by exact name (e.g. N42SH, Y30). Used only when the inner part is not in the library: the manual dimensions with the grade's Br at 20 °C and maximum temperature. Blank = the manual Br and no rating."),
            grade_outer: String = "" => param_rust_only("-", "Outer magnet grade (manual dimensions)",
                "As the inner grade, for the outer ring."),
            axial_length_mm: Option<f64> = None => param_rust_only("mm", "Axial magnet length, both rings",
                "Addendum A1. Blank = each ring's part or manual length. A value sets both rings' axial length and keeps everything else each ring has (part or manual cross-section, grade, Br, rating): blocks cut or stacked to length. The calibration factor still follows the part names (Calculator C42), and the retainer span, hub length and cup depth stay inputs (Metal design C172, C123, C124: no rule sizes them, Addendum A decision 28), so recheck them. Inverse sizing's default free variable.")
                .range(2.0, 50.8, 0.01),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// (Addendum A6, a Rust-only mode) and the manual Br and no rating otherwise.
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence),
/// its rating and grade from [`library::tmax_C`] and [`library::grade_id`], which apply E19.
#[allow(non_snake_case)]
pub fn resolve_magnets(m: &MagnetInputs, dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet) {
    let resolve =
```

with:

```rust
/// (Addendum A6, a Rust-only mode) and the manual Br and no rating otherwise.
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence),
/// its rating and grade from [`library::tmax_C`] and [`library::grade_id`], which apply E19.
/// The Rust-only `axial_length_mm` (Addendum A1), when set, replaces both rings' length.
#[allow(non_snake_case)]
pub fn resolve_magnets(m: &MagnetInputs, dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet) {
    let resolve =
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                },
            }
        };
    (
        resolve(
            &m.part_inner,
            &m.grade_inner,
            m.manual_inner_length_mm,
            m.manual_inner_width_mm,
            m.manual_inner_thickness_mm,
            m.manual_inner_br_T,
        ),
        resolve(
            &m.part_outer,
            &m.grade_outer,
            m.manual_outer_length_mm,
            m.manual_outer_width_mm,
            m.manual_outer_thickness_mm,
            m.manual_outer_br_T,
        ),
    )
}
```

with:

```rust
                },
            }
        };
    let with_length = |ring: ResolvedMagnet| match m.axial_length_mm {
        Some(length_mm) => ResolvedMagnet { length_mm, ..ring },
        None => ring,
    };
    (
        with_length(resolve(
            &m.part_inner,
            &m.grade_inner,
            m.manual_inner_length_mm,
            m.manual_inner_width_mm,
            m.manual_inner_thickness_mm,
            m.manual_inner_br_T,
        )),
        with_length(resolve(
            &m.part_outer,
            &m.grade_outer,
            m.manual_outer_length_mm,
            m.manual_outer_width_mm,
            m.manual_outer_thickness_mm,
            m.manual_outer_br_T,
        )),
    )
}
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

- [ ] **Step 4: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test schema
```

Expected: the bless run passes: `tests/data/input_schema.json` gains the `coupling.magnets.axial_length_mm` row (`"type": "opt_f64"`, `"default": null`, `"rust_only": true`).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib axial_length 2>&1 | grep "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test sizing 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed`, then `test result: ok. 1 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `122 passed`, the new `tests\sizing.rs` 1 passed, the others as in Task 4 (`compute_all_never_panics_on_extreme_inputs` now also sets the override to 0, −1, a tenth of 2 mm, 508 mm and ±1e300).

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a2/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task5.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task5.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task5.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/model.rs magcoupling-rs/tests/sizing.rs magcoupling-rs/tests/robustness.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): axial length override for both rings (Addendum A1)

Rust-only coupling.magnets.axial_length_mm: blank keeps each ring's length; a
value sets both rings' length and keeps the rest of each ring. The calibration
factor still follows the part names (decision A2-3); the housing inputs stay
inputs (decision 28). Inverse sizing's default free variable.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 6: Inverse sizing (spec A1: Torque → Magnets)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. The search is subtle, but the plan gives its exact code and its unit tests, so the task is transcription plus running tests and stays on `sonnet`.

Spec A1: a target torque; the hot-low torque with production variation (`metal.torque_hot_low_Nm`) must meet it by adjusting one
free variable (axial length by default; magnets per ring, discrete, poles even; ring radius); every other input fixed; "a bracketed
1-D search: bisection for the continuous variables, stepping for the discrete one"; "the smallest value that meets the target, or 'not
reachable' with the best value achieved inside the variable's slider range. It fails loudly and never extrapolates past the range."
The torque is not guaranteed monotone (on the default design it rises with the ring radius to about 14 mm, falls to about 20 mm and
rises again), so a continuous variable is sampled at the 65 ends of 64 equal cells, ascending, and refined wherever something can lie
between two samples: the first sample that meets is bisected against the one before it (a crossing); three consecutive valid samples
that rise and then do not rise get a golden-section peak search between the outer two, and a peak that meets has the crossing below it
bisected (a meeting interval inside one cell, 0.328 mm of radius or 0.778 mm of length: the case `a_meeting_interval_inside_one_cell_is_found`
pins, 2.0695 N·m with no back iron and a 0.5 mm gap, is met near 12.11 mm, where a grid-only scan would return 29.47 mm); where validity changes between
two samples the edge is bisected and its valid side sampled (a torque largest where the keyway starts to leave wall). Every peak is
offered as the best value, so "not reachable" reports the largest torque the search found, not only a grid value. The stated limit: a
hump whose rise and fall both lie inside one cell leaves no trace on the samples (a unit test pins it). Each value is a full
`compute_all` of the design with the variable changed (3 to 139 per solve on the default design; up to about 1,500 for a torque noisy
enough to show a peak at every other sample): the answer is exactly what the forward calculation shows, with no second copy of the
torque chain. Decision A2-4 sets what counts (`is_valid`): the blocks fit (`blocks_fit`: faceted blocks their flats, the Calculator's
C52/C59 comparison, now one function `flat_fits` for the checks and for sizing; arcs do not overlap at the magnet mid-radius, a
`pitch_share` of at most 1, where the fill C66/C67 would clamp and price overlapping arcs as fitting), the keyway leaves hub wall
(`model.hub_wall_past_key_mm` > 0), the end-effect factor is in range (Task 3: at short lengths f_end <= 0 gives a zero or negative
torque, a region the search handles explicitly by never counting it) and the torque is finite; a value meets when it counts and its
torque is at least the target. The space claim is not a condition (spec A1 shows exceeding it as a red callout); Task 7 reports it.
Errors are loud: a non-positive or non-finite target, or inputs that fail `validate()`.

The spec's tests and why each base design is fair: the round trips (1e-6) use designs where the original value is also the smallest
that meets its own torque: the torque grows with the length at any length (it goes as L − c_end · pole pitch); 4, 6 and 8 poles give
less than 8 and 10 on the default hub; the torque rises with the ring radius from the smallest radius whose flats fit (9.77 mm) to about
14 mm. The Review Focus edges each get a test: an odd base pole count, the smaller of two meeting intervals, a meeting interval inside
one cell, a target just below the true peak, a target met only at a validity edge, the f_end <= 0 region, overlapping arcs, a keyway
through the hub, no valid value at all (`best: None`), invalid targets and inputs, and extreme designs in `tests/robustness.rs`. The two
new threshold comparisons get their equality edges (`arcs_fit_exactly_at_a_pitch_share_of_one`, `the_hub_wall_counts_only_while_positive`).
`sizing.rs`'s unit tests drive the search alone with plain functions: each refinement, both directions of a validity edge, a plateau
(no peak search), stepping (the grid only), the bound on a peak search and the stated limit.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sizing.rs` (`FreeVariable`, `SCAN_CELLS`, `VALUE_TOLERANCE_MM`, `SizingPoint`, `SizingOutcome`, `SizingError`, `is_valid`, `solve`; the search and its unit tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs` (`flat_fits` (the flat checks' comparison, now shared), `pub fn pitch_share` (the fill's quotient, now shared) and `pub fn blocks_fit`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/mod.rs` (`pub mod sizing;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs` (the spec's sizing tests, the edges and the fit rules' equality edges)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs` (sizing on extreme designs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (layout and tests rows)

**Interfaces:**
- Consumes: `api::compute_all`, `api::DesignResults`, `DesignInputs::validate`, `meta::input_rows`, `meta::SliderRange`, `meta::SetError`; Task 3's `model::end_effect_in_range`; Task 5's `coupling.magnets.axial_length_mm`; the result `model.hub_wall_past_key_mm` (Calculator C53); the inputs `coupling.npole` (slider 4 to 40, step 2) and `coupling.inner_back_apothem_mm` (9.0 to 30.0 mm).
- Produces (module `engine::sizing`):
  - `pub const SCAN_CELLS: usize = 64`, `pub const VALUE_TOLERANCE_MM: f64 = 1e-9`;
  - `pub enum FreeVariable { AxialLength, MagnetsPerRing, RingRadius }` with `ALL` (the first is the default), `const fn path(self) -> &'static str`, `fn range(self) -> SliderRange` (from the metadata), `fn grid(self) -> Vec<f64>` (the coarse pass: 65 cell ends, or every even pole count), `fn apply(self, inputs: &DesignInputs, value: f64) -> DesignInputs`;
  - `pub struct SizingPoint { pub value: f64, pub torque_hot_low_Nm: f64, pub inputs: DesignInputs }`;
  - `pub enum SizingOutcome { Solved(SizingPoint), NotReachable { best: Option<SizingPoint> } }`; `pub enum SizingError { InvalidTarget(f64), InvalidInputs(Vec<SetError>) }`;
  - `pub fn is_valid(design: &DesignInputs, results: &DesignResults) -> bool` (decision A2-4; the evaluation inside `solve` and the tests read this one predicate);
  - `pub fn solve(inputs: &DesignInputs, variable: FreeVariable, target_Nm: f64) -> Result<SizingOutcome, SizingError>`;
  - private: `Sample<P>`, `Found<P>` and `Search<'a, P>` (the search over a grid, taking the evaluation as a function so its unit tests drive it with plain functions);
  - in `model`: `pub fn pitch_share(width_mm: f64, back_apothem_mm: f64, thickness_mm: f64, npole: f64) -> f64` (the fill C66/C67 before its min(1, ...), the same operations, so no bit moves) and `pub fn blocks_fit(ci: &CouplingInputs, r: &ModelResults) -> bool` (decision A2-4's fit rule).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs`, replace:

```rust
}

#[test]
fn compute_all_is_cheap_enough_to_run_every_frame() {
    // Spec: "milliseconds per call". Debug build, generous bound (a smoke check, not a benchmark).
    // A GUI frame recomputes and reads the dashboard numbers: compute_all plus headline.
```

with:

```rust
}

#[test]
fn sizing_never_panics_on_extreme_designs() {
    // Inverse sizing on designs no slider reaches: it must return an outcome or an error for
    // every free variable, never panic (gate 4 is a debug build, overflow checks on).
    use magcoupling::engine::sizing::{FreeVariable, solve};
    let mut designs = Vec::new();
    let mut d = DesignInputs::default();
    d.coupling.npole = i64::MAX;
    designs.push(d);
    let mut d = DesignInputs::default();
    d.metal.face_gap_mm = 1e300;
    designs.push(d);
    let mut d = DesignInputs::default();
    d.coupling.magnets.part_inner = String::new();
    d.coupling.magnets.manual_inner_thickness_mm = 0.0;
    d.coupling.backiron = 0;
    designs.push(d);
    let mut d = DesignInputs::default();
    d.metal.variation = 1.0; // the hot low torque is 0
    designs.push(d);
    for design in &designs {
        for variable in FreeVariable::ALL {
            let _ = solve(design, variable, 1.0);
        }
    }
}

#[test]
fn compute_all_is_cheap_enough_to_run_every_frame() {
    // Spec: "milliseconds per call". Debug build, generous bound (a smoke check, not a benchmark).
    // A GUI frame recomputes and reads the dashboard numbers: compute_all plus headline.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;

#[test]
fn a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass() {
```

with:

```rust

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::meta::SetErrorKind;
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
    FreeVariable, SCAN_CELLS, SizingError, SizingOutcome, SizingPoint, VALUE_TOLERANCE_MM,
    is_valid, solve,
};

/// The hot-low torque with production variation of a design (what sizing makes meet the target).
fn hot_low(inputs: &DesignInputs) -> f64 {
    compute_all(inputs).metal.torque_hot_low_Nm
}

/// The solved point, or a panic naming the outcome.
fn solved(outcome: Result<SizingOutcome, SizingError>) -> SizingPoint {
    match outcome {
        Ok(SizingOutcome::Solved(point)) => point,
        other => panic!("expected a solution, got {other:?}"),
    }
}

#[test]
fn a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust
        at_part.retainers.retainer_span_mm
    );
}
```

with:

```rust
        at_part.retainers.retainer_span_mm
    );
}

#[test]
fn each_free_variable_round_trips() {
    // Spec "Addendum testing": solving for the forward result's torque returns the original
    // value (1e-6). Each base design is one where that value is also the smallest that meets
    // the torque: the hot-low torque grows with the length (~ L - c_end * pole pitch) at any
    // length; T(4), T(6) and T(8) poles are below T(10) and T(8) on the default hub (0.40,
    // 0.79, 1.67 against 2.29 N m); the torque rises with the ring radius from the smallest
    // radius whose flats fit (9.77 mm) to about 14 mm.
    let cases: [(FreeVariable, &[f64]); 3] = [
        (FreeVariable::AxialLength, &[6.0, 12.7, 25.0]),
        (FreeVariable::MagnetsPerRing, &[8.0, 10.0]),
        (FreeVariable::RingRadius, &[10.15, 12.5]),
    ];
    for (variable, values) in cases {
        for &v0 in values {
            let base = variable.apply(&DesignInputs::default(), v0);
            let target = hot_low(&base);
            let p = solved(solve(&base, variable, target));
            assert!(
                (p.value - v0).abs() <= 1e-6,
                "{variable:?} {v0}: {}",
                p.value
            );
            assert!(p.torque_hot_low_Nm >= target, "{variable:?} {v0}");
            // Every other input stays fixed.
            assert_eq!(
                p.inputs,
                variable.apply(&base, p.value),
                "{variable:?} {v0}"
            );
            assert_eq!(p.torque_hot_low_Nm, hot_low(&p.inputs));
        }
    }
}

#[test]
fn a_target_beyond_the_range_is_not_reachable() {
    // Spec: "not reachable" with the best value achieved inside the variable's slider range;
    // it never extrapolates past the range.
    for variable in FreeVariable::ALL {
        let range = variable.range();
        match solve(&DesignInputs::default(), variable, 10.0) {
            Ok(SizingOutcome::NotReachable { best: Some(best) }) => {
                assert!(
                    (range.min..=range.max).contains(&best.value),
                    "{variable:?}: {}",
                    best.value
                );
                assert!(best.torque_hot_low_Nm < 10.0, "{variable:?}");
                assert_eq!(
                    best.inputs,
                    variable.apply(&DesignInputs::default(), best.value)
                );
                // The best of the values the search tried: no valid grid value does better.
                for value in variable.grid() {
                    let design = variable.apply(&DesignInputs::default(), value);
                    let r = compute_all(&design);
                    if is_valid(&design, &r) {
                        assert!(
                            r.metal.torque_hot_low_Nm <= best.torque_hot_low_Nm,
                            "{variable:?} {value}"
                        );
                    }
                }
                // The ring radius's torque peaks between two grid values: its best is the
                // refined peak, not a grid value.
                if variable == FreeVariable::RingRadius {
                    assert!(!variable.grid().contains(&best.value), "{}", best.value);
                }
            }
            other => panic!("{variable:?}: {other:?}"),
        }
    }
    // The length's torque grows with the length: its best is the slider's end, 50.8 mm.
    match solve(&DesignInputs::default(), FreeVariable::AxialLength, 10.0) {
        Ok(SizingOutcome::NotReachable { best: Some(best) }) => assert_eq!(best.value, 50.8),
        other => panic!("{other:?}"),
    }
}

#[test]
fn poles_stay_even() {
    // Spec: magnets per ring is discrete and the poles stay even, whatever the base design holds
    // (an odd pole count set straight on the struct: set() checks no step grid).
    let grid = FreeVariable::MagnetsPerRing.grid();
    assert_eq!(grid.first(), Some(&4.0));
    assert_eq!(grid.last(), Some(&40.0));
    assert!(grid.iter().all(|n| n % 2.0 == 0.0), "{grid:?}");
    let mut odd = DesignInputs::default();
    odd.coupling.npole = 11;
    for target in [0.3, 1.0, 1.7, 2.2] {
        let p = solved(solve(&odd, FreeVariable::MagnetsPerRing, target));
        assert_eq!(p.value % 2.0, 0.0, "{target}: {}", p.value);
        assert_eq!(p.inputs.coupling.npole % 2, 0, "{target}");
        assert_eq!(p.inputs.coupling.npole as f64, p.value);
    }
}

#[test]
fn the_smaller_of_two_meeting_intervals_is_returned() {
    // The torque is not monotone in the ring radius: on the default design it rises to about
    // 14 mm, falls to about 20 mm and rises again to the slider's end. At 2.5 N m both
    // 11.7 to 15.7 mm and 29.8 to 30 mm meet; the smallest value is the answer.
    let p = solved(solve(
        &DesignInputs::default(),
        FreeVariable::RingRadius,
        2.5,
    ));
    assert!(p.value > 11.0 && p.value < 13.0, "{}", p.value);
    let at =
        |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&DesignInputs::default(), radius));
    assert!(at(p.value) >= 2.5);
    assert!(
        at(p.value - 1e-6) < 2.5,
        "the boundary, to the bisection's tolerance"
    );
    assert!(at(30.0) >= 2.5 && at(20.0) < 2.5, "a second interval meets");
}

#[test]
fn a_meeting_interval_inside_one_cell_is_found() {
    // With no back iron and a 0.5 mm face gap the torque peaks near 12.12 mm of ring radius,
    // between two grid values (11.953125 and 12.28125 mm) that both miss 2.0695 N m: the
    // interval that meets lies inside one cell. The peak search finds it, and the answer is
    // its lower end, not the later interval near the slider's end (29.47 mm).
    let mut base = DesignInputs::default();
    base.coupling.backiron = 0;
    base.metal.face_gap_mm = 0.5;
    let target = 2.0695;
    let at = |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&base, radius));
    let p = solved(solve(&base, FreeVariable::RingRadius, target));
    assert!(p.value > 12.0 && p.value < 12.2, "{}", p.value);
    assert!(at(p.value) >= target);
    assert!(
        at(p.value - 1e-6) < target,
        "the lower end, to the bisection's tolerance"
    );
    // Both grid values around it miss: the case the coarse scan alone cannot see.
    let grid = FreeVariable::RingRadius.grid();
    let cell = grid
        .windows(2)
        .find(|w| w[0] <= p.value && p.value <= w[1])
        .unwrap();
    assert!(at(cell[0]) < target && at(cell[1]) < target, "{cell:?}");
}

#[test]
fn a_target_just_below_the_true_peak_is_solved() {
    // The default design's torque peaks at 2.59511225 N m near 13.591 mm of ring radius; the
    // nearest grid value, 13.59375 mm, reads 2.59511211. A target between the two is met only
    // near the peak, which no grid value reaches.
    let base = DesignInputs::default();
    let target = 2.5951122;
    let at = |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&base, radius));
    assert!(
        FreeVariable::RingRadius
            .grid()
            .into_iter()
            .all(|x| at(x) < target)
    );
    let p = solved(solve(&base, FreeVariable::RingRadius, target));
    assert!(p.value > 13.5 && p.value < 13.7, "{}", p.value);
    assert!(at(p.value) >= target && at(p.value - 1e-6) < target);
}

#[test]
fn a_target_met_only_where_the_keyway_starts_to_leave_wall_is_found() {
    // A 24 mm bore with a 4 mm keyway leaves hub wall past the key only above 16.05 mm of ring
    // radius (16.05 - 0.05 bond - 12 - 4), where the torque is already falling: 2.46 N m is
    // met just above that edge, missed at the next grid value and met again from about 29 mm.
    // The validity edge is sampled, so the answer is the edge, not the later interval.
    let mut base = DesignInputs::default();
    base.coupling.bore_mm = 24.0;
    base.coupling.keyway_depth_mm = 4.0;
    let target = 2.46;
    let at = |radius: f64| hot_low(&FreeVariable::RingRadius.apply(&base, radius));
    let next = FreeVariable::RingRadius
        .grid()
        .into_iter()
        .find(|&x| x > 16.05)
        .unwrap();
    assert!(at(next) < target && at(29.0) >= target, "{next}");
    let p = solved(solve(&base, FreeVariable::RingRadius, target));
    assert!((p.value - 16.05).abs() <= 1e-6, "{}", p.value);
    assert!(compute_all(&p.inputs).model.hub_wall_past_key_mm > 0.0);
    assert!(p.torque_hot_low_Nm >= target);
}

#[test]
fn short_lengths_where_f_end_is_not_positive_never_count() {
    // Audit M9: with c_end = 0.5 the end-effect factor is 0 or negative below about 4.4 mm
    // (c_end x pole pitch), and so is the torque. A tiny target is met just above that length,
    // never inside the out-of-range region, and the solved design reads "OK".
    let mut base = DesignInputs::default();
    base.coupling.c_end = 0.5;
    let short = compute_all(&FreeVariable::AxialLength.apply(&base, 2.0));
    assert_eq!(short.model.end_effect_check, END_EFFECT_OUT_OF_RANGE);
    assert!(short.metal.torque_hot_low_Nm < 0.0);
    let p = solved(solve(&base, FreeVariable::AxialLength, 1e-6));
    let r = compute_all(&p.inputs);
    assert_eq!(r.model.end_effect_check, "OK");
    let pitch = r.model.pole_pitch_mm;
    assert!(
        p.value > 0.5 * pitch && p.value < 0.5 * pitch + 0.01,
        "{} vs {}",
        p.value,
        0.5 * pitch
    );
}

#[test]
fn nothing_valid_in_the_range_reports_no_best_value() {
    // 25.4 mm wide manual blocks on the default hub fit no even pole count from 4 up (the inner
    // flat is 2 x 10.15 x tan(pi/N) mm): every value is invalid, so there is no best value.
    let mut base = DesignInputs::default();
    base.coupling.magnets.part_inner = String::new();
    base.coupling.magnets.part_outer = String::new();
    base.coupling.magnets.manual_inner_width_mm = 25.4;
    base.coupling.magnets.manual_outer_width_mm = 25.4;
    assert_eq!(
        solve(&base, FreeVariable::MagnetsPerRing, 0.1),
        Ok(SizingOutcome::NotReachable { best: None })
    );
}

#[test]
fn overlapping_arcs_never_count() {
    // Decision A2-4: arcs fit when their blocks do not overlap at the magnet mid-radius. On the
    // default hub 12 arcs of 6.35 mm share a 6.14 mm pitch there (the fill C66 clamps the
    // share to 1 and prices them as if they fitted): they never count, so 2.3 N m (between
    // 10 poles' 2.285 and 12 poles' 2.541 N m) is not reachable and the best is 10 poles.
    let mut arcs = DesignInputs::default();
    arcs.coupling.faceted = 0;
    let twelve = FreeVariable::MagnetsPerRing.apply(&arcs, 12.0);
    let r = compute_all(&twelve);
    let share = pitch_share(
        r.model.inner_width_mm,
        twelve.coupling.inner_back_apothem_mm,
        r.model.inner_thickness_mm,
        12.0,
    );
    assert!(share > 1.0 && r.model.fill_inner == 1.0, "{share}");
    assert!(!blocks_fit(&twelve.coupling, &r.model));
    assert!(r.metal.torque_hot_low_Nm >= 2.3);
    match solve(&arcs, FreeVariable::MagnetsPerRing, 2.3) {
        Ok(SizingOutcome::NotReachable { best: Some(best) }) => assert_eq!(best.value, 10.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn arcs_fit_exactly_at_a_pitch_share_of_one() {
    // The equality edge of the arc rule (a share of at most 1): manual inner arcs exactly one
    // mid-radius pitch wide fit; the next wider double does not.
    let mut d = DesignInputs::default();
    d.coupling.faceted = 0;
    d.coupling.magnets.part_inner = String::new();
    let a = d.coupling.inner_back_apothem_mm;
    let t = d.coupling.magnets.manual_inner_thickness_mm;
    let n = d.coupling.npole as f64;
    let pitch = 2.0 * std::f64::consts::PI * (a + t / 2.0) / n;
    d.coupling.magnets.manual_inner_width_mm = pitch;
    let r = compute_all(&d);
    let share = |r: &magcoupling::DesignResults| {
        pitch_share(r.model.inner_width_mm, a, r.model.inner_thickness_mm, n)
    };
    assert_eq!(share(&r), 1.0, "the equality");
    assert!(blocks_fit(&d.coupling, &r.model));
    d.coupling.magnets.manual_inner_width_mm = pitch.next_up();
    let r = compute_all(&d);
    assert!(share(&r) > 1.0);
    assert!(!blocks_fit(&d.coupling, &r.model));
}

#[test]
fn a_hub_the_keyway_breaks_through_never_counts() {
    // Decision A2-4: a keyway that leaves no hub wall (Calculator C53 <= 0) is not a design.
    // With a 16 mm bore and a 2.5 mm keyway the wall past the key is positive only above
    // 10.55 mm of ring radius (10.55 - 0.05 bond - 8 - 2.5), though the flats fit from
    // 9.77 mm: 0.5 N m is met at that edge.
    let mut base = DesignInputs::default();
    base.coupling.bore_mm = 16.0;
    base.coupling.keyway_depth_mm = 2.5;
    let p = solved(solve(&base, FreeVariable::RingRadius, 0.5));
    assert!(compute_all(&p.inputs).model.hub_wall_past_key_mm > 0.0);
    assert!((p.value - 10.55).abs() <= 1e-6, "{}", p.value);
    let below = FreeVariable::RingRadius.apply(&base, p.value - 1e-6);
    assert!(!is_valid(&below, &compute_all(&below)));
}

#[test]
fn the_hub_wall_counts_only_while_positive() {
    // The equality edge of the hub rule (wall past the keyway > 0): a keyway exactly as deep as
    // the hub wall leaves none and does not count; the next shallower double leaves some.
    let mut d = DesignInputs::default();
    let wall = compute_all(&d).model.hub_wall_mm;
    d.coupling.keyway_depth_mm = wall;
    let r = compute_all(&d);
    assert_eq!(r.model.hub_wall_past_key_mm, 0.0, "the equality");
    assert!(!is_valid(&d, &r));
    d.coupling.keyway_depth_mm = wall.next_down();
    let r = compute_all(&d);
    assert!(r.model.hub_wall_past_key_mm > 0.0);
    assert!(is_valid(&d, &r));
}

#[test]
fn invalid_targets_and_inputs_fail_loudly() {
    for target in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for variable in FreeVariable::ALL {
            match solve(&DesignInputs::default(), variable, target) {
                Err(SizingError::InvalidTarget(t)) => {
                    assert!(t == target || (t.is_nan() && target.is_nan()))
                }
                other => panic!("{target} {variable:?}: {other:?}"),
            }
        }
    }
    let mut bad = DesignInputs::default();
    bad.coupling.max_harmonic = 4; // outside its choices, set on the struct
    bad.metal.face_gap_mm = f64::NAN;
    match solve(&bad, FreeVariable::AxialLength, 2.0) {
        Err(SizingError::InvalidInputs(errors)) => {
            let found: Vec<(&str, &SetErrorKind)> =
                errors.iter().map(|e| (e.path.as_str(), &e.kind)).collect();
            assert_eq!(
                found,
                [
                    (
                        "coupling.max_harmonic",
                        &SetErrorKind::NotAChoice { code: 4 }
                    ),
                    ("metal.face_gap_mm", &SetErrorKind::NotFinite),
                ]
            );
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_default_design_sized_to_its_requirement() {
    // The default design's hot low torque, 2.285 N m, misses the 2.5 N m requirement ("Below hot
    // minimum"). Sized by length it meets it a little above 12.7 mm; sized by radius it meets it
    // with a cup wider than the 43 mm claim; sized by poles it cannot: 12 poles no longer fit
    // the default hub, so the best is the design's own 10.
    let base = DesignInputs::default();
    let required = base.metal.required_min_Nm;
    assert!(hot_low(&base) < required);
    let p = solved(solve(&base, FreeVariable::AxialLength, required));
    assert!(p.value > 12.7 && p.value < 20.0, "{}", p.value);
    assert_eq!(
        compute_all(&p.inputs).metal.hot_min_check,
        "Estimate covers hot min"
    );
    let p = solved(solve(&base, FreeVariable::RingRadius, required));
    assert!(
        compute_all(&p.inputs).metal.diameter_reserve_mm < 0.0,
        "{}",
        p.value
    );
    match solve(&base, FreeVariable::MagnetsPerRing, required) {
        Ok(SizingOutcome::NotReachable { best: Some(best) }) => assert_eq!(best.value, 10.0),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_search_constants_are_the_documented_ones() {
    assert_eq!((SCAN_CELLS, VALUE_TOLERANCE_MM), (64, 1e-9));
    assert_eq!(
        FreeVariable::ALL[0],
        FreeVariable::AxialLength,
        "the default"
    );
    for variable in FreeVariable::ALL {
        let grid = variable.grid();
        let range = variable.range();
        assert_eq!(
            (grid[0], *grid.last().unwrap()),
            (range.min, range.max),
            "{variable:?}"
        );
        if variable != FreeVariable::MagnetsPerRing {
            assert_eq!(grid.len(), SCAN_CELLS + 1, "{variable:?}");
        }
    }
    assert_eq!(
        FreeVariable::AxialLength.path(),
        "coupling.magnets.axial_length_mm"
    );
    assert_eq!(FreeVariable::MagnetsPerRing.path(), "coupling.npole");
    assert_eq!(
        FreeVariable::RingRadius.path(),
        "coupling.inner_back_apothem_mm"
    );
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test sizing 2>&1 | grep -E "^error" | head -2
```

Expected: ``error[E0432]: unresolved imports `magcoupling::engine::model::blocks_fit`, `magcoupling::engine::model::pitch_share` `` and ``error[E0432]: unresolved import `magcoupling::engine::sizing` ``.

- [ ] **Step 3: Add the sizing module and the shared fit rule**

`flat_fits` replaces the two inline comparisons of the Calculator's flat checks (`flat >= width`) and `pitch_share` the two fill quotients (the same operations in the same order), so the checks, the fill and sizing cannot disagree; the texts and every number stay as they are (bit for bit: every result of the default design, the workbook defaults and every differential input set).

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

with:

```markdown
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass, not the housing inputs. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass, not the housing inputs; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/mod.rs`, replace:

```rust
pub mod meta;
pub mod metal_design;
pub mod model;
pub mod sweeps;
pub mod temperature;
pub mod warnings;
```

with:

```rust
pub mod meta;
pub mod metal_design;
pub mod model;
pub mod sizing;
pub mod sweeps;
pub mod temperature;
pub mod warnings;
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    )
}

/// Measured correction only for the prototype's circuit (no iron, same poles, B842SH both rings).
pub fn select_calibration_factor(
    backiron: i64,
```

with:

```rust
    )
}

/// Whether a block fits its polygon flat: the comparison of the flat checks C52 and C59.
fn flat_fits(flat_mm: f64, width_mm: f64) -> bool {
    flat_mm >= width_mm
}

/// A block's share of its pole pitch at the magnet mid-radius: the width over
/// 2π (back apothem + thickness / 2) / N, the fill C66 (inner ring) and C67 (outer ring, from
/// the outer face apothem) before their min(1, ...). Above 1 the blocks overlap there.
pub fn pitch_share(width_mm: f64, back_apothem_mm: f64, thickness_mm: f64, npole: f64) -> f64 {
    width_mm / (2.0 * PI * (back_apothem_mm + thickness_mm / 2.0) / npole)
}

/// Whether both rings' blocks fit (Addendum A decision A2-4). Faceted blocks (`faceted` 1)
/// fit their polygon flats: the Calculator's C52 and C59 checks pass (their comparison,
/// [`flat_fits`]). Arcs (any other code; the flat checks read "n/a (arcs)") fit when neither
/// ring's blocks overlap at the magnet mid-radius: a [`pitch_share`] of at most 1, where C66
/// and C67 would otherwise clamp the fill and price overlapping arcs as if they fitted.
/// Inverse sizing counts only layouts that fit.
pub fn blocks_fit(ci: &CouplingInputs, r: &ModelResults) -> bool {
    if ci.faceted == 1 {
        flat_fits(r.inner_flat_width_mm, r.inner_width_mm)
            && flat_fits(r.outer_flat_width_mm, r.outer_width_mm)
    } else {
        let n = ci.npole as f64;
        let inner = pitch_share(
            r.inner_width_mm,
            ci.inner_back_apothem_mm,
            r.inner_thickness_mm,
            n,
        );
        let outer = pitch_share(
            r.outer_width_mm,
            r.outer_face_apothem_mm,
            r.outer_thickness_mm,
            n,
        );
        inner <= 1.0 && outer <= 1.0
    }
}

/// Measured correction only for the prototype's circuit (no iron, same poles, B842SH both rings).
pub fn select_calibration_factor(
    backiron: i64,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust

    let flat_i = 2.0 * a_i * (PI / N).tan();
    let chk_i = if ci.faceted == 1 {
        if flat_i >= mi.width_mm {
            format!("OK, {} mm slack", fmt_fixed(flat_i - mi.width_mm, 2))
        } else {
            "TOO NARROW: increase apothem or reduce poles".to_owned()
```

with:

```rust

    let flat_i = 2.0 * a_i * (PI / N).tan();
    let chk_i = if ci.faceted == 1 {
        if flat_fits(flat_i, mi.width_mm) {
            format!("OK, {} mm slack", fmt_fixed(flat_i - mi.width_mm, 2))
        } else {
            "TOO NARROW: increase apothem or reduce poles".to_owned()
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    let g_m = A_o - r_face_i;
    let flat_o = 2.0 * A_o * (PI / N).tan();
    let chk_o = if ci.faceted == 1 {
        if flat_o >= mo.width_mm {
            format!(
                "OK, blocks {} mm apart at the faces",
                fmt_fixed(flat_o - mo.width_mm, 2)
```

with:

```rust
    let g_m = A_o - r_face_i;
    let flat_o = 2.0 * A_o * (PI / N).tan();
    let chk_o = if ci.faceted == 1 {
        if flat_fits(flat_o, mo.width_mm) {
            format!(
                "OK, blocks {} mm apart at the faces",
                fmt_fixed(flat_o - mo.width_mm, 2)
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    };
    let R_g = r_face_i + g_m / 2.0;
    let tau_p = 2.0 * PI * R_g / N;
    let al_i = py_min(
        1.0,
        mi.width_mm / (2.0 * PI * (a_i + mi.thickness_mm / 2.0) / N),
    );
    let al_o = py_min(
        1.0,
        mo.width_mm / (2.0 * PI * (A_o + mo.thickness_mm / 2.0) / N),
    );

    let bri = mi.br_T * br_factor(alpha_br, ci.op_temp_C);
    let bro = mo.br_T * br_factor(alpha_br, ci.op_temp_C);
```

with:

```rust
    };
    let R_g = r_face_i + g_m / 2.0;
    let tau_p = 2.0 * PI * R_g / N;
    let al_i = py_min(1.0, pitch_share(mi.width_mm, a_i, mi.thickness_mm, N));
    let al_o = py_min(1.0, pitch_share(mo.width_mm, A_o, mo.thickness_mm, N));

    let bri = mi.br_T * br_factor(alpha_br, ci.op_temp_C);
    let bro = mo.br_T * br_factor(alpha_br, ci.op_temp_C);
```

Create `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/sizing.rs` with exactly this content:

```rust
//! Addendum A1: inverse sizing (Torque → Magnets).
//!
//! Spec A1: inverse sizing "takes a target torque and makes the hot-low torque with
//! production variation (`metal.torque_hot_low_Nm`) meet it, by adjusting ONE free variable
//! the user picks: axial magnet length (default), magnets per ring (discrete; poles stay
//! even), or ring radius. Every other input stays fixed. ... It returns the smallest value
//! that meets the target, or 'not reachable' with the best value achieved inside the
//! variable's slider range. It fails loudly and never extrapolates past the range."
//!
//! **Method.** The torque is not guaranteed monotone in the variable (on the default design
//! it rises with the ring radius to about 14 mm, falls to about 20 mm and rises again), so a
//! continuous variable is sampled in ascending order at the [`SCAN_CELLS`] + 1 ends of equal
//! cells of its slider range, and three refinements find what lies between two samples:
//!
//! - **a crossing**: the first sample that meets is bisected against the sample before it to
//!   [`VALUE_TOLERANCE_MM`]; a range minimum that meets is the answer as it is;
//! - **a peak**: where three consecutive valid samples rise and then do not rise
//!   (T0 < T1 ≥ T2, the middle one missing the target), a golden-section search between the
//!   outer two finds the peak to [`VALUE_TOLERANCE_MM`]; a peak that meets has the crossing
//!   below it bisected (so a meeting interval inside one cell is found), and every peak is
//!   offered as the best value;
//! - **a validity edge**: where validity ([`is_valid`]) changes between two samples, the edge
//!   is bisected to [`VALUE_TOLERANCE_MM`] and its valid side sampled, so a torque largest
//!   where the blocks start to fit, or the keyway starts to leave hub wall, is seen there.
//!
//! Magnets per ring (poles per ring, one block per pole) steps through the even values of its
//! slider, with no refinement. Each value is one [`compute_all`] of the design with only the
//! free variable changed, so the answer is what the forward calculation shows, and every
//! value tried lies inside the slider range.
//!
//! **Residual limits** (stated, not handled): a hump of the torque whose rise and fall both
//! lie inside one cell (0.328 mm of ring radius, 0.778 mm of axial length) leaves no trace on
//! the samples and is not seen, and neither is a validity change that reverts inside one
//! cell. Near a peak the torque is flat, so the peak search finds the peak torque to its
//! floating-point resolution only (about 1e-15 relative): a target within that of the true
//! peak may read "not reachable".
//!
//! **What counts** ([`is_valid`], decision A2-4): the blocks fit ([`blocks_fit`]: faceted
//! blocks their polygon flats, the Calculator's C52 and C59; arcs without overlapping at the
//! magnet mid-radius), the keyway leaves hub wall (C53 > 0), the end-effect factor is in
//! range ([`end_effect_in_range`], audit M9: at short lengths f_end ≤ 0 makes the torque 0 or
//! negative) and the hot-low torque is finite. A value meets when it counts and its hot-low
//! torque is at least the target. "Not reachable" reports the best valid value the search
//! evaluated (the largest hot-low torque among the samples, the validity edges and the
//! refined peaks), or none when no value is valid.
//!
//! The space claim is not a condition: a solution may exceed it, and the housing results show
//! by how much (spec A1: a red callout, not a constraint).

use super::api::{DesignInputs, DesignResults, compute_all};
use super::meta::{SetError, SliderRange, input_rows};
use super::model::{blocks_fit, end_effect_in_range};

/// Equal cells of the coarse scan over a continuous variable's slider range.
pub const SCAN_CELLS: usize = 64;

/// Every bisection and peak search stops when its bracket is this narrow [mm]: a solved value
/// meets the target and lies within it of the crossing its bracket holds.
pub const VALUE_TOLERANCE_MM: f64 = 1e-9;

/// The free variable of inverse sizing (spec A1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreeVariable {
    /// Axial magnet length of both rings (`coupling.magnets.axial_length_mm`): the default.
    AxialLength,
    /// Magnets per ring, one block per pole (`coupling.npole`): even values only.
    MagnetsPerRing,
    /// Ring radius: the inner magnet back apothem (`coupling.inner_back_apothem_mm`,
    /// Calculator C8; report 6.5: the variable starts from this input, not the pole sweep's rule).
    RingRadius,
}

impl FreeVariable {
    /// Every free variable; the first is the default.
    pub const ALL: [FreeVariable; 3] = [
        FreeVariable::AxialLength,
        FreeVariable::MagnetsPerRing,
        FreeVariable::RingRadius,
    ];

    /// The input the variable sets.
    pub const fn path(self) -> &'static str {
        match self {
            FreeVariable::AxialLength => "coupling.magnets.axial_length_mm",
            FreeVariable::MagnetsPerRing => "coupling.npole",
            FreeVariable::RingRadius => "coupling.inner_back_apothem_mm",
        }
    }

    /// The variable's slider range, read from its input's metadata (one source).
    pub fn range(self) -> SliderRange {
        input_rows(&DesignInputs::default())
            .into_iter()
            .find(|row| row.path == self.path())
            .and_then(|row| row.meta.range)
            .expect("every free variable is an input with a slider (tests/sizing.rs)")
    }

    /// The values the coarse pass evaluates, ascending: the [`SCAN_CELLS`] + 1 cell ends of a
    /// continuous variable, or every even whole number of the magnets-per-ring slider.
    pub fn grid(self) -> Vec<f64> {
        let r = self.range();
        match self {
            FreeVariable::MagnetsPerRing => (r.min as i64..=r.max as i64)
                .filter(|n| n % 2 == 0)
                .map(|n| n as f64)
                .collect(),
            FreeVariable::AxialLength | FreeVariable::RingRadius => (0..=SCAN_CELLS)
                .map(|i| {
                    if i == SCAN_CELLS {
                        r.max
                    } else {
                        r.min + (r.max - r.min) * i as f64 / SCAN_CELLS as f64
                    }
                })
                .collect(),
        }
    }

    /// The design `inputs` with this variable set to `value` and every other input kept.
    pub fn apply(self, inputs: &DesignInputs, value: f64) -> DesignInputs {
        let mut design = inputs.clone();
        match self {
            FreeVariable::AxialLength => design.coupling.magnets.axial_length_mm = Some(value),
            FreeVariable::MagnetsPerRing => design.coupling.npole = value as i64,
            FreeVariable::RingRadius => design.coupling.inner_back_apothem_mm = value,
        }
        design
    }
}

/// One value of the free variable, evaluated.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the result names
pub struct SizingPoint {
    /// The free variable's value (magnets per ring as a whole number).
    pub value: f64,
    /// The hot-low torque with production variation there (`metal.torque_hot_low_Nm`).
    pub torque_hot_low_Nm: f64,
    /// The design at that value: the inputs with only the free variable changed.
    pub inputs: DesignInputs,
}

/// What inverse sizing found.
#[derive(Clone, Debug, PartialEq)]
pub enum SizingOutcome {
    /// The smallest value in the slider range that meets the target, within the module docs'
    /// residual limits (a continuous variable to within [`VALUE_TOLERANCE_MM`] of its crossing).
    Solved(SizingPoint),
    /// No value the search evaluated meets the target: the best valid one (the largest
    /// hot-low torque, refined peaks included), or `None` when no value is valid.
    NotReachable { best: Option<SizingPoint> },
}

/// Why inverse sizing refused to run.
#[derive(Clone, Debug, PartialEq)]
pub enum SizingError {
    /// The target is not a positive, finite torque [N·m].
    InvalidTarget(f64),
    /// The inputs fail [`DesignInputs::validate`] (every offending path, in schema order).
    InvalidInputs(Vec<SetError>),
}

/// Whether a design counts for inverse sizing (decision A2-4): its blocks fit
/// ([`blocks_fit`]), the keyway leaves hub wall (`model.hub_wall_past_key_mm` > 0, Calculator
/// C53), the end-effect factor is in range ([`end_effect_in_range`]) and the hot-low torque is
/// finite. `results` is `compute_all(design)`.
pub fn is_valid(design: &DesignInputs, results: &DesignResults) -> bool {
    blocks_fit(&design.coupling, &results.model)
        && results.model.hub_wall_past_key_mm > 0.0
        && end_effect_in_range(results.model.f_end)
        && results.metal.torque_hot_low_Nm.is_finite()
}

/// Makes the hot-low torque of `inputs` meet `target_Nm` by adjusting `variable` alone (spec
/// A1; the module docs give the method, its limits and what counts).
#[allow(non_snake_case)] // unit suffix
pub fn solve(
    inputs: &DesignInputs,
    variable: FreeVariable,
    target_Nm: f64,
) -> Result<SizingOutcome, SizingError> {
    if !(target_Nm.is_finite() && target_Nm > 0.0) {
        return Err(SizingError::InvalidTarget(target_Nm));
    }
    inputs.validate().map_err(SizingError::InvalidInputs)?;
    let mut evaluate = |value: f64| {
        let design = variable.apply(inputs, value);
        let results = compute_all(&design);
        Sample {
            value,
            torque_Nm: results.metal.torque_hot_low_Nm,
            valid: is_valid(&design, &results),
            design,
        }
    };
    let search = Search::new(
        &mut evaluate,
        target_Nm,
        variable != FreeVariable::MagnetsPerRing,
    );
    Ok(match search.over(&variable.grid()) {
        Found::Meets(sample) => SizingOutcome::Solved(sample.into_point()),
        Found::Best(best) => SizingOutcome::NotReachable {
            best: best.map(Sample::into_point),
        },
    })
}

/// One evaluated value: its torque, whether it counts ([`is_valid`]), and the design there
/// (the inputs in [`solve`]; nothing in the search's unit tests).
#[derive(Clone, Debug)]
#[allow(non_snake_case)] // unit suffix
struct Sample<P> {
    value: f64,
    torque_Nm: f64,
    valid: bool,
    design: P,
}

impl<P> Sample<P> {
    #[allow(non_snake_case)] // unit suffix
    fn meets(&self, target_Nm: f64) -> bool {
        self.valid && self.torque_Nm >= target_Nm
    }

    /// The torque the peak search compares: a value that does not count has none at all.
    fn height(&self) -> f64 {
        if self.valid {
            self.torque_Nm
        } else {
            f64::NEG_INFINITY
        }
    }
}

impl Sample<DesignInputs> {
    fn into_point(self) -> SizingPoint {
        SizingPoint {
            value: self.value,
            torque_hot_low_Nm: self.torque_Nm,
            inputs: self.design,
        }
    }
}

/// What the search found.
#[derive(Debug)]
enum Found<P> {
    /// The smallest value that meets, within the module docs' limits.
    Meets(Sample<P>),
    /// Nothing met: the best valid sample, if any.
    Best(Option<Sample<P>>),
}

/// The search of [`solve`] over an ascending grid (the module docs' method). It takes the
/// evaluation as a function so its unit tests can drive it with plain functions.
#[allow(non_snake_case)] // unit suffix
struct Search<'a, P> {
    evaluate: &'a mut dyn FnMut(f64) -> Sample<P>,
    target_Nm: f64,
    /// A continuous variable: bisect crossings, search peaks, locate validity edges.
    refine: bool,
    /// The sample taken last: the lower end of a crossing's bisection.
    previous: Option<Sample<P>>,
    /// The last (up to) three valid samples of the current run of valid samples.
    window: Vec<Sample<P>>,
    /// The valid sample with the largest torque so far (the first of equals).
    best: Option<Sample<P>>,
}

impl<'a, P: Clone> Search<'a, P> {
    #[allow(non_snake_case)] // unit suffix
    fn new(evaluate: &'a mut dyn FnMut(f64) -> Sample<P>, target_Nm: f64, refine: bool) -> Self {
        Search {
            evaluate,
            target_Nm,
            refine,
            previous: None,
            window: Vec::new(),
            best: None,
        }
    }

    /// Runs the search over `grid` (ascending).
    fn over(mut self, grid: &[f64]) -> Found<P> {
        for &value in grid {
            let sample = (self.evaluate)(value);
            if let Some(found) = self.grid_sample(sample) {
                return Found::Meets(found);
            }
        }
        Found::Best(self.best)
    }

    /// Takes the next grid sample, after sampling the validity edge between it and the
    /// previous sample if validity changed there (a continuous variable only).
    fn grid_sample(&mut self, sample: Sample<P>) -> Option<Sample<P>> {
        let edge_before = match &self.previous {
            Some(previous) if self.refine && previous.valid != sample.valid => {
                Some(previous.clone())
            }
            _ => None,
        };
        if let Some(previous) = edge_before {
            let changed = !previous.valid;
            let (lo, hi) = self.bisect(previous.clone(), sample.clone(), |s| s.valid == changed);
            if previous.valid {
                // Valid, then not: the largest valid value, unless the bisection never moved.
                if lo.value > previous.value
                    && let Some(found) = self.take(lo)
                {
                    return Some(found);
                }
            } else {
                // Not valid, then valid: the smallest valid value. The invalid end nearest it
                // is the lower end of any crossing, so an edge that meets is the answer as is.
                self.previous = Some(lo);
                if hi.value < sample.value
                    && let Some(found) = self.take(hi)
                {
                    return Some(found);
                }
            }
        }
        self.take(sample)
    }

    /// Takes one sample, in ascending order: returns the answer when it meets (bisected
    /// against the sample before it); otherwise keeps the best and searches a peak the last
    /// three valid samples show.
    fn take(&mut self, sample: Sample<P>) -> Option<Sample<P>> {
        if sample.meets(self.target_Nm) {
            return Some(match self.previous.take() {
                Some(lo) if self.refine => self.crossing(lo, sample),
                _ => sample,
            });
        }
        if !sample.valid {
            self.window.clear();
            self.previous = Some(sample);
            return None;
        }
        self.offer(&sample);
        self.window.push(sample.clone());
        if self.window.len() > 3 {
            self.window.remove(0);
        }
        let hump = match &self.window[..] {
            [rise, top, fall]
                if self.refine
                    && rise.torque_Nm < top.torque_Nm
                    && top.torque_Nm >= fall.torque_Nm =>
            {
                Some((rise.clone(), top.clone(), fall.value))
            }
            _ => None,
        };
        if let Some((rise, top, fall)) = hump {
            let peak = self.peak(rise.value, top, fall);
            self.offer(&peak);
            if peak.meets(self.target_Nm) {
                return Some(self.crossing(rise, peak));
            }
        }
        self.previous = Some(sample);
        None
    }

    /// Keeps `sample` as the best if it counts and its torque beats the best so far.
    fn offer(&mut self, sample: &Sample<P>) {
        if sample.valid
            && self
                .best
                .as_ref()
                .is_none_or(|b| sample.torque_Nm > b.torque_Nm)
        {
            self.best = Some(sample.clone());
        }
    }

    /// The smallest value that meets between `lo` (does not meet) and `hi` (meets).
    #[allow(non_snake_case)] // unit suffix
    fn crossing(&mut self, lo: Sample<P>, hi: Sample<P>) -> Sample<P> {
        let target_Nm = self.target_Nm;
        self.bisect(lo, hi, |s| s.meets(target_Nm)).1
    }

    /// Bisects between `lo`, which lacks the property `has`, and `hi`, which has it, until the
    /// two are [`VALUE_TOLERANCE_MM`] apart or adjacent doubles; returns both ends.
    fn bisect(
        &mut self,
        mut lo: Sample<P>,
        mut hi: Sample<P>,
        has: impl Fn(&Sample<P>) -> bool,
    ) -> (Sample<P>, Sample<P>) {
        while hi.value - lo.value > VALUE_TOLERANCE_MM {
            let mid = lo.value + (hi.value - lo.value) / 2.0;
            if mid <= lo.value || mid >= hi.value {
                break; // adjacent doubles
            }
            let sample = (self.evaluate)(mid);
            if has(&sample) {
                hi = sample;
            } else {
                lo = sample;
            }
        }
        (lo, hi)
    }

    /// The golden-section search for the largest torque between `lo` and `hi`, which hold the
    /// sample `top` whose torque is at least both ends': narrows the bracket to
    /// [`VALUE_TOLERANCE_MM`] and returns the best sample it saw (`top` if none beats it).
    fn peak(&mut self, mut lo: f64, top: Sample<P>, mut hi: f64) -> Sample<P> {
        let ratio = (5.0_f64.sqrt() - 1.0) / 2.0; // 1 / golden ratio
        let mut best = top;
        let mut c = hi - (hi - lo) * ratio;
        let mut d = lo + (hi - lo) * ratio;
        let mut at_c = (self.evaluate)(c);
        let mut at_d = (self.evaluate)(d);
        loop {
            for sample in [&at_c, &at_d] {
                if sample.height() > best.height() {
                    best = sample.clone();
                }
            }
            if hi - lo <= VALUE_TOLERANCE_MM || !(lo < c && c < d && d < hi) {
                return best;
            }
            if at_c.height() >= at_d.height() {
                // The peak lies in [lo, d]: d's point becomes c's, a new c.
                hi = d;
                d = c;
                at_d = at_c;
                c = hi - (hi - lo) * ratio;
                at_c = (self.evaluate)(c);
            } else {
                // The peak lies in [c, hi]: c's point becomes d's, a new d.
                lo = c;
                c = d;
                at_c = at_d;
                d = lo + (hi - lo) * ratio;
                at_d = (self.evaluate)(d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! The search alone, driven by plain functions of the value (no model): each refinement,
    //! the stated limit and the evaluation count.
    use super::*;

    /// The grid 0, 1, ..., 10 (cells of 1).
    fn grid() -> Vec<f64> {
        (0..=10).map(f64::from).collect()
    }

    /// Runs the search of `torque` (`valid` says where it counts) and records every value it
    /// evaluated.
    fn run(
        torque: impl Fn(f64) -> f64,
        valid: impl Fn(f64) -> bool,
        target: f64,
        refine: bool,
    ) -> (Found<()>, Vec<f64>) {
        let mut seen = Vec::new();
        let mut evaluate = |value: f64| {
            seen.push(value);
            Sample {
                value,
                torque_Nm: torque(value),
                valid: valid(value),
                design: (),
            }
        };
        let found = Search::new(&mut evaluate, target, refine).over(&grid());
        (found, seen)
    }

    fn met(found: Found<()>) -> Sample<()> {
        match found {
            Found::Meets(sample) => sample,
            other => panic!("expected a value that meets, got {other:?}"),
        }
    }

    fn always(_: f64) -> bool {
        true
    }

    #[test]
    fn a_crossing_is_bisected_to_the_tolerance() {
        let (found, seen) = run(|x| x, always, 4.25, true);
        let s = met(found);
        assert!(
            s.torque_Nm >= 4.25 && s.value - 4.25 <= VALUE_TOLERANCE_MM,
            "{}",
            s.value
        );
        assert!(
            seen.iter().all(|v| (0.0..=10.0).contains(v)),
            "never outside the grid"
        );
    }

    #[test]
    fn a_first_value_that_meets_is_the_answer_as_it_is() {
        let (found, seen) = run(|x| x, always, -1.0, true);
        assert_eq!(met(found).value, 0.0);
        assert_eq!(seen, [0.0]);
    }

    #[test]
    fn a_hump_inside_one_cell_is_found_at_its_rising_crossing() {
        // Peak 1 at 4.5; the samples 4 and 5 read 0.75 and miss a target of 0.9.
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        let (found, seen) = run(torque, always, 0.9, true);
        let s = met(found);
        let crossing = 4.5 - 0.1_f64.sqrt();
        assert!(s.torque_Nm >= 0.9, "{}", s.torque_Nm);
        assert!(
            (s.value - crossing).abs() < 1e-8,
            "{} vs {crossing}",
            s.value
        );
        assert!(seen.iter().all(|v| (0.0..=10.0).contains(v)));
    }

    #[test]
    fn not_reachable_reports_the_refined_peak() {
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        match run(torque, always, 2.0, true).0 {
            Found::Best(Some(best)) => {
                assert!((best.value - 4.5).abs() < 1e-6, "{}", best.value);
                assert!(best.torque_Nm > 1.0 - 1e-12, "{}", best.torque_Nm);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_torque_falling_from_where_values_start_to_count_is_met_at_that_edge() {
        // Counts from 3.3 on, falling: the sample 4 (6.0) misses 6.65, the edge (6.7) meets.
        let (found, _) = run(|x| 10.0 - x, |x| x >= 3.3, 6.65, true);
        let s = met(found);
        assert!(
            s.valid && s.value >= 3.3 && s.value - 3.3 <= VALUE_TOLERANCE_MM,
            "{}",
            s.value
        );
    }

    #[test]
    fn a_torque_rising_to_where_values_stop_counting_is_met_before_that_edge() {
        // Counts up to 6.7, rising: the sample 6 misses 6.65, the sample 7 does not count.
        let (found, _) = run(|x| x, |x| x <= 6.7, 6.65, true);
        let s = met(found);
        assert!(
            (s.value - 6.65).abs() < 1e-8 && s.torque_Nm >= 6.65,
            "{}",
            s.value
        );
    }

    #[test]
    fn nothing_counting_reports_no_best_value() {
        let (found, seen) = run(|x| x, |_| false, 1.0, true);
        assert!(matches!(found, Found::Best(None)), "{found:?}");
        assert_eq!(seen, grid(), "no edge, no peak: only the grid");
    }

    #[test]
    fn a_plateau_searches_no_peak() {
        // Equal torques never rise, so no peak search runs: one evaluation per grid value,
        // and the best is the first of equals.
        let (found, seen) = run(|_| 1.0, always, 2.0, true);
        assert_eq!(seen, grid());
        match found {
            Found::Best(Some(best)) => assert_eq!(best.value, 0.0),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn stepping_evaluates_the_grid_only() {
        // The discrete variable (magnets per ring): no bisection, peak or edge between values.
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        let (found, seen) = run(torque, |x| x >= 3.3, 0.9, false);
        assert_eq!(seen, grid());
        assert!(
            matches!(found, Found::Best(Some(ref b)) if b.value == 4.0),
            "{found:?}"
        );
        let (found, _) = run(|x| x, always, 4.25, false);
        assert_eq!(met(found).value, 5.0);
    }

    #[test]
    fn a_hump_whose_rise_and_fall_are_inside_one_cell_is_the_stated_limit() {
        // The module docs' residual limit, pinned: a spike between 4 and 5 over a rising line
        // leaves every sample rising, so no peak search runs and the spike is not seen.
        let torque = |x: f64| x / 10.0 + if (x - 4.4).abs() < 0.05 { 5.0 } else { 0.0 };
        let (found, seen) = run(torque, always, 3.0, true);
        assert!(
            matches!(found, Found::Best(Some(ref b)) if b.value == 10.0),
            "{found:?}"
        );
        assert_eq!(seen, grid());
    }

    #[test]
    fn a_peak_search_is_bounded() {
        // One peak search: about 45 evaluations for a bracket of 2 narrowed to 1e-9, never
        // more than the golden ratio allows (plus the grid and a crossing's bisection).
        let torque = |x: f64| 1.0 - (x - 4.5) * (x - 4.5);
        let (_, seen) = run(torque, always, 2.0, true);
        let golden_steps = (VALUE_TOLERANCE_MM / 2.0).ln() / ((5.0_f64.sqrt() - 1.0) / 2.0).ln();
        assert!(
            seen.len() <= grid().len() + 2 + golden_steps.ceil() as usize,
            "{}",
            seen.len()
        );
    }
}
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test sizing 2>&1 | grep "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib sizing 2>&1 | grep "test result"
```

Expected: `test result: ok. 17 passed`, then `test result: ok. 11 passed` (the search's unit tests).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `133 passed`, `tests\sizing.rs` 17 passed, `tests\robustness.rs` 11 passed, the others as in Task 5.

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task6.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/sizing.rs magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/tests/sizing.rs magcoupling-rs/tests/robustness.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): inverse sizing, Torque to Magnets (spec A1)

One free variable (axial length by default, magnets per ring even only, ring
radius): a 64-cell scan of its slider range refined at the first crossing, at
each peak (golden section) and at each validity edge; the smallest value that
counts (blocks fit: flats, or arcs that do not overlap; the keyway leaves hub
wall; f_end > 0; decision A2-4) and meets the target, or NotReachable with the
best valid value found. Never outside the range; loud errors for a bad target
or invalid inputs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 7: Housing autofit and the space claim (spec A1; decisions 27, 28)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

Spec A1: "Housing autofit. Cup, sleeve, liner, cap and the axial stack are always derived from the magnet layout and the calculator's
existing clearance and wall rules, in both modes"; "Space claim. The envelope (43 mm diameter × 35 mm overall length, and the 20 mm
large-diameter bay ...). Exceeding it shows a red callout on the view naming the overshoot in mm per axis, plus a red dashboard badge";
testing: "the envelope-exceeded callout triggers exactly when a derived dimension exceeds the space claim, per axis". Decision 28 A
limits autofit to report 6.5's classes D (already derived: cup OD, pocket radius, wall at the flats, sleeve, liner and endplate
diameters, all results already) and R; decision 27 A keeps the cup wall an input with the rule's 2.0 mm as a suggestion. So this task
adds the suggestion (one computation: the verdict's advice quotes the same number, report 6.5's "no second evaluation of the rule") and
the space claim, read from the workbook's reserves (claim − dimension; exceeded exactly when below 0, so at the claim is inside), with
NaN reported as unknown rather than inside. The badge quotes each overshoot to 0.01 mm and at least 0.01 mm (a dimension 0.001 mm
past its claim reads "0.01 mm over", never "0.00 mm over"; Excel's CEILING through `compat::ceiling` would also do that, but its 1e-12
guard sits on the quotient, so two ulps of noise at 43 mm, 2.8000000000000114, would print "2.81"), and an axis that is not a number
reads "unknown" beside the exceeded ones instead of hiding them ("Space claim unknown" only when nothing is exceeded). Both are part of
every `compute_all`, so a design inverse sizing returns shows its overshoot too ("in both modes"). The class N dimensions (cup depth,
retainer span, hub, web, boss, endplates, cap) stay inputs in this task: no rule (decision 28; the M41 thread and the boss OD are M4
questions). So here no dimension the space claim reads moves with the axial length (the axial stack C134 and the large-diameter stack
C137 are sums of class N inputs), and a length-sized design reads "Inside the space claim" at any length: the default design sized to
its own 2.5 N·m requirement gets 13.77 mm magnets on the 13.0 mm hub, and sized to 9.9 N·m 50.62 mm magnets in the 15.5 mm cup, both
inside. `a_length_sized_design_reads_inside_the_space_claim_at_any_length` pins that behaviour until Task 7b, which implements the
user's choice on A2-8 (option B: the hub, the cup cavity and the retainer span follow the length override) and replaces that test on
purpose.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs` (`HousingResults` (Rust-only `housing.*`), `INSIDE_THE_SPACE_CLAIM`, `SPACE_CLAIM_UNKNOWN`, `compute`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/materials.rs` (Rust-only `materials.cup_wall_suggested_mm`, `NO_WALL_RULE`; the advice quotes the same number; test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs` (the `housing` result group)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/mod.rs` (`pub mod housing;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs` (the space claim per axis, in both modes; a tiny overshoot; a length-sized design (option (a) of decision A2-8, replaced in Task 7b))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (layout and tests rows)

**Interfaces:**
- Consumes: `MetalDesignResults::{diameter_reserve_mm, axial_reserve_mm, large_dia_reserve_mm}` (C136, C139, C138: claim minus dimension), `rotating_od_mm`, `axial_stack_mm`, `large_dia_stack_mm` (C135, C134, C137); the claims `metal.max_diameter_mm`, `max_overall_axial_mm`, `max_large_dia_axial_mm` (C131, C130, C129); `materials::compute`'s wall rule (`ceiling(t_bi, 0.1)`); Task 6's `sizing::solve` (the test of a sized design).
- Produces:
  - `materials.cup_wall_suggested_mm: NumOrText` (Rust-only; `Num(ceiling(t_bi, 0.1))`, or `Text(NO_WALL_RULE)` = `"n/a"` exactly when the check reads "No back iron");
  - module `engine::housing`: `pub struct HousingResults { pub diameter_overshoot_mm: f64, pub length_overshoot_mm: f64, pub bay_overshoot_mm: f64, pub space_claim_check: String }` (every field Rust-only), `pub const INSIDE_THE_SPACE_CLAIM: &str`, `pub const SPACE_CLAIM_UNKNOWN: &str`, `pub fn compute(mdr: &MetalDesignResults) -> HousingResults`;
  - `DesignResults::housing` (after `warnings`).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
    }

    #[test]
    fn e9_no_back_iron_replaces_the_wall_advice_only_at_code_0() {
        let mat = MaterialsInputs::default();
        let too_thin = "Too thin: raise Metal design C122 to at least 2.0 mm";
```

with:

```rust
    }

    #[test]
    fn the_wall_suggestion_is_the_number_the_advice_quotes() {
        // Addendum A1 autofit, decision 27: the wall stays an input and the rule's wall is a
        // suggestion, the back-iron need rounded up to 0.1 mm (the advice's own number).
        let mat = MaterialsInputs::default();
        let r = compute(&mat, 1.90415278222222, 1.8, 1, &parts(), Deviations::NONE);
        assert_eq!(r.cup_wall_suggested_mm, NumOrText::Num(2.0));
        assert_eq!(
            r.cup_wall_check,
            "Too thin: raise Metal design C122 to at least 2.0 mm"
        );
        // The suggestion does not depend on the wall, and a need of exactly 1.9 mm stays 1.9 mm
        // (the 1e-12 guard of `ceiling`): 19 steps of 0.1, as Excel's CEILING gives it.
        let r = compute(&mat, 1.90415278222222, 2.5, 1, &parts(), Deviations::NONE);
        assert_eq!(
            (r.cup_wall_suggested_mm, r.cup_wall_check.as_str()),
            (NumOrText::Num(2.0), "OK")
        );
        let r = compute(&mat, 1.9, 1.8, 1, &parts(), Deviations::NONE);
        assert_eq!(r.cup_wall_suggested_mm, NumOrText::Num(19.0 * 0.1));
        assert_eq!(
            r.cup_wall_check,
            "Too thin: raise Metal design C122 to at least 1.9 mm"
        );
        // No back iron (E9): no magnetic rule, as the check reads "No back iron"; without E9 the
        // workbook still advises a wall, and so does the suggestion.
        let e9 = Deviations::only(DeviationId::E9);
        let r = compute(&mat, 1.90415278222222, 1.8, 0, &parts(), e9);
        assert_eq!(
            (r.cup_wall_suggested_mm, r.cup_wall_check.as_str()),
            (NumOrText::Text(NO_WALL_RULE), "No back iron")
        );
        let r = compute(&mat, 1.90415278222222, 1.8, 0, &parts(), Deviations::NONE);
        assert_eq!(r.cup_wall_suggested_mm, NumOrText::Num(2.0));
    }

    #[test]
    fn e9_no_back_iron_replaces_the_wall_advice_only_at_code_0() {
        let mat = MaterialsInputs::default();
        let too_thin = "Too thin: raise Metal design C122 to at least 2.0 mm";
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::meta::SetErrorKind;
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
```

with:

```rust

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::housing::{INSIDE_THE_SPACE_CLAIM, SPACE_CLAIM_UNKNOWN};
use magcoupling::engine::meta::NumOrText;
use magcoupling::engine::meta::SetErrorKind;
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust
        "coupling.inner_back_apothem_mm"
    );
}
```

with:

```rust
        "coupling.inner_back_apothem_mm"
    );
}

#[test]
fn the_default_design_is_inside_its_space_claim() {
    // Report 6.5: rotating OD 42.8 mm (set by the cap) of 43; stack 31.8 of 35; large-diameter
    // stack 18.8 of the 20 mm bay. The autofit suggests the rule's 2.0 mm cup wall (decision 27).
    let r = compute_all(&DesignInputs::default());
    let h = &r.housing;
    assert_eq!(
        (
            h.diameter_overshoot_mm,
            h.length_overshoot_mm,
            h.bay_overshoot_mm
        ),
        (0.0, 0.0, 0.0)
    );
    assert_eq!(h.space_claim_check, INSIDE_THE_SPACE_CLAIM);
    assert!((r.metal.diameter_reserve_mm - 0.2).abs() < 1e-12);
    assert!((r.metal.axial_reserve_mm - 3.2).abs() < 1e-12);
    assert!((r.metal.large_dia_reserve_mm - 1.2).abs() < 1e-12);
    assert_eq!(r.materials.cup_wall_suggested_mm, NumOrText::Num(2.0));
}

#[test]
fn the_space_claim_is_exceeded_exactly_when_a_dimension_exceeds_it_per_axis() {
    // Spec "Addendum testing": the envelope-exceeded callout triggers exactly when a derived
    // dimension exceeds the space claim, per axis. Each claim is put at its dimension exactly
    // (the comparison at equality, asserted first), then 0.5 mm under and over it.
    type Claim = fn(&mut DesignInputs, f64);
    type Read = fn(&magcoupling::DesignResults) -> (f64, f64); // (dimension, overshoot)
    let axes: [(&str, Claim, Read); 3] = [
        (
            "diameter",
            |i, x| i.metal.max_diameter_mm = x,
            |r| (r.metal.rotating_od_mm, r.housing.diameter_overshoot_mm),
        ),
        (
            "overall length",
            |i, x| i.metal.max_overall_axial_mm = x,
            |r| (r.metal.axial_stack_mm, r.housing.length_overshoot_mm),
        ),
        (
            "large-diameter bay",
            |i, x| i.metal.max_large_dia_axial_mm = x,
            |r| (r.metal.large_dia_stack_mm, r.housing.bay_overshoot_mm),
        ),
    ];
    for (axis, set_claim, read) in axes {
        let (dimension, _) = read(&compute_all(&DesignInputs::default()));
        let at_claim = |claim: f64| {
            let mut inputs = DesignInputs::default();
            set_claim(&mut inputs, claim);
            compute_all(&inputs)
        };
        let r = at_claim(dimension);
        assert_eq!(
            read(&r).0,
            dimension,
            "{axis}: the claim sits at the dimension"
        );
        assert_eq!(read(&r).1, 0.0, "{axis}: at the claim is inside");
        assert_eq!(
            r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM,
            "{axis}"
        );
        let r = at_claim(dimension - 0.5);
        assert_eq!(read(&r).1, dimension - (dimension - 0.5), "{axis}");
        assert_eq!(
            r.housing.space_claim_check,
            format!("Exceeds the space claim: {axis} 0.50 mm over"),
            "only this axis"
        );
        let r = at_claim(dimension + 0.5);
        assert_eq!(read(&r).1, 0.0, "{axis}");
        assert_eq!(
            r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM,
            "{axis}"
        );
    }
}

#[test]
fn a_sized_design_shows_its_overshoot() {
    // Both modes: the space claim is part of every forward calculation, so a design inverse
    // sizing returns shows it too. Sized by ring radius to the 2.5 N m requirement, the cup
    // grows past the 43 mm claim; the axial stack does not move.
    let base = DesignInputs::default();
    let p = solved(solve(
        &base,
        FreeVariable::RingRadius,
        base.metal.required_min_Nm,
    ));
    let r = compute_all(&p.inputs);
    let over = r.metal.rotating_od_mm - base.metal.max_diameter_mm;
    assert!(over > 0.0, "{over}");
    assert_eq!(r.housing.diameter_overshoot_mm, over);
    assert_eq!(
        (r.housing.length_overshoot_mm, r.housing.bay_overshoot_mm),
        (0.0, 0.0)
    );
    assert!(
        r.housing
            .space_claim_check
            .starts_with("Exceeds the space claim: diameter "),
        "{}",
        r.housing.space_claim_check
    );
}

#[test]
fn several_axes_are_named_in_order_and_nan_is_unknown() {
    let mut inputs = DesignInputs::default();
    inputs.metal.max_diameter_mm = 40.0;
    inputs.metal.max_large_dia_axial_mm = 18.0;
    let h = compute_all(&inputs).housing;
    assert_eq!(
        h.space_claim_check,
        "Exceeds the space claim: diameter 2.80 mm over, large-diameter bay 0.80 mm over"
    );
    // A claim that is not a number (set on the struct: validate() names it) is not "inside",
    // and it hides no known overshoot: the axis reads unknown among the exceeded ones.
    inputs.metal.max_overall_axial_mm = f64::NAN;
    let h = compute_all(&inputs).housing;
    assert!(h.length_overshoot_mm.is_nan());
    assert_eq!(
        h.space_claim_check,
        "Exceeds the space claim: diameter 2.80 mm over, overall length unknown, \
         large-diameter bay 0.80 mm over"
    );
    // With nothing exceeded, a NaN axis makes the whole claim unknown.
    let mut inputs = DesignInputs::default();
    inputs.metal.max_overall_axial_mm = f64::NAN;
    assert_eq!(
        compute_all(&inputs).housing.space_claim_check,
        SPACE_CLAIM_UNKNOWN
    );
}

#[test]
fn a_tiny_overshoot_reads_at_least_a_hundredth() {
    // Any dimension past its claim reads at least "0.01 mm over", never "0.00 mm over": each
    // claim 0.001 mm under its dimension (the overshoot itself stays exact).
    type Claim = fn(&mut DesignInputs, f64);
    let axes: [(&str, Claim, f64); 3] = [
        ("diameter", |i, x| i.metal.max_diameter_mm = x, 42.8),
        (
            "overall length",
            |i, x| i.metal.max_overall_axial_mm = x,
            31.8,
        ),
        (
            "large-diameter bay",
            |i, x| i.metal.max_large_dia_axial_mm = x,
            18.8,
        ),
    ];
    let base = compute_all(&DesignInputs::default());
    let dimensions = [
        base.metal.rotating_od_mm,
        base.metal.axial_stack_mm,
        base.metal.large_dia_stack_mm,
    ];
    for ((axis, set_claim, nominal), dimension) in axes.into_iter().zip(dimensions) {
        assert!((dimension - nominal).abs() < 1e-12, "{axis}: {dimension}");
        let mut inputs = DesignInputs::default();
        set_claim(&mut inputs, dimension - 0.001);
        let h = compute_all(&inputs).housing;
        let over = [
            h.diameter_overshoot_mm,
            h.length_overshoot_mm,
            h.bay_overshoot_mm,
        ];
        assert!(
            over.iter().any(|&o| o > 0.0 && o < 0.005),
            "{axis}: {over:?}"
        );
        assert_eq!(
            h.space_claim_check,
            format!("Exceeds the space claim: {axis} 0.01 mm over"),
            "{axis}"
        );
    }
}

#[test]
fn a_length_sized_design_reads_inside_the_space_claim_at_any_length() {
    // Decision A2-8 as recommended (no engine rule): the axial length moves no dimension the
    // space claim reads (the axial stack C134 and the large-diameter stack C137 are sums of
    // class N inputs), so a length-sized design reads "Inside the space claim" at any length.
    // Sized to its own 2.5 N m requirement the default design's magnets (13.77 mm) outgrow
    // the 13.0 mm hub; sized to 9.9 N m (50.6 mm) they outgrow the 15.5 mm cup and the
    // 14.5 mm retainer span. The override's help says to recheck them; M4 decides the rule.
    let base = DesignInputs::default();
    for (target, longer_than) in [
        (base.metal.required_min_Nm, base.metal.hub_length_mm),
        (9.9, base.metal.cup_depth_mm),
    ] {
        let p = solved(solve(&base, FreeVariable::AxialLength, target));
        assert!(p.value > longer_than, "{target}: {}", p.value);
        let r = compute_all(&p.inputs);
        assert_eq!(
            r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM,
            "{target}"
        );
        assert_eq!(
            r.metal.axial_stack_mm,
            compute_all(&base).metal.axial_stack_mm
        );
    }
    let p = solved(solve(&base, FreeVariable::AxialLength, 9.9));
    assert!(p.value > 50.0 && p.value > base.metal.retainer_span_mm);
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test sizing 2>&1 | grep -E "^error" | head -2
```

Expected: ``error[E0432]: unresolved import `magcoupling::engine::housing` ``.

- [ ] **Step 3: Add the wall suggestion and the space claim**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
| `src/engine/temperature.rs` | `Temperature design` sheet: 7 input groups (37 cells), the adhesive table `ADHESIVES` with `selected_adhesive`, `TemperatureLinks`, `TemperatureResults` (130 cells) |
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
```

with:

```markdown
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27) |
| `src/engine/temperature.rs` | `Temperature design` sheet: 7 input groups (37 cells), the adhesive table `ADHESIVES` with `selected_adhesive`, `TemperatureLinks`, `TemperatureResults` (130 cells) |
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, and `space_claim_check`) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass, not the housing inputs; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass, not the housing inputs; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; a length-sized design reads inside the space claim at any length (decision A2-8: no rule reads the hub, cup depth or retainer span). |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
#[cfg(feature = "workbook-parity")]
use super::deviations::{REGISTRY, restore_workbook_defaults};
use super::grades;
use super::material_library;
use super::materials::{self, MaterialsInputs, MaterialsResults};
use super::meta::{ResultSet, SetError, TableLayout, Value, inputs, results, validate};
```

with:

```rust
#[cfg(feature = "workbook-parity")]
use super::deviations::{REGISTRY, restore_workbook_defaults};
use super::grades;
use super::housing::{self, HousingResults};
use super::material_library;
use super::materials::{self, MaterialsInputs, MaterialsResults};
use super::meta::{ResultSet, SetError, TableLayout, Value, inputs, results, validate};
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
            temperature: TemperatureResults,
            clamps: ClampResults,
            warnings: WarningResults,
        }
        tables {
            gap_sweep: SweepRow => TableLayout::RowsDown { sheet: "Gap sweep", first_row: 6 },
```

with:

```rust
            temperature: TemperatureResults,
            clamps: ClampResults,
            warnings: WarningResults,
            housing: HousingResults,
        }
        tables {
            gap_sweep: SweepRow => TableLayout::RowsDown { sheet: "Gap sweep", first_row: 6 },
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        magnet_cte_per_C: ti.mismatch.ndfeb_cte_per_C,
    });

    let alloy = if inputs.clamps.alloy == 1 {
        &materials::AL7075
    } else {
```

with:

```rust
        magnet_cte_per_C: ti.mismatch.ndfeb_cte_per_C,
    });

    // Addendum A1: the space claim, from the derived dimensions (both modes).
    let housing = housing::compute(&mdr);

    let alloy = if inputs.clamps.alloy == 1 {
        &materials::AL7075
    } else {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        temperature: temp,
        clamps: clr,
        warnings: warn,
        gap_sweep: gap,
        pole_sweep: pole,
    }
```

with:

```rust
        temperature: temp,
        clamps: clr,
        warnings: warn,
        housing,
        gap_sweep: gap,
        pole_sweep: pole,
    }
```

Create `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs` with exactly this content:

```rust
//! Addendum A1: housing autofit and the space claim.
//!
//! **Autofit** (report 6.5 classes; decisions 27 and 28). Class D, the housing dimensions
//! the calculator already derives from the magnet layout (the cup OD `model.cup_od_mm`, the
//! pocket corner radius, the ring wall at the flats, the sleeve and liner diameters, the
//! endplate OD), are results: they follow every change of the layout, in both modes. Class R:
//! the cup wall at the corners stays an input, and the wall rule's value is the suggestion
//! `materials.cup_wall_suggested_mm` (decision 27); the sleeve and liner thicknesses keep
//! their clearance check; the inner back apothem is an input (inverse sizing's ring radius).
//! Class N, the dimensions with no rule (cup depth, retainer span, hub length, web, boss,
//! endplates, cap), stay inputs (decision 28); the M41 cap thread below the cup OD and the
//! 22 mm against 25 mm boss OD are M4 design questions.
//!
//! **Space claim** (spec A1: "43 mm diameter × 35 mm overall length, and the 20 mm
//! large-diameter bay, from the metal-design inputs"). Each derived dimension against its
//! claim: the rotating OD (Metal design C135) against the diameter (C131), the axial stack
//! (C134) against the overall length (C130), the large-diameter stack (C137) against the bay
//! (C129). The overshoot is exceeded exactly when the dimension exceeds its claim (the
//! workbook's reserve, claim minus dimension, below 0); at the claim is inside. The badge
//! quotes each overshoot to 0.01 mm and at least 0.01 mm, so a dimension past its claim never
//! reads "0.00 mm over"; an axis that is not a number reads unknown, beside the exceeded ones.

use super::compat::{fmt_fixed, py_max};
use super::meta::{out_rust_only, results};
use super::metal_design::MetalDesignResults;

results! {
    /// The space claim, per axis (Addendum A1). Rust-only.
    pub struct HousingResults {
        fields {
            diameter_overshoot_mm: f64 => out_rust_only("mm", "Diameter beyond the space claim",
                "Addendum A1: the rotating OD (Metal design C135, the larger of the cup and the cap) minus the claimed diameter (C131) when it is larger; 0 inside or at the claim."),
            length_overshoot_mm: f64 => out_rust_only("mm", "Overall length beyond the space claim",
                "The axial stack (Metal design C134) minus the claimed overall length (C130) when it is larger; 0 inside or at the claim."),
            bay_overshoot_mm: f64 => out_rust_only("mm", "Large-diameter stack beyond its bay",
                "The large-diameter stack (Metal design C137) minus the claimed bay (C129) when it is larger; 0 inside or at the claim."),
            space_claim_check: String => out_rust_only("", "Space claim",
                "The dashboard badge: 'Inside the space claim', or 'Exceeds the space claim:' and each axis it exceeds with the overshoot in mm (at least 0.01), an axis that is not a number reading 'unknown'; 'Space claim unknown' when nothing is exceeded and an axis is not a number."),
        }
    }
}

/// `housing.space_claim_check` when every derived dimension is inside or at its claim.
pub const INSIDE_THE_SPACE_CLAIM: &str = "Inside the space claim";

/// `housing.space_claim_check` when no axis is exceeded and a dimension or a claim is not a
/// number (a value set on the struct; `DesignInputs::validate` names the input).
pub const SPACE_CLAIM_UNKNOWN: &str = "Space claim unknown: a dimension or a claim is not a number";

/// How far a derived dimension passes its claim [mm], from the reserve (claim − dimension):
/// its negative when below 0, else 0; NaN when the reserve is NaN.
fn overshoot(reserve_mm: f64) -> f64 {
    if reserve_mm < 0.0 {
        -reserve_mm
    } else if reserve_mm.is_nan() {
        f64::NAN
    } else {
        0.0
    }
}

/// An overshoot as the badge quotes it [mm]: two decimals, and at least 0.01, so a dimension
/// past its claim by less than 0.005 mm does not read "0.00 mm over".
fn quoted_overshoot(over_mm: f64) -> String {
    fmt_fixed(py_max(over_mm, 0.01), 2)
}

/// The space claim of a design, from its Metal design results.
pub fn compute(mdr: &MetalDesignResults) -> HousingResults {
    let axes = [
        ("diameter", overshoot(mdr.diameter_reserve_mm)),
        ("overall length", overshoot(mdr.axial_reserve_mm)),
        ("large-diameter bay", overshoot(mdr.large_dia_reserve_mm)),
    ];
    let check = if axes.iter().any(|&(_, over)| over > 0.0) {
        // Every axis not inside, in order: its overshoot, or unknown when it is not a number.
        let named: Vec<String> = axes
            .iter()
            .filter_map(|&(axis, over)| {
                if over > 0.0 {
                    Some(format!("{axis} {} mm over", quoted_overshoot(over)))
                } else if over.is_nan() {
                    Some(format!("{axis} unknown"))
                } else {
                    None
                }
            })
            .collect();
        format!("Exceeds the space claim: {}", named.join(", "))
    } else if axes.iter().any(|(_, over)| over.is_nan()) {
        SPACE_CLAIM_UNKNOWN.to_owned()
    } else {
        INSIDE_THE_SPACE_CLAIM.to_owned()
    };
    HousingResults {
        diameter_overshoot_mm: axes[0].1,
        length_overshoot_mm: axes[1].1,
        bay_overshoot_mm: axes[2].1,
        space_claim_check: check,
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
use super::compat::{ceiling, fmt_fixed};
use super::deviations::{DeviationId, Deviations};
use super::material_library::PartProperties;
use super::meta::{inputs, out, out_rust_only, param, param_rust_only, results};

inputs! {
    /// 4140 steel properties (Materials!C13:C19).
```

with:

```rust
use super::compat::{ceiling, fmt_fixed};
use super::deviations::{DeviationId, Deviations};
use super::material_library::PartProperties;
use super::meta::{NumOrText, inputs, out, out_rust_only, param, param_rust_only, results};

inputs! {
    /// 4140 steel properties (Materials!C13:C19).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
            cup_wall_check: String => out("", "Cup wall check",
                "A thicker corner wall grows the cup OD by twice the change; recheck the cap thread and envelope.",
                "Materials!C22"),
            hub_flats_under_mm: f64 => out("mm", "Machine the hub flats under by", "On the apothem.", "Materials!C27"),
            cup_pockets_over_mm: f64 => out("mm", "Machine the cup pockets over by", "On the apothem.", "Materials!C28"),
            bores_over_dia_mm: f64 => out("mm", "Machine bores over (on diameter)", "", "Materials!C29"),
```

with:

```rust
            cup_wall_check: String => out("", "Cup wall check",
                "A thicker corner wall grows the cup OD by twice the change; recheck the cap thread and envelope.",
                "Materials!C22"),
            cup_wall_suggested_mm: NumOrText => out_rust_only("mm", "Suggested cup wall at the pocket corners (autofit)",
                "Addendum A1 autofit, decision 27: the wall check's rule, the back-iron thickness needed rounded up to 0.1 mm (the number the check's advice quotes). The wall stays an input (Metal design C122). 'n/a' when the check reads 'No back iron'."),
            hub_flats_under_mm: f64 => out("mm", "Machine the hub flats under by", "On the apothem.", "Materials!C27"),
            cup_pockets_over_mm: f64 => out("mm", "Machine the cup pockets over by", "On the apothem.", "Materials!C28"),
            bores_over_dia_mm: f64 => out("mm", "Machine bores over (on diameter)", "", "Materials!C29"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
    }
}

/// Cup wall check against the back-iron need, and electroless-nickel pre-plate offsets.
/// `backiron` is the Calculator selector C6 in effect (1 steel, 0 none); only E9
/// reads it. `parts` names the materials in effect (Rust-only results).
```

with:

```rust
    }
}

/// Text of `materials.cup_wall_suggested_mm` when the wall check reads "No back iron":
/// no magnetic rule sizes an aluminium cup's wall.
pub const NO_WALL_RULE: &str = "n/a";

/// Cup wall check against the back-iron need, and electroless-nickel pre-plate offsets.
/// `backiron` is the Calculator selector C6 in effect (1 steel, 0 none); only E9
/// reads it. `parts` names the materials in effect (Rust-only results).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
    parts: &PartProperties,
    dev: Deviations,
) -> MaterialsResults {
    let check = if dev.is_on(DeviationId::E9) && backiron == 0 {
        "No back iron".to_owned() // as the Calculator's C105/C106 read
    } else if wall_corner_mm >= t_bi_req_mm {
        "OK".to_owned()
    } else {
        format!(
            "Too thin: raise Metal design C122 to at least {} mm",
            fmt_fixed(ceiling(t_bi_req_mm, 0.1), 1)
        )
    };
    let t = mat.nickel.thickness_mm;
```

with:

```rust
    parts: &PartProperties,
    dev: Deviations,
) -> MaterialsResults {
    // Addendum A1 autofit (decision 27): the rule's wall, the number the advice quotes.
    let suggested = ceiling(t_bi_req_mm, 0.1);
    let no_back_iron = dev.is_on(DeviationId::E9) && backiron == 0;
    let check = if no_back_iron {
        "No back iron".to_owned() // as the Calculator's C105/C106 read
    } else if wall_corner_mm >= t_bi_req_mm {
        "OK".to_owned()
    } else {
        format!(
            "Too thin: raise Metal design C122 to at least {} mm",
            fmt_fixed(suggested, 1)
        )
    };
    let t = mat.nickel.thickness_mm;
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
        backiron_thickness_needed_mm: t_bi_req_mm,
        cup_wall_corner_mm: wall_corner_mm,
        cup_wall_check: check,
        hub_flats_under_mm: t,
        cup_pockets_over_mm: t,
        bores_over_dia_mm: 2.0 * t,
```

with:

```rust
        backiron_thickness_needed_mm: t_bi_req_mm,
        cup_wall_corner_mm: wall_corner_mm,
        cup_wall_check: check,
        cup_wall_suggested_mm: if no_back_iron {
            NumOrText::Text(NO_WALL_RULE)
        } else {
            NumOrText::Num(suggested)
        },
        hub_flats_under_mm: t,
        cup_pockets_over_mm: t,
        bores_over_dia_mm: 2.0 * t,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/mod.rs`, replace:

```rust
pub mod constants;
pub mod deviations;
pub mod grades;
pub mod library;
pub mod material_library;
pub mod materials;
```

with:

```rust
pub mod constants;
pub mod deviations;
pub mod grades;
pub mod housing;
pub mod library;
pub mod material_library;
pub mod materials;
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test sizing 2>&1 | grep "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib wall_suggestion 2>&1 | grep "test result"
```

Expected: `test result: ok. 23 passed`, then `test result: ok. 1 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `134 passed`, `tests\sizing.rs` 23 passed, the others as in Task 6 (the housing group and the suggestion are Rust-only; the verdict text is unchanged, so parity holds).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task7.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task7.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task7.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/housing.rs magcoupling-rs/src/engine/materials.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/tests/sizing.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): housing autofit suggestion and the space claim (spec A1)

Decision 27: the cup wall stays an input; materials.cup_wall_suggested_mm is the
wall rule's value, the number the advice quotes. The space claim per axis
(diameter, overall length, large-diameter bay) from the workbook's reserves:
exceeded exactly when a dimension exceeds its claim, quoted to 0.01 mm and at
least 0.01 mm; a NaN axis reads unknown beside the exceeded ones. Part of every
forward calculation, so sized designs show it too (decision 28: class D/R; the
axial housing follows the length in Task 7b, decision A2-8).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 7b: The axial housing follows the magnet length (decision A2-8, option B)

**Model:** `sonnet` (the plan gives the exact code, and the replay proved every edit, see-it-fail and gate step; the design choices, which dimensions follow, from which length and with which floor, are settled in decision A2-8 and below, so none is left to the implementer; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

Decision A2-8, option B, chosen by the user on 2026-09-30: the axial housing grows with the magnets. With the Rust-only axial length
override set, each class N dimension a ring bounds follows that ring's length change from the ring's own length (its part's, or its
manual length in either manual mode), so the axial stack (Metal design C134), the large-diameter stack (C137, against the 20 mm bay) and
the space claim respond. Which dimensions, from the workbook formulas: C134 = C133 + C124 + C125 + C126 and C137 = C133 + C124 + C125
(`metal_design::compute`, `stack` and `large`); of their terms only the cup cavity depth C124 bounds a magnet (the outer ring sits in
the cavity), while the cap C133 and the web C125 are the thicknesses at its two ends and the boss C126 is the shaft interface, so C124
follows the outer ring. The hub length C123 enters only the hub's mass (Calculator C112: the hub section times C123), and the inner
ring sits on the hub's flats, so C123 follows the inner ring. The retainer span C172 prices the sleeve over the inner ring and the liner
inside the outer ring with one length (Metal design C46, echoed as C45), so it must cover both rings and follows the longer ring's
change: the longer ring's own length to the override (with both rings 12.7 mm, as at the defaults, all three grow by L − 12.7). The
endplates (C170, C171, C182) and the cap's thread engagement (C168) bound no magnet and stay as typed.

Each followed dimension is its input plus (L − the ring's own length), so it keeps the margin the user typed (0.3, 2.8 and 1.8 mm at the
defaults), and never less than the ring's length (the longer ring for the span): the physical minimum. Nothing is added to that
minimum because the workbook has no axial clearance input: its axial margins are these class N inputs themselves, and report 6.5 finds
two identities for the 15.5 mm cup depth (12.7 + 2.0 + 0.8 and 14.5 + 1.0), so neither is a rule to borrow. Because the margin is kept,
the floor binds only where an input is already shorter than its ring (a hub typed shorter than its magnets, or the 14.5 mm span under
15 mm manual rings); there, setting the override, even to the rings' own length, raises that dimension to the ring. `py_max` keeps a NaN
input NaN (never `f64::max`, which would put the ring's length in its place; a NaN override is NaN either way). With the override blank, `axial_housing` returns the three inputs untouched, with
no arithmetic, so every existing result is bit-identical (parity, the differential data and the registry probes never set a Rust-only
input). `api.rs` puts the values in effect into the Metal design inputs before the retainers, the mass estimate and Metal design, so
everything that reads them follows: the retainers' mass (C46; C45 shows the span in effect), the cup and hub masses (Calculator C111,
C112), the total mass and inertia (C114, C115), the heat capacity (Temperature design C141), both stacks and their reserves (Metal
design C134, C137 to C139), the hybrid length (C192) and the space claim. Only the override drives it: a manual length (C14, C24) or another part moves no housing
dimension, as in the workbook.

The default design sized by length to its own 2.5 N·m requirement gets 13.77 mm magnets and both stacks grow by 1.07 mm, to 32.87 mm
of the 35 mm length and 19.87 mm of the 20 mm bay: inside. Sized to 9.9 N·m (50.62 mm magnets) the cup cavity grows by 37.92 mm and the
badge reads "Exceeds the space claim: overall length 34.72 mm over, large-diameter bay 36.72 mm over", as the hand sums of C134 and
C137 give. The bay is crossed first, near 13.9 mm magnets, then the length, near 15.9 mm. The change L − 12.7 is not exact in binary, so
a typed 13.9 mm gives a large-diameter stack of 20.000000000000004 mm and reads "0.01 mm over" by Task 7's rule; the edge test finds
each crossing to the last bit instead of typing a decimal. This task replaces `a_length_sized_design_reads_inside_the_space_claim_at_any_length`
(Task 7, which pinned option (a)) and rewrites Task 5's assertion that the stack and the span stay put under the override.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs` (`AxialHousing`, `axial_housing` (private `follow`); `HousingResults` gains the three dimensions in effect; `compute` takes the axial housing; module docs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs` (the dimensions in effect replace C123, C124 and C172 before the retainers, the mass estimate and Metal design)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs` (the override's help)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs` (Task 5's override test; the length-sized space claim test (option (a)) replaced by nine tests)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/data/input_schema.json` (the override's help)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (housing.rs and tests/sizing.rs rows)

**Interfaces:**
- Consumes: `model::MagnetInputs` with Task 5's `axial_length_mm`, `model::resolve_magnets(m: &MagnetInputs, dev) -> (ResolvedMagnet, ResolvedMagnet)` (each ring's own length: its part's, or its manual length), `MetalDesignInputs::{hub_length_mm, cup_depth_mm, retainer_span_mm}` (Metal design C123, C124, C172), Task 7's `housing::compute` and `HousingResults`, `compat::py_max`.
- Produces:
  - module `engine::housing`: `pub struct AxialHousing { pub hub_length_mm: f64, pub cup_depth_mm: f64, pub retainer_span_mm: f64 }`, `pub fn axial_housing(md: &MetalDesignInputs, magnets: &MagnetInputs, dev: Deviations) -> AxialHousing`; `compute` becomes `pub fn compute(mdr: &MetalDesignResults, axial: &AxialHousing) -> HousingResults`;
  - Rust-only results `housing.hub_length_mm`, `housing.cup_depth_mm`, `housing.retainer_span_mm` (the values in effect, for M4 to draw), after `housing.space_claim_check`.

- [ ] **Step 1: Write the failing tests**

`tests/sizing.rs`: Task 5's override test now expects the axial stack and the retainer span to grow by the rings' change (20 − 12.7 = 7.3 mm). `a_length_sized_design_reads_inside_the_space_claim_at_any_length` is replaced by nine tests: with the override blank the three dimensions are the inputs, bit for bit, and the stacks are the workbook's sums of them (`with_the_override_blank_the_axial_housing_is_the_inputs`); the rings' own length moves no bit of any result (`setting_the_override_to_the_rings_own_length_changes_nothing`); the default design sized by length stays inside at 2.5 N·m and exceeds the claim at 9.9 N·m by the hand sums (`a_design_sized_by_length_grows_its_housing_and_can_exceed_the_space_claim`); a short override shrinks both stacks (`a_short_override_shrinks_the_axial_housing`); the hub, the cup cavity and the span follow the inner, the outer and the longer ring, with either ring the longer (`each_dimension_follows_the_ring_it_bounds`: with the inner ring the longer and no floor binding, the cavity takes the outer ring's change and the span the inner ring's, so a span that followed the outer ring, or a cavity that followed the longer ring, fails); no dimension ends shorter than its ring, checked at the floor's equality edge first, and the floor never hides a NaN input (`a_housing_dimension_never_ends_shorter_than_its_ring`: `f64::max` in place of `py_max` fails); each stack trips the claim exactly where it crosses it (`the_space_claim_trips_exactly_where_a_derived_stack_crosses_its_claim`, each edge found to the last bit and asserted at equality first); the masses follow (`the_housing_masses_follow_the_dimensions_in_effect`: equal to typing the values in effect into C123, C124 and C172); a NaN override reads unknown (`a_nan_override_leaves_the_axial_housing_unknown`).

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust
//! Addendum A1: the axial length override (Task 5), inverse sizing (Task 6) and the space
//! claim (Task 7), end to end through `compute_all`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::housing::{INSIDE_THE_SPACE_CLAIM, SPACE_CLAIM_UNKNOWN};
use magcoupling::engine::meta::NumOrText;
use magcoupling::engine::meta::SetErrorKind;
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
    FreeVariable, SCAN_CELLS, SizingError, SizingOutcome, SizingPoint, VALUE_TOLERANCE_MM,
```

with:

```rust
//! Addendum A1: the axial length override (Task 5), inverse sizing (Task 6), the space
//! claim (Task 7) and the axial housing that follows the override (Task 7b, decision A2-8),
//! end to end through `compute_all`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;
use magcoupling::engine::housing::{INSIDE_THE_SPACE_CLAIM, SPACE_CLAIM_UNKNOWN};
use magcoupling::engine::meta::NumOrText;
use magcoupling::engine::meta::SetErrorKind;
use magcoupling::engine::meta::{Value, result_rows};
use magcoupling::engine::model::{END_EFFECT_OUT_OF_RANGE, blocks_fit, pitch_share};
use magcoupling::engine::sizing::{
    FreeVariable, SCAN_CELLS, SizingError, SizingOutcome, SizingPoint, VALUE_TOLERANCE_MM,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust
    }
}

#[test]
fn a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass() {
    // Decision A2-3: the measured correction keys on the part names, as the workbook's C42
    // does, so a length override keeps it for the prototype's rings with no back iron (no
    // step in torque as a sized length passes the prototype's 12.7 mm). The magnets' mass
    // follows the length; the housing inputs do not (decision 28: no rule sizes them).
    let mut inputs = DesignInputs::default();
    inputs.coupling.backiron = 0;
    let at_part = compute_all(&inputs);
```

with:

```rust
    }
}

/// The default design with the axial length override at `length_mm`.
fn with_length(length_mm: f64) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.coupling.magnets.axial_length_mm = Some(length_mm);
    inputs
}

/// Every result of a design as (path, bit pattern), so NaN equals NaN and nothing is rounded.
fn result_bits(inputs: &DesignInputs) -> Vec<(String, String)> {
    result_rows(&compute_all(inputs))
        .into_iter()
        .map(|row| {
            let bits = match row.value {
                Value::Num(x) => format!("{:016x}", x.to_bits()),
                other => format!("{other:?}"),
            };
            (row.path, bits)
        })
        .collect()
}

#[test]
fn a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass() {
    // Decision A2-3: the measured correction keys on the part names, as the workbook's C42
    // does, so a length override keeps it for the prototype's rings with no back iron (no
    // step in torque as a sized length passes the prototype's 12.7 mm). The magnets' mass
    // follows the length, and so do the axial stack and the retainer span (decision A2-8: the
    // cup cavity and the span grow by the rings' change, 20 - 12.7 = 7.3 mm; Task 7b).
    let mut inputs = DesignInputs::default();
    inputs.coupling.backiron = 0;
    let at_part = compute_all(&inputs);
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust
    assert_eq!(at_part.model.f_cal, at_part.calibration.f_cal_updated);
    assert_eq!(long.model.f_cal, at_part.model.f_cal);
    assert!((long.mass.magnets_g / at_part.mass.magnets_g - 20.0 / 12.7).abs() < 1e-12);
    assert_eq!(long.metal.axial_stack_mm, at_part.metal.axial_stack_mm);
    assert_eq!(
        long.retainers.retainer_span_mm,
        at_part.retainers.retainer_span_mm
    );
}
```

with:

```rust
    assert_eq!(at_part.model.f_cal, at_part.calibration.f_cal_updated);
    assert_eq!(long.model.f_cal, at_part.model.f_cal);
    assert!((long.mass.magnets_g / at_part.mass.magnets_g - 20.0 / 12.7).abs() < 1e-12);
    let change = 20.0 - 12.7;
    assert!((long.metal.axial_stack_mm - at_part.metal.axial_stack_mm - change).abs() < 1e-12);
    assert!(
        (long.retainers.retainer_span_mm - at_part.retainers.retainer_span_mm - change).abs()
            < 1e-12
    );
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/sizing.rs`, replace:

```rust
}

#[test]
fn a_length_sized_design_reads_inside_the_space_claim_at_any_length() {
    // Decision A2-8 as recommended (no engine rule): the axial length moves no dimension the
    // space claim reads (the axial stack C134 and the large-diameter stack C137 are sums of
    // class N inputs), so a length-sized design reads "Inside the space claim" at any length.
    // Sized to its own 2.5 N m requirement the default design's magnets (13.77 mm) outgrow
    // the 13.0 mm hub; sized to 9.9 N m (50.6 mm) they outgrow the 15.5 mm cup and the
    // 14.5 mm retainer span. The override's help says to recheck them; M4 decides the rule.
    let base = DesignInputs::default();
    for (target, longer_than) in [
        (base.metal.required_min_Nm, base.metal.hub_length_mm),
        (9.9, base.metal.cup_depth_mm),
    ] {
        let p = solved(solve(&base, FreeVariable::AxialLength, target));
        assert!(p.value > longer_than, "{target}: {}", p.value);
        let r = compute_all(&p.inputs);
        assert_eq!(
            r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM,
            "{target}"
        );
        assert_eq!(
            r.metal.axial_stack_mm,
            compute_all(&base).metal.axial_stack_mm
        );
    }
    let p = solved(solve(&base, FreeVariable::AxialLength, 9.9));
    assert!(p.value > 50.0 && p.value > base.metal.retainer_span_mm);
}
```

with:

```rust
}

#[test]
fn with_the_override_blank_the_axial_housing_is_the_inputs() {
    // Decision A2-8 acts only through the override. Blank, the hub length, the cup cavity depth
    // and the retainer span that every result reads are the inputs themselves, bit for bit (no
    // floor either: a hub shorter than its ring stays as typed), so no existing result moves;
    // parity and the differential data never set the override.
    let mut odd = DesignInputs::default();
    odd.metal.hub_length_mm = 9.0;
    odd.metal.cup_depth_mm = 30.0;
    odd.metal.retainer_span_mm = 3.0;
    for inputs in [DesignInputs::default(), odd] {
        let m = &inputs.metal;
        let r = compute_all(&inputs);
        let h = &r.housing;
        assert_eq!(
            (
                h.hub_length_mm.to_bits(),
                h.cup_depth_mm.to_bits(),
                h.retainer_span_mm.to_bits(),
                r.retainers.retainer_span_mm.to_bits()
            ),
            (
                m.hub_length_mm.to_bits(),
                m.cup_depth_mm.to_bits(),
                m.retainer_span_mm.to_bits(),
                m.retainer_span_mm.to_bits()
            )
        );
        // Metal design C134 and C137 as the workbook sums them, from the inputs.
        assert_eq!(
            r.metal.axial_stack_mm,
            m.cap_axial_mm + m.cup_depth_mm + m.web_mm + m.boss_length_mm
        );
        assert_eq!(
            r.metal.large_dia_stack_mm,
            m.cap_axial_mm + m.cup_depth_mm + m.web_mm
        );
    }
}

#[test]
fn setting_the_override_to_the_rings_own_length_changes_nothing() {
    // The default rings are 12.7 mm library parts, and each followed dimension covers its ring
    // (13.0, 15.5 and 14.5 mm), so the override at 12.7 mm moves no bit of any result.
    assert_eq!(
        result_bits(&with_length(12.7)),
        result_bits(&DesignInputs::default())
    );
}

#[test]
fn a_design_sized_by_length_grows_its_housing_and_can_exceed_the_space_claim() {
    // Decision A2-8. Sized by length to its own 2.5 N m requirement, the default design's
    // magnets reach 13.77 mm and both stacks grow by the same 1.07 mm, to 32.87 mm of the 35 mm
    // length and 19.87 mm of the 20 mm bay: inside. Sized to 9.9 N m (50.62 mm magnets) the cup
    // cavity grows by 37.92 mm and both stacks pass their claims; the diameter does not move.
    // The hand computation: Metal design C134 and C137 with the cavity in effect (C124 plus the
    // outer ring's change from its 12.7 mm part), in the workbook's order.
    let base = DesignInputs::default();
    let m = &base.metal;
    for target in [m.required_min_Nm, 9.9] {
        let p = solved(solve(&base, FreeVariable::AxialLength, target));
        let r = compute_all(&p.inputs);
        let cavity = m.cup_depth_mm + (p.value - 12.7);
        let stack = m.cap_axial_mm + cavity + m.web_mm + m.boss_length_mm;
        let large = m.cap_axial_mm + cavity + m.web_mm;
        assert_eq!(r.housing.cup_depth_mm, cavity, "{target}");
        assert_eq!(
            (r.metal.axial_stack_mm, r.metal.large_dia_stack_mm),
            (stack, large),
            "{target}"
        );
        assert_eq!(r.housing.diameter_overshoot_mm, 0.0, "{target}");
        if target == m.required_min_Nm {
            assert!((p.value - 13.77).abs() < 0.005, "{}", p.value);
            assert_eq!(r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM);
        } else {
            assert!((p.value - 50.62).abs() < 0.005, "{}", p.value);
            assert_eq!(
                (r.housing.length_overshoot_mm, r.housing.bay_overshoot_mm),
                (
                    stack - m.max_overall_axial_mm,
                    large - m.max_large_dia_axial_mm
                )
            );
            assert_eq!(
                r.housing.space_claim_check,
                "Exceeds the space claim: overall length 34.72 mm over, \
                 large-diameter bay 36.72 mm over"
            );
        }
    }
}

#[test]
fn a_short_override_shrinks_the_axial_housing() {
    // 6.0 mm magnets, 6.7 mm shorter than the 12.7 mm parts: the hub 6.3 mm, the cup cavity
    // 8.8 mm and the retainer span 7.8 mm, each keeping its margin over the rings (0.3, 2.8 and
    // 1.8 mm), and both stacks 6.7 mm shorter (25.1 and 12.1 mm).
    let base = compute_all(&DesignInputs::default());
    let r = compute_all(&with_length(6.0));
    let change = 6.0 - 12.7;
    let h = &r.housing;
    assert_eq!(
        (h.hub_length_mm, h.cup_depth_mm, h.retainer_span_mm),
        (13.0 + change, 15.5 + change, 14.5 + change)
    );
    assert!((r.metal.axial_stack_mm - (base.metal.axial_stack_mm + change)).abs() < 1e-12);
    assert!((r.metal.large_dia_stack_mm - (base.metal.large_dia_stack_mm + change)).abs() < 1e-12);
    assert!((r.metal.axial_stack_mm - 25.1).abs() < 1e-12);
    assert_eq!(h.space_claim_check, INSIDE_THE_SPACE_CLAIM);
}

#[test]
fn each_dimension_follows_the_ring_it_bounds() {
    // Manual rings of 10 mm (inner) and 15 mm (outer), both set to 20 mm. The hub follows the
    // inner ring (+10 mm: 23.0), the cup cavity the outer ring (+5 mm: 20.5), and the retainer
    // span, one span for the sleeve and the liner, the longer ring: 14.5 + 5 = 19.5 mm, shorter
    // than the 20 mm rings (the typed span already left the 15 mm ring 0.5 mm uncovered), so
    // it is raised to 20.0 mm, the physical minimum.
    let manual = |inner_mm: f64, outer_mm: f64| {
        let mut inputs = with_length(20.0);
        inputs.coupling.magnets.part_inner = String::new();
        inputs.coupling.magnets.part_outer = String::new();
        inputs.coupling.magnets.manual_inner_length_mm = inner_mm;
        inputs.coupling.magnets.manual_outer_length_mm = outer_mm;
        compute_all(&inputs)
    };
    let r = manual(10.0, 15.0);
    assert_eq!(
        (r.model.inner_length_mm, r.model.outer_length_mm),
        (20.0, 20.0)
    );
    let h = &r.housing;
    assert_eq!(
        (h.hub_length_mm, h.cup_depth_mm, h.retainer_span_mm),
        (23.0, 20.5, 20.0)
    );
    assert_eq!(r.retainers.retainer_span_mm, 20.0);
    // The inner ring the longer (12 mm inner, 10 mm outer), every input over its ring so no
    // floor binds: the hub +8 mm (21.0), the cup cavity +10 mm (25.5, not the longer ring's
    // +8 mm: 23.5) and the span the longer, inner ring's +8 mm (22.5, not the outer ring's
    // +10 mm: 24.5).
    let r = manual(12.0, 10.0);
    let h = &r.housing;
    assert_eq!(
        (h.hub_length_mm, h.cup_depth_mm, h.retainer_span_mm),
        (21.0, 25.5, 22.5)
    );
    assert_eq!(r.retainers.retainer_span_mm, 22.5);
}

#[test]
fn a_housing_dimension_never_ends_shorter_than_its_ring() {
    // The physical minimum: with the override set, each followed dimension is at least its
    // ring's length (the workbook has no axial clearance input to add). An input with a margin
    // over its ring keeps it and never reaches the floor. The equality edge first: a 12.7 mm hub
    // under the 12.7 mm rings, set to 25.4 mm, where the margin rule lands on the ring exactly.
    let mut inputs = with_length(25.4);
    inputs.metal.hub_length_mm = 12.7;
    assert_eq!(
        12.7 + (25.4 - 12.7),
        25.4,
        "the margin rule lands on the ring"
    );
    assert_eq!(compute_all(&inputs).housing.hub_length_mm, 25.4);
    // A hub already shorter than its ring (12.0 mm under 12.7 mm rings) is raised to the ring:
    // at 20 mm the margin rule would give 19.3 mm. The one exception to "the ring's own length
    // changes nothing": the override at 12.7 mm raises it to 12.7 mm.
    inputs.metal.hub_length_mm = 12.0;
    inputs.coupling.magnets.axial_length_mm = Some(20.0);
    assert_eq!(compute_all(&inputs).housing.hub_length_mm, 20.0);
    inputs.coupling.magnets.axial_length_mm = Some(12.7);
    assert_eq!(compute_all(&inputs).housing.hub_length_mm, 12.7);
    // The ranges' extremes: 50.8 mm manual rings set to 2 mm, with every followed input at its
    // 3 mm minimum: the margin rule gives -45.8 mm, the floor 2 mm; the masses stay positive.
    let mut inputs = with_length(2.0);
    let magnets = &mut inputs.coupling.magnets;
    magnets.part_inner = String::new();
    magnets.part_outer = String::new();
    magnets.manual_inner_length_mm = 50.8;
    magnets.manual_outer_length_mm = 50.8;
    inputs.metal.hub_length_mm = 3.0;
    inputs.metal.cup_depth_mm = 3.0;
    inputs.metal.retainer_span_mm = 3.0;
    let r = compute_all(&inputs);
    let h = &r.housing;
    assert_eq!(
        (h.hub_length_mm, h.cup_depth_mm, h.retainer_span_mm),
        (2.0, 2.0, 2.0)
    );
    assert!(r.mass.hub_g > 0.0 && r.mass.cup_g > 0.0 && r.retainers.retainers_g > 0.0);
    // The floor never hides a NaN input (`py_max`; `f64::max` would put the ring's 20 mm in
    // its place): with the override set, a NaN hub, cup depth or span stays NaN.
    let mut inputs = with_length(20.0);
    inputs.metal.hub_length_mm = f64::NAN;
    inputs.metal.cup_depth_mm = f64::NAN;
    inputs.metal.retainer_span_mm = f64::NAN;
    let h = compute_all(&inputs).housing;
    assert!(h.hub_length_mm.is_nan() && h.cup_depth_mm.is_nan() && h.retainer_span_mm.is_nan());
}

/// The longest override length within 64 ulps of `start` whose derived dimension (`read`) is at
/// most `claim`: the last bit of length inside it.
fn last_length_inside(start: f64, claim: f64, read: fn(&magcoupling::DesignResults) -> f64) -> f64 {
    let at = |length_mm: f64| read(&compute_all(&with_length(length_mm)));
    let mut length_mm = start;
    for _ in 0..64 {
        if at(length_mm) <= claim {
            break;
        }
        length_mm = length_mm.next_down();
    }
    for _ in 0..64 {
        if at(length_mm.next_up()) > claim {
            break;
        }
        length_mm = length_mm.next_up();
    }
    assert!(
        at(length_mm) <= claim && at(length_mm.next_up()) > claim,
        "no edge within 64 ulps of {start}"
    );
    length_mm
}

#[test]
fn the_space_claim_trips_exactly_where_a_derived_stack_crosses_its_claim() {
    // The override moves both stacks and the claim compares them as before: exceeded only
    // above the claim. The cavity reaches the 20 mm bay first (about 13.9 mm magnets), then the
    // 35 mm length (about 15.9 mm). Each edge to the last bit: the length whose stack equals
    // the claim exactly (asserted first) is inside, one bit longer is over.
    let bay_edge = last_length_inside(13.9, 20.0, |r| r.metal.large_dia_stack_mm);
    let r = compute_all(&with_length(bay_edge));
    assert_eq!(
        r.metal.large_dia_stack_mm, 20.0,
        "the stack sits at the claim"
    );
    assert_eq!(r.housing.bay_overshoot_mm, 0.0);
    assert_eq!(r.housing.space_claim_check, INSIDE_THE_SPACE_CLAIM);
    let r = compute_all(&with_length(bay_edge.next_up()));
    assert!(r.housing.bay_overshoot_mm > 0.0);
    assert_eq!(
        r.housing.space_claim_check,
        "Exceeds the space claim: large-diameter bay 0.01 mm over"
    );
    let length_edge = last_length_inside(15.9, 35.0, |r| r.metal.axial_stack_mm);
    let r = compute_all(&with_length(length_edge));
    assert_eq!(r.metal.axial_stack_mm, 35.0, "the stack sits at the claim");
    assert_eq!(r.housing.length_overshoot_mm, 0.0);
    assert_eq!(
        r.housing.space_claim_check,
        "Exceeds the space claim: large-diameter bay 2.00 mm over"
    );
    let r = compute_all(&with_length(length_edge.next_up()));
    assert!(r.housing.length_overshoot_mm > 0.0);
    assert_eq!(
        r.housing.space_claim_check,
        "Exceeds the space claim: overall length 0.01 mm over, large-diameter bay 2.00 mm over"
    );
}

#[test]
fn the_housing_masses_follow_the_dimensions_in_effect() {
    // Bigger parts weigh more. At 20 mm magnets the hub, the cup and the retainers weigh
    // exactly what the default design weighs with the dimensions in effect typed into Metal
    // design C123, C124 and C172; the stacks and the hybrid length agree too, and the total
    // differs only by the magnets. The heat capacity grows with the mass.
    let base = compute_all(&DesignInputs::default());
    let long = compute_all(&with_length(20.0));
    let mut typed = DesignInputs::default();
    typed.metal.hub_length_mm = long.housing.hub_length_mm;
    typed.metal.cup_depth_mm = long.housing.cup_depth_mm;
    typed.metal.retainer_span_mm = long.housing.retainer_span_mm;
    let typed = compute_all(&typed);
    assert_eq!(
        (long.mass.hub_g, long.mass.cup_g, long.retainers.retainers_g),
        (
            typed.mass.hub_g,
            typed.mass.cup_g,
            typed.retainers.retainers_g
        )
    );
    assert_eq!(
        (
            long.metal.axial_stack_mm,
            long.metal.large_dia_stack_mm,
            long.metal.hybrid_length_mm
        ),
        (
            typed.metal.axial_stack_mm,
            typed.metal.large_dia_stack_mm,
            typed.metal.hybrid_length_mm
        )
    );
    let extra = long.mass.magnets_g - typed.mass.magnets_g;
    assert!((long.mass.total_g - typed.mass.total_g - extra).abs() < 1e-9);
    assert!(long.mass.hub_g > base.mass.hub_g);
    assert!(long.mass.cup_g > base.mass.cup_g);
    assert!(long.retainers.retainers_g > base.retainers.retainers_g);
    assert!(
        long.temperature.thermal.heat_capacity_J_K > base.temperature.thermal.heat_capacity_J_K
    );
}

#[test]
fn a_nan_override_leaves_the_axial_housing_unknown() {
    // A NaN set on the struct (validate() names the path): the dimensions in effect and both
    // stacks are NaN, and the space claim reads unknown, not inside.
    let r = compute_all(&with_length(f64::NAN));
    let h = &r.housing;
    assert!(h.hub_length_mm.is_nan() && h.cup_depth_mm.is_nan() && h.retainer_span_mm.is_nan());
    assert!(r.metal.axial_stack_mm.is_nan() && r.metal.large_dia_stack_mm.is_nan());
    assert_eq!(h.space_claim_check, SPACE_CLAIM_UNKNOWN);
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test sizing 2>&1 | grep -E "^error" | head -2
```

Expected: ``error[E0609]: no field `hub_length_mm` on type `&HousingResults` ``, then the same for `cup_depth_mm`.

- [ ] **Step 3: Make the axial housing follow the magnets**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, and `space_claim_check`) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
```

with:

```markdown
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass, not the housing inputs; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; a length-sized design reads inside the space claim at any length (decision A2-8: no rule reads the hub, cup depth or retainer span). |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        backiron: parts.backiron,
        ..inputs.coupling.clone()
    };
    let md = &MetalDesignInputs {
        steel_density_g_mm3: parts.steel.density_g_mm3,
        sleeve_density_g_mm3: parts.sleeve_liner.props.density_g_mm3,
        ..inputs.metal.clone()
    };
    // The cap is the only part the retainers price at Metal design C42.
```

with:

```rust
        backiron: parts.backiron,
        ..inputs.coupling.clone()
    };
    // Addendum A1 (decision A2-8): with the axial length override set, the hub length, the cup
    // cavity depth and the retainer span follow the magnets; blank, they are the inputs.
    let axial = housing::axial_housing(&inputs.metal, &ci.magnets, dev);
    let md = &MetalDesignInputs {
        steel_density_g_mm3: parts.steel.density_g_mm3,
        sleeve_density_g_mm3: parts.sleeve_liner.props.density_g_mm3,
        hub_length_mm: axial.hub_length_mm,
        cup_depth_mm: axial.cup_depth_mm,
        retainer_span_mm: axial.retainer_span_mm,
        ..inputs.metal.clone()
    };
    // The cap is the only part the retainers price at Metal design C42.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
    });

    // Addendum A1: the space claim, from the derived dimensions (both modes).
    let housing = housing::compute(&mdr);

    let alloy = if inputs.clamps.alloy == 1 {
        &materials::AL7075
```

with:

```rust
    });

    // Addendum A1: the space claim, from the derived dimensions (both modes).
    let housing = housing::compute(&mdr, &axial);

    let alloy = if inputs.clamps.alloy == 1 {
        &materials::AL7075
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs`, replace:

```rust
//! endplates, cap), stay inputs (decision 28); the M41 cap thread below the cup OD and the
//! 22 mm against 25 mm boss OD are M4 design questions.
//!
//! **Space claim** (spec A1: "43 mm diameter × 35 mm overall length, and the 20 mm
//! large-diameter bay, from the metal-design inputs"). Each derived dimension against its
//! claim: the rotating OD (Metal design C135) against the diameter (C131), the axial stack
```

with:

```rust
//! endplates, cap), stay inputs (decision 28); the M41 cap thread below the cup OD and the
//! 22 mm against 25 mm boss OD are M4 design questions.
//!
//! **The axial housing follows the magnets** (Addendum A decision A2-8, option B). With the
//! Rust-only axial length override (`coupling.magnets.axial_length_mm`) set, the three class N
//! dimensions a ring bounds follow that ring's length change from its own length (its part's,
//! or its manual length), so the stacks the space claim reads grow and shrink with the magnets
//! ([`axial_housing`]): the hub length (C123, the steel the inner ring sits on, priced at C112)
//! with the inner ring; the cup cavity depth (C124, the pocket the outer ring sits in, summed by
//! the axial stack C134 and the large-diameter stack C137 and priced at C111) with the outer
//! ring; the retainer span (C172, one span for both the sleeve over the inner ring and the liner
//! inside the outer ring, priced at C46) with the longer ring, since it must cover both. The
//! other class N dimensions bound no magnet: the cap (C133) and the web (C125) are thicknesses
//! at the two ends of the cavity, the boss (C126) is the shaft interface and the endplates
//! (C170, C171) are plates. Each followed dimension keeps its input's margin over its ring and
//! never ends shorter than the ring itself, the physical minimum (the workbook has no axial
//! clearance input to add: its margins, 0.3, 2.8 and 1.8 mm at the defaults, are these inputs,
//! and report 6.5 finds two identities for the cup depth, so neither is a rule). Blank, the
//! three are the inputs, bit for bit. Every result that reads them (the masses and so the heat
//! capacity, both stacks, the hybrid length) reads the values in effect, which `housing.*`
//! shows.
//!
//! **Space claim** (spec A1: "43 mm diameter × 35 mm overall length, and the 20 mm
//! large-diameter bay, from the metal-design inputs"). Each derived dimension against its
//! claim: the rotating OD (Metal design C135) against the diameter (C131), the axial stack
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs`, replace:

```rust
//! reads "0.00 mm over"; an axis that is not a number reads unknown, beside the exceeded ones.

use super::compat::{fmt_fixed, py_max};
use super::meta::{out_rust_only, results};
use super::metal_design::MetalDesignResults;

results! {
    /// The space claim, per axis (Addendum A1). Rust-only.
```

with:

```rust
//! reads "0.00 mm over"; an axis that is not a number reads unknown, beside the exceeded ones.

use super::compat::{fmt_fixed, py_max};
use super::deviations::Deviations;
use super::meta::{out_rust_only, results};
use super::metal_design::{MetalDesignInputs, MetalDesignResults};
use super::model::{MagnetInputs, resolve_magnets};

results! {
    /// The space claim, per axis (Addendum A1). Rust-only.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs`, replace:

```rust
                "The large-diameter stack (Metal design C137) minus the claimed bay (C129) when it is larger; 0 inside or at the claim."),
            space_claim_check: String => out_rust_only("", "Space claim",
                "The dashboard badge: 'Inside the space claim', or 'Exceeds the space claim:' and each axis it exceeds with the overshoot in mm (at least 0.01), an axis that is not a number reading 'unknown'; 'Space claim unknown' when nothing is exceeded and an axis is not a number."),
        }
    }
}
```

with:

```rust
                "The large-diameter stack (Metal design C137) minus the claimed bay (C129) when it is larger; 0 inside or at the claim."),
            space_claim_check: String => out_rust_only("", "Space claim",
                "The dashboard badge: 'Inside the space claim', or 'Exceeds the space claim:' and each axis it exceeds with the overshoot in mm (at least 0.01), an axis that is not a number reading 'unknown'; 'Space claim unknown' when nothing is exceeded and an axis is not a number."),
            hub_length_mm: f64 => out_rust_only("mm", "Steel inner hub axial length in effect",
                "Metal design C123; with the axial length override set, C123 plus the inner ring's length change, and at least the ring's length (decision A2-8)."),
            cup_depth_mm: f64 => out_rust_only("mm", "Cup cavity axial depth in effect",
                "Metal design C124; with the axial length override set, C124 plus the outer ring's length change, and at least the ring's length (decision A2-8). Both axial stacks sum it."),
            retainer_span_mm: f64 => out_rust_only("mm", "Retainer axial span in effect",
                "Metal design C172; with the axial length override set, C172 plus the longer ring's length change, and at least that ring's length (decision A2-8)."),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs`, replace:

```rust
    fmt_fixed(py_max(over_mm, 0.01), 2)
}

/// The space claim of a design, from its Metal design results.
pub fn compute(mdr: &MetalDesignResults) -> HousingResults {
    let axes = [
        ("diameter", overshoot(mdr.diameter_reserve_mm)),
        ("overall length", overshoot(mdr.axial_reserve_mm)),
```

with:

```rust
    fmt_fixed(py_max(over_mm, 0.01), 2)
}

/// The axial housing dimensions in effect (decision A2-8): Metal design C123, C124 and C172.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxialHousing {
    pub hub_length_mm: f64,
    pub cup_depth_mm: f64,
    pub retainer_span_mm: f64,
}

/// A dimension that bounds a ring, following the ring's length change: the input plus
/// (`length_mm` - `own_length_mm`), so it keeps the input's margin over the ring, and at least
/// `length_mm`, the ring itself (the physical minimum). NaN in, NaN out ([`py_max`]).
fn follow(input_mm: f64, length_mm: f64, own_length_mm: f64) -> f64 {
    py_max(input_mm + (length_mm - own_length_mm), length_mm)
}

/// The hub length, cup cavity depth and retainer span in effect. With the override blank, the
/// inputs, untouched. With it set, each follows the ring it bounds ([`follow`]) from the ring's
/// own length (its part's, or its manual length; the override gives both rings one length):
/// the hub the inner ring, the cup cavity the outer ring, the retainer span the longer ring.
pub fn axial_housing(
    md: &MetalDesignInputs,
    magnets: &MagnetInputs,
    dev: Deviations,
) -> AxialHousing {
    let Some(length_mm) = magnets.axial_length_mm else {
        return AxialHousing {
            hub_length_mm: md.hub_length_mm,
            cup_depth_mm: md.cup_depth_mm,
            retainer_span_mm: md.retainer_span_mm,
        };
    };
    let own = MagnetInputs {
        axial_length_mm: None,
        ..magnets.clone()
    };
    let (inner, outer) = resolve_magnets(&own, dev);
    AxialHousing {
        hub_length_mm: follow(md.hub_length_mm, length_mm, inner.length_mm),
        cup_depth_mm: follow(md.cup_depth_mm, length_mm, outer.length_mm),
        retainer_span_mm: follow(
            md.retainer_span_mm,
            length_mm,
            py_max(inner.length_mm, outer.length_mm),
        ),
    }
}

/// The space claim of a design, from its Metal design results, and the axial housing in
/// effect ([`axial_housing`]), which it reports.
pub fn compute(mdr: &MetalDesignResults, axial: &AxialHousing) -> HousingResults {
    let axes = [
        ("diameter", overshoot(mdr.diameter_reserve_mm)),
        ("overall length", overshoot(mdr.axial_reserve_mm)),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/housing.rs`, replace:

```rust
        length_overshoot_mm: axes[1].1,
        bay_overshoot_mm: axes[2].1,
        space_claim_check: check,
    }
}
```

with:

```rust
        length_overshoot_mm: axes[1].1,
        bay_overshoot_mm: axes[2].1,
        space_claim_check: check,
        hub_length_mm: axial.hub_length_mm,
        cup_depth_mm: axial.cup_depth_mm,
        retainer_span_mm: axial.retainer_span_mm,
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            grade_outer: String = "" => param_rust_only("-", "Outer magnet grade (manual dimensions)",
                "As the inner grade, for the outer ring."),
            axial_length_mm: Option<f64> = None => param_rust_only("mm", "Axial magnet length, both rings",
                "Addendum A1. Blank = each ring's part or manual length. A value sets both rings' axial length and keeps everything else each ring has (part or manual cross-section, grade, Br, rating): blocks cut or stacked to length. The calibration factor still follows the part names (Calculator C42), and the retainer span, hub length and cup depth stay inputs (Metal design C172, C123, C124: no rule sizes them, Addendum A decision 28), so recheck them. Inverse sizing's default free variable.")
                .range(2.0, 50.8, 0.01),
        }
    }
```

with:

```rust
            grade_outer: String = "" => param_rust_only("-", "Outer magnet grade (manual dimensions)",
                "As the inner grade, for the outer ring."),
            axial_length_mm: Option<f64> = None => param_rust_only("mm", "Axial magnet length, both rings",
                "Addendum A1. Blank = each ring's part or manual length. A value sets both rings' axial length and keeps everything else each ring has (part or manual cross-section, grade, Br, rating): blocks cut or stacked to length. The calibration factor still follows the part names (Calculator C42). The hub length, cup cavity depth and retainer span follow the length change (Metal design C123 with the inner ring, C124 with the outer ring, C172 with the longer ring; each at least its ring's length: decision A2-8), so both axial stacks and the space claim follow; housing.* shows them. Inverse sizing's default free variable.")
                .range(2.0, 50.8, 0.01),
        }
    }
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

- [ ] **Step 4: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test schema
```

Expected: the bless run passes: `tests/data/input_schema.json` changes in `coupling.magnets.axial_length_mm`'s `help` only.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test sizing 2>&1 | grep "test result"
```

Expected: `test result: ok. 31 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: `tests\sizing.rs` 31 passed, the others as in Task 7 (unit tests `134 passed`; parity, the differential data and every registry test unedited: none of them sets the override).

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a2/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task7b.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task7b.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task7b.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/housing.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/model.rs magcoupling-rs/tests/sizing.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): the axial housing follows the magnet length (decision A2-8)

Decision A2-8, option B (the user's choice): with the axial length override set,
the hub length (C123) follows the inner ring, the cup cavity depth (C124) the
outer ring and the retainer span (C172) the longer ring, each by the ring's
length change and never shorter than the ring, so both axial stacks, the masses
and the space claim follow; housing.* shows the values in effect. Blank, the
inputs are used untouched: every existing result is bit-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 8: One aluminium modulus: the library's 6061 takes E18's 68.9 GPa (decision A2-6)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

The A-1 carry-over (`docs/ai/04-memory.yaml`, A-1 decision A8): "E18 uses the report's Alliance 6061 modulus 68.9 GPa while the A5
library's 6061 record carries Kaiser's 68.3 GPa (decision 5): picking 6061 as back iron differs from the default aluminium body in
C104, C105 and C201 only (tests/material_links.rs pins it). Unify when the user decides." Decision A2-6 (recommended) unifies on E18's
value: the library's engine values are already "the workbook's number where one exists", and for 6061's modulus the number the
calculator computes with is E18's registered 68.9 GPa. So the approved E18 probe values stay untouched, and the only result that moves is
the non-default "6061-T6 aluminium" back-iron pick, which now equals the default no-back-iron design cell for cell (C104, C105, C201
took 68.3 before). Kaiser's 68.3 GPa stays the sourced reference, compared with the data file as before (the data file is evidence and
is never edited). The expansion coefficient was already equal (23.6e-6); it now reads the same constant, so the two cannot drift.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/material_library.rs` (6061-T6's engine expansion and modulus are `AL_HUB_CTE_PER_C` and `AL_HUB_MODULUS_GPA`; module doc)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/material_library.rs` (the engine-value rule admits E18's registered modulus for 6061)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/material_links.rs` (a 6061 back iron equals the default no-back-iron design, cell for cell)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (the two tests rows)

**Interfaces:**
- Consumes: `temperature::AL_HUB_CTE_PER_C` (23.6e-6 /°C) and `temperature::AL_HUB_MODULUS_GPA` (68.9 GPa), E18's Alliance values; `material_library::MATERIALS` (the `6061_T6` record, engine modulus 68.3, sourced 68.3 Kaiser).
- Produces: the `6061_T6` record's `engine.cte_per_C = AL_HUB_CTE_PER_C` and `engine.modulus_GPa = AL_HUB_MODULUS_GPA` (one literal each, in `temperature.rs`); its `modulus_GPa: Sourced` stays Kaiser's 68.3 (the data file's value). No default number moves.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/material_library.rs`, replace:

```rust
};
use magcoupling::engine::materials::{AL6061, Steel4140};
use magcoupling::engine::metal_design::MetalDesignInputs;
use magcoupling::engine::temperature::{SlipLossInputs, ThermalInputs};

const DATA: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-data.json";
/// Decision 6 A re-sources the 416 expansion coefficient (the data file keeps a placeholder).
```

with:

```rust
};
use magcoupling::engine::materials::{AL6061, Steel4140};
use magcoupling::engine::metal_design::MetalDesignInputs;
use magcoupling::engine::temperature::{AL_HUB_MODULUS_GPA, SlipLossInputs, ThermalInputs};

const DATA: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-data.json";
/// Decision 6 A re-sources the 416 expansion coefficient (the data file keeps a placeholder).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/material_library.rs`, replace:

```rust
            "{id} engine sigma"
        );
        assert_eq!(e.cp_J_kgK, engine_expected(j, "cp_J_kgK"), "{id} engine cp");
        assert_eq!(e.modulus_GPa, engine_expected(j, "E_GPa"), "{id} engine E");
        assert!(
            close(
                e.density_g_mm3,
```

with:

```rust
            "{id} engine sigma"
        );
        assert_eq!(e.cp_J_kgK, engine_expected(j, "cp_J_kgK"), "{id} engine cp");
        // Decision A2-6: 6061-T6's engine modulus is E18's registered 68.9 GPa (Alliance), so
        // a 6061 back iron equals the default aluminium body; its sourced value (Kaiser 68.3)
        // is compared with the data file above.
        let modulus = if id == "6061_T6" {
            AL_HUB_MODULUS_GPA
        } else {
            engine_expected(j, "E_GPa")
        };
        assert_eq!(e.modulus_GPa, modulus, "{id} engine E");
        assert!(
            close(
                e.density_g_mm3,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/material_links.rs`, replace:

```rust
        c304["Temperature design!C104"],
        aluminium["Temperature design!C104"]
    );
    // The library's 6061 differs from the workbook aluminium only in the modulus E18 reads
    // (68.3 GPa Kaiser against 68.9 GPa Alliance; decision table of the A-1 plan).
    let c6061 = with_back_iron(8);
    let changed = changed_results(&no_iron, &c6061, Deviations::ALL);
    let want: BTreeSet<String> = ["C104", "C105", "C201"]
        .iter()
        .map(|c| format!("Temperature design!{c}"))
        .collect();
    assert_eq!(changed, want);
}

#[test]
```

with:

```rust
        c304["Temperature design!C104"],
        aluminium["Temperature design!C104"]
    );
    // Decision A2-6: the library's 6061 engine values are the workbook aluminium's and E18's
    // (its modulus is E18's 68.9 GPa, the sourced Kaiser 68.3 GPa stays the reference), so
    // picking 6061 as back iron is the default no-back-iron design, cell for cell.
    let c6061 = with_back_iron(8);
    let changed = changed_results(&no_iron, &c6061, Deviations::ALL);
    assert!(changed.is_empty(), "{changed:?}");
}

#[test]
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --no-fail-fast --test material_library --test material_links 2>&1 | grep -E "test result|panicked|left|right"
```

Expected: `materials_equal_the_addendum_data_file` panics with `6061_T6 engine E`, `left: 68.3`, `right: 68.9`; `a_non_ferromagnetic_back_iron_is_the_hub_cup_and_boss_material` panics (C104, C105 and C201 still differ); `test result: FAILED. 3 passed; 1 failed` and `test result: FAILED. 10 passed; 1 failed`.

- [ ] **Step 3: Point the 6061 record at E18's constants**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `tests/differential.rs` | Every result of every seeded case equals the Python engine (`tests/data/differential/<group>.json`), deviations off. The full run (`differential/full.json`, `full_run_matches_python_on_every_case`) varies all 160 inputs at once and compares every result, groups and tables, so it also catches a `MODULES` entry that forgot an input group; `every_selector_pair_is_covered_in_the_full_run` checks that every pair of selector choices across groups occurs in it, and `every_selector_choice_appears_in_every_module_file` that each module file sets every selector it varies to every choice. `every_branch_is_reached` checks the `BRANCHES` table (every branch of every text result is hit), `every_text_result_has_a_branches_entry` that no text-producing result lacks a `BRANCHES` row (a new branch cannot land unchecked), and `every_varied_input_takes_two_values` that each varied input changes; the helpers corpus checks `compat` against Python exactly. Rust-only results (`ResultMeta::rust_only`, e.g. `clamps.length_note`) have no Python counterpart and are skipped; so are Rust-only selectors in the selector-coverage tests (`InputMeta::rust_only`: the generator never passes them to Python). |
| `tests/python_schema.rs` | Every ported field carries the Python label, unit, help, cell, choices and default; no Python field of a ported group is missing; each ported table has the Python field order, row count and cell of every value (`tables_match_the_python_layout`); `headline` has Python's keys, order and values at the defaults; inputs and scalar results are listed in the Python order (the GUI's tables and CSV export follow it); Rust-only inputs and results are skipped, and no Rust-only input may share a path with a Python one (`rust_only_inputs_are_unknown_to_python`). |
| `tests/grades.rs` | The grade table equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (engine literals bit for bit; the N42SH engine beta is the workbook's); every library part resolves to a grade; a part's workbook Br and Tmax equal its grade's except the registered differences; sintered NdFeB alpha and density equal the engine constants; only ferrite has a positive beta; every part cites its vendor page for coating and magnetization. |
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
```

with:

```markdown
| `tests/differential.rs` | Every result of every seeded case equals the Python engine (`tests/data/differential/<group>.json`), deviations off. The full run (`differential/full.json`, `full_run_matches_python_on_every_case`) varies all 160 inputs at once and compares every result, groups and tables, so it also catches a `MODULES` entry that forgot an input group; `every_selector_pair_is_covered_in_the_full_run` checks that every pair of selector choices across groups occurs in it, and `every_selector_choice_appears_in_every_module_file` that each module file sets every selector it varies to every choice. `every_branch_is_reached` checks the `BRANCHES` table (every branch of every text result is hit), `every_text_result_has_a_branches_entry` that no text-producing result lacks a `BRANCHES` row (a new branch cannot land unchecked), and `every_varied_input_takes_two_values` that each varied input changes; the helpers corpus checks `compat` against Python exactly. Rust-only results (`ResultMeta::rust_only`, e.g. `clamps.length_note`) have no Python counterpart and are skipped; so are Rust-only selectors in the selector-coverage tests (`InputMeta::rust_only`: the generator never passes them to Python). |
| `tests/python_schema.rs` | Every ported field carries the Python label, unit, help, cell, choices and default; no Python field of a ported group is missing; each ported table has the Python field order, row count and cell of every value (`tables_match_the_python_layout`); `headline` has Python's keys, order and values at the defaults; inputs and scalar results are listed in the Python order (the GUI's tables and CSV export follow it); Rust-only inputs and results are skipped, and no Rust-only input may share a path with a Python one (`rust_only_inputs_are_unknown_to_python`). |
| `tests/grades.rs` | The grade table equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (engine literals bit for bit; the N42SH engine beta is the workbook's); every library part resolves to a grade; a part's workbook Br and Tmax equal its grade's except the registered differences; sintered NdFeB alpha and density equal the engine constants; only ferrite has a positive beta; every part cites its vendor page for coating and magnetization. |
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for (6061-T6's engine modulus is E18's 68.9 GPa, decision A2-6); the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/material_library.rs`, replace:

```rust
//! [`Material::engine`] holds what the engine uses when a material is picked:
//! the workbook's number where the workbook has one (4140, 316L, 6061, 7075:
//! decisions 21 to 26 keep them as the defaults, the sourced value is the
//! reference), else the sourced value, as engine-unit literals. The default
//! choice of each part is the workbook's material, whose values ARE the
//! existing inputs (Materials C13 to C18, Temperature design C111, C139, C140,
//! Metal design C42, C44, C132): picking it changes nothing, so parity and the
//! differential tests hold (decision table of the A-1 plan).

/// A property as the data file selects it: the value in the data file's unit and
/// its source, or `None` where no source was found or the property does not apply.
```

with:

```rust
//! [`Material::engine`] holds what the engine uses when a material is picked:
//! the workbook's number where the workbook has one (4140, 316L, 6061, 7075:
//! decisions 21 to 26 keep them as the defaults, the sourced value is the
//! reference), else the sourced value, as engine-unit literals; 6061-T6's expansion
//! and modulus are correction E18's (the Alliance datasheet, 23.6e-6 /°C and 68.9 GPa:
//! [`AL_HUB_CTE_PER_C`], [`AL_HUB_MODULUS_GPA`]), so a 6061 back iron equals the default
//! aluminium body cell for cell (Addendum A-2 decision A2-6; Kaiser's 68.3 GPa stays the
//! sourced reference). The default
//! choice of each part is the workbook's material, whose values ARE the
//! existing inputs (Materials C13 to C18, Temperature design C111, C139, C140,
//! Metal design C42, C44, C132): picking it changes nothing, so parity and the
//! differential tests hold (decision table of the A-1 plan).

use super::temperature::{AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA};

/// A property as the data file selects it: the value in the data file's unit and
/// its source, or `None` where no source was found or the property does not apply.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/material_library.rs`, replace:

```rust
            sigma_S_m: 25000000.0,
            density_g_mm3: 0.0027,
            cp_J_kgK: 900.0,
            cte_per_C: 23.6e-6,
            modulus_GPa: 68.3,
        },
        needs_plating: false,
        notes: "Default cap and housing (the workbook's C42, C43 and C140). As a back iron: non-magnetic demonstration. mu_r: pure-aluminium proxy. M5.",
```

with:

```rust
            sigma_S_m: 25000000.0,
            density_g_mm3: 0.0027,
            cp_J_kgK: 900.0,
            cte_per_C: AL_HUB_CTE_PER_C,
            modulus_GPa: AL_HUB_MODULUS_GPA,
        },
        needs_plating: false,
        notes: "Default cap and housing (the workbook's C42, C43 and C140). As a back iron: non-magnetic demonstration. mu_r: pure-aluminium proxy. M5.",
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/material_library.rs`, replace:

```rust
    thermal: &super::temperature::ThermalInputs,
) -> PartProperties {
    use super::materials::AL6061;
    use super::temperature::{AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA};
    let workbook_steel = props(
        steel.conductivity_S_m,
        md.steel_density_g_mm3,
```

with:

```rust
    thermal: &super::temperature::ThermalInputs,
) -> PartProperties {
    use super::materials::AL6061;
    let workbook_steel = props(
        steel.conductivity_S_m,
        md.steel_density_g_mm3,
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test material_library --test material_links 2>&1 | grep "test result"
```

Expected: `test result: ok. 4 passed`, then `test result: ok. 11 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`, the counts as in Task 7b (`tests\deviations.rs` 54 passed: E18's registry probes and its report test are unchanged).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task8.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task8.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task8.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/material_library.rs magcoupling-rs/tests/material_library.rs magcoupling-rs/tests/material_links.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
fix(magcoupling-rs): one aluminium modulus, E18's 68.9 GPa for the library's 6061 (decision A2-6)

The A-1 open item A8: a 6061 back iron differed from the default aluminium body
in C104, C105 and C201 (68.3 against 68.9 GPa). The library's engine values are
the calculator's numbers; for 6061's modulus that is E18's registered value, so
the approved E18 probes stay and the 6061 pick now equals the default design.
Kaiser's 68.3 GPa stays the sourced reference.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 9: C45 keeps its NdFeB slider; a positive beta is typed (decision A2-5)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

The A-1 carry-over (`docs/ai/04-memory.yaml`, A-1 decisions A2 and A9): "C45's slider (-0.008 to -0.001) cannot enter ferrite's
+0.0035 with the source at 0: widen or split it". Decision A2-5 (recommended) does neither. Widening the slider changes
`tests/data/input_schema.json`'s range, from which `gen_differential.py` draws every random case and range-end case: the differential
data would be regenerated, against "the M2 parity and differential tests are unchanged" (and Python applies |β|, so a positive C45 has
no workbook meaning to compare with). Splitting it adds a second input for one quantity. A slider range does not reject a typed value
(the meta docs: "it does not reject typed values"), and M4's value box accepts typed entry for every input; E20 already handles a
positive β (A-1 Task 10). So C45 keeps the workbook's sintered-NdFeB range, its help says a positive value is typed, and a test pins
the path: `set()` accepts +0.0035, and with the coercivity source at 0 the default rings take E20's cold side (hot onsets +inf, the
150 °C rating as the hot limit, a cold limit of about −64 °C that the −40 °C minimum clears: "OK"). A graded magnet needs none of this:
its grade supplies β (Y30: +0.0035) with the source at 1.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs` (C45's help says how a positive beta is entered)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs` (a typed positive beta with the coercivity source at 0 takes E20's cold side)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/data/input_schema.json` (C45's help)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (robustness row)

**Interfaces:**
- Consumes: `temperature.demag.beta_hcj_per_C` (C45, slider −0.008 to −0.001, the E20 help), `temperature.demag.coercivity_source` (A-1's Rust-only selector), E20's cold-side branch (`ring_demag`: hot onsets +inf, the rating as the hot limit, `cold_limit_C`, `cold_check`).
- Produces: C45's help text (the rest of its metadata, and every number, unchanged).

- [ ] **Step 1: Write the failing test**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/robustness.rs`, replace:

```rust
}

#[test]
fn a_positive_beta_without_a_rating_has_no_hot_limit() {
    // E20: a positive beta typed into C45 (coercivity source 0) for manual magnets without a
    // grade: no knee on heating and no rating, so no magnet limit (+inf: the adhesive governs
```

with:

```rust
}

#[test]
fn a_typed_positive_beta_takes_the_cold_side() {
    // Decision A2-5: C45's slider stays the workbook's NdFeB range (the differential generator
    // samples from it), and a positive beta is typed: set() refuses only NaN, infinities and
    // codes outside choices. With the coercivity source at 0 (C44 and C45 for both rings) a
    // typed +0.0035 on the default rings (rated 150 C) takes E20's cold side: no knee on
    // heating, the rating as the hot limit, a cold limit that the -40 C minimum clears.
    let meta = input_rows(&DesignInputs::default())
        .into_iter()
        .find(|r| r.path == "temperature.demag.beta_hcj_per_C")
        .expect("C45")
        .meta;
    let range = meta.range.expect("a slider");
    assert_eq!((range.min, range.max), (-0.008, -0.001));
    assert!(meta.help.contains("typed"), "{}", meta.help);
    let mut inputs = DesignInputs::default();
    inputs
        .set("temperature.demag.coercivity_source", Value::Int(0))
        .unwrap();
    inputs
        .set("temperature.demag.beta_hcj_per_C", Value::Num(0.0035))
        .unwrap();
    let d = compute_all(&inputs).temperature.demag;
    assert_eq!(d.beta_used_per_C, 0.0035);
    assert_eq!(d.onset_skipping_C, f64::INFINITY);
    assert_eq!(d.magnet_limit_C, 150.0);
    assert!(
        matches!(d.cold_limit_C, NumOrText::Num(c) if c < -40.0),
        "{:?}",
        d.cold_limit_C
    );
    assert_eq!(d.cold_check, "OK");
}

#[test]
fn a_positive_beta_without_a_rating_has_no_hot_limit() {
    // E20: a positive beta typed into C45 (coercivity source 0) for manual magnets without a
    // grade: no knee on heating and no rating, so no magnet limit (+inf: the adhesive governs
```

- [ ] **Step 2: Run it to see it fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test robustness typed_positive 2>&1 | grep -E "panicked|test result"
```

Expected: `a_typed_positive_beta_takes_the_cold_side` panics at the help assertion (`meta.help.contains("typed")`); `test result: FAILED. 0 passed; 1 failed`. (The engine side already holds: A-1's E20 branch.)

- [ ] **Step 3: Reword C45's help**

E20's registry entry keeps the workbook's C45 help in `workbook_help` (`tests/python_schema.rs` compares Python's help with it, and the port's help must differ), so only the port's text changes.

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
```

with:

```markdown
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `e7_finds_the_peak_at_a_fill_of_exactly_0_4` pins E7 where the fifth harmonic vanishes (a ring at a fill of exactly 0.4). `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
                "N42SH ≥ 20 kOe. Correction E20: each magnet's grade supplies Hcj; this value is used for a magnet without a grade, or for both rings when the coercivity source is 0.", "Temperature design!C44")
                .range(800.0, 3000.0, 1.0),
            beta_hcj_per_C: f64 = -0.005 => param("1/°C", "Hcj temperature coefficient (effective, 20–150 °C)",
                "Correction E20: each magnet's grade supplies beta; this value is used for a magnet without a grade, or for both rings when the coercivity source is 0.", "Temperature design!C45")
                .range(-0.008, -0.001, 0.0001)
                .assumption(),
            knee_fraction: f64 = 0.9 => param("-", "Knee field as a fraction of Hcj",
```

with:

```rust
                "N42SH ≥ 20 kOe. Correction E20: each magnet's grade supplies Hcj; this value is used for a magnet without a grade, or for both rings when the coercivity source is 0.", "Temperature design!C44")
                .range(800.0, 3000.0, 1.0),
            beta_hcj_per_C: f64 = -0.005 => param("1/°C", "Hcj temperature coefficient (effective, 20–150 °C)",
                "Correction E20: each magnet's grade supplies beta; this value is used for a magnet without a grade, or for both rings when the coercivity source is 0. The slider covers sintered NdFeB (-0.8 to -0.1 %/°C); a positive value (hard ferrite, whose coercivity falls as it cools, e.g. +0.0035) is typed in, and E20's cold-side check then applies.", "Temperature design!C45")
                .range(-0.008, -0.001, 0.0001)
                .assumption(),
            knee_fraction: f64 = 0.9 => param("-", "Knee field as a fraction of Hcj",
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test schema
```

Expected: the bless run passes: `tests/data/input_schema.json` changes in C45's `help` only.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test robustness typed_positive 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: `tests\robustness.rs` 12 passed, the others as in Task 8.

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a2/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task9.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task9.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task9.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/temperature.rs magcoupling-rs/tests/robustness.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
docs(magcoupling-rs): C45 keeps its NdFeB slider; a positive beta is typed (decision A2-5)

Widening the slider would regenerate the differential data (the generator samples
from the range); splitting it would give one quantity two inputs. C45's help now
says a positive beta (hard ferrite) is typed, and a test pins E20's cold side for a
typed +0.0035 with the coercivity source at 0.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 10: Each grade-mode ring's own alpha(Br) and density (decision A2-7)

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; temperature scaling across four modules). Physics reviewer: spec A6 ("Grade: Br, Hcj, Hcb, (BH)max, α(Br) and β(Hcj) temperature coefficients, ..., density"), the A-1 plan's decisions A4 and A13, the E20 registry entry, and this plan's decision A2-7 (including its note on how each ring's reverse fields are scaled).

The A-1 carry-over (`docs/ai/04-memory.yaml`; A-1 decision A4 kept one alpha and the NdFeB density "as for every library part (all
NdFeB)", with "per-ring alpha and density from the grade (touches model, metal design, temperature and mass; no parity risk because the
mode is new): carried to A-2"). Decision A2-7 (recommended): a ring in the grade mode (manual dimensions with a grade picked, A-1 Task 4)
takes its grade's α(Br) and density; a library part and a manual magnet without a grade keep the calculator's single α (Calibration C22)
and the NdFeB density. A library part's grade is sintered NdFeB, whose α and density equal C22's default and the NdFeB constant, so the
two paths agree at the defaults; keeping C22 for library parts keeps it a live assumption (Task 4's smoke test) and keeps the NONE-mode
differential data (which vary C22 with library parts) exact. `ResolvedMagnet` gets its own signal, `from_grade`, because a library part
carries a grade too. What follows the per-ring coefficients: Br at the operating temperature (C69, C70) and so the pull-out; every torque
at another temperature, which goes with Br_i(T) · Br_o(T) (Metal design C8, C19, C111, C155; Temperature design C61, C16, C184, C186 and
the rows built on them); each ring's demagnetization onsets under E20 (its reverse fields scale with its own Br: the stored 3D fields do
not separate the opposing ring's share, M3's work); the magnets' mass (C110) and the inner block's mass for the bond load (C82). The
magnets' heat capacity keeps the NdFeB specific heat input (the grade table has none). Bit-identity with one coefficient: `x.powi(2)` is
`x * x` exactly, so th_i(T)·th_o(T) and (th_i(a)/th_i(b))·(th_o(a)/th_o(b)) reproduce the workbook's squares, and the magnets' mass keeps
the workbook's single product when both densities are equal. A-1's E20 unit test of mixed rings assumed one α ("the calculator's one
alpha (A4)"); its expectation for the torque at the limit now uses both rings' coefficients, and its no-E20 comparison holds the outer
coefficient equal. The E20 registry probes are unchanged (the ferrite probe sets C22 to Y30's −0.20 %/°C already).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs` (`ResolvedMagnet::from_grade`, `alpha_br()`, `density_g_mm3()`; Br at the operating temperature per ring; per-ring magnet mass; four Rust-only results; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs` (`compute` takes each ring's coefficient; the torque range uses their product; test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs` (`TemperatureLinks::{inner_alpha_br, outer_alpha_br, inner_magnet_density_g_mm3}` replace `alpha_br`; `torque_factor`; each ring's demag check with its own coefficient; the bond-load block mass; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs` (pass each ring's coefficient and density)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/grades.rs` (docs: alpha and density are read in the grade mode)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/assumptions.rs` (the Br coefficient's rationale names the grade mode)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/grades.rs` (two Y30 grade rings end to end)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (model and grades rows)

**Interfaces:**
- Consumes: `model::ResolvedMagnet` (`grade: Option<&'static Grade>`), `Grade::{alpha_br_per_C, density_g_mm3}` (sintered NdFeB: −0.0012 and 0.0075, the engine constants, pinned by `tests/grades.rs`; Y30 −0.002 and 0.005; SmCo −0.00035 to −0.00045; bonded −0.0014 and 0.0058), `constants::NDFEB_DENSITY_G_MM3`, `metal_design::compute(md, torque_op, torque_20C, op_temp, alpha_br, corner_gap, ...)`, `temperature::TemperatureLinks::alpha_br`.
- Produces:
  - `ResolvedMagnet::from_grade: bool` (true only for manual dimensions with a grade), `pub fn alpha_br(&self, calculator_alpha: f64) -> f64`, `pub fn density_g_mm3(&self) -> f64`;
  - Rust-only results `model.inner_alpha_br_per_C`, `outer_alpha_br_per_C`, `inner_magnet_density_g_mm3`, `outer_magnet_density_g_mm3` (C35 keeps showing Calibration C22);
  - `metal_design::compute(.., alpha_br: f64, alpha_inner: f64, alpha_outer: f64, corner_gap_mm, ..)` (two new parameters after `alpha_br`);
  - `TemperatureLinks { inner_alpha_br: f64, outer_alpha_br: f64, inner_magnet_density_g_mm3: f64, .. }` (the field `alpha_br` is gone); private `fn torque_factor(k: &TemperatureLinks, T: f64) -> f64`; `temperature.demag.alpha_br` (C43) shows the governing ring's coefficient.

- [ ] **Step 1: Write the failing tests**

`model.rs`: a Y30 inner ring takes −0.002 and 0.005, a gradeless manual ring and a library part follow the alpha argument, an NdFeB grade changes nothing; the magnets' mass per ring. `metal_design.rs`: the torque range with each ring's coefficient (the existing equality-edge test gains the two new arguments). `temperature.rs`: the fixtures carry the per-ring fields, the mixed-rings test expects both rings' coefficients in the torque at the limit, and a new test checks each ring's onsets and the block mass. `tests/grades.rs`: two Y30 grade rings with C22 at its default equal the A-1 way of typing −0.002 into C22 by hand.

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
    use super::*;

    #[test]
    fn checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8. Neither left-hand side (hot_low, min_run) reads its
        // right-hand side, so one run supplies it and a second run puts the check at equality.
```

with:

```rust
    use super::*;

    #[test]
    fn each_ring_scales_the_torque_with_its_own_coefficient() {
        // Decision A2-7: torque goes with Br_inner * Br_outer, each ring with its coefficient;
        // with one coefficient the products are the workbook's squares, bit for bit.
        let ret = RetainerResults {
            retainer_span_mm: 14.5,
            retainers_g: 3.131,
            sleeve_id_mm: 27.436,
            sleeve_od_mm: 27.636,
            liner_od_mm: 29.39,
            liner_id_mm: 28.99,
            endplate_od_mm: 27.436,
            cap_g: 2.322,
            endplates_g: 6.653,
        };
        let run = |alpha_inner: f64, alpha_outer: f64| {
            compute(
                &MetalDesignInputs::default(),
                2.6473,
                2.8487,
                50.0,
                -0.0012,
                alpha_inner,
                alpha_outer,
                1.0268,
                1.4,
                41.2,
                10,
                10.0,
                5.0,
                0.95,
                173.79,
                30.778,
                &ret,
                0.9,
                20.0,
                0.00785,
                Deviations::NONE,
            )
        };
        let md = MetalDesignInputs::default();
        let th = |alpha: f64, t: f64| br_factor(alpha, t);
        let one = run(-0.0012, -0.0012);
        assert_eq!(
            one.torque_cold_Nm,
            2.8487 * th(-0.0012, md.min_temp_C).powi(2)
        );
        assert_eq!(
            one.cold_for_hot_min_Nm,
            md.required_min_Nm * (th(-0.0012, md.min_temp_C) / th(-0.0012, 50.0)).powi(2)
        );
        let mixed = run(-0.002, -0.0012);
        assert_eq!(
            mixed.torque_cold_Nm,
            2.8487 * (th(-0.002, md.min_temp_C) * th(-0.0012, md.min_temp_C))
        );
        assert_eq!(
            mixed.required_20C_zero_scatter_Nm,
            md.required_min_Nm / (th(-0.002, 50.0) * th(-0.0012, 50.0))
        );
        // The prototype's baseline (C151) keeps the calculator's alpha: B842SH rings.
        assert_eq!(mixed.noiron_baseline_hot_Nm, one.noiron_baseline_hot_Nm);
        assert_eq!(mixed.alpha_br_per_C, -0.0012);
    }

    #[test]
    fn checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8. Neither left-hand side (hot_low, min_run) reads its
        // right-hand side, so one run supplies it and a second run puts the check at equality.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
                2.6473,
                2.8487,
                50.0,
                -0.0012,
                1.0268,
                1.4,
```

with:

```rust
                2.6473,
                2.8487,
                50.0,
                -0.0012,
                -0.0012,
                -0.0012,
                1.0268,
                1.4,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    }

    #[test]
    fn a_library_part_wins_over_a_grade_and_an_unknown_grade_is_manual() {
        let mut ci = CouplingInputs::default();
        ci.magnets.grade_inner = "Y30".to_owned(); // the part B842SH is in the library
```

with:

```rust
    }

    #[test]
    fn a_grade_ring_takes_its_grade_alpha_and_density() {
        // Decision A2-7: a ring in the grade mode (manual dimensions with a grade picked) takes
        // its grade's Br temperature coefficient and density. A library part (every one
        // sintered NdFeB) and a manual magnet without a grade keep the calculator's single
        // alpha (Calibration C22, here the `alpha_br` argument) and the NdFeB density.
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        let r = at(&ci);
        assert_eq!(
            (r.inner_alpha_br_per_C, r.outer_alpha_br_per_C),
            (-0.002, -0.0012)
        );
        assert_eq!(
            (r.inner_magnet_density_g_mm3, r.outer_magnet_density_g_mm3),
            (0.005, NDFEB_DENSITY_G_MM3)
        );
        assert_eq!(r.br_inner_T_op, 0.37 * br_factor(-0.002, 50.0));
        assert_eq!(r.br_outer_T_op, r.outer_br_T * br_factor(-0.0012, 50.0));
        assert_eq!(
            r.alpha_br_per_C, -0.0012,
            "C35 still shows the calculator's alpha"
        );
        // A gradeless manual magnet and a library part follow the argument, whatever it is.
        let mut manual = CouplingInputs::default();
        manual.magnets.part_inner = String::new();
        let r = compute(
            &manual,
            1.4,
            0.05,
            0.05,
            1.8,
            -0.001,
            1.5,
            0.95,
            0.95,
            2000.0,
            2.5,
            Deviations::NONE,
        );
        assert_eq!(
            (r.inner_alpha_br_per_C, r.outer_alpha_br_per_C),
            (-0.001, -0.001)
        );
        assert_eq!(r.inner_magnet_density_g_mm3, NDFEB_DENSITY_G_MM3);
        // An NdFeB grade in the grade mode has the calculator's default values: nothing moves.
        let mut n42 = CouplingInputs::default();
        n42.magnets.part_inner = String::new();
        n42.magnets.grade_inner = "N42".to_owned();
        let r = at(&n42);
        assert_eq!(
            (r.inner_alpha_br_per_C, r.inner_magnet_density_g_mm3),
            (-0.0012, NDFEB_DENSITY_G_MM3)
        );
    }

    #[test]
    fn a_library_part_wins_over_a_grade_and_an_unknown_grade_is_manual() {
        let mut ci = CouplingInputs::default();
        ci.magnets.grade_inner = "Y30".to_owned(); // the part B842SH is in the library
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    }

    #[test]
    fn e9_cup_and_boss_follow_the_hub_density() {
        // The Metal design defaults api::compute passes; retainers, hardware, cap and
        // end plates are left at 0 because only the densities matter here.
```

with:

```rust
    }

    #[test]
    fn a_grade_ring_weighs_at_its_grade_density() {
        // Decision A2-7: the magnets' mass prices each ring at its own density; rings of one
        // density keep the workbook's single product, bit for bit.
        let md = super::super::metal_design::MetalDesignInputs::default();
        let mass = |ci: &CouplingInputs| {
            let r = at(ci);
            let m = mass_estimate(
                ci,
                &r,
                md.bond_inner_mm,
                md.bond_outer_mm,
                md.cup_depth_mm,
                md.web_mm,
                md.hub_length_mm,
                md.boss_length_mm,
                md.boss_od_mm,
                md.steel_density_g_mm3,
                md.al_density_g_mm3,
                0.0,
                0.0,
                0.0,
                0.0,
                Deviations::NONE,
            );
            (m.magnets_g, r)
        };
        let (workbook, r) = mass(&CouplingInputs::default());
        let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
        let volume_o = r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm;
        assert_eq!(workbook, 10.0 * (volume_i + volume_o) * NDFEB_DENSITY_G_MM3);
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        let (ferrite, r) = mass(&ci);
        let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
        assert_eq!(
            ferrite,
            10.0 * (volume_i * 0.005 + volume_o * NDFEB_DENSITY_G_MM3)
        );
    }

    #[test]
    fn e9_cup_and_boss_follow_the_hub_density() {
        // The Metal design defaults api::compute passes; retainers, hardware, cap and
        // end plates are left at 0 because only the densities matter here.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            op_temp_C: 50.0,
            npole: 10,
            br20_T: 1.29,
            alpha_br: -0.0012,
            tmax_lib_C: NumOrText::Num(150.0),
            mu0: 1.256637e-6,
            pullout_op_Nm: 2.6473,
```

with:

```rust
            op_temp_C: 50.0,
            npole: 10,
            br20_T: 1.29,
            inner_alpha_br: -0.0012,
            outer_alpha_br: -0.0012,
            inner_magnet_density_g_mm3: 0.0075,
            tmax_lib_C: NumOrText::Num(150.0),
            mu0: 1.256637e-6,
            pullout_op_Nm: 2.6473,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        k.inner_length_mm = 1.0;
        k.inner_width_mm = 1.0;
        k.cold_high_Nm = 0.75; // tau_b = 0.75 MPa: fat = 0.2 * 15 / 0.75 = 4 (AA 326, 15 MPa)
        k.alpha_br = 0.0; // every temperature factor is exactly 1: amp = pullout_20C * (1 + variation)
        k.variation = 0.0;
        k.pullout_20C_Nm = 0.9375;
        let mut ti = TemperatureInputs::default();
```

with:

```rust
        k.inner_length_mm = 1.0;
        k.inner_width_mm = 1.0;
        k.cold_high_Nm = 0.75; // tau_b = 0.75 MPa: fat = 0.2 * 15 / 0.75 = 4 (AA 326, 15 MPa)
        k.inner_alpha_br = 0.0; // every temperature factor is exactly 1: amp = pullout_20C * (1 + variation)
        k.outer_alpha_br = 0.0;
        k.variation = 0.0;
        k.pullout_20C_Nm = 0.9375;
        let mut ti = TemperatureInputs::default();
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        let mut k = links();
        k.inner_grade = crate::engine::grades::grade("Y30");
        k.br20_T = 0.37;
        k.alpha_br = -0.002;
        k.tmax_lib_C = NumOrText::Num(250.0);
        k.outer_grade = k.inner_grade;
        k.outer_br20_T = k.br20_T;
        k.outer_tmax_lib_C = k.tmax_lib_C;
        k
    }

    /// `k` with the outer ring (Br, rating, grade) of `from`.
    fn with_outer_of(mut k: TemperatureLinks, from: &TemperatureLinks) -> TemperatureLinks {
        k.outer_br20_T = from.outer_br20_T;
        k.outer_tmax_lib_C = from.outer_tmax_lib_C;
        k.outer_grade = from.outer_grade;
        k
```

with:

```rust
        let mut k = links();
        k.inner_grade = crate::engine::grades::grade("Y30");
        k.br20_T = 0.37;
        k.inner_alpha_br = -0.002;
        k.inner_magnet_density_g_mm3 = 0.005;
        k.tmax_lib_C = NumOrText::Num(250.0);
        k.outer_grade = k.inner_grade;
        k.outer_br20_T = k.br20_T;
        k.outer_alpha_br = k.inner_alpha_br;
        k.outer_tmax_lib_C = k.tmax_lib_C;
        k
    }

    /// `k` with the outer ring (Br, alpha, rating, grade) of `from`.
    fn with_outer_of(mut k: TemperatureLinks, from: &TemperatureLinks) -> TemperatureLinks {
        k.outer_br20_T = from.outer_br20_T;
        k.outer_alpha_br = from.outer_alpha_br;
        k.outer_tmax_lib_C = from.outer_tmax_lib_C;
        k.outer_grade = from.outer_grade;
        k
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        let e20 = Deviations::only(DeviationId::E20);
        let ti = TemperatureInputs::default();
        let mixed = with_outer_of(links(), &ferrite_links());
        let mut ferrite = ferrite_links();
        ferrite.alpha_br = mixed.alpha_br; // the calculator's one alpha (A4)
        let r = compute(&ti, &mixed, e20);
        assert_eq!(
            (r.demag.demag_ring.as_str(), r.demag.cold_ring.as_str()),
```

with:

```rust
        let e20 = Deviations::only(DeviationId::E20);
        let ti = TemperatureInputs::default();
        let mixed = with_outer_of(links(), &ferrite_links());
        let ferrite = ferrite_links();
        let r = compute(&ti, &mixed, e20);
        assert_eq!(
            (r.demag.demag_ring.as_str(), r.demag.cold_ring.as_str()),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            want.cold_limit_C = cold.cold_limit_C;
            want.cold_ring = RING_OUTER.to_owned();
            want.cold_check = cold.cold_check;
            want
        });
        // The stored NdFeB fields are past Y30's knee at room temperature: the cold check fails.
        assert_eq!(r.demag.cold_check, "Below the cold demagnetization limit");
        assert_eq!(r.summary.verdict, VERDICT_CHECK);
        // Swapped, the rings trade places.
        let mut swapped = with_outer_of(ferrite_links(), &links());
        swapped.alpha_br = mixed.alpha_br;
        let s = compute(&ti, &swapped, e20);
        assert_eq!(
            (s.demag.demag_ring.as_str(), s.demag.cold_ring.as_str()),
```

with:

```rust
            want.cold_limit_C = cold.cold_limit_C;
            want.cold_ring = RING_OUTER.to_owned();
            want.cold_check = cold.cold_check;
            // Decision A2-7: the torque at the limit goes with both rings' Br, each ring with
            // its own coefficient (the NdFeB inner, the ferrite outer).
            want.torque_at_limit_Nm = mixed.pullout_20C_Nm
                * (br_factor(-0.0012, want.magnet_limit_C)
                    * br_factor(-0.002, want.magnet_limit_C));
            want
        });
        // The stored NdFeB fields are past Y30's knee at room temperature: the cold check fails.
        assert_eq!(r.demag.cold_check, "Below the cold demagnetization limit");
        assert_eq!(r.summary.verdict, VERDICT_CHECK);
        // Swapped, the rings trade places.
        let swapped = with_outer_of(ferrite_links(), &links());
        let s = compute(&ti, &swapped, e20);
        assert_eq!(
            (s.demag.demag_ring.as_str(), s.demag.cold_ring.as_str()),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        );
        assert_eq!(s.demag.magnet_limit_C, r.demag.magnet_limit_C);
        assert_eq!(s.demag.cold_limit_C, r.demag.cold_limit_C);
        assert_eq!(
            run(&ti, &mixed),
            run(&ti, &links()),
            "the workbook reads the inner ring only"
        );
    }
```

with:

```rust
        );
        assert_eq!(s.demag.magnet_limit_C, r.demag.magnet_limit_C);
        assert_eq!(s.demag.cold_limit_C, r.demag.cold_limit_C);
        // Without E20 the workbook reads the inner ring's grade, Br and rating only (the outer
        // ring's coefficient still scales the torques, decision A2-7).
        let mut one_alpha = mixed.clone();
        one_alpha.outer_alpha_br = links().outer_alpha_br;
        assert_eq!(
            run(&ti, &one_alpha),
            run(&ti, &links()),
            "the workbook reads the inner ring only"
        );
    }

    #[test]
    fn each_ring_is_checked_with_its_own_coefficient() {
        // Decision A2-7: with E20 each ring's onsets use its own Br coefficient (the reverse
        // field scales with that ring's Br), the block shows the governing ring's, and the
        // torques at the limits go with both rings' Br.
        let e20 = Deviations::only(DeviationId::E20);
        let ti = TemperatureInputs::default();
        let mut k = links();
        k.outer_alpha_br = -0.0009; // a hypothetical outer grade with a flatter Br curve
        let r = compute(&ti, &k, e20);
        let d = &ti.demag;
        let onset = |alpha: f64| {
            let offset = demag_onset_C(
                1.29 / (2.0 * k.mu0) / 1000.0,
                1592.0,
                -0.005,
                d.knee_fraction,
                alpha,
                0.0,
            ) - 150.0;
            demag_onset_C(
                d.h_rev_likepole_kA_m,
                1592.0,
                -0.005,
                d.knee_fraction,
                alpha,
                offset,
            )
        };
        let (inner, outer) = (onset(-0.0012), onset(-0.0009));
        let (governing, alpha) = if outer < inner {
            (outer, -0.0009)
        } else {
            (inner, -0.0012)
        };
        assert_eq!(r.demag.onset_skipping_C, governing);
        assert_eq!(r.demag.alpha_br, alpha);
        let limit = governing - d.design_margin_C;
        assert_eq!(
            r.demag.torque_at_limit_Nm,
            k.pullout_20C_Nm * (br_factor(-0.0012, limit) * br_factor(-0.0009, limit))
        );
        // The inner block's mass for the bond load uses the inner ring's density.
        let mut ferrite = links();
        ferrite.inner_magnet_density_g_mm3 = 0.005;
        let a = compute(&ti, &ferrite, e20).adhesive;
        assert_eq!(
            a.block_mass_g,
            k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 0.005
        );
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/tests/grades.rs`, replace:

```rust
            }
            other => panic!("{}: unexpected vendor {other}", spec.part),
        }
    }
}
```

with:

```rust
            }
            other => panic!("{}: unexpected vendor {other}", spec.part),
        }
    }
}

#[test]
fn a_grade_ring_scales_with_its_own_alpha_and_weighs_at_its_density() {
    // Decision A2-7 end to end: hard ferrite Y30 picked for manual dimensions on both rings,
    // with the calculator's alpha (Calibration C22) left at the NdFeB -0.0012: the rings take
    // Y30's -0.20 %/C and 5.0 g/cm3. Setting C22 to Y30's value by hand (the A-1 way) gives
    // the same Calculator torques and temperature limits; only C35 and the parts C22 alone
    // drives (the Calibration prototype, C151) differ.
    use magcoupling::{DesignInputs, compute_all};
    let mut graded = DesignInputs::default();
    let m = &mut graded.coupling.magnets;
    m.part_inner = String::new();
    m.part_outer = String::new();
    m.grade_inner = "Y30".to_owned();
    m.grade_outer = "Y30".to_owned();
    let r = compute_all(&graded);
    assert_eq!(
        (r.model.inner_alpha_br_per_C, r.model.outer_alpha_br_per_C),
        (-0.002, -0.002)
    );
    let volume = r.model.inner_length_mm * r.model.inner_width_mm * r.model.inner_thickness_mm;
    let volume_o = r.model.outer_length_mm * r.model.outer_width_mm * r.model.outer_thickness_mm;
    assert_eq!(r.mass.magnets_g, 10.0 * (volume + volume_o) * 0.005);
    let mut by_hand = graded.clone();
    by_hand.calibration.alpha_br_per_C = -0.002;
    let h = compute_all(&by_hand);
    assert_eq!(r.model.pullout_Nm, h.model.pullout_Nm);
    assert_eq!(r.metal.torque_cold_high_Nm, h.metal.torque_cold_high_Nm);
    assert_eq!(
        r.temperature.summary.governing_limit_C,
        h.temperature.summary.governing_limit_C
    );
    assert_eq!(r.temperature.demag.alpha_br, -0.002);
    assert_eq!(r.model.alpha_br_per_C, -0.0012);
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors: ``error[E0061]: this function takes 19 arguments but 21 arguments were supplied`` (metal design) and ``error[E0609]: no field `inner_alpha_br_per_C` on type `model::ModelResults` ``.

- [ ] **Step 3: Give each grade-mode ring its own coefficient and density**

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E14 (all applied), probes, and the `Deviations` switch |
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa` and `end_effect_check`) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27) |
```

with:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E14 (all applied), probes, and the `Deviations` switch |
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup; a grade-mode ring reads its alpha and density (decision A2-7) |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa` and `end_effect_check`) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below, and each ring's alpha(Br) and magnet density used: a grade-mode ring's grade, decision A2-7), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        m.pullout_20C_Nm,
        ci.op_temp_C,
        cal_in.alpha_br_per_C,
        m.corner_gap_mm,
        m.face_gap_mm,
        m.cup_od_mm,
```

with:

```rust
        m.pullout_20C_Nm,
        ci.op_temp_C,
        cal_in.alpha_br_per_C,
        m.inner_alpha_br_per_C,
        m.outer_alpha_br_per_C,
        m.corner_gap_mm,
        m.face_gap_mm,
        m.cup_od_mm,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        op_temp_C: ci.op_temp_C,
        npole: ci.npole,
        br20_T: m.inner_br_T,
        alpha_br: cal_in.alpha_br_per_C,
        tmax_lib_C: m.inner_tmax_C,
        mu0: ci.mu0,
        pullout_op_Nm: m.pullout_Nm,
```

with:

```rust
        op_temp_C: ci.op_temp_C,
        npole: ci.npole,
        br20_T: m.inner_br_T,
        inner_alpha_br: m.inner_alpha_br_per_C,
        outer_alpha_br: m.outer_alpha_br_per_C,
        inner_magnet_density_g_mm3: m.inner_magnet_density_g_mm3,
        tmax_lib_C: m.inner_tmax_C,
        mu0: ci.mu0,
        pullout_op_Nm: m.pullout_Nm,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/assumptions.rs`, replace:

```rust
        id: "br_temperature_coefficient",
        label: "Br temperature coefficient",
        paths: &["calibration.alpha_br_per_C"],
        rationale: "Reversible remanence coefficient: Br(T) = Br(20 °C) · (1 + α (T − 20 °C)), and torque scales with Br², on every sheet. −0.12 %/°C is the sintered NdFeB value of every library part's grade.",
        source: "Workbook Calibration!C22 (−0.0012 /°C); Addendum A grade table (K&J, sintered NdFeB).",
    },
    Assumption {
```

with:

```rust
        id: "br_temperature_coefficient",
        label: "Br temperature coefficient",
        paths: &["calibration.alpha_br_per_C"],
        rationale: "Reversible remanence coefficient: Br(T) = Br(20 °C) · (1 + α (T − 20 °C)), and torque scales with Br², on every sheet. −0.12 %/°C is the sintered NdFeB value of every library part's grade; a ring in the grade mode (manual dimensions with a grade) takes its grade's own coefficient (decision A2-7).",
        source: "Workbook Calibration!C22 (−0.0012 /°C); Addendum A grade table (K&J, sintered NdFeB).",
    },
    Assumption {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/grades.rs`, replace:

```rust
//! E20 the Hcj and beta of each ring's grade (the demagnetization block checks
//! both rings; the weaker governs, A-1 plan decision A13).
//! Hcb, (BH)max, mu_rec, the coefficient ranges and the reference beta are for
//! display. alpha(Br) and density are recorded but not read: the calculator
//! keeps its single alpha input (Calibration!C22) and the NdFeB magnet density
//! for every magnet (decision table of the A-1 plan).

/// K&J Magnetics, Neodymium Magnet Specifications & Tolerances (the NdFeB basis, decision D1/1).
pub const KJ_SPECS: &str = "https://www.kjmagnetics.com/neodymium-magnet-specifications.asp";
```

with:

```rust
//! E20 the Hcj and beta of each ring's grade (the demagnetization block checks
//! both rings; the weaker governs, A-1 plan decision A13).
//! Hcb, (BH)max, mu_rec, the coefficient ranges and the reference beta are for
//! display. alpha(Br) and density are read for a ring in the grade mode (manual
//! dimensions with a grade: Addendum A-2 decision A2-7); a library part (every one
//! sintered NdFeB, whose grade values equal them) keeps the calculator's single alpha
//! (Calibration!C22) and the NdFeB density.

/// K&J Magnetics, Neodymium Magnet Specifications & Tolerances (the NdFeB basis, decision D1/1).
pub const KJ_SPECS: &str = "https://www.kjmagnetics.com/neodymium-magnet-specifications.asp";
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/grades.rs`, replace:

```rust
    pub hcb_kA_m: f64,
    /// Maximum energy product [kJ/m³] (display).
    pub bhmax_kJ_m3: f64,
    /// Reversible temperature coefficient of Br [1/°C] (recorded, not read).
    pub alpha_br_per_C: f64,
    /// beta(Hcj) the engine uses with E20 [1/°C]: the reference value, except
    /// N42SH, which keeps the workbook's -0.005 (decision 18 A).
```

with:

```rust
    pub hcb_kA_m: f64,
    /// Maximum energy product [kJ/m³] (display).
    pub bhmax_kJ_m3: f64,
    /// Reversible temperature coefficient of Br [1/°C] (read in the grade mode, decision A2-7).
    pub alpha_br_per_C: f64,
    /// beta(Hcj) the engine uses with E20 [1/°C]: the reference value, except
    /// N42SH, which keeps the workbook's -0.005 (decision 18 A).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/grades.rs`, replace:

```rust
    pub mu_rec: Option<f64>,
    /// Maximum operating temperature [°C] (the calibration rating of the demag block).
    pub tmax_C: f64,
    /// Density [g/mm³] (recorded, not read).
    pub density_g_mm3: f64,
    pub sources: GradeSources,
    /// The report's resolutions (R#) and flags for this grade.
```

with:

```rust
    pub mu_rec: Option<f64>,
    /// Maximum operating temperature [°C] (the calibration rating of the demag block).
    pub tmax_C: f64,
    /// Density [g/mm³] (read in the grade mode, decision A2-7).
    pub density_g_mm3: f64,
    pub sources: GradeSources,
    /// The report's resolutions (R#) and flags for this grade.
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
/// cold limits, the radial clearance stack, duty, the axial stack, the optional
/// aluminium adapter and the hybrid mass. `ret` is [`retainers`]' result;
/// `cup_density_g_mm3` is the density the mass model gives the web
/// (`model::cup_boss_density`), read only by E16.
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn compute(
    md: &MetalDesignInputs,
```

with:

```rust
/// cold limits, the radial clearance stack, duty, the axial stack, the optional
/// aluminium adapter and the hybrid mass. `ret` is [`retainers`]' result;
/// `cup_density_g_mm3` is the density the mass model gives the web
/// (`model::cup_boss_density`), read only by E16. `alpha_br` is the calculator's single
/// alpha (C17; the prototype's baseline C151); `alpha_inner` and `alpha_outer` are each
/// ring's (decision A2-7: a grade-mode ring's grade, else `alpha_br`), and the torques go with
/// their product.
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn compute(
    md: &MetalDesignInputs,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
    torque_20C_Nm: f64,
    op_temp_C: f64,
    alpha_br: f64,
    corner_gap_mm: f64,
    face_gap_mm: f64,
    cup_od_mm: f64,
```

with:

```rust
    torque_20C_Nm: f64,
    op_temp_C: f64,
    alpha_br: f64,
    alpha_inner: f64,
    alpha_outer: f64,
    corner_gap_mm: f64,
    face_gap_mm: f64,
    cup_od_mm: f64,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
    dev: Deviations,
) -> MetalDesignResults {
    let th = |T: f64| br_factor(alpha_br, T); // Python lambda th
    let cold = torque_20C_Nm * th(md.min_temp_C).powi(2);
    let hot_low = torque_op_Nm * (1.0 - md.variation);
    let cold_high = cold * (1.0 + md.variation);
    let clearance = (ret.liner_id_mm - ret.sleeve_od_mm) / 2.0;
```

with:

```rust
    dev: Deviations,
) -> MetalDesignResults {
    let th = |T: f64| br_factor(alpha_br, T); // Python lambda th
    // Decision A2-7: torque ~ Br_i(T) Br_o(T), each ring with its coefficient; with one
    // coefficient these are th(T)**2 and (th(a) / th(b))**2, bit for bit.
    let th2 = |T: f64| br_factor(alpha_inner, T) * br_factor(alpha_outer, T);
    let ratio2 = |a: f64, b: f64| {
        (br_factor(alpha_inner, a) / br_factor(alpha_inner, b))
            * (br_factor(alpha_outer, a) / br_factor(alpha_outer, b))
    };
    let cold = torque_20C_Nm * th2(md.min_temp_C);
    let hot_low = torque_op_Nm * (1.0 - md.variation);
    let cold_high = cold * (1.0 + md.variation);
    let clearance = (ret.liner_id_mm - ret.sleeve_od_mm) / 2.0;
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
    let removed =
        PI / 4.0 * (md.adapter_pilot_dia_mm.powi(2) - bore_mm.powi(2)) * md.web_mm * web_density;
    let hybrid = mass_total_g - boss_mass_g - removed + adapter + md.adapter_hardware_g;
    let cold_for_min = md.required_min_Nm * (th(md.min_temp_C) / th(op_temp_C)).powi(2);
    MetalDesignResults {
        torque_op_Nm,
        torque_20C_Nm,
```

with:

```rust
    let removed =
        PI / 4.0 * (md.adapter_pilot_dia_mm.powi(2) - bore_mm.powi(2)) * md.web_mm * web_density;
    let hybrid = mass_total_g - boss_mass_g - removed + adapter + md.adapter_hardware_g;
    let cold_for_min = md.required_min_Nm * ratio2(md.min_temp_C, op_temp_C);
    MetalDesignResults {
        torque_op_Nm,
        torque_20C_Nm,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
        running_clearance_mm: min_run,
        op_temp_C,
        alpha_br_per_C: alpha_br,
        required_20C_Nm: md.required_min_Nm / (th(op_temp_C).powi(2) * (1.0 - md.variation)),
        hot_margin: torque_op_Nm / md.required_min_Nm - 1.0,
        corner_gap_mm,
        sleeve_liner_clearance_mm: clearance,
```

with:

```rust
        running_clearance_mm: min_run,
        op_temp_C,
        alpha_br_per_C: alpha_br,
        required_20C_Nm: md.required_min_Nm / (th2(op_temp_C) * (1.0 - md.variation)),
        hot_margin: torque_op_Nm / md.required_min_Nm - 1.0,
        corner_gap_mm,
        sleeve_liner_clearance_mm: clearance,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
        magnetic_cycles: slip_f * md.slip_event_s * md.life_events,
        slip_loss_W: loss,
        slip_energy_J: energy,
        required_20C_zero_scatter_Nm: md.required_min_Nm / th(op_temp_C).powi(2),
        torque_cold_zero_var_Nm: cold,
        axial_stack_mm: stack,
        rotating_od_mm: rot_od,
```

with:

```rust
        magnetic_cycles: slip_f * md.slip_event_s * md.life_events,
        slip_loss_W: loss,
        slip_energy_J: energy,
        required_20C_zero_scatter_Nm: md.required_min_Nm / th2(op_temp_C),
        torque_cold_zero_var_Nm: cold,
        axial_stack_mm: stack,
        rotating_od_mm: rot_od,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
//! `harmonic_amplitude`, `shear_stress`, shared with the sweeps). Two formulas
//! several sheets share live here once, so they stay bit-identical everywhere:
//! `corner_radius` (√(r_face² + (w/2)²)) and `br_factor` (1 + α (T − 20 °C)).
//! `mass_estimate` (Calculator rows 110-115) is ported with `MassResults`.
//!
//! The harmonic set is the Rust-only assumption `coupling.max_harmonic` (Addendum A3):
//! the odd harmonics 1, 3, ... up to 11 ([`ODD_HARMONICS`], [`harmonic_count`]), the
```

with:

```rust
//! `harmonic_amplitude`, `shear_stress`, shared with the sweeps). Two formulas
//! several sheets share live here once, so they stay bit-identical everywhere:
//! `corner_radius` (√(r_face² + (w/2)²)) and `br_factor` (1 + α (T − 20 °C)).
//! `mass_estimate` (Calculator rows 110-115) is ported with `MassResults`. A ring in the
//! grade mode (manual dimensions with a grade) takes its grade's alpha(Br) and density
//! (Addendum A-2 decision A2-7, [`ResolvedMagnet::alpha_br`], [`ResolvedMagnet::density_g_mm3`]).
//!
//! The harmonic set is the Rust-only assumption `coupling.max_harmonic` (Addendum A3):
//! the odd harmonics 1, 3, ... up to 11 ([`ODD_HARMONICS`], [`harmonic_count`]), the
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                "The library part's grade, or the grade picked for manual dimensions; blank for a manual magnet without a grade."),
            outer_grade: String => out_rust_only("", "Outer magnet grade used",
                "As the inner grade, for the outer ring."),
        }
    }
}
```

with:

```rust
                "The library part's grade, or the grade picked for manual dimensions; blank for a manual magnet without a grade."),
            outer_grade: String => out_rust_only("", "Outer magnet grade used",
                "As the inner grade, for the outer ring."),
            inner_alpha_br_per_C: f64 => out_rust_only("1/°C", "Inner Br temperature coefficient used",
                "Decision A2-7: the grade's for manual dimensions with a grade picked; else the calculator's alpha (Calibration C22, shown in C35)."),
            outer_alpha_br_per_C: f64 => out_rust_only("1/°C", "Outer Br temperature coefficient used",
                "As the inner coefficient, for the outer ring."),
            inner_magnet_density_g_mm3: f64 => out_rust_only("g/mm³", "Inner magnet density used",
                "Decision A2-7: the grade's for manual dimensions with a grade picked; else NdFeB, 7.5 g/cm³ (C110)."),
            outer_magnet_density_g_mm3: f64 => out_rust_only("g/mm³", "Outer magnet density used",
                "As the inner density, for the outer ring."),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    pub tmax_C: NumOrText,
    /// The part's grade, or the grade picked for manual dimensions (Addendum A6).
    pub grade: Option<&'static Grade>,
}

/// Library values when the part is found (workbook IFERROR/INDEX/MATCH); else the
```

with:

```rust
    pub tmax_C: NumOrText,
    /// The part's grade, or the grade picked for manual dimensions (Addendum A6).
    pub grade: Option<&'static Grade>,
    /// Whether the ring is in the grade mode: manual dimensions with a grade picked (the
    /// part is not in the library). Only then does the grade supply alpha(Br) and density.
    pub from_grade: bool,
}

impl ResolvedMagnet {
    /// The ring's reversible Br coefficient [1/°C]: its grade's in the grade mode (decision
    /// A2-7), else `calculator_alpha`, the calculator's single alpha (Calibration C22), which
    /// equals every library part's sintered NdFeB grade at its default.
    pub fn alpha_br(&self, calculator_alpha: f64) -> f64 {
        match self.grade {
            Some(g) if self.from_grade => g.alpha_br_per_C,
            _ => calculator_alpha,
        }
    }

    /// The ring's magnet density [g/mm³]: its grade's in the grade mode (decision A2-7), else
    /// the NdFeB density the workbook uses for every magnet.
    pub fn density_g_mm3(&self) -> f64 {
        match self.grade {
            Some(g) if self.from_grade => g.density_g_mm3,
            _ => NDFEB_DENSITY_G_MM3,
        }
    }
}

/// Library values when the part is found (workbook IFERROR/INDEX/MATCH); else the
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                    br_T: library::br_T(spec, dev),
                    tmax_C: NumOrText::Num(library::tmax_C(spec, dev)),
                    grade: grades::grade(library::grade_id(spec, dev)),
                },
                None => match grades::grade(grade) {
                    Some(g) => ResolvedMagnet {
```

with:

```rust
                    br_T: library::br_T(spec, dev),
                    tmax_C: NumOrText::Num(library::tmax_C(spec, dev)),
                    grade: grades::grade(library::grade_id(spec, dev)),
                    from_grade: false,
                },
                None => match grades::grade(grade) {
                    Some(g) => ResolvedMagnet {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                        br_T: g.br_T,
                        tmax_C: NumOrText::Num(g.tmax_C),
                        grade: Some(g),
                    },
                    None => ResolvedMagnet {
                        length_mm,
```

with:

```rust
                        br_T: g.br_T,
                        tmax_C: NumOrText::Num(g.tmax_C),
                        grade: Some(g),
                        from_grade: true,
                    },
                    None => ResolvedMagnet {
                        length_mm,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                        br_T,
                        tmax_C: NumOrText::Text(NOT_IN_LIBRARY),
                        grade: None,
                    },
                },
            }
```

with:

```rust
                        br_T,
                        tmax_C: NumOrText::Text(NOT_IN_LIBRARY),
                        grade: None,
                        from_grade: false,
                    },
                },
            }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    let al_i = py_min(1.0, pitch_share(mi.width_mm, a_i, mi.thickness_mm, N));
    let al_o = py_min(1.0, pitch_share(mo.width_mm, A_o, mo.thickness_mm, N));

    let bri = mi.br_T * br_factor(alpha_br, ci.op_temp_C);
    let bro = mo.br_T * br_factor(alpha_br, ci.op_temp_C);
    let h = shear_stress(
        bri,
        bro,
```

with:

```rust
    let al_i = py_min(1.0, pitch_share(mi.width_mm, a_i, mi.thickness_mm, N));
    let al_o = py_min(1.0, pitch_share(mo.width_mm, A_o, mo.thickness_mm, N));

    // Decision A2-7: each ring with its own coefficient (a grade-mode ring's grade, else C22).
    let (alpha_i, alpha_o) = (mi.alpha_br(alpha_br), mo.alpha_br(alpha_br));
    let bri = mi.br_T * br_factor(alpha_i, ci.op_temp_C);
    let bro = mo.br_T * br_factor(alpha_o, ci.op_temp_C);
    let h = shear_stress(
        bri,
        bro,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        outer_temp_check: temp_check(mo.tmax_C),
        inner_grade: mi.grade.map_or("", |g| g.id).to_owned(),
        outer_grade: mo.grade.map_or("", |g| g.id).to_owned(),
    }
}
```

with:

```rust
        outer_temp_check: temp_check(mo.tmax_C),
        inner_grade: mi.grade.map_or("", |g| g.id).to_owned(),
        outer_grade: mo.grade.map_or("", |g| g.id).to_owned(),
        inner_alpha_br_per_C: alpha_i,
        outer_alpha_br_per_C: alpha_o,
        inner_magnet_density_g_mm3: mi.density_g_mm3(),
        outer_magnet_density_g_mm3: mo.density_g_mm3(),
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    dev: Deviations,
) -> MassResults {
    let N = ci.npole as f64;
    let m_mag = N
        * (r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm
            + r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm)
        * NDFEB_DENSITY_G_MM3;
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density =
        cup_boss_density(ci.backiron, steel_density_g_mm3, al_density_g_mm3, dev);
```

with:

```rust
    dev: Deviations,
) -> MassResults {
    let N = ci.npole as f64;
    let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
    let volume_o = r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm;
    let (rho_i, rho_o) = (r.inner_magnet_density_g_mm3, r.outer_magnet_density_g_mm3);
    // Decision A2-7: each ring at its own density; rings of one density keep the workbook's
    // single product (bit for bit: N (V_i + V_o) rho).
    let m_mag = if rho_i == rho_o {
        N * (volume_i + volume_o) * rho_i
    } else {
        N * (volume_i * rho_i + volume_o * rho_o)
    };
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density =
        cup_boss_density(ci.backiron, steel_density_g_mm3, al_density_g_mm3, dev);
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct TemperatureLinks {
    pub op_temp_C: f64,                // Calculator C10
    pub npole: i64,                    // Calculator C5
    pub br20_T: f64,                   // Calculator C21
    pub alpha_br: f64,                 // Calibration C22
    pub tmax_lib_C: NumOrText,         // Calculator C22 ("n/a" for manual magnets)
    pub mu0: f64,                      // Calculator C43
    pub pullout_op_Nm: f64,            // Calculator C93
    pub pullout_20C_Nm: f64,           // Calculator C94
    pub inner_back_apothem_mm: f64,    // Calculator C8
    pub inner_length_mm: f64,          // Calculator C18
    pub inner_width_mm: f64,           // Calculator C19
    pub inner_thickness_mm: f64,       // Calculator C20
    pub hub_wall_mm: f64,              // Calculator C38
    pub active_length_mm: f64,         // Calculator C33
    pub outer_back_apothem_mm: f64,    // Calculator C60
    pub mass_magnets_g: f64,           // Calculator C110
    pub mass_cup_g: f64,               // Calculator C111
    pub mass_hub_g: f64,               // Calculator C112
    pub mass_boss_g: f64,              // Calculator C113
    pub slip_rpm: f64,                 // Metal design C85
    pub slip_event_s: f64,             // Metal design C87
    pub life_events: f64,              // Metal design C88
    pub measured_drag_Nm: Option<f64>, // Metal design C90
    pub cold_high_Nm: f64,             // Metal design C10
    pub required_min_Nm: f64,          // Metal design C7
    pub variation: f64,                // Metal design C18
    pub min_temp_C: f64,               // Metal design C16
    pub magnetic_cycles: f64,          // Metal design C89
    pub bond_inner_mm: f64,            // Metal design C120
    pub bond_outer_mm: f64,            // Metal design C121
    pub sleeve_mm: f64,                // Metal design C25
    pub liner_mm: f64,                 // Metal design C26
    pub sleeve_id_mm: f64,             // Metal design C175
    pub sleeve_od_mm: f64,             // Metal design C176
    pub liner_od_mm: f64,              // Metal design C177
    pub liner_id_mm: f64,              // Metal design C178
    pub cap_face_mm: f64,              // Metal design C167
    pub cup_wall_mm: f64,              // Metal design C122 (E17: the aluminium cup's wall)
    pub web_mm: f64,                   // Metal design C125 (E17: the aluminium web)
    pub hardware_g: f64,               // Metal design C128
    pub retainers_g: f64,              // Metal design C46
    pub cap_g: f64,                    // Metal design C180
    pub endplates_g: f64,              // Metal design C181
    /// The inner magnet's grade (`model::ResolvedMagnet::grade`): E20 reads its Hcj and beta.
    pub inner_grade: Option<&'static Grade>,
    /// E20 (Decisions to confirm, A13): the outer ring's Br at 20 °C (Calculator C31), rating
```

with:

```rust
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct TemperatureLinks {
    pub op_temp_C: f64, // Calculator C10
    pub npole: i64,     // Calculator C5
    pub br20_T: f64,    // Calculator C21
    /// Decision A2-7: each ring's Br coefficient (`model.inner_alpha_br_per_C`,
    /// `outer_alpha_br_per_C`): a grade-mode ring's grade, else Calibration C22.
    pub inner_alpha_br: f64,
    pub outer_alpha_br: f64,
    /// Decision A2-7: the inner ring's magnet density (`model.inner_magnet_density_g_mm3`),
    /// the bond-load block mass (C82; the workbook's literal 0.0075 for NdFeB).
    pub inner_magnet_density_g_mm3: f64,
    pub tmax_lib_C: NumOrText, // Calculator C22 ("n/a" for manual magnets)
    pub mu0: f64,              // Calculator C43
    pub pullout_op_Nm: f64,    // Calculator C93
    pub pullout_20C_Nm: f64,   // Calculator C94
    pub inner_back_apothem_mm: f64, // Calculator C8
    pub inner_length_mm: f64,  // Calculator C18
    pub inner_width_mm: f64,   // Calculator C19
    pub inner_thickness_mm: f64, // Calculator C20
    pub hub_wall_mm: f64,      // Calculator C38
    pub active_length_mm: f64, // Calculator C33
    pub outer_back_apothem_mm: f64, // Calculator C60
    pub mass_magnets_g: f64,   // Calculator C110
    pub mass_cup_g: f64,       // Calculator C111
    pub mass_hub_g: f64,       // Calculator C112
    pub mass_boss_g: f64,      // Calculator C113
    pub slip_rpm: f64,         // Metal design C85
    pub slip_event_s: f64,     // Metal design C87
    pub life_events: f64,      // Metal design C88
    pub measured_drag_Nm: Option<f64>, // Metal design C90
    pub cold_high_Nm: f64,     // Metal design C10
    pub required_min_Nm: f64,  // Metal design C7
    pub variation: f64,        // Metal design C18
    pub min_temp_C: f64,       // Metal design C16
    pub magnetic_cycles: f64,  // Metal design C89
    pub bond_inner_mm: f64,    // Metal design C120
    pub bond_outer_mm: f64,    // Metal design C121
    pub sleeve_mm: f64,        // Metal design C25
    pub liner_mm: f64,         // Metal design C26
    pub sleeve_id_mm: f64,     // Metal design C175
    pub sleeve_od_mm: f64,     // Metal design C176
    pub liner_od_mm: f64,      // Metal design C177
    pub liner_id_mm: f64,      // Metal design C178
    pub cap_face_mm: f64,      // Metal design C167
    pub cup_wall_mm: f64,      // Metal design C122 (E17: the aluminium cup's wall)
    pub web_mm: f64,           // Metal design C125 (E17: the aluminium web)
    pub hardware_g: f64,       // Metal design C128
    pub retainers_g: f64,      // Metal design C46
    pub cap_g: f64,            // Metal design C180
    pub endplates_g: f64,      // Metal design C181
    /// The inner magnet's grade (`model::ResolvedMagnet::grade`): E20 reads its Hcj and beta.
    pub inner_grade: Option<&'static Grade>,
    /// E20 (Decisions to confirm, A13): the outer ring's Br at 20 °C (Calculator C31), rating
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
#[allow(non_snake_case)] // unit suffixes, as the result names
struct RingDemag {
    br20_T: f64,
    tmax_lib_C: NumOrText,
    hcj20: f64,
    beta: f64,
```

with:

```rust
#[allow(non_snake_case)] // unit suffixes, as the result names
struct RingDemag {
    br20_T: f64,
    /// The ring's Br coefficient (decision A2-7).
    alpha_br: f64,
    tmax_lib_C: NumOrText,
    hcj20: f64,
    beta: f64,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
}

#[allow(non_snake_case)] // Python names (T)
fn ring_demag(
    d: &DemagInputs,
    k: &TemperatureLinks,
    br20_T: f64,
    tmax_lib_C: NumOrText,
    grade: Option<&'static Grade>,
    dev: Deviations,
```

with:

```rust
}

#[allow(non_snake_case)] // Python names (T)
#[allow(clippy::too_many_arguments)] // one ring's Br, coefficient, rating and grade
fn ring_demag(
    d: &DemagInputs,
    k: &TemperatureLinks,
    br20_T: f64,
    alpha_br: f64,
    tmax_lib_C: NumOrText,
    grade: Option<&'static Grade>,
    dev: Deviations,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let t_ref = if cold_side {
        f64::INFINITY
    } else {
        demag_onset_C(h_ref, hcj20, beta, d.knee_fraction, k.alpha_br, 0.0)
    };
    // the workbook errors out when the magnet is not in the library; Python leaves the onset uncalibrated
    let offset = match tmax_lib_C {
```

with:

```rust
    let t_ref = if cold_side {
        f64::INFINITY
    } else {
        demag_onset_C(h_ref, hcj20, beta, d.knee_fraction, alpha_br, 0.0)
    };
    // the workbook errors out when the magnet is not in the library; Python leaves the onset uncalibrated
    let offset = match tmax_lib_C {
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        if cold_side {
            f64::INFINITY
        } else {
            demag_onset_C(h, hcj20, beta, d.knee_fraction, k.alpha_br, offset)
        }
    };
    let onsets = [
```

with:

```rust
        if cold_side {
            f64::INFINITY
        } else {
            demag_onset_C(h, hcj20, beta, d.knee_fraction, alpha_br, offset)
        }
    };
    let onsets = [
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        on(d.h_rev_likepole_kA_m),
        on(d.h_rev_single_ring_kA_m),
    ];
    let thf = |T: f64| br_factor(k.alpha_br, T).powi(2);
    // E20 cold side: the hot limit is the rating. A magnet with no rating has no hot limit
    // (+inf, so the adhesive governs C12) and no torque at it (NaN, where the workbook
    // formula would give +inf).
```

with:

```rust
        on(d.h_rev_likepole_kA_m),
        on(d.h_rev_single_ring_kA_m),
    ];
    let thf = |T: f64| torque_factor(k, T);
    // E20 cold side: the hot limit is the rating. A magnet with no rating has no hot limit
    // (+inf, so the adhesive governs C12) and no torque at it (NaN, where the workbook
    // formula would give +inf).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    // E20 cold side: the skipping case (the largest reverse field) governs, as on the hot side.
    let cold = |h: f64| {
        if cold_side {
            NumOrText::Num(cold_onset_C(h, hcj20, beta, d.knee_fraction, k.alpha_br))
        } else {
            NumOrText::Text(NO_COLD_ONSET)
        }
```

with:

```rust
    // E20 cold side: the skipping case (the largest reverse field) governs, as on the hot side.
    let cold = |h: f64| {
        if cold_side {
            NumOrText::Num(cold_onset_C(h, hcj20, beta, d.knee_fraction, alpha_br))
        } else {
            NumOrText::Text(NO_COLD_ONSET)
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    };
    RingDemag {
        br20_T,
        tmax_lib_C,
        hcj20,
        beta,
```

with:

```rust
    };
    RingDemag {
        br20_T,
        alpha_br,
        tmax_lib_C,
        hcj20,
        beta,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        cold_limit,
        cold_ok,
    }
}

/// E20: whether cold limit `a` is higher than `b` (a number is higher than "n/a").
```

with:

```rust
        cold_limit,
        cold_ok,
    }
}

/// Torque at magnet temperature `T` over torque at 20 °C: Br_i(T) Br_o(T) / (Br_i Br_o), each
/// ring with its coefficient (decision A2-7); with one coefficient the workbook's
/// (1 + α (T − 20))², bit for bit.
#[allow(non_snake_case)] // Python name (T)
fn torque_factor(k: &TemperatureLinks, T: f64) -> f64 {
    br_factor(k.inner_alpha_br, T) * br_factor(k.outer_alpha_br, T)
}

/// E20: whether cold limit `a` is higher than `b` (a number is higher than "n/a").
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust

    // ---- demagnetization
    let d = &ti.demag;
    let inner = ring_demag(d, k, k.br20_T, k.tmax_lib_C, k.inner_grade, dev);
    // E20 (Decisions to confirm, A13): the outer ring is checked too, with its own Br, rating
    // and grade. The ring with the lower magnet limit governs and the block shows it whole
    // (the inner ring on a tie, so identical rings keep the inner ring's block bit for bit);
    // the cold side shows the ring with the higher cold limit and passes only if both rings
    // pass. The workbook checks the inner ring only.
    let outer = dev
        .is_on(DeviationId::E20)
        .then(|| ring_demag(d, k, k.outer_br20_T, k.outer_tmax_lib_C, k.outer_grade, dev));
    let (hot, hot_ring) = match outer {
        Some(o) if o.mag_lim < inner.mag_lim => (o, RING_OUTER),
        _ => (inner, RING_INNER),
```

with:

```rust

    // ---- demagnetization
    let d = &ti.demag;
    let inner = ring_demag(
        d,
        k,
        k.br20_T,
        k.inner_alpha_br,
        k.tmax_lib_C,
        k.inner_grade,
        dev,
    );
    // E20 (Decisions to confirm, A13): the outer ring is checked too, with its own Br, rating
    // and grade. The ring with the lower magnet limit governs and the block shows it whole
    // (the inner ring on a tie, so identical rings keep the inner ring's block bit for bit);
    // the cold side shows the ring with the higher cold limit and passes only if both rings
    // pass. The workbook checks the inner ring only.
    let outer = dev.is_on(DeviationId::E20).then(|| {
        ring_demag(
            d,
            k,
            k.outer_br20_T,
            k.outer_alpha_br,
            k.outer_tmax_lib_C,
            k.outer_grade,
            dev,
        )
    });
    let (hot, hot_ring) = match outer {
        Some(o) if o.mag_lim < inner.mag_lim => (o, RING_OUTER),
        _ => (inner, RING_INNER),
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let cold_ok = inner.cold_ok && outer.is_none_or(|o| o.cold_ok);
    let [on_al, on_po, on_lp, on_cu] = hot.onsets;
    let mag_lim = hot.mag_lim;
    let thf = |T: f64| br_factor(k.alpha_br, T).powi(2);
    let demag = DemagResults {
        br20_T: hot.br20_T,
        alpha_br: k.alpha_br,
        tmax_lib_C: hot.tmax_lib_C,
        h_ref_kA_m: hot.h_ref,
        t_ref_model_C: hot.t_ref,
```

with:

```rust
    let cold_ok = inner.cold_ok && outer.is_none_or(|o| o.cold_ok);
    let [on_al, on_po, on_lp, on_cu] = hot.onsets;
    let mag_lim = hot.mag_lim;
    let thf = |T: f64| torque_factor(k, T);
    let demag = DemagResults {
        br20_T: hot.br20_T,
        alpha_br: hot.alpha_br,
        tmax_lib_C: hot.tmax_lib_C,
        h_ref_kA_m: hot.h_ref,
        t_ref_model_C: hot.t_ref,
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    // ---- adhesive selection and loads
    let sel = selected_adhesive(ti.adhesive.selected);
    let area = k.inner_length_mm * k.inner_width_mm;
    let m_block = k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 0.0075; // literal, as Python
    let r_mid = k.inner_back_apothem_mm + k.inner_thickness_mm / 2.0;
    let Ft = k.cold_high_Nm / (npole * r_mid / 1000.0);
    let tau_b = Ft / area;
```

with:

```rust
    // ---- adhesive selection and loads
    let sel = selected_adhesive(ti.adhesive.selected);
    let area = k.inner_length_mm * k.inner_width_mm;
    // Python's literal 0.0075 (NdFeB); decision A2-7: a grade-mode inner ring's density.
    let m_block =
        k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * k.inner_magnet_density_g_mm3;
    let r_mid = k.inner_back_apothem_mm + k.inner_thickness_mm / 2.0;
    let Ft = k.cold_high_Nm / (npole * r_mid / 1000.0);
    let tau_b = Ft / area;
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --lib 2>&1 | grep "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test grades 2>&1 | grep "test result"
```

Expected: `test result: ok. 138 passed`, then `test result: ok. 10 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `138 passed`, `tests\grades.rs` 10 passed, the others as in Task 9 (parity, the differential data and every registry test unedited: library parts keep C22 and the NdFeB density).

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a2/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task10.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task10.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task10.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 5: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/metal_design.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/grades.rs magcoupling-rs/src/engine/assumptions.rs magcoupling-rs/tests/grades.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
feat(magcoupling-rs): each grade-mode ring's own alpha(Br) and density (decision A2-7)

A ring with manual dimensions and a grade takes the grade's Br coefficient and
density: Br at temperature, every torque at another temperature (Br_i Br_o),
its E20 onsets, the magnets' and the bond block's mass. Library parts and
gradeless magnets keep Calibration C22 and the NdFeB density: defaults, parity
and the differential data bit-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 11: Docs pass and the final gate

**Model:** `sonnet` (documentation with the exact text given; CLAUDE.md section 5). The final whole-branch review after this task runs on the session model (controller).

The repo rule (`docs/ai/01-meta.yaml`, coordination): after file-modifying work update 02-system (invariants, status),
03-structure (module layout), 04-memory (open questions) and 05-update-tracker; the crate README's layout and tests rows were kept
current task by task. `02-system.yaml` and `03-structure.yaml` did not parse as YAML before this plan (a plain scalar holding `: `,
e.g. 03-structure line 42); this task adds no new case of it (every new 03-structure value is quoted) and does not repair the old ones.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md` (status)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/lib.rs` (crate docs: the A-2 inputs, sizing and the assumptions panel)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/02-system.yaml` (magcoupling responsibility, invariants, status)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/03-structure.yaml` (the model and calibration entries; assumptions, sizing, housing; the tests list)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/04-memory.yaml` (the A-2 carries resolved; the M4, A-3 and M3 items this plan leaves; the E4 bondline item gets its owner and trigger)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/05-update-tracker.md` (the A-2 entry)

**Interfaces:**
- Consumes: everything above.
- Produces: docs that match the code (the repo rule); a green final gate; the default headline unchanged.

- [ ] **Step 1: Crate README status, crate docs and docs/ai**

In `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/02-system.yaml`, replace:

```yaml
      vendored Python oracle reference/magcoupling-py (workbook port).
      Workbook-exact except registered, approved corrections (deviation
      registry, M1 audit E1-E14, Addendum A E15-E20). Engine complete (M2), every module but fields3d;
      Addendum A-1 adds the grade table, materials per part with physics links, and six warnings.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
```

with:

```yaml
      vendored Python oracle reference/magcoupling-py (workbook port).
      Workbook-exact except registered, approved corrections (deviation
      registry, M1 audit E1-E14, Addendum A E15-E20). Engine complete (M2), every module but fields3d;
      Addendum A-1 adds the grade table, materials per part with physics links, and six warnings;
      Addendum A-2 adds the harmonic set, the assumptions registry, inverse sizing and the space claim, with an axial housing that follows the length override.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/02-system.yaml`, replace:

```yaml
      - A correction that changes more than 15 cells at defaults is a reviewed golden file (tests/data/deviations); one that changes none carries a registry probe.
      - The engine never panics on inputs; out-of-choice selector codes give NaN or "#N/A"; DesignInputs::validate() at input boundaries; no i64 arithmetic on input-derived values.
      - Rust-only inputs and results (rust_only metadata) have no cell and are skipped by the parity, metadata and differential tests; gen_differential.py never passes a Rust-only input to Python.
      - Every Rust-only selector's default reproduces the ported behaviour bit for bit: part materials at code 1 are the workbook inputs, a blank grade is the manual mode, coercivity_source 1 with the N42SH grade gives the workbook's 1592 kA/m and -0.005 /C.

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
```

with:

```yaml
      - A correction that changes more than 15 cells at defaults is a reviewed golden file (tests/data/deviations); one that changes none carries a registry probe.
      - The engine never panics on inputs; out-of-choice selector codes give NaN or "#N/A"; DesignInputs::validate() at input boundaries; no i64 arithmetic on input-derived values.
      - Rust-only inputs and results (rust_only metadata) have no cell and are skipped by the parity, metadata and differential tests; gen_differential.py never passes a Rust-only input to Python.
      - Every Rust-only selector's default reproduces the ported behaviour bit for bit: part materials at code 1 are the workbook inputs, a blank grade is the manual mode, coercivity_source 1 with the N42SH grade gives the workbook's 1592 kA/m and -0.005 /C, max_harmonic 5 sums 1, 3, 5 in the workbook's order, a blank axial_length_mm keeps each ring's length and the housing inputs as typed.
      - With the axial length override set, the hub length, cup cavity depth and retainer span in effect follow the rings they bound (housing::axial_housing, decision A2-8), each input plus its ring's length change from the ring's own length (for the retainer span, the longer ring's), and never shorter than that ring; every result that reads them (masses, heat capacity, both axial stacks, the space claim) reads the values in effect, which housing.* reports.
      - The E7 peak search (model::peak_off_half_pitch) returns None exactly when half a pitch is the maximum (1e-12 relative), so every E7-neutral design keeps the workbook expression bit for bit; it finds every root of dT/dx in cos^2 x by recursion on derivatives (no closed form, no scan grid).
      - Inverse sizing (sizing::solve) evaluates compute_all only inside the free variable's slider range; a continuous variable's 64-cell scan is refined between samples at the first crossing, at each peak (golden section) and at each validity edge, so only a torque hump whose rise and fall both lie inside one cell is unseen (the documented limit). A value counts only if sizing::is_valid holds (the blocks fit, faceted blocks on their flats and arcs without overlapping, model::blocks_fit; the keyway leaves hub wall; f_end > 0, model::end_effect_in_range; a finite hot-low torque) and meets when its hot-low torque reaches the target; poles stay even.
      - Library parts and gradeless manual magnets use Calibration C22 and the NdFeB density; only a grade-mode ring (manual dimensions with a grade) takes its grade's alpha(Br) and density (decision A2-7).

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/02-system.yaml`, replace:

```yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied; next: Addendum A-2 (parameters and sizing), A-3, then M4 GUI"
```

with:

```yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim; next: Addendum A-3 (explanations), then M4 GUI"
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/03-structure.yaml`, replace:

```yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
```

with:

```yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20) and A-2 (harmonic set, assumptions, inverse sizing, space claim) complete
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/03-structure.yaml`, replace:

```yaml
    warnings: warnings.rs (Addendum A5 material warnings: WARNING_RULES, six rules with text, Severity and note id; thresholds SLEEVE_SIGMA_BASELINE_S_M 1.35e6, LOW_SATURATION_T 1.7, CTE_MISMATCH_LIMIT_PER_C 15e-6; WarningResults = DesignResults::warnings, Rust-only)
    material_library: material_library.rs (Addendum A5 materials library MATERIALS, 14 entries cited per value; EngineProps = workbook numbers where they exist (decisions 21-26), else sourced; BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, CAP_HOUSING_CHOICES, code 1 = the workbook's material; resolve -> PartProperties: the circuit in effect, the design flux density, the steel, body (no back iron), sleeve and cap values the engine reads)
    grades: grades.rs (Addendum A6 grade table GRADES, 17 grades cited per value from docs/analyses/2026-09-30-magcoupling-addendum-a-data.json; grade(id); N42SH keeps the workbook Hcj 1592 and beta -0.005)
    calibration: calibration.rs (Calibration sheet, 23 result cells, 16 default inputs)
    model: "model.rs (Calculator sheet: inputs (Rust-only grade_inner/grade_outer: a grade for manual dimensions, Addendum A6), ModelResults 73 result cells (Rust-only inner_grade/outer_grade), HARMONICS [1, 3, 5], peak_off_half_pitch, peak_angle (the one E7 gate), tau_at and at_pull_out (E7), corner_radius and br_factor (shared across sheets); mass_estimate and MassResults, 6 result cells)"
    metal_design: "metal_design.rs (Metal design sheet: MetalDesignInputs 48 fields; retainers, 9 result cells; compute, MetalDesignResults 49 result cells; VALIDATION_ITEMS)"
    materials: "materials.rs (Materials sheet: inputs, plus the Rust-only part selectors materials.parts (back_iron, sleeve_liner, cap_housing); AL7075 and AL6061 alloys; ScrewClasses::proof, NaN for a code outside 1-3; compute, MaterialsResults 7 result cells)"
    temperature: "temperature.rs (Temperature design sheet: 7 input groups, 37 cells; ADHESIVES table and selected_adhesive, NO_ADHESIVE (NaN, \"#N/A\") for a code outside 1-4; TemperatureLinks; compute, TemperatureResults 130 result cells and the uncelled adhesive name; demag_onset_C, cold_onset_C (E20, positive beta), ring_demag (E20: each ring's check, the weaker governs, A13), volkersen_peak_shear_MPa; Rust-only coercivity_source and the free-space fields of E17; Rust-only demag results hcj20_used/beta_used, demag_ring, cold_ring and the cold side)"
    clamps: "clamps.rs (Shaft clamps and Clamp screw sizes sheets: 25 input cells; SCREW_SIZES, TABLE_COLUMNS, MACHINING_STEPS; ScrewRow with 34 fields and 165 table cells; screw_class_name, \"#N/A\" for a code outside 1-3; compute, ClampResults 33 result cells plus the Rust-only length_note; the screw table lists after the scalars)"
    sweeps: "sweeps.rs (Gap sweep and Pole sweep sheets: GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES; SweepRow with 26 fields, 338 and 156 table cells; SweepContext with 20 fields (Python's 19 plus bond_inner, read only by E4); gap_sweep, pole_sweep)"
    api: api.rs (DesignInputs/DesignResults groups in Python order, the complete Python DesignResults; compute resolves the part materials first (material_library::resolve) and feeds the values in effect; compute_all applies Deviations::ALL; headline and HEADLINE (the 15 dashboard numbers, Python order, read with ResultSet::get, about 0.4 us per call in release); DesignInputs::validate; test-only compute_all_with, DesignInputs::defaults_with)
    ported: [meta.rs, compat.rs, constants.rs, calibration.rs, library.rs, grades.rs, material_library.rs, warnings.rs, model.rs, metal_design.rs, materials.rs, temperature.rs, clamps.rs, sweeps.rs, api.rs]
    remaining: [fields3d (M3)]
  features: gui and app (declared empty, M4); workbook-parity (test-only, enabled by a self dev-dependency)
  tests: tests/ (the integration tests parity.rs, differential.rs, python_schema.rs, schema.rs, deviations.rs, static_data.rs, robustness.rs, grades.rs, material_library.rs; common/mod.rs holds the PORTED_INPUTS / PORTED_RESULTS ratchets)
  test_data: tests/data/ (reference_values.json = copy of the vendored snapshot; input_schema.json via MAGCOUPLING_BLESS=1 cargo test --test schema; python_schema.json, static_data.json, differential/<group>.json (10 result groups plus helpers.json) and differential/full.json via tools/gen_differential.py; deviations/E3.json, E4.json and E5.json, the golden files of the broad corrections, via MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells)
  gate: linkage-sim-rs/scripts/gate.sh gates 4-7 (magcoupling-rs test, clippy -D warnings, wasm32 check; Python parity suite + gen_differential.py --check)
```

with:

```yaml
    warnings: warnings.rs (Addendum A5 material warnings: WARNING_RULES, six rules with text, Severity and note id; thresholds SLEEVE_SIGMA_BASELINE_S_M 1.35e6, LOW_SATURATION_T 1.7, CTE_MISMATCH_LIMIT_PER_C 15e-6; WarningResults = DesignResults::warnings, Rust-only)
    material_library: material_library.rs (Addendum A5 materials library MATERIALS, 14 entries cited per value; EngineProps = workbook numbers where they exist (decisions 21-26), else sourced; BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, CAP_HOUSING_CHOICES, code 1 = the workbook's material; resolve -> PartProperties: the circuit in effect, the design flux density, the steel, body (no back iron), sleeve and cap values the engine reads)
    grades: grades.rs (Addendum A6 grade table GRADES, 17 grades cited per value from docs/analyses/2026-09-30-magcoupling-addendum-a-data.json; grade(id); N42SH keeps the workbook Hcj 1592 and beta -0.005)
    calibration: "calibration.rs (Calibration sheet, 23 result cells, 16 default inputs; compute takes the Calculator's harmonic set; Rust-only tau7_Pa-tau11_Pa and end_effect_check)"
    assumptions: "assumptions.rs (Addendum A3 panel: ASSUMPTIONS, 14 rows over the 15 inputs flagged .assumption() with rationale and source; states, modified, any_modified, reset_to_workbook_defaults; the dependency-graph traceability test is plan A-3's)"
    sizing: "sizing.rs (Addendum A1 inverse sizing: FreeVariable (axial length, magnets per ring, ring radius), SCAN_CELLS 64, VALUE_TOLERANCE_MM 1e-9, is_valid (decision A2-4), solve -> Solved / NotReachable {best}: the scan refined at the first crossing, each peak and each validity edge; SizingError; the search's unit tests drive it with plain functions)"
    housing: "housing.rs (Addendum A1 housing: axial_housing, the hub length, cup cavity depth and retainer span in effect, which follow the length override, decision A2-8; the space claim: HousingResults = DesignResults::housing, Rust-only overshoot per axis, space_claim_check and the three dimensions in effect; the autofit classes of report 6.5 in its module doc)"
    model: "model.rs (Calculator sheet: inputs (Rust-only grade_inner/grade_outer: a grade for manual dimensions, Addendum A6; max_harmonic, the A3 harmonic set, MAX_HARMONIC_CHOICES, harmonic_count; axial_length_mm, the A1 length override), ModelResults 73 result cells (Rust-only inner_grade/outer_grade, tau7_Pa-tau11_Pa, end_effect_check, each ring's alpha and magnet density), HARMONICS [1, 3, 5] (the workbook set) and ODD_HARMONICS [1..11], peak_off_half_pitch (one root search for any odd set, decision 29), peak_angle (the one E7 gate), tau_at and at_pull_out (E7), harmonic_sum/harmonic_slot, end_effect_in_range/end_effect_check (audit M9 flag), pitch_share and blocks_fit (the A2-4 fit rule: flats, or arcs that do not overlap), ResolvedMagnet::alpha_br/density_g_mm3 (grade mode, decision A2-7), corner_radius and br_factor (shared across sheets); mass_estimate and MassResults, 6 result cells)"
    metal_design: "metal_design.rs (Metal design sheet: MetalDesignInputs 48 fields; retainers, 9 result cells; compute, MetalDesignResults 49 result cells; VALIDATION_ITEMS)"
    materials: "materials.rs (Materials sheet: inputs, plus the Rust-only part selectors materials.parts (back_iron, sleeve_liner, cap_housing); AL7075 and AL6061 alloys; ScrewClasses::proof, NaN for a code outside 1-3; compute, MaterialsResults 7 result cells plus the Rust-only cup_wall_suggested_mm, the A1 autofit's wall, decision 27)"
    temperature: "temperature.rs (Temperature design sheet: 7 input groups, 37 cells; ADHESIVES table and selected_adhesive, NO_ADHESIVE (NaN, \"#N/A\") for a code outside 1-4; TemperatureLinks; compute, TemperatureResults 130 result cells and the uncelled adhesive name; demag_onset_C, cold_onset_C (E20, positive beta), ring_demag (E20: each ring's check, the weaker governs, A13), volkersen_peak_shear_MPa; Rust-only coercivity_source and the free-space fields of E17; Rust-only demag results hcj20_used/beta_used, demag_ring, cold_ring and the cold side)"
    clamps: "clamps.rs (Shaft clamps and Clamp screw sizes sheets: 25 input cells; SCREW_SIZES, TABLE_COLUMNS, MACHINING_STEPS; ScrewRow with 34 fields and 165 table cells; screw_class_name, \"#N/A\" for a code outside 1-3; compute, ClampResults 33 result cells plus the Rust-only length_note; the screw table lists after the scalars)"
    sweeps: "sweeps.rs (Gap sweep and Pole sweep sheets: GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES; SweepRow with 26 fields, 338 and 156 table cells; SweepContext with 20 fields (Python's 19 plus bond_inner, read only by E4); gap_sweep, pole_sweep)"
    api: api.rs (DesignInputs/DesignResults groups in Python order, the complete Python DesignResults; compute resolves the part materials first (material_library::resolve) and feeds the values in effect; compute_all applies Deviations::ALL; headline and HEADLINE (the 15 dashboard numbers, Python order, read with ResultSet::get, about 0.4 us per call in release); DesignInputs::validate; test-only compute_all_with, DesignInputs::defaults_with)
    ported: [meta.rs, compat.rs, constants.rs, calibration.rs, library.rs, grades.rs, material_library.rs, warnings.rs, model.rs, metal_design.rs, materials.rs, temperature.rs, clamps.rs, sweeps.rs, api.rs]
    addendum_only: [assumptions.rs, sizing.rs, housing.rs]
    remaining: [fields3d (M3)]
  features: gui and app (declared empty, M4); workbook-parity (test-only, enabled by a self dev-dependency)
  tests: tests/ (the integration tests parity.rs, differential.rs, python_schema.rs, schema.rs, deviations.rs, static_data.rs, robustness.rs, grades.rs, material_library.rs, material_links.rs, assumptions.rs, sizing.rs; common/mod.rs holds the PORTED_INPUTS / PORTED_RESULTS ratchets)
  test_data: tests/data/ (reference_values.json = copy of the vendored snapshot; input_schema.json via MAGCOUPLING_BLESS=1 cargo test --test schema; python_schema.json, static_data.json, differential/<group>.json (10 result groups plus helpers.json) and differential/full.json via tools/gen_differential.py; deviations/E3.json, E4.json and E5.json, the golden files of the broad corrections, via MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells)
  gate: linkage-sim-rs/scripts/gate.sh gates 4-7 (magcoupling-rs test, clippy -D warnings, wasm32 check; Python parity suite + gen_differential.py --check)
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/04-memory.yaml`, replace:

```yaml
  - "RESOLVED 2026-09-30 (user): the E9 physics residuals at backiron = 0 become corrections E15-E17 inside the Addendum A5 materials work, each verified like the M1 audit and applied only after the user approves its numbers"
  - "RESOLVED 2026-09-30 (user): all 31 Addendum A verification decisions take option A (docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md section 8; data in 2026-09-30-magcoupling-addendum-a-data.json): K&J-minimum NdFeB basis, SuperMagnetMan 60 C Tmax deviation, ferrite beta +0.35 %/C, Recoma 20/26/30, materials table with published-minimum yields, E15-E18 registered (E17 closed form T1, end factor 0.7, stored free-space fields, hub included; Deviations::with/without combinator), per-grade Hcj/beta demag fix as a registered deviation, workbook values stay the defaults (N42SH Hcj 1592, beta -0.0050, 4140 sigma/mu_r/cp/CTE, 316L, 6061), design_flux_density_T feeds the wall check, cup wall stays an input (autofit suggests), harmonic search generalized with a root-pair guard, A2 v1 scope = dashboard + geometry callouts + five chains (about 159 paths)"
  - "M3 must apply E3 and E5 inside fields3d: rerun it with the corrected N42SH Br (E3) and multiply the web integral by 4 in fields3d.run (E5); see each entry's corrected_formula in magcoupling-rs/src/engine/deviations.rs"
  - "Addendum A engine plan: make the harmonic set a parameter and generalize peak_off_half_pitch beyond [1, 3, 5] (decision D7). The generalized search must stay cancellation-free (as 2fb6548 made the closed form): the tests a_vanishing_fifth_harmonic_does_not_lose_the_peak, peak_off_half_pitch_finds_the_brute_force_maximum and e7_finds_the_peak_at_a_fill_of_exactly_0_4 must keep passing"
  - "RESOLVED (Addendum A-1): the per-grade demagnetization fix is E20 (decision 19): each ring is checked with its own Hcj and beta, Br and rating and the weaker ring governs (A-1 decision A13; the workbook read only the inner ring against the outer blocks' fields), C44/C45 as overrides through the Rust-only temperature.demag.coercivity_source; ferrite (positive beta) is limited on the cold side (hot onsets +inf, rating as hot limit, Rust-only cold onsets/limit/check in the verdict, both rings); tests e20_* incl. the ferrite cold case through the custom-dimension mode and the mixed rings"
  - "RESOLVED (Addendum A-1): the E9 residuals are E15 (heat capacity), E16 (removed web disc) and E17 (aluminium eddy losses, T1 with the free-space fields), plus E18 (the aluminium hub's mismatch screen); each reproduces the report's changed-cell tables"
  - "M3 must recompute and re-bless: E17's three Rust-only free-space fields (b_hub_free_T 0.07832 T, b_cup_free_T 0.08764 T, web_integral_free_T2m2 6.837e-6 T2m2, pinned at 4 s.f. from fields at Br 1.29 T; apply E3) and the reverse fields of a non-default grade (the E20 ferrite probe scales the stored NdFeB fields by 0.37/1.29 as a placeholder)"
  - "Open (Addendum A-1 decision A8): E18 uses the report's Alliance 6061 modulus 68.9 GPa while the A5 library's 6061 record carries Kaiser's 68.3 GPa (decision 5): picking 6061 as back iron differs from the default aluminium body in C104, C105 and C201 only (tests/material_links.rs pins it). Unify when the user decides"
  - "Addendum A-2 carries: the harmonic set as a parameter with the generalized peak search (decision 29); a per-ring alpha(Br) and magnet density for a non-NdFeB grade (A-1 decision A4 kept one alpha and the NdFeB density); A1 inverse sizing, housing autofit and the space claim"
  - "Addendum A-2 must allow for two A-1 overrides of assumption inputs: with E20 and coercivity_source 1 (the default), C44 and C45 change nothing for a magnet with a grade (every library part), and a library back iron with its own design flux density (1018) replaces Materials C13 in the wall check; the A3 test that each assumption moves its dependent results must skip or document them. C45's slider (-0.008 to -0.001) cannot enter ferrite's +0.0035 with the source at 0: widen or split it (A-1 decisions A2, A9)"
  - "M3: fields3d.run gives 1.034632e-5; times 4 (E5) and formatted .4g that is 4.139e-5, not the M2 default 4.14e-5 (C125 differs by about 0.00009 W). M3 should pin the M2 default or expect the difference"
  - "Tidy after M2: magcoupling-rs/src/engine/temperature.rs is about 1,350 lines, past the architecture's ~800-line split point; split mechanically (inputs, results, compute, tests) when no correction work is in flight"
  - "Addendum A3 (workbook literals become inputs): E4 as approved adds the inner bondline input to the pole-sweep keyed-wall term, while the block-fit term keeps the workbook literal + 0.05 (the same bondline), so at bond_inner != 0.05 the two terms of the max() disagree (sweeps.rs pole_sweep); E4's test and golden file probe only 0.05"
  - "M4/A3: f_end = 1 - c_end * pole pitch / L goes negative inside the slider ranges (L down to 2 mm, c_end up to 0.5; final review: 59 of 3,000 random Calculator runs, 785 sweep rows), giving a negative pull-out that E7 makes more negative. Audit M9, not approved for M2: add the audit's domain guard or at least a GUI flag on f_end <= 0"
  - "M4: the workbook-parity feature is guarded by convention only (any downstream Cargo.toml can enable it). A compile_error! on all(feature = app, feature = workbook-parity) was NOT added: the self dev-dependency enables workbook-parity for every cargo test, so cargo test --features app would stop compiling. Decide the guard in the M4 plan (e.g. a CI check that the shipped build's feature set excludes it)"
```

with:

```yaml
  - "RESOLVED 2026-09-30 (user): the E9 physics residuals at backiron = 0 become corrections E15-E17 inside the Addendum A5 materials work, each verified like the M1 audit and applied only after the user approves its numbers"
  - "RESOLVED 2026-09-30 (user): all 31 Addendum A verification decisions take option A (docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md section 8; data in 2026-09-30-magcoupling-addendum-a-data.json): K&J-minimum NdFeB basis, SuperMagnetMan 60 C Tmax deviation, ferrite beta +0.35 %/C, Recoma 20/26/30, materials table with published-minimum yields, E15-E18 registered (E17 closed form T1, end factor 0.7, stored free-space fields, hub included; Deviations::with/without combinator), per-grade Hcj/beta demag fix as a registered deviation, workbook values stay the defaults (N42SH Hcj 1592, beta -0.0050, 4140 sigma/mu_r/cp/CTE, 316L, 6061), design_flux_density_T feeds the wall check, cup wall stays an input (autofit suggests), harmonic search generalized with a root-pair guard, A2 v1 scope = dashboard + geometry callouts + five chains (about 159 paths)"
  - "M3 must apply E3 and E5 inside fields3d: rerun it with the corrected N42SH Br (E3) and multiply the web integral by 4 in fields3d.run (E5); see each entry's corrected_formula in magcoupling-rs/src/engine/deviations.rs"
  - "RESOLVED (Addendum A-2, decision 29): one general E7 peak search for any odd set up to 11 (every root of dT/dx in cos^2 x by recursion on derivatives), and the harmonic set is the Rust-only assumption coupling.max_harmonic (default 5 = 1, 3, 5) summed by the Calculator, both circuit sums, the sweeps and the Calibration prototype; the three cancellation tests and the fill-0.4 test pass unedited, and the closed form is the tests' reference (within 1e-12 rad)"
  - "RESOLVED (Addendum A-1): the per-grade demagnetization fix is E20 (decision 19): each ring is checked with its own Hcj and beta, Br and rating and the weaker ring governs (A-1 decision A13; the workbook read only the inner ring against the outer blocks' fields), C44/C45 as overrides through the Rust-only temperature.demag.coercivity_source; ferrite (positive beta) is limited on the cold side (hot onsets +inf, rating as hot limit, Rust-only cold onsets/limit/check in the verdict, both rings); tests e20_* incl. the ferrite cold case through the custom-dimension mode and the mixed rings"
  - "RESOLVED (Addendum A-1): the E9 residuals are E15 (heat capacity), E16 (removed web disc) and E17 (aluminium eddy losses, T1 with the free-space fields), plus E18 (the aluminium hub's mismatch screen); each reproduces the report's changed-cell tables"
  - "M3 must recompute and re-bless: E17's three Rust-only free-space fields (b_hub_free_T 0.07832 T, b_cup_free_T 0.08764 T, web_integral_free_T2m2 6.837e-6 T2m2, pinned at 4 s.f. from fields at Br 1.29 T; apply E3) and the reverse fields of a non-default grade (the E20 ferrite probe scales the stored NdFeB fields by 0.37/1.29 as a placeholder)"
  - "RESOLVED (Addendum A-2 decision A2-6): the library's 6061 engine modulus is E18's 68.9 GPa (Kaiser's 68.3 stays the sourced reference), so a 6061 back iron equals the default no-back-iron design cell for cell"
  - "RESOLVED (Addendum A-2): the harmonic set and peak search (Tasks 1-2), per-ring alpha(Br) and density for a grade-mode ring (decision A2-7), inverse sizing (sizing.rs), the housing autofit suggestion (materials.cup_wall_suggested_mm) and the space claim (housing.rs); with the length override set, the hub length, cup cavity depth and retainer span follow the rings they bound (decision A2-8, option B, housing::axial_housing), so a length-sized design shows its overshoot (the default design sized to 9.9 N m: 34.72 mm over the length, 36.72 mm over the bay)"
  - "RESOLVED (Addendum A-2): tests/assumptions.rs documents the C45 override (no effect with E20 and coercivity source 1; moves results with source 0); C45 keeps its NdFeB slider and a positive beta is typed (decision A2-5: widening would regenerate the differential data)"
  - "M3: fields3d.run gives 1.034632e-5; times 4 (E5) and formatted .4g that is 4.139e-5, not the M2 default 4.14e-5 (C125 differs by about 0.00009 W). M3 should pin the M2 default or expect the difference"
  - "Tidy after M2: magcoupling-rs/src/engine/temperature.rs is about 1,350 lines, past the architecture's ~800-line split point; split mechanically (inputs, results, compute, tests) when no correction work is in flight"
  - "Open, not in Addendum A-2's scope (the A3 panel lists no bondline assumption): E4 as approved adds the inner bondline input to the pole-sweep keyed-wall term, while the block-fit term keeps the workbook literal + 0.05 (the same bondline), so at bond_inner != 0.05 the two terms of the max() disagree (sweeps.rs pole_sweep); E4's test and golden file probe only 0.05. Owner: plan A-3 or a deviation proposal (changing E4 is a registered deviation and needs the user's approval); trigger: making the pole-sweep block-fit bondline an input, or any change to E4"
  - "RESOLVED (Addendum A-2, audit M9): the engine flags f_end <= 0 (model.end_effect_check, calibration.end_effect_check, model::end_effect_in_range for sweep rows); M4 greys the numbers computed from the pull-out; inverse sizing never counts such a value"
  - "M4 design questions (decision 28): the M41 cap thread sits below the 41.33 mm cup body OD; the boss OD is 22 mm on Metal design and 25 mm on Shaft clamps. M4 draws the axial housing in effect (housing.hub_length_mm, cup_depth_mm, retainer_span_mm: A-2 decision A2-8)"
  - "M4: the sizing mode, target torque and free variable are GUI state (A-2 decision A2-9); sizing::solve takes them as arguments. Decide whether design files and share links record them"
  - "Plan A-3: the spec's dependency-graph traceability test for assumptions (each assumption changes every dependent result and no independent one) needs the equation registry; its records must include the Rust-only tau7_Pa-tau11_Pa terms of the pull-out and the Calibration sums"
  - "M3: a grade-mode ring's reverse fields are scaled with its own alpha(Br) (decision A2-7); the stored 3D fields do not separate the opposing ring's share, which live fields per ring would"
  - "M4: the workbook-parity feature is guarded by convention only (any downstream Cargo.toml can enable it). A compile_error! on all(feature = app, feature = workbook-parity) was NOT added: the self dev-dependency enables workbook-parity for every cargo test, so cargo test --features app would stop compiling. Decide the guard in the M4 plan (e.g. a CI check that the shipped build's feature set excludes it)"
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/README.md`, replace:

```markdown
as recorded). Addendum A-1 (data and physics) complete: the A6 grade table and
the parts' vendor data, any grade with manual dimensions, the A5 materials
library with per-part selectors, physics links and six warnings, and E15 to E20
applied (Addendum A decisions, approved 2026-09-30). Next: Addendum A-2
(parameters and sizing), A-3 (explanations), then M4.

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
```

with:

```markdown
as recorded). Addendum A-1 (data and physics) complete: the A6 grade table and
the parts' vendor data, any grade with manual dimensions, the A5 materials
library with per-part selectors, physics links and six warnings, and E15 to E20
applied (Addendum A decisions, approved 2026-09-30). Addendum A-2 (parameters
and sizing) complete: the harmonic set up to 11 with one general E7 peak search,
the assumptions registry (A3), the end-effect validity flag, the axial length
override with the axial housing that follows it (hub length, cup cavity depth,
retainer span), inverse sizing (A1), the housing autofit suggestion and the space
claim, one aluminium modulus, and each grade-mode ring's own alpha and density
(decisions A2-1 to A2-9). Next: Addendum A-3 (explanations), then M4.

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
```

In `C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/src/lib.rs`, replace:

```rust
//! correction (the M1 math audit's E1 to E14, the Addendum A verification's E15
//! to E20) is registered in [`engine::deviations::REGISTRY`]. The Addendum A
//! inputs the Python engine does not have (part materials, grades, the
//! coercivity source, the E17 free-space fields) are Rust-only and default to
//! the ported behaviour.
//!
//! Spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`.
//!
```

with:

```rust
//! correction (the M1 math audit's E1 to E14, the Addendum A verification's E15
//! to E20) is registered in [`engine::deviations::REGISTRY`]. The Addendum A
//! inputs the Python engine does not have (part materials, grades, the
//! coercivity source, the E17 free-space fields, the harmonic set, the axial
//! length override) are Rust-only and default to the ported behaviour. Beyond
//! the forward calculation: [`engine::sizing::solve`] (inverse sizing, Addendum
//! A1) and [`engine::assumptions`] (the A3 assumptions panel).
//!
//! Spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`.
//!
```

- [ ] **Step 2: The update tracker**

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import datetime, pathlib
p = pathlib.Path(r"C:/Users/Cole/source/repos/lsim-mag-a2/docs/ai/05-update-tracker.md")
s = p.read_text(encoding="utf-8")
anchor = "Reverse chronological (newest at top).\n\n---\n\n"
assert s.count(anchor) == 1
entry = f"""## {datetime.date.today():%Y-%m-%d} — Magcoupling Addendum A-2: parameters and sizing (A3, A1)
- `magcoupling-rs`: one general E7 peak search for any odd harmonic set (decision 29: every root of
  dT/dx in cos^2 x by recursion on derivatives) and the harmonic set as the Rust-only assumption
  `coupling.max_harmonic` (1 to 11, default 5), summed by the Calculator, both circuit sums, the
  sweeps and the Calibration prototype (Rust-only tau7_Pa to tau11_Pa).
- The assumptions registry (`assumptions.rs`: the spec's 14 rows over 15 inputs, rationale and
  source, `modified`, `reset_to_workbook_defaults`); the audit M9 flag `end_effect_check`; the axial
  length override `coupling.magnets.axial_length_mm`; inverse sizing (`sizing.rs`: axial length,
  magnets per ring, ring radius; a 64-cell scan refined at the first crossing, each peak and each
  validity edge; counts a value only if its blocks fit (flats, or arcs that do not overlap), the
  keyway leaves hub wall and f_end > 0); the autofit wall suggestion
  `materials.cup_wall_suggested_mm` and the space claim (`housing.rs`, `DesignResults::housing`,
  overshoot per axis, at least 0.01 mm), with the hub length, cup cavity depth and retainer span
  following the length override (decision A2-8, option B: `housing::axial_housing`).
- Carry-overs: the library's 6061 takes E18's 68.9 GPa (A2-6); C45 keeps its NdFeB slider, a positive
  beta is typed (A2-5); a grade-mode ring takes its grade's alpha(Br) and density (A2-7).
- Parity (1,149 checks), the differential data and every registry probe unchanged; the default
  headline with every correction on is unchanged (2.688 N·m, 93.06 °C, M4 x 14).

"""
p.write_text(s.replace(anchor, anchor + entry, 1), encoding="utf-8", newline="\n")
print("tracker entry added")
EOF
```

- [ ] **Step 3: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task11.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task11.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/target/gate-task11.log
git -C C:/Users/Cole/source/repos/lsim-mag-a2 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 4: Check the default headline one last time**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a2/magcoupling-rs/Cargo.toml --test deviations all_corrections_together_give_the_reviewed_headline 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed` (pull-out 2.688 N·m, hot low 2.285, limit 93.06 °C, clamp M4 x 14: unchanged by every task).

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a2 add magcoupling-rs/README.md magcoupling-rs/src/lib.rs docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md
git -C C:/Users/Cole/source/repos/lsim-mag-a2 commit -F - <<'EOF'
docs(magcoupling): Addendum A-2 status, invariants and open items

README status and crate docs; docs/ai system invariants (the Rust-only defaults,
the peak search, sizing's rules, per-ring alpha), structure (assumptions, sizing,
housing), memory (the A-2 carries resolved; the M4 design questions, sizing
state, the A-3 traceability test, M3 fields per ring; the E4 bondline item's
owner and trigger) and the update tracker.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a2`; `git -C C:/Users/Cole/source/repos/lsim-mag-a2 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

## Self-review record

A critique of the first complete draft found one blocking defect (B1) and six non-blocking gaps (N1 to N6). Every finding is fixed below; everything else in the plan is kept. The changed Rust was re-verified the same way as the rest (the Verification record above): every task replayed on a fresh copy of the A-1 end state with each see-it-fail and gate step, `cargo test` also in a release build, and the bit dump of every existing result equal to the pre-revision one.

| Finding | What the critique changed | Where |
|---|---|---|
| B1 (blocking): the scan tested "meets" only at the 65 grid points, so a meeting interval inside one cell (0.328 mm of radius, 0.778 mm of length) was skipped. It returned a later bracket (no back iron and a 0.5 mm gap, 2.0695 N·m: 29.47 mm where about 12.11 mm meets) or a false "not reachable" (the default design, any ring-radius target in (2.595112108, 2.595112251] N·m), and "best" was only ever a grid sample | The search (the private `Search` in `sizing.rs`) now refines between samples: the first crossing (bisection); each peak (a golden-section search where three consecutive valid samples rise and then do not rise; a peak that meets has the crossing below it bisected; every peak is offered as the best value); each validity edge, in both directions (bisection, its valid side sampled). The remaining limit, a hump whose rise and fall both lie inside one cell, is stated in the module docs and pinned by a unit test. New tests: `a_meeting_interval_inside_one_cell_is_found` (the critique's repro, asserting both grid neighbours miss), `a_target_just_below_the_true_peak_is_solved` (asserting that no grid value meets), `a_target_met_only_where_the_keyway_starts_to_leave_wall_is_found`, and 11 unit tests that drive the search with plain functions. `a_target_beyond_the_range_is_not_reachable` now also asserts that the ring radius's best is the refined peak, not a grid value. With the peak and edge refinements switched off, those three tests, that assertion and four of the unit tests fail. Reworded: Architecture, "Settled here", Review Focus 2, the module and `SizingOutcome` docs, both README rows, `docs/ai` and the tracker entry. Every later README block that quotes those rows was regenerated from the rebuilt commits, so all of them carry the new text. Measured again: the cost line (now 3 to 139 `compute_all` calls per solve on the default design) and every test count. Task 6 stays on `sonnet` because the plan again gives the exact code, the search's unit tests included | Task 6; Tasks 7 to 11 (quoted context); plan head; Verification record |
| N1: arcs always counted as fitting, so sizing could return overlapping arcs (arcs, magnets per ring, 2.3 N·m: 12 poles of 6.35 mm on a 6.14 mm pitch, which the fill's min(1, ...) prices as if they fitted) | A2-4 is extended: arcs fit when neither ring's pitch share exceeds 1. The share is the width over 2π(apothem of the ring's inner surface + thickness/2)/N (the inner back apothem, or the outer face apothem), i.e. the fill C66/C67 before its clamp. `model::pitch_share` is that quotient. The fill now calls it, with the same operations in the same order, so no bit moves. `model::blocks_fit` also calls it; it replaces `flats_fit` and takes `&CouplingInputs`. Tests: `overlapping_arcs_never_count` (not reachable, best 10 poles) and the equality edge `arcs_fit_exactly_at_a_pitch_share_of_one` | Task 6; A2-4; Review Focus 1; Global Constraints |
| N2: validity ignored the hub (16 mm bore, 2.5 mm keyway, ring radius, 0.5 N·m: 9.77 mm with the wall past the keyway at −0.78 mm) | `sizing::is_valid` also requires `model.hub_wall_past_key_mm` > 0 (Calculator C53). Tests: `a_hub_the_keyway_breaks_through_never_counts` (now 10.55 mm) and the equality edge `the_hub_wall_counts_only_while_positive`. A2-4 states the rule, and its alternative (b) quotes the repro | Task 6; A2-4; Review Focus 1; Global Constraints |
| N3: with the default free variable, no dimension the space claim reads ever moves | A2-8 now states this plainly: a length-sized design reads "Inside the space claim" at any length up to 50.8 mm. It quotes the default design sized to its own 2.5 N·m requirement (13.77 mm magnets on the 13.0 mm hub) and to 9.9 N·m (50.6 mm in the 15.5 mm cup). It also describes alternative (b) concretely: each ring against its dimension, with overshoots and a flag, not a sizing rule. (a) stays the recommendation because neither the workbook nor the report says which ring each class N dimension bounds (decision 28 A keeps them as inputs with no rule). The user decides with this in view at Task 0. `a_length_sized_design_reads_inside_the_space_claim_at_any_length` (Task 7) pins the chosen behaviour, and the M4 design-questions line in `04-memory.yaml` quotes it | A2-8; Review Focus 4; Task 7; Task 11 |
| N4: an overshoot below 0.005 mm printed "0.00 mm over", and a NaN on one axis hid the known overshoots on the others | The badge quotes `fmt_fixed(py_max(over, 0.01), 2)`, so any positive overshoot reads at least 0.01 mm. This departs from the critique's literal suggestion, `fmt_fixed(ceiling(over, 0.01), 2)`: `compat::ceiling` applies its 1e-12 guard to the quotient, so two ulps of noise at 43 mm (2.8000000000000114) would print "2.81". The floor at 0.01 meets the finding and keeps round-to-nearest for every other overshoot. A NaN axis now reads "<axis> unknown" beside the exceeded axes, and `SPACE_CLAIM_UNKNOWN` appears only when nothing is exceeded. Tests: `a_tiny_overshoot_reads_at_least_a_hundredth` (each claim 0.001 mm under its dimension) and an extended `several_axes_are_named_in_order_and_nan_is_unknown` | Task 7; Settled here; Review Focus 3 and 4 |
| N5: the test re-derived validity from the flat check alone, leaving out the end effect and the finite-torque check | One public predicate, `sizing::is_valid(design, results)`, is used by the search, by `a_target_beyond_the_range_is_not_reachable` and by the equality-edge tests. N1 and N2 land in it | Task 6 |
| N6: the A3 carry-over (E4's pole-sweep bondline: the block-fit term keeps the literal 0.05 mm) stayed in memory verbatim, with no owner | The plan's out-of-scope sentence (Spec) now names it. Task 11 relabels the `04-memory.yaml` line "Open, not in Addendum A-2's scope" and gives its owner (plan A-3 or a deviation proposal; changing E4 needs the user's approval) and its trigger | Plan head; Task 11 |

## Amendment: the user's decisions of 2026-09-30

The user confirmed the recommended option on A2-1 to A2-7 and A2-9 and chose option B on A2-8: the axial housing grows with the magnets (the user's option B is the growth rule; the draft's (b), an overshoot check per ring, is superseded). Task 7b implements it on `sonnet`, because the plan gives the exact code and the design choices are settled in A2-8's row and Task 7b's intro: with the length override set, the hub length (C123) follows the inner ring, the cup cavity depth (C124) the outer ring and the retainer span (C172) the longer ring, each by the ring's change from its own length and never shorter than the ring; blank, nothing changes. Task 7's `a_length_sized_design_reads_inside_the_space_claim_at_any_length` (N3 above) stays in Task 7, which it describes, and Task 7b replaces it with nine tests; Task 5's override test now expects the stack and the span to grow. Also changed: the plan head (Decisions: confirmed; A2-8's row; Architecture; Review Focus 4 and a new 5; File Structure; Order; Global Constraints: model tiers and equality edges), Task 0's decision step (the answers are recorded, not asked), the intros of Tasks 5 and 7, Task 8's counts, Task 11's docs (the M4 design questions keep the M41 cap thread and the boss OD; the housing containment question is resolved) and its tracker entry. Tasks 8 to 11 were rebuilt on Task 7b and their edit blocks regenerated (their quoted README context now carries Task 7b's row). Re-verified as the rest: the whole plan replayed on a fresh copy of the A-1 end state with every see-it-fail and gate step, and the bit dump of every existing result with the override blank equals the pre-amendment dump in both builds.

An adversarial check of Task 7b then mutated its rule and found two choices no test pinned: a retainer span that followed the outer ring instead of the longer one, and a cup cavity that followed the longer ring instead of the outer one, both passed all 31 tests (the only ring pair with different lengths had the outer ring the longer, where each pair of rules agrees). So did `f64::max` in place of `py_max` in the floor, which the intro named as a choice (and which it justified by a NaN override; the choice matters for a NaN input). `each_dimension_follows_the_ring_it_bounds` now also takes the inner ring the longer (12 and 10 mm, set to 20 mm, no floor binding: hub 21.0, cavity 25.5, span 22.5 mm), and `a_housing_dimension_never_ends_shorter_than_its_ring` now also asserts that a NaN hub, cup depth or span stays NaN with the override set; each of the three mutations now fails. Tests only: no engine line, test count or result bit changes. Its independent bit comparison (the Task 7 and Task 7b trees, 49 designs in both modes: the defaults, the workbook defaults, manual rings either way round, a hub, cup and span typed shorter or longer than their rings, no back iron, 6 poles, arcs, grade-mode rings and 40 differential input sets) found no bit moved with the override blank; with it set, only the masses, the retainers' span and mass, the stacks, reserves and hybrid values, the thermal results that read the heat capacity and the space claim move, and at the rings' own length nothing moves.
