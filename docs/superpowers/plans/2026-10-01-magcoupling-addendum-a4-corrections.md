# Magcoupling Addendum A-4: Calibration Clamp, Cure Margin, Per-Grade Magnet Thermal Properties Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the four corrections the user approved on 2026-10-01 with the Addendum A-4 verification (all ten decisions option A): E21 (each ring's rating-calibration offset clamped at 0), E24 (the cure margin from the ring with the lower single-ring onset), E22 (the inner ring's grade CTE in the bond screen and warning rule 6, the screens comparing the size of the shear) and E23 (each ring's grade specific heat in the heat capacity), registered with cells, probes and an approval record of their own, the explain records and teaching notes kept true, and everything workbook-exact with the corrections off.

**Architecture:** Each correction is a branch at its formula behind `dev.is_on(DeviationId::Ek)`, the workbook form kept beside it, and one registry entry with its report's cells, probes and approval (`Approval::AddendumA4`, section 5 of the A-4 report). E21 and E24 refine E20 (`depends_on`), so they are inert without it; E22 and E23 read two cited fields new on the `Grade` record (`bond_plane_cte_per_C`, `specific_heat_J_kgK`), `None` for sintered NdFeB, so neither grade value moves a sintered NdFeB design (E22's size comparison can, at one slider corner: decision A4-4). Every value a corrected formula needs that the engine computed but did not expose becomes a Rust-only result (A-3's standing rule): each ring's applied offset and single-ring onset, the magnet CTE in effect, each ring's magnet mass and specific heat in effect; the equation records read them as terms, so the drift guard still proves every displayed formula against the engine with every correction on. The C95 and C138 inputs keep their labels and ranges (decision 10); their help, and that of C24, C50 and C101, is reworded and registered as workbook help.

**Tech Stack:** Rust 2024 edition, std only for the library (wasm32-unknown-unknown clean); `serde_json` (feature `float_roundtrip`) as the only dev-dependency; the vendored Python 3.12 oracle `reference/magcoupling-py/` run with `C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe` (only `--check`: no differential data changes); Git Bash for every command; `python` (3.12, on the PATH) for the in-place edit scripts.

**Spec:** the approved report `C:/Users/Cole/source/repos/lsim-mag-a4/docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md` (section 5: "**Approved (user, 2026-10-01): option A on all 10 decisions.**"; section 2.10's record corrections for E21; section 2.8's E24; sections 3.2, 4.1 and 4.7's E22 and E23 values, cells, probes and landing notes) with its data file `docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json`; the earlier report `docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (decision 19, E20, and its section 6.3 row B842-N52, which E21 supersedes). The calculator spec `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (Addendum A2 to A4) governs the explain layer. Save this plan in the repository as `docs/superpowers/plans/2026-10-01-magcoupling-addendum-a4-corrections.md` (the README and the sign-off record name that path).

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-a4`, branch `magcoupling/addendum-a4`, created by Task 0 from `main`, LF line endings with a worktree-scoped `core.autocrlf=false`. Every command uses absolute paths into it. Nothing is pushed.

## Decisions to confirm

These questions arose while turning the approved report into code; no approved decision settles them. This plan implements the recommended option of each (Task 0 records the user's answers; if the user picks another option, the task that implements it stops and escalates instead of improvising).

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| A4-1 | The report states E24's formula and examples but no registry cells or probe. | **Cells C24 and C25** (the verdict reads C24. At the default sliders no verdict moves for any ordered pair of grades or of library parts with any of the four adhesives, on both hubs and both bases (`e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict`). At a hot cure one does: Recoma 30 beside N42H, in either order, with EA 9514 (120 °C), the knee fraction C46 at 0.95 and the design margin C51 at 0 reads OK with C24 = 75.40 °C and CHECK with E24's −6.719 °C, because N42H's single-ring onset, 113.3 °C, is below the cure; only C24 and C25 move, on both bases (`e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure`). A two-slider search over every grade pair and part pair, C46 from 0.50 to 1.00 in steps of 0.01 and C51 from 0 to 10 °C in steps of 0.5 °C, with EA 9514 and 2214 Hi-Temp (the two hot cures), on the user basis, found flips only for this pair, at C46 0.94 to 0.97 and C51 at most 7 °C; a coarser search over all four adhesives and both bases found no other), and **one probe on E24's registry basis `NONE.with(E20)`: Recoma 30 inside, N42H outside**, the report's largest pre-existing overstatement (C24 169.87720052918982 -> 90.82967381692148 C, 79.05 C, section 2.8); the E21-created case (probe 4, N38UH beside Recoma 26: 142.57635668906815 C) is pinned by a test on `NONE.with(E20).with(E21).with(E24)`, since a registry probe runs only on the corrections the entry refines | (b) cells C24 only (then the flip above would change an unregistered cell); (c) make E24 depend on E21 too and probe N38UH beside Recoma 26 (contradicts the approved "E24 gated on E20") | 5 |
| A4-2 | Which teaching notes a correction makes stale. | **Two go back to Draft and are re-reviewed in Task 6**: `demagnetization` (E21 makes its sentence "the onsets are shifted so the reference magnet reaches its knee exactly at its rated temperature" false: a rating above the model's own onset no longer raises them) and `a5.cte_mismatch_with_magnets` (E22 changes what "the magnets'" expansion is in warning rule 6, and the note's only magnet value is sintered NdFeB's; a SmCo ring now shears the glue the other way, decision 9). `thermal_time_constant` stays reviewed: C = Σ m c still holds with each ring at its own c (E23) | (b) only `demagnetization` (the CTE note stays true for NdFeB but silent on ferrite and SmCo, which E22 now distinguishes) | 1, 3, 6 |
| A4-4 | Decision 9 (A) compares the size of the shear "under E22", and the report's premise is that only a grade makes the shear negative (IN1: "Without E22, s1 is positive at every slider setting"; decision 9: "E22 makes a negative mismatch reachable for the first time"). That premise is false at one slider corner: the default back iron's CTE input C17 (slider 5e-6 to 25e-6 /°C) set below C95 (slider up to 6e-6 /°C; every library hub material is at 9.0e-6 or more). With B842SH on both rings, C17 = 5e-6, C95 = 6e-6, C96 = 3.0 GPa and both bondlines at 0.01 mm, C104 is −22.6 MPa on the registry basis (−22.8 on the user basis), and turning E22 on turns C106 and C202 from "Below" to "Above" on both bases: a sintered NdFeB design moves. | **Keep the size comparison gated on E22 alone** (decision 9 as written: it fixes a misreading wherever the shear is negative, and −22.6 MPa is past the 15 MPa lap shear whichever way it acts). Every claim that no sintered NdFeB design moves is stated for the default sliders, and `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too` pins the corner (only C106 and C202 move, on both bases) | (b) gate the size comparison on a grade CTE being in effect (every sintered NdFeB design stays bit for bit, but users at that corner keep the misreading) | 3, 7 |
| A4-3 | `main` moved after the plan's base: 23ed7bd (the A-4 approval) was followed by acc0296, an independent physics review of the A4 notes (it rewords six notes and their review records, the README's "17 of the 17" sentence, warning rule 2's text and the tracker). | **Branch from `main` (acc0296) as Task 0 says.** The blocks this plan aims at those places are written to apply to both commits: a note goes to Draft by a script anchored on its id (whatever its review record says), and the README sentence edits anchor on the prefix both commits share. The plan was replayed on both bases | (b) branch from 23ed7bd exactly and merge `main` at the end (a conflict in `notes.rs`'s demagnetization review block and in the README sentence) | 0, 1, 3, 6 |

Settled by the report or this plan's design without a new decision (each is stated where it is implemented): E21's record carries the report's 33 cells (the provisional 31 plus C43 and C47) and its four probes (probe 4 included), and its approval cites decisions 1 and 3 (C101's help documents the C12 swing); the approval record is a new `Approval::AddendumA4 { decisions }` variant (the report's section 2.10 offered a variant or a report section; a variant keeps `every_entry_is_approved_in_its_report` checking the decisions by number); the clamp is `py_max(offset, 0.0)`, which is exactly the report's `offset < 0.0 => 0` (a NaN stays NaN, every offset of 0 or more and -0.0 bit for bit) and exactly the markup's `max(·, 0)`, and E24's minimum is `py_min`, the markup's `min` (the inner ring on a tie, so identical rings stay bit for bit); each ring's offset becomes a Rust-only result so the clamp is stated once per ring in the explorer and both the ring's limit and its single-ring onset read it (DRY); E23's per-ring masses are Rust-only mass results computed beside C110 (one source of N V ρ), and rings of one specific heat keep the workbook's single product C110 × c bit for bit (decision A2-7's pattern); the M2 basis helper `m2()` in `tests/deviations.rs` switches E21 to E24 off with the other Addendum corrections; C24's help is reworded under E24 (its C59 is the governing ring's, its margin the lower ring's).

## Global Constraints

Every task's requirements implicitly include this section.

- Crate `magcoupling-rs/` stays a sibling of `linkage-sim-rs/`: "No Cargo workspace conversion." The library is pure std with no `[dependencies]` outside the `gui` and `app` features; `cargo check --target wasm32-unknown-unknown --lib` stays clean. The `workbook-parity` feature is reachable only through the self dev-dependency.
- "One pure entry point `compute_all(&DesignInputs) -> Results`: no I/O, no global state." Its signature does not change. No input is added; `tests/data/input_schema.json` changes only in the `help` of C95 (Task 3) and C138 (Task 4), each rewritten by `MAGCOUPLING_BLESS=1 cargo test --test schema` and committed.
- Addendum A: "At default inputs and default assumptions every result stays workbook-exact." With `Deviations::NONE`: parity (1,149 checks), every differential file and `gen_differential.py --check` unchanged after every task. With every correction on, the default design (B842SH on both rings) is bit for bit unchanged on every result row, Rust-only rows included; so is, at the default sliders, every sintered NdFeB grade and library part with an offset of 0 or more (each correction's tests assert it; E22's size comparison departs from this only at decision A4-4's slider corner), and `all_corrections_together_give_the_reviewed_headline` passes unedited.
- The registry, exactly as the report states: E21 depends on E20 (33 cells, four probes, decisions 1 and 3); E22 (cells C104, C105, C106, C201, C202; probes E22-1, E22-2; decisions 4 to 7, 9, 10); E23 (the report's 27 registry-basis cells; probes E23-1, E23-2; decisions 4, 5, 8, 10); E24 depends on E20 (decision A4-1; decision 2). Probe values are the report's full-precision `f64`, compared by the parity rule.
- Values are written as typed, never converted at load time (A-4 data file, `engine_literal_rule`): `10.0e-6`, `14.0e-6`, `13.0e-6`, `700.0`, `370.0`, `350.0`, `420.0`; bonded NdFeB has no CTE (decision 7).
- Spec A2: "The equation shown is provably the one that produced the number." After every task `tests/explain.rs` passes unedited: the drift guard (every record reproduces the engine with every correction on, every `cases` arm taken, every value term at two values), the registry and drill-down tests and the A3 traceability tests. A new Rust-only in-effect field becomes a term wherever an explained record shows it.
- A teaching note whose text a correction makes stale goes back to `Review::Draft` in that correction's task and is signed off again only by Task 6's physics review (decision A4-2).
- `reference/magcoupling-py/`, the A-4 report, its data file and the Addendum A report are evidence: never edited. GUI code is not touched (`src/gui/`, `src/app.rs`, `src/bin/`): the M4-1 GUI is being built on another branch.
- Every gate run: `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate` line (12 gates); then `git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml` before every commit, and `cargo fmt --check` clean. Every block below is already in rustfmt's layout (it was produced by formatting); the record, table and symbol arrays carry `#[rustfmt::skip]`.
- The replace blocks quote the tree with LF line endings (Task 0 checks them). If a block's text is not found, stop and escalate; do not improvise a match. The README and `docs/ai/*` blocks quote whole rows or paragraphs and are the likeliest to drift if `main` moves again (decision A4-3).
- Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that wrote the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The commit blocks below write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the session model); a `sonnet` task replaces `Claude Opus 5.5` with its own model's name. Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`): each task updates `magcoupling-rs/README.md` where it changes the Differences table, the layout, the tests or the explorer; Task 7 updates `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml` and `05-update-tracker.md`; a YAML scalar containing `: ` is quoted.
- Model tiers (CLAUDE.md section 5), each task's **Model:** line: Tasks 1, 3, 4 and 5 (the corrections and their explain-record updates) and Task 6 (the physics review and the notes) run on the session model (omit `model` on purpose, comment `// session model: physics`); Tasks 0, 2 and 7 give the exact code or text and run on `sonnet` (set `model: 'sonnet'`). A `sonnet` attempt that ends `blocked`, leaves the gate red or is rejected in review escalates every retry to the session model.
- Commit subjects: `feat(magcoupling-rs): ...` for Tasks 1 to 5, `docs(magcoupling-rs): ...` for Task 6 and `docs(magcoupling): ...` for Task 7.

## Review Focus

Six input classes or failure modes the report implies but no single correction's red test is built around; each names the tests that pin it and its task.

1. **Rings whose clamp reorders the governing ring** (a ring with a negative offset beside one with a positive offset, in either order; an exact tie after the clamp). Expected: the block shows the ring with the lower clamped limit, C42, C43 and C47 follow it, an exact tie resolves to the inner ring, and with E24 the cure margin is the lower ring's whichever ring governs. Tests: E21's probes 3 and 4 (Task 1), `e21_reorders_the_rings_where_the_clamp_moves_a_limit_and_ties_stay_inner` (N52 beside N50, the report's one exact tie; Task 1), `e21_moves_only_pairs_with_a_negative_offset_and_never_raises_a_limit` (every ordered pair of different grades and of different parts: bit for bit unless a ring's offset is negative, C58, C60, C12 and C15 never rise, and the report's pair table, 214 pairs moved, 71 flips and 0 verdicts on the registry basis and 150, 39 and 0 on the user basis, 156 from Task 5 on; Task 1), `e24_with_e21_gives_the_lower_ring_s_margin_on_probe_4` and `e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict` (every ordered pair of grades and of parts, all four adhesives, both hubs, both bases; Task 5).
2. **The knee-fraction assumption slider below 0.7835 on the default part.** Expected: the default part's offset turns negative and E21 binds (first at 0.78), and at 0.70 the clamp turns the default design's verdict from OK to CHECK. Test: `e21_binds_on_the_default_part_only_below_the_knee_threshold` (Task 1).
3. **A magnet that expands more than its hub** (any SmCo grade on steel; a reachable back-iron pick: Recoma 20 on 416 stainless; without a grade, C95 set above the default back iron's C17, decision A4-4). Expected: the shear is negative and stays on display signed, and C106 and C202 compare its size, so a large negative shear reads "Above" (decision 9), sintered NdFeB included at the slider corner. Tests: `e22_compares_the_size_of_a_negative_shear` (registry basis, -19.1 MPa against the 15 MPa lap shear), `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too` (B842SH, C17 5e-6, C95 6e-6, C96 3.0 GPa, 0.01 mm bondlines: -22.6 MPa, only C106 and C202 move) and the unit test `e22_reads_the_inner_ring_s_grade_cte_and_compares_the_shear_s_size` (Task 3).
4. **A ferrite, SmCo or bonded ring on one side only, and bonded NdFeB anywhere.** Expected: E22 reads only the inner ring (an outer-only grade is inert; bonded has no CTE), E23 reads both rings and moves C141 by exactly that ring's mass times its specific heat's difference from C138, and two different rings by the sum of both rings' terms; at the default sliders nothing moves for sintered NdFeB or any library part. Tests: `e22_moves_only_a_ferrite_or_smco_inner_ring` and `e23_moves_only_a_ring_with_a_grade_specific_heat` (every grade in three placements, every part, both hubs, both bases, with the wiring to C95 and C138; Tasks 3 and 4) and `e23_moves_no_verdict_and_the_peak_by_hundredths_of_a_degree` (C141 for every ordered pair of grades, Y30 beside Recoma 26 included; Task 4).
5. **Values that bypass validation, and the differential corpus.** Expected: a NaN offset (a NaN rating) stays NaN through the clamp; E21 and E24 alone are NONE on every corpus case (the report's double gate, section 2.6); with decision 9's size comparison no differential case moves under E22 or E23 (the report checked neutrality before decision 9 existed). Tests: the unit test `e21_clamps_a_negative_offset_on_each_ring_and_keeps_the_rest` and `e21_alone_moves_no_differential_case` (Task 1), `e22_moves_no_differential_case` and `e23_moves_no_differential_case` (every corpus case, NONE against only the correction and every correction but it against all, every result row bit for bit; Tasks 3 and 4), `e24_alone_moves_no_differential_case` (Task 5); `records_never_panic_on_inputs_that_bypass_set` (unchanged) covers the new records.
6. **A hot cure beside a ring whose single-ring onset is below it.** Expected: at the default sliders E24 turns no verdict; with EA 9514's 120 °C cure, C46 at 0.95 and C51 at 0, Recoma 30 beside N42H (either order) turns from OK to CHECK, and only C24 and C25 move (decision A4-1). Test: `e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure` (Task 5).

## File Structure

Modified (under `C:/Users/Cole/source/repos/lsim-mag-a4/`; no file is created):

| File | Responsibility | Tasks |
|---|---|---|
| `magcoupling-rs/src/engine/deviations.rs` | `A4_REPORT`, `Approval::AddendumA4`, `DeviationId::E21` to `E24`, the four registry entries, `decision_list` (the evidence text), its unit tests | 1, 3, 4, 5 |
| `magcoupling-rs/src/engine/temperature.rs` | E21's clamp in `ring_demag`; E24's cure margin; `bond_plane_cte_per_C` and the size comparison (E22); `magnet_specific_heat_J_kgK` and the per-ring magnet heat term (E23), both through one private `grade_value_or`; the Rust-only results; C24, C50, C95, C101, C138 help; `TemperatureLinks` per-ring masses; unit tests | 1, 3, 4, 5 |
| `magcoupling-rs/src/engine/grades.rs` | `Grade::bond_plane_cte_per_C`, `specific_heat_J_kgK`, `GradeSources::bond_plane_cte`, `specific_heat`, two source constants | 2 |
| `magcoupling-rs/src/engine/model.rs` | `MassResults::magnets_inner_g`, `magnets_outer_g` (Rust-only) | 4 |
| `magcoupling-rs/src/engine/api.rs` | warning rule 6 reads the CTE in effect; the per-ring masses into the links | 3, 4 |
| `magcoupling-rs/src/engine/warnings.rs` | rule 6's help and input doc | 3 |
| `magcoupling-rs/src/engine/explain/tables.rs` | the grade table's `specific_heat_J_kgK` field | 2 |
| `magcoupling-rs/src/engine/explain/records/demagnetization.rs` | each ring's offset and single-ring onset (new), the per-ring limits and C50 (rewritten) | 1, 5 |
| `magcoupling-rs/src/engine/explain/records/slip_heating.rs` | each ring's magnet mass and specific heat (new), C141 and C24 (rewritten) | 4, 5 |
| `magcoupling-rs/src/engine/explain/notes.rs` | the `demagnetization` and `a5.cte_mismatch_with_magnets` notes: redrafted, then signed off | 1, 3, 6 |
| `magcoupling-rs/tests/deviations.rs` | the A-4 approval check, the helpers (Task 1: `a4_bases`, `grade_rings`, `part_rings`, `grade_pairs`, `part_pairs`, `results_with`, `same_value`, `assert_same_results`, `moved_cells`, `assert_a4_defaults_unchanged`, `assert_no_differential_case_moves_between`; Task 3: `grade_placements`, `HUBS`, `assert_no_differential_case_moves`) and each correction's tests | 1, 3, 4, 5 |
| `magcoupling-rs/tests/grades.rs`, `tests/common/mod.rs` | the thermal fields against the A-4 data file (`A4_DATA`) | 2 |
| `magcoupling-rs/tests/data/input_schema.json` | blessed: the help of C95 and C138 | 3, 4 |
| `magcoupling-rs/README.md` | Differences rows E21 to E24, layout, tests, the explorer and status | 1 to 7 |
| `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` | project state | 7 |

Order: E21 first (the approval variant every later entry uses), then the grade data alone (Task 2: nothing reads it, so it is a mechanical, separately reviewable step), then E22, E23 and E24 in id order (the registry is indexed by id), the physics review and the notes' sign-off (Task 6), and the docs (Task 7). E24 lands on the same branch as E21 (decision 2: "in the same change"); between Tasks 1 and 5 the branch shows E21's C24 overstatement, which E24 removes before anything merges.

## Verification record

This plan was replayed before it was handed over, on two bases: a fresh LF export of `main` at 23ed7bd (the base the task named) and one of `main` at acc0296 (decision A4-3). This revision, which fixes the critic's findings (Self-review record, at the end), was rebuilt from its blocks and replayed again on both bases. In each replay, the blocks of each task were applied in order with no formatting run between them, so `cargo fmt --check` proves every block is in rustfmt's layout. Each base had its own cargo target directory, and every command marked with an expected output was run and compared. The two replays print the same thing from Task 1 on, line for line, timings aside.

The replays reused the target directories of a first, fresh-target replay: git archive gives the files the commit's old timestamps, so cargo kept the earlier build for the untouched base. Task 0's base counts and its release-gate expectation therefore come from that first replay. Every later task's figures come from fresh builds, because each task edits `src/`.

- **Red steps** failed as stated: compile errors naming the missing items for Tasks 1 to 5 (the tallies in each Step 2 are this revision's), and the release gate naming the two drafts before Task 6.
- **After every task** these passed:
  - `cargo test` (the drift guard and the A3 traceability tests in `tests/explain.rs` included) and `cargo test --features app`;
  - `cargo clippy --all-targets -- -D warnings` and `cargo clippy --all-targets --features app -- -D warnings`;
  - `cargo check --target wasm32-unknown-unknown --lib` and `cargo fmt --check`;
  - `gen_differential.py --check` with the oracle Python.
- **Probes.** Every registry probe reproduced the report's full-precision values by the parity rule (`each_probe_shows_its_correction`), and changed only registered cells (`addendum_entries_name_every_cell_their_probes_change`; E21's 33-cell list holds on both bases).
- **The pair sweep** reproduced the report's E21 table after Tasks 1 to 4 (214, 71, 0 on the registry basis; 150, 39, 0 on the user basis) and gave 156 user pairs from Task 5 on, as Task 5 states.
- **Task 6.** The sign-off script ran as if both notes were approved, and the ignored release gate `release_notes_are_reviewed` then passed. The review sheet has 15 matching lines.

| After task | Unit tests | `tests/deviations.rs` | `tests/grades.rs` | `tests/explain.rs` | Equations |
|---|---|---|---|---|---|
| 0 (base) | 172 | 54 | 11 | 12 passed, 2 ignored | 392 |
| 1 | 173 | 64 | 11 | 12 passed, 2 ignored | 394 |
| 2 | 174 | 64 | 12 | 12 passed, 2 ignored | 394 |
| 3 | 175 | 70 | 12 | 12 passed, 2 ignored | 394 |
| 4 | 176 | 76 | 12 | 12 passed, 2 ignored | 398 |
| 5 to 7 | 177 | 82 | 12 | 12 passed, 2 ignored | 400 |

Every other binary keeps Task 0's counts throughout: assumptions 8, differential 19, material_library 4, material_links 11, parity 4, python_schema 7, robustness 12, schema 7, sizing 31, static_data 8, and the doc-tests 1 passed, 1 ignored. No workbook cell, differential value or earlier registry probe moves.

The equation count is the review sheet's row count (`cargo test --test explain review_sheet -- --ignored --nocapture`). Eight records are new: each ring's calibration offset (Task 1), magnet mass and specific heat (Task 4), and single-ring onset (Task 5). Five are rewritten: the per-ring magnet limits and C50 (Task 1), C141 (Task 4) and C24 (Task 5).

Found by the first replay, not in the report: on E22's registry basis (the workbook's 0.55 GPa adhesive), a Recoma 20 ring on a 416 stainless back iron (10.08e-6 /°C, the lowest back-iron CTE) shears the bond at -19.1 MPa (C104; C201 -3.40 MPa). That is past the 15 MPa lap shear, so decision 9's size comparison reads "Above" where a signed one would read "Below" (pinned by `e22_compares_the_size_of_a_negative_shear`). On the user basis (E1's 0.107 GPa) the same design reads -3.92 MPa, below the lap shear.

Found in review, not in the report: two cases, each pinned by a test.
- **Decision A4-4.** With C95 set above the default back iron's C17, a slider corner, a sintered NdFeB design shears the bond at -22.6 MPa, and the size comparison reads it "Above".
- **Decision A4-1.** With EA 9514's cure, C46 at 0.95 and C51 at 0, E24 turns Recoma 30 beside N42H from OK to CHECK.

The full `gate.sh` (all 12 gates, the linkage crate's included) ran on the final acc0296 tree of this revision: `GATE PASS`, with no `SKIP gate` line. Its last three lines were `1159 passed in 2.21s`, `differential data is current (...)` and `GATE PASS`.

`main` moved during this revision. Another session merged the M4-1 GUI as 42ab370, and at the time of checking, that commit still had conflict markers in `docs/ai/02-system.yaml`. An apply-only check on a 42ab370 export found that:
- every block of Tasks 1 to 6 applies there;
- with them applied (and the input schema blessed), `cargo test --features app` passes: 314 lib tests (the GUI's corrections index, which reads the registry, included) and 82 in `tests/deviations.rs`. `cargo clippy --all-targets --features app -- -D warnings` and `cargo fmt --check` are clean;
- only Task 7's first block, the README status paragraph, no longer matches, because M4-1 rewrote the lines after it.

Task 0 Step 1's rule covers this case: the first block that does not match stops the run, and the controller escalates it. Re-anchor that one block once `main` settles.

Not exercised by the replay: the physics review itself (Task 6 dispatches it and acts on its findings) and the user's answers to the Decisions to confirm.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` for Steps 1 to 6 (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 7 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: `main` at acc0296 (`merge: independent physics review of the A4 teaching notes ...`), or at 23ed7bd (`docs: record approval of the 10 A-4 decisions (E21, E24, E22, E23)`) if that merge is undone; both carry the approved report and data file.
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-a4` on the new branch `magcoupling/addendum-a4` with LF line endings, a confirmed green baseline, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

This machine's system gitconfig sets `core.autocrlf=true`, which would check the files out with CRLF; every replace block below quotes them with LF, as git stores them. So the worktree is created with LF and keeps a worktree-scoped `core.autocrlf=false`. A worktree-scoped setting needs `extensions.worktreeConfig true` in the main repository's own config (`C:/Users/Cole/source/repos/linkage_simulation/.git/config`): a permanent change to that repository, which stays after this worktree is removed and lets every worktree carry settings of its own (it changes nothing else). It is already `true` on this machine (an earlier plan set it), so the command below changes nothing here; it is kept so the step also works on a clone without it.

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 main
git -C C:/Users/Cole/source/repos/linkage_simulation diff --stat acc0296 main -- magcoupling-rs docs/ai
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/addendum-a4 C:/Users/Cole/source/repos/lsim-mag-a4 main
git -C C:/Users/Cole/source/repos/lsim-mag-a4 config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-a4 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-a4 status --short
```

Expected: the `log` line is `acc0296 merge: independent physics review of the A4 teaching notes (no statement wrong or misleading; 10 precisions applied)`; the `diff --stat` prints nothing. If `main` has moved past acc0296 and the `diff --stat` lists files, continue: the blocks are checked one by one, and the first that does not match is a stop-and-escalate (Global Constraints). Then `worktree add` prints `Preparing worktree (new branch 'magcoupling/addendum-a4')`; then `magcoupling/addendum-a4`; no status lines.

- [ ] **Step 2: Check the line endings**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 config --get core.autocrlf; head -c 2000 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs | tr -cd '\r' | wc -c
```

Expected: `false` and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-a4 rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-a4 reset -q --hard` and check again.

- [ ] **Step 3: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`; `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md` whole (the Differences, Deviations and Equation explorer sections are binding); the approved report `C:/Users/Cole/source/repos/lsim-mag-a4/docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md` whole (sections 2.10 and 5 above all) and its data file's `e21`, `e22_e23` and `grades` keys; decision 19 of `docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`; and this plan's Decisions to confirm.

- [ ] **Step 4: Run the crate's tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test explain release_notes_are_reviewed -- --ignored 2>&1 | grep "test result"
```

Expected: every binary `ok`, in order: unit tests `172 passed`; `tests\assumptions.rs` 8 passed; `tests\deviations.rs` 54 passed; `tests\differential.rs` 19 passed; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 11 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 12 passed; `tests\schema.rs` 7 passed; `tests\sizing.rs` 31 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. (Read the counts as offsets from Task 0's if Task 0 recorded different ones.) Then the release gate: `test result: ok. 1 passed` (all 17 notes reviewed).

- [ ] **Step 5: Check the oracle Python**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
cd C:/Users/Cole/source/repos/lsim-mag-a4/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `ls` prints the path; then `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`. If `ls` fails, stop and escalate: every gate run uses that interpreter.

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task0.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 7: Decisions**

The controller asks the user the plan's **Decisions to confirm** (A4-1 to A4-4) before Task 1 and records the answers in the execution notes. Every task implements its decision's recommended option; if the user picks another option, the task that implements it stops and escalates instead of improvising.

---

### Task 1: The A-4 approval record and E21, the calibration offset clamped at 0

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5): a physics correction to the demagnetization check and its explain records. Physics reviewer (Task 6 and the per-task review): the A-4 report sections 2.1 to 2.10.

Decision 1 (A) approves E21 as section 2.10 specifies it: "option A: C50 = MAX(C49 − C47, 0) per ring, gated on E20; C50 shows the applied offset, with the help text 'E21: clamped at 0; the rating never raises the onsets'" (the port's help texts begin "Correction E21:"), with that section's record corrections (C43 and C47 in the cells, probe 4, the corrected wording) and an approval record of its own; decision 3 (A): C101's help documents that the bond screen's hot swing follows C12. The clamp sits in `ring_demag` after the offset is formed, so E20 compares the clamped limits; `py_max(offset, 0.0)` is `if 0 > x { 0 } else { x }`, the report's `< 0.0` rule exactly. Each ring's applied offset becomes a Rust-only result, and the explorer's per-ring limit records read it in place of their `where` binding, so the clamp is stated once per ring; C50's record gains `max(·, 0)`. The `demagnetization` note's fourth sentence ("the onsets are shifted so the reference magnet reaches its knee exactly at its rated temperature") becomes false, so it is rewritten and goes back to Draft (decision A4-2).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs` (module doc, `A4_REPORT`, `DeviationId::E21`, `Approval::AddendumA4`, `decision_list`, the E21 entry, unit tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs` (module doc, the clamp, C50 and C101 help, two Rust-only results, a unit test and the E20 test's fixture)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/demagnetization.rs` (two new records, three rewritten)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs` (the `demagnetization` note)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs` (the A-4 approval check, the A-4 helpers, ten E21 tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md` (Differences intro and row E21, layout, the tests row (its probes and probe-cells sentences now say E15 onward), the notes count)

**Interfaces:**
- Consumes: `temperature::ring_demag` (E20's per-ring block, `RingDemag::offset`), `compat::py_max`, `tests/deviations.rs`'s `probe_base`, `inputs_with`, `cells_with`, `demag_with`, `both_rings`, `assert_sig4`, `assert_bit_for_bit_at_defaults`.
- Produces (read by Tasks 3 to 7):
  - `pub const A4_REPORT: &str` and `Approval::AddendumA4 { decisions: &'static [u32] }` (deviations.rs); `DeviationId::E21` (`ALL` has 21 ids);
  - Rust-only `DemagResults` fields `inner_calibration_offset_C: f64`, `outer_calibration_offset_C: f64` (paths `temperature.demag.inner_calibration_offset_C`, `outer_...`);
  - in `tests/deviations.rs`: `const VERDICT_OK`, `VERDICT_CHECK`; `fn a4_bases(id: DeviationId) -> [(&'static str, Deviations, Deviations); 2]` (`"registry"`: the entry's `depends_on` then with `id`; `"user"`: `ALL.without(id)` then `ALL`); `fn grade_rings(inner: &str, outer: &str) -> [(&'static str, Value); 4]`; `fn results_with(overrides: &[(&str, Value)], dev: Deviations) -> DesignResults`; `fn assert_same_results(a: &DesignResults, b: &DesignResults, what: &str)` (every result row, bit for bit, through `fn same_value(x: &Value, y: &Value) -> bool`); `fn moved_cells(a: &DesignResults, b: &DesignResults) -> Vec<String>` (the workbook cells that differ, in row order); `type Overrides = Vec<(&'static str, Value)>`; `fn part_rings(inner: &str, outer: &str) -> [(&'static str, Value); 2]`; `fn grade_pairs() -> Vec<(String, Overrides)>` (272 ordered pairs of different grades) and `fn part_pairs() -> Vec<(String, Overrides)>` (210 of different library parts); `fn assert_a4_defaults_unchanged(id: DeviationId)` (the default cells with only `id`, and every result row on both bases); `fn assert_no_differential_case_moves_between(bases: &[(Deviations, Deviations)], what: &str)` (every corpus case, every result row, on every core); `differential_files` and `load_cases` imported from `tests/common`; `type ClampRow`.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
use common::{
    cell_values_for, json_to_value, num, read_json, read_text, repo_path, report, snapshot,
    value_to_json,
};
use magcoupling::engine::api::{DesignInputs, compute_all, compute_all_with, headline};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::{
    ADDENDUM_REPORT, Approval, Deviation, DeviationClass, DeviationId, DeviationStatus, Deviations,
    Literal, Probe, REGISTRY, REPORT,
};
```

with:

```rust
use common::{
    cell_values_for, differential_files, json_to_value, load_cases, num, read_json, read_text,
    repo_path, report, snapshot, value_to_json,
};
use magcoupling::engine::api::{
    DesignInputs, DesignResults, compute_all, compute_all_with, headline,
};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::{
    A4_REPORT, ADDENDUM_REPORT, Approval, Deviation, DeviationClass, DeviationId, DeviationStatus,
    Deviations, Literal, Probe, REGISTRY, REPORT,
};
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
    assert!(
        section8.contains("**Approved (user, 2026-09-30): option A on all 31 decisions.**"),
        "section 8 records the approval"
    );
    for d in REGISTRY {
```

with:

```rust
    assert!(
        section8.contains("**Approved (user, 2026-09-30): option A on all 31 decisions.**"),
        "section 8 records the approval"
    );
    // Section 5 of the Addendum A-4 report: its approval line and numbered decisions.
    let a4 = read_text(&repo_path(A4_REPORT));
    let section5 = a4
        .split_once("\n## 5. Decisions for the user\n")
        .map(|(_, rest)| rest)
        .expect("the Addendum A-4 report has a section 5");
    assert!(
        section5.contains("**Approved (user, 2026-10-01): option A on all 10 decisions.**"),
        "section 5 records the approval"
    );
    for d in REGISTRY {
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
                assert_eq!(d.class, DeviationClass::Engine, "{}", d.id);
            }
        }
    }
}
```

with:

```rust
                assert_eq!(d.class, DeviationClass::Engine, "{}", d.id);
            }
            // E21 to E24 have no audit row: the A-4 report's tables name cells, not statuses.
            Approval::AddendumA4 { decisions } => {
                for n in decisions {
                    let heading = format!("{n}. **");
                    assert!(
                        section5.lines().any(|line| line.starts_with(&heading)),
                        "{}: section 5 has no decision {n}",
                        d.id
                    );
                }
                assert_eq!(d.class, DeviationClass::Engine, "{}", d.id);
            }
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
    // The Addendum A entries (E15 to E20) name every cell their probes change, downstream
    // cells included, so the registry alone says where each correction shows. The M1 entries
    // (E1 to E14) name the corrected and report-named cells only: their probes change
    // hundreds of sweep and downstream cells (E7, E8), and the broad ones keep golden files.
    let mut failures = Vec::new();
    for d in REGISTRY
        .iter()
        .filter(|d| matches!(d.approval, Approval::Addendum { .. }))
    {
```

with:

```rust
    // The Addendum A and A-4 entries (E15 onward) name every cell their probes change,
    // downstream cells included, so the registry alone says where each correction shows. The
    // M1 entries (E1 to E14) name the corrected and report-named cells only: their probes
    // change hundreds of sweep and downstream cells (E7, E8), and the broad ones keep golden
    // files.
    let mut failures = Vec::new();
    for d in REGISTRY
        .iter()
        .filter(|d| !matches!(d.approval, Approval::AuditRow))
    {
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
/// The report's M2 basis: E1 to E14 on, the Addendum corrections off.
fn m2() -> Deviations {
    [
        DeviationId::E15,
        DeviationId::E16,
        DeviationId::E17,
        DeviationId::E18,
        DeviationId::E19,
        DeviationId::E20,
    ]
```

with:

```rust
/// The report's M2 basis: E1 to E14 on, the Addendum corrections off.
fn m2() -> Deviations {
    [
        DeviationId::E15,
        DeviationId::E16,
        DeviationId::E17,
        DeviationId::E18,
        DeviationId::E19,
        DeviationId::E20,
        DeviationId::E21,
    ]
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
/// The verdict C25's two texts.
const VERDICT_OK: &str = "OK on temperature. Confirm drag torque and thermal cycling by test.";
const VERDICT_CHECK: &str = "CHECK: see the rows above.";

/// The A-4 report's two bases for correction `id` (its section 1, "Conventions"): the
/// registry basis (the corrections `id` refines, then `id` too; decision 15) and the user
/// basis (every correction but `id`, then every correction).
fn a4_bases(id: DeviationId) -> [(&'static str, Deviations, Deviations); 2] {
    let registry = probe_base(&REGISTRY[id.index()]);
    [
        ("registry", registry, registry.with(id)),
        ("user", Deviations::ALL.without(id), Deviations::ALL),
    ]
}

/// Input overrides by path, applied to the defaults.
type Overrides = Vec<(&'static str, Value)>;

/// Manual dimensions with `inner` and `outer` picked as the rings' grades.
fn grade_rings(inner: &str, outer: &str) -> [(&'static str, Value); 4] {
    [
        ("coupling.magnets.part_inner", Value::Text(String::new())),
        ("coupling.magnets.part_outer", Value::Text(String::new())),
        ("coupling.magnets.grade_inner", Value::Text(inner.into())),
        ("coupling.magnets.grade_outer", Value::Text(outer.into())),
    ]
}

/// The library part `inner` on the inner ring and `outer` on the outer ring.
fn part_rings(inner: &str, outer: &str) -> [(&'static str, Value); 2] {
    [
        ("coupling.magnets.part_inner", Value::Text(inner.into())),
        ("coupling.magnets.part_outer", Value::Text(outer.into())),
    ]
}

/// Every ordered pair of different grades for manual dimensions (272 pairs, the A-4
/// harness's section 2.3), labelled "inner/outer".
fn grade_pairs() -> Vec<(String, Overrides)> {
    use magcoupling::engine::grades::GRADES;
    let mut pairs = Vec::new();
    for gi in &GRADES {
        for go in GRADES.iter().filter(|go| go.id != gi.id) {
            let rings = grade_rings(gi.id, go.id).to_vec();
            pairs.push((format!("{}/{}", gi.id, go.id), rings));
        }
    }
    pairs
}

/// Every ordered pair of different library parts (210 pairs, the A-4 harness's section 2.3),
/// labelled "inner/outer".
fn part_pairs() -> Vec<(String, Overrides)> {
    use magcoupling::engine::library::MAGNET_LIBRARY;
    let mut pairs = Vec::new();
    for pi in &MAGNET_LIBRARY {
        for po in MAGNET_LIBRARY.iter().filter(|po| po.part != pi.part) {
            let rings = part_rings(pi.part, po.part).to_vec();
            pairs.push((format!("{}/{}", pi.part, po.part), rings));
        }
    }
    pairs
}

/// Every result for `overrides` on the defaults `dev` implies, under `dev`.
fn results_with(overrides: &[(&str, Value)], dev: Deviations) -> DesignResults {
    compute_all_with(&inputs_with(overrides, dev), dev)
}

/// Bit for bit: a NaN equals a NaN of the same bits.
fn same_value(x: &Value, y: &Value) -> bool {
    match (x, y) {
        (Value::Num(p), Value::Num(q)) => p.to_bits() == q.to_bits(),
        (p, q) => p == q,
    }
}

/// Every result row, Rust-only ones included, is bit for bit the same in `a` and `b`.
fn assert_same_results(a: &DesignResults, b: &DesignResults, what: &str) {
    let (rows_a, rows_b) = (result_rows(a), result_rows(b));
    assert_eq!(rows_a.len(), rows_b.len(), "{what}");
    for (x, y) in rows_a.iter().zip(&rows_b) {
        assert!(
            same_value(&x.value, &y.value),
            "{what}: {} moved: {:?} -> {:?}",
            x.path,
            x.value,
            y.value
        );
    }
}

/// The workbook cells whose result differs, bit for bit, from `a` to `b`, in row order.
fn moved_cells(a: &DesignResults, b: &DesignResults) -> Vec<String> {
    result_rows(a)
        .into_iter()
        .zip(result_rows(b))
        .filter(|(x, y)| !same_value(&x.value, &y.value))
        .filter_map(|(x, _)| x.cell)
        .collect()
}

/// An A-4 correction keeps the defaults: every default cell bit for bit with only `id` on,
/// and every result row, Rust-only ones included, on both of its bases.
fn assert_a4_defaults_unchanged(id: DeviationId) {
    assert_bit_for_bit_at_defaults(id);
    for (basis, before, after) in a4_bases(id) {
        assert_same_results(&results_with(&[], before), &results_with(&[], after), basis);
    }
}

/// No differential case (3,391 today, every corpus file) moves from `before` to `after`, for
/// each `(before, after)` in `bases`: every result row, Rust-only ones included, bit for bit.
fn assert_no_differential_case_moves_between(bases: &[(Deviations, Deviations)], what: &str) {
    let all: Vec<(String, BTreeMap<String, Value>)> = differential_files()
        .into_iter()
        .flat_map(|file| {
            load_cases(file)
                .into_iter()
                .map(move |case| (format!("{file} case {}", case.id), case.inputs))
        })
        .collect();
    assert!(!all.is_empty());
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    std::thread::scope(|s| {
        for chunk in all.chunks(all.len().div_ceil(threads)) {
            s.spawn(move || {
                for (label, overrides) in chunk {
                    for &(before, after) in bases {
                        let mut inputs = DesignInputs::defaults_with(before);
                        for (path, value) in overrides {
                            inputs
                                .set(path, value.clone())
                                .unwrap_or_else(|e| panic!("{label}: {e}"));
                        }
                        assert_same_results(
                            &compute_all_with(&inputs, before),
                            &compute_all_with(&inputs, after),
                            &format!("{what}, {label}"),
                        );
                    }
                }
            });
        }
    });
}

#[test]
fn e21_leaves_every_default_cell_bit_for_bit() {
    // Gated on E20, E21 alone is inert. On top of E20, and with every correction, the default
    // part's offset is positive (+10.43 C on the registry basis, +9.921 C with E3's 1.30 T),
    // so the clamp keeps every result, Rust-only ones included, bit for bit.
    assert_a4_defaults_unchanged(DeviationId::E21);
}

#[test]
fn e21_alone_moves_no_differential_case() {
    // A-4 report section 2.6, the double gate: E21 refines E20, so on its own it is NONE on
    // every corpus case, every result row bit for bit.
    let alone = (Deviations::NONE, Deviations::only(DeviationId::E21));
    assert_no_differential_case_moves_between(&[alone], "E21 alone");
}

#[test]
fn e21_without_e20_changes_nothing() {
    // N42's coercivity typed into C44 and C45 on the default part: without E20 the inputs act
    // for both rings and the offset is negative (-56.55 C), yet E21 alone changes nothing,
    // because it refines E20.
    let typed = [
        ("temperature.demag.hcj20_kA_m", Value::Num(954.9)),
        ("temperature.demag.beta_hcj_per_C", Value::Num(-0.0062)),
    ];
    let workbook = results_with(&typed, Deviations::NONE);
    assert!(workbook.temperature.demag.calibration_offset_C < 0.0);
    assert_same_results(
        &workbook,
        &results_with(&typed, Deviations::only(DeviationId::E21)),
        "E21 alone",
    );
}

#[test]
fn e21_clamps_exactly_the_negative_offsets() {
    // Every grade on both rings (manual dimensions), on both bases. A grade whose model
    // reference onset C49 is at or above its rating C47 moves nothing; below it, the offset
    // is 0 and the onsets are the uncalibrated model's (A-4 report section 2.2).
    use magcoupling::engine::grades::GRADES;
    use magcoupling::engine::temperature::demag_onset_C;
    let e21 = DeviationId::E21;
    let d = DesignInputs::default().temperature.demag;
    for (basis, before, after) in a4_bases(e21) {
        let mut clamped = Vec::new();
        for g in &GRADES {
            let rings = grade_rings(g.id, g.id);
            let (b, a) = (demag_with(&rings, before), demag_with(&rings, after));
            if b.calibration_offset_C >= 0.0 {
                assert_same_results(
                    &results_with(&rings, before),
                    &results_with(&rings, after),
                    &format!("{basis} {}", g.id),
                );
                continue;
            }
            clamped.push(g.id);
            let model = demag_onset_C(
                d.h_rev_likepole_kA_m,
                g.hcj20_kA_m,
                g.beta_hcj_per_C,
                d.knee_fraction,
                g.alpha_br_per_C,
                0.0,
            );
            assert_eq!(a.calibration_offset_C, 0.0, "{basis} {}", g.id);
            assert_eq!(a.onset_skipping_C, model, "{basis} {}", g.id);
            assert_eq!(
                a.magnet_limit_C,
                model - d.design_margin_C,
                "{basis} {}",
                g.id
            );
            assert!(a.onset_skipping_C < b.onset_skipping_C, "{basis} {}", g.id);
            // Both rings carry the grade: each ring's applied offset is the block's.
            assert_eq!(
                (a.inner_calibration_offset_C, a.outer_calibration_offset_C),
                (0.0, 0.0),
                "{basis} {}",
                g.id
            );
        }
        assert_eq!(
            clamped,
            [
                "N52",
                "N50",
                "N50M",
                "N38UH",
                "N35EH",
                "SmCo_2_17_26",
                "SmCo_2_17_30"
            ],
            "{basis}"
        );
    }
}

#[test]
fn e21_clamps_exactly_the_parts_rated_above_their_model_onset() {
    // A-4 report section 2.2, the library parts on both rings: the N52 parts move on both
    // bases; the SuperMagnetMan arcs (N50 and N50M, stored ratings 80 and 100 C) move on the
    // registry basis only, since on the user basis E19's 60 C rating is below their model
    // onset (offset +12.98 C). Every other part is bit for bit unchanged, and no governing
    // limit rises.
    use magcoupling::engine::library::MAGNET_LIBRARY;
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        let mut clamped = Vec::new();
        for spec in &MAGNET_LIBRARY {
            let rings = both_rings(spec.part);
            let what = format!("{basis} {}", spec.part);
            let (b, a) = (results_with(&rings, before), results_with(&rings, after));
            if b.temperature.demag.calibration_offset_C >= 0.0 {
                assert_same_results(&b, &a, &what);
                continue;
            }
            clamped.push(spec.part);
            assert_eq!(a.temperature.demag.calibration_offset_C, 0.0, "{what}");
            let limit = |r: &DesignResults| r.temperature.summary.governing_limit_C;
            assert!(limit(&a) < limit(&b), "{what}");
        }
        clamped.sort_unstable();
        let want: &[&str] = if basis == "registry" {
            &["B842-N52", "B882-N52", "M5026", "M5044", "M5045"]
        } else {
            &["B842-N52", "B882-N52"]
        };
        assert_eq!(clamped, want, "{basis}");
    }
}

/// A row of the A-4 report's table 2.3: (grade, C50 before, C60, C12 and C24 as [before,
/// after], C25 on both sides).
type ClampRow = (
    &'static str,
    f64,
    [f64; 2],
    [f64; 2],
    [f64; 2],
    &'static str,
);

#[test]
fn e21_changes_the_report_s_cells_to_the_report_s_figures() {
    // A-4 report section 2.3, the grade on both rings: the same figures on both bases.
    let rows: [ClampRow; 7] = [
        (
            "N52",
            -9.689,
            [0.1679, -9.521],
            [0.1679, -9.521],
            [59.77, 50.09],
            VERDICT_CHECK,
        ),
        (
            "N50",
            -6.138,
            [-3.383, -9.521],
            [-3.383, -9.521],
            [56.22, 50.09],
            VERDICT_CHECK,
        ),
        (
            "N50M",
            -7.535,
            [41.90, 34.37],
            [41.90, 34.37],
            [76.80, 69.27],
            VERDICT_CHECK,
        ),
        (
            "N38UH",
            -7.343,
            [131.9, 124.6],
            [120.0, 120.0],
            [149.9, 142.6],
            VERDICT_OK,
        ),
        (
            "N35EH",
            -3.574,
            [155.4, 151.9],
            [120.0, 120.0],
            [169.3, 165.8],
            VERDICT_OK,
        ),
        (
            "SmCo_2_17_26",
            -60.21,
            [161.9, 101.7],
            [120.0, 101.7],
            [265.2, 205.0],
            VERDICT_OK,
        ),
        (
            "SmCo_2_17_30",
            -0.4500,
            [46.27, 45.82],
            [46.27, 45.82],
            [169.9, 169.4],
            VERDICT_CHECK,
        ),
    ];
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        for (grade, offset, limit, governing, cure, verdict) in rows {
            let rings = grade_rings(grade, grade);
            let (b, a) = (cells_with(&rings, before), cells_with(&rings, after));
            let what = |cell: &str| format!("{basis} {grade} {cell}");
            assert_sig4(&what("C50"), &b["Temperature design!C50"], offset);
            assert_eq!(
                a["Temperature design!C50"],
                Value::Num(0.0),
                "{basis} {grade}"
            );
            for (cell, [want_b, want_a]) in [
                ("Temperature design!C60", limit),
                ("Temperature design!C12", governing),
                ("Temperature design!C24", cure),
            ] {
                assert_sig4(&what(cell), &b[cell], want_b);
                assert_sig4(&what(cell), &a[cell], want_a);
            }
            for c in [&b, &a] {
                assert_eq!(
                    c["Temperature design!C25"],
                    Value::Text(verdict.into()),
                    "{basis} {grade}"
                );
            }
        }
    }
}

#[test]
fn e21_typed_n42_coercivity_on_a_150_c_rating_reads_check() {
    // A-4 report section 2.5, the source-0 check: N42's Hcj and beta typed into C44 and C45 on
    // the default part (rated 150 C), with the coercivity source at 0. On the user basis the
    // clamp also turns the verdict from OK to CHECK.
    let typed = [
        ("temperature.demag.hcj20_kA_m", Value::Num(954.9)),
        ("temperature.demag.beta_hcj_per_C", Value::Num(-0.0062)),
        ("temperature.demag.coercivity_source", Value::Int(0)),
    ];
    // (basis, C50 before, C12 before, C24 before, C25 before); after: 0, 9.164, 60.51, CHECK.
    let want = [
        ("registry", -56.55, 65.71, 117.1, VERDICT_CHECK),
        ("user", -57.32, 66.48, 117.8, VERDICT_OK),
    ];
    for ((basis, before, after), (label, offset, governing, cure, verdict)) in
        a4_bases(DeviationId::E21).into_iter().zip(want)
    {
        assert_eq!(basis, label);
        let (b, a) = (cells_with(&typed, before), cells_with(&typed, after));
        assert_sig4(basis, &b["Temperature design!C50"], offset);
        assert_sig4(basis, &b["Temperature design!C12"], governing);
        assert_sig4(basis, &b["Temperature design!C24"], cure);
        assert_eq!(
            b["Temperature design!C25"],
            Value::Text(verdict.into()),
            "{basis}"
        );
        assert_eq!(a["Temperature design!C50"], Value::Num(0.0), "{basis}");
        assert_sig4(basis, &a["Temperature design!C12"], 9.164);
        assert_sig4(basis, &a["Temperature design!C24"], 60.51);
        assert_eq!(
            a["Temperature design!C25"],
            Value::Text(VERDICT_CHECK.into()),
            "{basis}"
        );
    }
}

#[test]
fn e21_binds_on_the_default_part_only_below_the_knee_threshold() {
    // A-4 report section 2.3: the default part's offset turns negative below a knee fraction
    // of 0.78349 (slider step 0.01): 0.79 moves nothing, 0.78 is the first clamped setting,
    // and at 0.70 the clamp turns the default design's verdict from OK to CHECK (C12 74.81 ->
    // 65.42 C, user basis).
    let knee = |k: f64| [("temperature.demag.knee_fraction", Value::Num(k))];
    let [_, (_, before, after)] = a4_bases(DeviationId::E21);
    assert_same_results(
        &results_with(&knee(0.79), before),
        &results_with(&knee(0.79), after),
        "knee 0.79",
    );
    let first = results_with(&knee(0.78), before).temperature.demag;
    assert!(
        first.calibration_offset_C < 0.0,
        "{}",
        first.calibration_offset_C
    );
    let (b, a) = (
        cells_with(&knee(0.70), before),
        cells_with(&knee(0.70), after),
    );
    assert_sig4("C12 before", &b["Temperature design!C12"], 74.81);
    assert_sig4("C12 after", &a["Temperature design!C12"], 65.42);
    assert_eq!(b["Temperature design!C25"], Value::Text(VERDICT_OK.into()));
    assert_eq!(
        a["Temperature design!C25"],
        Value::Text(VERDICT_CHECK.into())
    );
}

#[test]
fn e21_reorders_the_rings_where_the_clamp_moves_a_limit_and_ties_stay_inner() {
    // Probe 3's pair: B842 inside (N42, offset +12.68 C), B842-N52 outside (offset -9.689 C).
    // Clamped, the N52 ring has the lower limit: the block shows the outer ring. Each ring's
    // applied offset is a Rust-only result: the inner one is kept, the outer one is 0.
    use magcoupling::engine::temperature::{RING_INNER, RING_OUTER};
    let rings = part_rings("B842", "B842-N52");
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        let (b, a) = (demag_with(&rings, before), demag_with(&rings, after));
        assert_eq!(
            (b.demag_ring.as_str(), a.demag_ring.as_str()),
            (RING_INNER, RING_OUTER),
            "{basis}"
        );
        assert_eq!(a.inner_calibration_offset_C, b.inner_calibration_offset_C);
        assert_sig4(basis, &Value::Num(b.outer_calibration_offset_C), -9.689);
        assert_eq!(a.outer_calibration_offset_C, 0.0, "{basis}");
    }
    // N52 beside N50 is the one exact tie after the clamp (the A-4 report's "Ties"): both
    // skipping onsets are 0.4787157208430415 C, and the tie resolves to the inner ring.
    let tie = demag_with(&grade_rings("N52", "N50"), Deviations::ALL);
    assert_eq!(tie.onset_skipping_C, 0.4787157208430415);
    assert_eq!(tie.inner_magnet_limit_C, tie.outer_magnet_limit_C);
    assert_eq!(tie.demag_ring, RING_INNER);
}

#[test]
fn e21_moves_only_pairs_with_a_negative_offset_and_never_raises_a_limit() {
    // A-4 report sections 2.3 and 2.6, every ordered pair of different grades and of
    // different library parts (482 per basis): a pair in which neither ring's offset is
    // negative is bit for bit unchanged, Rust-only rows included; in every other pair the
    // clamp never raises the skipping onset C58, the magnet limit C60, the governing limit
    // C12 or the hot-day margin C15.
    for (basis, before, after) in a4_bases(DeviationId::E21) {
        let (mut moved, mut flips, mut verdicts) = (0, 0, 0);
        for (label, rings) in grade_pairs().into_iter().chain(part_pairs()) {
            let what = format!("{basis} {label}");
            let (b, a) = (results_with(&rings, before), results_with(&rings, after));
            let (db, da) = (&b.temperature.demag, &a.temperature.demag);
            if db.inner_calibration_offset_C >= 0.0 && db.outer_calibration_offset_C >= 0.0 {
                assert_same_results(&b, &a, &what);
                continue;
            }
            let (sb, sa) = (&b.temperature.summary, &a.temperature.summary);
            assert!(da.onset_skipping_C <= db.onset_skipping_C, "{what}: C58");
            assert!(da.magnet_limit_C <= db.magnet_limit_C, "{what}: C60");
            assert!(sa.governing_limit_C <= sb.governing_limit_C, "{what}: C12");
            assert!(sa.margin_hot_day_C <= sb.margin_hot_day_C, "{what}: C15");
            moved += usize::from(!moved_cells(&b, &a).is_empty());
            flips += usize::from(db.demag_ring != da.demag_ring);
            verdicts += usize::from(sb.verdict != sa.verdict);
        }
        // (pairs with a moved cell, governing-ring flips, verdict changes): the report's table.
        let want = match basis {
            "registry" => (214, 71, 0),
            _ => (150, 39, 0),
        };
        assert_eq!((moved, flips, verdicts), want, "{basis}");
    }
}

#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

In the unit tests of `temperature.rs` (the E20 test's fixture copies and blanks the new outer-ring field, as it does the others; then a new test):

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            // Plan A-3's per-ring results: the outer ring is the ferrite one.
            want.outer_hcj20_kA_m = cold.outer_hcj20_kA_m;
            want.outer_beta_per_C = cold.outer_beta_per_C;
            want.outer_magnet_limit_C = cold.outer_magnet_limit_C;
            want.outer_cold_limit_C = cold.outer_cold_limit_C;
```

with:

```rust
            // Plan A-3's per-ring results: the outer ring is the ferrite one.
            want.outer_hcj20_kA_m = cold.outer_hcj20_kA_m;
            want.outer_beta_per_C = cold.outer_beta_per_C;
            want.outer_magnet_limit_C = cold.outer_magnet_limit_C;
            want.outer_cold_limit_C = cold.outer_cold_limit_C;
            want.outer_calibration_offset_C = cold.outer_calibration_offset_C;
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        let workbook_cells = |mut r: TemperatureResults| {
            r.demag.outer_hcj20_kA_m = 0.0;
            r.demag.outer_beta_per_C = 0.0;
            r.demag.outer_magnet_limit_C = 0.0;
            r.demag.outer_cold_limit_C = NumOrText::Text("");
            r
        };
```

with:

```rust
        let workbook_cells = |mut r: TemperatureResults| {
            r.demag.outer_hcj20_kA_m = 0.0;
            r.demag.outer_beta_per_C = 0.0;
            r.demag.outer_magnet_limit_C = 0.0;
            r.demag.outer_cold_limit_C = NumOrText::Text("");
            r.demag.outer_calibration_offset_C = 0.0;
            r
        };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    #[test]
    fn each_ring_shows_its_own_block() {
```

with:

```rust
    #[test]
    fn e21_clamps_a_negative_offset_on_each_ring_and_keeps_the_rest() {
        // A-4 decision 1: on top of E20 each ring's offset is MAX(C49 - C47, 0). An N52 outer
        // ring (Br 1.45 T, rated 80 C) has a negative offset: clamped to 0, its limit is the
        // uncalibrated model's skipping onset less the margin. The N42SH inner ring's positive
        // offset is kept bit for bit, E21 without E20 is inert, and a NaN offset (a NaN
        // rating) stays NaN.
        let e20 = Deviations::only(DeviationId::E20);
        let both = e20.with(DeviationId::E21);
        let ti = TemperatureInputs::default();
        let mut k = links();
        k.outer_grade = crate::engine::grades::grade("N52");
        k.outer_br20_T = 1.45;
        k.outer_tmax_lib_C = NumOrText::Num(80.0);
        let before = compute(&ti, &k, e20).demag;
        let after = compute(&ti, &k, both).demag;
        assert!(before.outer_calibration_offset_C < 0.0);
        assert_eq!(after.outer_calibration_offset_C, 0.0);
        assert!(before.inner_calibration_offset_C > 0.0);
        assert_eq!(
            after.inner_calibration_offset_C.to_bits(),
            before.inner_calibration_offset_C.to_bits()
        );
        let d = &ti.demag;
        let n52 = k.outer_grade.expect("the N52 grade");
        let model = demag_onset_C(
            d.h_rev_likepole_kA_m,
            n52.hcj20_kA_m,
            n52.beta_hcj_per_C,
            d.knee_fraction,
            k.outer_alpha_br,
            0.0,
        );
        assert_eq!(after.outer_magnet_limit_C, model - d.design_margin_C);
        assert!(after.outer_magnet_limit_C < before.outer_magnet_limit_C);
        assert_eq!(
            compute(&ti, &k, Deviations::only(DeviationId::E21)),
            run(&ti, &k),
            "E21 refines E20"
        );
        k.outer_tmax_lib_C = NumOrText::Num(f64::NAN);
        assert!(
            compute(&ti, &k, both)
                .demag
                .outer_calibration_offset_C
                .is_nan()
        );
    }

    #[test]
    fn each_ring_shows_its_own_block() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test deviations 2>&1 | grep -E "^error" | sort | uniq -c
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --lib 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: compile errors only, the first command:

```text
1 error: could not compile `magcoupling-rs` (test "deviations") due to 21 previous errors
1 error[E0432]: unresolved import `magcoupling::engine::deviations::A4_REPORT`
1 error[E0599]: no variant named `AddendumA4` found for enum `Approval`
11 error[E0599]: no variant or associated item named `E21` found for enum `DeviationId` in the current scope
1 error[E0609]: no field `inner_calibration_offset_C` on type `&DemagResults`
3 error[E0609]: no field `inner_calibration_offset_C` on type `DemagResults`
1 error[E0609]: no field `outer_calibration_offset_C` on type `&DemagResults`
3 error[E0609]: no field `outer_calibration_offset_C` on type `DemagResults`
```

and the second:

```text
1 error: could not compile `magcoupling-rs` (lib test) due to 11 previous errors
2 error[E0599]: no variant or associated item named `E21` found for enum `deviations::DeviationId` in the current scope
3 error[E0609]: no field `inner_calibration_offset_C` on type `DemagResults`
6 error[E0609]: no field `outer_calibration_offset_C` on type `DemagResults`
```

- [ ] **Step 3: Implement E21, its record and its explain terms**

The registry: the A-4 report constant and approval variant, the id, and the entry with its 33 cells and four probes (values from the report's section 2.5 and the data file's `e21.probes`). The engine: the clamp and the two per-ring results. The explorer: each ring's offset record (the clamp), the per-ring limits reading it, C50's `max`. The note: the sentence and the source, then the script sets the note to Draft. The README.

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! parts) and E20 (decision 19: each magnet's own coercivity), all approved on
//! 2026-09-30. Everything else stays workbook-exact. This registry is the one
//! place that says where the port departs from the workbook and why.
```

with:

```rust
//! parts) and E20 (decision 19: each magnet's own coercivity), all approved on
//! 2026-09-30. The Addendum A-4 verification
//! (`docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md`) added
//! E21 (decision 1: the rating-calibration offset clamped at 0), approved on
//! 2026-10-01. Everything else stays workbook-exact. This registry is the one
//! place that says where the port departs from the workbook and why.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! - **Probes.** A correction that changes nothing at default inputs (E7 to
//!   E13, E15 to E20) lists `probes`: input overrides on which it shows (the
```

with:

```rust
//! - **Probes.** A correction that changes nothing at default inputs (E7 to
//!   E13, E15 onward) lists `probes`: input overrides on which it shows (the
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//!   `depends_on` (E15, E16 and E17 refine E9: without it the cup is steel).
```

with:

```rust
//!   `depends_on` (E15, E16 and E17 refine E9: without it the cup is steel;
//!   E21 refines E20: without it there is one calibration, the inner ring's).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
/// The Addendum A verification report E15 to E20 cite (section 8 holds the
/// decisions the user approved on 2026-09-30).
pub const ADDENDUM_REPORT: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md";
```

with:

```rust
/// The Addendum A verification report E15 to E20 cite (section 8 holds the
/// decisions the user approved on 2026-09-30).
pub const ADDENDUM_REPORT: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md";

/// The Addendum A-4 verification report E21 onward cite (section 5 holds the decisions the
/// user approved on 2026-10-01; its data file is
/// `docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json`).
pub const A4_REPORT: &str = "docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md";
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    E19,
    E20,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 20] = [
```

with:

```rust
    E19,
    E20,
    E21,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 21] = [
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        DeviationId::E19,
        DeviationId::E20,
    ];
```

with:

```rust
        DeviationId::E19,
        DeviationId::E20,
        DeviationId::E21,
    ];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    /// E15 to E18 also have an audit row there; E19 and E20 have none.
    Addendum { decisions: &'static [u32] },
}
```

with:

```rust
    /// E15 to E18 also have an audit row there; E19 and E20 have none.
    Addendum { decisions: &'static [u32] },
    /// These decisions of the Addendum A-4 verification report ([`A4_REPORT`],
    /// section 5), approved (option A) on 2026-10-01. No audit row.
    AddendumA4 { decisions: &'static [u32] },
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    /// Corrections this one refines; its probes run with them on, on both
    /// sides (decision 15). Empty for all but E15, E16 and E17 (on E9).
```

with:

```rust
    /// Corrections this one refines; its probes run with them on, on both
    /// sides (decision 15). Empty for all but E15, E16 and E17 (on E9) and
    /// E21 (on E20).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    /// Off-default checks for corrections neutral at defaults (E7 to E13,
    /// E15 to E20).
```

with:

```rust
    /// Off-default checks for corrections neutral at defaults (E7 to E13,
    /// E15 onward).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            Approval::AuditRow => REPORT,
            Approval::Addendum { .. } => ADDENDUM_REPORT,
        }
    }

    /// Where the evidence is: the audit report entry, or the Addendum A decisions.
    pub fn evidence(&self) -> String {
        match self.approval {
            Approval::AuditRow => format!("{REPORT}, entry {}", self.id),
            Approval::Addendum { decisions } => {
                let list: Vec<String> = decisions.iter().map(u32::to_string).collect();
                let noun = if decisions.len() == 1 {
                    "decision"
                } else {
                    "decisions"
                };
                format!("{ADDENDUM_REPORT}, section 8, {noun} {}", list.join(", "))
            }
        }
    }
}
```

with:

```rust
            Approval::AuditRow => REPORT,
            Approval::Addendum { .. } => ADDENDUM_REPORT,
            Approval::AddendumA4 { .. } => A4_REPORT,
        }
    }

    /// Where the evidence is: the audit report entry, or the Addendum A or A-4 decisions.
    pub fn evidence(&self) -> String {
        match self.approval {
            Approval::AuditRow => format!("{REPORT}, entry {}", self.id),
            Approval::Addendum { decisions } => {
                format!("{ADDENDUM_REPORT}, section 8, {}", decision_list(decisions))
            }
            Approval::AddendumA4 { decisions } => {
                format!("{A4_REPORT}, section 5, {}", decision_list(decisions))
            }
        }
    }
}

/// "decision 2" or "decisions 10, 11": how the evidence names a report's decisions.
fn decision_list(decisions: &[u32]) -> String {
    let list: Vec<String> = decisions.iter().map(u32::to_string).collect();
    let noun = if decisions.len() == 1 {
        "decision"
    } else {
        "decisions"
    };
    format!("{noun} {}", list.join(", "))
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
                    CellChange {
                        cell: "Temperature design!C25",
                        workbook: Literal::Text(
                            "OK on temperature. Confirm drag torque and thermal cycling by test.",
                        ),
                        corrected: Literal::Text("CHECK: see the rows above."),
                    },
                ],
            },
        ],
    },
];
```

with:

```rust
                    CellChange {
                        cell: "Temperature design!C25",
                        workbook: Literal::Text(
                            "OK on temperature. Confirm drag torque and thermal cycling by test.",
                        ),
                        corrected: Literal::Text("CHECK: see the rows above."),
                    },
                ],
            },
        ],
    },
    Deviation {
        id: DeviationId::E21,
        title: "The rating calibration raises a grade's demagnetization onsets above the model when its rating exceeds the model's reference onset",
        class: DeviationClass::Engine,
        approval: Approval::AddendumA4 { decisions: &[1, 3] },
        depends_on: &[DeviationId::E20],
        status: DeviationStatus::Applied,
        cells: &[
            "Temperature design!C7",
            "Temperature design!C8",
            "Temperature design!C9",
            "Temperature design!C10",
            "Temperature design!C12",
            "Temperature design!C13",
            "Temperature design!C15",
            "Temperature design!C19",
            "Temperature design!C23",
            "Temperature design!C24",
            "Temperature design!C42",
            "Temperature design!C43",
            "Temperature design!C47",
            "Temperature design!C48",
            "Temperature design!C49",
            "Temperature design!C50",
            "Temperature design!C56",
            "Temperature design!C57",
            "Temperature design!C58",
            "Temperature design!C59",
            "Temperature design!C60",
            "Temperature design!C61",
            "Temperature design!C101",
            "Temperature design!C104",
            "Temperature design!C105",
            "Temperature design!C106",
            "Temperature design!C150",
            "Temperature design!C151",
            "Temperature design!C152",
            "Temperature design!C153",
            "Temperature design!C181",
            "Temperature design!C182",
            "Temperature design!F12",
        ],
        corrected_formula: "C50 = MAX(C49 - C47, 0) for each ring (E20 checks both), applied before E20 picks the \
            weaker ring: the rating calibrates the model's onsets down, never up (equivalently, each onset is the lower \
            of the calibrated and the uncalibrated one). Gated on E20; a positive beta (the cold side) and a magnet \
            without a rating keep an offset of 0, and a NaN offset stays NaN. It takes the lower of two unvalidated \
            estimates: conservative for the demagnetization check (C58, C60, C12 and C15 never rise), not uniformly \
            downstream: the mismatch screen C101 to C106 relaxes where the magnets come to govern (decision 3, \
            documented in C101's help), and the shown cure margin C24 can rise under a governing-ring flip (E24 takes \
            the lower ring's). C50 shows the applied offset (its help: clamped at 0).",
        workbook_input_defaults: &[],
        workbook_help: &[
            ("temperature.demag.calibration_offset_C", ""),
            ("temperature.mismatch.worst_swing_C", ""),
        ],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[
            Probe {
                label: "B842-N52 (N52: rating 80 C, model reference onset 70.31 C) on both rings, the library path: \
                    offset -9.689 C clamped to 0 (supersedes Addendum A report 6.3 row B842-N52 under E20: C12 0.17 -> -9.52 C)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("B842-N52")),
                    ("coupling.magnets.part_outer", Literal::Text("B842-N52")),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C50",
                        workbook: Literal::Num(-9.689222777921628),
                        corrected: Literal::Num(0.0),
                    },
                    CellChange {
                        cell: "Temperature design!C58",
                        workbook: Literal::Num(10.16793849876467),
                        corrected: Literal::Num(0.4787157208430415),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(0.16793849876466993),
                        corrected: Literal::Num(-9.521284279156959),
                    },
                ],
            },
            Probe {
                label: "Recoma 26 (SmCo_2_17_26: rating 350 C, model reference onset 289.8 C) for manual dimensions on \
                    both rings: offset -60.21 C clamped to 0; the magnets govern instead of the adhesive",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("")),
                    ("coupling.magnets.part_outer", Literal::Text("")),
                    (
                        "coupling.magnets.grade_inner",
                        Literal::Text("SmCo_2_17_26"),
                    ),
                    (
                        "coupling.magnets.grade_outer",
                        Literal::Text("SmCo_2_17_26"),
                    ),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C50",
                        workbook: Literal::Num(-60.21319132809839),
                        corrected: Literal::Num(0.0),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(120.0),
                        corrected: Literal::Num(101.73342351672972),
                    },
                    CellChange {
                        cell: "Temperature design!F12",
                        workbook: Literal::Text("Adhesive governs."),
                        corrected: Literal::Text("Magnets govern (skipping case)."),
                    },
                ],
            },
            Probe {
                label: "B842 inner (N42, offset +12.68 C), B842-N52 outer (N52, offset -9.689 C): the clamp moves the \
                    governing ring from inner to outer, and the block shows the N52 ring",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("B842")),
                    ("coupling.magnets.part_outer", Literal::Text("B842-N52")),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C42",
                        workbook: Literal::Num(1.3),
                        corrected: Literal::Num(1.45),
                    },
                    CellChange {
                        cell: "Temperature design!C50",
                        workbook: Literal::Num(12.681126305465526),
                        corrected: Literal::Num(0.0),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(-3.5174216150834887),
                        corrected: Literal::Num(-9.521284279156959),
                    },
                    CellChange {
                        cell: "Temperature design!C24",
                        workbook: Literal::Num(47.83256962065032),
                        corrected: Literal::Num(50.08556444987687),
                    },
                ],
            },
            Probe {
                label: "N38UH inner, Recoma 26 outer (manual dimensions): the clamp moves the governing ring to \
                    Recoma 26, so the block's alpha(Br) C43 and rating C47 change (A-4 report probe 4; C24 here is \
                    the governing ring's, which E24 replaces by the lower ring's)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("")),
                    ("coupling.magnets.part_outer", Literal::Text("")),
                    ("coupling.magnets.grade_inner", Literal::Text("N38UH")),
                    (
                        "coupling.magnets.grade_outer",
                        Literal::Text("SmCo_2_17_26"),
                    ),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C43",
                        workbook: Literal::Num(-0.0012),
                        corrected: Literal::Num(-0.00035),
                    },
                    CellChange {
                        cell: "Temperature design!C47",
                        workbook: Literal::Num(180.0),
                        corrected: Literal::Num(350.0),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(120.0),
                        corrected: Literal::Num(101.73342351672972),
                    },
                    CellChange {
                        cell: "Temperature design!C24",
                        workbook: Literal::Num(149.91967217479288),
                        corrected: Literal::Num(205.01249772124206),
                    },
                ],
            },
        ],
    },
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            assert!(d.evidence().starts_with(d.report()), "{}", d.id);
            match d.approval {
                Approval::AuditRow => {
                    assert!(d.id.index() < 14, "{}: only E1 to E14 are M1 rows", d.id);
                    assert!(d.evidence().ends_with(&format!("entry {}", d.id)));
                }
                Approval::Addendum { decisions } => {
                    assert!(d.id.index() >= 14, "{}: E1 to E14 are M1 rows", d.id);
                    assert!(!decisions.is_empty(), "{} cites no decision", d.id);
                    assert!(decisions.iter().all(|n| (1..=31).contains(n)), "{}", d.id);
                }
            }
```

with:

```rust
            assert!(d.evidence().starts_with(d.report()), "{}", d.id);
            // An Addendum entry is newer than the report's first id and cites its decisions.
            let decided = |decisions: &[u32], first: DeviationId, last_decision: u32| {
                assert!(
                    d.id.index() >= first.index(),
                    "{}: the report starts at {first}",
                    d.id
                );
                assert!(!decisions.is_empty(), "{} cites no decision", d.id);
                assert!(
                    decisions.iter().all(|n| (1..=last_decision).contains(n)),
                    "{}",
                    d.id
                );
            };
            match d.approval {
                Approval::AuditRow => {
                    assert!(d.id.index() < 14, "{}: only E1 to E14 are M1 rows", d.id);
                    assert!(d.evidence().ends_with(&format!("entry {}", d.id)));
                }
                Approval::Addendum { decisions } => decided(decisions, DeviationId::E15, 31),
                Approval::AddendumA4 { decisions } => decided(decisions, DeviationId::E21, 10),
            }
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        assert_eq!(
            REGISTRY[DeviationId::E17.index()].evidence(),
            format!("{ADDENDUM_REPORT}, section 8, decisions 10, 11, 12, 13, 14, 15")
        );
    }
```

with:

```rust
        assert_eq!(
            REGISTRY[DeviationId::E17.index()].evidence(),
            format!("{ADDENDUM_REPORT}, section 8, decisions 10, 11, 12, 13, 14, 15")
        );
        assert_eq!(
            REGISTRY[DeviationId::E21.index()].evidence(),
            format!("{A4_REPORT}, section 5, decisions 1, 3")
        );
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! is not 1) and E20 (the demagnetization block checks each ring against its
//! own grade, Br and rating, and shows the ring with the lower limit; a positive
//! beta is limited on the cold side, [`cold_onset_C`]).
```

with:

```rust
//! is not 1), E20 (the demagnetization block checks each ring against its
//! own grade, Br and rating, and shows the ring with the lower limit; a positive
//! beta is limited on the cold side, [`cold_onset_C`]) and E21 (on top of E20:
//! each ring's rating-calibration offset C50 is clamped at 0, so a rating never
//! raises the onsets above the model's own).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            calibration_offset_C: f64 => out("°C", "Calibration offset (model minus library rating)", "", "Temperature design!C50"),
```

with:

```rust
            calibration_offset_C: f64 => out("°C", "Calibration offset (model minus library rating)",
                "Correction E21: clamped at 0; the rating never raises the onsets.", "Temperature design!C50"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            outer_cold_limit_C: NumOrText => out_rust_only("°C", "Outer ring: cold magnet limit",
                "Correction E20: as the inner ring's, for the outer ring."),
```

with:

```rust
            outer_cold_limit_C: NumOrText => out_rust_only("°C", "Outer ring: cold magnet limit",
                "Correction E20: as the inner ring's, for the outer ring."),
            inner_calibration_offset_C: f64 => out_rust_only("°C", "Inner ring: calibration offset applied",
                "Plan A-4: the inner ring's own offset (its C49 minus its C47), whichever ring governs; 0 with a positive beta or without a rating. Correction E21: clamped at 0, so the rating never raises the ring's onsets."),
            outer_calibration_offset_C: f64 => out_rust_only("°C", "Outer ring: calibration offset applied",
                "Plan A-4: as the inner ring's, for the outer ring."),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            worst_swing_C: f64 => out("°C", "Worst swing from the stress-free (cure) temperature", "", "Temperature design!C101"),
```

with:

```rust
            worst_swing_C: f64 => out("°C", "Worst swing from the stress-free (cure) temperature",
                "The larger of the cold swing (the cure temperature C77 minus the cold limit C100) and the hot swing (the governing limit C12 minus C77): the bond is screened over the range the design is rated to, so where the magnets govern, a lower magnet limit (correction E21 can lower it) shrinks the hot swing and the peak shears with it (Addendum A-4 decision 3).",
                "Temperature design!C101"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        NumOrText::Num(tmax) => t_ref - tmax,
        NumOrText::Text(_) => 0.0,
    };
```

with:

```rust
        NumOrText::Num(tmax) => t_ref - tmax,
        NumOrText::Text(_) => 0.0,
    };
    // E21 (refines E20; Addendum A-4 decision 1): the rating may lower the model's onsets,
    // never raise them. A negative offset (the model's reference onset below the rating) would
    // lift every onset above the model's own; clamped, the onsets are the uncalibrated model's
    // (each onset is the lower of the calibrated and the uncalibrated one). py_max(x, 0) is
    // `if 0 > x { 0 } else { x }`: a NaN offset stays NaN, and every offset of 0 or more
    // (-0.0 too) is kept bit for bit.
    let offset = if dev.is_on(DeviationId::E20) && dev.is_on(DeviationId::E21) {
        py_max(offset, 0.0)
    } else {
        offset
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        outer_magnet_limit_C: outer_block.mag_lim,
        outer_cold_limit_C: outer_block.cold_limit,
    };
```

with:

```rust
        outer_magnet_limit_C: outer_block.mag_lim,
        outer_cold_limit_C: outer_block.cold_limit,
        // Plan A-4: each ring's applied offset (E21 clamps it), the explorer's term for the
        // clamp.
        inner_calibration_offset_C: inner.offset,
        outer_calibration_offset_C: outer_block.offset,
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/demagnetization.rs`, replace:

```rust
use crate::engine::deviations::DeviationId::{E19, E20};
```

with:

```rust
use crate::engine::deviations::DeviationId::{E19, E20, E21};
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/demagnetization.rs`, replace:

```rust
    // A ring's magnet limit: its skipping onset minus the margin, the onset calibrated so the
    // reference magnet (permeance coefficient 1) reaches the knee at the rating; with a positive
    // beta (hard ferrite) the knee is never reached on heating and the rating is the limit.
    record("temperature.demag.inner_magnet_limit_C", "ϑ_{lim,i}",
        r#"cases({temperature.demag.inner_beta_per_C} > 0
                   => cases({model.inner_tmax_C} != "n/a" => {model.inner_tmax_C}; else => inf);
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.inner_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.inner_alpha_br_per_C}))
                       - [Δϑ_{cal,i}] - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m},
                 [H_{ref,i}] = frac({model.inner_br_T}, 2 * {coupling.mu0}) / 1000,
                 [Δϑ_{cal,i}] = cases({model.inner_tmax_C} != "n/a"
                     => 20 + frac([H_k] - [H_{ref,i}], [H_k] · abs({temperature.demag.inner_beta_per_C}) - [H_{ref,i}] · abs({model.inner_alpha_br_per_C}))
                        - {model.inner_tmax_C};
                     else => 0)"#).corrected(&[E20]),
    record("temperature.demag.outer_magnet_limit_C", "ϑ_{lim,o}",
        r#"cases({temperature.demag.outer_beta_per_C} > 0
                   => cases({model.outer_tmax_C} != "n/a" => {model.outer_tmax_C}; else => inf);
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.outer_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.outer_alpha_br_per_C}))
                       - [Δϑ_{cal,o}] - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m},
                 [H_{ref,o}] = frac({model.outer_br_T}, 2 * {coupling.mu0}) / 1000,
                 [Δϑ_{cal,o}] = cases({model.outer_tmax_C} != "n/a"
                     => 20 + frac([H_k] - [H_{ref,o}], [H_k] · abs({temperature.demag.outer_beta_per_C}) - [H_{ref,o}] · abs({model.outer_alpha_br_per_C}))
                        - {model.outer_tmax_C};
                     else => 0)"#).corrected(&[E20]),
```

with:

```rust
    // A ring's calibration offset: the model's onset for the reference magnet (permeance
    // coefficient 1) minus the ring's rating, so the reference magnet reaches its knee at the
    // rating; E21 clamps it at 0, so a rating never raises the onsets. None with a positive beta
    // (hard ferrite: the rating is not a knee rating) or without a rating.
    record("temperature.demag.inner_calibration_offset_C", "Δϑ_{cal,i}",
        r#"cases({temperature.demag.inner_beta_per_C} > 0 or {model.inner_tmax_C} = "n/a" => 0;
               else => max(20 + frac([H_k] - [H_{ref,i}], [H_k] · abs({temperature.demag.inner_beta_per_C}) - [H_{ref,i}] · abs({model.inner_alpha_br_per_C}))
                           - {model.inner_tmax_C}, 0))
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m},
                 [H_{ref,i}] = frac({model.inner_br_T}, 2 * {coupling.mu0}) / 1000"#).corrected(&[E20, E21]),
    record("temperature.demag.outer_calibration_offset_C", "Δϑ_{cal,o}",
        r#"cases({temperature.demag.outer_beta_per_C} > 0 or {model.outer_tmax_C} = "n/a" => 0;
               else => max(20 + frac([H_k] - [H_{ref,o}], [H_k] · abs({temperature.demag.outer_beta_per_C}) - [H_{ref,o}] · abs({model.outer_alpha_br_per_C}))
                           - {model.outer_tmax_C}, 0))
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m},
                 [H_{ref,o}] = frac({model.outer_br_T}, 2 * {coupling.mu0}) / 1000"#).corrected(&[E20, E21]),
    // A ring's magnet limit: its skipping onset, calibrated by the ring's offset, minus the
    // margin; with a positive beta (hard ferrite) the knee is never reached on heating and the
    // rating is the limit.
    record("temperature.demag.inner_magnet_limit_C", "ϑ_{lim,i}",
        r#"cases({temperature.demag.inner_beta_per_C} > 0
                   => cases({model.inner_tmax_C} != "n/a" => {model.inner_tmax_C}; else => inf);
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.inner_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.inner_alpha_br_per_C}))
                       - {temperature.demag.inner_calibration_offset_C} - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m}"#).corrected(&[E20]),
    record("temperature.demag.outer_magnet_limit_C", "ϑ_{lim,o}",
        r#"cases({temperature.demag.outer_beta_per_C} > 0
                   => cases({model.outer_tmax_C} != "n/a" => {model.outer_tmax_C}; else => inf);
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.outer_beta_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({model.outer_alpha_br_per_C}))
                       - {temperature.demag.outer_calibration_offset_C} - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m}"#).corrected(&[E20]),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/demagnetization.rs`, replace:

```rust
    record("temperature.demag.calibration_offset_C", "Δϑ_{cal}",
        r#"cases({temperature.demag.beta_used_per_C} > 0 => 0;
               {temperature.demag.tmax_lib_C} != "n/a" => {temperature.demag.t_ref_model_C} - {temperature.demag.tmax_lib_C};
               else => 0)"#).corrected(&[E20]),
```

with:

```rust
    record("temperature.demag.calibration_offset_C", "Δϑ_{cal}",
        r#"cases({temperature.demag.beta_used_per_C} > 0 => 0;
               {temperature.demag.tmax_lib_C} != "n/a" => max({temperature.demag.t_ref_model_C} - {temperature.demag.tmax_lib_C}, 0);
               else => 0)"#).corrected(&[E20, E21]),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
            "temperature.demag.inner_magnet_limit_C",
            "temperature.demag.outer_magnet_limit_C",
            "temperature.demag.demag_ring",
        ],
```

with:

```rust
            "temperature.demag.inner_magnet_limit_C",
            "temperature.demag.outer_magnet_limit_C",
            "temperature.demag.inner_calibration_offset_C",
            "temperature.demag.outer_calibration_offset_C",
            "temperature.demag.demag_ring",
        ],
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
            "The onsets are shifted so the reference magnet reaches its knee exactly at its rated temperature, and the design limit keeps a 10 °C margin below the skipping onset.",
            "With correction E20 each ring is checked with its own grade, and the ring with the lower limit governs.",
        ],
```

with:

```rust
            "Where the model puts the reference magnet's knee above its rated temperature, the onsets are shifted down until it sits at the rating; a rating above the model's own onset never raises them (correction E21), and the design limit keeps a 10 °C margin below the skipping onset.",
            "With correction E20 each ring is checked with its own grade, and the ring with the lower limit governs.",
        ],
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
            "Addendum A decision 19 (E20: each ring's own grade)",
        ],
```

with:

```rust
            "Addendum A decision 19 (E20: each ring's own grade)",
            "Addendum A-4 decision 1 (E21: the calibration offset clamped at 0; report section 2.1)",
        ],
```

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import pathlib

# A correction changed this note's text: it goes back to Draft (hidden by notes::note_for) until
# the physics review of Task 6 signs it off again. Anchored on the note id, whatever its review
# record says.
NOTE = "demagnetization"
p = pathlib.Path("C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs")
s = p.read_text(encoding="utf-8")
start = s.index(f'        id: "{NOTE}",\n')
at = s.index("        review: Review::Reviewed {\n", start)
end = s.index("        },\n", at) + len("        },\n")
assert s.find("        id: ", start + 1, end) == -1, f"{NOTE}: no reviewed block before the next note"
s = s[:at] + "        review: Review::Draft,\n" + s[end:]
p.write_text(s, encoding="utf-8", newline="\n")
print(f"{NOTE} is a draft again")
EOF
```

Expected: prints `demagnetization is a draft again`.

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
id) and E15 to E20 in the Addendum A verification report
(`docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the decisions
of its section 8 that each entry's `approval` cites), and registered in `src/engine/deviations.rs` with the cells it changes and
```

with:

```markdown
id), E15 to E20 in the Addendum A verification report
(`docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the decisions
of its section 8 that each entry's `approval` cites) and E21 in the Addendum A-4
verification report (`docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md`,
the decisions of its section 5), and registered in `src/engine/deviations.rs` with the cells it changes and
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
probe one on top of the corrections it refines (`depends_on`: E15 to E17 on E9,
decision 15), or to start from the workbook's defaults.
```

with:

```markdown
probe one on top of the corrections it refines (`depends_on`: E15 to E17 on E9,
E21 on E20; decision 15), or to start from the workbook's defaults.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
B842 on both rings: C12 23.06 → −3.52 °C; B842SH inside B842: 92.55 → −3.52 °C. N42SH keeps 1592 and −0.005 (decisions 17, 18), so defaults do not move | Addendum A decision 19 |
```

with:

```markdown
B842 on both rings: C12 23.06 → −3.52 °C; B842SH inside B842: 92.55 → −3.52 °C. N42SH keeps 1592 and −0.005 (decisions 17, 18), so defaults do not move | Addendum A decision 19 |
| E21 | Temperature design!C50 → C56-C61, C7-C10, C12, C13, C15, C19, C23, C24, C150-C153, C181, C182; on a governing-ring flip also C42, C43, C47-C49 and F12; where the magnets come to govern, C101, C104-C106 | on top of E20, C50 = C49 − C47 per ring, so a rating above the model's own reference onset raises every onset (N52 −9.689 °C, Recoma 26 −60.21 °C) | C50 = MAX(C49 − C47, 0) per ring, before the weaker ring is picked: a rating lowers the onsets, never raises them (each onset is the lower of the calibrated and the uncalibrated one); C50's help says so and C101's documents that the bond screen's hot swing follows C12; each ring's applied offset is a Rust-only result. B842-N52 on both rings: C12 0.17 → −9.52 °C; Recoma 26 on both rings: C12 120 → 101.7 °C (the magnets govern). The default design's offset is +9.92 °C: unchanged | Addendum A-4 decisions 1, 3 |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E20 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A), probes, and the `Deviations` switch |
```

with:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E21 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A, E21 from Addendum A-4), probes, and the `Deviations` switch |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
`TemperatureResults` (130 cells, plus the Rust-only results of E20: each ring's own Hcj, beta, magnet limit and cold limit, plan A-3) |
```

with:

```markdown
`TemperatureResults` (130 cells, plus the Rust-only results of E20: each ring's own Hcj, beta, magnet limit and cold limit, plan A-3; each ring's applied calibration offset, E21, plan A-4) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
`e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell.
```

with:

```markdown
`e20_mixed_rings_use_the_weaker_grade` and `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature`; `e15_leaves_every_default_cell_bit_for_bit` to `e20_leaves_every_default_cell_bit_for_bit` pin that none moves a default cell. The Addendum A-4 entries (E21 onward) cite the decisions of that report's section 5 and are checked on its two bases (the registry basis, on top of what each refines, and the user basis, every correction but it): for E21 `e21_leaves_every_default_cell_bit_for_bit` (every result, Rust-only ones included), `e21_without_e20_changes_nothing`, `e21_clamps_exactly_the_negative_offsets` (the seven clamped grades; every other grade bit for bit), `e21_clamps_exactly_the_parts_rated_above_their_model_onset` (the N52 parts on both bases, the arcs on the registry basis only), `e21_changes_the_report_s_cells_to_the_report_s_figures`, `e21_typed_n42_coercivity_on_a_150_c_rating_reads_check`, `e21_binds_on_the_default_part_only_below_the_knee_threshold`, `e21_reorders_the_rings_where_the_clamp_moves_a_limit_and_ties_stay_inner`, `e21_moves_only_pairs_with_a_negative_offset_and_never_raises_a_limit` (every ordered pair of different grades and of different parts: the report's pair table, and no limit rises) and `e21_alone_moves_no_differential_case` (the double gate: every corpus case). The A-4 helpers compare every result row bit for bit (`assert_same_results`, `moved_cells`) and run each correction's checks on both bases (`assert_a4_defaults_unchanged`, `assert_no_differential_case_moves_between`).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
A correction that changes nothing at defaults (E7 to E13, E15 to E20) carries registry probes
```

with:

```markdown
A correction that changes nothing at defaults (E7 to E13, E15 onward) carries registry probes
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
The Addendum A entries name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`)
```

with:

```markdown
The Addendum A and A-4 entries (E15 onward) name every cell their probes change, downstream cells included (`addendum_entries_name_every_cell_their_probes_change`)
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
the panel shows a note only once a physics reviewer has signed it off (`note_for`); 17 of the 17 are signed off (plan A-3 Task 16
```

with:

```markdown
the panel shows a note only once a physics reviewer has signed it off (`note_for`); 16 of the 17 are signed off (the demagnetization note is a draft again until its E21 sentence is reviewed, plan A-4; plan A-3 Task 16
```

- [ ] **Step 4: Format and run the tests**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: every binary `ok`, in order: unit tests `173 passed`; `tests\assumptions.rs` 8 passed; `tests\deviations.rs` 64 passed; `tests\differential.rs` 19 passed; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 11 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 12 passed; `tests\schema.rs` 7 passed; `tests\sizing.rs` 31 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. (Read the counts as offsets from Task 0's if Task 0 recorded different ones.) The probes reproduce the report (`each_probe_shows_its_correction`), every cell a probe changes is in the 33 (`addendum_entries_name_every_cell_their_probes_change`), the pair sweep reproduces the report's table (214, 71, 0 on the registry basis; 150, 39, 0 on the user basis) and E21 alone moves no corpus case, and the drift guard and traceability pass with the new records.

- [ ] **Step 5: Lints, wasm and the differential data**

Run:

```bash
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
cd C:/Users/Cole/source/repos/lsim-mag-a4/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: each cargo line ends `Finished ...` (no warning, no error); then `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`.

- [ ] **Step 6: The release gate is red until Task 6**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test explain release_notes_are_reviewed -- --ignored 2>&1 | grep -E "not yet reviewed|test result"
```

Expected: `not yet reviewed: ["demagnetization"]` in the panic line, then `test result: FAILED. 0 passed; 1 failed`. This is the accuracy gate working: the draft is hidden by `note_for` until Task 6.

- [ ] **Step 7: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task1.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task1.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 8: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 add magcoupling-rs/src/engine/deviations.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/explain/records/demagnetization.rs magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a4 commit -F - <<'EOF'
feat(magcoupling-rs): E21, the rating-calibration offset clamped at 0

On top of E20 each ring's offset is MAX(C49 - C47, 0): a rating lowers the
onsets, never raises them (A-4 decision 1). The entry carries the report's 33
cells and four probes and an approval record of its own (Approval::AddendumA4,
section 5); C50's and C101's help is reworded (decisions 1, 3). Each ring's
applied offset is a Rust-only result the explorer reads; the demagnetization
note is redrafted and back to Draft until Task 6's review. The default design
and every grade with an offset of 0 or more are bit for bit unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 2: The per-grade magnet thermal properties (data only)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). Nothing reads the new fields yet, so no result moves. A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

Decisions 5 to 8 (A): the resolved values (Y30 CTE ⊥ 10.0e-6 and cp 700.0; Recoma 20 14.0e-6 and 370.0; Recoma 26 and Recoma 30 cp 350.0), Sm2Co17 CTE ⊥ 13.0e-6 (Arnold), bonded NdFeB with no CTE (it keeps C95) and cp 420.0 (single source, flagged). The report's landing note: "Add the per-grade values to the `Grade` record (two `Option<f64>` fields), checked against the data file in `tests/grades.rs`." Each value cites its source in `GradeSources`, as every other grade value does. The explorer's grade table gains the specific heat (Task 4's records read it); the CTE has no explained consumer, so it is not a table field.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/grades.rs` (two `Grade` fields, two `GradeSources` fields, two source constants, the 17 entries by script, module doc)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/tables.rs` (the `specific_heat_J_kgK` field and a unit test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/common/mod.rs` (`A4_DATA`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/grades.rs` (`magnet_thermal_properties_equal_the_a4_data_file`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md` (the grades layout and tests rows)

**Interfaces:**
- Consumes: `grades::GRADES`, `GradeSources`, the existing source constants `ARNOLD_RECOMA`, `ECLIPSE_FERRITE`.
- Produces (read by Tasks 3 and 4):
  - `Grade::bond_plane_cte_per_C: Option<f64>` [1/°C] and `Grade::specific_heat_J_kgK: Option<f64>` [J/(kg·K)];
  - `GradeSources::bond_plane_cte: Option<&'static str>`, `GradeSources::specific_heat: Option<&'static str>`;
  - `pub const MS_SCHRAMBERG_HF_26_24: &str`, `pub const EAM_BONDED_NEO: &str`;
  - the explain table field `table("grades", key, "specific_heat_J_kgK")` (`none` for a sintered NdFeB grade and for a missing key);
  - `tests/common/mod.rs`: `pub const A4_DATA: &str`.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/common/mod.rs`, replace:

```rust
/// The Addendum A data file (evidence: never edited), relative to the repository root.
pub const ADDENDUM_DATA: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-data.json";
```

with:

```rust
/// The Addendum A data file (evidence: never edited), relative to the repository root.
pub const ADDENDUM_DATA: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-data.json";

/// The Addendum A-4 data file (evidence: never edited), relative to the repository root.
pub const A4_DATA: &str = "docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json";
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/grades.rs`, replace:

```rust
//! The A6 grade table and the part table against the approved Addendum A data
//! file (`docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`): every
//! value and every citation, every part resolving to a grade, and the places
//! where a part's workbook value differs from its grade on purpose.

mod common;

use common::{ADDENDUM_DATA, read_json, repo_path};
```

with:

```rust
//! The A6 grade table and the part table against the approved Addendum A data
//! file (`docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`): every
//! value and every citation, every part resolving to a grade, and the places
//! where a part's workbook value differs from its grade on purpose. The magnet
//! thermal properties (bond-plane CTE and specific heat) against the Addendum A-4
//! data file (`docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json`).

mod common;

use common::{A4_DATA, ADDENDUM_DATA, read_json, repo_path};
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/grades.rs`, replace:

```rust
#[test]
fn only_ferrite_has_a_positive_beta() {
```

with:

```rust
#[test]
fn magnet_thermal_properties_equal_the_a4_data_file() {
    // Addendum A-4 decisions 5 to 8 (option A): the five non-NdFeB grades carry the data
    // file's engine literals bit for bit (bonded NdFeB has no bond-plane CTE: its proxies are
    // 4x apart, decision 7), each value citing its source; every sintered NdFeB grade carries
    // none and keeps the C95 and C138 inputs (the data file's `ndfeb_unchanged`).
    let doc = read_json(&repo_path(A4_DATA));
    let thermal = doc["grades"].as_object().expect("a grades object");
    let ndfeb: Vec<&str> = doc["ndfeb_unchanged"]["grades"]
        .as_array()
        .expect("a grade list")
        .iter()
        .map(|g| g.as_str().expect("a grade id"))
        .collect();
    assert_eq!(
        thermal.len() + ndfeb.len(),
        GRADES.len(),
        "one list per grade"
    );
    for g in &GRADES {
        let id = g.id;
        let Some(j) = thermal.get(id) else {
            assert!(ndfeb.contains(&id), "{id} is in neither list");
            assert_eq!(g.family, GradeFamily::NdFeB, "{id}");
            assert_eq!(
                (g.bond_plane_cte_per_C, g.specific_heat_J_kgK),
                (None, None),
                "{id}"
            );
            assert_eq!(
                (g.sources.bond_plane_cte, g.sources.specific_heat),
                (None, None),
                "{id}"
            );
            continue;
        };
        assert_ne!(g.family, GradeFamily::NdFeB, "{id}");
        let lit = &j["engine_literals"];
        assert_eq!(
            g.bond_plane_cte_per_C,
            opt_f(&lit["bond_plane_cte_per_C"]),
            "{id} CTE"
        );
        assert_eq!(
            g.specific_heat_J_kgK,
            opt_f(&lit["specific_heat_J_kgK"]),
            "{id} specific heat"
        );
        // A value cites the data file's source for it; no value, no citation.
        let cite = |value: Option<f64>, field: &str| {
            value.and(opt_s(&j["fields"][field]["value_source_url"]))
        };
        assert_eq!(
            g.sources.bond_plane_cte,
            cite(g.bond_plane_cte_per_C, "cte_perp_1e-6_per_K"),
            "{id} CTE source"
        );
        assert_eq!(
            g.sources.specific_heat,
            cite(g.specific_heat_J_kgK, "cp_J_kgK"),
            "{id} specific heat source"
        );
    }
    // The approved values (A-4 report section 3.2), as typed.
    let pick = |id: &str| grade(id).map(|g| (g.bond_plane_cte_per_C, g.specific_heat_J_kgK));
    assert_eq!(pick("Y30"), Some((Some(10.0e-6), Some(700.0))));
    assert_eq!(pick("SmCo_1_5_20"), Some((Some(14.0e-6), Some(370.0))));
    assert_eq!(pick("SmCo_2_17_26"), Some((Some(13.0e-6), Some(350.0))));
    assert_eq!(pick("SmCo_2_17_30"), Some((Some(13.0e-6), Some(350.0))));
    assert_eq!(pick("Bonded_NdFeB_BCN19"), Some((None, Some(420.0))));
}

#[test]
fn only_ferrite_has_a_positive_beta() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
        assert_eq!(
            lookup("adhesives", &Value::Int(2), "cure_C").unwrap(),
            Some(Value::Num(120.0))
        );
    }
}
```

with:

```rust
        assert_eq!(
            lookup("adhesives", &Value::Int(2), "cure_C").unwrap(),
            Some(Value::Num(120.0))
        );
    }

    #[test]
    fn a_grade_without_a_specific_heat_reads_none() {
        // Addendum A-4: ferrite, SmCo and bonded NdFeB carry a specific heat (E23); a sintered
        // NdFeB grade has none (it keeps the C138 input), as does a key the table lacks.
        let cp = |id: &str| lookup("grades", &Value::Text(id.into()), "specific_heat_J_kgK");
        assert_eq!(cp("Y30").unwrap(), Some(Value::Num(700.0)));
        assert_eq!(cp("N42SH").unwrap(), Some(Value::None));
        assert_eq!(cp("").unwrap(), None);
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test grades 2>&1 | grep -E "^error" | sort | uniq -c
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --lib a_grade_without 2>&1 | grep -E "panicked|Err"
```

Expected: the first command,

```text
1 error: could not compile `magcoupling-rs` (test "grades") due to 12 previous errors
4 error[E0609]: no field `bond_plane_cte_per_C` on type `&Grade`
2 error[E0609]: no field `bond_plane_cte` on type `GradeSources`
4 error[E0609]: no field `specific_heat_J_kgK` on type `&Grade`
2 error[E0609]: no field `specific_heat` on type `GradeSources`
```

and the second, the panic of the new table test: `called `Result::unwrap()` on an `Err` value: "table grades has no field specific_heat_J_kgK"`.

- [ ] **Step 3: Add the fields, the values and the table field**

The struct fields and constants first, then a script that adds the four new lines to each of the 17 entries (it asserts it finds each grade's density lines), then the table field and the README.

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/grades.rs`, replace:

```rust
//! sintered NdFeB, whose grade values equal them) keeps the calculator's single alpha
//! (Calibration!C22) and the NdFeB density.
```

with:

```rust
//! sintered NdFeB, whose grade values equal them) keeps the calculator's single alpha
//! (Calibration!C22) and the NdFeB density.
//!
//! The magnet thermal properties (Addendum A-4, decisions 5 to 8, option A; data file
//! `docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json`, key `"grades"`): the
//! expansion coefficient in the bond plane, across the magnetization, and the specific
//! heat of the ferrite, SmCo and bonded NdFeB grades, each cited in [`GradeSources`] and
//! compared with the data file by `tests/grades.rs`. A sintered NdFeB grade has neither and
//! keeps the C95 and C138 inputs; bonded NdFeB has no CTE (decision 7) and its specific heat
//! is single-source (decision 8). Corrections E22 (the inner ring's CTE) and E23 (each
//! ring's specific heat) read them.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/grades.rs`, replace:

```rust
pub const ALLIANCE_BONDED_NEO: &str =
    "https://allianceorg.com/magnetic-materials/bonded-magnets/compression-bonded-neo/";
```

with:

```rust
pub const ALLIANCE_BONDED_NEO: &str =
    "https://allianceorg.com/magnetic-materials/bonded-magnets/compression-bonded-neo/";
/// MS-Schramberg HF 26/24 strontium ferrite, the Y30 proxy grade nearest by Br: "spec. heat
/// capacity approx. 700 J/(kg·K)" (Addendum A-4 decision 5).
pub const MS_SCHRAMBERG_HF_26_24: &str = "https://magnete.de/fileadmin/user_upload/Magnetische_Kenndaten/Gesinterte_Hartferritmagnete/HF_26-24_Sr_E.pdf";
/// EAM compression-bonded Neo physical properties, the BCN-19 proxy vendor: 0.42 W·s/(g·°C),
/// a single source (Addendum A-4 decision 8).
pub const EAM_BONDED_NEO: &str =
    "https://eamagnetics.com/wp-content/uploads/2021/08/EAM-ComprBond-Neo-PhysProp-1.pdf";
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/grades.rs`, replace:

```rust
    pub tmax: &'static str,
    pub density: &'static str,
}
```

with:

```rust
    pub tmax: &'static str,
    pub density: &'static str,
    /// The bond-plane CTE's source (Addendum A-4); `None` where the grade has no value.
    pub bond_plane_cte: Option<&'static str>,
    /// The specific heat's source (Addendum A-4); `None` where the grade has no value.
    pub specific_heat: Option<&'static str>,
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/grades.rs`, replace:

```rust
    /// Density [g/mm³] (read in the grade mode, decision A2-7).
    pub density_g_mm3: f64,
    pub sources: GradeSources,
```

with:

```rust
    /// Density [g/mm³] (read in the grade mode, decision A2-7).
    pub density_g_mm3: f64,
    /// Expansion coefficient in the bond plane, across the magnetization [1/°C] (Addendum
    /// A-4 decisions 5 to 7; correction E22 reads the inner ring's): ferrite and SmCo.
    /// `None` keeps the C95 input (sintered NdFeB; bonded NdFeB, decision 7).
    pub bond_plane_cte_per_C: Option<f64>,
    /// Specific heat [J/(kg·K)] (decisions 5 and 8; correction E23 reads each ring's):
    /// ferrite, SmCo and bonded NdFeB. `None` keeps the C138 input (sintered NdFeB).
    pub specific_heat_J_kgK: Option<f64>,
    pub sources: GradeSources,
```

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import pathlib
import re

p = pathlib.Path("C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/grades.rs")
s = p.read_text(encoding="utf-8")
# grade id: (bond-plane CTE, specific heat, CTE source, specific-heat source) as Rust text;
# every other grade (sintered NdFeB) gets None for all four.
THERMAL = {
    "SmCo_2_17_26": ("Some(13.0e-6)", "Some(350.0)", "Some(ARNOLD_RECOMA)", "Some(ARNOLD_RECOMA)"),
    "SmCo_2_17_30": ("Some(13.0e-6)", "Some(350.0)", "Some(ARNOLD_RECOMA)", "Some(ARNOLD_RECOMA)"),
    "SmCo_1_5_20": ("Some(14.0e-6)", "Some(370.0)", "Some(ARNOLD_RECOMA)", "Some(ARNOLD_RECOMA)"),
    "Y30": ("Some(10.0e-6)", "Some(700.0)", "Some(ECLIPSE_FERRITE)", "Some(MS_SCHRAMBERG_HF_26_24)"),
    "Bonded_NdFeB_BCN19": ("None", "Some(420.0)", "None", "Some(EAM_BONDED_NEO)"),
}
head = '    Grade {\n        id: "'
blocks = s.split(head)
assert len(blocks) == 18, len(blocks)
out = [blocks[0]]
for b in blocks[1:]:
    gid = b.split('"', 1)[0]
    cte, cp, cte_src, cp_src = THERMAL.get(gid, ("None", "None", "None", "None"))
    b, n = re.subn(
        r"(\n        density_g_mm3: [0-9.]+,\n)",
        lambda m: m.group(1) + f"        bond_plane_cte_per_C: {cte},\n        specific_heat_J_kgK: {cp},\n",
        b,
        count=1,
    )
    assert n == 1, gid
    b, n = re.subn(
        r"(\n            density: [A-Z_]+,\n)",
        lambda m: m.group(1) + f"            bond_plane_cte: {cte_src},\n            specific_heat: {cp_src},\n",
        b,
        count=1,
    )
    assert n == 1, gid
    out.append(b)
p.write_text(head.join(out), encoding="utf-8", newline="\n")
print("added the thermal fields to", len(blocks) - 1, "grades")
EOF
```

Expected: prints `added the thermal fields to 17 grades`.

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
    TableField { table: "grades", field: "density_g_mm3", symbol: "ρ_{grade}", unit: "g/mm³", label: "Grade density" },
```

with:

```rust
    TableField { table: "grades", field: "density_g_mm3", symbol: "ρ_{grade}", unit: "g/mm³", label: "Grade density" },
    TableField { table: "grades", field: "specific_heat_J_kgK", symbol: "c_{grade}", unit: "J/(kg·K)", label: "Grade specific heat (Addendum A-4; none for sintered NdFeB)" },
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
            "density_g_mm3" => num(g.density_g_mm3),
            _ => unreachable!("checked against TABLE_FIELDS"),
```

with:

```rust
            "density_g_mm3" => num(g.density_g_mm3),
            "specific_heat_J_kgK" => g.specific_heat_J_kgK.map_or(Value::None, num),
            _ => unreachable!("checked against TABLE_FIELDS"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup; a grade-mode ring reads its alpha and density (decision A2-7) |
```

with:

```markdown
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density; Addendum A-4: the bond-plane CTE of ferrite and SmCo and the specific heat of ferrite, SmCo and bonded NdFeB, `None` for sintered NdFeB), each value cited; `grade(id)` lookup; a grade-mode ring reads its alpha and density (decision A2-7) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
the magnets' mass and the bond block's mass (`mixed_rings_each_take_their_own_alpha_and_density_either_way_round`). |
```

with:

```markdown
the magnets' mass and the bond block's mass (`mixed_rings_each_take_their_own_alpha_and_density_either_way_round`). The Addendum A-4 magnet thermal properties equal `docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json` value for value and citation for citation: the bond-plane CTE of ferrite and SmCo, the specific heat of ferrite, SmCo and bonded NdFeB, none for sintered NdFeB (`magnet_thermal_properties_equal_the_a4_data_file`). |
```

- [ ] **Step 4: Format and run the tests**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: every binary `ok`, in order: unit tests `174 passed`; `tests\assumptions.rs` 8 passed; `tests\deviations.rs` 64 passed; `tests\differential.rs` 19 passed; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 12 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 12 passed; `tests\schema.rs` 7 passed; `tests\sizing.rs` 31 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. (Read the counts as offsets from Task 0's if Task 0 recorded different ones.)

- [ ] **Step 5: Lints, wasm and the differential data**

Run:

```bash
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
cd C:/Users/Cole/source/repos/lsim-mag-a4/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: each cargo line ends `Finished ...` (no warning, no error); then `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`.

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task2.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 add magcoupling-rs/src/engine/grades.rs magcoupling-rs/src/engine/explain/tables.rs magcoupling-rs/tests/common/mod.rs magcoupling-rs/tests/grades.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a4 commit -F - <<'EOF'
feat(magcoupling-rs): per-grade magnet CTE and specific heat in the grade table

The bond-plane CTE of ferrite and SmCo and the specific heat of ferrite, SmCo
and bonded NdFeB (A-4 decisions 5 to 8), each cited and checked against the
A-4 data file; sintered NdFeB has neither and keeps C95 and C138. The explain
grade table reads the specific heat. Nothing reads the fields yet: no result
moves.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 3: E22, the inner ring's grade CTE in the bond plane and the size of the shear

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5): a physics correction to the Volkersen screen and warning rule 6. Physics reviewer: the A-4 report sections 3.1, 4.3, 4.5, 4.7 (IN1) and decision 9; the M1 audit row E1.

Decisions 4, 5, 6, 7, 9 and 10 (A). The magnet CTE in the bond plane is the inner ring's grade value where the grade has one (`temperature::bond_plane_cte_per_C`, through a private `grade_value_or` that Task 4's specific heat reuses), else C95; the Volkersen screen and warning rule 6 both read it (the warning through the new Rust-only `temperature.mismatch.magnet_cte_per_C`, one source). Only the inner ring: the hub bonds it, and the cup bond is not screened. A grade that expands more than its hub makes the shear negative, so under E22 C106 compares |C104| with the lap shear and C202 |C201| with the endurance limit (decision 9); without E22 the workbook's signed test stays. Without a grade a negative mismatch needs the default back iron's C17 near its slider floor and below C95 (every library hub material is at 9.0e-6 /°C or more, C95 at most 6e-6); the size comparison then applies to sintered NdFeB too, as decision A4-4 recommends, and `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too` pins that corner. The report checked E22's neutrality on the differential corpus before decision 9 existed, so `e22_moves_no_differential_case` re-proves it with the size comparison. C95 keeps its label and range; its help is reworded and registered. The `a5.cte_mismatch_with_magnets` note goes back to Draft (decision A4-2). The mismatch screen has no explained record (it is outside decision 31's scope), so no record changes.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs` (module doc, `DeviationId::E22`, the E22 entry)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs` (module doc, `grade_value_or` and `bond_plane_cte_per_C`, the size comparison, C95 help, `MismatchResults::magnet_cte_per_C`, a unit test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/api.rs` (warning rule 6 reads the CTE in effect)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/warnings.rs` (rule 6's help and input doc)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs` (the `a5.cte_mismatch_with_magnets` note)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs` (helpers and six E22 tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/data/input_schema.json` (blessed: C95's help)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `Grade::bond_plane_cte_per_C` (Task 2); `a4_bases`, `grade_rings`, `Overrides`, `results_with`, `assert_same_results`, `moved_cells`, `assert_a4_defaults_unchanged`, `assert_no_differential_case_moves_between` (Task 1).
- Produces (read by Tasks 4 and 7):
  - `pub fn bond_plane_cte_per_C(input_per_C: f64, inner: Option<&Grade>, dev: Deviations) -> f64` (temperature.rs);
  - `fn grade_value_or(input: f64, value: Option<f64>, dev: Deviations, id: DeviationId) -> f64` (temperature.rs, private: the grade's value where it has one and `id` is on, else the input);
  - Rust-only `MismatchResults::magnet_cte_per_C: f64` (`temperature.mismatch.magnet_cte_per_C`); `DeviationId::E22`;
  - in `tests/deviations.rs`: `fn grade_placements(grade: &str) -> [(&'static str, Overrides); 3]` (`"both rings"`, `"inner only"`, `"outer only"`); `const HUBS: [(&str, i64); 2]` (the steel circuit and no back iron); `fn assert_no_differential_case_moves(id: DeviationId)` (Task 1's corpus check on the workbook basis, NONE against only `id`, and the user basis, every correction but `id` against all); `type ScreenRow`.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
        DeviationId::E20,
        DeviationId::E21,
    ]
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
```

with:

```rust
        DeviationId::E20,
        DeviationId::E21,
        DeviationId::E22,
    ]
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
/// The placements the A-4 harness ran for each grade (its section 1): the grade on both
/// rings, on the inner ring only and on the outer ring only (beside the default part B842SH).
fn grade_placements(grade: &str) -> [(&'static str, Overrides); 3] {
    let text = |s: &str| Value::Text(s.into());
    [
        ("both rings", grade_rings(grade, grade).to_vec()),
        (
            "inner only",
            vec![
                ("coupling.magnets.part_inner", text("")),
                ("coupling.magnets.grade_inner", text(grade)),
            ],
        ),
        (
            "outer only",
            vec![
                ("coupling.magnets.part_outer", text("")),
                ("coupling.magnets.grade_outer", text(grade)),
            ],
        ),
    ]
}

/// The A-4 harness's two hubs: the steel circuit (the default) and no back iron, which makes
/// the hub aluminium (the mass model; E18's screen and warning rule 6 follow it).
const HUBS: [(&str, i64); 2] = [("steel hub", 1), ("aluminium hub", 0)];

/// No differential case moves under correction `id`, from the workbook (NONE against only
/// `id`) nor for users (every correction but `id` against every correction).
fn assert_no_differential_case_moves(id: DeviationId) {
    let bases = [
        (Deviations::NONE, Deviations::only(id)),
        (Deviations::ALL.without(id), Deviations::ALL),
    ];
    assert_no_differential_case_moves_between(&bases, &id.to_string());
}

#[test]
fn e22_leaves_every_default_cell_bit_for_bit() {
    // The default part is sintered NdFeB: C95 stays the magnet CTE, and the shear is positive.
    assert_a4_defaults_unchanged(DeviationId::E22);
}

#[test]
fn e22_moves_only_a_ferrite_or_smco_inner_ring() {
    // A-4 report sections 4.1, 4.2 and 4.5: E22 reads the inner ring's grade CTE (the hub
    // bonds the inner ring), so a grade with one (ferrite, SmCo) moves the bond screen and
    // warning rule 6 when it is the inner ring, and nothing moves anywhere else: an outer-only
    // grade, sintered and bonded NdFeB, every library part, on both hubs and both bases, at
    // the default sliders (a slider corner moves sintered NdFeB too, decision A4-4:
    // `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too`). Wired as approved:
    // E22 equals E22 off with C95 set to the grade's value, on every row (the readings agree
    // because wherever the shear is negative here its size stays below both thresholds,
    // report IN1).
    use magcoupling::engine::grades::GRADES;
    use magcoupling::engine::library::MAGNET_LIBRARY;
    let mut configs: Vec<(String, Overrides, Option<f64>)> = Vec::new();
    for g in &GRADES {
        for (placement, overrides) in grade_placements(g.id) {
            let inner_cte = (placement != "outer only")
                .then_some(g.bond_plane_cte_per_C)
                .flatten();
            configs.push((format!("{} {placement}", g.id), overrides, inner_cte));
        }
    }
    for spec in &MAGNET_LIBRARY {
        configs.push((
            format!("{} both rings", spec.part),
            both_rings(spec.part).to_vec(),
            None,
        ));
    }
    let mut moved = 0;
    for (hub, backiron) in HUBS {
        for (basis, before, after) in a4_bases(DeviationId::E22) {
            for (label, overrides, inner_cte) in &configs {
                let mut o = overrides.clone();
                o.push(("coupling.backiron", Value::Int(backiron)));
                let what = format!("{label}, {hub}, {basis}");
                let (b, a) = (results_with(&o, before), results_with(&o, after));
                let Some(cte) = *inner_cte else {
                    assert_same_results(&b, &a, &what);
                    continue;
                };
                moved += 1;
                assert_eq!(a.temperature.mismatch.magnet_cte_per_C, cte, "{what}");
                assert_ne!(
                    b.temperature.mismatch.peak_shear_current_MPa,
                    a.temperature.mismatch.peak_shear_current_MPa,
                    "{what}"
                );
                let mut wired = inputs_with(&o, before);
                wired.temperature.mismatch.ndfeb_cte_per_C = cte;
                assert_same_results(
                    &compute_all_with(&wired, before),
                    &a,
                    &format!("{what}, wiring"),
                );
            }
        }
    }
    // Y30, Recoma 20, 26 and 30, on both rings and inside; two hubs; two bases.
    assert_eq!(moved, 4 * 2 * 2 * 2);
}

/// A row of the A-4 report's table 4.3 (the grade on both rings): (grade, basis, hub code,
/// C104 and C201 as [before, after], and as [before, after] whether C106 and C202 read
/// "Above" and warning rule 6 fires).
type ScreenRow = (
    &'static str,
    &'static str,
    i64,
    [f64; 2],
    [f64; 2],
    [bool; 2],
    [bool; 2],
    [bool; 2],
);

#[test]
fn e22_reproduces_the_report_s_bond_screen_table() {
    // A-4 report section 4.3 (bonded NdFeB keeps C95, decision 7: no row moves).
    const T: bool = true;
    const F: bool = false;
    let rows: [ScreenRow; 16] = [
        (
            "Y30",
            "registry",
            1,
            [64.07, 11.25],
            [11.37, 1.995],
            [T, F],
            [T, F],
            [F, F],
        ),
        (
            "Y30",
            "registry",
            0,
            [64.07, 11.25],
            [11.37, 1.995],
            [T, F],
            [T, F],
            [T, F],
        ),
        (
            "Y30",
            "user",
            1,
            [16.11, 2.829],
            [2.563, 0.4500],
            [T, F],
            [F, F],
            [F, F],
        ),
        (
            "Y30",
            "user",
            0,
            [28.63, 15.96],
            [4.655, 2.595],
            [T, T],
            [T, F],
            [T, F],
        ),
        (
            "SmCo_1_5_20",
            "registry",
            1,
            [64.07, -8.315],
            [11.37, -1.475],
            [T, F],
            [T, F],
            [F, F],
        ),
        (
            "SmCo_1_5_20",
            "registry",
            0,
            [64.07, -8.315],
            [11.37, -1.475],
            [T, F],
            [T, F],
            [T, F],
        ),
        (
            "SmCo_1_5_20",
            "user",
            1,
            [13.11, -1.701],
            [2.563, -0.3326],
            [F, F],
            [F, F],
            [F, F],
        ),
        (
            "SmCo_1_5_20",
            "user",
            0,
            [23.30, 9.166],
            [4.655, 1.832],
            [T, F],
            [T, F],
            [T, F],
        ),
        (
            "SmCo_2_17_26",
            "registry",
            1,
            [64.07, -3.424],
            [11.37, -0.6073],
            [T, F],
            [T, F],
            [F, F],
        ),
        (
            "SmCo_2_17_26",
            "registry",
            0,
            [64.07, -3.424],
            [11.37, -0.6073],
            [T, F],
            [T, F],
            [T, F],
        ),
        (
            "SmCo_2_17_26",
            "user",
            1,
            [13.11, -0.7004],
            [2.563, -0.1370],
            [F, F],
            [F, F],
            [F, F],
        ),
        (
            "SmCo_2_17_26",
            "user",
            0,
            [23.30, 10.12],
            [4.655, 2.022],
            [T, F],
            [T, F],
            [T, F],
        ),
        (
            "SmCo_2_17_30",
            "registry",
            1,
            [64.07, -3.424],
            [11.37, -0.6073],
            [T, F],
            [T, F],
            [F, F],
        ),
        (
            "SmCo_2_17_30",
            "registry",
            0,
            [64.07, -3.424],
            [11.37, -0.6073],
            [T, F],
            [T, F],
            [T, F],
        ),
        (
            "SmCo_2_17_30",
            "user",
            1,
            [10.19, -0.5446],
            [2.563, -0.1370],
            [F, F],
            [F, F],
            [F, F],
        ),
        (
            "SmCo_2_17_30",
            "user",
            0,
            [18.11, 7.869],
            [4.655, 2.022],
            [T, F],
            [T, F],
            [T, F],
        ),
    ];
    let bases = a4_bases(DeviationId::E22);
    for (grade, basis, backiron, c104, c201, above, daily_above, fires) in rows {
        let (_, before, after) = *bases.iter().find(|(b, _, _)| *b == basis).expect("a basis");
        let mut o = grade_rings(grade, grade).to_vec();
        o.push(("coupling.backiron", Value::Int(backiron)));
        let what = format!("{grade}, {basis}, backiron {backiron}");
        for (side, dev) in [(0, before), (1, after)] {
            let r = results_with(&o, dev);
            let (m, life) = (&r.temperature.mismatch, &r.temperature.adhesive_life);
            assert_sig4(&what, &Value::Num(m.peak_shear_current_MPa), c104[side]);
            assert_sig4(&what, &Value::Num(life.daily_peak_shear_MPa), c201[side]);
            assert_eq!(
                m.reading.starts_with("Above"),
                above[side],
                "{what} C106 {side}"
            );
            assert_eq!(
                life.daily_screen.starts_with("Above"),
                daily_above[side],
                "{what} C202 {side}"
            );
            assert_eq!(
                !r.warnings.cte_mismatch_with_magnets.is_empty(),
                fires[side],
                "{what} rule 6 {side}"
            );
        }
    }
}

#[test]
fn e22_compares_the_size_of_a_negative_shear() {
    // Decision 9: a grade that expands more than the hub shears the bond the other way; the
    // screens compare the shear's size and keep the signed value on display. Recoma 20
    // (14e-6 /C) on a 416 stainless back iron (10.08e-6 /C) on the registry basis (the
    // workbook's 0.55 GPa adhesive): C104 is below -15 MPa, AA 326's lap shear, so C106 reads
    // "Above"; a signed comparison would read "Below".
    let mut o = grade_rings("SmCo_1_5_20", "SmCo_1_5_20").to_vec();
    o.push(("materials.parts.back_iron", Value::Int(4)));
    let [(_, before, after), _] = a4_bases(DeviationId::E22);
    let r = results_with(&o, after);
    let (m, life) = (&r.temperature.mismatch, &r.temperature.adhesive_life);
    assert!(
        m.peak_shear_current_MPa < -15.0,
        "{}",
        m.peak_shear_current_MPa
    );
    assert_eq!(m.reading, "Above the lap-shear strength at the block ends");
    assert!(
        life.daily_peak_shear_MPa < -3.0,
        "{}",
        life.daily_peak_shear_MPa
    );
    assert_eq!(
        life.daily_screen,
        "Above the fatigue endurance: qualify by thermal cycling"
    );
    // Without E22 the same design shears the positive way (C95, -0.8e-6 /C).
    assert!(
        results_with(&o, before)
            .temperature
            .mismatch
            .peak_shear_current_MPa
            > 15.0
    );
}

#[test]
fn e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too() {
    // Decision 9 is gated on E22 alone (plan decision A4-4), so it also reads a sintered
    // NdFeB design whose mismatch a slider makes negative: the default back iron's CTE input
    // C17 at its 5e-6 /C floor, below C95 at its 6e-6 /C ceiling (every library hub material
    // is at 9.0e-6 /C or more). With the stiffest adhesive (C96 3.0 GPa) on 0.01 mm bondlines,
    // C104 is -22.6 MPa (registry basis) or -22.8 MPa (user basis): E22 reads it "Above" the
    // 15 MPa lap shear (C106) and the endurance limit (C202), where the workbook's signed
    // test reads "Below". C95 stays the magnet CTE, and no other cell moves.
    let o = [
        ("materials.steel.cte_per_C", Value::Num(5e-6)),
        ("temperature.mismatch.ndfeb_cte_per_C", Value::Num(6e-6)),
        (
            "temperature.mismatch.adhesive_shear_modulus_GPa",
            Value::Num(3.0),
        ),
        (
            "temperature.mismatch.recommended_bondline_mm",
            Value::Num(0.01),
        ),
        ("metal.bond_inner_mm", Value::Num(0.01)),
    ];
    for (basis, before, after) in a4_bases(DeviationId::E22) {
        let (b, a) = (results_with(&o, before), results_with(&o, after));
        let (mb, ma) = (&b.temperature.mismatch, &a.temperature.mismatch);
        assert_eq!(ma.magnet_cte_per_C, 6e-6, "{basis}");
        assert!(
            ma.peak_shear_current_MPa < -15.0,
            "{basis}: {}",
            ma.peak_shear_current_MPa
        );
        assert_eq!(mb.reading, "Below the lap-shear strength", "{basis}");
        assert_eq!(
            ma.reading, "Above the lap-shear strength at the block ends",
            "{basis}"
        );
        assert_eq!(
            b.temperature.adhesive_life.daily_screen, "Below the fatigue endurance",
            "{basis}"
        );
        assert_eq!(
            a.temperature.adhesive_life.daily_screen,
            "Above the fatigue endurance: qualify by thermal cycling",
            "{basis}"
        );
        assert_eq!(
            moved_cells(&b, &a),
            ["Temperature design!C106", "Temperature design!C202"],
            "{basis}"
        );
    }
}

#[test]
fn e22_moves_no_differential_case() {
    // A-4 report section 4.2, rechecked here with decision 9's size comparison: no case of
    // the corpus picks a grade, and every shear in the corpus is positive.
    assert_no_differential_case_moves(DeviationId::E22);
}

#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

In the unit tests of `temperature.rs`:

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    #[test]
    fn each_ring_shows_its_own_block() {
```

with:

```rust
    #[test]
    fn e22_reads_the_inner_ring_s_grade_cte_and_compares_the_shear_s_size() {
        // Addendum A-4 decisions 5 to 7 and 9. A SmCo inner ring (13e-6 /C) on a steel hub
        // (12.3e-6) turns the mismatch, and the shear, negative; a ferrite or SmCo outer ring,
        // a bonded NdFeB inner ring (no grade CTE, decision 7) and E22 off keep C95.
        let e22 = Deviations::only(DeviationId::E22);
        let ti = TemperatureInputs::default();
        let c95 = ti.mismatch.ndfeb_cte_per_C;
        let smco = crate::engine::grades::grade("SmCo_2_17_26");
        let mut k = links();
        k.inner_grade = smco;
        let r = compute(&ti, &k, e22);
        assert_eq!(r.mismatch.magnet_cte_per_C, 13.0e-6);
        assert!(r.mismatch.peak_shear_current_MPa < 0.0);
        assert_eq!(run(&ti, &k).mismatch.magnet_cte_per_C, c95, "E22 off");
        let mut outer = links();
        outer.outer_grade = smco;
        assert_eq!(compute(&ti, &outer, e22), run(&ti, &outer), "outer ring");
        let mut bonded = links();
        bonded.inner_grade = crate::engine::grades::grade("Bonded_NdFeB_BCN19");
        assert_eq!(
            compute(&ti, &bonded, e22),
            run(&ti, &bonded),
            "bonded NdFeB"
        );
        // The size is compared: with the workbook's stiffer adhesive (0.55 GPa) and a
        // hypothetical low-expansion hub (2e-6 /C), the negative shear exceeds AA 326's 15 MPa
        // lap shear, and the reading says so.
        let mut stiff = TemperatureInputs::default();
        stiff.mismatch.adhesive_shear_modulus_GPa = 0.55;
        k.steel_cte = 2.0e-6;
        let r = compute(&stiff, &k, e22);
        assert!(
            r.mismatch.peak_shear_current_MPa < -15.0,
            "{}",
            r.mismatch.peak_shear_current_MPa
        );
        assert_eq!(
            r.mismatch.reading,
            "Above the lap-shear strength at the block ends"
        );
    }

    #[test]
    fn each_ring_shows_its_own_block() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test deviations 2>&1 | grep -E "^error" | sort | uniq -c
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --lib 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: compile errors only, the first command:

```text
1 error: could not compile `magcoupling-rs` (test "deviations") due to 9 previous errors
7 error[E0599]: no variant or associated item named `E22` found for enum `DeviationId` in the current scope
1 error[E0609]: no field `magnet_cte_per_C` on type `&MismatchResults`
1 error[E0609]: no field `magnet_cte_per_C` on type `MismatchResults`
```

and the second:

```text
1 error: could not compile `magcoupling-rs` (lib test) due to 3 previous errors
1 error[E0599]: no variant or associated item named `E22` found for enum `deviations::DeviationId` in the current scope
2 error[E0609]: no field `magnet_cte_per_C` on type `MismatchResults`
```

- [ ] **Step 3: Implement E22**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! E21 (decision 1: the rating-calibration offset clamped at 0), approved on
//! 2026-10-01. Everything else stays workbook-exact. This registry is the one
```

with:

```rust
//! E21 (decision 1: the rating-calibration offset clamped at 0) and E22
//! (decisions 4 to 7, 9, 10: the inner ring's grade CTE in the bond plane),
//! approved on 2026-10-01. Everything else stays workbook-exact. This registry is the one
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    E20,
    E21,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 21] = [
```

with:

```rust
    E20,
    E21,
    E22,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 22] = [
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        DeviationId::E20,
        DeviationId::E21,
    ];
```

with:

```rust
        DeviationId::E20,
        DeviationId::E21,
        DeviationId::E22,
    ];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
                    CellChange {
                        cell: "Temperature design!C24",
                        workbook: Literal::Num(149.91967217479288),
                        corrected: Literal::Num(205.01249772124206),
                    },
                ],
            },
        ],
    },
];
```

with:

```rust
                    CellChange {
                        cell: "Temperature design!C24",
                        workbook: Literal::Num(149.91967217479288),
                        corrected: Literal::Num(205.01249772124206),
                    },
                ],
            },
        ],
    },
    Deviation {
        id: DeviationId::E22,
        title: "The adhesive mismatch screen and the expansion-mismatch warning use sintered NdFeB's bond-plane expansion for every magnet grade",
        class: DeviationClass::Engine,
        approval: Approval::AddendumA4 {
            decisions: &[4, 5, 6, 7, 9, 10],
        },
        depends_on: &[],
        status: DeviationStatus::Applied,
        cells: &[
            "Temperature design!C104",
            "Temperature design!C105",
            "Temperature design!C106",
            "Temperature design!C201",
            "Temperature design!C202",
        ],
        corrected_formula: "The magnet expansion in the bond plane (across the magnetization) is the inner ring's grade \
            value: Y30 10.0e-6, Recoma 20 14.0e-6, Recoma 26 and Recoma 30 13.0e-6 /C (decisions 5, 6); C95 (-0.8e-6) \
            stays for sintered NdFeB (every library part), a manual magnet without a grade and bonded NdFeB (decision 7). \
            Both consumers read it: the Volkersen screen (delta alpha = hub CTE - magnet CTE: C104, C105, C201) and \
            warning rule 6. Only the inner ring is read: the hub bonds it, and the cup bond is not screened. A grade that \
            expands more than the hub makes the shear negative, so C106 compares |C104| with the lap shear and C202 \
            compares |C201| with the endurance limit, the signed values kept on display (decision 9); so does C95 set \
            above the default back iron's C17, a slider corner where the size comparison reads sintered NdFeB too (plan \
            decision A4-4). C95 keeps its label and range; its help and the Rust-only \
            temperature.mismatch.magnet_cte_per_C show the value in effect (decision 10).",
        workbook_input_defaults: &[],
        workbook_help: &[(
            "temperature.mismatch.ndfeb_cte_per_C",
            "Across the magnetization.",
        )],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[
            Probe {
                label: "Recoma 26 (13.0e-6 /C) for manual dimensions on both rings, steel hub (A-4 report probe E22-1): \
                    the mismatch turns negative and the screen reads Below",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("")),
                    ("coupling.magnets.part_outer", Literal::Text("")),
                    (
                        "coupling.magnets.grade_inner",
                        Literal::Text("SmCo_2_17_26"),
                    ),
                    (
                        "coupling.magnets.grade_outer",
                        Literal::Text("SmCo_2_17_26"),
                    ),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C104",
                        workbook: Literal::Num(64.07367112080888),
                        corrected: Literal::Num(-3.423783953020314),
                    },
                    CellChange {
                        cell: "Temperature design!C106",
                        workbook: Literal::Text("Above the lap-shear strength at the block ends"),
                        corrected: Literal::Text("Below the lap-shear strength"),
                    },
                    CellChange {
                        cell: "Temperature design!C201",
                        workbook: Literal::Num(11.365659614702166),
                        corrected: Literal::Num(-0.6073253229230151),
                    },
                ],
            },
            Probe {
                label: "Y30 (10.0e-6 /C) for manual dimensions on both rings, steel hub (A-4 report probe E22-2)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("")),
                    ("coupling.magnets.part_outer", Literal::Text("")),
                    ("coupling.magnets.grade_inner", Literal::Text("Y30")),
                    ("coupling.magnets.grade_outer", Literal::Text("Y30")),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C104",
                        workbook: Literal::Num(64.07367112080888),
                        corrected: Literal::Num(11.249575845638201),
                    },
                    CellChange {
                        cell: "Temperature design!C202",
                        workbook: Literal::Text(
                            "Above the fatigue endurance: qualify by thermal cycling",
                        ),
                        corrected: Literal::Text("Below the fatigue endurance"),
                    },
                ],
            },
        ],
    },
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! beta is limited on the cold side, [`cold_onset_C`]) and E21 (on top of E20:
//! each ring's rating-calibration offset C50 is clamped at 0, so a rating never
//! raises the onsets above the model's own).
```

with:

```rust
//! beta is limited on the cold side, [`cold_onset_C`]), E21 (on top of E20:
//! each ring's rating-calibration offset C50 is clamped at 0, so a rating never
//! raises the onsets above the model's own) and E22 (the mismatch screen and
//! warning rule 6 read the inner ring's grade CTE, [`bond_plane_cte_per_C`],
//! and the screens compare the size of the shear).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            ndfeb_cte_per_C: f64 = -0.8e-6 => param("1/°C", "NdFeB expansion in the bond plane",
                "Across the magnetization.", "Temperature design!C95")
```

with:

```rust
            ndfeb_cte_per_C: f64 = -0.8e-6 => param("1/°C", "NdFeB expansion in the bond plane",
                "Across the magnetization. Sintered NdFeB: every library part, a manual magnet without a grade, and bonded NdFeB. Correction E22: a ferrite or SmCo inner ring uses its grade's value instead (temperature.mismatch.magnet_cte_per_C shows the value in effect).", "Temperature design!C95")
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            reading: String => out("", "Reading", "", "Temperature design!C106"),
        }
```

with:

```rust
            reading: String => out("", "Reading", "", "Temperature design!C106"),
            magnet_cte_per_C: f64 => out_rust_only("1/°C", "Magnet expansion in the bond plane used",
                "Correction E22: the inner ring's grade value for ferrite or SmCo (the hub bonds the inner ring; the cup bond is not screened), else C95. The screen and warning rule 6 read it."),
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
```

with:

```rust
/// A grade's own value where the grade has one and correction `id` is on, else the workbook
/// `input` (Addendum A-4: E22's bond-plane CTE, E23's specific heat).
fn grade_value_or(input: f64, value: Option<f64>, dev: Deviations, id: DeviationId) -> f64 {
    match value {
        Some(v) if dev.is_on(id) => v,
        _ => input,
    }
}

/// E22 (Addendum A-4 decisions 5 to 7): the magnet expansion in the bond plane, across the
/// magnetization, that the mismatch screen and warning rule 6 read: the inner ring's grade
/// value where the grade has one (ferrite, SmCo), else `input_per_C` (Temperature design C95:
/// sintered NdFeB, every library part, a manual magnet without a grade, bonded NdFeB). Only
/// the inner ring: the hub bonds it, and the cup bond is not screened.
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub fn bond_plane_cte_per_C(input_per_C: f64, inner: Option<&Grade>, dev: Deviations) -> f64 {
    let grade_cte = inner.and_then(|g| g.bond_plane_cte_per_C);
    grade_value_or(input_per_C, grade_cte, dev, DeviationId::E22)
}

/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let d_alpha = hub_cte - mm.ndfeb_cte_per_C;
```

with:

```rust
    let magnet_cte = bond_plane_cte_per_C(mm.ndfeb_cte_per_C, k.inner_grade, dev);
    let d_alpha = hub_cte - magnet_cte;
    // E22 (decision 9): a grade that expands more than the hub makes the mismatch, and the
    // shear, negative; the screens compare its size and the signed values stay on display.
    // Without E22 the workbook compares the signed value. Without a grade, a negative mismatch
    // needs the default back iron's C17 set below C95 (a slider corner: every library hub
    // material is at 9.0e-6 /C or more, C95 at most 6e-6); the size comparison then reads
    // sintered NdFeB too (plan decision A4-4).
    let size = |shear: f64| {
        if dev.is_on(DeviationId::E22) {
            shear.abs()
        } else {
            shear
        }
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        reading: if s1 > sel.lap_shear_MPa {
            "Above the lap-shear strength at the block ends"
        } else {
            "Below the lap-shear strength"
        }
        .to_owned(),
    };
```

with:

```rust
        reading: if size(s1) > sel.lap_shear_MPa {
            "Above the lap-shear strength at the block ends"
        } else {
            "Below the lap-shear strength"
        }
        .to_owned(),
        magnet_cte_per_C: magnet_cte,
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        daily_screen: if daily < sel.lap_shear_MPa * al.fatigue_endurance {
```

with:

```rust
        daily_screen: if size(daily) < sel.lap_shear_MPa * al.fatigue_endurance {
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        magnet_cte_per_C: ti.mismatch.ndfeb_cte_per_C,
    });
```

with:

```rust
        // E22: the inner ring's grade CTE where it has one, as the mismatch screen reads it.
        magnet_cte_per_C: temp.mismatch.magnet_cte_per_C,
    });
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/warnings.rs`, replace:

```rust
                "Fires when the hub's expansion coefficient differs from the magnets' (Temperature design C95) by more than 15e-6 /°C."),
```

with:

```rust
                "Fires when the hub's expansion coefficient differs from the magnets' (Temperature design C95; with correction E22 the inner ring's grade value, temperature.mismatch.magnet_cte_per_C) by more than 15e-6 /°C."),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/warnings.rs`, replace:

```rust
    /// The magnets' expansion coefficient in the bond plane (Temperature design C95) [1/°C].
```

with:

```rust
    /// The magnets' expansion coefficient in the bond plane [1/°C]: Temperature design C95,
    /// or with E22 the inner ring's grade value (`temperature.mismatch.magnet_cte_per_C`).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
            "Heating changes a part's length by α ΔT per unit length; NdFeB barely changes across its magnetization (about −0.8 × 10⁻⁶ /°C), while steel expands about 12 × 10⁻⁶ /°C and aluminium about 24 × 10⁻⁶ /°C.",
            "Where a magnet is glued to the hub, that difference is forced through the thin glue line as shear, largest at the block ends.",
            "The larger the mismatch and the temperature swing, the higher that shear, so a hub whose expansion differs from the magnets' by more than 15 × 10⁻⁶ /°C is flagged; the Volkersen screen puts a number on it.",
        ],
```

with:

```rust
            "Heating changes a part's length by α ΔT per unit length; across its magnetization sintered NdFeB barely changes (about −0.8 × 10⁻⁶ /°C), ferrite and SmCo expand about 10 to 14 × 10⁻⁶ /°C (correction E22 reads the inner ring's grade), steel about 12 × 10⁻⁶ /°C and aluminium about 24 × 10⁻⁶ /°C.",
            "Where a magnet is glued to the hub, that difference is forced through the thin glue line as shear, largest at the block ends.",
            "The larger the mismatch and the temperature swing, the higher that shear, so a hub whose expansion differs from the inner magnets' by more than 15 × 10⁻⁶ /°C is flagged; the Volkersen screen puts a number on it.",
            "A magnet that expands more than its hub, SmCo on steel for example, shears the glue the other way, so the screen compares the size of the shear with the glue's strength.",
        ],
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
            "Addendum A decision 16 (E18: the aluminium hub's expansion)",
        ],
```

with:

```rust
            "Addendum A decision 16 (E18: the aluminium hub's expansion)",
            "Addendum A-4 decisions 5, 6 and 9 (E22: the inner ring's grade CTE; the size of the shear compared)",
        ],
```

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import pathlib

# A correction changed this note's text: it goes back to Draft (hidden by notes::note_for) until
# the physics review of Task 6 signs it off again. Anchored on the note id, whatever its review
# record says.
NOTE = "a5.cte_mismatch_with_magnets"
p = pathlib.Path("C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs")
s = p.read_text(encoding="utf-8")
start = s.index(f'        id: "{NOTE}",\n')
at = s.index("        review: Review::Reviewed {\n", start)
end = s.index("        },\n", at) + len("        },\n")
assert s.find("        id: ", start + 1, end) == -1, f"{NOTE}: no reviewed block before the next note"
s = s[:at] + "        review: Review::Draft,\n" + s[end:]
p.write_text(s, encoding="utf-8", newline="\n")
print(f"{NOTE} is a draft again")
EOF
```

Expected: prints `a5.cte_mismatch_with_magnets is a draft again`.

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
of its section 8 that each entry's `approval` cites) and E21 in the Addendum A-4
```

with:

```markdown
of its section 8 that each entry's `approval` cites) and E21 and E22 in the Addendum A-4
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
The default design's offset is +9.92 °C: unchanged | Addendum A-4 decisions 1, 3 |
```

with:

```markdown
The default design's offset is +9.92 °C: unchanged | Addendum A-4 decisions 1, 3 |
| E22 | Temperature design!C95 → C104, C105, C201; verdicts C106, C202; warning rule 6 | sintered NdFeB's bond-plane expansion (C95, −0.8e-6 /°C) for every magnet | the inner ring's grade value (the hub bonds it; the cup bond is not screened): Y30 10.0e-6, Recoma 20 14.0e-6, Recoma 26 and 30 13.0e-6 /°C (Arnold); C95 stays for sintered NdFeB, every library part, a manual magnet without a grade and bonded NdFeB (its proxies disagree 4x). A grade that expands more than the hub makes the shear negative: C106 and C202 compare its size, the signed values stay on display; so does C95 set above the default back iron's C17 (a slider corner, so the size comparison reads sintered NdFeB too, plan decision A4-4). C95 keeps its label and range; its help and the Rust-only `temperature.mismatch.magnet_cte_per_C` show the value in effect. Recoma 26 on both rings, registry basis: C104 64.07 → −3.42 MPa, C106 "Above" → "Below"; aluminium hub, user basis: warning rule 6 goes silent | Addendum A-4 decisions 4-7, 9, 10 |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E21 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A, E21 from Addendum A-4), probes, and the `Deviations` switch |
```

with:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E22 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A, E21 and E22 from Addendum A-4), probes, and the `Deviations` switch |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
each ring's applied calibration offset, E21, plan A-4) |
```

with:

```markdown
each ring's applied calibration offset, E21, plan A-4; the magnet CTE in effect, E22) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
`e21_alone_moves_no_differential_case` (the double gate: every corpus case).
```

with:

```markdown
`e21_alone_moves_no_differential_case` (the double gate: every corpus case); for E22 `e22_leaves_every_default_cell_bit_for_bit`, `e22_moves_only_a_ferrite_or_smco_inner_ring` (every grade on both rings, inside and outside, every library part, both hubs, both bases, and the wiring to C95), `e22_reproduces_the_report_s_bond_screen_table`, `e22_compares_the_size_of_a_negative_shear` (decision 9), `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too` (the C17 and C95 slider corner, decision A4-4) and `e22_moves_no_differential_case` (every corpus case, Rust-only rows included).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
16 of the 17 are signed off (the demagnetization note is a draft again until its E21 sentence is reviewed, plan A-4;
```

with:

```markdown
15 of the 17 are signed off (the demagnetization and expansion-mismatch notes are drafts again until their E21 and E22 sentences are reviewed, plan A-4;
```

- [ ] **Step 4: Format, bless the input schema and run the tests**

C95's help changed, and `tests/data/input_schema.json` records every input's help (the generator reads only its ranges, so the differential data does not change).

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml
cd C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs && MAGCOUPLING_BLESS=1 cargo test --test schema 2>&1 | grep "test result"
git -C C:/Users/Cole/source/repos/lsim-mag-a4 diff --stat -- magcoupling-rs/tests/data/input_schema.json
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: `test result: ok. 7 passed`; the schema diff is `1 file changed, 1 insertion(+), 1 deletion(-)` (C95's `help` line only); then every binary `ok`, in order: unit tests `175 passed`; `tests\assumptions.rs` 8 passed; `tests\deviations.rs` 70 passed; `tests\differential.rs` 19 passed; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 12 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 12 passed; `tests\schema.rs` 7 passed; `tests\sizing.rs` 31 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. (Read the counts as offsets from Task 0's if Task 0 recorded different ones.)

- [ ] **Step 5: Lints, wasm and the differential data**

Run:

```bash
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
cd C:/Users/Cole/source/repos/lsim-mag-a4/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: each cargo line ends `Finished ...` (no warning, no error); then `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`.

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task3.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 add magcoupling-rs/src/engine/deviations.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/warnings.rs magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a4 commit -F - <<'EOF'
feat(magcoupling-rs): E22, the inner ring's grade CTE in the bond screen

The Volkersen screen and warning rule 6 read the inner ring's grade CTE across
the magnetization (ferrite, SmCo; A-4 decisions 4 to 7); C95 stays for sintered
and bonded NdFeB. A grade that expands more than the hub makes the shear
negative, so C106 and C202 compare its size (decision 9). C95 keeps its label
and range; its help and the Rust-only temperature.mismatch.magnet_cte_per_C
show the value in effect (decision 10). At the default sliders no sintered
NdFeB design moves (where C95 exceeds the default back iron's C17 the size
comparison reads its negative shear too, decision A4-4), and no differential
case moves; the expansion-mismatch note is back to Draft until Task 6.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 4: E23, each ring's grade specific heat in the heat capacity

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5): a physics correction to the thermal network and its explain records. Physics reviewer: the A-4 report sections 3.1, 4.4 and decision A2-7.

Decisions 4, 5, 8 and 10 (A). C141's magnet term is m_inner · c_inner + m_outer · c_outer, each ring's N V ρ at its own density (decision A2-7) and its grade's specific heat (`temperature::magnet_specific_heat_J_kgK`, through Task 3's `grade_value_or`), in both of E15's branches; rings of one specific heat keep the workbook's C110 × c bit for bit. The per-ring masses become Rust-only mass results computed beside C110 (one source), and the per-ring specific heats in effect are Rust-only thermal results (decision 10); the explorer gains a record for each and C141's record reads them. C138 keeps its label and range; its help is reworded and registered.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs` (module doc, `DeviationId::E23`, the E23 entry)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/model.rs` (`MassResults::magnets_inner_g`, `magnets_outer_g`; the mass test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs` (module doc, `magnet_specific_heat_J_kgK`, the magnet heat term, C138 help, two links fields, two Rust-only results, the fixture, a unit test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/api.rs` (the per-ring masses into the links)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/slip_heating.rs` (four new records, C141 rewritten)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs` (six tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/data/input_schema.json` (blessed: C138's help)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `Grade::specific_heat_J_kgK` and the table field `specific_heat_J_kgK` (Task 2); `a4_bases`, `grade_rings`, `Overrides`, `results_with`, `assert_same_results`, `assert_a4_defaults_unchanged` (Task 1); `grade_value_or`, `grade_placements`, `HUBS`, `assert_no_differential_case_moves` (Task 3).
- Produces (read by Tasks 5 and 7):
  - `pub fn magnet_specific_heat_J_kgK(input: f64, grade: Option<&Grade>, dev: Deviations) -> f64` (temperature.rs);
  - Rust-only `MassResults::magnets_inner_g`, `magnets_outer_g: f64` (`mass.magnets_inner_g`, `mass.magnets_outer_g`) and `ThermalResults::inner_magnet_c_J_kgK`, `outer_magnet_c_J_kgK: f64`; `TemperatureLinks::mass_magnets_inner_g`, `mass_magnets_outer_g: f64`; `DeviationId::E23`.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
        DeviationId::E21,
        DeviationId::E22,
    ]
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
```

with:

```rust
        DeviationId::E21,
        DeviationId::E22,
        DeviationId::E23,
    ]
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
#[test]
fn e23_leaves_every_default_cell_bit_for_bit() {
    // The default part is sintered NdFeB: both rings keep C138, the workbook's single product.
    assert_a4_defaults_unchanged(DeviationId::E23);
}

#[test]
fn e23_moves_only_a_ring_with_a_grade_specific_heat() {
    // A-4 report sections 4.1, 4.2 and 4.4: E23 reads each ring's grade specific heat, so a
    // ferrite, SmCo or bonded NdFeB ring on either side moves the heat capacity C141 by its
    // mass times its specific heat's difference from C138, and nothing moves for sintered
    // NdFeB or a library part; both hubs, both bases. A grade on both rings is wired as
    // approved: E23 equals E23 off with C138 set to the grade's value, on every row.
    use magcoupling::engine::grades::GRADES;
    use magcoupling::engine::library::MAGNET_LIBRARY;
    let c138 = DesignInputs::default().temperature.thermal.c_ndfeb;
    let mut configs: Vec<(String, Overrides, Option<f64>, &str)> = Vec::new();
    for g in &GRADES {
        for (placement, overrides) in grade_placements(g.id) {
            let label = format!("{} {placement}", g.id);
            configs.push((label, overrides, g.specific_heat_J_kgK, placement));
        }
    }
    for spec in &MAGNET_LIBRARY {
        let label = format!("{} both rings", spec.part);
        configs.push((label, both_rings(spec.part).to_vec(), None, "both rings"));
    }
    let mut moved = 0;
    for (hub, backiron) in HUBS {
        for (basis, before, after) in a4_bases(DeviationId::E23) {
            for (label, overrides, cp, placement) in &configs {
                let mut o = overrides.clone();
                o.push(("coupling.backiron", Value::Int(backiron)));
                let what = format!("{label}, {hub}, {basis}");
                let (b, a) = (results_with(&o, before), results_with(&o, after));
                let Some(cp) = *cp else {
                    assert_same_results(&b, &a, &what);
                    continue;
                };
                moved += 1;
                let (tb, ta) = (&b.temperature.thermal, &a.temperature.thermal);
                let mass = &a.mass;
                let (ring_mass, inner, outer) = match *placement {
                    "both rings" => (mass.magnets_g, cp, cp),
                    "inner only" => (mass.magnets_inner_g, cp, c138),
                    _ => (mass.magnets_outer_g, c138, cp),
                };
                assert_eq!(
                    (ta.inner_magnet_c_J_kgK, ta.outer_magnet_c_J_kgK),
                    (inner, outer),
                    "{what}"
                );
                let shift = ta.heat_capacity_J_K - tb.heat_capacity_J_K;
                let want = ring_mass * (cp - c138) / 1000.0;
                assert!(
                    (shift - want).abs() <= 1e-9 * want.abs(),
                    "{what}: C141 moved {shift}, not {want}"
                );
                if *placement == "both rings" {
                    let mut wired = inputs_with(&o, before);
                    wired.temperature.thermal.c_ndfeb = cp;
                    let w = compute_all_with(&wired, before);
                    // The input itself shows in effect when E23 is off; every other row agrees.
                    assert_eq!(
                        w.temperature.thermal.heat_capacity_J_K,
                        ta.heat_capacity_J_K
                    );
                    let mut w = w;
                    w.temperature.thermal.inner_magnet_c_J_kgK = ta.inner_magnet_c_J_kgK;
                    w.temperature.thermal.outer_magnet_c_J_kgK = ta.outer_magnet_c_J_kgK;
                    assert_same_results(&w, &a, &format!("{what}, wiring"));
                }
            }
        }
    }
    // Y30, Recoma 20, 26, 30 and bonded NdFeB, in three placements; two hubs; two bases.
    assert_eq!(moved, 5 * 3 * 2 * 2);
}

/// A row of the A-4 report's table 4.4 (user basis, steel hub): (grade, placement, C141 and
/// C143 as [before, after]).
type ThermalRow = (&'static str, &'static str, [f64; 2], [f64; 2]);

#[test]
fn e23_reproduces_the_report_s_thermal_table() {
    // A-4 report section 4.4 on the user basis; "one ring" is the inner ring here (with the
    // default dimensions the outer ring gives the same masses). The verdict never moves.
    let rows: [ThermalRow; 10] = [
        ("Y30", "both rings", [76.57, 83.22], [255.2, 277.4]),
        ("Y30", "inner only", [79.38, 82.71], [264.6, 275.7]),
        ("SmCo_1_5_20", "both rings", [84.22, 81.21], [280.7, 270.7]),
        ("SmCo_1_5_20", "inner only", [83.21, 81.70], [277.4, 272.3]),
        ("SmCo_2_17_26", "both rings", [83.99, 80.17], [280.0, 267.2]),
        ("SmCo_2_17_26", "inner only", [83.09, 81.18], [277.0, 270.6]),
        ("SmCo_2_17_30", "both rings", [83.99, 80.17], [280.0, 267.2]),
        ("SmCo_2_17_30", "inner only", [83.09, 81.18], [277.0, 270.6]),
        (
            "Bonded_NdFeB_BCN19",
            "both rings",
            [78.37, 77.78],
            [261.2, 259.3],
        ),
        (
            "Bonded_NdFeB_BCN19",
            "inner only",
            [80.28, 79.99],
            [267.6, 266.6],
        ),
    ];
    let [_, (_, before, after)] = a4_bases(DeviationId::E23);
    for (grade, placement, c141, c143) in rows {
        let (_, o) = grade_placements(grade)
            .into_iter()
            .find(|(p, _)| *p == placement)
            .expect("a placement");
        let (b, a) = (cells_with(&o, before), cells_with(&o, after));
        let what = format!("{grade} {placement}");
        for (cell, want) in [
            ("Temperature design!C141", c141),
            ("Temperature design!C143", c143),
        ] {
            assert_sig4(&format!("{what} {cell}"), &b[cell], want[0]);
            assert_sig4(&format!("{what} {cell}"), &a[cell], want[1]);
        }
        assert_eq!(
            b["Temperature design!C25"], a["Temperature design!C25"],
            "{what}"
        );
    }
    // The time to the limit, aluminium hub, one ring (C150).
    for (grade, want) in [
        ("Y30", [653.6, 689.6]),
        ("SmCo_1_5_20", [695.1, 678.8]),
        ("SmCo_2_17_26", [693.9, 673.1]),
    ] {
        let (_, mut o) = grade_placements(grade)
            .into_iter()
            .find(|(p, _)| *p == "inner only")
            .expect("a placement");
        o.push(("coupling.backiron", Value::Int(0)));
        let (b, a) = (cells_with(&o, before), cells_with(&o, after));
        assert_sig4(grade, &b["Temperature design!C150"], want[0]);
        assert_sig4(grade, &a["Temperature design!C150"], want[1]);
    }
}

#[test]
fn e23_moves_no_verdict_and_the_peak_by_hundredths_of_a_degree() {
    // A-4 report section 4.1: over every ordered pair of grades on both hubs and both bases,
    // E23 changes no verdict C25, and the peak magnet temperature C180 by at most 0.031 C
    // (Y30 on both rings, aluminium hub). Each ring is at its own specific heat, so C141 moves
    // by m_inner (c_inner - C138) + m_outer (c_outer - C138), mixed non-NdFeB rings (Y30
    // beside Recoma 26) included, and not at all for two sintered NdFeB rings.
    use magcoupling::engine::grades::{GRADES, Grade};
    let c138 = DesignInputs::default().temperature.thermal.c_ndfeb;
    let c = |g: &Grade| g.specific_heat_J_kgK.unwrap_or(c138);
    let mut largest: f64 = 0.0;
    for (hub, backiron) in HUBS {
        for (basis, before, after) in a4_bases(DeviationId::E23) {
            for gi in &GRADES {
                for go in &GRADES {
                    let mut o = grade_rings(gi.id, go.id).to_vec();
                    o.push(("coupling.backiron", Value::Int(backiron)));
                    let what = format!("{}/{}, {hub}, {basis}", gi.id, go.id);
                    let (b, a) = (results_with(&o, before), results_with(&o, after));
                    let (sb, sa) = (&b.temperature.summary, &a.temperature.summary);
                    assert_eq!(sb.verdict, sa.verdict, "{what}");
                    let peak = |r: &DesignResults| r.temperature.magnet_life.peak_C;
                    largest = largest.max((peak(&a) - peak(&b)).abs());
                    let (tb, ta, m) = (&b.temperature.thermal, &a.temperature.thermal, &a.mass);
                    assert_eq!(
                        (ta.inner_magnet_c_J_kgK, ta.outer_magnet_c_J_kgK),
                        (c(gi), c(go)),
                        "{what}"
                    );
                    let want = (m.magnets_inner_g * (c(gi) - c138)
                        + m.magnets_outer_g * (c(go) - c138))
                        / 1000.0;
                    let shift = ta.heat_capacity_J_K - tb.heat_capacity_J_K;
                    assert!(
                        (shift - want).abs() <= 1e-9 * want.abs(),
                        "{what}: C141 moved {shift}, not {want}"
                    );
                }
            }
        }
    }
    assert!(largest > 0.03 && largest < 0.0314, "{largest}");
}

#[test]
fn e22_and_e23_change_disjoint_cells() {
    // A-4 decision 4: the two corrections are split because their cells do not overlap, so
    // each is probed and approved alone.
    let cells = |id: DeviationId| REGISTRY[id.index()].cells;
    for cell in cells(DeviationId::E22) {
        assert!(!cells(DeviationId::E23).contains(cell), "{cell}");
    }
}

#[test]
fn e23_moves_no_differential_case() {
    // A-4 report section 4.2: no case of the corpus picks a grade.
    assert_no_differential_case_moves(DeviationId::E23);
}

#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

In the unit tests of `model.rs` (the mass test also checks each ring's own mass) and `temperature.rs`:

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/model.rs`, replace:

```rust
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
```

with:

```rust
            (m, r)
        };
        let (workbook, r) = mass(&CouplingInputs::default());
        let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
        let volume_o = r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm;
        assert_eq!(
            workbook.magnets_g,
            10.0 * (volume_i + volume_o) * NDFEB_DENSITY_G_MM3
        );
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        let (ferrite, r) = mass(&ci);
        let volume_i = r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm;
        assert_eq!(
            ferrite.magnets_g,
            10.0 * (volume_i * 0.005 + volume_o * NDFEB_DENSITY_G_MM3)
        );
        // Plan A-4 (E23): each ring's own mass, N V rho; together they are C110.
        assert_eq!(
            (ferrite.magnets_inner_g, ferrite.magnets_outer_g),
            (
                10.0 * volume_i * 0.005,
                10.0 * volume_o * NDFEB_DENSITY_G_MM3
            )
        );
        let sum = ferrite.magnets_inner_g + ferrite.magnets_outer_g;
        assert!((sum - ferrite.magnets_g).abs() <= 1e-12 * sum);
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    #[test]
    fn each_ring_shows_its_own_block() {
```

with:

```rust
    #[test]
    fn e23_prices_each_ring_at_its_own_specific_heat() {
        // Addendum A-4 decisions 5 and 8: a Y30 inner ring (700 J/(kg K)) beside an N42SH outer
        // ring (C138) adds its mass times 260 J/(kg K) to the heat capacity; Y30 on both rings
        // is C138 typed to 700, bit for bit (the workbook's single product); sintered NdFeB
        // and E23 off keep C138.
        let e23 = Deviations::only(DeviationId::E23);
        let ti = TemperatureInputs::default();
        let c138 = ti.thermal.c_ndfeb;
        let y30 = crate::engine::grades::grade("Y30");
        let mut k = links();
        k.inner_grade = y30;
        let (off, on) = (run(&ti, &k).thermal, compute(&ti, &k, e23).thermal);
        assert_eq!(
            (on.inner_magnet_c_J_kgK, on.outer_magnet_c_J_kgK),
            (700.0, c138)
        );
        assert_eq!(
            (off.inner_magnet_c_J_kgK, off.outer_magnet_c_J_kgK),
            (c138, c138)
        );
        let want = k.mass_magnets_inner_g * (700.0 - c138) / 1000.0;
        let shift = on.heat_capacity_J_K - off.heat_capacity_J_K;
        assert!((shift - want).abs() <= 1e-12 * want, "{shift} vs {want}");
        k.outer_grade = y30;
        let mut typed = TemperatureInputs::default();
        typed.thermal.c_ndfeb = 700.0;
        assert_eq!(
            compute(&ti, &k, e23).thermal.heat_capacity_J_K.to_bits(),
            run(&typed, &k).thermal.heat_capacity_J_K.to_bits()
        );
        assert_eq!(compute(&ti, &links(), e23), run(&ti, &links()), "NdFeB");
    }

    #[test]
    fn each_ring_shows_its_own_block() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test deviations 2>&1 | grep -E "^error" | sort | uniq -c
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --lib 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: compile errors only, the first command:

```text
1 error: could not compile `magcoupling-rs` (test "deviations") due to 19 previous errors
7 error[E0599]: no variant or associated item named `E23` found for enum `DeviationId` in the current scope
3 error[E0609]: no field `inner_magnet_c_J_kgK` on type `&ThermalResults`
1 error[E0609]: no field `inner_magnet_c_J_kgK` on type `ThermalResults`
2 error[E0609]: no field `magnets_inner_g` on type `&MassResults`
2 error[E0609]: no field `magnets_outer_g` on type `&MassResults`
3 error[E0609]: no field `outer_magnet_c_J_kgK` on type `&ThermalResults`
1 error[E0609]: no field `outer_magnet_c_J_kgK` on type `ThermalResults`
```

and the second:

```text
1 error: could not compile `magcoupling-rs` (lib test) due to 10 previous errors
1 error[E0599]: no variant or associated item named `E23` found for enum `deviations::DeviationId` in the current scope
2 error[E0609]: no field `inner_magnet_c_J_kgK` on type `ThermalResults`
2 error[E0609]: no field `magnets_inner_g` on type `MassResults`
2 error[E0609]: no field `magnets_outer_g` on type `MassResults`
1 error[E0609]: no field `mass_magnets_inner_g` on type `engine::temperature::TemperatureLinks`
2 error[E0609]: no field `outer_magnet_c_J_kgK` on type `ThermalResults`
```

- [ ] **Step 3: Implement E23 and its explain records**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! E21 (decision 1: the rating-calibration offset clamped at 0) and E22
//! (decisions 4 to 7, 9, 10: the inner ring's grade CTE in the bond plane),
//! approved on 2026-10-01. Everything else stays workbook-exact. This registry is the one
```

with:

```rust
//! E21 (decision 1: the rating-calibration offset clamped at 0), E22
//! (decisions 4 to 7, 9, 10: the inner ring's grade CTE in the bond plane) and
//! E23 (decisions 4, 5, 8, 10: each ring's grade specific heat), approved on
//! 2026-10-01. Everything else stays workbook-exact. This registry is the one
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    E21,
    E22,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 22] = [
```

with:

```rust
    E21,
    E22,
    E23,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 23] = [
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        DeviationId::E21,
        DeviationId::E22,
    ];
```

with:

```rust
        DeviationId::E21,
        DeviationId::E22,
        DeviationId::E23,
    ];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
                    CellChange {
                        cell: "Temperature design!C202",
                        workbook: Literal::Text(
                            "Above the fatigue endurance: qualify by thermal cycling",
                        ),
                        corrected: Literal::Text("Below the fatigue endurance"),
                    },
                ],
            },
        ],
    },
];
```

with:

```rust
                    CellChange {
                        cell: "Temperature design!C202",
                        workbook: Literal::Text(
                            "Above the fatigue endurance: qualify by thermal cycling",
                        ),
                        corrected: Literal::Text("Below the fatigue endurance"),
                    },
                ],
            },
        ],
    },
    Deviation {
        id: DeviationId::E23,
        title: "The heat capacity prices every magnet grade at sintered NdFeB's specific heat",
        class: DeviationClass::Engine,
        approval: Approval::AddendumA4 {
            decisions: &[4, 5, 8, 10],
        },
        depends_on: &[],
        status: DeviationStatus::Applied,
        cells: &[
            "Temperature design!C19",
            "Temperature design!C20",
            "Temperature design!C141",
            "Temperature design!C143",
            "Temperature design!C145",
            "Temperature design!C150",
            "Temperature design!C151",
            "Temperature design!C152",
            "Temperature design!C154",
            "Temperature design!C155",
            "Temperature design!C156",
            "Temperature design!C157",
            "Temperature design!C158",
            "Temperature design!C159",
            "Temperature design!C160",
            "Temperature design!C161",
            "Temperature design!C171",
            "Temperature design!C172",
            "Temperature design!C180",
            "Temperature design!C181",
            "Temperature design!C182",
            "Temperature design!C186",
            "Temperature design!C189",
            "Temperature design!C190",
            "Temperature design!C192",
            "Temperature design!C193",
            "Temperature design!C196",
        ],
        corrected_formula: "Each ring's magnets at its grade's specific heat: Y30 700.0, Recoma 20 370.0, Recoma 26 and \
            Recoma 30 350.0, bonded NdFeB 420.0 J/(kg K) (decisions 5, 8; the bonded value is single-source, flagged); C138 \
            (440) stays for sintered NdFeB (every library part) and a manual magnet without a grade. The magnets' term of \
            the heat capacity C141 is m_inner c_inner + m_outer c_outer (each ring's N V rho at its own density, decision \
            A2-7), and C110 x c when both rings share c (the workbook's product, bit for bit), in both of E15's branches; \
            the thermal cells downstream follow. C138 keeps its label and range; its help and the Rust-only \
            temperature.thermal.inner_magnet_c_J_kgK and outer_magnet_c_J_kgK show the values in effect (decision 10).",
        workbook_input_defaults: &[],
        workbook_help: &[("temperature.thermal.c_ndfeb", "")],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[
            Probe {
                label: "Y30 (700 J/(kg K)) for manual dimensions on both rings, steel hub (A-4 report probe E23-1)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("")),
                    ("coupling.magnets.part_outer", Literal::Text("")),
                    ("coupling.magnets.grade_inner", Literal::Text("Y30")),
                    ("coupling.magnets.grade_outer", Literal::Text("Y30")),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C141",
                        workbook: Literal::Num(76.57010154344385),
                        corrected: Literal::Num(83.21686244344386),
                    },
                    CellChange {
                        cell: "Temperature design!C143",
                        workbook: Literal::Num(255.23367181147952),
                        corrected: Literal::Num(277.3895414781462),
                    },
                    CellChange {
                        cell: "Temperature design!C180",
                        workbook: Literal::Num(65.88143223919256),
                        corrected: Literal::Num(65.86604472256508),
                    },
                ],
            },
            Probe {
                label: "Recoma 26 (350 J/(kg K)) for manual dimensions inside, B842SH outside (A-4 report probe E23-2: \
                    each ring at its own specific heat)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("")),
                    (
                        "coupling.magnets.grade_inner",
                        Literal::Text("SmCo_2_17_26"),
                    ),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C141",
                        workbook: Literal::Num(83.09415301144385),
                        corrected: Literal::Num(81.18448747594385),
                    },
                    CellChange {
                        cell: "Temperature design!C180",
                        workbook: Literal::Num(65.86630657629901),
                        corrected: Literal::Num(65.87048331422399),
                    },
                ],
            },
        ],
    },
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            added_inertia_kgm2: f64 => out("kg·m²", "Added inertia at 0.25 m from the swing axis",
                "Point-mass estimate.", "Calculator!C115"),
        }
```

with:

```rust
            added_inertia_kgm2: f64 => out("kg·m²", "Added inertia at 0.25 m from the swing axis",
                "Point-mass estimate.", "Calculator!C115"),
            magnets_inner_g: f64 => out_rust_only("g", "Magnets, inner ring",
                "Plan A-4: N V rho for the inner ring at its own density; correction E23 prices it at the ring's own specific heat."),
            magnets_outer_g: f64 => out_rust_only("g", "Magnets, outer ring",
                "Plan A-4: as the inner ring's, for the outer ring."),
        }
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        total_g: total,
        added_inertia_kgm2: total / 1000.0 * 0.25_f64.powi(2),
    }
}
```

with:

```rust
        total_g: total,
        added_inertia_kgm2: total / 1000.0 * 0.25_f64.powi(2),
        // Plan A-4 (E23): each ring's own mass; their sum is C110 up to rounding.
        magnets_inner_g: N * volume_i * rho_i,
        magnets_outer_g: N * volume_o * rho_o,
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! raises the onsets above the model's own) and E22 (the mismatch screen and
//! warning rule 6 read the inner ring's grade CTE, [`bond_plane_cte_per_C`],
//! and the screens compare the size of the shear).
```

with:

```rust
//! raises the onsets above the model's own), E22 (the mismatch screen and
//! warning rule 6 read the inner ring's grade CTE, [`bond_plane_cte_per_C`],
//! and the screens compare the size of the shear) and E23 (C141 prices each
//! ring's magnets at its grade's specific heat, [`magnet_specific_heat_J_kgK`]).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            c_ndfeb: f64 = 440.0 => param("J/(kg·K)", "Specific heat, NdFeB",
                "", "Temperature design!C138")
```

with:

```rust
            c_ndfeb: f64 = 440.0 => param("J/(kg·K)", "Specific heat, NdFeB",
                "Sintered NdFeB: every library part and a manual magnet without a grade. Correction E23: a ferrite, SmCo or bonded NdFeB ring uses its grade's specific heat instead (temperature.thermal.inner_magnet_c_J_kgK and outer_magnet_c_J_kgK show the values in effect).", "Temperature design!C138")
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    pub body_cte: f64,
    pub body_E_GPa: f64,
}
```

with:

```rust
    pub body_cte: f64,
    pub body_E_GPa: f64,
    /// Plan A-4 (E23): each ring's magnet mass, N V rho (`mass.magnets_inner_g`,
    /// `magnets_outer_g`); their sum is C110 (`mass_magnets_g`) up to rounding.
    pub mass_magnets_inner_g: f64,
    pub mass_magnets_outer_g: f64,
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            temp_at_fault_C: f64 => out("°C", "Magnet temperature at the fault trip time, high case", "", "Temperature design!C161"),
```

with:

```rust
            temp_at_fault_C: f64 => out("°C", "Magnet temperature at the fault trip time, high case", "", "Temperature design!C161"),
            inner_magnet_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Specific heat, inner ring magnets",
                "Correction E23: the inner ring's grade value for ferrite, SmCo or bonded NdFeB, else C138."),
            outer_magnet_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Specific heat, outer ring magnets",
                "Correction E23: as the inner ring's, for the outer ring."),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
```

with:

```rust
/// E23 (Addendum A-4 decisions 5 and 8): a ring's magnet specific heat [J/(kg·K)]: its
/// grade's (ferrite, SmCo, bonded NdFeB), else `input` (Temperature design C138: sintered
/// NdFeB, every library part, a manual magnet without a grade).
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub fn magnet_specific_heat_J_kgK(input: f64, grade: Option<&Grade>, dev: Deviations) -> f64 {
    let grade_c = grade.and_then(|g| g.specific_heat_J_kgK);
    grade_value_or(input, grade_c, dev, DeviationId::E23)
}

/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    // ---- thermal network
    let th = &ti.thermal;
```

with:

```rust
    // ---- thermal network
    let th = &ti.thermal;
    // E23: each ring's magnets at its own specific heat. Rings of one specific heat keep the
    // workbook's single product, C110 x c, bit for bit (decision A2-7's pattern).
    let c_mag_i = magnet_specific_heat_J_kgK(th.c_ndfeb, k.inner_grade, dev);
    let c_mag_o = magnet_specific_heat_J_kgK(th.c_ndfeb, k.outer_grade, dev);
    let magnets_heat = if c_mag_i == c_mag_o {
        k.mass_magnets_g * c_mag_i
    } else {
        k.mass_magnets_inner_g * c_mag_i + k.mass_magnets_outer_g * c_mag_o
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        (k.mass_magnets_g * th.c_ndfeb
            + (k.mass_cup_g + k.mass_boss_g) * c_cup
```

with:

```rust
        (magnets_heat
            + (k.mass_cup_g + k.mass_boss_g) * c_cup
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        (k.mass_magnets_g * th.c_ndfeb
            + (k.mass_cup_g + k.mass_hub_g + k.mass_boss_g + k.hardware_g) * k.steel_c
```

with:

```rust
        (magnets_heat
            + (k.mass_cup_g + k.mass_hub_g + k.mass_boss_g + k.hardware_g) * k.steel_c
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        rev95: 3.0 * tau_th * rev_s,
        temp_at_fault_C: T_fault,
    };
```

with:

```rust
        rev95: 3.0 * tau_th * rev_s,
        temp_at_fault_C: T_fault,
        inner_magnet_c_J_kgK: c_mag_i,
        outer_magnet_c_J_kgK: c_mag_o,
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            mass_magnets_g: 38.347,
```

with:

```rust
            mass_magnets_g: 38.347,
            mass_magnets_inner_g: 38.347 / 2.0,
            mass_magnets_outer_g: 38.347 / 2.0,
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        mass_magnets_g: mass.magnets_g,
```

with:

```rust
        mass_magnets_g: mass.magnets_g,
        mass_magnets_inner_g: mass.magnets_inner_g,
        mass_magnets_outer_g: mass.magnets_outer_g,
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/slip_heating.rs`, replace:

```rust
use crate::engine::deviations::DeviationId::{E5, E8, E9, E12, E15, E17};
```

with:

```rust
use crate::engine::deviations::DeviationId::{E5, E8, E9, E12, E15, E17, E23};
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/slip_heating.rs`, replace:

```rust
         where [V_i] = {model.inner_length_mm} * {model.inner_width_mm} * {model.inner_thickness_mm},
               [V_o] = {model.outer_length_mm} * {model.outer_width_mm} * {model.outer_thickness_mm}"),
```

with:

```rust
         where [V_i] = {model.inner_length_mm} * {model.inner_width_mm} * {model.inner_thickness_mm},
               [V_o] = {model.outer_length_mm} * {model.outer_width_mm} * {model.outer_thickness_mm}"),
    // Plan A-4 (E23): each ring's own mass, at its own density.
    record("mass.magnets_inner_g", "m_{mag,i}",
        "{coupling.npole} * [V_i] * {model.inner_magnet_density_g_mm3}
         where [V_i] = {model.inner_length_mm} * {model.inner_width_mm} * {model.inner_thickness_mm}"),
    record("mass.magnets_outer_g", "m_{mag,o}",
        "{coupling.npole} * [V_o] * {model.outer_magnet_density_g_mm3}
         where [V_o] = {model.outer_length_mm} * {model.outer_width_mm} * {model.outer_thickness_mm}"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/slip_heating.rs`, replace:

```rust
    // --- Thermal network: one lumped heat capacity and one conductance ---
    // E15: an aluminium hub, cup and boss at the body material's specific heat; hardware stays steel.
    record("temperature.thermal.heat_capacity_J_K", "C",
        "cases({materials.circuit_backiron} != 1
                 => ({mass.magnets_g} * {temperature.thermal.c_ndfeb} + ({mass.cup_g} + {mass.boss_g}) * {materials.body_c_J_kgK}
                     + {mass.hub_g} * {materials.body_c_J_kgK} + {metal.hardware_g} * {temperature.thermal.steel_c}
                     + ({retainers.retainers_g} + {retainers.endplates_g}) * {materials.sleeve_c_J_kgK} + {retainers.cap_g} * {materials.cap_c_J_kgK}) / 1000;
               else => ({mass.magnets_g} * {temperature.thermal.c_ndfeb}
                     + ({mass.cup_g} + {mass.hub_g} + {mass.boss_g} + {metal.hardware_g}) * {temperature.thermal.steel_c}
                     + ({retainers.retainers_g} + {retainers.endplates_g}) * {materials.sleeve_c_J_kgK} + {retainers.cap_g} * {materials.cap_c_J_kgK}) / 1000)")
        .corrected(&[E9, E15]),
```

with:

```rust
    // --- Thermal network: one lumped heat capacity and one conductance ---
    // E23: each ring's magnets at its grade's specific heat (ferrite, SmCo, bonded NdFeB), else
    // the NdFeB input.
    record("temperature.thermal.inner_magnet_c_J_kgK", "c_{mag,i}",
        r#"cases([c_{grade}] != none => [c_{grade}]; else => {temperature.thermal.c_ndfeb})
           where [c_{grade}] = table("grades", {model.inner_grade}, "specific_heat_J_kgK")"#).corrected(&[E23]),
    record("temperature.thermal.outer_magnet_c_J_kgK", "c_{mag,o}",
        r#"cases([c_{grade}] != none => [c_{grade}]; else => {temperature.thermal.c_ndfeb})
           where [c_{grade}] = table("grades", {model.outer_grade}, "specific_heat_J_kgK")"#).corrected(&[E23]),
    // E15: an aluminium hub, cup and boss at the body material's specific heat; hardware stays
    // steel. E23: rings of one specific heat keep the workbook's single product.
    record("temperature.thermal.heat_capacity_J_K", "C",
        "cases({materials.circuit_backiron} != 1
                 => ([Q_{mag}] + ({mass.cup_g} + {mass.boss_g}) * {materials.body_c_J_kgK}
                     + {mass.hub_g} * {materials.body_c_J_kgK} + {metal.hardware_g} * {temperature.thermal.steel_c}
                     + ({retainers.retainers_g} + {retainers.endplates_g}) * {materials.sleeve_c_J_kgK} + {retainers.cap_g} * {materials.cap_c_J_kgK}) / 1000;
               else => ([Q_{mag}]
                     + ({mass.cup_g} + {mass.hub_g} + {mass.boss_g} + {metal.hardware_g}) * {temperature.thermal.steel_c}
                     + ({retainers.retainers_g} + {retainers.endplates_g}) * {materials.sleeve_c_J_kgK} + {retainers.cap_g} * {materials.cap_c_J_kgK}) / 1000)
         where [Q_{mag}] = cases({temperature.thermal.inner_magnet_c_J_kgK} = {temperature.thermal.outer_magnet_c_J_kgK}
                                   => {mass.magnets_g} * {temperature.thermal.inner_magnet_c_J_kgK};
                                 else => {mass.magnets_inner_g} * {temperature.thermal.inner_magnet_c_J_kgK}
                                         + {mass.magnets_outer_g} * {temperature.thermal.outer_magnet_c_J_kgK})")
        .corrected(&[E9, E15, E23]),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
of its section 8 that each entry's `approval` cites) and E21 and E22 in the Addendum A-4
```

with:

```markdown
of its section 8 that each entry's `approval` cites) and E21 to E23 in the Addendum A-4
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
aluminium hub, user basis: warning rule 6 goes silent | Addendum A-4 decisions 4-7, 9, 10 |
```

with:

```markdown
aluminium hub, user basis: warning rule 6 goes silent | Addendum A-4 decisions 4-7, 9, 10 |
| E23 | Temperature design!C138 → C141 and the thermal rows downstream (C143, C145, C150-C152, C154-C161, C171, C172, C180-C182, C186, C189, C190, C192, C193, C196, C19, C20) | sintered NdFeB's specific heat (C138, 440 J/(kg·K)) for every magnet | each ring's magnets at its grade's specific heat: Y30 700, Recoma 20 370, Recoma 26 and 30 350, bonded NdFeB 420 (single-source, flagged) J/(kg·K); C138 stays for sintered NdFeB, every library part and a manual magnet without a grade. C141's magnet term is m_inner c_inner + m_outer c_outer (Rust-only `mass.magnets_inner_g`, `magnets_outer_g`), and C110 × c when both rings share c (bit for bit). C138 keeps its label and range; its help and the Rust-only `temperature.thermal.inner_magnet_c_J_kgK`, `outer_magnet_c_J_kgK` show the values in effect. Y30 on both rings, registry basis: C141 76.57 → 83.22 J/K; no verdict moves, and the peak by at most 0.031 °C | Addendum A-4 decisions 4, 5, 8, 10 |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E22 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A, E21 and E22 from Addendum A-4), probes, and the `Deviations` switch |
```

with:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E23 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A, E21 to E23 from Addendum A-4), probes, and the `Deviations` switch |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
each ring's applied calibration offset, E21, plan A-4; the magnet CTE in effect, E22) |
```

with:

```markdown
each ring's applied calibration offset, E21, plan A-4; the magnet CTE in effect, E22; each ring's magnet specific heat in effect, E23) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
(the C17 and C95 slider corner, decision A4-4) and `e22_moves_no_differential_case` (every corpus case, Rust-only rows included).
```

with:

```markdown
(the C17 and C95 slider corner, decision A4-4) and `e22_moves_no_differential_case` (every corpus case, Rust-only rows included); for E23 `e23_leaves_every_default_cell_bit_for_bit`, `e23_moves_only_a_ring_with_a_grade_specific_heat` (C141 moves by the ring's mass times its specific heat's difference from C138; the wiring to C138), `e23_reproduces_the_report_s_thermal_table`, `e23_moves_no_verdict_and_the_peak_by_hundredths_of_a_degree` (every ordered grade pair, both hubs, both bases; C141 by each ring's mass times its specific heat's difference from C138, mixed rings included) and `e23_moves_no_differential_case`; `e22_and_e23_change_disjoint_cells` (decision 4).
```

- [ ] **Step 4: Format, bless the input schema and run the tests**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml
cd C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs && MAGCOUPLING_BLESS=1 cargo test --test schema 2>&1 | grep "test result"
git -C C:/Users/Cole/source/repos/lsim-mag-a4 diff --stat -- magcoupling-rs/tests/data/input_schema.json
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: `test result: ok. 7 passed`; the schema diff is `1 file changed, 1 insertion(+), 1 deletion(-)` (C138's `help` line only); then every binary `ok`, in order: unit tests `176 passed`; `tests\assumptions.rs` 8 passed; `tests\deviations.rs` 76 passed; `tests\differential.rs` 19 passed; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 12 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 12 passed; `tests\schema.rs` 7 passed; `tests\sizing.rs` 31 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. (Read the counts as offsets from Task 0's if Task 0 recorded different ones.) The drift guard takes both arms of C141's new `cases` (one specific heat: the default; two: the "grade mode" and ferrite augmentations).

- [ ] **Step 5: Lints, wasm and the differential data**

Run:

```bash
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
cd C:/Users/Cole/source/repos/lsim-mag-a4/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: each cargo line ends `Finished ...` (no warning, no error); then `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`.

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task4.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task4.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 add magcoupling-rs/src/engine/deviations.rs magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/explain/records/slip_heating.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a4 commit -F - <<'EOF'
feat(magcoupling-rs): E23, each ring's grade specific heat in the heat capacity

C141 prices each ring's magnets at its grade's specific heat (ferrite, SmCo,
bonded NdFeB; A-4 decisions 4, 5, 8): m_inner c_inner + m_outer c_outer, and
C110 x c bit for bit when both rings share it. Each ring's mass and specific
heat in effect are Rust-only results the explorer reads; C138 keeps its label
and range, its help reworded (decision 10). No verdict moves, the peak by at
most 0.031 C; no sintered NdFeB design and no differential case moves.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 5: E24, the cure margin from the lower single-ring onset

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5): a physics correction to the verdict's cure margin and its explain records. Physics reviewer: the A-4 report section 2.8.

Decision 2 (A): "Land C24 = MIN(C59_inner, C59_outer) − cure as its own correction (proposed E24, gated on E20) in the same change as E21." Under E20 the block, C24 included, is the governing ring's (the lower magnet limit C60), and that ring can have the higher single-ring onset; E24 takes the lower of both rings' own C59 (`py_min`, the inner ring's on a tie, so identical rings stay bit for bit). The block's onsets C56 to C59 and the summary C7, C8 stay the governing ring's. Each ring's single-ring onset becomes a Rust-only result, with a record each reading the ring's offset from Task 1, and C24's record takes their minimum. The registry entry's cells and probe are decision A4-1's; `e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure` pins the slider setting at which E24 turns a verdict, which is why C25 is registered. Task 5 also updates one line of Task 1's test `e21_moves_only_pairs_with_a_negative_offset_and_never_raises_a_limit`: the user basis's pair count goes from the report's 150 to 156, because with E24 in every correction a clamp on a ring that does not govern lowers that ring's single-ring onset, which E24 reads, so C24 moves in six more pairs (N50M beside N42M, and N38UH or N35EH beside Recoma 20, each in both orders; only C24 moves there).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs` (module doc, `depends_on` docs, `DeviationId::E24`, the E24 entry)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs` (module doc, the cure margin, C24 help, two Rust-only results, the E20 test's fixture, a unit test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/demagnetization.rs` (two new records)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/slip_heating.rs` (C24 rewritten)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs` (six tests, and the user-basis pair count of Task 1's pair sweep)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `DemagResults::inner_calibration_offset_C`, `outer_calibration_offset_C` and the offset records (Task 1); `compat::py_min`; the test helpers of Task 1 (`grade_pairs`, `part_pairs`, `part_rings`, `moved_cells`, `assert_a4_defaults_unchanged`, `assert_no_differential_case_moves_between` among them) and Task 3 (`HUBS`).
- Produces (read by Task 7): Rust-only `DemagResults::inner_onset_single_ring_C`, `outer_onset_single_ring_C: f64` (`temperature.demag.inner_onset_single_ring_C`, `outer_...`); `DeviationId::E24` (`ALL` has 24 ids).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
        DeviationId::E22,
        DeviationId::E23,
    ]
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
```

with:

```rust
        DeviationId::E22,
        DeviationId::E23,
        DeviationId::E24,
    ]
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
        // (pairs with a moved cell, governing-ring flips, verdict changes): the report's table.
        let want = match basis {
            "registry" => (214, 71, 0),
            _ => (150, 39, 0),
        };
```

with:

```rust
        // (pairs with a moved cell, governing-ring flips, verdict changes): the report's table,
        // but for six more pairs on the user basis: there E24, in every correction, reads the
        // single-ring onset of a ring E21 clamps that does not govern, so C24 moves.
        let want = match basis {
            "registry" => (214, 71, 0),
            _ => (156, 39, 0),
        };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
#[test]
fn e24_leaves_every_default_cell_bit_for_bit() {
    // Identical rings: both single-ring onsets are the governing ring's, bit for bit.
    assert_a4_defaults_unchanged(DeviationId::E24);
}

#[test]
fn e24_alone_moves_no_differential_case() {
    // The double gate, as for E21 (A-4 report section 2.6): E24 refines E20, so on its own it
    // is NONE on every corpus case, every result row bit for bit.
    let alone = (Deviations::NONE, Deviations::only(DeviationId::E24));
    assert_no_differential_case_moves_between(&[alone], "E24 alone");
}

#[test]
fn e24_reproduces_the_report_s_cure_margins() {
    // A-4 report section 2.8 (with E21, so on the user basis): the shown C24 before E24 and
    // the lower ring's after it, in either ring order.
    let pairs: [(&str, &str, f64, f64); 7] = [
        ("N38UH", "SmCo_2_17_26", 205.0, 142.6),
        ("N35EH", "SmCo_2_17_26", 205.0, 165.8),
        ("SmCo_2_17_26", "SmCo_1_5_20", 205.0, 169.7),
        ("SmCo_2_17_30", "N38UH", 169.4, 142.6),
        ("SmCo_2_17_30", "N35EH", 169.4, 165.8),
        ("SmCo_2_17_30", "N42H", 169.4, 90.83),
        ("SmCo_2_17_30", "N42SH", 169.4, 121.4),
    ];
    let [_, (_, before, after)] = a4_bases(DeviationId::E24);
    for (a, b, shown, lower) in pairs {
        for (inner, outer) in [(a, b), (b, a)] {
            let rings = grade_rings(inner, outer);
            let what = format!("{inner}/{outer}");
            let cure = |dev| cells_with(&rings, dev)["Temperature design!C24"].clone();
            assert_sig4(&what, &cure(before), shown);
            assert_sig4(&what, &cure(after), lower);
        }
    }
    // The library parts: B842 (N42) beside B842-N52 (A-4 report probe 3), either order.
    for (inner, outer) in [("B842", "B842-N52"), ("B842-N52", "B842")] {
        let rings = part_rings(inner, outer);
        let what = format!("{inner}/{outer}");
        assert_sig4(
            &what,
            &cells_with(&rings, before)["Temperature design!C24"],
            50.09,
        );
        assert_sig4(
            &what,
            &cells_with(&rings, after)["Temperature design!C24"],
            47.83,
        );
    }
}

#[test]
fn e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict() {
    // Every ordered pair of grades and of library parts, alike or not, with each of the four
    // adhesives, both hubs, both bases: C24 is the lower of the two rings' single-ring onsets
    // less the cure temperature, never above the governing ring's, bit for bit where the
    // governing ring has the lower onset; no verdict moves at the default sliders (A-4 report
    // section 2.8: the lowest single-ring C24 of any grade is BCN-19's 12.26 C, above 10 C;
    // a slider setting that turns one is the next test's).
    use magcoupling::engine::grades::GRADES;
    use magcoupling::engine::library::MAGNET_LIBRARY;
    // (label, rings, whether the rings are library parts)
    let mut configs: Vec<(String, Overrides, bool)> = Vec::new();
    for (label, rings) in grade_pairs() {
        configs.push((label, rings, false));
    }
    for g in &GRADES {
        let rings = grade_rings(g.id, g.id).to_vec();
        configs.push((format!("{} both rings", g.id), rings, false));
    }
    for (label, rings) in part_pairs() {
        configs.push((label, rings, true));
    }
    for spec in &MAGNET_LIBRARY {
        let rings = both_rings(spec.part).to_vec();
        configs.push((format!("{} both rings", spec.part), rings, true));
    }
    for adhesive in 1..=4 {
        for (hub, backiron) in HUBS {
            for (basis, before, after) in a4_bases(DeviationId::E24) {
                // How many grade configurations and part configurations C24 moves in.
                let mut moved = [0, 0];
                for (label, rings, part) in &configs {
                    let mut o = rings.clone();
                    o.push(("coupling.backiron", Value::Int(backiron)));
                    o.push(("temperature.adhesive.selected", Value::Int(adhesive)));
                    let what = format!("{label}, adhesive {adhesive}, {hub}, {basis}");
                    let (b, a) = (results_with(&o, before), results_with(&o, after));
                    let (d, cure) = (&a.temperature.demag, a.temperature.adhesive.cure_C);
                    let lower = d.inner_onset_single_ring_C.min(d.outer_onset_single_ring_C);
                    let (sb, sa) = (&b.temperature.summary, &a.temperature.summary);
                    assert_eq!(sa.cure_margin_C, lower - cure, "{what}");
                    assert!(sa.cure_margin_C <= sb.cure_margin_C, "{what}");
                    if d.onset_single_ring_C == lower {
                        assert_same_results(&b, &a, &what);
                    }
                    assert_eq!(sb.verdict, sa.verdict, "{what}");
                    if sa.cure_margin_C != sb.cure_margin_C {
                        moved[usize::from(*part)] += 1;
                    }
                }
                // The same with every adhesive and hub: the rings decide which onset is lower.
                let want = if basis == "registry" {
                    [20, 0]
                } else {
                    [26, 28]
                };
                assert_eq!(moved, want, "adhesive {adhesive}, {hub}, {basis}");
            }
        }
    }
    let bonded = grade_rings("Bonded_NdFeB_BCN19", "Bonded_NdFeB_BCN19");
    assert_sig4(
        "BCN-19",
        &cells_with(&bonded, Deviations::ALL)["Temperature design!C24"],
        12.26,
    );
}

#[test]
fn e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure() {
    // Decision A4-1: C25 is registered because E24 can turn the verdict, though at no default
    // slider setting (the test above). Recoma 30 beside N42H, in either order, bonded with
    // EA 9514 (cured at 120 C), with the knee fraction C46 at 0.95 and the design margin C51
    // at 0: Recoma 30 governs and its single-ring onset gives C24 = 75.40 C, so the verdict
    // reads OK; N42H's single-ring onset, 113.3 C, is below the cure, so with E24 C24 is
    // -6.719 C and the verdict reads CHECK. Only C24 and C25 move, on both bases.
    for (inner, outer) in [("SmCo_2_17_30", "N42H"), ("N42H", "SmCo_2_17_30")] {
        let mut o = grade_rings(inner, outer).to_vec();
        o.extend([
            ("temperature.adhesive.selected", Value::Int(2)),
            ("temperature.demag.knee_fraction", Value::Num(0.95)),
            ("temperature.demag.design_margin_C", Value::Num(0.0)),
        ]);
        for (basis, before, after) in a4_bases(DeviationId::E24) {
            let what = format!("{inner}/{outer}, {basis}");
            let (b, a) = (results_with(&o, before), results_with(&o, after));
            let (sb, sa) = (&b.temperature.summary, &a.temperature.summary);
            assert_sig4(&what, &Value::Num(sb.cure_margin_C), 75.40);
            assert_sig4(&what, &Value::Num(sa.cure_margin_C), -6.719);
            assert_eq!(
                (sb.verdict.as_str(), sa.verdict.as_str()),
                (VERDICT_OK, VERDICT_CHECK),
                "{what}"
            );
            assert_eq!(
                moved_cells(&b, &a),
                ["Temperature design!C24", "Temperature design!C25"],
                "{what}"
            );
        }
    }
}

#[test]
fn e24_with_e21_gives_the_lower_ring_s_margin_on_probe_4() {
    // A-4 report probe 4 (N38UH inside, Recoma 26 outside): E21 moves the block to Recoma 26,
    // whose C24 is 205.0 C; with E24 the margin is N38UH's, 142.57635668906815 C (the note
    // under decision 2 A).
    let rings = grade_rings("N38UH", "SmCo_2_17_26");
    let base = Deviations::NONE
        .with(DeviationId::E20)
        .with(DeviationId::E21);
    let cure = |dev| cells_with(&rings, dev)["Temperature design!C24"].clone();
    assert_eq!(cure(base), Value::Num(205.01249772124206));
    assert_eq!(
        cure(base.with(DeviationId::E24)),
        Value::Num(142.57635668906815)
    );
}

#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

In the unit tests of `temperature.rs` (the E20 test's fixture again; then a new test):

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            want.outer_cold_limit_C = cold.outer_cold_limit_C;
            want.outer_calibration_offset_C = cold.outer_calibration_offset_C;
```

with:

```rust
            want.outer_cold_limit_C = cold.outer_cold_limit_C;
            want.outer_calibration_offset_C = cold.outer_calibration_offset_C;
            want.outer_onset_single_ring_C = cold.outer_onset_single_ring_C;
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            r.demag.outer_calibration_offset_C = 0.0;
            r
```

with:

```rust
            r.demag.outer_calibration_offset_C = 0.0;
            r.demag.outer_onset_single_ring_C = 0.0;
            r
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    #[test]
    fn each_ring_shows_its_own_block() {
```

with:

```rust
    #[test]
    fn e24_takes_the_lower_single_ring_onset_of_both_rings() {
        // Addendum A-4 decision 2: under E20 the block is the governing ring's, and that ring
        // need not have the lower single-ring onset C59. Recoma 26 inside (rated 350 C)
        // governs once E21 clamps its offset, while the N38UH ring outside has the lower C59:
        // E24 takes the outer ring's; without E24 the margin is the governing ring's; without
        // E20 there is one ring and E24 is inert.
        let base = Deviations::only(DeviationId::E20).with(DeviationId::E21);
        let ti = TemperatureInputs::default();
        let cure = selected_adhesive(ti.adhesive.selected).cure_C;
        let mut k = links();
        k.inner_grade = crate::engine::grades::grade("SmCo_2_17_26");
        k.br20_T = 1.0;
        k.inner_alpha_br = -0.00035;
        k.tmax_lib_C = NumOrText::Num(350.0);
        k.outer_grade = crate::engine::grades::grade("N38UH");
        k.outer_br20_T = 1.22;
        k.outer_tmax_lib_C = NumOrText::Num(180.0);
        let before = compute(&ti, &k, base);
        let after = compute(&ti, &k, base.with(DeviationId::E24));
        assert_eq!(before.demag.demag_ring, RING_INNER);
        let d = &after.demag;
        assert!(d.outer_onset_single_ring_C < d.inner_onset_single_ring_C);
        assert_eq!(
            d.inner_onset_single_ring_C,
            before.demag.onset_single_ring_C
        );
        assert_eq!(
            before.summary.cure_margin_C,
            d.inner_onset_single_ring_C - cure
        );
        assert_eq!(
            after.summary.cure_margin_C,
            d.outer_onset_single_ring_C - cure
        );
        assert_eq!(
            compute(&ti, &k, Deviations::only(DeviationId::E24)),
            run(&ti, &k),
            "E24 refines E20"
        );
    }

    #[test]
    fn each_ring_shows_its_own_block() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test deviations 2>&1 | grep -E "^error" | sort | uniq -c
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --lib 2>&1 | grep -E "^error" | sort | uniq -c
```

Expected: compile errors only, the first command:

```text
1 error: could not compile `magcoupling-rs` (test "deviations") due to 9 previous errors
7 error[E0599]: no variant or associated item named `E24` found for enum `DeviationId` in the current scope
1 error[E0609]: no field `inner_onset_single_ring_C` on type `&DemagResults`
1 error[E0609]: no field `outer_onset_single_ring_C` on type `&DemagResults`
```

and the second:

```text
1 error: could not compile `magcoupling-rs` (lib test) due to 10 previous errors
2 error[E0599]: no variant or associated item named `E24` found for enum `deviations::DeviationId` in the current scope
3 error[E0609]: no field `inner_onset_single_ring_C` on type `&DemagResults`
2 error[E0609]: no field `outer_onset_single_ring_C` on type `&DemagResults`
3 error[E0609]: no field `outer_onset_single_ring_C` on type `DemagResults`
```

- [ ] **Step 3: Implement E24 and its explain records**

The probe's values are the engine's at full precision on E24's registry basis `NONE.with(E20)`: 169.87720052918982 C (the governing Recoma 30 ring's, the report's "shown before") and 90.82967381692148 C (N42H's, the report's 90.83).

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! E23 (decisions 4, 5, 8, 10: each ring's grade specific heat), approved on
//! 2026-10-01. Everything else stays workbook-exact. This registry is the one
```

with:

```rust
//! E23 (decisions 4, 5, 8, 10: each ring's grade specific heat) and E24
//! (decision 2: the cure margin from the lower ring), approved on 2026-10-01.
//! Everything else stays workbook-exact. This registry is the one
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//!   E21 refines E20: without it there is one calibration, the inner ring's).
```

with:

```rust
//!   E21 and E24 refine E20: without it there is one ring, the inner one).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    /// sides (decision 15). Empty for all but E15, E16 and E17 (on E9) and
    /// E21 (on E20).
```

with:

```rust
    /// sides (decision 15). Empty for all but E15, E16 and E17 (on E9) and
    /// E21 and E24 (on E20).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    E22,
    E23,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 23] = [
```

with:

```rust
    E22,
    E23,
    E24,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 24] = [
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        DeviationId::E22,
        DeviationId::E23,
    ];
```

with:

```rust
        DeviationId::E22,
        DeviationId::E23,
        DeviationId::E24,
    ];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
                    CellChange {
                        cell: "Temperature design!C180",
                        workbook: Literal::Num(65.86630657629901),
                        corrected: Literal::Num(65.87048331422399),
                    },
                ],
            },
        ],
    },
];
```

with:

```rust
                    CellChange {
                        cell: "Temperature design!C180",
                        workbook: Literal::Num(65.86630657629901),
                        corrected: Literal::Num(65.87048331422399),
                    },
                ],
            },
        ],
    },
    Deviation {
        id: DeviationId::E24,
        title: "Under E20 the cure margin is the governing ring's, which need not have the lower single-ring demagnetization onset",
        class: DeviationClass::Engine,
        approval: Approval::AddendumA4 { decisions: &[2] },
        depends_on: &[DeviationId::E20],
        status: DeviationStatus::Applied,
        cells: &["Temperature design!C24", "Temperature design!C25"],
        corrected_formula: "C24 = MIN(C59_inner, C59_outer) - C77: the cure margin is the lower of both rings' \
            single-ring onsets (each ring's own block, E20) less the cure temperature. The governing ring is the one with \
            the lower magnet limit C60, and its C59 can be the higher one, so the shown C24 overstated the real minimum \
            (by up to 79.05 C under E20 alone, Recoma 30 beside N42H, and up to 62.44 C once E21 moves the governing ring, \
            N38UH beside Recoma 26). Gated on E20 (without it only the inner ring is checked); the block's onsets C56 to \
            C59 and the summary C7, C8 stay the governing ring's. Each ring's single-ring onset is a Rust-only result. \
            The verdict C25 reads C24; no verdict moves at default inputs (the lowest single-ring C24 of any grade or part \
            is 12.26 C, BCN-19), but one can at a hot cure: Recoma 30 beside N42H with EA 9514 (120 C), the knee fraction \
            C46 at 0.95 and the design margin C51 at 0 turns from OK to CHECK (N42H's single-ring onset, 113.3 C, is \
            below the cure).",
        workbook_input_defaults: &[],
        workbook_help: &[("temperature.summary.cure_margin_C", "")],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "Recoma 30 inner, N42H outer (manual dimensions): Recoma 30 has the lower magnet limit and governs, \
                but N42H has the lower single-ring onset (A-4 report section 2.8, the E20 defect before E21: 79.05 C)",
            inputs: &[
                ("coupling.magnets.part_inner", Literal::Text("")),
                ("coupling.magnets.part_outer", Literal::Text("")),
                (
                    "coupling.magnets.grade_inner",
                    Literal::Text("SmCo_2_17_30"),
                ),
                ("coupling.magnets.grade_outer", Literal::Text("N42H")),
            ],
            expect: &[CellChange {
                cell: "Temperature design!C24",
                workbook: Literal::Num(169.87720052918982),
                corrected: Literal::Num(90.82967381692148),
            }],
        }],
    },
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! and the screens compare the size of the shear) and E23 (C141 prices each
//! ring's magnets at its grade's specific heat, [`magnet_specific_heat_J_kgK`]).
```

with:

```rust
//! and the screens compare the size of the shear), E23 (C141 prices each
//! ring's magnets at its grade's specific heat, [`magnet_specific_heat_J_kgK`])
//! and E24 (on top of E20: the cure margin C24 is the lower of both rings'
//! single-ring onsets less the cure temperature).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            cure_margin_C: f64 => out("°C", "Cure margin below the single-ring demag onset", "", "Temperature design!C24"),
```

with:

```rust
            cure_margin_C: f64 => out("°C", "Cure margin below the single-ring demag onset",
                "Correction E24: the lower of both rings' single-ring onsets (temperature.demag.inner_onset_single_ring_C, outer_onset_single_ring_C) less the cure temperature C77; the onset C59 shows the governing ring's.",
                "Temperature design!C24"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            outer_calibration_offset_C: f64 => out_rust_only("°C", "Outer ring: calibration offset applied",
                "Plan A-4: as the inner ring's, for the outer ring."),
```

with:

```rust
            outer_calibration_offset_C: f64 => out_rust_only("°C", "Outer ring: calibration offset applied",
                "Plan A-4: as the inner ring's, for the outer ring."),
            inner_onset_single_ring_C: f64 => out_rust_only("°C", "Inner ring: single-ring onset during an adhesive cure",
                "Plan A-4: the inner ring's own C59, whichever ring governs (+inf with a positive beta). Correction E24 takes the lower of both rings' for the cure margin C24."),
            outer_onset_single_ring_C: f64 => out_rust_only("°C", "Outer ring: single-ring onset during an adhesive cure",
                "Plan A-4: as the inner ring's, for the outer ring."),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        inner_calibration_offset_C: inner.offset,
        outer_calibration_offset_C: outer_block.offset,
    };
```

with:

```rust
        inner_calibration_offset_C: inner.offset,
        outer_calibration_offset_C: outer_block.offset,
        // Plan A-4: each ring's single-ring onset, which E24 compares.
        inner_onset_single_ring_C: inner.onsets[3],
        outer_onset_single_ring_C: outer_block.onsets[3],
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    // ---- summary
    let cure_margin = on_cu - sel.cure_C;
```

with:

```rust
    // ---- summary
    // E24 (refines E20; Addendum A-4 decision 2): the cure margin is the lower of both rings'
    // single-ring onsets; the governing ring (the lower magnet limit) can have the higher one.
    // py_min keeps the inner ring's on a tie, so identical rings stay bit for bit.
    let single_ring_onset = if dev.is_on(DeviationId::E20) && dev.is_on(DeviationId::E24) {
        py_min(inner.onsets[3], outer_block.onsets[3])
    } else {
        on_cu
    };
    let cure_margin = single_ring_onset - sel.cure_C;
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/demagnetization.rs`, replace:

```rust
                       - {temperature.demag.outer_calibration_offset_C} - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m}"#).corrected(&[E20]),
```

with:

```rust
                       - {temperature.demag.outer_calibration_offset_C} - {temperature.demag.design_margin_C})
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m}"#).corrected(&[E20]),
    // A ring's single-ring onset (its own C59): the field of the ring alone on its carrier
    // during an adhesive cure; E24 compares both rings'.
    record("temperature.demag.inner_onset_single_ring_C", "ϑ_{sr,i}",
        "cases({temperature.demag.inner_beta_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_single_ring_kA_m},
                                 [H_k] · abs({temperature.demag.inner_beta_per_C}) - {temperature.demag.h_rev_single_ring_kA_m} · abs({model.inner_alpha_br_per_C}))
                       - {temperature.demag.inner_calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m}").corrected(&[E20]),
    record("temperature.demag.outer_onset_single_ring_C", "ϑ_{sr,o}",
        "cases({temperature.demag.outer_beta_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_single_ring_kA_m},
                                 [H_k] · abs({temperature.demag.outer_beta_per_C}) - {temperature.demag.h_rev_single_ring_kA_m} · abs({model.outer_alpha_br_per_C}))
                       - {temperature.demag.outer_calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m}").corrected(&[E20]),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/slip_heating.rs`, replace:

```rust
use crate::engine::deviations::DeviationId::{E5, E8, E9, E12, E15, E17, E23};
```

with:

```rust
use crate::engine::deviations::DeviationId::{E5, E8, E9, E12, E15, E17, E20, E23, E24};
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/records/slip_heating.rs`, replace:

```rust
    record("temperature.summary.cure_margin_C", "Δϑ_{cure}", "{temperature.demag.onset_single_ring_C} - {temperature.adhesive.cure_C}"),
```

with:

```rust
    // E24: the lower of both rings' single-ring onsets, whichever ring governs.
    record("temperature.summary.cure_margin_C", "Δϑ_{cure}",
        "min({temperature.demag.inner_onset_single_ring_C}, {temperature.demag.outer_onset_single_ring_C}) - {temperature.adhesive.cure_C}")
        .corrected(&[E20, E24]),
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
of its section 8 that each entry's `approval` cites) and E21 to E23 in the Addendum A-4
```

with:

```markdown
of its section 8 that each entry's `approval` cites) and E21 to E24 in the Addendum A-4
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
probe one on top of the corrections it refines (`depends_on`: E15 to E17 on E9,
E21 on E20; decision 15), or to start from the workbook's defaults.
```

with:

```markdown
probe one on top of the corrections it refines (`depends_on`: E15 to E17 on E9,
E21 and E24 on E20; decision 15), or to start from the workbook's defaults.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
no verdict moves, and the peak by at most 0.031 °C | Addendum A-4 decisions 4, 5, 8, 10 |
```

with:

```markdown
no verdict moves, and the peak by at most 0.031 °C | Addendum A-4 decisions 4, 5, 8, 10 |
| E24 | Temperature design!C24 (feeds the verdict C25) | on top of E20, C24 = the governing ring's C59 − C77; the governing ring (the lower magnet limit C60) can have the higher single-ring onset, so C24 overstated the real minimum (Recoma 30 beside N42H: by 79.05 °C; with E21, N38UH beside Recoma 26: by 62.44 °C) | C24 = MIN(C59 inner, C59 outer) − C77, each ring's own onset a Rust-only result (`temperature.demag.inner_onset_single_ring_C`, `outer_onset_single_ring_C`); the block's onsets stay the governing ring's, and C24's help says so. Identical rings are bit for bit; no verdict moves at default inputs (the lowest single-ring C24 of any grade or part is 12.26 °C, BCN-19), but one can at a hot cure: Recoma 30 beside N42H with EA 9514 (120 °C), C46 at 0.95 and C51 at 0 turns from OK to CHECK (N42H's single-ring onset, 113.3 °C, is below the cure) | Addendum A-4 decision 2 |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E23 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A, E21 to E23 from Addendum A-4), probes, and the `Deviations` switch |
```

with:

```markdown
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E24 (all applied: E1 to E14 from the M1 audit, E15 to E20 from Addendum A, E21 to E24 from Addendum A-4), probes, and the `Deviations` switch |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
each ring's applied calibration offset, E21, plan A-4; the magnet CTE in effect, E22; each ring's magnet specific heat in effect, E23) |
```

with:

```markdown
each ring's applied calibration offset, E21, plan A-4; the magnet CTE in effect, E22; each ring's magnet specific heat in effect, E23; each ring's single-ring onset, E24) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
`e23_moves_no_differential_case`; `e22_and_e23_change_disjoint_cells` (decision 4).
```

with:

```markdown
`e23_moves_no_differential_case`; `e22_and_e23_change_disjoint_cells` (decision 4); for E24 `e24_leaves_every_default_cell_bit_for_bit`, `e24_alone_moves_no_differential_case` (the double gate), `e24_reproduces_the_report_s_cure_margins`, `e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict` (every ordered pair of grades and of parts, all four adhesives, both hubs, both bases), `e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure` (decision A4-1: why C25 is registered) and `e24_with_e21_gives_the_lower_ring_s_margin_on_probe_4`.
```

- [ ] **Step 4: Format and run the tests**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: every binary `ok`, in order: unit tests `177 passed`; `tests\assumptions.rs` 8 passed; `tests\deviations.rs` 82 passed; `tests\differential.rs` 19 passed; `tests\explain.rs` 12 passed, 2 ignored; `tests\grades.rs` 12 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 12 passed; `tests\schema.rs` 7 passed; `tests\sizing.rs` 31 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. (Read the counts as offsets from Task 0's if Task 0 recorded different ones.)

- [ ] **Step 5: Lints, wasm and the differential data**

Run:

```bash
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets -- -D warnings 2>&1 | tail -1
cargo clippy --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings 2>&1 | tail -1
cargo check --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --target wasm32-unknown-unknown --lib 2>&1 | tail -1
cd C:/Users/Cole/source/repos/lsim-mag-a4/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: each cargo line ends `Finished ...` (no warning, no error); then `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`.

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task5.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task5.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task5.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 add magcoupling-rs/src/engine/deviations.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/explain/records/demagnetization.rs magcoupling-rs/src/engine/explain/records/slip_heating.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a4 commit -F - <<'EOF'
feat(magcoupling-rs): E24, the cure margin from the lower ring

Under E20 the cure margin C24 was the governing ring's, which need not have the
lower single-ring onset (up to 79.05 C over the minimum, 62.44 C after E21).
E24 (on E20, A-4 decision 2) takes MIN(C59 inner, C59 outer) - C77; each ring's
single-ring onset is a Rust-only result the explorer reads. Registry: cells C24
and C25, probe Recoma 30 beside N42H (decision A4-1). No verdict moves at default
inputs; a hot cure with a lowered knee fraction and design margin turns one
(pinned); identical rings are bit for bit. Task 1's pair sweep counts 156 user
pairs now, the report's 150 plus six where only C24 moves.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 6: Physics review of the changed records and notes, and the notes' sign-off

**Model:** session model throughout (omit `model`, comment `// session model: physics`; CLAUDE.md section 5): the reviewer is a fresh context that did not write Tasks 1 to 5; the spec's accuracy gate names it.

The corrections changed what thirteen equation records state (eight new, five rewritten) and redrafted two teaching notes (decision A4-2). Each review step gives the reviewer the rendered records or the notes and the report sections they come from; the reviewer answers APPROVED or lists findings. A finding in a record is fixed in its formula (the drift guard must stay green); a finding in a note is fixed in its sentences (2 to 6, sources kept) and sent back to the same reviewer. A note the reviewer will not approve stays a draft, hidden by `note_for`, and its id comes out of the sign-off script's list, which keeps the release gate red (name it in the commit body and in Task 7's memory entry).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs` (each approved note's `review`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md` (how many notes are signed off)

**Interfaces:**
- Consumes: Tasks 1, 4 and 5's records; the two Draft notes of Tasks 1 and 3; `tests/explain.rs`'s ignored `review_sheet` and `release_notes_are_reviewed`.
- Produces: both notes `Review::Reviewed { reviewer, date, record }` (if approved), so `notes::note_for` shows them and the M4 release gate passes.

- [ ] **Step 1: The release gate is red, naming the two drafts**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test explain release_notes_are_reviewed -- --ignored 2>&1 | grep -E "not yet reviewed|test result"
```

Expected: `not yet reviewed: ["demagnetization", "a5.cte_mismatch_with_magnets"]` in the panic line, then `test result: FAILED. 0 passed; 1 failed`.

- [ ] **Step 2: Review the changed records**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E 'calibration_offset_C|_magnet_limit_C|onset_single_ring_C|cure_margin_C|magnets_inner_g|magnets_outer_g|magnet_c_J_kgK|heat_capacity_J_K' > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/a4-review-sheet.md; wc -l < C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/a4-review-sheet.md
```

Expected: `15` (the 13 records, plus the unchanged `onset_single_ring_C` and `cold_onset_single_ring_C`, which the pattern also matches).

Dispatch the physics reviewer (session model) with the prompt: "Review these equation records of the magcoupling calculator (`C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/a4-review-sheet.md`: result path, workbook cell, the rendered formula, the corrections it names) against the approved Addendum A-4 report `C:/Users/Cole/source/repos/lsim-mag-a4/docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md` and the engine (`src/engine/temperature.rs` `ring_demag` and the summary and thermal network of `compute`, `src/engine/model.rs` `mass_estimate`). E21 (section 2.1): each ring's offset is MAX(C49 − C47, 0), 0 with a positive beta or without a rating, and the limits and single-ring onsets subtract it. E24 (section 2.8): C24 is the lower of both rings' single-ring onsets less the cure temperature. E23 (sections 3.1, 4.4): C141's magnet term is m_inner c_inner + m_outer c_outer, C110 · c when both share c. For each record: is the formula the engine's, are the symbols plain and distinct, does it name the corrections it embodies? Answer APPROVED, or list each finding with the record path and the fix." Record the answer; fix any finding in its record and re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test explain` until `test result: ok. 12 passed; 0 failed; 2 ignored`.

- [ ] **Step 3: Review the two redrafted notes**

Dispatch the physics reviewer (session model) with the prompt: "Review these teaching notes of the magcoupling calculator (`C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs`) for physics accuracy at Physics 2 level against the sources each cites: `demagnetization` (its sentence on the onset shift, rewritten for correction E21: the A-4 report section 2.1 and decision 1; the rest of the note was reviewed before and is unchanged) and `a5.cte_mismatch_with_magnets` (its first and third sentences and its new fourth, rewritten for correction E22: the report sections 3.2, 3.3 and 4.7 IN1, decision 9; warning rule 6 in `src/engine/warnings.rs` and `temperature::bond_plane_cte_per_C`). Every sentence true and plainly worded, every number the engine's or the report's, the sources cited. Answer APPROVED per note, or list each finding with the note id and the fix." Record each answer; fix any finding as the intro says, then run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --lib notes 2>&1 | grep "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test explain notes_link 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed` (the notes unit tests), then `test result: ok. 1 passed`.

- [ ] **Step 4: Sign the approved notes off**

The script records the reviewer, today's date and this review; remove from `APPROVED` any id the reviewer did not approve.

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import datetime
import pathlib

# The ids the physics reviewer approved in Steps 2 and 3. Remove any id the reviewer did not
# approve: that note stays a draft (hidden by notes::note_for) and the release gate stays red.
APPROVED = ["demagnetization", "a5.cte_mismatch_with_magnets"]
REVIEWER = "physics reviewer (session model)"
RECORD = (
    "plan A-4 Task 6 physics review of the E21 and E22 sentences against "
    "docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md (sections 2.1, 3.3, 4.7): "
    "the sign-off commit's body (what was checked, any fixes); the note's other sentences keep their earlier review"
)
today = f"{datetime.date.today():%Y-%m-%d}"
p = pathlib.Path("C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/src/engine/explain/notes.rs")
s = p.read_text(encoding="utf-8")
draft = "        review: Review::Draft,\n"
for note_id in APPROVED:
    start = s.index(f'        id: "{note_id}",\n')
    at = s.index(draft, start)
    assert s.find("        id: ", start + 1, at) == -1, f"{note_id}: no draft review line before the next note"
    reviewed = (
        "        review: Review::Reviewed {\n"
        f'            reviewer: "{REVIEWER}",\n'
        f'            date: "{today}",\n'
        f'            record: "{RECORD}",\n'
        "        },\n"
    )
    s = s[:at] + reviewed + s[at + len(draft):]
p.write_text(s, encoding="utf-8", newline="\n")
r = pathlib.Path("C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md")
t = r.read_text(encoding="utf-8")
old = (
    "15 of the 17 are signed off (the demagnetization and expansion-mismatch notes are drafts again "
    "until their E21 and E22 sentences are reviewed, plan A-4;"
)
assert t.count(old) == 1
drafts = [i for i in ("demagnetization", "a5.cte_mismatch_with_magnets") if i not in APPROVED]
new = f"{17 - len(drafts)} of the 17 are signed off (plan A-4 Task 6 re-reviewed the two notes its corrections changed"
t = t.replace(old, new + (f", still drafts: {', '.join(drafts)};" if drafts else ";"))
r.write_text(t, encoding="utf-8", newline="\n")
print(f"{len(APPROVED)} notes signed off on {today}")
EOF
```

Expected: prints `2 notes signed off on <today's date>` when both are approved.

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --lib notes 2>&1 | grep "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test explain release_notes_are_reviewed -- --ignored 2>&1 | grep "test result"
```

Expected: `cargo fmt --check` prints nothing (the script writes rustfmt's layout); `test result: ok. 3 passed`; then `test result: ok. 1 passed` when both notes are approved (with a note left a draft it fails naming it: record that in the commit body and in Task 7's memory entry, and continue).

- [ ] **Step 5: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task6.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 6: Commit**

Write the reviewer's answers (Steps 2 and 3, each finding and its fix) into the commit body below in place of its last paragraph: the notes' review record names this commit's body.

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 add magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a4 commit -F - <<'EOF'
docs(magcoupling-rs): sign off the notes A-4 redrafted after the physics review

The physics reviewer checked the thirteen equation records E21, E23 and E24
changed and the two notes E21 and E22 redrafted (demagnetization,
a5.cte_mismatch_with_magnets) against the A-4 report; approved notes are
Review::Reviewed again, so note_for shows them and the M4 release gate passes.

Review: records APPROVED; demagnetization APPROVED;
a5.cte_mismatch_with_magnets APPROVED (replace with the reviewer's answers and
any fixes).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 7: The README status, docs/ai and the update tracker

**Model:** `sonnet` (the plan gives the exact text; CLAUDE.md section 5: set `model: 'sonnet'`). A `blocked` end or a rejected review escalates every retry to the session model.

The repo rule (`docs/ai/01-meta.yaml`): after file-modifying work, update `02-system.yaml` (invariants), `03-structure.yaml` (module layout), `04-memory.yaml` (resolve the three open items A-4 closes, refresh the A-1 offsets, as the report's section 2.2 asks, and record the report's carry-overs) and `05-update-tracker.md`. If Task 6 left a note a draft, add a line to the memory block naming it.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md` (status, the explorer's plan line)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md`

**Interfaces:**
- Consumes: Tasks 1 to 6.
- Produces: the project state for the next session.

- [ ] **Step 1: Write the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
traceability test over every assumption, and the 17 A4 teaching notes behind a physics-review gate.
Next: the M4 GUI plans.
```

with:

```markdown
traceability test over every assumption, and the 17 A4 teaching notes behind a physics-review gate.
Addendum A-4 (corrections) complete: E21 clamps each ring's rating-calibration offset at 0, E24
takes the cure margin from the ring with the lower single-ring onset, E22 reads the inner ring's
grade CTE in the bond screen and warning rule 6 (the screens compare the size of the shear), and
E23 prices each ring's magnets at its grade's specific heat (A-4 decisions 1 to 10, approved
2026-10-01; per-grade CTE and specific heat in the grade table, cited); at the default sliders
the default design and every sintered NdFeB part are unchanged (E22's size comparison also reads
a sintered NdFeB design's negative shear when C95 is set above the default back iron's C17), and
every equation record still reproduces the engine.
Next: the M4 GUI plans.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/README.md`, replace:

```markdown
`START_HERE` opens them in the spec's order (torque chain, back iron, temperature, demagnetization, slip heating, clamps). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.
```

with:

```markdown
`START_HERE` opens them in the spec's order (torque chain, back iron, temperature, demagnetization, slip heating, clamps). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`; the A-4 corrections (E21 to E24) added
eight records (each ring's calibration offset and single-ring onset, each ring's magnet mass and specific heat) and
rewrote five (the per-ring limits, C50, C24, C141): `docs/superpowers/plans/2026-10-01-magcoupling-addendum-a4-corrections.md`.
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/02-system.yaml`, replace:

```yaml
      registry, M1 audit E1-E14, Addendum A E15-E20). Engine complete (M2), every module but fields3d;
```

with:

```yaml
      registry, M1 audit E1-E14, Addendum A E15-E20, Addendum A-4 E21-E24). Engine complete (M2), every module but fields3d;
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/02-system.yaml`, replace:

```yaml
      Addendum A-3 adds the equation explorer's engine side (evaluable equation records proven by a drift guard, the dependency graph, the A3 traceability test and the A4 teaching notes).
```

with:

```yaml
      Addendum A-3 adds the equation explorer's engine side (evaluable equation records proven by a drift guard, the dependency graph, the A3 traceability test and the A4 teaching notes).
      Addendum A-4 adds four corrections, E21 (the calibration offset clamped at 0), E24 (the cure margin from the lower ring), E22 (the inner ring's grade CTE in the bond screen) and E23 (each ring's grade specific heat).
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/02-system.yaml`, replace:

```yaml
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
```

with:

```yaml
      - Every departure from the workbook is an approved, registered deviation (E1-E24 in deviations.rs; E15-E17 are probed on top of E9 and E21, E24 on top of E20, decision 15). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
      - "Sintered NdFeB (every library part, every grade of family NdFeB) keeps the workbook's magnet CTE (C95) and specific heat (C138): Grade::bond_plane_cte_per_C and specific_heat_J_kgK are None there, so E22's grade CTE and E23 move only a ferrite, SmCo or bonded NdFeB design (bonded has a specific heat but no CTE, A-4 decision 7). E22's size comparison (decision 9) reads every negative shear, so it also moves a sintered NdFeB design whose C95 is set above the default back iron's C17 (a slider corner; every library hub material is at 9.0e-6 /C or more, plan A-4 decision A4-4)."
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/02-system.yaml`, replace:

```yaml
Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes; next: the M4 GUI plans"
```

with:

```yaml
Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes. Addendum A-4 complete: E21 calibration clamp, E24 cure margin from the lower ring, E22 inner-ring grade CTE (size-of-shear screens), E23 per-ring grade specific heat; next: the M4 GUI plans"
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/03-structure.yaml`, replace:

```yaml
A-2 (harmonic set, assumptions, inverse sizing, space claim) and A-3 (equation explorer engine side - evaluable records, drift guard, A3 traceability, A4 notes) complete; M4 infrastructure
```

with:

```yaml
A-2 (harmonic set, assumptions, inverse sizing, space claim), A-3 (equation explorer engine side - evaluable records, drift guard, A3 traceability, A4 notes) and A-4 (corrections E21-E24) complete; M4 infrastructure
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/03-structure.yaml`, replace:

```yaml
    deviations: deviations.rs (REGISTRY of the M1 corrections E1-E14 and the Addendum A corrections E15-E20 (report docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md, section 8 decisions via Approval::Addendum); depends_on (E15-E17 on E9);
```

with:

```yaml
    deviations: deviations.rs (REGISTRY of the M1 corrections E1-E14, the Addendum A corrections E15-E20 (report docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md, section 8 decisions via Approval::Addendum) and the Addendum A-4 corrections E21-E24 (A4_REPORT, section 5 decisions via Approval::AddendumA4); depends_on (E15-E17 on E9, E21 and E24 on E20);
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/03-structure.yaml`, replace:

```yaml
    grades: grades.rs (Addendum A6 grade table GRADES, 17 grades cited per value from docs/analyses/2026-09-30-magcoupling-addendum-a-data.json; grade(id); N42SH keeps the workbook Hcj 1592 and beta -0.005)
```

with:

```yaml
    grades: grades.rs (Addendum A6 grade table GRADES, 17 grades cited per value from docs/analyses/2026-09-30-magcoupling-addendum-a-data.json; grade(id); N42SH keeps the workbook Hcj 1592 and beta -0.005; Addendum A-4 bond_plane_cte_per_C and specific_heat_J_kgK, None for sintered NdFeB, cited from docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json)
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/03-structure.yaml`, replace:

```yaml
mass_estimate and MassResults, 6 result cells)"
```

with:

```yaml
mass_estimate and MassResults, 6 result cells and the Rust-only magnets_inner_g/magnets_outer_g, plan A-4)"
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/03-structure.yaml`, replace:

```yaml
Rust-only demag results hcj20_used/beta_used, demag_ring, cold_ring and the cold side)"
```

with:

```yaml
Rust-only demag results hcj20_used/beta_used, demag_ring, cold_ring and the cold side; plan A-4 - E21's clamp in ring_demag with each ring's calibration offset and single-ring onset, E24's cure margin, bond_plane_cte_per_C (E22, magnet_cte_per_C) and magnet_specific_heat_J_kgK (E23, each ring's magnet specific heat))"
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/04-memory.yaml`, replace:

```yaml
  - "Open (Addendum A-1 final review; for A-2/M4, no change in A-1): E20 applies the workbook's rating calibration (C50 = model onset t_ref minus the rating, temperature.rs ring_demag) to every grade. Where a rating exceeds the grade's uncalibrated t_ref the offset is negative and raises every onset, so the check is less conservative than the uncalibrated model. Measured at 8958deb (manual dimensions, the grade on both rings, other inputs default, every correction on): N52 -9.7 C (approved as-is: report 6.3 row B842-N52, C12 30.73 -> 0.17 C), N50 -6.1 (80 C rating), N50M -7.5, N38UH -7.3, N35EH -3.6, Recoma 26 -18.6 (report R11: Arnold's 350 C 'may be considerably lower at low load line'); large positive offsets the other way: BCN-19 +89.9, Recoma 30 +46, Recoma 20 +255. Candidate: clamp the offset at >= 0 as a later registered correction (user decision)"
  - "Open (Addendum A-1 Task 13 review; not in A-2 scope): the CTE-mismatch warning and E18's bond screen compare the hub with the NdFeB bond-plane CTE (Temperature design C95) for every grade, so a ferrite or SmCo ring reads the NdFeB mismatch. The A6 grade table carries no CTE; a per-grade magnet CTE (sourced, like the rest of the table) would fix both. Needs sourcing and the user's approval as a correction"
  - "Open (Addendum A-2 Task 10 review; same family as the per-grade CTE item): the magnets' heat capacity prices each ring's own mass (decision A2-7) at the NdFeB specific heat for every grade, so a ferrite or SmCo ring's thermal mass is off; a sourced per-grade specific heat would fix it. Needs sourcing and the user's approval as a correction"
```

with:

```yaml
  - "RESOLVED (Addendum A-4, E21, decision 1): each ring's rating-calibration offset is clamped at 0 on top of E20 (C50 = MAX(C49 - C47, 0) per ring). The offsets before the clamp, after decision A2-7 (the grade on both rings, every correction but E21): N52 -9.689 C, N50 -6.138, N50M -7.535, N38UH -7.343, N35EH -3.574, Recoma 26 -60.21, Recoma 30 -0.450 (the seven clamped grades); BCN-19 +94.28, Recoma 20 +217.3, the default N42SH +9.921 unchanged. The A-1 figures quoted here before (8958deb: Recoma 26 -18.6, Recoma 30 +46) predate A2-7's per-grade alpha(Br)"
  - "RESOLVED (Addendum A-4, E22, decisions 4-7, 9, 10): the mismatch screen and warning rule 6 read the inner ring's grade CTE across the magnetization (Y30 10e-6, Recoma 20 14e-6, Recoma 26 and 30 13e-6 /C; sintered and bonded NdFeB keep C95); C106 and C202 compare the size of the shear, the signed values shown"
  - "RESOLVED (Addendum A-4, E23, decisions 4, 5, 8, 10): C141 prices each ring's magnets at its grade's specific heat (Y30 700, Recoma 20 370, Recoma 26 and 30 350, bonded NdFeB 420 single-source; sintered NdFeB keeps C138), C110 x c when both rings share it"
  - "RESOLVED (Addendum A-4, E24, decision 2): under E20 the cure margin C24 is the lower of both rings' single-ring onsets less the cure temperature (it overstated the minimum by up to 79.05 C before E21 and 62.44 C after it); the block's onsets stay the governing ring's"
  - "Open (A-4 report IN3 carry-over, no decision needed now): the mismatch screen reads C97 = 160 GPa for every grade, while the sheets print 140 GPa (Recoma) and 150 GPa (MS-Schramberg ferrite); with E22 the shear at the sheet modulus is under 1 % lower, so 160 is conservative and no reading changes"
  - "Open (A-4 report section 2.1, recorded with E21): the uncalibrated onsets use the stored reverse fields C52-C55, which do not scale with a ring's Br, so for a 1.45 T N52 ring the clamped (uncalibrated) onset can itself be optimistic; M3's live fields per ring remove this"
```

In `C:/Users/Cole/source/repos/lsim-mag-a4/docs/ai/05-update-tracker.md`, replace:

```markdown
# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

```

with:

```markdown
# Update Tracker

Meaningful changes to the project that future sessions should know about.
Reverse chronological (newest at top).

---

## 2026-10-01 — Magcoupling Addendum A-4: corrections E21 to E24
- `magcoupling-rs`: four registered corrections, approved with the A-4 verification report
  (`docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md`, section 5, all ten
  decisions option A), each with an approval record of its own (`Approval::AddendumA4`):
  - E21 (on E20): each ring's rating-calibration offset is clamped at 0, so a rating never raises
    the onsets above the model's own (N52 parts: C12 0.17 -> -9.52 C; Recoma 26: the magnets now
    govern at 101.7 C). C50's help says so; C101's documents that the bond screen's hot swing follows
    C12 (decision 3).
  - E24 (on E20): the cure margin C24 is the lower of both rings' single-ring onsets less the cure
    temperature; under E20 it was the governing ring's, which overstated it by up to 79.05 C.
  - E22: the bond screen and warning rule 6 read the inner ring's grade CTE across the
    magnetization (ferrite and SmCo); the screens compare the size of the shear, which a grade that
    expands more than the hub makes negative (decision 9).
  - E23: the heat capacity prices each ring's magnets at its grade's specific heat (ferrite, SmCo,
    bonded NdFeB); C110 x c, bit for bit, when both rings share it.
- Grade table: cited bond-plane CTE and specific heat per non-NdFeB grade (`tests/grades.rs` against
  the A-4 data file); sintered NdFeB has none and keeps C95 and C138.
- Rust-only results: each ring's applied calibration offset and single-ring onset, the magnet CTE in
  effect, each ring's magnet mass and specific heat in effect. C95, C138, C24, C50 and C101 keep their
  labels; their help is reworded (registered as workbook help).
- Explorer: eight new records and five rewritten (each still proven by the drift guard; the A3
  traceability test unchanged); the demagnetization and expansion-mismatch notes were redrafted for
  E21 and E22 and signed off again after a physics review (plan A-4 Task 6).
- Unchanged: `Deviations::NONE` parity, the differential data, and at the default sliders the default
  design and every sintered NdFeB grade and library part, bit for bit; no A-4 correction moves a
  differential case (E21 and E24 alone are inert, the double gate). On the registry basis a Recoma 20
  ring on a 416 back iron shears the bond at -19.1 MPa, which decision 9 reads as above the 15 MPa lap
  shear (`e22_compares_the_size_of_a_negative_shear`); the same size comparison reads a sintered NdFeB
  design at the slider corner where C95 exceeds the default back iron's C17 (plan decision A4-4,
  `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too`). E24 turns no verdict at the
  default sliders, but Recoma 30 beside N42H with EA 9514's 120 °C cure, C46 at 0.95 and C51 at 0
  turns from OK to CHECK (`e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure`).

```

- [ ] **Step 2: Check the YAML parses and the tests still pass**

`02-system.yaml` does not parse as a whole on `main` (line 14, a plain scalar with `: ` in `architecture_invariants`, outside this plan), so the check parses the two places this task edits there on their own: the `magcoupling:` component mapping and the status line; `03-structure.yaml` and `04-memory.yaml` parse whole. (A `: ` inside a plain multi-line scalar is what it catches.)

Run:

```bash
python - <<'EOF'
import re

import yaml

root = "C:/Users/Cole/source/repos/lsim-mag-a4"
t = open(f"{root}/docs/ai/02-system.yaml", encoding="utf-8").read().split("\n")
start = t.index("  magcoupling:")
end = next(i for i in range(start + 1, len(t)) if re.match(r"(  )?\S", t[i]))
yaml.safe_load("\n".join(t[start:end]))
yaml.safe_load(next(line for line in t if line.startswith('  magcoupling: "M2 complete')))
for f in ("docs/ai/03-structure.yaml", "docs/ai/04-memory.yaml"):
    yaml.safe_load(open(f"{root}/{f}", encoding="utf-8"))
print("yaml ok")
EOF
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --test deviations all_corrections_together_give_the_reviewed_headline 2>&1 | grep "test result"
```

Expected: `yaml ok`, then `test result: ok. 1 passed` (the default headline with every correction on, unedited since A-3).

- [ ] **Step 3: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a4/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task7.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task7.log; grep -c "SKIP gate" C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/target/gate-task7.log
git -C C:/Users/Cole/source/repos/lsim-mag-a4 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a4/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's pytest summary (`1159 passed in ...s`), `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate` line); `git checkout` restores the PNGs the linkage tests rewrite; `cargo fmt --check` prints nothing.

- [ ] **Step 4: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a4 add magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md
git -C C:/Users/Cole/source/repos/lsim-mag-a4 commit -F - <<'EOF'
docs(magcoupling): Addendum A-4 status, invariants and open items

E21 to E24 applied (A-4 decisions 1 to 10): the README status and plan line,
the registry range and the sintered NdFeB invariant in 02-system, the layout in
03-structure, the three open items A-4 resolves with the refreshed offsets and
the report's two carry-overs in 04-memory, and the tracker entry.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```


---

## Self-review record

An independent critic reviewed the previous revision of this plan (replayed on acc0296) and found no blocking defect: the registry against the report, the gates, the bit-for-bit defaults, NONE parity and the differential data all held. It raised one false invariant, which needed a text fix and a decision for the user, and seven smaller items. This revision fixes every one. Then the whole plan was rebuilt from its blocks and replayed again on both bases (Verification record above). Each finding and what changed:

1. **False invariant: E22's size comparison can move a sintered NdFeB design.** With the default back iron, the hub CTE is the Materials!C17 slider (5e-6 to 25e-6 /°C), and C95 reaches 6e-6, so a negative mismatch is reachable without any grade. Reproduced: B842SH on both rings, C17 = 5e-6, C95 = 6e-6, C96 = 3.0 GPa, both bondlines at 0.01 mm. C104 is −22.61 MPa on the registry basis and −22.77 on the user basis, and E22 turns C106 and C202 from "Below" to "Above" on both bases; no other cell moves. Fixed:
   - The wording, everywhere it claimed otherwise: the Architecture paragraph, the "settled" list (the gating moved to a decision), Global Constraints, Review Focus items 3 and 4, Task 3's prose, the comment in `e22_moves_no_differential_case` ("every shear in the corpus is positive"), the shipped comment at the size comparison in `temperature.rs`, E22's `corrected_formula`, the README's E22 row, Task 3's commit body, and Task 7's 02-system invariant, README status line and tracker entry. Each now says that no sintered NdFeB design moves *at the default sliders*, and names the slider corner where the size comparison reads one.
   - New decision **A4-4**, recommending decision 9 as written (gated on E22 alone). The alternative is to gate the comparison on a grade CTE being in effect.
   - New Task 3 test `e22_compares_the_size_of_a_negative_shear_on_sintered_ndfeb_too` pins the corner on both bases: C104 below −15 MPa, C106 and C202 "Above" with E22 and "Below" without it, C95 in effect, and `moved_cells` exactly C106 and C202.
2. **A4-1: nothing showed that C25 can change under E24.**
   - A slider search found a reachable flip: Recoma 30 beside N42H, in either order, with EA 9514 (120 °C), the knee fraction C46 at 0.95 and the design margin C51 at 0. C24 goes from 75.40 to −6.719 °C (N42H's single-ring onset, 113.3 °C, is below the cure), and the verdict goes from OK to CHECK on both bases. Only C24 and C25 move. The fine search, on the user basis, covered every grade pair and part pair with both hot-cure adhesives, C46 from 0.50 to 1.00 in steps of 0.01 and C51 from 0 to 10 °C in steps of 0.5 °C. Flips occur only for this pair, with EA 9514 or 2214 Hi-Temp, C46 from 0.94 to 0.97 and C51 at most 7 °C. A coarser search over all four adhesives and both bases found no other.
   - A4-1 now states this, and C25 stays registered. The new Task 5 test `e24_turns_the_verdict_where_the_lower_ring_cannot_take_a_hot_cure` pins the flip.
   - The default-slider claim is now tested too: `e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict` runs all four adhesives.
   - E24's `corrected_formula` and README row now name the flip.
3. **E21's pair claims were not pinned.** New Task 1 test `e21_moves_only_pairs_with_a_negative_offset_and_never_raises_a_limit` covers the 272 grade pairs and 210 part pairs on both bases:
   - A pair in which both offsets are 0 or more is bit for bit unchanged. In every other pair C58, C60, C12 and C15 never rise.
   - It pins the report's table: 214 pairs moved, 71 flips and 0 verdicts on the registry basis; 150, 39 and 0 on the user basis. A pair counts as moved when a workbook cell changes.
   - The user count is the report's 150 through Task 4 and becomes 156 at Task 5 (checked against the engine at each stage). Task 5 updates that one line and says why: with E24 in every correction, a clamp on the ring that does not govern lowers that ring's single-ring onset, which E24 reads. C24 then moves in six more pairs: N50M beside N42M, and N38UH or N35EH beside Recoma 20, each in both orders, with only C24 moving.
4. **Section 2.6's double gate was missing.**
   - Task 1 adds `assert_no_differential_case_moves_between(bases, what)`, the corpus loop, run on every core. `e21_alone_moves_no_differential_case` and, in Task 5, `e24_alone_moves_no_differential_case` call it with NONE against `only(id)`.
   - Task 3's `assert_no_differential_case_moves(id)` is now a thin wrapper over it, run on the workbook and user bases, so there is one implementation.
5. **Test gaps in E24 and E23.**
   - (a) `e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict` now covers every ordered pair of library parts as well as of grades, rings alike or not, with all four adhesives on both hubs and both bases. It pins how many configurations C24 moves in: grades and parts 20 and 0 on the registry basis, 26 and 28 on the user basis, the same for every adhesive and hub.
   - (b) `e23_moves_no_verdict_and_the_peak_by_hundredths_of_a_degree` now asserts, for every ordered grade pair (Y30 beside Recoma 26 included), each ring's specific heat in effect and C141's shift = (m_inner (c_inner − C138) + m_outer (c_outer − C138)) / 1000. The tolerance is 1e-9 relative; the largest error measured is 4.3e-14, the smallest shift 0.297 J/K, and two sintered NdFeB rings shift by exactly 0.
6. **Docs.**
   - Task 1's README edits change "E15 to E20) carries registry probes" to "E15 onward)". The probe-cells sentence now reads "The Addendum A and A-4 entries (E15 onward)", because `addendum_entries_name_every_cell_their_probes_change` checks the A-4 entries too. Both strings are the same on both bases.
   - The Task 6 sign-off script's comment names Steps 2 and 3.
7. **DRY.**
   - `bond_plane_cte_per_C` (Task 3) and `magnet_specific_heat_J_kgK` (Task 4) share one private `grade_value_or(input, value, dev, id)`, and both public wrappers stay.
   - The four `e2x_leaves_every_default_cell_bit_for_bit` tests call one `assert_a4_defaults_unchanged(id)`.
   - Row comparison goes through `same_value`: `assert_same_results` and the new `moved_cells`, which the pair, corner and flip tests use, both call it.
   - The part-pair overrides are written once (`part_rings`), and so are the pair lists (`grade_pairs`, `part_pairs`).
   - `type Overrides` and the `differential_files`/`load_cases` import moved from Task 3 to Task 1, their first user.
8. **Model tiers and Task 0.** The tiers already followed CLAUDE.md section 5 and are unchanged. Task 0's prose now says that `extensions.worktreeConfig true` is a permanent change to the main repository's config. It is already `true` on this machine, so the step is a no-op there.
9. **Task 6's record string** (cosmetic). The verify trees were built before the string gained "; the note's other sentences keep their earlier review". Every tree of this revision is rebuilt from the current blocks, so the replayed Task 6 commit carries the plan's string.

New tests, by task: Task 1 adds 2, Task 3 adds 1 and Task 5 adds 2, all in `tests/deviations.rs`. Two tests were extended in place: `e23_moves_no_verdict_and_the_peak_by_hundredths_of_a_degree` and `e24_takes_the_lower_ring_for_every_pair_and_moves_no_verdict`. The four defaults tests now call the shared helper. No unit test, explain record, probe or registry cell changed. The only registry text that changed is the `corrected_formula` of E22 and E24.
