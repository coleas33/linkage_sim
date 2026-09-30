# Magcoupling Addendum A-1: Data and Physics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Addendum A engine data and physics on the finished M2 engine: the A6 grade and part tables (any grade with manual dimensions), the A5 materials library with per-part selectors, physics links and six warnings, and the approved corrections E15 to E20, while the defaults stay workbook-exact.

**Architecture:** Two static, cited tables (`grades.rs`, `material_library.rs`) built from the approved data file; every new input is Rust-only (a new `InputMeta::rust_only` flag the generator and the metadata tests honour), and its default reproduces the ported behaviour bit for bit, so `Deviations::NONE` parity (1,149 checks) and the differential tests never move. The part materials resolve once in `api::compute` into the values the engine reads (the default choice IS the workbook's inputs). Each correction is a registered deviation in its own commit, with the report's changed-cell tables or probes as expected values; E15 to E17 are probed on top of E9 through a new `Deviations::with`/`without` combinator.

**Tech Stack:** Rust 2024 edition, std only for the library (wasm32-unknown-unknown clean); `serde_json` (feature `float_roundtrip`) as the only dev-dependency; the vendored Python 3.12 oracle `reference/magcoupling-py/` run with `C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe`; Git Bash for every command.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-a1/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, Addendum A: A5, A6 and "Addendum testing" (the Materials and Grades items). A1 (sizing, autofit), A2 (explorer), A3 (assumptions), A4 (notes) and the GUI are OUT of scope: later plans. Authorities read with it: the verification report `C:/Users/Cole/source/repos/lsim-mag-a1/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (all 31 decisions approved as option A, section 8) and its data file `.../docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` (every value here comes from it, with its citation); the M2 plan `C:/Users/Cole/source/repos/linkage_simulation/docs/superpowers/plans/2026-09-29-magcoupling-m2-engine-port.md` and its architecture `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/m2-plan/architecture.md` (sections 4 to 9 are binding: naming, cell mapping, deviation mechanics, the porting pattern, the translation rules); `magcoupling-rs/README.md` (its short form).

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-a1`, branch `magcoupling/addendum-a1` (from `main` at `102672a`: the M2 engine with parity, differential tests and E1-E14). Every command uses absolute paths into it. Nothing is pushed.

## Decisions to confirm

**Confirmed (user, 2026-09-30): the recommended option on all of A1-A13.** No task stops on a decision.

These questions arose while turning the 31 approved decisions into code; none of the 31 settles them. Each row's **recommended** option is what this plan implements (the controller asks them once, in Task 0). If the user picks an alternative, the named task stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternative | Tasks |
|---|---|---|---|---|
| A1 | Registry ids for decisions 2 and 19 (the report gives none) | **E19** = the SuperMagnetMan arcs (decision 2), **E20** = each magnet's own coercivity (decision 19), in decision order | the two swapped | 2, 9, 10 |
| A2 | Decision 19 keeps C44/C45 "as overrides that win when set": how "set" is represented | **An explicit Rust-only selector** `temperature.demag.coercivity_source` (1 = the magnet's grade, default; 0 = C44 and C45); the M4 GUI flips it to 0 when the user edits either slider. With this choice, and every correction on, C44 and C45 change nothing for a magnet that has a grade (every library part, the default B842SH included) until the source is 0: typing -0.007 into C45 leaves C12 at 93.06 °C. C45 is an assumption input (A3), so Addendum A-2's test that each assumption moves its dependent results must allow for it, and C45's slider (-0.008 to -0.001) cannot enter ferrite's +0.0035 with the source at 0 (A-2 widens or splits it; recorded in `04-memory.yaml`, Task 14). The C44 and C45 labels stay (schema parity); their help says when they act, and E20's `workbook_help` records the workbook text | "set" = differs from the declared default (implicit: typing 1592 would not override); `Option` inputs (breaks the Python schema default) | 10 |
| A3 | What the results mean for a positive beta (ferrite): decision 19 includes "the positive-beta branch" but not its outputs | **No knee on heating:** hot onsets C7-C9 and C56-C59 = +inf, no calibration offset (C50 = 0; the rating is not a cold-side knee rating), hot limit C10/C60 = the grade's rating; **new Rust-only cold onsets** (four cases), **cold limit** (skipping onset + C51) and **cold check** against the minimum temperature (Metal design C16), which also gates the verdict C25 | write the cold onsets into the existing onset cells (two meanings in one field); report the cold side only, verdict unchanged | 10 |
| A4 | What a grade picked for manual dimensions supplies | **Br at 20 °C and Tmax per ring** (with E20 also Hcj and beta); alpha(Br) and magnet density stay the calculator's single alpha input (Calibration!C22) and the NdFeB density, as for every library part (all NdFeB); the M4 GUI offers the grade's alpha as a preset; the ferrite probe sets alpha = -0.20 %/C explicitly | per-ring alpha and density from the grade (touches model, metal design, temperature and mass; no parity risk because the mode is new): carried to A-2 | 4, 10 |
| A5 | How a part-material choice combines with the existing inputs (report 6.1 point 1) | **The default choice is the workbook's material, whose values ARE the inputs**; another choice supplies its library engine values in place of those inputs; a code outside the choices gives NaN and "#N/A" (decision D3) | GUI presets that write the inputs, engine unchanged: cannot separate the cap, the aluminium adapter and the no-back-iron body, which share Metal design C42 and Temperature design C140 | 12 |
| A6 | "The bare backiron selector stays available as an override" | **Effective C6 = 0 for a non-ferromagnetic back iron, else the C6 input**, so C6 = 0 still selects the free-space circuit with a steel choice | the material decides alone; C6 ignored once a library back iron is picked | 12 |
| A7 | The hub, cup and boss material with no back iron (E9, E15-E18 assume aluminium) | **The workbook's aluminium** (C42, C140, Materials!C43 and E18's 6061) by default and when C6 = 0 overrides a steel choice; **a non-ferromagnetic pick (304, 6061) is its own body material** (density, cp, sigma, CTE, modulus) | always the workbook aluminium (a 304 back iron priced as aluminium) | 12 |
| A8 | E18's aluminium modulus: its report row uses Alliance 68.9 GPa; the approved materials table has Kaiser 68.3 GPa for 6061 (decision 5) | **Keep both as approved:** E18 uses 68.9 (its probe basis); the library keeps 68.3; picking "6061" as back iron differs from the default aluminium body only in C104, C105, C201 (pinned by a test); unify later | E18 reads the library (68.3): M2-basis C104 20.75 and workbook-basis 73.74 instead of the report's 20.76 and 73.87; or change the library's 6061 modulus to 68.9 | 8, 12 |
| A9 | Design flux density of a library steel that has none (12L14, 416, 17-4PH): decision 20 covers 4140 (1.5 T) and 1018 (1.7 T, the workbook's comment) | **Keep Materials C13 (1.5 T)** for the wall check and fire the low-saturation warning. Consequences to weigh: Materials C13 is an assumption input (A3), and once a back iron with its own design flux density is picked (1018) C13 no longer acts on the wall check (changing it from 1.5 to 1.2 T leaves the verdict at "OK"); picking 1018 turns the default wall verdict (Materials C22) from "Too thin: raise Metal design C122 to at least 2.0 mm" to "OK" (pinned by `the_design_flux_density_feeds_the_wall_check`) | take a lower bound as the design value (17-4PH 1.07 or 1.35 T at 140 Oe, 416's 1.60 T Bsat): promotes a lower bound, which the report forbids | 12, 13 |
| A10 | Warning conditions and thresholds (the data file: "a model choice") | non-ferromagnetic back iron = the **free-space circuit in effect** (a non-magnetic pick or C6 = 0); sleeve sigma **> 1.35e6 S/m** (316L, C111); low saturation = steel back iron with **Bsat < 1.7 T** (the highest design flux density the workbook names for a back-iron steel: Materials C13's comment on 1018, also 1018's design value in the wall check; a design value used as the threshold because no source gives a saturation threshold) **or no design flux density of its own**; a back-iron code outside its choices fires no material rule (decision D3); plating = 4140, 1018 or 12L14 with **Materials C26 = 0**; CTE mismatch = hub in effect against C95 **> 15e-6 /°C** (above 4140's 13.1e-6, below 304's 17.7e-6) | other thresholds, e.g. relative to the default design | 13 |
| A11 | Spec A5 names conductivity, density, CTE and modulus as links, not specific heat | **Specific heat follows the material too** (heat capacity), for the back iron, the body, the sleeve and the cap | keep cp at the inputs (a PEEK sleeve heats like 316L) | 12 |
| A12 | No listed sleeve can meet the ferromagnetic-sleeve rule (spec: 316L, Ti-5, Inconel 625, PEEK) | **Keep the spec's choices per part**; `warnings.rs` tests the rule directly | offer every library material in every selector | 12, 13 |
| A13 | Which ring E20 checks. C52-C55 are the OUTER blocks' worst reverse fields (their help: "Outer blocks (inner 341)"), yet the workbook's block reads only the inner ring's Br (C21) and rating (C22); report 6.3 warns the choice becomes visible once each ring has a grade. Checking the inner ring only, B842SH inside B842 reads 93.06 °C and "OK" while the same parts swapped read -3.52 °C and "CHECK"; grade N38UH inside B842 reads 120.00 °C, slightly less conservative than the workbook's 119.03 °C | **(a) Each ring with its own grade (or C44/C45), Br and rating; the ring with the lower magnet limit governs.** The block (C42, C47-C61, the Hcj and beta used) shows that ring whole, so C56 still follows from C52, C50 and the Hcj used; the inner ring on a tie, so identical rings (the default, every probe of the report) are the inner ring's block bit for bit. The cold side shows the ring with the higher cold limit and passes only if both rings pass. Rust-only `demag_ring` and `cold_ring` name the rings shown. The mixed pair reads -3.52 °C and "CHECK" in both orders. The governing ring is chosen on the magnet limit (the skipping case), so the cure margin C24 reads that ring's single-ring onset | (b) the outer ring's grade, Br and rating, because C52-C55 are the outer blocks' fields (a weaker inner ring would then read "OK"); (c) keep the inner ring, as the workbook does (the mixed pair keeps reading "OK") | 10 |

Settled here without a decision (the report asked the plan to fill them): coating and magnetization direction of every library part, read from the vendors' product pages on 2026-09-30 (K&J: "Nickel-Copper-Nickel (Ni-Cu-Ni)", "Magnetized Through Thickness" on all twelve blocks; SuperMagnetMan: "Nickel", "Radially magnetized" on the three arcs), each cited by page; the 416 expansion coefficient re-sourced as decision 6 A asks (the Rolled Alloys 416 sheet prints "Coefficient of Thermal Expansion* 5.6 in/in F x 10-6" in its 212 F column, footnote "* 70F to indicated temperature": 5.6e-6 in/in/F from 70 to 212 F = 10.08e-6 /K over 21-100 °C, quoted in the material's notes; Smiths prints 9.9 with no range).

## Global Constraints

Every task's requirements implicitly include this section.

- Crate `magcoupling-rs/` stays a sibling of `linkage-sim-rs/`: "No Cargo workspace conversion." The library is pure std with no `[dependencies]`; `cargo check --target wasm32-unknown-unknown --lib` stays clean. The `workbook-parity` feature is reachable only through the self dev-dependency.
- "One pure entry point `compute_all(&DesignInputs) -> Results`: no I/O, no global state, milliseconds per call." Its signature does not change.
- "Text verdicts are reproduced character for character." "Selector integer codes match the workbook (`backiron`: 1 steel, 0 none, etc.)." "Units as the package: mm, N·m, °C, T, kA/m, MPa, W, J, rpm, g."
- Addendum A: "At default inputs and default assumptions every result stays workbook-exact, so the M2 parity and differential tests are unchanged." After every task: `Deviations::NONE` parity (1,149 checks: numbers 1e-9 relative, 1e-12 absolute, text exact) and every differential file pass unchanged, and the default results with every correction on keep the M2 headline (pull-out 2.688 N·m, limit 93.06 °C, clamp M4 x 14; `all_corrections_together_give_the_reviewed_headline` is never edited).
- Every new input is Rust-only (`param_rust_only`): no workbook cell, never passed to Python, and its default reproduces the ported behaviour bit for bit.
- Deviation registry (spec M2): "cells, workbook value at defaults, corrected formula, evidence link"; with deviations on, only registered fields may differ; a correction neutral at defaults carries probes; E15-E17 are probed on top of E9 (decision 15 A). One commit per correction, never in a feature commit. The test-only switch is never exposed to users.
- Library values are the data file's engine-unit literals, typed as literals and never converted at load time (report 2.5: -0.55/100 is not -0.0055); every value keeps its source citation in a data field, and `tests/grades.rs` and `tests/material_library.rs` compare them with the data file.
- `reference/magcoupling-py/`: its engine (`magcoupling/`) is never edited and its `pytest` stays green; only `tools/gen_differential.py` changes (Task 1). The Addendum A report and data file are evidence: never edited.
- Porting and translation rules (architecture sections 7 and 8, README): operand order never changed; `py_min`/`py_max`; Python names verbatim; `#[allow(non_snake_case)]` where names carry units; never `+`, `-` or `*` on an `i64` from an input; catch-all `else` kept; a selector code outside its choices gives NaN or `"#N/A"`, never another row (decision D3), except a two-way choice, which falls through to its else branch as the workbook's two-way IFs do (README, "Invalid inputs"): the Rust-only `coercivity_source` is the one new two-way selector, so any code but 1 means C44 and C45 for both rings (Review Focus 5).
- Every gate run: `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate 7/7` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them). Gate 5 is `cargo clippy --all-targets -- -D warnings` on `magcoupling-rs`.
- `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml` before every commit (it breaks lines, never reorders operands). Each edit block below is shown as written before formatting; later blocks quote the formatted text, so run every `cargo fmt` step where it appears: skipping one makes a later "replace" block miss its text.
- Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that wrote the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The commit blocks below write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the session model); a `sonnet` task replaces `Claude Opus 5.5` with its own model's name. Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`): each task updates `magcoupling-rs/README.md` and `docs/ai/03-structure.yaml` where it changes layout, tests or corrections; Task 14 updates `02-system.yaml`, `04-memory.yaml` and `05-update-tracker.md`. Read `docs/ai/*.yaml` before starting (Task 0).
- Model tiers (CLAUDE.md section 5): each task's **Model:** line. Tasks 0-4, 11 and 14 give exact code or data and run on `sonnet` (set `model: 'sonnet'`); Tasks 5-10, 12 and 13 are physics (corrections, new physics links and rules) and run on the session model (omit `model` on purpose, comment `// session model: physics`), as do their physics reviews, whose prompt names the report rows and decisions of the task. A `sonnet` attempt that ends `blocked`, leaves the gate red or is rejected in review is retried on the session model. The final whole-branch review runs on the session model.
- Equality edges (architecture section 7 step 8): each new threshold comparison gets a unit test at exact equality that asserts the equality first (`cold_check_passes_at_equality`, `thresholds_are_strict_at_equality`).
- Commit subjects: `feat(magcoupling-rs): ...` for features, `fix(magcoupling-rs): apply E<k> ...` for corrections, `docs(magcoupling): ...` for docs.

## Review Focus

Six inputs the spec implies but no parity or differential case holds (every new input is Rust-only, so the differential data only ever holds its default). Each names the test that pins it and its task.

1. **Grade text that almost matches, or a grade set while the part is in the library** (`"y30"`, `"Y30 "`, `"N 42"`; `grade_inner = "Y30"` with part B842SH). Expected: exact-text lookup like the part lookup; a near miss is today's manual mode (manual Br, `"n/a"`, uncalibrated), and a library part always wins over the grade. Test: `a_library_part_wins_over_a_grade_and_an_unknown_grade_is_manual` (Task 4).
2. **C6 = 0 typed with a ferromagnetic library back iron** (1018 picked, backiron 0). Expected: the free-space circuit and the workbook's aluminium hub, cup and boss, not a steel cup priced into a free-space design. Test: `the_backiron_input_overrides_a_ferromagnetic_choice` (Task 12).
3. **A value typed into an input that a pick would supply** (316L conductivity typed at 5e6 S/m with the default sleeve; the steel's expansion coefficient typed at 16e-6 /°C with the default back iron). Expected: the warning fires as it would for a picked material, and only that warning: the rules read the materials in effect. Test: `each_warning_fires_on_the_design_that_meets_its_condition` (Task 13).
4. **A ferrite grade with the reverse fields left at the stored NdFeB values** (all above Y30's 162 kA/m knee at 20 °C). Expected: no panic and no false cold-weather margin: the cold onsets lie above the operating temperature, the cold check fails, the verdict reads CHECK. Test: `e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature` (Task 10).
5. **A Rust-only selector code outside its choices set on the struct** (material codes 99, 0, -1; coercivity source 7). Expected: never another material: NaN properties and `"#N/A"`; the coercivity source, a two-way choice, falls through to the inputs (Global Constraints); `validate()` names each path; an unknown back iron fires no material warning. Tests: `a_code_outside_the_choices_gives_nan_not_another_material` (Task 12), `an_invalid_coercivity_source_uses_the_inputs` (Task 10).
6. **Different grades on the two rings** (B842SH inside B842, and swapped; an NdFeB ring beside a ferrite ring). Expected (A13): the weaker ring governs whichever side it is on: -3.52 °C and "CHECK" in both orders, the block showing B842 (`demag_ring`); with a ferrite ring the hot side is the NdFeB ring's and the cold side the ferrite's, and the verdict needs both. Tests: `e20_mixed_rings_use_the_weaker_grade` (Task 10, and the registry's third E20 probe), `e20_checks_both_rings_each_side_from_the_weaker` (Task 10, unit).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/`):

| File | Responsibility | Task |
|---|---|---|
| `src/engine/grades.rs` | A6 grade table: 17 grades, every value cited; `grade(id)`; `N42SH` | 3 |
| `src/engine/material_library.rs` | A5 materials: 14 materials cited per value, engine values, each part's choices (Task 11); `resolve` into `PartProperties` (Task 12) | 11, 12 |
| `src/engine/warnings.rs` | The six A5 warning rules: text, severity, note id, thresholds, `WarningResults` | 13 |
| `tests/grades.rs` | Grades and parts against the data file; parts resolve to grades | 3, 9 |
| `tests/material_library.rs` | Materials against the data file; defaults equal the inputs | 11 |
| `tests/material_links.rs` | The physics links and the warnings end to end | 12, 13 |

Modified: `src/engine/{meta,deviations,library,model,metal_design,materials,temperature,api,mod}.rs`, `src/lib.rs`, `tests/{schema,python_schema,differential,deviations,robustness}.rs`, `tests/data/input_schema.json` (blessed), `reference/magcoupling-py/tools/gen_differential.py`, `magcoupling-rs/README.md`, `docs/ai/{02-system,03-structure,04-memory}.yaml`, `docs/ai/05-update-tracker.md`. No differential, parity or golden data file changes (they are the proof the defaults did not move).

Order: harness first (Rust-only inputs, the registry), then the A6 data (grades, parts, the grade mode), then the corrections E15-E18 against the report on the existing aluminium constants (their probes then guard the materials rewiring), E19 and E20 (which read the grade data), and last the A5 materials: the library, the links, the warnings.

## Verification record

Every edit, script and test command of Tasks 1 to 14 was replayed, task by task and in this order, on a fresh copy of `magcoupling-rs`, `reference/magcoupling-py`, `docs/ai` and `docs/analyses` taken from `102672a` (a scratch directory, not the worktree): each "see it fail" step failed with the error it names, each later step passed, and after every task `cargo test`, `cargo clippy --all-targets -- -D warnings`, the wasm32 check, `cargo fmt --check` and `gen_differential.py --check` (against the mirrored oracle) passed. The gate script itself was not run in the replay (the copy has no `linkage-sim-rs`); its four magcoupling gates were run one by one. Task 0's checks (baseline counts, oracle pytest `1159 passed`) were measured when the plan was written. The oracle's own `pytest` (1159 passed) is unaffected: no engine file under `reference/magcoupling-py/magcoupling/` changes. Final replayed state: unit tests 107, `deviations.rs` 54, `differential.rs` 19, `grades.rs` 9, `material_library.rs` 4, `material_links.rs` 11, `parity.rs` 4, `python_schema.rs` 7, `robustness.rs` 9, `schema.rs` 7, `static_data.rs` 8, doc-tests 1 (1 ignored). The registry probes and report tables reproduce the report:
- E15: all 23 cells in each of three columns.
- E16: at full precision.
- E17: all 41 cells in two columns, plus the six full-precision values.
- E18: both bases, with the verdict flips.
- E20: B842 C12 23.06 → -3.52 °C, and every part of report 6.3.

The mixed rings of decision A13 give -3.52 °C and "CHECK" in both orders, where the workbook gives 92.55 °C with B842SH inside. Every Addendum A probe changes only cells its entry lists (`addendum_entries_name_every_cell_their_probes_change`).

---

## Tasks

### Task 0: Baseline and context

**Model:** `sonnet` for Steps 1-5 (running commands and reporting output; CLAUDE.md section 5). Step 6 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: branch `magcoupling/addendum-a1` at `102672a` (created from `main`).
- Produces: a confirmed green baseline and the controller's record of the answers to the Decisions to confirm.

- [ ] **Step 1: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`; `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` whole (the Deviations and Translation rules sections are binding); the spec's Addendum A (`C:/Users/Cole/source/repos/lsim-mag-a1/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, sections A5, A6 and "Addendum testing"); the verification report `C:/Users/Cole/source/repos/lsim-mag-a1/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (sections 2 to 6 and 8) and this plan's Decisions to confirm.

- [ ] **Step 2: Check the branch and a clean tree**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-a1 log --oneline -1
git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short
```

Expected: `magcoupling/addendum-a1`; `102672a docs: record approval of the 31 Addendum A decisions; spec text fixes (N50/N50M, 20 mm bay, design flux density)`; no status lines.

- [ ] **Step 3: Run the crate's tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: in order: unit tests `88 passed`; `tests\deviations.rs` 35 passed; `tests\differential.rs` 19 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 6 passed; `tests\robustness.rs` 7 passed; `tests\schema.rs` 7 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`.

- [ ] **Step 4: Check the oracle Python**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
cd C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `ls` prints the path; then `differential data is current (...)`. If `ls` fails, stop and escalate: every gate run of this plan uses that interpreter.

- [ ] **Step 5: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task0.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Decisions**

The controller shows the user this plan's **Decisions to confirm** table in one message and records the answers in the plan's execution notes. Every task implements the recommended option; a task that depends on a decision names it in its intro. If the user picks an alternative that this plan gives no concrete variant for, that task stops and escalates instead of improvising.

---

### Task 1: Rust-only inputs (`InputMeta::rust_only`, `param_rust_only`)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

Why (report section 6.1 point 2, M2 Task 15's `ResultMeta::rust_only` for results): Addendum A adds inputs the Python
engine does not have (the part material selectors, the magnet grade per ring, the coercivity source, the E17 free-space fields).
Without this flag `gen_differential.py` would pass them to Python's `set_input` and fail, and the metadata tests would look
them up in Python's schema. The first real Rust-only inputs arrive in Task 4; that task's gate proves the generator filter.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/meta.rs` (the flag, the builder, a unit test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/schema.rs` (export the flag; Rust-only exactly when uncelled)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs` (skip Rust-only inputs; guard test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/differential.rs` (`selectors()` skips Rust-only selectors)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py/tools/gen_differential.py` (leave Rust-only inputs out)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/data/input_schema.json` (`"rust_only"` on every row)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Tests and Regenerating sections)

**Interfaces:**
- Consumes: `meta::InputMeta` (fields `name, ty, unit, label, help, cell, choices, range, assumption`), `param(unit, label, help, cell) -> InputMeta`.
- Produces:
  - `pub rust_only: bool` on `InputMeta` (false from `param`);
  - `pub const fn param_rust_only(unit: &'static str, label: &'static str, help: &'static str) -> InputMeta` (cell `None`, `rust_only: true`; chain `.range()`, `.log()`, `.choices()` as for `param`);
  - `input_schema.json` rows carry `"rust_only"`; `gen_differential.py` drops those rows before it builds any case, so a Rust-only input keeps its Rust default in every parity and differential case;
  - the tests skip Rust-only inputs where they compare with Python (`python_schema.rs`) or index differential cases by selector path (`differential.rs::selectors`).

- [ ] **Step 1: Write the failing unit test**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/meta.rs`, replace:

```rust
    #[test]
    fn only_out_rust_only_marks_a_result_rust_only() {
```

with:

```rust
    #[test]
    fn only_param_rust_only_marks_an_input_rust_only() {
        let rust = param_rust_only("-", "Grade", "").choices(&[(0, "none")]);
        assert!(rust.rust_only && rust.cell.is_none());
        assert_eq!(rust.choices, &[(0, "none")]);
        let workbook = param("-", "Part", "", "S!C1");
        assert!(!workbook.rust_only && workbook.cell == Some("S!C1"));
        assert!(!Leaf::FIELDS.iter().any(|m| m.rust_only));
    }

    #[test]
    fn only_out_rust_only_marks_a_result_rust_only() {
```

- [ ] **Step 2: Run it to see it fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --lib only_param_rust_only 2>&1 | grep -E "^error" | head -3
```

Expected: a compile error, ``error[E0425]: cannot find function `param_rust_only` in this scope``.

- [ ] **Step 3: Add the flag and the builder**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/meta.rs`, replace:

```rust
    pub range: Option<SliderRange>,
    /// A model assumption (Addendum A3) rather than a design input.
    pub assumption: bool,
}
```

with:

```rust
    pub range: Option<SliderRange>,
    /// A model assumption (Addendum A3) rather than a design input.
    pub assumption: bool,
    /// An input the Python engine does not have (Addendum A5/A6 selectors, the
    /// E17 free-space fields): no workbook cell; the metadata-parity tests skip
    /// it and the differential generator never passes it to Python, so it
    /// keeps its default in every parity and differential case.
    pub rust_only: bool,
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/meta.rs`, replace:

```rust
        cell: Some(cell),
        choices: &[],
        range: None,
        assumption: false,
    }
}

impl InputMeta {
```

with:

```rust
        cell: Some(cell),
        choices: &[],
        range: None,
        assumption: false,
        rust_only: false,
    }
}

/// Declares an input the Python engine does not have (no workbook cell), in
/// the order of [`param`] without the cell. Chain `.range()` or `.choices()`
/// as for any input.
pub const fn param_rust_only(
    unit: &'static str,
    label: &'static str,
    help: &'static str,
) -> InputMeta {
    InputMeta {
        name: "",
        ty: FieldType::F64,
        unit,
        label,
        help,
        cell: None,
        choices: &[],
        range: None,
        assumption: false,
        rust_only: true,
    }
}

impl InputMeta {
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --lib only_param_rust_only 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed` (the unit-test binary; the others are filtered out).

- [ ] **Step 4: Teach the harness and the generator about Rust-only inputs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/schema.rs`, replace:

```rust
                "assumption": m.assumption,
            })
```

with:

```rust
                "assumption": m.assumption,
                "rust_only": m.rust_only,
            })
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/schema.rs`, replace:

```rust
    let mut failures = Vec::new();
    // A Rust-only result has no Python counterpart, so no workbook cell either.
```

with:

```rust
    let mut failures = Vec::new();
    // A Rust-only input or result has no Python counterpart, so no workbook cell
    // either; every other input names its workbook cell.
    for r in &inputs {
        assert_eq!(
            r.meta.rust_only,
            r.meta.cell.is_none(),
            "{}: an input is Rust-only exactly when it has no cell",
            r.path
        );
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/schema.rs`, replace:

```rust
//! then regenerate the differential data (see `magcoupling-rs/README.md`).
```

with:

```rust
//! then regenerate the differential data (see `magcoupling-rs/README.md`).
//! Rust-only inputs (`InputMeta::rust_only`) are exported with `"rust_only": true`;
//! the generator leaves them out, so the Python engine never sees them.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs`, replace:

```rust
    for row in input_rows(&DesignInputs::defaults_with(Deviations::NONE)) {
        let Some(py) = python.get(&row.path) else {
```

with:

```rust
    for row in input_rows(&DesignInputs::defaults_with(Deviations::NONE))
        .into_iter()
        .filter(|r| !r.meta.rust_only)
    {
        let Some(py) = python.get(&row.path) else {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs`, replace:

```rust
    let inputs: Vec<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .map(|r| r.path)
        .collect();
```

with:

```rust
    let inputs: Vec<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| !r.meta.rust_only)
        .map(|r| r.path)
        .collect();
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs`, replace:

```rust
//! is compared against the recorded workbook text instead.
```

with:

```rust
//! is compared against the recorded workbook text instead. Rust-only inputs and
//! results (`rust_only`) have no Python counterpart and are skipped.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs`, replace:

```rust
#[test]
fn ported_results_carry_the_python_metadata() {
```

with:

```rust
#[test]
fn rust_only_inputs_are_unknown_to_python() {
    // A Rust-only input that Python also has would be skipped by the metadata test above.
    let python = python_rows();
    let clashes: Vec<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| r.meta.rust_only && python.contains_key(&r.path))
        .map(|r| r.path)
        .collect();
    assert!(clashes.is_empty(), "Rust-only inputs Python has: {clashes:?}");
}

#[test]
fn ported_results_carry_the_python_metadata() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/differential.rs`, replace:

```rust
/// Selector inputs and their codes, from the Rust metadata.
fn selectors() -> Vec<(String, Vec<i64>)> {
    input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| !r.meta.choices.is_empty())
```

with:

```rust
/// Selector inputs and their codes, from the Rust metadata. Rust-only selectors
/// are not in the differential data (the generator never passes them to Python).
fn selectors() -> Vec<(String, Vec<i64>)> {
    input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| !r.meta.choices.is_empty() && !r.meta.rust_only)
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py/tools/gen_differential.py`, replace:

```python
    magcoupling-rs/tests/data/input_schema.json
        types, workbook defaults, choices and slider ranges of every input;
        rewrite it with `MAGCOUPLING_BLESS=1 cargo test --test schema`.
```

with:

```python
    magcoupling-rs/tests/data/input_schema.json
        types, workbook defaults, choices and slider ranges of every input;
        rewrite it with `MAGCOUPLING_BLESS=1 cargo test --test schema`.
        Rust-only inputs ("rust_only": true: the Addendum A selectors and the
        E17 free-space fields) are left out: the Python engine has no such
        input, so every case keeps them at their Rust defaults.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py/tools/gen_differential.py`, replace:

```python
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["inputs"]
```

with:

```python
    schema = [f for f in json.loads(schema_path.read_text(encoding="utf-8"))["inputs"]
              if not f.get("rust_only", False)]  # the Python engine has no Rust-only input
```

`reference/magcoupling-py/tools/` is ours (the engine under `magcoupling/` stays untouched).

- [ ] **Step 5: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test schema
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: the bless run passes (it rewrites `tests/data/input_schema.json`: `"rust_only": false` added to all 161 rows, nothing else); then every binary `ok`, no `FAILED`: unit tests 89 passed, `python_schema.rs` 7 passed.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|panicked"
```

Expected: no output.

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 6: Update the crate README**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
Rust-only results (`ResultMeta::rust_only`, e.g. `clamps.length_note`) have no Python counterpart and are skipped. |
```

with:

```markdown
Rust-only results (`ResultMeta::rust_only`, e.g. `clamps.length_note`) have no Python counterpart and are skipped; so are Rust-only selectors in the selector-coverage tests (`InputMeta::rust_only`: the generator never passes them to Python). |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
inputs and scalar results are listed in the Python order (the GUI's tables and CSV export follow it); Rust-only results are skipped. |
```

with:

```markdown
inputs and scalar results are listed in the Python order (the GUI's tables and CSV export follow it); Rust-only inputs and results are skipped, and no Rust-only input may share a path with a Python one (`rust_only_inputs_are_unknown_to_python`). |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none);
```

with:

```markdown
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell);
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
Run 1, then 2, when an input, a range, a choice or a table layout changes; run 3
when a broad correction changes cells.
```

with:

```markdown
Run 1, then 2, when an input, a range, a choice or a table layout changes; run 3
when a broad correction changes cells. A Rust-only input (declared with
`param_rust_only`, exported with `"rust_only": true`) is left out by generator 2:
the Python engine has no such input, so every case keeps its Rust default and no
data file changes when one is added.
```

- [ ] **Step 7: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task1.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task1.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 8: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/meta.rs magcoupling-rs/tests/schema.rs magcoupling-rs/tests/python_schema.rs magcoupling-rs/tests/differential.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md reference/magcoupling-py/tools/gen_differential.py
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
feat(magcoupling-rs): Rust-only inputs (InputMeta::rust_only, param_rust_only)

Addendum A adds inputs the Python engine does not have. The flag mirrors
ResultMeta::rust_only: no cell, skipped by the metadata-parity tests and the
selector-coverage tests, and left out by gen_differential.py (report 6.1).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 2: Registry for the Addendum A corrections: approval evidence, `depends_on`, `with`/`without`, E15 to E20 as Planned

**Model:** `sonnet` (exact code; infrastructure, no physics). Escalate as in Task 1.

Decision 15 A: E15-E17 refine E9, so their probes run on top of it. The ids E19 (decision 2) and E20 (decision 19)
are this plan's (Decisions to confirm, A1). The Addendum report's rows still read `pending (decision N)` and section 8
records the approval: the approval test checks the section-8 line, each cited decision, and that a row (E15-E18) names
its first decision. The report is evidence and is not edited.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs` (ids E15-E20, `ADDENDUM_REPORT`, `Approval`, `depends_on`, `with`/`without`, six Planned entries, unit tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs` (approval test against both reports; probes run on their base; every cell an Addendum probe changes is registered)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Deviations section)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (the deviations line)

**Interfaces:**
- Consumes: `REGISTRY`, `Deviation`, `DeviationId`, `Deviations` (`ALL`, `is_on`; test-only `NONE`, `only`).
- Produces:
  - `DeviationId::{E15, E16, E17, E18, E19, E20}`; `DeviationId::ALL: [DeviationId; 20]`;
  - `pub const ADDENDUM_REPORT: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md"`;
  - `pub enum Approval { AuditRow, Addendum { decisions: &'static [u32] } }`;
  - `Deviation` gains `pub approval: Approval` and `pub depends_on: &'static [DeviationId]`, `Deviation::report(&self) -> &'static str`; `evidence()` names the Addendum decisions for E15-E20;
  - test-only `Deviations::with(self, id) -> Self` and `Deviations::without(self, id) -> Self` (`const fn`);
  - six `Planned` entries: E15 (decisions 8, 15; depends on E9), E16 (9, 14, 15; E9), E17 (10-15; E9), E18 (16), E19 (2), E20 (19);
  - `tests/deviations.rs`: `fn probe_base(d: &Deviation) -> Deviations` (the `depends_on` set); `each_probe_shows_its_correction` compares `probe_base(d)` against `probe_base(d).with(d.id)`; `fn probe_cells(probe: &Probe, dev: Deviations) -> BTreeMap<String, Value>` (`at_probe` reads one cell of it); `addendum_entries_name_every_cell_their_probes_change` (E15-E20 list every cell their probes change, so each entry's `cells` is complete from the moment its probes land).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            assert!(!d.cells.is_empty(), "{} names no cell", d.id);
            assert!(d.evidence().starts_with(REPORT), "{}", d.id);
            assert!(d.evidence().ends_with(&format!("entry {}", d.id)));
        }
    }
```

with:

```rust
            assert!(!d.cells.is_empty(), "{} names no cell", d.id);
            assert!(d.evidence().starts_with(d.report()), "{}", d.id);
            match d.approval {
                Approval::AuditRow => {
                    assert!(d.id.index() < 14, "{}: only E1 to E14 are M1 rows", d.id);
                    assert!(d.evidence().ends_with(&format!("entry {}", d.id)));
                }
                Approval::Addendum { decisions } => {
                    assert!(d.id.index() >= 14, "{}: E1 to E14 are M1 rows", d.id);
                    assert!(!decisions.is_empty(), "{} cites no decision", d.id);
                    assert!(
                        decisions.iter().all(|n| (1..=31).contains(n)),
                        "{}",
                        d.id
                    );
                }
            }
            for dep in d.depends_on {
                assert!(
                    dep.index() < d.id.index(),
                    "{} depends on a later {dep}",
                    d.id
                );
            }
        }
    }

    #[test]
    fn evidence_names_the_report_and_the_decisions() {
        assert_eq!(
            REGISTRY[DeviationId::E1.index()].evidence(),
            format!("{REPORT}, entry E1")
        );
        assert_eq!(
            REGISTRY[DeviationId::E15.index()].evidence(),
            format!("{ADDENDUM_REPORT}, section 8, decision 8")
        );
        assert_eq!(
            REGISTRY[DeviationId::E17.index()].evidence(),
            format!("{ADDENDUM_REPORT}, section 8, decisions 10, 11, 12, 13, 14, 15")
        );
    }

    #[test]
    fn with_and_without_add_and_remove_one_correction() {
        let (e9, e15) = (DeviationId::E9, DeviationId::E15);
        let both = Deviations::NONE.with(e9).with(e15);
        for id in DeviationId::ALL {
            assert_eq!(both.is_on(id), id == e9 || id == e15, "{id}");
            assert_eq!(Deviations::ALL.without(e15).is_on(id), id != e15, "{id}");
        }
        assert_eq!(Deviations::NONE.with(e9), Deviations::only(e9));
        assert_eq!(both.without(e15), Deviations::only(e9));
        assert_eq!(Deviations::ALL.without(e9).with(e9), Deviations::ALL);
        assert_eq!(Deviations::NONE.without(e9), Deviations::NONE);
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        assert_eq!(
            REGISTRY[DeviationId::E15.index()].evidence(),
            format!("{ADDENDUM_REPORT}, section 8, decision 8")
        );
```

with:

```rust
        assert_eq!(
            REGISTRY[DeviationId::E18.index()].evidence(),
            format!("{ADDENDUM_REPORT}, section 8, decision 16")
        );
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
use magcoupling::engine::deviations::{
    Deviation, DeviationClass, DeviationId, DeviationStatus, Deviations, Literal, Probe, REGISTRY,
    REPORT,
};
```

with:

```rust
use magcoupling::engine::deviations::{
    ADDENDUM_REPORT, Approval, Deviation, DeviationClass, DeviationId, DeviationStatus, Deviations,
    Literal, Probe, REGISTRY, REPORT,
};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn every_entry_is_an_approved_row_of_the_audit_report() {
    let text = read_text(&repo_path(REPORT));
    for d in REGISTRY {
        let prefix = format!("| {} |", d.id);
        let row = text
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{REPORT} has no row for {}", d.id));
        assert!(
            row.contains("| approved (user, 2026-09-29) |"),
            "{} is not approved in the report",
            d.id
        );
        let documentation = row.contains("(documentation)");
        assert_eq!(
            documentation,
            d.class == DeviationClass::Documentation,
            "{}: class vs report",
            d.id
        );
    }
}
```

with:

```rust
#[test]
fn every_entry_is_approved_in_its_report() {
    let audit = read_text(&repo_path(REPORT));
    let addendum = read_text(&repo_path(ADDENDUM_REPORT));
    // Section 8 of the Addendum A report: the approval line and the numbered decisions.
    let section8 = addendum
        .split_once("\n## 8. Decisions for the user\n")
        .map(|(_, rest)| rest)
        .expect("the Addendum A report has a section 8");
    assert!(
        section8.contains("**Approved (user, 2026-09-30): option A on all 31 decisions.**"),
        "section 8 records the approval"
    );
    for d in REGISTRY {
        match d.approval {
            Approval::AuditRow => {
                let prefix = format!("| {} |", d.id);
                let row = audit
                    .lines()
                    .find(|line| line.starts_with(&prefix))
                    .unwrap_or_else(|| panic!("{REPORT} has no row for {}", d.id));
                assert!(
                    row.contains("| approved (user, 2026-09-29) |"),
                    "{} is not approved in the report",
                    d.id
                );
                let documentation = row.contains("(documentation)");
                assert_eq!(
                    documentation,
                    d.class == DeviationClass::Documentation,
                    "{}: class vs report",
                    d.id
                );
            }
            Approval::Addendum { decisions } => {
                for n in decisions {
                    let heading = format!("{n}. **");
                    assert!(
                        section8.lines().any(|line| line.starts_with(&heading)),
                        "{}: section 8 has no decision {n}",
                        d.id
                    );
                }
                // E15 to E18 have an audit row ("| E18 (candidate) |" too); its status
                // column names the first decision that approves the entry.
                let prefix = format!("| {} ", d.id);
                if let Some(row) = addendum.lines().find(|line| line.starts_with(&prefix)) {
                    let status = row.trim_end_matches('|').rsplit('|').next().unwrap_or("");
                    assert!(
                        status.contains("decision") && status.contains(&decisions[0].to_string()),
                        "{}: row status {status:?} does not name decision {}",
                        d.id,
                        decisions[0]
                    );
                }
                assert_eq!(d.class, DeviationClass::Engine, "{}", d.id);
            }
        }
    }
}

#[test]
fn e15_to_e18_have_audit_rows_and_e19_e20_decisions_only() {
    let addendum = read_text(&repo_path(ADDENDUM_REPORT));
    for id in DeviationId::ALL.into_iter().skip(14) {
        let prefix = format!("| {id} ");
        let has_row = addendum.lines().any(|line| line.starts_with(&prefix));
        let expected = matches!(
            id,
            DeviationId::E15 | DeviationId::E16 | DeviationId::E17 | DeviationId::E18
        );
        assert_eq!(has_row, expected, "{id}");
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
        for probe in d.probes {
            for change in probe.expect {
                // Always run the workbook side (it must not panic); a workbook error value
                // (`#DIV/0!`, where Python raises) has no Rust counterpart to compare with.
                let workbook = at_probe(change.cell, probe, Deviations::NONE);
```

with:

```rust
        // Decision 15: a correction that refines others is probed on top of them.
        let base = probe_base(d);
        for probe in d.probes {
            for change in probe.expect {
                // Always run the workbook side (it must not panic); a workbook error value
                // (`#DIV/0!`, where Python raises) has no Rust counterpart to compare with.
                let workbook = at_probe(change.cell, probe, base);
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
                let corrected = at_probe(change.cell, probe, Deviations::only(d.id));
```

with:

```rust
                let corrected = at_probe(change.cell, probe, base.with(d.id));
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
/// With every correction off, the cell still holds the workbook snapshot value.
```

with:

```rust
/// The corrections a probe of `d` runs on, on both sides: those `d` refines (decision 15).
fn probe_base(d: &Deviation) -> Deviations {
    d.depends_on
        .iter()
        .fold(Deviations::NONE, |base, &id| base.with(id))
}

/// With every correction off, the cell still holds the workbook snapshot value.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
/// The value at one cell for a probe's inputs, applied to the defaults `dev` implies.
fn at_probe(cell: &str, probe: &Probe, dev: Deviations) -> Value {
    let mut inputs = DesignInputs::defaults_with(dev);
    for &(path, value) in probe.inputs {
        inputs
            .set(path, value.to_value())
            .unwrap_or_else(|e| panic!("probe {:?}: {e}", probe.label));
    }
    cell_values_for(&inputs, dev)
        .remove(cell)
        .unwrap_or_else(|| panic!("{cell} is not a cell of the port"))
}
```

with:

```rust
/// Every value with a workbook cell for a probe's inputs, applied to the defaults `dev` implies.
fn probe_cells(probe: &Probe, dev: Deviations) -> BTreeMap<String, Value> {
    let mut inputs = DesignInputs::defaults_with(dev);
    for &(path, value) in probe.inputs {
        inputs
            .set(path, value.to_value())
            .unwrap_or_else(|e| panic!("probe {:?}: {e}", probe.label));
    }
    cell_values_for(&inputs, dev)
}

/// The value at one cell for a probe's inputs, applied to the defaults `dev` implies.
fn at_probe(cell: &str, probe: &Probe, dev: Deviations) -> Value {
    probe_cells(probe, dev)
        .remove(cell)
        .unwrap_or_else(|| panic!("{cell} is not a cell of the port"))
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn every_applied_engine_correction_is_visible_somewhere() {
```

with:

```rust
#[test]
fn addendum_entries_name_every_cell_their_probes_change() {
    // The Addendum A entries (E15 to E20) name every cell their probes change, downstream
    // cells included, so the registry alone says where each correction shows. The M1 entries
    // (E1 to E14) name the corrected and report-named cells only: their probes change
    // hundreds of sweep and downstream cells (E7, E8), and the broad ones keep golden files.
    let mut failures = Vec::new();
    for d in REGISTRY
        .iter()
        .filter(|d| matches!(d.approval, Approval::Addendum { .. }))
    {
        let base = probe_base(d);
        for probe in d.probes {
            let changed = changed_cells(&probe_cells(probe, base), &probe_cells(probe, base.with(d.id)));
            for cell in changed.iter().filter(|c| !d.cells.contains(&c.as_str())) {
                failures.push(format!("{} {:?}: {cell} changes but is not in cells", d.id, probe.label));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_applied_engine_correction_is_visible_somewhere() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors, among them ``unresolved imports `magcoupling::engine::deviations::ADDENDUM_REPORT`, `magcoupling::engine::deviations::Approval` ``.

- [ ] **Step 3: Give every existing entry its approval and dependencies**

Fourteen entries plus the unit tests' `FAKE` entry get the same two lines after `class:`; the script inserts them and checks the count.

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import re, pathlib
p = pathlib.Path(r"C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs")
s = p.read_text(encoding="utf-8")
s, n = re.subn(
    r"(\n(\s*)class: DeviationClass::(Engine|Documentation),\n)",
    lambda m: m.group(1) + f"{m.group(2)}approval: Approval::AuditRow,\n{m.group(2)}depends_on: &[],\n",
    s,
)
assert n == 15, n
p.write_text(s, encoding="utf-8", newline="\n")
print("inserted", n)
EOF
```

- [ ] **Step 4: Add the ids, the Addendum report, `Approval`, `with`/`without` and the six Planned entries**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! The M1 math audit (`docs/analyses/2026-09-29-magcoupling-math-audit.md`)
//! found fourteen engine errors, E1 to E14, and the user approved every
//! correction on 2026-09-29 (E14 is documentation only). Everything else stays
//! workbook-exact. This registry is the one place that says where the port
//! departs from the workbook and why.
```

with:

```rust
//! The M1 math audit (`docs/analyses/2026-09-29-magcoupling-math-audit.md`)
//! found fourteen engine errors, E1 to E14, and the user approved every
//! correction on 2026-09-29 (E14 is documentation only). The Addendum A
//! verification (`docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`)
//! added E15 to E18 (its audit rows: the E9 residuals at back iron = 0 and the
//! aluminium hub's mismatch screen), E19 (decision 2: the SuperMagnetMan arc
//! parts) and E20 (decision 19: each magnet's own coercivity), all approved on
//! 2026-09-30. Everything else stays workbook-exact. This registry is the one
//! place that says where the port departs from the workbook and why.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! - **Probes.** A correction that changes nothing at default inputs (E7 to
//!   E13) lists `probes`: input overrides on which it shows (the report's
//!   off-default example) and the cells it changes there, with the workbook
//!   and the corrected value.
```

with:

```rust
//! - **Probes.** A correction that changes nothing at default inputs (E7 to
//!   E13, E15 to E20) lists `probes`: input overrides on which it shows (the
//!   report's off-default example) and the cells it changes there, with the
//!   workbook and the corrected value.
//! - **Dependencies.** A correction that refines another one lists it in
//!   `depends_on` (E15, E16 and E17 refine E9: without it the cup is steel).
//!   Its probes run with the dependencies on, on both sides (decision 15).
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
//! Users always get [`Deviations::ALL`] (`compute_all`). The `workbook-parity`
//! feature, enabled only for tests, adds [`Deviations::NONE`] and
//! [`Deviations::only`], so workbook parity and the differential tests compare
//! against the workbook and the Python engine exactly, and each deviation can
//! be checked alone. The GUI never shows the switch.
```

with:

```rust
//! Users always get [`Deviations::ALL`] (`compute_all`). The `workbook-parity`
//! feature, enabled only for tests, adds [`Deviations::NONE`],
//! [`Deviations::only`], [`Deviations::with`] and [`Deviations::without`], so
//! workbook parity and the differential tests compare against the workbook and
//! the Python engine exactly, and each deviation can be checked alone or on top
//! of the ones it refines. The GUI never shows the switch.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
/// The M1 audit report every entry cites.
pub const REPORT: &str = "docs/analyses/2026-09-29-magcoupling-math-audit.md";
```

with:

```rust
/// The M1 audit report E1 to E14 cite.
pub const REPORT: &str = "docs/analyses/2026-09-29-magcoupling-math-audit.md";

/// The Addendum A verification report E15 to E20 cite (section 8 holds the
/// decisions the user approved on 2026-09-30).
pub const ADDENDUM_REPORT: &str =
    "docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md";
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    E13,
    E14,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 14] = [
```

with:

```rust
    E13,
    E14,
    E15,
    E16,
    E17,
    E18,
    E19,
    E20,
}

impl DeviationId {
    /// Every identifier, in report order.
    pub const ALL: [DeviationId; 20] = [
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        DeviationId::E13,
        DeviationId::E14,
    ];
```

with:

```rust
        DeviationId::E13,
        DeviationId::E14,
        DeviationId::E15,
        DeviationId::E16,
        DeviationId::E17,
        DeviationId::E18,
        DeviationId::E19,
        DeviationId::E20,
    ];
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
/// Whether the correction is in the code yet.
```

with:

```rust
/// Where the user approved a correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Approval {
    /// Its row of the M1 audit report ([`REPORT`]), approved on 2026-09-29.
    AuditRow,
    /// These decisions of the Addendum A verification report
    /// ([`ADDENDUM_REPORT`], section 8), approved (option A) on 2026-09-30.
    /// E15 to E18 also have an audit row there; E19 and E20 have none.
    Addendum { decisions: &'static [u32] },
}

/// Whether the correction is in the code yet.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    pub title: &'static str,
    pub class: DeviationClass,
    pub status: DeviationStatus,
```

with:

```rust
    pub title: &'static str,
    pub class: DeviationClass,
    /// Where the user approved it.
    pub approval: Approval,
    /// Corrections this one refines; its probes run with them on, on both
    /// sides (decision 15). Empty for all but E15, E16 and E17 (on E9).
    pub depends_on: &'static [DeviationId],
    pub status: DeviationStatus,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    /// Off-default checks for corrections neutral at defaults (E7 to E13).
    pub probes: &'static [Probe],
}

impl Deviation {
    /// Where the evidence is: the audit report entry.
    pub fn evidence(&self) -> String {
        format!("{REPORT}, entry {}", self.id)
    }
}
```

with:

```rust
    /// Off-default checks for corrections neutral at defaults (E7 to E13,
    /// E15 to E20).
    pub probes: &'static [Probe],
}

impl Deviation {
    /// The report that approves this entry.
    pub fn report(&self) -> &'static str {
        match self.approval {
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

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
    /// TEST-ONLY. Exactly one correction, to check it in isolation.
    pub const fn only(id: DeviationId) -> Self {
        Self { mask: id.bit() }
    }
}
```

with:

```rust
    /// TEST-ONLY. Exactly one correction, to check it in isolation.
    pub const fn only(id: DeviationId) -> Self {
        Self { mask: id.bit() }
    }

    /// TEST-ONLY. This set with `id` switched on as well (decision 15: E15 to
    /// E17 are probed on top of E9, `NONE.with(E9).with(E15)`).
    pub const fn with(self, id: DeviationId) -> Self {
        Self {
            mask: self.mask | id.bit(),
        }
    }

    /// TEST-ONLY. This set with `id` switched off (`ALL.without(E18)`: what
    /// users see, less one correction).
    pub const fn without(self, id: DeviationId) -> Self {
        Self {
            mask: self.mask & !id.bit(),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        workbook_help: &[(
            "clamps.boss_od_mm",
            "At 22 mm only M3 fits; two of them need a 14.5 mm clamp.",
        )],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
];
```

with:

```rust
        workbook_help: &[(
            "clamps.boss_od_mm",
            "At 22 mm only M3 fits; two of them need a 14.5 mm clamp.",
        )],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E15,
        title: "Heat capacity prices the aluminium cup, boss and hub at steel specific heat (E9 residual 1)",
        class: DeviationClass::Engine,
        approval: Approval::Addendum {
            decisions: &[8, 15],
        },
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Planned,
        cells: &[
            "Calculator!C6",
            "Calculator!C111",
            "Calculator!C112",
            "Calculator!C113",
            "Temperature design!C140",
            "Temperature design!C141",
            "Temperature design!C143",
            "Temperature design!C145",
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
            "Temperature design!C20",
        ],
        corrected_formula: "C141 = [m_mag c_NdFeB + (Calculator!C111 + Calculator!C113) c_cup + Calculator!C112 c_hub \
            + Metal design!C128 Materials!C16 + (retainers + endplates) Temperature design!C139 + cap Temperature design!C140] / 1000; \
            c_cup = C140 when the cup is aluminium (E9's gate: C6 != 1 and E9 on), otherwise Materials!C16; \
            c_hub = C140 when C6 != 1, otherwise Materials!C16. The gate reads the same flag the density reads.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E16,
        title: "The disc removed from the web in the aluminium-adapter variant is priced at steel density (E9 residual 2)",
        class: DeviationClass::Engine,
        approval: Approval::Addendum {
            decisions: &[9, 14, 15],
        },
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Planned,
        cells: &[
            "Calculator!C6",
            "Calculator!C111",
            "Metal design!C147",
            "Metal design!C148",
            "Metal design!C149",
            "Metal design!C189",
            "Metal design!C191",
        ],
        corrected_formula: "C189 = pi/4 (C185^2 - Calculator!C39^2) C125 rho_cup, where rho_cup is the density \
            Calculator!C111 uses for the web (C42 when aluminium under E9, otherwise C132), passed from the mass model \
            as one source of truth. The labels C147 and C189 stay (schema parity); their help is reworded (decision 14).",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E17,
        title: "The cup, web and hub eddy losses use the steel skin-depth model, steel constants and steel-circuit fields for aluminium parts (E9 residual 3)",
        class: DeviationClass::Engine,
        approval: Approval::Addendum {
            decisions: &[10, 11, 12, 13, 14, 15],
        },
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Planned,
        cells: &[
            "Calculator!C6",
            "Calculator!C8",
            "Calculator!C20",
            "Calculator!C33",
            "Calculator!C38",
            "Calculator!C60",
            "Metal design!C120",
            "Metal design!C121",
            "Metal design!C122",
            "Metal design!C125",
            "Materials!C43",
            "Temperature design!C114",
            "Temperature design!C116",
            "Temperature design!C17",
            "Temperature design!C18",
            "Temperature design!C19",
            "Temperature design!C20",
            "Temperature design!C22",
            "Temperature design!C123",
            "Temperature design!C124",
            "Temperature design!C125",
            "Temperature design!C130",
            "Temperature design!C131",
            "Temperature design!C132",
            "Temperature design!C134",
            "Temperature design!C145",
            "Temperature design!C146",
            "Temperature design!C147",
            "Temperature design!C148",
            "Temperature design!C149",
            "Temperature design!C150",
            "Temperature design!C151",
            "Temperature design!C154",
            "Temperature design!C155",
            "Temperature design!C156",
            "Temperature design!C157",
            "Temperature design!C161",
            "Temperature design!C169",
            "Temperature design!C170",
            "Temperature design!C171",
            "Temperature design!C172",
            "Temperature design!C173",
            "Temperature design!C175",
            "Temperature design!C176",
            "Temperature design!C177",
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
        corrected_formula: "At C6 = 0 price each aluminium part with the low-Reynolds closed form T1: \
            P = f_end sigma_Al w_e^2 B_free^2 / (2 k^2) d_eff A, d_eff = (1 - e^(-2 k d)) / (2 k), k = p / r, \
            sigma_Al = Materials!C43, f_end = C114, A = 2 pi r L (L = Calculator!C33). Hub: r = Calculator!C8 - Metal design!C120, \
            d = Calculator!C38, B_free = 0.07832 T when the cup is aluminium (E9 on), C116 / 2 when it is steel. \
            Cup: r = Calculator!C60 + Metal design!C121, d = Metal design!C122, B_free = 0.08764 T. \
            Web: (r_mid / p)^2 replaces 1/k^2, r_mid = Calculator!C8 + Calculator!C20 / 2, d = Metal design!C125, \
            and the free-space integral 6.837e-6 T^2 m^2 replaces A B^2. The three free-space fields are Rust-only inputs \
            pinned at 4 s.f. (decision 12; M3 computes them live). The labels C123 to C125 stay; their help is reworded (decision 14).",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E18,
        title: "The adhesive thermal-mismatch screen uses 4140's CTE and modulus for a hub that the mass model makes aluminium",
        class: DeviationClass::Engine,
        approval: Approval::Addendum {
            decisions: &[16],
        },
        depends_on: &[],
        status: DeviationStatus::Planned,
        cells: &[
            "Calculator!C6",
            "Temperature design!C94",
            "Temperature design!C98",
            "Temperature design!C104",
            "Temperature design!C105",
            "Temperature design!C106",
            "Temperature design!C201",
            "Temperature design!C202",
        ],
        corrected_formula: "With C6 != 1 (E15's hub gate: the hub material follows the hub density rule) the Volkersen \
            screen uses aluminium 6061-T6 for the hub: CTE 23.6e-6 /C and modulus 68.9 GPa (Alliance 6061-T6 datasheet); \
            C94 and C98 still show Materials!C17 and C18.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E19,
        title: "The SuperMagnetMan arc parts carry a maximum temperature above the vendor's 60 C, and M5045's grade contradicts its specification grid",
        class: DeviationClass::Engine,
        approval: Approval::Addendum { decisions: &[2] },
        depends_on: &[],
        status: DeviationStatus::Planned,
        cells: &[
            "Calculator!C11",
            "Calculator!C12",
            "Calculator!C22",
            "Calculator!C32",
            "Calculator!C107",
            "Calculator!C108",
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
            "Temperature design!C47",
            "Temperature design!C50",
            "Temperature design!C56",
            "Temperature design!C57",
            "Temperature design!C58",
            "Temperature design!C59",
            "Temperature design!C60",
            "Temperature design!C61",
            "Temperature design!C150",
            "Temperature design!C151",
            "Temperature design!C152",
            "Temperature design!C153",
            "Temperature design!C181",
            "Temperature design!C182",
        ],
        corrected_formula: "Tmax of library rows M5044, M5045 and M5026 = 60 C (the vendor's specification grid, \
            supermagnetman.com/products/m5044, m5045, m5026); M5045 maps to grade N50 (the grid's 'Neodymium 50'; \
            the title's N50M is the unsafe reading of a self-contradicting page). Br stays at the workbook's 1.42 T.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
    Deviation {
        id: DeviationId::E20,
        title: "The demagnetization check uses one N42SH coercivity curve for every magnet and checks only the inner ring",
        class: DeviationClass::Engine,
        approval: Approval::Addendum { decisions: &[19] },
        depends_on: &[],
        status: DeviationStatus::Planned,
        cells: &[
            "Calculator!C11",
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
            "Temperature design!C25",
            "Temperature design!C42",
            "Temperature design!C44",
            "Temperature design!C45",
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
            "Temperature design!C150",
            "Temperature design!C151",
            "Temperature design!C152",
            "Temperature design!C153",
            "Temperature design!C181",
            "Temperature design!C182",
        ],
        corrected_formula: "Each ring is checked against its own magnet: Hcj(20 C) and beta(Hcj) from its grade (a library \
            part's, or the grade picked for manual dimensions), its Br (Calculator!C21, C31) and its rating (C22, C32); \
            C44 and C45 stay as overrides that win when the coercivity source is set to the inputs, and a magnet without \
            a grade uses them. Both rings meet the stored reverse fields C52 to C55, and the ring with the lower magnet \
            limit governs: C42 and C47 to C61 show that ring (the inner ring on a tie; the A-1 plan's decision A13). A \
            positive beta (hard ferrite) takes the signed onset form, where the knee falls as the magnet cools: the \
            onset is a cold limit, and the cold check passes only if both rings pass. \
            N42SH keeps the workbook's 1592 kA/m and -0.005 /C (decisions 17 A, 18 A), so the default design does not move.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
];
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`: unit tests 91 passed (two new), `deviations.rs` 37 passed (the approval test replaces the old one, plus `e15_to_e18_have_audit_rows_and_e19_e20_decisions_only` and `addendum_entries_name_every_cell_their_probes_change`, which a Planned entry passes: it has no probe). A Planned entry changes nothing, so `each_deviation_alone_changes_exactly_its_registered_cells` passes for E15-E20 with no registered cells.

- [ ] **Step 5: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
Every correction below is approved in the M1 math audit report
(`docs/analyses/2026-09-29-magcoupling-math-audit.md`, the row with the same
id) and registered in `src/engine/deviations.rs`
```

with:

```markdown
Every correction below is approved, E1 to E14 in the M1 math audit report
(`docs/analyses/2026-09-29-magcoupling-math-audit.md`, the row with the same
id) and E15 to E20 in the Addendum A verification report
(`docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the decisions
of its section 8 that each entry's `approval` cites), and registered in `src/engine/deviations.rs`
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
`Deviations::only`, `DesignInputs::defaults_with`): the parity and differential
tests to compare against the workbook and the Python engine exactly, and the
registry, metadata, robustness and unit tests to isolate one correction or to
start from the workbook's defaults.
```

with:

```markdown
`Deviations::only`, `Deviations::with`, `Deviations::without`,
`DesignInputs::defaults_with`): the parity and differential
tests to compare against the workbook and the Python engine exactly, and the
registry, metadata, robustness and unit tests to isolate one correction, to
probe one on top of the corrections it refines (`depends_on`: E15 to E17 on E9,
decision 15), or to start from the workbook's defaults.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    deviations: deviations.rs (REGISTRY of the M1 corrections E1-E14, all fourteen Applied; Deviations switch; registry Probe entries and changes_file golden-file names; NONE/only() and restore_workbook_defaults behind the test-only workbook-parity feature)
```

with:

```yaml
    deviations: deviations.rs (REGISTRY of the M1 corrections E1-E14 and the Addendum A corrections E15-E20 (report docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md, section 8 decisions: Approval::Addendum); depends_on (E15-E17 on E9); Deviations switch; registry Probe entries and changes_file golden-file names; NONE/only()/with()/without() and restore_workbook_defaults behind the test-only workbook-parity feature)
```

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task2.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 7: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/deviations.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
feat(magcoupling-rs): registry for the Addendum A corrections (E15-E20 planned)

Approval evidence per entry (the M1 row, or the Addendum A decisions),
depends_on and the test-only Deviations::with/without (decision 15), and
E15-E20 as Planned entries. Nothing changes numerically.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 3: A6 grade table, the part table's vendor data, E3 read from the grade

**Model:** `sonnet` (data transcription with the exact code given; CLAUDE.md section 5). Escalate as in Task 1.

Grade data lives once (spec A6), with one accepted repetition: the part rows keep their own Br and Tmax next to the grade
table, because `tests/static_data.rs` compares the part table with Python's and the workbook values must stay there. The copy cannot
drift silently: `library_parts_agree_with_their_grade_except_where_registered` (`tests/grades.rs`) requires every part's Br and Tmax
to equal its grade's except where a difference is intended, and pins each of those: the N42SH rows' 1.29 T (E3) and the
SuperMagnetMan arcs' 1.42 T (decision 2 A keeps the workbook Br). The N42SH grade's engine beta is the workbook's
-0.005 /C (decision 18 A; Arnold's -0.0055 is `beta_hcj_reference_per_C`) and its Hcj the workbook's 1592 kA/m (decision 17 A),
so the grade is default-neutral when E20 reads it (Task 10). alpha(Br) and density are recorded, not read (Decisions to
confirm, A4). Coating and magnetization direction were read from each vendor's product page on 2026-09-30 (the report left
them for this plan): K&J prints "Nickel-Copper-Nickel (Ni-Cu-Ni)" and "Magnetized Through Thickness" on all twelve blocks,
SuperMagnetMan "Nickel" and "Radially magnetized" on the three arcs. Decision 1 A: the K&J basis for every NdFeB value and the
15 parts as mapped; K&J's four other N42SH stock blocks (B421SH, BX088SH, BY042SH, BY0X02SH) stay recorded in the report, not added,
and no added grade has a K&J stock block, so the part table keeps its 15 rows.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/grades.rs` (the 17 grades, cited per value)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/grades.rs` (every value and citation against the data file; parts resolve to grades)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/mod.rs` (`pub mod grades;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs` (vendor page, coating, magnetization per part; E3's value from the N42SH grade)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Layout and Tests tables)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (engine_files, ported, tests)

**Interfaces:**
- Consumes: `library::{MagnetSpec, MAGNET_LIBRARY, lookup, br_T, N42SH_BR_CORRECTED_T}`, `constants::NDFEB_DENSITY_G_MM3`, `temperature::DemagInputs`, `calibration::CalibrationInputs`.
- Produces:
  - `grades::{Grade, GradeFamily, GradeSources, GRADES: [Grade; 17], N42SH: &Grade, grade(id: &str) -> Option<&'static Grade>}`; `Grade` fields `id, name, family, br_T, hcj20_kA_m, hcb_kA_m, bhmax_kJ_m3, alpha_br_per_C, beta_hcj_per_C` (what E20 uses), `beta_hcj_reference_per_C` (sourced), `coefficient_range_C: Option<[f64; 2]>, mu_rec: Option<f64>, tmax_C, density_g_mm3, sources, notes`;
  - URL consts `grades::{KJ_SPECS, ARNOLD_RECOMA, ARNOLD_APEEM_2006, ARNOLD_TN_0303, ECLIPSE_FERRITE, ALLIANCE_C5, ALLIANCE_BONDED_NEO}`;
  - `MagnetSpec` gains `page, coating, magnetization: &'static str`; consts `library::{KJ_COATING, KJ_MAGNETIZATION, SMM_COATING, SMM_MAGNETIZATION}`;
  - `library::N42SH_BR_CORRECTED_T == grades::N42SH.br_T` (1.30).

- [ ] **Step 1: Write the failing tests**

Create `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/grades.rs` with exactly this content:

```rust
//! The A6 grade table and the part table against the approved Addendum A data
//! file (`docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`): every
//! value and every citation, every part resolving to a grade, and the places
//! where a part's workbook value differs from its grade on purpose.

mod common;

use common::{read_json, repo_path};
use magcoupling::engine::constants::NDFEB_DENSITY_G_MM3;
use magcoupling::engine::grades::{GRADES, Grade, GradeFamily, N42SH, grade};
use magcoupling::engine::library::{MAGNET_LIBRARY, N42SH_BR_CORRECTED_T};
use magcoupling::engine::temperature::DemagInputs;

const DATA: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-data.json";

fn data() -> serde_json::Value {
    read_json(&repo_path(DATA))
}

fn f(json: &serde_json::Value) -> f64 {
    json.as_f64()
        .unwrap_or_else(|| panic!("{json} is a number"))
}

fn opt_f(json: &serde_json::Value) -> Option<f64> {
    json.as_f64()
}

fn opt_s(json: &serde_json::Value) -> Option<&str> {
    json.as_str()
}

#[test]
fn grades_equal_the_addendum_data_file() {
    let doc = data();
    let grades = doc["grades"].as_object().expect("a grades object");
    // serde_json keeps object keys sorted: compare the sets (the table itself keeps
    // the data file's order).
    let ids: std::collections::BTreeSet<&str> = grades.keys().map(String::as_str).collect();
    let rust: std::collections::BTreeSet<&str> = GRADES.iter().map(|g| g.id).collect();
    assert_eq!(rust, ids, "grade ids");
    assert_eq!(GRADES.len(), ids.len(), "no duplicate id");
    for g in &GRADES {
        let j = &grades[g.id];
        let fields = &j["fields"];
        let lit = &j["engine_literals"];
        let id = g.id;
        assert_eq!(g.name, j["display_name"].as_str().expect("a name"), "{id}");
        // Engine literals, bit for bit (typed as literals, never converted).
        assert_eq!(g.br_T, f(&lit["br_T"]), "{id} Br");
        assert_eq!(g.hcj20_kA_m, f(&lit["hcj20_kA_m"]), "{id} Hcj");
        assert_eq!(g.alpha_br_per_C, f(&lit["alpha_br_per_C"]), "{id} alpha");
        assert_eq!(
            g.beta_hcj_reference_per_C,
            f(&lit["beta_hcj_per_C"]),
            "{id} beta"
        );
        assert_eq!(g.tmax_C, f(&lit["tmax_C"]), "{id} Tmax");
        assert_eq!(g.density_g_mm3, f(&lit["density_g_mm3"]), "{id} density");
        // The engine's beta keeps a workbook default where the data file records one.
        let engine_beta = opt_f(&j["workbook_default_literals"]["beta_hcj_per_C"])
            .unwrap_or_else(|| f(&lit["beta_hcj_per_C"]));
        assert_eq!(g.beta_hcj_per_C, engine_beta, "{id} engine beta");
        // Display fields, in the data file's own units.
        assert_eq!(g.hcb_kA_m, f(&fields["Hcb_kA_m"]["value"]), "{id} Hcb");
        assert_eq!(
            g.bhmax_kJ_m3,
            f(&fields["BHmax_kJ_m3"]["value"]),
            "{id} BHmax"
        );
        assert_eq!(g.mu_rec, opt_f(&fields["mu_rec"]["value"]), "{id} mu_rec");
        let range = fields["alpha_Br_pct_per_C"]["temp_range_C"]
            .as_array()
            .map(|r| [f(&r[0]), f(&r[1])]);
        assert_eq!(g.coefficient_range_C, range, "{id} range");
        // Every value cites the data file's source.
        let s = &g.sources;
        for (what, rust, key) in [
            ("br", s.br, "Br_T"),
            ("hcj", s.hcj, "Hcj_kA_m"),
            ("hcb", s.hcb, "Hcb_kA_m"),
            ("bhmax", s.bhmax, "BHmax_kJ_m3"),
            ("alpha", s.alpha, "alpha_Br_pct_per_C"),
            ("beta", s.beta, "beta_Hcj_pct_per_C"),
            ("tmax", s.tmax, "Tmax_C"),
            ("density", s.density, "density_g_cm3"),
        ] {
            assert_eq!(
                Some(rust),
                opt_s(&fields[key]["value_source_url"]),
                "{id} {what} source"
            );
        }
        assert_eq!(
            s.mu_rec,
            opt_s(&fields["mu_rec"]["value_source_url"]),
            "{id} mu_rec source"
        );
        assert_eq!(
            s.coefficient_range,
            opt_s(&fields["alpha_Br_pct_per_C"]["temp_range_source_url"]),
            "{id} range source"
        );
    }
}

#[test]
fn every_library_part_resolves_to_a_grade() {
    // Spec, Addendum testing: "every library part resolves to a grade" (the workbook's
    // grade text, N50M included).
    for spec in &MAGNET_LIBRARY {
        assert!(
            grade(spec.grade).is_some(),
            "{} names {}",
            spec.part,
            spec.grade
        );
    }
}

#[test]
fn library_parts_agree_with_their_grade_except_where_registered() {
    // Grade data lives once: a part's workbook Br and Tmax equal its grade's, except
    // the N42SH rows' 1.29 T (E3 corrects it to the grade's 1.30 T) and the SuperMagnetMan
    // arcs' 1.42 T (decision 2 A keeps the workbook Br; K&J's N50 minimum is 1.41 T).
    for spec in &MAGNET_LIBRARY {
        let g: &Grade = grade(spec.grade).expect("resolves");
        assert_eq!(spec.tmax_C, g.tmax_C, "{} Tmax", spec.part);
        let expected_br = match (spec.grade, spec.vendor) {
            ("N42SH", _) => 1.29,
            (_, "SuperMagnetMan") => 1.42,
            _ => g.br_T,
        };
        assert_eq!(spec.br_T, expected_br, "{} Br", spec.part);
    }
}

#[test]
fn e3_corrects_to_the_n42sh_grade_minimum() {
    assert_eq!(N42SH_BR_CORRECTED_T, N42SH.br_T);
    assert_eq!(N42SH.br_T, 1.30);
}

#[test]
fn n42sh_keeps_the_workbook_coercivity() {
    // Decisions 17 A and 18 A: the default part's grade is default-neutral for E20.
    let demag = DemagInputs::default();
    assert_eq!(N42SH.hcj20_kA_m, demag.hcj20_kA_m);
    assert_eq!(N42SH.beta_hcj_per_C, demag.beta_hcj_per_C);
    assert_eq!(N42SH.beta_hcj_reference_per_C, -0.0055);
}

#[test]
fn sintered_ndfeb_grades_are_bit_equal_to_the_engine_constants() {
    // Report 2.5: alpha(Br) and density of every sintered NdFeB grade equal the engine's
    // -0.0012 /°C (Calibration!C22) and 0.0075 g/mm³, as literals.
    let alpha = magcoupling::engine::calibration::CalibrationInputs::default().alpha_br_per_C;
    for g in GRADES.iter().filter(|g| g.family == GradeFamily::NdFeB) {
        assert_eq!(g.alpha_br_per_C, alpha, "{}", g.id);
        assert_eq!(g.density_g_mm3, NDFEB_DENSITY_G_MM3, "{}", g.id);
    }
}

#[test]
fn only_ferrite_has_a_positive_beta() {
    for g in &GRADES {
        let positive = g.beta_hcj_per_C > 0.0;
        assert_eq!(positive, g.family == GradeFamily::Ferrite, "{}", g.id);
        assert_eq!(
            g.beta_hcj_per_C.signum(),
            g.beta_hcj_reference_per_C.signum(),
            "{}",
            g.id
        );
    }
    assert_eq!(grade("Y30").map(|g| g.beta_hcj_per_C), Some(0.0035)); // decision 3 A
}

#[test]
fn every_part_cites_its_vendor_page_for_coating_and_magnetization() {
    for spec in &MAGNET_LIBRARY {
        let page = spec.page;
        match spec.vendor {
            "K&J" => {
                assert_eq!(
                    page,
                    format!(
                        "https://www.kjmagnetics.com/proddetail.asp?prod={}",
                        spec.part
                    )
                );
                assert_eq!(
                    spec.coating, "Nickel-Copper-Nickel (Ni-Cu-Ni)",
                    "{}",
                    spec.part
                );
                assert_eq!(
                    spec.magnetization, "Magnetized through thickness",
                    "{}",
                    spec.part
                );
            }
            "SuperMagnetMan" => {
                assert_eq!(
                    page,
                    format!(
                        "https://supermagnetman.com/products/{}",
                        spec.part.to_lowercase()
                    )
                );
                assert_eq!(spec.coating, "Nickel", "{}", spec.part);
                assert_eq!(spec.magnetization, "Radially magnetized", "{}", spec.part);
            }
            other => panic!("{}: unexpected vendor {other}", spec.part),
        }
    }
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test grades 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors, among them ``unresolved import `magcoupling::engine::grades` ``.

- [ ] **Step 3: Create the grade table**

Create `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/grades.rs` with exactly this content:

```rust
//! Magnet grades (spec Addendum A6): one record per grade, so grade data lives
//! once. Library parts name their grade ([`crate::engine::library::MagnetSpec::grade`]);
//! a grade can also be picked for manual dimensions
//! (`coupling.magnets.grade_inner`, `grade_outer`).
//!
//! Data: `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`, key
//! `"grades"`, approved with the Addendum A verification (decisions 1, 3, 4, 17
//! and 18, option A; resolutions R1 to R17 in its section 2.4). Every value
//! cites its source in [`GradeSources`]; `tests/grades.rs` compares every value
//! and citation with the data file. Values are the data file's engine-unit
//! literals (`engine_literals`), typed as literals and never converted at load
//! time (-0.55 / 100 is -0.0055000000000000005, not -0.0055).
//!
//! What the engine reads: Br and Tmax of a grade picked for manual dimensions
//! (`model::resolve_magnets`), E3's corrected N42SH Br ([`N42SH`]), and with
//! E20 the Hcj and beta of each ring's grade (the demagnetization block checks
//! both rings; the weaker governs, A-1 plan decision A13).
//! Hcb, (BH)max, mu_rec, the coefficient ranges and the reference beta are for
//! display. alpha(Br) and density are recorded but not read: the calculator
//! keeps its single alpha input (Calibration!C22) and the NdFeB magnet density
//! for every magnet (decision table of the A-1 plan).

/// K&J Magnetics, Neodymium Magnet Specifications & Tolerances (the NdFeB basis, decision D1/1).
pub const KJ_SPECS: &str = "https://www.kjmagnetics.com/neodymium-magnet-specifications.asp";
/// Arnold Recoma sintered SmCo, combined datasheet 160301 (p4 Recoma 20, p8 Recoma 26, p12 Recoma 30).
pub const ARNOLD_RECOMA: &str =
    "https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf";
/// Arnold (Constantinides), APEEM 2006, slide 23: recoil permeability "about 1.05".
pub const ARNOLD_APEEM_2006: &str = "https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Manufacturing-and-performance-comparison-between-bonded-and-sintered-permanent-magnets-Constantinides-APEEM-2006-psn-hi-res.pdf";
/// Arnold TECHNotes TN 0303 (family coefficients, averages over about 20 to 120 °C).
pub const ARNOLD_TN_0303: &str =
    "https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/TN_0303_rev_150715.pdf";
/// Eclipse Magnetics, Ferrite/Ceramic Magnets Datasheet.
pub const ECLIPSE_FERRITE: &str =
    "https://www.eclipsemagnetics.com/site/assets/files/19602/ferrite_ceramic_datasheet.pdf";
/// Alliance LLC, Ferrite C-5 (the +0.35 %/°C beta, decision 3 A).
pub const ALLIANCE_C5: &str =
    "https://allianceorg.com/magnetic-materials/ceramic-magnets/ferrite-c-5/";
/// Alliance LLC, Compression Bonded Neo (BCN-19).
pub const ALLIANCE_BONDED_NEO: &str =
    "https://allianceorg.com/magnetic-materials/bonded-magnets/compression-bonded-neo/";

/// The material family of a grade.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GradeFamily {
    /// Sintered NdFeB.
    NdFeB,
    /// Sintered Sm2Co17.
    SmCo2_17,
    /// Sintered SmCo5.
    SmCo1_5,
    /// Sintered hard (strontium) ferrite: its beta(Hcj) is positive.
    Ferrite,
    /// Isotropic compression-bonded NdFeB.
    BondedNdFeB,
}

/// Where each value of a grade comes from (a URL of the data file's `sources`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GradeSources {
    pub br: &'static str,
    pub hcj: &'static str,
    pub hcb: &'static str,
    pub bhmax: &'static str,
    pub alpha: &'static str,
    pub beta: &'static str,
    /// The measured range of alpha and beta; `None` where none is printed.
    pub coefficient_range: Option<&'static str>,
    /// `None` where mu_rec is not sourced.
    pub mu_rec: Option<&'static str>,
    pub tmax: &'static str,
    pub density: &'static str,
}

/// One magnet grade (the selected value of each field: the published minimum
/// where one exists, report section 2.1).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct Grade {
    /// The data file's key; parts and the grade inputs name a grade by it.
    pub id: &'static str,
    /// Display name.
    pub name: &'static str,
    pub family: GradeFamily,
    /// Remanence at 20 °C [T].
    pub br_T: f64,
    /// Intrinsic coercivity at 20 °C [kA/m].
    pub hcj20_kA_m: f64,
    /// Normal coercivity [kA/m] (display).
    pub hcb_kA_m: f64,
    /// Maximum energy product [kJ/m³] (display).
    pub bhmax_kJ_m3: f64,
    /// Reversible temperature coefficient of Br [1/°C] (recorded, not read).
    pub alpha_br_per_C: f64,
    /// beta(Hcj) the engine uses with E20 [1/°C]: the reference value, except
    /// N42SH, which keeps the workbook's -0.005 (decision 18 A).
    pub beta_hcj_per_C: f64,
    /// The sourced beta(Hcj) [1/°C]. Positive for hard ferrite: its coercivity
    /// falls as it cools, so its demagnetization risk is at cold.
    pub beta_hcj_reference_per_C: f64,
    /// The temperature range [°C] alpha and beta were measured over.
    pub coefficient_range_C: Option<[f64; 2]>,
    /// Recoil permeability (display; class-level value).
    pub mu_rec: Option<f64>,
    /// Maximum operating temperature [°C] (the calibration rating of the demag block).
    pub tmax_C: f64,
    /// Density [g/mm³] (recorded, not read).
    pub density_g_mm3: f64,
    pub sources: GradeSources,
    /// The report's resolutions (R#) and flags for this grade.
    pub notes: &'static str,
}

/// Every grade, in the data file's order.
pub const GRADES: [Grade; 17] = [
    Grade {
        id: "N35",
        name: "N35",
        family: GradeFamily::NdFeB,
        br_T: 1.17,
        hcj20_kA_m: 954.9,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 262.6,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17 (K&J low end, not stated as guaranteed).",
    },
    Grade {
        id: "N42",
        name: "N42",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 954.9,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N48",
        name: "N48",
        family: GradeFamily::NdFeB,
        br_T: 1.38,
        hcj20_kA_m: 954.9,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 358.1,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N48-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N48-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N48-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17 (Hcj: K&J 954.9, Arnold 875).",
    },
    Grade {
        id: "N52",
        name: "N52",
        family: GradeFamily::NdFeB,
        br_T: 1.45,
        hcj20_kA_m: 875.4,
        hcb_kA_m: 836.0,
        bhmax_kJ_m3: 393.9,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 60.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R3 (Hcb 836 Arnold: K&J's 891.3 exceeds Hcj), R4 (Tmax 80 K&J; Arnold catalog 60, Eclipse 70), R16, R17.",
    },
    Grade {
        id: "N50",
        name: "N50",
        family: GradeFamily::NdFeB,
        br_T: 1.41,
        hcj20_kA_m: 875.4,
        hcb_kA_m: 875.4,
        bhmax_kJ_m3: 382.0,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0062,
        beta_hcj_reference_per_C: -0.0062,
        coefficient_range_C: Some([20.0, 80.0]),
        mu_rec: Some(1.05),
        tmax_C: 80.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "Library grade (M5044, M5026; M5045 under E19), not in the spec's A6 list (decision 30). R5, R16, R17.",
    },
    Grade {
        id: "N42M",
        name: "N42M",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 1114.1,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.006,
        beta_hcj_reference_per_C: -0.006,
        coefficient_range_C: Some([20.0, 100.0]),
        mu_rec: Some(1.05),
        tmax_C: 100.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42M-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42M-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42M-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N50M",
        name: "N50M",
        family: GradeFamily::NdFeB,
        br_T: 1.41,
        hcj20_kA_m: 1114.1,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 382.0,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.00675,
        beta_hcj_reference_per_C: -0.00675,
        coefficient_range_C: Some([20.0, 100.0]),
        mu_rec: Some(1.05),
        tmax_C: 100.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50M-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50M-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50M-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "Library grade (M5045 as stored in the workbook), not in the spec's A6 list. R1 (Hcb), R6 (Tmax), R16, R17.",
    },
    Grade {
        id: "N42H",
        name: "N42H",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 1352.8,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0057,
        beta_hcj_reference_per_C: -0.0057,
        coefficient_range_C: Some([20.0, 120.0]),
        mu_rec: Some(1.05),
        tmax_C: 120.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42H-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42H-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42H-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N42SH",
        name: "N42SH",
        family: GradeFamily::NdFeB,
        br_T: 1.3,
        hcj20_kA_m: 1592.0,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 318.3,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.005,
        beta_hcj_reference_per_C: -0.0055,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 150.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "Default grade (B842SH on both rings). Hcj 1592 Arnold (decision 17 A; K&J 1591.5). The engine keeps the workbook beta -0.005 /C; Arnold's -0.0055 is the reference (decision 18 A). R16, R17.",
    },
    Grade {
        id: "N38UH",
        name: "N38UH",
        family: GradeFamily::NdFeB,
        br_T: 1.22,
        hcj20_kA_m: 1989.4,
        hcb_kA_m: 907.2,
        bhmax_kJ_m3: 286.5,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0051,
        beta_hcj_reference_per_C: -0.0051,
        coefficient_range_C: Some([20.0, 180.0]),
        mu_rec: Some(1.05),
        tmax_C: 180.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38UH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38UH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38UH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N35EH",
        name: "N35EH",
        family: GradeFamily::NdFeB,
        br_T: 1.17,
        hcj20_kA_m: 2387.3,
        hcb_kA_m: 859.4,
        bhmax_kJ_m3: 262.6,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.0047,
        beta_hcj_reference_per_C: -0.0047,
        coefficient_range_C: Some([20.0, 200.0]),
        mu_rec: Some(1.05),
        tmax_C: 200.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35EH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35EH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35EH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R16, R17.",
    },
    Grade {
        id: "N33AH",
        name: "N33AH",
        family: GradeFamily::NdFeB,
        br_T: 1.14,
        hcj20_kA_m: 2705.6,
        hcb_kA_m: 811.7,
        bhmax_kJ_m3: 246.7,
        alpha_br_per_C: -0.0012,
        beta_hcj_per_C: -0.00375,
        beta_hcj_reference_per_C: -0.00375,
        coefficient_range_C: Some([20.0, 220.0]),
        mu_rec: Some(1.05),
        tmax_C: 220.0,
        density_g_mm3: 0.0075,
        sources: GradeSources {
            br: KJ_SPECS,
            hcj: KJ_SPECS,
            hcb: KJ_SPECS,
            bhmax: KJ_SPECS,
            alpha: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N33AH-151021.pdf",
            beta: "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N33AH-151021.pdf",
            coefficient_range: Some(
                "https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N33AH-151021.pdf",
            ),
            mu_rec: Some(KJ_SPECS),
            tmax: KJ_SPECS,
            density: KJ_SPECS,
        },
        notes: "R2 ((BH)max K&J 246.7; Arnold 215 min), R16, R17.",
    },
    Grade {
        id: "SmCo_2_17_26",
        name: "Recoma 26 (Sm2Co17 grade 26)",
        family: GradeFamily::SmCo2_17,
        br_T: 1.0,
        hcj20_kA_m: 1200.0,
        hcb_kA_m: 680.0,
        bhmax_kJ_m3: 185.0,
        alpha_br_per_C: -0.00035,
        beta_hcj_per_C: -0.00247,
        beta_hcj_reference_per_C: -0.00247,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 350.0,
        density_g_mm3: 0.0083,
        sources: GradeSources {
            br: ARNOLD_RECOMA,
            hcj: ARNOLD_RECOMA,
            hcb: ARNOLD_RECOMA,
            bhmax: ARNOLD_RECOMA,
            alpha: ARNOLD_RECOMA,
            beta: ARNOLD_RECOMA,
            coefficient_range: Some(ARNOLD_RECOMA),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ARNOLD_RECOMA,
            density: ARNOLD_RECOMA,
        },
        notes: "Arnold Recoma 26, the low-Hcj sub-grade (decision 4 A; typ Hcj 2000). R9, R10, R11 (Tmax 350: may be considerably lower at a low load line), R14 (mu_rec class value).",
    },
    Grade {
        id: "SmCo_2_17_30",
        name: "Recoma 30 (Sm2Co17 grade 30)",
        family: GradeFamily::SmCo2_17,
        br_T: 1.09,
        hcj20_kA_m: 1040.0,
        hcb_kA_m: 700.0,
        bhmax_kJ_m3: 215.0,
        alpha_br_per_C: -0.00035,
        beta_hcj_per_C: -0.0025,
        beta_hcj_reference_per_C: -0.0025,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 250.0,
        density_g_mm3: 0.0083,
        sources: GradeSources {
            br: ARNOLD_RECOMA,
            hcj: ARNOLD_RECOMA,
            hcb: ARNOLD_RECOMA,
            bhmax: ARNOLD_RECOMA,
            alpha: ARNOLD_RECOMA,
            beta: ARNOLD_RECOMA,
            coefficient_range: Some(ARNOLD_RECOMA),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ARNOLD_RECOMA,
            density: ARNOLD_RECOMA,
        },
        notes: "Arnold Recoma 30, the low-Hcj sub-grade (decision 4 A; 30HE/30S reach 1500/1750). R9, R10, R14.",
    },
    Grade {
        id: "SmCo_1_5_20",
        name: "Recoma 20 (SmCo5 grade 20)",
        family: GradeFamily::SmCo1_5,
        br_T: 0.85,
        hcj20_kA_m: 2000.0,
        hcb_kA_m: 640.0,
        bhmax_kJ_m3: 140.0,
        alpha_br_per_C: -0.00045,
        beta_hcj_per_C: -0.0019,
        beta_hcj_reference_per_C: -0.0019,
        coefficient_range_C: Some([20.0, 150.0]),
        mu_rec: Some(1.05),
        tmax_C: 250.0,
        density_g_mm3: 0.0084,
        sources: GradeSources {
            br: ARNOLD_RECOMA,
            hcj: ARNOLD_RECOMA,
            hcb: ARNOLD_RECOMA,
            bhmax: ARNOLD_RECOMA,
            alpha: ARNOLD_RECOMA,
            beta: ARNOLD_RECOMA,
            coefficient_range: Some(ARNOLD_RECOMA),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ARNOLD_RECOMA,
            density: ARNOLD_RECOMA,
        },
        notes: "Arnold Recoma 20 (decision 4 A). R8 (beta -0.19 is 37-52 % smaller than the SmCo5 family values: the non-conservative side), R10, R14.",
    },
    Grade {
        id: "Y30",
        name: "Ferrite Y30 (= C5)",
        family: GradeFamily::Ferrite,
        br_T: 0.37,
        hcj20_kA_m: 180.0,
        hcb_kA_m: 175.0,
        bhmax_kJ_m3: 26.0,
        alpha_br_per_C: -0.002,
        beta_hcj_per_C: 0.0035,
        beta_hcj_reference_per_C: 0.0035,
        coefficient_range_C: Some([20.0, 120.0]),
        mu_rec: Some(1.05),
        tmax_C: 250.0,
        density_g_mm3: 0.005,
        sources: GradeSources {
            br: ECLIPSE_FERRITE,
            hcj: ECLIPSE_FERRITE,
            hcb: ECLIPSE_FERRITE,
            bhmax: ECLIPSE_FERRITE,
            alpha: ECLIPSE_FERRITE,
            beta: ALLIANCE_C5,
            coefficient_range: Some(ARNOLD_TN_0303),
            mu_rec: Some(ARNOLD_APEEM_2006),
            tmax: ECLIPSE_FERRITE,
            density: ECLIPSE_FERRITE,
        },
        notes: "Positive beta: coercivity falls as the magnet cools (decision 3 A: +0.35 %/C, Alliance C-5; Eclipse and TN 0303 give +0.27). Coefficient range borrowed from TN 0303's Ferrite 8 row (R15). Density: range midpoint (R12). R14.",
    },
    Grade {
        id: "Bonded_NdFeB_BCN19",
        name: "Bonded NdFeB (Alliance BCN-19)",
        family: GradeFamily::BondedNdFeB,
        br_T: 0.65,
        hcj20_kA_m: 880.0,
        hcb_kA_m: 416.0,
        bhmax_kJ_m3: 72.0,
        alpha_br_per_C: -0.0014,
        beta_hcj_per_C: -0.0036,
        beta_hcj_reference_per_C: -0.0036,
        coefficient_range_C: None,
        mu_rec: None,
        tmax_C: 140.0,
        density_g_mm3: 0.0058,
        sources: GradeSources {
            br: ALLIANCE_BONDED_NEO,
            hcj: ALLIANCE_BONDED_NEO,
            hcb: ALLIANCE_BONDED_NEO,
            bhmax: ALLIANCE_BONDED_NEO,
            alpha: ALLIANCE_BONDED_NEO,
            beta: ALLIANCE_BONDED_NEO,
            coefficient_range: None,
            mu_rec: None,
            tmax: ALLIANCE_BONDED_NEO,
            density: ALLIANCE_BONDED_NEO,
        },
        notes: "Alliance BCN-19 (about 0.65 T). alpha and beta are Alliance's operating-point coefficients (Bd, Hd), no range printed (R13). mu_rec not sourced (Arnold gives 1.1-1.7 for isotropic bonded Neo). Density: range midpoint (R12).",
    },
];

/// The default part's grade (B842SH on both rings): E3's corrected Br is its
/// published minimum (decision D1).
pub const N42SH: &Grade = &GRADES[8];

/// The grade named by exact text (like the part lookup); `None` for an empty or
/// unknown name.
pub fn grade(id: &str) -> Option<&'static Grade> {
    if id.is_empty() {
        return None;
    }
    GRADES.iter().find(|g| g.id == id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n42sh_is_the_named_grade() {
        assert_eq!(N42SH.id, "N42SH");
        assert_eq!(grade("N42SH"), Some(N42SH));
    }

    #[test]
    fn lookup_is_exact_text() {
        assert_eq!(grade("Y30").map(|g| g.family), Some(GradeFamily::Ferrite));
        for miss in ["", "n42sh", "N42SH ", "N 42", "Recoma 26"] {
            assert_eq!(grade(miss), None, "{miss:?}");
        }
        let ids: std::collections::BTreeSet<&str> = GRADES.iter().map(|g| g.id).collect();
        assert_eq!(ids.len(), GRADES.len(), "grade ids are unique");
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/mod.rs`, replace:

```rust
pub mod deviations;
pub mod library;
```

with:

```rust
pub mod deviations;
pub mod grades;
pub mod library;
```

- [ ] **Step 4: Give the parts their vendor data and E3 its grade**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
//! Stock magnet library ('Magnet library' sheet).
//!
//! Port of `reference/magcoupling-py/magcoupling/library.py`. The calculator
//! looks magnets up by exact part text, like the workbook's INDEX/MATCH.
//! Dimensions in mm, Br is the 20 °C remanence in tesla, tmax the supplier
//! rating in °C. The rows hold the WORKBOOK values (1.29 T for the N42SH rows),
//! so the static-data test still compares them with the Python engine. The
//! approved correction E3 (N42SH remanence, decision D1) is applied by
//! [`br_T`], which the Calculator calls where it resolves a part
//! (`model::resolve_magnets`). The 3D field run (`fields3d`, M3) must be rerun
//! with the corrected Br.

use super::deviations::{DeviationId, Deviations};

/// E3: the N42SH rows' remanence with the approved correction: the vendor's
/// grade minimum (K&J 13.0 kG), like the library's other K&J rows (decision D1).
pub const N42SH_BR_CORRECTED_T: f64 = 1.30;

/// The 20 °C remanence the Calculator uses for a library part: the row's
/// workbook value, or with E3 on the corrected N42SH value.
#[allow(non_snake_case)]
pub fn br_T(spec: &MagnetSpec, dev: Deviations) -> f64 {
    if dev.is_on(DeviationId::E3) && spec.grade == "N42SH" {
        N42SH_BR_CORRECTED_T
    } else {
        spec.br_T
    }
}
```

with:

```rust
//! Stock magnet library ('Magnet library' sheet): the part table of Addendum A6.
//!
//! Port of `reference/magcoupling-py/magcoupling/library.py`. The calculator
//! looks magnets up by exact part text, like the workbook's INDEX/MATCH.
//! Dimensions in mm, Br is the 20 °C remanence in tesla, tmax the supplier
//! rating in °C. The rows hold the WORKBOOK values (1.29 T for the N42SH rows),
//! so the static-data test still compares them with the Python engine. Each
//! part names its grade ([`crate::engine::grades`], where grade data lives
//! once); `tests/grades.rs` checks that every part resolves to a grade and
//! that its workbook Br and Tmax equal the grade's, except where a registered
//! correction or decision says otherwise. Coating and magnetization direction
//! come from the vendor's product page ([`MagnetSpec::page`], read 2026-09-30).
//! The approved correction E3 (N42SH remanence, decision D1) is applied by
//! [`br_T`], which the Calculator calls where it resolves a part
//! (`model::resolve_magnets`). The 3D field run (`fields3d`, M3) must be rerun
//! with the corrected Br.

use super::deviations::{DeviationId, Deviations};
use super::grades;

/// E3: the N42SH rows' remanence with the approved correction: the N42SH
/// grade's published minimum (K&J 13.0 kG), like the library's other K&J rows
/// (decision D1).
pub const N42SH_BR_CORRECTED_T: f64 = grades::N42SH.br_T;

/// The 20 °C remanence the Calculator uses for a library part: the row's
/// workbook value, or with E3 on the corrected N42SH value.
#[allow(non_snake_case)]
pub fn br_T(spec: &MagnetSpec, dev: Deviations) -> f64 {
    if dev.is_on(DeviationId::E3) && spec.grade == grades::N42SH.id {
        N42SH_BR_CORRECTED_T
    } else {
        spec.br_T
    }
}

/// K&J's plating on every library block (product pages, 2026-09-30).
pub const KJ_COATING: &str = "Nickel-Copper-Nickel (Ni-Cu-Ni)";
/// K&J's magnetization direction on every library block.
pub const KJ_MAGNETIZATION: &str = "Magnetized through thickness";
/// SuperMagnetMan's coating on the three arc parts (specification grids).
pub const SMM_COATING: &str = "Nickel";
/// SuperMagnetMan's magnetization direction on the three arc parts.
pub const SMM_MAGNETIZATION: &str = "Radially magnetized";
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
    /// Supplier maximum operating temperature [°C].
    pub tmax_C: f64,
    pub notes: &'static str,
}
```

with:

```rust
    /// Supplier maximum operating temperature [°C].
    pub tmax_C: f64,
    pub notes: &'static str,
    /// The vendor's product page: the source of `coating` and `magnetization`.
    pub page: &'static str,
    /// Plating or coating, as the vendor states it.
    pub coating: &'static str,
    /// Magnetization direction, as the vendor states it.
    pub magnetization: &'static str,
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
        grade,
        br_T,
        tmax_C,
        notes,
    }
}
```

with:

```rust
        grade,
        br_T,
        tmax_C,
        notes,
        page: "",
        coating: "",
        magnetization: "",
    }
}

impl MagnetSpec {
    /// The row with the vendor's product page and what it states.
    const fn listed(
        self,
        page: &'static str,
        coating: &'static str,
        magnetization: &'static str,
    ) -> Self {
        Self {
            page,
            coating,
            magnetization,
            ..self
        }
    }
}
```

Then replace the whole `MAGNET_LIBRARY` const (each row gains `.listed(page, coating, magnetization)`):

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
/// The library, in the Python row order (`_ROWS`).
pub const MAGNET_LIBRARY: [MagnetSpec; 15] = [
    row(
        "B842SH",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1/2 x 1/4 x 1/8 in, magnetized through 1/8 in",
    ),
    row(
        "B842",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N42",
        1.30,
        80.0,
        "",
    ),
    row(
        "B842-N52",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N52",
        1.45,
        80.0,
        "",
    ),
    row(
        "B822",
        "K&J",
        "block",
        [12.7, 3.17, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/8 x 1/8 in",
    ),
    row(
        "B862",
        "K&J",
        "block",
        [12.7, 9.5, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 3/8 x 1/8 in",
    ),
    row(
        "B882",
        "K&J",
        "block",
        [12.7, 12.7, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/2 x 1/8 in",
    ),
    row(
        "B882-N52",
        "K&J",
        "block",
        [12.7, 12.7, 3.17],
        "N52",
        1.45,
        80.0,
        "",
    ),
    row(
        "B861",
        "K&J",
        "block",
        [12.7, 9.5, 1.59],
        "N42",
        1.30,
        80.0,
        "1/2 x 3/8 x 1/16 in",
    ),
    row(
        "B881",
        "K&J",
        "block",
        [12.7, 12.7, 1.59],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/2 x 1/16 in (check stock)",
    ),
    row(
        "B442",
        "K&J",
        "block",
        [6.35, 6.35, 3.17],
        "N42",
        1.30,
        80.0,
        "1/4 x 1/4 x 1/8 in",
    ),
    row(
        "BX042SH",
        "K&J",
        "block",
        [25.4, 6.35, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1 x 1/4 x 1/8 in",
    ),
    row(
        "BX082SH",
        "K&J",
        "block",
        [25.4, 12.7, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1 x 1/2 x 1/8 in",
    ),
    row(
        "M5044",
        "SuperMagnetMan",
        "arc",
        [13.0, 5.5, 1.31],
        "N50",
        1.42,
        80.0,
        "22.60 OD x 19.97 ID x 13, 29.8 deg, 12 pcs; width = mean arc length",
    ),
    row(
        "M5045",
        "SuperMagnetMan",
        "arc",
        [6.56, 5.6, 1.12],
        "N50M",
        1.42,
        100.0,
        "22.70 OD x 20.47 ID x 6.56, 12 pcs",
    ),
    row(
        "M5026",
        "SuperMagnetMan",
        "arc",
        [15.0, 6.4, 1.67],
        "N50",
        1.42,
        80.0,
        "26.60 OD x 23.26 ID x 15, 12 pcs",
    ),
];
```

with:

```rust
/// The library, in the Python row order (`_ROWS`).
pub const MAGNET_LIBRARY: [MagnetSpec; 15] = [
    row(
        "B842SH",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1/2 x 1/4 x 1/8 in, magnetized through 1/8 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B842SH",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B842",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N42",
        1.30,
        80.0,
        "",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B842",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B842-N52",
        "K&J",
        "block",
        [12.7, 6.35, 3.17],
        "N52",
        1.45,
        80.0,
        "",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B842-N52",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B822",
        "K&J",
        "block",
        [12.7, 3.17, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/8 x 1/8 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B822",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B862",
        "K&J",
        "block",
        [12.7, 9.5, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 3/8 x 1/8 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B862",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B882",
        "K&J",
        "block",
        [12.7, 12.7, 3.17],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/2 x 1/8 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B882",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B882-N52",
        "K&J",
        "block",
        [12.7, 12.7, 3.17],
        "N52",
        1.45,
        80.0,
        "",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B882-N52",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B861",
        "K&J",
        "block",
        [12.7, 9.5, 1.59],
        "N42",
        1.30,
        80.0,
        "1/2 x 3/8 x 1/16 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B861",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B881",
        "K&J",
        "block",
        [12.7, 12.7, 1.59],
        "N42",
        1.30,
        80.0,
        "1/2 x 1/2 x 1/16 in (check stock)",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B881",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "B442",
        "K&J",
        "block",
        [6.35, 6.35, 3.17],
        "N42",
        1.30,
        80.0,
        "1/4 x 1/4 x 1/8 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=B442",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "BX042SH",
        "K&J",
        "block",
        [25.4, 6.35, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1 x 1/4 x 1/8 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=BX042SH",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "BX082SH",
        "K&J",
        "block",
        [25.4, 12.7, 3.17],
        "N42SH",
        1.29,
        150.0,
        "1 x 1/2 x 1/8 in",
    )
    .listed(
        "https://www.kjmagnetics.com/proddetail.asp?prod=BX082SH",
        KJ_COATING,
        KJ_MAGNETIZATION,
    ),
    row(
        "M5044",
        "SuperMagnetMan",
        "arc",
        [13.0, 5.5, 1.31],
        "N50",
        1.42,
        80.0,
        "22.60 OD x 19.97 ID x 13, 29.8 deg, 12 pcs; width = mean arc length",
    )
    .listed(
        "https://supermagnetman.com/products/m5044",
        SMM_COATING,
        SMM_MAGNETIZATION,
    ),
    row(
        "M5045",
        "SuperMagnetMan",
        "arc",
        [6.56, 5.6, 1.12],
        "N50M",
        1.42,
        100.0,
        "22.70 OD x 20.47 ID x 6.56, 12 pcs",
    )
    .listed(
        "https://supermagnetman.com/products/m5045",
        SMM_COATING,
        SMM_MAGNETIZATION,
    ),
    row(
        "M5026",
        "SuperMagnetMan",
        "arc",
        [15.0, 6.4, 1.67],
        "N50",
        1.42,
        80.0,
        "26.60 OD x 23.26 ID x 15, 12 pcs",
    )
    .listed(
        "https://supermagnetman.com/products/m5026",
        SMM_COATING,
        SMM_MAGNETIZATION,
    ),
];
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`: unit tests 93 passed (the two in `grades.rs`), `grades.rs` 8 passed, `static_data.rs` 8 passed (the workbook rows are unchanged).

- [ ] **Step 5: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows and the exact-text lookup; `br_T` applies E3 |
```

with:

```markdown
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `tests/static_data.rs` |
```

with:

```markdown
| `tests/grades.rs` | The grade table equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (engine literals bit for bit; the N42SH engine beta is the workbook's); every library part resolves to a grade; a part's workbook Br and Tmax equal its grade's except the registered differences; sintered NdFeB alpha and density equal the engine constants; only ferrite has a positive beta; every part cites its vendor page for coating and magnetization. |
| `tests/static_data.rs` |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    library: library.rs (Magnet library rows, workbook values, and the exact-text lookup; br_T applies E3)
```

with:

```yaml
    library: library.rs (Magnet library rows = the A6 part table: workbook values, grade, vendor page, coating, magnetization; the exact-text lookup; br_T applies E3 with grades::N42SH.br_T)
    grades: grades.rs (Addendum A6 grade table GRADES, 17 grades cited per value from docs/analyses/2026-09-30-magcoupling-addendum-a-data.json; grade(id); N42SH keeps the workbook Hcj 1592 and beta -0.005)
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    ported: [meta.rs, compat.rs, constants.rs, calibration.rs, library.rs, model.rs
```

with:

```yaml
    ported: [meta.rs, compat.rs, constants.rs, calibration.rs, library.rs, grades.rs, model.rs
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
  tests: tests/ (the seven integration tests parity.rs, differential.rs, python_schema.rs, schema.rs, deviations.rs, static_data.rs, robustness.rs;
```

with:

```yaml
  tests: tests/ (the integration tests parity.rs, differential.rs, python_schema.rs, schema.rs, deviations.rs, static_data.rs, robustness.rs, grades.rs;
```

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task3.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 7: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/grades.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/src/engine/library.rs magcoupling-rs/tests/grades.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
feat(magcoupling-rs): A6 grade table and part vendor data

Seventeen grades from the approved Addendum A data file, every value cited
(decisions 1, 3, 4, 17, 18); the parts gain vendor page, coating and
magnetization (vendor pages, 2026-09-30); E3's corrected Br is the N42SH
grade's. No result changes.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 4: Any grade with manual dimensions (Rust-only `grade_inner`, `grade_outer`)

**Model:** `sonnet` (the plan gives the exact code). Escalate as in Task 1.

Spec A6: "Custom dimensions stay available: pick any grade with manual dimensions." Report 6.3: this must be a NEW mode, because
the existing manual mode (no grade, manual Br, uncalibrated demag) is exercised by the differential data (`''`, `b842sh`) and
must stay as it is. Default `""` keeps every existing case bit for bit; the generator never passes these inputs (Task 1).
The ferrite cold case (Task 10) runs through this mode: no library part is ferrite (report section 3).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs` (two Rust-only inputs, `ResolvedMagnet::grade`, `resolve_magnets`, two Rust-only results, unit tests)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/data/input_schema.json` (the two inputs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Layout row of model.rs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (the model line)

**Interfaces:**
- Consumes: `grades::{grade, Grade}` (Task 3), `param_rust_only` (Task 1), `meta::out_rust_only`.
- Produces:
  - inputs `coupling.magnets.grade_inner: String = ""` and `grade_outer: String = ""` (Rust-only; exact grade id, e.g. `"Y30"`);
  - `ResolvedMagnet { length_mm, width_mm, thickness_mm, br_T, tmax_C: NumOrText, grade: Option<&'static Grade> }`;
  - `resolve_magnets`: a library part as before (its grade from the row); else, when the grade text names a grade, the manual dimensions with the grade's `br_T` and `tmax_C`; else the manual magnet as before (`"n/a"`, no grade);
  - Rust-only results `model.inner_grade: String`, `model.outer_grade: String` (grade id or `""`).

- [ ] **Step 1: Write the failing unit tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    #[test]
    fn part_lookup_is_exact_text() {
```

with:

```rust
    #[test]
    fn a_grade_gives_manual_dimensions_its_br_and_rating() {
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.grade_inner = "Y30".to_owned();
        ci.magnets.manual_inner_br_T = 1.2; // ignored: the grade supplies Br
        let r = at(&ci);
        assert_eq!(r.inner_br_T, 0.37);
        assert_eq!(r.inner_tmax_C, NumOrText::Num(250.0));
        assert_eq!(r.inner_temp_check, "OK");
        assert_eq!(r.inner_grade, "Y30");
        assert_eq!(
            (r.inner_length_mm, r.inner_thickness_mm),
            (ci.magnets.manual_inner_length_mm, ci.magnets.manual_inner_thickness_mm)
        );
        // The outer ring keeps its library part.
        assert_eq!((r.outer_br_T, r.outer_grade.as_str()), (1.29, "N42SH"));
    }

    #[test]
    fn a_library_part_wins_over_a_grade_and_an_unknown_grade_is_manual() {
        let mut ci = CouplingInputs::default();
        ci.magnets.grade_inner = "Y30".to_owned(); // the part B842SH is in the library
        let r = at(&ci);
        assert_eq!(r, at(&CouplingInputs::default()));
        assert_eq!(r.inner_grade, "N42SH");
        for text in ["", "y30", "Y30 ", "N42"] {
            let mut ci = CouplingInputs::default();
            ci.magnets.part_inner = String::new();
            ci.magnets.grade_inner = text.to_owned();
            let r = at(&ci);
            if text == "N42" {
                assert_eq!((r.inner_br_T, r.inner_grade.as_str()), (1.30, "N42"));
                continue;
            }
            assert_eq!(r.inner_br_T, ci.magnets.manual_inner_br_T, "{text:?}");
            assert_eq!(r.inner_tmax_C, NumOrText::Text(NOT_IN_LIBRARY), "{text:?}");
            assert_eq!(r.inner_grade, "", "{text:?}");
        }
    }

    #[test]
    fn part_lookup_is_exact_text() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --lib a_grade_gives 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors, among them ``no field `grade_inner` on type `model::MagnetInputs` ``.

- [ ] **Step 3: Add the inputs, the grade in `resolve_magnets` and the two results**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
use super::deviations::{DeviationId, Deviations};
use super::library::{self, lookup};
use super::meta::{NumOrText, inputs, out, param, results};
```

with:

```rust
use super::deviations::{DeviationId, Deviations};
use super::grades::{self, Grade};
use super::library::{self, lookup};
use super::meta::{NumOrText, inputs, out, out_rust_only, param, param_rust_only, results};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            manual_outer_br_T: f64 = 1.30 => param("T", "Manual outer Br at 20 °C",
                "", "Calculator!C27")
                .range(0.2, 1.5, 0.001),
        }
    }
}
```

with:

```rust
            manual_outer_br_T: f64 = 1.30 => param("T", "Manual outer Br at 20 °C",
                "", "Calculator!C27")
                .range(0.2, 1.5, 0.001),
            grade_inner: String = "" => param_rust_only("-", "Inner magnet grade (manual dimensions)",
                "Addendum A6: a grade of the grade table, by exact name (e.g. N42SH, Y30). Used only when the inner part is not in the library: the manual dimensions with the grade's Br at 20 °C and maximum temperature. Blank = the manual Br and no rating."),
            grade_outer: String = "" => param_rust_only("-", "Outer magnet grade (manual dimensions)",
                "As the inner grade, for the outer ring."),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            outer_temp_check: String => out("", "Outer magnet temperature check",
                "Library rating only.", "Calculator!C108"),
        }
    }
}
```

with:

```rust
            outer_temp_check: String => out("", "Outer magnet temperature check",
                "Library rating only.", "Calculator!C108"),
            inner_grade: String => out_rust_only("", "Inner magnet grade used",
                "The library part's grade, or the grade picked for manual dimensions; blank for a manual magnet without a grade."),
            outer_grade: String => out_rust_only("", "Outer magnet grade used",
                "As the inner grade, for the outer ring."),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// One magnet ring as the Calculator uses it (Python `ResolvedMagnet`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct ResolvedMagnet {
    pub length_mm: f64,
    pub width_mm: f64,
    pub thickness_mm: f64,
    pub br_T: f64,
    /// Library rating, or `NOT_IN_LIBRARY` for manual magnets.
    pub tmax_C: NumOrText,
}

/// Library values when the part is found, else the manual values (workbook IFERROR/INDEX/MATCH).
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence).
#[allow(non_snake_case)]
pub fn resolve_magnets(m: &MagnetInputs, dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet) {
    let resolve = |part: &str, length_mm: f64, width_mm: f64, thickness_mm: f64, br_T: f64| {
        match lookup(part) {
            Some(spec) => ResolvedMagnet {
                length_mm: spec.length_mm,
                width_mm: spec.width_mm,
                thickness_mm: spec.thickness_mm,
                br_T: library::br_T(spec, dev),
                tmax_C: NumOrText::Num(spec.tmax_C),
            },
            None => ResolvedMagnet {
                length_mm,
                width_mm,
                thickness_mm,
                br_T,
                tmax_C: NumOrText::Text(NOT_IN_LIBRARY),
            },
        }
    };
    (
        resolve(
            &m.part_inner,
            m.manual_inner_length_mm,
            m.manual_inner_width_mm,
            m.manual_inner_thickness_mm,
            m.manual_inner_br_T,
        ),
        resolve(
            &m.part_outer,
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
/// One magnet ring as the Calculator uses it (Python `ResolvedMagnet`, plus the grade).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct ResolvedMagnet {
    pub length_mm: f64,
    pub width_mm: f64,
    pub thickness_mm: f64,
    pub br_T: f64,
    /// Library or grade rating, or `NOT_IN_LIBRARY` for manual magnets without a grade.
    pub tmax_C: NumOrText,
    /// The part's grade, or the grade picked for manual dimensions (Addendum A6).
    pub grade: Option<&'static Grade>,
}

/// Library values when the part is found (workbook IFERROR/INDEX/MATCH); else the
/// manual dimensions, with the grade's Br and rating when a grade is picked
/// (Addendum A6, a Rust-only mode) and the manual Br and no rating otherwise.
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence).
#[allow(non_snake_case)]
pub fn resolve_magnets(m: &MagnetInputs, dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet) {
    let resolve = |part: &str,
                   grade: &str,
                   length_mm: f64,
                   width_mm: f64,
                   thickness_mm: f64,
                   br_T: f64| match lookup(part) {
        Some(spec) => ResolvedMagnet {
            length_mm: spec.length_mm,
            width_mm: spec.width_mm,
            thickness_mm: spec.thickness_mm,
            br_T: library::br_T(spec, dev),
            tmax_C: NumOrText::Num(spec.tmax_C),
            grade: grades::grade(spec.grade),
        },
        None => match grades::grade(grade) {
            Some(g) => ResolvedMagnet {
                length_mm,
                width_mm,
                thickness_mm,
                br_T: g.br_T,
                tmax_C: NumOrText::Num(g.tmax_C),
                grade: Some(g),
            },
            None => ResolvedMagnet {
                length_mm,
                width_mm,
                thickness_mm,
                br_T,
                tmax_C: NumOrText::Text(NOT_IN_LIBRARY),
                grade: None,
            },
        },
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

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        inner_temp_check: temp_check(mi.tmax_C),
        outer_temp_check: temp_check(mo.tmax_C),
    }
}
```

with:

```rust
        inner_temp_check: temp_check(mi.tmax_C),
        outer_temp_check: temp_check(mo.tmax_C),
        inner_grade: mi.grade.map_or("", |g| g.id).to_owned(),
        outer_grade: mo.grade.map_or("", |g| g.id).to_owned(),
    }
}
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

- [ ] **Step 4: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test schema
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: the bless run adds the two rows (`"rust_only": true`, `"cell": null`, type `text`); every binary `ok`, unit tests 95 passed.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|panicked"
```

Expected: no output.

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python). This is the first proof that the Task 1 filter works: without it the generator would pass `coupling.magnets.grade_inner` to Python and fail.

- [ ] **Step 5: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs, `ModelResults` (73 cells), the fixed harmonic set `HARMONICS` (1, 3, 5), and the mass estimate `MassResults` (6 cells) |
```

with:

```markdown
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6), `ModelResults` (73 cells, plus the Rust-only `inner_grade` and `outer_grade`), the fixed harmonic set `HARMONICS` (1, 3, 5), and the mass estimate `MassResults` (6 cells) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    model: "model.rs (Calculator sheet: inputs, ModelResults 73 result cells,
```

with:

```yaml
    model: "model.rs (Calculator sheet: inputs (Rust-only grade_inner/grade_outer: a grade for manual dimensions, Addendum A6), ModelResults 73 result cells (Rust-only inner_grade/outer_grade),
```

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task4.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task4.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 7: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/model.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
feat(magcoupling-rs): any grade with manual dimensions

Rust-only inputs coupling.magnets.grade_inner/grade_outer: a manual magnet
with a grade takes the grade's Br and rating (spec A6). A library part wins;
an unknown grade is the existing manual mode. Defaults unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 5: E15: heat capacity at each part's own specific heat

**Model:** session model (omit `model`; CLAUDE.md section 5: a `risk: physics` correction). Physics reviewer: the Addendum A report row E15 and section 5.4 (E15 table).

Decision 8 A. C141 = [m_mag c_NdFeB + (cup + boss) c_cup + hub c_hub + hardware c_steel + (retainers + endplates) c_316 + cap c_Al] / 1000,
c_cup = C140 when the cup is aluminium (E9's gate), c_hub = C140 when C6 != 1, both from the same helpers the masses read.
With every part steel the workbook expression stays, so the default cells stay bit for bit. Expected values: the report's
section 5.4 E15 table, all three columns (workbook + E9, M2, standalone), 23 cells each.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs` (`cup_is_aluminium`, `hub_is_aluminium`: the one gate for mass and heat)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs` (links `cup_aluminium`, `hub_aluminium`; C141 under E15)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs` (fill the two links)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs` (E15 Applied, probe on E9)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs` (report tables, `m2()`, helpers)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Differences table)

**Interfaces:**
- Consumes: `Deviations::with/without`, `probe_base` (Task 2); `model::mass_estimate`; `TemperatureLinks`.
- Produces:
  - `pub(crate) fn model::cup_is_aluminium(backiron: i64, dev: Deviations) -> bool` (E9 on and C6 != 1) and `pub(crate) fn model::hub_is_aluminium(backiron: i64) -> bool` (C6 != 1); `mass_estimate` reads them;
  - `TemperatureLinks { .., cup_aluminium: bool, hub_aluminium: bool }`;
  - test helpers in `tests/deviations.rs`: `assert_sig4(what, got, want)` (the report's 4 significant figures), `cells_with(overrides, dev)`, `type ReportRows`, `assert_report_table(label, overrides, before, after, rows)` (exactly the listed cells change, each to the report's figure), `const NO_BACK_IRON`, `fn m2() -> Deviations` (E1-E14 on: the report's M2 basis).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e14_the_22mm_boss_statement_is_true() {
```

with:

```rust
/// `got` is the Addendum A report's figure `want`, printed to 4 significant
/// figures (report section 5.1): within half a unit of the 4th figure.
fn assert_sig4(what: &str, got: &Value, want: f64) {
    let got = num(got);
    let half = 0.5 * 10f64.powi(want.abs().log10().floor() as i32 - 3);
    assert!(
        (got - want).abs() <= half * (1.0 + 1e-9),
        "{what}: {got} is not the report's {want} (4 s.f.)"
    );
}

/// Every celled value for the defaults with `overrides` applied, under `dev`.
fn cells_with(overrides: &[(&str, Value)], dev: Deviations) -> BTreeMap<String, Value> {
    let mut inputs = DesignInputs::defaults_with(dev);
    for (path, value) in overrides {
        inputs.set(path, value.clone()).expect("a valid input");
    }
    cell_values_for(&inputs, dev)
}

/// A report table of changed cells: (cell, before, after) at 4 significant figures.
type ReportRows<'a> = &'a [(&'a str, f64, f64)];

/// `before` -> `after` changes exactly the report's cells, to the report's figures.
fn assert_report_table(
    label: &str,
    overrides: &[(&str, Value)],
    before: Deviations,
    after: Deviations,
    rows: ReportRows,
) {
    let (b, a) = (cells_with(overrides, before), cells_with(overrides, after));
    let want: BTreeSet<String> = rows.iter().map(|(c, _, _)| (*c).to_owned()).collect();
    assert_eq!(changed_cells(&b, &a), want, "{label}: changed cells");
    for &(cell, was, now) in rows {
        assert_sig4(&format!("{label} {cell} before"), &b[cell], was);
        assert_sig4(&format!("{label} {cell} after"), &a[cell], now);
    }
}

const NO_BACK_IRON: [(&str, Value); 1] = [("coupling.backiron", Value::Int(0))];

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
    .into_iter()
    .fold(Deviations::ALL, Deviations::without)
}

#[test]
fn e15_heat_capacity_matches_the_report() {
    // Report 5.4, E15, all three columns: workbook + E9 (the probe basis, decision 15), M2
    // (every other correction on) and standalone (E9 off: only the hub moves).
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let e15 = DeviationId::E15;
    let main: ReportRows = &[
        ("Temperature design!C20", 66.01, 65.92),
        ("Temperature design!C141", 45.78, 63.02),
        ("Temperature design!C143", 152.6, 210.1),
        ("Temperature design!C145", 0.005411, 0.003931),
        ("Temperature design!C154", 0.05411, 0.03931),
        ("Temperature design!C155", 0.1623, 0.1179),
        ("Temperature design!C156", 616.1, 848.0),
        ("Temperature design!C157", 205.4, 282.7),
        ("Temperature design!C158", 5087.0, 7002.0),
        ("Temperature design!C159", 457.8, 630.2),
        ("Temperature design!C160", 1.526e4, 2.101e4),
        ("Temperature design!C161", 65.32, 65.23),
        ("Temperature design!C171", 0.005411, 0.003931),
        ("Temperature design!C172", 0.01623, 0.01179),
        ("Temperature design!C180", 66.01, 65.92),
        ("Temperature design!C181", 36.54, 36.63),
        ("Temperature design!C182", 26.54, 26.63),
        ("Temperature design!C186", 1.63, 1.631),
        ("Temperature design!C189", 66.01, 65.92),
        ("Temperature design!C190", 53.99, 54.08),
        ("Temperature design!C192", 1.875, 1.875),
        ("Temperature design!C193", 0.1981, 0.1982),
        ("Temperature design!C196", 7.571, 7.569),
    ];
    assert_report_table("E15 on E9", &NO_BACK_IRON, e9, e9.with(e15), main);
    let m2_rows: ReportRows = &[
        ("Temperature design!C20", 66.12, 66.02),
        ("Temperature design!C141", 45.78, 63.02),
        ("Temperature design!C143", 152.6, 210.1),
        ("Temperature design!C145", 0.00601, 0.004366),
        ("Temperature design!C154", 0.0601, 0.04366),
        ("Temperature design!C155", 0.1803, 0.131),
        ("Temperature design!C156", 554.7, 763.5),
        ("Temperature design!C157", 184.9, 254.5),
        ("Temperature design!C158", 5087.0, 7002.0),
        ("Temperature design!C159", 457.8, 630.2),
        ("Temperature design!C160", 1.526e4, 2.101e4),
        ("Temperature design!C161", 65.36, 65.26),
        ("Temperature design!C171", 0.00601, 0.004366),
        ("Temperature design!C172", 0.01803, 0.0131),
        ("Temperature design!C180", 66.12, 66.02),
        ("Temperature design!C181", 36.93, 37.03),
        ("Temperature design!C182", 26.93, 27.03),
        ("Temperature design!C186", 1.63, 1.63),
        ("Temperature design!C189", 66.12, 66.02),
        ("Temperature design!C190", 53.88, 53.98),
        ("Temperature design!C192", 1.874, 1.875),
        ("Temperature design!C193", 0.1981, 0.1981),
        ("Temperature design!C196", 7.573, 7.571),
    ];
    assert_report_table("E15 on M2", &NO_BACK_IRON, m2(), m2().with(e15), m2_rows);
    let standalone: ReportRows = &[
        ("Temperature design!C20", 65.89, 65.88),
        ("Temperature design!C141", 74.19, 77.98),
        ("Temperature design!C143", 247.3, 259.9),
        ("Temperature design!C145", 0.003339, 0.003177),
        ("Temperature design!C154", 0.03339, 0.03177),
        ("Temperature design!C155", 0.1002, 0.0953),
        ("Temperature design!C156", 998.3, 1049.0),
        ("Temperature design!C157", 332.8, 349.8),
        ("Temperature design!C158", 8243.0, 8664.0),
        ("Temperature design!C159", 741.9, 779.8),
        ("Temperature design!C160", 2.473e4, 2.599e4),
        ("Temperature design!C161", 65.2, 65.19),
        ("Temperature design!C171", 0.003339, 0.003177),
        ("Temperature design!C172", 0.01002, 0.00953),
        ("Temperature design!C180", 65.89, 65.88),
        ("Temperature design!C181", 36.66, 36.67),
        ("Temperature design!C182", 26.66, 26.67),
        ("Temperature design!C186", 1.631, 1.631),
        ("Temperature design!C189", 65.89, 65.88),
        ("Temperature design!C190", 54.11, 54.12),
        ("Temperature design!C192", 1.876, 1.876),
        ("Temperature design!C193", 0.1982, 0.1982),
        ("Temperature design!C196", 7.569, 7.569),
    ];
    assert_report_table(
        "E15 alone",
        &NO_BACK_IRON,
        Deviations::NONE,
        Deviations::only(e15),
        standalone,
    );
    // C19 and C150 stay "never" in every column; C137 (the steel specific heat shown) does not move.
    for dev in [e9.with(e15), m2().with(e15), Deviations::only(e15)] {
        let c = cells_with(&NO_BACK_IRON, dev);
        assert_eq!(
            c["Temperature design!C19"],
            Value::Text("never: steady state stays below the limit".into())
        );
        assert_eq!(c["Temperature design!C137"], Value::Num(473.0));
    }
}

#[test]
fn e15_leaves_every_default_cell_bit_for_bit() {
    // At defaults (C6 = 1) every part is steel and the workbook expression stays.
    assert_bit_for_bit_at_defaults(DeviationId::E15);
}

#[test]
fn e14_the_22mm_boss_statement_is_true() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations e15 2>&1 | grep -E "panicked|test result" -A2 | head -8
```

Expected: `e15_heat_capacity_matches_the_report` FAILS with `E15 on E9: changed cells` (left `{}`: nothing changes yet, right the 23 report cells); `e15_leaves_every_default_cell_bit_for_bit` passes.

- [ ] **Step 3: Implement E15**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// Calculator rows 110-115. Gross solids: no holes, slots or threads subtracted.
```

with:

```rust
/// Whether the cup wall, rear web and boss are aluminium: E9, with no
/// intentional back iron (`backiron` is Calculator!C6). The one gate the mass
/// model (C111, C113), the heat capacity (E15) and the removed-disc mass (E16)
/// read, so they cannot disagree.
pub(crate) fn cup_is_aluminium(backiron: i64, dev: Deviations) -> bool {
    dev.is_on(DeviationId::E9) && backiron != 1
}

/// Whether the keyed hub is aluminium: the workbook prices it so whenever C6 is
/// not 1 (C112), with or without E9. Read by the mass model, E15 and E18.
pub(crate) fn hub_is_aluminium(backiron: i64) -> bool {
    backiron != 1
}

/// Calculator rows 110-115. Gross solids: no holes, slots or threads subtracted.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density = if dev.is_on(DeviationId::E9) && ci.backiron != 1 {
        al_density_g_mm3
    } else {
        steel_density_g_mm3
    };
```

with:

```rust
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density = if cup_is_aluminium(ci.backiron, dev) {
        al_density_g_mm3
    } else {
        steel_density_g_mm3
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    let m_hub = (hub_area - PI * (ci.bore_mm / 2.0).powi(2))
        * hub_length_mm
        * (if ci.backiron == 1 {
            steel_density_g_mm3
        } else {
            al_density_g_mm3
        });
```

with:

```rust
    let m_hub = (hub_area - PI * (ci.bore_mm / 2.0).powi(2))
        * hub_length_mm
        * (if hub_is_aluminium(ci.backiron) {
            al_density_g_mm3
        } else {
            steel_density_g_mm3
        });
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    pub steel_E_GPa: f64,              // Materials C18
    pub al6061_sigma_S_m: f64,         // Materials C43
}
```

with:

```rust
    pub steel_E_GPa: f64,              // Materials C18
    pub al6061_sigma_S_m: f64,         // Materials C43
    /// E9: the cup and boss are aluminium (`model::cup_is_aluminium`).
    pub cup_aluminium: bool,
    /// The hub is aluminium, C6 != 1 (`model::hub_is_aluminium`).
    pub hub_aluminium: bool,
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    // ---- thermal network
    let th = &ti.thermal;
    let C = (k.mass_magnets_g * th.c_ndfeb
        + (k.mass_cup_g + k.mass_hub_g + k.mass_boss_g + k.hardware_g) * k.steel_c
        + (k.retainers_g + k.endplates_g) * th.c_316
        + k.cap_g * th.c_aluminium)
        / 1000.0;
```

with:

```rust
    // ---- thermal network
    let th = &ti.thermal;
    // E15: an aluminium cup, boss or hub at aluminium's specific heat, on the gates the
    // masses read; the keys and screws (hardware) stay steel. With every part steel the
    // workbook expression stays, bit for bit.
    let C = if dev.is_on(DeviationId::E15) && k.hub_aluminium {
        let c_cup = if k.cup_aluminium {
            th.c_aluminium
        } else {
            k.steel_c
        };
        (k.mass_magnets_g * th.c_ndfeb
            + (k.mass_cup_g + k.mass_boss_g) * c_cup
            + k.mass_hub_g * th.c_aluminium
            + k.hardware_g * k.steel_c
            + (k.retainers_g + k.endplates_g) * th.c_316
            + k.cap_g * th.c_aluminium)
            / 1000.0
    } else {
        (k.mass_magnets_g * th.c_ndfeb
            + (k.mass_cup_g + k.mass_hub_g + k.mass_boss_g + k.hardware_g) * k.steel_c
            + (k.retainers_g + k.endplates_g) * th.c_316
            + k.cap_g * th.c_aluminium)
            / 1000.0
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            steel_E_GPa: 205.0,
            al6061_sigma_S_m: 2.5e7,
        }
    }
```

with:

```rust
            steel_E_GPa: 205.0,
            al6061_sigma_S_m: 2.5e7,
            cup_aluminium: false,
            hub_aluminium: false,
        }
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        steel_E_GPa: mat_in.steel.modulus_GPa,
        al6061_sigma_S_m: materials::AL6061.conductivity_S_m,
    };
```

with:

```rust
        steel_E_GPa: mat_in.steel.modulus_GPa,
        al6061_sigma_S_m: materials::AL6061.conductivity_S_m,
        cup_aluminium: model::cup_is_aluminium(ci.backiron, dev),
        hub_aluminium: model::hub_is_aluminium(ci.backiron),
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Planned,
        cells: &[
            "Calculator!C6",
            "Calculator!C111",
            "Calculator!C112",
            "Calculator!C113",
            "Temperature design!C140",
```

with:

```rust
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Applied,
        cells: &[
            "Calculator!C6",
            "Calculator!C111",
            "Calculator!C112",
            "Calculator!C113",
            "Temperature design!C140",
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            c_hub = C140 when C6 != 1, otherwise Materials!C16. The gate reads the same flag the density reads.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
```

with:

```rust
            c_hub = C140 when C6 != 1, otherwise Materials!C16. The gate reads the same flag the density reads.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "no intentional back iron (coupling.backiron = 0), on top of E9 (report 5.4, main column)",
            inputs: &[("coupling.backiron", Literal::Int(0))],
            expect: &[
                CellChange {
                    cell: "Temperature design!C141",
                    workbook: Literal::Num(45.78201892981087),
                    corrected: Literal::Num(63.01541872000666),
                },
                CellChange {
                    cell: "Temperature design!C143",
                    workbook: Literal::Num(152.60672976603624),
                    corrected: Literal::Num(210.05139573335552),
                },
                CellChange {
                    cell: "Temperature design!C145",
                    workbook: Literal::Num(0.00541065752941102),
                    corrected: Literal::Num(0.003930955795673762),
                },
                CellChange {
                    cell: "Temperature design!C20",
                    workbook: Literal::Num(66.01060704628003),
                    corrected: Literal::Num(65.92282367380993),
                },
            ],
        }],
    },
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! 0 rev and 0 N·m at C19, C23, C150 to C153), and E13 (a heating power of
//! exactly 0, from a measured drag of 0, gives +inf rotations per °C at C156
//! and C157 instead of Python's ZeroDivisionError).
```

with:

```rust
//! 0 rev and 0 N·m at C19, C23, C150 to C153), E13 (a heating power of
//! exactly 0, from a measured drag of 0, gives +inf rotations per °C at C156
//! and C157 instead of Python's ZeroDivisionError), and E15 (C141 prices an
//! aluminium cup, boss and hub at C140, on the gates the masses read).
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, `deviations.rs` 39 passed; parity and differential unchanged (E15 is off in NONE).

- [ ] **Step 4: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| E14 | Shaft clamps!C35 help; README | "At 22 mm only M3 fits; two of them need a 14.5 mm clamp." | "At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3." | E14 (documentation) |
```

with:

```markdown
| E14 | Shaft clamps!C35 help; README | "At 22 mm only M3 fits; two of them need a 14.5 mm clamp." | "At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3." | E14 (documentation) |
| E15 | Temperature design!C141 (feeds C143, C145, C154-C161, C171, C172, C180-C182, C186, C189, C190, C192, C193, C196, C20) | cup, boss and hub at 4140's specific heat even when the mass model makes them aluminium | aluminium parts at C140 (900 J/(kg·K)): the cup and boss on E9's gate, the hub when C6 is not 1; hardware stays steel. Back iron 0 on E9: C141 45.78 → 63.02 J/K, no verdict changes | Addendum A row E15 (decisions 8, 15) |
```

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task5.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task5.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task5.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/deviations.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
fix(magcoupling-rs): apply E15 heat capacity at each part's own specific heat

With no back iron the aluminium cup, boss and hub are priced at C140, on the
gates the masses read (decision 8). Reproduces the report's E15 table in all
three columns; the default cells stay bit for bit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 6: E16: the removed web disc at the web's density; reworded result help (decision 14)

**Model:** session model (omit `model`; a `risk: physics` correction). Physics reviewer: report row E16, section 5.4 (E16 table), decision 14.

Decision 9 A: C189 = pi/4 (C185^2 - C39^2) C125 rho_cup, rho_cup passed from the mass model (the same helper C111 and C113
use). Standalone (E9 off) the web is steel and nothing moves. Decision 14 A: the labels C147 "Selected one-piece steel-cup mass"
and C189 "Steel removed for optional larger pilot bore" stay (schema parity with Python); the help is reworded and the workbook's
(empty) help is recorded in `workbook_help`, which the metadata tests now accept for results.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs` (`cup_boss_density`, the one density of the web)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs` (C189 under E16; C147 and C189 help)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs` (pass the web density)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs` (E16 Applied, probe on E9, `workbook_help` for two results)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs` (E16 tables; reworded help of a result)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs` (a result's reworded help compares with the recorded workbook text)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Differences table, Deviations section)

**Interfaces:**
- Consumes: `model::cup_is_aluminium` (Task 5), test helpers `assert_report_table`, `cells_with`, `m2` (Task 5).
- Produces:
  - `pub(crate) fn model::cup_boss_density(backiron: i64, steel_density_g_mm3: f64, al_density_g_mm3: f64, dev: Deviations) -> f64`;
  - `metal_design::compute(.., proto_test_temp_C: f64, cup_density_g_mm3: f64, dev: Deviations)` (one more parameter before `dev`);
  - `workbook_help` may name a scalar result path; `tests/python_schema.rs` and `reworded_help_is_recorded_for_real_fields` accept it.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs`, replace:

```rust
#[test]
fn ported_results_carry_the_python_metadata() {
    let python = python_rows();
    let mut failures = Vec::new();
```

with:

```rust
#[test]
fn ported_results_carry_the_python_metadata() {
    let python = python_rows();
    let reworded = reworded_help();
    let mut failures = Vec::new();
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/python_schema.rs`, replace:

```rust
        let m = row.meta;
        if py.kind != "result" {
            failures.push(format!("{}: Python kind is {:?}", row.path, py.kind));
        }
        compare(
            &mut failures,
            &row.path,
            (m.label, m.unit, m.help, m.cell),
            py,
        );
```

with:

```rust
        let m = row.meta;
        if py.kind != "result" {
            failures.push(format!("{}: Python kind is {:?}", row.path, py.kind));
        }
        // Python keeps the workbook help where a correction rewords it (decision 14).
        let help = reworded.get(row.path.as_str()).copied().unwrap_or(m.help);
        compare(
            &mut failures,
            &row.path,
            (m.label, m.unit, help, m.cell),
            py,
        );
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn reworded_help_is_recorded_for_real_fields() {
    let inputs = input_rows(&DesignInputs::default());
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
    {
        for &(path, workbook) in d.workbook_help {
            if path.contains("[*]") {
                continue; // table columns: checked against the workbook headers in tests/schema.rs
            }
            let row = inputs
                .iter()
                .find(|r| r.path == path)
                .unwrap_or_else(|| panic!("{}: {path} is not an input", d.id));
            assert_ne!(
                row.meta.help, workbook,
                "{}: {path} help is not reworded",
                d.id
            );
        }
    }
}
```

with:

```rust
#[test]
fn reworded_help_is_recorded_for_real_fields() {
    // An input path, a scalar result path (decision 14: E16's and E17's labels stay, their
    // help is reworded), or a table column (checked against the workbook in tests/schema.rs).
    let inputs = input_rows(&DesignInputs::default());
    let results = result_rows(&compute_all(&DesignInputs::default()));
    for d in REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
    {
        for &(path, workbook) in d.workbook_help {
            if path.contains("[*]") {
                continue; // table columns: checked against the workbook headers in tests/schema.rs
            }
            let help = inputs
                .iter()
                .find(|r| r.path == path)
                .map(|r| r.meta.help)
                .or_else(|| {
                    results
                        .iter()
                        .find(|r| r.path == path && !r.meta.rust_only)
                        .map(|r| r.meta.help)
                })
                .unwrap_or_else(|| panic!("{}: {path} is not an input or a result", d.id));
            assert_ne!(help, workbook, "{}: {path} help is not reworded", d.id);
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e15_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
#[test]
fn e16_removed_disc_matches_the_report() {
    // Report 5.4, E16: the same four cells in the main and M2 columns; standalone (E9 off:
    // the web is steel) nothing moves. C147, C188, C47, C114 and C192 do not change.
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let e16 = DeviationId::E16;
    let rows: ReportRows = &[
        ("Metal design!C148", 101.5, 103.8),
        ("Metal design!C149", -4.689, -6.954),
        ("Metal design!C189", 3.453, 1.188),
        ("Metal design!C191", 101.5, 103.8),
    ];
    assert_report_table("E16 on E9", &NO_BACK_IRON, e9, e9.with(e16), rows);
    assert_report_table("E16 on M2", &NO_BACK_IRON, m2(), m2().with(e16), rows);
    assert_report_table(
        "E16 alone",
        &NO_BACK_IRON,
        Deviations::NONE,
        Deviations::only(e16),
        &[],
    );
    // Full precision (report 5.4): the disc at 2.7 g/cm³.
    let c = cells_with(&NO_BACK_IRON, e9.with(e16));
    assert_eq!(c["Metal design!C189"], Value::Num(1.1875220230569419));
    assert_eq!(c["Metal design!C191"], Value::Num(103.76539290918065));
    assert_eq!(c["Metal design!C149"], Value::Num(-6.95366329618038));
}

#[test]
fn e16_leaves_every_default_cell_bit_for_bit() {
    // At defaults the web is steel: the density E16 reads is C132 itself.
    assert_bit_for_bit_at_defaults(DeviationId::E16);
}

#[test]
fn e15_leaves_every_default_cell_bit_for_bit() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations e16 2>&1 | grep -E "panicked|test result" -A2 | head -8
```

Expected: `e16_removed_disc_matches_the_report` FAILS with `E16 on E9: changed cells` (left `{}`).

- [ ] **Step 3: Implement E16 and reword the two help texts**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// Whether the keyed hub is aluminium: the workbook prices it so whenever C6 is
```

with:

```rust
/// The density of the cup wall, rear web and boss [g/mm³]: aluminium (C42) when
/// [`cup_is_aluminium`], else steel (C132). The mass model's C111 and C113 use it,
/// and E16 prices the disc bored out of the web with it (one source of truth).
pub(crate) fn cup_boss_density(
    backiron: i64,
    steel_density_g_mm3: f64,
    al_density_g_mm3: f64,
    dev: Deviations,
) -> f64 {
    if cup_is_aluminium(backiron, dev) {
        al_density_g_mm3
    } else {
        steel_density_g_mm3
    }
}

/// Whether the keyed hub is aluminium: the workbook prices it so whenever C6 is
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density = if cup_is_aluminium(ci.backiron, dev) {
        al_density_g_mm3
    } else {
        steel_density_g_mm3
    };
```

with:

```rust
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density =
        cup_boss_density(ci.backiron, steel_density_g_mm3, al_density_g_mm3, dev);
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied E8 (Metal design!C175).
```

with:

```rust
//! Deviations touching this sheet (see
//! [`crate::engine::deviations::REGISTRY`]): applied E8 (Metal design!C175)
//! and E16 (C189: the disc bored out of the web for the adapter pilot is priced
//! at the cup's density, aluminium with no back iron under E9).
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
            steel_cup_mass_g: f64 => out("g", "Selected one-piece steel-cup mass", "", "Metal design!C147"),
```

with:

```rust
            steel_cup_mass_g: f64 => out("g", "Selected one-piece steel-cup mass",
                "4140 at back iron = 1; the aluminium cup when there is no back iron (correction E9).", "Metal design!C147"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
            adapter_steel_removed_g: f64 => out("g", "Steel removed for optional larger pilot bore", "", "Metal design!C189"),
```

with:

```rust
            adapter_steel_removed_g: f64 => out("g", "Steel removed for optional larger pilot bore",
                "Priced at the cup's density: 4140 at back iron = 1, 6061 aluminium when there is no back iron (corrections E9 and E16).",
                "Metal design!C189"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
/// The Metal design sheet (Python `metal_design.compute`): torque at the hot and
/// cold limits, the radial clearance stack, duty, the axial stack, the optional
/// aluminium adapter and the hybrid mass. `ret` is [`retainers`]' result.
```

with:

```rust
/// The Metal design sheet (Python `metal_design.compute`): torque at the hot and
/// cold limits, the radial clearance stack, duty, the axial stack, the optional
/// aluminium adapter and the hybrid mass. `ret` is [`retainers`]' result;
/// `cup_density_g_mm3` is the density the mass model gives the web
/// (`model::cup_boss_density`), read only by E16.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
    proto_measured_Nm: f64,
    proto_test_temp_C: f64,
    _dev: Deviations,
) -> MetalDesignResults {
```

with:

```rust
    proto_measured_Nm: f64,
    proto_test_temp_C: f64,
    cup_density_g_mm3: f64,
    dev: Deviations,
) -> MetalDesignResults {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
    let removed = PI / 4.0
        * (md.adapter_pilot_dia_mm.powi(2) - bore_mm.powi(2))
        * md.web_mm
        * md.steel_density_g_mm3;
```

with:

```rust
    // E16: the disc lies in the web, so it is priced at the web's density (E9 makes it
    // aluminium with no back iron); the workbook always uses steel.
    let web_density = if dev.is_on(DeviationId::E16) {
        cup_density_g_mm3
    } else {
        md.steel_density_g_mm3
    };
    let removed = PI / 4.0
        * (md.adapter_pilot_dia_mm.powi(2) - bore_mm.powi(2))
        * md.web_mm
        * web_density;
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/metal_design.rs`, replace:

```rust
                &ret,
                0.9,
                20.0,
                Deviations::NONE,
            )
        };
```

with:

```rust
                &ret,
                0.9,
                20.0,
                0.00785,
                Deviations::NONE,
            )
        };
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        &ret,
        cal_in.measured_torque_Nm,
        cal_in.test_temp_C,
        dev,
    );
```

with:

```rust
        &ret,
        cal_in.measured_torque_Nm,
        cal_in.test_temp_C,
        model::cup_boss_density(
            ci.backiron,
            md.steel_density_g_mm3,
            md.al_density_g_mm3,
            dev,
        ),
        dev,
    );
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            decisions: &[9, 14, 15],
        },
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Planned,
```

with:

```rust
            decisions: &[9, 14, 15],
        },
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Applied,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            as one source of truth. The labels C147 and C189 stay (schema parity); their help is reworded (decision 14).",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
```

with:

```rust
            as one source of truth. The labels C147 and C189 stay (schema parity); their help is reworded (decision 14).",
        workbook_input_defaults: &[],
        workbook_help: &[
            ("metal.steel_cup_mass_g", ""),
            ("metal.adapter_steel_removed_g", ""),
        ],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "no intentional back iron (coupling.backiron = 0), on top of E9 (report 5.4, E16)",
            inputs: &[("coupling.backiron", Literal::Int(0))],
            expect: &[
                CellChange {
                    cell: "Metal design!C189",
                    workbook: Literal::Num(3.4526103262951824),
                    corrected: Literal::Num(1.1875220230569419),
                },
                CellChange {
                    cell: "Metal design!C191",
                    workbook: Literal::Num(101.5003046059424),
                    corrected: Literal::Num(103.76539290918065),
                },
                CellChange {
                    cell: "Metal design!C148",
                    workbook: Literal::Num(101.5003046059424),
                    corrected: Literal::Num(103.76539290918065),
                },
                CellChange {
                    cell: "Metal design!C149",
                    workbook: Literal::Num(-4.6885749929421365),
                    corrected: Literal::Num(-6.95366329618038),
                },
            ],
        }],
    },
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, `deviations.rs` 41 passed; `python_schema.rs` 7 passed (the reworded help compares with the recorded workbook text).

- [ ] **Step 4: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| E15 | Temperature design!C141 (feeds C143, C145, C154-C161, C171, C172, C180-C182, C186, C189, C190, C192, C193, C196, C20) | cup, boss and hub at 4140's specific heat even when the mass model makes them aluminium | aluminium parts at C140 (900 J/(kg·K)): the cup and boss on E9's gate, the hub when C6 is not 1; hardware stays steel. Back iron 0 on E9: C141 45.78 → 63.02 J/K, no verdict changes | Addendum A row E15 (decisions 8, 15) |
```

with:

```markdown
| E15 | Temperature design!C141 (feeds C143, C145, C154-C161, C171, C172, C180-C182, C186, C189, C190, C192, C193, C196, C20) | cup, boss and hub at 4140's specific heat even when the mass model makes them aluminium | aluminium parts at C140 (900 J/(kg·K)): the cup and boss on E9's gate, the hub when C6 is not 1; hardware stays steel. Back iron 0 on E9: C141 45.78 → 63.02 J/K, no verdict changes | Addendum A row E15 (decisions 8, 15) |
| E16 | Metal design!C189 → C191, C148, C149 (C147, C189 help) | the disc bored out of the web for the adapter pilot at steel density | at the web's density, one source with the mass model (`model::cup_boss_density`): aluminium with no back iron under E9. Back iron 0 on E9: C189 3.453 → 1.188 g, C191 101.5 → 103.8 g | Addendum A row E16 (decisions 9, 14, 15) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
help text, record the workbook's text in `workbook_help` (an input path, or a
table column as `group.table[*].field`):
```

with:

```markdown
help text, record the workbook's text in `workbook_help` (an input path, a
scalar result path (decision 14: E16 and E17 keep the workbook's labels and
reword the help), or a table column as `group.table[*].field`):
```

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task6.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/metal_design.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/deviations.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/tests/python_schema.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
fix(magcoupling-rs): apply E16 removed web disc at the web's density

The disc bored out of the web is priced at the density the mass model gives
the web (decision 9): aluminium with no back iron under E9. C147 and C189 keep
their labels; the help is reworded (decision 14).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 7: E17: aluminium eddy losses with the low-Reynolds closed form and the free-space fields

**Model:** session model (omit `model`; a `risk: physics` correction). Physics reviewer: report row E17, sections 5.4 (E17 table), 5.5 and 5.6.

Decisions 10-15 A: T1 (never non-conservative, within 1.5 % of the exact layered solution at the operating point), end factor
C114 = 0.7 (the workbook's thin-conductor convention), three stored free-space fields pinned at 4 significant figures (the probe
basis: only these literals reproduce the report's full-precision values, section 5.6 item 2), hub included, labels kept with
reworded help. The hub field with a steel cup (E9 off) is C116/2 = 0.1035 T (the amended standalone column). The headline
flip of C19 ("never" to 440.3 s) sits inside the model's uncertainty (report 5.5.5): the README row says so.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs` (three Rust-only free-space inputs, links `cup_wall_mm`, `web_mm`, T1 under E17, reworded help of C123-C125, unit test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs` (fill the two links)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs` (E17 Applied, probe on E9 at full precision, `workbook_help`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs` (the 41-cell tables, full precision, standalone, the report's headline table)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/data/input_schema.json` (the three inputs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Differences table)

**Interfaces:**
- Consumes: `TemperatureLinks { cup_aluminium, hub_aluminium, .. }` (Task 5), test helpers (Task 5).
- Produces:
  - inputs `temperature.slip_loss.b_hub_free_T = 0.07832`, `b_cup_free_T = 0.08764` (range 0 to 1, step 1e-5), `web_integral_free_T2m2 = 6.837e-6` (log range 1e-8 to 1e-3), all Rust-only;
  - `TemperatureLinks { .., cup_wall_mm: f64 /* Metal design C122 */, web_mm: f64 /* C125 */ }`;
  - under E17: hub (C6 != 1) and cup and web (E9's aluminium cup) priced with T1: P = f_end sigma_Al w_e^2 B^2 / (2 k^2) d_eff 2 pi r L, d_eff = (1 - e^(-2kd)) / (2k), k = p / r; web: (r_mid/p)^2 d_eff times the free integral.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e16_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
#[test]
fn e17_aluminium_eddy_losses_match_the_report() {
    // Report 5.4, E17 (T1, end factor C114 = 0.7, the 4-s.f. free-space fields): 41 cells in
    // the main column (workbook + E9, decision 15) and the M2 column. C152 stays "never" and
    // no verdict text changes.
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let e17 = DeviationId::E17;
    let main: ReportRows = &[
        ("Temperature design!C17", 73.26, 74.73),
        ("Temperature design!C18", 89.77, 94.18),
        ("Temperature design!C20", 66.01, 66.19),
        ("Temperature design!C22", 0.6881, 0.8105),
        ("Temperature design!C123", 0.2259, 0.1942),
        ("Temperature design!C124", 1.353, 1.543),
        ("Temperature design!C125", 0.0914, 0.3737),
        ("Temperature design!C130", 2.477, 2.918),
        ("Temperature design!C131", 0.01183, 0.01393),
        ("Temperature design!C132", 2.477, 2.918),
        ("Temperature design!C134", 7.431, 8.754),
        ("Temperature design!C145", 0.005411, 0.006374),
        ("Temperature design!C146", 8.257, 9.726),
        ("Temperature design!C147", 24.77, 29.18),
        ("Temperature design!C148", 73.26, 74.73),
        ("Temperature design!C149", 89.77, 94.18),
        ("Temperature design!C154", 0.05411, 0.06374),
        ("Temperature design!C155", 0.1623, 0.1912),
        ("Temperature design!C156", 616.1, 523.0),
        ("Temperature design!C157", 205.4, 174.3),
        ("Temperature design!C161", 65.32, 65.38),
        ("Temperature design!C169", 0.2477, 0.2918),
        ("Temperature design!C170", 0.7431, 0.8754),
        ("Temperature design!C171", 0.005411, 0.006374),
        ("Temperature design!C172", 0.01623, 0.01912),
        ("Temperature design!C173", 14.86, 17.51),
        ("Temperature design!C175", 0.2294, 0.2702),
        ("Temperature design!C176", 0.6881, 0.8105),
        ("Temperature design!C177", 0.2477, 0.2918),
        ("Temperature design!C180", 66.01, 66.19),
        ("Temperature design!C181", 36.54, 36.36),
        ("Temperature design!C182", 26.54, 26.36),
        ("Temperature design!C186", 1.63, 1.63),
        ("Temperature design!C189", 66.01, 66.19),
        ("Temperature design!C190", 53.99, 53.81),
        ("Temperature design!C192", 1.875, 1.874),
        ("Temperature design!C193", 0.1981, 0.198),
        ("Temperature design!C196", 7.571, 7.575),
    ];
    // C19, C150 and C151 are text ("never") before and numbers after: checked below.
    let text_cells = [
        "Temperature design!C19",
        "Temperature design!C150",
        "Temperature design!C151",
    ];
    let check = |label: &str, before: Deviations, after: Deviations, rows: ReportRows, t19: f64, r151: f64| {
        let (b, a) = (cells_with(&NO_BACK_IRON, before), cells_with(&NO_BACK_IRON, after));
        let mut want: BTreeSet<String> = rows.iter().map(|(c, _, _)| (*c).to_owned()).collect();
        want.extend(text_cells.iter().map(|c| (*c).to_owned()));
        assert_eq!(changed_cells(&b, &a), want, "{label}: changed cells");
        for &(cell, was, now) in rows {
            assert_sig4(&format!("{label} {cell} before"), &b[cell], was);
            assert_sig4(&format!("{label} {cell} after"), &a[cell], now);
        }
        for cell in ["Temperature design!C19", "Temperature design!C150"] {
            assert_eq!(
                b[cell],
                Value::Text("never: steady state stays below the limit".into()),
                "{label} {cell}"
            );
            assert_sig4(&format!("{label} {cell}"), &a[cell], t19);
        }
        assert_eq!(b["Temperature design!C151"], Value::Text("never".into()));
        assert_sig4(&format!("{label} C151"), &a["Temperature design!C151"], r151);
        assert_eq!(
            a["Temperature design!C152"],
            Value::Text("never: steady state stays below the limit".into()),
            "{label}: the estimate never reaches the limit"
        );
    };
    check("E17 on E9", e9, e9.with(e17), main, 440.3, 1.468e4);
    let m2_rows: ReportRows = &[
        ("Temperature design!C17", 74.17, 74.73),
        ("Temperature design!C18", 92.51, 94.18),
        ("Temperature design!C20", 66.12, 66.19),
        ("Temperature design!C22", 0.7643, 0.8105),
        ("Temperature design!C123", 0.2259, 0.1942),
        ("Temperature design!C124", 1.353, 1.543),
        ("Temperature design!C125", 0.3656, 0.3737),
        ("Temperature design!C130", 2.751, 2.918),
        ("Temperature design!C131", 0.01314, 0.01393),
        ("Temperature design!C132", 2.751, 2.918),
        ("Temperature design!C134", 8.254, 8.754),
        ("Temperature design!C145", 0.00601, 0.006374),
        ("Temperature design!C146", 9.171, 9.726),
        ("Temperature design!C147", 27.51, 29.18),
        ("Temperature design!C148", 74.17, 74.73),
        ("Temperature design!C149", 92.51, 94.18),
        ("Temperature design!C154", 0.0601, 0.06374),
        ("Temperature design!C155", 0.1803, 0.1912),
        ("Temperature design!C156", 554.7, 523.0),
        ("Temperature design!C157", 184.9, 174.3),
        ("Temperature design!C161", 65.36, 65.38),
        ("Temperature design!C169", 0.2751, 0.2918),
        ("Temperature design!C170", 0.8254, 0.8754),
        ("Temperature design!C171", 0.00601, 0.006374),
        ("Temperature design!C172", 0.01803, 0.01912),
        ("Temperature design!C173", 16.51, 17.51),
        ("Temperature design!C175", 0.2548, 0.2702),
        ("Temperature design!C176", 0.7643, 0.8105),
        ("Temperature design!C177", 0.2751, 0.2918),
        ("Temperature design!C180", 66.12, 66.19),
        ("Temperature design!C181", 36.93, 36.87),
        ("Temperature design!C182", 26.93, 26.87),
        ("Temperature design!C186", 1.63, 1.63),
        ("Temperature design!C189", 66.12, 66.19),
        ("Temperature design!C190", 53.88, 53.81),
        ("Temperature design!C192", 1.874, 1.874),
        ("Temperature design!C193", 0.1981, 0.198),
        ("Temperature design!C196", 7.573, 7.575),
    ];
    check("E17 on M2", m2(), m2().with(e17), m2_rows, 497.0, 1.657e4);

    // Full precision, workbook + E9 basis, with the 4-s.f. fields pinned (report 5.4, decision 12).
    let c = cells_with(&NO_BACK_IRON, e9.with(e17));
    for (cell, want) in [
        ("Temperature design!C123", 0.1942432063711941),
        ("Temperature design!C124", 1.5432858065214725),
        ("Temperature design!C125", 0.3736960055840505),
        ("Temperature design!C130", 2.917946124198304),
        ("Temperature design!C18", 94.17946124198303),
        ("Temperature design!C19", 440.3079945062734),
    ] {
        assert!(parity_close(&c[cell], &Value::Num(want)), "{cell}: {:?} vs {want}", c[cell]);
    }

    // Standalone (E9 off, amended): the cup and web stay steel; the aluminium hub sees half
    // the doubled steel-circuit field, C116 / 2 = 0.1035 T (report 5.6, correction 1).
    let alone = cells_with(&NO_BACK_IRON, Deviations::only(e17));
    let workbook = cells_with(&NO_BACK_IRON, Deviations::NONE);
    assert_sig4("E17 alone C123", &alone["Temperature design!C123"], 0.3392);
    assert_sig4("E17 alone C130", &alone["Temperature design!C130"], 2.590);
    assert_sig4("E17 alone C18", &alone["Temperature design!C18"], 90.90);
    for cell in ["Temperature design!C124", "Temperature design!C125"] {
        assert_eq!(alone[cell], workbook[cell], "{cell}: the steel cup and web keep the workbook formula");
    }
}

#[test]
fn e15_to_e17_together_match_the_reports_headline_table() {
    // Report 5.2 at back iron = 0: the combined columns. No correction changes the torque,
    // the governing limit or the verdict C25.
    let e9 = Deviations::NONE.with(DeviationId::E9);
    let all3 = |base: Deviations| {
        base.with(DeviationId::E15)
            .with(DeviationId::E16)
            .with(DeviationId::E17)
    };
    let c = cells_with(&NO_BACK_IRON, all3(e9));
    assert_sig4("C18", &c["Temperature design!C18"], 94.18);
    assert_sig4("C19", &c["Temperature design!C19"], 606.1);
    assert_sig4("C151", &c["Temperature design!C151"], 2.020e4);
    assert_sig4("C20", &c["Temperature design!C20"], 66.09);
    let m = cells_with(&NO_BACK_IRON, all3(m2()));
    assert_sig4("M2 C19", &m["Temperature design!C19"], 684.1);
    assert_sig4("M2 C151", &m["Temperature design!C151"], 2.280e4);
    assert_sig4("M2 C20", &m["Temperature design!C20"], 66.09);
    assert_sig4("M2 C12", &m["Temperature design!C12"], 93.06);
    assert_sig4("M2 C23", &m["Temperature design!C23"], 0.04019);
    for (label, cells) in [("E9", &c), ("M2", &m)] {
        assert_sig4(&format!("{label} C93"), &cells["Calculator!C93"], 1.697);
        assert_eq!(
            cells["Temperature design!C152"],
            Value::Text("never: steady state stays below the limit".into())
        );
        assert_eq!(
            cells["Temperature design!C25"],
            Value::Text("OK on temperature. Confirm drag torque and thermal cycling by test.".into())
        );
    }
}

#[test]
fn e17_leaves_every_default_cell_bit_for_bit() {
    // At defaults every loss term is steel: the skin-limited workbook formulas stay.
    assert_bit_for_bit_at_defaults(DeviationId::E17);
}

#[test]
fn e16_leaves_every_default_cell_bit_for_bit() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    #[test]
    fn e13_changes_only_the_rotations_per_degree() {
```

with:

```rust
    #[test]
    fn e17_prices_aluminium_parts_with_the_low_reynolds_closed_form() {
        // Report 5.5.2-5.5.3 at the fixture (10 poles, 2000 rpm slip): an independent
        // evaluation of T1 per part, P = f_end sigma w^2 B^2 (1 - e^(-2kd)) / (4 k^3) 2 pi r L,
        // against the engine's P = f_end sigma w^2 B^2 / (2 k^2) d_eff 2 pi r L.
        use crate::engine::constants::MU0;
        let e17 = Deviations::only(DeviationId::E17);
        let mut k = links();
        k.cup_aluminium = true; // E9 with no back iron
        k.hub_aluminium = true;
        let ti = TemperatureInputs::default();
        let sl = &ti.slip_loss;
        let r = compute(&ti, &k, e17);
        let (pp, sigma, f_end) = (5.0_f64, k.al6061_sigma_S_m, sl.end_factor);
        let we = 2.0 * PI * pp * k.slip_rpm / 60.0;
        let length = k.active_length_mm / 1000.0;
        let t1 = |b: f64, radius: f64, d: f64| {
            let wave = pp / radius;
            f_end * sigma * we * we * b * b * (1.0 - (-2.0 * wave * d).exp())
                / (4.0 * wave * wave * wave)
                * 2.0
                * PI
                * radius
                * length
        };
        let r_cup = (k.outer_back_apothem_mm + k.bond_outer_mm) / 1000.0;
        let r_hub = (k.inner_back_apothem_mm - k.bond_inner_mm) / 1000.0;
        let r_web = (k.inner_back_apothem_mm + k.inner_thickness_mm / 2.0) / 1000.0;
        let close = |got: f64, want: f64| (got - want).abs() <= 1e-12 * want.abs();
        let want_cup = t1(sl.b_cup_free_T, r_cup, k.cup_wall_mm / 1000.0);
        let want_hub = t1(sl.b_hub_free_T, r_hub, k.hub_wall_mm / 1000.0);
        assert!(close(r.slip_loss.cup_W, want_cup), "{} {want_cup}", r.slip_loss.cup_W);
        assert!(close(r.slip_loss.hub_W, want_hub), "{} {want_hub}", r.slip_loss.hub_W);
        let wave_web = pp / r_web;
        let want_web = f_end * sigma * we * we / 2.0 * (r_web / pp).powi(2)
            * (1.0 - (-2.0 * wave_web * k.web_mm / 1000.0).exp())
            / (2.0 * wave_web)
            * sl.web_integral_free_T2m2;
        assert!(close(r.slip_loss.web_W, want_web), "{} {want_web}", r.slip_loss.web_W);
        // The report's regime numbers (4 s.f.): k = p / r and the aluminium skin depth, which
        // exceeds every part (d / delta < 1): the premise of T1.
        let sig4 = |x: f64, want: f64| {
            (x - want).abs() <= 0.5 * 10f64.powi(want.log10().floor() as i32 - 3)
        };
        assert!(sig4(pp / r_cup, 278.7) && sig4(pp / r_hub, 495.0) && sig4(pp / r_web, 426.1));
        let delta_al = (2.0 / (we * MU0 * sigma)).sqrt();
        assert!(sig4(delta_al * 1000.0, 7.797), "{delta_al}");
        for d_mm in [k.cup_wall_mm, k.web_mm, k.hub_wall_mm] {
            assert!(d_mm / 1000.0 < delta_al, "{d_mm} mm");
        }
        // A steel cup (E9 off) gives the aluminium hub half the doubled steel-circuit field.
        k.cup_aluminium = false;
        let steel_cup = compute(&ti, &k, e17);
        let want_hub = t1(sl.b_hub_T / 2.0, r_hub, k.hub_wall_mm / 1000.0);
        assert!(close(steel_cup.slip_loss.hub_W, want_hub));
        assert_eq!(steel_cup.slip_loss.cup_W, run(&ti, &k).slip_loss.cup_W);
        assert_eq!(steel_cup.slip_loss.web_W, run(&ti, &k).slip_loss.web_W);
    }

    #[test]
    fn e13_changes_only_the_rotations_per_degree() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations e17 2>&1 | grep -E "panicked|test result" -A2 | head -8
```

Expected: `e17_aluminium_eddy_losses_match_the_report` FAILS with `E17 on E9: changed cells`; `e15_to_e17_together_match_the_reports_headline_table` FAILS (C19 is still "never").

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --lib e17_prices 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors, among them ``no field `b_cup_free_T` `` and ``no field `cup_wall_mm` ``.

- [ ] **Step 3: Implement E17**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
use super::compat::{py_max, py_min, text0};
use super::deviations::{DeviationId, Deviations};
use super::meta::{NumOrText, inputs, out, out_uncelled, param, results};
```

with:

```rust
use super::compat::{py_max, py_min, text0};
use super::deviations::{DeviationId, Deviations};
use super::meta::{NumOrText, inputs, out, out_uncelled, param, param_rust_only, results};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            high_multiplier: f64 = 3.0 => param("-", "High-case multiplier on the estimate",
                "Set to 1 once measured.", "Temperature design!C133")
                .range(1.0, 10.0, 0.1),
        }
    }
}
```

with:

```rust
            high_multiplier: f64 = 3.0 => param("-", "High-case multiplier on the estimate",
                "Set to 1 once measured.", "Temperature design!C133")
                .range(1.0, 10.0, 0.1),
            b_hub_free_T: f64 = 0.07832 => param_rust_only("T", "Opposite-ring field at an aluminium hub (fundamental, free space)",
                "Correction E17, no back iron with an aluminium cup: 3D, no steel image and not doubled (the steel-circuit field is C116). Stored at 4 significant figures (decision 12) until M3 computes it live.")
                .range(0.0, 1.0, 0.00001),
            b_cup_free_T: f64 = 0.08764 => param_rust_only("T", "Opposite-ring field at an aluminium cup (fundamental, free space)",
                "Correction E17, no back iron: 3D, no steel image and not doubled (the steel-circuit field is C117). Stored at 4 significant figures (decision 12) until M3 computes it live.")
                .range(0.0, 1.0, 0.00001),
            web_integral_free_T2m2: f64 = 6.837e-6 => param_rust_only("T²·m²", "Rear-web end field of an aluminium web, ∫B² dA (free space)",
                "Correction E17, no back iron: the inner ring alone, no steel image (the steel-circuit value is C121). Stored at 4 significant figures (decision 12) until M3 computes it live.")
                .range(1e-8, 1e-3, 1e-9)
                .log(),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    pub cap_face_mm: f64,              // Metal design C167
```

with:

```rust
    pub cap_face_mm: f64,              // Metal design C167
    pub cup_wall_mm: f64,              // Metal design C122 (E17: the aluminium cup's wall)
    pub web_mm: f64,                   // Metal design C125 (E17: the aluminium web)
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            hub_W: f64 => out("W", "Hub surface (solid steel)", "", "Temperature design!C123"),
            cup_W: f64 => out("W", "Cup surface (solid steel)", "", "Temperature design!C124"),
            web_W: f64 => out("W", "Rear web (solid steel)", "", "Temperature design!C125"),
```

with:

```rust
            hub_W: f64 => out("W", "Hub surface (solid steel)",
                "Solid 4140 with back iron (skin-limited formula). With no back iron the hub is 6061 aluminium: low-Reynolds form of correction E17.",
                "Temperature design!C123"),
            cup_W: f64 => out("W", "Cup surface (solid steel)",
                "Solid 4140 with back iron (skin-limited formula). With no back iron the cup is 6061 aluminium (E9): low-Reynolds form of correction E17.",
                "Temperature design!C124"),
            web_W: f64 => out("W", "Rear web (solid steel)",
                "Solid 4140 with back iron (skin-limited formula). With no back iron the web is 6061 aluminium (E9): low-Reynolds form of correction E17.",
                "Temperature design!C125"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let p_hub = surface(sl.b_hub_T, r_hub);
    let p_cup = surface(sl.b_cup_T, r_cup);
    let p_web = k.steel_sigma_S_m * we.powi(2) * (delta / 1000.0) / 4.0
        * ((r_mid / 1000.0) / pp).powi(2)
        * sl.web_integral_T2m2;
```

with:

```rust
    // E17 (report 5.5.3): an aluminium part is resistance-limited, not skin-limited, and
    // sees the free-space field. Low-Reynolds closed form T1 with the thin-conductor end
    // factor: P = f_end sigma_Al we^2 B^2 / (2 k^2) d_eff 2 pi r L, k = p / r and
    // d_eff = (1 - e^(-2 k d)) / (2 k), d the part's thickness [m].
    let e17 = dev.is_on(DeviationId::E17);
    let d_eff = |kk: f64, d_m: f64| (1.0 - (-2.0 * kk * d_m).exp()) / (2.0 * kk);
    let aluminium_surface = |B: f64, r: f64, d_m: f64| {
        let kk = pp / r;
        sl.end_factor * k.al6061_sigma_S_m * we.powi(2) * B.powi(2) / (2.0 * kk.powi(2))
            * d_eff(kk, d_m)
            * 2.0
            * PI
            * r
            * L
    };
    let p_hub = if e17 && k.hub_aluminium {
        // With a steel cup (E9 off) the outer ring has its first-order image in the cup:
        // half the doubled steel-circuit field (report 5.6, amended).
        let b = if k.cup_aluminium {
            sl.b_hub_free_T
        } else {
            sl.b_hub_T / 2.0
        };
        aluminium_surface(b, r_hub, k.hub_wall_mm / 1000.0)
    } else {
        surface(sl.b_hub_T, r_hub)
    };
    let p_cup = if e17 && k.cup_aluminium {
        aluminium_surface(sl.b_cup_free_T, r_cup, k.cup_wall_mm / 1000.0)
    } else {
        surface(sl.b_cup_T, r_cup)
    };
    let p_web = if e17 && k.cup_aluminium {
        // (r_mid / p)^2 replaces 1 / k^2 and the free-space integral replaces A B^2.
        let r_w = r_mid / 1000.0;
        sl.end_factor * k.al6061_sigma_S_m * we.powi(2) / 2.0
            * (r_w / pp).powi(2)
            * d_eff(pp / r_w, k.web_mm / 1000.0)
            * sl.web_integral_free_T2m2
    } else {
        k.steel_sigma_S_m * we.powi(2) * (delta / 1000.0) / 4.0
            * ((r_mid / 1000.0) / pp).powi(2)
            * sl.web_integral_T2m2
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            cap_face_mm: 0.8,
            hardware_g: 6.0,
```

with:

```rust
            cap_face_mm: 0.8,
            cup_wall_mm: 1.8,
            web_mm: 2.5,
            hardware_g: 6.0,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        cap_face_mm: md.cap_axial_mm,
        hardware_g: md.hardware_g,
```

with:

```rust
        cap_face_mm: md.cap_axial_mm,
        cup_wall_mm: md.cup_wall_corner_mm,
        web_mm: md.web_mm,
        hardware_g: md.hardware_g,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            decisions: &[10, 11, 12, 13, 14, 15],
        },
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Planned,
```

with:

```rust
            decisions: &[10, 11, 12, 13, 14, 15],
        },
        depends_on: &[DeviationId::E9],
        status: DeviationStatus::Applied,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            pinned at 4 s.f. (decision 12; M3 computes them live). The labels C123 to C125 stay; their help is reworded (decision 14).",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
```

with:

```rust
            pinned at 4 s.f. (decision 12; M3 computes them live). The labels C123 to C125 stay; their help is reworded (decision 14).",
        workbook_input_defaults: &[],
        workbook_help: &[
            ("temperature.slip_loss.hub_W", ""),
            ("temperature.slip_loss.cup_W", ""),
            ("temperature.slip_loss.web_W", ""),
        ],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "no intentional back iron (coupling.backiron = 0), on top of E9 (report 5.4, E17, full precision)",
            inputs: &[("coupling.backiron", Literal::Int(0))],
            expect: &[
                CellChange {
                    cell: "Temperature design!C123",
                    workbook: Literal::Num(0.22590854790913922),
                    corrected: Literal::Num(0.1942432063711941),
                },
                CellChange {
                    cell: "Temperature design!C124",
                    workbook: Literal::Num(1.3530776316850626),
                    corrected: Literal::Num(1.5432858065214725),
                },
                CellChange {
                    cell: "Temperature design!C125",
                    workbook: Literal::Num(0.09140096902640166),
                    corrected: Literal::Num(0.3736960055840505),
                },
                CellChange {
                    cell: "Temperature design!C130",
                    workbook: Literal::Num(2.4771082543421903),
                    corrected: Literal::Num(2.917946124198304),
                },
                CellChange {
                    cell: "Temperature design!C18",
                    workbook: Literal::Num(89.7710825434219),
                    corrected: Literal::Num(94.17946124198303),
                },
                CellChange {
                    cell: "Temperature design!C19",
                    workbook: Literal::Text("never: steady state stays below the limit"),
                    corrected: Literal::Num(440.3079945062734),
                },
            ],
        }],
    },
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! and C157 instead of Python's ZeroDivisionError), and E15 (C141 prices an
//! aluminium cup, boss and hub at C140, on the gates the masses read).
```

with:

```rust
//! and C157 instead of Python's ZeroDivisionError), E15 (C141 prices an
//! aluminium cup, boss and hub at C140, on the gates the masses read), and
//! E17 (with no back iron the aluminium hub, cup and web losses C123-C125 take
//! the low-Reynolds closed form T1 with the Rust-only free-space fields).
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

- [ ] **Step 4: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test schema
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: the bless run adds the three rows (`"rust_only": true`); every binary `ok`: unit tests 96 passed, `deviations.rs` 44 passed.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|panicked"
```

Expected: no output.

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 5: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| E16 | Metal design!C189 → C191, C148, C149 (C147, C189 help) | the disc bored out of the web for the adapter pilot at steel density | at the web's density, one source with the mass model (`model::cup_boss_density`): aluminium with no back iron under E9. Back iron 0 on E9: C189 3.453 → 1.188 g, C191 101.5 → 103.8 g | Addendum A row E16 (decisions 9, 14, 15) |
```

with:

```markdown
| E16 | Metal design!C189 → C191, C148, C149 (C147, C189 help) | the disc bored out of the web for the adapter pilot at steel density | at the web's density, one source with the mass model (`model::cup_boss_density`): aluminium with no back iron under E9. Back iron 0 on E9: C189 3.453 → 1.188 g, C191 101.5 → 103.8 g | Addendum A row E16 (decisions 9, 14, 15) |
| E17 | Temperature design!C123, C124, C125 → C130-C134 and the thermal rows (41 cells); 3 Rust-only inputs | the steel skin-limited formula, 4140's σ and μr and the steel-circuit fields for the aluminium hub, cup and web | at C6 = 0 the low-Reynolds closed form T1 (end factor C114, σ = Materials!C43) with the free-space fields `temperature.slip_loss.b_hub_free_T` 0.07832 T, `b_cup_free_T` 0.08764 T, `web_integral_free_T2m2` 6.837e-6 T²·m² (4 s.f., decision 12; M3 computes them live); with E9 off the hub sees C116/2. Back iron 0 on E9: C130 2.477 → 2.918 W, C18 89.77 → 94.18 °C, C19 "never" → 440.3 s (inside the model's uncertainty: finite only for f_end ≥ 0.646) | Addendum A row E17 (decisions 10-15) |
```

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task7.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task7.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task7.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/deviations.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
fix(magcoupling-rs): apply E17 aluminium eddy losses (T1, free-space fields)

With no back iron the aluminium hub, cup and web are resistance-limited: the
low-Reynolds closed form T1 with end factor C114 and three Rust-only free-space
fields pinned at 4 s.f. (decisions 10-15). Reproduces the report's 41-cell
tables and full-precision values; defaults bit for bit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 8: E18: the adhesive mismatch screen of an aluminium hub

**Model:** session model (omit `model`; a `risk: physics` correction). Physics reviewer: report row E18 and decision 16.

Decision 16 A: register E18 with E15's hub gate. The report's expected values: on the workbook basis C104 46.13 → 73.87 MPa
(both verdicts already "Above"); on the M2 basis (E1's 0.107 GPa) C104 11.68 → 20.76, C105 6.071 → 11.03, C201 2.563 → 4.655 MPa
and both verdicts flip. The report's 68.9 GPa is kept although the library's 6061 record carries Kaiser's 68.3 GPa (decision 5):
with 68.3 the M2-basis C104 would read 20.75 and the workbook-basis 73.74, off the report at the 4th figure (Decisions to
confirm, A8).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs` (`AL_HUB_CTE_PER_C`, `AL_HUB_MODULUS_GPA`; the hub's CTE and modulus under E18)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs` (E18 Applied, workbook-basis probe)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs` (workbook and M2 bases; the flips; `ALL.without(E18)`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Differences table)

**Interfaces:**
- Consumes: `TemperatureLinks::hub_aluminium` (Task 5), `Deviations::without` (Task 2), `m2()` (Task 5).
- Produces: `pub const temperature::AL_HUB_CTE_PER_C: f64 = 23.6e-6` and `pub const temperature::AL_HUB_MODULUS_GPA: f64 = 68.9` (the report's Alliance 6061-T6 values; Task 12 routes them through the body material).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e17_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
#[test]
fn e18_aluminium_hub_mismatch_matches_the_report() {
    // Report row E18. On the workbook basis (C96 = 0.55 GPa) the screens already read
    // "Above": only the numbers move. On the M2 basis (E1's 0.107 GPa) both verdicts flip.
    let e18 = DeviationId::E18;
    let workbook: ReportRows = &[
        ("Temperature design!C104", 46.13, 73.87),
        ("Temperature design!C105", 26.73, 45.1),
        ("Temperature design!C201", 11.37, 19.18),
    ];
    assert_report_table(
        "E18 workbook",
        &NO_BACK_IRON,
        Deviations::NONE,
        Deviations::only(e18),
        workbook,
    );
    let (b, a) = (
        cells_with(&NO_BACK_IRON, m2()),
        cells_with(&NO_BACK_IRON, m2().with(e18)),
    );
    let flips = [
        (
            "Temperature design!C106",
            "Below the lap-shear strength",
            "Above the lap-shear strength at the block ends",
        ),
        (
            "Temperature design!C202",
            "Below the fatigue endurance",
            "Above the fatigue endurance: qualify by thermal cycling",
        ),
    ];
    let mut want: BTreeSet<String> = ["C104", "C105", "C201"]
        .iter()
        .map(|c| format!("Temperature design!{c}"))
        .collect();
    want.extend(flips.iter().map(|(c, _, _)| (*c).to_owned()));
    assert_eq!(changed_cells(&b, &a), want, "E18 on M2: changed cells");
    for (cell, was, now) in [
        ("Temperature design!C104", 11.68, 20.76),
        ("Temperature design!C105", 6.071, 11.03),
        ("Temperature design!C201", 2.563, 4.655),
    ] {
        assert_sig4(&format!("M2 {cell} before"), &b[cell], was);
        assert_sig4(&format!("M2 {cell} after"), &a[cell], now);
    }
    for (cell, was, now) in flips {
        assert_eq!(b[cell], Value::Text(was.into()), "{cell}");
        assert_eq!(a[cell], Value::Text(now.into()), "{cell}");
    }
    // C94 and C98 still show the steel inputs.
    assert_eq!(a["Temperature design!C94"], Value::Num(12.3e-6));
    assert_eq!(a["Temperature design!C98"], Value::Num(205.0));
    // What users see: every correction on, less E18, against every correction on.
    let (without, with) = (
        cells_with(&NO_BACK_IRON, Deviations::ALL.without(e18)),
        cells_with(&NO_BACK_IRON, Deviations::ALL),
    );
    assert_eq!(changed_cells(&without, &with), want, "E18 on ALL");
}

#[test]
fn e18_leaves_every_default_cell_bit_for_bit() {
    // At defaults the hub is steel: the workbook's CTE and modulus stay.
    assert_bit_for_bit_at_defaults(DeviationId::E18);
}

#[test]
fn e17_leaves_every_default_cell_bit_for_bit() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations e18 2>&1 | grep -E "panicked|test result" -A2 | head -8
```

Expected: `e18_aluminium_hub_mismatch_matches_the_report` FAILS with `E18 workbook: changed cells`.

- [ ] **Step 3: Implement E18**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
/// The text of an unbroken-slip time that never reaches the limit.
```

with:

```rust
/// E18: the expansion coefficient of the aluminium hub [1/°C] (6061-T6, Alliance
/// datasheet <https://www.allianceorg.com/pdfs/alumext/6061t6.pdf>: 23.6e-6 /°C;
/// Addendum A report, row E18).
pub const AL_HUB_CTE_PER_C: f64 = 23.6e-6;
/// E18: the elastic modulus of the aluminium hub [GPa] (6061-T6, the same Alliance
/// datasheet: 68.9 GPa).
pub const AL_HUB_MODULUS_GPA: f64 = 68.9;

/// The text of an unbroken-slip time that never reaches the limit.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let dT = py_max(sel.cure_C - k.min_temp_C, gov - sel.cure_C);
    let d_alpha = k.steel_cte - mm.ndfeb_cte_per_C;
```

with:

```rust
    let dT = py_max(sel.cure_C - k.min_temp_C, gov - sel.cure_C);
    // E18: the blocks bond to the hub, which the mass model makes aluminium when C6 != 1
    // (E15's hub gate); the workbook screens 4140 whatever the hub is. C94 and C98 still
    // show the steel inputs.
    let (hub_cte, hub_E_GPa) = if dev.is_on(DeviationId::E18) && k.hub_aluminium {
        (AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA)
    } else {
        (k.steel_cte, k.steel_E_GPa)
    };
    let d_alpha = hub_cte - mm.ndfeb_cte_per_C;
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        k.bond_inner_mm,
        mm.ndfeb_modulus_GPa,
        k.inner_thickness_mm,
        k.steel_E_GPa,
        k.hub_wall_mm,
        k.inner_length_mm,
    );
```

with:

```rust
        k.bond_inner_mm,
        mm.ndfeb_modulus_GPa,
        k.inner_thickness_mm,
        hub_E_GPa,
        k.hub_wall_mm,
        k.inner_length_mm,
    );
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        mm.recommended_bondline_mm,
        mm.ndfeb_modulus_GPa,
        k.inner_thickness_mm,
        k.steel_E_GPa,
        k.hub_wall_mm,
        k.inner_length_mm,
    );
```

with:

```rust
        mm.recommended_bondline_mm,
        mm.ndfeb_modulus_GPa,
        k.inner_thickness_mm,
        hub_E_GPa,
        k.hub_wall_mm,
        k.inner_length_mm,
    );
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        approval: Approval::Addendum { decisions: &[16] },
        depends_on: &[],
        status: DeviationStatus::Planned,
```

with:

```rust
        approval: Approval::Addendum { decisions: &[16] },
        depends_on: &[],
        status: DeviationStatus::Applied,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            C94 and C98 still show Materials!C17 and C18.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
```

with:

```rust
            C94 and C98 still show Materials!C17 and C18.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[Probe {
            label: "no intentional back iron (coupling.backiron = 0), workbook basis (report row E18: C104 46.13 -> 73.87 MPa)",
            inputs: &[("coupling.backiron", Literal::Int(0))],
            expect: &[
                CellChange {
                    cell: "Temperature design!C104",
                    workbook: Literal::Num(46.12653767935644),
                    corrected: Literal::Num(73.8667679636259),
                },
                CellChange {
                    cell: "Temperature design!C105",
                    workbook: Literal::Num(26.728261752019268),
                    corrected: Literal::Num(45.09855207959816),
                },
                CellChange {
                    cell: "Temperature design!C201",
                    workbook: Literal::Num(11.365659614702167),
                    corrected: Literal::Num(19.1772587685732),
                },
            ],
        }],
    },
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! E17 (with no back iron the aluminium hub, cup and web losses C123-C125 take
//! the low-Reynolds closed form T1 with the Rust-only free-space fields).
```

with:

```rust
//! E17 (with no back iron the aluminium hub, cup and web losses C123-C125 take
//! the low-Reynolds closed form T1 with the Rust-only free-space fields) and
//! E18 (the mismatch screen C104-C106, C201, C202 bonds to an aluminium hub when
//! C6 is not 1).
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, `deviations.rs` 46 passed.

- [ ] **Step 4: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| E17 | Temperature design!C123, C124, C125 → C130-C134 and the thermal rows (41 cells); 3 Rust-only inputs | the steel skin-limited formula, 4140's σ and μr and the steel-circuit fields for the aluminium hub, cup and web | at C6 = 0 the low-Reynolds closed form T1 (end factor C114, σ = Materials!C43) with the free-space fields `temperature.slip_loss.b_hub_free_T` 0.07832 T, `b_cup_free_T` 0.08764 T, `web_integral_free_T2m2` 6.837e-6 T²·m² (4 s.f., decision 12; M3 computes them live); with E9 off the hub sees C116/2. Back iron 0 on E9: C130 2.477 → 2.918 W, C18 89.77 → 94.18 °C, C19 "never" → 440.3 s (inside the model's uncertainty: finite only for f_end ≥ 0.646) | Addendum A row E17 (decisions 10-15) |
```

with:

```markdown
| E17 | Temperature design!C123, C124, C125 → C130-C134 and the thermal rows (41 cells); 3 Rust-only inputs | the steel skin-limited formula, 4140's σ and μr and the steel-circuit fields for the aluminium hub, cup and web | at C6 = 0 the low-Reynolds closed form T1 (end factor C114, σ = Materials!C43) with the free-space fields `temperature.slip_loss.b_hub_free_T` 0.07832 T, `b_cup_free_T` 0.08764 T, `web_integral_free_T2m2` 6.837e-6 T²·m² (4 s.f., decision 12; M3 computes them live); with E9 off the hub sees C116/2. Back iron 0 on E9: C130 2.477 → 2.918 W, C18 89.77 → 94.18 °C, C19 "never" → 440.3 s (inside the model's uncertainty: finite only for f_end ≥ 0.646) | Addendum A row E17 (decisions 10-15) |
| E18 | Temperature design!C104, C105, C201; verdicts C106, C202 | 4140's expansion coefficient and modulus (C94, C98) for a hub the mass model makes aluminium | at C6 ≠ 1 (E15's hub gate) the screen uses 6061-T6, 23.6e-6 /°C and 68.9 GPa (Alliance datasheet); C94 and C98 still show the steel inputs. M2 basis at back iron 0: C104 11.68 → 20.76 MPa, C106 and C202 read "Above ..." | Addendum A row E18 (decision 16) |
```

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task8.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task8.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task8.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/deviations.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
fix(magcoupling-rs): apply E18 mismatch screen of an aluminium hub

With C6 != 1 the blocks bond to an aluminium hub: 6061-T6 expansion and
modulus in the Volkersen screen (decision 16). On the M2 basis C106 and C202
flip to "Above", as the report states.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 9: E19: the SuperMagnetMan arcs follow the vendor's specification grid

**Model:** session model (omit `model`; a registered deviation). Reviewer: decision 2 and report section 3.

Decision 2 A: the vendor's grid (supermagnetman.com/products/m5044, m5045, m5026, read 2026-09-30) gives "Max Working Temp 60 C" on
all three arcs and "Neodymium 50" for M5045, whose title says N50M. Br stays at the workbook's 1.42 T. The report did not count
the Tmax cells; this task does: 26 cells per arc on the workbook basis, 20 on the M2 basis (the E12 guard zeroes the six time and
drag cells there). The grade mapping has no numeric effect until E20 reads it (Task 10).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs` (`vendor_tmax_C`, `vendor_grade`, `tmax_C()`, `grade_id()`, unit test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs` (`resolve_magnets` reads them)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs` (E19 Applied, two probes)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs` (the three arcs, both bases; M5045's grade)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/grades.rs` (each grade lists its parts (both mappings of the data file))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Differences table, Layout row of library.rs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (the library line)

**Interfaces:**
- Consumes: `MagnetSpec`, `MAGNET_LIBRARY` (Task 3), `ResolvedMagnet::grade` (Task 4).
- Produces:
  - `MagnetSpec { .., vendor_tmax_C: Option<f64>, vendor_grade: Option<&'static str> }` (60 °C on the three arcs; `Some("N50")` on M5045);
  - `pub fn library::tmax_C(spec: &MagnetSpec, dev: Deviations) -> f64` and `pub fn library::grade_id(spec: &MagnetSpec, dev: Deviations) -> &'static str` (E19 applies the vendor's values);
  - `resolve_magnets` rates and grades a library part through them.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e19_applies_the_vendor_grid_to_exactly_the_three_arcs() {
        let e19 = Deviations::only(DeviationId::E19);
        for spec in &MAGNET_LIBRARY {
            let arc = spec.vendor == "SuperMagnetMan";
            assert_eq!(spec.vendor_tmax_C.is_some(), arc, "{}", spec.part);
            assert_eq!(tmax_C(spec, Deviations::NONE), spec.tmax_C, "{}", spec.part);
            assert_eq!(grade_id(spec, Deviations::NONE), spec.grade, "{}", spec.part);
            let want = if arc { 60.0 } else { spec.tmax_C };
            assert_eq!(tmax_C(spec, e19), want, "{}", spec.part);
            assert_eq!(tmax_C(spec, Deviations::ALL), want, "{}", spec.part);
            let want_grade = if spec.part == "M5045" { "N50" } else { spec.grade };
            assert_eq!(grade_id(spec, e19), want_grade, "{}", spec.part);
        }
        assert_eq!(lookup("M5045").map(|s| s.grade), Some("N50M"), "the workbook row");
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e18_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
#[test]
fn e19_supermagnetman_arcs_follow_the_vendor_grid() {
    // Decision 2 A: the vendor's 60 C on all three arcs moves every rating-calibrated demag
    // cell by the rating change (the calibration offset absorbs it), and nothing else: 26
    // cells on the workbook basis (the E12 guard zeroes 6 of them on the M2 basis). The
    // temperature checks C107 and C108 stay "OK" at 50 C.
    let e19 = DeviationId::E19;
    for (part, stored) in [("M5044", 80.0), ("M5045", 100.0), ("M5026", 80.0)] {
        let rings = [
            ("coupling.magnets.part_inner", Value::Text(part.into())),
            ("coupling.magnets.part_outer", Value::Text(part.into())),
        ];
        for (basis, before, count) in [("workbook", Deviations::NONE, 26), ("M2", m2(), 20)] {
            let (b, a) = (cells_with(&rings, before), cells_with(&rings, before.with(e19)));
            let changed = changed_cells(&b, &a);
            assert_eq!(changed.len(), count, "{part} {basis}: {changed:?}");
            assert_eq!(b["Calculator!C22"], Value::Num(stored), "{part}");
            assert_eq!(a["Calculator!C22"], Value::Num(60.0), "{part}");
            let shift = num(&b["Temperature design!C12"]) - num(&a["Temperature design!C12"]);
            assert!((shift - (stored - 60.0)).abs() < 1e-9, "{part} {basis}: C12 shift {shift}");
            for cell in ["Calculator!C107", "Calculator!C108"] {
                assert_eq!(a[cell], Value::Text("OK".into()), "{part} {cell}");
            }
        }
    }
    // M5045 maps to the grid's N50 (read by E20); its Br stays the workbook's 1.42 T.
    let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
    inputs.coupling.magnets.part_inner = "M5045".into();
    let off = compute_all_with(&inputs, Deviations::NONE).model;
    let on = compute_all_with(&inputs, Deviations::only(e19)).model;
    assert_eq!((off.inner_grade.as_str(), on.inner_grade.as_str()), ("N50M", "N50"));
    assert_eq!((off.inner_br_T, on.inner_br_T), (1.42, 1.42));
}

#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
    // The default part is B842SH: the vendor grid applies to the arcs only.
    assert_bit_for_bit_at_defaults(DeviationId::E19);
}

#[test]
fn e18_leaves_every_default_cell_bit_for_bit() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/grades.rs`, replace:

```rust
use magcoupling::engine::constants::NDFEB_DENSITY_G_MM3;
use magcoupling::engine::grades::{GRADES, Grade, GradeFamily, N42SH, grade};
use magcoupling::engine::library::{MAGNET_LIBRARY, N42SH_BR_CORRECTED_T};
```

with:

```rust
use magcoupling::engine::constants::NDFEB_DENSITY_G_MM3;
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::grades::{GRADES, Grade, GradeFamily, N42SH, grade};
use magcoupling::engine::library::{MAGNET_LIBRARY, N42SH_BR_CORRECTED_T, grade_id};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/grades.rs`, replace:

```rust
#[test]
fn library_parts_agree_with_their_grade_except_where_registered() {
```

with:

```rust
#[test]
fn each_grade_lists_the_parts_that_resolve_to_it() {
    // The data file's `library_parts` is decision 2 A (E19 on: M5045 is N50);
    // `library_parts_under_decision_2_B_or_C` is the workbook's own mapping (E19 off).
    let doc = data();
    for g in &GRADES {
        for (dev, key) in [
            (Deviations::ALL, "library_parts"),
            (Deviations::NONE, "library_parts_under_decision_2_B_or_C"),
        ] {
            let listed = &doc["grades"][g.id][key];
            let listed = if listed.is_null() {
                &doc["grades"][g.id]["library_parts"]
            } else {
                listed
            };
            let want: Vec<&str> = listed
                .as_array()
                .expect("a part list")
                .iter()
                .map(|p| p.as_str().expect("a part"))
                .collect();
            let got: Vec<&str> = MAGNET_LIBRARY
                .iter()
                .filter(|spec| grade_id(spec, dev) == g.id)
                .map(|spec| spec.part)
                .collect();
            assert_eq!(got, want, "{} {key}", g.id);
        }
    }
}

#[test]
fn library_parts_agree_with_their_grade_except_where_registered() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations e19 2>&1 | grep -E "panicked|test result" -A2 | head -8
```

Expected: `e19_supermagnetman_arcs_follow_the_vendor_grid` FAILS (`M5044 workbook`: no cell changes yet).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test grades 2>&1 | grep -E "^error" | head -2
```

Expected: a compile error, ``unresolved import `magcoupling::engine::library::grade_id` ``.

- [ ] **Step 3: Implement E19**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
/// K&J's plating on every library block (product pages, 2026-09-30).
```

with:

```rust
/// The maximum operating temperature the Calculator uses for a library part: the
/// row's workbook value, or with E19 on the vendor's own statement where it differs
/// (the SuperMagnetMan arcs: 60 °C, decision 2 A).
#[allow(non_snake_case)]
pub fn tmax_C(spec: &MagnetSpec, dev: Deviations) -> f64 {
    match spec.vendor_tmax_C {
        Some(vendor) if dev.is_on(DeviationId::E19) => vendor,
        _ => spec.tmax_C,
    }
}

/// The grade of a library part: the row's workbook grade, or with E19 on the
/// vendor's specification grid where it contradicts it (M5045: N50, decision 2 A).
pub fn grade_id(spec: &MagnetSpec, dev: Deviations) -> &'static str {
    match spec.vendor_grade {
        Some(vendor) if dev.is_on(DeviationId::E19) => vendor,
        _ => spec.grade,
    }
}

/// K&J's plating on every library block (product pages, 2026-09-30).
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
    /// Magnetization direction, as the vendor states it.
    pub magnetization: &'static str,
}
```

with:

```rust
    /// Magnetization direction, as the vendor states it.
    pub magnetization: &'static str,
    /// E19: the vendor's maximum working temperature where it differs from the
    /// workbook row (read only through [`tmax_C`]).
    pub vendor_tmax_C: Option<f64>,
    /// E19: the vendor grid's grade where it contradicts the workbook row
    /// (read only through [`grade_id`]).
    pub vendor_grade: Option<&'static str>,
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
        page: "",
        coating: "",
        magnetization: "",
    }
}
```

with:

```rust
        page: "",
        coating: "",
        magnetization: "",
        vendor_tmax_C: None,
        vendor_grade: None,
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
        Self {
            page,
            coating,
            magnetization,
            ..self
        }
    }
}
```

with:

```rust
        Self {
            page,
            coating,
            magnetization,
            ..self
        }
    }

    /// E19: what the vendor's page states where it differs from the workbook row.
    #[allow(non_snake_case)]
    const fn vendor_states(self, tmax_C: f64, grade: Option<&'static str>) -> Self {
        Self {
            vendor_tmax_C: Some(tmax_C),
            vendor_grade: grade,
            ..self
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
    .listed(
        "https://supermagnetman.com/products/m5044",
        SMM_COATING,
        SMM_MAGNETIZATION,
    ),
```

with:

```rust
    .listed(
        "https://supermagnetman.com/products/m5044",
        SMM_COATING,
        SMM_MAGNETIZATION,
    )
    .vendor_states(60.0, None),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
    .listed(
        "https://supermagnetman.com/products/m5045",
        SMM_COATING,
        SMM_MAGNETIZATION,
    ),
```

with:

```rust
    .listed(
        "https://supermagnetman.com/products/m5045",
        SMM_COATING,
        SMM_MAGNETIZATION,
    )
    .vendor_states(60.0, Some("N50")),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/library.rs`, replace:

```rust
    .listed(
        "https://supermagnetman.com/products/m5026",
        SMM_COATING,
        SMM_MAGNETIZATION,
    ),
```

with:

```rust
    .listed(
        "https://supermagnetman.com/products/m5026",
        SMM_COATING,
        SMM_MAGNETIZATION,
    )
    .vendor_states(60.0, None),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                    br_T: library::br_T(spec, dev),
                    tmax_C: NumOrText::Num(spec.tmax_C),
                    grade: grades::grade(spec.grade),
```

with:

```rust
                    br_T: library::br_T(spec, dev),
                    tmax_C: NumOrText::Num(library::tmax_C(spec, dev)),
                    grade: grades::grade(library::grade_id(spec, dev)),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence).
#[allow(non_snake_case)]
pub fn resolve_magnets(
```

with:

```rust
/// A library part's Br comes from [`library::br_T`], which applies E3 (the N42SH remanence),
/// its rating and grade from [`library::tmax_C`] and [`library::grade_id`], which apply E19.
#[allow(non_snake_case)]
pub fn resolve_magnets(
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        approval: Approval::Addendum { decisions: &[2] },
        depends_on: &[],
        status: DeviationStatus::Planned,
```

with:

```rust
        approval: Approval::Addendum { decisions: &[2] },
        depends_on: &[],
        status: DeviationStatus::Applied,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            the title's N50M is the unsafe reading of a self-contradicting page). Br stays at the workbook's 1.42 T.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
```

with:

```rust
            the title's N50M is the unsafe reading of a self-contradicting page). Br stays at the workbook's 1.42 T.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[
            Probe {
                label: "M5044 on both rings (stored 80 C, vendor grid 60 C)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("M5044")),
                    ("coupling.magnets.part_outer", Literal::Text("M5044")),
                ],
                expect: &[
                    CellChange {
                        cell: "Calculator!C22",
                        workbook: Literal::Num(80.0),
                        corrected: Literal::Num(60.0),
                    },
                    CellChange {
                        cell: "Calculator!C32",
                        workbook: Literal::Num(80.0),
                        corrected: Literal::Num(60.0),
                    },
                    CellChange {
                        cell: "Temperature design!C50",
                        workbook: Literal::Num(73.7958582391511),
                        corrected: Literal::Num(93.7958582391511),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(29.18110148932614),
                        corrected: Literal::Num(9.181101489326139),
                    },
                ],
            },
            Probe {
                label: "M5045 on both rings (stored 100 C, vendor grid 60 C)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("M5045")),
                    ("coupling.magnets.part_outer", Literal::Text("M5045")),
                ],
                expect: &[
                    CellChange {
                        cell: "Calculator!C22",
                        workbook: Literal::Num(100.0),
                        corrected: Literal::Num(60.0),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(49.18110148932614),
                        corrected: Literal::Num(9.181101489326139),
                    },
                ],
            },
        ],
    },
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`: unit tests 97 passed, `deviations.rs` 48 passed, `grades.rs` 9 passed.

- [ ] **Step 4: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| E18 | Temperature design!C104, C105, C201; verdicts C106, C202 | 4140's expansion coefficient and modulus (C94, C98) for a hub the mass model makes aluminium | at C6 ≠ 1 (E15's hub gate) the screen uses 6061-T6, 23.6e-6 /°C and 68.9 GPa (Alliance datasheet); C94 and C98 still show the steel inputs. M2 basis at back iron 0: C104 11.68 → 20.76 MPa, C106 and C202 read "Above ..." | Addendum A row E18 (decision 16) |
```

with:

```markdown
| E18 | Temperature design!C104, C105, C201; verdicts C106, C202 | 4140's expansion coefficient and modulus (C94, C98) for a hub the mass model makes aluminium | at C6 ≠ 1 (E15's hub gate) the screen uses 6061-T6, 23.6e-6 /°C and 68.9 GPa (Alliance datasheet); C94 and C98 still show the steel inputs. M2 basis at back iron 0: C104 11.68 → 20.76 MPa, C106 and C202 read "Above ..." | Addendum A row E18 (decision 16) |
| E19 | Calculator!C22, C32 → the rating-calibrated demag cells (C47, C50, C56-C61, C7-C10, C12, C13, C15, C24, C181, C182) | M5044, M5045, M5026 rated 80, 100, 80 °C; M5045 graded N50M | the vendor's specification grid: 60 °C for all three, M5045 graded N50; Br stays 1.42 T. M5044 on both rings: C12 29.18 → 9.18 °C | Addendum A decision 2 |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br |
```

with:

```markdown
and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
the exact-text lookup; br_T applies E3 with grades::N42SH.br_T)
```

with:

```yaml
the exact-text lookup; br_T applies E3 with grades::N42SH.br_T; tmax_C and grade_id apply E19, the vendor grid of the SuperMagnetMan arcs)
```

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task9.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task9.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task9.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/library.rs magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/deviations.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/tests/grades.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
fix(magcoupling-rs): apply E19 SuperMagnetMan arcs per the vendor grid

The vendor's 60 C on M5044, M5045 and M5026, and N50 for M5045 (decision 2).
Br stays 1.42 T. 26 cells move per arc on the workbook basis.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 10: E20: each magnet's own coercivity, with the positive-beta (ferrite) branch

**Model:** session model (omit `model`; a `risk: physics` correction and new physics). Physics reviewer: decisions 3, 17, 18, 19, report sections 2.5, 3 and 6.3, and this plan's Decisions to confirm A2, A3 and A13.

Decision 19 A: gated so that NONE keeps the single input; probes on B842 at both rings (report 6.3: C12 23.06 → -3.52 C) and a
ferrite cold case through the custom-dimension mode; the positive-beta branch included; C44/C45 stay as overrides that win when
set, made explicit by the Rust-only `coercivity_source` (Decisions to confirm, A2). The positive-beta semantics are A3: with beta > 0
the knee is reached on cooling, never on heating, so the hot onsets are +inf, the hot limit is the grade's rating (no calibration
offset: the rating is not a cold-side knee rating), and the cold side is new Rust-only results. The ferrite probe sets alpha(Br) to
Y30's -0.20 %/C (the calculator keeps one alpha, A4) and scales the four stored reverse fields by 0.37/1.29 (the fields scale with Br;
M3 computes them live). The workbook's demag block reads only the inner ring's Br and rating against C52-C55, which are the OUTER blocks'
reverse fields (report 6.3: the choice becomes visible once each ring has a grade). E20 checks each ring with its own grade, Br and
rating; the ring with the lower magnet limit governs and the block (C42, C47-C61, the Hcj and beta used) shows it whole, the inner ring
on a tie, so identical rings are bit for bit the inner ring's block; the cold side shows the ring with the higher cold limit and passes
only if both rings pass (Decisions to confirm, A13). The B842 and ferrite probes use one part on both rings; a third probe (B842SH inside
B842) pins A13. The C44 and C45 labels stay (schema parity); their help now names E20 and the registry records the workbook text. With
a positive beta and no rating (a manual magnet, source 0) there is no hot limit: C60 = +inf (the adhesive governs C12) and C61 = NaN.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs` (Rust-only `coercivity_source`; links `inner_grade` and the outer ring's Br, rating and grade; `ring_demag` (one ring's check; the weaker ring governs, A13); the grade's Hcj and beta; `cold_onset_C`; the cold side; the C44 and C45 help; unit tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs` (fill `inner_grade` and the outer ring's links)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs` (E20 Applied: the recorded C44 and C45 help, the B842, ferrite and mixed-ring probes)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs` (all 15 parts against report 6.3; the override; the ferrite cold case; unscaled ferrite; mixed rings)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/robustness.rs` (a coercivity source outside its choices; a positive beta with no rating)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/data/input_schema.json` (the selector)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Differences table)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (the temperature line)

**Interfaces:**
- Consumes: `grades::Grade` (Task 3), `model::ModelResults::{inner_grade, outer_grade, outer_br_T, outer_tmax_C}` (Task 4), `library::grade_id` (E19, Task 9), `TemperatureLinks::min_temp_C`.
- Produces:
  - input `temperature.demag.coercivity_source: i64 = 1` (Rust-only; 1 each magnet's grade, 0 C44 and C45 for both rings; a two-way choice: any code but 1 falls through to 0);
  - `TemperatureLinks { .., inner_grade: Option<&'static Grade>, outer_br20_T: f64, outer_tmax_lib_C: NumOrText, outer_grade: Option<&'static Grade> }` (the outer ring's Calculator C31, C32 and grade);
  - `pub fn temperature::cold_onset_C(h_rev_kA_m: f64, hcj20: f64, beta: f64, knee: f64, alpha_br: f64) -> f64` and `pub const NO_COLD_ONSET: &str = "n/a"`; `pub const temperature::{RING_INNER, RING_OUTER}` (`"inner"`, `"outer"`); private `ring_demag` (one ring's check) and `higher_cold_limit`;
  - Rust-only `DemagResults` fields `hcj20_used_kA_m: f64, beta_used_per_C: f64, demag_ring: String, cold_onset_aligned_C, cold_onset_pullout_C, cold_onset_skipping_C, cold_onset_single_ring_C, cold_limit_C: NumOrText, cold_ring: String, cold_check: String`;
  - with E20 on, a positive beta with no rating gives C60 = +inf and C61 = NaN;
  - the verdict C25 also requires the cold check (always true for beta <= 0).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
/// The demagnetization results for `overrides` under `dev`.
fn demag_with(
    overrides: &[(&str, Value)],
    dev: Deviations,
) -> magcoupling::engine::temperature::DemagResults {
    let mut inputs = DesignInputs::defaults_with(dev);
    for (path, value) in overrides {
        inputs.set(path, value.clone()).expect("a valid input");
    }
    compute_all_with(&inputs, dev).temperature.demag
}

fn both_rings(part: &str) -> [(&'static str, Value); 2] {
    [
        ("coupling.magnets.part_inner", Value::Text(part.into())),
        ("coupling.magnets.part_outer", Value::Text(part.into())),
    ]
}

#[test]
fn e20_each_part_uses_its_own_coercivity() {
    // Report 6.3, governing limit C12 with the part's own Hcj and beta (E20 alone: the
    // stored ratings, M5045 as the workbook's N50M). The N42SH parts do not move: the grade
    // keeps the workbook's 1592 kA/m and -0.005 /C (decisions 17 A, 18 A).
    let e20 = DeviationId::E20;
    for (part, workbook, corrected) in [
        ("B842", 23.06, -3.52),
        ("B822", 23.06, -3.52),
        ("B862", 23.06, -3.52),
        ("B882", 23.06, -3.52),
        ("B861", 23.06, -3.52),
        ("B881", 23.06, -3.52),
        ("B442", 23.06, -3.52),
        ("B842-N52", 30.73, 0.17),
        ("B882-N52", 30.73, 0.17),
        ("M5044", 29.18, -2.50),
        ("M5045", 49.18, 42.50),
        ("M5026", 29.18, -2.50),
        ("B842SH", 92.55, 92.55),
        ("BX042SH", 92.55, 92.55),
        ("BX082SH", 92.55, 92.55),
    ] {
        let rings = both_rings(part);
        let (b, a) = (
            cells_with(&rings, Deviations::NONE),
            cells_with(&rings, Deviations::only(e20)),
        );
        let limit = |c: &BTreeMap<String, Value>| num(&c["Temperature design!C12"]);
        assert!((limit(&b) - workbook).abs() <= 0.005, "{part}: {}", limit(&b));
        assert!((limit(&a) - corrected).abs() <= 0.005, "{part}: {}", limit(&a));
    }
    // B842 changes 24 cells (report 6.3) and shows the N42 grade's Hcj and beta.
    let rings = both_rings("B842");
    let changed = changed_cells(
        &cells_with(&rings, Deviations::NONE),
        &cells_with(&rings, Deviations::only(e20)),
    );
    assert_eq!(changed.len(), 24, "{changed:?}");
    let demag = demag_with(&rings, Deviations::only(e20));
    assert_eq!((demag.hcj20_used_kA_m, demag.beta_used_per_C), (954.9, -0.0062));
    let workbook = demag_with(&rings, Deviations::NONE);
    assert_eq!((workbook.hcj20_used_kA_m, workbook.beta_used_per_C), (1592.0, -0.005));
    // With E19 too, M5045 is N50 (the vendor grid): Hcj 875.4, beta -0.62 %/C.
    let m5045 = demag_with(&both_rings("M5045"), Deviations::ALL);
    assert_eq!((m5045.hcj20_used_kA_m, m5045.beta_used_per_C), (875.4, -0.0062));
    let n50m = demag_with(&both_rings("M5045"), Deviations::only(e20));
    assert_eq!((n50m.hcj20_used_kA_m, n50m.beta_used_per_C), (1114.1, -0.00675));
}

#[test]
fn e20_the_hcj_and_beta_inputs_override_the_grade_when_selected() {
    // Decision 19: C44 and C45 stay as overrides that win when set: the coercivity source 0.
    let mut rings = both_rings("B842").to_vec();
    rings.push(("temperature.demag.coercivity_source", Value::Int(0)));
    let e20 = Deviations::only(DeviationId::E20);
    assert!(changed_cells(&cells_with(&rings, Deviations::NONE), &cells_with(&rings, e20)).is_empty());
    rings.push(("temperature.demag.hcj20_kA_m", Value::Num(954.9)));
    rings.push(("temperature.demag.beta_hcj_per_C", Value::Num(-0.0062)));
    let typed = cells_with(&rings, e20);
    let graded = cells_with(&both_rings("B842"), e20);
    assert_eq!(
        typed["Temperature design!C12"], graded["Temperature design!C12"],
        "the grade's values typed into C44 and C45 give the same limit"
    );
    // Manual magnets without a grade always use the inputs (both rings manual: a library
    // ring beside a manual one is checked with its own grade and rating, A13).
    let manual = both_rings("");
    assert!(changed_cells(&cells_with(&manual, Deviations::NONE), &cells_with(&manual, e20)).is_empty());
}

#[test]
fn e20_ferrite_is_limited_on_the_cold_side() {
    // Spec, Addendum testing: "the demag check uses the part's own Hcj(T), including a
    // ferrite cold-case test", through the custom-dimension mode (no library part is ferrite).
    let e20 = &REGISTRY[DeviationId::E20.index()];
    let ferrite = &e20.probes[1];
    let overrides: Vec<(&str, Value)> = ferrite
        .inputs
        .iter()
        .map(|&(path, value)| (path, value.to_value()))
        .collect();
    let on = demag_with(&overrides, Deviations::only(DeviationId::E20));
    let off = demag_with(&overrides, Deviations::NONE);
    // The workbook takes |beta| and reports hot onsets near 240 C: "OK" for a magnet that
    // demagnetizes on every like-pole pass below about 100 C.
    assert!(off.onset_skipping_C > 200.0 && off.cold_check.starts_with("n/a"));
    assert_eq!((on.hcj20_used_kA_m, on.beta_used_per_C), (180.0, 0.0035));
    assert_eq!(on.onset_skipping_C, f64::INFINITY, "no knee on heating");
    assert_eq!(on.calibration_offset_C, 0.0, "the rating is not a cold-side knee rating");
    assert_eq!(on.magnet_limit_C, 250.0, "the hot limit is the grade's rating");
    let num_of = |x: NumOrText| match x {
        NumOrText::Num(v) => v,
        NumOrText::Text(t) => panic!("expected a number, got {t:?}"),
    };
    // The aligned field is below the knee at 20 C: its cold onset lies below 20 C. The
    // skipping field is past it: the magnet survives only above about 101 C.
    let aligned = num_of(on.cold_onset_aligned_C);
    let skipping = num_of(on.cold_onset_skipping_C);
    assert!(aligned < 20.0 && (aligned - (-57.82)).abs() < 0.005, "{aligned}");
    assert!((skipping - 100.90).abs() < 0.005, "{skipping}");
    assert_eq!(num_of(on.cold_limit_C), skipping + 10.0);
    assert_eq!(on.cold_check, "Below the cold demagnetization limit");
}

#[test]
fn e20_mixed_rings_use_the_weaker_grade() {
    // A-1 plan decision A13: C52 to C55 are the outer blocks' reverse fields, and the workbook
    // checks only the inner ring's Br and rating against them. With E20 each ring is checked
    // with its own grade, Br and rating and the weaker ring governs, on either side: B842 (N42)
    // beside B842SH (N42SH) gives B842's limit in both orders, and the block shows B842.
    use magcoupling::engine::temperature::{RING_INNER, RING_OUTER};
    let b842 = cells_with(&both_rings("B842"), Deviations::ALL);
    for (inner, outer, governing) in [("B842SH", "B842", RING_OUTER), ("B842", "B842SH", RING_INNER)] {
        let rings = [
            ("coupling.magnets.part_inner", Value::Text(inner.into())),
            ("coupling.magnets.part_outer", Value::Text(outer.into())),
        ];
        let cells = cells_with(&rings, Deviations::ALL);
        let limit = &cells["Temperature design!C12"];
        assert_report("Temperature design!C12", limit, -3.52, 0.005);
        assert_eq!(limit, &b842["Temperature design!C12"], "{inner}/{outer}");
        assert_eq!(
            cells["Temperature design!C25"],
            Value::Text("CHECK: see the rows above.".into()),
            "{inner}/{outer}"
        );
        let demag = demag_with(&rings, Deviations::ALL);
        assert_eq!(demag.demag_ring, governing, "{inner}/{outer}");
        assert_eq!((demag.hcj20_used_kA_m, demag.beta_used_per_C), (954.9, -0.0062));
        assert_eq!(demag.tmax_lib_C, NumOrText::Num(80.0), "the block shows B842");
    }
    // Without E20 the workbook reads the inner ring only: B842SH inside reads OK.
    let stronger_inside = [
        ("coupling.magnets.part_inner", Value::Text("B842SH".into())),
        ("coupling.magnets.part_outer", Value::Text("B842".into())),
    ];
    let workbook = cells_with(&stronger_inside, Deviations::NONE);
    assert_report("Temperature design!C12", &workbook["Temperature design!C12"], 92.55, 0.005);
    assert_eq!(demag_with(&stronger_inside, Deviations::NONE).demag_ring, RING_INNER);
    // Identical rings tie: the inner ring's block, bit for bit (the default design).
    assert_eq!(demag_with(&both_rings("B842"), Deviations::ALL).demag_ring, RING_INNER);
}

#[test]
fn e20_leaves_every_default_cell_bit_for_bit() {
    // The default part's grade N42SH carries the workbook's own Hcj and beta.
    assert_bit_for_bit_at_defaults(DeviationId::E20);
}

#[test]
fn e19_leaves_every_default_cell_bit_for_bit() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    #[test]
    fn e13_changes_only_the_rotations_per_degree() {
```

with:

```rust
    /// The fixture with hard-ferrite magnets (Y30) on both rings and fields below their knee.
    fn ferrite_links() -> TemperatureLinks {
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
    }

    #[test]
    fn e20_checks_both_rings_each_side_from_the_weaker() {
        // A-1 plan decision A13: an NdFeB inner ring (N42SH) with a ferrite outer ring (Y30).
        // The hot side reads the NdFeB ring (the ferrite has no knee on heating), the cold side
        // the ferrite ring (NdFeB has none), and the verdict needs both. Without E20 only the
        // inner ring is read.
        let e20 = Deviations::only(DeviationId::E20);
        let ti = TemperatureInputs::default();
        let mixed = with_outer_of(links(), &ferrite_links());
        let mut ferrite = ferrite_links();
        ferrite.alpha_br = mixed.alpha_br; // the calculator's one alpha (A4)
        let r = compute(&ti, &mixed, e20);
        assert_eq!(
            (r.demag.demag_ring.as_str(), r.demag.cold_ring.as_str()),
            (RING_INNER, RING_OUTER)
        );
        assert_eq!(r.demag, {
            let mut want = compute(&ti, &links(), e20).demag;
            let cold = compute(&ti, &ferrite, e20).demag;
            want.cold_onset_aligned_C = cold.cold_onset_aligned_C;
            want.cold_onset_pullout_C = cold.cold_onset_pullout_C;
            want.cold_onset_skipping_C = cold.cold_onset_skipping_C;
            want.cold_onset_single_ring_C = cold.cold_onset_single_ring_C;
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
            (RING_OUTER, RING_INNER)
        );
        assert_eq!(s.demag.magnet_limit_C, r.demag.magnet_limit_C);
        assert_eq!(s.demag.cold_limit_C, r.demag.cold_limit_C);
        assert_eq!(run(&ti, &mixed), run(&ti, &links()), "the workbook reads the inner ring only");
    }

    #[test]
    fn cold_onset_is_where_the_knee_meets_the_reverse_field() {
        // Hk (1 + beta dT) = H (1 + alpha dT) at the cold onset, and the magnet is past the
        // knee below it (beta > 0: the knee falls as it cools).
        let (hcj, beta, knee, alpha) = (180.0, 0.0035, 0.9, -0.002);
        for h in [50.0, 102.0, 160.0, 248.0] {
            let t = cold_onset_C(h, hcj, beta, knee, alpha);
            let hk_at = |temp: f64| knee * hcj * (1.0 + beta * (temp - 20.0));
            let h_at = |temp: f64| h * (1.0 + alpha * (temp - 20.0));
            assert!((hk_at(t) - h_at(t)).abs() <= 1e-9 * h, "{h}: {t}");
            assert!(hk_at(t - 1.0) < h_at(t - 1.0) && hk_at(t + 1.0) > h_at(t + 1.0), "{h}");
            assert_eq!(t < 20.0, h < knee * hcj, "{h}: below 20 C exactly when under the knee");
        }
    }

    #[test]
    fn cold_check_passes_at_equality() {
        // `min_temp >= cold_limit`: the cold onset does not read the minimum temperature, so
        // one run supplies it and a second run puts the check at exact equality.
        let e20 = Deviations::only(DeviationId::E20);
        let mut ti = TemperatureInputs::default();
        ti.demag.h_rev_aligned_kA_m = 90.0;
        ti.demag.h_rev_pullout_kA_m = 120.0;
        ti.demag.h_rev_likepole_kA_m = 140.0;
        ti.demag.h_rev_single_ring_kA_m = 110.0;
        let mut k = ferrite_links();
        let limit = match compute(&ti, &k, e20).demag.cold_limit_C {
            NumOrText::Num(x) => x,
            NumOrText::Text(t) => panic!("{t}"),
        };
        k.min_temp_C = limit;
        let r = compute(&ti, &k, e20);
        assert_eq!(r.demag.cold_limit_C, NumOrText::Num(k.min_temp_C));
        assert_eq!(r.demag.cold_check, "OK");
        k.min_temp_C = limit - 0.001;
        assert_eq!(
            compute(&ti, &k, e20).demag.cold_check,
            "Below the cold demagnetization limit"
        );
        assert_eq!(compute(&ti, &k, e20).summary.verdict, VERDICT_CHECK);
    }

    #[test]
    fn a_negative_beta_never_takes_the_cold_side() {
        // NdFeB and SmCo (beta < 0) keep the workbook form under E20; the cold side is "n/a".
        let e20 = Deviations::only(DeviationId::E20);
        let ti = TemperatureInputs::default();
        let k = links(); // N42SH: the workbook's own 1592 kA/m and -0.005 /C
        let r = compute(&ti, &k, e20);
        assert_eq!(r, run(&ti, &k), "the default grade is default-neutral");
        assert_eq!(r.demag.cold_limit_C, NumOrText::Text(NO_COLD_ONSET));
        assert_eq!(r.demag.cold_check, "n/a (coercivity rises as the magnet cools)");
        // Without E20 a positive beta typed into C45 (set() takes it) keeps the workbook's |beta|.
        let mut typed = TemperatureInputs::default();
        typed.demag.coercivity_source = 0;
        typed.demag.beta_hcj_per_C = 0.005;
        let workbook = run(&typed, &k);
        assert!(workbook.demag.onset_skipping_C.is_finite());
        assert_eq!(workbook.demag.cold_check, "n/a (coercivity rises as the magnet cools)");
        // With E20 the same typed beta takes the cold side.
        let corrected = compute(&typed, &k, e20);
        assert_eq!(corrected.demag.onset_skipping_C, f64::INFINITY);
        assert!(matches!(corrected.demag.cold_onset_skipping_C, NumOrText::Num(_)));
    }

    #[test]
    fn e13_changes_only_the_rotations_per_degree() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/robustness.rs`, replace:

```rust
#[test]
fn compute_all_is_cheap_enough_to_run_every_frame() {
```

with:

```rust
#[test]
fn an_invalid_coercivity_source_uses_the_inputs() {
    // A Rust-only selector set on the struct (bypassing set()): any code but 1 means the Hcj
    // and beta inputs for both rings, as the catch-all else of a two-way IF (Global
    // Constraints); validate() names it.
    let mut inputs = DesignInputs::default();
    inputs.coupling.magnets.part_inner = "B842".into();
    let graded = compute_all(&inputs).temperature.demag;
    inputs.temperature.demag.coercivity_source = 7;
    let res = compute_all(&inputs).temperature.demag;
    assert_eq!(graded.hcj20_used_kA_m, 954.9);
    assert_eq!(res.hcj20_used_kA_m, inputs.temperature.demag.hcj20_kA_m);
    let errors = inputs.validate().expect_err("an invalid code");
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].path, "temperature.demag.coercivity_source");
    assert_eq!(errors[0].kind, SetErrorKind::NotAChoice { code: 7 });
}

#[test]
fn a_positive_beta_without_a_rating_has_no_hot_limit() {
    // E20: a positive beta typed into C45 (coercivity source 0) for manual magnets without a
    // grade: no knee on heating and no rating, so no magnet limit (+inf: the adhesive governs
    // C12) and no torque at it (NaN, where the workbook formula would give +inf).
    let mut inputs = DesignInputs::default();
    inputs.coupling.magnets.part_inner = String::new();
    inputs.coupling.magnets.part_outer = String::new();
    inputs.temperature.demag.coercivity_source = 0;
    inputs.temperature.demag.beta_hcj_per_C = 0.0035;
    let t = compute_all(&inputs).temperature;
    assert_eq!(t.demag.magnet_limit_C, f64::INFINITY);
    assert!(t.demag.torque_at_limit_Nm.is_nan());
    assert_eq!(t.summary.governing_limit_C, t.summary.adhesive_limit_C);
    assert_eq!(t.summary.governing_note, "Adhesive governs.");
    assert_eq!(
        t.summary.verdict,
        "OK on temperature. Confirm drag torque and thermal cycling by test."
    );
}

#[test]
fn compute_all_is_cheap_enough_to_run_every_frame() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/deviations.rs`, replace:

```rust
#[test]
fn e20_leaves_every_default_cell_bit_for_bit() {
```

with:

```rust
#[test]
fn e20_ferrite_with_the_stored_ndfeb_fields_is_past_its_knee_at_room_temperature() {
    // Review Focus 5: a ferrite grade with the reverse fields left at the stored NdFeB
    // values (354 to 863 kA/m, all above Y30's 162 kA/m knee). The cold onsets then lie
    // ABOVE the operating temperature: the magnet is demagnetized wherever it runs, and the
    // cold check and the verdict say so; nothing panics or reads as a cold-weather margin.
    let ferrite = [
        ("coupling.magnets.part_inner", Value::Text(String::new())),
        ("coupling.magnets.part_outer", Value::Text(String::new())),
        ("coupling.magnets.grade_inner", Value::Text("Y30".into())),
        ("coupling.magnets.grade_outer", Value::Text("Y30".into())),
    ];
    let demag = demag_with(&ferrite, Deviations::ALL);
    let cold = |x: NumOrText| match x {
        NumOrText::Num(v) => v,
        NumOrText::Text(t) => panic!("{t}"),
    };
    for onset in [
        demag.cold_onset_aligned_C,
        demag.cold_onset_pullout_C,
        demag.cold_onset_skipping_C,
        demag.cold_onset_single_ring_C,
    ] {
        assert!(cold(onset) > 50.0, "{onset:?}");
    }
    assert_eq!(demag.cold_check, "Below the cold demagnetization limit");
    assert_eq!(
        cells_with(&ferrite, Deviations::ALL)["Temperature design!C25"],
        Value::Text("CHECK: see the rows above.".into())
    );
}

#[test]
fn e20_leaves_every_default_cell_bit_for_bit() {
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations e20 2>&1 | grep -E "^error" | head -2
```

Expected: compile errors, among them ``no field `hcj20_used_kA_m` ``.

- [ ] **Step 3: Implement E20**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
use super::compat::{py_max, py_min, text0};
use super::deviations::{DeviationId, Deviations};
use super::meta::{NumOrText, inputs, out, out_uncelled, param, param_rust_only, results};
```

with:

```rust
use super::compat::{py_max, py_min, text0};
use super::deviations::{DeviationId, Deviations};
use super::grades::Grade;
use super::meta::{
    NumOrText, inputs, out, out_rust_only, out_uncelled, param, param_rust_only, results,
};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            h_rev_single_ring_kA_m: f64 = 569.0 => param("kA/m", "3D worst reverse field, single ring on its carrier",
                "Adhesive-cure case (inner ring alone: 545).", "Temperature design!C55")
                .range(0.0, 1500.0, 1.0),
        }
    }
}
```

with:

```rust
            h_rev_single_ring_kA_m: f64 = 569.0 => param("kA/m", "3D worst reverse field, single ring on its carrier",
                "Adhesive-cure case (inner ring alone: 545).", "Temperature design!C55")
                .range(0.0, 1500.0, 1.0),
            coercivity_source: i64 = 1 => param_rust_only("-", "Coercivity for the demagnetization check",
                "Correction E20: 1 = each magnet's own grade (its library part, or the grade picked for manual dimensions); 0 = Hcj and beta above (C44, C45) for both rings, which then override the grades. A magnet without a grade always uses C44 and C45; a code other than 1 falls through to them, as the else of a two-way IF.")
                .choices(&[(1, "magnet grade"), (0, "Hcj and beta inputs")]),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
                "N42SH ≥ 20 kOe.", "Temperature design!C44")
```

with:

```rust
                "N42SH ≥ 20 kOe. Correction E20: each magnet's grade supplies Hcj; this value is used for a magnet without a grade, or for both rings when the coercivity source is 0.", "Temperature design!C44")
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            beta_hcj_per_C: f64 = -0.005 => param("1/°C", "Hcj temperature coefficient (effective, 20–150 °C)",
                "", "Temperature design!C45")
```

with:

```rust
            beta_hcj_per_C: f64 = -0.005 => param("1/°C", "Hcj temperature coefficient (effective, 20–150 °C)",
                "Correction E20: each magnet's grade supplies beta; this value is used for a magnet without a grade, or for both rings when the coercivity source is 0.", "Temperature design!C45")
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    pub steel_sigma_S_m: f64,          // Materials C14
```

with:

```rust
    /// The inner magnet's grade (`model::ResolvedMagnet::grade`): E20 reads its Hcj and beta.
    pub inner_grade: Option<&'static Grade>,
    /// E20 (Decisions to confirm, A13): the outer ring's Br at 20 °C (Calculator C31), rating
    /// (C32) and grade, for its own demagnetization check.
    pub outer_br20_T: f64,
    pub outer_tmax_lib_C: NumOrText,
    pub outer_grade: Option<&'static Grade>,
    pub steel_sigma_S_m: f64,          // Materials C14
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            torque_at_service_Nm: f64 => out("N·m", "Pull-out torque at the service maximum", "", "Temperature design!C62"),
        }
    }
}
```

with:

```rust
            torque_at_service_Nm: f64 => out("N·m", "Pull-out torque at the service maximum", "", "Temperature design!C62"),
            hcj20_used_kA_m: f64 => out_rust_only("kA/m", "Hcj at 20 °C used",
                "Correction E20: the governing ring's grade, or C44 when that magnet has no grade or the source is set to the inputs."),
            beta_used_per_C: f64 => out_rust_only("1/°C", "Hcj temperature coefficient used",
                "Correction E20: the governing ring's grade's, or C45. Positive for hard ferrite: its coercivity falls as it cools."),
            demag_ring: String => out_rust_only("", "Ring the demagnetization block shows",
                "Correction E20: both rings are checked, each against its own grade, Br and rating; the ring with the lower magnet limit governs and the block shows it (the inner ring on a tie). Without E20 the workbook checks the inner ring only."),
            cold_onset_aligned_C: NumOrText => out_rust_only("°C", "Cold demag onset, rings aligned",
                "Correction E20, positive beta only: below this temperature the reverse field exceeds the knee. 'n/a' when coercivity rises as the magnet cools."),
            cold_onset_pullout_C: NumOrText => out_rust_only("°C", "Cold demag onset at pull-out", ""),
            cold_onset_skipping_C: NumOrText => out_rust_only("°C", "Cold demag onset while skipping", ""),
            cold_onset_single_ring_C: NumOrText => out_rust_only("°C", "Cold demag onset, single ring on its carrier", ""),
            cold_limit_C: NumOrText => out_rust_only("°C", "Cold magnet limit",
                "The skipping cold onset plus the design margin (C51): the minimum magnet temperature (Metal design C16) must not be below it."),
            cold_ring: String => out_rust_only("", "Ring the cold-side results show",
                "Correction E20: the ring with the higher cold limit (the inner ring when neither has one)."),
            cold_check: String => out_rust_only("", "Cold demagnetization check",
                "Against the minimum magnet temperature (Metal design C16); it passes only if both rings pass."),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
```

with:

```rust
/// E20, positive beta (hard ferrite): the temperature at which the reverse field
/// (scaling with Br, `alpha_br` signed) meets the knee (scaling with Hcj, `beta`
/// signed): Hk (1 + beta dT) = H (1 + alpha dT). With beta > 0 the knee falls as
/// the magnet cools, so the magnet demagnetizes BELOW this temperature: a cold
/// limit, never calibrated to the (hot) rating. For beta <= 0 it equals
/// [`demag_onset_C`] with no offset, but the engine keeps that form there (parity).
#[allow(non_snake_case)]
pub fn cold_onset_C(h_rev_kA_m: f64, hcj20: f64, beta: f64, knee: f64, alpha_br: f64) -> f64 {
    let hk = knee * hcj20;
    20.0 + (hk - h_rev_kA_m) / (h_rev_kA_m * alpha_br - hk * beta)
}

/// The text of a cold-side result when the coercivity rises as the magnet cools.
pub const NO_COLD_ONSET: &str = "n/a";

/// Which ring a demagnetization result shows (E20 checks both rings, Decisions to
/// confirm A13).
pub const RING_INNER: &str = "inner";
/// See [`RING_INNER`].
pub const RING_OUTER: &str = "outer";

/// One ring's demagnetization check: the workbook's block (C47 to C61) for a ring with
/// Br `br20_T`, rating `tmax_lib_C` and grade `grade`. The workbook checks the inner ring
/// only; E20 checks the outer ring too (Decisions to confirm, A13).
#[derive(Clone, Copy, Debug)]
#[allow(non_snake_case)] // unit suffixes, as the result names
struct RingDemag {
    br20_T: f64,
    tmax_lib_C: NumOrText,
    hcj20: f64,
    beta: f64,
    h_ref: f64,
    t_ref: f64,
    offset: f64,
    /// Aligned, pull-out, skipping, single ring (C56 to C59).
    onsets: [f64; 4],
    mag_lim: f64,
    torque_at_limit_Nm: f64,
    /// E20 cold side, in the order of `onsets`; "n/a" unless beta > 0.
    cold_onsets: [NumOrText; 4],
    cold_limit: NumOrText,
    cold_ok: bool,
}

#[allow(non_snake_case)] // Python names (T)
fn ring_demag(
    d: &DemagInputs,
    k: &TemperatureLinks,
    br20_T: f64,
    tmax_lib_C: NumOrText,
    grade: Option<&'static Grade>,
    dev: Deviations,
) -> RingDemag {
    // E20: the ring's own grade, unless the source is set to the inputs (C44, C45), which
    // then win; a magnet without a grade uses the inputs, as the workbook does. A code other
    // than 1 falls through to the inputs, as the else of a two-way IF.
    let (hcj20, beta) = match grade {
        Some(g) if dev.is_on(DeviationId::E20) && d.coercivity_source == 1 => {
            (g.hcj20_kA_m, g.beta_hcj_per_C)
        }
        _ => (d.hcj20_kA_m, d.beta_hcj_per_C),
    };
    // E20: with a positive beta (hard ferrite) the knee is reached on cooling, never on
    // heating: the hot onsets are +inf, the rating is the hot limit and the cold side below
    // decides. The workbook takes |beta| whatever its sign.
    let cold_side = dev.is_on(DeviationId::E20) && beta > 0.0;
    let h_ref = br20_T / (2.0 * k.mu0) / 1000.0;
    let t_ref = if cold_side {
        f64::INFINITY
    } else {
        demag_onset_C(h_ref, hcj20, beta, d.knee_fraction, k.alpha_br, 0.0)
    };
    // the workbook errors out when the magnet is not in the library; Python leaves the onset uncalibrated
    let offset = match tmax_lib_C {
        _ if cold_side => 0.0, // E20: the rating is not a knee rating on the cold side
        NumOrText::Num(tmax) => t_ref - tmax,
        NumOrText::Text(_) => 0.0,
    };
    let on = |h: f64| {
        if cold_side {
            f64::INFINITY
        } else {
            demag_onset_C(h, hcj20, beta, d.knee_fraction, k.alpha_br, offset)
        }
    };
    let onsets = [
        on(d.h_rev_aligned_kA_m),
        on(d.h_rev_pullout_kA_m),
        on(d.h_rev_likepole_kA_m),
        on(d.h_rev_single_ring_kA_m),
    ];
    let thf = |T: f64| br_factor(k.alpha_br, T).powi(2);
    // E20 cold side: the hot limit is the rating. A magnet with no rating has no hot limit
    // (+inf, so the adhesive governs C12) and no torque at it (NaN, where the workbook
    // formula would give +inf).
    let (mag_lim, torque_at_limit_Nm) = if cold_side {
        match tmax_lib_C {
            NumOrText::Num(tmax) => (tmax, k.pullout_20C_Nm * thf(tmax)),
            NumOrText::Text(_) => (f64::INFINITY, f64::NAN),
        }
    } else {
        let lim = onsets[2] - d.design_margin_C;
        (lim, k.pullout_20C_Nm * thf(lim))
    };
    // E20 cold side: the skipping case (the largest reverse field) governs, as on the hot side.
    let cold = |h: f64| {
        if cold_side {
            NumOrText::Num(cold_onset_C(h, hcj20, beta, d.knee_fraction, k.alpha_br))
        } else {
            NumOrText::Text(NO_COLD_ONSET)
        }
    };
    let cold_onsets = [
        cold(d.h_rev_aligned_kA_m),
        cold(d.h_rev_pullout_kA_m),
        cold(d.h_rev_likepole_kA_m),
        cold(d.h_rev_single_ring_kA_m),
    ];
    let cold_limit = match cold_onsets[2] {
        NumOrText::Num(c) => NumOrText::Num(c + d.design_margin_C),
        NumOrText::Text(_) => NumOrText::Text(NO_COLD_ONSET),
    };
    let cold_ok = match cold_limit {
        NumOrText::Num(limit) => k.min_temp_C >= limit,
        NumOrText::Text(_) => true,
    };
    RingDemag {
        br20_T,
        tmax_lib_C,
        hcj20,
        beta,
        h_ref,
        t_ref,
        offset,
        onsets,
        mag_lim,
        torque_at_limit_Nm,
        cold_onsets,
        cold_limit,
        cold_ok,
    }
}

/// E20: whether cold limit `a` is higher than `b` (a number is higher than "n/a").
fn higher_cold_limit(a: NumOrText, b: NumOrText) -> bool {
    match (a, b) {
        (NumOrText::Num(x), NumOrText::Num(y)) => x > y,
        (NumOrText::Num(_), NumOrText::Text(_)) => true,
        (NumOrText::Text(_), _) => false,
    }
}

/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    // ---- demagnetization
    let d = &ti.demag;
    let h_ref = k.br20_T / (2.0 * k.mu0) / 1000.0;
    let t_ref = demag_onset_C(
        h_ref,
        d.hcj20_kA_m,
        d.beta_hcj_per_C,
        d.knee_fraction,
        k.alpha_br,
        0.0,
    );
    // the workbook errors out when the magnet is not in the library; Python leaves the onset uncalibrated
    let offset = match k.tmax_lib_C {
        NumOrText::Num(tmax) => t_ref - tmax,
        NumOrText::Text(_) => 0.0,
    };
    let on = |h: f64| {
        demag_onset_C(
            h,
            d.hcj20_kA_m,
            d.beta_hcj_per_C,
            d.knee_fraction,
            k.alpha_br,
            offset,
        )
    };
    let (on_al, on_po, on_lp, on_cu) = (
        on(d.h_rev_aligned_kA_m),
        on(d.h_rev_pullout_kA_m),
        on(d.h_rev_likepole_kA_m),
        on(d.h_rev_single_ring_kA_m),
    );
    let mag_lim = on_lp - d.design_margin_C;
    let thf = |T: f64| br_factor(k.alpha_br, T).powi(2);
    let demag = DemagResults {
        br20_T: k.br20_T,
        alpha_br: k.alpha_br,
        tmax_lib_C: k.tmax_lib_C,
        h_ref_kA_m: h_ref,
        t_ref_model_C: t_ref,
        calibration_offset_C: offset,
        onset_aligned_C: on_al,
        onset_pullout_C: on_po,
        onset_skipping_C: on_lp,
        onset_single_ring_C: on_cu,
        magnet_limit_C: mag_lim,
        torque_at_limit_Nm: k.pullout_20C_Nm * thf(mag_lim),
        torque_at_service_Nm: k.pullout_op_Nm,
    };
```

with:

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
    };
    let (cold, cold_ring) = match outer {
        Some(o) if higher_cold_limit(o.cold_limit, inner.cold_limit) => (o, RING_OUTER),
        _ => (inner, RING_INNER),
    };
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
        calibration_offset_C: hot.offset,
        onset_aligned_C: on_al,
        onset_pullout_C: on_po,
        onset_skipping_C: on_lp,
        onset_single_ring_C: on_cu,
        magnet_limit_C: mag_lim,
        torque_at_limit_Nm: hot.torque_at_limit_Nm,
        torque_at_service_Nm: k.pullout_op_Nm,
        hcj20_used_kA_m: hot.hcj20,
        beta_used_per_C: hot.beta,
        demag_ring: hot_ring.to_owned(),
        cold_onset_aligned_C: cold.cold_onsets[0],
        cold_onset_pullout_C: cold.cold_onsets[1],
        cold_onset_skipping_C: cold.cold_onsets[2],
        cold_onset_single_ring_C: cold.cold_onsets[3],
        cold_limit_C: cold.cold_limit,
        cold_ring: cold_ring.to_owned(),
        cold_check: match cold.cold_limit {
            NumOrText::Text(_) => "n/a (coercivity rises as the magnet cools)",
            NumOrText::Num(_) if cold_ok => "OK",
            NumOrText::Num(_) => "Below the cold demagnetization limit",
        }
        .to_owned(),
    };
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let ok = margin_hot > 0.0
        && (mag_lim - peak) > 0.0
        && (sel.design_limit_C - peak) > 0.0
        && cure_margin >= 10.0;
```

with:

```rust
    let ok = margin_hot > 0.0
        && (mag_lim - peak) > 0.0
        && (sel.design_limit_C - peak) > 0.0
        && cure_margin >= 10.0
        && cold_ok; // E20: always true unless the coercivity falls on cooling
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            al6061_sigma_S_m: 2.5e7,
            cup_aluminium: false,
            hub_aluminium: false,
        }
    }
```

with:

```rust
            al6061_sigma_S_m: 2.5e7,
            cup_aluminium: false,
            hub_aluminium: false,
            inner_grade: crate::engine::grades::grade("N42SH"),
            outer_br20_T: 1.29,
            outer_tmax_lib_C: NumOrText::Num(150.0),
            outer_grade: crate::engine::grades::grade("N42SH"),
        }
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        endplates_g: ret.endplates_g,
        steel_sigma_S_m: mat_in.steel.conductivity_S_m,
```

with:

```rust
        endplates_g: ret.endplates_g,
        inner_grade: grades::grade(&m.inner_grade),
        outer_br20_T: m.outer_br_T,
        outer_tmax_lib_C: m.outer_tmax_C,
        outer_grade: grades::grade(&m.outer_grade),
        steel_sigma_S_m: mat_in.steel.conductivity_S_m,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
use super::deviations::Deviations;
```

with:

```rust
use super::deviations::Deviations;
use super::grades;
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
        approval: Approval::Addendum { decisions: &[19] },
        depends_on: &[],
        status: DeviationStatus::Planned,
```

with:

```rust
        approval: Approval::Addendum { decisions: &[19] },
        depends_on: &[],
        status: DeviationStatus::Applied,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/deviations.rs`, replace:

```rust
            N42SH keeps the workbook's 1592 kA/m and -0.005 /C (decisions 17 A, 18 A), so the default design does not move.",
        workbook_input_defaults: &[],
        workbook_help: &[],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[],
    },
```

with:

```rust
            N42SH keeps the workbook's 1592 kA/m and -0.005 /C (decisions 17 A, 18 A), so the default design does not move.",
        workbook_input_defaults: &[],
        workbook_help: &[
            ("temperature.demag.hcj20_kA_m", "N42SH ≥ 20 kOe."),
            ("temperature.demag.beta_hcj_per_C", ""),
        ],
        changes_at_defaults: &[],
        changes_file: None,
        probes: &[
            Probe {
                label: "B842 (N42: Hcj 954.9 kA/m, beta -0.62 %/C) on both rings (report 6.3: C12 23.06 -> -3.52 C)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("B842")),
                    ("coupling.magnets.part_outer", Literal::Text("B842")),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C50",
                        workbook: Literal::Num(79.92129544736366),
                        corrected: Literal::Num(12.681126305465526),
                    },
                    CellChange {
                        cell: "Temperature design!C58",
                        workbook: Literal::Num(33.05566428111358),
                        corrected: Literal::Num(6.482578384916511),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(23.05566428111358),
                        corrected: Literal::Num(-3.5174216150834887),
                    },
                ],
            },
            Probe {
                label: "hard ferrite Y30 for manual dimensions on both rings, alpha(Br) -0.20 %/C, the four \
                    reverse fields scaled from the stored NdFeB values by 0.37 / 1.29 (the M3 placeholder): \
                    the knee is reached on cooling",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("")),
                    ("coupling.magnets.part_outer", Literal::Text("")),
                    ("coupling.magnets.grade_inner", Literal::Text("Y30")),
                    ("coupling.magnets.grade_outer", Literal::Text("Y30")),
                    ("calibration.alpha_br_per_C", Literal::Num(-0.002)),
                    ("temperature.demag.h_rev_aligned_kA_m", Literal::Num(102.0)),
                    ("temperature.demag.h_rev_pullout_kA_m", Literal::Num(227.0)),
                    ("temperature.demag.h_rev_likepole_kA_m", Literal::Num(248.0)),
                    ("temperature.demag.h_rev_single_ring_kA_m", Literal::Num(163.0)),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C58",
                        workbook: Literal::Num(240.54277641656867),
                        corrected: Literal::Num(f64::INFINITY),
                    },
                    CellChange {
                        cell: "Temperature design!C60",
                        workbook: Literal::Num(230.54277641656867),
                        corrected: Literal::Num(250.0),
                    },
                    CellChange {
                        cell: "Temperature design!C25",
                        workbook: Literal::Text(
                            "OK on temperature. Confirm drag torque and thermal cycling by test.",
                        ),
                        corrected: Literal::Text("CHECK: see the rows above."),
                    },
                ],
            },
            Probe {
                label: "B842SH inner (N42SH), B842 outer (N42): the outer ring governs (A-1 plan decision A13)",
                inputs: &[
                    ("coupling.magnets.part_inner", Literal::Text("B842SH")),
                    ("coupling.magnets.part_outer", Literal::Text("B842")),
                ],
                expect: &[
                    CellChange {
                        cell: "Temperature design!C42",
                        workbook: Literal::Num(1.29),
                        corrected: Literal::Num(1.3),
                    },
                    CellChange {
                        cell: "Temperature design!C47",
                        workbook: Literal::Num(150.0),
                        corrected: Literal::Num(80.0),
                    },
                    CellChange {
                        cell: "Temperature design!C12",
                        workbook: Literal::Num(92.55004986453575),
                        corrected: Literal::Num(-3.5174216150834887),
                    },
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
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
//! the low-Reynolds closed form T1 with the Rust-only free-space fields) and
//! E18 (the mismatch screen C104-C106, C201, C202 bonds to an aluminium hub when
//! C6 is not 1).
```

with:

```rust
//! the low-Reynolds closed form T1 with the Rust-only free-space fields), E18
//! (the mismatch screen C104-C106, C201, C202 bonds to an aluminium hub when C6
//! is not 1) and E20 (the demagnetization block checks each ring against its
//! own grade, Br and rating, and shows the ring with the lower limit; a positive
//! beta is limited on the cold side, [`cold_onset_C`]).
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

- [ ] **Step 4: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test schema
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: the bless run adds `temperature.demag.coercivity_source` (`"rust_only": true`, choices 1 and 0); every binary `ok`: unit tests 101 passed, `deviations.rs` 54 passed, `robustness.rs` 9 passed.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|panicked"
```

Expected: no output.

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 5: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| E19 | Calculator!C22, C32 → the rating-calibrated demag cells (C47, C50, C56-C61, C7-C10, C12, C13, C15, C24, C181, C182) | M5044, M5045, M5026 rated 80, 100, 80 °C; M5045 graded N50M | the vendor's specification grid: 60 °C for all three, M5045 graded N50; Br stays 1.42 T. M5044 on both rings: C12 29.18 → 9.18 °C | Addendum A decision 2 |
```

with:

```markdown
| E19 | Calculator!C22, C32 → the rating-calibrated demag cells (C47, C50, C56-C61, C7-C10, C12, C13, C15, C24, C181, C182) | M5044, M5045, M5026 rated 80, 100, 80 °C; M5045 graded N50M | the vendor's specification grid: 60 °C for all three, M5045 graded N50; Br stays 1.42 T. M5044 on both rings: C12 29.18 → 9.18 °C | Addendum A decision 2 |
| E20 | Temperature design!C44, C45 → C42, C47-C61, C7-C10, C12 and the margins | one N42SH coercivity curve (C44 1592 kA/m, C45 −0.5 %/°C) for every magnet, and only the inner ring's Br and rating against the outer blocks' reverse fields | each ring's own grade (library part, or the grade picked for manual dimensions), Br and rating, unless `temperature.demag.coercivity_source` = 0 picks C44 and C45; the ring with the lower magnet limit governs and the block shows it (`temperature.demag.demag_ring`); a positive beta (ferrite) has no knee on heating: hot onsets +inf, the grade's rating is the hot limit, and Rust-only cold onsets, cold limit and cold check (against Metal design C16, both rings) feed the verdict. A positive beta with no rating has no hot limit: C60 = +inf (the adhesive governs C12) and C61 = NaN, which exporters must handle as they handle E13's +inf. B842 on both rings: C12 23.06 → −3.52 °C; B842SH inside B842: 92.55 → −3.52 °C. N42SH keeps 1592 and −0.005 (decisions 17, 18), so defaults do not move | Addendum A decision 19 |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
TemperatureLinks; compute, TemperatureResults 130 result cells and the uncelled adhesive name; demag_onset_C, volkersen_peak_shear_MPa)
```

with:

```yaml
TemperatureLinks; compute, TemperatureResults 130 result cells and the uncelled adhesive name; demag_onset_C, cold_onset_C (E20, positive beta), ring_demag (E20: each ring's check, the weaker governs, A13), volkersen_peak_shear_MPa; Rust-only coercivity_source and the free-space fields of E17; Rust-only demag results hcj20_used/beta_used, demag_ring, cold_ring and the cold side)
```

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task10.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task10.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task10.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 7: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/src/engine/deviations.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/tests/robustness.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
fix(magcoupling-rs): apply E20 each magnet's own coercivity

Each ring is checked with its own grade (Hcj, beta), Br and rating unless
the Rust-only coercivity source picks C44 and C45 (decision 19); the ring
with the lower limit governs (A13). A positive beta (ferrite) is limited on
the cold side. B842: C12 23.06 -> -3.52 C; N42SH is default-neutral.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 11: A5 materials library

**Model:** `sonnet` (data transcription with the exact code given). Escalate as in Task 1.

Decisions 5-7 and 20-26 A: the library carries the sourced values as reference; `Material::engine` is what the engine uses when
the material is picked: the workbook's number where the data file records one (4140: 4.5e6 S/m, 473, 12.3e-6, 205, 7.85; 316L: 1.35e6,
500, 8.0; 6061: 2.5e7, 900, 2.7; 7075: 1.9e7), else the sourced value. `the_default_materials_are_the_workbook_inputs` proves each
default material's engine values are bit-equal to the inputs they stand for, which is why picking the default changes nothing (Task 12).
Decision 6 A asks for the 416 expansion coefficient to be re-sourced from a standard-416 sheet; the data file kept Zapp's 10.5 as a
placeholder. Rolled Alloys' 416 sheet prints 5.6e-6 in/in/F from 70 to 212 F (21 to 100 C, the range every other entry uses):
10.08e-6 /K, cited in the record; Smiths prints 9.9 with no range. Decision 7 A: 1018 and 12L14 keep the higher conductivity (flagged).

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/material_library.rs` (14 materials, cited per value; engine values; each part's choices)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/material_library.rs` (every value and citation against the data file; default materials equal the inputs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/mod.rs` (`pub mod material_library;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Layout and Tests tables)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (engine_files, ported, tests)

**Interfaces:**
- Consumes: `materials::{Steel4140, AL6061}`, `metal_design::MetalDesignInputs`, `temperature::{SlipLossInputs, ThermalInputs}` (defaults only, in the tests).
- Produces:
  - `material_library::{Sourced, EngineProps, Material, MATERIALS: [Material; 14], material(id) -> Option<&'static Material>}`; `Material` fields `id, label, name, condition, ferromagnetic, ferromagnetic_source, mu_r, bsat_T, sigma_S_m, density_g_cm3, cte_1e6_per_K, modulus_GPa, yield_MPa, cp_J_kgK` (each `Sourced { value: Option<f64>, source: Option<&'static str> }`), `design_flux_density_T: Option<f64>, engine: EngineProps, needs_plating: bool, notes`;
  - `EngineProps { sigma_S_m, density_g_mm3, cp_J_kgK, cte_per_C, modulus_GPa }` (engine-unit literals);
  - `BACK_IRON_CHOICES: [(i64, &str); 8]`, `SLEEVE_LINER_CHOICES: [(i64, &str); 4]`, `CAP_HOUSING_CHOICES: [(i64, &str); 3]` (code 1 = the workbook's material), `chosen(choices, code) -> Option<&'static Material>`.

- [ ] **Step 1: Write the failing tests**

Create `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/material_library.rs` with exactly this content:

```rust
//! The A5 materials library against the approved Addendum A data file
//! (`docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`): every value
//! and citation, the engine values (workbook numbers where the workbook has
//! them, decisions 21 to 26), and the default materials equal to the inputs.

mod common;

use common::{read_json, repo_path};
use magcoupling::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, MATERIALS, SLEEVE_LINER_CHOICES, Sourced, material,
};
use magcoupling::engine::materials::{AL6061, Steel4140};
use magcoupling::engine::metal_design::MetalDesignInputs;
use magcoupling::engine::temperature::{SlipLossInputs, ThermalInputs};

const DATA: &str = "docs/analyses/2026-09-30-magcoupling-addendum-a-data.json";
/// Decision 6 A re-sources the 416 expansion coefficient (the data file keeps a placeholder).
const RA_416: &str = "https://www.rolledalloys.com/wp-content/uploads/416_stainless-steel-data-sheet-rolled-alloys.pdf";

fn data_materials() -> Vec<serde_json::Value> {
    read_json(&repo_path(DATA))["materials"]
        .as_array()
        .expect("a materials array")
        .clone()
}

/// The data file's selected value and source of a field, as a [`Sourced`] holds them.
fn assert_sourced(what: &str, rust: Sourced, field: &serde_json::Value) {
    let value = field["value"].as_f64();
    let source = if value.is_some() {
        field["source_url"].as_str()
    } else {
        None
    };
    assert_eq!((rust.value, rust.source), (value, source), "{what}");
}

/// Engine value = the workbook default the data file records, else the sourced value.
fn engine_expected(m: &serde_json::Value, key: &str) -> f64 {
    m["workbook_defaults"][key]["default"]
        .as_f64()
        .or_else(|| m["fields"][key]["value"].as_f64())
        .unwrap_or_else(|| panic!("{} {key}", m["id"]))
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-15 * want.abs()
}

#[test]
fn materials_equal_the_addendum_data_file() {
    let data = data_materials();
    let ids: Vec<&str> = data
        .iter()
        .map(|m| m["id"].as_str().expect("an id"))
        .collect();
    let rust: Vec<&str> = MATERIALS.iter().map(|m| m.id).collect();
    assert_eq!(rust, ids, "ids and order");
    for (m, j) in MATERIALS.iter().zip(&data) {
        let id = m.id;
        let f = &j["fields"];
        assert_eq!(m.name, j["name"].as_str().expect("a name"), "{id}");
        assert_eq!(
            m.ferromagnetic,
            j["ferromagnetic"]["value"].as_bool().expect("a bool"),
            "{id}"
        );
        assert_eq!(
            Some(m.ferromagnetic_source),
            j["ferromagnetic"]["source_url"].as_str(),
            "{id}"
        );
        for (what, rust, key) in [
            ("mu_r", m.mu_r, "mu_r"),
            ("bsat", m.bsat_T, "bsat_T"),
            ("sigma", m.sigma_S_m, "sigma_S_m"),
            ("density", m.density_g_cm3, "density_g_cm3"),
            ("E", m.modulus_GPa, "E_GPa"),
            ("yield", m.yield_MPa, "yield_MPa"),
            ("cp", m.cp_J_kgK, "cp_J_kgK"),
        ] {
            assert_sourced(&format!("{id} {what}"), rust, &f[key]);
        }
        if id == "416_annealed" {
            assert_eq!(
                f["cte_1e-6_per_K"]["value"].as_f64(),
                Some(10.5),
                "the placeholder"
            );
            assert_eq!(
                m.cte_1e6_per_K,
                Sourced {
                    value: Some(10.08),
                    source: Some(RA_416)
                }
            );
        } else {
            assert_sourced(&format!("{id} cte"), m.cte_1e6_per_K, &f["cte_1e-6_per_K"]);
        }
        assert_eq!(
            m.design_flux_density_T,
            j["design_flux_density_T"]["value"].as_f64(),
            "{id} design flux density"
        );
        // Engine values: the workbook's number where one exists, else the sourced value.
        let e = &m.engine;
        assert_eq!(
            e.sigma_S_m,
            engine_expected(j, "sigma_S_m"),
            "{id} engine sigma"
        );
        assert_eq!(e.cp_J_kgK, engine_expected(j, "cp_J_kgK"), "{id} engine cp");
        assert_eq!(e.modulus_GPa, engine_expected(j, "E_GPa"), "{id} engine E");
        assert!(
            close(
                e.density_g_mm3,
                engine_expected(j, "density_g_cm3") / 1000.0
            ),
            "{id} engine density {}",
            e.density_g_mm3
        );
        let cte = if id == "416_annealed" {
            10.08
        } else {
            engine_expected(j, "cte_1e-6_per_K")
        };
        assert!(
            close(e.cte_per_C, cte * 1e-6),
            "{id} engine cte {}",
            e.cte_per_C
        );
    }
}

#[test]
fn the_default_materials_are_the_workbook_inputs() {
    // Picking the default material of a part changes nothing: its engine values are
    // bit-equal to the inputs the engine reads (decisions 21 to 26).
    let steel = Steel4140::default();
    let m4140 = material("4140_annealed").expect("4140").engine;
    assert_eq!(m4140.sigma_S_m, steel.conductivity_S_m);
    assert_eq!(m4140.cp_J_kgK, steel.specific_heat_J_kgK);
    assert_eq!(m4140.cte_per_C, steel.cte_per_C);
    assert_eq!(m4140.modulus_GPa, steel.modulus_GPa);
    let md = MetalDesignInputs::default();
    assert_eq!(m4140.density_g_mm3, md.steel_density_g_mm3);
    assert_eq!(
        material("4140_annealed").and_then(|m| m.design_flux_density_T),
        Some(steel.bsat_T)
    );
    let (sl, th) = (SlipLossInputs::default(), ThermalInputs::default());
    let m316 = material("316L_annealed").expect("316L").engine;
    assert_eq!(m316.sigma_S_m, sl.sigma_316_S_m);
    assert_eq!(m316.cp_J_kgK, th.c_316);
    assert_eq!(m316.density_g_mm3, md.sleeve_density_g_mm3);
    let m6061 = material("6061_T6").expect("6061").engine;
    assert_eq!(m6061.sigma_S_m, AL6061.conductivity_S_m);
    assert_eq!(m6061.cp_J_kgK, th.c_aluminium);
    assert_eq!(m6061.density_g_mm3, md.al_density_g_mm3);
    for (choices, default) in [
        (&BACK_IRON_CHOICES[..], "4140_annealed"),
        (&SLEEVE_LINER_CHOICES[..], "316L_annealed"),
        (&CAP_HOUSING_CHOICES[..], "6061_T6"),
    ] {
        assert_eq!(choices[0], (1, default));
    }
}

#[test]
fn each_part_offers_the_spec_choices_with_their_roles() {
    // Spec A5 choices; the data file's roles agree.
    let data = data_materials();
    let roles = |id: &str| -> String {
        data.iter()
            .find(|m| m["id"] == id)
            .map(|m| m["roles"].to_string())
            .unwrap_or_default()
    };
    for (choices, role) in [
        (&BACK_IRON_CHOICES[..], "back_iron"),
        (&SLEEVE_LINER_CHOICES[..], "sleeve_liner"),
        (&CAP_HOUSING_CHOICES[..], "cap_housing"),
    ] {
        for &(_, id) in choices {
            assert!(roles(id).contains(role), "{id} is not a {role} material");
        }
    }
    let non_magnetic: Vec<&str> = BACK_IRON_CHOICES
        .iter()
        .filter_map(|&(_, id)| material(id))
        .filter(|m| !m.ferromagnetic)
        .map(|m| m.id)
        .collect();
    assert_eq!(
        non_magnetic,
        ["304_annealed", "6061_T6"],
        "the demonstration back irons"
    );
    let sleeves_ferromagnetic = SLEEVE_LINER_CHOICES
        .iter()
        .filter_map(|&(_, id)| material(id))
        .any(|m| m.ferromagnetic);
    assert!(!sleeves_ferromagnetic, "no listed sleeve is ferromagnetic");
}

#[test]
fn plain_and_low_alloy_steels_need_plating() {
    // The data file's warning input: 4140, 1018 and 12L14; stainless and non-ferrous do not.
    let plated: Vec<&str> = MATERIALS
        .iter()
        .filter(|m| m.needs_plating)
        .map(|m| m.id)
        .collect();
    assert_eq!(
        plated,
        ["4140_annealed", "1018_hot_rolled", "12L14_cold_drawn"]
    );
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test material_library 2>&1 | grep -E "^error" | head -2
```

Expected: a compile error, ``unresolved import `magcoupling::engine::material_library` ``.

- [ ] **Step 3: Create the library**

Create `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/material_library.rs` with exactly this content:

```rust
//! Materials per part (spec Addendum A5): the library, the choices each part
//! offers, and (with the selectors) what a choice feeds into the engine.
//!
//! Data: `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`, key
//! `"materials"`, approved with the Addendum A verification (decisions 5 to 7
//! and 20 to 26, option A; resolutions M1 to M24 in its section 4.2). Each
//! property keeps the data file's selected value and its source in a
//! [`Sourced`]; `tests/material_library.rs` compares every value and citation
//! with the data file. One value differs from the data file on purpose: the 416
//! expansion coefficient, re-sourced from a standard-416 sheet as decision 6 A
//! asks (Rolled Alloys, 5.6e-6 /F from 70 to 212 F = 10.08e-6 /K; the data file
//! kept Zapp's 10.5 as a placeholder).
//!
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sourced {
    pub value: Option<f64>,
    pub source: Option<&'static str>,
}

/// A sourced property.
const fn sourced(value: f64, source: &'static str) -> Sourced {
    Sourced {
        value: Some(value),
        source: Some(source),
    }
}

/// A property with no source (a null in the data file, with its reason there).
const NOT_SOURCED: Sourced = Sourced {
    value: None,
    source: None,
};

/// What the engine uses when a material is picked (engine units).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct EngineProps {
    /// Electrical conductivity [S/m]: slip losses.
    pub sigma_S_m: f64,
    /// Density [g/mm³]: mass and heat capacity.
    pub density_g_mm3: f64,
    /// Specific heat [J/(kg·K)]: heat capacity.
    pub cp_J_kgK: f64,
    /// Expansion coefficient [1/°C]: the bond-stress screen.
    pub cte_per_C: f64,
    /// Elastic modulus [GPa]: the bond-stress screen.
    pub modulus_GPa: f64,
}

/// One library material.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct Material {
    /// The data file's id.
    pub id: &'static str,
    /// The selector's text.
    pub label: &'static str,
    pub name: &'static str,
    pub condition: &'static str,
    /// Selects the steel circuit (as back iron) and short-circuits the gap (as sleeve).
    pub ferromagnetic: bool,
    pub ferromagnetic_source: &'static str,
    /// Relative permeability (reference only: secant, maximum or chart-read values;
    /// no source gives the incremental value at the magnet bias, so the engine keeps
    /// Materials C15 for every steel).
    pub mu_r: Sourced,
    /// Saturation flux density [T]: informational; drives the low-saturation warning.
    pub bsat_T: Sourced,
    pub sigma_S_m: Sourced,
    pub density_g_cm3: Sourced,
    /// Expansion coefficient [1e-6/K].
    pub cte_1e6_per_K: Sourced,
    pub modulus_GPa: Sourced,
    /// Yield strength [MPa] (published minimum where one exists; not read by the engine).
    pub yield_MPa: Sourced,
    pub cp_J_kgK: Sourced,
    /// The wall check's design flux density [T] (decision 20): workbook values only,
    /// 4140 1.5 T (Materials C13) and 1018 about 1.7 T (that cell's comment).
    pub design_flux_density_T: Option<f64>,
    pub engine: EngineProps,
    /// Plain or low-alloy steel: needs plating against corrosion.
    pub needs_plating: bool,
    /// The report's resolutions (M#) and flags.
    pub notes: &'static str,
}

/// Every library material, in the data file's order.
pub const MATERIALS: [Material; 14] = [
    Material {
        id: "4140_annealed",
        label: "4140 annealed",
        name: "AISI 4140 (EN 42CrMo4), annealed",
        condition: "Annealed (815-870 C), ~197 HB. Workbook plan: annealed 4140 with electroless nickel.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        mu_r: sourced(
            363.0,
            "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            4330000.0,
            "https://www.lucefin.com/wp-content/files_mf/152353604042CrMo4.pdf",
        ),
        density_g_cm3: sourced(7.85, "https://www.azom.com/article.aspx?ArticleID=6769"),
        cte_1e6_per_K: sourced(12.2, "https://www.azom.com/article.aspx?ArticleID=6769"),
        modulus_GPa: sourced(
            205.0,
            "https://otaisteel.com/aisi-4140-material-data-sheet/",
        ),
        yield_MPa: sourced(
            415.0,
            "https://otaisteel.com/aisi-4140-material-data-sheet/",
        ),
        cp_J_kgK: sourced(
            461.0,
            "https://www.lucefin.com/wp-content/files_mf/152353604042CrMo4.pdf",
        ),
        design_flux_density_T: Some(1.5),
        engine: EngineProps {
            sigma_S_m: 4500000.0,
            density_g_mm3: 0.00785,
            cp_J_kgK: 473.0,
            cte_per_C: 12.3e-6,
            modulus_GPa: 205.0,
        },
        needs_plating: true,
        notes: "Default back iron (the Materials sheet's inputs). mu_r 363 is the secant value at 1.5 T, not the incremental value the skin depth needs (decision 22). M13.",
    },
    Material {
        id: "1018_hot_rolled",
        label: "1018 hot rolled",
        name: "AISI 1018 (UNS G10180)",
        condition: "Hot rolled bar for yield; annealed for resistivity, CTE and specific heat (as the sources state). Cold drawn yield is 370 MPa.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            5848000.0,
            "https://www.azom.com/article.aspx?ArticleID=6115",
        ),
        density_g_cm3: sourced(7.87, "https://www.azom.com/article.aspx?ArticleID=6115"),
        cte_1e6_per_K: sourced(
            12.0,
            "https://www.theworldmaterial.com/astm-sae-aisi-1018-carbon-steel/",
        ),
        modulus_GPa: sourced(205.0, "https://www.azom.com/article.aspx?ArticleID=6115"),
        yield_MPa: sourced(
            220.0,
            "https://www.theworldmaterial.com/astm-sae-aisi-1018-carbon-steel/",
        ),
        cp_J_kgK: sourced(
            486.0,
            "https://www.theworldmaterial.com/astm-sae-aisi-1018-carbon-steel/",
        ),
        design_flux_density_T: Some(1.7),
        engine: EngineProps {
            sigma_S_m: 5848000.0,
            density_g_mm3: 0.00787,
            cp_J_kgK: 486.0,
            cte_per_C: 12.0e-6,
            modulus_GPa: 205.0,
        },
        needs_plating: true,
        notes: "Aggregator sources only; sigma derived at 20 C, flagged +-30 % (decision 7 A; 7 % IACS alternate 4.06e6). Design flux 1.7 T: the workbook's comment on Materials!C13.",
    },
    Material {
        id: "12L14_cold_drawn",
        label: "12L14 cold drawn",
        name: "AISI 12L14 (UNS G12144) resulfurized, leaded free-machining steel",
        condition: "Cold drawn bar (yield). Other properties: condition not stated by the sources.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.azom.com/article.aspx?ArticleID=6604",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(5747000.0, "https://www.theworldmaterial.com/12l14-steel/"),
        density_g_cm3: sourced(7.87, "https://www.azom.com/article.aspx?ArticleID=6604"),
        cte_1e6_per_K: sourced(11.5, "https://www.azom.com/article.aspx?ArticleID=6604"),
        modulus_GPa: sourced(200.0, "https://www.theworldmaterial.com/12l14-steel/"),
        yield_MPa: sourced(415.0, "https://www.azom.com/article.aspx?ArticleID=6604"),
        cp_J_kgK: sourced(472.0, "https://www.theworldmaterial.com/12l14-steel/"),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 5747000.0,
            density_g_mm3: 0.00787,
            cp_J_kgK: 472.0,
            cte_per_C: 11.5e-6,
            modulus_GPa: 200.0,
        },
        needs_plating: true,
        notes: "Aggregator sources only; sigma flagged +-30 % (decision 7 A; 7.1 % IACS alternate 4.12e6). Contains lead (RoHS/REACH). No sourced design flux density.",
    },
    Material {
        id: "416_annealed",
        label: "416 stainless, annealed",
        name: "Type 416 (UNS S41600) free-machining martensitic stainless",
        condition: "Annealed (Carpenter: anneal 650-760 C cool in air, ~187 HB). Hardened values listed as alternates.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.carpentertechnology.com/blog/magnetic-properties-of-stainless-steels",
        mu_r: sourced(
            110.0,
            "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        ),
        bsat_T: sourced(
            1.6,
            "https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf",
        ),
        sigma_S_m: sourced(
            1754000.0,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        density_g_cm3: sourced(
            7.64,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        cte_1e6_per_K: sourced(
            10.08,
            "https://www.rolledalloys.com/wp-content/uploads/416_stainless-steel-data-sheet-rolled-alloys.pdf",
        ),
        modulus_GPa: sourced(
            200.0,
            "https://www.smithmetal.com/pdf/stainless/416-stainless.pdf",
        ),
        yield_MPa: sourced(
            276.0,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        cp_J_kgK: sourced(
            460.5,
            "https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1754000.0,
            density_g_mm3: 0.00764,
            cp_J_kgK: 460.5,
            cte_per_C: 10.08e-6,
            modulus_GPa: 200.0,
        },
        needs_plating: false,
        notes: "Standard martensitic 416 (decision 6 A): Bsat 1.60 T is a proxy (SMAG Table 4, 410 at 0.15 % C); CTE 10.08e-6 re-sourced from Rolled Alloys (the sheet prints 'Coefficient of Thermal Expansion* 5.6 in/in F x 10-6' in its 212 F column, footnote '* 70F to indicated temperature': 5.6e-6 /F over 70 to 212 F = 10.08e-6 /K over 21 to 100 C); yield 276 is typical (no minimum published). mu_r 110 secant at 1.5 T.",
    },
    Material {
        id: "17-4PH_H1150",
        label: "17-4PH H1150",
        name: "17-4 PH (UNS S17400) precipitation-hardened stainless, condition H1150",
        condition: "H1150 (aged 4 h at 1150 F / 621 C, air cool). Chosen as the default because it is the stable, tough, lower-distortion condition; H900 is listed separately.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        mu_r: sourced(
            76.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1250000.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        density_g_cm3: sourced(
            7.82,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cte_1e6_per_K: sourced(
            11.9,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        modulus_GPa: sourced(
            199.9,
            "https://www.rolledalloys.com/wp-content/uploads/17-4_Data-sheet-rolled-alloys.pdf",
        ),
        yield_MPa: sourced(
            725.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cp_J_kgK: sourced(
            460.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1250000.0,
            density_g_mm3: 0.00782,
            cp_J_kgK: 460.0,
            cte_per_C: 11.9e-6,
            modulus_GPa: 199.9,
        },
        needs_plating: false,
        notes: "Low induction: ARMCO Fig. 3 gives about 1.07 T at 140 Oe (a lower bound on Bsat, not Bsat). sigma is ARMCO's H900 figure as a proxy (M24). mu_r chart-read.",
    },
    Material {
        id: "17-4PH_H900",
        label: "17-4PH H900",
        name: "17-4 PH (UNS S17400), condition H900",
        condition: "H900 (aged 1 h at 900 F / 482 C, air cool). Maximum strength, lowest toughness.",
        ferromagnetic: true,
        ferromagnetic_source: "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        mu_r: sourced(
            96.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1250000.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        density_g_cm3: sourced(
            7.8,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cte_1e6_per_K: sourced(
            10.8,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        modulus_GPa: sourced(
            199.9,
            "https://www.rolledalloys.com/wp-content/uploads/17-4_Data-sheet-rolled-alloys.pdf",
        ),
        yield_MPa: sourced(
            1170.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        cp_J_kgK: sourced(
            460.0,
            "https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1250000.0,
            density_g_mm3: 0.0078,
            cp_J_kgK: 460.0,
            cte_per_C: 10.8e-6,
            modulus_GPa: 199.9,
        },
        needs_plating: false,
        notes: "ARMCO Fig. 3 gives about 1.35 T at 140 Oe (a lower bound on Bsat). mu_r chart-read.",
    },
    Material {
        id: "304_annealed",
        label: "304 stainless (non-magnetic)",
        name: "AISI 304 (UNS S30400) austenitic stainless, annealed",
        condition: "Annealed sheet/strip (AK Steel data sheet).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.carpentertechnology.com/blog/magnetic-properties-of-stainless-steels",
        mu_r: sourced(
            1.02,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1389000.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        density_g_cm3: sourced(
            8.03,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        cte_1e6_per_K: sourced(
            16.9,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        modulus_GPa: sourced(
            193.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        yield_MPa: sourced(
            205.0,
            "https://www.sandmeyersteel.com/wp-content/uploads/Alloy304-304L-APR2013.pdf",
        ),
        cp_J_kgK: sourced(
            500.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1389000.0,
            density_g_mm3: 0.00803,
            cp_J_kgK: 500.0,
            cte_per_C: 16.9e-6,
            modulus_GPa: 193.0,
        },
        needs_plating: false,
        notes: "Non-magnetic demonstration back iron: an open magnetic circuit. mu_r 1.02 is AK Steel's upper limit at 200 Oe. M3 (yield minimum), M20.",
    },
    Material {
        id: "6061_T6",
        label: "6061-T6 aluminium",
        name: "Aluminium 6061-T6 (T651)",
        condition: "T6 temper (solution heat treated and artificially aged).",
        ferromagnetic: false,
        ferromagnetic_source: "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        mu_r: sourced(
            1.000022,
            "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            24940000.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        density_g_cm3: sourced(
            2.7,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        cte_1e6_per_K: sourced(
            23.6,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        modulus_GPa: sourced(
            68.3,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        yield_MPa: sourced(
            241.0,
            "https://www.aerometalsalliance.com/resources/data-sheets/view/Aluminium-Alloy-QQ-A-25011-T6-Sheet_200",
        ),
        cp_J_kgK: sourced(
            896.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 25000000.0,
            density_g_mm3: 0.0027,
            cp_J_kgK: 900.0,
            cte_per_C: 23.6e-6,
            modulus_GPa: 68.3,
        },
        needs_plating: false,
        notes: "Default cap and housing (the workbook's C42, C43 and C140). As a back iron: non-magnetic demonstration. mu_r: pure-aluminium proxy. M5.",
    },
    Material {
        id: "7075_T6",
        label: "7075-T6 aluminium",
        name: "Aluminium 7075-T6 (T651)",
        condition: "T6 temper (solution heat treated and artificially aged).",
        ferromagnetic: false,
        ferromagnetic_source: "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        mu_r: sourced(
            1.000022,
            "https://en.wikipedia.org/wiki/Permeability_(electromagnetism)",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            19140000.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        density_g_cm3: sourced(
            2.8,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        cte_1e6_per_K: sourced(
            23.4,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        modulus_GPa: sourced(
            71.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        yield_MPa: sourced(
            462.0,
            "https://www.aerometalsalliance.com/resources/data-sheets/view/Aluminium-Alloy-QQ-A-25012-T6-Sheet_203",
        ),
        cp_J_kgK: sourced(
            960.0,
            "https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 19000000.0,
            density_g_mm3: 0.0028,
            cp_J_kgK: 960.0,
            cte_per_C: 23.4e-6,
            modulus_GPa: 71.0,
        },
        needs_plating: false,
        notes: "The workbook uses 7075-T6 for clamp collars and adapters (clamps.alloy), which this choice does not change. M6.",
    },
    Material {
        id: "316L_annealed",
        label: "316L annealed",
        name: "316L (UNS S31603) austenitic stainless, annealed",
        condition: "Annealed sheet/strip/plate (AK Steel and Sandmeyer data sheets).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.sandmeyersteel.com/wp-content/uploads/316-316l-317l-spec-sheet.pdf",
        mu_r: sourced(
            1.02,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1351000.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        density_g_cm3: sourced(
            7.99,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        cte_1e6_per_K: sourced(
            16.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        modulus_GPa: sourced(
            193.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        yield_MPa: sourced(
            172.0,
            "https://www.sandmeyersteel.com/wp-content/uploads/316-316l-317l-spec-sheet.pdf",
        ),
        cp_J_kgK: sourced(
            500.0,
            "https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1350000.0,
            density_g_mm3: 0.008,
            cp_J_kgK: 500.0,
            cte_per_C: 16.0e-6,
            modulus_GPa: 193.0,
        },
        needs_plating: false,
        notes: "Default sleeve and liner (the workbook's C111, C139 and Metal design C44, which also sets the endplates). M4, M21.",
    },
    Material {
        id: "Ti6Al4V_annealed",
        label: "Ti-6Al-4V grade 5",
        name: "Titanium grade 5 (Ti-6Al-4V, UNS R56400), annealed",
        condition: "Annealed (700-785 C per ASM sheet).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        mu_r: sourced(
            1.00005,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            595200.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        density_g_cm3: sourced(
            4.42,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        cte_1e6_per_K: sourced(
            9.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        modulus_GPa: sourced(
            113.8,
            "https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf",
        ),
        yield_MPa: sourced(
            828.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        cp_J_kgK: sourced(
            580.0,
            "https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 595200.0,
            density_g_mm3: 0.00442,
            cp_J_kgK: 580.0,
            cte_per_C: 9.0e-6,
            modulus_GPa: 113.8,
        },
        needs_plating: false,
        notes: "About 0.44 times the conductivity of 316L. sigma at 0 C (4-6 % high at room temperature). M7, M22, M23.",
    },
    Material {
        id: "IN625_annealed",
        label: "Inconel 625",
        name: "INCONEL alloy 625 (UNS N06625), annealed",
        condition: "Annealed (resistivity: annealed 2100 F / 1 h; yield: annealed rod, bar, plate nominal range).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        mu_r: sourced(
            1.0006,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            775200.0,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        density_g_cm3: sourced(
            8.44,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        cte_1e6_per_K: sourced(
            12.8,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        modulus_GPa: sourced(
            207.5,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        yield_MPa: sourced(
            414.0,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        cp_J_kgK: sourced(
            410.0,
            "https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 775200.0,
            density_g_mm3: 0.00844,
            cp_J_kgK: 410.0,
            cte_per_C: 12.8e-6,
            modulus_GPa: 207.5,
        },
        needs_plating: false,
        notes: "About 0.57 times the conductivity of 316L. Yield: lower bound of the composite range.",
    },
    Material {
        id: "PEEK_unfilled",
        label: "PEEK",
        name: "PEEK, unfilled (Ensinger TECAPEEK natural stock shapes; Victrex 450G as cross-check)",
        condition: "Unfilled, natural/beige; stock-shape values at 23 C. Not a conductor.",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1e-13,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        density_g_cm3: sourced(
            1.31,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        cte_1e6_per_K: sourced(
            50.0,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        modulus_GPa: sourced(
            4.0,
            "https://www.victrex.com/-/media/downloads/datasheets/victrex_tds_450g.pdf",
        ),
        yield_MPa: sourced(
            98.0,
            "https://www.victrex.com/-/media/downloads/datasheets/victrex_tds_450g.pdf",
        ),
        cp_J_kgK: sourced(
            1100.0,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1e-13,
            density_g_mm3: 0.00131,
            cp_J_kgK: 1100.0,
            cte_per_C: 50.0e-6,
            modulus_GPa: 4.0,
        },
        needs_plating: false,
        notes: "Conductivity is an upper bound from the volume resistivity: zero for slip loss. CTE three times 316L. M14 to M17.",
    },
    Material {
        id: "POM_H_acetal",
        label: "Acetal (POM-H)",
        name: "Acetal, POM homopolymer (Delrin 150 series / TECAFORM AD natural)",
        condition: "HOMOPOLYMER (not copolymer), natural stock shape, 73 F (23 C).",
        ferromagnetic: false,
        ferromagnetic_source: "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        mu_r: NOT_SOURCED,
        bsat_T: NOT_SOURCED,
        sigma_S_m: sourced(
            1e-13,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        density_g_cm3: sourced(
            1.41,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        cte_1e6_per_K: sourced(
            122.4,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        modulus_GPa: sourced(3.1, "https://cdn.thomasnet.com/ccp/00072207/74788.pdf"),
        yield_MPa: sourced(
            75.84,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        cp_J_kgK: sourced(
            1465.0,
            "https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0",
        ),
        design_flux_density_T: None,
        engine: EngineProps {
            sigma_S_m: 1e-13,
            density_g_mm3: 0.00141,
            cp_J_kgK: 1465.0,
            cte_per_C: 122.4e-6,
            modulus_GPa: 3.1,
        },
        needs_plating: false,
        notes: "Homopolymer (Delrin 150). Conductivity is an upper bound: zero for slip loss. CTE about ten times aluminium. M9, M10.",
    },
];

/// The material with this data-file id (exact text).
pub fn material(id: &str) -> Option<&'static Material> {
    MATERIALS.iter().find(|m| m.id == id)
}

/// Back iron (hub, cup and boss) choices: selector code and material id. Code 1,
/// the workbook's 4140, is the default: the Materials sheet inputs.
pub const BACK_IRON_CHOICES: [(i64, &str); 8] = [
    (1, "4140_annealed"),
    (2, "1018_hot_rolled"),
    (3, "12L14_cold_drawn"),
    (4, "416_annealed"),
    (5, "17-4PH_H1150"),
    (6, "17-4PH_H900"),
    (7, "304_annealed"),
    (8, "6061_T6"),
];

/// Sleeve and liner choices (the endplates follow them, as Metal design C44 does).
/// Code 1, the workbook's 316L, is the default.
pub const SLEEVE_LINER_CHOICES: [(i64, &str); 4] = [
    (1, "316L_annealed"),
    (2, "Ti6Al4V_annealed"),
    (3, "IN625_annealed"),
    (4, "PEEK_unfilled"),
];

/// Cap and housing choices (the engine models the cap; the aluminium adapter and the
/// clamp alloy keep their own inputs). Code 1, the workbook's 6061-T6, is the default.
pub const CAP_HOUSING_CHOICES: [(i64, &str); 3] =
    [(1, "6061_T6"), (2, "7075_T6"), (3, "POM_H_acetal")];

/// The material a selector code picks among `choices`; `None` for a code outside them.
pub fn chosen(choices: &[(i64, &'static str)], code: i64) -> Option<&'static Material> {
    choices
        .iter()
        .find(|&&(c, _)| c == code)
        .and_then(|&(_, id)| material(id))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_choice_names_a_library_material_and_codes_start_at_one() {
        for choices in [
            &BACK_IRON_CHOICES[..],
            &SLEEVE_LINER_CHOICES[..],
            &CAP_HOUSING_CHOICES[..],
        ] {
            for (i, &(code, id)) in choices.iter().enumerate() {
                assert_eq!(code, i as i64 + 1, "{id}");
                assert!(material(id).is_some(), "{id}");
            }
            assert_eq!(chosen(choices, 0), None);
            assert_eq!(chosen(choices, choices.len() as i64 + 1), None);
        }
        assert_eq!(
            chosen(&BACK_IRON_CHOICES, 7).map(|m| m.ferromagnetic),
            Some(false)
        );
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/mod.rs`, replace:

```rust
pub mod library;
```

with:

```rust
pub mod library;
pub mod material_library;
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`: unit tests 102 passed, `material_library.rs` 4 passed.

- [ ] **Step 4: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/materials.rs` |
```

with:

```markdown
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `tests/static_data.rs` |
```

with:

```markdown
| `tests/material_library.rs` | The materials library equals `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` value for value and citation for citation (the 416 CTE re-sourced, decision 6); engine values are the workbook's where the data file records one, else the sourced ones; the default material of each part is bit-equal to the inputs it stands for; the choices match the spec's roles; plain and low-alloy steels need plating. |
| `tests/static_data.rs` |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    grades: grades.rs (
```

with:

```yaml
    material_library: material_library.rs (Addendum A5 materials library MATERIALS, 14 entries cited per value; EngineProps = workbook numbers where they exist (decisions 21-26), else sourced; BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, CAP_HOUSING_CHOICES, code 1 = the workbook's material)
    grades: grades.rs (
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
library.rs, grades.rs, model.rs
```

with:

```yaml
library.rs, grades.rs, material_library.rs, model.rs
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
static_data.rs, robustness.rs, grades.rs;
```

with:

```yaml
static_data.rs, robustness.rs, grades.rs, material_library.rs;
```

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task11.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task11.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task11.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/material_library.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/tests/material_library.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
feat(magcoupling-rs): A5 materials library

Fourteen materials from the approved Addendum A data file, every value cited
(decisions 5-7, 20-26); engine values keep the workbook's numbers where they
exist. The 416 CTE is re-sourced (Rolled Alloys, decision 6). Not wired yet.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 12: Per-part material selectors and the physics links

**Model:** session model (omit `model`; physics links across several modules, CLAUDE.md section 5). Physics reviewer: spec A5 (physics links), report sections 4.3 and 6.4, decisions 20-26, and this plan's Decisions to confirm A5-A7, A9, A11.

Spec A5 physics links, with the rule that keeps parity (report 6.1 point 1): the default choice of each part IS the workbook's material,
whose values are the existing inputs, so a default design (and every parity and differential case) is bit for bit unchanged; any other
choice supplies its library engine values in place of those inputs (A5). A ferromagnetic back iron keeps the steel circuit and supplies
conductivity, density, specific heat, expansion and modulus, and its design flux density where the library has one (1018: 1.7 T; else
Materials C13 stays, A9); a non-ferromagnetic one (304, 6061) selects the free-space circuit and becomes the hub, cup and boss
material, while the C6 input still overrides a ferromagnetic choice toward "no back iron" (A6, A7). Incremental permeability stays C15
(no source gives it at the magnet bias). Specific heat follows the material too (A11). A code outside the choices gives NaN and "#N/A"
(decision D3). `tests/material_links.rs` pins each link by equivalence: picking 1018 equals typing 1018's values into the inputs.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/material_library.rs` (`PartMaterial`, `PartProperties`, `resolve`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs` (Rust-only `PartMaterialInputs` group `materials.parts`; four Rust-only results; `compute` takes the parts)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs` (body links for E15, E17, E18)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs` (resolve the parts; feed the inputs in effect)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/material_links.rs` (picking equals typing; the circuit; the body; the override; the wall check; invalid codes)
- Modify (generated): `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/data/input_schema.json` (three selectors)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Layout and Tests tables, Invalid inputs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (materials, material_library and api lines)

**Interfaces:**
- Consumes: `material_library::{MATERIALS, EngineProps, chosen, *_CHOICES}` (Task 11), `temperature::{AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA}` (Task 8), `TemperatureLinks` (Tasks 5, 7, 10).
- Produces:
  - inputs `materials.parts.back_iron` (codes 1-8), `sleeve_liner` (1-4), `cap_housing` (1-3), Rust-only, default 1;
  - `material_library::{PartMaterial { material, is_default, props }, PartProperties { backiron, design_flux_T, back_iron, steel, body, sleeve_liner, cap }, resolve(choice, steel, backiron, md, slip, thermal) -> PartProperties, NO_MATERIAL}`;
  - `materials::compute(mat, t_bi_req_mm, wall_corner_mm, backiron, parts: &PartProperties, dev)`; Rust-only results `materials.circuit_backiron: i64`, `back_iron_material`, `sleeve_liner_material`, `cap_material: String`;
  - `TemperatureLinks { .., body_sigma_S_m, body_c, body_cte, body_E_GPa }` (E15, E17, E18 read the body; the cap keeps `al6061_sigma_S_m` and `th.c_aluminium`).

- [ ] **Step 1: Write the failing tests**

Create `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/material_links.rs` with exactly this content:

```rust
//! Addendum A5 physics links: what a part's material choice feeds into the
//! engine. Picking a steel or a sleeve material equals typing its library values
//! into the inputs it stands for; a non-ferromagnetic back iron selects the
//! free-space circuit and becomes the hub, cup and boss material; the backiron
//! input still overrides a ferromagnetic choice; the default choices change nothing.

mod common;

use std::collections::{BTreeMap, BTreeSet};

use magcoupling::engine::api::{DesignInputs, compute_all, compute_all_with};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, SLEEVE_LINER_CHOICES, material,
};
use magcoupling::engine::materials::PartMaterialInputs;
use magcoupling::engine::meta::{InputSet, SetErrorKind, Value, input_rows, result_rows};

/// Every value with a workbook cell (inputs and results) for `inputs` under `dev`.
fn cells(inputs: &DesignInputs, dev: Deviations) -> BTreeMap<String, Value> {
    let results = compute_all_with(inputs, dev);
    let mut map = BTreeMap::new();
    for row in input_rows(inputs) {
        if let Some(cell) = row.meta.cell {
            map.insert(cell.to_owned(), row.value);
        }
    }
    for row in result_rows(&results) {
        if let Some(cell) = row.cell {
            map.insert(cell, row.value);
        }
    }
    map
}

/// Cells whose RESULT value differs (input cells are what the user typed).
fn changed_results(
    inputs_a: &DesignInputs,
    inputs_b: &DesignInputs,
    dev: Deviations,
) -> BTreeSet<String> {
    let result_cells = |inputs: &DesignInputs| -> BTreeMap<String, Value> {
        result_rows(&compute_all_with(inputs, dev))
            .into_iter()
            .filter_map(|r| r.cell.map(|c| (c, r.value)))
            .collect()
    };
    let (a, b) = (result_cells(inputs_a), result_cells(inputs_b));
    a.iter()
        .filter(|(cell, v)| !parity_close(v, &b[*cell]))
        .map(|(cell, _)| cell.clone())
        .collect()
}

fn num(v: &Value) -> f64 {
    match v {
        Value::Num(x) => *x,
        other => panic!("expected a number, got {other:?}"),
    }
}

fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= 1e-12 * want.abs()
}

fn with_back_iron(code: i64) -> DesignInputs {
    let mut inputs = DesignInputs::default();
    inputs.materials.parts.back_iron = code;
    inputs
}

#[test]
fn each_selector_offers_the_library_choices() {
    let rows = input_rows(&PartMaterialInputs::default());
    for (path, choices) in [
        ("back_iron", &BACK_IRON_CHOICES[..]),
        ("sleeve_liner", &SLEEVE_LINER_CHOICES[..]),
        ("cap_housing", &CAP_HOUSING_CHOICES[..]),
    ] {
        let row = rows.iter().find(|r| r.path == path).expect(path);
        assert!(row.meta.rust_only, "{path}");
        let want: Vec<(i64, &str)> = choices
            .iter()
            .map(|&(code, id)| (code, material(id).expect("a material").label))
            .collect();
        assert_eq!(row.meta.choices, &want[..], "{path}");
        assert_eq!(row.value, Value::Int(1), "{path}: the workbook's material");
    }
}

#[test]
fn the_default_choices_change_nothing() {
    for dev in [Deviations::NONE, Deviations::ALL] {
        let base = DesignInputs::defaults_with(dev);
        let mut explicit = base.clone();
        explicit
            .set("materials.parts.back_iron", Value::Int(1))
            .expect("a choice");
        explicit
            .set("materials.parts.sleeve_liner", Value::Int(1))
            .expect("a choice");
        explicit
            .set("materials.parts.cap_housing", Value::Int(1))
            .expect("a choice");
        assert_eq!(
            compute_all_with(&explicit, dev),
            compute_all_with(&base, dev)
        );
    }
    let res = compute_all(&DesignInputs::default()).materials;
    assert_eq!(
        (
            res.circuit_backiron,
            res.back_iron_material.as_str(),
            res.sleeve_liner_material.as_str(),
            res.cap_material.as_str()
        ),
        (1, "4140 annealed", "316L annealed", "6061-T6 aluminium")
    );
}

#[test]
fn picking_a_steel_equals_typing_its_values() {
    // A ferromagnetic back iron supplies conductivity, density, specific heat, expansion
    // and modulus, and its design flux density where the library has one (1018: 1.7 T);
    // otherwise Materials C13 stays. Incremental permeability stays C15.
    for &(code, id) in &BACK_IRON_CHOICES[1..6] {
        let m = material(id).expect("a material");
        assert!(m.ferromagnetic, "{id}");
        let picked = with_back_iron(code);
        let mut typed = DesignInputs::default();
        let e = m.engine;
        let s = &mut typed.materials.steel;
        s.conductivity_S_m = e.sigma_S_m;
        s.specific_heat_J_kgK = e.cp_J_kgK;
        s.cte_per_C = e.cte_per_C;
        s.modulus_GPa = e.modulus_GPa;
        if let Some(b) = m.design_flux_density_T {
            s.bsat_T = b;
        }
        typed.metal.steel_density_g_mm3 = e.density_g_mm3;
        for dev in [Deviations::NONE, Deviations::ALL] {
            assert!(
                changed_results(&picked, &typed, dev).is_empty(),
                "{id}: {:?}",
                changed_results(&picked, &typed, dev)
            );
        }
        assert_eq!(compute_all(&picked).materials.back_iron_material, m.label);
    }
}

#[test]
fn picking_a_sleeve_equals_typing_its_values() {
    // The sleeve and liner (and the endplates, as Metal design C44 prices them).
    for &(code, id) in &SLEEVE_LINER_CHOICES[1..] {
        let e = material(id).expect("a material").engine;
        let mut picked = DesignInputs::default();
        picked.materials.parts.sleeve_liner = code;
        let mut typed = DesignInputs::default();
        typed.temperature.slip_loss.sigma_316_S_m = e.sigma_S_m;
        typed.temperature.thermal.c_316 = e.cp_J_kgK;
        typed.metal.sleeve_density_g_mm3 = e.density_g_mm3;
        for dev in [Deviations::NONE, Deviations::ALL] {
            assert!(changed_results(&picked, &typed, dev).is_empty(), "{id}");
        }
    }
}

#[test]
fn the_cap_choice_prices_the_cap_only() {
    // The cap's mass, loss and heat; the aluminium adapter (C188) keeps Metal design C42.
    let base = cells(&DesignInputs::default(), Deviations::ALL);
    for &(code, id) in &CAP_HOUSING_CHOICES[1..] {
        let e = material(id).expect("a material").engine;
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.cap_housing = code;
        let c = cells(&inputs, Deviations::ALL);
        let ratio = |cell: &str| num(&c[cell]) / num(&base[cell]);
        assert!(
            close(ratio("Metal design!C180"), e.density_g_mm3 / 0.0027),
            "{id} cap mass"
        );
        assert!(
            close(ratio("Temperature design!C128"), e.sigma_S_m / 2.5e7),
            "{id} cap loss"
        );
        assert_eq!(
            c["Temperature design!C112"],
            Value::Num(e.sigma_S_m),
            "{id}"
        );
        assert_eq!(
            c["Metal design!C188"], base["Metal design!C188"],
            "{id} adapter"
        );
        assert_eq!(c["Calculator!C93"], base["Calculator!C93"], "{id} torque");
    }
}

#[test]
fn a_non_ferromagnetic_back_iron_selects_the_free_space_circuit() {
    // Spec, Addendum testing: "a non-ferromagnetic back iron switches the circuit factor".
    // 304 and 6061 give the torque of the no-back-iron circuit (the geometry factor
    // s_free), as the backiron input 0 does, and E9's "No back iron".
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0;
    let free = cells(&no_iron, Deviations::ALL);
    let steel = cells(&DesignInputs::default(), Deviations::ALL);
    for code in [7, 8] {
        let c = cells(&with_back_iron(code), Deviations::ALL);
        assert_eq!(c["Calculator!C93"], free["Calculator!C93"], "{code}");
        assert_ne!(c["Calculator!C93"], steel["Calculator!C93"], "{code}");
        assert_eq!(c["Materials!C22"], Value::Text("No back iron".into()));
        assert_eq!(
            compute_all(&with_back_iron(code))
                .materials
                .circuit_backiron,
            0
        );
    }
}

#[test]
fn a_non_ferromagnetic_back_iron_is_the_hub_cup_and_boss_material() {
    // With no back iron the body is the picked material: density (E9, C111-C113),
    // specific heat (E15), conductivity (E17) and expansion and modulus (E18). The
    // default choice with C6 = 0 keeps the workbook's aluminium.
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0;
    let aluminium = cells(&no_iron, Deviations::ALL);
    let c304 = cells(&with_back_iron(7), Deviations::ALL);
    let e304 = material("304_annealed").expect("304").engine;
    let ratio = |cell: &str| num(&c304[cell]) / num(&aluminium[cell]);
    for cell in ["Calculator!C111", "Calculator!C112", "Calculator!C113"] {
        assert!(close(ratio(cell), e304.density_g_mm3 / 0.0027), "{cell}");
    }
    // E17's closed form is linear in conductivity (same fields and geometry).
    for cell in [
        "Temperature design!C123",
        "Temperature design!C124",
        "Temperature design!C125",
    ] {
        assert!(close(ratio(cell), e304.sigma_S_m / 2.5e7), "{cell}");
    }
    assert_ne!(
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
fn the_backiron_input_overrides_a_ferromagnetic_choice() {
    // C6 = 0 with 1018 picked: the free-space circuit and the workbook's aluminium body.
    let mut inputs = with_back_iron(2);
    inputs.coupling.backiron = 0;
    let res = compute_all(&inputs);
    assert_eq!(res.materials.circuit_backiron, 0);
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0;
    let aluminium = compute_all(&no_iron);
    assert_eq!(res.mass.cup_g, aluminium.mass.cup_g, "the E9 aluminium cup");
    assert_eq!(res.model.pullout_Nm, aluminium.model.pullout_Nm);
}

#[test]
fn the_design_flux_density_feeds_the_wall_check() {
    // Decision 20: t_bi = B_gap tau_p / (pi B_design). 1018's 1.7 T (the workbook's comment)
    // thins the wall needed by 1.5/1.7 and the 1.8 mm wall passes; 17-4PH has no sourced
    // design value, so Materials C13 (1.5 T) stays.
    let base = cells(&DesignInputs::default(), Deviations::ALL);
    let c1018 = cells(&with_back_iron(2), Deviations::ALL);
    let need = |c: &BTreeMap<String, Value>| num(&c["Calculator!C104"]);
    assert!(close(need(&c1018), need(&base) * 1.5 / 1.7));
    assert_eq!(c1018["Calculator!C36"], Value::Num(1.7));
    assert_eq!(
        base["Materials!C22"],
        Value::Text("Too thin: raise Metal design C122 to at least 2.0 mm".into())
    );
    assert_eq!(c1018["Materials!C22"], Value::Text("OK".into()));
    let c174 = cells(&with_back_iron(5), Deviations::ALL);
    assert_eq!(need(&c174), need(&base));
    assert_eq!(c174["Calculator!C36"], Value::Num(1.5));
}

#[test]
fn a_code_outside_the_choices_gives_nan_not_another_material() {
    // Decision D3 for the new selectors: never another material; validate() names it.
    let mut inputs = DesignInputs::default();
    inputs.materials.parts.back_iron = 99;
    inputs.materials.parts.sleeve_liner = 0;
    inputs.materials.parts.cap_housing = -1;
    let res = compute_all(&inputs);
    assert_eq!(res.materials.back_iron_material, "#N/A");
    assert!(res.mass.cup_g.is_nan() && res.temperature.slip_loss.sleeve_W.is_nan());
    assert!(res.retainers.cap_g.is_nan());
    let errors = inputs.validate().expect_err("three invalid codes");
    let paths: Vec<(&str, &SetErrorKind)> =
        errors.iter().map(|e| (e.path.as_str(), &e.kind)).collect();
    assert_eq!(
        paths,
        [
            (
                "materials.parts.back_iron",
                &SetErrorKind::NotAChoice { code: 99 }
            ),
            (
                "materials.parts.sleeve_liner",
                &SetErrorKind::NotAChoice { code: 0 }
            ),
            (
                "materials.parts.cap_housing",
                &SetErrorKind::NotAChoice { code: -1 }
            ),
        ]
    );
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test material_links 2>&1 | grep -E "^error" | head -2
```

Expected: compile errors, among them ``unresolved import `magcoupling::engine::materials::PartMaterialInputs` ``.

- [ ] **Step 3: Resolve the parts**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/material_library.rs`, replace:

```rust
/// The material a selector code picks among `choices`; `None` for a code outside them.
pub fn chosen(choices: &[(i64, &'static str)], code: i64) -> Option<&'static Material> {
    choices
        .iter()
        .find(|&&(c, _)| c == code)
        .and_then(|&(_, id)| material(id))
}
```

with:

```rust
/// The material a selector code picks among `choices`; `None` for a code outside them.
pub fn chosen(choices: &[(i64, &'static str)], code: i64) -> Option<&'static Material> {
    choices
        .iter()
        .find(|&&(c, _)| c == code)
        .and_then(|&(_, id)| material(id))
}

/// What a code outside the choices supplies (decision D3: never another material):
/// NaN properties, so the results say the input is invalid; `validate()` names it.
const NO_PROPS: EngineProps = EngineProps {
    sigma_S_m: f64::NAN,
    density_g_mm3: f64::NAN,
    cp_J_kgK: f64::NAN,
    cte_per_C: f64::NAN,
    modulus_GPa: f64::NAN,
};

/// The name results show for a code outside the choices.
pub const NO_MATERIAL: &str = "#N/A";

/// One part's material in effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartMaterial {
    /// The library record picked; `None` for a code outside the choices.
    pub material: Option<&'static Material>,
    /// Whether the pick is the part's default (the workbook's material: the inputs).
    pub is_default: bool,
    /// The values the engine reads for this part.
    pub props: EngineProps,
}

impl PartMaterial {
    /// The selector text of the pick, or [`NO_MATERIAL`].
    pub fn label(&self) -> &'static str {
        self.material.map_or(NO_MATERIAL, |m| m.label)
    }
}

/// What the parts' materials feed into the engine. `api::compute` builds it from
/// the selectors and the inputs; at the default choices every value is the input
/// it stands for, bit for bit.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct PartProperties {
    /// The circuit Calculator C6 selects in effect: 0 (free space) for a
    /// non-ferromagnetic back iron, else the C6 input, which thus overrides a
    /// ferromagnetic choice toward "no back iron".
    pub backiron: i64,
    /// The design flux density of the wall check (Materials C13 in effect).
    pub design_flux_T: f64,
    /// The back-iron pick (its name and properties, for display and the warnings).
    pub back_iron: PartMaterial,
    /// The steel circuit's values: the Materials inputs, or a ferromagnetic pick's.
    pub steel: EngineProps,
    /// The hub, cup and boss when there is no back iron: the workbook's aluminium
    /// (Metal design C42, Temperature design C140, Materials C43, and E18's 6061
    /// expansion and modulus), or a non-ferromagnetic pick's own values.
    pub body: EngineProps,
    /// Sleeve, liner and endplates (expansion and modulus not read).
    pub sleeve_liner: PartMaterial,
    /// The cap (expansion and modulus not read).
    pub cap: PartMaterial,
}

/// The workbook values of a part, as [`EngineProps`].
#[allow(non_snake_case)]
const fn props(
    sigma_S_m: f64,
    density_g_mm3: f64,
    cp_J_kgK: f64,
    cte_per_C: f64,
    modulus_GPa: f64,
) -> EngineProps {
    EngineProps {
        sigma_S_m,
        density_g_mm3,
        cp_J_kgK,
        cte_per_C,
        modulus_GPa,
    }
}

/// A part's material: the default pick stands for `workbook` (the inputs); another
/// pick supplies its engine values; a code outside the choices gives [`NO_PROPS`].
fn part(choices: &[(i64, &'static str)], code: i64, workbook: EngineProps) -> PartMaterial {
    let material = chosen(choices, code);
    let is_default = code == choices[0].0;
    let props = match material {
        Some(_) if is_default => workbook,
        Some(m) => m.engine,
        None => NO_PROPS,
    };
    PartMaterial {
        material,
        is_default,
        props,
    }
}

/// Resolves the three part selectors against the inputs (Addendum A5 physics links):
/// a ferromagnetic back iron keeps the steel circuit and supplies the steel values and
/// its design flux density where the library has one (else Materials C13 stays); a
/// non-ferromagnetic one selects the free-space circuit and becomes the hub, cup and
/// boss material; the sleeve and cap picks supply their conductivity, density and
/// specific heat. Incremental permeability always stays Materials C15 (no source
/// gives it at the magnet bias).
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub fn resolve(
    choice: &super::materials::PartMaterialInputs,
    steel: &super::materials::Steel4140,
    backiron: i64,
    md: &super::metal_design::MetalDesignInputs,
    slip: &super::temperature::SlipLossInputs,
    thermal: &super::temperature::ThermalInputs,
) -> PartProperties {
    use super::materials::AL6061;
    use super::temperature::{AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA};
    let workbook_steel = props(
        steel.conductivity_S_m,
        md.steel_density_g_mm3,
        steel.specific_heat_J_kgK,
        steel.cte_per_C,
        steel.modulus_GPa,
    );
    let back_iron = part(&BACK_IRON_CHOICES, choice.back_iron, workbook_steel);
    let non_magnetic = back_iron.material.is_some_and(|m| !m.ferromagnetic);
    let design_flux_T = match back_iron.material {
        Some(m) if !back_iron.is_default => m.design_flux_density_T.unwrap_or(steel.bsat_T),
        Some(_) => steel.bsat_T,
        None => f64::NAN,
    };
    let aluminium = props(
        AL6061.conductivity_S_m,
        md.al_density_g_mm3,
        thermal.c_aluminium,
        AL_HUB_CTE_PER_C,
        AL_HUB_MODULUS_GPA,
    );
    PartProperties {
        backiron: if non_magnetic { 0 } else { backiron },
        design_flux_T,
        back_iron,
        // A non-magnetic pick leaves the steel values at the inputs: only cells the
        // workbook still prices as steel with no back iron (E9 off) read them.
        steel: if non_magnetic {
            workbook_steel
        } else {
            back_iron.props
        },
        body: if non_magnetic {
            back_iron.props
        } else {
            aluminium
        },
        sleeve_liner: part(
            &SLEEVE_LINER_CHOICES,
            choice.sleeve_liner,
            props(
                slip.sigma_316_S_m,
                md.sleeve_density_g_mm3,
                thermal.c_316,
                f64::NAN,
                f64::NAN,
            ),
        ),
        cap: part(
            &CAP_HOUSING_CHOICES,
            choice.cap_housing,
            props(
                AL6061.conductivity_S_m,
                md.al_density_g_mm3,
                thermal.c_aluminium,
                f64::NAN,
                f64::NAN,
            ),
        ),
    }
}
```

- [ ] **Step 4: Wire the selectors and the links**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
use super::compat::{ceiling, fmt_fixed};
use super::deviations::{DeviationId, Deviations};
use super::meta::{inputs, out, param, results};
```

with:

```rust
use super::compat::{ceiling, fmt_fixed};
use super::deviations::{DeviationId, Deviations};
use super::material_library::PartProperties;
use super::meta::{inputs, out, out_rust_only, param, param_rust_only, results};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
inputs! {
    /// Every Materials input, grouped as the Python `MaterialsInputs`.
    pub struct MaterialsInputs {
        fields {}
        groups {
            steel: Steel4140,
            nickel: ElectrolessNickel,
            screws: ScrewClasses,
        }
    }
}
```

with:

```rust
inputs! {
    /// The material of each part (Addendum A5, Rust-only selectors). Code 1 of each
    /// is the workbook's material, whose values are the inputs; the choices are the
    /// library records of `material_library` (`BACK_IRON_CHOICES` and the others,
    /// tested equal to these texts).
    pub struct PartMaterialInputs {
        fields {
            back_iron: i64 = 1 => param_rust_only("-", "Back iron material (hub, cup and boss)",
                "Addendum A5. 1 = the workbook's 4140: the steel inputs above. Another choice supplies its library conductivity, density, specific heat, expansion and modulus, and its design flux density where the library has one (else C13 stays). A non-ferromagnetic choice selects the free-space circuit and becomes the hub, cup and boss material; with a ferromagnetic one, Calculator C6 = 0 still selects the free-space circuit (the override).")
                .choices(&[
                    (1, "4140 annealed"),
                    (2, "1018 hot rolled"),
                    (3, "12L14 cold drawn"),
                    (4, "416 stainless, annealed"),
                    (5, "17-4PH H1150"),
                    (6, "17-4PH H900"),
                    (7, "304 stainless (non-magnetic)"),
                    (8, "6061-T6 aluminium"),
                ]),
            sleeve_liner: i64 = 1 => param_rust_only("-", "Sleeve and liner material",
                "Addendum A5. 1 = the workbook's 316L (Temperature design C111 and C139, Metal design C44). Another choice supplies its conductivity, density and specific heat; the endplates follow it, as C44 prices them.")
                .choices(&[
                    (1, "316L annealed"),
                    (2, "Ti-6Al-4V grade 5"),
                    (3, "Inconel 625"),
                    (4, "PEEK"),
                ]),
            cap_housing: i64 = 1 => param_rust_only("-", "Cap and housing material",
                "Addendum A5. 1 = the workbook's 6061-T6 (Materials C43, Temperature design C140, Metal design C42). Another choice supplies the cap's conductivity, density and specific heat; the aluminium adapter and the clamp alloy keep their own inputs.")
                .choices(&[
                    (1, "6061-T6 aluminium"),
                    (2, "7075-T6 aluminium"),
                    (3, "Acetal (POM-H)"),
                ]),
        }
    }
}

inputs! {
    /// Every Materials input, grouped as the Python `MaterialsInputs` (plus the
    /// Rust-only part selectors).
    pub struct MaterialsInputs {
        fields {}
        groups {
            steel: Steel4140,
            nickel: ElectrolessNickel,
            screws: ScrewClasses,
            parts: PartMaterialInputs,
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
            ods_under_dia_mm: f64 => out("mm", "Machine outside diameters under (on diameter)", "", "Materials!C30"),
        }
    }
}
```

with:

```rust
            ods_under_dia_mm: f64 => out("mm", "Machine outside diameters under (on diameter)", "", "Materials!C30"),
            circuit_backiron: i64 => out_rust_only("-", "Back-iron circuit in effect",
                "1 = steel circuit, 0 = free space: Calculator C6, or 0 for a non-ferromagnetic back iron (Addendum A5)."),
            back_iron_material: String => out_rust_only("", "Back iron material", ""),
            sleeve_liner_material: String => out_rust_only("", "Sleeve and liner material", ""),
            cap_material: String => out_rust_only("", "Cap and housing material", ""),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
/// Cup wall check against the back-iron need, and electroless-nickel pre-plate offsets.
/// `backiron` is the Calculator selector C6 (1 steel, 0 none); only E9 reads it.
pub fn compute(
    mat: &MaterialsInputs,
    t_bi_req_mm: f64,
    wall_corner_mm: f64,
    backiron: i64,
    dev: Deviations,
) -> MaterialsResults {
```

with:

```rust
/// Cup wall check against the back-iron need, and electroless-nickel pre-plate offsets.
/// `backiron` is the Calculator selector C6 in effect (1 steel, 0 none); only E9
/// reads it. `parts` names the materials in effect (Rust-only results).
pub fn compute(
    mat: &MaterialsInputs,
    t_bi_req_mm: f64,
    wall_corner_mm: f64,
    backiron: i64,
    parts: &PartProperties,
    dev: Deviations,
) -> MaterialsResults {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
        bores_over_dia_mm: 2.0 * t,
        ods_under_dia_mm: 2.0 * t,
    }
}
```

with:

```rust
        bores_over_dia_mm: 2.0 * t,
        ods_under_dia_mm: 2.0 * t,
        circuit_backiron: backiron,
        back_iron_material: parts.back_iron.label().to_owned(),
        sleeve_liner_material: parts.sleeve_liner.label().to_owned(),
        cap_material: parts.cap.label().to_owned(),
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::material_library::resolve;
    use crate::engine::metal_design::MetalDesignInputs;
    use crate::engine::temperature::{SlipLossInputs, ThermalInputs};

    /// The default parts (every selector at code 1).
    fn parts() -> PartProperties {
        let mat = MaterialsInputs::default();
        resolve(
            &mat.parts,
            &mat.steel,
            1,
            &MetalDesignInputs::default(),
            &SlipLossInputs::default(),
            &ThermalInputs::default(),
        )
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
compute(&mat, needed, corner, 1, Deviations::NONE)
```

with:

```rust
compute(&mat, needed, corner, 1, &parts(), Deviations::NONE)
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
compute(&mat, 1.90415278222222, 1.8, 1, Deviations::NONE)
```

with:

```rust
compute(&mat, 1.90415278222222, 1.8, 1, &parts(), Deviations::NONE)
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
|backiron, dev| compute(&mat, 1.90415278222222, 1.8, backiron, dev).cup_wall_check;
```

with:

```rust
|backiron, dev| compute(&mat, 1.90415278222222, 1.8, backiron, &parts(), dev).cup_wall_check;
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
compute(&mat, 1.8, 1.8, 0, e9).cup_wall_check
```

with:

```rust
compute(&mat, 1.8, 1.8, 0, &parts(), e9).cup_wall_check
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    /// E9: the cup and boss are aluminium (`model::cup_is_aluminium`).
    pub cup_aluminium: bool,
    /// The hub is aluminium, C6 != 1 (`model::hub_is_aluminium`).
    pub hub_aluminium: bool,
```

with:

```rust
    /// E9: the cup and boss are aluminium (`model::cup_is_aluminium`).
    pub cup_aluminium: bool,
    /// The hub is aluminium, C6 != 1 (`model::hub_is_aluminium`).
    pub hub_aluminium: bool,
    /// The hub, cup and boss material without back iron (Addendum A5,
    /// `material_library::PartProperties::body`): the workbook's aluminium by default
    /// (C43, C140 and E18's 6061), or a non-ferromagnetic back-iron pick. E15, E17 and
    /// E18 read it.
    pub body_sigma_S_m: f64,
    pub body_c: f64,
    pub body_cte: f64,
    pub body_E_GPa: f64,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let C = if dev.is_on(DeviationId::E15) && k.hub_aluminium {
        let c_cup = if k.cup_aluminium {
            th.c_aluminium
        } else {
            k.steel_c
        };
        (k.mass_magnets_g * th.c_ndfeb
            + (k.mass_cup_g + k.mass_boss_g) * c_cup
            + k.mass_hub_g * th.c_aluminium
```

with:

```rust
    let C = if dev.is_on(DeviationId::E15) && k.hub_aluminium {
        let c_cup = if k.cup_aluminium { k.body_c } else { k.steel_c };
        (k.mass_magnets_g * th.c_ndfeb
            + (k.mass_cup_g + k.mass_boss_g) * c_cup
            + k.mass_hub_g * k.body_c
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        sl.end_factor * k.al6061_sigma_S_m * we.powi(2) * B.powi(2) / (2.0 * kk.powi(2))
```

with:

```rust
        sl.end_factor * k.body_sigma_S_m * we.powi(2) * B.powi(2) / (2.0 * kk.powi(2))
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        sl.end_factor * k.al6061_sigma_S_m * we.powi(2) / 2.0
            * (r_w / pp).powi(2)
```

with:

```rust
        sl.end_factor * k.body_sigma_S_m * we.powi(2) / 2.0
            * (r_w / pp).powi(2)
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    let (hub_cte, hub_E_GPa) = if dev.is_on(DeviationId::E18) && k.hub_aluminium {
        (AL_HUB_CTE_PER_C, AL_HUB_MODULUS_GPA)
    } else {
```

with:

```rust
    let (hub_cte, hub_E_GPa) = if dev.is_on(DeviationId::E18) && k.hub_aluminium {
        (k.body_cte, k.body_E_GPa)
    } else {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            cup_aluminium: false,
            hub_aluminium: false,
            inner_grade: crate::engine::grades::grade("N42SH"),
```

with:

```rust
            cup_aluminium: false,
            hub_aluminium: false,
            body_sigma_S_m: 2.5e7,
            body_c: 900.0,
            body_cte: AL_HUB_CTE_PER_C,
            body_E_GPa: AL_HUB_MODULUS_GPA,
            inner_grade: crate::engine::grades::grade("N42SH"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
        let (pp, sigma, f_end) = (5.0_f64, k.al6061_sigma_S_m, sl.end_factor);
```

with:

```rust
        let (pp, sigma, f_end) = (5.0_f64, k.body_sigma_S_m, sl.end_factor);
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
use super::materials::{self, MaterialsInputs, MaterialsResults};
```

with:

```rust
use super::material_library;
use super::materials::{self, MaterialsInputs, MaterialsResults};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
// Python api.compute_all lines 52-99; Python local names.
fn compute(inputs: &DesignInputs, dev: Deviations) -> DesignResults {
    let (ci, md, cal_in, mat_in) = (
        &inputs.coupling,
        &inputs.metal,
        &inputs.calibration,
        &inputs.materials,
    );
```

with:

```rust
// Python api.compute_all lines 52-99; Python local names.
fn compute(inputs: &DesignInputs, dev: Deviations) -> DesignResults {
    let (cal_in, mat_in) = (&inputs.calibration, &inputs.materials);
    // Addendum A5: the parts' materials in effect. At the default choices every value is
    // the input it stands for, so the copies below equal the inputs bit for bit.
    let parts = material_library::resolve(
        &mat_in.parts,
        &mat_in.steel,
        inputs.coupling.backiron,
        &inputs.metal,
        &inputs.temperature.slip_loss,
        &inputs.temperature.thermal,
    );
    let ci = &CouplingInputs {
        backiron: parts.backiron,
        ..inputs.coupling.clone()
    };
    let md = &MetalDesignInputs {
        steel_density_g_mm3: parts.steel.density_g_mm3,
        sleeve_density_g_mm3: parts.sleeve_liner.props.density_g_mm3,
        ..inputs.metal.clone()
    };
    // The cap is the only part the retainers price at Metal design C42.
    let md_retainers = &MetalDesignInputs {
        al_density_g_mm3: parts.cap.props.density_g_mm3,
        ..md.clone()
    };
    let mut ti = inputs.temperature.clone();
    ti.slip_loss.sigma_316_S_m = parts.sleeve_liner.props.sigma_S_m;
    ti.thermal.c_316 = parts.sleeve_liner.props.cp_J_kgK;
    ti.thermal.c_aluminium = parts.cap.props.cp_J_kgK;
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        cal_in.alpha_br_per_C,
        mat_in.steel.bsat_T,
        f_cal,
```

with:

```rust
        cal_in.alpha_br_per_C,
        parts.design_flux_T,
        f_cal,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
    let ret = metal_design::retainers(
        md,
```

with:

```rust
    let ret = metal_design::retainers(
        md_retainers,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        md.steel_density_g_mm3,
        md.al_density_g_mm3,
        ret.retainers_g,
```

with:

```rust
        md.steel_density_g_mm3,
        parts.body.density_g_mm3,
        ret.retainers_g,
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        model::cup_boss_density(
            ci.backiron,
            md.steel_density_g_mm3,
            md.al_density_g_mm3,
            dev,
        ),
```

with:

```rust
        model::cup_boss_density(
            ci.backiron,
            md.steel_density_g_mm3,
            parts.body.density_g_mm3,
            dev,
        ),
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
    let matr = materials::compute(
        mat_in,
        m.backiron_needed_mm,
        md.cup_wall_corner_mm,
        ci.backiron,
        dev,
    );
```

with:

```rust
    let matr = materials::compute(
        mat_in,
        m.backiron_needed_mm,
        md.cup_wall_corner_mm,
        ci.backiron,
        &parts,
        dev,
    );
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        steel_sigma_S_m: mat_in.steel.conductivity_S_m,
        steel_mu_r: mat_in.steel.mu_r_incremental,
        steel_c: mat_in.steel.specific_heat_J_kgK,
        steel_cte: mat_in.steel.cte_per_C,
        steel_E_GPa: mat_in.steel.modulus_GPa,
        al6061_sigma_S_m: materials::AL6061.conductivity_S_m,
        cup_aluminium: model::cup_is_aluminium(ci.backiron, dev),
        hub_aluminium: model::hub_is_aluminium(ci.backiron),
    };
    let temp = temperature::compute(&inputs.temperature, &links, dev);
```

with:

```rust
        steel_sigma_S_m: parts.steel.sigma_S_m,
        steel_mu_r: mat_in.steel.mu_r_incremental,
        steel_c: parts.steel.cp_J_kgK,
        steel_cte: parts.steel.cte_per_C,
        steel_E_GPa: parts.steel.modulus_GPa,
        al6061_sigma_S_m: parts.cap.props.sigma_S_m,
        cup_aluminium: model::cup_is_aluminium(ci.backiron, dev),
        hub_aluminium: model::hub_is_aluminium(ci.backiron),
        body_sigma_S_m: parts.body.sigma_S_m,
        body_c: parts.body.cp_J_kgK,
        body_cte: parts.body.cte_per_C,
        body_E_GPa: parts.body.modulus_GPa,
    };
    let temp = temperature::compute(&ti, &links, dev);
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

- [ ] **Step 5: Bless the input schema and run everything**

Run:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test schema
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: the bless run adds the three selectors; every binary `ok`: `material_links.rs` 10 passed; `parity.rs`, `differential.rs` and `deviations.rs` unchanged (the default choices are the inputs, bit for bit).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|panicked"
```

Expected: no output.

Run:

```bash
cd C:/Users/Cole/source/repos/lsim-mag-a1/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python).

- [ ] **Step 6: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells) |
```

with:

```markdown
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `tests/static_data.rs` |
```

with:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material; C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material. |
| `tests/static_data.rs` |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
Call `DesignInputs::validate()` where inputs enter: it returns
every offending path, in schema order, with the reason `set` would give.
```

with:

```markdown
Call `DesignInputs::validate()` where inputs enter: it returns
every offending path, in schema order, with the reason `set` would give. The
Rust-only selectors follow the same rule: a material code outside its choices
gives NaN properties and the name `"#N/A"`, never another material; the
two-way coercivity source (E20) falls through to C44 and C45, as a workbook IF.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    materials: "materials.rs (Materials sheet: inputs; AL7075 and AL6061 alloys;
```

with:

```yaml
    materials: "materials.rs (Materials sheet: inputs, plus the Rust-only part selectors materials.parts (back_iron, sleeve_liner, cap_housing); AL7075 and AL6061 alloys;
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, CAP_HOUSING_CHOICES, code 1 = the workbook's material)
```

with:

```yaml
BACK_IRON_CHOICES, SLEEVE_LINER_CHOICES, CAP_HOUSING_CHOICES, code 1 = the workbook's material; resolve -> PartProperties: the circuit in effect, the design flux density, the steel, body (no back iron), sleeve and cap values the engine reads)
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    api: api.rs (DesignInputs/DesignResults groups in Python order, the complete Python DesignResults;
```

with:

```yaml
    api: api.rs (DesignInputs/DesignResults groups in Python order, the complete Python DesignResults; compute resolves the part materials first (material_library::resolve) and feeds the values in effect;
```

- [ ] **Step 7: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task12.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task12.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task12.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 8: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/material_library.rs magcoupling-rs/src/engine/materials.rs magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/tests/material_links.rs magcoupling-rs/tests/data/input_schema.json magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
feat(magcoupling-rs): per-part material selectors and physics links

Rust-only selectors materials.parts.{back_iron, sleeve_liner, cap_housing}.
The default choice is the workbook's material (the inputs); another choice
supplies its library values; a non-ferromagnetic back iron selects the
free-space circuit and becomes the hub, cup and boss material. Defaults bit
for bit.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 13: The six material warnings as engine outputs

**Model:** session model (omit `model`; the rule conditions are physics judgment, Decisions to confirm A10). Physics reviewer: spec A5 (warning rules), report section 4 and the data file's `warning_rule_inputs`.

Spec A5: plain language, colour-coded (Severity), linked to a teaching note (the id), and "each warning rule fires exactly on its
condition" (Addendum testing). The rules read the materials in effect, so a typed input can fire them as a pick does. Conditions and
thresholds (A10): non-ferromagnetic back iron = the free-space circuit in effect (a non-magnetic pick, or C6 = 0); ferromagnetic
sleeve = the pick is ferromagnetic (no listed sleeve is: unit-tested directly); high-conductivity sleeve = sigma in effect above 316L's
1.35e6 S/m; low saturation = a steel back iron with Bsat below 1.7 T (the highest design flux density the workbook names for a
back-iron steel, Materials C13's comment: a design value used as a threshold, not a saturation figure) or no design flux density of its
own (416, 12L14, 17-4PH); uncoated low-alloy steel = 4140, 1018 or 12L14 with no plating (Materials C26 = 0); CTE mismatch = the hub in
effect differs from the magnets (C95) by more than 15e-6 /C (not the default 4140's 13.1e-6; 304 17.7e-6 and aluminium 24.4e-6 are
above). A back-iron code outside its choices is no material (decision D3): `back_iron_known` keeps the two back-iron material rules
silent, and the invalid-code test of Task 12 now also checks that no warning fires. The tests check each rule both ways: it fires on its
condition and stays quiet when only a neighbouring condition holds (unplated stainless, a typed expansion coefficient).

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/warnings.rs` (`WARNING_RULES`, thresholds, `WarningInputs`, `WarningResults`, `compute`, unit tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/mod.rs` (`pub mod warnings;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs` (`DesignResults::warnings`; the inputs in effect)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/material_links.rs` (each warning end to end, fired and quiet; no warning for an unknown back iron)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (Layout and Tests tables)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml` (engine_files, ported)

**Interfaces:**
- Consumes: `material_library::PartProperties` (Task 12), `materials::MaterialsInputs::nickel`, `temperature::MismatchInputs::ndfeb_cte_per_C`.
- Produces:
  - `warnings::{Severity { Warning, Caution }, WarningRule { id, severity, note_id, text }, WARNING_RULES: [WarningRule; 6], SLEEVE_SIGMA_BASELINE_S_M = 1.35e6, LOW_SATURATION_T = 1.7, CTE_MISMATCH_LIMIT_PER_C = 15e-6, WarningInputs, WarningResults, compute(&WarningInputs) -> WarningResults}`; `WarningInputs::back_iron_known: bool` (false for a back-iron code outside its choices: the low-saturation and plating rules then stay silent, decision D3);
  - `DesignResults::warnings: WarningResults` (six Rust-only `String` results, `warnings.<rule id>`: the rule's text when it fires, else empty); the rule ids are the data file's `warning_rule_inputs` keys; note ids `a5.<rule id>` are placeholders for Addendum A4.

- [ ] **Step 1: Write the failing end-to-end test**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/tests/material_links.rs`, replace:

```rust
            (
                "materials.parts.cap_housing",
                &SetErrorKind::NotAChoice { code: -1 }
            ),
        ]
    );
}
```

with:

```rust
            (
                "materials.parts.cap_housing",
                &SetErrorKind::NotAChoice { code: -1 }
            ),
        ]
    );
    // An unknown back iron is no material: no warning rule reads it (decision D3).
    assert_eq!(fired(&inputs), Vec::<&str>::new());
}

/// The warning rules that fire for `inputs` (every correction on), by id.
fn fired(inputs: &DesignInputs) -> Vec<&'static str> {
    use magcoupling::engine::meta::ResultSet;
    use magcoupling::engine::warnings::WARNING_RULES;
    let w = compute_all(inputs).warnings;
    WARNING_RULES
        .iter()
        .filter(|rule| w.get(rule.id) == Some(Value::Text(rule.text.to_owned())))
        .map(|rule| rule.id)
        .collect()
}

#[test]
fn each_warning_fires_on_the_design_that_meets_its_condition() {
    // Spec, Addendum testing: "each warning rule fires exactly on its condition", end to
    // end. The ferromagnetic-sleeve rule has no listed sleeve that meets it (all four are
    // non-magnetic); src/engine/warnings.rs tests it directly.
    let open_circuit = ["non_ferromagnetic_back_iron", "cte_mismatch_with_magnets"];
    assert_eq!(fired(&DesignInputs::default()), Vec::<&str>::new());
    let mut no_iron = DesignInputs::default();
    no_iron.coupling.backiron = 0; // the workbook's aluminium hub, cup and boss
    assert_eq!(fired(&no_iron), open_circuit);
    for code in [7, 8] {
        assert_eq!(fired(&with_back_iron(code)), open_circuit, "{code}");
    }
    for code in [3, 4, 5, 6] {
        // 12L14 and 17-4PH have no design flux density of their own; 416's Bsat is 1.60 T.
        assert_eq!(fired(&with_back_iron(code)), ["low_saturation"], "{code}");
    }
    assert_eq!(
        fired(&with_back_iron(2)),
        Vec::<&str>::new(),
        "1018: 1.7 T sourced"
    );
    for code in 2..=4 {
        let mut inputs = DesignInputs::default();
        inputs.materials.parts.sleeve_liner = code;
        assert_eq!(fired(&inputs), Vec::<&str>::new(), "sleeve {code}");
    }
    let mut hot = DesignInputs::default();
    hot.temperature.slip_loss.sigma_316_S_m = 5e6; // typed: more conductive than 316L
    assert_eq!(fired(&hot), ["high_conductivity_sleeve_or_liner"]);
    let mut stiff = DesignInputs::default();
    stiff.materials.steel.cte_per_C = 16e-6; // typed: 16.8e-6 /°C from the magnets' -0.8e-6
    assert_eq!(fired(&stiff), ["cte_mismatch_with_magnets"]);
    for code in [1, 2, 3] {
        let mut bare = with_back_iron(code);
        bare.materials.nickel.thickness_mm = 0.0;
        let want: Vec<&str> = if code == 3 {
            vec!["low_saturation", "uncoated_low_alloy_steel"]
        } else {
            vec!["uncoated_low_alloy_steel"]
        };
        assert_eq!(fired(&bare), want, "{code}");
    }
    for code in [4, 5, 6] {
        // 416 and 17-4PH are stainless: unplated, only their saturation warns.
        let mut bare = with_back_iron(code);
        bare.materials.nickel.thickness_mm = 0.0;
        assert_eq!(fired(&bare), ["low_saturation"], "{code}");
    }
    let mut bare_304 = with_back_iron(7);
    bare_304.materials.nickel.thickness_mm = 0.0;
    assert_eq!(fired(&bare_304), open_circuit, "stainless needs no plating");
}
```

- [ ] **Step 2: Run it to see it fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test material_links 2>&1 | grep -E "^error" | head -2
```

Expected: compile errors, among them ``unresolved import `magcoupling::engine::warnings` ``.

- [ ] **Step 3: Create the rules (with their unit tests) and wire them**

Create `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/warnings.rs` with exactly this content:

```rust
//! Material consequence warnings (spec Addendum A5): six rules, each an engine
//! output with plain-language text, a severity and a teaching-note id (the
//! notes are Addendum A4's; the ids are placeholders until then).
//!
//! Each rule reads the materials IN EFFECT (`material_library::PartProperties`
//! and the inputs), so a value typed into an input can fire a rule as a library
//! pick does. A result is the rule's text when it fires and empty otherwise;
//! [`WARNING_RULES`] carries the severity and note id of each, in result order.
//! The three thresholds are model choices (no source; decision table of the
//! A-1 plan): [`SLEEVE_SIGMA_BASELINE_S_M`], [`LOW_SATURATION_T`],
//! [`CTE_MISMATCH_LIMIT_PER_C`].

use super::meta::{out_rust_only, results};

/// How serious a warning is (the GUI's badge colour).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    /// Red: the coupling does not work as designed (torque).
    Warning,
    /// Amber: a design consequence to check (heat, wall, corrosion, bond stress).
    Caution,
}

/// One rule: its result field, severity, teaching note and text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarningRule {
    /// The result field (`warnings.<id>`).
    pub id: &'static str,
    pub severity: Severity,
    /// Addendum A4 teaching-note id (placeholder until the notes exist).
    pub note_id: &'static str,
    /// What the result reads when the rule fires.
    pub text: &'static str,
}

/// The sleeve conductivity above which slip heating exceeds the workbook design's:
/// 316L, Temperature design!C111's default [S/m]. No listed sleeve exceeds it; a
/// typed conductivity can.
pub const SLEEVE_SIGMA_BASELINE_S_M: f64 = 1.35e6;

/// Saturation below which a back iron is flagged [T]: 1.7 T, the highest design
/// flux density the workbook names for a back-iron steel (Materials!C13's comment,
/// "1018 would be about 1.7 T", which the library keeps as 1018's
/// `design_flux_density_T`, decision 20). It is a design flux density, not a
/// saturation figure: no source gives a saturation threshold, so the rule is a
/// model choice (A-1 plan, decision A10). A steel that saturates below it cannot
/// carry the flux the best-sourced back iron is designed for.
pub const LOW_SATURATION_T: f64 = 1.7;

/// Expansion mismatch between the magnets and the part they bond to above which
/// the bond stress is "higher" [1/°C]: above the default design's 4140 hub
/// (12.3e-6 - (-0.8e-6) = 13.1e-6), below 304 stainless (17.7e-6).
pub const CTE_MISMATCH_LIMIT_PER_C: f64 = 15e-6;

/// The six rules, in the order of [`WarningResults`].
pub const WARNING_RULES: [WarningRule; 6] = [
    WarningRule {
        id: "non_ferromagnetic_back_iron",
        severity: Severity::Warning,
        note_id: "a5.non_ferromagnetic_back_iron",
        text: "Non-ferromagnetic back iron: the magnetic circuit is open, so torque drops, a strong stray field extends outside the coupling, and the part collects ferrous chips and debris.",
    },
    WarningRule {
        id: "ferromagnetic_sleeve_or_liner",
        severity: Severity::Warning,
        note_id: "a5.ferromagnetic_sleeve_or_liner",
        text: "Ferromagnetic sleeve or liner: it short-circuits the gap flux, so torque collapses.",
    },
    WarningRule {
        id: "high_conductivity_sleeve_or_liner",
        severity: Severity::Caution,
        note_id: "a5.high_conductivity_sleeve_or_liner",
        text: "High-conductivity sleeve or liner: more eddy current than 316L, so more slip heating.",
    },
    WarningRule {
        id: "low_saturation",
        severity: Severity::Caution,
        note_id: "a5.low_saturation",
        text: "Low or unsourced saturation: the wall check assumes a design flux density this back iron may not reach, so its walls need to be thicker.",
    },
    WarningRule {
        id: "uncoated_low_alloy_steel",
        severity: Severity::Caution,
        note_id: "a5.uncoated_low_alloy_steel",
        text: "Uncoated low-alloy steel: it corrodes, so it needs plating (the workbook plans electroless nickel).",
    },
    WarningRule {
        id: "cte_mismatch_with_magnets",
        severity: Severity::Caution,
        note_id: "a5.cte_mismatch_with_magnets",
        text: "Large expansion mismatch between the magnets and the hub they are bonded to: higher bond stress over temperature swings.",
    },
];

results! {
    /// The material warnings (Rust-only): each the rule's text when it fires, else empty.
    pub struct WarningResults {
        fields {
            non_ferromagnetic_back_iron: String => out_rust_only("", "Non-ferromagnetic back iron",
                "Fires when the circuit in effect is free space: a non-ferromagnetic back iron, or Calculator C6 = 0."),
            ferromagnetic_sleeve_or_liner: String => out_rust_only("", "Ferromagnetic sleeve or liner",
                "Fires when the sleeve and liner material is ferromagnetic."),
            high_conductivity_sleeve_or_liner: String => out_rust_only("", "High-conductivity sleeve or liner",
                "Fires when the sleeve conductivity in effect exceeds 316L's 1.35e6 S/m."),
            low_saturation: String => out_rust_only("", "Low saturation",
                "Fires for a ferromagnetic back iron whose saturation is below 1.7 T (the highest design flux density the workbook names for a back-iron steel) or that has no design flux density of its own (the wall check then keeps Materials C13)."),
            uncoated_low_alloy_steel: String => out_rust_only("", "Uncoated low-alloy steel",
                "Fires for a plain or low-alloy steel back iron with no plating (Materials C26 = 0)."),
            cte_mismatch_with_magnets: String => out_rust_only("", "Expansion mismatch with the magnets",
                "Fires when the hub's expansion coefficient differs from the magnets' (Temperature design C95) by more than 15e-6 /°C."),
        }
    }
}

/// What the rules read: the materials in effect.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffixes, as the engine's names
pub struct WarningInputs {
    /// Calculator C6 in effect (1 steel circuit; anything else is free space, as the engine takes it).
    pub circuit_backiron: i64,
    /// The sleeve and liner material is ferromagnetic.
    pub sleeve_ferromagnetic: bool,
    /// The sleeve conductivity in effect [S/m].
    pub sleeve_sigma_S_m: f64,
    /// The back-iron code names a library material. False for a code outside its
    /// choices: that is no material (decision D3), so rules 4 and 5 stay silent.
    pub back_iron_known: bool,
    /// The back iron's saturation flux density, where sourced [T].
    pub back_iron_bsat_T: Option<f64>,
    /// The back iron has its own design flux density (the workbook's for 4140, the library's).
    pub back_iron_design_flux_sourced: bool,
    /// The back iron is plain or low-alloy steel.
    pub back_iron_needs_plating: bool,
    /// Electroless nickel thickness (Materials C26) [mm].
    pub plating_mm: f64,
    /// The hub's expansion coefficient in effect [1/°C] (the steel circuit's, or the
    /// body's with no back iron).
    pub hub_cte_per_C: f64,
    /// The magnets' expansion coefficient in the bond plane (Temperature design C95) [1/°C].
    pub magnet_cte_per_C: f64,
}

/// Evaluates the six rules.
pub fn compute(w: &WarningInputs) -> WarningResults {
    let steel = w.circuit_backiron == 1;
    let fired = [
        !steel,
        w.sleeve_ferromagnetic,
        w.sleeve_sigma_S_m > SLEEVE_SIGMA_BASELINE_S_M,
        steel
            && w.back_iron_known
            && (w.back_iron_bsat_T.is_some_and(|b| b < LOW_SATURATION_T)
                || !w.back_iron_design_flux_sourced),
        steel && w.back_iron_known && w.back_iron_needs_plating && w.plating_mm <= 0.0,
        (w.hub_cte_per_C - w.magnet_cte_per_C).abs() > CTE_MISMATCH_LIMIT_PER_C,
    ];
    let text = |i: usize| {
        if fired[i] {
            WARNING_RULES[i].text.to_owned()
        } else {
            String::new()
        }
    };
    WarningResults {
        non_ferromagnetic_back_iron: text(0),
        ferromagnetic_sleeve_or_liner: text(1),
        high_conductivity_sleeve_or_liner: text(2),
        low_saturation: text(3),
        uncoated_low_alloy_steel: text(4),
        cte_mismatch_with_magnets: text(5),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::meta::{ResultSet, Value};

    /// The default design's inputs: no rule fires.
    fn quiet() -> WarningInputs {
        WarningInputs {
            circuit_backiron: 1,
            sleeve_ferromagnetic: false,
            sleeve_sigma_S_m: 1.35e6,
            back_iron_known: true,
            back_iron_bsat_T: None,
            back_iron_design_flux_sourced: true,
            back_iron_needs_plating: true,
            plating_mm: 0.015,
            hub_cte_per_C: 12.3e-6,
            magnet_cte_per_C: -0.8e-6,
        }
    }

    /// A change to [`quiet`] that meets one rule's condition.
    type Meet = fn(&mut WarningInputs);

    /// Which rules fired, by index.
    fn fired(w: &WarningInputs) -> Vec<usize> {
        let r = compute(w);
        WARNING_RULES
            .iter()
            .enumerate()
            .filter(|(_, rule)| r.get(rule.id) != Some(Value::Text(String::new())))
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn the_rules_are_the_result_fields_in_order() {
        let names: Vec<&str> = WarningResults::FIELDS.iter().map(|m| m.name).collect();
        let ids: Vec<&str> = WARNING_RULES.iter().map(|r| r.id).collect();
        assert_eq!(names, ids);
        for rule in &WARNING_RULES {
            assert!(!rule.text.is_empty() && rule.note_id == format!("a5.{}", rule.id));
        }
        assert!(WarningResults::FIELDS.iter().all(|m| m.rust_only));
    }

    #[test]
    fn each_rule_fires_exactly_on_its_condition() {
        // Spec, Addendum testing: "each warning rule fires exactly on its condition".
        assert_eq!(fired(&quiet()), Vec::<usize>::new());
        let cases: [(usize, Meet); 6] = [
            (0, |w| w.circuit_backiron = 0),
            (1, |w| w.sleeve_ferromagnetic = true),
            (2, |w| w.sleeve_sigma_S_m = 1.3500001e6),
            (3, |w| w.back_iron_bsat_T = Some(1.6)),
            (4, |w| w.plating_mm = 0.0),
            (5, |w| w.hub_cte_per_C = 16.9e-6),
        ];
        for (rule, set) in cases {
            let mut w = quiet();
            set(&mut w);
            assert_eq!(fired(&w), vec![rule], "{}", WARNING_RULES[rule].id);
            assert_eq!(
                compute(&w).get(WARNING_RULES[rule].id),
                Some(Value::Text(WARNING_RULES[rule].text.to_owned()))
            );
        }
        // Low saturation also fires for a back iron with no design flux density of its own.
        let mut w = quiet();
        w.back_iron_design_flux_sourced = false;
        assert_eq!(fired(&w), vec![3]);
        // Plating is needed for plain and low-alloy steel only: unplated stainless is quiet.
        let mut w = quiet();
        w.back_iron_needs_plating = false;
        w.plating_mm = 0.0;
        assert_eq!(fired(&w), Vec::<usize>::new());
    }

    #[test]
    fn an_unknown_back_iron_fires_no_material_rule() {
        // Decision D3: a back-iron code outside its choices is no material, so its missing
        // saturation, design flux density and plating need fire nothing.
        let mut w = quiet();
        w.back_iron_known = false;
        w.back_iron_bsat_T = Some(1.0);
        w.back_iron_design_flux_sourced = false;
        w.plating_mm = 0.0;
        assert_eq!(fired(&w), Vec::<usize>::new());
    }

    #[test]
    fn thresholds_are_strict_at_equality() {
        // Each comparison at exact equality does not fire (the threshold is the last quiet value).
        let mut w = quiet();
        w.sleeve_sigma_S_m = SLEEVE_SIGMA_BASELINE_S_M;
        w.back_iron_bsat_T = Some(LOW_SATURATION_T);
        w.hub_cte_per_C = CTE_MISMATCH_LIMIT_PER_C;
        w.magnet_cte_per_C = 0.0;
        assert_eq!(
            w.hub_cte_per_C - w.magnet_cte_per_C,
            CTE_MISMATCH_LIMIT_PER_C
        );
        assert_eq!(fired(&w), Vec::<usize>::new());
    }

    #[test]
    fn steel_rules_need_the_steel_circuit() {
        // With free space in effect there is no steel back iron to saturate or to plate.
        let mut w = quiet();
        w.circuit_backiron = 0;
        w.back_iron_bsat_T = Some(1.0);
        w.back_iron_design_flux_sourced = false;
        w.plating_mm = 0.0;
        assert_eq!(fired(&w), vec![0]);
        w.circuit_backiron = 7; // an invalid code is free space, as the engine takes it
        assert_eq!(fired(&w), vec![0]);
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/mod.rs`, replace:

```rust
pub mod temperature;
```

with:

```rust
pub mod temperature;
pub mod warnings;
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
use super::temperature::{self, TemperatureInputs, TemperatureResults};
```

with:

```rust
use super::temperature::{self, TemperatureInputs, TemperatureResults};
use super::warnings::{self, WarningResults};
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
            temperature: TemperatureResults,
            clamps: ClampResults,
        }
        tables {
```

with:

```rust
            temperature: TemperatureResults,
            clamps: ClampResults,
            warnings: WarningResults,
        }
        tables {
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
    let temp = temperature::compute(&ti, &links, dev);
```

with:

```rust
    let temp = temperature::compute(&ti, &links, dev);
    // Addendum A5: the material consequence warnings, on the materials in effect.
    let back_iron = parts.back_iron.material;
    let warn = warnings::compute(&warnings::WarningInputs {
        circuit_backiron: parts.backiron,
        sleeve_ferromagnetic: parts.sleeve_liner.material.is_some_and(|m| m.ferromagnetic),
        sleeve_sigma_S_m: parts.sleeve_liner.props.sigma_S_m,
        back_iron_known: back_iron.is_some(),
        back_iron_bsat_T: back_iron.and_then(|m| m.bsat_T.value),
        back_iron_design_flux_sourced: back_iron.is_some_and(|m| {
            parts.back_iron.is_default || m.design_flux_density_T.is_some()
        }),
        back_iron_needs_plating: back_iron.is_some_and(|m| m.needs_plating),
        plating_mm: mat_in.nickel.thickness_mm,
        hub_cte_per_C: if parts.backiron == 1 {
            parts.steel.cte_per_C
        } else {
            parts.body.cte_per_C
        },
        magnet_cte_per_C: ti.mismatch.ndfeb_cte_per_C,
    });
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/engine/api.rs`, replace:

```rust
        temperature: temp,
        clamps: clr,
```

with:

```rust
        temperature: temp,
        clamps: clr,
        warnings: warn,
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`: unit tests 107 passed (five in `warnings.rs`), `material_links.rs` 11 passed.

- [ ] **Step 4: Update the docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/api.rs` |
```

with:

```markdown
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
| `src/engine/api.rs` |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
a code outside the choices gives NaN, not another material. |
```

with:

```markdown
a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
    material_library: material_library.rs (
```

with:

```yaml
    warnings: warnings.rs (Addendum A5 material warnings: WARNING_RULES, six rules with text, Severity and note id; thresholds SLEEVE_SIGMA_BASELINE_S_M 1.35e6, LOW_SATURATION_T 1.7, CTE_MISMATCH_LIMIT_PER_C 15e-6; WarningResults = DesignResults::warnings, Rust-only)
    material_library: material_library.rs (
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/03-structure.yaml`, replace:

```yaml
material_library.rs, model.rs
```

with:

```yaml
material_library.rs, warnings.rs, model.rs
```

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task13.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task13.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task13.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 6: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/src/engine/warnings.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/src/engine/api.rs magcoupling-rs/tests/material_links.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
feat(magcoupling-rs): the six material warnings as engine outputs

Non-ferromagnetic back iron, ferromagnetic sleeve, high-conductivity sleeve,
low saturation, uncoated low-alloy steel and CTE mismatch, each with text,
severity and a teaching-note id, on the materials in effect (spec A5).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

### Task 14: Docs pass and the final gate

**Model:** `sonnet` (documentation with the exact text given). The final whole-branch review after this task runs on the session model (controller).

The repo rule (docs/ai/01-meta.yaml, coordination): after file-modifying work update 02-system (invariants, status), 04-memory (open questions) and 05-update-tracker; 03-structure was kept current task by task.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md` (status)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/lib.rs` (crate docs name both reports)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/02-system.yaml` (magcoupling responsibility, invariants, status)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/04-memory.yaml` (resolve the A6 and E9-residual items; the new open items)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/05-update-tracker.md` (the A-1 entry)

**Interfaces:**
- Consumes: everything above.
- Produces: docs that match the code (repo rule); a green final gate.

- [ ] **Step 1: Crate README status and crate docs**

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/README.md`, replace:

```markdown
**Status:** M2 complete: every Python module except `fields3d` (M3) is ported;
workbook parity covers all 1,149 checks; E1 to E14 applied (decisions D1 to D7
as recorded).
```

with:

```markdown
**Status:** M2 complete: every Python module except `fields3d` (M3) is ported;
workbook parity covers all 1,149 checks; E1 to E14 applied (decisions D1 to D7
as recorded). Addendum A-1 (data and physics) complete: the A6 grade table and
the parts' vendor data, any grade with manual dimensions, the A5 materials
library with per-part selectors, physics links and six warnings, and E15 to E20
applied (Addendum A decisions, approved 2026-09-30). Next: Addendum A-2
(parameters and sizing), A-3 (explanations), then M4.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/src/lib.rs`, replace:

```rust
//! workbook snapshot and the Python engine, except where an approved
//! correction from the M1 math audit is registered in
//! [`engine::deviations::REGISTRY`].
```

with:

```rust
//! workbook snapshot and the Python engine, except where an approved
//! correction (the M1 math audit's E1 to E14, the Addendum A verification's E15
//! to E20) is registered in [`engine::deviations::REGISTRY`]. The Addendum A
//! inputs the Python engine does not have (part materials, grades, the
//! coercivity source, the E17 free-space fields) are Rust-only and default to
//! the ported behaviour.
```

- [ ] **Step 2: docs/ai: system, memory, tracker**

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/02-system.yaml`, replace:

```yaml
      Workbook-exact except registered, approved corrections (deviation
      registry, M1 audit E1-E14). Engine complete (M2), every module but fields3d.
```

with:

```yaml
      Workbook-exact except registered, approved corrections (deviation
      registry, M1 audit E1-E14, Addendum A E15-E20). Engine complete (M2), every module but fields3d;
      Addendum A-1 adds the grade table, materials per part with physics links, and six warnings.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/02-system.yaml`, replace:

```yaml
      - Every departure from the workbook is an approved, registered deviation (E1-E14 in deviations.rs).
```

with:

```yaml
      - Every departure from the workbook is an approved, registered deviation (E1-E20 in deviations.rs; E15-E17 are probed on top of E9, decision 15).
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/02-system.yaml`, replace:

```yaml
      - Rust-only results (rust_only metadata) have no cell and are skipped by the parity, metadata and differential tests.
```

with:

```yaml
      - Rust-only inputs and results (rust_only metadata) have no cell and are skipped by the parity, metadata and differential tests; gen_differential.py never passes a Rust-only input to Python.
      - Every Rust-only selector's default reproduces the ported behaviour bit for bit: part materials at code 1 are the workbook inputs, a blank grade is the manual mode, coercivity_source 1 with the N42SH grade gives the workbook's 1592 kA/m and -0.005 /C.
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/02-system.yaml`, replace:

```yaml
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied; next: M3 fields3d, then M4 GUI"
```

with:

```yaml
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied; next: Addendum A-2 (parameters and sizing), A-3, then M4 GUI"
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/04-memory.yaml`, replace:

```yaml
  - "Addendum A6 owns the per-grade demagnetization fix: the addendum calls it a correctness fix in M2, but the M2 plan scoped all of Addendum A out. Today the demag check uses one N42SH Hcj curve for every magnet; with grade data each magnet uses its own Hcj(T), including ferrite (opposite-sign beta, demag risk at cold). The Addendum A engine plan must include it with a ferrite cold-case test"
```

with:

```yaml
  - "RESOLVED (Addendum A-1): the per-grade demagnetization fix is E20 (decision 19): each ring is checked with its own Hcj and beta, Br and rating and the weaker ring governs (A-1 decision A13; the workbook read only the inner ring against the outer blocks' fields), C44/C45 as overrides through the Rust-only temperature.demag.coercivity_source; ferrite (positive beta) is limited on the cold side (hot onsets +inf, rating as hot limit, Rust-only cold onsets/limit/check in the verdict, both rings); tests e20_* incl. the ferrite cold case through the custom-dimension mode and the mixed rings"
```

In `C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/04-memory.yaml`, replace:

```yaml
  - "E9 physics residuals at backiron = 0 (not approved corrections; A5 per-part materials is the natural owner, each needs user approval and a registry entry): (1) heat capacity prices cup + hub + boss at steel c (temperature.rs, C141 45.8 J/K; about 63.0 J/K with aluminium c = 900, +38 %, conservative: overstates the rise); (2) the removed web disc is priced at steel density (metal_design.rs, C148/C191 hybrid about 2.27 g low, C149 -4.69 g, C147 label says steel cup); (3) cup and web eddy losses use steel sigma and mu_r (temperature.rs, C121-C130) though E9 makes them aluminium"
```

with:

```yaml
  - "RESOLVED (Addendum A-1): the E9 residuals are E15 (heat capacity), E16 (removed web disc) and E17 (aluminium eddy losses, T1 with the free-space fields), plus E18 (the aluminium hub's mismatch screen); each reproduces the report's changed-cell tables"
  - "M3 must recompute and re-bless: E17's three Rust-only free-space fields (b_hub_free_T 0.07832 T, b_cup_free_T 0.08764 T, web_integral_free_T2m2 6.837e-6 T2m2, pinned at 4 s.f. from fields at Br 1.29 T; apply E3) and the reverse fields of a non-default grade (the E20 ferrite probe scales the stored NdFeB fields by 0.37/1.29 as a placeholder)"
  - "Open (Addendum A-1 decision A8): E18 uses the report's Alliance 6061 modulus 68.9 GPa while the A5 library's 6061 record carries Kaiser's 68.3 GPa (decision 5): picking 6061 as back iron differs from the default aluminium body in C104, C105 and C201 only (tests/material_links.rs pins it). Unify when the user decides"
  - "Addendum A-2 carries: the harmonic set as a parameter with the generalized peak search (decision 29); a per-ring alpha(Br) and magnet density for a non-NdFeB grade (A-1 decision A4 kept one alpha and the NdFeB density); A1 inverse sizing, housing autofit and the space claim"
  - "Addendum A-2 must allow for two A-1 overrides of assumption inputs: with E20 and coercivity_source 1 (the default), C44 and C45 change nothing for a magnet with a grade (every library part), and a library back iron with its own design flux density (1018) replaces Materials C13 in the wall check; the A3 test that each assumption moves its dependent results must skip or document them. C45's slider (-0.008 to -0.001) cannot enter ferrite's +0.0035 with the source at 0: widen or split it (A-1 decisions A2, A9)"
```

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import datetime, pathlib
p = pathlib.Path(r"C:/Users/Cole/source/repos/lsim-mag-a1/docs/ai/05-update-tracker.md")
s = p.read_text(encoding="utf-8")
anchor = "Reverse chronological (newest at top).\n\n---\n\n"
assert s.count(anchor) == 1
entry = f"""## {datetime.date.today():%Y-%m-%d} — Magcoupling Addendum A-1: data and physics (grades, materials, E15-E20)
- `magcoupling-rs`: the A6 grade table (`grades.rs`, 17 grades cited per value) and the parts' vendor
  data (coating, magnetization, vendor page); any grade with manual dimensions (Rust-only
  `coupling.magnets.grade_inner`/`grade_outer`); the A5 materials library (`material_library.rs`, 14
  materials) with Rust-only per-part selectors `materials.parts.*` whose default is the workbook
  material (the inputs), physics links (circuit, design flux density, conductivity, density, specific
  heat, expansion and modulus; the C6 override) and six warnings (`warnings.rs`, `DesignResults::warnings`).
- Corrections E15 (heat capacity), E16 (removed web disc), E17 (aluminium eddy losses, T1 with three
  Rust-only free-space fields), E18 (aluminium hub mismatch screen), E19 (SuperMagnetMan arcs per the
  vendor grid) and E20 (each ring's own coercivity, Br and rating, the weaker ring governing; ferrite limited
  on the cold side), each with the
  report's changed-cell tables or probes as expected values. Registry: `Approval`, `depends_on`,
  `Deviations::with/without` (decision 15).
- Rust-only inputs (`param_rust_only`): no cell, skipped by the metadata tests, left out by
  gen_differential.py. Parity (1,149 checks) and the differential data are unchanged; the default
  headline with every correction on is unchanged (2.688 N·m, 93.06 °C, M4 x 14).

"""
p.write_text(s.replace(anchor, anchor + entry, 1), encoding="utf-8", newline="\n")
print("tracker entry added")
EOF
```

- [ ] **Step 3: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml
```

Expected: no output (it breaks lines, never reorders operands).

`cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) run inside it; `cargo fmt --check` must also be clean before the commit.

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a1/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task14.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task14.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/target/gate-task14.log
git -C C:/Users/Cole/source/repos/lsim-mag-a1 checkout -- docs/chebyshev_lambda
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs.

- [ ] **Step 4: Check the default headline one last time**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a1/magcoupling-rs/Cargo.toml --test deviations all_corrections_together_give_the_reviewed_headline 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed` (pull-out 2.688 N·m, hot low 2.285, limit 93.06 °C, clamp M4 x 14: unchanged by every task).

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a1 add magcoupling-rs/README.md magcoupling-rs/src/lib.rs docs/ai/02-system.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md
git -C C:/Users/Cole/source/repos/lsim-mag-a1 commit -F - <<'EOF'
docs(magcoupling): Addendum A-1 status, invariants and open items

README status and crate docs; docs/ai system invariants (Rust-only inputs,
selector defaults), memory (A6 fix and E9 residuals resolved; M3 re-bless,
the E18 modulus question, A-2 carries) and the update tracker.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

Expected: one commit on `magcoupling/addendum-a1`; `git -C C:/Users/Cole/source/repos/lsim-mag-a1 status --short` prints nothing (no PNG under docs/chebyshev_lambda).

---

## Self-review record

A plan critique found no blocking defect and eleven non-blocking findings. This revision fixes all eleven. Nothing else in the plan changed.

Every changed edit, script and test command was replayed with the rest of the plan, and each task's gate passed (Verification record). A few text-only changes were made after that replay started:
- the Files and Interfaces notes of Tasks 2 and 13;
- two README sentences on the `new` side of existing edits: Task 10's E20 row and Task 12's "Invalid inputs" sentence.

None of them moves any `old` text that a later edit matches. The README sentences were mirrored into the scratch crate, and its tests still pass.

1. **E20 checked only the inner ring.** C52 to C55 are the outer blocks' reverse fields, so a weaker outer ring read "OK": B842SH inside B842 gave 93.06 °C. What changed:
   - A new decision row, A13, offers three options. The recommended one, (a), is implemented: each ring is checked with its own grade, Br and rating. The alternatives are (b) the outer ring only and (c) the inner ring only.
   - The ring with the lower magnet limit governs, and the block shows that ring whole: C42, C47 to C61, and the Hcj and beta used. The inner ring wins a tie.
   - The cold side shows the ring with the higher cold limit, and the cold check passes only if both rings pass.
   - Task 10 adds `TemperatureLinks::{outer_br20_T, outer_tmax_lib_C, outer_grade}`, which api.rs fills from Calculator C31, C32 and the outer grade. It also adds a private `ring_demag`, the Rust-only results `demag_ring` and `cold_ring`, and `RING_INNER`/`RING_OUTER`.
   - New tests:
     - `e20_mixed_rings_use_the_weaker_grade`: in both orders, C12 = -3.5174 °C and the verdict reads "CHECK: see the rows above.", and the block shows B842.
     - `e20_checks_both_rings_each_side_from_the_weaker` (unit test): an NdFeB ring beside a ferrite ring.
     - A third E20 probe, B842SH inside B842: C42 1.29 → 1.30 T, C47 150 → 80 °C, C12 92.55 → -3.52 °C, C25 "OK ..." → "CHECK ...".
   - These texts now say which ring is used: E20's title and corrected formula (Task 2), the README row, the input and result help, the module docs, `grades.rs` docs, `03-structure.yaml`, and the Task 14 memory and tracker strings.
   - `e20_the_hcj_and_beta_inputs_override_the_grade_when_selected` now makes both rings manual, because a library ring beside a manual one is now checked too.
   - Identical rings keep the inner ring's block bit for bit. The headline test and every earlier expected value are unchanged.
2. **The coercivity selector and the material picks override assumption inputs.**
   - Rows A2 and A9 now say so, with the measured examples. Row A2 also notes for A-2 that the A3 assumption test must allow for this, and that C45's slider cannot enter ferrite's +0.0035.
   - Task 14 records the same notes in `04-memory.yaml`.
   - E20's Applied entry records the workbook help of C44 ("N42SH ≥ 20 kOe.") and C45 (empty) in `workbook_help`. The port's help for both now says when they apply.
   - The labels stay, for schema parity. The Planned entry keeps `workbook_help` empty, as `planned_entries_change_nothing_yet` requires.
3. **An invalid back-iron code fired `low_saturation`.**
   - `WarningInputs` gains `back_iron_known` (`back_iron.is_some()`), which the low-saturation and plating rules now require.
   - New unit test: `an_unknown_back_iron_fires_no_material_rule`.
   - `a_code_outside_the_choices_gives_nan_not_another_material` now also asserts that no warning fires. The assertion is added in Task 13, where the warnings exist.
4. **The warning tests did not check that each rule stays quiet off its condition.**
   - `each_rule_fires_exactly_on_its_condition` now checks that unplated stainless is quiet.
   - The end-to-end test adds codes 4 to 6 unplated (only `low_saturation` fires) and a steel expansion coefficient typed at 16e-6 /°C (only `cte_mismatch_with_magnets` fires).
   - Review Focus 3 names the typed expansion coefficient.
5. **The 1.7 T threshold was justified as a saturation figure.** `LOW_SATURATION_T`'s doc, the warning's help, row A10 and Task 13's intro now call it the highest design flux density the workbook names for a back-iron steel. Row A9 adds that picking 1018 turns the default wall verdict to "OK".
6. **The 416 expansion coefficient's citation.** The value was already right. The 416 notes and the "Settled here" paragraph now quote the Rolled Alloys line and its footnote.
7. **E19 and E20 did not list every cell their probes change.**
   - E19 now lists all 30 cells. E20 lists all 34, including A13's C42, C47 and C48 and the downstream C101, C104 and C105.
   - A new Task 2 test, `addendum_entries_name_every_cell_their_probes_change`, enforces this for E15 to E20. The M1 entries keep their report-named lists, because their probes change hundreds of downstream cells.
   - A new helper, `probe_cells`, takes the probe setup out of `at_probe`.
8. **The coercivity selector broke the plan's invalid-code rule.** The Global Constraint now states the exception the README already gives: a two-way choice falls through to its else branch, as the workbook's two-way IFs do. `coercivity_source` is the only new selector of that kind. The README's "Invalid inputs" sentence (Task 12) now says the same. The code and its test are unchanged.
9. **Python edits fenced as Rust.** The four `gen_differential.py` edit blocks in Task 1 are now fenced as python. The plan builder now picks the fence language from the file extension, and no other block changed.
10. **A positive beta with no rating gave +inf torque.** With E20 on, C60 = +inf, so the adhesive governs C12, and C61 is now NaN instead of +inf. The README's E20 row says so, and adds that exporters must handle NaN as they handle E13's +inf. New robustness test: `a_positive_beta_without_a_rating_has_no_hot_limit`.
11. **The part rows repeat Br and Tmax next to the grade table.** Task 3's intro now states this repetition and accepts it, and names the test that keeps the two copies from drifting.

Review Focus gains item 6, the mixed rings. The expected test counts in each task and in the Verification record were re-measured from the replay.
