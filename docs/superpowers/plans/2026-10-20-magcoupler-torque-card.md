# Magcoupler slice 3, the torque card: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The Size view gains a read-only card, between the readouts and the drawing, that shows for one fixed step of each Size control (Outside diameter ±1 mm, Overall length ±1 mm, Magnet face gap ±0.1 mm, Magnets per ring ±2) the model's change in hot-low torque, the size changes, the dashboard checks whose level flips and the margin change of each counted check that already fails, with plain words wherever a step has no honest number.

**Architecture:**
- **A pure module** `magcoupling-rs/src/gui/torque_card.rs` (no egui): the card's types, each control stepped through its own slider's mapping, the order of the validity tests, the jump and calibration rules, the flips and margins over the counted dashboard checks, and the card's lines.
- **Shared helpers first:** one number format for the Size tab in `gui/format.rs` (decision 10B); `check_terms` in `gui/guidance.rs` (the production half of the fix tests' `terms()`); `slider_minimum` in `gui/size_controls.rs`; the card's ban list in `engine/explain/plain.rs`; the shared test designs.
- **Wiring:** the card lives inside the existing `ConsequenceCache` (one key: the design shown, the Minimum OD and the clamp pause), in place of the N±2 neighbour lines, which leave the Magnets per ring control (decision 7A).
- **Panel:** a collapsible card with a Why link to the `torque_from_shear` note; it starts closed while the Equation panel is open, in a narrow centre region, or when it would push the drawing under its 160 pt floor; the user's own choice wins for the session; its body scrolls above the drawing's floor.
- **Unchanged:** no engine input or result, no Python-locked row, the clamp section stays paused, Auto is unchanged.

**Tech Stack:** Rust 1.89; egui 0.32.3; the headless egui harness (`magcoupling-rs/src/gui/test_support.rs`); `linkage-sim-rs/scripts/gate.sh`, where gate 12 runs the Python oracle from `reference/magcoupling-py/.venv`; the gui-smoke Playwright workflow (`.claude/workflows/gui-smoke.js`).

**Spec:** `docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md` (revision 5, commit dae024f; approved by the owner "as the reviews leave it" for the night run of 2026-10-19). The owner's decisions and the controller's rulings are in `.superpowers/sdd/2026-10-19-magcoupler-slice3/owner-decisions.md`. The research, contract, drafts, integration record and critique are in the same gitignored folder.

**How this plan was made:**
1. A binding contract assigned the tasks.
2. Stage 1: the format and prerequisites drafter built T1-T2 in a scratch prototype; the G-3 and smoke drafter wrote T7-T8; the docs and release drafter wrote T9-T10.
3. Stage 2: two drafters wrote T3-T4 (the card module) and T5-T6 (wiring and layout) on copies of that prototype.
4. An integrator built T1-T6 and T8 in one prototype and measured each task commit; a consistency and physics critic checked the drafts against the spec.
5. The CLAUDE.md plan review (Architecture, Code Quality, Tests, Performance) ran as four reviewer agents on the drafts and the integrated code. The controller took every recommended option on the owner's behalf (owner-decisions.md, "Spec approval and the night run"), recorded below for the owner to re-rule at the STOP.
6. A second integration built the adopted critique amendments and plan-review options into the same chain and measured it again. Every amendment and ruling is folded into the task text, and a replay check confirmed that the plan's blocks rebuild the measured commits byte for byte.

Where a line number has moved because of an earlier task, trust the code and keep the intent.

## Global Constraints

- **Implementer model:** BL-073 is `risk: physics`, so every task and every review runs on the session model (CLAUDE.md section 5; precedence).
- **Commit trailers**, exactly:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
  ```
- **Commit messages:** written with the Write tool to `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T<n>.txt`, or `T<n>-fix-<k>.txt` and the names T7 and T10 give. Then `git commit -F <file>`. Never pipe a heredoc.
- **Staging:** files by name only. Never `git add -A` or `git add .`. Never stage the owner's untracked `docs/analyses/2026-05-28-press-4bar-analysis.md`.
- **TDD:** red, then green. A task that adds tests only shows its red by planted faults, each undone afterwards.
- **Before each commit**, run all of these:
  - the touched suites;
  - `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`;
  - the same without `--features app`;
  - `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`.
- **Docs ship with the code,** in the task that makes them stale. T9 sweeps what is left and quotes texts read from the code, never drafts.
- **Never edit:**
  - Python-locked rows (`tests/python_schema.rs`), `reference/`, `tests/data/python_schema.json` or the differential files;
  - the committed note sign-off sheet.
- **No engine input or result is added,** so `tests/data/input_schema.json` does not change.
- **Gate 1** rewrites `docs/chebyshev_lambda/*.png`. Restore them with `git checkout -- docs/chebyshev_lambda`.
- **The clamp section stays paused** (`engine::clamps::PAUSED = true`):
  - no on-screen text says "clamp" while paused; the card names the clamp's check only on an unpaused panel (test 7a);
  - the paused section's texts are not edited.
- **The card's decisions** (owner, 2026-10-19):
  - steps (3A): Outside diameter ±1 mm, Overall length ±1 mm, Magnet face gap ±0.1 mm, Magnets per ring ±2, each through its own slider's mapping;
  - rows in the Size controls' fixed order, each stating its step; no sorting (6A);
  - read-only, labels through `label_for`, no links in rows, no apply (11A);
  - words on the failing side only, the other side keeps its number; no numbers when the design shown does not fit, has no prediction or a hot-low or pull-out not above zero (9A, T-12);
  - a length side whose overlapping magnet length changes by anything other than the step reads `JUMP` (O-1, widened);
  - a step that changes the calibration factor shows the true number with `CALIBRATION_CHANGE` (16A), never explained as physics;
  - no Temperature limits margin (O-2, T-13);
  - one number format for the Size tab (10B): one decimal of percent; N·m to the fewest decimals that keep the true sign and are not zero, capped at two more than the usual (T-9); a row's two N·m at one precision (T-14).
- **On-screen text** follows the spec's traps (section 4.4):
  - no ranking: `CARD_RANKING_WORDS` (strongest, biggest, best, most, lever, dominant, ranks) never appear in a card text;
  - a step is never extrapolated; a jump never reads as a slope; a near-zero row is never "no effect" or "free";
  - no ordering of the three torques, no "guaranteed", no "protects";
  - Auto wording never says the torque rises where Auto would resize instead;
  - plain words, glyph-safe, `plain_text_violations` empty;
  - every new string is a named constant; its words are drafts until T7's G-3 review, and tests read the constants.
- **Performance (13A):** the card computes live inside `ConsequenceCache`, once per changed design; a browser drag frame's p95 must stay at or under 16.7 ms (T10).
- **Release:** STOP for the owner's review before any merge or push, and ask for deploy authorization there. A push to `main` deploys.

## Review Focus

These five inputs or conditions are the most likely to bite a person using the card, and the spec gives none of them a test of its own. Each has its test in the task named, except item 5, which waits for the owner's rule.
1. **A drag at the frame-slowest design.** With Auto on the ring size, every frame builds the design shown through two geometries, and a drawn frame there costs more than at the smoke design (measured in the browser: a median of 1.9 ms against 1.3-1.4 ms, before the card). A person dragging there expects the same smooth frames. T10 Step 7 drags in the browser at (a) with Auto on the ring size as well as at the smoke design, timing only the frames that draw; either p95 over 16.7 ms runs Step 7b (ruling C-6, plan review 4.1A-4.3A).
2. **A drag with Auto on.** Every frame of a drag changes the design shown, and the card must compute once for each design it draws (13A), never twice. T5: `a_drag_computes_the_card_once_for_each_design_it_draws`.
3. **A design shown past a slider's end,** from a design file or a link: a face gap of 5.5 or 6.0 mm, an axial override of 60 mm, the Minimum OD at its top. Both sides of that row read "at the end of its range", never an invented number. T3: `a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range`; T7's G-3 brief probes the words for the owner.
4. **Two sides that read alike, and the boss plateau.** A row whose two sides both read -0.005 N·m (-0.3 %) while only one crosses the Required torque flips Hot-low torque on that side only (T-14). Where the boss sets the outside diameter, one step reads the slider's own lift, +7.55 % (R-13). Both are on the STOP sheet, and both join the test table, so every table test runs on them (plan review 2.2A). T4: `a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required` and `where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift`.
5. **A design file's degenerate values.** A design file or a link can carry any finite value, and the card runs on the first view of every design it opens. At a count of 2 the polygon degenerates and the card quotes "Space claim: Rotating diameter -3266247870639104.00 mm"; a calibration coefficient of 1e-9 or 1e300 gives percents of eleven or more digits and lines thousands of characters long; a variation just under 1 gives sides of "-0.00 N·m" beside a true percent. A person expects no such number, and the app must never fail on load. T4: `the_card_never_panics_at_the_values_a_design_file_can_carry` holds the no-panic and plain-words part at 47 such values (plan review 1.1A). What the numbers read there is the owner's rule (ruling C-7): T7's G-3 brief probes the class and T10's STOP sheet asks the owner.

## Controller rulings on the critique (binding; folded into the tasks)

- **C-1 (I-1 to I-16, the integrator's amendments).** Adopted as measured. Where an integration amendment and a critique amendment fix the same defect, the integration text wins, because it was measured in the integrated prototype:
  - I-1 over A-1 (the `gui::torque_gap_text` re-export, in T1);
  - I-2 to I-6 over A-2 (T5's Old blocks are T3's and T4's text), with A-2's deletion of the fact "If Task 3's text differs, remove what Task 4 leaves";
  - I-7, which is A-3;
  - I-8 over A-4 and A-5 (T5 rewrites the stale "which call them" lines and the whole cache clause of 02-system :116);
  - I-9 to I-12 over A-6's figures;
  - I-13 over A-8's skip-if strings;
  - I-16 over A-9 (Step 7b's card hunk, exact).
- **C-2 (A-7, and A-8's remainder; must).**
  - A-7: T10 Step 1 expects at least 10 commits since dae024f: the plan's commit and the nine commit steps of T1-T9; T7's fix commits add to them.
  - Every marker in T9 Step 10 is a string its edit or the earlier task's text leaves in the docs: S3's marker is the text T5 writes ("one key in three parts, slice 3").
  - R11 and C4 are deleted from T9: T3, T4 and T5 own those lines, and `COLD_HIGH` and `WALL_NEEDED` live in `gui/guidance.rs`.
- **C-3 (A-4's guard).** T9 Step 4's docs sweep gains "which call them", so the stale phrase cannot come back. If it matches a correct docs line at T8, the sweep takes a narrower phrase that matches none.
- **C-4 (the should-amendments that change code; adopted for DRY and for coverage).**
  - A-10: the screen dump takes `card_designs()`'s keys (spec 8.6, "the same keys"), and so do `check_shipped.py`'s quotes and the STOP sheet's probes.
  - A-12: one `FACE_GAP`, guidance's, made `pub(crate)`; the card and the test table import it.
  - A-13: `sizing_range(inputs, variable, minimum_diameter_mm)` moves to `gui/sizing.rs` as `pub(crate)`; the panel's method and the test table's `auto_shown` both call it.
  - A-14: one `TORQUE_DECIMALS = 2`, in `gui/format.rs`; `torque_gap_text` and the card read it. The spec's literal test pins stay literals.
  - A-24: one `smoke_inputs()` in `gui/test_support.rs`; the table's (b), T5's test 10, RF-2's test and T6's `smoke_design()` build from it.
  - A-11 is already met by I-3, which builds the count-row tests on `card_at` and `side_at`.
  - A-19, A-20 and A-21: RF-2's test in T5, RF-3's in T3, and RF-4's two in T4.
  - Every count, filter figure, red and Old block they move was measured again in the second integration.
- **C-5 (the should-amendments that change text).**
  - A-15, A-16, A-17 and A-22 are adopted as written.
  - A-23: the plan review's section below is titled "## Plan review rulings". T10 Step 10.3 reads the build ledger at `.superpowers/sdd/2026-10-20-magcoupler-torque-card/progress.md` (the SDD workspace of this plan's file name), and falls back to `.superpowers/sdd/2026-10-19-magcoupler-slice3/progress.md` before it stops.
- **C-6 (A-18, rewritten by plan review 4.1A, 4.2A and 4.3A: the frame check measures, it does not project).**
  - T10 Step 6's probe times (a), (b), (e), (6 magnets) and (Auto ring) in rounds, every design once per round, on a quiet machine (no cargo or rustc running, the gate finished), and prints the share link of (a) with Auto on the ring size.
  - T10 Step 7 drags in the browser at the smoke design (gui-smoke with `frame_check`) and at (a) with Auto on the ring size, the frame-slowest design, each with T8's frame timer, which keeps only the frames that draw. Either p95 over 16.7 ms runs Step 7b.
  - The critique's "+2.3 ms" came from the machine's slow regime, which starts a few seconds into every run; the projection script is dropped. The STOP sheet quotes the measured figures.
- **C-7 (RF-5, widened by plan review 1.1A to a design file's degenerate values).**
  - No rule is added: 9A names only fit, prediction, hot-low and pull-out, and a new rule is the owner's.
  - T4's robustness test holds that the card is computed, repeatable and in plain words at 47 such values, at both back irons and at the Minimum OD's top.
  - T7's G-3 brief probes the class (A-16's probe 12, 1.1A's probe 13), and T10's STOP sheet shows the measured quotes (a count of 2, a calibration coefficient of 1e-9 or 1e300, a variation just under 1) and asks the owner (A-22, widened).
  - The critic's guard test for a count of 2, red today, is not added; it lands with the owner's rule.
- **C-8 (the drafters' deviations).** All adopted as drafted, except where C-1 to C-7 say otherwise: DA-1 to DA-10, DV-B1 to DV-B11, DC-1 to DC-14, D-G-1 to D-G-8, and DEV-1 to DEV-8 (I-8 moves DEV-4 into T5). The Deviation index after the tasks renumbers them from PD-7.
- **C-9 (the drafters' questions).** Each is ruled as its drafter recommended:
  - `WALL_NEEDED` and `COLD_HIGH` live in `gui/guidance.rs` beside `check_terms`.
  - The self-cleaning `expect` attributes stay; T4 deletes them.
  - T5 and T9 take their Old text after T1 and T2.
  - The parse error that `docs/ai/02-system.yaml` already has at line 15 is not fixed in this slice. Docs checks parse each edited item alone.
  - Test 1a imports guidance's check paths once T4 makes them `pub(crate)`.
  - `CardDesign`, `card_designs()`, `card_at` and `side_at` live in `gui/test_support.rs`.
  - The review dump draws each design in its own Auto state, and all three Auto states at the four designs whose form differs under Auto.
  - G-3 findings on the owner's 9A words, on reused shipped strings, on the number format or proposing a new rule go to the STOP sheet; T7 does not apply them.
  - T8 owns README :618-631 and :697.
  - T10 runs gui-smoke with `{"frame_check": true}` on the local server, and with `{"url": "https://colesorkness.com"}` after the deploy.
  - The public surface outside the crate:
    - `CARD_HEADING`, `card`, `Control`, `Side`, `TorqueCard::{lines, rows}` and `Change.hot_low_Nm`;
    - `ConsequenceCache::get(inputs, results, minimum_diameter_mm, clamps_paused)`, with `clamps_paused` last.
  - T10's timing covers (a), (b), (e), (6 magnets) and (Auto ring), in rounds (C-6).
  - No document cites a red Temperature limits case: `verdict_level` makes the check amber at worst.
- **C-10 (the plan review).** The CLAUDE.md plan review ran as four reviewer agents on this plan (BIG CHANGE: Architecture, Code Quality, Tests, Performance, at most four issues each). The controller took every recommended option on the owner's behalf under the night-run approval. "## Plan review rulings" below records each one, and the owner re-rules any of them at the STOP.

## Plan deviations from the spec

- **PD-1 (C-4, plan review 2.3A).** Shared names the spec does not name, kept once for DRY:
  - guidance's `FACE_GAP`, made `pub(crate)`;
  - `sizing_range` in `gui/sizing.rs`;
  - `TORQUE_DECIMALS` in `gui/format.rs`;
  - `smoke_inputs()` and `boss_plateau()` in `gui/test_support.rs`;
  - `SliderMapping` in `gui/size_controls.rs`: one mapping per Size slider, applied by the controls, the card's step, test 1's oracle and the test table. The count side sets the count through `InputSet::set` of a whole number, as the slider does, where spec 3.3 names `FreeVariable::MagnetsPerRing.apply`: the same design.
- **PD-2 (C-4, plan review).** Tests beyond the spec's list:
  - RF-2's drag test (T5), RF-3's past-the-end test (T3) and RF-4's two tests (T4);
  - the card at a design file's degenerate values (T4, 1.1A) and a count step to a design without size controls (T4, 2.1A);
  - `SliderMapping`'s own test (T2, 2.3A), a Required edit reaching the card in its own frame (T5, 4.4A) and a window resize deciding the card again with no user choice (T6, 3.3A);
  - stronger checks in tests 10, 22 and 24: the Minimum OD as the key's third part (3.2A), the drawing's floor beside the Equation panel (3.1A), and a body that really scrolls (3.4A).
- **PD-3 (C-6, plan review 4.1A to 4.3A).** The release measures more than spec 7.2's smoke drag:
  - T8's frame timer keeps only the frames that draw;
  - T10 Step 6 times every design in rounds on a quiet machine;
  - T10 Step 7 drags in the browser at (a) with Auto on the ring size as well as at the smoke design, each p95 at or under 16.7 ms.
- **PD-4 (C-1, I-1).** T1 adds the `gui::torque_gap_text` re-export: spec 8.6 test 27 and 9.2 name it, and no task owned it.
- **PD-5 (plan review 2.2A).** T-14's one-flip design and R-13's boss plateau join `card_designs()`, the table spec section 8 builds from section 2.1's designs. The screen dump gains the table's two count overflows, and one script checks the dump's cards against the review dump's, design by design.
- **PD-6 (plan review 4.4A).** The panel builds the design shown once per view, after the Required edit, and hands it to the card and the drawing. Spec 6.1 has the card call `shown_inputs()` itself. The design is the same, and the frame builds it once less.
- **The drafters' deviations** are in the Deviation index after the tasks, from PD-7, each with its ruling.

## Plan review rulings (controller for the owner, 2026-10-20: BIG CHANGE, four reviewer agents, every recommended option)

The CLAUDE.md plan review ran as four fresh-context reviewers on the drafts, the amendments and the integrated code: Architecture, Code Quality, Tests and Performance, at most four issues each. Under the night-run approval the controller took every recommended option, and each is folded into the tasks. The owner re-rules any of them at the STOP.

| # | Question | Answer |
|---|---|---|
| 1.1 | A design file or a link can carry any finite value, and the card runs on every link's first view. Guard the card against failing there, and ask the owner about the whole class of degenerate values, not a count of 2 alone? | **1.1A.** T4 adds `the_card_never_panics_at_the_values_a_design_file_can_carry` (47 values, both back irons, the Minimum OD at its top) with a planted fault. T7's G-3 brief names the class (probe 13). T10's STOP sheet widens A-22's bullet to the class with the measured quotes. No new rule (C-7). |
| 1.2 | The STOP sheet's card lines come from the screen dump's own retyped list of designs. Check it against the table the tests pin? | **1.2A**, met by 2.2A's one check script, run at T9 and at T10 Step 9. |
| 2.1 | If the frame check misses, Step 7b would change spec 2.2's size-change rule on a count side whose stepped design has no size controls (reachable from a design file). Keep the rule and pin it? | **2.1A.** Step 7b reads `size_of` on the diameter, length and gap rows and `dimensions` on the count row. T4 adds `a_step_to_a_design_without_size_controls_draws_no_size_change` with a planted fault, and Step 7b's guard names it. |
| 2.2 | T-14's one-flip design and R-13's boss plateau live outside the one test table, and nothing checks the screen dump's copy of the table. One table and one check? | **2.2A.** Both join `card_designs()` in T4 (`boss_plateau()` in `test_support`). RF-4's tests read them by key, and test 1a pins the plateau. The screen dump gains the table's two count overflows, and T9 and T10 Step 9 run one script that checks the dump's cards against the review dump's (PD-5). |
| 2.3 | The Size sliders' mappings are written twice in production code and twice more in the tests. One mapping? | **2.3A.** `SliderMapping` in `gui/size_controls.rs` (T2, beside `slider_minimum`, with its own test), applied by `controls_ui`, the card's step, test 1's oracle and the test table (T3). The count side sets the count as the slider does (PD-1). |
| 3.1 | egui raises a scrolling body to 64 points, so beside the Equation panel a card the user opened pushes the drawing under its 160-point floor. Keep the floor? | **3.1A.** `.min_scrolled_height(cap)` after `.max_height(cap)`, with a comment naming egui's default. Test 22 asserts the drawing's floor at every step, with a planted fault. |
| 3.2 | No panel test reaches the Minimum OD: the card could ask the cache with another minimum and every test would pass. Pin it? | **3.2A.** Test 10 gains the key's third part: a Minimum OD set computes the card once, and the drawn Outside diameter row moves to "at the end of its range". Planted fault. |
| 3.3 | No test resizes the window while the user has made no choice. Pin the screen half of the card's view key? | **3.3A.** T6 adds `the_card_is_decided_again_when_the_window_is_resized_with_no_user_choice`, with a planted fault. |
| 3.4 | Test 24 is named for the body's scrolling but never scrolls. Make it scroll? | **3.4A.** Test 24 turns the mouse wheel over the body until the last row is whole in view, the drawing keeping its floor. Planted fault. |
| 4.1 | T8's timer times every display refresh: in a measured drag, 72 % of the callbacks drew nothing. Time only the frames that draw? | **4.1A.** One `FRAME_TIMER` constant marks a callback by a WebGL draw call. T8's stub and the app.rs test pin it, red first. |
| 4.2 | C-6's projection rests on a slow-regime artifact and on the wrong design. Measure instead? | **4.2A.** T10 Step 7 also drags in the browser at (a) with Auto on the ring size, the frame-slowest design, from a share link the probe prints. Either p95 over 16.7 ms runs Step 7b. The projection script is dropped (C-6 rewritten). |
| 4.3 | T10 Step 6 times the designs one after another on a machine that halves its speed mid-run, and a cargo build may run beside it. Time in rounds on a quiet machine? | **4.3A.** The probe times every design once per round for five rounds. Steps 6 and 7 start with no cargo or rustc running and the gate finished. Step 9.2's background build starts after Step 7. |
| 4.4 | The card builds the design shown once more every frame, and spec 3.5's same-frame rule has no test. Build it once? | **4.4A.** The panel builds the design shown once, after the Required edit, and hands it to the card and the drawing. T5 adds `a_required_edit_reaches_the_card_in_its_own_frame`, with a planted fault (PD-6). |

No option was declined: none contradicts the spec, an owner decision or a ruling once C-6 and C-7 were rewritten for 4.2A and 1.1A. The plan review files are kept in the drafting workspace.

## Before Task 1 (controller)

The plan is committed once, with its review folded in, on `magcoupling/slice3-torque-card`. Every task's `git status --short` check expects only the owner's `?? docs/analyses/2026-05-28-press-4bar-analysis.md`. The build runs the plan with superpowers:subagent-driven-development:
1. The workspace is the SDD workspace of this plan's file name, `.superpowers/sdd/2026-10-20-magcoupler-torque-card/`. Its ledger, `progress.md`, starts with `# SDD ledger — plan: docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md`.
2. Each task's brief comes from the skill's `task-brief` script. The implementer reads its brief and this head, never the whole plan.
3. Progress and rulings go into the build ledger, not into this plan. A plan edit made anyway, a ticked checkbox included, is committed by name with its own message file before the next task's status check.

---

## Tasks

### Task 1: The Size tab's number format (decision 10B): `gui/format.rs`, the readout's gap and the moved decimals search

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec section 5 in full, T-9 and the formatter half of T-14, and tests 12, 13 and 14. One number format for the Size tab, in `gui/format.rs`:
- `fewest_decimals_apart` and `MOST_DECIMALS` move from `gui/guidance.rs` unchanged (the search `pub(crate)`, `MOST_DECIMALS` private to `format.rs`). `band_quotes`, `window_quotes` and `input_text_below` (spec 5's `:2127` caller) import them.
- New: `CHANGE_EXTRA_DECIMALS = 2`; `TORQUE_DECIMALS = 2`, the decimals a torque change starts from (the readout's gap here, the card's N·m from Task 4: one constant, critique A-14); `reads_not_zero` (a finite number that is not zero, the one test of "a digit that is not zero"); `change_decimals` (the fewest decimals from `from` up to the cap that read a digit that is not zero, through `fewest_decimals_apart`, else `from`: T-9); `signed_fixed`; `signed_change`; and `shared_change_decimals`, T-14's rule for a row's two N·m (the larger of the sides' `change_decimals(.., 2)`, each N·m then drawn through `signed_fixed`). The spec names no function for T-14's rule; test 13 pins it here, and Task 4's `lines` calls it (plan deviation PD-8).
- `signed_percent` reads one decimal: `fraction × 100` through `signed_fixed(.., 1)`, so an exact zero of either sign reads "+0.0 %" and a value that rounds to zero keeps its true sign (code review 5 SF-4).
- `torque_gap_text` takes the format: a margin `signed_change(gap, TORQUE_DECIMALS)`, a shortfall `fixed(|gap|, change_decimals(gap, TORQUE_DECIMALS))`, the percent `signed_percent(gap / required)` with the gap's own sign, so the `replacen('+', "-", 1)` re-signing goes; its doc comment follows. The prototype's probe at spec 5's designs reads (a) "short 0.21 N·m (-8.6 %)", (b) "margin +0.00 N·m (+0.0 %)", (c) "short 1.06 N·m (-42.3 %)", (e) "short 0.83 N·m (-33.1 %)" and (e2) "margin +0.17 N·m (+11.5 %)", spec 5's table exactly.
- `gui/mod.rs` re-exports `torque_gap_text` as `gui::torque_gap_text`: the slice's screen dump (Task 9) reads it from outside the crate, and `size_controls` is a private module (spec 8.6 test 27 and 9.2 name the re-export; plan deviation PD-4).

Two facts the implementer meets on the way:
- `shared_change_decimals` has no production caller until Task 4, so the lib build would warn `dead_code`, an error under clippy's `-D warnings`. It carries `#[cfg_attr(not(test), expect(dead_code, reason = ...))]`. When Task 4's `lines` calls it, the compiler reports the expectation unfulfilled (checked in the prototype: "this lint expectation is unfulfilled"), so Task 4 must delete the attribute in the same commit (plan deviation PD-9). Task 2 puts the same attribute on `check_terms`.
- The neighbour lines under Magnets per ring read `signed_percent` through `Neighbour::line` until Task 5 moves them into the card. From this commit they read one decimal too: "At 8: pull-out 1.97 N·m (-26.8 %), cup OD 42.4 mm" (was "(-27 %)"), 10B's one percent rule for the Size tab.

Line numbers are at dae024f (whose code is 69fa505's); trust the code. Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`. Every code block is rustfmt's layout: in the plan-time prototype, replaying this task's Old/New blocks on 69fa505 reproduces the task's files byte for byte, and then both suites, both clippy runs and fmt are green.

**Files:**
- Modify: `magcoupling-rs/src/gui/format.rs` (the moved search, the new functions, `signed_percent`; test 12 in place of the whole-percent test, test 13, the search's test), `magcoupling-rs/src/gui/guidance.rs` (the search leaves; its import; the what-moves-what comment; the neighbour line's test), `magcoupling-rs/src/gui/size_controls.rs` (`torque_gap_text` and its doc; test 14: the gap tests and the readout texts' test), `magcoupling-rs/src/gui/mod.rs` (the re-export `gui::torque_gap_text`)
- Modify (docs): `magcoupling-rs/README.md` (the gap's format in the Size-tab paragraph, `:203`; the `src/gui/` module list's `format.rs`, `:155`; the `src/gui/format.rs` test row, `:695`), `docs/ai/03-structure.yaml` (the `gui` line's `format.rs` clause and, for the re-export, its `size_controls.rs` clause, `:68`), `docs/ai/02-system.yaml` (`:116`'s `torque_gap_text` clause: the gap's number format, spec 9.2)
- Test: `magcoupling-rs/src/gui/format.rs` (`a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign`, test 12; `a_change_keeps_its_sign_and_shows_a_digit_that_is_not_zero`, test 13; `the_decimals_search_takes_the_fewest_from_its_start_else_the_most`); `magcoupling-rs/src/gui/size_controls.rs` (`the_gap_reads_short_or_a_margin_against_the_requirement` and `the_readout_texts_have_glyphs_and_plain_words`, test 14); `magcoupling-rs/src/gui/guidance.rs` (`the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not`)

**Interfaces:**
- Consumes: `gui::format::{fixed, non_finite_text}` (unchanged).
- Produces, in `magcoupling-rs/src/gui/format.rs`, every item `pub(crate)`:
  - `fn fewest_decimals_apart<T>(from: usize, quotes: impl Fn(usize) -> T, apart: impl Fn(&T) -> bool) -> (usize, T)` (moved, unchanged; its fallback `const MOST_DECIMALS: usize = 15` private);
  - `const CHANGE_EXTRA_DECIMALS: usize = 2`;
  - `const TORQUE_DECIMALS: usize = 2` (the Size tab's N·m decimals: `torque_gap_text` reads it, and Task 4's card);
  - `fn reads_not_zero(text: &str) -> bool`: Task 4's `shows(change, from)` is `change.is_finite() && reads_not_zero(&fixed(change, from + CHANGE_EXTRA_DECIMALS))`;
  - `fn change_decimals(x: f64, from: usize) -> usize`;
  - `fn signed_fixed(x: f64, decimals: usize) -> String`;
  - `fn signed_change(x: f64, from: usize) -> String`;
  - `fn shared_change_decimals(changes: &[f64], from: usize) -> Option<usize>` (`None` for a row with no number): Task 4 draws a row's two N·m as `signed_fixed(change_Nm, decimals)` at `shared_change_decimals(&[the number sides' change_Nm], TORQUE_DECIMALS)`, and deletes the `expect(dead_code)` attribute;
  - `fn signed_percent(fraction: f64) -> String`, now to one decimal.
- `gui::size_controls::torque_gap_text(hot_low_Nm: f64, required_Nm: f64) -> (String, bool)`: the signature unchanged, the new format.
- `gui::torque_gap_text`, a `pub use size_controls::torque_gap_text;` in `gui/mod.rs` (for the slice's screen dump, spec 8.6 test 27).

- [ ] **Step 1: Write the tests of the new format that compile at HEAD**

(a) `magcoupling-rs/src/gui/format.rs`, the test module's last test: the whole-percent test is replaced by spec test 12 (its values are the spec's deliberate literal pins, plus the old test's three to a tenth).

Old:
```rust
    #[test]
    fn a_change_reads_as_a_whole_signed_percentage_never_minus_zero() {
        assert_eq!(signed_percent(0.1121), "+11 %");
        assert_eq!(signed_percent(-0.2679), "-27 %");
        assert_eq!(signed_percent(1.1279), "+113 %");
        assert_eq!(signed_percent(0.0), "+0 %");
        // A small fall rounds to zero: no minus sign on a zero.
        assert_eq!(signed_percent(-0.004), "+0 %");
        assert_eq!(signed_percent(-0.0), "+0 %");
        assert_eq!(signed_percent(f64::NAN), "NaN %");
        assert_eq!(signed_percent(f64::NEG_INFINITY), "-inf %");
    }
}
```

New:
```rust
    #[test]
    fn a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign() {
        // Decision 10B (slice 3 spec section 5): one decimal of percent, the Size tab's one
        // percent rule, on the torque card and the readout's gap alike.
        assert_eq!(signed_percent(0.0349), "+3.5 %");
        assert_eq!(signed_percent(-0.0037), "-0.4 %");
        assert_eq!(signed_percent(-0.0028), "-0.3 %");
        assert_eq!(signed_percent(0.1121), "+11.2 %");
        assert_eq!(signed_percent(-0.2679), "-26.8 %");
        assert_eq!(signed_percent(1.1279), "+112.8 %");
        // A change that rounds to zero keeps its true sign, so a small fall never reads as a
        // rise ...
        assert_eq!(signed_percent(-0.0004), "-0.0 %");
        assert_eq!(signed_percent(0.0004), "+0.0 %");
        // ... while an exact zero of either sign reads "+" (the house rule; `{:+.1}` alone
        // prints -0.0 as "-0.0").
        assert_eq!(signed_percent(0.0), "+0.0 %");
        assert_eq!(signed_percent(-0.0), "+0.0 %");
        assert_eq!(signed_percent(f64::NAN), "NaN %");
        assert_eq!(signed_percent(f64::INFINITY), "+inf %");
        assert_eq!(signed_percent(f64::NEG_INFINITY), "-inf %");
    }
}
```

(b) `magcoupling-rs/src/gui/size_controls.rs`: the gap tests of spec 5 (test 14) take the new texts, each the value the new rule gives ("short 0.001 N·m (-0.0 %)" was "short 0.00 N·m (-0 %)"), plus a margin at three decimals and a shortfall past the cap. `a_requirement_that_is_no_positive_torque_drops_the_percent` is unchanged.

Old:
```rust
    #[test]
    fn the_gap_reads_short_or_a_margin_against_the_requirement() {
        assert_eq!(
            torque_gap_text(2.21, 2.5),
            ("short 0.29 N·m (-12 %)".to_owned(), false)
        );
        assert_eq!(
            torque_gap_text(2.9, 2.5),
            ("margin +0.40 N·m (+16 %)".to_owned(), true)
        );
        // The engine's hot minimum check passes at equality (it fails `hot_low < required`).
        assert_eq!(
            torque_gap_text(2.5, 2.5),
            ("margin +0.00 N·m (+0 %)".to_owned(), true)
        );
        // A shortfall too small to show still reads short, with its sign.
        assert_eq!(
            torque_gap_text(2.499, 2.5),
            ("short 0.00 N·m (-0 %)".to_owned(), false)
        );
    }
```

New:
```rust
    #[test]
    fn the_gap_reads_short_or_a_margin_against_the_requirement() {
        // Decision 10B: N·m to the fewest decimals, from two, that read a digit that is not
        // zero, and one decimal of percent with the gap's own sign.
        assert_eq!(
            torque_gap_text(2.21, 2.5),
            ("short 0.29 N·m (-11.6 %)".to_owned(), false)
        );
        assert_eq!(
            torque_gap_text(2.9, 2.5),
            ("margin +0.40 N·m (+16.0 %)".to_owned(), true)
        );
        // The engine's hot minimum check passes at equality (it fails `hot_low < required`).
        assert_eq!(
            torque_gap_text(2.5, 2.5),
            ("margin +0.00 N·m (+0.0 %)".to_owned(), true)
        );
        // A shortfall too small for two decimals shows its digit, and its percent keeps its
        // sign, so a design a hair under its requirement never reads like a pass.
        assert_eq!(
            torque_gap_text(2.499, 2.5),
            ("short 0.001 N·m (-0.0 %)".to_owned(), false)
        );
        assert_eq!(
            torque_gap_text(2.504, 2.5),
            ("margin +0.004 N·m (+0.2 %)".to_owned(), true)
        );
        // Past the cap (two more decimals, T-9) a gap reads two decimals: float noise is not a
        // digit to show. The word and the percent's sign still say which side it is on.
        assert_eq!(
            torque_gap_text(2.49999, 2.5),
            ("short 0.00 N·m (-0.0 %)".to_owned(), false)
        );
    }
```

`the_readout_texts_have_glyphs_and_plain_words` (test 14's other half) holds the two new forms to the glyphs and the plain-words guard too:

Old:
```rust
            torque_gap_text(2.21, 2.5).0,
            torque_gap_text(2.9, 2.5).0,
            torque_gap_text(2.0, 0.0).0,
```

New:
```rust
            torque_gap_text(2.21, 2.5).0,
            torque_gap_text(2.9, 2.5).0,
            torque_gap_text(2.0, 0.0).0,
            // Decision 10B's new forms: an exact zero and a shortfall at three decimals.
            torque_gap_text(2.5, 2.5).0,
            torque_gap_text(2.499, 2.5).0,
```

(c) `magcoupling-rs/src/gui/guidance.rs`, `the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not`: the neighbour line's percent was pinned by the old whole-percent rule (`{:+.0}` of a rounded value). It now reads through `signed_percent`, whose own tests pin the rule, so the test pins the line's template and not the rule twice. (Task 5 moves this test to the card's tests.)

Old:
```rust
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert_eq!(
            lines[1],
            format!(
                "At {}: pull-out {:.2} N·m ({:+.0} %), cup OD {:.1} mm",
                npole - 2,
                fewer.model.pullout_Nm,
                (100.0 * (fewer.model.pullout_Nm / results.model.pullout_Nm - 1.0)).round(),
                fewer.model.cup_od_mm
            )
        );
```

New:
```rust
        assert_eq!(lines[0], RING_SIZE_STAYS);
        // The change as the Size tab writes every percent (decision 10B: `signed_percent`,
        // one decimal, pinned in `gui::format`'s tests).
        assert_eq!(
            lines[1],
            format!(
                "At {}: pull-out {:.2} N·m ({}), cup OD {:.1} mm",
                npole - 2,
                fewer.model.pullout_Nm,
                signed_percent(fewer.model.pullout_Nm / results.model.pullout_Nm - 1.0),
                fewer.model.cup_od_mm
            )
        );
```

- [ ] **Step 2: Run them and see the format's two tests fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign the_gap_reads_short_or_a_margin_against_the_requirement the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not the_readout_texts_have_glyphs_and_plain_words`

Expected:
```text
test gui::format::tests::a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign ... FAILED
test gui::size_controls::tests::the_gap_reads_short_or_a_margin_against_the_requirement ... FAILED
test gui::guidance::tests::the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not ... ok
test gui::size_controls::tests::the_readout_texts_have_glyphs_and_plain_words ... ok
thread 'gui::format::tests::a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign' panicked at src\gui\format.rs:299:9:
  left: "+3 %"
 right: "+3.5 %"
thread 'gui::size_controls::tests::the_gap_reads_short_or_a_margin_against_the_requirement' panicked at src\gui\size_controls.rs:421:9:
  left: ("short 0.29 N·m (-12 %)", false)
 right: ("short 0.29 N·m (-11.6 %)", false)
test result: FAILED. 2 passed; 2 failed; 0 ignored; 0 measured; 930 filtered out
```
The two that pass are kept valid by the change, not red tests: the neighbour pin reads `signed_percent` itself, and the new gap forms have glyphs today.

- [ ] **Step 3: Write test 13 and the decimals search's test**

`magcoupling-rs/src/gui/format.rs`, after test 12 (the end of the file): spec test 13 (`signed_change(.., 2)`, `change_decimals` at the cap, `signed_fixed`, a row's shared decimals with T-14's two rows), plus edges the spec leaves implicit: zero and non-finite values read `from` whatever it is, a `from` past what the search covers reads itself, and a size change from one decimal (spec 9a's 0.0046 and 6e-5). The search's own test pins its fallback, which `change_decimals` guards against.

Old:
```rust
        assert_eq!(signed_percent(f64::NEG_INFINITY), "-inf %");
    }
}
```

New:
```rust
        assert_eq!(signed_percent(f64::NEG_INFINITY), "-inf %");
    }

    #[test]
    fn a_change_keeps_its_sign_and_shows_a_digit_that_is_not_zero() {
        // Decision 10B: a change to the fewest decimals, from its own (N·m 2), that keep its
        // true sign and read a digit that is not zero, never "+0.00" for a fall or "-0.00" for
        // a rise; at most CHANGE_EXTRA_DECIMALS more (T-9), so float noise reads its first
        // decimals with its true sign.
        assert_eq!(CHANGE_EXTRA_DECIMALS, 2);
        for (x, text) in [
            (0.2010, "+0.20"),
            (-0.0063, "-0.01"),
            (-0.0046, "-0.005"),
            (-0.00012, "-0.0001"),
            (0.0, "+0.00"),
            (-0.0, "+0.00"),
            (-0.00004, "-0.00"),
            (1e-10, "+0.00"),
            (f64::NAN, "NaN"),
            (f64::INFINITY, "+inf"),
            (f64::NEG_INFINITY, "-inf"),
        ] {
            assert_eq!(signed_change(x, 2), text, "{x}");
        }
        // The cap: where the search reads a digit only past `from` + CHANGE_EXTRA_DECIMALS
        // (-0.00004 at 5 decimals, 1e-10 at 10, 1e-20 at none up to MOST_DECIMALS, where it
        // falls back to MOST_DECIMALS), the change reads `from`.
        for (x, past_the_cap) in [(-0.00004, 5), (1e-10, 10), (1e-20, MOST_DECIMALS)] {
            let (found, _) = fewest_decimals_apart(2, |d| fixed(x, d), |t| reads_not_zero(t));
            assert_eq!(found, past_the_cap, "{x}");
            assert_eq!(change_decimals(x, 2), 2, "{x}");
        }
        assert_eq!(change_decimals(-0.0063, 2), 2);
        assert_eq!(change_decimals(-0.0046, 2), 3);
        assert_eq!(change_decimals(-0.00012, 2), 4, "at the cap itself");
        // A size change starts at one decimal (the sliders' 0.1 mm): 0.0046 mm needs three,
        // 6e-5 mm is past the cap there.
        assert_eq!(change_decimals(0.0046, 1), 3);
        assert_eq!(change_decimals(6e-5, 1), 1);
        // Zero and numbers that are not finite read `from`, whatever `from` is; a `from` past
        // what the search covers reads itself, never fewer decimals.
        for x in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(change_decimals(x, 1), 1, "{x}");
        }
        assert_eq!(change_decimals(0.5, MOST_DECIMALS + 1), MOST_DECIMALS + 1);
        assert_eq!(change_decimals(1e-17, MOST_DECIMALS - 1), MOST_DECIMALS - 1);
        // signed_fixed: exactly `decimals` places with the true sign; an exact zero reads "+".
        assert_eq!(signed_fixed(-0.0063, 3), "-0.006");
        assert_eq!(signed_fixed(-0.0046, 3), "-0.005");
        assert_eq!(signed_fixed(-0.00004, 2), "-0.00");
        assert_eq!(signed_fixed(0.0, 3), "+0.000");
        assert_eq!(signed_fixed(-0.0, 3), "+0.000");
        assert_eq!(signed_fixed(f64::NAN, 2), "NaN");
        assert_eq!(signed_fixed(f64::NEG_INFINITY, 2), "-inf");
        // A row's two N·m at one precision (T-14): the larger of the sides' decimals. (e)'s
        // Outside diameter row, -0.0063 and -0.0046 N·m, reads "-0.006" and "-0.005" (each on
        // its own: "-0.01" and "-0.005"); (e2)+Auto length's, -0.00477 and -0.00502, reads
        // "-0.005" both. A row with one number side keeps that side's own; a row of words has
        // none.
        assert_eq!(shared_change_decimals(&[-0.0063, -0.0046], 2), Some(3));
        assert_eq!(shared_change_decimals(&[-0.00477, -0.00502], 2), Some(3));
        assert_eq!(signed_fixed(-0.00477, 3), signed_fixed(-0.00502, 3));
        assert_eq!(shared_change_decimals(&[-0.0046], 2), Some(3));
        assert_eq!(shared_change_decimals(&[0.2008], 2), Some(2));
        assert_eq!(shared_change_decimals(&[], 2), None);
    }

    #[test]
    fn the_decimals_search_takes_the_fewest_from_its_start_else_the_most() {
        // The one search for more decimals (BL-059 NB-1; the guidance's band and window quotes,
        // a typed input below its slider's bottom, and the Size tab's change decimals).
        let digit = |text: &String| reads_not_zero(text);
        assert_eq!(
            fewest_decimals_apart(2, |d| fixed(-0.0046, d), digit),
            (3, "-0.005".to_owned())
        );
        // Never fewer than its start, even where fewer would read apart.
        assert_eq!(
            fewest_decimals_apart(2, |d| fixed(0.5, d), digit),
            (2, "0.50".to_owned())
        );
        // None up to MOST_DECIMALS: MOST_DECIMALS and its quote.
        assert_eq!(
            fewest_decimals_apart(2, |d| fixed(1e-20, d), digit),
            (MOST_DECIMALS, fixed(1e-20, MOST_DECIMALS))
        );
        // What reads apart here: a finite number that is not zero, of either sign.
        for (text, apart) in [
            ("-0.005", true),
            ("0.50", true),
            ("-0.00", false),
            ("0.000", false),
            ("NaN", false),
            ("+inf", false),
            ("", false),
        ] {
            assert_eq!(reads_not_zero(text), apart, "{text:?}");
        }
    }
}
```

- [ ] **Step 4: Run them and see the build fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::format`

Expected: the lib test target does not compile (the functions do not exist in `format.rs` yet):
```text
error[E0425]: cannot find function `change_decimals` in this scope      (9 of these)
error[E0425]: cannot find function `fewest_decimals_apart` in this scope      (4 of these)
error[E0425]: cannot find function `reads_not_zero` in this scope      (3 of these)
error[E0425]: cannot find function `shared_change_decimals` in this scope      (5 of these)
error[E0425]: cannot find function `signed_change` in this scope
error[E0425]: cannot find function `signed_fixed` in this scope      (9 of these)
error[E0425]: cannot find value `CHANGE_EXTRA_DECIMALS` in this scope
error[E0425]: cannot find value `MOST_DECIMALS` in this scope      (7 of these)
error: could not compile `magcoupling-rs` (lib test) due to 39 previous errors
```

- [ ] **Step 5: The format (`gui/format.rs`) and the moved search (`gui/guidance.rs`)**

`magcoupling-rs/src/gui/format.rs`: the search and its fallback arrive (unchanged but for `pub(crate)` and a doc that names its callers without links into `guidance.rs`), then the new functions with `TORQUE_DECIMALS`, and `signed_percent` to one decimal.

Old:
```rust
/// A change, as a fraction, as a whole signed percentage with its unit: 0.1121 reads `+11 %`,
/// -0.2679 `-27 %`. A change that rounds to zero reads `+0 %`, never `-0 %`; one that is not
/// finite reads as its [`non_finite_text`] and the unit.
pub(crate) fn signed_percent(fraction: f64) -> String {
    if let Some(text) = non_finite_text(fraction) {
        return format!("{text} %");
    }
    let percent = (fraction * 100.0).round();
    // round() keeps the sign of a small fall (-0.4 rounds to -0.0, which prints "-0").
    let percent = if percent == 0.0 { 0.0 } else { percent };
    format!("{percent:+.0} %")
}
```

New:
```rust
/// The most decimals [`fewest_decimals_apart`] gives a number near one: f64 holds no more of
/// its digits.
const MOST_DECIMALS: usize = 15;

/// The one search for more decimals (BL-059 NB-1): of the guidance's band and window quotes
/// (`gui::guidance`'s `band_quotes` and `window_quotes`), its quote of an input typed below its
/// slider's bottom, and the Size tab's change decimals ([`change_decimals`]). The fewest
/// decimals, from `from` up to [`MOST_DECIMALS`], whose `quotes` read `apart`, with those
/// quotes; else [`MOST_DECIMALS`] and its quotes.
pub(crate) fn fewest_decimals_apart<T>(
    from: usize,
    quotes: impl Fn(usize) -> T,
    apart: impl Fn(&T) -> bool,
) -> (usize, T) {
    (from..=MOST_DECIMALS)
        .map(|decimals| (decimals, quotes(decimals)))
        .find(|(_, quoted)| apart(quoted))
        .unwrap_or_else(|| (MOST_DECIMALS, quotes(MOST_DECIMALS)))
}

/// How many decimals past its own a change may take to show a digit that is not zero
/// ([`change_decimals`]; slice 3 spec T-9). A change still zero there is float noise, such as
/// Auto's solve residual, and reads its own decimals with its true sign. The torque card draws
/// a size or margin change only where it shows a digit within them.
pub(crate) const CHANGE_EXTRA_DECIMALS: usize = 2;

/// The decimals a torque change starts from on the Size tab (spec section 5: N·m to two
/// decimals, then [`change_decimals`]): the readout's gap (`torque_gap_text`) and the torque
/// card's N·m read this one constant.
pub(crate) const TORQUE_DECIMALS: usize = 2;

/// Whether `text` reads a finite number that is not zero: "-0.005" does; "-0.00", "NaN" and
/// "+inf" do not. What [`change_decimals`] looks for, decimal by decimal.
pub(crate) fn reads_not_zero(text: &str) -> bool {
    text.parse::<f64>().is_ok_and(|x| x.is_finite() && x != 0.0)
}

/// The decimals of a change `x` (decision 10B): the fewest, from `from` up to `from` +
/// [`CHANGE_EXTRA_DECIMALS`], at which [`fixed`] reads a digit that is not zero
/// ([`fewest_decimals_apart`]; -0.0046 from 2 takes 3, "-0.005"). `from` where `x` is zero or
/// not finite, or reads zero even at the cap (T-9: -0.00004 from 2 reads "-0.00").
pub(crate) fn change_decimals(x: f64, from: usize) -> usize {
    if x == 0.0 || !x.is_finite() {
        return from;
    }
    let (decimals, text) = fewest_decimals_apart(
        from,
        |decimals| fixed(x, decimals),
        |text| reads_not_zero(text),
    );
    // The search has no cap of its own: it falls back to MOST_DECIMALS when no decimal reads
    // a digit, or when `from` is past MOST_DECIMALS. Its answer counts only inside the cap,
    // and only where it reads a digit.
    if (from..=from + CHANGE_EXTRA_DECIMALS).contains(&decimals) && reads_not_zero(&text) {
        decimals
    } else {
        from
    }
}

/// `x` to exactly `decimals` places with its true sign, as the Size tab writes a change: a
/// value that rounds to zero keeps its sign ("-0.00"), an exact zero of either sign reads "+"
/// (the house rule; `{:+}` alone prints -0.0 as "-0"), and a number that is not finite reads as
/// its [`non_finite_text`].
pub(crate) fn signed_fixed(x: f64, decimals: usize) -> String {
    match non_finite_text(x) {
        Some(text) => text.to_owned(),
        None if x == 0.0 => format!("{:+.decimals$}", 0.0),
        None => format!("{x:+.decimals$}"),
    }
}

/// A change `x` at its own decimals: [`signed_fixed`] at [`change_decimals`] from `from` (the
/// Size tab's N·m from [`TORQUE_DECIMALS`], a size change in mm from 1).
pub(crate) fn signed_change(x: f64, from: usize) -> String {
    signed_fixed(x, change_decimals(x, from))
}

/// The decimals a row of changes shares (slice 3 spec T-14: a torque card row's two N·m at one
/// precision): the larger of their [`change_decimals`] from `from`, so each keeps its true sign
/// and a digit that is not zero, and the two read in their true ratio (-0.0063 and -0.0046 N·m
/// read "-0.006" and "-0.005", not "-0.01" and "-0.005"). `None` for a row with no number.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the torque card's lines call it (slice 3 Task 4, which removes this)"
    )
)]
pub(crate) fn shared_change_decimals(changes: &[f64], from: usize) -> Option<usize> {
    changes.iter().map(|&x| change_decimals(x, from)).max()
}

/// A change, as a fraction, as a signed percentage to one decimal with its unit (decision 10B:
/// the Size tab's one percent rule): 0.0349 reads `+3.5 %`, -0.2679 `-26.8 %`. One that rounds
/// to zero keeps its true sign (`-0.0 %`), an exact zero of either sign reads `+0.0 %`, and one
/// that is not finite reads as its [`non_finite_text`] and the unit ([`signed_fixed`]).
pub(crate) fn signed_percent(fraction: f64) -> String {
    format!("{} %", signed_fixed(fraction * 100.0, 1))
}
```

`magcoupling-rs/src/gui/guidance.rs`: the search leaves (moved, not copied); `band_quotes`, `window_quotes` and `input_text_below` call it through the import. The doc links `[`fewest_decimals_apart`]` at `input_text_below` resolve through the import.

Old:
```rust
use crate::gui::format::{
    SIGNIFICANT_DIGITS, fixed, format_value, last_digit_step, signed_percent, with_unit,
};
```

New:
```rust
use crate::gui::format::{
    SIGNIFICANT_DIGITS, fewest_decimals_apart, fixed, format_value, last_digit_step,
    signed_percent, with_unit,
};
```

Old:
```rust
/// The most decimals [`fewest_decimals_apart`] gives a number near one: f64 holds no more of
/// its digits.
const MOST_DECIMALS: usize = 15;

/// The one search for more decimals of [`band_quotes`] and [`window_quotes`] (BL-059 NB-1): the
/// fewest decimals, from `from` up to [`MOST_DECIMALS`], whose `quotes` read `apart`, with those
/// quotes; else [`MOST_DECIMALS`] and its quotes.
fn fewest_decimals_apart<T>(
    from: usize,
    quotes: impl Fn(usize) -> T,
    apart: impl Fn(&T) -> bool,
) -> (usize, T) {
    (from..=MOST_DECIMALS)
        .map(|decimals| (decimals, quotes(decimals)))
        .find(|(_, quoted)| apart(quoted))
        .unwrap_or_else(|| (MOST_DECIMALS, quotes(MOST_DECIMALS)))
}

/// The band and the widest band the rating allows, as gearbox Problem sentence 2's first remedy
```

New:
```rust
/// The band and the widest band the rating allows, as gearbox Problem sentence 2's first remedy
```

The what-moves-what comment (spec 9.2, `gui/guidance.rs:2711-2712`) states the new rule:

Old:
```rust
// What moves what (spec 3.3, 4.1). Lengths read to 0.1 mm (the Size tab sliders' precision),
// torques to 0.01 N·m and changes to whole percent.
```

New:
```rust
// What moves what (spec 3.3, 4.1). Lengths read to 0.1 mm (the Size tab sliders' precision),
// torques to 0.01 N·m and changes to one decimal of percent (`signed_percent`, decision 10B).
```

- [ ] **Step 6: The readout's gap (`gui/size_controls.rs`) and its re-export (`gui/mod.rs`)**

Old:
```rust
use crate::gui::format::{fixed, signed_percent};
```

New:
```rust
use crate::gui::format::{TORQUE_DECIMALS, change_decimals, fixed, signed_change, signed_percent};
```

Old:
```rust
/// The hot-low torque's gap to the requirement, and whether it meets it: `hot_low_Nm >=
/// required_Nm`, as the engine's hot minimum check (which fails `hot_low < required`). Two
/// decimals of N·m and a whole percent of the requirement, each signed: "short 0.29 N·m
/// (-12 %)", "margin +0.40 N·m (+16 %)". The percent is left out when the requirement is not a
/// positive torque (the slider stops at its metadata minimum; a design file need not).
#[allow(non_snake_case)] // unit suffix
pub fn torque_gap_text(hot_low_Nm: f64, required_Nm: f64) -> (String, bool) {
    let meets = hot_low_Nm >= required_Nm;
    let gap_Nm = (hot_low_Nm - required_Nm).abs();
    let (word, torque) = if meets {
        ("margin", format!("+{}", fixed(gap_Nm, 2)))
    } else {
        ("short", fixed(gap_Nm, 2))
    };
    let text = if required_Nm > 0.0 {
        // `signed_percent` writes a change that rounds to zero as "+0 %". A shortfall keeps its
        // sign ("-0 %"), so a design a hair under its requirement never reads like a pass.
        let percent = signed_percent(gap_Nm / required_Nm);
        let percent = if meets {
            percent
        } else {
            percent.replacen('+', "-", 1)
        };
        format!("{word} {torque} N·m ({percent})")
    } else {
        format!("{word} {torque} N·m")
    };
    (text, meets)
}
```

New:
```rust
/// The hot-low torque's gap to the requirement, and whether it meets it: `hot_low_Nm >=
/// required_Nm`, as the engine's hot minimum check (which fails `hot_low < required`). The Size
/// tab's one number format (decision 10B): the N·m to the fewest decimals, from two
/// ([`TORQUE_DECIMALS`]), that read a digit that is not zero ([`change_decimals`], at most two
/// more), a margin signed, and one decimal of percent of the requirement with the gap's own sign
/// ([`signed_percent`]): "short 0.29 N·m (-11.6 %)", "margin +0.40 N·m (+16.0 %)", "short
/// 0.001 N·m (-0.0 %)", so a design a hair under its requirement never reads like a pass. The
/// percent is left out when the requirement is not a positive torque (the slider stops at its
/// metadata minimum; a design file need not).
#[allow(non_snake_case)] // unit suffix
pub fn torque_gap_text(hot_low_Nm: f64, required_Nm: f64) -> (String, bool) {
    let meets = hot_low_Nm >= required_Nm;
    let gap_Nm = hot_low_Nm - required_Nm;
    let torque = if meets {
        format!("margin {} N·m", signed_change(gap_Nm, TORQUE_DECIMALS))
    } else {
        format!(
            "short {} N·m",
            fixed(gap_Nm.abs(), change_decimals(gap_Nm, TORQUE_DECIMALS))
        )
    };
    let text = if required_Nm > 0.0 {
        format!("{torque} ({})", signed_percent(gap_Nm / required_Nm))
    } else {
        torque
    };
    (text, meets)
}
```

`magcoupling-rs/src/gui/mod.rs`: `torque_gap_text` becomes public as `gui::torque_gap_text`, for the slice's screen dump (Task 9 reads it from outside the crate, and `size_controls` is a private module; spec 8.6 test 27 and 9.2 name the re-export; plan deviation PD-4):

Old:
```rust
pub use panel::{
    CentreView, MagcouplingPanel, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX, WELCOME,
};
```

New:
```rust
pub use panel::{
    CentreView, MagcouplingPanel, PanelRequest, SHOW_DETAILS, SIZING_LOG_PREFIX, WELCOME,
};
// The Size tab's gap text, public for the slice-3 screen dump (`size_controls` is private).
pub use size_controls::torque_gap_text;
```

The panel tests that build the expected gap with `torque_gap_text` itself (`gui/panel.rs:4982`, `:5091`, `:5150`, `:5647`, `:5701`) pass unchanged; `:5150`'s `gap.starts_with("margin +")` holds (a margin is signed by `signed_change`).

- [ ] **Step 7: Run the format's and the gap's tests and see them pass; plant four faults**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::format the_gap_reads_short_or_a_margin_against_the_requirement a_requirement_that_is_no_positive_torque_drops_the_percent the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not the_readout_texts_have_glyphs_and_plain_words`

Expected: `test result: ok. 19 passed; 0 failed; 0 ignored; 0 measured; 917 filtered out` (the 15 tests of `gui::format` and the four named).

Each planted fault below is one edit of `magcoupling-rs/src/gui/format.rs`, run with the command given, then undone (`git diff --stat` shows the file as after Step 5 again). They show the edges of tests 12 and 13 bite, and that the gap tests' literal pins hold `TORQUE_DECIMALS`:

| Planted fault | Run (`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- <filter>`) | Expected |
|---|---|---|
| In `change_decimals`, `&& reads_not_zero(&text) {` becomes ` {` (the search's fallback taken as an answer) | `gui::format` | `a_change_keeps_its_sign_and_shows_a_digit_that_is_not_zero` fails at `format.rs:449`: `left: 15`, `right: 14` (`change_decimals(1e-17, MOST_DECIMALS - 1)`); `14 passed; 1 failed` |
| `signed_percent`'s body becomes `format!("{:+.1} %", fraction * 100.0)` (no house rule) | `a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign` | fails at `format.rs:400`: `left: "-0.0 %"`, `right: "+0.0 %"` |
| In `shared_change_decimals`, `.max()` becomes `.min()` | `a_change_keeps_its_sign_and_shows_a_digit_that_is_not_zero` | fails at `format.rs:463`: `left: Some(2)`, `right: Some(3)` ((e)'s Outside diameter row) |
| `pub(crate) const TORQUE_DECIMALS: usize = 2;` becomes `= 3;` (critique A-14) | the step's run filter (`gui::format the_gap_reads_short_or_a_margin_against_the_requirement a_requirement_that_is_no_positive_torque_drops_the_percent the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not the_readout_texts_have_glyphs_and_plain_words`) | 2 fail (`17 passed; 2 failed`): `the_gap_reads_short_or_a_margin_against_the_requirement` at `size_controls.rs:420`: `left: ("short 0.290 N·m (-11.6 %)", false)`, `right: ("short 0.29 N·m (-11.6 %)", false)`; `a_requirement_that_is_no_positive_torque_drops_the_percent` at `size_controls.rs:455`: `left: ("margin +2.000 N·m", true)`, `right: ("margin +2.00 N·m", true)` |

- [ ] **Step 8: Docs: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`, `docs/ai/02-system.yaml`**

(a) README, the Size-tab paragraph (around `:202-204`; spec 9.2's `:203`). Old (three whole lines):
```text
links carry it. **Hot-low torque** (`metal.torque_hot_low_Nm`) is shown with its gap to
Required torque, "short X N·m (-P %)" in red or "margin +X N·m (+P %)" in green, and **Pull-out
torque (nominal, at T)** is the nominal figure at the live operating temperature. While the
```
New (seven lines; the quoted readout stays on one line, so a grep for it finds it):
```text
links carry it. **Hot-low torque** (`metal.torque_hot_low_Nm`) is shown with its gap to
Required torque, "short X N·m (-P.P %)" in red or "margin +X N·m (+P.P %)" in green, and
**Pull-out torque (nominal, at T)** is the nominal figure at the live operating temperature.
The gap is in the Size tab's one number format (decision 10B, `gui::format`): X to the fewest
decimals, from two, that read a digit that is not zero (at most two more), and one decimal of
percent with the gap's own sign, so a design a hair under its requirement reads
"short 0.001 N·m (-0.0 %)". While the
```

(b) README, the Layout table's `src/gui/` row (around `:155`), its `format.rs` entry. Old (a substring of one line):
```text
`format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell)
```
New:
```text
`format.rs` (`format_value`, `with_unit`: the display text of every value, `+inf` and `NaN` included; `search_haystack`, `search_needle`: what the results search and the inputs filter match: the display label, the catalogue label, the path and the cell; the Size tab's one number format, slice 3 decision 10B: `fixed`, `signed_fixed`, `signed_change` and `change_decimals` (a change to the fewest decimals, from its own (`TORQUE_DECIMALS`, two, for N·m: the readout's gap and the torque card), that keep its true sign and read a digit that is not zero, at most `CHANGE_EXTRA_DECIMALS` more), `shared_change_decimals` (a torque card row's two N·m at one precision), the one-decimal `signed_percent`, and `fewest_decimals_apart`, the one search for more decimals)
```

(c) README, the Tests table's `src/gui/format.rs` row (around `:699` after (a)'s four lines; `:695` at HEAD). Old (the row's end):
```text
the step `mm_at_least` and the gearbox Problem's limits round by |
```
New:
```text
the step `mm_at_least` and the gearbox Problem's limits round by; the Size tab's one number format (slice 3, decision 10B): a change as a percentage to one decimal with its true sign (an exact zero of either sign "+0.0 %", a fall that rounds to zero "-0.0 %"), as N·m to the fewest decimals, from two, that read a digit that is not zero, at most two more (-0.0046 reads "-0.005", -0.00004 "-0.00"), a row's two N·m at one precision (-0.0063 and -0.0046 read "-0.006" and "-0.005"), the decimals search's fallback, and the readout's gap in that format (`size_controls.rs`: "short 0.29 N·m (-11.6 %)", "short 0.001 N·m (-0.0 %)") |
```

(d) `docs/ai/03-structure.yaml`, the `gui` line (`:68`, a double-quoted scalar: no `"` may be added). Old:
```text
format.rs gained format::fixed and format::signed_percent (the gap text's one formatting)
```
New:
```text
format.rs gained format::fixed and the Size tab's one number format (slice 3, decision 10B: signed_fixed, signed_change and change_decimals with CHANGE_EXTRA_DECIMALS and reads_not_zero, TORQUE_DECIMALS (a torque change's own two decimals, which torque_gap_text and the torque card read), shared_change_decimals for a torque card row's two N·m, a one-decimal signed_percent, and fewest_decimals_apart moved from guidance.rs; torque_gap_text reads them)
```

(e) `docs/ai/02-system.yaml`, `:116`'s Size-tab clause (spec 9.2: "the gap's number format"; Task 5 rewrites the same line's neighbour clause, Task 10 its timing figure). Old:
```text
with its live gap to the required torque (torque_gap_text)
```
New:
```text
with its live gap to the required torque (torque_gap_text, in the Size tab's one number format, decision 10B: the N·m to the fewest decimals, from two, that read a digit that is not zero, and one decimal of percent with the gap's own sign)
```

(f) `docs/ai/03-structure.yaml`, the `gui` line's `size_controls.rs` clause: the re-export of Step 6. Old:
```text
LINKED_HOUSING, torque_gap_text, pullout_label,
```
New:
```text
LINKED_HOUSING, torque_gap_text (public as gui::torque_gap_text, a pub use in gui/mod.rs, for the slice-3 screen dump), pullout_label,
```

Run: `grep -c "(-P %)\|(+P %)" magcoupling-rs/README.md`

Expected: `0`.

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`.

`docs/ai/02-system.yaml` does not parse at 69fa505 (`could not find expected ':'` at line 15, pre-existing; BL-064's release notes record it), so check the edited item alone:

Run: `python -c "import yaml; t=open('docs/ai/02-system.yaml', encoding='utf-8').read().split(chr(10)); l=[x for x in t if 'torque_gap_text, in the Size tab' in x]; assert len(l) == 1; print(type(yaml.safe_load(l[0].strip())).__name__)"`

Expected: `list`.

- [ ] **Step 9: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary `ok`, the counts of 69fa505 (the GUI is not built without `--features app`):

| binary | result |
|---|---|
| lib | 233 passed, 1 ignored |
| assumptions | 13 passed |
| deviations | 54 passed |
| differential | 19 passed |
| explain | 63 passed, 1 ignored |
| gearbox | 13 passed |
| grades | 12 passed |
| material_library | 4 passed |
| material_links | 14 passed |
| parity | 4 passed |
| python_schema | 7 passed |
| robustness | 13 passed |
| schema | 11 passed |
| shaft_joint | 40 passed |
| size_controls | 38 passed |
| sizing | 45 passed |
| static_data | 8 passed |
| doc-tests | 1 passed, 1 ignored |

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `930 passed; 0 failed; 6 ignored` (928 at 69fa505: test 12 replaces the whole-percent test; test 13 and the search's test are new); the two app binaries 0 tests; `explain` 64 passed, 1 ignored; every other binary as above.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`, then the same without `--features app`.

Expected: `Finished`, no warning, both.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/02-system.yaml
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/gui/format.rs
 M magcoupling-rs/src/gui/guidance.rs
 M magcoupling-rs/src/gui/mod.rs
 M magcoupling-rs/src/gui/size_controls.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
```

- [ ] **Step 10: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T1.txt` with the Write tool:
```text
feat(magcoupler): one number format for the Size tab (BL-073 T1, decision 10B)

The Size tab's changes read with their true sign and a digit that is not zero.
gui::format gains change_decimals (the fewest decimals, from a change's own and
at most CHANGE_EXTRA_DECIMALS = 2 more, that read a digit; float noise past them
reads its own decimals with its sign, T-9), reads_not_zero, signed_fixed,
signed_change and shared_change_decimals (a torque card row's two N·m at one
precision, T-14; its caller, the card's lines, comes in Task 4, which removes
its expect(dead_code)). signed_percent reads one decimal: an exact zero of
either sign "+0.0 %", a fall that rounds to zero "-0.0 %". fewest_decimals_apart
and MOST_DECIMALS move from gui/guidance.rs unchanged: one search for more
decimals. TORQUE_DECIMALS = 2, the decimals a torque change starts from, is one
constant for the readout's gap and the torque card's N·m (Task 4).

The readout's gap takes the format, its percent signed by the gap itself:
"short 0.29 N·m (-11.6 %)", "margin +0.40 N·m (+16.0 %)", and a design a hair
under its requirement "short 0.001 N·m (-0.0 %)" (was "short 0.00 N·m (-0 %)").
A number-format change to shipped UI, no wording change (owner 10B). The
Magnets per ring neighbour lines read the same percent until the card takes
them (Task 5). gui/mod.rs re-exports torque_gap_text as gui::torque_gap_text
for the slice's screen dump (size_controls is private; spec 8.6 test 27).

Tests: the whole-percent test is replaced by one to a tenth (spec test 12);
new tests pin the change decimals, their cap, signed_fixed and a row's shared
decimals (test 13) and the decimals search's fallback; the gap tests take the
new texts (test 14). The README, docs/ai/02-system.yaml and 03-structure.yaml
follow.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/format.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/size_controls.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T1.txt
git status --short
```

Expected: one commit of 7 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 1 moves** (each confirmed by running in the plan-time prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui::format::tests::a_change_reads_as_a_whole_signed_percentage_never_minus_zero` | replaced by `a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign` (spec test 12; the old test's 0.1121, -0.2679 and 1.1279 kept, to a tenth) | 10B: one decimal; the old pin "-0.004 -> +0 %" is now "-0.4 %", and a fall that rounds to zero keeps its sign ("-0.0 %") |
| `gui::size_controls::tests::the_gap_reads_short_or_a_margin_against_the_requirement` | its four texts to the new rule (spec 5: "-11.6 %", "+16.0 %", "+0.0 %", "short 0.001 N·m (-0.0 %)"); two cases added (a margin at three decimals, a shortfall past the cap) | 10B on the readout's gap |
| `gui::size_controls::tests::the_readout_texts_have_glyphs_and_plain_words` | two texts added (the exact zero and the three-decimal shortfall) | the new forms held to the glyphs and the guard (test 14) |
| `gui::guidance::tests::the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not` | the expected percent through `signed_percent`, not `{:+.0}` of a rounded value | the neighbour lines read `signed_percent` until Task 5 moves them (and this test) to the card |
| `gui::size_controls::tests::a_requirement_that_is_no_positive_torque_drops_the_percent`; `gui/panel.rs`'s tests at `:4982`, `:5091`, `:5150`, `:5647`, `:5701` | unchanged, pass | no percent there, or the expected gap is built with `torque_gap_text` |

---

### Task 2: Shared prerequisites: `check_terms` and the axis table, `slider_minimum`, the card's ban list and its matcher, and the shared test designs

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec sections 3.3 (`slider_minimum`), 3.6 (`check_terms`, `CheckTerm`, `Quoted`, `SPACE_CLAIM_AXIS_SIZES`), 4.4 (`CARD_RANKING_WORDS` and its matcher) and section 8's intro (the shared test designs), with tests 1a and 9. Each prerequisite lands with its own test, before the card module that reads them:
- `check_terms(check, inputs, results) -> Vec<CheckTerm>` in `gui/guidance.rs`: the production half of the fix tests' test-private `terms()`. Each counted check's terms in a fixed order, margins larger-is-better and `passes` as the verdict compares, bit for bit as `terms()` built them; each term carries the quantity the card quotes for it (`Quoted { path, name, decimals }`, spec 3.6's table: none for Hot-low torque, Temperature limits (T-13), the end effect and the fit). `terms()` now calls it and adds only the clamp's own term (`clamp_margin`); `check_terms` gives the clamp none. The test's `Term` becomes `CheckTerm` and `joint_term` moves with it (moved, never copied). `check_terms` has no production caller until Task 4's margins, so it carries Task 1's `expect(dead_code)` attribute (plan deviation PD-9), which seeds what it reads (`CheckTerm`, `Quoted`, `joint_term`, `JointSide::quoted`): the lib build warns nothing.
- `WALL_NEEDED` ("wall needed") and `COLD_HIGH` ("cold-high") are the cup wall's and the gearbox check's `Quoted.name`. `check_terms` must name them here, before `torque_card.rs` exists (Task 3), so both constants live in `gui/guidance.rs` beside the table that names them, as `SPACE_CLAIM_AXIS_SIZES` names the axes; spec 4.1 puts them in `torque_card.rs` (plan deviation PD-7). Task 4 imports them and defines no copies; the G-3 review (Task 7) sets their words here.
- `SPACE_CLAIM_AXIS_SIZES: [(&str, &str); 3]`: each axis's name and size path in `axis_reserves`' order, typed once. The space claim's Problem reads it (its overshoot and claim paths stay in the Problem; its sentences are unchanged, and `the_space_claim_problem_names_each_axis_past_its_claim` passes as it is), and `check_terms` quotes from it.
- `slider_minimum(legacy: &SizeControls, preference: Option<f64>) -> f64` in `gui/size_controls.rs`: the Outside diameter slider's lowest value, the expression `controls_ui` typed (`gui/size_controls.rs:248-250`), now called by it and, from Task 3, by the card. Beside it, `SliderMapping` (plan review 2.3): one mapping per Size slider (`Diameter { minimum }`, `Length`, `Input { path }`), whose `design(inputs, value)` is the design that slider makes of a value; `controls_ui` applies it on an edit (the two mapping sites it typed go), and from Task 3 the card's step, test 1's oracle and the test table apply it too. The count is set as the slider sets it, `InputSet::set` of a whole number (plan deviation PD-1).
- `CARD_RANKING_WORDS` and its whole-word matcher `ranking_words_in(text) -> Vec<&'static str>` in `engine/explain/plain.rs`, beside `TORQUE_ORDER_WORDS` (the spec names no matcher: plan deviation PD-10). A word matches whole, in any case, with an `s` added or taken off; it returns the listed words a text says, in the list's order, each once, as `plain_text_violations` returns its patterns. Task 4's test 16 calls it over every card line.
- The shared test designs in `gui/test_support.rs`, each doc comment stating its section 2.1 fact, and test 1a, `the_shared_card_designs_read_as_section_2_1_says`, which pins every one. It sits in a new `mod tests` at the end of `gui/test_support.rs`: the file is compiled for tests only (`gui/mod.rs:32-33`), so the test compiles before the card module exists and reads `level`, `counted_guides`, `predicts`, `dimensions` and `fixed`, nothing of the card. (The `gui/guidance.rs` tests keep their own `no_back_iron(i: &mut DesignInputs)` edit; never import `test_support::no_back_iron` there.)

The facts, from the prototype (each read to the decimals spec 2.1 quotes): the mixed pair and its reverse read hot-low 1.0099 N·m; the mixed pair fails Hot-low torque, Running clearance, Cup wall and Shaft joints, the reverse pair Hot-low torque, Running clearance and Cup wall, both with Temperature limits at caution (spec 2.1 names only "3 of 9 and 1 caution" for the reverse pair). Every other fact equals spec 2.1's.

Line numbers are at dae024f (whose code is 69fa505's) or, in files Task 1 changed, after Task 1; trust the code. Every code block is rustfmt's layout: replaying this task's Old/New blocks on Task 1's files reproduces this task's files byte for byte in the plan-time prototype.

**Files:**
- Modify: `magcoupling-rs/src/engine/explain/plain.rs` (`CARD_RANKING_WORDS`, `ranking_words_in`, the matcher's test), `magcoupling-rs/src/gui/guidance.rs` (`check_terms`, `CheckTerm`, `Quoted`, `joint_term`, `WALL_NEEDED`, `COLD_HIGH`, `JointSide::quoted`, `SPACE_CLAIM_AXIS_SIZES`, the space claim's Problem reading it; the test module's `terms()`, the removed `Term`, test 9), `magcoupling-rs/src/gui/size_controls.rs` (`slider_minimum` and `controls_ui`'s one call; `SliderMapping` and `controls_ui`'s two mapping sites, which apply it; their tests), `magcoupling-rs/src/gui/test_support.rs` (the twelve designs; test 1a)
- Modify (docs): `magcoupling-rs/README.md` (the Layout table's `src/engine/explain/` row, `:153`, and `src/gui/` row, `:155`: `guidance.rs`, `size_controls.rs`, `test_support.rs`; the Tests table's rows for `src/engine/explain/plain.rs`, `src/gui/panel.rs` and `src/gui/guidance.rs`, and a new `src/gui/test_support.rs` row), `docs/ai/03-structure.yaml` (the `explain` line, `:64`; the `gui` line, `:68`: `guidance.rs`, `size_controls.rs`, `test_support.rs`)
- Test: `magcoupling-rs/src/engine/explain/plain.rs` (`the_card_ranking_words_match_whole_in_any_case_with_an_s_added_or_taken_off`, the matcher half of spec test 16), `magcoupling-rs/src/gui/guidance.rs` (`check_terms_keeps_the_fix_tests_margins`, test 9), `magcoupling-rs/src/gui/size_controls.rs` (`the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown` and `each_size_slider_makes_its_design_through_its_own_mapping`; spec test 6's "slider_minimum equals the controls' own minimum at every design" stays Task 3's), `magcoupling-rs/src/gui/test_support.rs` (`the_shared_card_designs_read_as_section_2_1_says`, test 1a)

**Interfaces:**
- Consumes: Task 1's `gui::format::fixed` (test 1a); `gui::guidance::{level, counted_guides, axis_reserves, shaft_joints, joint_side, JOINT_SIDES, SPACE_CLAIM_AXES, TemperatureTerm}`; `engine::shaft_joint::{joints, CAUSES, Cause}`; `engine::api::shaft_joint_links`; `gui::plain_text::label_for`; `engine::size_controls::{SizeControls, dimensions, with_ring_radius}`; `gui::size_controls::predicts`; `engine::clamps::PAUSED`.
- Produces:
  - `engine::explain::plain`: `pub const CARD_RANKING_WORDS: [&str; 7] = ["strongest", "biggest", "best", "most", "lever", "dominant", "ranks"]`; `pub fn ranking_words_in(text: &str) -> Vec<&'static str>`.
  - `gui::guidance`:
    - `pub const WALL_NEEDED: &str = "wall needed"` and `pub const COLD_HIGH: &str = "cold-high"` (drafts; Task 7 sets them);
    - `#[derive(Clone, Copy, Debug, PartialEq, Eq)] pub(crate) struct Quoted { pub path: &'static str, pub name: Option<&'static str>, pub decimals: usize }`;
    - `#[derive(Clone, Debug, PartialEq)] pub(crate) struct CheckTerm { pub name: String, pub margin: f64, pub passes: bool, pub quoted: Option<Quoted> }`. `name` is the fix tests' term name (never drawn): "hot-low torque"; "running clearance"; "cup wall"; the eight `TemperatureTerm` names as they print (`HotDayMagnet(Inner)` ...); "diameter", "overall length", "large-diameter bay"; "gearbox input torque"; "inner hub slips", "inner hub bore stress", "inner hub fitting", "inner hub key", then the cup's four; "end-effect factor"; "inner ring", "outer ring";
    - `pub(crate) fn check_terms(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Vec<CheckTerm>`: empty for the clamp and for a path that is no guide's check (no panic in production); the shaft joints always eight terms in one order, every one passing while the check cannot be made or its joint is not judged. Task 4 reads only `quoted` for the margins (the card draws the guide's label, then `name` when it is `Some`, then the change through `with_unit` of `result_info(path)`'s unit at `decimals`) and deletes the `expect(dead_code)` attribute;
    - the quoted quantity per check, spec 3.6's table: Running clearance `metal.min_running_clearance_mm`, no name, 2; Cup wall `materials.backiron_thickness_needed_mm`, `WALL_NEEDED`, 2; Gearbox input torque `metal.cold_high_input_Nm`, `COLD_HIGH`, 2; Space claim each axis's size (`metal.rotating_od_mm`, `metal.axial_stack_mm`, `metal.large_dia_stack_mm`), named "Rotating diameter", "Overall length", "Large-diameter stack", 2; Shaft joints per (joint, cause) the joint's needed minimum (µm, 2), bore stress (MPa, 1), fitting temperature (°C, 1) or key pressure (MPa, 1), named `label_for(path)`; none for the rest;
    - `pub(crate) const SPACE_CLAIM_AXIS_SIZES: [(&str, &str); 3]` (name, size path);
    - private: `fn joint_term(joint: &str, cause: Cause) -> String` (moved from the tests) and `JointSide::quoted(&self, cause: Cause) -> (&'static str, usize)` beside `JointSide::fixes`.
  - `gui::size_controls`: `pub(crate) fn slider_minimum(legacy: &SizeControls, preference: Option<f64>) -> f64`; `pub(crate) enum SliderMapping { Diameter { minimum: f64 }, Length, Input { path: &'static str } }` with `pub(crate) fn design(self, inputs: &DesignInputs, value: f64) -> Result<DesignInputs, String>` (derives `Clone, Copy, Debug, PartialEq`).
  - `gui::test_support` (all `pub(crate) fn ... -> crate::DesignInputs`): `mixed_lengths()`, `reverse_lengths()`, `no_back_iron()`, `no_back_iron_negative_calibration()`, `end_effect_at_a_step()`, `odd_count()`, `keyway_through()`, `short_cavity()`, `temperature_edge()`, `forty_narrow_magnets()`, `zero_torque()`, `with_variation(variation: f64)`.

- [ ] **Step 1: Write the failing tests**

(a) `magcoupling-rs/src/engine/explain/plain.rs`, the end of the test module: the matcher's own positive and negative examples (spec test 16's matching rule: "levers", "Rank" match; "almost", "bestow" do not). Task 4 runs it over every card line.

Old:
```rust
        for text in [
            "The gearbox check compares cold-high at the gearbox input with its rating.",
            "The sleeve protects the magnets.",
            "",
        ] {
            assert!(!mentions_gearbox_protection(text), "{text}");
        }
    }
}
```

New:
```rust
        for text in [
            "The gearbox check compares cold-high at the gearbox input with its rating.",
            "The sleeve protects the magnets.",
            "",
        ] {
            assert!(!mentions_gearbox_protection(text), "{text}");
        }
    }

    #[test]
    fn the_card_ranking_words_match_whole_in_any_case_with_an_s_added_or_taken_off() {
        // Slice 3 spec 4.4 (owner decision 14A): the torque card's ban list, owner-confirmed.
        assert_eq!(
            CARD_RANKING_WORDS,
            [
                "strongest",
                "biggest",
                "best",
                "most",
                "lever",
                "dominant",
                "ranks"
            ]
        );
        // Each listed word says itself: as it is, in capitals, with an s added, and inside a
        // sentence.
        for listed in CARD_RANKING_WORDS {
            for text in [
                listed.to_owned(),
                listed.to_uppercase(),
                format!("{listed}s"),
                format!("The {listed}, here."),
            ] {
                assert_eq!(ranking_words_in(&text), [listed], "{text:?}");
            }
        }
        // The spec's examples, an s taken off ("rank"), and several words in the list's order,
        // each once.
        for (text, said) in [
            ("levers", &["lever"][..]),
            ("Rank", &["ranks"]),
            ("rank", &["ranks"]),
            ("the most-used lever", &["most", "lever"]),
            (
                "Ranks: the STRONGEST lever, then the strongest LEVERS.",
                &["strongest", "lever", "ranks"],
            ),
        ] {
            assert_eq!(ranking_words_in(text), said, "{text:?}");
        }
        // A listed word inside a longer word is not that word.
        for text in [
            "almost",
            "bestow",
            "mostly",
            "utmost",
            "leverage",
            "dominance",
            "biggestly",
            "",
        ] {
            assert_eq!(ranking_words_in(text), Vec::<&str>::new(), "{text:?}");
        }
    }
}
```

(b) `magcoupling-rs/src/gui/size_controls.rs`, the end of `mod tests` (before `mod auto_tests`): `slider_minimum`'s rule at the defaults, at a loaded design smaller than the range's start (the ring at 7.5 mm through the slider's own mapping) and at preferences inside and past the Minimum OD's range; and the controls' own value, which a loaded smaller design pins on its first frame, is `slider_minimum`'s. Then `SliderMapping`'s own test (plan review 2.3): each Size slider makes its design through its own mapping, the engine's resize functions and the inputs' own `set` (the count a whole number), and a value its mapping refuses says why.

Old:
```rust
        // Spec 3.5's help caps: 1-2 sentences, at most 30 words.
        for hover in [DIAMETER_HOVER, LENGTH_HOVER] {
            assert!(word_count(hover) <= 30, "{hover}");
            assert!(
                hover.ends_with('.') && (1..=2).contains(&sentence_count(hover)),
                "{hover}"
            );
        }
    }
}
```

New:
```rust
        // Spec 3.5's help caps: 1-2 sentences, at most 30 words.
        for hover in [DIAMETER_HOVER, LENGTH_HOVER] {
            assert!(word_count(hover) <= 30, "{hover}");
            assert!(
                hover.ends_with('.') && (1..=2).contains(&sentence_count(hover)),
                "{hover}"
            );
        }
    }

    #[test]
    fn the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown() {
        // Slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for the
        // controls and the torque card.
        use crate::engine::size_controls::with_ring_radius;
        let size =
            |inputs: &DesignInputs| dimensions(inputs).expect("the design has size controls");
        // Unset: the range's start at the default floor, below the diameter shown ...
        let defaults = size(&DesignInputs::default());
        assert!(defaults.diameter_range.min < defaults.diameter_mm);
        assert_eq!(slider_minimum(&defaults, None), defaults.diameter_range.min);
        // ... or the diameter shown, where a loaded design is smaller than that start (the ring
        // at 7.5 mm through the Outside diameter slider's own mapping).
        let smaller = with_ring_radius(&DesignInputs::default(), 7.5)
            .expect("7.5 mm is a ring radius the Outside diameter reaches");
        let smaller_size = size(&smaller);
        assert!(smaller_size.diameter_mm < smaller_size.diameter_range.min);
        assert_eq!(
            slider_minimum(&smaller_size, None),
            smaller_size.diameter_mm
        );
        // Set: the preference itself inside the Minimum OD's range (the diameter shown, as at a
        // user's Minimum OD of 42.8 mm, included), else the nearer end of that range.
        let range = defaults.minimum_diameter_range;
        for inside in [
            range.min,
            (range.min + range.max) / 2.0,
            defaults.diameter_mm,
        ] {
            assert_eq!(slider_minimum(&defaults, Some(inside)), inside);
        }
        assert_eq!(slider_minimum(&defaults, Some(range.min - 1.0)), range.min);
        assert_eq!(slider_minimum(&defaults, Some(range.max + 1.0)), range.max);
        // The controls' own: on its first frame a loaded smaller design pins its starting floor,
        // the slider's minimum; the defaults pin none.
        let ctx = egui::Context::default();
        let pinned = |inputs: &DesignInputs| {
            let consequences = Consequences::of(inputs, &compute_all(inputs), None);
            let mut minimum = None;
            sized_frame(&ctx, SCREEN, Vec::new(), |ui| {
                minimum = controls_ui(ui, inputs, None, &consequences, None).minimum_diameter_mm;
            });
            minimum
        };
        assert_eq!(pinned(&smaller), Some(slider_minimum(&smaller_size, None)));
        assert_eq!(pinned(&DesignInputs::default()), None);
    }

    #[test]
    fn each_size_slider_makes_its_design_through_its_own_mapping() {
        // Slice 3 spec 3.3 (owner 3A): one mapping per Size-tab slider, which the controls apply
        // on an edit and the torque card on each step. The Outside diameter through
        // resize_diameter_with_minimum at the slider's lowest value, the Overall length through
        // resize_length, the gap and the magnet count as their inputs (the count whole); a value
        // a mapping refuses says why.
        let inputs = DesignInputs::default();
        let size = dimensions(&inputs).expect("the defaults have size controls");
        let minimum = slider_minimum(&size, None);
        let diameter = size.diameter_mm + 1.0;
        assert_eq!(
            SliderMapping::Diameter { minimum }.design(&inputs, diameter),
            resize_diameter_with_minimum(&inputs, diameter, minimum)
        );
        let length = size.overall_length_mm + 1.0;
        assert_eq!(
            SliderMapping::Length.design(&inputs, length),
            resize_length(&inputs, length)
        );
        let gap = SliderMapping::Input {
            path: "metal.face_gap_mm",
        }
        .design(&inputs, 1.5)
        .expect("a gap on its slider");
        let mut expected = inputs.clone();
        expected.metal.face_gap_mm = 1.5;
        assert_eq!(gap, expected);
        let count = SliderMapping::Input {
            path: FreeVariable::MagnetsPerRing.path(),
        }
        .design(&inputs, 8.0)
        .expect("a count on its slider");
        let mut expected = inputs.clone();
        expected.coupling.npole = 8;
        assert_eq!(count, expected);
        assert!(
            SliderMapping::Length
                .design(&inputs, size.length_range.max + 1.0)
                .is_err()
        );
        assert!(
            SliderMapping::Input {
                path: "metal.face_gap_mm",
            }
            .design(&inputs, f64::NAN)
            .is_err()
        );
    }
}
```

(c) `magcoupling-rs/src/gui/guidance.rs`, the test module. Its `test_support` import gains `keyway_through`:

Old:
```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, drawn_texts, hot_day, hub_overstressed, is_shaft_joint,
        mixed_rings, n42_parts, only_the_clamp_fails, only_the_gearbox_fails, only_the_joint_fails,
        passing_design, reversed_hub_band, shaft_joint_input_rows, shaft_joint_result_rows,
        short_magnets, sized_frame, slider_range, small_boss, with_poles,
    };
```

New:
```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, drawn_texts, hot_day, hub_overstressed, is_shaft_joint,
        keyway_through, mixed_rings, n42_parts, only_the_clamp_fails, only_the_gearbox_fails,
        only_the_joint_fails, passing_design, reversed_hub_band, shaft_joint_input_rows,
        shaft_joint_result_rows, short_magnets, sized_frame, slider_range, small_boss, with_poles,
    };
```

The test-private `Term` goes (Step 4's `CheckTerm` replaces it; the fix tests read its three shared fields):

Old:
```rust
    // ------------------------------------------------------------------ the fixes help

    /// One term of a check's verdict: its margin, larger is better, and whether it passes as
    /// the verdict compares it.
    #[derive(Clone, Debug)]
    struct Term {
        name: String,
        margin: f64,
        passes: bool,
    }

    /// The clamp's margin for the fix tests:
```

New:
```rust
    // ------------------------------------------------------------------ the fixes help

    /// The clamp's margin for the fix tests:
```

`terms()` reads the production table and adds the clamp's own term; `joint_term` leaves the tests (Step 4 adds it to production, moved):

Old:
```rust
    /// The terms of `check`'s verdict in a fixed order (two designs' lists pair up). The
    /// temperature terms are the guide's own ([`TemperatureTerm`]: a test holds them to the
    /// verdict), named as they print.
    fn terms(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Vec<Term> {
        let term = |name: &str, margin: f64, passes| Term {
            name: name.to_owned(),
            margin,
            passes,
        };
        let at_least_zero = |name, margin: f64| term(name, margin, margin >= 0.0);
        let (m, md) = (&results.model, &results.metal);
        match check {
            HOT_MIN => vec![at_least_zero(
                "hot-low torque",
                md.torque_hot_low_Nm - inputs.metal.required_min_Nm,
            )],
            CLEARANCE => vec![at_least_zero(
                "running clearance",
                md.min_running_clearance_mm - inputs.metal.residual_target_mm,
            )],
            CUP_WALL => vec![at_least_zero(
                "cup wall",
                results.materials.cup_wall_corner_mm
                    - results.materials.backiron_thickness_needed_mm,
            )],
            TEMPERATURE => TemperatureTerm::ALL
                .into_iter()
                .map(|t| {
                    let margin = t.margin(inputs, results);
                    term(&format!("{t:?}"), margin, !t.fails(inputs, results))
                })
                .collect(),
            CLAMP => vec![term(
                "screw size",
                clamp_margin(inputs, results),
                results.clamps.index > 0,
            )],
            SPACE_CLAIM => vec![
                at_least_zero("diameter", md.diameter_reserve_mm),
                at_least_zero("overall length", md.axial_reserve_mm),
                at_least_zero("large-diameter bay", md.large_dia_reserve_mm),
            ],
            GEARBOX => vec![at_least_zero(
                "gearbox input torque",
                inputs.coupling.gearbox_input_rating_Nm - md.cold_high_input_Nm,
            )],
            // The engine's four terms of each joint, with the guide's margins (spec 5.2). A joint
            // that is not judged (no wall, a reversed band, or the check cannot be made) passes
            // them; with no keyway the key terms pass with a margin of +inf.
            SHAFT_JOINT => {
                // Not judged while the check cannot be made (`shaft_joints` is an `Err`).
                let judged = shaft_joints(inputs, results).is_ok();
                let links = shaft_joint_links(inputs, results);
                shaft_joint::joints(&inputs.shaft_joint, &links)
                    .iter()
                    .flat_map(|joint| {
                        CAUSES.map(|cause| {
                            let passes = !(judged && joint.determinate() && joint.fails(cause));
                            term(&joint_term(joint.name, cause), joint.margin(cause), passes)
                        })
                    })
                    .collect()
            }
            END_EFFECT => vec![term("end-effect factor", m.f_end, m.f_end > 0.0)],
            // The engine's rule ring by ring; the margin is the flat's slack, or what is left
            // of the arcs' share of the circle.
            FIT_CHECK => ring_fits(&inputs.coupling, m)
                .into_iter()
                .zip(["inner ring", "outer ring"])
                .map(|(ring, name)| {
                    let margin = match ring {
                        RingFit::Flat { width_mm, flat_mm } => flat_mm - width_mm,
                        RingFit::Arc { share } => 1.0 - share,
                    };
                    term(name, margin, ring.fits())
                })
                .collect(),
            _ => panic!("no terms for {check}"),
        }
    }

    /// The name of the shaft joint `joint`'s term `cause` in [`terms`]: "inner hub slips", "cup
    /// bore stress", "inner hub fitting", "cup key".
    fn joint_term(joint: &str, cause: Cause) -> String {
        let term = match cause {
            Cause::Slips => "slips",
            Cause::BoreStress => "bore stress",
            Cause::CannotBeFitted => "fitting",
            Cause::Key => "key",
        };
        format!("{joint} {term}")
    }

    /// The margin of the term `name` of `check`.
```

New:
```rust
    /// The terms of `check`'s verdict in a fixed order (two designs' lists pair up): the
    /// production table ([`check_terms`]: the temperature terms are the guide's own,
    /// [`TemperatureTerm`], which a test holds to the verdict), and the clamp's own term, whose
    /// margin is the fix tests' ([`clamp_margin`]). Panics for a path that is no guide's check.
    fn terms(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Vec<CheckTerm> {
        assert!(guide(check).is_some(), "no terms for {check}");
        let mut terms = check_terms(check, inputs, results);
        if check == CLAMP {
            terms.push(CheckTerm {
                name: "screw size".to_owned(),
                margin: clamp_margin(inputs, results),
                passes: results.clamps.index > 0,
                quoted: None,
            });
        }
        terms
    }

    /// The margin of the term `name` of `check`.
```

Test 9 follows `failing_term_names`. Over every test design it holds `terms()` to `check_terms` term by term (the clamp's one term the tests' own); at the defaults it pins spec 3.6's table, each margin bit for bit by the table's expression and each quoted quantity's path, name and decimals; every quoted path is a number result in mm, µm, MPa, °C or N·m; the shaft joints keep eight terms in one order at a judged design, at two not judged (no wall past a keyway through the hub, a reversed band) and where the check cannot be made (a safety factor of 0: every term passes); and `SPACE_CLAIM_AXES`, `axis_reserves` and `SPACE_CLAIM_AXIS_SIZES` name the axes in one order (each reserve is its claim less the size the table names, bit for bit, at every test design; one step of each fix of an axis, from the defaults, shrinks that size and raises that reserve).

Old:
```rust
    fn failing_term_names(
        check: &str,
        inputs: &DesignInputs,
        results: &DesignResults,
    ) -> Vec<String> {
        terms(check, inputs, results)
            .into_iter()
            .filter(|t| !t.passes)
            .map(|t| t.name)
            .collect()
    }
```

New:
```rust
    fn failing_term_names(
        check: &str,
        inputs: &DesignInputs,
        results: &DesignResults,
    ) -> Vec<String> {
        terms(check, inputs, results)
            .into_iter()
            .filter(|t| !t.passes)
            .map(|t| t.name)
            .collect()
    }

    #[test]
    fn check_terms_keeps_the_fix_tests_margins() {
        // Slice 3 spec 3.6 (test 9): the counted checks' terms are one production table,
        // `check_terms`, which the torque card reads; the fix tests read it through `terms()`,
        // which adds only the clamp's own term, and pass unchanged.
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            for guide in &CHECK_GUIDES {
                let shared = check_terms(guide.check, &inputs, &results);
                let tested = terms(guide.check, &inputs, &results);
                if guide.check == CLAMP {
                    assert!(shared.is_empty(), "{name}: the clamp has no shared term");
                    let [screw] = tested.as_slice() else {
                        panic!("{name}: the clamp's one term: {tested:?}")
                    };
                    assert_eq!(screw.name, "screw size");
                    assert!(same(screw.margin, clamp_margin(&inputs, &results)));
                    assert_eq!(screw.passes, results.clamps.index > 0, "{name}");
                    assert_eq!(screw.quoted, None);
                    continue;
                }
                assert!(!shared.is_empty(), "{name}: {} has terms", guide.check);
                assert_eq!(tested.len(), shared.len(), "{name}: {}", guide.check);
                for (t, s) in tested.iter().zip(&shared) {
                    assert!(
                        t.name == s.name
                            && same(t.margin, s.margin)
                            && t.passes == s.passes
                            && t.quoted == s.quoted,
                        "{name}: {t:?} against {s:?}"
                    );
                }
                // Every quoted quantity is a number result in a unit of spec 3.6's table, drawn
                // through its metadata (never typed).
                for quoted in shared.iter().filter_map(|t| t.quoted) {
                    let unit = result_info(quoted.path)
                        .unwrap_or_else(|| panic!("{} is a result", quoted.path))
                        .meta
                        .unit;
                    assert!(
                        ["mm", "µm", "MPa", "°C", "N·m"].contains(&unit),
                        "{}: {unit}",
                        quoted.path
                    );
                    assert!(
                        matches!(results.get(quoted.path), Some(Value::Num(_))),
                        "{name}: {}",
                        quoted.path
                    );
                }
            }
            assert!(check_terms("no.such.check", &inputs, &results).is_empty());
        }

        // Spec 3.6's table at the defaults: each check's terms, margins (larger is better) and
        // quoted quantities. Hot-low torque, Temperature limits (T-13), the end effect and the
        // magnets' fit quote none.
        let defaults = DesignInputs::default();
        let before = compute_all(&defaults);
        let (md, materials) = (&before.metal, &before.materials);
        type Row = (String, u64, bool, Option<Quoted>);
        let read = |check: &str| -> Vec<Row> {
            check_terms(check, &defaults, &before)
                .into_iter()
                .map(|t| (t.name, t.margin.to_bits(), t.passes, t.quoted))
                .collect()
        };
        let row = |name: &str, margin: f64, quoted| -> Row {
            (name.to_owned(), margin.to_bits(), margin >= 0.0, quoted)
        };
        let quote = |path, name, decimals| {
            Some(Quoted {
                path,
                name,
                decimals,
            })
        };
        assert_eq!(
            read(HOT_MIN),
            [row(
                "hot-low torque",
                md.torque_hot_low_Nm - defaults.metal.required_min_Nm,
                None
            )]
        );
        assert_eq!(
            read(CLEARANCE),
            [row(
                "running clearance",
                md.min_running_clearance_mm - defaults.metal.residual_target_mm,
                quote("metal.min_running_clearance_mm", None, 2)
            )]
        );
        assert_eq!(
            read(CUP_WALL),
            [row(
                "cup wall",
                materials.cup_wall_corner_mm - materials.backiron_thickness_needed_mm,
                quote(
                    "materials.backiron_thickness_needed_mm",
                    Some(WALL_NEEDED),
                    2
                )
            )]
        );
        assert_eq!(
            read(GEARBOX),
            [row(
                "gearbox input torque",
                defaults.coupling.gearbox_input_rating_Nm - md.cold_high_input_Nm,
                quote("metal.cold_high_input_Nm", Some(COLD_HIGH), 2)
            )]
        );
        let axis = |index: usize| {
            let (name, size) = SPACE_CLAIM_AXIS_SIZES[index];
            quote(size, Some(name), 2)
        };
        assert_eq!(
            read(SPACE_CLAIM),
            [
                row("diameter", md.diameter_reserve_mm, axis(0)),
                row("overall length", md.axial_reserve_mm, axis(1)),
                row("large-diameter bay", md.large_dia_reserve_mm, axis(2)),
            ]
        );
        for check in [HOT_MIN, TEMPERATURE, END_EFFECT, FIT_CHECK] {
            assert!(
                read(check).iter().all(|(.., quoted)| quoted.is_none()),
                "{check}"
            );
        }
        assert_eq!(read(TEMPERATURE).len(), TemperatureTerm::ALL.len());

        // The shaft joints: eight terms in one order whether both joints are judged (the
        // defaults; the inner hub slipping), one is not (no wall past a keyway through the hub,
        // a reversed band) or the check cannot be made (a safety factor of 0), where every term
        // passes. Each quotes its joint's result for its cause, named by its label.
        let names: Vec<String> = [shaft_joint::INNER_HUB, shaft_joint::CUP]
            .into_iter()
            .flat_map(|joint| CAUSES.map(|cause| joint_term(joint, cause)))
            .collect();
        assert_eq!(
            names,
            [
                "inner hub slips",
                "inner hub bore stress",
                "inner hub fitting",
                "inner hub key",
                "cup slips",
                "cup bore stress",
                "cup fitting",
                "cup key",
            ]
        );
        let unmade = design(|i| i.shaft_joint.safety_factor = 0.0);
        for (name, inputs) in [
            ("defaults", defaults.clone()),
            ("joint_slips", only_the_joint_fails()),
            ("keyway through", keyway_through()),
            ("reversed band", breaking(reversed_hub_band)),
            ("cannot be made", unmade.clone()),
        ] {
            let joint_terms = check_terms(SHAFT_JOINT, &inputs, &compute_all(&inputs));
            let read_names: Vec<&str> = joint_terms.iter().map(|t| t.name.as_str()).collect();
            assert_eq!(read_names, names, "{name}");
        }
        let unmade_results = compute_all(&unmade);
        assert!(shaft_joints(&unmade, &unmade_results).is_err());
        assert!(
            check_terms(SHAFT_JOINT, &unmade, &unmade_results)
                .iter()
                .all(|t| t.passes)
        );
        let joints_quoted: Vec<Option<Quoted>> = read(SHAFT_JOINT)
            .into_iter()
            .map(|(.., quoted)| quoted)
            .collect();
        let per_cause: Vec<Option<Quoted>> = JOINT_SIDES
            .iter()
            .flat_map(|side| {
                [
                    (side.needed, 2),
                    (side.stress, 1),
                    (side.fit_temp, 1),
                    (side.key, 1),
                ]
            })
            .map(|(path, decimals)| quote(path, Some(label_for(path)), decimals))
            .collect();
        assert_eq!(joints_quoted, per_cause);

        // SPACE_CLAIM_AXES, axis_reserves and SPACE_CLAIM_AXIS_SIZES name the three axes in one
        // order: each reserve is its claim less the size SPACE_CLAIM_AXIS_SIZES names, bit for
        // bit, at every test design; and one step of each fix of an axis, from the defaults,
        // shrinks that size and raises that reserve.
        let claims = [
            "metal.max_diameter_mm",
            "metal.max_overall_axial_mm",
            "metal.max_large_dia_axial_mm",
        ];
        for (name, inputs) in designs() {
            let results = compute_all(&inputs);
            let reserves = axis_reserves(&results);
            for (index, (&(_, size), claim)) in
                SPACE_CLAIM_AXIS_SIZES.iter().zip(claims).enumerate()
            {
                let claim = inputs.get(claim).and_then(number).expect("a claim input");
                let size = result_number(&results, size);
                assert!(same(reserves[index], claim - size), "{name}: axis {index}");
            }
        }
        for (index, axis_fixes) in SPACE_CLAIM_AXES.iter().enumerate() {
            let size = SPACE_CLAIM_AXIS_SIZES[index].1;
            for &(path, direction) in *axis_fixes {
                let moved = moves(&defaults, &before, path, direction);
                assert!(!moved.is_empty(), "{path} moves from the defaults");
                for design in moved {
                    let after = compute_all(&design);
                    assert!(
                        result_number(&after, size) < result_number(&before, size),
                        "{path}: {size}"
                    );
                    assert!(
                        axis_reserves(&after)[index] > axis_reserves(&before)[index],
                        "{path}: axis {index}"
                    );
                }
            }
        }
    }
```

(d) `magcoupling-rs/src/gui/test_support.rs`, the end of the file: test 1a, in the file's own `mod tests`. It reads each fact through `fixed` to the decimals spec 2.1 quotes, the measured calibration factor bit for bit as `calibration.f_cal_updated`, and the checks by result path (the guides' labels are G-3's).

Old:
```rust
    let mut text = String::new();
    part.read_to_string(&mut text)
        .expect("an XML part is UTF-8");
    text
}
```

New:
```rust
    let mut text = String::new();
    part.read_to_string(&mut text)
        .expect("an XML part is UTF-8");
    text
}

mod tests {
    use super::*;
    use crate::engine::clamps::PAUSED;
    use crate::engine::model::blocks_fit;
    use crate::engine::size_controls::dimensions;
    use crate::gui::dashboard::Level;
    use crate::gui::format::fixed;
    use crate::gui::guidance::{counted_guides, level};
    use crate::gui::size_controls::predicts;
    use crate::{DesignInputs, compute_all};

    // The counted checks the facts name, by result path.
    const HOT_MIN: &str = "metal.hot_min_check";
    const CLEARANCE: &str = "metal.clearance_check";
    const CUP_WALL: &str = "materials.cup_wall_check";
    const TEMPERATURE: &str = "temperature.summary.verdict";
    const SHAFT_JOINT: &str = "shaft_joint.check";
    const GEARBOX: &str = "gearbox.cold_high_check";

    /// The counted checks (the shaft-clamp section paused, as shipped) at `want` on `inputs`, in
    /// the guides' order.
    fn checks_at(inputs: &DesignInputs, want: Level) -> Vec<&'static str> {
        let results = compute_all(inputs);
        counted_guides(PAUSED)
            .filter(|guide| level(guide, inputs, &results) == want)
            .map(|guide| guide.check)
            .collect()
    }

    #[test]
    fn the_shared_card_designs_read_as_section_2_1_says() {
        // Slice 3 spec 2.1, each number to the decimals the section quotes it to, so a changed
        // default that moves a design fails here first, not inside a card test.
        let defaults_fail = [HOT_MIN, CLEARANCE, CUP_WALL, GEARBOX];
        let none: [&str; 0] = [];
        let hot_low = |inputs: &DesignInputs| compute_all(inputs).metal.torque_hot_low_Nm;

        // The mixed-length pair and its reverse.
        let mixed = compute_all(&mixed_lengths());
        assert_eq!(fixed(mixed.metal.torque_hot_low_Nm, 4), "1.0099");
        assert_eq!(fixed(mixed.model.active_length_mm, 2), "6.35");
        assert_eq!(
            checks_at(&mixed_lengths(), Level::Bad),
            [HOT_MIN, CLEARANCE, CUP_WALL, SHAFT_JOINT]
        );
        assert_eq!(checks_at(&mixed_lengths(), Level::Caution), [TEMPERATURE]);
        assert_eq!(fixed(hot_low(&reverse_lengths()), 4), "1.0099");
        assert_eq!(
            checks_at(&reverse_lengths(), Level::Bad),
            [HOT_MIN, CLEARANCE, CUP_WALL]
        );
        assert_eq!(checks_at(&reverse_lengths(), Level::Caution), [TEMPERATURE]);

        // No back iron, and with its calibration coefficient at -0.95: the measured factor on
        // the design shown, the coefficient itself on its Magnets per ring -2 design.
        let two_fewer = |mut inputs: DesignInputs| {
            inputs.coupling.npole -= 2;
            inputs
        };
        for (name, inputs) in [
            ("no back iron", no_back_iron()),
            (
                "its coefficient at -0.95",
                no_back_iron_negative_calibration(),
            ),
        ] {
            let results = compute_all(&inputs);
            assert_eq!(
                results.model.f_cal.to_bits(),
                results.calibration.f_cal_updated.to_bits(),
                "{name}: the measured factor"
            );
            assert_eq!(fixed(results.model.f_cal, 4), "1.0495", "{name}");
            assert_eq!(
                fixed(results.metal.torque_hot_low_Nm, 4),
                "1.4428",
                "{name}"
            );
            assert_eq!(fixed(results.model.pullout_Nm, 4), "1.6974", "{name}");
            assert_eq!(
                checks_at(&inputs, Level::Bad),
                [HOT_MIN, CLEARANCE, SHAFT_JOINT],
                "{name}"
            );
            let fewer = compute_all(&two_fewer(inputs.clone()));
            assert_eq!(
                fewer.model.f_cal, inputs.calibration.f_cal_original,
                "{name}: its -2 design takes the coefficient"
            );
        }
        let fewer = compute_all(&two_fewer(no_back_iron_negative_calibration()));
        assert_eq!(fewer.model.f_cal, -0.95);
        assert_eq!(fixed(fewer.metal.torque_hot_low_Nm, 4), "-0.8505");
        assert_eq!(fixed(fewer.model.pullout_Nm, 4), "-1.0006");
        assert_eq!(fixed(fewer.model.f_end, 3), "0.870");
        assert!(predicts(&fewer, fewer.model.pullout_Nm));
        assert!(predicts(&fewer, fewer.metal.torque_hot_low_Nm));

        // The end effect at a step: in range at 5.0 mm, not at 4.0 mm.
        let at_a_step = compute_all(&end_effect_at_a_step());
        assert_eq!(fixed(at_a_step.metal.torque_hot_low_Nm, 4), "0.1196");
        assert!(predicts(&at_a_step, at_a_step.metal.torque_hot_low_Nm));
        let mut shorter = end_effect_at_a_step();
        shorter.coupling.magnets.axial_length_mm = Some(4.0);
        let shorter = compute_all(&shorter);
        assert!(!predicts(&shorter, shorter.metal.torque_hot_low_Nm));

        // An odd count: no size controls.
        assert_eq!(fixed(hot_low(&odd_count()), 4), "2.0295");
        assert!(dimensions(&odd_count()).is_none());

        // The keyway through the hub: no wall past it, so the joints are not judged.
        let keyway = compute_all(&keyway_through());
        assert!(keyway.model.hub_wall_past_key_mm < 0.0);
        assert_eq!(checks_at(&keyway_through(), Level::Caution), [SHAFT_JOINT]);

        // The short cavity: the defaults' torque and checks, a shorter Overall length.
        let size = dimensions(&short_cavity()).expect("the short cavity has size controls");
        assert_eq!(fixed(hot_low(&short_cavity()), 4), "2.2852");
        assert_eq!(checks_at(&short_cavity(), Level::Bad), defaults_fail);
        assert_eq!(
            [
                size.overall_length_mm,
                size.length_range.min,
                size.length_range.max,
            ]
            .map(|mm| fixed(mm, 2)),
            ["27.30", "18.30", "67.10"]
        );

        // The temperature edge: the defaults' torque and checks, Temperature limits passing.
        assert_eq!(fixed(hot_low(&temperature_edge()), 4), "2.2852");
        assert_eq!(checks_at(&temperature_edge(), Level::Bad), defaults_fail);
        assert_eq!(checks_at(&temperature_edge(), Level::Caution), none);

        // Forty narrow magnets fit.
        let forty = compute_all(&forty_narrow_magnets());
        assert!(blocks_fit(&forty_narrow_magnets().coupling, &forty.model));
        assert_eq!(fixed(forty.metal.torque_hot_low_Nm, 4), "0.4857");

        // Zero torque: both torques 0 N·m and both shown (the end effect in range).
        let zero = compute_all(&zero_torque());
        assert_eq!(
            (zero.model.pullout_Nm, zero.metal.torque_hot_low_Nm),
            (0.0, 0.0)
        );
        assert!(predicts(&zero, zero.model.pullout_Nm));
        assert!(predicts(&zero, zero.metal.torque_hot_low_Nm));
        assert_eq!(fixed(zero.model.f_end, 3), "0.896");
        assert_eq!(checks_at(&zero_torque(), Level::Bad), [HOT_MIN, CLEARANCE]);
        assert_eq!(
            checks_at(&zero_torque(), Level::Caution),
            [SHAFT_JOINT, GEARBOX]
        );

        // The variation allowance at 1.0 and 1.5: hot-low 0 and below it, the pull-out shown.
        let one = compute_all(&with_variation(1.0));
        assert_eq!(one.metal.torque_hot_low_Nm, 0.0);
        assert_eq!(fixed(one.model.pullout_Nm, 4), "2.6885");
        assert_eq!(fixed(one.model.f_end, 3), "0.896");
        assert!(predicts(&one, one.model.pullout_Nm));
        assert_eq!(fixed(hot_low(&with_variation(1.5)), 4), "-1.3442");
    }
}
```

- [ ] **Step 2: Run them and see the builds fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib --no-run`

Expected:
```text
error[E0425]: cannot find value `CARD_RANKING_WORDS` in this scope      (2 of these)
error[E0425]: cannot find function `ranking_words_in` in this scope      (3 of these)
error: could not compile `magcoupling-rs` (lib test) due to 5 previous errors
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib --no-run`

Expected (each line once unless counted):
```text
warning: unused import: `super::*`      (test_support.rs's new tests module: the designs do not exist yet)
error[E0412]: cannot find type `CheckTerm` in this scope
error[E0412]: cannot find type `Quoted` in this scope      (3 of these)
error[E0422]: cannot find struct, variant or union type `CheckTerm` in this scope
error[E0422]: cannot find struct, variant or union type `Quoted` in this scope
error[E0425]: cannot find function `check_terms` in this scope      (6 of these)
error[E0425]: cannot find function `joint_term` in this scope
error[E0425]: cannot find function `slider_minimum` in this scope      (7 of these)
error[E0433]: failed to resolve: use of undeclared type `SliderMapping`      (6 of these)
error[E0425]: cannot find function `ranking_words_in` in this scope      (3 of these)
error[E0425]: cannot find value `CARD_RANKING_WORDS` in this scope      (2 of these)
error[E0425]: cannot find value `COLD_HIGH` in this scope
error[E0425]: cannot find value `WALL_NEEDED` in this scope
error[E0425]: cannot find value `SPACE_CLAIM_AXIS_SIZES` in this scope      (3 of these)
error[E0425]: cannot find function `mixed_lengths` in this scope      (3 of these)
error[E0425]: cannot find function `reverse_lengths` in this scope      (3 of these)
error[E0425]: cannot find function `no_back_iron` in this scope
error[E0425]: cannot find function `no_back_iron_negative_calibration` in this scope      (2 of these)
error[E0425]: cannot find function `end_effect_at_a_step` in this scope      (2 of these)
error[E0425]: cannot find function `odd_count` in this scope      (2 of these)
error[E0425]: cannot find function `keyway_through` in this scope      (2 of these)
error[E0425]: cannot find function `short_cavity` in this scope      (3 of these)
error[E0425]: cannot find function `temperature_edge` in this scope      (3 of these)
error[E0425]: cannot find function `forty_narrow_magnets` in this scope      (2 of these)
error[E0425]: cannot find function `zero_torque` in this scope      (3 of these)
error[E0425]: cannot find function `with_variation` in this scope      (2 of these)
error[E0432]: unresolved import `crate::gui::test_support::keyway_through`
error: could not compile `magcoupling-rs` (lib test) due to 65 previous errors; 1 warning emitted
```

- [ ] **Step 3: The card's ban list and its matcher (`engine/explain/plain.rs`)**

Old:
```rust
/// Words that would order the coupling's torques (BL-064 spec 6.2, trap 2): no shaft-joint text
/// says them.
pub const TORQUE_ORDER_WORDS: [&str; 3] = ["strongest", "most expected", "most torque expected"];
```

New:
```rust
/// Words that would order the coupling's torques (BL-064 spec 6.2, trap 2): no shaft-joint text
/// says them.
pub const TORQUE_ORDER_WORDS: [&str; 3] = ["strongest", "most expected", "most torque expected"];

/// Words that would rank the Size tab's controls on the torque card (slice 3 spec 4.4, owner
/// decision 14A: its rows keep the controls' order and make no sorted claim): no card text says
/// them, matched by [`ranking_words_in`].
pub const CARD_RANKING_WORDS: [&str; 7] = [
    "strongest",
    "biggest",
    "best",
    "most",
    "lever",
    "dominant",
    "ranks",
];

/// The words of [`CARD_RANKING_WORDS`] that `text` says, in that list's order, each once. A word
/// of `text` (a run of letters and digits) says a listed word when it is that word whole, in any
/// case, or that word with an `s` added or taken off: "levers" and "Rank" say "lever" and
/// "ranks"; "almost" and "bestow" say none.
pub fn ranking_words_in(text: &str) -> Vec<&'static str> {
    let lower = text.to_lowercase();
    CARD_RANKING_WORDS
        .into_iter()
        .filter(|&listed| {
            words(&lower).any(|word| {
                word == listed
                    || word.strip_suffix('s') == Some(listed)
                    || listed.strip_suffix('s') == Some(word)
            })
        })
        .collect()
}
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --lib -- engine::explain::plain`

Expected: `test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 228 filtered out`.

- [ ] **Step 4: The check terms and the axis table (`gui/guidance.rs`)**

The imports (`CAUSES` and `label_for` were test-only; the test module's own imports of them stay and are harmless, as its `ring_fits` already is):

Old:
```rust
use crate::engine::shaft_joint::{
    self, CannotBeMade, Cause, Floor, HubFitLimit, ImpossibleValue, JointTerms, NoWall,
};
```

New:
```rust
use crate::engine::shaft_joint::{
    self, CAUSES, CannotBeMade, Cause, Floor, HubFitLimit, ImpossibleValue, JointTerms, NoWall,
};
```

Old:
```rust
use crate::gui::plain_text::display_label;
```

New:
```rust
use crate::gui::plain_text::{display_label, label_for};
```

Each joint's quoted result, beside its fixes (`JOINT_SIDES` already names every path):

Old:
```rust
impl JointSide {
    /// The fixes of the joint's `cause` term.
    const fn fixes(&self, cause: Cause) -> &'static [(&'static str, Direction)] {
        match cause {
            Cause::Slips => self.slip_fixes,
            Cause::BoreStress => self.stress_fixes,
            Cause::CannotBeFitted => self.fitting_fixes,
            Cause::Key => self.key_fixes,
        }
    }
}
```

New:
```rust
impl JointSide {
    /// The fixes of the joint's `cause` term.
    const fn fixes(&self, cause: Cause) -> &'static [(&'static str, Direction)] {
        match cause {
            Cause::Slips => self.slip_fixes,
            Cause::BoreStress => self.stress_fixes,
            Cause::CannotBeFitted => self.fitting_fixes,
            Cause::Key => self.key_fixes,
        }
    }

    /// The result the torque card quotes for the joint's `cause` term, and its decimals (slice 3
    /// spec 3.6): the smallest measured interference that holds [µm], the bore stress [MPa], the
    /// fitting temperature needed [°C] and the key pressure [MPa].
    const fn quoted(&self, cause: Cause) -> (&'static str, usize) {
        match cause {
            Cause::Slips => (self.needed, 2),
            Cause::BoreStress => (self.stress, 1),
            Cause::CannotBeFitted => (self.fit_temp, 1),
            Cause::Key => (self.key, 1),
        }
    }
}
```

The axes' names and sizes, typed once, after `axis_reserves`:

Old:
```rust
/// Each space-claim axis's reserve, in [`SPACE_CLAIM_AXES`]' order: the claim less the size,
/// larger is better, below zero when the axis is over its claim.
fn axis_reserves(results: &DesignResults) -> [f64; 3] {
    let m = &results.metal;
    [
        m.diameter_reserve_mm,
        m.axial_reserve_mm,
        m.large_dia_reserve_mm,
    ]
}
```

New:
```rust
/// Each space-claim axis's reserve, in [`SPACE_CLAIM_AXES`]' order: the claim less the size,
/// larger is better, below zero when the axis is over its claim.
fn axis_reserves(results: &DesignResults) -> [f64; 3] {
    let m = &results.metal;
    [
        m.diameter_reserve_mm,
        m.axial_reserve_mm,
        m.large_dia_reserve_mm,
    ]
}

/// Each space-claim axis's name and the result path of the size its reserve is the claim less,
/// in [`axis_reserves`]' order: the one table of the axes' names and sizes (slice 3 spec 3.6).
/// The space claim's Problem names an axis past its claim by it, and [`check_terms`] quotes the
/// size on the torque card.
pub(crate) const SPACE_CLAIM_AXIS_SIZES: [(&str, &str); 3] = [
    ("Rotating diameter", "metal.rotating_od_mm"),
    ("Overall length", "metal.axial_stack_mm"),
    ("Large-diameter stack", "metal.large_dia_stack_mm"),
];
```

The space claim's Problem reads them (its overshoot, overshoot path, claim path and claim name stay in its own table; the sentences are unchanged):

Old:
```rust
        let h = &results.housing;
        // (axis, its overshoot, the overshoot's path, the dimension's, the claim's, the claim's name)
        let axes = [
            (
                "Rotating diameter",
                h.diameter_overshoot_mm,
                "housing.diameter_overshoot_mm",
                "metal.rotating_od_mm",
                "metal.max_diameter_mm",
                "claim",
            ),
            (
                "Overall length",
                h.length_overshoot_mm,
                "housing.length_overshoot_mm",
                "metal.axial_stack_mm",
                "metal.max_overall_axial_mm",
                "claim",
            ),
            (
                "Large-diameter stack",
                h.bay_overshoot_mm,
                "housing.bay_overshoot_mm",
                "metal.large_dia_stack_mm",
                "metal.max_large_dia_axial_mm",
                "bay",
            ),
        ];
        let sentences = axes
            .iter()
            .filter_map(|&(axis, over, over_path, dimension, claim, name)| {
```

New:
```rust
        let h = &results.housing;
        // Each axis's overshoot, the overshoot's path, the claim's and the claim's name, in
        // SPACE_CLAIM_AXIS_SIZES' order (the axis's name and its dimension's path).
        let overshoots = [
            (
                h.diameter_overshoot_mm,
                "housing.diameter_overshoot_mm",
                "metal.max_diameter_mm",
                "claim",
            ),
            (
                h.length_overshoot_mm,
                "housing.length_overshoot_mm",
                "metal.max_overall_axial_mm",
                "claim",
            ),
            (
                h.bay_overshoot_mm,
                "housing.bay_overshoot_mm",
                "metal.max_large_dia_axial_mm",
                "bay",
            ),
        ];
        let sentences = SPACE_CLAIM_AXIS_SIZES
            .iter()
            .zip(overshoots)
            .filter_map(|(&(axis, dimension), (over, over_path, claim, name))| {
```

The check terms, after `failing` and before the fixes. Each margin is the arm `terms()` had, bit for bit; the space claim's now reads `axis_reserves` (the same three fields), as spec 3.6 asks.

Old:
```rust
/// Whether `check` fails or asks for a look in this design: a Problem is built only then.
fn failing(check: &str, inputs: &DesignInputs, results: &DesignResults) -> bool {
    level_of(check, inputs, results) != Level::Good
}
```

New:
```rust
/// Whether `check` fails or asks for a look in this design: a Problem is built only then.
fn failing(check: &str, inputs: &DesignInputs, results: &DesignResults) -> bool {
    level_of(check, inputs, results) != Level::Good
}

// ------------------------------------------------------------------ the check terms

/// The name the torque card gives the cup wall's quoted quantity, the wall the flux needs
/// (`materials.backiron_thickness_needed_mm`): "Cup wall: wall needed +0.07 mm" (slice 3 spec
/// 4.1; a draft until the slice's G-3 review sets it).
pub const WALL_NEEDED: &str = "wall needed";

/// The name the torque card gives the gearbox check's quoted quantity, cold-high at the gearbox
/// input (`metal.cold_high_input_Nm`): the torque it is (slice 3 spec T-11), "Gearbox input
/// torque: cold-high +0.03 N·m"; a draft until the slice's G-3 review sets it.
pub const COLD_HIGH: &str = "cold-high";

/// The quantity the torque card quotes for a term of a check that fails (slice 3 spec 3.6): the
/// result at `path`, named `name` after the check's label where the label alone does not name
/// it, to `decimals` places of the result's own unit (its metadata's, never typed).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Quoted {
    pub path: &'static str,
    pub name: Option<&'static str>,
    pub decimals: usize,
}

/// One term of a counted check's verdict (slice 3 spec 3.6): its name, its margin (larger is
/// better), whether it passes as the verdict compares it, and the quantity the torque card
/// quotes for it, if any.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CheckTerm {
    pub name: String,
    pub margin: f64,
    pub passes: bool,
    pub quoted: Option<Quoted>,
}

/// The terms of `check`'s verdict, in a fixed order so two designs' lists pair up: the one table
/// the fix tests' margins and the torque card's read (slice 3 spec 3.6). The temperature terms
/// are the guide's own ([`TemperatureTerm`]), named as they print, and quote nothing (T-13: they
/// read stored 3D fields that do not follow a step). The shaft joints' are the engine's four
/// causes of each joint, inner hub first ([`shaft_joint::joints`]), always eight: a joint not
/// judged (no wall, a reversed band) or a check that cannot be made passes them. The clamp has
/// none here (the fix tests add their own; its verdict moves with no single slider step), nor
/// has a path that is no guide's check.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the torque card's margins call it (slice 3 Task 4, which removes this)"
    )
)]
pub(crate) fn check_terms(
    check: &str,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> Vec<CheckTerm> {
    let term = |name: &str, margin: f64, passes, quoted| CheckTerm {
        name: name.to_owned(),
        margin,
        passes,
        quoted,
    };
    let at_least_zero = |name, margin: f64, quoted| term(name, margin, margin >= 0.0, quoted);
    let quote = |path, name, decimals| {
        Some(Quoted {
            path,
            name,
            decimals,
        })
    };
    let (m, md) = (&results.model, &results.metal);
    match check {
        // The row's own number is this term's quantity: never quoted again.
        HOT_MIN => vec![at_least_zero(
            "hot-low torque",
            md.torque_hot_low_Nm - inputs.metal.required_min_Nm,
            None,
        )],
        CLEARANCE => vec![at_least_zero(
            "running clearance",
            md.min_running_clearance_mm - inputs.metal.residual_target_mm,
            quote("metal.min_running_clearance_mm", None, 2),
        )],
        CUP_WALL => vec![at_least_zero(
            "cup wall",
            results.materials.cup_wall_corner_mm - results.materials.backiron_thickness_needed_mm,
            quote(
                "materials.backiron_thickness_needed_mm",
                Some(WALL_NEEDED),
                2,
            ),
        )],
        TEMPERATURE => TemperatureTerm::ALL
            .into_iter()
            .map(|t| {
                let margin = t.margin(inputs, results);
                term(&format!("{t:?}"), margin, !t.fails(inputs, results), None)
            })
            .collect(),
        SPACE_CLAIM => ["diameter", "overall length", "large-diameter bay"]
            .into_iter()
            .zip(axis_reserves(results))
            .zip(SPACE_CLAIM_AXIS_SIZES)
            .map(|((name, reserve), (axis, size))| {
                at_least_zero(name, reserve, quote(size, Some(axis), 2))
            })
            .collect(),
        GEARBOX => vec![at_least_zero(
            "gearbox input torque",
            inputs.coupling.gearbox_input_rating_Nm - md.cold_high_input_Nm,
            quote("metal.cold_high_input_Nm", Some(COLD_HIGH), 2),
        )],
        // The engine's four terms of each joint, with the guide's margins (BL-064 spec 5.2). A
        // joint that is not judged (no wall, a reversed band, or the check cannot be made) passes
        // them; with no keyway the key terms pass with a margin of +inf.
        SHAFT_JOINT => {
            // Not judged while the check cannot be made (`shaft_joints` is an `Err`).
            let judged = shaft_joints(inputs, results).is_ok();
            let links = shaft_joint_links(inputs, results);
            shaft_joint::joints(&inputs.shaft_joint, &links)
                .iter()
                .flat_map(|joint| {
                    CAUSES.map(|cause| {
                        let passes = !(judged && joint.determinate() && joint.fails(cause));
                        let quoted = joint_side(joint.name).and_then(|(_, side)| {
                            let (path, decimals) = side.quoted(cause);
                            quote(path, Some(label_for(path)), decimals)
                        });
                        term(
                            &joint_term(joint.name, cause),
                            joint.margin(cause),
                            passes,
                            quoted,
                        )
                    })
                })
                .collect()
        }
        END_EFFECT => vec![term("end-effect factor", m.f_end, m.f_end > 0.0, None)],
        // The engine's rule ring by ring; the margin is the flat's slack, or what is left of the
        // arcs' share of the circle.
        FIT_CHECK => ring_fits(&inputs.coupling, m)
            .into_iter()
            .zip(["inner ring", "outer ring"])
            .map(|(ring, name)| {
                let margin = match ring {
                    RingFit::Flat { width_mm, flat_mm } => flat_mm - width_mm,
                    RingFit::Arc { share } => 1.0 - share,
                };
                term(name, margin, ring.fits(), None)
            })
            .collect(),
        // CLAMP, and any path that is no guide's check.
        _ => Vec::new(),
    }
}

/// The name of the shaft joint `joint`'s term `cause` in [`check_terms`]: "inner hub slips",
/// "cup bore stress", "inner hub fitting", "cup key".
fn joint_term(joint: &str, cause: Cause) -> String {
    let term = match cause {
        Cause::Slips => "slips",
        Cause::BoreStress => "bore stress",
        Cause::CannotBeFitted => "fitting",
        Cause::Key => "key",
    };
    format!("{joint} {term}")
}
```

- [ ] **Step 5: The slider's minimum and each slider's mapping (`gui/size_controls.rs`)**

Old:
```rust
use crate::engine::size_controls::{
    dimensions, dimensions_with_minimum_diameter, resize_diameter_with_minimum, resize_length,
};
```

New:
```rust
use crate::engine::size_controls::{
    SizeControls, dimensions, dimensions_with_minimum_diameter, resize_diameter_with_minimum,
    resize_length,
};
```

`slider_minimum`, and beside it `SliderMapping` (plan review 2.3), before `ControlsOutput`:

Old:
```rust
pub(crate) struct ControlsOutput {
    pub inputs: DesignInputs,
```

New:
```rust
/// The Outside diameter slider's lowest value (slice 3 spec 3.3): the user's Minimum OD
/// `preference`, else the diameter range's start at the default floor, or the diameter shown
/// where that is lower (a loaded smaller design starts with its own diameter exposed), held to
/// the Minimum OD's own range. One expression for the controls and the torque card, each of
/// which already holds the design's [`dimensions`] (`legacy`), so it runs no geometry.
pub(crate) fn slider_minimum(legacy: &SizeControls, preference: Option<f64>) -> f64 {
    let range = legacy.minimum_diameter_range;
    preference
        .unwrap_or(legacy.diameter_range.min.min(legacy.diameter_mm))
        .clamp(range.min, range.max)
}

/// What a Size-tab slider does with a value (spec 3.3): the design its own mapping makes of it.
/// The controls apply it on an edit and the torque card on each step of each control (slice 3,
/// owner 3A: each control through its own slider's mapping), so a card side is the design the
/// slider itself makes, and a change of a mapping reaches both.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SliderMapping {
    /// The Outside diameter slider: [`resize_diameter_with_minimum`] at the slider's lowest
    /// value `minimum` ([`slider_minimum`]).
    Diameter { minimum: f64 },
    /// The Overall length slider: [`resize_length`].
    Length,
    /// The Magnet face gap or the Magnets per ring slider: its input at `path` set to the value
    /// ([`InputSet::set`]), the magnet count as a whole number.
    Input { path: &'static str },
}

impl SliderMapping {
    /// The design the slider makes of `value` from `inputs`, or why its mapping refuses it.
    pub(crate) fn design(self, inputs: &DesignInputs, value: f64) -> Result<DesignInputs, String> {
        match self {
            SliderMapping::Diameter { minimum } => {
                resize_diameter_with_minimum(inputs, value, minimum)
            }
            SliderMapping::Length => resize_length(inputs, value),
            SliderMapping::Input { path } => {
                let value = if path == FreeVariable::MagnetsPerRing.path() {
                    Value::Int(value as i64)
                } else {
                    Value::Num(value)
                };
                let mut design = inputs.clone();
                design.set(path, value).map_err(|error| error.to_string())?;
                Ok(design)
            }
        }
    }
}

pub(crate) struct ControlsOutput {
    pub inputs: DesignInputs,
```

`controls_ui` calls it (one expression; `range` stays, the Minimum OD slider and the floor line read it):

Old:
```rust
            let range = legacy.minimum_diameter_range;
            let mut minimum = minimum_diameter_mm
                .unwrap_or(legacy.diameter_range.min.min(legacy.diameter_mm))
                .clamp(range.min, range.max);
```

New:
```rust
            let range = legacy.minimum_diameter_range;
            let mut minimum = slider_minimum(&legacy, minimum_diameter_mm);
```

`controls_ui`'s size loop carries each slider's own mapping (plan review 2.3): the loop's head and its two tuples gain it,

Old:
```rust
            for (label, hover, path, hint, mut value, range, line) in [
                (
                    DIAMETER,
                    DIAMETER_HOVER,
                    "coupling.inner_back_apothem_mm",
                    DIAMETER_HINT_KEY,
                    size.diameter_mm,
                    size.diameter_range,
                    consequences.diameter_line(),
                ),
                (
                    LENGTH,
                    LENGTH_HOVER,
                    "coupling.magnets.axial_length_mm",
                    LENGTH_HINT_KEY,
                    size.overall_length_mm,
                    size.length_range,
                    consequences.length_line(),
                ),
            ] {
```

New:
```rust
            for (label, hover, path, hint, mut value, range, mapping, line) in [
                (
                    DIAMETER,
                    DIAMETER_HOVER,
                    "coupling.inner_back_apothem_mm",
                    DIAMETER_HINT_KEY,
                    size.diameter_mm,
                    size.diameter_range,
                    SliderMapping::Diameter { minimum },
                    consequences.diameter_line(),
                ),
                (
                    LENGTH,
                    LENGTH_HOVER,
                    "coupling.magnets.axial_length_mm",
                    LENGTH_HINT_KEY,
                    size.overall_length_mm,
                    size.length_range,
                    SliderMapping::Length,
                    consequences.length_line(),
                ),
            ] {
```

and its edit applies it, in place of the two mapping calls it typed:

Old:
```rust
                if response.changed() {
                    let resized = if path == "coupling.inner_back_apothem_mm" {
                        resize_diameter_with_minimum(&output.inputs, value, minimum)
                    } else {
                        resize_length(&output.inputs, value)
                    };
                    match resized {
                        Ok(resized) => output.inputs = resized,
                        Err(error) => output.error = Some(error),
                    }
                }
```

New:
```rust
                if response.changed() {
                    match mapping.design(&output.inputs, value) {
                        Ok(resized) => output.inputs = resized,
                        Err(error) => output.error = Some(error),
                    }
                }
```

The gap and count loop sets its input through the same mapping (`SliderMapping::Input`, the count as a whole number, as before; the error text is unchanged: `error.to_string()` is `format!("{error}")`):

Old:
```rust
            if response.changed() {
                let value = if path == "coupling.npole" {
                    Value::Int(value as i64)
                } else {
                    Value::Num(value)
                };
                if let Err(error) = output.inputs.set(path, value) {
                    output.error = Some(format!("{error}"));
                }
            }
```

New:
```rust
            if response.changed() {
                match (SliderMapping::Input { path }).design(&output.inputs, value) {
                    Ok(design) => output.inputs = design,
                    Err(error) => output.error = Some(error),
                }
            }
```

- [ ] **Step 6: The shared test designs (`gui/test_support.rs`)**

After `shaft_joints_off_default`, before `cup_without_wall_or_key_bearing`. The section comment names the test that pins the docs; each doc names its spec 2.1 key:

Old:
```rust
/// The defaults with a 16 mm bore in a 12 mm boss and no boss length (BL-064): the cup has no
```

New:
```rust
// The torque card's shared test designs (slice 3 spec section 2.1). Each doc states the
// design's fact there, and `tests::the_shared_card_designs_read_as_section_2_1_says` pins it, so
// a changed default that moves one fails there first, not inside a card test. "Counted checks"
// are the 9 guides that count while the shaft-clamp section is paused, as shipped.

/// The mixed-length pair (slice 3 spec 2.1, "mixed"): the library's B442 (6.35 mm long) inside
/// and B842SH (12.7 mm) outside, the axial override blank. Hot-low 1.0099 N·m over an active
/// length of 6.35 mm; it fails 4 of the 9 counted checks (Hot-low torque, Running clearance, Cup
/// wall, Shaft joints) and asks for a look at Temperature limits. Apart from [`mixed_rings`],
/// which mixes grades.
pub(crate) fn mixed_lengths() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.magnets.part_inner = "B442".to_owned();
    inputs.coupling.magnets.part_outer = "B842SH".to_owned();
    inputs
}

/// The mixed-length pair the other way round (slice 3 spec 2.1, "reverse"): B842SH inside, B442
/// outside, the axial override blank. Hot-low 1.0099 N·m, as [`mixed_lengths`]; it fails 3 of
/// the 9 counted checks (Hot-low torque, Running clearance, Cup wall) and asks for a look at
/// Temperature limits.
pub(crate) fn reverse_lengths() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.magnets.part_inner = "B842SH".to_owned();
    inputs.coupling.magnets.part_outer = "B442".to_owned();
    inputs
}

/// No back iron (slice 3 spec 2.1, "(c)"): `coupling.backiron` 0, the example's measured
/// calibration factor (1.0495, `model.f_cal`). Hot-low 1.4428 N·m; it fails 3 of the 9 counted
/// checks (Hot-low torque, Running clearance, Shaft joints). Its Magnets per ring `-2` design
/// takes the calibration coefficient itself, 0.95.
pub(crate) fn no_back_iron() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.backiron = 0;
    inputs
}

/// [`no_back_iron`] with a design file's calibration coefficient of -0.95
/// (`calibration.f_cal_original`; `validate()` passes it). The measured factor cancels it, so the
/// design reads as [`no_back_iron`]: factor 1.0495, hot-low 1.4428 and pull-out 1.6974 N·m,
/// failing the same 3 of 9. Its Magnets per ring `-2` design (8 poles) takes the coefficient
/// itself: factor -0.95, hot-low -0.8505 and pull-out -1.0006 N·m with the end-effect factor
/// 0.870, in range (slice 3 spec 2.1, "(c), `f_cal_original` -0.95").
pub(crate) fn no_back_iron_negative_calibration() -> crate::DesignInputs {
    let mut inputs = no_back_iron();
    inputs.calibration.f_cal_original = -0.95;
    inputs
}

/// An end-effect factor near its edge (slice 3 spec 2.1, "end effect at a step"): `c_end` 0.5
/// and the axial override at 5.0 mm. Hot-low 0.1196 N·m, with the end effect in range; one
/// millimetre shorter (the override at 4.0 mm) it is not.
pub(crate) fn end_effect_at_a_step() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.c_end = 0.5;
    inputs.coupling.magnets.axial_length_mm = Some(5.0);
    inputs
}

/// An odd magnet count from a design file (slice 3 spec 2.1, "odd count"): `coupling.npole` 9,
/// which the slider (even counts) does not take. Hot-low 2.0295 N·m; the Size tab has no size
/// controls for it (`engine::size_controls::dimensions` is `None`).
pub(crate) fn odd_count() -> crate::DesignInputs {
    with_poles(9)
}

/// A keyway through the hub (slice 3 spec 2.1, "keyway through"): a 16 mm bore with a 2.5 mm
/// keyway. The hub wall past the keyway is below zero (`model.hub_wall_past_key_mm`), so the
/// shaft joints have no wall to judge and ask for a look (Caution).
pub(crate) fn keyway_through() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.coupling.bore_mm = 16.0;
    inputs.coupling.keyway_depth_mm = 2.5;
    inputs
}

/// A cup cavity input shorter than the outer ring (slice 3 spec 2.1, "short cavity"):
/// `metal.cup_depth_mm` 11.0 against B842SH's 12.7 mm, the axial override blank. The design
/// reads as the defaults (hot-low 2.2852 N·m, failing the same 4 of 9 checks); its Overall
/// length is 27.30 mm on a range of 18.30 to 67.10 mm.
pub(crate) fn short_cavity() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.cup_depth_mm = 11.0;
    inputs
}

/// A hot day just under the temperature verdict's edge (slice 3 spec 2.1, "temperature edge":
/// the edge is at a hot-day ambient of 82.0913 °C): `temperature.duty.hot_ambient_C` 82.08.
/// Hot-low 2.2852 N·m, failing the defaults' 4 of 9 checks, with Temperature limits passing.
pub(crate) fn temperature_edge() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.temperature.duty.hot_ambient_C = 82.08;
    inputs
}

/// Forty narrow magnets that fit (slice 3 spec 2.1, "npole 40"): `coupling.npole` 40, both
/// parts blank and both rings' manual blocks 1.2 mm wide. The blocks fit; hot-low 0.4857 N·m.
pub(crate) fn forty_narrow_magnets() -> crate::DesignInputs {
    let mut inputs = with_poles(40);
    let magnets = &mut inputs.coupling.magnets;
    magnets.part_inner.clear();
    magnets.part_outer.clear();
    magnets.manual_inner_width_mm = 1.2;
    magnets.manual_outer_width_mm = 1.2;
    inputs
}

/// An unmagnetized inner ring from a design file (slice 3 spec 2.1, "zero torque"): both parts
/// blank and `coupling.magnets.manual_inner_br_T` 0 (the slider stops at 0.2 T; `set()` and
/// `validate()` check only that it is finite). The model predicts zero: pull-out and hot-low
/// 0 N·m with the end-effect factor 0.896, in range. It fails 2 of the 9 counted checks (Hot-low
/// torque, Running clearance) and asks for a look at 2 (Shaft joints, Gearbox input torque).
pub(crate) fn zero_torque() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    let magnets = &mut inputs.coupling.magnets;
    magnets.part_inner.clear();
    magnets.part_outer.clear();
    magnets.manual_inner_br_T = 0.0;
    inputs
}

/// The defaults with the variation allowance at `variation` (`metal.variation`; a design file
/// can hold any finite value). At 1.0 hot-low is 0 N·m while the model predicts (pull-out
/// 2.6885 N·m, end-effect factor 0.896); at 1.5 hot-low is -1.3442 N·m (slice 3 spec 2.1,
/// "variation 1.0").
pub(crate) fn with_variation(variation: f64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.variation = variation;
    inputs
}

/// The defaults with a 16 mm bore in a 12 mm boss and no boss length (BL-064): the cup has no
```

- [ ] **Step 7: Run the new tests and the fix tests and see them pass; plant six faults**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- check_terms_keeps_the_fix_tests_margins the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown each_size_slider_makes_its_design_through_its_own_mapping the_shared_card_designs_read_as_section_2_1_says the_card_ranking_words_match_whole_in_any_case_with_an_s_added_or_taken_off the_space_claim_problem_names_each_axis_past_its_claim`

Expected: `test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 935 filtered out`.

The eleven fix tests that read `terms()`, `margin()` or `failing_term_names()`, unchanged:

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red every_fix_moves_its_check_toward_passing_over_its_slider the_shaft_joints_fail_on_these_test_designs_and_pass_on_every_other raising_a_minimum_is_not_offered_where_one_step_reverses_the_band lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band raising_the_key_contact_is_not_offered_where_the_key_has_no_bearing_length with_no_keyway_both_key_terms_pass_with_an_infinite_margin each_shaft_joint_fix_moves_its_term_toward_passing_all_along_its_slider where_the_magnets_onset_binds_the_joint_offers_its_band_not_the_grade with_a_cool_adhesive_the_inner_hub_cannot_be_fitted_and_its_maximum_comes_down the_shaft_joint_problem_quotes_the_live_numbers`

Expected: `test result: ok. 11 passed; 0 failed; 0 ignored; 0 measured; 930 filtered out`.

Each planted fault below is one edit, run with the filter given (`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- <filter>`; the matcher's without `--features app`), then undone. Steps 3 to 6 add code beside tests that compile red in Step 2, and tests 1a and 9 hold refactors that change no behaviour, so these show each test bites:

| Planted fault | Filter | Expected |
|---|---|---|
| `short_cavity`: `inputs.metal.cup_depth_mm = 11.0;` becomes `12.0;` | `the_shared_card_designs_read_as_section_2_1_says` | fails at `test_support.rs:789`: `left: ["28.30", "18.30", "67.10"]`, `right: ["27.30", "18.30", "67.10"]` |
| `SPACE_CLAIM_AXIS_SIZES`: its first two rows swapped | `check_terms_keeps_the_fix_tests_margins the_space_claim_problem_names_each_axis_past_its_claim` | both fail; the Problem reads `"Rotating diameter 42.80 mm is 34.90 mm over the 35.00 mm claim. Large-diameter stack ..."` against `"Overall length 69.90 mm is 34.90 mm over the 35.00 mm claim. ..."`; `0 passed; 2 failed` |
| `ranking_words_in`: the line `\|\| listed.strip_suffix('s') == Some(word)` deleted (no `s` taken off) | `the_card_ranking_words_match_whole_in_any_case_with_an_s_added_or_taken_off` (no `--features app`) | fails at `plain.rs:515`: `"Rank"`, `left: []`, `right: ["ranks"]` |
| `slider_minimum`: `.clamp(range.min, range.max)` becomes `.max(range.min)` | `the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown` | fails at `size_controls.rs:637`: `left: 81.0`, `right: 80.0` (a preference past the Minimum OD range's top) |
| `check_terms`: `quote("metal.min_running_clearance_mm", None, 2),` becomes `.., 1),` | `check_terms_keeps_the_fix_tests_margins` | fails at the table at the defaults: `left: [("running clearance", .., false, Some(Quoted { path: "metal.min_running_clearance_mm", name: None, decimals: 1 }))]`, `right: [.. decimals: 2 ..]` |
| `SliderMapping::Length => resize_length(inputs, value),` becomes `SliderMapping::Length => resize_length(inputs, value + 0.1),` (plan review 2.3) | `each_size_slider_makes_its_design_through_its_own_mapping` | fails at `size_controls.rs:669`, its Length assertion: `left: Ok(DesignInputs { .. axial_length_mm: Some(13.799999999999997) .. })`, `right: Ok(DesignInputs { .. axial_length_mm: Some(13.699999999999996) .. })`. Run over `gui::size_controls gui::panel` it is the one test that fails (`200 passed; 1 failed`): no panel test pins an Overall length edit's design |

- [ ] **Step 8: Docs: `magcoupling-rs/README.md` and `docs/ai/03-structure.yaml`**

Twelve edits. Each "Old" is a substring of one line (the README's line numbers are after Task 1's four added lines); replace only it.

(a) `docs/ai/03-structure.yaml`, the `explain` line (`:64`, a double-quoted scalar). Old:
```text
TORQUE_ORDER_WORDS (BL-064, spec 6.2 trap 2: the words that would order the coupling's torques, the one list the shaft-joint guide's and note's tests ban), pure std so gate 4 runs it
```
New:
```text
TORQUE_ORDER_WORDS (BL-064, spec 6.2 trap 2: the words that would order the coupling's torques, the one list the shaft-joint guide's and note's tests ban), CARD_RANKING_WORDS and ranking_words_in (slice 3 spec 4.4, owner decision 14A: the torque card's ban list, strongest, biggest, best, most, lever, dominant and ranks, each matched whole in any case with an s added or taken off), pure std so gate 4 runs it
```

(b) The `gui` line (`:68`), `guidance.rs`'s clause. Old:
```text
counted_guides (those guides, in order), level, fix_text,
```
New:
```text
counted_guides (those guides, in order), level, check_terms -> Vec<CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>}> (slice 3 spec 3.6: each counted check's terms in a fixed order, the one table the fix tests' margins and the torque card's read; none for the clamp, whose own term the fix tests add; joint_term names the shaft joints' eight; WALL_NEEDED and COLD_HIGH name the cup wall's and the gearbox check's quoted quantities, drafts until the slice's G-3 review), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and size path in axis_reserves' order, read by the space claim's Problem and by check_terms), fix_text,
```

(c) The `gui` line, `size_controls.rs`'s clause. Old:
```text
variable_slider, change_text; ControlsOutput.note
```
New:
```text
variable_slider, change_text, slider_minimum (slice 3 spec 3.3: the Outside diameter slider's lowest value, one expression for controls_ui and the torque card); SliderMapping (each Size slider's own mapping of a value to a design: controls_ui applies it on an edit, the torque card on each step); ControlsOutput.note
```

(d) The `gui` line, `test_support.rs`'s clause (spec 9.2: "the `test_support.rs` clause gains the new test designs"; Task 6 adds `drawing_rect`). Old:
```text
cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the lookups
```
New:
```text
cup_without_wall_or_key_bearing (BL-064: a 16 mm bore in a 12 mm boss with no boss length, the cup's Lame rows NaN and its key pressure +inf), the torque card's shared test designs (slice 3 spec 2.1, each fact in its doc and pinned by its tests module's the_shared_card_designs_read_as_section_2_1_says): mixed_lengths and reverse_lengths (B442 and B842SH, each way round), no_back_iron and no_back_iron_negative_calibration (a design file's coefficient of -0.95), end_effect_at_a_step (c_end 0.5, the override at 5.0 mm), odd_count (9 magnets), keyway_through (a 16 mm bore, a 2.5 mm keyway), short_cavity (an 11 mm cup cavity), temperature_edge (a hot-day ambient of 82.08 °C), forty_narrow_magnets (40 blocks 1.2 mm wide), zero_torque (an unmagnetized inner ring) and with_variation(variation), the lookups
```

(e) README, the Layout table's `src/engine/explain/` row (around `:157`). Old:
```text
(`plain.rs`: `BANNED`, `plain_text_violations`, and the shared `word_count` and `sentence_count`; pure std so gate 4 runs it, re-exported by `gui::guidance`; slice 2, spec 3.4)
```
New:
```text
(`plain.rs`: `BANNED`, `plain_text_violations`, and the shared `word_count` and `sentence_count`; pure std so gate 4 runs it, re-exported by `gui::guidance`; slice 2, spec 3.4; slice 3 adds the torque card's ban list, `CARD_RANKING_WORDS`, and its whole-word match, `ranking_words_in`, spec 4.4)
```

(f) README, the Layout table's `src/gui/` row (around `:159`), `guidance.rs`'s entry. Old:
```text
`counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `all_fixes`,
```
New:
```text
`counted_guides` (the guides that count, `counts`, while the shaft-clamp section is paused or not), `check_terms` (`CheckTerm`, `Quoted`; slice 3: each counted check's terms in a fixed order, the one table the fix tests and the torque card read), `SPACE_CLAIM_AXIS_SIZES`, `all_fixes`,
```

(g) The same row, `size_controls.rs`'s entry (Task 5 rewrites its "consequence lines"). Old:
```text
`size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines and Why links)
```
New:
```text
`size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines and Why links; `slider_minimum`, the Outside diameter slider's lowest value, one expression with the torque card; `SliderMapping`, each Size slider's own mapping of a value to a design, one mapping for the controls and the torque card's steps)
```

(h) The same row, `test_support.rs`'s entry. Old:
```text
`test_support.rs` (headless egui helpers, tests only)
```
New:
```text
`test_support.rs` (headless egui helpers and the shared test designs, the torque card's among them; tests only)
```

(i) README, the Tests table's `src/engine/explain/plain.rs` row (around `:688`; spec 9.2's `:684`), its end. Old:
```text
(BL-059: the one rule the note, guide and overlay bans call) |
```
New:
```text
(BL-059: the one rule the note, guide and overlay bans call); the torque card's ban list (`CARD_RANKING_WORDS`, slice 3 spec 4.4: strongest, biggest, best, most, lever, dominant, ranks) matches a word whole, in any case, with an s added or taken off ("levers" and "Rank" match; "almost", "bestow", "mostly" and "utmost" do not), each listed word once, in the list's order |
```

(j) The `src/gui/guidance.rs` row (around `:697`; spec 9.2's `:693`, whose neighbour clauses Task 5 moves to the card's row), its end. Old:
```text
and with neither judged both sentences show, the inner hub's first, with no fix. |
```
New:
```text
and with neither judged both sentences show, the inner hub's first, with no fix. Slice 3: the counted checks' terms are one production table, `check_terms`, which the fix tests read through `terms()` (adding only the clamp's own term) and pass unchanged; it quotes spec 3.6's quantity per term (the running clearance, the wall needed, cold-high at the gearbox input, each space-claim axis's size, each shaft joint's result for its cause, named by its label; none for hot-low, the temperature terms, the end effect or the fit), each a number result in mm, µm, MPa, °C or N·m; the shaft joints' eight terms keep one order whether judged, not judged or not made (every term then passing), and the space claim's fixes, reserves and axis sizes one axis order (`check_terms_keeps_the_fix_tests_margins`). |
```

(k) The `src/gui/panel.rs` row (around `:690`), its `size_controls.rs` clause. Old:
```text
and Auto's no-valid-value line names the ring radius; slice 2, help batch A (T7):
```
New:
```text
and Auto's no-valid-value line names the ring radius; the Outside diameter slider's lowest value is one expression, `slider_minimum` (the user's Minimum OD, else the range's start or a loaded smaller design's own diameter, held to the Minimum OD's range), which the controls pin on a loaded smaller design's first frame (slice 3); each Size slider makes its design through its own mapping, `SliderMapping`, which the torque card's steps share (`each_size_slider_makes_its_design_through_its_own_mapping`, slice 3); slice 2, help batch A (T7):
```

(l) A new Tests-table row for `src/gui/test_support.rs`, before the `src/gui/plain_text.rs` row. Old (the start of that row):
```text
| `src/gui/plain_text.rs` (feature `gui`) |
```
New (the new row, a line break, then that row's start):
```text
| `src/gui/test_support.rs` (feature `gui`) | The torque card's shared test designs read as the slice 3 spec's section 2.1 says, each to the decimals the section quotes (`the_shared_card_designs_read_as_section_2_1_says`): the mixed-length pair and its reverse (hot-low 1.0099 N·m; the pair's active length 6.35 mm), no back iron (the measured factor 1.0495 on the design shown and the coefficient itself on its two-fewer-magnets design, also with a design file's coefficient of -0.95, there hot-low -0.8505 N·m), the end effect at a step (in range at 5.0 mm, not at 4.0 mm), the odd count (no size controls), the keyway through the hub (the shaft joints not judged), the short cavity (Overall length 27.30 mm on 18.30 to 67.10 mm), the temperature edge (Temperature limits passing), forty narrow magnets (the blocks fit), zero torque (pull-out and hot-low 0 N·m, both shown) and the variation allowance at 1.0 and 1.5 (hot-low 0 and -1.3442 N·m) |
| `src/gui/plain_text.rs` (feature `gui`) |
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`.

- [ ] **Step 9: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: lib `234 passed; 0 failed; 1 ignored` (the matcher's test is engine code, built in both suites); every other binary as in Task 1 Step 9.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `935 passed; 0 failed; 6 ignored` (930 after Task 1, plus test 9, test 1a, `slider_minimum`'s test, `SliderMapping`'s test and the matcher's); the two app binaries 0 tests; `explain` 64 passed, 1 ignored; every other binary as in Task 1 Step 9.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`, then the same without `--features app`.

Expected: `Finished`, no warning, both (the `expect(dead_code)` on `check_terms` holds: nothing outside the tests calls it yet).

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/engine/explain/plain.rs
 M magcoupling-rs/src/gui/guidance.rs
 M magcoupling-rs/src/gui/size_controls.rs
 M magcoupling-rs/src/gui/test_support.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
```

- [ ] **Step 10: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T2.txt` with the Write tool:
```text
feat(magcoupler): the torque card's shared prerequisites (BL-073 T2)

- gui::guidance::check_terms -> Vec<CheckTerm>: the counted checks' terms in a
  fixed order, the production half of the fix tests' test-private terms(),
  which now calls it and adds only the clamp's own term (the eleven fix tests
  that read it pass unchanged). Each term carries the quantity the torque card
  quotes (Quoted: path, name, decimals; spec 3.6): the running clearance, the
  wall needed (WALL_NEEDED), cold-high at the gearbox input (COLD_HIGH), each
  space-claim axis's size, each shaft joint's result for its cause; none for
  hot-low, the temperature terms (T-13), the end effect or the fit. The test's
  Term and joint_term move with it. Its caller, the card's margins, comes in
  Task 4, which removes its expect(dead_code).
- SPACE_CLAIM_AXIS_SIZES: the space-claim axes' names and size paths, typed
  once, read by the space claim's Problem (its text unchanged) and check_terms.
- gui::size_controls::slider_minimum: the Outside diameter slider's lowest
  value, one expression for controls_ui and the card; and SliderMapping, each
  Size slider's own mapping of a value to a design (plan review 2.3), which
  controls_ui applies on an edit and the card will apply on each step.
- engine::explain::plain::CARD_RANKING_WORDS and ranking_words_in: the card's
  ban list (owner 14A), each word matched whole in any case with an s added
  or taken off.
- gui::test_support: the card's shared test designs (spec 2.1), each fact in
  its doc and pinned by the_shared_card_designs_read_as_section_2_1_says
  (test 1a).

The README and docs/ai/03-structure.yaml follow.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/engine/explain/plain.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/size_controls.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T2.txt
git status --short
```

Expected: one commit of 6 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 2 moves** (each confirmed by running in the plan-time prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui/guidance.rs`'s test helper `terms()` (and its `Term`, `joint_term`) | `terms()` returns `check_terms` plus the clamp's own term (`CheckTerm`); `Term` gone; `joint_term` moved to production with `check_terms` | spec 3.6: one table for the fix tests and the card (DRY); the helper still panics for a path that is no guide's check |
| the eleven fix tests that read `terms()`, `margin()` or `failing_term_names()` (`each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red`, `every_fix_moves_its_check_toward_passing_over_its_slider`, `the_shaft_joints_fail_on_these_test_designs_and_pass_on_every_other`, `raising_a_minimum_is_not_offered_where_one_step_reverses_the_band`, `lowering_a_maximum_is_not_offered_where_one_step_reverses_the_band`, `raising_the_key_contact_is_not_offered_where_the_key_has_no_bearing_length`, `with_no_keyway_both_key_terms_pass_with_an_infinite_margin`, `each_shaft_joint_fix_moves_its_term_toward_passing_all_along_its_slider`, `where_the_magnets_onset_binds_the_joint_offers_its_band_not_the_grade`, `with_a_cool_adhesive_the_inner_hub_cannot_be_fitted_and_its_maximum_comes_down`, `the_shaft_joint_problem_quotes_the_live_numbers`) | unchanged, pass | their margins come from `check_terms`, bit for bit |
| `gui::guidance::tests::the_space_claim_problem_names_each_axis_past_its_claim` | unchanged, passes | the Problem reads its axis names and sizes from `SPACE_CLAIM_AXIS_SIZES`; its sentences are the same |
| the Size tab's Minimum OD tests (`gui/panel.rs`) | unchanged, pass | `controls_ui` calls `slider_minimum`, its own expression, and applies `SliderMapping` on an edit (the same designs) |

---

### Task 3: The torque card's module, part 1: its types, each control's step through its own slider, and the order of the tests (`gui/torque_card.rs`)

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec sections 3.1 (the types), 3.2 (`card`, `rows`) and 3.3 (each control's step and the order of the tests), T-2 and T-5, with tests 1, 3, 4, 5 and 6 of section 8.1. A new pure module, `magcoupling-rs/src/gui/torque_card.rs` (no egui), public (`pub mod torque_card;`):
- the types of 3.1 under their binding names: `Control` (`ALL`, `step`, `label`, `held_by`), `Words`, `Flip`, `MarginChange`, `Change`, `Side`, `Row` and `TorqueCard`, with `TorqueCard::rows`;
- `card(inputs, results, minimum_diameter_mm, clamps_paused)`: each control one fixed step each way through its slider's own mapping (3.3: each through `size_controls::SliderMapping`, the mapping the Size tab's controls apply (Task 2, plan review 2.3): the diameter at `slider_minimum`, the length, the gap, and the count after `count_step`, set as the slider sets it), in the order of the tests. Test 0 is the design shown, by the readouts' own rule (`size_controls::predicts`, through one helper, `predicts_both`, that step 6 reads too). Then the size rows only with size controls (T-5); the slider's range within `STEP_SLACK_MM` (1e-6 mm; a value within the slack past an end is taken as that end); the mapping; `side_words`; on the Overall length row the widened jump rule (`|Δactive − sign × step| > STEP_SLACK_MM`; an overlap that is not a number reads the jump too); then a `Change` whatever its sign, with `calibration_changed`;
- `side_words(design, results, keyway)` and `count_step(npole, more)`, moved from the neighbour lines in `gui/guidance.rs` (`neighbour_outcome` and `neighbour_counts`: moved, not copied). The neighbour lines call them until Task 5 removes the lines; `prediction` stays with the lines and leaves with them.
- One of each shared piece (critique A-12, A-13, A-24): guidance's `FACE_GAP` becomes `pub(crate)` and the card and its table import it (no second constant); `gui::sizing::sizing_range(inputs, variable, minimum_diameter_mm)` holds the panel's rule for the range Auto searches, which the panel's own `sizing_range` method and the table's `auto_shown` call; `test_support::smoke_inputs()` is the gui-smoke design's inputs, which the table's (b) and Tasks 5 and 6 build from. They land in Step 1, (e) and (f), so the table compiles at Step 2 and the red is the card's alone. The panel's production code then has no use of `FreeVariable` until Task 5's `auto()`, so its import moves to the panel's test module here and back in Task 5 (plan deviation PD-59).
- A test beyond the spec's (critique A-20, RF-3): `a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range`, a design file's gap of 5.5 mm, axial override of 60 mm and the Minimum OD at its top: both sides of that row read the end of its range, and a side back on the slider is the slider's own design.

The card's tests read one table of the spec's section 2 designs, `card_designs()` (code review 4 SF-4). It lives in `gui/test_support.rs`, beside the shared designs it reads, as `pub(crate)`: `CardDesign { key, inputs, minimum_diameter_mm }` with `card()` and `side(control, more)`, and the helpers `card_at` and `side_at`. Spec 8 puts the table in the card's test module, but the G-3 review dump that Task 5 extends (`guidance_strings_for_review`, in `gui/guidance.rs`'s tests; Task 7's interface) reads the same table, and no other module can reach a test module's items (plan deviation PD-17). Each key is the spec's section 2.1 key (2.3's for the count overflow), without its Markdown. Where a row holds several designs, ": <what differs>" follows the key, as Task 7's sheet greps them: "slider tops: gap 5.0", "(a), Min OD 45: 43.3 mm", "variation 1.0: 1.5". An Auto design is solved as the panel solves it, through `SizingRunner` and `gui::sizing::sized` over the panel's search range.

Facts the implementer meets on the way:
- The flips and the margins (3.6) and every text (4.1) are Task 4's. So here a `Change`'s `flips` and `margins` are empty, and `card` does not read `clamps_paused` yet. The unused parameter would warn `unused_variables`, which is an error under clippy's `-D warnings`. It carries `#[expect(unused_variables, reason = ...)]`, and Task 4 deletes it when the flips read the pause. The prototype checked it: the expectation is fulfilled here and clippy is clean. Task 1 and Task 2 use the same pattern (plan deviation PD-9).
- Every public item is reachable from outside the crate (`gui` and `torque_card` are public modules), so the module raises no `dead_code` warning. The `pub(crate)` items `side_words` and `count_step` have their callers in the neighbour lines; the card reads guidance's `FACE_GAP`.
- `rows()` and `card_designs()` land here, not in Task 4, because tests 1, 3, 4, 5 and 6 read them (plan deviation PD-17). Task 4 adds each design's Auto state to the table (`CardDesign.auto`). Its readers are tests 8 and 18, so adding the field now would warn `dead_code`.
- The neighbour lines keep their shipped words (`NEIGHBOUR_DOES_NOT_FIT`, `NEIGHBOUR_NO_PREDICTION`) until Task 4 moves them into the card as `SIDE_DOES_NOT_FIT` and `SIDE_NO_PREDICTION`. Their no-prediction rule becomes the readouts' (`predicts`), which equals the shipped one (spec 3.3 step 6), so the neighbour lines' seven tests pass unchanged.
- The neighbour rule's own test, `neighbours_are_even_counts_inside_the_slider_and_never_overflow`, calls `neighbour_counts`, which moves. Its pins move to `count_step_keeps_even_counts_inside_the_slider_and_never_overflows`, one side at a time. Its last assertion (an odd count draws the ring-size line and no neighbour line) reads the neighbour lines, which still ship. That assertion joins `at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown` until Task 5 moves that test.
- Two more tests go beyond the spec's (plan deviation PD-23). `each_control_is_a_size_tab_slider_with_its_step` pins the controls' order and steps, each label equal to the Size tab's drawn label constant, and `held_by` as the inverse of `size_controls::variable_slider`. `a_side_reads_a_number_exactly_when_sizing_counts_its_design` pins `side_words` against `sizing::is_valid` in its order, at every even count of the neighbour lines' designs, and decision D-1 with the keyway through the hub.
- The card's tests take about 0.4 s: the table solves eight Auto designs, and every card runs `compute_all` once for each side it compares.

Line numbers are after Task 2 (dae024f's code with Tasks 1 and 2); trust the code. Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`. Every code block is rustfmt's layout. In the plan-time prototype, replaying this task's blocks in the order given on Task 2's files reproduces this task's files byte for byte, and both suites, both clippy runs and fmt are then green.

**Files:**
- Create: `magcoupling-rs/src/gui/torque_card.rs`
- Modify: `magcoupling-rs/src/gui/mod.rs` (`pub mod torque_card;`), `magcoupling-rs/src/gui/guidance.rs` (its `FACE_GAP` `pub(crate)`; the neighbour lines call `count_step` and `side_words`; `neighbour_counts` and `neighbour_outcome` leave; the neighbour rule's test moves), `magcoupling-rs/src/gui/sizing.rs` (`sizing_range`; its `ring_radius_range` import), `magcoupling-rs/src/gui/panel.rs` (its `sizing_range` method calls `gui::sizing::sizing_range`; `ring_radius_range` leaves its imports and `FreeVariable` moves to its test module until Task 5)
- Modify (tests): `magcoupling-rs/src/gui/test_support.rs` (`smoke_inputs`; the table: `CardDesign`, `card_at`, `side_at`, `auto_shown`, `card_designs`)
- Modify (docs): `magcoupling-rs/README.md` (the Layout table's `src/gui/` row, `:155`: the module, the table in `test_support.rs`'s entry and `sizing.rs`'s `sizing_range`; the Tests table: a new `src/gui/torque_card.rs` row after the `src/gui/guidance.rs` row), `docs/ai/02-system.yaml` (`key_files`, `:82`), `docs/ai/03-structure.yaml` (the `gui` line, `:68`: the module's clause, the table's and `sizing.rs`'s)
- Test: `magcoupling-rs/src/gui/torque_card.rs`. New: `each_control_is_a_size_tab_slider_with_its_step`; `each_number_is_the_difference_of_two_compute_all_runs` (test 1); `the_words_follow_the_order_of_the_tests` (test 3); `the_design_shown_decides_first` (test 4); `the_size_rows_follow_the_size_controls` (test 5); `the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum` (test 6); `a_side_reads_a_number_exactly_when_sizing_counts_its_design`; `a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range`. Moved: `count_step_keeps_even_counts_inside_the_slider_and_never_overflows`.
- Test: `magcoupling-rs/src/gui/guidance.rs`: `at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown` takes the odd-count assertion.

**Interfaces:**
- Consumes:
  - Task 2's `gui::size_controls::slider_minimum(legacy: &SizeControls, preference: Option<f64>) -> f64` with `SliderMapping` beside it, and Task 2's test designs (`mixed_lengths`, `reverse_lengths`, `no_back_iron`, `no_back_iron_negative_calibration`, `end_effect_at_a_step`, `odd_count`, `keyway_through`, `short_cavity`, `temperature_edge`, `forty_narrow_magnets`, `zero_torque`, `with_variation`);
  - `gui::size_controls::{self, predicts, variable_slider, DIAMETER, LENGTH, GAP, MAGNETS}`;
  - `engine::size_controls::{SizeControls, dimensions, dimensions_with_minimum_diameter}` (the mappings come through `SliderMapping`);
  - `engine::sizing::FreeVariable` (`ALL`, `path`, `range`, `apply`), `engine::model::blocks_fit`, `engine::meta::SliderRange`;
  - `gui::inputs::InputCatalogue`, `gui::plain_text::label_for`, `gui::dashboard::Level`, `gui::guidance::FACE_GAP` (made `pub(crate)` here);
  - test-only, for the card's tests: `engine::size_controls::{resize_diameter_with_minimum, resize_length}` (tests 3 and 6 call them directly), `engine::sizing::is_valid`, `engine::clamps::PAUSED`, `gui::format::fixed`, `gui::test_support::{end_effect_at_a_step, keyway_through, no_back_iron, short_magnets, slider_range}`;
  - test-only, for the table in `gui/test_support.rs`: `gui::sizing::{SizingMode, SizingRunner, SizingState, sized, sizing_range}`, `engine::size_controls::{dimensions, dimensions_with_minimum_diameter, with_ring_radius}`, `gui::size_controls::{SliderMapping, slider_minimum}`, `gui::guidance::FACE_GAP`, and the module's own `hot_day`, `only_the_joint_fails`, `passing_design`, `short_magnets` and `with_poles`.
- Produces, in `magcoupling-rs/src/gui/torque_card.rs`. Every type derives `Clone, Debug, PartialEq`; `Control` and `Words` also derive `Copy, Eq`.
  - `pub enum Control { OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing }`. Methods: `pub const ALL: [Control; 4]`, `pub const fn step(self) -> f64`, `pub fn label(self) -> &'static str`, `pub fn held_by(self) -> Option<FreeVariable>`; private `fn signed_step(self, more: bool) -> f64`.
  - `pub enum Words { EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump }`.
  - `pub struct Flip { pub check: &'static str, pub to: Level }`.
  - `pub struct MarginChange { pub check: &'static str, pub name: Option<&'static str>, pub change: f64, pub unit: &'static str, pub decimals: usize }`.
  - `pub struct Change { pub hot_low_Nm: f64, pub change_Nm: f64, pub fraction: f64, pub calibration_changed: bool, pub diameter_change_mm: Option<f64>, pub length_change_mm: Option<f64>, pub flips: Vec<Flip>, pub margins: Vec<MarginChange> }`.
  - `pub enum Side { Words(Words), Change(Box<Change>) }`.
  - `pub struct Row { pub control: Control, pub fewer: Side, pub more: Side }`.
  - `pub enum TorqueCard { ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows(Vec<Row>) }`, with `pub fn rows(&self) -> &[Row]`.
  - `pub fn card(inputs: &DesignInputs, results: &DesignResults, minimum_diameter_mm: Option<f64>, clamps_paused: bool) -> TorqueCard`.
  - Crate-visible: `pub(crate) const STEP_SLACK_MM: f64 = 1e-6`, `pub(crate) fn side_words(design: &DesignInputs, results: &DesignResults, keyway: bool) -> Option<Words>`, `pub(crate) fn count_step(npole: i64, more: bool) -> Result<i64, Words>`.
  - Test-only, in the card's test module, which Task 4's tests read: `find(designs, key)`, `sides(row)`, `slider_step(control, inputs, minimum, more)` (through `SliderMapping`), `words(&Side)`, `number(&Side)`, `NUMBER_SIDES` (263).
- Produces, in `magcoupling-rs/src/gui/test_support.rs` (tests only, all `pub(crate)`):
  - `struct CardDesign { key: &'static str, inputs: DesignInputs, minimum_diameter_mm: Option<f64> }`, with `fn card(&self) -> TorqueCard` and `fn side(&self, control: Control, more: bool) -> Side`;
  - `fn card_at(inputs: &DesignInputs, minimum: Option<f64>) -> TorqueCard`, at the shipped clamp pause;
  - `fn side_at(inputs: &DesignInputs, minimum: Option<f64>, control: Control, more: bool) -> Side`;
  - `fn card_designs() -> Vec<CardDesign>`: 50 designs;
  - `fn smoke_inputs() -> crate::DesignInputs`: the gui-smoke share link's inputs before Auto solves them, the defaults with a 1.5 mm face gap (critique A-24);
  - private: `fn auto_shown(inputs, variable, minimum) -> DesignInputs`, over `gui::sizing::sizing_range`.
- Produces, in `magcoupling-rs/src/gui/sizing.rs`: `pub(crate) fn sizing_range(inputs: &DesignInputs, variable: FreeVariable, minimum_diameter_mm: Option<f64>) -> SliderRange`, the range Auto searches (critique A-13). In `gui::guidance`: `FACE_GAP` is `pub(crate)` (critique A-12); the card defines no second constant.

- [ ] **Step 1: Write the failing tests**

(a) Create `magcoupling-rs/src/gui/torque_card.rs` with its module doc, its imports and its test module; Step 3 adds the code between the imports and the tests. Test 1 rebuilds each number side's stepped design itself (`slider_step`: the stepped value held to the slider's range, then the mapping the Size tab's controls apply, `SliderMapping`), rather than reading it through the card. After test 6, a test beyond the spec's reads a design shown past a slider's end (critique A-20).

```rust
//! The torque card (slice 3 spec, BL-073; owner decisions 1A-17A): for each Size control, what
//! one fixed step each way does to the hot-low torque of the design shown. Pure: no egui. The
//! panel draws it and the screen dump prints it.
//!
//! Each side is the design the control's own slider mapping makes at exactly one step (owner
//! 3A: Outside diameter ±1 mm at the user's Minimum OD, Overall length ±1 mm, Magnet face gap
//! ±0.1 mm, Magnets per ring ±2), and its number is the difference of two `compute_all` runs. A
//! side with no honest number reads words, by the order of tests of spec 3.3: the design shown
//! first ([`TorqueCard::ShownDoesNotFit`], [`TorqueCard::ShownNoPrediction`],
//! [`TorqueCard::ShownNoHotLow`]), then the slider's range within [`STEP_SLACK_MM`], the
//! slider's mapping, the validity rule every row shares ([`side_words`]) and, on the Overall
//! length row, the jump.

use crate::engine::meta::SliderRange;
use crate::engine::model::blocks_fit;
use crate::engine::size_controls::{SizeControls, dimensions, dimensions_with_minimum_diameter};
use crate::engine::sizing::FreeVariable;
use crate::gui::dashboard::Level;
use crate::gui::guidance::FACE_GAP;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
use crate::gui::size_controls::{self, SliderMapping, predicts, slider_minimum};
use crate::{DesignInputs, DesignResults, compute_all};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::clamps::PAUSED;
    use crate::engine::size_controls::{resize_diameter_with_minimum, resize_length};
    use crate::engine::sizing::is_valid;
    use crate::gui::format::fixed;
    use crate::gui::test_support::{
        CardDesign, card_designs, end_effect_at_a_step, keyway_through, no_back_iron,
        short_magnets, side_at, slider_range,
    };

    /// The number sides of the cards of section 2's designs ([`card_designs`]): the tests that
    /// read every one count them, so none can pass by reading none.
    const NUMBER_SIDES: usize = 263;

    /// The design keyed `key` in `designs` ([`card_designs`]).
    fn find<'a>(designs: &'a [CardDesign], key: &str) -> &'a CardDesign {
        designs
            .iter()
            .find(|design| design.key == key)
            .unwrap_or_else(|| panic!("no card design {key}"))
    }

    /// A row's two sides, each with whether it is the `+` side.
    fn sides(row: &Row) -> [(bool, &Side); 2] {
        [(false, &row.fewer), (true, &row.more)]
    }

    /// The design one step of `control` makes from `inputs` through the control's own slider,
    /// written here and not through the card: the stepped value held to the slider's range
    /// (which a number side lies within [`STEP_SLACK_MM`] of), then the mapping the Size tab's
    /// controls apply ([`SliderMapping`]).
    fn slider_step(
        control: Control,
        inputs: &DesignInputs,
        minimum: Option<f64>,
        more: bool,
    ) -> DesignInputs {
        let step = control.signed_step(more);
        let size = || dimensions(inputs).expect("a size row has size controls");
        let held = |value: f64, range: SliderRange| value.clamp(range.min, range.max);
        let (mapping, value) = match control {
            Control::OutsideDiameter => {
                let minimum = slider_minimum(&size(), minimum);
                let slider =
                    dimensions_with_minimum_diameter(inputs, minimum).expect("a diameter range");
                (
                    SliderMapping::Diameter { minimum },
                    held(slider.diameter_mm + step, slider.diameter_range),
                )
            }
            Control::OverallLength => {
                let size = size();
                (
                    SliderMapping::Length,
                    held(size.overall_length_mm + step, size.length_range),
                )
            }
            Control::FaceGap => (
                SliderMapping::Input { path: FACE_GAP },
                held(inputs.metal.face_gap_mm + step, slider_range(FACE_GAP)),
            ),
            Control::MagnetsPerRing => (
                SliderMapping::Input {
                    path: FreeVariable::MagnetsPerRing.path(),
                },
                inputs.coupling.npole as f64 + step,
            ),
        };
        mapping.design(inputs, value).expect("on the slider")
    }

    /// The words of `side`; panics on a number.
    fn words(side: &Side) -> Words {
        match side {
            Side::Words(words) => *words,
            Side::Change(change) => panic!("a number, not words: {change:?}"),
        }
    }

    /// The number of `side`; panics on words.
    fn number(side: &Side) -> Change {
        match side {
            Side::Change(change) => (**change).clone(),
            Side::Words(words) => panic!("words, not a number: {words:?}"),
        }
    }

    #[test]
    fn each_control_is_a_size_tab_slider_with_its_step() {
        // Owner 2A, 3A, 6A and 11A: the four Size controls in the tab's order, each one's step,
        // the label drawn above its slider, and the Auto variable that sets it (the inverse of
        // `size_controls::variable_slider`; the gap is never Auto's).
        use size_controls::{DIAMETER, GAP, LENGTH, MAGNETS};
        assert_eq!(
            Control::ALL.map(|control| (control.label(), control.step(), control.held_by())),
            [
                (DIAMETER, 1.0, Some(FreeVariable::RingRadius)),
                (LENGTH, 1.0, Some(FreeVariable::AxialLength)),
                (GAP, 0.1, None),
                (MAGNETS, 2.0, Some(FreeVariable::MagnetsPerRing)),
            ]
        );
        for variable in FreeVariable::ALL {
            let held: Vec<Control> = Control::ALL
                .into_iter()
                .filter(|control| control.held_by() == Some(variable))
                .collect();
            assert_eq!(held.len(), 1, "{variable:?}");
            assert_eq!(held[0].label(), size_controls::variable_slider(variable));
        }
    }

    #[test]
    fn each_number_is_the_difference_of_two_compute_all_runs() {
        // Spec 8.1 test 1 (owner 1A): every number side, at every design of section 2, is the
        // stepped design's compute_all less the design shown's, bit for bit, the stepped design
        // rebuilt here through the slider's own mapping.
        let mut numbers = 0;
        for design in card_designs() {
            let before = compute_all(&design.inputs);
            let card = card(&design.inputs, &before, design.minimum_diameter_mm, PAUSED);
            for row in card.rows() {
                for (more, side) in sides(row) {
                    let Side::Change(change) = side else {
                        continue;
                    };
                    let what = format!("{} {:?} {more}", design.key, row.control);
                    let stepped = slider_step(
                        row.control,
                        &design.inputs,
                        design.minimum_diameter_mm,
                        more,
                    );
                    let after = compute_all(&stepped);
                    let hot_low = after.metal.torque_hot_low_Nm;
                    let change_nm = hot_low - before.metal.torque_hot_low_Nm;
                    assert_eq!(change.hot_low_Nm.to_bits(), hot_low.to_bits(), "{what}");
                    assert_eq!(change.change_Nm.to_bits(), change_nm.to_bits(), "{what}");
                    assert_eq!(
                        change.fraction.to_bits(),
                        (change_nm / before.metal.torque_hot_low_Nm).to_bits(),
                        "{what}"
                    );
                    // The sizes the sliders would show; none exactly where either design has no
                    // size controls.
                    let sizes = match (dimensions(&design.inputs), dimensions(&stepped)) {
                        (Some(shown), Some(stepped)) => [
                            Some(stepped.diameter_mm - shown.diameter_mm),
                            Some(stepped.overall_length_mm - shown.overall_length_mm),
                        ],
                        _ => [None, None],
                    };
                    assert_eq!(
                        [change.diameter_change_mm, change.length_change_mm]
                            .map(|mm| mm.map(f64::to_bits)),
                        sizes.map(|mm| mm.map(f64::to_bits)),
                        "{what}"
                    );
                    assert_eq!(
                        change.calibration_changed,
                        after.model.f_cal != before.model.f_cal,
                        "{what}"
                    );
                    numbers += 1;
                }
            }
        }
        assert_eq!(numbers, NUMBER_SIDES);
    }

    #[test]
    fn the_words_follow_the_order_of_the_tests() {
        // Spec 8.1 test 3: one design per word, in the order of spec 3.3's tests.
        use Control::{FaceGap, MagnetsPerRing, OutsideDiameter, OverallLength};
        let designs = card_designs();
        let design = |key| find(&designs, key);
        let a = design("(a)");

        // Step 4: the blocks do not fit, a step down the ring or two magnets more.
        assert_eq!(words(&a.side(OutsideDiameter, false)), Words::DoesNotFit);
        assert_eq!(words(&a.side(MagnetsPerRing, true)), Words::DoesNotFit);

        // Step 2: off the slider's range, the diameter's at the user's Minimum OD (owner 9A). At
        // 42.8 mm, the diameter shown, the -1 mm side leaves it (at the default it reached the
        // fit test above). At 45 mm, above the diameter shown, both sides lie below the range;
        // at 43.3 mm only the -1 mm side does, and the +1 mm side is (a)'s.
        for (key, more) in [
            ("(a), Min OD 42.8", false),
            ("(a), Min OD 45", false),
            ("(a), Min OD 45", true),
            ("(a), Min OD 45: 43.3 mm", false),
        ] {
            assert_eq!(
                words(&design(key).side(OutsideDiameter, more)),
                Words::EndOfRange,
                "{key} {more}"
            );
        }
        assert_eq!(
            number(&design("(a), Min OD 45: 43.3 mm").side(OutsideDiameter, true)),
            number(&a.side(OutsideDiameter, true))
        );
        // Each slider's own ends: the gap at its bottom and top, the count at 4 and 40 (forty
        // narrow magnets, which fit), the Outside diameter and the Overall length at theirs.
        for (key, control, more) in [
            ("gap at 0.3", FaceGap, false),
            ("slider tops: gap 5.0", FaceGap, true),
            ("npole 4", MagnetsPerRing, false),
            ("npole 40", MagnetsPerRing, true),
            ("slider tops: Outside diameter 80", OutsideDiameter, true),
            ("slider tops: Overall length 69.9", OverallLength, true),
            ("slider tops: Overall length 21.1", OverallLength, false),
        ] {
            assert_eq!(
                words(&design(key).side(control, more)),
                Words::EndOfRange,
                "{key}"
            );
        }
        // A count from a design file the slider does not take: odd, or overflowing (the design
        // shown does not fit there, so no row is reached; `count_step` itself is pinned by
        // `count_step_keeps_even_counts_inside_the_slider_and_never_overflows`).
        for more in [false, true] {
            assert_eq!(
                words(&design("odd count").side(MagnetsPerRing, more)),
                Words::NotOnTheSlider
            );
        }
        for key in ["count overflow: i64::MAX", "count overflow: i64::MIN"] {
            assert_eq!(design(key).card(), TorqueCard::ShownDoesNotFit, "{key}");
        }

        // Step 5: the keyway through the hub, on the diameter, length and gap rows; not on the
        // count row (decision D-1), whose -2 side is the number of the keyway intact.
        let keyway = design("keyway through");
        for control in [OverallLength, FaceGap] {
            for more in [false, true] {
                assert_eq!(
                    words(&keyway.side(control, more)),
                    Words::KeywayBreaksThrough,
                    "{control:?} {more}"
                );
            }
        }
        let mut intact = keyway.inputs.clone();
        intact.coupling.keyway_depth_mm = 0.0;
        assert_eq!(
            number(&keyway.side(MagnetsPerRing, false))
                .change_Nm
                .to_bits(),
            number(&side_at(&intact, None, MagnetsPerRing, false))
                .change_Nm
                .to_bits()
        );

        // Step 6: no prediction, a step shorter or two magnets fewer at the end effect's edge.
        let edge = design("end effect at a step");
        assert_eq!(words(&edge.side(OverallLength, false)), Words::NoPrediction);
        assert_eq!(
            words(&edge.side(MagnetsPerRing, false)),
            Words::NoPrediction
        );

        // Step 7: the jump, words whenever the overlapping magnet length changes by anything but
        // the step (O-1). The mixed pair jumps both ways; its reverse moves by the step.
        for more in [false, true] {
            assert_eq!(
                words(&design("mixed").side(OverallLength, more)),
                Words::Jump
            );
            assert!(matches!(
                design("reverse").side(OverallLength, more),
                Side::Change(_)
            ));
        }
        // The cup cavity shorter than the outer ring: at 11.0 mm both sides jump (red first: 9A as
        // written read the +1 mm side, -0.70 mm of active length, as a number); at 12.0 and
        // 12.5 mm the +1 mm side moves it +0.30 and +0.80 mm; at 10.7 mm it moves -1.00 mm, the
        // step's size the wrong way (an unsigned comparison read it as a number). At 12.7 and
        // 15.5 mm both sides move it by exactly the step.
        let cavity = |mm: f64| {
            let mut inputs = DesignInputs::default();
            inputs.metal.cup_depth_mm = mm;
            inputs
        };
        assert_eq!(
            words(&design("short cavity").side(OverallLength, false)),
            Words::Jump
        );
        for mm in [11.0, 12.0, 12.5, 10.7] {
            assert_eq!(
                words(&side_at(&cavity(mm), None, OverallLength, true)),
                Words::Jump,
                "{mm}"
            );
        }
        for mm in [12.7, 15.5] {
            for more in [false, true] {
                assert!(
                    matches!(
                        side_at(&cavity(mm), None, OverallLength, more),
                        Side::Change(_)
                    ),
                    "{mm} {more}"
                );
            }
        }

        // Step 8: a number whatever its sign. (c) with a design file's calibration coefficient
        // of -0.95: two magnets fewer take the coefficient itself, a hot-low below zero; and (c)
        // at 8 poles with a measured torque of 0: two magnets more take the prototype's measured
        // factor, 0, a hot-low of exactly 0 (owner 16A; code review 4 MF-2, physics review 5
        // SF-2).
        let negative = design("(c), f_cal_original -0.95");
        let fewer = number(&negative.side(MagnetsPerRing, false));
        let stepped = compute_all(&FreeVariable::MagnetsPerRing.apply(&negative.inputs, 8.0));
        assert!(fewer.calibration_changed);
        assert!(fewer.hot_low_Nm < 0.0, "{fewer:?}");
        assert_eq!(
            fewer.hot_low_Nm.to_bits(),
            stepped.metal.torque_hot_low_Nm.to_bits()
        );
        let mut unmeasured = no_back_iron();
        unmeasured.coupling.npole = 8;
        unmeasured.calibration.measured_torque_Nm = 0.0;
        let more = number(&side_at(&unmeasured, None, MagnetsPerRing, true));
        assert!(more.calibration_changed);
        assert_eq!(more.hot_low_Nm, 0.0, "{more:?}");

        // A step that lands on an end (the slack of step 2; code review 2 MF-1): one slider
        // step inside each end, the side is a number, the end's own design bit for bit (red
        // first: an exact range test read the length's +1 mm side at 68.9 mm as the end of its
        // range, 1.42e-14 mm past it).
        for (key, more) in [
            ("steps to an end: (a) at length 68.9", true),
            ("steps to an end: (a) at length 22.1", false),
        ] {
            let shown = design(key);
            let lengths = dimensions(&shown.inputs)
                .expect("size controls")
                .length_range;
            let end = if more { lengths.max } else { lengths.min };
            let at_end = resize_length(&shown.inputs, end).expect("the range's end");
            assert_eq!(
                number(&shown.side(OverallLength, more))
                    .hot_low_Nm
                    .to_bits(),
                compute_all(&at_end).metal.torque_hot_low_Nm.to_bits(),
                "{key}"
            );
        }
        for (key, control, more) in [
            ("steps to an end: (a) at gap 4.9", FaceGap, true),
            ("steps to an end: (a) at gap 0.4", FaceGap, false),
            (
                "steps to an end: (a) at Outside diameter 79",
                OutsideDiameter,
                true,
            ),
            (
                "steps to an end: (e) at Outside diameter 79",
                OutsideDiameter,
                true,
            ),
        ] {
            assert!(
                matches!(design(key).side(control, more), Side::Change(_)),
                "{key}"
            );
        }
        // (e) with its Minimum OD 1 mm under the diameter shown: the -1 mm side lands on the
        // range's start, the number at the default preference.
        assert_eq!(
            number(&design("steps to an end: (e), Min OD 41.8").side(OutsideDiameter, false)),
            number(&design("(e)").side(OutsideDiameter, false))
        );
    }

    #[test]
    fn the_design_shown_decides_first() {
        // Spec 8.1 test 4: the design shown decides before any row, by the readouts' own rule
        // (physics review 3 MF-2).
        let designs = card_designs();
        let design = |key| find(&designs, key).inputs.clone();
        let mut negative_coefficient = DesignInputs::default();
        negative_coefficient.calibration.f_cal_original = -0.95;
        let mut negative_pull_out = negative_coefficient.clone();
        negative_pull_out.metal.variation = 1.5;
        for (name, inputs, shown) in [
            // The end effect out of range: the readouts show no prediction.
            (
                "short magnets",
                design("short magnets"),
                TorqueCard::ShownNoPrediction,
            ),
            // The model predicts and the readouts show both torques, but hot-low, the percent's
            // base, is not above zero (a design file's variation).
            (
                "variation 1.0",
                design("variation 1.0"),
                TorqueCard::ShownNoHotLow,
            ),
            (
                "variation 1.0: 1.5",
                design("variation 1.0: 1.5"),
                TorqueCard::ShownNoHotLow,
            ),
            // Zero torque, shown as 0 N·m by both readouts (red first: revision 2's rule read
            // "no prediction" for a pull-out that is not above zero).
            (
                "zero torque",
                design("zero torque"),
                TorqueCard::ShownNoHotLow,
            ),
            // Both torques below zero; then the pull-out below zero with hot-low above it, which
            // a rule on hot-low alone would let through to the rows.
            (
                "f_cal_original -0.95",
                negative_coefficient,
                TorqueCard::ShownNoHotLow,
            ),
            (
                "f_cal_original -0.95, variation 1.5",
                negative_pull_out,
                TorqueCard::ShownNoHotLow,
            ),
            (
                "shown does not fit",
                design("shown does not fit"),
                TorqueCard::ShownDoesNotFit,
            ),
        ] {
            let results = compute_all(&inputs);
            let torques = [results.model.pullout_Nm, results.metal.torque_hot_low_Nm];
            // What the readouts show (`size_controls::predicts`), and the torques' signs.
            let shows_both = torques.iter().all(|&torque| predicts(&results, torque));
            let both_above_zero = torques.iter().all(|&torque| torque > 0.0);
            match shown {
                TorqueCard::ShownNoPrediction => assert!(!shows_both, "{name}"),
                TorqueCard::ShownNoHotLow => {
                    assert!(shows_both && !both_above_zero, "{name}: {torques:?}")
                }
                _ => assert!(!blocks_fit(&inputs.coupling, &results.model), "{name}"),
            }
            if name == "f_cal_original -0.95, variation 1.5" {
                assert!(torques[0] < 0.0 && torques[1] > 0.0, "{torques:?}");
            }
            let card = card(&inputs, &results, None, PAUSED);
            assert_eq!(card, shown, "{name}");
            assert!(card.rows().is_empty(), "{name}");
        }
    }

    #[test]
    fn the_size_rows_follow_the_size_controls() {
        // Spec 8.1 test 5 (T-5): the Outside diameter and Overall length rows only where the
        // Size tab draws their sliders; every row in the controls' order (owner 6A).
        let mut without_size_controls = Vec::new();
        for design in card_designs() {
            let card = design.card();
            let controls: Vec<Control> = card.rows().iter().map(|row| row.control).collect();
            if !matches!(card, TorqueCard::Rows(_)) {
                assert!(controls.is_empty(), "{}", design.key);
            } else if dimensions(&design.inputs).is_some() {
                assert_eq!(controls, Control::ALL, "{}", design.key);
            } else {
                assert_eq!(
                    controls,
                    [Control::FaceGap, Control::MagnetsPerRing],
                    "{}",
                    design.key
                );
                without_size_controls.push(design.key);
            }
        }
        assert_eq!(without_size_controls, ["odd count"]);
    }

    #[test]
    fn the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum() {
        // Spec 8.1 test 6 (owner 9A: the slider's own range at the user's Minimum OD).
        let designs = card_designs();
        let design = |key| find(&designs, key);
        // At (e) the lowest Minimum OD and the default map a step each way to the same rings,
        // 9.688060 and 10.611940 mm, so the row is the same.
        let (e, lowest) = (design("(e)"), design("(e), low Min OD"));
        let size = dimensions(&e.inputs).expect("(e) has size controls");
        for (step, ring) in [(-1.0, "9.688060"), (1.0, "10.611940")] {
            for preference in [e.minimum_diameter_mm, lowest.minimum_diameter_mm] {
                let minimum = slider_minimum(&size, preference);
                let stepped =
                    resize_diameter_with_minimum(&e.inputs, size.diameter_mm + step, minimum)
                        .expect("on the slider");
                assert_eq!(fixed(stepped.coupling.inner_back_apothem_mm, 6), ring);
            }
        }
        assert_eq!(lowest.card().rows()[0], e.card().rows()[0]);
        assert_eq!(e.card().rows()[0].control, Control::OutsideDiameter);
        // At (a) with Minimum OD 42.8 mm, the diameter shown, the slider starts there: the -1 mm
        // side leaves it and the +1 mm side reads as at the default.
        let (a, at_shown) = (design("(a)"), design("(a), Min OD 42.8"));
        assert_eq!(
            slider_minimum(
                &dimensions(&a.inputs).expect("(a) has size controls"),
                at_shown.minimum_diameter_mm
            ),
            42.8
        );
        assert_eq!(
            at_shown.side(Control::OutsideDiameter, false),
            Side::Words(Words::EndOfRange)
        );
        assert_eq!(
            at_shown.side(Control::OutsideDiameter, true),
            a.side(Control::OutsideDiameter, true)
        );
        // At every design of section 2 the row reads the Outside diameter slider at the user's
        // Minimum OD: the range at `slider_minimum` (the controls' own expression) starts at it,
        // and a side reads the end of its range exactly when its step lands more than
        // STEP_SLACK_MM outside that range.
        let mut rows = 0;
        for design in &designs {
            let card = design.card();
            let Some(row) = card
                .rows()
                .iter()
                .find(|row| row.control == Control::OutsideDiameter)
            else {
                continue;
            };
            let legacy = dimensions(&design.inputs).expect("a diameter row has size controls");
            let minimum = slider_minimum(&legacy, design.minimum_diameter_mm);
            let slider = dimensions_with_minimum_diameter(&design.inputs, minimum)
                .expect("a diameter range");
            assert_eq!(slider.diameter_range.min, minimum, "{}", design.key);
            for (more, side) in sides(row) {
                let requested = slider.diameter_mm + Control::OutsideDiameter.signed_step(more);
                let range = slider.diameter_range;
                let off =
                    requested < range.min - STEP_SLACK_MM || requested > range.max + STEP_SLACK_MM;
                assert_eq!(
                    *side == Side::Words(Words::EndOfRange),
                    off,
                    "{} {more}",
                    design.key
                );
            }
            rows += 1;
        }
        // Not vacuous: every diameter row of section 2's designs.
        assert_eq!(rows, 42);
    }

    #[test]
    fn a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range() {
        // A design file or link can carry a value past a slider's end (`validate()` checks only
        // that a number is finite). A side more than STEP_SLACK_MM outside the slider's range
        // reads the end of its range (spec 3.3 step 2): on both sides where the design shown
        // itself lies past an end; a side that lands back on the slider is a number, the
        // slider's own design.
        use Control::{FaceGap, OutsideDiameter, OverallLength};
        let with = |edit: fn(&mut DesignInputs)| {
            let mut inputs = DesignInputs::default();
            edit(&mut inputs);
            inputs
        };
        let gap_above = with(|i| i.metal.face_gap_mm = 5.5);
        let gap_below = with(|i| i.metal.face_gap_mm = 0.2);
        let long = with(|i| i.coupling.magnets.axial_length_mm = Some(60.0));
        for (inputs, control) in [(&gap_above, FaceGap), (&long, OverallLength)] {
            for more in [false, true] {
                assert_eq!(
                    words(&side_at(inputs, None, control, more)),
                    Words::EndOfRange,
                    "{control:?} {more}"
                );
            }
        }
        assert_eq!(
            words(&side_at(&gap_below, None, FaceGap, false)),
            Words::EndOfRange
        );
        let up = number(&side_at(&gap_below, None, FaceGap, true));
        let stepped = compute_all(&slider_step(FaceGap, &gap_below, None, true));
        assert_eq!(
            up.hot_low_Nm.to_bits(),
            stepped.metal.torque_hot_low_Nm.to_bits()
        );
        // The Minimum OD at its slider's top: the diameter's range is that one value, above the
        // diameter shown, so both sides lie off it.
        let top = dimensions(&DesignInputs::default())
            .expect("size controls")
            .minimum_diameter_range
            .max;
        for more in [false, true] {
            assert_eq!(
                words(&side_at(
                    &DesignInputs::default(),
                    Some(top),
                    OutsideDiameter,
                    more
                )),
                Words::EndOfRange
            );
        }
    }

    #[test]
    fn count_step_keeps_even_counts_inside_the_slider_and_never_overflows() {
        // Moved with the neighbour lines' rule (`neighbour_counts`, gui/guidance.rs), one side
        // at a time: two magnets each way, an even count inside the magnet slider's range.
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        assert_eq!(count_step(fewest + 4, false), Ok(fewest + 2));
        assert_eq!(count_step(fewest + 4, true), Ok(fewest + 6));
        // At the slider's ends only the step inside it.
        assert_eq!(count_step(fewest, false), Err(Words::EndOfRange));
        assert_eq!(count_step(most, true), Err(Words::EndOfRange));
        // A count from a design file may be odd or past the slider: odd inside the range is not
        // a value the slider takes; past it, the range decides first.
        for more in [false, true] {
            assert_eq!(count_step(fewest + 3, more), Err(Words::NotOnTheSlider));
        }
        assert_eq!(count_step(most + 1, false), Err(Words::NotOnTheSlider));
        assert_eq!(count_step(most + 1, true), Err(Words::EndOfRange));
        assert_eq!(count_step(most + 2, false), Ok(most));
        assert_eq!(count_step(most + 2, true), Err(Words::EndOfRange));
        assert_eq!(count_step(fewest - 2, true), Ok(fewest));
        assert_eq!(count_step(fewest - 2, false), Err(Words::EndOfRange));
        // Never overflowing: checked arithmetic reads the end of the range.
        for more in [false, true] {
            assert_eq!(count_step(most + 4, more), Err(Words::EndOfRange));
            assert_eq!(count_step(i64::MAX, more), Err(Words::EndOfRange));
            assert_eq!(count_step(i64::MIN, more), Err(Words::EndOfRange));
        }
    }

    #[test]
    fn a_side_reads_a_number_exactly_when_sizing_counts_its_design() {
        // Spec 3.3 steps 4-6: `side_words` is sizing's validity rule (`is_valid`) in its order,
        // one rule for every row. Moved with the neighbour lines' tests of gui/guidance.rs: their
        // designs at every even count (the defaults, magnets too short for the model, the end
        // effect near its edge) and the keyway through the hub, broken and intact.
        let range = FreeVariable::MagnetsPerRing.range();
        let counts = (range.min as i64..=range.max as i64).step_by(2);
        let mut intact = keyway_through();
        intact.coupling.keyway_depth_mm = 0.0;
        let at = |inputs: &DesignInputs, npole: i64| {
            let design = FreeVariable::MagnetsPerRing.apply(inputs, npole as f64);
            let results = compute_all(&design);
            (design, results)
        };
        let mut found = Vec::new();
        for base in [
            DesignInputs::default(),
            short_magnets(),
            end_effect_at_a_step(),
            keyway_through(),
            intact.clone(),
        ] {
            for npole in counts.clone() {
                let (design, results) = at(&base, npole);
                let read = side_words(&design, &results, true);
                assert_eq!(
                    read.is_none(),
                    is_valid(&design, &results),
                    "{npole}: {read:?}"
                );
                found.extend(read);
            }
        }
        // Not vacuous: every word of steps 4-6 is met.
        for word in [
            Words::DoesNotFit,
            Words::KeywayBreaksThrough,
            Words::NoPrediction,
        ] {
            assert!(found.contains(&word), "{word:?}");
        }
        // Decision D-1: with the keyway left out, a keyway through the hub changes no side's
        // words; each count reads as with the keyway intact (the wall past it does not depend on
        // the count).
        for npole in counts {
            let (broken, broken_results) = at(&keyway_through(), npole);
            let (whole, whole_results) = at(&intact, npole);
            assert_eq!(
                side_words(&broken, &broken_results, false),
                side_words(&whole, &whole_results, true),
                "{npole}"
            );
        }
        // The order: the fit before the keyway, the keyway before the prediction.
        let (crowded, crowded_results) = at(&keyway_through(), 12);
        assert_eq!(
            side_words(&crowded, &crowded_results, true),
            Some(Words::DoesNotFit)
        );
        let mut short_and_keyed = short_magnets();
        short_and_keyed.coupling.bore_mm = keyway_through().coupling.bore_mm;
        short_and_keyed.coupling.keyway_depth_mm = keyway_through().coupling.keyway_depth_mm;
        let results = compute_all(&short_and_keyed);
        assert_eq!(
            side_words(&short_and_keyed, &results, true),
            Some(Words::KeywayBreaksThrough)
        );
        assert_eq!(
            side_words(&short_and_keyed, &results, false),
            Some(Words::NoPrediction)
        );
    }
}
```

(b) `magcoupling-rs/src/gui/mod.rs`:

Old (`magcoupling-rs/src/gui/mod.rs`, from line 32):
```rust
#[cfg(test)]
pub(crate) mod test_support;
pub mod trace;
```

New:
```rust
#[cfg(test)]
pub(crate) mod test_support;
pub mod torque_card;
pub mod trace;
```

(c) `magcoupling-rs/src/gui/test_support.rs`: the gui-smoke design's inputs (`smoke_inputs`, critique A-24) and the table, after `with_variation`:

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 316):
```rust
pub(crate) fn with_variation(variation: f64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.variation = variation;
    inputs
}
```

New:
```rust
pub(crate) fn with_variation(variation: f64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.variation = variation;
    inputs
}

/// The gui-smoke share link's design inputs (slice 3 spec 2.1, "(b)"), before Auto on the
/// magnet length solves them: the defaults with a 1.5 mm face gap.
pub(crate) fn smoke_inputs() -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.face_gap_mm = 1.5;
    inputs
}

// The torque card's one table of designs (slice 3 spec section 8, code review 4 SF-4): here,
// beside the designs it reads, so any test module reads the one table.

/// One design of the torque card's table ([`card_designs`]) as the panel shows it: its key,
/// the design shown (an Auto design solved as the panel solves it) and the Size tab's Minimum
/// OD preference.
pub(crate) struct CardDesign {
    pub(crate) key: &'static str,
    pub(crate) inputs: crate::DesignInputs,
    pub(crate) minimum_diameter_mm: Option<f64>,
}

impl CardDesign {
    /// The design's card, at its Minimum OD preference ([`card_at`]).
    pub(crate) fn card(&self) -> crate::gui::torque_card::TorqueCard {
        card_at(&self.inputs, self.minimum_diameter_mm)
    }

    /// The side of the design's `control` row, `more` its `+` side ([`side_at`]).
    pub(crate) fn side(
        &self,
        control: crate::gui::torque_card::Control,
        more: bool,
    ) -> crate::gui::torque_card::Side {
        side_at(&self.inputs, self.minimum_diameter_mm, control, more)
    }
}

/// The card of `inputs` as the panel asks for it: with its own results, the Minimum OD
/// preference `minimum` and the shaft-clamp section's shipped pause.
pub(crate) fn card_at(
    inputs: &crate::DesignInputs,
    minimum: Option<f64>,
) -> crate::gui::torque_card::TorqueCard {
    let results = crate::compute_all(inputs);
    crate::gui::torque_card::card(inputs, &results, minimum, crate::engine::clamps::PAUSED)
}

/// The side of `inputs`' card ([`card_at`], at the Minimum OD preference `minimum`) on
/// `control`'s row, `more` its `+` side.
pub(crate) fn side_at(
    inputs: &crate::DesignInputs,
    minimum: Option<f64>,
    control: crate::gui::torque_card::Control,
    more: bool,
) -> crate::gui::torque_card::Side {
    let card = card_at(inputs, minimum);
    let row = card
        .rows()
        .iter()
        .find(|row| row.control == control)
        .unwrap_or_else(|| panic!("a {control:?} row: {card:?}"));
    if more {
        row.more.clone()
    } else {
        row.fewer.clone()
    }
}

/// The design the panel shows from `inputs` with Auto on `variable` at the Minimum OD
/// preference `minimum`: the solver's value, or its best where the requirement is out of
/// reach (`SizingRunner`), set through `gui::sizing::sized`, over the range the panel
/// searches (`gui::sizing::sizing_range`, the panel's own rule).
fn auto_shown(
    inputs: &crate::DesignInputs,
    variable: crate::engine::sizing::FreeVariable,
    minimum: Option<f64>,
) -> crate::DesignInputs {
    use crate::gui::sizing::{SizingMode, SizingRunner, SizingState, sized, sizing_range};
    let range = sizing_range(inputs, variable, minimum);
    let state = SizingState {
        mode: SizingMode::TorqueToMagnets,
        variable,
    };
    let mut runner = SizingRunner::default();
    runner.solve_now(inputs, &state, range);
    let value = runner
        .value(variable)
        .unwrap_or_else(|| panic!("Auto on {variable:?} shows a value"));
    sized(variable, inputs, value)
}

/// The torque card's designs: those of the slice 3 spec's section 2.1 and its Edges, each with
/// its Minimum OD preference (an Auto design solved as the panel solves it).
/// The one table the card's tests read (code review 4 SF-4), by key or all of it, so a design
/// added once reaches every test. Each key is the spec's, without its Markdown, with
/// ": <what differs>" where a row holds several designs.
pub(crate) fn card_designs() -> Vec<CardDesign> {
    use crate::DesignInputs;
    use crate::engine::size_controls::{
        dimensions, dimensions_with_minimum_diameter, with_ring_radius,
    };
    use crate::engine::sizing::FreeVariable::{AxialLength, MagnetsPerRing, RingRadius};
    use crate::gui::guidance::FACE_GAP;
    use crate::gui::size_controls::{SliderMapping, slider_minimum};
    let a = DesignInputs::default;
    let e = || with_poles(8);
    let e2 = || {
        let mut inputs = with_poles(8);
        inputs.metal.required_min_Nm = 1.5;
        inputs
    };
    let size = |inputs: &DesignInputs| dimensions(inputs).expect("a design with size controls");
    let lowest = |inputs: &DesignInputs| size(inputs).minimum_diameter_range.min;
    // A design one slider mapping away (the Size tab's own, `SliderMapping`): the Outside
    // diameter at the default Minimum OD, the Overall length, the gap.
    let at_diameter = |inputs: &DesignInputs, mm: f64| {
        SliderMapping::Diameter {
            minimum: slider_minimum(&size(inputs), None),
        }
        .design(inputs, mm)
        .expect("a diameter on the slider")
    };
    let at_length = |inputs: &DesignInputs, mm: f64| {
        SliderMapping::Length
            .design(inputs, mm)
            .expect("a length on the slider")
    };
    let at_gap = |mm: f64| {
        SliderMapping::Input { path: FACE_GAP }
            .design(&a(), mm)
            .expect("a gap on the slider")
    };
    let diameter_top = |inputs: &DesignInputs| {
        let minimum = slider_minimum(&size(inputs), None);
        dimensions_with_minimum_diameter(inputs, minimum)
            .expect("a diameter range")
            .diameter_range
            .max
    };
    let lengths = size(&a()).length_range;
    let plain = |key, inputs| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: None,
    };
    let at_minimum = |key, inputs, minimum: f64| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: Some(minimum),
    };
    let solved = |key, inputs: DesignInputs, variable, minimum: Option<f64>| CardDesign {
        key,
        inputs: auto_shown(&inputs, variable, minimum),
        minimum_diameter_mm: minimum,
    };
    let hot = {
        let mut inputs = a();
        hot_day(&mut inputs);
        inputs
    };
    vec![
        plain("(a)", a()),
        solved("(b)", smoke_inputs(), AxialLength, None),
        plain("(c)", no_back_iron()),
        plain("(e)", e()),
        plain("(e2)", e2()),
        plain("mixed", mixed_lengths()),
        plain("reverse", reverse_lengths()),
        solved("(a)+Auto length", a(), AxialLength, None),
        solved("(a)+Auto count", a(), MagnetsPerRing, None),
        solved("(a)+Auto ring", a(), RingRadius, None),
        solved("(e2)+Auto length", e2(), AxialLength, None),
        solved("(e2)+Auto count", e2(), MagnetsPerRing, None),
        solved("(e2)+Auto ring", e2(), RingRadius, None),
        solved(
            "(e2)+Auto ring, low Min OD",
            e2(),
            RingRadius,
            Some(lowest(&e2())),
        ),
        plain("short magnets", short_magnets()),
        plain("end effect at a step", end_effect_at_a_step()),
        plain("odd count", odd_count()),
        at_minimum("(e), low Min OD", e(), lowest(&e())),
        at_minimum("(a), Min OD 42.8", a(), 42.8),
        at_minimum("(a), Min OD 45", a(), 45.0),
        at_minimum("(a), Min OD 45: 43.3 mm", a(), 43.3),
        plain(
            "shown does not fit",
            with_ring_radius(&a(), 9.5).expect("a ring the diameter slider reaches"),
        ),
        plain("keyway through", keyway_through()),
        plain("gap at 0.3", at_gap(0.3)),
        plain("npole 4", with_poles(4)),
        plain("hot day", hot),
        plain("only the joint fails", only_the_joint_fails()),
        plain("passing", passing_design()),
        plain("short cavity", short_cavity()),
        plain("temperature edge", temperature_edge()),
        plain("npole 40", forty_narrow_magnets()),
        plain("slider tops: gap 5.0", at_gap(5.0)),
        plain(
            "slider tops: Outside diameter 80",
            at_diameter(&a(), diameter_top(&a())),
        ),
        plain(
            "slider tops: Overall length 69.9",
            at_length(&a(), lengths.max),
        ),
        plain(
            "slider tops: Overall length 21.1",
            at_length(&a(), lengths.min),
        ),
        plain("steps to an end: (a) at length 68.9", at_length(&a(), 68.9)),
        plain("steps to an end: (a) at length 22.1", at_length(&a(), 22.1)),
        plain("steps to an end: (e) at length 68.9", at_length(&e(), 68.9)),
        plain("steps to an end: (e) at length 22.1", at_length(&e(), 22.1)),
        plain("steps to an end: (a) at gap 4.9", at_gap(4.9)),
        plain("steps to an end: (a) at gap 0.4", at_gap(0.4)),
        plain(
            "steps to an end: (a) at Outside diameter 79",
            at_diameter(&a(), diameter_top(&a()) - 1.0),
        ),
        plain(
            "steps to an end: (e) at Outside diameter 79",
            at_diameter(&e(), diameter_top(&e()) - 1.0),
        ),
        at_minimum(
            "steps to an end: (e), Min OD 41.8",
            e(),
            size(&e()).diameter_mm - 1.0,
        ),
        plain("variation 1.0", with_variation(1.0)),
        plain("variation 1.0: 1.5", with_variation(1.5)),
        plain("zero torque", zero_torque()),
        plain(
            "(c), f_cal_original -0.95",
            no_back_iron_negative_calibration(),
        ),
        plain("count overflow: i64::MAX", with_poles(i64::MAX)),
        plain("count overflow: i64::MIN", with_poles(i64::MIN)),
    ]
}
```

(d) `magcoupling-rs/src/gui/guidance.rs`, the test module. The neighbour rule's test leaves; its pins are now `count_step_keeps_even_counts_inside_the_slider_and_never_overflows` above, one side at a time. Its odd-count assertion joins the slider-ends neighbour test before it, which reads the neighbour lines until Task 5 moves them:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 9205):
```rust
        }
    }

    #[test]
    fn neighbours_are_even_counts_inside_the_slider_and_never_overflow() {
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        assert_eq!(neighbour_counts(fewest + 4), [fewest + 2, fewest + 6]);
        // A count from a design file may be odd or past the slider.
        assert!(
            neighbour_counts(fewest + 3).is_empty(),
            "odd: no even neighbour"
        );
        assert_eq!(neighbour_counts(most + 2), [most]);
        assert_eq!(neighbour_counts(fewest - 2), [fewest]);
        assert!(neighbour_counts(most + 4).is_empty());
        assert!(neighbour_counts(i64::MAX).is_empty());
        assert!(neighbour_counts(i64::MIN).is_empty());
        // An odd count shows the ring-size line and no neighbour.
        let odd = with_magnets(&DesignInputs::default(), fewest + 5);
        let lines = consequences_of(&odd, None).magnet_lines(odd.coupling.npole, None);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert!(!lines.iter().any(|l| l.starts_with("At ")), "{lines:?}");
    }
```

New:
```rust
        }
        // An odd count (a design file's) shows the ring-size line and no neighbour: the slider
        // takes neither count next to it (`torque_card::count_step`, whose own test pins the
        // counts it takes).
        let odd = with_magnets(&DesignInputs::default(), fewest + 5);
        let lines = consequences_of(&odd, None).magnet_lines(odd.coupling.npole, None);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert!(!lines.iter().any(|l| l.starts_with("At ")), "{lines:?}");
    }
```

(e) `magcoupling-rs/src/gui/guidance.rs`: its `FACE_GAP`, visible to the crate, is the one constant of the gap's input path, which the card and its table read (critique A-12):

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 161):
```rust
const FACE_GAP: &str = "metal.face_gap_mm";
```

New:
```rust
pub(crate) const FACE_GAP: &str = "metal.face_gap_mm";
```

(f) `magcoupling-rs/src/gui/sizing.rs` and `magcoupling-rs/src/gui/panel.rs`: the range Auto searches, one rule for the panel and the card's table (critique A-13). `gui/sizing.rs` imports `ring_radius_range`,

Old (`magcoupling-rs/src/gui/sizing.rs`, from line 22):
```rust
use crate::engine::size_controls::{dimensions, with_ring_radius};
use crate::engine::sizing::{FreeVariable, SizingError, SizingOutcome, solve_in};
```

New:
```rust
use crate::engine::size_controls::{dimensions, ring_radius_range, with_ring_radius};
use crate::engine::sizing::{FreeVariable, SizingError, SizingOutcome, solve_in};
```

and gains `sizing_range` after `sized`:

Old (`magcoupling-rs/src/gui/sizing.rs`, from line 112):
```rust
        _ => variable.apply(inputs, value),
    }
}
```

New:
```rust
        _ => variable.apply(inputs, value),
    }
}

/// The range Auto searches for `variable` (spec guidance 3.5; slice 3: one rule for the panel
/// and the torque card's test table): the ring size over the ring radii of the Size tab's
/// diameter range, its Minimum OD `minimum_diameter_mm` included ([`ring_radius_range`]), or
/// over its slider where the size controls define no range (an odd magnet count); every other
/// variable over its slider.
pub(crate) fn sizing_range(
    inputs: &DesignInputs,
    variable: FreeVariable,
    minimum_diameter_mm: Option<f64>,
) -> SliderRange {
    match variable {
        FreeVariable::RingRadius => ring_radius_range(inputs, minimum_diameter_mm)
            .unwrap_or_else(|| FreeVariable::RingRadius.range()),
        other => other.range(),
    }
}
```

The panel's imports lose `ring_radius_range`, and `FreeVariable` with it: the panel's production code reads `FreeVariable` again only from Task 5's `auto()` (clippy's `unused import` otherwise), so it moves to the panel's test module below until then (plan deviation PD-59):

Old (`magcoupling-rs/src/gui/panel.rs`, from line 31):
```rust
use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};
use crate::engine::size_controls::ring_radius_range;
use crate::engine::sizing::{FreeVariable, SizingOutcome};
```

New:
```rust
use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};
use crate::engine::sizing::SizingOutcome;
```

The panel's `gui::sizing` import gains `sizing_range`:

Old (`magcoupling-rs/src/gui/panel.rs`, from line 61):
```rust
use crate::gui::sizing::{
    REQUIREMENT_INPUT, SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, held_by_auto,
    with_own_value,
};
```

New:
```rust
use crate::gui::sizing::{
    REQUIREMENT_INPUT, SOLVED_PREFIX, SOLVING, SizingMode, SizingRunner, SizingState, held_by_auto,
    sizing_range, with_own_value,
};
```

The panel's `sizing_range` method calls it; its callers are unchanged:

Old (`magcoupling-rs/src/gui/panel.rs`, from line 418):
```rust
    /// The range Auto's solve searches (spec guidance 3.5): ring size over the ring radii of the
    /// Size tab's diameter range, its Minimum OD included
    /// ([`crate::engine::size_controls::ring_radius_range`]), or over its slider when the size
    /// controls define no range (an odd magnet count); every other free variable over its slider.
    /// Read from the inputs, never the design shown, so a solve cannot move its own range.
    fn sizing_range(&self) -> SliderRange {
        match self.sizing.variable {
            FreeVariable::RingRadius => ring_radius_range(&self.inputs, self.minimum_diameter_mm)
                .unwrap_or_else(|| FreeVariable::RingRadius.range()),
            variable => variable.range(),
        }
    }
```

New:
```rust
    /// The range Auto's solve searches ([`crate::gui::sizing::sizing_range`], spec guidance 3.5),
    /// read from the inputs, never the design shown, so a solve cannot move its own range.
    fn sizing_range(&self) -> SliderRange {
        sizing_range(&self.inputs, self.sizing.variable, self.minimum_diameter_mm)
    }
```

The panel's test module imports `FreeVariable` itself until Task 5:

Old (`magcoupling-rs/src/gui/panel.rs`, from line 1592):
```rust
    use crate::engine::sizing::SizingOutcome;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
```

New:
```rust
    use crate::engine::sizing::{FreeVariable, SizingOutcome};
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
```

- [ ] **Step 2: Run them and see the build fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib --no-run`

Expected: the library itself builds, with ten unused-import warnings (the module's imports wait for Step 3), and the lib test target does not compile:
```text
error[E0433]: failed to resolve: use of undeclared type `Words`      (32 of these)
error[E0433]: failed to resolve: use of undeclared type `Control`      (15 of these)
error[E0425]: cannot find function `count_step` in this scope      (14 of these)
error[E0433]: failed to resolve: use of undeclared type `TorqueCard`      (11 of these)
error[E0433]: failed to resolve: use of undeclared type `Side`      (10 of these)
error[E0425]: cannot find function `side_words` in this scope      (6 of these)
error[E0412]: cannot find type `Side` in this scope      (3 of these)
error[E0412]: cannot find type `Control` in this scope      (3 of these)
error[E0432]: unresolved import `Control`      (2 of these)
error[E0412]: cannot find type `TorqueCard` in module `crate::gui::torque_card`      (2 of these)
error[E0412]: cannot find type `Control` in module `crate::gui::torque_card`      (2 of these)
error[E0412]: cannot find type `Side` in module `crate::gui::torque_card`      (2 of these)
error[E0425]: cannot find value `STEP_SLACK_MM` in this scope      (2 of these)
error[E0425]: cannot find function `card` in this scope      (2 of these)
error[E0425]: cannot find function `card` in module `crate::gui::torque_card`
error[E0412]: cannot find type `Row` in this scope
error[E0412]: cannot find type `Words` in this scope
error[E0412]: cannot find type `Change` in this scope
error: could not compile `magcoupling-rs` (lib test) due to 110 previous errors; 5 warnings emitted
```

- [ ] **Step 3: The card's types and steps (`gui/torque_card.rs`)**

The code between the imports and the test module: the slack, the types, `card`, `side_words`, `count_step`, each control's slider, and steps 2 to 8 of the side's tests:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 23):
```rust
use crate::{DesignInputs, DesignResults, compute_all};

#[cfg(test)]
mod tests {
```

New:
```rust
use crate::{DesignInputs, DesignResults, compute_all};

/// The one slack of the card's range test and its jump rule [mm] (spec 3.3 steps 2 and 7): far
/// above the float noise of the sliders' mappings (a step that lands 1.42e-14 mm past a range's
/// end, 3.6e-15 mm in `resize_length`'s subtraction), far below the sliders' 0.01 and 0.1 mm
/// steps and any real change.
pub(crate) const STEP_SLACK_MM: f64 = 1e-6;

/// The Magnets per ring step (owner 3A): two magnets, so an even count stays even.
const COUNT_STEP: i64 = 2;

/// A Size-tab control the card steps (owner 2A), in the controls' order (6A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    OutsideDiameter,
    OverallLength,
    FaceGap,
    MagnetsPerRing,
}

impl Control {
    /// The four controls in the Size tab's order: the card's rows keep it (owner 6A).
    pub const ALL: [Control; 4] = [
        Control::OutsideDiameter,
        Control::OverallLength,
        Control::FaceGap,
        Control::MagnetsPerRing,
    ];

    /// The step of decision 3A: 1.0 mm, 1.0 mm, 0.1 mm, 2 magnets.
    pub const fn step(self) -> f64 {
        match self {
            Control::OutsideDiameter | Control::OverallLength => 1.0,
            Control::FaceGap => 0.1,
            Control::MagnetsPerRing => COUNT_STEP as f64,
        }
    }

    /// The label drawn above the control's slider: `size_controls::DIAMETER` and `LENGTH` (no
    /// input carries those two names), and the gap's and the count's input labels (owner 11A).
    pub fn label(self) -> &'static str {
        match self {
            Control::OutsideDiameter => size_controls::DIAMETER,
            Control::OverallLength => size_controls::LENGTH,
            Control::FaceGap => label_for(FACE_GAP),
            Control::MagnetsPerRing => label_for(FreeVariable::MagnetsPerRing.path()),
        }
    }

    /// The Auto variable whose slider is this control: the inverse of
    /// [`size_controls::variable_slider`]; `None` for the gap, which Auto never sets.
    pub fn held_by(self) -> Option<FreeVariable> {
        FreeVariable::ALL
            .into_iter()
            .find(|&variable| size_controls::variable_slider(variable) == self.label())
    }

    /// The step on the `+` side (`more`) or the `-` side, with its sign.
    fn signed_step(self, more: bool) -> f64 {
        if more { self.step() } else { -self.step() }
    }
}

/// Why a side shows words, not a number (owner 9A, 16A; spec T-4), in the order of the tests
/// that find them (spec 3.3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Words {
    /// The step leaves its slider's range (the diameter's at the user's Minimum OD, owner 9A),
    /// or the slider's mapping refuses it.
    EndOfRange,
    /// An odd magnet count from a design file, which the slider does not take.
    NotOnTheSlider,
    /// The stepped design's blocks do not fit.
    DoesNotFit,
    /// The keyway leaves no hub wall past it (not on the count row: decision D-1).
    KeywayBreaksThrough,
    /// The model has no prediction for the stepped design.
    NoPrediction,
    /// The overlapping magnet length does not change by the step (the Overall length row).
    Jump,
}

/// A dashboard check whose level differs on the stepped design (owner 8A).
#[derive(Clone, Debug, PartialEq)]
pub struct Flip {
    /// The check's guide label.
    pub check: &'static str,
    /// Its level on the stepped design.
    pub to: Level,
}

/// The change of a failing check's own quantity (owner 8A; spec 3.6). `unit` is the result's
/// own (`result_info(path).meta.unit`: mm, µm, MPa, °C or N·m for the quoted paths), drawn
/// through `with_unit` as `result_text` does; no unit is typed.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginChange {
    /// The check's guide label.
    pub check: &'static str,
    /// The quantity's name, where the label alone does not name it.
    pub name: Option<&'static str>,
    pub change: f64,
    pub unit: &'static str,
    /// The decimals the change starts from (section 5's rule).
    pub decimals: usize,
}

/// A side with a number.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub struct Change {
    /// The stepped design's hot-low torque (`metal.torque_hot_low_Nm`).
    pub hot_low_Nm: f64,
    /// That hot-low less the design shown's.
    pub change_Nm: f64,
    /// `change_Nm` over the design shown's hot-low.
    pub fraction: f64,
    /// Whether the stepped design takes another calibration factor (`model.f_cal`, the factor
    /// `select_calibration_factor` picks; owner 16A).
    pub calibration_changed: bool,
    /// The Outside diameter the slider would show, less the design shown's ([`dimensions`]);
    /// `None` where either design has no size controls.
    pub diameter_change_mm: Option<f64>,
    /// The Overall length likewise.
    pub length_change_mm: Option<f64>,
    /// The counted checks whose level differs, in `CHECK_GUIDES` order.
    pub flips: Vec<Flip>,
    /// The failing checks' quantities that move and show (spec 3.6).
    pub margins: Vec<MarginChange>,
}

/// One side of a row: words, or a number with what the step costs.
#[derive(Clone, Debug, PartialEq)]
pub enum Side {
    Words(Words),
    Change(Box<Change>),
}

/// One control's row: its `-` side, then its `+` side.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub control: Control,
    pub fewer: Side,
    pub more: Side,
}

/// The card of a design. `ShownNoHotLow`: the model predicts (the readouts show both torques),
/// but hot-low, the percent's base, is not above zero (a design file's variation of 1 or more;
/// physics review 2 MF-1), or the pull-out is not above zero (a magnet Br or calibration
/// coefficient of 0 or less; physics review 3 MF-2).
#[derive(Clone, Debug, PartialEq)]
pub enum TorqueCard {
    ShownDoesNotFit,
    ShownNoPrediction,
    ShownNoHotLow,
    Rows(Vec<Row>),
}

impl TorqueCard {
    /// The rows: empty for the three one-line forms.
    pub fn rows(&self) -> &[Row] {
        match self {
            TorqueCard::Rows(rows) => rows,
            TorqueCard::ShownDoesNotFit
            | TorqueCard::ShownNoPrediction
            | TorqueCard::ShownNoHotLow => &[],
        }
    }
}

/// The card of `inputs`, the design shown, whose results are `results` (`compute_all(inputs)`),
/// with the Size tab's Minimum OD preference (`None`: unset), as the panel holds it, and the
/// shaft-clamp pause (`clamps_paused`, the panel's field): the flips and margins run over
/// `counted_guides(clamps_paused)`.
pub fn card(
    inputs: &DesignInputs,
    results: &DesignResults,
    minimum_diameter_mm: Option<f64>,
    #[expect(
        unused_variables,
        reason = "the flips and margins read it (slice 3 Task 4, which removes this)"
    )]
    clamps_paused: bool,
) -> TorqueCard {
    // Test 0 (spec 3.3): the design shown, by the readouts' own rule.
    if !blocks_fit(&inputs.coupling, &results.model) {
        return TorqueCard::ShownDoesNotFit;
    }
    if !predicts_both(results) {
        return TorqueCard::ShownNoPrediction;
    }
    if !(results.model.pullout_Nm > 0.0 && results.metal.torque_hot_low_Nm > 0.0) {
        return TorqueCard::ShownNoHotLow;
    }
    let shown = Shown {
        inputs,
        results,
        size: dimensions(inputs),
    };
    TorqueCard::Rows(
        Control::ALL
            .into_iter()
            .filter_map(|control| {
                // Test 1: a size row only where the Size tab draws its slider (T-5).
                let slider = Slider::of(control, &shown, minimum_diameter_mm)?;
                Some(Row {
                    control,
                    fewer: shown.side(control, &slider, false),
                    more: shown.side(control, &slider, true),
                })
            })
            .collect(),
    )
}

/// Why a stepped design reads words, not a number, by sizing's own validity rule
/// ([`crate::engine::sizing::is_valid`]) in its order (spec 3.3 steps 4-6): its blocks do not
/// fit; with `keyway`, the keyway leaves no hub wall past it (`model.hub_wall_past_key_mm`, NaN
/// included); the model has no prediction, by the readouts' own rule ([`predicts`]: the end
/// effect out of range, or a torque that is not finite). `None` where the design counts.
/// `results` is `compute_all(design)`. The Magnets per ring row passes `keyway` false (decision
/// D-1): the wall past the keyway does not depend on the count, so the design shown says so.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation is the point: a NaN wall fails it
pub(crate) fn side_words(
    design: &DesignInputs,
    results: &DesignResults,
    keyway: bool,
) -> Option<Words> {
    if !blocks_fit(&design.coupling, &results.model) {
        Some(Words::DoesNotFit)
    } else if keyway && !(results.model.hub_wall_past_key_mm > 0.0) {
        Some(Words::KeywayBreaksThrough)
    } else if !predicts_both(results) {
        Some(Words::NoPrediction)
    } else {
        None
    }
}

/// One step of the Magnets per ring slider from `npole`, `more` the `+` side (owner 3A: two
/// magnets): the count, or why the slider does not take it. The slider takes the even counts
/// of its range (one block per pole): a step past the range, or one that overflows (a design
/// file's count), reads [`Words::EndOfRange`]; an odd count inside the range, which a design
/// file can hold, [`Words::NotOnTheSlider`].
pub(crate) fn count_step(npole: i64, more: bool) -> Result<i64, Words> {
    let range = FreeVariable::MagnetsPerRing.range();
    let stepped = if more {
        npole.checked_add(COUNT_STEP)
    } else {
        npole.checked_sub(COUNT_STEP)
    };
    match stepped {
        Some(count) if (range.min..=range.max).contains(&(count as f64)) => {
            if count % 2 == 0 {
                Ok(count)
            } else {
                Err(Words::NotOnTheSlider)
            }
        }
        _ => Err(Words::EndOfRange),
    }
}

/// Whether the readouts show both torques of `results` as numbers ([`predicts`]: the end
/// effect in range and the torque finite). The card's "no prediction" rule for the design shown
/// and for every side (spec 3.3 tests 0 and 6), so the readouts, the design shown and each side
/// read one rule.
fn predicts_both(results: &DesignResults) -> bool {
    predicts(results, results.model.pullout_Nm)
        && predicts(results, results.metal.torque_hot_low_Nm)
}

/// `value` on a slider's `range` (spec 3.3 step 2): `None` when it is not finite or lies more
/// than [`STEP_SLACK_MM`] past an end; else the value, taken as that end within the slack past
/// it. The mappings' own range checks are exact, and a step from a value on the slider's grid
/// can land a few ulps past the end the slider reaches exactly.
fn on_slider(value: f64, range: SliderRange) -> Option<f64> {
    (value.is_finite() && value >= range.min - STEP_SLACK_MM && value <= range.max + STEP_SLACK_MM)
        .then(|| value.clamp(range.min, range.max))
}

/// The Magnet face gap slider's range: its input's, as the Size tab draws it.
fn face_gap_range() -> SliderRange {
    InputCatalogue::get()
        .entry(FACE_GAP)
        .and_then(|entry| entry.meta.range)
        .expect("the face gap has a slider")
}

/// The design shown, its results and its size controls ([`dimensions`]: `None` where the Size
/// tab draws no size slider): what each side is compared with.
struct Shown<'a> {
    inputs: &'a DesignInputs,
    results: &'a DesignResults,
    size: Option<SizeControls>,
}

/// How a control's own slider maps a value to a design, with the range it takes (spec 3.3): the
/// range here, the mapping the controls' own ([`SliderMapping`]).
enum Slider {
    /// The Outside diameter slider at the user's Minimum OD: its range there, and its mapping at
    /// `minimum` ([`slider_minimum`]).
    Diameter { size: SizeControls, minimum: f64 },
    /// The Overall length slider: its mapping over the size controls' length range.
    Length { size: SizeControls },
    /// The Magnet face gap slider: its input set inside its input's range.
    FaceGap { range: SliderRange },
    /// The Magnets per ring slider: [`count_step`], then the count set as the slider sets it,
    /// the ring radius kept.
    MagnetsPerRing,
}

impl Slider {
    /// The slider of `control` on the design shown, at the Minimum OD preference `minimum`;
    /// `None` where the Size tab draws no such slider (T-5): a size row without size controls
    /// (an odd or out-of-range count, or a geometry they cannot define), or the diameter row
    /// without a diameter range at that Minimum OD.
    fn of(control: Control, shown: &Shown, minimum: Option<f64>) -> Option<Slider> {
        match control {
            Control::OutsideDiameter => {
                let minimum = slider_minimum(&shown.size?, minimum);
                let size = dimensions_with_minimum_diameter(shown.inputs, minimum)?;
                Some(Slider::Diameter { size, minimum })
            }
            Control::OverallLength => Some(Slider::Length { size: shown.size? }),
            Control::FaceGap => Some(Slider::FaceGap {
                range: face_gap_range(),
            }),
            Control::MagnetsPerRing => Some(Slider::MagnetsPerRing),
        }
    }

    /// The design one step of `control` from `inputs` makes, `more` its `+` side; else why the
    /// slider does not take that step (spec 3.3 steps 2 and 3): a value more than
    /// [`STEP_SLACK_MM`] past an end of its range, one the mapping refuses, or a count
    /// [`count_step`] does not take.
    fn step(
        &self,
        control: Control,
        inputs: &DesignInputs,
        more: bool,
    ) -> Result<DesignInputs, Words> {
        let step = control.signed_step(more);
        let on = |value: f64, range: SliderRange| on_slider(value, range).ok_or(Words::EndOfRange);
        match *self {
            // Each side through the slider's own mapping (`SliderMapping`, the controls' too).
            Slider::Diameter { size, minimum } => {
                let requested = on(size.diameter_mm + step, size.diameter_range)?;
                SliderMapping::Diameter { minimum }
                    .design(inputs, requested)
                    .map_err(|_| Words::EndOfRange)
            }
            Slider::Length { size } => {
                let requested = on(size.overall_length_mm + step, size.length_range)?;
                SliderMapping::Length
                    .design(inputs, requested)
                    .map_err(|_| Words::EndOfRange)
            }
            Slider::FaceGap { range } => {
                let gap = on(inputs.metal.face_gap_mm + step, range)?;
                SliderMapping::Input { path: FACE_GAP }
                    .design(inputs, gap)
                    .map_err(|_| Words::EndOfRange)
            }
            Slider::MagnetsPerRing => {
                let npole = count_step(inputs.coupling.npole, more)?;
                SliderMapping::Input {
                    path: FreeVariable::MagnetsPerRing.path(),
                }
                .design(inputs, npole as f64)
                .map_err(|_| Words::EndOfRange)
            }
        }
    }
}

impl Shown<'_> {
    /// The side of `control`'s row one step of its `slider` away, `more` the `+` side, by the
    /// order of tests (spec 3.3 steps 2-8).
    fn side(&self, control: Control, slider: &Slider, more: bool) -> Side {
        let design = match slider.step(control, self.inputs, more) {
            Ok(design) => design,
            Err(words) => return Side::Words(words),
        };
        let results = compute_all(&design);
        // Steps 4-6: sizing's validity rule, the keyway left out on the count row (D-1).
        let keyway = control != Control::MagnetsPerRing;
        if let Some(words) = side_words(&design, &results, keyway) {
            return Side::Words(words);
        }
        // Step 7: a length step that is not a step of the overlapping magnet length.
        if control == Control::OverallLength && !self.moves_by_the_step(&results, more) {
            return Side::Words(Words::Jump);
        }
        Side::Change(Box::new(self.change(&design, &results)))
    }

    /// Whether the overlapping magnet length (`model.active_length_mm`) of the stepped design,
    /// whose results are `stepped`, moves by exactly the Overall length step, within
    /// [`STEP_SLACK_MM`] (spec 3.3 step 7: owner 9A, widened by question O-1). A change by
    /// anything else, more or less or the other way, is a jump, and so is a length that is not
    /// a number.
    fn moves_by_the_step(&self, stepped: &DesignResults, more: bool) -> bool {
        let moved = stepped.model.active_length_mm - self.results.model.active_length_mm;
        (moved - Control::OverallLength.signed_step(more)).abs() <= STEP_SLACK_MM
    }

    /// The number side of the stepped `design`, whose results are `stepped` (spec 3.3 step 8):
    /// its hot-low against the design shown's, whatever its sign, and what the step costs.
    #[allow(non_snake_case)] // unit suffix
    fn change(&self, design: &DesignInputs, stepped: &DesignResults) -> Change {
        let shown_hot_low_Nm = self.results.metal.torque_hot_low_Nm;
        let hot_low_Nm = stepped.metal.torque_hot_low_Nm;
        let change_Nm = hot_low_Nm - shown_hot_low_Nm;
        let (diameter_change_mm, length_change_mm) = match (self.size, dimensions(design)) {
            (Some(shown), Some(after)) => (
                Some(after.diameter_mm - shown.diameter_mm),
                Some(after.overall_length_mm - shown.overall_length_mm),
            ),
            _ => (None, None),
        };
        Change {
            hot_low_Nm,
            change_Nm,
            fraction: change_Nm / shown_hot_low_Nm,
            calibration_changed: stepped.model.f_cal != self.results.model.f_cal,
            diameter_change_mm,
            length_change_mm,
            flips: Vec::new(),
            margins: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
```

- [ ] **Step 4: The neighbour lines call the moved rule (`gui/guidance.rs`)**

The import:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 71):
```rust
use crate::gui::size_controls;
use crate::{DesignInputs, DesignResults, compute_all};
```

New:
```rust
use crate::gui::size_controls;
use crate::gui::torque_card::{Words, count_step, side_words};
use crate::{DesignInputs, DesignResults, compute_all};
```

`neighbours` steps the count with `count_step` and reads the design with `side_words` (the keyway left out, D-1). `neighbour_counts` and `neighbour_outcome` leave (moved), and `prediction` stays:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 3053):
```rust
/// The neighbours of `inputs` (`results` are its own), each one `compute_all` of the design
/// with only the magnet count changed.
fn neighbours(inputs: &DesignInputs, results: &DesignResults) -> Vec<Neighbour> {
    let baseline = prediction(results);
    neighbour_counts(inputs.coupling.npole)
        .into_iter()
        .map(|npole| {
            let design = FreeVariable::MagnetsPerRing.apply(inputs, npole as f64);
            let outcome = neighbour_outcome(&design, &compute_all(&design), baseline);
            Neighbour { npole, outcome }
        })
        .collect()
}

/// Two fewer and two more than `npole`, ascending, keeping the even counts inside the magnet
/// slider (one block per pole; the slider steps by 2). A count from a design file may be odd
/// or past the slider, and none overflows.
fn neighbour_counts(npole: i64) -> Vec<i64> {
    let range = FreeVariable::MagnetsPerRing.range();
    [npole.checked_sub(2), npole.checked_add(2)]
        .into_iter()
        .flatten()
        .filter(|&n| n % 2 == 0 && (range.min..=range.max).contains(&(n as f64)))
        .collect()
}

/// The pull-out of the design shown when the model predicts one: the end effect in range and
/// a positive, finite torque. Without it a neighbour's change has nothing to compare with.
fn prediction(results: &DesignResults) -> Option<f64> {
    let pullout = results.model.pullout_Nm;
    (end_effect_in_range(results.model.f_end) && pullout.is_finite() && pullout > 0.0)
        .then_some(pullout)
}

/// What a neighbour design does (`results` are its own; `baseline` is [`prediction`] of the
/// design shown). The conditions of [`crate::engine::sizing::is_valid`] but the keyway's: the
/// hub wall past the keyway (apothem - bondline - bore / 2 - keyway depth) does not depend on
/// the magnet count, so a neighbour fails it exactly when the design shown does, the Size tab
/// already says so, and the torques still compare.
fn neighbour_outcome(
    design: &DesignInputs,
    results: &DesignResults,
    baseline: Option<f64>,
) -> NeighbourOutcome {
    if !blocks_fit(&design.coupling, &results.model) {
        return NeighbourOutcome::DoesNotFit;
    }
    let pullout = results.model.pullout_Nm;
    if !end_effect_in_range(results.model.f_end)
        || !pullout.is_finite()
        || !results.metal.torque_hot_low_Nm.is_finite()
    {
        return NeighbourOutcome::NoPrediction;
    }
    NeighbourOutcome::Torque {
        pullout_Nm: pullout,
        change: baseline.map_or(f64::NAN, |base| pullout / base - 1.0),
        cup_od_mm: results.model.cup_od_mm,
    }
}
```

New:
```rust
/// The neighbours of `inputs` (`results` are its own), each one `compute_all` of the design
/// with only the magnet count changed: two fewer and two more where the magnet slider takes
/// the count ([`count_step`], the torque card's count row's step), read by the card's validity
/// rule ([`side_words`]) but the keyway's (decision D-1: the hub wall past the keyway does not
/// depend on the magnet count, so a neighbour fails it exactly when the design shown does, the
/// Size tab already says so, and the torques still compare).
fn neighbours(inputs: &DesignInputs, results: &DesignResults) -> Vec<Neighbour> {
    let baseline = prediction(results);
    [false, true]
        .into_iter()
        .filter_map(|more| count_step(inputs.coupling.npole, more).ok())
        .map(|npole| {
            let design = FreeVariable::MagnetsPerRing.apply(inputs, npole as f64);
            let results = compute_all(&design);
            let outcome = match side_words(&design, &results, false) {
                None => NeighbourOutcome::Torque {
                    pullout_Nm: results.model.pullout_Nm,
                    change: baseline.map_or(f64::NAN, |base| results.model.pullout_Nm / base - 1.0),
                    cup_od_mm: results.model.cup_od_mm,
                },
                Some(Words::DoesNotFit) => NeighbourOutcome::DoesNotFit,
                // The keyway left out, the only other words are the prediction's.
                Some(_) => NeighbourOutcome::NoPrediction,
            };
            Neighbour { npole, outcome }
        })
        .collect()
}

/// The pull-out of the design shown when the model predicts one: the end effect in range and
/// a positive, finite torque. Without it a neighbour's change has nothing to compare with.
fn prediction(results: &DesignResults) -> Option<f64> {
    let pullout = results.model.pullout_Nm;
    (end_effect_in_range(results.model.f_end) && pullout.is_finite() && pullout > 0.0)
        .then_some(pullout)
}
```

`blocks_fit` and `end_effect_in_range` stay imported, because the guides and `prediction` read them.

- [ ] **Step 5: Run the card's tests and the neighbour lines' tests and see them pass; plant fourteen faults**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::torque_card`

Expected: `test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 940 filtered out` (about 0.4 s).

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- each_neighbour_is_the_design_with_only_the_magnet_count_changed the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown a_keyway_through_the_hub_changes_no_neighbour_s_reading with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented with_magnets_too_short_for_the_model_every_neighbour_reads_words`

Expected: `test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 942 filtered out`. The neighbour lines read the moved rule, and no reading changes.

Each planted fault below is one edit of `magcoupling-rs/src/gui/torque_card.rs`; each Old is unique. Run it with `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::torque_card`, then undo it by the reverse edit (the file is new, so `git checkout` cannot restore it). The module's nine tests pass again after each. Together they show that each rule of spec 3.3 has a test that bites, including the "red first" of the spec's tests 3 and 4:

F1. The exact range test (no slack).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
    (value.is_finite() && value >= range.min - STEP_SLACK_MM && value <= range.max + STEP_SLACK_MM)
        .then(|| value.clamp(range.min, range.max))
```

New:
```rust
    (value.is_finite() && value >= range.min && value <= range.max).then_some(value)
```

Expected: 2 fail (`7 passed; 2 failed`): `the_words_follow_the_order_of_the_tests` panics `words, not a number: EndOfRange` (a step that lands within the slack past a range's end), and `each_number_is_the_difference_of_two_compute_all_runs` reads `left: 261`, `right: 263`.

F2. The jump rule as 9A wrote it.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        (moved - Control::OverallLength.signed_step(more)).abs() <= STEP_SLACK_MM
```

New:
```rust
        let _ = more;
        moved.abs() <= Control::OverallLength.step() + STEP_SLACK_MM
```

Expected: 2 fail: `the_words_follow_the_order_of_the_tests` panics `a number, not words: Change { hot_low_Nm: 2.144621922273862, change_Nm: -0.14058292852202037, ...` (the 11.0 mm cavity's `+1 mm` side, whose overlap moves -0.70 mm), and test 1 reads `left: 264`, `right: 263`.

F3. The jump rule unsigned.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        (moved - Control::OverallLength.signed_step(more)).abs() <= STEP_SLACK_MM
```

New:
```rust
        let _ = more;
        (moved.abs() - Control::OverallLength.step()).abs() <= STEP_SLACK_MM
```

Expected: 1 fails (`8 passed; 1 failed`): `the_words_follow_the_order_of_the_tests` panics `a number, not words: Change { hot_low_Nm: 2.084372095764424, change_Nm: -0.20083275503145837, ...` (the 10.7 mm cavity's `+1 mm` side, whose overlap moves the step's size the wrong way).

F4. Revision 2's shown rule (a pull-out not above zero reads no prediction).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
    if !predicts_both(results) {
        return TorqueCard::ShownNoPrediction;
```

New:
```rust
    if !(predicts_both(results) && results.model.pullout_Nm > 0.0) {
        return TorqueCard::ShownNoPrediction;
```

Expected: 1 fails: `the_design_shown_decides_first` at `zero torque`: `left: ShownNoPrediction`, `right: ShownNoHotLow`.

F5. A shown rule on hot-low alone.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
    if !(results.model.pullout_Nm > 0.0 && results.metal.torque_hot_low_Nm > 0.0) {
```

New:
```rust
    if !(results.metal.torque_hot_low_Nm > 0.0) {
```

Expected: 1 fails: `the_design_shown_decides_first` at `f_cal_original -0.95, variation 1.5`: `left: Rows([...])`, `right: ShownNoHotLow`.

F6. The keyway on the count row (no D-1).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        let keyway = control != Control::MagnetsPerRing;
```

New:
```rust
        let keyway = true;
```

Expected: 2 fail: `the_words_follow_the_order_of_the_tests` panics `words, not a number: KeywayBreaksThrough` (the keyway through the hub, count `-2`), and test 1 reads `left: 262`, `right: 263`.

F7. The count's step unchecked.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        npole.checked_add(COUNT_STEP)
    } else {
        npole.checked_sub(COUNT_STEP)
```

New:
```rust
        Some(npole + COUNT_STEP)
    } else {
        Some(npole - COUNT_STEP)
```

Expected: 1 fails: `count_step_keeps_even_counts_inside_the_slider_and_never_overflows` panics `attempt to subtract with overflow`.

F8. The fraction as the prototype wrote it (stepped over shown, less one).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
            fraction: change_Nm / shown_hot_low_Nm,
```

New:
```rust
            fraction: hot_low_Nm / shown_hot_low_Nm - 1.0,
```

Expected: 1 fails: test 1 at `(a) OverallLength true` (the fraction's bits differ).

F9. No calibration note.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
            calibration_changed: stepped.model.f_cal != self.results.model.f_cal,
```

New:
```rust
            calibration_changed: false,
```

Expected: 2 fail: test 1 at `(c) MagnetsPerRing false` (`left: false`, `right: true`), and test 3 at `assertion failed: fewer.calibration_changed`.

F10. The diameter through resize_diameter (no Minimum OD).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                SliderMapping::Diameter { minimum }
                    .design(inputs, requested)
                    .map_err(|_| Words::EndOfRange)
```

New:
```rust
                let _ = minimum;
                crate::engine::size_controls::resize_diameter(inputs, requested)
                    .map_err(|_| Words::EndOfRange)
```

Expected: 2 fail: `the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum` at `(e2)+Auto ring, low Min OD false` (`left: true`, `right: false`), and test 1 reads `left: 262`, `right: 263`.

F11. The diameter's range at the default Minimum OD.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                let minimum = slider_minimum(&shown.size?, minimum);
```

New:
```rust
                let minimum = slider_minimum(&shown.size?, None);
```

Expected: 4 fail (`5 passed; 4 failed`): test 3 at `(a), Min OD 42.8 false` (`left: DoesNotFit`, `right: EndOfRange`), test 6 (`left: Words(DoesNotFit)`), test 1 at `(a), Min OD 45 OutsideDiameter true`, and `a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range` (the Minimum OD at its top: `left: DoesNotFit`, `right: EndOfRange`).

F12. The odd count read as the end of the range.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                Err(Words::NotOnTheSlider)
```

New:
```rust
                Err(Words::EndOfRange)
```

Expected: 2 fail: `count_step_keeps_even_counts_inside_the_slider_and_never_overflows` (`left: Err(EndOfRange)`, `right: Err(NotOnTheSlider)`) and test 3 (`left: EndOfRange`, `right: NotOnTheSlider`).

F13. The rows out of the controls' order.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        Control::ALL
            .into_iter()
            .filter_map(|control| {
```

New:
```rust
        Control::ALL
            .into_iter()
            .rev()
            .filter_map(|control| {
```

Expected: 2 fail: `the_size_rows_follow_the_size_controls` at `(a)` (`left: [MagnetsPerRing, FaceGap, OverallLength, OutsideDiameter]`), and test 6 (`left: MagnetsPerRing`).

F14. The gap stepped from the value shown held to its slider (critique A-20: a design shown past an end).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                let gap = on(inputs.metal.face_gap_mm + step, range)?;
```

New:
```rust
                let gap = on(inputs.metal.face_gap_mm.clamp(range.min, range.max) + step, range)?;
```

Expected: 1 fails (`8 passed; 1 failed`): `a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range` panics `a number, not words: Change { hot_low_Nm: 0.8019875288278033, change_Nm: 0.11777746364743968, ...` (the 5.5 mm gap's `-0.1 mm` side, stepped from the slider's top).

- [ ] **Step 6: Docs: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`**

The module and the table ship with their docs. Task 4 extends these entries with what it adds. When the neighbour lines leave, Task 5 rewrites "which call them" (its Step 7 (j2) and (r2)).

(a) README, the Layout table's `src/gui/` row (`:155`): the module's entry, after the entry for `size_controls.rs`.

Old (`magcoupling-rs/README.md`, from line 155):
```text
one mapping for the controls and the torque card's steps), `sizing.rs` (`SizingMode`,
```

New:
```text
one mapping for the controls and the torque card's steps), `torque_card.rs` (the torque card, slice 3, BL-073; pure, no egui: `card` -> `TorqueCard` (`ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow`, or `Rows`, one `Row` per Size control (`Control`), read by `rows`), each control one fixed step each way through its slider's own mapping (`size_controls::SliderMapping`, the one the controls apply), in the order of tests of its spec's section 3.3: a side's `Words`, or its `Change` (the stepped design's hot-low, its change in N·m and as a fraction of the design shown's, the Outside diameter and Overall length changes, and whether the calibration factor changes); `side_words` (sizing's validity rule, every row's, the keyway left out on the count row: decision D-1) and `count_step` (two magnets each way: checked, even, inside the slider), both moved from the N-2 and N+2 lines, which call them; `STEP_SLACK_MM`, the one slack of the range test and the jump rule), `sizing.rs` (`SizingMode`,
```

(a2) The same row, `sizing.rs`'s entry: `sizing_range` (critique A-13).

Old (`magcoupling-rs/README.md`, from line 155):
```text
`SizingRunner`: debounced inverse sizing over a search range)
```

New:
```text
`SizingRunner`: debounced inverse sizing over a search range; `sizing_range`, the range Auto searches, one rule for the panel and the torque card's test table, slice 3)
```

(b) The same row, the entry for `test_support.rs`:

Old (`magcoupling-rs/README.md`, from line 155):
```text
headless egui helpers and the shared test designs, the torque card's among them; tests only
```

New:
```text
headless egui helpers and the shared test designs, the torque card's among them with their one table, `card_designs()`; tests only
```

(c) README, the Tests table: a new `src/gui/torque_card.rs` row, before the row that starts with `src/gui/explorer.rs` (that is, after the `src/gui/guidance.rs` row).

Old (`magcoupling-rs/README.md`, from line 698):
```text
| `src/gui/explorer.rs`, `src/engine/explain/notes.rs`,
```

New:
```text
| `src/gui/torque_card.rs` (feature `gui`) | The torque card (slice 3, BL-073): its four controls are the Size tab's, in the tab's order, each with its step, its slider's label and the Auto variable that sets it (`each_control_is_a_size_tab_slider_with_its_step`); over one table of designs, `card_designs()` in `gui/test_support.rs` (every design of its spec's section 2.1 with its Minimum OD preference, an Auto design solved as the panel solves it): every number is the difference of two `compute_all` runs of a design the test rebuilds through the slider's own mapping, bit for bit, the size changes against `dimensions` (none exactly where either design has none) and the calibration note against `model.f_cal` (`each_number_is_the_difference_of_two_compute_all_runs`); each side's words in the order of the tests, a step that lands within `STEP_SLACK_MM` of a slider's end a number, the jump words on every length side whose overlapping magnet length changes by anything other than the step, a count side whose calibration switch takes its torque to or below zero a number (`the_words_follow_the_order_of_the_tests`); the design shown decides first, by the readouts' own rule (`the_design_shown_decides_first`); the size rows only where the Size tab draws them (`the_size_rows_follow_the_size_controls`); the diameter row steps as the slider does at the user's Minimum OD (`the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum`); a design shown past a slider's end (a design file's gap of 5.5 mm, an axial override of 60 mm, the Minimum OD at its slider's top) reads the end of its range on both sides of that row, and a side back on the slider is the slider's own design (`a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range`); moved with the N-2 and N+2 lines' rule from `gui/guidance.rs`: `count_step` keeps even counts inside 4 to 40 and never overflows (`count_step_keeps_even_counts_inside_the_slider_and_never_overflows`), and a side reads a number exactly when sizing counts its design, the keyway left out on the count row (`a_side_reads_a_number_exactly_when_sizing_counts_its_design`). |
| `src/gui/explorer.rs`, `src/engine/explain/notes.rs`,
```

(d) `docs/ai/02-system.yaml`, `key_files` (`:82`):

Old (`docs/ai/02-system.yaml`, from line 82):
```text
src/gui/guidance.rs, src/gui/size_controls.rs,
```

New:
```text
src/gui/guidance.rs, src/gui/torque_card.rs, src/gui/size_controls.rs,
```

(e) `docs/ai/03-structure.yaml`, the `gui` line (`:68`). It is a double-quoted scalar, so no `"` may be added. The module's clause, after the clause for `size_controls.rs`:

Old (`docs/ai/03-structure.yaml`, from line 68):
```text
a hint line under each of its five controls), sizing.rs (SizingMode,
```

New:
```text
a hint line under each of its five controls), torque_card.rs (slice 3, BL-073; pure, no egui: Control (OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing; ALL, step, label, held_by), Words (EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump), Flip, MarginChange, Change, Side, Row, TorqueCard (ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows; rows()); card(inputs, results, minimum_diameter_mm, clamps_paused): each Size control one step each way through its slider's own mapping (size_controls::SliderMapping, the one the controls apply), in the order of tests of its spec's section 3.3 (the design shown by the readouts' own rule; on the slider within STEP_SLACK_MM, the one slack of the range test and the jump rule; the mapping; side_words, the validity rule of every row, with D-1 for the count; the jump: words whenever the overlapping magnet length changes by anything other than the step); count_step (the count's step: checked, even, inside 4 to 40); side_words and count_step moved from the N-2 and N+2 lines, which call them), sizing.rs (SizingMode,
```

(f) The same line, the clause for `test_support.rs`: the table, after Task 2's designs:

Old (`docs/ai/03-structure.yaml`, from line 68):
```text
zero_torque (an unmagnetized inner ring) and with_variation(variation), the lookups
```

New:
```text
zero_torque (an unmagnetized inner ring), with_variation(variation) and smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap), and their one table for the torque card's tests, card_designs() (CardDesign {key, inputs, minimum_diameter_mm} with card() and side(), card_at, side_at: every design of spec 2.1 and its Edges at its Minimum OD preference, an Auto design solved as the panel solves it), the lookups
```

(f2) The same line, `sizing.rs`'s clause: `sizing_range` (critique A-13).

Old (`docs/ai/03-structure.yaml`, from line 68):
```text
SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), DEBOUNCE_S 0.25)
```

New:
```text
SizingRunner (update, status, shown, is_current, solve_now; the search range is part of its SolveKey), sizing_range (slice 3: the range Auto searches for a variable at the Minimum OD, one rule for the panel and the torque card's test table), DEBOUNCE_S 0.25)
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`.

`docs/ai/02-system.yaml` does not parse at 69fa505 (pre-existing, near line 15; Task 1 says so), so check the edited line alone:

Run: `python -c "import yaml; t=open('docs/ai/02-system.yaml', encoding='utf-8').read().split(chr(10)); l=[x for x in t if 'src/gui/torque_card.rs' in x]; assert len(l) == 1; print(type(yaml.safe_load(l[0].strip())).__name__)"`

Expected: `dict`.

- [ ] **Step 7: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary `ok`, with Task 2's counts (the module is behind the `gui` feature):

| binary | result |
|---|---|
| lib | 234 passed, 1 ignored |
| assumptions | 13 passed |
| deviations | 54 passed |
| differential | 19 passed |
| explain | 63 passed, 1 ignored |
| gearbox | 13 passed |
| grades | 12 passed |
| material_library | 4 passed |
| material_links | 14 passed |
| parity | 4 passed |
| python_schema | 7 passed |
| robustness | 13 passed |
| schema | 11 passed |
| shaft_joint | 40 passed |
| size_controls | 38 passed |
| sizing | 45 passed |
| static_data | 8 passed |
| doc-tests | 1 passed, 1 ignored |

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `943 passed; 0 failed; 6 ignored` (935 after Task 2, plus nine new tests, less the one moved out of `gui/guidance.rs`); the two app binaries run 0 tests; `explain` 64 passed, 1 ignored; every other binary as above.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`, then the same without `--features app`.

Expected: `Finished`, no warning, both times.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/02-system.yaml
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/gui/guidance.rs
 M magcoupling-rs/src/gui/mod.rs
 M magcoupling-rs/src/gui/panel.rs
 M magcoupling-rs/src/gui/sizing.rs
 M magcoupling-rs/src/gui/test_support.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
?? magcoupling-rs/src/gui/torque_card.rs
```

- [ ] **Step 8: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T3.txt` with the Write tool:
```text
feat(magcoupler): the torque card's module: its types and each control's step (BL-073 T3)

gui/torque_card.rs (pure, no egui; slice 3 spec 3.1-3.3). card() steps each Size
control one fixed step each way through the control's own slider mapping (owner
3A; size_controls::SliderMapping, the controls' own: Outside diameter +/-1 mm at
the user's Minimum OD through slider_minimum, Overall length +/-1 mm, the face
gap +/-0.1 mm, the count +/-2 through count_step, set as its slider sets it), in
the order of tests of spec 3.3: the design shown by the readouts' own rule
(ShownDoesNotFit; ShownNoPrediction where predicts fails; ShownNoHotLow for a
pull-out or hot-low not above zero), the size rows only with size controls
(T-5), the slider's range within STEP_SLACK_MM (1e-6 mm, the one slack of the
range test and the jump rule), the mapping, side_words (sizing's validity rule;
decision D-1 on the count row), and on the Overall length row the widened jump
rule (owner question O-1: words whenever the overlapping magnet length changes by
anything other than the step). A number side holds the stepped hot-low, its
change and fraction, the size changes and whether the calibration factor changes
(owner 16A), whatever its sign. The flips, the margins and the card's texts come
with the next commit; until then the clamp pause is an expected-unused parameter.

side_words and count_step move from the N-2 and N+2 lines (neighbour_outcome and
neighbour_counts: moved, not copied), which call them until they leave the Size
tab. The neighbour rule's test moves to count_step's; its odd-count assertion
joins the slider-ends neighbour test.

One of each shared piece: guidance's FACE_GAP, now visible to the crate, is the
card's and the table's; gui::sizing::sizing_range is the one rule for the range
Auto searches, which the panel calls and the table's Auto designs read; and
test_support::smoke_inputs is the gui-smoke design's inputs.

Tests (spec 8.1 tests 1, 3, 4, 5 and 6, red first) over one table of the spec's
section 2 designs, card_designs(), in gui/test_support.rs beside the designs it
reads, so any test module reads the one table; with count_step's and side_words'
own, the controls' steps and labels against the Size tab's, and a design shown
past a slider's end (both sides of that row at the end of their range). The
README, docs/ai/02-system.yaml and 03-structure.yaml name the module and the
table.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/torque_card.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/src/gui/mod.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/sizing.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T3.txt
git status --short
```

Expected: one commit of 9 files. `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 3 moves** (each confirmed by a run in the plan-time prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui::guidance::tests::neighbours_are_even_counts_inside_the_slider_and_never_overflow` | -> `gui::torque_card::tests::count_step_keeps_even_counts_inside_the_slider_and_never_overflows`. Its pins move one side at a time (`Ok(count)`, `Err(EndOfRange)`, `Err(NotOnTheSlider)`), with two more: a count at either end of the slider, and an odd count past the range, which reads the range first | `neighbour_counts` moved into `count_step` (spec T-2, 6.3) |
| (its last assertion: an odd count draws the ring-size line and no neighbour line) | -> appended to `gui::guidance::tests::at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown` | it reads the neighbour lines, which ship until Task 5 |
| the other seven neighbour tests of spec 6.3: `each_neighbour_is_the_design_with_only_the_magnet_count_changed`, `the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not`, `at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown`, `a_keyway_through_the_hub_changes_no_neighbour_s_reading`, `with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it`, `a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented`, `with_magnets_too_short_for_the_model_every_neighbour_reads_words` | unchanged here; they stay for Task 5, which moves them into the card's test module as the count row's tests (spec 6.3) | they guard the neighbour lines, which now read `count_step` and `side_words`, until Task 5 removes the lines |

---

### Task 4: The torque card's module, part 2: what a step costs, the card's texts and its lines (`gui/torque_card.rs`)

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec sections 3.4 (the Auto marking, drawn outside the cached card), 3.6 (what a step costs), 4.1 (the strings), 4.2 (the reused strings), 4.4 (the traps) and section 5's T-14 (a row's two N·m at one precision), with tests 2, 2a, 7, 7a, 8, 9a, 15, 16, 17 and 18 of section 8. Tests 3 and 4 gain their drawn lines. In `gui/torque_card.rs`:
- What a number side costs (3.6). `Change.flips`: the counted checks whose level differs on the stepped design (`counted_guides(clamps_paused)` and `level`, in the guides' order). `Change.margins`: of each check that fails on both designs (Bad and Bad), each `check_terms` term that fails on either design and quotes a quantity (`Quoted`), where its change shows. `shows` means finite, with a digit that is not zero at `CHANGE_EXTRA_DECIMALS` past the quantity's own decimals. The change is in its result's own unit (`result_info(path).meta.unit`). "Either" catches a step that puts a second term of a failing check over its limit, which no flip shows: at (b), the diameter's `+1 mm` side and the Rotating diameter. Never the hot-low or the temperature check: Task 2's `check_terms` quotes nothing for either (T-13).
- The strings of 4.1, each a named `pub const` (drafts; Task 7's G-3 review sets them): `CARD_HEADING`, `CAPTION`, `AUTO_HOLDS_LENGTH`, `AUTO_HOLDS_RING`, `AUTO_HOLDS_COUNT`, `END_OF_RANGE`, `NOT_ON_THE_SLIDER`, `SIDE_DOES_NOT_FIT`, `SIDE_NO_PREDICTION`, `KEYWAY_BREAKS_THROUGH`, `JUMP`, `CALIBRATION_CHANGE`, `SHOWN_DOES_NOT_FIT`, `SHOWN_NO_PREDICTION`, `SHOWN_NO_HOT_LOW`, `FLIP_FAILS`, `FLIP_PASSES`, `FLIP_CAUTION`. `WALL_NEEDED` and `COLD_HIGH` are Task 2's, in `gui/guidance.rs` (plan deviation PD-7); the card defines no copies.
- The templates:
  - `Words::text`;
  - `Flip::text`: "{label}: {word}";
  - `MarginChange::text`: "{label} {change} {unit}", or "{label}: {name} {change} {unit}";
  - `Change::costs`: the calibration note, then the size changes (never the row's own dimension, and only where one shows), the flips and the margins;
  - `Row::line`: "{label}: {fewer}; {more}", where a side is "{step} -> {N·m} N·m ({percent})" followed by its costs.
  
  A row's two N·m share `shared_change_decimals(&[...], TORQUE_DECIMALS)` (T-14, Task 1's rule) and are drawn through `signed_fixed`. Every other change is drawn through `signed_change` at its own decimals, and the percent through `signed_percent`.
- `TorqueCard::lines(auto)`. The three one-line forms draw their line alone (T-12). The rows form draws `CAPTION`, the Auto line while Auto is on, then each row; the row that Auto holds (`Control::held_by`) reads `held_line(control)` = "{label}: {SET_BY_AUTO}" (owner 5A). The card does not depend on the Auto state, so the cache keeps one card for every Auto state (spec 3.4).
- The neighbour lines' two words move into the card as `SIDE_DOES_NOT_FIT` and `SIDE_NO_PREDICTION`. `NEIGHBOUR_DOES_NOT_FIT` and `NEIGHBOUR_NO_PREDICTION` leave `gui/guidance.rs` (moved, not copied), and its neighbour lines read the card's constants until Task 5 removes them. The Size tab's two exploration lines (`gui/panel.rs`) are built from `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` after one private `EXPLORATION_ONLY` = "Exploration only: " (4.2), with the words unchanged. `smaller_exploration_shows_magnet_fit_and_keyway_breakthrough` builds its expected texts from the same constants.

Facts the implementer meets on the way:
- `check_terms` (Task 2) and `shared_change_decimals` (Task 1) get their production caller here. Delete their `#[cfg_attr(not(test), expect(dead_code, ...))]` attributes in this commit, or the build reports the expectation unfulfilled (plan deviation PD-9). The `#[expect(unused_variables)]` on `card`'s `clamps_paused` goes too, as `Shown` now holds the pause.
- `gui/guidance.rs` opens two things to the crate. `result_number` becomes `pub(crate)`: the card reads each quoted quantity with it, the one reader of a result as a number. Its eight check paths (`HOT_MIN` ... `SPACE_CLAIM`) become `pub(crate)`, so the card's tests can name checks; the alternative, retyping eight paths in the tests, is the repetition the owner flags (plan deviation PD-22). Task 2's test 1a typed six of them as local copies while they were private; it now imports them, and the copies go.
- Test 2a pins (c)'s percent at the side's own value, `signed_percent(-0.41052)`. Spec 2.3's -41.05 % reads "-41.0 %" through `signed_percent`, while the side is -41.0516 % and reads "-41.1 %" (plan deviation PD-25).
- No design reachable from the inputs has Temperature limits red: `verdict_level` (`gui/dashboard.rs`, the temperature verdict) maps "OK" to Good and "CHECK" to Caution only. So T-13's "even when red on both designs" has no design to show it. Test 7 asserts instead, at every design of the table, that the temperature check's terms quote nothing and that its level is never Bad (plan deviation PD-19).
- Test 18 triples every number of a copy of the card (the spec says "moves each") and compares each design's lines with the copy's: the same skeleton, and every drawn number different. Doubling was not enough. It left a drawn number unchanged ("+0.01" from 0.0072 to 0.0144). A gap step's Running clearance change is the step itself at every design, so a second design cannot move it. A torque that changes in proportion keeps its percent (plan deviation PD-18).
- `card()` takes about 1 ms per design in release (median of five means of 200: (a) 0.98 ms, (b) 0.96 ms, (c) 1.00 ms, (e) 2.05 ms), the budget Task 10 checks.
- The table (`card_designs()` in `gui/test_support.rs`) gains each design's Auto state, `CardDesign.auto`, which tests 8 and 18 read.
- The table gains T-14's one-flip design and R-13's boss plateau (plan review 2.2; `test_support::boss_plateau`, pinned by test 1a), so tests 1, 7, 15, 16, 17 and 18 run on the two lines the STOP sheet asks the owner about, and the two tests of those cases (critique A-21) read them by key. Their not-vacuous counts move: `NUMBER_SIDES` 263 -> 277 and test 6's rows 42 -> 44 (Task 3's constants, edited here), test 7's quotes 500 -> 517, test 16's texts 1010 -> 1056, test 18's numbers 1002 -> 1049.
- Two tests beyond the spec's from the plan review: a step to a design without size controls draws no size change (2.1, the rule Task 10's Step 7b must keep), and the card is drawn without a panic, the same card twice, in plain words with its glyphs, at 47 values a design file can carry, at both back irons and at the Minimum OD's top (1.1; no new rule: ruling C-7).
- The card's N·m start from Task 1's `TORQUE_DECIMALS` (`gui::format`, critique A-14); a size change starts from the card's own `SIZE_DECIMALS = 1`.

Line numbers are after Task 3; trust the code. Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`. Every code block is rustfmt's layout. In the plan-time prototype, replaying this task's blocks in the order given on Task 3's files reproduces this task's files byte for byte, and both suites, both clippy runs and fmt are then green.

**Files:**
- Modify: `magcoupling-rs/src/gui/torque_card.rs`:
  - the module doc and imports (`TORQUE_DECIMALS` from `gui::format`); `SIZE_DECIMALS` and the strings;
  - `Control::unit` and `Control::step_text`; `Words::text`; `Flip::text`; `MarginChange::text`; `Change::costs`;
  - `Row::torque_decimals`, `Row::line`, `Row::side_text`; `TorqueCard::lines`; `held_line` and `auto_line`;
  - `card` and `Shown` with the pause; `shows`; the flips and margins in `Shown::change`, and `Shown::quoted_changes`.
- Modify: `magcoupling-rs/src/gui/guidance.rs` (the import; the check paths `pub(crate)`; the attribute on `check_terms`; `result_number` `pub(crate)`; the two `NEIGHBOUR_*` words leave; `Neighbour::line`), `magcoupling-rs/src/gui/format.rs` (the attribute on `shared_change_decimals`), `magcoupling-rs/src/gui/panel.rs` (the import, `EXPLORATION_ONLY`, the two exploration lines)
- Modify (tests): `magcoupling-rs/src/gui/test_support.rs` (`CardDesign.auto`, `card_designs()`; `boss_plateau`; the table's `T-14 one flip` and `R-13 boss plateau`; test 1a imports the check paths and pins the plateau), `magcoupling-rs/src/gui/panel.rs` (`smaller_exploration_shows_magnet_fit_and_keyway_breakthrough`)
- Modify (docs): `magcoupling-rs/README.md` (the Layout table's `src/gui/` row, the `torque_card.rs` entry; the Tests table's `src/gui/torque_card.rs` row), `docs/ai/03-structure.yaml` (the `gui` line: the clauses for `torque_card.rs` and `card_designs()`)
- Test: `magcoupling-rs/src/gui/torque_card.rs`. New: `the_card_reads_the_spec_s_numbers` (test 2); `the_card_s_lines_read_as_drafted` (test 2a); `the_flips_and_the_margins_name_the_checks_that_move` (test 7); `the_card_names_the_clamp_only_while_unpaused` (test 7a); `the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s` (test 8); `size_changes_and_margins_are_drawn_only_when_they_show` (test 9a); `no_card_number_reads_zero_or_disagrees_with_its_percent` (test 15); `the_card_texts_never_rank_the_controls` (test 16); `every_card_text_has_its_glyphs` (test 17); `the_card_s_strings_are_constants_and_its_numbers_come_from_results` (test 18); `a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required`; `where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift`; `a_step_to_a_design_without_size_controls_draws_no_size_change`; `the_card_never_panics_at_the_values_a_design_file_can_carry`. Extended to the drawn line: `the_words_follow_the_order_of_the_tests` (test 3) and `the_design_shown_decides_first` (test 4).

**Interfaces:**
- Consumes:
  - Task 1's `gui::format::{fixed, reads_not_zero, CHANGE_EXTRA_DECIMALS, TORQUE_DECIMALS, signed_fixed, signed_change, signed_percent, shared_change_decimals, with_unit}`;
  - Task 2's `gui::guidance::{check_terms, CheckTerm, Quoted, WALL_NEEDED, COLD_HIGH, SPACE_CLAIM_AXIS_SIZES}` and `engine::explain::plain::ranking_words_in`;
  - Task 3's module and its table in `gui/test_support.rs`;
  - `gui::guidance::{counted_guides, level, guide}`, `gui::dashboard::result_info`, `gui::size_controls::SET_BY_AUTO`;
  - test-only: `engine::explain::plain::{TORQUE_ORDER_WORDS, mentions_gearbox_protection, plain_text_violations}` and `gui::test_support::{SCREEN, sized_frame, assert_glyphs, clamp_at_its_limit, only_the_clamp_fails}`.
- Produces, in `gui/torque_card.rs`:
  - the `pub const` strings above (`&str`);
  - `impl Words { pub fn text(self) -> &'static str }`;
  - `impl TorqueCard { pub fn lines(&self, auto: Option<FreeVariable>) -> Vec<String> }`;
  - `pub fn held_line(control: Control) -> String`;
  - private: `SIZE_DECIMALS = 1` (the N·m start from `gui::format::TORQUE_DECIMALS`), `Control::unit`, `Control::step_text`, `Flip::text`, `MarginChange::text`, `Change::costs(control) -> Vec<String>`, `Row::torque_decimals`, `Row::line`, `Row::side_text`, `auto_line(variable) -> &'static str`, `shows(change, from) -> bool`, `Shown::quoted_changes`;
  - `card`'s signature is unchanged; a `Change` now carries `flips` and `margins`.
- Produces, in `gui::guidance`: `pub(crate) const HOT_MIN, CLEARANCE, CUP_WALL, TEMPERATURE, CLAMP, SHAFT_JOINT, GEARBOX, SPACE_CLAIM` and `pub(crate) fn result_number(results: &DesignResults, path: &str) -> f64`. Removed: `NEIGHBOUR_DOES_NOT_FIT`, `NEIGHBOUR_NO_PREDICTION`.
- Produces, in `gui::panel`: private `const EXPLORATION_ONLY: &str = "Exploration only: "`.
- Produces, in `gui::test_support`: `CardDesign.auto: Option<FreeVariable>` (`pub(crate)`); `pub(crate) fn boss_plateau() -> crate::DesignInputs`; `card_designs()` holds 52 designs.

- [ ] **Step 1: Write the failing tests**

(a) `magcoupling-rs/src/gui/torque_card.rs`, the test module. The imports: the checks by their paths, the plain-words guards, the glyph helpers, the clamp's designs, and `card_at`:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 458):
```rust
    use super::*;
    use crate::engine::clamps::PAUSED;
    use crate::engine::size_controls::{resize_diameter_with_minimum, resize_length};
    use crate::engine::sizing::is_valid;
    use crate::gui::format::fixed;
    use crate::gui::test_support::{
        CardDesign, card_designs, end_effect_at_a_step, keyway_through, no_back_iron,
        short_magnets, side_at, slider_range,
    };
```

New:
```rust
    use super::*;
    use crate::engine::clamps::PAUSED;
    use crate::engine::explain::plain::{
        TORQUE_ORDER_WORDS, mentions_gearbox_protection, plain_text_violations, ranking_words_in,
    };
    use crate::engine::size_controls::{resize_diameter_with_minimum, resize_length};
    use crate::engine::sizing::is_valid;
    use crate::gui::guidance::{
        CLAMP, CLEARANCE, COLD_HIGH, CUP_WALL, GEARBOX, HOT_MIN, SHAFT_JOINT, SPACE_CLAIM,
        SPACE_CLAIM_AXIS_SIZES, TEMPERATURE, WALL_NEEDED, guide,
    };
    use crate::gui::test_support::{
        CardDesign, SCREEN, assert_glyphs, card_at, card_designs, clamp_at_its_limit,
        end_effect_at_a_step, keyway_through, no_back_iron, only_the_clamp_fails, short_magnets,
        side_at, sized_frame, slider_range,
    };
```

Test 3 draws the side whose calibration switch takes the hot-low to zero (physics review 5 SF-2):

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 790):
```rust
        let more = number(&side_at(&unmeasured, None, MagnetsPerRing, true));
        assert!(more.calibration_changed);
        assert_eq!(more.hot_low_Nm, 0.0, "{more:?}");
```

New:
```rust
        let more = number(&side_at(&unmeasured, None, MagnetsPerRing, true));
        assert!(more.calibration_changed);
        assert_eq!(more.hot_low_Nm, 0.0, "{more:?}");
        // It is drawn as every number side: its N·m with its true sign at the row's decimals, and
        // -100.0 %, the hot-low gone (physics review 5 SF-2).
        assert_eq!(more.fraction, -1.0);
        let card = card_at(&unmeasured, None);
        let row = &card.rows()[3];
        assert!(
            side_texts(&card.lines(None)[4], row)[1].starts_with(&format!(
                "+2 -> {} ({}), {CALIBRATION_CHANGE}",
                torque(more.change_Nm, row.torque_decimals()),
                signed_percent(-1.0)
            )),
            "{:?}",
            card.lines(None)
        );
```

Test 4: each one-line form draws its one line alone, whatever the Auto state (T-12):

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 926):
```rust
            let card = card(&inputs, &results, None, PAUSED);
            assert_eq!(card, shown, "{name}");
            assert!(card.rows().is_empty(), "{name}");
        }
```

New:
```rust
            let card = card(&inputs, &results, None, PAUSED);
            assert_eq!(card, shown, "{name}");
            assert!(card.rows().is_empty(), "{name}");
            // Its one line alone: no caption, no Auto line, whatever the Auto state (T-12).
            let line = match card {
                TorqueCard::ShownDoesNotFit => SHOWN_DOES_NOT_FIT,
                TorqueCard::ShownNoPrediction => SHOWN_NO_PREDICTION,
                TorqueCard::ShownNoHotLow => SHOWN_NO_HOT_LOW,
                TorqueCard::Rows(_) => unreachable!("{name}"),
            };
            for auto in [None].into_iter().chain(FreeVariable::ALL.map(Some)) {
                assert_eq!(card.lines(auto), [line], "{name} {auto:?}");
            }
        }
```

The new tests, after the last test: 2, 2a, 7, 7a, 8 and 9a; then the T-14 one-flip test and R-13's boss-plateau test (critique A-21), the size-controls test (plan review 2.1) and the robustness test (plan review 1.1); then 15, 16, 17 and 18, with their helpers:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 1194):
```rust
        assert_eq!(
            side_words(&short_and_keyed, &results, false),
            Some(Words::NoPrediction)
        );
    }
}
```

New:
```rust
        assert_eq!(
            side_words(&short_and_keyed, &results, false),
            Some(Words::NoPrediction)
        );
    }

    // ------------------------------------------------------------------ what a step costs, drawn

    /// The units a quoted quantity's result carries (spec 3.6): mm, µm, MPa, °C and N·m.
    const QUOTED_UNITS: [&str; 5] = ["mm", "µm", "MPa", "°C", "N·m"];

    /// The label of the guide of `check`.
    fn label(check: &str) -> &'static str {
        guide(check).expect("a guide's check").label
    }

    /// The four rows of `design`'s card, in order.
    fn four_rows(design: &CardDesign) -> [Row; 4] {
        <[Row; 4]>::try_from(design.card().rows().to_vec()).expect("four rows")
    }

    /// A number side as section 2.3's tables print it: the change in N·m to 4 decimals and in
    /// percent to 2, its flips, and each quoted change to 4 decimals.
    #[derive(Debug, PartialEq)]
    struct Pinned {
        torque: String,
        percent: String,
        flips: Vec<Flip>,
        quoted: Vec<(&'static str, Option<&'static str>, String)>,
    }

    impl Pinned {
        fn of(side: &Side) -> Pinned {
            let change = number(side);
            Pinned {
                torque: signed_fixed(change.change_Nm, 4),
                percent: signed_fixed(100.0 * change.fraction, 2),
                flips: change.flips,
                quoted: change
                    .margins
                    .iter()
                    .map(|margin| (margin.check, margin.name, signed_fixed(margin.change, 4)))
                    .collect(),
            }
        }
    }

    /// Section 2.3's number side: the N·m and the percent as its tables print them, the flips
    /// (each check by its path, and its level on the stepped design) and the quoted changes
    /// (check, name, change).
    fn pin(
        torque: &str,
        percent: &str,
        flips: &[(&str, Level)],
        quoted: &[(&str, Option<&'static str>, &str)],
    ) -> Pinned {
        Pinned {
            torque: torque.to_owned(),
            percent: percent.to_owned(),
            flips: flips
                .iter()
                .map(|&(check, to)| Flip {
                    check: label(check),
                    to,
                })
                .collect(),
            quoted: quoted
                .iter()
                .map(|&(check, name, change)| (label(check), name, change.to_owned()))
                .collect(),
        }
    }

    #[test]
    fn the_card_reads_the_spec_s_numbers() {
        // Spec 8.1 test 2: section 2.3's values (C), to its decimals: N·m to 4, % to 2, mm to 4.
        let designs = card_designs();
        let rows = |key| four_rows(find(&designs, key));
        let (wall, cold) = (Some(WALL_NEEDED), Some(COLD_HIGH));
        let rotating = Some(SPACE_CLAIM_AXIS_SIZES[0].0);
        let stack = Some(SPACE_CLAIM_AXIS_SIZES[2].0);
        let does_not_fit = Side::Words(Words::DoesNotFit);

        // (a), the defaults: no size change, no calibration change (the cap sets the Outside
        // diameter at every stepped design).
        let [diameter, length, gap, count] = rows("(a)");
        assert_eq!(diameter.fewer, does_not_fit);
        assert_eq!(
            Pinned::of(&diameter.more),
            pin(
                "+0.0798",
                "+3.49",
                &[(SPACE_CLAIM, Level::Bad)],
                &[
                    (CLEARANCE, None, "+0.0125"),
                    (CUP_WALL, wall, "+0.0651"),
                    (GEARBOX, cold, "+0.0281"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&length.fewer),
            pin("-0.2008", "-8.79", &[], &[(GEARBOX, cold, "-0.0707")])
        );
        assert_eq!(
            Pinned::of(&length.more),
            pin("+0.2008", "+8.79", &[], &[(GEARBOX, cold, "+0.0707")])
        );
        assert_eq!(
            Pinned::of(&gap.fewer),
            pin(
                "+0.0759",
                "+3.32",
                &[],
                &[
                    (CLEARANCE, None, "-0.1000"),
                    (CUP_WALL, wall, "+0.0182"),
                    (GEARBOX, cold, "+0.0267"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&gap.more),
            pin(
                "-0.0733",
                "-3.21",
                &[],
                &[
                    (CLEARANCE, None, "+0.1000"),
                    (CUP_WALL, wall, "-0.0177"),
                    (GEARBOX, cold, "-0.0258"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&count.fewer),
            pin(
                "-0.6121",
                "-26.79",
                &[(GEARBOX, Level::Good)],
                &[(CUP_WALL, wall, "+0.4797")]
            )
        );
        assert_eq!(count.more, does_not_fit);

        // (b), the gui-smoke share link's design with Auto on the magnet length, solved: the
        // held row is computed (drawn as Auto's, test 8); the diameter +1 mm side puts the
        // Rotating diameter over the claim, a second axis of a check that already fails.
        let [diameter, length, gap, count] = rows("(b)");
        assert_eq!(diameter.fewer, does_not_fit);
        assert_eq!(
            Pinned::of(&diameter.more),
            pin(
                "+0.0892",
                "+3.57",
                &[],
                &[
                    (CLEARANCE, None, "+0.0125"),
                    (CUP_WALL, wall, "+0.0643"),
                    (GEARBOX, cold, "+0.0314"),
                    (SPACE_CLAIM, rotating, "+1.0000"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&length.fewer),
            pin(
                "-0.1945",
                "-7.78",
                &[(HOT_MIN, Level::Bad), (SPACE_CLAIM, Level::Good)],
                &[(GEARBOX, cold, "-0.0685")]
            )
        );
        assert_eq!(
            Pinned::of(&length.more),
            pin(
                "+0.1945",
                "+7.78",
                &[],
                &[(GEARBOX, cold, "+0.0685"), (SPACE_CLAIM, stack, "+1.0000")]
            )
        );
        assert_eq!(
            Pinned::of(&gap.fewer),
            pin(
                "+0.0827",
                "+3.31",
                &[],
                &[
                    (CLEARANCE, None, "-0.1000"),
                    (CUP_WALL, wall, "+0.0177"),
                    (GEARBOX, cold, "+0.0291"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&gap.more),
            pin(
                "-0.0798",
                "-3.19",
                &[(HOT_MIN, Level::Bad)],
                &[
                    (CLEARANCE, None, "+0.1000"),
                    (CUP_WALL, wall, "-0.0173"),
                    (GEARBOX, cold, "-0.0281"),
                ]
            )
        );
        assert_eq!(
            Pinned::of(&count.fewer),
            pin(
                "-0.6579",
                "-26.32",
                &[(HOT_MIN, Level::Bad)],
                &[(CUP_WALL, wall, "+0.4753"), (GEARBOX, cold, "-0.2317")]
            )
        );
        assert_eq!(count.more, does_not_fit);

        // (c), no back iron: the count -2 side takes the calibration coefficient, a change of
        // factor its number includes (owner 16A).
        let [.., count] = rows("(c)");
        let fewer = number(&count.fewer);
        assert_eq!(
            [
                signed_fixed(fewer.change_Nm, 4),
                signed_fixed(100.0 * fewer.fraction, 2)
            ],
            ["-0.5923", "-41.05"]
        );
        assert!(fewer.calibration_changed);

        // (e), the owner's case: the diameter row under 1 % both ways, and two magnets fewer
        // grow the Outside diameter.
        let [diameter, _, _, count] = rows("(e)");
        for (side, torque, percent) in [
            (&diameter.fewer, "-0.0063", "-0.37"),
            (&diameter.more, "-0.0046", "-0.28"),
        ] {
            let change = number(side);
            assert_eq!(signed_fixed(change.change_Nm, 4), torque);
            assert_eq!(signed_fixed(100.0 * change.fraction, 2), percent);
        }
        let grown = number(&count.fewer)
            .diameter_change_mm
            .expect("a size change");
        assert_eq!(signed_fixed(grown, 4), "+2.2307");

        // The mixed-length pair jumps both ways; its diameter +1 mm side is a number. The
        // reverse pair moves the active length by the step.
        let [diameter, length, ..] = rows("mixed");
        assert_eq!(
            [length.fewer, length.more],
            [Side::Words(Words::Jump), Side::Words(Words::Jump)]
        );
        assert_eq!(signed_fixed(number(&diameter.more).change_Nm, 4), "+0.0300");
        let [_, length, ..] = rows("reverse");
        for (side, percent) in [(&length.fewer, "-19.89"), (&length.more, "+19.89")] {
            assert_eq!(signed_fixed(100.0 * number(side).fraction, 2), percent);
        }
    }

    /// The N·m of a side as drawn: `x` at `decimals` places with its true sign, and the unit.
    fn torque(x: f64, decimals: usize) -> String {
        with_unit(signed_fixed(x, decimals), "N·m")
    }

    /// A quoted or size change as drawn: `x` at its own decimals from `from`, and `unit`.
    fn quoted(x: f64, from: usize, unit: &str) -> String {
        with_unit(signed_change(x, from), unit)
    }

    /// The unit of the result at `path`, as the card quotes it.
    fn unit_of(path: &str) -> &'static str {
        result_info(path).expect("a result").meta.unit
    }

    #[test]
    fn the_card_s_lines_read_as_drafted() {
        // Spec 8.1 test 2a (code review 2 MF-5): whole lines, built from the constants, the
        // labels, the step texts and section 2.3's numbers through the Size tab's format at the
        // row's decimals (T-14). The drafted blocks of 2.3 are these lines; a G-3 word change
        // flows through.
        let designs = card_designs();
        let lines = |key| find(&designs, key).card().lines(None);
        let [od, length, gap, count] = Control::ALL.map(Control::label);
        let (clearance, cup_wall, gearbox, space) = (
            label(CLEARANCE),
            label(CUP_WALL),
            label(GEARBOX),
            label(SPACE_CLAIM),
        );
        let mm = |x: f64| quoted(x, 2, "mm");
        let cold = |x: f64| quoted(x, 2, "N·m");

        // (a): the order within a side is the number, the size changes, the flips, then the
        // quoted changes in the checks' order.
        assert_eq!(
            lines("(a)"),
            [
                CAPTION.to_owned(),
                format!(
                    "{od}: -1 mm -> {SIDE_DOES_NOT_FIT}; +1 mm -> {} ({}), {space}: {FLIP_FAILS}, \
                     {clearance} {}, {cup_wall}: {WALL_NEEDED} {}, {gearbox}: {COLD_HIGH} {}",
                    torque(0.0798, 2),
                    signed_percent(0.0349),
                    mm(0.0125),
                    mm(0.0651),
                    cold(0.0281)
                ),
                format!(
                    "{length}: -1 mm -> {} ({}), {gearbox}: {COLD_HIGH} {}; +1 mm -> {} ({}), \
                     {gearbox}: {COLD_HIGH} {}",
                    torque(-0.2008, 2),
                    signed_percent(-0.0879),
                    cold(-0.0707),
                    torque(0.2008, 2),
                    signed_percent(0.0879),
                    cold(0.0707)
                ),
                format!(
                    "{gap}: -0.1 mm -> {} ({}), {clearance} {}, {cup_wall}: {WALL_NEEDED} {}, \
                     {gearbox}: {COLD_HIGH} {}; +0.1 mm -> {} ({}), {clearance} {}, \
                     {cup_wall}: {WALL_NEEDED} {}, {gearbox}: {COLD_HIGH} {}",
                    torque(0.0759, 2),
                    signed_percent(0.0332),
                    mm(-0.1),
                    mm(0.0182),
                    cold(0.0267),
                    torque(-0.0733, 2),
                    signed_percent(-0.0321),
                    mm(0.1),
                    mm(-0.0177),
                    cold(-0.0258)
                ),
                format!(
                    "{count}: -2 -> {} ({}), {gearbox}: {FLIP_PASSES}, {cup_wall}: {WALL_NEEDED} \
                     {}; +2 -> {SIDE_DOES_NOT_FIT}",
                    torque(-0.6121, 2),
                    signed_percent(-0.2679),
                    mm(0.4797)
                ),
            ]
        );

        // (e): the diameter row's two N·m at the row's 3 decimals (T-14: each on its own would
        // read "-0.01" and "-0.005"), and two magnets fewer grow the Outside diameter.
        let e = lines("(e)");
        assert_eq!(
            e[1],
            format!(
                "{od}: -1 mm -> {} ({}), {clearance} {}, {cup_wall}: {WALL_NEEDED} {}; \
                 +1 mm -> {} ({}), {space}: {FLIP_FAILS}, {clearance} {}, {cup_wall}: \
                 {WALL_NEEDED} {}",
                torque(-0.0063, 3),
                signed_percent(-0.0037),
                mm(-0.013),
                mm(-0.079),
                torque(-0.0046, 3),
                signed_percent(-0.0028),
                mm(0.0122),
                mm(0.079)
            )
        );
        assert_eq!(
            e[4],
            format!(
                "{count}: -2 -> {} ({}), {od} {}, {space}: {FLIP_FAILS}, {cup_wall}: \
                 {WALL_NEEDED} {}; +2 -> {} ({}), {gearbox}: {FLIP_FAILS}, {cup_wall}: \
                 {WALL_NEEDED} {}",
                torque(-0.8868, 2),
                signed_percent(-0.53),
                quoted(2.2307, 1, "mm"),
                mm(0.7995),
                torque(0.6121, 2),
                signed_percent(0.3658),
                mm(-0.4797)
            )
        );

        // (c): the calibration note right after the number, then the shaft joints' quoted
        // needs, named by their results' labels. (Its fraction to 5 decimals: -41.05 % at two
        // reads -41.0 %, while the side, -41.0516 %, reads -41.1 %.)
        let joints = label(SHAFT_JOINT);
        let need = |path: &str, x: f64| {
            format!(
                "{joints}: {} {}",
                label_for(path),
                quoted(x, 2, unit_of(path))
            )
        };
        assert_eq!(
            lines("(c)")[4],
            format!(
                "{count}: -2 -> {} ({}), {CALIBRATION_CHANGE}, {}, {}; +2 -> {SIDE_DOES_NOT_FIT}",
                torque(-0.5923, 2),
                signed_percent(-0.41052),
                need("shaft_joint.hub_needed_um", -2.43),
                need("shaft_joint.cup_needed_um", -1.9182)
            )
        );

        // The mixed pair's and the short cavity's Overall length rows: the jump both ways.
        let jumps = format!("{length}: -1 mm -> {JUMP}; +1 mm -> {JUMP}");
        for key in ["mixed", "short cavity"] {
            assert_eq!(lines(key)[2], jumps, "{key}");
        }

        // The odd count: no size rows; the gap's numbers, and the count's two sides in words.
        assert_eq!(
            lines("odd count"),
            [
                CAPTION.to_owned(),
                format!(
                    "{gap}: -0.1 mm -> {} ({}), {clearance} {}, {cup_wall}: {WALL_NEEDED} {}, \
                     {gearbox}: {COLD_HIGH} {}; +0.1 mm -> {} ({}), {clearance} {}, \
                     {cup_wall}: {WALL_NEEDED} {}, {gearbox}: {COLD_HIGH} {}",
                    torque(0.0653, 2),
                    signed_percent(0.0322),
                    mm(-0.1),
                    mm(0.0202),
                    cold(0.0230),
                    torque(-0.0628, 2),
                    signed_percent(-0.031),
                    mm(0.1),
                    mm(-0.0197),
                    cold(-0.0221)
                ),
                format!("{count}: -2 -> {NOT_ON_THE_SLIDER}; +2 -> {NOT_ON_THE_SLIDER}"),
            ]
        );

        // (b) with Auto on the magnet length: the caption, the Auto line, then the held row in
        // its place.
        let b = find(&designs, "(b)")
            .card()
            .lines(Some(FreeVariable::AxialLength));
        assert_eq!(b[..2], [CAPTION, AUTO_HOLDS_LENGTH]);
        assert_eq!(b[3], held_line(Control::OverallLength));
        assert_eq!(b[3], format!("{length}: {}", size_controls::SET_BY_AUTO));
        assert_eq!(b.len(), 6);
    }

    #[test]
    fn the_flips_and_the_margins_name_the_checks_that_move() {
        // Spec 8.1 test 7 (owner 8A; spec 3.6). At every design of section 2, against each side's
        // stepped design rebuilt here: the flips are exactly the counted checks whose level
        // differs, in the guides' order; each quoted change is of a check that fails on both
        // designs (never Hot-low torque, which is the row's own number, nor Temperature limits,
        // whose terms read stored 3D fields: T-13), is drawn (`shows`), and is in its result's
        // own unit, one of mm, µm, MPa, °C and N·m.
        let designs = card_designs();
        let mut quotes = 0;
        for design in &designs {
            let card = design.card();
            let before = compute_all(&design.inputs);
            for row in card.rows() {
                for (more, side) in sides(row) {
                    let Side::Change(change) = side else {
                        continue;
                    };
                    let what = format!("{} {:?} {more}", design.key, row.control);
                    let stepped = slider_step(
                        row.control,
                        &design.inputs,
                        design.minimum_diameter_mm,
                        more,
                    );
                    let after = compute_all(&stepped);
                    let flips: Vec<Flip> = counted_guides(PAUSED)
                        .filter_map(|guide| {
                            let to = level(guide, &stepped, &after);
                            (level(guide, &design.inputs, &before) != to).then_some(Flip {
                                check: guide.label,
                                to,
                            })
                        })
                        .collect();
                    assert_eq!(change.flips, flips, "{what}");
                    for margin in &change.margins {
                        let guide = counted_guides(PAUSED)
                            .find(|guide| guide.label == margin.check)
                            .unwrap_or_else(|| panic!("{what}: {margin:?}"));
                        assert_eq!(level(guide, &design.inputs, &before), Level::Bad, "{what}");
                        assert_eq!(level(guide, &stepped, &after), Level::Bad, "{what}");
                        assert!(![HOT_MIN, TEMPERATURE].contains(&guide.check), "{what}");
                        assert!(shows(margin.change, margin.decimals), "{what}: {margin:?}");
                        assert!(QUOTED_UNITS.contains(&margin.unit), "{what}: {margin:?}");
                        // The unit and the decimals are those of a term the check quotes.
                        assert!(
                            check_terms(guide.check, &design.inputs, &before)
                                .iter()
                                .filter_map(|term| term.quoted)
                                .any(|q| q.name == margin.name
                                    && q.decimals == margin.decimals
                                    && unit_of(q.path) == margin.unit),
                            "{what}: {margin:?}"
                        );
                        quotes += 1;
                    }
                }
            }
        }
        // Not vacuous: every quoted change at section 2's designs.
        assert_eq!(quotes, 517);

        let design = |key| find(&designs, key);
        // (a) gap -0.1 mm quotes Running clearance -0.10 mm (owner 8A's example), and the
        // gearbox's quantity by its name (T-11).
        let gap = number(&design("(a)").side(Control::FaceGap, false));
        assert_eq!(
            (gap.margins[0].check, gap.margins[0].name),
            (label(CLEARANCE), None)
        );
        assert_eq!(signed_change(gap.margins[0].change, 2), "-0.10");
        let gearbox = gap.margins.last().expect("the gearbox's quantity");
        assert_eq!(
            (gearbox.check, gearbox.name),
            (label(GEARBOX), Some(COLD_HIGH))
        );
        // A check that fails on both designs, whose quantity the step does not move, is not
        // quoted: at (a) a length step leaves the running clearance and the wall needed as they
        // were.
        let length = number(&design("(a)").side(Control::OverallLength, true));
        assert_eq!(
            length.margins.iter().map(|m| m.check).collect::<Vec<_>>(),
            [label(GEARBOX)]
        );
        // A term that fails on the stepped design only is quoted: at (b) the diameter +1 mm side
        // puts the Rotating diameter over the space claim, whose large-diameter stack already
        // fails, so no flip shows it.
        let b = design("(b)");
        let plus = number(&b.side(Control::OutsideDiameter, true));
        let rotating = SPACE_CLAIM_AXIS_SIZES[0].0;
        assert!(plus.flips.is_empty(), "{plus:?}");
        assert!(
            plus.margins.iter().any(|m| m.name == Some(rotating)),
            "{plus:?}"
        );
        let rotating_passes = |inputs: &DesignInputs| {
            check_terms(SPACE_CLAIM, inputs, &compute_all(inputs))[0].passes
        };
        assert!(rotating_passes(&b.inputs));
        assert!(!rotating_passes(&slider_step(
            Control::OutsideDiameter,
            &b.inputs,
            None,
            true
        )));
        // Only the joint fails: every number side quotes the inner hub's needed minimum.
        let need = label_for("shaft_joint.hub_needed_um");
        for row in design("only the joint fails").card().rows() {
            for (_, side) in sides(row) {
                if let Side::Change(change) = side {
                    assert!(
                        change.margins.iter().any(|m| m.name == Some(need)),
                        "{row:?}"
                    );
                }
            }
        }
        // The temperature edge: Temperature limits turns to a caution on three sides.
        let edge = design("temperature edge").card();
        let caution = Flip {
            check: label(TEMPERATURE),
            to: Level::Caution,
        };
        let cautions: Vec<(Control, bool)> = edge
            .rows()
            .iter()
            .flat_map(|row| {
                sides(row)
                    .into_iter()
                    .filter(
                        |(_, side)| matches!(side, Side::Change(c) if c.flips.contains(&caution)),
                    )
                    .map(move |(more, _)| (row.control, more))
            })
            .collect();
        assert_eq!(
            cautions,
            [
                (Control::OutsideDiameter, true),
                (Control::OverallLength, true),
                (Control::MagnetsPerRing, false)
            ]
        );
        // Temperature limits is never quoted (T-13). Its verdict is amber at worst (`verdict_level`
        // reads OK or CHECK), so it never fails on both designs; and its terms quote nothing at
        // any design, so no level could let one through.
        let temperature = guide(TEMPERATURE).expect("the temperature guide");
        for design in &designs {
            let results = compute_all(&design.inputs);
            assert!(
                check_terms(TEMPERATURE, &design.inputs, &results)
                    .iter()
                    .all(|term| term.quoted.is_none()),
                "{}",
                design.key
            );
            assert_ne!(
                level(temperature, &design.inputs, &results),
                Level::Bad,
                "{}",
                design.key
            );
        }
    }

    #[test]
    fn the_card_names_the_clamp_only_while_unpaused() {
        // Spec 8.1 test 7a (code review M1): the flips run over the guides that count at the
        // panel's pause. Unpaused, the clamp's guide counts: at the clamp's limit the Overall
        // length +1 mm side flips it to fails, and no clamp quantity is quoted (`check_terms`
        // gives the clamp none). Paused, as shipped, no side names it.
        let at_limit = clamp_at_its_limit();
        let results = compute_all(&at_limit);
        let clamp = label(CLAMP);
        let names_the_clamp = |card: &TorqueCard| {
            card.rows().iter().any(|row| {
                sides(row).into_iter().any(|(_, side)| match side {
                    Side::Change(change) => {
                        change.flips.iter().any(|flip| flip.check == clamp)
                            || change.margins.iter().any(|margin| margin.check == clamp)
                    }
                    Side::Words(_) => false,
                })
            })
        };
        let unpaused = card(&at_limit, &results, None, false);
        let longer = number(&unpaused.rows()[1].more);
        assert!(
            longer.flips.contains(&Flip {
                check: clamp,
                to: Level::Bad
            }),
            "{longer:?}"
        );
        assert!(names_the_clamp(&unpaused));
        assert!(unpaused.rows().iter().all(|row| {
            sides(row).into_iter().all(|(_, side)| match side {
                Side::Change(change) => change.margins.iter().all(|m| m.check != clamp),
                Side::Words(_) => true,
            })
        }));
        assert!(!names_the_clamp(&card(&at_limit, &results, None, true)));
        for inputs in [at_limit, only_the_clamp_fails()] {
            let paused = card_at(&inputs, None);
            assert!(!names_the_clamp(&paused));
            for line in paused.lines(None) {
                assert!(!line.to_lowercase().contains("clamp"), "{line}");
            }
        }
    }

    #[test]
    fn the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s() {
        // Spec 8.1 test 8 (owner 5A; spec 3.4): Auto's marking is drawn by `lines`, outside the
        // card. For each Auto variable at (a) and (e2): the held row, the one `held_by` names,
        // reads `held_line` after the caption and the Auto line; the other rows read as with
        // Auto off, which shows the held row's numbers; one card serves both.
        let designs = card_designs();
        for key in ["(a)", "(e2)"] {
            let card = find(&designs, key).card();
            assert_eq!(
                card,
                find(&designs, key).card(),
                "the card is the design's alone"
            );
            let plain = card.lines(None);
            assert_eq!(plain[0], CAPTION);
            for variable in FreeVariable::ALL {
                let held = Control::ALL
                    .into_iter()
                    .position(|control| control.held_by() == Some(variable))
                    .expect("a control for every variable");
                let mut expected = plain.clone();
                expected.insert(1, auto_line(variable).to_owned());
                expected[2 + held] = held_line(Control::ALL[held]);
                assert_eq!(card.lines(Some(variable)), expected, "{key} {variable:?}");
                // Auto off, the held row shows its numbers.
                assert!(
                    plain[1 + held].starts_with(&format!("{}: -", Control::ALL[held].label())),
                    "{}",
                    plain[1 + held]
                );
            }
        }
        // The three Auto lines, each the variable's own.
        assert_eq!(
            FreeVariable::ALL.map(auto_line),
            [AUTO_HOLDS_LENGTH, AUTO_HOLDS_COUNT, AUTO_HOLDS_RING]
        );
        // The odd count has no size rows (T-5): with Auto on the ring size or the magnet length
        // the Auto line is drawn and no row is held; with Auto on the count, its row is.
        let odd = find(&designs, "odd count").card();
        let plain = odd.lines(None);
        assert_eq!(plain.len(), 3);
        for variable in [FreeVariable::RingRadius, FreeVariable::AxialLength] {
            assert_eq!(
                odd.lines(Some(variable)),
                [
                    CAPTION.to_owned(),
                    auto_line(variable).to_owned(),
                    plain[1].clone(),
                    plain[2].clone()
                ]
            );
        }
        assert_eq!(
            odd.lines(Some(FreeVariable::MagnetsPerRing)),
            [
                CAPTION.to_owned(),
                AUTO_HOLDS_COUNT.to_owned(),
                plain[1].clone(),
                held_line(Control::MagnetsPerRing)
            ]
        );
        // Each Auto design of section 2 draws its held row once.
        for design in &designs {
            let Some(variable) = design.auto else {
                continue;
            };
            let lines = design.card().lines(Some(variable));
            let held = lines
                .iter()
                .filter(|line| line.ends_with(size_controls::SET_BY_AUTO))
                .count();
            assert_eq!(held, 1, "{}: {lines:?}", design.key);
        }
    }

    #[test]
    fn size_changes_and_margins_are_drawn_only_when_they_show() {
        // Spec 8.1 test 9a (code review M2): section 5's cap keeps float noise off the screen.
        for from in [1, 2] {
            for x in [
                0.0,
                1e-15,
                -1e-15,
                3.6e-15,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                assert!(!shows(x, from), "{x} from {from}");
            }
            assert!(shows(0.0046, from), "from {from}");
        }
        assert!(shows(6e-5, 2));
        assert!(!shows(6e-5, 1));
        // The odd count has no size controls: its gap sides carry no size change, and its card
        // equals a second card of it (no NaN stands for "no value").
        let designs = card_designs();
        let odd = find(&designs, "odd count");
        for (_, side) in sides(&odd.card().rows()[0]) {
            let change = number(side);
            assert_eq!(
                (change.diameter_change_mm, change.length_change_mm),
                (None, None)
            );
        }
        assert_eq!(odd.card(), odd.card());
        // At (a) no side draws a size change: the cap sets the Outside diameter at every stepped
        // design, and the length moves by float noise only.
        let size_labels = [Control::OutsideDiameter, Control::OverallLength].map(Control::label);
        let draws_a_size = |change: &Change, control: Control| {
            change.costs(control).iter().any(|cost| {
                size_labels
                    .iter()
                    .any(|label| cost.starts_with(&format!("{label} ")))
            })
        };
        let a = find(&designs, "(a)");
        for row in a.card().rows() {
            for (_, side) in sides(row) {
                if let Side::Change(change) = side {
                    assert!(!draws_a_size(change, row.control), "{row:?}");
                }
            }
        }
        // A side with a planted 1e-15 mm size change draws nothing for it; 0.04 mm draws, but
        // never on the row of its own dimension.
        let mut planted = number(&a.side(Control::FaceGap, true));
        let drawn = planted.costs(Control::FaceGap);
        planted.diameter_change_mm = Some(1e-15);
        planted.length_change_mm = Some(-1e-15);
        assert_eq!(planted.costs(Control::FaceGap), drawn);
        planted.diameter_change_mm = Some(0.04);
        assert!(draws_a_size(&planted, Control::FaceGap));
        assert!(!draws_a_size(&planted, Control::OutsideDiameter));
    }

    #[test]
    fn a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required() {
        // Spec T-14 and 10.1 step 6 (the one-flip line on the STOP sheet): the table's "T-14 one
        // flip", (e2) with Auto on the magnet length shown with Auto off and Required halfway
        // between its Outside diameter sides' hot-low. Both sides draw the same N·m and percent
        // at the row's decimals, and only the side whose hot-low is below Required flips
        // Hot-low torque.
        let designs = card_designs();
        let design = find(&designs, "T-14 one flip");
        let required = design.inputs.metal.required_min_Nm;
        let card = design.card();
        let row = &card.rows()[0];
        assert_eq!(row.control, Control::OutsideDiameter);
        let hot_low = Flip {
            check: label(HOT_MIN),
            to: Level::Bad,
        };
        let sides = [number(&row.fewer), number(&row.more)];
        let flips = sides.each_ref().map(|side| side.flips.contains(&hot_low));
        let below = sides.each_ref().map(|side| side.hot_low_Nm < required);
        assert_eq!(flips, below);
        assert_eq!(flips.iter().filter(|&&flip| flip).count(), 1, "{flips:?}");
        let lines = card.lines(None);
        let [fewer, more] = side_texts(&lines[1], row);
        assert_eq!(drawn_number(fewer), drawn_number(more), "{fewer} | {more}");
    }

    #[test]
    fn where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift() {
        // Spec R-13 (on the STOP sheet; no diameter jump rule): the table's "R-13 boss plateau".
        // The boss sets the Outside diameter at the slider's bottom: -1 mm reads the end of its
        // range, and +1 mm, which lifts the cup from its own size to the boss plus the step, is
        // a number (+7.55 %), as the slider does.
        let designs = card_designs();
        let design = find(&designs, "R-13 boss plateau");
        assert_eq!(
            design.side(Control::OutsideDiameter, false),
            Side::Words(Words::EndOfRange)
        );
        let more = number(&design.side(Control::OutsideDiameter, true));
        assert_eq!(signed_fixed(100.0 * more.fraction, 2), "+7.55");
    }

    #[test]
    fn a_step_to_a_design_without_size_controls_draws_no_size_change() {
        // Spec 2.2: a size change is the sliders' own (`dimensions`) of both designs, none where
        // either has no size controls. A design file can put the ring past its slider: at 6
        // magnets on a ring of 38.0 mm the Size tab draws its size controls, while at 4 magnets
        // (the count row's -2 side) it cannot define them, so that side draws its number and no
        // size change.
        use crate::gui::test_support::with_poles;
        let mut inputs = with_poles(6);
        inputs.coupling.inner_back_apothem_mm = 38.0;
        let mut four = inputs.clone();
        four.coupling.npole = 4;
        assert!(dimensions(&inputs).is_some());
        assert!(dimensions(&four).is_none());
        let fewer = number(&side_at(&inputs, None, Control::MagnetsPerRing, false));
        assert_eq!(
            (fewer.diameter_change_mm, fewer.length_change_mm),
            (None, None)
        );
        let card = card_at(&inputs, None);
        let line = card.lines(None).pop().expect("the count row's line");
        let [fewer_text, _] = side_texts(&line, &card.rows()[3]);
        assert!(
            !fewer_text.contains(Control::OutsideDiameter.label()),
            "{fewer_text}"
        );
    }

    #[test]
    fn the_card_never_panics_at_the_values_a_design_file_can_carry() {
        // A design file or a share link can carry any finite value (`validate()` checks only
        // that a number is finite), and the panel draws the card on the first view of every
        // design it opens, so a panic here ends the app (wasm). At each such value, at both
        // back irons and at the Minimum OD's top, the card is computed and equals a second
        // card of the same design (no field holds a NaN, spec 3.1), and every line of every
        // Auto state is drawn in plain words with the font's glyphs. What the numbers read at
        // the degenerate ones (a count of 2, a calibration coefficient of 1e-9 or 1e300, a
        // variation just under 1) is the owner's to rule at the release stop.
        type Edit = fn(&mut DesignInputs);
        let edits: [(&str, Edit); 47] = [
            ("npole 0", |d| d.coupling.npole = 0),
            ("npole 1", |d| d.coupling.npole = 1),
            ("npole 2", |d| d.coupling.npole = 2),
            ("npole 3", |d| d.coupling.npole = 3),
            ("npole -2", |d| d.coupling.npole = -2),
            ("npole 42", |d| d.coupling.npole = 42),
            ("npole 1e6", |d| d.coupling.npole = 1_000_000),
            ("npole i64::MAX", |d| d.coupling.npole = i64::MAX),
            ("npole i64::MIN", |d| d.coupling.npole = i64::MIN),
            ("gap 0", |d| d.metal.face_gap_mm = 0.0),
            ("gap -1", |d| d.metal.face_gap_mm = -1.0),
            ("gap 5.5", |d| d.metal.face_gap_mm = 5.5),
            ("gap 1e300", |d| d.metal.face_gap_mm = 1e300),
            ("override 0.1", |d| {
                d.coupling.magnets.axial_length_mm = Some(0.1)
            }),
            ("override 60", |d| {
                d.coupling.magnets.axial_length_mm = Some(60.0)
            }),
            ("override 1e300", |d| {
                d.coupling.magnets.axial_length_mm = Some(1e300)
            }),
            ("override -1", |d| {
                d.coupling.magnets.axial_length_mm = Some(-1.0)
            }),
            ("ring 0", |d| d.coupling.inner_back_apothem_mm = 0.0),
            ("ring 0.5", |d| d.coupling.inner_back_apothem_mm = 0.5),
            ("ring 200", |d| d.coupling.inner_back_apothem_mm = 200.0),
            ("ring 1e300", |d| d.coupling.inner_back_apothem_mm = 1e300),
            ("ring -1", |d| d.coupling.inner_back_apothem_mm = -1.0),
            ("cup depth 0", |d| d.metal.cup_depth_mm = 0.0),
            ("cup depth 1e300", |d| d.metal.cup_depth_mm = 1e300),
            ("boss 0", |d| d.metal.boss_od_mm = 0.0),
            ("boss 200", |d| d.metal.boss_od_mm = 200.0),
            ("cap 0", |d| d.metal.cap_od_mm = 0.0),
            ("cap 1e300", |d| d.metal.cap_od_mm = 1e300),
            ("bore 0", |d| d.coupling.bore_mm = 0.0),
            ("bore 1e300", |d| d.coupling.bore_mm = 1e300),
            ("variation 1 - 1e-10", |d| d.metal.variation = 1.0 - 1e-10),
            ("variation -1", |d| d.metal.variation = -1.0),
            ("variation 1e300", |d| d.metal.variation = 1e300),
            ("inner Br 1e-6 T", |d| {
                d.coupling.magnets.part_inner.clear();
                d.coupling.magnets.part_outer.clear();
                d.coupling.magnets.manual_inner_br_T = 1e-6;
            }),
            ("inner Br -1 T", |d| {
                d.coupling.magnets.part_inner.clear();
                d.coupling.magnets.part_outer.clear();
                d.coupling.magnets.manual_inner_br_T = -1.0;
            }),
            ("no back iron, 8 magnets, f_cal_original 1e-9", |d| {
                d.coupling.backiron = 0;
                d.coupling.npole = 8;
                d.calibration.f_cal_original = 1e-9;
            }),
            ("no back iron, 8 magnets, f_cal_original 1e-300", |d| {
                d.coupling.backiron = 0;
                d.coupling.npole = 8;
                d.calibration.f_cal_original = 1e-300;
            }),
            ("no back iron, 8 magnets, f_cal_original 1e-320", |d| {
                d.coupling.backiron = 0;
                d.coupling.npole = 8;
                d.calibration.f_cal_original = 1e-320;
            }),
            ("f_cal_original 1e300", |d| {
                d.calibration.f_cal_original = 1e300
            }),
            ("required 1e-300", |d| d.metal.required_min_Nm = 1e-300),
            ("required -1", |d| d.metal.required_min_Nm = -1.0),
            ("c_end 0", |d| d.coupling.c_end = 0.0),
            ("c_end 1e300", |d| d.coupling.c_end = 1e300),
            ("gearbox rating 0", |d| {
                d.coupling.gearbox_input_rating_Nm = 0.0
            }),
            ("residual target 1e300", |d| {
                d.metal.residual_target_mm = 1e300
            }),
            ("hot ambient 1e300", |d| {
                d.temperature.duty.hot_ambient_C = 1e300
            }),
            ("hot ambient -1e300", |d| {
                d.temperature.duty.hot_ambient_C = -1e300
            }),
        ];
        let ctx = egui::Context::default();
        // The fonts load on the context's first frame.
        sized_frame(&ctx, SCREEN, Vec::new(), |_| {});
        let top = dimensions(&DesignInputs::default())
            .expect("size controls")
            .minimum_diameter_range
            .max;
        let mut rows = 0;
        for backiron in [1, 0] {
            for (name, edit) in edits {
                for minimum in [None, Some(top)] {
                    let mut inputs = DesignInputs::default();
                    inputs.coupling.backiron = backiron;
                    edit(&mut inputs);
                    let what = format!("{name}, back iron {backiron}, Minimum OD {minimum:?}");
                    let shown = card_at(&inputs, minimum);
                    assert_eq!(shown, card_at(&inputs, minimum), "{what}");
                    rows += shown.rows().len();
                    for auto in [None].into_iter().chain(FreeVariable::ALL.map(Some)) {
                        let lines = shown.lines(auto);
                        assert!(!lines.is_empty(), "{what}");
                        for line in &lines {
                            assert!(!line.is_empty(), "{what}");
                            assert_eq!(
                                plain_text_violations(line),
                                Vec::<&str>::new(),
                                "{what}: {line}"
                            );
                            assert_glyphs(&ctx, line, &what);
                        }
                    }
                }
            }
        }
        // Not vacuous: most of these designs draw rows.
        assert!(rows > 200, "{rows} rows");
    }

    /// The side texts of a row's line, "{label}: {fewer}; {more}": the two sides.
    fn side_texts<'a>(line: &'a str, row: &Row) -> [&'a str; 2] {
        let texts = line
            .strip_prefix(&format!("{}: ", row.control.label()))
            .unwrap_or_else(|| panic!("a row's line: {line}"));
        <[&str; 2]>::try_from(texts.split("; ").collect::<Vec<_>>())
            .unwrap_or_else(|_| panic!("two sides: {line}"))
    }

    /// The N·m and the percent a number side's text draws: "{step} -> {N·m} N·m ({percent} %)".
    fn drawn_number(side: &str) -> (&str, &str) {
        let (_, number) = side.split_once(" -> ").expect("a step");
        let (torque, rest) = number.split_once(" N·m (").expect("a number side");
        let (percent, _) = rest.split_once(" %)").expect("a percent");
        (torque, percent)
    }

    #[test]
    fn no_card_number_reads_zero_or_disagrees_with_its_percent() {
        // Spec 8.3 test 15 (decision 10B): at every design of section 2, each number side's N·m
        // reads a digit that is not zero unless the change is under the format's cap (T-9),
        // and its sign is its percent's.
        let designs = card_designs();
        let mut numbers = 0;
        for design in &designs {
            let card = design.card();
            let lines = card.lines(None);
            for (row, line) in card.rows().iter().zip(&lines[1..]) {
                for ((_, side), text) in sides(row).into_iter().zip(side_texts(line, row)) {
                    let Side::Change(change) = side else {
                        continue;
                    };
                    let (torque, percent) = drawn_number(text);
                    assert!(
                        reads_not_zero(torque) || !shows(change.change_Nm, TORQUE_DECIMALS),
                        "{}: {text}",
                        design.key
                    );
                    assert_eq!(
                        torque.chars().next(),
                        percent.chars().next(),
                        "{}: {text}",
                        design.key
                    );
                    numbers += 1;
                }
            }
        }
        assert_eq!(numbers, NUMBER_SIDES);
        // Below zero: (c) with a design file's calibration coefficient of -0.95, two magnets
        // fewer.
        let card = find(&designs, "(c), f_cal_original -0.95").card();
        let count = &card.rows()[3];
        let lines = card.lines(None);
        let (torque, percent) = drawn_number(side_texts(&lines[4], count)[0]);
        assert!(number(&count.fewer).hot_low_Nm < 0.0);
        assert!(
            torque.starts_with('-') && percent.starts_with('-'),
            "{torque} {percent}"
        );
        // The one-line forms draw no number.
        for key in [
            "short magnets",
            "shown does not fit",
            "variation 1.0",
            "zero torque",
        ] {
            for line in find(&designs, key).card().lines(None) {
                assert!(!line.chars().any(|c| c.is_ascii_digit()), "{key}: {line}");
            }
        }
    }

    /// Every line the card draws at every design of section 2, with each Auto state, and its
    /// heading: what the word and glyph tests scan.
    fn every_card_text() -> Vec<(String, String)> {
        let mut texts = vec![("heading".to_owned(), CARD_HEADING.to_owned())];
        for design in card_designs() {
            let card = design.card();
            for auto in [None].into_iter().chain(FreeVariable::ALL.map(Some)) {
                for line in card.lines(auto) {
                    texts.push((format!("{} {auto:?}", design.key), line));
                }
            }
        }
        texts
    }

    #[test]
    fn the_card_texts_never_rank_the_controls() {
        // Spec 8.4 test 16 (owner 14A; spec 4.4): no card text ranks the controls
        // (`CARD_RANKING_WORDS`; its matcher's own examples are in `engine::explain::plain`),
        // orders the torques, says the coupling protects the gearbox, names the paused clamp
        // section, or says what the plain-words guard bans.
        let texts = every_card_text();
        // Not vacuous: the heading, and every line at every design with each Auto state.
        assert_eq!(texts.len(), 1056);
        for (what, text) in &texts {
            assert_eq!(ranking_words_in(text), Vec::<&str>::new(), "{what}: {text}");
            let lower = text.to_lowercase();
            for word in TORQUE_ORDER_WORDS {
                assert!(!lower.contains(word), "{what}: {text}");
            }
            assert!(!mentions_gearbox_protection(text), "{what}: {text}");
            assert!(!lower.contains("clamp"), "{what}: {text}");
            assert_eq!(
                plain_text_violations(text),
                Vec::<&str>::new(),
                "{what}: {text}"
            );
        }
    }

    #[test]
    fn every_card_text_has_its_glyphs() {
        // Spec 8.4 test 17.
        let ctx = egui::Context::default();
        // The fonts load on the context's first frame.
        sized_frame(&ctx, SCREEN, Vec::new(), |_| {});
        for (what, text) in every_card_text() {
            assert_glyphs(&ctx, &text, &what);
        }
    }

    /// `line` with each number it draws replaced by '#' (its skeleton), and those numbers. A
    /// side's step ("-1", "+0.1", "-2" before " mm -> " or " -> ") is the template's, so it
    /// stays in the skeleton.
    fn numbers_in(line: &str) -> (String, Vec<String>) {
        let chars: Vec<char> = line.chars().collect();
        let (mut skeleton, mut numbers) = (String::new(), Vec::new());
        let mut i = 0;
        while i < chars.len() {
            let signed =
                matches!(chars[i], '+' | '-') && chars.get(i + 1).is_some_and(char::is_ascii_digit);
            if !(signed || chars[i].is_ascii_digit()) {
                skeleton.push(chars[i]);
                i += 1;
                continue;
            }
            let mut end = i + 1;
            while end < chars.len() && (chars[end].is_ascii_digit() || chars[end] == '.') {
                end += 1;
            }
            let number: String = chars[i..end].iter().collect();
            let after: String = chars[end..].iter().collect();
            if after.starts_with(" mm -> ") || after.starts_with(" -> ") {
                skeleton.push_str(&number);
            } else {
                skeleton.push('#');
                numbers.push(number);
            }
            i = end;
        }
        (skeleton, numbers)
    }

    #[test]
    fn the_card_s_strings_are_constants_and_its_numbers_come_from_results() {
        // Spec 8.4 test 18. Each design's card against a copy whose every number (each change in
        // N·m and as a fraction, each size change and each quoted change) is tripled: every line
        // keeps its skeleton, so its words are the template's and the constants', and differs at
        // every number it draws, so each number is read from the card (whose fields test 1 holds
        // to compute_all). Three times a drawn value moves its first drawn digit; two designs
        // could not show it, since a gap step's Running clearance change is the step itself at
        // every design, and a torque that changes in proportion keeps every percent.
        let tripled = |card: &TorqueCard| -> TorqueCard {
            let mut card = card.clone();
            if let TorqueCard::Rows(rows) = &mut card {
                for row in rows {
                    for side in [&mut row.fewer, &mut row.more] {
                        if let Side::Change(change) = side {
                            change.change_Nm *= 3.0;
                            change.fraction *= 3.0;
                            change.diameter_change_mm =
                                change.diameter_change_mm.map(|mm| 3.0 * mm);
                            change.length_change_mm = change.length_change_mm.map(|mm| 3.0 * mm);
                            for margin in &mut change.margins {
                                margin.change *= 3.0;
                            }
                        }
                    }
                }
            }
            card
        };
        let mut numbers = 0;
        for design in card_designs() {
            let card = design.card();
            let moved = tripled(&card);
            for (line, moved_line) in card.lines(design.auto).iter().zip(moved.lines(design.auto)) {
                let (skeleton, drawn) = numbers_in(line);
                let (moved_skeleton, moved_drawn) = numbers_in(&moved_line);
                assert_eq!(skeleton, moved_skeleton, "{}", design.key);
                for (number, moved_number) in drawn.iter().zip(&moved_drawn) {
                    assert_ne!(number, moved_number, "{}: {line}", design.key);
                    numbers += 1;
                }
            }
        }
        // Not vacuous: every number drawn at section 2's designs.
        assert_eq!(numbers, 1049);
    }
}
```

Task 3's two not-vacuous pins move with the two designs the table gains in (b) (plan review 2.2): `NUMBER_SIDES` 263 -> 277,

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 477):
```rust
    const NUMBER_SIDES: usize = 263;
```

New:
```rust
    const NUMBER_SIDES: usize = 277;
```

and test 6's diameter rows 42 -> 44:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 1039):
```rust
        assert_eq!(rows, 42);
```

New:
```rust
        assert_eq!(rows, 44);
```

(b) `magcoupling-rs/src/gui/test_support.rs`: the table gains each design's Auto state, and T-14's one-flip design and R-13's boss plateau (plan review 2.2):

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 333):
```rust
/// One design of the torque card's table ([`card_designs`]) as the panel shows it: its key,
/// the design shown (an Auto design solved as the panel solves it) and the Size tab's Minimum
/// OD preference.
pub(crate) struct CardDesign {
    pub(crate) key: &'static str,
    pub(crate) inputs: crate::DesignInputs,
    pub(crate) minimum_diameter_mm: Option<f64>,
}
```

New:
```rust
/// One design of the torque card's table ([`card_designs`]) as the panel shows it: its key,
/// the design shown (an Auto design solved as the panel solves it), the Size tab's Minimum OD
/// preference and the Auto state.
pub(crate) struct CardDesign {
    pub(crate) key: &'static str,
    pub(crate) inputs: crate::DesignInputs,
    pub(crate) minimum_diameter_mm: Option<f64>,
    pub(crate) auto: Option<crate::engine::sizing::FreeVariable>,
}
```

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 414):
```rust
/// its Minimum OD preference (an Auto design solved as the panel solves it).
```

New:
```rust
/// its Minimum OD preference and Auto state (an Auto design solved as the panel solves it).
```

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 462):
```rust
    let plain = |key, inputs| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: None,
    };
    let at_minimum = |key, inputs, minimum: f64| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: Some(minimum),
    };
    let solved = |key, inputs: DesignInputs, variable, minimum: Option<f64>| CardDesign {
        key,
        inputs: auto_shown(&inputs, variable, minimum),
        minimum_diameter_mm: minimum,
    };
```

New:
```rust
    let plain = |key, inputs| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: None,
        auto: None,
    };
    let at_minimum = |key, inputs, minimum: f64| CardDesign {
        key,
        inputs,
        minimum_diameter_mm: Some(minimum),
        auto: None,
    };
    let solved = |key, inputs: DesignInputs, variable, minimum: Option<f64>| CardDesign {
        key,
        inputs: auto_shown(&inputs, variable, minimum),
        minimum_diameter_mm: minimum,
        auto: Some(variable),
    };
```

R-13's boss plateau, a shared design after `with_variation` (plan review 2.2):

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 316):
```rust
pub(crate) fn with_variation(variation: f64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.variation = variation;
    inputs
}
```

New:
```rust
pub(crate) fn with_variation(variation: f64) -> crate::DesignInputs {
    let mut inputs = crate::DesignInputs::default();
    inputs.metal.variation = variation;
    inputs
}

/// The shaft boss setting the Outside diameter (slice 3 spec R-13, the boss plateau): 8 magnets
/// per ring on a ring of 8.0 mm, the cap at 30 mm with its thread at 29 mm, and the boss at
/// 40 mm. The magnets fit and the boss sets the Outside diameter, 40.00 mm, the slider's bottom
/// (its range 40.00 to 85.41 mm): one +1 mm step lifts the cup from its own size to the boss
/// plus the step.
pub(crate) fn boss_plateau() -> crate::DesignInputs {
    let mut inputs = with_poles(8);
    inputs.coupling.inner_back_apothem_mm = 8.0;
    inputs.metal.cap_od_mm = 30.0;
    inputs.metal.cap_thread_dia_mm = 29.0;
    inputs.metal.boss_od_mm = 40.0;
    inputs
}
```

In `card_designs()`, T-14's one-flip design, after the hot day:

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 494):
```rust
    let hot = {
        let mut inputs = a();
        hot_day(&mut inputs);
        inputs
    };
```

New:
```rust
    let hot = {
        let mut inputs = a();
        hot_day(&mut inputs);
        inputs
    };
    // T-14's one-flip row (slice 3 spec 10.1 step 6): (e2) with Auto on the magnet length,
    // shown with Auto off and Required halfway between its Outside diameter sides' hot-low, so
    // the two sides read alike while only one crosses Required.
    let one_flip = {
        use crate::gui::torque_card::Side;
        let mut shown = auto_shown(&e2(), AxialLength, None);
        let card = card_at(&shown, None);
        let row = &card.rows()[0];
        let hot_low = |side: &Side| match side {
            Side::Change(change) => change.hot_low_Nm,
            Side::Words(words) => panic!("an Outside diameter side reads {words:?}"),
        };
        shown.metal.required_min_Nm = (hot_low(&row.fewer) + hot_low(&row.more)) / 2.0;
        shown
    };
```

The table's two new entries, after (c) with `f_cal_original` -0.95:

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 589):
```rust
        plain(
            "(c), f_cal_original -0.95",
            no_back_iron_negative_calibration(),
        ),
        plain("count overflow: i64::MAX", with_poles(i64::MAX)),
```

New:
```rust
        plain(
            "(c), f_cal_original -0.95",
            no_back_iron_negative_calibration(),
        ),
        plain("T-14 one flip", one_flip),
        plain("R-13 boss plateau", boss_plateau()),
        plain("count overflow: i64::MAX", with_poles(i64::MAX)),
```

Test 1a (`the_shared_card_designs_read_as_section_2_1_says`, in the same file's test module) pins the boss plateau's facts at its end:

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 1108):
```rust
        assert_eq!(fixed(hot_low(&with_variation(1.5)), 4), "-1.3442");
    }
}
```

New:
```rust
        assert_eq!(fixed(hot_low(&with_variation(1.5)), 4), "-1.3442");

        // The boss plateau (R-13): the magnets fit, and the boss sets the Outside diameter at the
        // bottom of the slider's range.
        let boss = boss_plateau();
        let size = dimensions(&boss).expect("the boss plateau has size controls");
        assert!(blocks_fit(&boss.coupling, &compute_all(&boss).model));
        assert_eq!(size.diameter_mm, boss.metal.boss_od_mm);
        assert_eq!(
            [
                size.diameter_mm,
                size.diameter_range.min,
                size.diameter_range.max
            ]
            .map(|mm| fixed(mm, 2)),
            ["40.00", "40.00", "85.41"]
        );
    }
}
```

(c) `magcoupling-rs/src/gui/panel.rs`, `smaller_exploration_shows_magnet_fit_and_keyway_breakthrough`: its expected texts come from the constants:

Old (`magcoupling-rs/src/gui/panel.rs`, from line 4897):
```rust
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
        assert_eq!(
            count(
                &output,
                "Exploration only: magnets do not fit at this diameter."
            ),
            1
        );
        assert_eq!(
            count(
                &output,
                "Exploration only: the keyway breaks through the hub."
            ),
            1
        );
```

New:
```rust
        harness.click_text("Size and torque");
        let output = harness.frame(Vec::new());
        // The lines' words are the torque card's side words (slice 3 spec 4.2).
        assert_eq!(
            count(
                &output,
                &format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter.")
            ),
            1
        );
        assert_eq!(
            count(
                &output,
                &format!("{EXPLORATION_ONLY}{KEYWAY_BREAKS_THROUGH}.")
            ),
            1
        );
```

- [ ] **Step 2: Run them and see the build fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib --no-run`

Expected: the library builds with no warning; the lib test target does not compile:
```text
error[E0599]: no method named `lines` found for enum `torque_card::TorqueCard` in the current scope      (21 of these)
error[E0425]: cannot find function `signed_percent` in this scope      (14 of these)
error[E0425]: cannot find function `signed_fixed` in this scope      (12 of these)
error[E0425]: cannot find value `CAPTION` in this scope      (6 of these)
error[E0425]: cannot find function `shows` in this scope      (6 of these)
error[E0425]: cannot find function `level` in this scope      (5 of these)
error[E0425]: cannot find value `SIDE_DOES_NOT_FIT` in this scope      (4 of these)
error[E0425]: cannot find value `FLIP_FAILS` in this scope      (3 of these)
error[E0425]: cannot find function `check_terms` in this scope      (3 of these)
error[E0425]: cannot find function `held_line` in this scope      (3 of these)
error[E0277]: the size for values of type `str` cannot be known at compilation time      (3 of these)
error[E0425]: cannot find value `EXPLORATION_ONLY` in this scope      (2 of these)
error[E0425]: cannot find value `CALIBRATION_CHANGE` in this scope      (2 of these)
error[E0425]: cannot find function `with_unit` in this scope      (2 of these)
error[E0425]: cannot find function `signed_change` in this scope      (2 of these)
error[E0425]: cannot find value `AUTO_HOLDS_LENGTH` in this scope      (2 of these)
error[E0425]: cannot find function `counted_guides` in this scope      (2 of these)
error[E0425]: cannot find value `AUTO_HOLDS_COUNT` in this scope      (2 of these)
error[E0425]: cannot find function `auto_line` in this scope      (2 of these)
error[E0599]: no method named `costs` found for struct `torque_card::Change` in the current scope      (2 of these)
error[E0425]: cannot find value `KEYWAY_BREAKS_THROUGH` in this scope
error[E0425]: cannot find value `SHOWN_DOES_NOT_FIT` in this scope
error[E0425]: cannot find value `SHOWN_NO_PREDICTION` in this scope
error[E0425]: cannot find value `SHOWN_NO_HOT_LOW` in this scope
error[E0425]: cannot find function `fixed` in this scope
error[E0425]: cannot find function `result_info` in this scope
error[E0425]: cannot find value `FLIP_PASSES` in this scope
error[E0425]: cannot find value `JUMP` in this scope
error[E0425]: cannot find value `NOT_ON_THE_SLIDER` in this scope
error[E0425]: cannot find value `auto_line` in this scope
error[E0425]: cannot find value `AUTO_HOLDS_RING` in this scope
error[E0425]: cannot find function `reads_not_zero` in this scope
error[E0425]: cannot find value `TORQUE_DECIMALS` in this scope
error[E0425]: cannot find value `CARD_HEADING` in this scope
error[E0603]: constant `CLAMP` is private
error[E0603]: constant `CLEARANCE` is private
error[E0603]: constant `CUP_WALL` is private
error[E0603]: constant `GEARBOX` is private
error[E0603]: constant `HOT_MIN` is private
error[E0603]: constant `SHAFT_JOINT` is private
error[E0603]: constant `SPACE_CLAIM` is private
error[E0603]: constant `TEMPERATURE` is private
error[E0599]: no method named `torque_decimals` found for reference `&torque_card::Row` in the current scope
error[E0599]: no method named `costs` found for reference `&torque_card::Change` in the current scope
error: could not compile `magcoupling-rs` (lib test) due to 122 previous errors
```

- [ ] **Step 3: What a step costs, and the card's lines (`gui/torque_card.rs`)**

(a) The module doc names what a step costs and how the card is drawn:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 11):
```rust
//! slider's mapping, the validity rule every row shares ([`side_words`]) and, on the Overall
//! length row, the jump.
```

New:
```rust
//! slider's mapping, the validity rule every row shares ([`side_words`]) and, on the Overall
//! length row, the jump. A number side also says what the step costs (owner 8A, spec 3.6): the
//! change in Outside diameter and Overall length, the counted dashboard checks whose level flips,
//! and the change of each failing check's own quantity ([`check_terms`]).
//!
//! [`TorqueCard::lines`] draws it: every text a named constant here (drafts until the slice's G-3
//! review sets them), every number from the results through the Size tab's one number format
//! (decision 10B, `gui::format`), a row's two N·m at one precision (T-14). The Auto marking is
//! drawn there too, outside the card the cache keeps, so one card serves every Auto state.
```

(b) The imports:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 25):
```rust
use crate::gui::dashboard::Level;
use crate::gui::guidance::FACE_GAP;
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
use crate::gui::size_controls::{self, SliderMapping, predicts, slider_minimum};
```

New:
```rust
use crate::gui::dashboard::{Level, result_info};
use crate::gui::format::{
    CHANGE_EXTRA_DECIMALS, TORQUE_DECIMALS, fixed, reads_not_zero, shared_change_decimals,
    signed_change, signed_fixed, signed_percent, with_unit,
};
use crate::gui::guidance::{FACE_GAP, check_terms, counted_guides, level, result_number};
use crate::gui::inputs::InputCatalogue;
use crate::gui::plain_text::label_for;
use crate::gui::size_controls::{self, SET_BY_AUTO, SliderMapping, predicts, slider_minimum};
```

(c) After `COUNT_STEP`: the decimals a size change starts from (a torque change's are Task 1's `TORQUE_DECIMALS`, critique A-14), and the card's texts (spec 4.1 and 4.2):

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 42):
```rust
/// The Magnets per ring step (owner 3A): two magnets, so an even count stays even.
const COUNT_STEP: i64 = 2;
```

New:
```rust
/// The Magnets per ring step (owner 3A): two magnets, so an even count stays even.
const COUNT_STEP: i64 = 2;

/// The decimals a size change starts from (spec section 5): the sliders' 0.1 mm. A torque change
/// starts from the Size tab's own, [`TORQUE_DECIMALS`] (`gui::format`).
const SIZE_DECIMALS: usize = 1;

// The card's texts (spec 4.1 and 4.2): drafts until the slice's G-3 review sets the words. A
// row's label is its Size control's ([`Control::label`]), a check's its guide's.

/// The card's heading (named apart from `size_controls::HEADING` and `panel::HEADING`).
pub const CARD_HEADING: &str = "One step of each size control";
/// The first line of the rows (owner 12A): every number on the card is a change.
pub const CAPTION: &str =
    "The change in the model's hot-low torque after one step of one control, everything else held.";
/// The Auto line with Auto on the magnet length (owner 5A): what Auto does after a change, and
/// what the rows keep.
pub const AUTO_HOLDS_LENGTH: &str =
    "Auto sets the magnet length again after a change; each row keeps the length shown.";
/// The Auto line with Auto on the ring size.
pub const AUTO_HOLDS_RING: &str =
    "Auto sizes the ring again after a change; each row keeps the ring radius shown.";
/// The Auto line with Auto on the magnet count.
pub const AUTO_HOLDS_COUNT: &str =
    "Auto sets the magnet count again after a change; each row keeps the count shown.";
/// A side off its slider's range ([`Words::EndOfRange`]; owner 9A's words).
pub const END_OF_RANGE: &str = "at the end of its range";
/// An odd count's sides ([`Words::NotOnTheSlider`]; T-4).
pub const NOT_ON_THE_SLIDER: &str = "not a value the slider takes";
/// A side whose blocks do not fit ([`Words::DoesNotFit`]): the neighbour lines' shipped words,
/// moved here; the Size tab's exploration line is built from it.
pub const SIDE_DOES_NOT_FIT: &str = "magnets do not fit";
/// A side the model has no prediction for ([`Words::NoPrediction`]): the neighbour lines'
/// shipped words, moved here (named apart from `size_controls::NO_PREDICTION`).
pub const SIDE_NO_PREDICTION: &str = "no prediction";
/// A side whose keyway leaves no hub wall past it ([`Words::KeywayBreaksThrough`]; T-4): the
/// shipped words of the Size tab's exploration line, which is built from it.
pub const KEYWAY_BREAKS_THROUGH: &str = "the keyway breaks through the hub";
/// A length side that is not a step of the overlapping magnet length ([`Words::Jump`]; owner
/// 9A, widened by question O-1). It names no cause.
pub const JUMP: &str = "not a step: the overlapping magnet length does not change by the step";
/// After a number whose stepped design takes another calibration factor (owner 16A): it names
/// the change, never a cause.
pub const CALIBRATION_CHANGE: &str = "includes a calibration change";
/// The card of a design shown whose magnets do not fit ([`TorqueCard::ShownDoesNotFit`]).
pub const SHOWN_DOES_NOT_FIT: &str =
    "No step is compared while the magnets of the design shown do not fit.";
/// The card of a design shown with no torque prediction ([`TorqueCard::ShownNoPrediction`]: its
/// readout shows `NO_PREDICTION` or `NO_TORQUE`).
pub const SHOWN_NO_PREDICTION: &str =
    "No step is compared while the design shown has no torque prediction.";
/// The card of a design shown whose readouts show a pull-out or hot-low not above zero
/// ([`TorqueCard::ShownNoHotLow`]): true for either torque.
pub const SHOWN_NO_HOT_LOW: &str = "No step is compared while the design shown has a pull-out or hot-low torque that is not above zero.";
/// A flip's words, drawn "{label}: {word}" ([`Flip`]): a phrase inside a sentence, not the
/// dashboard badge's own words (`Level::word`).
pub const FLIP_FAILS: &str = "fails";
pub const FLIP_PASSES: &str = "passes";
pub const FLIP_CAUTION: &str = "caution";
```

(d) `Control`'s unit and step text, after `signed_step`:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 148):
```rust
    /// The step on the `+` side (`more`) or the `-` side, with its sign.
    fn signed_step(self, more: bool) -> f64 {
        if more { self.step() } else { -self.step() }
    }
}
```

New:
```rust
    /// The step on the `+` side (`more`) or the `-` side, with its sign.
    fn signed_step(self, more: bool) -> f64 {
        if more { self.step() } else { -self.step() }
    }

    /// The unit of the control's step: millimetres, or none for the magnet count.
    fn unit(self) -> &'static str {
        match self {
            Control::OutsideDiameter | Control::OverallLength | Control::FaceGap => "mm",
            Control::MagnetsPerRing => "",
        }
    }

    /// A side's step as its row writes it (spec 4.1): "-1 mm", "+0.1 mm", "-2".
    fn step_text(self, more: bool) -> String {
        with_unit(format!("{:+}", self.signed_step(more)), self.unit())
    }
}
```

(e) `Words::text`, after the enum:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 182):
```rust
    /// The overlapping magnet length does not change by the step (the Overall length row).
    Jump,
}
```

New:
```rust
    /// The overlapping magnet length does not change by the step (the Overall length row).
    Jump,
}

impl Words {
    /// The words the side shows.
    pub fn text(self) -> &'static str {
        match self {
            Words::EndOfRange => END_OF_RANGE,
            Words::NotOnTheSlider => NOT_ON_THE_SLIDER,
            Words::DoesNotFit => SIDE_DOES_NOT_FIT,
            Words::KeywayBreaksThrough => KEYWAY_BREAKS_THROUGH,
            Words::NoPrediction => SIDE_NO_PREDICTION,
            Words::Jump => JUMP,
        }
    }
}
```

(f) `Flip::text`, after the struct:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 205):
```rust
    /// Its level on the stepped design.
    pub to: Level,
}
```

New:
```rust
    /// Its level on the stepped design.
    pub to: Level,
}

impl Flip {
    /// "{label}: fails", "passes" or "caution" (owner 8A).
    fn text(&self) -> String {
        let word = match self.to {
            Level::Bad => FLIP_FAILS,
            Level::Good => FLIP_PASSES,
            Level::Caution => FLIP_CAUTION,
        };
        format!("{}: {word}", self.check)
    }
}
```

(g) `MarginChange::text`, after the struct:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 232):
```rust
    /// The decimals the change starts from (section 5's rule).
    pub decimals: usize,
}
```

New:
```rust
    /// The decimals the change starts from (section 5's rule).
    pub decimals: usize,
}

impl MarginChange {
    /// "{label} {change} {unit}", or "{label}: {name} {change} {unit}" where it has a name
    /// (spec 3.6), the change at its own decimals ([`signed_change`]).
    fn text(&self) -> String {
        let change = with_unit(signed_change(self.change, self.decimals), self.unit);
        match self.name {
            None => format!("{} {change}", self.check),
            Some(name) => format!("{}: {name} {change}", self.check),
        }
    }
}
```

(h) `Change::costs`, after the struct:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 268):
```rust
    /// The failing checks' quantities that move and show (spec 3.6).
    pub margins: Vec<MarginChange>,
}
```

New:
```rust
    /// The failing checks' quantities that move and show (spec 3.6).
    pub margins: Vec<MarginChange>,
}

impl Change {
    /// What the step costs, in the order of spec 3.6: the calibration note, the size changes
    /// (Outside diameter, then Overall length: never the row's own `control`, and only where
    /// one shows, [`shows`]), the flips, then the failing checks' quantities.
    fn costs(&self, control: Control) -> Vec<String> {
        let mut costs = Vec::new();
        if self.calibration_changed {
            costs.push(CALIBRATION_CHANGE.to_owned());
        }
        for (size, change) in [
            (Control::OutsideDiameter, self.diameter_change_mm),
            (Control::OverallLength, self.length_change_mm),
        ] {
            if let Some(change) =
                change.filter(|&change| size != control && shows(change, SIZE_DECIMALS))
            {
                let change = with_unit(signed_change(change, SIZE_DECIMALS), size.unit());
                costs.push(format!("{} {change}", size.label()));
            }
        }
        costs.extend(self.flips.iter().map(Flip::text));
        costs.extend(self.margins.iter().map(MarginChange::text));
        costs
    }
}
```

(i) `Row`: the decimals of its two N·m (T-14), its line and each side's text:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 305):
```rust
/// One control's row: its `-` side, then its `+` side.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub control: Control,
    pub fewer: Side,
    pub more: Side,
}
```

New:
```rust
/// One control's row: its `-` side, then its `+` side.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub control: Control,
    pub fewer: Side,
    pub more: Side,
}

impl Row {
    /// The decimals of the row's two N·m (T-14: one precision for both, the larger of their
    /// own, [`shared_change_decimals`]); a row of words draws none.
    fn torque_decimals(&self) -> usize {
        let changes: Vec<f64> = [&self.fewer, &self.more]
            .into_iter()
            .filter_map(|side| match side {
                Side::Change(change) => Some(change.change_Nm),
                Side::Words(_) => None,
            })
            .collect();
        shared_change_decimals(&changes, TORQUE_DECIMALS).unwrap_or(TORQUE_DECIMALS)
    }

    /// "{label}: {fewer}; {more}" (spec 4.1), the two sides' N·m at the row's one precision.
    fn line(&self) -> String {
        let decimals = self.torque_decimals();
        format!(
            "{}: {}; {}",
            self.control.label(),
            self.side_text(&self.fewer, false, decimals),
            self.side_text(&self.more, true, decimals)
        )
    }

    /// One side (spec 4.1): "{step} -> {words}", or "{step} -> {N·m} N·m ({percent})" with the
    /// N·m at the row's `decimals`, then ", " and each cost.
    fn side_text(&self, side: &Side, more: bool, decimals: usize) -> String {
        let step = self.control.step_text(more);
        match side {
            Side::Words(words) => format!("{step} -> {}", words.text()),
            Side::Change(change) => {
                let torque = with_unit(signed_fixed(change.change_Nm, decimals), "N·m");
                let number = format!("{step} -> {torque} ({})", signed_percent(change.fraction));
                std::iter::once(number)
                    .chain(change.costs(self.control))
                    .collect::<Vec<_>>()
                    .join(", ")
            }
        }
    }
}
```

(j) `TorqueCard::lines` before `rows`; `held_line` and `auto_line` after the impl:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 368):
```rust
impl TorqueCard {
    /// The rows: empty for the three one-line forms.
    pub fn rows(&self) -> &[Row] {
        match self {
            TorqueCard::Rows(rows) => rows,
            TorqueCard::ShownDoesNotFit
            | TorqueCard::ShownNoPrediction
            | TorqueCard::ShownNoHotLow => &[],
        }
    }
}
```

New:
```rust
impl TorqueCard {
    /// The lines under the heading, as drawn and as the screen dump prints them. For the rows:
    /// [`CAPTION`], the Auto line while Auto is on (`auto` is its variable), then one line per
    /// row, the row Auto holds reading [`held_line`] (owner 5A; spec 3.4: the marking is drawn
    /// here, outside the card the cache keeps). For the three one-line forms, their line alone,
    /// with no caption and no Auto line (T-12).
    pub fn lines(&self, auto: Option<FreeVariable>) -> Vec<String> {
        match self {
            TorqueCard::ShownDoesNotFit => vec![SHOWN_DOES_NOT_FIT.to_owned()],
            TorqueCard::ShownNoPrediction => vec![SHOWN_NO_PREDICTION.to_owned()],
            TorqueCard::ShownNoHotLow => vec![SHOWN_NO_HOT_LOW.to_owned()],
            TorqueCard::Rows(rows) => {
                let mut lines = vec![CAPTION.to_owned()];
                lines.extend(auto.map(|variable| auto_line(variable).to_owned()));
                lines.extend(rows.iter().map(|row| {
                    if auto.is_some() && row.control.held_by() == auto {
                        held_line(row.control)
                    } else {
                        row.line()
                    }
                }));
                lines
            }
        }
    }

    /// The rows: empty for the three one-line forms.
    pub fn rows(&self) -> &[Row] {
        match self {
            TorqueCard::Rows(rows) => rows,
            TorqueCard::ShownDoesNotFit
            | TorqueCard::ShownNoPrediction
            | TorqueCard::ShownNoHotLow => &[],
        }
    }
}

/// The held row's line, "{label}: {SET_BY_AUTO}" (owner 5A): [`TorqueCard::lines`] draws it,
/// and `app.rs`'s smoke test asserts it in the gui-smoke prompt (`size_controls` is a private
/// module, so `app.rs` reaches the words only here).
pub fn held_line(control: Control) -> String {
    format!("{}: {SET_BY_AUTO}", control.label())
}

/// The Auto line while Auto is on `variable` (owner 5A).
fn auto_line(variable: FreeVariable) -> &'static str {
    match variable {
        FreeVariable::AxialLength => AUTO_HOLDS_LENGTH,
        FreeVariable::RingRadius => AUTO_HOLDS_RING,
        FreeVariable::MagnetsPerRing => AUTO_HOLDS_COUNT,
    }
}
```

(k) `card` reads the pause, so its `expect` goes:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 428):
```rust
    minimum_diameter_mm: Option<f64>,
    #[expect(
        unused_variables,
        reason = "the flips and margins read it (slice 3 Task 4, which removes this)"
    )]
    clamps_paused: bool,
) -> TorqueCard {
```

New:
```rust
    minimum_diameter_mm: Option<f64>,
    clamps_paused: bool,
) -> TorqueCard {
```

(l) `card` hands the pause to the design shown:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 441):
```rust
    let shown = Shown {
        inputs,
        results,
        size: dimensions(inputs),
    };
```

New:
```rust
    let shown = Shown {
        inputs,
        results,
        size: dimensions(inputs),
        clamps_paused,
    };
```

(m) `shows`, before `face_gap_range`:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 529):
```rust
/// The Magnet face gap slider's range: its input's, as the Size tab draws it.
```

New:
```rust
/// Whether a size or margin change is drawn (spec 3.6): it is finite and reads a digit that is
/// not zero at `from` + [`CHANGE_EXTRA_DECIMALS`] decimals, the cap of section 5's format. So
/// float noise (3.6e-15 mm) is never drawn as "+0.0 mm", nor is a NaN or an infinite change (a
/// key with no bearing area, +inf against +inf).
fn shows(change: f64, from: usize) -> bool {
    change.is_finite() && reads_not_zero(&fixed(change, from + CHANGE_EXTRA_DECIMALS))
}

/// The Magnet face gap slider's range: its input's, as the Size tab draws it.
```

(n) `Shown` holds the pause:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 545):
```rust
/// The design shown, its results and its size controls ([`dimensions`]: `None` where the Size
/// tab draws no size slider): what each side is compared with.
struct Shown<'a> {
    inputs: &'a DesignInputs,
    results: &'a DesignResults,
    size: Option<SizeControls>,
}
```

New:
```rust
/// The design shown, its results and its size controls ([`dimensions`]: `None` where the Size
/// tab draws no size slider): what each side is compared with, over the guides that count at
/// the shaft-clamp pause `clamps_paused`.
struct Shown<'a> {
    inputs: &'a DesignInputs,
    results: &'a DesignResults,
    size: Option<SizeControls>,
    clamps_paused: bool,
}
```

(o) `Shown::change` builds the flips and the margins; `quoted_changes` follows it:

Old (`magcoupling-rs/src/gui/torque_card.rs`, from line 679):
```rust
        Change {
            hot_low_Nm,
            change_Nm,
            fraction: change_Nm / shown_hot_low_Nm,
            calibration_changed: stepped.model.f_cal != self.results.model.f_cal,
            diameter_change_mm,
            length_change_mm,
            flips: Vec::new(),
            margins: Vec::new(),
        }
    }
}
```

New:
```rust
        // Each counted check whose level differs, and of each check that fails on both designs
        // the quantities of its failing terms (spec 3.6).
        let mut flips = Vec::new();
        let mut margins = Vec::new();
        for guide in counted_guides(self.clamps_paused) {
            let before = level(guide, self.inputs, self.results);
            let after = level(guide, design, stepped);
            if before != after {
                flips.push(Flip {
                    check: guide.label,
                    to: after,
                });
            }
            if before == Level::Bad && after == Level::Bad {
                margins.extend(self.quoted_changes(guide.check, guide.label, design, stepped));
            }
        }
        Change {
            hot_low_Nm,
            change_Nm,
            fraction: change_Nm / shown_hot_low_Nm,
            calibration_changed: stepped.model.f_cal != self.results.model.f_cal,
            diameter_change_mm,
            length_change_mm,
            flips,
            margins,
        }
    }

    /// The quoted quantities of `check` (its guide labelled `label`), a check that fails on the
    /// design shown and on the stepped `design`, whose results are `stepped`: of the terms of
    /// [`check_terms`], paired term by term, each that fails on either design and quotes a
    /// quantity, where its change shows ([`shows`]), in the result's own unit. "Either" catches
    /// a step that puts a second term of a failing check over its limit, which no flip shows.
    fn quoted_changes(
        &self,
        check: &str,
        label: &'static str,
        design: &DesignInputs,
        stepped: &DesignResults,
    ) -> Vec<MarginChange> {
        check_terms(check, self.inputs, self.results)
            .into_iter()
            .zip(check_terms(check, design, stepped))
            .filter(|(before, after)| !(before.passes && after.passes))
            .filter_map(|(before, _)| before.quoted)
            .filter_map(|quoted| {
                let change =
                    result_number(stepped, quoted.path) - result_number(self.results, quoted.path);
                shows(change, quoted.decimals).then(|| MarginChange {
                    check: label,
                    name: quoted.name,
                    change,
                    unit: result_info(quoted.path).map_or("", |info| info.meta.unit),
                    decimals: quoted.decimals,
                })
            })
            .collect()
    }
}
```

- [ ] **Step 4: `gui/guidance.rs`, `gui/format.rs`, `gui/panel.rs`**

(a) `gui/guidance.rs`, the import of the card's words:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 72):
```rust
use crate::gui::torque_card::{Words, count_step, side_words};
```

New:
```rust
use crate::gui::torque_card::{
    SIDE_DOES_NOT_FIT, SIDE_NO_PREDICTION, Words, count_step, side_words,
};
```

The check paths, visible to the crate for the card's tests:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 149):
```rust
// The dashboard checks the guides cover (`DASHBOARD`'s check column; a test compares).
const HOT_MIN: &str = "metal.hot_min_check";
const CLEARANCE: &str = "metal.clearance_check";
const CUP_WALL: &str = "materials.cup_wall_check";
const TEMPERATURE: &str = "temperature.summary.verdict";
const CLAMP: &str = "clamps.recommended";
const SHAFT_JOINT: &str = "shaft_joint.check";
const GEARBOX: &str = "gearbox.cold_high_check";
const SPACE_CLAIM: &str = "housing.space_claim_check";
```

New:
```rust
// The dashboard checks the guides cover (`DASHBOARD`'s check column; a test compares). The
// torque card's tests name the checks it flips and quotes by them.
pub(crate) const HOT_MIN: &str = "metal.hot_min_check";
pub(crate) const CLEARANCE: &str = "metal.clearance_check";
pub(crate) const CUP_WALL: &str = "materials.cup_wall_check";
pub(crate) const TEMPERATURE: &str = "temperature.summary.verdict";
pub(crate) const CLAMP: &str = "clamps.recommended";
pub(crate) const SHAFT_JOINT: &str = "shaft_joint.check";
pub(crate) const GEARBOX: &str = "gearbox.cold_high_check";
pub(crate) const SPACE_CLAIM: &str = "housing.space_claim_check";
```

Test 1a (`the_shared_card_designs_read_as_section_2_1_says`, Task 2's, in the test module of `gui/test_support.rs`) imports them in place of its six copies:

Old (`magcoupling-rs/src/gui/test_support.rs`, from line 954):
```rust
    use crate::gui::guidance::{counted_guides, level};
    use crate::gui::size_controls::predicts;
    use crate::{DesignInputs, compute_all};

    // The counted checks the facts name, by result path.
    const HOT_MIN: &str = "metal.hot_min_check";
    const CLEARANCE: &str = "metal.clearance_check";
    const CUP_WALL: &str = "materials.cup_wall_check";
    const TEMPERATURE: &str = "temperature.summary.verdict";
    const SHAFT_JOINT: &str = "shaft_joint.check";
    const GEARBOX: &str = "gearbox.cold_high_check";
```

New:
```rust
    use crate::gui::guidance::{
        CLEARANCE, CUP_WALL, GEARBOX, HOT_MIN, SHAFT_JOINT, TEMPERATURE, counted_guides, level,
    };
    use crate::gui::size_controls::predicts;
    use crate::{DesignInputs, compute_all};
```

`check_terms` has its caller now (the card's margins), so its attribute goes:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 733):
```rust
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the torque card's margins call it (slice 3 Task 4, which removes this)"
    )
)]
pub(crate) fn check_terms(
```

New:
```rust
pub(crate) fn check_terms(
```

`result_number`, the card's reader of a quoted quantity:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 1312):
```rust
/// The result at `path` as a number ([`number`]); NaN for a text, no value or an unknown path.
fn result_number(results: &DesignResults, path: &str) -> f64 {
```

New:
```rust
/// The result at `path` as a number ([`number`]); NaN for a text, no value or an unknown path.
/// The torque card reads its quoted quantities with it.
pub(crate) fn result_number(results: &DesignResults, path: &str) -> f64 {
```

The two neighbour words leave, moved into the card as `SIDE_DOES_NOT_FIT` and `SIDE_NO_PREDICTION`:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 2906):
```rust
/// A neighbour line's words when its blocks do not fit.
const NEIGHBOUR_DOES_NOT_FIT: &str = "magnets do not fit";

/// A neighbour line's words when the model has no number there.
const NEIGHBOUR_NO_PREDICTION: &str = "no prediction";

/// What two magnets fewer or two more would do (spec 3.3).
```

New (only the last line of Old stays, as context; the lines before it go):
```rust
/// What two magnets fewer or two more would do (spec 3.3).
```

`Neighbour::line` reads the card's words:

Old (`magcoupling-rs/src/gui/guidance.rs`, from line 2955):
```rust
            NeighbourOutcome::DoesNotFit => format!("{at} {NEIGHBOUR_DOES_NOT_FIT}"),
            NeighbourOutcome::NoPrediction => format!("{at} {NEIGHBOUR_NO_PREDICTION}"),
```

New:
```rust
            NeighbourOutcome::DoesNotFit => format!("{at} {SIDE_DOES_NOT_FIT}"),
            NeighbourOutcome::NoPrediction => format!("{at} {SIDE_NO_PREDICTION}"),
```

(b) `gui/format.rs`: `shared_change_decimals` has its caller now (`Row::torque_decimals`):

Old (`magcoupling-rs/src/gui/format.rs`, from line 171):
```rust
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the torque card's lines call it (slice 3 Task 4, which removes this)"
    )
)]
pub(crate) fn shared_change_decimals(changes: &[f64], from: usize) -> Option<usize> {
```

New:
```rust
pub(crate) fn shared_change_decimals(changes: &[f64], from: usize) -> Option<usize> {
```

(c) `gui/panel.rs`, the import:

Old (`magcoupling-rs/src/gui/panel.rs`, from line 65):
```rust
use crate::gui::spreadsheet::Snapshot;
```

New:
```rust
use crate::gui::spreadsheet::Snapshot;
use crate::gui::torque_card::{KEYWAY_BREAKS_THROUGH, SIDE_DOES_NOT_FIT};
```

`EXPLORATION_ONLY`, after `ASSUMPTIONS_NOTE`:

Old (`magcoupling-rs/src/gui/panel.rs`, from line 158):
```rust
/// The line at the top of the Assumptions view.
pub const ASSUMPTIONS_NOTE: &str = "The model's assumptions, apart from the design inputs (each \
     also stays in its input group). The equation panel tags them and what they flow into.";
```

New:
```rust
/// The line at the top of the Assumptions view.
pub const ASSUMPTIONS_NOTE: &str = "The model's assumptions, apart from the design inputs (each \
     also stays in its input group). The equation panel tags them and what they flow into.";

/// The start of the Size tab's exploration lines: a design the checks rule out (its magnets do
/// not fit, or the keyway breaks through the hub) is still drawn, for exploring. Each line ends
/// in the torque card's side words ([`SIDE_DOES_NOT_FIT`], [`KEYWAY_BREAKS_THROUGH`]), so a
/// change of those words reaches both (slice 3 spec 4.2).
const EXPLORATION_ONLY: &str = "Exploration only: ";
```

The two exploration lines in `view_ui`, built from the card's side words:

Old (`magcoupling-rs/src/gui/panel.rs`, from line 915):
```rust
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        "Exploration only: magnets do not fit at this diameter.",
                    );
                }
                let key_wall = self.results.model.hub_wall_past_key_mm;
                if !key_wall.is_finite() || key_wall <= 0.0 {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        "Exploration only: the keyway breaks through the hub.",
                    );
```

New:
```rust
                if !crate::engine::model::blocks_fit(&shown.coupling, &self.results.model) {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter."),
                    );
                }
                let key_wall = self.results.model.hub_wall_past_key_mm;
                if !key_wall.is_finite() || key_wall <= 0.0 {
                    ui.colored_label(
                        ui.visuals().error_fg_color,
                        format!("{EXPLORATION_ONLY}{KEYWAY_BREAKS_THROUGH}."),
                    );
```

- [ ] **Step 5: Run the tests and see them pass; plant fourteen faults**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::torque_card smaller_exploration_shows_magnet_fit_and_keyway_breakthrough`

Expected: `test result: ok. 24 passed; 0 failed; 0 ignored; 0 measured; 939 filtered out` (the card's 23 and the panel's one; about 1.4 s).

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- each_neighbour_is_the_design_with_only_the_magnet_count_changed the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown a_keyway_through_the_hub_changes_no_neighbour_s_reading with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented with_magnets_too_short_for_the_model_every_neighbour_reads_words`

Expected: `test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 956 filtered out`. The neighbour lines read the card's words, unchanged.

Each planted fault below is one edit; each Old is unique in its file. G1 to G11, G13 and G14 edit `magcoupling-rs/src/gui/torque_card.rs` and run with `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- gui::torque_card`. G12 edits `magcoupling-rs/src/gui/panel.rs` and runs with `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- smaller_exploration_shows_magnet_fit_and_keyway_breakthrough`. Undo each by the reverse edit (both files hold this task's uncommitted work, so never `git checkout` them); the tests pass again after each:

G1. Each N·m at its own decimals (no T-14).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                let torque = with_unit(signed_fixed(change.change_Nm, decimals), "N·m");
```

New:
```rust
                let _ = decimals;
                let torque = with_unit(signed_change(change.change_Nm, TORQUE_DECIMALS), "N·m");
```

Expected: 2 fail (`21 passed; 2 failed`): `the_card_s_lines_read_as_drafted` at (e)'s diameter row, each N·m at its own decimals: `left: "Outside diameter: -1 mm -> -0.01 N·m (-0.4 %), Running clearance -0.01 mm, Cup wall: wall needed -0.08 mm; +1 mm -> -0.005 N·m (-0.3 %), ..."`; and `a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required`: `left: ("-0.005", "-0.3")`, `right: ("-0.01", "-0.3")` (the two sides no longer read alike).

G2. Margins of terms failing on the design shown only.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
            .filter(|(before, after)| !(before.passes && after.passes))
```

New:
```rust
            .filter(|(before, _)| !before.passes)
```

Expected: 3 fail (`20 passed; 3 failed`): `the_card_reads_the_spec_s_numbers` at (b)'s diameter `+1 mm` side, with no Rotating diameter quoted; `the_flips_and_the_margins_name_the_checks_that_move` (`left: 511`, `right: 517`); and `the_card_s_strings_are_constants_and_its_numbers_come_from_results` (`left: 1043`, `right: 1049`).

G3. A change drawn whenever it is not exactly zero.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
    change.is_finite() && reads_not_zero(&fixed(change, from + CHANGE_EXTRA_DECIMALS))
```

New:
```rust
    let _ = (from, CHANGE_EXTRA_DECIMALS, reads_not_zero(""));
    change != 0.0
```

Expected: 2 fail: `size_changes_and_margins_are_drawn_only_when_they_show` panics `0.000000000000001 from 1`, and test 7 reads `left: 524`, `right: 517`.

G4. The flips at the shipped pause, not the panel's.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        for guide in counted_guides(self.clamps_paused) {
```

New:
```rust
        for guide in counted_guides(crate::engine::clamps::PAUSED) {
```

Expected: 1 fails: `the_card_names_the_clamp_only_while_unpaused`: the unpaused card's Overall length `+1 mm` side has no clamp flip (the panic prints its `Change`).

G5. The flips before the size changes.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        costs.extend(self.flips.iter().map(Flip::text));
        costs.extend(self.margins.iter().map(MarginChange::text));
        costs

```

New:
```rust
        costs.extend(self.margins.iter().map(MarginChange::text));
        costs.insert(0, String::new());
        costs.remove(0);
        let mut flips: Vec<String> = self.flips.iter().map(Flip::text).collect();
        flips.extend(costs);
        flips

```

Expected: 2 fail: test 2a at (e)'s count row (`left: "Magnets per ring: -2 -> -0.89 N·m (-53.0 %), Space claim: fails, Outside diameter +2.2 mm, ..."`), and test 3 at the hot-low-0 side's drawn text.

G6. A row's own dimension drawn as a size change.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                change.filter(|&change| size != control && shows(change, SIZE_DECIMALS))
```

New:
```rust
                change.filter(|&change| (size != control || size == control) && shows(change, SIZE_DECIMALS))
```

Expected: 3 fail: test 9a ((a)'s diameter row draws its own size change), test 2a (`... +1 mm -> +0.08 N·m (+3.5 %), Outside diameter +1.0 mm, ...`) and test 18 (`left: 1175`, `right: 1049`).

G7. The held row left as numbers.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                    if auto.is_some() && row.control.held_by() == auto {
```

New:
```rust
                    if auto.is_some() && row.control.held_by() == auto && false {
```

Expected: 3 fail: test 8 at `(a) AxialLength`; test 2a, where (b)'s held row is drawn as numbers (`left: "Overall length: -1 mm -> -0.19 N·m (-7.8 %), ..."`); and test 18 (`left: 1096`, `right: 1049`).

G8. The one-line form with the caption.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
            TorqueCard::ShownNoHotLow => vec![SHOWN_NO_HOT_LOW.to_owned()],
```

New:
```rust
            TorqueCard::ShownNoHotLow => vec![CAPTION.to_owned(), SHOWN_NO_HOT_LOW.to_owned()],
```

Expected: 2 fail: test 4 at `variation 1.0 None` (`left` holds the caption, then `SHOWN_NO_HOT_LOW`) and test 16 (`left: 1068`, `right: 1056`).

G9. A quoted change in a typed unit.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
                    unit: result_info(quoted.path).map_or("", |info| info.meta.unit),
```

New:
```rust
                    unit: "mm",
```

Expected: 2 fail: test 2a ((a)'s gearbox change drawn in mm) and test 7 at `(a) OutsideDiameter true: MarginChange { check: "Gearbox input torque", ..., unit: "mm", ... }`.

G10. A number typed in a text.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
pub const CALIBRATION_CHANGE: &str = "includes a calibration change";
```

New:
```rust
pub const CALIBRATION_CHANGE: &str = "includes a calibration change of 9 %";
```

Expected: 1 fails: test 18 at (c)'s line, `assertion `left != right` failed` with `left: "9"`: the typed number does not move with the card's numbers.

G11. A ranking word in the caption.

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
    "The change in the model's hot-low torque after one step of one control, everything else held.";
```

New:
```rust
    "The change in the model's hot-low torque after one step of one control, the most telling, everything else held.";
```

Expected: 1 fails: test 16 at `(a) None`: `left: ["most"]`, `right: []`.

G12. The exploration line typed apart from the side words.

Old (`magcoupling-rs/src/gui/panel.rs`):
```rust
                        format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter."),
```

New:
```rust
                        format!("{EXPLORATION_ONLY}the magnets do not fit at this diameter."),
```

Expected: 1 fails (`0 passed; 1 failed; ... 962 filtered out`): `smaller_exploration_shows_magnet_fit_and_keyway_breakthrough` reads `left: 0`, `right: 1`.

G13. A sanity bound on the change, which a design file's values break (plan review 1.1).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
        Change {
            hot_low_Nm,
            change_Nm,
```

New:
```rust
        assert!(
            (change_Nm / shown_hot_low_Nm).abs() < 1e6,
            "no step moves hot-low a millionfold"
        );
        Change {
            hot_low_Nm,
            change_Nm,
```

Expected: 1 fails (`22 passed; 1 failed`): `the_card_never_panics_at_the_values_a_design_file_can_carry` panics `no step moves hot-low a millionfold`.

G14. A stepped design without size controls read as a size change of zero (plan review 2.1).

Old (`magcoupling-rs/src/gui/torque_card.rs`):
```rust
            _ => (None, None),
```

New:
```rust
            (Some(_), None) => (Some(0.0), Some(0.0)),
            _ => (None, None),
```

Expected: 1 fails (`22 passed; 1 failed`): `a_step_to_a_design_without_size_controls_draws_no_size_change`: `left: (Some(0.0), Some(0.0))`, `right: (None, None)`.

- [ ] **Step 6: Docs: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`**

(a) README, the Layout table's `src/gui/` row, Task 3's `torque_card.rs` entry: what a step costs, `lines`, `held_line` and the texts:

Old (`magcoupling-rs/README.md`, from line 155):
```text
and whether the calibration factor changes); `side_words`
```

New:
```text
whether the calibration factor changes, the counted checks whose level flips (`Flip`, over `counted_guides` at the panel's shaft-clamp pause) and the failing checks' quantities the step moves (`MarginChange`, from `guidance::check_terms`, drawn where `shows`)); `lines(auto)` draws it, a row's two N·m at one precision and the row Auto holds as `held_line`, every text a named constant (`CARD_HEADING`, `CAPTION`, the three Auto lines, the side words, `JUMP`, `CALIBRATION_CHANGE`, the three one-line forms and the flip words; `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` also end the Size tab's exploration lines); `side_words`
```

(b) README, the Tests table's `src/gui/torque_card.rs` row: the table's Auto state and its two new designs, then this task's tests at the row's end:

Old (`magcoupling-rs/README.md`, from line 698):
```text
(every design of its spec's section 2.1 with its Minimum OD preference, an Auto
```

New:
```text
(every design of its spec's section 2.1, T-14's one-flip design and R-13's boss plateau, with its Minimum OD preference and Auto state, an Auto
```

Old (`magcoupling-rs/README.md`, from line 698):
```text
(`a_side_reads_a_number_exactly_when_sizing_counts_its_design`). |
```

New:
```text
(`a_side_reads_a_number_exactly_when_sizing_counts_its_design`); the spec's numbers, and whole lines built from the constants so a reworded text flows through (`the_card_reads_the_spec_s_numbers`, `the_card_s_lines_read_as_drafted`); the flips are exactly the counted checks whose level differs, and each quoted change is of a check that fails on both designs, never the hot-low or the temperature check, in its result's own unit (`the_flips_and_the_margins_name_the_checks_that_move`), the clamp named only while unpaused (`the_card_names_the_clamp_only_while_unpaused`); the held row and the Auto line, drawn outside the card (`the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s`); size changes and margins drawn only when they show (`size_changes_and_margins_are_drawn_only_when_they_show`); a row whose two sides read alike flips Hot-low torque only on the side whose hot-low is below Required (T-14, the table's "T-14 one flip": `a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required`); where the boss sets the Outside diameter (the table's "R-13 boss plateau") -1 mm reads the end of its range and +1 mm the slider's own lift, +7.55 % (`where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift`); a step to a design without size controls (a design file's 6 magnets on a 38 mm ring, two fewer) draws its number and no size change (`a_step_to_a_design_without_size_controls_draws_no_size_change`); no number reads zero or disagrees with its percent (`no_card_number_reads_zero_or_disagrees_with_its_percent`); no card text ranks the controls, orders the torques, says the coupling protects the gearbox or names the clamp, and every one passes the plain-words guard (`the_card_texts_never_rank_the_controls`) and has its glyphs (`every_card_text_has_its_glyphs`); at finite values a design file or link can carry past every slider (a count from 0 to 1e6 and i64's ends; gaps, overrides, rings, cup depths, bosses, caps and bores at 0, negative or 1e300; a variation, magnet Br or calibration coefficient near 0 or at 1e300), at both back irons and at the Minimum OD's top, the card is drawn without a panic, equals a second card of the same design, and every line of every Auto state passes the plain-words guard and has its glyphs (`the_card_never_panics_at_the_values_a_design_file_can_carry`); every number is read from the card and every other word is the template's (`the_card_s_strings_are_constants_and_its_numbers_come_from_results`). |
```

(b2) The Tests table's `src/gui/test_support.rs` row: test 1a pins the boss plateau.

Old (`magcoupling-rs/README.md`, from line 701):
```text
the variation allowance at 1.0 and 1.5 (hot-low 0 and -1.3442 N·m) |
```

New:
```text
the variation allowance at 1.0 and 1.5 (hot-low 0 and -1.3442 N·m), and the boss plateau (R-13: the magnets fit and the boss sets the Outside diameter at 40.00 mm, the bottom of its slider's 40.00 to 85.41 mm) |
```

(c) `docs/ai/03-structure.yaml`, the `gui` line, Task 3's `torque_card.rs` clause:

Old (`docs/ai/03-structure.yaml`, from line 68):
```text
side_words and count_step moved from the N-2 and N+2 lines, which call them), sizing.rs (SizingMode,
```

New:
```text
side_words and count_step moved from the N-2 and N+2 lines, which call them; each change with its calibration note, its flips over counted_guides at the panel's shaft-clamp pause and the failing checks' quantities check_terms quotes, drawn where shows; held_line; lines(auto), a row's two N·m at one precision (T-14); every text a named constant: CARD_HEADING, CAPTION, the three Auto lines, END_OF_RANGE, NOT_ON_THE_SLIDER, SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION (the neighbour lines' words, moved), KEYWAY_BREAKS_THROUGH (SIDE_DOES_NOT_FIT and it end the Size tab's exploration lines, panel.rs EXPLORATION_ONLY), JUMP, CALIBRATION_CHANGE, the three one-line forms and the flip words; WALL_NEEDED and COLD_HIGH are guidance.rs's), sizing.rs (SizingMode,
```

(d) The same line, the `card_designs()` clause:

Old (`docs/ai/03-structure.yaml`, from line 68):
```text
CardDesign {key, inputs, minimum_diameter_mm} with card() and side(), card_at, side_at: every design of spec 2.1 and its Edges at its Minimum OD preference, an Auto
```

New:
```text
CardDesign {key, inputs, minimum_diameter_mm, auto} with card() and side(), card_at, side_at: every design of spec 2.1 and its Edges, T-14's one-flip design and R-13's boss plateau, at its Minimum OD preference and Auto state, an Auto
```

(d2) The same line, `test_support.rs`'s designs: `boss_plateau`.

Old (`docs/ai/03-structure.yaml`, from line 68):
```text
with_variation(variation) and smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap), and their one table
```

New:
```text
with_variation(variation), smoke_inputs (the gui-smoke share link's inputs before Auto solves them: the defaults with a 1.5 mm face gap) and boss_plateau (R-13: the shaft boss setting the Outside diameter), and their one table
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); print('ok')"`

Expected: `ok`.

- [ ] **Step 7: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: every binary as after Task 3 (lib `234 passed; 0 failed; 1 ignored`; the table in Task 3's Step 7).

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `957 passed; 0 failed; 6 ignored` (943 after Task 3, plus fourteen new tests); the two app binaries run 0 tests; `explain` 64 passed, 1 ignored; every other binary as after Task 3.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`, then the same without `--features app`.

Expected: `Finished`, no warning, both times.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/gui/format.rs
 M magcoupling-rs/src/gui/guidance.rs
 M magcoupling-rs/src/gui/panel.rs
 M magcoupling-rs/src/gui/test_support.rs
 M magcoupling-rs/src/gui/torque_card.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
```

- [ ] **Step 8: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T4.txt` with the Write tool:
```text
feat(magcoupler): what a torque card step costs, and the card's lines (BL-073 T4)

gui/torque_card.rs, slice 3 spec 3.4, 3.6, 4.1, 4.2 and 5 (T-14). A number side
now carries the counted checks whose level flips (over counted_guides at the
panel's shaft-clamp pause) and the failing checks' quantities the step moves:
of each check that fails on both designs, each check_terms term that fails on
either and quotes a quantity, where its change shows (finite and a digit at two
decimals past its own, the format's cap), in the result's own unit. Never the
hot-low (the row's own number) nor the temperature check (T-13: its terms quote
nothing).

TorqueCard::lines draws the card: the caption, the Auto line and held_line for
the row Auto holds (owner 5A, outside the cached card), then each row as
"{label}: {fewer}; {more}", a side "{step} -> {N·m} N·m ({percent})" then its
calibration note, size changes (never the row's own), flips and quoted changes;
a row's two N·m at one precision (T-14, shared_change_decimals); the three
one-line forms alone (T-12). Every text is a named constant (spec 4.1 drafts,
set by the G-3 review in T7). The neighbour lines' side words move here as
SIDE_DOES_NOT_FIT and SIDE_NO_PREDICTION; the Size tab's exploration lines are
built from SIDE_DOES_NOT_FIT and KEYWAY_BREAKS_THROUGH with panel.rs's
EXPLORATION_ONLY, words unchanged. check_terms and shared_change_decimals have
their caller: their expect(dead_code) goes. guidance's result_number and its
check paths are visible to the crate, so the card reads results and its tests,
and the shared designs' test, name checks without retyping them. The card's
N·m start from gui::format's TORQUE_DECIMALS, as the readout's gap does.

Tests (spec 8 tests 2, 2a, 7, 7a, 8, 9a, 15, 16, 17, 18; tests 3 and 4 extended
to the drawn line), red first; card_designs() in gui/test_support.rs gains each
design's Auto state, and T-14's one-flip design and R-13's boss plateau
(test_support::boss_plateau, pinned by test 1a), which the tests of those two
STOP-sheet cases read by key. A step to a design without size controls draws no
size change, and at 47 values a design file can carry the card is drawn without
a panic (plan review 1.1, 2.1, 2.2). The README and docs/ai/03-structure.yaml
follow.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/torque_card.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/format.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/README.md docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T4.txt
git status --short
```

Expected: one commit of 7 files. `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 4 moves** (each confirmed by a run in the plan-time prototype). Task 4 moves no test. It changes four in place and adds four beyond the spec's list:

| Test | Old -> new | Why |
|---|---|---|
| `gui::panel::tests::smaller_exploration_shows_magnet_fit_and_keyway_breakthrough` | stays; its two expected texts are built from `EXPLORATION_ONLY`, `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` in place of the typed literals, with the words unchanged | spec 4.2 and 6.3 (code review 3 S9); planted fault G12 shows that it still bites |
| `gui::torque_card::tests::the_words_follow_the_order_of_the_tests` and `the_design_shown_decides_first` (Task 3's tests 3 and 4) | stay; extended to the drawn line (the hot-low-0 side's text; each one-line form's `lines(auto)` alone for every Auto state) | spec 8.1 tests 3 and 4 name the drawn text, which exists from this task |
| `gui::test_support::tests::the_shared_card_designs_read_as_section_2_1_says` (Task 2's test 1a) | stays; it imports `HOT_MIN`, `CLEARANCE`, `CUP_WALL`, `TEMPERATURE`, `SHAFT_JOINT` and `GEARBOX` from `gui::guidance` in place of its six local copies, its assertions unchanged; its end pins the boss plateau (the magnets fit; the Outside diameter 40.00 mm, the slider's 40.00 to 85.41 mm) | the paths are `pub(crate)` from this task; one definition each (DRY); plan review 2.2 |
| `gui::torque_card::tests::a_row_whose_two_sides_read_alike_flips_hot_low_only_where_it_crosses_required` | new | critique A-21 (RF-4), on the table's "T-14 one flip" (plan review 2.2) |
| `gui::torque_card::tests::where_the_boss_sets_the_outside_diameter_one_step_reads_the_slider_s_own_lift` | new | critique A-21 (RF-4), on the table's "R-13 boss plateau" (plan review 2.2) |
| `gui::torque_card::tests::a_step_to_a_design_without_size_controls_draws_no_size_change` | new | plan review 2.1: spec 2.2's rule on its stepped side, which Task 10 Step 7b keeps |
| `gui::torque_card::tests::the_card_never_panics_at_the_values_a_design_file_can_carry` | new | plan review 1.1 and ruling C-7: the input boundary (a design file's finite values), in the pattern of `tests/robustness.rs` |

The neighbour tests of spec 6.3 run unchanged here, reading `SIDE_DOES_NOT_FIT` and `SIDE_NO_PREDICTION` through `Neighbour::line`. They stay for Task 5, which moves them into the card's test module as the count row's tests (spec 6.3). In this table, "test 3" is `the_words_follow_the_order_of_the_tests`. For Task 5, the tests that already hold each one's point:

| Neighbour test (spec 6.3) | Task | The card's tests that already hold its point |
|---|---|---|
| `each_neighbour_is_the_design_with_only_the_magnet_count_changed` | Task 5 | test 1 (every count side is `compute_all` of `FreeVariable::MagnetsPerRing.apply`, bit for bit); test 2 ((a)'s count row: -0.6121 N·m, `+2` does not fit) |
| `the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not` | Task 5 | test 2a ((a)'s and (e)'s drawn count rows, the cup's growth as "Outside diameter +2.2 mm") |
| `at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown` (with Task 3's odd-count assertion) | Task 5 | test 3 (npole 4 `-2` and npole 40 `+2` read `EndOfRange`; the odd count's two sides `NotOnTheSlider`); `count_step_keeps_even_counts_inside_the_slider_and_never_overflows` |
| `neighbours_are_even_counts_inside_the_slider_and_never_overflow` | moved in Task 3 | `count_step_keeps_even_counts_inside_the_slider_and_never_overflows` |
| `a_keyway_through_the_hub_changes_no_neighbour_s_reading` | Task 5 | test 3 (keyway through: the length and gap rows read `KeywayBreaksThrough`; the count's `-2` side is the number of the keyway intact, bit for bit); `a_side_reads_a_number_exactly_when_sizing_counts_its_design` (D-1) |
| `with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it` | Task 5 | `a_side_reads_a_number_exactly_when_sizing_counts_its_design` |
| `a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented` | Task 5 | test 3 (end effect at a step: the length's `-1 mm` side and the count's `-2` side read `NoPrediction`); test 4 |
| `with_magnets_too_short_for_the_model_every_neighbour_reads_words` | Task 5 | test 4 (short magnets: `ShownNoPrediction`, no rows) |

---

### Task 5: Wiring: the card in the consequences cache and on the Size tab, the neighbour lines off Magnets per ring

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec sections 3.2 (the cache and the two call sites), 3.4 (the Auto marking outside the cache), 3.5 (where the panel asks), 4.3 (the first line under Magnets per ring), 6.1 (placement, the header row, the heading's toggle) and 6.3 (every shipped test that reads the moved lines, the Why-link counts, the helpers that gain `clamps_paused`); tests 10 and 11; owner decisions 4A, 5A, 7A, 12A. One commit, so the Size tab never lacks the N±2 information: the neighbour lines leave Magnets per ring and the card appears in the same commit.

- `Consequences.card: TorqueCard` replaces `neighbours`. `Consequences::of` and `ConsequenceCache::get` gain `clamps_paused`; the cache keeps it in `CachedConsequences` and in its equality test (one key in three parts, as `FixCache` keys on the pause). Both panel call sites pass `self.clamps_paused`.
- `magnet_lines` gives `RING_SIZE_STAYS` (Auto off and Auto on the magnet length; its clause changed per 4.3) and the fit line. `AUTO_RESIZES_RING` and `AUTO_RESETS_LENGTH` are removed; the card's Auto lines (`AUTO_HOLDS_*`, Task 4) take their rule.
- The neighbour code leaves `gui/guidance.rs` (`Neighbour`, `NeighbourOutcome`, `Neighbour::line`, `neighbours`, `prediction`), and with it the imports `end_effect_in_range` and `signed_percent`, and guidance's import of the card's `SIDE_DOES_NOT_FIT`, `SIDE_NO_PREDICTION`, `Words`, `count_step` and `side_words`. Its seven tests move to the card's test module as the count row's tests, rewritten to the card's types and hot-low.
- The Size tab draws the card in place of the label under the readouts: `torque_card_ui`, a `CollapsingState` with id `"magcoupling_torque_card"` (`TORQUE_CARD_ID`), open (Task 6 decides open or closed), its header one `horizontal_wrapped` row inside `show_header`: the heading, a `Label` that senses clicks and toggles the card as its arrow does, then `why_note_link(ui, SIZE_TAB_NOTE)`, whose click asks for the note through `readouts.open_note`. The body draws `card.lines(auto)` as weak wrapped labels (`weak_line`). The card draws the design shown after this frame's edits, which `view_ui` builds once (`self.shown_inputs()`, after the Required slider's edit) and hands to the card and the drawing (plan review 4.4, plan deviation PD-6); the card runs its own `compute_all` of it and asks with `self.minimum_diameter_mm` and `self.clamps_paused`.
- Tests beyond the spec's: a drag of the gap slider with Auto on the magnet length computes the card once per design it draws (critique A-19, RF-2); a typed Required value reaches the card in its own frame (plan review 4.4); test 10 covers the key's third part, the Minimum OD (plan review 3.2). The smoke design's inputs are `test_support::smoke_inputs()` (critique A-24).
- `MagcouplingPanel::auto()` returns the variable Auto sets while it is on. `inputs_ui` and the card share it (one expression, DRY).
- `end_effect_edge()` moves from `gui/guidance.rs`'s tests to `gui/test_support.rs`: guidance's `every_consequence_line_has_its_glyphs_and_plain_words` and the count row's moved no-prediction test both read it.
- `guidance_strings_for_review` gains Task 7's card block (Task 7's interface, verbatim) and the card heading's Why-link place.

Facts the implementer meets on the way:
- The red is a build failure: the tests call the new signatures and fields. Nine planted faults in Step 5 show that the new tests bite.
- Task 4 must have defined, in `gui/torque_card.rs`: `card`, `TorqueCard` (with `lines` and `rows`), `Row`, `Side`, `Change`, `Words`, `Control` (with `label` and `held_by`), `count_step`, `held_line`, and the constants `CARD_HEADING`, `CAPTION`, `AUTO_HOLDS_LENGTH`, `AUTO_HOLDS_RING`, `AUTO_HOLDS_COUNT`, `END_OF_RANGE`, `SIDE_DOES_NOT_FIT` and `SHOWN_NO_PREDICTION`. In `gui/panel.rs` it must have defined `EXPLORATION_ONLY`, and the `use crate::gui::torque_card::{KEYWAY_BREAKS_THROUGH, SIDE_DOES_NOT_FIT};` line that Step 4 extends. In `gui/test_support.rs`: `CardDesign { key, inputs, minimum_diameter_mm, auto }` and `card_designs()`.
- The Old blocks of the neighbour code (Step 3) are Task 4's text: guidance's `torque_card` import names `SIDE_DOES_NOT_FIT` and `SIDE_NO_PREDICTION`, the two `NEIGHBOUR_*` words are gone (Task 4 moved them into the card), `Neighbour::line` reads the card's words, and `neighbours()` calls `count_step` and `side_words` (Task 3).
- Line numbers are after Task 4. Trust the code. Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`. Every code block is rustfmt's layout: replaying this task's Old/New blocks on the commit before it reproduces the task's eight files byte for byte (CRLF normalised; the plan's replay check). Then both suites, both clippy runs and fmt are green.

**Files:**
- Modify: `magcoupling-rs/src/gui/guidance.rs` (the module doc; the imports; `SIZE_TAB_NOTE`'s doc; `RING_SIZE_STAYS`, `magnet_count_line`; the neighbour code removed; `Consequences`, `of`, `magnet_lines`; `ConsequenceCache`), `magcoupling-rs/src/gui/panel.rs` (`TORQUE_CARD_ID`, the `consequences` field's doc, `auto()`, `inputs_ui`'s call, `torque_card_ui`, `view_ui`), `magcoupling-rs/src/gui/size_controls.rs` (the module doc)
- Modify (tests): `magcoupling-rs/src/gui/guidance.rs`, `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/size_controls.rs` (`the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown`'s `Consequences::of`), `magcoupling-rs/src/gui/torque_card.rs` (the count row's moved tests), `magcoupling-rs/src/gui/test_support.rs` (`end_effect_edge`)
- Modify (docs): `magcoupling-rs/README.md` (the `src/gui/` module list's `guidance.rs` and `panel.rs` entries; the Size tab's lines and a paragraph on the card; where the Why links sit; the Tests table's `src/gui/panel.rs`, `src/gui/guidance.rs` and `src/gui/torque_card.rs` rows), `docs/ai/02-system.yaml` (`:115`'s Why links, `:116`'s Size-tab clause), `docs/ai/03-structure.yaml` (the `gui` line's `guidance.rs`, `panel.rs` and `test_support.rs` clauses; `manual_size_gui`)
- Test: `magcoupling-rs/src/gui/panel.rs` (new `the_size_tab_draws_the_torque_card_in_place_of_the_label` and `idle_frames_compute_the_card_once_and_an_edit_again`, test 10; test 11 in `idle_frames_on_the_first_view_change_no_input` and `idle_frames_change_no_input`; the moved-line and Why-link tests of spec 6.3), `magcoupling-rs/src/gui/guidance.rs` (the cache test, the first-line test, the ring-size wording test, `without_a_breakdown_or_a_fit_there_is_no_line_for_it`), `magcoupling-rs/src/gui/torque_card.rs` (seven count-row tests moved from `gui/guidance.rs`)

**Interfaces:**
- Consumes: Task 4's `gui::torque_card::{card, TorqueCard, Row, Side, Words, Control, count_step, held_line, CARD_HEADING, CAPTION, AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT, END_OF_RANGE, SIDE_DOES_NOT_FIT, SHOWN_NO_PREDICTION}`; Task 4's `gui/panel.rs` `EXPLORATION_ONLY`; `gui::test_support::{card_designs, CardDesign, card_at, side_at, odd_count, forty_narrow_magnets, end_effect_at_a_step, short_magnets, with_poles, smoke_inputs}` (Tasks 2-4; `smoke_inputs` is Task 3's); the card test module's `sides`, `number` and `words` (Task 3); `gui::dashboard::why_note_link`; `gui::input_ui::weak_line`; `Readouts::open_note`.
- Produces:
  - `gui::guidance`: `Consequences { pub card: TorqueCard, pub fit, pub fit_unlocked, pub diameter, pub length }`; `Consequences::of(inputs: &DesignInputs, results: &DesignResults, minimum_diameter_mm: Option<f64>, clamps_paused: bool) -> Consequences`; `ConsequenceCache::get(&mut self, inputs, results, minimum_diameter_mm: Option<f64>, clamps_paused: bool) -> &Consequences`; `magnet_count_line(auto)`: `Some(RING_SIZE_STAYS)` for `None` and `Some(AxialLength)`, else `None`; `RING_SIZE_STAYS` = "The ring radius stays the same when you change the magnet count; the cup can still change size." (a draft until Task 7).
  - `gui::panel` (private): `const TORQUE_CARD_ID: &str = "magcoupling_torque_card"`; `fn auto(&self) -> Option<FreeVariable>`; `fn torque_card_ui(&mut self, ui: &mut egui::Ui, readouts: &mut Readouts, shown: &DesignInputs)` (Task 6 adds `card: CardFrame` before `shown`).
  - `gui::test_support::end_effect_edge() -> DesignInputs` (`pub(crate)`).
  - Removed: `gui::guidance::{Neighbour, NeighbourOutcome, AUTO_RESIZES_RING, AUTO_RESETS_LENGTH}`.

- [ ] **Step 1: Write the failing tests**

(a) `magcoupling-rs/src/gui/guidance.rs`, the test module. Its `test_support` import takes `end_effect_edge`, which moves there (b):

Old:
```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, drawn_texts, hot_day, hub_overstressed, is_shaft_joint,
        keyway_through, mixed_rings, n42_parts, only_the_clamp_fails, only_the_gearbox_fails,
        only_the_joint_fails, passing_design, reversed_hub_band, shaft_joint_input_rows,
        shaft_joint_result_rows, short_magnets, sized_frame, slider_range, small_boss, with_poles,
    };
```

New:
```rust
    use crate::gui::test_support::{
        assert_glyphs, clamp_at_its_limit, drawn_texts, end_effect_edge, hot_day, hub_overstressed,
        is_shaft_joint, keyway_through, mixed_rings, n42_parts, only_the_clamp_fails,
        only_the_gearbox_fails, only_the_joint_fails, passing_design, reversed_hub_band,
        shaft_joint_input_rows, shaft_joint_result_rows, short_magnets, sized_frame, slider_range,
        small_boss, with_poles,
    };
```

The links test's comment names the card heading's link:

Old:
```rust
        // Spec 3.7: the Size tab's Required and Hot-low links lead where the hot minimum
        // check's Why does, the banner's where the end-effect check's does, and the links
        // under Magnets per ring and Outside diameter to Start here's first step.
```

New:
```rust
        // Spec 3.7: the Size tab's Required and Hot-low links lead where the hot minimum
        // check's Why does, the banner's where the end-effect check's does, and the links
        // under Magnets per ring and Outside diameter and on the torque card's heading (slice 3
        // spec 6.1) to Start here's first step.
```

`no_guide_text_calls_the_ring_radius_ring_size` lists the card's three Auto lines in place of the two removed constants (spec 4.3: they take their rule):

Old:
```rust
        // One name per quantity (spec 3.5): the input row says Ring radius, so no guide, fix,
        // problem or consequence line says "ring size".
        let mut texts = every_text();
        texts.extend([RING_SIZE_STAYS, AUTO_RESIZES_RING, AUTO_RESETS_LENGTH].map(str::to_owned));
```

New:
```rust
        // One name per quantity (spec 3.5): the input row says Ring radius, so no guide, fix,
        // problem or consequence line says "ring size", nor the torque card's Auto lines, which
        // took over the rule of the two Auto lines that stood under Magnets per ring (slice 3
        // spec 4.3).
        use crate::gui::torque_card::{AUTO_HOLDS_COUNT, AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING};
        let mut texts = every_text();
        texts.extend(
            [
                RING_SIZE_STAYS,
                AUTO_HOLDS_LENGTH,
                AUTO_HOLDS_RING,
                AUTO_HOLDS_COUNT,
            ]
            .map(str::to_owned),
        );
```

The helpers and the neighbour tests. `consequences_of` passes the shipped pause, `clamps::PAUSED` (spec 6.3; 13 callers). `end_effect_edge` moves to `gui/test_support.rs` (b). The seven neighbour tests move to the card's test module (c) as the count row's tests. (`neighbours_are_even_counts_inside_the_slider_and_never_overflow` calls `neighbour_counts`, which Task 3 moves into `count_step`, so it moved with Task 3 as test 3's pin.) Old: from the helpers' doc to the first fit-line test:

Old:
```rust
    /// The consequences of `inputs` as the panel builds them: with the design's own results.
    fn consequences_of(inputs: &DesignInputs, minimum_diameter_mm: Option<f64>) -> Consequences {
        Consequences::of(inputs, &compute_all(inputs), minimum_diameter_mm)
    }

    /// `inputs` with `npole` magnets per ring and nothing else changed.
    fn with_magnets(inputs: &DesignInputs, npole: i64) -> DesignInputs {
        FreeVariable::MagnetsPerRing.apply(inputs, npole as f64)
    }

    /// A design whose own end effect is just out of range while two more magnets bring it back
    /// in and fit: the first ring radius of the solver's grid where two more magnets fit, the
    /// magnets cut halfway between the two counts' end-effect limits (f_end = 1 - c_end * pole
    /// pitch / L is zero at L = c_end * pole pitch, and the pitch falls as the count rises).
    fn end_effect_edge() -> DesignInputs {
        let base = DesignInputs::default();
        let more = base.coupling.npole + 2;
        let radius = FreeVariable::RingRadius
            .grid()
            .into_iter()
            .find(|&radius| {
                let design = with_magnets(&FreeVariable::RingRadius.apply(&base, radius), more);
                blocks_fit(&design.coupling, &compute_all(&design).model)
            })
            .expect("two more magnets fit somewhere on the ring-radius slider");
        let mut design = FreeVariable::RingRadius.apply(&base, radius);
        // The end-effect coefficient's slider maximum: the length lands inside its slider.
        design.coupling.c_end = 0.5;
        let pitch = |npole: i64| {
            compute_all(&with_magnets(&design, npole))
                .model
                .pole_pitch_mm
        };
        let length = design.coupling.c_end * (pitch(base.coupling.npole) + pitch(more)) / 2.0;
        design.coupling.magnets.axial_length_mm = Some(length);
        design
    }

    #[test]
    fn each_neighbour_is_the_design_with_only_the_magnet_count_changed() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let consequences = Consequences::of(&inputs, &results, None);
        let counts: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
        assert_eq!(counts, [npole - 2, npole + 2]);
        for neighbour in &consequences.neighbours {
            let design = with_magnets(&inputs, neighbour.npole);
            let r = compute_all(&design);
            match neighbour.outcome {
                NeighbourOutcome::Torque {
                    pullout_Nm: pullout,
                    change,
                    cup_od_mm,
                } => {
                    assert!(
                        crate::engine::sizing::is_valid(&design, &r),
                        "{neighbour:?}"
                    );
                    assert_eq!(pullout, r.model.pullout_Nm);
                    assert_eq!(change, r.model.pullout_Nm / results.model.pullout_Nm - 1.0);
                    assert_eq!(cup_od_mm, r.model.cup_od_mm);
                }
                NeighbourOutcome::DoesNotFit => {
                    assert!(!blocks_fit(&design.coupling, &r.model), "{neighbour:?}")
                }
                NeighbourOutcome::NoPrediction => {
                    assert!(!end_effect_in_range(r.model.f_end), "{neighbour:?}")
                }
            }
        }
        // The owner's case (spec 1): two fewer magnets fit, lose torque and grow the cup; two
        // more do not fit at this ring size.
        match consequences.neighbours[0].outcome {
            NeighbourOutcome::Torque {
                change, cup_od_mm, ..
            } => {
                assert!(change < 0.0, "{change}");
                assert!(cup_od_mm > results.model.cup_od_mm, "{cup_od_mm}");
            }
            other => panic!("{} magnets fit at the default ring: {other:?}", npole - 2),
        }
        assert_eq!(
            consequences.neighbours[1].outcome,
            NeighbourOutcome::DoesNotFit
        );
    }

    #[test]
    fn the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not() {
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let fewer = compute_all(&with_magnets(&inputs, npole - 2));
        let lines = Consequences::of(&inputs, &results, None).magnet_lines(npole, None);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        // The change as the Size tab writes every percent (decision 10B: `signed_percent`,
        // one decimal, pinned in `gui::format`'s tests).
        assert_eq!(
            lines[1],
            format!(
                "At {}: pull-out {:.2} N·m ({}), cup OD {:.1} mm",
                npole - 2,
                fewer.model.pullout_Nm,
                signed_percent(fewer.model.pullout_Nm / results.model.pullout_Nm - 1.0),
                fewer.model.cup_od_mm
            )
        );
        assert_eq!(lines[2], format!("At {}: magnets do not fit", npole + 2));
        assert_eq!(lines.len(), 4, "the fit line closes the list: {lines:?}");
    }

    #[test]
    fn at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown() {
        // Spec 5: at 4 or 40 magnets, only the neighbour inside the slider range.
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        for (npole, inside, outside) in
            [(fewest, fewest + 2, fewest - 2), (most, most - 2, most + 2)]
        {
            let inputs = with_magnets(&DesignInputs::default(), npole);
            let consequences = consequences_of(&inputs, None);
            let counts: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
            assert_eq!(counts, [inside], "{npole} magnets");
            let lines = consequences.magnet_lines(npole, None);
            assert!(
                lines
                    .iter()
                    .any(|l| l.starts_with(&format!("At {inside}:"))),
                "{lines:?}"
            );
            assert!(
                !lines
                    .iter()
                    .any(|l| l.starts_with(&format!("At {outside}:"))),
                "{lines:?}"
            );
        }
        // An odd count (a design file's) shows the ring-size line and no neighbour: the slider
        // takes neither count next to it (`torque_card::count_step`, whose own test pins the
        // counts it takes).
        let odd = with_magnets(&DesignInputs::default(), fewest + 5);
        let lines = consequences_of(&odd, None).magnet_lines(odd.coupling.npole, None);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert!(!lines.iter().any(|l| l.starts_with("At ")), "{lines:?}");
    }

    #[test]
    fn a_keyway_through_the_hub_changes_no_neighbour_s_reading() {
        // The hub wall past the keyway does not depend on the magnet count, so a neighbour
        // fails it exactly when the design shown does (the Size tab says so already).
        let mut broken = DesignInputs::default();
        broken.coupling.bore_mm = 16.0; // the bore and keyway sliders' maxima
        broken.coupling.keyway_depth_mm = 4.0;
        let mut intact = broken.clone();
        intact.coupling.keyway_depth_mm = 0.0;
        assert!(compute_all(&broken).model.hub_wall_past_key_mm <= 0.0);
        assert!(compute_all(&intact).model.hub_wall_past_key_mm > 0.0);
        let read = |inputs: &DesignInputs| neighbours(inputs, &compute_all(inputs));
        assert_eq!(read(&broken), read(&intact));
        assert!(
            matches!(read(&broken)[0].outcome, NeighbourOutcome::Torque { .. }),
            "{:?}",
            read(&broken)
        );
    }

    #[test]
    fn with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it() {
        let range = FreeVariable::MagnetsPerRing.range();
        for design in [
            DesignInputs::default(),
            crate::gui::test_support::short_magnets(),
            end_effect_edge(),
        ] {
            for npole in (range.min as i64..=range.max as i64).step_by(2) {
                let inputs = with_magnets(&design, npole);
                for neighbour in neighbours(&inputs, &compute_all(&inputs)) {
                    let at = with_magnets(&design, neighbour.npole);
                    let r = compute_all(&at);
                    assert!(r.model.hub_wall_past_key_mm > 0.0);
                    assert_eq!(
                        matches!(neighbour.outcome, NeighbourOutcome::Torque { .. }),
                        crate::engine::sizing::is_valid(&at, &r),
                        "{npole} -> {neighbour:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented() {
        // Spec 5: words, not numbers. The design shown has no prediction either, so the
        // neighbour that has one shows its torque without a change against a meaningless one.
        let inputs = end_effect_edge();
        let results = compute_all(&inputs);
        assert!(
            !end_effect_in_range(results.model.f_end),
            "{}",
            results.model.f_end
        );
        let npole = inputs.coupling.npole;
        let consequences = Consequences::of(&inputs, &results, None);
        assert_eq!(
            consequences.neighbours[0],
            Neighbour {
                npole: npole - 2,
                outcome: NeighbourOutcome::NoPrediction
            }
        );
        let more = compute_all(&with_magnets(&inputs, npole + 2));
        match consequences.neighbours[1].outcome {
            NeighbourOutcome::Torque {
                pullout_Nm: pullout,
                change,
                cup_od_mm,
            } => {
                assert_eq!(pullout, more.model.pullout_Nm);
                assert!(change.is_nan(), "{change}");
                assert_eq!(cup_od_mm, more.model.cup_od_mm);
            }
            other => panic!("{other:?}"),
        }
        let lines = consequences.magnet_lines(npole, None);
        assert_eq!(lines[1], format!("At {}: no prediction", npole - 2));
        assert_eq!(
            lines[2],
            format!(
                "At {}: pull-out {:.2} N·m, cup OD {:.1} mm",
                npole + 2,
                more.model.pullout_Nm,
                more.model.cup_od_mm
            )
        );
    }

    #[test]
    fn with_magnets_too_short_for_the_model_every_neighbour_reads_words() {
        let inputs = crate::gui::test_support::short_magnets();
        let npole = inputs.coupling.npole;
        let lines = consequences_of(&inputs, None).magnet_lines(npole, None);
        let neighbour_lines: Vec<&String> = lines.iter().filter(|l| l.starts_with("At ")).collect();
        assert_eq!(neighbour_lines.len(), 2, "{lines:?}");
        for (line, count) in neighbour_lines.into_iter().zip([npole - 2, npole + 2]) {
            let r = compute_all(&with_magnets(&inputs, count));
            let words = if !blocks_fit(&with_magnets(&inputs, count).coupling, &r.model) {
                "magnets do not fit"
            } else if !end_effect_in_range(r.model.f_end) {
                "no prediction"
            } else {
                panic!("{count} magnets give a torque on short magnets: {line}")
            };
            assert_eq!(*line, format!("At {count}: {words}"));
        }
    }
```

New:
```rust
    /// The consequences of `inputs` as the panel builds them: with the design's own results, at
    /// the shipped shaft-clamp pause.
    fn consequences_of(inputs: &DesignInputs, minimum_diameter_mm: Option<f64>) -> Consequences {
        Consequences::of(
            inputs,
            &compute_all(inputs),
            minimum_diameter_mm,
            crate::engine::clamps::PAUSED,
        )
    }

    /// `inputs` with `npole` magnets per ring and nothing else changed.
    fn with_magnets(inputs: &DesignInputs, npole: i64) -> DesignInputs {
        FreeVariable::MagnetsPerRing.apply(inputs, npole as f64)
    }
```

`without_a_breakdown_or_a_fit_there_is_no_line_for_it` builds the struct with the card (spec 6.3; its assertions are unchanged):

Old:
```rust
        let empty = Consequences {
            neighbours: Vec::new(),
```

New:
```rust
        let empty = Consequences {
            card: TorqueCard::Rows(Vec::new()),
```

`every_consequence_line_has_its_glyphs_and_plain_words`: its comment states the line forms that remain under the controls:

Old:
```rust
        // Spec tests 3 and 11, over every line form: a torque, a torque without a change, both
        // word forms, the three fit forms and both breakdowns.
```

New:
```rust
        // Spec tests 3 and 11, over every line form under the controls: the ring-size line, the
        // three fit forms and both breakdowns (the neighbour lines' torque and word forms are the
        // torque card's count row since slice 3: its own tests hold its lines).
```

`the_first_magnet_line_follows_the_auto_state`, rewritten to spec 4.3, with spec 6.3's line counts (Auto off and on the magnet length 2, on the ring size and on the count 1). The two Auto lines' texts no longer exist, so no literal stays:

Old:
```rust
    #[test]
    fn the_first_magnet_line_follows_the_auto_state() {
        // Physics review 3, N4: the ring keeps its radius only with Auto off. With Auto on the
        // ring size a new count is solved again, and on the magnet length the length is; the
        // neighbour lines keep the present size, and the first line says so. With Auto on the
        // count its slider is locked, so no line speaks of changing it.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let inputs = DesignInputs::default();
        let npole = inputs.coupling.npole;
        let consequences = consequences_of(&inputs, None);
        let off = consequences.magnet_lines(npole, None);
        let states: [(Option<FreeVariable>, Option<&str>); 4] = [
            (None, Some(RING_SIZE_STAYS)),
            (
                Some(FreeVariable::RingRadius),
                Some(
                    "Auto resizes the ring when you change the magnet count; the lines below keep the present ring radius.",
                ),
            ),
            (
                Some(FreeVariable::AxialLength),
                Some(
                    "Auto resets the magnet length when you change the magnet count; the lines below keep the present length.",
                ),
            ),
            (Some(FreeVariable::MagnetsPerRing), None),
        ];
        for (auto, first) in states {
            assert_eq!(magnet_count_line(auto), first, "{auto:?}");
            let lines = consequences.magnet_lines(npole, auto);
            // The first line alone follows the state: the rest are the Auto-off lines.
            match first {
                Some(first) => {
                    assert_eq!(lines[0], first, "{auto:?}");
                    assert_eq!(lines[1..], off[1..], "{auto:?}");
                    crate::gui::test_support::assert_glyphs(&ctx, first, "a magnet count line");
                    assert_eq!(plain_text_violations(first), Vec::<&str>::new(), "{first}");
                }
                None => assert_eq!(lines[..], off[1..], "{auto:?}"),
            }
        }
    }
```

New:
```rust
    #[test]
    fn the_first_magnet_line_follows_the_auto_state() {
        // Slice 3 spec 4.3 (physics review 3, N4, as the card leaves it): the ring keeps its
        // radius with Auto off and with Auto on the magnet length (that Auto moves no ring), so
        // both draw the ring-size line. With Auto on the ring size a new count sizes the ring
        // again, which the torque card's Auto line says (decision 5A), so no line under Magnets
        // per ring speaks of it; with Auto on the count its slider is locked. The fit line
        // follows in every state, and no line names a neighbouring count (decision 7A: the card's
        // count row): 2 lines with the ring-size line, 1 without.
        let ctx = egui::Context::default();
        let _ = ctx.run(egui::RawInput::default(), |_| {});
        let inputs = DesignInputs::default();
        let npole = inputs.coupling.npole;
        let consequences = consequences_of(&inputs, None);
        let off = consequences.magnet_lines(npole, None);
        let states: [(Option<FreeVariable>, Option<&str>); 4] = [
            (None, Some(RING_SIZE_STAYS)),
            (Some(FreeVariable::AxialLength), Some(RING_SIZE_STAYS)),
            (Some(FreeVariable::RingRadius), None),
            (Some(FreeVariable::MagnetsPerRing), None),
        ];
        for (auto, first) in states {
            assert_eq!(magnet_count_line(auto), first, "{auto:?}");
            let lines = consequences.magnet_lines(npole, auto);
            assert_eq!(
                lines.len(),
                1 + usize::from(first.is_some()),
                "{auto:?}: {lines:?}"
            );
            // The first line alone follows the state: the rest is the Auto-off fit line.
            match first {
                Some(first) => {
                    assert_eq!(lines[0], first, "{auto:?}");
                    assert_eq!(lines[1..], off[1..], "{auto:?}");
                }
                None => assert_eq!(lines[..], off[1..], "{auto:?}"),
            }
        }
        // The ring-size line's words (its clause is new in slice 3): glyphs and plain words.
        crate::gui::test_support::assert_glyphs(&ctx, RING_SIZE_STAYS, "a magnet count line");
        assert_eq!(plain_text_violations(RING_SIZE_STAYS), Vec::<&str>::new());
    }
```

`the_cache_computes_once_per_design_and_minimum_and_follows_an_edit` passes the pause to every `get` and `of` (spec 6.3). It reads the card in place of the neighbours. It adds the pause as a key part, and a pin that the pause reaches the card: at `clamp_at_its_limit()` the paused and unpaused cards differ (Step 5, fault b):

Old:
```rust
    #[test]
    fn the_cache_computes_once_per_design_and_minimum_and_follows_an_edit() {
        // Spec 4.1 and test 5: computed again only when the inputs or the minimum change; what
        // it returns is always Consequences::of the design asked about.
        let mut cache = ConsequenceCache::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let first = cache.get(&inputs, &results, None).clone();
        assert_eq!(first, Consequences::of(&inputs, &results, None));
        for _ in 0..5 {
            assert_eq!(cache.get(&inputs, &results, None), &first);
        }
        assert_eq!(
            cache.computed, 1,
            "a design asked about again is not computed again"
        );
        // An edit: the neighbours' numbers follow it.
        let mut hot = inputs.clone();
        hot.coupling.op_temp_C += 30.0;
        let hot_results = compute_all(&hot);
        let edited = cache.get(&hot, &hot_results, None).clone();
        assert_eq!(cache.computed, 2);
        assert_eq!(edited, Consequences::of(&hot, &hot_results, None));
        assert_ne!(edited.neighbours, first.neighbours);
        let fewer = compute_all(&with_magnets(&hot, hot.coupling.npole - 2));
        assert!(
            matches!(
                edited.neighbours[0].outcome,
                NeighbourOutcome::Torque { pullout_Nm: pullout, .. } if pullout == fewer.model.pullout_Nm
            ),
            "{:?}",
            edited.neighbours
        );
        // A new minimum diameter is a new question too.
        let lowest = crate::engine::size_controls::dimensions(&hot)
            .expect("a size")
            .minimum_diameter_range
            .min;
        assert_eq!(
            cache.get(&hot, &hot_results, Some(lowest)),
            &Consequences::of(&hot, &hot_results, Some(lowest))
        );
        assert_eq!(cache.computed, 3);
        cache.get(&hot, &hot_results, Some(lowest));
        assert_eq!(cache.computed, 3);
    }
```

New:
```rust
    #[test]
    fn the_cache_computes_once_per_design_and_minimum_and_follows_an_edit() {
        // Spec 4.1 and test 5: computed again only when the inputs, the minimum or the
        // shaft-clamp pause change (slice 3 spec 3.2: one key in three parts); what it returns is
        // always Consequences::of the design asked about.
        use crate::gui::torque_card::{Control, Side};
        let paused = crate::engine::clamps::PAUSED;
        let mut cache = ConsequenceCache::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let first = cache.get(&inputs, &results, None, paused).clone();
        assert_eq!(first, Consequences::of(&inputs, &results, None, paused));
        for _ in 0..5 {
            assert_eq!(cache.get(&inputs, &results, None, paused), &first);
        }
        assert_eq!(
            cache.computed, 1,
            "a design asked about again is not computed again"
        );
        // An edit: the torque card's numbers follow it (its count row is the old neighbour
        // lines, in hot-low).
        let mut hot = inputs.clone();
        hot.coupling.op_temp_C += 30.0;
        let hot_results = compute_all(&hot);
        let edited = cache.get(&hot, &hot_results, None, paused).clone();
        assert_eq!(cache.computed, 2);
        assert_eq!(edited, Consequences::of(&hot, &hot_results, None, paused));
        assert_ne!(edited.card, first.card);
        let fewer = compute_all(&with_magnets(&hot, hot.coupling.npole - 2));
        let count = edited
            .card
            .rows()
            .iter()
            .find(|row| row.control == Control::MagnetsPerRing)
            .expect("the count row");
        assert!(
            matches!(
                &count.fewer,
                Side::Change(change) if change.hot_low_Nm == fewer.metal.torque_hot_low_Nm
            ),
            "{count:?}"
        );
        // A new minimum diameter is a new question too.
        let lowest = crate::engine::size_controls::dimensions(&hot)
            .expect("a size")
            .minimum_diameter_range
            .min;
        assert_eq!(
            cache.get(&hot, &hot_results, Some(lowest), paused),
            &Consequences::of(&hot, &hot_results, Some(lowest), paused)
        );
        assert_eq!(cache.computed, 3);
        cache.get(&hot, &hot_results, Some(lowest), paused);
        assert_eq!(cache.computed, 3);
        // And so is the other pause: the card's flips and margins run over the guides that count
        // (an unpaused test panel counts the shaft clamp's).
        assert_eq!(
            cache.get(&hot, &hot_results, Some(lowest), !paused),
            &Consequences::of(&hot, &hot_results, Some(lowest), !paused)
        );
        assert_eq!(cache.computed, 4);
        cache.get(&hot, &hot_results, Some(lowest), !paused);
        assert_eq!(cache.computed, 4);
        // The pause reaches the card: at the clamp's limit one step of length flips the shaft
        // clamp, which counts only while its section is not paused (spec 3.2, test 7a).
        let limit = clamp_at_its_limit();
        let limit_results = compute_all(&limit);
        assert_ne!(
            Consequences::of(&limit, &limit_results, None, false).card,
            Consequences::of(&limit, &limit_results, None, true).card
        );
    }
```

(b) `magcoupling-rs/src/gui/test_support.rs`: `end_effect_edge`, moved from `gui/guidance.rs`'s tests. Its body is the same; the guidance test helper `with_magnets` becomes a local closure. It goes right before `cup_without_wall_or_key_bearing`:

Old:
```rust
/// The defaults with a 16 mm bore in a 12 mm boss and no boss length (BL-064): the cup has no
```

New:
```rust
/// A design whose own end effect is just out of range while two more magnets bring it back in
/// and fit: the first ring radius of the solver's grid where two more magnets fit, the magnets
/// cut halfway between the two counts' end-effect limits (f_end = 1 - c_end * pole pitch / L is
/// zero at L = c_end * pole pitch, and the pitch falls as the count rises). Moved from
/// `gui::guidance`'s tests (slice 3): they and the torque card's count-row tests read it.
pub(crate) fn end_effect_edge() -> crate::DesignInputs {
    use crate::engine::sizing::FreeVariable;
    let with_magnets = |inputs: &crate::DesignInputs, npole: i64| {
        FreeVariable::MagnetsPerRing.apply(inputs, npole as f64)
    };
    let base = crate::DesignInputs::default();
    let more = base.coupling.npole + 2;
    let radius = FreeVariable::RingRadius
        .grid()
        .into_iter()
        .find(|&radius| {
            let design = with_magnets(&FreeVariable::RingRadius.apply(&base, radius), more);
            crate::engine::model::blocks_fit(&design.coupling, &crate::compute_all(&design).model)
        })
        .expect("two more magnets fit somewhere on the ring-radius slider");
    let mut design = FreeVariable::RingRadius.apply(&base, radius);
    // The end-effect coefficient's slider maximum: the length lands inside its slider.
    design.coupling.c_end = 0.5;
    let pitch = |npole: i64| {
        crate::compute_all(&with_magnets(&design, npole))
            .model
            .pole_pitch_mm
    };
    let length = design.coupling.c_end * (pitch(base.coupling.npole) + pitch(more)) / 2.0;
    design.coupling.magnets.axial_length_mm = Some(length);
    design
}

/// The defaults with a 16 mm bore in a 12 mm boss and no boss length (BL-064): the cup has no
```

(c) `magcoupling-rs/src/gui/torque_card.rs`, at the end of the test module, after its last test. The seven moved tests, rewritten to the card's types and hot-low, with one local helper. The Old block is the end of Task 4's last test, test 18:

Old:
```rust
        // Not vacuous: every number drawn at section 2's designs.
        assert_eq!(numbers, 1049);
    }
}
```

New:
```rust
        // Not vacuous: every number drawn at section 2's designs.
        assert_eq!(numbers, 1049);
    }

    // The count row: the neighbour lines' tests, moved from `gui/guidance.rs` with the lines
    // (slice 3 T5, spec 6.3; decision 7A), rewritten to the card's types and hot-low.

    #[test]
    fn the_count_row_s_sides_are_the_design_with_only_the_magnet_count_changed() {
        // Was each_neighbour_is_the_design_with_only_the_magnet_count_changed: each side is
        // compute_all of the design with only `coupling.npole` changed, the ring radius kept
        // (FreeVariable::MagnetsPerRing.apply, as the neighbour lines were), now in hot-low, or
        // words where its blocks do not fit.
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let [fewer, more] =
            [false, true].map(|more| side_at(&inputs, None, Control::MagnetsPerRing, more));
        for (count, side) in [(npole - 2, &fewer), (npole + 2, &more)] {
            let design = FreeVariable::MagnetsPerRing.apply(&inputs, count as f64);
            assert_eq!(
                design.coupling.inner_back_apothem_mm,
                inputs.coupling.inner_back_apothem_mm
            );
            let r = compute_all(&design);
            match side {
                Side::Change(change) => {
                    assert!(is_valid(&design, &r), "{count}");
                    assert_eq!(change.hot_low_Nm, r.metal.torque_hot_low_Nm, "{count}");
                    assert_eq!(
                        change.change_Nm,
                        r.metal.torque_hot_low_Nm - results.metal.torque_hot_low_Nm,
                        "{count}"
                    );
                }
                Side::Words(Words::DoesNotFit) => {
                    assert!(!blocks_fit(&design.coupling, &r.model), "{count}")
                }
                other => panic!("{count}: {other:?}"),
            }
        }
        // The owner's case (spec 1): two fewer magnets fit and lose hot-low torque, the cup
        // growing under the cap; two more do not fit at this ring size.
        let fewer = number(&fewer);
        assert!(fewer.change_Nm < 0.0, "{fewer:?}");
        let fewer_cup =
            compute_all(&FreeVariable::MagnetsPerRing.apply(&inputs, (npole - 2) as f64))
                .model
                .cup_od_mm;
        assert!(fewer_cup > results.model.cup_od_mm, "{fewer_cup}");
        assert_eq!(words(&more), Words::DoesNotFit);
    }

    #[test]
    fn the_count_row_says_each_side_s_hot_low_change_or_why_not() {
        // Was the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not: the count's
        // numbers are the card's count row, in hot-low and the Size tab's number format, and the
        // lines under Magnets per ring keep the ring-size line and the fit line only (7A).
        use crate::gui::guidance::{Consequences, RING_SIZE_STAYS};
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let npole = inputs.coupling.npole;
        let fewer = compute_all(&FreeVariable::MagnetsPerRing.apply(&inputs, (npole - 2) as f64));
        let change = fewer.metal.torque_hot_low_Nm - results.metal.torque_hot_low_Nm;
        let label = Control::MagnetsPerRing.label();
        let line = card(&inputs, &results, None, PAUSED)
            .lines(None)
            .into_iter()
            .find(|line| line.starts_with(&format!("{label}: ")))
            .expect("the count row's line");
        // The row's one number side, at its own decimals (T-14).
        let fewer_text = format!(
            "{label}: -2 -> {} N·m ({})",
            signed_change(change, TORQUE_DECIMALS),
            signed_percent(change / results.metal.torque_hot_low_Nm)
        );
        assert!(line.starts_with(&fewer_text), "{line}");
        assert!(
            line.ends_with(&format!("; +2 -> {SIDE_DOES_NOT_FIT}")),
            "{line}"
        );
        let lines = Consequences::of(&inputs, &results, None, PAUSED).magnet_lines(npole, None);
        assert_eq!(
            lines.len(),
            2,
            "the ring-size line and the fit line: {lines:?}"
        );
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert!(
            !lines.iter().any(|line| line.starts_with("At ")),
            "{lines:?}"
        );
    }

    #[test]
    fn at_either_end_of_the_magnet_slider_the_count_side_past_it_reads_its_end() {
        // Was at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown: at 4
        // magnets the -2 side, at 40 the +2 side reads 9A's "at the end of its range", and the
        // side inside the slider is the design with that count. 40 blocks fit only when narrow
        // (the default parts do not fit 40: that card is ShownDoesNotFit). An odd count (a
        // design file's) draws the ring-size line under Magnets per ring and no count: its
        // count row reads NOT_ON_THE_SLIDER both ways (the_words_follow_the_order_of_the_tests).
        use crate::gui::guidance::{Consequences, RING_SIZE_STAYS};
        use crate::gui::test_support::{forty_narrow_magnets, odd_count, with_poles};
        let range = FreeVariable::MagnetsPerRing.range();
        let (fewest, most) = (range.min as i64, range.max as i64);
        for (inputs, past_is_more) in [(with_poles(fewest), false), (forty_narrow_magnets(), true)]
        {
            let npole = inputs.coupling.npole;
            assert_eq!(npole, if past_is_more { most } else { fewest });
            let [fewer, more] =
                [false, true].map(|more| side_at(&inputs, None, Control::MagnetsPerRing, more));
            let (past, inside, inside_count) = if past_is_more {
                (&more, &fewer, npole - 2)
            } else {
                (&fewer, &more, npole + 2)
            };
            assert_eq!(*past, Side::Words(Words::EndOfRange), "{npole}");
            let at = compute_all(&FreeVariable::MagnetsPerRing.apply(&inputs, inside_count as f64));
            match inside {
                Side::Change(change) => {
                    assert_eq!(
                        change.hot_low_Nm, at.metal.torque_hot_low_Nm,
                        "{inside_count}"
                    )
                }
                Side::Words(words) => assert_ne!(*words, Words::EndOfRange, "{inside_count}"),
            }
            let step = if past_is_more { "+2" } else { "-2" };
            let line = card_at(&inputs, None)
                .lines(None)
                .pop()
                .expect("the count row is the last line");
            assert!(
                line.contains(&format!("{step} -> {END_OF_RANGE}")),
                "{line}"
            );
        }
        let odd = odd_count();
        let lines = Consequences::of(&odd, &compute_all(&odd), None, PAUSED)
            .magnet_lines(odd.coupling.npole, None);
        assert_eq!(lines[0], RING_SIZE_STAYS);
        assert!(
            !lines.iter().any(|line| line.starts_with("At ")),
            "{lines:?}"
        );
    }

    #[test]
    fn a_keyway_through_the_hub_changes_no_count_side_s_hot_low() {
        // Was a_keyway_through_the_hub_changes_no_neighbour_s_reading: the hub wall past the
        // keyway does not depend on the magnet count, so the count row skips the keyway (D-1)
        // and its numbers are the same with the keyway through the hub or not (the Size tab's
        // exploration line says it breaks through; the other rows say so on each side).
        let mut broken = DesignInputs::default();
        broken.coupling.bore_mm = 16.0; // the bore and keyway sliders' maxima
        broken.coupling.keyway_depth_mm = 4.0;
        let mut intact = broken.clone();
        intact.coupling.keyway_depth_mm = 0.0;
        assert!(compute_all(&broken).model.hub_wall_past_key_mm <= 0.0);
        assert!(compute_all(&intact).model.hub_wall_past_key_mm > 0.0);
        // Each side's hot-low numbers, or its words.
        let read = |inputs: &DesignInputs| -> Vec<Result<(f64, f64, f64), Words>> {
            [false, true]
                .map(|more| side_at(inputs, None, Control::MagnetsPerRing, more))
                .into_iter()
                .map(|side| match side {
                    Side::Change(change) => {
                        Ok((change.hot_low_Nm, change.change_Nm, change.fraction))
                    }
                    Side::Words(words) => Err(words),
                })
                .collect()
        };
        assert_eq!(read(&broken), read(&intact));
        assert!(read(&broken)[0].is_ok(), "{:?}", read(&broken));
    }

    #[test]
    fn with_hub_wall_past_the_keyway_a_count_side_reads_a_number_exactly_when_sizing_counts_it() {
        // Was with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it:
        // with wall past the keyway, a count side is a number exactly when its design counts for
        // inverse sizing (sizing::is_valid), over the slider's even counts, at the defaults and at
        // a design whose end effect is in range while two magnets fewer take it out (its -2
        // side reads "no prediction").
        let range = FreeVariable::MagnetsPerRing.range();
        let (mut numbers, mut words) = (0, 0);
        for design in [DesignInputs::default(), end_effect_at_a_step()] {
            for npole in (range.min as i64..=range.max as i64).step_by(2) {
                let inputs = FreeVariable::MagnetsPerRing.apply(&design, npole as f64);
                let shown = card_at(&inputs, None);
                // A design shown that does not fit or predicts nothing has no rows (test 0).
                let Some(row) = shown
                    .rows()
                    .iter()
                    .find(|row| row.control == Control::MagnetsPerRing)
                else {
                    continue;
                };
                for (more, side) in sides(row) {
                    let Ok(count) = count_step(npole, more) else {
                        assert_eq!(*side, Side::Words(Words::EndOfRange), "{npole}");
                        continue;
                    };
                    let at = FreeVariable::MagnetsPerRing.apply(&inputs, count as f64);
                    let r = compute_all(&at);
                    assert!(r.model.hub_wall_past_key_mm > 0.0, "{npole} -> {count}");
                    let number = matches!(side, Side::Change(_));
                    assert_eq!(number, is_valid(&at, &r), "{npole} -> {count}: {side:?}");
                    if number {
                        numbers += 1;
                    } else {
                        words += 1;
                    }
                }
            }
        }
        // Not vacuous: both outcomes occur.
        assert!(numbers > 0 && words > 0, "{numbers} numbers, {words} words");
    }

    #[test]
    fn with_no_prediction_on_the_design_shown_the_card_invents_no_change() {
        // Was a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented: the
        // design shown has no prediction (its end effect just out of range) while two more
        // magnets would have one. The neighbour line showed that count's torque with no change;
        // the card compares nothing there (test 0, owner 9A): its one line, no row.
        use crate::gui::test_support::end_effect_edge;
        let inputs = end_effect_edge();
        let results = compute_all(&inputs);
        assert!(
            !crate::engine::model::end_effect_in_range(results.model.f_end),
            "{}",
            results.model.f_end
        );
        let more = FreeVariable::MagnetsPerRing.apply(&inputs, (inputs.coupling.npole + 2) as f64);
        assert!(
            is_valid(&more, &compute_all(&more)),
            "two more magnets predict"
        );
        let shown = card(&inputs, &results, None, PAUSED);
        assert_eq!(shown, TorqueCard::ShownNoPrediction);
        assert!(shown.rows().is_empty());
        for auto in [None].into_iter().chain(FreeVariable::ALL.map(Some)) {
            assert_eq!(shown.lines(auto), [SHOWN_NO_PREDICTION], "{auto:?}");
        }
    }

    #[test]
    fn with_magnets_too_short_for_the_model_the_card_names_no_count() {
        // Was with_magnets_too_short_for_the_model_every_neighbour_reads_words: at short magnets
        // (f_end below 0) the card draws its one line and no row, so no count's number shows,
        // and the lines under Magnets per ring name no count either.
        let inputs = short_magnets();
        let results = compute_all(&inputs);
        let shown = card(&inputs, &results, None, PAUSED);
        assert_eq!(shown, TorqueCard::ShownNoPrediction);
        assert_eq!(shown.lines(None), [SHOWN_NO_PREDICTION]);
        let lines = crate::gui::guidance::Consequences::of(&inputs, &results, None, PAUSED)
            .magnet_lines(inputs.coupling.npole, None);
        assert!(
            !lines.iter().any(|line| line.starts_with("At ")),
            "{lines:?}"
        );
    }
}
```

(d) `magcoupling-rs/src/gui/size_controls.rs`: Task 2's `the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown` builds `Consequences` for the controls. It passes the shipped pause:

Old:
```rust
            let consequences = Consequences::of(inputs, &compute_all(inputs), None);
```

New:
```rust
            let consequences = Consequences::of(
                inputs,
                &compute_all(inputs),
                None,
                crate::engine::clamps::PAUSED,
            );
```

(e) `magcoupling-rs/src/gui/panel.rs`, the test module. The imports: the two removed constants go (spec 6.3, the compiler and clippy's `-D warnings` catch them), and the card's names come in:

Old:
```rust
    use crate::gui::guidance::{
        AUTO_RESETS_LENGTH, AUTO_RESIZES_RING, Consequences, RING_SIZE_STAYS, plain_text_violations,
    };
```

New:
```rust
    use crate::gui::guidance::{Consequences, RING_SIZE_STAYS, plain_text_violations};
```

Old:
```rust
        tooltips_at_once, with_poles,
    };
    use crate::headline;
```

New:
```rust
        tooltips_at_once, with_poles,
    };
    use crate::gui::torque_card::{CAPTION, Control, held_line};
    use crate::headline;
```

And the `test_support` import gains `smoke_inputs` (Task 3; critique A-24):

Old:
```rust
    use crate::gui::test_support::{
        SCREEN, clamp_at_its_limit, drawn_texts, hot_day, hub_overstressed, key_event, key_tap,
        last_characters, only_the_clamp_fails, only_the_joint_fails, primary_button, select_all,
        short_magnets, sized_frame, sized_frame_at, text_center, text_rect, text_rects,
        tooltips_at_once, with_poles,
    };
```

New:
```rust
    use crate::gui::test_support::{
        SCREEN, clamp_at_its_limit, drawn_texts, hot_day, hub_overstressed, key_event, key_tap,
        last_characters, only_the_clamp_fails, only_the_joint_fails, primary_button, select_all,
        short_magnets, sized_frame, sized_frame_at, smoke_inputs, text_center, text_rect,
        text_rects, tooltips_at_once, with_poles,
    };
```

Test 11, the first view: the harness's own case, then the card open and closed (by a click on its heading), both at the shipped pause (spec 8.2):

Old:
```rust
    #[test]
    fn idle_frames_on_the_first_view_change_no_input() {
        // The Size tab's controls (the Required slider among them), like every input row, write
        // only on an edit (decision M41-2): idle frames on the first view keep values off the
        // sliders' steps as they are.
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.coupling.inner_back_apothem_mm += 0.01234;
        design.inputs.coupling.magnets.axial_length_mm = Some(12.7123);
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        design.inputs.metal.required_min_Nm = 2.2345;
        let mut harness = Harness::first_view();
        harness.panel.open_design(design.clone());
        let mut output = harness.frame(Vec::new());
        for _ in 0..15 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
        assert_eq!(count(&output, WELCOME), 1);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), 0);
    }
```

New:
```rust
    #[test]
    fn idle_frames_on_the_first_view_change_no_input() {
        // The Size tab's controls (the Required slider among them), like every input row, write
        // only on an edit (decision M41-2): idle frames on the first view keep values off the
        // sliders' steps as they are. Slice 3 test 11: so do they under the torque card, open and
        // closed by a click on its heading, at the shipped clamp pause (the first case is the
        // harness's own, unpaused).
        let mut design = Design::default();
        design.inputs.metal.face_gap_mm = 1.4123;
        design.inputs.coupling.inner_back_apothem_mm += 0.01234;
        design.inputs.coupling.magnets.axial_length_mm = Some(12.7123);
        design.inputs.metal.measured_drag_Nm = Some(0.012345);
        design.inputs.metal.required_min_Nm = 2.2345;
        for card in [None, Some(true), Some(false)] {
            let mut harness = Harness::first_view();
            harness.panel.clamps_paused = card.is_some();
            harness.panel.open_design(design.clone());
            let mut output = harness.frame(Vec::new());
            if card == Some(false) {
                harness.click(text_center(&output, CARD_HEADING));
            }
            for _ in 0..15 {
                output = harness.frame(Vec::new());
            }
            assert_eq!(harness.panel.centre, CentreView::SizeAndTorque);
            assert_eq!(count(&output, WELCOME), 1);
            assert_eq!(
                count(&output, CAPTION),
                usize::from(card != Some(false)),
                "{card:?}: the card's caption shows exactly while it is open"
            );
            assert_eq!(harness.panel.design(), design, "{card:?}");
            assert_eq!(harness.panel.last_error, None, "{card:?}");
            assert_eq!(harness.panel.history.undo_len(), 0, "{card:?}");
        }
    }
```

Test 11, `idle_frames_change_no_input`: after its two orders, the Size tab with the card open and then closed, at the shipped pause:

Old:
```rust
        let total = InputCatalogue::get().all().count();
        assert_eq!(count(&output, &format!("{total} of {total} inputs")), 1);
        assert_eq!(count(&output, "Vacuum permeability"), 2);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), 0);
    }
```

New:
```rust
        let total = InputCatalogue::get().all().count();
        assert_eq!(count(&output, &format!("{total} of {total} inputs")), 1);
        assert_eq!(count(&output, "Vacuum permeability"), 2);
        assert_eq!(harness.panel.design(), design);
        assert_eq!(harness.panel.last_error, None);
        assert_eq!(harness.panel.history.undo_len(), 0);
        // Slice 3 test 11: the Size tab under the torque card, open and then closed by a click
        // on its heading, at the shipped clamp pause: idle frames still change no input.
        harness.panel.clamps_paused = true;
        harness.click_text(CentreView::SizeAndTorque.label());
        for open in [true, false] {
            let mut output = harness.frame(Vec::new());
            if !open {
                harness.click(text_center(&output, CARD_HEADING));
            }
            for _ in 0..15 {
                output = harness.frame(Vec::new());
            }
            assert_eq!(count(&output, CAPTION), usize::from(open), "open {open}");
            assert_eq!(harness.panel.design(), design, "open {open}");
            assert_eq!(harness.panel.last_error, None, "open {open}");
            assert_eq!(harness.panel.history.undo_len(), 0, "open {open}");
        }
    }
```

`shown_consequences` passes the panel's own pause (spec 6.3; 11 callers):

Old:
```rust
    /// The what-moves-what lines of the design the panel shows, as the Size tab computes them:
    /// that design, its own results, the panel's Minimum OD.
    fn shown_consequences(harness: &Harness) -> Consequences {
        let shown = harness.panel.shown_inputs();
        Consequences::of(
            &shown,
            &compute_all(&shown),
            harness.panel.minimum_diameter_mm,
        )
    }
```

New:
```rust
    /// The what-moves-what lines and the torque card of the design the panel shows, as the Size
    /// tab computes them: that design, its own results, the panel's Minimum OD and its
    /// shaft-clamp pause.
    fn shown_consequences(harness: &Harness) -> Consequences {
        let shown = harness.panel.shown_inputs();
        Consequences::of(
            &shown,
            &compute_all(&shown),
            harness.panel.minimum_diameter_mm,
            harness.panel.clamps_paused,
        )
    }
```

The card in place of the label (spec 6.1): a new test, and the replaced label kept as a constant of the test module (Task 6's tests read it too); then a typed Required value reaching the card in its own frame (spec 3.5's same-frame rule, plan review 4.4):

Old:
```rust
    #[test]
    fn size_tab_draws_each_what_moves_what_line_under_its_control() {
```

New:
```rust
    /// The label the torque card replaced under the Size tab's readouts (slice 3 spec 6.1), kept
    /// as the record of what the card took the place of.
    const REPLACED_LABEL: &str = "Build checks are shown in the dashboard. The geometry drawing fits the view; use the dimensions to compare sizes.";

    #[test]
    fn the_size_tab_draws_the_torque_card_in_place_of_the_label() {
        // Slice 3 spec 6.1: under the readouts, the card's heading with its Why link, then the
        // card's lines for the design shown, each drawn once, one under the next; the label it
        // replaced is gone. A click on the heading closes the card, as one on its arrow does,
        // and another opens it; neither edits the design.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let output = harness.frame(Vec::new());
        assert_eq!(count(&output, REPLACED_LABEL), 0);
        assert_eq!(count(&output, CARD_HEADING), 1);
        let lines = shown_consequences(&harness).card.lines(None);
        assert_eq!(lines.first().map(String::as_str), Some(CAPTION));
        let rect = |output: &egui::FullOutput, text: &str| {
            text_rect(output, text).unwrap_or_else(|| panic!("no {text:?}"))
        };
        let mut above = rect(&output, CARD_HEADING);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
            let at = rect(&output, line);
            assert!(
                above.bottom() <= at.top() + 0.5,
                "{line:?} is not under the line above it"
            );
            above = at;
        }
        for open in [false, true] {
            let output = harness.frame(Vec::new());
            harness.click(text_center(&output, CARD_HEADING));
            // Past the collapse animation.
            let output = harness.frame_after(1.0, Vec::new());
            assert_eq!(count(&output, CARD_HEADING), 1, "open {open}");
            for line in &lines {
                assert_eq!(
                    count(&output, line),
                    usize::from(open),
                    "open {open}: {line}"
                );
            }
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        assert_eq!(harness.panel.history.undo_len(), 0);
    }

    #[test]
    fn a_required_edit_reaches_the_card_in_its_own_frame() {
        // Slice 3 spec 3.5: the card draws the design shown after the frame's edits, the Required
        // slider's included, which the readouts above it make. At the defaults (hot-low
        // 2.29 N·m), Required 2.4 N·m lies under the Overall length +1 mm side's hot-low
        // (2.49 N·m), so that side's Hot-low torque flips: the frame that applies the typed
        // value draws the edited design's card, not the one before the edit.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        harness.panel.clamps_paused = true;
        let output = harness.frame(Vec::new());
        let before = shown_consequences(&harness).card.lines(None);
        let (_, value_box) = beside(&output, REQUIRED).remove(0);
        harness.click(value_box.center());
        harness.frame([select_all(), vec![egui::Event::Text("2.4".to_owned())]].concat());
        let output = harness.frame(key_tap(egui::Key::Enter));
        assert_eq!(harness.panel.inputs().metal.required_min_Nm, 2.4);
        let after = shown_consequences(&harness).card.lines(None);
        assert_ne!(
            after, before,
            "Required 2.4 N·m moves a Hot-low torque flip"
        );
        for line in &after {
            assert_eq!(count(&output, line), 1, "{line}");
        }
    }

    #[test]
    fn size_tab_draws_each_what_moves_what_line_under_its_control() {
```

`size_tab_draws_each_what_moves_what_line_under_its_control`: two lines under Magnets per ring at the defaults (spec 6.3's "4 -> 2"):

Old:
```rust
        let lines = consequences.magnet_lines(10, None);
        assert_eq!(lines.first().map(String::as_str), Some(RING_SIZE_STAYS));
        let mut above = rect(MAGNETS);
```

New:
```rust
        let lines = consequences.magnet_lines(10, None);
        assert_eq!(lines.first().map(String::as_str), Some(RING_SIZE_STAYS));
        // The ring-size line and the fit line (slice 3, decision 7A: the lines for two magnets
        // fewer and two more are the torque card's count row).
        assert_eq!(lines.len(), 2, "{lines:?}");
        let mut above = rect(MAGNETS);
```

`size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter`: two lines at 8 magnets, and the card's lines drawn (the owner's case answered on screen):

Old:
```rust
        let fit = consequences
            .magnet_lines(8, None)
            .into_iter()
            .find(|line| line.starts_with("Fits down to "))
            .expect("a fit line at the slider's floor");
        assert!(fit.ends_with("set Minimum OD lower to reach it."), "{fit}");
        assert_eq!(count(&output, &fit), 1, "{fit}");
    }
```

New:
```rust
        let fit = consequences
            .magnet_lines(8, None)
            .into_iter()
            .find(|line| line.starts_with("Fits down to "))
            .expect("a fit line at the slider's floor");
        assert!(fit.ends_with("set Minimum OD lower to reach it."), "{fit}");
        assert_eq!(count(&output, &fit), 1, "{fit}");
        // Slice 3: two lines under Magnets per ring, and the torque card above the drawing
        // answers the owner's question in numbers (decision 7A moved the neighbour lines there):
        // every one of its lines is drawn.
        assert_eq!(consequences.magnet_lines(8, None).len(), 2);
        for line in consequences.card.lines(None) {
            assert_eq!(count(&output, &line), 1, "{line}");
        }
    }
```

`size_tab_first_magnet_line_follows_the_auto_state`, rewritten to spec 4.3: the ring-size line with Auto off and on the length, none on the ring or the count; the card's Auto line of each state, and no other. Its loop tail, the lines drawn one under the next, is unchanged:

Old:
```rust
    #[test]
    fn size_tab_first_magnet_line_follows_the_auto_state() {
        // Physics review 3, N4: the line under Magnets per ring says what a new count does to
        // the size Auto sets: the ring keeps its radius with Auto off, Auto resizes the ring
        // or resets the length when it sets them, and with Auto on the count (its slider
        // locked) no line speaks of changing it. Only the state's own line is drawn, under
        // Magnets per ring, and the neighbour lines are drawn with each.
        let firsts = [RING_SIZE_STAYS, AUTO_RESIZES_RING, AUTO_RESETS_LENGTH];
        let states = [
            (None, Some(RING_SIZE_STAYS)),
            (Some(FreeVariable::RingRadius), Some(AUTO_RESIZES_RING)),
            (Some(FreeVariable::AxialLength), Some(AUTO_RESETS_LENGTH)),
            (Some(FreeVariable::MagnetsPerRing), None),
        ];
        for (auto, first) in states {
            let mut harness = size_tab(SCREEN, DesignInputs::default());
            if let Some(variable) = auto {
                harness.panel.sizing.variable = variable;
                harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
            }
            harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            for line in firsts {
                let wanted = usize::from(Some(line) == first);
                assert_eq!(count(&output, line), wanted, "{auto:?}: {line}");
            }
            let shown = harness.panel.shown_inputs();
            let lines = shown_consequences(&harness).magnet_lines(shown.coupling.npole, auto);
            assert_eq!(
                lines.first().map(String::as_str) == first,
                first.is_some(),
                "{auto:?}"
            );
```

New:
```rust
    #[test]
    fn size_tab_first_magnet_line_follows_the_auto_state() {
        // Slice 3 spec 4.3 (physics review 3, N4, as the card leaves it): the line under
        // Magnets per ring says what a new count does to the ring: it stays with Auto off and
        // with Auto on the magnet length; with Auto on the ring size the torque card's Auto line
        // says Auto sizes the ring again (decision 5A), and with Auto on the count (its slider
        // locked) no line speaks of changing it. The lines under the control are drawn one under
        // the next, and the card draws the Auto line of the state and no other.
        use crate::gui::torque_card::{AUTO_HOLDS_COUNT, AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING};
        let states = [
            (None, Some(RING_SIZE_STAYS), None),
            (
                Some(FreeVariable::AxialLength),
                Some(RING_SIZE_STAYS),
                Some(AUTO_HOLDS_LENGTH),
            ),
            (Some(FreeVariable::RingRadius), None, Some(AUTO_HOLDS_RING)),
            (
                Some(FreeVariable::MagnetsPerRing),
                None,
                Some(AUTO_HOLDS_COUNT),
            ),
        ];
        for (auto, first, auto_line) in states {
            let mut harness = size_tab(SCREEN, DesignInputs::default());
            if let Some(variable) = auto {
                harness.panel.sizing.variable = variable;
                harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
            }
            harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
            harness.frame(Vec::new());
            let output = harness.frame(Vec::new());
            assert_eq!(
                count(&output, RING_SIZE_STAYS),
                usize::from(first.is_some()),
                "{auto:?}"
            );
            for line in [AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT] {
                let wanted = usize::from(Some(line) == auto_line);
                assert_eq!(count(&output, line), wanted, "{auto:?}: {line}");
            }
            let shown = harness.panel.shown_inputs();
            let lines = shown_consequences(&harness).magnet_lines(shown.coupling.npole, auto);
            assert_eq!(
                lines.len(),
                1 + usize::from(first.is_some()),
                "{auto:?}: {lines:?}"
            );
            assert_eq!(
                lines.first().map(String::as_str) == first,
                first.is_some(),
                "{auto:?}"
            );
```

`size_tab_what_moves_what_lines_follow_each_edit`: the card's lines follow each edit too (its comment named the neighbours):

Old:
```rust
    #[test]
    fn size_tab_what_moves_what_lines_follow_each_edit() {
        // Spec 6 test 5: the lines are recomputed when the design changes (the cache keys on
        // it), from the magnet count's own slider and from any other input.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let at_10 = shown_consequences(&harness).magnet_lines(10, None);
```

New:
```rust
    #[test]
    fn size_tab_what_moves_what_lines_follow_each_edit() {
        // Spec 6 test 5: the lines are recomputed when the design changes (the cache keys on
        // it), from the magnet count's own slider and from any other input; so is the torque
        // card that took over the neighbour lines (slice 3, decision 7A).
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        let at_10 = shown_consequences(&harness).magnet_lines(10, None);
        let card_at_10 = shown_consequences(&harness).card.lines(None);
```

Old:
```rust
        for line in at_10.iter().filter(|line| !at_8.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        // Another input: a wider face gap moves every neighbour's pull-out and cup.
        harness.panel.inputs.metal.face_gap_mm = 2.0;
        let output = harness.frame(Vec::new());
        let wider = shown_consequences(&harness);
        let lines = wider.magnet_lines(8, None);
        assert_ne!(lines, at_8);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in at_8.iter().filter(|line| !lines.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
```

New:
```rust
        for line in at_10.iter().filter(|line| !at_8.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        let card_at_8 = shown_consequences(&harness).card.lines(None);
        assert_ne!(card_at_8, card_at_10);
        for line in &card_at_8 {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in card_at_10.iter().filter(|line| !card_at_8.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        // Another input: a wider face gap moves the smallest diameter that fits and every step
        // of the card.
        harness.panel.inputs.metal.face_gap_mm = 2.0;
        let output = harness.frame(Vec::new());
        let wider = shown_consequences(&harness);
        let lines = wider.magnet_lines(8, None);
        assert_ne!(lines, at_8);
        for line in &lines {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in at_8.iter().filter(|line| !lines.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
        let card_wider = wider.card.lines(None);
        assert_ne!(card_wider, card_at_8);
        for line in &card_wider {
            assert_eq!(count(&output, line), 1, "{line}");
        }
        for line in card_at_8.iter().filter(|line| !card_wider.contains(line)) {
            assert_eq!(count(&output, line), 0, "stale: {line}");
        }
```

`size_tab_magnet_slider_ends_show_only_the_neighbour_in_range` reads the card's count row (its name kept, the spec's). At 4 magnets the `-2` side, at 40 the `+2` side reads `END_OF_RANGE`. The default blocks do not fit 40, so its top end is `forty_narrow_magnets()`. Two lines remain under the control at a slider end (6.3's "3 -> 2"):

Old:
```rust
    #[test]
    fn size_tab_magnet_slider_ends_show_only_the_neighbour_in_range() {
        // Spec 5: at 4 or 40 magnets per ring the line for 2 or 42 is left out.
        for (npole, inside, outside) in [(4, 6, 2), (40, 38, 42)] {
            let mut harness = size_tab(SCREEN, with_poles(npole));
            let output = harness.frame(Vec::new());
            let consequences = shown_consequences(&harness);
            let neighbours: Vec<i64> = consequences.neighbours.iter().map(|n| n.npole).collect();
            assert_eq!(neighbours, [inside], "{npole}");
            let lines = consequences.magnet_lines(npole, None);
            assert!(lines.contains(&RING_SIZE_STAYS.to_owned()), "{lines:?}");
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            let texts = drawn_texts(&output);
            assert!(
                !texts
                    .iter()
                    .any(|t| t.starts_with(&format!("At {outside}:"))),
                "{npole}: {texts:?}"
            );
        }
    }
```

New:
```rust
    #[test]
    fn size_tab_magnet_slider_ends_show_only_the_neighbour_in_range() {
        // Spec 5, as the torque card's count row has it (slice 3, decision 7A): at 4 or 40
        // magnets per ring the side past the slider reads "at the end of its range", drawn on
        // the card, the side inside it is a step, and no text names 2 or 42 magnets. The
        // default blocks do not fit 40 magnets (the card reads its one line there); forty
        // narrow blocks do.
        use crate::gui::test_support::forty_narrow_magnets;
        use crate::gui::torque_card::END_OF_RANGE;
        for (inputs, past, inside) in [
            (with_poles(4), "-2", "+2"),
            (forty_narrow_magnets(), "+2", "-2"),
        ] {
            let npole = inputs.coupling.npole;
            let mut harness = size_tab(SCREEN, inputs);
            let output = harness.frame(Vec::new());
            let consequences = shown_consequences(&harness);
            let row = format!("{}: ", Control::MagnetsPerRing.label());
            let count_line = consequences
                .card
                .lines(None)
                .into_iter()
                .find(|line| line.starts_with(&row))
                .unwrap_or_else(|| panic!("{npole}: no count row"));
            assert!(
                count_line.contains(&format!("{past} -> {END_OF_RANGE}")),
                "{count_line}"
            );
            assert!(
                !count_line.contains(&format!("{inside} -> {END_OF_RANGE}")),
                "{count_line}"
            );
            assert_eq!(count(&output, &count_line), 1, "{count_line}");
            // Two lines under Magnets per ring at a slider end (three before the card): the
            // ring-size line and the fit line.
            let lines = consequences.magnet_lines(npole, None);
            assert_eq!(lines.len(), 2, "{lines:?}");
            assert!(lines.contains(&RING_SIZE_STAYS.to_owned()), "{lines:?}");
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{line}");
            }
            let outside = if past == "-2" { npole - 2 } else { npole + 2 };
            let texts = drawn_texts(&output);
            assert!(
                !texts.iter().any(|t| t.contains(&format!("At {outside}:"))),
                "{npole}: {texts:?}"
            );
        }
    }
```

`size_tab_new_texts_have_glyphs_and_plain_words`: the card's heading and lines are new texts of the Size tab:

Old:
```rust
        let mut new_texts = vec![REQUIRED.to_owned(), HOT_LOW.to_owned()];
```

New:
```rust
        let mut new_texts = vec![
            REQUIRED.to_owned(),
            HOT_LOW.to_owned(),
            CARD_HEADING.to_owned(),
        ];
```

Old:
```rust
            let mut texts = consequences.magnet_lines(shown.coupling.npole, None);
            if dimensions(&shown).is_some() {
```

New:
```rust
            let mut texts = consequences.magnet_lines(shown.coupling.npole, None);
            // The torque card's lines (slice 3), the neighbour lines' numbers among them.
            texts.extend(consequences.card.lines(None));
            if dimensions(&shown).is_some() {
```

The Why-link counts (spec 6.3, code review 2 MF-3): 3 links to `SIZE_TAB_NOTE` (under Magnets per ring, under Outside diameter, the card's heading), each clicked, and 2 to `TORQUE_BAND_NOTE`. The rest of the test is unchanged:

Old:
```rust
    #[test]
    fn the_size_tab_s_why_links_open_their_notes() {
        // Spec 3.7 (S2-8): under Magnets per ring and Outside diameter a link to how the
        // coupling makes torque; at the end of the Required and Hot-low rows a link to the
        // hot minimum check's note. Each opens its note on its own in the Equation panel.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explorer::reviewed_note;
        use crate::gui::guidance::{SIZE_TAB_NOTE, TORQUE_BAND_NOTE};
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        for id in [SIZE_TAB_NOTE, TORQUE_BAND_NOTE] {
            let text =
                why_note_text(reviewed_note(id).expect("a Size-tab link's note is reviewed"));
            for index in 0..2 {
                harness.panel.explorer = Explorer::default();
                let output = harness.frame(Vec::new());
                // The Required and Hot-low links end wrapped rows: click their text.
                let links = last_characters(&output, &text);
                assert_eq!(links.len(), 2, "{text}");
                harness.click(links[index]);
```

New:
```rust
    #[test]
    fn the_size_tab_s_why_links_open_their_notes() {
        // Spec 3.7 (S2-8): under Magnets per ring and Outside diameter, and on the torque card's
        // heading (slice 3 spec 6.1, drawn open or closed), a link to how the coupling makes
        // torque; at the end of the Required and Hot-low rows a link to the hot minimum check's
        // note. Each opens its note on its own in the Equation panel.
        use crate::gui::dashboard::why_note_text;
        use crate::gui::explorer::reviewed_note;
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        for (id, links) in [(SIZE_TAB_NOTE, 3), (TORQUE_BAND_NOTE, 2)] {
            let text =
                why_note_text(reviewed_note(id).expect("a Size-tab link's note is reviewed"));
            for index in 0..links {
                harness.panel.explorer = Explorer::default();
                let output = harness.frame(Vec::new());
                // The Required and Hot-low links end wrapped rows: click their text.
                let at = last_characters(&output, &text);
                assert_eq!(at.len(), links, "{text}");
                harness.click(at[index]);
```

At an odd count, 2 links (Magnets per ring and the card's heading):

Old:
```rust
        // An odd magnet count has no diameter range: the Outside diameter control and its link
        // are not drawn; the link under Magnets per ring is.
```

New:
```rust
        // An odd magnet count has no diameter range: the Outside diameter control and its link
        // are not drawn; the link under Magnets per ring is, and the torque card's heading's (the
        // card keeps its gap and count rows, slice 3 T-5).
```

Old:
```rust
        let text = why_note_text(reviewed_note(SIZE_TAB_NOTE).unwrap());
        assert_eq!(text_rects(&output, &text).len(), 1);
    }
```

New:
```rust
        let text = why_note_text(reviewed_note(SIZE_TAB_NOTE).unwrap());
        assert_eq!(text_rects(&output, &text).len(), 2);
    }
```

After `idle_frames_compute_the_fixes_once_and_an_edit_again`: a drag of the gap slider with Auto on the magnet length computes the card once for each design it draws (critique A-19, RF-2; from `smoke_inputs()`), then test 10, whose pattern follows the fixes test's, with the key's third part, the Minimum OD (plan review 3.2):

Old:
```rust
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.4;
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.fixes.computed, computed + 1);
    }
```

New:
```rust
        harness.panel.inputs.metal.cup_wall_corner_mm = 2.4;
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.fixes.computed, computed + 1);
    }

    #[test]
    fn a_drag_computes_the_card_once_for_each_design_it_draws() {
        // Slice 3 spec 3.5 and decision 13A: on a drag frame the card asks for the design after
        // the frame's edit (one miss) and the next frame's controls ask for that same design (a
        // hit), so a drag computes the consequences once per design drawn, never twice; at the
        // smoke design with Auto on the magnet length, the drag the browser frame check times.
        let mut harness = size_tab(SCREEN, smoke_inputs());
        harness.panel.clamps_paused = true;
        harness.panel.sizing.variable = FreeVariable::AxialLength;
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let gap = harness.size_widget(GAP).rect;
        let mut at = gap.left_center() + egui::vec2(20.0, 0.0);
        harness.frame(vec![egui::Event::PointerMoved(at)]);
        harness.frame(vec![primary_button(at, true)]);
        harness.frame(Vec::new());
        let computed = harness.panel.consequences.computed;
        let mut designs = 0;
        for i in 0..20 {
            let before = harness.panel.shown_inputs();
            at += egui::vec2(if i % 2 == 0 { 3.0 } else { -3.0 }, 0.0);
            harness.frame(vec![egui::Event::PointerMoved(at)]);
            designs += usize::from(harness.panel.shown_inputs() != before);
        }
        harness.frame(vec![primary_button(at, false)]);
        harness.frame(Vec::new());
        assert!(designs >= 10, "the drag moved the gap on {designs} frames");
        assert_eq!(harness.panel.consequences.computed - computed, designs);
    }

    #[test]
    fn idle_frames_compute_the_card_once_and_an_edit_again() {
        // Slice 3 test 10: the torque card lives in the ConsequenceCache, keyed on the design
        // shown, the Minimum OD and the clamp pause (spec 3.2), with the Auto marking applied
        // outside it (spec 3.4). Idle frames compute nothing; a gap edit computes once (the frame
        // that draws it asks with the edited design, the next frame's controls hit); a toggle of
        // the clamp pause computes once; so does a new Minimum OD, which the drawn Outside
        // diameter row follows.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        harness.panel.clamps_paused = true;
        harness.frame(Vec::new());
        let computed = harness.panel.consequences.computed;
        for _ in 0..5 {
            harness.frame(Vec::new());
        }
        assert_eq!(harness.panel.consequences.computed, computed, "idle frames");
        let gap = harness.panel.design_widgets[2];
        harness.ctx.memory_mut(|m| m.request_focus(gap));
        harness.frame(Vec::new());
        harness.frame(key_tap(egui::Key::ArrowRight));
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert!(
            harness.panel.inputs().metal.face_gap_mm > DesignInputs::default().metal.face_gap_mm
        );
        assert_eq!(
            harness.panel.consequences.computed,
            computed + 1,
            "one gap edit"
        );
        harness.panel.clamps_paused = false;
        for _ in 0..3 {
            harness.frame(Vec::new());
        }
        assert_eq!(
            harness.panel.consequences.computed,
            computed + 2,
            "the pause toggled"
        );
        // The key's third part, the Minimum OD (owner 9A: the diameter row steps the slider at
        // the user's Minimum OD): set at the diameter shown, it computes once, and the drawn
        // Outside diameter row reads the slider's end on its -1 mm side, where with none set it
        // read that the magnets do not fit.
        use crate::gui::torque_card::END_OF_RANGE;
        let diameter_row = |harness: &Harness| {
            let start = format!("{}: ", Control::OutsideDiameter.label());
            shown_consequences(harness)
                .card
                .lines(None)
                .into_iter()
                .find(|line| line.starts_with(&start))
                .expect("an Outside diameter row")
        };
        let unset = diameter_row(&harness);
        assert!(
            unset.contains(&format!("-1 mm -> {SIDE_DOES_NOT_FIT};")),
            "{unset}"
        );
        let shown = dimensions(&harness.panel.shown_inputs()).expect("size controls");
        harness.panel.minimum_diameter_mm = Some(shown.diameter_mm);
        let mut output = harness.frame(Vec::new());
        for _ in 0..3 {
            output = harness.frame(Vec::new());
        }
        assert_eq!(
            harness.panel.consequences.computed,
            computed + 3,
            "the Minimum OD set"
        );
        let at_shown = diameter_row(&harness);
        assert!(
            at_shown.contains(&format!("-1 mm -> {END_OF_RANGE};")),
            "{at_shown}"
        );
        assert_eq!(count(&output, &at_shown), 1, "{at_shown}");
        assert_eq!(count(&output, &unset), 0, "stale: {unset}");
        // (i) The smoke design, Auto on the magnet length, solved: turning Auto off adopts the
        // solved length (decision M41-7), so the design shown, and the card, stay; nothing is
        // computed, and the Overall length row, held while Auto was on, reads its numbers.
        let mut harness = size_tab(SCREEN, smoke_inputs());
        harness.panel.clamps_paused = true;
        harness.panel.sizing.variable = FreeVariable::AxialLength;
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        let held = shown_consequences(&harness).card;
        let held_row = held_line(Control::OverallLength);
        assert_eq!(
            count(&output, &held_row),
            1,
            "the held row while Auto is on"
        );
        let computed = harness.panel.consequences.computed;
        harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);
        harness.frame(Vec::new());
        let output = harness.frame(Vec::new());
        assert_eq!(
            harness.panel.consequences.computed, computed,
            "Auto turned off"
        );
        assert_eq!(shown_consequences(&harness).card, held);
        assert_eq!(count(&output, &held_row), 0);
        let length = held
            .lines(None)
            .into_iter()
            .find(|line| line.starts_with(&format!("{}: ", Control::OverallLength.label())))
            .expect("a length row");
        assert_eq!(count(&output, &length), 1, "{length}");
        // (ii) The defaults, Auto on the magnet count: the best count is the one shown (10), so
        // turning Auto on, letting it solve and turning it off computes nothing.
        let mut harness = size_tab(SCREEN, DesignInputs::default());
        harness.panel.clamps_paused = true;
        harness.frame(Vec::new());
        let computed = harness.panel.consequences.computed;
        harness.panel.sizing.variable = FreeVariable::MagnetsPerRing;
        harness.panel.set_sizing_mode(SizingMode::TorqueToMagnets);
        harness.frame(Vec::new());
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.shown_inputs(), DesignInputs::default());
        harness.panel.set_sizing_mode(SizingMode::MagnetsToTorque);
        harness.frame(Vec::new());
        harness.frame(Vec::new());
        assert_eq!(
            harness.panel.consequences.computed, computed,
            "Auto on the count"
        );
    }
```

- [ ] **Step 2: Run them and see the build fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib --no-run`

Expected: the lib test target does not compile (the new signatures and the `card` field do not exist yet), 49 errors:
```text
     17 error[E0609]: no field `card` on type `gui::guidance::Consequences`
     12 error[E0061]: this function takes 3 arguments but 4 arguments were supplied
      7 error[E0425]: cannot find value `CARD_HEADING` in this scope
      7 error[E0061]: this method takes 3 arguments but 4 arguments were supplied
      3 error[E0277]: the size for values of type `str` cannot be known at compilation time
      1 error[E0560]: struct `gui::guidance::Consequences` has no field named `card`
      1 error[E0433]: failed to resolve: use of undeclared type `TorqueCard`
      1 error[E0425]: cannot find value `SIZE_TAB_NOTE` in this scope
error: could not compile `magcoupling-rs` (lib test) due to 49 previous errors
```
(Counted with `... --no-run 2>&1 | grep -E "^error" | sort | uniq -c | sort -rn`.) The build without `--features app` still compiles: the GUI is not built there.

- [ ] **Step 3: The consequences, the cache and the lines under Magnets per ring (`gui/guidance.rs`)**

The module doc (spec 9.2, code review 3 S4): the neighbour lines are the card's now, and the cache keys on the pause too.

Old:
```rust
//! What moves what (spec 3.3): [`Consequences`] are the lines under the Size tab's controls:
//! two magnets fewer and two more (each one `compute_all` of the design with only the count
//! changed; words, not numbers, where the blocks do not fit or the model has no prediction),
//! the smallest outside diameter at which the count fits, and what sets the outside diameter
//! and the overall length ([`crate::engine::size_controls`]). Every number comes from the
//! engine. [`ConsequenceCache`] keeps them until the inputs or the minimum diameter change.
```

New:
```rust
//! What moves what (spec 3.3): [`Consequences`] are the lines under the Size tab's controls
//! (the smallest outside diameter at which the count fits, and what sets the outside diameter
//! and the overall length, [`crate::engine::size_controls`]) and the torque card the Size tab
//! draws between its readouts and its drawing ([`crate::gui::torque_card`]: one step of each
//! size control each way, its count row the lines for two magnets fewer and two more that
//! stood under Magnets per ring until slice 3, decision 7A). Every number comes from the
//! engine. [`ConsequenceCache`] keeps them until the inputs, the minimum diameter or the
//! shaft-clamp pause change.
```

The imports. `end_effect_in_range` and `signed_percent` lose their last users (`prediction`, `Neighbour::line`); the card module's names come in. The third block's Old is Task 3's import of the moved functions:

Old:
```rust
use crate::engine::model::{
    ResolvedMagnet, RingFit, blocks_fit, end_effect_in_range, resolve_magnets, ring_fits,
};
```

New:
```rust
use crate::engine::model::{ResolvedMagnet, RingFit, blocks_fit, resolve_magnets, ring_fits};
```

Old:
```rust
use crate::gui::format::{
    SIGNIFICANT_DIGITS, fewest_decimals_apart, fixed, format_value, last_digit_step,
    signed_percent, with_unit,
};
```

New:
```rust
use crate::gui::format::{
    SIGNIFICANT_DIGITS, fewest_decimals_apart, fixed, format_value, last_digit_step, with_unit,
};
```

Old:
```rust
use crate::gui::torque_card::{
    SIDE_DOES_NOT_FIT, SIDE_NO_PREDICTION, Words, count_step, side_words,
};
```

New:
```rust
use crate::gui::torque_card::{TorqueCard, card};
```

`SIZE_TAB_NOTE`'s doc (spec 9.2):

Old:
```rust
/// Where the Why links under Magnets per ring and Outside diameter on the Size tab lead (spec
/// 3.7): how the coupling makes torque, Start here's first step.
```

New:
```rust
/// Where the Size tab's Why links under Magnets per ring and Outside diameter (spec 3.7) and on
/// the torque card's heading (slice 3 spec 6.1) lead: how the coupling makes torque, Start
/// here's first step.
```

The lines under Magnets per ring. `RING_SIZE_STAYS` takes 4.3's clause. `magnet_count_line` follows 4.3. The two Auto constants go. The neighbour words and types leave with the lines they built (spec T-2: the card's `side_words` and `SIDE_*` words serve all four rows now):

Old:
```rust
// What moves what (spec 3.3, 4.1). Lengths read to 0.1 mm (the Size tab sliders' precision),
// torques to 0.01 N·m and changes to one decimal of percent (`signed_percent`, decision 10B).

/// The first line under Magnets per ring with Auto off: the ring radius
/// (`coupling.inner_back_apothem_mm`) is its own input; the magnet count only sets how many
/// blocks go around it.
pub const RING_SIZE_STAYS: &str = "The ring radius stays the same when you change the magnet count; the cup OD below can still change.";

/// The first line under Magnets per ring with Auto on the ring size (physics review 3, N4): a
/// new count changes the solve, so Auto solves the ring radius again, while the neighbour
/// lines are built at the present ring.
pub const AUTO_RESIZES_RING: &str = "Auto resizes the ring when you change the magnet count; the lines below keep the present ring radius.";

/// The first line under Magnets per ring with Auto on the magnet length (physics review 3,
/// N4): the ring stays, but Auto solves the length again, while the neighbour lines are built
/// at the present length.
pub const AUTO_RESETS_LENGTH: &str = "Auto resets the magnet length when you change the magnet count; the lines below keep the present length.";

/// The first line under Magnets per ring for the Auto state `auto` (`None`: Auto off): what a
/// new magnet count does to the size Auto sets. `None` with Auto on the magnet count itself:
/// its slider is locked, so no one changes the count there.
pub fn magnet_count_line(auto: Option<FreeVariable>) -> Option<&'static str> {
    match auto {
        None => Some(RING_SIZE_STAYS),
        Some(FreeVariable::RingRadius) => Some(AUTO_RESIZES_RING),
        Some(FreeVariable::AxialLength) => Some(AUTO_RESETS_LENGTH),
        Some(FreeVariable::MagnetsPerRing) => None,
    }
}

/// What two magnets fewer or two more would do (spec 3.3).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)] // unit suffix, as the engine's names
pub enum NeighbourOutcome {
    /// The blocks fit and the model predicts a torque.
    Torque {
        /// The neighbour's pull-out torque at the operating temperature (`model.pullout_Nm`).
        pullout_Nm: f64,
        /// Its change as a fraction of the pull-out of the design shown; NaN when that design
        /// has no prediction itself, so there is nothing to compare with.
        change: f64,
        /// The neighbour's cup outer diameter (`model.cup_od_mm`).
        cup_od_mm: f64,
    },
    /// The blocks do not fit at this ring size ([`blocks_fit`]).
    DoesNotFit,
    /// The blocks fit but the model has no number: the end effect is out of range
    /// ([`end_effect_in_range`]) or a torque is not finite.
    NoPrediction,
}

/// A magnet count next to the design's, and what it would do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Neighbour {
    pub npole: i64,
    pub outcome: NeighbourOutcome,
}

impl Neighbour {
    /// "At N: ..." (spec 3.3): the torque and cup, or why there is no number.
    fn line(&self) -> String {
        let at = format!("At {}:", self.npole);
        match self.outcome {
            NeighbourOutcome::Torque {
                pullout_Nm: pullout,
                change,
                cup_od_mm,
            } => {
                let percent = if change.is_finite() {
                    format!(" ({})", signed_percent(change))
                } else {
                    String::new()
                };
                format!(
                    "{at} pull-out {}{percent}, cup OD {}",
                    with_unit(fixed(pullout, 2), "N·m"),
                    with_unit(fixed(cup_od_mm, 1), "mm")
                )
            }
            NeighbourOutcome::DoesNotFit => format!("{at} {SIDE_DOES_NOT_FIT}"),
            NeighbourOutcome::NoPrediction => format!("{at} {SIDE_NO_PREDICTION}"),
        }
    }
}
```

New:
```rust
// What moves what (spec 3.3, 4.1). Lengths read to 0.1 mm (the Size tab sliders' precision).
// Two magnets fewer and two more are the torque card's count row (slice 3, decision 7A), in
// the Size tab's one number format (decision 10B).

/// The first line under Magnets per ring with Auto off or on the magnet length: the ring radius
/// (`coupling.inner_back_apothem_mm`) is its own input, the magnet count only sets how many
/// blocks go around it, and Auto on the magnet length moves no ring. The cup can still change
/// size: what a count step does to it is the torque card's count row now (slice 3 spec 4.3;
/// decision 7A moved the neighbour lines' "cup OD" there).
pub const RING_SIZE_STAYS: &str = "The ring radius stays the same when you change the magnet count; the cup can still change size.";

/// The first line under Magnets per ring for the Auto state `auto` (`None`: Auto off): what a
/// new magnet count does to the ring (slice 3 spec 4.3). [`RING_SIZE_STAYS`] with Auto off and
/// with Auto on the magnet length, which moves no ring. `None` with Auto on the ring size, which
/// sizes the ring again after a count change (the torque card's Auto line says so, decision
/// 5A), and with Auto on the magnet count itself, whose slider is locked.
pub fn magnet_count_line(auto: Option<FreeVariable>) -> Option<&'static str> {
    match auto {
        None | Some(FreeVariable::AxialLength) => Some(RING_SIZE_STAYS),
        Some(FreeVariable::RingRadius | FreeVariable::MagnetsPerRing) => None,
    }
}
```

`Consequences`: the card in place of the neighbours, `of` with the pause, `magnet_lines` without the neighbour lines:

Old:
```rust
pub struct Consequences {
    /// Two magnets fewer and two more, ascending, where the count stays an even count inside
    /// the magnet slider.
    pub neighbours: Vec<Neighbour>,
```

New:
```rust
pub struct Consequences {
    /// The torque card of the design (slice 3, [`card`]): one step of each size control each
    /// way, drawn between the Size tab's readouts and its drawing.
    pub card: TorqueCard,
```

Old:
```rust
    /// The consequences of `inputs`, whose results are `results` (`compute_all(inputs)`: each
    /// neighbour's change compares with this design's pull-out), with the Size tab's minimum
    /// diameter preference (`None`: the slider's own floor).
    pub fn of(
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
    ) -> Self {
```

New:
```rust
    /// The consequences of `inputs`, whose results are `results` (`compute_all(inputs)`: each
    /// step of the torque card compares with this design's hot-low), with the Size tab's minimum
    /// diameter preference (`None`: the slider's own floor) and the shaft-clamp pause
    /// (`clamps_paused`, the panel's: the card's flips and margins run over the guides that
    /// count, slice 3 spec 3.2).
    pub fn of(
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
        clamps_paused: bool,
    ) -> Self {
```

Old:
```rust
        Self {
            neighbours: neighbours(inputs, results),
            fit,
```

New:
```rust
        Self {
            card: card(inputs, results, minimum_diameter_mm, clamps_paused),
            fit,
```

Old:
```rust
    /// The lines under Magnets per ring: the line for the Auto state `auto`
    /// ([`magnet_count_line`]; `None`: Auto off), one line per neighbour, then the smallest
    /// diameter that fits. `npole` is the count the consequences were built for
    /// (`inputs.coupling.npole`).
    pub fn magnet_lines(&self, npole: i64, auto: Option<FreeVariable>) -> Vec<String> {
        let mut lines: Vec<String> = magnet_count_line(auto)
            .map(str::to_owned)
            .into_iter()
            .collect();
        lines.extend(self.neighbours.iter().map(Neighbour::line));
        lines.extend(self.fit.map(|fit| fit_line(fit, self.fit_unlocked, npole)));
```

New:
```rust
    /// The lines under Magnets per ring (decision 7A): the line for the Auto state `auto`
    /// ([`magnet_count_line`]; `None`: Auto off), then the smallest diameter that fits. `npole`
    /// is the count the consequences were built for (`inputs.coupling.npole`). Two magnets fewer
    /// and two more are the torque card's count row.
    pub fn magnet_lines(&self, npole: i64, auto: Option<FreeVariable>) -> Vec<String> {
        let mut lines: Vec<String> = magnet_count_line(auto)
            .map(str::to_owned)
            .into_iter()
            .collect();
        lines.extend(self.fit.map(|fit| fit_line(fit, self.fit_unlocked, npole)));
```

`neighbours` and `prediction` go (spec T-2: `prediction`'s only caller was `neighbours`; the card's test 0 and step 6 read the readouts' own rule, `size_controls::predicts`). Old: the end of `impl Consequences` and the two functions after it:

Old:
```rust
}

/// The neighbours of `inputs` (`results` are its own), each one `compute_all` of the design
/// with only the magnet count changed: two fewer and two more where the magnet slider takes
/// the count ([`count_step`], the torque card's count row's step), read by the card's validity
/// rule ([`side_words`]) but the keyway's (decision D-1: the hub wall past the keyway does not
/// depend on the magnet count, so a neighbour fails it exactly when the design shown does, the
/// Size tab already says so, and the torques still compare).
fn neighbours(inputs: &DesignInputs, results: &DesignResults) -> Vec<Neighbour> {
    let baseline = prediction(results);
    [false, true]
        .into_iter()
        .filter_map(|more| count_step(inputs.coupling.npole, more).ok())
        .map(|npole| {
            let design = FreeVariable::MagnetsPerRing.apply(inputs, npole as f64);
            let results = compute_all(&design);
            let outcome = match side_words(&design, &results, false) {
                None => NeighbourOutcome::Torque {
                    pullout_Nm: results.model.pullout_Nm,
                    change: baseline.map_or(f64::NAN, |base| results.model.pullout_Nm / base - 1.0),
                    cup_od_mm: results.model.cup_od_mm,
                },
                Some(Words::DoesNotFit) => NeighbourOutcome::DoesNotFit,
                // The keyway left out, the only other words are the prediction's.
                Some(_) => NeighbourOutcome::NoPrediction,
            };
            Neighbour { npole, outcome }
        })
        .collect()
}

/// The pull-out of the design shown when the model predicts one: the end effect in range and
/// a positive, finite torque. Without it a neighbour's change has nothing to compare with.
fn prediction(results: &DesignResults) -> Option<f64> {
    let pullout = results.model.pullout_Nm;
    (end_effect_in_range(results.model.f_end) && pullout.is_finite() && pullout > 0.0)
        .then_some(pullout)
}
```

New:
```rust
}
```

The cache: one key in three parts (spec 3.2):

Old:
```rust
/// [`Consequences`] kept for the design they were built for (spec 4.1): the panel asks every
/// frame, and the neighbours' `compute_all` runs and the fit search are not per-frame work. They
/// are computed again only when the inputs or the minimum diameter change.
```

New:
```rust
/// [`Consequences`] kept for the design they were built for (spec 4.1): the panel asks every
/// frame, and the torque card's stepped `compute_all` runs and the fit search are not per-frame
/// work. They are computed again only when the inputs, the minimum diameter or the shaft-clamp
/// pause change: one key in three parts (slice 3 spec 3.2), as [`FixCache`] keys on the pause.
```

Old:
```rust
/// What a [`ConsequenceCache`] holds: the question and its answer.
struct CachedConsequences {
    inputs: DesignInputs,
    minimum_diameter_mm: Option<f64>,
    consequences: Consequences,
}

impl ConsequenceCache {
    /// [`Consequences::of`] `inputs` (whose results are `results`) with the minimum diameter
    /// preference, computed only when the inputs or the minimum differ from the last call's.
    pub fn get(
        &mut self,
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
    ) -> &Consequences {
        let current = self.last.as_ref().is_some_and(|last| {
            last.inputs == *inputs && last.minimum_diameter_mm == minimum_diameter_mm
        });
        if !current {
            self.computed += 1;
            self.last = Some(CachedConsequences {
                inputs: inputs.clone(),
                minimum_diameter_mm,
                consequences: Consequences::of(inputs, results, minimum_diameter_mm),
            });
        }
```

New:
```rust
/// What a [`ConsequenceCache`] holds: the question (the design, the Minimum OD and the
/// shaft-clamp pause) and its answer.
struct CachedConsequences {
    inputs: DesignInputs,
    minimum_diameter_mm: Option<f64>,
    clamps_paused: bool,
    consequences: Consequences,
}

impl ConsequenceCache {
    /// [`Consequences::of`] `inputs` (whose results are `results`) with the minimum diameter
    /// preference and the shaft-clamp pause, computed only when the inputs, the minimum or the
    /// pause differ from the last call's.
    pub fn get(
        &mut self,
        inputs: &DesignInputs,
        results: &DesignResults,
        minimum_diameter_mm: Option<f64>,
        clamps_paused: bool,
    ) -> &Consequences {
        let current = self.last.as_ref().is_some_and(|last| {
            last.inputs == *inputs
                && last.minimum_diameter_mm == minimum_diameter_mm
                && last.clamps_paused == clamps_paused
        });
        if !current {
            self.computed += 1;
            self.last = Some(CachedConsequences {
                inputs: inputs.clone(),
                minimum_diameter_mm,
                clamps_paused,
                consequences: Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused),
            });
        }
```

- [ ] **Step 4: The card on the Size tab (`gui/panel.rs`) and the controls' module doc (`gui/size_controls.rs`)**

The imports. The second block's Old is Task 4's line:

Old:
```rust
use crate::gui::guidance::{
    ConsequenceCache, FixCache, SIZE_TAB_ALIASES, TORQUE_BAND_NOTE, within_minimum_diameter,
};
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, hint_line, input_row, slider};
```

New:
```rust
use crate::gui::guidance::{
    ConsequenceCache, FixCache, SIZE_TAB_ALIASES, SIZE_TAB_NOTE, TORQUE_BAND_NOTE,
    within_minimum_diameter,
};
use crate::gui::history::History;
use crate::gui::input_ui::{CHANGED_DOT, RowEdit, hint_line, input_row, slider, weak_line};
```

Old:
```rust
use crate::gui::torque_card::{KEYWAY_BREAKS_THROUGH, SIDE_DOES_NOT_FIT};
```

New:
```rust
use crate::gui::torque_card::{CARD_HEADING, KEYWAY_BREAKS_THROUGH, SIDE_DOES_NOT_FIT};
```

The card's id, beside the Equation panel's:

Old:
```rust
const EQUATION_PANEL_ID: &str = "magcoupling_equation_panel";
```

New:
```rust
const EQUATION_PANEL_ID: &str = "magcoupling_equation_panel";

/// The id of the torque card's collapsing state on the Size tab (slice 3 spec 6.1): egui keeps
/// the card open or closed under it.
const TORQUE_CARD_ID: &str = "magcoupling_torque_card";
```

The cache field's doc (spec 9.2):

Old:
```rust
    /// The Size tab's what-moves-what lines (spec 3.3) for the design shown and the minimum OD,
    /// recomputed only when either changes: view state, in no design file, link or undo step.
```

New:
```rust
    /// The Size tab's what-moves-what lines (spec 3.3) and its torque card (slice 3) for the
    /// design shown, the minimum OD and the shaft-clamp pause, recomputed only when one of the
    /// three changes: view state, in no design file, link or undo step.
```

`auto()`, one expression for the controls and the card (spec 6.1 names `:1134-1135`'s expression):

Old:
```rust
    /// The sizing state.
    pub fn sizing(&self) -> &SizingState {
        &self.sizing
    }
```

New:
```rust
    /// The sizing state.
    pub fn sizing(&self) -> &SizingState {
        &self.sizing
    }

    /// The variable Auto sets while it is on (Torque → Magnets), `None` while it is off: the
    /// size the Size tab's controls lock and the torque card holds (slice 3 spec 6.1).
    fn auto(&self) -> Option<FreeVariable> {
        (self.sizing.mode == SizingMode::TorqueToMagnets).then_some(self.sizing.variable)
    }
```

`auto()` reads `FreeVariable` again, so its import comes back to the panel's own imports (Task 3 moved it into the test module while the production code had no use of it; plan deviation PD-59),

Old:
```rust
use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};
use crate::engine::sizing::SizingOutcome;
```

New:
```rust
use crate::engine::meta::{InputSet, ResultSet, SliderRange, Value};
use crate::engine::sizing::{FreeVariable, SizingOutcome};
```

and leaves the test module's:

Old:
```rust
    use crate::engine::sizing::{FreeVariable, SizingOutcome};
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
```

New:
```rust
    use crate::engine::sizing::SizingOutcome;
    use crate::gui::dashboard::{END_EFFECT_BANNER, STORED_3D_LABEL, result_info};
```

Old:
```rust
            self.set_sizing_mode(asked.mode);
            let auto =
                (self.sizing.mode == SizingMode::TorqueToMagnets).then_some(self.sizing.variable);
            let shown = self.shown_inputs();
```

New:
```rust
            self.set_sizing_mode(asked.mode);
            let auto = self.auto();
            let shown = self.shown_inputs();
```

The controls' call passes the pause (spec 3.5), and its comment follows:

Old:
```rust
            // The cache keys on the design alone, so it is handed that design's own results:
            // `self.results` can lag the inputs by a frame (the header's assumptions reset edits
            // them after the last compute), and a lagging pull-out would stick in the lines until
            // the next edit. One compute_all (about 17 us) per frame of this tab; the neighbour
            // runs and the fit search rerun only when the design or the minimum changes.
            let shown_results = compute_all(&shown);
            let consequences = self.consequences.get(&shown, &shown_results, minimum);
```

New:
```rust
            // The cache keys on the design (with the minimum and the clamp pause), so it is
            // handed that design's own results: `self.results` can lag the inputs by a frame (the
            // header's assumptions reset edits them after the last compute), and a lagging
            // number would stick in the lines until the next edit. One compute_all (about 17 us)
            // per frame of this tab; the torque card's stepped runs and the fit search rerun only
            // when the design, the minimum or the pause changes.
            let shown_results = compute_all(&shown);
            let consequences =
                self.consequences
                    .get(&shown, &shown_results, minimum, self.clamps_paused);
```

`torque_card_ui`, before `view_ui` (spec 6.1); it takes the design shown, `shown`, from `view_ui` (plan review 4.4). `HeaderResponse` keeps the closure's return private until `.body(..)`, so the heading's click is recorded in a local and toggles the header before `.body(..)` (code review 5 SF-5):

Old:
```rust
    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
```

New:
```rust
    /// The torque card (slice 3 spec 6.1), between the Size tab's readouts and its drawing: a
    /// collapsing header, open, whose one wrapping row holds the heading, which opens and closes
    /// the card as its arrow does, and the Why link to how the coupling makes torque; under it
    /// the card's lines for `shown`, the design shown after this frame's edits (spec 3.5), weak
    /// like the lines under the controls. The card's own `compute_all` is defensive: the cache's
    /// key and the results it is handed come from one design whatever a later change puts between
    /// the panel's results and this view.
    fn torque_card_ui(&mut self, ui: &mut egui::Ui, readouts: &mut Readouts, shown: &DesignInputs) {
        let results = compute_all(shown);
        let auto = self.auto();
        let lines = self
            .consequences
            .get(
                shown,
                &results,
                self.minimum_diameter_mm,
                self.clamps_paused,
            )
            .card
            .lines(auto);
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            egui::Id::new(TORQUE_CARD_ID),
            true,
        );
        // The heading toggles the card as the arrow does. `HeaderResponse` keeps the closure's
        // return private until `body`, so the click is recorded here.
        let mut heading_clicked = false;
        let mut header = state.show_header(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                heading_clicked = ui
                    .add(egui::Label::new(CARD_HEADING).sense(egui::Sense::click()))
                    .clicked();
                if why_note_link(ui, SIZE_TAB_NOTE) {
                    readouts.open_note(SIZE_TAB_NOTE);
                }
            });
        });
        if heading_clicked {
            header.toggle();
        }
        header.body(|ui| {
            for line in &lines {
                weak_line(ui, line);
            }
        });
    }

    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
```

The card in place of the label (spec 6.1: the dashboard and the drawing say what it said). `view_ui` builds the design shown once, after the Required slider's edit, and hands it to the card and the drawing (plan review 4.4, plan deviation PD-6):

Old:
```rust
                ui.label("Build checks are shown in the dashboard. The geometry drawing fits the view; use the dimensions to compare sizes.");
                geometry_ui(
                    ui,
                    &self.shown_inputs(),
```

New:
```rust
                // The design shown after this view's one edit, the Required slider's above (slice
                // 3 spec 3.5): the card and the drawing below read this one copy, built once a
                // frame (with Auto on the ring size each build runs two geometries).
                let shown_after_edits = self.shown_inputs();
                self.torque_card_ui(ui, readouts, &shown_after_edits);
                geometry_ui(
                    ui,
                    &shown_after_edits,
```

`magcoupling-rs/src/gui/size_controls.rs`, the module doc (spec 9.2):

Old:
```rust
//! Under each control, its hint line ([`hint_line`], slice 2 spec 3.5) and its what-moves-what
//! lines (spec 3.3: [`Consequences`], computed by the guidance layer). The panel's centre view
//! draws the torque readouts (spec 3.2) with the texts here: [`REQUIRED`], [`HOT_LOW`] with
//! [`torque_gap_text`], [`pullout_label`], and [`NO_PREDICTION`] or [`NO_TORQUE`] where the model
//! makes none ([`predicts`]).
```

New:
```rust
//! Under each control, its hint line ([`hint_line`], slice 2 spec 3.5) and its what-moves-what
//! lines (spec 3.3: [`Consequences`], computed by the guidance layer); the lines for two magnets
//! fewer and two more are the torque card's count row since slice 3 (decision 7A). The panel's
//! centre view draws the torque readouts (spec 3.2) with the texts here: [`REQUIRED`],
//! [`HOT_LOW`] with [`torque_gap_text`], [`pullout_label`], and [`NO_PREDICTION`] or
//! [`NO_TORQUE`] where the model makes none ([`predicts`]), then the torque card.
```

- [ ] **Step 5: Run the tests and see them pass; plant nine faults**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_size_tab_draws_the_torque_card_in_place_of_the_label idle_frames_compute_the_card_once_and_an_edit_again idle_frames_on_the_first_view_change_no_input idle_frames_change_no_input the_size_tab_s_why_links_open_their_notes without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring size_tab_ the_first_magnet_line_follows_the_auto_state the_cache_computes_once_per_design_and_minimum_and_follows_an_edit no_guide_text_calls_the_ring_radius_ring_size without_a_breakdown_or_a_fit_there_is_no_line_for_it every_consequence_line_has_its_glyphs_and_plain_words the_count_row a_keyway_through_the_hub_changes_no_count_side_s_hot_low with_hub_wall_past_the_keyway_a_count_side at_either_end_of_the_magnet_slider_the_count_side with_no_prediction_on_the_design_shown with_magnets_too_short_for_the_model_the_card the_slider_minimum_is_the_preference a_drag_computes_the_card_once_for_each_design_it_draws a_required_edit_reaches_the_card_in_its_own_frame`

Expected: `test result: ok. 44 passed; 0 failed; 0 ignored; 0 measured; 923 filtered out` (the filter `size_tab_` also selects Task 3's `each_control_is_a_size_tab_slider_with_its_step`).

Each planted fault below is one edit, run with the filter given (`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- <filter>`), then undone. The red of Step 2 is a build failure, so these show that the new tests bite:

| Planted fault | Filter | Expected |
|---|---|---|
| (a) `ConsequenceCache::get`'s key drops the pause (`&& last.clamps_paused == clamps_paused` deleted) | `idle_frames_compute_the_card_once_and_an_edit_again the_cache_computes_once_per_design_and_minimum_and_follows_an_edit` | both fail. The cache test fails at the unpaused question: `left` is the paused card, `right` the unpaused one, whose flips add "Shaft clamp". Test 10 fails `the pause toggled`, `left: 2`, `right: 3` |
| (b) `Consequences::of` passes `crate::engine::clamps::PAUSED` to `card` in place of `clamps_paused` (the line `card: card(inputs, results, minimum_diameter_mm, clamps_paused),`) | `the_cache_computes_once_per_design_and_minimum_and_follows_an_edit` | fails at its last assertion, `assertion `left != right` failed`: the two cards at the clamp's limit are equal |
| (c) `magnet_count_line`: `None => Some(RING_SIZE_STAYS), Some(_) => None` (Auto on the length loses the ring-size line) | `the_first_magnet_line_follows_the_auto_state size_tab_first_magnet_line_follows_the_auto_state` | both fail at `Some(AxialLength)`: the guidance test `left: None`, `right: Some("The ring radius stays the same when you change the magnet count; the cup can still change size.")`; the panel test `left: 0`, `right: 1` |
| (d) `torque_card_ui`: the three lines `if heading_clicked { header.toggle(); }` deleted | `the_size_tab_draws_the_torque_card_in_place_of_the_label idle_frames_on_the_first_view_change_no_input idle_frames_change_no_input` | all three fail: the card test at `open false: The change in the model's hot-low torque after one step of one control, everything else held.` `left: 1`, `right: 0`; the idle tests at `Some(false): the card's caption shows exactly while it is open` and `open false` (`left: 1`, `right: 0`) |
| (e) `torque_card_ui`: `if why_note_link(ui, SIZE_TAB_NOTE) { readouts.open_note(SIZE_TAB_NOTE); }` becomes `why_note_link(ui, SIZE_TAB_NOTE);` | `the_size_tab_s_why_links_open_their_notes` | fails: `Why: How the coupling makes torque #2` (the card's link, the third copy, opens nothing) |
| (f) `inputs_ui` asks with `crate::engine::clamps::PAUSED` in place of `self.clamps_paused` (the cache's ask, `.get(&shown, &shown_results, minimum, self.clamps_paused);`) | `idle_frames_compute_the_card_once_and_an_edit_again` | fails `the pause toggled`, `left: 10`, `right: 6`: on an unpaused panel the controls and the card ask with two keys, each frame computing twice |
| (g) `torque_card_ui` asks the cache with `None` in place of `self.minimum_diameter_mm` (plan review 3.2) | `idle_frames_compute_the_card_once_and_an_edit_again` | fails `the Minimum OD set`, `left: 12`, `right: 5`: the controls ask with the minimum, the card without, two computations a frame |
| (h) in `view_ui`, `self.torque_card_ui(ui, readouts, &shown_after_edits);` becomes `self.torque_card_ui(ui, readouts, shown);` (plan review 4.4: the design before the Required edit) | `a_required_edit_reaches_the_card_in_its_own_frame` | fails at the Overall length row: `assertion `left == right` failed: Overall length: -1 mm -> -0.20 N·m (-8.8 %), Gearbox input torque: cold-high -0.07 N·m; +1 mm -> +0.20 N·m (+8.8 %), Hot-low torque: passes, Gearbox input torque: cold-high +0.07 N·m`, `left: 0`, `right: 1` |
| (i) in `view_ui`, `self.torque_card_ui(ui, readouts, &shown_after_edits);` becomes `self.torque_card_ui(ui, readouts, &self.inputs.clone());` (critique A-19: the card asks with the inputs, not the design shown) | `a_drag_computes_the_card_once_for_each_design_it_draws` | fails `left: 44`, `right: 20`: with Auto on, the card's design and the controls' differ, two computations a frame |

- [ ] **Step 6: The review dump's card block (`guidance_strings_for_review`)**

The `#[ignore]` test that Task 7 runs for the G-3 review. It gains the card at every design of `card_designs()`, after the "What moves what" block and before `// The Size tab constants.` This is Task 7's interface block, verbatim. It also gains a place of its own for the card heading's Why link (spec 9.2):

Old:
```rust
        // The Size tab constants.
        emit("size tab/no prediction", sc::NO_PREDICTION);
```

New:
```rust
        // The torque card (slice 3; spec sections 4 and 10.1 step 2), for its G-3 review: the
        // heading, then the lines `TorqueCard::lines` draws at every design of `card_designs()`,
        // each in its own Auto state as the panel draws it, at the shipped clamp pause; then every
        // Auto state at the four designs whose card takes a form of its own under Auto (the odd
        // count has no size rows, so Auto on ring size or magnet length holds no row; the three
        // one-line forms draw no Auto line: T-12, test 8). Tag: "card/<key>/<Auto state>/<line>",
        // the line numbered from 00.
        {
            use crate::gui::test_support::card_designs;
            use crate::gui::torque_card::{CARD_HEADING, card};
            let state = |auto: Option<FreeVariable>| {
                auto.map_or_else(|| "Auto off".to_owned(), |v| format!("Auto {v:?}"))
            };
            emit("card/heading", CARD_HEADING);
            let cards: Vec<_> = card_designs()
                .into_iter()
                .map(|d| {
                    let shown = card(
                        &d.inputs,
                        &compute_all(&d.inputs),
                        d.minimum_diameter_mm,
                        crate::engine::clamps::PAUSED,
                    );
                    (d, shown)
                })
                .collect();
            for (d, shown) in &cards {
                for (n, line) in shown.lines(d.auto).iter().enumerate() {
                    emit(&format!("card/{}/{}/{n:02}", d.key, state(d.auto)), line);
                }
            }
            let design = |key: &str| {
                cards
                    .iter()
                    .find(|(d, _)| d.key == key)
                    .unwrap_or_else(|| panic!("card_designs() has no design {key:?}"))
            };
            for key in [
                "odd count",
                "short magnets",
                "shown does not fit",
                "variation 1.0",
            ] {
                let (_, shown) = design(key);
                for variable in FreeVariable::ALL {
                    for (n, line) in shown.lines(Some(variable)).iter().enumerate() {
                        emit(
                            &format!("card/{key}/{}/{n:02}", state(Some(variable))),
                            line,
                        );
                    }
                }
            }
            // The readout's gap above the card, in the card's number format (decision 10B),
            // for the review to read in context.
            for key in ["(a)", "(b)", "(c)", "(e)", "(e2)"] {
                let (d, _) = design(key);
                let results = compute_all(&d.inputs);
                let (gap, _) = sc::torque_gap_text(
                    results.metal.torque_hot_low_Nm,
                    d.inputs.metal.required_min_Nm,
                );
                emit(&format!("size tab/torque gap/{key}"), &gap);
            }
        }

        // The Size tab constants.
        emit("size tab/no prediction", sc::NO_PREDICTION);
```

Old:
```rust
            (
                "size tab/under Magnets per ring and Outside diameter",
                SIZE_TAB_NOTE,
            ),
```

New:
```rust
            (
                "size tab/under Magnets per ring and Outside diameter",
                SIZE_TAB_NOTE,
            ),
            ("size tab/the torque card's heading", SIZE_TAB_NOTE),
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > magcoupling-rs/target/t5-guidance.txt 2>&1; grep -c '^\[card/' magcoupling-rs/target/t5-guidance.txt; grep -n -e '^\[card/heading\]' -e '^\[size tab/torque gap/' -e "^\[why link/size tab/the torque card's heading\]" -e '^\[size tab/under Magnets per ring' -e '^TOTAL STRINGS' magcoupling-rs/target/t5-guidance.txt`

Expected: `260` card lines (the heading and 259 `[card/<key>/` lines, at Task 4's 52 designs), then these lines, each after its line number in the dump (217, 218, 251, 511 to 515, 556 and 570); the dump's own run reads `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 966 filtered out`:
```text
[size tab/under Magnets per ring/Auto None] The ring radius stays the same when you change the magnet count; the cup can still change size.
[size tab/under Magnets per ring/Auto Some(AxialLength)] The ring radius stays the same when you change the magnet count; the cup can still change size.
[card/heading] One step of each size control
[size tab/torque gap/(a)] short 0.21 N·m (-8.6 %)
[size tab/torque gap/(b)] margin +0.00 N·m (+0.0 %)
[size tab/torque gap/(c)] short 1.06 N·m (-42.3 %)
[size tab/torque gap/(e)] short 0.83 N·m (-33.1 %)
[size tab/torque gap/(e2)] margin +0.17 N·m (+11.5 %)
[why link/size tab/the torque card's heading] Why: How the coupling makes torque
TOTAL STRINGS: 564
```
`TOTAL STRINGS` is 305 plus the `[card/<key>/` lines (Task 7's formula): 311 at 69fa505, less the 12 neighbour lines and the first line under Magnets per ring with Auto on the ring size, plus the heading, the five gap lines and the new Why-link place: 305 + 259 = 564.

- [ ] **Step 7: Docs: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`**

Twenty edits. Each Old but (c)'s is a substring of one line; replace only it. Texts the card draws are quoted as drafted; Task 7 updates any it rewords (its Step 5).

(a) README, the Layout table's `src/gui/` row, `guidance.rs`'s entry:

Old:
```text
`design_status`, `Consequences`, `ConsequenceCache`, `magnet_count_line`,
```

New:
```text
`design_status`, `Consequences` (the lines under the Size tab's controls and its torque card, `card`), `ConsequenceCache` (one key: the design shown, the Minimum OD and the shaft-clamp pause), `magnet_count_line`,
```

(b) The same row, `panel.rs`'s entry:

Old:
```text
the assumptions banner, the welcome strip, the sizing controls)
```

New:
```text
the assumptions banner, the welcome strip, the sizing controls, the Size tab's torque card between the readouts and the drawing (`torque_card_ui`, slice 3))
```

(c) The Size tab's lines under the controls (spec 9.2's `:214-226`), and a paragraph on the card (several lines):

Old:
```text
Under the controls, short lines say what moves what, each read from the engine (the neighbour
magnet counts from `compute_all` of the design with only `coupling.npole` changed; the
diameters and lengths from `engine::size_controls`), never typed. For **Magnets per ring**:
a first line that follows the Auto state (`guidance::magnet_count_line`): with Auto off, "The
ring radius stays the same when you change the magnet count; the cup OD below can still
change."; with Auto on the ring radius, "Auto resizes the ring when you change the magnet count;
the lines below keep the present ring radius."; with Auto on the magnet length, "Auto resets the
magnet length when you change the magnet count; the lines below keep the present length."; and
none with Auto on the magnet count, whose slider is locked. Then the pull-out and cup OD at N-2 and N+2 inside the slider range (a neighbour whose
blocks do not fit, or whose end effect is out of range, reads in words; one that fails only the
keyway still shows its torque, since the keyway does not depend on the count), and the
smallest outside diameter that fits N magnets (where only the Minimum OD floor stops it, the
line says how far it could go and that lowering Minimum OD reaches it). For **Outside
diameter**: "Set by: cap A, cup B, boss C mm", the largest being the outside diameter. For
**Overall length**: "Magnets L mm + housing H mm".
```

New:
```text
Under the controls, short lines say what moves what, each read from the engine (the diameters
and lengths from `engine::size_controls`), never typed. For **Magnets per ring**: a first line
that follows the Auto state (`guidance::magnet_count_line`, slice 3 spec 4.3): with Auto off and
with Auto on the magnet length, "The ring radius stays the same when you change the magnet
count; the cup can still change size."; none with Auto on the ring radius (the torque card's
Auto line says Auto sizes the ring again after a change) or on the magnet count, whose slider is
locked. Then the smallest outside diameter that fits N magnets (where only the Minimum OD floor
stops it, the line says how far it could go and that lowering Minimum OD reaches it). For
**Outside diameter**: "Set by: cap A, cup B, boss C mm", the largest being the outside
diameter. For **Overall length**: "Magnets L mm + housing H mm". The lines for two magnets
fewer and two more moved to the torque card's count row (slice 3, decision 7A).

**The torque card** (slice 3, `gui::torque_card`; spec
`docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md`) sits between the readouts
and the drawing, in place of a label that pointed at the dashboard: a collapsing header, "One
step of each size control", whose heading opens and closes it as its arrow does, with the
"Why: How the coupling makes torque" link after it; under it a caption and one row per Size
control (Outside diameter, Overall length, Magnet face gap, Magnets per ring), each the change in
the model's hot-low torque after one step each way (1 mm, 1 mm, 0.1 mm, 2 magnets) through the
control's own slider mapping, with what the step costs, or words where a side has no number.
While Auto is on, an Auto line says Auto sets the size again after a change, and the held
control's row reads "Set by Auto; turn Auto off to edit". The panel draws it for the design shown
after the frame's edits, from the `ConsequenceCache` (one key: the design, the Minimum OD and the
shaft-clamp pause; the Auto marking is applied outside the cache).
```

(d) Where the Why links sit (spec 9.2's `:362`):

Old:
```text
sits under Magnets per ring and Outside diameter on the Size tab (to the note on how the coupling makes torque)
```

New:
```text
sits under Magnets per ring and Outside diameter on the Size tab and on the torque card's heading (to the note on how the coupling makes torque)
```

(e), (f), (g) The Tests table's `src/gui/panel.rs` row: the slice-2 T19 clause's places, the odd count's links, and Task 5's tests at the row's end:

Old:
```text
the Size tab's Why links under Magnets per ring and Outside diameter open the torque note
```

New:
```text
the Size tab's Why links under Magnets per ring and Outside diameter, and on the torque card's heading (slice 3), open the torque note
```

Old:
```text
with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link is;
```

New:
```text
with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link and the torque card's are;
```

Old:
```text
the gearbox check's row once with a red badge (`table_badges_on`, shared with the end-effect test) |
```

New:
```text
the gearbox check's row once with a red badge (`table_badges_on`, shared with the end-effect test); slice 3 (T5): the Size tab draws the torque card in place of the label under the readouts, its heading and each of its lines once, one under the next, a click on the heading closing and opening it as one on its arrow does (`the_size_tab_draws_the_torque_card_in_place_of_the_label`); a typed Required value reaches the card in the frame that applies it (`a_required_edit_reaches_the_card_in_its_own_frame`); under Magnets per ring two lines, the ring-size line with Auto off or on the magnet length and the fit line, the card drawing the Auto line of each Auto state, the count's steps (at 4 and 40 magnets the side past the slider reading its end) and following each edit; idle frames compute the card once, a gap edit, a clamp-pause toggle and a new Minimum OD once each (the drawn Outside diameter row then reading its slider's end), and turning Auto off at a solved length or Auto on the count at its best computes nothing (`idle_frames_compute_the_card_once_and_an_edit_again`); a drag of the gap slider with Auto on the magnet length computes the card once for each design it draws (`a_drag_computes_the_card_once_for_each_design_it_draws`); idle frames change no input with the card open and closed, at the clamp pause |
```

(h), (i) The `src/gui/guidance.rs` row: its neighbour clauses go (spec 9.2, code review 3 S5), the cache's key and the ring-size wording follow:

Old:
```text
the consequence numbers equal `compute_all` of the neighbour design and follow an input edit; neighbours outside 4 to 40 magnets are not shown;
```

New:
```text
the consequence lines follow an input edit, and the `ConsequenceCache` computes once per design, Minimum OD and shaft-clamp pause, the pause reaching the torque card (slice 3); the lines under Magnets per ring are the ring-size line, with Auto off or on the magnet length, and the fit line (slice 3: the neighbour lines' tests moved to the torque card's count row);
```

Old:
```text
no guide, fix, Problem or consequence text says "ring size"
```

New:
```text
no guide, fix, Problem or consequence text, nor the torque card's Auto lines, says "ring size"
```

(j) The `src/gui/torque_card.rs` row (Task 3's): the moved tests, at the row's start:

Old:
```text
| `src/gui/torque_card.rs` (feature `gui`) |
```

New:
```text
| `src/gui/torque_card.rs` (feature `gui`) | The count row is the neighbour lines that stood under Magnets per ring (decision 7A; their tests moved here from `gui/guidance.rs`, rewritten to the card's types and hot-low): each side is the design with only the count changed, or words where its blocks do not fit; at 4 and 40 magnets the side past the slider reads its end; the count row skips the keyway (D-1); a count side reads a number exactly when sizing counts its design; with no prediction on the design shown, or magnets too short for the model, the card compares nothing and draws its one line.
```

(j2) README, the Layout table's `src/gui/` row, Task 3's `torque_card.rs` entry: the N-2 and N+2 lines no longer call them.

Old:
```text
both moved from the N-2 and N+2 lines, which call them;
```

New:
```text
both moved from the N-2 and N+2 lines that the count row replaced;
```

(k) `docs/ai/02-system.yaml`, `:115` (reachability, spec 9.2, code review 2 MF-3):

Old:
```text
sit under Magnets per ring and Outside diameter (guidance::SIZE_TAB_NOTE)
```

New:
```text
sit under Magnets per ring and Outside diameter and on the torque card's heading (guidance::SIZE_TAB_NOTE; slice 3)
```

(l), (m), (n) `:116`: the lines under the controls, the card, the cache's key. The "under 1.6 ms" figure stays for Task 10's measurement:

Old:
```text
Under the controls, Consequences::of says what moves what: the neighbour magnet counts N-2 and N+2 inside the slider range (compute_all of the design with only coupling.npole changed; decision D-1: a neighbour that fails only the keyway still shows its torque, and one reads in words only when its blocks do not fit, its end effect is out of range or its torque is not finite), a first line that follows the Auto state (guidance::magnet_count_line: RING_SIZE_STAYS with Auto off, The ring radius stays the same when you change the magnet count; the cup OD below can still change.; AUTO_RESIZES_RING with Auto on the ring size; AUTO_RESETS_LENGTH with Auto on the magnet length; none with Auto on the magnet count, whose slider is locked), the smallest outside diameter
```

New:
```text
Under the controls, Consequences::of says what moves what: a first line that follows the Auto state (guidance::magnet_count_line, slice 3 spec 4.3: RING_SIZE_STAYS with Auto off and with Auto on the magnet length, The ring radius stays the same when you change the magnet count; the cup can still change size.; none with Auto on the ring size, whose rule the torque card's Auto line takes, or on the magnet count, whose slider is locked), the smallest outside diameter
```

Old:
```text
reusing the expressions dimensions uses).
```

New:
```text
reusing the expressions dimensions uses). Between the readouts and the drawing the torque card (slice 3, gui::torque_card, drawn by MagcouplingPanel::torque_card_ui in place of a label; spec docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md): a collapsing header (CARD_HEADING) whose heading toggles it as its arrow does, with the Why link to SIZE_TAB_NOTE, then for each Size control the change in hot-low torque after one step each way through the control's own slider mapping, with its costs, or words; its count row is the N-2 and N+2 lines that stood under Magnets per ring (decision 7A). Consequences.card holds it, computed for the design shown after the frame's edits (its own compute_all, defensive); the Auto marking (the held row, the Auto line) is applied outside the cache.
```

Old:
```text
A ConsequenceCache recomputes only when the inputs or the Minimum OD change; the FixCache likewise only when the inputs change
```

New:
```text
A ConsequenceCache recomputes only when the inputs, the Minimum OD or the shaft-clamp pause change (one key in three parts, slice 3, as the FixCache keys on the pause); the FixCache likewise only when the inputs or the pause change
```

(o) `docs/ai/03-structure.yaml`, the `gui` line, `guidance.rs`'s clause. It drops `Neighbour`, `NeighbourOutcome`, "neighbours N-2 and N+2" and the two Auto constants (spec 9.2, code review 3 S5):

Old:
```text
Consequences::of with magnet_lines, diameter_line and length_line (neighbours N-2 and N+2, the fit, the diameter and length breakdowns), Neighbour, NeighbourOutcome, ConsequenceCache, magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS, AUTO_RESIZES_RING, AUTO_RESETS_LENGTH, or none with Auto on the count),
```

New:
```text
Consequences::of(inputs, results, minimum_diameter_mm, clamps_paused) with magnet_lines, diameter_line and length_line (the torque card, Consequences.card, the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count),
```

(p) The same line, `panel.rs`'s clause:

Old:
```text
the Key design group's Auto status line (AUTO_STATUS_START) and run_sizing)
```

New:
```text
the Key design group's Auto status line (AUTO_STATUS_START), the Size tab's torque card (torque_card_ui, slice 3: a collapsing header under TORQUE_CARD_ID between the readouts and the drawing, the heading toggling it as its arrow does, the Why link to SIZE_TAB_NOTE, the card's lines for the design shown after the frame's edits; auto(), the variable Auto sets, shared with the controls) and run_sizing)
```

(q) The same line, `test_support.rs`'s clause (after Task 2's designs):

Old:
```text
an Auto design solved as the panel solves it), the lookups
```

New:
```text
an Auto design solved as the panel solves it); end_effect_edge (moved from gui/guidance.rs's tests in slice 3: the end effect just out of range while two more magnets bring it back in, read by those tests and the torque card's count-row tests), the lookups
```

(r) `manual_size_gui` (spec 9.2's `:73`):

Old:
```text
above its consequence lines"
```

New:
```text
above its consequence lines; since slice 3 the torque card between the readouts and the drawing (gui::torque_card, drawn by panel.rs torque_card_ui and held in the same ConsequenceCache), and under Magnets per ring the ring-size line and the fit line only (the lines for two magnets fewer and two more are the card's count row)"
```

(r2) `docs/ai/03-structure.yaml`, the `gui` line, Task 3's `torque_card.rs` clause, likewise.

Old:
```text
side_words and count_step moved from the N-2 and N+2 lines, which call them;
```

New:
```text
side_words and count_step moved from the N-2 and N+2 lines that the count row replaced;
```

Run: `python -c "import yaml; yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8')); t=open('docs/ai/02-system.yaml', encoding='utf-8').read().split(chr(10)); [print(type(yaml.safe_load([x for x in t if k in x][0].strip())).__name__) for k in ['The first view is Size and torque', 'Reachability (guidance slice 2']]"`

Expected: `list` twice. (`docs/ai/03-structure.yaml` parses whole. `docs/ai/02-system.yaml` does not parse whole with PyYAML at 69fa505 either: line 15 is pre-existing. So its two edited lines are checked as Task 1 checks its line.)

- [ ] **Step 8: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `961 passed; 0 failed; 6 ignored` (Task 4's 957 plus 4: the seven neighbour tests leave `gui/guidance.rs` and arrive in `gui/torque_card.rs`, and four panel tests are new). The two app binaries run 0 tests. `explain` 64 passed, 1 ignored. Every other binary is as in Task 1 Step 9: assumptions 13, deviations 54, differential 19, gearbox 13, grades 12, material_library 4, material_links 14, parity 4, python_schema 7, robustness 13, schema 11, shaft_joint 40, size_controls 38, sizing 45, static_data 8; doc-tests 1 passed, 1 ignored.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: lib `234 passed; 0 failed; 1 ignored` (no GUI test runs without `--features app`); every other binary as above, `explain` 63 passed, 1 ignored.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`, then the same without `--features app`.

Expected: `Finished`, no warning, both.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/02-system.yaml
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/gui/guidance.rs
 M magcoupling-rs/src/gui/panel.rs
 M magcoupling-rs/src/gui/size_controls.rs
 M magcoupling-rs/src/gui/test_support.rs
 M magcoupling-rs/src/gui/torque_card.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
```

- [ ] **Step 9: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T5.txt` with the Write tool:
```text
feat(magcoupler): the torque card on the Size tab, from the consequences cache (BL-073 T5)

The Size tab draws the torque card between its readouts and its drawing, in
place of the label that pointed at the dashboard (spec 6.1): a collapsing
header whose heading opens and closes it as its arrow does, with the Why link
to how the coupling makes torque, and the card's lines for the design shown
after this frame's edits (spec 3.5), one copy of which the card and the drawing
read. Open for now; Task 6 decides open or closed.

The card lives in the ConsequenceCache: Consequences.card replaces the
neighbours, and Consequences::of and ConsequenceCache::get take the shaft-clamp
pause, one key in three parts (spec 3.2); both panel call sites pass the
panel's pause. The Auto marking stays outside the cache (spec 3.4).

The N-2 and N+2 lines leave Magnets per ring (owner 7A): their tests move to the
card's count row, rewritten to its types and hot-low, and Neighbour,
NeighbourOutcome, Neighbour::line and prediction leave gui/guidance.rs (the
two neighbour words moved in Task 4). The first line under Magnets per ring
follows spec 4.3:
RING_SIZE_STAYS (its clause now "the cup can still change size") with Auto off
or on the magnet length, none with Auto on the ring size or the count;
AUTO_RESIZES_RING and AUTO_RESETS_LENGTH are removed, the card's Auto lines
taking their rule.

Tests: test 10 (the card computed once per design, a gap edit, a pause toggle
and a new Minimum OD once each, the drawn diameter row following the Minimum
OD, Auto toggles that keep the design shown none); a drag computes the card
once for each design it draws; a typed Required value reaches the card in its
own frame; test 11 (idle frames change no input with the card open and
closed); the card drawn in place of the label; the Size tab's Why links 3 and
2 to the torque note; the moved-line tests and their counts; the review dump
gains the card at every shared design. The README and docs/ai/02-system.yaml
and 03-structure.yaml follow.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/size_controls.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/src/gui/torque_card.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T5.txt
git status --short
```

Expected: one commit of 8 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 5 moves** (each confirmed in the prototype), and its tests beyond the spec's:

| Test | Old -> new | Why |
|---|---|---|
| `gui::guidance::tests::each_neighbour_is_the_design_with_only_the_magnet_count_changed` | moved to `gui::torque_card::tests::the_count_row_s_sides_are_the_design_with_only_the_magnet_count_changed`: each side is `compute_all` of the design with only the count changed, in hot-low (`hot_low_Nm`, `change_Nm`), or `Words::DoesNotFit`; the owner's case kept (-2 fits and loses hot-low, the cup grows; +2 does not fit) | spec 6.3, T-2, 7A: the neighbour lines are the card's count row |
| `gui::guidance::tests::the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not` | moved to `the_count_row_says_each_side_s_hot_low_change_or_why_not`: the count row's line in the Size tab's format (`signed_change` from `TORQUE_DECIMALS`, `signed_percent`: a row with one number side is at that side's own decimals), `+2 -> SIDE_DOES_NOT_FIT`, and two lines left under Magnets per ring | the same; the line is the card's now |
| `gui::guidance::tests::at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown` | moved to `at_either_end_of_the_magnet_slider_the_count_side_past_it_reads_its_end`: at 4 magnets `-2`, at forty narrow magnets `+2` reads `Words::EndOfRange` and `END_OF_RANGE`; the inside side is that count's design; Task 3's odd-count assertion comes along (at `odd_count()` the first line under Magnets per ring is `RING_SIZE_STAYS` and no line names a count) | the same; 40 default blocks do not fit, so the card would be `ShownDoesNotFit` there |
| `gui::guidance::tests::a_keyway_through_the_hub_changes_no_neighbour_s_reading` | moved to `a_keyway_through_the_hub_changes_no_count_side_s_hot_low`: the count sides' hot-low numbers and words are the same with the keyway through or intact (D-1); `-2` is a number | the same; the flips and margins may differ (the shaft joints read the keyway), so only the hot-low numbers are compared |
| `gui::guidance::tests::with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it` | moved to `with_hub_wall_past_the_keyway_a_count_side_reads_a_number_exactly_when_sizing_counts_it`, at the defaults and `end_effect_at_a_step()` (a design shown that predicts while a count side does not), not vacuous (both outcomes occur) | the same; short magnets and `end_effect_edge` predict nothing on the design shown, which then has no rows (test 0) |
| `gui::guidance::tests::a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented` | moved to `with_no_prediction_on_the_design_shown_the_card_invents_no_change`: at `end_effect_edge()` (two more magnets predict) the card is `ShownNoPrediction`, no row, its one line in every Auto state | the same; 9A: no numbers at all while the design shown has no prediction |
| `gui::guidance::tests::with_magnets_too_short_for_the_model_every_neighbour_reads_words` | moved to `with_magnets_too_short_for_the_model_the_card_names_no_count`: `ShownNoPrediction`, its one line, no line under Magnets per ring names a count | the same |
| `gui::guidance::tests::neighbours_are_even_counts_inside_the_slider_and_never_overflow` | not Task 5's: it calls `neighbour_counts`, which Task 3 moves into `count_step` (test 3 keeps it for `count_step`, spec 8.1) | Task 3 moved it |
| `gui::guidance::tests::the_cache_computes_once_per_design_and_minimum_and_follows_an_edit` | every `get` and `of` takes the pause; reads `card` (the count row's `-2` hot-low) in place of `neighbours`; the pause as a third key part; the pause reaches the card at the clamp's limit | spec 3.2, 6.3 (code review 5 SF-2) |
| `gui::guidance::tests::without_a_breakdown_or_a_fit_there_is_no_line_for_it` | `card: TorqueCard::Rows(Vec::new())` in place of `neighbours: Vec::new()`; assertions unchanged | spec 6.3 (code review 4 SF-3) |
| `gui::guidance::tests::the_first_magnet_line_follows_the_auto_state` | rewritten to 4.3: `RING_SIZE_STAYS` with Auto off and on the length, none on the ring or the count; the line counts 2, 2, 1, 1 | spec 4.3, 6.3 |
| `gui::guidance::tests::no_guide_text_calls_the_ring_radius_ring_size` | lists `AUTO_HOLDS_LENGTH`, `AUTO_HOLDS_RING`, `AUTO_HOLDS_COUNT` in place of the two removed constants | spec 6.3, 4.3: the card's Auto lines take their rule |
| `gui::guidance::tests::every_consequence_line_has_its_glyphs_and_plain_words` | comment only (the line forms left under the controls); passes | spec 6.3; the card's lines are held to the glyphs and the guards by tests 16 and 17 |
| `the_fit_line_names_a_diameter_that_fits_and_one_step_below_does_not`, `the_fit_line_s_edge_forms_name_the_ends_of_the_diameter_range`, `with_minimum_od_unset_the_fit_line_says_how_far_it_could_go_and_what_unlocks_it`, `at_8_magnets_the_ring_size_stays_and_the_cap_sets_the_diameter` (`gui::guidance::tests`) | unchanged, pass | they read the fit line by `.pop()` / `.last()` and the first line by index: no count of lines (plan deviation PD-35); the counts that move are pinned in the tests above and below |
| `gui::guidance::tests::the_size_tab_and_banner_links_lead_where_their_checks_do` | comment names the card heading's link | spec 9.2 |
| `gui::guidance::tests::guidance_strings_for_review` (`#[ignore]`) | gains the card block and the heading's Why-link place | spec 6.3, Task 7's interface |
| `gui::guidance::tests::consequences_of` (helper, 13 callers) | passes `clamps::PAUSED` | spec 6.3 (code review 5 SF-2) |
| `gui::guidance::tests::end_effect_edge` (helper) | moved to `gui::test_support::end_effect_edge` | plan deviation PD-29: read by guidance's glyph test and the card's moved no-prediction test |
| `gui::size_controls::tests::the_slider_minimum_is_the_preference_else_the_range_start_or_the_diameter_shown` (Task 2's) | its `Consequences::of` passes `clamps::PAUSED` | the new signature (a third call site spec 6.3 does not list: Task 2 added it) |
| `gui::panel::tests::shown_consequences` (helper, 11 callers) | passes `harness.panel.clamps_paused` | spec 6.3 (code review 5 SF-2) |
| `gui::panel::tests::the_size_tab_s_why_links_open_their_notes` | 3 links to `SIZE_TAB_NOTE`, 2 to `TORQUE_BAND_NOTE`, each clicked | spec 6.3 (code review 2 MF-3) |
| `gui::panel::tests::without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring` | 1 -> 2 links | the same; the odd count keeps the gap and count rows (T-5) |
| `gui::panel::tests::size_tab_draws_each_what_moves_what_line_under_its_control` | pins 2 lines under Magnets per ring | spec 6.3's "4 -> 2" at (a) |
| `gui::panel::tests::size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter` | pins 2 lines at 8 magnets, and the card's lines drawn | spec 6.3's "4 -> 2" at (e); the owner's case on screen |
| `gui::panel::tests::size_tab_first_magnet_line_follows_the_auto_state` | rewritten to 4.3: the ring-size line per state, the card's Auto line per state, the line counts | spec 4.3, 6.3 |
| `gui::panel::tests::size_tab_what_moves_what_lines_follow_each_edit` | the card's lines follow each edit (drawn, the stale ones gone) | spec 6.3: the neighbour lines it followed are the card's |
| `gui::panel::tests::size_tab_magnet_slider_ends_show_only_the_neighbour_in_range` | reads the card's count row: `-2` at 4 and `+2` at forty narrow magnets read `END_OF_RANGE`; 2 lines under the control; no "At 2:" / "At 42:" text | spec 6.3's "3 -> 2" at a slider end |
| `gui::panel::tests::size_tab_new_texts_have_glyphs_and_plain_words` | the card's heading and lines join the new texts | spec 6.3 |
| `gui::panel::tests::size_tab_narrow_window_wraps_the_new_lines_inside_their_sides`, `size_tab_smallest_diameter_line_searches_with_the_minimum_od` | unchanged, pass | they read `magnet_lines` (now two lines) and the fit line; nothing they pin moved |
| `gui::panel::tests::idle_frames_on_the_first_view_change_no_input`, `idle_frames_change_no_input` | test 11: the card open and closed, at the shipped pause | spec 8.2 |
| `gui::panel::tests::idle_frames_compute_the_card_once_and_an_edit_again` (test 10) | new | spec test 10, with the key's third part, the Minimum OD (plan review 3.2) |
| `gui::panel::tests::a_drag_computes_the_card_once_for_each_design_it_draws` | new | critique A-19 (RF-2): decision 13A's one computation per drawn design under Auto |
| `gui::panel::tests::a_required_edit_reaches_the_card_in_its_own_frame` | new | plan review 4.4: spec 3.5's same-frame rule |

---

### Task 6: The card opens or closes for the view, and keeps the drawing's floor

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec section 6.2 in full, 6.1's arrow and heading as the user's choice, 8.5 and tests 19 to 25; owner decision 4A ("collapsible, open by default; starts closed while the Equation panel is open or the window is narrow; the drawing keeps its 160 pt floor"), T-7 and T-10.

- Two session fields, view state like `welcome_dismissed` (in no file, link or undo step, kept by Reset all and every load). `card_choice: Option<bool>` is the user's own open or close, set by a click on the arrow or on the heading; it wins for the session. `card_start: Option<(CardView, bool)>` is the default and the view it was decided for.
- `CardView { equation_open, screen_width, screen_height }`: the Equation panel's state and `ctx.screen_rect()` rounded to whole points. `centre_ui` reads it once at its top, with the centre region's width, as a `CardFrame` it hands `view_ui` and the card. A design change, or a side panel widening to its content, never changes the view, so it never decides the default again (code review 3 MF-2).
- `card_default`, while the user has made no choice and the view differs from `card_start`'s: open iff (rule 1) the Equation panel is closed, (rule 2) the region is at least `CARD_OPEN_MIN_WIDTH` = 400 pt wide, and (rule 3) `available_height - card_open_height(ui, lines) - item_spacing.y >= drawing_floor(ui)` at the card.
- `card_open_height(ui, lines)`, a free function in `gui/panel.rs`, measures the open card before it is drawn. The header row is laid out as drawn: the heading, then the Why link continuing after it, wrapping under it in a narrow region, as egui 0.32.3's `widgets/label.rs:192-213` lays out a label in a wrapping row. Each line is a galley wrapped at the body's width (the region less the indent), with the item spacing.
- `list_row(ui)` and `drawing_floor(ui)` in `gui/geometry_view.rs`: one expression for the callout list's row, which `geometry_ui` now reads, and for the room the drawing needs to keep `MIN_DRAWING_HEIGHT` with `MIN_LIST_ROWS` rows (215.1 pt in the app's style, 212.1 in the harness's).
- The body scrolls in a `ScrollArea` capped at `max(available - header - 2 × item_spacing.y - drawing_floor, one body row)`, measured inside the body as what is left under the header less one spacing and the floor. The cap is also the body's least height while it scrolls (`min_scrolled_height`): egui's default, 64 pt, would hold a scrolling body taller than the cap and push the drawing under its floor beside the Equation panel (plan review 3.1).
- `test_support::drawing_rect(output)` is the layout tests' probe of the drawing: the one rect with the extreme background colour and 4 pt corners. No production field is added for a test: `view_ui` keeps dropping `geometry_ui`'s `GeometryLayout` (code review S11).
- `geometry_view::MIN_DRAWING_HEIGHT` becomes `pub(crate)`, so the tests compare with it, not a typed 160 (plan deviation PD-32).

Facts the implementer meets on the way:
- The red is a build failure: the tests read the new fields, constants and functions. Eleven planted faults in Step 5 show each rule and pin bites.
- Every layout test sets the shaft-clamp pause before the card's first frame, which is the frame that decides its default (code review 3 S3). It does so through one helper, `card_tab`: start on the Geometry tab, set the pause, the design (an Auto design solves there) and the Equation panel, then click the Size tab. The helper returns the deciding frame's output and steps past the collapse animation.
- The headless harness draws in egui's default style (item spacing 8 x 3), not the app's (6 x 4), so the tests assert outcomes with margin in both styles and pin no figure measured in the app's style (spec 6.2). The figures below are the harness's, from the plan-time prototype. They equal spec 6.2 and 8.5's harness figures: at 1280 x 800 the open card is 173 / 190 pt at (a) / (b), bodies 152 / 169 under a cap of 371.9. At 930 x 1024 it is 355 / 400, bodies 320 / 365 under 473.9 / 459.9. At 930 x 800 the bodies are capped at 249.9 / 235.9. At 1080 x 745 rule 3 leaves -83.1 at no back iron and +153.9 at short magnets. At 1080 x 800 it leaves +83.9 at (a) and -28.1 at (c). The header is 18 pt, and 32 pt wrapped at 930 wide, against the replaced label's 28 and 42. The pre-measure equals the drawn height to 0.0 pt.
- Spec 8.5 test 21 at 1280 x 800 cannot tell rule 1 apart: there the open Equation panel leaves so little room that rule 3 alone closes the card (a planted fault removing rule 1 passed it). So test 21 adds a 1280 x 2000 case, where rules 2 and 3 alone would open the card (the test asserts the room), and only rule 1 closes it (plan deviation PD-33).
- Line numbers are after Task 5. Trust the code. Every code block is rustfmt's layout: replaying this task's Old/New blocks on Task 5's commit reproduces the task's six files byte for byte (CRLF normalised; the plan's replay check).

**Files:**
- Modify: `magcoupling-rs/src/gui/panel.rs` (`CARD_OPEN_MIN_WIDTH`, `CardView`, `CardFrame`, the fields `card_choice` and `card_start`, `centre_ui`, `view_ui`, `torque_card_ui`, `card_default`, `card_open_height`), `magcoupling-rs/src/gui/geometry_view.rs` (`MIN_DRAWING_HEIGHT` `pub(crate)`, `list_row`, `drawing_floor`, `geometry_ui`'s row)
- Modify (tests): `magcoupling-rs/src/gui/panel.rs`, `magcoupling-rs/src/gui/geometry_view.rs`, `magcoupling-rs/src/gui/test_support.rs` (`drawing_rect`)
- Modify (docs): `magcoupling-rs/README.md` (the `src/gui/` module list's `panel.rs` and `geometry_view.rs` entries; the card paragraph Task 5 wrote; the Tests table's `src/gui/panel.rs` row and the `src/gui/geometry.rs`, `src/gui/geometry_view.rs` row), `docs/ai/02-system.yaml` (`:116`), `docs/ai/03-structure.yaml` (the `gui` line's `geometry_view.rs`, `panel.rs` and `test_support.rs` clauses)
- Test: `magcoupling-rs/src/gui/panel.rs` (tests 19, 20 with 20's Why-link half, 20a, 21, 22, the resize test (`the_card_is_decided_again_when_the_window_is_resized_with_no_user_choice`, plan review 3.3), 23, 24, 25; `no_centre_view_paints_or_takes_clicks_in_the_equation_panel` with the card open and closed), `magcoupling-rs/src/gui/geometry_view.rs` (`the_drawing_floor_is_the_room_that_keeps_the_drawing_over_three_list_rows`)

**Interfaces:**
- Consumes: Task 5's `torque_card_ui`, `TORQUE_CARD_ID`, `auto()`, `shown_consequences`, `REPLACED_LABEL`; Task 4's `CARD_HEADING`, `TorqueCard::lines`, and in `gui/panel.rs` `EXPLORATION_ONLY` with `SIDE_DOES_NOT_FIT` (test 23 builds the exploration line from them); `gui::dashboard::why_note_text`; `gui::explorer::reviewed_note`; `gui::test_support::{no_back_iron, short_magnets, with_poles, region_frame, last_characters, text_rects, smoke_inputs}` (`smoke_inputs`, Task 3's, builds the tests' `smoke_design()`: critique A-24); egui 0.32.3's `CollapsingState`, `HeaderResponse`, `ScrollArea`, `WidgetText::into_layout_job`.
- Produces:
  - `gui::panel` (private): `const CARD_OPEN_MIN_WIDTH: f32 = 400.0`; `struct CardView { equation_open: bool, screen_width: u32, screen_height: u32 }` with `CardView::of(ctx: &egui::Context, equation_open: bool) -> CardView`; `struct CardFrame { view: CardView, region_width: f32 }`; fields `card_choice: Option<bool>`, `card_start: Option<(CardView, bool)>`; `fn card_default(&mut self, ui: &egui::Ui, card: CardFrame, lines: &[String]) -> bool`; `fn card_open_height(ui: &egui::Ui, lines: &[String]) -> f32`; `fn torque_card_ui(&mut self, ui, readouts, card: CardFrame, shown: &DesignInputs)`; `fn view_ui(&mut self, ui, shown, readouts, card: CardFrame)`.
  - `gui::geometry_view`: `pub(crate) const MIN_DRAWING_HEIGHT: f32 = 160.0`; `pub(crate) fn list_row(ui: &egui::Ui) -> f32`; `pub(crate) fn drawing_floor(ui: &egui::Ui) -> f32`.
  - `gui::test_support::drawing_rect(output: &egui::FullOutput) -> egui::Rect` (`pub(crate)`; panics unless exactly one drawing is painted).

- [ ] **Step 1: Write the failing tests**

(a) `magcoupling-rs/src/gui/test_support.rs`, before `drawn_texts`: the drawing's probe (spec 8 intro):

Old:
```rust
/// Every text egui drew in a frame (widgets and painter text), nested shapes
```

New:
```rust
/// The geometry drawing's painted area in a frame (slice 3 spec 8 intro): the one rect filled
/// with the extreme background colour of egui's default visuals (the headless harness's style)
/// with 4-point corners, as `geometry_view::geometry_ui` paints it under the drawing. Panics
/// unless exactly one such rect was painted: the clamp drawing and the notes' diagrams paint
/// the same plate, so a test reads the drawing with neither on screen.
pub(crate) fn drawing_rect(output: &egui::FullOutput) -> egui::Rect {
    let background = egui::Visuals::default().extreme_bg_color;
    let plates: Vec<egui::Rect> = flat_shapes(output)
        .into_iter()
        .filter_map(|shape| match shape {
            egui::Shape::Rect(r)
                if r.fill == background && r.corner_radius == egui::CornerRadius::same(4) =>
            {
                Some(r.rect)
            }
            _ => None,
        })
        .collect();
    assert_eq!(plates.len(), 1, "one drawing painted: {plates:?}");
    plates[0]
}

/// Every text egui drew in a frame (widgets and painter text), nested shapes
```

(b) `magcoupling-rs/src/gui/geometry_view.rs`, the test module, before `the_view_stays_in_a_short_region_the_drawing_shrinking_and_the_list_scrolling`. The floor's own pin: in a region of exactly `drawing_floor` the drawing keeps 160 points, and a point less it does not:

Old:
```rust
    #[test]
    fn the_view_stays_in_a_short_region_the_drawing_shrinking_and_the_list_scrolling() {
```

New:
```rust
    #[test]
    fn the_drawing_floor_is_the_room_that_keeps_the_drawing_over_three_list_rows() {
        // Slice 3 spec 6.2: drawing_floor is MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows of the
        // callout list (list_row, the row geometry_ui reserves). In a region of exactly that
        // height the drawing of the defaults (a list of more than three rows) keeps its 160
        // points; a point less and it shrinks below them.
        let ctx = egui::Context::default();
        let inputs = DesignInputs::default();
        let results = compute_all(&inputs);
        let size = Vec2::new(1000.0, 900.0);
        let mut floor = f32::NAN;
        sized_frame(&ctx, size, Vec::new(), |ui| {
            let row = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y;
            assert_eq!(list_row(ui), row);
            floor = drawing_floor(ui);
            assert_eq!(floor, MIN_DRAWING_HEIGHT + MIN_LIST_ROWS * row);
        });
        for (height, keeps) in [(floor, true), (floor - 1.0, false)] {
            let region = Rect::from_min_size(Pos2::new(20.0, 30.0), Vec2::new(700.0, height));
            let mut drawing = f32::NAN;
            for _ in 0..2 {
                crate::gui::test_support::region_frame(&ctx, size, region, |ui| {
                    let layout =
                        geometry_ui(ui, &inputs, &results, false, &mut Readouts::default());
                    drawing = layout.rect.height();
                });
            }
            assert_eq!(
                drawing >= MIN_DRAWING_HEIGHT - 1e-3,
                keeps,
                "a region of {height}: a drawing of {drawing}"
            );
        }
    }

    #[test]
    fn the_view_stays_in_a_short_region_the_drawing_shrinking_and_the_list_scrolling() {
```

(c) `magcoupling-rs/src/gui/panel.rs`, the test module. The imports:

Old:
```rust
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::guidance::{Consequences, RING_SIZE_STAYS, plain_text_violations};
```

New:
```rust
    use crate::gui::format::{format_value, with_unit};
    use crate::gui::geometry_view::MIN_DRAWING_HEIGHT;
    use crate::gui::guidance::{Consequences, RING_SIZE_STAYS, plain_text_violations};
```

Old:
```rust
    use crate::gui::test_support::{
        SCREEN, clamp_at_its_limit, drawn_texts, hot_day, hub_overstressed, key_event, key_tap,
        last_characters, only_the_clamp_fails, only_the_joint_fails, primary_button, select_all,
        short_magnets, sized_frame, sized_frame_at, smoke_inputs, text_center, text_rect,
        text_rects, tooltips_at_once, with_poles,
    };
```

New:
```rust
    use crate::gui::test_support::{
        SCREEN, clamp_at_its_limit, drawing_rect, drawn_texts, hot_day, hub_overstressed,
        key_event, key_tap, last_characters, no_back_iron, only_the_clamp_fails,
        only_the_joint_fails, primary_button, select_all, short_magnets, sized_frame,
        sized_frame_at, smoke_inputs, text_center, text_rect, text_rects, tooltips_at_once,
        with_poles,
    };
```

The 8.5 helpers (`smoke_design()` from `test_support::smoke_inputs()`, critique A-24) and tests 19 to 25, with the resize test after test 22 (plan review 3.3), test 22 checking the drawing's floor beside the Equation panel (plan review 3.1) and test 24 scrolling the body with the mouse wheel (plan review 3.4), before `size_tab_draws_each_what_moves_what_line_under_its_control`:

Old:
```rust
    #[test]
    fn size_tab_draws_each_what_moves_what_line_under_its_control() {
```

New:
```rust
    // The torque card's open or closed and its room (slice 3 spec 6.2, tests 19 to 25). Each
    // test pauses the shaft-clamp section before the card's first frame, the one that decides
    // its default (`card_tab`), and steps frames past the collapse animation before it measures.

    /// `inputs` as a design with Auto off.
    fn design_of(inputs: DesignInputs) -> Design {
        Design {
            inputs,
            sizing: SizingState::default(),
        }
    }

    /// The gui-smoke share link's design (slice 3 spec 2.1, (b)): the defaults with a 1.5 mm face
    /// gap, Auto on the magnet length.
    fn smoke_design() -> Design {
        Design {
            inputs: smoke_inputs(),
            sizing: SizingState {
                mode: SizingMode::TorqueToMagnets,
                variable: FreeVariable::AxialLength,
            },
        }
    }

    /// A harness on a screen of `size` showing the Size tab for `design` (an Auto one solved),
    /// with the shaft-clamp section paused and the Equation panel open or not, and the frame
    /// that drew the card first, the one that decided its default (slice 3 spec 8.5). The
    /// harness starts on the Geometry tab, which draws no card, and the pause, the design and
    /// the Equation panel are set there, so the default is decided for them, at the shipped
    /// pause; the last frames step past the collapse animation.
    fn card_tab(
        size: egui::Vec2,
        design: Design,
        equation_open: bool,
    ) -> (Harness, egui::FullOutput) {
        let mut harness = Harness::on_screen(size);
        harness.panel.clamps_paused = true;
        harness.panel.open_design(design);
        if equation_open {
            harness.panel.explorer.open_note(SIZE_TAB_NOTE);
        }
        harness.frame(Vec::new());
        // An Auto design: the solve runs once the design has been still for the debounce.
        harness.frame_after(DEBOUNCE_S + 0.01, Vec::new());
        harness.frame(Vec::new());
        assert_eq!(harness.panel.card_start, None, "no card drawn yet");
        let deciding = harness.click_text(CentreView::SizeAndTorque.label());
        assert!(
            harness.panel.card_start.is_some(),
            "the click's frame drew the card"
        );
        harness.frame_after(1.0, Vec::new());
        harness.frame(Vec::new());
        (harness, deciding)
    }

    /// Whether the torque card is open, as egui keeps it.
    fn card_open(harness: &Harness) -> bool {
        egui::collapsing_header::CollapsingState::load(&harness.ctx, egui::Id::new(TORQUE_CARD_ID))
            .expect("the card has been drawn")
            .is_open()
    }

    /// The card's Why link in a frame: its text rect and where a click lands on its last
    /// character. Of the Size tab's links to how the coupling makes torque, the card's is the
    /// one at or right of the drawing's left edge (the other two are on the inputs side).
    fn card_link(output: &egui::FullOutput) -> (egui::Rect, egui::Pos2) {
        let text = why_note_text(reviewed_note(SIZE_TAB_NOTE).expect("the note is reviewed"));
        let left = drawing_rect(output).left();
        let rects: Vec<egui::Rect> = text_rects(output, &text)
            .into_iter()
            .filter(|rect| rect.left() >= left)
            .collect();
        let at: Vec<egui::Pos2> = last_characters(output, &text)
            .into_iter()
            .filter(|at| at.x >= left)
            .collect();
        assert_eq!((rects.len(), at.len()), (1, 1), "{rects:?}");
        (rects[0], at[0])
    }

    /// The clip rect of the first drawn text equal to `needle`: for the card's heading, the
    /// centre view's region (`confined` clips to it).
    fn text_clip(output: &egui::FullOutput, needle: &str) -> egui::Rect {
        fn has(shape: &egui::Shape, needle: &str) -> bool {
            match shape {
                egui::Shape::Vec(nested) => nested.iter().any(|s| has(s, needle)),
                egui::Shape::Text(text) => text.galley.text() == needle,
                _ => false,
            }
        }
        output
            .shapes
            .iter()
            .find(|clipped| has(&clipped.shape, needle))
            .map(|clipped| clipped.clip_rect)
            .unwrap_or_else(|| panic!("no {needle:?} drawn"))
    }

    /// The card's pre-measure of `lines` ([`card_open_height`]) and the drawing's floor
    /// ([`drawing_floor`]), measured as the panel measures them on a deciding frame, in a scratch
    /// frame of a region `width` points wide in the harness's style (egui's default).
    fn scratch_measure(width: f32, lines: &[String]) -> (f32, f32) {
        let mut measured = (f32::NAN, f32::NAN);
        let region = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(width, 4000.0));
        crate::gui::test_support::region_frame(
            &egui::Context::default(),
            egui::vec2(width + 100.0, 4000.0),
            region,
            |ui| measured = (card_open_height(ui, lines), drawing_floor(ui)),
        );
        measured
    }

    /// The height of the label the card replaced, laid out as it was under the readouts (a
    /// wrapped label at the region's width `width`) in the harness's style [points].
    fn replaced_label_height(harness: &Harness, width: f32) -> f32 {
        let font = harness.ctx.style().text_styles[&egui::TextStyle::Body].clone();
        harness
            .ctx
            .fonts(|f| f.layout(REPLACED_LABEL.to_owned(), font, egui::Color32::WHITE, width))
            .size()
            .y
    }

    #[test]
    fn the_card_sits_between_the_readouts_and_the_drawing() {
        // Slice 3 spec 8.5 test 19: at 1280 x 800, at the defaults and at the smoke design
        // (Auto on the magnet length), the card's heading is under the Pull-out readout and
        // above the drawing; the card starts open, every one of its lines is drawn, the drawing
        // keeps its 160 points and the label the card replaced is gone.
        for design in [design_of(DesignInputs::default()), smoke_design()] {
            let (mut harness, _) = card_tab(egui::vec2(1280.0, 800.0), design, false);
            let output = harness.frame(Vec::new());
            let case = format!("{:?}", harness.panel.sizing);
            assert!(card_open(&harness), "{case}");
            let heading = text_rect(&output, CARD_HEADING).expect("the heading");
            let label = pullout_label(harness.panel.results().metal.op_temp_C);
            let pull_out = centre_rect(&output, &label);
            let drawing = drawing_rect(&output);
            assert!(
                pull_out.bottom() <= heading.top(),
                "{case}: {pull_out:?} {heading:?}"
            );
            assert!(
                heading.bottom() <= drawing.top(),
                "{case}: {heading:?} {drawing:?}"
            );
            let lines = shown_consequences(&harness)
                .card
                .lines(harness.panel.auto());
            for line in &lines {
                assert_eq!(count(&output, line), 1, "{case}: {line}");
            }
            assert!(
                drawing.height() >= MIN_DRAWING_HEIGHT,
                "{case}: {drawing:?}"
            );
            assert_eq!(count(&output, REPLACED_LABEL), 0, "{case}");
        }
    }

    #[test]
    fn the_card_starts_closed_in_a_narrow_window_and_the_drawing_gains() {
        // Slice 3 spec 8.5 test 20 (rule 2, T-10): at 930 x 800, with the Equation panel closed,
        // the card starts closed at the defaults and at short magnets, whose one-line form rule
        // 3 alone would open; closed, it is no taller than the label it replaced, laid out at the
        // region's width, so the drawing is at least as tall as before the card. At 1280 x 800
        // short magnets start open, and the card closed by its heading is again no taller than
        // the label there.
        for inputs in [DesignInputs::default(), short_magnets()] {
            let one_line = inputs == short_magnets();
            let (mut harness, _) = card_tab(egui::vec2(930.0, 800.0), design_of(inputs), false);
            let output = harness.frame(Vec::new());
            let region = drawing_rect(&output).width();
            assert!(!card_open(&harness), "one-line form {one_line}");
            assert!(region < CARD_OPEN_MIN_WIDTH, "{region}");
            let heading = text_rect(&output, CARD_HEADING).expect("the heading");
            let spacing = harness.ctx.style().spacing.item_spacing.y;
            let closed = drawing_rect(&output).top() - heading.top() - spacing;
            let label = replaced_label_height(&harness, region);
            assert!(
                closed <= label,
                "closed {closed} against the label's {label}"
            );
            if one_line {
                // Rule 3 alone would open the one-line form here: the width rule closes it.
                let room = text_clip(&output, CARD_HEADING).bottom() - heading.top();
                let lines = shown_consequences(&harness).card.lines(None);
                let (open_height, floor) = scratch_measure(region, &lines);
                assert!(
                    room - open_height - spacing >= floor,
                    "room {room}, card {open_height}, floor {floor}"
                );
            }
        }
        let (harness, _) = card_tab(egui::vec2(1280.0, 800.0), design_of(short_magnets()), false);
        assert!(card_open(&harness), "the one-line form in a wide window");
        let (mut harness, _) = card_tab(
            egui::vec2(1280.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        let output = harness.frame(Vec::new());
        harness.click(text_center(&output, CARD_HEADING));
        let output = harness.frame_after(1.0, Vec::new());
        assert!(!card_open(&harness));
        let heading = text_rect(&output, CARD_HEADING).expect("the heading");
        let spacing = harness.ctx.style().spacing.item_spacing.y;
        let closed = drawing_rect(&output).top() - heading.top() - spacing;
        let label = replaced_label_height(&harness, drawing_rect(&output).width());
        assert!(
            closed <= label,
            "closed {closed} against the label's {label}"
        );
        // At 930 x 800 the header row wraps (spec 6.1): the card's Why link continues under the
        // heading inside the centre region, and a click on its last character opens how the
        // coupling makes torque (test 21 clicks it in a wide window).
        let (mut harness, _) = card_tab(
            egui::vec2(930.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        let output = harness.frame(Vec::new());
        let (link, at) = card_link(&output);
        let heading = text_rect(&output, CARD_HEADING).expect("the heading");
        assert!(
            link.right() <= drawing_rect(&output).right(),
            "{link:?} past {:?}",
            drawing_rect(&output)
        );
        assert!(
            link.bottom() > heading.bottom(),
            "the link wraps under the heading"
        );
        harness.click(at);
        assert_eq!(harness.panel.explorer.note(), Some(SIZE_TAB_NOTE));
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_card_starts_closed_when_it_would_take_the_drawing_under_its_floor() {
        // Slice 3 spec 8.5 test 20a (rule 3 deciding alone): at 1080 x 745, with the Equation
        // panel closed and the welcome strip shown. On the deciding frame the view has the
        // Equation panel closed and the centre region (the drawing's width: the drawing fills
        // it) is at least CARD_OPEN_MIN_WIDTH wide, so rules 1 and 2 pass. No back iron's open
        // card would take the drawing under its floor: it starts closed; the one-line form at
        // short magnets leaves room: it starts open.
        for (inputs, open) in [(no_back_iron(), false), (short_magnets(), true)] {
            let (harness, deciding) = card_tab(egui::vec2(1080.0, 745.0), design_of(inputs), false);
            let (view, default) = harness.panel.card_start.expect("decided");
            assert!(!view.equation_open);
            let region = drawing_rect(&deciding).width();
            assert!(region >= CARD_OPEN_MIN_WIDTH, "{region}");
            assert_eq!(count(&deciding, WELCOME), 1);
            assert_eq!((default, card_open(&harness)), (open, open));
            assert_eq!(harness.panel.card_choice, None);
        }
    }

    #[test]
    fn the_card_starts_closed_while_the_equation_panel_is_open() {
        // Slice 3 spec 8.5 test 21 (rule 1): at 1280 x 800 the card starts closed while the
        // Equation panel is open, and so it does on a screen tall enough for the open card above
        // the panel. Started open, a click on the card's own Why link opens the panel, and with
        // no user choice the card closes (decision 4A); the panel closing opens it again.
        let wide = egui::vec2(1280.0, 800.0);
        let (harness, _) = card_tab(wide, design_of(DesignInputs::default()), true);
        assert!(!card_open(&harness));
        // On a tall screen the open card fits above the Equation panel, so rules 2 and 3 alone
        // would open it there: rule 1 closes it on its own.
        let (mut harness, _) = card_tab(
            egui::vec2(1280.0, 2000.0),
            design_of(DesignInputs::default()),
            true,
        );
        assert!(!card_open(&harness));
        let output = harness.frame(Vec::new());
        let heading = text_rect(&output, CARD_HEADING).expect("the heading");
        let room = text_clip(&output, CARD_HEADING).bottom() - heading.top();
        let region = drawing_rect(&output).width();
        let (open_height, floor) =
            scratch_measure(region, &shown_consequences(&harness).card.lines(None));
        let spacing = harness.ctx.style().spacing.item_spacing.y;
        assert!(region >= CARD_OPEN_MIN_WIDTH, "{region}");
        assert!(
            room - open_height - spacing >= floor,
            "room {room}, card {open_height}, floor {floor}"
        );
        let (mut harness, _) = card_tab(wide, design_of(DesignInputs::default()), false);
        assert!(card_open(&harness));
        let output = harness.frame(Vec::new());
        let (_, at) = card_link(&output);
        harness.click(at);
        assert_eq!(harness.panel.explorer.note(), Some(SIZE_TAB_NOTE));
        harness.frame_after(1.0, Vec::new());
        assert!(
            !card_open(&harness),
            "the Equation panel's opening closes it"
        );
        assert_eq!(harness.panel.card_choice, None);
        harness.panel.explorer.close();
        harness.frame_after(1.0, Vec::new());
        assert!(card_open(&harness), "and its closing opens it again");
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_user_s_open_or_close_wins_for_the_session() {
        // Slice 3 spec 8.5 test 22: at 930 x 800 the card starts closed; a click on its heading
        // opens it, and it stays open after a design edit, a resize to 1280 x 800, the Equation
        // panel opening, Reset all and a load. A click on its arrow then closes it and one on
        // its heading opens it again: each is the user's choice.
        let (mut harness, _) = card_tab(
            egui::vec2(930.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(!card_open(&harness));
        let output = harness.frame(Vec::new());
        harness.click(text_center(&output, CARD_HEADING));
        harness.frame_after(1.0, Vec::new());
        assert_eq!(harness.panel.card_choice, Some(true));
        // Open by the user's choice, and its body capped so the drawing keeps its 160 points,
        // the Equation panel's opening at 1280 x 800 included.
        let still_open = |harness: &mut Harness, what: &str| {
            harness.frame_after(1.0, Vec::new());
            let output = harness.frame(Vec::new());
            assert!(card_open(harness), "{what}");
            assert_eq!(harness.panel.card_choice, Some(true), "{what}");
            let drawing = drawing_rect(&output);
            assert!(
                drawing.height() >= MIN_DRAWING_HEIGHT - 0.01,
                "{what}: {drawing:?}"
            );
        };
        still_open(&mut harness, "the click");
        harness.panel.inputs.metal.face_gap_mm = 2.0;
        still_open(&mut harness, "a design edit");
        harness.screen = egui::vec2(1280.0, 800.0);
        still_open(&mut harness, "a resize");
        harness.panel.explorer.open_note(SIZE_TAB_NOTE);
        still_open(&mut harness, "the Equation panel opening");
        harness.panel.explorer.close();
        harness.click_text(RESET_ALL);
        still_open(&mut harness, "Reset all");
        harness
            .panel
            .load_design_file(&design_to_json(&design_of(with_poles(8))))
            .expect("the file loads");
        still_open(&mut harness, "a load");
        // The arrow, half an indent left of the heading's text, closes it; the heading opens it.
        let output = harness.frame(Vec::new());
        let heading = text_rect(&output, CARD_HEADING).expect("the heading");
        let indent = harness.ctx.style().spacing.indent;
        harness.click(egui::pos2(
            heading.left() - indent / 2.0,
            heading.center().y,
        ));
        harness.frame_after(1.0, Vec::new());
        assert_eq!(harness.panel.card_choice, Some(false));
        assert!(!card_open(&harness));
        let output = harness.frame(Vec::new());
        harness.click(text_center(&output, CARD_HEADING));
        harness.frame_after(1.0, Vec::new());
        assert_eq!(harness.panel.card_choice, Some(true));
        assert!(card_open(&harness));
    }

    #[test]
    fn the_card_is_decided_again_when_the_window_is_resized_with_no_user_choice() {
        // Slice 3 spec 6.2: with no user choice, the default is decided again when the view
        // changes, and a resize of the window is one of its triggers (test 22 resizes only after
        // the user's click, which wins). Started closed at 930 x 800 (rule 2), the card opens
        // when the window widens to 1280 x 800 and closes when it narrows again; each time the
        // view it was decided for is the new screen, and no user choice is recorded.
        let (mut harness, _) = card_tab(
            egui::vec2(930.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(!card_open(&harness));
        for (width, open) in [(1280.0, true), (930.0, false)] {
            harness.screen = egui::vec2(width, 800.0);
            harness.frame(Vec::new());
            harness.frame_after(1.0, Vec::new());
            harness.frame(Vec::new());
            let (view, default) = harness.panel.card_start.expect("decided");
            assert_eq!(
                (view.screen_width, view.screen_height, default),
                (width as u32, 800, open),
                "{width}"
            );
            assert_eq!(card_open(&harness), open, "{width}");
            assert_eq!(harness.panel.card_choice, None, "{width}");
        }
        assert_eq!(harness.panel.inputs(), &DesignInputs::default());
    }

    #[test]
    fn the_card_never_opens_or_closes_under_a_drag() {
        // Slice 3 spec 8.5 test 23: the card's default is decided for the view (the Equation
        // panel and the screen), never on a design change. At 1280 x 800, from the defaults,
        // open: twenty drag frames on the gap slider; the Outside diameter dragged down across
        // the fit edge (the exploration line appears above the card); Required dragged across
        // hot-low; an assumption changed, so the assumptions banner takes a line of the header
        // and the region loses that height. Through each the card stays open and the view it was
        // decided for does not change.
        let (mut harness, _) = card_tab(
            egui::vec2(1280.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(card_open(&harness));
        let start = harness.panel.card_start;
        let holds = |harness: &mut Harness, what: &str| {
            harness.frame_after(1.0, Vec::new());
            let output = harness.frame(Vec::new());
            assert!(card_open(harness), "{what}");
            assert_eq!(harness.panel.card_start, start, "{what}");
            assert_eq!(harness.panel.card_choice, None, "{what}");
            output
        };
        // A drag on a slider's rail: press, moves, release, a frame each.
        let drag = |harness: &mut Harness, from: egui::Pos2, moves: &[egui::Vec2]| {
            harness.frame(vec![egui::Event::PointerMoved(from)]);
            harness.frame(vec![primary_button(from, true)]);
            let mut at = from;
            for &step in moves {
                at += step;
                harness.frame(vec![egui::Event::PointerMoved(at)]);
            }
            harness.frame(vec![primary_button(at, false)]);
        };
        let gap = harness.size_widget(GAP).rect;
        let to_and_fro: Vec<egui::Vec2> = (0..20)
            .map(|i| egui::vec2(if i % 2 == 0 { 3.0 } else { -3.0 }, 0.0))
            .collect();
        let before = harness.panel.inputs().metal.face_gap_mm;
        drag(
            &mut harness,
            gap.left_center() + egui::vec2(20.0, 0.0),
            &to_and_fro,
        );
        assert_ne!(
            harness.panel.inputs().metal.face_gap_mm,
            before,
            "the drag moved the gap"
        );
        holds(&mut harness, "twenty gap drag frames");
        let diameter = harness.size_widget(DIAMETER).rect;
        drag(
            &mut harness,
            diameter.left_center() + egui::vec2(2.0, 0.0),
            &[egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0)],
        );
        let output = holds(&mut harness, "the diameter across the fit edge");
        let fit = format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter.");
        assert_eq!(count(&output, &fit), 1, "the exploration line shows");
        let required = harness
            .ctx
            .read_response(
                harness
                    .panel
                    .requirement_widget
                    .expect("the Required slider"),
            )
            .expect("drawn")
            .rect;
        let hot_low = harness.panel.results().metal.torque_hot_low_Nm;
        assert!(harness.panel.inputs().metal.required_min_Nm > hot_low);
        drag(
            &mut harness,
            required.left_center() + egui::vec2(2.0, 0.0),
            &[egui::vec2(1.0, 0.0), egui::vec2(-1.0, 0.0)],
        );
        assert!(
            harness.panel.inputs().metal.required_min_Nm < hot_low,
            "across hot-low"
        );
        holds(&mut harness, "Required across hot-low");
        harness.panel.inputs.metal.variation = 0.2;
        let output = holds(&mut harness, "the assumptions banner");
        let banner = assumptions_banner(harness.panel.inputs(), true).expect("a banner");
        assert_eq!(count(&output, &banner), 1);
        // At 1080 x 800, from the defaults, open with no user choice: no back iron widens the
        // dashboard to its content and the centre region narrows (428.0 to 425.7 points in the
        // harness's style), where rule 3 would close the card; it stays open, its view unchanged.
        let (mut harness, _) = card_tab(
            egui::vec2(1080.0, 800.0),
            design_of(DesignInputs::default()),
            false,
        );
        assert!(card_open(&harness));
        let start = harness.panel.card_start;
        let wide = drawing_rect(&harness.frame(Vec::new())).width();
        harness.panel.inputs.coupling.backiron = 0;
        for _ in 0..6 {
            harness.frame(Vec::new());
        }
        let output = harness.frame_after(1.0, Vec::new());
        let narrow = drawing_rect(&output).width();
        assert!(narrow < wide, "the region narrows: {wide} -> {narrow}");
        assert!(card_open(&harness));
        assert_eq!(harness.panel.card_start, start);
        let heading = text_rect(&output, CARD_HEADING).expect("the heading");
        let room = text_clip(&output, CARD_HEADING).bottom() - heading.top();
        let lines = shown_consequences(&harness).card.lines(None);
        let (open_height, floor) = scratch_measure(narrow, &lines);
        let spacing = harness.ctx.style().spacing.item_spacing.y;
        assert!(
            room - open_height - spacing < floor,
            "rule 3 at no back iron would close it: room {room}, card {open_height}, floor {floor}"
        );
    }

    #[test]
    fn the_card_body_scrolls_above_the_drawing_s_floor() {
        // Slice 3 spec 8.5 test 24: opened by a click on its heading at 930 x 800, where rule 2
        // starts it closed and its body is taller than the room, the card's body scrolls and the
        // drawing keeps its 160 points, at the defaults and at no back iron (the tallest card).
        for inputs in [DesignInputs::default(), no_back_iron()] {
            let (mut harness, _) = card_tab(egui::vec2(930.0, 800.0), design_of(inputs), false);
            let output = harness.frame(Vec::new());
            harness.click(text_center(&output, CARD_HEADING));
            let output = harness.frame_after(1.0, Vec::new());
            assert!(card_open(&harness));
            let drawing = drawing_rect(&output);
            assert!(drawing.height() >= MIN_DRAWING_HEIGHT - 0.01, "{drawing:?}");
            // The body is capped, so its last row starts out of view (not painted, or cut by the
            // body's clip); the mouse wheel over the body brings it into view, whole, and the
            // drawing keeps its floor.
            let last = shown_consequences(&harness)
                .card
                .lines(None)
                .pop()
                .expect("a row");
            assert!(
                text_rect(&output, &last)
                    .is_none_or(|row| row.bottom() > text_clip(&output, &last).bottom() + 1.0),
                "nothing to scroll: {last}"
            );
            let body = text_center(&output, CAPTION);
            harness.frame(vec![egui::Event::PointerMoved(body)]);
            harness.frame(vec![egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta: egui::vec2(0.0, -5000.0),
                modifiers: egui::Modifiers::NONE,
            }]);
            // egui spreads a wheel step over several frames.
            let mut output = harness.frame(Vec::new());
            for _ in 0..20 {
                output = harness.frame(Vec::new());
            }
            let row = text_rect(&output, &last).expect("the last row in view");
            let clip = text_clip(&output, &last);
            assert!(clip.contains_rect(row), "{row:?} cut by {clip:?}");
            let drawing = drawing_rect(&output);
            assert!(drawing.height() >= MIN_DRAWING_HEIGHT - 0.01, "{drawing:?}");
            assert!(card_open(&harness));
        }
    }

    #[test]
    fn the_card_s_measured_height_is_the_drawn_height() {
        // Slice 3 spec 8.5 test 25: the card's pre-measure, card_open_height in a scratch frame
        // of the centre region's width, equals its drawn open height (the heading's top to the
        // drawing's, less an item spacing) within a point, at the defaults and the smoke design,
        // at 1280 x 800, where its header is one row, and at 930 x 1024, where the row wraps and
        // a click on the heading opens the card (rule 2 starts it closed); the bodies are under
        // their caps there. At 930 x 800 the bodies are over the cap: the drawn body is the cap,
        // the room at the card less the header, two item spacings and the drawing's floor.
        let body_row = |harness: &Harness| {
            let font = harness.ctx.style().text_styles[&egui::TextStyle::Body].clone();
            harness.ctx.fonts(|f| f.row_height(&font))
        };
        for size in [
            egui::vec2(1280.0, 800.0),
            egui::vec2(930.0, 1024.0),
            egui::vec2(930.0, 800.0),
        ] {
            for design in [design_of(DesignInputs::default()), smoke_design()] {
                let (mut harness, _) = card_tab(size, design, false);
                if !card_open(&harness) {
                    let output = harness.frame(Vec::new());
                    harness.click(text_center(&output, CARD_HEADING));
                    harness.frame_after(1.0, Vec::new());
                }
                let output = harness.frame(Vec::new());
                let case = format!("{size:?} {:?}", harness.panel.sizing.mode);
                let spacing = harness.ctx.style().spacing.item_spacing.y;
                let heading = text_rect(&output, CARD_HEADING).expect("the heading");
                let drawing = drawing_rect(&output);
                let drawn = drawing.top() - heading.top() - spacing;
                let header = heading.bottom().max(card_link(&output).0.bottom()) - heading.top();
                let body = drawn - header - spacing;
                let lines = shown_consequences(&harness)
                    .card
                    .lines(harness.panel.auto());
                let (measured, floor) = scratch_measure(drawing.width(), &lines);
                let room = text_clip(&output, CARD_HEADING).bottom() - heading.top();
                let cap = (room - header - 2.0 * spacing - floor).max(body_row(&harness));
                if size.y < 1000.0 && size.x < 1000.0 {
                    assert!(measured > drawn + 1.0, "{case}: the body is capped");
                    assert!((body - cap).abs() <= 1.0, "{case}: body {body}, cap {cap}");
                } else {
                    assert!(body < cap, "{case}: body {body} under the cap {cap}");
                    assert!(
                        (measured - drawn).abs() <= 1.0,
                        "{case}: measured {measured}, drawn {drawn}"
                    );
                }
            }
        }
    }

    #[test]
    fn size_tab_draws_each_what_moves_what_line_under_its_control() {
```

Test 21's second half (spec 8.5): `no_centre_view_paints_or_takes_clicks_in_the_equation_panel` runs the Size tab with the card closed (the Equation panel's opening leaves it so) and opened by the user, after its other views:

Old:
```rust
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
        assert!(
            failures.is_empty(),
```

New:
```rust
            // Slice 3 test 21: the Size tab's torque card closed, as the Equation panel's opening
            // leaves it, and opened by the user; neither paints into the panel.
            for choice in [None, Some(true)] {
                harness.panel.card_choice = choice;
                harness.panel.centre = CentreView::SizeAndTorque;
                harness.frame_after(1.0, Vec::new());
                let output = harness.frame(Vec::new());
                let case = format!("{size:?}, dragged up {dragged}, the card's choice {choice:?}");
                if let Some(overlap) = equation_panel_overlap(&harness.ctx, &output) {
                    failures.push(format!("{case}: {overlap}"));
                }
            }
            assert_eq!(harness.panel.inputs(), &DesignInputs::default());
        }
        assert!(
            failures.is_empty(),
```

- [ ] **Step 2: Run them and see the build fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib --no-run`

Expected: the lib test target does not compile, 27 errors:
```text
      9 error[E0609]: no field `card_choice` on type `gui::panel::MagcouplingPanel`
      8 error[E0609]: no field `card_start` on type `gui::panel::MagcouplingPanel`
      3 error[E0425]: cannot find value `CARD_OPEN_MIN_WIDTH` in this scope
      2 error[E0425]: cannot find function `drawing_floor` in this scope
      1 error[E0603]: constant `MIN_DRAWING_HEIGHT` is private
      1 error[E0425]: cannot find function `why_note_text` in this scope
      1 error[E0425]: cannot find function `reviewed_note` in this scope
      1 error[E0425]: cannot find function `list_row` in this scope
      1 error[E0425]: cannot find function `card_open_height` in this scope
error: could not compile `magcoupling-rs` (lib test) due to 27 previous errors
```

- [ ] **Step 3: The drawing's floor (`gui/geometry_view.rs`)**

Old:
```rust
const MIN_DRAWING_HEIGHT: f32 = 160.0;
```

New:
```rust
pub(crate) const MIN_DRAWING_HEIGHT: f32 = 160.0;
```

Old:
```rust
    let g = geometry(inputs, results, clamps_paused);
    let row = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y;
```

New:
```rust
    let g = geometry(inputs, results, clamps_paused);
    let row = list_row(ui);
```

`list_row` and `drawing_floor`, after `drawing_height`, before `paint_view`:

Old:
```rust
/// Paints one view's pieces, dashed lines and dimensions, and registers each dimension as a
```

New:
```rust
/// The height of a row of the callout list under the drawing [points]: a body text row and the
/// item spacing, as [`geometry_ui`] reserves it (one expression for it and [`drawing_floor`]).
pub(crate) fn list_row(ui: &egui::Ui) -> f32 {
    ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y
}

/// The room the drawing needs to keep [`MIN_DRAWING_HEIGHT`] with [`MIN_LIST_ROWS`] rows of the
/// callout list under it [points] (slice 3 spec 6.2): what the Size tab's torque card leaves
/// above the drawing, starting closed rather than take it and capping its body above it.
pub(crate) fn drawing_floor(ui: &egui::Ui) -> f32 {
    MIN_DRAWING_HEIGHT + MIN_LIST_ROWS * list_row(ui)
}

/// Paints one view's pieces, dashed lines and dimensions, and registers each dimension as a
```

- [ ] **Step 4: Open or closed, the pre-measure and the capped body (`gui/panel.rs`)**

The imports:

Old:
```rust
use crate::gui::dashboard::{
    Level, dashboard_ui, end_effect_banner, end_effect_banner_ui, why_note_link,
};
use crate::gui::explorer::{
    EQUATION_PANEL, Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights,
};
use crate::gui::format::{format_value, search_needle, with_unit};
use crate::gui::geometry_view::geometry_ui;
```

New:
```rust
use crate::gui::dashboard::{
    Level, dashboard_ui, end_effect_banner, end_effect_banner_ui, why_note_link, why_note_text,
};
use crate::gui::explorer::{
    EQUATION_PANEL, Explorer, FOCUS_WIDTH, explorer_ui, held_reason, panel_heights, reviewed_note,
};
use crate::gui::format::{format_value, search_needle, with_unit};
use crate::gui::geometry_view::{drawing_floor, geometry_ui};
```

The constant and the two types, after `TORQUE_CARD_ID`:

Old:
```rust
const TORQUE_CARD_ID: &str = "magcoupling_torque_card";
```

New:
```rust
const TORQUE_CARD_ID: &str = "magcoupling_torque_card";

/// The narrowest centre region the torque card starts open in [points] (slice 3 spec T-10,
/// decision 4A's "the window is narrow" as a width rule): between the 278 points of a
/// 930 x 800 window's region and the 628 of a 1280 x 800 window's.
const CARD_OPEN_MIN_WIDTH: f32 = 400.0;

/// The view the torque card's default is decided for (slice 3 spec 6.2): whether the Equation
/// panel is open, and the screen's size in whole points. A design change never changes it, so
/// the card never opens or closes under a drag; nor does the centre region's width, which a side
/// panel widening to fit a design's content moves (code review 3 MF-2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CardView {
    equation_open: bool,
    screen_width: u32,
    screen_height: u32,
}

impl CardView {
    /// This frame's view: whether the Equation panel is open (`equation_open`), and `ctx`'s
    /// screen rounded to whole points.
    fn of(ctx: &egui::Context, equation_open: bool) -> Self {
        let screen = ctx.screen_rect();
        Self {
            equation_open,
            screen_width: screen.width().round() as u32,
            screen_height: screen.height().round() as u32,
        }
    }
}

/// What the torque card's open-or-closed rule reads of a frame (slice 3 spec 6.2), taken at the
/// top of the centre region before the Equation panel takes its part: the view, and the region's
/// width [points] (rule 2, read on a deciding frame only).
#[derive(Clone, Copy, Debug)]
struct CardFrame {
    view: CardView,
    region_width: f32,
}
```

The fields, after `welcome_dismissed`, and their start:

Old:
```rust
    /// Whether the session has hidden the welcome strip (spec 3.1): a view choice, never in a
    /// design file, a share link or the undo history, and kept by reset all and every load.
    welcome_dismissed: bool,
```

New:
```rust
    /// Whether the session has hidden the welcome strip (spec 3.1): a view choice, never in a
    /// design file, a share link or the undo history, and kept by reset all and every load.
    welcome_dismissed: bool,
    /// The user's own open or close of the Size tab's torque card (slice 3 spec 6.2), set by a
    /// click on its arrow or its heading: it wins for the session. A view choice, as
    /// `welcome_dismissed` is.
    card_choice: Option<bool>,
    /// The torque card's default and the view it was decided for (slice 3 spec 6.2): decided
    /// again only when the view differs, while the user has made no choice. A view choice too.
    card_start: Option<(CardView, bool)>,
```

Old:
```rust
            welcome_dismissed: false,
```

New:
```rust
            welcome_dismissed: false,
            card_choice: None,
            card_start: None,
```

`centre_ui` reads the frame once, before the Equation panel takes its part of the region (spec 6.2 rule 2: `gui/panel.rs:788-793`), and hands it to `view_ui`:

Old:
```rust
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        // The Equation panel docks at the bottom of the centre region (decision M43-1). It
```

New:
```rust
    fn centre_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
        // What the torque card's open-or-closed rule reads (slice 3 spec 6.2), once, before the
        // Equation panel takes its part of the region.
        let card = CardFrame {
            view: CardView::of(ui.ctx(), self.explorer.open),
            region_width: ui.available_rect_before_wrap().width(),
        };
        // The Equation panel docks at the bottom of the centre region (decision M43-1). It
```

Old:
```rust
                self.view_ui(ui, shown, readouts);
```

New:
```rust
                self.view_ui(ui, shown, readouts, card);
```

Old:
```rust
    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
    /// readouts (`readouts`). Each view fits itself to `ui`'s available height.
    fn view_ui(&mut self, ui: &mut egui::Ui, shown: &DesignInputs, readouts: &mut Readouts) {
```

New:
```rust
    /// The view of the centre region the tabs select, of `shown`, the design shown, its values
    /// readouts (`readouts`), with what the torque card's open-or-closed rule reads of the frame
    /// (`card`). Each view fits itself to `ui`'s available height.
    fn view_ui(
        &mut self,
        ui: &mut egui::Ui,
        shown: &DesignInputs,
        readouts: &mut Readouts,
        card: CardFrame,
    ) {
```

Old:
```rust
                self.torque_card_ui(ui, readouts, &shown_after_edits);
```

New:
```rust
                self.torque_card_ui(ui, readouts, card, &shown_after_edits);
```

`torque_card_ui`: its doc and parameter, the state set from the user's choice or the default, the user's click recorded, the capped body; then `card_default` after it:

Old:
```rust
    /// The torque card (slice 3 spec 6.1), between the Size tab's readouts and its drawing: a
    /// collapsing header, open, whose one wrapping row holds the heading, which opens and closes
    /// the card as its arrow does, and the Why link to how the coupling makes torque; under it
    /// the card's lines for `shown`, the design shown after this frame's edits (spec 3.5), weak
    /// like the lines under the controls. The card's own `compute_all` is defensive: the cache's
    /// key and the results it is handed come from one design whatever a later change puts between
    /// the panel's results and this view.
    fn torque_card_ui(&mut self, ui: &mut egui::Ui, readouts: &mut Readouts, shown: &DesignInputs) {
```

New:
```rust
    /// The torque card (slice 3 spec 6.1), between the Size tab's readouts and its drawing: a
    /// collapsing header whose one wrapping row holds the heading, which opens and closes the
    /// card as its arrow does, and the Why link to how the coupling makes torque; under it the
    /// card's lines for `shown`, the design shown after this frame's edits (spec 3.5), weak like
    /// the lines under the controls, in a body that scrolls above the drawing's floor (spec 6.2).
    /// Open or closed: the user's own choice for the session, else the default for `card`'s view
    /// ([`MagcouplingPanel::card_default`]). The card's own `compute_all` is defensive: the
    /// cache's key and the results it is handed come from one design whatever a later change
    /// puts between the panel's results and this view.
    fn torque_card_ui(
        &mut self,
        ui: &mut egui::Ui,
        readouts: &mut Readouts,
        card: CardFrame,
        shown: &DesignInputs,
    ) {
```

Old:
```rust
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            egui::Id::new(TORQUE_CARD_ID),
            true,
        );
        // The heading toggles the card as the arrow does. `HeaderResponse` keeps the closure's
        // return private until `body`, so the click is recorded here.
```

New:
```rust
        let open = match self.card_choice {
            Some(choice) => choice,
            None => self.card_default(ui, card, &lines),
        };
        let mut state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            egui::Id::new(TORQUE_CARD_ID),
            open,
        );
        state.set_open(open);
        // The heading toggles the card as the arrow does. `HeaderResponse` keeps the closure's
        // return private until `body`, so the click is recorded here.
```

Old:
```rust
        if heading_clicked {
            header.toggle();
        }
        header.body(|ui| {
            for line in &lines {
                weak_line(ui, line);
            }
        });
    }
```

New:
```rust
        if heading_clicked {
            header.toggle();
        }
        // A click on the arrow (inside `show_header`) or on the heading is the user's choice.
        if header.is_open() != open {
            self.card_choice = Some(header.is_open());
        }
        header.body(|ui| {
            // The body scrolls inside a cap that keeps the drawing's floor under the card: the
            // region left under the header, less the spacing before the drawing and its floor,
            // and at least one row. The cap is also the body's least height while it scrolls:
            // egui's own (64 points) would hold it taller and push the drawing under its floor
            // beside the Equation panel.
            let cap = (ui.available_height() - ui.spacing().item_spacing.y - drawing_floor(ui))
                .max(ui.text_style_height(&egui::TextStyle::Body));
            egui::ScrollArea::vertical()
                .id_salt("magcoupling_torque_card_body")
                .max_height(cap)
                .min_scrolled_height(cap)
                .show(ui, |ui| {
                    for line in &lines {
                        weak_line(ui, line);
                    }
                });
        });
    }

    /// The torque card's default for `card`'s view (slice 3 spec 6.2): the one decided for that
    /// view, else decided now and kept with it, so a design change never decides it again. Open
    /// iff the Equation panel is closed (rule 1), the centre region is at least
    /// [`CARD_OPEN_MIN_WIDTH`] wide (rule 2) and the open card leaves the drawing its floor
    /// (rule 3: `ui`'s height at the card, less the card's height measured before it is drawn,
    /// [`card_open_height`] of `lines`, and the spacing before the drawing, is at least
    /// [`drawing_floor`]).
    fn card_default(&mut self, ui: &egui::Ui, card: CardFrame, lines: &[String]) -> bool {
        if let Some((view, open)) = self.card_start
            && view == card.view
        {
            return open;
        }
        let room =
            ui.available_height() - card_open_height(ui, lines) - ui.spacing().item_spacing.y;
        let open = !card.view.equation_open
            && card.region_width >= CARD_OPEN_MIN_WIDTH
            && room >= drawing_floor(ui);
        self.card_start = Some((card.view, open));
        open
    }
```

`card_open_height`, a free function before `confined`:

Old:
```rust
/// Draws `add` in a child of `ui` (salted `id_salt`) confined to the space left in `ui`: the
```

New:
```rust
/// The torque card's height when open [points], measured before it is drawn (slice 3 spec 6.2):
/// its header row laid out as [`MagcouplingPanel::torque_card_ui`] draws it (beside the arrow,
/// an indent wide, the heading, then the Why link continuing after it and wrapping under it in a
/// narrow region, as egui lays a label out in a wrapping row, the row at least an interactive
/// widget's height), then the item spacing and each of `lines` as a galley wrapped at the body's
/// width (the row's, less the body's indent), with the item spacing between them.
fn card_open_height(ui: &egui::Ui, lines: &[String]) -> f32 {
    let spacing = ui.spacing();
    let width = ui.available_width() - spacing.indent;
    // A label's galley as egui lays it out: wrapped at `width`, its first row starting
    // `indentation` in and at least `first_row` tall.
    let galley = |text: &str, indentation: f32, first_row: f32| {
        let mut job =
            std::sync::Arc::unwrap_or_clone(egui::WidgetText::from(text).into_layout_job(
                ui.style(),
                egui::FontSelection::Default,
                ui.text_valign(),
            ));
        job.wrap.max_width = width;
        job.first_row_min_height = first_row;
        job.halign = egui::Align::Min;
        job.justify = false;
        if let Some(section) = job.sections.first_mut() {
            section.leading_space = indentation;
        }
        ui.fonts(|fonts| fonts.layout_job(job))
    };
    let row = spacing.interact_size.y;
    let heading = galley(CARD_HEADING, 0.0, row).size();
    let header = match reviewed_note(SIZE_TAB_NOTE) {
        Some(note) => galley(
            &why_note_text(note),
            heading.x + spacing.item_spacing.x,
            row,
        )
        .size()
        .y
        .max(heading.y),
        None => heading.y,
    };
    let body: f32 = lines
        .iter()
        .map(|line| galley(line, 0.0, 0.0).size().y)
        .sum::<f32>()
        + spacing.item_spacing.y * lines.len().saturating_sub(1) as f32;
    header + spacing.item_spacing.y + body
}

/// Draws `add` in a child of `ui` (salted `id_salt`) confined to the space left in `ui`: the
```

- [ ] **Step 5: Run the tests and see them pass; plant eleven faults**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib -- the_card_sits_between the_card_starts_closed the_card_never_opens the_card_body_scrolls the_card_s_measured_height the_user_s_open_or_close_wins_for_the_session the_drawing_floor_is_the_room no_centre_view_paints_or_takes_clicks_in_the_equation_panel the_size_tab_draws_the_torque_card_in_place_of_the_label the_size_tab_s_why_links_open_their_notes idle_frames_ the_card_is_decided_again_when_the_window_is_resized`

Expected: `test result: ok. 21 passed; 0 failed; 0 ignored; 0 measured; 956 filtered out`.

Each planted fault below is one edit in `gui/panel.rs` ((h)'s in `gui/geometry_view.rs`), run with the filter given, then undone:

| Planted fault | Filter | Expected |
|---|---|---|
| (a) A key that reads the region's width: in `card_default`, the condition `&& view == card.view` becomes `&& view == card.view && card.region_width >= 427.0` (the decision is made again once a side panel narrows the region) | `the_card_never_opens_or_closes_under_a_drag` | fails `assertion failed: card_open(&harness)` at no back iron, 1080 x 800 (the region 428.0 -> 425.7 pt; rule 3 closes it there) |
| (b) Rule 2 removed (the line `&& card.region_width >= CARD_OPEN_MIN_WIDTH` deleted) | `the_card_starts_closed_in_a_narrow_window_and_the_drawing_gains` | fails `one-line form true`: at 930 x 800 the one-line form opens |
| (c) Rule 3 removed (`&& room >= drawing_floor(ui);` becomes `;`) | `the_card_starts_closed_when_it_would_take_the_drawing_under_its_floor` | fails `left: (true, true)`, `right: (false, false)` at no back iron, 1080 x 745 |
| (d) Rule 1 removed (`let open = !card.view.equation_open` becomes `let open = true`) | `the_card_starts_closed_while_the_equation_panel_is_open` | fails `assertion failed: !card_open(&harness)` at 1280 x 2000 (at 1280 x 800 rule 3 closes it too; plan deviation PD-33) |
| (e) The user's choice not recorded (the three lines `if header.is_open() != open { self.card_choice = Some(header.is_open()); }` deleted) | `the_user_s_open_or_close_wins_for_the_session` | fails `left: None`, `right: Some(true)` after the heading's click |
| (f) No cap (`.max_height(cap)` becomes `.max_height(f32::INFINITY + cap)`) | `the_card_body_scrolls the_card_s_measured_height` | two fail: `the_card_body_scrolls_above_the_drawing_s_floor` (the drawing `[[336.0 642.0] - [614.0 731.9]]`, 89.9 pt) and `the_card_s_measured_height_is_the_drawn_height` (`[930.0 800.0] MagnetsToTorque: the body is capped`) |
| (g) The pre-measure without the wrap (in `card_open_height`, the link's indentation `heading.x + spacing.item_spacing.x,` becomes `0.0,`) | `the_card_s_measured_height_is_the_drawn_height` | fails `[930.0 1024.0] MagnetsToTorque: measured 341, drawn 355` (the 32 pt wrapped header measured as one 18 pt row) |
| (h) `drawing_floor` over two rows (`MIN_DRAWING_HEIGHT + (MIN_LIST_ROWS - 1.0) * list_row(ui)`, `gui/geometry_view.rs`) | `the_drawing_floor_is_the_room_that_keeps_the_drawing_over_three_list_rows the_card_body_scrolls_above_the_drawing_s_floor` | both fail: `left: 194.75`, `right: 212.125`; the drawing under 160 pt at 930 x 800 (`[[336.0 589.2] - [614.0 731.9]]`) |
| (i) The body's least height left at egui's 64 pt (the line `.min_scrolled_height(cap)` deleted) | `the_user_s_open_or_close_wins_for_the_session the_card_body_scrolls the_card_s_measured_height` | fails `the Equation panel opening: [[336.0 264.0] - [964.0 397.5]]` (the drawing 133.5 pt at 1280 x 800 beside the Equation panel); tests 24 and 25 still pass (`2 passed; 1 failed`) |
| (j) The screen left out of the view (in `CardView::of`, `let screen` becomes `let _screen`, and `screen_width: screen.width().round() as u32,` and `screen_height: screen.height().round() as u32,` become `screen_width: 0,` and `screen_height: 0,`) | `gui::panel` | fails `the_card_is_decided_again_when_the_window_is_resized_with_no_user_choice` at `1280`: `left: (0, 0, false)`, `right: (1280, 800, true)`; every other panel test passes with it (`200 passed; 1 failed`) |
| (k) The body not scrollable (`.enable_scrolling(false)` on its own line after `.id_salt("magcoupling_torque_card_body")`) | `gui::panel` | fails `the_card_body_scrolls_above_the_drawing_s_floor`: `the last row in view`; every other panel test passes with it (`200 passed; 1 failed`) |

- [ ] **Step 6: Docs: `magcoupling-rs/README.md`, `docs/ai/02-system.yaml`, `docs/ai/03-structure.yaml`**

Nine edits; each Old is a substring of one line (several are Task 5's text); replace only it.

(a) README, the `src/gui/` row, `panel.rs`'s entry (Task 5's clause):

Old:
```text
(`torque_card_ui`, slice 3))
```

New:
```text
(`torque_card_ui`, slice 3; open or closed by `card_default` for a `CardView`, its height measured before it is drawn by `card_open_height`, `CARD_OPEN_MIN_WIDTH`))
```

(b) The same row, `geometry_view.rs`'s entry:

Old:
```text
`geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`)
```

New:
```text
`geometry_view.rs` (`geometry_ui`: both views to scale, the callout list, `Transform`, `side_by_side`, `arrowhead`; `list_row` and `drawing_floor`, the room the drawing needs to keep 160 points over three list rows, which the torque card leaves it, slice 3)
```

(c) The card paragraph Task 5 wrote, its last sentence:

Old:
```text
the Auto marking is applied outside the cache).
```

New:
```text
the Auto marking is applied outside the cache). It starts open, but closed while the Equation panel is open (a click on its own Why link opens the panel, so the card then closes), in a centre region under 400 points wide (`CARD_OPEN_MIN_WIDTH`; a 930 x 800 window's), or where the open card, measured before it is drawn, would take the drawing under its floor of 160 points over three list rows; that default is decided again only when the Equation panel opens or closes or the window is resized, never on a design change, and a click on the heading or the arrow is the user's choice for the session (kept by Reset all and every load). Its body scrolls above the drawing's floor.
```

(d) The Tests table's `src/gui/panel.rs` row, after Task 5's clause:

Old:
```text
idle frames change no input with the card open and closed, at the clamp pause |
```

New:
```text
idle frames change no input with the card open and closed, at the clamp pause; slice 3 (T6, headless, each test pausing the clamp section before the card's first frame): at 1280 x 800 the card sits under the Pull-out readout and above the drawing, open, every line drawn and the drawing at least 160 points, at the defaults and the smoke design; at 930 x 800 it starts closed (the region under 400 points), at short magnets too, closed no taller than the label it replaced, and its Why link wraps under the heading inside the region and opens the torque note; at 1080 x 745 no back iron starts closed (its open card would take the drawing under its floor) and short magnets open; with no user choice a window resize decides it again (open at 1280 x 800, closed again at 930 x 800); the Equation panel closes it and its closing opens it, and no view, the card open or closed, paints into the panel; a click on the heading or the arrow wins for the session, through a design edit, a resize, the Equation panel (the drawing keeping its 160 points beside it), Reset all and a load; drags of the gap, the Outside diameter across the fit edge and Required across hot-low, the assumptions banner, and a side panel narrowing the region at no back iron leave it as decided; opened at 930 x 800 its body scrolls (the mouse wheel brings its last row into view) and the drawing keeps 160 points; its measured height is its drawn height within a point at 1280 x 800 and 930 x 1024, and at 930 x 800 its body is the cap (`the_card_s_measured_height_is_the_drawn_height`) |
```

(e) The `src/gui/geometry.rs`, `src/gui/geometry_view.rs` row (spec 9.2's `:687`):

Old:
```text
the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`);
```

New:
```text
the drawing keeping 160 points while the list keeps three rows, then shrinking to scale (`drawing_height`); in a region of exactly `drawing_floor` (slice 3: the room the torque card leaves) the drawing keeps its 160 points, and a point less it shrinks below them;
```

(f) `docs/ai/02-system.yaml`, `:116`, after Task 5's card sentence:

Old:
```text
the Auto marking (the held row, the Auto line) is applied outside the cache.
```

New:
```text
the Auto marking (the held row, the Auto line) is applied outside the cache. Open or closed (spec 6.2): the user's own click on the heading or the arrow wins for the session (card_choice, view state like welcome_dismissed); without it the default (card_start) is decided for a CardView, the Equation panel's state and the screen in whole points read at the top of centre_ui, so a design change, or a side panel widening to its content, never decides it again: open iff the Equation panel is closed, the centre region is at least CARD_OPEN_MIN_WIDTH (400 pt) wide and the open card, pre-measured by card_open_height, leaves the drawing its drawing_floor (160 pt and three list rows); the body is a ScrollArea capped above that floor.
```

(g) `docs/ai/03-structure.yaml`, the `gui` line, `geometry_view.rs`'s clause:

Old:
```text
geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest);
```

New:
```text
geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); list_row and drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows, which the Size tab's torque card leaves the drawing, slice 3);
```

(h) The same line, `panel.rs`'s clause (Task 5's text):

Old:
```text
shared with the controls) and run_sizing)
```

New:
```text
shared with the controls; open or closed by card_default: the user's card_choice for the session, else card_start, the default decided for a CardView (the Equation panel's state and the screen in whole points, read once at the top of centre_ui with the region's width as a CardFrame), open iff the Equation panel is closed, the region is at least CARD_OPEN_MIN_WIDTH wide and the open card, pre-measured by card_open_height, leaves the drawing its drawing_floor; the body a ScrollArea capped above that floor) and run_sizing)
```

(i) The same line, `test_support.rs`'s clause (Task 5's text):

Old:
```text
read by those tests and the torque card's count-row tests), the lookups
```

New:
```text
read by those tests and the torque card's count-row tests), drawing_rect (the geometry drawing's painted rect in a frame, which the torque card's layout tests read), the lookups
```

Run the YAML check of Task 5 Step 7.

Expected: `list` twice.

- [ ] **Step 7: Both suites, clippy both ways, fmt**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app`

Expected: lib `971 passed; 0 failed; 6 ignored` (Task 5's count plus 10: tests 19, 20, 20a, 21, 22, 23, 24 and 25, the resize test, and the floor's test). Every other binary is as in Task 5 Step 8.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml`

Expected: lib `234 passed; 0 failed; 1 ignored`; every other binary as in Task 5 Step 8.

Run: `cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings`, then the same without `--features app`.

Expected: `Finished`, no warning, both.

Run: `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then `cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check`.

Expected: no output, exit 0.

Run: `git status --short`

Expected:
```text
 M docs/ai/02-system.yaml
 M docs/ai/03-structure.yaml
 M magcoupling-rs/README.md
 M magcoupling-rs/src/gui/geometry_view.rs
 M magcoupling-rs/src/gui/panel.rs
 M magcoupling-rs/src/gui/test_support.rs
?? docs/analyses/2026-05-28-press-4bar-analysis.md
```

- [ ] **Step 8: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T6.txt` with the Write tool:
```text
feat(magcoupler): the torque card opens or closes for the view and keeps the drawing's floor (BL-073 T6)

The card's open state follows spec 6.2 (owner 4A). The user's own click on the
heading or the arrow wins for the session (card_choice, view state like the
welcome strip's). Without it the default (card_start) is decided for a
CardView: the Equation panel's state and the screen in whole points, read once
at the top of centre_ui. So a design change, or a side panel widening to its
content, never opens or closes the card. Open iff the Equation panel is closed,
the centre region is at least CARD_OPEN_MIN_WIDTH (400 pt) wide, and the open
card, measured before it is drawn (card_open_height: the header row wrapped as
drawn, each line a wrapped galley), leaves the drawing its floor (drawing_floor
in gui/geometry_view.rs: 160 pt and three list rows, the row one expression
with geometry_ui's, list_row). The body scrolls in a ScrollArea capped above
that floor, the cap also its least height while it scrolls (egui's own,
64 pt, would push the drawing under its floor beside the Equation panel).

Tests 19 to 25 (headless, at 1280 x 800, 930 x 800, 930 x 1024, 1080 x 745,
1080 x 800; each pausing the clamp section before the card's first frame), a
window resize deciding the card again with no user choice, the Equation
panel's isolation with the card open and closed, and the floor's own test.
test_support::drawing_rect reads the drawing's painted rect; no field is
added for a test. Test 21 adds a 1280 x 2000 case where only rule 1 closes the
card. The README and docs/ai/02-system.yaml and 03-structure.yaml follow.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/geometry_view.rs magcoupling-rs/src/gui/test_support.rs magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T6.txt
git status --short
```

Expected: one commit of 6 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 6 moves** (each confirmed in the prototype):

| Test | Old -> new | Why |
|---|---|---|
| `gui::panel::tests::no_centre_view_paints_or_takes_clicks_in_the_equation_panel` | gains the Size tab with the card closed and opened by the user, at each of its five sizes | spec 8.5 test 21: it passes with the card open and closed |
| every Task 5 card test (`the_size_tab_draws_the_torque_card_in_place_of_the_label`, test 10, test 11, the Why-link tests) | unchanged, pass | at their 1280 x 1024 and taller screens rules 1 to 3 open the card, and a heading click is now the user's choice |
| `gui::geometry_view::tests::the_drawing_keeps_its_floor_only_while_the_list_keeps_a_few_rows`, `the_view_stays_in_a_short_region_the_drawing_shrinking_and_the_list_scrolling` | unchanged, pass | `geometry_ui` reads its row through `list_row`, the same expression |

---

### Task 7: G-3 review of every card string, and its record

**Implementer:** session model, with the G-3 review that the controller dispatches inside the task (Step 2, at most three rounds).

Spec section 4 (the text and its traps, 4.4) and 10.1 step 2 (the review's scope); owner decision 15A (a G-3 physics review of every card string before the release) under slice-1 decision G-3. One fresh-context session-model physics review reads every on-screen string slice 3 adds or changes, rendered at every design of spec section 2 in its own Auto state, and at the forms the card takes under each Auto state. The audience is mechanically literate engineers who are not magnet specialists. Nothing may be false or misleading at any design of section 2 or at any reachable input. The words of spec 4.1 are drafts until this task, and every test reads the constants (test 2a builds whole lines from them), so a word this review sets flows through the tests without an edit. Must-fix items are applied where each string lives, once, then re-reviewed: at most three rounds. The record goes into the help-review report under a new `## Slice 3 torque card` section.

This task changes no rule, no number format and none of the owner's own words. Those findings go to the owner at T10's STOP, as spec 2.3 rules for the Minimum OD 45 words ("other words would go beyond 9A, so any change goes to the owner at the STOP"); the table "What this task applies, and what goes to the owner" below gives the rule for each string.

Line numbers are approximate; trust the code and grep for names. Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`.

**Files:**
- Modify, only as the review asks:
  - `magcoupling-rs/src/gui/torque_card.rs`: the constants of spec 4.1; `TorqueCard::lines` only for a ruling on a template or on the one change drawn at two precisions (D3 below);
  - `magcoupling-rs/src/gui/guidance.rs`: `RING_SIZE_STAYS` (spec 4.3); `WALL_NEEDED` and `COLD_HIGH`, which T2 placed beside `check_terms` (plan deviation PD-7);
  - the tests that pin a reworded string by a literal (Step 3's grep; the card's own tests read the constants);
  - the README and `docs/ai` lines that T1 to T6 wrote and that quote a reworded string (Step 5).
- Modify: `docs/analyses/2026-10-14-magcoupler-help-review.md`: append `## Slice 3 torque card` after the last line of `## BL-064 shaft joints`, the file's last section. The working copy is CRLF (`core.autocrlf` true; the index stores LF either way).
- Gitignored outputs: `magcoupling-rs/target/g3/S3-guidance.txt`, `S3-note.txt` and `S3-strings.diff`; `.superpowers/sdd/2026-10-19-magcoupler-slice3/t7-g3-brief-r<N>.md` and `t7-g3-r<N>.md`.
- Unchanged: `tests/data/input_schema.json` (no input's label or help is in scope); `.claude/workflows/gui-smoke.js` (T8 quotes the words this task sets).

**Interfaces:**
- Consumes:
  - T1: `gui/format.rs`'s `signed_fixed`, `signed_change`, `change_decimals`, `CHANGE_EXTRA_DECIMALS` and the one-decimal `signed_percent`; `torque_gap_text` (`gui/size_controls.rs`) on the new rule.
  - T2: `CARD_RANKING_WORDS` and its whole-word matcher (`engine/explain/plain.rs`); `check_terms`, `CheckTerm`, `Quoted`, `SPACE_CLAIM_AXIS_SIZES`, `WALL_NEEDED` and `COLD_HIGH` (`gui/guidance.rs`); the shared test designs of `gui/test_support.rs`.
  - T3 and T4 (`gui/torque_card.rs`): the constants `CARD_HEADING`, `CAPTION`, `AUTO_HOLDS_LENGTH`, `AUTO_HOLDS_RING`, `AUTO_HOLDS_COUNT`, `END_OF_RANGE`, `NOT_ON_THE_SLIDER`, `JUMP`, `CALIBRATION_CHANGE`, `SHOWN_DOES_NOT_FIT`, `SHOWN_NO_PREDICTION`, `SHOWN_NO_HOT_LOW`, `FLIP_FAILS`, `FLIP_PASSES`, `FLIP_CAUTION`, `KEYWAY_BREAKS_THROUGH`, `SIDE_DOES_NOT_FIT` and `SIDE_NO_PREDICTION`; `card`, `TorqueCard::lines`, `held_line`, `Control::label`; and the table `card_designs() -> Vec<CardDesign>`, `CardDesign { key: &'static str, inputs: DesignInputs, minimum_diameter_mm: Option<f64>, auto: Option<FreeVariable> }`, reachable from `gui::guidance`'s tests (in `gui/test_support.rs`, ruling C-9). Its keys are spec 2.1's Key column without the Markdown backticks; a 2.1 row that holds several designs keeps its key for the first and adds ": <what differs>" for each other (for example "(a), Min OD 45: 43.3 mm", "variation 1.0: 1.5", "slider tops: gap 5.0"); T-14's one-flip design and R-13's boss plateau are "T-14 one flip" and "R-13 boss plateau" (plan review 2.2).
  - T5: `RING_SIZE_STAYS`'s 4.3 clause and `magnet_count_line`; the review dump's card block, exactly as below.
  - T6: no on-screen string (`CARD_OPEN_MIN_WIDTH` is a width).
- Produces:
  - the G-3-set words of every string below. T8 quotes `CARD_HEADING`, `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole in `gui-smoke.js`; T9 quotes the final texts in the docs; T10's STOP sheet lists every text;
  - the record (`## Slice 3 torque card`), with its list of items for the owner at the STOP (T10's STOP sheet, spec 10.1 step 6, lists each).

**The review dump's card block: the interface T5 produces.** T5 extends the `#[ignore]` test `guidance_strings_for_review` (`gui/guidance.rs`, the test module's last test) with the block below, right after the "What moves what" block and before `// The Size tab constants.`, and adds one place to its list of Why links. The block is rustfmt's layout; it compiles against Task 4's card module, and Task 5 Step 6 runs it (clippy clean both ways).

```rust
        // The torque card (slice 3; spec sections 4 and 10.1 step 2), for its G-3 review: the
        // heading, then the lines `TorqueCard::lines` draws at every design of `card_designs()`,
        // each in its own Auto state as the panel draws it, at the shipped clamp pause; then every
        // Auto state at the four designs whose card takes a form of its own under Auto (the odd
        // count has no size rows, so Auto on ring size or magnet length holds no row; the three
        // one-line forms draw no Auto line: T-12, test 8). Tag: "card/<key>/<Auto state>/<line>",
        // the line numbered from 00.
        {
            use crate::gui::test_support::card_designs;
            use crate::gui::torque_card::{CARD_HEADING, card};
            let state = |auto: Option<FreeVariable>| {
                auto.map_or_else(|| "Auto off".to_owned(), |v| format!("Auto {v:?}"))
            };
            emit("card/heading", CARD_HEADING);
            let cards: Vec<_> = card_designs()
                .into_iter()
                .map(|d| {
                    let shown = card(
                        &d.inputs,
                        &compute_all(&d.inputs),
                        d.minimum_diameter_mm,
                        crate::engine::clamps::PAUSED,
                    );
                    (d, shown)
                })
                .collect();
            for (d, shown) in &cards {
                for (n, line) in shown.lines(d.auto).iter().enumerate() {
                    emit(&format!("card/{}/{}/{n:02}", d.key, state(d.auto)), line);
                }
            }
            let design = |key: &str| {
                cards
                    .iter()
                    .find(|(d, _)| d.key == key)
                    .unwrap_or_else(|| panic!("card_designs() has no design {key:?}"))
            };
            for key in [
                "odd count",
                "short magnets",
                "shown does not fit",
                "variation 1.0",
            ] {
                let (_, shown) = design(key);
                for variable in FreeVariable::ALL {
                    for (n, line) in shown.lines(Some(variable)).iter().enumerate() {
                        emit(
                            &format!("card/{key}/{}/{n:02}", state(Some(variable))),
                            line,
                        );
                    }
                }
            }
            // The readout's gap above the card, in the card's number format (decision 10B),
            // for the review to read in context.
            for key in ["(a)", "(b)", "(c)", "(e)", "(e2)"] {
                let (d, _) = design(key);
                let results = compute_all(&d.inputs);
                let (gap, _) = sc::torque_gap_text(
                    results.metal.torque_hot_low_Nm,
                    d.inputs.metal.required_min_Nm,
                );
                emit(&format!("size tab/torque gap/{key}"), &gap);
            }
        }

```

and, in the list of Why links further down (spec 9.2: the review dump's place tag gains a place of its own):

```rust
            (
                "size tab/under Magnets per ring and Outside diameter",
                SIZE_TAB_NOTE,
            ),
            ("size tab/the torque card's heading", SIZE_TAB_NOTE),
```

What the sheet then holds, besides the guide strings it printed before:
- `[card/heading]` once;
- for every design of `card_designs()`, `[card/<key>/Auto off/NN]` (or `Auto AxialLength`, `Auto RingRadius`, `Auto MagnetsPerRing` for an Auto design), one tag per line of `lines(auto)`, numbered from `00`: the caption, the Auto line, then the rows; a one-line form has its one line `00`;
- `[card/<key>/Auto <Variable>/NN]` for each of the three Auto variables at "odd count", "short magnets", "shown does not fit" and "variation 1.0" (the Auto designs of 2.3 already show every held row and every Auto line);
- `[size tab/torque gap/<key>]` at (a), (b), (c), (e) and (e2), and `[why link/size tab/the torque card's heading] Why: How the coupling makes torque`;
- the first line under Magnets per ring at `Auto None` and `Auto Some(AxialLength)`, now `RING_SIZE_STAYS` with its 4.3 clause, and nothing at `Auto Some(RingRadius)` (T5's `magnet_count_line`);
- `TOTAL STRINGS:` 305 plus the number of `[card/<key>/` lines: 311 at 69fa505, less T5's 12 neighbour lines and the first line under Magnets per ring at Auto on ring size, plus the heading, the five gap lines and the new Why-link place. With Task 4's table: `TOTAL STRINGS: 564` = 305 + 259; `grep -c '^\[card/'` prints 260.

**The strings under review.** "Drawn" is where the string shows with the clamp section paused (shipped). The approximate places are the spec's; the final lines are what Step 1's grep prints.

| # | Strings | Where they live | Drawn |
|---|---|---|---|
| 1 | `CARD_HEADING` "One step of each size control" | `gui/torque_card.rs`, `pub const CARD_HEADING` | The card's collapsing header on the Size tab, open or closed, in every form; a clickable label that toggles the card (spec 6.1), with the Why link after it |
| 2 | `CAPTION` "The change in the model's hot-low torque after one step of one control, everything else held." (owner 12A's caption line) | `gui/torque_card.rs`, `pub const CAPTION` | The first body line of the rows form |
| 3 | `AUTO_HOLDS_LENGTH`, `AUTO_HOLDS_RING`, `AUTO_HOLDS_COUNT` (owner 5A's Auto line) | `gui/torque_card.rs`, three `pub const` | The second body line while Auto is on that variable |
| 4 | `SHOWN_DOES_NOT_FIT`, `SHOWN_NO_PREDICTION`, `SHOWN_NO_HOT_LOW` | `gui/torque_card.rs`, three `pub const` | Alone under the heading, no caption and no Auto line (T-12) |
| 5 | The side words of this spec: `NOT_ON_THE_SLIDER` (T-4), `JUMP` (O-1), `KEYWAY_BREAKS_THROUGH` (T-4) | `gui/torque_card.rs`, three `pub const`; `KEYWAY_BREAKS_THROUGH` also builds the Size tab's exploration line in `gui/panel.rs` (`format!("{EXPLORATION_ONLY}{KEYWAY_BREAKS_THROUGH}.")`) | A side with no number; the exploration line above the card |
| 6 | `CALIBRATION_CHANGE` "includes a calibration change" (owner 16A's example) | `gui/torque_card.rs`, `pub const CALIBRATION_CHANGE` | Right after a side's number when the calibration factor differs |
| 7 | The flip words `FLIP_FAILS`, `FLIP_PASSES`, `FLIP_CAUTION`, drawn "{guide label}: {word}" | `gui/torque_card.rs`, three `pub const` | A side's flips, in the dashboard's check order |
| 8 | The quoted quantities' names `WALL_NEEDED` "wall needed" and `COLD_HIGH` "cold-high" (T-11), in "{label}: {name} {change} {unit}" | `pub const WALL_NEEDED` and `pub const COLD_HIGH`, beside `check_terms` in `gui/guidance.rs`, where `Quoted.name` reads them (plan deviation PD-7) | A side's margins: "Cup wall: wall needed +0.07 mm", "Gearbox input torque: cold-high +0.03 N·m" |
| 9 | The templates of `TorqueCard::lines`: a row "{label}: {fewer}; {more}"; a side "{step} -> {N·m} N·m ({percent})" then ", " and each cost; a step "{±step} mm" from `Control::step` ("-1 mm", "+0.1 mm", "-2"); a size change "{Outside diameter / Overall length} {change} mm"; a margin "{label} {change} {unit}" or "{label}: {name} {change} {unit}" | `gui/torque_card.rs`, `impl TorqueCard`, `pub fn lines` and the helpers it calls | Every row |
| 10 | `RING_SIZE_STAYS`, its new clause (spec 4.3): "The ring radius stays the same when you change the magnet count; the cup can still change size." | `gui/guidance.rs`, `pub const RING_SIZE_STAYS`, through `magnet_count_line` | The first line under Magnets per ring, Auto off and Auto on the magnet length |
| 11 | The owner's side words (9A): `END_OF_RANGE` "at the end of its range", `SIDE_DOES_NOT_FIT` "magnets do not fit", `SIDE_NO_PREDICTION` "no prediction" | `gui/torque_card.rs`, three `pub const`; `SIDE_DOES_NOT_FIT` also builds the exploration line "Exploration only: magnets do not fit at this diameter." (`gui/panel.rs`) | A side with no number; the exploration line |
| 12 | The numbers' format (decision 10B; T-9, T-14) | `gui/format.rs`: `signed_fixed`, `signed_change`, `change_decimals`, `signed_percent` | Every number on the card; the readout's gap above it |

Read in context only (judged for how the card's words read beside them; changed only where a card word makes them wrong):
- C1: the reused strings of spec 4.2 in their new place: `SET_BY_AUTO` "Set by Auto; turn Auto off to edit" in the held row (`held_line`; `gui/size_controls.rs`, shipped under the locked slider), the control labels (`size_controls::DIAMETER`, `LENGTH`, `label_for("metal.face_gap_mm")`, `label_for("coupling.npole")`), the guide labels (`CheckGuide.label`), the Space claim axis names (`SPACE_CLAIM_AXIS_SIZES`) and the shaft joints' result labels (`label_for`).
- C2: the card's Why link "Why: How the coupling makes torque" (`why_note_link(ui, SIZE_TAB_NOTE)`) and its note `torque_from_shear` (sheet `S3-note.txt`; its own physics review set its words).
- C3: the readout's gap line above the card, "short 0.21 N·m (-8.6 %)" at (a) (T1's format; the words unchanged).
- C4: what leaves the screen: the label the card replaces ("Build checks are shown in the dashboard. The geometry drawing fits the view; use the dimensions to compare sizes."), the neighbour lines under Magnets per ring and `AUTO_RESIZES_RING` / `AUTO_RESETS_LENGTH` (T-8: with the card closed nothing near Magnets per ring says Auto sizes again after a count change).

**What this task applies, and what goes to the owner.**

| A finding on | Applied in this task? | Where it goes |
|---|---|---|
| Strings 1 to 8 and 10 (the drafts of spec 4.1 and 4.3) | Yes: the reviewer's words win within the traps | The constant, once (Step 3) |
| String 9 (a template) or D3 (the one change at two precisions; the spec leaves it to this review) | Yes, with a red test first | `TorqueCard::lines` (Step 3) |
| String 11 (owner 9A's words) or C1 (a shipped string reused) | No | The record's "For the owner at the STOP" list, with the reviewer's proposed text (spec 2.3 Edges, (a) with Minimum OD 45) |
| String 12 (the number format: owner 10B, read by T-9 and T-14) | No | The same list |
| A new rule (a diameter jump rule for R-13's boss plateau, a temperature margin against T-13, a stored-3D marker on a flip, a different routing of the design shown) | No | The same list: each goes beyond the owner's decisions |

- [ ] **Step 1: Locate the strings and write the review sheets**

```bash
git grep -n -A1 -E 'const (CARD_HEADING|CAPTION|AUTO_HOLDS_LENGTH|AUTO_HOLDS_RING|AUTO_HOLDS_COUNT|END_OF_RANGE|NOT_ON_THE_SLIDER|JUMP|COLD_HIGH|CALIBRATION_CHANGE|SHOWN_DOES_NOT_FIT|SHOWN_NO_PREDICTION|SHOWN_NO_HOT_LOW|FLIP_FAILS|FLIP_PASSES|FLIP_CAUTION|WALL_NEEDED|KEYWAY_BREAKS_THROUGH|SIDE_DOES_NOT_FIT|SIDE_NO_PREDICTION): &str' -- magcoupling-rs/src/gui/
git grep -n -e 'pub fn held_line' -e 'pub fn lines' -e 'format!' -- magcoupling-rs/src/gui/torque_card.rs
git grep -n -A1 -e 'pub const RING_SIZE_STAYS' -e 'pub fn magnet_count_line' -e 'SPACE_CLAIM_AXIS_SIZES: \[' -- magcoupling-rs/src/gui/guidance.rs
git grep -n -e 'const EXPLORATION_ONLY' -e '{EXPLORATION_ONLY}' -- magcoupling-rs/src/gui/panel.rs
git grep -n -e 'pub const SET_BY_AUTO' -- magcoupling-rs/src/gui/size_controls.rs
mkdir -p magcoupling-rs/target/g3
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > magcoupling-rs/target/g3/S3-guidance.txt
NOTES_FOR_REVIEW=torque_from_shear NOTES_REVIEW_OUT=target/g3/S3-note.txt cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture > /dev/null
git diff -U0 main -- magcoupling-rs/src/gui/torque_card.rs magcoupling-rs/src/gui/guidance.rs magcoupling-rs/src/gui/panel.rs magcoupling-rs/src/gui/size_controls.rs magcoupling-rs/src/gui/format.rs magcoupling-rs/src/engine/explain/plain.rs > magcoupling-rs/target/g3/S3-strings.diff
grep -c '^\[card/' magcoupling-rs/target/g3/S3-guidance.txt
grep -n -e '^\[card/heading\]' -e '^\[size tab/torque gap/' -e "^\[why link/size tab/the torque card's heading\]" -e '^\[size tab/under Magnets per ring' -e '^TOTAL STRINGS' magcoupling-rs/target/g3/S3-guidance.txt
missing=0
for key in '(a)' '(b)' '(c)' '(e)' '(e2)' 'mixed' 'reverse' '(a)+Auto length' '(a)+Auto count' '(a)+Auto ring' '(e2)+Auto length' '(e2)+Auto count' '(e2)+Auto ring' '(e2)+Auto ring, low Min OD' 'short magnets' 'end effect at a step' 'odd count' '(e), low Min OD' '(a), Min OD 42.8' '(a), Min OD 45' 'shown does not fit' 'keyway through' 'gap at 0.3' 'npole 4' 'hot day' 'only the joint fails' 'passing' 'short cavity' 'temperature edge' 'npole 40' 'variation 1.0' 'zero torque' '(c), f_cal_original -0.95' 'T-14 one flip' 'R-13 boss plateau'; do
  n=$(grep -c -F "[card/$key/" magcoupling-rs/target/g3/S3-guidance.txt); printf '%4d  %s\n' "$n" "$key"; [ "$n" -gt 0 ] || missing=$((missing+1))
done
for prefix in 'slider tops' 'steps to an end'; do
  n=$(grep -c -F "[card/$prefix" magcoupling-rs/target/g3/S3-guidance.txt); printf '%4d  %s\n' "$n" "$prefix"; [ "$n" -gt 0 ] || missing=$((missing+1))
done
echo "missing: $missing"
```
Expected:
- Each `git grep` prints its lines (rustfmt puts a long literal on the line after its `const`, which `-A1` shows). Copy them into the brief.
- The dump ends `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 976 filtered out` (after Task 6), and the notes dump writes `S3-note.txt` (`torque_from_shear` at the five probe designs, starting `==== torque_from_shear (Reviewed)`).
- The sheet holds what "The review dump's card block" above lists: `[card/heading] One step of each size control`, the five gap lines, the card heading's Why-link place and `TOTAL STRINGS:` 305 plus the `[card/<key>/` lines. Measured after Task 6: `grep -c '^\[card/'` prints `260`, and the `grep -n` prints:
  ```text
  214:[size tab/under Magnets per ring/Auto None] The ring radius stays the same when you change the magnet count; the cup can still change size.
  215:[size tab/under Magnets per ring/Auto Some(AxialLength)] The ring radius stays the same when you change the magnet count; the cup can still change size.
  248:[card/heading] One step of each size control
  508:[size tab/torque gap/(a)] short 0.21 N·m (-8.6 %)
  509:[size tab/torque gap/(b)] margin +0.00 N·m (+0.0 %)
  510:[size tab/torque gap/(c)] short 1.06 N·m (-42.3 %)
  511:[size tab/torque gap/(e)] short 0.83 N·m (-33.1 %)
  512:[size tab/torque gap/(e2)] margin +0.17 N·m (+11.5 %)
  553:[why link/size tab/the torque card's heading] Why: How the coupling makes torque
  567:TOTAL STRINGS: 564
  ```
- The loop prints a count of at least 1 for every key (at least 3 for a rows design: the caption and its rows) and ends `missing: 0`. Measured after Task 6:
  ```text
     5  (a)
     6  (b)
     5  (c)
     5  (e)
     5  (e2)
     5  mixed
     5  reverse
     6  (a)+Auto length
     6  (a)+Auto count
     6  (a)+Auto ring
     6  (e2)+Auto length
     6  (e2)+Auto count
     6  (e2)+Auto ring
     6  (e2)+Auto ring, low Min OD
     4  short magnets
     5  end effect at a step
    15  odd count
     5  (e), low Min OD
     5  (a), Min OD 42.8
     5  (a), Min OD 45
     4  shown does not fit
     5  keyway through
     5  gap at 0.3
     5  npole 4
     5  hot day
     5  only the joint fails
     5  passing
     5  short cavity
     5  temperature edge
     5  npole 40
     4  variation 1.0
     1  zero torque
     5  (c), f_cal_original -0.95
     5  T-14 one flip
     5  R-13 boss plateau
    20  slider tops
    45  steps to an end
  missing: 0
  ```
  A missing key is a design `card_designs()` lacks (or holds under another key): stop and hand it to the controller (T4's table), since the reviewer must see every design 10.1 step 2 names.

- [ ] **Step 2 (controller): the G-3 review, one fresh-context reviewer, at most three rounds**

Dispatch it with the Agent tool: subagent `general-purpose`, `model` omitted (session model: physics, CLAUDE.md section 5), fresh context. Save the brief as `.superpowers/sdd/2026-10-19-magcoupler-slice3/t7-g3-brief-r<N>.md` and each answer verbatim as `t7-g3-r<N>.md` there. Round 1's brief, verbatim, with Step 1's `git grep` output pasted where it says so:

```text
You are a fresh-context physics reviewer (session model) doing the G-3 review of slice 3's new and changed on-screen
texts, the torque card, in the magnetic coupling calculator (crate magcoupling-rs; repo
C:/Users/Cole/source/repos/linkage_simulation, branch magcoupling/slice3-torque-card). Read-only: do not edit the
repository and run no git command that changes state. Probe the engine only from a scratch crate outside the
repository, in your scratchpad, with its own CARGO_TARGET_DIR there (if a link step fails with LNK1104, retry once,
then use a shorter CARGO_TARGET_DIR). Its Cargo.toml:
    [package]
    name = "slice3g3"
    version = "0.1.0"
    edition = "2024"
    publish = false
    [dependencies]
    magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs", features = ["gui"] }
    [workspace]
The library is `magcoupling`. The card of a design `inputs` is
    let results = magcoupling::compute_all(&inputs);
    let card = magcoupling::gui::torque_card::card(&inputs, &results, minimum_od_mm, magcoupling::engine::clamps::PAUSED);
    let lines: Vec<String> = card.lines(auto);
with `minimum_od_mm` the Size tab's Minimum OD preference (None: unset), `auto` None or
Some(magcoupling::engine::sizing::FreeVariable::{AxialLength, RingRadius, MagnetsPerRing}), and the heading
magcoupling::gui::torque_card::CARD_HEADING above the lines. A design shown under Auto is
magcoupling::gui::sizing::sized(variable, &inputs, value) at the solved value spec section 2.1 gives. The test
designs in gui::test_support are crate-private: build every design from spec section 2.1's Edits column. The guards:
magcoupling::engine::explain::plain::{plain_text_violations, mentions_gearbox_protection, TORQUE_ORDER_WORDS,
CARD_RANKING_WORDS} and the whole-word matcher beside CARD_RANKING_WORDS. The first build takes several minutes. Use
no Markdown heading in your answer (no line that starts with "#"): it is pasted verbatim into a report whose "## "
lines are section boundaries.

Audience: mechanically literate engineers who are not magnetics specialists. Plain words. Every string must be true
and not misleading at every design of spec section 2 and at every reachable input: every value inside its slider,
the typed values a design file or link can carry that the spec names (an odd magnet count, a variation of 1.0 or
1.5, a magnet Br of 0, a calibration coefficient of 0 or -0.95, a Minimum OD above the diameter shown), a design
file's value past a slider's end (the face gap above 5.0 mm, an axial override past the length range, a count of 2),
and a design file's base that no slider reaches, near zero or huge (a variation of 1 - 1e-10, a manual inner Br of
1e-6 T, a calibration coefficient of 1e-9 or 1e-300 with no back iron at 8 magnets or of 1e300, a Required torque of
1e-300). The sliders:
Outside diameter (its range from the Minimum OD preference, up to 80 mm; the Minimum OD slider runs to 80 mm),
Overall length (21.10-69.90 mm at the defaults), Magnet face gap 0.3-5.0 mm (0.01 mm steps), Magnets per ring 4-40
(even counts).

What slice 3 is: the spec docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md (read sections 1, 2,
3.3, 3.6, 4, 5, 6.1 and 10; section 1.1 holds the owner's binding decisions 1A-17A, section 1.3 the answered owner
questions O-1 and O-2). The Size tab gains one card between the torque readouts and the drawing. For each of the four
Size controls (Outside diameter +-1 mm, Overall length +-1 mm, Magnet face gap +-0.1 mm, Magnets per ring +-2) it
shows what one step each way does to the model's hot-low torque at the gearbox output, in N·m and %, with what the
step costs: the change in Outside diameter and Overall length, the dashboard checks whose level flips, and the change
of each failing check's own quantity (never a Temperature limits quantity: T-13). Each side is the design the slider's
own mapping makes at exactly one step; every number is a difference of two compute_all runs. A side with no honest
number shows words: the magnets do not fit, no prediction, the end of the slider's range, not a value the slider
takes (an odd count), the keyway breaks through the hub, or not a step (the overlapping magnet length changes by
anything other than the step: O-1). The design shown decides first: if its magnets do not fit, it has no
prediction, or its pull-out or hot-low is not above zero, one line stands alone. While Auto is on, the held
control's row reads "Set by Auto; turn Auto off to edit" and one line says Auto sizes again after a change (5A). A
step that changes the calibration factor shows the true number with a short note (16A). Numbers follow 10B: one
decimal of percent; N·m to the fewest decimals that keep the true sign and are not zero, at most two more than usual
(T-9); a row's two N·m at one precision (T-14).

The strings under review, with file and line (from git grep):
<paste Step 1's git grep outputs>
 1 CARD_HEADING, the card's heading (a clickable label that opens and closes the card, with its Why link after it);
 2 CAPTION, the first line of the rows form (owner 12A: "the model's numbers for one step, everything else held");
 3 AUTO_HOLDS_LENGTH, AUTO_HOLDS_RING, AUTO_HOLDS_COUNT, the Auto line;
 4 SHOWN_DOES_NOT_FIT, SHOWN_NO_PREDICTION, SHOWN_NO_HOT_LOW, each drawn alone;
 5 the side words NOT_ON_THE_SLIDER, JUMP and KEYWAY_BREAKS_THROUGH (the last also builds the Size tab's
   exploration line "Exploration only: the keyway breaks through the hub.");
 6 CALIBRATION_CHANGE;
 7 FLIP_FAILS, FLIP_PASSES, FLIP_CAUTION, drawn "{check}: {word}";
 8 WALL_NEEDED and COLD_HIGH, the names of two quoted quantities;
 9 the templates of TorqueCard::lines: a row "{label}: {fewer}; {more}"; a side "{step} -> {N·m} N·m ({percent})" then
   ", " and each cost; a step "-1 mm", "+0.1 mm", "-2"; a size change "{Outside diameter | Overall length} {change}
   mm"; a margin "{check} {change} {unit}" or "{check}: {name} {change} {unit}";
10 RING_SIZE_STAYS's new clause (spec 4.3), under Magnets per ring with Auto off and on the magnet length;
11 the owner's side words (decision 9A): END_OF_RANGE, SIDE_DOES_NOT_FIT (also the exploration line "Exploration
   only: magnets do not fit at this diameter.") and SIDE_NO_PREDICTION;
12 the format of every number on the card (decision 10B, T-9, T-14), and the readout's gap above it.

What you may change, and what goes to the owner. Your words win over the drafts for strings 1 to 10, within the
traps; for 9 say the line as it would read. Strings 11 and 12 are the owner's own decisions, and the context lines
below are shipped strings in a new place: judge them where they stand, and if one is false or misleading there, write
the must-fix with your proposed text marked "for the owner"; it goes to the owner at the release stop, not into this
change (the spec rules so for the Minimum OD 45 words, D5). A new rule (a diameter jump rule for the boss plateau
R-13, a temperature margin against T-13, a stored-3D marker on a flip, a different routing of the design shown) goes
beyond the owner's decisions: propose it "for the owner".

Read in context only:
- C1: the reused strings in their new place: "Set by Auto; turn Auto off to edit" in the held row; the control
  labels "Outside diameter", "Overall length", "Magnet face gap", "Magnets per ring"; the dashboard check labels; the
  Space claim axis names "Rotating diameter", "Overall length", "Large-diameter stack"; the shaft joints' result
  labels (e.g. "Inner hub smallest measured interference that holds").
- C2: the card's Why link "Why: How the coupling makes torque" and its note torque_from_shear (sheet S3-note.txt;
  its own physics review set its words; the note compares a millimetre of magnet length with a millimetre of ring
  radius, while the card's Outside diameter step is about 0.48 mm of ring radius: spec 4.2, for the lessons slice).
- C3: the readout's gap line above the card in the new format ("short 0.21 N·m (-8.6 %)" at the defaults).
- C4: what leaves the screen: the label the card replaces ("Build checks are shown in the dashboard. The geometry
  drawing fits the view; use the dimensions to compare sizes."), the N-2 / N+2 lines under Magnets per ring and their
  two Auto first lines; with the card closed (a 930 x 800 window, or the Equation panel open) nothing near Magnets per
  ring says Auto sizes again after a count change (T-8).

Sheets in magcoupling-rs/target/g3/:
- S3-guidance.txt (the ignored test guidance_strings_for_review): every guide string as before, and the card:
  "[card/heading]"; "[card/<design key>/<Auto state>/<line number>]" for every design of spec section 2.1 in its
  own Auto state, then each Auto state at the odd count and the three one-line forms; "[size tab/torque gap/<key>]";
  "[why link/size tab/the torque card's heading]";
- S3-note.txt: the note torque_from_shear at the five probe designs, for consistency;
- S3-strings.diff: the source diff against main of torque_card.rs, guidance.rs, panel.rs, size_controls.rs,
  format.rs and plain.rs.

Read the code behind every string: gui/torque_card.rs whole (card, the order of the tests of spec 3.3, side_words,
count_step, shows, lines, held_line); gui/guidance.rs: check_terms, WALL_NEEDED, COLD_HIGH, RING_SIZE_STAYS,
magnet_count_line, SPACE_CLAIM_AXIS_SIZES; gui/format.rs: signed_fixed, signed_change, change_decimals, signed_percent;
gui/size_controls.rs: torque_gap_text, predicts, SET_BY_AUTO; gui/panel.rs: torque_card_ui and the exploration lines.

Traps (each one written in a string is a must-fix; sources: the spec's section 4.4, slice-1 spec section 8, slice-2
spec section 8):
 1. Ranking the controls: "X is the strongest / biggest / best lever", any comparison between rows. The card ban
    list CARD_RANKING_WORDS: strongest, biggest, best, most, lever, dominant, ranks, matched as whole words in any
    case with an s added or taken off ("levers" and "Rank" match; "almost" does not). Rows keep the controls'
    fixed order (6A). Slice 1's "radius is the strongest lever at a fixed pole count" is one form of it.
 2. A small change shown as "+0 %" or "-0.00"; a number whose sign disagrees with its percent; two numbers that read
    equal where they are not (BL-068); two sides of a row whose N·m misstate their ratio ("-0.01" against "-0.005"
    for a true 1.35:1, T-14).
 3. Extrapolating a step: "each mm adds X", any rate per mm or per magnet. Each side states its step and its own
    change only.
 4. A jump read as a slope. A length side whose overlapping magnet length does not change by the step reads JUMP
    (O-1, the short cavity's +1 mm side included). Not guarded, and on the owner's sheet: the diameter row's boss
    plateau (R-13; the sheet's `[card/R-13 boss plateau/` lines).
 5. "Torque grows as R_g²" without "at fixed shear stress"; "a bigger coupling gives more torque" (at 8 magnets on
    the default ring both diameter steps lower hot-low).
 6. "Fewer poles reach across the gap better"; "fewer poles keep the same size" (at 8 magnets the count -2 grows the
    Outside diameter 2.2 mm); "fewer poles always improve torque per mass".
 7. The calibration switch explained as physics. CALIBRATION_CHANGE names the change, never a cause (16A).
 8. f_cal, c_end, the variation, μ0 or the bench inputs presented as things to change; "the end factor is
    accurate"; "edit the calibration inputs to match your design"; "the bench calibration applies beyond
    prototype-matching designs".
 9. "Guaranteed" pull-out; any ordering of the torques (TORQUE_ORDER_WORDS: "strongest", "most expected", "most
    torque expected"); "the rings re-lock as soon as the load falls below pull-out" (they re-lock only when the slip
    speed falls). Every N·m on the card is hot-low at the output except the gearbox phrase's, which names its own
    torque (T-11) and compares none.
10. "More torque is safer"; "the coupling protects the gearbox" (mentions_gearbox_protection): cold-high exceeds the
    gearbox input rating at the defaults, and nothing checks the drive line beyond that rating.
11. A change near 0 % called "no effect", "free" or "negligible": every cost is listed and the true sign kept.
12. "A hotter-rated grade is free"; stored 3D fields that follow the design, its geometry or its grade. The card
    quotes no Temperature limits quantity (T-13); its Temperature limits flip is the dashboard's own verdict on the
    stepped design.
13. An elasticity in °C.
14. Under Auto, "the torque rises" where Auto would size again after the change.
15. "clamp" in any case while the clamp section is paused (shipped). Unpaused, the card may name "Shaft clamp"
    (test 7a); no shipped panel is unpaused.
16. The other slice-2 traps, none expected here: "each ring uses its own α"; "416, 17-4PH or 12L14 are clean
    swaps" or "416 stainless is a safe back-iron fix"; "steel gives the torque shown" without the saturation caveat;
    "more mass lowers the steady temperature"; "the coupling can slip indefinitely"; "the clamp is sized to the
    drive"; "raising Boss outside diameter enlarges the drawn boss"; "model ferrite by typing a positive β"; "the
    expansion caution counts the temperature swing"; "1.5 T is 4140's saturation".
17. Plain words and glyphs: no cell id, no backlog or decision id ("BL-073", "9A"), no code name ("ShownNoHotLow",
    "f_cal"); plain_text_violations empty; "->", not U+2192; negatives with the formatter's ASCII "-"; "N·m", "µm",
    "°C" as the house writes them.
18. No number typed in any text: every number comes from the results or from a control's step (spec 4: "no literal
    number is typed in a text").

Decisions the spec hands to you; rule each KEEP or CHANGE, with the reason:
D1. T-11 (R-12): "cold-high" in the gearbox phrase, "Gearbox input torque: cold-high +0.03 N·m" at the defaults'
    Outside diameter +1 mm. It is the one N·m on the card that is not hot-low at the output, and it names a third
    torque on the Size tab (the dashboard's Problem calls it "Cold-high at the gearbox input"). Keep it, reword it,
    or drop the name ("Gearbox input torque +0.03 N·m").
D2. SHOWN_NO_HOT_LOW, at variation 1.0 (the readouts above show the pull-out 2.69 N·m and hot-low 0.00 N·m), at
    variation 1.5 (hot-low -1.34 N·m), at zero torque (a magnet Br of 0: both 0.00 N·m), at a calibration
    coefficient of -0.95 (both below zero) and at -0.95 with variation 1.5 (pull-out -2.69, hot-low +1.34 N·m): true
    in every case routed to it?
D3. One change at two precisions (spec 3.6): at 4 magnets the gap row draws "Outside diameter -0.3 mm" and "Space
    claim: Rotating diameter -0.28 mm" on one side, and the count +2 side "Outside diameter -9.3 mm" beside "Rotating
    diameter -9.31 mm"; the diameter row's "Space claim: Rotating diameter +1.00 mm" repeats the row's own step (the
    smoke design (b) and 4 magnets). Keep both (the spec's choice: each is true and says something the other does
    not; the Rotating diameter is the larger of the cup and the cap, while the Outside diameter also counts the shaft
    boss), or (A) draw no size change whose quantity a quoted axis on the same side states, or (B) draw no quote that
    repeats the row's own step.
D4. "everything else held" (CAPTION) beside rows that list size changes (at 8 magnets the count -2 side's "Outside
    diameter +2.2 mm"; the diameter row's cap following the cup: consequences of the step, not held inputs), and
    beside the reverse pair's length row (B842SH inner, B442 outer): each step trims the longer inner ring 12.7 ->
    5.35 / 7.35 mm and the hub with it (13.0 -> 5.65 / 7.65 mm) with no flip and no quoted cost, the one place where
    the caption is not literally true and no cost shows it. Its number is the model's honest value (the model has no
    overhang term); BL-074 records the trim.
D5. The Minimum OD 45 words: at the defaults with Minimum OD 45.0 mm both Outside diameter sides read "at the end of
    its range" (41.80 and 43.80 mm lie below the slider's 45.00-80.00 mm range) while the slider's handle sits at
    its bottom end and moves up; at 43.3 mm only the -1 mm side does. END_OF_RANGE is the owner's word (9A): say
    whether it misleads here and what you would write, "for the owner". And where the design shown itself lies
    past an end (the face gap at 5.5 mm, an axial override of 60 mm): both sides read END_OF_RANGE; say whether it
    misleads, "for the owner".
D6. The temperature edge (the defaults with temperature.duty.hot_ambient_C 82.08): the Outside diameter +1 mm,
    Overall length +1 mm and count -2 sides flip "Temperature limits: caution" (the slip-fault margins move about
    0.04-0.06 °C a step, partly through the stored slip-heating field). The card's flip carries no stored-3D marker;
    the dashboard row does. Is the flip as written true and not misleading?
D7. No back iron with calibration.f_cal_original -0.95 (a design file's value): "Magnets per ring: -2 -> -2.29 N·m
    (-158.9 %), includes a calibration change, Shaft joints: passes; +2 -> magnets do not fit". A side below zero
    shown as its number with the note (16A); "Shaft joints: passes" is the dashboard's own verdict on the stepped
    design.
D8. RING_SIZE_STAYS's new clause, "...; the cup can still change size." (the old "the cup OD below can still change"
    pointed at the neighbour lines, which move into the card).
D9. The words beyond 9A (T-4, R-4): KEYWAY_BREAKS_THROUGH (four sides at a 16 mm bore with a 2.5 mm keyway; it also
    builds the shipped exploration line) and NOT_ON_THE_SLIDER (both count sides at an odd count of 9, a design
    file's).
D10. The one-line forms draw no caption and no Auto line, Auto on or off (T-12).

Caps and the tests that hold the words:
- No length cap, but every extra wrapped line costs height. At 1280 x 800 the open card at the smoke design (196 pt)
  leaves the drawing's list every row with 7.4 pt to spare, and 172.9 pt before the card would start closed (tests
  19 and 25 fail first). Prefer the draft's length or shorter, and say what a longer text costs.
- The tests read the constants (test 2a builds whole lines from them), so a word change flows through. Test 16 runs
  the ban list, TORQUE_ORDER_WORDS, mentions_gearbox_protection, "clamp" and plain_text_violations over every line at
  every design and Auto state; test 17 the glyphs.
- The gui-smoke prompt (plan T8) quotes CARD_HEADING, CAPTION, AUTO_HOLDS_LENGTH and the held Overall length row
  after this review: nothing to keep for it.

Probe at least, from compute_all and torque_card::card:
(1) the card at the defaults (a), the smoke design (b), no back iron (c), 8 magnets (e) and (e) with Required 1.5
    N·m (e2), against spec section 2.3's tables and drafted lines; the Auto designs of 2.3, each held row and Auto
    line;
(2) the short cavity (cup_depth_mm 11.0): its Overall length row reads JUMP both sides (active length 12.70 -> 10.00
    and 12.70 -> 12.00 mm; the +1 mm side would have printed -0.1406 N·m, -6.15 %); also 12.0 and 12.5 mm;
(3) the designs of D1 to D10;
(4) the mixed pair (B442 inner, B842SH outer, blank override): JUMP both ways (the numbers it would have shown:
    +106.39 % and +146.16 %);
(5) end effect at a step (c_end 0.5, axial override 5.0 mm): "no prediction" on the length -1 mm and count -2
    sides; the length +1 mm side +0.20 N·m (+167.9 %), true by the model (R-7);
(6) the odd count (9): no size rows; both count sides NOT_ON_THE_SLIDER;
(7) the slider ends: the gap at 0.3 (-0.1 mm), 4 magnets (-2), 40 narrow magnets (+2), the gap at 5.0 (+0.1 mm), the
    Outside diameter at 80 mm (+1 mm), the Overall length at 69.90 (+1 mm) and 21.10 mm (-1 mm): "at the end of its
    range"; one step inside each end: 68.90 mm +1 mm "+0.20 N·m (+2.1 %)"; at 22.10 mm the Outside diameter +1 mm
    "+0.004 N·m (+1.1 %)", the smallest number side (T-9);
(8) the one-line forms: short magnets, shown does not fit (ring 9.5 mm), variation 1.0, zero torque, each with the
    readouts above the card;
(9) hot day, only the joint fails and the passing design: the flips and the shaft-joint quotes ("Shaft joints:
    Inner hub smallest measured interference that holds -0.04 µm");
(10) (e2) with Auto on the magnet length, then Auto off with Required between its two diameter sides' hot-low
    (1.494977 and 1.495232 N·m): both sides read "-0.005 N·m (-0.3 %)" while only one flips Hot-low torque (T-14's
    row, on the owner's sheet; the sheet's `[card/T-14 one flip/` lines);
(11) the readout's gap above the card (S3-guidance.txt, "size tab/torque gap/<key>");
(12) the defaults with coupling.npole 2 (a design file's): what the card draws, and whether any number on it is
    physically meaningless (a new rule is the owner's);
(13) the same class's other members, each from the defaults (the owner's rule too, as (12)): no back iron at 8 magnets
    with calibration.f_cal_original 1e-9 and 1e-300 (the count +2 side, which the calibration switch takes from about
    9e-10 N·m to 1.44 N·m); metal.variation 1 - 1e-10, and both parts blank with
    coupling.magnets.manual_inner_br_T 1e-6 (every side's N·m at T-9's cap beside its percent, under a hot-low readout
    of about 2.7e-10 or 1.8e-6 N·m); calibration.f_cal_original 1e300 (the card's lines against the readouts'
    2.830e300 and 2.405e300 N·m); metal.required_min_Nm 1e-300 (the readout's gap): what each line reads and whether
    any number on it misleads, "for the owner".

Return, for the record. It is pasted verbatim into docs/analyses/2026-10-14-magcoupler-help-review.md, under
"## Slice 3 torque card", so no line of it may start with "#".
1. "Reviewer: physics reviewer (session model), fresh context, <date>; round <n>."
2. Read: what you read; the probes you ran, with their numbers.
3. A table, one row per string (its number and constant, as listed above): PASS or MUST-FIX, with the reason.
4. Each MUST-FIX: the string's place (constant or template), Old, New (the exact text), Why (the engine fact and
   the probe that make the old text false or misleading) and, where a claim can be pinned, the test assertion. Mark
   each "applied here" (strings 1 to 10) or "for the owner" (strings 11 and 12, the context lines, a new rule).
5. SHOULD-FIX offered, in the same form.
6. D1 to D10: KEEP or CHANGE, each with its reason; a CHANGE in the form of 4.
7. Traps checked, 1 to 18, one line each. Typed numbers: every number typed in a string (none is allowed).
8. The context lines: fine, or why not.
9. Verdict: PASS or MUST_FIX. MUST_FIX only for items applied here; the items for the owner are listed apart and do
   not hold the verdict.
Your words win over the drafts within the traps: write the text you would ship.
```

Rounds 2 and 3 use the same brief with three changes, and each round is a new fresh context:
- add the round's changes, each string old -> new, and the tests Step 3 added;
- add "Re-review only the changed strings and every string whose meaning depends on them; round 1's findings on the
  unchanged strings stand";
- give the earlier answers' paths.

- [ ] **Step 3: Apply the must-fix items: constants first, tests follow through the constants**

Apply only the items marked "applied here" (the table "What this task applies, and what goes to the owner"). Copy each item marked "for the owner" into the record's list (Step 4). Edit each applied string only where it lives, once:
- **A constant of strings 1 to 8** (`gui/torque_card.rs`; `WALL_NEEDED` and `COLD_HIGH` in `gui/guidance.rs`): change the literal. The tests follow through the constant: test 2a (whole lines built with `format!` from the constants), test 4 (the one-line forms), tests 15 to 18 (every design of `card_designs()`), and `smaller_exploration_shows_magnet_fit_and_keyway_breakthrough` (`gui/panel.rs`, built from `EXPLORATION_ONLY`, `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH`). A new `KEYWAY_BREAKS_THROUGH` also rewords the shipped exploration line above the card: add that line, old -> new, to the record's list for the STOP sheet (spec 4.3).
- **D1 ruled "drop the name"**: the gearbox term's `Quoted.name` becomes `None` in `check_terms`' gearbox arm (`gui/guidance.rs`), so its phrase reads "{check} {change} {unit}", and `COLD_HIGH` goes. Its five other readers change in the same red-first edit: T4's test 2 (`cold = Some(COLD_HIGH)`), T4's test 2a (`{gearbox}: {COLD_HIGH} {}`), T4's test 7 (`(label(GEARBOX), Some(COLD_HIGH))`), T2's test 9 (`quote("metal.cold_high_input_Nm", Some(COLD_HIGH), 2)`) and `check_terms`' gearbox arm itself, with the card test module's import of `COLD_HIGH`; then `git grep -n COLD_HIGH -- magcoupling-rs/src` prints nothing. Test 7's expected gearbox phrase ("Gearbox input torque: cold-high +0.03 N·m" at (a)'s Outside diameter +1 mm) changes in the same edit, built as `format!("{} {} {}", label, signed_change(change, 2), unit)` from the guide's label and the result's unit: see it fail on the old code first.
- **`RING_SIZE_STAYS`** (string 10, `gui/guidance.rs`): change the literal. The tests read the constant: `the_first_magnet_line_follows_the_auto_state`, `no_guide_text_calls_the_ring_radius_ring_size`, `size_tab_first_magnet_line_follows_the_auto_state` and the line-count tests T5 moved; check with `git grep -n -F "the cup can still change size" -- magcoupling-rs/src` (only the constant).
- **A template of string 9, or D3 ruled (A) or (B)**: the change is local to `TorqueCard::lines` (spec 3.6). Test first, in test 2a (`the_card_s_lines_read_as_drafted`): write the expected line in the ruled form and see it fail on the old `lines`, then change `lines`, then see it pass.
  - D3 (A), "no size change whose quantity a quoted axis on the same side states": add 4 magnets' (`with_poles(4)`) Magnet face gap row to test 2a, its `-0.1 mm` side without "Outside diameter -0.3 mm" and with "Space claim: Rotating diameter -0.28 mm" (each part built with `format!` from `size_controls::DIAMETER`, the axis name of `SPACE_CLAIM_AXIS_SIZES` and `signed_change` of the side's own fields), and its count `+2` side without "Outside diameter -9.3 mm".
  - D3 (B), "no quote that repeats the row's own step": test 2a's (b) lines (already pinned) lose "Space claim: Rotating diameter +1.00 mm" on the Outside diameter row's `+1 mm` side; add 4 magnets' Outside diameter row without the quote on either side.
  - A separator or step text (" -> ", "; ", ", ", the step form): test 2a's `format!` strings change with it, since they type the templates.
- **A must-fix whose claim a test can check** (the brief's item 4 names the assertion): write that assertion first, in the card's test module, see it fail on the old text, then apply the text.

Then:
- run `git grep -n -F "<old text>" -- magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai .claude/workflows` for each changed string. Expected: only the constant's own line, and any README or `docs/ai` line that Step 5 updates. A test literal that has a constant takes the constant;
- run `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib`: `test result: ok.`;
- rewrite the sheets (Step 1), commit the round's fixes (Step 7, "a round that changed strings"), then dispatch the next round (Step 2).

At most three rounds. If round 3 still returns a must-fix marked "applied here", **stop** and report to the controller with the open items: it is an owner decision. Should-fix items stay offered and are not applied unless the controller says so.

- [ ] **Step 4: Write the record in `docs/analyses/2026-10-14-magcoupler-help-review.md`**

Append the block below at the end of the file, after the last line of `## BL-064 shaft joints`. Angle-bracket slots are filled from the reviewer's answers and from Step 3; everything else is verbatim. Leave out "Round 2" and "Round 3" if they did not run. If an answer has a line starting with "#" despite the brief, indent that line by two spaces and say so where the reports begin. The only new line that starts with "## " is the section's heading. The block is indented by two spaces in this plan only, so that its heading lines do not read as the plan's own sections: write it into the file without that indent.

  ```markdown
  ## Slice 3 torque card

  Slice 3's G-3 review of every new and changed on-screen text of the torque card (plan
  `docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md`, Task 7) is recorded in this section, with each
  round's report as returned.

  - Scope: the torque card's strings (spec `docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md`,
    sections 4 and 10.1 step 2), reviewed under the same G-3 rule:
    - `gui/torque_card.rs`: `CARD_HEADING`, `CAPTION`, the Auto lines (`AUTO_HOLDS_LENGTH`, `AUTO_HOLDS_RING`,
      `AUTO_HOLDS_COUNT`), the one-line forms (`SHOWN_DOES_NOT_FIT`, `SHOWN_NO_PREDICTION`, `SHOWN_NO_HOT_LOW`), the
      side words (`NOT_ON_THE_SLIDER`, `JUMP`, `KEYWAY_BREAKS_THROUGH`), `CALIBRATION_CHANGE`, the flip words
      (`FLIP_FAILS`, `FLIP_PASSES`, `FLIP_CAUTION`) and the templates of `TorqueCard::lines`;
    - `gui/guidance.rs`: the quoted quantities' names (`WALL_NEEDED`, `COLD_HIGH`) and `RING_SIZE_STAYS`'s new clause;
    - judged in place, for the owner where they need a change: the owner's side words (`END_OF_RANGE`,
      `SIDE_DOES_NOT_FIT`, `SIDE_NO_PREDICTION`) and the number format (decision 10B, T-9, T-14);
    - read in context: the reused strings in their new place (`SET_BY_AUTO` in the held row, the control, check, axis
      and shaft-joint labels), the card's Why link and its note `torque_from_shear`, the readout's gap line, and what
      leaves the screen (the replaced label, the neighbour lines and their two Auto first lines).
  - Read: the sheets `magcoupling-rs/target/g3/S3-guidance.txt` (`guidance_strings_for_review`: the card at every
    design of the spec's section 2.1 in its own Auto state, each Auto state at the odd count and the three one-line
    forms, the gap lines), `S3-note.txt` (`notes_for_review`, `torque_from_shear`) and `S3-strings.diff`
    (`git diff -U0 main` of the source files); the spec, sections 1 to 6 and 10; the code behind every string. Each
    round probed the engine from a scratch crate outside the repository:
    - round 1 (<date>): <the probes, with their numbers, as returned>;
    - round 2 (<date>): <as returned>.
  - The decisions the spec handed to this review (D1 to D10): <each, KEEP or CHANGE, as ruled>.
  - Round 1: <n> must-fix items applied here, text applied verbatim (constants first, tests through the constants):
    - `<the constant or template>`.
      - Old: "<text>"
      - New: "<text>"
      - Why: <the engine fact and the probe, as returned>.

    Should-fix offered, not applied (open for the controller): <"none", or SF-n with its text and why>.

    Recorded, no change asked: <the reviewer's observations, as returned>.
  - Round 2: <PASS, no must-fix | n must-fix items, as above>. It re-reviewed the changed strings and every string
    whose meaning depends on them; round 1's findings on the unchanged strings stand.
  - For the owner at the STOP (T10's STOP sheet lists each): <"none", or each item marked "for the owner": the string, its
    text, the reviewer's proposed text and why; and every shipped line a changed constant rewords, such as the
    exploration line built from `KEYWAY_BREAKS_THROUGH`>.
  - What happened to the should-fix items: <applied (where) or left open, each>.
  - Tests that followed the changes: <each test changed or added, and why | "none: every test reads the strings
    through their constants">.
  - Traps: none of the 18 is written in any card string. <Or the trap a round found, and its fix.>
  - Verdict: PASS (round <n>).

  ### Report for the record, round 1 (as returned)

  <round 1's answer, verbatim>

  ### Report for the record, round 2 (as returned)

  <round 2's answer, verbatim>
  ```

Run:
```bash
awk '/^## Slice 3 torque card/{s=1; next} s && /^## /{print NR": "$0}' docs/analyses/2026-10-14-magcoupler-help-review.md
grep -n '^## ' docs/analyses/2026-10-14-magcoupler-help-review.md | tail -2
git ls-files --eol docs/analyses/2026-10-14-magcoupler-help-review.md
```
Expected: the `awk` prints nothing (no pasted line starts with "## "); the `grep` ends with `2722:## BL-064 shaft joints` and `<line>:## Slice 3 torque card`; `git ls-files --eol` prints `i/lf    w/crlf` (the file had 5050 lines, every one CRLF). If it prints `w/mixed`, make the working copy CRLF again (the index stores LF either way):
```bash
python -c "p='docs/analyses/2026-10-14-magcoupler-help-review.md'; d=open(p,'rb').read().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n'); open(p,'wb').write(d)"
```

- [ ] **Step 5: Docs that quote a changed string**

For every string the review changed, run `git grep -n -F "<old text>" -- magcoupling-rs/README.md docs/ai`. It must print nothing once the README and `docs/ai` lines quote the new text. T1 to T6 wrote those lines (T5's README lines on the card and the lines under Magnets per ring, spec 9.2); this step updates only their words. If nothing changed, there is nothing to do here. T9 quotes the final texts in `docs/ai/05-update-tracker.md`.

- [ ] **Step 6: Suites, clippy both ways, fmt**

```bash
cargo test --manifest-path magcoupling-rs/Cargo.toml
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path magcoupling-rs/Cargo.toml
cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check
```
Expected:
- every `test result:` line reads `ok`; the counts are T6's (app library `971 passed; 0 failed; 6 ignored`, library without features `234 passed; 0 failed; 1 ignored`), plus one in the app library per assertion test a must-fix added (Step 3), none if the review changed no string or changed constants only;
- both clippy runs are clean;
- `--check` prints nothing.

- [ ] **Step 7: Commit (one commit per round that changed strings, then the record)**

Run `git status --short`. Expected: only the files this task touched, and `?? docs/analyses/2026-05-28-press-4bar-analysis.md` (never staged).

**A round that changed strings** (Step 3, before the next round's dispatch): stage each changed file by name (the sources, the tests, the doc lines of Step 5). Write the message with the Write tool to `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T7-fix-<round>.txt`:

```text
fix(magcoupler): slice 3 G-3 review round <round>: <the strings changed> (BL-073 T7)

<one paragraph: each string old -> new in a few words, and the probe that made the
old text false or misleading, from the reviewer's answer; each test added or changed>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

```bash
git add <each changed source, test and doc file, by name>
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T7-fix-<round>.txt
```

**The record**, after the last round, with the message file `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T7.txt`:

```text
docs(magcoupler): slice 3 G-3 review of the torque card's texts, PASS at round <n> (BL-073 T7)

One fresh-context session-model review of every new and changed on-screen text of the
torque card: the heading, the caption, the three Auto lines, the three one-line forms,
the side words, the calibration note, the flip and margin phrases, the row templates
and RING_SIZE_STAYS's new clause, rendered at every design of the spec's section 2 in
its own Auto state (the short cavity, the temperature edge, variation 1.0, zero torque,
no back iron with f_cal_original -0.95 and the Minimum OD at 45 mm among them), with
its rulings on "cold-high", SHOWN_NO_HOT_LOW, the one change at two precisions and
"everything else held". <n> items for the owner at the STOP. Recorded in
docs/analyses/2026-10-14-magcoupler-help-review.md, "## Slice 3 torque card".

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

```bash
git add docs/analyses/2026-10-14-magcoupler-help-review.md
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T7.txt
git status --short
```
Expected: one commit of 1 file; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 7 moves:** none at plan time. A ruling changes these, each in Step 3 and in its own commit, and the record names it:

| Test | Old -> new | Why |
|---|---|---|
| `torque_card::tests::the_card_s_lines_read_as_drafted` (test 2a) | its `format!` strings, or 4 magnets' rows added | a ruling on a template, or D3 (A) or (B) |
| `torque_card::tests::the_flips_and_the_margins_name_the_checks_that_move` (test 7) | its gearbox phrase without "cold-high" | D1 ruled "drop the name" |
| any test that types an old text (Step 3's grep) | the literal -> the constant | a reworded constant |

---

### Task 8: gui-smoke: the torque card step and the browser drag frame-time check

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec 7.2 (the browser frame-time check), 8.6 test 26 and 6.3's gui-smoke items. The calculator's smoke step gains two steps after its existing checks, on the share link opened again with a timer on every frame the app draws (plan review 4.1: `FRAME_TIMER` keeps a callback only if it made a WebGL draw call): the card step (the card's heading, caption, Auto line and four rows, quoted whole as `torque_card.rs` holds them after T7's review) and the frame-time step (one drag of the Magnet face gap slider, read frame by frame). The console collection and the browser close that ended the existing checks move after both. The script itself judges the drag from its numbers and fails the run on it only with `frame_check: true`, so a machine's speed never fails an ordinary run or the post-deploy run. `app.rs`'s `the_smoke_test_share_link_opens_its_design` asserts that the prompt quotes the card's four texts whole and names the three other rows by their starts (test 26), and that it carries `FRAME_TIMER` (plan review 4.1), so the prompt cannot go stale (the R14 rule). T10 runs the workflow: with `{"frame_check": true}` on the local server (the release run, spec 10.1 step 5), and with `{"url": "https://colesorkness.com"}` alone after the deploy.

Line numbers are 69fa505's (`gui-smoke.js` and `app.rs` are unchanged since; the README's have moved with Tasks 1 to 7); trust the code. Commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation`.

**Files:**
- Modify: `.claude/workflows/gui-smoke.js` (LF line ends: `.gitattributes` has `.claude/workflows/*.js text eol=lf`), `magcoupling-rs/src/app.rs` (the test module only).
- Modify (docs): `magcoupling-rs/README.md`: the gui-smoke paragraph (around :618-631) and the `src/app.rs` row of the Tests table (around :697). Spec 9.2 does not list them; this task makes them stale, so it updates them (T9's sweep finds them done).
- Gitignored helpers it writes: `magcoupling-rs/target/gui-smoke-stub.cjs` (runs the script's own code with a stub agent) and `magcoupling-rs/target/gui-smoke.check.js` (the syntax check's wrapped copy).

**Interfaces:**
- Consumes: T4's `gui::torque_card::{CARD_HEADING, CAPTION, AUTO_HOLDS_LENGTH, Control, held_line}` (`Control::label`) with T7's words; `crate::gui::SIZING_LOG_PREFIX` ("magcoupling sizing: ", already asserted through the solved line).
- Produces:
  - in `gui-smoke.js`: the calculator schema's required booleans `torque_card`, `drag_moved` and `drag_frames_within_60fps`, and the optional numbers `drag_frames` (integer), `drag_frame_ms_median`, `drag_frame_ms_p95` and `drag_frame_ms_max` (ms; absent when the drag did not run); `FRAME_BUDGET_MS = 16.7` and `MIN_DRAG_FRAMES = 30`; `FRAME_TIMER`, the drag's frame timer, a page script that keeps a callback's time only if the callback made a WebGL draw call (plan review 4.1); the argument `frame_check` (`ARGS.frame_check === true`; default false). `drag_frames_within_60fps` is the script's verdict: `drag_moved === true`, an integer `drag_frames >= MIN_DRAG_FRAMES` and a finite `drag_frame_ms_p95 <= FRAME_BUDGET_MS`; with `frame_check` it also gates the calculator's `passed`.
  - test 26 in `app::tests::the_smoke_test_share_link_opens_its_design`.

- [ ] **Step 1: Write the failing test**

`magcoupling-rs/src/app.rs`, the test `the_smoke_test_share_link_opens_its_design`: after the R14 assertions, the card's (spec 8.6 test 26).

Old:
```rust
        let fit =
            crate::gui::guidance::guide(crate::gui::guidance::FIT_CHECK).expect("the fit guide");
        assert!(script.contains(fit.label), "the dashboard's last row");
        // The app opens the link and the solve succeeds after its debounce.
```

New:
```rust
        let fit =
            crate::gui::guidance::guide(crate::gui::guidance::FIT_CHECK).expect("the fit guide");
        assert!(script.contains(fit.label), "the dashboard's last row");
        // Slice 3 (spec 8.6, test 26): the card step quotes the torque card whole as the page
        // draws it at the smoke design (Auto on the magnet length): the heading, the caption, the
        // Auto line and the held row; it names the other three rows by their starts. The G-3
        // review (plan T7) set the words before the prompt quoted them; this keeps it current.
        {
            use crate::gui::torque_card::{
                AUTO_HOLDS_LENGTH, CAPTION, CARD_HEADING, Control, held_line,
            };
            for (what, text) in [
                ("the card's heading", CARD_HEADING.to_owned()),
                ("the card's caption", CAPTION.to_owned()),
                ("the card's Auto line", AUTO_HOLDS_LENGTH.to_owned()),
                ("the held row", held_line(Control::OverallLength)),
            ] {
                assert!(
                    script.contains(&text),
                    "the smoke prompt quotes {what} whole: {text}"
                );
            }
            for control in [
                Control::OutsideDiameter,
                Control::FaceGap,
                Control::MagnetsPerRing,
            ] {
                let start = format!("{}:", control.label());
                assert!(
                    script.contains(&start),
                    "the smoke prompt names the row starting {start}"
                );
            }
        }
        // The drag's timer keeps only the frames the app draws (performance review 4.1): eframe
        // asks for an animation frame at every display refresh, so the timer marks a frame by its
        // WebGL draw call.
        assert!(
            script.contains("const FRAME_TIMER = `") && script.contains("drawElements"),
            "the smoke prompt's frame timer keeps only the frames the app draws"
        );
        // The app opens the link and the solve succeeds after its debounce.
```

Each message prints the text the prompt must quote, as `torque_card.rs` holds it: Step 4 reads them from it.

- [ ] **Step 2: Run the test and see it fail**

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link_opens_its_design`

Expected (the heading as T7 left it; the draft shown):
```text
thread 'app::tests::the_smoke_test_share_link_opens_its_design' panicked at src\app.rs:284:17:
the smoke prompt quotes the card's heading whole: One step of each size control
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 976 filtered out
```
(The heading's assertion fails first. The frame timer's assertion is red first too: with Step 1's block and Step 5's
edits made without `FRAME_TIMER`, the test fails `panicked at src\app.rs:304:9:` with `the smoke prompt's frame timer
keeps only the frames the app draws`.)

- [ ] **Step 3: The script's frame rule, red first: a stub-agent run**

The workflow script has no test runner, so its new rule is checked by running the script's own code under node with a stub in place of the agents. Write `magcoupling-rs/target/gui-smoke-stub.cjs` (gitignored by `magcoupling-rs/.gitignore`'s `/target`) with the Write tool:

```js
// Runs .claude/workflows/gui-smoke.js's own code under node with a stub agent (no browser), to
// check the rule the script applies to the calculator's drag after its agent returns (slice-3
// spec 7.2 steps 3 and 4). Run from the repository root: node magcoupling-rs/target/gui-smoke-stub.cjs
const fs = require('fs')
const body = fs
  .readFileSync('.claude/workflows/gui-smoke.js', 'utf8')
  .replace(/^export const meta/m, 'const meta')
const AsyncFunction = (async () => {}).constructor
const workflow = new AsyncFunction('args', 'agent', body)

// A calculator agent's reading of a good run; each case changes one thing.
const reading = {
  passed: true,
  torque_card: true,
  drag_moved: true,
  drag_frames: 60,
  drag_frame_ms_median: 4.1,
  drag_frame_ms_p95: 9.8,
  drag_frame_ms_max: 12.0,
  drag_frames_within_60fps: true,
}
const cases = [
  ['a fast drag, frame_check', { frame_check: true }, reading],
  ['a slow drag, frame_check', { frame_check: true }, { ...reading, drag_frame_ms_p95: 21.3 }],
  ['a slow drag, an ordinary run', {}, { ...reading, drag_frame_ms_p95: 21.3 }],
  ['a slow drag, frame_check false', { frame_check: false }, { ...reading, drag_frame_ms_p95: 21.3 }],
  ['12 frames, frame_check', { frame_check: true }, { ...reading, drag_frames: 12 }],
  ['no p95, frame_check', { frame_check: true }, { ...reading, drag_frame_ms_p95: null }],
  ['the handle not held, frame_check', { frame_check: true }, { ...reading, drag_moved: false }],
  ['the agent returned nothing, frame_check', { frame_check: true }, null],
]

// The drag's frame timer (FRAME_TIMER) in a stand-in window: two display refreshes, the first
// draws nothing (eframe's idle callback), the second makes a WebGL draw call. The timer must keep
// the second alone (performance review 4.1); T8's first timer, the control, keeps both.
const FIRST_TIMER = "() => { window.__frameMs = []; const raf = window.requestAnimationFrame.bind(window); window.requestAnimationFrame = cb => raf(t => { const s = performance.now(); cb(t); window.__frameMs.push(performance.now() - s); }); }"
const keptOf = source => {
  const queue = []
  const window = { requestAnimationFrame: cb => queue.push(cb), WebGL2RenderingContext: function () {} }
  window.WebGL2RenderingContext.prototype.drawElements = function () {}
  new Function('window', 'performance', `(${source})()`)(window, { now: () => 0 })
  const gl = new window.WebGL2RenderingContext()
  window.requestAnimationFrame(() => {})
  window.requestAnimationFrame(() => gl.drawElements())
  while (queue.length) queue.shift()(0)
  return window.__frameMs.length
}
const frameTimer = /^const FRAME_TIMER = `([^`]*)`$/m.exec(body)
console.log(`the first timer keeps ${keptOf(FIRST_TIMER)} of 2 refreshes`)
console.log(frameTimer ? `FRAME_TIMER keeps ${keptOf(frameTimer[1])} of 2 refreshes` : 'no FRAME_TIMER in the script')

;(async () => {
  for (const [name, args, result] of cases) {
    const stub = async () => (result ? { ...result } : null)
    const out = await workflow(JSON.stringify({ ...args, linkage: false }), stub)
    const m = out.magcoupling
    console.log(`${name}: passed=${out.passed} within=${m.drag_frames_within_60fps}`)
  }
})()
```

Run: `node magcoupling-rs/target/gui-smoke-stub.cjs`

Expected (red: today's script takes the agent's word, so a slow drag passes the release run; and it has no `FRAME_TIMER`):
```text
the first timer keeps 2 of 2 refreshes
no FRAME_TIMER in the script
a fast drag, frame_check: passed=true within=true
a slow drag, frame_check: passed=true within=true
a slow drag, an ordinary run: passed=true within=true
a slow drag, frame_check false: passed=true within=true
12 frames, frame_check: passed=true within=true
no p95, frame_check: passed=true within=true
the handle not held, frame_check: passed=true within=true
the agent returned nothing, frame_check: passed=false within=undefined
```

- [ ] **Step 4: Read the card's words as T7 set them**

```bash
git grep -n -A1 -E 'const (CARD_HEADING|CAPTION|AUTO_HOLDS_LENGTH): &str' -- magcoupling-rs/src/gui/torque_card.rs
git grep -n -A2 -e 'pub fn held_line' -- magcoupling-rs/src/gui/torque_card.rs
git grep -n -e 'pub const SET_BY_AUTO' -- magcoupling-rs/src/gui/size_controls.rs
```
The edits of Step 5 quote spec 4.1's drafts: "One step of each size control" (twice: the first view and the card step), "The change in the model's hot-low torque after one step of one control, everything else held.", "Auto sets the magnet length again after a change; each row keeps the length shown." and "Overall length: Set by Auto; turn Auto off to edit". Where T7's record changed one, write the set text in its place in edits F and G: copy it from the grep, or from Step 2's failure message, whole. The three row starts are the control labels ("Outside diameter:", "Magnet face gap:", "Magnets per ring:"), which T7 does not change.

- [ ] **Step 5: The script's edits**

Nine edits to `.claude/workflows/gui-smoke.js`, in this order (each Old is unique in the file when it is applied; together they take `wc -l` from 107 to 147).

(A) The meta description (`:3`) names the two new checks. Old:
```js
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator a Why link opens a note, an input tooltip shows no spreadsheet details, and the dashboard scrolls at 1280x800',
```
New:
```js
  description: 'Smoke-test the locally served web build (the hub page at /tools/, the linkage app at /tools/linkage/, the calculator at /tools/magcoupler/): loads, canvas present, console clean; on the calculator a Why link opens a note, an input tooltip shows no spreadsheet details, the dashboard scrolls at 1280x800, the torque card shows its rows between the readouts and the drawing, and a drag of the gap slider is timed frame by frame (frame_check: true fails a p95 over one 60 fps frame)',
```

(B) `MAGCOUPLING_SCHEMA` (`:21-39`): three required booleans and the drag's numbers (spec 8.6 test 26, 7.2 step 3). Old:
```js
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'why_link_opened', 'tooltip_plain', 'dashboard_scrolls'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    first_view: { type: 'boolean' },
    explorer_ready: { type: 'boolean' },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
    details_toggle: { type: 'boolean' },
    why_link_opened: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
    dashboard_scrolls: { type: 'boolean' },
    screenshot_note: { type: 'string' },
```
New:
```js
  type: 'object', required: ['passed', 'canvas_present', 'console_errors', 'share_link_loaded', 'sizing_solved', 'design_file_loaded', 'first_view', 'explorer_ready', 'details_toggle', 'why_link_opened', 'tooltip_plain', 'dashboard_scrolls', 'torque_card', 'drag_moved', 'drag_frames_within_60fps'],
  properties: {
    passed: { type: 'boolean' },
    canvas_present: { type: 'boolean' },
    console_errors: { type: 'array', items: { type: 'string' } },
    first_view: { type: 'boolean' },
    explorer_ready: { type: 'boolean' },
    share_link_loaded: { type: 'boolean' },
    sizing_solved: { type: 'boolean' },
    design_file_loaded: { type: 'boolean' },
    details_toggle: { type: 'boolean' },
    why_link_opened: { type: 'boolean' },
    tooltip_plain: { type: 'boolean' },
    dashboard_scrolls: { type: 'boolean' },
    torque_card: { type: 'boolean' },
    drag_moved: { type: 'boolean' },
    drag_frames: { type: 'integer' },
    drag_frame_ms_median: { type: 'number' },
    drag_frame_ms_p95: { type: 'number' },
    drag_frame_ms_max: { type: 'number' },
    drag_frames_within_60fps: { type: 'boolean' },
    screenshot_note: { type: 'string' },
```

(C) The comment over `MAGCOUPLING_TOOLS` (`:56-57`; the tool list itself already has `browser_run_code_unsafe`). Old:
```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Load design
// button, a Why link, the Geometry tab), uploads a file and moves the mouse (a tooltip, a scroll).
```
New:
```js
// The calculator's step also sets the window to 1280 x 800, clicks the canvas (the Load design
// button, a Why link, the Geometry tab), uploads a file, moves the mouse (a tooltip, a scroll)
// and drags the Magnet face gap slider while a timer reads each frame it draws (spec 7.2).
```

(D) The frame check's two numbers, the drag's frame timer `FRAME_TIMER` (plan review 4.1) and the argument `frame_check` (`:60-63`). Old:
```js
const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const url = ARGS.url || 'http://localhost:8080'
```
New:
```js
// The calculator's drag frame-time check (slice-3 spec 7.2): one 60 fps frame, and the fewest
// drawn drag frames that show the timer caught eframe's drawing (fewer: it missed it).
const FRAME_BUDGET_MS = 16.7
const MIN_DRAG_FRAMES = 30

// The drag's frame timer (slice-3 spec 7.2), installed in the page before the app starts.
// eframe 0.32's web runner asks for an animation frame at every display refresh and draws only
// when something changed (eframe src/web/events.rs, paint_and_schedule): in a drag at 165 Hz,
// 458 of 640 callbacks drew nothing (performance review 4.1). The timer keeps a callback only
// if it made a WebGL draw call (eframe's glow painter draws every frame with drawElements), so
// drag_frames counts drawn frames and the median and the p95 are theirs.
const FRAME_TIMER = `() => { window.__frameMs = []; window.__drew = false; for (const gl of [window.WebGL2RenderingContext, window.WebGLRenderingContext]) { if (!gl) continue; for (const name of ['drawElements', 'drawArrays']) { const draw = gl.prototype[name]; if (typeof draw !== 'function') continue; gl.prototype[name] = function (...args) { window.__drew = true; return draw.apply(this, args); }; } } const raf = window.requestAnimationFrame.bind(window); window.requestAnimationFrame = cb => raf(t => { window.__drew = false; const s = performance.now(); cb(t); const ms = performance.now() - s; if (window.__drew) window.__frameMs.push(ms); }); }`

const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
if (ARGS.selftest) { return { ok: true } }

const url = ARGS.url || 'http://localhost:8080'
// Only the release run sets frame_check: true (slice-3 spec 7.2 step 4); a slow drag frame then
// fails the run. Otherwise the drag's times are recorded and never fail an ordinary or live run.
const frameCheck = ARGS.frame_check === true
```

(E) The calculator step's comment (`:77-84`) gains the card step and the frame timing. Old:
```js
// a teaching note, an input row's tooltip shows no spreadsheet details while the toggle is
// off, and the dashboard scrolls to its last row at 1280 x 800. src/app.rs checks that the
// prompt names the strip, the toggle, the note's parts, the tab and the row as the page does.
```
New:
```js
// a teaching note, an input row's tooltip shows no spreadsheet details while the toggle is
// off, and the dashboard scrolls to its last row at 1280 x 800. Then (slice 3, spec 7.2 and
// 8.6) the link is opened again with a timer on every frame the app draws: on that fresh page the
// torque card shows its heading, caption, Auto line and four rows between the readouts and the
// drawing, and one drag of the gap slider is timed frame by frame. The script judges the drag
// from its numbers (FRAME_BUDGET_MS, MIN_DRAG_FRAMES) and fails the run on it only with
// frame_check: true. src/app.rs checks that the prompt names the strip, the toggle, the note's
// parts, the tab, the row and the card's words as the page does.
```

(F) The first view (`:91`) names the card between the readouts and the drawing (spec 6.3). Old:
```js
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, dashboard on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the middle region shows the first view with no click: at its top the line "Set the size -> read the torque -> open any red check to see why and what to change." with a "Hide" button, under it a row of tabs starting "Size and torque" (the selected tab), and under the tabs the torque readouts above a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. details_toggle=true only if the header row at the top of the page (it may wrap to a second line) shows, after the "Equation panel" button, a checkbox labelled "Show spreadsheet details", unticked.
```
New:
```js
Wait 8 seconds (WASM init, then the sizing solve after its 0.25 s debounce). Snapshot the page and confirm a <canvas> element with id "magcoupling_canvas" exists. Collect the console messages at level "info" (it includes errors and warnings). Take a screenshot and judge whether it shows the calculator rendered in a dark theme (the size controls on the left, dashboard on the right, the Size and torque view in the middle) vs a blank page. first_view=true only if the middle region shows the first view with no click: at its top the line "Set the size -> read the torque -> open any red check to see why and what to change." with a "Hide" button, under it a row of tabs starting "Size and torque" (the selected tab), and under the tabs the torque readouts, then the heading "One step of each size control" with a link starting "Why:" beside it, open, with lines of text under it, then a drawing of two rings of coloured magnet blocks inside a grey cup around a shaft. details_toggle=true only if the header row at the top of the page (it may wrap to a second line) shows, after the "Equation panel" button, a checkbox labelled "Show spreadsheet details", unticked.
```

(G) The console collection and the browser close (`:98`) move after two new steps: the card step on the share link opened again with the frame timer (spec 7.2 step 1, 8.6 test 26), then the drag (7.2 steps 2 and 3). Old:
```js
If a target of these last three checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes. Collect the console messages at level "info" once more, then close the browser.
```
New:
```js
If a target of these last three checks cannot be found after two attempts (each with a fresh screenshot), set that field false and say why in notes.
Then the torque card and the drag's frame times, on the page opened again. Not on the page above: the design file loaded there has no sizing, so no row reads "Set by Auto" and no Auto line shows; the dashboard's Why link opened the Equation panel, which closes the card; and the tooltip step left the Geometry tab selected, where no card is drawn. Put a timer on every frame the app draws and open the share link again with browser_run_code_unsafe, code: async (page) => { await page.addInitScript(${FRAME_TIMER}); await page.goto('${page}'); }
Wait 8 seconds (WASM init, then the sizing solve) and take a screenshot. torque_card=true only if the middle region shows, between the torque readouts and the drawing, the heading "One step of each size control" with a link starting "Why:" beside it, open, and under it, in this order: the line "The change in the model's hot-low torque after one step of one control, everything else held.", the line "Auto sets the magnet length again after a change; each row keeps the length shown.", then four rows: one starting "Outside diameter:", the row "Overall length: Set by Auto; turn Auto off to edit", one starting "Magnet face gap:" and one starting "Magnets per ring:"; and the drawing still shows below the card. A row may wrap onto more lines.
Then the drag: in a fresh screenshot find the handle of the "Magnet face gap" slider among the size controls on the left (it reads 1.50 mm; Auto on the magnet length leaves this slider free). Drag it and read the frame times in one call of browser_run_code_unsafe, code: async (page) => { const x = X, y = Y; let sizing = 0; const onConsole = m => { if (m.text().includes('magcoupling sizing: ')) sizing += 1; }; page.on('console', onConsole); await page.mouse.move(x, y); await page.evaluate(() => { window.__frameMs = []; }); await page.mouse.down(); for (let i = 0; i < 60; i++) { await page.mouse.move(i % 2 === 0 ? x + 3 : x - 3, y); await page.waitForTimeout(50); } await page.mouse.up(); const ms = await page.evaluate(() => window.__frameMs.slice()); page.off('console', onConsole); const s = [...ms].sort((a, b) => a - b); const rank = q => (s.length ? s[Math.ceil(q * s.length) - 1] : null); return { frames: s.length, median: rank(0.5), p95: rank(0.95), max: s.length ? s[s.length - 1] : null, sizing_lines: sizing }; } (X and Y the handle's centre in CSS pixels of the screenshot; the timer is emptied just before the press, so only the drag's frames count, not the load, the solve or the screenshots). Then take a screenshot. drag_moved=true only if the Magnet face gap now reads a value other than 1.50 mm (the drag held the handle). From the returned object: drag_frames=frames, drag_frame_ms_median=median, drag_frame_ms_p95=p95 and drag_frame_ms_max=max (milliseconds per frame, nearest rank). drag_frames_within_60fps=true only if drag_moved, drag_frames >= ${MIN_DRAG_FRAMES} and drag_frame_ms_p95 <= ${FRAME_BUDGET_MS} (this script checks it again from your numbers); under ${MIN_DRAG_FRAMES} frames the timer saw the app draw too few frames: say so in notes. In notes give sizing_lines, the number of "magcoupling sizing: " console lines logged during the drag (expected 0: the moves come 50 ms apart, so the 0.25 s debounce never runs a solve mid-drag). If the handle cannot be found after two attempts (each with a fresh screenshot), set drag_moved=false, leave the four numbers out and say why in notes.
Collect the console messages at level "info" once more (it covers the reopened page and the drag), then close the browser.
```

(H) The agent's `passed` (`:100`) requires `torque_card` and leaves the drag to the script. Old:
```js
passed=true only if: page loaded, canvas present, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, why_link_opened, tooltip_plain, dashboard_scrolls, and zero console messages of type error (warnings are OK — put them in notes). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
```
New:
```js
passed=true only if: page loaded, canvas present, first_view, details_toggle, share_link_loaded, sizing_solved, design_file_loaded, explorer_ready, why_link_opened, tooltip_plain, dashboard_scrolls, torque_card, and zero console messages of type error (warnings are OK — put them in notes); leave the drag's fields out of passed (this script judges them). List every console error string verbatim in console_errors. If navigation fails entirely, passed=false with the failure in notes — the server may not be running (serve linkage-sim-rs/web/ after running scripts/build_magcoupling_web.sh).`,
```

(I) The fallback (`:102`) gains the fields, and the script judges the drag (spec 7.2 steps 3 and 4). Old:
```js
  ) || { passed: false, canvas_present: false, console_errors: [], first_view: false, explorer_ready: false, share_link_loaded: false, sizing_solved: false, design_file_loaded: false, details_toggle: false, why_link_opened: false, tooltip_plain: false, dashboard_scrolls: false, notes: 'smoke agent returned no result (agent error)' }
```
New:
```js
  ) || { passed: false, canvas_present: false, console_errors: [], first_view: false, explorer_ready: false, share_link_loaded: false, sizing_solved: false, design_file_loaded: false, details_toggle: false, why_link_opened: false, tooltip_plain: false, dashboard_scrolls: false, torque_card: false, drag_moved: false, drag_frames: 0, drag_frames_within_60fps: false, notes: 'smoke agent returned no result (agent error)' }
  // The drag's verdict (slice-3 spec 7.2 steps 3 and 4), from its numbers by this script, not the
  // agent's arithmetic: recorded on every run, and failing the run only with frame_check: true.
  const p95 = magcoupling.drag_frame_ms_p95
  magcoupling.drag_frames_within_60fps = magcoupling.drag_moved === true
    && Number.isInteger(magcoupling.drag_frames) && magcoupling.drag_frames >= MIN_DRAG_FRAMES
    && Number.isFinite(p95) && p95 <= FRAME_BUDGET_MS
  if (frameCheck) magcoupling.passed = magcoupling.passed && magcoupling.drag_frames_within_60fps
```

- [ ] **Step 6: Check the script's syntax**

`node --check .claude/workflows/gui-smoke.js` proves nothing for this file: the `export` makes Node 22 treat it as an ES module, and it then exits 0 even with a planted syntax error (`const broken = {` appended: exit 0). The workflow runs the file as the body of an async function, so check it that way:

```bash
{ echo 'async function workflow(args, agent) {'; sed 's/^export const meta/const meta/' .claude/workflows/gui-smoke.js; echo '}'; } > magcoupling-rs/target/gui-smoke.check.js
node --check magcoupling-rs/target/gui-smoke.check.js; echo "exit $?"
git ls-files --eol .claude/workflows/gui-smoke.js
```
Expected: `exit 0` (the same wrapped check of a copy with `const broken = {` appended, or with a stray `]` in the schema, exits 1); the `git ls-files --eol` line reads `i/lf`, `w/lf` and `attr/text eol=lf` for the file (the edits kept its LF line ends).

- [ ] **Step 7: Run the stub and the test again and see them pass**

Run: `node magcoupling-rs/target/gui-smoke-stub.cjs`

Expected:
```text
the first timer keeps 2 of 2 refreshes
FRAME_TIMER keeps 1 of 2 refreshes
a fast drag, frame_check: passed=true within=true
a slow drag, frame_check: passed=false within=false
a slow drag, an ordinary run: passed=true within=false
a slow drag, frame_check false: passed=true within=false
12 frames, frame_check: passed=false within=false
no p95, frame_check: passed=false within=false
the handle not held, frame_check: passed=false within=false
the agent returned nothing, frame_check: passed=false within=false
```

A planted fault: `FRAME_TIMER` with ` if (window.__drew)` removed before `window.__frameMs.push(ms);` prints `FRAME_TIMER keeps 2 of 2 refreshes`; undo it.

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib the_smoke_test_share_link_opens_its_design`

Expected: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 976 filtered out`. Planted in the plan-time prototype, each of these failed it with its text, then passed once undone: the caption with "one step of each control" in place of "one step of one control" (`the smoke prompt quotes the card's caption whole: The change in the model's hot-low torque after one step of one control, everything else held.`), and the row start written "Magnets per ring" without its colon (`the smoke prompt names the row starting Magnets per ring:`).

- [ ] **Step 8: Docs: `magcoupling-rs/README.md`**

Four edits; each Old is a substring of one line (the working copy is CRLF), unique in the file.

(a) The gui-smoke paragraph's first view (around :620). Old:
```text
geometry drawing in the first screenshot (the first view: no click)
```
New:
```text
torque card (open, between the readouts and the drawing) and geometry drawing in the first screenshot (the first view: no click)
```

(b) The same paragraph's last check (around :628). Old:
```text
in a 1280 x 800 window; all with zero console
```
New:
```text
in a 1280 x 800 window; then (slice 3) it opens the link again with a timer on every frame the app draws (eframe asks for an animation frame at every display refresh; the timer keeps those that make a WebGL draw call), checks the torque card's caption, Auto line and four rows (the Overall length row held by Auto among them), and drags the Magnet face gap slider 3 px each way 60 times, 50 ms apart: the drag's drawn-frame count, median, p95 and slowest frame are recorded on every run, and with `frame_check: true` (the release run) at least 30 frames and a p95 within one 60 fps frame (16.7 ms) are required; all with zero console
```

(c) The same paragraph's test clause (around :630). Old:
```text
toggle, the note's parts, the Geometry tab and the last row as the page draws them).
```
New:
```text
toggle, the note's parts, the Geometry tab, the last row and the card's heading, caption, Auto line, held row and row starts as the page draws them).
```

(d) The Tests table's `src/app.rs` row (around :697). Old:
```text
the Geometry tab and the Magnets fit row as the page draws them;
```
New:
```text
the Geometry tab and the Magnets fit row as the page draws them, and its card step quotes `torque_card::CARD_HEADING`, `CAPTION`, `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` whole and names the other three rows by their starts (slice 3, test 26);
```

Run: `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain` (it reads the README). Expected: `test result: ok.` at T7's count (64 passed, 1 ignored).

- [ ] **Step 9: Suites, clippy both ways, fmt**

```bash
cargo test --manifest-path magcoupling-rs/Cargo.toml
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path magcoupling-rs/Cargo.toml
cargo fmt --manifest-path magcoupling-rs/Cargo.toml -- --check
git status --short
```
Expected:
- every `test result:` line reads `ok`, at T7's counts: this task extends one test and adds none (app library 971 passed, 6 ignored; library without features 234 passed, 1 ignored; `explain` 64 passed, 1 ignored);
- both clippy runs are clean; `--check` prints nothing (Step 1's block is rustfmt's layout);
- `git status --short`:
  ```text
   M .claude/workflows/gui-smoke.js
   M magcoupling-rs/README.md
   M magcoupling-rs/src/app.rs
  ?? docs/analyses/2026-05-28-press-4bar-analysis.md
  ```

- [ ] **Step 10: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T8.txt` with the Write tool:

```text
test(magcoupler): gui-smoke checks the torque card and times a drag frame by frame (BL-073 T8)

The calculator's smoke step opens the share link again with a timer on every frame
the app draws (spec 7.2). On that fresh page it checks the torque card between the readouts and
the drawing: the heading, the caption, the Auto line, the held Overall length row and
the other three rows' starts, quoted as the G-3 review set them. Then it drags the
Magnet face gap slider 3 px each way 60 times, 50 ms apart, and reads the drag's frames.
The script judges them itself (at least 30 frames, a p95 within one 60 fps frame,
16.7 ms) and fails the run on them only with frame_check: true, the release run; an
ordinary or a live run records them. The console collection and the browser close move
after both steps, and the first view names the card.

app.rs's the_smoke_test_share_link_opens_its_design asserts that the prompt quotes the
card's heading, caption, Auto line and held row whole and names the other three rows by
their starts (test 26). README: the gui-smoke paragraph and the app.rs test row.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```

Run:
```bash
git add .claude/workflows/gui-smoke.js magcoupling-rs/src/app.rs magcoupling-rs/README.md
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T8.txt
git status --short
```

Expected: one commit of 3 files; `git status --short` then shows only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

**Existing tests Task 8 moves** (confirmed by running in the prototype):

| Test | Old -> new | Why |
|---|---|---|
| `app::tests::the_smoke_test_share_link_opens_its_design` (`src/app.rs:211-284`) | asserts the link, the design file, the log lines and the R14 names -> also the card's four texts whole and its three row starts, and the drag's frame timer `FRAME_TIMER` | spec 8.6 test 26 (6.3's gui-smoke item); plan review 4.1 |

---

### Task 9: Docs and backlog: BL-073 to BL-075 and BL-065's line, the README and docs/ai, and the screen dump that reads what ships

**Implementer:** session model (BL-073 is risk: physics; CLAUDE.md section 5).

Spec section 9. This task writes the docs no code task makes stale (the backlog items, the status and NEXT lines,
`04-memory`, the tracker), writes the README, `02-system` and `03-structure` lines spec 9.2 names unless an earlier
task already wrote them (each edit below says how to tell), and sweeps every stale text with exact grep commands. It
also builds the slice's copy of the BL-064 screen-dump tool, with the card and gap records of spec 8.6 test 27, and
the slice's copy of `check_shipped.py`: every on-screen text and number this task's docs quote is read from what the
branch ships, never from a draft (BL-059 ruling B-1). Task 10 reuses both for the STOP sheet (plan deviation PD-50: the screen
dump is built here because the T9 docs need it, and T10 re-runs it and adds main's side).

All commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation` in Git Bash, on branch
`magcoupling/slice3-torque-card`. Edits are made with the Edit tool (exact old text -> exact new text; the docs files
are CRLF, which the Edit tool matches). `YYYY-MM-DD` means the date `date +%F` prints on the day the step runs. `W`
below is `.superpowers/sdd/2026-10-19-magcoupler-slice3` (gitignored).

Rules for this task:
- Docs only, plus the gitignored tool and check (the tool builds into the gitignored `magcoupling-rs/target/`).
- `docs/ai/05-update-tracker.md` history is never rewritten: this task adds one new top entry.
- Each edit of Steps 6-9 names the task whose code makes the line stale and a "skip if" string: when the file
  already holds that string, an earlier task wrote the fact; skip the edit. When the file holds neither the old
  text nor the skip-if string, an earlier task rewrote the line another way: read it; if it states what the new text
  states, skip the edit and name it in the commit body; otherwise STOP and report the line to the controller.
- A line Step 4's table gives to T1, T5 or T6 (their docs lines and code comments, which are not edited here) that is
  still stale means that task's docs step was not applied: apply that task's step from this plan verbatim (no new
  wording here), and name it in the commit body.
- Every on-screen text or number this task quotes is checked against what the branch ships (Step 3). On a wording
  change the shipped text wins; a number that differs is a STOP.

**Files:**
- Create (gitignored): `$W/screen-dump/src/main.rs` (BL-064's, copied and extended), `$W/screen-dump/after/Cargo.toml`
  and the `Cargo.lock` beside it, `$W/check_shipped.py` (BL-064's, copied and extended), `$W/same_designs.py` (plan review 2.2 and 1.2: the
  dump's designs against the review dump's), `$W/screen-T9.tsv`, `$W/g3-T9.txt` (the review dump it reads),
  `$W/msg/T9.txt`; the tool's build goes to `magcoupling-rs/target/screen-dump/` (gitignored).
- Modify: `docs/ai/backlog.yaml` (BL-073, BL-074 and BL-075 before BL-055; one line in BL-065's notes),
  `magcoupling-rs/README.md` (the status paragraph ~:46; the `src/gui/` module list ~:155; the Size-tab lines under
  the controls and a new torque-card paragraph ~:214-228; the Why links ~:362; the test table's rows for
  `src/engine/explain/plain.rs` ~:684, `src/gui/panel.rs` ~:686, `src/gui/geometry_view.rs` ~:687 and
  `src/gui/guidance.rs` ~:693, and a new `src/gui/torque_card.rs` row), `docs/ai/02-system.yaml` (`key_files` ~:82, the
  reachability invariant ~:115, the Size-tab invariant ~:116, one `known_limitations` entry, `current_statuses`
  ~:488), `docs/ai/03-structure.yaml` (the `explain:` entry ~:64, the `gui:` entry ~:68, `manual_size_gui:` ~:73),
  `docs/ai/04-memory.yaml` (a new active issue; NEXT ~:13), `docs/ai/05-update-tracker.md` (a new top entry).
- Test: no new test. Step 11 runs `tests/explain.rs`'s docs-count tests (they read the README and `03-structure.yaml`),
  both suites, both clippy runs and `cargo fmt --check`.

**Interfaces:**
- Consumes: T1-T8 committed: `gui::torque_card::{card, CARD_HEADING, Control, Side}` (public; `TorqueCard::lines` and
  `rows`, `Change::hot_low_Nm`), the `gui::torque_gap_text` re-export, `guidance::magnet_count_line` (its spec 4.3
  rule), the G-3 record `## Slice 3 torque card` (T7), gui-smoke's card and frame steps (T8); the public API BL-064's
  tool read (unchanged at 69fa505) and `engine::size_controls::{dimensions, dimensions_with_minimum_diameter,
  resize_diameter_with_minimum, resize_length, ring_radius_range, with_ring_radius}`,
  `engine::sizing::{FreeVariable, SizingOutcome, solve_in}`, `gui::sizing::sized`, `engine::meta::input_rows`.
- Produces (Task 10 edits these exact strings, after the owner's OK unless it says otherwise): the tracker heading
  `## 2026-10-19 — Magcoupler guidance slice 3: the torque card, one step of each size control (BL-073, branch magcoupling/slice3-torque-card)`
  and its last bullet's last sentence `the card's percents read the model's end factor (BL-065).` (T10 Step 8 appends a
  bullet after it); BL-073 `  status: in_progress` (the only one in the file) and its notes' last sentence
  `Deploy authorization is asked at the STOP.` (T10 Step 8 appends after it); the 04-memory active issue that starts
  `  - "Magcoupler torque card (guidance slice 3, BL-073, branch magcoupling/slice3-torque-card; spec` with its phrase
  `built (T1-T9); the release checks (T10: whole-branch review, GATE PASS with gate 12, web build, the native timing and gui-smoke with the browser frame check) and then the owner's review of the diff come before the merge and the push (risk: physics).`;
  the NEXT phrase `then slice 3, the torque card, built on branch magcoupling/slice3-torque-card`; the 02-system phrase
  `built on branch magcoupling/slice3-torque-card`; the screen dump `$W/screen-dump/` (Task 10 adds its `before/`
  side) and `$W/check_shipped.py`.

- [ ] **Step 1: Preconditions**

Run:
```bash
git branch --show-current
git status --short
git log --oneline dae024f..HEAD
grep -c 'pub mod torque_card;' magcoupling-rs/src/gui/mod.rs
grep -c 'torque_gap_text' magcoupling-rs/src/gui/mod.rs
grep -c 'pub const CARD_HEADING' magcoupling-rs/src/gui/torque_card.rs
grep -c '^## Slice 3 torque card' docs/analyses/2026-10-14-magcoupler-help-review.md
grep -c 'frame_check' .claude/workflows/gui-smoke.js
```
Expected: `magcoupling/slice3-torque-card`; `git status` prints only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`;
the log holds the plan's commit (and the commit of any later plan edit) and the commit of every commit step of T1-T8,
each with the subject that step names (T7 may add review-fix commits); then `1`, at least `1` (the re-export the
screen dump reads, `pub use size_controls::torque_gap_text;`), `1`, `1` and at least `1`. Anything else: STOP and finish
the owning task first (a missing re-export: Task 1 Step 6 adds it, plan deviation PD-4).

- [ ] **Step 2: Build the screen dump and read what the branch ships**

1. Copy BL-064's tool:
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
mkdir -p $W/screen-dump/src $W/screen-dump/after
cp .superpowers/sdd/2026-10-17-bl064-shaft-joint/screen-dump/src/main.rs $W/screen-dump/src/main.rs
```
2. Five edits to `$W/screen-dump/src/main.rs` with the Edit tool (the file is LF).

(E1) The tool's doc comment. Old:
```rust
//! BL-064 release tool (not part of the crate): prints what the calculator shows at the
//! example design (the defaults) as one record per line, `kind<TAB>key<TAB>text`, so two
//! checkouts can be compared line by line. Built against one checkout's `magcoupling-rs`
//! (feature `gui`) by `before/Cargo.toml` or `after/Cargo.toml`. Every text comes from the
//! code's own tables and functions; nothing here is typed from a draft.
```
New:
```rust
//! Slice-3 release tool (not part of the crate; BL-064's, copied): prints what the calculator
//! shows at the example design (the defaults) as one record per line, `kind<TAB>key<TAB>text`,
//! so two checkouts can be compared line by line. Built against one checkout's
//! `magcoupling-rs` (feature `gui`) by `before/Cargo.toml` or `after/Cargo.toml`. Every text
//! comes from the code's own tables and functions; nothing here is typed from a draft. Slice 3
//! adds the first line under Magnets per ring in each Auto state (`size tab`) and, behind this
//! tool's own feature `card` (on in `after/Cargo.toml` only), the torque card at every design of
//! the slice-3 spec's section 2.1 (`card`) and the readout's gap at five of them (`gap`).
```

(E2) Old:
```rust
use magcoupling::engine::meta::{InputSet, ResultSet, Value, input_rows, result_rows};
```
New:
```rust
use magcoupling::engine::meta::{InputSet, ResultSet, Value, input_rows, result_rows};
use magcoupling::engine::sizing::FreeVariable;
```

(E3) Old:
```rust
use magcoupling::gui::guidance::{CHECK_GUIDES, counted_guides, design_status, design_status_text};
```
New:
```rust
use magcoupling::gui::guidance::{
    CHECK_GUIDES, counted_guides, design_status, design_status_text, magnet_count_line,
};
```

(E4) The first line under Magnets per ring in each Auto state, on both checkouts (`magnet_count_line` is public on
`main` too). Old:
```rust
    // The share link of the default design.
```
New:
```rust
    // The first line under Magnets per ring in each Auto state (slice 3 rewords it and drops
    // the two Auto lines: its spec's section 4.3).
    for auto in [None].into_iter().chain(FreeVariable::ALL.map(Some)) {
        out(
            "size tab",
            &format!("magnet count line/Auto {auto:?}"),
            magnet_count_line(auto).unwrap_or("(none)"),
        );
    }

    // The share link of the default design.
```

(E5) The end of `main` (its last loop), then the card module, behind the tool's own feature `card` (spec 8.6 test
27: one record per line drawn, `card<TAB><design key> NN<TAB><line>`, `00` the heading; the designs are `card_designs()`'s,
under its keys (critique A-10): every design of spec section 2.1 with its Minimum OD and Auto state, T-14's one-flip
design and R-13's boss plateau for the STOP sheet, and the table's two count overflows (plan review 2.2); and `gap` records of `torque_gap_text` at (a), (b), (c), (e) and (e2)). Old:
```rust
        out("count", key, &n.to_string());
    }
}
```
New:
```rust
        out("count", key, &n.to_string());
    }

    #[cfg(feature = "card")]
    card::print();
}

/// Slice 3 (its spec's section 8.6, test 27): the torque card at every design of the spec's
/// section 2.1 with its Minimum OD and Auto state, one `card` record per line drawn
/// (`<design key> 00` the heading, then `lines(auto)` from `01`), and the readout's gap at (a),
/// (b), (c), (e) and (e2). Public items only: `gui::torque_card` and the `gui::torque_gap_text`
/// re-export. The design keys are the spec's section 2.1 keys, the same as the card tests'
/// `card_designs()` (a test module, which this tool cannot read): `same_designs.py` checks that
/// every design both name draws the same lines here as in the review dump, which reads it.
#[cfg(feature = "card")]
mod card {
    use super::out;
    use magcoupling::engine::clamps::PAUSED;
    use magcoupling::engine::meta::{SliderRange, input_rows};
    use magcoupling::engine::size_controls::{
        dimensions, dimensions_with_minimum_diameter, resize_diameter_with_minimum, resize_length,
        ring_radius_range, with_ring_radius,
    };
    use magcoupling::engine::sizing::{FreeVariable, SizingOutcome, solve_in};
    use magcoupling::gui::sizing::sized;
    use magcoupling::gui::torque_card::{CARD_HEADING, Control, Side, card};
    use magcoupling::gui::torque_gap_text;
    use magcoupling::{DesignInputs, compute_all};

    /// One design of the spec's section 2.1: its key, its inputs (or why they could not be
    /// built), the Size tab's Minimum OD preference and the Auto variable (`None`: Auto off).
    struct Design {
        key: String,
        built: Result<DesignInputs, String>,
        minimum: Option<f64>,
        auto: Option<FreeVariable>,
    }

    /// A design shown with Auto off.
    fn design(key: &str, built: Result<DesignInputs, String>, minimum: Option<f64>) -> Design {
        Design {
            key: key.to_owned(),
            built,
            minimum,
            auto: None,
        }
    }

    /// The design shown with Auto on `variable` from `base` ([`auto_shown`]).
    fn auto(
        key: &str,
        base: &DesignInputs,
        variable: FreeVariable,
        minimum: Option<f64>,
    ) -> Design {
        Design {
            key: key.to_owned(),
            built: auto_shown(base, variable, minimum),
            minimum,
            auto: Some(variable),
        }
    }

    /// `base` with `edit` applied.
    fn with(base: &DesignInputs, edit: impl FnOnce(&mut DesignInputs)) -> DesignInputs {
        let mut d = base.clone();
        edit(&mut d);
        d
    }

    /// The slider range of the input at `path` (its metadata).
    fn slider(path: &str) -> Result<SliderRange, String> {
        input_rows(&DesignInputs::default())
            .into_iter()
            .find(|row| row.path == path)
            .and_then(|row| row.meta.range)
            .ok_or_else(|| format!("{path} has no slider"))
    }

    /// The Outside diameter slider's minimum at the Minimum OD preference `preference`: the
    /// expression of the crate-private `gui::size_controls::slider_minimum`, restated.
    fn slider_minimum(inputs: &DesignInputs, preference: Option<f64>) -> Result<f64, String> {
        let legacy = dimensions(inputs).ok_or("no size controls")?;
        let range = legacy.minimum_diameter_range;
        Ok(preference
            .unwrap_or(legacy.diameter_range.min.min(legacy.diameter_mm))
            .clamp(range.min, range.max))
    }

    /// `inputs` at the Outside diameter `requested` through the slider's own mapping at the
    /// default Minimum OD.
    fn at_diameter(inputs: &DesignInputs, requested: f64) -> Result<DesignInputs, String> {
        let minimum = slider_minimum(inputs, None)?;
        resize_diameter_with_minimum(inputs, requested, minimum)
    }

    /// The Outside diameter slider's top at the default Minimum OD.
    fn diameter_top(inputs: &DesignInputs) -> Result<f64, String> {
        let minimum = slider_minimum(inputs, None)?;
        dimensions_with_minimum_diameter(inputs, minimum)
            .map(|size| size.diameter_range.max)
            .ok_or_else(|| String::from("no diameter range"))
    }

    /// The design the panel shows with Auto on `variable`: the solved value, else the best
    /// value, through `gui::sizing::sized`, searched over the range the panel searches (ring
    /// size over `ring_radius_range` at the Minimum OD, as `MagcouplingPanel::sizing_range`).
    fn auto_shown(
        inputs: &DesignInputs,
        variable: FreeVariable,
        minimum: Option<f64>,
    ) -> Result<DesignInputs, String> {
        let range = match variable {
            FreeVariable::RingRadius => {
                ring_radius_range(inputs, minimum).unwrap_or_else(|| variable.range())
            }
            other => other.range(),
        };
        match solve_in(inputs, variable, inputs.metal.required_min_Nm, range) {
            Ok(SizingOutcome::Solved(point)) => Ok(sized(variable, inputs, point.value)),
            Ok(SizingOutcome::NotReachable { best: Some(best) }) => {
                Ok(sized(variable, inputs, best.value))
            }
            other => Err(format!("Auto shows no design: {other:?}")),
        }
    }

    /// The hot-low torque of a side with a number.
    #[allow(non_snake_case)] // unit suffix
    fn hot_low_Nm(side: &Side) -> Option<f64> {
        match side {
            Side::Change(change) => Some(change.hot_low_Nm),
            Side::Words(_) => None,
        }
    }

    /// The design of T-14's one-flip line (spec 10.1 step 6): (e2)+Auto length's design shown,
    /// Auto off, with Required halfway between its Outside diameter sides' hot-low torques, so
    /// the two sides read the same N·m while only one flips Hot-low torque.
    fn one_flip(e2: &DesignInputs) -> Result<DesignInputs, String> {
        let mut shown = auto_shown(e2, FreeVariable::AxialLength, None)?;
        let results = compute_all(&shown);
        let shown_card = card(&shown, &results, None, PAUSED);
        let row = shown_card
            .rows()
            .iter()
            .find(|row| row.control == Control::OutsideDiameter)
            .ok_or("no Outside diameter row")?;
        let (Some(fewer), Some(more)) = (hot_low_Nm(&row.fewer), hot_low_Nm(&row.more)) else {
            return Err(String::from("an Outside diameter side has no number"));
        };
        shown.metal.required_min_Nm = (fewer + more) / 2.0;
        Ok(shown)
    }

    /// Every design of `card_designs()`, under its key: the spec's section 2.1 in its order, T-14's
    /// one-flip design, R-13's boss plateau and the two count overflows.
    fn designs() -> Vec<Design> {
        let a = DesignInputs::default();
        let c = with(&a, |d| d.coupling.backiron = 0);
        let e = with(&a, |d| d.coupling.npole = 8);
        let e2 = with(&e, |d| d.metal.required_min_Nm = 1.5);
        let passing = with(&a, |d| {
            d.metal.face_gap_mm = 1.8;
            d.metal.cup_wall_corner_mm = 2.0;
            d.metal.required_min_Nm = 1.6;
            d.coupling.gearbox_input_rating_Nm = 1.0;
        });
        let lowest =
            |inputs: &DesignInputs| dimensions(inputs).map(|size| size.minimum_diameter_range.min);
        let no_size = || String::from("no size controls");
        let mut v = vec![
            design("(a)", Ok(a.clone()), None),
            auto(
                "(b)",
                &with(&a, |d| d.metal.face_gap_mm = 1.5),
                FreeVariable::AxialLength,
                None,
            ),
            design("(c)", Ok(c.clone()), None),
            design("(e)", Ok(e.clone()), None),
            design("(e2)", Ok(e2.clone()), None),
            design(
                "mixed",
                Ok(with(&a, |d| {
                    d.coupling.magnets.part_inner = String::from("B442");
                    d.coupling.magnets.part_outer = String::from("B842SH");
                })),
                None,
            ),
            design(
                "reverse",
                Ok(with(&a, |d| {
                    d.coupling.magnets.part_inner = String::from("B842SH");
                    d.coupling.magnets.part_outer = String::from("B442");
                })),
                None,
            ),
            auto("(a)+Auto length", &a, FreeVariable::AxialLength, None),
            auto("(a)+Auto count", &a, FreeVariable::MagnetsPerRing, None),
            auto("(a)+Auto ring", &a, FreeVariable::RingRadius, None),
            auto("(e2)+Auto length", &e2, FreeVariable::AxialLength, None),
            auto("(e2)+Auto count", &e2, FreeVariable::MagnetsPerRing, None),
            auto("(e2)+Auto ring", &e2, FreeVariable::RingRadius, None),
            auto(
                "(e2)+Auto ring, low Min OD",
                &e2,
                FreeVariable::RingRadius,
                lowest(&e2),
            ),
            design(
                "short magnets",
                Ok(with(&a, |d| {
                    d.coupling.c_end = 0.5;
                    d.coupling.magnets.part_inner.clear();
                    d.coupling.magnets.part_outer.clear();
                    d.coupling.magnets.manual_inner_length_mm = 2.0;
                    d.coupling.magnets.manual_outer_length_mm = 2.0;
                })),
                None,
            ),
            design(
                "end effect at a step",
                Ok(with(&a, |d| {
                    d.coupling.c_end = 0.5;
                    d.coupling.magnets.axial_length_mm = Some(5.0);
                })),
                None,
            ),
            design("odd count", Ok(with(&a, |d| d.coupling.npole = 9)), None),
            design("(e), low Min OD", Ok(e.clone()), lowest(&e)),
            design(
                "(a), Min OD 42.8",
                Ok(a.clone()),
                dimensions(&a).map(|size| size.diameter_mm),
            ),
            design("(a), Min OD 45", Ok(a.clone()), Some(45.0)),
            design("(a), Min OD 45: 43.3 mm", Ok(a.clone()), Some(43.3)),
            design(
                "shown does not fit",
                with_ring_radius(&a, 9.5).ok_or_else(|| String::from("with_ring_radius refused")),
                None,
            ),
            design(
                "keyway through",
                Ok(with(&a, |d| {
                    d.coupling.bore_mm = 16.0;
                    d.coupling.keyway_depth_mm = 2.5;
                })),
                None,
            ),
            design(
                "gap at 0.3",
                slider("metal.face_gap_mm").map(|r| with(&a, |d| d.metal.face_gap_mm = r.min)),
                None,
            ),
            design(
                "npole 4",
                slider("coupling.npole").map(|r| with(&a, |d| d.coupling.npole = r.min as i64)),
                None,
            ),
            design(
                "hot day",
                Ok(with(&a, |d| {
                    d.temperature.duty.hot_ambient_C = 80.0;
                    d.temperature.duty.driving_rise_C = 60.0;
                })),
                None,
            ),
            design(
                "only the joint fails",
                Ok(with(&passing, |d| {
                    d.shaft_joint.hub_interference_min_um = 8.0
                })),
                None,
            ),
            design("passing", Ok(passing.clone()), None),
            design(
                "short cavity",
                Ok(with(&a, |d| d.metal.cup_depth_mm = 11.0)),
                None,
            ),
            design(
                "temperature edge",
                Ok(with(&a, |d| d.temperature.duty.hot_ambient_C = 82.08)),
                None,
            ),
            design(
                "npole 40",
                Ok(with(&a, |d| {
                    d.coupling.npole = 40;
                    d.coupling.magnets.part_inner.clear();
                    d.coupling.magnets.part_outer.clear();
                    d.coupling.magnets.manual_inner_width_mm = 1.2;
                    d.coupling.magnets.manual_outer_width_mm = 1.2;
                })),
                None,
            ),
            design(
                "slider tops: gap 5.0",
                slider("metal.face_gap_mm").map(|r| with(&a, |d| d.metal.face_gap_mm = r.max)),
                None,
            ),
            design(
                "slider tops: Outside diameter 80",
                diameter_top(&a).and_then(|top| at_diameter(&a, top)),
                None,
            ),
            design(
                "slider tops: Overall length 69.9",
                dimensions(&a)
                    .ok_or_else(no_size)
                    .and_then(|size| resize_length(&a, size.length_range.max)),
                None,
            ),
            design(
                "slider tops: Overall length 21.1",
                dimensions(&a)
                    .ok_or_else(no_size)
                    .and_then(|size| resize_length(&a, size.length_range.min)),
                None,
            ),
        ];
        // One step inside each end, on the sliders' grids as the spec's section 2.1 gives them.
        for (name, base) in [("(a)", &a), ("(e)", &e)] {
            for length in [68.9, 22.1] {
                v.push(design(
                    &format!("steps to an end: {name} at length {length}"),
                    resize_length(base, length),
                    None,
                ));
            }
            v.push(design(
                &format!("steps to an end: {name} at Outside diameter 79"),
                at_diameter(base, 79.0),
                None,
            ));
        }
        for gap in [4.9, 0.4] {
            v.push(design(
                &format!("steps to an end: (a) at gap {gap}"),
                Ok(with(&a, |d| d.metal.face_gap_mm = gap)),
                None,
            ));
        }
        v.push(design(
            "steps to an end: (e), Min OD 41.8",
            Ok(e.clone()),
            dimensions(&e).map(|size| size.diameter_mm - 1.0),
        ));
        for (key, variation) in [("variation 1.0", 1.0), ("variation 1.0: 1.5", 1.5)] {
            v.push(design(
                key,
                Ok(with(&a, |d| d.metal.variation = variation)),
                None,
            ));
        }
        v.push(design(
            "zero torque",
            Ok(with(&a, |d| {
                d.coupling.magnets.part_inner.clear();
                d.coupling.magnets.part_outer.clear();
                d.coupling.magnets.manual_inner_br_T = 0.0;
            })),
            None,
        ));
        v.push(design(
            "(c), f_cal_original -0.95",
            Ok(with(&c, |d| d.calibration.f_cal_original = -0.95)),
            None,
        ));
        v.push(design("T-14 one flip", one_flip(&e2), None));
        // The spec's risk R-13: the shaft boss sets the Outside diameter (the boss above the cup
        // and the cap), so one +1 mm step lifts the cup by several millimetres.
        v.push(design(
            "R-13 boss plateau",
            Ok(with(&a, |d| {
                d.coupling.npole = 8;
                d.coupling.inner_back_apothem_mm = 8.0;
                d.metal.cap_od_mm = 30.0;
                d.metal.cap_thread_dia_mm = 29.0;
                d.metal.boss_od_mm = 40.0;
            })),
            None,
        ));
        // The table's count overflows (a design file's count at an end of i64): the design shown
        // does not fit, so the card draws its one line.
        for (key, npole) in [
            ("count overflow: i64::MAX", i64::MAX),
            ("count overflow: i64::MIN", i64::MIN),
        ] {
            v.push(design(
                key,
                Ok(with(&a, |d| d.coupling.npole = npole)),
                None,
            ));
        }
        v
    }

    /// The `card` and `gap` records.
    pub fn print() {
        let designs = designs();
        for d in &designs {
            let inputs = match &d.built {
                Ok(inputs) => inputs,
                Err(why) => {
                    out(
                        "card",
                        &format!("{} 00", d.key),
                        &format!("(not built: {why})"),
                    );
                    continue;
                }
            };
            let results = compute_all(inputs);
            let shown = card(inputs, &results, d.minimum, PAUSED);
            out("card", &format!("{} 00", d.key), CARD_HEADING);
            for (i, line) in shown.lines(d.auto).iter().enumerate() {
                out("card", &format!("{} {:02}", d.key, i + 1), line);
            }
        }
        // The readout's gap (torque_gap_text) at five designs, as the Size tab draws it: green
        // where hot-low meets the requirement, red where it is short.
        for key in ["(a)", "(b)", "(c)", "(e)", "(e2)"] {
            let text = match designs.iter().find(|d| d.key == key).map(|d| &d.built) {
                Some(Ok(inputs)) => {
                    let results = compute_all(inputs);
                    let (text, meets) = torque_gap_text(
                        results.metal.torque_hot_low_Nm,
                        inputs.metal.required_min_Nm,
                    );
                    format!("{text} [{}]", if meets { "green" } else { "red" })
                }
                Some(Err(why)) => format!("(not built: {why})"),
                None => String::from("(no such design)"),
            };
            out("gap", key, &text);
        }
    }
}
```

3. Write `$W/screen-dump/after/Cargo.toml` with the Write tool, exactly (Task 10 writes a `before/` twin with the
   feature off):
```toml
[package]
name = "screen-dump-after"
version = "0.1.0"
edition = "2024"
publish = false

# Its own workspace: a slice-3 release tool, not part of any crate of the repository.
[workspace]

[[bin]]
name = "screen-dump-after"
path = "../src/main.rs"

[features]
# The torque card's and the readout gap's records read the branch's public items, which main
# does not have (gui::torque_card and the gui::torque_gap_text re-export): on here only.
default = ["card"]
card = []

[dependencies]
magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs", features = ["gui"] }
```

4. Run (the first build takes a few minutes):
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
cp magcoupling-rs/Cargo.lock $W/screen-dump/after/Cargo.lock
CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/screen-dump" cargo run --quiet --manifest-path $W/screen-dump/after/Cargo.toml > $W/screen-T9.tsv
awk -F'\t' '{print $1}' $W/screen-T9.tsv | sort | uniq -c
awk -F'\t' '$1=="card" && $2 ~ / 00$/' $W/screen-T9.tsv | wc -l
awk -F'\t' '$1=="card" && $3 ~ /^\(not built/' $W/screen-T9.tsv
awk -F'\t' '$1=="gap" || $1=="size tab"' $W/screen-T9.tsv
git status --short
```
Expected: the record counts `18 assumption`, `2 bond screen`, `290 card`, `4 corner`, `20 count`, `5 gap`, `10 guide label`,
`10 guide why`, `187 input`, `34 input section`, `21 note`, `1112 result`, `18 row`, `18 row hover`, `1 share link`,
`4 size tab`, `15 start here`, `2 status`, `3 stored 3D`, `19 verdict` (every kind but `card`, `gap` and `size tab` as
BL-064 left them: slice 3 adds no input, result, row, check, note or section, measured on the tool at 69fa505 with the
feature off; `290 card` is 52 designs, 36 of them in the rows form with four rows and no Auto line (6 records each:
the heading, the caption, four rows), the 8 Auto designs (7: the Auto line too), the odd count (4: its gap and count
rows only, T-5) and the 7 one-line forms (2: short magnets, shown does not fit, variation 1.0 and 1.5, zero torque, the two
count overflows)); `52` designs; no `(not built` line; five `gap` lines and four `size tab` lines; `git status` prints only the owner's
untracked file. A different `card` count: read the records of the designs whose count differs (`awk -F'\t' '$1=="card"'
$W/screen-T9.tsv`) against the card's spec before going on, and say in the commit body what differs and why.
Measured after Task 8, the dump holds among its records `T-14 one flip 02` "Outside diameter: -1 mm -> -0.005 N·m
(-0.3 %), Running clearance -0.01 mm, Cup wall: wall needed -0.08 mm; +1 mm -> -0.005 N·m (-0.3 %), Hot-low torque:
fails, Space claim: fails, ...", `R-13 boss plateau 02` "Outside diameter: -1 mm -> at the end of its range; +1 mm ->
+0.12 N·m (+7.5 %), Running clearance +0.05 mm, Cup wall: wall needed +0.25 mm", and the two count overflows "No step
is compared while the magnets of the design shown do not fit."

- [ ] **Step 3: Check every text and number this task quotes against what ships (ruling B-1)**

1. Copy BL-064's check:
```bash
cp .superpowers/sdd/2026-10-17-bl064-shaft-joint/check_shipped.py .superpowers/sdd/2026-10-19-magcoupler-slice3/check_shipped.py
```
2. Five edits to `$W/check_shipped.py` with the Edit tool. Its BL-064 quotes stay: slice 3 must leave them true.

(K1) Old:
```python
"""BL-064 T9 Step 3 (ruling B-1 of BL-059, kept for BL-064): every on-screen text and number
Task 9's docs quote, checked against what the branch ships (the screen-dump tool's records).
Prints one line per quote: OK, or MISMATCH with the shipped text; exits 1 on any MISMATCH.
BEFORE_QUOTES are checked against main's dump (screen-dump/before) when it is given, and are
printed NOT CHECKED, without failing, when it is not.
```
New:
```python
"""Slice-3 T9 Step 3 (BL-064's tool, copied; ruling B-1 of BL-059): every on-screen text and
number the docs and the STOP sheet quote, checked against what the branch ships (the
screen-dump tool's records). BL-064's quotes are kept: slice 3 must leave them true. Prints one
line per quote: OK, or MISMATCH with the shipped text and whether its numbers are the quote's
(a wording change) or not; exits 1 on any MISMATCH. BEFORE_QUOTES are checked against main's
dump (screen-dump/before) when it is given, and are printed NOT CHECKED, without failing, when
it is not.
```

(K2) Old:
```python
import sys

# (kind, key, how, the text Task 9 quotes, where Task 9 quotes it)
# how: "has" = the record contains the text; "starts" = it starts with it; "ends" = it ends
# with it; "is" = equal.
```
New:
```python
import re
import sys

# (kind, key, how, the text quoted, where it is quoted)
# how: "has" = the record contains the text; "starts" = it starts with it; "ends" = it ends
# with it; "is" = equal; "lacks" = the record exists and does not contain it; "absent" = there
# is no such record.
```

(K3) The slice-3 quotes, at the end of `QUOTES` (the card's lines are the spec's drafted lines, section 2.3 and its
prototype's `out-run6.txt`; the docs of Steps 5-9 quote the texts and numbers whose `quoted in:` names them). Old:
```python
    ("input", "shaft_joint.output_shaft_fit_C", "has", "] default 20.00 slider -20 to 40 step 1", "tracker (Review leftovers, LO-76: its slider's bottom, -20 °C)"),
]
```
New:
```python
    ("input", "shaft_joint.output_shaft_fit_C", "has", "] default 20.00 slider -20 to 40 step 1", "tracker (Review leftovers, LO-76: its slider's bottom, -20 °C)"),
    # Slice 3 (spec section 8.6, test 27): every card line and gap text the docs and the STOP
    # sheet quote. The card texts are the spec's drafts until the G-3 review sets the words; on a
    # MISMATCH whose numbers are the same, the shipped words win (Task 9 Step 3's rule).
    ("card", "(a) 00", "is", "One step of each size control", "README card paragraph, tracker (the heading)"),
    ("card", "(a) 01", "is", "The change in the model's hot-low torque after one step of one control, everything else held.", "README card paragraph (the caption)"),
    ("card", "(a) 02", "is", "Outside diameter: -1 mm -> magnets do not fit; +1 mm -> +0.08 N·m (+3.5 %), Space claim: fails, Running clearance +0.01 mm, Cup wall: wall needed +0.07 mm, Gearbox input torque: cold-high +0.03 N·m", "STOP sheet; README card paragraph ('magnets do not fit', 'Space claim: fails', 'Cup wall: wall needed +0.07 mm')"),
    ("card", "(a) 03", "is", "Overall length: -1 mm -> -0.20 N·m (-8.8 %), Gearbox input torque: cold-high -0.07 N·m; +1 mm -> +0.20 N·m (+8.8 %), Gearbox input torque: cold-high +0.07 N·m", "STOP sheet; BL-065 notes (the length row's 8.8 %)"),
    ("card", "(a) 04", "is", "Magnet face gap: -0.1 mm -> +0.08 N·m (+3.3 %), Running clearance -0.10 mm, Cup wall: wall needed +0.02 mm, Gearbox input torque: cold-high +0.03 N·m; +0.1 mm -> -0.07 N·m (-3.2 %), Running clearance +0.10 mm, Cup wall: wall needed -0.02 mm, Gearbox input torque: cold-high -0.03 N·m", "STOP sheet"),
    ("card", "(a) 05", "is", "Magnets per ring: -2 -> -0.61 N·m (-26.8 %), Gearbox input torque: passes, Cup wall: wall needed +0.48 mm; +2 -> magnets do not fit", "STOP sheet; BL-065 notes ((a)'s count -2, -26.8 %)"),
    ("card", "(a) 06", "absent", "", "STOP sheet (four rows at (a))"),
    ("card", "(b) 02", "is", "Auto sets the magnet length again after a change; each row keeps the length shown.", "STOP sheet ((b)'s Auto line); README card paragraph"),
    ("card", "(b) 04", "is", "Overall length: Set by Auto; turn Auto off to edit", "STOP sheet ((b)'s held row); README card paragraph"),
    ("card", "(c) 05", "is", "Magnets per ring: -2 -> -0.59 N·m (-41.1 %), includes a calibration change, Shaft joints: Inner hub smallest measured interference that holds -2.43 µm, Shaft joints: Cup smallest measured interference that holds -1.92 µm; +2 -> magnets do not fit", "STOP sheet; README card paragraph ('includes a calibration change')"),
    ("card", "(e) 02", "is", "Outside diameter: -1 mm -> -0.006 N·m (-0.4 %), Running clearance -0.01 mm, Cup wall: wall needed -0.08 mm; +1 mm -> -0.005 N·m (-0.3 %), Space claim: fails, Running clearance +0.01 mm, Cup wall: wall needed +0.08 mm", "STOP sheet (T-14's shared decimals); README card paragraph, tracker, 04-memory (the owner's case)"),
    ("card", "(e) 02", "lacks", "-1 mm -> -0.01 N·m", "STOP sheet (T-14: the per-number form does not ship)"),
    ("card", "(e) 03", "is", "Overall length: -1 mm -> -0.15 N·m (-9.1 %); +1 mm -> +0.15 N·m (+9.1 %), Gearbox input torque: fails", "STOP sheet; README card paragraph, tracker, 04-memory (the owner's case)"),
    ("card", "(e) 05", "is", "Magnets per ring: -2 -> -0.89 N·m (-53.0 %), Outside diameter +2.2 mm, Space claim: fails, Cup wall: wall needed +0.80 mm; +2 -> +0.61 N·m (+36.6 %), Gearbox input torque: fails, Cup wall: wall needed -0.48 mm", "STOP sheet"),
    ("card", "mixed 03", "is", "Overall length: -1 mm -> not a step: the overlapping magnet length does not change by the step; +1 mm -> not a step: the overlapping magnet length does not change by the step", "STOP sheet (the mixed pair's JUMP); README card paragraph"),
    ("card", "reverse 03", "is", "Overall length: -1 mm -> -0.20 N·m (-19.9 %); +1 mm -> +0.20 N·m (+19.9 %)", "STOP sheet (the reverse pair: numbers)"),
    ("card", "short cavity 03", "is", "Overall length: -1 mm -> not a step: the overlapping magnet length does not change by the step; +1 mm -> not a step: the overlapping magnet length does not change by the step", "STOP sheet (O-1: the short cavity's +1 mm side reads words)"),
    ("card", "temperature edge 02", "has", "+1 mm -> +0.08 N·m (+3.5 %), Temperature limits: caution, Space claim: fails", "STOP sheet (the temperature edge's flips)"),
    ("card", "temperature edge 03", "has", "+1 mm -> +0.20 N·m (+8.8 %), Temperature limits: caution", "STOP sheet (the temperature edge's flips)"),
    ("card", "temperature edge 05", "has", "-2 -> -0.61 N·m (-26.8 %), Temperature limits: caution", "STOP sheet (the temperature edge's flips)"),
    ("card", "(c), f_cal_original -0.95 05", "is", "Magnets per ring: -2 -> -2.29 N·m (-158.9 %), includes a calibration change, Shaft joints: passes; +2 -> magnets do not fit", "STOP sheet (round 4: a count side below zero shows its number)"),
    ("card", "(a), Min OD 45 02", "is", "Outside diameter: -1 mm -> at the end of its range; +1 mm -> at the end of its range", "STOP sheet (the Minimum OD 45 words)"),
    ("card", "(a), Min OD 45: 43.3 mm 02", "starts", "Outside diameter: -1 mm -> at the end of its range; +1 mm -> +0.08 N·m (+3.5 %)", "STOP sheet (the Minimum OD 45 words, 43.3 beside them)"),
    ("card", "(a), Min OD 42.8 02", "starts", "Outside diameter: -1 mm -> at the end of its range;", "README card paragraph ('at the end of its range')"),
    ("card", "end effect at a step 03", "starts", "Overall length: -1 mm -> no prediction;", "README card paragraph ('no prediction')"),
    ("card", "odd count 03", "is", "Magnets per ring: -2 -> not a value the slider takes; +2 -> not a value the slider takes", "README card paragraph ('not a value the slider takes')"),
    ("card", "keyway through 03", "is", "Overall length: -1 mm -> the keyway breaks through the hub; +1 mm -> the keyway breaks through the hub", "README card paragraph ('the keyway breaks through the hub')"),
    ("card", "T-14 one flip 02", "is", "Outside diameter: -1 mm -> -0.005 N·m (-0.3 %), Running clearance -0.01 mm, Cup wall: wall needed -0.08 mm; +1 mm -> -0.005 N·m (-0.3 %), Hot-low torque: fails, Space claim: fails, Running clearance +0.01 mm, Cup wall: wall needed +0.08 mm", "STOP sheet (T-14's one-flip line)"),
    ("card", "R-13 boss plateau 02", "starts", "Outside diameter: -1 mm -> at the end of its range; +1 mm -> +0.12 N·m (+7.5 %)", "STOP sheet (R-13: the boss plateau reads a number)"),
    ("card", "short magnets 01", "is", "No step is compared while the design shown has no torque prediction.", "STOP sheet (a one-line form, T-12)"),
    ("card", "short magnets 02", "absent", "", "STOP sheet (the one line drawn alone, T-12)"),
    ("card", "shown does not fit 01", "is", "No step is compared while the magnets of the design shown do not fit.", "STOP sheet (a one-line form, T-12)"),
    ("card", "shown does not fit 02", "absent", "", "STOP sheet (the one line drawn alone, T-12)"),
    ("card", "variation 1.0 01", "is", "No step is compared while the design shown has a pull-out or hot-low torque that is not above zero.", "STOP sheet (SHOWN_NO_HOT_LOW)"),
    ("card", "variation 1.0 02", "absent", "", "STOP sheet (the one line drawn alone, T-12)"),
    ("card", "zero torque 01", "is", "No step is compared while the design shown has a pull-out or hot-low torque that is not above zero.", "STOP sheet (SHOWN_NO_HOT_LOW at zero torque)"),
    ("card", "zero torque 02", "absent", "", "STOP sheet (the one line drawn alone, T-12)"),
    ("gap", "(a)", "is", "short 0.21 N·m (-8.6 %) [red]", "tracker (Number format); STOP sheet"),
    ("gap", "(b)", "is", "margin +0.00 N·m (+0.0 %) [green]", "STOP sheet"),
    ("gap", "(c)", "is", "short 1.06 N·m (-42.3 %) [red]", "STOP sheet"),
    ("gap", "(e)", "is", "short 0.83 N·m (-33.1 %) [red]", "STOP sheet"),
    ("gap", "(e2)", "is", "margin +0.17 N·m (+11.5 %) [green]", "STOP sheet"),
    ("size tab", "magnet count line/Auto None", "is", "The ring radius stays the same when you change the magnet count; the cup can still change size.", "README Size-tab paragraph, 02-system Size-tab invariant"),
    ("size tab", "magnet count line/Auto Some(AxialLength)", "is", "The ring radius stays the same when you change the magnet count; the cup can still change size.", "README Size-tab paragraph, 02-system Size-tab invariant (Auto on the magnet length)"),
    ("size tab", "magnet count line/Auto Some(RingRadius)", "is", "(none)", "README Size-tab paragraph, 02-system Size-tab invariant"),
    ("size tab", "magnet count line/Auto Some(MagnetsPerRing)", "is", "(none)", "README Size-tab paragraph, 02-system Size-tab invariant"),
]
```

(K4) `main`'s side: its share link (no input added) and its first lines under Magnets per ring; no card. Old:
```python
BEFORE_QUOTES = [
    ("share link", "default design", "is", "2462 characters", "tracker (Engine bullet: grows from 2,462)"),
]
```
New:
```python
BEFORE_QUOTES = [
    ("share link", "default design", "is", "2643 characters", "the STOP sheet (no input added: main's link is the branch's length)"),
    ("size tab", "magnet count line/Auto None", "is", "The ring radius stays the same when you change the magnet count; the cup OD below can still change.", "the STOP sheet (the shipped first line, before)"),
    ("size tab", "magnet count line/Auto Some(RingRadius)", "is", "Auto resizes the ring when you change the magnet count; the lines below keep the present ring radius.", "the STOP sheet (AUTO_RESIZES_RING, before); T10's bundle check"),
    ("size tab", "magnet count line/Auto Some(AxialLength)", "is", "Auto resets the magnet length when you change the magnet count; the lines below keep the present length.", "the STOP sheet (AUTO_RESETS_LENGTH, before)"),
    ("card", "(a) 00", "absent", "", "the STOP sheet (main has no card)"),
]

# The numbers in a text, in order: a MISMATCH with the same numbers is a wording change.
NUMBERS = re.compile(r"[-+]?\d+(?:\.\d+)?")
```

(K5) The two new kinds of match, and a MISMATCH says whether its numbers are the quote's. Old:
```python
        got = records.get((kind, key))
        ok = got is not None and (
            (how == "is" and got == want)
            or (how == "starts" and got.startswith(want))
            or (how == "ends" and got.endswith(want))
            or (how == "has" and want in got)
        )
        bad += not ok
        print(f"{'OK      ' if ok else 'MISMATCH'} {side}[{kind}] {key}: wants {how} {want!r}"
              + ("" if ok else f"; ships {got!r}; quoted in: {where}"))
```
New:
```python
        got = records.get((kind, key))
        if how == "absent":
            ok = got is None
        else:
            ok = got is not None and (
                (how == "is" and got == want)
                or (how == "starts" and got.startswith(want))
                or (how == "ends" and got.endswith(want))
                or (how == "has" and want in got)
                or (how == "lacks" and want not in got)
            )
        bad += not ok
        note = ""
        if not ok and got is not None and how in ("is", "starts", "ends", "has"):
            same = NUMBERS.findall(want) == NUMBERS.findall(got)
            note = (" (the same numbers: a wording change)" if same
                    else " (NUMBERS DIFFER: not a wording change)")
        print(f"{'OK      ' if ok else 'MISMATCH'} {side}[{kind}] {key}: wants {how} {want!r}"
              + ("" if ok else f"; ships {got!r}{note}; quoted in: {where}"))
```

3. Run:
```bash
python -I .superpowers/sdd/2026-10-19-magcoupler-slice3/check_shipped.py .superpowers/sdd/2026-10-19-magcoupler-slice3/screen-T9.tsv
```
Expected: every quote line starts `OK` (111 lines), the stored-3D line included, the five `BEFORE_QUOTES` print `NOT
CHECKED (no before dump given)`, and the last line is `0 mismatch(es)` (exit 0).

On a `MISMATCH`:
1. `ships None`, or `NUMBERS DIFFER`: not a wording change. STOP and report the line: a missing record means the owning
   task's code is not in; a number differs from the spec's and its card test (tests 2 and 2a pin these numbers).
2. `the same numbers: a wording change` (the G-3 review of T7 reworded a draft): the shipped text wins. Wherever Steps
   5-9 quote the expected text (the line's `quoted in:` names where; "STOP sheet" alone means no docs line here),
   write the shipped words instead, the sentence otherwise unchanged; change that entry's text in `QUOTES` to the
   shipped one; run the check again, until `0 mismatch(es)`; add one body line per substitution to the commit message
   (Step 12), for example `Quotes the shipped JUMP words "<shipped>" in place of the draft.`

4. The dump's designs are the table's (plan review 2.2 and 1.2, plan deviation PD-58): the dump restates `card_designs()`'s
   designs, so one script checks each restated design against the tested one. Write
   `.superpowers/sdd/2026-10-19-magcoupler-slice3/same_designs.py` with the Write tool, exactly:
```python
"""Every card the screen dump prints for a key of card_designs() is the card the G-3 review dump
prints for that key: the dump restates the table's designs (a separate crate cannot read the
test module), so this ties each restated design to the tested one. Usage:
  python -I same_designs.py SCREEN_TSV REVIEW_DUMP
Prints one line per shared key and exits 1 on any difference or on a table key the dump lacks.
"""
import re
import sys

screen_path, review_path = sys.argv[1], sys.argv[2]

# The screen dump: card<TAB><key> NN<TAB><line>, NN 00 the heading, the lines from 01.
screen = {}
for line in open(screen_path, encoding='utf-8'):
    kind, key, text = (line.rstrip('\n').split('\t') + ['', ''])[:3]
    if kind != 'card':
        continue
    design, number = key.rsplit(' ', 1)
    if number != '00':
        screen.setdefault(design, []).append(text)

# The review dump: [card/<key>/<Auto state>/NN] <line>, the lines from 00; each design's own
# Auto state is its first block (the extra Auto states of four designs follow the table).
review, state_of = {}, {}
tag = re.compile(r'^\[card/(.+)/(Auto [A-Za-z]+)/(\d\d)\] (.*)$')
for line in open(review_path, encoding='utf-8'):
    m = tag.match(line.rstrip('\n'))
    if not m:
        continue
    design, state, _, text = m.groups()
    state_of.setdefault(design, state)
    if state == state_of[design]:
        review.setdefault(design, []).append(text)

bad = 0
for design in review:
    if design not in screen:
        print(f'MISSING  {design}: the table has it, the screen dump does not')
        bad += 1
    elif screen[design] != review[design]:
        print(f'DIFFERS  {design}:')
        for s, r in zip(screen[design], review[design]):
            if s != r:
                print(f'  dump  : {s}\n  table : {r}')
        bad += 1
    else:
        print(f'SAME     {design} ({len(review[design])} lines)')
extra = [d for d in screen if d not in review]
print(f'only in the dump (no table design): {extra}')
print(f'{bad} difference(s)')
sys.exit(1 if bad else 0)
```

   Then run:
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > $W/g3-T9.txt
python -I $W/same_designs.py $W/screen-T9.tsv $W/g3-T9.txt
```

   Expected (measured after Task 8): 52 lines starting `SAME`, then `only in the dump (no table design): []` and `0
   difference(s)`, exit 0. On a `DIFFERS` or `MISSING` line, the dump's restated design or helper differs from the
   tested table: fix the dump's `designs()`, never the table, and run both commands again. (A planted one-character
   drift in the dump's `passing 02` line printed `DIFFERS  passing:` with the two lines and exited 1.)

- [ ] **Step 4: The stale-text sweep, before the edits (red)**

Run:
```bash
STALE='\(-P %\)|\(\+P %\)|from the engine \(the neighbour|cup OD below can still|Auto resizes the ring when you change|Auto resets the magnet length when you change|lines below keep the present|cup OD at N-2 and N\+2|counts N-2 and N\+2 inside|neighbours N-2 and N\+2|Neighbour, NeighbourOutcome|RING_SIZE_STAYS, AUTO_RESIZES_RING|AUTO_RESIZES_RING with Auto on|AUTO_RESETS_LENGTH with Auto on|recomputes only when the inputs or the Minimum OD change|Outside diameter on the Size tab \(to the note|Magnets per ring and Outside diameter \(guidance::SIZE_TAB_NOTE\)|Why links under Magnets per ring and Outside diameter open the torque note|the Magnets per ring link is;|neighbours outside 4 to 40 magnets|of the neighbour design|[Nn]ext: the torque sensitivity card|slice 3, a torque sensitivity card|about 30 compute_all calls|under 1\.6 ms per changed design|which call them'
grep -n -o -E ".{0,30}($STALE).{0,30}" magcoupling-rs/README.md docs/ai/01-meta.yaml docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml | cut -c1-170
echo ---
CSTALE='a whole percent of the requirement|\(-12 %\)|\(\+16 %\)|a whole signed percentage|two magnets fewer and two more \(each|until the inputs or the minimum diameter change|Why links under Magnets per ring and Outside diameter on the Size tab lead|changes to whole percent|new magnet count does to the size Auto sets|the neighbours. .compute_all. runs|computed again only when the inputs or the minimum diameter change|when the inputs or the minimum differ from the last call|under Magnets per ring and Outside diameter to Start here|for the design shown and the minimum OD,|const AUTO_RESIZES_RING|const AUTO_RESETS_LENGTH|enum NeighbourOutcome|struct Neighbour |fn neighbour_outcome|fn neighbour_counts|cup OD below can still change|label\("Build checks are shown in the dashboard'
grep -rn -o -E ".{0,30}($CSTALE).{0,30}" magcoupling-rs/src magcoupling-rs/tests --include=*.rs | cut -c1-170
```
What the patterns catch (measured at 69fa505: 24 matches on 21 README and docs/ai lines from the first command, 23 code
lines from the second, every one a text slice 3 makes stale), each with the task that writes its new text:

| Text at 69fa505 | Where | Written by |
|---|---|---|
| "short X N·m (-P %)", "margin +X N·m (+P %)" | README ~:203 | T1 |
| "Next: the torque sensitivity card" | README ~:46 | T9 (R1) |
| "(the neighbour", "the cup OD below can still", "Auto resizes the ring when you change", "the lines below keep the present", "cup OD at N-2 and N+2" | README ~:214-222 | T5 (Step 7 (c); T9's R3 is its guard) |
| "Outside diameter on the Size tab (to the note" | README ~:362 | T5 (Step 7 (d); R4 is its guard) |
| "Why links under Magnets per ring and Outside diameter open the torque note", "the Magnets per ring link is;" | README ~:686 | T5 (Step 7 (e), (f); R6 and R7 are their guards) |
| "of the neighbour design", "neighbours outside 4 to 40 magnets" | README ~:693 | T5 (Step 7 (h); R10 is its guard) |
| "(about 30 compute_all calls)" | 02-system :106 | T10 (Step 8: the count, with the measured times) |
| "Magnets per ring and Outside diameter (guidance::SIZE_TAB_NOTE)" | 02-system :115 | T5 (Step 7 (k); S4 is its guard) |
| "counts N-2 and N+2 inside", "cup OD below can still", "AUTO_RESIZES_RING with Auto on", "AUTO_RESETS_LENGTH with Auto on", "recomputes only when the inputs or the Minimum OD change" | 02-system :116 | T5 (Step 7 (l) to (n); S2 and S3 are their guards) |
| "under 1.6 ms per changed design" | 02-system :116 | T10 (Step 8: the measured figures) |
| "next: the torque sensitivity card" | 02-system :488 | T9 (S5) |
| "neighbours N-2 and N+2", "Neighbour, NeighbourOutcome", "RING_SIZE_STAYS, AUTO_RESIZES_RING" | 03-structure :68 | T5 (Step 7 (o); C2 is its guard) |
| "slice 3, a torque sensitivity card" | 04-memory :13 | T9 (M2) |
| code: `size_controls.rs:175-176` (`torque_gap_text`'s doc: "a whole percent", "(-12 %)", "(+16 %)") and the test's literals `:421`, `:425`; `format.rs:89` ("a whole signed percentage") | | T1 |
| code: `guidance.rs:34`, `:38` (the module doc), `:606` (`SIZE_TAB_NOTE`'s doc), `:2712` (the what-moves-what comment), `:2717` (`RING_SIZE_STAYS`), `:2722`, `:2727` (the two Auto lines), `:2730` (`magnet_count_line`'s doc), `:2750`, `:2770`, `:2902`, `:2924` (the neighbour code), `:2989-2990`, `:3008` (the cache's docs), `:3928` (a test comment); `panel.rs:310` (the `consequences` field's doc), `:936` (the label the card replaces) | | T5 |

Patterns are chosen so no new text matches them; `which call them` keeps the phrase Task 5 removed from coming back
(ruling C-3). Expected after T1-T8 (the text is what matters; the line numbers move with the rows earlier tasks add),
measured: the first command prints exactly README `:46` (R1), 02-system `:106` and `:116` (T10), `:488` (S5) and
04-memory `:13` (M2), and nothing after `---`. Any other line: it belongs to the task the table names;
apply that task's docs or code-comment step from this plan as the Rules say, and note it for the commit body (Step
12). A line that matches but is correct as it stands (a false positive; none is expected): STOP and report it to the
controller with the line, rather than editing the pattern list.

(The module doc of `gui/size_controls.rs`, `:6-7`, which spec 9.2 lists, stays true after slice 3: its controls still
draw their what-moves-what lines, and the card is the panel's, drawn with `torque_card`'s texts; no pattern covers it.)

- [ ] **Step 5: `docs/ai/backlog.yaml`: BL-073, BL-074 and BL-075, and BL-065's line**

1. The three items, exactly as spec 9.1 gives them, but for BL-073's `status`, which is `in_progress` until the owner's
   OK (T10 Step 11 sets `fixed`), after BL-072 and before `- id: BL-055`. Old (unique):
```text
- id: BL-055
```
New:
```yaml
- id: BL-073
  title: "Magcoupler Size tab: a torque card that shows what one step of each size control does to hot-low torque, and what it costs"
  dimension: gui
  risk: physics
  evidence: "Guidance order G-1 (docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md:34): the torque sensitivity card after the note rewrite; the trigger case (spec1:8-9): 10 -> 8 magnets and a lower target did not shrink the coupling and nothing on screen said why. At 8 magnets on the default ring (compute_all, main 69fa505) Outside diameter -1 / +1 mm moves hot-low -0.37 / -0.28 % while Overall length moves it -9.05 / +9.05 %; nothing on screen says so. Owner decisions 1A-17A (2026-10-19), spec docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md."
  acceptance: "gui/torque_card.rs (pure) and its card on the Size tab between the readouts and the drawing, per the spec: four rows (Outside diameter, Overall length, Magnet face gap, Magnets per ring) at ±1 mm, ±1 mm, ±0.1 mm, ±2 through each slider's own mapping; every number a difference of two compute_all runs; words where a side does not fit, has no prediction, leaves its slider, jumps or breaks the keyway; flips and failing-check quantities per side; the Auto marking; the N±2 lines moved off Magnets per ring; one Size-tab number format (one decimal of percent, N·m to the fewest decimals that keep the true sign, a row's two sides at one precision) on the card and the readout's gap; the tests of the spec's section 8 red first; G-3 physics review of every card string; whole-branch review; GATE PASS with gate 12 running; web build; gui-smoke with the card step and a browser drag-frame p95 at or under 16.7 ms; docs/ai 02-05 and the README; STOP with the sheet; the owner's OK of the diff."
  priority: 2
  status: in_progress
  notes: "Owner decisions 1A-17A (2026-10-19, owner-decisions.md in the gitignored slice-3 research). Reverses spec1 section 3.3 for the neighbour lines (7A) and the readout's whole-percent format (10B). The controller's rulings on the owner's night-run approval, on the STOP sheet for the owner to reverse: the jump rule widened (owner question O-1: words whenever the overlapping magnet length changes by anything other than the step) and no temperature margin (O-2). BL-065 stays open (12A): the card ships the model's numbers. Deploy authorization is asked at the STOP."
- id: BL-074
  title: "Overall length from a blank override jumps when the housing does not hold the outer ring's own length or the inner part is shorter: the first slider move sets both rings to one length (B442 inner, B842SH outer: -1 mm reads +106 %; matched parts in an 11 mm cup cavity: +1 mm moves hot-low -6.2 %)"
  dimension: gui
  risk: physics
  evidence: "magcoupling-rs/src/engine/size_controls.rs:453-466: resize_length sets both rings' axial override to requested - length_range.min + axial_range.min (:460); from a blank override the length range follows the outer ring the cup cavity holds (:220-226), the cavity following the ring as max(cup_depth + L - own, L) (engine/housing.rs:106-108). With part_inner B442 (6.35 mm) and part_outer B842SH (12.7 mm), overall length 31.80 -> 30.80 mm gives override 11.7 mm, active length 6.35 -> 11.70 mm and hot-low 1.0099 -> 2.0844 N m (+106.39 %); +1 mm gives +146.16 % (compute_all, main 69fa505; slice-3 spec section 2.3). The reverse pair moves the active length by the step but trims the longer inner ring to the outer's length less or plus the step (12.7 -> 5.35 / 7.35 mm, the hub length following it, 13.0 -> 5.65 / 7.65 mm). Matched parts move it by the step only while the cup cavity input (metal.cup_depth_mm) is at least the outer ring's length: at 11.0 mm (input range 3-40) -1 mm sets both rings to 10.0 mm (active -2.70 mm) and +1 mm to 12.0 mm (active -0.70 mm, hot-low -6.15 %) (slice-3 spec physics review 1; spec section 2.3). No doc records it (grep of docs/ai and the README). The slice-3 torque card shows words on every first step that is not a step (its jump rule, widened by owner question O-1, answered 2026-10-19); the slider itself does it unannounced. A diameter step can jump the same way (slice-3 spec R-13): where the shaft boss sets the Outside diameter (metal.boss_od_mm, slider 12-40 mm, above both the cup and the cap, all under Detailed inputs), the slider's mapping lifts the cup from its own size to the boss plus the step in one step (requested_cup = requested - cap_offset.max(0), engine/size_controls.rs:431-432). At npole 8, ring 8.0 mm, cap 30 mm, cap thread 29 mm and boss 40 mm, +1 mm of Outside diameter lifts the cup +3.22 mm (ring 8.00 -> 9.49 mm, against 0.46 mm of ring for 1 mm elsewhere at 8 magnets) and hot-low +7.55 %, which the card shows as a number (it has no diameter jump rule); with the default boss (22 mm) and magnets the boss never sets the diameter (slice-3 spec, section 3d of its prototype's out-run5.txt)."
  acceptance: "owner decides whether the jump is intended, for a blank override whose housing does not hold the outer ring's own length, or a shorter inner part. If intended: the Overall length hover (LENGTH_HOVER, gui/size_controls.rs:46), the README Size-tab paragraph and docs/ai/02-system.yaml say that a first move from a blank override sets both rings to one length, with headless tests pinning the mixed pair's, the reverse pair's and the short cavity's first step. If not: red tests first at the mixed pair and the short cavity (the first step moves the active length by 1 mm, or is refused with a reason), then a mapping change in engine/size_controls.rs chosen with the owner (for example the override counted from the shorter ring, or a per-ring step), the card's jump rule kept as a guard, the README and docs/ai updated; G-3 review of any new text; GATE PASS with gate 12 running"
  priority: 3
  status: open
  notes: "Filed with slice 3 (owner decision 17A, 2026-10-19), from the slice-3 context critique's finding 6, widened by the slice-3 spec's physics review 1 (MF-2, SF-5: the short cavity and the reverse pair's trim). The torque card's jump rule (spec section 3.3 step 7, test 3; widened by owner question O-1) keeps the card honest either way on the length row; the diameter row's boss plateau has no such rule (spec R-13)."
- id: BL-075
  title: "BL-064 fold-in review notes with substance: a too-cold gearbox shaft hides a judged red joint, the joint links read the inputs not compute's copies, the fitting source quotes results not links, and test and DRY polish"
  dimension: quality
  risk: physics
  evidence: "Gitignored records .superpowers/sdd/2026-10-18-bl064-shaft-joint/lo-B1-review.md .. lo-B12-review.md and lo-final-review.md (Minor items; ruling B-6 in rulings.md left them unfiled). (1) lo-B1 M-2, lo-final M-1 O-2: CannotBeMade::GearboxShaftTooCold (magcoupling-rs/src/engine/shaft_joint.rs:912, returned at :960) stops every joint being judged though the gearbox shaft's temperature enters only the cup's fitting term: compute_all at 69fa505 with the hub minimum at 10 um: gearbox shaft at -20 C reads 'The fit slips (inner hub): holds 3.48 N·m against 7.65 N·m', at -21 C (a file or link value; the slider stops at -20 C, OUTPUT_SHAFT_COLDEST_C :175) 'Shaft joint check unknown: the gearbox shaft is below its slider's bottom' (amber; slice-3 spec scratch crate, src/bin/bl075.rs; the review's own example said 3.85 N·m). (2) lo-B2 M-1: joint_links (magcoupling-rs/src/engine/api.rs:177-200) reads the eight geometry values from inputs, so the_joints_rebuilt_from_the_results_are_the_engine_s no longer guards against compute overriding one in ci or md. (3) lo-B4 M-2: fitting_source (magcoupling-rs/src/gui/guidance.rs:2193) chooses its arm on the links but quotes the adhesive limit and the inner onset from results (onset_texts :2226); a planted fault read 'above the 49.93 C allowed: the magnets' single-ring onset, 91.35 C, less 10.00 C'. (4) Polish: one keyed rule written twice (lo-B1 M-5, shaft_joint.rs); 10f64.powi(-(decimals as i32)) typed twice (lo-B4 M-3, guidance.rs:1917, :2354); retyped test designs and lookups (lo-B8 M-2 and M-3, lo-B9 M-3 and M-4); unpinned edges (lo-B1 M-6, lo-B5 M-2, lo-B7 M-2, lo-B10 M-2); four inline paths in the joint guide (lo-final M-5)."
  acceptance: "owner rules item (1) (keep the amber with its reason recorded, or refuse only the cup's fitting term so the inner hub stays judged, red test first at hub minimum 10 um and gearbox shaft -21 C); (2) joint_links takes &CouplingInputs and &MetalDesignInputs, compute passes ci and md, the rebuilt-joints test regains its guard (a planted override fails it); (3) fitting_source quotes links.adhesive_limit_C and links.inner_onset_single_ring_C through the links shaft_joint_problem already built (pins unchanged; a planted fault in joint_links fails one); (4) the listed polish, each with a test that fails before it where it is behaviour; G-3 re-check of any on-screen text; GATE PASS with gate 12 running"
  priority: 3
  status: open
  notes: "Filed with slice 3 (owner decision 17B, 2026-10-19). Ruling B-6 (2026-10-19) kept the 70 Minor notes in the review records unless the owner asked to file them; these are the ones with substance. Item (1) sits against build ruling B-5 note 1's 'a real red stays red'."
- id: BL-055
```
2. BL-065's notes gain spec 9.1's one line. Old (unique):
```text
The end_effect note's 3D band sentence names the steel case meanwhile."
```
New:
```text
The end_effect note's 3D band sentence names the steel case meanwhile. The slice-3 torque card's percents are a consumer: re-run its probe rows when BL-065 lands (in the model with c_end scaled to the steel slice, the length row reads about 9.3 %/mm against 8.8 %, and (a)'s count -2 about -28.1 % against -26.8 %; the N·m of the length row is unchanged)."
```
(Its "8.8 %" and "-26.8 %" are (a)'s card numbers, which Step 3 checked.)

- [ ] **Step 6: `magcoupling-rs/README.md`**

(R1) No code task makes this line stale. Skip if `grep -c -F 'Guidance slice 3 complete' magcoupling-rs/README.md` prints 1 or more. Old:
```text
Next: the torque sensitivity card (slice 3), then M3 (live 3D fields); BL-065 to BL-072 are open in `docs/ai/backlog.yaml`.
```
New:
```text
Guidance slice 3 complete (spec `docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md`, BL-073, owner decisions 1A-17A): the torque card on the Size tab, one fixed step each way of each size control with its change in hot-low torque and what the step costs, and one number format for the Size tab (see [The panel](#the-panel)). Next: guided lessons, plot captions and drawing titles, then M3 (live 3D fields); BL-065 to BL-072, BL-074 and BL-075 are open in `docs/ai/backlog.yaml`.
```

(R2) Made stale by T3's code; written here unless T3 wrote it. Skip if `grep -c -F 'torque_card.rs` (' magcoupling-rs/README.md` prints 1 or more. Old:
```text
`size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines and Why links)
```
New:
```text
`torque_card.rs` (the torque card, pure: `card`, `TorqueCard` with its `lines` and `rows`, `Control`, `Words`, `held_line`, `count_step`, `side_words`, `STEP_SLACK_MM` and the card's texts; slice 3, BL-073), `size_controls.rs` (the Size and torque tab's controls, torque readouts, Auto toggle, hint lines, consequence lines and Why links; since slice 3 the N-2 and N+2 lines are the torque card's Magnets per ring row, and `slider_minimum`, the Outside diameter slider's minimum, is shared with the card)
```

(R3) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F -e 'the first line until slice 3' -e 'fewer and two more moved to the torque' magcoupling-rs/README.md` prints 1 or more. Old:
```text
Under the controls, short lines say what moves what, each read from the engine (the neighbour
magnet counts from `compute_all` of the design with only `coupling.npole` changed; the
diameters and lengths from `engine::size_controls`), never typed. For **Magnets per ring**:
a first line that follows the Auto state (`guidance::magnet_count_line`): with Auto off, "The
ring radius stays the same when you change the magnet count; the cup OD below can still
change."; with Auto on the ring radius, "Auto resizes the ring when you change the magnet count;
the lines below keep the present ring radius."; with Auto on the magnet length, "Auto resets the
magnet length when you change the magnet count; the lines below keep the present length."; and
none with Auto on the magnet count, whose slider is locked. Then the pull-out and cup OD at N-2 and N+2 inside the slider range (a neighbour whose
blocks do not fit, or whose end effect is out of range, reads in words; one that fails only the
keyway still shows its torque, since the keyway does not depend on the count), and the
smallest outside diameter that fits N magnets (where only the Minimum OD floor stops it, the
line says how far it could go and that lowering Minimum OD reaches it). For **Outside
diameter**: "Set by: cap A, cup B, boss C mm", the largest being the outside diameter. For
**Overall length**: "Magnets L mm + housing H mm".
```
New:
```text
Under the controls, short lines say what moves what, each read from the engine (the
diameters and lengths from `engine::size_controls`), never typed. For **Magnets per ring**: a
first line that follows the Auto state (`guidance::magnet_count_line`): with Auto off or on the
magnet length, "The ring radius stays the same when you change the magnet count; the cup can
still change size."; none with Auto on the ring radius or the magnet count (the torque card's
Auto line says what Auto does after a change). Then the smallest outside diameter that fits N
magnets (where only the Minimum OD floor stops it, the line says how far it could go and that
lowering Minimum OD reaches it). The pull-out at two magnets fewer and two more, which followed
the first line until slice 3, is now the torque card's Magnets per ring row, in hot-low. For
**Outside diameter**: "Set by: cap A, cup B, boss C mm", the largest being the outside
diameter. For **Overall length**: "Magnets L mm + housing H mm".

**The torque card** (slice 3, BL-073; `gui::torque_card`, pure, no egui) sits between the
torque readouts and the drawing: a collapsible block headed "One step of each size control"
with a Why link to the note on how the coupling makes torque. Its first line is "The change in
the model's hot-low torque after one step of one control, everything else held.", then one row
per Size control in the controls' order, each a fixed step both ways through its slider's own
mapping: Outside diameter ±1 mm (at the Minimum OD), Overall length ±1 mm, Magnet face gap
±0.1 mm and Magnets per ring ±2. A side reads the change in hot-low torque in N·m and percent,
the difference of two `compute_all` runs (a row's two N·m at one precision), then what the step
costs: the change in Outside diameter or Overall length, each dashboard check whose level
flips ("Space claim: fails") and the change of a failing check's own quantity ("Cup wall: wall
needed +0.07 mm"); never the hot-low torque, which the row's number already is, nor the
temperature check, whose terms read the stored 3D fields. A side with no number says why:
"magnets do not fit", "no prediction", "at the end of its range" (the slider's own range, the
Minimum OD's included), "not a value the slider takes" (an odd count from a file), "the keyway
breaks through the hub", or on the Overall length row "not a step: the overlapping magnet
length does not change by the step" (a first move from a blank override with a shorter part or
a short cup cavity, BL-074). A step that changes the calibration factor adds "includes a
calibration change". At 8 magnets on the default ring the Outside diameter row reads -1 mm ->
-0.006 N·m (-0.4 %) and +1 mm -> -0.005 N·m (-0.3 %), the Overall length row -1 mm -> -0.15
N·m (-9.1 %). With Auto on, the card describes the design shown: Auto's row reads, on the
magnet length, "Overall length: Set by Auto; turn Auto off to edit", and one line says Auto sets
that size again after a change. A design shown whose magnets do not fit, with no torque
prediction, or whose pull-out or hot-low torque is not above zero gets one line in place of the
rows. The card starts open at 1280 x 800 and closed while the Equation panel is open, while the
centre region is narrower than 400 points, or where the open card would take the drawing under
its 160-point floor; a click on its heading or its arrow opens or closes it for the session,
and its body scrolls above the drawing's floor.
```

(R4) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F 'heading (to the note on how the coupling makes torque)' magcoupling-rs/README.md` prints 1 or more. Old:
```text
sits under Magnets per ring and Outside diameter on the Size tab (to the note on how the coupling makes torque)
```
New:
```text
sits under Magnets per ring and Outside diameter on the Size tab and on the torque card's heading (to the note on how the coupling makes torque)
```

(R5) Made stale by T2's code; written here unless T2 wrote it. Skip if `grep -c -F 'CARD_RANKING_WORDS' magcoupling-rs/README.md` prints 1 or more. Old:
```text
and neither word alone (BL-059: the one rule the note, guide and overlay bans call) |
```
New:
```text
and neither word alone (BL-059: the one rule the note, guide and overlay bans call); `CARD_RANKING_WORDS` (BL-073: strongest, biggest, best, most, lever, dominant, ranks), the torque card's ban list, matches a word whole and case-insensitive with an s added or taken off: "levers" and "Rank" match, "almost" and "bestow" do not |
```

(R6) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F -e 'heading open the torque note' -e 'heading (slice 3), open the torque note' magcoupling-rs/README.md` prints 1 or more. Old:
```text
the Size tab's Why links under Magnets per ring and Outside diameter open the torque note,
```
New:
```text
the Size tab's Why links under Magnets per ring and Outside diameter and on the torque card's heading open the torque note,
```

(R7) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F -e 'heading link are;' -e 'Magnets per ring link and the torque card' magcoupling-rs/README.md` prints 1 or more. Old:
```text
with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link is;
```
New:
```text
with an odd magnet count the Outside diameter control and its link are not drawn and the Magnets per ring link and the card's heading link are;
```

(R8) Made stale by T6's code; written here unless T6 wrote it. Skip if `grep -c -F -e 'the_card_sits_between_the_readouts_and_the_drawing' -e 'slice 3 (T6, headless' magcoupling-rs/README.md` prints 1 or more. Old:
```text
(`table_badges_on`, shared with the end-effect test) |
```
New:
```text
(`table_badges_on`, shared with the end-effect test); slice 3 (BL-073): the torque card sits between the Pull-out readout and the drawing, open on the first view at 1280 x 800 with every row drawn and the drawing at least 160 points (`the_card_sits_between_the_readouts_and_the_drawing`); it starts closed in a 930 x 800 window, where its header is no taller than the label it replaced and its Why link stays inside the centre region (`the_card_starts_closed_in_a_narrow_window_and_the_drawing_gains`), where the open card would take the drawing under its floor (`the_card_starts_closed_when_it_would_take_the_drawing_under_its_floor`) and while the Equation panel is open (`the_card_starts_closed_while_the_equation_panel_is_open`); a click on its heading or its arrow decides for the session, through design edits, a resize, the Equation panel, Reset all and a load (`the_user_s_open_or_close_wins_for_the_session`); no drag, exploration line, assumptions banner or side panel's growth opens or closes it (`the_card_never_opens_or_closes_under_a_drag`); its body scrolls above the drawing's floor (`the_card_body_scrolls_above_the_drawing_s_floor`) and its pre-measured height is the drawn one, a capped body the cap (`the_card_s_measured_height_is_the_drawn_height`); idle frames compute the card once, and an edit or a pause toggle once more, an Auto toggle that keeps the design shown none (`idle_frames_compute_the_card_once_and_an_edit_again`), and the two idle-frame tests run with the card open and closed |
```

(R9) Made stale by T6's code; written here unless T6 wrote it. Skip if `grep -c -F 'drawing_floor' magcoupling-rs/README.md` prints 1 or more. Old:
```text
then shrinking to scale (`drawing_height`);
```
New:
```text
then shrinking to scale (`drawing_height`); `drawing_floor` is that room, `MIN_DRAWING_HEIGHT` plus three rows of `list_row` (one row height for the list and the floor), which the Size tab's torque card keeps clear (slice 3);
```

(R10) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F 'check_terms_keeps_the_fix_tests_margins' magcoupling-rs/README.md` prints 1 or more. Old:
```text
the consequence numbers equal `compute_all` of the neighbour design and follow an input edit; neighbours outside 4 to 40 magnets are not shown;
```
New:
```text
the consequences follow an input edit and are computed once per design, Minimum OD and shaft-clamp pause (`the_cache_computes_once_per_design_and_minimum_and_follows_an_edit`; the tests of the N-2 and N+2 lines moved to `gui/torque_card.rs` with them in slice 3); `check_terms`, the counted checks' terms that the fix tests and the torque card read, keeps the fix tests' margins, the shaft joints' eight terms in one order whether judged or not, and one axis order in `SPACE_CLAIM_AXES`, `axis_reserves` and `SPACE_CLAIM_AXIS_SIZES` (`check_terms_keeps_the_fix_tests_margins`);
```

(R3) quotes `RING_SIZE_STAYS`, the card's heading, caption and side words, the flip and margin phrases, the held row
and (e)'s numbers as Step 3 checked them. The README's `src/gui/torque_card.rs` test row is Tasks 3, 4 and 5's (ruling
C-2); Step 11 checks its test names against the code.)

- [ ] **Step 7: `docs/ai/02-system.yaml`**

(S1) Made stale by T3's code; written here unless T3 wrote it. Skip if `grep -c -F 'src/gui/torque_card.rs' docs/ai/02-system.yaml` prints 1 or more. Old:
```text
src/gui/guidance.rs, src/gui/size_controls.rs,
```
New:
```text
src/gui/guidance.rs, src/gui/torque_card.rs, src/gui/size_controls.rs,
```

(S2) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F -e 'Auto line says what Auto does after a change' -e 'whose rule the torque card' docs/ai/02-system.yaml` prints 1 or more. Old:
```text
Under the controls, Consequences::of says what moves what: the neighbour magnet counts N-2 and N+2 inside the slider range (compute_all of the design with only coupling.npole changed; decision D-1: a neighbour that fails only the keyway still shows its torque, and one reads in words only when its blocks do not fit, its end effect is out of range or its torque is not finite), a first line that follows the Auto state (guidance::magnet_count_line: RING_SIZE_STAYS with Auto off, The ring radius stays the same when you change the magnet count; the cup OD below can still change.; AUTO_RESIZES_RING with Auto on the ring size; AUTO_RESETS_LENGTH with Auto on the magnet length; none with Auto on the magnet count, whose slider is locked), the smallest outside diameter
```
New:
```text
Under the controls, Consequences::of says what moves what: a first line under Magnets per ring that follows the Auto state (guidance::magnet_count_line: RING_SIZE_STAYS with Auto off or on the magnet length, The ring radius stays the same when you change the magnet count; the cup can still change size.; none with Auto on the ring size or the magnet count, where the torque card's Auto line says what Auto does after a change), the smallest outside diameter
```

(S3) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F -e 'one key changes, in three parts' -e 'one key in three parts, slice 3' docs/ai/02-system.yaml` prints 1 or more. Old:
```text
(the engine::size_controls breakdowns, reusing the expressions dimensions uses). A ConsequenceCache recomputes only when the inputs or the Minimum OD change; the FixCache likewise only when the inputs change
```
New:
```text
(the engine::size_controls breakdowns, reusing the expressions dimensions uses), and the torque card (slice 3, BL-073; Consequences.card from gui::torque_card::card, drawn between the torque readouts and the drawing): one fixed step each way of each Size control through its slider's own mapping (Outside diameter ±1 mm at the Minimum OD, Overall length ±1 mm, Magnet face gap ±0.1 mm, Magnets per ring ±2: the N-2 and N+2 lines that left Magnets per ring, in hot-low), each side the change in hot-low torque in N·m and percent, every number the difference of two compute_all runs (a row's two N·m at one precision), then the size change, the counted checks whose level flips and the change of a failing check's own quantity (guidance::check_terms; never the hot-low or the temperature check), or words where no number shows, by the order of tests of its spec's section 3.3 (the design shown by the readouts' own rule; then the slider's range within STEP_SLACK_MM, the mapping, fit, the keyway with decision D-1 for the count, the prediction, and on the length row the jump: words whenever the overlapping magnet length changes by anything other than the step). Auto's held row and its Auto line are drawn outside the cache (TorqueCard::lines), so an Auto toggle that keeps the design shown recomputes nothing. The card's open or closed default is decided only when its CardView changes (the Equation panel opened or closed, the screen resized, in whole points), never on a design change: open when the Equation panel is closed, the centre region is at least CARD_OPEN_MIN_WIDTH (400 pt) wide and the open card leaves the drawing its floor (geometry_view::drawing_floor); a click on its heading or arrow decides for the session (card_choice), and its body is capped above the drawing's floor. A ConsequenceCache recomputes only when its one key changes, in three parts: the design shown, the Minimum OD and the shaft-clamp pause; the FixCache only when the inputs or the pause change
```

(S4) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F 'heading (guidance::SIZE_TAB_NOTE' docs/ai/02-system.yaml` prints 1 or more. Old:
```text
sit under Magnets per ring and Outside diameter (guidance::SIZE_TAB_NOTE)
```
New:
```text
sit under Magnets per ring and Outside diameter and on the torque card's heading (guidance::SIZE_TAB_NOTE; slice 3)
```

(S5) No code task makes this line stale. Skip if `grep -c -F 'built on branch magcoupling/slice3-torque-card' docs/ai/02-system.yaml` prints 1 or more. Old:
```text
; next: the torque sensitivity card (slice 3), then M3 (live 3D fields)"
```
New:
```text
; the torque card (guidance slice 3, BL-073, owner decisions 1A-17A) built on branch magcoupling/slice3-torque-card: gui::torque_card and its card on the Size tab between the torque readouts and the drawing (one fixed step each way of each Size control, its change in hot-low torque and what the step costs, words where no number shows; the N-2 and N+2 lines moved into it from under Magnets per ring) and one Size-tab number format (one decimal of percent) on the card and the readout's gap; next: guided lessons, plot captions and drawing titles, then M3 (live 3D fields)"
```

(S6) No code task makes this line stale. Skip if `grep -c -F 'torque card (BL-073, gui::torque_card) shows' docs/ai/02-system.yaml` prints 1 or more. Old:
```text
are listed once, in docs/ai/04-memory.yaml."
```
New:
```text
are listed once, in docs/ai/04-memory.yaml."
  - "The magcoupler's torque card (BL-073, gui::torque_card) shows the model's numbers for one step of each Size control with everything else held, not a slope: its percents read the model's end factor, which BL-065 has not checked with magnetic steel behind the magnets (re-run the card's probe rows when BL-065 lands). The Overall length row reads words whenever the overlapping magnet length changes by anything other than the step (a first move from a blank override with a shorter part or a short cup cavity, BL-074), but the Outside diameter row has no such rule: where the shaft boss sets the Outside diameter (its inputs are under Detailed inputs), one step can lift the cup by several millimetres and still reads as a number (slice-3 spec risk R-13, in BL-074's evidence)."
```

(S3 leaves the timing parenthetical after it, "(both together measured at under 1.6 ms ...)", and `:106`'s drag-frame
count to Task 10 Step 8, which writes the measured figures. S6 is a new `known_limitations` entry after the shaft-joint
check's. `02-system.yaml` does not parse as YAML at 69fa505, near line 15, before this branch: leave that alone; Step 10
parses each line this task writes on its own.)

- [ ] **Step 8: `docs/ai/03-structure.yaml`**

(C1) Made stale by T2's code; written here unless T2 wrote it. Skip if `grep -c -F 'CARD_RANKING_WORDS' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
TORQUE_ORDER_WORDS (BL-064, spec 6.2 trap 2: the words that would order the coupling's torques, the one list the shaft-joint guide's and note's tests ban), pure std
```
New:
```text
TORQUE_ORDER_WORDS (BL-064, spec 6.2 trap 2: the words that would order the coupling's torques, the one list the shaft-joint guide's and note's tests ban), CARD_RANKING_WORDS and its whole-word matcher (BL-073: the torque card's ban list, strongest, biggest, best, most, lever, dominant and ranks, each matched whole and case-insensitive with an s added or taken off), pure std
```

(C2) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F 'one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
Consequences::of with magnet_lines, diameter_line and length_line (neighbours N-2 and N+2, the fit, the diameter and length breakdowns), Neighbour, NeighbourOutcome, ConsequenceCache, magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS, AUTO_RESIZES_RING, AUTO_RESETS_LENGTH, or none with Auto on the count)
```
New:
```text
Consequences::of (with clamps_paused) with card (the torque card, torque_card::card), magnet_lines, diameter_line and length_line (the fit, the diameter and length breakdowns), ConsequenceCache (one key in three parts: the design shown, the Minimum OD and the shaft-clamp pause), magnet_count_line (the first line under Magnets per ring, by Auto state: RING_SIZE_STAYS with Auto off or on the magnet length, none with Auto on the ring size or the count)
```

(C3) Made stale by T2's code; written here unless T2 wrote it. Skip if `grep -c -F 'check_terms' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults),
```
New:
```text
check_terms -> CheckTerm {name, margin, passes, quoted: Option<Quoted {path, name, decimals}>} (every counted check's terms, larger is better, from one table: the fix tests' terms() adds only the clamp's, and the torque card quotes the failing ones; joint_term names the shaft joints'), SPACE_CLAIM_AXIS_SIZES (each space-claim axis's name and dimension path in axis_reserves' order, read by the Problem's axis table and check_terms), default_reverse_fields (the temperature Problem's caveat while the reverse-field inputs hold their defaults),
```

(C5) Made stale by T2's code; written here unless T2 wrote it. Skip if `grep -c -F 'slider_minimum' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
variable_slider, change_text;
```
New:
```text
variable_slider, change_text, slider_minimum (the Outside diameter slider's minimum at the Minimum OD preference: one expression for the controls and the torque card);
```

(C6) Made stale by T6's code; written here unless T6 wrote it. Skip if `grep -c -F 'drawing_floor' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest);
```
New:
```text
geometry_view.rs (geometry_ui -> GeometryLayout (the drawing keeps 160 points while the callout list keeps three rows, then shrinks to scale; the list scrolls in the rest); drawing_floor (that room, MIN_DRAWING_HEIGHT plus MIN_LIST_ROWS rows of list_row, which the Size tab's torque card keeps clear) and list_row (the list's row height: one expression for geometry_ui and drawing_floor);
```

(C7) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F 'torque_card_ui' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
the welcome strip (WELCOME, welcome_dismissed), the Key design group's Auto status line (AUTO_STATUS_START) and run_sizing)
```
New:
```text
the welcome strip (WELCOME, welcome_dismissed), the Key design group's Auto status line (AUTO_STATUS_START) and run_sizing; the Size tab's torque card (slice 3: torque_card_ui, a CollapsingState with id magcoupling_torque_card where the label above the drawing was, its header row wrapped with the heading, which toggles it, and the Why link; card_choice, the user's own open or close, wins for the session; card_start holds the default and the CardView (the Equation panel open, the screen's width and height in whole points) it was decided for, decided again only when that view changes: open when the Equation panel is closed, the centre region is at least CARD_OPEN_MIN_WIDTH (400 pt) wide and card_open_height, the open card measured before it is drawn, leaves the drawing its drawing_floor; the body capped above that floor))
```

(C8) Made stale by T2's code; written here unless T2 wrote it. Skip if `grep -c -F 'no_back_iron_negative_calibration' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement),
```
New:
```text
and clamp_at_its_limit (the defaults with the clamp's capacity just covering its requirement), the torque card's designs (slice 3; each doc comment states its spec section 2.1 fact, which the_shared_card_designs_read_as_section_2_1_says pins): mixed_lengths and reverse_lengths (a B442 and a B842SH ring, each way round), no_back_iron, end_effect_at_a_step, odd_count, keyway_through, short_cavity, temperature_edge, forty_narrow_magnets, zero_torque, with_variation(variation) and no_back_iron_negative_calibration,
```

(C9) Made stale by T6's code; written here unless T6 wrote it. Skip if `grep -c -F 'drawing_rect' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
text_rect, text_rects, text_center, text_color,
```
New:
```text
text_rect, text_rects, text_center, text_color, drawing_rect (the painted rect of the geometry drawing, which the torque card's layout tests measure),
```

(C10) Made stale by T5's code; written here unless T5 wrote it. Skip if `grep -c -F 'since slice 3 the torque card' docs/ai/03-structure.yaml` prints 1 or more. Old:
```text
since slice 2 a hint line (plain_text::HINTS) under each control, above its consequence lines"
```
New:
```text
since slice 2 a hint line (plain_text::HINTS) under each control, above its consequence lines; since slice 3 the torque card (gui::torque_card) in the centre between the torque readouts and the drawing, and under Magnets per ring only the first line and the fit line (the N-2 and N+2 lines are the card's count row)"
```

(The format clauses, "format.rs gained format::fixed and format::signed_percent ..." and `torque_gap_text`'s, are T1's;
the `gui::torque_gap_text` re-export's mention is T1's (its Step 8 (f), plan deviation PD-4); the `torque_card.rs`
clause is T3's and T4's (ruling C-2). Step 11 checks these facts.)

- [ ] **Step 9: `docs/ai/04-memory.yaml` and the tracker's new top entry**

(M1) No code task makes this line stale. Skip if `grep -c -F 'Magcoupler torque card (guidance slice 3, BL-073' docs/ai/04-memory.yaml` prints 1 or more. Old:
```text
  - "Payload weights Track 2: before merge
```
New:
```text
  - "Magcoupler torque card (guidance slice 3, BL-073, branch magcoupling/slice3-torque-card; spec docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md, plan docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md; owner decisions 1A-17A and the controller's night-run rulings, .superpowers/sdd/2026-10-19-magcoupler-slice3/owner-decisions.md): built (T1-T9); the release checks (T10: whole-branch review, GATE PASS with gate 12, web build, the native timing and gui-smoke with the browser frame check) and then the owner's review of the diff come before the merge and the push (risk: physics). The Size tab's torque card (gui::torque_card) shows one fixed step each way of each Size control, its change in hot-low torque in N·m and percent and what it costs, or words where no number shows; the N-2 and N+2 lines left Magnets per ring for it, and the Size tab reads one decimal of percent. At 8 magnets on the default ring the Outside diameter row reads -1 mm -> -0.006 N·m (-0.4 %) and +1 mm -> -0.005 N·m (-0.3 %), the Overall length row -1 mm -> -0.15 N·m (-9.1 %). Open from it: BL-074 (the Overall length slider's first move from a blank override sets both rings to one length; the owner decides whether that is intended), BL-075 (the BL-064 fold-in's review notes with substance), and BL-065's dependency (the card's percents read the model's end factor: re-run its probe rows when BL-065 lands)."
  - "Payload weights Track 2: before merge
```

(M2) No code task makes this line stale. Skip if `grep -c -F 'then slice 3, the torque card, built on branch magcoupling/slice3-torque-card' docs/ai/04-memory.yaml` prints 1 or more. Old:
```text
BL-064, the shaft joints, shipped 2026-10-19): slice 3, a torque sensitivity card; then guided lessons,
```
New:
```text
BL-064, the shaft joints, shipped 2026-10-19; then slice 3, the torque card, built on branch magcoupling/slice3-torque-card): guided lessons,
```

(M3) No code task makes this line stale. Skip if `grep -c -F 'percents re-run when it lands' docs/ai/04-memory.yaml` prints 1 or more. Old:
```text
BL-065 (a 3D check of the end factor with magnetic steel behind the magnets; the owner to place it),
```
New:
```text
BL-065 (a 3D check of the end factor with magnetic steel behind the magnets; the owner to place it; the torque card's percents re-run when it lands),
```

(M4) No code task makes this line stale. Skip if `grep -c -F 'From slice 3: BL-074' docs/ai/04-memory.yaml` prints 1 or more. Old:
```text
BL-072 (the temperature verdict's cure margin reads the governing ring's onset: the owner-approved E24, to land with E21)."
```
New:
```text
BL-072 (the temperature verdict's cure margin reads the governing ring's onset: the owner-approved E24, to land with E21). From slice 3: BL-074 (the Overall length slider's first move from a blank override sets both rings to one length; the owner decides whether that is intended), BL-075 (the BL-064 fold-in's review notes with substance)."
```

Then `docs/ai/05-update-tracker.md`, the new top entry (quotes checked in Step 3: (e)'s numbers and (a)'s gap). Old
(unique):
```text
## 2026-10-17 — Magcoupler shaft joints: the keyed shrink fits that hold the rotors on their shafts (BL-064, branch magcoupling/bl064-shaft-joint)
```
New:
```markdown
## 2026-10-19 — Magcoupler guidance slice 3: the torque card, one step of each size control (BL-073, branch magcoupling/slice3-torque-card)
- Why: the owner went from 10 to 8 magnets, lowered the torque target and expected the coupling to shrink; it did not, and nothing on screen said why (slice 1's trigger; decision G-1 put a torque sensitivity card third). The card shows, for each of the four Size controls, what one fixed step each way does to the hot-low torque and what the step costs. At 8 magnets on the default ring the Outside diameter row reads -1 mm -> -0.006 N·m (-0.4 %) and +1 mm -> -0.005 N·m (-0.3 %), the Overall length row -1 mm -> -0.15 N·m (-9.1 %). Spec `docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md` (revision 5, committed dae024f after five review rounds); plan `docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md` (T1-T10); research and the owner's decisions (gitignored) `.superpowers/sdd/2026-10-19-magcoupler-slice3/`.
- Owner decisions (2026-10-19, rounds 1-5): **1A** the step card, live while dragging; **2A** rows for the four Size controls only (Outside diameter, Overall length, Magnet face gap, Magnets per ring), hot-low only; **3A** fixed steps of ±1 mm, ±1 mm, ±0.1 mm and ±2 magnets; **4A** between the torque readouts and the drawing in place of the fixed label, collapsible and open by default, closed while the Equation panel is open or the window is narrow, the drawing keeping its 160 pt floor; **5A** with Auto on, the card describes the design shown: the held row reads "set by Auto" and one line says Auto sizes again after a change; **6A** the controls' fixed order, each row stating its step, no sorting and no ranking words; **7A** the N±2 lines under Magnets per ring move into the card, in hot-low (this reverses slice 1's spec section 3.3 for those lines), while the first line and the smallest-diameter line stay; **8A** each side's change in Outside diameter and Overall length, the dashboard checks whose level flips and the change of each failing check's own quantity; **9A** words on the failing side only, none of the card's numbers where the design shown does not fit or has no prediction, full validity for the size and gap rows and decision D-1 for the count, and words where the overlapping magnet length jumps; **10B** one decimal of percent and N·m to the fewest decimals that keep the true sign, on the card and on the readout's gap (a change to the shipped whole-percent format; it closes slice 1's S19, percent precision, deferred in 04-memory at 5a1126a and dropped from it in 47bdcd7, so no doc held it at 69fa505); **11A** read-only rows, labels through label_for; **12A** no new note: a Why link to the torque note and one caption line, BL-065 left open; **13A** live recompute inside the one ConsequenceCache, a browser drag frame measured before release, an optimization only if it misses 60 fps; **14A** the full test set; **15A** BL-073, risk physics, released only after the STOP and the owner's OK; **16A** a step that changes the calibration factor shows the true number with a short note; **17A, 17B** BL-074 and BL-075 filed with this slice.
- The night run (2026-10-19, `owner-decisions.md`): the owner approved the spec ("Approved. Run throughout the night to implement /ultracode as needed"), and on that approval the controller ruled, each on the STOP sheet for the owner to reverse: owner question O-1 answered "widen" (the Overall length row reads words whenever the overlapping magnet length changes by anything other than the step, so a short cup cavity's +1 mm side reads words where 9A as written showed -6.2 %); O-2 answered "quote none" (no Temperature limits quantity on the card, its flips still shown, T-13); spec review round 3: the design shown decides first by the readouts' own rule, with one line in place of the rows where the magnets do not fit, the model has no prediction, or a pull-out or hot-low torque is not above zero (`ShownNoHotLow`), the card's open state keyed on the screen's size (`CardView`), the layout test's windows, and the shaft boss plateau recorded as risk R-13; round 4: the card's header row wraps, so its Why link stays inside a narrow centre region, and a count side whose calibration switch takes its torque to or below zero shows its number with the calibration note. The plan review's rulings are in the plan's head.
- Number format (T1): `gui/format.rs` gains `signed_fixed`, `signed_change`, `change_decimals` and `CHANGE_EXTRA_DECIMALS` (a change to the fewest decimals, from its unit's own up to two more, that keep its true sign and a digit that is not zero; past that cap its own decimals with its sign), `TORQUE_DECIMALS` (a torque change's own two decimals, one constant for the readout's gap and the card), and `fewest_decimals_apart` with `MOST_DECIMALS` moved there from `gui/guidance.rs`; `signed_percent` reads one decimal, an exact zero of either sign "+0.0 %", a value that rounds to zero keeping its sign; `torque_gap_text` follows the rule (at the example design "short 0.21 N·m (-8.6 %)", was "(-9 %)").
- Shared pieces (T2): `guidance::check_terms` (`CheckTerm`, `Quoted`; the production half of the fix tests' terms, which now call it), `SPACE_CLAIM_AXIS_SIZES` (the Problem's axis table and the quoted axes read one table), `size_controls::slider_minimum` (the Outside diameter slider's minimum, one expression for the controls and the card) and `size_controls::SliderMapping` (each Size slider's own mapping of a value to a design, which the controls apply on an edit and the card on each step), `CARD_RANKING_WORDS` and its whole-word matcher in `engine/explain/plain.rs`, and the card's shared test designs in `gui/test_support.rs`.
- The card (T3, T4): `gui/torque_card.rs`, pure: `card(inputs, results, minimum_diameter_mm, clamps_paused) -> TorqueCard` (`ShownDoesNotFit`, `ShownNoPrediction`, `ShownNoHotLow` or `Rows`), each Size control stepped through its slider's own mapping and tested in the order of the spec's section 3.3, with `STEP_SLACK_MM` the one slack of the range test and the jump rule; `side_words` and `count_step` moved there from the neighbour lines (moved, not copied); flips over the counted guides and the failing checks' quantities through `check_terms`; every text a named constant set by the G-3 review; a row's two N·m at one precision (T-14); `card_designs()`, the one table of the spec's section 2 designs the card's tests loop over, T-14's one-flip design and R-13's boss plateau among them; one rule for the range Auto searches, `gui::sizing::sizing_range`, which the panel and that table read.
- Wiring and layout (T5, T6): `Consequences.card` in place of the neighbour lines, the cache's one key in three parts (the design shown, the Minimum OD, the shaft-clamp pause); `RING_SIZE_STAYS` reworded, `AUTO_RESIZES_RING` and `AUTO_RESETS_LENGTH` removed, `magnet_count_line` per the spec's section 4.3; the card drawn in the Size view in place of the label above the drawing (`torque_card_ui`, a `CollapsingState` whose wrapped header row holds the heading, which toggles it, and the Why link), its default decided only when the `CardView` changes and the user's own click winning for the session; `geometry_view::drawing_floor` and `list_row`; `test_support::drawing_rect`.
- Reviews and smoke (T7, T8): every card string set by the G-3 physics review (`docs/analyses/2026-10-14-magcoupler-help-review.md`, `## Slice 3 torque card`); gui-smoke gains the card step (the texts quoted whole, held to the code by `app.rs`'s smoke test) and a browser drag-frame timing on the gap slider that keeps only the frames the app draws (`FRAME_TIMER`), which gates `passed` only with `frame_check`.
- Docs and backlog (T9): the README (the status, the module list, the Size-tab lines and the card, the Why links, the test rows), docs/ai 02-05; BL-073 (`in_progress` until the owner's OK), BL-074 (the Overall length slider's first move from a blank override sets both rings to one length, with the shaft boss plateau of risk R-13 in its evidence), BL-075 (the BL-064 fold-in's review notes with substance) and one line in BL-065's notes (the card's percents are a consumer of its fix). Not covered by the card: the diameter row's boss plateau (R-13) and the reverse pair's trimmed inner ring (BL-074), both recorded; the card's percents read the model's end factor (BL-065).

## 2026-10-17 — Magcoupler shaft joints: the keyed shrink fits that hold the rotors on their shafts (BL-064, branch magcoupling/bl064-shaft-joint)
```

- [ ] **Step 10: Check the YAML and the item shapes**

Run:
```bash
python - <<'EOF'
import yaml
b = yaml.safe_load(open('docs/ai/backlog.yaml', encoding='utf-8'))
ids = [x['id'] for x in b]
assert ids[:21] == ['BL-0%d' % n for n in range(56, 76)] + ['BL-055'], ids[:21]
keys = ('id', 'title', 'dimension', 'risk', 'evidence', 'acceptance', 'priority', 'status', 'notes')
assert all(tuple(x) == keys for x in b), [x['id'] for x in b if tuple(x) != keys]
item = {x['id']: x for x in b}
assert [x['id'] for x in b if x['status'] == 'in_progress'] == ['BL-073']
assert item['BL-073']['notes'].endswith('Deploy authorization is asked at the STOP.')
assert (item['BL-073']['dimension'], item['BL-073']['risk'], item['BL-073']['priority']) == ('gui', 'physics', 2)
assert (item['BL-074']['dimension'], item['BL-074']['risk'], item['BL-074']['priority'], item['BL-074']['status']) == ('gui', 'physics', 3, 'open')
assert (item['BL-075']['dimension'], item['BL-075']['risk'], item['BL-075']['priority'], item['BL-075']['status']) == ('quality', 'physics', 3, 'open')
assert 'R-13' in item['BL-074']['notes'] and 'boss' in item['BL-074']['evidence']
assert "torque card's percents are a consumer" in item['BL-065']['notes']
m = yaml.safe_load(open('docs/ai/04-memory.yaml', encoding='utf-8'))
assert any(i.startswith('Magcoupler torque card (guidance slice 3, BL-073') for i in m['active_issues'])
assert any('then slice 3, the torque card, built on branch magcoupling/slice3-torque-card' in q for q in m['open_questions'])
assert not any('slice 3, a torque sensitivity card' in q for q in m['open_questions'])
yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8'))
lines = open('docs/ai/02-system.yaml', encoding='utf-8').read().splitlines()
for marker in ('src/gui/torque_card.rs', 'one key in three parts, slice 3', "and on the torque card's heading (guidance::SIZE_TAB_NOTE",
               'built on branch magcoupling/slice3-torque-card', "The magcoupler's torque card (BL-073, gui::torque_card)"):
    hit = [line for line in lines if marker in line]
    assert len(hit) == 1, (marker, len(hit))
    yaml.safe_load('k:\n' + hit[0])
print(len(b), len(m['active_issues']), len(m['open_questions']))
EOF
```
Expected: `75 6 66` (72 backlog items at 69fa505 plus BL-073 to BL-075; 5 active issues plus slice 3's; 66 open
questions, NEXT edited in place). An `AssertionError` names the broken item: fix the edit, run again. (Measured after
Task 8 with this task's edits: `75 6 66`, each of the five markers once in `02-system.yaml`.)

- [ ] **Step 11: The sweep again (green), the new facts, the docs tests and the suites**

Run Step 4's two commands again. Expected: the first prints exactly the two T10 lines, 02-system `:106` "(about 30
compute_all calls)" and `:116` "under 1.6 ms per changed design" (Task 10 Step 8 writes the measured figures), and
nothing after `---` (measured after Task 8 with this task's edits).

Then the new facts (each written by the task named):
```bash
grep -c 'src/gui/torque_card.rs' docs/ai/02-system.yaml
grep -c 'torque_card.rs (' docs/ai/03-structure.yaml
grep -c '`torque_card.rs` (' magcoupling-rs/README.md
grep -c '^| `src/gui/torque_card.rs` (feature `gui`) |' magcoupling-rs/README.md
grep -c -E 'signed_change|change_decimals' docs/ai/03-structure.yaml magcoupling-rs/README.md
grep -c -F '(-P.P %)' magcoupling-rs/README.md
grep -c 'gui::torque_gap_text' docs/ai/03-structure.yaml
grep -c 'CARD_RANKING_WORDS' docs/ai/03-structure.yaml magcoupling-rs/README.md
grep -c 'check_terms' docs/ai/03-structure.yaml magcoupling-rs/README.md
grep -c 'slider_minimum' docs/ai/03-structure.yaml magcoupling-rs/README.md
grep -c 'drawing_floor' docs/ai/03-structure.yaml magcoupling-rs/README.md
grep -c 'BL-07[345]' docs/ai/backlog.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md
for t in the_card_sits_between_the_readouts_and_the_drawing the_card_starts_closed_in_a_narrow_window_and_the_drawing_gains the_card_starts_closed_when_it_would_take_the_drawing_under_its_floor the_card_starts_closed_while_the_equation_panel_is_open the_user_s_open_or_close_wins_for_the_session the_card_never_opens_or_closes_under_a_drag the_card_body_scrolls_above_the_drawing_s_floor the_card_s_measured_height_is_the_drawn_height idle_frames_compute_the_card_once_and_an_edit_again the_cache_computes_once_per_design_and_minimum_and_follows_an_edit check_terms_keeps_the_fix_tests_margins each_number_is_the_difference_of_two_compute_all_runs the_card_reads_the_spec_s_numbers the_card_s_lines_read_as_drafted the_words_follow_the_order_of_the_tests the_design_shown_decides_first the_size_rows_follow_the_size_controls the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum the_flips_and_the_margins_name_the_checks_that_move the_card_names_the_clamp_only_while_unpaused the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s size_changes_and_margins_are_drawn_only_when_they_show no_card_number_reads_zero_or_disagrees_with_its_percent the_card_texts_never_rank_the_controls every_card_text_has_its_glyphs the_card_s_strings_are_constants_and_its_numbers_come_from_results the_shared_card_designs_read_as_section_2_1_says; do printf '%s %s\n' "$t" "$(grep -rl "fn $t()" magcoupling-rs/src | tr '\n' ' ')"; done
```
Expected: `1`; at least `1`; `1`; `1`; at least `1` in each file (T1's format clauses: `03-structure.yaml`'s
format.rs entry and the README's `src/gui/format.rs` test row); at least `1` (T1's README `:203`); at least `1` (the
re-export's mention, T1's); at least `1` in each; at least `1` in each; at least `1` in each; at
least `1` in each; at least `1` in each file; and every test name followed by the file that holds it (the nine
panel tests from `the_card_sits_between_the_readouts_and_the_drawing` to `idle_frames_compute_the_card_once_and_an_edit_again` by
`magcoupling-rs/src/gui/panel.rs`, `the_cache_computes_once_per_design_and_minimum_and_follows_an_edit` and
`check_terms_keeps_the_fix_tests_margins` by `magcoupling-rs/src/gui/guidance.rs`,
`the_shared_card_designs_read_as_section_2_1_says` by `magcoupling-rs/src/gui/test_support.rs`, and the other fifteen by
`magcoupling-rs/src/gui/torque_card.rs`). Measured after Task 8 with this task's edits: `1`; `1`; `1`; `1`; `1` in each
file; `1`; `1`; `1` and `2` (03-structure, README) for each of `CARD_RANKING_WORDS`, `check_terms`, `slider_minimum` and
`drawing_floor`; `3`, `2` and `3` for `BL-07[345]` (backlog, 04-memory, tracker).
A `0`: the owning task's docs step is missing: apply it as the Rules say, then run this step again. A test name
followed by nothing: the README names a test the code does not have: if the code has it under another name, write that
name in the README row (and say so in the commit body); if it has no such test, STOP: the owning task's test is
missing.

Then the docs-count tests, the suites, the lints and the format check:
```bash
cargo test --manifest-path magcoupling-rs/Cargo.toml --test explain the_docs_count_the_notes_and_the_steps
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --test explain the_docs_count
cargo test --manifest-path magcoupling-rs/Cargo.toml 2>&1 | grep -E '^test result|FAILED|panicked'
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app 2>&1 | grep -E '^test result|FAILED|panicked'
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings
cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check
git status --short
```
Expected: `test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 63 filtered out` (without features) and
`test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 63 filtered out` (`--features app`); every
`test result:` line of both suites `ok` with `0 failed`, and no `FAILED` or `panicked` line; Task 9 changes no code, so
every count is the one Task 8's suite step recorded (measured: `lib` 234 passed and 1 ignored, 971 passed and 6
ignored with `--features app`; `assumptions` 13, `deviations` 54, `differential` 19,
`explain` 63 passed and 1 ignored, 64 and 1 with `--features app`; `gearbox` 13, `grades` 12, `material_library` 4,
`material_links` 14, `parity` 4, `python_schema` 7, `robustness` 13, `schema` 11, `shaft_joint` 40, `size_controls`
38, `sizing` 45, `static_data` 8; the doc-tests 1 passed and 1 ignored; the two app binaries 0); both clippy runs end
without a warning or an error; `cargo fmt --check` prints nothing; `git status --short` shows ` M`
lines only for the five files this task edits (`docs/ai/03-structure.yaml` is unchanged: every C edit is skipped or
deleted; and any a Step 4 or Step 11 fallback edited) and
`?? docs/analyses/2026-05-28-press-4bar-analysis.md`.

- [ ] **Step 12: Commit**

Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T9.txt` with the Write tool, exactly:
```text
docs(magcoupler): slice 3 docs pass: the tracker entry, 04-memory, BL-073 to BL-075 and BL-065's line, the README and docs/ai

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
If Step 3 substituted a shipped text, an edit of Steps 6-9 was skipped or reworded per the Rules, or Step 4 or Step 11
made you apply an earlier task's docs step, add one body line per item before the trailers (a blank line before the
body and after it), for example `Applies T5's code-comment step for SIZE_TAB_NOTE's doc, which T5's commit missed.` or
`R4 skipped: T5 wrote the card's Why link into the README line.` (Measured after Task 8: every skipped edit finds
its skip-if string and no shipped text is substituted, so the message has no body line.) Then:
```bash
git add docs/ai/backlog.yaml docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md magcoupling-rs/README.md
git status --short
```
Also stage, by name, every file a Step 4 or Step 11 fallback edited (the body names each). Expected `git status
--short`: `M ` lines for exactly the staged files and `?? docs/analyses/2026-05-28-press-4bar-analysis.md` (never staged).
Then:
```bash
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T9.txt
git log --oneline -1
git status --short
```
Expected: one new commit of 5 files (`docs/ai/03-structure.yaml` is unchanged: every C edit is skipped or deleted, and
its `git add` stages nothing; more only with a fallback) whose subject starts `docs(magcoupler): slice 3 docs pass`; `git status --short` then prints only the owner's untracked file.

**Existing tests Task 9 moves:** none (docs only; the docs-count tests pass unedited).

---

### Task 10: Release: preconditions, whole-branch review and one fix wave, the gate with gate 12, web build, the native timing and the browser frame check, the measured figures, the screen dump and the STOP sheet; after the OK: BL-073 fixed, merge, push, deploy check, live smoke

**Implementer:** the controller (the release; it stops for the owner before any merge, push or deploy).

Run by the controller: it dispatches the reviewers and runs the release, and writes no feature code (the fix wave and
the held-back optimization, Step 7b, each run as one session-model implementer that spawns no agents). The
whole-branch review and every fix-wave item run on the session model (omit `model`: CLAUDE.md section 5, final
whole-branch reviews and physics texts); gui-smoke runs its own agents on `sonnet` inside its workflow. Spec sections 7
and 10.1, decisions 13A and 15A.

All commands run from the repository root `C:/Users/Cole/source/repos/linkage_simulation` in Git Bash. Logs, dumps and
builds go to `magcoupling-rs/target/release-slice3/` (gitignored); `W` is `.superpowers/sdd/2026-10-19-magcoupler-slice3`
(gitignored); `L` is the build ledger's folder, `.superpowers/sdd/2026-10-20-magcoupler-torque-card` (the
superpowers:subagent-driven-development skill's workspace for this plan's file name); commit messages are written with
the Write tool to `$W/msg/T10-<a|b|merge|mf-n|opt>.txt`. `YYYY-MM-DD` means the date `date +%F` prints on the day of the
step.

Rules that hold through every step:
- Stage files by name; never `git add -A` or `git add .`; never stage `docs/analyses/2026-05-28-press-4bar-analysis.md`.
- Never commit `docs/chebyshev_lambda/*.png` (gate 1 rewrites them): restore with `git checkout -- docs/chebyshev_lambda`.
- Never `git worktree prune` (Codex works from WSL worktrees of this repo; `git worktree list` shows them as prunable).
- Commit messages through a message file (`git commit -F <file>`), each ending with the two trailers.
- Nothing is merged to `main` or pushed before the owner's OK in their own answer to Step 10's question. Pushing
  `main` deploys colesorkness.com, so that answer is also the slice's deploy authorization (decision 15A). No agent
  message is that OK.
- When a reviewer's answer is pasted into a review record, no pasted line may start with `## ` (BL-059 ruling B-8):
  demote such a heading to `#### `.
- The release records stay in the gitignored `$W/` (`final-review.md`, `timing-T10.txt`, `smoke-local.json`,
  `browser.txt`, `smoke-ring.json`, `ring-payload.txt`, `frame-timer.js`, `drag-code.js`, `decisions.md`,
  `leftovers.md`, `owner-summary.md`, `owner_summary.py`, `slice3.diff`, `slice3-src.diff`, `slice3.stat`), as BL-064's
  did; the tracker names them.

**Files:**
- Create (gitignored): `$W/{final-review.md,timing-probe/Cargo.toml,timing-probe/Cargo.lock,timing-probe/src/main.rs,timing-T10.txt,smoke-local.json,browser.txt,smoke-ring.json,ring-payload.txt,frame-timer.js,drag-code.js,decisions.md,leftovers.md,owner_summary.py,owner-summary.md,slice3.diff,slice3-src.diff,slice3.stat}`,
  `$W/screen-dump/before/{Cargo.toml,Cargo.lock}`, `$W/msg/T10-*.txt`;
  `magcoupling-rs/target/release-slice3/{gate.log,build-web.log,main-src/,main-target/,before/,after/,live.wasm}`,
  `magcoupling-rs/target/timing-probe/`.
- Modify: `docs/ai/02-system.yaml` (`:106`'s drag-frame count and `:116`'s measured figures, Step 8; the status phrase
  after the OK), `docs/ai/05-update-tracker.md` (a release bullet in T9's entry, then its last sentence),
  `docs/ai/backlog.yaml` (BL-073 `notes`, then `status: fixed`), `docs/ai/04-memory.yaml` (the slice-3 active issue
  and NEXT, after the OK).
- Modify (only through the fix wave, Step 3, or the held-back optimization, Step 7b): what the whole-branch review's
  must-fix items name; for Step 7b `magcoupling-rs/src/engine/size_controls.rs`, `magcoupling-rs/tests/size_controls.rs`,
  `magcoupling-rs/src/gui/torque_card.rs`, `magcoupling-rs/README.md` and `docs/ai/03-structure.yaml`.
- Modify (outside the repo, after the deploy): the auto-memory
  `C:/Users/Cole/.claude/projects/C--Users-Cole-source-repos-linkage-simulation/memory/project_magcoupler_guidance_overhaul.md`
  and its index line in `MEMORY.md` there.

**Interfaces:**
- Consumes: T1-T9 committed; T9's exact strings (its Interfaces), its screen dump (`$W/screen-dump/`) and
  `$W/check_shipped.py`; the review dumps (ignored tests `guidance_strings_for_review`, which T5 extends with the
  card's lines, `overlay_strings_for_review`, `notes_for_review`); the oracle venv
  `reference/magcoupling-py/.venv/Scripts/python.exe`; `linkage-sim-rs/scripts/{gate.sh,build_web.sh,serve_web.sh}`;
  the `gui-smoke` skill (`.claude/workflows/gui-smoke.js` as T8 leaves it: `ARGS.frame_check`, `FRAME_TIMER`, and in its `magcoupling`
  result `torque_card`, `drag_moved`, `drag_frames`, `drag_frame_ms_median`, `drag_frame_ms_p95`, `drag_frame_ms_max`
  and `drag_frames_within_60fps`); the public API `gui::guidance::{ConsequenceCache, FixCache}` (`get` with
  `clamps_paused`, spec 3.2), `gui::torque_card::card`, `engine::size_controls::{DiameterFit, FIT_TOLERANCE_MM,
  dimensions, dimensions_with_minimum_diameter, ring_radius_range, smallest_fitting_diameter}`, `engine::sizing::{FreeVariable,
  SizingOutcome, solve_in}`, `gui::sizing::{SizingMode, SizingState, sized}` and `gui::session::{Design, encode_share_payload}`
  (Step 6's probe, which also prints the ring design's share link: performance review 4.2); `gh` (logged in; the deploy workflow file
  is `deploy-web.yml`, "Deploy Web to Vercel", run on every push to `main`); the Playwright MCP tools; SendUserFile and
  AskUserQuestion.
- Produces: the release-check records and the measured figures in `02-system.yaml`; after the OK, BL-073 `fixed`, the
  merge commit on `main`, the push, a checked deploy and live smoke, the local branch deleted, the memory updated.

**Before the deploy** (checked 2026-10-20 on the live file, the deploy of 69fa505): the live
`magcoupling-web_bg.wasm` holds `main`'s Auto ring-size line ("Auto resizes the ring when you change the magnet count;
the lines below keep the present ring radius.") once and the card's draft heading ("One step of each size control")
nowhere. Steps 5 and 12 count both: a bundle with the card holds the heading and not the line (the slice removes
`AUTO_RESIZES_RING`, spec 4.3).

- [ ] **Step 1: Preconditions and the release invariants**

Run:
```bash
git branch --show-current
git status --short
git log --oneline dae024f..HEAD | wc -l
git log --oneline -1
grep -rn -A2 '#\[ignore' magcoupling-rs/src magcoupling-rs/tests | grep -o 'fn [a-z_0-9]*' | sort -u
cargo fmt --manifest-path magcoupling-rs/Cargo.toml --check
mkdir -p magcoupling-rs/target/release-slice3
BASE="$(git merge-base main HEAD)"; echo "$BASE"; git rev-parse 69fa505
git diff "$BASE" HEAD --stat -- reference magcoupling-rs/tests/data magcoupling-rs/tests/python_schema.rs magcoupling-rs/tests/parity.rs magcoupling-rs/tests/differential.rs magcoupling-rs/tests/deviations.rs magcoupling-rs/src/engine/deviations.rs
echo ---
git diff "$BASE" HEAD --stat -- magcoupling-rs/src/engine magcoupling-rs/tests
echo ---
git diff "$BASE" HEAD --stat -- magcoupling-rs/src/engine/sizing.rs magcoupling-rs/src/gui/sizing.rs magcoupling-rs/src/engine/clamps.rs
grep -c '^pub const PAUSED: bool = true;' magcoupling-rs/src/engine/clamps.rs
echo ---
ls .superpowers/sdd/2026-10-19-magcoupler-slice3/screen-dump/after/Cargo.toml .superpowers/sdd/2026-10-19-magcoupler-slice3/check_shipped.py
```
Expected:
- `magcoupling/slice3-torque-card`; only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`;
- at least `10` commits since dae024f (the plan's commit and the nine commit steps of T1-T9, each with the subject that
  step names; a later plan edit and T7's review-fix commits add to them); the last commit is T9's
  (`docs(magcoupler): slice 3 docs pass ...`);
- the ignored functions are `fn guidance_strings_for_review`, `fn help_for_review`, `fn notes_before_snapshot`,
  `fn notes_for_review`, `fn notes_signoff_sheet`, `fn overlay_strings_for_review`, `fn review_sheet` (as at 69fa505),
  plus only a review tool a plan step adds by name (check each extra with
  `grep -c "fn <name>" docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md`, at least `1`);
- `cargo fmt --check` prints nothing; the merge base equals 69fa505's full hash (if `main` moved, the base stays
  69fa505 until Step 11 merges);
- the first `git diff --stat` prints nothing (no Python-locked row, oracle file, data file or deviation changed; the
  input schema `tests/data/input_schema.json` too: no input is added, so gate 12's parity and the schema test hold);
- the second lists `magcoupling-rs/src/engine/explain/plain.rs` (T2's ban list) and nothing else: slice 3 changes no
  engine physics, input or result (spec 10.2);
- the third lists `magcoupling-rs/src/gui/sizing.rs` alone, 18 insertions and 1 deletion (T3's `sizing_range`, the range
  Auto searches, moved there from the panel with its import: critique A-13; Auto's solve and the clamp section
  unchanged), then `1` (the section still paused);
- both of T9's files are listed.

Anything else: STOP and finish the owning task first.

- [ ] **Step 2: Whole-branch review (fresh context, session model)**

Dispatch one reviewer on the session model (omit `model`) with this prompt, verbatim:

> You are the whole-branch reviewer for slice 3 of the magnetic coupling calculator's guidance overhaul, the torque
> card (BL-073, risk: physics; branch `magcoupling/slice3-torque-card` of C:/Users/Cole/source/repos/linkage_simulation,
> crate `magcoupling-rs`). Read the spec `docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md` (revision
> 5, binding: the owner decisions 1A-17A of section 1.1; the technical decisions T-1 to T-14 of section 1.2; the
> answered questions O-1, widen, and O-2, quote none, of section 1.3; sections 2 to 9; the risks of section 10.3), the
> owner decisions and the controller's rulings `.superpowers/sdd/2026-10-19-magcoupler-slice3/owner-decisions.md`
> (binding, every section), the plan `docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md` (the rulings, plan
> deviations and plan-review answers in its head are accepted changes to the spec), the G-3 record
> `docs/analyses/2026-10-14-magcoupler-help-review.md` (`## Slice 3 torque card`), then the whole diff `git diff
> main...magcoupling/slice3-torque-card`. Read-only: change no file and run no command that changes the repository
> (you may run `cargo test` with `CARGO_TARGET_DIR` set to a folder outside the repository).
> Check, each with file:line evidence:
> (1) every owner decision 1A-17A and every ruling of owner-decisions.md is implemented as the spec says, or the plan
> records a deviation the controller accepted; in particular 7A (the N±2 lines left Magnets per ring and are the card's
> count row, in hot-low; the first line and the smallest-diameter line stay), 9A widened by O-1 (`|stepped active -
> shown active - sign x step| > STEP_SLACK_MM` reads `Jump`, one `JUMP` text), 10B (one percent rule and one formatter on
> the card and in `torque_gap_text`), 12A (no new note; the heading's Why link to `SIZE_TAB_NOTE`; no caveat line), 13A
> (the one `ConsequenceCache` key, now in three parts; one miss per drag frame), 16A (`calibration_changed` from
> `model.f_cal`, the note right after the number, a side's number shown whatever its sign), the round-3 rulings (test 0
> on the readouts' own rule through `size_controls::predicts`; `ShownNoHotLow` for a pull-out or hot-low not above zero;
> the `CardView` keyed on the Equation panel and the screen's size in whole points; R-13 recorded, no diameter jump
> rule) and the round-4 rulings (the header row in `horizontal_wrapped` inside `show_header`; a count side below zero a
> number with the note);
> (2) T-1 to T-14, each as the spec states it;
> (3) `gui/torque_card.rs` against sections 3.1-3.6: the public types and signatures under their binding names (among
> them `STEP_SLACK_MM`, `count_step`, `side_words`, `ShownNoHotLow`, `held_line`, `card`, `lines`, `rows`); the order of
> tests 0-8; each side stepped through its slider's own mapping, `SliderMapping`, which the controls apply too (plan
> review 2.3: `slider_minimum` with `resize_diameter_with_minimum`, never `resize_diameter`; `resize_length`;
> `InputSet::set`, the count as a whole number, plan deviation PD-1; `count_step` with checked arithmetic); the slack used in
> steps 2 and 7 only and named once; no `unwrap` or `expect` on a mapping's result; the flips over
> `counted_guides(clamps_paused)`; the margins only through `check_terms` and `shows`, never the hot-low or the
> temperature check, each unit the result's own through `with_unit`; every type `Clone, Debug, PartialEq` and no NaN
> standing for "no value"; the neighbour code moved, not copied (no second copy of the validity rule or the count step
> left in `gui/guidance.rs`; `prediction` gone);
> (4) the wiring of sections 3.2, 3.4 and 3.5 and the GUI of 6.1-6.2: `Consequences.card`; `Consequences::of` and
> `ConsequenceCache::get` take `clamps_paused`, both panel call sites pass `self.clamps_paused`, the cache's key holds
> it; the card draws the design shown that `view_ui` builds once after the Required edit (`self.shown_inputs()`) and
> hands to the card and the drawing (plan deviation PD-6), with its own `compute_all`; the Auto marking outside the cache;
> the header row (the heading a `Label` with a click sense that toggles before `.body(..)`, and `why_note_link(ui,
> SIZE_TAB_NOTE)`); the open-or-closed rules 1-3 with `CARD_OPEN_MIN_WIDTH` 400 pt, decided only when the `CardView`
> changes, `card_choice` winning for the session; the capped `ScrollArea` body; `drawing_floor` and `list_row`, one
> expression with `geometry_ui`'s; the label of the old `gui/panel.rs:936` removed; the exploration lines built from
> `EXPLORATION_ONLY`, `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH`;
> (5) section 5's number format: `signed_percent` at one decimal, an exact zero of either sign "+0.0 %", a value that
> rounds to zero keeping its sign; `change_decimals` capped at `from + CHANGE_EXTRA_DECIMALS` through
> `fewest_decimals_apart`; `signed_fixed`; a row's two N·m at one precision (T-14); `torque_gap_text` with no
> `replacen` re-signing and its doc comment on the new rule;
> (6) every on-screen text the branch adds or changes (section 4.1's constants, section 4.3's `RING_SIZE_STAYS`, the
> templates and the phrases as rendered at the designs of section 2) against the traps of section 4.4: no ranking or
> comparative word (`CARD_RANKING_WORDS` by its whole-word rule), no `TORQUE_ORDER_WORDS`, nothing that says the
> coupling protects the gearbox, no per-mm or slope wording, the calibration change never explained as physics, no "no
> effect" or "free", no assumption named as a thing to change, no "clamp" in any case while the clamp section is paused,
> plain words and glyphs; each text equals its `## Slice 3 torque card` record (list any text edited after its review);
> every number from the results;
> (7) every test of section 8 exists under its spec name and asserts values computed from `compute_all` or named
> constants (the spec's deliberate literal pins of tests 2, 12, 13 and section 5's gap tests excepted); the section 6.3
> moves are done as the plan's task tables say; no existing assertion weakened and no test newly ignored; the layout
> tests set the pause before the deciding frame and pin no figure measured in the app's style;
> (8) unchanged where the spec says so: no engine input, result or physics (`git diff main...HEAD --
> magcoupling-rs/src/engine` touches only `engine/explain/plain.rs`'s ban list, and `engine/size_controls.rs` only if
> the plan's held-back optimization step ran), `tests/data/input_schema.json`, the Python-locked rows, `reference/`, the
> differential and deviation files; `engine::clamps::PAUSED` still `true` and the paused section's texts unedited;
> Auto's solve unchanged;
> (9) the docs (`magcoupling-rs/README.md`, `docs/ai/02-system.yaml` to `05-update-tracker.md`, BL-065 and BL-073 to
> BL-075 in `docs/ai/backlog.yaml`) describe the code as it is: no stale text (the N±2 lines under Magnets per ring, the
> two removed Auto lines, a whole percent, the cache's old key), every quoted text the shipped one; 02-system's "under 1.6
> ms" and "about 30 compute_all calls" are the release's to replace with measured figures and may still stand;
> (10) gui-smoke (`.claude/workflows/gui-smoke.js`): the card step at its fixed place quotes the shipped texts whole;
> the frame-time step reads only the drag's frames, timing only the frames the app draws (`FRAME_TIMER`, plan review
> 4.1), and gates `passed` only with `frame_check`; the console collection and the browser's close come after both
> steps; `src/app.rs`'s `the_smoke_test_share_link_opens_its_design` asserts the four texts, the three row starts and
> the frame timer.
> Return PASS, or a numbered list of must-fix items (file:line, the defect, the exact replacement text or the test to
> add) and a separate list of suggestions. A must-fix that changes an on-screen text says so.

Save the reviewer's full answer to `.superpowers/sdd/2026-10-19-magcoupler-slice3/final-review.md`.

- [ ] **Step 3: One fix wave, then the re-review (only if Step 2 returned must-fix items)**

For each must-fix item, in order, one commit each, on the session model (an implementer that does not spawn agents):
1. Write the failing test the item names (or one that shows the defect), run it, see it fail.
2. Apply the fix.
3. A fix that changes an on-screen text (a card constant, `RING_SIZE_STAYS`, a template or phrase): one fresh
   session-model G-3 reviewer on the changed strings, rendered at the designs of spec section 2 (dump them with
   `cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture`
   and with T9's screen dump,
   `CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/screen-dump" cargo run --quiet --manifest-path .superpowers/sdd/2026-10-19-magcoupler-slice3/screen-dump/after/Cargo.toml`),
   with the spec's traps (section 4.4) and the G-3 brief of Task 7; its must-fix items applied; its report appended
   under `## Slice 3 torque card` of `docs/analyses/2026-10-14-magcoupler-help-review.md` as
   `### Re-review after the whole-branch review (YYYY-MM-DD)`. A changed text that gui-smoke's card step quotes
   (`CARD_HEADING`, `CAPTION`, `AUTO_HOLDS_LENGTH`, `held_line`) is changed in `.claude/workflows/gui-smoke.js` in the
   same commit (`app.rs`'s `the_smoke_test_share_link_opens_its_design` fails until it is). A docs line that quotes the
   changed text: re-run T9's Step 2 dump and its Step 3 check with the new text in `QUOTES`, and write the shipped text
   in the docs line.
4. Run the touched suites (`cargo test --manifest-path magcoupling-rs/Cargo.toml --features app <filter>` and the same
   without `--features app`), both clippy runs
   (`cargo clippy --manifest-path magcoupling-rs/Cargo.toml --all-targets --features app -- -D warnings` and the same
   without `--features app`) and `cargo fmt --manifest-path magcoupling-rs/Cargo.toml`, then write
   `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-mf-<n>.txt`:
```text
fix(magcoupler): <what, in a few words>, from the slice-3 whole-branch review

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
   stage the changed files by name and `git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-mf-<n>.txt`.

Then dispatch a fresh session-model reviewer on `git diff <the commit before the fix wave>..HEAD` with the must-fix list,
asking for PASS or the items still open. Append its answer to `final-review.md`. A second round of must-fix items:
apply them the same way and re-review once more; a third: STOP and report to the owner.

- [ ] **Step 4: The gate, with gate 12 running**

`gate.sh` finds the oracle's Python in this order: `$MAGCOUPLING_PYTHON` (used as given, so a wrong value fails the
gate), then `reference/magcoupling-py/.venv/Scripts/python.exe` (or `.venv/bin/python`) in this checkout or another
worktree. It prints `oracle Python: <path>` when gate 12 runs, and `SKIP gate 12/12: no Python for
reference/magcoupling-py ...` (without failing) when none is found. Naming the venv explicitly makes a SKIP impossible.

Run in the background (about 6 minutes warm, longer cold):
```bash
MAGCOUPLING_PYTHON="$PWD/reference/magcoupling-py/.venv/Scripts/python.exe" bash linkage-sim-rs/scripts/gate.sh 2>&1 | tee magcoupling-rs/target/release-slice3/gate.log
```
When it exits, run:
```bash
L=magcoupling-rs/target/release-slice3/gate.log
tail -n 1 $L
grep -c 'SKIP gate 12/12' $L
grep -n 'oracle Python:' $L
grep -c '^== gate [0-9]*/12' $L
grep -E '^[0-9]+ passed' $L
grep -n 'differential data is current' $L
git checkout -- docs/chebyshev_lambda
git status --short
```
Expected: `GATE PASS`; `0`; `oracle Python: /c/Users/Cole/source/repos/linkage_simulation/reference/magcoupling-py/.venv/Scripts/python.exe`;
`12`; one pytest summary line `<N> passed ...` with no `failed` in it; `differential data is current (...)`; after the
restore, `git status --short` prints only `?? docs/analyses/2026-05-28-press-4bar-analysis.md`.
On a failure: STOP the release. The failing gate's output names the test; fix it as a must-fix of Step 3 (red test, fix,
the reviews it needs, commit) and run this step again from the top. Never edit `gate.sh` to pass.

- [ ] **Step 5: Web build, and the card is in the bundle**

Run:
```bash
D=magcoupling-rs/target/release-slice3
B=$D/build-web.log
bash linkage-sim-rs/scripts/build_web.sh 2>&1 | tee $B
grep -n 'parity guard:\|Magcoupling build complete!' $B
WASM=linkage-sim-rs/web/tools/magcoupler/magcoupling-web_bg.wasm
HEADING="$(sed -n '/const CARD_HEADING:/,/;/p' magcoupling-rs/src/gui/torque_card.rs | grep -o '"[^"]*"' | head -1 | tr -d '"')"
OLD_AUTO="$(git show "$(git merge-base main HEAD)":magcoupling-rs/src/gui/guidance.rs | sed -n '/^pub const AUTO_RESIZES_RING:/,/;/p' | grep -o '"[^"]*"' | head -1 | tr -d '"')"
echo "[$HEADING]"
echo "[$OLD_AUTO]"
[ -n "$HEADING" ] && grep -c -a -F "$HEADING" $WASM
[ -n "$OLD_AUTO" ] && grep -c -a -F "$OLD_AUTO" $WASM
git status --short
```
Expected: `parity guard: magcoupling-web builds without workbook-parity; magcoupling-rs features: ...` and
`Magcoupling build complete!`; the heading in brackets as `torque_card.rs` holds it after the G-3 review
(`[One step of each size control]` unless the review reworded it; `[]` means the constant was not found: STOP and read
`torque_card.rs`, since an empty text would match everything); `main`'s Auto ring-size line in brackets
(`[Auto resizes the ring when you change the magnet count; the lines below keep the present ring radius.]`); at least
`1`; `0`; `git status` unchanged (the bundles are gitignored). If the release link fails with `LNK1104` (Windows
MAX_PATH, 02-system lessons), build from a short path: `subst X: "$(cygpath -w "$PWD")"`, then `cd /x && bash
linkage-sim-rs/scripts/build_web.sh`, then `subst X: /d`.

- [ ] **Step 6: The native time per changed design, with the card (spec 7.1's method)**

The probe times, in a release build, the work a drag frame that changes the design adds: the `ConsequenceCache`'s miss,
which now holds the card, and the `FixCache`'s miss, asked for two designs one face-gap slider step apart in turn so
every call misses; each figure the minimum and the median of five rounds, every design timed once per round (the mean
of 200 calls after 20 warm-up calls), as the spec's section 7.1 measured, in rounds because this machine runs the first
seconds of a run about twice as fast as the rest (performance review 4.3). It times (a), (b), (e), 6 magnets and (a)
with Auto on the ring size (ruling C-6). It also counts the fit search's `compute_all` calls (its bisection replayed on
its own answer), which `02-system.yaml`'s drag-frame line states, and prints the share link of (a) with Auto on the
ring size, which Step 7 drags in the browser (performance review 4.2).

1. The machine is quiet (performance review 4.3): no cargo or rustc runs, and Step 4's gate has exited. Run:
```bash
tasklist | grep -c -i -E '^(cargo|rustc)\.exe'
tail -n 1 magcoupling-rs/target/release-slice3/gate.log
```
   Expected: `0` and `GATE PASS`. Otherwise wait for them to finish. Step 7 runs the same two commands before it starts
   the server.

2. Write `$W/timing-probe/Cargo.toml` with the Write tool, exactly:
```toml
[package]
name = "timing-probe"
version = "0.1.0"
edition = "2024"
publish = false

# Its own workspace: a slice-3 release probe, not part of any crate of the repository.
[workspace]

[dependencies]
magcoupling-rs = { path = "C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs", features = ["gui"] }

[profile.release]
debug = false
```
3. Write `$W/timing-probe/src/main.rs` with the Write tool, exactly:
```rust
//! Slice-3 release probe (not part of the crate): the native time per changed design of the
//! Size tab's two caches with the torque card, at (a), (b), (e), 6 magnets and (a) with Auto on
//! the ring size, by the method of the slice-3 spec's section 7.1 (the mean of 200 calls after
//! 20 warm-up calls; the minimum and the median of five, in a release build), in five rounds
//! that each time every design once, back to back: this machine runs the first seconds of a run
//! about twice as fast as the rest (spec 7.1's "about twice"), and rounds put every design's
//! median in the same regime where one design after another would split them (performance
//! review 4.3). Each call asks a cache for the other of two designs one face-gap slider step
//! apart, so every call misses, as on a drag frame that changes the design. Also, for scale,
//! each cache alone, the card alone and `compute_all`; the fit search's `compute_all` calls, its
//! bisection replayed on its own answer (docs/ai/02-system.yaml's drag-frame count); and the
//! share link's payload of (a) with Auto on the ring size, the design Step 7 drags in the
//! browser a second time (performance review 4.2).

use std::hint::black_box;
use std::time::Instant;

use magcoupling::engine::clamps::PAUSED;
use magcoupling::engine::size_controls::{
    DiameterFit, FIT_TOLERANCE_MM, dimensions, dimensions_with_minimum_diameter, ring_radius_range,
    smallest_fitting_diameter,
};
use magcoupling::engine::sizing::{FreeVariable, SizingOutcome, solve_in};
use magcoupling::gui::guidance::{ConsequenceCache, FixCache};
use magcoupling::gui::session::{Design, encode_share_payload};
use magcoupling::gui::sizing::{SizingMode, SizingState, sized};
use magcoupling::gui::torque_card::card;
use magcoupling::{DesignInputs, DesignResults, compute_all};

/// The mean time of 200 calls of `f` after 20 warm-up calls [ms].
fn mean_ms(f: &mut dyn FnMut()) -> f64 {
    for _ in 0..20 {
        f();
    }
    let start = Instant::now();
    for _ in 0..200 {
        f();
    }
    1e3 * start.elapsed().as_secs_f64() / 200.0
}

/// The minimum and the median of five rounds' figures [ms].
fn min_median(runs: &[f64]) -> (f64, f64) {
    let mut runs = runs.to_vec();
    runs.sort_by(f64::total_cmp);
    (runs[0], runs[runs.len() / 2])
}

/// The `compute_all` calls of one fit search (`smallest_fitting_diameter`) at `inputs` with
/// the Minimum OD `minimum`: its geometry, one; each diameter tried, one; the bisection
/// replayed on the search's own answer (a diameter fits from the answer up, so each middle
/// goes the way the search took it).
fn one_fit_search(inputs: &DesignInputs, minimum: Option<f64>) -> usize {
    let size = match minimum {
        None => dimensions(inputs),
        Some(minimum) => dimensions_with_minimum_diameter(inputs, minimum),
    };
    let Some(range) = size.map(|size| size.diameter_range) else {
        return 1;
    };
    match smallest_fitting_diameter(inputs, inputs.coupling.npole, minimum) {
        None => 1,
        Some(DiameterFit::AtMinimum(_)) => 2,
        Some(DiameterFit::Nowhere(_)) => 3,
        Some(DiameterFit::At(edge)) => {
            let mut calls = 3;
            let (mut low, mut high) = (range.min, range.max);
            while high - low > FIT_TOLERANCE_MM {
                let middle = low + (high - low) / 2.0;
                if middle <= low || middle >= high {
                    break;
                }
                calls += 1;
                if middle >= edge {
                    high = middle;
                } else {
                    low = middle;
                }
            }
            calls
        }
    }
}

/// The `compute_all` calls of the fit searches `Consequences::of` runs at `inputs` with the
/// Minimum OD unset: the search, and where the slider's floor stops it (`AtMinimum`) one
/// `dimensions` and the search again at the lowest Minimum OD (its `fit_unlocked`).
fn fit_search_calls(inputs: &DesignInputs) -> usize {
    let first = one_fit_search(inputs, None);
    match smallest_fitting_diameter(inputs, inputs.coupling.npole, None) {
        Some(DiameterFit::AtMinimum(_)) => {
            let lowest = dimensions(inputs).map(|size| size.minimum_diameter_range.min);
            first + 1 + lowest.map_or(0, |lowest| one_fit_search(inputs, Some(lowest)))
        }
        _ => first,
    }
}

/// A design timed in every round: the design shown and the design one face-gap slider step
/// away, with their results, its own two caches, and the five figures' rounds.
struct Timed {
    key: &'static str,
    pair: [(DesignInputs, DesignResults); 2],
    consequences: ConsequenceCache,
    fixes: FixCache,
    calls: usize,
    /// Both caches, the ConsequenceCache, the FixCache, the card, compute_all.
    rounds: [Vec<f64>; 5],
}

impl Timed {
    fn new(key: &'static str, shown: DesignInputs) -> Self {
        let mut moved = shown.clone();
        moved.metal.face_gap_mm += 0.01; // one step of the gap slider: another design
        let pair = [
            (shown.clone(), compute_all(&shown)),
            (moved.clone(), compute_all(&moved)),
        ];
        Self {
            key,
            pair,
            consequences: ConsequenceCache::default(),
            fixes: FixCache::default(),
            calls: 0,
            rounds: Default::default(),
        }
    }

    /// One round: each figure once, the caches asked for the other design of the pair each call.
    fn round(&mut self) {
        let Self {
            pair,
            consequences,
            fixes,
            calls,
            rounds,
            ..
        } = self;
        rounds[0].push(mean_ms(&mut || {
            let (inputs, results) = &pair[*calls % 2];
            *calls += 1;
            black_box(consequences.get(inputs, results, None, PAUSED));
            black_box(fixes.get(inputs, results, PAUSED));
        }));
        rounds[1].push(mean_ms(&mut || {
            let (inputs, results) = &pair[*calls % 2];
            *calls += 1;
            black_box(consequences.get(inputs, results, None, PAUSED));
        }));
        rounds[2].push(mean_ms(&mut || {
            let (inputs, results) = &pair[*calls % 2];
            *calls += 1;
            black_box(fixes.get(inputs, results, PAUSED));
        }));
        let (inputs, results) = &pair[0];
        rounds[3].push(mean_ms(&mut || {
            black_box(card(inputs, results, None, PAUSED));
        }));
        rounds[4].push(mean_ms(&mut || {
            black_box(compute_all(inputs));
        }));
    }
}

fn main() {
    let a = DesignInputs::default();
    // (b): the gui-smoke share link's design, face gap 1.5 mm with Auto on the magnet length,
    // shown at its solved length.
    let mut smoke = a.clone();
    smoke.metal.face_gap_mm = 1.5;
    let variable = FreeVariable::AxialLength;
    let b = match solve_in(
        &smoke,
        variable,
        smoke.metal.required_min_Nm,
        variable.range(),
    ) {
        Ok(SizingOutcome::Solved(point)) => sized(variable, &smoke, point.value),
        other => panic!("(b) does not solve: {other:?}"),
    };
    let mut e = a.clone();
    e.coupling.npole = 8;
    let mut six = a.clone();
    six.coupling.npole = 6;
    let ring = match solve_in(
        &a,
        FreeVariable::RingRadius,
        a.metal.required_min_Nm,
        ring_radius_range(&a, None).unwrap_or_else(|| FreeVariable::RingRadius.range()),
    ) {
        Ok(SizingOutcome::Solved(point)) => sized(FreeVariable::RingRadius, &a, point.value),
        other => panic!("(a) with Auto on the ring size does not solve: {other:?}"),
    };
    let ring_link = encode_share_payload(&Design {
        inputs: a.clone(),
        sizing: SizingState {
            mode: SizingMode::TorqueToMagnets,
            variable: FreeVariable::RingRadius,
        },
    });
    let mut designs = [
        Timed::new("(a)", a),
        Timed::new("(b)", b),
        Timed::new("(e)", e),
        Timed::new("(6 magnets)", six),
        Timed::new("(Auto ring)", ring),
    ];
    for _ in 0..5 {
        for design in designs.iter_mut() {
            design.round();
        }
    }
    println!(
        "native release; each figure: min / median of five rounds, each round the mean of 200 calls after 20 warm-up at every design in turn [ms]"
    );
    for design in &designs {
        let [both, consequences, fixes, card_alone, one] =
            design.rounds.each_ref().map(|runs| min_median(runs));
        println!(
            "{}: both caches per changed design {:.3} / {:.3}; ConsequenceCache {:.3} / {:.3}; FixCache {:.3} / {:.3}; card {:.3} / {:.3}; compute_all {:.3} / {:.3}; fit search: {} compute_all calls",
            design.key,
            both.0,
            both.1,
            consequences.0,
            consequences.1,
            fixes.0,
            fixes.1,
            card_alone.0,
            card_alone.1,
            one.0,
            one.1,
            fit_search_calls(&design.pair[0].0),
        );
    }
    println!("ring share payload: {ring_link}");
}
```
4. Run (the first release build takes a few minutes; run nothing else heavy meanwhile, as the times are wall-clock):
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
cp magcoupling-rs/Cargo.lock $W/timing-probe/Cargo.lock
CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/timing-probe" cargo run --release --quiet --manifest-path $W/timing-probe/Cargo.toml | tee $W/timing-T10.txt
git status --short
```
Expected: seven lines: the header `native release; each figure: min / median of five rounds, each round the mean of 200
calls after 20 warm-up at every design in turn [ms]`; then `(a): both caches per changed design A / B; ConsequenceCache
...; FixCache ...; card ...; compute_all ...; fit search: 29 compute_all calls`, then `(b)` (29), `(e)` (32: its search
stops at the slider's floor and runs again at the lowest Minimum OD), `(6 magnets)` (32) and `(Auto ring)` (29) in the
same form; then `ring share payload: ` and the payload (2,646 characters on the plan's code; it decodes back to the
example design with Auto on the ring size). The fit search is unchanged by slice 3: the counts of (a), (b) and (e) were
measured on 69fa505's search. Measured on the plan's code after Task 9 (a quiet machine, 49 s): medians (a) 3.404, (b)
3.424, (e) 3.850, (6 magnets) 4.305 and (Auto ring) 3.805 ms, with `compute_all` medians 0.035-0.039 ms, one regime;
the minimum can come from round 1's faster seconds ((a) 1.692 ms), so Step 8 quotes the medians. The spec's section 7.1
put the card near 1.0 ms, the consequences' other work near 0.7 ms and the fixes near 0.13 ms at (a) on this machine,
so both caches near 1.8 ms there, or up to about twice that on a run where the machine is slow (later spec runs were,
about two times). `git status` prints only the owner's untracked file. A compile error in the probe: the code's public
API differs from spec 3.2's (a binding interface): STOP and report.

- [ ] **Step 7: gui-smoke on the local server, with the browser frame check**

1. Run Step 6's two precondition commands again (expected `0` and `GATE PASS`). Then start the server in the
   background: `bash linkage-sim-rs/scripts/serve_web.sh 8080` (it serves `linkage-sim-rs/web/`
   at `http://localhost:8080/tools/`, `/tools/linkage/` and `/tools/magcoupler/`).
2. Invoke the `gui-smoke` skill with the arguments `{"frame_check": true}` (it smokes `http://localhost:8080`). Write
   the whole result object it returns, verbatim, to `$W/smoke-local.json` with the Write tool.
3. Run:
```bash
python - <<'EOF'
import json
r = json.load(open('.superpowers/sdd/2026-10-19-magcoupler-slice3/smoke-local.json', encoding='utf-8'))
m = r['magcoupling']
print(r.get('passed'), r.get('linkage', {}).get('passed'), m.get('passed'), m.get('console_errors'))
print({k: m.get(k) for k in ('first_view', 'details_toggle', 'share_link_loaded', 'sizing_solved', 'design_file_loaded',
                             'explorer_ready', 'why_link_opened', 'tooltip_plain', 'dashboard_scrolls', 'torque_card')})
print({k: m.get(k) for k in ('drag_moved', 'drag_frames', 'drag_frame_ms_median', 'drag_frame_ms_p95',
                             'drag_frame_ms_max', 'drag_frames_within_60fps')})
EOF
```
Expected: `True True True []`; every listed field `True`; `drag_moved` `True`, `drag_frames` at least 30,
`drag_frame_ms_p95` at or under 16.7 and `drag_frames_within_60fps` `True` (the threshold of spec 7.2: the 95th
percentile of the drag's drawn frames within one 60 fps frame).
- A canvas-click field false (`design_file_loaded`, `why_link_opened`, `tooltip_plain`, `dashboard_scrolls`) whose notes
  say the target was not found: run the skill once more. A field false twice, `torque_card` false, or any console
  error: STOP and treat it as a must-fix (Step 3), with the smoke's notes and screenshots as evidence.
- `drag_moved` false or fewer than 30 frames: the measurement failed, not the card (the notes say whether the timer
  saw the app draw too few frames or the handle was not held): run the skill once more; twice: STOP and treat it as a must-fix of
  the smoke's frame step (Task 8's code).
- A valid drag (moved, at least 30 frames) with `drag_frame_ms_p95` above 16.7: the frame check missed. Run Step 7b
  (13A's held-back optimization), then Steps 4 to 7 again. If it still misses: STOP and report the figures to the
  owner (decision 13A: the owner decides).
3b. The drag at the frame-slowest design: the example design with Auto on the ring size (performance review 4.2).
   Run:
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
sed -n 's/^ring share payload: //p' $W/timing-T10.txt > $W/ring-payload.txt
sed -n 's/^const FRAME_TIMER = `\(.*\)`$/\1/p' .claude/workflows/gui-smoke.js > $W/frame-timer.js
grep -o 'async (page) => { const x = X, y = Y;.*sizing_lines: sizing }; }' .claude/workflows/gui-smoke.js > $W/drag-code.js
wc -c $W/ring-payload.txt $W/frame-timer.js $W/drag-code.js
```
   Expected: `2647`, `633` and `760` bytes (measured on the plan's code; the payload file holds its newline). If any is
   0, STOP: Step 6's probe or T8's script differs from the plan. Then, with the Playwright tools (load `browser_resize`,
   `browser_run_code_unsafe`, `browser_take_screenshot` and `browser_close` through ToolSearch):
   - resize to 1280 x 800;
   - run `async (page) => { await page.addInitScript(<frame-timer.js>); await page.goto('http://localhost:8080/tools/magcoupler/?m=<ring-payload.txt>'); await page.waitForTimeout(8000); }`
     with the two files' texts in place;
   - take a screenshot: the Size tab, "Auto: smallest ring radius", the Outside diameter slider locked. Find the Magnet
     face gap slider's handle (it reads 1.40 mm);
   - run `drag-code.js`'s code with that handle's X and Y in CSS pixels;
   - take a screenshot, then close the browser.

   Write `$W/smoke-ring.json` with the Write tool:
   `{"drag_moved": <the gap now reads a value other than 1.40 mm>, "drag_frames": <frames>, "drag_frame_ms_median": <median>, "drag_frame_ms_p95": <p95>, "drag_frame_ms_max": <max>, "sizing_lines": <sizing_lines>}`.
   Expected: `drag_moved` true, `drag_frames` at least 30, `drag_frame_ms_p95` at or under 16.7, `sizing_lines` 0.
   - Not moved, or under 30 frames: run this item once more. Twice: STOP (T8's drag code).
   - A valid drag over 16.7 ms: Step 7b, then Steps 4 to 7 again, both drags.
4. The browser the frames were measured in, for the docs line: with the Playwright MCP tools (load
   `mcp__plugin_playwright_playwright__browser_navigate`, `..._browser_evaluate` and `..._browser_close` through
   ToolSearch), open `http://localhost:8080/tools/`, evaluate `() => navigator.userAgent`, and close the browser. Write
   the browser's name and major version from the user agent (for example `Chrome 141`) to `$W/browser.txt` with the Write
   tool, one line.
5. Stop the server (stop its background task). `git status --short` must again print only the owner's untracked file
   (the smoke writes and deletes `.playwright-mcp/magcoupling-smoke-design.json`, gitignored).

- [ ] **Step 7b: Only if a frame check of Step 7 missed (the smoke drag or the ring drag): the held-back optimization (decision 13A)**

One session-model implementer, one commit; the base geometry's reuse of spec 7.2: each stepped design's size read from
the results the side already computed, through one engine function that `geometry` itself calls (DRY), in place of
`dimensions(&stepped)` (a geometry and so one more `compute_all` and `input_rows` per side; spec 7.1 measured the card
at about half its time with it).

1. The red test, in `magcoupling-rs/tests/size_controls.rs`. Old:
```rust
    resize_diameter_with_minimum, resize_length, ring_radius_range, smallest_fitting_diameter,
    with_ring_radius,
};
```
New:
```rust
    resize_diameter_with_minimum, resize_length, ring_radius_range, size_of,
    smallest_fitting_diameter, with_ring_radius,
};
```
and at the end of the file, after its last test, this test:
```rust
#[test]
fn the_size_read_from_the_results_is_the_size_dimensions_reads() {
    // Slice 3's held-back optimization (its spec section 7.2): `size_of(inputs, results)`,
    // which reads no geometry, is `dimensions`' Outside diameter and overall length bit for bit
    // wherever `dimensions` has them: the five probe designs, every even magnet count of the
    // slider at the defaults, and a boss that sets the outside diameter.
    use magcoupling::engine::explain::probes::probe_designs;
    let mut designs: Vec<(String, DesignInputs)> = probe_designs()
        .into_iter()
        .map(|(name, design)| (name.to_owned(), design))
        .collect();
    for npole in (4..=40).step_by(2) {
        let mut design = DesignInputs::default();
        design.coupling.npole = npole;
        designs.push((format!("{npole} magnets"), design));
    }
    let mut boss = DesignInputs::default();
    boss.metal.boss_od_mm = 40.0;
    designs.push((String::from("a 40 mm boss"), boss));
    let mut compared = 0;
    for (name, design) in designs {
        let Some(size) = dimensions(&design) else {
            continue;
        };
        let (diameter_mm, length_mm) = size_of(&design, &compute_all(&design));
        assert_eq!(diameter_mm.to_bits(), size.diameter_mm.to_bits(), "{name}");
        assert_eq!(
            length_mm.to_bits(),
            size.overall_length_mm.to_bits(),
            "{name}"
        );
        compared += 1;
    }
    assert!(compared >= 20, "{compared}");
}
```
Run `cargo test --manifest-path magcoupling-rs/Cargo.toml --test size_controls`. Expected: it does not compile
(`error[E0432]: unresolved import` ... `size_of`).

2. `magcoupling-rs/src/engine/size_controls.rs`. Old:
```rust
use super::api::{DesignInputs, compute_all};
```
New:
```rust
use super::api::{DesignInputs, DesignResults, compute_all};
```
Old (in `geometry`):
```rust
    let results = compute_all(inputs);
    let cup = results.model.cup_od_mm;
    let md = &inputs.metal;
    let diameter = DiameterBreakdown {
        cap_mm: md.cap_od_mm,
        cup_mm: cup,
        boss_mm: md.boss_od_mm,
    };
    let outside = diameter.outside_mm();
    let overall_length = results.metal.axial_stack_mm;
```
New:
```rust
    let results = compute_all(inputs);
    let cup = results.model.cup_od_mm;
    let md = &inputs.metal;
    let diameter = diameter_of(inputs, &results);
    let (outside, overall_length) = size_of(inputs, &results);
```
Old:
```rust
/// Current outside diameter (cup, cap or boss) and overall axial stack. `None` means the
/// geometry or the intersection of its supported input ranges cannot define these controls.
pub fn dimensions(
```
New:
```rust
/// The three diameters the outside diameter is the largest of, from a design and its results:
/// the one place they are read.
fn diameter_of(inputs: &DesignInputs, results: &DesignResults) -> DiameterBreakdown {
    DiameterBreakdown {
        cap_mm: inputs.metal.cap_od_mm,
        cup_mm: results.model.cup_od_mm,
        boss_mm: inputs.metal.boss_od_mm,
    }
}

/// The outside diameter and the overall length [mm] of `inputs` from its own results
/// (`compute_all(inputs)`), as [`dimensions`] reads them: the largest of the cap, the cup and
/// the boss, and the axial stack. A caller that already holds the results reads the size with
/// no geometry of its own (slice 3's held-back optimization: the torque card's stepped
/// designs). It checks no range, so it gives numbers where [`dimensions`] is `None`.
pub fn size_of(inputs: &DesignInputs, results: &DesignResults) -> (f64, f64) {
    (
        diameter_of(inputs, results).outside_mm(),
        results.metal.axial_stack_mm,
    )
}

/// Current outside diameter (cup, cap or boss) and overall axial stack. `None` means the
/// geometry or the intersection of its supported input ranges cannot define these controls.
pub fn dimensions(
```
Run the same test. Expected: `test result: ok. 39 passed` (38 at 69fa505 plus this one; checked on 69fa505's
engine with this change, and clippy clean both ways).

3. `magcoupling-rs/src/gui/torque_card.rs`, two edits. Task 4's `Shown::change` reads each stepped design's size: its
   `design` is the stepped design, `stepped` its results, and `self.size` the design shown's `dimensions`. The import
   first (Task 3's line, which `SliderMapping` leaves without the two resize functions, plan review 2.3). Old:
```rust
use crate::engine::size_controls::{SizeControls, dimensions, dimensions_with_minimum_diameter};
```
   New:
```rust
use crate::engine::size_controls::{
    SizeControls, dimensions, dimensions_with_minimum_diameter, size_of,
};
```
   Then the size change. A step that keeps the magnet count reads its size from the results the side already holds; a
   count step still reads `dimensions` (spec 2.2's rule on its stepped side, plan review 2.1). Old:
```rust
        let (diameter_change_mm, length_change_mm) = match (self.size, dimensions(design)) {
            (Some(shown), Some(after)) => (
                Some(after.diameter_mm - shown.diameter_mm),
                Some(after.overall_length_mm - shown.overall_length_mm),
            ),
            _ => (None, None),
        };
```
   New:
```rust
        // The size the sliders would show on the stepped design; none where either design has
        // no size controls (spec 2.2). A step that keeps the magnet count keeps them, so its size
        // is read from the results the side already holds, with no geometry of its own (decision
        // 13A's held-back optimization); a count step can leave them undefined (a design file's
        // ring past its slider), so it reads `dimensions`.
        let after = if design.coupling.npole == self.inputs.coupling.npole {
            Some(size_of(design, stepped))
        } else {
            dimensions(design).map(|size| (size.diameter_mm, size.overall_length_mm))
        };
        let (diameter_change_mm, length_change_mm) = match (self.size, after) {
            (Some(shown), Some((diameter_mm, length_mm))) => (
                Some(diameter_mm - shown.diameter_mm),
                Some(length_mm - shown.overall_length_mm),
            ),
            _ => (None, None),
        };
```
   Tests 1 (`each_number_is_the_difference_of_two_compute_all_runs`), 9a
   (`size_changes_and_margins_are_drawn_only_when_they_show`) and
   `a_step_to_a_design_without_size_controls_draws_no_size_change` (Task 4: a count step to a design without size
   controls draws no size change) pin every size change against `dimensions`, `None` exactly where either design has
   none, and must pass unedited: the count row still reads `dimensions`, since a design file's ring past its slider can
   leave the size controls undefined there. If any of them fails: revert this step's card change and STOP, and report
   to the owner with Step 7's figures. (Measured on the plan's code after Task 9, with Step 7b.1, 7b.2, these two edits
   and 7b.4: `tests/size_controls.rs` 39 passed; the card module 30 passed, tests 1, 9a and
   `a_step_to_a_design_without_size_controls_draws_no_size_change` unedited; app library 971 passed, 6 ignored; both
   clippy runs clean; `cargo fmt -- --check` clean. With a size read through `size_of` on every side, the count step's
   included, that last test fails `left: (Some(23.76620273735618), Some(0.0))`, `right: (None, None)`: the guard fires.)

4. Docs, in the same commit: `magcoupling-rs/README.md`'s `src/engine/size_controls.rs` row (~:150). Old:
```text
and `with_ring_radius` (a ring radius applied through the Outside diameter slider's own mapping, so the cap follows the cup). The torque model is untouched |
```
New:
```text
and `with_ring_radius` (a ring radius applied through the Outside diameter slider's own mapping, so the cap follows the cup); since slice 3 `size_of` (the outside diameter and overall length a design's own results give, as `dimensions` reads them, with no geometry of its own: the torque card's stepped designs). The torque model is untouched |
```
   and `docs/ai/03-structure.yaml`'s engine `size_controls` entry (~:51). Old:
```text
so the cap OD and cap thread follow the cup; None where that mapping refuses), each reusing dimensions' own expressions)
```
New:
```text
so the cap OD and cap thread follow the cup; None where that mapping refuses), each reusing dimensions' own expressions; since slice 3 size_of (the outside diameter and overall length from a design's own results, as dimensions reads them, with no geometry: the torque card's stepped designs, the held-back optimization of decision 13A))
```
   and the README's `tests/size_controls.rs` test row (~:678). Old:
```text
(the cap follows the cup; out-of-range requests return `None`). |
```
New:
```text
(the cap follows the cup; out-of-range requests return `None`); the size read from a design's results is the size `dimensions` reads, bit for bit (`the_size_read_from_the_results_is_the_size_dimensions_reads`, slice 3's held-back optimization). |
```

5. Both suites, both clippy runs, `cargo fmt`, then `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-opt.txt`:
```text
perf(magcoupler): the torque card reads each stepped design's size from its own results (decision 13A)

The browser frame check missed one 60 fps frame (Step 7: <the figures>). The card's
stepped designs that keep the magnet count read their size through
engine::size_controls::size_of, which geometry itself calls, in place of a geometry per
side; a count step still reads dimensions (a design file's ring past its slider can leave
its size controls undefined). Tests 1, 9a and
a_step_to_a_design_without_size_controls_draws_no_size_change pin every size change
against dimensions, unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
   (with Step 7's figures in place of `<the figures>`), stage the five files by name, commit, and run Steps 4 to 7
   again. The whole-branch reviewer of Step 2 re-reviews this commit as Step 3's re-review does.

- [ ] **Step 8: Record the release checks and the measured figures, and commit**

1. The figures, read from the files Steps 6 and 7 wrote (nothing typed):
```bash
python - <<'EOF'
import json, re, subprocess
W = '.superpowers/sdd/2026-10-19-magcoupler-slice3'
date = subprocess.run(['date', '+%F'], capture_output=True, text=True, check=True).stdout.strip()
probe = {}
for line in open(f'{W}/timing-T10.txt', encoding='utf-8'):
    m = re.match(r'^\((\w+)\): both caches per changed design ([0-9.]+) / ([0-9.]+);.*fit search: (\d+) compute_all calls', line)
    if m:
        probe[m.group(1)] = (m.group(3), int(m.group(4)))
m = json.load(open(f'{W}/smoke-local.json', encoding='utf-8'))['magcoupling']
r = json.load(open(f'{W}/smoke-ring.json', encoding='utf-8'))
browser = open(f'{W}/browser.txt', encoding='utf-8').read().strip()
calls = 22 + probe['a'][1] + 3
print(f'(about {calls} compute_all calls at the example design: up to 22 for the card\'s eight steps, {probe["a"][1]} for the fit search, one for each of the two breakdowns and the panel\'s own for the card); this is accepted (measured {date}: the Size-tab invariant below).')
print(f'(both together, with the torque card, measured {date} at {probe["a"][0]} ms per changed design at the example design, {probe["b"][0]} ms at the gui-smoke design and {probe["e"][0]} ms at 8 magnets, native release, the median of five runs of the mean of 200 calls; in {browser} at 1280 x 800 a drag of the Magnet face gap slider drew {m["drag_frames"]} frames at a median of {m["drag_frame_ms_median"]} ms and a 95th percentile of {m["drag_frame_ms_p95"]} ms, the slowest {m["drag_frame_ms_max"]} ms; at the example design with Auto on the ring size {r["drag_frames"]} frames at a median of {r["drag_frame_ms_median"]} ms and a 95th percentile of {r["drag_frame_ms_p95"]} ms; both within one 60 fps frame).')
EOF
```
   Expected: two lines, `(about 54 compute_all calls at the example design: ...` (22 + 29 + 3) and `(both together, with
   the torque card, measured YYYY-MM-DD at ...`.
2. `docs/ai/02-system.yaml` `:106` (with the first printed line in place of the parenthesis and the clause after it).
   Old:
```text
The consequence lines recompute on every drag frame (about 30 compute_all calls); this is accepted.
```
   New: `The consequence lines and the torque card recompute on every drag frame that changes the design ` followed by the
   first printed line.
3. `docs/ai/02-system.yaml` `:116`. Old:
```text
(both together measured at under 1.6 ms per changed design, native release, so a drag frame stays within budget).
```
   New: the second printed line.
4. `docs/ai/05-update-tracker.md`: a bullet after T9's last bullet of the slice-3 entry. Write `passed with no
   must-fix` if Step 2 returned PASS, else `passed after its fix wave of <N> items (re-reviewed)` with the count of Step
   3's commits; add `, after the held-back optimization (Step 7b)` after `the browser frame check` if Step 7b ran. Old:
```text
the card's percents read the model's end factor (BL-065).

## 2026-10-17 — Magcoupler shaft joints
```
New:
```text
the card's percents read the model's end factor (BL-065).
- T10 release checks (YYYY-MM-DD): the whole-branch review passed with no must-fix (record: .superpowers/sdd/2026-10-19-magcoupler-slice3/final-review.md); `gate.sh` GATE PASS with gate 12 running (the oracle venv reference/magcoupling-py/.venv); the web build carries the card (its heading is in magcoupling-web_bg.wasm, main's Auto ring-size line is not); the Size tab's two caches with the card take <A> ms per changed design at the example design, native release (record: .superpowers/sdd/2026-10-19-magcoupler-slice3/timing-T10.txt); a gui-smoke run passed on the local server with the card step and the browser frame check (<BROWSER>: <K> drag frames, median <M> ms, 95th percentile <P> ms, and <P2> ms at the ring design; record: .superpowers/sdd/2026-10-19-magcoupler-slice3/smoke-local.json). Not yet done: the owner's review of the diff, the merge, the push and the deploy check.

## 2026-10-17 — Magcoupler shaft joints
```
   with `<A>` the probe's (a) median, `<BROWSER>` the line of `browser.txt`, and `<K>`, `<M>`, `<P>` the smoke's
   `drag_frames`, `drag_frame_ms_median` and `drag_frame_ms_p95`, as Step 8.1 printed them, and `<P2>`
   `smoke-ring.json`'s `drag_frame_ms_p95`.
5. `docs/ai/backlog.yaml`, BL-073: append to its `notes:` string, right before the closing `"` (anchor:
   `Deploy authorization is asked at the STOP."`, BL-073's alone):
   ` Release checks done YYYY-MM-DD (T10): whole-branch review PASS, GATE PASS with gate 12 running, web build, the native timing and gui-smoke with the browser frame check PASS (p95 <P> ms; <P2> ms at the ring design); waiting for the owner's review of the diff and the deploy authorization.`
6. Check, then commit:
```bash
python - <<'EOF'
import yaml
b = yaml.safe_load(open('docs/ai/backlog.yaml', encoding='utf-8'))
item = {x['id']: x for x in b}
assert item['BL-073']['status'] == 'in_progress'
assert item['BL-073']['notes'].endswith("waiting for the owner's review of the diff and the deploy authorization.")
lines = open('docs/ai/02-system.yaml', encoding='utf-8').read().splitlines()
for marker in ('compute_all calls at the example design: up to 22', 'with the torque card, measured'):
    hit = [line for line in lines if marker in line]
    assert len(hit) == 1, (marker, len(hit))
    yaml.safe_load('k:\n' + hit[0])
assert not any('under 1.6 ms' in line or 'about 30 compute_all' in line for line in lines)
print('ok')
EOF
```
   Expected: `ok`. Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-a.txt`:
```text
docs(magcoupler): record the slice-3 release checks and the measured drag-frame figures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
then:
```bash
git add docs/ai/02-system.yaml docs/ai/05-update-tracker.md docs/ai/backlog.yaml
git status --short
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-a.txt
```
Expected: the three files staged (`M `) and the owner's file untracked; one new commit.

- [ ] **Step 9: The screen dump again, main's side, and every quote checked**

1. The branch's side (after the fix wave), and its review dumps:
```bash
D=magcoupling-rs/target/release-slice3
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
mkdir -p $D/after $D/before
CARGO_TARGET_DIR="$PWD/magcoupling-rs/target/screen-dump" cargo run --quiet --manifest-path $W/screen-dump/after/Cargo.toml > $D/after/screen.tsv
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > $D/after/guidance-strings.txt
cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture > $D/after/overlay-strings.txt
NOTES_REVIEW_OUT="$PWD/$D/after/notes.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture
grep -c '^test result: ok. 1 passed' $D/after/guidance-strings.txt $D/after/overlay-strings.txt
python -I $W/same_designs.py $D/after/screen.tsv $D/after/guidance-strings.txt
```
Expected: the notes run prints `21 notes at 5 designs: .../notes.txt` and `test result: ok. 1 passed`; the `grep -c`
prints `1` for each dump file; `same_designs.py` prints 52 lines starting `SAME` and `0 difference(s)` (T9 Step 3).

2. `main`'s side, from a clean copy of the merge base (the first build there takes several minutes; it may run in the
   background from Step 8 on, after the timed Steps 6 and 7):
```bash
D=magcoupling-rs/target/release-slice3
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
BASE="$(git merge-base main HEAD)"
rm -rf $D/main-src && mkdir -p $D/main-src $D/before
git archive "$BASE" | tar -x -C $D/main-src
(cd $D/main-src && export CARGO_TARGET_DIR="$PWD/../main-target" &&
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib guidance_strings_for_review -- --ignored --nocapture > ../before/guidance-strings.txt &&
  cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib overlay_strings_for_review -- --ignored --nocapture > ../before/overlay-strings.txt &&
  NOTES_REVIEW_OUT="$PWD/../before/notes.txt" cargo test --manifest-path magcoupling-rs/Cargo.toml --features app --lib notes_for_review -- --ignored --nocapture)
mkdir -p $W/screen-dump/before
sed -e 's/screen-dump-after/screen-dump-before/g' -e '/^default = \["card"\]$/d' -e 's/: on here only\./: off on this side (main)./' -e "s#C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs#$(cygpath -m "$PWD/$D/main-src/magcoupling-rs")#" $W/screen-dump/after/Cargo.toml > $W/screen-dump/before/Cargo.toml
cp $D/main-src/magcoupling-rs/Cargo.lock $W/screen-dump/before/Cargo.lock
grep -c 'default = \["card"\]' $W/screen-dump/before/Cargo.toml
CARGO_TARGET_DIR="$PWD/$D/main-target" cargo run --quiet --manifest-path $W/screen-dump/before/Cargo.toml > $D/before/screen.tsv
grep -c '^test result: ok. 1 passed' $D/before/guidance-strings.txt $D/before/overlay-strings.txt
awk -F'\t' '{print $1}' $D/before/screen.tsv | sort | uniq -c | tr '\n' ' '
git status --short
```
Expected: the notes run prints `21 notes at 5 designs: ...`; `0` (the `card` feature is off on `main`'s side: `main`
has neither `gui::torque_card` nor the gap re-export); the `grep -c` prints `1` for each file; the record counts of
T9 Step 2 without `card` and `gap` (`18 assumption 2 bond screen 4 corner 20 count 10 guide label 10 guide why 187
input 34 input section 21 note 1112 result 18 row 18 row hover 1 share link 4 size tab 15 start here 2 status 3
stored 3D 19 verdict`, measured with the tool at 69fa505); `git status` shows only the owner's untracked file
(`main-src/` and the dumps are under the gitignored `magcoupling-rs/target/`).

3. Every quote, on both sides:
```bash
python -I .superpowers/sdd/2026-10-19-magcoupler-slice3/check_shipped.py magcoupling-rs/target/release-slice3/after/screen.tsv magcoupling-rs/target/release-slice3/before/screen.tsv
```
Expected: every line starts `OK`, `main`'s five `(main)` lines included, and the last line is `0 mismatch(es)`. A
`MISMATCH` here means a text changed after T9 (the fix wave) and a docs line or `QUOTES` entry was not updated with it:
apply Step 3's docs rule, commit the docs (a `docs(magcoupler): ...` commit with its own message file), and run this
step again.

- [ ] **Step 10: STOP: the owner's sheet and the diff, then ask for the OK**

1. The diffs:
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
git diff main...HEAD > $W/slice3.diff
git diff main...HEAD -- . ':(exclude)magcoupling-rs/tests' ':(exclude)*_test*' > $W/slice3-src.diff
git diff --stat main...HEAD > $W/slice3.stat
tail -n 1 $W/slice3.stat
```
Expected: the stat's last line counts the files changed. `slice3-src.diff` is the production and docs part of the diff:
it leaves out `magcoupling-rs/tests` and test files, while the in-file `mod tests` blocks stay in the full diff only.

2. Write `$W/decisions.md` with the Write tool, exactly this text, then append the plan's review section to it:
```markdown
## 6. Rulings and readings for you to keep or reverse

Each stands unless you name it in your answer to question 1.

The controller's rulings on your night-run approval ("Approved. Run throughout the night to implement /ultracode as
needed", 2026-10-19; `.superpowers/sdd/2026-10-19-magcoupler-slice3/owner-decisions.md`):
1. O-1, the jump rule widened: the Overall length row reads words whenever the overlapping magnet length changes by
   anything other than the step, not only by more than it. The short cup cavity's +1 mm side reads words where your
   9A as written showed -6.2 % for +1 mm (section 2, "short cavity").
2. O-2, no temperature quantity: the card quotes no Temperature limits margin (its terms read the stored 3D fields of
   the example design); its flips still show. The check is amber at worst, so it is never red on both designs (section
   2, "temperature edge").
3. The design shown decides first, by the readouts' own rule: one line in place of the rows where its magnets do not
   fit, the model has no prediction, or its pull-out or hot-low torque is not above zero (section 2: short magnets,
   shown does not fit, variation 1.0, zero torque).
4. The card's open or closed default is decided only when the Equation panel opens or closes or the window is
   resized: never by a design change, and never by a side panel widening, your drag of its edge included.
5. The layout tests' windows: 1280 x 800 and 930 x 1024, with the body's cap checked at 930 x 800.
6. Spec review round 4: the card's header row wraps, so in a narrow window its Why link continues under the heading;
   and a count side whose calibration switch takes its torque to or below zero shows its number with the calibration
   note (section 2, "(c), f_cal_original -0.95").
7. R-13, the shaft boss plateau: where the shaft boss sets the Outside diameter (its inputs are under Detailed
   inputs), one +1 mm step lifts the cup by several millimetres and still reads as a number (section 2, "R-13 boss
   plateau"); there is no diameter jump rule, which would go beyond 9A. Recorded in BL-074's evidence.

The spec's own decisions (section 1.2); T-9, T-12, T-13 and T-14 narrow or read one of yours:
- T-1. One pure module draws nothing itself: the panel draws the card, the release's screen dump prints it.
- T-2. The N±2 lines' logic moved into the card, not copied: one rule for every row's words, and the readouts' own
  prediction rule throughout.
- T-3. "Set by Auto" is drawn outside the card's cache, so an Auto toggle recomputes nothing.
- T-4. Two word forms beyond 9A's: a keyway that breaks through the hub, and a count no slider takes (an odd count
  from a file); section 2 shows their words.
- T-5. The two size rows are left out where the Size tab draws no such slider (an odd count from a file).
- T-6. Each failing check's quantity comes from one table of check terms, the one the fix tests read.
- T-7. The card's open state is decided when the view changes, never on a design change; its body scrolls above the
  drawing's floor.
- T-8. With the card closed (930 x 800, or the Equation panel open), nothing near Magnets per ring says Auto re-sizes
  after a count change: the two Auto lines left with the N±2 lines.
- T-9 (10B, float noise). A change is drawn to at most two more decimals than its unit's own; a smaller one reads
  "+0.00" or "-0.00" with its sign. No design of the spec comes near it; a design file's degenerate base does (see
  "What else changes on screen").
- T-10. "The window is narrow" is a width rule: the card starts closed while the centre region is under 400 points.
- T-11. The gearbox quantity names the torque its check compares, cold-high at the gearbox input: a third torque named
  on the Size tab, as the G-3 review left it (section 2).
- T-12 (5A and 9A). The three one-line forms draw their line alone, with no caption and no Auto line.
- T-13 (8A). No Temperature limits quantity on the card (O-2 above).
- T-14 (10B, per row). A row's two N·m share their decimals: (e)'s Outside diameter row reads "-0.006" and "-0.005"
  (section 2), where each to its own fewest decimals would read "-0.01" and "-0.005" for a true ratio of 1.35:1.

What else changes on screen:
- The first view's drawing is shorter with the card open: about 185 pt at the example design in place of 335.5 pt at
  1280 x 800 (the spec's headless measure in the app's style, section 6.2).
- A click on the card's heading, as on its arrow, opens or closes it, and that choice holds for the session.
- The card's own Why link opens the Equation panel, which closes the card while you have made no choice (4A).
- With Minimum OD above the diameter shown, both Outside diameter sides read the end-of-range words while the
  slider's handle sits at its bottom end (section 2, "(a), Min OD 45"; at 43.3 mm only the -1 mm side does).
- With a design file's value past a slider's end (the face gap above 5.0 mm, an axial override past the length range,
  the Minimum OD at its slider's top), both sides of that row read "at the end of its range" (as with Minimum OD 45).
- A design file or link can carry any finite value, and at a degenerate one the card draws numbers no slider reaches;
  no rule of the spec keeps them off the card (a test pins only that it draws them without fault), and the G-3 record
  renders them (its probes 12 and 13): yours to rule. At 2 magnets per ring (a degenerate polygon) it quotes a Space
  claim change of about 3e15 mm; with no back iron at 8 magnets and a calibration coefficient of 1e-9 its +2 side
  reads about +161 billion percent (1.44 N·m from about 9e-10), and with 1e-300 a 300-digit percent; at a variation
  just under 1, or a magnet Br of 1e-6 T, every side's N·m reads +0.00 or -0.00 beside its percent (T-9's cap, which
  no design of the spec reaches); at a calibration coefficient of 1e300 the card's lines run past 4,000 characters
  while the readouts above read about 2.8e300 N·m; and at a Required torque of 1e-300 the readout's gap shows a
  300-digit percent, as before this slice. A rule could bound the Size tab's number format (beyond it, the readouts'
  four significant digits), check a loaded file's values against what the model can mean, or keep such numbers off
  the card.
- Two sides of a row can read the same N·m and percent while only one flips Hot-low torque (section 2, "T-14 one
  flip").
- The first line under Magnets per ring keeps its sentence with its last clause reworded (the spec's R-5), and the two
  Auto lines under it are gone (T-8); section 3 shows both.
- BL-065 stays open: the card's percents read the model's end factor, and BL-065's notes say which rows to re-run.

The plan review's answers (the controller took each recommended option on your behalf; from the plan's head):
```
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
awk '/^## Plan review/{on=1} on && /^## / && !/^## Plan review/{exit} on' docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md | sed 's/^## /#### /' >> $W/decisions.md
tail -n 5 $W/decisions.md
```
   Expected: the plan's `## Plan review ...` section appended, its heading demoted to `#### `; empty output means the plan
   has no section that starts `## Plan review`: STOP and ask the controller where the plan review's answers are.

3. Write `$W/leftovers.md` with the Write tool, with this text; its section 7 is read from these, never from memory:
```markdown
## 7. Leftovers from the reviews (none blocks the merge; your call)

<every deferred minor of the build ledger, every should-fix item the slice's G-3 record leaves open and every
suggestion of final-review.md, one bullet each: what, where (file:line), the offered fix; "on screen" when it changes a
shown text (it then needs its own G-3 re-review); or the single line "None.">

## 8. Checks done

The G-3 physics review of every card string (docs/analyses/2026-10-14-magcoupler-help-review.md, Slice 3 torque card),
the whole-branch review, gate.sh GATE PASS with the Python oracle running (gate 12), the web build (the card is in the
bundle), the native timing of the Size tab's two caches with the card, the web smoke test with the card step and the
browser frame check, and a second browser drag at the example design with Auto on the ring size.
```
```bash
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
L=.superpowers/sdd/2026-10-20-magcoupler-torque-card/progress.md
[ -f "$L" ] || L=$W/progress.md
head -n 1 $L
grep -n 'minor (deferred)' $L
grep -n -i -A40 'suggestion' $W/final-review.md | head -80
awk '/^## /{on=($0=="## Slice 3 torque card")} on' docs/analyses/2026-10-14-magcoupler-help-review.md | grep -n -i -E 'should-fix|SF-[0-9]|open'
```
   `$L` is the build ledger (the superpowers:subagent-driven-development skill's `progress.md` for this plan's file
   name). Its first line names the plan file (`# SDD ledger — plan: ...2026-10-20-magcoupler-torque-card.md`); the grep
   prints every deferred minor of T1-T10, or nothing. When the plan's workspace has no ledger, `$L` falls back to
   `$W/progress.md`. If `head` still fails (no ledger in either), STOP and ask the controller where the task reviews'
   minor findings were recorded: section 7 is never written as "None." for want of the file.

4. Write `$W/owner_summary.py` with the Write tool, exactly:
```python
"""Slice-3 T10: the owner's STOP sheet, every on-screen text and number the branch adds or
changes, before (main) -> after (the branch), with the release's measurements.

Both sides are read by the same tools from each checkout, in BEFORE_DIR and AFTER_DIR:
screen.tsv (the screen-dump tool; the branch's has the card and gap records),
guidance-strings.txt and overlay-strings.txt (the crate's review dumps) and notes.txt
(notes_for_review). The shipped literals that are no dump's (the readout gap's test cases, the
exploration lines, the label the card replaces) are read from each checkout's source, main's
through `git show BASE:path`. The measurements are the controller's files: the gui-smoke result
(SMOKE_JSON) and the timing probe's output (TIMING_TXT). Nothing here is typed from a draft; the
decisions, leftovers and checks sections are the controller's files (EXTRA.md ...).

Usage:
  python -I owner_summary.py BEFORE_DIR AFTER_DIR BASE STAT_LINE SMOKE_JSON TIMING_TXT OUT [EXTRA.md ...]
(run from the repository root: it reads magcoupling-rs/src and calls git)
"""

import collections
import json
import os
import re
import subprocess
import sys

TAG = re.compile(r"^\[([^\]]+)\] (.*)$")
DESIGN_KINDS = ("fix", "note", "problem")
# The card at the spec's probe designs (section 10.1 step 6), in its order.
PROBES = [
    "(a)", "(b)", "(c)", "(e)", "(e2)", "mixed", "reverse", "short cavity", "temperature edge",
    "(c), f_cal_original -0.95", "(a), Min OD 45", "(a), Min OD 45: 43.3 mm", "T-14 one flip",
    "R-13 boss plateau", "short magnets", "shown does not fit", "variation 1.0", "zero torque",
]
# The guidance dump's Size-tab tags for the lines under Magnets per ring (section 3 shows them).
MAGNET_TAGS = ("size tab/under Magnets per ring/", "size tab/magnets/")
GAP_CASE = re.compile(
    r'torque_gap_text\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*,\s*\(\s*"([^"]*)"\.to_owned\(\)\s*,\s*(true|false)\s*\)')
STR_CONST = r'const {name}: &str =\s*"((?:[^"\\]|\\.)*)";'


def read_screen(path):
    """kind -> key -> text, in file order."""
    kinds = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if line:
                kind, key, text = line.split("\t", 2)
                kinds.setdefault(kind, collections.OrderedDict())[key] = text
    return kinds


def read_guidance(path):
    """tag -> its texts, in file order (a fix or note tag holds several)."""
    tags = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = TAG.match(line.rstrip("\r\n"))
            if m:
                tags.setdefault(m.group(1), []).append(m.group(2))
    return tags


def read_overlay(path):
    """(table, path) -> the text the overlay shows."""
    rows = collections.OrderedDict()
    key = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = TAG.match(line)
            if m:
                key = (m.group(1), m.group(2))
            elif key is not None and line.startswith("  now: "):
                rows[key] = line[len("  now: "):]
    return rows


def read_notes(path):
    """note id -> its block (title, sources and every probe design), as lines."""
    blocks = collections.OrderedDict()
    note = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            m = re.match(r"^==== (\S+) \(", line)
            if m:
                note = m.group(1)
                blocks[note] = []
            elif note is not None:
                blocks[note].append(line)
    return blocks


def source(rev, path):
    """A file of the checkout (`rev` None) or of commit `rev`, as text."""
    if rev is None:
        with open(path, encoding="utf-8") as f:
            return f.read()
    return subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True,
                          encoding="utf-8", check=True).stdout


def str_const(text, name):
    """The value of `const NAME: &str = "...";` in a Rust source, or None."""
    m = re.search(STR_CONST.format(name=name), text)
    return m.group(1).replace('\\"', '"') if m else None


def cell(text):
    if text is None or text == "":
        return "(none)"
    return text.replace("|", "\\|").replace("\n", "<br>")


def changes(before, after):
    """(key, before, after) for every key whose text differs, before's order first."""
    keys = list(before) + [k for k in after if k not in before]
    return [(k, before.get(k), after.get(k)) for k in keys if before.get(k) != after.get(k)]


def table(rows, head=("Where", "Before (main)", "After (this branch)")):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return out


def get(screen, kind, key):
    """The text of one screen record, or None where the checkout has none."""
    return screen.get(kind, {}).get(key)


def card_lines(screen, design):
    """The card's records at `design`, in order: (NN, text)."""
    prefix = design + " "
    return [(k[len(prefix):], v) for k, v in screen.get("card", {}).items()
            if k.startswith(prefix) and re.fullmatch(r"\d\d", k[len(prefix):])]


def timing(path):
    """The timing probe's lines, design -> its figures."""
    out = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"^(\([^)]+\)): (.*)$", line.rstrip("\r\n"))
            if m:
                out[m.group(1)] = m.group(2)
    return out


def main(argv):
    before_dir, after_dir, base, stat, smoke_json, timing_txt, out_path = argv[1:8]
    extras = argv[8:]
    sb = read_screen(f"{before_dir}/screen.tsv")
    sa = read_screen(f"{after_dir}/screen.tsv")
    gb = read_guidance(f"{before_dir}/guidance-strings.txt")
    ga = read_guidance(f"{after_dir}/guidance-strings.txt")
    ob = read_overlay(f"{before_dir}/overlay-strings.txt")
    oa = read_overlay(f"{after_dir}/overlay-strings.txt")
    nb = read_notes(f"{before_dir}/notes.txt")
    na = read_notes(f"{after_dir}/notes.txt")
    with open(smoke_json, encoding="utf-8") as f:
        smoke = json.load(f)
    calc = smoke.get("magcoupling") or {}
    probe = timing(timing_txt)
    panel_b = source(base, "magcoupling-rs/src/gui/panel.rs")
    panel_a = source(None, "magcoupling-rs/src/gui/panel.rs")
    card_a = source(None, "magcoupling-rs/src/gui/torque_card.rs")
    sizes_b = source(base, "magcoupling-rs/src/gui/size_controls.rs")
    sizes_a = source(None, "magcoupling-rs/src/gui/size_controls.rs")
    lines = []
    w = lines.append

    w("# Guidance slice 3, the torque card: what changes on screen")
    w("")
    w(f"Branch `magcoupling/slice3-torque-card` against `main` ({base}): {stat.strip()}. The full")
    w("diff is attached (slice3.diff), and its production and docs part (slice3-src.diff). Every")
    w("text below is what the site shows, with the shaft-clamp section paused, read by the same")
    w("tools from each checkout.")
    w("")

    # 1. At a glance.
    label_b = re.search(r'ui\.label\("(Build checks are shown[^"]*)"\)', panel_b)
    label_a = re.search(r'ui\.label\("(Build checks are shown[^"]*)"\)', panel_a)
    native = "; ".join(f"{k}: {v.split(';')[0]}" for k, v in probe.items()) or None
    browser = None
    if calc.get("drag_frames") is not None:
        browser = (f"{calc.get('drag_frames')} drag frames: median {calc.get('drag_frame_ms_median')} ms, "
                   f"95th percentile {calc.get('drag_frame_ms_p95')} ms, slowest {calc.get('drag_frame_ms_max')} ms "
                   f"(within one 60 fps frame: {calc.get('drag_frames_within_60fps')}; the drag moved the gap: {calc.get('drag_moved')})")
    ring_path = os.path.join(os.path.dirname(smoke_json), "smoke-ring.json")
    ring = {}
    if os.path.exists(ring_path):
        with open(ring_path, encoding="utf-8") as f:
            ring = json.load(f)
    browser_ring = None
    if ring.get("drag_frames") is not None:
        browser_ring = (f"{ring.get('drag_frames')} drag frames: median {ring.get('drag_frame_ms_median')} ms, "
                        f"95th percentile {ring.get('drag_frame_ms_p95')} ms, slowest {ring.get('drag_frame_ms_max')} ms "
                        f"(the drag moved the gap: {ring.get('drag_moved')})")
    glance = [
        ("The card's heading, between the torque readouts and the drawing", "(no card)", get(sa, "card", "(a) 00")),
        ("The line the card replaces there", label_b.group(1) if label_b else None,
         "(still drawn: a finding)" if label_a else "(removed: the card is drawn there)"),
        ("Dashboard status line", get(sb, "status", "paused"), get(sa, "status", "paused")),
        ("The readout's gap at the example design", "(main's format: see section 3)", get(sa, "gap", "(a)")),
        ("Native time per changed design, both caches with the card (min / median, ms)",
         "under 1.6 ms (docs/ai/02-system.yaml at main)", native),
        ("A drag frame in the browser (gap slider, 1280 x 800, local server)", "(not measured)", browser),
        ("A drag frame in the browser at the example design with Auto on the ring size", "(not measured)", browser_ring),
        ("gui-smoke on the local server", None, f"passed: {smoke.get('passed')}; the card step: {calc.get('torque_card')}"),
        ("Inputs, results, share link of the default design",
         f"{get(sb, 'count', 'inputs')} inputs, {get(sb, 'count', 'results')} results, {get(sb, 'share link', 'default design')}",
         f"{get(sa, 'count', 'inputs')} inputs, {get(sa, 'count', 'results')} results, {get(sa, 'share link', 'default design')}"),
    ]
    w("## 1. At a glance")
    w("")
    lines += table([(f"{i}. {a}", b, c) for i, (a, b, c) in enumerate(glance, 1)])
    w("")

    # 2. The card at the probe designs.
    w("## 2. The card at the probe designs (the heading, then each line as drawn)")
    w("")
    for design in PROBES:
        found = card_lines(sa, design)
        w(f"### {design}")
        w("")
        w("```text")
        if found:
            lines.extend(f"{nn}  {text}" for nn, text in found)
        else:
            w("(no card records)")
        w("```")
        w("")

    # 3. The Size tab's other changed texts.
    w("## 3. The Size tab's other texts that change")
    w("")
    w("### The lines under Magnets per ring (the guidance dump's Size-tab designs)")
    w("")
    rows = []
    for tag in list(gb) + [t for t in ga if t not in gb]:
        if tag.startswith(MAGNET_TAGS) and gb.get(tag) != ga.get(tag):
            rows.append((tag, "<br>".join(gb.get(tag, [])) or None, "<br>".join(ga.get(tag, [])) or None))
    lines += table(rows) if rows else ["No change."]
    w("")
    w("### The first line under Magnets per ring, by Auto state")
    w("")
    rows = changes(sb.get("size tab", {}), sa.get("size tab", {}))
    lines += table(rows) if rows else ["No change."]
    w("")
    w("### The readout's gap to the requirement")
    w("")
    w("At the designs of the card's spec (the branch; main has no public gap text to dump):")
    w("")
    lines += table([(k, None, v) for k, v in sa.get("gap", {}).items()])
    w("")
    w("The shipped test cases (`torque_gap_text(hot-low, required)`), as each checkout's test pins them:")
    w("")
    cases_b = {(m.group(1), m.group(2)): m.group(3) for m in GAP_CASE.finditer(sizes_b)}
    cases_a = {(m.group(1), m.group(2)): m.group(3) for m in GAP_CASE.finditer(sizes_a)}
    lines += table([(f"hot-low {k[0]} N·m, required {k[1]} N·m", cases_b.get(k), cases_a.get(k))
                    for k in list(cases_b) + [k for k in cases_a if k not in cases_b]])
    w("")
    w("### The exploration lines above the card")
    w("")
    before_lines = re.findall(r'"(Exploration only: [^"]*)"', panel_b)
    eo = str_const(panel_a, "EXPLORATION_ONLY")
    fit_word = str_const(card_a, "SIDE_DOES_NOT_FIT")
    key_word = str_const(card_a, "KEYWAY_BREAKS_THROUGH")
    built = {
        'format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter.")': (eo, fit_word, " at this diameter."),
        'format!("{EXPLORATION_ONLY}{KEYWAY_BREAKS_THROUGH}.")': (eo, key_word, "."),
    }
    after_lines = []
    for code, (head, word, tail) in built.items():
        ok = code in panel_a and head is not None and word is not None
        after_lines.append(f"{head}{word}{tail}" if ok else f"(built otherwise: read gui/panel.rs for {code})")
    lines += table([(f"line {i + 1}", b, a) for i, (b, a) in
                    enumerate(zip(before_lines + [None] * 2, after_lines))])
    w("")

    # 4. Everything else, before -> after.
    w("## 4. Every other text or number that changes, before -> after")
    w("")
    card_texts = set(sa.get("card", {}).values())
    guide_rows = []
    for tag in list(gb) + [t for t in ga if t not in gb]:
        if tag.startswith(MAGNET_TAGS):
            continue
        b, a = gb.get(tag), ga.get(tag)
        # The card's own lines, which Task 5 adds to the guidance dump under tags that name the
        # card, and any line the screen dump's card records hold: section 2 shows the card.
        new_card_lines = b is None and a and ("card" in tag.lower() or all(t in card_texts for t in a))
        if b == a or new_card_lines:
            continue
        guide_rows.append((tag, "<br>".join(b or []) or None, "<br>".join(a or []) or None))
    sections = [("Guides and the Size tab (the guidance dump; the card's own lines are in section 2)", guide_rows)]
    sections.append(("On-screen overlay labels and helps",
                     [(f"{t} {p}", b, a) for (t, p), b, a in changes(ob, oa)]))
    for kind, title in (
        ("row", "Dashboard rows"), ("row hover", "Dashboard row hovers"),
        ("guide label", "Check guides' labels"), ("guide why", "Check guides' Why sentences"),
        ("stored 3D", "The stored-3D label and its hover"),
        ("input section", "Input groups and sections"), ("input", "Inputs"), ("result", "Results"),
        ("assumption", "Assumptions view"), ("note", "Teaching notes (title, shown or hidden)"),
        ("start here", "Start here"), ("verdict", "Shaft-joint verdicts at their probes"),
        ("corner", "The BL-067 corner"), ("bond screen", "The bond screen"),
    ):
        sections.append((title, changes(sb.get(kind, {}), sa.get(kind, {}))))
    for title, rows in sections:
        w(f"### {title}")
        w("")
        lines += table(rows) if rows else ["No change."]
        w("")
    changed_notes = [n for n in nb if n in na and nb[n] != na[n]]
    new_notes = [n for n in na if n not in nb]
    w(f"Teaching notes whose drawn text changed at a probe design: {', '.join(changed_notes) or 'none'}; new notes: {', '.join(new_notes) or 'none'}.")
    w("")

    # 5. Counts.
    w("## 5. Pinned counts that moved")
    w("")
    rows = changes(sb.get("count", {}), sa.get("count", {}))
    lines += table(rows, ("Count", "Before (main)", "After (this branch)")) if rows else ["None."]
    w("")

    for extra in extras:
        with open(extra, encoding="utf-8") as f:
            w(f.read().rstrip())
        w("")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    empty = [r for r in glance if r[2] in (None, "")]
    missing = [d for d in PROBES if not card_lines(sa, d)]
    print(f"{out_path}: {len(lines)} lines; at-a-glance rows with no after text: {len(empty)}; "
          f"probe designs with no card: {len(missing)}; gap cases before/after: {len(cases_b)}/{len(cases_a)}")


if __name__ == "__main__":
    main(sys.argv)
```

5. Build the sheet and check it:
```bash
D=magcoupling-rs/target/release-slice3
W=.superpowers/sdd/2026-10-19-magcoupler-slice3
python -I $W/owner_summary.py $D/before $D/after "$(git rev-parse --short "$(git merge-base main HEAD)")" "$(tail -n 1 $W/slice3.stat)" $W/smoke-local.json $W/timing-T10.txt $W/owner-summary.md $W/decisions.md $W/leftovers.md
OUT=$W/owner-summary.md
grep -c '^## ' $OUT
grep -c '(no card records)' $OUT
grep -c 'a finding' $OUT
sed -n '/^## 5\. Pinned counts that moved/,/^## 6/p' $OUT
```
Expected: the script prints `.../owner-summary.md: <N> lines; at-a-glance rows with no after text: 0; probe designs with
no card: 0; gap cases before/after: 7/7`; `8` (sections 1 to 5 and the extras' 6 to 8); `0`; `0` (the label the card
replaces is gone); section 5 reads `None.` (no input, result, check, guide, note, row or section count moves). Then read
the file once in full: section 1's first row the shipped heading, its native row Step 6's figures and its two browser rows
Step 7's (the smoke drag and the ring drag); section 2 every probe design's card; section 3 the Magnets per ring lines without the N±2 lines and with the
reworded first line, the Auto lines gone, the gap at one decimal, the exploration lines as built from the constants (a
`(built otherwise ...)` cell: read `gui/panel.rs` and write the lines into the sheet by hand from the code); section 4
`No change.` in every part (the guidance dump's tags that name the card hold the card's own lines, which section 2
shows). Any other change is a finding: read the owning task before sending anything. (Measured with the script's own `read_screen`
and `card_lines` on Task 9's screen dump: every probe design finds its card, 6 records at each rows-form design,
7 at `(b)`, 2 at each one-line form, and `(a), Min OD 45: 43.3 mm` and `(a), Min OD 45` 6 each.)

6. Send the owner three files with SendUserFile: `$W/owner-summary.md`, `$W/slice3.diff` (the full diff) and
   `$W/slice3-src.diff` (the production and docs part).

7. Ask with AskUserQuestion (recommended options first), with these two questions (ask the second only if section 7
   lists at least one leftover):
   - Question 1: "1. Slice 3, the torque card (BL-073), is built and checked: whole-branch review passed, GATE PASS with
     gate 12, the web build carries the card, gui-smoke passed with the browser frame check (95th percentile <P> ms, and <P2>
     ms at the ring design, of a 16.7 ms frame). The sheet and the diff are attached; section 6 lists the rulings made on your night-run approval.
     Merge to main and push? The push deploys colesorkness.com/tools/magcoupler, so this is also the slice's deploy
     authorization."
     - Option "1A OK: merge, push and deploy (recommended)": the texts and numbers are as reviewed, and every ruling of
       section 6 stands.
     - Option "1B Change first": name the rows, texts or rulings of the sheet to change; each change is a must-fix of
       Step 3 (with its re-review), then Steps 4 to 10 again.
     - Option "1C Hold": do not merge yet; the branch stays as it is.
   - Question 2: "2. The reviews' leftovers (section 7 of the sheet; none blocks the merge): what should happen to
     them?"
     - Option "2A Fold them in before the merge (recommended)": BL-059's and BL-064's precedent; each docs item is
       edited, each on-screen item gets its G-3 re-review, then Steps 4 to 9 run again before the merge.
     - Option "2B File them as one backlog item and merge now".
     - Option "2C Leave them in the review records and merge now".

Do nothing further until the owner answers in their own message. 1B: run the changes as Step 3 says, then Steps 4 to
10 again. 1C: stop here and report the branch state. 1A with 2A: fold the leftovers in as Step 3 does (one commit per
item, `msg/T10-lo-<n>.txt`, subject `fix(magcoupler): <what>, from the slice-3 review leftovers`; each on-screen item
with its G-3 re-review appended under `## Slice 3 torque card` as `### Re-review of the review leftovers
(YYYY-MM-DD)`), run Steps 4 to 9 again, then Step 11. 1A with 2B: file one backlog item (next free id after BL-075,
`risk` the highest of its items, `status: open`) listing them, in Step 11's commit. 1A with 2C: Step 11.

- [ ] **Step 11: After the OK only: record it, merge, push**

1. `docs/ai/backlog.yaml`, BL-073: replace `  status: in_progress` (the only one in the file, BL-073's) with
   `  status: fixed`, and append to its `notes:` string, right before the closing `"` (anchor: `waiting for the owner's
   review of the diff and the deploy authorization."` with its closing quote, BL-073's alone):
   ` The owner reviewed the diff and gave the OK, with the deploy authorization, on YYYY-MM-DD (AGENTS.md risk: physics); merged to main and pushed the same day.`
   With 2A, write instead
   ` The owner reviewed the diff and gave the OK, with the deploy authorization, on YYYY-MM-DD (AGENTS.md risk: physics), after the review leftovers were folded in at the owner's choice (2A); merged to main and pushed the same day.`
2. `docs/ai/05-update-tracker.md`: in the T10 release bullet of Step 8, replace
   `Not yet done: the owner's review of the diff, the merge, the push and the deploy check.` with
   `The owner reviewed the diff and gave the OK, with the deploy authorization, on YYYY-MM-DD; merged to main and pushed; the deploy run and the live smoke are checked in the session that pushed.`
3. `docs/ai/04-memory.yaml`:
   - in the slice-3 active issue, replace
     `Magcoupler torque card (guidance slice 3, BL-073, branch magcoupling/slice3-torque-card; spec` with
     `Magcoupler torque card (guidance slice 3, BL-073; spec`, and replace
     `built (T1-T9); the release checks (T10: whole-branch review, GATE PASS with gate 12, web build, the native timing and gui-smoke with the browser frame check) and then the owner's review of the diff come before the merge and the push (risk: physics).`
     with `released YYYY-MM-DD after the owner's review of the diff.`;
   - in the NEXT item, replace `then slice 3, the torque card, built on branch magcoupling/slice3-torque-card` with
     `slice 3, the torque card, shipped YYYY-MM-DD`.
4. `docs/ai/02-system.yaml`: replace `built on branch magcoupling/slice3-torque-card` with `merged and deployed YYYY-MM-DD`.
5. Check the YAML still parses:
```bash
python - <<'EOF'
import yaml
b = yaml.safe_load(open('docs/ai/backlog.yaml', encoding='utf-8'))
ids = [x['id'] for x in b]
assert ids[:21] == ['BL-0%d' % n for n in range(56, 76)] + ['BL-055'], ids[:21]
keys = ('id', 'title', 'dimension', 'risk', 'evidence', 'acceptance', 'priority', 'status', 'notes')
assert all(tuple(x) == keys for x in b), [x['id'] for x in b if tuple(x) != keys]
item = {x['id']: x for x in b}
assert item['BL-073']['status'] == 'fixed' and not [x['id'] for x in b if x['status'] == 'in_progress']
assert 'merged to main and pushed the same day.' in item['BL-073']['notes']
m = yaml.safe_load(open('docs/ai/04-memory.yaml', encoding='utf-8'))
assert any(i.startswith('Magcoupler torque card (guidance slice 3, BL-073; spec') for i in m['active_issues'])
assert any('slice 3, the torque card, shipped' in q for q in m['open_questions'])
yaml.safe_load(open('docs/ai/03-structure.yaml', encoding='utf-8'))
lines = open('docs/ai/02-system.yaml', encoding='utf-8').read().splitlines()
hit = [line for line in lines if '(guidance slice 3, BL-073, owner decisions 1A-17A) merged and deployed' in line]
assert len(hit) == 1, len(hit)
yaml.safe_load('k:\n' + hit[0])
assert not any('built on branch magcoupling/slice3-torque-card' in line for line in lines)
print(len(b), len(m['active_issues']), len(m['open_questions']))
EOF
```
   Expected: `75 6 66`. With 2B's item, placed after BL-075, the first assertion's `range(56, 76)` becomes
   `range(56, 77)` and the expected line `76 6 66`.
6. Write `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-b.txt`:
```text
docs(magcoupler): record the owner's OK of slice 3, the torque card (BL-073)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
   and `.superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-merge.txt`:
```text
Merge magcoupling/slice3-torque-card: the magcoupler torque card (guidance slice 3, BL-073)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01UW1Rcc7PfLxGH7NQJ5srin
```
7. Commit on the branch, then merge and push:
```bash
git add docs/ai/backlog.yaml docs/ai/05-update-tracker.md docs/ai/04-memory.yaml docs/ai/02-system.yaml
git status --short
git commit -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-b.txt
git checkout main
git pull --ff-only origin main
git merge --no-ff magcoupling/slice3-torque-card -F .superpowers/sdd/2026-10-19-magcoupler-slice3/msg/T10-merge.txt
git status --short
git log --oneline -1
git push origin main
```
Expected: the four files staged and the owner's file untracked; the pull prints `Already up to date.` (main was 69fa505,
as `origin/main`); if `main` moved, STOP: merge `main` into the branch, then Steps 4 to 10 again, then here; the merge
has no conflicts; `git status` shows only the owner's untracked file; the log's subject is
`Merge magcoupling/slice3-torque-card: the magcoupler torque card (guidance slice 3, BL-073)`; the push succeeds
(`69fa505..<merge> main -> main`).

- [ ] **Step 12: Watch the deploy; prove the new bundle is live; live smoke**

Run:
```bash
git rev-parse HEAD
gh run list --workflow deploy-web.yml --branch main --limit 1 --json databaseId,headSha,status,conclusion
```
Expected: the run's `headSha` equals `git rev-parse HEAD` (if the newest run is still 69fa505's, wait a few seconds and
list again). Then, with that run's `databaseId`:
```bash
gh run watch <databaseId> --exit-status
```
Expected: the run ends `completed` with exit status 0 (BL-064's run, 39323543717, took the same path). On a failure:
`gh run view <databaseId> --log-failed`, report it to the owner, and fix on a new branch with its own red test; never
push again blindly.

Then the live bundle (Vercel serves the new files only once the deploy is done):
```bash
D=magcoupling-rs/target/release-slice3
curl -sS --compressed -o $D/live.wasm -w '%{http_code} %{size_download}\n' https://colesorkness.com/tools/magcoupler/magcoupling-web_bg.wasm
HEADING="$(sed -n '/const CARD_HEADING:/,/;/p' magcoupling-rs/src/gui/torque_card.rs | grep -o '"[^"]*"' | head -1 | tr -d '"')"
OLD_AUTO="$(git show 69fa505:magcoupling-rs/src/gui/guidance.rs | sed -n '/^pub const AUTO_RESIZES_RING:/,/;/p' | grep -o '"[^"]*"' | head -1 | tr -d '"')"
echo "[$HEADING]"
[ -n "$HEADING" ] && grep -c -a -F "$HEADING" $D/live.wasm
[ -n "$OLD_AUTO" ] && grep -c -a -F "$OLD_AUTO" $D/live.wasm
```
Expected: `200 <size>`; the heading in brackets, as in Step 5; at least `1`; `0` (before the deploy the live file gave
`0` and `1`, checked 2026-10-20). If it still gives the old counts, wait a minute and fetch again, at most three times;
then report to the owner.

Then invoke the `gui-smoke` skill with the arguments `{"url": "https://colesorkness.com"}` (the same steps against the
live site, with no `frame_check`: the frame fields are recorded and do not gate `passed`; `/` redirects to `/tools/`
there). Expected: `passed: true` with every calculator field, `torque_card` included, `true` and no console error, as
in Step 7. A failed field: report the smoke's notes and screenshots to the owner (the site is already live).

- [ ] **Step 13: Delete the local branch; update the memory; report**

1. Run:
```bash
git branch -d magcoupling/slice3-torque-card
git branch --show-current
git status --short
```
Expected: `Deleted branch magcoupling/slice3-torque-card (was <hash>).`; `main`; only the owner's untracked file. (The
branch was never pushed, so there is no remote branch to delete. Do not touch the other worktrees.)
2. The auto-memory (outside the repo), with the merge's short hash (`git rev-parse --short HEAD`), the run id of Step 12
   and the figures of Step 8.1:
   - In `C:/Users/Cole/.claude/projects/C--Users-Cole-source-repos-linkage-simulation/memory/project_magcoupler_guidance_overhaul.md`,
     replace the whole bullet that starts `- Slice 3 (torque sensitivity card) STARTED 2026-10-19` with:
```text
- Slice 3 torque card SHIPPED YYYY-MM-DD (BL-073): the Size tab's torque card (gui::torque_card), one fixed step each way of each Size control (+/-1 mm OD at the Minimum OD, +/-1 mm length, +/-0.1 mm gap, +/-2 magnets) with its hot-low change in N.m and % and what it costs (size change, flips, failing checks' quantities), words where no number shows; the N+/-2 lines moved into it from under Magnets per ring; one Size-tab number format (one decimal of percent). Owner decisions 1A-17A; the night-run rulings (O-1 widen, O-2 quote none, test 0 on the readouts' rule with ShownNoHotLow, CardView keyed on the screen, the wrapped header, a count side below zero shown, R-13 recorded) and the plan review's answers stand unless the owner reverses them. Spec docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md, plan docs/superpowers/plans/2026-10-20-magcoupler-torque-card.md. Browser drag frame p95 <P> ms in <BROWSER>; native both caches <A> ms per changed design. The owner reviewed the diff and said OK, with the deploy authorization; merge <short hash> on main, pushed; deploy run <run id> green; the live wasm carries the card's heading; live gui-smoke PASS; branch deleted; the gitignored workspaces .superpowers/sdd/2026-10-19-magcoupler-slice3/ (owner-decisions.md, final-review.md, owner-summary.md, the screen-dump tool, the timing probe) and 2026-10-20-magcoupler-torque-card/ (the build ledger) are kept. Open: BL-065 (the card's percents re-run when it lands), BL-074 (the length slider's first-move jump; the owner decides), BL-075 (the BL-064 fold-in's notes). Next on the roadmap: guided lessons, plot captions and drawing titles; re-confirm the deploy authorization per slice.
```
   - In `MEMORY.md` there, on the `project_magcoupler_guidance_overhaul.md` line, replace
     `slice 3 brainstorming started 2026-10-19` with `slice 3 torque card shipped YYYY-MM-DD (<short hash>)`. If the
     controller reworded that line during the night, replace its slice-3 phrase the same way.
3. Tell the owner, in a few lines: merged and deployed (the merge's short hash), the deploy run's link
   (`https://github.com/coleas33/linkage_sim/actions/runs/<run id>`), the live wasm check and the live smoke result,
   and what stays open: BL-065 (the card's percents), BL-074 and BL-075, then guided lessons.

**Existing tests Task 10 moves:** none (a fix-wave item names any test it moves in its own commit; Step 7b adds one
test to `tests/size_controls.rs` and moves none).

---

## Deviation index

PD-1 to PD-6 are in the head. The drafters' deviations follow, renumbered from PD-7 (ruling C-8), each with its drafter's
label, where it lives and its ruling; then the second integration's departures from the adopted text, PD-58 to PD-62.
"Adopted" means the controller accepted it as drafted (ruling C-8, or the ruling named).

- **PD-7** (DA-1; T2 Step 4, T4 Step 3, T7): `WALL_NEEDED` and `COLD_HIGH` are `pub const`s in `gui/guidance.rs` beside
  `check_terms`, not in `gui/torque_card.rs` as spec 4.1 says: `Quoted.name` reads them from Task 2 on, before the card
  module exists. Task 4 imports them and defines no copies; Task 7's G-3 review sets their words there. Adopted (C-9;
  critique A-15 carries it into Task 7's text).
- **PD-8** (DA-2; T1 Step 5, T4 Step 3): two helpers the spec does not name, `shared_change_decimals` (T-14's row rule)
  and `reads_not_zero` (the one test of "a digit that is not zero", shared by `change_decimals` and the card's `shows`).
  Adopted.
- **PD-9** (DA-3; T1 Step 5, T2 Step 4, T4 Step 4): `shared_change_decimals` and `check_terms` carry
  `expect(dead_code)` until Task 4 gives them a production caller, so each task is green alone under `clippy -D
  warnings`; Task 4's commit deletes both. Adopted (C-9).
- **PD-10** (DA-4; T2 Step 3): the ban list's matcher is named `ranking_words_in(text)`: the listed words a text says,
  whole, in any case, with an s added or taken off. Adopted.
- **PD-11** (DA-5; T2 Step 1): test 1a sits in a new `mod tests` at the end of `gui/test_support.rs`. Its own copies of
  the check paths are superseded at Task 4, which imports guidance's paths (PD-22). Adopted as PD-22 amends it.
- **PD-12** (DA-6; T1 Step 8, T2 Step 8): docs lines each task updates beyond spec 9.2's list, because its code makes them
  stale; Tasks 5 and 9 take their Old text after Tasks 1 and 2. Adopted (C-9).
- **PD-13** (DA-7; T1 Step 3, T2 Step 1): tests beyond the spec's list: the decimals search's own test, test 13's zero,
  non-finite, past-the-cap and size cases, the gap test's margin and shortfall cases, and `slider_minimum`'s own test.
  Adopted.
- **PD-14** (DA-8; T2 Step 4): `check_terms` returns no terms for a path that is not a guide's check and never panics in
  production; the fix tests' `terms()` keeps its panic. Adopted.
- **PD-15** (DA-9; T2 Step 6): spec 2.1 names no failing checks for "reverse"; the table's doc and test 1a record the
  measured ones (Hot-low torque, Running clearance and Cup wall failing, Temperature limits at caution). Adopted.
- **PD-16** (DA-10; T1 header): from Task 1 to Task 4 the N±2 lines read one-decimal percents (decision 10B's one
  percent rule), until Task 5 moves them into the card; no commit loses the N±2 information. Adopted.
- **PD-17** (DV-B1; T3 Step 1 (c)): `rows()` and `card_designs()` land in Task 3, and the table lives in
  `gui/test_support.rs` (`pub(crate)`: `CardDesign` with `card()` and `side()`, `card_at`, `side_at`), not in the card's
  test module, because the review dump in `gui/guidance.rs`'s tests reads the same table. Adopted (C-9).
- **PD-18** (DV-B2; T4 Step 1, test 18): test 18 triples every number of a copy of the card and checks that each line
  keeps its skeleton while every drawn number differs, where spec 8.4 says "move each number". Adopted.
- **PD-19** (DV-B3; T4 Step 1, test 7): no reachable design has Temperature limits red (`verdict_level` maps the
  temperature verdict to Good or Caution), so test 7 asserts that its terms quote nothing and its level is never Bad.
  Adopted (C-9; critique A-17 carries it into Task 10's STOP sheet).
- **PD-20** (DV-B4; T3 Step 1): `count_step`'s `i64::MAX` and `i64::MIN` pins live in
  `count_step_keeps_even_counts_inside_the_slider_and_never_overflows`; test 3 asserts that the two overflow designs read
  `ShownDoesNotFit`. Adopted; the Task 9 sentence that credited the pins elsewhere left with R11 (C-2).
- **PD-21** (DV-B5; T4 Step 3, T5 Step 3): the two neighbour words move into the card in Task 4 as `SIDE_DOES_NOT_FIT`
  and `SIDE_NO_PREDICTION`, and `Neighbour::line` reads them until Task 5; Task 5's Old blocks are Task 4's text.
  Adopted (C-1: I-2 to I-6).
- **PD-22** (DV-B6; T4 Steps 1 (b) and 4): guidance's eight check-path constants and `result_number` become `pub(crate)`; the card
  reads quoted quantities through `result_number`, and test 1a imports the paths in place of its six copies. Adopted.
- **PD-23** (DV-B7; T3 Step 1, T4 Step 1): tests beyond the spec: `each_control_is_a_size_tab_slider_with_its_step` and
  `a_side_reads_a_number_exactly_when_sizing_counts_its_design` (Task 3), and tests 3 and 4 extended to the drawn line
  (Task 4). Adopted.
- **PD-24** (DV-B8; T3 Step 1, T4 Step 1): `card_designs()` holds spec 2.1's rows and Edges, 2.3's count overflows, (a)
  at Minimum OD 43.3 mm and variation 1.5: 50 designs, 52 once Task 4 adds T-14's one-flip design and R-13's boss
  plateau (plan review 2.2). The not-vacuous pins move with them: `NUMBER_SIDES` 277, quotes 517, texts 1056, numbers
  1049. Adopted, as plan review 2.2 amends it.
- **PD-25** (DV-B9; T4 Step 1, test 2a): (c)'s count `-2` side is drawn from its own -41.0516 % ("-41.1 %"), not
  spec 2.3's rounded -41.05 %. Adopted.
- **PD-26** (DV-B10; T3 Step 3, T4 Step 3): `card()` carries `#[expect(unused_variables)]` on `clamps_paused` until
  Task 4's flips read it; Task 4 deletes it. Adopted.
- **PD-27** (DV-B11; T3 Step 3, T4 Step 3): visibilities the spec does not state: `Words::text` public,
  `Row::torque_decimals` private; the face gap's path is guidance's one `FACE_GAP`, made `pub(crate)` (critique A-12), in
  place of the card's own constant. Adopted, as A-12 amends it.
- **PD-28** (DC-1; T3 Step 1 (c)): the review dump reads `card_designs()` from `gui/test_support.rs`, since every test
  module is private (the same placement as PD-17). Adopted (C-9).
- **PD-29** (DC-2; T5 Step 1 (b)): `end_effect_edge()` moves from guidance's tests to `gui/test_support.rs`, read by
  guidance's glyph test and the count row's moved no-prediction test. Adopted.
- **PD-30** (DC-3; T5 Step 4): `MagcouplingPanel::auto()`, one expression for the controls and the card. Adopted.
- **PD-31** (DC-4; T5 Step 4, T6 Step 4): private names the spec does not give: `TORQUE_CARD_ID`, `CardFrame`,
  `card_default`, and `CardView`'s whole-point screen fields. Adopted.
- **PD-32** (DC-5; T6 Step 3): `geometry_view::MIN_DRAWING_HEIGHT` becomes `pub(crate)`, so tests compare with it, not
  a typed 160. Adopted.
- **PD-33** (DC-6; T6 Step 1, test 21): test 21 adds a 1280 x 2000 case, where only rule 1 closes the card; at the
  spec's 1280 x 800 rule 3 alone closes it. Adopted.
- **PD-34** (DC-7; T5 Step 1 (c)): the seven moved neighbour tests are renamed for the count row (the old names name the
  removed `Neighbour`). Adopted.
- **PD-35** (DC-8; T5 Step 1): the four fit-line tests spec 6.3 lists pin no line count and pass unchanged; 6.3's
  counts are pinned in the first-line, size-tab, owner-case and slider-end tests instead. Adopted.
- **PD-36** (DC-9; T5 Step 1 (d)): Task 5 also edits Task 2's `slider_minimum` test, a third `Consequences::of` call
  site spec 6.3 does not list. Adopted.
- **PD-37** (DC-10; T5 Step 1 (a)): the cache test keeps its spec name, gains the pause as a key part and pins that the
  pause reaches the card at `clamp_at_its_limit()`. Adopted.
- **PD-38** (DC-11; T5 Step 1 (a)): `no_guide_text_calls_the_ring_radius_ring_size` lists the card's three Auto lines in
  place of the two removed constants. Adopted.
- **PD-39** (DC-12; T6 Step 1, test 23): test 23's assumption edit sets the input between frames, and its drags are
  pointer drags on each slider's rail. Adopted.
- **PD-40** (DC-13; T6 Step 1, test 20): test 20 asserts outright that rule 3 alone would open the one-line form at
  930 x 800. Adopted.
- **PD-41** (DC-14; T5 Step 1 (e), test 10): test 10 checks "the held row's computed sides" as the whole cached card
  equal before and after Auto turns off, with the held line drawn and then gone. Adopted.
- **PD-42** (D-G-1; T8 Steps 3, 6 and 7): `node --check` is run on a wrapped copy of `gui-smoke.js` (the plain check
  passes a broken file), and a stub-agent run executes the script's own code. Adopted.
- **PD-43** (D-G-2; T8 Step 5 (D), (I)): the script, not the agent, computes `drag_frames_within_60fps` and gates
  `passed` on it only with `frame_check`. Adopted.
- **PD-44** (D-G-3; T3 Step 1 (c), T4 Step 1 (b)): `CardDesign` and `card_designs()` are `pub(crate)` in
  `gui/test_support.rs`, so the review dump and the card's tests share one table (as PD-17). Adopted (C-9).
- **PD-45** (D-G-4; T5 Step 6, T7): the review dump draws each design in its own Auto state, and all three Auto states at
  the four designs whose form differs under Auto. Adopted (C-9).
- **PD-46** (D-G-5; T5 Step 6): the review dump also prints the readout's gap at (a), (b), (c), (e) and (e2), for the
  reviewer's context. Adopted.
- **PD-47** (D-G-6; T8 Step 8): Task 8 edits the README's gui-smoke paragraph (~:618-631) and the `src/app.rs` test row
  (~:697), which spec 9.2 does not list. Adopted (C-9).
- **PD-48** (D-G-7; T8 Step 5 (B), (I)): the schema's details: `drag_frames` an integer, the four drag numbers optional,
  the three booleans required, and the fallback object's values. Adopted.
- **PD-49** (D-G-8; T7, the table "What this task applies"): a G-3 finding on the owner's 9A words, a reused shipped
  string, the number format or a new rule goes to the STOP sheet, not into Task 7's change. Adopted (C-9).
- **PD-50** (DEV-1; T9 Steps 2 and 3): the screen dump and `check_shipped.py` are built in Task 9, which needs them for
  its quotes; Task 10 re-runs them and adds `main`'s side. Adopted.
- **PD-51** (DEV-2; T9 Step 5): BL-073 is written `status: in_progress`, not spec 9.1's `open`; Task 10 Step 11 sets
  `fixed` after the OK. Adopted.
- **PD-52** (DEV-3; T9 Step 7, S6): one `known_limitations` entry in `02-system.yaml` that spec 9.2 does not list (a
  step, not a slope; BL-065's end factor; R-13's plateau). Adopted.
- **PD-53** (DEV-4; T5 Step 7 (n)): the 02-system :116 rewrite also corrects the FixCache clause, "only when the inputs
  or the pause change". Adopted, moved into Task 5 by I-8 (C-1, C-8).
- **PD-54** (DEV-5; T9 Step 2, E5): the screen dump's designs are `card_designs()`'s, under its keys (critique A-10),
  with its two count overflows (plan review 2.2); T-14's one-flip design and R-13's plateau are in the table itself.
  Adopted, as A-10 and plan review 2.2 amend it.
- **PD-55** (DEV-6; T9 Step 4): the module doc of `gui/size_controls.rs` stays true after slice 3, so no sweep pattern
  covers it. Adopted.
- **PD-56** (DEV-7; T10 Step 8.1): 02-system :106's count is computed, not measured: 22 for the card, the probe's fit
  search (29 at the example design) and 3, about 54. Adopted.
- **PD-57** (DEV-8; T9 Step 2, sub-step 4): the dump's `card` record count is derived from the card's forms; measured
  after Task 8 it is 290 at 52 designs. Adopted, with the measured figure.
- **PD-58** (second integration; T9 Step 3 sub-step 4, T10 Step 9.1): one check script, `same_designs.py`, ties the
  screen dump's cards to the review dump's, design by design; plan review 1.2's own `check_card_dumps.py` is not
  written. Ruling 1.2A ("met by 2.2A's one check script").
- **PD-59** (second integration; T3 Step 1 (f), T5 Step 4): the panel's `FreeVariable` import moves into its test module
  at Task 3, where `sizing_range` leaves the production code no use of it (`clippy -D warnings`), and back at Task 5's
  `auto()`. Adopted: each task stays green alone.
- **PD-60** (second integration; T1 Step 5): `TORQUE_DECIMALS`' doc names spec section 5's rule and its two readers, not
  critique A-14's "its readouts' `fixed(x, 2)`", since the readouts show four significant digits. Adopted.
- **PD-61** (second integration; T9 Step 2, E5): plan review 2.2's one-line `v.push(...)` of the count overflows is
  written in rustfmt's layout, so the tool stays rustfmt-clean apart from BL-064's own line. Adopted.
- **PD-62** (second integration; T9 Step 9): the tracker entry names the review's shared pieces where they ship
  (`TORQUE_DECIMALS`, `SliderMapping`, the table's two designs and `sizing_range`, `FRAME_TIMER`). Adopted: the owner's
  rule that docs ship with the code.

## Per-task test counts

Measured on the second integration's commits, every task green alone (both suites, both clippy runs with
`-D warnings`, `cargo fmt -- --check`). Tasks 7 and 10 had no measured commit. "App lib" is `cargo test --features app
--lib`; "Lib" is the same without `--features app`. Every row also has: app lib 6 ignored, lib 1 ignored, `explain` 1
ignored, doc-tests 1 passed and 1 ignored.

| After | App lib | Lib | explain with app | explain without app | Other binaries that move |
|---|---|---|---|---|---|
| 69fa505 (the base) | 928 | 233 | 64 | 63 | (the base) |
| T1 | 930 | 233 | 64 | 63 | none |
| T2 | 935 | 234 | 64 | 63 | none |
| T3 | 943 | 234 | 64 | 63 | none |
| T4 | 957 | 234 | 64 | 63 | none |
| T5 | 961 | 234 | 64 | 63 | none |
| T6 | 971 | 234 | 64 | 63 | none |
| T7 | 971, plus one per assertion test a must-fix adds | 234 | 64 | 63 | none |
| T8 | as after T7 (971 with no T7 test) | 234 | 64 | 63 | none |
| T9 | as after T8 (docs only) | as after T8 | 64 | 63 | none |
| T10 | as after T9, plus a test per fix-wave item | as after T9, plus any | 64 | 63 | `size_controls` 39 (from 38), only if Step 7b runs |

These never move: assumptions 13, deviations 54, differential 19, gearbox 13, grades 12, material_library 4,
material_links 14, parity 4, python_schema 7 (gate 12's parity untouched), robustness 13, schema 11, shaft_joint 40,
sizing 45, static_data 8; the two app binaries run 0 tests.

## Amendment ledger

Every amendment, ruling and review item, and where it lives in the tasks above (or why it lives nowhere).

- **I-1** (C-1 over A-1; PD-4): T1 header, Interfaces, Step 6 (the `gui::torque_gap_text` re-export) and Step 8 (f)
  (03-structure's clause); T9 Step 1 and Step 8's note (the re-export is Task 1's).
- **I-2** (C-1 over A-2): T5 Step 1 (a) (guidance's test block with Task 3's odd-count assertion) and the moved-tests
  table's `at_either_end_...` row.
- **I-3** (C-1 over A-2; meets A-11): T5 Step 1 (c), the moved count-row tests on Task 4's test module (`card_at`,
  `side_at`, the anchor 1049).
- **I-4** (C-1 over A-2): T5 Step 3, guidance's `torque_card` import as Task 4 leaves it.
- **I-5** (C-1 over A-2): T5 Step 3, the neighbour types' block as Task 4 leaves it.
- **I-6** (C-1 over A-2): T5 Step 3, `neighbours()` and `prediction()` as Task 3 leaves them.
- **I-7** (C-1; it is A-3): T5 Step 7 (q), 03-structure's `test_support.rs` clause.
- **I-8** (C-1 over A-4 and A-5; PD-53): T5 Step 7: "Twenty edits", (n)'s whole cache clause of 02-system :116, (j2) and
  (r2).
- **I-9** (C-1 over A-6): T5 header (the neighbour-code bullet; the Facts bullet on Task 4's text, A-2's deleted fact),
  Interfaces, Steps 2, 5, 6 and 8, Step 9's message and the moved-tests table, each figure as J-5 measured it again.
- **I-10** (C-1 over A-6): T6 Steps 2, 5 and 7, each figure as J-6 measured it again.
- **I-11** (C-1 over A-6): T7 Step 1's expected output (976 filtered out, the grep lines, the coverage loop) and "What
  the sheet then holds" (564 = 305 + 259; 260), as J-7 measured them again.
- **I-12** (C-1 over A-6): T8 Steps 2, 7 and 9 (976 filtered out; 971 and 234), as J-8 measured them again.
- **I-13** (C-1 over A-8's skip-if strings): T9 Steps 6 and 7, the two-pattern skip-ifs of R3, R6, R7, R8, S2 and S3;
  T9 Step 4's owner table (one owner per line: Task 5's text, Task 9's edits its guards).
- **I-14**: T9 Step 4, `CSTALE`'s two patterns and the measured "Expected after T1-T8".
- **I-15**: T10 Step 7b.1, the test's last assertion in rustfmt's layout.
- **I-16** (C-1 over A-9): T10 Step 7b.3, the two edits of `gui/torque_card.rs`, with 2.3's import and 2.1's size
  reading (J-10), the guard sentence and the measured line.
- **J-1** (A-14; PD-60): T1 header, Interfaces, Step 5 (`TORQUE_DECIMALS`), Step 6 (the gap reads it), Step 7 (the
  fourth planted fault), Steps 9 and 10.
- **J-2** (plan review 2.3): T2 header, Files, Interfaces, Step 1 (b) (`SliderMapping`'s test), Step 2 (65 errors),
  Step 5 (`SliderMapping`; `controls_ui`'s loops apply it), Steps 7, 9 and 10, the existing-tests row.
- **J-3** (A-12, A-13, A-20, A-24, plan review 2.3; PD-59): T3 header, Files, Interfaces, Step 1 (a) (A-20's test), (c)
  (`smoke_inputs`), (e) (`FACE_GAP`) and (f) (`sizing_range`; the panel's import), Step 2 (110 errors), Steps 5 to 8.
- **J-4** (A-14, A-21, plan review 1.1, 2.1, 2.2, 2.3): T4 header, Files, Interfaces, Step 1 (the four new tests; the two
  table designs and `boss_plateau`; the moved pins), Step 2 (122 errors), Step 3 (c), Step 5 (G13, G14 and the
  figures), Steps 6 to 8, the tests table.
- **J-5** (A-19, A-24, plan review 2.2, 3.2, 4.4): T5 header, Interfaces, Step 1 (the drag and Required tests, test
  10's Minimum OD part, the import), Step 2 (49 errors), Step 4 (the card's `shown`; PD-59's import back), Step 5 (44
  passed; faults (g) to (i)), Steps 6 to 9, the tests table.
- **J-6** (A-24, plan review 3.1, 3.3, 3.4, 4.4): T6 header, Files, Interfaces, Step 1 (`smoke_design`, test 22's
  floor, the resize test, test 24), Step 2 (27 errors), Step 4 (the call, the signature, the body's least height), Step
  5 (21 passed; faults (i) to (k)), Steps 7 and 8.
- **J-7** (plan review 2.2): T7 Step 1 (the loop's two keys, the measured expected output); its brief items are C-5's
  and 2.2's text, below.
- **J-8** (plan review 4.1): T8 header, Interfaces, Steps 1, 2, 3 (the stub's timer block), 5, 7, 8, 9 and 10.
- **J-9** (A-10, C-2, C-3, I-13, I-14, plan review 1.2, 2.2; PD-58, PD-61, PD-62): T9 Files, Step 2 (E5's keys, count
  overflows and doc; the expected counts), Step 3 (K3's key; sub-step 4, `same_designs.py`), Step 4, Steps 6 to 9 (R11
  and C4 deleted; the tracker's names), Steps 10 to 12.
- **J-10** (A-10, plan review 2.1, 2.2, 2.3, 4.2, 4.3; C-6): T10 Step 6 (the probe, its Expected), Step 7 (3b's
  measured sizes), Step 7b (the card's two edits, the measured line), Steps 8 and 11 (anchors measured), Step 9.1, Step
  10.4 (`PROBES`).
- **A-1** (must): lives in I-1 (C-1).
- **A-2** (must): I-2 to I-6 (C-1); its deletion of "If Task 3's text differs, remove what Task 4 leaves": T5 header.
- **A-3** (must): I-7 (C-1).
- **A-4** (must): I-8 (C-1); its guard: T9 Step 4's `STALE` gains `which call them` (C-3).
- **A-5** (must): I-8 (C-1), T5 Step 7 (n).
- **A-6** (must): I-9 to I-12 (C-1), re-measured by J-5 to J-8.
- **A-7** (must; C-2): T10 Step 1, at least 10 commits since dae024f.
- **A-8** (must): its skip-if strings lose to I-13 (C-1); its remainder (C-2): T9 Step 10's marker is Task 5's
  "one key in three parts, slice 3", and R11 and C4 are deleted from T9 Steps 6 and 8. A-8's own marker text lives
  nowhere (C-2 names Task 5's text).
- **A-9** (must): I-16 (C-1), as J-10 amends it: T10 Step 7b.3.
- **A-10** (C-4): T9 Step 2 (E5's keys), Step 3 (K3's key); T10 Step 10.4 (`PROBES`); PD-54.
- **A-11** (C-4): met by I-3, T5 Step 1 (c).
- **A-12** (C-4): T3 Step 1 (e) (guidance's `FACE_GAP` `pub(crate)`), the card and the table import it; T2 Step 5's gap
  mapping; PD-27.
- **A-13** (C-4): T3 Step 1 (f) (`gui::sizing::sizing_range`), the panel's method and the table's `auto_shown` call it;
  T10 Step 1's third diff.
- **A-14** (C-4; PD-60): T1 Steps 5 to 7 and 10 (J-1); T4 header, Interfaces and Step 3 (c) (J-4).
- **A-15** (C-5): T7 Step 3's D1 bullet (the five readers); T7's Files, Interfaces and strings table name
  `gui/guidance.rs`.
- **A-16** (C-5): T7 Step 2's brief: the reachable-input sentence, D5 widened, probe (12).
- **A-17** (C-5): T10 Step 10.2, `decisions.md` item 2.
- **A-18** (C-6 as plan review 4.2 and 4.3 rewrite it): T10 Step 6 (the probe's 6 magnets and Auto ring) and Step 7
  (3b, the browser drag at the ring design). Its projection script lives nowhere: C-6 drops it.
- **A-19** (C-4): T5 Step 1 (e) (`a_drag_computes_the_card_once_for_each_design_it_draws`), Step 5 fault (i), header and
  tests table.
- **A-20** (C-4): T3 Step 1 (a) (`a_design_shown_past_a_slider_s_end_reads_the_end_of_its_range`).
- **A-21** (C-4): T4 Step 1 (a) (the T-14 and R-13 tests, on the table's keys by plan review 2.2) and the tests table.
- **A-22** (C-5, widened by plan review 1.1): T10 Step 10.2, `decisions.md`'s two bullets (past a slider's end; a
  design file's degenerate values) and the T-9 bullet.
- **A-23** (C-5): the head's "## Plan review rulings" title (verbatim head); T10 Step 10.3's ledger and its fallback.
- **A-24** (C-4): T3 Step 1 (c) (`smoke_inputs()`), the table's (b); T5 Interfaces, Step 1 (test 10 and the drag test);
  T6 Interfaces and Step 1 (`smoke_design()`).
- **C-1**: I-1 to I-16, as above.
- **C-2**: A-7 and A-8's remainder, as above.
- **C-3**: A-4's guard, T9 Step 4 (measured: the phrase matches no docs line after Task 8, so no narrower phrase).
- **C-4**: A-10 to A-14, A-19 to A-21 and A-24, as above, each measured again by J-1 to J-10.
- **C-5**: A-15, A-16, A-17, A-22 and A-23, as above.
- **C-6**: T10 Steps 6 and 7 (plan review 4.2 and 4.3); the head's Review Focus item 1 and PD-3. The projection script
  lives nowhere (dropped by this ruling).
- **C-7**: T4 Step 1 (the robustness test, plan review 1.1), T7 Step 2 (probes 12 and 13), T10 Step 10.2 (the widened
  bullet). The critic's guard test for a count of 2 lives nowhere: it lands with the owner's rule.
- **C-8**: the Deviation index, PD-7 to PD-57; DEV-4 in T5 Step 7 (n) (PD-53).
- **C-9**: each question where its ruling lands: `WALL_NEEDED` and `COLD_HIGH` (PD-7); the `expect` attributes (PD-9);
  Tasks 5 and 9 on Tasks 1 and 2's text (PD-12); the 02-system parse error left (T5 Step 7, T9 Step 10, each line parsed
  alone); test 1a's imported paths (PD-22); the table in `gui/test_support.rs` (PD-17, PD-28, PD-44); the dump's Auto
  states (PD-45); G-3 findings to the STOP sheet (PD-49); T8's README lines (PD-47); T10's gui-smoke runs (Steps 7 and
  12); the public surface (T3 to T5 Interfaces); T10's timing designs (Step 6); no red Temperature limits case (PD-19,
  T10 Step 10.2 item 2).
- **C-10**: the head's "## Plan review rulings"; each option as below.
- **Plan review 1.1** (1.1A): T4 Step 1 (`the_card_never_panics_at_the_values_a_design_file_can_carry`), Step 5 (G13),
  Step 6, the tests table; T7 Step 2 (the reachable-input sentence, probe 13); T10 Step 10.2 (the widened bullet, the T-9
  bullet).
- **Plan review 1.2** (1.2A): met by 2.2's one script (PD-58): T9 Step 2 (E5's doc line), Step 3 sub-step 4; T10 Step
  9.1. Its `check_card_dumps.py`, its Files entry and its T10 Step 9.3 command live nowhere (ruling 1.2A).
- **Plan review 2.1** (2.1A): T4 Step 1 (`a_step_to_a_design_without_size_controls_draws_no_size_change`), Step 5 (G14),
  the tests table; T10 Step 7b.3 (the size reading, the guard sentence, the measured line) and Step 7b.5 (the message).
- **Plan review 2.2** (2.2A; PD-5): T4 Step 1 (the two designs, `boss_plateau`, test 1a's pin, the moved pins) and Step
  6; T5 Step 1 (c) (1049) and Step 6 (260 and 564); T7 Steps 1 and 2 (the loop's keys, trap 4, probe 10); T9 Steps 2
  and 3; T10 Step 9.1.
- **Plan review 2.3** (2.3A; PD-1): T2 Steps 1, 2, 5, 7, 9 and 10 (`SliderMapping`, its test, `controls_ui`); T3 (the
  card's step, test 1's oracle and the table through it); T4's Old blocks; T9 Step 9 (the tracker); T10 Step 2 item (3)
  and Step 7b.3 (the import).
- **Plan review 3.1** (3.1A): T6 header, Step 1 (test 22's floor), Step 4 (`.min_scrolled_height(cap)`), Step 5 fault
  (i).
- **Plan review 3.2** (3.2A): T5 Step 1 (test 10's Minimum OD part), Step 5 fault (g), Step 7 (g).
- **Plan review 3.3** (3.3A): T6 Files, Step 1 (the resize test), Step 5 (its filter and fault (j)), Steps 7 and 8.
- **Plan review 3.4** (3.4A): T6 Step 1 (test 24 turns the mouse wheel), Step 5 fault (k).
- **Plan review 4.1** (4.1A): T8 header, Interfaces, Steps 1 to 3, 5, 7, 8 and 10 (`FRAME_TIMER`); T9 Step 9; T10 Step
  2 item (10) and Step 7 (drawn frames; 3b's extraction). Its plan-head sentence for PD-3 is the head's (verbatim).
- **Plan review 4.2** (4.2A): T10 Files and Rules (the four records), Interfaces, Step 6 (the ring payload), Step 7 (3b),
  Step 8 (8.1, 8.4, 8.5), Step 10 (`owner_summary.py`'s ring row, question 1). Its "delete C-6's projection block"
  lives nowhere: the drafts never carried that block. Its head edits are the head's (verbatim).
- **Plan review 4.3** (4.3A): T10 Step 6 (rounds; the quiet-machine item; the probe; the Expected), Step 7 item 1, Step
  9.2 (the background build from Step 8 on).
- **Plan review 4.4** (4.4A; PD-6): T5 header, Interfaces, Step 1 (`a_required_edit_reaches_the_card_in_its_own_frame`),
  Step 4 (`view_ui` builds the design once), Step 5 fault (h), Step 7; T6 Interfaces and Step 4; T10 Step 2 item (4).
- **RF-1** (A-18): T10 Steps 6 and 7, as C-6 is rewritten; the critique's projection lives nowhere.
- **RF-2** (A-19): T5's drag test.
- **RF-3** (A-20): T3's past-the-end test.
- **RF-4** (A-21): T4's two tests.
- **RF-5** (C-7): no rule; T4's robustness test, T7's probes 12 and 13, T10's STOP-sheet bullet. Its guard test for a
  count of 2 lives nowhere until the owner rules.
- **Assembly** (decided for the spec, the head's rulings and the owner's decisions; no question asked):
  - the deviations are numbered from PD-7, as the head says (plan review 4.1's "from PD-5" is superseded);
  - T9 Step 4's owner table names Task 5 for the lines its guarded edits now guard (I-13's one owner per line);
  - T10 Step 1's diffs read as measured: `plain.rs` alone, then `gui/sizing.rs` alone (A-13);
  - T10 Step 2's reviewer checks name `SliderMapping` (2.3), the design shown built once (4.4) and `FRAME_TIMER` (4.1);
  - T5 Step 6 and T7 Step 1 quote the measured grep lines; T4's G13 quotes only the measured panic;
  - pointers into the drafting workspace (the contract, the drafters' questions) and stand-in provenance are removed;
    the drafters' labels in the task text are the PD numbers above.
