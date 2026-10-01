# Magcoupling Addendum A-3: Explainability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the engine side of the Addendum A equation explorer on the A-2 engine: an evaluable equation record for every result in decision 31's scope (the five chains, the dashboard and the geometry callouts) and for every term those results need to reach the inputs, proven against the engine by a drift guard; the dependency graph and "used by"; the A3 traceability test over every assumption with the styling data; and the 17 A4 teaching notes behind a physics-review gate.

**Architecture:** One text per explained result is both the display and the evaluation: a small formula markup whose terms are input and result paths. The M4 typesetter draws its parse tree and the drift guard evaluates the same tree over the engine's own term values (the parity rule, at the defaults and at every differential case under 16 augmentations, every value term of every record seen at two values), so the equation shown is provably the one that produced the number, with no Rust closure. The engine stays as ported: a value a formula needs that the engine computed but did not expose becomes a Rust-only result, a pure read that changes no formula. Records are one step deep and plain data (`const` arrays, one file per chain); `Registry::build()` parses and checks them once (each path read in one unit; nothing the typesetter would draw two ways) and gives `equation_for`, `used_by`, the dependency graph, term rows, the A3 term styles, a selector code's choice labels and the corrections upstream of a result, owned by the caller (no global state). Traceability is two one-sided tests (soundness for every input; sensitivity for every assumption, algebraic cancellations listed with reasons). Teaching notes are static data keyed by id that the panel shows only once a physics reviewer has signed them off.

**Tech Stack:** Rust 2024 edition, std only for the library (wasm32-unknown-unknown clean); `serde_json` (feature `float_roundtrip`) as the only dev-dependency; the vendored Python 3.12 oracle `reference/magcoupling-py/` run with `C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe` (only `--check`: no data file changes); Git Bash for every command.

**Spec:** `C:/Users/Cole/source/repos/lsim-mag-a3/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, Addendum A: A2 (equation explorer: the explanation layer, the drift guard; hover, the panel and the typesetter are M4's and consume this plan's registry), A3 ("Traceability"), A4 (teaching notes) and "Addendum testing" (the equation drift guard; the assumptions traceability test), with M4 "Layout" (the geometry callouts). Scope: decision 31 of `docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (section 7: the dashboard, the geometry callouts and the five chains; every other field shows its workbook cell). Decisions it builds on: Addendum A decisions 1 to 31, the A-1 plan's A1 to A13 (`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a1-data-physics.md`) and the A-2 plan's A2-1 to A2-9 (`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a2-parameters-sizing.md`). The notes are drafted from the M1 audit `docs/analyses/2026-09-29-magcoupling-math-audit.md`.

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-a3`, branch `magcoupling/addendum-a3`, created by Task 0 from the final head of `magcoupling/addendum-a2`. Every command uses absolute paths into it. Nothing is pushed.

## Decisions to confirm

**Confirmed (user, 2026-09-30): G1-G5 as recommended, and the two spec rewordings Task 17 applies (Q1: the A2 equation record's evaluable formula text replaces the 'eval closure'; Q5: the A3 traceability test's precise wording with the listed cancellations).** No task stops on a decision.

These questions arose while turning decision 31 and the spec's A2 to A4 into code; no approved decision settles them. This plan implements the recommended option of each (Task 0 records the user's answers; if the user picks another option, the task that implements it stops and escalates instead of improvising).

| Id | Question | Recommended (implemented here) | Alternatives | Tasks |
|---|---|---|---|---|
| G1 | Which numbers are "the geometry callouts" of decision 31? The report left them to be counted "when the A1 view is specified", and no one has specified that view yet. | **The numbers the M4 geometry view draws**: the face gap, the corner gap and the running clearance (spec M4 "Layout"), the clearance's two parts (the nominal sleeve-to-liner clearance and the adverse movement), and per axis the overshoot past the space claim with the dimension it measures (spec A1: "a red callout ... naming the overshoot in mm per axis"): 11 paths, 10 of them new, so the scope holds **169** paths, all explained | (b) only the three dimension callouts M4 names (face gap, corner gap, running clearance): 160 paths, and the overshoot callouts show only their values; (c) also the badge text (G2) | 13 |
| G2 | `housing.space_claim_check`, A-2's dashboard badge text ("Exceeds the space claim: diameter 1.23 mm over, overall length unknown"), joins the exceeded axes, which the markup can state only with a new list-join function | **Leave it unexplained** (it shows its label and value): it is a badge, not one of the report's 15 headline paths, and the three overshoot numbers it quotes are explained (G1) | (b) add a `join(sep, ...)` markup function that skips empty texts and record it (about 20 lines plus tests) | 13 |
| G3 | The order of the record batches. This plan's design lists them in the spec's "start here" order (torque, temperature, demagnetization, slip heating, clamps, dashboard), but the chains are entangled: the demagnetization margins read the peak magnet temperature (slip heating) and the slip-heating time to the limit reads the governing limit (demagnetization), so neither temperature nor demagnetization can flip to explained alone | **Follow the closure**: demagnetization block and governing limit (Task 8, its own drill-down test), then slip heating, which flips both chains (Task 9), then temperature, clamps, dashboard and the geometry callouts | (b) one large batch for demagnetization, slip heating and temperature together (about 140 records in one task, one reviewer sheet) | 8 to 13 |
| G4 | E20 checks the outer ring only when the correction is on; the records need each ring's block as Rust-only results, which need a value with E20 off | **Compute the outer ring's block with E20 off too** (from the C44 and C45 inputs, as the workbook would): it governs nothing and changes no workbook cell; one E20 unit test compares through a helper that blanks only the four new outer-ring fields | (b) NaN placeholders with E20 off: rejected, NaN is not equal to NaN, so the engine's whole-struct equality assertions would fail; (c) repeat the inner ring's values: misleading | 7 |
| G5 | Which notes, and the diagram kinds | **17 notes**: the spec's nine ideas, the physics behind each of the six A5 warnings, ferrite's cold-side demagnetization (spec A6 asks for that note) and the one-point calibration (the bench correction's cancellation of the assumed factor, found by the traceability test); `START_HERE` with 11 entries in the spec's order; **two more diagram kinds**, the knee with load lines (`DemagKnee`) and the first-order heating curve (`HeatingCurve`), beside the tracer's four, for M4 to paint | (b) 16 notes (no calibration note) and the four diagram kinds only | 15, 16 |
| G6 | The spec's A2 gives each record "an `eval` closure over term values" beside its display markup; with a closure per record the drift guard would prove the closure, not the formula shown | **The formula markup is the evaluation**: the drift guard evaluates the displayed tree (Task 2's evaluator); `Eval::Custom` stays an escape hatch the registry test pins at zero uses; Task 17 rewords the spec's A2 line to say so | (b) a Rust closure per record beside the markup (392 closures kept in step by hand, and the guard proves them, not what is shown); (c) keep the spec's wording and record the difference in this plan only | 2, 3, 17 |
| G7 | The spec's A3 test, "changing each assumption changes every dependent result and no independent one, using the equation registry's dependency graph", is false as worded: verdict text is piecewise constant, a `min` loser or an untaken branch has zero sensitivity at a design point, and some dependencies cancel algebraically (five found) | **Two one-sided checks**: soundness for every input (nothing outside its dependency set moves, and every input changes some result somewhere) and sensitivity for the 15 assumption inputs (every numeric result on the active path moves above rounding at some design point, the cancellations listed with reasons); Task 17 rewords the spec's A3 test line to that | (b) keep the wording and test it at the defaults with a per-pair exception list (hundreds of entries, brittle as records change); (c) soundness only (nothing checks that an assumption reaches what it should) | 4, 14, 17 |

Settled by this plan's design without a new decision (each is stated where it is implemented): a value a formula needs that the engine did not expose becomes a Rust-only result, a pure read, with the full suite green; every explained chain drills down to inputs (223 of the 392 equations are that closure); the symbols are T torque, σ shear stress, ϑ temperature, φ electrical angle, τ_p pole pitch, λ fill, `^{cal}` for the prototype, selectors as upright words; the explorer explains the engine with every correction on, and each record names the corrections it embodies (the M4 "corrected vs workbook" marker); the drift guard starts every differential case from the corrected defaults and runs every augmentation (each is load-bearing: without the harmonic sets the 7 to 11 records compare 0 with 0); the registry is built once and owned by the caller; records are defined over validated inputs (a test pins that inputs bypassing `set` never make a record panic); a repeated sub-expression (the Br(T) factor) stays in each record, as each record is one workbook cell; formatted verdicts are stated in the markup (`concat`, `fmt`, `fmtnum`); the registry refuses a formula that reads a path in two units or that the typesetter would draw two ways (an inline `a / b` or a Σ as a factor, a power of a symbol that already has a superscript), and the plain rendering parenthesizes a stacked fraction that is a factor (Task 3); the drift guard requires every value term of every record at two values (Task 3; Tasks 8 and 9 add the augmentations that needs), and soundness requires every input to change some result at some design point, two inputs no result reads listed with reasons (Task 4); the panel reads a selector code's choice labels and the corrections upstream of a result from the registry (Task 4).

## Global Constraints

Every task's requirements implicitly include this section.

- Crate `magcoupling-rs/` stays a sibling of `linkage-sim-rs/`: "No Cargo workspace conversion." The library is pure std with no `[dependencies]`; `cargo check --target wasm32-unknown-unknown --lib` stays clean. The `workbook-parity` feature is reachable only through the self dev-dependency.
- "One pure entry point `compute_all(&DesignInputs) -> Results`: no I/O, no global state, milliseconds per call." Its signature does not change; the explanation layer reads its results and never feeds it.
- Addendum A: "At default inputs and default assumptions every result stays workbook-exact, so the M2 parity and differential tests are unchanged." After every task: `Deviations::NONE` parity (1,149 checks: numbers 1e-9 relative, 1e-12 absolute, text exact), every differential file and every registry test pass unchanged, and `all_corrections_together_give_the_reviewed_headline` (the default headline with every correction on) passes.
- The engine stays as ported: no input is added, no formula changes, the deviation registry is not edited. Every new result is Rust-only (`out_rust_only`) and a pure read of a value the engine computes (Tasks 1 and 7), with one exception, decision G4: Task 7 computes the outer ring's demagnetization block anew with E20 off (from the C44 and C45 inputs, as the workbook would), a value that governs nothing and moves no workbook cell; A-2's "Rust-only" rules hold (no cell; skipped by the parity, metadata and differential tests).
- `reference/magcoupling-py/` is not edited; `gen_differential.py --check` must report the data current after every task. The Addendum A report, its data file and the M1 audit are evidence: never edited.
- Spec A2: "The equation shown is provably the one that produced the number." Every record's formula markup is what the drift guard evaluates; `the_registry_is_consistent` pins zero `Eval::Custom` records. A record states its result one step from its terms, never recomputes upstream (a value the engine computes but does not expose may be a `where` binding when it has no result of its own), takes unit, label and workbook cell from the result's metadata, and names the corrections it embodies. Every term has a symbol, no two paths share one, and every explained chain drills down to inputs.
- Drift guard tolerance is the parity rule (1e-9 relative, 1e-12 absolute; text exact; NaN equals NaN); records hold over every input `set` accepts.
- Every gate run: `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh` must end `GATE PASS` with no `SKIP gate 7/7` line; then `git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda` (the linkage tests rewrite those PNGs; never commit them).
- `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml` before every commit, and `cargo fmt --check` clean. Every block below is already in rustfmt's layout (it was produced by formatting); the record, table and symbol arrays carry `#[rustfmt::skip]`, so their one-line-per-entry layout stays.
- The replace blocks quote the A-2 end state as the A-2 plan's replay produced it, with LF line endings (Task 0 creates the worktree with a worktree-scoped `core.autocrlf=false` and checks it: this machine's system gitconfig sets it to true). If a block's text is not found, stop and escalate; do not improvise a match. The likely mismatch points are the blocks in `magcoupling-rs/README.md` and `docs/ai/*.yaml` (they quote whole table rows or paragraphs as context); the update tracker is edited by a script that anchors on its header only.
- Every commit message ends with exactly two lines: a `Co-Authored-By:` line naming the model that wrote the commit, then `Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9`. The commit blocks below write `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (the session model); a `sonnet` task replaces `Claude Opus 5.5` with its own model's name. Nothing is pushed.
- Docs change with code (repo rule, `docs/ai/01-meta.yaml`): each task updates `magcoupling-rs/README.md` where it changes layout, tests, the explorer section or a chain's status; Task 17 updates `docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, `05-update-tracker.md` and the spec's A2 and A3 wording. Read `docs/ai/*.yaml` before starting (Task 0).
- Model tiers (CLAUDE.md section 5): each task's **Model:** line. Tasks 0, 2 to 6, 8 to 14 and 17 give the exact code or text and run on `sonnet` (set `model: 'sonnet'`; a new `CANCELLATIONS` entry is a physics judgement, reviewed on the session model); Tasks 1, 7, 15 and 16 run on the session model (omit `model` on purpose, comment `// session model: physics`): a shared physics expression, a change to what the engine computes, and the teaching prose and its review. Every batch's physics review step runs on the session model. A `sonnet` attempt that ends `blocked`, leaves the gate red or is rejected in review escalates every retry to the session model.
- Physics review (spec "Testing and validation summary"; this plan's accuracy gate): each batch's review sheet is checked by a session-model reviewer whose prompt names the relevant M1 derivations before the batch commits; the notes are checked in Task 16 and only approved notes become `Reviewed`.
- Commit subjects: `feat(magcoupling-rs): ...` for Tasks 1 to 13 and 15, `test(magcoupling-rs): ...` for Task 14, `docs(magcoupling-rs): ...` for Task 16 and `docs(magcoupling): ...` for Task 17.

## Review Focus

Five input classes the spec implies but no task's red step exercises directly. Each names the tests that pin it and its task.

1. **Hard ferrite on a ring** (Y30 with its positive β from the grade, or a positive β typed into C45 with the coercivity source at 0, on a rated grade-mode ring or on a manual magnet with no grade and so no rating). Expected: the cold side decides (the cold onsets, the cold limit, the cold check from the ring with the higher cold limit); the hot onsets are infinite; the rating is the hot limit, and with no rating there is no hot limit and no torque at it (NaN); the other ring's hot block governs when it is lower. Tests: the drift guard's "ferrite inputs", "ferrite inputs, beta 0.002" and "unrated ferrite" augmentations with every `cases` arm taken and every value term at two values (Tasks 8 and 3), `each_ring_shows_its_own_block` (Task 7), and the traceability test's design points under the "ferrite inputs" and "unrated ferrite" augmentations, where the typed positive β is nudged around its value and to the slider's ends (Tasks 4 and 14). The "coercivity from the inputs" trace point runs at the default β (−0.005): it covers the inputs' coercivity, not ferrite.
2. **A bench drag entered, including a drag of exactly 0** (E13). Expected: the used and high losses become the drag power, the steady rises follow, a drag of 0 is no heating (the time to the limit "never"); the records reproduce the engine and soundness holds. Tests: the drift guard over the differential data, which vary the measured drag (Task 9: `used_W` and `high_W` take both arms), and the trace points "coercivity from the inputs, drag measured" and "drag measured as 0 (E13)" (Task 14).
3. **A hot-day start at or above the governing limit** (E12). Expected: the time to the limit is 0 and the critical drag 0, never negative. Tests: the drift guard (Task 9: the `time_to_limit_*` records take their E12 arm at some of its points) and the trace point "hot-day start above the limit (E12)" (Task 14).
4. **Non-default part materials** (a ferromagnetic back iron other than 4140, a non-ferromagnetic one, the sleeve and cap picks). Expected: the steel values, the body (hub, cup, boss) values, the sleeve's and the cap's are the pick's library values where the engine uses them and the inputs elsewhere; the free-space circuit for a non-ferromagnetic back iron. Tests: the augmentations "back iron 1018/304/6061" (Task 3) and "sleeve Ti, cap acetal" (Task 9), `the_materials_in_effect_are_the_picks_values` (Task 7), and the trace points under 1018, 6061 and the sleeve and cap picks (Task 14).
5. **Inputs that bypass `set`** (an invalid harmonic code, a selector code outside its choices, NaN written into the struct, as a design file or share link could hold). Expected: the records are defined over validated inputs (the GUI validates at its boundaries, decision D3), so a record may differ from the engine there, but evaluating it gives a value or an `EvalError`, never a panic; `validate()` names every bad input; an overshoot over a NaN reserve is NaN, as the engine's. Test: `records_never_panic_on_inputs_that_bypass_set` (Task 14).

## File Structure

Created (under `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/`):

| File | Responsibility | Task |
|---|---|---|
| `src/engine/explain/mod.rs` | The module and its re-exports | 2, 3, 4, 5, 7 |
| `src/engine/explain/markup.rs` | Formula and symbol grammar, parser, parse tree, visitors; the typesetting rules the registry enforces | 2, 3, 6 |
| `src/engine/explain/eval.rs` | Evaluator, `Trace`, `TermSource` | 2, 6 |
| `src/engine/explain/tables.rs` | The static engine tables a formula reads (`TABLE_FIELDS`, `lookup`) | 2, 7 |
| `src/engine/explain/record.rs` | `Record`, `Family`, `Eval`: the authoring form | 3 |
| `src/engine/explain/records/mod.rs` | `RECORDS`, `FAMILIES`: one entry per batch file | 3, 8 to 13 |
| `src/engine/explain/records/torque.rs` | Batch 1, the torque chain and its closure (127 equations) | 3 |
| `src/engine/explain/records/demagnetization.rs` | Batch 2: the demagnetization block, both rings, the cold side, the governing limit (39) | 8 |
| `src/engine/explain/records/slip_heating.rs` | Batch 3: slip losses, the thermal network, the mass and housing closure, the margins (84) | 9 |
| `src/engine/explain/records/temperature.rs` | Batch 4: torque against temperature, ratings, the summary (16) | 10 |
| `src/engine/explain/records/clamps.rs` | Batch 5: the screw table's families and the clamp scalars (100) | 11 |
| `src/engine/explain/records/dashboard.rs` | Batch 6: the headline paths outside the chains (17) | 12 |
| `src/engine/explain/records/geometry.rs` | Batch 7: the geometry callouts (9) | 13 |
| `src/engine/explain/registry.rs` | `Registry` (build and its checks, lookups, graph, term rows, styles, choice labels, upstream corrections), `Equation`, `Design`, `TermKind`, `TermRow`, `TermStyle`, `CONVERSIONS`, `RESULT_CHOICES` | 3, 4 |
| `src/engine/explain/symbols.rs` | `SYMBOLS`: display symbols of inputs and leaf terms | 3, 8, 9, 11 to 13 |
| `src/engine/explain/scope.rs` | `SCOPE`: decision 31's chains and the geometry callouts, status per chain | 3, 9 to 13 |
| `src/engine/explain/notes.rs` | `NOTES`, `START_HERE`, `Review`, `Diagram`, `note_for`: the A4 notes and the accuracy gate | 5, 15, 16 |
| `src/engine/explain/render.rs` | `plain`: the one-line text rendering (tooltip fallback, export, the review sheet) | 3, 6 |
| `tests/explain.rs` | Drift guard (with the term-level check), registry, scope and drill-down, A3 traceability and styling, the panel lookups, note links, release gate, review sheet | 3, 4, 5, 8, 9, 13, 14, 15 |

Modified: `src/engine/{model,calibration,sweeps}.rs` (Task 1), `src/engine/{temperature,materials}.rs` (Task 7), `src/engine/mod.rs` (Task 2), `tests/common/mod.rs` and `tests/differential.rs` (Task 3: the shared differential loader), `README.md` (every task), `docs/ai/{02-system,03-structure,04-memory}.yaml`, `docs/ai/05-update-tracker.md` and the spec (Task 17). No differential, parity, golden or Python file changes; `tests/data/input_schema.json` does not change (no input is added).

Order: the engine's torque terms first (Task 1), then the language (Task 2), the registry with the torque records and the drift guard (Task 3), traceability (Task 4) and the notes module (Task 5): that is the tracer slice, whole and green. Then the markup and engine additions the remaining chains need (Tasks 6, 7), the record batches in closure order (Tasks 8 to 13, decision G3), traceability made non-vacuous over every assumption once every chain is explained (Task 14), the notes (Task 15) and their review (Task 16), and the docs (Task 17).

## Verification record

This plan was replayed before it was handed over. The base was the post-A-2 tree (the A-2 plan's proven end state; it equals the A-2 replay's tree except the update tracker's dated A-2 entry, around which Task 17's tracker script anchors). Every task's blocks were applied in order to a fresh LF copy of it, with a dedicated cargo target directory, and every command the plan marks with an expected output was run and compared:

- **Red steps** failed as stated (compile errors naming the missing items for Tasks 1 to 7; the batches' structure tests listing the paths without a record; Task 15's notes tests on the stubs).
- **After every task**, the magcoupling gates passed: `cargo test`, `cargo clippy --all-targets -- -D warnings`, `cargo check --target wasm32-unknown-unknown --lib`, `gen_differential.py --check` with the oracle Python, and `cargo fmt --check`. (`gate.sh` itself also runs the linkage crate's gates 1 to 3, which this plan does not touch, and the oracle's pytest suite, which it does not change.)
- **Task 16**'s sign-off script ran as if every note were approved (17 notes reviewed), and the ignored release gate `release_notes_are_reviewed` then passed; Task 17's tracker script ran; `all_corrections_together_give_the_reviewed_headline` passed at the end.
- The plan's commits were produced by formatting each state with rustfmt, and the final `src/` and `tests/` equal, file for file, the tree in which the records were developed and proven.
- This replay is the second: the first was reviewed independently, every finding was fixed in the commit that owns it (the Self-review record lists them), and the whole plan was replayed again from the post-A-2 base with every check above.

| After task | Unit tests | `tests/explain.rs` | Equations (drift guard input sets) |
|---|---|---|---|
| 0 (base) | 138 | none | none |
| 1 | 139 | none | none |
| 2 | 159 | none | none |
| 3 | 162 | 5 passed, 1 ignored | 127 (40,693) |
| 4 | 162 | 9 passed, 1 ignored | 127 |
| 5 | 165 | 10 passed, 2 ignored | 127 |
| 6 | 167 | 10 passed, 2 ignored | 127 |
| 7 | 170 | 10 passed, 2 ignored | 127 |
| 8 | 170 | 11 passed, 2 ignored | 166 (50,866) |
| 9 | 170 | 11 passed, 2 ignored | 250 (57,648) |
| 10 to 13 | 170 | 11 passed, 2 ignored | 266, 366, 383, 392 |
| 14 to 17 | 170 | 12 passed, 2 ignored | 392 |

Every other binary keeps Task 0's counts throughout (parity 4, differential 19, deviations 54, and so on): no workbook cell, differential value or registry probe moves.

At the end: 392 equations (torque 127, demagnetization 39, slip heating 84, temperature 16, clamps 100, dashboard 17, geometry 9) over the 169 scope paths, all explained and every one drilling down to inputs; 223 of the 392 are the closure those paths need. 145 leaf symbols, no two paths sharing one; zero custom evaluations. The drift guard evaluates every equation at 57,648 input sets (the defaults, then 3,391 differential cases as generated and under 16 augmentations) in about 10 s of a debug build with 32 threads, every `cases` arm taken except the two listed unreachable and every value term of every record seen at two values. Soundness holds for all 172 inputs at 114 design points, each nudged a step both ways and to its slider's ends, and every input but the two no result reads (listed with reasons) changes some result there; at those points 4 of the dependency graph's 870 non-redundant edges could be dropped unseen (named in the Self-review record); sensitivity holds for the 15 assumption inputs with the 5 listed cancellations, and each assumption reaches and moves an explained result. The engine gained 48 Rust-only results, all reads (Task 1: 24 model and 7 calibration; Task 7: 8 temperature and 9 materials).

Not exercised by the replay: the physics reviews (each is a dispatched review whose findings the task acts on), the user's answers to the Decisions to confirm, and the linkage crate's own gates.

---

## Tasks

### Task 0: Worktree, baseline and context

**Model:** `sonnet` for Steps 1 to 6 (creating the worktree, running commands and reporting output; CLAUDE.md section 5). Step 7 is the controller's.

Verification only; nothing to write.

**Files:**
- none

**Interfaces:**
- Consumes: branch `magcoupling/addendum-a2` at its final head (the A-2 plan's Task 11 commit, `docs(magcoupling): Addendum A-2 status, invariants and open items`, reviewed and gate-green).
- Produces: the worktree `C:/Users/Cole/source/repos/lsim-mag-a3` on the new branch `magcoupling/addendum-a3` with LF line endings, a confirmed green baseline, and the controller's record of the user's answers to the Decisions to confirm.

- [ ] **Step 1: Create the worktree**

This machine's system gitconfig sets `core.autocrlf=true`, which would check the files out with CRLF; every replace block below quotes them with LF, as git stores them. So the worktree is created with LF and keeps a worktree-scoped `core.autocrlf=false` (`extensions.worktreeConfig` lets one worktree carry its own setting; the main checkout and the other worktrees keep theirs).

Run:

```bash
git -C C:/Users/Cole/source/repos/linkage_simulation log --oneline -1 magcoupling/addendum-a2
git -C C:/Users/Cole/source/repos/linkage_simulation config extensions.worktreeConfig true
git -C C:/Users/Cole/source/repos/linkage_simulation -c core.autocrlf=false worktree add -b magcoupling/addendum-a3 C:/Users/Cole/source/repos/lsim-mag-a3 magcoupling/addendum-a2
git -C C:/Users/Cole/source/repos/lsim-mag-a3 config --worktree core.autocrlf false
git -C C:/Users/Cole/source/repos/lsim-mag-a3 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: the `log` line is A-2's final commit, `docs(magcoupling): Addendum A-2 status, invariants and open items` (if it is not, A-2 has not finished: stop and escalate); `worktree add` prints `Preparing worktree (new branch 'magcoupling/addendum-a3')`; then `magcoupling/addendum-a3`; no status lines.

- [ ] **Step 2: Check the line endings**

No replace block would match a CRLF file, so check before the first edit.

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 config --get core.autocrlf; head -c 2000 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs | tr -cd '\r' | wc -c
```

Expected: `false` for `core.autocrlf` (the worktree's own setting) and `0` carriage returns. If the count is not 0: run `git -C C:/Users/Cole/source/repos/lsim-mag-a3 rm -rq --cached . && git -C C:/Users/Cole/source/repos/lsim-mag-a3 reset -q --hard` (it rewrites the files with the worktree's setting) and check again.

- [ ] **Step 3: Read the project context**

Read `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/01-meta.yaml`, `02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`; `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` whole (the Deviations and Translation rules sections are binding); the spec's Addendum A (`C:/Users/Cole/source/repos/lsim-mag-a3/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, sections A2, A3, A4 and "Addendum testing"); the verification report `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md` (section 7, "A2 scope", and decision 31); the M1 audit `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-29-magcoupling-math-audit.md` (the physics reviews cite its rows); and this plan's Decisions to confirm.

- [ ] **Step 4: Run the crate's tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```

Expected: in order (each `Running` line, then its result): unit tests `138 passed`; `tests\assumptions.rs` 8 passed; `tests\deviations.rs` 54 passed; `tests\differential.rs` 19 passed; `tests\grades.rs` 10 passed; `tests\material_library.rs` 4 passed; `tests\material_links.rs` 11 passed; `tests\parity.rs` 4 passed; `tests\python_schema.rs` 7 passed; `tests\robustness.rs` 12 passed; `tests\schema.rs` 7 passed; `tests\sizing.rs` 31 passed; `tests\static_data.rs` 8 passed; doc-tests `1 passed; 0 failed; 1 ignored`. These are the counts of the tree this plan was proven against (A-2's replay); if a count differs but every binary is `ok`, record the actual counts, continue, and read every later task's counts as offsets from them.

- [ ] **Step 5: Check the oracle Python**

Run:

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
cd C:/Users/Cole/source/repos/lsim-mag-a3/reference/magcoupling-py && C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe tools/gen_differential.py --check
```

Expected: `ls` prints the path; then `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`: no data file changes (the generator never passes a Rust-only input to Python). If `ls` fails, stop and escalate: every gate run of this plan uses that interpreter.

- [ ] **Step 6: Run the gate**

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task0.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 7: Decisions**

The controller asks the user the plan's **Decisions to confirm** (G1 to G7) before Task 1 and records the answers in the execution notes. Every task implements its decision's recommended option and names the decision in its intro; if the user picks another option, the task that implements it stops and escalates instead of improvising.

---

### Task 1: Expose the torque chain's terms as Rust-only results

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5): it moves the shared harmonic amplitude into one function (`Harmonic::amplitude`) that the shear stress and the pull-out both call, and the claim that every existing result stays bit for bit needs physics judgement. Physics reviewer: M1 audit rows M3, M4, M7 and E7.

The equation explorer's records (this plan's Architecture) need every term to be an input or a result path. The engine
already computes harmonics 7 to 11 for every design (`shear_stress` returns all six; `let [h1, h3, h5, ..] = h` drops the rest),
the per-harmonic torque-angle amplitude and the E7 angles; this task exposes them as Rust-only results. Each is a pure read of a
value the engine computes, and no formula changes: the one shared expression, the amplitude, moves into `Harmonic::amplitude`, which
`shear_stress`, `at_pull_out` and the new results all call, so their values are bit-identical. Parity, differential and schema tests
skip Rust-only results by design. The red test pins the reads against the engine's own law (each summed shear stress is its
amplitude at the pull-out angle; harmonics 7 to 11 have their parts whether or not they are summed).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs` (`Harmonic::amplitude`, `HALF_PITCH_RAD`, `at_pull_out` returning its angle; 24 Rust-only results; a test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/calibration.rs` (7 Rust-only results (`amp1_Pa`..`amp11_Pa`, `pullout_angle_rad`))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/sweeps.rs` (`let (h_pull, _) = at_pull_out(..)`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the model and calibration layout rows)

**Interfaces:**
- Consumes: `model::shear_stress` (all six harmonics computed, `[Harmonic; 6]`), `model::at_pull_out`, `model::peak_angle`, `calibration::compute`'s `amp_n` closure and `peak`.
- Produces (read by the records of Tasks 3 and 8 onward):
  - `pub const HALF_PITCH_RAD: f64 = PI / 2.0` (model.rs): the E7 angle when half a pitch is the maximum;
  - `impl Harmonic { pub fn amplitude(&self, backiron: i64, mu0: f64) -> f64 }`: `bi bo S / (2 mu0)` in the circuit in effect, the one expression `shear_stress` and `at_pull_out` share;
  - `pub(crate) fn at_pull_out(h: &[Harmonic], backiron: i64, mu0: f64, dev: Deviations) -> (Vec<Harmonic>, Option<f64>)`: now also returns the E7 angle (`None`: half a pitch);
  - Rust-only `ModelResults` fields `k7`, `k9`, `k11`, `b_i7`..`b_i11`, `b_o7`..`b_o11`, `s7_iron`..`s11_iron`, `s7_free`..`s11_free`, `amp1_Pa`..`amp11_Pa`, `pullout_angle_rad`, `iron_circuit_angle_rad`, `free_circuit_angle_rad`;
  - Rust-only `CalibrationResults` fields `amp1_Pa`..`amp11_Pa`, `pullout_angle_rad`.

- [ ] **Step 1: Write the failing test**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        let (off, on) = case(1.44, 1.59, 1.32, 3.17);
        assert!(on.gap_flux_density_T < off.gap_flux_density_T);
    }
}
```

with:

```rust
        let (off, on) = case(1.44, 1.59, 1.32, 3.17);
        assert!(on.gap_flux_density_T < off.gap_flux_density_T);
    }

    #[test]
    fn the_explorer_terms_are_the_engine_s_own_values() {
        // Plan A-3: the Rust-only terms of the equation explorer are reads of what the engine
        // computes. Every harmonic up to 11 has its parts whether or not it is summed; each
        // amplitude is B_i B_o S / (2 mu0) in the circuit in effect (steel here); each summed
        // shear stress is its amplitude at the pull-out angle; the angles are pi/2 or inside
        // [0, pi/2]; the prototype's terms obey the same law at its own angle.
        use crate::engine::api::{DesignInputs, compute_all};
        use crate::engine::meta::{InputSet, Value};
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        for code in [5, 11] {
            let mut inputs = DesignInputs::default();
            inputs
                .set("coupling.max_harmonic", Value::Int(code))
                .unwrap();
            let res = compute_all(&inputs);
            let r = &res.model;
            let parts = [
                (1, r.b_i1, r.b_o1, r.s1_iron, r.amp1_Pa, r.tau1_Pa),
                (3, r.b_i3, r.b_o3, r.s3_iron, r.amp3_Pa, r.tau3_Pa),
                (5, r.b_i5, r.b_o5, r.s5_iron, r.amp5_Pa, r.tau5_Pa),
                (7, r.b_i7, r.b_o7, r.s7_iron, r.amp7_Pa, r.tau7_Pa),
                (9, r.b_i9, r.b_o9, r.s9_iron, r.amp9_Pa, r.tau9_Pa),
                (11, r.b_i11, r.b_o11, r.s11_iron, r.amp11_Pa, r.tau11_Pa),
            ];
            for (n, bi, bo, s, amp, tau) in parts {
                assert!(
                    bi != 0.0 && s != 0.0,
                    "harmonic {n}'s parts exist (code {code})"
                );
                assert!(
                    close(amp, bi * bo * s / (2.0 * MU0)),
                    "harmonic {n}: amplitude (code {code})"
                );
                let want = if i64::from(n) <= code {
                    amp * (f64::from(n) * r.pullout_angle_rad).sin()
                } else {
                    0.0
                };
                assert!(close(tau, want), "harmonic {n}: shear stress (code {code})");
            }
            for angle in [
                r.pullout_angle_rad,
                r.iron_circuit_angle_rad,
                r.free_circuit_angle_rad,
            ] {
                assert!((0.0..=HALF_PITCH_RAD).contains(&angle), "{angle}");
            }
            let c = &res.calibration;
            for (n, amp, tau) in [
                (1, c.amp1_Pa, c.tau1_Pa),
                (3, c.amp3_Pa, c.tau3_Pa),
                (11, c.amp11_Pa, c.tau11_Pa),
            ] {
                let want = if i64::from(n) <= code {
                    amp * (f64::from(n) * c.pullout_angle_rad).sin()
                } else {
                    0.0
                };
                assert!(close(tau, want), "prototype harmonic {n} (code {code})");
            }
        }
    }
}
```

- [ ] **Step 2: Run it to see it fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib the_explorer_terms 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors: ``error[E0425]: cannot find value `HALF_PITCH_RAD` in this scope``, then ``error[E0609]: no field `amp1_Pa` on type `&model::ModelResults` `` and the same for `amp3_Pa` (more follow without `head`).

- [ ] **Step 3: Expose the terms**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup; a grade-mode ring reads its alpha and density (decision A2-7) |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa` and `end_effect_check`) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below, and each ring's alpha(Br) and magnet density used: a grade-mode ring's grade, decision A2-7), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27) |
```

with:

```markdown
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows (the A6 part table: workbook Br and Tmax, grade, vendor page, coating, magnetization) and the exact-text lookup; `br_T` applies E3 with the N42SH grade's Br; `tmax_C` and `grade_id` apply E19 (the vendor's grid) |
| `src/engine/grades.rs` | Addendum A6 grade table `GRADES` (17 grades: Br, Hcj, Hcb, (BH)max, alpha, beta, mu_rec, Tmax, density), each value cited; `grade(id)` lookup; a grade-mode ring reads its alpha and density (decision A2-7) |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells, plus the Rust-only `tau7_Pa` to `tau11_Pa`, `end_effect_check`, and the explorer's torque-angle amplitudes `amp1_Pa` to `amp11_Pa` and E7 angle `pullout_angle_rad`, plan A-3) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below, each ring's alpha(Br) and magnet density used: a grade-mode ring's grade, decision A2-7, and the equation explorer's terms, Rust-only reads, plan A-3: the wave number, amplitudes and geometry factors of harmonics 7 to 11, `k7` to `s11_free`, each harmonic's torque-angle amplitude `amp1_Pa` to `amp11_Pa`, `Harmonic::amplitude` being the one expression the shear stress and the pull-out share, and the E7 angles `pullout_angle_rad`, `iron_circuit_angle_rad` and `free_circuit_angle_rad`), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, out_rust_only, param, results};
use super::model::{
    ODD_HARMONICS, br_factor, corner_radius, end_effect_check, harmonic_count, harmonic_slot,
    harmonic_sum, peak_angle, tau_at,
};

inputs! {
```

with:

```rust
use super::deviations::Deviations;
use super::meta::{NumOrText, inputs, out, out_rust_only, param, results};
use super::model::{
    HALF_PITCH_RAD, ODD_HARMONICS, br_factor, corner_radius, end_effect_check, harmonic_count,
    harmonic_slot, harmonic_sum, peak_angle, tau_at,
};

inputs! {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
                "Summed when the highest harmonic is 9 or more; 0 otherwise."),
            tau11_Pa: f64 => out_rust_only("Pa", "Shear stress, harmonic 11",
                "Summed when the highest harmonic is 11; 0 otherwise."),
            torque_2d_Nm: f64 => out("N·m", "2D torque before end and calibration factors", "", "Calibration!C43"),
            original_model_Nm: f64 => out("N·m", "Original model pull-out torque",
                "Same value as model_torque_Nm.", "Calibration!C44"),
```

with:

```rust
                "Summed when the highest harmonic is 9 or more; 0 otherwise."),
            tau11_Pa: f64 => out_rust_only("Pa", "Shear stress, harmonic 11",
                "Summed when the highest harmonic is 11; 0 otherwise."),
            amp1_Pa: f64 => out_rust_only("Pa", "Harmonic 1 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): the prototype's harmonic 1 shear stress at electrical angle φ is this times sin(1φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp3_Pa: f64 => out_rust_only("Pa", "Harmonic 3 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): the prototype's harmonic 3 shear stress at electrical angle φ is this times sin(3φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp5_Pa: f64 => out_rust_only("Pa", "Harmonic 5 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): the prototype's harmonic 5 shear stress at electrical angle φ is this times sin(5φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp7_Pa: f64 => out_rust_only("Pa", "Harmonic 7 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): the prototype's harmonic 7 shear stress at electrical angle φ is this times sin(7φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp9_Pa: f64 => out_rust_only("Pa", "Harmonic 9 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): the prototype's harmonic 9 shear stress at electrical angle φ is this times sin(9φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp11_Pa: f64 => out_rust_only("Pa", "Harmonic 11 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): the prototype's harmonic 11 shear stress at electrical angle φ is this times sin(11φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            pullout_angle_rad: f64 => out_rust_only("rad", "Pull-out electrical angle",
                "E7 (plan A-3: a term of the equation explorer): the electrical angle θ at which the prototype's torque-angle curve Σ τ_n sin(nθ) peaks; π/2 (half a pole pitch) when that is the maximum."),
            torque_2d_Nm: f64 => out("N·m", "2D torque before end and calibration factors", "", "Calibration!C43"),
            original_model_Nm: f64 => out("N·m", "Original model pull-out torque",
                "Same value as model_torque_Nm.", "Calibration!C44"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/calibration.rs`, replace:

```rust
        tau7_Pa: t(3),
        tau9_Pa: t(4),
        tau11_Pa: t(5),
        torque_2d_Nm: t2d,
        original_model_Nm: model,
        fea_interp_Nm: interp,
```

with:

```rust
        tau7_Pa: t(3),
        tau9_Pa: t(4),
        tau11_Pa: t(5),
        amp1_Pa: amp_n(1),
        amp3_Pa: amp_n(3),
        amp5_Pa: amp_n(5),
        amp7_Pa: amp_n(7),
        amp9_Pa: amp_n(9),
        amp11_Pa: amp_n(11),
        pullout_angle_rad: peak.unwrap_or(HALF_PITCH_RAD),
        torque_2d_Nm: t2d,
        original_model_Nm: model,
        fea_interp_Nm: interp,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                "Summed when the highest harmonic is 9 or more; 0 otherwise."),
            tau11_Pa: f64 => out_rust_only("Pa", "Harmonic 11 shear stress",
                "Summed when the highest harmonic is 11; 0 otherwise."),
            tau_Pa: f64 => out("Pa", "Total magnetic shear stress at pull-out",
                "PM-PM couplings typically 100–250 kPa.", "Calculator!C89"),
            area_lever_m3: f64 => out("m³", "Gap area × lever arm (2π R_g² L)",
```

with:

```rust
                "Summed when the highest harmonic is 9 or more; 0 otherwise."),
            tau11_Pa: f64 => out_rust_only("Pa", "Harmonic 11 shear stress",
                "Summed when the highest harmonic is 11; 0 otherwise."),
            k7: f64 => out_rust_only("1/m", "Harmonic 7 wave number",
                "Plan A-3 (a term of the equation explorer): computed for every harmonic up to 11, summed or not, as k1 to k5 are."),
            b_i7: f64 => out_rust_only("T", "Harmonic 7 inner amplitude", "As b_i1 to b_i5, for harmonic 7."),
            b_o7: f64 => out_rust_only("T", "Harmonic 7 outer amplitude", "As b_o1 to b_o5, for harmonic 7."),
            s7_iron: f64 => out_rust_only("-", "Harmonic 7 geometry factor with back iron", "As s1_iron to s5_iron, for harmonic 7."),
            s7_free: f64 => out_rust_only("-", "Harmonic 7 geometry factor without back iron", "As s1_free to s5_free, for harmonic 7."),
            k9: f64 => out_rust_only("1/m", "Harmonic 9 wave number",
                "Plan A-3 (a term of the equation explorer): computed for every harmonic up to 11, summed or not, as k1 to k5 are."),
            b_i9: f64 => out_rust_only("T", "Harmonic 9 inner amplitude", "As b_i1 to b_i5, for harmonic 9."),
            b_o9: f64 => out_rust_only("T", "Harmonic 9 outer amplitude", "As b_o1 to b_o5, for harmonic 9."),
            s9_iron: f64 => out_rust_only("-", "Harmonic 9 geometry factor with back iron", "As s1_iron to s5_iron, for harmonic 9."),
            s9_free: f64 => out_rust_only("-", "Harmonic 9 geometry factor without back iron", "As s1_free to s5_free, for harmonic 9."),
            k11: f64 => out_rust_only("1/m", "Harmonic 11 wave number",
                "Plan A-3 (a term of the equation explorer): computed for every harmonic up to 11, summed or not, as k1 to k5 are."),
            b_i11: f64 => out_rust_only("T", "Harmonic 11 inner amplitude", "As b_i1 to b_i5, for harmonic 11."),
            b_o11: f64 => out_rust_only("T", "Harmonic 11 outer amplitude", "As b_o1 to b_o5, for harmonic 11."),
            s11_iron: f64 => out_rust_only("-", "Harmonic 11 geometry factor with back iron", "As s1_iron to s5_iron, for harmonic 11."),
            s11_free: f64 => out_rust_only("-", "Harmonic 11 geometry factor without back iron", "As s1_free to s5_free, for harmonic 11."),
            amp1_Pa: f64 => out_rust_only("Pa", "Harmonic 1 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,1 B_o,1 S_1 / (2 μ0) in the circuit in effect, so harmonic 1's shear stress at electrical angle φ is this times sin(1φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp3_Pa: f64 => out_rust_only("Pa", "Harmonic 3 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,3 B_o,3 S_3 / (2 μ0) in the circuit in effect, so harmonic 3's shear stress at electrical angle φ is this times sin(3φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp5_Pa: f64 => out_rust_only("Pa", "Harmonic 5 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,5 B_o,5 S_5 / (2 μ0) in the circuit in effect, so harmonic 5's shear stress at electrical angle φ is this times sin(5φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp7_Pa: f64 => out_rust_only("Pa", "Harmonic 7 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,7 B_o,7 S_7 / (2 μ0) in the circuit in effect, so harmonic 7's shear stress at electrical angle φ is this times sin(7φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp9_Pa: f64 => out_rust_only("Pa", "Harmonic 9 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,9 B_o,9 S_9 / (2 μ0) in the circuit in effect, so harmonic 9's shear stress at electrical angle φ is this times sin(9φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            amp11_Pa: f64 => out_rust_only("Pa", "Harmonic 11 torque-angle amplitude",
                "Plan A-3 (a term of the equation explorer): B_i,11 B_o,11 S_11 / (2 μ0) in the circuit in effect, so harmonic 11's shear stress at electrical angle φ is this times sin(11φ); the E7 peak search reads these. Computed whether or not the harmonic is summed."),
            tau_Pa: f64 => out("Pa", "Total magnetic shear stress at pull-out",
                "PM-PM couplings typically 100–250 kPa.", "Calculator!C89"),
            area_lever_m3: f64 => out("m³", "Gap area × lever arm (2π R_g² L)",
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                "", "Calculator!C95"),
            pullout_noiron_Nm: f64 => out("N·m", "Raw no-back-iron prediction, same layout",
                "", "Calculator!C96"),
            ripple_freq_Hz: f64 => out("Hz", "Torque ripple frequency at the design slip speed",
                "", "Calculator!C97"),
            gearbox_input_ripple_Nm: f64 => out("N·m", "Estimated gearbox input torque ripple amplitude",
```

with:

```rust
                "", "Calculator!C95"),
            pullout_noiron_Nm: f64 => out("N·m", "Raw no-back-iron prediction, same layout",
                "", "Calculator!C96"),
            pullout_angle_rad: f64 => out_rust_only("rad", "Pull-out electrical angle",
                "E7 (plan A-3: a term of the equation explorer): the electrical angle θ at which the torque-angle curve Σ τ_n sin(nθ) of the harmonics summed peaks; π/2 (half a pole pitch, the workbook's assumption) when that is the maximum. Every τ_n is taken at it."),
            iron_circuit_angle_rad: f64 => out_rust_only("rad", "Pull-out electrical angle, steel circuit (C95)",
                "E7: as pullout_angle_rad, for the steel-backed circuit sum of C95."),
            free_circuit_angle_rad: f64 => out_rust_only("rad", "Pull-out electrical angle, free-space circuit (C96)",
                "E7: as pullout_angle_rad, for the no-back-iron circuit sum of C96."),
            ripple_freq_Hz: f64 => out("Hz", "Torque ripple frequency at the design slip speed",
                "", "Calculator!C97"),
            gearbox_input_ripple_Nm: f64 => out("N·m", "Estimated gearbox input torque ripple amplitude",
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            self.s_free
        }
    }
}

/// S_n: sinh ratio for a steel-backed circuit, exponential form for free-space rings.
```

with:

```rust
            self.s_free
        }
    }

    /// The torque-angle amplitude [Pa] in the circuit `backiron` selects, B_in B_on S_n / (2 μ0):
    /// the harmonic's shear stress at electrical angle x is this times sin(n x) (E7). The one
    /// expression for `shear_stress`, `at_pull_out` and the `amp*_Pa` results.
    pub fn amplitude(&self, backiron: i64, mu0: f64) -> f64 {
        self.bi * self.bo / (2.0 * mu0) * self.s(backiron)
    }
}

/// S_n: sinh ratio for a steel-backed circuit, exponential form for free-space rings.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
            s_free,
            tau: 0.0,
        };
        hn.tau = bi * bo / (2.0 * mu0) * hn.s(backiron) * (nf * PI / 2.0).sin();
        hn
    })
}
```

with:

```rust
            s_free,
            tau: 0.0,
        };
        hn.tau = hn.amplitude(backiron, mu0) * (nf * PI / 2.0).sin();
        hn
    })
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
/// E7 for one circuit: every harmonic's `tau` at the true pull-out angle when
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
    }
    h_pull
}

/// Calculator sheet. Linked values come from Metal design, Calibration and Materials (see `api`).
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
```

with:

```rust
/// E7 for one circuit: every harmonic's `tau` at the true pull-out angle when
/// half a pitch is not the maximum ([`peak_angle`] on the amplitudes
/// B_in,n·B_on,n/(2μ0)·S_n of the circuit `backiron` selects). Returns `h`
/// unchanged, bit for bit, when E7 is off or half a pitch is the maximum, with the angle
/// [`peak_angle`] found (`None`: half a pitch). `h` is the harmonic set summed (Addendum A3).
/// Shared by [`compute`] and the sweep rows.
pub(crate) fn at_pull_out(
    h: &[Harmonic],
    backiron: i64,
    mu0: f64,
    dev: Deviations,
) -> (Vec<Harmonic>, Option<f64>) {
    let amplitude = |x: &Harmonic| x.amplitude(backiron, mu0);
    let mut h_pull = h.to_vec();
    let amplitudes: Vec<f64> = h.iter().map(amplitude).collect();
    let peak = peak_angle(&amplitudes, dev);
    if let Some(x) = peak {
        for hn in h_pull.iter_mut() {
            hn.tau = tau_at(amplitude(hn), hn.n, x);
        }
    }
    (h_pull, peak)
}

/// Half a pole pitch as an electrical angle, π/2: where the workbook evaluates every
/// harmonic, and what the angle results (`pullout_angle_rad`, the circuit angles and the
/// Calibration's) read when [`peak_angle`] keeps it (returns `None`).
pub const HALF_PITCH_RAD: f64 = PI / 2.0;

/// Calculator sheet. Linked values come from Metal design, Calibration and Materials (see `api`).
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    let used = &h[..count.unwrap_or(0)];
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    // `used` (the half-pitch terms) stays for the per-circuit sums below.
    let h_pull = at_pull_out(used, ci.backiron, ci.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let tau = harmonic_sum(count, taus.iter().copied()); // Python sum(): left fold from 0
    let AL = 2.0 * PI * (R_g / 1000.0).powi(2) * (L / 1000.0);
```

with:

```rust
    let used = &h[..count.unwrap_or(0)];
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    // `used` (the half-pitch terms) stays for the per-circuit sums below.
    let (h_pull, pull_peak) = at_pull_out(used, ci.backiron, ci.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let tau = harmonic_sum(count, taus.iter().copied()); // Python sum(): left fold from 0
    let AL = 2.0 * PI * (R_g / 1000.0).powi(2) * (L / 1000.0);
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
    let T_pull20 = T_pull * (mi.br_T * mo.br_T) / (bri * bro);
    // sum(bi * bo * S * sin(n pi/2) for n) / (2 mu0) * ...: note S inside the product, /(2 mu0) after the sum
    // E7: each circuit at the maximum of its own torque-angle curve.
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients: Vec<f64> = used.iter().map(|x| x.bi * x.bo * s(x)).collect();
        match peak_angle(&coefficients, dev) {
            Some(x) => harmonic_sum(
                count,
                used.iter()
```

with:

```rust
    let T_pull20 = T_pull * (mi.br_T * mo.br_T) / (bri * bro);
    // sum(bi * bo * S * sin(n pi/2) for n) / (2 mu0) * ...: note S inside the product, /(2 mu0) after the sum
    // E7: each circuit at the maximum of its own torque-angle curve.
    // Each gives the sum and the E7 angle it is taken at (`None`: half a pitch).
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients: Vec<f64> = used.iter().map(|x| x.bi * x.bo * s(x)).collect();
        let peak = peak_angle(&coefficients, dev);
        let sum = match peak {
            Some(x) => harmonic_sum(
                count,
                used.iter()
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
                used.iter()
                    .map(|x| x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin()),
            ),
        }
    };
    let T_iron = circuit(|x| x.s_iron) / (2.0 * ci.mu0) * AL * f_end * f_cal_original;
    let T_noiron = circuit(|x| x.s_free) / (2.0 * ci.mu0) * AL * f_end * f_cal;

    let floor_ = py_max(ci.drive_torque_Nm * ci.drive_safety_factor, required_min_Nm);
    // E10: in the series circuit each magnet contributes its own MMF (Br·t), not the mean Br.
```

with:

```rust
                used.iter()
                    .map(|x| x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin()),
            ),
        };
        (sum, peak)
    };
    let (iron_sum, iron_peak) = circuit(|x| x.s_iron);
    let (free_sum, free_peak) = circuit(|x| x.s_free);
    let T_iron = iron_sum / (2.0 * ci.mu0) * AL * f_end * f_cal_original;
    let T_noiron = free_sum / (2.0 * ci.mu0) * AL * f_end * f_cal;

    let floor_ = py_max(ci.drive_torque_Nm * ci.drive_safety_factor, required_min_Nm);
    // E10: in the series circuit each magnet contributes its own MMF (Br·t), not the mean Br.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        };
        text.to_owned()
    };
    let [h1, h3, h5, ..] = h;
    let tau_n = |i: usize| harmonic_slot(count, &taus, i);

    ModelResults {
```

with:

```rust
        };
        text.to_owned()
    };
    let [h1, h3, h5, h7, h9, h11] = h;
    let tau_n = |i: usize| harmonic_slot(count, &taus, i);

    ModelResults {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        tau7_Pa: tau_n(3),
        tau9_Pa: tau_n(4),
        tau11_Pa: tau_n(5),
        tau_Pa: tau,
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
```

with:

```rust
        tau7_Pa: tau_n(3),
        tau9_Pa: tau_n(4),
        tau11_Pa: tau_n(5),
        k7: h7.k,
        b_i7: h7.bi,
        b_o7: h7.bo,
        s7_iron: h7.s_iron,
        s7_free: h7.s_free,
        k9: h9.k,
        b_i9: h9.bi,
        b_o9: h9.bo,
        s9_iron: h9.s_iron,
        s9_free: h9.s_free,
        k11: h11.k,
        b_i11: h11.bi,
        b_o11: h11.bo,
        s11_iron: h11.s_iron,
        s11_free: h11.s_free,
        amp1_Pa: h[0].amplitude(ci.backiron, ci.mu0),
        amp3_Pa: h[1].amplitude(ci.backiron, ci.mu0),
        amp5_Pa: h[2].amplitude(ci.backiron, ci.mu0),
        amp7_Pa: h[3].amplitude(ci.backiron, ci.mu0),
        amp9_Pa: h[4].amplitude(ci.backiron, ci.mu0),
        amp11_Pa: h[5].amplitude(ci.backiron, ci.mu0),
        tau_Pa: tau,
        area_lever_m3: AL,
        torque_2d_Nm: T2D,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/model.rs`, replace:

```rust
        pullout_20C_Nm: T_pull20,
        pullout_iron_Nm: T_iron,
        pullout_noiron_Nm: T_noiron,
        ripple_freq_Hz: N / 2.0 * slip_rpm / 60.0,
        gearbox_input_ripple_Nm: T_pull / (ci.gear_ratio * ci.gear_efficiency),
        gearbox_reference_Nm: ci.gearbox_input_rating_Nm * ci.gear_ratio * ci.gear_efficiency,
```

with:

```rust
        pullout_20C_Nm: T_pull20,
        pullout_iron_Nm: T_iron,
        pullout_noiron_Nm: T_noiron,
        pullout_angle_rad: pull_peak.unwrap_or(HALF_PITCH_RAD),
        iron_circuit_angle_rad: iron_peak.unwrap_or(HALF_PITCH_RAD),
        free_circuit_angle_rad: free_peak.unwrap_or(HALF_PITCH_RAD),
        ripple_freq_Hz: N / 2.0 * slip_rpm / 60.0,
        gearbox_input_ripple_Nm: T_pull / (ci.gear_ratio * ci.gear_efficiency),
        gearbox_reference_Nm: ci.gearbox_input_rating_Nm * ci.gear_ratio * ci.gear_efficiency,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/sweeps.rs`, replace:

```rust
    // Addendum A3: the Calculator's harmonic set. E7: every harmonic at the true pull-out
    // angle when half a pitch is not the maximum (as the model).
    let count = harmonic_count(ctx.max_harmonic);
    let h_pull = at_pull_out(&h[..count.unwrap_or(0)], ctx.backiron, ctx.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let U = harmonic_sum(count, taus.iter().copied());
    let V = U * 2.0 * PI * (H / 1000.0).powi(2) * (ctx.L / 1000.0);
```

with:

```rust
    // Addendum A3: the Calculator's harmonic set. E7: every harmonic at the true pull-out
    // angle when half a pitch is not the maximum (as the model).
    let count = harmonic_count(ctx.max_harmonic);
    let (h_pull, _) = at_pull_out(&h[..count.unwrap_or(0)], ctx.backiron, ctx.mu0, dev);
    let taus: Vec<f64> = h_pull.iter().map(|x| x.tau).collect();
    let U = harmonic_sum(count, taus.iter().copied());
    let V = U * 2.0 * PI * (H / 1000.0).powi(2) * (ctx.L / 1000.0);
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib the_explorer_terms 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `139 passed` (Task 0's 138 and this task's test); the integration binaries and the doc-tests as in Task 0.

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task1.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task1.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 5: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/model.rs magcoupling-rs/src/engine/calibration.rs magcoupling-rs/src/engine/sweeps.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): expose the torque chain's terms as Rust-only results

Harmonics 7 to 11's parts, each harmonic's torque-angle amplitude and the E7
angles (pull-out, both circuit sums, the prototype), for the equation explorer.
Pure reads: the shared amplitude is one function (Harmonic::amplitude) that the
shear stress and the pull-out call, so every existing result is bit-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 2: The formula markup, its evaluator and the static tables

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

The markup is the explorer's core decision (this plan's Architecture): one text per result is both what the M4 panel typesets
and what the drift guard evaluates, so the equation shown is provably the one that produced the number. The grammar (in
`markup.rs`'s module docs) has no implicit multiplication; `{path}` is a term (an input or result path, `{path|m}` converted to
another unit); `cases(..)` are piecewise formulas whose NaN comparisons are false as in the engine; `sum(n in H: ..)` folds over the
harmonic set as Python's `sum()`; `peak(n in H: A)` is the E7 angle of the amplitudes A_n, decided exactly as the engine decides it
(`model::peak_off_half_pitch`); `table("t", key, "f")` reads a static engine table with every correction on (E3 for library Br);
`where [S] = e` names a sub-expression. The evaluator traces which terms moved the result (value terms) and which were only read to
decide something piecewise-constant (condition terms): the A3 traceability test (Task 4) reads it. This task writes the three
modules test-first: Step 1 creates each file with its unit tests only.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs` (the module, three submodules for now)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs` (formula and symbol grammar, parser, parse tree, visitors; unit tests)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs` (evaluator, `Trace`, `TermSource`; unit tests)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs` (`TABLE_FIELDS`, `field`, `lookup` (magnets, grades, back iron); unit tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/mod.rs` (`pub mod explain;`)

**Interfaces:**
- Consumes: `model::{HALF_PITCH_RAD, ODD_HARMONICS, harmonic_count, peak_off_half_pitch}` (Task 1 and A-2), `compat::{py_min, py_max}`, `library::{lookup, br_T}`, `grades::grade`, `material_library::{BACK_IRON_CHOICES, chosen}`, `meta::Value`.
- Produces:
  - `markup::parse(src: &str, index: Option<u32>) -> Result<Formula, ParseError>`; `Formula { body: Expr, bindings: Vec<Binding>, cases_count: usize }` with `visit`/`visit_mut`; `Expr` (`Num { value, text }`, `Text`, `NoneLit`, `Pi`, `Term(TermRef)`, `FamilyTerm(TermRef)`, `Index`, `Local(usize)`, `Neg`, `Bin(BinOp, ..)`, `Frac`, `Call(Func, Vec<Expr>)`, `Paren`, `Sum(IndexSet, ..)`, `Cases { id, arms, otherwise }`, `Peak(IndexSet, ..)`, `Table { table, key, field }`); `TermRef { path, unit: Option<String>, scale: f64 }`; `Symbol::parse(&str) -> Result<Symbol, String>` and `Symbol::plain()`; `IndexSet::Harmonics.selector() == "coupling.max_harmonic"`;
  - `eval::TermSource` (`fn value(&self, path: &str) -> Option<Value>`, implemented for `BTreeMap<String, Value>`), `eval::EvalError(String)`, `eval::Trace { value_terms, condition_terms, arms }`, `eval::evaluate(&Formula, &dyn TermSource, Option<&mut Trace>) -> Result<Value, EvalError>`;
  - `tables::TABLE_FIELDS: &[TableField]` (`table`, `field`, `symbol`, `unit`, `label`), `tables::field(table, field) -> Option<&TableField>`, `tables::lookup(table, &Value, field) -> Result<Option<Value>, String>`.

- [ ] **Step 1: Write the failing tests**

`mod.rs` declares the three modules; each module file holds only its `#[cfg(test)] mod tests` for now.

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs` with exactly this content:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::explain::markup::parse;

    fn src(pairs: &[(&str, Value)]) -> BTreeMap<String, Value> {
        pairs
            .iter()
            .map(|(p, v)| ((*p).to_owned(), v.clone()))
            .collect()
    }

    fn eval(markup: &str, s: &BTreeMap<String, Value>) -> (Value, Trace) {
        let f = parse(markup, None).unwrap();
        let mut t = Trace::default();
        (evaluate(&f, s, Some(&mut t)).unwrap(), t)
    }

    #[test]
    fn arithmetic_and_functions() {
        let s = src(&[("a.x", Value::Num(3.0)), ("a.y", Value::Int(4))]);
        assert_eq!(eval("sqrt({a.x}^2 + {a.y}^2)", &s).0, Value::Num(5.0));
        assert_eq!(eval("frac({a.x}, 2) · 2 - -1", &s).0, Value::Num(4.0));
        assert_eq!(
            eval("exp(0) + ln(1) + abs(-2) + ceil(0.2) + floor(1.8)", &s).0,
            Value::Num(5.0)
        );
        assert_eq!(eval("2^0.5", &s).0, Value::Num(2f64.powf(0.5)));
    }

    #[test]
    fn unit_scale_applies_to_numbers() {
        let s = src(&[("a.len_mm", Value::Num(12.0))]);
        let mut f = parse("{a.len_mm|m}", None).unwrap();
        f.visit_mut(&mut |e| {
            if let Expr::Term(r) = e {
                r.scale = 1e-3;
            }
        });
        assert_eq!(evaluate(&f, &s, None).unwrap(), Value::Num(12.0 * 1e-3));
    }

    #[test]
    fn cases_trace_value_and_condition_terms() {
        let s = src(&[
            ("a.x", Value::Num(1.0)),
            ("a.y", Value::Num(2.0)),
            ("a.v", Value::Num(7.0)),
            ("a.w", Value::Num(9.0)),
        ]);
        let (v, t) = eval(r#"cases({a.x} < {a.y} => {a.v}; else => {a.w})"#, &s);
        assert_eq!(v, Value::Num(7.0));
        assert_eq!(t.value_terms, ["a.v".to_owned()].into());
        assert_eq!(
            t.condition_terms,
            ["a.x".to_owned(), "a.y".to_owned()].into()
        );
        assert_eq!(t.arms, [(0, 0)].into());
        let (v, t) = eval(r#"cases({a.x} > {a.y} => {a.v}; else => "no")"#, &s);
        assert_eq!(v, Value::Text("no".into()));
        assert_eq!(t.arms, [(0, 1)].into());
        // NaN comparisons are false, as the engine's `if`.
        let nan = src(&[("a.x", Value::Num(f64::NAN))]);
        assert_eq!(
            eval(r#"cases({a.x} < 1 => 1; else => 2)"#, &nan).0,
            Value::Num(2.0)
        );
    }

    #[test]
    fn text_and_none_compare_exactly() {
        let s = src(&[("a.p", Value::Text("B842SH".into())), ("a.d", Value::None)]);
        assert_eq!(
            eval(
                r#"cases({a.p} = "B842SH" and {a.d} = none => 1; else => 0)"#,
                &s
            )
            .0,
            Value::Num(1.0)
        );
        assert_eq!(
            eval(r#"cases({a.p} != "B842" => 1; else => 0)"#, &s).0,
            Value::Num(1.0)
        );
        let f = parse(r#"cases({a.p} < "C" => 1; else => 0)"#, None).unwrap();
        assert!(evaluate(&f, &s, None).is_err(), "text has no order");
    }

    #[test]
    fn min_max_follow_python_and_trace_the_winner() {
        let s = src(&[("a.x", Value::Num(1.0)), ("a.y", Value::Num(2.0))]);
        let (v, t) = eval("min({a.y}, {a.x})", &s);
        assert_eq!(v, Value::Num(1.0));
        assert_eq!(t.value_terms, ["a.x".to_owned()].into());
        assert_eq!(t.condition_terms, ["a.y".to_owned()].into());
        let (v, t) = eval("max(1, {a.x})", &s);
        assert_eq!(v, Value::Num(1.0), "a tie keeps the first, as Python");
        assert!(t.value_terms.is_empty());
        // py_min(NaN, 1) is NaN: the first argument is kept when the second does not win.
        let nan = src(&[("a.x", Value::Num(f64::NAN))]);
        assert!(matches!(eval("min({a.x}, 1)", &nan).0, Value::Num(x) if x.is_nan()));
        assert_eq!(eval("min(1, {a.x})", &nan).0, Value::Num(1.0));
    }

    #[test]
    fn a_sum_runs_over_the_harmonic_set() {
        let mut s = src(&[("coupling.max_harmonic", Value::Int(5))]);
        for n in ODD_HARMONICS {
            s.insert(format!("m.t{n}"), Value::Num(f64::from(n)));
        }
        let (v, t) = eval("sum(n in H: {m.t#} * n)", &s);
        assert_eq!(v, Value::Num(1.0 + 9.0 + 25.0));
        assert!(t.condition_terms.contains("coupling.max_harmonic"));
        assert!(t.value_terms.contains("m.t5") && !t.value_terms.contains("m.t7"));
        s.insert("coupling.max_harmonic".into(), Value::Int(4));
        assert!(
            matches!(eval("sum(n in H: {m.t#})", &s).0, Value::Num(x) if x.is_nan()),
            "invalid set"
        );
    }

    #[test]
    fn peak_is_the_e7_angle_of_the_amplitudes() {
        let mut s = src(&[("coupling.max_harmonic", Value::Int(3))]);
        // A small third harmonic: half a pitch stays the peak; a large one moves it.
        s.insert("m.a1".into(), Value::Num(1.0));
        s.insert("m.a3".into(), Value::Num(0.05));
        let (v, t) = eval("peak(n in H: {m.a#})", &s);
        assert_eq!(v, Value::Num(HALF_PITCH_RAD));
        assert!(
            t.value_terms.is_empty(),
            "the angle does not move with the amplitudes here"
        );
        s.insert("m.a3".into(), Value::Num(0.5));
        let (v, t) = eval("peak(n in H: {m.a#})", &s);
        assert_eq!(v, Value::Num(peak_off_half_pitch(&[1.0, 0.5]).unwrap()));
        assert!(t.value_terms.contains("m.a1") && t.value_terms.contains("m.a3"));
    }

    #[test]
    fn tables_read_engine_data_by_key() {
        let s = src(&[
            ("c.part", Value::Text("B842SH".into())),
            ("c.none", Value::Text("".into())),
        ]);
        let (v, t) = eval(r#"table("magnets", {c.part}, "length_mm")"#, &s);
        assert_eq!(
            v,
            Value::Num(crate::engine::library::lookup("B842SH").unwrap().length_mm)
        );
        assert!(t.value_terms.is_empty() && t.condition_terms.contains("c.part"));
        assert_eq!(
            eval(r#"table("magnets", {c.none}, "length_mm")"#, &s).0,
            Value::None
        );
    }

    #[test]
    fn bindings_are_lazy_and_traced_once_used() {
        let s = src(&[
            ("a.x", Value::Num(1.0)),
            ("a.y", Value::Num(2.0)),
            ("a.b", Value::Int(1)),
        ]);
        let (v, t) = eval(
            "cases({a.b} = 1 => [S] * 10; else => 0) where [S] = {a.x} + {a.y}",
            &s,
        );
        assert_eq!(v, Value::Num(30.0));
        assert!(t.value_terms.contains("a.x") && t.value_terms.contains("a.y"));
        let s0 = src(&[
            ("a.x", Value::Num(1.0)),
            ("a.y", Value::Num(2.0)),
            ("a.b", Value::Int(0)),
        ]);
        let (_, t) = eval(
            "cases({a.b} = 1 => [S] * 10; else => 0) where [S] = {a.x} + {a.y}",
            &s0,
        );
        assert!(t.value_terms.is_empty(), "the binding was never needed");
    }

    #[test]
    fn errors_name_the_problem() {
        let s = src(&[("a.t", Value::Text("x".into()))]);
        for (markup, want) in [
            ("{a.missing}", "unknown term"),
            ("{a.t} + 1", "expected a number"),
            (r#"table("nope", 1, "x")"#, "no table"),
        ] {
            let err = evaluate(&parse(markup, None).unwrap(), &s, None).unwrap_err();
            assert!(err.0.contains(want), "{markup}: {err}");
        }
    }
}
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs` with exactly this content:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn num(v: f64) -> Expr {
        Expr::Num {
            value: v,
            text: v.to_string(),
        }
    }

    fn term(p: &str) -> Expr {
        Expr::Term(TermRef {
            path: p.into(),
            unit: None,
            scale: 1.0,
        })
    }

    #[test]
    fn precedence_and_associativity() {
        // a + b * c ^ d ^ e: power right-assoc and tighter than product, product tighter than sum.
        let f = parse("{a.x} + {a.y} * {a.z} ^ 2 ^ 3", None).unwrap();
        let expected = Expr::Bin(
            BinOp::Add,
            Box::new(term("a.x")),
            Box::new(Expr::Bin(
                BinOp::Mul,
                Box::new(term("a.y")),
                Box::new(Expr::Bin(
                    BinOp::Pow,
                    Box::new(term("a.z")),
                    Box::new(Expr::Bin(
                        BinOp::Pow,
                        Box::new(num(2.0)),
                        Box::new(num(3.0)),
                    )),
                )),
            )),
        );
        assert_eq!(f.body, expected);
        // -x^2 is -(x^2); a - b - c is (a - b) - c; a / b · c is (a / b) · c.
        let f = parse("-{a.x}^2", None).unwrap();
        assert!(
            matches!(f.body, Expr::Neg(ref inner) if matches!(**inner, Expr::Bin(BinOp::Pow, ..)))
        );
        let f = parse("{a.x} - {a.y} - {a.z}", None).unwrap();
        assert!(
            matches!(f.body, Expr::Bin(BinOp::Sub, ref l, _) if matches!(**l, Expr::Bin(BinOp::Sub, ..)))
        );
        let f = parse("{a.x} / {a.y} · {a.z}", None).unwrap();
        assert!(
            matches!(f.body, Expr::Bin(BinOp::Dot, ref l, _) if matches!(**l, Expr::Bin(BinOp::Div, ..)))
        );
    }

    #[test]
    fn numbers_keep_their_text() {
        let f = parse("1e-3 + 0.5 + 20 + 2.5E+2", None).unwrap();
        let mut texts = Vec::new();
        f.visit(&mut |e| {
            if let Expr::Num { text, value } = e {
                texts.push((text.clone(), *value));
            }
        });
        assert_eq!(
            texts,
            vec![
                ("1e-3".into(), 1e-3),
                ("0.5".into(), 0.5),
                ("20".into(), 20.0),
                ("2.5E+2".into(), 250.0)
            ]
        );
    }

    #[test]
    fn family_members_substitute_the_index() {
        let f = parse("frac(n * {m.b_i#|m}, 2) where [S_#] = 1", Some(3));
        // An unused binding is refused.
        assert!(f.is_err());
        let f = parse(
            "frac(n * {m.b_i#|m}, 2) · [S_#] where [S_#] = {m.s#}",
            Some(3),
        )
        .unwrap();
        assert_eq!(f.bindings[0].symbol, "S_3");
        let mut paths = Vec::new();
        f.visit(&mut |e| {
            if let Expr::Term(t) = e {
                paths.push((t.path.clone(), t.unit.clone()));
            }
        });
        assert_eq!(
            paths,
            vec![("m.b_i3".into(), Some("m".into())), ("m.s3".into(), None)]
        );
        assert!(matches!(f.body, Expr::Bin(BinOp::Dot, ref l, _)
            if matches!(**l, Expr::Frac(ref a, _) if matches!(**a, Expr::Bin(BinOp::Mul, ref n, _) if **n == Expr::Num { value: 3.0, text: "3".into() }))));
    }

    #[test]
    fn a_sum_binds_n_and_family_terms() {
        let f = parse("sum(n in H: {m.tau#} * n)", None).unwrap();
        match &f.body {
            Expr::Sum(IndexSet::Harmonics, body) => {
                assert!(matches!(**body, Expr::Bin(BinOp::Mul, ref a, ref b)
                    if matches!(**a, Expr::FamilyTerm(ref t) if t.path == "m.tau#") && **b == Expr::Index));
            }
            other => panic!("{other:?}"),
        }
        assert!(parse("n + 1", None).is_err(), "n outside a sum");
        assert!(parse("{m.x#}", None).is_err(), "# outside a sum");
        assert!(parse("sum(n in H: sum(n in H: n))", None).is_err());
        assert!(
            parse("sum(n in H: n)", Some(1)).is_err(),
            "a sum inside a family"
        );
    }

    #[test]
    fn cases_and_conditions() {
        let f = parse(
            r#"cases({a.x} < {a.y} and {a.p} = "B842SH" => "Below"; {a.z} != none or 1 >= 2 => 1; else => cases(1 <= 2 => 0; else => 1))"#,
            None,
        )
        .unwrap();
        assert_eq!(f.cases_count, 2);
        match &f.body {
            Expr::Cases {
                id,
                arms,
                otherwise,
            } => {
                assert_eq!(*id, 0);
                assert_eq!(arms.len(), 2);
                assert!(matches!(arms[0].0, Cond::And(..)));
                assert!(matches!(arms[1].0, Cond::Or(..)));
                assert!(matches!(**otherwise, Expr::Cases { id: 1, .. }));
            }
            other => panic!("{other:?}"),
        }
        assert!(parse("cases(else => 1)", None).is_err(), "needs an arm");
        assert!(parse("cases(1 < 2 => 1)", None).is_err(), "needs else");
    }

    #[test]
    fn where_bindings_resolve_in_order() {
        let f = parse("[A] + [B] where [A] = 1, [B] = [A] * 2", None).unwrap();
        assert_eq!(f.bindings.len(), 2);
        assert!(
            matches!(f.bindings[1].expr, Expr::Bin(BinOp::Mul, ref a, _) if **a == Expr::Local(0))
        );
        assert!(
            parse("[A] where [A] = [B], [B] = 1", None).is_err(),
            "used before bound"
        );
        assert!(parse("[A]", None).is_err(), "not bound");
        assert!(
            parse("[A] where [A] = 1, [A] = 2", None).is_err(),
            "bound twice"
        );
    }

    #[test]
    fn calls_check_their_arity_and_names() {
        assert!(parse("min(1)", None).is_err());
        assert!(parse("sin(1, 2)", None).is_err());
        assert!(parse("foo(1)", None).is_err());
        assert!(parse("max(1, 2, 3)", None).is_ok());
        assert!(parse("2{a.x}", None).is_err(), "no implicit multiplication");
        assert!(matches!(
            parse("peak(n in H: {m.a#})", None).unwrap().body,
            Expr::Peak(IndexSet::Harmonics, _)
        ));
        assert!(
            parse("peak(n in H: n)", Some(3)).is_err(),
            "a peak inside a family"
        );
        let t = parse(r#"table("magnets", {c.part}, "br_T")"#, None).unwrap();
        assert!(
            matches!(t.body, Expr::Table { ref table, ref field, .. } if table == "magnets" && field == "br_T")
        );
        assert!(
            parse("table(magnets, {c.part}, br_T)", None).is_err(),
            "names are quoted"
        );
        assert!(parse("{a b}", None).is_err());
        assert!(parse("(1 + 2", None).is_err());
    }

    #[test]
    fn symbols_split_into_base_and_scripts() {
        assert_eq!(
            Symbol::parse("S_{3}^{iron}").unwrap(),
            Symbol {
                base: "S".into(),
                sub: Some("3".into()),
                sup: Some("iron".into())
            }
        );
        assert_eq!(Symbol::parse("τ_p").unwrap().plain(), "τ_p");
        assert_eq!(Symbol::parse("T_{pull,20}").unwrap().plain(), "T_{pull,20}");
        assert_eq!(
            Symbol::parse("N").unwrap(),
            Symbol {
                base: "N".into(),
                sub: None,
                sup: None
            }
        );
        for bad in ["", "_i", "B_", "B_{}", "B_{i", "B_i x", "B^2_i"] {
            assert!(Symbol::parse(bad).is_err(), "{bad}");
        }
    }
}
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs` with exactly this content:

```rust
//! Addendum A2 equation explorer, engine side: the explanation layer.
//!
//! - [`markup`]: the formula and symbol grammar, parser and tree;
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron).

pub mod eval;
pub mod markup;
pub mod tables;

pub use eval::{EvalError, TermSource, Trace};
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs` with exactly this content:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_field_reads_and_a_missing_key_is_none() {
        let key = |t: &str| match t {
            "magnets" => Value::Text("B842SH".into()),
            "grades" => Value::Text("N42SH".into()),
            _ => Value::Int(1),
        };
        for f in TABLE_FIELDS {
            assert!(
                lookup(f.table, &key(f.table), f.field).unwrap().is_some(),
                "{}.{}",
                f.table,
                f.field
            );
            let missing = if f.table == "back_iron" {
                Value::Int(99)
            } else {
                Value::Text(String::new())
            };
            assert_eq!(
                lookup(f.table, &missing, f.field).unwrap(),
                None,
                "{}.{}",
                f.table,
                f.field
            );
        }
        assert!(lookup("magnets", &Value::Int(1), "br_T").is_err());
        assert!(lookup("magnets", &key("magnets"), "density").is_err());
        assert!(lookup("nope", &key("magnets"), "br_T").is_err());
    }

    #[test]
    fn the_back_iron_table_knows_the_non_ferromagnetic_choices() {
        let ferro = |c| lookup("back_iron", &Value::Int(c), "ferromagnetic").unwrap();
        assert_eq!(ferro(1), Some(Value::Int(1)), "4140");
        let non: Vec<i64> = BACK_IRON_CHOICES
            .iter()
            .map(|&(c, _)| c)
            .filter(|&c| ferro(c) == Some(Value::Int(0)))
            .collect();
        assert_eq!(non.len(), 2, "304 and 6061 (spec A5)");
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/mod.rs`, replace:

```rust
pub mod compat;
pub mod constants;
pub mod deviations;
pub mod grades;
pub mod housing;
pub mod library;
```

with:

```rust
pub mod compat;
pub mod constants;
pub mod deviations;
pub mod explain;
pub mod grades;
pub mod housing;
pub mod library;
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib explain 2>&1 | grep -E "^error" | head -3
```

Expected: ``error[E0432]: unresolved imports `eval::EvalError`, `eval::TermSource`, `eval::Trace` `` and ``error: could not compile `magcoupling-rs` (lib) due to 1 previous error`` (each module file holds only its tests).

- [ ] **Step 3: Write the three modules**

Each replace block inserts the module's implementation above its tests.

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
//! Evaluates a parsed formula over term values: what the drift guard compares with the
//! engine's result, and (traced) what the traceability test reads to know which terms a
//! result actually depends on at a design point.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::fmt;

use super::markup::{BinOp, Cond, Expr, Formula, Func, IndexSet, RelOp};
use super::tables;
use crate::engine::compat::{py_max, py_min};
use crate::engine::meta::Value;
use crate::engine::model::{HALF_PITCH_RAD, ODD_HARMONICS, harmonic_count, peak_off_half_pitch};

/// Where term values come from: an input or result path to its value, `None` for an
/// unknown path.
pub trait TermSource {
    fn value(&self, path: &str) -> Option<Value>;
}

/// Why a formula could not be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub struct EvalError(pub String);

impl fmt::Display for EvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// What an evaluation read, for the traceability test and branch coverage.
///
/// A term is a **value term** when its value flows into the result along the path taken
/// (arithmetically, through the chosen `cases` arm, through the winner of a `min`/`max`),
/// so nudging it moves the result. It is a **condition term** when it was read only to
/// decide something piecewise-constant: a `cases` condition, the loser of a `min`/`max`, the
/// argument of `ceil`/`floor`, the selector of a Σ's index set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub value_terms: BTreeSet<String>,
    pub condition_terms: BTreeSet<String>,
    /// For each `cases` evaluated, its id and the arm taken (`arms.len()` for `else`).
    pub arms: BTreeSet<(usize, usize)>,
}

impl Trace {
    fn absorb(&mut self, other: Trace, as_condition: bool) {
        if as_condition {
            self.condition_terms.extend(other.value_terms);
        } else {
            self.value_terms.extend(other.value_terms);
        }
        self.condition_terms.extend(other.condition_terms);
        self.arms.extend(other.arms);
    }
}

/// An evaluated value: a number (inputs' integers become numbers), a text or none.
#[derive(Clone, Debug, PartialEq)]
enum V {
    Num(f64),
    Text(String),
    None,
}

impl V {
    fn from_value(v: Value) -> V {
        match v {
            Value::Num(x) => V::Num(x),
            Value::Int(i) => V::Num(i as f64),
            Value::Text(s) => V::Text(s),
            Value::None => V::None,
        }
    }

    fn into_value(self) -> Value {
        match self {
            V::Num(x) => Value::Num(x),
            V::Text(s) => Value::Text(s),
            V::None => Value::None,
        }
    }

    fn num(&self, what: &str) -> Result<f64, EvalError> {
        match self {
            V::Num(x) => Ok(*x),
            other => Err(EvalError(format!(
                "{what}: expected a number, got {other:?}"
            ))),
        }
    }
}

/// Evaluates `formula` over `src`. With `trace`, records what was read ([`Trace`]).
/// `index` is the Σ variable's value when evaluating inside a Σ (callers pass `None`).
pub fn evaluate(
    formula: &Formula,
    src: &dyn TermSource,
    trace: Option<&mut Trace>,
) -> Result<Value, EvalError> {
    let mut ev = Evaluator {
        formula,
        src,
        locals: vec![None; formula.bindings.len()],
    };
    let mut local_trace = Trace::default();
    let v = ev.expr(&formula.body, None, &mut local_trace)?;
    if let Some(t) = trace {
        t.absorb(local_trace, false);
    }
    Ok(v.into_value())
}

struct Evaluator<'a> {
    formula: &'a Formula,
    src: &'a dyn TermSource,
    /// Memoized bindings: the value and what evaluating it read.
    locals: Vec<Option<(V, Trace)>>,
}

impl Evaluator<'_> {
    fn term(&self, path: &str, scale: f64, t: &mut Trace) -> Result<V, EvalError> {
        let v = self
            .src
            .value(path)
            .ok_or_else(|| EvalError(format!("unknown term {path}")))?;
        t.value_terms.insert(path.to_owned());
        let v = V::from_value(v);
        Ok(match v {
            V::Num(x) if scale != 1.0 => V::Num(x * scale),
            other => other,
        })
    }

    fn num(&mut self, e: &Expr, n: Option<u32>, t: &mut Trace) -> Result<f64, EvalError> {
        self.expr(e, n, t)?.num("operand")
    }

    fn expr(&mut self, e: &Expr, n: Option<u32>, t: &mut Trace) -> Result<V, EvalError> {
        Ok(match e {
            Expr::Num { value, .. } => V::Num(*value),
            Expr::Text(s) => V::Text(s.clone()),
            Expr::NoneLit => V::None,
            Expr::Pi => V::Num(PI),
            Expr::Term(r) => self.term(&r.path, r.scale, t)?,
            Expr::FamilyTerm(r) => {
                let n =
                    n.ok_or_else(|| EvalError(format!("family term {} outside a Σ", r.path)))?;
                self.term(&r.path.replace('#', &n.to_string()), r.scale, t)?
            }
            Expr::Index => V::Num(f64::from(
                n.ok_or_else(|| EvalError("n outside a Σ".into()))?,
            )),
            Expr::Peak(set, body) => {
                // The amplitudes over the set; the engine's E7 search picks the angle. Their
                // terms move the angle only where it leaves half a pitch.
                let mut at = Trace::default();
                let amplitudes = match self.members(*set, &mut at)? {
                    None => Vec::new(),
                    Some(members) => members
                        .iter()
                        .map(|&k| self.num(body, Some(k), &mut at))
                        .collect::<Result<Vec<f64>, EvalError>>()?,
                };
                let peak = peak_off_half_pitch(&amplitudes);
                t.absorb(at, peak.is_none());
                V::Num(peak.unwrap_or(HALF_PITCH_RAD))
            }
            Expr::Table { table, key, field } => {
                let mut kt = Trace::default();
                let key = self.expr(key, n, &mut kt)?.into_value();
                t.absorb(kt, true); // the key selects a row: piecewise constant
                match tables::lookup(table, &key, field) {
                    Ok(Some(v)) => V::from_value(v),
                    Ok(None) => V::None,
                    Err(e) => return Err(EvalError(e)),
                }
            }
            Expr::Local(k) => {
                if self.locals[*k].is_none() {
                    let binding = &self.formula.bindings[*k].expr;
                    let mut lt = Trace::default();
                    let v = self.expr(binding, n, &mut lt)?;
                    self.locals[*k] = Some((v, lt));
                }
                let (v, lt) = self.locals[*k].clone().expect("just evaluated");
                t.absorb(lt, false);
                v
            }
            Expr::Neg(a) => V::Num(-self.num(a, n, t)?),
            Expr::Paren(a) => self.expr(a, n, t)?,
            Expr::Bin(op, a, b) => {
                let (x, y) = (self.num(a, n, t)?, self.num(b, n, t)?);
                V::Num(match op {
                    BinOp::Add => x + y,
                    BinOp::Sub => x - y,
                    BinOp::Mul | BinOp::Dot => x * y,
                    BinOp::Div => x / y,
                    BinOp::Pow => pow(x, y),
                })
            }
            Expr::Frac(a, b) => {
                let (x, y) = (self.num(a, n, t)?, self.num(b, n, t)?);
                V::Num(x / y)
            }
            Expr::Call(func, args) => self.call(*func, args, n, t)?,
            Expr::Sum(set, body) => {
                match self.members(*set, t)? {
                    None => V::Num(f64::NAN), // an invalid set: every harmonic sum is NaN (D3)
                    Some(members) => {
                        let mut acc = 0.0; // Python sum(): a left fold from 0
                        for &k in members {
                            acc += self.num(body, Some(k), t)?;
                        }
                        V::Num(acc)
                    }
                }
            }
            Expr::Cases {
                id,
                arms,
                otherwise,
            } => {
                for (i, (cond, value)) in arms.iter().enumerate() {
                    let mut ct = Trace::default();
                    let holds = self.cond(cond, n, &mut ct)?;
                    t.absorb(ct, true);
                    if holds {
                        t.arms.insert((*id, i));
                        return self.expr(value, n, t);
                    }
                }
                t.arms.insert((*id, arms.len()));
                self.expr(otherwise, n, t)?
            }
        })
    }

    /// The members of an index set, as the engine picks them (`model::harmonic_count`);
    /// `None` for an invalid set. Reads the set's selector as a condition term.
    fn members(
        &mut self,
        set: IndexSet,
        t: &mut Trace,
    ) -> Result<Option<&'static [u32]>, EvalError> {
        let selector = set.selector();
        let code = self
            .src
            .value(selector)
            .ok_or_else(|| EvalError(format!("unknown term {selector}")))?;
        t.condition_terms.insert(selector.to_owned());
        match (set, code) {
            (IndexSet::Harmonics, Value::Int(c)) => {
                Ok(harmonic_count(c).map(|k| &ODD_HARMONICS[..k]))
            }
            (_, other) => Err(EvalError(format!(
                "{selector}: expected a code, got {other:?}"
            ))),
        }
    }

    fn call(
        &mut self,
        func: Func,
        args: &[Expr],
        n: Option<u32>,
        t: &mut Trace,
    ) -> Result<V, EvalError> {
        if matches!(func, Func::Min | Func::Max) {
            // Python's min/max fold; only the winner's terms are value terms.
            let mut best: Option<(f64, Trace)> = None;
            for a in args {
                let mut at = Trace::default();
                let x = self.num(a, n, &mut at)?;
                best = Some(match best {
                    None => (x, at),
                    Some((b, bt)) => {
                        // compat::py_min(b, x) is x exactly when x < b (py_max: x > b).
                        let pick = if func == Func::Min {
                            py_min(b, x)
                        } else {
                            py_max(b, x)
                        };
                        let x_wins = if func == Func::Min { x < b } else { x > b };
                        debug_assert!(pick.to_bits() == if x_wins { x } else { b }.to_bits());
                        if x_wins {
                            t.absorb(bt, true);
                            (x, at)
                        } else {
                            t.absorb(at, true);
                            (b, bt)
                        }
                    }
                });
            }
            let (x, bt) = best.expect("min/max has at least two arguments (parser)");
            t.absorb(bt, false);
            return Ok(V::Num(x));
        }
        let mut at = Trace::default();
        let x = self.num(&args[0], n, &mut at)?;
        let piecewise = matches!(func, Func::Ceil | Func::Floor);
        t.absorb(at, piecewise);
        Ok(V::Num(match func {
            Func::Sqrt => x.sqrt(),
            Func::Sin => x.sin(),
            Func::Cos => x.cos(),
            Func::Tan => x.tan(),
            Func::Sinh => x.sinh(),
            Func::Cosh => x.cosh(),
            Func::Tanh => x.tanh(),
            Func::Exp => x.exp(),
            Func::Ln => x.ln(),
            Func::Abs => x.abs(),
            Func::Ceil => x.ceil(),
            Func::Floor => x.floor(),
            Func::Min | Func::Max => unreachable!("handled above"),
        }))
    }

    fn cond(&mut self, c: &Cond, n: Option<u32>, t: &mut Trace) -> Result<bool, EvalError> {
        Ok(match c {
            Cond::And(a, b) => self.cond(a, n, t)? && self.cond(b, n, t)?,
            Cond::Or(a, b) => self.cond(a, n, t)? || self.cond(b, n, t)?,
            Cond::Rel(op, a, b) => {
                let (x, y) = (self.expr(a, n, t)?, self.expr(b, n, t)?);
                match (&x, &y) {
                    (V::Num(x), V::Num(y)) => match op {
                        RelOp::Lt => x < y,
                        RelOp::Le => x <= y,
                        RelOp::Gt => x > y,
                        RelOp::Ge => x >= y,
                        RelOp::Eq => x == y,
                        RelOp::Ne => x != y,
                    },
                    _ => match op {
                        RelOp::Eq => x == y,
                        RelOp::Ne => x != y,
                        _ => return Err(EvalError(format!("cannot order {x:?} and {y:?}"))),
                    },
                }
            }
        })
    }
}

/// `x ^ y`: an integer power by `powi` (as the engine writes squares), else `powf`.
fn pow(x: f64, y: f64) -> f64 {
    if y.fract() == 0.0 && y.abs() <= 64.0 {
        x.powi(y as i32)
    } else {
        x.powf(y)
    }
}

/// A [`TermSource`] over a fixed map (unit tests, documentation examples).
impl TermSource for BTreeMap<String, Value> {
    fn value(&self, path: &str) -> Option<Value> {
        self.get(path).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

````rust
//! The equation markup (Addendum A2): one text per explained result that is both what the
//! equation panel typesets and what the drift guard evaluates, so the equation shown is the
//! one that produced the number (the spec's A2 guarantee).
//!
//! # Formula grammar (EBNF)
//!
//! ```text
//! formula  = expr , [ "where" , binding , { "," , binding } ] ;
//! binding  = local , "=" , expr ;                 (* a named sub-expression *)
//! local    = "[" , symbol , "]" ;                 (* its name, in symbol markup *)
//! expr     = additive ;
//! cases    = "cases" , "(" , arm , { ";" , arm } , ";" , "else" , "=>" , expr , ")" ;
//! arm      = cond , "=>" , expr ;
//! cond     = conj , { "or" , conj } ;
//! conj     = rel , { "and" , rel } ;
//! rel      = additive , ( "<" | "<=" | ">" | ">=" | "=" | "!=" ) , additive ;
//! additive = product , { ( "+" | "-" ) , product } ;
//! product  = unary , { ( "*" | "·" | "/" ) , unary } ;
//! unary    = "-" , unary | power ;
//! power    = atom , [ "^" , unary ] ;             (* right-associative *)
//! atom     = number | text | term | local | "n" | "pi" | "π" | "none"
//!          | call | cases | sum | peak | table | "(" , expr , ")" ;
//! number   = digit , { digit } , [ "." , digit , { digit } ] , [ ( "e" | "E" ) , [ "-" | "+" ] , digit , { digit } ] ;
//! text     = '"' , { char - '"' } , '"' ;
//! term     = "{" , path , [ "|" , unit ] , "}" ;   (* an input or result path; "#" = the index *)
//! call     = func , "(" , expr , { "," , expr } , ")" ;
//! func     = "frac" | "sqrt" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "exp"
//!          | "ln" | "abs" | "min" | "max" | "ceil" | "floor" ;
//! sum      = "sum" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! peak     = "peak" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! table    = "table" , "(" , text , "," , expr , "," , text , ")" ;
//! ```
//!
//! Whitespace separates tokens and is otherwise ignored. There is no implicit
//! multiplication: `2*{coupling.mu0}`, never `2{coupling.mu0}`.
//!
//! # Meaning (what the evaluator computes and the typesetter draws)
//!
//! | Markup | Evaluates to | Typeset as |
//! |---|---|---|
//! | `{model.gap_radius_mm}` | the value at that path (a result, else an input) | the term's display symbol, in the term's colour |
//! | `{model.gap_radius_mm\|m}` | the value converted to the unit after `\|` (the registry resolves the factor; unknown pairs are refused) | the same symbol; the term list shows the value in that unit |
//! | `a * b` | a × b | a thin space (juxtaposition); `×` between two numbers |
//! | `a · b` | a × b | a centred dot |
//! | `a / b` | a ÷ b | inline `a/b` |
//! | `frac(a, b)` | a ÷ b | a stacked fraction |
//! | `a ^ b` | a to the power b | b as a superscript (outer parentheses of b dropped) |
//! | `sqrt(a)` | √a | a radical |
//! | `exp(a)` | e to the a | `e` with a as a superscript |
//! | `abs(a)`, `ceil(a)`, `floor(a)` | \|a\|, ⌈a⌉, ⌊a⌋ | those brackets |
//! | `sin(a)` .. `ln(a)` | the function | upright name, argument in parentheses |
//! | `min(a, b, ..)`, `max(..)` | Python's `min`/`max`, folded left to right (`compat::py_min`) | upright name |
//! | `(a)` | a | parentheses (always shown) |
//! | `sum(n in H: e)` | Σ over n in the harmonic set H (1, 3, ... up to `coupling.max_harmonic`), a left fold from 0 as Python's `sum()`; NaN for an invalid set | Σ with `n ∈ H` beneath |
//! | `cases(c1 => e1; ...; else => e)` | the first arm whose condition holds, else `e` (a NaN comparison is false, as in the engine) | a left brace, one row per arm: `e1   if c1` |
//! | `a < b` ... `a != b`, `and`, `or` | comparisons (numbers, or exact text; `none` matches an unset optional input) | `<`, `≤`, `>`, `≥`, `=`, `≠`, `and`, `or` |
//! | `"text"` | that text (a verdict) | the text, quoted |
//! | `n` | the index: the family member's index, or the Σ variable | italic n (a family member shows its digit) |
//! | `peak(n in H: A)` | the electrical angle φ in [0, π/2] at which Σ_{n∈H} A_n sin(nφ) is largest, A_n the expression: π/2 unless another maximum beats it by more than 1e-12 relative, exactly as E7 decides (`model::peak_off_half_pitch`); π/2 for an invalid set | `arg max` over φ of Σ_{n∈H} A sin(nφ) |
//! | `table("magnets", k, "br_T")` | a field of a static engine table, by key; `none` when the key is not in it ([`super::tables`] lists the tables and fields) | the field's symbol with the key: `B_{r,lib}(part_i)` |
//! | `... where [S_n] = e` | `[S_n]` stands for e (evaluated when first used) | the formula, then one line per binding: `S_n = e` |
//!
//! # Symbol markup
//!
//! A display symbol (a record's, a leaf term's, a local's) is `base [_ script] [^ script]`,
//! where `base` is one or more characters other than `_ ^ { }` and space, and `script` is
//! one character or `{...}`: `τ_p`, `B_{i,3}`, `S_{3}^{iron}`, `T_{pull,20}`. A `#` in a
//! family's symbol is its index (`B_{i,#}` is `B_{i,3}` for harmonic 3 and `B_{i,n}` under a Σ).
//! [`Symbol::parse`] splits it for the typesetter.

use std::fmt;

/// A parse error, at a character offset of the markup.
#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    pub at: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "at {}: {}", self.at, self.message)
    }
}

/// A reference to an input or result path.
#[derive(Clone, Debug, PartialEq)]
pub struct TermRef {
    /// The path; inside a Σ a family path, with `#` for the index.
    pub path: String,
    /// The unit the formula wants the value in, when not the term's own.
    pub unit: Option<String>,
    /// The factor from the term's unit to `unit`, set by the registry (1 without `unit`).
    pub scale: f64,
}

/// Binary operators. `Mul` and `Dot` both multiply; they differ only in how they are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    /// `*`: drawn as juxtaposition.
    Mul,
    /// `·`: drawn as a centred dot.
    Dot,
    /// `/`: drawn inline.
    Div,
    Pow,
}

/// Functions of the markup (`frac` has its own node).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Func {
    Sqrt,
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Abs,
    Min,
    Max,
    Ceil,
    Floor,
}

impl Func {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "sqrt" => Func::Sqrt,
            "sin" => Func::Sin,
            "cos" => Func::Cos,
            "tan" => Func::Tan,
            "sinh" => Func::Sinh,
            "cosh" => Func::Cosh,
            "tanh" => Func::Tanh,
            "exp" => Func::Exp,
            "ln" => Func::Ln,
            "abs" => Func::Abs,
            "min" => Func::Min,
            "max" => Func::Max,
            "ceil" => Func::Ceil,
            "floor" => Func::Floor,
            _ => return None,
        })
    }

    /// The markup name.
    pub const fn name(self) -> &'static str {
        match self {
            Func::Sqrt => "sqrt",
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Sinh => "sinh",
            Func::Cosh => "cosh",
            Func::Tanh => "tanh",
            Func::Exp => "exp",
            Func::Ln => "ln",
            Func::Abs => "abs",
            Func::Min => "min",
            Func::Max => "max",
            Func::Ceil => "ceil",
            Func::Floor => "floor",
        }
    }

    /// Whether the function takes any number (at least two) of arguments.
    const fn variadic(self) -> bool {
        matches!(self, Func::Min | Func::Max)
    }
}

/// The index sets a Σ can run over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexSet {
    /// The harmonics summed: 1, 3, ... up to `coupling.max_harmonic` (Addendum A3).
    Harmonics,
}

impl IndexSet {
    /// The input that chooses the set: an implicit term of every Σ over it.
    pub const fn selector(self) -> &'static str {
        match self {
            IndexSet::Harmonics => "coupling.max_harmonic",
        }
    }
}

/// Comparison operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// A condition of a `cases` arm.
#[derive(Clone, Debug, PartialEq)]
pub enum Cond {
    Rel(RelOp, Expr, Expr),
    And(Box<Cond>, Box<Cond>),
    Or(Box<Cond>, Box<Cond>),
}

/// A formula expression: the typesetter's input and the evaluator's.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// A number; `text` is how the markup wrote it (the typesetter shows that).
    Num {
        value: f64,
        text: String,
    },
    /// A text literal (a verdict).
    Text(String),
    /// `none`: an unset optional input.
    NoneLit,
    /// π.
    Pi,
    /// An input or result path.
    Term(TermRef),
    /// A family term inside a Σ: `path` holds `#` for the index.
    FamilyTerm(TermRef),
    /// The Σ index `n`.
    Index,
    /// A `where` binding, by position.
    Local(usize),
    Neg(Box<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Frac(Box<Expr>, Box<Expr>),
    Call(Func, Vec<Expr>),
    /// Parentheses the markup wrote (drawn; transparent to evaluation).
    Paren(Box<Expr>),
    Sum(IndexSet, Box<Expr>),
    /// `id` numbers the `cases` of a formula in parse order (the branch-coverage key).
    Cases {
        id: usize,
        arms: Vec<(Cond, Expr)>,
        otherwise: Box<Expr>,
    },
    /// The E7 pull-out angle of the amplitudes A_n (the expression, per index).
    Peak(IndexSet, Box<Expr>),
    /// A field of a static engine table ([`super::tables`]), by key.
    Table {
        table: String,
        key: Box<Expr>,
        field: String,
    },
}

/// A named sub-expression (`where [S_n] = ...`).
#[derive(Clone, Debug, PartialEq)]
pub struct Binding {
    /// Its display name, in symbol markup, the index substituted.
    pub symbol: String,
    pub expr: Expr,
}

/// A parsed formula.
#[derive(Clone, Debug, PartialEq)]
pub struct Formula {
    pub body: Expr,
    pub bindings: Vec<Binding>,
    /// How many `cases` the formula holds (their ids are 0 .. this).
    pub cases_count: usize,
}

impl Formula {
    /// Calls `f` on every expression node: the body, then each binding, depth first.
    pub fn visit(&self, f: &mut dyn FnMut(&Expr)) {
        visit_expr(&self.body, f);
        for b in &self.bindings {
            visit_expr(&b.expr, f);
        }
    }

    /// Mutable [`Formula::visit`].
    pub fn visit_mut(&mut self, f: &mut dyn FnMut(&mut Expr)) {
        visit_expr_mut(&mut self.body, f);
        for b in &mut self.bindings {
            visit_expr_mut(&mut b.expr, f);
        }
    }
}

fn visit_cond(c: &Cond, f: &mut dyn FnMut(&Expr)) {
    match c {
        Cond::Rel(_, a, b) => {
            visit_expr(a, f);
            visit_expr(b, f);
        }
        Cond::And(a, b) | Cond::Or(a, b) => {
            visit_cond(a, f);
            visit_cond(b, f);
        }
    }
}

fn visit_expr(e: &Expr, f: &mut dyn FnMut(&Expr)) {
    f(e);
    match e {
        Expr::Neg(a) | Expr::Paren(a) | Expr::Sum(_, a) | Expr::Peak(_, a) => visit_expr(a, f),
        Expr::Table { key, .. } => visit_expr(key, f),
        Expr::Bin(_, a, b) | Expr::Frac(a, b) => {
            visit_expr(a, f);
            visit_expr(b, f);
        }
        Expr::Call(_, args) => args.iter().for_each(|a| visit_expr(a, f)),
        Expr::Cases {
            arms, otherwise, ..
        } => {
            for (c, a) in arms {
                visit_cond(c, f);
                visit_expr(a, f);
            }
            visit_expr(otherwise, f);
        }
        Expr::Num { .. }
        | Expr::Text(_)
        | Expr::NoneLit
        | Expr::Pi
        | Expr::Term(_)
        | Expr::FamilyTerm(_)
        | Expr::Index
        | Expr::Local(_) => {}
    }
}

fn visit_cond_mut(c: &mut Cond, f: &mut dyn FnMut(&mut Expr)) {
    match c {
        Cond::Rel(_, a, b) => {
            visit_expr_mut(a, f);
            visit_expr_mut(b, f);
        }
        Cond::And(a, b) | Cond::Or(a, b) => {
            visit_cond_mut(a, f);
            visit_cond_mut(b, f);
        }
    }
}

fn visit_expr_mut(e: &mut Expr, f: &mut dyn FnMut(&mut Expr)) {
    f(e);
    match e {
        Expr::Neg(a) | Expr::Paren(a) | Expr::Sum(_, a) | Expr::Peak(_, a) => visit_expr_mut(a, f),
        Expr::Table { key, .. } => visit_expr_mut(key, f),
        Expr::Bin(_, a, b) | Expr::Frac(a, b) => {
            visit_expr_mut(a, f);
            visit_expr_mut(b, f);
        }
        Expr::Call(_, args) => args.iter_mut().for_each(|a| visit_expr_mut(a, f)),
        Expr::Cases {
            arms, otherwise, ..
        } => {
            for (c, a) in arms {
                visit_cond_mut(c, f);
                visit_expr_mut(a, f);
            }
            visit_expr_mut(otherwise, f);
        }
        Expr::Num { .. }
        | Expr::Text(_)
        | Expr::NoneLit
        | Expr::Pi
        | Expr::Term(_)
        | Expr::FamilyTerm(_)
        | Expr::Index
        | Expr::Local(_) => {}
    }
}

/// Parses a formula. `index` is the family member's index (`Some(3)` for harmonic 3): each
/// `#` in a term path or a local's symbol becomes its digits and `n` becomes that number;
/// `None` for a plain record, where `n` and `#` are allowed only inside a Σ.
pub fn parse(src: &str, index: Option<u32>) -> Result<Formula, ParseError> {
    let mut p = Parser {
        chars: src.chars().collect(),
        pos: 0,
        index,
        in_sum: false,
        cases_count: 0,
        local_names: Vec::new(),
    };
    let body = p.expr()?;
    let mut bindings = Vec::new();
    if p.eat_word("where") {
        loop {
            p.skip_ws();
            let at = p.pos;
            let symbol = p.local_symbol()?;
            if bindings.iter().any(|b: &Binding| b.symbol == symbol) {
                return Err(p.error_at(at, format!("[{symbol}] is bound twice")));
            }
            p.expect('=')?;
            let expr = p.expr()?;
            bindings.push(Binding { symbol, expr });
            if !p.eat(',') {
                break;
            }
        }
    }
    p.skip_ws();
    if p.pos < p.chars.len() {
        return Err(p.error(format!("unexpected '{}'", p.chars[p.pos])));
    }
    // Resolve every local reference to its binding; a binding may use only earlier ones.
    let names: Vec<String> = bindings.iter().map(|b| b.symbol.clone()).collect();
    let mut formula = Formula {
        body,
        bindings,
        cases_count: p.cases_count,
    };
    let mut used = vec![false; names.len()];
    let resolve = |e: &mut Expr, limit: usize, used: &mut Vec<bool>| -> Result<(), String> {
        let mut result = Ok(());
        visit_expr_mut(e, &mut |node| {
            if let Expr::Local(i) = node {
                let name = &p.local_names[*i];
                match names.iter().position(|n| n == name) {
                    Some(k) if k < limit => {
                        used[k] = true;
                        *node = Expr::Local(k);
                    }
                    Some(_) => result = Err(format!("[{name}] is used before it is bound")),
                    None => result = Err(format!("[{name}] is not bound")),
                }
            }
        });
        result
    };
    resolve(&mut formula.body, names.len(), &mut used).map_err(|m| p.error_at(0, m))?;
    for k in 0..formula.bindings.len() {
        let mut expr = std::mem::replace(&mut formula.bindings[k].expr, Expr::NoneLit);
        resolve(&mut expr, k, &mut used).map_err(|m| p.error_at(0, m))?;
        formula.bindings[k].expr = expr;
    }
    if let Some(k) = used.iter().position(|u| !u) {
        return Err(p.error_at(0, format!("[{}] is bound but never used", names[k])));
    }
    Ok(formula)
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    index: Option<u32>,
    in_sum: bool,
    cases_count: usize,
    /// Local names in order of first reference (resolved to bindings after the parse).
    local_names: Vec<String>,
}

const KEYWORDS: &[&str] = &[
    "cases", "else", "and", "or", "where", "sum", "peak", "table", "in", "n", "pi", "none", "H",
    "frac",
];

impl Parser {
    fn error(&self, message: String) -> ParseError {
        self.error_at(self.pos, message)
    }

    fn error_at(&self, at: usize, message: String) -> ParseError {
        ParseError { at, message }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos).copied()
    }

    fn peek2(&mut self) -> Option<char> {
        self.skip_ws();
        self.chars.get(self.pos + 1).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn eat_str(&mut self, s: &str) -> bool {
        self.skip_ws();
        let n = s.chars().count();
        if self.chars.len() >= self.pos + n
            && self.chars[self.pos..self.pos + n]
                .iter()
                .copied()
                .eq(s.chars())
        {
            self.pos += n;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char) -> Result<(), ParseError> {
        if self.eat(c) {
            Ok(())
        } else {
            let got = self
                .peek()
                .map_or("the end".to_owned(), |g| format!("'{g}'"));
            Err(self.error(format!("expected '{c}', got {got}")))
        }
    }

    fn is_word_char(c: char) -> bool {
        c.is_alphabetic() || c == '_'
    }

    /// The identifier at the cursor, without consuming it.
    fn peek_word(&mut self) -> Option<String> {
        self.skip_ws();
        let start = self.pos;
        let mut end = start;
        while end < self.chars.len() && Self::is_word_char(self.chars[end]) {
            end += 1;
        }
        (end > start).then(|| self.chars[start..end].iter().collect())
    }

    fn eat_word(&mut self, word: &str) -> bool {
        if self.peek_word().as_deref() == Some(word) {
            self.pos += word.chars().count();
            true
        } else {
            false
        }
    }

    fn expect_word(&mut self, word: &str) -> Result<(), ParseError> {
        if self.eat_word(word) {
            Ok(())
        } else {
            Err(self.error(format!("expected '{word}'")))
        }
    }

    fn substitute(&self, text: &str) -> Result<String, String> {
        match self.index {
            Some(n) => Ok(text.replace('#', &n.to_string())),
            None if text.contains('#') && !self.in_sum => {
                Err(format!("'#' outside a Σ in a plain record: {text}"))
            }
            None => Ok(text.to_owned()),
        }
    }

    fn expr(&mut self) -> Result<Expr, ParseError> {
        self.additive()
    }

    /// `cases(...)`, after the keyword.
    fn cases(&mut self) -> Result<Expr, ParseError> {
        self.expect('(')?;
        let id = self.cases_count;
        self.cases_count += 1;
        let mut arms = Vec::new();
        loop {
            if self.eat_word("else") {
                if arms.is_empty() {
                    return Err(self.error("cases needs at least one arm before else".into()));
                }
                if !self.eat_str("=>") {
                    return Err(self.error("expected '=>' after else".into()));
                }
                let otherwise = self.expr()?;
                self.expect(')')?;
                return Ok(Expr::Cases {
                    id,
                    arms,
                    otherwise: Box::new(otherwise),
                });
            }
            let cond = self.cond()?;
            if !self.eat_str("=>") {
                return Err(self.error("expected '=>' after a condition".into()));
            }
            let value = self.expr()?;
            arms.push((cond, value));
            self.expect(';')?;
        }
    }

    fn cond(&mut self) -> Result<Cond, ParseError> {
        let mut left = self.conj()?;
        while self.eat_word("or") {
            let right = self.conj()?;
            left = Cond::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn conj(&mut self) -> Result<Cond, ParseError> {
        let mut left = self.rel()?;
        while self.eat_word("and") {
            let right = self.rel()?;
            left = Cond::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn rel(&mut self) -> Result<Cond, ParseError> {
        let a = self.additive()?;
        let op = if self.eat_str("<=") {
            RelOp::Le
        } else if self.eat_str(">=") {
            RelOp::Ge
        } else if self.eat_str("!=") {
            RelOp::Ne
        } else if self.peek() == Some('=') && self.peek2() != Some('>') {
            self.pos += 1;
            RelOp::Eq
        } else if self.eat('<') {
            RelOp::Lt
        } else if self.eat('>') {
            RelOp::Gt
        } else {
            return Err(self.error("expected a comparison".into()));
        };
        let b = self.additive()?;
        Ok(Cond::Rel(op, a, b))
    }

    fn additive(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.product()?;
        loop {
            let op = match self.peek() {
                Some('+') => BinOp::Add,
                Some('-') => BinOp::Sub,
                _ => return Ok(left),
            };
            self.pos += 1;
            let right = self.product()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
    }

    fn product(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                Some('*') => BinOp::Mul,
                Some('·') => BinOp::Dot,
                Some('/') => BinOp::Div,
                _ => return Ok(left),
            };
            self.pos += 1;
            let right = self.unary()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
    }

    fn unary(&mut self) -> Result<Expr, ParseError> {
        if self.eat('-') {
            Ok(Expr::Neg(Box::new(self.unary()?)))
        } else {
            self.power()
        }
    }

    fn power(&mut self) -> Result<Expr, ParseError> {
        let base = self.atom()?;
        if self.eat('^') {
            let exponent = self.unary()?;
            Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exponent)))
        } else {
            Ok(base)
        }
    }

    fn number(&mut self) -> Result<Expr, ParseError> {
        let start = self.pos;
        let digits = |p: &mut Parser| {
            let s = p.pos;
            while p.pos < p.chars.len() && p.chars[p.pos].is_ascii_digit() {
                p.pos += 1;
            }
            p.pos > s
        };
        digits(self);
        if self.pos < self.chars.len() && self.chars[self.pos] == '.' {
            self.pos += 1;
            if !digits(self) {
                return Err(self.error("expected digits after '.'".into()));
            }
        }
        if self.pos < self.chars.len() && matches!(self.chars[self.pos], 'e' | 'E') {
            let save = self.pos;
            self.pos += 1;
            if self.pos < self.chars.len() && matches!(self.chars[self.pos], '+' | '-') {
                self.pos += 1;
            }
            if !digits(self) {
                self.pos = save; // not an exponent
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        let value = text
            .parse::<f64>()
            .map_err(|e| self.error_at(start, format!("bad number {text}: {e}")))?;
        Ok(Expr::Num { value, text })
    }

    fn delimited(&mut self, close: char, what: &str) -> Result<String, ParseError> {
        let start = self.pos;
        let mut text = String::new();
        loop {
            match self.chars.get(self.pos) {
                None => return Err(self.error_at(start, format!("unterminated {what}"))),
                Some(&c) if c == close => {
                    self.pos += 1;
                    return Ok(text);
                }
                Some(&c) => {
                    text.push(c);
                    self.pos += 1;
                }
            }
        }
    }

    fn quoted(&mut self, what: &str) -> Result<String, ParseError> {
        if self.eat('"') {
            self.delimited('"', what)
        } else {
            Err(self.error(format!("expected the {what} as a quoted text")))
        }
    }

    fn local_symbol(&mut self) -> Result<String, ParseError> {
        self.expect('[')?;
        let at = self.pos;
        let raw = self.delimited(']', "local name")?;
        let symbol = self
            .substitute(raw.trim())
            .map_err(|m| self.error_at(at, m))?;
        Symbol::parse(&symbol).map_err(|m| self.error_at(at, m))?;
        Ok(symbol)
    }

    fn atom(&mut self) -> Result<Expr, ParseError> {
        let at = self.pos;
        match self.peek() {
            None => Err(self.error("unexpected end of formula".into())),
            Some(c) if c.is_ascii_digit() => self.number(),
            Some('"') => {
                self.pos += 1;
                Ok(Expr::Text(self.delimited('"', "text")?))
            }
            Some('{') => {
                self.pos += 1;
                let raw = self.delimited('}', "term")?;
                let (path, unit) = match raw.split_once('|') {
                    Some((p, u)) => (p.trim().to_owned(), Some(u.trim().to_owned())),
                    None => (raw.trim().to_owned(), None),
                };
                if path.is_empty() || path.contains(char::is_whitespace) {
                    return Err(self.error_at(at, format!("bad term path '{path}'")));
                }
                let family = path.contains('#') && self.index.is_none();
                let path = self.substitute(&path).map_err(|m| self.error_at(at, m))?;
                let term = TermRef {
                    path,
                    unit,
                    scale: 1.0,
                };
                Ok(if family {
                    Expr::FamilyTerm(term)
                } else {
                    Expr::Term(term)
                })
            }
            Some('[') => {
                let symbol = self.local_symbol()?;
                let i = match self.local_names.iter().position(|n| *n == symbol) {
                    Some(i) => i,
                    None => {
                        self.local_names.push(symbol);
                        self.local_names.len() - 1
                    }
                };
                Ok(Expr::Local(i))
            }
            Some('(') => {
                self.pos += 1;
                let inner = self.expr()?;
                self.expect(')')?;
                Ok(Expr::Paren(Box::new(inner)))
            }
            Some('π') => {
                self.pos += 1;
                Ok(Expr::Pi)
            }
            Some(_) => {
                let word = self
                    .peek_word()
                    .ok_or_else(|| self.error(format!("unexpected '{}'", self.chars[self.pos])))?;
                self.pos += word.chars().count();
                match word.as_str() {
                    "pi" => Ok(Expr::Pi),
                    "none" => Ok(Expr::NoneLit),
                    "cases" => self.cases(),
                    "n" => match self.index {
                        Some(n) => Ok(Expr::Num {
                            value: f64::from(n),
                            text: n.to_string(),
                        }),
                        None if self.in_sum => Ok(Expr::Index),
                        None => Err(self.error_at(at, "'n' outside a Σ in a plain record".into())),
                    },
                    "frac" => {
                        self.expect('(')?;
                        let num = self.expr()?;
                        self.expect(',')?;
                        let den = self.expr()?;
                        self.expect(')')?;
                        Ok(Expr::Frac(Box::new(num), Box::new(den)))
                    }
                    "sum" | "peak" => {
                        if self.index.is_some() || self.in_sum {
                            return Err(self
                                .error_at(at, format!("'{word}' inside a family record or a Σ")));
                        }
                        self.expect('(')?;
                        self.expect_word("n")?;
                        self.expect_word("in")?;
                        self.expect_word("H")?;
                        self.expect(':')?;
                        self.in_sum = true;
                        let body = self.expr();
                        self.in_sum = false;
                        let body = Box::new(body?);
                        self.expect(')')?;
                        Ok(if word == "sum" {
                            Expr::Sum(IndexSet::Harmonics, body)
                        } else {
                            Expr::Peak(IndexSet::Harmonics, body)
                        })
                    }
                    "table" => {
                        self.expect('(')?;
                        let table = self.quoted("table name")?;
                        self.expect(',')?;
                        let key = self.expr()?;
                        self.expect(',')?;
                        let field = self.quoted("field name")?;
                        self.expect(')')?;
                        Ok(Expr::Table {
                            table,
                            key: Box::new(key),
                            field,
                        })
                    }
                    name => {
                        let func = Func::from_name(name).ok_or_else(|| {
                            if KEYWORDS.contains(&name) {
                                self.error_at(at, format!("'{name}' is not allowed here"))
                            } else {
                                self.error_at(at, format!("unknown name '{name}'"))
                            }
                        })?;
                        self.expect('(')?;
                        let mut args = vec![self.expr()?];
                        while self.eat(',') {
                            args.push(self.expr()?);
                        }
                        self.expect(')')?;
                        let ok = if func.variadic() {
                            args.len() >= 2
                        } else {
                            args.len() == 1
                        };
                        if !ok {
                            return Err(self.error_at(
                                at,
                                format!(
                                    "{name} takes {} argument(s), got {}",
                                    if func.variadic() {
                                        "two or more"
                                    } else {
                                        "one"
                                    },
                                    args.len()
                                ),
                            ));
                        }
                        Ok(Expr::Call(func, args))
                    }
                }
            }
        }
    }
}

/// A display symbol split for the typesetter: `B_{i,3}` is base `B`, subscript `i,3`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Symbol {
    pub base: String,
    pub sub: Option<String>,
    pub sup: Option<String>,
}

impl Symbol {
    /// Parses symbol markup (module docs); `#` must already be substituted by the caller
    /// when it stands for an index.
    pub fn parse(text: &str) -> Result<Symbol, String> {
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        let mut base = String::new();
        while i < chars.len() && !matches!(chars[i], '_' | '^' | '{' | '}' | ' ') {
            base.push(chars[i]);
            i += 1;
        }
        if base.is_empty() {
            return Err(format!("symbol '{text}' has no base"));
        }
        let script = |i: &mut usize| -> Result<String, String> {
            match chars.get(*i) {
                Some('{') => {
                    let start = *i + 1;
                    let end = chars[start..]
                        .iter()
                        .position(|&c| c == '}')
                        .map(|k| start + k)
                        .ok_or_else(|| format!("symbol '{text}': unclosed '{{'"))?;
                    *i = end + 1;
                    let s: String = chars[start..end].iter().collect();
                    if s.is_empty() {
                        return Err(format!("symbol '{text}': empty script"));
                    }
                    Ok(s)
                }
                Some(&c) if !matches!(c, '_' | '^' | '}' | ' ') => {
                    *i += 1;
                    Ok(c.to_string())
                }
                _ => Err(format!("symbol '{text}': missing script")),
            }
        };
        let mut sub = None;
        let mut sup = None;
        if chars.get(i) == Some(&'_') {
            i += 1;
            sub = Some(script(&mut i)?);
        }
        if chars.get(i) == Some(&'^') {
            i += 1;
            sup = Some(script(&mut i)?);
        }
        if i != chars.len() {
            return Err(format!(
                "symbol '{text}': unexpected text after the scripts"
            ));
        }
        Ok(Symbol { base, sub, sup })
    }

    /// Plain-text form: `B_{i,3}` stays `B_i,3`-free of braces for one-character scripts.
    pub fn plain(&self) -> String {
        let script = |s: &str| {
            if s.chars().count() == 1 {
                s.to_owned()
            } else {
                format!("{{{s}}}")
            }
        };
        let mut out = self.base.clone();
        if let Some(s) = &self.sub {
            out.push('_');
            out.push_str(&script(s));
        }
        if let Some(s) = &self.sup {
            out.push('^');
            out.push_str(&script(s));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
````

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
#[cfg(test)]
mod tests {
    use super::*;
```

with:

```rust
//! The static engine tables a formula can read with `table("name", key, "field")`: the
//! magnet library, the grade table and the back-iron materials, each field as the engine
//! uses it with every approved correction on (the explorer describes what users see).
//!
//! A lookup returns `none` when the key is not in the table (a part name that is not a
//! library part, a blank grade), which is how the engine's own selections branch.

use crate::engine::deviations::Deviations;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::material_library::{BACK_IRON_CHOICES, chosen};
use crate::engine::meta::Value;

/// One readable field: its table, name, display symbol and unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableField {
    pub table: &'static str,
    pub field: &'static str,
    /// Symbol markup; the typesetter writes the key after it in parentheses.
    pub symbol: &'static str,
    pub unit: &'static str,
    pub label: &'static str,
}

/// Every field a formula may read. The registry refuses any other `table(...)`.
#[rustfmt::skip]
pub const TABLE_FIELDS: &[TableField] = &[
    TableField { table: "magnets", field: "length_mm", symbol: "L_{lib}", unit: "mm", label: "Library part axial length" },
    TableField { table: "magnets", field: "width_mm", symbol: "w_{lib}", unit: "mm", label: "Library part tangential width" },
    TableField { table: "magnets", field: "thickness_mm", symbol: "t_{lib}", unit: "mm", label: "Library part radial thickness" },
    TableField { table: "magnets", field: "br_T", symbol: "B_{r,lib}", unit: "T", label: "Library part remanence at 20 °C (E3: the N42SH grade's)" },
    TableField { table: "grades", field: "br_T", symbol: "B_{r,grade}", unit: "T", label: "Grade remanence at 20 °C" },
    TableField { table: "grades", field: "alpha_br_per_C", symbol: "α_{grade}", unit: "1/°C", label: "Grade Br temperature coefficient" },
    TableField { table: "back_iron", field: "ferromagnetic", symbol: "ferro", unit: "-", label: "Back-iron material is ferromagnetic (1) or not (0)" },
];

/// The field's metadata, or `None` if a formula may not read it.
pub fn field(table: &str, field: &str) -> Option<&'static TableField> {
    TABLE_FIELDS
        .iter()
        .find(|f| f.table == table && f.field == field)
}

/// The value of `field` in `table` at `key`: `Ok(None)` when the key is not in the table,
/// `Err` for a table or field no formula may read, or a key of the wrong type.
pub fn lookup(table: &str, key: &Value, field_name: &str) -> Result<Option<Value>, String> {
    if field(table, field_name).is_none() {
        return Err(if TABLE_FIELDS.iter().any(|f| f.table == table) {
            format!("table {table} has no field {field_name}")
        } else {
            format!("no table {table}")
        });
    }
    let text = |k: &Value| match k {
        Value::Text(s) => Ok(s.clone()),
        other => Err(format!("table {table}: key {other:?} is not a text")),
    };
    Ok(match table {
        "magnets" => library::lookup(&text(key)?).map(|spec| {
            Value::Num(match field_name {
                "length_mm" => spec.length_mm,
                "width_mm" => spec.width_mm,
                "thickness_mm" => spec.thickness_mm,
                "br_T" => library::br_T(spec, Deviations::ALL),
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }),
        "grades" => grades::grade(&text(key)?).map(|g| {
            Value::Num(match field_name {
                "br_T" => g.br_T,
                "alpha_br_per_C" => g.alpha_br_per_C,
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }),
        "back_iron" => {
            // A code: an integer input, or the same number after arithmetic promoted it.
            let code = match key {
                Value::Int(c) => *c,
                Value::Num(x) if x.fract() == 0.0 && x.abs() < 1e15 => *x as i64,
                other => return Err(format!("table back_iron: key {other:?} is not a code")),
            };
            chosen(&BACK_IRON_CHOICES, code).map(|m| Value::Int(i64::from(m.ferromagnetic)))
        }
        _ => unreachable!("checked against TABLE_FIELDS"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib explain 2>&1 | grep "test result"
```

Expected: `test result: ok. 20 passed` (the markup, evaluator and tables unit tests).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task2.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/mod.rs magcoupling-rs/src/engine/explain
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): the equation markup, its evaluator and the static tables

One text per explained result that is both what the panel typesets and what
the drift guard evaluates: the formula and symbol grammar and parser, an
evaluator that traces value and condition terms, and the engine tables a
formula reads (magnet library, grades, back iron), corrections on.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 3: The registry, the torque-chain records and the drift guard

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. The physics review in Step 6 runs on the session model.

A record states what one result IS, one step from its terms (`σ = Σ σ_n`, never the chain behind them); unit, label and
workbook cell come from the result's metadata, so they cannot drift. The registry parses and checks every record once: each term
path exists, each unit conversion is known (`CONVERSIONS`), each table field is readable, every term has a symbol and no two paths
share one, no leaf symbol is dead, and the graph has no cycle; a formula reads each path in one unit (the term list shows each
term once, in that unit), and holds nothing the typesetter would draw two ways (it draws `a / b` inline and only the parentheses
written, so an inline `a / b` or a Σ as a factor, or a power of a symbol that already has a superscript, must be parenthesized: the
rules beside the markup table); a bad record fails `Registry::build` with every reason listed. The plain rendering (the review
sheet the physics reviewer signs) parenthesizes a stacked fraction that is a factor, so it reads as the tree does. The
torque chain (decision 31's first chain, 54 scope paths) is the tracer: with the geometry, magnet resolution, material circuit and
calibration records its terms need, every term drills down to an input. The drift guard evaluates every equation over the engine's
own term values at the defaults, at every differential case (3,391, loaded exactly as `tests/differential.rs` loads them, now
through the shared `common::load_cases`) and at each case again under 11 augmentations, the Rust-only inputs the Python generator
never varies (harmonic sets 1, 3, 7, 9, 11; back irons 1018, 304, 6061; the grade mode; the axial override; everything at once),
without which the harmonic 7 to 11 records would compare 0 with 0; it requires every `cases` arm to be taken (two are listed
unreachable with their reasons), every numeric record to vary, every value term of every record (a term its value moves with at
a point) to take two values somewhere, since a term seen at one value cannot be told from a literal of it, and every E7 angle to
leave half a pitch somewhere. The points are
independent and run on every core.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/record.rs` (`Record`, `Family`, `Eval`, `record`, `family`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` (`RECORDS`, `FAMILIES`: one entry per batch)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/torque.rs` (the torque chain: 73 records and 9 families (127 equations))
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs` (`Registry`, `Equation`, `Design`, `TermKind`, `TermRow`, `CONVERSIONS`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs` (`SYMBOLS`: display symbols of inputs and cell-only terms)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs` (`SCOPE`: decision 31's chains and dashboard paths, status per chain)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/render.rs` (`plain`: one-line text rendering (a stacked fraction as a factor in parentheses))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs` (the typesetting rules the registry enforces, beside the markup table)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs` (every submodule but notes; re-exports)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/common/mod.rs` (`Case`, `load_cases`, `FULL`, `differential_files` (moved from differential.rs))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/differential.rs` (uses the shared loader)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (drift guard, registry, scope, drill-down, review sheet)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (layout row, tests row, the explorer section)

**Interfaces:**
- Consumes: Task 2's `markup`, `eval`, `tables`; Task 1's Rust-only results; `api::{DesignInputs, DesignResults, compute_all}`, `meta::{input_rows, result_rows, InputMeta, ResultMeta, InputSet, ResultSet}`, `compat::parity_close`, `deviations::DeviationId`.
- Produces (every later task uses these):
  - `record::record(target, symbol, formula) -> Record` (const), `Record::corrected(&'static [DeviationId])`, `Record::custom(CustomEval)`, `record::family(Record, &'static [u32]) -> Family`;
  - `records::RECORDS: &[&[Record]]`, `records::FAMILIES: &[&[Family]]` (a batch appends its file's slices);
  - `Registry::build() -> Registry` (panics listing every problem), `try_build`, `from_parts(&[Record], &[Family], &[(&str, &str)])`, `equations()`, `equation_for(path)`, `used_by(path) -> &[String]`, `is_leaf_input`, `term_kind(path) -> Option<TermKind>`, `symbol(path)`, `family_symbol(template)`, `family_members(template, &dyn TermSource)`, `graph()`, `upstream(path) -> BTreeSet<String>`, `term_values`, `term_rows(&Equation, &dyn TermSource) -> Vec<TermRow>`, `evaluate(&Equation, &dyn TermSource, Option<&mut Trace>)`;
  - `registry::Design { inputs: &DesignInputs, results: &DesignResults }` (a `TermSource`: a result path, else an input path);
  - `symbols::SYMBOLS: &[(&str, &str)]`; `scope::{SCOPE, Chain { id, status, paths }, Status::{Explained, Pending}}`; `render::plain(&Registry, symbol, &Formula) -> String`;
  - in `tests/explain.rs`: `augmentations()`, `guard_points()`, `UNREACHABLE_ARMS`, the tests `every_record_reproduces_the_engine_everywhere`, `input_and_result_paths_are_disjoint`, `the_registry_is_consistent`, `review_sheet` (ignored), `every_explained_chain_drills_down_to_inputs`, `explained_chains_have_every_record_and_scope_paths_exist`.

- [ ] **Step 1: Write the failing tests**

The shared differential loader moves into `tests/common/mod.rs` (DRY: the drift guard loads the cases exactly as the differential test does); `tests/explain.rs` holds the guard and the structure tests.

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/common/mod.rs`, replace:

```rust
    };
    format!("{} failure(s):\n{}{tail}", failures.len(), shown.join("\n"))
}
```

with:

```rust
    };
    format!("{} failure(s):\n{}{tail}", failures.len(), shown.join("\n"))
}

/// The data file whose cases vary every input group and compare every result.
pub const FULL: &str = "full";

/// A generated differential case (`tests/data/differential/<module>.json`): inputs by path,
/// and the Python results of its module. Shared by `differential.rs` and the drift guard.
pub struct Case {
    pub id: u64,
    pub tag: String,
    pub inputs: BTreeMap<String, Value>,
    pub results: BTreeMap<String, Value>,
}

pub fn load_cases(module: &str) -> Vec<Case> {
    let doc = read_json(&data_path(&format!("differential/{module}.json")));
    assert_eq!(doc["module"], module);
    let paths = |key: &str| -> Vec<String> {
        doc[key]
            .as_array()
            .unwrap_or_else(|| panic!("{module}: {key} is an array"))
            .iter()
            .map(|p| p.as_str().expect("a path").to_owned())
            .collect()
    };
    let (input_paths, result_paths) = (paths("input_paths"), paths("result_paths"));
    doc["cases"]
        .as_array()
        .expect("a cases array")
        .iter()
        .map(|c| {
            let id = c["id"].as_u64().expect("a case id");
            let zip = |paths: &[String], key: &str| -> BTreeMap<String, Value> {
                let values = c[key]
                    .as_array()
                    .unwrap_or_else(|| panic!("case {id}: {key}"));
                assert_eq!(
                    values.len(),
                    paths.len(),
                    "{module} case {id}: {key} length"
                );
                paths
                    .iter()
                    .cloned()
                    .zip(values.iter().map(json_to_value))
                    .collect()
            };
            Case {
                id,
                tag: c["tag"].as_str().expect("a case tag").to_owned(),
                inputs: zip(&input_paths, "inputs"),
                results: zip(&result_paths, "results"),
            }
        })
        .collect()
}

/// Every differential data file with cases: one per ported result group, then the full run.
pub fn differential_files() -> Vec<&'static str> {
    PORTED_RESULTS
        .iter()
        .map(|p| p.group)
        .chain([FULL])
        .collect()
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/differential.rs`, replace:

```rust

use std::collections::{BTreeMap, BTreeSet};

use common::{PORTED_RESULTS, data_path, group_of, json_to_value, read_json, report};
use magcoupling::engine::api::{DesignInputs, compute_all, compute_all_with};
use magcoupling::engine::compat::{
    ceiling, floor_, fmt_fixed, fmt_num, parity_close, py_repr, text0,
};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputSet, Value, input_rows, result_rows};

/// The data file whose cases vary every input group and compare every result.
const FULL: &str = "full";

/// A generated case: inputs by path, and the Python results of its module.
struct Case {
    id: u64,
    tag: String,
    inputs: BTreeMap<String, Value>,
    results: BTreeMap<String, Value>,
}

fn load_cases(module: &str) -> Vec<Case> {
    let doc = read_json(&data_path(&format!("differential/{module}.json")));
    assert_eq!(doc["module"], module);
    let paths = |key: &str| -> Vec<String> {
        doc[key]
            .as_array()
            .unwrap_or_else(|| panic!("{module}: {key} is an array"))
            .iter()
            .map(|p| p.as_str().expect("a path").to_owned())
            .collect()
    };
    let (input_paths, result_paths) = (paths("input_paths"), paths("result_paths"));
    doc["cases"]
        .as_array()
        .expect("a cases array")
        .iter()
        .map(|c| {
            let id = c["id"].as_u64().expect("a case id");
            let zip = |paths: &[String], key: &str| -> BTreeMap<String, Value> {
                let values = c[key]
                    .as_array()
                    .unwrap_or_else(|| panic!("case {id}: {key}"));
                assert_eq!(
                    values.len(),
                    paths.len(),
                    "{module} case {id}: {key} length"
                );
                paths
                    .iter()
                    .cloned()
                    .zip(values.iter().map(json_to_value))
                    .collect()
            };
            Case {
                id,
                tag: c["tag"].as_str().expect("a case tag").to_owned(),
                inputs: zip(&input_paths, "inputs"),
                results: zip(&result_paths, "results"),
            }
        })
        .collect()
}

/// The Rust results of `module` for a case, deviations off.
fn rust_results(module: &str, case: &Case) -> Result<BTreeMap<String, Value>, String> {
```

with:

```rust

use std::collections::{BTreeMap, BTreeSet};

use common::{Case, FULL, PORTED_RESULTS, data_path, group_of, load_cases, read_json, report};
use magcoupling::engine::api::{DesignInputs, compute_all, compute_all_with};
use magcoupling::engine::compat::{
    ceiling, floor_, fmt_fixed, fmt_num, parity_close, py_repr, text0,
};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::{FieldType, InputSet, Value, input_rows, result_rows};

/// The Rust results of `module` for a case, deviations off.
fn rust_results(module: &str, case: &Case) -> Result<BTreeMap<String, Value>, String> {
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` with exactly this content:

```rust
//! Addendum A2 explanation layer: the drift guard, the registry's structure and the v1
//! scope.
//!
//! **Drift guard.** Every equation record is evaluated over the engine's own term values and
//! must reproduce the engine's result by the parity rule (1e-9 relative, 1e-12 absolute;
//! text exact), corrections on (`compute_all`, what users see), at: the defaults; every
//! differential case (`tests/data/differential/*.json`, 3,391 cases, starting from the
//! corrected defaults); and each of those again under every **augmentation**, the Rust-only
//! inputs the Python generator never varies (the harmonic set, the back-iron material, the
//! grade mode, the axial override), without which the tau7-tau11 records would compare
//! 0 with 0. Anti-vacuity checks require every `cases` arm of every record to be taken, every
//! numeric record and every value term of every record to take two values, and the E7
//! angles to leave half a pitch.

mod common;

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::FRAC_PI_2;

use common::{differential_files, load_cases, report};
use magcoupling::engine::api::{DesignInputs, compute_all};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::explain::markup::{Expr, Symbol};
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, render};
use magcoupling::engine::meta::{InputSet, ResultSet, Value, input_rows, result_rows};

/// One input set the guard evaluates at.
struct Point {
    label: String,
    inputs: DesignInputs,
}

/// Rust-only inputs the differential generator never varies, applied on top of each case.
fn augmentations() -> Vec<(&'static str, Vec<(&'static str, Value)>)> {
    let text = |s: &str| Value::Text(s.to_owned());
    let mut v: Vec<(&'static str, Vec<(&'static str, Value)>)> = vec![("as generated", vec![])];
    for (label, code) in [
        ("harmonics 1", 1),
        ("harmonics 3", 3),
        ("harmonics 7", 7),
        ("harmonics 9", 9),
        ("harmonics 11", 11),
    ] {
        v.push((label, vec![("coupling.max_harmonic", Value::Int(code))]));
    }
    for (label, code) in [
        ("back iron 1018", 2),
        ("back iron 304", 7),
        ("back iron 6061", 8),
    ] {
        v.push((label, vec![("materials.parts.back_iron", Value::Int(code))]));
    }
    v.push((
        "grade mode",
        vec![
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("N52")),
            ("coupling.magnets.part_outer", text("")),
            ("coupling.magnets.grade_outer", text("Y30")),
        ],
    ));
    v.push((
        "axial 8 mm",
        vec![("coupling.magnets.axial_length_mm", Value::Num(8.0))],
    ));
    v.push((
        "everything",
        vec![
            ("coupling.max_harmonic", Value::Int(11)),
            ("materials.parts.back_iron", Value::Int(8)),
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("N35EH")),
            ("coupling.magnets.axial_length_mm", Value::Num(20.0)),
        ],
    ));
    v
}

/// The defaults, then every differential case under every augmentation.
fn guard_points() -> Vec<Point> {
    let mut points = vec![Point {
        label: "defaults".into(),
        inputs: DesignInputs::default(),
    }];
    let augmentations = augmentations();
    for file in differential_files() {
        for case in load_cases(file) {
            let mut base = DesignInputs::default();
            for (path, value) in &case.inputs {
                base.set(path, value.clone())
                    .unwrap_or_else(|e| panic!("{file} case {}: {e}", case.id));
            }
            for (label, sets) in &augmentations {
                let mut inputs = base.clone();
                for (path, value) in sets {
                    inputs
                        .set(path, value.clone())
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                }
                points.push(Point {
                    label: format!("{file} case {} ({}), {label}", case.id, case.tag),
                    inputs,
                });
            }
        }
    }
    points
}

fn same(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) if x.is_nan() && y.is_nan() => true,
        _ => parity_close(a, b),
    }
}

/// What the guard learned across all points.
#[derive(Default)]
struct Coverage {
    arms: BTreeMap<String, BTreeSet<(usize, usize)>>,
    values: BTreeMap<String, BTreeSet<u64>>,
    angles_off_half_pitch: BTreeMap<String, usize>,
    /// Per equation (in `Registry::equations` order): each value term's distinct values,
    /// up to two.
    term_values: Vec<BTreeMap<String, Vec<Value>>>,
}

/// Adds `v` to `seen` unless it holds it already or holds two (two show the term varies;
/// every NaN is one value).
fn note_value(seen: &mut Vec<Value>, v: Value) {
    let same = |a: &Value| match (a, &v) {
        (Value::Num(x), Value::Num(y)) => x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
        (a, b) => a == b,
    };
    if seen.len() < 2 && !seen.iter().any(same) {
        seen.push(v);
    }
}

/// Evaluates every record at every point of `points`: the failures and the coverage.
fn guard(registry: &Registry, points: &[Point]) -> (Vec<String>, Coverage) {
    let mut failures = Vec::new();
    let mut cov = Coverage {
        term_values: vec![BTreeMap::new(); registry.equations().len()],
        ..Coverage::default()
    };
    for p in points {
        let results = compute_all(&p.inputs);
        let src = Design {
            inputs: &p.inputs,
            results: &results,
        };
        for (i, eq) in registry.equations().iter().enumerate() {
            let want = results
                .get(&eq.target)
                .expect("a record targets a result path");
            let mut trace = Trace::default();
            match registry.evaluate(eq, &src, Some(&mut trace)) {
                Ok(got) if same(&got, &want) => {}
                Ok(got) => failures.push(format!(
                    "{} at {}: record {got:?}, engine {want:?}",
                    eq.target, p.label
                )),
                Err(e) => failures.push(format!("{} at {}: {e}", eq.target, p.label)),
            }
            cov.arms
                .entry(eq.target.clone())
                .or_default()
                .extend(trace.arms);
            let terms = &mut cov.term_values[i];
            for term in trace.value_terms {
                match terms.get_mut(&term) {
                    Some(seen) => {
                        if seen.len() < 2 {
                            note_value(seen, src.value(&term).unwrap_or(Value::None));
                        }
                    }
                    None => {
                        let v = src.value(&term).unwrap_or(Value::None);
                        terms.insert(term, vec![v]);
                    }
                }
            }
            if let Value::Num(x) = want {
                cov.values
                    .entry(eq.target.clone())
                    .or_default()
                    .insert(x.to_bits());
                if eq.target.ends_with("angle_rad") && x != FRAC_PI_2 {
                    *cov.angles_off_half_pitch
                        .entry(eq.target.clone())
                        .or_default() += 1;
                }
            }
        }
    }
    (failures, cov)
}

impl Coverage {
    fn merge(&mut self, other: Coverage) {
        for (k, v) in other.arms {
            self.arms.entry(k).or_default().extend(v);
        }
        for (k, v) in other.values {
            self.values.entry(k).or_default().extend(v);
        }
        for (k, n) in other.angles_off_half_pitch {
            *self.angles_off_half_pitch.entry(k).or_default() += n;
        }
        if self.term_values.is_empty() {
            self.term_values = other.term_values;
        } else {
            for (mine, theirs) in self.term_values.iter_mut().zip(other.term_values) {
                for (term, values) in theirs {
                    let seen = mine.entry(term).or_default();
                    for v in values {
                        note_value(seen, v);
                    }
                }
            }
        }
    }
}

/// `cases` arms no input can reach, each with its reason (the anti-vacuity check skips them).
const UNREACHABLE_ARMS: &[(&str, usize, usize, &str)] = &[
    (
        "model.tau1_Pa",
        0,
        1,
        "the fundamental is in every harmonic set (the smallest choice is 1)",
    ),
    ("calibration.tau1_Pa", 0, 1, "as model.tau1_Pa"),
];

#[test]
fn every_record_reproduces_the_engine_everywhere() {
    let registry = Registry::build();
    let points = guard_points();
    // The defaults, then every differential case under every augmentation (no stale count:
    // the data files are regenerated).
    let cases: usize = differential_files()
        .into_iter()
        .map(|f| load_cases(f).len())
        .sum();
    assert!(cases > 0);
    assert_eq!(
        points.len(),
        1 + cases * augmentations().len(),
        "{} points",
        points.len()
    );
    // The points are independent: evaluate them on every core (about 1 s instead of 30 in a
    // debug build).
    let threads = std::thread::available_parallelism().map_or(4, usize::from);
    let chunk = points.len().div_ceil(threads);
    let (mut failures, mut cov) = (Vec::new(), Coverage::default());
    std::thread::scope(|s| {
        let handles: Vec<_> = points
            .chunks(chunk)
            .map(|c| s.spawn(|| guard(&registry, c)))
            .collect();
        for h in handles {
            let (f, c) = h.join().expect("a guard thread");
            failures.extend(f);
            cov.merge(c);
        }
    });
    assert!(failures.is_empty(), "drift guard: {}", report(&failures));

    // Anti-vacuity: every arm of every `cases` taken somewhere.
    let mut untaken = Vec::new();
    for eq in registry.equations() {
        let mut ids = BTreeMap::new();
        eq.formula.visit(&mut |e| {
            if let Expr::Cases { id, arms, .. } = e {
                ids.insert(*id, arms.len());
            }
        });
        let taken = cov.arms.get(&eq.target).cloned().unwrap_or_default();
        for (id, n) in ids {
            for arm in 0..=n {
                let exempt = UNREACHABLE_ARMS
                    .iter()
                    .any(|&(t, i, a, _)| t == eq.target && i == id && a == arm);
                if taken.contains(&(id, arm)) {
                    assert!(
                        !exempt,
                        "{}: cases {id} arm {arm} is listed unreachable but is taken",
                        eq.target
                    );
                } else if !exempt {
                    untaken.push(format!("{}: cases {id} arm {arm}", eq.target));
                }
            }
        }
    }
    assert!(untaken.is_empty(), "arms never taken: {}", report(&untaken));
    // Every value term of every record takes two values somewhere: a term the guard sees at
    // one value only is indistinguishable there from a literal of that value, so a record
    // could lose a real dependency and still pass.
    let mut single = Vec::new();
    for (eq, terms) in registry.equations().iter().zip(&cov.term_values) {
        for (term, seen) in terms {
            if seen.len() < 2 {
                single.push(format!("{} reads {term} only as {:?}", eq.target, seen[0]));
            }
        }
    }
    assert!(
        single.is_empty(),
        "value terms seen at one value only: {}",
        report(&single)
    );
    // Every numeric record varies across the points (it is exercised, not a constant).
    let constant: Vec<&String> = cov
        .values
        .iter()
        .filter(|(_, v)| v.len() < 2)
        .map(|(k, _)| k)
        .collect();
    assert!(constant.is_empty(), "records that never vary: {constant:?}");
    // E7 leaves half a pitch for every angle record somewhere.
    for eq in registry
        .equations()
        .iter()
        .filter(|e| e.target.ends_with("angle_rad"))
    {
        let n = cov
            .angles_off_half_pitch
            .get(&eq.target)
            .copied()
            .unwrap_or(0);
        assert!(n > 0, "{}: never off half a pitch", eq.target);
    }
    // The harmonic 7 to 11 records are nonzero somewhere (the augmentations reach them).
    for n in [7, 9, 11] {
        for t in [format!("model.tau{n}_Pa"), format!("calibration.tau{n}_Pa")] {
            let values = &cov.values[&t];
            assert!(
                values.iter().any(|&b| f64::from_bits(b) != 0.0),
                "{t} is always 0"
            );
        }
    }
}

#[test]
fn input_and_result_paths_are_disjoint() {
    // `Design` looks a term up among the results, then the inputs: no path may be both.
    let inputs = DesignInputs::default();
    let ins: BTreeSet<String> = input_rows(&inputs).into_iter().map(|r| r.path).collect();
    let outs: BTreeSet<String> = result_rows(&compute_all(&inputs))
        .into_iter()
        .map(|r| r.path)
        .collect();
    assert!(
        ins.is_disjoint(&outs),
        "{:?}",
        ins.intersection(&outs).collect::<Vec<_>>()
    );
}

#[test]
fn the_registry_is_consistent() {
    let r = Registry::build();
    for eq in r.equations() {
        assert_eq!(
            r.equation_for(&eq.target).map(|e| &e.target),
            Some(&eq.target)
        );
        assert_eq!(r.term_kind(&eq.target), Some(TermKind::Explained));
        for t in &eq.terms {
            assert!(
                r.used_by(t).contains(&eq.target),
                "{t} used by {}",
                eq.target
            );
            assert!(r.term_kind(t).is_some(), "{t}");
            assert!(Symbol::parse(r.symbol(t).expect("every term has a symbol")).is_ok());
        }
        // The plain rendering names every term by its symbol, never by its path.
        let text = render::plain(&r, &eq.symbol, &eq.formula);
        assert!(!text.contains('['), "{}: {text}", eq.target);
    }
    // A stacked fraction as a factor is parenthesized in the plain line, so it reads as the
    // tree does.
    let f_end = r.equation_for("model.f_end").unwrap();
    assert_eq!(
        render::plain(&r, &f_end.symbol, &f_end.formula),
        "f_{end} = 1 − c_{end} · (τ_p/L)"
    );
    // No custom evals in the tracer: every displayed formula is the evaluated one.
    let custom: Vec<&str> = r
        .equations()
        .iter()
        .filter(|e| matches!(e.eval, Eval::Custom(_)))
        .map(|e| e.target.as_str())
        .collect();
    assert!(custom.is_empty(), "{custom:?}");
    assert!(r.is_leaf_input("metal.variation") && !r.is_leaf_input("metal.torque_hot_low_Nm"));
    assert_eq!(
        r.term_kind("metal.variation"),
        Some(TermKind::Input { assumption: true })
    );
    assert_eq!(r.family_symbol("model.b_i#").as_deref(), Some("B_{i,n}"));
    // The term list shows each term in the unit the formula reads it in.
    let inputs = DesignInputs::default();
    let results = compute_all(&inputs);
    let src = Design {
        inputs: &inputs,
        results: &results,
    };
    let rows = r.term_rows(r.equation_for("model.k3").unwrap(), &src);
    let rg = rows
        .iter()
        .find(|x| x.path == "model.gap_radius_mm")
        .unwrap();
    assert_eq!(rg.unit, "m");
    assert_eq!(
        rg.value,
        Some(Value::Num(results.model.gap_radius_mm * 1e-3))
    );
    assert_eq!(rg.symbol, "R_g");
    let n = rows.iter().find(|x| x.path == "coupling.npole").unwrap();
    assert_eq!(
        (n.unit.as_str(), n.kind),
        ("-", TermKind::Input { assumption: false })
    );
    let mut eleven = DesignInputs::default();
    eleven.set("coupling.max_harmonic", Value::Int(11)).unwrap();
    for (inputs, want) in [(DesignInputs::default(), 3), (eleven, 6)] {
        let results = compute_all(&inputs);
        let members = r.family_members(
            "model.b_i#",
            &Design {
                inputs: &inputs,
                results: &results,
            },
        );
        assert_eq!(members.len(), want);
        assert_eq!(members[0], "model.b_i1");
    }
}

/// The physics reviewer's sheet for a batch: every record's path, cell, symbol, rendered
/// formula and corrections, as a Markdown table on stdout. Run with
/// `cargo test --test explain review_sheet -- --ignored --nocapture`.
#[test]
#[ignore = "a review tool, not a check"]
fn review_sheet() {
    let r = Registry::build();
    println!("| Result | Cell | Formula | Corrections |\n|---|---|---|---|");
    for eq in r.equations() {
        let formula = render::plain(&r, &eq.symbol, &eq.formula).replace('|', "\\|");
        let cell = eq.cell.as_deref().unwrap_or("Rust-only");
        println!(
            "| `{}` | {cell} | {formula} | {:?} |",
            eq.target, eq.corrections
        );
    }
}

#[test]
fn every_explained_chain_drills_down_to_inputs() {
    // Every term below every record of an explained chain is an input or has a record:
    // nothing stops at a cell-only result.
    let r = Registry::build();
    let explained: Vec<_> = SCOPE
        .iter()
        .filter(|c| c.status == Status::Explained)
        .collect();
    assert!(explained.iter().any(|c| c.id == "torque"));
    for chain in explained {
        for path in chain.paths {
            for up in r.upstream(&path.replace("[]", "[0]")) {
                assert!(
                    matches!(
                        r.term_kind(&up),
                        Some(TermKind::Input { .. } | TermKind::Explained)
                    ),
                    "{}: {path} depends on {up}, which is neither an input nor explained",
                    chain.id
                );
            }
        }
    }
}

#[test]
fn explained_chains_have_every_record_and_scope_paths_exist() {
    let r = Registry::build();
    let results: BTreeSet<String> = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .map(|x| x.path)
        .collect();
    let mut missing = Vec::new();
    for chain in SCOPE {
        for path in chain.paths {
            // A table column (`clamps.table[].x`) exists when its first row does.
            let first_row = path.replace("[]", "[0]");
            assert!(
                results.contains(&first_row),
                "{}: {path} is not a result path",
                chain.id
            );
            if chain.status == Status::Explained && r.equation_for(&first_row).is_none() {
                missing.push(format!("{}: {path}", chain.id));
            }
        }
    }
    assert!(missing.is_empty(), "{missing:?}");
    let union: BTreeSet<&str> = SCOPE.iter().flat_map(|c| c.paths.iter().copied()).collect();
    assert_eq!(
        union.len(),
        159,
        "decision 31: the chains and the dashboard (report section 7)"
    );
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "^error" | head -3
```

Expected: ``error[E0432]: unresolved import `magcoupling::engine::explain::record` ``, then the same for `registry` and `scope` (more follow without `head`).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test differential 2>&1 | grep "test result"
```

Expected: `test result: ok. 19 passed`: the differential test is unchanged by the move of its loader.

- [ ] **Step 3: Write the registry, the records and the rest of the module**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

with:

```markdown
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
The gate runs `gen_differential.py --check`, so stale data fails it. The
snapshot copy must equal `reference/magcoupling-py/tests/reference_values.json`
(a test checks).

## Porting a module
```

with:

```markdown
The gate runs `gen_differential.py --check`, so stale data fails it. The
snapshot copy must equal `reference/magcoupling-py/tests/reference_values.json`
(a test checks).

## Equation explorer (Addendum A2 to A4)

`src/engine/explain/` explains results. Each record states one result from its terms in a
small markup that is both what the M4 equation panel typesets and what the drift guard
evaluates, so the equation shown is the one that produced the number. Its terms are input
and result paths (hover keys, colours, drill-down and slider links); unit, label and
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
|---|---|---|
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | pending |
| slip heating | `records/slip_heating.rs` | pending |
| temperature | `records/temperature.rs` | pending |
| clamps | `records/clamps.rs` | pending |
| dashboard | `records/dashboard.rs` | pending |

## Porting a module
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
//! | `{model.gap_radius_mm\|m}` | the value converted to the unit after `\|` (the registry resolves the factor; unknown pairs are refused) | the same symbol; the term list shows the value in that unit |
//! | `a * b` | a × b | a thin space (juxtaposition); `×` between two numbers |
//! | `a · b` | a × b | a centred dot |
//! | `a / b` | a ÷ b | inline `a/b` |
//! | `frac(a, b)` | a ÷ b | a stacked fraction |
//! | `a ^ b` | a to the power b | b as a superscript (outer parentheses of b dropped) |
//! | `sqrt(a)` | √a | a radical |
```

with:

```rust
//! | `{model.gap_radius_mm\|m}` | the value converted to the unit after `\|` (the registry resolves the factor; unknown pairs are refused) | the same symbol; the term list shows the value in that unit |
//! | `a * b` | a × b | a thin space (juxtaposition); `×` between two numbers |
//! | `a · b` | a × b | a centred dot |
//! | `a / b` | a ÷ b | inline `a/b` (never a factor: see below) |
//! | `frac(a, b)` | a ÷ b | a stacked fraction |
//! | `a ^ b` | a to the power b | b as a superscript (outer parentheses of b dropped) |
//! | `sqrt(a)` | √a | a radical |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
//! | `peak(n in H: A)` | the electrical angle φ in [0, π/2] at which Σ_{n∈H} A_n sin(nφ) is largest, A_n the expression: π/2 unless another maximum beats it by more than 1e-12 relative, exactly as E7 decides (`model::peak_off_half_pitch`); π/2 for an invalid set | `arg max` over φ of Σ_{n∈H} A sin(nφ) |
//! | `table("magnets", k, "br_T")` | a field of a static engine table, by key; `none` when the key is not in it ([`super::tables`] lists the tables and fields) | the field's symbol with the key: `B_{r,lib}(part_i)` |
//! | `... where [S_n] = e` | `[S_n]` stands for e (evaluated when first used) | the formula, then one line per binding: `S_n = e` |
//!
//! # Symbol markup
//!
```

with:

```rust
//! | `peak(n in H: A)` | the electrical angle φ in [0, π/2] at which Σ_{n∈H} A_n sin(nφ) is largest, A_n the expression: π/2 unless another maximum beats it by more than 1e-12 relative, exactly as E7 decides (`model::peak_off_half_pitch`); π/2 for an invalid set | `arg max` over φ of Σ_{n∈H} A sin(nφ) |
//! | `table("magnets", k, "br_T")` | a field of a static engine table, by key; `none` when the key is not in it ([`super::tables`] lists the tables and fields) | the field's symbol with the key: `B_{r,lib}(part_i)` |
//! | `... where [S_n] = e` | `[S_n]` stands for e (evaluated when first used) | the formula, then one line per binding: `S_n = e` |
//!
//! The typesetter draws `a / b` inline and only the parentheses the markup writes, so the
//! registry refuses markup that would read two ways: an inline `a / b` as an operand of `*`,
//! `·` or `/` (`a/b c` reads as a/(b c): write `frac(a, b)` or `(a / b)`); a `sum` or `peak`
//! as an operand of `*`, `·`, `/` or `^`, or left of `+` or `-` (write `(sum(...))`); and a
//! power of a term, local or `exp` whose symbol already has a superscript (write
//! `({calibration.gap_radius_mm})^2`, never `R_g^{cal}^2`). It also refuses a formula that
//! reads one path in two units: the term list shows each term once, in the unit it is read in.
//!
//! # Symbol markup
//!
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs`, replace:

```rust
//! Addendum A2 equation explorer, engine side: the explanation layer.
//!
//! - [`markup`]: the formula and symbol grammar, parser and tree;
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron).

pub mod eval;
pub mod markup;
pub mod tables;

pub use eval::{EvalError, TermSource, Trace};
```

with:

```rust
//! Addendum A2 equation explorer, engine side: the explanation layer.
//!
//! Each explained result has an equation record ([`record::Record`]): its target path, a
//! display symbol, a formula in a small markup ([`markup`]) whose terms ARE input or result
//! paths, and the approved corrections it embodies. Unit, label and workbook cell come from
//! the result's metadata. The formula is both what the panel typesets and what the drift
//! guard evaluates ([`eval`]), so the equation shown is provably the one that produced the
//! number: `tests/explain.rs` evaluates every record over the engine's own term values at
//! the defaults and at every differential and augmented input set, corrections on, and
//! requires 1e-9 relative agreement (the parity rule).
//!
//! The engine code stays as ported. Where a formula needs a term the engine computed but did
//! not expose (the harmonics 7 to 11 parts, the torque-angle amplitudes, the E7 angles), the
//! engine exposes it as a Rust-only result: a pure read, no formula change.
//!
//! - [`markup`]: the formula and symbol grammar, parser and tree;
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron);
//! - [`record`]: the authoring form, [`records`]: the records, one file per batch;
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term rows;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
//! - [`scope`]: the v1 scope (decision 31) and which chains are written;
//! - [`render`]: a plain-text rendering (tooltips fallback, export, tests).

pub mod eval;
pub mod markup;
pub mod record;
pub mod records;
pub mod registry;
pub mod render;
pub mod scope;
pub mod symbols;
pub mod tables;

pub use eval::{EvalError, TermSource, Trace};
pub use registry::{Design, Equation, Registry, TermKind, TermRow};
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/record.rs` with exactly this content:

```rust
//! The authoring form of an equation record: what a records file declares, as `const` data.
//!
//! A record says what one result IS, one step from its terms: `σ = Σ_{n∈H} σ_n`, never the
//! chain behind the terms. Each term is an input or result path, so its value comes from the
//! engine, the hover key is the path, and a click drills into that term's own record. The
//! label, unit and workbook cell are not repeated here: the registry reads them from the
//! result's metadata by path.

use std::fmt;

use super::eval::{EvalError, TermSource};
use crate::engine::deviations::DeviationId;
use crate::engine::meta::Value;

/// A Rust evaluation over a record's terms (it sees only the formula's terms).
pub type CustomEval = fn(&dyn TermSource) -> Result<Value, EvalError>;

/// How the drift guard computes a record's value.
#[derive(Clone, Copy)]
pub enum Eval {
    /// The formula markup itself: the guard proves the displayed formula. The default.
    Markup,
    /// Escape hatch for what the markup cannot state (for example a verdict that formats a
    /// number into its text). The display markup is still parsed for its terms, and the
    /// function sees only those; the registry counts these records and the physics review
    /// checks each display against its function by hand.
    Custom(CustomEval),
}

impl fmt::Debug for Eval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Eval::Markup => f.write_str("Markup"),
            Eval::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

/// One explained result.
#[derive(Clone, Copy, Debug)]
pub struct Record {
    /// The result path it explains (in a [`Family`], `#` for the index).
    pub target: &'static str,
    /// Its display symbol, in symbol markup (in a family, `#` for the index, braced: `σ_{#}`).
    pub symbol: &'static str,
    /// Its formula, in formula markup (see [`super::markup`]).
    pub formula: &'static str,
    pub eval: Eval,
    /// The approved corrections this formula embodies (E-ids): the GUI's "corrected vs
    /// workbook" marker names them, and the traceability of a correction starts here.
    pub corrections: &'static [DeviationId],
}

/// A record whose value is its formula (the usual case).
pub const fn record(target: &'static str, symbol: &'static str, formula: &'static str) -> Record {
    Record {
        target,
        symbol,
        formula,
        eval: Eval::Markup,
        corrections: &[],
    }
}

impl Record {
    /// Names the approved corrections the formula embodies.
    pub const fn corrected(self, corrections: &'static [DeviationId]) -> Self {
        Self {
            corrections,
            ..self
        }
    }

    /// Replaces the markup evaluation by a Rust function (see [`Eval::Custom`]).
    pub const fn custom(self, eval: CustomEval) -> Self {
        Self {
            eval: Eval::Custom(eval),
            ..self
        }
    }
}

/// One record per index (the harmonics): `target`, `symbol` and the formula's term paths
/// carry `#`, and the formula's `n` is the index.
#[derive(Clone, Copy, Debug)]
pub struct Family {
    pub record: Record,
    pub indices: &'static [u32],
}

/// A family of records, one per index.
pub const fn family(record: Record, indices: &'static [u32]) -> Family {
    Family { record, indices }
}
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` with exactly this content:

```rust
//! The equation records, one file per batch (plan A-3). A batch adds its file here and
//! flips its chain to `Explained` in [`super::scope`].

pub mod torque;

use super::record::{Family, Record};

/// Every batch's plain records.
pub const RECORDS: &[&[Record]] = &[torque::RECORDS];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES];
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/torque.rs` with exactly this content:

```rust
//! Torque chain (plan A-3 batch 1, the tracer): the harmonics (τ1 to τ11 as the shear
//! stresses σ_n), the total shear stress, end effect, calibration factor, pull-out, and the
//! hot and cold torques, plus the geometry, remanence and magnet-resolution records their
//! terms need, so every term drills down to an input.
//!
//! Each formula is transcribed from the engine function named beside it, with every approved
//! correction on (the explorer shows what users see). `tests/explain.rs` evaluates each one
//! over the engine's own term values at the defaults and every differential and augmented
//! input set: a transcription error fails there, never silently.
//!
//! Symbols: σ shear stress, T torque, ϑ temperature, φ electrical angle, τ_p pole pitch
//! (plan A-3; `symbols.rs` lists the conventions).

use crate::engine::deviations::DeviationId::{E3, E7, E8};
use crate::engine::explain::record::{Family, Record, family, record};
use crate::engine::model::ODD_HARMONICS;

/// The harmonic families: one record per odd harmonic up to 11, summed or not.
#[rustfmt::skip]
pub const FAMILIES: &[Family] = &[
    // model::shear_stress: k = n (N/2) / R_g.
    family(record("model.k#", "k_{#}",
        "frac(n * {coupling.npole}, 2 * {model.gap_radius_mm|m})"), &ODD_HARMONICS),
    // model::harmonic_amplitude: Br 4/(nπ) sin(n α π/2), square-wave magnetization.
    family(record("model.b_i#", "B_{i,#}",
        "{model.br_inner_T_op} · frac(4, n * π) · sin(frac(n * π * {model.fill_inner}, 2))"), &ODD_HARMONICS),
    family(record("model.b_o#", "B_{o,#}",
        "{model.br_outer_T_op} · frac(4, n * π) · sin(frac(n * π * {model.fill_outer}, 2))"), &ODD_HARMONICS),
    // model::geometry_factor, steel-backed circuit.
    family(record("model.s#_iron", "S_{#}^{iron}",
        "frac(sinh({model.k#} * {model.inner_thickness_mm|m}) * sinh({model.k#} * {model.outer_thickness_mm|m}), \
         sinh({model.k#} * ({model.inner_thickness_mm|m} + {model.outer_thickness_mm|m} + {model.face_gap_mm|m})))"),
        &ODD_HARMONICS),
    // model::geometry_factor, free-space rings.
    family(record("model.s#_free", "S_{#}^{free}",
        "frac((1 - exp(-{model.k#} * {model.inner_thickness_mm|m})) * (1 - exp(-{model.k#} * {model.outer_thickness_mm|m})) \
         * exp(-{model.k#} * {model.face_gap_mm|m}), 2)"),
        &ODD_HARMONICS),
    // model::Harmonic::amplitude, in the circuit in effect (A5: a non-ferromagnetic back iron is free space).
    family(record("model.amp#_Pa", "A_{#}",
        "frac({model.b_i#} * {model.b_o#}, 2 * {coupling.mu0}) · \
         cases({materials.circuit_backiron} = 1 => {model.s#_iron}; else => {model.s#_free})"),
        &ODD_HARMONICS),
    // model::compute via at_pull_out and harmonic_slot: summed harmonics at the E7 angle; 0 when left out.
    family(record("model.tau#_Pa", "σ_{#}",
        "cases(n <= {coupling.max_harmonic} => {model.amp#_Pa} · sin(n * {model.pullout_angle_rad}); else => 0)")
        .corrected(&[E7]), &ODD_HARMONICS),
    // calibration::compute amp_n: the prototype, identical free-space rings.
    family(record("calibration.amp#_Pa", "A_{#}^{cal}",
        "frac((4 * {calibration.br_test_T} / (n * π))^2 * sin(frac(n * π * {calibration.fill_inner}, 2)) \
         * sin(frac(n * π * {calibration.fill_outer}, 2)), 2 * {calibration.mu0}) \
         · frac((1 - exp(-[k_{#}^{cal}] * {calibration.magnet_thickness_mm|m}))^2 * exp(-[k_{#}^{cal}] * {calibration.flat_gap_mm|m}), 2) \
         where [k_{#}^{cal}] = frac(n * {calibration.poles_per_ring}, 2 * {calibration.gap_radius_mm|m})"),
        &ODD_HARMONICS),
    family(record("calibration.tau#_Pa", "σ_{#}^{cal}",
        "cases(n <= {coupling.max_harmonic} => {calibration.amp#_Pa} · sin(n * {calibration.pullout_angle_rad}); else => 0)")
        .corrected(&[E7]), &ODD_HARMONICS),
];

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- Calculator: shear stress and pull-out (model::compute) ---
    record("model.tau_Pa", "σ", "sum(n in H: {model.tau#_Pa})"),
    record("model.pullout_angle_rad", "φ_{pull}", "peak(n in H: {model.amp#_Pa})").corrected(&[E7]),
    record("model.iron_circuit_angle_rad", "φ_{iron}",
        "peak(n in H: {model.b_i#} * {model.b_o#} * {model.s#_iron})").corrected(&[E7]),
    record("model.free_circuit_angle_rad", "φ_{free}",
        "peak(n in H: {model.b_i#} * {model.b_o#} * {model.s#_free})").corrected(&[E7]),
    record("model.area_lever_m3", "A_L", "2 * π * {model.gap_radius_mm|m}^2 * {model.active_length_mm|m}"),
    record("model.torque_2d_Nm", "T_{2D}", "{model.tau_Pa} * {model.area_lever_m3}"),
    record("model.f_end", "f_{end}", "1 - {coupling.c_end} · frac({model.pole_pitch_mm}, {model.active_length_mm})"),
    record("model.pullout_Nm", "T_{pull}", "{model.torque_2d_Nm} * {model.f_end} * {model.f_cal}"),
    record("model.pullout_20C_Nm", "T_{pull,20}",
        "{model.pullout_Nm} · frac({model.inner_br_T} * {model.outer_br_T}, {model.br_inner_T_op} * {model.br_outer_T_op})"),
    record("model.pullout_iron_Nm", "T_{iron}",
        "frac(sum(n in H: {model.b_i#} * {model.b_o#} * {model.s#_iron} * sin(n * {model.iron_circuit_angle_rad})), 2 * {coupling.mu0}) \
         * {model.area_lever_m3} * {model.f_end} * {calibration.f_cal_original}").corrected(&[E7]),
    record("model.pullout_noiron_Nm", "T_{free}",
        "frac(sum(n in H: {model.b_i#} * {model.b_o#} * {model.s#_free} * sin(n * {model.free_circuit_angle_rad})), 2 * {coupling.mu0}) \
         * {model.area_lever_m3} * {model.f_end} * {model.f_cal}").corrected(&[E7]),
    // model::select_calibration_factor: the measured correction only for the prototype's own circuit.
    record("model.f_cal", "f_{cal}",
        r#"cases({materials.circuit_backiron} = 0 and {coupling.npole} = {calibration.poles_per_ring}
               and {coupling.magnets.part_inner} = "B842SH" and {coupling.magnets.part_outer} = "B842SH"
               => {calibration.f_cal_updated};
             else => {calibration.f_cal_original})"#),

    // --- Calculator: the geometry and remanence the torque terms read ---
    record("model.gap_radius_mm", "R_g", "{model.inner_face_radius_mm} + {model.face_gap_mm} / 2"),
    record("model.pole_pitch_mm", "τ_p", "frac(2 * π * {model.gap_radius_mm}, {coupling.npole})"),
    // model::pitch_share, clamped at 1 (C66, C67).
    record("model.fill_inner", "λ_i",
        "min(1, frac({model.inner_width_mm}, 2 * π * ({coupling.inner_back_apothem_mm} + {model.inner_thickness_mm} / 2) / {coupling.npole}))"),
    record("model.fill_outer", "λ_o",
        "min(1, frac({model.outer_width_mm}, 2 * π * ({model.outer_face_apothem_mm} + {model.outer_thickness_mm} / 2) / {coupling.npole}))"),
    // model::br_factor, each ring with its own coefficient (decision A2-7).
    record("model.br_inner_T_op", "B_{r,i,op}",
        "{model.inner_br_T} * (1 + {model.inner_alpha_br_per_C} * ({coupling.op_temp_C} - 20))"),
    record("model.br_outer_T_op", "B_{r,o,op}",
        "{model.outer_br_T} * (1 + {model.outer_alpha_br_per_C} * ({coupling.op_temp_C} - 20))"),
    record("model.active_length_mm", "L", "min({model.inner_length_mm}, {model.outer_length_mm})"),
    record("model.face_gap_mm", "g", "{model.outer_face_apothem_mm} - {model.inner_face_radius_mm}"),
    record("model.inner_face_radius_mm", "r_{face,i}", "{coupling.inner_back_apothem_mm} + {model.inner_thickness_mm}"),
    record("model.outer_face_apothem_mm", "A_o", "{model.inner_corner_radius_mm} + {model.corner_gap_mm}"),
    // model::corner_radius for flat blocks; the face radius for arcs.
    record("model.inner_corner_radius_mm", "r_{corner,i}",
        "cases({coupling.faceted} = 1 => sqrt({model.inner_face_radius_mm}^2 + ({model.inner_width_mm} / 2)^2);
               else => {model.inner_face_radius_mm})"),
    // E8: the corner gap from the inner corner radius C55 (the face radius for arcs).
    record("model.corner_gap_mm", "g_c",
        "{metal.face_gap_mm} - ({model.inner_corner_radius_mm} - {model.inner_face_radius_mm})").corrected(&[E8]),

    // --- Calculator: the magnets in effect (model::resolve_magnets) ---
    // The axial override wins; else the library part; else the manual value (a grade-mode
    // ring keeps the manual dimensions).
    record("model.inner_length_mm", "L_i",
        r#"cases({coupling.magnets.axial_length_mm} != none => {coupling.magnets.axial_length_mm};
               [L_{lib}] != none => [L_{lib}];
               else => {coupling.magnets.manual_inner_length_mm})
           where [L_{lib}] = table("magnets", {coupling.magnets.part_inner}, "length_mm")"#),
    record("model.outer_length_mm", "L_o",
        r#"cases({coupling.magnets.axial_length_mm} != none => {coupling.magnets.axial_length_mm};
               [L_{lib}] != none => [L_{lib}];
               else => {coupling.magnets.manual_outer_length_mm})
           where [L_{lib}] = table("magnets", {coupling.magnets.part_outer}, "length_mm")"#),
    record("model.inner_width_mm", "w_i",
        r#"cases([w_{lib}] != none => [w_{lib}]; else => {coupling.magnets.manual_inner_width_mm})
           where [w_{lib}] = table("magnets", {coupling.magnets.part_inner}, "width_mm")"#),
    record("model.outer_width_mm", "w_o",
        r#"cases([w_{lib}] != none => [w_{lib}]; else => {coupling.magnets.manual_outer_width_mm})
           where [w_{lib}] = table("magnets", {coupling.magnets.part_outer}, "width_mm")"#),
    record("model.inner_thickness_mm", "t_i",
        r#"cases([t_{lib}] != none => [t_{lib}]; else => {coupling.magnets.manual_inner_thickness_mm})
           where [t_{lib}] = table("magnets", {coupling.magnets.part_inner}, "thickness_mm")"#),
    record("model.outer_thickness_mm", "t_o",
        r#"cases([t_{lib}] != none => [t_{lib}]; else => {coupling.magnets.manual_outer_thickness_mm})
           where [t_{lib}] = table("magnets", {coupling.magnets.part_outer}, "thickness_mm")"#),
    // E3: a library part's Br is its grade's (N42SH 1.30 T); a grade-mode ring takes its grade's.
    record("model.inner_br_T", "B_{r,i}",
        r#"cases([B_{lib}] != none => [B_{lib}]; [B_{grade}] != none => [B_{grade}];
               else => {coupling.magnets.manual_inner_br_T})
           where [B_{lib}] = table("magnets", {coupling.magnets.part_inner}, "br_T"),
                 [B_{grade}] = table("grades", {coupling.magnets.grade_inner}, "br_T")"#).corrected(&[E3]),
    record("model.outer_br_T", "B_{r,o}",
        r#"cases([B_{lib}] != none => [B_{lib}]; [B_{grade}] != none => [B_{grade}];
               else => {coupling.magnets.manual_outer_br_T})
           where [B_{lib}] = table("magnets", {coupling.magnets.part_outer}, "br_T"),
                 [B_{grade}] = table("grades", {coupling.magnets.grade_outer}, "br_T")"#).corrected(&[E3]),
    // model::ResolvedMagnet::alpha_br: the grade's only in the grade mode (decision A2-7).
    record("model.inner_alpha_br_per_C", "α_i",
        r#"cases(table("magnets", {coupling.magnets.part_inner}, "br_T") = none and [α_{grade}] != none => [α_{grade}];
               else => {calibration.alpha_br_per_C})
           where [α_{grade}] = table("grades", {coupling.magnets.grade_inner}, "alpha_br_per_C")"#),
    record("model.outer_alpha_br_per_C", "α_o",
        r#"cases(table("magnets", {coupling.magnets.part_outer}, "br_T") = none and [α_{grade}] != none => [α_{grade}];
               else => {calibration.alpha_br_per_C})
           where [α_{grade}] = table("grades", {coupling.magnets.grade_outer}, "alpha_br_per_C")"#),
    // material_library::resolve: a non-ferromagnetic back iron selects the free-space circuit (A5).
    record("materials.circuit_backiron", "circuit",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => 0; else => {coupling.backiron})"#),

    // --- Calibration: the prototype model and the one-point correction (calibration::compute) ---
    record("calibration.poles_per_ring", "N^{cal}", "{calibration.total_magnets} / 2"),
    record("calibration.face_radius_mm", "r_{face}^{cal}", "{calibration.apothem_mm} + {calibration.magnet_thickness_mm}"),
    record("calibration.corner_radius_mm", "r_{corner}^{cal}",
        "sqrt(({calibration.face_radius_mm})^2 + ({calibration.magnet_width_mm} / 2)^2)"),
    record("calibration.corner_gap_mm", "g_c^{cal}",
        "cases({calibration.gap_definition} = 0 => {calibration.spacing_mm};
               else => {calibration.spacing_mm} - ({calibration.corner_radius_mm} - {calibration.face_radius_mm}))"),
    record("calibration.outer_face_apothem_mm", "A_o^{cal}", "{calibration.corner_radius_mm} + {calibration.corner_gap_mm}"),
    record("calibration.flat_gap_mm", "g^{cal}", "{calibration.outer_face_apothem_mm} - {calibration.face_radius_mm}"),
    record("calibration.gap_radius_mm", "R_g^{cal}", "{calibration.face_radius_mm} + {calibration.flat_gap_mm} / 2"),
    record("calibration.fill_inner", "λ_i^{cal}",
        "min(1, frac({calibration.magnet_width_mm}, 2 * π * ({calibration.apothem_mm} + {calibration.magnet_thickness_mm} / 2) / {calibration.poles_per_ring}))"),
    record("calibration.fill_outer", "λ_o^{cal}",
        "min(1, frac({calibration.magnet_width_mm}, 2 * π * ({calibration.outer_face_apothem_mm} + {calibration.magnet_thickness_mm} / 2) / {calibration.poles_per_ring}))"),
    record("calibration.br_test_T", "B_{r,test}",
        "{calibration.br_T} * (1 + {calibration.alpha_br_per_C} * ({calibration.test_temp_C} - 20))"),
    record("calibration.pole_pitch_mm", "τ_p^{cal}", "frac(2 * π * {calibration.gap_radius_mm}, {calibration.poles_per_ring})"),
    record("calibration.f_end", "f_{end}^{cal}",
        "1 - {calibration.c_end} · frac({calibration.pole_pitch_mm}, {calibration.magnet_length_mm})"),
    record("calibration.pullout_angle_rad", "φ^{cal}", "peak(n in H: {calibration.amp#_Pa})").corrected(&[E7]),
    record("calibration.torque_2d_Nm", "T_{2D}^{cal}",
        "(sum(n in H: {calibration.tau#_Pa})) * 2 * π * ({calibration.gap_radius_mm|m})^2 * {calibration.magnet_length_mm|m}"),
    record("calibration.original_model_Nm", "T_{orig}^{cal}",
        "{calibration.torque_2d_Nm} * {calibration.f_end} * {calibration.f_cal_original}"),
    record("calibration.model_torque_Nm", "T_{model}^{cal}", "{calibration.original_model_Nm}"),
    record("calibration.measured_over_model", "r^{cal}", "frac({calibration.measured_torque_Nm}, {calibration.model_torque_Nm})"),
    record("calibration.f_cal_updated", "f_{cal,1}", "{calibration.f_cal_original} * {calibration.measured_over_model}"),
    record("calibration.model_error", "ε^{cal}", "frac({calibration.model_torque_Nm}, {calibration.measured_torque_Nm}) - 1"),
    record("calibration.fea_interp_Nm", "T_{3D}",
        r#"cases({calibration.corner_gap_mm} >= 1 and {calibration.corner_gap_mm} <= 1.5
               => {calibration.fea_torque1_Nm} + ({calibration.fea_torque2_Nm} - {calibration.fea_torque1_Nm}) · frac({calibration.corner_gap_mm} - 1, 0.5);
             else => "outside range")"#),
    record("calibration.fea_interp_error", "ε_{3D}",
        r#"cases({calibration.fea_interp_Nm} != "outside range" => frac({calibration.fea_interp_Nm}, {calibration.measured_torque_Nm}) - 1;
             else => "n.a.")"#),

    // --- Metal design: hot and cold torque (metal_design::compute) ---
    record("metal.op_temp_C", "ϑ_{op,MD}", "{coupling.op_temp_C}"),
    record("metal.alpha_br_per_C", "α_{MD}", "{calibration.alpha_br_per_C}"),
    record("metal.torque_op_Nm", "T_{op}", "{model.pullout_Nm}"),
    record("metal.torque_20C_Nm", "T_{20}", "{model.pullout_20C_Nm}"),
    record("metal.torque_cold_Nm", "T_{cold}",
        "{metal.torque_20C_Nm} * (1 + {model.inner_alpha_br_per_C} * ({metal.min_temp_C} - 20)) \
         * (1 + {model.outer_alpha_br_per_C} * ({metal.min_temp_C} - 20))"),
    record("metal.torque_hot_low_Nm", "T_{hot,low}", "{metal.torque_op_Nm} * (1 - {metal.variation})"),
    record("metal.torque_cold_high_Nm", "T_{cold,high}", "{metal.torque_cold_Nm} * (1 + {metal.variation})"),
    record("metal.hot_min_check", "C_{hot}",
        r#"cases({metal.torque_hot_low_Nm} < {metal.required_min_Nm} => "Below hot minimum"; else => "Estimate covers hot min")"#),
    record("metal.required_20C_Nm", "T_{req,20}",
        "frac({metal.required_min_Nm}, (1 + {model.inner_alpha_br_per_C} * ({metal.op_temp_C} - 20)) \
         * (1 + {model.outer_alpha_br_per_C} * ({metal.op_temp_C} - 20)) * (1 - {metal.variation}))"),
    record("metal.hot_margin", "m_{hot}", "frac({metal.torque_op_Nm}, {metal.required_min_Nm}) - 1"),
    record("metal.required_20C_zero_scatter_Nm", "T_{req,20,v=0}",
        "frac({metal.required_min_Nm}, (1 + {model.inner_alpha_br_per_C} * ({metal.op_temp_C} - 20)) \
         * (1 + {model.outer_alpha_br_per_C} * ({metal.op_temp_C} - 20)))"),
    record("metal.torque_cold_zero_var_Nm", "T_{cold,v=0}", "{metal.torque_cold_Nm}"),
    record("metal.noiron_baseline_hot_Nm", "T_{proto,hot}",
        "{calibration.measured_torque_Nm} * (frac(1 + {metal.alpha_br_per_C} * ({metal.op_temp_C} - 20), \
         1 + {metal.alpha_br_per_C} * ({calibration.test_temp_C} - 20)))^2"),
    record("metal.cold_for_hot_min_Nm", "T_{cold,req}",
        "{metal.required_min_Nm} · frac(1 + {model.inner_alpha_br_per_C} * ({metal.min_temp_C} - 20), 1 + {model.inner_alpha_br_per_C} * ({metal.op_temp_C} - 20)) \
         · frac(1 + {model.outer_alpha_br_per_C} * ({metal.min_temp_C} - 20), 1 + {model.outer_alpha_br_per_C} * ({metal.op_temp_C} - 20))"),
    record("metal.cold_for_hot_min_input_Nm", "T_{cold,req,in}",
        "frac({metal.cold_for_hot_min_Nm}, {coupling.gear_ratio} * {coupling.gear_efficiency})"),
    record("metal.cold_high_Nm", "T_{cold,high,MD}", "{metal.torque_cold_high_Nm}"),
    record("metal.cold_high_input_Nm", "T_{cold,high,in}",
        "frac({metal.cold_high_Nm}, {coupling.gear_ratio} * {coupling.gear_efficiency})"),
];
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs` with exactly this content:

```rust
//! The equation registry: every record parsed and checked once, indexed by target, with the
//! dependency graph the explorer and the traceability test read.
//!
//! Built explicitly ([`Registry::build`]) and owned by the caller (the GUI builds it once at
//! start-up), so the engine keeps no global state. Everything a frame needs is a map lookup
//! plus `ResultSet::get`/`InputSet::get` per term.

use std::collections::{BTreeMap, BTreeSet};

use super::eval::{self, EvalError, TermSource, Trace};
use super::markup::{self, BinOp, Expr, Formula, Func, IndexSet, Symbol};
use super::record::{Eval, Family, Record};
use super::records;
use super::symbols::SYMBOLS;
use super::tables;
use crate::engine::api::{DesignInputs, DesignResults, compute_all};
use crate::engine::deviations::DeviationId;
use crate::engine::meta::{
    InputMeta, InputSet, ResultMeta, ResultSet, Value, input_rows, result_rows,
};
use crate::engine::model::{ODD_HARMONICS, harmonic_count};

/// Unit conversions a term reference may ask for (`{path|unit}`): from, to, factor.
pub const CONVERSIONS: &[(&str, &str, f64)] = &[
    ("mm", "m", 1e-3),
    ("kA/m", "A/m", 1e3),
    ("MPa", "Pa", 1e6),
    ("GPa", "Pa", 1e9),
    ("g", "kg", 1e-3),
];

/// The factor that converts a value in `from` to `to`.
pub fn conversion(from: &str, to: &str) -> Option<f64> {
    CONVERSIONS
        .iter()
        .find(|&&(f, t, _)| f == from && t == to)
        .map(|&(_, _, k)| k)
}

/// A design's values as term values: a result path, else an input path (the two sets of
/// paths are disjoint: `tests/explain.rs`).
pub struct Design<'a> {
    pub inputs: &'a DesignInputs,
    pub results: &'a DesignResults,
}

impl TermSource for Design<'_> {
    fn value(&self, path: &str) -> Option<Value> {
        self.results.get(path).or_else(|| self.inputs.get(path))
    }
}

/// A source restricted to one record's terms (what a custom eval sees).
struct Restricted<'a> {
    inner: &'a dyn TermSource,
    allowed: &'a [String],
}

impl TermSource for Restricted<'_> {
    fn value(&self, path: &str) -> Option<Value> {
        if self.allowed.iter().any(|a| a == path) {
            self.inner.value(path)
        } else {
            None
        }
    }
}

/// One explained result, parsed and checked.
#[derive(Clone, Debug)]
pub struct Equation {
    /// The result path it explains.
    pub target: String,
    /// Its display symbol (symbol markup, index substituted).
    pub symbol: String,
    /// The formula markup as authored (a family's template; see `index`).
    pub source: &'static str,
    /// The family member's index, for a family record.
    pub index: Option<u32>,
    /// The parsed formula: the typesetter's input.
    pub formula: Formula,
    pub eval: Eval,
    /// Every term the formula can read, in first-appearance order: the static dependencies
    /// (every `cases` branch, every harmonic a Σ may include, a Σ's set selector, a table
    /// key). Each is an input or result path.
    pub terms: Vec<String>,
    /// From the result's metadata.
    pub label: &'static str,
    pub unit: &'static str,
    /// Its workbook cell (`None` for a Rust-only result).
    pub cell: Option<String>,
    pub corrections: &'static [DeviationId],
}

/// One row of the equation panel's term list.
#[derive(Clone, Debug, PartialEq)]
pub struct TermRow {
    pub path: String,
    /// Symbol markup.
    pub symbol: String,
    /// The value in `unit` (converted when the formula asks for another unit, `{path|m}`).
    pub value: Option<Value>,
    pub unit: String,
    pub kind: TermKind,
}

/// What a path is to the explorer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermKind {
    /// A design input or an assumption: a leaf; its slider is highlighted.
    Input { assumption: bool },
    /// A result with an equation record: a click drills into it.
    Explained,
    /// A result without a record: shows its label, value and workbook cell only.
    CellOnly,
}

/// The registry.
pub struct Registry {
    equations: Vec<Equation>,
    by_target: BTreeMap<String, usize>,
    used_by: BTreeMap<String, Vec<String>>,
    symbols: BTreeMap<String, String>,
    family_symbols: BTreeMap<String, String>,
    inputs: BTreeMap<String, &'static InputMeta>,
    results: BTreeMap<String, &'static ResultMeta>,
}

impl Registry {
    /// Builds the registry from every records file. Panics listing every problem;
    /// `tests/explain.rs` keeps the records clean, so a shipped build never does.
    pub fn build() -> Registry {
        match Self::try_build() {
            Ok(r) => r,
            Err(errors) => panic!("equation records:\n{}", errors.join("\n")),
        }
    }

    /// [`Registry::build`], returning every problem instead of panicking.
    pub fn try_build() -> Result<Registry, Vec<String>> {
        let records: Vec<Record> = records::RECORDS
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();
        let families: Vec<Family> = records::FAMILIES
            .iter()
            .flat_map(|f| f.iter().copied())
            .collect();
        Self::from_parts(&records, &families, SYMBOLS)
    }

    /// Builds a registry from the given records and leaf symbols (tests build small ones).
    pub fn from_parts(
        plain: &[Record],
        families: &[Family],
        leaf_symbols: &[(&'static str, &'static str)],
    ) -> Result<Registry, Vec<String>> {
        let defaults = DesignInputs::default();
        let schema_results = compute_all(&defaults);
        let inputs: BTreeMap<String, &'static InputMeta> = input_rows(&defaults)
            .into_iter()
            .map(|r| (r.path, r.meta))
            .collect();
        let result_rows = result_rows(&schema_results);
        let cells: BTreeMap<String, Option<String>> = result_rows
            .iter()
            .map(|r| (r.path.clone(), r.cell.clone()))
            .collect();
        let results: BTreeMap<String, &'static ResultMeta> =
            result_rows.into_iter().map(|r| (r.path, r.meta)).collect();
        let mut errors = Vec::new();

        // Expand the families into one record per index.
        let mut authored: Vec<(Record, Option<u32>)> = plain.iter().map(|r| (*r, None)).collect();
        let mut family_symbols = BTreeMap::new();
        for f in families {
            if matches!(f.record.eval, Eval::Custom(_)) {
                errors.push(format!(
                    "{}: a family record cannot have a custom eval",
                    f.record.target
                ));
            }
            // Each member's symbol is parsed below, which catches an unbraced index (k_11).
            if !f.record.target.contains('#') || !f.record.symbol.contains('#') {
                errors.push(format!(
                    "{}: a family's target and symbol need '#'",
                    f.record.target
                ));
            }
            family_symbols.insert(f.record.target.to_owned(), f.record.symbol.to_owned());
            authored.extend(f.indices.iter().map(|&n| (f.record, Some(n))));
        }

        let unit_of = |path: &str| -> Option<&'static str> {
            results
                .get(path)
                .map(|m| m.unit)
                .or_else(|| inputs.get(path).map(|m| m.unit))
        };
        let exists = |path: &str| results.contains_key(path) || inputs.contains_key(path);

        let mut equations = Vec::new();
        for (rec, index) in authored {
            let sub = |s: &str| match index {
                Some(n) => s.replace('#', &n.to_string()),
                None => s.to_owned(),
            };
            let target = sub(rec.target);
            let symbol = sub(rec.symbol);
            let at = |m: String| format!("{target}: {m}");
            if let Err(m) = Symbol::parse(&symbol) {
                errors.push(at(m));
            }
            let Some(meta) = results.get(&target) else {
                errors.push(at("not a result path".into()));
                continue;
            };
            let mut formula = match markup::parse(rec.formula, index) {
                Ok(f) => f,
                Err(e) => {
                    errors.push(at(format!("markup {e}")));
                    continue;
                }
            };
            // Resolve unit conversions and check every term, family member and table read.
            let mut terms: Vec<String> = Vec::new();
            let add = |p: String, terms: &mut Vec<String>| {
                if !terms.contains(&p) {
                    terms.push(p);
                }
            };
            let mut problems = Vec::new();
            // The unit each path is read in: one per formula, so the term list (one row per
            // term, in that unit) satisfies the equation shown.
            let mut read_in: BTreeMap<String, Option<String>> = BTreeMap::new();
            let mut two_units: BTreeSet<String> = BTreeSet::new();
            formula.visit_mut(&mut |e| {
                let (r, family) = match e {
                    Expr::Term(r) => (r, false),
                    Expr::FamilyTerm(r) => (r, true),
                    Expr::Sum(set, _) | Expr::Peak(set, _) => {
                        add(set.selector().to_owned(), &mut terms);
                        return;
                    }
                    Expr::Table { table, field, .. } => {
                        if tables::field(table, field).is_none() {
                            problems.push(format!("table {table} has no readable field {field}"));
                        }
                        return;
                    }
                    _ => return,
                };
                {
                    let members: Vec<String> = if family {
                        ODD_HARMONICS
                            .iter()
                            .map(|n| r.path.replace('#', &n.to_string()))
                            .collect()
                    } else {
                        vec![r.path.clone()]
                    };
                    for m in &members {
                        if !exists(m) {
                            problems.push(format!("term {m} is not an input or result path"));
                        } else if m == &target {
                            problems.push("the record reads its own target".into());
                        }
                        match read_in.get(m) {
                            Some(u) if *u != r.unit => {
                                two_units.insert(m.clone());
                            }
                            Some(_) => {}
                            None => {
                                read_in.insert(m.clone(), r.unit.clone());
                            }
                        }
                        add(m.clone(), &mut terms);
                    }
                    if let Some(u) = &r.unit {
                        match unit_of(&members[0]).and_then(|from| conversion(from, u)) {
                            Some(k) => r.scale = k,
                            None => problems.push(format!(
                                "no conversion from {} ({:?}) to {u}",
                                members[0],
                                unit_of(&members[0])
                            )),
                        }
                    }
                }
            });
            problems.extend(
                two_units
                    .into_iter()
                    .map(|m| format!("{m} is read in two units")),
            );
            errors.extend(problems.into_iter().map(&at));
            equations.push(Equation {
                target: target.clone(),
                symbol,
                source: rec.formula,
                index,
                formula,
                eval: rec.eval,
                terms,
                label: meta.label,
                unit: meta.unit,
                cell: cells.get(&target).cloned().flatten(),
                corrections: rec.corrections,
            });
        }

        let mut by_target = BTreeMap::new();
        for (i, eq) in equations.iter().enumerate() {
            if by_target.insert(eq.target.clone(), i).is_some() {
                errors.push(format!("{}: two records", eq.target));
            }
        }
        let mut used_by: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for eq in &equations {
            for t in &eq.terms {
                used_by
                    .entry(t.clone())
                    .or_default()
                    .push(eq.target.clone());
            }
        }

        // Symbols: a record's own, else the leaf table; one symbol per path, every term has
        // one, no leaf entry is dead, and no two paths share a symbol.
        let mut symbols: BTreeMap<String, String> = equations
            .iter()
            .map(|e| (e.target.clone(), e.symbol.clone()))
            .collect();
        for &(path, symbol) in leaf_symbols {
            if by_target.contains_key(path) {
                errors.push(format!(
                    "{path}: has a record; its symbol is the record's (remove it from SYMBOLS)"
                ));
            } else if !used_by.contains_key(path) {
                errors.push(format!("{path}: in SYMBOLS but no record reads it"));
            } else if !exists(path) {
                errors.push(format!(
                    "{path}: in SYMBOLS but not an input or result path"
                ));
            }
            if let Err(m) = Symbol::parse(symbol) {
                errors.push(format!("{path}: {m}"));
            }
            if symbols.insert(path.to_owned(), symbol.to_owned()).is_some() {
                errors.push(format!("{path}: listed twice"));
            }
        }
        for term in used_by.keys() {
            if !symbols.contains_key(term) {
                errors.push(format!("{term}: a term with no symbol (add it to SYMBOLS)"));
            }
        }
        let mut owners: BTreeMap<&str, &str> = BTreeMap::new();
        for (path, symbol) in &symbols {
            if let Some(other) = owners.insert(symbol.as_str(), path.as_str()) {
                errors.push(format!(
                    "symbol {symbol} is used by both {other} and {path}"
                ));
            }
        }

        // What the panel draws must read as what is evaluated.
        for eq in &equations {
            let symbol_of = |e: &Expr| -> Option<String> {
                match e {
                    Expr::Term(r) => symbols.get(&r.path).cloned(),
                    Expr::FamilyTerm(r) => family_symbols.get(&r.path).map(|s| s.replace('#', "n")),
                    Expr::Local(k) => eq.formula.bindings.get(*k).map(|b| b.symbol.clone()),
                    _ => None,
                }
            };
            for m in typesetting_problems(&eq.formula, &symbol_of) {
                errors.push(format!("{}: {m}", eq.target));
            }
        }

        // Cycles would make drill-down loop; the graph must be a DAG.
        let graph: BTreeMap<&str, &[String]> = equations
            .iter()
            .map(|e| (e.target.as_str(), e.terms.as_slice()))
            .collect();
        for eq in &equations {
            if closure(&graph, &eq.target).1 {
                errors.push(format!(
                    "{}: its terms lead back to it (a cycle)",
                    eq.target
                ));
            }
        }

        if errors.is_empty() {
            Ok(Registry {
                equations,
                by_target,
                used_by,
                symbols,
                family_symbols,
                inputs,
                results,
            })
        } else {
            Err(errors)
        }
    }

    /// Every equation, in authoring order (plain records, then family members).
    pub fn equations(&self) -> &[Equation] {
        &self.equations
    }

    /// The equation of a result path, if it has one.
    pub fn equation_for(&self, path: &str) -> Option<&Equation> {
        self.by_target.get(path).map(|&i| &self.equations[i])
    }

    /// The results whose equation reads `path` (the panel's "used by" list), sorted by
    /// authoring order; empty when none does.
    pub fn used_by(&self, path: &str) -> &[String] {
        self.used_by.get(path).map_or(&[], Vec::as_slice)
    }

    /// Whether `path` is an input: a leaf of every drill-down.
    pub fn is_leaf_input(&self, path: &str) -> bool {
        self.inputs.contains_key(path)
    }

    /// What `path` is to the explorer; `None` for a path that is neither input nor result.
    pub fn term_kind(&self, path: &str) -> Option<TermKind> {
        if let Some(m) = self.inputs.get(path) {
            Some(TermKind::Input {
                assumption: m.assumption,
            })
        } else if self.by_target.contains_key(path) {
            Some(TermKind::Explained)
        } else if self.results.contains_key(path) {
            Some(TermKind::CellOnly)
        } else {
            None
        }
    }

    /// The display symbol (symbol markup) of a record's target or of a term.
    pub fn symbol(&self, path: &str) -> Option<&str> {
        self.symbols.get(path).map(String::as_str)
    }

    /// The generic symbol of a family term inside a Σ (`model.b_i#` gives `B_{i,n}`).
    pub fn family_symbol(&self, template: &str) -> Option<String> {
        self.family_symbols
            .get(template)
            .map(|s| s.replace('#', "n"))
    }

    /// The members of a family term inside a Σ or `peak` for a design: the paths of the
    /// harmonics summed (`model.b_i#` gives `model.b_i1`, `model.b_i3`, `model.b_i5` for the
    /// workbook's set), so the panel colours and lists exactly the terms in play.
    pub fn family_members(&self, template: &str, src: &dyn TermSource) -> Vec<String> {
        let count = match src.value(IndexSet::Harmonics.selector()) {
            Some(Value::Int(code)) => harmonic_count(code).unwrap_or(0),
            _ => 0,
        };
        ODD_HARMONICS[..count]
            .iter()
            .map(|n| template.replace('#', &n.to_string()))
            .collect()
    }

    /// The dependency graph: each explained result and its terms.
    pub fn graph(&self) -> impl Iterator<Item = (&str, &[String])> {
        self.equations
            .iter()
            .map(|e| (e.target.as_str(), e.terms.as_slice()))
    }

    /// Every path `path` depends on, transitively (terms of terms), itself excluded.
    pub fn upstream(&self, path: &str) -> BTreeSet<String> {
        let graph: BTreeMap<&str, &[String]> = self.graph().collect();
        closure(&graph, path).0
    }

    /// The value of every term of `eq`, in `eq.terms` order, in each term's own unit.
    pub fn term_values(&self, eq: &Equation, src: &dyn TermSource) -> Vec<(String, Option<Value>)> {
        eq.terms.iter().map(|t| (t.clone(), src.value(t))).collect()
    }

    /// The panel's term list for `eq`: symbol, value and unit of every term, in `eq.terms`
    /// order, each in the unit the formula reads it in (`{model.gap_radius_mm|m}` lists R_g
    /// in m), so the numbers shown satisfy the equation shown.
    pub fn term_rows(&self, eq: &Equation, src: &dyn TermSource) -> Vec<TermRow> {
        let mut wanted: BTreeMap<String, (String, f64)> = BTreeMap::new();
        eq.formula.visit(&mut |e| {
            if let Expr::Term(r) | Expr::FamilyTerm(r) = e
                && let Some(u) = &r.unit
            {
                for n in ODD_HARMONICS {
                    wanted
                        .entry(r.path.replace('#', &n.to_string()))
                        .or_insert((u.clone(), r.scale));
                }
            }
        });
        eq.terms
            .iter()
            .map(|t| {
                let own = self
                    .results
                    .get(t)
                    .map(|m| m.unit)
                    .or_else(|| self.inputs.get(t).map(|m| m.unit))
                    .unwrap_or("");
                let (unit, scale) = wanted.get(t).cloned().unwrap_or((own.to_owned(), 1.0));
                let value = src.value(t).map(|v| match v {
                    Value::Num(x) if scale != 1.0 => Value::Num(x * scale),
                    other => other,
                });
                TermRow {
                    path: t.clone(),
                    symbol: self.symbols.get(t).cloned().unwrap_or_default(),
                    value,
                    unit,
                    kind: self
                        .term_kind(t)
                        .expect("every term is an input or result (build)"),
                }
            })
            .collect()
    }

    /// Evaluates `eq` over `src` (the drift guard; the GUI never needs it per frame).
    pub fn evaluate(
        &self,
        eq: &Equation,
        src: &dyn TermSource,
        trace: Option<&mut Trace>,
    ) -> Result<Value, EvalError> {
        match eq.eval {
            Eval::Markup => eval::evaluate(&eq.formula, src, trace),
            Eval::Custom(f) => {
                let restricted = Restricted {
                    inner: src,
                    allowed: &eq.terms,
                };
                let v = f(&restricted)?;
                if let Some(t) = trace {
                    // A custom eval's sensitivity is unknown: its terms count as read, not as moving it.
                    t.condition_terms.extend(eq.terms.iter().cloned());
                }
                Ok(v)
            }
        }
    }
}

/// What the typesetter would draw ambiguously, so that the formula shown would not read as
/// the one evaluated (the markup table in [`super::markup`] draws `a / b` inline and only the
/// parentheses the markup writes): an inline `a / b` as an operand of a product or of another
/// inline quotient (`a/b c` reads as a/(b c)); a Σ or `peak` as an operand of a product,
/// quotient or power, or left of a sum (where it ends is unclear); a power of a symbol, local
/// or `exp` that already carries a superscript (`R_g^{cal}^2`). Parentheses resolve each.
fn typesetting_problems(
    formula: &Formula,
    symbol_of: &dyn Fn(&Expr) -> Option<String>,
) -> Vec<String> {
    let inline_div = |x: &Expr| matches!(x, Expr::Bin(BinOp::Div, ..));
    let big = |x: &Expr| matches!(x, Expr::Sum(..) | Expr::Peak(..));
    let mut out = Vec::new();
    formula.visit(&mut |e| {
        let Expr::Bin(op, a, b) = e else { return };
        match op {
            BinOp::Mul | BinOp::Dot | BinOp::Div => {
                if inline_div(a) || inline_div(b) {
                    out.push("an inline a / b as an operand of a product or quotient: write frac(a, b) or (a / b)".to_owned());
                }
                if big(a) || big(b) {
                    out.push("a Σ or peak as an operand of a product or quotient: wrap it in parentheses".to_owned());
                }
            }
            BinOp::Pow => {
                let base = match &**a {
                    Expr::Call(Func::Exp, _) => Some("exp".to_owned()),
                    other => symbol_of(other)
                        .filter(|s| Symbol::parse(s).is_ok_and(|s| s.sup.is_some())),
                };
                if let Some(s) = base {
                    out.push(format!("a power of {s}, which has a superscript: wrap it in parentheses"));
                }
                if big(a) {
                    out.push("a Σ or peak raised to a power: wrap it in parentheses".to_owned());
                }
            }
            BinOp::Add | BinOp::Sub => {
                if big(a) {
                    out.push("a Σ or peak left of + or −: wrap it in parentheses".to_owned());
                }
            }
        }
    });
    out
}

/// Every node reachable from `start` along `graph`'s edges, `start` excluded, and whether
/// `start` is reachable from itself (a cycle).
fn closure(graph: &BTreeMap<&str, &[String]>, start: &str) -> (BTreeSet<String>, bool) {
    let mut seen = BTreeSet::new();
    let mut stack: Vec<&str> = vec![start];
    let mut cycle = false;
    while let Some(p) = stack.pop() {
        for t in graph.get(p).copied().unwrap_or(&[]) {
            if t == start {
                cycle = true;
            }
            if seen.insert(t.clone()) {
                stack.push(t);
            }
        }
    }
    (seen, cycle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::explain::record::{family, record};

    #[test]
    fn the_shipped_records_build() {
        let r = Registry::build();
        assert!(!r.equations().is_empty());
    }

    #[test]
    fn a_bad_record_is_refused_with_every_reason() {
        let errors = Registry::from_parts(
            &[
                record("model.nope", "X", "1"),
                record("model.f_end", "f_{end}", "{model.nope} + 1"),
                record("model.tau_Pa", "σ", "{model.gap_radius_mm|K}"),
                record("model.torque_2d_Nm", "T_{2D}", "1 +"),
                record(
                    "model.pullout_Nm",
                    "T_{pull}",
                    "{model.gap_radius_mm|m} * {model.gap_radius_mm}",
                ),
                record(
                    "model.k3",
                    "k_3",
                    "{coupling.npole} / 2 * {model.gap_radius_mm}",
                ),
                record("model.f_cal", "f^{cal}", "2"),
                record("model.pole_pitch_mm", "τ_p", "{model.f_cal}^2"),
                record(
                    "model.area_lever_m3",
                    "A",
                    "sum(n in H: {model.tau#_Pa}) * 2",
                ),
            ],
            &[
                family(record("model.k#", "k_#", "n"), &[1, 11]),
                family(record("model.k", "k_{#}", "n"), &[1]),
            ],
            &[],
        )
        .err()
        .expect("refused");
        let all = errors.join("\n");
        for want in [
            "model.nope: not a result path",
            "term model.nope is not",
            "no conversion from model.gap_radius_mm",
            "model.torque_2d_Nm: markup",
            "a family's target and symbol need",
            "model.k11: symbol 'k_11': unexpected text",
            "model.pullout_Nm: model.gap_radius_mm is read in two units",
            "model.k3: an inline a / b as an operand of a product",
            "model.pole_pitch_mm: a power of f^{cal}, which has a superscript",
            "model.area_lever_m3: a Σ or peak as an operand of a product",
        ] {
            assert!(all.contains(want), "missing '{want}' in:\n{all}");
        }
    }

    #[test]
    fn a_cycle_and_a_missing_symbol_are_refused() {
        let errors = Registry::from_parts(
            &[
                record("model.f_end", "f_{end}", "{model.pullout_Nm}"),
                record("model.pullout_Nm", "T_{pull}", "{model.f_end}"),
            ],
            &[],
            &[],
        )
        .err()
        .expect("refused");
        assert!(errors.iter().any(|e| e.contains("a cycle")), "{errors:?}");
    }
}
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/render.rs` with exactly this content:

```rust
//! A plain-text rendering of a formula: the results-table tooltip fallback, CSV/JSON export
//! and test output. The M4 typesetter draws the same tree ([`super::markup::Expr`]) with
//! egui (stacked fractions, scripts, radicals, Σ, braces); this is its one-line shadow.

use super::markup::{BinOp, Cond, Expr, Formula, Func, RelOp, Symbol};
use super::registry::Registry;
use super::tables;

/// `symbol = formula [, where ...]` in plain text, symbols from the registry.
pub fn plain(registry: &Registry, target_symbol: &str, formula: &Formula) -> String {
    let mut r = Renderer { registry, formula };
    let mut out = format!("{} = {}", sym(target_symbol), r.expr(&formula.body));
    for (i, b) in formula.bindings.iter().enumerate() {
        let lead = if i == 0 { ", where" } else { "," };
        out.push_str(&format!("{lead} {} = {}", sym(&b.symbol), r.expr(&b.expr)));
    }
    out
}

fn sym(markup: &str) -> String {
    Symbol::parse(markup).map_or_else(|_| markup.to_owned(), |s| s.plain())
}

struct Renderer<'a> {
    registry: &'a Registry,
    formula: &'a Formula,
}

impl Renderer<'_> {
    fn term(&self, path: &str, family: bool) -> String {
        let symbol = if family {
            self.registry.family_symbol(path)
        } else {
            self.registry.symbol(path).map(str::to_owned)
        };
        symbol.map_or_else(|| format!("[{path}]"), |s| sym(&s))
    }

    /// An operand of a product, quotient or power base: a stacked fraction, drawn inline here
    /// as `a/b`, is wrapped so the line reads as the tree does (`(π/4) (D² − d²)`).
    fn factor(&mut self, e: &Expr) -> String {
        let s = self.expr(e);
        if matches!(e, Expr::Frac(..)) {
            format!("({s})")
        } else {
            s
        }
    }

    /// Wraps compound operands of an inline `/` so the text reads as the fraction does.
    fn grouped(&mut self, e: &Expr) -> String {
        let s = self.expr(e);
        let atomic = matches!(
            e,
            Expr::Num { .. }
                | Expr::Text(_)
                | Expr::NoneLit
                | Expr::Pi
                | Expr::Term(_)
                | Expr::FamilyTerm(_)
                | Expr::Index
                | Expr::Local(_)
                | Expr::Paren(_)
                | Expr::Call(..)
                | Expr::Table { .. }
        );
        if atomic { s } else { format!("({s})") }
    }

    fn expr(&mut self, e: &Expr) -> String {
        match e {
            Expr::Num { text, .. } => text.clone(),
            Expr::Text(t) => format!("\"{t}\""),
            Expr::NoneLit => "none".into(),
            Expr::Pi => "π".into(),
            Expr::Term(r) => self.term(&r.path, false),
            Expr::FamilyTerm(r) => self.term(&r.path, true),
            Expr::Index => "n".into(),
            Expr::Local(k) => sym(&self.formula.bindings[*k].symbol),
            Expr::Neg(a) => format!("−{}", self.expr(a)),
            Expr::Paren(a) => format!("({})", self.expr(a)),
            Expr::Bin(op, a, b) => {
                let (x, y) = match op {
                    BinOp::Add | BinOp::Sub => (self.expr(a), self.expr(b)),
                    BinOp::Pow => (self.factor(a), self.expr(b)),
                    BinOp::Mul | BinOp::Dot | BinOp::Div => (self.factor(a), self.factor(b)),
                };
                match op {
                    BinOp::Add => format!("{x} + {y}"),
                    BinOp::Sub => format!("{x} − {y}"),
                    BinOp::Mul => {
                        let numbers =
                            matches!(**a, Expr::Num { .. }) && matches!(**b, Expr::Num { .. });
                        // A family member's index 1 as a factor is not drawn (k_1 = N/(2 R_g)).
                        let unit_factor = matches!(**a, Expr::Num { value, .. } if value == 1.0);
                        if unit_factor && !numbers {
                            y
                        } else if numbers {
                            format!("{x} × {y}")
                        } else {
                            format!("{x} {y}")
                        }
                    }
                    BinOp::Dot => format!("{x} · {y}"),
                    BinOp::Div => format!("{x}/{y}"),
                    BinOp::Pow => {
                        let y = match &**b {
                            Expr::Paren(inner) => self.expr(inner),
                            _ => y,
                        };
                        if y.chars().count() == 1 {
                            format!("{x}^{y}")
                        } else {
                            format!("{x}^{{{y}}}")
                        }
                    }
                }
            }
            Expr::Frac(a, b) => format!("{}/{}", self.grouped(a), self.grouped(b)),
            Expr::Call(f, args) => {
                let a: Vec<String> = args.iter().map(|x| self.expr(x)).collect();
                match f {
                    Func::Sqrt => format!("√({})", a[0]),
                    Func::Exp => format!("e^{{{}}}", a[0]),
                    Func::Abs => format!("|{}|", a[0]),
                    Func::Ceil => format!("⌈{}⌉", a[0]),
                    Func::Floor => format!("⌊{}⌋", a[0]),
                    _ => format!("{}({})", f.name(), a.join(", ")),
                }
            }
            Expr::Sum(_, body) => format!("Σ_{{n∈H}} {}", self.expr(body)),
            Expr::Peak(_, body) => format!(
                "argmax_{{0≤φ≤π/2}} Σ_{{n∈H}} {} sin(nφ)",
                self.grouped(body)
            ),
            Expr::Table { table, key, field } => {
                let s = tables::field(table, field)
                    .map_or_else(|| format!("{table}.{field}"), |f| sym(f.symbol));
                format!("{s}({})", self.expr(key))
            }
            Expr::Cases {
                arms, otherwise, ..
            } => {
                let mut rows: Vec<String> = arms
                    .iter()
                    .map(|(c, v)| format!("{} if {}", self.expr(v), self.cond(c)))
                    .collect();
                rows.push(format!("{} otherwise", self.expr(otherwise)));
                format!("{{ {} }}", rows.join("; "))
            }
        }
    }

    fn cond(&mut self, c: &Cond) -> String {
        match c {
            Cond::And(a, b) => format!("{} and {}", self.cond(a), self.cond(b)),
            Cond::Or(a, b) => format!("{} or {}", self.cond(a), self.cond(b)),
            Cond::Rel(op, a, b) => {
                let op = match op {
                    RelOp::Lt => "<",
                    RelOp::Le => "≤",
                    RelOp::Gt => ">",
                    RelOp::Ge => "≥",
                    RelOp::Eq => "=",
                    RelOp::Ne => "≠",
                };
                format!("{} {op} {}", self.expr(a), self.expr(b))
            }
        }
    }
}
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs` with exactly this content:

```rust
//! The v1 scope of the equation explorer (Addendum A decision 31): the dashboard, the
//! geometry callouts and the five chains, 159 paths plus the callouts (report section 7,
//! the chain lists of the A2 scope count). Every other result shows its label, value and
//! workbook cell. A path may sit in two chains; it has one record.
//!
//! Each batch of plan A-3 writes one chain's records and flips its status to `Explained`;
//! `tests/explain.rs` then requires a record for every path of every explained chain. The
//! geometry callouts are listed when the A1 view is specified (M4).

/// Whether a chain's records are written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Explained,
    Pending,
}

/// One chain (or the dashboard) of the v1 scope.
#[derive(Clone, Copy, Debug)]
pub struct Chain {
    pub id: &'static str,
    pub status: Status,
    pub paths: &'static [&'static str],
}

/// The v1 scope, in the spec's "start here" order, then the dashboard paths outside the chains.
pub const SCOPE: &[Chain] = &[
    Chain {
        id: "torque",
        status: Status::Explained,
        paths: &[
            "model.tau1_Pa",
            "model.tau3_Pa",
            "model.tau5_Pa",
            "model.tau_Pa",
            "calibration.tau1_Pa",
            "calibration.tau3_Pa",
            "calibration.tau5_Pa",
            "model.k1",
            "model.b_i1",
            "model.b_o1",
            "model.s1_iron",
            "model.s1_free",
            "model.k3",
            "model.b_i3",
            "model.b_o3",
            "model.s3_iron",
            "model.s3_free",
            "model.k5",
            "model.b_i5",
            "model.b_o5",
            "model.s5_iron",
            "model.s5_free",
            "model.area_lever_m3",
            "model.torque_2d_Nm",
            "model.pullout_Nm",
            "model.pullout_20C_Nm",
            "model.pullout_iron_Nm",
            "model.pullout_noiron_Nm",
            "calibration.torque_2d_Nm",
            "calibration.original_model_Nm",
            "calibration.model_torque_Nm",
            "model.f_end",
            "calibration.f_end",
            "model.f_cal",
            "calibration.f_cal_updated",
            "calibration.measured_over_model",
            "calibration.model_error",
            "calibration.fea_interp_Nm",
            "calibration.fea_interp_error",
            "metal.torque_op_Nm",
            "metal.torque_20C_Nm",
            "metal.torque_cold_Nm",
            "metal.torque_hot_low_Nm",
            "metal.torque_cold_high_Nm",
            "metal.hot_min_check",
            "metal.required_20C_Nm",
            "metal.hot_margin",
            "metal.required_20C_zero_scatter_Nm",
            "metal.torque_cold_zero_var_Nm",
            "metal.noiron_baseline_hot_Nm",
            "metal.cold_for_hot_min_Nm",
            "metal.cold_for_hot_min_input_Nm",
            "metal.cold_high_Nm",
            "metal.cold_high_input_Nm",
        ],
    },
    Chain {
        id: "temperature",
        status: Status::Pending,
        paths: &[
            "calibration.br_test_T",
            "model.inner_br_T",
            "model.outer_br_T",
            "model.alpha_br_per_C",
            "model.br_inner_T_op",
            "model.br_outer_T_op",
            "temperature.demag.br20_T",
            "temperature.demag.alpha_br",
            "metal.op_temp_C",
            "metal.alpha_br_per_C",
            "model.pullout_Nm",
            "model.pullout_20C_Nm",
            "metal.torque_op_Nm",
            "metal.torque_20C_Nm",
            "metal.torque_cold_Nm",
            "temperature.demag.torque_at_limit_Nm",
            "temperature.demag.torque_at_service_Nm",
            "temperature.summary.torque_hot_day_Nm",
            "temperature.summary.torque_hot_day_note",
            "temperature.magnet_life.torque_hot_day_Nm",
            "temperature.magnet_life.torque_hot_day_check",
            "temperature.magnet_life.torque_peak_Nm",
            "temperature.adhesive_life.torque_peak_var_Nm",
            "temperature.summary.service_max_C",
            "temperature.summary.onset_aligned_C",
            "temperature.summary.onset_pullout_C",
            "temperature.summary.onset_skipping_C",
            "temperature.summary.magnet_limit_C",
            "temperature.summary.adhesive_limit_C",
            "temperature.summary.governing_limit_C",
            "temperature.summary.governing_note",
            "temperature.summary.margin_service_C",
            "temperature.summary.hot_day_start_C",
            "temperature.summary.margin_hot_day_C",
            "temperature.summary.cure_margin_C",
            "temperature.summary.verdict",
            "temperature.demag.tmax_lib_C",
            "temperature.demag.magnet_limit_C",
            "model.inner_tmax_C",
            "model.outer_tmax_C",
            "model.inner_temp_check",
            "model.outer_temp_check",
            "temperature.adhesive.design_limit_C",
            "temperature.mismatch.cold_limit_C",
            "temperature.duty.hot_day_start_C",
        ],
    },
    Chain {
        id: "demagnetization",
        status: Status::Pending,
        paths: &[
            "temperature.demag.h_ref_kA_m",
            "temperature.demag.t_ref_model_C",
            "temperature.demag.calibration_offset_C",
            "temperature.demag.onset_aligned_C",
            "temperature.demag.onset_pullout_C",
            "temperature.demag.onset_skipping_C",
            "temperature.demag.onset_single_ring_C",
            "temperature.demag.magnet_limit_C",
            "temperature.summary.onset_aligned_C",
            "temperature.summary.onset_pullout_C",
            "temperature.summary.onset_skipping_C",
            "temperature.demag.torque_at_limit_Nm",
            "temperature.demag.torque_at_service_Nm",
            "temperature.summary.margin_service_C",
            "temperature.summary.margin_hot_day_C",
            "temperature.summary.cure_margin_C",
            "temperature.magnet_life.margin_onset_C",
            "temperature.magnet_life.margin_limit_C",
        ],
    },
    Chain {
        id: "slip_heating",
        status: Status::Pending,
        paths: &[
            "temperature.slip_loss.steel_sigma_S_m",
            "temperature.slip_loss.steel_mu_r",
            "temperature.slip_loss.cap_sigma_S_m",
            "temperature.slip_loss.skin_depth_mm",
            "temperature.slip_loss.hub_W",
            "temperature.slip_loss.cup_W",
            "temperature.slip_loss.web_W",
            "temperature.slip_loss.sleeve_W",
            "temperature.slip_loss.liner_W",
            "temperature.slip_loss.cap_W",
            "temperature.slip_loss.magnets_W",
            "temperature.slip_loss.total_W",
            "temperature.slip_loss.drag_Nm",
            "temperature.slip_loss.used_W",
            "temperature.slip_loss.high_W",
            "metal.slip_freq_Hz",
            "metal.slip_loss_W",
            "metal.slip_energy_J",
            "temperature.thermal.steel_c",
            "temperature.thermal.heat_capacity_J_K",
            "temperature.thermal.time_constant_s",
            "temperature.thermal.t95_s",
            "temperature.thermal.rev_per_tau",
            "temperature.thermal.rev95",
            "temperature.thermal.time_to_limit_high",
            "temperature.thermal.time_to_limit_est",
            "temperature.thermal.rotations_to_limit_high",
            "temperature.thermal.critical_drag_Nm",
            "temperature.summary.time_to_limit_high",
            "temperature.summary.critical_drag_Nm",
        ],
    },
    Chain {
        id: "clamps",
        status: Status::Pending,
        paths: &[
            "clamps.screw_proof_MPa",
            "clamps.joint_preload_N",
            "clamps.tightening_Nm",
            "clamps.table[].preload_strength_N",
            "clamps.table[].preload_strip_N",
            "clamps.table[].preload_N",
            "clamps.table[].tightening_Nm",
            "clamps.clamp_factor",
            "clamps.required_Nm",
            "clamps.max_torque_Nm",
            "clamps.screws",
            "clamps.capacity_Nm",
            "clamps.sf_coupling",
            "clamps.joint_torque_Nm",
            "clamps.joint_sf",
            "clamps.table[].torque_per_screw_Nm",
            "clamps.table[].screws_needed",
            "clamps.table[].clamp_torque_Nm",
            "clamps.table[].sf_coupling",
        ],
    },
    Chain {
        id: "dashboard",
        status: Status::Pending,
        paths: &[
            "model.gearbox_input_ripple_Nm",
            "model.cup_od_mm",
            "mass.total_g",
            "metal.min_running_clearance_mm",
            "metal.clearance_check",
            "materials.cup_wall_check",
            "clamps.recommended",
        ],
    },
];
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs` with exactly this content:

```rust
//! Display symbols of the terms that have no record of their own: every input a formula
//! reads, and any result shown with its workbook cell only. A record's target takes the
//! record's symbol. The registry refuses a term without a symbol, an entry no formula
//! reads, an entry for a path with a record, and two paths with one symbol.
//!
//! Conventions (plan A-3): T torque, σ shear stress, ϑ temperature, φ electrical
//! angle, τ_p pole pitch; `^{cal}` marks the Calibration prototype; a selector is an upright
//! word (`faceted`), which the typesetter writes with the choice's label in conditions.

/// (path, symbol markup), grouped by input group.
pub const SYMBOLS: &[(&str, &str)] = &[
    // coupling
    ("coupling.npole", "N"),
    ("coupling.backiron", "backiron"),
    ("coupling.faceted", "faceted"),
    ("coupling.inner_back_apothem_mm", "a_i"),
    ("coupling.op_temp_C", "ϑ_{op}"),
    ("coupling.c_end", "c_{end}"),
    ("coupling.max_harmonic", "N_h"),
    ("coupling.mu0", "μ_0"),
    ("coupling.gear_ratio", "i_g"),
    ("coupling.gear_efficiency", "η_g"),
    ("coupling.magnets.part_inner", "part_i"),
    ("coupling.magnets.part_outer", "part_o"),
    ("coupling.magnets.grade_inner", "grade_i"),
    ("coupling.magnets.grade_outer", "grade_o"),
    ("coupling.magnets.axial_length_mm", "L_{ax}"),
    ("coupling.magnets.manual_inner_length_mm", "L_{i,man}"),
    ("coupling.magnets.manual_outer_length_mm", "L_{o,man}"),
    ("coupling.magnets.manual_inner_width_mm", "w_{i,man}"),
    ("coupling.magnets.manual_outer_width_mm", "w_{o,man}"),
    ("coupling.magnets.manual_inner_thickness_mm", "t_{i,man}"),
    ("coupling.magnets.manual_outer_thickness_mm", "t_{o,man}"),
    ("coupling.magnets.manual_inner_br_T", "B_{r,i,man}"),
    ("coupling.magnets.manual_outer_br_T", "B_{r,o,man}"),
    // metal
    ("metal.face_gap_mm", "g_{face}"),
    ("metal.min_temp_C", "ϑ_{min}"),
    ("metal.variation", "v"),
    ("metal.required_min_Nm", "T_{req}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
    ("calibration.spacing_mm", "s^{cal}"),
    ("calibration.gap_definition", "gapdef"),
    ("calibration.test_temp_C", "ϑ_{test}"),
    ("calibration.apothem_mm", "a^{cal}"),
    ("calibration.magnet_length_mm", "L^{cal}"),
    ("calibration.magnet_width_mm", "w^{cal}"),
    ("calibration.magnet_thickness_mm", "t^{cal}"),
    ("calibration.br_T", "B_r^{cal}"),
    ("calibration.alpha_br_per_C", "α"),
    ("calibration.c_end", "c_{end}^{cal}"),
    ("calibration.f_cal_original", "f_{cal,0}"),
    ("calibration.mu0", "μ_0^{cal}"),
    ("calibration.fea_torque1_Nm", "T_{3D,1}"),
    ("calibration.fea_torque2_Nm", "T_{3D,2}"),
    // materials
    ("materials.parts.back_iron", "material_{BI}"),
];
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

- [ ] **Step 4: Run the drift guard and the structure tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 5 passed; 0 failed; 1 ignored` (the drift guard, the disjoint paths, the registry, the drill-down and the scope tests; `review_sheet` is ignored), in about 4 s with 32 threads: the guard evaluates 127 equations at 40,693 input sets (the defaults, then 3,391 differential cases, each as generated and under 11 augmentations).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib explain 2>&1 | grep "test result"
```

Expected: `test result: ok. 23 passed` (Task 2's 20 and the registry's 3).

- [ ] **Step 5: Physics review of the torque chain (session model)**

Dispatch the physics reviewer on the **session model** (omit `model`, comment `// session model: physics`; CLAUDE.md section 5) with the prompt below, and give it the sheet this command prints:

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E "model\.|calibration\.|metal\.|materials\."
```

Expected: one Markdown table row per record of the batch: path, workbook cell (or `Rust-only`), the rendered formula and the corrections it embodies.

Reviewer prompt: "Review the torque-chain equation records of the magcoupling equation explorer (the sheet below) against the M1 audit (docs/analyses/2026-09-29-magcoupling-math-audit.md) rows M3 (planar S_n), M4 and M7 (the square-wave harmonics), M5 (steel at the magnet backs), M8 and M9 (the end factor), E7 (pull-out at the true peak angle), E8 (the corner gap) and E3 (library Br), and Addendum A decision 29 and A-2 decision A2-7. The drift guard already proves each formula reproduces the engine; review what numbers cannot: (1) each symbol follows the conventions of `src/engine/explain/symbols.rs` (T torque, σ shear stress, ϑ temperature, φ angle, τ_p pole pitch) and no two paths share one; (2) each formula is the physics the M1 audit confirmed for that cell, stated one step from its terms; (3) each record names the corrections it embodies; (4) these reasoning claims hold: σ_n is A_n sin(n φ_pull) for a summed harmonic and 0 otherwise, A_n = B_i,n B_o,n S_n/(2 μ0) in the circuit in effect; a non-ferromagnetic back iron selects the free-space circuit; f_cal is the bench correction only for the prototype's own rings, pole count and circuit; the hot and cold torques scale with each ring's own Br coefficient. Answer APPROVED, or list each finding with the record and the fix."

On APPROVED, continue. On findings: fix each in the batch's records file (a symbol in `symbols.rs`), re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain` until green, and send the changed rows back to the same reviewer; the task does not commit until the reviewer approves. A finding that would change the engine is out of scope: stop and escalate.

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task3.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 7: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain magcoupling-rs/tests/common/mod.rs magcoupling-rs/tests/differential.rs magcoupling-rs/tests/explain.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): the equation registry, the torque-chain records and the drift guard

Records state each result one step from its terms; the registry checks every
path, unit, table field and symbol, refuses a path read in two units and
markup the typesetter would draw two ways, and builds the dependency graph. The torque
chain and its closure (127 equations) drill down to inputs, and the drift guard
proves every record against the engine at the defaults and every differential
case under 11 augmentations, every cases arm taken and every value term varied.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 4: A3 traceability and the styling data

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Step 5 (the cancellations) is a physics review on the session model.

The spec's A3 test, "changing each assumption changes every dependent result and no independent one, using the equation
registry's dependency graph", is false as worded in three ways: verdict text is piecewise constant, a `min` or an untaken branch is a
static dependency with zero sensitivity at a design point, and some dependencies cancel algebraically. So it is two one-sided checks.
**Soundness, for every input** (172): at 51 design points, each nudge (a step both ways inside the slider, or around a value typed outside it such as a
ferrite beta; the slider's two ends, so a threshold flips its verdict; every other choice; another part and manual; set and unset)
leaves every explained result outside `Registry::downstream(input)` bit-identical, which proves the graph complete; and every input
has an accepted nudge that changes some result at some point, so the check is not vacuous for it (the two inputs no result reads
are listed in `INERT_INPUTS` with their reasons). **Sensitivity, for the 15 assumption inputs**: every numeric result on the input's active path (reachable along the value
terms of each record's trace at that point) must move by more than 1e-10 relative for some nudge at some point; what never moves is a
spurious edge or a cancellation, and a cancellation must be listed in `CANCELLATIONS` with its reason (a listed pair that moves fails).
Five exist today, each a physics fact. The styling data the M4 panel reads (an assumption term, the changed-from-default dot, a result
downstream of a modified assumption) comes from the registry's precomputed `upstream_inputs` and A-2's `assumptions` module; it styles
the panel's terms only (a cell-only result has no dependencies in the registry, so it is never marked affected). For the panel the
registry also gives a selector code's choice labels (`choices`; `materials.circuit_backiron` borrows `coupling.backiron`'s through
`RESULT_CHOICES`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker: the
pull-out's own record names none, but E7 and E8 act upstream of it). Task 17 rewords the spec's A3 test line to what this checks
(decision G7).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (the A3 traceability tests (soundness and its non-vacuity check, sensitivity, `CANCELLATIONS`, `INERT_INPUTS`), the styling test and the panel lookups test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs` (`TermStyle`, `upstream_inputs`, `downstream`, `term_style`, `modified_assumptions_upstream`; `RESULT_CHOICES`, `choices`, `corrections_upstream`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs` (re-export `TermStyle`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the explain.rs tests row, the explorer section)

**Interfaces:**
- Consumes: Task 3's registry and records; `assumptions::{ASSUMPTIONS, Assumption, modified, any_modified, reset_to_workbook_defaults}` (A-2); `eval::Trace`.
- Produces:
  - `registry::TermStyle { kind: TermKind, changed_from_default: bool, affected_by_modified_assumption: bool }`;
  - `Registry::upstream_inputs(path) -> Option<&BTreeSet<String>>` (precomputed transitive leaves), `Registry::downstream(path) -> BTreeSet<String>` (transitive users), `Registry::term_style(path, &DesignInputs) -> Option<TermStyle>`, `Registry::modified_assumptions_upstream(path, &DesignInputs) -> Vec<&'static Assumption>`;
  - for the M4 panel: `registry::RESULT_CHOICES: &[(&str, &str)]` (a result holding a selector code, and the input whose choices label it), `Registry::choices(path) -> &'static [(i64, &'static str)]`, `Registry::corrections_upstream(path) -> Vec<DeviationId>` (the record's own corrections, then every upstream record's, each once);
  - in `tests/explain.rs`: `nudges`, `trace_points`, `check_traceability(paths, sensitivity) -> Vec<String>`, `trace_at` and its `PointTrace`, `CANCELLATIONS`, `INERT_INPUTS`, the tests `each_assumption_moves_what_depends_on_it_and_nothing_else`, `no_input_moves_a_result_the_graph_says_is_independent_of_it`, `a_modified_assumption_styles_its_term_and_what_it_flows_into`, `the_panel_labels_selector_codes_and_names_upstream_corrections`.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
//! Addendum A2 explanation layer: the drift guard, the registry's structure and the v1
//! scope.
//!
//! **Drift guard.** Every equation record is evaluated over the engine's own term values and
//! must reproduce the engine's result by the parity rule (1e-9 relative, 1e-12 absolute;
```

with:

```rust
//! Addendum A2/A3 explanation layer: the drift guard, the registry's structure, the v1
//! scope and the A3 traceability test.
//!
//! **Drift guard.** Every equation record is evaluated over the engine's own term values and
//! must reproduce the engine's result by the parity rule (1e-9 relative, 1e-12 absolute;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
//! 0 with 0. Anti-vacuity checks require every `cases` arm of every record to be taken, every
//! numeric record and every value term of every record to take two values, and the E7
//! angles to leave half a pitch.

mod common;
```

with:

```rust
//! 0 with 0. Anti-vacuity checks require every `cases` arm of every record to be taken, every
//! numeric record and every value term of every record to take two values, and the E7
//! angles to leave half a pitch.
//!
//! **Traceability (A3).** For each input (the assumptions first, as the spec asks, then every
//! input), at several design points: nudging it changes no explained result outside its
//! static dependency set (the registry's graph), and changes every numeric result on its
//! active dependency path (the branch taken, the min/max winner; see `eval::Trace`).

mod common;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
use std::f64::consts::FRAC_PI_2;

use common::{differential_files, load_cases, report};
use magcoupling::engine::api::{DesignInputs, compute_all};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::explain::markup::{Expr, Symbol};
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, render};
use magcoupling::engine::meta::{InputSet, ResultSet, Value, input_rows, result_rows};

/// One input set the guard evaluates at.
struct Point {
```

with:

```rust
use std::f64::consts::FRAC_PI_2;

use common::{differential_files, load_cases, report};
use magcoupling::engine::api::{DesignInputs, DesignResults, compute_all};
use magcoupling::engine::assumptions::{self, ASSUMPTIONS};
use magcoupling::engine::compat::parity_close;
use magcoupling::engine::deviations::DeviationId;
use magcoupling::engine::explain::markup::{Expr, Symbol};
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, render};
use magcoupling::engine::library;
use magcoupling::engine::meta::{
    FieldType, InputMeta, InputSet, ResultSet, Value, input_rows, result_rows,
};

/// One input set the guard evaluates at.
struct Point {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
        "decision 31: the chains and the dashboard (report section 7)"
    );
}
```

with:

```rust
        "decision 31: the chains and the dashboard (report section 7)"
    );
}

// ---------------------------------------------------------------------------------------
// A3 traceability
// ---------------------------------------------------------------------------------------

/// The nudged values of an input: a step both ways inside its slider for a number (around
/// the value when it is typed outside the slider), one step for a count, and for either the
/// slider's two ends; every other choice for a selector; another library part and "manual"
/// for a part name; set and unset for an optional number.
fn nudges(meta: &InputMeta, value: &Value) -> Vec<Value> {
    match (meta.ty, value) {
        (FieldType::I64, Value::Int(code)) if !meta.choices.is_empty() => meta
            .choices
            .iter()
            .map(|&(c, _)| c)
            .filter(|c| c != code)
            .map(Value::Int)
            .collect(),
        (FieldType::I64, Value::Int(n)) => {
            let r = meta.range.expect("a count has a slider");
            let step = r.step.max(1.0) as i64;
            let mut v = vec![if (*n + step) as f64 <= r.max {
                n + step
            } else {
                n - step
            }];
            // The slider's ends too: a threshold flips its verdict only under a large move.
            for end in [r.min as i64, r.max as i64] {
                if end != *n && !v.contains(&end) {
                    v.push(end);
                }
            }
            v.into_iter().map(Value::Int).collect()
        }
        (FieldType::F64 | FieldType::OptF64, Value::Num(x)) => {
            // Both ways inside the slider: a min() tie moves the result one way only. A value
            // typed outside the slider (hard ferrite's positive beta) is nudged where it is.
            let r = meta.range.expect("a number has a slider");
            let inside = |y: &f64| (r.min..=r.max).contains(y);
            let step = r.step.max(1e-3 * x.abs()).max(1e-12);
            let mut ys: Vec<f64> = if inside(x) {
                [x + step, x - step].into_iter().filter(inside).collect()
            } else {
                vec![x + step, x - step]
            };
            // The slider's ends too: a threshold (a minimum wall, a required torque) flips its
            // verdict only under a large move, and a slider narrower than the step (μ0's) is
            // still nudged.
            for end in [r.min, r.max] {
                if end != *x && !ys.contains(&end) {
                    ys.push(end);
                }
            }
            let mut v: Vec<Value> = ys.into_iter().map(Value::Num).collect();
            if meta.ty == FieldType::OptF64 {
                v.push(Value::None);
            }
            v
        }
        (FieldType::OptF64, Value::None) => {
            let r = meta.range.expect("a number has a slider");
            vec![Value::Num((r.min + r.max) / 2.0)]
        }
        (FieldType::Text, Value::Text(s)) if meta.name.starts_with("part_") => {
            let other = library::MAGNET_LIBRARY
                .iter()
                .map(|m| m.part)
                .find(|p| p != s)
                .expect("two parts");
            vec![Value::Text(other.to_owned()), Value::Text(String::new())]
        }
        (FieldType::Text, Value::Text(s)) if meta.name.starts_with("grade_") => {
            let g = if s == "N52" { "Y30" } else { "N52" };
            vec![Value::Text(g.to_owned()), Value::Text(String::new())]
        }
        (FieldType::Text, _) => vec![Value::Text("Loctite AA 326 + SF 7649".into())],
        other => panic!("{}: no nudge for {other:?}", meta.name),
    }
}

/// Design points for the traceability test: the defaults; the measured prototype's own
/// circuit (6061 back iron, so the bench correction is the calibration factor) at a test
/// temperature off 20 °C; a grade-mode ring with eleven harmonics at six poles; and a dozen
/// full-run cases under four of the drift guard's augmentations.
fn trace_points() -> Vec<(String, DesignInputs)> {
    let design = |sets: &[(&str, Value)]| {
        let mut inputs = DesignInputs::default();
        for (p, v) in sets {
            inputs.set(p, v.clone()).unwrap();
        }
        inputs
    };
    let text = |s: &str| Value::Text(s.to_owned());
    let mut pts = vec![
        ("defaults".to_owned(), DesignInputs::default()),
        (
            "prototype circuit, test at 35 °C".to_owned(),
            design(&[
                ("materials.parts.back_iron", Value::Int(8)),
                ("calibration.test_temp_C", Value::Num(35.0)),
            ]),
        ),
        (
            "grade, 11 harmonics, 6 poles, test at 35 °C".to_owned(),
            design(&[
                ("coupling.max_harmonic", Value::Int(11)),
                ("coupling.magnets.part_inner", text("")),
                ("coupling.magnets.grade_inner", text("N52")),
                ("coupling.npole", Value::Int(6)),
                ("calibration.test_temp_C", Value::Num(35.0)),
            ]),
        ),
    ];
    let augmentations: Vec<_> = augmentations()
        .into_iter()
        .filter(|(l, _)| {
            [
                "as generated",
                "harmonics 11",
                "back iron 6061",
                "grade mode",
            ]
            .contains(l)
        })
        .collect();
    for case in load_cases("full").into_iter().take(12) {
        for (label, sets) in &augmentations {
            let mut inputs = DesignInputs::default();
            for (path, value) in case
                .inputs
                .iter()
                .map(|(p, v)| (p.as_str(), v))
                .chain(sets.iter().map(|(p, v)| (*p, v)))
            {
                inputs.set(path, value.clone()).unwrap();
            }
            pts.push((format!("full case {}, {label}", case.id), inputs));
        }
    }
    pts
}

/// Any change at all: what soundness forbids for an independent result.
fn changed(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => x.to_bits() != y.to_bits() && !(x.is_nan() && y.is_nan()),
        _ => a != b,
    }
}

/// A change above rounding noise (1e-10 relative): what sensitivity requires, so that an
/// algebraic cancellation cannot pass on its last-bit wobble.
fn moved_beyond_rounding(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Num(x), Value::Num(y)) => (x - y).abs() > 1e-10 * x.abs().max(y.abs()),
        _ => a != b,
    }
}

/// Each term's users along value edges at a design point: the records whose value moves
/// with that term there (each record's trace, `eval::Trace::value_terms`).
fn value_edges(
    r: &Registry,
    inputs: &DesignInputs,
    results: &DesignResults,
) -> BTreeMap<String, Vec<String>> {
    let src = Design { inputs, results };
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for eq in r.equations() {
        let mut t = Trace::default();
        r.evaluate(eq, &src, Some(&mut t))
            .expect("the drift guard evaluates every record");
        for term in t.value_terms {
            edges.entry(term).or_default().push(eq.target.clone());
        }
    }
    edges
}

/// The explained results on `input`'s active path: reachable from it along value edges.
fn active_downstream(value_edges: &BTreeMap<String, Vec<String>>, input: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut stack = vec![input.to_owned()];
    while let Some(p) = stack.pop() {
        for user in value_edges.get(&p).into_iter().flatten() {
            if seen.insert(user.clone()) {
                stack.push(user.clone());
            }
        }
    }
    seen
}

/// Checks both sides of traceability for every input in `paths`; returns failures.
///
/// Soundness (nothing independent moves) holds at every point and nudge. Sensitivity (the
/// active path moves) is judged across the points together: an input need move a result at
/// one point where it is on its active path, so a point where a factor happens to vanish
/// (the test temperature at 20 °C in `1 + α (ϑ − 20)`) does not fail it. What never moves at
/// any point although the graph says it depends is an algebraic cancellation, which must be
/// listed in [`CANCELLATIONS`] with its reason (and a listed pair that does move fails).
fn check_traceability(paths: &[String], sensitivity: bool) -> Vec<String> {
    let r = Registry::build();
    let points = trace_points();
    let mut failures = Vec::new();
    let mut must_move: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut moved_pairs: BTreeSet<(String, String)> = BTreeSet::new();
    let mut effective: BTreeSet<String> = BTreeSet::new();
    std::thread::scope(|s| {
        let handles: Vec<_> = points
            .iter()
            .map(|(label, base)| s.spawn(|| trace_at(&r, label, base, paths)))
            .collect();
        for h in handles {
            let t = h.join().expect("a traceability thread");
            failures.extend(t.failures);
            for (k, at) in t.must_move {
                must_move.entry(k).or_insert(at);
            }
            moved_pairs.extend(t.moved);
            effective.extend(t.effective);
        }
    });
    // Not vacuous: every input checked had a nudge `set` accepted that changed some result
    // (explained or not) at some design point, so soundness was put to the test for it;
    // an input no result reads is listed in INERT_INPUTS instead.
    for input in paths {
        let inert = INERT_INPUTS.iter().any(|&(i, _)| i == input);
        match (effective.contains(input), inert) {
            (false, false) => failures.push(format!(
                "{input}: no accepted nudge changed any result at any design point"
            )),
            (true, true) => failures.push(format!(
                "INERT_INPUTS lists {input}, but a nudge changed a result: remove the entry"
            )),
            _ => {}
        }
    }
    if !sensitivity {
        return failures;
    }
    for ((input, result), at) in &must_move {
        let key = (input.clone(), result.clone());
        let listed = CANCELLATIONS
            .iter()
            .any(|&(i, r, _)| i == input && r == result);
        if !moved_pairs.contains(&key) && !listed {
            failures.push(format!("{input} never moves {result}, which is on its active path (first at {at}): a spurious edge, or a cancellation to list"));
        }
    }
    for &(input, result, _) in CANCELLATIONS {
        if paths.iter().any(|p| p == input)
            && moved_pairs.contains(&(input.to_owned(), result.to_owned()))
        {
            failures.push(format!(
                "CANCELLATIONS lists {input} → {result}, but it moves: remove the entry"
            ));
        }
    }
    failures
}

/// Inputs no result reads (the workbook shows them, or only the drawing reads them): no
/// nudge can change anything, so the non-vacuity check of [`check_traceability`] skips them
/// (and fails if a listed one ever changes a result).
const INERT_INPUTS: &[(&str, &str)] = &[
    (
        "materials.steel.density_g_cm3",
        "Materials C19 is shown for reference: the mass model reads Metal design C132 (the input's help says so)",
    ),
    (
        "clamps.key_width_mm",
        "only the drawing reads it (Python drawing.py); the key pressure divides by the contact height",
    ),
];

/// Pairs (input, result).
type Pairs = BTreeSet<(String, String)>;

/// What one design point of [`check_traceability`] found.
struct PointTrace {
    /// The soundness failures.
    failures: Vec<String>,
    /// The pairs on an active path that did not move, with where.
    must_move: BTreeMap<(String, String), String>,
    /// The pairs that moved above rounding.
    moved: Pairs,
    /// The inputs an accepted nudge changed some result of (explained or not).
    effective: BTreeSet<String>,
}

/// One design point of [`check_traceability`].
fn trace_at(r: &Registry, label: &str, base: &DesignInputs, paths: &[String]) -> PointTrace {
    let metas: BTreeMap<String, &'static InputMeta> = input_rows(&DesignInputs::default())
        .into_iter()
        .map(|x| (x.path, x.meta))
        .collect();
    let mut failures = Vec::new();
    let mut must_move: BTreeMap<(String, String), String> = BTreeMap::new();
    let mut moved_pairs = Pairs::new();
    let mut effective = BTreeSet::new();
    {
        let base_results = compute_all(base);
        let base_rows = result_rows(&base_results);
        let edges = value_edges(r, base, &base_results);
        for path in paths {
            let meta = metas[path];
            let value = base.get(path).expect("an input path");
            let statics = r.downstream(path);
            let active = active_downstream(&edges, path);
            let _ = &statics;
            for nudged in nudges(meta, &value) {
                let mut inputs = base.clone();
                if inputs.set(path, nudged.clone()).is_err() {
                    continue; // outside the choices this design allows
                }
                let results = compute_all(&inputs);
                let mut any = false;
                for eq in r.equations() {
                    let (b, a) = (
                        base_results.get(&eq.target).unwrap(),
                        results.get(&eq.target).unwrap(),
                    );
                    let moved = changed(&b, &a);
                    any |= moved;
                    if moved && !statics.contains(&eq.target) {
                        failures.push(format!("{label}: {path} → {nudged:?} moves {} ({b:?} → {a:?}), which the graph says does not depend on it", eq.target));
                    }
                    let key = (path.clone(), eq.target.clone());
                    if moved_beyond_rounding(&b, &a) {
                        moved_pairs.insert(key);
                    } else if matches!(b, Value::Num(x) if x.is_finite())
                        && active.contains(&eq.target)
                    {
                        must_move
                            .entry(key)
                            .or_insert_with(|| format!("{label}, {nudged:?}"));
                    }
                }
                if !any && !effective.contains(path) {
                    // No explained result moved: did anything (a result outside the scope)?
                    any = result_rows(&results)
                        .iter()
                        .zip(&base_rows)
                        .any(|(a, b)| changed(&a.value, &b.value));
                }
                if any {
                    effective.insert(path.clone());
                }
            }
        }
    }
    PointTrace {
        failures,
        must_move,
        moved: moved_pairs,
        effective,
    }
}

/// Dependencies the graph names that cancel algebraically: the formula reads the input on
/// the way, but the result never moves with it. Each is a fact worth knowing (a teaching
/// note may cite it); `check_traceability` fails if a listed pair ever moves.
const CANCELLATIONS: &[(&str, &str, &str)] = &[
    (
        "calibration.f_cal_original",
        "calibration.f_cal_updated",
        "f_cal,1 = f_cal,0 · T_meas / T_model and T_model = T_2D f_end f_cal,0: the assumed factor cancels, so the bench correction is T_meas / (T_2D f_end)",
    ),
    (
        "calibration.alpha_br_per_C",
        "model.pullout_angle_rad",
        "alpha scales every harmonic amplitude by the same Br(T) ratio, and a common scale leaves the peak angle where it is",
    ),
    (
        "calibration.alpha_br_per_C",
        "model.iron_circuit_angle_rad",
        "as model.pullout_angle_rad",
    ),
    (
        "calibration.alpha_br_per_C",
        "model.free_circuit_angle_rad",
        "as model.pullout_angle_rad",
    ),
    (
        "calibration.alpha_br_per_C",
        "calibration.pullout_angle_rad",
        "as model.pullout_angle_rad, through the prototype's Br at the test temperature",
    ),
];

#[test]
fn each_assumption_moves_what_depends_on_it_and_nothing_else() {
    // Spec A3 testing: the fifteen assumption inputs (the fourteen rows; end effect is two).
    let paths: Vec<String> = ASSUMPTIONS
        .iter()
        .flat_map(|a| a.paths.iter().map(|p| (*p).to_owned()))
        .collect();
    assert_eq!(paths.len(), 15);
    let failures = check_traceability(&paths, true);
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn no_input_moves_a_result_the_graph_says_is_independent_of_it() {
    // Soundness for every input: the graph is complete. (Sensitivity is asserted for the
    // assumptions, as the spec asks: for all inputs the E8 corner geometry alone cancels
    // dozens of structural paths, e.g. the inner width reaches A_o through r_corner twice
    // with opposite signs.)
    let paths: Vec<String> = input_rows(&DesignInputs::default())
        .into_iter()
        .map(|x| x.path)
        .collect();
    let failures = check_traceability(&paths, false);
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn a_modified_assumption_styles_its_term_and_what_it_flows_into() {
    let r = Registry::build();
    let mut inputs = DesignInputs::default();
    assert!(!assumptions::any_modified(&inputs));
    let style = |p: &str, i: &DesignInputs| r.term_style(p, i).expect("a known path");
    assert!(!style("metal.variation", &inputs).changed_from_default);
    inputs.set("metal.variation", Value::Num(0.2)).unwrap();
    assert!(assumptions::any_modified(&inputs), "the banner shows");
    let v = style("metal.variation", &inputs);
    assert_eq!(v.kind, TermKind::Input { assumption: true });
    assert!(v.changed_from_default);
    assert!(style("metal.torque_hot_low_Nm", &inputs).affected_by_modified_assumption);
    assert!(
        !style("model.pullout_Nm", &inputs).affected_by_modified_assumption,
        "variation does not reach the pull-out"
    );
    assert_eq!(
        r.modified_assumptions_upstream("metal.torque_cold_high_Nm", &inputs)
            .iter()
            .map(|a| a.id)
            .collect::<Vec<_>>(),
        ["production_variation"]
    );
    // A design input changed from its default gets the dot but is not an assumption.
    inputs.set("coupling.npole", Value::Int(12)).unwrap();
    let n = style("coupling.npole", &inputs);
    assert_eq!(n.kind, TermKind::Input { assumption: false });
    assert!(n.changed_from_default);
    assumptions::reset_to_workbook_defaults(&mut inputs);
    assert!(!assumptions::any_modified(&inputs), "the banner clears");
    assert!(!style("metal.torque_hot_low_Nm", &inputs).affected_by_modified_assumption);
    assert!(
        style("coupling.npole", &inputs).changed_from_default,
        "the reset keeps design inputs"
    );
}

#[test]
fn the_panel_labels_selector_codes_and_names_upstream_corrections() {
    let r = Registry::build();
    // A selector input shows its choice labels; a result holding a code borrows its input's.
    assert_eq!(
        r.choices("coupling.backiron"),
        [(1, "steel circuit"), (0, "no back iron")]
    );
    assert_eq!(
        r.choices("materials.circuit_backiron"),
        r.choices("coupling.backiron")
    );
    assert_eq!(
        r.choices("materials.parts.back_iron")[0],
        (1, "4140 annealed")
    );
    assert!(r.choices("coupling.npole").is_empty() && r.choices("model.pullout_Nm").is_empty());
    // The "corrected vs workbook" marker: the pull-out's own record names no correction, but
    // E7 acts upstream of it (the pull-out angle) and E8 in the corner gap.
    assert!(
        r.equation_for("model.pullout_Nm")
            .unwrap()
            .corrections
            .is_empty()
    );
    let pullout = r.corrections_upstream("model.pullout_Nm");
    assert!(
        pullout.contains(&DeviationId::E7) && pullout.contains(&DeviationId::E8),
        "{pullout:?}"
    );
    // A record's own corrections come first, each once.
    assert_eq!(
        r.corrections_upstream("model.pullout_angle_rad")[0],
        DeviationId::E7
    );
    for (i, c) in pullout.iter().enumerate() {
        assert!(!pullout[..i].contains(c), "{c:?} twice in {pullout:?}");
    }
    assert!(
        r.corrections_upstream("coupling.npole").is_empty(),
        "an input"
    );
}
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "^error" | head -3
```

Expected: ``error[E0599]: no method named `downstream` found for reference `&Registry` in the current scope``, then the same for `term_style` and `modified_assumptions_upstream` (`choices` and `corrections_upstream` follow without `head`).

- [ ] **Step 3: Add the styling data to the registry**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

with:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs`, replace:

```rust
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron);
//! - [`record`]: the authoring form, [`records`]: the records, one file per batch;
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term rows;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
//! - [`scope`]: the v1 scope (decision 31) and which chains are written;
//! - [`render`]: a plain-text rendering (tooltips fallback, export, tests).
```

with:

```rust
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron);
//! - [`record`]: the authoring form, [`records`]: the records, one file per batch;
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term styles;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
//! - [`scope`]: the v1 scope (decision 31) and which chains are written;
//! - [`render`]: a plain-text rendering (tooltips fallback, export, tests).
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs`, replace:

```rust
pub mod tables;

pub use eval::{EvalError, TermSource, Trace};
pub use registry::{Design, Equation, Registry, TermKind, TermRow};
```

with:

```rust
pub mod tables;

pub use eval::{EvalError, TermSource, Trace};
pub use registry::{Design, Equation, Registry, TermKind, TermRow, TermStyle};
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
use super::symbols::SYMBOLS;
use super::tables;
use crate::engine::api::{DesignInputs, DesignResults, compute_all};
use crate::engine::deviations::DeviationId;
use crate::engine::meta::{
    InputMeta, InputSet, ResultMeta, ResultSet, Value, input_rows, result_rows,
```

with:

```rust
use super::symbols::SYMBOLS;
use super::tables;
use crate::engine::api::{DesignInputs, DesignResults, compute_all};
use crate::engine::assumptions::{self, Assumption};
use crate::engine::deviations::DeviationId;
use crate::engine::meta::{
    InputMeta, InputSet, ResultMeta, ResultSet, Value, input_rows, result_rows,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
    ("GPa", "Pa", 1e9),
    ("g", "kg", 1e-3),
];

/// The factor that converts a value in `from` to `to`.
pub fn conversion(from: &str, to: &str) -> Option<f64> {
```

with:

```rust
    ("GPa", "Pa", 1e9),
    ("g", "kg", 1e-3),
];

/// Results that hold a selector code, each with the input whose choices label it (the panel
/// shows `materials.circuit_backiron = 1` as "steel circuit"): [`Registry::choices`].
pub const RESULT_CHOICES: &[(&str, &str)] = &[("materials.circuit_backiron", "coupling.backiron")];

/// The factor that converts a value in `from` to `to`.
pub fn conversion(from: &str, to: &str) -> Option<f64> {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
    CellOnly,
}

/// The registry.
pub struct Registry {
    equations: Vec<Equation>,
```

with:

```rust
    CellOnly,
}

/// How the explorer styles a term for the current inputs (Addendum A3 traceability).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TermStyle {
    pub kind: TermKind,
    /// An input that differs from its default (the changed-from-default dot).
    pub changed_from_default: bool,
    /// A result some modified assumption flows into (by the dependency graph).
    pub affected_by_modified_assumption: bool,
}

/// The registry.
pub struct Registry {
    equations: Vec<Equation>,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
    family_symbols: BTreeMap<String, String>,
    inputs: BTreeMap<String, &'static InputMeta>,
    results: BTreeMap<String, &'static ResultMeta>,
}

impl Registry {
```

with:

```rust
    family_symbols: BTreeMap<String, String>,
    inputs: BTreeMap<String, &'static InputMeta>,
    results: BTreeMap<String, &'static ResultMeta>,
    upstream_inputs: BTreeMap<String, BTreeSet<String>>,
    defaults: DesignInputs,
}

impl Registry {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
        let results: BTreeMap<String, &'static ResultMeta> =
            result_rows.into_iter().map(|r| (r.path, r.meta)).collect();
        let mut errors = Vec::new();

        // Expand the families into one record per index.
        let mut authored: Vec<(Record, Option<u32>)> = plain.iter().map(|r| (*r, None)).collect();
```

with:

```rust
        let results: BTreeMap<String, &'static ResultMeta> =
            result_rows.into_iter().map(|r| (r.path, r.meta)).collect();
        let mut errors = Vec::new();
        for &(result, input) in RESULT_CHOICES {
            if !results.contains_key(result)
                || inputs.get(input).is_none_or(|m| m.choices.is_empty())
            {
                errors.push(format!(
                    "RESULT_CHOICES: {result} must be a result and {input} a selector input"
                ));
            }
        }

        // Expand the families into one record per index.
        let mut authored: Vec<(Record, Option<u32>)> = plain.iter().map(|r| (*r, None)).collect();
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
            .iter()
            .map(|e| (e.target.as_str(), e.terms.as_slice()))
            .collect();
        for eq in &equations {
            if closure(&graph, &eq.target).1 {
                errors.push(format!(
                    "{}: its terms lead back to it (a cycle)",
                    eq.target
                ));
            }
        }

        if errors.is_empty() {
```

with:

```rust
            .iter()
            .map(|e| (e.target.as_str(), e.terms.as_slice()))
            .collect();
        let mut upstream_inputs = BTreeMap::new();
        for eq in &equations {
            let (seen, cycle) = closure(&graph, &eq.target);
            if cycle {
                errors.push(format!(
                    "{}: its terms lead back to it (a cycle)",
                    eq.target
                ));
            }
            let leaves: BTreeSet<String> = seen
                .into_iter()
                .filter(|p| inputs.contains_key(p))
                .collect();
            upstream_inputs.insert(eq.target.clone(), leaves);
        }

        if errors.is_empty() {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
                family_symbols,
                inputs,
                results,
            })
        } else {
            Err(errors)
```

with:

```rust
                family_symbols,
                inputs,
                results,
                upstream_inputs,
                defaults,
            })
        } else {
            Err(errors)
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
        self.symbols.get(path).map(String::as_str)
    }

    /// The generic symbol of a family term inside a Σ (`model.b_i#` gives `B_{i,n}`).
    pub fn family_symbol(&self, template: &str) -> Option<String> {
        self.family_symbols
```

with:

```rust
        self.symbols.get(path).map(String::as_str)
    }

    /// The choice labels of the selector code at `path`: an input's own, or for a result that
    /// holds a code ([`RESULT_CHOICES`]) its input's; empty for any other path.
    pub fn choices(&self, path: &str) -> &'static [(i64, &'static str)] {
        let input = RESULT_CHOICES
            .iter()
            .find(|&&(result, _)| result == path)
            .map_or(path, |&(_, input)| input);
        self.inputs.get(input).map_or(&[], |m| m.choices)
    }

    /// The corrections a result embodies: its own record's, then those of every explained
    /// result upstream of it (in path order), each once. The M4 "corrected vs workbook" marker
    /// reads this: a correction acts where it is applied (E7 in the pull-out angle) and flows
    /// into everything downstream, so `model.pullout_Nm`, whose record names none, shows E7.
    /// Empty for an input or a cell-only result.
    pub fn corrections_upstream(&self, path: &str) -> Vec<DeviationId> {
        let Some(eq) = self.equation_for(path) else {
            return Vec::new();
        };
        let upstream = self.upstream(path);
        let mut out: Vec<DeviationId> = Vec::new();
        let theirs = upstream
            .iter()
            .filter_map(|p| self.equation_for(p))
            .flat_map(|e| e.corrections.iter());
        for c in eq.corrections.iter().chain(theirs) {
            if !out.contains(c) {
                out.push(*c);
            }
        }
        out
    }

    /// The generic symbol of a family term inside a Σ (`model.b_i#` gives `B_{i,n}`).
    pub fn family_symbol(&self, template: &str) -> Option<String> {
        self.family_symbols
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
    pub fn upstream(&self, path: &str) -> BTreeSet<String> {
        let graph: BTreeMap<&str, &[String]> = self.graph().collect();
        closure(&graph, path).0
    }

    /// The value of every term of `eq`, in `eq.terms` order, in each term's own unit.
```

with:

```rust
    pub fn upstream(&self, path: &str) -> BTreeSet<String> {
        let graph: BTreeMap<&str, &[String]> = self.graph().collect();
        closure(&graph, path).0
    }

    /// The inputs `path` depends on, transitively (precomputed).
    pub fn upstream_inputs(&self, path: &str) -> Option<&BTreeSet<String>> {
        self.upstream_inputs.get(path)
    }

    /// Every explained result that depends on `path`, transitively.
    pub fn downstream(&self, path: &str) -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![path.to_owned()];
        while let Some(p) = stack.pop() {
            for user in self.used_by(&p) {
                if seen.insert(user.clone()) {
                    stack.push(user.clone());
                }
            }
        }
        seen
    }

    /// The value of every term of `eq`, in `eq.terms` order, in each term's own unit.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/registry.rs`, replace:

```rust
                Ok(v)
            }
        }
    }
}
```

with:

```rust
                Ok(v)
            }
        }
    }

    /// How to style `path` for `inputs` (A3: assumption terms, the changed-from-default dot,
    /// results a modified assumption flows into). It styles the panel's terms (inputs and
    /// explained results): the registry knows no dependencies of a cell-only result, so one is
    /// never marked affected.
    pub fn term_style(&self, path: &str, inputs: &DesignInputs) -> Option<TermStyle> {
        let kind = self.term_kind(path)?;
        let changed_from_default = match kind {
            TermKind::Input { .. } => inputs.get(path) != self.defaults.get(path),
            _ => false,
        };
        let affected_by_modified_assumption =
            !self.modified_assumptions_upstream(path, inputs).is_empty();
        Some(TermStyle {
            kind,
            changed_from_default,
            affected_by_modified_assumption,
        })
    }

    /// The modified assumptions (A3 rows) that flow into `path`: the tooltip's
    /// "depends on modified assumptions" list. An assumption input itself counts.
    pub fn modified_assumptions_upstream(
        &self,
        path: &str,
        inputs: &DesignInputs,
    ) -> Vec<&'static Assumption> {
        let upstream = self.upstream_inputs.get(path);
        assumptions::modified(inputs)
            .into_iter()
            .filter(|a| {
                a.paths
                    .iter()
                    .any(|p| *p == path || upstream.is_some_and(|u| u.contains(*p)))
            })
            .collect()
    }
}
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 9 passed; 0 failed; 1 ignored` (Task 3's five, the two traceability tests, the styling test and the panel lookups test).

- [ ] **Step 4: Review the cancellations (session model)**

Dispatch the physics reviewer on the **session model** with this prompt: "Check each entry of `CANCELLATIONS` in `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` against the records it names (`src/engine/explain/records/torque.rs`): the assumed calibration factor f_cal,0 cancels from the bench correction f_cal,1 = f_cal,0 · T_meas / (T_2D f_end f_cal,0); the Br coefficient α scales every harmonic amplitude by one Br(T) ratio, which cannot move a peak angle (the pull-out angle, both circuit angles, the prototype's). Confirm each is an algebraic identity, not a missing or spurious dependency. Answer APPROVED or list the entry and the reason." On a finding, stop and escalate: a spurious edge is a record bug.

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task4.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task4.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/registry.rs magcoupling-rs/src/engine/explain/mod.rs magcoupling-rs/tests/explain.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): A3 traceability over the dependency graph, and the styling data

Soundness for every input (nudging it moves no explained result outside its
dependency set, at 51 design points; every input but two inert ones changes
some result there) and sensitivity for the 15 assumption
inputs (every numeric result on the active path moves somewhere above
rounding), five algebraic cancellations listed with reasons. term_style and
modified_assumptions_upstream give the A3 panel its styling data; choices and
corrections_upstream give it a selector's labels and the corrected marker.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 5: The A4 notes module and the release gate

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. The four drafted notes stay drafts here; Task 16 is their physics review.

Teaching notes are static data keyed by id; each lists the equations it explains, and `note_for(path)` finds the note for
the equation the panel has open (each equation has at most one note: the panel shows "the note for the equation currently open",
singular). The six A5 warning rules link to their notes by id (A-1 left the ids as placeholders), and the "start here" order follows
the spec. The accuracy gate: `note_for` returns only notes marked `Reviewed`, so drafts never reach users; a reviewed note names its
review record. This task lands the module with the torque chain's four notes drafted from the M1 derivations they cite and stubs that
fix the other ids; Task 15 drafts the rest and Task 16 is the physics review that signs them off. `release_notes_are_reviewed` is
ignored: it is the M4 hands-on checklist's line (every "start here" note and every A5 note reviewed), red until Task 16.

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs` (`Note`, `Review`, `Diagram`, `NOTES` (four torque notes drafted, twelve stubs), `START_HERE`, `note_for`; unit tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs` (`pub mod notes;`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (the note links test and the ignored release gate)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the explain.rs tests row, the explorer section)

**Interfaces:**
- Consumes: Task 3's registry and scope; `warnings::WARNING_RULES` (A-1: each rule's `note_id`).
- Produces:
  - `notes::Note { id, title, equations: &[&str], sentences: &[&str], watch_out: Option<&str>, diagram: Option<Diagram>, sources: &[&str], review: Review }`, `notes::Review::{Draft, Reviewed { reviewer, date, record }}`, `notes::Diagram::{SquareWaveHarmonics, FluxPathBackIron, TorqueAngle, EndFringing}`;
  - `notes::NOTES`, `notes::START_HERE: &[(&str, &str)]` (note id, the equation it opens), `notes::covers(entry, path) -> bool` (a `#` template covers its members), `notes::note(id)`, `notes::note_for_any_status(path)`, `notes::note_for(path)` (reviewed notes only: the accuracy gate);
  - in `tests/explain.rs`: `notes_link_to_records_and_each_equation_has_at_most_one`, `release_notes_are_reviewed` (ignored: the M4 release gate).

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
//! Addendum A2/A3 explanation layer: the drift guard, the registry's structure, the v1
//! scope and the A3 traceability test.
//!
//! **Drift guard.** Every equation record is evaluated over the engine's own term values and
//! must reproduce the engine's result by the parity rule (1e-9 relative, 1e-12 absolute;
```

with:

```rust
//! Addendum A2/A3 explanation layer: the drift guard, the registry's structure, the v1
//! scope, the teaching-note links and the A3 traceability test.
//!
//! **Drift guard.** Every equation record is evaluated over the engine's own term values and
//! must reproduce the engine's result by the parity rule (1e-9 relative, 1e-12 absolute;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, render};
use magcoupling::engine::library;
use magcoupling::engine::meta::{
    FieldType, InputMeta, InputSet, ResultSet, Value, input_rows, result_rows,
```

with:

```rust
use magcoupling::engine::explain::record::Eval;
use magcoupling::engine::explain::registry::{Design, Registry, TermKind};
use magcoupling::engine::explain::scope::{SCOPE, Status};
use magcoupling::engine::explain::{TermSource, Trace, notes, render};
use magcoupling::engine::library;
use magcoupling::engine::meta::{
    FieldType, InputMeta, InputSet, ResultSet, Value, input_rows, result_rows,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
    }
}

/// The physics reviewer's sheet for a batch: every record's path, cell, symbol, rendered
/// formula and corrections, as a Markdown table on stdout. Run with
/// `cargo test --test explain review_sheet -- --ignored --nocapture`.
```

with:

```rust
    }
}

/// The A4 accuracy gate for release (the M4 hands-on checklist runs it): every note the
/// "start here" order opens and every note an A5 warning links to has passed the physics
/// review. Red until the notes are reviewed; drafts never reach users meanwhile (`note_for`).
#[test]
#[ignore = "the M4 release gate: red until the notes are reviewed"]
fn release_notes_are_reviewed() {
    let mut ids: Vec<&str> = notes::START_HERE.iter().map(|(id, _)| *id).collect();
    ids.extend(
        magcoupling::engine::warnings::WARNING_RULES
            .iter()
            .map(|r| r.note_id),
    );
    let drafts: Vec<&str> = ids
        .into_iter()
        .filter(|id| {
            !matches!(
                notes::note(id).map(|n| n.review),
                Some(notes::Review::Reviewed { .. })
            )
        })
        .collect();
    assert!(drafts.is_empty(), "not yet reviewed: {drafts:?}");
}

/// The physics reviewer's sheet for a batch: every record's path, cell, symbol, rendered
/// formula and corrections, as a Markdown table on stdout. Run with
/// `cargo test --test explain review_sheet -- --ignored --nocapture`.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
        159,
        "decision 31: the chains and the dashboard (report section 7)"
    );
}

// ---------------------------------------------------------------------------------------
```

with:

```rust
        159,
        "decision 31: the chains and the dashboard (report section 7)"
    );
}

#[test]
fn notes_link_to_records_and_each_equation_has_at_most_one() {
    let r = Registry::build();
    let results: BTreeSet<String> = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .map(|x| x.path)
        .collect();
    for n in notes::NOTES {
        for entry in n.equations {
            let members: Vec<&String> =
                results.iter().filter(|p| notes::covers(entry, p)).collect();
            assert!(
                !members.is_empty(),
                "note {}: {entry} names no result",
                n.id
            );
            if n.sentences.is_empty() {
                continue; // a stub fixes an id; its links are checked when it is drafted
            }
            for m in members {
                assert!(
                    r.equation_for(m).is_some(),
                    "note {}: {m} has no record",
                    n.id
                );
            }
        }
    }
    for eq in r.equations() {
        let owners: Vec<&str> = notes::NOTES
            .iter()
            .filter(|n| n.equations.iter().any(|e| notes::covers(e, &eq.target)))
            .map(|n| n.id)
            .collect();
        assert!(owners.len() <= 1, "{}: notes {owners:?}", eq.target);
    }
    for (id, path) in notes::START_HERE {
        assert!(results.contains(*path), "start here {id}: {path}");
        let explained_chain = SCOPE
            .iter()
            .any(|c| c.status == Status::Explained && c.paths.contains(path));
        if explained_chain {
            assert!(
                r.equation_for(path).is_some(),
                "start here {id}: {path} has no record"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "^error" | head -3
```

Expected: ``error[E0432]: unresolved import `magcoupling::engine::explain::notes` `` and ``error: could not compile `magcoupling-rs` (test "explain") due to 1 previous error``.

- [ ] **Step 3: Write the notes module**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

with:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs`, replace:

```rust
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term styles;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
//! - [`scope`]: the v1 scope (decision 31) and which chains are written;
//! - [`render`]: a plain-text rendering (tooltips fallback, export, tests).

pub mod eval;
pub mod markup;
pub mod record;
pub mod records;
pub mod registry;
```

with:

```rust
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term styles;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
//! - [`scope`]: the v1 scope (decision 31) and which chains are written;
//! - [`notes`]: the A4 teaching notes, the "start here" order and the accuracy gate;
//! - [`render`]: a plain-text rendering (tooltips fallback, export, tests).

pub mod eval;
pub mod markup;
pub mod notes;
pub mod record;
pub mod records;
pub mod registry;
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs` with exactly this content:

```rust
//! Addendum A4 teaching notes: static data keyed by note id, linked to the equations each
//! explains and to the A5 warning rules that cite them (`warnings::WarningRule::note_id`).
//!
//! The accuracy gate: a note is drafted from the M1 derivations it cites (`sources`) and
//! shows in the GUI only once a physics reviewer has checked it (`Review::Reviewed`, naming
//! the review record). [`note_for`] returns reviewed notes only; [`note_for_any_status`] is
//! for the review tooling. `tests/explain.rs` checks the links, the 2-6 sentence rule of a
//! reviewed note, and that each equation has at most one note.

/// A small diagram the M4 panel paints beside a note (egui painter, no image files).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diagram {
    /// The magnetization's rectangular wave (blocks and gaps) and its first odd harmonics.
    SquareWaveHarmonics,
    /// Flux paths across the gap with and without back iron.
    FluxPathBackIron,
    /// Torque against electrical angle, the pull-out point marked.
    TorqueAngle,
    /// Field fringing at the magnet ends.
    EndFringing,
}

/// Where a note stands in the accuracy gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Review {
    /// Written, not yet checked: hidden from users.
    Draft,
    /// Checked by the physics reviewer: `record` names the review (report section or commit).
    Reviewed {
        reviewer: &'static str,
        date: &'static str,
        record: &'static str,
    },
}

/// One teaching note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Note {
    pub id: &'static str,
    pub title: &'static str,
    /// The equations it explains: result paths, or a family's target with `#` (all its members).
    pub equations: &'static [&'static str],
    /// 2 to 6 plain sentences at Physics 2 level (empty while a stub).
    pub sentences: &'static [&'static str],
    pub watch_out: Option<&'static str>,
    pub diagram: Option<Diagram>,
    /// The M1 derivations and report rows it is drafted from.
    pub sources: &'static [&'static str],
    pub review: Review,
}

const fn stub(id: &'static str, title: &'static str, equations: &'static [&'static str]) -> Note {
    Note {
        id,
        title,
        equations,
        sentences: &[],
        watch_out: None,
        diagram: None,
        sources: &[],
        review: Review::Draft,
    }
}

/// Every note. The torque chain's four are drafted (plan A-3 tracer); the rest are stubs
/// that fix the ids the "start here" order and the A5 warnings link to.
pub const NOTES: &[Note] = &[
    Note {
        id: "harmonics",
        title: "Harmonic decomposition",
        equations: &[
            "model.b_i#",
            "model.b_o#",
            "model.amp#_Pa",
            "model.tau_Pa",
            "calibration.amp#_Pa",
        ],
        sentences: &[
            "Each ring's magnetization alternates north and south around the circle, so along the gap it is a rectangular wave, blocks separated by gaps, not a smooth sine.",
            "Such a wave is a sum of sine waves at odd multiples of its basic frequency: harmonic n has n times as many wavelengths around the ring and an amplitude of 4/(nπ) times the remanence, scaled by sin(nπλ/2), where λ is the fraction of each pole the magnet fills; the gaps (λ < 1) are why that factor appears.",
            "Harmonic n of one ring pulls only on harmonic n of the other, so the total shear stress is a sum with one term per harmonic.",
            "Higher harmonics have shorter wavelengths, and their fields fade across the gap much faster, which is why the workbook keeps only 1, 3 and 5.",
        ],
        watch_out: Some(
            "A fill of exactly 0.4 makes the fifth harmonic vanish, because sin(5π·0.4/2) = sin(π) = 0: that is geometry, not an error.",
        ),
        diagram: Some(Diagram::SquareWaveHarmonics),
        sources: &[
            "M1 audit M3, M4, M7 (the planar harmonic model against exact 2D sections)",
            "Addendum A decision 29 (odd harmonics up to 11)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "back_iron_factor",
        title: "Back iron: the sinh factor against free space",
        equations: &["model.s#_iron", "model.s#_free", "model.k#"],
        sentences: &[
            "The geometry factor S_n says how much of harmonic n's field from one ring reaches through the other ring.",
            "With steel behind both rings, the field lines cross the gap and close through the steel instead of spreading into the air behind the magnets, and the factor takes the sinh form.",
            "The steel makes the field meet its surface at right angles, so the growing and decaying exponentials across the gap combine into sinh; in free space only the decaying one remains.",
            "Without back iron each ring's field leaks out behind it as well as across the gap, so the free-space factor, with its decay e^(−kg), is smaller.",
            "Both fall off with k·g, the gap measured against the harmonic's wavelength, so a small gap matters most for the high harmonics.",
        ],
        watch_out: None,
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "M1 audit M3 (planar factor against the cylindrical one)",
            "M1 audit M5 (steel at the magnet backs)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "pullout_angle",
        title: "Pull-out torque against rotation angle",
        equations: &[
            "model.tau#_Pa",
            "model.pullout_angle_rad",
            "model.iron_circuit_angle_rad",
            "model.free_circuit_angle_rad",
            "model.pullout_Nm",
            "calibration.pullout_angle_rad",
            "calibration.tau#_Pa",
        ],
        sentences: &[
            "Twist one ring against the other and the torque rises from zero, peaks and falls again: the peak is the pull-out torque, the most the coupling carries before it slips.",
            "Harmonic n contributes A_n sin(nφ), where φ is the electrical angle, 90° at half a pole pitch.",
            "With the fundamental alone the peak is at φ = 90°; a strong third harmonic can move it, so the calculator takes the angle where the summed curve is highest (correction E7).",
        ],
        watch_out: None,
        diagram: Some(Diagram::TorqueAngle),
        sources: &[
            "M1 audit E7 (T3-RC2)",
            "Addendum A decision 29 (one peak search for any harmonic set)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "end_effect",
        title: "End effect",
        equations: &["model.f_end", "calibration.f_end"],
        sentences: &[
            "The 2D model treats the rings as infinitely long, but real magnets end, and near each end the field fringes outward and carries less torque.",
            "The factor f_end = 1 − c_end·τ_p/L takes off a total of about c_end pole pitches, half at each end, from the active length L.",
            "It is empirical: within about 1 % of 3D for magnets longer than about 3 mm, but it turns negative for very short magnets, which the end-effect check flags.",
        ],
        watch_out: Some(
            "Below L = c_end·τ_p the factor is zero or negative and every torque computed from it is meaningless.",
        ),
        diagram: Some(Diagram::EndFringing),
        sources: &["M1 audit M8 and M9 (T3-RC1, T3-RC3)"],
        review: Review::Draft,
    },
    stub(
        "br_temperature",
        "Remanence against temperature, and torque ∝ Br²",
        &[
            "model.br_inner_T_op",
            "model.br_outer_T_op",
            "model.pullout_20C_Nm",
        ],
    ),
    stub(
        "demagnetization",
        "Demagnetization: knee, permeance and the onset temperatures",
        &[],
    ),
    stub("slip_heating", "Eddy-current slip loss and skin depth", &[]),
    stub("thermal_time_constant", "The thermal time constant", &[]),
    stub("clamp_preload", "Clamp preload and friction", &[]),
    stub(
        "ferrite_cold_demag",
        "Ferrite: the demagnetization risk is at cold",
        &[],
    ),
    stub(
        "a5.non_ferromagnetic_back_iron",
        "Non-ferromagnetic back iron",
        &["materials.circuit_backiron"],
    ),
    stub(
        "a5.ferromagnetic_sleeve_or_liner",
        "Ferromagnetic sleeve or liner",
        &[],
    ),
    stub(
        "a5.high_conductivity_sleeve_or_liner",
        "High-conductivity sleeve or liner",
        &[],
    ),
    stub("a5.low_saturation", "Low saturation", &[]),
    stub(
        "a5.uncoated_low_alloy_steel",
        "Uncoated low-alloy steel",
        &[],
    ),
    stub(
        "a5.cte_mismatch_with_magnets",
        "Expansion mismatch with the magnets",
        &[],
    ),
];

/// The suggested reading order (spec A4 "Start here": torque chain → back iron →
/// temperature → demagnetization → slip heating → clamps): (note id, the equation it opens).
pub const START_HERE: &[(&str, &str)] = &[
    ("harmonics", "model.tau_Pa"),
    ("back_iron_factor", "model.s1_iron"),
    ("br_temperature", "model.br_inner_T_op"),
    ("demagnetization", "temperature.demag.onset_skipping_C"),
    ("slip_heating", "temperature.slip_loss.total_W"),
    ("clamp_preload", "clamps.capacity_Nm"),
];

/// Whether a note's equation entry (a path or a `#` template) covers `path`.
pub fn covers(entry: &str, path: &str) -> bool {
    match entry.split_once('#') {
        None => entry == path,
        Some((prefix, suffix)) => path
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix(suffix))
            .is_some_and(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())),
    }
}

/// The note with this id.
pub fn note(id: &str) -> Option<&'static Note> {
    NOTES.iter().find(|n| n.id == id)
}

/// The note explaining `path`, whatever its review status (review tooling, tests).
pub fn note_for_any_status(path: &str) -> Option<&'static Note> {
    NOTES
        .iter()
        .find(|n| n.equations.iter().any(|e| covers(e, path)))
}

/// The note the equation panel shows for `path`: reviewed notes only (the accuracy gate).
pub fn note_for(path: &str) -> Option<&'static Note> {
    note_for_any_status(path).filter(|n| matches!(n.review, Review::Reviewed { .. }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::warnings::WARNING_RULES;

    #[test]
    fn templates_cover_their_members_only() {
        assert!(covers("model.tau#_Pa", "model.tau11_Pa"));
        assert!(covers("model.tau#_Pa", "model.tau1_Pa"));
        assert!(!covers("model.tau#_Pa", "model.tau_Pa"));
        assert!(!covers("model.tau#_Pa", "model.taux_Pa"));
        assert!(covers("model.f_end", "model.f_end"));
    }

    #[test]
    fn ids_are_unique_and_every_link_resolves() {
        for (i, n) in NOTES.iter().enumerate() {
            assert!(NOTES[..i].iter().all(|m| m.id != n.id), "{} twice", n.id);
        }
        for rule in &WARNING_RULES {
            assert!(
                note(rule.note_id).is_some(),
                "warning {} links to missing note {}",
                rule.id,
                rule.note_id
            );
        }
        for (id, _) in START_HERE {
            assert!(note(id).is_some(), "start-here note {id}");
        }
    }

    #[test]
    fn a_draft_is_hidden_and_a_reviewed_note_is_complete() {
        assert!(note_for("model.f_end").is_none(), "drafts stay hidden");
        assert_eq!(
            note_for_any_status("model.f_end").map(|n| n.id),
            Some("end_effect")
        );
        for n in NOTES {
            if n.sentences.is_empty() {
                assert_eq!(
                    n.review,
                    Review::Draft,
                    "{}: a stub cannot be reviewed",
                    n.id
                );
                continue;
            }
            assert!(
                (2..=6).contains(&n.sentences.len()),
                "{}: 2 to 6 sentences",
                n.id
            );
            assert!(
                !n.sources.is_empty(),
                "{}: a drafted note cites its M1 sources",
                n.id
            );
        }
    }
}
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 10 passed; 0 failed; 2 ignored` (`release_notes_are_reviewed` and `review_sheet` are ignored).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib notes 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed` (the notes module's unit tests).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task5.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task5.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task5.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/src/engine/explain/mod.rs magcoupling-rs/tests/explain.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): the A4 teaching-notes module and its release gate

Notes are static data linked to the equations they explain and to the A5
warning rules; note_for shows only reviewed notes (the accuracy gate). The
torque chain's four notes are drafted from the M1 rows they cite; the other
ids are stubs. The ignored release gate is the M4 checklist's line.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 6: Markup for the remaining chains: inf, nan, rounding to a step, formatted text

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model.

The remaining chains state a few things the tracer's markup could not: an onset that is never reached (`inf`, hard ferrite
heated: E20's positive-beta side), a torque at a limit that does not exist (`nan`, the engine's NaN there), a screw count
(`ceilto`, `floorto`: the clamp table) and two verdicts that format a number into their text (`materials.cup_wall_check` "Too
thin: raise ... to at least 2.0 mm" and `clamps.recommended` "ISO 4762 M4 x 12, class 12.9"). Stating them in the markup keeps
every displayed formula the evaluated one, with no Rust closure: `Eval::Custom` stays an escape hatch that the registry test pins at
zero uses.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs` (keywords `inf`, `nan`; functions `ceilto`, `floorto`, `fmt`, `fmtnum`, `concat` (arity table); docs; a test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs` (their evaluation and trace classification; a test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/render.rs` (their plain-text rendering)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the explorer section)

**Interfaces:**
- Consumes: Task 2's markup and evaluator; `compat::{ceiling, floor_, fmt_fixed, fmt_num}`.
- Produces (the batches use them):
  - `inf` (+∞, drawn ∞) and `nan` (an undefined value, drawn "undefined") as number literals;
  - `ceilto(a, s)`, `floorto(a, s)`: Excel's CEILING and FLOOR with the engine's 1e-12 guard (`compat::ceiling`, `compat::floor_`), both arguments condition terms (piecewise constant);
  - `fmt(a, d)` (`compat::fmt_fixed`, d a whole number 0 to 12), `fmtnum(a)` (`compat::fmt_num`): text, a value term; `concat(t, ...)`: two or more texts joined (a number must be formatted first);
  - `Func::{CeilTo, FloorTo, Fmt, FmtNum, Concat}` and a private arity table replacing `Func::variadic`.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs`, replace:

```rust
    }

    #[test]
    fn unit_scale_applies_to_numbers() {
        let s = src(&[("a.len_mm", Value::Num(12.0))]);
        let mut f = parse("{a.len_mm|m}", None).unwrap();
```

with:

```rust
    }

    #[test]
    fn rounding_to_a_step_and_formatting_follow_the_engine() {
        let s = src(&[
            ("a.t", Value::Num(1.90415278222222)),
            ("a.l", Value::Num(12.0)),
            ("a.w", Value::Num(1.9)),
        ]);
        let (v, t) = eval("ceilto({a.t}, 0.1)", &s);
        assert_eq!(v, Value::Num(ceiling(1.90415278222222, 0.1)));
        assert!(
            t.value_terms.is_empty() && t.condition_terms.contains("a.t"),
            "piecewise constant"
        );
        assert_eq!(
            eval("ceilto({a.w}, 0.1)", &s).0,
            Value::Num(19.0 * 0.1),
            "the 1e-12 guard"
        );
        assert_eq!(eval("floorto(7.5, 2)", &s).0, Value::Num(6.0));
        let (v, t) = eval(
            r#"concat("at least ", fmt(ceilto({a.t}, 0.1), 1), " mm")"#,
            &s,
        );
        assert_eq!(v, Value::Text("at least 2.0 mm".into()));
        assert!(t.condition_terms.contains("a.t"));
        let (v, t) = eval(r#"concat("M3 x ", fmtnum({a.l}))"#, &s);
        assert_eq!(v, Value::Text("M3 x 12".into()));
        assert!(
            t.value_terms.contains("a.l"),
            "a formatted number moves its text"
        );
        assert_eq!(eval("inf", &s).0, Value::Num(f64::INFINITY));
        assert!(matches!(eval("nan", &s).0, Value::Num(x) if x.is_nan()));
        for bad in [r#"concat("a", 1)"#, "fmt({a.t}, 0.5)", "fmt({a.t}, 13)"] {
            assert!(
                evaluate(&parse(bad, None).unwrap(), &s, None).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn unit_scale_applies_to_numbers() {
        let s = src(&[("a.len_mm", Value::Num(12.0))]);
        let mut f = parse("{a.len_mm|m}", None).unwrap();
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
    }

    #[test]
    fn symbols_split_into_base_and_scripts() {
        assert_eq!(
            Symbol::parse("S_{3}^{iron}").unwrap(),
```

with:

```rust
    }

    #[test]
    fn the_text_and_rounding_functions_check_their_arity() {
        assert!(parse("ceilto({a.x}, 0.1) + floorto({a.x}, 1)", None).is_ok());
        assert!(parse("ceilto({a.x})", None).is_err(), "ceilto takes 2");
        assert!(parse("fmt({a.x}, 1, 2)", None).is_err(), "fmt takes 2");
        assert!(
            parse(r#"concat("a")"#, None).is_err(),
            "concat takes 2 or more"
        );
        assert!(parse(r#"concat("a", fmt({a.x}, 1), fmtnum({a.y}), " mm")"#, None).is_ok());
        let f = parse("inf", None).unwrap();
        assert!(
            matches!(f.body, Expr::Num { value, ref text } if value == f64::INFINITY && text == "∞")
        );
        let f = parse("nan", None).unwrap();
        assert!(
            matches!(f.body, Expr::Num { value, ref text } if value.is_nan() && text == "undefined")
        );
        assert!(
            parse("[inf] where [inf] = 1", None).is_ok(),
            "a local's name is a symbol, not a keyword"
        );
    }

    #[test]
    fn symbols_split_into_base_and_scripts() {
        assert_eq!(
            Symbol::parse("S_{3}^{iron}").unwrap(),
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib explain 2>&1 | grep -E "^error" | head -3
```

Expected: ``error[E0425]: cannot find function `ceiling` in this scope`` and ``error: could not compile `magcoupling-rs` (lib test) due to 1 previous error`` (the new evaluator test names the new imports).

- [ ] **Step 3: Extend the markup, the evaluator and the rendering**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

with:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Formatted verdicts are stated in the markup too (`concat`, `fmt`, `fmtnum`), with `ceilto` and `floorto` for Excel's CEILING and FLOOR and the literals `inf` and `nan`, so no record needs a Rust closure. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs`, replace:

```rust

use super::markup::{BinOp, Cond, Expr, Formula, Func, IndexSet, RelOp};
use super::tables;
use crate::engine::compat::{py_max, py_min};
use crate::engine::meta::Value;
use crate::engine::model::{HALF_PITCH_RAD, ODD_HARMONICS, harmonic_count, peak_off_half_pitch};
```

with:

```rust

use super::markup::{BinOp, Cond, Expr, Formula, Func, IndexSet, RelOp};
use super::tables;
use crate::engine::compat::{ceiling, floor_, fmt_fixed, fmt_num, py_max, py_min};
use crate::engine::meta::Value;
use crate::engine::model::{HALF_PITCH_RAD, ODD_HARMONICS, harmonic_count, peak_off_half_pitch};
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs`, replace:

```rust
/// (arithmetically, through the chosen `cases` arm, through the winner of a `min`/`max`),
/// so nudging it moves the result. It is a **condition term** when it was read only to
/// decide something piecewise-constant: a `cases` condition, the loser of a `min`/`max`, the
/// argument of `ceil`/`floor`, the selector of a Σ's index set.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub value_terms: BTreeSet<String>,
```

with:

```rust
/// (arithmetically, through the chosen `cases` arm, through the winner of a `min`/`max`),
/// so nudging it moves the result. It is a **condition term** when it was read only to
/// decide something piecewise-constant: a `cases` condition, the loser of a `min`/`max`, the
/// arguments of `ceil`/`floor`/`ceilto`/`floorto`, the selector of a Σ's index set. A
/// formatted number (`fmt`, `fmtnum`) is a value term: its text moves with it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub value_terms: BTreeSet<String>,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs`, replace:

```rust
            t.absorb(bt, false);
            return Ok(V::Num(x));
        }
        let mut at = Trace::default();
        let x = self.num(&args[0], n, &mut at)?;
        let piecewise = matches!(func, Func::Ceil | Func::Floor);
```

with:

```rust
            t.absorb(bt, false);
            return Ok(V::Num(x));
        }
        match func {
            Func::Concat => {
                // Texts only: a number must be formatted explicitly (fmt, fmtnum).
                let mut out = String::new();
                for a in args {
                    match self.expr(a, n, t)? {
                        V::Text(s) => out.push_str(&s),
                        other => {
                            return Err(EvalError(format!(
                                "concat: expected a text, got {other:?}"
                            )));
                        }
                    }
                }
                return Ok(V::Text(out));
            }
            Func::CeilTo | Func::FloorTo => {
                // Rounding to a step is piecewise constant in both arguments.
                let mut at = Trace::default();
                let (x, step) = (
                    self.num(&args[0], n, &mut at)?,
                    self.num(&args[1], n, &mut at)?,
                );
                t.absorb(at, true);
                return Ok(V::Num(if func == Func::CeilTo {
                    ceiling(x, step)
                } else {
                    floor_(x, step)
                }));
            }
            Func::Fmt => {
                let x = self.num(&args[0], n, t)?;
                let mut dt = Trace::default();
                let d = self.num(&args[1], n, &mut dt)?;
                t.absorb(dt, true);
                if d.fract() != 0.0 || !(0.0..=12.0).contains(&d) {
                    return Err(EvalError(format!(
                        "fmt: {d} decimals (a whole number 0 to 12)"
                    )));
                }
                return Ok(V::Text(fmt_fixed(x, d as usize)));
            }
            Func::FmtNum => return Ok(V::Text(fmt_num(self.num(&args[0], n, t)?))),
            _ => {}
        }
        let mut at = Trace::default();
        let x = self.num(&args[0], n, &mut at)?;
        let piecewise = matches!(func, Func::Ceil | Func::Floor);
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/eval.rs`, replace:

```rust
            Func::Abs => x.abs(),
            Func::Ceil => x.ceil(),
            Func::Floor => x.floor(),
            Func::Min | Func::Max => unreachable!("handled above"),
        }))
    }
```

with:

```rust
            Func::Abs => x.abs(),
            Func::Ceil => x.ceil(),
            Func::Floor => x.floor(),
            Func::Min
            | Func::Max
            | Func::CeilTo
            | Func::FloorTo
            | Func::Fmt
            | Func::FmtNum
            | Func::Concat => {
                unreachable!("handled above")
            }
        }))
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
//! product  = unary , { ( "*" | "·" | "/" ) , unary } ;
//! unary    = "-" , unary | power ;
//! power    = atom , [ "^" , unary ] ;             (* right-associative *)
//! atom     = number | text | term | local | "n" | "pi" | "π" | "none"
//!          | call | cases | sum | peak | table | "(" , expr , ")" ;
//! number   = digit , { digit } , [ "." , digit , { digit } ] , [ ( "e" | "E" ) , [ "-" | "+" ] , digit , { digit } ] ;
//! text     = '"' , { char - '"' } , '"' ;
//! term     = "{" , path , [ "|" , unit ] , "}" ;   (* an input or result path; "#" = the index *)
//! call     = func , "(" , expr , { "," , expr } , ")" ;
//! func     = "frac" | "sqrt" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "exp"
//!          | "ln" | "abs" | "min" | "max" | "ceil" | "floor" ;
//! sum      = "sum" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! peak     = "peak" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! table    = "table" , "(" , text , "," , expr , "," , text , ")" ;
```

with:

```rust
//! product  = unary , { ( "*" | "·" | "/" ) , unary } ;
//! unary    = "-" , unary | power ;
//! power    = atom , [ "^" , unary ] ;             (* right-associative *)
//! atom     = number | text | term | local | "n" | "pi" | "π" | "none" | "inf" | "nan"
//!          | call | cases | sum | peak | table | "(" , expr , ")" ;
//! number   = digit , { digit } , [ "." , digit , { digit } ] , [ ( "e" | "E" ) , [ "-" | "+" ] , digit , { digit } ] ;
//! text     = '"' , { char - '"' } , '"' ;
//! term     = "{" , path , [ "|" , unit ] , "}" ;   (* an input or result path; "#" = the index *)
//! call     = func , "(" , expr , { "," , expr } , ")" ;
//! func     = "frac" | "sqrt" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" | "exp"
//!          | "ln" | "abs" | "min" | "max" | "ceil" | "floor" | "ceilto" | "floorto"
//!          | "fmt" | "fmtnum" | "concat" ;
//! sum      = "sum" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! peak     = "peak" , "(" , "n" , "in" , "H" , ":" , expr , ")" ;
//! table    = "table" , "(" , text , "," , expr , "," , text , ")" ;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
//! | `sqrt(a)` | √a | a radical |
//! | `exp(a)` | e to the a | `e` with a as a superscript |
//! | `abs(a)`, `ceil(a)`, `floor(a)` | \|a\|, ⌈a⌉, ⌊a⌋ | those brackets |
//! | `sin(a)` .. `ln(a)` | the function | upright name, argument in parentheses |
//! | `min(a, b, ..)`, `max(..)` | Python's `min`/`max`, folded left to right (`compat::py_min`) | upright name |
//! | `(a)` | a | parentheses (always shown) |
```

with:

```rust
//! | `sqrt(a)` | √a | a radical |
//! | `exp(a)` | e to the a | `e` with a as a superscript |
//! | `abs(a)`, `ceil(a)`, `floor(a)` | \|a\|, ⌈a⌉, ⌊a⌋ | those brackets |
//! | `ceilto(a, s)`, `floorto(a, s)` | a rounded up (down) to a multiple of s, as Excel's CEILING (FLOOR) with the engine's 1e-12 guard (`compat::ceiling`, `compat::floor_`) | ⌈a⌉ with s beneath (⌊a⌋ likewise) |
//! | `fmt(a, d)` | the text of a with d decimals (`compat::fmt_fixed`, Python's `f"{a:.{d}f}"`); d a whole number 0 to 12 | a, with d decimals |
//! | `fmtnum(a)` | the text of a as the workbook prints a number (`compat::fmt_num`: a whole number without ".0") | a |
//! | `concat(t, ...)` | the texts joined (every argument a text) | the texts side by side |
//! | `inf` | +∞ (an onset the knee is never reached at) | ∞ |
//! | `nan` | not a number: a quantity the engine leaves undefined (a torque at a limit that does not exist) | "undefined" |
//! | `sin(a)` .. `ln(a)` | the function | upright name, argument in parentheses |
//! | `min(a, b, ..)`, `max(..)` | Python's `min`/`max`, folded left to right (`compat::py_min`) | upright name |
//! | `(a)` | a | parentheses (always shown) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
    Max,
    Ceil,
    Floor,
}

impl Func {
```

with:

```rust
    Max,
    Ceil,
    Floor,
    CeilTo,
    FloorTo,
    Fmt,
    FmtNum,
    Concat,
}

impl Func {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
            "max" => Func::Max,
            "ceil" => Func::Ceil,
            "floor" => Func::Floor,
            _ => return None,
        })
    }
```

with:

```rust
            "max" => Func::Max,
            "ceil" => Func::Ceil,
            "floor" => Func::Floor,
            "ceilto" => Func::CeilTo,
            "floorto" => Func::FloorTo,
            "fmt" => Func::Fmt,
            "fmtnum" => Func::FmtNum,
            "concat" => Func::Concat,
            _ => return None,
        })
    }
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
            Func::Max => "max",
            Func::Ceil => "ceil",
            Func::Floor => "floor",
        }
    }

    /// Whether the function takes any number (at least two) of arguments.
    const fn variadic(self) -> bool {
        matches!(self, Func::Min | Func::Max)
    }
}
```

with:

```rust
            Func::Max => "max",
            Func::Ceil => "ceil",
            Func::Floor => "floor",
            Func::CeilTo => "ceilto",
            Func::FloorTo => "floorto",
            Func::Fmt => "fmt",
            Func::FmtNum => "fmtnum",
            Func::Concat => "concat",
        }
    }

    /// How many arguments the function takes: `(least, most)`, `None` for no upper bound.
    const fn arity(self) -> (usize, Option<usize>) {
        match self {
            Func::Min | Func::Max | Func::Concat => (2, None),
            Func::CeilTo | Func::FloorTo | Func::Fmt => (2, Some(2)),
            _ => (1, Some(1)),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust

const KEYWORDS: &[&str] = &[
    "cases", "else", "and", "or", "where", "sum", "peak", "table", "in", "n", "pi", "none", "H",
    "frac",
];

impl Parser {
```

with:

```rust

const KEYWORDS: &[&str] = &[
    "cases", "else", "and", "or", "where", "sum", "peak", "table", "in", "n", "pi", "none", "H",
    "frac", "inf", "nan",
];

impl Parser {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
                match word.as_str() {
                    "pi" => Ok(Expr::Pi),
                    "none" => Ok(Expr::NoneLit),
                    "cases" => self.cases(),
                    "n" => match self.index {
                        Some(n) => Ok(Expr::Num {
```

with:

```rust
                match word.as_str() {
                    "pi" => Ok(Expr::Pi),
                    "none" => Ok(Expr::NoneLit),
                    "inf" => Ok(Expr::Num {
                        value: f64::INFINITY,
                        text: "∞".into(),
                    }),
                    "nan" => Ok(Expr::Num {
                        value: f64::NAN,
                        text: "undefined".into(),
                    }),
                    "cases" => self.cases(),
                    "n" => match self.index {
                        Some(n) => Ok(Expr::Num {
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/markup.rs`, replace:

```rust
                            args.push(self.expr()?);
                        }
                        self.expect(')')?;
                        let ok = if func.variadic() {
                            args.len() >= 2
                        } else {
                            args.len() == 1
                        };
                        if !ok {
                            return Err(self.error_at(
                                at,
                                format!(
                                    "{name} takes {} argument(s), got {}",
                                    if func.variadic() {
                                        "two or more"
                                    } else {
                                        "one"
                                    },
                                    args.len()
                                ),
                            ));
                        }
                        Ok(Expr::Call(func, args))
```

with:

```rust
                            args.push(self.expr()?);
                        }
                        self.expect(')')?;
                        let (least, most) = func.arity();
                        if args.len() < least || most.is_some_and(|m| args.len() > m) {
                            let wanted = match (least, most) {
                                (1, Some(1)) => "one".to_owned(),
                                (l, Some(m)) if l == m => format!("{l}"),
                                (l, _) => format!("{l} or more"),
                            };
                            return Err(self.error_at(
                                at,
                                format!("{name} takes {wanted} argument(s), got {}", args.len()),
                            ));
                        }
                        Ok(Expr::Call(func, args))
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/render.rs`, replace:

```rust
                    Func::Abs => format!("|{}|", a[0]),
                    Func::Ceil => format!("⌈{}⌉", a[0]),
                    Func::Floor => format!("⌊{}⌋", a[0]),
                    _ => format!("{}({})", f.name(), a.join(", ")),
                }
            }
```

with:

```rust
                    Func::Abs => format!("|{}|", a[0]),
                    Func::Ceil => format!("⌈{}⌉", a[0]),
                    Func::Floor => format!("⌊{}⌋", a[0]),
                    Func::CeilTo => format!("⌈{}⌉_{{{}}}", a[0], a[1]),
                    Func::FloorTo => format!("⌊{}⌋_{{{}}}", a[0], a[1]),
                    Func::Fmt => format!("{} (to {} decimals)", a[0], a[1]),
                    Func::FmtNum => a[0].clone(),
                    Func::Concat => a.join(" ⧺ "),
                    _ => format!("{}({})", f.name(), a.join(", ")),
                }
            }
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib explain 2>&1 | grep "test result"
```

Expected: `test result: ok. 28 passed` (the explain module's unit tests, this task's two among them).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep "test result"
```

Expected: `test result: ok. 10 passed; 0 failed; 2 ignored`: every record still reproduces the engine.

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task6.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/markup.rs magcoupling-rs/src/engine/explain/eval.rs magcoupling-rs/src/engine/explain/render.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): markup literals inf and nan, rounding to a step, formatted text

ceilto and floorto (Excel CEILING and FLOOR with the engine's guard), fmt,
fmtnum and concat, so the clamp table and the two formatted verdicts are
stated and proven in the markup, with no Rust closure.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 7: Engine exposures and tables for the remaining chains

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5): it changes what the engine computes for Rust-only results (the outer ring's demagnetization block is now computed with E20 off too) and edits an E20 test. Physics reviewer: Addendum A decision 19 (E20) and A-1 decision A13.

Same rule as Task 1: a value a formula needs that the engine computed but did not expose becomes a Rust-only result, a pure
read. E20 checks both rings and shows the weaker one's block; the records need each ring's own Hcj, beta, magnet limit and cold limit
to show which ring governs and why (the per-ring results read each ring's `RingDemag`). Without E20 the workbook checks the inner ring
only; the outer ring's block is now computed all the same (governing nothing, decision G4), because a NaN placeholder would make
the engine's whole-struct equality assertions fail (NaN is not equal to NaN). The E20 test `e20_checks_both_rings_each_side_from_the_weaker`
gains the outer ring's four fields in its expected struct, and its "workbook reads the inner ring only" assertion compares through a
`workbook_cells` helper that blanks only those four Rust-only fields, so every workbook cell is still compared. The materials in
effect (A5: the steel, the hub, cup and boss without back iron, the sleeve, the cap) are read from `resolve`'s `PartProperties`. The
tables give the records every library value they read, corrections on (E19 for the SuperMagnetMan arcs' rating and M5045's grade).
Every field is added here, once, though some are first read by a later batch (DRY: one tables task).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/temperature.rs` (8 Rust-only per-ring results (E20); the outer ring's block always computed; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/materials.rs` (9 Rust-only materials-in-effect results; a test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs` (magnets tmax and grade (E19), grades id, Hcj, beta, Tmax and density, the part materials, aluminium alloys, adhesives, screw sizes; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs` (the tables line of the module docs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the temperature, materials and explain layout rows)

**Interfaces:**
- Consumes: `temperature::ring_demag` and its `RingDemag` (A-1), `material_library::PartProperties` from `resolve` (A-1), `library::{tmax_C, grade_id}` (E19), `grades::Grade`, `temperature::ADHESIVES`, `clamps::SCREW_SIZES`, `materials::{AL7075, AL6061}`.
- Produces:
  - Rust-only `temperature.demag.{inner,outer}_hcj20_kA_m`, `{inner,outer}_beta_per_C`, `{inner,outer}_magnet_limit_C` (f64) and `{inner,outer}_cold_limit_C` (`NumOrText`, "n/a" unless beta > 0): each ring's own E20 block;
  - Rust-only `materials.steel_density_g_mm3`, `body_sigma_S_m`, `body_density_g_mm3`, `body_c_J_kgK`, `sleeve_sigma_S_m`, `sleeve_density_g_mm3`, `sleeve_c_J_kgK`, `cap_density_g_mm3`, `cap_c_J_kgK`;
  - table fields: `magnets.tmax_C`, `magnets.grade` (text); `grades.id` (text), `hcj20_kA_m`, `beta_hcj_per_C`, `tmax_C`, `density_g_mm3`; `back_iron.sigma_S_m`, `density_g_mm3`, `cp_J_kgK`, `design_flux_density_T` (none where the workbook gives none); `sleeve_liner.*` and `cap_housing.*` (sigma, density, cp); `aluminium.{conductivity_S_m, shear_MPa, head_pressure_limit_MPa, key_bearing_allow_MPa}` keyed by alloy name; `adhesives.{design_limit_C, cure_C, lap_shear_MPa}` keyed by code; `screw_sizes.{name, d_mm, As_mm2, hole_mm, head_mm, hex_mm}` keyed by row (0 = M2.5).

- [ ] **Step 1: Write the failing tests**

`each_ring_shows_its_own_block` (temperature.rs), `the_materials_in_effect_are_the_picks_values` (materials.rs) and `corrected_library_fields_read_as_the_engine_uses_them` (tables.rs).

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
            .collect();
        assert_eq!(non.len(), 2, "304 and 6061 (spec A5)");
    }
}
```

with:

```rust
            .collect();
        assert_eq!(non.len(), 2, "304 and 6061 (spec A5)");
    }

    #[test]
    fn corrected_library_fields_read_as_the_engine_uses_them() {
        // E19: M5045's vendor grid (N50) and every SuperMagnetMan arc's 60 °C.
        let m = |part: &str, f: &str| lookup("magnets", &Value::Text(part.into()), f).unwrap();
        assert_eq!(m("M5045", "grade"), Some(Value::Text("N50".into())));
        assert_eq!(m("M5044", "tmax_C"), Some(Value::Num(60.0)));
        assert_eq!(m("B842SH", "br_T"), Some(Value::Num(1.30)), "E3");
        let g = |f: &str| lookup("grades", &Value::Text("Y30".into()), f).unwrap();
        assert_eq!(
            g("beta_hcj_per_C"),
            Some(Value::Num(0.0035)),
            "ferrite: positive beta"
        );
        assert_eq!(
            lookup("screw_sizes", &Value::Num(2.0), "name").unwrap(),
            Some(Value::Text("M4".into()))
        );
        assert_eq!(
            lookup("adhesives", &Value::Int(2), "cure_C").unwrap(),
            Some(Value::Num(120.0))
        );
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
    }

    #[test]
    fn proof_follows_the_class_code_and_is_nan_outside_it() {
        let s = ScrewClasses::default();
        assert_eq!([s.proof(1), s.proof(2), s.proof(3)], [970.0, 830.0, 450.0]);
```

with:

```rust
    }

    #[test]
    fn the_materials_in_effect_are_the_picks_values() {
        // Plan A-3: the Rust-only materials in effect read what `resolve` picked. At the
        // defaults they are the inputs; a non-ferromagnetic back iron (6061) becomes the hub,
        // cup and boss material and leaves the steel at the inputs; the sleeve and cap picks
        // supply their library values.
        use crate::engine::material_library::material;
        let md = MetalDesignInputs::default();
        let with = |back_iron: i64, sleeve_liner: i64, cap_housing: i64| {
            let mut mat = MaterialsInputs::default();
            mat.parts.back_iron = back_iron;
            mat.parts.sleeve_liner = sleeve_liner;
            mat.parts.cap_housing = cap_housing;
            let p = resolve(
                &mat.parts,
                &mat.steel,
                1,
                &md,
                &SlipLossInputs::default(),
                &ThermalInputs::default(),
            );
            compute(&mat, 1.9, 1.8, p.backiron, &p, Deviations::NONE)
        };
        let r = with(1, 1, 1);
        assert_eq!(
            (
                r.steel_density_g_mm3,
                r.body_sigma_S_m,
                r.body_density_g_mm3,
                r.sleeve_density_g_mm3,
                r.cap_density_g_mm3
            ),
            (
                md.steel_density_g_mm3,
                AL6061.conductivity_S_m,
                md.al_density_g_mm3,
                md.sleeve_density_g_mm3,
                md.al_density_g_mm3
            )
        );
        let al = material("6061_T6").unwrap().engine;
        let r = with(8, 1, 1);
        assert_eq!(
            (r.body_sigma_S_m, r.body_density_g_mm3, r.body_c_J_kgK),
            (al.sigma_S_m, al.density_g_mm3, al.cp_J_kgK)
        );
        assert_eq!(
            r.steel_density_g_mm3, md.steel_density_g_mm3,
            "a non-ferromagnetic pick leaves the steel at the inputs"
        );
        let (ti, pom) = (
            material("Ti6Al4V_annealed").unwrap().engine,
            material("POM_H_acetal").unwrap().engine,
        );
        let r = with(1, 2, 3);
        assert_eq!(
            (r.sleeve_sigma_S_m, r.sleeve_density_g_mm3, r.sleeve_c_J_kgK),
            (ti.sigma_S_m, ti.density_g_mm3, ti.cp_J_kgK)
        );
        assert_eq!(
            (r.cap_density_g_mm3, r.cap_c_J_kgK),
            (pom.density_g_mm3, pom.cp_J_kgK)
        );
    }

    #[test]
    fn proof_follows_the_class_code_and_is_nan_outside_it() {
        let s = ScrewClasses::default();
        assert_eq!([s.proof(1), s.proof(2), s.proof(3)], [970.0, 830.0, 450.0]);
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
    }

    #[test]
    fn each_ring_is_checked_with_its_own_coefficient() {
        // Decision A2-7: with E20 each ring's onsets use its own Br coefficient (the reverse
        // field scales with that ring's Br), the block shows the governing ring's, and the
```

with:

```rust
    }

    #[test]
    fn each_ring_shows_its_own_block() {
        // Plan A-3: the per-ring results read each ring's own block. On the mixed rings (an
        // NdFeB inner ring, a ferrite outer ring) the inner ring's equal the NdFeB-only run's
        // and the outer ring's the ferrite-only run's, whichever ring the sheet shows.
        let e20 = Deviations::only(DeviationId::E20);
        let ti = TemperatureInputs::default();
        let mixed = compute(&ti, &with_outer_of(links(), &ferrite_links()), e20).demag;
        let ndfeb = compute(&ti, &links(), e20).demag;
        let ferrite = compute(&ti, &ferrite_links(), e20).demag;
        assert_eq!(
            (
                mixed.inner_hcj20_kA_m,
                mixed.inner_beta_per_C,
                mixed.inner_magnet_limit_C
            ),
            (
                ndfeb.hcj20_used_kA_m,
                ndfeb.beta_used_per_C,
                ndfeb.magnet_limit_C
            )
        );
        assert_eq!(
            (
                mixed.outer_hcj20_kA_m,
                mixed.outer_beta_per_C,
                mixed.outer_magnet_limit_C
            ),
            (
                ferrite.hcj20_used_kA_m,
                ferrite.beta_used_per_C,
                ferrite.magnet_limit_C
            )
        );
        assert_eq!(mixed.inner_cold_limit_C, NumOrText::Text(NO_COLD_ONSET));
        assert_eq!(mixed.outer_cold_limit_C, ferrite.cold_limit_C);
        // The sheet shows the ring with the lower magnet limit: the NdFeB ring.
        assert!(mixed.inner_magnet_limit_C < mixed.outer_magnet_limit_C);
        assert_eq!(mixed.magnet_limit_C, mixed.inner_magnet_limit_C);
    }

    #[test]
    fn each_ring_is_checked_with_its_own_coefficient() {
        // Decision A2-7: with E20 each ring's onsets use its own Br coefficient (the reverse
        // field scales with that ring's Br), the block shows the governing ring's, and the
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib 2>&1 | grep -E "^error" | head -3
```

Expected: compile errors: ``error[E0609]: no field `steel_density_g_mm3` on type `MaterialsResults` `` and two more `no field` errors on `MaterialsResults` (the temperature test's `no field` errors follow without `head`).

- [ ] **Step 3: Expose the values and extend the tables**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below, each ring's alpha(Br) and magnet density used: a grade-mode ring's grade, decision A2-7, and the equation explorer's terms, Rust-only reads, plan A-3: the wave number, amplitudes and geometry factors of harmonics 7 to 11, `k7` to `s11_free`, each harmonic's torque-angle amplitude `amp1_Pa` to `amp11_Pa`, `Harmonic::amplitude` being the one expression the shear stress and the pull-out share, and the E7 angles `pullout_angle_rad`, `iron_circuit_angle_rad` and `free_circuit_angle_rad`), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27) |
| `src/engine/temperature.rs` | `Temperature design` sheet: 7 input groups (37 cells), the adhesive table `ADHESIVES` with `selected_adhesive`, `TemperatureLinks`, `TemperatureResults` (130 cells) |
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
```

with:

```markdown
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs (plus the Rust-only grade per ring for manual dimensions, Addendum A6, and the axial length override `coupling.magnets.axial_length_mm`, Addendum A1), `ModelResults` (73 cells, plus the Rust-only `inner_grade`, `outer_grade`, `tau7_Pa` to `tau11_Pa` and `end_effect_check`, the audit M9 flag: `END_EFFECT_OUT_OF_RANGE` when f_end is 0 or below, each ring's alpha(Br) and magnet density used: a grade-mode ring's grade, decision A2-7, and the equation explorer's terms, Rust-only reads, plan A-3: the wave number, amplitudes and geometry factors of harmonics 7 to 11, `k7` to `s11_free`, each harmonic's torque-angle amplitude `amp1_Pa` to `amp11_Pa`, `Harmonic::amplitude` being the one expression the shear stress and the pull-out share, and the E7 angles `pullout_angle_rad`, `iron_circuit_angle_rad` and `free_circuit_angle_rad`), the harmonic set (the Rust-only assumption `coupling.max_harmonic`, odd harmonics up to 11, `MAX_HARMONIC_CHOICES`, `harmonic_count`; `HARMONICS`, the workbook's 1, 3, 5, is the default), the E7 peak search, and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/material_library.rs` | Addendum A5 materials library `MATERIALS` (14 materials: ferromagnetic, mu_r, Bsat, sigma, density, CTE, E, yield, cp, design flux density), every value cited; `EngineProps` (what the engine uses when a material is picked: the workbook's number where one exists); each part's choices |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the Rust-only part selectors `materials.parts` (Addendum A5), the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells, plus the Rust-only circuit and material names and `cup_wall_suggested_mm`, the autofit's cup wall, decision 27, and the materials in effect: the steel density, the hub, cup and boss values without back iron, the sleeve's and the cap's, plan A-3) |
| `src/engine/temperature.rs` | `Temperature design` sheet: 7 input groups (37 cells), the adhesive table `ADHESIVES` with `selected_adhesive`, `TemperatureLinks`, `TemperatureResults` (130 cells, plus the Rust-only results of E20: each ring's own Hcj, beta, magnet limit and cold limit, plan A-3) |
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/warnings.rs` | Addendum A5 material warnings: `WARNING_RULES` (six rules: text, `Severity`, teaching-note id), the three model-choice thresholds, `WarningResults` (Rust-only, `warnings.<rule id>`) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

with:

```markdown
| `src/engine/housing.rs` | Addendum A1 housing autofit (which dimensions are derived, suggested or left as inputs: decisions 27, 28; with the axial length override set, `axial_housing` makes the hub length, the cup cavity depth and the retainer span follow the rings they bound, each at least its ring's length: decision A2-8) and the space claim: `HousingResults` (Rust-only, `housing.*`: the overshoot per axis, diameter, overall length and large-diameter bay, `space_claim_check`, and the hub length, cup depth and retainer span in effect) |
| `src/engine/sizing.rs` | Addendum A1 inverse sizing (Torque → Magnets): `FreeVariable` (axial length, the default; magnets per ring, even only; ring radius), `solve` (a coarse scan of `SCAN_CELLS` cells, refined between samples at the first crossing, at each peak and at each validity edge to `VALUE_TOLERANCE_MM`; the smallest value that meets the target, or `NotReachable` with the best valid value it evaluated; the stated limit: a torque hump whose rise and fall both lie inside one cell), `is_valid` (a value counts only if its blocks fit, faceted blocks on their flats and arcs without overlapping, the keyway leaves hub wall and f_end > 0), `SizingOutcome`, `SizingError` |
| `src/engine/assumptions.rs` | Addendum A3 assumptions panel: `ASSUMPTIONS` (the spec's 14 rows over the 15 inputs flagged `.assumption()`: label, input paths, rationale, source), `states`, `modified`, `any_modified` (the "assumptions modified" banner), `reset_to_workbook_defaults` |
| `src/engine/explain/` | Addendum A2 equation explorer, engine side (plan A-3): the formula markup (`markup.rs`: grammar, parser, parse tree), its evaluator with a trace of what a result depends on (`eval.rs`), the static tables a formula reads (`tables.rs`: magnets, grades, the part materials, aluminium alloys, adhesives, screw sizes), the authoring form (`record.rs`) and the records, one file per chain (`records/`), the registry (`registry.rs`: `Registry::build`, `equation_for`, `used_by`, the dependency graph, `term_rows`), display symbols (`symbols.rs`), the v1 scope (`scope.rs`, decision 31) and a plain-text rendering (`render.rs`) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy (`reference_values.json`), exported schemas (`input_schema.json`, `python_schema.json`), the Python static tables (`static_data.json`), differential data (`differential/`) and the golden files of the broad corrections (`deviations/E3.json`, `E4.json`, `E5.json`) |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/mod.rs`, replace:

```rust
//!
//! - [`markup`]: the formula and symbol grammar, parser and tree;
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, back iron);
//! - [`record`]: the authoring form, [`records`]: the records, one file per batch;
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term styles;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
```

with:

```rust
//!
//! - [`markup`]: the formula and symbol grammar, parser and tree;
//! - [`eval`]: the evaluator, with a trace of what a result depends on at a design point;
//! - [`tables`]: the static engine tables a formula can read (magnet library, grades, part
//!   materials, aluminium alloys, adhesives, screw sizes);
//! - [`record`]: the authoring form, [`records`]: the records, one file per batch;
//! - [`registry`]: `Registry::build`, `equation_for`, `used_by`, the dependency graph, term styles;
//! - [`symbols`]: display symbols of inputs and other leaf terms;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
//! The static engine tables a formula can read with `table("name", key, "field")`: the
//! magnet library, the grade table and the back-iron materials, each field as the engine
//! uses it with every approved correction on (the explorer describes what users see).
//!
//! A lookup returns `none` when the key is not in the table (a part name that is not a
//! library part, a blank grade), which is how the engine's own selections branch.

use crate::engine::deviations::Deviations;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::material_library::{BACK_IRON_CHOICES, chosen};
use crate::engine::meta::Value;

/// One readable field: its table, name, display symbol and unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

with:

```rust
//! The static engine tables a formula can read with `table("name", key, "field")`: the
//! magnet library, the grade table, the part materials (back iron, sleeve and liner, cap and
//! housing), the aluminium alloys, the adhesives and the clamp screw sizes, each field as the
//! engine uses it with every approved correction on (the explorer describes what users see).
//!
//! A lookup returns `none` when the key is not in the table (a part name that is not a
//! library part, a blank grade, a selector code outside the choices), which is how the
//! engine's own selections branch. A material or adhesive is keyed by its selector code, a
//! screw size by its row of the screw table (0 for M2.5), an alloy by its name.

use crate::engine::clamps::SCREW_SIZES;
use crate::engine::deviations::Deviations;
use crate::engine::grades;
use crate::engine::library;
use crate::engine::material_library::{
    BACK_IRON_CHOICES, CAP_HOUSING_CHOICES, SLEEVE_LINER_CHOICES, chosen,
};
use crate::engine::materials::{AL6061, AL7075};
use crate::engine::meta::Value;
use crate::engine::temperature::ADHESIVES;

/// One readable field: its table, name, display symbol and unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
    TableField { table: "magnets", field: "width_mm", symbol: "w_{lib}", unit: "mm", label: "Library part tangential width" },
    TableField { table: "magnets", field: "thickness_mm", symbol: "t_{lib}", unit: "mm", label: "Library part radial thickness" },
    TableField { table: "magnets", field: "br_T", symbol: "B_{r,lib}", unit: "T", label: "Library part remanence at 20 °C (E3: the N42SH grade's)" },
    TableField { table: "grades", field: "br_T", symbol: "B_{r,grade}", unit: "T", label: "Grade remanence at 20 °C" },
    TableField { table: "grades", field: "alpha_br_per_C", symbol: "α_{grade}", unit: "1/°C", label: "Grade Br temperature coefficient" },
    TableField { table: "back_iron", field: "ferromagnetic", symbol: "ferro", unit: "-", label: "Back-iron material is ferromagnetic (1) or not (0)" },
];

/// The field's metadata, or `None` if a formula may not read it.
```

with:

```rust
    TableField { table: "magnets", field: "width_mm", symbol: "w_{lib}", unit: "mm", label: "Library part tangential width" },
    TableField { table: "magnets", field: "thickness_mm", symbol: "t_{lib}", unit: "mm", label: "Library part radial thickness" },
    TableField { table: "magnets", field: "br_T", symbol: "B_{r,lib}", unit: "T", label: "Library part remanence at 20 °C (E3: the N42SH grade's)" },
    TableField { table: "magnets", field: "tmax_C", symbol: "ϑ_{max,lib}", unit: "°C", label: "Library part maximum operating temperature (E19: the vendor's where it differs)" },
    TableField { table: "magnets", field: "grade", symbol: "grade_{lib}", unit: "", label: "Library part grade (E19: the vendor grid's where it differs)" },
    TableField { table: "grades", field: "id", symbol: "grade_{tab}", unit: "", label: "The grade's key, when the table has it" },
    TableField { table: "grades", field: "br_T", symbol: "B_{r,grade}", unit: "T", label: "Grade remanence at 20 °C" },
    TableField { table: "grades", field: "alpha_br_per_C", symbol: "α_{grade}", unit: "1/°C", label: "Grade Br temperature coefficient" },
    TableField { table: "grades", field: "hcj20_kA_m", symbol: "H_{cj,grade}", unit: "kA/m", label: "Grade intrinsic coercivity at 20 °C" },
    TableField { table: "grades", field: "beta_hcj_per_C", symbol: "β_{grade}", unit: "1/°C", label: "Grade Hcj temperature coefficient (positive for hard ferrite)" },
    TableField { table: "grades", field: "tmax_C", symbol: "ϑ_{max,grade}", unit: "°C", label: "Grade maximum operating temperature" },
    TableField { table: "grades", field: "density_g_mm3", symbol: "ρ_{grade}", unit: "g/mm³", label: "Grade density" },
    TableField { table: "back_iron", field: "ferromagnetic", symbol: "ferro", unit: "-", label: "Back-iron material is ferromagnetic (1) or not (0)" },
    TableField { table: "back_iron", field: "sigma_S_m", symbol: "σ_{BI}", unit: "S/m", label: "Back-iron material conductivity (library)" },
    TableField { table: "back_iron", field: "density_g_mm3", symbol: "ρ_{BI}", unit: "g/mm³", label: "Back-iron material density (library)" },
    TableField { table: "back_iron", field: "cp_J_kgK", symbol: "c_{BI}", unit: "J/(kg·K)", label: "Back-iron material specific heat (library)" },
    TableField { table: "back_iron", field: "design_flux_density_T", symbol: "B_{des,BI}", unit: "T", label: "Back-iron material design flux density (decision 20; none where the workbook gives none)" },
    TableField { table: "sleeve_liner", field: "sigma_S_m", symbol: "σ_{SL}", unit: "S/m", label: "Sleeve and liner material conductivity (library)" },
    TableField { table: "sleeve_liner", field: "density_g_mm3", symbol: "ρ_{SL}", unit: "g/mm³", label: "Sleeve and liner material density (library)" },
    TableField { table: "sleeve_liner", field: "cp_J_kgK", symbol: "c_{SL}", unit: "J/(kg·K)", label: "Sleeve and liner material specific heat (library)" },
    TableField { table: "cap_housing", field: "sigma_S_m", symbol: "σ_{cap}", unit: "S/m", label: "Cap material conductivity (library)" },
    TableField { table: "cap_housing", field: "density_g_mm3", symbol: "ρ_{cap}", unit: "g/mm³", label: "Cap material density (library)" },
    TableField { table: "cap_housing", field: "cp_J_kgK", symbol: "c_{cap}", unit: "J/(kg·K)", label: "Cap material specific heat (library)" },
    TableField { table: "aluminium", field: "conductivity_S_m", symbol: "σ_{Al}", unit: "S/m", label: "Aluminium alloy conductivity (Materials C38, C43)" },
    TableField { table: "aluminium", field: "shear_MPa", symbol: "τ_{Al}", unit: "MPa", label: "Aluminium alloy shear strength (Materials C35, C40)" },
    TableField { table: "aluminium", field: "head_pressure_limit_MPa", symbol: "p_{head,Al}", unit: "MPa", label: "Aluminium alloy limiting pressure under a screw head (Materials C36, C41)" },
    TableField { table: "aluminium", field: "key_bearing_allow_MPa", symbol: "p_{key,Al}", unit: "MPa", label: "Aluminium alloy key bearing allowable (Materials C37, C42)" },
    TableField { table: "adhesives", field: "design_limit_C", symbol: "ϑ_{adh}", unit: "°C", label: "Adhesive design limit (Temperature design C65:C74)" },
    TableField { table: "adhesives", field: "cure_C", symbol: "ϑ_{cure}", unit: "°C", label: "Adhesive cure (stress-free) temperature" },
    TableField { table: "adhesives", field: "lap_shear_MPa", symbol: "τ_{lap}", unit: "MPa", label: "Adhesive lap shear at 22 °C (TDS)" },
    TableField { table: "screw_sizes", field: "name", symbol: "size", unit: "", label: "Screw size (Clamp screw sizes row 5)" },
    TableField { table: "screw_sizes", field: "d_mm", symbol: "d", unit: "mm", label: "Screw nominal diameter" },
    TableField { table: "screw_sizes", field: "As_mm2", symbol: "A_s", unit: "mm²", label: "Screw tensile stress area" },
    TableField { table: "screw_sizes", field: "hole_mm", symbol: "d_h", unit: "mm", label: "Clearance hole (ISO 273 medium)" },
    TableField { table: "screw_sizes", field: "head_mm", symbol: "d_k", unit: "mm", label: "Head diameter (ISO 4762 maximum)" },
    TableField { table: "screw_sizes", field: "hex_mm", symbol: "s_{hex}", unit: "mm", label: "Hex key" },
];

/// The field's metadata, or `None` if a formula may not read it.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
        Value::Text(s) => Ok(s.clone()),
        other => Err(format!("table {table}: key {other:?} is not a text")),
    };
    Ok(match table {
        "magnets" => library::lookup(&text(key)?).map(|spec| {
            Value::Num(match field_name {
                "length_mm" => spec.length_mm,
                "width_mm" => spec.width_mm,
                "thickness_mm" => spec.thickness_mm,
                "br_T" => library::br_T(spec, Deviations::ALL),
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }),
        "grades" => grades::grade(&text(key)?).map(|g| {
            Value::Num(match field_name {
                "br_T" => g.br_T,
                "alpha_br_per_C" => g.alpha_br_per_C,
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }),
        "back_iron" => {
            // A code: an integer input, or the same number after arithmetic promoted it.
            let code = match key {
                Value::Int(c) => *c,
                Value::Num(x) if x.fract() == 0.0 && x.abs() < 1e15 => *x as i64,
                other => return Err(format!("table back_iron: key {other:?} is not a code")),
            };
            chosen(&BACK_IRON_CHOICES, code).map(|m| Value::Int(i64::from(m.ferromagnetic)))
        }
        _ => unreachable!("checked against TABLE_FIELDS"),
    })
```

with:

```rust
        Value::Text(s) => Ok(s.clone()),
        other => Err(format!("table {table}: key {other:?} is not a text")),
    };
    // A code: an integer input, or the same number after arithmetic promoted it.
    let code = |k: &Value| match k {
        Value::Int(c) => Ok(*c),
        Value::Num(x) if x.fract() == 0.0 && x.abs() < 1e15 => Ok(*x as i64),
        other => Err(format!("table {table}: key {other:?} is not a code")),
    };
    let num = Value::Num;
    Ok(match table {
        "magnets" => library::lookup(&text(key)?).map(|spec| match field_name {
            "length_mm" => num(spec.length_mm),
            "width_mm" => num(spec.width_mm),
            "thickness_mm" => num(spec.thickness_mm),
            "br_T" => num(library::br_T(spec, Deviations::ALL)),
            "tmax_C" => num(library::tmax_C(spec, Deviations::ALL)),
            "grade" => Value::Text(library::grade_id(spec, Deviations::ALL).to_owned()),
            _ => unreachable!("checked against TABLE_FIELDS"),
        }),
        "grades" => grades::grade(&text(key)?).map(|g| match field_name {
            "id" => Value::Text(g.id.to_owned()),
            "br_T" => num(g.br_T),
            "alpha_br_per_C" => num(g.alpha_br_per_C),
            "hcj20_kA_m" => num(g.hcj20_kA_m),
            "beta_hcj_per_C" => num(g.beta_hcj_per_C),
            "tmax_C" => num(g.tmax_C),
            "density_g_mm3" => num(g.density_g_mm3),
            _ => unreachable!("checked against TABLE_FIELDS"),
        }),
        "back_iron" | "sleeve_liner" | "cap_housing" => {
            let choices: &[(i64, &'static str)] = match table {
                "back_iron" => &BACK_IRON_CHOICES,
                "sleeve_liner" => &SLEEVE_LINER_CHOICES,
                _ => &CAP_HOUSING_CHOICES,
            };
            chosen(choices, code(key)?).map(|m| match field_name {
                "ferromagnetic" => Value::Int(i64::from(m.ferromagnetic)),
                "sigma_S_m" => num(m.engine.sigma_S_m),
                "density_g_mm3" => num(m.engine.density_g_mm3),
                "cp_J_kgK" => num(m.engine.cp_J_kgK),
                "design_flux_density_T" => m.design_flux_density_T.map_or(Value::None, num),
                _ => unreachable!("checked against TABLE_FIELDS"),
            })
        }
        "aluminium" => {
            let name = text(key)?;
            [AL7075, AL6061]
                .into_iter()
                .find(|a| a.name == name)
                .map(|a| {
                    num(match field_name {
                        "conductivity_S_m" => a.conductivity_S_m,
                        "shear_MPa" => a.shear_MPa,
                        "head_pressure_limit_MPa" => a.head_pressure_limit_MPa,
                        "key_bearing_allow_MPa" => a.key_bearing_allow_MPa,
                        _ => unreachable!("checked against TABLE_FIELDS"),
                    })
                })
        }
        "adhesives" => {
            let c = code(key)?;
            (1..=ADHESIVES.len() as i64).contains(&c).then(|| {
                let a = &ADHESIVES[(c - 1) as usize];
                num(match field_name {
                    "design_limit_C" => a.design_limit_C,
                    "cure_C" => a.cure_C,
                    "lap_shear_MPa" => a.lap_shear_MPa,
                    _ => unreachable!("checked against TABLE_FIELDS"),
                })
            })
        }
        "screw_sizes" => {
            let row = code(key)?;
            usize::try_from(row)
                .ok()
                .and_then(|r| SCREW_SIZES.get(r))
                .map(|s| match field_name {
                    "name" => Value::Text(s.name.to_owned()),
                    "d_mm" => num(s.d_mm),
                    "As_mm2" => num(s.As_mm2),
                    "hole_mm" => num(s.hole_mm),
                    "head_mm" => num(s.head_mm),
                    "hex_mm" => num(s.hex_mm),
                    _ => unreachable!("checked against TABLE_FIELDS"),
                })
        }
        _ => unreachable!("checked against TABLE_FIELDS"),
    })
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
mod tests {
    use super::*;

    #[test]
    fn every_field_reads_and_a_missing_key_is_none() {
        let key = |t: &str| match t {
            "magnets" => Value::Text("B842SH".into()),
            "grades" => Value::Text("N42SH".into()),
            _ => Value::Int(1),
        };
        for f in TABLE_FIELDS {
            assert!(
                lookup(f.table, &key(f.table), f.field).unwrap().is_some(),
                "{}.{}",
                f.table,
                f.field
            );
            let missing = if f.table == "back_iron" {
                Value::Int(99)
            } else {
                Value::Text(String::new())
            };
            assert_eq!(
                lookup(f.table, &missing, f.field).unwrap(),
                None,
```

with:

```rust
mod tests {
    use super::*;

    /// A key each table holds, and one it does not.
    fn keys(table: &str) -> (Value, Value) {
        match table {
            "magnets" => (Value::Text("B842SH".into()), Value::Text(String::new())),
            "grades" => (Value::Text("N42SH".into()), Value::Text(String::new())),
            "aluminium" => (Value::Text("6061-T6".into()), Value::Text("2024-T3".into())),
            "screw_sizes" => (Value::Int(0), Value::Int(5)),
            _ => (Value::Int(1), Value::Int(99)),
        }
    }

    #[test]
    fn every_field_reads_and_a_missing_key_is_none() {
        for f in TABLE_FIELDS {
            let (held, missing) = keys(f.table);
            assert!(
                lookup(f.table, &held, f.field).unwrap().is_some(),
                "{}.{}",
                f.table,
                f.field
            );
            assert_eq!(
                lookup(f.table, &missing, f.field).unwrap(),
                None,
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/tables.rs`, replace:

```rust
            );
        }
        assert!(lookup("magnets", &Value::Int(1), "br_T").is_err());
        assert!(lookup("magnets", &key("magnets"), "density").is_err());
        assert!(lookup("nope", &key("magnets"), "br_T").is_err());
    }

    #[test]
```

with:

```rust
            );
        }
        assert!(lookup("magnets", &Value::Int(1), "br_T").is_err());
        assert!(lookup("magnets", &keys("magnets").0, "density").is_err());
        assert!(lookup("nope", &keys("magnets").0, "br_T").is_err());
        assert!(
            lookup("adhesives", &Value::Text("1".into()), "cure_C").is_err(),
            "a code key"
        );
    }

    #[test]
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
            back_iron_material: String => out_rust_only("", "Back iron material", ""),
            sleeve_liner_material: String => out_rust_only("", "Sleeve and liner material", ""),
            cap_material: String => out_rust_only("", "Cap and housing material", ""),
        }
    }
}
```

with:

```rust
            back_iron_material: String => out_rust_only("", "Back iron material", ""),
            sleeve_liner_material: String => out_rust_only("", "Sleeve and liner material", ""),
            cap_material: String => out_rust_only("", "Cap and housing material", ""),
            steel_density_g_mm3: f64 => out_rust_only("g/mm³", "Back-iron steel density in effect",
                "Plan A-3 (a term of the equation explorer): Metal design C132, or a ferromagnetic back-iron pick's library density (Addendum A5); the mass model prices the steel parts with it."),
            body_sigma_S_m: f64 => out_rust_only("S/m", "Hub, cup and boss conductivity without back iron",
                "The workbook's 6061-T6 (Materials C43), or a non-ferromagnetic back-iron pick's (Addendum A5); correction E17 prices the aluminium parts' slip losses with it."),
            body_density_g_mm3: f64 => out_rust_only("g/mm³", "Hub, cup and boss density without back iron",
                "Metal design C42, or a non-ferromagnetic back-iron pick's (Addendum A5): the aluminium hub, cup and boss (E9)."),
            body_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Hub, cup and boss specific heat without back iron",
                "Temperature design C140, or a non-ferromagnetic back-iron pick's (Addendum A5): correction E15's heat capacity."),
            sleeve_sigma_S_m: f64 => out_rust_only("S/m", "Sleeve and liner conductivity in effect",
                "Temperature design C111, or the sleeve and liner pick's (Addendum A5)."),
            sleeve_density_g_mm3: f64 => out_rust_only("g/mm³", "Sleeve, liner and endplate density in effect",
                "Metal design C44, or the sleeve and liner pick's (Addendum A5)."),
            sleeve_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Sleeve, liner and endplate specific heat in effect",
                "Temperature design C139, or the sleeve and liner pick's (Addendum A5)."),
            cap_density_g_mm3: f64 => out_rust_only("g/mm³", "Cap density in effect",
                "Metal design C42, or the cap pick's (Addendum A5)."),
            cap_c_J_kgK: f64 => out_rust_only("J/(kg·K)", "Cap specific heat in effect",
                "Temperature design C140, or the cap pick's (Addendum A5)."),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/materials.rs`, replace:

```rust
        back_iron_material: parts.back_iron.label().to_owned(),
        sleeve_liner_material: parts.sleeve_liner.label().to_owned(),
        cap_material: parts.cap.label().to_owned(),
    }
}
```

with:

```rust
        back_iron_material: parts.back_iron.label().to_owned(),
        sleeve_liner_material: parts.sleeve_liner.label().to_owned(),
        cap_material: parts.cap.label().to_owned(),
        steel_density_g_mm3: parts.steel.density_g_mm3,
        body_sigma_S_m: parts.body.sigma_S_m,
        body_density_g_mm3: parts.body.density_g_mm3,
        body_c_J_kgK: parts.body.cp_J_kgK,
        sleeve_sigma_S_m: parts.sleeve_liner.props.sigma_S_m,
        sleeve_density_g_mm3: parts.sleeve_liner.props.density_g_mm3,
        sleeve_c_J_kgK: parts.sleeve_liner.props.cp_J_kgK,
        cap_density_g_mm3: parts.cap.props.density_g_mm3,
        cap_c_J_kgK: parts.cap.props.cp_J_kgK,
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
                "Correction E20: the ring with the higher cold limit (the inner ring when neither has one)."),
            cold_check: String => out_rust_only("", "Cold demagnetization check",
                "Against the minimum magnet temperature (Metal design C16); it passes only if both rings pass."),
        }
    }
}
```

with:

```rust
                "Correction E20: the ring with the higher cold limit (the inner ring when neither has one)."),
            cold_check: String => out_rust_only("", "Cold demagnetization check",
                "Against the minimum magnet temperature (Metal design C16); it passes only if both rings pass."),
            inner_hcj20_kA_m: f64 => out_rust_only("kA/m", "Inner ring: Hcj at 20 °C used",
                "Plan A-3 (a term of the equation explorer): correction E20, the inner ring's grade, or C44 when it has no grade or the coercivity source is set to the inputs."),
            inner_beta_per_C: f64 => out_rust_only("1/°C", "Inner ring: Hcj temperature coefficient used",
                "Correction E20: the inner ring's grade's, or C45."),
            inner_magnet_limit_C: f64 => out_rust_only("°C", "Inner ring: magnet design limit",
                "Correction E20: the inner ring's own block (C47 to C60), whichever ring governs: its skipping onset minus the design margin, or with a positive beta its rating (+inf without one)."),
            inner_cold_limit_C: NumOrText => out_rust_only("°C", "Inner ring: cold magnet limit",
                "Correction E20, positive beta only: the inner ring's skipping cold onset plus the design margin; 'n/a' otherwise."),
            outer_hcj20_kA_m: f64 => out_rust_only("kA/m", "Outer ring: Hcj at 20 °C used",
                "Correction E20: the outer ring's grade, or C44. Without E20 the workbook checks the inner ring only, and the outer ring's block (from C44 and C45) governs nothing."),
            outer_beta_per_C: f64 => out_rust_only("1/°C", "Outer ring: Hcj temperature coefficient used",
                "Correction E20: the outer ring's grade's, or C45."),
            outer_magnet_limit_C: f64 => out_rust_only("°C", "Outer ring: magnet design limit",
                "Correction E20: as the inner ring's, for the outer ring."),
            outer_cold_limit_C: NumOrText => out_rust_only("°C", "Outer ring: cold magnet limit",
                "Correction E20: as the inner ring's, for the outer ring."),
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            dev,
        )
    });
    let (hot, hot_ring) = match outer {
        Some(o) if o.mag_lim < inner.mag_lim => (o, RING_OUTER),
        _ => (inner, RING_INNER),
```

with:

```rust
            dev,
        )
    });
    // Plan A-3: the outer ring's own block for the per-ring results the explorer reads; without
    // E20 it governs nothing (the workbook checks the inner ring only).
    let outer_block = outer.unwrap_or_else(|| {
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

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            NumOrText::Num(_) => "Below the cold demagnetization limit",
        }
        .to_owned(),
    };

    // ---- adhesive selection and loads
```

with:

```rust
            NumOrText::Num(_) => "Below the cold demagnetization limit",
        }
        .to_owned(),
        // Plan A-3: each ring's own values, read from its block, so the explorer can show which
        // ring governs and why.
        inner_hcj20_kA_m: inner.hcj20,
        inner_beta_per_C: inner.beta,
        inner_magnet_limit_C: inner.mag_lim,
        inner_cold_limit_C: inner.cold_limit,
        outer_hcj20_kA_m: outer_block.hcj20,
        outer_beta_per_C: outer_block.beta,
        outer_magnet_limit_C: outer_block.mag_lim,
        outer_cold_limit_C: outer_block.cold_limit,
    };

    // ---- adhesive selection and loads
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
            want.cold_limit_C = cold.cold_limit_C;
            want.cold_ring = RING_OUTER.to_owned();
            want.cold_check = cold.cold_check;
            // Decision A2-7: the torque at the limit goes with both rings' Br, each ring with
            // its own coefficient (the NdFeB inner, the ferrite outer).
            want.torque_at_limit_Nm = mixed.pullout_20C_Nm
```

with:

```rust
            want.cold_limit_C = cold.cold_limit_C;
            want.cold_ring = RING_OUTER.to_owned();
            want.cold_check = cold.cold_check;
            // Plan A-3's per-ring results: the outer ring is the ferrite one.
            want.outer_hcj20_kA_m = cold.outer_hcj20_kA_m;
            want.outer_beta_per_C = cold.outer_beta_per_C;
            want.outer_magnet_limit_C = cold.outer_magnet_limit_C;
            want.outer_cold_limit_C = cold.outer_cold_limit_C;
            // Decision A2-7: the torque at the limit goes with both rings' Br, each ring with
            // its own coefficient (the NdFeB inner, the ferrite outer).
            want.torque_at_limit_Nm = mixed.pullout_20C_Nm
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/temperature.rs`, replace:

```rust
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
```

with:

```rust
        assert_eq!(s.demag.cold_limit_C, r.demag.cold_limit_C);
        // Without E20 the workbook reads the inner ring's grade, Br and rating only (the outer
        // ring's coefficient still scales the torques, decision A2-7).
        // Plan A-3's outer-ring results show that ring's block all the same; it governs nothing.
        let workbook_cells = |mut r: TemperatureResults| {
            r.demag.outer_hcj20_kA_m = 0.0;
            r.demag.outer_beta_per_C = 0.0;
            r.demag.outer_magnet_limit_C = 0.0;
            r.demag.outer_cold_limit_C = NumOrText::Text("");
            r
        };
        let mut one_alpha = mixed.clone();
        one_alpha.outer_alpha_br = links().outer_alpha_br;
        assert_eq!(
            workbook_cells(run(&ti, &one_alpha)),
            workbook_cells(run(&ti, &links())),
            "the workbook reads the inner ring only"
        );
    }
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib -- each_ring_shows_its_own_block the_materials_in_effect corrected_library_fields e20_checks_both_rings 2>&1 | grep "test result"
```

Expected: `test result: ok. 4 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml 2>&1 | grep -E "test result|FAILED"
```

Expected: every binary `ok`, no `FAILED`: unit tests `170 passed`; `tests\explain.rs` `10 passed; 0 failed; 2 ignored`; the other binaries as in Task 0.

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task7.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task7.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task7.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 5: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/temperature.rs magcoupling-rs/src/engine/materials.rs magcoupling-rs/src/engine/explain/tables.rs magcoupling-rs/src/engine/explain/mod.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): expose each ring's demagnetization block and the materials in effect

Rust-only reads for the explorer: each ring's Hcj, beta, magnet limit and cold
limit (E20; the outer ring's block is computed with E20 off too, governing
nothing) and the part materials in effect (A5). The static tables gain every
library value the remaining records read, corrections on (E19).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 8: Batch 2: the demagnetization block, both rings, the cold side and the governing limit

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Step 6, the physics review, runs on the session model.

**Batch order (decision G3).** Each batch writes its chain's records and the closure its terms need, then flips the chain to
`Explained`, whereupon every path must have a record and drill down to inputs. The chains are entangled through two values: the
demagnetization margins read the peak magnet temperature (slip heating), and the slip-heating time to the limit reads the governing
limit (the demagnetization block). So the batches follow that closure: this batch writes the demagnetization block and the governing
limit; Task 9 adds slip heating and flips both chains; Task 10 flips temperature. This batch's red test checks that the block and
the governing limit drill down to inputs (the helper `cell_only_below`, shared with the chain test).
**The physics.** The onsets are the workbook's linear crossing of the reverse field (scaling with Br) and the knee (scaling with
Hcj): ϑ = 20 + (H_k − H)/(H_k |β| − H |α|) − Δϑ_cal, H_k = k_knee H_cj, offset so the reference magnet (permeance coefficient 1, H =
Br/(2 μ0)) reaches its knee at its rating. E20 checks each ring with its own grade, Br and rating; the ring with the lower magnet limit
governs and the sheet shows its block, so the shown block selects each ring's inputs and keeps the formula visible, while each ring's
limit (Rust-only, Task 7) is one record with `where` bindings for its knee, reference field and offset (values the engine computes
but does not expose per ring). With a positive beta (hard ferrite) the knee falls as the magnet cools: no hot onset (`inf`), the
rating is the hot limit (`nan` torque at a limit that does not exist), and the cold side checks the ring with the higher cold limit.
`cold_check` rests on an equivalence, not a transcription: both rings pass exactly when the minimum temperature is at or above the
higher cold limit. Three augmentations reach the cold side: "ferrite inputs" (coercivity from the inputs with a positive beta, both
rings in the grade mode with different Br coefficients and densities), "ferrite inputs, beta 0.002" (a second positive beta, the
rings swapped, so every cold-side record sees two betas and the outer ring a second Br coefficient and density: the drift guard's
term-level check needs each value term at two values) and "unrated ferrite" (manual magnets with no grade).

**Files:**
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/demagnetization.rs` (39 records)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` (the batch's file)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs` (12 input symbols)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (`cell_only_below`; the batch's drill-down test; three ferrite augmentations)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the demagnetization row)

**Interfaces:**
- Consumes: Task 7's per-ring results and tables (`grades.hcj20_kA_m`, `beta_hcj_per_C`, `tmax_C`, `id`; `magnets.tmax_C`, `grade`; `adhesives`); Task 6's `inf` and `nan`; Task 3's registry.
- Produces: records for `model.{inner,outer}_grade`, `model.{inner,outer}_tmax_C`, each ring's `temperature.demag.{inner,outer}_*` block, the block the sheet shows (`demag_ring`, `br20_T`, `alpha_br`, `tmax_lib_C`, `hcj20_used_kA_m`, `beta_used_per_C`, `h_ref_kA_m`, `t_ref_model_C`, `calibration_offset_C`, the four onsets, `magnet_limit_C`, `torque_at_limit_Nm`, `torque_at_service_Nm`), the cold side (`cold_ring`, `cold_limit_C`, `cold_check`, the four cold onsets), `temperature.adhesive.{design_limit_C, cure_C}`, `temperature.summary.governing_limit_C`, `temperature.duty.hot_day_start_C`; in `tests/explain.rs`, `fn cell_only_below(r: &Registry, path: &str) -> Vec<String>` and the test `the_demagnetization_block_and_the_governing_limit_drill_down_to_inputs`.

- [ ] **Step 1: Write the failing test**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
    }
}

#[test]
fn every_explained_chain_drills_down_to_inputs() {
    // Every term below every record of an explained chain is an input or has a record:
```

with:

```rust
    }
}

/// The terms below `path` (transitively) that are neither inputs nor explained: where a
/// drill-down from `path` would stop at a cell-only result. Empty when it reaches inputs.
fn cell_only_below(r: &Registry, path: &str) -> Vec<String> {
    r.upstream(path)
        .into_iter()
        .filter(|up| {
            !matches!(
                r.term_kind(up),
                Some(TermKind::Input { .. } | TermKind::Explained)
            )
        })
        .collect()
}

#[test]
fn every_explained_chain_drills_down_to_inputs() {
    // Every term below every record of an explained chain is an input or has a record:
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
    assert!(explained.iter().any(|c| c.id == "torque"));
    for chain in explained {
        for path in chain.paths {
            for up in r.upstream(&path.replace("[]", "[0]")) {
                assert!(
                    matches!(
                        r.term_kind(&up),
                        Some(TermKind::Input { .. } | TermKind::Explained)
                    ),
                    "{}: {path} depends on {up}, which is neither an input nor explained",
                    chain.id
                );
            }
        }
    }
}
```

with:

```rust
    assert!(explained.iter().any(|c| c.id == "torque"));
    for chain in explained {
        for path in chain.paths {
            let stops = cell_only_below(&r, &path.replace("[]", "[0]"));
            assert!(
                stops.is_empty(),
                "{}: {path} depends on {stops:?}, neither inputs nor explained",
                chain.id
            );
        }
    }
}

#[test]
fn the_demagnetization_block_and_the_governing_limit_drill_down_to_inputs() {
    // Batch 2: both rings' blocks (E20), the block the sheet shows, the cold side and the
    // governing limit reach inputs; the chains that read them flip once their other terms do.
    let r = Registry::build();
    for path in [
        "temperature.summary.governing_limit_C",
        "temperature.demag.h_ref_kA_m",
        "temperature.demag.t_ref_model_C",
        "temperature.demag.calibration_offset_C",
        "temperature.demag.onset_aligned_C",
        "temperature.demag.onset_pullout_C",
        "temperature.demag.onset_skipping_C",
        "temperature.demag.onset_single_ring_C",
        "temperature.demag.magnet_limit_C",
        "temperature.demag.torque_at_limit_Nm",
        "temperature.demag.torque_at_service_Nm",
        "temperature.demag.cold_check",
        "temperature.demag.cold_onset_skipping_C",
        "temperature.duty.hot_day_start_C",
        "temperature.adhesive.cure_C",
    ] {
        assert!(r.equation_for(path).is_some(), "{path} has no record");
        let stops = cell_only_below(&r, path);
        assert!(
            stops.is_empty(),
            "{path} depends on {stops:?}, neither inputs nor explained"
        );
    }
}
```

- [ ] **Step 2: Run it to see it fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain the_demagnetization_block 2>&1 | grep -E "panicked|has no record" | head -3
```

Expected: the test panics at its first path: `temperature.summary.governing_limit_C has no record` (after the `thread 'the_demagnetization_block_and_the_governing_limit_drill_down_to_inputs' panicked at tests\explain.rs:...` line).

- [ ] **Step 3: Write the records, their symbols and the augmentations**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override, the coercivity from the inputs with two ferrite betas), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| Chain (decision 31) | Records | Status |
|---|---|---|
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | pending |
| slip heating | `records/slip_heating.rs` | pending |
| temperature | `records/temperature.rs` | pending |
| clamps | `records/clamps.rs` | pending |
```

with:

```markdown
| Chain (decision 31) | Records | Status |
|---|---|---|
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | records written; explained with the slip-heating records (its margins read the peak magnet temperature) |
| slip heating | `records/slip_heating.rs` | pending |
| temperature | `records/temperature.rs` | pending |
| clamps | `records/clamps.rs` | pending |
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/demagnetization.rs` with exactly this content:

```rust
//! Demagnetization (plan A-3 batch 2): the Temperature design block C42 to C62 for both rings
//! (correction E20: each ring against its own grade, Br and rating; the ring with the lower
//! magnet limit governs and the block shows it), the positive-beta cold side of hard ferrite,
//! and the governing temperature limit with the adhesive limit and the hot-day start it is
//! compared with.
//!
//! Each formula is transcribed from the engine function named beside it (`temperature.rs`),
//! with every approved correction on. The onsets are the workbook's linear crossing of the
//! reverse field (scaling with Br) and the knee (scaling with Hcj):
//! ϑ = 20 + (H_k − H) / (H_k |β| − H |α|) − Δϑ_cal, H_k = k_knee · H_cj.
//!
//! Symbols: ϑ temperature, H field [kA/m], β the Hcj coefficient, α the Br coefficient.

use crate::engine::deviations::DeviationId::{E19, E20};
use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- The rings' grades and ratings (model::resolve_magnets, E19 on library parts) ---
    record("model.inner_grade", "grade_{i,used}",
        r#"cases([g_{lib}] != none => [g_{lib}]; [g_{in}] != none => [g_{in}]; else => "")
           where [g_{lib}] = table("magnets", {coupling.magnets.part_inner}, "grade"),
                 [g_{in}] = table("grades", {coupling.magnets.grade_inner}, "id")"#).corrected(&[E19]),
    record("model.outer_grade", "grade_{o,used}",
        r#"cases([g_{lib}] != none => [g_{lib}]; [g_{in}] != none => [g_{in}]; else => "")
           where [g_{lib}] = table("magnets", {coupling.magnets.part_outer}, "grade"),
                 [g_{in}] = table("grades", {coupling.magnets.grade_outer}, "id")"#).corrected(&[E19]),
    record("model.inner_tmax_C", "ϑ_{max,i}",
        r#"cases([ϑ_{lib}] != none => [ϑ_{lib}]; [ϑ_{grade}] != none => [ϑ_{grade}]; else => "n/a")
           where [ϑ_{lib}] = table("magnets", {coupling.magnets.part_inner}, "tmax_C"),
                 [ϑ_{grade}] = table("grades", {coupling.magnets.grade_inner}, "tmax_C")"#).corrected(&[E19]),
    record("model.outer_tmax_C", "ϑ_{max,o}",
        r#"cases([ϑ_{lib}] != none => [ϑ_{lib}]; [ϑ_{grade}] != none => [ϑ_{grade}]; else => "n/a")
           where [ϑ_{lib}] = table("magnets", {coupling.magnets.part_outer}, "tmax_C"),
                 [ϑ_{grade}] = table("grades", {coupling.magnets.grade_outer}, "tmax_C")"#).corrected(&[E19]),

    // --- Each ring's coercivity and limits (temperature::ring_demag, E20) ---
    // The ring's grade supplies Hcj and beta unless the coercivity source is the inputs (0);
    // a magnet without a grade always uses the inputs.
    record("temperature.demag.inner_hcj20_kA_m", "H_{cj,i}",
        r#"cases({temperature.demag.coercivity_source} = 1 and [H_{grade}] != none => [H_{grade}];
               else => {temperature.demag.hcj20_kA_m})
           where [H_{grade}] = table("grades", {model.inner_grade}, "hcj20_kA_m")"#).corrected(&[E20]),
    record("temperature.demag.outer_hcj20_kA_m", "H_{cj,o}",
        r#"cases({temperature.demag.coercivity_source} = 1 and [H_{grade}] != none => [H_{grade}];
               else => {temperature.demag.hcj20_kA_m})
           where [H_{grade}] = table("grades", {model.outer_grade}, "hcj20_kA_m")"#).corrected(&[E20]),
    record("temperature.demag.inner_beta_per_C", "β_i",
        r#"cases({temperature.demag.coercivity_source} = 1 and [β_{grade}] != none => [β_{grade}];
               else => {temperature.demag.beta_hcj_per_C})
           where [β_{grade}] = table("grades", {model.inner_grade}, "beta_hcj_per_C")"#).corrected(&[E20]),
    record("temperature.demag.outer_beta_per_C", "β_o",
        r#"cases({temperature.demag.coercivity_source} = 1 and [β_{grade}] != none => [β_{grade}];
               else => {temperature.demag.beta_hcj_per_C})
           where [β_{grade}] = table("grades", {model.outer_grade}, "beta_hcj_per_C")"#).corrected(&[E20]),
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
    // temperature::cold_onset_C: with beta > 0 the knee falls as the magnet cools; the ring's
    // cold limit is its skipping cold onset plus the margin.
    record("temperature.demag.inner_cold_limit_C", "ϑ_{cold,i}",
        r#"cases({temperature.demag.inner_beta_per_C} > 0
                   => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                {temperature.demag.h_rev_likepole_kA_m} · {model.inner_alpha_br_per_C} - [H_k] · {temperature.demag.inner_beta_per_C})
                      + {temperature.demag.design_margin_C};
               else => "n/a")
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.inner_hcj20_kA_m}"#).corrected(&[E20]),
    record("temperature.demag.outer_cold_limit_C", "ϑ_{cold,o}",
        r#"cases({temperature.demag.outer_beta_per_C} > 0
                   => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                {temperature.demag.h_rev_likepole_kA_m} · {model.outer_alpha_br_per_C} - [H_k] · {temperature.demag.outer_beta_per_C})
                      + {temperature.demag.design_margin_C};
               else => "n/a")
           where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.outer_hcj20_kA_m}"#).corrected(&[E20]),

    // --- The block the sheet shows: the governing ring (temperature::compute) ---
    record("temperature.demag.demag_ring", "ring_{hot}",
        r#"cases({temperature.demag.outer_magnet_limit_C} < {temperature.demag.inner_magnet_limit_C} => "outer"; else => "inner")"#)
        .corrected(&[E20]),
    record("temperature.demag.br20_T", "B_{r,ring}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {model.outer_br_T}; else => {model.inner_br_T})"#).corrected(&[E20]),
    record("temperature.demag.alpha_br", "α_{ring}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C})"#)
        .corrected(&[E20]),
    record("temperature.demag.tmax_lib_C", "ϑ_{max}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {model.outer_tmax_C}; else => {model.inner_tmax_C})"#).corrected(&[E20]),
    record("temperature.demag.hcj20_used_kA_m", "H_{cj}",
        r#"cases({temperature.demag.demag_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m}; else => {temperature.demag.inner_hcj20_kA_m})"#)
        .corrected(&[E20]),
    record("temperature.demag.beta_used_per_C", "β",
        r#"cases({temperature.demag.demag_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C})"#)
        .corrected(&[E20]),
    // The reverse field of a magnet at permeance coefficient 1: Br / (2 μ0), in kA/m.
    record("temperature.demag.h_ref_kA_m", "H_{ref}", "frac({temperature.demag.br20_T}, 2 * {coupling.mu0}) / 1000"),
    record("temperature.demag.t_ref_model_C", "ϑ_{ref}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_ref_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_ref_kA_m} · abs({temperature.demag.alpha_br})))
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.calibration_offset_C", "Δϑ_{cal}",
        r#"cases({temperature.demag.beta_used_per_C} > 0 => 0;
               {temperature.demag.tmax_lib_C} != "n/a" => {temperature.demag.t_ref_model_C} - {temperature.demag.tmax_lib_C};
               else => 0)"#).corrected(&[E20]),
    record("temperature.demag.onset_aligned_C", "ϑ_{al}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_aligned_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_aligned_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.onset_pullout_C", "ϑ_{po}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_pullout_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_pullout_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.onset_skipping_C", "ϑ_{sk}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_likepole_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.onset_single_ring_C", "ϑ_{sr}",
        "cases({temperature.demag.beta_used_per_C} > 0 => inf;
               else => 20 + frac([H_k] - {temperature.demag.h_rev_single_ring_kA_m},
                                 [H_k] · abs({temperature.demag.beta_used_per_C}) - {temperature.demag.h_rev_single_ring_kA_m} · abs({temperature.demag.alpha_br}))
                       - {temperature.demag.calibration_offset_C})
         where [H_k] = {temperature.demag.knee_fraction} * {temperature.demag.hcj20_used_kA_m}").corrected(&[E20]),
    record("temperature.demag.magnet_limit_C", "ϑ_{mag}",
        r#"cases({temperature.demag.beta_used_per_C} > 0
                   => cases({temperature.demag.tmax_lib_C} != "n/a" => {temperature.demag.tmax_lib_C}; else => inf);
               else => {temperature.demag.onset_skipping_C} - {temperature.demag.design_margin_C})"#).corrected(&[E20]),

    // --- The cold side (E20, positive beta): the ring with the higher cold limit ---
    record("temperature.demag.cold_ring", "ring_{cold}",
        r#"cases({temperature.demag.outer_cold_limit_C} != "n/a" and {temperature.demag.inner_cold_limit_C} != "n/a"
                   and {temperature.demag.outer_cold_limit_C} > {temperature.demag.inner_cold_limit_C} => "outer";
               {temperature.demag.outer_cold_limit_C} != "n/a" and {temperature.demag.inner_cold_limit_C} = "n/a" => "outer";
               else => "inner")"#).corrected(&[E20]),
    record("temperature.demag.cold_limit_C", "ϑ_{cold}",
        r#"cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_cold_limit_C};
               else => {temperature.demag.inner_cold_limit_C})"#).corrected(&[E20]),
    // Both rings pass exactly when the minimum temperature is at or above the higher cold limit.
    record("temperature.demag.cold_check", "C_{cold}",
        r#"cases({temperature.demag.cold_limit_C} = "n/a" => "n/a (coercivity rises as the magnet cools)";
               {metal.min_temp_C} >= {temperature.demag.cold_limit_C} => "OK";
               else => "Below the cold demagnetization limit")"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_aligned_C", "ϑ_{cold,al}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_aligned_kA_m}, {temperature.demag.h_rev_aligned_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_pullout_C", "ϑ_{cold,po}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_pullout_kA_m}, {temperature.demag.h_rev_pullout_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_skipping_C", "ϑ_{cold,sk}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_likepole_kA_m}, {temperature.demag.h_rev_likepole_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),
    record("temperature.demag.cold_onset_single_ring_C", "ϑ_{cold,sr}",
        r#"cases([β_c] > 0 => 20 + frac([H_k] - {temperature.demag.h_rev_single_ring_kA_m}, {temperature.demag.h_rev_single_ring_kA_m} · [α_c] - [H_k] · [β_c]);
               else => "n/a")
           where [β_c] = cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_beta_per_C}; else => {temperature.demag.inner_beta_per_C}),
                 [α_c] = cases({temperature.demag.cold_ring} = "outer" => {model.outer_alpha_br_per_C}; else => {model.inner_alpha_br_per_C}),
                 [H_k] = {temperature.demag.knee_fraction} * cases({temperature.demag.cold_ring} = "outer" => {temperature.demag.outer_hcj20_kA_m};
                                                                  else => {temperature.demag.inner_hcj20_kA_m})"#).corrected(&[E20]),

    // --- Torque at the limits (decision A2-7: each ring's Br with its own coefficient) ---
    // With a positive beta and no rating there is no hot limit (+inf) and no torque at it.
    record("temperature.demag.torque_at_limit_Nm", "T_{mag}",
        r#"cases({temperature.demag.beta_used_per_C} > 0 and {temperature.demag.tmax_lib_C} = "n/a" => nan;
               else => {model.pullout_20C_Nm} * ((1 + {model.inner_alpha_br_per_C} * ({temperature.demag.magnet_limit_C} - 20))
                                              * (1 + {model.outer_alpha_br_per_C} * ({temperature.demag.magnet_limit_C} - 20))))"#).corrected(&[E20]),
    record("temperature.demag.torque_at_service_Nm", "T_{svc}", "{model.pullout_Nm}"),

    // --- The governing limit and what it is compared with (temperature::compute) ---
    record("temperature.adhesive.design_limit_C", "ϑ_{adh}",
        r#"table("adhesives", {temperature.adhesive.selected}, "design_limit_C")"#),
    record("temperature.adhesive.cure_C", "ϑ_{cure}", r#"table("adhesives", {temperature.adhesive.selected}, "cure_C")"#),
    record("temperature.summary.governing_limit_C", "ϑ_{gov}",
        "min({temperature.demag.magnet_limit_C}, {temperature.adhesive.design_limit_C})"),
    record("temperature.duty.hot_day_start_C", "ϑ_{hot}", "{temperature.duty.hot_ambient_C} + {temperature.duty.driving_rise_C}"),
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
//! The equation records, one file per batch (plan A-3). A batch adds its file here and
//! flips its chain to `Explained` in [`super::scope`].

pub mod torque;

use super::record::{Family, Record};

/// Every batch's plain records.
pub const RECORDS: &[&[Record]] = &[torque::RECORDS];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES];
```

with:

```rust
//! The equation records, one file per batch (plan A-3). A batch adds its file here and
//! flips its chain to `Explained` in [`super::scope`].

pub mod demagnetization;
pub mod torque;

use super::record::{Family, Record};

/// Every batch's plain records.
pub const RECORDS: &[&[Record]] = &[torque::RECORDS, demagnetization::RECORDS];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
//! word (`faceted`), which the typesetter writes with the choice's label in conditions.

/// (path, symbol markup), grouped by input group.
pub const SYMBOLS: &[(&str, &str)] = &[
    // coupling
    ("coupling.npole", "N"),
```

with:

```rust
//! word (`faceted`), which the typesetter writes with the choice's label in conditions.

/// (path, symbol markup), grouped by input group.
#[rustfmt::skip]
pub const SYMBOLS: &[(&str, &str)] = &[
    // coupling
    ("coupling.npole", "N"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("calibration.fea_torque2_Nm", "T_{3D,2}"),
    // materials
    ("materials.parts.back_iron", "material_{BI}"),
];
```

with:

```rust
    ("calibration.fea_torque2_Nm", "T_{3D,2}"),
    // materials
    ("materials.parts.back_iron", "material_{BI}"),
    // temperature
    ("temperature.duty.hot_ambient_C", "ϑ_{amb}"),
    ("temperature.duty.driving_rise_C", "Δϑ_{drive}"),
    ("temperature.demag.hcj20_kA_m", "H_{cj,in}"),
    ("temperature.demag.beta_hcj_per_C", "β_{in}"),
    ("temperature.demag.knee_fraction", "k_{knee}"),
    ("temperature.demag.design_margin_C", "Δϑ_m"),
    ("temperature.demag.h_rev_aligned_kA_m", "H_{al}"),
    ("temperature.demag.h_rev_pullout_kA_m", "H_{po}"),
    ("temperature.demag.h_rev_likepole_kA_m", "H_{sk}"),
    ("temperature.demag.h_rev_single_ring_kA_m", "H_{sr}"),
    ("temperature.demag.coercivity_source", "source"),
    ("temperature.adhesive.selected", "adhesive"),
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
    v.push((
        "axial 8 mm",
        vec![("coupling.magnets.axial_length_mm", Value::Num(8.0))],
    ));
    v.push((
        "everything",
```

with:

```rust
    v.push((
        "axial 8 mm",
        vec![("coupling.magnets.axial_length_mm", Value::Num(8.0))],
    ));
    // E20's cold side from the inputs: hard ferrite's positive beta for both rings, first on
    // rated magnets (both rings in the grade mode, bonded NdFeB and ferrite, so the rings' Br
    // coefficients and densities differ and either ring can hold the higher cold limit), then
    // on manual magnets with no grade and so no rating (no hot limit).
    v.push((
        "ferrite inputs",
        vec![
            ("temperature.demag.coercivity_source", Value::Int(0)),
            ("temperature.demag.beta_hcj_per_C", Value::Num(0.0035)),
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("Bonded_NdFeB_BCN19")),
            ("coupling.magnets.part_outer", text("")),
            ("coupling.magnets.grade_outer", text("Y30")),
        ],
    ));
    v.push((
        "unrated ferrite",
        vec![
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.part_outer", text("")),
            ("temperature.demag.beta_hcj_per_C", Value::Num(0.0035)),
        ],
    ));
    // A second positive beta from the inputs with the rings swapped (ferrite inside, bonded
    // NdFeB outside), so every cold-side record sees two betas, and the outer ring a second
    // Br coefficient and density.
    v.push((
        "ferrite inputs, beta 0.002",
        vec![
            ("temperature.demag.coercivity_source", Value::Int(0)),
            ("temperature.demag.beta_hcj_per_C", Value::Num(0.002)),
            ("coupling.magnets.part_inner", text("")),
            ("coupling.magnets.grade_inner", text("Y30")),
            ("coupling.magnets.part_outer", text("")),
            ("coupling.magnets.grade_outer", text("Bonded_NdFeB_BCN19")),
        ],
    ));
    v.push((
        "everything",
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

- [ ] **Step 4: Run the drift guard and the structure tests**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 11 passed; 0 failed; 2 ignored`: the guard now evaluates 166 equations at 50,866 input sets (14 augmentations), every `cases` arm taken and every value term varied.

- [ ] **Step 5: Physics review of the demagnetization block (session model)**

Dispatch the physics reviewer on the **session model** (omit `model`, comment `// session model: physics`; CLAUDE.md section 5) with the prompt below, and give it the sheet this command prints:

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E "temperature\.demag|governing|adhesive\.|hot_day|_grade|_tmax"
```

Expected: one Markdown table row per record of the batch: path, workbook cell (or `Rust-only`), the rendered formula and the corrections it embodies.

Reviewer prompt: "Review the demagnetization records of the magcoupling equation explorer (the sheet below) against the M1 audit (docs/analyses/2026-09-29-magcoupling-math-audit.md) row M13 and its onset-calibration ruling, Addendum A decisions 3, 18 and 19 (E20) and A-1 decision A13 (both rings; the cold side from the ring with the higher cold limit). The drift guard already proves each formula reproduces the engine; review what numbers cannot: (1) each symbol follows the conventions of `src/engine/explain/symbols.rs` (T torque, σ shear stress, ϑ temperature, φ angle, τ_p pole pitch) and no two paths share one; (2) each formula is the physics the M1 audit confirmed for that cell, stated one step from its terms; (3) each record names the corrections it embodies; (4) these reasoning claims hold: each ring's limit binds its own knee, reference field and offset; with a positive beta the hot onsets are infinite and the rating is the hot limit; `cold_check` passes exactly when the minimum temperature is at or above the higher cold limit (both rings pass); the shown block is the ring with the lower magnet limit, the inner ring on a tie. Answer APPROVED, or list each finding with the record and the fix."

On APPROVED, continue. On findings: fix each in the batch's records file (a symbol in `symbols.rs`), re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain` until green, and send the changed rows back to the same reviewer; the task does not commit until the reviewer approves. A finding that would change the engine is out of scope: stop and escalate.

- [ ] **Step 6: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task8.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task8.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task8.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 7: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/records/demagnetization.rs magcoupling-rs/src/engine/explain/records/mod.rs magcoupling-rs/src/engine/explain/symbols.rs magcoupling-rs/tests/explain.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): explorer batch 2, the demagnetization block and the governing limit

Each ring's E20 block, the block the sheet shows, ferrite's cold side, the
adhesive limit and the governing limit, proven by the drift guard (three ferrite
augmentations reach the positive-beta branches at two betas). The chain flips with slip
heating (its margins read the peak magnet temperature).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 9: Batch 3: slip heating, the thermal network and their closure; demagnetization and slip heating explained

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Step 5, the physics review, runs on the session model.

This batch completes the closure Task 8 began, so both chains flip: the red step is the flip itself, and the structure
tests then name every path still missing a record. **Slip losses** (Temperature design C109 to C134): the opposite ring's field sweeps
each part at the field frequency f_e = p n / 60; solid steel is skin-limited (P/A = σ ω_e² B² δ / (4 k²), δ the skin depth);
with no back iron the hub, cup and web are aluminium, resistance-limited, in the free-space field (E17's form T1, with the
thin-conductor end factor); thin shells lose σ (ω_s r B)²/2 per volume (ω_s the slip speed, ω_e = p ω_s); the magnets σ ω_e² B² w²/24 per volume. The engine has a third
hub branch (half the steel-circuit field for an aluminium hub in a steel cup) that needs E9 off; with every correction on the cup is
aluminium whenever the hub is, so the record states two branches and says why. **Thermal network**: one heat capacity C = Σ m c (E15:
an aluminium hub, cup and boss at the body material's specific heat) and one conductance G; the time to the limit solves the
first-order rise (E12: a start at or above the limit reaches it at once; "never" when the steady temperature stays below it). The
mass model, the axial housing in effect (A-2 decision A2-8) and the retainers are closure: the heat capacity reads them. Two
augmentations: the sleeve, liner and cap picks (Ti and acetal) take the library branches of their materials, and "E17 fields"
sets E17's three free-space fields (Rust-only inputs no differential case varies) off their defaults under an aluminium back iron,
so the drift guard sees the hub, cup and web losses read each at two values. The losses are written with stacked fractions
(`frac(σ ω_e² B² δ, 4 k²)`), never an inline `a / b` as a factor, and the magnets' loss reads each block dimension in m only.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs` (flip `demagnetization` and `slip_heating`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/slip_heating.rs` (84 records)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` (the batch's file)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs` (54 input symbols)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (the sleeve and cap augmentation and the E17 fields one)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the demagnetization and slip-heating rows)

**Interfaces:**
- Consumes: Task 8's records (the governing limit, the hot-day start, the demagnetization block); Task 7's materials in effect and tables (`back_iron`, `sleeve_liner`, `cap_housing`, `aluminium`, `grades.density_g_mm3`); A-2's `housing.*` axial dimensions.
- Produces: records for the materials in effect (`temperature.slip_loss.steel_sigma_S_m`, `steel_mu_r`, `cap_sigma_S_m`, `temperature.thermal.steel_c`, `materials.{steel_density_g_mm3, body_*, sleeve_*, cap_density_g_mm3, cap_c_J_kgK}`), the duty block, the seven losses and their total, drag, used and high powers, the mass model (`mass.{magnets_g, cup_g, hub_g, boss_g}`, `model.{inner,outer}_magnet_density_g_mm3`, `model.{pocket_corner_radius_mm, cup_od_mm, hub_wall_mm, outer_back_apothem_mm}`), the axial housing in effect (`housing.{hub_length_mm, cup_depth_mm, retainer_span_mm}`), the retainers (`retainers.{sleeve_id_mm, sleeve_od_mm, liner_od_mm, liner_id_mm, endplate_od_mm, retainers_g, cap_g, endplates_g}`), the thermal network (heat capacity, time constant, steady rises, time to the limit, critical drag, the fault-trip temperature), the slip life rows the peak reads, `temperature.magnet_life.peak_C`, the metal-design slip rows and the demagnetization margins and summary rows.

- [ ] **Step 1: Write the failing test: flip the two chains**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs`, replace:

```rust
    },
    Chain {
        id: "demagnetization",
        status: Status::Pending,
        paths: &[
            "temperature.demag.h_ref_kA_m",
            "temperature.demag.t_ref_model_C",
```

with:

```rust
    },
    Chain {
        id: "demagnetization",
        status: Status::Explained,
        paths: &[
            "temperature.demag.h_ref_kA_m",
            "temperature.demag.t_ref_model_C",
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs`, replace:

```rust
    },
    Chain {
        id: "slip_heating",
        status: Status::Pending,
        paths: &[
            "temperature.slip_loss.steel_sigma_S_m",
            "temperature.slip_loss.steel_mu_r",
```

with:

```rust
    },
    Chain {
        id: "slip_heating",
        status: Status::Explained,
        paths: &[
            "temperature.slip_loss.steel_sigma_S_m",
            "temperature.slip_loss.steel_mu_r",
```

- [ ] **Step 2: Run the structure tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain explained_chains 2>&1 | grep -E "panicked|slip_heating: " | head -4
```

Expected: `explained_chains_have_every_record_and_scope_paths_exist` panics listing every path without a record, starting `["demagnetization: temperature.summary.onset_aligned_C", "demagnetization: temperature.summary.onset_pullout_C", ...` and continuing with the slip-heating paths (`"slip_heating: temperature.slip_loss.steel_sigma_S_m"`, ...).

- [ ] **Step 3: Write the records, their symbols and the augmentation**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override, the coercivity from the inputs with two ferrite betas), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override, the coercivity from the inputs with two ferrite betas, the sleeve and cap materials, E17's free-space fields), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| Chain (decision 31) | Records | Status |
|---|---|---|
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | records written; explained with the slip-heating records (its margins read the peak magnet temperature) |
| slip heating | `records/slip_heating.rs` | pending |
| temperature | `records/temperature.rs` | pending |
| clamps | `records/clamps.rs` | pending |
| dashboard | `records/dashboard.rs` | pending |
```

with:

```markdown
| Chain (decision 31) | Records | Status |
|---|---|---|
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | explained |
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | pending |
| clamps | `records/clamps.rs` | pending |
| dashboard | `records/dashboard.rs` | pending |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
//! flips its chain to `Explained` in [`super::scope`].

pub mod demagnetization;
pub mod torque;

use super::record::{Family, Record};

/// Every batch's plain records.
pub const RECORDS: &[&[Record]] = &[torque::RECORDS, demagnetization::RECORDS];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES];
```

with:

```rust
//! flips its chain to `Explained` in [`super::scope`].

pub mod demagnetization;
pub mod slip_heating;
pub mod torque;

use super::record::{Family, Record};

/// Every batch's plain records.
pub const RECORDS: &[&[Record]] = &[
    torque::RECORDS,
    demagnetization::RECORDS,
    slip_heating::RECORDS,
];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES];
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/slip_heating.rs` with exactly this content:

```rust
//! Slip heating (plan A-3 batch 3): the eddy-current slip losses (Temperature design C109 to
//! C134, corrections E5, E17), the heat capacity and thermal time constant (C137 to C161, E15),
//! the time and drag to the limit (E12), the metal-design slip rows, and what they need to
//! reach inputs: the materials in effect (Addendum A5), the mass model (C110 to C114, E8, E9)
//! with the axial housing (decision A2-8), the sleeve, liner, cap and endplates (Metal design
//! C175 to C181, E8), and the peak magnet temperature the demagnetization margins read.
//!
//! Each formula is transcribed from the engine function named beside it, with every approved
//! correction on (`temperature.rs`, `model.rs`, `metal_design.rs`, `housing.rs`,
//! `material_library.rs`).
//!
//! Symbols: P power, ω_e the field's angular frequency (p ω_s), ω_s the slip speed, σ conductivity, ρ density, c specific heat, m mass.

use crate::engine::deviations::DeviationId::{E5, E8, E9, E12, E15, E17};
use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- The materials in effect (material_library::resolve) ---
    // A ferromagnetic pick other than the default supplies the steel values; the default and a
    // non-ferromagnetic pick keep the Materials inputs.
    record("temperature.slip_loss.steel_sigma_S_m", "σ_{st}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 1
                   => table("back_iron", {materials.parts.back_iron}, "sigma_S_m");
               else => {materials.steel.conductivity_S_m})"#),
    record("temperature.slip_loss.steel_mu_r", "μ_r", "{materials.steel.mu_r_incremental}"),
    record("temperature.thermal.steel_c", "c_{st}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 1
                   => table("back_iron", {materials.parts.back_iron}, "cp_J_kgK");
               else => {materials.steel.specific_heat_J_kgK})"#),
    record("materials.steel_density_g_mm3", "ρ_{st}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 1
                   => table("back_iron", {materials.parts.back_iron}, "density_g_mm3");
               else => {metal.steel_density_g_mm3})"#),
    // Without back iron the hub, cup and boss are the workbook's 6061-T6, or a
    // non-ferromagnetic pick (A5: the pick becomes that material).
    record("materials.body_sigma_S_m", "σ_{body}",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => table("back_iron", {materials.parts.back_iron}, "sigma_S_m");
               else => table("aluminium", "6061-T6", "conductivity_S_m"))"#),
    record("materials.body_density_g_mm3", "ρ_{body}",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => table("back_iron", {materials.parts.back_iron}, "density_g_mm3");
               else => {metal.al_density_g_mm3})"#),
    record("materials.body_c_J_kgK", "c_{body}",
        r#"cases(table("back_iron", {materials.parts.back_iron}, "ferromagnetic") = 0 => table("back_iron", {materials.parts.back_iron}, "cp_J_kgK");
               else => {temperature.thermal.c_aluminium})"#),
    record("materials.sleeve_sigma_S_m", "σ_{sl}",
        r#"cases({materials.parts.sleeve_liner} = 1 => {temperature.slip_loss.sigma_316_S_m};
               else => table("sleeve_liner", {materials.parts.sleeve_liner}, "sigma_S_m"))"#),
    record("materials.sleeve_density_g_mm3", "ρ_{sl}",
        r#"cases({materials.parts.sleeve_liner} = 1 => {metal.sleeve_density_g_mm3};
               else => table("sleeve_liner", {materials.parts.sleeve_liner}, "density_g_mm3"))"#),
    record("materials.sleeve_c_J_kgK", "c_{sl}",
        r#"cases({materials.parts.sleeve_liner} = 1 => {temperature.thermal.c_316};
               else => table("sleeve_liner", {materials.parts.sleeve_liner}, "cp_J_kgK"))"#),
    record("temperature.slip_loss.cap_sigma_S_m", "σ_{cap,eff}",
        r#"cases({materials.parts.cap_housing} = 1 => table("aluminium", "6061-T6", "conductivity_S_m");
               else => table("cap_housing", {materials.parts.cap_housing}, "sigma_S_m"))"#),
    record("materials.cap_density_g_mm3", "ρ_{cap,eff}",
        r#"cases({materials.parts.cap_housing} = 1 => {metal.al_density_g_mm3};
               else => table("cap_housing", {materials.parts.cap_housing}, "density_g_mm3"))"#),
    record("materials.cap_c_J_kgK", "c_{cap,eff}",
        r#"cases({materials.parts.cap_housing} = 1 => {temperature.thermal.c_aluminium};
               else => table("cap_housing", {materials.parts.cap_housing}, "cp_J_kgK"))"#),

    // --- Duty (temperature::compute) ---
    record("temperature.duty.slip_rpm", "n_{slip}", "{metal.slip_rpm}"),
    record("temperature.duty.slip_rad_s", "ω_s", "{temperature.duty.slip_rpm} * 2 * π / 60"),
    record("temperature.duty.pole_pairs", "p", "{coupling.npole} / 2"),
    // The opposite ring's field sweeps each part p times per slip revolution.
    record("temperature.duty.field_freq_Hz", "f_e", "{temperature.duty.pole_pairs} * {temperature.duty.slip_rpm} / 60"),
    record("temperature.duty.field_omega_rad_s", "ω_e", "2 * π * {temperature.duty.field_freq_Hz}"),

    // --- Geometry the losses read ---
    record("model.hub_wall_mm", "t_{hub}", "{coupling.inner_back_apothem_mm} - {metal.bond_inner_mm} - {coupling.bore_mm} / 2"),
    record("model.outer_back_apothem_mm", "A_{back}", "{model.outer_face_apothem_mm} + {model.outer_thickness_mm}"),
    record("temperature.adhesive.inner_mid_radius_mm", "r_{mid}", "{coupling.inner_back_apothem_mm} + {model.inner_thickness_mm} / 2"),
    // metal_design::retainers: E8, the sleeve clears the inner corner radius C55.
    record("retainers.sleeve_id_mm", "D_{sl,i}", "2 * ({model.inner_corner_radius_mm} + {metal.sleeve_bedding_mm})").corrected(&[E8]),
    record("retainers.sleeve_od_mm", "D_{sl,o}", "{retainers.sleeve_id_mm} + 2 * {metal.sleeve_mm}"),
    record("retainers.liner_od_mm", "D_{ln,o}", "2 * ({model.outer_face_apothem_mm} - {metal.liner_bedding_mm})"),
    record("retainers.liner_id_mm", "D_{ln,i}", "{retainers.liner_od_mm} - 2 * {metal.liner_mm}"),
    record("retainers.endplate_od_mm", "D_{ep}", "{retainers.sleeve_id_mm}"),

    // --- Slip losses (temperature::compute) ---
    // Steel skin depth at the field frequency: δ = √(2 / (ω_e μ0 μ_r σ)).
    record("temperature.slip_loss.skin_depth_mm", "δ",
        "sqrt(frac(2, {temperature.duty.field_omega_rad_s} * {coupling.mu0} * {temperature.slip_loss.steel_mu_r} * {temperature.slip_loss.steel_sigma_S_m})) * 1000"),
    // Solid steel is skin-limited; with no back iron the hub is aluminium (resistance-limited,
    // E17's low-Reynolds form T1 with the free-space field and the thin-conductor end factor).
    // The engine's third branch, half the steel-circuit field for an aluminium hub in a steel
    // cup, needs E9 off: with every correction on the cup is aluminium whenever the hub is.
    record("temperature.slip_loss.hub_W", "P_{hub}",
        "cases({materials.circuit_backiron} != 1
                 => frac({temperature.slip_loss.end_factor} * {materials.body_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2
                         * {temperature.slip_loss.b_hub_free_T}^2, 2 * [k]^2) * frac(1 - exp(-2 * [k] * {model.hub_wall_mm|m}), 2 * [k])
                    * 2 * π * [r] * {model.active_length_mm|m};
               else => frac({temperature.slip_loss.steel_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.b_hub_T}^2
                         * {temperature.slip_loss.skin_depth_mm|m}, 4 * [k]^2) * 2 * π * [r] * {model.active_length_mm|m})
         where [r] = {coupling.inner_back_apothem_mm|m} - {metal.bond_inner_mm|m},
               [k] = {temperature.duty.pole_pairs} / [r]").corrected(&[E17]),
    record("temperature.slip_loss.cup_W", "P_{cup}",
        "cases({materials.circuit_backiron} != 1
                 => frac({temperature.slip_loss.end_factor} * {materials.body_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2
                         * {temperature.slip_loss.b_cup_free_T}^2, 2 * [k]^2) * frac(1 - exp(-2 * [k] * {metal.cup_wall_corner_mm|m}), 2 * [k])
                    * 2 * π * [r] * {model.active_length_mm|m};
               else => frac({temperature.slip_loss.steel_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.b_cup_T}^2
                         * {temperature.slip_loss.skin_depth_mm|m}, 4 * [k]^2) * 2 * π * [r] * {model.active_length_mm|m})
         where [r] = {model.outer_back_apothem_mm|m} + {metal.bond_outer_mm|m},
               [k] = {temperature.duty.pole_pairs} / [r]").corrected(&[E17]),
    // The rear web sees the end field ∫B² dA (E5: the steel-surface value); (r_mid / p)² replaces 1/k².
    record("temperature.slip_loss.web_W", "P_{web}",
        "cases({materials.circuit_backiron} != 1
                 => frac({temperature.slip_loss.end_factor} * {materials.body_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2, 2)
                    * ({temperature.adhesive.inner_mid_radius_mm|m} / {temperature.duty.pole_pairs})^2
                    * frac(1 - exp(-2 * [k] * {metal.web_mm|m}), 2 * [k]) * {temperature.slip_loss.web_integral_free_T2m2};
               else => frac({temperature.slip_loss.steel_sigma_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.skin_depth_mm|m}, 4)
                    * ({temperature.adhesive.inner_mid_radius_mm|m} / {temperature.duty.pole_pairs})^2 * {temperature.slip_loss.web_integral_T2m2})
         where [k] = {temperature.duty.pole_pairs} / {temperature.adhesive.inner_mid_radius_mm|m}").corrected(&[E5, E17]),
    // Thin shells: the loss per volume σ (ω_s r B)² / 2, times the end factor.
    record("temperature.slip_loss.sleeve_W", "P_{sl}",
        "frac({temperature.slip_loss.end_factor} * {materials.sleeve_sigma_S_m} * {metal.sleeve_mm|m} * ({temperature.duty.slip_rad_s} * [r])^2
              * {temperature.slip_loss.b_sleeve_T}^2, 2) * 2 * π * [r] * {model.active_length_mm|m}
         where [r] = frac({retainers.sleeve_id_mm|m} + {retainers.sleeve_od_mm|m}, 4)"),
    record("temperature.slip_loss.liner_W", "P_{ln}",
        "frac({temperature.slip_loss.end_factor} * {materials.sleeve_sigma_S_m} * {metal.liner_mm|m} * ({temperature.duty.slip_rad_s} * [r])^2
              * {temperature.slip_loss.b_liner_T}^2, 2) * 2 * π * [r] * {model.active_length_mm|m}
         where [r] = frac({retainers.liner_od_mm|m} + {retainers.liner_id_mm|m}, 4)"),
    record("temperature.slip_loss.cap_W", "P_{cap}",
        "{temperature.slip_loss.end_factor} * {temperature.slip_loss.cap_sigma_S_m} * {metal.cap_axial_mm|m}
         * {temperature.duty.slip_rad_s}^2 * {temperature.slip_loss.cap_integral_T2m4}"),
    // Eddy currents inside the blocks: σ ω_e² B² w² / 24 per volume, both rings.
    record("temperature.slip_loss.magnets_W", "P_{mag}",
        "frac({temperature.slip_loss.sigma_ndfeb_S_m} * {temperature.duty.field_omega_rad_s}^2 * {temperature.slip_loss.b_magnet_T}^2
              * {model.inner_width_mm|m}^2, 24) * ({model.inner_length_mm|m} * {model.inner_width_mm|m} * {model.inner_thickness_mm|m})
         * 2 * {coupling.npole}"),
    record("temperature.slip_loss.total_W", "P_{tot}",
        "{temperature.slip_loss.hub_W} + {temperature.slip_loss.cup_W} + {temperature.slip_loss.web_W} + {temperature.slip_loss.sleeve_W}
         + {temperature.slip_loss.liner_W} + {temperature.slip_loss.cap_W} + {temperature.slip_loss.magnets_W}"),
    record("temperature.slip_loss.drag_Nm", "T_{drag}", "frac({temperature.slip_loss.total_W}, {temperature.duty.slip_rad_s})"),
    // A bench drag replaces the estimate (and the high case) once entered.
    record("temperature.slip_loss.used_W", "P_{use}",
        "cases({metal.measured_drag_Nm} != none => {metal.measured_drag_Nm} * {temperature.duty.slip_rad_s};
               else => {temperature.slip_loss.total_W})"),
    record("temperature.slip_loss.high_W", "P_{hi}",
        "cases({metal.measured_drag_Nm} != none => {metal.measured_drag_Nm} * {temperature.duty.slip_rad_s};
               else => {temperature.slip_loss.total_W} * {temperature.slip_loss.high_multiplier})"),

    // --- The mass model the heat capacity reads (model::mass_estimate) ---
    // Decision A2-7: a grade-mode ring's density, else NdFeB's 7.5 g/cm³.
    record("model.inner_magnet_density_g_mm3", "ρ_{mag,i}",
        r#"cases(table("magnets", {coupling.magnets.part_inner}, "br_T") = none and [ρ_{grade}] != none => [ρ_{grade}]; else => 0.0075)
           where [ρ_{grade}] = table("grades", {coupling.magnets.grade_inner}, "density_g_mm3")"#),
    record("model.outer_magnet_density_g_mm3", "ρ_{mag,o}",
        r#"cases(table("magnets", {coupling.magnets.part_outer}, "br_T") = none and [ρ_{grade}] != none => [ρ_{grade}]; else => 0.0075)
           where [ρ_{grade}] = table("grades", {coupling.magnets.grade_outer}, "density_g_mm3")"#),
    record("mass.magnets_g", "m_{mag}",
        "cases({model.inner_magnet_density_g_mm3} = {model.outer_magnet_density_g_mm3}
                 => {coupling.npole} * ([V_i] + [V_o]) * {model.inner_magnet_density_g_mm3};
               else => {coupling.npole} * ([V_i] * {model.inner_magnet_density_g_mm3} + [V_o] * {model.outer_magnet_density_g_mm3}))
         where [V_i] = {model.inner_length_mm} * {model.inner_width_mm} * {model.inner_thickness_mm},
               [V_o] = {model.outer_length_mm} * {model.outer_width_mm} * {model.outer_thickness_mm}"),
    record("model.pocket_corner_radius_mm", "r_{pocket}",
        "cases({coupling.faceted} = 1 => frac({model.outer_back_apothem_mm} + {metal.bond_outer_mm}, cos(π / {coupling.npole}));
               else => {model.outer_back_apothem_mm} + {metal.bond_outer_mm})"),
    record("model.cup_od_mm", "D_{cup}", "2 * ({model.pocket_corner_radius_mm} + {metal.cup_wall_corner_mm})"),
    // E9: with no back iron the cup and boss are aluminium (the body material), as the hub is.
    record("mass.cup_g", "m_{cup}",
        "((π * ({model.cup_od_mm} / 2)^2 - [A_{cav}]) * {housing.cup_depth_mm}
          + π * (({model.cup_od_mm} / 2)^2 - ({coupling.bore_mm} / 2)^2) * {metal.web_mm}) * [ρ]
         where [A_{cav}] = cases({coupling.faceted} != 1 => π * ({model.outer_back_apothem_mm} + {metal.bond_outer_mm})^2;
                                 else => {coupling.npole} * ({model.outer_back_apothem_mm} + {metal.bond_outer_mm})^2 * tan(π / {coupling.npole})),
               [ρ] = cases({materials.circuit_backiron} != 1 => {materials.body_density_g_mm3}; else => {materials.steel_density_g_mm3})")
        .corrected(&[E8, E9]),
    record("mass.hub_g", "m_{hub}",
        "([A_{hub}] - π * ({coupling.bore_mm} / 2)^2) * {housing.hub_length_mm} * [ρ]
         where [A_{hub}] = cases({coupling.faceted} = 1 => {coupling.npole} * ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm})^2 * tan(π / {coupling.npole});
                                 else => π * ({coupling.inner_back_apothem_mm} - {metal.bond_inner_mm})^2),
               [ρ] = cases({materials.circuit_backiron} != 1 => {materials.body_density_g_mm3}; else => {materials.steel_density_g_mm3})"),
    record("mass.boss_g", "m_{boss}",
        "π * (({metal.boss_od_mm} / 2)^2 - ({coupling.bore_mm} / 2)^2) * {metal.boss_length_mm}
         * cases({materials.circuit_backiron} != 1 => {materials.body_density_g_mm3}; else => {materials.steel_density_g_mm3})")
        .corrected(&[E9]),
    // housing::axial_housing (decision A2-8): with the axial length override set, a dimension
    // that bounds a ring follows the ring's length change, and is never shorter than the ring.
    record("housing.hub_length_mm", "L_{hub}",
        r#"cases({coupling.magnets.axial_length_mm} != none
                   => max({metal.hub_length_mm} + ({coupling.magnets.axial_length_mm} - [L_{i,own}]), {coupling.magnets.axial_length_mm});
               else => {metal.hub_length_mm})
           where [L_{i,own}] = cases(table("magnets", {coupling.magnets.part_inner}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_inner}, "length_mm");
                                     else => {coupling.magnets.manual_inner_length_mm})"#),
    record("housing.cup_depth_mm", "L_{cav}",
        r#"cases({coupling.magnets.axial_length_mm} != none
                   => max({metal.cup_depth_mm} + ({coupling.magnets.axial_length_mm} - [L_{o,own}]), {coupling.magnets.axial_length_mm});
               else => {metal.cup_depth_mm})
           where [L_{o,own}] = cases(table("magnets", {coupling.magnets.part_outer}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_outer}, "length_mm");
                                     else => {coupling.magnets.manual_outer_length_mm})"#),
    record("housing.retainer_span_mm", "L_{ret}",
        r#"cases({coupling.magnets.axial_length_mm} != none
                   => max({metal.retainer_span_mm} + ({coupling.magnets.axial_length_mm} - max([L_{i,own}], [L_{o,own}])), {coupling.magnets.axial_length_mm});
               else => {metal.retainer_span_mm})
           where [L_{i,own}] = cases(table("magnets", {coupling.magnets.part_inner}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_inner}, "length_mm");
                                     else => {coupling.magnets.manual_inner_length_mm}),
                 [L_{o,own}] = cases(table("magnets", {coupling.magnets.part_outer}, "length_mm") != none
                                       => table("magnets", {coupling.magnets.part_outer}, "length_mm");
                                     else => {coupling.magnets.manual_outer_length_mm})"#),
    record("retainers.retainers_g", "m_{ret}",
        "frac(π, 4) * ({retainers.sleeve_od_mm}^2 - {retainers.sleeve_id_mm}^2 + {retainers.liner_od_mm}^2 - {retainers.liner_id_mm}^2)
         * {housing.retainer_span_mm} * {materials.sleeve_density_g_mm3}"),
    record("retainers.cap_g", "m_{cap}",
        "(frac(π, 4) * ({metal.cap_od_mm}^2 - {retainers.liner_id_mm}^2) * {metal.cap_axial_mm}
          + frac(π, 4) * ({metal.cap_od_mm}^2 - {metal.cap_thread_dia_mm}^2) * {metal.cap_thread_engagement_mm}) * {materials.cap_density_g_mm3}"),
    record("retainers.endplates_g", "m_{ep}",
        "frac(π, 4) * (({retainers.endplate_od_mm}^2 - {coupling.bore_mm}^2) * {metal.front_endplate_mm}
                  + ({retainers.endplate_od_mm}^2 - {metal.rear_endplate_hole_mm}^2) * {metal.rear_endplate_mm}) * {materials.sleeve_density_g_mm3}"),

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
    record("temperature.thermal.time_constant_s", "τ_{th}", "frac({temperature.thermal.heat_capacity_J_K}, {temperature.thermal.conductance_W_K})"),
    record("temperature.thermal.t95_s", "t_{95}", "3 * {temperature.thermal.time_constant_s}"),
    record("temperature.thermal.rev_per_tau", "N_{τ}", "{temperature.thermal.time_constant_s} * ({temperature.duty.slip_rpm} / 60)"),
    record("temperature.thermal.rev95", "N_{95}", "3 * {temperature.thermal.time_constant_s} * ({temperature.duty.slip_rpm} / 60)"),
    record("temperature.thermal.steady_rise_est_C", "Δϑ_{ss,est}", "frac({temperature.slip_loss.used_W}, {temperature.thermal.conductance_W_K})"),
    record("temperature.thermal.steady_rise_high_C", "Δϑ_{ss,hi}", "frac({temperature.slip_loss.high_W}, {temperature.thermal.conductance_W_K})"),
    record("temperature.thermal.steady_est_C", "ϑ_{ss,est}", "{temperature.duty.hot_day_start_C} + {temperature.thermal.steady_rise_est_C}"),
    record("temperature.thermal.steady_high_C", "ϑ_{ss,hi}", "{temperature.duty.hot_day_start_C} + {temperature.thermal.steady_rise_high_C}"),
    // First-order heating toward the steady temperature: ϑ(t) = ϑ_hot + Δϑ_ss (1 − e^(−t/τ)).
    // E12: a start at or above the limit reaches it at once.
    record("temperature.thermal.time_to_limit_high", "t_{lim,hi}",
        r#"cases({temperature.duty.hot_day_start_C} >= {temperature.summary.governing_limit_C} => 0;
               {temperature.thermal.steady_high_C} <= {temperature.summary.governing_limit_C} => "never: steady state stays below the limit";
               else => -{temperature.thermal.time_constant_s}
                       * ln(1 - frac({temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}, {temperature.thermal.steady_rise_high_C})))"#)
        .corrected(&[E12]),
    record("temperature.thermal.time_to_limit_est", "t_{lim,est}",
        r#"cases({temperature.duty.hot_day_start_C} >= {temperature.summary.governing_limit_C} => 0;
               {temperature.thermal.steady_est_C} <= {temperature.summary.governing_limit_C} => "never: steady state stays below the limit";
               else => -{temperature.thermal.time_constant_s}
                       * ln(1 - frac({temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}, {temperature.thermal.steady_rise_est_C})))"#)
        .corrected(&[E12]),
    record("temperature.thermal.rotations_to_limit_high", "N_{lim,hi}",
        r#"cases({temperature.thermal.time_to_limit_high} != "never: steady state stays below the limit"
                   => {temperature.thermal.time_to_limit_high} * ({temperature.duty.slip_rpm} / 60);
               else => "never")"#),
    // E12: slip never cools the magnets, so the critical drag is not negative.
    record("temperature.thermal.critical_drag_Nm", "T_{crit}",
        "max(0, {temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}) * {temperature.thermal.conductance_W_K}
         / {temperature.duty.slip_rad_s}").corrected(&[E12]),
    record("temperature.summary.time_to_limit_high", "t_{lim,hi}^{sum}", "{temperature.thermal.time_to_limit_high}"),
    record("temperature.summary.critical_drag_Nm", "T_{crit}^{sum}", "{temperature.thermal.critical_drag_Nm}"),
    record("temperature.thermal.temp_at_fault_C", "ϑ_{fault}",
        "{temperature.duty.hot_day_start_C} + {temperature.thermal.steady_rise_high_C}
         * (1 - exp(-{temperature.duty.fault_trip_s} / {temperature.thermal.time_constant_s}))"),

    // --- Slip life and the peak magnet temperature ---
    record("temperature.slip_life.rise_per_event_high_C", "Δϑ_{ev,hi}",
        "{temperature.slip_loss.high_W} * {metal.slip_event_s} / {temperature.thermal.heat_capacity_J_K}"),
    record("temperature.slip_life.slip_hours", "t_{slip}", "{metal.life_events} * {metal.slip_event_s} / 3600"),
    record("temperature.slip_life.slip_duty", "D_{slip}", "frac({temperature.slip_life.slip_hours}, {temperature.duty.life_hours})"),
    // The hot day, plus the fault-limited slip or one event, whichever is hotter, plus the
    // average heating of the life slip duty.
    record("temperature.magnet_life.peak_C", "ϑ_{peak}",
        "max({temperature.thermal.temp_at_fault_C}, {temperature.duty.hot_day_start_C} + {temperature.slip_life.rise_per_event_high_C})
         + {temperature.slip_life.slip_duty} * {temperature.thermal.steady_rise_high_C}"),

    // --- Metal design slip rows (metal_design::compute) ---
    record("metal.slip_freq_Hz", "f_{slip}", "frac({coupling.npole}, 2) * frac({metal.slip_rpm}, 60)"),
    record("metal.slip_loss_W", "P_{slip,MD}",
        r#"cases({metal.measured_drag_Nm} != none => {metal.measured_drag_Nm} * 2 * π * {metal.slip_rpm} / 60; else => "not measured")"#),
    record("metal.slip_energy_J", "E_{slip,MD}",
        r#"cases({metal.measured_drag_Nm} != none => {metal.slip_loss_W} * {metal.slip_event_s}; else => "not measured")"#),

    // --- The demagnetization margins and the summary rows (temperature::compute) ---
    record("temperature.summary.service_max_C", "ϑ_{svc}", "{coupling.op_temp_C}"),
    record("temperature.summary.onset_aligned_C", "ϑ_{al}^{sum}", "{temperature.demag.onset_aligned_C}"),
    record("temperature.summary.onset_pullout_C", "ϑ_{po}^{sum}", "{temperature.demag.onset_pullout_C}"),
    record("temperature.summary.onset_skipping_C", "ϑ_{sk}^{sum}", "{temperature.demag.onset_skipping_C}"),
    record("temperature.summary.margin_service_C", "Δϑ_{svc}",
        "{temperature.summary.governing_limit_C} - {temperature.summary.service_max_C}"),
    record("temperature.summary.margin_hot_day_C", "Δϑ_{hot}",
        "{temperature.summary.governing_limit_C} - {temperature.duty.hot_day_start_C}"),
    record("temperature.summary.cure_margin_C", "Δϑ_{cure}", "{temperature.demag.onset_single_ring_C} - {temperature.adhesive.cure_C}"),
    record("temperature.magnet_life.margin_onset_C", "Δϑ_{sk}", "{temperature.demag.onset_skipping_C} - {temperature.magnet_life.peak_C}"),
    record("temperature.magnet_life.margin_limit_C", "Δϑ_{mag}", "{temperature.demag.magnet_limit_C} - {temperature.magnet_life.peak_C}"),
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("coupling.mu0", "μ_0"),
    ("coupling.gear_ratio", "i_g"),
    ("coupling.gear_efficiency", "η_g"),
    ("coupling.magnets.part_inner", "part_i"),
    ("coupling.magnets.part_outer", "part_o"),
    ("coupling.magnets.grade_inner", "grade_i"),
```

with:

```rust
    ("coupling.mu0", "μ_0"),
    ("coupling.gear_ratio", "i_g"),
    ("coupling.gear_efficiency", "η_g"),
    ("coupling.bore_mm", "d_{bore}"),
    ("coupling.magnets.part_inner", "part_i"),
    ("coupling.magnets.part_outer", "part_o"),
    ("coupling.magnets.grade_inner", "grade_i"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("metal.min_temp_C", "ϑ_{min}"),
    ("metal.variation", "v"),
    ("metal.required_min_Nm", "T_{req}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
```

with:

```rust
    ("metal.min_temp_C", "ϑ_{min}"),
    ("metal.variation", "v"),
    ("metal.required_min_Nm", "T_{req}"),
    ("metal.slip_rpm", "n_s"),
    ("metal.slip_event_s", "t_{ev}"),
    ("metal.life_events", "N_{ev}"),
    ("metal.measured_drag_Nm", "T_{drag,meas}"),
    ("metal.bond_inner_mm", "b_i"),
    ("metal.bond_outer_mm", "b_o"),
    ("metal.sleeve_mm", "t_{sl}"),
    ("metal.liner_mm", "t_{ln}"),
    ("metal.sleeve_bedding_mm", "g_{sl}"),
    ("metal.liner_bedding_mm", "g_{ln}"),
    ("metal.cup_wall_corner_mm", "t_{wall}"),
    ("metal.hub_length_mm", "L_{hub,in}"),
    ("metal.cup_depth_mm", "L_{cav,in}"),
    ("metal.retainer_span_mm", "L_{ret,in}"),
    ("metal.web_mm", "t_{web}"),
    ("metal.boss_length_mm", "L_{boss}"),
    ("metal.boss_od_mm", "D_{boss}"),
    ("metal.hardware_g", "m_{hw}"),
    ("metal.cap_axial_mm", "t_{cap}"),
    ("metal.cap_od_mm", "D_{cap}"),
    ("metal.cap_thread_dia_mm", "D_{thr}"),
    ("metal.cap_thread_engagement_mm", "L_{thr}"),
    ("metal.front_endplate_mm", "t_{ep,f}"),
    ("metal.rear_endplate_mm", "t_{ep,r}"),
    ("metal.rear_endplate_hole_mm", "d_{ep}"),
    ("metal.steel_density_g_mm3", "ρ_{st,in}"),
    ("metal.al_density_g_mm3", "ρ_{Al,in}"),
    ("metal.sleeve_density_g_mm3", "ρ_{316,in}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("calibration.fea_torque2_Nm", "T_{3D,2}"),
    // materials
    ("materials.parts.back_iron", "material_{BI}"),
    // temperature
    ("temperature.duty.hot_ambient_C", "ϑ_{amb}"),
    ("temperature.duty.driving_rise_C", "Δϑ_{drive}"),
    ("temperature.demag.hcj20_kA_m", "H_{cj,in}"),
    ("temperature.demag.beta_hcj_per_C", "β_{in}"),
    ("temperature.demag.knee_fraction", "k_{knee}"),
```

with:

```rust
    ("calibration.fea_torque2_Nm", "T_{3D,2}"),
    // materials
    ("materials.parts.back_iron", "material_{BI}"),
    ("materials.parts.sleeve_liner", "material_{SL}"),
    ("materials.parts.cap_housing", "material_{cap}"),
    ("materials.steel.conductivity_S_m", "σ_{st,in}"),
    ("materials.steel.mu_r_incremental", "μ_{r,in}"),
    ("materials.steel.specific_heat_J_kgK", "c_{st,in}"),
    // temperature
    ("temperature.duty.hot_ambient_C", "ϑ_{amb}"),
    ("temperature.duty.driving_rise_C", "Δϑ_{drive}"),
    ("temperature.duty.fault_trip_s", "t_{trip}"),
    ("temperature.duty.life_hours", "t_{life}"),
    ("temperature.demag.hcj20_kA_m", "H_{cj,in}"),
    ("temperature.demag.beta_hcj_per_C", "β_{in}"),
    ("temperature.demag.knee_fraction", "k_{knee}"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("temperature.demag.h_rev_single_ring_kA_m", "H_{sr}"),
    ("temperature.demag.coercivity_source", "source"),
    ("temperature.adhesive.selected", "adhesive"),
];
```

with:

```rust
    ("temperature.demag.h_rev_single_ring_kA_m", "H_{sr}"),
    ("temperature.demag.coercivity_source", "source"),
    ("temperature.adhesive.selected", "adhesive"),
    ("temperature.slip_loss.sigma_316_S_m", "σ_{316,in}"),
    ("temperature.slip_loss.sigma_ndfeb_S_m", "σ_{NdFeB}"),
    ("temperature.slip_loss.end_factor", "f_{thin}"),
    ("temperature.slip_loss.b_hub_T", "B_{hub}"),
    ("temperature.slip_loss.b_cup_T", "B_{cup}"),
    ("temperature.slip_loss.b_sleeve_T", "B_{sl}"),
    ("temperature.slip_loss.b_liner_T", "B_{ln}"),
    ("temperature.slip_loss.cap_integral_T2m4", "I_{cap}"),
    ("temperature.slip_loss.web_integral_T2m2", "I_{web}"),
    ("temperature.slip_loss.b_magnet_T", "B_{mag}"),
    ("temperature.slip_loss.high_multiplier", "k_{hi}"),
    ("temperature.slip_loss.b_hub_free_T", "B_{hub,free}"),
    ("temperature.slip_loss.b_cup_free_T", "B_{cup,free}"),
    ("temperature.slip_loss.web_integral_free_T2m2", "I_{web,free}"),
    ("temperature.thermal.c_ndfeb", "c_{NdFeB}"),
    ("temperature.thermal.c_316", "c_{316,in}"),
    ("temperature.thermal.c_aluminium", "c_{Al,in}"),
    ("temperature.thermal.conductance_W_K", "G"),
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
//! differential case (`tests/data/differential/*.json`, 3,391 cases, starting from the
//! corrected defaults); and each of those again under every **augmentation**, the Rust-only
//! inputs the Python generator never varies (the harmonic set, the back-iron material, the
//! grade mode, the axial override), without which the tau7-tau11 records would compare
//! 0 with 0. Anti-vacuity checks require every `cases` arm of every record to be taken, every
//! numeric record and every value term of every record to take two values, and the E7
//! angles to leave half a pitch.
//!
//! **Traceability (A3).** For each input (the assumptions first, as the spec asks, then every
//! input), at several design points: nudging it changes no explained result outside its
```

with:

```rust
//! differential case (`tests/data/differential/*.json`, 3,391 cases, starting from the
//! corrected defaults); and each of those again under every **augmentation**, the Rust-only
//! inputs the Python generator never varies (the harmonic set, the back-iron material, the
//! grade mode, the axial override, the coercivity from the inputs with two ferrite betas,
//! the sleeve and cap materials, E17's free-space fields), without which the tau7-tau11
//! records would compare 0 with 0 and the ferrite and material branches would never run.
//! Anti-vacuity checks require every `cases` arm of every record to be taken, every numeric
//! record and every value term of every record to take two values, and the E7 angles to
//! leave half a pitch.
//!
//! **Traceability (A3).** For each input (the assumptions first, as the spec asks, then every
//! input), at several design points: nudging it changes no explained result outside its
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
    ] {
        v.push((label, vec![("materials.parts.back_iron", Value::Int(code))]));
    }
    v.push((
        "grade mode",
        vec![
```

with:

```rust
    ] {
        v.push((label, vec![("materials.parts.back_iron", Value::Int(code))]));
    }
    // The sleeve, liner and cap picks (A5): each supplies its library values in place of the inputs.
    v.push((
        "sleeve Ti, cap acetal",
        vec![
            ("materials.parts.sleeve_liner", Value::Int(2)),
            ("materials.parts.cap_housing", Value::Int(3)),
        ],
    ));
    // E17's free-space fields (Rust-only inputs) off their defaults, with an aluminium back
    // iron so the free-space branches of the hub, cup and web losses read them.
    v.push((
        "E17 fields",
        vec![
            ("materials.parts.back_iron", Value::Int(8)),
            ("temperature.slip_loss.b_hub_free_T", Value::Num(0.1)),
            ("temperature.slip_loss.b_cup_free_T", Value::Num(0.12)),
            (
                "temperature.slip_loss.web_integral_free_T2m2",
                Value::Num(1e-5),
            ),
        ],
    ));
    v.push((
        "grade mode",
        vec![
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 11 passed; 0 failed; 2 ignored`: 250 equations at 57,648 input sets (16 augmentations).

- [ ] **Step 4: Physics review of slip heating (session model)**

Dispatch the physics reviewer on the **session model** (omit `model`, comment `// session model: physics`; CLAUDE.md section 5) with the prompt below, and give it the sheet this command prints:

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E "slip_loss|thermal\.|slip_life|magnet_life|mass\.|retainers\.|housing\.|materials\.|duty\.|summary\.|metal\.slip"
```

Expected: one Markdown table row per record of the batch: path, workbook cell (or `Rust-only`), the rendered formula and the corrections it embodies.

Reviewer prompt: "Review the slip-heating records of the magcoupling equation explorer (the sheet below) against the M1 audit (docs/analyses/2026-09-29-magcoupling-math-audit.md) rows M11 (thin-skin steel losses), M12 (magnet eddy loss), P1 (the shell end factor), E5 (the web field at the steel surface), E12 and E13, the placeholder table (conductance, driving rise, slip event), and Addendum A report section 5.5 with decisions 8 (E15), 10 to 13 (E17) and 9 (E16), and A-2 decision A2-8 (the axial housing). The drift guard already proves each formula reproduces the engine; review what numbers cannot: (1) each symbol follows the conventions of `src/engine/explain/symbols.rs` (T torque, σ shear stress, ϑ temperature, φ angle, τ_p pole pitch) and no two paths share one; (2) each formula is the physics the M1 audit confirmed for that cell, stated one step from its terms; (3) each record names the corrections it embodies; (4) these reasoning claims hold: the hub record's two branches cover every design with every correction on (the engine's third branch needs E9 off); the heat capacity prices an aluminium hub, cup and boss at the body material's specific heat; the peak magnet temperature is the hot day plus the hotter of the fault-limited slip and one event, plus the life-duty average. Answer APPROVED, or list each finding with the record and the fix."

On APPROVED, continue. On findings: fix each in the batch's records file (a symbol in `symbols.rs`), re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain` until green, and send the changed rows back to the same reviewer; the task does not commit until the reviewer approves. A finding that would change the engine is out of scope: stop and escalate.

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task9.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task9.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task9.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/scope.rs magcoupling-rs/src/engine/explain/records/slip_heating.rs magcoupling-rs/src/engine/explain/records/mod.rs magcoupling-rs/src/engine/explain/symbols.rs magcoupling-rs/tests/explain.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): explorer batch 3, slip heating and the thermal network

The eddy losses (E5, E17), the heat capacity (E15) and time constant, the time
and drag to the limit (E12), the mass model and axial housing they read, the
peak magnet temperature and the demagnetization margins. Demagnetization and
slip heating are explained: every path has a record and reaches inputs.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 10: Batch 4: temperature explained

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Step 5, the physics review, runs on the session model.

The temperature chain is Br(T) and torque against temperature (torque ∝ Br_i(ϑ) Br_o(ϑ), each ring with its own coefficient,
A-2 decision A2-7), the magnet ratings and their checks, and the summary: which limit governs, the margins and the verdict. The
Br(T) factor 1 + α (ϑ − 20) appears in each torque record: each record is one workbook cell and the factor is what a student should
see (the guard proves every copy). The verdict reads E20's cold check: OK needs margin at the hot-day start, both limits above the
peak, 10 °C of cure margin and no ring below its cold limit.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs` (flip `temperature`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/temperature.rs` (16 records)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` (the batch's file)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the temperature row)

**Interfaces:**
- Consumes: Tasks 8 and 9's records (the demagnetization block, the peak temperature, the margins, the governing limit).
- Produces: records for `model.alpha_br_per_C`, `model.{inner,outer}_temp_check`, `temperature.magnet_life.{torque_hot_day_Nm, torque_hot_day_check, torque_peak_Nm}`, `temperature.adhesive_life.{torque_peak_var_Nm, margin_C}`, `temperature.mismatch.cold_limit_C` and the summary rows `magnet_limit_C`, `adhesive_limit_C`, `governing_note`, `hot_day_start_C`, `torque_hot_day_Nm`, `torque_hot_day_note`, `verdict`.

- [ ] **Step 1: Write the failing test: flip the chain**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs`, replace:

```rust
    },
    Chain {
        id: "temperature",
        status: Status::Pending,
        paths: &[
            "calibration.br_test_T",
            "model.inner_br_T",
```

with:

```rust
    },
    Chain {
        id: "temperature",
        status: Status::Explained,
        paths: &[
            "calibration.br_test_T",
            "model.inner_br_T",
```

- [ ] **Step 2: Run the structure tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain explained_chains 2>&1 | grep -E "panicked|temperature: " | head -3
```

Expected: the test panics listing the temperature paths without a record, starting `["temperature: model.alpha_br_per_C", "temperature: temperature.summary.torque_hot_day_Nm", ...`.

- [ ] **Step 3: Write the records**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | explained |
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | pending |
| clamps | `records/clamps.rs` | pending |
| dashboard | `records/dashboard.rs` | pending |
```

with:

```markdown
| torque | `records/torque.rs` | explained |
| demagnetization | `records/demagnetization.rs` | explained |
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | pending |
| dashboard | `records/dashboard.rs` | pending |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust

pub mod demagnetization;
pub mod slip_heating;
pub mod torque;

use super::record::{Family, Record};
```

with:

```rust

pub mod demagnetization;
pub mod slip_heating;
pub mod temperature;
pub mod torque;

use super::record::{Family, Record};
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
    torque::RECORDS,
    demagnetization::RECORDS,
    slip_heating::RECORDS,
];

/// Every batch's families.
```

with:

```rust
    torque::RECORDS,
    demagnetization::RECORDS,
    slip_heating::RECORDS,
    temperature::RECORDS,
];

/// Every batch's families.
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/temperature.rs` with exactly this content:

```rust
//! Temperature (plan A-3 batch 4): Br(T) and torque against temperature (torque ∝ Br_i Br_o,
//! each ring with its own coefficient, decision A2-7), the magnet ratings and their checks,
//! and the temperature summary: the limits, which one governs, the margins and the verdict
//! (Temperature design C6 to C25, C180 to C192; Calculator C35, C107, C108).
//!
//! The Br(T) factor 1 + α (ϑ − 20) appears in each torque record because each record is one
//! workbook cell and the factor is what the student should see (plan A-3).
//! Each formula is transcribed from the engine function named beside it, corrections on.

use crate::engine::deviations::DeviationId::E20;
use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- Calculator: the coefficient and the rating checks (model::compute) ---
    record("model.alpha_br_per_C", "α_{calc}", "{calibration.alpha_br_per_C}"),
    record("model.inner_temp_check", "C_{ϑ,i}",
        r#"cases({model.inner_tmax_C} = "n/a" => "unknown"; {coupling.op_temp_C} <= {model.inner_tmax_C} => "OK";
               else => "OVER the magnet rating")"#),
    record("model.outer_temp_check", "C_{ϑ,o}",
        r#"cases({model.outer_tmax_C} = "n/a" => "unknown"; {coupling.op_temp_C} <= {model.outer_tmax_C} => "OK";
               else => "OVER the magnet rating")"#),

    // --- Torque against temperature (temperature::compute: magnet life, adhesive life) ---
    // Reversible: T(ϑ) = T_20 · (1 + α_i (ϑ − 20)) (1 + α_o (ϑ − 20)).
    record("temperature.magnet_life.torque_hot_day_Nm", "T_{hot}",
        "{model.pullout_20C_Nm} * ((1 + {model.inner_alpha_br_per_C} * ({temperature.duty.hot_day_start_C} - 20))
                                * (1 + {model.outer_alpha_br_per_C} * ({temperature.duty.hot_day_start_C} - 20)))"),
    record("temperature.magnet_life.torque_hot_day_check", "C_{T,hot}",
        r#"cases({temperature.magnet_life.torque_hot_day_Nm} >= {metal.required_min_Nm} => "Meets it nominally (no variation allowance)";
               else => "Below it")"#),
    record("temperature.magnet_life.torque_peak_Nm", "T_{peak}",
        "{model.pullout_20C_Nm} * ((1 + {model.inner_alpha_br_per_C} * ({temperature.magnet_life.peak_C} - 20))
                                * (1 + {model.outer_alpha_br_per_C} * ({temperature.magnet_life.peak_C} - 20)))"),
    record("temperature.adhesive_life.torque_peak_var_Nm", "T_{peak,var}",
        "{temperature.magnet_life.torque_peak_Nm} * (1 + {metal.variation})"),
    record("temperature.adhesive_life.margin_C", "Δϑ_{adh}", "{temperature.adhesive.design_limit_C} - {temperature.magnet_life.peak_C}"),
    record("temperature.mismatch.cold_limit_C", "ϑ_{min,mm}", "{metal.min_temp_C}"),

    // --- The summary block (temperature::compute) ---
    record("temperature.summary.magnet_limit_C", "ϑ_{mag}^{sum}", "{temperature.demag.magnet_limit_C}"),
    record("temperature.summary.adhesive_limit_C", "ϑ_{adh}^{sum}", "{temperature.adhesive.design_limit_C}"),
    record("temperature.summary.governing_note", "C_{gov}",
        r#"cases({temperature.demag.magnet_limit_C} <= {temperature.adhesive.design_limit_C} => "Magnets govern (skipping case).";
               else => "Adhesive governs.")"#),
    record("temperature.summary.hot_day_start_C", "ϑ_{hot}^{sum}", "{temperature.duty.hot_day_start_C}"),
    record("temperature.summary.torque_hot_day_Nm", "T_{hot}^{sum}", "{temperature.magnet_life.torque_hot_day_Nm}"),
    record("temperature.summary.torque_hot_day_note", "C_{T,hot}^{sum}", "{temperature.magnet_life.torque_hot_day_check}"),
    // OK needs margin at the hot-day start, both limits above the peak, 10 °C of cure margin,
    // and (E20) no ring below its cold limit.
    record("temperature.summary.verdict", "V_{temp}",
        r#"cases({temperature.summary.margin_hot_day_C} > 0 and {temperature.magnet_life.margin_limit_C} > 0
                   and {temperature.adhesive_life.margin_C} > 0 and {temperature.summary.cure_margin_C} >= 10
                   and {temperature.demag.cold_check} != "Below the cold demagnetization limit"
                   => "OK on temperature. Confirm drag torque and thermal cycling by test.";
               else => "CHECK: see the rows above.")"#).corrected(&[E20]),
];
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 11 passed; 0 failed; 2 ignored` (266 equations).

- [ ] **Step 4: Physics review of the temperature chain (session model)**

Dispatch the physics reviewer on the **session model** (omit `model`, comment `// session model: physics`; CLAUDE.md section 5) with the prompt below, and give it the sheet this command prints:

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E "temp_check|alpha_br_per_C|magnet_life\.torque|adhesive_life|mismatch\.cold|summary\.(magnet|adhesive|governing_note|hot_day|torque|verdict)"
```

Expected: one Markdown table row per record of the batch: path, workbook cell (or `Rust-only`), the rendered formula and the corrections it embodies.

Reviewer prompt: "Review the temperature-chain records of the magcoupling equation explorer (the sheet below) against the M1 audit (docs/analyses/2026-09-29-magcoupling-math-audit.md) coverage of the Temperature design summary (C6 to C25) and the life blocks (C180 to C192), A-2 decision A2-7 (each ring's coefficient) and Addendum A decision 19 (E20's cold check in the verdict). The drift guard already proves each formula reproduces the engine; review what numbers cannot: (1) each symbol follows the conventions of `src/engine/explain/symbols.rs` (T torque, σ shear stress, ϑ temperature, φ angle, τ_p pole pitch) and no two paths share one; (2) each formula is the physics the M1 audit confirmed for that cell, stated one step from its terms; (3) each record names the corrections it embodies; (4) these reasoning claims hold: torque at a temperature is the 20 °C pull-out times each ring's Br(T) factor; the verdict's five conditions are the engine's, the cold one read through `cold_check`. Answer APPROVED, or list each finding with the record and the fix."

On APPROVED, continue. On findings: fix each in the batch's records file (a symbol in `symbols.rs`), re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain` until green, and send the changed rows back to the same reviewer; the task does not commit until the reviewer approves. A finding that would change the engine is out of scope: stop and escalate.

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task10.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task10.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task10.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/scope.rs magcoupling-rs/src/engine/explain/records/temperature.rs magcoupling-rs/src/engine/explain/records/mod.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): explorer batch 4, the temperature chain

Torque against temperature with each ring's Br coefficient, the ratings and
their checks, which limit governs, the margins and the verdict. The temperature
chain is explained.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 11: Batch 5: clamps explained

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Step 5, the physics review, runs on the session model.

A slotted clamp holds by friction: each screw's preload F squeezes the jaws, and one screw holds μ F d k. The preload is a
share of the proof load, capped by stripping the aluminium thread; the screws needed, the screws that fit along the clamp and the
geometry (the screw axis, the head seat, the grip, the far jaw's thread, all chords of the boss) decide which size works, and the
first size that works is recommended. The table's columns repeat per screw size, so each is a family over the five rows (the same
mechanism as the harmonics), and a size's standard data (diameter, stress area, hole, head) are read from the screw table rather
than recorded (a constant record could never vary). The recommended size's scalars pick their row by `clamps.index`.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs` (flip `clamps`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/clamps.rs` (15 records and 17 families over the 5 screw sizes (100 equations))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` (the batch's file and families)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs` (24 input symbols)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the clamps row)

**Interfaces:**
- Consumes: Task 7's tables (`screw_sizes`, `aluminium`); Task 6's `ceilto` and `floorto`; Task 3's `record::family` (a family indexed by table row: `#` becomes the row, `n` the row number).
- Produces: families `clamps.table[#].{offset_mm, wall_out_mm, head_fits, grip_mm, thread_avail_mm, engagement_req_mm, geometry_ok, preload_strength_N, preload_strip_N, preload_N, torque_per_screw_Nm, screws_needed, screws_fit, works, clamp_torque_Nm, sf_coupling, tightening_Nm}` over rows 0 to 4; records `clamps.{shaft_mm, boss_radius_mm, max_torque_Nm, required_Nm, clamp_factor, screw_proof_MPa, al_shear_MPa, index, screws, tightening_Nm, capacity_Nm, sf_coupling, joint_preload_N, joint_torque_Nm, joint_sf}`.

- [ ] **Step 1: Write the failing test: flip the chain**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs`, replace:

```rust
    },
    Chain {
        id: "clamps",
        status: Status::Pending,
        paths: &[
            "clamps.screw_proof_MPa",
            "clamps.joint_preload_N",
```

with:

```rust
    },
    Chain {
        id: "clamps",
        status: Status::Explained,
        paths: &[
            "clamps.screw_proof_MPa",
            "clamps.joint_preload_N",
```

- [ ] **Step 2: Run the structure tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain explained_chains 2>&1 | grep -E "panicked|clamps: " | head -3
```

Expected: the test panics listing the clamp paths without a record, starting `["clamps: clamps.screw_proof_MPa", "clamps: clamps.joint_preload_N", ...` (a table column is listed as `clamps.table[].preload_N`).

- [ ] **Step 3: Write the records and their symbols**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| demagnetization | `records/demagnetization.rs` | explained |
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | pending |
| dashboard | `records/dashboard.rs` | pending |

## Porting a module
```

with:

```markdown
| demagnetization | `records/demagnetization.rs` | explained |
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | explained |
| dashboard | `records/dashboard.rs` | pending |

## Porting a module
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/clamps.rs` with exactly this content:

```rust
//! Clamps (plan A-3 batch 5): the shaft clamp's preload and friction torque, per screw size
//! (the 'Clamp screw sizes' table, one family record per column over the five sizes, row 0 =
//! M2.5) and for the size recommended, and the adapter joint (Shaft clamps C14 to C69).
//!
//! A clamp holds by friction: each screw's preload F presses the jaws on the shaft, and the
//! torque one screw holds is μ F d k (friction, preload, shaft diameter, clamp factor). The
//! preload is a share of the screw's proof load, capped by stripping the aluminium thread.
//! Each formula is transcribed from `clamps::compute` with every approved correction on; a
//! screw size's standard data (diameter, stress area, hole, head) are read from the table.

use crate::engine::explain::record::{Family, Record, family, record};

/// The screw table's rows: M2.5, M3, M4, M5, M6.
const ROWS: &[u32] = &[0, 1, 2, 3, 4];

/// One record per screw size for each column the chain reads.
#[rustfmt::skip]
pub const FAMILIES: &[Family] = &[
    // Geometry: the screw axis sits half the bore, the ligament and half the clearance hole off
    // the shaft axis; the head seat, the grip and the far jaw's thread are chords of the boss.
    family(record("clamps.table[#].offset_mm", "e_{#}",
        r#"{clamps.shaft_mm} / 2 + {clamps.ligament_mm} + table("screw_sizes", n, "hole_mm") / 2"#), ROWS),
    family(record("clamps.table[#].wall_out_mm", "w_{out,#}",
        r#"{clamps.boss_radius_mm} - {clamps.table[#].offset_mm} - table("screw_sizes", n, "hole_mm") / 2"#), ROWS),
    family(record("clamps.table[#].head_fits", "fit_{head,#}",
        r#"cases({clamps.table[#].offset_mm} + table("screw_sizes", n, "head_mm") / 2 <= {clamps.boss_radius_mm} => 1; else => 0)"#), ROWS),
    family(record("clamps.table[#].grip_mm", "g_{grip,#}",
        r#"cases({clamps.table[#].head_fits} = 1
                   => sqrt({clamps.boss_radius_mm}^2 - ({clamps.table[#].offset_mm} + table("screw_sizes", n, "head_mm") / 2)^2) - {clamps.slit_mm} / 2;
               else => 0)"#), ROWS),
    family(record("clamps.table[#].thread_avail_mm", "L_{thr,#}",
        "cases({clamps.table[#].offset_mm} < {clamps.boss_radius_mm}
                 => sqrt({clamps.boss_radius_mm}^2 - {clamps.table[#].offset_mm}^2) - {clamps.slit_mm} / 2;
               else => 0)"), ROWS),
    family(record("clamps.table[#].engagement_req_mm", "L_{e,#}", r#"{clamps.engagement_x_d} * table("screw_sizes", n, "d_mm")"#), ROWS),
    family(record("clamps.table[#].geometry_ok", "ok_{geo,#}",
        "cases({clamps.table[#].wall_out_mm} >= {clamps.wall_out_mm} and {clamps.table[#].head_fits} = 1
                 and {clamps.table[#].grip_mm} >= {clamps.grip_min_mm} and {clamps.table[#].thread_avail_mm} >= {clamps.table[#].engagement_req_mm} => 1;
               else => 0)"), ROWS),
    // Strength: the preload from the screw's proof load, and the thread-stripping cap (shear
    // area about 0.6 π d L_e in the aluminium, with a safety factor); the smaller governs.
    family(record("clamps.table[#].preload_strength_N", "F_{b,#}",
        r#"{clamps.preload_fraction} * {clamps.screw_proof_MPa} * table("screw_sizes", n, "As_mm2")"#), ROWS),
    family(record("clamps.table[#].preload_strip_N", "F_{s,#}",
        r#"0.6 * π * table("screw_sizes", n, "d_mm") * {clamps.table[#].engagement_req_mm} * {clamps.al_shear_MPa} / {clamps.strip_sf}"#), ROWS),
    family(record("clamps.table[#].preload_N", "F_{#}", "min({clamps.table[#].preload_strength_N}, {clamps.table[#].preload_strip_N})"), ROWS),
    // Capacity: friction × preload × shaft diameter × clamp factor per screw.
    family(record("clamps.table[#].torque_per_screw_Nm", "T_{per,#}",
        "{clamps.friction} * {clamps.table[#].preload_N} * {clamps.shaft_mm|m} * {clamps.clamp_factor}"), ROWS),
    family(record("clamps.table[#].screws_needed", "n_{need,#}",
        "ceilto(frac({clamps.required_Nm}, {clamps.table[#].torque_per_screw_Nm}), 1)"), ROWS),
    // Fit: screws spaced a head diameter plus 1 mm along the clamp, inside the axial margins.
    family(record("clamps.table[#].screws_fit", "n_{fit,#}",
        r#"cases([s] >= 0 => floorto([s] / (table("screw_sizes", n, "head_mm") + 1), 1) + 1; else => 0)
           where [s] = {clamps.clamp_length_mm} - 2 * {clamps.axial_margin_mm} - (table("screw_sizes", n, "head_mm") + 0.5)"#), ROWS),
    family(record("clamps.table[#].works", "ok_{#}",
        "cases({clamps.table[#].geometry_ok} = 1 and {clamps.table[#].screws_needed} <= {clamps.table[#].screws_fit} => 1; else => 0)"), ROWS),
    family(record("clamps.table[#].clamp_torque_Nm", "T_{cl,#}",
        "{clamps.table[#].screws_needed} * {clamps.table[#].torque_per_screw_Nm}"), ROWS),
    family(record("clamps.table[#].sf_coupling", "S_{#}",
        "{clamps.table[#].screws_needed} * {clamps.table[#].torque_per_screw_Nm} / {clamps.max_torque_Nm}"), ROWS),
    family(record("clamps.table[#].tightening_Nm", "T_{tight,#}",
        r#"{clamps.nut_factor} * {clamps.table[#].preload_N} * table("screw_sizes", n, "d_mm") / 1000"#), ROWS),
];

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    record("clamps.shaft_mm", "d_{shaft}", "{coupling.bore_mm}"),
    record("clamps.boss_radius_mm", "R_{boss}", "{clamps.boss_od_mm} / 2"),
    record("clamps.max_torque_Nm", "T_{max}", "{metal.torque_cold_high_Nm}"),
    record("clamps.required_Nm", "T_{cl,req}", "{clamps.max_torque_Nm} * {clamps.safety_factor}"),
    record("clamps.clamp_factor", "k_{cl}",
        "cases({clamps.clamp_type} = 1 => {clamps.factor_one_piece}; else => {clamps.factor_two_piece})"),
    // ScrewClasses::proof: 12.9, 10.9 or A4-70 (the workbook's CHOOSE order).
    record("clamps.screw_proof_MPa", "σ_p",
        "cases({clamps.screw_class} = 1 => {materials.screws.proof_12_9_MPa}; {clamps.screw_class} = 2 => {materials.screws.proof_10_9_MPa};
               else => {materials.screws.yield_A4_70_MPa})"),
    record("clamps.al_shear_MPa", "τ_{Al,cl}",
        r#"cases({clamps.alloy} = 1 => table("aluminium", "7075-T6", "shear_MPa"); else => table("aluminium", "6061-T6", "shear_MPa"))"#),
    // The first size that works (0 when none does).
    record("clamps.index", "i_{size}",
        "cases({clamps.table[0].works} = 1 => 1; {clamps.table[1].works} = 1 => 2; {clamps.table[2].works} = 1 => 3;
               {clamps.table[3].works} = 1 => 4; {clamps.table[4].works} = 1 => 5; else => 0)"),
    record("clamps.screws", "n_{screws}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].screws_needed}; {clamps.index} = 2 => {clamps.table[1].screws_needed};
               {clamps.index} = 3 => {clamps.table[2].screws_needed}; {clamps.index} = 4 => {clamps.table[3].screws_needed};
               {clamps.index} = 5 => {clamps.table[4].screws_needed}; else => "")"#),
    record("clamps.tightening_Nm", "T_{tight}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].tightening_Nm}; {clamps.index} = 2 => {clamps.table[1].tightening_Nm};
               {clamps.index} = 3 => {clamps.table[2].tightening_Nm}; {clamps.index} = 4 => {clamps.table[3].tightening_Nm};
               {clamps.index} = 5 => {clamps.table[4].tightening_Nm}; else => "")"#),
    record("clamps.capacity_Nm", "T_{cap}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].clamp_torque_Nm}; {clamps.index} = 2 => {clamps.table[1].clamp_torque_Nm};
               {clamps.index} = 3 => {clamps.table[2].clamp_torque_Nm}; {clamps.index} = 4 => {clamps.table[3].clamp_torque_Nm};
               {clamps.index} = 5 => {clamps.table[4].clamp_torque_Nm}; else => "")"#),
    record("clamps.sf_coupling", "S_{cl}",
        r#"cases({clamps.index} = 1 => {clamps.table[0].sf_coupling}; {clamps.index} = 2 => {clamps.table[1].sf_coupling};
               {clamps.index} = 3 => {clamps.table[2].sf_coupling}; {clamps.index} = 4 => {clamps.table[3].sf_coupling};
               {clamps.index} = 5 => {clamps.table[4].sf_coupling}; else => "")"#),
    // The adapter joint: M3 screws on a bolt circle, friction on the nickel-plated face.
    record("clamps.joint_preload_N", "F_{joint}", "{clamps.table[1].preload_N}"),
    record("clamps.joint_torque_Nm", "T_{joint}",
        "{clamps.joint_friction} * {clamps.joint_screws} * {clamps.joint_preload_N} * {clamps.joint_bolt_circle_mm} / 2000"),
    record("clamps.joint_sf", "S_{joint}", "frac({clamps.joint_torque_Nm}, {clamps.required_Nm})"),
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
//! The equation records, one file per batch (plan A-3). A batch adds its file here and
//! flips its chain to `Explained` in [`super::scope`].

pub mod demagnetization;
pub mod slip_heating;
pub mod temperature;
```

with:

```rust
//! The equation records, one file per batch (plan A-3). A batch adds its file here and
//! flips its chain to `Explained` in [`super::scope`].

pub mod clamps;
pub mod demagnetization;
pub mod slip_heating;
pub mod temperature;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
    demagnetization::RECORDS,
    slip_heating::RECORDS,
    temperature::RECORDS,
];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES];
```

with:

```rust
    demagnetization::RECORDS,
    slip_heating::RECORDS,
    temperature::RECORDS,
    clamps::RECORDS,
];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES, clamps::FAMILIES];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("materials.steel.conductivity_S_m", "σ_{st,in}"),
    ("materials.steel.mu_r_incremental", "μ_{r,in}"),
    ("materials.steel.specific_heat_J_kgK", "c_{st,in}"),
    // temperature
    ("temperature.duty.hot_ambient_C", "ϑ_{amb}"),
    ("temperature.duty.driving_rise_C", "Δϑ_{drive}"),
```

with:

```rust
    ("materials.steel.conductivity_S_m", "σ_{st,in}"),
    ("materials.steel.mu_r_incremental", "μ_{r,in}"),
    ("materials.steel.specific_heat_J_kgK", "c_{st,in}"),
    ("materials.screws.proof_12_9_MPa", "σ_{p,12.9}"),
    ("materials.screws.proof_10_9_MPa", "σ_{p,10.9}"),
    ("materials.screws.yield_A4_70_MPa", "σ_{y,A4-70}"),
    // temperature
    ("temperature.duty.hot_ambient_C", "ϑ_{amb}"),
    ("temperature.duty.driving_rise_C", "Δϑ_{drive}"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("temperature.thermal.c_316", "c_{316,in}"),
    ("temperature.thermal.c_aluminium", "c_{Al,in}"),
    ("temperature.thermal.conductance_W_K", "G"),
];
```

with:

```rust
    ("temperature.thermal.c_316", "c_{316,in}"),
    ("temperature.thermal.c_aluminium", "c_{Al,in}"),
    ("temperature.thermal.conductance_W_K", "G"),
    // clamps
    ("clamps.safety_factor", "S_{req}"),
    ("clamps.friction", "μ"),
    ("clamps.clamp_type", "type"),
    ("clamps.factor_one_piece", "k_{1p}"),
    ("clamps.factor_two_piece", "k_{2p}"),
    ("clamps.alloy", "alloy"),
    ("clamps.screw_class", "class"),
    ("clamps.preload_fraction", "φ_p"),
    ("clamps.nut_factor", "K"),
    ("clamps.engagement_x_d", "k_{eng}"),
    ("clamps.strip_sf", "S_{strip}"),
    ("clamps.boss_od_mm", "D_{boss,cl}"),
    ("clamps.clamp_length_mm", "L_{cl}"),
    ("clamps.slit_mm", "s_{slit}"),
    ("clamps.ligament_mm", "l_{lig}"),
    ("clamps.wall_out_mm", "w_{out,min}"),
    ("clamps.grip_min_mm", "g_{grip,min}"),
    ("clamps.axial_margin_mm", "a_{ax}"),
    ("clamps.joint_screws", "n_{joint}"),
    ("clamps.joint_bolt_circle_mm", "D_{bc}"),
    ("clamps.joint_friction", "μ_{joint}"),
];
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 11 passed; 0 failed; 2 ignored` (366 equations).

- [ ] **Step 4: Physics review of the clamps (session model)**

Dispatch the physics reviewer on the **session model** (omit `model`, comment `// session model: physics`; CLAUDE.md section 5) with the prompt below, and give it the sheet this command prints:

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E "clamps\."
```

Expected: one Markdown table row per record of the batch: path, workbook cell (or `Rust-only`), the rendered formula and the corrections it embodies.

Reviewer prompt: "Review the clamp records of the magcoupling equation explorer (the sheet below) against the M1 audit (docs/analyses/2026-09-29-magcoupling-math-audit.md) rows M14 (the thread-stripping area) and E2 (the screw length and the slit), and the spec's M1 clamp description. The drift guard already proves each formula reproduces the engine; review what numbers cannot: (1) each symbol follows the conventions of `src/engine/explain/symbols.rs` (T torque, σ shear stress, ϑ temperature, φ angle, τ_p pole pitch) and no two paths share one; (2) each formula is the physics the M1 audit confirmed for that cell, stated one step from its terms; (3) each record names the corrections it embodies; (4) these reasoning claims hold: the friction torque per screw is μ F d k with d the shaft diameter; the preload is the smaller of the proof-load share and the stripping cap; a size works when its geometry fits and the screws needed fit along the clamp. Answer APPROVED, or list each finding with the record and the fix."

On APPROVED, continue. On findings: fix each in the batch's records file (a symbol in `symbols.rs`), re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain` until green, and send the changed rows back to the same reviewer; the task does not commit until the reviewer approves. A finding that would change the engine is out of scope: stop and escalate.

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task11.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task11.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task11.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/scope.rs magcoupling-rs/src/engine/explain/records/clamps.rs magcoupling-rs/src/engine/explain/records/mod.rs magcoupling-rs/src/engine/explain/symbols.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): explorer batch 5, the shaft clamps

Preload, the stripping cap and the friction torque per screw size (families over
the screw table's rows), the size recommended and the adapter joint. The clamps
chain is explained.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 12: Batch 6: the dashboard explained

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Step 5, the physics review, runs on the session model.

The dashboard's headline paths outside the five chains (report section 7.1). Two are formatted verdicts, stated with Task
6's text functions: the wall check quotes the rule's wall rounded up to 0.1 mm (decision 27's suggestion), the screw recommendation
the size, the even length (E2) and the class. The wall rule is t = B_gap τ_p / (π B_des) with E10's gap flux density (each magnet's
own MMF) and the design flux density in effect (decision 20). For M4: the screw table is keyed by row, so the typesetter shows the
size name for a `screw_sizes` key.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs` (flip `dashboard`)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/dashboard.rs` (12 records and 1 family (17 equations))
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` (the batch's file and family)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs` (8 input symbols)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the dashboard row)

**Interfaces:**
- Consumes: Task 6's `concat`, `fmt`, `fmtnum`, `ceilto`; Task 7's `back_iron.design_flux_density_T` and `screw_sizes.name`; Tasks 9 and 11's records.
- Produces: records for the seven headline paths outside the chains (`model.gearbox_input_ripple_Nm`, `mass.total_g`, `metal.min_running_clearance_mm`, `metal.clearance_check`, `materials.cup_wall_check`, `clamps.recommended`; `model.cup_od_mm` has its record from Task 9) and their closure (`metal.sleeve_liner_clearance_mm`, `metal.adverse_movement_mm`, `model.bsat_T`, `model.gap_flux_density_T`, `model.backiron_needed_mm`, `materials.cup_wall_suggested_mm`, the family `clamps.table[#].length_mm`).

- [ ] **Step 1: Write the failing test: flip the chain**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs`, replace:

```rust
    },
    Chain {
        id: "dashboard",
        status: Status::Pending,
        paths: &[
            "model.gearbox_input_ripple_Nm",
            "model.cup_od_mm",
```

with:

```rust
    },
    Chain {
        id: "dashboard",
        status: Status::Explained,
        paths: &[
            "model.gearbox_input_ripple_Nm",
            "model.cup_od_mm",
```

- [ ] **Step 2: Run the structure tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain explained_chains 2>&1 | grep -E "panicked|dashboard: " | head -3
```

Expected: the test panics with `["dashboard: model.gearbox_input_ripple_Nm", "dashboard: mass.total_g", "dashboard: metal.min_running_clearance_mm", "dashboard: metal.clearance_check", "dashboard: materials.cup_wall_check", "dashboard: clamps.recommended"]` (the cup OD has had its record since Task 9).

- [ ] **Step 3: Write the records and their symbols**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | explained |
| dashboard | `records/dashboard.rs` | pending |

## Porting a module
```

with:

```markdown
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | explained |
| dashboard | `records/dashboard.rs` | explained |

## Porting a module
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/dashboard.rs` with exactly this content:

```rust
//! Dashboard (plan A-3 batch 6): the headline paths outside the five chains (report section
//! 7.1): the gearbox input ripple, the cup OD, the rotating mass, the running clearance and
//! its check, the cup wall check and the recommended clamp screw, with what they need to reach
//! inputs: the back-iron wall rule (Calculator C36, C103, C104, E10), the wall suggestion
//! (decision 27), the sleeve-to-liner clearance and the screw length (E2).
//!
//! Two verdicts are formatted text; the markup states them with `concat`, `fmt` and `fmtnum`,
//! so they are proven like every other record (plan A-3). Each formula is transcribed
//! from the engine function named beside it, corrections on.

use crate::engine::deviations::DeviationId::{E2, E9, E10};
use crate::engine::explain::record::{Family, Record, family, record};

/// One record per screw size.
#[rustfmt::skip]
pub const FAMILIES: &[Family] = &[
    // clamps::compute, E2: the screw crosses the open slit before it reaches the far jaw; the
    // length is rounded up to an even millimetre.
    family(record("clamps.table[#].length_mm", "L_{scr,#}",
        "cases({clamps.table[#].head_fits} = 1
                 => ceilto({clamps.table[#].grip_mm} + {clamps.slit_mm} + {clamps.table[#].engagement_req_mm}, 2);
               else => 0)").corrected(&[E2]), &[0, 1, 2, 3, 4]),
];

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- Gearbox (model::compute) ---
    record("model.gearbox_input_ripple_Nm", "T_{ripple,in}",
        "frac({model.pullout_Nm}, {coupling.gear_ratio} * {coupling.gear_efficiency})"),

    // --- Rotating mass (model::mass_estimate) ---
    record("mass.total_g", "m_{tot}",
        "{mass.magnets_g} + {mass.cup_g} + {mass.hub_g} + {mass.boss_g} + {retainers.retainers_g} + {metal.hardware_g}
         + {retainers.cap_g} + {retainers.endplates_g}"),

    // --- Running clearance (metal_design::compute) ---
    record("metal.sleeve_liner_clearance_mm", "c_{nom}", "({retainers.liner_id_mm} - {retainers.sleeve_od_mm}) / 2"),
    record("metal.adverse_movement_mm", "Σ_{adv}",
        "{metal.shaft_displacement_mm} + {metal.runout_mm} + {metal.deflection_mm} + {metal.thermal_mm} + {metal.sleeve_form_mm}
         + {metal.magnet_position_mm}"),
    record("metal.min_running_clearance_mm", "c_{run}", "{metal.sleeve_liner_clearance_mm} - {metal.adverse_movement_mm}"),
    record("metal.clearance_check", "C_{run}",
        r#"cases({metal.min_running_clearance_mm} < {metal.residual_target_mm} => "Below target"; else => "Meets assumed target")"#),

    // --- The back-iron wall (model::compute, materials::compute) ---
    // The design flux density in effect (decision 20): a library back iron's own, else Materials C13.
    record("model.bsat_T", "B_{des}",
        r#"cases({materials.parts.back_iron} != 1 and table("back_iron", {materials.parts.back_iron}, "design_flux_density_T") != none
                   => table("back_iron", {materials.parts.back_iron}, "design_flux_density_T");
               else => {materials.steel.bsat_T})"#),
    // E10: in the series circuit each magnet contributes its own MMF, B_r t.
    record("model.gap_flux_density_T", "B_{gap}",
        "frac({model.br_inner_T_op} * {model.inner_thickness_mm} + {model.br_outer_T_op} * {model.outer_thickness_mm},
              {model.inner_thickness_mm} + {model.outer_thickness_mm} + {model.face_gap_mm})").corrected(&[E10]),
    // The flux of half a pole, B_gap τ_p / π per unit length, carried by the wall at B_des.
    record("model.backiron_needed_mm", "t_{bi}",
        "{model.gap_flux_density_T} * {model.pole_pitch_mm} / (π * {model.bsat_T})").corrected(&[E10]),
    // Decision 27: the rule's wall, rounded up to 0.1 mm (the number the check's advice quotes).
    record("materials.cup_wall_suggested_mm", "t_{wall,sug}",
        r#"cases({materials.circuit_backiron} = 0 => "n/a"; else => ceilto({model.backiron_needed_mm}, 0.1))"#).corrected(&[E9]),
    record("materials.cup_wall_check", "C_{wall}",
        r#"cases({materials.circuit_backiron} = 0 => "No back iron";
               {metal.cup_wall_corner_mm} >= {model.backiron_needed_mm} => "OK";
               else => concat("Too thin: raise Metal design C122 to at least ", fmt({materials.cup_wall_suggested_mm}, 1), " mm"))"#)
        .corrected(&[E9, E10]),

    // --- The recommended clamp screw (clamps::compute) ---
    record("clamps.recommended", "screw",
        r#"cases({clamps.index} = 0 => "None: enlarge the boss or the clamp length";
               else => concat("ISO 4762 ", table("screw_sizes", {clamps.index} - 1, "name"), " x ", fmtnum([L]), ", class ", [cls]))
           where [L] = cases({clamps.index} = 1 => {clamps.table[0].length_mm}; {clamps.index} = 2 => {clamps.table[1].length_mm};
                             {clamps.index} = 3 => {clamps.table[2].length_mm}; {clamps.index} = 4 => {clamps.table[3].length_mm};
                             else => {clamps.table[4].length_mm}),
                 [cls] = cases({clamps.screw_class} = 1 => "12.9"; {clamps.screw_class} = 2 => "10.9"; else => "A4-70")"#)
        .corrected(&[E2]),
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
//! flips its chain to `Explained` in [`super::scope`].

pub mod clamps;
pub mod demagnetization;
pub mod slip_heating;
pub mod temperature;
```

with:

```rust
//! flips its chain to `Explained` in [`super::scope`].

pub mod clamps;
pub mod dashboard;
pub mod demagnetization;
pub mod slip_heating;
pub mod temperature;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
    slip_heating::RECORDS,
    temperature::RECORDS,
    clamps::RECORDS,
];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES, clamps::FAMILIES];
```

with:

```rust
    slip_heating::RECORDS,
    temperature::RECORDS,
    clamps::RECORDS,
    dashboard::RECORDS,
];

/// Every batch's families.
pub const FAMILIES: &[&[Family]] = &[torque::FAMILIES, clamps::FAMILIES, dashboard::FAMILIES];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("metal.steel_density_g_mm3", "ρ_{st,in}"),
    ("metal.al_density_g_mm3", "ρ_{Al,in}"),
    ("metal.sleeve_density_g_mm3", "ρ_{316,in}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
```

with:

```rust
    ("metal.steel_density_g_mm3", "ρ_{st,in}"),
    ("metal.al_density_g_mm3", "ρ_{Al,in}"),
    ("metal.sleeve_density_g_mm3", "ρ_{316,in}"),
    ("metal.shaft_displacement_mm", "δ_{shaft}"),
    ("metal.runout_mm", "δ_{ro}"),
    ("metal.deflection_mm", "δ_{defl}"),
    ("metal.thermal_mm", "δ_{th}"),
    ("metal.sleeve_form_mm", "δ_{form}"),
    ("metal.magnet_position_mm", "δ_{pos}"),
    ("metal.residual_target_mm", "c_{res}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("materials.parts.back_iron", "material_{BI}"),
    ("materials.parts.sleeve_liner", "material_{SL}"),
    ("materials.parts.cap_housing", "material_{cap}"),
    ("materials.steel.conductivity_S_m", "σ_{st,in}"),
    ("materials.steel.mu_r_incremental", "μ_{r,in}"),
    ("materials.steel.specific_heat_J_kgK", "c_{st,in}"),
```

with:

```rust
    ("materials.parts.back_iron", "material_{BI}"),
    ("materials.parts.sleeve_liner", "material_{SL}"),
    ("materials.parts.cap_housing", "material_{cap}"),
    ("materials.steel.bsat_T", "B_{des,in}"),
    ("materials.steel.conductivity_S_m", "σ_{st,in}"),
    ("materials.steel.mu_r_incremental", "μ_{r,in}"),
    ("materials.steel.specific_heat_J_kgK", "c_{st,in}"),
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 11 passed; 0 failed; 2 ignored` (383 equations).

- [ ] **Step 4: Physics review of the dashboard (session model)**

Dispatch the physics reviewer on the **session model** (omit `model`, comment `// session model: physics`; CLAUDE.md section 5) with the prompt below, and give it the sheet this command prints:

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E "ripple|total_g|clearance|bsat|gap_flux|backiron_needed|cup_wall|recommended|length_mm"
```

Expected: one Markdown table row per record of the batch: path, workbook cell (or `Rust-only`), the rendered formula and the corrections it embodies.

Reviewer prompt: "Review the dashboard records of the magcoupling equation explorer (the sheet below) against the M1 audit (docs/analyses/2026-09-29-magcoupling-math-audit.md) rows M1 and M2 (the back-iron requirement), E10 (the gap flux density), E6 and E2, and Addendum A decisions 20 (the design flux density) and 27 (the wall suggestion). The drift guard already proves each formula reproduces the engine; review what numbers cannot: (1) each symbol follows the conventions of `src/engine/explain/symbols.rs` (T torque, σ shear stress, ϑ temperature, φ angle, τ_p pole pitch) and no two paths share one; (2) each formula is the physics the M1 audit confirmed for that cell, stated one step from its terms; (3) each record names the corrections it embodies; (4) these reasoning claims hold: the wall check's advice quotes the suggestion (the rule's wall rounded up to 0.1 mm); the recommended screw is the first size that works, at its even length. Answer APPROVED, or list each finding with the record and the fix."

On APPROVED, continue. On findings: fix each in the batch's records file (a symbol in `symbols.rs`), re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain` until green, and send the changed rows back to the same reviewer; the task does not commit until the reviewer approves. A finding that would change the engine is out of scope: stop and escalate.

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task12.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task12.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task12.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/scope.rs magcoupling-rs/src/engine/explain/records/dashboard.rs magcoupling-rs/src/engine/explain/records/mod.rs magcoupling-rs/src/engine/explain/symbols.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): explorer batch 6, the dashboard

The headline paths outside the chains: the gearbox ripple, the rotating mass,
the running clearance and its check, the cup wall check and the recommended
screw, the two formatted verdicts stated in the markup. The dashboard is
explained.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 13: Batch 7: the geometry callouts (decision G1)

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Step 5, the physics review, runs on the session model.

Decision 31 includes "the geometry callouts" and the report left them to be counted when the A1 view is specified. Decision
G1 lists them as the numbers the M4 geometry view draws: the face gap, the corner gap and the running clearance (spec M4 "Layout"),
the clearance's two parts (the nominal sleeve-to-liner clearance and the adverse movement), and, per axis, the overshoot past the
space claim with the dimension it measures (spec A1: a red callout naming the overshoot in mm per axis). Ten of the eleven are new
(the running clearance is a dashboard path), so the scope's union is 169 paths. The badge text `housing.space_claim_check` is not
a callout and stays unexplained (decision G2). The overshoot is max(−reserve, 0), which matches the engine's NaN for a NaN reserve.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs` (the `geometry` chain, explained; the module docs)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (the scope's path count, 169)
- Create: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/geometry.rs` (9 records)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs` (the batch's file)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs` (3 input symbols)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the geometry row)

**Interfaces:**
- Consumes: Tasks 3, 9 and 12's records (the face and corner gaps, the cup OD, the running clearance, the axial housing in effect).
- Produces: the `geometry` chain in `SCOPE` (11 paths, 10 new); records for `metal.{rotating_od_mm, axial_stack_mm, large_dia_stack_mm, diameter_reserve_mm, axial_reserve_mm, large_dia_reserve_mm}` and `housing.{diameter_overshoot_mm, length_overshoot_mm, bay_overshoot_mm}`.

- [ ] **Step 1: Write the failing test: add the chain**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs`, replace:

```rust
//!
//! Each batch of plan A-3 writes one chain's records and flips its status to `Explained`;
//! `tests/explain.rs` then requires a record for every path of every explained chain. The
//! geometry callouts are listed when the A1 view is specified (M4).

/// Whether a chain's records are written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

with:

```rust
//!
//! Each batch of plan A-3 writes one chain's records and flips its status to `Explained`;
//! `tests/explain.rs` then requires a record for every path of every explained chain. The
//! geometry callouts (plan A-3 decision G1) are the numbers the M4 geometry view draws: the
//! face gap, the corner gap and the running clearance (spec M4 "Layout"), the clearance's
//! two parts, and the overshoot per axis past the space claim with the dimension it measures
//! (spec A1); 10 of the 11 are new (the running clearance is a dashboard path), so the
//! scope holds 169 paths.

/// Whether a chain's records are written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/scope.rs`, replace:

```rust
        ],
    },
    Chain {
        id: "dashboard",
        status: Status::Explained,
        paths: &[
```

with:

```rust
        ],
    },
    Chain {
        id: "geometry",
        status: Status::Explained,
        paths: &[
            "model.face_gap_mm",
            "model.corner_gap_mm",
            "metal.min_running_clearance_mm",
            "metal.sleeve_liner_clearance_mm",
            "metal.adverse_movement_mm",
            "metal.rotating_od_mm",
            "metal.axial_stack_mm",
            "metal.large_dia_stack_mm",
            "housing.diameter_overshoot_mm",
            "housing.length_overshoot_mm",
            "housing.bay_overshoot_mm",
        ],
    },
    Chain {
        id: "dashboard",
        status: Status::Explained,
        paths: &[
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
    let union: BTreeSet<&str> = SCOPE.iter().flat_map(|c| c.paths.iter().copied()).collect();
    assert_eq!(
        union.len(),
        159,
        "decision 31: the chains and the dashboard (report section 7)"
    );
}
```

with:

```rust
    let union: BTreeSet<&str> = SCOPE.iter().flat_map(|c| c.paths.iter().copied()).collect();
    assert_eq!(
        union.len(),
        169,
        "decision 31: the chains and the dashboard (report section 7), and plan A-3 decision G1's 10 new geometry callouts"
    );
}
```

- [ ] **Step 2: Run the structure tests to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain explained_chains 2>&1 | grep -E "panicked|geometry: " | head -3
```

Expected: the test panics with `["geometry: metal.rotating_od_mm", "geometry: metal.axial_stack_mm", "geometry: metal.large_dia_stack_mm", "geometry: housing.diameter_overshoot_mm", "geometry: housing.length_overshoot_mm", "geometry: housing.bay_overshoot_mm"]` (the face gap, the corner gap, the running clearance and its two parts already have records).

- [ ] **Step 3: Write the records and their symbols**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | explained |
| dashboard | `records/dashboard.rs` | explained |

## Porting a module
```

with:

```markdown
| slip heating | `records/slip_heating.rs` | explained |
| temperature | `records/temperature.rs` | explained |
| clamps | `records/clamps.rs` | explained |
| geometry callouts (decision G1) | `records/geometry.rs` | explained |
| dashboard | `records/dashboard.rs` | explained |

## Porting a module
```

Create `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/geometry.rs` with exactly this content:

```rust
//! Geometry callouts (plan A-3 batch 7): the numbers the M4 geometry view draws (spec M4
//! "Layout": face gap, corner gap and running clearance; spec A1: the overshoot per axis past
//! the space claim), with the derived dimensions and reserves the overshoots compare
//! (Metal design C134 to C139). Decision G1 of plan A-3 lists them.
//!
//! The face gap, the corner gap and the running clearance already have records (the torque
//! chain and the dashboard); this batch adds the clearance's two parts and the space claim.
//! Each formula is transcribed from `metal_design::compute` and `housing::compute`.

use crate::engine::explain::record::{Record, record};

/// The plain records.
#[rustfmt::skip]
pub const RECORDS: &[Record] = &[
    // --- The derived dimensions and their reserves (metal_design::compute) ---
    record("metal.rotating_od_mm", "D_{rot}", "max({model.cup_od_mm}, {metal.cap_od_mm})"),
    record("metal.axial_stack_mm", "L_{stack}",
        "{metal.cap_axial_mm} + {housing.cup_depth_mm} + {metal.web_mm} + {metal.boss_length_mm}"),
    record("metal.large_dia_stack_mm", "L_{large}", "{metal.cap_axial_mm} + {housing.cup_depth_mm} + {metal.web_mm}"),
    record("metal.diameter_reserve_mm", "r_D", "{metal.max_diameter_mm} - {metal.rotating_od_mm}"),
    record("metal.axial_reserve_mm", "r_L", "{metal.max_overall_axial_mm} - {metal.axial_stack_mm}"),
    record("metal.large_dia_reserve_mm", "r_{bay}", "{metal.max_large_dia_axial_mm} - {metal.large_dia_stack_mm}"),

    // --- The space claim per axis (housing::compute): how far past the claim, 0 inside it ---
    record("housing.diameter_overshoot_mm", "o_D", "max(-{metal.diameter_reserve_mm}, 0)"),
    record("housing.length_overshoot_mm", "o_L", "max(-{metal.axial_reserve_mm}, 0)"),
    record("housing.bay_overshoot_mm", "o_{bay}", "max(-{metal.large_dia_reserve_mm}, 0)"),
];
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
pub mod clamps;
pub mod dashboard;
pub mod demagnetization;
pub mod slip_heating;
pub mod temperature;
pub mod torque;
```

with:

```rust
pub mod clamps;
pub mod dashboard;
pub mod demagnetization;
pub mod geometry;
pub mod slip_heating;
pub mod temperature;
pub mod torque;
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/records/mod.rs`, replace:

```rust
    temperature::RECORDS,
    clamps::RECORDS,
    dashboard::RECORDS,
];

/// Every batch's families.
```

with:

```rust
    temperature::RECORDS,
    clamps::RECORDS,
    dashboard::RECORDS,
    geometry::RECORDS,
];

/// Every batch's families.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/symbols.rs`, replace:

```rust
    ("metal.sleeve_form_mm", "δ_{form}"),
    ("metal.magnet_position_mm", "δ_{pos}"),
    ("metal.residual_target_mm", "c_{res}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
```

with:

```rust
    ("metal.sleeve_form_mm", "δ_{form}"),
    ("metal.magnet_position_mm", "δ_{pos}"),
    ("metal.residual_target_mm", "c_{res}"),
    ("metal.max_diameter_mm", "D_{max}"),
    ("metal.max_overall_axial_mm", "L_{max}"),
    ("metal.max_large_dia_axial_mm", "L_{bay}"),
    // calibration
    ("calibration.measured_torque_Nm", "T_{meas}"),
    ("calibration.total_magnets", "n_{mag}"),
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 11 passed; 0 failed; 2 ignored`: 392 equations, every one of the scope's 169 paths explained.

- [ ] **Step 4: Physics review of the geometry callouts (session model)**

Dispatch the physics reviewer on the **session model** (omit `model`, comment `// session model: physics`; CLAUDE.md section 5) with the prompt below, and give it the sheet this command prints:

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain review_sheet -- --ignored --nocapture 2>/dev/null | grep -E "rotating_od|stack|reserve|overshoot"
```

Expected: one Markdown table row per record of the batch: path, workbook cell (or `Rust-only`), the rendered formula and the corrections it embodies.

Reviewer prompt: "Review the geometry-callout records of the magcoupling equation explorer (the sheet below) against the Metal design rows C134 to C139 and A-2's space claim (housing.rs: the overshoot per axis, decision A2-8's axial housing in effect). The drift guard already proves each formula reproduces the engine; review what numbers cannot: (1) each symbol follows the conventions of `src/engine/explain/symbols.rs` (T torque, σ shear stress, ϑ temperature, φ angle, τ_p pole pitch) and no two paths share one; (2) each formula is the physics the M1 audit confirmed for that cell, stated one step from its terms; (3) each record names the corrections it embodies; (4) these reasoning claims hold: each overshoot is how far the dimension passes its claim, 0 inside or at the claim. Answer APPROVED, or list each finding with the record and the fix."

On APPROVED, continue. On findings: fix each in the batch's records file (a symbol in `symbols.rs`), re-run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain` until green, and send the changed rows back to the same reviewer; the task does not commit until the reviewer approves. A finding that would change the engine is out of scope: stop and escalate.

- [ ] **Step 5: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task13.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task13.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task13.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 6: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/scope.rs magcoupling-rs/tests/explain.rs magcoupling-rs/src/engine/explain/records/geometry.rs magcoupling-rs/src/engine/explain/records/mod.rs magcoupling-rs/src/engine/explain/symbols.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): explorer batch 7, the geometry callouts

Decision G1: the numbers the M4 geometry view draws (face gap, corner gap,
running clearance and its parts, the overshoot per axis with the dimension it
measures). The scope holds 169 paths, every one explained.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 14: A3 traceability over every assumption

**Model:** `sonnet` (the plan gives the exact code; CLAUDE.md section 5: set `model: 'sonnet'` on the implementer). A `blocked` end, a red gate or a rejected review escalates every retry to the session model. Any new `CANCELLATIONS` entry is a physics judgement: it is reviewed on the session model before it is added (none is expected).

Task 4 asserted the spec's A3 test for the 15 assumption inputs, but with only the torque chain explained the thermal and
clamp assumptions had no explained result downstream, so their check passed by absence. Now that every chain is explained (Task 13),
this task makes the test say so: every assumption path reaches an explained result, and each moves one at some design point; the
design points reach every chain's branches (E20 from the inputs and ferrite, E12, E13, a bench drag, 1018, the axial override that the
housing follows, the sleeve and cap picks), so soundness (for all 172 inputs, now at 114 design points instead of 51) has no blind
branch either; at them only 4 of the graph's 870 non-redundant edges could be dropped unseen. Inputs that bypass `set` give each
record a value or an `EvalError`, counted, never a panic. It is a test-only task: the records of Tasks 8 to 13 already satisfy
it, so the run passes at once; what it adds is that a future record or chain change that leaves an assumption without explained
results, or an input that moves nothing, fails here. No new cancellation appears.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (trace points for every chain's branches; the non-vacuity checks; the assumption test's preconditions; the bypass-set test)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the explain.rs tests row)

**Interfaces:**
- Consumes: Task 4's `check_traceability`, `trace_points`, `CANCELLATIONS`; every chain explained (Task 13).
- Produces: `trace_points()` with three more explicit points (the coercivity from the inputs with a bench drag, a drag of exactly 0 (E13), a hot-day start above the limit (E12)) and nine augmentations (adding back iron 1018, the axial override, the two ferrite ones and the sleeve and cap picks); `check_traceability` fails when an input checked moves no explained result at any point; `each_assumption_moves_what_depends_on_it_and_nothing_else` requires every chain explained and every assumption path to have explained results downstream; `records_never_panic_on_inputs_that_bypass_set` counts every record's value or `EvalError`.

- [ ] **Step 1: Strengthen the traceability tests**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override, the coercivity from the inputs with two ferrite betas, the sleeve and cap materials, E17's free-space fields), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

with:

```markdown
| `tests/material_links.rs` | Addendum A5 physics links: each selector offers the library's choices; the default choices change nothing; picking a steel or a sleeve equals typing its values into the inputs; the cap choice prices the cap only; a non-ferromagnetic back iron selects the free-space circuit and becomes the hub, cup and boss material (6061 equals the default no-back-iron design cell for cell, decision A2-6); C6 = 0 overrides a ferromagnetic choice; the design flux density feeds the wall check; a code outside the choices gives NaN, not another material; each warning fires on the design that meets its condition (`src/engine/warnings.rs` tests each rule, its equality edges and the steel-only rules directly). |
| `tests/assumptions.rs` | Addendum A3: every input flagged `.assumption()` is in exactly one `ASSUMPTIONS` row and the rows are the spec's v1 set; each row's source cites the workbook cell of every input it sets; the workbook defaults are the shipped defaults; changing one assumption flags exactly its row, the reset restores every assumption and keeps every design input; each assumption moves a result at the default design (the Hcj coefficient only with the coercivity source at 0, E20). The dependency-graph traceability test is plan A-3's. |
| `tests/sizing.rs` | Addendum A1 end to end: the axial length override keeps the measured calibration factor (part names, decision A2-3) and moves the magnets' mass; inverse sizing round-trips each free variable (1e-6), reports "not reachable" with the best value inside the range (a refined peak), keeps poles even (from an odd base too), returns the smaller of two meeting intervals, finds a meeting interval inside one grid cell, a target just below the true peak and a target met only at a validity edge, never counts f_end <= 0, blocks that do not fit (flats, or arcs that overlap) or a keyway through the hub (each rule at its equality edge), and refuses an invalid target or invalid inputs; the space claim is exceeded exactly when a derived dimension exceeds its claim, per axis (at the claim is inside), a sized design shows its overshoot, several axes are named in order, a NaN claim reads unknown (beside any exceeded axis) and a tiny overshoot reads at least 0.01 mm; the axial housing follows the override (decision A2-8): blank it changes no bit, at the rings' own length nothing moves, the hub, the cup cavity and the retainer span follow the inner, the outer and the longer ring, a short override shrinks both stacks, no dimension ends shorter than its ring (the floor at its equality edge), the masses follow, a length-sized design can exceed the space claim (the default design sized to 9.9 N·m: 34.72 mm over the length, 36.72 mm over the bay, as the hand sums give), each stack trips the claim exactly where it crosses it, and a NaN override reads unknown. |
| `tests/explain.rs` | Addendum A2 explanation layer. The drift guard: every equation record, evaluated over the engine's own term values, reproduces the engine's result by the parity rule (1e-9 relative, 1e-12 absolute; text exact) at the defaults and at every differential case under every augmentation (the Rust-only inputs the generator never varies: the harmonic set, the back-iron material, the grade mode, the axial override, the coercivity from the inputs with two ferrite betas, the sleeve and cap materials, E17's free-space fields), corrections on; every `cases` arm is taken (or listed unreachable with its reason), every numeric record and every value term of every record varies and every E7 angle leaves half a pitch somewhere. The registry is consistent (every term has a symbol, a kind and a reverse edge; no custom evals; no formula reads a path in two units or would typeset two ways); every path of an explained chain has a record and drills down to inputs. A3 traceability: nudging any input leaves every explained result outside its dependency set bit-identical (soundness, every input), and each assumption moves every numeric result on its active path above rounding at some design point (sensitivity), the algebraic cancellations listed with their reasons; every input has a nudge (a step both ways, the slider's ends, every other choice) that changes some result at some design point (the two inputs no result reads listed); with every chain explained, each of the 15 assumption inputs reaches an explained result and moves one at some design point (the A3 test is not vacuous); a modified assumption styles its term and the results it flows into; a selector code shows its choice labels and a result names the corrections upstream of it. Every A4 note links to records and each equation has at most one note; `release_notes_are_reviewed` (ignored) is the M4 release gate. `review_sheet` (ignored) prints the physics reviewer's sheet. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), NaN and infinities written straight into the struct, where `set` cannot refuse them, with the corrections off and on (`compute_all_never_panics_on_non_finite_struct_literals`; `validate()` names the path), a positive beta typed into C45 with the coercivity source at 0 (E20's cold side; the slider stays the NdFeB range, decision A2-5), and a debug-build time bound per frame (`compute_all` plus `headline`). The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none; an input is Rust-only exactly when it has no cell); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust

/// Design points for the traceability test: the defaults; the measured prototype's own
/// circuit (6061 back iron, so the bench correction is the calibration factor) at a test
/// temperature off 20 °C; a grade-mode ring with eleven harmonics at six poles; and a dozen
/// full-run cases under four of the drift guard's augmentations.
fn trace_points() -> Vec<(String, DesignInputs)> {
    let design = |sets: &[(&str, Value)]| {
        let mut inputs = DesignInputs::default();
```

with:

```rust

/// Design points for the traceability test: the defaults; the measured prototype's own
/// circuit (6061 back iron, so the bench correction is the calibration factor) at a test
/// temperature off 20 °C; a grade-mode ring with eleven harmonics at six poles; the
/// coercivity from the inputs with a bench drag entered; a hot-day start above the limit
/// (E12); and a dozen full-run cases under nine of the drift guard's augmentations (the
/// back-iron, grade, axial-override, ferrite and part-material branches of every chain).
fn trace_points() -> Vec<(String, DesignInputs)> {
    let design = |sets: &[(&str, Value)]| {
        let mut inputs = DesignInputs::default();
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
                ("calibration.test_temp_C", Value::Num(35.0)),
            ]),
        ),
    ];
    let augmentations: Vec<_> = augmentations()
        .into_iter()
```

with:

```rust
                ("calibration.test_temp_C", Value::Num(35.0)),
            ]),
        ),
        // The Hcj inputs act for both rings, and a bench drag replaces the loss estimate; a drag
        // of exactly 0 is no heating at all (E13).
        (
            "coercivity from the inputs, drag measured".to_owned(),
            design(&[
                ("temperature.demag.coercivity_source", Value::Int(0)),
                ("metal.measured_drag_Nm", Value::Num(0.02)),
            ]),
        ),
        (
            "drag measured as 0 (E13)".to_owned(),
            design(&[("metal.measured_drag_Nm", Value::Num(0.0))]),
        ),
        (
            "hot-day start above the limit (E12)".to_owned(),
            design(&[("temperature.duty.driving_rise_C", Value::Num(45.0))]),
        ),
    ];
    let augmentations: Vec<_> = augmentations()
        .into_iter()
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
            [
                "as generated",
                "harmonics 11",
                "back iron 6061",
                "grade mode",
            ]
            .contains(l)
        })
```

with:

```rust
            [
                "as generated",
                "harmonics 11",
                "back iron 1018",
                "back iron 6061",
                "grade mode",
                "axial 8 mm",
                "ferrite inputs",
                "unrated ferrite",
                "sleeve Ti, cap acetal",
            ]
            .contains(l)
        })
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
            ));
        }
    }
    failures
}
```

with:

```rust
            ));
        }
    }
    // Not vacuous: every input checked moves some explained result at some point.
    for input in paths {
        if !moved_pairs.iter().any(|(i, _)| i == input) {
            failures.push(format!(
                "{input} moves no explained result at any design point"
            ));
        }
    }
    failures
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust

#[test]
fn each_assumption_moves_what_depends_on_it_and_nothing_else() {
    // Spec A3 testing: the fifteen assumption inputs (the fourteen rows; end effect is two).
    let paths: Vec<String> = ASSUMPTIONS
        .iter()
        .flat_map(|a| a.paths.iter().map(|p| (*p).to_owned()))
        .collect();
    assert_eq!(paths.len(), 15);
    let failures = check_traceability(&paths, true);
    assert!(failures.is_empty(), "{}", report(&failures));
}
```

with:

```rust

#[test]
fn each_assumption_moves_what_depends_on_it_and_nothing_else() {
    // Spec A3 testing: the fifteen assumption inputs (the fourteen rows; end effect is two),
    // every one of them with explained results downstream now that every chain is explained.
    let r = Registry::build();
    let paths: Vec<String> = ASSUMPTIONS
        .iter()
        .flat_map(|a| a.paths.iter().map(|p| (*p).to_owned()))
        .collect();
    assert_eq!(paths.len(), 15);
    assert!(
        SCOPE.iter().all(|c| c.status == Status::Explained),
        "every chain of decision 31 is explained"
    );
    for p in &paths {
        assert!(
            !r.downstream(p).is_empty(),
            "{p}: no explained result depends on it"
        );
    }
    let failures = check_traceability(&paths, true);
    assert!(failures.is_empty(), "{}", report(&failures));
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
        .collect();
    let failures = check_traceability(&paths, false);
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
```

with:

```rust
        .collect();
    let failures = check_traceability(&paths, false);
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn records_never_panic_on_inputs_that_bypass_set() {
    // The records are proven over validated inputs (the drift guard sets every input through
    // `set`); a design file or share link can hold what `set` refuses (decision D3: the GUI
    // validates at its boundaries). Evaluating every record over such inputs, written straight
    // into the struct, gives a value or an EvalError, never a panic: an invalid harmonic code,
    // selector codes outside their choices, a NaN temperature and a NaN reserve.
    let r = Registry::build();
    let mut inputs = DesignInputs::default();
    inputs.coupling.max_harmonic = 4;
    inputs.temperature.adhesive.selected = 0;
    inputs.clamps.screw_class = 9;
    inputs.materials.parts.back_iron = 99;
    inputs.coupling.op_temp_C = f64::NAN;
    inputs.metal.max_diameter_mm = f64::NAN;
    assert!(
        inputs.validate().is_err(),
        "validate() names every one of them"
    );
    let results = compute_all(&inputs);
    let src = Design {
        inputs: &inputs,
        results: &results,
    };
    let (mut values, mut errors) = (0, 0);
    for eq in r.equations() {
        match r.evaluate(eq, &src, None) {
            Ok(_) => values += 1,
            Err(_) => errors += 1,
        }
    }
    // Every record returned (a value or an EvalError), and most give a value.
    assert_eq!(values + errors, r.equations().len());
    assert!(values > errors, "{values} values, {errors} errors");
}

#[test]
```

- [ ] **Step 2: Run them**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain -- each_assumption no_input_moves 2>&1 | grep "test result"
```

Expected: `test result: ok. 2 passed` (about 3 s with 32 threads).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 12 passed; 0 failed; 2 ignored` (`records_never_panic_on_inputs_that_bypass_set` is the new one).

- [ ] **Step 3: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task14.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task14.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task14.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 4: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/tests/explain.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
test(magcoupling-rs): A3 traceability over every assumption, every chain explained

Every assumption path reaches an explained result and moves one at some design
point; the design points reach every chain's branches (E20 from the inputs and
ferrite, E12, E13, a bench drag, 1018, the axial override, the sleeve and cap
picks); inputs that bypass set never make a record panic.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 15: The remaining A4 teaching notes and the start-here order

**Model:** session model (omit `model` on purpose, comment `// session model: physics`; CLAUDE.md section 5): teaching prose about the physics. The plan gives every sentence; the implementer still checks each against the source it cites before committing, and Task 16 is the independent physics review.

Spec A4 asks for about 15 to 20 notes on genuinely tricky ideas, 2 to 6 sentences each at Physics 2 level, an optional "watch
out" line and an optional small diagram, drafted from the M1 derivations, each checked by a physics reviewer before release. The list
(decision G5): the spec's nine ideas (harmonic decomposition; the back-iron sinh factor against free space; pull-out torque against
angle; end effect; Br(T) and torque ∝ Br²; demagnetization, knee and permeance; eddy-current slip loss and skin depth; the thermal
time constant; clamp preload and friction), the physics behind each of the six A5 warnings, ferrite's cold-side demagnetization (spec
A6: "it gets a teaching note"), and the one-point calibration (the bench correction's cancellation of the assumed factor, which the
traceability test found). Each note cites the M1 rows and decisions it is drafted from; the numbers it quotes are the engine's at the
default design (a 30 °C rise costs (1 − 0.0012·30)² ≈ 0.93 of the torque; the default skin depth is 1.30 mm; a 6061 back iron takes the
default pull-out from 2.688 to 1.697 N·m, more than a third less, including the switch of the
calibration factor from 0.95 to the bench's 1.049 that this circuit takes); the field's angular frequency ω_e and the slip speed
ω_s, a factor p apart, are never written as one ω. Each equation has at most one note, and each "start here" entry opens
its note's own equation (a test). Two diagram kinds join the four (the knee and load lines; the first-order heating curve), which M4
paints. The red test: every note has 2 to 6 sentences and sources (the stubs fail), the count is 15 to 20, and the "start here"
entries are covered.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs` (13 notes drafted (the stubs replaced, a calibration note added), two diagram kinds, `START_HERE` with 11 entries; tests)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs` (the note links test checks every note)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (the explorer section)

**Interfaces:**
- Consumes: Task 5's notes module; every chain's records (each note's equations must have records).
- Produces: `NOTES` with 17 notes (`harmonics`, `back_iron_factor`, `pullout_angle`, `end_effect`, `calibration`, `br_temperature`, `demagnetization`, `ferrite_cold_demag`, `slip_heating`, `thermal_time_constant`, `clamp_preload` and the six `a5.*`), all `Review::Draft`; `Diagram::{DemagKnee, HeatingCurve}`; `START_HERE` = harmonics → pull-out angle → end effect → calibration → back iron → Br(T) → demagnetization → ferrite's cold side → slip heating → thermal time constant → clamps, each entry the note's own equation.

- [ ] **Step 1: Write the failing tests**

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
                rule.note_id
            );
        }
        for (id, _) in START_HERE {
            assert!(note(id).is_some(), "start-here note {id}");
        }
    }

    #[test]
    fn a_draft_is_hidden_and_a_reviewed_note_is_complete() {
        assert!(note_for("model.f_end").is_none(), "drafts stay hidden");
        assert_eq!(
            note_for_any_status("model.f_end").map(|n| n.id),
            Some("end_effect")
        );
        for n in NOTES {
            if n.sentences.is_empty() {
                assert_eq!(
                    n.review,
                    Review::Draft,
                    "{}: a stub cannot be reviewed",
                    n.id
                );
                continue;
            }
            assert!(
                (2..=6).contains(&n.sentences.len()),
                "{}: 2 to 6 sentences",
```

with:

```rust
                rule.note_id
            );
        }
        for (id, path) in START_HERE {
            let n = note(id).unwrap_or_else(|| panic!("start-here note {id}"));
            assert!(
                n.equations.iter().any(|e| covers(e, path)),
                "start here {id}: the note does not explain {path}"
            );
        }
    }

    #[test]
    fn a_draft_is_hidden_and_every_note_is_complete() {
        assert_eq!(
            note_for_any_status("model.f_end").map(|n| n.id),
            Some("end_effect")
        );
        assert!(
            (15..=20).contains(&NOTES.len()),
            "spec A4: about 15 to 20 notes"
        );
        for n in NOTES {
            assert!(
                (2..=6).contains(&n.sentences.len()),
                "{}: 2 to 6 sentences",
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
            );
            assert!(
                !n.sources.is_empty(),
                "{}: a drafted note cites its M1 sources",
                n.id
            );
        }
    }
}
```

with:

```rust
            );
            assert!(
                !n.sources.is_empty(),
                "{}: a note cites the M1 derivations it is drafted from",
                n.id
            );
            // The accuracy gate: the panel shows a note only once it is reviewed.
            let reviewed = matches!(n.review, Review::Reviewed { .. });
            for entry in n.equations {
                let path = entry.replace('#', "1");
                assert_eq!(
                    note_for_any_status(&path).map(|m| m.id),
                    Some(n.id),
                    "{path}"
                );
                assert_eq!(note_for(&path).is_some(), reviewed, "{}: {path}", n.id);
            }
        }
    }
}
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/tests/explain.rs`, replace:

```rust
                "note {}: {entry} names no result",
                n.id
            );
            if n.sentences.is_empty() {
                continue; // a stub fixes an id; its links are checked when it is drafted
            }
            for m in members {
                assert!(
                    r.equation_for(m).is_some(),
```

with:

```rust
                "note {}: {entry} names no result",
                n.id
            );
            for m in members {
                assert!(
                    r.equation_for(m).is_some(),
```

- [ ] **Step 2: Run them to see them fail**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib notes 2>&1 | grep -E "panicked|2 to 6|start-here" | head -3
```

Expected: two of the three notes tests panic: `ids_are_unique_and_every_link_resolves` with `start here demagnetization: the note does not explain temperature.demag.onset_skipping_C` (a stub lists no equation) and `a_draft_is_hidden_and_every_note_is_complete` with `br_temperature: 2 to 6 sentences` (a stub has none).

- [ ] **Step 3: Draft the notes**

Before applying, read each sentence against the M1 row it cites (`C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-29-magcoupling-math-audit.md`) and the decision it names; a sentence you cannot trace to its source is a stop-and-escalate, not a rewrite.

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Formatted verdicts are stated in the markup too (`concat`, `fmt`, `fmtnum`), with `ceilto` and `floorto` for Excel's CEILING and FLOOR and the literals `inf` and `nan`, so no record needs a Rust closure. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

with:

```markdown
workbook cell come from the result's metadata, and where a formula needs a value the engine
computed but did not expose, the engine exposes it as a Rust-only result (a read, no formula
change). The registry is built explicitly (`Registry::build()`, a few ms) and owned by the
caller: no global state. Formatted verdicts are stated in the markup too (`concat`, `fmt`, `fmtnum`), with `ceilto` and `floorto` for Excel's CEILING and FLOOR and the literals `inf` and `nan`, so no record needs a Rust closure. For A3 it gives each term's style (`term_style`: assumption, changed from default, downstream of a modified assumption) and names the modified assumptions upstream of a result; for the panel it gives a selector code's choice labels (`choices`) and the corrections embodied upstream of a result (`corrections_upstream`, the corrected-vs-workbook marker). The A4 teaching notes are static data (`notes.rs`), each listing the equations it explains; the panel shows a note only once a physics reviewer has signed it off (`note_for`). There are 17: the spec's list (harmonics, the back-iron factor, pull-out against angle, end effect, Br(T), demagnetization, slip loss and skin depth, the thermal time constant, clamp preload, and the physics behind each of the six A5 warnings) plus ferrite's cold side and the one-point calibration; `START_HERE` opens them in the spec's order (torque chain, back iron, temperature, demagnetization, slip heating, clamps). Plan:
`docs/superpowers/plans/2026-09-30-magcoupling-addendum-a3-explainability.md`.

| Chain (decision 31) | Records | Status |
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
    TorqueAngle,
    /// Field fringing at the magnet ends.
    EndFringing,
}

/// Where a note stands in the accuracy gate.
```

with:

```rust
    TorqueAngle,
    /// Field fringing at the magnet ends.
    EndFringing,
    /// The intrinsic demagnetization curve with its knee, and the load lines of a few
    /// permeance coefficients.
    DemagKnee,
    /// First-order heating toward a steady temperature, the time constant marked.
    HeatingCurve,
}

/// Where a note stands in the accuracy gate.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
    pub review: Review,
}

const fn stub(id: &'static str, title: &'static str, equations: &'static [&'static str]) -> Note {
    Note {
        id,
        title,
        equations,
        sentences: &[],
        watch_out: None,
        diagram: None,
        sources: &[],
        review: Review::Draft,
    }
}

/// Every note. The torque chain's four are drafted (plan A-3 tracer); the rest are stubs
/// that fix the ids the "start here" order and the A5 warnings link to.
pub const NOTES: &[Note] = &[
    Note {
        id: "harmonics",
```

with:

```rust
    pub review: Review,
}

/// Every note: 17, the spec's list (harmonics, back iron, pull-out against angle, end effect,
/// Br(T), demagnetization, slip loss, the thermal time constant, clamps and the six A5
/// warnings) plus ferrite's cold side (spec A6) and the one-point calibration.
pub const NOTES: &[Note] = &[
    Note {
        id: "harmonics",
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`, replace:

```rust
        sources: &["M1 audit M8 and M9 (T3-RC1, T3-RC3)"],
        review: Review::Draft,
    },
    stub(
        "br_temperature",
        "Remanence against temperature, and torque ∝ Br²",
        &[
            "model.br_inner_T_op",
            "model.br_outer_T_op",
            "model.pullout_20C_Nm",
        ],
    ),
    stub(
        "demagnetization",
        "Demagnetization: knee, permeance and the onset temperatures",
        &[],
    ),
    stub("slip_heating", "Eddy-current slip loss and skin depth", &[]),
    stub("thermal_time_constant", "The thermal time constant", &[]),
    stub("clamp_preload", "Clamp preload and friction", &[]),
    stub(
        "ferrite_cold_demag",
        "Ferrite: the demagnetization risk is at cold",
        &[],
    ),
    stub(
        "a5.non_ferromagnetic_back_iron",
        "Non-ferromagnetic back iron",
        &["materials.circuit_backiron"],
    ),
    stub(
        "a5.ferromagnetic_sleeve_or_liner",
        "Ferromagnetic sleeve or liner",
        &[],
    ),
    stub(
        "a5.high_conductivity_sleeve_or_liner",
        "High-conductivity sleeve or liner",
        &[],
    ),
    stub("a5.low_saturation", "Low saturation", &[]),
    stub(
        "a5.uncoated_low_alloy_steel",
        "Uncoated low-alloy steel",
        &[],
    ),
    stub(
        "a5.cte_mismatch_with_magnets",
        "Expansion mismatch with the magnets",
        &[],
    ),
];

/// The suggested reading order (spec A4 "Start here": torque chain → back iron →
/// temperature → demagnetization → slip heating → clamps): (note id, the equation it opens).
pub const START_HERE: &[(&str, &str)] = &[
    ("harmonics", "model.tau_Pa"),
    ("back_iron_factor", "model.s1_iron"),
    ("br_temperature", "model.br_inner_T_op"),
    ("demagnetization", "temperature.demag.onset_skipping_C"),
    ("slip_heating", "temperature.slip_loss.total_W"),
    ("clamp_preload", "clamps.capacity_Nm"),
];
```

with:

```rust
        sources: &["M1 audit M8 and M9 (T3-RC1, T3-RC3)"],
        review: Review::Draft,
    },
    Note {
        id: "calibration",
        title: "The one-point calibration",
        equations: &[
            "model.f_cal",
            "calibration.f_cal_updated",
            "calibration.measured_over_model",
            "calibration.model_error",
        ],
        sentences: &[
            "The 2D harmonic model is a few per cent off real flat blocks, so the workbook scales it by a calibration factor.",
            "For the measured prototype's own rings and circuit (no back iron) it uses the bench result: f_cal,1 = f_cal,0 · T_meas / T_model, which puts the model on the 1.8 N·m measurement.",
            "Because T_model already contains f_cal,0, the assumed factor cancels: the bench correction is T_meas divided by the uncalibrated model, whatever f_cal,0 was.",
            "Every other design keeps the assumed 0.95, since one measurement cannot say how the model's error changes with the design.",
        ],
        watch_out: Some(
            "The bench value has two significant figures and an assumed 20 °C test temperature; it sits 2.5 % above the 3D pull-out (M1 audit P2).",
        ),
        diagram: None,
        sources: &[
            "M1 audit M4 and M6 (the model's bias and the calibration that absorbs it)",
            "M1 audit P2 (the bench value against physics)",
            "Plan A-3 traceability: the f_cal,0 cancellation (tests/explain.rs CANCELLATIONS)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "br_temperature",
        title: "Remanence against temperature, and torque ∝ Br²",
        equations: &[
            "model.br_inner_T_op",
            "model.br_outer_T_op",
            "model.pullout_20C_Nm",
            "metal.torque_cold_Nm",
            "temperature.magnet_life.torque_hot_day_Nm",
            "temperature.magnet_life.torque_peak_Nm",
            "temperature.demag.torque_at_limit_Nm",
        ],
        sentences: &[
            "A magnet's remanence Br, the flux density it keeps with no applied field, falls reversibly as it warms: Br(ϑ) = Br(20 °C) · (1 + α (ϑ − 20 °C)), with α about −0.12 %/°C for sintered NdFeB.",
            "Each ring's field is proportional to its own Br, and the shear stress is a product of the two rings' fields, so the torque scales as Br,i · Br,o: the square of Br when both rings are one grade.",
            "So a 30 °C rise costs about 7 % of the torque: (1 − 0.0012 · 30)² ≈ 0.93.",
            "The loss is reversible: cool the magnet and the torque comes back, unless it passed its demagnetization limit on the way, which is a separate, permanent loss.",
        ],
        watch_out: Some(
            "Each ring keeps its own coefficient: a ferrite ring (α about −0.2 %/°C) loses torque faster with heat than an NdFeB one.",
        ),
        diagram: None,
        sources: &[
            "M1 audit, confirmed: the Br(T) and torque-temperature rows (Calculator C69, C70, C94; Metal design C8)",
            "Addendum A decision A2-7 (each ring with its own coefficient)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "demagnetization",
        title: "Demagnetization: knee, permeance and the onset temperatures",
        equations: &[
            "temperature.demag.h_ref_kA_m",
            "temperature.demag.t_ref_model_C",
            "temperature.demag.calibration_offset_C",
            "temperature.demag.onset_aligned_C",
            "temperature.demag.onset_pullout_C",
            "temperature.demag.onset_skipping_C",
            "temperature.demag.onset_single_ring_C",
            "temperature.demag.magnet_limit_C",
            "temperature.demag.inner_magnet_limit_C",
            "temperature.demag.outer_magnet_limit_C",
            "temperature.demag.demag_ring",
        ],
        sentences: &[
            "Inside a magnet the field points against its magnetization; the stronger this reverse field H, the closer the magnet is to losing magnetization for good.",
            "The loss starts at the knee of the intrinsic curve, taken as H_k = 0.9 · Hcj, and Hcj falls as NdFeB heats (β about −0.5 %/°C), so each reverse field has an onset temperature where the falling knee meets it; the reverse field itself shrinks with Br(T), which is why the onset needs both α and β.",
            "The reverse field depends on the magnet's surroundings, its permeance: a reference magnet on the load line B = −μ0 H (permeance coefficient 1) sees Br/(2 μ0), and like poles of the other ring facing it while the coupling skips push it highest, 863 kA/m here, so the skipping onset sets the limit.",
            "The onsets are shifted so the reference magnet reaches its knee exactly at its rated temperature, and the design limit keeps a 10 °C margin below the skipping onset.",
            "With correction E20 each ring is checked with its own grade, and the ring with the lower limit governs.",
        ],
        watch_out: Some(
            "Hcj(T) is linear over 20–150 °C only: onsets computed above 150 °C are extrapolations, and a ±20 % change of the slope there moves the default limit by about ±3 °C (M1 audit M13).",
        ),
        diagram: Some(Diagram::DemagKnee),
        sources: &[
            "M1 audit M13 (knee temperature outside the coefficient range)",
            "M1 audit rulings: the onset calibration (it uses 9.83 °C of the 10 °C margin)",
            "Addendum A decision 19 (E20: each ring's own grade)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "ferrite_cold_demag",
        title: "Ferrite: the demagnetization risk is at cold",
        equations: &[
            "temperature.demag.inner_cold_limit_C",
            "temperature.demag.outer_cold_limit_C",
            "temperature.demag.cold_ring",
            "temperature.demag.cold_limit_C",
            "temperature.demag.cold_check",
            "temperature.demag.cold_onset_aligned_C",
            "temperature.demag.cold_onset_pullout_C",
            "temperature.demag.cold_onset_skipping_C",
            "temperature.demag.cold_onset_single_ring_C",
        ],
        sentences: &[
            "In hard ferrite the coercivity rises as it warms (β about +0.35 %/°C), the opposite of NdFeB, so its knee falls as the magnet cools.",
            "Its remanence still rises as it cools, so the reverse field grows while the knee shrinks: below some temperature the reverse field passes the knee and the magnet loses magnetization.",
            "The calculator finds that cold onset, ϑ = 20 + (H_k − H)/(H α − H_k β), and checks the minimum magnet temperature against the skipping cold onset plus the 10 °C margin; on heating a ferrite magnet never reaches its knee, so its rating is the hot limit.",
        ],
        watch_out: Some(
            "A ferrite ring can pass every hot check and still fail at −40 °C: read the cold check too.",
        ),
        diagram: Some(Diagram::DemagKnee),
        sources: &[
            "Spec A6 (ferrite's opposite-sign beta gets a teaching note)",
            "Addendum A decisions 3 (Y30 beta +0.35 %/°C) and 19 (E20's positive-beta branch)",
            "A-1 plan decision A13 (both rings; the cold side from the ring with the higher cold limit)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "slip_heating",
        title: "Eddy-current slip loss and skin depth",
        equations: &[
            "temperature.duty.field_omega_rad_s",
            "temperature.slip_loss.skin_depth_mm",
            "temperature.slip_loss.hub_W",
            "temperature.slip_loss.cup_W",
            "temperature.slip_loss.web_W",
            "temperature.slip_loss.cap_W",
            "temperature.slip_loss.magnets_W",
            "temperature.slip_loss.total_W",
            "temperature.slip_loss.drag_Nm",
        ],
        sentences: &[
            "When the coupling slips, each ring's alternating field sweeps past the other ring's parts at the field frequency f = p · n / 60, and by Faraday's law it drives eddy currents in every conductor it crosses: the steel cup and hub, the sleeve, the liner, the cap and the magnets themselves.",
            "The currents turn power into heat, and that power is also the drag torque times the slip speed.",
            "In steel the currents crowd into a skin of depth δ = √(2/(ω_e μ0 μr σ)), with ω_e = 2πf the field's angular frequency, about 1.3 mm at the default slip, so the steel loss grows as speed to the power 1.5 rather than 2.",
            "Thin shells such as the sleeve and liner are thinner than their skin depth, so their loss follows σ (ω_s r B)²/2 per unit volume, with ω_s the slip speed in rad/s, the square of speed.",
        ],
        watch_out: Some(
            "Aluminium parts (no back iron) are resistance-limited, not skin-limited, and see the weaker free-space field (correction E17); every loss here is an estimate with a factor-3 high case until a bench drag is measured.",
        ),
        diagram: None,
        sources: &[
            "M1 audit M11 (thin-skin steel losses against Stoll's half-space)",
            "M1 audit M12 (magnet eddy loss)",
            "M1 audit P1 (the shell end factor)",
            "Addendum A report section 5.5 (E17: the low-Reynolds form for aluminium parts)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "thermal_time_constant",
        title: "The thermal time constant",
        equations: &[
            "temperature.thermal.heat_capacity_J_K",
            "temperature.thermal.time_constant_s",
            "temperature.thermal.t95_s",
            "temperature.thermal.steady_rise_est_C",
            "temperature.thermal.steady_rise_high_C",
            "temperature.thermal.time_to_limit_high",
            "temperature.thermal.time_to_limit_est",
            "temperature.thermal.temp_at_fault_C",
            "temperature.thermal.critical_drag_Nm",
        ],
        sentences: &[
            "The calculator treats the rotating coupling as one lump with heat capacity C = Σ m c, losing heat to its surroundings through one conductance G.",
            "Heated with power P, its temperature approaches the steady rise P/G along ϑ(t) = ϑ_start + (P/G)(1 − e^(−t/τ)), with time constant τ = C/G: 63 % of the rise after τ and 95 % after 3τ.",
            "The steady rise P/G does not depend on C; C (with G) sets only how fast it is reached; one short slip event adds only P t / C.",
            "The time to the limit solves that curve for the limit temperature, and it is 'never' when the steady temperature stays below the limit.",
        ],
        watch_out: Some(
            "G is a placeholder (0.3 W/K, not measured): a 10 % change moves the high-case steady temperature by about 2.5 °C (M1 audit placeholder table).",
        ),
        diagram: Some(Diagram::HeatingCurve),
        sources: &[
            "M1 audit placeholder inputs (conductance: −2.48 °C per +10 %)",
            "Addendum A decision 8 (E15: aluminium parts at their own specific heat)",
            "M1 audit E12 (a start above the limit)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "clamp_preload",
        title: "Clamp preload and friction",
        equations: &[
            "clamps.table[#].preload_strength_N",
            "clamps.table[#].preload_strip_N",
            "clamps.table[#].preload_N",
            "clamps.table[#].torque_per_screw_Nm",
            "clamps.table[#].screws_needed",
            "clamps.capacity_Nm",
        ],
        sentences: &[
            "A slotted clamp holds the shaft by friction: tightening each screw stretches it to a preload F, which squeezes the jaws onto the shaft.",
            "Friction resists slipping on both jaws at the shaft radius, so one screw holds about T = μ F d k, with d the shaft diameter and k the share of the screw force that reaches the shaft.",
            "The preload is 75 % of the screw's proof load, unless the aluminium thread would strip first; the smaller of the two governs.",
            "The clamp must hold the cold-high torque times a safety factor of 2, which sets how many screws are needed.",
        ],
        watch_out: Some(
            "μ = 0.15 assumes a degreased bore; an oily bore (about 0.10) cuts the capacity by a third.",
        ),
        diagram: None,
        sources: &[
            "M1 audit M14 (the thread-stripping area)",
            "M1 audit E2 (the screw length and the slit)",
            "Addendum A3 assumptions: clamp friction and preload fraction",
        ],
        review: Review::Draft,
    },
    Note {
        id: "a5.non_ferromagnetic_back_iron",
        title: "Non-ferromagnetic back iron",
        equations: &["materials.circuit_backiron"],
        sentences: &[
            "Steel behind the magnets gives each ring's flux an easy return path, so the flux crosses the gap and closes through the steel instead of spreading out behind the magnets.",
            "Stainless 304 and aluminium have a relative permeability near 1, like air, so choosing one switches the calculator to the free-space circuit: the geometry factor loses its sinh form and the torque drops by more than a third (2.69 to 1.70 N·m at the default design).",
            "The flux that no longer closes through steel leaks out around the coupling, so a strong stray field extends outside it and pulls in steel chips and debris.",
        ],
        watch_out: None,
        diagram: Some(Diagram::FluxPathBackIron),
        sources: &[
            "Spec A5 warning rules",
            "M1 audit M5 (steel at the magnet backs)",
            "Addendum A report section 5 (the no-back-iron circuit, E15 to E17)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "a5.ferromagnetic_sleeve_or_liner",
        title: "Ferromagnetic sleeve or liner",
        equations: &[],
        sentences: &[
            "The sleeve and liner sit in the magnetic gap, the one place the flux must cross from ring to ring.",
            "A ferromagnetic sleeve or liner offers the flux an easy path along itself, from one pole to its neighbour on the same ring, so much of it never crosses the gap.",
            "Flux that short-circuits this way carries no torque, so the torque collapses; that is why the retainers are non-magnetic 316L, titanium, Inconel or PEEK.",
        ],
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Addendum A report section 4 (the materials table: ferromagnetic or not)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "a5.high_conductivity_sleeve_or_liner",
        title: "High-conductivity sleeve or liner",
        equations: &[
            "temperature.slip_loss.sleeve_W",
            "temperature.slip_loss.liner_W",
        ],
        sentences: &[
            "The sleeve and liner sit in the strongest alternating field during slip, and as thin shells their eddy loss is proportional to their conductivity: σ t (ω_s r B)²/2 per unit area, with ω_s the slip speed.",
            "A material that conducts better than 316L (1.35 × 10⁶ S/m) therefore heats more for the same slip, raising the steady temperature and shortening the time to the limit.",
            "Titanium and Inconel conduct less than 316L and PEEK almost not at all, so none of the listed choices fires this warning; a typed conductivity can.",
        ],
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "M1 audit P1 (the shell loss and its end factor)",
            "Addendum A report section 4 (the conductivities)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "a5.low_saturation",
        title: "Low saturation",
        equations: &["model.bsat_T", "model.backiron_needed_mm"],
        sentences: &[
            "The back-iron wall carries each pole's flux around to its neighbour, and the wall it needs is t = B_gap τ_p / (π B_des): the lower the flux density the steel is allowed, the thicker the wall.",
            "The design value B_des sits below saturation, 1.5 T for annealed 4140; a steel whose saturation is below 1.7 T (1018's design value, the highest the workbook names) is flagged, and so is a steel with no design value of its own, for which the wall check uses the design flux density input (1.5 T by default).",
            "Past saturation the steel's permeability collapses, flux leaks out of the circuit and the torque falls, so such a back iron needs thicker walls.",
        ],
        watch_out: Some(
            "The wall formula itself reads 12 to 16 % thin against exact 2D sections (M1 audit M1), so walls near the limit deserve margin.",
        ),
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Addendum A decision 20 (the design flux density feeds the wall check)",
            "M1 audit M1 and M2 (the back-iron requirement)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "a5.uncoated_low_alloy_steel",
        title: "Uncoated low-alloy steel",
        equations: &[],
        sentences: &[
            "Plain and low-alloy steels such as 4140, 1018 and 12L14 rust in humid air; the stainless grades in the list (416, 17-4PH, 304) resist it.",
            "The workbook therefore plans a high-phosphorus electroless nickel plating, which is non-magnetic as plated and adds about 0.015 mm per surface.",
            "With no plating thickness entered, a plain or low-alloy steel back iron is flagged, because it needs a coating.",
        ],
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "Workbook Materials C26 (electroless nickel, 0.013–0.025 mm)",
        ],
        review: Review::Draft,
    },
    Note {
        id: "a5.cte_mismatch_with_magnets",
        title: "Expansion mismatch with the magnets",
        equations: &[],
        sentences: &[
            "Heating changes a part's length by α ΔT per unit length; NdFeB barely changes across its magnetization (about −0.8 × 10⁻⁶ /°C), while steel expands about 12 × 10⁻⁶ /°C and aluminium about 24 × 10⁻⁶ /°C.",
            "Where a magnet is glued to the hub, that difference is forced through the thin glue line as shear, largest at the block ends.",
            "The larger the mismatch and the temperature swing, the higher that shear, so a hub whose expansion differs from the magnets' by more than 15 × 10⁻⁶ /°C is flagged; the Volkersen screen puts a number on it.",
        ],
        watch_out: None,
        diagram: None,
        sources: &[
            "Spec A5 warning rules",
            "M1 audit E1 (the Volkersen shear-lag screen)",
            "Addendum A decision 16 (E18: the aluminium hub's expansion)",
        ],
        review: Review::Draft,
    },
];

/// The suggested reading order (spec A4 "Start here": torque chain → back iron →
/// temperature → demagnetization → slip heating → clamps): (note id, the equation it opens).
pub const START_HERE: &[(&str, &str)] = &[
    ("harmonics", "model.tau_Pa"),
    ("pullout_angle", "model.pullout_Nm"),
    ("end_effect", "model.f_end"),
    ("calibration", "model.f_cal"),
    ("back_iron_factor", "model.s1_iron"),
    ("br_temperature", "model.pullout_20C_Nm"),
    ("demagnetization", "temperature.demag.onset_skipping_C"),
    ("ferrite_cold_demag", "temperature.demag.cold_check"),
    ("slip_heating", "temperature.slip_loss.total_W"),
    (
        "thermal_time_constant",
        "temperature.thermal.time_constant_s",
    ),
    ("clamp_preload", "clamps.capacity_Nm"),
];
```

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib notes 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed`.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain 2>&1 | grep -E "test result|FAILED"
```

Expected: `test result: ok. 12 passed; 0 failed; 2 ignored` (every note's equations have records; each equation has at most one note).

- [ ] **Step 4: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task15.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task15.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task15.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 5: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/tests/explain.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
feat(magcoupling-rs): the A4 teaching notes and the start-here order

17 notes at Physics 2 level drafted from the M1 derivations they cite: the spec's
list, the six A5 warnings, ferrite's cold side and the one-point calibration,
with two more diagram kinds. All drafts: Task 16 is their physics review.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 16: Physics review of the notes and the sign-off (the A4 accuracy gate)

**Model:** session model throughout (omit `model`, comment `// session model: physics`; CLAUDE.md section 5): the reviewer of Steps 1 to 5 is the physics reviewer the spec's accuracy gate names; it is a fresh context that did not write the notes.

Spec A4: "notes are drafted from the M1 derivations, and each one is checked by a physics reviewer before release." Each review
step gives the reviewer a group of notes (`cargo test` prints nothing about prose, so the reviewer reads `notes.rs` directly) and the
derivations they cite. The reviewer answers APPROVED or lists findings per note. A finding is fixed in that note's sentences (keep 2 to
6 sentences and the sources, re-run `cargo test --lib notes` and the explain tests, send the changed note back to the same reviewer);
a note the reviewer will not approve stays a draft, hidden by `note_for`, and its id comes out of the sign-off script's list, which
keeps the release gate red (the M4 checklist then shows it). The sign-off script records the reviewer, today's date and this review.

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs` (each approved note's `review` becomes `Review::Reviewed { .. }`)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (how many notes are signed off)

**Interfaces:**
- Consumes: Task 15's 17 drafted notes; `tests/explain.rs`'s ignored `release_notes_are_reviewed`.
- Produces: every approved note marked `Review::Reviewed { reviewer, date, record }`, so `notes::note_for` shows it; with all 17 approved, the M4 release gate passes.

- [ ] **Step 1: Review the torque-chain notes**

Dispatch the physics reviewer (session model) with the prompt: "Review these teaching notes of the magcoupling calculator (`C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`) for physics accuracy at Physics 2 level against the sources each cites (the M1 audit `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-29-magcoupling-math-audit.md`, the Addendum A report `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the spec's A4 and A5) and the equation records each lists (`src/engine/explain/records/`): every sentence true and plainly worded, every number the engine's at the default design, the watch-out line justified, the diagram fitting. Answer APPROVED per note, or list each finding with the note id and the fix. Notes: `harmonics`, `back_iron_factor`, `pullout_angle`, `end_effect`, `calibration`. Sources: M1 rows M3, M4, M5, M7, M8, M9, E7, P2; decision 29; the f_cal,0 cancellation in tests/explain.rs CANCELLATIONS." Record each answer.

- [ ] **Step 2: Review the temperature and demagnetization notes**

Dispatch the physics reviewer (session model) with the prompt: "Review these teaching notes of the magcoupling calculator (`C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`) for physics accuracy at Physics 2 level against the sources each cites (the M1 audit `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-29-magcoupling-math-audit.md`, the Addendum A report `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the spec's A4 and A5) and the equation records each lists (`src/engine/explain/records/`): every sentence true and plainly worded, every number the engine's at the default design, the watch-out line justified, the diagram fitting. Answer APPROVED per note, or list each finding with the note id and the fix. Notes: `br_temperature`, `demagnetization`, `ferrite_cold_demag`. Sources: M1 row M13 and the onset-calibration ruling; decisions 3, 18, 19; A-1 decision A13; A-2 decision A2-7. Check the permeance-coefficient-1 reference field Br/(2 μ0) and the cold-onset formula against `temperature::cold_onset_C`." Record each answer.

- [ ] **Step 3: Review the slip-heating, thermal and clamp notes**

Dispatch the physics reviewer (session model) with the prompt: "Review these teaching notes of the magcoupling calculator (`C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`) for physics accuracy at Physics 2 level against the sources each cites (the M1 audit `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-29-magcoupling-math-audit.md`, the Addendum A report `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the spec's A4 and A5) and the equation records each lists (`src/engine/explain/records/`): every sentence true and plainly worded, every number the engine's at the default design, the watch-out line justified, the diagram fitting. Answer APPROVED per note, or list each finding with the note id and the fix. Notes: `slip_heating`, `thermal_time_constant`, `clamp_preload`. Sources: M1 rows M11, M12, M14, P1, E2, E12 and the placeholder table; report section 5.5 (E17); decision 8 (E15). Check that the field's angular frequency ω_e = 2πf (in the skin depth) and the slip speed ω_s (in the shell loss) are never confused: they differ by the pole-pair count." Record each answer.

- [ ] **Step 4: Review the A5 warning notes**

Dispatch the physics reviewer (session model) with the prompt: "Review these teaching notes of the magcoupling calculator (`C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs`) for physics accuracy at Physics 2 level against the sources each cites (the M1 audit `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-29-magcoupling-math-audit.md`, the Addendum A report `C:/Users/Cole/source/repos/lsim-mag-a3/docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`, the spec's A4 and A5) and the equation records each lists (`src/engine/explain/records/`): every sentence true and plainly worded, every number the engine's at the default design, the watch-out line justified, the diagram fitting. Answer APPROVED per note, or list each finding with the note id and the fix. Notes: `a5.non_ferromagnetic_back_iron`, `a5.ferromagnetic_sleeve_or_liner`, `a5.high_conductivity_sleeve_or_liner`, `a5.low_saturation`, `a5.uncoated_low_alloy_steel`, `a5.cte_mismatch_with_magnets`. Sources: the spec's A5 warning rules, `src/engine/warnings.rs` (each rule's condition and threshold), report section 4 (the materials table), M1 rows M1, M2, M5, E1; decisions 16 and 20. In `a5.high_conductivity_sleeve_or_liner` the shell loss uses the slip speed ω_s, not the field's angular frequency ω_e." Record each answer.

- [ ] **Step 5: Act on the findings**

Fix each finding in its note as the intro says and re-run the two commands below until green; every note the reviewer approves stays in the sign-off list, every other id is removed from `APPROVED` in the next step's script.

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib notes 2>&1 | grep "test result"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain notes_link 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed` (the notes unit tests), then `test result: ok. 1 passed`.

- [ ] **Step 6: Sign the approved notes off**

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import datetime, pathlib
# The ids the physics reviewer approved in Steps 1 to 4. Remove any id the reviewer did not
# approve: that note stays a draft (hidden by notes::note_for) and the release gate stays red.
APPROVED = [
    "harmonics", "back_iron_factor", "pullout_angle", "end_effect", "calibration",
    "br_temperature", "demagnetization", "ferrite_cold_demag",
    "slip_heating", "thermal_time_constant", "clamp_preload",
    "a5.non_ferromagnetic_back_iron", "a5.ferromagnetic_sleeve_or_liner", "a5.high_conductivity_sleeve_or_liner",
    "a5.low_saturation", "a5.uncoated_low_alloy_steel", "a5.cte_mismatch_with_magnets",
]
REVIEWER = "physics reviewer (session model)"
RECORD = "plan A-3 Task 16 physics review of the A4 notes against the cited M1 derivations"
today = f"{datetime.date.today():%Y-%m-%d}"
p = pathlib.Path(r"C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/src/engine/explain/notes.rs")
s = p.read_text(encoding="utf-8")
for note_id in APPROVED:
    start = s.index(f'        id: "{note_id}",\n')
    at = s.index("        review: Review::Draft,\n", start)
    assert s.find("        id: ", start + 1, at) == -1, f"{note_id}: its review line was not found before the next note"
    s = s[:at] + (f'        review: Review::Reviewed {{ reviewer: "{REVIEWER}", date: "{today}", record: "{RECORD}" }},\n'
                  ) + s[at + len("        review: Review::Draft,\n"):]
p.write_text(s, encoding="utf-8", newline="\n")
r = pathlib.Path(r"C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md")
t = r.read_text(encoding="utf-8")
old = "the panel shows a note only once a physics reviewer has signed it off (`note_for`)."
assert t.count(old) == 1
n = len(APPROVED)
t = t.replace(old, old[:-1] + f"; {n} of the 17 are signed off (plan A-3 Task 16).")
r.write_text(t, encoding="utf-8", newline="\n")
print(f"{n} notes reviewed on {today}")
EOF
```

The script prints `17 notes reviewed on <today's date>` when every note is approved.

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --lib notes 2>&1 | grep "test result"
```

Expected: `test result: ok. 3 passed` (a reviewed note shows through `note_for`; the test checks it for every note).

- [ ] **Step 7: Run the release gate**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test explain release_notes_are_reviewed -- --ignored 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed` when all 17 are approved (every "start here" note and every A5 note reviewed). With a note left a draft it fails naming it: record that in the commit body and in Task 17's memory entry, and continue.

- [ ] **Step 8: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task16.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task16.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task16.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 9: Commit**

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/src/engine/explain/notes.rs magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
docs(magcoupling-rs): sign off the A4 teaching notes after the physics review

The physics reviewer checked each note against the M1 derivations it cites;
approved notes are Review::Reviewed (reviewer, date, record), so note_for shows
them. With all 17 approved the M4 release gate passes; a note left a draft
keeps it red (name it here and in Task 17's memory entry).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

### Task 17: Docs pass and the final gate

**Model:** `sonnet` (documentation with the exact text given; CLAUDE.md section 5). The final whole-branch review after this task runs on the session model (controller).

The repo rule (`docs/ai/01-meta.yaml`): after file-modifying work update 02-system (invariants, status), 03-structure (module
layout), 04-memory (open questions) and 05-update-tracker; the crate README's layout, tests and explorer rows were kept current task
by task. The spec gets two wording fixes this plan's design rests on (decisions G6 and G7): A2's "an `eval` closure over term values" becomes the evaluable
markup (no closure, a closure only as a counted escape hatch), and the A3 test line states what Tasks 4 and 14 check (no result outside
an assumption's dependents moves; every numeric result on its active path moves at some design point, cancellations listed).

**Files:**
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md` (status)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/02-system.yaml` (responsibility, two invariants, status)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/03-structure.yaml` (the crate's role, the `explain` entry, the tests list)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/04-memory.yaml` (the A-3 traceability item resolved; decisions G1 to G7; the M4 items; the E4 bondline owner)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (A2: the formula is what the guard evaluates; the A3 test as checked)
- Modify: `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/05-update-tracker.md` (the A-3 entry)

**Interfaces:**
- Consumes: everything above.
- Produces: docs that match the code (the repo rule); a green final gate; the default headline unchanged.

- [ ] **Step 1: Crate README status, docs/ai and the spec**

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/02-system.yaml`, replace:

```yaml
      registry, M1 audit E1-E14, Addendum A E15-E20). Engine complete (M2), every module but fields3d;
      Addendum A-1 adds the grade table, materials per part with physics links, and six warnings;
      Addendum A-2 adds the harmonic set, the assumptions registry, inverse sizing and the space claim, with an axial housing that follows the length override.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
```

with:

```yaml
      registry, M1 audit E1-E14, Addendum A E15-E20). Engine complete (M2), every module but fields3d;
      Addendum A-1 adds the grade table, materials per part with physics links, and six warnings;
      Addendum A-2 adds the harmonic set, the assumptions registry, inverse sizing and the space claim, with an axial housing that follows the length override.
      Addendum A-3 adds the equation explorer's engine side: evaluable equation records proven by a drift guard, the dependency graph, the A3 traceability test and the A4 teaching notes.
    key_files: [src/engine/meta.rs, src/engine/compat.rs, src/engine/deviations.rs, src/engine/api.rs]
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/02-system.yaml`, replace:

```yaml
      - The E7 peak search (model::peak_off_half_pitch) returns None exactly when half a pitch is the maximum (1e-12 relative), so every E7-neutral design keeps the workbook expression bit for bit; it finds every root of dT/dx in cos^2 x by recursion on derivatives (no closed form, no scan grid).
      - Inverse sizing (sizing::solve) evaluates compute_all only inside the free variable's slider range; a continuous variable's 64-cell scan is refined between samples at the first crossing, at each peak (golden section) and at each validity edge, so only a torque hump whose rise and fall both lie inside one cell is unseen (the documented limit). A value counts only if sizing::is_valid holds (the blocks fit, faceted blocks on their flats and arcs without overlapping, model::blocks_fit; the keyway leaves hub wall; f_end > 0, model::end_effect_in_range; a finite hot-low torque) and meets when its hot-low torque reaches the target; poles stay even.
      - Library parts and gradeless manual magnets use Calibration C22 and the NdFeB density; only a grade-mode ring (manual dimensions with a grade) takes its grade's alpha(Br) and density (decision A2-7).

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
```

with:

```yaml
      - The E7 peak search (model::peak_off_half_pitch) returns None exactly when half a pitch is the maximum (1e-12 relative), so every E7-neutral design keeps the workbook expression bit for bit; it finds every root of dT/dx in cos^2 x by recursion on derivatives (no closed form, no scan grid).
      - Inverse sizing (sizing::solve) evaluates compute_all only inside the free variable's slider range; a continuous variable's 64-cell scan is refined between samples at the first crossing, at each peak (golden section) and at each validity edge, so only a torque hump whose rise and fall both lie inside one cell is unseen (the documented limit). A value counts only if sizing::is_valid holds (the blocks fit, faceted blocks on their flats and arcs without overlapping, model::blocks_fit; the keyway leaves hub wall; f_end > 0, model::end_effect_in_range; a finite hot-low torque) and meets when its hot-low torque reaches the target; poles stay even.
      - Library parts and gradeless manual magnets use Calibration C22 and the NdFeB density; only a grade-mode ring (manual dimensions with a grade) takes its grade's alpha(Br) and density (decision A2-7).
      - Every equation record (src/engine/explain/records/) reproduces the engine's result by the parity rule at the defaults and at every differential case under every augmentation, corrections on, every value term of every record taking two values there (the drift guard, tests/explain.rs); the registry refuses a formula that reads a path in two units or would typeset two ways; the formula shown is the one evaluated (no custom evals); a record reads its terms' engine values and never recomputes upstream; a value the engine computed but did not expose becomes a Rust-only result (a read); every path of the explorer's scope (169) drills down to inputs.
      - A3 traceability: nudging any input moves no explained result outside its dependency set (soundness; every input but the two no result reads changes some result at some design point), and each assumption moves every numeric result on its active path at some design point above rounding, with the algebraic cancellations listed with reasons (sensitivity).

dataflow_on_user_change: |
  user edit -> state.blueprint mutation -> state.push_undo() -> state.rebuild()
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/02-system.yaml`, replace:

```yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim; next: Addendum A-3 (explanations), then M4 GUI"
```

with:

```yaml
  dxf_import: works native + WASM via drag-drop; selection-based conversions
  actuator_workflow: two-pass solve gives correct reactions with actuator loads
  non_grashof_sweep: NaN-padded across unreachable angles; playback ping-pongs
  magcoupling: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied. Addendum A-1 complete: grades, materials per part, warnings, E15-E20 applied. Addendum A-2 complete: harmonic set, assumptions registry, end-effect flag, inverse sizing, space claim. Addendum A-3 complete: equation explorer engine side, A3 traceability, A4 notes; next: M4 GUI"
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/03-structure.yaml`, replace:

```yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20) and A-2 (harmonic set, assumptions, inverse sizing, space claim) complete
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
```

with:

```yaml

magcoupling_rs:
  path: magcoupling-rs/ (repo root, sibling of linkage-sim-rs/; separate crate, NOT a workspace member)
  role: Rust port of reference/magcoupling-py (spec docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md); M2 complete - every Python module except fields3d (M3) is ported, workbook parity covers all 1,149 checks, E1-E14 applied; Addendum A-1 (grades, materials, E15-E20), A-2 (harmonic set, assumptions, inverse sizing, space claim) and A-3 (equation explorer engine side: evaluable records, drift guard, A3 traceability, A4 notes) complete
  guide: magcoupling-rs/README.md (tests, regenerating data, porting a module, Python-to-Rust translation rules, applying deviations)
  lib: magcoupling (src/lib.rs re-exports compute_all, headline, DesignInputs, DesignResults)
  engine: src/engine/ (pure std, wasm32-clean; one module per Python module)
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/03-structure.yaml`, replace:

```yaml
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

with:

```yaml
    sweeps: "sweeps.rs (Gap sweep and Pole sweep sheets: GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES; SweepRow with 26 fields, 338 and 156 table cells; SweepContext with 20 fields (Python's 19 plus bond_inner, read only by E4); gap_sweep, pole_sweep)"
    api: api.rs (DesignInputs/DesignResults groups in Python order, the complete Python DesignResults; compute resolves the part materials first (material_library::resolve) and feeds the values in effect; compute_all applies Deviations::ALL; headline and HEADLINE (the 15 dashboard numbers, Python order, read with ResultSet::get, about 0.4 us per call in release); DesignInputs::validate; test-only compute_all_with, DesignInputs::defaults_with)
    ported: [meta.rs, compat.rs, constants.rs, calibration.rs, library.rs, grades.rs, material_library.rs, warnings.rs, model.rs, metal_design.rs, materials.rs, temperature.rs, clamps.rs, sweeps.rs, api.rs]
    explain: "explain/ (Addendum A2-A4 equation explorer, engine side, plan A-3: markup.rs formula and symbol grammar, parser and tree (inf, nan, ceilto, floorto, fmt, fmtnum, concat); eval.rs evaluator with Trace (value and condition terms, arms taken); tables.rs static tables a formula reads; record.rs Record/Family/Eval and records/ (torque, demagnetization, slip_heating, temperature, clamps, dashboard, geometry; #[rustfmt::skip] data); registry.rs Registry::build, equation_for, used_by, graph, upstream, downstream, upstream_inputs, term_rows, family_members, term_style, modified_assumptions_upstream, choices, corrections_upstream; symbols.rs SYMBOLS; scope.rs SCOPE (decision 31 plus the G1 geometry callouts, 169 paths, all explained); notes.rs NOTES (17 A4 notes, START_HERE, Review gate, note_for); render.rs plain text)"
    addendum_only: [assumptions.rs, sizing.rs, housing.rs, explain/]
    remaining: [fields3d (M3)]
  features: gui and app (declared empty, M4); workbook-parity (test-only, enabled by a self dev-dependency)
  tests: tests/ (the integration tests parity.rs, differential.rs, python_schema.rs, schema.rs, deviations.rs, static_data.rs, robustness.rs, grades.rs, material_library.rs, material_links.rs, assumptions.rs, sizing.rs, explain.rs (the drift guard, the registry, the scope, the A3 traceability and the A4 note links); common/mod.rs holds the shared differential Case loader and the PORTED_INPUTS / PORTED_RESULTS ratchets)
  test_data: tests/data/ (reference_values.json = copy of the vendored snapshot; input_schema.json via MAGCOUPLING_BLESS=1 cargo test --test schema; python_schema.json, static_data.json, differential/<group>.json (10 result groups plus helpers.json) and differential/full.json via tools/gen_differential.py; deviations/E3.json, E4.json and E5.json, the golden files of the broad corrections, via MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells)
  gate: linkage-sim-rs/scripts/gate.sh gates 4-7 (magcoupling-rs test, clippy -D warnings, wasm32 check; Python parity suite + gen_differential.py --check)
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/04-memory.yaml`, replace:

```yaml
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

with:

```yaml
  - "RESOLVED (Addendum A-2): tests/assumptions.rs documents the C45 override (no effect with E20 and coercivity source 1; moves results with source 0); C45 keeps its NdFeB slider and a positive beta is typed (decision A2-5: widening would regenerate the differential data)"
  - "M3: fields3d.run gives 1.034632e-5; times 4 (E5) and formatted .4g that is 4.139e-5, not the M2 default 4.14e-5 (C125 differs by about 0.00009 W). M3 should pin the M2 default or expect the difference"
  - "Tidy after M2: magcoupling-rs/src/engine/temperature.rs is about 1,350 lines, past the architecture's ~800-line split point; split mechanically (inputs, results, compute, tests) when no correction work is in flight"
  - "Open, not in Addendum A-2's scope (the A3 panel lists no bondline assumption): E4 as approved adds the inner bondline input to the pole-sweep keyed-wall term, while the block-fit term keeps the workbook literal + 0.05 (the same bondline), so at bond_inner != 0.05 the two terms of the max() disagree (sweeps.rs pole_sweep); E4's test and golden file probe only 0.05. Owner: a deviation proposal (plan A-3 explains the engine as it is and did not change E4) (changing E4 is a registered deviation and needs the user's approval); trigger: making the pole-sweep block-fit bondline an input, or any change to E4"
  - "RESOLVED (Addendum A-2, audit M9): the engine flags f_end <= 0 (model.end_effect_check, calibration.end_effect_check, model::end_effect_in_range for sweep rows); M4 greys the numbers computed from the pull-out; inverse sizing never counts such a value"
  - "M4 design questions (decision 28): the M41 cap thread sits below the 41.33 mm cup body OD; the boss OD is 22 mm on Metal design and 25 mm on Shaft clamps. M4 draws the axial housing in effect (housing.hub_length_mm, cup_depth_mm, retainer_span_mm: A-2 decision A2-8)"
  - "M4: the sizing mode, target torque and free variable are GUI state (A-2 decision A2-9); sizing::solve takes them as arguments. Decide whether design files and share links record them"
  - "RESOLVED (Addendum A-3): the dependency-graph traceability test (tests/explain.rs: soundness for every input, sensitivity for the 15 assumption inputs, each reaching an explained result, with the cancellations listed with reasons); the records include tau7_Pa-tau11_Pa and the Calibration sums (model.tau#_Pa, calibration.tau#_Pa families)"
  - "RESOLVED (Addendum A-3 decisions G1-G7, as the user confirmed them before execution): G1 the geometry callouts are the face gap, corner gap, running clearance and its two parts, and the overshoot per axis with the dimension it measures (169 scope paths); G2 housing.space_claim_check stays unexplained (its text joins the exceeded axes); G3 the record batches follow the chain closure (demagnetization, slip heating, temperature, clamps, dashboard, geometry); G4 the outer ring's demagnetization block is computed with E20 off too and governs nothing; G5 17 notes (the spec's list, ferrite's cold side, the one-point calibration) and two more diagram kinds, DemagKnee and HeatingCurve; G6 the spec's A2 eval closure became the evaluable markup (a Rust closure only as a counted escape hatch, none used); G7 the spec's A3 test became two one-sided checks (soundness for every input; sensitivity on the active path at some design point, above rounding, the algebraic cancellations listed)"
  - "M4 (from plan A-3): the typesetter draws explain::markup::Expr; a selector compared with a code shows the choice label (explain::Registry::choices; materials.circuit_backiron borrows coupling.backiron's, explain::registry::RESULT_CHOICES); the corrected-vs-workbook marker reads Registry::corrections_upstream (a record's own corrections, then every upstream record's); term_style styles the panel's terms only (a cell-only result is never marked affected); a screw_sizes table key is a row index, shown as the size name; paint the six notes::Diagram kinds; build explain::Registry once at start-up and own it in the app state; only reviewed notes show (notes::note_for); tests/explain.rs release_notes_are_reviewed (ignored) is on the hands-on checklist"
  - "M3: a grade-mode ring's reverse fields are scaled with its own alpha(Br) (decision A2-7); the stored 3D fields do not separate the opposing ring's share, which live fields per ring would"
  - "M4: the workbook-parity feature is guarded by convention only (any downstream Cargo.toml can enable it). A compile_error! on all(feature = app, feature = workbook-parity) was NOT added: the self dev-dependency enables workbook-parity for every cargo test, so cargo test --features app would stop compiling. Decide the guard in the M4 plan (e.g. a CI check that the shipped build's feature set excludes it)"
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, replace:

```markdown

- **Explanation layer.** Each explained result registers an equation record:
  target id, display symbol, a display formula in a small markup (fractions,
  sub/superscripts, Σ, √), its term ids, unit, workbook cell, and an `eval`
  closure over term values. The engine code stays as ported (parity), and
  the explanation layer is separate.
- **Drift guard (test).** For every equation record, `eval` over the engine's
  term values must reproduce the engine's result (1e-9 relative) at defaults
  and at the M2 differential-test input sets. The equation shown is provably
  the one that produced the number.
```

with:

```markdown

- **Explanation layer.** Each explained result registers an equation record:
  target id, display symbol, a display formula in a small markup (fractions,
  sub/superscripts, Σ, √), its term ids, unit and workbook cell. The formula
  markup is itself what the drift guard evaluates over the term values; a
  Rust closure only where the markup cannot state it, counted and reviewed
  by hand (plan A-3: none). The engine code stays as ported (parity), and
  the explanation layer is separate.
- **Drift guard (test).** For every equation record, its evaluation over the engine's
  term values must reproduce the engine's result (1e-9 relative) at defaults
  and at the M2 differential-test input sets. The equation shown is provably
  the one that produced the number.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, replace:

```markdown
- **Equation drift guard:** runs over every equation record (A2).
- **Assumptions:**
  - at defaults, results are bit-identical to workbook parity;
  - changing each assumption changes every dependent result and no
    independent one, using the equation registry's dependency graph.
- **Materials:**
  - each warning rule fires exactly on its condition;
  - a non-ferromagnetic back iron switches the circuit factor.
```

with:

```markdown
- **Equation drift guard:** runs over every equation record (A2).
- **Assumptions:**
  - at defaults, results are bit-identical to workbook parity;
  - changing each assumption changes no result outside its dependents in the
    equation registry's dependency graph, and every numeric result on its
    active path at some design point, above rounding (algebraic
    cancellations listed with their reasons).
- **Materials:**
  - each warning rule fires exactly on its condition;
  - a non-ferromagnetic back iron switches the circuit factor.
```

In `C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/README.md`, replace:

```markdown
override with the axial housing that follows it (hub length, cup cavity depth,
retainer span), inverse sizing (A1), the housing autofit suggestion and the space
claim, one aluminium modulus, and each grade-mode ring's own alpha and density
(decisions A2-1 to A2-9). Next: Addendum A-3 (explanations), then M4.

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
```

with:

```markdown
override with the axial housing that follows it (hub length, cup cavity depth,
retainer span), inverse sizing (A1), the housing autofit suggestion and the space
claim, one aluminium modulus, and each grade-mode ring's own alpha and density
(decisions A2-1 to A2-9). Addendum A-3 (explanations) complete: the engine side of the equation
explorer (`src/engine/explain/`: one evaluable markup per result, proven by the drift guard, over
decision 31's paths, the geometry callouts and the terms they need to reach inputs), the A3
traceability test over every assumption, and the 17 A4 teaching notes behind a physics-review gate.
Next: M4.

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
```

- [ ] **Step 2: The update tracker**

Run this script (it edits the file in place and asserts what it expects to find):

```bash
python - <<'EOF'
import datetime, pathlib
p = pathlib.Path(r"C:/Users/Cole/source/repos/lsim-mag-a3/docs/ai/05-update-tracker.md")
s = p.read_text(encoding="utf-8")
anchor = "Reverse chronological (newest at top).\n\n---\n\n"
assert s.count(anchor) == 1
entry = f"""## {datetime.date.today():%Y-%m-%d} — Magcoupling Addendum A-3: explainability (A2, A3 traceability, A4)
- `magcoupling-rs`: the engine side of the equation explorer (`src/engine/explain/`): one formula
  markup per explained result that is both what the M4 panel typesets and what the drift guard
  evaluates (no Rust closures); the registry (`Registry::build`, `equation_for`, `used_by`, the
  dependency graph, term rows and A3 styles); 392 equations over decision 31's chains, the dashboard
  and the geometry callouts (decision G1: 169 scope paths, every one drilling down to inputs).
- The drift guard proves every record against the engine at the defaults and every differential
  case under 16 augmentations (harmonic sets, back irons, grade mode, axial override, ferrite from
  the inputs at two betas and unrated, the sleeve and cap picks, E17's free-space fields), every
  `cases` arm taken and every value term of every record varied.
- A3 traceability: soundness for every input, sensitivity for the 15 assumption inputs with five
  cancellations listed; every assumption reaches and moves an explained result.
- A4: 17 teaching notes drafted from the M1 derivations, physics-reviewed and signed off; the
  start-here order; `note_for` shows reviewed notes only; the release gate is green.
- Engine: Rust-only reads only (harmonics 7-11 parts, the amplitudes and E7 angles, each ring's E20
  block, the materials in effect); parity, the differential data and every registry probe unchanged.

"""
p.write_text(s.replace(anchor, anchor + entry, 1), encoding="utf-8", newline="\n")
print("tracker entry added")
EOF
```

- [ ] **Step 3: Format, then run the gate**

Run:

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml
```

Expected: no output (every block below is already in rustfmt's layout).

The gate runs `cargo test`, `cargo clippy --all-targets -- -D warnings` (gate 5) and the wasm32 check (gate 6) on `magcoupling-rs`, the linkage crate's gates and the Python oracle (gate 7).

Run:

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-a3/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task17.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task17.log; grep -c "SKIP gate 7/7" C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/target/gate-task17.log
git -C C:/Users/Cole/source/repos/lsim-mag-a3 checkout -- docs/chebyshev_lambda
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check
```

Expected: `exit=0`; the last three lines are the oracle's `1159 passed in ...s` summary, `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` and `GATE PASS`; the `grep -c` prints `0` (no `SKIP gate 7/7` line); `git checkout` restores the regenerated PNGs. Then `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --check` prints nothing.

- [ ] **Step 4: Check the default headline one last time**

Run:

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-a3/magcoupling-rs/Cargo.toml --test deviations all_corrections_together_give_the_reviewed_headline 2>&1 | grep "test result"
```

Expected: `test result: ok. 1 passed` (the default headline with every correction on: unchanged by every task).

- [ ] **Step 5: Commit**

This task runs on `sonnet`: in the `Co-Authored-By:` line replace `Claude Opus 5.5` with your own model's name (the line names the model that wrote the commit; Global Constraints).

Run:

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-a3 add magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md
git -C C:/Users/Cole/source/repos/lsim-mag-a3 commit -F - <<'EOF'
docs(magcoupling): Addendum A-3 status, invariants and open items

README status; docs/ai system invariants (the drift guard, A3 traceability),
structure (the explain module), memory (the A-3 traceability item resolved,
decisions G1-G7, the M4 items) and the update tracker; the spec's A2 and A3
wording follows the evaluable markup and the two-sided traceability test.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
git -C C:/Users/Cole/source/repos/lsim-mag-a3 status --short
```

Expected: one commit on `magcoupling/addendum-a3`; `status --short` prints nothing (no PNG under docs/chebyshev_lambda, no stray file).

---

## Self-review

Run against the spec (Addendum A2 to A4 and "Addendum testing"), decision 31 and the A-1 and A-2 decisions after writing the plan.

**1. Spec coverage.**

| Requirement | Where |
|---|---|
| A2 explanation layer: a record per explained result (target, symbol, formula markup with fractions, scripts, Σ, √, terms, unit, cell); the engine stays as ported | Tasks 2 and 3 (markup, records, registry; unit, label and cell from the metadata), 6 (the markup the other chains need), 1 and 7 (Rust-only reads only, and decision G4's outer-ring block) |
| A2 drift guard: every record reproduces the engine (1e-9) at the defaults and the differential input sets | Task 3 (the guard, with 11 augmentations and the anti-vacuity checks, the term-level one among them), Tasks 8 and 9 (five more augmentations) |
| A2 hover, panel, breadcrumb, "used by", leaf highlighting, typesetter | M4's; this plan gives the data: `equation_for`, `term_rows`, `used_by`, `is_leaf_input`, `term_kind`, `symbol`, `family_symbol`, `family_members`, the parse tree, a parse tree the typesetter cannot draw two ways (Task 3), `choices` and `corrections_upstream` (Task 4), `render::plain` for the fallback |
| Decision 31's scope: the dashboard, the geometry callouts and the five chains | Tasks 3 and 8 to 13 (169 paths, decision G1 for the callouts), each chain drilling down to inputs |
| A3 traceability: a changed assumption flows through every dependent equation; the assumption styling, the changed dot, the banner and the reset | Task 4 (soundness and sensitivity; `term_style`, `modified_assumptions_upstream`; the banner and reset are A-2's `assumptions` module), Task 14 (non-vacuous over every assumption) |
| Addendum testing: "changing each assumption changes every dependent result and no independent one, using the equation registry's dependency graph" | Tasks 4 and 14, with the spec's wording made checkable in Task 17 (decision G7; cancellations listed) |
| A4 notes: 2 to 6 sentences at Physics 2 level, an optional watch-out and diagram, about 15 to 20 ideas on the spec's list, the A5 warnings' physics | Tasks 5 and 15 (17 notes, decision G5; the unit tests enforce the sentence count and sources) |
| A4 "Explain" follows the open equation; "start here" order | `note_for(path)` (Task 5); `START_HERE`, each entry its note's own equation (Task 15) |
| A4 accuracy gate: drafted from the M1 derivations, checked by a physics reviewer before release | Each note cites its sources (Task 15); Task 16's reviews and sign-off; `note_for` hides drafts; the release gate |
| A6: ferrite's opposite-sign β "gets a teaching note" | `ferrite_cold_demag` (Task 15) |
| Physics changes get a dedicated physics reviewer naming the M1 derivations | Every batch's review step (Tasks 3, 8 to 13), the cancellations (Tasks 4 and 14), Tasks 1 and 7 on the session model, Task 16 |

Gaps found and closed while writing: the traceability test passed by absence for the thermal and clamp assumptions while only the torque chain was explained (Task 14 makes it fail on an assumption with no explained result); the geometry callouts had no list (decision G1); inputs that bypass `set` had no explorer test (Review Focus 5, Task 14).

**2. Placeholder scan.** The assembler refuses the plan if it contains a placeholder marker or a scratch path; every code step carries the code (each block was produced from a commit that compiled and passed), every command its expected output from the replay.

**3. Type consistency.** Every name a task's Interfaces block produces is used by later tasks under the same name; the replay applied every task's blocks in order on the post-A-2 base and ran each task's commands, so a renamed type, field or function would have failed there.

**4. Review Focus.** Each of the five lines names the tests that pin it, in the task that owns the code.

## Self-review record

An independent review of the replayed plan returned four blocking findings and six groups of smaller ones. Each is fixed in the task that owns the code, the plan was rebuilt from the fixed commits and replayed again from the post-A-2 base (the Verification record above is that replay). What changed, finding by finding:

| # | Finding | Fix (task) | Proof |
|---|---|---|---|
| 1 | Blocking: the drift guard saw 18 (record, value term) pairs at one term value only (the cold side against both betas and the outer ring's Br coefficient, the hub, cup and web losses against E17's free-space fields, the magnet mass against the outer density), so a term could be replaced by a literal and pass. | The guard records each value term's distinct values per record and fails on any seen at one value (Task 3). Two augmentations reach the 18: "ferrite inputs, beta 0.002" (the rings swapped, a second positive beta, the outer ring bonded NdFeB with its own α and density; Task 8) and "E17 fields" (the three free-space fields off their defaults under a 6061 back iron; Task 9). A typed value outside its slider (a positive β) is now nudged around itself instead of dropped (Task 4). Review Focus 1 now names the ferrite augmentations, not the "coercivity from the inputs" point, which runs at the default β. | With the two augmentations removed the new check lists exactly those 18 pairs; with them it passes. Replacing β by the literal 0.0035 in `temperature.demag.inner_cold_limit_C` now fails the guard. The two μ0 inputs need no exemption: the differential cases already set each to its range ends. |
| 2 | Blocking: `temperature.slip_loss.magnets_W` read the inner width in m and in mm, so its term list could not satisfy the equation shown. | The volume reads all three block dimensions in m (no `1e-9`); `Registry::build` refuses a formula that reads one path in two units (Task 3), with a case in `a_bad_record_is_refused_with_every_reason`. | No other record trips the check; the drift guard holds at 1e-9. |
| 3 | Blocking: 18 equations could not be typeset unambiguously (an inline `a / b` as a factor, `R_g^{cal}^2`, a Σ whose extent is unclear). | `Registry::build` refuses, unless parenthesized, an inline `a / b` as an operand of `*`, `·` or `/`; a Σ or `peak` as an operand of a product, quotient or power or left of `+`/`−`; and a power of a term, local or `exp` whose symbol has a superscript (Task 3; the rules sit beside the markup table, with four bad-record cases). The flagged records use `frac`, `|m` conversions or parentheses (Tasks 3, 8, 9, 11). The plain rendering, the sheet the physics reviewer signs, wraps a stacked fraction that is a factor (`(π/4) (D² − d²)`), pinned on `model.f_end`. | The rules flag 20 equations: the reviewed 17 (one of them both a double superscript and a Σ factor) and three `a / b / c` reads of the reference field (`temperature.demag.h_ref_kA_m` and the two ring limits' bindings). All are rewritten; the guard reproduces each at 1e-9. |
| 4 | Blocking (physics): two notes wrote ω for both the field's angular frequency and the slip speed, a factor p = 5 apart. | `slip_heating` writes δ = √(2/(ω_e μ0 μr σ)) with ω_e = 2πf and the shell loss σ (ω_s r B)²/2 with ω_s the slip speed; `a5.high_conductivity_sleeve_or_liner` likewise; the records file's symbol line and shell comment name both (Tasks 9 and 15). Task 16's reviewer prompt asks for exactly this check. | The notes unit tests (2 to 6 sentences, sources) and the note links test pass. |
| 5 | Physics wording: the steady rise depends on P too; the 6061 drop is more than a third (2.69 to 1.70 N·m, the bench factor included); the low-saturation threshold is 1.7 T and the fallback is the design flux input; f_end takes off c_end pitches in total, half at each end; the magnetization is a rectangular wave; optionally, both α and β in the onset, and why sinh. | All seven sentences rewritten (Tasks 5 and 15); the harmonics note's second sentence and the `SquareWaveHarmonics` doc follow the rectangular wave. The low-saturation sentence also says a steel with no design value of its own is flagged, as `warnings.rs` does. | Every note keeps 2 to 6 sentences; Task 16 reviews them. |
| 6 | Soundness blind spots: 81 of 870 non-redundant graph edges could be dropped unseen at the design points (verdict thresholds, μ0 never nudged, the housing's axial branch, the clamp checks). | "axial 8 mm" joins the trace points (Task 14); every number and count is also nudged to its slider's two ends, which flips threshold verdicts and nudges μ0, whose slider is narrower than the step (Task 4); soundness fails for an input that no accepted nudge changes any result of at any point, the two inputs no result reads listed with reasons (Task 4). | An exhaustive edge-drop census over the final records (each edge dropped in turn, detected if some accepted nudge moves a result only that edge reaches): 81 undetected before (the review's count), 57 with the axial point and the outside-slider nudge but no slider ends, 4 with them (`temperature.demag.{inner,outer}_magnet_limit_C` against each ring's Br, `temperature.demag.cold_onset_skipping_C` against `cold_ring`, `clamps.table[1].geometry_ok` against `engagement_req_mm`). The non-vacuity check found the two inert inputs itself (`materials.steel.density_g_cm3`, `clamps.key_width_mm`). |
| 7 | The bypass-set test asserted only that some record evaluated. | It counts values and `EvalError`s and asserts they sum to every equation, values the majority (Task 14). | Passes. |
| 8 | Governance: two spec rewordings were listed as settled without a decision; a Global Constraint contradicted decision G4. | Decisions G6 (the markup is the evaluation) and G7 (two one-sided traceability checks) join the table; Task 0 asks G1 to G7; Task 17's memory entry records them; the Rust-only constraint names G4's exception. | Text only. |
| 9 | M4 API gaps: choice labels of a result holding a selector code; the corrected-vs-workbook marker when the correction sits upstream; styling of cell-only results. | `Registry::choices` with `RESULT_CHOICES` (checked at build), `Registry::corrections_upstream` (own corrections first, each once), and `term_style` documented as styling panel terms only (Task 4); `the_panel_labels_selector_codes_and_names_upstream_corrections` pins them; docs/ai records them for M4 (Task 17). | Passes from Task 4 on. |
| 10 | Model tiers: a new cancellation is a physics judgement; Task 16's commit claimed a green release gate unconditionally. | Task 14's Model line and the Global Constraint send any new `CANCELLATIONS` entry to the session model (none appeared: the census and sensitivity runs added none); Task 16's commit body is conditional on all 17 notes being approved. | Text only. |

The replay ran Task 16's sign-off script as if every note were approved, so the notes show as reviewed in the replayed tree; the physics reviews themselves (each batch's and Task 16's) have not run and remain the plan's accuracy gate.
