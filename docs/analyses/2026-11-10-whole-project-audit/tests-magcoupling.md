# Audit findings: tests-magcoupling
Scope read: all 16 magcoupling-rs/tests/*.rs read by structure and by search (helpers, ignores, data users, assertion-free tests; explain.rs, deviations.rs, shaft_joint.rs, sizing.rs skimmed in part), tests/common/mod.rs in full, all of tests/data/** (checked which test reads each file), reference/magcoupling-py layout, README, pyproject, tools, and its users (gate.sh, docs, Rust tests).

## MT-01: Test setup helpers copied between test files instead of living in tests/common
- Kind: inefficient
- Where: magcoupling-rs/tests/gearbox.rs:45 and tests/shaft_joint.rs:32 (`fn design(edit)`, identical 5 lines); tests/gearbox.rs:52 `custom_dimensions`, tests/shaft_joint.rs:52 `manual`, tests/grades.rs:416 (inline), tests/explain.rs:2685 `on_grade`, tests/grades.rs:260 and :308 (both rings on manual dimensions with a grade: `part_inner.clear(); part_outer.clear(); grade_inner = ..; grade_outer = ..`); the bare `part_inner = String::new(); part_outer = String::new()` pair again at sizing.rs:299, 1291, 1347 and robustness.rs:382; tests/explain.rs:2669 and tests/shaft_joint.rs:136 (`fn slider(path)`, same body, only the panic text differs); tests/deviations.rs:35 and tests/schema.rs:33 (`const BLESS_VAR: &str = "MAGCOUPLING_BLESS"`)
- Evidence: gearbox.rs:45-49 and shaft_joint.rs:32-36 are the same function `fn design(edit: impl Fn(&mut DesignInputs)) -> DesignInputs { let mut inputs = DesignInputs::default(); edit(&mut inputs); inputs }`. gearbox.rs:54-57 and shaft_joint.rs:54-57 and grades.rs:416-419 are the same four statements (`clear()` x2, `grade_inner = x.to_owned()`, `grade_outer = ..`). explain.rs:2669 and shaft_joint.rs:136 both are `input_meta(path).range.unwrap_or_else(|| panic!(..))`. common/mod.rs already holds `input_meta`, so `slider` belongs beside it.
- Why it matters: the owner flags repetition; a change to how a design is put on manual dimensions (a new field) must be made in 5 places, and the copies already drift (clear() in some, String::new() in others, different panic texts).
- Suggested action: move `design`, a `manual_grades(base, inner, outer)` and `slider` into tests/common/mod.rs (it already allows dead code per test crate); keep the per-file `quoted` and `close` helpers, which differ on purpose (different verdict formats and tolerances 1e-12, 1e-15, 1e-9). Share BLESS_VAR as a common const.
- Fix class: needs-judgment
- Confidence: certain for design, slider, BLESS_VAR (read both copies); likely for the grade-pair helpers (each copy differs slightly in what else it sets).

## MT-02: Differential JSON (about 10 MB) is parsed again by almost every test that needs it
- Kind: inefficient
- Where: magcoupling-rs/tests/common/mod.rs:324 (`load_cases`, no cache); callers tests/differential.rs:43 (once per module test, 10 tests), :323, :349 (all 10 modules again), :491, :515 (all 10 again); tests/explain.rs:893 and :1093 (both load all 12 files), :5179; tests/deviations.rs:2030 (all 12 files)
- Evidence: `grep OnceLock|LazyLock|thread_local tests/` finds nothing. Sizes: gap_sweep.json 2.1 MB, full.json 1.9 MB, temperature.json 1.5 MB, pole_sweep.json 1.2 MB, clamps.json 1.1 MB. In differential.rs alone the 10 module files are read and parsed about 4 times each (about 40 parses) in one test binary, explain.rs twice over all 12, deviations.rs once.
- Why it matters: repeated serde_json parse of megabytes in a gate step that runs in debug builds; cost not measured here.
- Suggested action: cache the parsed cases per module in common (a `OnceLock<BTreeMap<..>>` returning `&'static [Case]`), so each binary parses a file once.
- Fix class: needs-judgment
- Confidence: likely (call counts are certain; the time saved is not measured, a `cargo test --test differential -- --report-time` would settle it)

## MT-03: tests/data/reference_values.json is a 147 kB byte-for-byte copy kept equal by a test
- Kind: not needed
- Where: magcoupling-rs/tests/data/reference_values.json; reference/magcoupling-py/tests/reference_values.json; tests/parity.rs:88-98 (`snapshot_copy_equals_the_vendored_snapshot`); tests/common/mod.rs:210-218 (`snapshot`); README.md:790
- Evidence: parity.rs:89-93 reads both and asserts equality ("copy the vendored file again"). The same test crate already reads other files outside tests/data through `repo_path` (grades.rs:16, material_library.rs:20, deviations.rs:200, 283), so a copy is not needed to reach it.
- Why it matters: two copies of one fixture plus a test whose only job is to detect the copies differing; each snapshot refresh is a manual copy step.
- Suggested action: have `snapshot()` read `repo_path("reference/magcoupling-py/tests/reference_values.json")`, delete the copy and the equality test, and update README lines 164, 703, 790.
- Fix class: owner-decision (the copy may be kept so that magcoupling-rs can be built or published alone)
- Confidence: likely; the owner's reason for a standalone crate would settle it

## MT-04: A no-op `let _ = &statics;` left in the traceability test
- Kind: not needed
- Where: magcoupling-rs/tests/explain.rs:5372
- Evidence: `let statics = r.downstream(path); let active = ..; let _ = &statics;` and then `statics` is used at :5387 (`!statics.contains(&eq.target)`), so the statement silences nothing.
- Why it matters: leftover from an earlier version where `statics` was unused; reads as if a value were being discarded.
- Suggested action: delete line 5372.
- Fix class: mechanical-safe
- Confidence: certain

## MT-05: Test and generator doc comments no longer match what is tested
- Kind: inconsistent
- Where: magcoupling-rs/tests/static_data.rs:1-2; reference/magcoupling-py/tools/gen_differential.py:35
- Evidence: static_data.rs:1-2 "Static engine tables (magnet library, and later adhesives, alloys, screw sizes, checklists)": tests exist for all of them (static_data.rs:70-200: harmonics, validation items, alloys, adhesives, screw sizes, machining steps, sweep variables, table columns). gen_differential.py:35 ends "... and the sweep variables so far".
- Why it matters: "later" and "so far" say the work is unfinished when it is done.
- Suggested action: drop "and later" and "so far" (gen_differential.py is the unchanged oracle's tool: edit only if the owner allows, otherwise fix the Rust side only).
- Fix class: mechanical-safe (Rust side); owner-decision (the oracle's tools folder is marked do-not-edit only for magcoupling/, so low risk)
- Confidence: certain

## MT-06: The ignore guard checks six files, but its name and doc say "only the review tools are ignored"
- Kind: inconsistent
- Where: magcoupling-rs/tests/explain.rs:1712-1765
- Evidence: `sources` lists notes.rs, plain_text.rs, explorer.rs, guidance.rs, dashboard.rs and tests/explain.rs; the other 15 tests/*.rs files and every other src file are not scanned. Today every `#[ignore]` in magcoupling-rs (7) has a reason string and sits in a scanned file, so the test passes.
- Why it matters: an `#[ignore]` added to, say, tests/sizing.rs would silently disable a check and this guard would stay green.
- Suggested action: scan every tests/*.rs and src/**/*.rs (a directory walk instead of `include_str!` of six names), or narrow the test's name and doc to the six files.
- Fix class: needs-judgment
- Confidence: certain (read the list); impact is potential only

## MT-07: The vendored oracle's independent audit suite and its tools are run by nothing automatic
- Kind: inconsistent
- Where: reference/magcoupling-py/audit/ (17 test files under audit/tests, 11 modules under audit/references, audit/tools/coverage_table.py, audit/tools/group_candidates.py, audit/tools/placeholder_sensitivity.py); linkage-sim-rs/scripts/gate.sh:140-141; docs/ai/03-structure.yaml:29-31; .github/ (no mention)
- Evidence: gate 12 runs `"$PY" -m pytest -q tests -p no:cacheprovider` and `gen_differential.py --check`, not `audit/tests`. No file under .github mentions pytest or audit. The baseline pytest run covered the linkage package's `tests`, not this folder. 03-structure.yaml:29 lists `audit_suite` and `audit_tools` as commands to run by hand. `coverage_table.py` and `group_candidates.py` are named by no code, only by that yaml line.
- Why it matters: the audit suite (about 40 modules, scipy-based) can rot unseen; its health is unknown to this audit (I did not run pytest).
- Suggested action: either add `audit/tests` to gate 12 (the yaml already has a `both_in_one_session` command), or record that it is by-hand only and when it was last green.
- Fix class: owner-decision
- Confidence: likely; running `pytest -q audit/tests` in the oracle venv would settle whether it still passes

## MT-08: The oracle's README tells readers to extend and edit the code that reference/README.md says never to edit
- Kind: inconsistent
- Where: reference/magcoupling-py/README.md:5 ("written to be dropped into another repository and driven from a GUI"), :61-76 ("Using it from a GUI", "Examples"), and the "Extending" section (new magnet in library.py, new adhesive, new screw size, change model.HARMONICS); reference/README.md:5-9
- Evidence: reference/README.md: "Do not edit `magcoupling-py/magcoupling/`: parity and differential tests for the Rust port compare against it unchanged." The oracle README's Extending section says "add a `MagnetSpec` row in `library.py`" and "change `model.HARMONICS`". Its "Differences from the workbook" also describes the Python engine, not the Rust port that now ships.
- Why it matters: following the Extending section changes the oracle, and the Rust differential data (gate 12 `--check`) then goes stale.
- Suggested action: add a short banner at the README top ("vendored oracle, do not edit; see reference/README.md") and drop or mark the Extending section as historical.
- Fix class: needs-judgment
- Confidence: certain on the contradiction

## MT-09: Two example front ends in the oracle that nothing runs, tests or links to
- Kind: not needed
- Where: reference/magcoupling-py/examples/gui_integration.py, reference/magcoupling-py/examples/tk_demo.py
- Evidence: the only mentions are reference/magcoupling-py/README.md:75 and tk_demo.py's own docstring (`Run: python examples/tk_demo.py`). No gate, test or Rust file uses them; the GUI they illustrate is now the Rust egui app.
- Why it matters: dead sample code in an oracle folder whose contract is "unchanged, used only for parity".
- Suggested action: delete both files and the README sentence, or leave as is.
- Fix class: owner-decision
- Confidence: certain they are unused; the owner decides whether they have value as history

## Leads settled
- leads 1, `magcoupling-rs` [dev-dependencies] `magcoupling-rs`: not real. It is the self dev-dependency that turns on the `workbook-parity` feature for the integration tests (Cargo.toml:113-114).
- leads 3, `reference/magcoupling-py/audit/tests/conftest.py` `pytest_configure`, `pytest_collection_modifyitems`, `pytest_runtest_makereport`, `pytest_sessionfinish`: not real. They are pytest hooks found by name.
- leads 3, `tests/...` (setup_method, _solve_sixbar, build_non_grashof): outside my part (linkage Python).
- leads 4, duplicated runs inside my part: none listed except `magcoupling-rs/src/engine/explain/scope.rs` and `notes.rs` (src, not mine). My own duplicate search found the helper copies in MT-01.
- leads 6, `edge_points` at tests/explain.rs:195 (561 lines): not a finding, it is one data table of labelled edge inputs with its own `on_edge` and `joint_edge` checks. Same for the 253-line `the_shaft_joint_note_s_claims_hold_at_every_probe_design`.
- leads 6, `compute` at reference/magcoupling-py/magcoupling/temperature.py:413 (135 lines): not a finding, the oracle is kept unchanged on purpose.
- leads 6, source file sizes of tests/explain.rs (6076 lines) and tests/deviations.rs (2784): not a finding by itself.
- baseline "identical files reference_values.json": real, see MT-03.
- baseline `#[ignore]` count 7: all 7 in magcoupling-rs carry a reason string and are the review tools the guard in explain.rs lists; only the guard's reach is a finding (MT-06).
- Data files: every file under tests/data is read by a test (differential/*.json by load_cases or helpers.json by differential.rs:563; deviations/E3-E5 by the registry's `changes_file`; input_schema, python_schema, static_data, reference_values by schema.rs, python_schema.rs, static_data.rs, parity.rs). None unused.
- Tests that can never fail: none found. The 9 one-line differential tests delegate to `check_module`, which asserts. `sizing_never_panics_on_extreme_designs` and `robustness.rs` `let _ = compute_all(..)` are deliberate no-panic checks.
- Mojibake seen in deviations.rs:126 etc. through PowerShell is the terminal's, not the file's (Grep for the mojibake bytes finds none).

## Not read
- tests/explain.rs lines about 2700-5000 and 5400-6076, tests/sizing.rs, tests/shaft_joint.rs, tests/deviations.rs (lines 140-2784 beyond searches) were searched and sampled, not read line by line.
- reference/magcoupling-py/magcoupling/*.py engine modules (do-not-edit oracle), audit/references/*.py and audit/tests/*.py bodies: checked only for imports and users.
- Python parity tests (tests/test_parity.py, test_api.py) read only at their headers.
