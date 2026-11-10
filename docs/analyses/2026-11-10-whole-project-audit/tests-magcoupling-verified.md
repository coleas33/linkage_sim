# Verified: tests-magcoupling

## MT-01: Test setup helpers copied between test files instead of living in tests/common
- Verdict: PARTLY. Holds: `fn design(edit)` is the same 5 lines at gearbox.rs:45-49 and shaft_joint.rs:32-36; `fn slider(path)` is the same body at explain.rs:2669-2673 and shaft_joint.rs:136-140 (only the panic text differs: "has no slider" vs "has a slider"); `const BLESS_VAR: &str = "MAGCOUPLING_BLESS"` at deviations.rs:35 and schema.rs:33; the four-statement "both rings on manual dimensions with a grade" appears at gearbox.rs:54-57 (`custom_dimensions`), shaft_joint.rs:54-57 (`manual`) and grades.rs:416-419; the `part_inner/part_outer = String::new()` pair at sizing.rs:299, 1291, 1347 and robustness.rs:382. Does not hold as stated: grades.rs:308 sets only one ring (inner or outer, inside an if/else) and explain.rs:2685 `on_grade` handles each ring by a `Rings` choice, so those two are not copies of the same helper; grades.rs:260 is both rings but with `String::new()` and a fixed "Y30".
- Checked: Grep of the helper names and of `part_inner = String::new|part_inner.clear` over magcoupling-rs/tests (read each hit with context); tests/common/mod.rs has `#![allow(dead_code)]` (line 4) and `input_meta` (136); every one of gearbox.rs, shaft_joint.rs, explain.rs, grades.rs, deviations.rs and schema.rs has `mod common;`.
- Safe to apply without judgment: no. It edits several test crates, the grade-pair helper has variants that differ on purpose, and the tests are what the gate runs.

## MT-02: Differential JSON (about 10 MB) is parsed again by almost every test that needs it
- Verdict: PARTLY. Holds: `load_cases` (common/mod.rs:324-365) reads and parses with no cache; Grep for OnceLock|LazyLock|thread_local under tests/ finds nothing; callers are differential.rs:43, 349, 491, 515, explain.rs:893-894, 1093-1095, 5179 and deviations.rs:2030-2033; file sizes are as stated (clamps 1.13 MB, full 1.94 MB, gap_sweep 2.10 MB, pole_sweep 1.16 MB, temperature 1.48 MB; the 10 module files plus full total about 10.2 MB). Not as stated: differential.rs:323 sits in `every_branch_is_reached`, which already keeps a local per-test cache (lines 319-323), so it parses each module once, not "all 10 again" beyond that test; and `differential_files()` returns 10 modules plus `full` = 11 files, not 12 (helpers.json is not loaded by `load_cases`). The time saved is unmeasured, as the finder says.
- Checked: common/mod.rs:61-110, 313-374; differential.rs:310-360; the Grep above; `Get-ChildItem` sizes of tests/data/differential.
- Safe to apply without judgment: no. A process-wide cache changes how tests share state and the return type of `load_cases` (callers take owned `Vec<Case>`), and the benefit is unmeasured.

## MT-03: tests/data/reference_values.json is a 147 kB byte-for-byte copy kept equal by a test
- Verdict: CONFIRMED
- Checked: both files are 146,849 bytes with the same SHA-256 (7B8AB886...1B07). parity.rs:87-98 `snapshot_copy_equals_the_vendored_snapshot` compares them; common/mod.rs:210-218 `snapshot()` reads the copy; `repo_path` (152-156) already reaches files outside tests/data; README.md lines 164, 703, 790 name the copy (790: "The snapshot copy must equal ... (a test checks)"); engine/calibration.rs:276 and docs/ai/03-structure.yaml name the snapshot too. No stated reason for the copy was found in README or docs.
- Safe to apply without judgment: no. It deletes a fixture and a test, and whether magcoupling-rs must build or publish alone is the owner's decision.

## MT-04: A no-op `let _ = &statics;` left in the traceability test
- Verdict: CONFIRMED
- Checked: explain.rs:5370-5372 (`let statics = r.downstream(path); let active = active_downstream(...); let _ = &statics;`), and `statics` is read at 5387 (`!statics.contains(&eq.target)`), so the borrow statement silences nothing. Nothing else mentions line 5372.
- Safe to apply without judgment: yes. A `let _ = &x;` on a value used later has no effect, so deleting it changes no behaviour on any target or feature.
- Exact change: magcoupling-rs/tests/explain.rs line 5372, delete `            let _ = &statics;`

## MT-05: Test and generator doc comments no longer match what is tested
- Verdict: CONFIRMED
- Checked: static_data.rs:1-2 reads "Static engine tables (magnet library, and later adhesives, alloys, screw sizes, checklists)"; the tests at static_data.rs:70-200 already cover harmonics, validation items, alloys, adhesives, screw sizes and machining steps, and sweep variables (Grep of `fn ` lists 8 tests). gen_differential.py:35 ends "and the sweep variables so far". No test or script reads either comment text.
- Safe to apply without judgment: yes for the Rust side (doc comment only); no for gen_differential.py (it sits in the vendored oracle's tools folder, owner decides).
- Exact change: magcoupling-rs/tests/static_data.rs lines 1-2, old
  `//! Static engine tables (magnet library, and later adhesives, alloys, screw`
  `//! sizes, checklists) equal the Python engine's, value for value.`
  new
  `//! Static engine tables (magnet library, adhesives, alloys, screw sizes,`
  `//! checklists) equal the Python engine's, value for value.`

## MT-06: The ignore guard checks six files, but its name and doc say "only the review tools are ignored"
- Verdict: PARTLY. Holds: the guard (explain.rs:1712-1765) scans exactly six sources (notes.rs, plain_text.rs, explorer.rs, guidance.rs, dashboard.rs, tests/explain.rs); an `#[ignore]` anywhere else (the other tests/*.rs or other src files) would not be seen. Does not hold: its doc is scoped to "in the files that hold them" (1712-1714), so the doc does not claim more than the six files; only the test's name is general. The count of 7 holds: Grep `#\[ignore` over magcoupling-rs finds tests/explain.rs:1871, explorer.rs:2438 and 2941, guidance.rs:9682, plain_text.rs:1239 and 1405, notes.rs:2423 (dashboard.rs, though scanned, has none); all carry a reason string and sit in scanned files.
- Checked: explain.rs:1712-1765; the Grep above.
- Safe to apply without judgment: no. A directory walk or a renamed test is a design choice (and `include_str!` of 6 names is how the guard stays compile-time).

## MT-07: The vendored oracle's independent audit suite and its tools are run by nothing automatic
- Verdict: CONFIRMED (small miscount: audit/references has 13 modules, not 11; audit/tests has 17 test files as stated)
- Checked: linkage-sim-rs/scripts/gate.sh:140-141 runs only `pytest -q tests` and `gen_differential.py --check`; Grep of `audit|pytest|python|gate` in .github/workflows/deploy-web.yml (the only file under .github) finds nothing; docs/ai/03-structure.yaml:29-31 lists `audit_suite`, `both_in_one_session` and `audit_tools` as commands to run by hand; Grep for audit/tests, coverage_table, group_candidates and placeholder_sensitivity outside docs/superpowers finds only docs and yaml. Globbed audit/tests (19 files incl. conftest.py and __init__.py), audit/references, audit/tools. Not run (pytest forbidden).
- Safe to apply without judgment: no. Adding the suite to the gate is the owner's call and its current health is unknown.

## MT-08: The oracle's README tells readers to extend and edit the code that reference/README.md says never to edit
- Verdict: CONFIRMED
- Checked: reference/README.md:5-8 "Do not edit `magcoupling-py/magcoupling/`: parity and differential tests for the Rust port compare against it unchanged." reference/magcoupling-py/README.md:5 ("written to be dropped into another repository and driven from a GUI"), 58 ("Using it from a GUI"), 75 (Examples), 221-226 (Extending: "add a `MagnetSpec` row in `library.py`", "change `model.HARMONICS`" with the note that it breaks parity on purpose).
- Safe to apply without judgment: no. The README is part of the vendored oracle folder, and what to keep as history is the owner's call.

## MT-09: Two example front ends in the oracle that nothing runs, tests or links to
- Verdict: CONFIRMED
- Checked: Grep for `gui_integration|tk_demo` over the whole worktree: only reference/magcoupling-py/README.md:75 and tk_demo.py:3 (its own docstring). Glob examples/* shows exactly the two files. No pyproject, gate, test or Rust hit.
- Safe to apply without judgment: no. The files sit in the vendored oracle and the owner decides whether to keep them as history.

## Seen while checking
- (none)
