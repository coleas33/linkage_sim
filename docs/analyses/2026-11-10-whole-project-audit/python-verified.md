# Verified: python
## PY-01: Three Python force elements are used nowhere and tested nowhere
- Verdict: CONFIRMED
- Checked: git grep for the module and class names over the whole worktree. Under src, tests and scripts the only hits are each file's own class line and `_id` default (bearing_friction.py:25,48; gas_spring.py:29,54; linear_actuator.py:19,42). No io/serialization path names them. README.md:12 and :34 and docs/FEATURES.md:28 do claim support, and docs/history/ROADMAP_IMPLEMENTATION.md:103-104 says "tested via integration benchmarks" (no test names them, so that doc line is not backed). Rust code names its own variants, not these files.
- Safe to apply without judgment: no, owner decision (delete or test).

## PY-02: Dead public functions and one dead dataclass in src/linkage_sim
- Verdict: CONFIRMED
- Checked: git grep over the worktree: ToggleSweepPoint 1 hit (toggle.py:76), transmission_angle_from_poses 1 (transmission.py:79), make_terminal_angle_event 1 (events.py:91), set_angle 2 hits (core/state.py:132 and the unrelated Rust `set_angle` in linkage-sim-rs/src/core/state.rs:132). No `__all__`, import, test, script or doc names the Python ones. External users cannot be ruled out from here.
- Safe to apply without judgment: no, public names of a library package.

## PY-03: A test with no assertion can never fail
- Verdict: PARTLY (the missing assertion holds; "can never fail" is slightly too strong)
- Checked: tests/test_mechanical_advantage.py:95-104: calls mechanical_advantage(..., "crank", "rocker", output_coord="theta"), then only comments; name and docstring talk about ground input, which is not exercised. A read-only AST scan of every test_* function under tests/ finds this as the only function with no assert / raises / assert_* call. It would still fail if the call raised.
- Safe to apply without judgment: no, the intended expectation must be chosen.

## PY-04: The four golden mechanism builders are copied between export_golden.py and test_golden_fixtures.py
- Verdict: CONFIRMED
- Checked: line numbers match (export_golden.py:50, 187, 264, 393-503; test_golden_fixtures.py:54, 90, 109, 153). A read-only AST comparison (docstrings removed) shows build_fourbar_driven, build_fourbar_undriven, build_slidercrank_driven, find_sixbar_initial, solve_sixbar_at, solve_slidercrank_at identical; build_sixbar_driven and solve_fourbar_at differ in code (the test builds the ternary inline, the exporter calls make_ternary_link), so "still identical" holds for six of eight pairs. Rust golden_fixtures.rs comments name export_golden.py as claimed (not re-opened).
- Safe to apply without judgment: no.

## PY-05: The ternary-plate constructor and the six-bar initial-guess solver are pasted 7 and 4 times
- Verdict: PARTLY
- Checked: `_make_ternary` in view_sixbar_A1/A2/B2/B3 has identical AST (same hash) in all four; further copies exist: view_sixbar.py:57 and export_golden.py:393 (`make_ternary_link`), tests/test_numerical_consistency.py:118, tests/test_regression_baselines.py:96, tests/test_sixbar_ternary.py:39, inline at test_golden_fixtures.py:159-165, so the constructor count is at least as stated. The `_find_initial` functions in the four view scripts are not identical (four different hashes); only the shape repeats, so "pasted 4 times" is overstated. bodies.py has no make_ternary (line 4 docstring mentions ternary plates).
- Safe to apply without judgment: no.

## PY-06: The same four-bar and slider-crank builders are rewritten in about 20 test files and 10 scripts
- Verdict: CONFIRMED
- Checked: git grep -l 'make_ground(O2=(0.0, 0.0), O4=' gives 47 files (35 under tests, 12 under scripts), more than the finder's counts but the same conclusion; tests/conftest.py does not exist; 19 test files define a build_fourbar / _build_fourbar / make_fourbar function.
- Safe to apply without judgment: no.

## PY-07: Two golden files are used only by the Python tests, and the Python loader turns a missing fixture into a skip
- Verdict: CONFIRMED
- Checked: git grep fourbar_dynamics / sixbar_kinematics: no hit in linkage-sim-rs; hits in scripts/export_golden.py, tests/test_golden_fixtures.py (487, 529, 567, 579, 591), docs. load_golden at test_golden_fixtures.py:43-48 does `pytest.skip` when the path is missing; 20 load_golden( call sites. docs/analyses/2026-10-06-open-physics-review-response.md:120 says fourbar_dynamics.json is badly off. git ls-files shows all 9 golden files tracked.
- Safe to apply without judgment: no, owner decision.

## PY-08: Sweep-based finite-difference tests can pass without checking anything
- Verdict: PARTLY
- Checked: `continue` lines match exactly in test_numerical_consistency.py (316, 321, 407, 412, 447, 477, 507, 533, 552, 757, 762) with no counter and no assert on how many angles were compared; same shape (skip an unsolved sample, no count) in test_sixbar_ternary.py:416, test_textbook_benchmarks.py:351 and test_fourbar_benchmark.py:356. Not the same pattern: test_regression_baselines.py:501 and 503 are deliberate exclusions of near-toggle angles, and test_failure_modes.py:498 follows `assert sweep.n_converged == len(angles)` (line 489), so that one cannot go silent.
- Safe to apply without judgment: no.

## PY-09: scripts/export_gifs.py behaves differently from the other scripts and hides failures
- Verdict: CONFIRMED (line numbers off)
- Checked: export_gifs.py:53-55 (not 37-40) `os.path.join("scripts", ...)`, `open(path).read().replace("if __name__", "if False")`, `exec(code, ns)`; 97 default output_dir "output/gifs"; 327-329 `BUILD FAILED` then `continue` (script ends normally, exit 0); 30-36 imports of `_compute_bounds`, `_draw_bodies`, `_draw_ground`, `_draw_joints`. export_golden.py inserts `Path(__file__).parent.parent / "src"` into sys.path; view_fourbar.py:74,92 and view_crank_rocker.py:99,116 have `main()` guarded by `__name__`. SCRIPT_NAMES (73-84) has 11 entries and no view_fourbar.
- Safe to apply without judgment: no.

## PY-10: 19 MB of generated GIFs in output/gifs are used by nothing
- Verdict: CONFIRMED
- Checked: git ls-files output gives 11 files, 19,710,549 bytes. git grep for output/gifs hits only scripts/export_gifs.py:14, 97, 340; git grep for ".gif" outside the Rust tree and scripts finds nothing; .gitignore:48 is the comment "GIFs are committed for reference" and line 49 `# output/` is itself commented out.
- Safe to apply without judgment: no, owner decision.

## Seen while checking
- None.
