# Audit findings: python
Scope read: src/linkage_sim (symbol-level dead-code scan by AST over every def/class, plus targeted reads of toggle, events, forward_dynamics, force_plots), tests (AST scan for assertion-free tests, swallowed excepts, skip/continue patterns; read test_golden_fixtures, test_numerical_consistency, test_mechanical_advantage), scripts (export_golden, export_gifs, view_crank_rocker, view_fourbar, ternary helpers of the sixbar views), data/benchmarks/golden, output/gifs, pyproject.toml.

Who uses the Python side: Rust reads 7 of the 9 golden files (linkage-sim-rs/tests/golden_fixtures.rs); linkage-sim-rs/src only names Python in comments ("ported from linkage_sim.analysis.envelopes", "matches the Python reference"); gate 12 runs the vendored reference/magcoupling-py oracle, not src/linkage_sim; deploy-web.yml runs no pytest. Nothing in either crate imports linkage_sim or calls scripts/. src/linkage_sim is used by tests/** and scripts/** only.

## PY-01: Three Python force elements are used nowhere and tested nowhere
- Kind: not needed
- Where: src/linkage_sim/forces/gas_spring.py (128 lines), src/linkage_sim/forces/bearing_friction.py (101), src/linkage_sim/forces/linear_actuator.py (105); claims at README.md:12,28,34 and docs/FEATURES.md:28
- Evidence: `git grep -E "bearing_friction|linear_actuator|gas_spring|BearingFriction|LinearActuator|GasSpring" -- src tests scripts` finds only each file's own class line and `_id` default. No import in src, tests or scripts; no test_*.py names them (test_friction.py covers forces/friction.py). The AST scan lists GasSpring, BearingFriction and LinearActuator as names occurring exactly once in src+tests+scripts.
- Why it matters: 334 lines of untested physics code that the README presents as "all 12 force element types supported in both Python and Rust"; if they are wrong nothing would show, and if they are not needed they only add reading load.
- Suggested action: either add one small test per element (the owner prefers tests), or delete them and say in README/FEATURES that the Python reference lacks gas spring, bearing friction and actuator.
- Fix class: owner-decision
- Confidence: certain (search covered src, tests, scripts; nothing outside those imports Python)

## PY-02: Dead public functions and one dead dataclass in src/linkage_sim
- Kind: not needed
- Where: src/linkage_sim/analysis/toggle.py:76 `ToggleSweepPoint`; src/linkage_sim/analysis/transmission.py:79 `transmission_angle_from_poses`; src/linkage_sim/core/state.py:132 `State.set_angle`; src/linkage_sim/solvers/events.py:91 `make_terminal_angle_event`
- Evidence: each name occurs once in the whole worktree (its definition). `ToggleSweepPoint` is not even used in toggle.py, which has no sweep function returning it. tests/test_nonsmooth_dynamics.py imports other names from solvers/events.py but not this one.
- Why it matters: untested API that nothing calls; ToggleSweepPoint is a leftover of a sweep feature that was never written.
- Suggested action: delete the four definitions (and their imports if any), or add tests if the owner wants them kept as API.
- Fix class: needs-judgment (public names of a library package, though no user exists in the repo)
- Confidence: certain for "no user in the repo"; "no external user" cannot be proven from here

## PY-03: A test with no assertion can never fail
- Kind: broken
- Where: tests/test_mechanical_advantage.py:95 `test_ground_input_returns_zero`
- Evidence: the body calls `mechanical_advantage(mech.state, q_dot, "crank", "rocker", output_coord="theta")`, stores `result`, then ends with the comment "Test with rocker->rocker (same body, should give MA=1 trivially)" and no assert. The name says "ground input returns zero" and the docstring says "Ground body has zero velocity", but neither case is exercised. It is the only assertion-free test in tests/ by AST scan.
- Why it matters: counts as a passing test in the 1088 while checking nothing; the ground-input behaviour is untested.
- Suggested action: pass "ground" as the input body and assert the expected result (or the raised error), or delete the test.
- Fix class: needs-judgment (the intended expectation must be chosen from mechanical_advantage's contract)
- Confidence: certain

## PY-04: The four golden mechanism builders are copied between export_golden.py and test_golden_fixtures.py
- Kind: inconsistent
- Where: scripts/export_golden.py:50 `build_fourbar_driven`, :187 `build_fourbar_undriven`, :264 `build_slidercrank_driven`, :393-:503 sixbar builder and `find_sixbar_initial`; tests/test_golden_fixtures.py:54, :90, :109, :153 (builders), plus `solve_*_at` helpers
- Evidence: tests/test_golden_fixtures.py:154 docstring "Watt I 6-bar with ternary coupler (matches export_golden.py)"; linkage-sim-rs/tests/golden_fixtures.rs:59,111 say "matches export_golden.py::build_fourbar_driven". leads.md lists 40-, 35-, 28-, 22-, 15-line identical runs between the two files.
- Why it matters: the test is only meaningful if its builder equals the exporter's; two hand-synced copies can drift apart and the test would then compare different mechanisms (or the file would be regenerated without the test noticing). A third copy lives in the Rust test.
- Suggested action: move the builders to one module importable by both (tests/__init__.py exists, so a tests/_builders.py or scripts importing from it works; export_golden already does sys.path.insert).
- Fix class: needs-judgment
- Confidence: certain for the duplication; likely that the copies are still identical (not diffed line by line)

## PY-05: The ternary-plate constructor and the six-bar initial-guess solver are pasted 7 and 4 times
- Kind: inefficient
- Where: `_make_ternary` in scripts/view_sixbar_A1.py:59, view_sixbar_A2.py:56, view_sixbar_B2.py:56, view_sixbar_B3.py:47; `make_ternary_link` in scripts/view_sixbar.py:57 and scripts/export_golden.py:393; `_make_ternary_link` in tests/test_numerical_consistency.py:118; the same body inline in tests/test_golden_fixtures.py:159-165. `_find_initial` in view_sixbar_A1.py:133, A2:128, B2:123, B3:114 (plus `_initial_guess` view_sixbar.py:149, `find_sixbar_initial` export_golden.py:456, `_find_sixbar_initial` in test_numerical_consistency)
- Evidence: the four `_make_ternary` bodies are character-identical (Body(id=...), three add_attachment_point calls, cg_local = mean of the two local points / 3, mass, Izz_cg). src/linkage_sim/core/bodies.py has make_bar and make_ground only; its docstring line 4 already describes ternary plates. leads.md: 22-line runs view_sixbar_B3/B2/A2 vs A1.
- Why it matters: the owner flags repetition; any fix to the cg formula must be made 8 times.
- Suggested action: add `make_ternary` to core/bodies.py (with a test in tests/test_bodies.py, which already has test_ternary_plate) and have the scripts and tests call it; put the sixbar initial-guess search in one scripts helper.
- Fix class: needs-judgment
- Confidence: certain

## PY-06: The same four-bar and slider-crank builders are rewritten in about 20 test files and 10 scripts
- Kind: inefficient
- Where: tests/test_kinematics.py:14, test_mechanical_advantage.py:14, test_mechanism.py:17, test_inverse_dynamics.py:18, test_fourbar_benchmark.py:32, test_benchmark_damped.py:22, test_benchmark_inertia.py:28, test_benchmark_fourbar_spring.py:24, test_benchmark_fourbar_gravity.py:29, test_failure_modes.py:42, test_interactive_viewer.py:35,66,105, test_numerical_consistency.py:65,88, test_golden_fixtures.py:54,109, test_regression_baselines.py, test_statics.py, test_reactions.py, test_virtual_work.py, test_toggle.py, test_viewer.py, test_animation.py and others; scripts/view_*.py (view_crank_rocker.py:42 and view_fourbar.py:35 differ only in numbers)
- Evidence: `git grep -c 'make_ground(O2=(0.0, 0.0), O4='` matches in 47 files under tests/ and scripts/. There is no tests/conftest.py and no shared builder module. leads.md lists 14-line identical runs between scripts/export_golden.py:57 and about 20 test files.
- Why it matters: the biggest single source of repetition in the Python side (the whole fourbar block, 14 lines, many times). Different parameters per file are real, but the shape is one function.
- Suggested action: one parametrised `build_fourbar(a, b, c, d, masses, driver=True)` and `build_slidercrank(...)` in a shared tests helper (and scripts helper); migrate files gradually. Keep a file's own builder where it truly differs.
- Fix class: needs-judgment
- Confidence: certain for the repetition; the right shared signature needs a decision

## PY-07: Two golden files are used only by the Python tests, and the Python loader turns a missing fixture into a skip
- Kind: inconsistent
- Where: data/benchmarks/golden/fourbar_dynamics.json and sixbar_kinematics.json (no reference in linkage-sim-rs); tests/test_golden_fixtures.py:44-46 `load_golden`
- Evidence: linkage-sim-rs/tests/golden_fixtures.rs loads only fourbar_kinematics, slidercrank_kinematics, fourbar_statics, slidercrank_statics, fourbar_inverse_dynamics, slidercrank_inverse_dynamics, pendulum_dynamics. `load_golden` does `if not path.exists(): pytest.skip(...)`. The tracked fixture fourbar_dynamics.json is documented as badly off in docs/analyses/2026-10-06-open-physics-review-response.md:120 (energy swings 42.3 J of 73.5 J, starting velocity violates the constraints); it is nevertheless still in the tracked set and still tested by tests/test_golden_fixtures.py:487-529.
- Why it matters: a deleted or renamed fixture silently turns about 19 tests into skips (they pass the pytest summary); and the Rust port has no check against two of the nine fixtures, one of which the project's own review calls wrong.
- Suggested action: make `load_golden` fail (pytest.fail or let FileNotFoundError raise); decide whether the two files get Rust tests or are marked Python-only (and fourbar_dynamics regenerated or dropped).
- Fix class: owner-decision
- Confidence: certain for the facts above; whether the owner wants them kept is the open point

## PY-08: Sweep-based finite-difference tests can pass without checking anything
- Kind: broken
- Where: tests/test_numerical_consistency.py:316,321,407,412,447,477,507,533,552,757,762 (`continue` when an index or solution is missing); same pattern at tests/test_regression_baselines.py:501,503, test_sixbar_ternary.py:416, test_textbook_benchmarks.py:351, test_fourbar_benchmark.py:356, test_failure_modes.py:498
- Evidence: `if idx_center is None or idx_minus is None or idx_plus is None: continue` then `if q_center is None or ...: continue`; no counter and no assert that at least one angle was compared. I ran the key lookup read-only for the 6-bar velocity test: all 6 test angles match today (so it checks now), but if the solver stops converging on that range or the float key rounding changes, the test turns into a no-op with a green result. The angle->index dict loop is also pasted three times (lines 283-322, 381-412, 736-762; leads.md).
- Why it matters: tests that cannot fail when the thing under test breaks give false confidence.
- Suggested action: count compared angles and `assert n_checked == len(angles_test)` (or fail on a missing solution); move the lookup into one helper.
- Fix class: needs-judgment
- Confidence: likely (the pattern is certain; the other files were not each read to see whether a count exists)

## PY-09: scripts/export_gifs.py behaves differently from the other scripts and hides failures
- Kind: inconsistent
- Where: scripts/export_gifs.py:37-40 (`_load_mechanism`), :97 (default `output_dir`), :327-331 (build failure), :27-34 imports of `_compute_bounds`, `_draw_bodies`, `_draw_ground`, `_draw_joints`; scripts/export_golden.py:21-22; scripts/view_*.py
- Evidence: `path = os.path.join("scripts", f"{script_name}.py")` and `code = open(path).read().replace("if __name__", "if False")` then `exec(code, ns)`: it only works when the shell's cwd is the repo root, the file handle is never closed, and the viewer scripts are run by string-patching their source. A failure prints `BUILD FAILED: ...` then `continue`, so the script exits 0 with a missing GIF. export_golden.py makes itself cwd-proof (`sys.path.insert(0, .../src)` and `Path(__file__)`) while export_gifs and view_*.py rely on `pip install -e` and the cwd. view_fourbar.py is in no SCRIPT_NAMES entry, so it has no GIF.
- Why it matters: a broken viewer script is only noticed by reading the output; running from another directory gives a FileNotFoundError that looks like a script bug.
- Suggested action: use `Path(__file__).parent`, `importlib` (the view scripts already have `main()` guarded by `__name__`, so a plain import works without the source patch), and exit non-zero when a build fails.
- Fix class: needs-judgment
- Confidence: certain for the code reading; not run (read-only audit)

## PY-10: 19 MB of generated GIFs in output/gifs are used by nothing
- Kind: not needed
- Where: output/gifs/view_*.gif (11 files, 1.2 to 2.4 MB each)
- Evidence: the only references to output/gifs are scripts/export_gifs.py (writer) and a .gitignore comment ("GIFs are committed for reference", .gitignore:48). No README, doc, HTML page or test names them (`git grep output/gifs` hits only those two). They are reproducible from export_gifs.py.
- Why it matters: they bloat every clone and cannot be reviewed; they go stale whenever a view script or the solver changes (view_sixbar.py is one of them).
- Suggested action: untrack and gitignore output/, or link them from a doc if they are meant as reference.
- Fix class: owner-decision
- Confidence: likely (nothing in the worktree uses them; the owner may keep them for some outside use)

## Leads settled
- reference/magcoupling-py/audit/tests/conftest.py pytest hooks (lines 23-52): not in my part; not real in any case (pytest calls hooks by name).
- src/linkage_sim/analysis/toggle.py `ToggleSweepPoint`: real (PY-02).
- analysis/transmission.py `transmission_angle_from_poses`: real (PY-02).
- core/state.py `set_angle`: real (PY-02).
- forces/gas_spring.py `GasSpring`: real (PY-01), and so are BearingFriction and LinearActuator, which the lead list missed.
- solvers/events.py `make_terminal_angle_event`: real (PY-02).
- tests/* `setup_method` entries: not real; pytest runs setup_method by name.
- tests/test_numerical_consistency.py `_solve_sixbar` (line 217): real; the name occurs once (its definition), the test methods use `_find_sixbar_initial` and position_sweep instead. Mechanical-safe to delete.
- tests/test_toggle.py `build_non_grashof` (line 37): real; occurs once, never called. Mechanical-safe to delete.
- Duplicate runs between scripts/export_golden.py, tests/test_golden_fixtures.py and the view_*.py: real (PY-04, PY-05, PY-06).
- Duplicate runs inside tests/test_numerical_consistency.py (:381-412 vs :287, :736-762 vs :294): real (PY-08).
- Duplicate runs tests/test_prismatic.py:450-466 vs :379, test_coupler_trace.py:167-182 vs :98, test_failure_modes.py:579-593 vs :290: not examined; small and in-file, not worth a finding.
- Rust-side runs (singular_behavior.rs, validation.rs): outside my part.
- output/gifs/*.gif "named N times": real as unused (PY-10); the counts come from the stems appearing in scripts and source, not from any document.
- Python-side "unused" check of mypy/pytest in the gate: already in the baseline; not repeated.

## Not read
- src/linkage_sim: only the symbols above were read. Not read in full: analysis/crank_selection.py, motor_sizing.py, force_breakdown.py, envelopes.py, validation.py; core/constraints.py, mechanism.py; solvers/assembly.py, kinematics.py, statics.py; io/serialization.py; viz/interactive_viewer.py, viewer.py. The dead-symbol scan did cover them, but not for bugs.
- tests: about 45 of the 57 files were only scanned by AST (assertion-free, swallowed excepts, continue/skip patterns), not read.
- scripts: view_chebyshev, view_double_crank, view_double_rocker, view_parallelogram, view_slider_crank and the bodies of view_sixbar*.py were not read beyond their helpers.
- data/benchmarks/golden/*.json: not opened (only file names checked).
