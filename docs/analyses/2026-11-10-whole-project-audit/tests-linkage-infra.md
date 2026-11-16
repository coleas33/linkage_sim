# Audit findings: tests-linkage-infra
Scope read: linkage-sim-rs/tests/*.rs (all 15 files, 7 read in full, golden_fixtures/singular_behavior/force_zone/geometry/compound_force_integration skimmed or read at their helpers and weak spots), linkage-sim-rs/scripts/* (all 6), .github/workflows/deploy-web.yml, .claude/workflows/*.js (committed worktree copies and the main-checkout copies of audit-campaign.js and fix-campaign.js), .claude/agents, .claude/hooks, .claude/skills/fbd-derive (main checkout), linkage-sim-rs/web/* (all 7 files), .gitignore, .gitattributes, pyproject.toml.

## INF-01: Gate 7 re-runs every test gate 4 already ran
- Kind: inefficient
- Where: linkage-sim-rs/scripts/gate.sh:88-89 (gate 4), :97-98 (gate 7); same overlap for clippy at :91-92 (gate 5) and :100-101 (gate 8)
- Evidence: gate 4 is `cargo test --manifest-path "$MAGCOUPLING"` (default features = `[]`, magcoupling-rs/Cargo.toml `default = []`); gate 7 is `cargo test --manifest-path "$MAGCOUPLING" --features app`. Features only add code: `app` implies `gui`, and the only feature-gated tests are two `#[cfg(feature = "gui")]` blocks (magcoupling-rs/tests/explain.rs:1505, :1773). So gate 7 compiles and runs all of tests/ (explain.rs 262 kB, deviations.rs 112 kB, shaft_joint.rs 84 kB, sizing.rs 64 kB ...) a second time.
- Why it matters: the largest test suite in the repo runs twice on every gate; the fix loop runs the gate once per item and once more per rework.
- Suggested action: keep the engine-alone proof as `cargo check` / `cargo test --lib --no-run` (gate 6 already checks the wasm32 lib with no features), or run gate 4 with a filter and let gate 7 carry the full suite. Same question for clippy gate 5 vs 8 (gate 5 buys the no-feature dead-code view only).
- Fix class: needs-judgment (the owner may want the engine-alone test run kept as a purity proof)
- Confidence: likely; would be settled by comparing `cargo test` test counts of gate 4 and gate 7 output (gate 7 should be gate 4 plus the two gui blocks and the gui/app unit tests).

## INF-02: Gate 2 (`cargo clippy --all-targets`, linkage-sim-rs) can only fail on a compile error
- Kind: broken (a step that practically cannot fail)
- Where: linkage-sim-rs/scripts/gate.sh:82-83
- Evidence: `cargo clippy --all-targets` with no `-D warnings`; the baseline reports 259 lib warnings plus 3 in tests/property_tests.rs and it exits 0. Every magcoupling-rs gate uses `-- -D warnings` (:92, :101, :104, :105). The header (:6-10) states the magcoupling-rs gates run "with warnings as errors", so the linkage gate is a deliberate exception, but nothing says what gate 2 is for.
- Why it matters: gate 1 (`cargo test --all`) already compiled every target; gate 2 then adds only clippy's deny-by-default lints, so a new warning of any kind (including a new `collapsible_if`, or an unused variable) passes the gate.
- Suggested action: either state in the gate's echo/comment that it only catches deny-level lints until BL-003 is done, or add a count-based ratchet (fail if the warning count rises). Do not switch to `-D warnings` before BL-003 lands.
- Fix class: owner-decision
- Confidence: certain for the behaviour; BL-003 covers the clippy bulk but not the gate's weakness.

## INF-03: Gate 11's empty-pin FAIL branch is unreachable
- Kind: broken
- Where: linkage-sim-rs/scripts/gate.sh:124, :127-130
- Evidence: line 124 is `CLI_PIN="$(grep -oE 'wasm-bindgen-cli@[0-9.]+' "$REPO_ROOT/.github/workflows/deploy-web.yml" | head -n 1 | cut -d@ -f2)"` under `set -euo pipefail`. If deploy-web.yml loses the pin, `grep` exits 1, pipefail makes the assignment fail and `set -e` aborts the script on line 124. Line 127 `[[ -z "$CLI_PIN" || ... ]]` and its message "deploy-web.yml installs wasm-bindgen-cli '', ..." can therefore never print for the empty case.
- Why it matters: the gate still exits non-zero, but with no message at all, in a gate whose other failures all say what is wrong.
- Suggested action: add `|| true` to the CLI_PIN assignment so line 127 reports it.
- Fix class: mechanical-safe
- Confidence: likely (follows from bash semantics; would be settled by running gate 11 against a copy of deploy-web.yml without the pin).

## INF-04: deploy-web.yml compiles wasm-bindgen-cli from source on every run, so its cache entry buys nothing
- Kind: inefficient
- Where: .github/workflows/deploy-web.yml:26-27 (caches `~/.cargo/bin/`), :36-37 (`cargo install wasm-bindgen-cli@0.2.114 --force`)
- Evidence: the cache step restores `~/.cargo/bin/` (which would hold the installed `wasm-bindgen`), then `cargo install ... --force` rebuilds and replaces it unconditionally.
- Why it matters: every push to main spends several minutes recompiling a tool whose version is pinned and cached.
- Suggested action: drop `--force` (cargo install is a no-op when that version is already installed) or guard with `wasm-bindgen --version | grep -q 0.2.114`; consider `--locked`.
- Fix class: needs-judgment (CI change, cannot be tested locally)
- Confidence: likely; settled by a CI run's timing after the change.

## INF-05: Docs say the deploy uses "stable" Rust, the workflow pins 1.89.0, and nothing else pins a toolchain
- Kind: inconsistent
- Where: docs/guides/WASM_DEPLOYMENT.md:83 ("Install the stable Rust toolchain"); .github/workflows/deploy-web.yml:18-20 (`dtolnay/rust-toolchain@stable` with `toolchain: "1.89.0"`); no `rust-toolchain.toml`, no `rust-version` in either Cargo.toml (git ls-files / git grep found none)
- Evidence: the action ref says `@stable`, its `toolchain` input overrides it to 1.89.0. Only the deploy workflow and a 2026-08-08 plan (docs/superpowers/plans/2026-08-08-agentic-test-improve-loop.md:9, "Rust 1.89.0") name the version; local builds and gate.sh use whatever rustup default is installed.
- Why it matters: the release wasm that ships is built with a compiler the gate never ran on; a local toolchain bump (edition 2024 code) can pass the gate and fail in CI, or the reverse. The doc also misstates the pin.
- Suggested action: add `rust-toolchain.toml` (or `rust-version`) matching 1.89.0 and have the workflow read it; correct the doc line. Also note the deploy job runs neither gate.sh nor any test before `vercel deploy --prod` (no document promises one).
- Fix class: owner-decision
- Confidence: certain for the doc/workflow disagreement; the "no gate before deploy" point is an observation only.

## INF-06: audit-campaign.js and fix-campaign.js hard-code the main checkout's absolute path as ROOT
- Kind: inconsistent
- Where: .claude/workflows/audit-campaign.js:36 and .claude/workflows/fix-campaign.js:34 (the same line in both the committed worktree copies and the newer uncommitted main-checkout copies; the model-pin edits in the main copies do not touch it)
- Evidence: `const ROOT = 'C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs'`. fix-campaign.js:42 then tells the agent "work in the main working tree", and :47 runs `bash scripts/gate.sh` from ROOT. gate.sh meanwhile locates itself with `$(dirname "$0")` and finds the oracle's `.venv` in sibling worktrees (:55-77), so the repo is designed to be used from several worktrees.
- Why it matters: the workflows cannot run in a worktree (this audit's own `lsim-audit`) or another clone: every agent is pointed at the main checkout, so a fix campaign started in a worktree edits, tests and stashes (`git stash push -u`, :56, :91) the wrong tree.
- Suggested action: take ROOT from `args.root` with the current default, or derive it from the workflow file's location; keep `git stash` helpers scoped to that root.
- Fix class: needs-judgment
- Confidence: certain for the hard-coded path; whether the owner wants workflows to run elsewhere is the judgment.

## INF-07: gui-smoke.js re-derives the calculator's `passed` from its checks but takes the linkage step's `passed` on the agent's word
- Kind: inconsistent
- Where: .claude/workflows/gui-smoke.js:126-133 (linkage step, no re-check), :182-184 (calculator re-check); linkage-sim-rs/scripts/gui_smoke_verdict_test.cjs:259-266 (only `passed: false` is tested for the linkage step)
- Evidence: `magcoupling.passed = magcoupling.passed === true && CALCULATOR_CHECKS.every(...) && console_errors.length === 0` (the comment at :179-181 says why an agent's own `passed` is not trusted). For the linkage step the result is `... || noReading(SMOKE_SCHEMA)` and line 196 uses `linkage.passed` directly even though SMOKE_SCHEMA requires `canvas_present`, `hub_page`, `title_warning`, `console_errors`. The verdict test has no case "hub_page false with passed true".
- Why it matters: an agent reading `hub_page: false`, `canvas_present: false`, `title_warning: true` or a non-empty console_errors with `passed: true` passes the smoke run for the linkage app; BL-037 (title warning) is exactly a check that must not be skippable.
- Suggested action: apply the same re-derivation to the linkage step (canvas_present && hub_page && !title_warning && console_errors empty) and add the matching cases to gui_smoke_verdict_test.cjs.
- Fix class: needs-judgment
- Confidence: certain (read both code paths); the committed and main copies of gui-smoke.js are identical (25416 bytes each).

## INF-08: rust-test-fixer agent hard-codes `model: opus`, against the repo's model rule
- Kind: inconsistent
- Where: C:/Users/Cole/source/repos/linkage_simulation/.claude/agents/rust-test-fixer.md:7 (main checkout only; .claude/agents is git-ignored)
- Evidence: frontmatter `model: opus`. CLAUDE.md section 5 says every agent definition names `sonnet`, `haiku` or `fable` and "Never hard-code an Opus alias or version"; the role (repair test-crate compile errors, mechanical, tests/ only) is in the `sonnet` list. fbd-math-reviewer.md:5 correctly says `model: fable`.
- Why it matters: the one agent definition left on the old rule bypasses the controller/decider split and its cost control.
- Suggested action: set `model: sonnet` (mechanical repair; escalate to fable on failure per section 5).
- Fix class: owner-decision
- Confidence: certain.

## INF-09: Post-edit cargo-check hook matches tests/*.rs but only checks the lib
- Kind: broken
- Where: C:/Users/Cole/source/repos/linkage_simulation/.claude/hooks/cargo-check-on-rs-edit.ps1:1-3 and :38-40 (regex `/linkage-sim-rs/(src|tests)/.*\.rs$`), :58 (`cargo check --features native --lib`)
- Evidence: the header says "Edit/Write to any *.rs file under linkage-sim-rs/src/" while the regex also accepts `tests/`; the command checks `--lib` only, which never compiles tests/*.rs.
- Why it matters: after an edit to an integration test the hook runs a check that cannot see the edit, stays silent, and so reports a clean result for a test file that may not compile.
- Suggested action: when the path is under tests/, run `cargo check --features native --tests` (the command rust-test-fixer.md already uses); fix the header comment.
- Fix class: mechanical-safe
- Confidence: certain.

## INF-10: fbd-derive skill and fbd-math-reviewer agent cite `evaluation.rs:424-425` for the actuator sign convention; it now sits at 437-438
- Kind: inconsistent
- Where: C:/Users/Cole/source/repos/linkage_simulation/.claude/skills/fbd-derive/SKILL.md (the "Actuator" line and the reference list: `forces/elements/evaluation.rs:424–425`); C:/Users/Cole/source/repos/linkage_simulation/.claude/agents/fbd-math-reviewer.md ("Actuator sign convention" item 5)
- Evidence: linkage-sim-rs/src/forces/elements/evaluation.rs:424-425 is `let v_rel = unit.dot(...)` and a blank line; `let force_on_b = unit * net_force_along_unit; let force_on_a = -force_on_b;` is at :437-438 in `evaluate_linear_actuator` (:389). The convention itself still matches (force_on_b = +F u, force_on_a = -F u).
- Why it matters: both documents are the "ground truth" an FBD reviewer follows; a wrong line points the reviewer at the end-stop damping code.
- Suggested action: cite `evaluate_linear_actuator` by name (it survives edits), not a line range.
- Fix class: mechanical-safe
- Confidence: certain.

## INF-11: tests/braindump_repro.rs is tracked and still the repro BL-017 depends on, yet its header says "TEMPORARY ... Not for commit"
- Kind: inconsistent
- Where: linkage-sim-rs/tests/braindump_repro.rs:1-4, :20; docs/ai/backlog.yaml BL-017 (open, evidence and acceptance name `braindump_repro bl010_side_sizing_mode_sweep_panics_debug_assert`), BL-010 notes ("Repro in tests/braindump_repro.rs (untracked)")
- Evidence: `//! TEMPORARY audit repro tests for BL-017. Not for commit.` and `//! (BL-011 repro promoted ... BL-010 repros promoted ...)`; `git ls-files` lists the file; the test is named `bl010_side_sizing_mode_sweep_panics_debug_assert` but its own comment says the root cause was BL-022, and it no longer panics (the name states the opposite of what it checks now).
- Why it matters: a reader deleting "temporary" files would remove the only test for BL-017's remaining acceptance; the file, the test name and the backlog note ("untracked") each say something different.
- Suggested action: rename to a regression test (for example tests/sizing_sweep_finite.rs, test `chebyshev_sizing_sweep_has_finite_forces`), rewrite the header, and update BL-010/BL-017 text in the same change.
- Fix class: needs-judgment
- Confidence: certain.

## INF-12: The same standard 4-bar is hand-built five times across two test files, with its initial-guess helper written twice
- Kind: inefficient (repetition)
- Where: linkage-sim-rs/tests/golden_fixtures.rs:62-86 (`build_fourbar_driven`), :306-328 (`build_fourbar_gravity`), :813-839 (`build_fourbar_expression_driver`), :967-983 (inlined again inside `spring_affects_statics_driver_torque`), :94-108 (`fourbar_initial_guess`); linkage-sim-rs/tests/singular_behavior.rs:26-53 (`build_standard_fourbar`), :55-79 (`..._with_gravity`), :87-101 (`fourbar_initial_guess`)
- Evidence: all build ground O2 (0,0)/O4 (4,0), crank 1.0 (mass 2.0, Izz 0.01), coupler 3.0 (3.0, 0.05), rocker "D","C" 2.0 (2.0, 0.02), four revolute joints J1-J4 and the identity driver `|t| t, |_t| 1.0, |_t| 0.0`; singular_behavior's `fourbar_initial_guess` and golden_fixtures' (:94-108) are the same 14 lines (the machine candidate "singular_behavior.rs:79-101 vs golden_fixtures.rs:86"). Differences are one `add_coupler_point` and one `add_force(Gravity)`.
- Why it matters: a change to the reference mechanism (e.g. the Python generator's) must be made in five places, and the golden tests compare against data from `scripts/export_golden.py`, which is its own copy again (baseline: 133 duplicated lines between export_golden.py and test_golden_fixtures.py, Python side not in this part).
- Suggested action: add `linkage-sim-rs/tests/common/mod.rs` (magcoupling-rs/tests/common already does this) with one `standard_fourbar(opts)` builder and one `fourbar_initial_guess`; keep per-file differences as parameters.
- Fix class: needs-judgment (tests only, no behaviour change, but touches golden-fixture tests)
- Confidence: certain.

## INF-13: `first_actuator` is copied between two test files, and two helpers in one file have identical bodies
- Kind: inefficient (repetition)
- Where: linkage-sim-rs/tests/actuator_force_label.rs:15-27 and linkage-sim-rs/tests/compound_actuator_rebuild.rs:29-41 (`first_actuator`, identical except the `expect` message); linkage-sim-rs/tests/compound_actuator_rebuild.rs:61-67 (`pin_to_pin_length`) and :71-77 (`element_length`)
- Evidence: both `first_actuator` bodies are `state.mechanism.as_ref().expect("mechanism built").forces().iter().find_map(|f| match f { ForceElement::LinearActuator(la) => Some(la.clone()), _ => None })`; `pin_to_pin_length(state, pins: &LinearActuatorElement)` and `element_length(state, la: &LinearActuatorElement)` are the same eight lines (two `body_point_global` calls and `(b - a).norm()`) under two names.
- Why it matters: repetition the owner flags aggressively; the two length helpers can silently diverge, and then the "pin-to-pin equals element length" check at :137-142 would compare two different things.
- Suggested action: one `actuator_length(state, la)` helper and one shared `first_actuator` in tests/common/mod.rs (see INF-12).
- Fix class: mechanical-safe (tests only)
- Confidence: certain.

## INF-14: Assertions sit inside `if let Ok` / `if converged` / `return Ok(())` guards, so these tests pass when the solver fails every case
- Kind: broken (tests that can never fail for a whole class of regression)
- Where: linkage-sim-rs/tests/property_tests.rs:134-137 (`None => return Ok(())`), :139-155 (`if let Ok(result)` then `if result.converged`), :167-178 (`return Ok(())` on `!converged`, `if let Ok(q_dot)`); linkage-sim-rs/tests/golden_fixtures.rs:1012-1014 (`continue` when not converged) and :1038-1045 (`if let Ok(result) = vw` around the virtual-work `assert!`)
- Evidence: in `constraint_residual_near_zero_after_solve` and `velocity_satisfies_constraint` the only assertions are inside those guards, and nothing counts how many generated cases reached an assertion. In `spring_affects_statics_driver_torque` the virtual-work agreement is checked only when `virtual_work_check` returns `Ok`; an `Err` skips it silently.
- Why it matters: if `solve_position` started returning `Err` or `converged == false` for all 4-bar inputs (or `virtual_work_check` always errored), both property tests and the virtual-work cross-check would still go green.
- Suggested action: count cases that reached an assertion and require a minimum (as `compared > 300` does in wheel_contact_sweep.rs:73); turn `if let Ok` into `.expect(...)` where an error is not an accepted outcome.
- Fix class: needs-judgment
- Confidence: certain for the structure; whether any current case reaches the guard-protected assertions needs one run with a counter.

## INF-15: `expression_driver_matches_constant_speed` (property test) tests the `meval` crate, not this repo; `meval` is also listed twice in Cargo.toml
- Kind: broken (a test that checks nothing of the repo) and not needed (duplicate dependency line)
- Where: linkage-sim-rs/tests/property_tests.rs:324-362; linkage-sim-rs/Cargo.toml:26 (`meval = "0.2"` in [dependencies]) and :62 (again in [dev-dependencies])
- Evidence: the test parses `format!("{}*t+{}", omega, theta_0)` with `expr.parse::<meval::Expr>().unwrap().bind("t")` and compares to `omega * t + theta_0`; no repo type is called. The golden_fixtures.rs test of the same name (`expression_driver_matches_constant_speed`, :841) is the one that uses `add_expression_driver`.
- Why it matters: it can only fail if a third-party crate breaks, and the same name in two files hides which one covers the repo's expression driver. The [dev-dependencies] entry is redundant with [dependencies].
- Suggested action: delete the property test, or rewrite it to build `add_expression_driver` and compare with `add_constant_speed_driver` over random omega/theta_0; remove the dev-dependency line.
- Fix class: needs-judgment
- Confidence: certain.

## INF-16: Direct dependencies `wasm-bindgen-futures` and `uuid` of linkage-sim-rs are named nowhere in its code
- Kind: not needed
- Where: linkage-sim-rs/Cargo.toml:38 (`uuid = { version = "1", features = ["js"] }`), :42 (`wasm-bindgen-futures = "0.4"`)
- Evidence: `Grep uuid|wasm_bindgen_futures|wasm-bindgen-futures` over linkage-sim-rs (excluding Cargo.lock) matches only those two Cargo.toml lines. Cargo.lock shows `wasm-bindgen-futures` also reached through eframe and rfd, and `uuid` through `dxf` 0.6.1, which depends on uuid with the `serde` feature only.
- Why it matters: `wasm-bindgen-futures` is a leftover direct dependency (the backlog.yaml:593 mention of it is about magcoupling-rs, a different crate). `uuid` with `js` may be there on purpose: on wasm32 dxf's uuid needs the `js` feature of getrandom, which only this direct line turns on.
- Suggested action: remove `wasm-bindgen-futures`; for `uuid` either keep it with a comment ("turns on getrandom/js for dxf on wasm32") or test removal with gate 3's wasm check.
- Fix class: needs-judgment
- Confidence: likely for wasm-bindgen-futures; unsure for uuid (settled by `cargo check` of LINKAGE_WEB_ARGS after removing the line).

## INF-17: .gitignore carries a stale heading and two commented-out rules
- Kind: not needed
- Where: .gitignore, the block "# Golden test fixtures ... committed after Phase 4 completion for Rust port / # data/benchmarks/golden/  # now tracked / *.zip / settings.json / settings.local.json / .cursorignore" and the block "# Output ... GIFs are committed for reference / # output/"
- Evidence: the heading describes golden fixtures, but the live rules under it are `*.zip`, `settings.json`, `settings.local.json`, `.cursorignore`; the only rule it describes is commented out ("now tracked"). `# output/` is likewise a disabled rule. `settings.json` is unanchored, so it ignores a file of that name at any depth (`.claude/*` already covers .claude/settings.json).
- Why it matters: leftovers that read as live policy; the unanchored `settings.json` can hide a future tracked config in web/ or a crate.
- Suggested action: delete the two commented-out rules and the stale heading, group the four live rules under an honest heading, and anchor `settings.json` if only `.claude/` was meant.
- Fix class: mechanical-safe (comment lines); anchoring the pattern is owner-decision
- Confidence: certain for the leftovers.

## Leads settled
- `uuid` (leads 1): not settled to real: no use in code, but likely feature-forcing for dxf on wasm32; see INF-16 (unsure).
- `wasm-bindgen-futures` (leads 1): real, see INF-16 (likely).
- `magcoupling-rs` [dev-dependencies] self-entry (leads 1): not real; it is the documented self dev-dependency that turns on the test-only `workbook-parity` feature (magcoupling-rs/Cargo.toml comment at the `workbook-parity` feature, gate 10).
- Duplicated runs singular_behavior.rs vs validation.rs:307 and golden_fixtures.rs:86 (leads 4, my part only): real, INF-12; singular_behavior.rs:298-317 (parallelogram builder) and :679-697 (short-crank slider) are the same builder shape, but with different geometry each, so left out as a bare pattern.
- tests/singular_behavior.rs vs src/io/serialization.rs:50 (16 lines) and validation.rs:307: not real as repetition worth a finding (revolute-joint boilerplate that any mechanism builder needs; INF-12's shared builder removes most of it).
- `gui-smoke.js` through `docs/chebyshev_lambda/*.png` and `output/gifs/*.gif` (leads 5): outside my part except that fix-campaign.js:102 names `docs/chebyshev_lambda/*.png` as never to be staged; BL-012 covers their being rewritten by cargo test.
- Python `setup_method` / `_solve_sixbar` / `build_non_grashof` candidates (leads 3): outside my part (tests/*.py), not read.

## Not read
- tests/*.py, scripts/*.py (Python tests and scripts: not in my part list beyond pyproject.toml).
- Bodies of linkage-sim-rs/tests/geometry_tests.rs, force_zone_tests.rs, compound_force_integration.rs, singular_behavior.rs (tests 108-286, 330-670, 726-end) and golden_fixtures.rs (160-300, 330-808, 840-950): only helpers, signatures and `if let` / weak-assertion greps were checked, so a weak assertion inside them could be missing from this report.
- .claude/skills other than fbd-derive (none exist); .claude/settings.local.json (not in scope, secrets possible).
