# Verified: tests-linkage-infra

## INF-01: Gate 7 re-runs every test gate 4 already ran
- Verdict: CONFIRMED (the overlap); whether to drop it is the owner's call
- Checked: gate.sh:88-89 (`cargo test --manifest-path "$MAGCOUPLING"`), :97-98 (`... --features app`), :91-92 and :100-101 (clippy, same pair). magcoupling-rs/Cargo.toml:13 `default = []`, :21 `gui`, :34 `app` implies `gui`. `Grep` for `cfg(feature = "gui"|"app")` or `cfg(not(feature ...))` in magcoupling-rs: only `tests/explain.rs:1505`, `:1773` and `src/lib.rs:33,36`; no `cfg(not(feature))` anywhere, so features only add code and gate 7 runs a superset of gate 4. Bins have `required-features = ["app"]` (Cargo.toml:129,134). Not done: comparing test counts of the two runs. Gate 5 is not wholly redundant with gate 8: it is the only no-feature dead-code view, as the finder says.
- Safe to apply without judgment: no, the owner may want the engine-alone run kept as a purity proof.

## INF-02: Gate 2 (`cargo clippy --all-targets`, linkage-sim-rs) can only fail on a compile error
- Verdict: CONFIRMED (by reading; the title slightly overstates, since deny-by-default clippy lints also fail it, as the finding's body says)
- Checked: gate.sh:82-83 `cargo clippy --all-targets` with no `-D warnings`; :92, :101, :104, :105 have `-- -D warnings`; header :6-10 says the magcoupling-rs gates run with warnings as errors. No `#![deny]`/`#![forbid]`/`[lints]` in linkage-sim-rs, and no clippy.toml, `.cargo/config` or rust-toolchain file in the worktree, so nothing else turns warnings into errors. backlog.yaml:239 BL (clippy warning debt, about 289 lib warnings) is the linked debt item. The 259 and 3 warning counts come from the baseline and were not re-run.
- Safe to apply without judgment: no, owner decision (comment, ratchet or leave).

## INF-03: Gate 11's empty-pin FAIL branch is unreachable
- Verdict: CONFIRMED
- Checked: gate.sh:26 `set -euo pipefail`; :124 `CLI_PIN="$(grep -oE 'wasm-bindgen-cli@[0-9.]+' ".../deploy-web.yml" | head -n 1 | cut -d@ -f2)"`; :127 `[[ -z "$CLI_PIN" || ... ]]` and :128 message. deploy-web.yml:37 holds the pin today, so the path is dormant. With the pin gone, grep exits 1, pipefail makes the pipeline fail, and an assignment from a command substitution takes that status, so `set -e` exits at :124 and :127 is never reached. `lock_versions` (awk, :51-53) exits 0, so :126 does not share the problem.
- Safe to apply without judgment: yes (identical when the pin exists; when it is missing the gate still exits 1, now with its message).
- Exact change: linkage-sim-rs/scripts/gate.sh line 124, old `| head -n 1 | cut -d@ -f2)"` new `| head -n 1 | cut -d@ -f2)" || true`

## INF-04: deploy-web.yml compiles wasm-bindgen-cli from source on every run, so its cache entry buys nothing
- Verdict: PARTLY. The rebuild on every run holds; the suggested fix "drop `--force`" is not safe.
- Checked: deploy-web.yml:23-34 caches `~/.cargo/bin/` (plus registry and the two `target/` dirs), :36-37 `cargo install wasm-bindgen-cli@0.2.114 --force`. `--force` makes cargo rebuild and overwrite regardless of what was restored, so the cached binary is replaced. But the cache does not include `~/.cargo/.crates.toml` and `.crates2.json`, which is where cargo records installed packages; a restored `~/.cargo/bin/wasm-bindgen` with no such record is, as far as I know, rejected by `cargo install` without `--force` ("binary already exists in destination"), so dropping `--force` is likely to fail the job. The guard alternative (`wasm-bindgen --version | grep -q 0.2.114`) would work. Not run: CI timing.
- Safe to apply without judgment: no, a CI change that cannot be tested locally.

## INF-05: Docs say the deploy uses "stable" Rust, the workflow pins 1.89.0, and nothing else pins a toolchain
- Verdict: CONFIRMED
- Checked: WASM_DEPLOYMENT.md:83 "Install the stable Rust toolchain"; deploy-web.yml:18-21 `dtolnay/rust-toolchain@stable` with `toolchain: "1.89.0"`. Glob for `rust-toolchain*`, `.cargo/config*`, `clippy.toml`: none; Grep `rust-version` in the three Cargo.toml files: none (both crates use `edition = "2024"`). "1.89" appears only in deploy-web.yml and two plans (2026-08-08-agentic-test-improve-loop.md:9, 2026-09-29-payload-weights-track2.md:9). deploy-web.yml has no gate.sh or test step before `vercel deploy --prod`.
- Safe to apply without judgment: no, owner decision (adding a toolchain file changes every local build).

## INF-06: audit-campaign.js and fix-campaign.js hard-code the main checkout's absolute path as ROOT
- Verdict: CONFIRMED (the hard-coded path); whether workflows should run elsewhere is the judgment
- Checked: `.claude/workflows/fix-campaign.js:34` and `audit-campaign.js:36` `const ROOT = 'C:/Users/Cole/source/repos/linkage_simulation/linkage-sim-rs'` in both the worktree copies and the main-checkout copies; no `args.root`. fix-campaign.js:42 "work in the main working tree", :47 and :80 `bash scripts/gate.sh` from ROOT, :56 and :91 `git stash push -u` in ROOT. The "main working tree" wording shows the design is deliberate, so a run from another worktree is a design change, not a slip.
- Safe to apply without judgment: no.

## INF-07: gui-smoke.js re-derives the calculator's `passed` from its checks but takes the linkage step's `passed` on the agent's word
- Verdict: CONFIRMED
- Checked: gui-smoke.js:8-19 `SMOKE_SCHEMA` requires `passed, canvas_present, console_errors, hub_page, title_warning`; :126-133 linkage step ends `|| noReading(SMOKE_SCHEMA)` and nothing re-checks it; :182-184 re-derives the calculator's `passed`; :196 `(linkage ? linkage.passed : true) && ...`. gui_smoke_verdict_test.cjs:141-142 `LINKAGE_GOOD`, :259-266 cases cover only `passed: false` and "agent returned nothing" for the linkage step; no case has `hub_page: false` or `title_warning: true` with `passed: true`. The committed and main-checkout copies were not byte-compared here.
- Safe to apply without judgment: no, it changes the verdict logic of a gate (13).

## INF-08: rust-test-fixer agent hard-codes `model: opus`, against the repo's model rule
- Verdict: CONFIRMED
- Checked: C:/Users/Cole/source/repos/linkage_simulation/.claude/agents/rust-test-fixer.md:6 `model: opus` (the finder says :7; frontmatter line 6). Its scope is tests/ compile repair (lines 15-26), mechanical. fbd-math-reviewer.md:5 `model: fable`. CLAUDE.md section 5 (as loaded in this session) says every agent definition names its model and never hard-code an Opus alias. The file is git-ignored (`.gitignore:30`), so it is not in the worktree.
- Safe to apply without judgment: no, owner decision, and the file is outside the worktree.

## INF-09: Post-edit cargo-check hook matches tests/*.rs but only checks the lib
- Verdict: CONFIRMED
- Checked: C:/Users/Cole/source/repos/linkage_simulation/.claude/hooks/cargo-check-on-rs-edit.ps1:1-2 header says "under linkage-sim-rs/src/"; :36 comment "src/ or tests/"; :39 regex `/linkage-sim-rs/(src|tests)/.*\.rs$`; :58 `cargo check --features native --lib`. `--lib` compiles only the library target, so an integration test under tests/ is never compiled by this hook. rust-test-fixer.md:33 uses `--tests`. Hook is untracked (outside the worktree).
- Safe to apply without judgment: no, it changes the owner's local hook and its run time, and lies outside the worktree.

## INF-10: fbd-derive skill and fbd-math-reviewer agent cite `evaluation.rs:424-425` for the actuator sign convention; it now sits at 437-438
- Verdict: CONFIRMED
- Checked: evaluation.rs:424 `let v_rel = unit.dot(...)`, :425 blank; :437-438 `let force_on_b = unit * net_force_along_unit; let force_on_a = -force_on_b;` inside `evaluate_linear_actuator` (:389). `.claude/skills/fbd-derive/SKILL.md:101` and `:198` cite `evaluation.rs:424–425`; `.claude/agents/fbd-math-reviewer.md:49` cites it and :99-100 and :109 say "~line 424" / "line 424". The sign convention itself (force_on_b = +F u, force_on_a = -F u) matches the code. Both files are untracked, in the main checkout only.
- Safe to apply without judgment: yes (prose only; nothing reads these line numbers), but the files are outside the worktree.
- Exact change: SKILL.md:101 old `(`forces/elements/evaluation.rs:424–425`)` new `(`forces/elements/evaluation.rs`, `evaluate_linear_actuator`)`; SKILL.md:198 old `linkage-sim-rs/src/forces/elements/evaluation.rs:424–425` new `linkage-sim-rs/src/forces/elements/evaluation.rs` (`evaluate_linear_actuator`); fbd-math-reviewer.md:49 old `forces/elements/evaluation.rs:424–425` new `forces/elements/evaluation.rs` (`evaluate_linear_actuator`); fbd-math-reviewer.md:99-100 and :109 drop the "~line 424" and "line 424" wording.

## INF-11: tests/braindump_repro.rs is tracked and still the repro BL-017 depends on, yet its header says "TEMPORARY ... Not for commit"
- Verdict: CONFIRMED (one small slip: the "(untracked)" note is in BL-017's notes, backlog.yaml:386, not BL-010's)
- Checked: braindump_repro.rs:1 `//! TEMPORARY audit repro tests for BL-017. Not for commit.`, :19-35 one test `bl010_side_sizing_mode_sweep_panics_debug_assert` whose comment says the cause was BL-022 and "panicked ... before BL-022" and that now asserts only `any finite` force; it is tracked (`git ls-files`). backlog.yaml:378-386 BL-017 `status: open`, acceptance "same repro test with #[should_panic] REMOVED passes", notes "Repro in tests/braindump_repro.rs (untracked)"; backlog.yaml:413 also names it. actuator_force_label.rs:4 refers to it. docs/analyses/2026-10-06-open-physics-review-response.md:160 says it "would now pass even if every pose failed".
- Safe to apply without judgment: no, a rename touches the test name, the backlog text and the BL-017 acceptance.

## INF-12: The same standard 4-bar is hand-built five times across two test files, with its initial-guess helper written twice
- Verdict: CONFIRMED (the count is low: six builders)
- Checked: golden_fixtures.rs:62-86, :306-328, :813-839 and the inline copy at :967-983 (4 builders); singular_behavior.rs:26-53 and :55-79 (2 more, so six, not five); the ground O2/O4, bar lengths, masses, J1-J4 and identity driver are the same in all. `fourbar_initial_guess` is identical except a comment at golden_fixtures.rs:94-108 and singular_behavior.rs:87-101; property_tests.rs:54 has a third helper with another signature that calls `try_fourbar_initial_q0`. Differences: coupler point, Gravity force, expression driver. `Glob **/tests/common/**` finds only magcoupling-rs/tests/common/mod.rs. There are also many in-crate unit-test copies (e.g. solver/reactions.rs:714, kinematics.rs:130) that the finding does not count.
- Safe to apply without judgment: no, a new shared module touching golden-fixture tests.

## INF-13: `first_actuator` is copied between two test files, and two helpers in one file have identical bodies
- Verdict: CONFIRMED
- Checked: actuator_force_label.rs:15-27 and compound_actuator_rebuild.rs:29-41 are identical except the `expect` text ("sample has a LinearActuator" vs "mechanism has a LinearActuator"); compound_actuator_rebuild.rs:61-67 `pin_to_pin_length` and :71-77 `element_length` have the same body; used at :137-138 and :207 with different elements (original pins vs the remapped one), so merging them is a rename, not a behaviour change.
- Safe to apply without judgment: no, it edits tests and needs the shared module of INF-12 for the cross-file part.

## INF-14: Assertions sit inside `if let Ok` / `if converged` / `return Ok(())` guards, so these tests pass when the solver fails every case
- Verdict: CONFIRMED for the structure; whether current cases reach the assertions was not run
- Checked: property_tests.rs:134-137 (`None => return Ok(())`), :139-155 (`if let Ok(result)` then `if result.converged` around the only `prop_assert!`s), :167-178 (`return Ok(())` on `!converged`, `if let Ok(q_dot)`); golden_fixtures.rs:1012-1014 (`continue` when not converged), :1039-1045 (`if let Ok(result) = vw` around the `assert!`). No counter of cases that reached an assertion in these tests; `any_torque_different` (:1003, :1048) does catch the all-skipped case for the spring test only.
- Safe to apply without judgment: no, the fix changes test strictness and may expose failing cases.

## INF-15: `expression_driver_matches_constant_speed` (property test) tests the `meval` crate, not this repo; `meval` is also listed twice in Cargo.toml
- Verdict: CONFIRMED
- Checked: property_tests.rs:324-362 parses `format!("{}*t+{}", ...)` with `meval::Expr` and compares to `omega*t+theta_0`; no repo type is called. golden_fixtures.rs:842 has a test of the same name that uses `add_expression_driver` (:832). Cargo.toml:26 `meval = "0.2"` under [dependencies] and :62 again under [dev-dependencies]; integration tests see regular dependencies, so :62 adds nothing. The repo uses meval in src (core/driver.rs:7, forces/elements/time_modulation.rs:59,115, gui/property_panel/force_editor.rs:1360-1365).
- Safe to apply without judgment: yes for the redundant dev-dependency line; no for the test (deleting or rewriting it is a choice).
- Exact change: linkage-sim-rs/Cargo.toml line 62: delete `meval = "0.2"` (the line under `[dev-dependencies]`, after `proptest = "1"`).

## INF-16: Direct dependencies `wasm-bindgen-futures` and `uuid` of linkage-sim-rs are named nowhere in its code
- Verdict: PARTLY. `wasm-bindgen-futures` holds; `uuid` is not shown to be unneeded.
- Checked: `Grep` `uuid|wasm_bindgen_futures|wasm-bindgen-futures|spawn_local|JsFuture` over linkage-sim-rs excluding Cargo.lock: only Cargo.toml:38 and :42 (2 hits), so neither is used in code. Cargo.lock: `wasm-bindgen-futures` 0.4.64 is also in the graph through other crates (finder: eframe, rfd), so removing the direct line would edit Cargo.lock's linkage-sim-rs entry. Cargo.lock `uuid` 1.22.0 depends on `js-sys` and `wasm-bindgen`, which are what uuid's `js` feature adds, so the direct `features = ["js"]` line is doing something; docs/ai/02-system.yaml:411 documents it as required for WASM. wasm32 build not run.
- Safe to apply without judgment: no, a manifest and lockfile change that only a wasm32 check can prove.

## INF-17: .gitignore carries a stale heading and two commented-out rules
- Verdict: CONFIRMED
- Checked: .gitignore:42 `# Golden test fixtures ... committed after Phase 4 completion for Rust port`, :43 `# data/benchmarks/golden/  # now tracked`, :44-47 live rules `*.zip`, `settings.json`, `settings.local.json`, `.cursorignore`, :48 `# Output -- GIFs are committed for reference`, :49 `# output/`. `git ls-files` tracks `data/benchmarks/golden/*.json` and `output/gifs/*.gif`, so both disabled rules are obsolete; `git ls-files` lists no `settings.json`, `settings.local.json`, `.cursorignore` or `*.zip`. `settings.json` is unanchored (any depth).
- Safe to apply without judgment: yes for the comment lines only (the two disabled rules and their headings); no for anchoring `settings.json` (owner decision).
- Exact change: .gitignore lines 48-49: delete both (`# Output ... GIFs are committed for reference` and `# output/`); line 42 old `# Golden test fixtures — committed after Phase 4 completion for Rust port` new `# Archives and local config files`; line 43: delete `# data/benchmarks/golden/  # now tracked`.

## Seen while checking
- docs-living DOC-10 quotes "tests/golden_fixtures.rs has 35 test functions"; it has 11 `#[test]` functions.
