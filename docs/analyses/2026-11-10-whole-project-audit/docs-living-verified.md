# Verified: docs-living

## DOC-01: README.md "File Structure" tree describes a Python layout that does not exist
- Verdict: CONFIRMED
- Checked: README.md:108-158 read; `git ls-files src` (53 files): no `core/load_cases.py`, `analysis/synthesis.py`, `analysis/optimization.py`, no `gui/`, `util/` holds only `__init__.py`; `git ls-files data` is 9 `data/benchmarks/golden/*.json` only; `git ls-files tests` has no `test_dynamics.py` or `test_units.py`. Real files missing from the tree exist (`core/point_mass.py`, `solvers/events.py`, `mass_matrix.py`, `sweep.py`, `viz/`, `forces/*`, `analysis/coupler.py`). ARCHITECTURE.md:265 does name `util/units.py`, which does not exist. Small correction: EXTENSIBILITY.md:178 names `analysis/linearization.py`, which is a stated future module, not one of the tree's invented files.
- Safe to apply without judgment: no, the replacement is either a new listing or a pointer to 03-structure.yaml, an owner choice.

## DOC-02: README.md and WASM_DEPLOYMENT.md say PNG/SVG/GIF/DXF export and HTML reports do not work in the browser; the code ships them
- Verdict: PARTLY. The documents contradict the code and each other; whether the browser export works at run time is not provable from reading.
- Checked: README.md:197 ("PNG/SVG/GIF/DXF export, autosave, and HTML reports are native-only") and :180 ("desktop and browser") read; WASM_DEPLOYMENT.md:111-116 read (item 2 "No PNG / SVG / GIF export"); `linkage-sim-rs/scripts/magcoupling_shipped.sh:25` `LINKAGE_WEB_ARGS=(--bin linkage-web ... --features raster)`; Cargo.toml:11-15 says raster is available on web; `src/gui/export/download.rs:106` `#[cfg(target_arch = "wasm32")] fn download_bytes_impl` builds a Blob and clicks an anchor; `menu_bar.rs` calls `export::download::download_text` many times (e.g. :253, :325, :357, :398, :426, :465, :494, :533) and gates only the PNG/GIF items with `#[cfg(feature = "raster")]` (:587), which the web build enables. Not checked: a click on Export PNG in a served build.
- Safe to apply without judgment: no, the owner should confirm the web exports work before the text changes.

## DOC-03: docs/guides/SAMPLES.md lists 17 of the 30 samples, with names and groups the GUI does not use
- Verdict: CONFIRMED
- Checked: SAMPLES.md tables 7 + 4 + 1 + 5 = 17. `linkage-sim-rs/src/gui/samples/mod.rs`: `all()` (:161) has 30 variants; `category()` (:124) gives 11 "4-Bar Mechanisms", 7 "6-Bar Mechanisms", 12 "Specialty Mechanisms". The 13 missing names all appear in `label()` and none in the guide. The guide's "Chebyshev (4-2-5-5)" is `Chebyshev Lambda (Straight-Line)` in `label()`. Slider-Crank is in the gallery's 4-Bar group, Quick-Return and Scotch Yoke in Specialty.
- Safe to apply without judgment: no, regenerating the guide writes new descriptive text and drops the old text.

## DOC-04: docs/FEATURES.md "Planned / Future" lists shipped features and lists one item twice
- Verdict: CONFIRMED
- Checked: FEATURES.md:284 "Required power curve", :285 "Actuator speed plot ... just needs display", :286 "Force margin visualization", :337 and :338 both "Mobile support" read. FEATURES.md:44 lists actuator speed and actuator power among the 15 plot tabs; `plot_panel/mod.rs:28-29` has `PlotTab::ActuatorSpeed`, `ActuatorPower`, and :200/:210/:335 use them. `plot_panel/actuator.rs:22,55,60,100-101` takes `actuator_rated_force` and draws rated lines and a safety-factor overlay. `git grep` for "Required power curve|Actuator speed plot" hits only FEATURES.md:284-285.
- Safe to apply without judgment: yes for the two shipped items; no for the force-margin and mobile-support lines (they need wording).
- Exact change: docs/FEATURES.md lines 284-285: delete both lines (the same two features are already listed at :44).

## DOC-05: linkage-sim-rs/README.md is in the scope list but does not exist; the README points to a SYSTEM.md whose count is stale
- Verdict: CONFIRMED
- Checked: `git ls-files linkage-sim-rs` shows `SYSTEM.md` and no README.md; README.md:82 says "File map of all 107 Rust source files"; `git ls-files linkage-sim-rs/src` has 137 `.rs` files. Stem search of those 137 against SYSTEM.md: exactly 30 are named nowhere (analysis/gravity_breakdown.rs, gui/menu_bar.rs, gui/theme.rs, solver/reactions.rs, all of solver/inverse_kinematics/ and others). SYSTEM.md:5 itself says 107 files, 45,941 lines, dated 2026-04-10. No document links to a linkage-sim-rs/README.md.
- Safe to apply without judgment: yes (README text only; nothing greps "107 Rust" except README.md:82 and SYSTEM.md:5).
- Exact change: README.md:82 old `File map of all 107 Rust source files` new `File map of the Rust source files`

## DOC-06: SOLIDWORKS_IMPORT.md "Manual JSON" example cannot be loaded by the app
- Verdict: CONFIRMED
- Checked: SOLIDWORKS_IMPORT.md:72-116 uses `"J1": {"Revolute": {...}}` and `"D1": {"ConstantSpeed": {...}}`; Field Reference says `Prismatic` "includes `axis` vector and `offset`". `src/io/schema.rs:191` and `:319` are `#[serde(tag = "type", rename_all = "snake_case")]` on `DriverJson` and `JointJson`; `MechanismJson` (:52-66) holds `joints: HashMap<String, JointJson>`, `drivers: HashMap<String, DriverJson>`, `forces: Vec<ForceElement>`. `Prismatic` (:339-349) has `axis_local_i: [f64; 2]` and `delta_theta_0`, no `offset`. No custom `Deserialize` or `untagged` in `src/io`. `src/gui/samples/custom_6bar.json` has 11 `"type":` entries. An externally tagged joint has no `type` field, so serde rejects it.
- Safe to apply without judgment: no, the example should be rewritten from a real exported file.

## DOC-07: ARCHITECTURE.md and README describe a Python design (plugin registry, `asteval`, friction tiers, driver types) that was never built or differs from the code
- Verdict: CONFIRMED
- Checked: ARCHITECTURE.md:146-161 force table lists `viscous_friction`, `regularized_coulomb`, `motor_droop`, `custom` (plugin registry); `git grep` of those names in `src` and `linkage-sim-rs/src` has 0 hits. `force_element.rs:17-30` has 13 variants (Gravity, LinearSpring, TorsionSpring, LinearDamper, RotaryDamper, ExternalForce, ExternalTorque, GasSpring, BearingFriction, JointLimit, Motor, LinearActuator, ForceZone). ARCHITECTURE.md:92-96 driver types `constant_speed | expression | interpolated`; Rust `DriverJson` has `ConstantSpeed` and `Expression` (schema.rs:193, :202). ARCHITECTURE.md:33,47 use `Izz_cg`. ARCHITECTURE.md:326 shows `trait ForceElement`; it is an enum. README.md:99 says "Python: `asteval` / Rust: `meval` ... Shipped"; `git grep -i asteval` in pyproject.toml and src: 0 hits; `meval` is used in `linkage-sim-rs/src/core/driver.rs:7`. Not re-read line by line: ARCHITECTURE.md:277-305 and :323-329.
- Safe to apply without judgment: no, whether the plan text stays as design intent is an owner decision.

## DOC-08: VALIDATION.md "Versioned Benchmark Suite" and its run command do not match the repo
- Verdict: CONFIRMED
- Checked: VALIDATION.md:125-145 says benchmarks are JSON files in `data/benchmarks/` with `benchmark_id` and `mechanism_file`; `git ls-files data` has only 9 `golden/*.json` snapshots; `git grep "benchmark_id|mechanism_file"` outside docs: 0 hits; no `benchmarks/fourbar_grashof.json`. Line 235 is `python -m pytest tests/ -k benchmark --benchmark-report`; `git grep benchmark-report` hits only that line; no `pytest_addoption`, no `pytest-benchmark` in the repo, so pytest rejects the option. Real files: `tests/test_benchmark_*.py` (5), `test_fourbar_benchmark.py`, `test_slidercrank_benchmark.py`, `test_textbook_benchmarks.py`.
- Safe to apply without judgment: yes for the command line only; the format and planned-list sections need a rewrite (no).
- Exact change: docs/architecture/VALIDATION.md:235 old `python -m pytest tests/ -k benchmark --benchmark-report` new `python -m pytest tests/ -k benchmark`

## DOC-09: docs/ai/02-system.yaml has at least seven syntax errors, not one
- Verdict: CONFIRMED
- Checked: read-only PyYAML parse: 02-system.yaml fails (line 14 col 5); 01-meta, 03-structure, 04-memory, backlog.yaml parse. With `: ` replaced in memory at lines 13, 80, 149, 153, 229, 299 (each a plain scalar holding `: `, read at those lines), the parse still fails at line 413 ("expected <block end>"); lines 412-414 are the `forces/elements/mod.rs ... #[cfg(test)]` item, where ` #` starts a comment inside a plain scalar. So six colon sites plus the :412 block. Already known as backlog.yaml BL-015 (title says "scanner error line 14-15", acceptance is the safe_load check), which only knew of the first error. The audit workflow reads this file (audit-campaign.js:37, :43 and others).
- Safe to apply without judgment: no, seven multi-line edits that must keep the content unchanged, and BL-015 owns the fix (acceptance: content semantically unchanged).

## DOC-10: Hard-coded counts in the living documents are stale
- Verdict: PARTLY. The stale counts hold; one sub-claim is wrong.
- Checked: README.md:82 and SYSTEM.md:5 say 107 files; `git ls-files linkage-sim-rs/src` has 137 `.rs` files and 30 are named nowhere in SYSTEM.md (see DOC-05). README.md:98, :162 and FEATURES.md:38 say "644 tests"; `git grep -c "#\[test\]"` gives 977 in src plus 117 in tests = 1094 (confirmed). TRAJECTORY_MODE.md:192 says "668+ lib tests". WRONG sub-claim: "`tests/golden_fixtures.rs` has 35 test functions"; it has 11 `#[test]` (24 `fn`s in all, the rest helpers). The text does not say what the 644 counts, so it may mean golden comparisons. Python 1088 not checked (baseline only).
- Safe to apply without judgment: no for the set (what 644 counts is unclear; SYSTEM.md is a dated generated map); the README.md:82 line alone is covered by DOC-05.

## DOC-11: 03-structure.yaml omits modules the update rule says it must list
- Verdict: CONFIRMED
- Checked: stems searched in docs/ai/03-structure.yaml: no hit for `inverse_kinematics`, `control_target`, `derivatives`, `preferences`, `trajectory_ops`, `trajectory_panel`, `profile_input`, `target_picker`, `plot_panel/trajectory.rs`, `eq_rendering`, `error.rs`; those files all exist in `git ls-files`. `io.files` (line 110) is `[schema, serialization, to_json]`; `io/from_json.rs` and `io/error.rs` exist (the one `from_json` hit is in the magcoupling `gui:` line 68). `feature_flags.native` (line 196) is `[rfd, resvg, gif, open, dirs]`; Cargo.toml:10 `native = ["raster", "dep:rfd", "dep:open", "dep:dirs"]`, :15 `raster`. Line 218 says `name/description/category`; the method is `label()` (mod.rs:52). Rule at 01-meta.yaml:24 read via finder; not re-read.
- Safe to apply without judgment: yes for the one-word rename; no for the module additions (they need descriptions).
- Exact change: docs/ai/03-structure.yaml:218 old `add SampleMechanism variant + all() + name/description/category` new `add SampleMechanism variant + all() + label/description/category`

## DOC-12: EXTENSIBILITY.md promises a registry, schema-generated panels and generic serialization; the Rust code has none of them
- Verdict: CONFIRMED
- Checked: EXTENSIBILITY.md:13 ("Solver changes required: none"), :19-22 (register in the force element registry), :96-123 Rules 1-4 (SCHEMA dict, no per-type serialization). `git grep -i "registry|SCHEMA"` over `linkage-sim-rs/src/forces`, `src/gui/property_panel` and `src/linkage_sim/forces`: one hit, an unrelated comment about the file schema version (element_types.rs:347). 03-structure.yaml:199-204 `add_new_force_element` lists five hand edits (element_types, evaluation, force_element enum + dispatch, force_render, force_editor). `draw_force_element_details` is at force_editor.rs:61 in a 1,416-line file (the "1,108 lines" is from leads.md, not rechecked). Analysis code matches on the enum (`analysis/gravity_breakdown.rs:190`).
- Safe to apply without judgment: no, whether Rules 3-4 remain goals is an owner decision.

## DOC-13: Rules in CLAUDE.md section 5 and AGENTS.md point at agent files and skills that are not in the repository, and the one agent file that exists breaks the rule
- Verdict: CONFIRMED
- Checked: `.gitignore:30` `.claude/*`, :31 `!.claude/workflows/`; `git ls-files .claude` lists only the three workflows. Main checkout `.claude/` holds `agents/fbd-math-reviewer.md`, `agents/rust-test-fixer.md`, `skills/fbd-derive/SKILL.md`, `hooks/*`, all untracked. `fbd-math-reviewer.md:5` `model: fable`; `rust-test-fixer.md:6` `model: opus` (the finder's :5 here and :7 in INF-08 are both off by one). AGENTS.md:23 (main) / :22 (worktree copy) requires the fbd-math-reviewer pass for `risk: physics` items. The worktree AGENTS.md lacks the model-roles line that the main copy has at :21.
- Safe to apply without judgment: no, owner decision (track the agents or declare them local; the file is outside git).

## DOC-14: 04-memory.yaml `open_questions` is 60 percent closed items, and several open ones are stale
- Verdict: PARTLY. The size and counts hold; the two stale items are partly shown, the third is not.
- Checked: PyYAML parse of 04-memory.yaml `open_questions`: 66 entries, 36,187 characters; 40 start with RESOLVED or DONE, 21,028 characters. "Tidy after M2" says about 1,350 lines; `magcoupling-rs/src/engine/temperature.rs` is 2,545 non-blank lines, so stale. "Deploy (M4-1)" says nothing pushed and that the next push ships /magcoupling/; 01-meta.yaml:3 says the site is live at /tools/magcoupler/; but the entry itself already ends "(since 2026-10-06 the calculator is at /tools/magcoupler/)", so it is half-corrected. "Reaction-force validation reference choice" is open and its topic equals 01-meta.yaml:16 `active_focus`; that the choice is already made is not provable from these files.
- Safe to apply without judgment: no, the resolved entries may hold detail the tracker lacks.

## DOC-15: AGENTS.md says CLAUDE.md "holds the project's working rules"; CLAUDE.md holds none of the repository's rules, and has stray characters
- Verdict: CONFIRMED
- Checked: AGENTS.md:20 (main and worktree) says CLAUDE.md holds the working rules; the rules are at AGENTS.md:22-27 (docs/ai, backlog, gate, never commit chebyshev PNGs, nothing pushed without the go). CLAUDE.md (both copies) has no `docs/ai`, `AGENTS`, `chebyshev` or `pushed`. U+FEFF count by script: 60 characters on 30 lines (7-11, 16-20, 25-29, 34-37, 42-45, 50-54, 57-58), 2 at the start of each of those 30 lines, in both copies.
- Safe to apply without judgment: no, CLAUDE.md is the owner's own prompt text.

## DOC-16: FEATURES.md claims "zero-warning compilation" for the WASM build; the baseline shows hundreds of warnings
- Verdict: NOT CONFIRMED
- Checked: FEATURES.md:64 has the phrase (also docs/history/ROADMAP_IMPLEMENTATION.md:459). gate.sh:82-83 runs `cargo clippy --all-targets` without `-D warnings` and :85-86 a wasm `cargo check`; backlog.yaml:239 records about 289 clippy warnings. Those are clippy lints; the claim is about compilation of the WASM build, and no read shows rustc warnings on wasm32. The finder rated it "unsure" itself.
- Safe to apply without judgment: no.

## DOC-17: README.md Rust tree lists 8 of 15 integration test files and omits modules
- Verdict: CONFIRMED
- Checked: README.md:201-227 tests block names 8 files; `git ls-files linkage-sim-rs/tests` has 15; missing are `actuator_force_label.rs`, `braindump_repro.rs`, `compound_actuator_rebuild.rs`, `dxf_import_test.rs`, `gravity_breakdown_reference.rs`, `web_layout.rs`, `wheel_contact_sweep.rs` (exactly the seven named). `src/bin/` has `linkage_gui.rs` and `linkage_web.rs`; the tree lists only `linkage_gui`. No `geometry/`, `error.rs`, `solver/reactions.rs` or `solver/inverse_kinematics/` in the tree (all exist). Line 205 says "12 variants"; the enum has 13.
- Safe to apply without judgment: no, deleting versus regenerating the tree is a choice.

## Seen while checking
- README.md:98 and :205 and FEATURES.md say 12 force element types; `ForceElement` has 13 variants (force_element.rs:17-30).
