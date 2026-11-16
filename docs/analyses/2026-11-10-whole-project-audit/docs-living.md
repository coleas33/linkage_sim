# Audit findings: docs-living
Scope read: README.md, AGENTS.md and CLAUDE.md (main-checkout copies), docs/FEATURES.md, docs/ai/01-meta.yaml, guides, WASM/samples docs; architecture, reference, ai yaml structure (see Not read).

## DOC-01: README.md "File Structure" tree describes a Python layout that does not exist
- Kind: inconsistent
- Where: README.md:108-158
- Evidence: the tree is rooted at `linkage-sim/` with `core/bodies.py`, `core/load_cases.py`, `analysis/synthesis.py`, `analysis/optimization.py`, `gui/*.py` (canvas, property_panel, animation, plot_panel), `util/units.py`, `util/expressions.py`, `util/plugin_registry.py`, `data/templates/`, `data/examples/`, and tests `test_dynamics.py`, `test_units.py`. The real Python package is `src/linkage_sim/` (git ls-files src): no `load_cases.py`, `synthesis.py`, `optimization.py`, no `gui/` and no `util/*.py` beyond `__init__.py`; `data/` holds only `benchmarks/`; `tests/` has no `test_dynamics.py` or `test_units.py`. Real files missing from the tree: `core/point_mass.py`, `solvers/events.py`, `solvers/mass_matrix.py`, `solvers/sweep.py`, `viz/`, `forces/*`, `analysis/coupler.py` etc.
- Why it matters: the first thing a reader sees about the repo layout is a plan from the Phase 1 design, not the tree; `ARCHITECTURE.md:265` and `EXTENSIBILITY.md:178` point at the same invented files.
- Suggested action: replace the tree with the real `src/linkage_sim/` listing (or delete it and point to docs/ai/03-structure.yaml); drop the Phase 5/6 modules that were never built.
- Fix class: needs-judgment
- Confidence: certain

## DOC-02: README.md and WASM_DEPLOYMENT.md say PNG/SVG/GIF/DXF export and HTML reports do not work in the browser; the code ships them
- Kind: inconsistent
- Where: README.md:197; docs/guides/WASM_DEPLOYMENT.md:111-116
- Evidence: README:197 "file dialogs, PNG/SVG/GIF/DXF export, autosave, and HTML reports are native-only"; WASM_DEPLOYMENT:114 "No PNG / SVG / GIF export -- Export functions rely on native filesystem access". But `linkage-sim-rs/scripts/magcoupling_shipped.sh` builds the web bundle with `--features raster` (LINKAGE_WEB_ARGS), `src/gui/export/download.rs:106` has a `#[cfg(target_arch = "wasm32")]` Blob download path, and `menu_bar.rs:419,487,524,587,626` call it for Export SVG, DXF, HTML report, PNG and GIF with no wasm gate. README:180 itself says the animated HTML export works "desktop and browser".
- Why it matters: users and agents are told to expect no export on the web, and the README contradicts itself.
- Suggested action: rewrite both limitation lists to what is gated: file dialogs (rfd), autosave, recent files; state that exports download through the browser. The owner should confirm the web exports work before the text changes.
- Fix class: needs-judgment
- Confidence: likely (settle by clicking Export PNG in the served web build)

## DOC-03: docs/guides/SAMPLES.md lists 17 of the 30 samples, with names and groups the GUI does not use
- Kind: inconsistent
- Where: docs/guides/SAMPLES.md:5-40; README.md:166; docs/FEATURES.md:52,181
- Evidence: SAMPLES.md has tables "Four-Bar" (7), "Slider-Crank Variants" (4), "Special Purpose" (1), "Six-Bar" (5) = 17 entries. `SampleMechanism::all()` in `linkage-sim-rs/src/gui/samples/mod.rs` has 30 variants in 3 categories (11 "4-Bar Mechanisms", 7 "6-Bar Mechanisms", 12 "Specialty Mechanisms"). Missing from the guide: Parallelogram Press, Parallelogram + Actuator, Chebyshev Lambda + Actuator, Hoeken, Roberts, Offset Slider-Crank, Whitworth Quick-Return, Bell Crank, Coupler Curve (figure-8), 6-Bar Watt II, Pantograph, Strandbeest, Custom 6-Bar Press. The guide's "Chebyshev (4-2-5-5)" is `Chebyshev Lambda (Straight-Line)` in the app, and it files Slider-Crank, Quick-Return, Scotch Yoke under other groups than the gallery does.
- Why it matters: the guide the README sends readers to for "built-in sample mechanisms" is 13 samples short.
- Suggested action: regenerate the guide from `label()`/`description()`/`category()`; keep the three gallery categories.
- Fix class: mechanical-safe
- Confidence: certain

## DOC-04: docs/FEATURES.md "Planned / Future" lists shipped features and lists one item twice
- Kind: inconsistent
- Where: docs/FEATURES.md:284-285,286,337-338 (against :44,:206-210)
- Evidence: Planned says "Required power curve -- P = F x v ..." and "Actuator speed plot -- dL/dt ... just needs display"; `PlotTab::ActuatorSpeed` and `PlotTab::ActuatorPower` exist (`src/gui/plot_panel/mod.rs:27-28`) and FEATURES.md:44 lists both as shipped. "Force margin visualization" (rated force input) is partly shipped: `plot_panel/actuator.rs:22,29` takes `actuator_rated_force` and `ACTUATOR_FORCE_TAB_TIP` mentions a green safe-zone band. Lines 337 and 338 are both "Mobile support" with different text ("3 commits saved in git, ready to cherry-pick").
- Why it matters: the roadmap section tells the owner work is open that is done, and the duplicate invites the question which one is current.
- Suggested action: move the speed and power items to Implemented, mark force margin partial, merge the two Mobile support lines.
- Fix class: mechanical-safe
- Confidence: certain

## DOC-05: linkage-sim-rs/README.md is in the scope list but does not exist; the README points to a SYSTEM.md whose count is stale
- Kind: inconsistent
- Where: README.md:82
- Evidence: `git ls-files linkage-sim-rs` has no README.md (only SYSTEM.md, Cargo files, scripts, web). README.md:82 says `linkage-sim-rs/SYSTEM.md` is the "File map of all 107 Rust source files"; `git ls-files linkage-sim-rs/src '*.rs'` gives 137 files.
- Why it matters: the count is wrong by 30, so the claim "all" files are mapped cannot be true.
- Suggested action: drop the number from the README line (it goes stale each commit) and check SYSTEM.md for the 30 unlisted files.
- Fix class: mechanical-safe
- Confidence: likely (SYSTEM.md itself was not diffed against the 137 files)

## DOC-06: SOLIDWORKS_IMPORT.md "Manual JSON" example cannot be loaded by the app
- Kind: broken
- Where: docs/guides/SOLIDWORKS_IMPORT.md:72-116 and the Field Reference table at :135-139
- Evidence: the guide writes joints as `"J1": {"Revolute": {"body_i": ...}}` and drivers as `"D1": {"ConstantSpeed": {...}}` (externally tagged). `linkage-sim-rs/src/io/schema.rs:191` and `:319` declare `#[serde(tag = "type", rename_all = "snake_case")]` on `DriverJson` and `JointJson`, and the shipped sample `src/gui/samples/custom_6bar.json` has `"J1": {"body_i": "link_1", ..., "type": "revolute"}` and `"D1": {..., "type": "constant_speed"}`. The guide also says `Prismatic` "includes `axis` vector and offset"; the schema field is `axis_local_i` and there is no `offset`.
- Why it matters: a user who builds the file as the guide says gets a parse error on load; the guide is the only document for hand-written JSON.
- Suggested action: rewrite the example from a real exported file (internally tagged, `"type": "revolute"`), fix the Prismatic row, and say that `forces` is an array of tagged `ForceElement` values.
- Fix class: needs-judgment
- Confidence: certain (serde tag attribute and shipped sample both read)

## DOC-07: ARCHITECTURE.md and README describe a Python design (plugin registry, `asteval`, friction tiers, driver types) that was never built or differs from the code
- Kind: inconsistent
- Where: docs/architecture/ARCHITECTURE.md:52,92-98,146-161,163-175,277-305,323-329; README.md:99
- Evidence: the force table lists `viscous_friction`, `regularized_coulomb`, `motor_droop`, `custom` (plugin registry); none exists in `src/linkage_sim/forces/` (it has `friction.py` with `CoulombFriction`, `joint_limit.py`, `bearing_friction.py`) or in the Rust `ForceElement` enum (`force_element.rs:17-31`: 13 variants incl. `ExternalTorque`, `BearingFriction`, `JointLimit`, `ForceZone`). Driver types "constant_speed | expression | interpolated": Python `core/drivers.py` has only `RevoluteDriver`; Rust `DriverJson` has `ConstantSpeed` and `Expression`. JointConstraint "type: revolute | prismatic | fixed" omits `CamFollower` (`core/constraint/mod.rs:31-36`). The JSON example uses `"Izz_cg"` and `"point_masses": {}`; the schema uses `izz_cg` and a per-body `point_masses` list. `ForceElement.evaluate()` is shown as a trait (ARCHITECTURE.md:326); it is an enum. README.md:99 says the expression evaluator is "Python: `asteval` / Rust: `meval` ... Shipped"; `pyproject.toml` has no asteval and `src/` has no evaluator.
- Why it matters: the document called "Core data model" is a Phase-1 plan; a reader (or agent) following it will look for types and fields that do not exist.
- Suggested action: mark the sections that are plans (friction tiers, plugin registry, interpolated driver) as design intent, bring the force table, driver list and JSON example to the Rust schema, and fix README:99 to "Rust: meval; Python: none".
- Fix class: needs-judgment
- Confidence: certain for each named item; whether the owner wants the plan text kept is an owner decision

## DOC-08: VALIDATION.md "Versioned Benchmark Suite" and its run command do not match the repo
- Kind: broken
- Where: docs/architecture/VALIDATION.md:127-139 (format), :180-232 (planned list), :233-236 (command)
- Evidence: "Benchmarks are stored in `data/benchmarks/` as JSON files with expected results ... `mechanism_file: benchmarks/fourbar_grashof.json`". `git ls-files data` has only `data/benchmarks/golden/*.json` (9 exported solver snapshots, no `benchmark_id`/`tests` schema). The run command `python -m pytest tests/ -k benchmark --benchmark-report` uses a flag that exists nowhere (`git grep benchmark-report` hits only this line); pytest rejects unknown options. The real benchmarks are `tests/test_benchmark_*.py`, `test_fourbar_benchmark.py`, `test_slidercrank_benchmark.py`, `test_textbook_benchmarks.py`. Mechanism ids in the "Planned" tables (`fourbar_grashof_cr`, `pendulum_damped`, ...) are not the test names.
- Why it matters: the command in the document fails when followed.
- Suggested action: replace the command with `python -m pytest tests -k benchmark`, and rewrite the section around the test files and golden fixtures that exist.
- Fix class: needs-judgment
- Confidence: certain

## DOC-09: docs/ai/02-system.yaml has at least seven syntax errors, not one
- Kind: broken
- Where: docs/ai/02-system.yaml:13, :80, :149, :153, :229, :299, and the block at :409-414 (the `forces/elements/mod.rs` item, which breaks after the earlier ones are fixed)
- Evidence: the baseline reported line 13 only. With line 13 patched in memory, PyYAML fails next at line 80 ("longer depends on the crate: its Tools menu links..." a plain scalar holding ": "); patching each ": " in turn, it then stops at 149 ("same rule: every per-source ..."), 153 ("required_totals: for an actuator..."), 229 ("pointer.press_origin(): egui reports ..."), 299 ("gravity_magnitude: the element carries ..."), and finally cannot recover at :413.
- Why it matters: the first file agents are told to read (AGENTS.md:22, 01-meta.yaml `before_work`) cannot be parsed by any YAML tool; fixing only line 13 leaves it broken.
- Suggested action: quote or block-scalar (`|`/`>`) every plain multi-line scalar that contains ": " or starts with a bracket, then add a YAML-parse check of docs/ai/*.yaml to the gate (a 10-line python parse fits next to gate 11's pin check).
- Fix class: mechanical-safe
- Confidence: certain (parse run on the committed file; errors after :413 not enumerated)

## DOC-10: Hard-coded counts in the living documents are stale
- Kind: inconsistent
- Where: README.md:82 ("107 Rust source files"), :98 and :162 ("644 tests"); docs/FEATURES.md:38 ("644 tests"); docs/guides/TRAJECTORY_MODE.md:192 ("668+ lib tests"); linkage-sim-rs/SYSTEM.md:4 (107 files, 45,941 lines, dated 2026-04-10)
- Evidence: `git ls-files linkage-sim-rs/src` has 137 `.rs` files, and 30 of them are named nowhere in SYSTEM.md. `git grep -h "#\[test\]"` in linkage-sim-rs gives 1094 (977 in src, 117 in tests). The Python suite is 1088 passed (baseline). "644 tests" is quoted as the count validated against golden fixtures; `tests/golden_fixtures.rs` has 35 test functions.
- Why it matters: three documents give three different numbers, none the current one.
- Suggested action: drop the counts (they go stale on every commit) or generate them in the gate; if kept, say what is counted.
- Fix class: mechanical-safe
- Confidence: certain for 107 and 668+; "644" may count golden comparisons rather than tests, which the text does not say

## DOC-11: 03-structure.yaml omits modules the update rule says it must list
- Kind: inconsistent
- Where: docs/ai/03-structure.yaml (`modules`, `gui_submodules`, `feature_flags`, `where_things_belong.add_new_sample_mechanism`); rule at docs/ai/01-meta.yaml:24 ("update 03-structure.yaml if module layout changed")
- Evidence: a stem search of `linkage-sim-rs/src/**/*.rs` against 03-structure.yaml finds no mention of `solver/inverse_kinematics/` (control_target, derivatives, severity, test_helpers: the whole trajectory-mode solver), `io/from_json.rs`, `io/error.rs`, `gui/state/preferences.rs`, `gui/state/trajectory_ops.rs`, `gui/trajectory_panel/` (profile_input, target_picker), `gui/plot_panel/trajectory.rs`, `gui/eq_rendering.rs`, `error.rs`. `modules.io.files` lists only schema, serialization, to_json. `feature_flags.native` lists `rfd, resvg, gif, open, dirs` but not the `raster` feature (Cargo.toml: `native = ["raster", ...]`; the web build enables `raster` alone). The sample recipe says "name/description/category"; the method is `label()`.
- Why it matters: the map agents use to find where things belong has no entry for trajectory mode, the largest feature added since it was written.
- Suggested action: add the missing modules and the `raster` flag; rename `name` to `label`.
- Fix class: mechanical-safe
- Confidence: certain

## DOC-12: EXTENSIBILITY.md promises a registry, schema-generated panels and generic serialization; the Rust code has none of them
- Kind: inconsistent
- Where: docs/architecture/EXTENSIBILITY.md:13-22 (ForceElement steps 1-4), :96-123 (Rules 1-4), against docs/ai/03-structure.yaml `where_things_belong.add_new_force_element`
- Evidence: EXTENSIBILITY says adding a ForceElement needs "Solver changes required: none", step 3 "Register the type name in the force element registry", Rule 2 "Mechanism Holds Generic Collections", Rule 3 "GUI Panels Are Auto-Generated from Schemas" (a `SCHEMA = {...}` dict per type), Rule 4 "No per-type serialization code". `git grep -i "registry\|SCHEMA"` over `linkage-sim-rs/src/forces`, `src/gui/property_panel` and `src/linkage_sim/forces` finds no registry or schema. 03-structure.yaml lists the real recipe: five hand edits (element_types.rs, evaluation.rs, force_element.rs, `gui/canvas/rendering/force_render.rs`, `gui/property_panel/force_editor.rs`, a 1,108-line function per leads.md). Analysis code also matches on the variant (`analysis/validation.rs`, `force_breakdown.rs`, `virtual_work.rs`, `forces/compound.rs`).
- Why it matters: Rule 1 (no type switching) is stated as an invariant and is not true of the code; a contributor following the document looks for a registry that is not there.
- Suggested action: rewrite the "To add" lists to the real Rust procedure (point to 03-structure.yaml) and mark Rules 3 and 4 as unmet goals.
- Fix class: owner-decision (whether Rules 3-4 remain goals)
- Confidence: certain for the missing registry and schema

## DOC-13: Rules in CLAUDE.md section 5 and AGENTS.md point at agent files and skills that are not in the repository, and the one agent file that exists breaks the rule
- Kind: inconsistent
- Where: CLAUDE.md section 5 (main-checkout copy: "Every agent definition (`.claude/agents/*.md`) ... names its model"; committed copy: "Never hard-code an Opus alias or version"); AGENTS.md:21,23 (main-checkout copy); docs/ai/01-meta.yaml:21-24; main checkout `.claude/agents/rust-test-fixer.md:5`
- Evidence: `.gitignore:30` is `.claude/*` with only `!.claude/workflows/` re-included, so `git ls-files .claude` lists the three workflows only; `.claude/agents/fbd-math-reviewer.md`, `rust-test-fixer.md`, `.claude/skills/fbd-derive/SKILL.md` and the hooks exist only in the owner's checkout. AGENTS.md:23 requires "the fbd-math-reviewer pass" for every `risk: physics` item, an agent a clone (or Codex, whom AGENTS.md addresses) does not have. In the main checkout `rust-test-fixer.md` has `model: opus`, against both copies of CLAUDE.md section 5 and against its own job (edits `tests/` only: mechanical, so `sonnet`). backlog.yaml:259 records that this file's premise is stale; it does not cover the model line.
- Why it matters: the model-selection rule is checked nowhere and broken in the only agent file it governs; the physics-review gate names a tool nobody else can run.
- Suggested action: either track `.claude/agents/` (and the skill) or say in AGENTS.md that they are local; set `rust-test-fixer.md` to `sonnet` (or delete it with the item at backlog.yaml:259).
- Fix class: owner-decision
- Confidence: certain (gitignore and file contents read in both checkouts)

## DOC-14: 04-memory.yaml `open_questions` is 60 percent closed items, and several open ones are stale
- Kind: not needed
- Where: docs/ai/04-memory.yaml `open_questions` (66 entries, 36 kB; 40 start with RESOLVED or DONE, 21 kB); entries "Tidy after M2", "Deploy (M4-1)", "Reaction-force validation reference choice"
- Evidence: parsed list: 66 entries, 40 matching `^(RESOLVED|DONE)`. "Tidy after M2: `magcoupling-rs/src/engine/temperature.rs` is about 1,350 lines" while the file is 2,632 lines. "Deploy (M4-1): nothing pushed ... ships /magcoupling/ on the next push" while 01-meta.yaml:3 says the calculator is live at /tools/magcoupler/ and the old path redirects. "Reaction-force validation reference choice: which of options A/B/C/Hybrid ... Blocks the implementation plan for the current active_focus" while 01-meta.yaml:16 is that same focus and the work is under way.
- Why it matters: every agent must read this file before every job (AGENTS.md:22); 21 kB of it is history already recorded in 05-update-tracker.md and the plans, and the stale open items mislead.
- Suggested action: move RESOLVED/DONE entries to the tracker (or delete where the tracker has them), and correct or close the three stale items.
- Fix class: needs-judgment
- Confidence: likely (the resolved entries may carry details not in the tracker; diffing against 05-update-tracker.md would settle it)

## DOC-15: AGENTS.md says CLAUDE.md "holds the project's working rules"; CLAUDE.md holds none of the repository's rules, and has stray characters
- Kind: inconsistent
- Where: AGENTS.md:20 (both copies); CLAUDE.md:1-58, stray characters on lines 7-11, 54, 57-58 (both copies)
- Evidence: CLAUDE.md is a pasted generic prompt ("Claude Code Prompt for Plan Mode #prompts", the plan-review script, the behavioral guidelines) plus section 5. It never mentions `docs/ai/`, `backlog.yaml`, the gate, `docs/chebyshev_lambda/*.png`, "nothing is pushed without the user's go", or "docs change with code"; those live in AGENTS.md:22-27 (and in the owner's memory). Claude Code loads CLAUDE.md automatically, so a fresh Claude session does not see the repository's own rules. The file also holds 60 U+FEFF characters (two at the start of each of 30 bullet lines) left from a paste.
- Why it matters: the pointer runs the wrong way: the auto-loaded file lacks the rules, and the file that has them says it defers to the first.
- Suggested action: move the AGENTS.md "Project rules" list into CLAUDE.md (or have CLAUDE.md say "read AGENTS.md first"); strip the U+FEFF characters.
- Fix class: owner-decision (CLAUDE.md is the owner's own prompt text)
- Confidence: certain

## DOC-16: FEATURES.md claims "zero-warning compilation" for the WASM build; the baseline shows hundreds of warnings
- Kind: inconsistent
- Where: docs/FEATURES.md:64
- Evidence: "WebAssembly build (feature flags, WASM entry point, zero-warning compilation)". The baseline reports 259 clippy warnings in the lib, about 25 rustc unused-import warnings and 11 "never used" items on the native build; gate 3 only runs a WASM `cargo check` and fails on no warning, and gate 2 does not deny warnings for linkage-sim-rs (magcoupling-rs does, gates 5 and 8).
- Why it matters: a status claim the tools contradict.
- Suggested action: delete the phrase.
- Fix class: mechanical-safe
- Confidence: unsure (the claim may have been true for the wasm32 target alone when written; a wasm32 `cargo check` would settle it)

## DOC-17: README.md Rust tree lists 8 of 15 integration test files and omits modules
- Kind: inconsistent
- Where: README.md:201-227
- Evidence: the `tests/` block names 8 files; `git ls-files linkage-sim-rs/tests` has 15 (missing: `actuator_force_label.rs`, `braindump_repro.rs`, `compound_actuator_rebuild.rs`, `dxf_import_test.rs`, `gravity_breakdown_reference.rs`, `web_layout.rs`, `wheel_contact_sweep.rs`). The `src/` block has no `geometry/`, no `error.rs`, no `bin/linkage_web.rs` (only `linkage_gui`, though the web bundle builds `linkage-web`), no `solver/reactions.rs`, `solver/inverse_kinematics/`, `forces/compound.rs`, `analysis/gravity_breakdown.rs`; `io/` shows only serialization.
- Why it matters: a second copy of the layout that goes stale; docs/ai/03-structure.yaml is the maintained one.
- Suggested action: delete the second tree and link 03-structure.yaml, or regenerate it.
- Fix class: mechanical-safe
- Confidence: certain

## Leads settled
- leads.md 1, `uuid`: not real. docs/ai/02-system.yaml:411 says the `dxf` crate pulls `uuid`, which needs `features = ["js"]` for WASM; the direct declaration exists to turn that feature on (not a code use).
- leads.md 1, `wasm-bindgen-futures`: outside my part (Cargo.toml); no `spawn_local`, `JsFuture` or `wasm_bindgen_futures` in linkage-sim-rs; docs name it only in backlog/plans for magcoupling-rs. Likely unused, left to the code finder.
- leads.md 1, `magcoupling-rs` dev-dependency on itself: not real; `magcoupling-rs/Cargo.toml:114` is the self dev-dependency that turns on `workbook-parity` for tests, and `magcoupling_shipped.sh` explains it.
- leads.md 2 `set_angle`, `is_cam_follower`, `update_prismatic_axis`: code, not in my part; no document names them.
- leads.md 5 (docs/chebyshev_lambda PNGs, output/gifs): outside my part. AGENTS.md:26 names the PNGs as test-rewritten files that must never be committed, yet they are tracked.
- leads.md 6 `draw_force_element_details` (1,108 lines): used only as evidence in DOC-12.

## Not read
- docs/history/** (a record; no living document sends readers to it as current except README.md:88-89 and :98, which label it History)
- docs/architecture/NUMERICAL_FORMULATION.md, ENGINEERING_OUTPUTS.md, ANALYSIS_MODES.md (only skimmed for paths, status words and the forward-dynamics section; no maths checked)
- docs/reference/FORCE_ELEMENTS.md (headings only: 13 elements, README and FEATURES say 12)
- docs/ai/02-system.yaml and docs/ai/backlog.yaml beyond structure checks (parse, ids, statuses, fields); docs/ai/04-memory.yaml beyond `open_questions` and `active_issues`
- docs/ai/05-update-tracker.md (outside the list)
- linkage-sim-rs/README.md does not exist (DOC-05)