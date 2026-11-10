# Audit findings: mag-engine-explain
Scope read: api.rs, meta.rs (all non-test and the macros), warnings.rs, picks.rs, assumptions.rs, compat.rs, constants.rs, mod.rs, deviations.rs (header, types, E21-E24 entries, tail and tests; the E1-E20 data rows skimmed), explain/{mod,scope,record,registry,eval(1-480),probes,render,note_text,plain(1-230),tables,symbols(1-60)}.rs, explain/markup.rs (1-230 and by search), explain/notes.rs (types, tail, tests start; note texts by search), explain/records/{mod,dashboard,geometry,temperature,clamps}.rs and the heads of demagnetization.rs and shaft_joint.rs


## MEX-01: Assumption helpers rebuild the defaults and the whole input schema on every call, and the GUI calls them every frame
- Kind: inefficient
- Where: magcoupling-rs/src/engine/assumptions.rs:213-247 (`states`, `modified`); callers magcoupling-rs/src/gui/panel.rs:1449 (`assumptions_ui`), :1725 (`assumptions_banner`), magcoupling-rs/src/engine/explain/registry.rs:659-690 (`term_style` -> `modified_assumptions_upstream` -> `assumptions::modified`), magcoupling-rs/src/gui/explorer.rs:1071 (once per term row), :1001
- Evidence: `modified()` and `states()` each start with `let defaults = DesignInputs::default();` and `states()` also calls `input_rows(inputs)` (allocates a String path and a Value for every input, hundreds of them) only to look up 18 units. `Registry` already holds `self.defaults` (registry.rs:149, used at :662) but `modified_assumptions_upstream` ignores it. explorer.rs:1071 calls `registry().term_style(&row.path, ...)` inside the per-row egui loop, so each visible term row of each frame rebuilds a default `DesignInputs` and compares 19 paths by string match. registry.rs:7-9 claims "what a frame asks per shown value is a map lookup", which is not true of `term_style`.
- Why it matters: egui repaints on interaction (every slider drag frame); this is avoidable allocation on the hot UI path and contradicts the registry's own doc.
- Suggested action: compare against a cached default (`Registry.defaults`, or a `static OnceLock<DesignInputs>`), read units from `InputSet` metadata (or a cached unit table) instead of `input_rows`, and compute `modified` once per frame for the term list.
- Fix class: needs-judgment
- Confidence: likely (a GUI timing run would give the size; the structure is certain)

## MEX-02: Stale "placeholder" and "until then" comments in warnings.rs: the teaching notes exist
- Kind: inconsistent
- Where: magcoupling-rs/src/engine/warnings.rs:2-4, :30-31
- Evidence: header: "a teaching-note id (the notes are Addendum A4's; the ids are placeholders until then)"; field doc: "Addendum A4 teaching-note id (placeholder until the notes exist)". But the same file's test calls `note(rule.note_id).unwrap()` (warnings.rs:488, :517) and `the_rules_are_the_result_fields_in_order` pins `note_id == "a5.<id>"`; `explain/notes.rs` defines all six `a5.*` notes.
- Why it matters: a reader is told the ids may still change; they are pinned by tests and by the notes.
- Suggested action: reword both comments to say the id names the `a5.*` note in `explain::notes`.
- Fix class: mechanical-safe
- Confidence: certain

## MEX-03: assumptions.rs says the traceability test "belongs to plan A-3" and does not exist; it does
- Kind: inconsistent
- Where: magcoupling-rs/src/engine/assumptions.rs:24-26; docs/ai/03-structure.yaml:49 repeats it ("the dependency-graph traceability test is plan A-3's")
- Evidence: "The spec's traceability test (each assumption changes every dependent result and no independent one, by the equation registry's dependency graph) needs the A2 equation registry and belongs to plan A-3." The registry is built (explain/registry.rs) and `tests/explain.rs:5454 each_assumption_moves_what_depends_on_it_and_nothing_else` is that test.
- Why it matters: the comment sends a reader looking for work that is done.
- Suggested action: replace the paragraph with a pointer to `tests/explain.rs::each_assumption_moves_what_depends_on_it_and_nothing_else`; fix the YAML phrase with it.
- Fix class: mechanical-safe
- Confidence: certain

## MEX-04: `Registry::term_values` and `Registry::graph` are unused; `Registry::choices` and `RESULT_CHOICES` are used by tests only
- Kind: not needed
- Where: magcoupling-rs/src/engine/explain/registry.rs:38-40, :508-516 (`choices`), :551-556 (`graph`), :583-586 (`term_values`)
- Evidence: `git grep` for `term_values(`, `.graph()` finds no caller in src, tests, scripts or web (the `term_values` hits in tests/explain.rs:969 are an unrelated struct field); `choices(` is called only at tests/explain.rs:5595-5606; the GUI's mentions of `Registry::choices` (gui/typeset.rs:19) are doc comments. `build` already holds the same graph as a local map (registry.rs:415-418) rather than calling `graph()`.
- Why it matters: public API nothing uses is maintenance weight and misleads readers about what the GUI reads.
- Suggested action: delete `term_values` and `graph`; either delete `choices` with `RESULT_CHOICES` and its test or point the GUI at it if the label is meant to show.
- Fix class: needs-judgment
- Confidence: likely (a wasm or external user of the lib would settle it; none found)

## MEX-05: `scope::Status` and the `Pending` variant are scaffolding of a finished plan: no chain is, or can be, Pending
- Kind: not needed
- Where: magcoupling-rs/src/engine/explain/scope.rs:16-29 and the eight `status: Status::Explained` lines; magcoupling-rs/tests/explain.rs:1906, 1969, 2000, 2179, 5465
- Evidence: `Status::Pending` is never constructed anywhere (git grep finds only the definition); tests/explain.rs:5465 asserts `SCOPE.iter().all(|c| c.status == Status::Explained)`. The module doc still says "Each batch of plan A-3 writes one chain's records and flips its status to `Explained`" and "the dashboard, the geometry callouts and the five chains" while SCOPE holds eight `Chain` entries (six chains, the dashboard, the geometry callouts). The same doc says "193 paths" (scope.rs:14) while tests/explain.rs:1976-1979 pins the union at 194 (E24 added the outer ring onset): the doc was not updated.
- Why it matters: a status flag that can only have one value, plus a doc that miscounts the chains.
- Suggested action: drop `Status` and the field and the test filters (or keep and fix the doc); correct "five chains".
- Fix class: needs-judgment
- Confidence: certain for "never Pending"; the removal is a taste call


## MEX-06: `notes_before_snapshot` is a write-once generator that can only fail
- Kind: not needed
- Where: magcoupling-rs/src/engine/explain/notes.rs:2422-2433 (the `#[ignore]` test), :1404-1444 (`before::snapshot_text`)
- Evidence: `#[ignore = "writes docs/analyses/2026-10-14-magcoupler-notes-before.txt once, before the first note is rewritten (T4)"]`; its body starts `assert!(!path.exists(), "... never written again")` and then `std::fs::write(&path, before::snapshot_text())`. The committed file exists (it is `include_str!`-ed at :1415 and read by `the_before_snapshot_holds_the_seventeen_notes_slice_2_started_from`), so `cargo test -- --ignored` can only fail on the assert; if the file were ever deleted, it would write the CURRENT notes in place of the historical ones.
- Why it matters: a test that can never pass, and whose success path would falsify the record it names; T4 finished weeks ago.
- Suggested action: delete the test (keep `snapshot_text`/`parse` while `the_before_snapshot_format_reads_back` uses them).
- Fix class: needs-judgment
- Confidence: certain that it cannot pass; removal is the owner's call (it is the only ignored test in the engine's notes)

## MEX-07: The custom-eval escape hatch is dead code that a test insists stays dead
- Kind: not needed
- Where: magcoupling-rs/src/engine/explain/record.rs:15-37 (`CustomEval`, `Eval::Custom`, Debug arm) and :74-80 (`Record::custom`); magcoupling-rs/src/engine/explain/registry.rs:63-77 (`Restricted`), :209-214 (the "family record cannot have a custom eval" check), :638-651 (the `Eval::Custom` arm of `evaluate`)
- Evidence: `git grep -E "\.custom\(|Eval::Custom"` finds no record using it; tests/explain.rs:1274-1281 collects `Eval::Custom` equations and asserts the list is empty ("No custom evals in the tracer"). record.rs:23-27 documents it as an "escape hatch" the registry "counts".
- Why it matters: about 50 lines of branches, a wrapper type and an error check that no shipped record reaches; each is untested except through the assertion that they are never used.
- Suggested action: remove the variant, `custom`, `Restricted` and the two registry branches (and the test) until a record needs them; `Eval` then reduces to `Markup` (or the field goes).
- Fix class: needs-judgment
- Confidence: certain it is unused; whether to keep a reserved hook is the owner's choice

## MEX-08: `TableField.unit` and `TableField.label` are never read (37 rows each)
- Kind: not needed
- Where: magcoupling-rs/src/engine/explain/tables.rs:23-32 (fields), :36-77 (data)
- Evidence: only `.symbol` is read, at gui/typeset.rs:797, :1653 and render.rs:142; `tables::field(...)` in registry.rs:278 only tests `is_none()`. No src or test reads `.unit` or `.label` of a `TableField` (registry reads units from result/input metadata). Several labels also carry spreadsheet and decision ids ("decision 20", "Materials C38, C43", "E3:"), which the plain-words rule bans for text on screen, so showing them later would need a rewrite anyway.
- Why it matters: 74 hand-typed strings that nothing displays or checks can drift silently.
- Suggested action: drop `unit` and `label` from `TableField` and the table, or wire them to the term list and add a test.
- Fix class: needs-judgment
- Confidence: likely (a read of the whole worktree found no user; a macro or doc-driven reader would settle it)

## MEX-09: Markup functions `ceil`, `floor`, `cosh`, `tanh` are implemented, typeset and documented but no record uses them
- Kind: not needed
- Where: magcoupling-rs/src/engine/explain/markup.rs:51, :128-209 (`Func` variants, `from_name`, `name`); eval.rs:348, :356-357, :361-362; render.rs:126-127; typeset.rs (the matching arms); tests markup/eval/typeset
- Evidence: a count over records/*.rs gives 0 uses of `ceil(`, `floor(`, `cosh(`, `tanh(` (ceilto 3, floorto 1, sinh 3, cos 1 are used). The only callers are unit tests (eval.rs:441, gui/typeset.rs:1998).
- Why it matters: speculative grammar surface to keep in three layers (parser, evaluator, typesetter) and in the doc table.
- Suggested action: leave out until a record needs them, or keep and say so; low value either way.
- Fix class: owner-decision
- Confidence: certain they are unused by records; value of removal is small

## MEX-10: The 23 shaft joint paths are typed in three places (SCOPE chain, note `equations`, and the records file)
- Kind: inconsistent
- Where: magcoupling-rs/src/engine/explain/scope.rs:248-276 and magcoupling-rs/src/engine/explain/notes.rs:905-934 (the same 23 paths, in the same order, plus the one extra single-ring onset in the note); one record per path in records/shaft_joint.rs
- Evidence: the machine lead (17 identical lines scope.rs:258-274 vs notes.rs:916) is real. Drift is guarded: tests/explain.rs:2002 `assert_eq!(chain.paths, numeric.as_slice())` ties the chain to the result schema and notes.rs:1661 pins the note's rows, so an edit that misses one list fails a test.
- Why it matters: repetition the owner flags, though a test makes it safe; each new shaft joint result means three edits.
- Suggested action: build the note's list from the chain (a `const` slice shared by both) or accept the repetition and say so in a comment.
- Fix class: needs-judgment
- Confidence: certain it is duplicated; low value

## MEX-11: api.rs says JSON export "is M4"; the design file export exists
- Kind: inconsistent
- Where: magcoupling-rs/src/engine/api.rs:22-23 (and the test comment at :550 "996 at M2")
- Evidence: "`to_dict` is not ported: JSON export is M4." The GUI saves and loads design files through `gui::session::design_to_json` (used at magcoupling-rs/src/app.rs:198) and `Panel::load_design_file`. The row-count comment "// 996 at M2" gives a number from a milestone ago beside `assert!(rows.len() > 900)`.
- Why it matters: a stale milestone pointer in the module's main doc.
- Suggested action: say JSON export is `gui::session::design_to_json`; drop or refresh the "996 at M2" figure.
- Fix class: mechanical-safe
- Confidence: certain for the JSON sentence

## Leads settled
- `magcoupling-rs/src/engine/explain/scope.rs:258-274` and `notes.rs:916` (17 duplicated lines): real, the shaft joint path list typed twice; guarded by tests, low value (MEX-10).
- `magcoupling-rs` [dev-dependencies] `magcoupling-rs`: outside my files (Cargo.toml); not read, not settled here.
- Other leads of leads.md name files outside my part; none of the unused-item candidates (section 2) are in my files.

## Not read
- engine/explain/notes.rs: the 21 note bodies (lines ~158-1290) and most of the tests (1530-2425) were only searched, not read line by line.
- engine/explain/markup.rs: parser and Symbol (lines 230-1035) and its tests were not read.
- engine/explain/eval.rs 480-617 (tests), engine/explain/plain.rs 230-505 (helpers and tests).
- engine/explain/records/{torque,slip_heating}.rs, the bodies of records/demagnetization.rs (110-239) and records/shaft_joint.rs (120-230); engine/explain/symbols.rs 60-211; engine/explain/scope.rs lines 80-230 (path lists).
- engine/deviations.rs REGISTRY rows E1 to E20 (lines 330-1560) and the `restore` test module (1990-1996).
- engine/meta.rs (to line 1439 of its test module; the remaining few test lines), engine/api.rs, assumptions.rs, compat.rs, constants.rs, picks.rs, warnings.rs, mod.rs and explain/{mod,record,registry,probes,render,note_text,tables}.rs were read in full.

