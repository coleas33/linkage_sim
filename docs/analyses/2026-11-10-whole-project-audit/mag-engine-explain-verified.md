# Verified: mag-engine-explain

## MEX-01: Assumption helpers rebuild the defaults and the whole input schema on every call, and the GUI calls them every frame
- Verdict: CONFIRMED (structure; the cost per call is not measured, as the finder said)
- Checked: assumptions.rs:213-247 (`states` builds `DesignInputs::default()` and `input_rows(inputs)` for 18 unit lookups; `modified` builds `DesignInputs::default()`); registry.rs:659-690 (`term_style` -> `modified_assumptions_upstream` -> `assumptions::modified(inputs)` at :682, while `self.defaults` is held and used at :662); gui/explorer.rs:1070-1072 (`term_style` per term row inside the draw loop); callers by search: panel.rs:1449, :1725, spreadsheet.rs:547, explorer.rs:1071, :1596. meta.rs:701-711 `input_rows` pushes a `String` path per leaf. registry.rs:9 doc claim read.
- Safe to apply without judgment: no, the fix changes signatures and caching (needs-judgment, as the finder said).

## MEX-02: Stale "placeholder" and "until then" comments in warnings.rs: the teaching notes exist
- Verdict: CONFIRMED
- Checked: warnings.rs:2-3 and :30 read; the rules carry `note_id: "a5.<id>"` (:60-90); the test at :218 pins `rule.note_id == format!("a5.{}", rule.id)` and :488/:517 call `note(...)`. `include_str!` search finds no reader of warnings.rs text. Search "placeholder|until then" in warnings.rs: 2 hits, both the stale ones.
- Safe to apply without judgment: yes (two doc comments, no code)
- Exact change:
  - magcoupling-rs/src/engine/warnings.rs line 3: old `//! notes are Addendum A4's; the ids are placeholders until then).` new `//! notes are Addendum A4's; each id names the `a5.*` note in `explain::notes`).`
  - magcoupling-rs/src/engine/warnings.rs line 30: old `    /// Addendum A4 teaching-note id (placeholder until the notes exist).` new `    /// Addendum A4 teaching-note id: the `a5.*` note in `explain::notes`.`

## MEX-03: assumptions.rs says the traceability test "belongs to plan A-3" and does not exist; it does
- Verdict: CONFIRMED
- Checked: assumptions.rs:24-26 read; tests/explain.rs:5454 `each_assumption_moves_what_depends_on_it_and_nothing_else` exists; docs/ai/03-structure.yaml:49 has the phrase. Two more copies of the stale sentence the finder did not list: magcoupling-rs/tests/assumptions.rs:6 ("belongs to plan A-3") and :268 ("the dependency-graph version is plan A-3's"). tests/explain.rs:1780 does `include_str!` of docs/ai/03-structure.yaml (it checks counts), so a YAML edit is read by a test; the wording of that phrase is not one it checks, but I did not read every claim.
- Safe to apply without judgment: yes for the assumptions.rs comment only; no for the YAML and tests/assumptions.rs (a test reads the YAML; keep those for the owner).
- Exact change:
  - magcoupling-rs/src/engine/assumptions.rs lines 24-26: old
    `//! The spec's traceability test (each assumption changes every dependent result and no`
    `//! independent one, by the equation registry's dependency graph) needs the A2 equation`
    `//! registry and belongs to plan A-3.`
    new
    `//! The spec's traceability test (each assumption changes every dependent result and no`
    `//! independent one, by the equation registry's dependency graph) is`
    `//! `tests/explain.rs::each_assumption_moves_what_depends_on_it_and_nothing_else`.`

## MEX-04: `Registry::term_values` and `Registry::graph` are unused; `Registry::choices` and `RESULT_CHOICES` are used by tests only
- Verdict: PARTLY. `term_values` (registry.rs:584) and `graph` (:552) have no caller anywhere (search over rs, js, html, sh, py, toml: only the definition; the `term_values` hits in tests/explain.rs are an unrelated struct field). The `choices` / `RESULT_CHOICES` half is REFUTED: `Registry::choices` is called by shipped GUI code at gui/typeset.rs:745 (`self.registry.choices(&r.path)`), :1342 and :1501 (and the explorer.rs:584 hit is `.choices(path)` on the same registry), and `RESULT_CHOICES` feeds `choices` (registry.rs:195, :511) and the build check. The finder read the typeset.rs mentions as doc comments; they are code.
- Checked: registry.rs:548-590, typeset.rs:743-746 and :1339-1343, searches above.
- Safe to apply without judgment: no (deleting public registry API that docs/ai/03-structure.yaml:64 lists by name, `graph` among them; owner's call).

## MEX-05: `scope::Status` and the `Pending` variant are scaffolding of a finished plan: no chain is, or can be, Pending
- Verdict: CONFIRMED
- Checked: scope.rs:1-29; `Status::Pending` / `Pending` search over rs, js, html, py, sh: the definition (scope.rs:20) only; SCOPE has 8 `Chain` entries (18 matches = struct + 8 chains + 8 status lines + the const); tests/explain.rs:1906, 1969, 2000, 2179, 5465 read the field; the doc says "five chains" (:2) and "193 paths" (:14) while tests/explain.rs:1976-1979 pins 194.
- Safe to apply without judgment: no (the removal edits tests; the doc count fix alone is a taste call the finder marked as such).

## MEX-06: `notes_before_snapshot` is a write-once generator that can only fail
- Verdict: CONFIRMED that it cannot pass; the finder's second point is PARTLY wrong. notes.rs:2422-2433 asserts `!path.exists()` and the committed file exists, so a `--ignored` run only fails. But the "if the file were deleted it would write the CURRENT notes" case cannot occur: `before::SNAPSHOT` (notes.rs:1414) is an `include_str!` of that file, so the test module would not compile without it.
- Checked: notes.rs:1404-1444, :2422-2439; tests/explain.rs:1716-1765 (`only_the_review_tools_are_ignored` lists `notes_before_snapshot` in TOOLS, an allow-list: removing the test does not fail it, but the list would then name a test that does not exist); docs mentions: docs/ai/05-update-tracker.md:56, 03-structure.yaml:64, README.md:714 and several plans.
- Safe to apply without judgment: no (the owner's call; docs name it).

## MEX-07: The custom-eval escape hatch is dead code that a test insists stays dead
- Verdict: CONFIRMED
- Checked: search for `Eval::Custom|.custom(|CustomEval|Restricted` over rs, js, html, py, sh: only record.rs:16-34, :74-77, registry.rs:64-77, :209, :640-641 and tests/explain.rs:1278; no record calls `.custom(`; tests/explain.rs:1274-1281 asserts the list is empty.
- Safe to apply without judgment: no (removes a variant and edits a test; owner's choice to keep a reserved hook).

## MEX-08: `TableField.unit` and `TableField.label` are never read (37 rows each)
- Verdict: CONFIRMED, with a miscount: TABLE_FIELDS has 40 rows (tables.rs:37-76), so 80 strings, not 74.
- Checked: search for `TABLE_FIELDS`, `tables::field`, `field(` over the worktree: readers are typeset.rs:797 and :1653 (`f.symbol`), render.rs:142-143 (`f.symbol`), registry.rs:278 (`is_none()`), tables.rs:80-90 (`.table`, `.field`) and tables.rs:209-216 (test reads `f.table`, `f.field`). Nothing reads `.unit` or `.label`.
- Safe to apply without judgment: no (data deletion across 40 rows; the finder offered wiring them instead).

## MEX-09: Markup functions `ceil`, `floor`, `cosh`, `tanh` are implemented, typeset and documented but no record uses them
- Verdict: CONFIRMED
- Checked: search for `ceil(|floor(|cosh(|tanh(` in magcoupling-rs/src: hits in records/*.rs: none; the only formula-grammar users are eval.rs:441 and gui/typeset.rs:1998 (tests). `Func::Ceil|Floor|Cosh|Tanh` arms read at markup.rs:158-192, eval.rs:348-362, render.rs:126-127, typeset.rs:651-652. Records could still spell them with spacing the pattern misses; I did not read every record body, only searched.
- Safe to apply without judgment: no (grammar surface; owner decision).

## MEX-10: The 23 shaft joint paths are typed in three places (SCOPE chain, note `equations`, and the records file)
- Verdict: CONFIRMED
- Checked: scope.rs:251-275 (23 paths) and notes.rs:909-934 (the same 23 in the same order plus `temperature.demag.inner_onset_single_ring_C`); records/shaft_joint.rs holds one record per path; tests/explain.rs:2002 `assert_eq!(chain.paths, numeric.as_slice())` guards the chain. notes.rs:1661 pin not opened.
- Safe to apply without judgment: no (refactor; the finder rates it low value).

## MEX-11: api.rs says JSON export "is M4"; the design file export exists
- Verdict: CONFIRMED for the JSON sentence; the "996 at M2" half is a stale figure beside `rows.len() > 900`, harmless and not worth a change.
- Checked: api.rs:23 read; `design_to_json` is `pub fn` at gui/session.rs:210, used at app.rs:198, gui/panel.rs:1256; gui/results_table.rs:364 also has a JSON export of the results.
- Safe to apply without judgment: yes (doc comment; gui is a cargo feature, so no intra-doc link, plain backticks)
- Exact change: magcoupling-rs/src/engine/api.rs line 23: old `//! bypass `set`. `to_dict` is not ported: JSON export is M4.` new `//! bypass `set`. `to_dict` is not ported: design files are `gui::session::design_to_json`.`

## Seen while checking
- None.
