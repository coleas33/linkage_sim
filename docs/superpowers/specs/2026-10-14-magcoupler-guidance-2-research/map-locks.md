# Slice 2 locks map: what pins or locks text in this repo

Scope: magcoupler calculator (`magcoupling-rs`), slice 2 = (A) rewrite the 17 teaching notes and add new
ones, (B) fill the 87 empty input helps, (C) remove on-screen jargon. Read-only survey, 2026-10-14.
Repo root `R` = `C:/Users/Cole/source/repos/linkage_simulation`; `M` = `R/magcoupling-rs`.

Legend. HARD = a test or the gate goes red. PROCESS = a written rule a reviewer or the owner will hold
you to. FREE = checked, nothing pins it. Every row ends with what it means for slice 2.

---------------------------------------------------------------------------------------------------
## 0. The short version

1. The input helps (B) are the big lock. All 87 empty helps are on ported inputs (87 of 161 ported,
   0 of 11 Rust-only). Ported label, unit, help, cell and choices are compared byte for byte with the
   vendored Python oracle. Fill them on the GUI side (an overlay keyed by input path); do not touch
   `engine` metadata, `python_schema.json`, `input_schema.json` or the oracle.
2. A teaching note is hidden until `Review::Reviewed`; any text edit is expected to go back to Draft and
   get a fresh physics review (session model). Hidden notes break four existing GUI tests, so the
   rewrite must land Draft -> reviewed within the slice, and `release_notes_are_reviewed` (currently
   `#[ignore]`) should be un-ignored by this slice, as 04-memory.yaml:82 says.
3. `NOTES.len()` must be 15..=20 (notes.rs:631-633). 17 + the audit's four candidates = 21 fails.
   Pick three, fold one into an existing note, or amend that test and spec A4 on purpose.
4. Each note is 2..=6 sentences and cites at least one source. `sources` is drawn on screen and 15 of 17
   notes carry banned jargon in it today. Sentences, watch-outs and titles are free text otherwise.
5. The slice-1 plain-words guard (`plain_text_violations`, 6 patterns) is only run over guidance text.
   It is `pub`, but lives behind the `gui` feature; the notes live in the engine. Notes and help are
   not checked by any plain-words test today (the glyph tests do cover notes and input tooltips).
6. Many "jargon" strings are pinned by tests by exact text (tooltips, `Rust-only ...`, `E3 E7 E8`,
   `Calculator!C93`, `Model assumption (Addendum A3)`, `Corrected vs workbook:`), and assumption
   `source` text is required to contain cell refs and "Addendum A3" (tests/assumptions.rs:123-127).
   Each is a deliberate test edit, not a free rewrite.
7. Nothing is snapshot or golden for human text. The only text pinned from outside Rust is the
   welcome line in `.claude/workflows/gui-smoke.js:83` and the console log prefixes it greps.
8. Push to main deploys with no test step (deploy-web.yml); the gate and a gui-smoke run are the only
   safety net. Nothing is pushed without the owner's go (AGENTS.md:26).

---------------------------------------------------------------------------------------------------
## 1. Process rules (PROCESS)

| # | Rule | Where | What it means for slice 2 |
|---|---|---|---|
| P1 | Plan-review protocol: before any code, ask BIG vs SMALL change; per issue give numbered issues, lettered options, recommended option first, use AskUserQuestion, one section at a time | `R/CLAUDE.md:3-67` | Slice 2 needs its own spec/plan pass with the owner; options must be numbered/lettered, recommended first. |
| P2 | Behavioural rules: surgical changes, simplicity, red test before fix, define verifiable goals | `R/CLAUDE.md:71-131` | Every text change needs a test that fails first (AGENTS.md:23 says the same). Do not touch adjacent text. |
| P3 | Model selection: physics, design, planning and final reviews on the session model; mechanical doc/YAML edits, mapping, finders on `sonnet`; never downgrade a retry | `R/CLAUDE.md:133-154` | Note rewrite and the physics review run on the session model (omit `model`, comment why). Help-string transcription and docs edits may run on sonnet. |
| P4 | Read `docs/ai/*.yaml` before any job; update them after file-modifying work; do not edit them on read-only work | `R/docs/ai/01-meta.yaml:17-25`, memory `feedback_docs_ai_workflow.md` | This map is read-only so docs/ai was not touched. The build must update 02/03/04/05 (see section 9). |
| P5 | Every code change ships with doc changes in the same change | memory `feedback_docs_with_code.md`, AGENTS.md:23 | README tables and docs/ai edits are part of each task, not a follow-up. |
| P6 | G-3: every new physics sentence gets one fresh-context session-model physics review per slice; numbers in hints come from results; "no Draft/Reviewed gate for hints" | spec `R/docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md:36`; `02-system.yaml:107` | Applies to help strings and any new hint text. It does NOT lift the Draft/Reviewed gate on teaching notes (that gate is code, section 2). Slice 2 should record a G-style decision for itself. |
| P7 | Spec section 8 physics traps must not be written (radius strongest lever, fewer poles keep size or always improve torque per mass, pull-out guaranteed, calibration beyond prototype designs, 416 stainless safe back-iron fix, stored 3D fields follow geometry/grade) | spec `...guidance-design.md:197-204` | The rewrite and new notes (hot-low/cold-high, back-iron wall rule, after slip) must avoid these; the back-iron wall note is exactly where the 416 trap sits. |
| P8 | `risk: physics` backlog items need a physics reviewer pass AND the owner's explicit review of the diff before merge; not `fixed` until then. fbd-math-reviewer is linkage-specific; magcoupling uses a physics reviewer prompt naming the M1 derivations | `R/AGENTS.md:22`; `R/docs/ai/backlog.yaml:1-5`; calculator spec `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md:262-266` | Open a BL entry (next after BL-055) with `risk: physics`. Review prompt must cite the M1 derivations a note's `sources` name. |
| P9 | Run the gate as `MAGCOUPLING_PYTHON=<python with oracle deps> bash linkage-sim-rs/scripts/gate.sh`; it must end `GATE PASS` | `R/AGENTS.md:24` | See section 8 for steps and runtime. |
| P10 | Nothing is pushed without the user's go; pushing `main` deploys. Memory: re-confirm deploy authorization for later slices (slice 1's G-8 was for slice 1 only) | `R/AGENTS.md:26`; memory `project_magcoupler_guidance_overhaul.md`; `01-meta.yaml:12` | Ask for fresh authorization before merge/push. |
| P11 | Rustdoc: indented lines in `///`/`//!` become doctests; fence ASCII math as ```text | `02-system.yaml:407-408` | Applies to any doc comment written in notes.rs, the new help module, guidance.rs. `cargo test` runs doctests (gate 4). |
| P12 | Windows MAX_PATH: a release build under a deep directory fails with LNK1104; build from a short path or `subst` | `02-system.yaml:451` | Repo path is already long (`C:\Users\Cole\source\repos\linkage_simulation\magcoupling-rs`); use a short worktree for release/web builds. |
| P13 | Egui glyphs: default fonts lack U+2192 and U+2264; write `->` and `<=`. `glyph_safe` maps only U+03D1 -> theta, U+207B -> macron, U+221D -> `~` | `02-system.yaml:444`; `M/src/gui/typeset.rs:271-278` | Any other exotic character in a note or help string fails a glyph test (section 4). |
| P14 | Symbol edits: a symbol a note quotes changes with the note, which sends it to Draft and needs a new physics review. Open item M43-13: `thermal_time_constant` writes tau/theta_start where records show tau_th/theta_hot | `04-memory.yaml:74,83` | If the rewrite aligns symbols with the records (good for plain reading), that is a text edit under the same gate. |

---------------------------------------------------------------------------------------------------
## 2. Teaching notes (A): the accuracy gate and everything that pins a note

### 2.1 The gate itself (HARD in tests, PROCESS for who reviews)

| # | Rule | Where | What it means for slice 2 |
|---|---|---|---|
| N1 | A note shows only when `Review::Reviewed {reviewer, date, record}`; `Draft` is hidden. `note_for` and `explorer::reviewed_note` both filter on it. | `M/src/engine/explain/notes.rs:30-41, 583-586`; `M/src/gui/explorer.rs:110-113` | A rewritten note left Draft disappears from the Equation panel, the dashboard Why links and the warning links. |
| N2 | Nothing in code says who may flip a note to Reviewed (a struct literal). Convention in the 17 records: "physics reviewer (session model)" or "independent physics review 2026-10-01", `record` = a commit or report section. | `notes.rs:87-91, 110-113` etc.; precedent flow in `05-update-tracker.md:340-364` | Process: edit as Draft -> fresh-context session-model physics review -> apply fixes -> flip with a `record` naming the review. P3/P6/P8 say the reviewer is the session model, never sonnet. |
| N3 | The `record` should name a COMMITTED artifact. `.superpowers/` is gitignored (`R/.gitignore:56`), so the slice-1 `physics-fixes.md` records are not in the repo. | `.gitignore:56`; `04-memory.yaml:7` | Put the review report in `docs/analyses/` (as `2026-10-06-open-physics-review-response.md` precedent) or cite a commit hash. A record cannot name its own commit; flip in a follow-up commit or cite the review commit. |
| N4 | Editing a Reviewed note, or a symbol it quotes, "sends it back to Draft and needs a new physics review". | `04-memory.yaml:74,83`; `notes.rs:1-10` | All 17 rewrites are a full re-review, not a spot check. Budget it. |
| N5 | `release_notes_are_reviewed` (every START_HERE note and every warning note is Reviewed) exists but is `#[ignore]`; it "stays ignore only because M4-3 edits no engine test, so un-ignoring it is the next engine change's step". | `M/tests/explain.rs:1033-1055`; `04-memory.yaml:82` | Slice 2 edits `notes.rs` (engine) so it is that "next engine change": remove the `#[ignore]` (gate 4 and 7 currently show "2 ignored" in tests/explain.rs). Decide whether new notes join the check. |
| N6 | Spec A4: notes are "2-6 plain-language sentences at Physics 2 level", "about 15-20 notes", optional watch-out, optional diagram, "checked by a physics reviewer before release" | `R/docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md:352-371` | The owner now wants non-specialist engineers and "say what to change": that is a spec amendment (new decision), not just a rewrite. |

### 2.2 Structure tests on `NOTES` (HARD)

| # | Rule | Where | What it means for slice 2 |
|---|---|---|---|
| N7 | `NOTES.len()` in `15..=20` ("spec A4: about 15 to 20 notes") | `notes.rs:629-633` | 17 + 4 candidates = 21 FAILS. 3 new notes = 20 passes with no headroom. Fold "back-iron wall rule" into the existing `a5.low_saturation` rewrite (it already owns `model.backiron_needed_mm`), or raise the bound together with the spec. |
| N8 | Each note has `sentences.len()` in `2..=6` | `notes.rs:634-639` | A "what to change" block cannot be added as extra sentences beyond 6. `Note` has no field for it; adding one (e.g. `change: &[&str]`) means editing all 17 struct literals plus new ones plus `note_ui` (`explorer.rs:571-599`). No test or code outside notes.rs constructs a `Note` (checked by grep), so the blast radius is notes.rs + explorer.rs. |
| N9 | `sources` non-empty for every note ("a note cites the M1 derivations it is drafted from") | `notes.rs:640-644` | New notes need a non-empty `sources`. See N10 for what is drawn. |
| N10 | `sources` is DRAWN on screen as `Sources: ...` (small weak label) through `glyph_safe` | `explorer.rs:588-598` (const `SOURCES` at 104) | Jargon today: 15 of 17 notes' `sources` match a banned pattern (cell refs, `E\d`, "Addendum", "workbook"), e.g. "Calculator C69, C70, C94; Metal design C8", "Addendum A decision 29", "Spec A5 warning rules", "M1 audit M3". No test pins the drawn "Sources:" line, so you may plain-word it, move provenance to a non-drawn field, or stop drawing it; the field itself must stay non-empty (N9). |
| N11 | Ids unique; every warning rule's `note_id` resolves; every START_HERE `(id, path)` has a note that covers that path | `notes.rs:603-622` | Do not rename or drop an id that a warning, a guide, a ROW_LEARN row or START_HERE uses (N13-N16). |
| N12 | Warning rule ids tie to note ids: `rule.note_id == format!("a5.{}", rule.id)` | `M/src/engine/warnings.rs:215` (ids at 60-90) | The six `a5.*` note ids are frozen. Titles may change (see N15). |
| N13 | Equation links: every `equations` entry covers at least one real result at the default design, each such result has an equation record, and each result path has AT MOST ONE note | `M/tests/explain.rs:1173-1205` | A new note may not claim a path an existing note owns. Already owned: `model.pullout_Nm` (pullout_angle), `model.tau_Pa`/`b_i#`/`amp#_Pa` (harmonics), `metal.torque_cold_Nm` and `model.pullout_20C_Nm` (br_temperature), `model.backiron_needed_mm` and `model.bsat_T` (a5.low_saturation), `temperature.slip_loss.*` and `drag_Nm` (slip_heating), `temperature.thermal.*` (thermal_time_constant), `temperature.demag.*` (demagnetization, ferrite_cold_demag). Not owned (verified by grep in notes.rs): `metal.torque_hot_low_Nm`, `metal.torque_cold_high_Nm`, `model.cup_od_mm`, `mass.total_g`, `model.gearbox_input_ripple_Nm`. A note with `equations: &[]` is legal (three a5 notes already do). "After the coupling slips" will have to be `&[]` or use unowned paths. |
| N14 | START_HERE: `results.contains(path)`; if the path's chain is "Explained", a record must exist. The bar text is `Start here N of LEN: <glyph_safe title>`. | `tests/explain.rs:1206-1217`; `M/src/gui/explorer.rs:114-123` | Appending new steps at the END is safe. Inserting at index 0 breaks the hard-coded title test (N17). |
| N15 | `a_draft_is_hidden_and_every_note_is_complete` also asserts `note_for_any_status("model.f_end")` is `end_effect` | `notes.rs:625-629` | Keep the `end_effect` note owning `model.f_end`. |

### 2.3 GUI tests that pin note text or ids (HARD)

| # | Rule | Where | What it means for slice 2 |
|---|---|---|---|
| N16 | `count(output, note.sentences[0]) == 1` against the RAW string, while `note_ui` draws `glyph_safe(sentence)`. Run for `pullout_angle`, for `WARNING_RULES[0]`'s note (`a5.non_ferromagnetic_back_iron`), and for every `Learn::Note` in `CHECK_GUIDES` (`a5.low_saturation`, `demagnetization`, `clamp_preload`, `end_effect`). | `M/src/gui/panel.rs:6998, 7265, 7505` | `sentences[0]` of those notes must contain none of U+03D1, U+207B, U+221D (else the raw string is never drawn) and must be one drawn label (no manual line breaks). Keep sentence 0 plain ASCII-ish. |
| N17 | Hard-coded title: `"{START_HERE_BUTTON} 1 of {}: Harmonic decomposition"` | `M/src/gui/explorer.rs:1084` (assert at 1083-1086) | Renaming the `harmonics` note title edits this test. |
| N18 | Titles are used as click targets: `click_text("{WHY}: {glyph_safe(title)}")` and `click_text(glyph_safe(title))`; `count(title)` must be >= 1 on the dashboard for `pullout_angle` and `end_effect` and grows by exactly 1 when Explain is on | `panel.rs:6978-7005, 7255-7260, 7493-7500` | Titles must stay unique across notes and must not collide with any other drawn label (a collision makes `click_text` hit the wrong widget). Dashboard Why links draw the TITLE, so long titles widen the dashboard (it already scrolls at 1280x800). |
| N19 | `a_note_by_id_shows_only_once_reviewed`, `every_start_here_step_opens_an_equation_with_its_reviewed_note` (START_HERE path -> `note_for(path)` is that id) | `explorer.rs:1090-1109` | A new note whose `equations` takes over a START_HERE path breaks this. |
| N20 | Diagram points: the torque-angle diagram must paint >= 1 filled circle of radius 3.5; the flux-path diagram must paint none (the warning test) | `panel.rs:6965-6970, 6999, 7266-7269` | Reusing or changing a `Diagram` variant's drawing is pinned by these point counts. A new diagram needs a new `Diagram` variant and an arm in `gui/diagrams.rs::diagram_shapes`; notes.rs docs list the 6 variants. |
| N21 | Glyph guard over every note (title, sentences, watch-out, sources through `glyph_safe`) in a 1280x3000 harness | `panel.rs:7064-7079` (test `every_teaching_note_shows_with_glyphs_in_the_default_fonts`); also every START_HERE bar `7040-7062` | Already covers new/rewritten notes automatically (iterates `NOTES`). Taller notes still fit the 3000 pt harness. Characters not in egui's default fonts fail here, with the offending note id in the message. |

### 2.4 Learn links from the dashboard (HARD)

| # | Rule | Where | What it means for slice 2 |
|---|---|---|---|
| N22 | Every `Learn::Note(id)` in `CHECK_GUIDES`/`ROW_LEARN` must resolve to a REVIEWED note | `M/src/gui/guidance.rs:2363-2379` (`every_learn_link_resolves`) | A guide cannot link a new note until that note is Reviewed. Currently linked: `a5.low_saturation` (188), `demagnetization` (197), `clamp_preload` (209), `end_effect` (275), `pullout_angle` (305, 318), `br_temperature` (306). |
| N23 | For every DASHBOARD value row whose path a reviewed note covers, the row's Why link MUST equal `Learn::Note(that note)` | `guidance.rs:2381-2400` (`every_dashboard_row_has_a_why_link`); `ROW_LEARN` at `guidance.rs:302-331` | If a new note claims `metal.torque_cold_high_Nm` (now `Learn::Assumption("production_variation")`, line 313) or `model.cup_od_mm` / `mass.total_g` (now `Learn::None(..)`, 322/326), you must update ROW_LEARN in the same change or this test fails. Note `metal.torque_hot_low_Nm` is a check row: it leads where `HOT_MIN_GUIDE.learn` leads (line 164, an assumption), so a hot-low/cold-high note is wired by editing that guide's `learn`. |
| N24 | Dashboard Why text for a hidden note is `NOTE_NOT_REVIEWED` ("Its teaching note has not passed review."); `NO_NOTE_YET` otherwise | `M/src/gui/dashboard.rs:106-110, 745-760`; tests `dashboard.rs:1920-1955` | That is what users see if a rewritten note is left Draft. |
| N25 | `reviewed_note("end_effect")` is used by name in a dashboard test | `dashboard.rs:1925` | Keep `end_effect` Reviewed. |

### 2.5 What is NOT pinned (FREE)

- The wording of every sentence and watch-out other than sentences[0] constraints (N16) and the length bounds (N8).
- Numbers inside notes. They are literals about the default design ("2.69 to 1.54 N m", "0.95 to 1.049", "1.70 N m", "863 kA/m", "336 C", "1.3 mm"); no test ties them to the engine (grep over src/tests found only an unrelated comment at `tests/deviations.rs:1854`). G-3's "numbers from results" applies to guidance, not notes. A slice-2 drift test for the quoted numbers would be new work.
- The `sources` wording (N10), `diagram` choice, `equations` lists (within N13).

---------------------------------------------------------------------------------------------------
## 3. Input help (B): the python boundary

All paths below are under `M/`. Counts verified from `tests/data/input_schema.json` (Rust export) and
`tests/data/python_schema.json`: 172 Rust inputs, 11 Rust-only, 161 ported; 87 empty helps, all 87 on
ported inputs by group: metal 31, coupling 15, clamps 15, temperature 12, calibration 8, materials 6;
4 of the 87 are assumptions. The Python oracle has 89 empty input helps; the 2 difference is two inputs a
registered correction already gave help, each through `workbook_help: [(path, "")]` in `src/engine/deviations.rs`:
E1 `temperature.mismatch.adhesive_shear_modulus_GPa` (line 288) and E20 `temperature.demag.beta_hcj_per_C` (line 1298).
That is the only precedent for engine-side help on a ported input, and both ride on a value-changing correction.

| # | Rule | Where | What it means for slice 2 |
|---|---|---|---|
| H1 | Every ported input's label, unit, help, cell, choices and default equal Python's (`compare`, `choices`, defaults). A reworded help is allowed only when an APPLIED deviation lists it in `workbook_help` (recording the workbook text) | `tests/python_schema.rs:1-14, 51-136` (test `ported_inputs_carry_the_python_metadata_and_defaults` at 94-136); `tests/common/mod.rs:189-198`; `src/engine/deviations.rs:218-225` | Setting `help` on any of the 87 in `engine` FAILS this test. The only sanctioned engine route is a deviation, and a deviation is a physics correction that needs an audit-report row (H4). Wrong tool for pure help text. |
| H2 | `tests/deviations.rs::reworded_help_is_recorded_for_real_fields` requires each `workbook_help` path to be a real field whose port help differs from the recorded text | `tests/deviations.rs:2004-2030`; README `745-770` (Deviations rule) | Confirms H1: help changes are modelled as corrections of the workbook. |
| H3 | `tests/data/input_schema.json` includes `help` and must equal the live metadata; refresh with `MAGCOUPLING_BLESS=1 cargo test --test schema` | `tests/schema.rs:31-78` | Any engine help edit (even on the 11 Rust-only inputs) forces a re-bless of this file. `reference/magcoupling-py/tools/gen_differential.py` reads only paths, types, ranges, choices and defaults from it (`gen_differential.py:402-412`), so help edits alone do not make `--check` stale. |
| H4 | Deviation registry entries must be approved rows of the audit report; the oracle is read-only | `README.md:753-770`; `docs/ai/03-structure.yaml:12-13` ("engine: magcoupling/ (READ-ONLY, never edit"); `02-system.yaml:88,89` | You cannot add a registry entry for wording, and you cannot edit `reference/magcoupling-py` (parity and differential data come from the unchanged oracle). |
| H5 | The 85 non-empty ported helps are locked the same way ("Same as the measured prototype.", "Even; one block per pole." at `src/engine/model.rs:158-160`, ...) | H1 | The owner's complaint that `npole` help never says it leaves the radius alone cannot be fixed in the engine string either. An overlay must be able to ADD to a non-empty workbook help, not only fill empties. |
| H6 | `help` is read in exactly three places: `gui/inputs.rs:910` (`input_tooltip`), `gui/size_controls.rs:322` (Size-tab labels, `on_hover_text(entry.meta.help)`), `gui/dashboard.rs:409` (`result_tooltip`). No other consumer (checked by grep over src). Exports (CSV/JSON/xlsx) do not carry help. | grep result | A GUI overlay changes three call sites. Zero engine, data-file, or export changes. |
| H7 | `tooltips_carry_help_path_cell_range_and_default` pins the FULL tooltip for `metal.face_gap_mm`: `"Same as the measured prototype.\nmetal.face_gap_mm\nMetal design!C119\nSlider 0.3 to 5 mm, step 0.01\nDefault: 1.400"`, plus `Default: ...` tails and the two jargon lines for `coupling.max_harmonic` | `M/src/gui/inputs.rs:1380-1406` | Changing the tooltip layout (help first, then path, cell, slider, default) is a deliberate edit of this test. Appending plain help after the workbook help line changes this exact string. |
| H8 | Glyph guard over EVERY input tooltip (so every new help string is checked) | `panel.rs:4017` in `every_text_the_panel_shows_has_glyphs_in_the_default_fonts` (3946-4048) | New help strings get glyph coverage for free if they flow through `input_tooltip`. A separate overlay shown elsewhere must be added to that test. |
| H9 | No test requires non-empty help (audit: "No test requires non-empty help"; `tests/schema.rs` only checks label/path/cell well-formedness at 206-263) | `tests/schema.rs:206` | Add the missing guard: every input has help from the engine OR the overlay; no overlay key names a non-input path. |
| H10 | No plain-words test covers help or tooltips; `input_tooltip` itself prints `path`, the workbook cell, `Rust-only input (no workbook cell)` and `Model assumption (Addendum A3)` | `inputs.rs:907-940` | A plain-words guard over help must test the help text only (or the whole tooltip only after the jargon lines are removed, see section 5). |

Also keep in mind for results: Python has 305 of 331 results with empty help; the dashboard tooltip
(`dashboard.rs:404-420`) uses the same locked source. Not in scope for the 87 but same lock.

---------------------------------------------------------------------------------------------------
## 4. Existing guards (slice 1) and what they cover

| # | Guard | Where | Covers | Reusable for notes/help? |
|---|---|---|---|---|
| G1 | `BANNED: [&str; 6]` = `C\d{2,3}`, `E\d{1,2}`, `workbook`, `Addendum`, `Rust-only`, `Calculator!`; `plain_text_violations(text) -> Vec<&'static str>`; allowed phrase `Workbook example` | `M/src/gui/guidance.rs:1437-1494` (`ALLOWED_WORKBOOK` 1452); unit test of the patterns `3286-3324` | Pure `pub fn`, no egui. | YES, `pub mod guidance` (`gui/mod.rs:17`). But `gui` is an optional feature (`Cargo.toml` features) and `notes.rs` is engine, so an engine-only test (gate 4) cannot call it. Put the notes/help plain-words test in a `gui` test (e.g. `explorer.rs` or `guidance.rs` tests, run by gate 7 `--features app`), or move `BANNED`/`plain_text_violations` to a pure-std engine module (DRY; engine stays pure std per `02-system.yaml:95`). |
| G2 | Hard-coded "clean" list includes `"ISO 4762 M4 x 14, class 12.9"`, `"E123"`, `"C1"` | `guidance.rs:3311-3323` | Shows what `plain_text_violations` must NOT flag. | A broad `M\d` ban for "M3"/"M1 audit" would flag the screw name M4 and fail this test. Add targeted patterns instead ("M1 audit", "until M3", "drawing.py", "Spec A", "decision"), and extend `BANNED` plus its length-6 array, the zip, and the unit test together. |
| G3 | `every_text()` (private test helper) = guide labels, Whys, reasons, fix texts, fallbacks, Problems, trade-off notes, status lines for 8 designs | `guidance.rs:2239-2269`; used by `no_guide_text_carries_workbook_jargon` (3326) and `every_guide_text_has_glyphs_in_the_default_fonts` (3334) | Guidance only. | NOT reusable as is (private to guidance tests, guidance texts only). Write `every_note_text()` and `every_help_text()` beside it. |
| G4 | Dashboard whole-guide check: plain words over Whys, reasons and fallbacks; glyphs over those plus assumption rationales (rationales excluded from plain words on purpose: "An assumption's rationale is the Assumptions view's own text, which may cite cells: glyphs only") | `M/src/gui/dashboard.rs:2062-2160` | Guide texts. | The explicit exemption for assumption rationales is what slice 2's jargon removal collides with (A1 below). |
| G5 | Consequence lines checked for banned patterns | `guidance.rs:4515, 4557` | Size-tab lines. | Done in slice 1. |
| G6 | `assert_glyphs(ctx, text, what)` (egui default fonts at proportional 14, `has_glyph`) and `drawn_texts(output)`; `pub(crate)` in a `#[cfg(test)]` module | `M/src/gui/test_support.rs:151-192`; `gui/mod.rs:31-32` | Any text drawn or passed. | YES for any gui test. Panel-wide: `every_text_the_panel_shows_has_glyphs_in_the_default_fonts` (panel.rs:3946) covers drawn text in every view and group, every input tooltip, warning-rule texts, dashboard tooltips and row tooltips; `every_teaching_note_shows_with_glyphs...` (7064) covers notes. |
| G7 | Typesetter glyph rules: `glyph_safe` + "a test lays out every equation and checks every character, another every teaching note" | `02-system.yaml:114`; `typeset.rs:271-278` | Equations and notes. | Already applies to the rewrite. |

Gap summary: no automated plain-words check exists for notes (title, sentences, watch-out, sources),
input help, input/result tooltips, the Assumptions view, geometry notes, explorer strings, warning
texts or the results table. Today's notes would fail it as follows (counted from `notes.rs`):

| Banned hit | Notes (sentences) | Notes (watch_out) | Notes (sources) |
|---|---|---|---|
| `workbook` | harmonics, calibration, a5.low_saturation, a5.uncoated_low_alloy_steel | - | a5.uncoated_low_alloy_steel |
| `E\d{1,2}` correction ids | pullout_angle, demagnetization, thermal_time_constant | calibration, slip_heating | pullout_angle, calibration, demagnetization, ferrite_cold_demag, slip_heating, thermal_time_constant, clamp_preload, a5.non_ferromagnetic_back_iron, a5.cte_mismatch_with_magnets |
| `C\d{2,3}` cell refs | - | - | calibration, br_temperature, a5.uncoated_low_alloy_steel |
| `Addendum` | - | - | 13 of 17 |

(Not covered by the six patterns but equally on screen: "M1 audit M9" in watch-outs and sources,
"Spec A5", "correction E7".)

---------------------------------------------------------------------------------------------------
## 5. On-screen jargon (C): where it lives and what pins it

Engine strings that cross the python boundary (verdicts, labels, units, choice texts) are HARD-locked
(section 6). The strings below are GUI-side or Rust-only, so removing them is allowed, but each is
pinned by tests that must be edited deliberately.

| # | String (what the owner sees) | Where | Pinned by | What it means for slice 2 |
|---|---|---|---|---|
| J1 | `Rust-only input (no workbook cell)` in every Rust-only input tooltip | `gui/inputs.rs:914-917` | `inputs.rs:1400` (and the full-string test H7) | Replace with plain words ("Added by this calculator, not in the original spreadsheet" or drop); edit test at `inputs.rs:1398-1402`. |
| J2 | `Model assumption (Addendum A3)` | `inputs.rs:936-938` | `inputs.rs:1401` (`ends_with`) | Same. |
| J3 | `Rust-only result (no workbook cell)` in result tooltips | `gui/dashboard.rs:413-417` | `dashboard.rs:1484, 1504` | Same. |
| J4 | Workbook cell and path lines (`Calculator!C93`, `model.pullout_Nm`) in every tooltip, plus the search hint "Search label, path or cell" | `inputs.rs:913-917`; `dashboard.rs:412-417`; `results_table.rs:52` (`SEARCH_HINT`) | `dashboard.rs:1474` (`"\nmodel.pullout_Nm\nCalculator!C93"`), `panel.rs:2619, 2675-2739` (`count(...,"Calculator!C92")`), `inputs.rs:1382, 1479-1488` (search by cell and path), `format.rs:120`, `results_table.rs:875, 951` | Search-by-cell and path is a feature (the search tests pin it) and the CSV/JSON/xlsx exports carry cells by design (`spreadsheet.rs:53` "Workbook cell", decision X-1). Decide per surface: hide from hover text vs keep for traceability. |
| J5 | Correction markers `E3 E7 E8` beside values; tooltip `Corrected vs workbook: E3: ...  at the default design, with E3 alone: workbook 2.647, corrected 2.688` | `gui/corrections.rs:162-190` (marker 162, tooltip 176-188); `dashboard.rs:459` | `corrections.rs:221-296`, `dashboard.rs:1330-1334, 1500`, `panel.rs:2572`, `results_table.rs:775, 825`, `spreadsheet.rs:949` | The markers are a feature with 6+ pinning tests and an export column. Plain relabelling is possible (the ids come from the registry), but is the widest-reaching of the jargon edits. |
| J6 | `3D values from the workbook` label and the note "...workbook's stored 3D values: they do not follow geometry changes until M3 computes them live." | `dashboard.rs:206-210` (`STORED_3D_LABEL`, `STORED_3D_NOTE`) | `panel.rs` stored-3D tests (README row 590), `dashboard.rs` STORED_3D_ROWS tests | Contains both "workbook" and "M3" (milestone name). Rewording is safe; keep the physics (the stored reverse fields do not follow geometry or grade: spec section 8 trap list item). |
| J7 | `Reset to workbook defaults`; assumption dot hover `Changed from the workbook default`; status `Assumptions reset to the workbook defaults`; banner doc "spec Addendum A3" (doc comment only) | `gui/panel.rs:129, 1021-1026, 1157` | via constants in tests (`RESET_ASSUMPTIONS`) and `panel.rs:6757` | Constants are used by name in tests, so changing the text is cheap; the hover and status strings are literals. |
| J8 | Input-order toggle `Workbook groups` | `gui/inputs.rs:80` (`InputOrder::Workbook`) | panel tests use `InputOrder::Workbook.label()` (no literal) | Safe to rename; also the A/B glyph test iterates both orders (`panel.rs:3974`). |
| J9 | Choice text `5: 1, 3, 5 (workbook)` (Rust-only input `coupling.max_harmonic`) | engine choice label; shown via `explorer.rs value_text` | `explorer.rs:1170`, `panel.rs:6906` | Rust-only so engine-editable (re-bless `input_schema.json`, H3) and two GUI tests to edit. |
| J10 | Explorer: `No equation record: the value, its label and its workbook cell are what the explorer knows.`; `Rust-only result` fallback | `gui/explorer.rs:73-76, 644, 795` | explorer tests (constants used by name) | Safe. |
| J11 | Results table `Rust-only` cell column text | `gui/results_table.rs:55-56` (`RUST_ONLY`), `spreadsheet.rs:284` | `spreadsheet.rs:844`, `results_table.rs:889` | Also an export value (xlsx/CSV); changing it changes exports. |
| J12 | Geometry design-check notes cite cells: `Design check: the cap thread diameter ... (Metal design C169) is below the cup body OD ... (C165)` | `gui/geometry.rs:~783-800` (`DESIGN_CHECK` const at 245-247) | geometry tests | GUI-side text; free to plain-word with a test edit. |
| J13 | Clamp drawing: `drawing.py's only layout` in `ONE_PIECE_ONLY`; `NO_SCREW_FITS` doc says `drawing.py's ValueError text` (only `ONE_PIECE_ONLY` is shown) | `gui/clamp_drawing.rs:33-39` | clamp_drawing tests (README row 593 "the one-piece note", "drawing.py's message") | The visible text names a Python file. Constant used by name; the many `drawing.py` mentions elsewhere are doc comments. |
| J14 | Assumptions view: each row draws `Source: <assumption.source>`; sources cite cells and "Addendum A3", e.g. "Spec Addendum A3 (workbook 1, 3, 5; ...); workbook Calculator rows 71 to 88; M2 decision D7 ..." | `gui/panel.rs:1141-1180`; text in `src/engine/assumptions.rs:47+` | HARD: `tests/assumptions.rs:123-127` requires `source` to CONTAIN each input's workbook cell, or "Addendum A3" for a Rust-only input; `panel.rs:6757` draws it verbatim; `dashboard.rs:2125-2137` exempts rationales from plain words | Removing "Addendum A3"/cell refs from what is drawn means either editing that test or drawing a plain `source` and keeping the cited one elsewhere (e.g. hover). The `label`/`rationale` fields have only a non-empty check (`assumptions.rs:112-118`) and are free to reword. |
| J15 | Engine Rust-only warning text: `Uncoated low-alloy steel: ... (the workbook plans electroless nickel)` | `src/engine/warnings.rs:85` | Tests read `rule.text` at runtime (`material_links.rs:345`, `panel.rs:7240+`), no literal | FREE to edit. Only the six `a5.*` ids are frozen (N12). Also drawn through the glyph test (`panel.rs:4019`). |
| J16 | WELCOME strip text "Set the size -> read the torque -> open any red check to see why and what to change." | `gui/panel.rs:68-69` (count tests at `panel.rs:1768-1781`) | `.claude/workflows/gui-smoke.js:83` (the smoke agent looks for this exact line with a "Hide" button) | Change the constant AND the smoke prompt together. |

Not jargon but pinned and easy to trip: the ring radius has several names, and the engine label of
`coupling.inner_back_apothem_mm` ("Inner magnet back apothem, including bondline", `src/engine/model.rs:167`)
is a ported label, so HARD-locked by `python_schema.rs` (H1). The only GUI-side renaming mechanism today is
`SIZE_TAB_ALIASES` (`guidance.rs:152-154`) which maps the ring radius to the Size tab's "Outside diameter" for
fix text. Any further display-name layer for labels must be GUI-side too, and it must not change the label
used by search/trace, which is matched against the real label (`gui/format.rs:33`, search tests `inputs.rs:1479+`).

---------------------------------------------------------------------------------------------------
## 6. Python reference boundary: does any human-readable text cross it?

| # | Rule | Where | What it means for slice 2 |
|---|---|---|---|
| X1 | Oracle package is `R/reference/magcoupling-py/magcoupling/` (`_fields.py`, `api.py`, `model.py`, `metal_design.py`, ...). READ-ONLY, "never edit; parity and differential tests compare against it unchanged". `tools/gen_differential.py` writes `python_schema.json`, `static_data.json`, `differential/*.json` | `docs/ai/03-structure.yaml:10-16`; `02-system.yaml:87-88` | Do not edit anything under `reference/`. |
| X2 | Text that crosses: input/result label, unit, help, cell, choice texts (python_schema.rs, H1); result TEXT values compared exactly (parity "text exact", differential `BRANCHES` list of verdict strings such as "Below hot minimum", "TOO NARROW: increase apothem or reduce poles", "Too thin", "OVER the magnet rating") | `tests/parity.rs:1-8`; `tests/differential.rs:74-160`; `02-system.yaml:85` | Engine verdict strings, labels and choice texts are not slice-2 material. The guidance layer wraps them (that is why slice 1 was GUI-side). |
| X3 | Rust-only inputs/results are skipped by parity, metadata and differential tests; `gen_differential.py` never passes them to Python; no Rust-only input may share a path with a Python one | `02-system.yaml:89`; `tests/python_schema.rs:129-142` | Help on the 11 Rust-only inputs, `warnings.rs` texts, choice labels of Rust-only selectors and `clamps.length_note` are engine-editable (re-bless `input_schema.json` for input metadata only). |
| X4 | Teaching notes, `Note`, `START_HERE`, assumptions' `rationale` and `label`, warning texts and every `gui/` string are NOT in any Python-compared data. `ASSUMPTIONS` `source` is constrained by H/J14. | `tests/python_schema.rs` + `differential.rs` read only metadata and numeric/verdict results | The note rewrite crosses no Python boundary. |
| X5 | Tables' column header label/unit/note equal the workbook headers except where a correction rewords the note | `tests/schema.rs:306-370`; README `588` | Do not touch table column text. |
| X6 | `gen_differential.py --check` (gate 12) exits 1 if a committed data file differs from a fresh run; it runs against the committed `input_schema.json` minus Rust-only | `gen_differential.py:402-436` | Help edits alone cannot make it stale; a change to ranges, choices or defaults would. Slice 2 should change none of those. |

---------------------------------------------------------------------------------------------------
## 7. Snapshots, golden files, files read by tests

| # | Item | Where | Text? | What it means for slice 2 |
|---|---|---|---|---|
| S1 | No `insta`, `.snap`, `expect_test` or "expected text" files (searched Cargo.toml, src, tests) | - | - | No snapshot to refresh for notes or help. |
| S2 | Golden files `tests/data/deviations/E3.json, E4.json, E5.json` (numbers, `include_str!` by `gui/corrections.rs:36-46`) | `M/tests/data/deviations/` | Cell names and numbers only | Unaffected. |
| S3 | `tests/data/input_schema.json` (Rust-written, includes `help`, "Do not edit by hand") | H3 | Yes, help | Unaffected if help stays out of the engine (recommended); otherwise re-bless. |
| S4 | `tests/data/python_schema.json` (Python-written, includes help/labels) | H1 | Yes | Never edit; regenerated only by the oracle script. |
| S5 | `tests/data/reference_values.json` equals the oracle's copy (a test checks) | README `648-650` | Workbook labels/values | Unaffected. |
| S6 | `include_str!` of source files: `app.rs:215` reads `.claude/workflows/gui-smoke.js` and extracts `MAGCOUPLING_SMOKE_PAYLOAD` and `MAGCOUPLING_SMOKE_DESIGN_FILE`; `app.rs:260` reads `web/tools/magcoupler/index.html` (canvas id, `<title>TITLE</title>`); `app/theme.rs:77,109` compares theme source files | `M/src/app.rs:207-257, 259-270` | Mostly structural | Do not reformat or rename the `const NAME = '...'` lines in gui-smoke.js. The test also asserts the log prefixes (`SHARE_LINK_LOADED`, `DESIGN_FILE_LOADED`, `SIZING_LOG_PREFIX`+`SOLVED_PREFIX`, `REGISTRY_LOG_PREFIX`) appear in the script, and that Auto sizes the default-requirement length into 14.0..14.5 mm. Notes/help edits do not touch these; a change to defaults, requirement, input paths or the share format would (regenerate the payload via `MagcouplingPanel::share_link`). |
| S7 | `tests/web_layout.rs` (linkage-sim-rs) keeps pages, vercel.json, scripts in step; reads page structure, not magcoupler on-screen text | `R/linkage-sim-rs/tests/web_layout.rs` | No | Unaffected. |

---------------------------------------------------------------------------------------------------
## 8. gui-smoke, web build and the gate

### 8.1 gui-smoke (`.claude/workflows/gui-smoke.js`, a Playwright agent, `model: 'sonnet'`)

| # | What it reads | Where | What it means for slice 2 |
|---|---|---|---|
| W1 | `first_view` is true only if the middle region shows the exact welcome line "Set the size -> read the torque -> open any red check to see why and what to change." with a "Hide" button, a tab row starting "Size and torque", torque readouts and the two-ring drawing | `gui-smoke.js:83` | Pins `panel.rs:68-75` (WELCOME, HIDE_WELCOME). Edit the smoke prompt if you change them (J16). |
| W2 | Header buttons "Save design", "Load design", "Copy share link" located by screenshot; click centre of "Load design" | `gui-smoke.js:85` | Do not rename or reorder those header buttons without editing the prompt. |
| W3 | Console log substrings: `magcoupling: loaded the design from the share link`, `magcoupling: loaded a design file`, `magcoupling sizing: Solved at`, `magcoupling explorer: ` | `gui-smoke.js:86-87`; checked in `app.rs:207-257` | Keep the log prefixes (they are not user-facing text). |
| W4 | Zero console errors; `title_warning` (BL-037); the linkage page and hub checks | `gui-smoke.js:62-70, 88` | Notes/help do not affect these. |
| W5 | Run requirement: "Before merging a batch that touched src/gui/ or magcoupling-rs/src/gui/" (`meta.whenToUse`), needs `scripts/build_magcoupling_web.sh` then `scripts/serve_web.sh`. Slice 1's run covered "dashboard scrolling at 1280x800" | `gui-smoke.js:1-6`; `04-memory.yaml:7` | Slice 2 touches `src/gui/` so a smoke run is required. Longer note text and a hover overlay make the dashboard taller; re-check scrolling and that no console error appears when opening notes. The smoke does not open a note or a tooltip today, so it will not catch a bad note by itself. |
| W6 | The build goes through `magcoupling_shipped.sh` and fails if a magcoupling-rs unit has `workbook-parity` | `02-system.yaml:96` | Use the scripts, never ad hoc cargo, for the web bundle. |

### 8.2 CI and deploy

`.github/workflows/deploy-web.yml`: on push to `main`, installs wasm-bindgen-cli 0.2.114, runs
`scripts/build_web.sh` and deploys to Vercel. It runs NO tests. The gate is the only test gate and it
is local. `01-meta.yaml:12`: "direct pushes to main; Vercel auto-deploys WASM on every push".

### 8.3 `linkage-sim-rs/scripts/gate.sh` (12 steps, `--full` adds a 13th: release WASM builds)

| Gate | Step | Slice 2 relevance |
|---|---|---|
| 1-3 | linkage-sim-rs: `cargo test --all`, `cargo clippy --all-targets`, WASM check | Independent of magcoupling since 2026-10-06, still part of GATE PASS. |
| 4 | `cargo test` in magcoupling-rs (engine only, no features) | Runs `notes.rs` unit tests and `tests/explain.rs`; the engine plain-words test cannot use `gui::guidance` here. |
| 5 | `cargo clippy --all-targets -- -D warnings` | Warnings are errors. |
| 6 | `cargo check --target wasm32 --lib` (engine) | Engine must stay pure std (`02-system.yaml:95`). |
| 7 | `cargo test --features app` | Runs all `src/gui/*` tests (panel, guidance, explorer, dashboard, inputs, app); the right home for notes/help plain-words tests. |
| 8-9 | clippy `--features app` and wasm32 clippy of gui lib and the web entry, `-D warnings` | Same. |
| 10 | workbook-parity guard on the shipped native and wasm builds, with negative controls | Unaffected. |
| 11 | wasm-bindgen version parity of both lock files with deploy-web.yml | Unaffected. |
| 12 | Oracle pytest (1,159 tests) and `gen_differential.py --check`; finds Python via `$MAGCOUPLING_PYTHON`, then a `.venv` in this checkout or any git worktree; SKIPs (does not fail) when none | The `.venv` is NOT in this checkout (`reference/magcoupling-py/.venv` absent); the last recorded run found `C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe` through the worktree scan. If that worktree goes away, gate 12 prints SKIP and AGENTS.md:24 still requires `MAGCOUPLING_PYTHON`. |

Runtime: not documented anywhere in the repo. Evidence from the scratchpad log `gate2.log` of the slice-1
run (warm cache; file times 11:10:32 for `gate.log` and 11:16:45 for `gate2.log`, so about 6 minutes
warm, inferred): gate 1 test time 181 s (976 tests), `tests/explain.rs` 20 s per run (gates 4 and 7),
670 magcoupling lib tests in about 2 s, oracle pytest 5.6 s, the rest are cached clippy/check runs of
1-10 s. A cold build is much longer and unrecorded. Test counts (static `#[test]` grep): 671 in
`M/src`, 240 in `M/tests`.

Note: `gate 4` shows `tests/explain.rs` as "12 passed; 2 ignored" (`release_notes_are_reviewed` and
`review_sheet`), confirming N5.

---------------------------------------------------------------------------------------------------
## 9. Docs that must change with slice 2 (P4, P5)

| Doc | Where | Change |
|---|---|---|
| `docs/ai/02-system.yaml` | line 107 (guidance layer; "Known limitation (slice 2): a fix can name an input Auto holds"), 113-114 (typesetter/glyph tests), 476 (`current_statuses.magcoupling`: "next: guidance slice 2 ...") ; new invariants for: the help overlay (GUI-side, not engine), note field(s), the plain-words guard over notes/help, the NOTES bound, the note review record | Update when facts change; keep short and factual (`feedback_docs_ai_workflow.md`). |
| `docs/ai/03-structure.yaml` | line 61 (`explain:` notes.rs), 65 (`gui:` file list: guidance.rs, explorer.rs, any new help module), 72 (tests list) | Add the new module and any new test file. |
| `docs/ai/04-memory.yaml` | line 7 (slice-1 open/deferred items, to be closed or carried), 11 (NEXT slice-2 entry), 82 (un-ignore `release_notes_are_reviewed`), 83 (M43-13 tau/theta note), 74 (symbol renames) | Move slice 2 from NEXT to done/active; resolve 82 and 83. |
| `docs/ai/05-update-tracker.md` | top of file, reverse chronological | A slice-2 entry in the slice-1 style (decisions, tests red first, G-8 style release note). |
| `docs/ai/backlog.yaml` | after BL-055 (line 8-16) | New BL entry, `risk: physics`, acceptance line. |
| `magcoupling-rs/README.md` | line 29 ("17 A4 teaching notes behind a physics-review gate"), 150-151 (module table rows `explain/`, `gui/`), 318-321 (guidance paragraph), 460-475 (Teaching notes paragraph, `START_HERE`), 577-597 test tables: `tests/explain.rs` (585), `tests/schema.rs` (588, only if engine metadata changes), `src/gui/panel.rs` (590), `src/gui/guidance.rs` (597), plus rows for any new test file or module | The README test table is the project's test index; each new test or file needs a row. |
| Spec | `docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md:206-210` ("Out of scope: Note rewrites and the three new notes (slice 2, with the 87 empty help strings and on-screen jargon ...)") and calculator spec A4 (`2026-09-28-magcoupling-calculator-design.md:352-371`) | Write a slice-2 spec (or amend A4 for the new audience and N7/N8). |
| `.claude/workflows/gui-smoke.js` | line 83 and any prompt text that quotes on-screen strings | Only if WELCOME or header button names change. |
| Slice-1 deferred items recorded in docs (not locks, but docs must stay true when closed) | `04-memory.yaml:7` list (hot-limit caveat for far geometry, hot-low block when blocks do not fit, clamp boss vs drawn boss, S4 per-ring cure fixes, S5 screw class fix, S6 cup-wall vs diameter note, S7 twelve_poles fit pricing, S11 unreachable slider note, S12 label aliases, S15 overall-length levers in the space claim, S17 fit-vs-diameter-claim note, S18 cap floor over the thread, S19 percent precision); `02-system.yaml:107` and `README.md:314-317` ("A fix can name an input Auto holds ... known limitation"); memory `project_magcoupler_guidance_overhaul.md` (dashboard ~2x taller; FixCache recomputes every drag frame; Cap OD/thread rows not locked under ring-size Auto; blank-grade manual magnets uncalibrated at `temperature.rs ~791`) | If slice 2 takes any of these, update the sentences that call them limitations. |

---------------------------------------------------------------------------------------------------
## 10. Decisions this map forces on the slice-2 plan (numbered, recommended option first)

1. Where do the 87 helps live? A. GUI-side overlay keyed by input path (recommended): three call sites
   (H6), no engine, data-file or oracle change, glyph test free via `input_tooltip` (H8), add a guard that
   every input has engine-or-overlay help and every overlay key is a real input. B. Engine metadata plus
   a `workbook_help` registry entry per field: fails H1/H2 and invents audit rows (H4). C. Edit the
   oracle: forbidden (X1). Also decide: overlay replaces or appends to workbook help (H5: needed for the
   `npole` case).
2. How many new notes? A. Three notes and fold the back-iron wall rule into the `a5.low_saturation`
   rewrite (N7, N13). B. Four notes and raise the 15..=20 bound plus amend spec A4 on purpose.
3. Where does "what to change" go? A. A new `Note` field drawn by `note_ui` (N8; touches notes.rs and
   explorer.rs only). B. Inside the six-sentence budget.
4. What to do with the drawn `Sources:` line (N10): plain-word the field, or keep provenance in a field
   that is not drawn. Both keep the `!sources.is_empty()` test green.
5. Where does the plain-words guard for notes/help live (G1)? A. Move `BANNED` and
   `plain_text_violations` to a pure-std engine module so the engine tests (gate 4) and gui tests share one
   definition (DRY). B. Keep it in `gui::guidance` and test notes from a gui test (gate 7 only).
   Either way extend `BANNED` carefully (G2: no blanket `M\d`).
6. Review flow for notes (N2-N5): who is the reviewer agent (session model), where the committed record
   lives (N3), and whether `release_notes_are_reviewed` is un-ignored and widened to new notes (N5).
7. Jargon scope (C): which of J1-J16 are in this slice. J1-J3, J6-J10, J12, J13, J15, J16 are cheap; J4,
   J5, J11 touch search, markers and exports; J14 collides with `tests/assumptions.rs:123-127`.
8. Authorization: new spec/plan pass (P1), new BL entry (P8), fresh deploy go (P10).
