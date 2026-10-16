# Teaching-note mechanism map (notes-mech, slice 2)

Scope: how a note is stored, ordered, gated, rendered, reached and tested. No physics. Read-only
pass over `magcoupling-rs` (repo untouched). All paths below are relative to
`C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs/` unless they start with `docs/`.
Scratch helpers (not in repo): `scratchpad/s2/parse_notes.py`, `stats.py`, `notes_parsed.json`.

## 0. Ten-second version

- A note is a `const` Rust struct in `src/engine/explain/notes.rs` (17 of them). Text is PLAIN `&'static str`:
  no markup, no inline math, no links to notes or inputs, no live numbers.
- Nothing in `markup.rs`, `symbols.rs`, `registry.rs`, `render.rs`, `scope.rs`, `eval.rs`, `tables.rs`,
  `record.rs` touches notes (the only mentions are doc comments). Notes link to the engine only through
  `Note.equations` (result paths) and `WarningRule.note_id` (`src/engine/warnings.rs`).
- Hard caps that bite slice 2: `NOTES.len()` must be 15..=20 (now 17, so 3 new notes fit, a 4th fails the
  test at `notes.rs:630-633`); each note must have 2..=6 `sentences` elements (`notes.rs:635-639`).
- Every note is `Review::Reviewed` today and ~10 tests go red if any linked note is `Draft`. So a rewrite
  must stay `Reviewed` in the same commit (or the test list in section 6 must be updated).
- NO test limits words per sentence, jargon, or "says what to change" in note text (audit TC-34 is right).
  `plain_text_violations` (guidance.rs:1457) is applied to guide texts only, never to `NOTES`.
- No Python parity/differential suite and no JSON fixture pins any note text. Only one note title is pinned
  verbatim: "Harmonic decomposition" at `src/gui/explorer.rs:1084`.

## 1. The `Note` struct and every field

`src/engine/explain/notes.rs:44-57` (`#[derive(Clone, Copy, Debug, PartialEq, Eq)]`, so every field must be
`const`-constructible and `'static`; there is no `Default`, so adding a field means editing all 17 literals).

| Field | Type | Meaning / who reads it |
|---|---|---|
| `id` | `&'static str` | Key. Unique (`notes.rs:604-606`). Convention `a5.<rule id>` for the six warning notes, pinned by `warnings.rs:215` (`rule.note_id == format!("a5.{}", rule.id)`). `Explorer::open_note(&'static str)` and `Learn::Note(&'static str)` need `'static`. |
| `title` | `&'static str` | Drawn bold at the top of the note (`explorer.rs:572`), as the header's selected label when the note is opened alone (`explorer.rs:454-456`), and as the CLICKABLE LINK TEXT of every "Why:" link (`dashboard.rs:753`, `:810`) and in the Start-here bar (`explorer.rs:117`). Everything goes through `glyph_safe`. |
| `equations` | `&[&str]` | Result paths the note explains; `#` stands for a digit run (a family: `model.tau#_Pa`, `clamps.table[#].preload_N`). `covers()` at `notes.rs:561-569`. Drives: Explain toggle (`note_for(path)`), Start-here, the "Explains" link row when opened alone. May be empty (3 notes are: `a5.ferromagnetic_sleeve_or_liner`, `a5.uncoated_low_alloy_steel`, `a5.cte_mismatch_with_magnets`); such a note is reachable only by a link. |
| `sentences` | `&[&str]` | 2..=6 ELEMENTS (doc says "plain sentences at Physics 2 level", `notes.rs:50`). Each element is drawn as its own wrapped `egui::Label` (`explorer.rs:573-575`), so an element is one visual paragraph. No per-element length limit. |
| `watch_out` | `Option<&str>` | One string, drawn in `warn_fg_color` with the hard-coded prefix `WATCH_OUT = "Watch out"` (`explorer.rs:101`, `:576-584`). The only "extra" slot; there is no "what to change"/"fix" field. 10 of 17 notes have one. |
| `diagram` | `Option<Diagram>` | One of 6 enum kinds (`notes.rs:14-28`): `SquareWaveHarmonics, FluxPathBackIron, TorqueAngle, EndFringing, DemagKnee, HeatingCurve`. Painted by `gui/diagrams.rs` (300x130 pt, `DIAGRAM_SIZE`), schematic not data. 8 of 17 notes have one; `DemagKnee` is shared by 2. |
| `sources` | `&[&str]` | MUST be non-empty (`notes.rs:640-644`, content untested). Drawn on screen as a small weak line `Sources: a; b; c` (`explorer.rs:588-598`) - this is where "M1 audit M9", "Addendum A decision 29", "Calibration!C21" reach users. |
| `review` | `Review` | `Draft` or `Reviewed { reviewer, date, record }` (all three `&'static str`, free text, NEVER checked for content). Gate: `note_for` / `explorer::reviewed_note` return only `Reviewed` notes (`notes.rs:584-586`, `explorer.rs:111-113`). Today: 17 Reviewed, 0 Draft. Reviewer strings: 6 "independent physics review 2026-10-01" (harmonics, calibration, br_temperature, demagnetization, ferrite_cold_demag, slip_heating), 11 "physics reviewer (session model)" (commit ee11bfd record). |

No other fields (no group, order, level, link list, glossary terms, symbol list). `Diagram` and `Review`
enums are the only supporting types in `notes.rs`.

## 2. Order and grouping

- `NOTES` array order (`notes.rs:62-539`): harmonics, back_iron_factor, pullout_angle, end_effect, calibration,
  br_temperature, demagnetization, ferrite_cold_demag, slip_heating, thermal_time_constant, clamp_preload, then the
  six `a5.*` in `WARNING_RULES` order. The UI never iterates `NOTES` for display (only tests do); order is cosmetic.
  `note_for_any_status` takes the FIRST note whose `equations` cover a path, but ownership is unique (tests).
- THE ONLY ORDER THE USER SEES is `START_HERE: &[(&str, &str)]` (`notes.rs:543-558`), 11 steps of
  (note id, the equation path it opens): harmonics `model.tau_Pa`; pullout_angle `model.pullout_Nm`; end_effect
  `model.f_end`; calibration `model.f_cal`; back_iron_factor `model.s1_iron`; br_temperature `model.pullout_20C_Nm`;
  demagnetization `temperature.demag.onset_skipping_C`; ferrite_cold_demag `temperature.demag.cold_check`;
  slip_heating `temperature.slip_loss.total_W`; thermal_time_constant `temperature.thermal.time_constant_s`;
  clamp_preload `clamps.capacity_Nm`. The six `a5.*` notes are not in it.
- There is NO grouping/sections/"levels" in code: no group field, no index/list-of-all-notes UI, no glossary.
  "Start here" is a flat walk; the doc comment calls the logical groups "torque chain -> back iron -> temperature ->
  demagnetization -> slip heating -> clamps" but nothing encodes group boundaries.
- The Start-here button tooltip hard-codes those group words (`explorer.rs:486`); change it if the order changes.
- `START_HERE` length is dynamic everywhere (`START_HERE.len()`), so adding/removing steps needs no constant edit.

## 3. Lookups (API)

`notes.rs`: `covers(entry, path)` (:561), `note(id)` (:572), `note_for_any_status(path)` (:577, tests/review tooling),
`note_for(path)` (:584, reviewed only). `explorer.rs`: `reviewed_note(id)` (:111, the gate for every by-id use),
`start_here_text(step)` (:116, `"Start here {n} of {len}: {glyph_safe(title)}"`, empty title if not reviewed),
`warning_note(rule)` (`dashboard.rs:141`).

## 4. How a note is rendered and reached in the GUI

Panel: `src/gui/explorer.rs` (`Explorer`, `explorer_ui`, `note_ui` :571, `note_alone_ui` :603, `equation_body` :633).
The "Equation panel" is docked at the foot of the centre region (`panel.rs:743-752`), closed until opened.

Entry points, with the code that creates them:
1. Header toggle "Equation panel" (`panel.rs:997-1007`) then click a value (readout hover/click ->
   `Explorer::open_path`, `explorer.rs:220`), then tick the "Explain" checkbox (`explorer.rs:491`, `explain` default
   false, pinned by `explorer.rs:1058` and `panel.rs:1761`). Shows `notes::note_for(&eq.target)` between the
   corrections line and the Terms list (`explorer.rs:678-687`); no reviewed note -> `NO_NOTE`
   "No teaching note for this equation." (:98, :538, :683). It follows drilling.
2. "Start here" button in the panel header (`explorer.rs:484-490`) -> `Explorer::start(0)` -> opens the step's path with
   Explain ON; Previous / Next / Stop bar (:495-515).
3. Dashboard "Why:" link per row (`dashboard.rs:736-778`, `WHY_LINK="Why:"`): `Learn::Note(id)` renders a link whose
   text is `glyph_safe(note.title)`; click -> `Readouts::open_note(id)` (`readouts.rs:140`) -> `ReadoutEvents.note` ->
   `Explorer::end_frame` -> `open_note` (`explorer.rs:229, 358-361`). `Learn::Assumption` links to the input row via
   `focus_input` and shows the rationale on hover; `Learn::None(reason)` shows "no note yet" (`NO_NOTE_YET`,
   dashboard.rs:107) with the reason as hover text; a non-reviewed `Learn::Note` shows "no note yet" with
   `NOTE_NOT_REVIEWED`.
4. Warnings block (`dashboard.rs:792-819`): for each fired `WARNING_RULES` rule, a link "Why: {note.title}" (`WHY` =
   "Why", :94) -> `open_note(rule.note_id)`.
5. A note opened alone (3 and 4) replaces the trail (`explorer.rs:229-235`); header shows the title as a selected label,
   body shows the "Explains" row (:603-628: links for each non-`#` `equations` path that has a record, link text =
   the record's symbol via `crumb`, hover = label; click = `open_path`, which clears the note), then `note_ui`.

`note_ui` layout (`explorer.rs:571-599`): title (strong) -> each `sentences` element as a wrapped Label -> optional
"Watch out: ..." (warn colour) -> optional diagram -> small weak "Sources: ...".

What is NOT reachable / not available: no link from a note to another note; no link from note text to an input row
(the plumbing exists - `Explorer::focus_input`, `Readouts::focus_input`, `Explorer::open_note` - and `explorer_ui` holds
`&mut Explorer`, but `note_ui(ui, &Note)` has no handle and draws only Labels); no input row links to a note (grep of
`input_ui.rs`/`inputs.rs`/`pickers.rs`/`size_controls.rs`: no note or Explain reference; the only input-row link out is
`TRACE_HINT`, which traces equations, not notes); the Size tab and the end-effect banner link to nothing; the
Equation panel's empty text (`EMPTY_TEXT`) and tooltips do not mention Explain or Start here.

## 5. Markup features available to a note

None beyond Unicode in a string.
- Equation markup (`markup.rs`: `frac(a,b)`, `{path}` terms, `sum`, `peak`, `cases`...) and symbol markup (`ϑ_{op}`)
  are for equation RECORDS only; the typesetter draws them (`typeset.rs`). Notes cannot embed a record, a term chip,
  a colour, a stacked fraction, or a sub/superscript. They fake it with text: `S_n`, `e^(−kg)`, `sin(nπλ/2)`,
  `A_n sin(nφ)`, `10⁻⁶`, `Br²`.
- `symbols.rs:19-29` says a symbol a note quotes "changes with the note, and a changed note goes back to Draft", but no
  test checks that note symbols match the registry's (the thermal note writes `τ`, `ϑ_start` where the panel shows
  `τ_th`, `ϑ_hot`: docs/ai/04-memory.yaml:83; audit TC-20, PA-15).
- `glyph_safe` (`typeset.rs:274-278`) rewrites ϑ U+03D1 -> θ, U+207B superscript minus -> ¯, ∝ U+221D -> ~ on the way to
  the screen. Characters currently used in titles/sentences/watch_out (all pass `has_glyph` in the tests):
  `° ± ² · × Δ Σ α β δ λ μ π σ τ φ ω ϑ – ⁶ ⁻ − √ ∝ ≈`. Any new character must pass `assert_glyphs`
  (`test_support.rs:183-192`) in the glyph tests (section 6).
- Numbers in a note are literals (`2.69 to 1.54 N·m`, `863 kA/m`, `0.3 W/K`, `336 °C`); no test ties them to the
  engine. `Note` has no function-valued field, so live numbers would need a struct change.

## 6. Every constraint on note content/structure (file:line)

### 6a. Unit tests in `src/engine/explain/notes.rs`
- :593-600 `templates_cover_their_members_only` - `covers()` semantics (`#` = one or more digits; `model.tau#_Pa` does
  not cover `model.tau_Pa`).
- :602-622 `ids_are_unique_and_every_link_resolves` - unique ids (:604-606); every `WARNING_RULES[i].note_id` resolves
  (:607-614); every `START_HERE (id, path)`: note exists and one of its `equations` covers `path` (:615-621).
- :624-657 `a_draft_is_hidden_and_every_note_is_complete`:
  - :626-629 `note_for_any_status("model.f_end")` is `end_effect` (pins ownership of f_end);
  - :630-633 `(15..=20).contains(&NOTES.len())`, message "spec A4: about 15 to 20 notes";
  - :635-639 `(2..=6).contains(&n.sentences.len())`;
  - :640-644 `!n.sources.is_empty()`;
  - :647-655 each `equations` entry (with `#` -> `1`) resolves back to this note via `note_for_any_status`, and
    `note_for(path).is_some() == reviewed`.

### 6b. Integration tests in `tests/explain.rs` (plain `cargo test`)
- :1173-1218 `notes_link_to_records_and_each_equation_has_at_most_one`: each `equations` entry names at least one
  result row of the DEFAULT design (`result_rows(compute_all(&DesignInputs::default()))`) (:1184-1188); every member
  has an equation record (:1189-1195); no equation record is owned by two notes (:1198-1205); each `START_HERE` path is
  a default-design result and, if in an Explained chain of `SCOPE`, has a record (:1206-1217).
- :1033-1055 `release_notes_are_reviewed` (`#[ignore]`, run by hand): every START_HERE note and every warning note must
  be `Reviewed`.
- :1060-1073 `review_sheet` (`#[ignore]`): prints records, not notes.

### 6c. GUI tests (need `--features gui`/`app`; gate 7 runs `cargo test --features app`)
`src/gui/explorer.rs`:
- :1055-1088 start_here walk; :1081-1087 pins `start_here_text(0) == "Start here 1 of {START_HERE.len()}: Harmonic decomposition"`
  -> VERBATIM TITLE PIN and "harmonics is START_HERE[0]". Explain default off (:1058).
- :1090-1096 `note_for(path).id == id` for every START_HERE step (path must have a record).
- :1098-1109 `reviewed_note(id)` is Some exactly when Reviewed; "no.such.note" is None.
- :1111-1124 open a note alone (uses id `"a5.low_saturation"`): replaces trail; `open_path`/`start` clear it.

`src/gui/panel.rs` (headless egui harness; `count()` matches a drawn text EXACTLY, :1682-1684):
- :6972-7015 Explain toggle. Needs notes `pullout_angle` and `end_effect` to exist and have their TITLES drawn >= 1 on the
  dashboard (they are the Why links); title count +1 when Explain is on; `pullout.sentences[0]` (RAW, not
  glyph_safe'd) drawn exactly once; the pull-out note's diagram paints >= 1 marked point (circle radius 3.5,
  `diagram_points`, :6965-6970); drilling to `model.f_end` shows the `end_effect` title; `mass.total_g` must have NO
  reviewed note (`NO_NOTE` drawn once, :7006-7010); toggling off removes `NO_NOTE`.
- :7017-7038 Start here: `count(start_here_text(0)) == 1` and `count(note(START_HERE[0].0).title) == 1` (so step 1's
  title must not also be a Why link on the default dashboard); Next/Previous/Stop.
- :7040-7062 each step: every drawn text has glyphs; bar ends with `glyph_safe(title)`.
- :7064-7079 every note, opened alone at 1280x3000: `glyph_safe(title)` is among drawn texts; every drawn text
  (sentences, watch-out, sources, diagram labels) has glyphs.
- :7239-7277 warning link: `rule[0]` (a5.non_ferromagnetic_back_iron) link text `"Why: {glyph_safe(title)}"` is
  clickable; its `sentences[0]` drawn once; its diagram marks NO point (`diagram_points == 0`, :7266-7269, so keep a
  no-circle diagram or none); the note's one equation link has text "circuit" (the symbol of
  `materials.circuit_backiron`) and opens `materials.circuit_backiron` (:7270-7276) - pins that note's `equations`.
- :7483-7508 every `Learn::Note` of `CHECK_GUIDES` (a5.low_saturation, demagnetization, clamp_preload, end_effect):
  title click opens it; `sentences[0]` drawn once, RAW. So `sentences[0]` of any linked note must not contain
  U+03D1, U+207B or U+221D (glyph_safe would change the drawn text) - currently true for all 17 (verified).

`src/gui/dashboard.rs`:
- :1509-1530 every `WARNING_RULES` note is reviewed and `warning_note(rule).id == rule.note_id`.
- :1634-1757 Why links: `WHY_LINK` drawn `DASHBOARD.len()+1` times (18); for each distinct link text in
  `ROW_LEARN + fit`, `count(texts, link) == (number of rows using it)` - so a note TITLE must not equal any other drawn
  dashboard text (guide labels like "Hot-low torque", "End-effect model", "Cup wall", result labels) or the count doubles.
- :1921-1958 `Learn::Note(end_effect)` click emits `events.note`; `Learn::Note("no_such_note")` shows `NO_NOTE_YET` with
  `NOTE_NOT_REVIEWED` on hover.
- :2060-2143 plain-text + glyph checks over guide strings (labels, Whys, reasons, fix texts, `NOTE_NOT_REVIEWED`...):
  NOT over note text.

`src/gui/guidance.rs`:
- :2363-2379 `every_learn_link_resolves`: every `Learn::Note` in `CHECK_GUIDES` and `ROW_LEARN` is reviewed.
- :2381-2400 `every_dashboard_row_has_a_why_link`: `ROW_LEARN` is exactly `DASHBOARD` in order; a row with a check leads
  where its guide does; a row WITHOUT a check whose path is covered by a reviewed note MUST be `Learn::Note(that id)`
  (:2392-2395). Consequence: if a new note claims `metal.torque_cold_high_Nm`, `model.cup_od_mm` or `mass.total_g`
  (rows with no check), `ROW_LEARN` must change in the same commit (mass.total_g must stay unowned anyway, see panel
  :7006).
- :2299-2309 `every_why_is_one_short_sentence`: <= 30 words, one sentence - applies to `CheckGuide.why` ONLY.
- :3325-3339 `no_guide_text_carries_workbook_jargon` / glyphs over `every_text()` (:2239-2269: guide labels, Whys,
  `Learn::None` reasons, fixes, Problems, status). NOT notes.

`src/gui/diagrams.rs` tests: `KINDS: [Diagram; 6]` (:485-492); :529-554 every diagram draws inside its rect with labels,
and the six label sets are pairwise distinct; a new `Diagram` variant needs the enum (notes.rs:14), the `match` in
`diagram_shapes` (diagrams.rs:110-117), `KINDS`, and a shape fn.

`src/engine/warnings.rs`: :56 `WARNING_RULES: [WarningRule; 6]`, :209-218 `note_id == "a5.<id>"` and the rules mirror
`WarningResults` fields.

### 6d. What no test checks (gaps)
Words/sentence, sentence count semantics beyond 2..=6, "says what to change", jargon (`plain_text_violations` /
`BANNED` = `C\d{2,3}`, `E\d{1,2}`, workbook, Addendum, Rust-only, Calculator!), `sources` content, review record
content, symbol agreement with records, numbers vs engine, that Explain exists for a Start-here title, one-sentence
elements. A slice-2 plain-language test would be new (reuse `guidance::plain_text_violations`, `BANNED` at
guidance.rs:1442-1449).

## 7. Are note texts pinned by Python parity/differential suites or fixtures?

No. Evidence:
- `grep` of note titles over the whole repo (excluding `target/`): hits only in `notes.rs`, `explorer.rs:1084`,
  `warnings.rs` (result labels such as "Low saturation", a different string), and three old plan docs.
- `tests/parity.rs`, `tests/differential.rs`, `tests/python_schema.rs`, `tests/schema.rs`, `tests/static_data.rs`,
  `tests/data/**/*.json`: the only "notes" matches are the magnet library's `notes` field (`tests/static_data.rs:41`,
  `static_data.json`) and an unrelated workbook cell comment in `reference_values.json`. `reference/magcoupling-py`: the
  only `notes` is `library.py` (library field). `python_schema.json` locks engine labels/help/verdict strings, not notes.
- Gate (`linkage-sim-rs/scripts/gate.sh` gates 4-12) and `magcoupling_shipped.sh` check features/builds only.
- The warning TEXTS in `warnings.rs` (`WARNING_RULES[i].text`) are separate strings that are asserted equal to the
  result value at `warnings.rs:236-239`; changing a warning's text does not touch notes (and is Rust-only, not in
  python_schema).

## 8. Per-note table (17)

"El" = number of `sentences` elements; "words/el" = whitespace words per element; WO = watch_out words; all are
`Review::Reviewed`. "Jargon" = what `plain_text_violations` (plus audit-style tokens) finds today in the visible text.

| # | id | title | El | words/el (total incl. WO) | WO | diagram | START_HERE | Dashboard / warning links | Jargon on screen |
|---|----|-------|----|---------------------------|----|---------|-----------|---------------------------|------------------|
| 1 | `harmonics` | Harmonic decomposition | 4 | 27/85/26/61 (221) | 22 | SquareWaveHarmonics | #1 | none | s4 "workbook"; src "Addendum A decision 29", "M1 audit M3, M4, M7" |
| 2 | `back_iron_factor` | Back iron: the sinh factor against free space | 5 | 19/32/31/26/22 (130) | - | FluxPathBackIron | #5 | none | src "M1 audit M3/M5" |
| 3 | `pullout_angle` | Pull-out torque against rotation angle | 3 | 30/17/32 (79) | - | TorqueAngle | #2 | rows `model.pullout_Nm`, `model.gearbox_input_ripple_Nm` (guidance.rs:305, 318) | s3 "(correction E7)"; src "M1 audit E7 (T3-RC2)", "Addendum A decision 29" |
| 4 | `end_effect` | End effect | 3 | 25/25/60 (164) | 54 | EndFringing | #3 | `END_EFFECT_GUIDE.learn` (guidance.rs:275) -> row `model.end_effect_check` (:342) | s3 "M1 skeptic"; WO "M1 audit M9"; src "M1 audit M8 and M9 (T3-RC1, T3-RC3)" |
| 5 | `calibration` | The one-point calibration | 4 | 54/44/22/46 (211) | 45 | none | #4 | none | s1 "workbook", "M1 audit M4, M5"; WO "correction E3", "M1 audit P2"; src "Calibration!C21", "tests/explain.rs CANCELLATIONS" |
| 6 | `br_temperature` | Remanence against temperature, and torque ∝ Br² (draws "~") | 4 | 38/39/19/53 (171) | 22 | none | #6 | row `model.pullout_20C_Nm` (guidance.rs:306) | src "Calculator C69, C70, C94; Metal design C8" |
| 7 | `demagnetization` | Demagnetization: knee, permeance and the onset temperatures | 5 | 31/58/53/29/19 (247) | 57 | DemagKnee | #7 | `TEMPERATURE_GUIDE.learn` (guidance.rs:197) -> rows `temperature.summary.governing_limit_C`, `margin_hot_day_C`, `verdict` (:332-339) | s5 "correction E20"; WO "M1 audit M13"; src "E20", "Addendum A decision 19" |
| 8 | `ferrite_cold_demag` | Ferrite: the demagnetization risk is at cold | 3 | 25/30/51 (156) | 50 | DemagKnee | #8 | none | src "Spec A6", "Addendum A decisions 3 and 19 (E20)", "A-1 plan decision A13" |
| 9 | `slip_heating` | Eddy-current slip loss and skin depth | 4 | 65/18/45/46 (208) | 34 | none | #9 | none | WO "correction E17"; src "M1 audit M11 (Stoll)", "Addendum A report section 5.5 (E17)" |
| 10 | `thermal_time_constant` | The thermal time constant | 4 | 26/37/29/38 (156) | 26 | HeatingCurve | #10 | none | s4 "correction E12"; WO "M1 audit placeholder table"; src "E15", "E12" |
| 11 | `clamp_preload` | Clamp preload and friction | 4 | 24/38/31/20 (131) | 18 | none | #11 | `CLAMP_GUIDE.learn` (guidance.rs:209) -> row `clamps.recommended` (:340) | src "M1 audit M14, E2", "Addendum A3 assumptions" |
| 12 | `a5.non_ferromagnetic_back_iron` | Non-ferromagnetic back iron | 3 | 30/74/28 (132) | - | FluxPathBackIron | - | warning `non_ferromagnetic_back_iron` | src "Spec A5", "Addendum A report section 5 (E15 to E17)" |
| 13 | `a5.ferromagnetic_sleeve_or_liner` | Ferromagnetic sleeve or liner | 3 | 20/39/46 (105) | - | none | - | warning `ferromagnetic_sleeve_or_liner`; `equations` empty | src "Spec A5", "Addendum A report section 4", "Plan A-3 Task 16" |
| 14 | `a5.high_conductivity_sleeve_or_liner` | High-conductivity sleeve or liner | 3 | 48/29/26 (103) | - | none | - | warning `high_conductivity_sleeve_or_liner` | src "Spec A5", "M1 audit P1", "Addendum A report section 4" |
| 15 | `a5.low_saturation` | Low saturation | 3 | 37/57/24 (142) | 24 | none | - | warning `low_saturation` AND `CUP_WALL_GUIDE.learn` (guidance.rs:188) -> row `materials.cup_wall_check` (:330) | s2 "workbook"; WO "M1 audit M1"; src "Addendum A decision 20", "M1 audit M1 and M2" |
| 16 | `a5.uncoated_low_alloy_steel` | Uncoated low-alloy steel | 3 | 25/21/19 (65) | - | none | - | warning `uncoated_low_alloy_steel`; `equations` empty | s2 "workbook"; src "Workbook Materials C26" |
| 17 | `a5.cte_mismatch_with_magnets` | Expansion mismatch with the magnets | 3 | 37/24/38 (99) | - | none | - | warning `cte_mismatch_with_magnets`; `equations` empty | s3 "Volkersen screen"; src "M1 audit E1", "Addendum A decision 16 (E18)"; s1/s3 contain U+207B (drawn as ¯) |

Density (61 elements): median 31 words, mean 35.5, max 85 (harmonics s2); 31 elements > 30 words, 17 > 40, 11 > 50.
Per-note totals: min 65, median 142, max 247. `plain_text_violations` hits in visible note text today: titles none;
sentences/watch-out: harmonics s4, pullout_angle s3, calibration s1 + WO, demagnetization s5, slip_heating WO,
thermal_time_constant s4, a5.low_saturation s2, a5.uncoated_low_alloy_steel s2; the Sources line: 15 of 17 notes hit a banned
pattern (Addendum, E-id, C-cell or workbook); only back_iron_factor and end_effect are clean (`stats.py` in the scratchpad).
Not banned by the existing regexes but still internal: "M1 audit M<n>", "T3-RC<n>", "Spec A5", "Plan A-3", "decision N",
"skeptic", "placeholder table".

Which dashboard rows link to NO note (all `Learn::None` / `Learn::Assumption`): `metal.torque_hot_low_Nm`,
`metal.hot_min_check`, `metal.torque_cold_high_Nm` (Assumption `production_variation`, guidance.rs:164, 313),
`model.cup_od_mm`, `mass.total_g`, `metal.min_running_clearance_mm`, `metal.clearance_check`,
`housing.space_claim_check` (+ the Magnets fit row) ("No note covers ... yet", guidance.rs:179, 218, 284, 322, 326).
Notes with no dashboard link and no warning: harmonics, back_iron_factor, calibration, ferrite_cold_demag, slip_heating,
thermal_time_constant (reachable only by Start here or Explain on an equation).
Inputs that link to a note: none.

## 9. How a NEW note must be wired (checklist)

1. `notes.rs`: append a `Note { .. }` to `NOTES`.
   - unique `id` (use `a5.` ONLY for a warning note, whose id is forced by the rule id);
   - `sentences.len()` in 2..=6; `sources` non-empty; `review: Review::Reviewed{..}` if anything links it (see 6c);
   - `equations`: every entry must (a) match >= 1 result row at the default design, (b) have an equation record for
     every member, (c) not be claimed by another note. `equations: &[]` is legal (link-only note).
   - **Count**: 17 + 3 = 20 passes; 21 fails `notes.rs:630-633` - raise the range (and the message "spec A4: about 15 to
     20 notes", the `notes.rs:59-61` doc, README.md:29/662, docs/ai/03-structure.yaml:61) if you add a fourth.
2. Optional `START_HERE` step: `(id, path)` with `path` covered by the note, a default-design result with a record
   (tests above). If the new note becomes step 1, `explorer.rs:1084` ("Harmonic decomposition") must change;
   `panel.rs:7027` needs the title to appear exactly once at that moment. Update the tooltip words at `explorer.rs:486`.
3. Optional dashboard link: edit `CheckGuide.learn` (guidance.rs:156-285) and/or `ROW_LEARN` (:304-343; fixed-size array
   keyed to `DASHBOARD`). Satisfy 6c guidance tests (rows without a check that a reviewed note covers must lead to it).
   Remove/replace the matching `Learn::None("No note covers ... yet")` reasons (they are also in `every_text()`).
   Title must be unique among drawn dashboard text (dashboard.rs:1719-1735).
4. Optional warning link: `WarningRule.note_id` (`a5.<id>`), array length 6 and `WarningResults` fields are type-level.
5. Diagram: reuse one of 6 (mind the :7266 no-point assertion for the non-ferromagnetic note and the >= 1 point
   assertion for the pull-out note), or add a variant in 3 places (enum, `diagram_shapes`, `KINDS`) with labels distinct
   from the other six.
6. Docs with the code (project rule): `README.md` lines 29, 41, 460-466, 585, 662 ("17 of the 17 are signed off");
   `docs/ai/02-system.yaml` (70, 76, 114), `03-structure.yaml:61` ("17 A4 notes"), `04-memory.yaml` (72-74, 83: a note edit
   "sends the note back to Draft and needs a physics review"), `05-update-tracker.md`, `backlog.yaml`.
7. Verify: `cargo test --manifest-path magcoupling-rs/Cargo.toml` (notes.rs unit tests + `tests/explain.rs`) and
   `cargo test --features app` (explorer/panel/dashboard/guidance/diagrams tests), then the repo gate.

## 10. Equation ownership for the candidate new notes (what is free to claim)

From `src/engine/explain/records/*.rs`: 275 record targets (families once); 77 are owned by a note, 198 unowned.
No equation is owned twice. Free to claim for the audit's candidates:
- Torque chain (T = sigma * 2 pi R_g^2 L, pole pitch, fill): `model.area_lever_m3`, `model.torque_2d_Nm`,
  `model.gap_radius_mm`, `model.pole_pitch_mm`, `model.fill_inner`, `model.fill_outer`, `model.active_length_mm`,
  `model.face_gap_mm`, `model.inner_thickness_mm`, `model.outer_thickness_mm`, `model.inner_face_radius_mm`,
  `model.outer_face_apothem_mm`. (Already owned: `model.tau_Pa`/`tau#_Pa` harmonics/pullout_angle, `model.pullout_Nm`,
  `model.f_end`, `model.f_cal`, `model.k#`.)
- Hot-low / cold-high: `metal.torque_op_Nm`, `metal.torque_20C_Nm`, `metal.torque_hot_low_Nm`,
  `metal.torque_cold_high_Nm`, `metal.hot_min_check`, `metal.hot_margin`, `metal.required_20C_Nm`,
  `metal.cold_high_Nm`, `metal.cold_high_input_Nm`, `metal.cold_for_hot_min_Nm`, `metal.op_temp_C`,
  `metal.alpha_br_per_C` (`metal.torque_cold_Nm` is owned by `br_temperature`). Claiming `metal.torque_cold_high_Nm`
  forces a `ROW_LEARN` edit (6c).
- Back-iron wall rule: `materials.cup_wall_check`, `materials.cup_wall_suggested_mm` are free; `model.backiron_needed_mm`
  and `model.bsat_T` are OWNED by `a5.low_saturation` (a new wall-rule note either leaves them or moves them, editing
  both notes in one commit).
- After the coupling slips: free: `temperature.summary.time_to_limit_high`, `temperature.summary.critical_drag_Nm`,
  `temperature.thermal.steady_est_C`, `steady_high_C`, `rotations_to_limit_high`, `temperature.slip_life.*`,
  `metal.slip_loss_W`, `metal.slip_energy_J`, `temperature.summary.*` (onsets, margins, verdict, governing_limit_C).
  Owned: `temperature.slip_loss.*_W`, `temperature.thermal.time_constant_s`, `time_to_limit_*`, `critical_drag_Nm`
  (thermal_time_constant), all `temperature.demag.*` listed in section 8.
- `mass.total_g` must stay unowned (panel.rs:7006-7010 uses it as the "no note" example).

## 11. Process and risk notes for slice 2

- Rewriting a note keeps its `Reviewed` status in code; nothing enforces re-review (only the docs convention
  "a note edit sends it back to Draft and needs a physics review"). If a rewrite is set to `Draft` for review, all of these
  go red: explorer.rs:1081-1096, panel.rs:6972-7062, 7239-7277, 7483-7508, dashboard.rs:1509-1530 and 1634-1757,
  guidance.rs:2363-2400, tests/explain.rs:1033 (ignored). So: edit text, run the fresh-context physics review (CLAUDE.md
  model rule: physics keeps the session model), then update `Review::Reviewed { reviewer, date, record }` to name it, all
  before the gate. The `record` string is free text; today it points at commit ee11bfd / the 2026-10-01 independent review.
- The slice-1 spec (G-3) exempts hints from a Draft/Reviewed gate; notes keep the gate (spec A4 "Accuracy gate",
  notes.rs:4-10).
- Slice-1 spec section 8 physics traps (radius strongest lever; fewer poles shrink the ring; pull-out guaranteed / re-lock;
  bench calibration beyond the prototype; 416 stainless as a safe fix; stored 3D demag fields follow geometry/grade) apply
  to note text too; existing `calibration` and `ferrite_cold_demag` notes already touch two of them.
- Because `sentences` is capped at 6 elements and each is one paragraph, "short plain sentences + what to change" needs
  either shorter elements within the 6 cap, the single `watch_out` slot, or a new field (e.g. `change: &[&str]`) added to
  all 17 literals and drawn in `note_ui` (the only renderer). A new field changes no test unless a test constructs `Note`
  (none does outside `notes.rs`).
- Inline links to inputs/notes inside note text would need `note_ui` to take `&mut Explorer` (it is reachable from
  `equation_body` and `note_alone_ui`, both inside `explorer_ui` which owns `&mut Explorer`); egui `Label` cannot embed a
  link, so text would have to be split into runs. Tests that depend on whole-sentence labels: panel.rs:6998, 7265, 7505
  (`count(sentences[0]) == 1`). Splitting `sentences[0]` into several labels would break these three.
- Other on-screen strings in the panel that are note-adjacent and would be touched by jargon cleanup: `EMPTY_TEXT`,
  `NO_EQUATION` (explorer.rs:64-75, "workbook cell"), `CORRECTIONS` line "Embodies corrections: E7, E8" (:78, :673-677),
  the `eq.cell` fallback "Rust-only result" (:644, :795), `NO_NOTE` (:98; pinned by panel.rs:7010-7014 as a constant, not literal).
  Tests reference those by constant, so wording can change freely except where a test pastes the literal
  (`dashboard.rs:1500-1504` pins "Corrected vs workbook:" and "Rust-only result (no workbook cell)").
