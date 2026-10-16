# Magcoupler guidance, slice 2: notes that say what to change, plain words on screen

Date: 2026-10-14. Branch: `magcoupling/guidance-2` (from `main` 5a1126a, slice 1 merged and
deployed). Status: decisions made by the owner on 2026-10-14 (all ten recommended options, rulings
R1-R14 approved). Slice 1's spec (`2026-10-13-magcoupler-guidance-design.md`) stays in force
except where section 2 amends it.

Research, read-only, 2026-10-14 (9 agents; committed beside this spec in
`2026-10-14-magcoupler-guidance-2-research/`):
- `synthesis.md`: facts F1-F36 with file:line evidence, the scope inventory, the decisions, the
  task sketch and the risks.
- `map-notes-mech.md`, `map-help.md` (section 5 lists all 87 empty helps with what each input
  physically is), `map-jargon.md` (85 jargon rows, Appendix A: 83 help/label declarations,
  Appendix C: jargon inside the notes), `map-locks.md` (every lock and pin, J1-J16).
- `phys-notes-a.md`, `phys-notes-b.md`, `phys-notes-c.md`: a physics read of all 17 notes, each
  with problems, must-keep facts, a what-to-change list, traps and a draft rewrite.
- `phys-new-notes.md`: the three new notes, drafted and checked against the engine.

The drafts are starting points, not approved text. The per-note physics review (section 4) is
the authority for every sentence that ships.

## 1. Why

Slice 1 put cause, fix and consequence where the user looks. The audit's other themes remain:
- **T4.** The notes are accurate but written for a specialist. Median sentence 31 words, maximum
  85; terms used before they are defined; audit ids in the text; no note teaches how the coupling
  makes torque, the hot-low/cold-high band, or what happens after it slips.
- **T6 (note part).** Notes present editable defaults and stored 3D fields as facts about the
  user's design ("863 kA/m here").
- **T7.** One radius has four names; users see "workbook", "Calculator!C93", "E3 E7", "Addendum
  A3", "M3", "drawing.py", "Rust-only"; 87 of 172 inputs have no help, and about 30 more have
  help that is terse, stale or wrong.
- **T5 (rest).** Notes are reached only through Explain, Start here and dashboard Why links.

Audience (owner, 2026-10-13): mechanically literate engineers who are not magnetics specialists.
Pain points: notes too dense, they do not say what to change, topics missing.

## 2. Decisions (owner, 2026-10-14)

| Id | Decision |
|---|---|
| S2-1 | Every note has three visible parts: **What it is** (3-5 sentences), **What to change** (structured entries: an input, a direction, the effect and its trade-off; the input's on-screen label is drawn live and links to the input row), **Watch out** (0-2 entries). Sources sit under a fold that starts closed. |
| S2-2 | Notes may quote live values from the user's design (`{result.path}` placeholders), rendered from the current results; never in sentence 0 or a title, never inside a comparative claim. |
| S2-3 | Three new notes: `torque_from_shear` "How the coupling makes torque" (Start here step 1), `hot_low_cold_high` "Hot-low and cold-high: the torque band", `when_slipping` "When the coupling slips". The cup-wall rule is delivered by rewriting `a5.low_saturation` as "Back-iron wall and saturation" (id frozen). 20 notes, at the cap. |
| S2-4 | One name per quantity on screen, through a GUI-side display-name overlay (section 3.5). Exports, files and links keep the original spreadsheet labels and paths; search matches both. |
| S2-5 | A header toggle **Show spreadsheet details**, off by default, gates every provenance line (cells, paths, E-badges, Source lines). "Workbook example design" becomes **Example design** (amends slice-1 G-6), hover "The starting values from the original spreadsheet." |
| S2-6 | Per-note physics gate: a fresh-context session-model reviewer checks each note at the five probe designs (3.2); it ships as Reviewed in the same commit, with the review committed. Before merge the owner signs off a generated old-vs-new sheet. |
| S2-7 | Help: fill the 87 empty helps and replace about 30 wrong, stale or terse ones through a GUI overlay; a short always-visible hint line under the Key design inputs and the five Size-tab controls. |
| S2-8 | Reachability bundle: Start here reordered (torque note first), Why links from the Size tab (Magnets per ring, Outside diameter), Required, Hot-low and the end-effect banner, and "No teaching note" replaced by the nearest note. Explain stays off by default (G-5 unchanged). |
| S2-9 | Text and GUI only. Engine gaps go to the backlog as `risk: physics` items: E21 (calibration offset clamp), E22 (per-grade bond-plane CTE), a gearbox cold-high check (needs the owner's limit; 0.805 N·m cold-high against a 0.6 N·m rating today), and the clamp boss diameter link. Notes state these limits as watch-outs. |
| S2-10 | Release: merge to `main` and push (deploys) only when `gate.sh` ends GATE PASS **with gate 12 running** (the oracle venv exists, no SKIP), every per-note review and the whole-branch review pass, gui-smoke passes, and the owner has signed off the note sheet. Then check the deploy run. |

Rulings (approved defaults):
- **R1. Help lives in a GUI overlay.** It is keyed by input path (`gui/plain_text.rs`). Engine
  metadata, `python_schema.json`, the Python oracle and the deviation registry are never touched
  for wording (F18). Rust-only helps are edited in the engine and `tests/data/input_schema.json`
  is re-blessed (F21).
- **R2. One plain-words guard.** `BANNED` and `plain_text_violations` move from
  `gui/guidance.rs` to a pure-std engine module (`engine/explain/plain.rs`), re-exported by
  `gui::guidance`, so gate 4 can test the notes. Targeted patterns are added (section 3.4); never
  a blanket `M\d` (F26).
- **R3. Equation ownership,** one note per equation (F8). The table is in section 3.3.
- **R4. Start here has 14 steps** (section 3.3).
- **R5. Titles** are plain and unique against every drawn dashboard label (F12), with no ∝, ϑ or
  superscript minus.
- **R6. Symbols match the equation panel.** σ means shear stress only; conductivity is written in
  words or as σ_st and σ_sl; τ_th and ϑ_hot as the panel shows them (closes M43-13).
- **R7. Diagrams.** TC-22: draw the fundamental dashed beside the summed curve (the point-count
  pin still holds). Fix the TC-24 labels. The ferrite note drops the NdFeB DemagKnee diagram. No
  new Diagram variants.
- **R8. No glossary type.** Each term is glossed in plain words at its first use along Start
  here.
- **R9. Cup-wall verdict on screen.** "Too thin: raise Metal design C122 ..." is shown with the
  input's display label. The number is parsed from the engine text after `verdict_level` has
  read it (F25). The engine string is unchanged.
- **R10. Assumption rationales.** The 8 with jargon are rewritten; `end_effect` reads "within 2 %
  of free-space 3D for magnets 3.2 mm and longer" (`assumptions.rs:59`); `calibration_factor`'s
  "own rings" is fixed (PA-18). `source` keeps its cells, shown only under the toggle (F24).
- **R11. Warning tests pinned to the library:** low_saturation fires only for back-iron codes
  {3,4,5,6}; cte for {7,8} and for Back iron 0; no sleeve choice fires the ferromagnetic or the
  high-conductivity warning; uncoated fires for {1,2,3} at plating thickness 0.
- **R12. Process.** Open BL-056 (`risk: physics`). Docs change with every task. Ruling at
  execution: work on the branch in the main checkout as slice 1 did (its gate passed there); a
  worktree is used only if one is needed, on a short path (F35).
- **R13.** `a5.ferromagnetic_sleeve_or_liner` gets the shortest rewrite (its warning never fires
  on the library): jargon out, plus "the torque model has no term for it".
- **R14. Smoke.** The welcome strip and header buttons stay unchanged (F28), apart from the new
  toggle. The smoke prompt opens one note through a Why link and one input tooltip, and checks
  dashboard scrolling at 1280x800.

## 3. Design

### 3.1 Note structure (S2-1)

`engine/explain/notes.rs` (pure std):

```rust
pub enum Step { Raise, Lower, Choose }

/// One "What to change" entry. The GUI draws "<Step word> <display label of input>: <text>",
/// the label a link that focuses the input row.
pub struct Change {
    pub input: &'static str,                  // an input path that exists in the catalogue
    pub step: Step,                           // Choose for pickers and choice inputs
    /// The result this entry claims to move, and which way (Some for Raise/Lower on numeric
    /// inputs; checked by a test at the probe designs where the input is free).
    pub aim: Option<(&'static str, Sense)>,
    pub text: &'static str,                   // effect and trade-off, <= 25 words
}
pub enum Sense { Up, Down }

pub struct Note {
    pub id, title, equations, diagram, sources, review,   // unchanged meaning
    pub sentences: &'static [&'static str],   // "What it is": 3-5, each <= 25 words
    pub change: &'static [Change],            // 1-5 entries (0 allowed only for R13's note)
    pub watch_out: &'static [&'static str],   // 0-2, each <= 30 words
}
```

Rules:
- Sentence 0 answers the question of the row that links to the note, in plain words, with no
  symbol, no placeholder and no ϑ, U+207B or ∝ (F10). It is drawn as one label.
- Formulas appear only where they carry the idea; the equation is already above the note in
  Explain.
- `change` entries name inputs by path only; a label is never typed (slice-1 spec 4.1). The
  label comes from the display-name overlay (3.5).
- An entry whose input is held by Auto or overridden by a part picker on the current design is
  drawn with "(held by Auto)" or "(set by the <picker label> pick)" instead of a link.
- Every entry names its trade-off when the move pushes another check toward red (a longer magnet
  raises cold-high and the clamp demand; a smaller gap tightens running clearance; a thicker wall
  grows the cup).
- The GUI draws: the sentences; a "What to change" heading and its entries; a "Watch out" heading
  and its entries; a "Where this comes from" fold (closed) holding `sources`.
- Caps are enforced by tests. They are written red with the struct change, marked `#[ignore]`
  while the old text is in place, and un-ignored when the last note is rewritten.

### 3.2 Live values (S2-2)

- Syntax: `{result.path}` inside `sentences[1..]`, `change[].text` and `watch_out`. A pure-std
  parser splits text into literal and placeholder segments; the renderer formats the value with
  the dashboard's value formatter and unit. A non-finite value renders as "no prediction".
- Placeholders never appear in sentence 0, a title or `sources`. They appear only in "here: X" or
  "on this design, X" clauses, never inside a comparison whose truth depends on the value.
- **The five probe designs**, used by tests and reviews:
  1. the default (Example) design;
  2. the default at 8 magnets per ring;
  3. a non-ferromagnetic back iron (304);
  4. Y30 ferrite on custom dimensions;
  5. a 20 mm ring radius.
- Tests: every placeholder names a result present at the default design; the text rendered at
  all five designs contains no `{` or `}` and passes the plain-words and glyph guards.

### 3.3 New notes, equation ownership, Start here (S2-3, R3, R4)

Ownership (each equation has at most one note; `mass.total_g` stays unowned):

| Note | Owns |
|---|---|
| `torque_from_shear` (new) | `model.area_lever_m3`, `model.torque_2d_Nm`, `model.gap_radius_mm`, `model.pole_pitch_mm`, `model.active_length_mm` |
| `hot_low_cold_high` (new) | hot-low, cold-high, the hot-minimum check and margin, the required-at-20 °C rows, the cold-for-hot-minimum rows, `torque_op`, `torque_20C` |
| `when_slipping` (new) | `model.gearbox_input_ripple_Nm`, `metal.slip_freq_Hz` |
| `harmonics` | the fill rows |
| `br_temperature` | keeps `metal.torque_cold_Nm`, adds the three alpha records |
| `clamp_preload` | adds `clamps.max_torque_Nm`, `clamps.required_Nm` |
| `slip_heating` | the duty-frequency rows |
| `thermal_time_constant` | `magnet_life.peak_C`, the `steady_*_C` rows, the rise-per-event rows |
| `demagnetization` | the onset margins |
| `a5.low_saturation` | adds `cup_wall_check`, `cup_wall_suggested_mm`, `gap_flux_density_T` |
| `a5.high_conductivity_sleeve_or_liner` | adds `materials.sleeve_sigma_S_m` |

The exact paths are verified against the records when the plan is written
(`synthesis.md` 3.11 R3; `phys-new-notes.md` section 6).

Wiring: `HOT_MIN_GUIDE.learn` becomes `hot_low_cold_high` (`guidance.rs:164`); the cold-high
ROW_LEARN entry follows its owner (`guidance.rs:309-313`); the ripple ROW_LEARN entry moves from
`pullout_angle` to `when_slipping` (`guidance.rs:316-319`). The step-1 note cannot also be a
dashboard Why target (F11), so `torque_from_shear` owns no dashboard row.

Start here (14 steps): 1 `torque_from_shear`, 2 `harmonics`, 3 `back_iron_factor`,
4 `pullout_angle`, 5 `end_effect`, 6 `calibration`, 7 `br_temperature`, 8 `hot_low_cold_high`,
9 `when_slipping`, 10 `slip_heating`, 11 `thermal_time_constant`, 12 `demagnetization`,
13 `ferrite_cold_demag` (its step label reads "(ferrite rings only)", TC-32), 14 `clamp_preload`.
The pinned step-1 title and the tooltip (`explorer.rs:486, 1084`) follow.

### 3.4 The plain-words guard (R2)

Moved to `engine/explain/plain.rs`; `gui::guidance` re-exports it. Existing patterns stay;
added: "M1 audit", "skeptic", "Plan A-", `decision \d`, "drawing.py", `\.rs\b`, "Metal design",
"Temperature design", "Shaft clamps", "Calibration!", `correction E\d`, and a whole-word `M\d`
audit id with a whitelist for screw and thread sizes (M2.5-M6, M41 x 0.5) and grade names
(B842SH, N42SH, Y30, M5045). The guard's own example tests grow with it.

Applied to: every note field (rendered at the five designs), every overlay string (labels,
help, hints), every drawn tooltip with the toggle off, the assumption rationales (the exemption at
`dashboard.rs:2120-2133` is dropped), the warning texts and the explorer's own strings.

### 3.5 Display names and help overlay (S2-4, S2-7, R1)

`gui/plain_text.rs` holds two path-keyed tables and two accessors:
- `label_for(path) -> &str`: the display label, falling back to the catalogue label.
- `help_for(path) -> &str`: the overlay help, falling back to the engine help.

Canonical names:

| Quantity | Name on screen |
|---|---|
| `coupling.inner_back_apothem_mm` | Ring radius (the shaft centre to the back of the inner magnets) |
| the Size-tab diameter control | Outside diameter (the largest of cup, cap and boss; it follows the ring radius) |
| `coupling.npole` | Magnets per ring |
| `metal.face_gap_mm` | Magnet face gap |
| `model.gap_radius_mm` | Gap radius R_g |
| `metal.required_min_Nm` | Required torque |
| `model.pullout_Nm` | Pull-out torque (nominal) |
| `metal.torque_hot_low_Nm` | Hot-low torque |
| `metal.torque_cold_high_Nm` | Cold-high torque |
| `metal.variation` | Production variation allowance |
| the workbook | the original spreadsheet |

No torque is ever called "guaranteed" (slice-1 spec 8).

The display label is used on input rows, in fix text (C-3 uniqueness runs on display labels), in
equation term names and on dashboard rows. Search matches the display label, the catalogue label,
the path and the cell. Exports keep the catalogue labels. GUI-owned strings are renamed ("ring
size" -> "ring radius"; Auto dropdown; Size-tab footnotes; plot legends; "Nm" -> "N·m").

Help style (all overlay and rewritten help): 1-2 sentences, at most 30 words: what the input
physically is, which results it moves and which way, and the trade-off. Say when a part picker
overrides the input or Auto holds it. Calibration inputs describe the bench prototype, not the
user's design. Optional-adapter fields change mass and length only. No default numbers (the
tooltip prints "Default:"), no spreadsheet or audit ids.

Scope: the 87 empty helps (`map-help.md` section 5) and the overrides listed in `synthesis.md`
2.3 (npole, face gap, `required_min`, `beta_hcj`, the grade-mode alpha/density/rating helps,
`b_cup_T`, and the "3D.", "User supplied.", "Placeholder.", "UNCONFIRMED." family). With the
toggle on, a tooltip whose help was overridden also shows the original as "Original spreadsheet
note: ...". Rust-only input helps and the jargon result helps (`map-jargon.md` Appendix A) are
rewritten in the engine.

Hint lines: one short weak-text line under each Key design input and each of the five Size-tab
controls, from the same table (a `hint_for(path)` accessor; a hint is at most 12 words).

A coverage test: every input has non-empty help from the engine or the overlay; no overlay key
names a missing path; every overlay string passes the guard and the glyph test.

### 3.6 Provenance toggle (S2-5)

- One header toggle "Show spreadsheet details", off by default. GUI-only: not saved, not in share
  links, not in undo.
- It gates: the tooltip path and cell lines; the results-table cell and marker columns; the
  dashboard E-badges (with a one-line legend when on); the explorer's "(Calculator!C93)" and
  "Embodies corrections" lines (titles on hover); the assumption Source lines; the
  "Original spreadsheet note" line (3.5).
- With the toggle on, the marker hover reads "Differs from the original spreadsheet: <title>, was
  X, now Y" with no repo path.
- The note "Where this comes from" fold is closed by default either way.
- Exports and search are unchanged.
- The status line says "Example design: fails N of 8 checks"; its hover says "The starting values
  from the original spreadsheet." The gui-smoke constants follow.

### 3.7 Reachability (S2-8)

- Start here order (3.3).
- The Equation panel's empty text mentions Start here and Explain (TC-29b).
- "No teaching note" offers the nearest reviewed note upstream of the row, or the assumption
  rationale (TC-29a, TC-30).
- New Why links: under Magnets per ring and Outside diameter on the Size tab, to
  `torque_from_shear` (link text "How the coupling makes torque", not the bare step-1 title text
  if F11's pin forbids it); Required and Hot-low on the Size tab, to `hot_low_cold_high`; the
  end-effect banner, to `end_effect`.
- Explain stays off by default.

### 3.8 Other jargon

The cheap GUI strings of `map-locks.md` J1-J3, J6-J10, J12, J13, J15, J16 and `map-jargon.md`
C-4..C-6, C-8, C-11, A-10 ("Autofit"); the R9 cup-wall display; the stored-3D wording ("3D values
from the workbook ... until M3" becomes a plain statement that the values are fixed and do not
follow the design); "1, 3, 5 (workbook)" becomes "1, 3, 5 (default)". Engine verdict strings and
Python-locked labels, help, cells and choices are never edited (F18, F25).

## 4. Review (S2-6)

Per note, inside the task that writes it:
1. The implementer writes the note from the draft and the engine.
2. A fresh-context session-model physics reviewer (never sonnet) reads the note rendered at the
   five probe designs, the engine code behind every claim, the traps (section 8) and the
   `change` entries.
3. Must-fix items are applied; changed sentences are re-reviewed. Up to three rounds.
4. **Reviewed** means: the reviewer returns PASS with no open must-fix; every number is
   reproduced from `compute_all`; every `change` direction is confirmed by a probe (and by the
   `aim` test where it has one).
5. In the same commit the note becomes `Reviewed { reviewer: "physics reviewer (session model)",
   date, record: "docs/analyses/2026-10-14-magcoupler-notes-review.md#<id>" }` and the review
   section is appended to that committed report.

`release_notes_are_reviewed` (`tests/explain.rs:1037`) is un-ignored and widened to every note
linked from START_HERE, `CHECK_GUIDES`, `ROW_LEARN` and the warnings.

Help, display labels, hint lines, the verdict display and other GUI text follow slice-1 G-3: one
session-model physics review per batch, numbers from results, no Draft gate.

Owner sign-off: an ignored test prints a sheet (old text vs new text for each note, rendered at
the default design, with its change and watch-out entries) to
`docs/analyses/2026-10-14-magcoupler-notes-signoff.md`. The owner reads it and signs off before
merge.

## 5. Edge cases

- A placeholder whose result is non-finite or missing (end effect out of range, ferrite-only
  rows on NdFeB): "no prediction"; the sentence still reads correctly.
- A change entry whose input is held by Auto, overridden by a picker, or at its slider end: drawn
  as plain text with the reason, never as a link that does nothing.
- A note opened while its owner row is greyed (ferrite rows on NdFeB): the step label says
  "(ferrite rings only)".
- Display-label collisions: C-3 disambiguation and `every_input_has_its_own_fix_text` run on
  display labels.
- Search for an old label, a path or a cell still finds the input.
- The toggle off hides provenance in every drawn place; on, it shows all of it; no test or export
  depends on it being on except the tests that switch it on explicitly.
- A narrow window: hint lines wrap; the Equation panel scrolls.

## 6. Tests

1. Note struct caps (3-5 sentences <= 25 words, 1-5 changes, 0-2 watch-outs <= 30 words, sentence-0
   rule), red with the struct change, un-ignored at the end.
2. Every `change.input` exists; every `aim` direction holds at each probe design where the input is
   free and finite.
3. Placeholders: parse, render at five designs, no braces left, guard and glyph clean,
   "no prediction" for non-finite.
4. Equation ownership (one note per equation, every entry a default-design result with a record).
5. Start here: 14 steps in the order of 3.3; the pinned step-1 title.
6. Wiring: `HOT_MIN_GUIDE.learn`, the cold-high and ripple ROW_LEARN entries, every Learn::Note
   resolves to a Reviewed note.
7. `release_notes_are_reviewed`, widened.
8. Plain-words guard: moved, extended, and applied to every surface of 3.4; its own positive and
   negative examples (screw sizes and grades stay clean).
9. Overlay coverage: every input has help; no stray keys; every label, help and hint passes the
   guard and the glyph test; search finds old labels, paths and cells.
10. Toggle: off hides every provenance line (one headless test per site), on shows them; the
    pinned tooltip tests switch it on; not saved, not in links.
11. Reachability: each new Why link opens its note (headless click test); "No teaching note" never
    shows on a dashboard row.
12. R11 warning tests.
13. gui-smoke: the welcome strip and header (plus the toggle), one note via a Why link, one tooltip,
    dashboard scrolling at 1280x800.

## 7. Release (S2-10)

1. Each task leaves `cargo test` green for the crates it touches; GATE PASS at the end of the build.
2. Whole-branch review on the session model; one fix wave; re-review.
3. `bash linkage-sim-rs/scripts/gate.sh` ends GATE PASS with gate 12 running (the oracle venv at
   `reference/magcoupling-py/.venv`).
4. Web build and gui-smoke.
5. BL-056 notes and the update tracker record the outcomes; the sign-off sheet is generated.
6. **Stop for the owner's sign-off of the note sheet.**
7. Then: `git checkout main && git merge --no-ff magcoupling/guidance-2`, push, watch "Deploy Web to
   Vercel", check the live page.

## 8. Physics traps (must not be written)

Slice-1 spec section 8 still applies. Added (verified in `synthesis.md` section 5.2 and the
`phys-*.md` trap lists):
- "Fewer poles reach across the gap better" (net torque falls; S_1 peaks near the default pitch).
- "Torque grows as R_g²" stated without "at fixed shear stress" (torque dips between 14 and 20 mm
  ring radius).
- "The rings re-lock below pull-out" (they re-lock only when the slip speed falls).
- "The end factor is accurate" (optimistic by about 11 % at a 4.4 mm gap).
- "Edit the calibration inputs to match your design" (gives a meaningless factor).
- "Each ring uses its own α" (only rings with a picked grade do).
- "416, 17-4PH or 12L14 are clean swaps" (all trip low saturation).
- "Steel gives the torque shown" without the saturation caveat.
- "A hotter-rated grade is free" (torque falls 12-23 %; until E21 its onset can be optimistic).
- "More mass lowers the steady temperature" (it only slows the heating).
- "The coupling can slip indefinitely" (the margin is small and the conductance a placeholder).
- "The clamp is sized to the drive" (it is sized to cold-high times the safety factor).
- "Raising Boss outside diameter enlarges the drawn boss" (it does not).
- "Model ferrite by typing a positive β" (the GUI clamps it away).
- "The expansion caution counts the temperature swing" (it does not).
- "1.5 T is 4140's saturation" (unsourced).
- "The coupling protects the gearbox" (cold-high exceeds the 0.6 N·m input rating; nothing checks
  it).

## 9. Out of scope

Engine physics changes (E21, E22, the gearbox check, the clamp boss link: backlog items under
S2-9); the torque sensitivity card (slice 3); guided lessons, plot captions and drawing titles;
Explain on by default; links between notes; new Diagram variants.
