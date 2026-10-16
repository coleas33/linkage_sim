# Magcoupler guidance slice 2: synthesis for the spec and plan

Read-only synthesis, 2026-10-14, at HEAD `5a1126a` (slice 1 merged). Sources:
`s2/map-locks.md`, `s2/map-notes-mech.md`, `s2/map-help.md`, `s2/map-jargon.md`, `s2/phys-notes-a.md`
(torque-chain notes), `s2/phys-notes-b.md` (temperature and clamp notes), `s2/phys-notes-c.md` (the six
A5 warning notes), `s2/phys-new-notes.md`, and `audit.json` (synthesis themes T4, T6, T7 and their kept
findings). The key citations were spot-checked against the repo. Paths are relative to `magcoupling-rs/`
unless they start with `docs/`, `.claude/`, `AGENTS.md`, `CLAUDE.md` or `linkage-sim-rs/`.

Slice 2 is: (A) rewrite the 17 teaching notes for mechanically literate engineers who are not
magnetics specialists, so that each says what to change, and add three new notes; (B) fill the 87
empty input helps; (C) remove on-screen jargon. The owner's pain points are: the notes are too dense,
they do not say what to change, and topics are missing (`docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md:25-28`).

---------------------------------------------------------------------------------------------------
## 1. Facts slice 2 must respect

### 1.1 The note gate and note structure (HARD: tests go red)

| # | Fact | Where |
|---|---|---|
| F1 | A note shows only when it is `Review::Reviewed`. `note_for` and `explorer::reviewed_note` filter on that status, so a Draft note is hidden. | `src/engine/explain/notes.rs:30-41, 583-586`; `src/gui/explorer.rs:110-113` |
| F2 | If a note that something links to is Draft, about 10 GUI tests go red: the explorer tests, the Explain/Start-here/warning/guide tests in the panel, the dashboard Why tests and the guidance link tests. Users meanwhile see "Its teaching note has not passed review." | `src/gui/explorer.rs:1081-1109`; `src/gui/panel.rs:6972-7062, 7239-7277, 7483-7508`; `src/gui/dashboard.rs:106-110, 1509-1530, 1634-1757`; `src/gui/guidance.rs:2364-2400` |
| F3 | Editing a note's text, its title, or a symbol it quotes sends it back to Draft. It then needs a fresh-context physics review on the session model. This is a convention: nothing in code enforces re-review. | `notes.rs:1-10`; `docs/ai/04-memory.yaml:74, 83`; `src/engine/explain/symbols.rs:19-29`; `CLAUDE.md` section 5 |
| F4 | The review `record` is free text. It should name a committed artifact, but `.superpowers/` is gitignored. A record cannot name its own commit. | `.gitignore:56`; precedent `docs/analyses/2026-10-06-open-physics-review-response.md` |
| F5 | `release_notes_are_reviewed` checks that every START_HERE note and every warning note is Reviewed. It is marked `#[ignore]`, and un-ignoring it is "the next engine change's step". Slice 2 is that engine change. | `tests/explain.rs:1037-1038`; `docs/ai/04-memory.yaml:82` |
| F6 | `NOTES.len()` must be in 15..=20. There are 17 now, so at most 3 new notes fit; a 4th fails. Each note needs 2..=6 `sentences`, non-empty `sources`, and a unique id with every link resolving. `end_effect` must own `model.f_end`. | `notes.rs:631, 636, 641, 603-622, 626-629` |
| F7 | `Note` is a `const` struct with no `Default`. Adding a field means editing all 17 literals and `note_ui`, the only renderer. No code outside `notes.rs` constructs a `Note`. | `notes.rs:45-57`; `src/gui/explorer.rs:571-599` |
| F8 | One note per equation. Every `equations` entry must be a result of the default design that has a record. | `tests/explain.rs:1174-1218` |
| F9 | The six `a5.*` note ids are frozen, because `rule.note_id == "a5.{rule.id}"`. | `src/engine/warnings.rs:215` |
| F10 | `sentences[0]` is counted as the RAW string for `pullout_angle`, the first warning note and every guide-linked note. So sentence 0 must not contain ϑ, U+207B or ∝, must not contain a placeholder, and must be drawn as one label. | `src/gui/panel.rs:6998, 7265, 7505` |
| F11 | The step-1 title is pinned verbatim as "Harmonic decomposition". The step-1 title must be drawn exactly once while step 1 is open, so the step-1 note cannot also be a dashboard Why target. The tooltip text with the group words is hard-coded. | `src/gui/explorer.rs:1084, 486`; `src/gui/panel.rs:7027` |
| F12 | A note title must not equal any other text drawn on the dashboard, because link counts must match exactly. | `src/gui/dashboard.rs:1719-1735` |
| F13 | Every `Learn::Note` must resolve to a Reviewed note. A dashboard VALUE row (`model.pullout_Nm`, `model.pullout_20C_Nm`, `metal.torque_cold_high_Nm`, `model.gearbox_input_ripple_Nm`, `model.cup_od_mm`, `mass.total_g`) that a reviewed note covers must link to that note. `mass.total_g` must stay unowned. | `src/gui/guidance.rs:2364-2400`; `src/gui/dashboard.rs:150-158`; `src/gui/panel.rs:7006-7010` |
| F14 | Diagram pins: the pull-out diagram paints at least one point (circle r = 3.5), and the non-ferromagnetic note's diagram paints none. A new `Diagram` variant needs changes in three places, and its labels must be distinct. | `src/gui/panel.rs:6965-6970, 7266-7269`; `notes.rs:14`; `src/gui/diagrams.rs:110-117, 485-492, 529-554` |
| F15 | Glyphs: the default egui fonts are used. `glyph_safe` maps only ϑ, U+207B and ∝, so write `->` and `<=`. Every note field and every input tooltip is glyph-tested. | `src/gui/typeset.rs:271-278`; `src/gui/panel.rs:7065-7079, 3946-4048` (tooltips at :4017) |
| F16 | `sources` is drawn on screen as "Sources: ...", and 15 of the 17 notes hit a banned pattern there. No test pins the drawn line, only `!sources.is_empty()`. | `src/gui/explorer.rs:104, 588-598` |
| F17 | Numbers in notes are untested literals about the default design. | map-locks 2.5 |

### 1.2 Input help and labels (HARD)

| # | Fact | Where |
|---|---|---|
| F18 | Each of the 161 ported inputs must match Python's label, unit, help, cell and choices byte for byte. All 87 empty helps are on ported inputs, so filling any of them in the engine fails the test. The only sanctioned reword path is an applied deviation's `workbook_help`, and a deviation is a physics correction with an audit row, not wording. The oracle is read-only. | `tests/python_schema.rs:72-91, 94-136`; `src/engine/deviations.rs:218-225`; `tests/deviations.rs:2004-2030`; `docs/ai/03-structure.yaml:12-13` |
| F19 | Help is read in exactly three places: the input tooltip, the two Size-tab slider hovers, and the result tooltip. It is not searched, not exported, and not in design files or links. | `src/gui/inputs.rs:907-940` (via `input_ui.rs:114-117, 140`); `src/gui/size_controls.rs:322`; `src/gui/dashboard.rs:404-420` |
| F20 | The full tooltip of `metal.face_gap_mm` is pinned, including "Same as the measured prototype.", the path, `Metal design!C119`, slider and default. The harmonic tooltip's "Rust-only input (no workbook cell)" and "Model assumption (Addendum A3)" are pinned too. | `src/gui/inputs.rs:1380-1406` (1400-1401) |
| F21 | `tests/data/input_schema.json` must equal the live metadata. Any engine help edit, even on a Rust-only input, needs `MAGCOUPLING_BLESS=1 cargo test --test schema`. | `tests/schema.rs:60-78` |
| F22 | Labels feed C-3 disambiguation, the rule that no two inputs give the same fix text, and the search haystack (label + path + cell). The query "gearbox" must match exactly 3 inputs. | `src/gui/guidance.rs:857-891, 2467-2477`; `src/gui/format.rs:33-35`; `src/gui/inputs.rs:1478-1496`; `src/gui/panel.rs:2485, 2519` |
| F23 | `SIZE_TAB_ALIASES` is the only GUI renaming mechanism. It has one entry: the ring radius maps to "Outside diameter" for fix text. | `src/gui/guidance.rs:154` |
| F24 | An assumption's `source` must contain the workbook cell of every input it sets, or "Addendum A3". Rationales are exempt from the plain-words check on purpose. | `tests/assumptions.rs:123-127`; `src/gui/dashboard.rs:2120-2133` |
| F25 | Verdict strings are Python-locked through the differential data. `verdict_level` matches them exactly, for example the prefix "Too thin: ". | `tests/differential.rs:74-160`; `src/gui/dashboard.rs:257-336` |

### 1.3 Guards, smoke, gate, process

| # | Fact | Where |
|---|---|---|
| F26 | The plain-words guard is `BANNED` (6 patterns), `plain_text_violations` and `ALLOWED_WORKBOOK = "Workbook example"`. It lives in the `gui` feature and runs only over guidance texts. Its clean list includes "ISO 4762 M4 x 14, class 12.9", so a blanket `M\d` ban is impossible. | `src/gui/guidance.rs:1442-1494, 3311-3323, 3326-3339` |
| F27 | Text pinned by tests: `inputs.rs:1400-1401`; `dashboard.rs:1474, 1484, 1500, 1504`; `panel.rs:2619` (the results table draws `Calculator!C93` once); `panel.rs:6701` ("Embodies corrections"); `explorer.rs:1170` and `panel.rs:6906` ("1, 3, 5 (workbook)"); `geometry.rs:1166-1167, 1192`; `corrections.rs:221-296`; `spreadsheet.rs:948`. | map-locks J1-J16; map-jargon 5.2 |
| F28 | gui-smoke checks the exact welcome line, the header buttons and the console prefixes. `app.rs` reads `gui-smoke.js` constants. The smoke run never opens a note or a tooltip. | `.claude/workflows/gui-smoke.js:83-88`; `src/app.rs:207-257` |
| F29 | The gate has 12 steps. Gate 4 runs engine tests with no gui feature. Gate 7 runs `--features app`, which is where gui guards run. Gate 12 needs `MAGCOUPLING_PYTHON`; there is no `.venv` in this checkout, so gate 12 otherwise SKIPs. About 6 minutes warm. | `linkage-sim-rs/scripts/gate.sh`; `AGENTS.md:24`; map-locks 8.3 |
| F30 | A push to main deploys with no CI tests, and nothing is pushed without the owner's go. Slice 1's G-8 authorization was for slice 1 only. | `.github/workflows/deploy-web.yml:37-46`; `AGENTS.md:26`; memory `project_magcoupler_guidance_overhaul.md` |
| F31 | A `risk: physics` item needs the physics reviewer pass AND the owner's explicit review of the diff before merge. A failing test comes first, and docs change with the code. | `AGENTS.md:22-23`; `docs/ai/backlog.yaml:1-5` |
| F32 | Model rule: physics text, its review, design and final reviews run on the session model. Mechanical work runs on sonnet. Never downgrade a retry. | `CLAUDE.md` section 5 |
| F33 | Slice-1 decisions still in force: G-3 (one session-model physics review per slice for hint text, numbers from results, no Draft gate for hints), G-5 (Explain off by default), G-6 (the "Workbook example design" label). | spec `2026-10-13-magcoupler-guidance-design.md:36, 38-39`; `src/gui/guidance.rs:1381` |
| F34 | Spec section 8 traps must not be written: radius as the strongest lever; fewer poles keep the size; pull-out guaranteed or re-lock below pull-out; calibration beyond the prototype; 416 as a safe fix; stored 3D fields follow the design. The physics pass adds about 30 more (section 5). | spec `:197-204` |
| F35 | Rustdoc: indented lines in doc comments become doctests, so use ```text fences. A release build fails on MAX_PATH from the long repo path. | `docs/ai/02-system.yaml:407-408, 451` |
| F36 | Docs that must change with the code: the README note count, module and test tables (`README.md:29, 41, 150-151, 318-321, 460-475, 577-597, 662`); `docs/ai/02-system.yaml:107, 113-114, 476`; `03-structure.yaml:61, 65, 72`; `04-memory.yaml:7, 11, 74, 82, 83`; `05-update-tracker.md`; `backlog.yaml` (a new BL after BL-055). | map-locks 9 |

---------------------------------------------------------------------------------------------------
## 2. Scope inventory

### 2.1 Notes to rewrite: all 17

All 17 carry visible jargon, in their sentences or their Sources line. Under decision 1-A each also
gains a "What to change" part, so every note is a text edit, goes back to Draft and needs a full
re-review (F3).

Density today: 61 sentence elements; median 31 words, mean 35.5, maximum 85. 31 elements are over 30
words, 17 over 40 and 11 over 50. Per-note totals run from 65 to 247 words (median 142). In this table,
"s" plus a number means sentence N of the note (s2 = sentence 2), and WO means its watch-out line.

| # | Note (Start here step / links) | Sev | Main problems (finding ids) |
|---|---|---|---|
| 1 | `harmonics` (step 1; title pinned) | H | 85- and 61-word sentences. s4 names the wrong main reason higher harmonics are small: the amplitude product (about 1/40) outweighs the gap fade. Says "workbook". Omits that at low fill the 3rd harmonic moves pull-out by 12-33 % and shifts its angle. Never ties fill to the user's levers (TC-4, VZ-9, phys-a 2). |
| 2 | `back_iron_factor` (step 5) | M | s1 misdefines S_n. Gives no magnitude for the steel gain (x1.76). Misses the e^(-πg/τ_p) gap rule and that S_1 has a best pole pitch; beware the trap "fewer poles reach across better". Silent that the torque assumes unsaturated steel while the default design fails the cup wall (TC-11, PA-9). |
| 3 | `pullout_angle` (step 2; dashboard Why for pull-out and ripple, `guidance.rs:305, 318`) | M | Never states the T_pull = T_2D·f_end·f_cal equation it opens. A_n and electrical angle are undefined, with no mechanical twist. s3 is incomplete: only a same-sign 3rd harmonic above A_1/9 moves the peak (PA-8). Says "(correction E7)". Says nothing about what happens after slip (TC-12). |
| 4 | `end_effect` (step 3; `guidance.rs:275`) | M | Accuracy is overstated: within 2 %, not 1 %, and only checked in free space (PA-14). Omits that it is optimistic at wide gaps (+11 % at 4.4 mm), which is where the slice-1 clearance fix sends users. Uses default literals 1.3 mm and 3 mm. Says "M1 skeptic" and "M1 audit M9". Gives no per-mm lever (TC-13). |
| 5 | `calibration` (step 4) | M | States editable defaults (0.95, 1.8 N·m, 20 °C) as constants. s3 is a reviewer's proof. Misses the 10.5 % step from the rule when the pole count changes without steel (TC-14). Misses that editing the prototype inputs gives a meaningless factor (1.57, +65 %). The watch-out is an audit trail. |
| 6 | `br_temperature` (step 6; `guidance.rs:306`) | M | The watch-out misleads: only rings with a picked grade keep their own α. The title renders as "torque ~ Br²" (TC-18). s4 front-loads demagnetization. "30 °C / 7 %" holds only at the defaults. Does not say which temperature each torque row uses (PA-4). |
| 7 | `demagnetization` (step 7; `guidance.rs:197`) | H | "863 kA/m here" presents a stored 3D default as the user's design (spec 8 trap 6; PA-1, TC-5). States the onset shift as fact, though it is non-conservative for N52, N50, N38UH, N35EH and SmCo (PA-16). Entries of 58, 53 and 57 words. Hcj, knee and permeance are undefined. No lever. Says "E20" and "M1 audit M13". |
| 8 | `ferrite_cold_demag` (step 8) | M | "336 °C cold limit" reads as a bug (TC-19). Omits that a typed positive β is clamped away (PA-2) and that Y30 data cover only 20-120 °C (PA-17). Its step opens a check that reads "n/a" on the NdFeB default (TC-32). States the margin as a constant. |
| 9 | `slip_heating` (step 9) | H | Silent that the stored slip fields never follow the design: 2.751 -> 5.397 W at a 15 mm apothem with the fields held (PA-1). 65-word opener. σ collides with shear stress (TC-15). States speed^1.5 as physics (PA-19). No lever: a measured drag replaces the estimate. |
| 10 | `thermal_time_constant` (step 10) | L | Writes τ and ϑ_start where the panel shows τ_th and ϑ_hot (M43-13, TC-20). Says "correction E12" and "M1 audit placeholder table". Does not explain the peak the check uses. The 0.54 °C margin must come from the results. |
| 11 | `clamp_preload` (step 11; `guidance.rs:209`) | M | Does not say why the clamp is sized to cold-high (TC-16). States editable defaults (0.75, 1.5, 2, μ 0.15) as fixed. Says "proof", but A4-70 uses yield stress. Misses the clamp boss vs drawn boss trap. Says "Addendum A3" and "M1 audit M14, E2". |
| 12 | `a5.non_ferromagnetic_back_iron` (red warning) | M | 74-word sentence. "More than 40 %" where the dashboard shows 37 % (TC-10). Mentions the calibration side effect without stating the rule. Misses the "Back iron = no back iron" path and its consequences. The audit's suggested fix (416) is a trap. |
| 13 | `a5.ferromagnetic_sleeve_or_liner` (warning never fires; TC-6 dropped) | L | Only its jargon matters to users. It implies a torque drop the model does not compute. |
| 14 | `a5.high_conductivity_sleeve_or_liner` (amber) | L | Presents stored shell fields as the design's own. States the 0.7 end factor as a constant. The threshold is a code constant equal to the 316L input's default. Uses a bare σ. |
| 15 | `a5.low_saturation` (amber; also the Why of the default design's red cup-wall check, `guidance.rs:188`) | H | Explain shows "Low saturation" for the wall rule on every design. Conflates design flux density with saturation (TC-17, PA-5). Implies the shown torque drops (PA-9). The "12-16 % thin" watch-out holds at the defaults only. Says "workbook" and "M1 audit M1". |
| 16 | `a5.uncoated_low_alloy_steel` (amber) | L | Overstates that 416 resists corrosion. Lists 304, which cannot be a back iron. Does not name its input (TC-21). Offers 416/17-4PH without the low-saturation trade-off. Says "workbook". |
| 17 | `a5.cte_mismatch_with_magnets` (amber) | L | Uses the NdFeB CTE for every grade (PA-13). Shows the "10¯⁶" glyph. Its "so" implies the flag counts the temperature swing. "Volkersen screen" names a method, not a results row. Silent that the modulus does not follow the selected adhesive. |

Totals: High 4, Medium 8, Low 5. Physics-reviewed drafts for all 17 exist in `phys-notes-a/b/c.md`
(about 20-26 words per sentence, each with a "what to change" table and a traps list).

### 2.2 New notes recommended: 3 (decision 3)

- **(a) `torque_from_shear`, "How the coupling makes torque".** Start here step 1. It owns
  `model.area_lever_m3`, `torque_2d_Nm`, `gap_radius_mm`, `pole_pitch_mm` and `active_length_mm`.
- **(b) `hot_low_cold_high`, "Hot-low and cold-high: the torque band".** It owns the hot-low, cold-high
  and hot-minimum rows, rewires `HOT_MIN_GUIDE.learn` (`guidance.rs:164`), and forces the cold-high
  ROW_LEARN entry (`guidance.rs:309-313`).
- **(d) `when_slipping`, "When the coupling slips".** It owns `model.gearbox_input_ripple_Nm`, which
  forces ROW_LEARN `guidance.rs:316-319`, and `metal.slip_freq_Hz`.
- **(c) the back-iron wall rule** is delivered by rewriting `a5.low_saturation` as "Back-iron wall and
  saturation". The cup-wall guide already links there, and the id is frozen.

Total after the slice: 20 notes, exactly at the cap (F6). Drafts are in `phys-new-notes.md` 3.1-3.5.

### 2.3 Input help

- **87 empty helps.** All are on ported, Python-locked inputs (F18).
  - By workflow group: Shaft, key and clamps 15; Calibration and model 9; Requirements and operating
    conditions 5; Magnets and rings 8; Gap and clearances 7; Materials 20; Housing and retainers 20;
    Thermal and demagnetization 3.
  - 68 are in ordinary sections and 19 under Advanced. 4 are model assumptions (`calibration.c_end`,
    `temperature.demag.knee_fraction`, `design_margin_C`, `clamps.preload_fraction`). 1 is in Key
    design (`coupling.magnets.part_outer`).
  - By package: metal 31; clamps 15; coupling 15 (7 plus 8 under magnets); temperature 12;
    calibration 8; materials 6.
  - `map-help.md` section 5 holds a verified "what it physically is" line for each.
  - Natural clusters that can share wording:
    - the six running-clearance allowances (one sum);
    - the manual block fields;
    - cap, endplate and retainer;
    - the six optional-adapter fields (mass and length only);
    - the eight bench-prototype fields;
    - the ten clamp-geometry fields;
    - the three specific heats;
    - the two screw-class stresses.
  - Material inputs apply only while their part picker is 1 (`material_library.rs:855-921`).
- **Non-empty help with jargon: 19 of 85.**
  - 10 are on Rust-only inputs, free to edit in the engine plus a re-bless.
  - 4 are on ported inputs already reworded through `workbook_help`. They are free if the new text
    differs from the recorded workbook text.
  - 5 are ported and locked: `shaft_displacement` and `magnet_position` ("UNCONFIRMED."),
    `cup_wall_corner`, `alpha_br` ("sheet"), and the steel `density` ("C132").
- **About 30 terse, stale or wrong ported helps** that are worth overriding: npole ("Even; one block per
  pole."), face gap ("Same as the measured prototype.", which is also the Size-tab hover),
  `required_min` ("At the 50 °C magnet temperature."), `beta_hcj` (tells users to type a positive value
  the GUI clamps, PA-2), the alpha/density/rating helps that are wrong in grade mode (PA-11), `b_cup_T`
  (PA-12), and the "3D.", "User supplied.", "Placeholder." and "UNCONFIRMED." family.
- **Result help with jargon:** 83 declarations in total (map-jargon Appendix A), mostly Rust-only
  results such as "Plan A-3", "E7" and "until M3 computes it live". Only one is a dashboard row:
  "End-effect model check", whose help cites "Audit M9".

### 2.4 On-screen jargon by group

The map-jargon inventory has 85 rows:

| Group | Rows | Examples |
|---|---|---|
| Radius and size names | 13 (R-1..R-11, N-1, N-2) | "Inner magnet back apothem, including bondline" / "Ring radius" / "ring size" / "Outside diameter"; "apothem" in verdicts; "Number of poles per ring" against "Magnets per ring" |
| Torque names | 16 (T-1..T-16) | "Required minimum service pull-out torque", "Hot-side low torque including assumed variation", "Below hot minimum" naming two different comparisons, unit "Nm" against "N·m" |
| Workbook and cell references | 24 (W-1..W-24) | tooltip lines `Calculator!C93` and `model.pullout_Nm`; "Rust-only ..."; "3D values from the workbook ... until M3"; "Reset to workbook defaults"; "Workbook groups"; 14 assumption sources; "Too thin: raise Metal design C122"; "1, 3, 5 (workbook)" |
| Audit and spec ids, badges | 11 (A-1..A-11) | `E3 E7 E8` beside 14 of the 17 dashboard values at the defaults; the "Corrected vs workbook" hover with a repo path; "Embodies corrections"; the note Sources line; "Model assumption (Addendum A3)"; "Autofit" |
| Code and file names | 11 (C-1..C-11) | "drawing.py's only layout"; dotted paths; "equation record"; "The differential tests cover the slider range only."; `f_end` in the banner; "(refresh after a 3D run)" |
| Other | 10 (O-1..O-10) | four names for the face gap; "candidate" and "prototype"; two "Verdict" rows; σ/τ/θ symbol collisions; undefined domain terms; `+inf`; note titles |

Inside the notes: 15 sentence or watch-out lines and 19 Sources lines carry jargon (map-jargon
Appendix C).

Cost tiers for the GUI strings (map-locks J1-J16):
- **Cheap** (a constant or one test edit): J1-J3, J6-J10, J12, J13, J15, J16.
- **Wide** (touches search, markers or exports): J4 (cells and paths), J5 (E-badges), J11 ("Rust-only"
  table column).
- **Collides with a test:** J14 (assumption sources, F24).

Unsafe suggestion from the audit: UI-50's "Guaranteed-minimum torque" breaks spec section 8.

---------------------------------------------------------------------------------------------------
## 3. Decisions for the owner

Each decision lists its options with the recommended option first, then effort, risk and impact. Only
decisions that change what gets built are listed. Defaults I would take without asking are in 3.11.

### 1. Note format, structure and length budget

Context: a note holds 2-6 sentence elements and one optional watch-out. It has no slot for "what to
change" (`notes.rs:45-57, 636`). Density is a median of 31 words and a maximum of 85.

- **1-A (recommended). Three visible parts, with provenance folded away, through a struct change.**
  - **What it is:** `sentences`, 3-5 entries of at most 25 words, one idea each. Sentence 0 answers the
    question of the dashboard row that links to the note, in plain words, with no symbols and no
    placeholders (F10).
  - **What to change:** a new field `change: &[(&str /*input path*/, &str /*direction -> effect; trade-off*/)]`
    with 1-5 entries.
    - The input label is drawn at display time from `InputCatalogue` and the label overlay, so no
      label is ever typed (slice-1 rule, spec 4.1).
    - Each entry is a link that focuses the input row.
  - **Watch out:** `watch_out: &[&str]`, 0-2 entries of at most 30 words.
  - **Where this comes from:** `sources`, drawn under a fold that starts closed.
  - Formulas appear only where they carry the idea, because the equation already sits above the note
    in Explain.
  - Caps are enforced by tests written red with the drafts and un-ignored once the set is reviewed
    (TC-34 advises setting caps after the drafts).
  - **Effort:** medium. The struct, 17 literals, `note_ui`/`note_alone_ui`, the focus link and its tests.
  - **Risk:** low to medium. `Note` is a const struct, and sentence 0 is counted raw.
  - **Impact:** `notes.rs`, `explorer.rs` and panel tests. Labels cannot go stale (DRY), and every note
    says what to change in a consistent place.
- **1-B. The same three parts inside today's struct.**
  - The last 1-2 sentences say "To change: ..." with labels typed as text. The watch-out stays one
    string. A test checks that each quoted label exists.
  - **Effort:** low.
  - **Risk:** medium. Labels are typed twice (DRY), there are no links, and the 6-entry cap squeezes the
    drafts, which already use 5-6 entries.
  - **Impact:** `notes.rs` only.
- **1-C. Plain summary always shown, details folded (VZ-9).**
  - Formulas, caveats and sources go inside a "Details" fold.
  - **Effort:** medium.
  - **Risk:** high for accuracy. The caveats the reviewers insist stay visible would hide in the fold:
    the stored 3D fields, wide-gap optimism and the calibration step (TC-5 check). It still needs a
    what-to-change slot.

### 2. Numbers inside notes: live values or literals

Context: the notes quote default-design numbers that no test checks (F17). The physics drafts are
written with `{path}` placeholders ("never substitute the default value", phys-a:12). The TC-8 check
warns that qualitative words around a live number can become false off-default.

- **2-A (recommended). Live placeholders.**
  - A `{result.path}` in a sentence, watch-out or change text is rendered from the current results
    with `format_value`. A non-finite value, or an end effect out of range, renders as "no prediction".
  - Placeholders never appear in sentence 0 or in titles. They appear only in "here: X" clauses, never
    inside a comparative claim.
  - **Tests:**
    - every placeholder is a result path at the default design;
    - the text rendered at the default design and at four probe designs (8 poles; 304 back iron; Y30 on
      custom dimensions; a 20 mm ring radius) contains no "{";
    - that rendered text passes the plain-words and glyph guards.
  - The reviewer reads each note rendered at the same five designs.
  - **Effort:** medium. A pure-std parser in the engine, the renderer, and tests.
  - **Risk:** medium. Wording can drift off-default; the five-design review is the mitigation.
  - **Impact:** notes speak about the user's own design and fix "863 kA/m here" (T6).
- **2-B. No live numbers.**
  - Editable defaults are written as "(X by default)" with the on-screen label. Validation figures are
    scoped ("in a free-space 3D check at 10 poles and a 1.4 mm gap"). "Here" is never used.
  - **Effort:** low.
  - **Risk:** low.
  - **Impact:** notes stay generic, and users read their own numbers off the dashboard. About 25
    placeholder sites in the drafts must be reworded.
- **2-C. Literals now, placeholders in a later slice.**
  - **Effort:** low now, plus a second review of every note later.
  - **Risk:** low.
  - **Impact:** double review cost.

### 3. Which three new notes, and the note cap

Context: the cap is 15..=20 and there are 17 notes (F6). The owner named four candidates and said
"three".

- **3-A (recommended, R1).**
  - New notes: (a) "How the coupling makes torque" (Start here step 1), (b) "Hot-low and cold-high: the
    torque band", and (d) "When the coupling slips".
  - (c) the wall rule comes from rewriting `a5.low_saturation` as "Back-iron wall and saturation". The
    cup-wall guide already links there (`guidance.rs:188`), and the id is frozen (F9).
  - **Effort:** 3 notes plus 1 rewrite that is already planned. Wiring: START_HERE,
    `explorer.rs:1084/486`, `HOT_MIN_GUIDE.learn` and two ROW_LEARN rows.
  - **Risk:** medium. (d) states un-modelled behaviour (re-locking) qualitatively, and the a5 note
    carries both the warning and the wall rule within its budget.
  - **Impact:** the cap sits at exactly 20, with no headroom.
- **3-B (R2).**
  - New notes (a), (b), and (c) on its own as "How thick the cup wall must be". (d) is spread across a
    `pullout_angle` watch-out, a `slip_heating` sentence and a `clamp_preload` sentence.
  - **Effort:** the same as 3-A.
  - **Risk:** medium. Slip is still scattered, so the CG-13 complaint stays unfixed, and the Start-here
    pull-out note grows.
- **3-C (R3).**
  - All four notes. Raise the cap to 15..=21 and amend spec A4 ("about 15 to 20"). The strings to
    change are `notes.rs:59-61` and `:631-633`, `README.md:29/662` and `03-structure.yaml:61`.
  - **Effort:** one more note review.
  - **Risk:** low.
  - **Impact:** the cleanest result, with one idea per note, but it departs from both "three new notes"
    and A4.

### 4. Canonical names for the ring radius and the torques, and how they reach the screen

Context: one radius has four names on screen. They are the Python-locked input label (`model.rs:167`),
the Size-tab "Outside diameter" (`size_controls.rs:26`), "Ring radius" and "ring size" in Auto
(`sizing.rs:84, 94`), and "apothem" in verdicts. Labels are Python-locked (F18). The only renaming
mechanism is `SIZE_TAB_ALIASES` (F23).

The canonical set is the same in every option:

| Quantity | Name on screen | Notes |
|---|---|---|
| `coupling.inner_back_apothem_mm` | **Ring radius** | the shaft centre to the back of the inner magnets |
| Size-tab control | **Outside diameter** | the largest of cup, cap and boss; it follows the ring radius |
| `coupling.npole` | **Magnets per ring** | |
| `metal.face_gap_mm` | **Magnet face gap** | |
| `model.gap_radius_mm` | **Gap radius R_g** | |
| `metal.required_min_Nm` | **Required torque** | |
| `model.pullout_Nm` | **Pull-out torque (nominal)** | |
| `metal.torque_hot_low_Nm` | **Hot-low torque** | |
| `metal.torque_cold_high_Nm` | **Cold-high torque** | |
| `metal.variation` | **Production variation allowance** | |
| the workbook | **"the original spreadsheet"** | |

No torque is ever called "guaranteed".

- **4-A (recommended). A display-label overlay for a short listed set.**
  - It covers about 6 inputs and 3 dashboard results, and shares one `gui/plain_text.rs` table with the
    help overlay.
  - The display label is used on input rows, in fix text, in equation term names and on the dashboard.
    Search matches the display label, the workbook label, the path and the cell. Exports keep the
    workbook labels.
  - GUI-owned strings are renamed, for example "ring size" to "ring radius".
  - **Effort:** medium.
    - Five label read sites: `input_ui.rs:115`, `inputs.rs:685`, `guidance.rs:868-871`,
      `explorer.rs:373`, and the dashboard row label.
    - Pinned literals: `panel.rs:5446`, `sizing.rs:797`, `guidance.rs:3850-3859, 4537`,
      `size_controls.rs:528`, `trace.rs:144`.
    - C-3 uniqueness then runs on display labels.
  - **Risk:** medium. Screen and export names differ, explicitly and by design.
  - **Impact:** one name per quantity on screen.
- **4-B. GUI-owned strings and hovers only.**
  - Workbook labels stay on input rows and the dashboard. The help says, for example, "called Ring
    radius on the Size tab".
  - **Effort:** low.
  - **Risk:** the input row still reads "Inner magnet back apothem, including bondline", so four names
    become three.
- **4-C. One name for both controls.**
  - The Size-tab slider becomes "Ring radius", in radius millimetres, and Outside diameter becomes a
    readout.
  - **Effort:** high. It redoes the slice-1 Size tab, its tests and the Auto choices.
  - **Risk:** it reverses UI the owner approved yesterday.

### 5. Provenance on screen: workbook cells, paths, E-badges, note Sources, assumption Sources

Context:
- Every tooltip shows the dotted path and the cell (`inputs.rs:913-917`, `dashboard.rs:412-417`).
- 14 of the 17 dashboard rows carry E-badges at the defaults (`dashboard.rs:563-569`). Their hover
  cites repo paths (`corrections.rs:175-195`).
- Note Sources are drawn (F16). Assumption Source lines are drawn (`panel.rs:1171-1176`) and must keep
  their cells (F24).
- Exports keep everything (`spreadsheet.rs:53, 68, 80, 377-383`).

- **5-A (recommended). One header toggle, "Show spreadsheet details".**
  - It is off by default and GUI-only: not saved, not in links, not in undo.
  - It gates every provenance line:
    - the tooltip path and cell lines;
    - the results-table cell and marker columns;
    - the dashboard E-badges, with a one-line legend when the toggle is on;
    - the explorer's "(Calculator!C93)" and "Embodies corrections" lines, with titles on hover (TC-28);
    - the assumption Source lines.
  - Note Sources sit under the closed fold from decision 1 either way.
  - The marker hover becomes "Differs from the original spreadsheet: <title>, was X, now Y", with no
    repo path.
  - Search still matches cells and paths. Exports do not change.
  - **Sub-point needing your yes or no:** rename "Workbook example design" to "Example design", with
    the hover "the starting values from the original spreadsheet". This amends G-6 (spec:39, F33).
  - **Effort:** medium. One bool, about 8 drawing sites, and about 12 pinned tests that switch the
    toggle on.
  - **Risk:** low.
  - **Impact:** traceability stays one click away for owner and reviewer, and the noise is gone for
    users.
- **5-B. Remove provenance from the screen entirely; the exports are the record.**
  - **Effort:** low to medium.
  - **Risk:** reviewers lose the on-screen link to cells. A search for "C93" still matches, but the user
    cannot see why. The E-ids become invisible.
- **5-C. Keep it visible with plain labels.**
  - For example "Original spreadsheet cell: ...", plus a legend for the E-ids.
  - **Effort:** low.
  - **Risk:** the noise stays on every hover and on 14 dashboard rows, so the complaint is only half
    fixed.

### 6. The Draft/Reviewed gate: who reviews, and what counts as Reviewed

Context: a Draft note is hidden and turns about 10 tests red (F1, F2). The review record is free text,
and `.superpowers/` is gitignored (F4). `risk: physics` needs the reviewer AND the owner's diff review
(F31). `release_notes_are_reviewed` is ignored (F5).

- **6-A (recommended). A per-note gate inside each note task.**
  - **Flow:**
    1. Write the draft.
    2. A fresh-context session-model physics reviewer, never sonnet, checks it against the engine at
       the five designs of decision 2 and against the spec section 8 traps plus the new trap list.
    3. Apply the fixes, then re-review the changed sentences.
    4. In the same commit, flip the note to
       `Reviewed { reviewer: "physics reviewer (session model)", date, record: "docs/analyses/2026-10-xx-magcoupler-notes-review.md, <id>" }`.
       The report is committed.
  - **"Reviewed" means:**
    - the reviewer returns PASS with no open must-fix;
    - every number is reproduced from `compute_all`;
    - every "what to change" direction is confirmed by a probe.
  - Un-ignore `release_notes_are_reviewed`, and widen it to every note linked from START_HERE,
    `CHECK_GUIDES`, `ROW_LEARN` and the warnings.
  - Before merge you sign off a generated review sheet that shows old against new text for each note
    (`AGENTS.md:22`).
  - Help text, display labels, the verdict display map and other GUI text follow G-3: one session-model
    review per batch and no Draft gate.
  - **Effort:** 20 reviews, each with one fix round. This uses a lot of usage.
  - **Risk:** low.
  - **Impact:** every commit leaves the gate green.
- **6-B. As 6-A, plus an independent second review.**
  - Codex reviews from the WSL worktree, as in the 2026-10-06 handover, for the 3 new notes and the 4
    High notes.
  - **Effort:** 7 more reviews plus the handover.
  - **Risk:** the lowest.
  - **Impact:** slower.
- **6-C. One review of all 20 notes at the end.**
  - **Effort:** the lowest.
  - **Risk:** high. Until then the commits would carry Reviewed records that describe other text, and a
    late failure blocks the whole release.

### 7. Help text: style, scope, and whether it stays hover-only

Context: 87 of 172 inputs have no help, and every help string is hover-only (F19). Terse workbook help
sits on the two Size-tab hovers.

The style is the same in every option:
- 1-2 sentences, at most 30 words: what the input physically is, which results it moves and in which
  direction, and the trade-off.
- Say so when a part picker overrides the input, or when Auto holds it.
- For calibration inputs, say they describe the bench prototype, not your design. For the
  optional-adapter fields, say they change mass only.
- No default numbers, because the tooltip already prints "Default:". No spreadsheet or audit ids.

- **7-A (recommended).**
  - A GUI overlay replaces the help on screen for the 87 empty inputs and for about 30 non-empty ones
    that are jargon, terse, stale or wrong (2.3). Under the toggle from decision 5, the workbook text
    stays visible as "Original spreadsheet note".
  - Rust-only helps are rewritten in the engine, with a re-bless.
  - Hover stays the main surface. In addition, one always-visible weak hint line sits under the 10
    Key design inputs (`inputs.rs:27-38`) and the 5 Size-tab controls.
  - **Effort:** about 120 strings in 4 batches, plus tests.
  - **Risk:** medium. Each "raises X" is a physics claim, covered by the G-3 review.
  - **Impact:** every input explains itself, and the npole and face-gap hovers are fixed.
- **7-B.**
  - The overlay fills only the 87 empty helps. The non-empty workbook help stays, and help stays
    hover-only.
  - **Effort:** lower.
  - **Risk:** the Size-tab hovers keep "Same as the measured prototype." and "Even; one block per
    pole.". The stale "At the 50 °C" stays, and so does the advice to type a positive β that the GUI
    clamps away (PA-2).
- **7-C. 7-A plus inline help under every input row.**
  - **Effort:** 7-A plus layout work.
  - **Risk:** Detailed inputs grow much taller, the side panel widens (UI-34), and glyph and layout
    tests churn.

### 8. Does slice 2 also make the notes reachable?

Context:
- Explain is off by default (G-5, A4, M43-2).
- Nothing links from the Size tab, the input rows or the end-effect banner to a note.
- NO_NOTE is a dead end (`explorer.rs:98`).
- Six notes are reachable only through Start here or Explain.

- **8-A (recommended). A small, code-only bundle.**
  - Start here is reordered with the torque note first, and the ferrite step reads "(ferrite rings
    only)" (TC-32).
  - The Equation panel's empty text mentions Start here and Explain (TC-29b).
  - NO_NOTE offers the nearest upstream reviewed note, or the assumption rationale (TC-29a, TC-30).
  - New "Why" links:
    - the Magnets per ring and Outside diameter lines lead to the torque note, and the link text is not
      the bare step-1 title (F11);
    - Required and Hot-low lead to the hot-low note;
    - the end-effect banner leads to `end_effect` (CG-10).
  - Explain stays off by default.
  - **Effort:** low to medium, with a headless test per link.
  - **Risk:** low. G-3 review of the hint text.
  - **Impact:** the rewrites are reached where the 10 -> 8 confusion happens.
- **8-B. Only what the new notes force.**
  - That is the Start-here order and the dashboard links the tests require.
  - **Effort:** the lowest.
  - **Risk:** most users never open the rewritten notes. The audit put it this way: approach A "only
    pays off once B makes the notes reachable".
- **8-C. 8-A, plus Explain on by default, note links on input rows, `see_also` links between notes
  (TC-26), and see-plot links (TC-25).**
  - **Effort:** high. It adds two more `Note` fields.
  - **Risk:** it reverses A4, M43-2 and G-5.

### 9. Engine physics gaps the notes would otherwise have to caveat

Context, from the physics pass (outside the note text):
- E21, the calibration offset clamp, is approved but not landed. Negative offsets raise onsets:
  N38UH +7.3 °C, N52 +9.7 °C.
- E22 is per-grade bond-plane CTE. Today the NdFeB value is used for every grade (PA-13).
- Cold-high at the gearbox input is 0.805 N·m against the 0.6 N·m rating, and nothing checks it (T-G).
- The clamp's "Boss outside diameter" (25 mm) is unlinked from the drawn "Shaft boss outside diameter"
  (22 mm).
- The Y30 coefficient range is never read (PA-17).
- The stored 3D fields are frozen (PA-1).

- **9-A (recommended). Slice 2 stays text and GUI only.**
  - Notes and help describe current behaviour, with explicit watch-outs.
  - Open separate `risk: physics` backlog items for E21, E22, a gearbox cold-high check and the
    clamp-boss link. The gearbox check needs your answer: is 0.6 N·m the right limit?
  - A note that a later fix changes is edited in that fix's commit.
  - **Effort:** none in this slice.
  - **Risk:** the demagnetization and CTE notes carry caveats that a later correction removes, which
    means a second Draft cycle for those two notes.
  - **Impact:** no registry, golden-file or parity change.
- **9-B. Land E21 and E22 first, on this branch.**
  - That means registry entries, audit rows, probes and a review. Then write those two notes against the
    corrected engine.
  - **Effort:** two more physics items.
  - **Risk:** engine numbers change inside a text slice.
  - **Impact:** a simpler demagnetization note.
- **9-C. 9-B plus a new gearbox cold-high dashboard check, with its guide.**
  - **Effort:** high.
  - **Risk:** needs a requirement decision first.

### 10. Release and deploy authorization for slice 2

Context: a push to main deploys with no CI tests, and slice 1's G-8 covered slice 1 only (F30).

- **10-A (recommended). As in G-8, with one addition.**
  - Merge to main and push (which deploys) once all of these hold:
    - `gate.sh` ends GATE PASS, with `MAGCOUPLING_PYTHON` set so gate 12 runs rather than SKIPs;
    - every note review and the whole-branch review pass;
    - gui-smoke passes;
    - you have signed off the note review sheet (`AGENTS.md:22`).
  - Then check the deploy run.
  - **Effort:** none extra.
  - **Risk:** low.
- **10-B. Two releases.**
  - First, the infrastructure, help, vocabulary and jargon work deploys on gate pass. Second, the notes
    deploy after their reviews and your sign-off.
  - **Effort:** two gate and smoke runs.
  - **Risk:** lower per release. The help ships sooner, and the vocabulary the notes quote is already
    live.
- **10-C. No standing authorization.**
  - Stop at GATE PASS with the branch ready, and you push.
  - **Effort:** none.
  - **Risk:** none.
  - **Impact:** slower.

### 3.11 Rulings I would make (defaults; tell me if you object)

- **R1. Help lives in a GUI-side overlay.** It is keyed by path, in `gui/plain_text.rs`. Engine metadata,
  `python_schema.json`, the oracle and the deviation registry are never touched for wording (F18).
  Rust-only helps are edited in place, and `input_schema.json` is re-blessed. Add a coverage test: every
  input has help from the engine or the overlay, and no overlay key names a missing path.
- **R2. One plain-words guard, for DRY.**
  - Move `BANNED` and `plain_text_violations` (`guidance.rs:1442-1494`) to a pure-std engine module.
    That lets gate 4 test the notes, and `gui::guidance` re-exports it.
  - Extend the patterns with targeted entries, never a blanket `M\d` (F26):
    - "M1 audit", "skeptic", "Plan A-", `decision \d`, "drawing.py", `\.rs\b`;
    - "Metal design", "Temperature design", "Shaft clamps", "Calibration!";
    - `correction E\d`;
    - whole-word "M3", with a whitelist for screws and threads (M2.5-M6, M41 × 0.5) and grade names
      (B842SH, N42SH, Y30, M5045).
  - Apply the guard to:
    - every note field and every overlay string;
    - every drawn tooltip with the toggle off;
    - assumption rationales (drop the exemption at `dashboard.rs:2120-2133`);
    - warning texts and explorer strings.
- **R3. Equation ownership, one note per equation (F8).**
  - `harmonics`: the fill rows.
  - `torque_from_shear`: area_lever, torque_2d, gap_radius, pole_pitch and active_length.
  - `hot_low_cold_high`: hot-low, cold-high, hot_min_check, hot_margin, the required_20C rows, the
    cold-for-hot-min rows, torque_op and torque_20C.
  - `br_temperature`: keeps `metal.torque_cold_Nm` and adds the three alpha records.
  - `clamp_preload`: adds `clamps.max_torque_Nm` and `required_Nm`.
  - `when_slipping`: the gearbox ripple and `metal.slip_freq_Hz`.
  - `slip_heating`: the duty frequency rows.
  - `thermal_time_constant`: `magnet_life.peak_C`, `steady_*_C` and the rise-per-event rows.
  - `demagnetization`: the onset margins.
  - `a5.low_saturation`: adds `cup_wall_check`, `cup_wall_suggested_mm` and `gap_flux_density_T`.
  - `a5.high_conductivity_sleeve_or_liner`: adds `materials.sleeve_sigma_S_m`.
  - Under 3-B or 3-C, `pullout_angle` keeps the ripple instead. `mass.total_g` stays unowned.
- **R4. Start-here order, 14 steps.**
  1. torque_from_shear
  2. harmonics
  3. back_iron_factor
  4. pullout_angle
  5. end_effect
  6. calibration
  7. br_temperature
  8. hot_low_cold_high
  9. when_slipping
  10. slip_heating
  11. thermal_time_constant
  12. demagnetization
  13. ferrite_cold_demag
  14. clamp_preload

  Update the tooltip and the pinned title (`explorer.rs:486, 1084`). This order avoids the forward
  reference that TC-3's order creates by putting pull-out before harmonics.
- **R5. Titles** use the physics drafts' plain titles. Each is unique against every drawn dashboard
  label (F12) and contains no ∝, ϑ or superscript minus.
- **R6. Symbols match the equation panel.** σ means shear stress only; conductivity is written in words,
  or as σ_st and σ_sl. Use τ_th and ϑ_hot. This closes M43-13 (`04-memory.yaml:83`).
- **R7. Diagrams.**
  - Fix TC-22: draw the fundamental dashed and keep the summed curve, so the point-count pin still
    holds (F14).
  - Fix the TC-24 labels.
  - The ferrite note drops the NdFeB DemagKnee diagram, which teaches the wrong direction.
  - No new Diagram variants; TC-23 waits for the lessons slice.
- **R8. No glossary type.** Each term is glossed in plain words at its first use along Start here, and
  the reviewer checks it.
- **R9. Dashboard verdict display map, for the cup-wall verdict only.** "Too thin: raise Metal design
  C122 ..." becomes the on-screen input label, with the number parsed from the engine text after
  `verdict_level` has read it (F25). Other verdicts keep the engine wording (slice-1 spec 3.4).
- **R10. Assumption rationales.**
  - Rewrite the 8 that carry jargon.
  - Fix `end_effect`'s "within 1 %" to "within 2 % of free-space 3D for 3.2 mm and longer"
    (`assumptions.rs:59`).
  - Fix `calibration_factor`'s "own rings" (`assumptions.rs:66`, PA-18).
  - `source` keeps its cells, behind the toggle (F24).
- **R11. Warning tests pinned to the library (phys-c D8).**
  - low_saturation fires only for back-iron codes {3,4,5,6}.
  - cte fires for {7,8} and for Back iron 0.
  - No sleeve choice fires the ferromagnetic or the high-conductivity warning.
  - uncoated fires for {1,2,3} at a plating thickness of 0.
- **R12. Process.**
  - Open BL-056 (`risk: physics`).
  - Write a slice-2 spec that amends A4 for the new audience and format.
  - Work in a worktree on a short path (F35).
  - Docs change with every task.
- **R13. `a5.ferromagnetic_sleeve_or_liner`** gets the shortest possible rewrite (its warning never
  fires): clear its jargon, and add "the torque model has no term for it".
- **R14. Smoke test.**
  - The welcome strip and the header buttons stay unchanged (F28).
  - Extend the smoke prompt to open one note through a Why link and one input tooltip.
  - Check dashboard scrolling at 1280x800.

---------------------------------------------------------------------------------------------------
## 4. Proposed task breakdown, assuming the recommended options

Model labels follow `CLAUDE.md` section 5. "sonnet" means the plan gives the exact code. "session"
means physics text, its review, or design judgment. Notes are one file, so the note tasks run in
sequence. Each task carries its own doc edits (F36) and writes a failing test before the change.

| Task | What | Model | Depends on | Files |
|---|---|---|---|---|
| T0 | Slice-2 spec (amends A4) and plan; the owner rules on decisions 1-10; open BL-056; short-path worktree | session | none | `docs/superpowers/specs/`, `docs/superpowers/plans/`, `docs/ai/backlog.yaml` |
| T1 | Move the plain-words guard to a pure-std engine module and re-export it; extend `BANNED` with the R2 patterns; update the guard's own example tests | sonnet | T0 | `src/engine/explain/mod.rs`, new `src/engine/explain/plain.rs`, `src/gui/guidance.rs` (1442-1494, 3286-3339) |
| T2 | `gui/plain_text.rs` overlay mechanism: `help_for` and `label_for`; wire the 3 help sites and 5 label sites; search on both labels; C-3 on display labels; coverage, plain-words and glyph tests. No content beyond seed rows. | sonnet | T1 | new `src/gui/plain_text.rs`, `src/gui/mod.rs`, `inputs.rs`, `input_ui.rs`, `size_controls.rs`, `dashboard.rs`, `guidance.rs`, `explorer.rs` |
| T3 | Provenance toggle (5-A): header toggle, gate about 8 drawing sites, legend, reworded marker hover; switch the toggle on in the pinned tests | sonnet | T1 | `panel.rs`, `inputs.rs`, `dashboard.rs`, `results_table.rs`, `explorer.rs`, `corrections.rs` |
| T4 | `Note` struct (1-A): add `change`, turn `watch_out` into a slice, put Sources under a fold, render change entries as focus links (labels via T2). The 17 literals change mechanically with their text unchanged, so they stay Reviewed. Add cap tests marked `#[ignore]`. | sonnet | T2 | `src/engine/explain/notes.rs`, `src/gui/explorer.rs`, panel tests |
| T5 | Placeholders (2-A): pure-std segment parser, renderer, non-finite as words; tests at five designs; no placeholder in sentence 0 | sonnet | T4 | `notes.rs` (or new `explain/note_text.rs`), `explorer.rs` |
| T6 | Vocabulary (4-A): display labels for the listed set; rename GUI-owned strings (ring size, Auto dropdown, Size-tab footnote O-7, Outside diameter hover, plot legends, sweep "Nm"); update the pinned literals | session (hover wording), sonnet (renames) | T2 | `plain_text.rs`, `sizing.rs`, `size_controls.rs`, `guidance.rs`, `plots.rs`, `src/engine/sweeps.rs` |
| T7a-d | Help content (7-A) in 4 batches: (a) shaft/clamps and requirements; (b) magnets, gap and calibration; (c) materials; (d) housing, thermal and the non-empty overrides. Inline hints for Key design and the Size tab. G-3 review per batch. | session | T2, T6 | `plain_text.rs`, `input_ui.rs` (hint line), `size_controls.rs` |
| T8 | Rust-only help and label rewrites (10 inputs, the 4 reworded ports, about 70 result helps from Appendix A); "1, 3, 5 (default)"; re-bless `input_schema.json` | session | T1 | `src/engine/{model,materials,temperature,housing,calibration,clamps,metal_design,warnings}.rs`, `tests/data/input_schema.json` |
| T9 | Cheap GUI jargon strings: J1-J3, J6-J10, J12, J13, J15, C-4..C-6, C-8, C-11, A-10 "Autofit"; the R9 verdict map; "Example design" if 5-A's sub-point is yes; the stored-3D wording | sonnet (renames), session (stored-3D and verdict wording) | T3 | `panel.rs`, `inputs.rs`, `dashboard.rs`, `explorer.rs`, `geometry.rs`, `clamp_drawing.rs`, `input_ui.rs`, `trace.rs`, `src/engine/warnings.rs` |
| T10 | Assumption rationales (R10); drop the plain-words exemption | session | T1 | `src/engine/assumptions.rs`, `src/gui/dashboard.rs` |
| T11 | New note (a), `torque_from_shear`: START_HERE[0], `explorer.rs:1084/486`; per-note review | session | T4, T5, T6 | `notes.rs`, `explorer.rs`, `docs/analyses/<review>.md` |
| T12 | Rewrite harmonics, back_iron_factor and pullout_angle; TC-22/TC-24 diagram fixes; per-note reviews | session | T11 | `notes.rs`, `diagrams.rs`, review report |
| T13 | Rewrite end_effect and calibration; per-note reviews | session | T12 | `notes.rs`, review report |
| T14 | New note (b) plus the br_temperature rewrite; `HOT_MIN_GUIDE.learn`, ROW_LEARN cold-high, START_HERE; reviews | session | T13 | `notes.rs`, `guidance.rs` (164, 309-313), review report |
| T15 | New note (d) plus the slip_heating and thermal_time_constant rewrites; ROW_LEARN ripple; HeatingCurve label; reviews | session | T14 | `notes.rs`, `guidance.rs` (316-319), `diagrams.rs`, review report |
| T16 | Rewrite demagnetization, ferrite_cold_demag and clamp_preload; the ferrite step label; reviews | session | T15 | `notes.rs`, `explorer.rs` (`start_here_text`), review report |
| T17 | The six a5 notes, low_saturation becoming "Back-iron wall and saturation"; R11 library-pinned warning tests (sonnet sub-step); reviews | session (+ sonnet tests) | T16 | `notes.rs`, `src/engine/warnings.rs` tests, review report |
| T18 | Un-ignore the cap and plain-words note tests and `release_notes_are_reviewed`; widen it to every linked note | sonnet | T17 | `notes.rs` tests, `tests/explain.rs` |
| T19 | Reachability (8-A): empty text, the NO_NOTE next step, Size-tab and banner Why links, START_HERE order check | sonnet (code), session (hint wording via G-3) | T11, T14 | `explorer.rs`, `size_controls.rs`, `panel.rs`, `dashboard.rs` |
| T20 | Docs consistency pass: README counts, tables and test index; `docs/ai` 02-05; tracker entry; smoke prompt (R14) | sonnet | all | `README.md`, `docs/ai/*`, `.claude/workflows/gui-smoke.js` |
| T21 | Release: generated note review sheet; whole-branch review; gate.sh with `MAGCOUPLING_PYTHON`; web build and gui-smoke; owner sign-off; deploy per decision 10 | session (review), sonnet (runs) | all | none |

Review gates:
- **Per task:** a fresh-context review of the diff. Sonnet reviews the mechanical tasks (T1-T5, T18,
  T20); the session model reviews everything with physics text.
- **Per note:** a session-model physics review inside each task T11-T17, with the report committed
  before the flip to Reviewed (6-A).
- **G-3 batch review:** for help, labels and GUI hint text (T6-T10, T19).
- **Whole branch:** a session-model review before T21.
- **gate.sh:** GATE PASS at the end of every task. Gates 4 and 7 cover engine and GUI; gate 12 needs
  `MAGCOUPLING_PYTHON`.
- **gui-smoke:** before merge, because `src/gui/` changed.
- **Owner:** explicit review of the note diff (`AGENTS.md:22`).

Parallelism: T2, T3, T8 and T10 can run in parallel after T1, since they touch disjoint files. T7a-d
run in sequence because they share one file. T11-T17 run strictly in sequence because they share
`notes.rs`.

---------------------------------------------------------------------------------------------------
## 5. Risks: what could go wrong for a user

**1. Hidden or blocked notes.** A rewritten note left Draft disappears, and Why links show "Its teaching
note has not passed review." (F2). A late review failure either blocks the release or tempts someone to
ship stale Reviewed records. Mitigation: the per-note gate inside each task (6-A).

**2. New physics errors from simplifying.** The verified traps a plain rewrite tends to fall into:
- "Fewer poles reach across the gap better": net torque falls, and S_1 peaks near the default pitch.
- "Fewer poles keep the size": the cup grows, and Outside diameter is set by the cap.
- "Radius is the strongest lever": length is +0.236 N·m/mm, against +0.18 for the ring radius and
  +0.094 for Outside diameter.
- "T grows as R_g²": only at fixed σ. Torque dips between 14 and 20 mm of ring radius.
- "The rings re-lock below pull-out": they re-lock only when the slip speed falls.
- "The end factor is accurate": it is optimistic by 11 % at a 4.4 mm gap, which is where the slice-1
  clearance fix sends users.
- "Edit the calibration inputs to match your design": that gives a meaningless factor of 1.57.
- "Each ring uses its own α": only rings with a picked grade do.
- "416, 17-4PH or 12L14 are clean swaps": all three trip low saturation. With plating set, 4140 and 1018
  are the only steel back irons that fire no warning.
- "Steel gives the torque shown": only if it does not saturate, and the default fails the wall check.
- "The stored 3D reverse and slip fields follow the design": the demagnetization limit stays 93.06 °C
  at every pole count.
- "A hotter-rated grade is free": torque falls 12-23 %. Until E21, its onset can be raised
  non-conservatively.
- "More mass lowers the steady temperature": it only slows the heating.
- "'Never' means the coupling can slip indefinitely": the margin is 0.54 °C on a placeholder
  conductance.
- "The clamp is sized to the drive": it is sized to cold-high times the safety factor.
- "Raising Boss outside diameter enlarges the drawn boss": it does not.
- "Model ferrite by typing a positive β": the GUI clamps it away.
- "The expansion caution counts the temperature swing": it does not.
- "1.5 T is 4140's saturation": that figure is unsourced.

Full lists: phys-notes-a 8, phys-notes-b 6, phys-notes-c 7, phys-new-notes 2.

**3. Live numbers beside words that turn false off-default.** For example "the fundamental carries
most of the torque" at low fill, or the margin sign. NaN or inf from an out-of-range end effect.
Mitigation: the five-design review and the "no prediction" rendering (2-A).

**4. "What to change" advice that holds at the defaults but not elsewhere.**
- Advice to raise an input that Auto holds (slice-1 known limitation, `02-system.yaml:107`), or one a
  part picker overrides (material inputs apply only at picker 1).
- Advice that pushes a different check red: a longer magnet raises cold-high, the clamp demand and
  gearbox swings. A smaller gap fails running clearance. A thicker wall grows the cup.

Every change entry therefore names its trade-off, and the reviewer probes its direction.

**5. Two names for one thing during or after the rename.** Labels on screen (overlay) differ from the
export and workbook labels. Search must still find old labels and cells. A note that quotes a label
typed as text would be stranded by a later rename; structured change entries (1-A) avoid that.

**6. Lost traceability.** Hidden E-badges also hide that a value differs from the original spreadsheet.
Mitigation: the toggle, the exports, and the wording "Differs from the original spreadsheet" when the
toggle is on.

**7. Model limits read as design facts.**
- The stored 3D fields: temperature limits and slip heat are borrowed from the default design.
- "Hot" in hot-low means the 50 °C operating input, not the 65 °C hot-day start (T-E). The required
  torque's workbook help says "At the 50 °C" and goes stale.
- Cold-high exceeds the gearbox input rating (0.805 against 0.6 N·m) and nothing checks it. No note may
  say the coupling protects the gearbox.
- "Nominal hot margin" (+7.5 %) sits beside a failing hot-low (-8.6 %), and "Below hot minimum" names
  two different comparisons (T-F).

**8. Layout and glyphs.** Longer notes and inline hints make the dashboard and the Equation panel
taller. The dashboard already scrolls at 1280x800, and the bottom-docked panel crushes the geometry
(VZ-12). Glyph failures: →, ≤, U+207B and ∝ render badly or fail the tests.

**9. Shipping risk.** A push deploys with no CI, so the local gate is the only test gate. Gate 12 SKIPs
silently without `MAGCOUPLING_PYTHON`. The smoke run never opens a note or a tooltip unless its prompt
is extended (R14).

**10. Review cost and pacing.** About 20 note reviews, about 5 help and label review batches, and a
whole-branch review. Under usage limits the slice may stall midway, with notes reviewed but help
unshipped. Decision 10-B lowers this risk.

**11. Unreachable rewrites.** Under 8-B, the improved notes stay behind Explain for most users, and the
slice's effort goes unseen.
