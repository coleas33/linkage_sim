# Slice 2 jargon map: user-visible internal vocabulary in the magcoupler GUI and engine text

Read-only audit of `C:/Users/Cole/source/repos/linkage_simulation/magcoupling-rs` at HEAD `5a1126a`
(slice 1 merged). Paths are relative to `magcoupling-rs/` unless noted. Line numbers are the
start line of the string literal (I lexed every `.rs` file under `src/`, kept only string
literals outside comments and outside the `#[cfg(test)]` tail, then read each hit in context).
The audit screenshots in `scratchpad/shots/` pre-date slice 1; I re-checked every claim
against the current source, and say where slice 1 already changed something.

Nothing in the repo was modified. The only files I wrote are this one and a throwaway helper
`s2/_lex.py` / `s2/_dump.py` / `s2/_hits.py` in the same scratch folder.

---

## 0. Bottom line (read this first)

1. **Provenance text is the bulk of it, and it comes from four mechanisms**, not from hundreds
   of separate strings:
   (a) the hover on every input and result adds a dotted path line and a workbook-cell line
   (`inputs.rs:910-917`, `dashboard.rs:406-430`);
   (b) the grey `E3 E7 E8` correction ids beside dashboard values and in an unlabelled column
   of the results table (`dashboard.rs:566-568`, `results_table.rs:733-740`), with a hover that
   cites repo file paths (`corrections.rs:175-195`, `deviations.rs:249-262`);
   (c) the assumption `source` line, which is all workbook cells and audit ids
   (`assumptions.rs:53-144`, shown at `panel.rs:1171-1176`), and the note `Sources:` line
   (`explorer.rs:588-598`);
   (d) about 83 help strings written for reviewers, not users (Appendix A).
   Fix those four and most of the list below goes away.
2. **The E3/E7/E8 badges are workbook-correction ids, not evidence levels.** E1 to E14 are rows
   of the M1 math audit; E15 to E20 come from the Addendum A verification. A badge means "an
   approved correction to the original Excel workbook changes this cell". They are drawn by
   `ui.small(&line.marker)` and have no hover of their own. Section 3.
3. **What is locked by tests.** The label, unit, help, cell and choices of every ported input
   and result equal the Python schema exactly (`tests/python_schema.rs`), so the 87 empty
   helps (all on ported inputs) and every ported label cannot be edited in the engine without
   re-baselining Python data. A GUI-side map is the only cheap route. Rust-only fields (11
   inputs, about 93 results) are free apart from re-blessing `tests/data/input_schema.json`.
   Verdict strings are pinned by the Python differential data and are matched exactly by
   `verdict_level` (`dashboard.rs:257-336`). Section 1.
4. **Slice 1 already ships a reusable guard**: `guidance::plain_text_violations` and
   `BANNED` (`guidance.rs:1439-1494`). It is applied to the check guides, the Size tab text and
   the welcome strip only. Slice 2 should apply it to every other surface (list in 5.3).
5. **Slice 1's own text still mixes names**: on the Size tab alone the one ring radius is
   "Outside diameter" (slider), "ring radius" and "ring size" (consequence lines) and "cup
   OD". Section 2.1.
6. **One audit suggestion is unsafe**: UI-50 proposes naming hot-low torque "Guaranteed-minimum
   torque". The slice-1 spec section 8 forbids "Pull-out torque is guaranteed". Do not use it.

---

## 1. Constraint map: what pins which kind of text

| Text kind | Where defined | Pinned by | Consumed by other code | Consequence for slice 2 |
|---|---|---|---|---|
| Ported input label, unit, help, cell, choices (161 Python inputs) | `engine/*.rs` via `param(...)` | `tests/python_schema.rs::ported_inputs_carry_the_python_metadata_and_defaults` (exact), `tests/data/input_schema.json` (blessed, `tests/schema.rs`) | label feeds `guidance::fix_label`, `trace.rs`, search, equation term rows | Edit only through a GUI-side label/hint map. The registry's `workbook_help` mechanism (`deviations.rs:228-232`, `tests/common/mod.rs:192`) exists but needs a deviation entry per field. |
| Ported result label, unit, help, cell (331 Python results) | `engine/*.rs` via `out(...)` | `tests/python_schema.rs::ported_results_carry_the_python_metadata` (exact) | `term_label()` (`explorer.rs:370`), results table, exports | Same: GUI-side map only. |
| Table-row labels (sweeps) | `engine/sweeps.rs:51-79` | only path and cell (`tables_match_the_python_layout`) | results table, plots read the fields by path | Labels are free to change. No literal pin found for `Inner apothem`, `R_g`, `Pull-out at T_op`. |
| Rust-only input meta (11) | `param_rust_only` | `input_schema.json` (re-bless with `MAGCOUPLING_BLESS=1 cargo test --test schema`) | tooltips | Free to reword; re-bless. Literal pins: `inputs.rs:1395-1403` (harmonic tooltip). |
| Rust-only result meta (about 93) | `out_rust_only` | non-empty label only (`tests/schema.rs:204-260`) | tooltips, results table, equation term rows | Free to reword. |
| Verdict strings (`Below hot minimum`, `Too thin: raise Metal design C122 ...`, `TOO NARROW: ...`) | `engine/model.rs`, `metal_design.rs`, `materials.rs`, `temperature.rs` | Python snapshot `tests/data/reference_values.json`, `tests/data/differential/full.json`, branch list `tests/differential.rs:89-291`, `tests/deviations.rs:511,540` | `dashboard::verdict_level` matches the exact text; `plots.rs:317-318` matches the sweep status; equation records repeat the text as formula (`records/dashboard.rs:44-77`, `records/torque.rs:212`, `records/temperature.rs:19-57`, `records/demagnetization.rs:167-177`) and `tests/explain.rs` evaluates them against the engine | Do not edit in the engine. Use a display-layer map `(path, text) -> plain text` applied only to the dashboard value (and the guide). The results table, the equation panel and the exports keep the workbook wording (as slice-1 spec 3.4 already says). |
| Workbook cell, as data (`meta.cell`, `ResultInfo.cell`) | meta tables | `tests/schema.rs`, `python_schema.rs`, reference snapshot | `CorrectionIndex` is keyed by cell (`corrections.rs:71,137`); `search_haystack` (`format.rs:35`); xlsx "Workbook cell" columns (`spreadsheet.rs:53,68,80`); CSV and JSON exports (`results_table.rs:44,330,349`); explorer line (`explorer.rs:641-645`) | Keep the data. Change only what is drawn. |
| Correction ids `E1..E20`, `title`, `evidence()` | `engine/deviations.rs` | `tests/deviations.rs`, `corrections.rs:221-248`, `dashboard.rs:1330`, `results_table.rs:775`, `panel.rs:6701` | `marker_text` -> `DashboardLine.marker`, `TableEntry.marker` -> xlsx columns (`spreadsheet.rs:356,537`); `Registry::corrections_upstream` -> explorer line | Keep the data and the export. Gate the drawing. |
| Assumption `rationale`, `source` | `engine/assumptions.rs` | `tests/assumptions.rs:108-129`: `source` must contain the workbook cell of every input it sets, or "Addendum A3" for Rust-only | `panel.rs:1167-1176`, `dashboard.rs:768` (rationale as hover), xlsx Assumptions sheet (`spreadsheet.rs:557-558`) | Split into two fields (plain rationale for the screen, `source` kept for the export and the test), or relax the test. |
| Note `sentences`, `watch_out`, `sources`, `title` | `engine/explain/notes.rs` | in-file tests: 2 to 6 sentences, non-empty `sources`, 15 to 20 notes (`notes.rs:624-655`); `tests/explain.rs` release gate: every start-here and warning note must be `Review::Reviewed` | `explorer.rs:571-599` draws sentences, watch-out and `Sources:` | `sources` must stay non-empty. Render it collapsed or only in the export. Every rewritten note needs a fresh review record or it drops to Draft and disappears. |
| Plain-words guard | `guidance.rs:1439-1494` | `guidance.rs:3287-3330` | used by 8 test sites (`dashboard.rs:2138`, `guidance.rs:3328,4515,4557`, `panel.rs:1962,4857`, `size_controls.rs:466,572`) | Extend `BANNED` and apply to every surface below. |

---

## 2. Inventory

Columns: **String (file:line)** | **Where on screen** | **Pinned by test?** | **Consumed by code?** |
**Proposed plain replacement**. "Const only" means tests reference the constant, so changing the
text breaks nothing.

### 2.1 Radius and size names

One physical quantity, the radius from the shaft axis to the back face of the inner magnets
(input `coupling.inner_back_apothem_mm`, default 10.15 mm, symbol `a_i`), is called, on screen:

`Inner magnet back apothem, including bondline` | `Ring radius` | `ring size` | `Outside diameter`
(the Size tab slider that edits it) and, in the equation panel and sweeps, `a_i`, `Inner apothem`,
`apothem`. A different, derived radius is `Gap mean radius` / `R_g` / `gap radius` (the middle of
the air gap: ring radius + inner magnet thickness + half the face gap). A different diameter is
`Cup outer diameter`. Rows:

| # | String (file:line) | Where on screen | Pinned? | Consumed? | Proposed replacement |
|---|---|---|---|---|---|
| R-1 | `Inner magnet back apothem, including bondline` (`engine/model.rs:167`, input label, Calculator!C8) | Detailed inputs > Magnets and rings (closed by default, not in Key design); equation-panel term list and breadcrumb; trace banner; xlsx Inputs | `python_schema.json` + `input_schema.json` | `InputCatalogue` label feeds `fix_label` (`guidance.rs:857`), trace, search | GUI label map: **`Ring radius`**. Hover: "Distance from the shaft centre to the back of the inner magnets (hub surface plus bond line). The magnet count does not change it." |
| R-2 | help `Machined hub apothem plus the inner bondline.` (`model.rs:168`) | same row's hover | same | none | hint map, text as R-1 hover |
| R-3 | `Outside diameter` (`gui/size_controls.rs:26` `DIAMETER`) | Size tab slider; status "Outside diameter 42.8 -> 40.1 mm" (`size_controls.rs:77`, `sizing.rs:357`); fix text "Raise Outside diameter" (`guidance.rs:154` `SIZE_TAB_ALIASES`); guidance "Smallest outside diameter that fits N magnets" (`guidance.rs:1962`) | const only | `variable_slider` maps ring size to it | Keep as the name of the Size-tab control, but define it once (hover, none today): "Largest outside diameter of the rotating parts (cup, cap and boss). Moving it changes the ring radius, the cap OD and the cap thread together." It is `max(cup, cap, boss)`, not the ring (`engine/size_controls.rs:63`) |
| R-4 | `Ring radius` (`gui/sizing.rs:84`, `variable_label`); `The ring radius stays the same when you change the magnet count; the cup OD below can still change.` (`guidance.rs:1726`) | xlsx Summary "Free variable" cell; Size tab line under Magnets per ring; `change_text` fallback `size_controls.rs:86,528` | literal at `size_controls.rs:528` and `guidance.rs:4537` | spreadsheet | Keep `Ring radius` as the canonical name. Add the bridge sentence on the Size tab: "Outside diameter follows the ring radius." |
| R-5 | `ring size` (`gui/sizing.rs:94` `auto_choice`); `No valid value of {} in range` -> "No valid value of ring size in range" (`sizing.rs:332`); `Auto resizes the ring ... present ring size.` (`guidance.rs:1731`); `at a given ring size` (`guidance.rs:282`); `do not fit at this ring size` (`guidance.rs:1697`) | Auto row dropdown "Auto: smallest [ring size] that meets the torque requirement"; status line; fit guide Why and Problem | literals: `panel.rs:5446`, `sizing.rs:797`, `guidance.rs:3850-3859,4537` | none | Use **`ring radius`** in Auto's dropdown and the fit text; keep "outside diameter" only where the slider is named. Update the 6 literal tests. |
| R-6 | `Cup outer diameter` (`model.rs:254`, python label, C62); `cup OD` (`guidance.rs:1726` and consequence lines); `Steel cup body OD` (`metal_design.rs:309`); `Threaded cap OD` (`metal_design.rs`, input C166) | dashboard row 7; Size tab lines; detailed inputs | python parity for labels | `cup_od_mm` read by guidance | Keep labels; add one hover on the dashboard row: "Outside diameter of the steel ring around the outer magnets. It can be smaller than the Outside diameter, which also counts the cap." |
| R-7 | `Diameter: rotating OD 42.80 mm of 43.00 mm`, `Overall length: axial stack ...`, `Large-diameter bay: stack ...` (`geometry.rs:727-750`, formatted `geometry.rs:585-597`) | callout list under the geometry drawing (Size tab and Geometry tab) | built from callouts; no literal pin found for `rotating OD` | callout `path` fields | `Outside diameter 42.80 mm of the 43.00 mm space claim` (same word as the slider) |
| R-8 | `Proposed rotating outside diameter` (`metal_design.rs:289`, C135) | results table, hover of callout 6 | python parity | none | leave; mention in hover of R-7 that this is the same number |
| R-9 | `Gap mean radius` (`model.rs:256`), `Pole pitch at the gap radius` (`model.rs:257`), `Gap area × lever arm (2π R_g² L)` (`model.rs:326`), `Mean gap radius` (`calibration.rs:104`), sweep column `R_g` (`sweeps.rs:59`) | results table (hover only on dashboard), equation panel term list, symbol `R_g` in the equation (`records/torque.rs:90`) | python parity for model labels; sweeps free | `R_g` symbol pinned `tests/explain.rs:979` | Define once, in the new torque note and the results hover: "Gap radius R_g: the radius of the middle of the air gap, ring radius + inner magnet thickness + half the gap." Rename `R_g` column to `Gap radius`. |
| R-10 | `apothem` as a user word: verdicts `TOO NARROW: increase apothem or reduce poles` and `TOO NARROW: increase gap/apothem or reduce poles` (`model.rs:937,952`); `Inner apothem`, `Outer face apothem` (`sweeps.rs:54,57`); `Prototype inner hub apothem` (`calibration.rs:55`); `Outer magnet face apothem` (`calibration.rs:102`); `Outer block face apothem`, `Outer block back apothem` (`model.rs:244,251`); legend `Pull-out torque (rows at their smallest fitting apothem)` (`plots.rs:151`) | results table rows `Inner flat check` / `Outer flat check` (verdict text), Gap sweep and Pole sweep groups, Pole sweep plot legend (none of these are dashboard rows) | verdicts: differential tests + `full.json` + `verdict_level` (`dashboard.rs:267-268`); plot const only | `PULL_OUT_SMALLEST_APOTHEM` const | Engine verdicts stay. Plot legend: `Pull-out torque (each row at the smallest ring radius where its magnets fit)`. Sweep columns: `Ring radius`, `Outer magnet face radius`. Verdict display map: `Too narrow for the magnets: raise the ring radius or use fewer magnets`. |
| R-11 | `Inner block face radius`, `Inner block corner radius`, `Pocket corner radius`, `Inner block mid radius` (`model.rs:241-253`, `temperature.rs:523`); guidance `at their mid-radius` (`guidance.rs:1711`) | results table; guidance fit problem | python parity (model) | none | leave; they are distinct quantities. Only the guidance phrase needs "(midway through the magnet thickness)" once. |

Companion cluster, the magnet count (same pain, same fix):

| # | String (file:line) | Where | Pinned? | Replacement |
|---|---|---|---|---|
| N-1 | `Number of poles per ring` (`model.rs:158`, help `Even; one block per pole.`) | Detailed inputs, Key design row | python parity | GUI label map `Magnets per ring (one block per pole)`; hint "Sets how many blocks go around the ring; it does not change the ring radius." |
| N-2 | `Magnets per ring` (`size_controls.rs:29`, `sizing.rs:83`), `magnet count` (`sizing.rs:93`), `blocks per ring` (`guidance.rs:1697`), `Poles per ring` (plot axis `plots.rs:466`, `calibration.rs:98`), `reduce poles` (verdicts) | Size tab, Auto dropdown, guidance, pole-sweep axis | const only / literals in guidance tests | Canonical: `magnets per ring`; "blocks" only for the individual pieces; axis `Magnets per ring`. |

### 2.2 Torque names

Canonical set I propose (matches the slice-1 vocabulary; avoids spec section 8 traps):

- **Required torque**: `metal.required_min_Nm`, the least torque the coupling must carry, hot.
- **Pull-out torque (nominal)**: `model.pullout_Nm`, the most torque the coupling carries before
  it slips, at the operating magnet temperature, from the model, before production scatter. An
  estimate, not guaranteed.
- **Hot-low torque**: `metal.torque_hot_low_Nm` = pull-out at the operating temperature less the
  production-variation allowance. The number the requirement is checked against. Still an
  estimate; never "guaranteed".
- **Cold-high torque**: `metal.torque_cold_high_Nm` = cold pull-out plus the allowance; the most
  the coupling could pass; it sizes the clamp.
- **Required floor** (`model.required_floor_Nm`) is a different number (the larger of the driving
  need times its safety factor and the required torque); keep it separate and name it.

| # | String (file:line) | Where on screen | Pinned? | Consumed? | Replacement |
|---|---|---|---|---|---|
| T-1 | `Required` (`size_controls.rs:133`) | Size tab slider | const | sets `metal.required_min_Nm` | `Required torque`, hover "The least torque the coupling must carry when hot. The checks compare Hot-low torque with it." |
| T-2 | `Required minimum service pull-out torque` (`metal_design.rs:25`, C7; help `At the 50 °C magnet temperature.`, a literal 50 that goes stale when the operating temperature changes) | Detailed inputs, Requirements group | python parity | fix text (`guidance.rs`) | GUI label map `Required torque (hot)`; hover generated from `coupling.op_temp_C`. |
| T-3 | `Target hot-low torque` (`panel.rs:113` `TARGET_LABEL`) | xlsx Summary only since slice 1 (`spreadsheet.rs:31`) | const (`panel.rs`/`spreadsheet.rs` tests) | export | `Required torque` (it is the same number since G-2). |
| T-4 | `Required minimum` (`plots.rs:154`), `Required floor` (`plots.rs:162`), `that meets the torque requirement` (`size_controls.rs:34`) | torque-vs-temperature legend; sweep plot legends; Auto row | const | none | `Required torque` and `Required torque or driving need (larger)`; Auto: "that meets the required torque". |
| T-5 | `Required nominal 20 °C torque for hot minimum` (`metal_design.rs:274`, C19), `Required floor: larger of traction need and service minimum` (`model.rs:352`) | results table | python parity / Rust-only | none | add hovers only |
| T-6 | `Pull-out torque at operating temperature` (`model.rs:333`, help `Analytical estimate, not a guaranteed minimum.`) | dashboard row 1; hover | python parity | `plots.rs:107` | keep; add the one-line definition above to the hover (never defined on screen today). |
| T-7 | `Pull-out torque (nominal, at 50 °C)` (`size_controls.rs:154`) | Size tab third readout | `panel.rs:4331-4346` via `pullout_label` | none | keep. Replaced "Expected torque" in slice 1; `panel.rs:4342` asserts the old word is gone. |
| T-8 | `Pull-out torque` (`plots.rs:148`), `Pull-out at the operating temperature [N·m]` (`plots.rs:589`), `Pull-out at T_op` / `Pull-out at 20 C` (`sweeps.rs:75-76`), unit `Nm` in 76 sweep rows vs `N·m` elsewhere | plot legends and axis; sweep table rows | const; sweeps not pinned | none | `Pull-out torque (nominal)`; unit `N·m` everywhere. |
| T-9 | `Predicted candidate torque at operating temperature` (`metal_design.rs:265`), `Original model pull-out torque`, `2D pull-out torque (infinite length)` (`model.rs`) | results table | python parity | none | hover: "same as Pull-out torque" for the first. |
| T-10 | `Hot-side low torque including assumed variation` (`metal_design.rs:268`, C9) | dashboard row 3 | python parity | `STORED/END_EFFECT_ROWS` | GUI label alias on the dashboard: `Hot-low torque`. |
| T-11 | `Hot-low` (`size_controls.rs:137`), `Hot-low torque` (`guidance.rs:158` guide label), `hot-low torque` (`sizing.rs:322`, `size_controls.rs:36` hover), `Hot-low torque {} is below the required {}...` (`guidance.rs:1556`) | Size tab, guide block title, Auto/solved status | const / literals in `panel.rs:4331`, `guidance.rs` | status prefix matched `Solved at ` | One name `Hot-low torque` everywhere, with the hover from the canonical set above on the Size tab label. |
| T-12 | `Hot minimum with variation allowance` (`metal_design.rs:270`, C11, check label), `Estimate covers hot min` / `Below hot minimum` (`metal_design.rs:457-459`) | dashboard row 4 value; results table | verdict: Python data + `verdict_level` | equation record `records/torque.rs:212` | display map for the dashboard value only: `Hot-low torque meets the required torque` / `Hot-low torque is below the required torque`. |
| T-13 | `Below hot minimum` (`model.rs:1159`, `model.verdict`), `Nominal only: hot test` (`model.rs:1161`), sweep status `below hot minimum` / `nominal: test needed` (`sweeps.rs:173,175`), plot `BELOW_MINIMUM`, `NOMINAL` (`plots.rs:158-159,317-318`) | results table `Verdict`; gap and pole sweep tables and plots | Python data, `tests/deviations.rs:511,540`, `differential.rs:117` | `plots.rs` matches the status exactly | **Same words, different test**: here it compares the *nominal* pull-out with the *required floor* (`model.rs:1158`, `sweeps.rs:172`); T-12 compares *hot-low* with the *required torque*. Plot legends: `Below the required floor` / `Meets it nominally: confirm with a hot test`. |
| T-14 | `Cold torque with assumed positive variation` (`metal_design.rs:269`), `Predicted candidate cold high, with allowance` (C157), `Candidate cold high, input equivalent` (C158) | dashboard row 5; results table | python parity | clamp sizing | dashboard alias `Cold-high torque`; guidance already says "the highest torque the coupling can pass (cold, plus the 15 % variation allowance)" (`guidance.rs:1597`): reuse. |
| T-15 | `Symmetric torque variation allowance` (`metal_design.rs:31`, unit `fraction`, shows "0.150 fraction"), `Production variation` (`assumptions.rs:71`, label), `Low: minus the variation allowance` / `High: plus the variation allowance` (`plots.rs:152-153`) | input row, Assumptions view, plot legend | python parity (input) | none | `Production variation allowance`, shown as `15 %`. |
| T-16 | `Meets it nominally (no variation allowance)` / `Below it` (`temperature.rs:1332`, label `Against the requirement`) | temperature verdict block, results table | Python data + `verdict_level` (`dashboard.rs:286-287`) | `records/temperature.rs:31` | display map: `Pull-out at the hot-day temperature meets the required torque (before variation)`. |

### 2.3 Workbook and cell references

| # | String (file:line) | Where on screen | Pinned? | Consumed? | Replacement |
|---|---|---|---|---|---|
| W-1 | cell line `Calculator!C93` etc. in every input hover (`inputs.rs:914-917`) and result hover (`dashboard.rs:413-417`), plus the dotted path line (`inputs.rs:912`, `dashboard.rs:412`) | hover of every input label, slider, dashboard row, table row, geometry callout, plot readout | literal strings: `inputs.rs:1379-1383` (exact tooltip for `metal.face_gap_mm`), `dashboard.rs:1474` (`"\nmodel.pullout_Nm\nCalculator!C93"`) | `meta.cell` data stays | Show only when a header toggle "Show spreadsheet details" is on, labelled `Original spreadsheet cell: Calculator!C93` and `Internal name: model.pullout_Nm`; omit when off. |
| W-2 | `Rust-only input (no workbook cell)` (`inputs.rs:915`), `Rust-only result (no workbook cell)` (`dashboard.rs:416`) | hover of the 11 Rust-only inputs and about 93 Rust-only results (incl. dashboard rows Space claim, End-effect check) | `inputs.rs:1400`, `dashboard.rs:1484,1504`; `guidance.rs:3300` uses it as the guard's sample | none | Omit the line (toggle off); toggle on: `Not in the original spreadsheet`. |
| W-3 | `Rust-only` as the cell text (`results_table.rs:56` `RUST_ONLY`) | results-table cell column (every Rust-only row), CSV `cell` empty, xlsx "Workbook cell" (`spreadsheet.rs:286`) | const; `spreadsheet.rs:680,849` | export | on screen: blank or `new`; export: `not in original`. |
| W-4 | results-table row = label, value, **cell**, **marker** columns, no header (`results_table.rs:733-740`); hint `Search label, path or cell` (`results_table.rs:53`); inputs filter hint `Filter inputs by label, path or cell` (`inputs.rs:802`) | Results table tab; left inputs filter | consts (`SEARCH_HINT` `panel.rs:2622,2702…`, `FILTER_HINT` `panel.rs:42,1206,2482,3990`); **`panel.rs:2619` asserts the results table draws `Calculator!C93` exactly once** (the cell column), so a toggle needs that test to switch it on | `search_haystack` (`format.rs:35`) matches cells and paths | Columns gated by the toggle; hints `Search by name`. Keep the cell/path matching (power users type `C93`). |
| W-5 | explorer: `{label} ({cell})` e.g. `Pull-out torque at operating temperature (Calculator!C93)` (`explorer.rs:641-645`), `Rust-only result` (`explorer.rs:644,795`) | first line under the equation panel heading, and for cell-only results | `panel.rs:6676-6702` (exact line `... (Calculator!C93)`) | `eq.cell` from the registry | toggle: otherwise just the label. |
| W-6 | `No equation record: the value, its label and its workbook cell are what the explorer knows.` (`explorer.rs:75`) | equation panel for any result with no equation record (opened from the results table or a term link) | const `panel.rs:6659-6675` | none | `This value has no equation page yet; its name and value are shown above.` |
| W-7 | `3D values from the workbook` (`dashboard.rs:207` `STORED_3D_LABEL`) | grey line after the last temperature row; xlsx Summary note (`spreadsheet.rs:345`) | const (`dashboard.rs:1353`, `panel.rs:2570` count == 1) | `spreadsheet.rs` | `Temperature limits use 3D field values for the default design`. |
| W-8 | `The demagnetization reverse fields and the slip-loss fields are the workbook's stored 3D values: they do not follow geometry changes until M3 computes them live.` (`dashboard.rs:210`) | hover of W-7 and tail of the three temperature-row hovers (`dashboard.rs:423-425`) | const only | none | `The temperature rows use 3D magnetic field values worked out for the default design. They do not change when you change the geometry or the magnet grade, so treat the temperature limits as approximate for other designs.` Do not promise a live model (M3 is on hold). Slice 1 already has plain wording for the same fact: `guidance.rs:1238,1271`. |
| W-9 | `Workbook example design` (`guidance.rs:1381` `WORKBOOK_EXAMPLE`; slice-1 spec G-6 allows it) | first line of the dashboard: "Workbook example design: fails 3 of 8 checks" | `guidance.rs:3307,3321,3996-4000`, `ALLOWED_WORKBOOK` (`guidance.rs:1452`), `dashboard` via const | status line | `Example design: fails 3 of 8 checks` (hover: "The starting values from the original spreadsheet"). Needs owner sign-off because G-6 named the workbook explicitly. |
| W-10 | `Reset to workbook defaults` (`panel.rs:129`), `Every assumption back to its workbook default; the design inputs stay as they are` (`panel.rs:1021`), `Assumptions reset to the workbook defaults` (`panel.rs:1026`), `Changed from the workbook default` (`panel.rs:1157`) | Assumptions-modified banner button and hover; status line; changed dot hover in Assumptions view | const / literal only at the definition | none | `Reset assumptions to the starting values`, `Changed from the starting value`. |
| W-11 | `Workbook groups` (`inputs.rs:80`, `InputOrder::Workbook`) | order toggle "By workflow / Workbook groups" above the inputs | labels used via `InputOrder::label()` in all panel tests | none | `By original sheet` (or "By original spreadsheet sheet"). |
| W-12 | ASSUMPTIONS_NOTE `The model's assumptions, apart from the design inputs (each also stays in its input group). The equation panel tags them and what they flow into.` (`panel.rs:135`) | top of the Assumptions view | const (`panel.rs:1150`) | none | `These are modelling choices, not requirements. Change one to see how sensitive the answer is; each shows why it has its value.` (not workbook jargon, but circular; UI-31) |
| W-13 | assumption sources, all 14 (`assumptions.rs:53,60,67,74,81,88,95,102,109,116,123,130,137,144`): e.g. `Workbook Calculator!C41 and Calibration!C23 (0.15); M1 audit M8 (error band against 3D) and M9 (negative for short magnets).` | small grey `Source: ...` line under every assumption (`panel.rs:1171-1176`); xlsx Assumptions sheet | **`tests/assumptions.rs:123-127`: source must contain every input's cell, or "Addendum A3"** | export (`spreadsheet.rs:558`) | Keep the field for the export and the test. Show on screen a plain `Why this value` instead (the rationale, rewritten) and put `source` behind the details toggle. |
| W-14 | assumption rationales with jargon (8 of 14): `the workbook sums 1, 3 and 5` (52), `(end_effect_check)` (59), `Calibration C9`, `Calculator C42`, `The audit found` (66), `(decision A2-7)` (80), `With correction E20` (87), `the audit found that calibration uses 9.83 °C` (94), `with E20` (101), `Shaft clamps C66` (136) (`assumptions.rs`) | rationale text under every assumption; hover of the Why link on dashboard rows (`dashboard.rs:768`) | `tests/dashboard` uses the const (`dashboard.rs:1945`); the guard test exempts rationales ("may cite cells", `dashboard.rs:2118`) | export | rewrite 6 rationales; then remove the exemption at `dashboard.rs:2118-2128`. |
| W-15 | `Metal design C169` / `C165`, `Metal design C127` / `Shaft clamps C35` in the two Design check notes (`geometry.rs:798,808`) | amber lines under the geometry drawing (Size and Geometry tabs) | literal tests `geometry.rs:1166-1167`, prefix `geometry_view.rs:523` | none | `Design check: the cap thread diameter 41.00 mm is below the cup body OD 41.33 mm`; `Design check: the boss OD is 22.00 mm in the housing inputs and 25.00 mm in the clamp inputs`. |
| W-16 | verdict `Too thin: raise Metal design C122 to at least 2.0 mm` (`materials.rs:244`; copy in `records/dashboard.rs:67`) | dashboard row 11 (Cup wall check) value; results table | Python data (`reference_values.json:1356`, `differential/full.json`); `verdict_level` (`dashboard.rs:324` prefix `Too thin: `); `tests/explain.rs` evaluates the record | `materials.rs:310-353` tests | display map on the dashboard only: `Too thin: raise Minimum outer return ring wall to at least 2.0 mm` (the on-screen input name, as the guide already says). Parse the number from the engine text. |
| W-17 | cell refs in ported-input helps: `Same as 1018; the mass model uses Metal design C132.` (`materials.rs:44`) | hover of Back-iron steel density | python parity | none | hint map |
| W-18 | cell refs in Rust-only helps: housing (`housing.rs:53-65`, 6), materials (`materials.rs:133-214`, 14), model (`model.rs:147,182,342,344,370,374`), temperature (`temperature.rs:105,257-263,475-506`), warnings (`warnings.rs:99-109`, 4) | hover of those inputs and results. Rows a user meets without opening the results table: Key design input `Axial magnet length, both rings` (`inputs.rs:30`, the long `Addendum A1` / `C42 C123 C124 C172` hover, `model.rs:147`) and dashboard row 17 `End-effect model check` (`Audit M9`) | `input_schema.json` for the inputs only | none | rewrite in plain words (Appendix A lists all 83). |
| W-19 | `(the workbook plans electroless nickel)` (`warnings.rs:85`, rule text) | dashboard "Material warnings" list when the uncoated-steel caution fires; xlsx Summary | `warnings.rs` tests compare the const only | export | `(plating, for example electroless nickel, is the usual fix)`. |
| W-20 | `1, 3, 5 (workbook)` (`model.rs:54`, choice of `coupling.max_harmonic`) | Assumptions view dropdown and the equation panel term value `5: 1, 3, 5 (workbook)` | `explorer.rs:1170`, `panel.rs:6906`, `input_schema.json:170` | choice code 5 | `1, 3, 5 (default)`. Update the 3 pins. |
| W-21 | `This design and its results as an .xlsx workbook laid out as this page (values, no formulas)` (`panel.rs:990`), `The page's layout as an .xlsx workbook (values, no formulas)` (`results_table.rs:504`) | hover of Export spreadsheet / Export XLSX | none | none | "workbook" here means the exported file, a different sense from W-1..W-20; say `spreadsheet file`. Also two names for one action: `Export spreadsheet` (`panel.rs:90`) vs `Export XLSX` (`results_table.rs:50`). |
| W-22 | xlsx export headers `Workbook cell`, `Corrected vs workbook`, `Workbook default`, `Source`, `Path`, `Corrections applied (every approved correction is on)` (`spreadsheet.rs:53,68,70,80,82,92,95,106`) | the downloaded spreadsheet | `spreadsheet.rs` tests via `column(&RESULT_COLUMNS, "Corrected vs workbook")` (`spreadsheet.rs:948`) | export | This is the right home for traceability. Rename headers `Original spreadsheet cell`, `Differs from original spreadsheet`, `Starting value`; keep the E-id legend (already there, `spreadsheet.rs:378-383`). Change both the header consts and the 3 test lookups. |
| W-23 | engine sweep help `... (workbook 'Corner gap (mm)')` (`sweeps.rs:53`) | results-table hover of every `Swept variable` | none | none | `Corner gap in the gap sweep; poles per ring in the pole sweep` |
| W-24 | helps with `the workbook's ...` on reworded ports (`temperature.rs:200,246,559-565`, `metal_design.rs:299,312`, `clamps.rs:237`) | hovers | the Python text is already recorded in `workbook_help`; Rust text is free | none | plain rewrite; keep `workbook_help` untouched |

### 2.4 Audit and spec ids, and the correction badges

| # | String (file:line) | Where on screen | Pinned? | Consumed? | Replacement |
|---|---|---|---|---|---|
| A-1 | `E3 E7 E8` marker text (`corrections.rs:163-169`; drawn `dashboard.rs:566-568`, `results_table.rs:740`) | beside 14 of the 17 dashboard values at the defaults; a column in the results table (over 100 rows) | `corrections.rs:236`, `dashboard.rs:1330`, `results_table.rs:775` | xlsx marker columns | Section 3. Gate behind the toggle; legend line when on. |
| A-2 | `Corrected vs workbook:` + `E3: <title>` + `at the default design, with E3 alone: workbook 2.647, corrected 2.688` + `docs/analyses/2026-09-29-magcoupling-math-audit.md, entry E3` (`corrections.rs:175-195`, `deviations.rs:249-262`) | tail of every hover of a marked value | `corrections.rs:239,244,247`, `dashboard.rs:1500`, `results_table.rs:825` | tooltip only | `Differs from the original spreadsheet:` then the plain title and `was 2.647, now 2.688`; drop the repo path. The 20 `title` strings are already near plain English (list in Section 3). |
| A-3 | `Embodies corrections: E7, E8, E3` (`explorer.rs:78,676`) | line in the equation panel under the value | `panel.rs:6701` (exact) | `Registry::corrections_upstream` | `Changed from the original spreadsheet: ...titles` or hide behind the toggle. |
| A-4 | note `sources` (`explorer.rs:104,588-598`; data `notes.rs:84-536`), e.g. `Sources: M1 audit M3, M4, M7 (the planar harmonic model against exact 2D ...); Addendum A decision 29 (odd harmonics up to 11)`, `Plan A-3 traceability: the f_cal,0 cancellation (tests/explain.rs CANCELLATIONS)` (`notes.rs:188`), `Plan A-3 Task 16 physics review, the saturation cap ...` (`notes.rs:439`), `Workbook Materials C26 ...` (`notes.rs:509`) | small grey last line of every note in the equation panel | `notes.rs` test requires non-empty `sources` | `explorer.rs:592`; `Review::Reviewed.record` is not shown | Keep data; draw under a collapsed `Where this comes from`, or only in the review/export. |
| A-5 | ids inside note sentences and watch-outs (full list, Appendix C): `(correction E7)` `notes.rs:131`; `one M1 skeptic found it 11 % high` `152`; `M1 audit M9` `155`; `M1 audit M4 ... M5 ... the workbook scales` `175`; `(correction E3) ... (M1 audit P2)` `181`; `With correction E20` `249`; `M1 audit M13` `252`; `correction E17` `321`; `correction E12` `354`; `M1 audit placeholder table` `357`; `the highest the workbook names` `478`; `M1 audit M1` `482`; `The workbook therefore plans` `502`; `the Volkersen screen` `524`; `the workbook keeps only 1, 3 and 5` `77` | body text of 12 of the 17 notes | none beyond the review gate | none | The note rewrite removes them; add a test: no sentence matches `M1 audit|skeptic|correction E\d|workbook|Addendum|Plan A-`. |
| A-6 | `Audit M9 (the user's decision: a flag, no physics change): ...` (`model.rs:331`, `calibration.rs:110`) | hover of dashboard row 17 `End-effect model check` | none | none | `The model gives no usable prediction when the magnets are no longer than about 0.15 of a pole pitch (end-effect factor <= 0). Everything computed from the pull-out is then greyed.` |
| A-7 | help texts for Rust-only results: `Plan A-3 (a term of the equation explorer): ...` (calibration.rs:121-131, model.rs:294-322, materials.rs:198), `E7 (plan A-3 ...)` (`model.rs:340`, `calibration.rs:133`), `Addendum A1/A3/A5/A6`, `decision A2-7/A2-8/12`, `Correction E17/E20 ...`, `until M3 computes it live` (`temperature.rs:257,260,263`) | hovers in the results table and equation term lists (about 62 results, 9 inputs) | none for results; `input_schema.json` for inputs | none | plain rewrite; or hide hovers for rows that exist only to feed the explorer (the 6 `ampN` and 3 `kN` rows) |
| A-8 | `Model assumption (Addendum A3)` (`inputs.rs:937`) | last line of the hover of the 15 assumption inputs | `inputs.rs:1401` | `meta.assumption` | `A modelling choice; see the Assumptions tab`. |
| A-9 | `M3` (`dashboard.rs:210`); `M41 × 0.5 concept.` (`metal.cap_thread_dia_mm` help, C169): **a thread size, not an id** | M3 in W-8; the thread is genuine | python parity for the thread | none | M3 only. |
| A-10 | `Autofit` (`geometry.rs:250,787`: `Autofit: the wall rule asks for at least 2.000 mm at the pocket corners (input 1.800 mm)`); label `Suggested cup wall at the pocket corners (autofit)` (`materials.rs:187`, help `Addendum A1 autofit, decision 27 ...`) | grey line under the geometry drawing; results-table row | `geometry.rs:1192`, `geometry_view.rs:525` (literal `Autofit:`) | `cup_wall_suggested_mm` | Names a feature that does not exist. `Wall rule: the cup wall should be at least 2.000 mm at the pocket corners (you have 1.800 mm)`; label `Wall the rule asks for`. |
| A-11 | `Spec`/`decision`/`M2 decision D7` etc. in `Source:` lines (W-13) and `Review` records (not drawn) | see W-13 | | | |

### 2.5 Code and file names

| # | String (file:line) | Where on screen | Pinned? | Consumed? | Replacement |
|---|---|---|---|---|---|
| C-1 | `The drawing shows the one-piece slotted clamp (drawing.py's only layout).` (`clamp_drawing.rs:39`, drawn `clamp_drawing.rs:812`) | note under the Clamp tab drawing when the two-piece clamp is selected | const `clamp_drawing.rs:1207` | none | `Only the one-piece slotted clamp is drawn; a two-piece clamp has no drawing.` |
| C-2 | `Rust-only ...` strings | see W-2, W-3, W-5 | | | |
| C-3 | dotted paths: hover line (`inputs.rs:912`, `dashboard.rs:412`); equation term label hover `.on_hover_text(&row.path)` (`explorer.rs:723`); xlsx `Path` columns (`spreadsheet.rs:54,69,81,89`); CSV header `path,label,value,unit,cell` (`results_table.rs:44`); load/refusal messages `coupling.npole: no such input` (`meta.rs:492-503`), `Sizing refused: invalid inputs (coupling.npole, ...)` (`sizing.rs:340`) | hovers; exports; status line after a bad file or an invalid design | `inputs.rs:1379`, `dashboard.rs:1474`, `results_table.rs:861` | exports and search | toggle for hovers; for the two messages map each path through `term_label` (UI-9). |
| C-4 | `equation record` (`explorer.rs:75`, tag `no equation record` `explorer.rs:768`; `trace.rs:75` `(its results have no equation record)`; `results_table.rs:68` `only results with an equation record are traced`) | equation-panel term tags, trace banner, empty results-table text | const `panel.rs:6659-6675`, `results_table` tests | none | `no equation shown for this value`; trace: "no result shown here uses it" |
| C-5 | `The differential tests cover the slider range only.` (`input_ui.rs:122`) | hover of the `outside the slider range` note on an input | none | none | `Your value is outside the slider range. It is kept until you edit it.` |
| C-6 | `Click the label to trace the explained results it drives` (`input_ui.rs:31`), `Tracing {label}: drives N explained results` (`trace.rs:77`) | hover of every input label; banner | const | none | "Click to highlight every result this input affects". "trace"/"explained" are internal. |
| C-7 | code names inside help texts: `coupling.max_harmonic`, `tau7_Pa, tau9_Pa and tau11_Pa` (`model.rs:182`), `pullout_angle_rad` (`model.rs:342`), `housing.* shows them` (`model.rs:147`), `end_effect_check` (`assumptions.rs:60`), `model.pullout_Nm`-style names none | hovers, Assumptions rationale | `input_schema.json` | none | plain rewrite |
| C-8 | `f_end` (`dashboard.rs:420,498`: `End-effect model out of range (f_end = -0.12): the pull-out and the numbers computed from it are greyed.`) | red banner over the dashboard and results table; hover of every greyed row | const `END_EFFECT_BANNER`; `plots.rs:161` | `end_effect_out_of_range` | `...(end-effect factor -0.12)`; the factor is shown elsewhere as `End-effect factor`. |
| C-9 | `Free variable`, `Target hot-low torque` (`panel.rs:113,116`) | **xlsx Summary only** after slice 1 (no longer drawn on screen) | const | `spreadsheet.rs:31` | `Solve for`, `Required torque`. |
| C-10 | console log prefixes `magcoupling sizing: `, `magcoupling explorer: ` (`panel.rs:119`, `readouts.rs:35`) | browser console only, not on screen; the web smoke test greps them | `gui-smoke` | yes | leave. Listed so nobody renames them. |
| C-11 | `3D results: stored reverse fields (refresh after a 3D run)`, `3D results: stored slip-loss fields (refresh after a 3D run)`, `3D results: reference torques` (`inputs.rs:534,540,551`) | group headings in Detailed inputs > Calibration and model | none | `SECTION_LABELS` fixture tests use paths | `Reverse fields from 3D analysis (default design)`, `Slip-loss fields from 3D analysis (default design)`, `3D reference torques`. Drop "refresh after a 3D run" (a developer workflow). |

### 2.6 Other

| # | String (file:line) | Where on screen | Note / replacement |
|---|---|---|---|
| O-1 | `Candidate flat-face magnetic gap` (`metal_design.rs:87`, C119, help `Same as the measured prototype.`), `Magnet face gap` (`size_controls.rs:28`, hover is this same help `size_controls.rs:322`), `Face gap` (`geometry.rs:473`), `Gap at the flat centres (effective gap)` (`model.rs`), `Corner gap` (`geometry.rs:481`, plot axis `plots.rs:457`), `Effective flat-centre gap` (`calibration.rs`) | Size tab, Key design, geometry list, plot axis | Same gap, four names. Canonical `Magnet face gap`; define "face gap = clearance between the inner and outer magnet faces at their centres; corner gap = the same at the block corners". Gap-sweep axis currently says `Corner gap` while the control is `Magnet face gap`. |
| O-2 | `candidate` and `prototype` (labels `Predicted candidate torque ...`, `Candidate cold high`, `Prototype inner hub apothem`, `Prototype magnets` section `inputs.rs:522`) | results table, Calibration group | workbook words for "this design" and "the measured test coupling". Gloss once in the Calibration group heading hover. |
| O-3 | `Verdict` is the label of two results (`model.verdict` C102, `temperature.summary.verdict` C25); the dashboard's `Verdict` row is the temperature one (`dashboard.rs:173`). 18 labels are duplicated (UI-42) | dashboard row 14; results table | GUI alias on the dashboard: `Temperature verdict`. |
| O-4 | symbol collisions in equations and notes: `σ` = shear stress (`records/torque.rs:45,64`) vs `σ` = conductivity (`notes.rs:318,455`); `τ_p` pole pitch (`torque.rs:91`) vs `τ_n` in help (`model.rs:340`) vs `τ: 63 %` time constant (`diagrams.rs:443`); `θ` in help (`model.rs:340`, `calibration.rs:133`) vs `φ` electrical angle (`symbols`, `diagrams.rs:289`) vs `ϑ` temperature | equation panel, notes, diagrams, hovers | PA-15. Fix while rewriting notes: conductivity `σ_el`, time constant `τ_th`, angle `φ` everywhere. |
| O-5 | undefined magnetics terms on first sight: `Br`, `Hcj`, `knee`, `remanence`, `coercivity`, `permeance`, `skipping`, `like poles facing`, `fill`, `pole pitch`, `back iron`, `end effect`, `bondline`, `f_end`, `A_n` (labels like `Intrinsic coercivity Hcj at 20 °C (grade minimum)`, help `once per pole pass while skipping`) | labels (python-locked), notes, guidance | These are domain words, not project jargon; define each once in the notes (the three new notes plus glossary hovers). Not banned by the guard. |
| O-6 | raw number sentinels `+inf`, `-inf`, `NaN` (`format.rs:52`), unit text `fraction` (shows `0.150 fraction`), unit `Nm` vs `N·m` (`sweeps.rs:73-77`) | dashboard/limits ("+inf °C" where a limit does not apply), table, exports | `+inf` -> `no limit` on screen; keep the token in CSV/JSON (shared by design, `format.rs:46-52`). |
| O-7 | `Diameter and length edits resize the associated housing, keeping its allowances. Gap and magnet count remain your choices.` (`size_controls.rs:30`, drawn `:344`) | Size tab footnote under the sliders | Misdescribes what the Outside diameter slider does (it moves the ring radius, cap OD and cap thread). `Outside diameter changes the ring radius, and the cap OD and cap thread follow it. Length changes the magnet length and the housing around it.` |
| O-8 | `A suggested order: torque, back iron, temperature, demagnetization, slip heating, clamps` (`explorer.rs:486`) vs the bar `Start here 1 of 11: Harmonic decomposition` (`explorer.rs:116-123`) | hover of the Start here button | Hover lists 6 topics, the walk has 11 steps starting with Fourier harmonics. Update with the new note order. |
| O-9 | note titles with internal vocabulary: `Back iron: the sinh factor against free space`, `Demagnetization: knee, permeance and the onset temperatures`, `Eddy-current slip loss and skin depth`, `Remanence against temperature, and torque ∝ Br²`, `The one-point calibration` (`notes.rs:95,230,302,198,167`) | equation panel heading, Start-here bar, dashboard Why links (`dashboard.rs:~765`) | rewrite with the note. |
| O-10 | `Why:`-link target `no note yet` (`dashboard.rs:107`) | 15 of 17 dashboard rows had it pre-slice-1; slice 1 links assumptions | shrinks when the three new notes land; keep the text. |

---

## 3. The `E3 E7 E8` badges

**What they are.** Correction ids from the approved-deviation registry (`engine/deviations.rs`,
`REGISTRY`). E1 to E14 are the rows of the M1 math audit
(`docs/analyses/2026-09-29-magcoupling-math-audit.md`); E15 to E20 come from the Addendum A
verification (`docs/analyses/2026-09-30-magcoupling-addendum-a-verification.md`). Each is "an
error found in the original Excel workbook that the Rust port corrects". 19 are `Applied` and
`Engine` class and therefore mark values; E14 rewords help only and never marks
(`corrections.rs:158-160`). They are not evidence levels, confidence grades or equation numbers.

**When a value gets one.** `CorrectionIndex::build` (`corrections.rs:86-127`) ties a workbook
cell to a correction when (a) the correction changes the cell at the defaults (the registry's
`changes_at_defaults`, or the golden files for the broad E3, E4, E5), (b) the audit report names
the cell, or (c) one of its probes shows the correction changing the cell off-default. So E3
(the N42SH remanence 1.29 to 1.30 T) lands on most torque values; E20 and E19 on the temperature
limits; E9 and E10 on the cup wall; E8 on clearance and on the pull-out; E2 on the clamp screw.
A marker says "this number differs from the original spreadsheet", which is not "this number is
less certain".

**Where drawn.**
- Dashboard: `ui.small(&line.marker)` right of the value (`dashboard.rs:563-569`). Small grey
  text, no hover of its own; the row's hover carries the detail. At the defaults 14 of the 17
  rows carry one (pull-out E3 E7 E8; 20 °C E3; hot-low E3; hot-min check E8; cold-high E3;
  ripple E3; mass E9; running clearance E8; clearance screening E8; cup wall E9 E10; governing
  limit E3 E19 E20; hot-day margin E3 E19 E20; temperature verdict E20; recommended screw E2 E8;
  per audit screenshots 02 and 13, markers unchanged by slice 1). Cup OD, space claim and the
  end-effect check carry none.
- Results table: the fourth, unlabelled column (`results_table.rs:739-740`). The test requires
  more than 100 and fewer than half of the rows to carry one (`corrections.rs:315-334`).
- Hover: `marker_tooltip` (`corrections.rs:175-195`) appended to the result tooltip
  (`dashboard.rs:426-428`).
- Equation panel: `Embodies corrections: E7, E8, E3` (`explorer.rs:676`), the order being the
  registry's, not sorted.
- Exports: `Corrected vs workbook` columns on the Dashboard and Results sheets
  (`spreadsheet.rs:356,537`), and the Summary lists every applied correction with its title
  (`spreadsheet.rs:377-383`). **That list is the only legend that exists anywhere.** On screen
  the ids are never defined.

**Does it serve a purpose?** Yes, for the owner and a reviewer: traceability to the original
workbook and to the audit (spec M4 "Layout": "corrected values carry a 'corrected vs workbook'
marker with the deviation in its tooltip"). For an engineer using the calculator it is noise
that looks like an equation number or a warning, and the hover cites repo paths a visitor on
colesorkness.com cannot open. Nothing in the engine reads the badge text; only the exports and
tests do.

**Recommendation.** Keep the data and the export. On screen, draw the badge only when a header
toggle "Show spreadsheet details" is on (GUI-only state like `welcome_dismissed`, not in files,
links or undo), with a one-line legend above the dashboard when on: "E-numbers are corrections
to the original spreadsheet; hover a value to see what changed". Reword the hover to
`Differs from the original spreadsheet:` + the plain title + `was 2.647, now 2.688` and drop the
repo path. If the owner wants no toggle, drop the ids and keep a single `≠`-style mark with the
same hover.

The 20 titles as the hover shows them (`deviations.rs`, near-plain already; E14 is not shown):
E1 Adhesive shear modulus does not match the selected adhesive. E2 Clamp screw length leaves
out the clamp slit. E3 Library remanence of the N42SH parts is below the vendor's published
minimum. E4 Pole sweep counts the inner bondline as hub wall. E5 Rear-web eddy loss uses the
free-space field instead of the field at the steel surface. E6 Cup wall at the flats counts the
outer bondline as steel. E7 Pull-out is taken at half a pole pitch, which is not always the
maximum. E8 Arc-magnet mode reuses flat-block corner geometry in three formulas. E9 'No back
iron' is applied to the torque but not to the cup's mass and wall check. E10 Gap flux density
averages Br instead of summing each magnet's MMF. E11 22 °C adhesive fatigue screen ignores
the fatigue-endurance input. E12 No guard for a hot-day start already above the temperature
limit. E13 A measured drag of exactly 0 stops the whole calculation. E15 to E17 end with
"(E9 residual n)": strip that suffix on screen (it points at another id). E15 Heat capacity
prices the aluminium cup, boss and hub at steel specific heat. E16 The disc removed from the web
in the aluminium-adapter variant is priced at steel density. E17 The cup, web and hub eddy
losses use the steel skin-depth model, steel constants and steel-circuit fields for aluminium
parts. E18 The adhesive thermal-mismatch screen uses 4140's CTE and modulus for a hub that the
mass model makes aluminium. E19 The SuperMagnetMan arc parts carry a maximum temperature above the vendor's 60 C,
and M5045's grade contradicts its specification grid. E20 The demagnetization check uses one
N42SH coercivity curve for every magnet and checks only the inner ring. Several are too long
for a hover; the export keeps the full titles.

---

## 4. Proposed plain vocabulary (one term per quantity)

| Quantity | Use on screen | Never on screen |
|---|---|---|
| `coupling.inner_back_apothem_mm` | **Ring radius** ("distance from the shaft centre to the back of the inner magnets") | apothem, inner back apothem, ring size, hub apothem |
| Size-tab slider | **Outside diameter** (largest of cup, cap, boss; follows the ring radius) | rotating OD |
| `model.cup_od_mm` | Cup outer diameter | |
| `model.gap_radius_mm` | Gap radius R_g | Gap mean radius, Mean gap radius (pick one) |
| `coupling.npole` | **Magnets per ring** (one block per pole) | poles per ring, magnet count |
| `metal.face_gap_mm` | **Magnet face gap** | Candidate flat-face magnetic gap, effective gap |
| `metal.required_min_Nm` | **Required torque** | Target hot-low torque, Required minimum service pull-out torque (input label stays, python-locked), requirement |
| `model.pullout_Nm` | **Pull-out torque (nominal)** | Expected torque |
| `metal.torque_hot_low_Nm` | **Hot-low torque** | Hot-side low torque, hot minimum, "guaranteed" anything |
| `metal.torque_cold_high_Nm` | **Cold-high torque** | Cold torque with assumed positive variation |
| `metal.variation` | **Production variation allowance** (15 %) | Symmetric torque variation allowance, assumed variation |
| original Excel model | **the original spreadsheet** (once, defined in the details toggle hover) | workbook, Calculator!, Rust-only, M3 |
| corrections | **differs from the original spreadsheet** | corrected vs workbook, E-ids (details only) |

---

## 5. How to remove it: mechanisms, tests to change, open decisions

### 5.1 Mechanisms (smallest set that clears the inventory)

1. **GUI-side label and hint maps keyed by input or result path** (new, e.g. `gui/plain_text.rs`
   or inside `inputs.rs`): ring radius and magnets-per-ring labels (R-1, N-1), 87 new helps (the
   slice's item B), the cup-wall verdict (W-16), the dashboard alias labels (T-10, T-14, O-3).
   The engine metadata stays parity-locked. This is the only route for ported fields.
2. **One provenance toggle** ("Show spreadsheet details", default off) gating W-1..W-5, A-1..A-3,
   C-3. One new bool in `MagcouplingPanel`; the data (`meta.cell`, `marker`) is unchanged.
3. **Rust-only text rewrite in place** (Appendix A, W-18, A-6, A-7, A-8, C-7): free, then re-bless
   `tests/data/input_schema.json` for the inputs.
4. **Assumptions**: add a plain `why` field (or rewrite `rationale`) and keep `source` for the
   export and `tests/assumptions.rs:123-127`.
5. **Notes rewrite** removes A-5; render `sources` collapsed (A-4).
6. **Display-layer verdict map** for the dashboard value only (T-12, T-13, T-16, W-16, R-10).

### 5.2 Tests that pin text and must change with it

- `inputs.rs:1379-1383` (exact tooltip), `1395-1403` (harmonic tooltip), `1499-1501`.
- `dashboard.rs:1330,1474,1484,1500,1504`; `results_table.rs:775,825,875,951` (CSV and JSON rows keep
  the cell: unchanged); `panel.rs:2619` (cell column on screen); `corrections.rs:221-248`
  (only if the tooltip format changes).
- `explorer.rs:1170`, `panel.rs:6701,6906` and `6659-6702` (cell line and `Embodies corrections`).
- `geometry.rs:1166-1167,1192`, `geometry_view.rs:523,525`.
- `guidance.rs:3287-3330` (the guard's own examples), `3307,3321`, `3996` (status line),
  `3850-3859,4537` (ring size), `panel.rs:5446`, `sizing.rs:797`, `size_controls.rs:528`.
- `spreadsheet.rs:948` and the `column(...)` lookups, if export headers are renamed.
- `tests/assumptions.rs:123-127` if `source` stops carrying cells.
- Tests that only use the constants (`STORED_3D_LABEL`, `NO_EQUATION`, `RESET_ASSUMPTIONS`,
  `InputOrder::Workbook.label()`, `ONE_PIECE_ONLY`, `FILTER_HINT`, `SEARCH_HINT`) need no edit.

### 5.3 Surfaces the slice-1 guard does not cover (apply `plain_text_violations` here)

Dashboard labels and values and every hover (`result_tooltip`, `input_tooltip`), geometry notes,
the explorer lines and tags, notes (`sentences`, `watch_out`, `title`), assumption `rationale`,
results-table hints and empty texts, the Assumptions banner and button, plot legends and axes,
clamp notes, status lines. Extend `BANNED` (`guidance.rs:1442`) with: `M1 audit`, `skeptic`,
`Plan A-`, `decision`, `Addendum`, `drawing.py`, `.rs`, `Metal design`, `Temperature design`,
`Shaft clamps`, `Calibration!`, `M3` (whole word, not a screw size), `correction E\d`,
`Rust`. Whitelist: screw sizes `M2.5 M3 M4 M6`, `M41 × 0.5`, grades `N42SH B842SH Y30 M5045`,
`Hcj`, `Br`, `ISO 4762`.

### 5.4 Decisions for the owner

1. Is the badge toggle acceptable, or should the ids go entirely (single mark)? (3)
2. Rename `Workbook example design` (slice-1 G-6 named it)? (W-9)
3. Canonical radius wording: `Ring radius` for the input and `Outside diameter` for the slider,
   joined by one sentence, or one name for both? (2.1)
4. Dashboard verdicts: plain display map on the dashboard only, or leave the engine wording
   everywhere and rely on the guide blocks? (T-12, W-16)
5. Keep `Calculator!C93` style cells visible to power users behind the toggle, or only in the
   export? (W-1)
6. Assumption `source`: split field, or relax `tests/assumptions.rs`? (W-13)

---

## Appendix A: help and label declarations with jargon (83)

Found by scanning `param`, `param_rust_only`, `out`, `out_rust_only` declarations for cell
references, `E<n>`, `Audit M<n>`, `Plan A-<n>`, `decision`, `Addendum`, `workbook`, `Rust-only`
in the label or help. All reach the screen only as hover text (input tooltip, result tooltip,
equation term hover), except where noted. Kind: I = input, R = result; `ro` = Rust-only (free to
edit).

- `calibration.rs`: 110 R ro (End-effect model check, `Audit M9`); 115 R ro; 121,123,125,127,129,131 R ro (harmonic torque-angle amplitude, `Plan A-3`, `E7`); 133 R ro (Pull-out electrical angle).
- `clamps.rs`: 237 R (Screw length, `correction E2`); 262 R ro (Screw length note, `correction E2`).
- `housing.rs`: 52, 54, 56 R ro (beyond the space claim, `Metal design C135/C131/C134/C130/C137/C129`); 60, 62, 64 R ro (`C123/C124/C172`, `decision A2-8`).
- `materials.rs`: 44 I (Density, `Metal design C132`); 133, 145, 153 I ro (part materials, `Addendum A5`, `the workbook's 4140/316L/6061-T6`, `C13 C43 C44 C42 C111 C139 C140`); 187 R ro (autofit); 193, 198, 200, 202, 204, 206, 208, 210, 212, 214 R ro (`... in effect`, cells, `Plan A-3`, `E9 E15 E17`).
- `metal_design.rs`: 299, 312 R (`the workbook's aluminium`, `E9 E16`, `Addendum A5`).
- `model.rs`: 147 I ro (Axial magnet length, `Addendum A1`, `C42 C123 C124 C172`, `decision A2-8`); 182 I ro (Highest odd harmonic summed, `Addendum A3`, `C95 C96`, `Rust-only tau7_Pa`); 288, 294, 300, 306 R ro; 312-322 R ro (`Plan A-3`, `E7`); 331 R ro (End-effect check, `Audit M9`, dashboard row 17); 340, 342, 344 R ro (`E7`, `C95 C96`, `the workbook's assumption`); 370, 374 R ro (`decision A2-7`, `C22 C35 C110`).
- `sweeps.rs`: 52 R table (Swept variable, `workbook 'Corner gap (mm)'`).
- `temperature.rs`: 78, 81 I (`Correction E20`); 105 I ro (`C44 C45`); 200 I (`Correction E1: the workbook's 0.55 GPa`); 246 I (`correction E5`, `the workbook's 1.035e-5`, `E17`); 257, 260, 263 I ro (`E17`, `C116 C117 C121`, `decision 12`, `until M3 computes it live`); 475-506 R ro (16 rows: `Correction E20`, `C44 C45 C47 to C60 C51`, `Plan A-3`, `the workbook checks the inner ring only`); 559, 562, 565 R (`the workbook's 6061 aluminium`, `E9 E17`).
- `warnings.rs`: 99, 105, 107, 109 R ro (`Calculator C6`, `Materials C13 C26`, `Temperature design C95`, `the workbook names`).

Dashboard rows whose hover carries jargon: `End-effect model check` (A-6) only. The other 16
dashboard rows' helps are clean (`Analytical estimate, not a guaranteed minimum.` etc.).

## Appendix B: false positives (do not "fix")

Screw sizes `M2.5`, `M3`, `M4`, `M6` (`clamps.rs:62,241`, `Through the M6 vent port`,
`Allowable preload per M3`, `Pilot + 3 × M3`); thread `M41 × 0.5` (`metal.cap_thread_dia_mm`
help); magnet and grade names `B842SH`, `N42SH`, `Y30`, `M5044/M5045/M5026` (library parts);
`ISO 4762 M4 x 14, class 12.9`; `E = 0.300 GPa` (Young's modulus symbol in
`temperature.rs:200`); `H900`, `H1150` (steel conditions); the `Level` words `OK / Check /
Fails`. `grades.rs` and `material_library.rs` `notes`/`condition` fields with `R16`, `decision
17 A`, `M13` are **not drawn** (only `condition` is, `pickers.rs:103`). `Review::Reviewed.record`
(`commit ee11bfd`) is not drawn. Table-field labels in `explain/tables.rs` (`decision 20`,
`Materials C38`) are not drawn (only their symbols are, `render.rs:142`, `typeset.rs:794`).

## Appendix C: jargon hits inside `engine/explain/notes.rs` (current line numbers)

Sentences and watch-outs: 77, 131, 152, 155, 175, 181, 249, 252, 321, 354, 357, 478, 482, 502, 524.
`sources` lines (drawn as `Sources: ...`): 84-85, 107-108, 136-137, 158, 185-188, 219-220,
256-258, 291-292, 325-328, 361-363, 393-395, 416-417, 438-439, 463-464, 487-488, 509, 530-531.
Not drawn: the 17 `review: Review::Reviewed { record: ... }` strings (lines 90, 113, 142, 162,
193, 225, 263, 297, 333, 368, 400, 422, 444, 469, 493, 514, 536; they say `plan A-3 Task 16
physics review ... commit ee11bfd` or `independent physics review 2026-10-01 ...`). Slice-1 guidance text (`guidance.rs`) is
clean of every id above.
