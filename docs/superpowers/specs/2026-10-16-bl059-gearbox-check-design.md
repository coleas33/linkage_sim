# BL-059: a gearbox check of the coupling's cold-high torque — design

- Date: 2026-10-16. Branch `magcoupling/bl059-gearbox-check` from `main` at 3cc0952 (guidance slice 2 shipped).
- Backlog: BL-059 (`risk: physics`, priority 2). Research (gitignored): `.superpowers/sdd/2026-10-16-bl059-gearbox-check/`:
  `research-architecture.md`, `research-physics.md`, `research-surfaces.md`, `synthesis.md`.
- `M` = `magcoupling-rs`. Torques are in N·m; every torque in this spec names its side of the gearbox.

## 1. Purpose

Nothing in the calculator compares the coupling's strongest expected torque with the gearbox it sits on. The drive line is
motor -> GAM 5:1 gearbox -> magnetic coupling (on the gearbox **output**) -> wheel (owner, 2026-10-14). When the wheel is
held, the coupling passes up to its pull-out torque before it slips, and while it slips the gearbox carries a fully
reversing swing of the same amplitude. The strongest expected pull-out is **cold-high**: the pull-out at the minimum magnet
temperature with the + production allowance (`metal.torque_cold_high_Nm`).

BL-059 adds one counted dashboard check, with its slice-1 guide (Problem, Why, Fix), that compares cold-high with the
gearbox rating, and corrects the texts that say nothing checks it.

Success: an engineer sees, on the dashboard, whether the coupling's cold-high torque is within the gearbox rating; when it
is not, the guide says by how much, what to change, and — when no magnet size, gap or count can pass both this and the
hot minimum check (hot-low against the requirement) — says so with numbers and names what would have to change instead.

## 2. Owner decisions (2026-10-16, all the recommended options)

| # | Decision |
|---|---|
| 1A | **Basis: the gearbox input.** Compare `metal.cold_high_input_Nm` (cold-high ÷ ratio ÷ efficiency) with `coupling.gearbox_input_rating_Nm` (0.6). That is cold-high ≤ rating × ratio × efficiency = 2.85 at the output (the workbook's own `model.gearbox_reference_Nm`). It is never unconservative: in the forward-drive half of each slip cycle the input carries T/(i·η). |
| 2A | **Build now with 0.6**; the owner checks the GAM datasheet line later (an input or output figure; nominal, acceleration or emergency-stop rating). BL-059's notes record the open question. |
| 3A | **Counted and red.** The example design reads "fails 4 of 8 checks" while the shaft-clamp section is paused. |
| 4A | **Say the conflict in the guide.** When no magnet size, gap or count can pass both this check and the hot minimum check, the Problem adds a computed sentence with the numbers and what would have to change. |
| 5A | **Auto unchanged.** Auto still sizes for the hot minimum check (hot-low against the requirement) only. |
| 6A | **Fixes: Lower Axial magnet length, Raise Magnet face gap.** Both also lower hot-low; the existing trade-off and side-effect lines say so. |
| 7A | **Change only what becomes false or misleading:** the two "nothing checks the gearbox" watch-outs, the rating's help, the new guide texts, and an overlay that names the ripple row as the nominal swing at the operating temperature (label in section 5). |
| 8A | **Fatigue goes to the backlog.** A new item: the repeated slip swing against the gearbox's nominal rating. |
| R | **Release: stop for the owner's review of the diff** (AGENTS.md `risk: physics`) after the reviews, GATE PASS with gate 12 and gui-smoke; merge and push (which deploys) only on the owner's OK. |

Technical decisions taken as recommended (synthesis 9-14): a new Rust-only result group `gearbox`; the check greys with
the end-effect rows; the Why link goes to `hot_low_cold_high`; the "protects the gearbox" ban stays and widens to guide and
help texts; `passing_design()` gets a gearbox rating of 1.0; no new note sign-off sheet (the owner reviews the diff).

## 3. Physics basis (numbers from `compute_all`, main at 3cc0952)

- `metal_design.rs`: cold = T20·th_i(Tmin)·th_o(Tmin); hot_low = T_op·(1 − v); cold_high = cold·(1 + v);
  `cold_high_input_Nm = cold_high / (i·η)` (`:509`). `model.gearbox_reference_Nm` = rating·i·η = 2.85 (`model.rs:1157`).
- The band **B = cold_high / hot_low = [th_i(Tmin)·th_o(Tmin) / (th_i(Top)·th_o(Top))]·(1 + v)/(1 − v)**. T20, Br,
  geometry, magnet count and back iron cancel. At the defaults (−40/50 °C, α −0.0012 /°C, v 0.15) B = 1.6731 for every
  NdFeB design; Y30 gives 1.9207.
- A design that just meets the requirement R has cold-high R·B at the output and R·B/(i·η) at the input:
  2.5·1.6731 = 4.183, 0.881 at the input. The largest R compatible with the rating is rating·i·η/B = 1.703.

| design | hot-low | cold-high (output) | at the input | check (≤ 0.6) |
|---|---|---|---|---|
| default | 2.2852 | 3.8233 | 0.8049 | fail |
| 20 mm ring radius | 2.1584 | 3.6111 | 0.7602 | fail |
| 8 magnets per ring | 1.6731 | 2.7992 | 0.5893 | pass (1.8 % margin) |
| 304 back iron | 1.4428 | 2.4138 | 0.5082 | pass |
| Y30 ferrite, custom dimensions | 0.1760 | 0.3381 | 0.0712 | pass |
| N33AH both rings, custom-dimension base | 1.7573 | 2.9401 | 0.6190 | **fail** (passes a 3.0 output basis: pins 1A) |
| Auto on length, defaults (13.770 mm) | 2.5000 | 4.1827 | 0.8806 | fail |
| Auto on length, minimum temperature 20 °C (its slider's top), v 0 | 2.5000 | 2.6902 | 0.5664 | pass (the Problem is None) |
| default, c_end 1.5 (end-effect model out of range; outside its 0-0.5 slider, so tests use `short_magnets()`) | −0.1031 | −0.1726 | −0.0363 | greyed, not counted |

What the check does **not** cover (it is necessary, not sufficient):
- the fully reversing slip swing as a fatigue load (16.7 cycles, about 33 reversals, per 0.1 s event; 3.33e8 cycles per
  life at the defaults, `metal.magnetic_cycles`); a peak rating says nothing about that (8A);
- backlash impacts at each reversal, the ring-down at re-lock, torsional resonance crossings as the slip frequency sweeps
  0-167 Hz, and the drive line's stored kinetic energy at a jam: each can exceed cold-high at the shaft;
- the gearbox's own temperature range (the old GAM catalog lists −10 or −20 °C; the coupling's minimum is −40 °C).

## 4. Engine

**New module `M/src/engine/gearbox.rs`** (Rust-only, like `housing.rs`):
- `GearboxResults { cold_high_check: String }`, declared with `out_rust_only` (no cell). Engine label
  "Cold-high torque at the gearbox input against its rating"; help (≤ 30 words, plain): "Compares the cold-high torque at
  the gearbox input, cold-high divided by the ratio and efficiency, with the gearbox input torque rating. A peak check
  only."
- A pure `compute(cold_high_input_Nm: f64, rating_Nm: f64) -> GearboxResults`:
  - pass when `cold_high_input_Nm <= rating_Nm` (equality passes; the hot-minimum check fails only strictly too);
  - fail when strictly above, and when `cold_high_input_Nm` is +∞;
  - "unknown" when either value is NaN.
- Verdict texts, as named constants (plain words, glyph-safe, no internal paths):
  - `WITHIN_THE_GEARBOX_RATING` = "Within the gearbox input rating";
  - fail: `ABOVE_THE_GEARBOX_RATING` prefix "Above the gearbox input rating", then ": {q} N·m against {r} N·m". r is the
    rating to 2 decimals. q is the cold-high at the input to 2 decimals and at least 0.01 above the quoted rating,
    `fmt_fixed(py_max(x, r_quoted + 0.01), 2)`, as `housing::quoted_overshoot` does (`housing.rs:89-93`), so a red verdict
    never shows two equal numbers. Tested at the defaults ("0.80 N·m against 0.60 N·m") and at the defaults with the
    rating at 0.80, a slider step ("0.81 N·m against 0.80 N·m");
  - `GEARBOX_CHECK_UNKNOWN` = "Gearbox check unknown: a torque is not a number".
- `M/src/engine/mod.rs`: `pub mod gearbox;`. `M/src/engine/api.rs`: a `gearbox: GearboxResults` field beside `housing`,
  computed beside `housing::compute` from `metal.cold_high_input_Nm` and `ci.gearbox_input_rating_Nm`.
- **Unchanged:** every Python-ported input, default, range, label, help and result; `tests/data/*.json`;
  `input_schema.json`; `reference/magcoupling-py`. Gate 12 parity is unaffected (a new Rust-only group only).

## 5. GUI

**Dashboard (`M/src/gui/dashboard.rs`):**
- `DASHBOARD` 17 -> 18 rows: `("gearbox.cold_high_check", Some(same))` at index 15, right after the 15 headline rows, so
  `paths[15..]` = [gearbox, space claim, end effect]. The GUI-only Magnets fit row stays last (the smoke checks it).
- `CHECKS` 30 -> 31, the new check appended last (its schema-order test, `dashboard.rs:1320-1328`); `verdict_level`: Good for `WITHIN_…`, Bad for the `ABOVE_…` prefix, Caution for `…UNKNOWN`. Every
  verdict text stays classified (an unclassified text reads green: `guidance.rs` `check_level(..).unwrap_or(Good)`).
- `END_EFFECT_ROWS` 7 -> 8: the check greys, uncounted, when the end-effect model is out of range (at c_end 1.5 the
  negative cold-high would otherwise read as a pass). The headline-only test is re-scoped to "the DASHBOARD rows that move
  with c_end" in `headline_moved_by`; the greyed list at `dashboard.rs:1612` gains the row.

**Guide (`M/src/gui/guidance.rs`):** `GEARBOX_GUIDE` in `CHECK_GUIDES` (8 -> 9) between the clamp guide and the space-claim
guide, so the guides stay in dashboard order; counted; its `ROW_LEARN` entry and `Learn::Note("hot_low_cold_high")`.
Its `label` (required and distinct, `guidance.rs:2300`; it shows in the trade-off line at the defaults) is
"Gearbox input torque". Numbers use the house readout format (`result_text`, 4 significant digits, `guidance.rs:10-11`).
- **Problem** (live numbers; drafts, final words set by the G-3 review):
  - fail, sentence 1: "Cold-high at the gearbox input is {cold_high_input} N·m, above its {rating} N·m input rating."
  - fail, sentence 2, only when the conflict holds: "No magnet size, gap or count passes both this and the hot minimum:
    cold-high is {B} times hot-low here, so meeting the {required} N·m requirement puts {R·B/(i·η)} N·m on the gearbox
    input. It takes a gearbox rated for that, a requirement of at most {rating·i·η/B} N·m, or a band of at most
    {rating·i·η/R} times (a warmer minimum temperature, a smaller measured variation or SmCo magnets narrow it)."
    At the defaults: 1.673, 0.8806, 1.703 and 1.140. No lever in the parentheses passes alone at the defaults (variation 0
    gives 0.6509; minimum 20 °C gives 0.7663; SmCo 2:17 gives 0.7581); the sentence names the target, not a single lever.
  - unknown (Caution; the Problem must be Some there, `guidance.rs:3444-3450`): "The gearbox check cannot be made:
    cold-high at the gearbox input or the rating is not a number."
  - The conflict holds when `hot_low > 0` and `required · (cold_high / hot_low) / (i·η) > rating`, every term read from
    the results and inputs (`metal.required_min_Nm`, `metal.torque_cold_high_Nm`, `metal.torque_hot_low_Nm`,
    `coupling.gear_ratio`, `coupling.gear_efficiency`, `coupling.gearbox_input_rating_Nm`). It is well defined at every
    reachable input: the variation tops out at 0.5, so hot-low ≤ 0 only when the end-effect model is out of range (the row
    greys); a NaN makes the comparison false; ratio ≥ 1 and efficiency ≥ 0.5, so i·η is never 0.
  - Only the Why has a cap (30 words, one sentence, `guidance.rs:2308-2318`); the Problem has none, but each sentence above
    stays one sentence. `the_problems_quote_numbers_from_the_results` (`:3577`) gets a GEARBOX case that recomputes B and
    the derived numbers from the results.
- **Why** (one sentence): "Up to cold-high torque reaches the gearbox output before the coupling slips, and in each swing
  while it slips; at the gearbox input it must stay within the rating."
- **Fixes:** `Fixes::Static`: Lower `coupling.magnets.axial_length_mm`, Raise `metal.face_gap_mm` (both monotone over their
  sliders). The existing "One input pulls two ways" trade-off line and the face-gap side effect print in the block. No new
  `SIDE_EFFECTS` entry: at the defaults it would duplicate the trade-off line, and on a passing or Caution check its outcome
  ("rises further above its rating") would be false (`side_effects` also fires on Caution, `guidance.rs:550`).
- `terms()`: the margin is rating − cold-high at the input. The doc comments' check counts move ("eight dashboard checks").
- **Not offered as fixes:** the variation (an assumption input), the minimum temperature (an environment fact), the rating
  or ratio (not a real part). They are named only in Problem sentence 2 and in the torque-band note.

**Overlays (`M/src/gui/plain_text.rs`):**
- `coupling.gearbox_input_rating_Nm` HELP rewritten: "The gearbox's input torque rating. The gearbox check compares the
  cold-high torque at the gearbox input with it." (the engine help stays Python-locked; the rating's type is still open,
  2A, so no text calls it a peak rating).
- `model.gearbox_input_ripple_Nm`: a LABELS and HELP row (label cap 47 characters, `plain_text.rs:1001-1004`). Label
  "Nominal slip swing at the gearbox input"; the help says it is the pull-out at the operating temperature without the
  production allowance, so it reads below cold-high at the input (0.566 against 0.805 at the defaults). No "warm" in the
  label: the operating-temperature slider reaches below the minimum temperature.
  `the_canonical_names_are_the_spec_s` (`plain_text.rs:978`) gains the row.
- The new check's on-screen label comes from the engine label unless the G-3 review asks for an overlay.

**Result groups:** `PACKAGE_LABELS` 12 -> 13 ("Gearbox"). Exports (CSV, JSON, XLSX) pick the new row up without code
changes; the share link and design files store inputs only, so they do not change.

**Auto (5A):** no change to `sizing.rs` or the Size tab. Under Auto on any variable, both fixes are undone by the re-solve:
Auto brings hot-low back to the requirement and B does not depend on geometry (`gui/sizing.rs:10-18`). So under a solved
Auto the check fails exactly when Problem sentence 2 shows, and that sentence is the guide's real answer there. Under
Auto on length the length fix also points at a locked row, as the space claim's length fix already does; documented, not
special-cased.

## 6. Notes and texts (physics review)

- `hot_low_cold_high` watch_out[1] (`notes.rs:584`) and `when_slipping` watch_out[0] (`notes.rs:745`) say nothing checks
  the gearbox; both become false. Drafts (the physics reviewer sets the words; caps and plain-words rules hold):
  - torque band: "The gearbox check compares cold-high at the gearbox input, {metal.cold_high_input_Nm} here, with its
    input rating; repeated slips also load the gearbox in fatigue, which nothing checks."
  - slipping (27 words): "The gearbox output carries these swings; cold they can reach cold-high,
    {metal.cold_high_input_Nm} at its input here, and the gearbox check compares only that peak with its rating."
  - Their sources name the check and the rating's open datasheet question; their review dates move.
- Each note gets a fresh-context session-model physics re-review, recorded as a `### BL-059 re-review (<review date>)`
  subsection under its section of `docs/analyses/2026-10-14-magcoupler-notes-review.md` (history is not rewritten).
- The guide's label, its Problem (fail sentences 1-2 and the unknown text) and Why, the three engine verdict texts, the
  two overlay rows and the engine label and help get one G-3 review (plain words for
  engineers who are not magnet specialists; nothing false at the default, the probe designs or any reachable input),
  recorded in `docs/analyses/2026-10-14-magcoupler-help-review.md` under a new `## BL-059 gearbox check` section.
- Traps the reviewers check: name the side of the gearbox for every torque; efficiency raises the input torque (never
  "reduces the load"); cold-high is an expectation, "up to" or "can reach", never "at most" and never what every unit
  reaches; never "the coupling protects the gearbox" (a pass means only that the modelled cold-high peak is within the
  input rating); "peak" names the compared torque, never the rating (its type is open, 2A); the ripple row is the nominal
  swing at the operating temperature; geometry cannot narrow the band; name the side for hot-low too (the output); no
  "hot-day" in these texts (the hot minimum check uses the operating temperature); no internal paths, ids or "clamp"
  while that section is paused.

## 7. Tests (red first where it is behaviour)

1. **Engine (`gearbox.rs`):** the defaults fail with the "Above" text carrying 0.80 and 0.60; 8 magnets, 304 and Y30 pass;
   20 mm fails; the N33AH design (0.6190 at the input) fails, pinning basis 1A; exactly at the rating passes and the next
   float above fails; NaN cold-high or NaN rating reads unknown; +∞ reads above. `tests/robustness.rs` keeps running over
   the extremes.
2. **Rust-only:** `meta.rust_only`, no cell; the parity suites (`python_schema`, `parity`, `differential`, `deviations`)
   pass unchanged; the label and help pass the schema tests.
3. **The tripwire's replacement:** `nothing_checks_a_gearbox_torque_against_its_rating_yet` becomes
   `the_gearbox_check_is_a_dashboard_row_with_its_guide` (exactly one gearbox check row, in `CHECKS`, with a guide whose
   learn is the torque-band note; the ripple row stays a value row).
4. **Classification and greying:** Good, Bad and Caution for the three texts; on `short_magnets()` (end-effect model out
   of range) the row is greyed, uncounted, absent from `failing_checks` and the results-table badge.
5. **Status:** the defaults read "Example design: fails 4 of 8 checks" while paused (4 of 9 unpaused);
   `passing_design()` passes all 8; with its rating back at 0.6 it fails the gearbox alone.
6. **Problem:** present exactly when the check fails or is unknown; at the defaults it carries the live 0.8049, 0.6,
   1.673, 0.8806, 1.703 and 1.140 and sentence 2; on the gearbox-only design (`passing_design()` with the rating back at
   0.6: cold-high at the input 0.7070 fails; R·B/(i·η) = 0.5636) the Problem is present and sentence 2 absent; on minimum
   temperature 20 °C (the slider's top), variation 0, Auto length (11.902 mm, 0.5664) the check passes and the Problem is
   None; the unknown text shows on a NaN rating; plain-words, glyph and clamp-ban tests cover it.
7. **Fixes:** both fixes show at the defaults and move the check toward passing over their sliders; the trade-off and
   face-gap side-effect lines appear; the fix tests run over the existing designs plus "gearbox only" (`passing_design()`
   with the rating back at 0.6). "Near the gearbox limit" (the defaults with the rating set within one length step's rise,
   about 0.0007 N·m at the input, above cold-high at the input) is a dedicated test, not a `designs()` entry (there the
   hot minimum guide's only fix is held back by `helps()`, which would break
   `each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red` by construction): it asserts the
   hot-low block shows `NO_FIX` and the gearbox check stays green.
8. **Text bans:** no guide or overlay text says the coupling protects the gearbox (new); the note ban stays.
9. **Notes:** the torque-band and slipping note tests pin the new watch-outs at the five probe designs (the check fails
   exactly when `cold_high_input_Nm > rating`; cold-high at the input exceeds the ripple row); caps, sentence-zero,
   plain-words and `release_notes_are_reviewed` pass with the new records.
10. **Panel and exports:** the gearbox row's own Why link (not the first copy of the title drawn, which is the hot-low row's)
    opens the torque-band note; every dashboard row opens in Explain; CSV and JSON
    carry the row with an empty cell column; the XLSX Summary lists 18 dashboard lines.
11. **Auto (documents 5A):** Auto on length at the defaults solves 13.770 mm and the gearbox check is red there.
12. **Pinned tests and counts** that move, each confirmed by running: status texts; failing lists; row counts
    (`panel.rs:3141` 7 -> 8); `trace.rs` totals (392, 1086 -> 1087);
    `two_failing_checks_that_pull_one_input_apart_say_so` (`guidance.rs:3276`: the defaults now pull the length both
    ways); `raising_the_face_gap_says_it_lowers_the_failing_hot_low_torque` ("no other guide gets the note", `:4930`);
    `the_canonical_names_are_the_spec_s` (`plain_text.rs:978`); the greyed list (`dashboard.rs:1612`); the comments that
    name the tripwire (`tests/explain.rs:2497`, `:2836`).

## 8. Docs and backlog

- `M/README.md`: the dashboard rows and counts, the checks and guides, the Why targets, the test table rows, the
  protect ban, and the Python-parity paragraph (a Rust-only group added).
- `docs/ai/02-system.yaml`, `03-structure.yaml` (module entry, `addendum_only` += `gearbox.rs`, CHECKS 31, CHECK_GUIDES 9),
  `04-memory.yaml`, `05-update-tracker.md` (a new top entry).
- `docs/ai/backlog.yaml`:
  - BL-059: the basis stated as 1A (input-side, 2.85 at the output; the old "equivalently" corrected), the open datasheet
    question (2A), status `fixed` on merge;
  - new **BL-066** (`risk: physics`): a repeated-slip gearbox check of the slip swing against the gearbox's nominal
    (continuous) rating; needs the owner's datasheet value as a new input.

## 9. Release

1. Red tests, implementation, both clippy runs, fmt, the touched suites.
2. The physics re-review of the two notes and the G-3 review of the guide, overlay and engine texts; must-fix items
   applied and re-reviewed.
3. A fresh-context session-model review of the whole branch.
4. `MAGCOUPLING_PYTHON=<venv python> bash linkage-sim-rs/scripts/gate.sh`: GATE PASS with gate 12 running (no SKIP).
5. `build_web.sh`, then gui-smoke on the local server (Magnets fit still the last dashboard row; a Why link opens a note).
6. **STOP:** send the owner a short summary of every changed on-screen text and number. Merge (`--no-ff`) and push only on
   the owner's OK; then watch the deploy and run gui-smoke on https://colesorkness.com.

## 10. Out of scope

- A fatigue or nominal-rating check (BL-066), a new input of any kind, and any change to Auto.
- Re-wording the 14 "shaft-joint and gearbox loads rise" change entries (still true).
- The motor, backlash, stiffness and inertia dynamics above cold-high (not modelled).
- The shaft-to-hub joint (BL-064) and the paused clamp section.

## 11. Risks

- **A path without "gearbox"** would let the tripwire pass silently: the result path is `gearbox.cold_high_check` and the
  tripwire is replaced in the same commit.
- **Fix hiding:** once the check passes, a hot-minimum length fix that would redden it within one slider step is hidden;
  the "near the gearbox limit" design covers that path.
- **Equality:** the check compares in one form only (the input side) and is tested exactly at equality.
- **Dashboard height:** one more row; Magnets fit stays last, so the smoke's scroll check still holds.
- **Parity:** any edit to a Python-locked row fails gate 12; all on-screen rewording goes through the overlay.
