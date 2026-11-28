# Magcoupler R4 batch: text review (G-3)

**Date:** 2026-11-28

**Reviewer:** a fresh-context reviewer on the session model, Opus 5.5, standing in for fable while fable is out.
Fable's physics re-review of this diff is BL-130, on 2026-12-01.

**Branch:** `magcoupling/r4-physics` (worktree `lsim-plan`), `main...HEAD` at b78a1b9: BL-067, BL-083, BL-084 and
BL-087, all `risk: physics`.

**Rule:** spec `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, decision G-3 and section 8.3. Every
new or changed on-screen string gets a text review, and a string that states physics also gets a physics review.
Read: the four backlog entries, the spec and the form of `docs/analyses/2026-10-14-magcoupler-help-review.md`. The
builders' and reviewers' notes beyond the backlog entries were not read.

**Verdict:** MUST_FIX. Two items:

- MF-1 blocks the public site. The shaft joints check's help says "the required torque", but on this page Required
  torque is the hot-low requirement.
- MF-2 does not block. An older lesson step is false in the BL-067 corner.

There are six should-fix items and three nits.

## How it was checked

- **Strings.** Listed from `git diff main...HEAD -- magcoupling-rs/src`. Each string was read against the code that
  picks it (`gui/guidance.rs` `gearbox_problem`, `term_sentence`, `checked_pullout_name`, `check_terms`;
  `gui/size_controls.rs` `torque_gap_text`; `gui/geometry.rs` `cap_skirt_mm` and `wall_cautions`; `gui/panel.rs`;
  `gui/explain.rs` `touched_guides`; `gui/plain_text.rs` HELP) and against the engine (`engine/gearbox.rs`
  `checked_pullout_Nm`, `engine/shaft_joint.rs`, `engine/metal_design.rs`: hot-low is
  `torque_op × (1 − v)`, cold-high is `torque_20C × th²(min) × (1 + v)`).
- **Engine probe.** A scratch crate built on the engine with no GUI features, in the scratchpad (no repo file
  touched). It ran `compute_all`, `api::checked_pullout` and `api::gearbox_checked_input` on
  `DesignInputs::default()`:
  - Defaults: hot-low 2.285204851 N·m, cold-high 3.823310 N·m, cold-high at the input 0.8049 N·m, rating 0.6 N·m.
  - BL-067 corner (operating −40 °C, minimum 20 °C, variation 0.1): hot-low 2.992156, cold-high 3.182327, operating
    pull-out 3.324618, checked pull-out 3.657079 at the output and 0.769911 at the input, cold-high at the input
    0.669964, joint torque 7.314159 N·m.
  - Equal temperatures: on the operating = minimum sweep (−40 to 20 °C in 0.5 °C steps × variations 0, 0.05, 0.1,
    0.15 and 0.2, 605 designs), the operating pull-out with its allowance comes out above cold-high on 180 designs,
    by float noise alone. See SF-1.
- **Python oracle** (`reference/magcoupling-py`, its venv).
  - The same corner gives operating pull-out 3.273666, cold-high 3.133557 and hot-low 2.946300. Its default design's
    pull-out is 1.55 % below the Rust default's.
  - The temperature and allowance factors agree with the engine to five digits: cold-high over the operating
    pull-out is 0.95720 in both, and hot-low over it is 0.9 in both.
  - The oracle has neither check, so it confirms the band physics only. It shows the same float-noise flip at equal
    temperatures (180 of 605).
- **Rendering.** The branch's web build was served at `http://localhost:8080/tools/magcoupler/`. The served wasm was
  checked for the new strings first. Driven with Playwright at 1280 x 800.
  - The designs were opened as share links (`?m=`: deflated, URL-safe base64 design JSON, built by a scratch script).
    Values a link leaves out take the engine defaults. Each value of the BL-067 corner sits on its slider in Detailed
    inputs; the branch's test `below_the_minimum_temperature_the_texts_name_the_operating_pull_out_with_its_allowance`
    sets them through `with_number`.
  - The BL-083 shortfall cannot be typed: the Required torque box rounds to 0.01 N·m. A share link reaches it, with
    Required = hot-low + 1e-7.
  - The cap break-through cannot be reached from the Size tab: the Outside diameter control moves the cap thread
    with the cap (screenshot 24). It is reached through Detailed inputs (Threaded cap OD and Cap thread nominal
    diameter), or through a link or a file.
  - The browser was shared with another session, which moved the first tab to another local build mid-run. All
    renders after that were made in a tab of this review's own. Every render shown here was checked to be on
    port 8080. At the end this review closed its own tab only, so the other session's tab was left open.
- **Screenshots** (saved under `.playwright-mcp/r4` in the main checkout and copied to the session scratchpad
  `r4-text-review`):

| # | File | Shows |
|---|------|-------|
| 0 | `00-starting-design.png` | the clean address |
| 1 | `01-hot-low-entry-starting.png` | Hot-low tile's Explain entry: What it is (BL-084) |
| 2 | `02-cold-high-entry-starting.png` | Cold-high tile's entry: What it is, Checks it touches with two links |
| 3 | `03-bl067-corner-loaded.png` | BL-067 corner with rating 0.72 N·m and inner hub interference minimum 8 µm |
| 4 | `04-bl067-corner-gearbox-problem.png` | gearbox Problem naming the operating pull-out with its allowance |
| 5 | `05-bl067-corner-shaft-joints-problem.png` | shaft joints Problem's slip sentence |
| 6, 7 | `06-...torque-card.png`, `07-...scrolled.png` | the torque card in the corner (gearbox term quotes nothing) |
| 8 | `08-bl067-corner-cold-high-entry.png` | Cold-high entry in the corner (both checks it names fail) |
| 9, 10 | `09-...torque-band-note.png`, `10-...watch-out.png` | torque-band note, sentence 3 and watch-out |
| 11 | `11-bl067-corner-shaft-joint-note.png` | shaft-joint note, sentence 2 |
| 12 | `12-bl067-corner-required-torque-equation.png` | the joint torque's equation, `S_fit max(T_cold,high, T_op (1 + v))` |
| 13, 14 | `13-...slip-note-watch-out.png`, `14-...sources.png` | slip note watch-out and its source lines |
| 15 | `15-equal-temps-0C-gearbox-problem.png` | operating = minimum = 0 °C: sentence 1 names the operating pull-out, sentence 2 cold-high |
| 16, 17 | `16-cap-thread-past-od-42.0-42.2.png`, `17-...42.0-42.0.png` | the red Exploration only line |
| 18 | `18-cap-thin-skirt-0.05mm-amber.png` | the amber skirt caution for a 0.05 mm skirt |
| 19 | `19-hot-low-hair-shortfall-tile.png` | the Hot-low tile at Required = hot-low + 1e-7 (BL-083) |
| 20 | `20-results-table-slip-swing-entry.png` | the slip swing row's entry (ripple help) |
| 21 | `21-results-table-gearbox-check-row.png` | the gearbox check row and its label |
| 22 | `22-results-table-shaft-joint-check-hover.png` | the shaft joints check row's hover (MF-1) |
| 23 | `23-results-table-gearbox-check-hover.png` | the gearbox check row's hover |
| 24 | `24-outside-diameter-entry-cap-thread-follows.png` | the Outside diameter entry: the cap and its thread follow the cup |

## Every string

Verdicts:

- **pass:** true, plain and consistent.
- **should:** true, with a wording or consistency point.
- **MUST:** wrong, misleading or unreadable for an engineer using the calculator.

| # | Where | String (as drawn) | Verdict |
|---|-------|-------------------|---------|
| S1 | engine label, `gearbox.cold_high_check` (results table) | "Torque at the gearbox input against its rating" (was "Cold-high torque at the gearbox input against its rating") | pass |
| S2 | engine help, `gearbox.cold_high_check` (row hover) | "Cold-high, or the operating pull-out with allowance if larger, at the gearbox output divided by ratio and efficiency, against the gearbox input rating. Fatigue from repeated slips is not checked." | pass |
| S3 | engine help, `shaft_joint.safety_factor` | "Each fit alone, and the key alone, must hold this times cold-high, or the operating pull-out with allowance if larger. While the coupling slips, the torque can reverse many times." | should (SF-5: scope of "times") |
| S4 | engine help, `shaft_joint.required_Nm` | "Cold-high, or the operating pull-out with the production allowance if larger, times the safety factor. Each fit alone must hold it; the key alone is checked at it." | pass |
| S5 | engine help, `shaft_joint.check` (row hover) | "Each rotor's shrink fit must hold the required torque when loosest, stay elastic when tightest and be fitted within the allowed temperature; the key alone is a backup." | **MUST (MF-1)** |
| S6 | equation record, `shaft_joint.required_Nm` | `T_fit,req = S_fit max(T_cold,high, T_op (1 + v))` | pass (term label of T_op: SF-4) |
| S7 | note source line, `CHECKED_PULLOUT_SOURCE` (torque-band, slip and shaft-joint notes) | "Assumption of 2026-11-10, under the owner's ruling that open questions take the conservative answer: the gearbox and shaft joint checks compare the larger of cold-high and the operating pull-out with the allowance" | pass (SF-6: the older line above it) |
| S8 | torque-band note, sentence 3 | "Cold-high, here {…}, is the pull-out at the minimum temperature plus that allowance: the most torque expected before it slips, unless running below the minimum." | pass (nit N-2) |
| S9 | torque-band note, watch-out 2 | "The gearbox check compares cold-high at the gearbox input, {…} here, or the operating pull-out with allowance if larger, with its rating; nothing checks the gearbox's fatigue from repeated slips." | pass |
| S10 | slip note, watch-out 1 | "The gearbox output carries these swings; cold they can reach cold-high, {…} at the gearbox input here; the gearbox check compares that, or the operating pull-out with allowance if larger." | pass |
| S11 | shaft-joint note, sentence 2 | "Loosest, the inner hub holds {…} here, the cup {…}; each needs {…}, cold-high or a larger operating pull-out with allowance, times the safety factor." | pass (nit N-2) |
| S12 | gearbox Problem, sentence 1 | "{Cold-high \| The operating pull-out with its allowance} at the gearbox input is {torque}, above its {rating} input rating." | should (SF-1) |
| S13 | shaft joints Problem, slip sentence | "At its loosest the {joint} holds {…}, less than the {…} it must hold: {cold-high \| the operating pull-out with its allowance}, {…}, times the safety factor of {…}." | should (SF-1) |
| S14 | torque card, gearbox term | "Gearbox input torque: cold-high ±x N·m" no longer drawn where the operating pull-out is compared (removal) | pass |
| S15 | HELP, `coupling.gearbox_input_rating_Nm` | "The gearbox's input torque rating. The gearbox check compares the cold-high torque at the gearbox input with it, or the operating pull-out with allowance if larger." | should (SF-5: clause placement) |
| S16 | HELP, `model.gearbox_input_ripple_Nm` | "The pull-out at the gearbox output at operating temperature, without the production allowance, divided by ratio and efficiency. The gearbox check compares cold-high, or this plus allowance if larger." ("Slip swing: " removed; the row's label says it) | should (SF-5: "at the gearbox input" dropped) |
| S17 | HELP, `metal.torque_hot_low_Nm` (Hot-low tile, row, hover) | "The pull-out at the operating magnet temperature less the production variation allowance. It must reach the required torque; a hotter operating temperature or a larger allowance lowers it." | pass (nit N-3) |
| S18 | HELP, `metal.torque_cold_high_Nm` (Cold-high tile, row, hover) | "The pull-out at the minimum magnet temperature plus the production variation allowance. The shaft joint and gearbox checks read it; a colder minimum or a larger allowance raises it." | should (SF-3) |
| S19 | Cold-high entry, Checks it touches | "Shaft joints" and "Gearbox input torque", each a link with its badge (new) | pass |
| S20 | Size tab, error colour | "Exploration only: the cap thread breaks through the cap." | pass |
| S21 | Size tab, amber | "Cap wall past its thread (skirt): {skirt}; guide minimum is 0.5 mm." now only for 0 < skirt < 0.5 mm (words unchanged) | pass |
| S22 | Hot-low tile and Size tab gap | "short 1.000e-7 N·m (-0.0 %)" in place of "short 0.00 N·m (-0.0 %)" | pass (nit N-1) |
| S23 | removed | "Cold-high torque at the gearbox input against its rating"; "Slip swing: "; "short 0.00 N·m"; the shaft-joint helps' "cold-high times the safety factor" | pass |
| S24 | unchanged, made false by BL-067 | torque-band note, change step for Minimum magnet temperature: "a colder minimum raises cold-high and with it the shaft-joint and gearbox loads; enter the coldest the magnets really get." | **MUST (MF-2)** |
| S25 | unchanged, made incomplete by BL-067 | gearbox guide Why ("… at the gearbox input it must stay within the rating"), shaft joints guide Why ("each fit must carry cold-high with margin …"), torque-band note sentence 4 ("must carry cold-high with margin …") | should (SF-2) |

Each physics claim held on the engine where the table says pass:

- S17's and S18's directions hold one slider step each way, on the example and Y30 designs (the branch's test does
  this too).
- S20: a nominal skirt at zero already puts the tapped root past the outside diameter, so at or below zero is a
  break-through.
- S2, S4 and S6 match `checked_pullout_Nm` and `joint_specs`.
- S12 and S13 quote 0.7699 N·m at the gearbox input and 3.657 N·m at the output in the corner, as the probe gives.

No new string says that the coupling protects the gearbox, uses a spreadsheet cell name, names a ring by colour, or
is fitted to one user's hardware.

## Findings

### Must fix

**MF-1. The shaft joints check's help calls the joint torque "the required torque" (S5). Blocks the public site.**

- What the page shows: the results-table hover of "Shaft joints: keyed shrink fits" says each rotor's shrink fit
  "must hold the required torque when loosest" (screenshot 22).
- Why it misleads: on this page, Required torque is the hot-low requirement, the Size tab's first slider (1.35 N·m
  on the starting design, drawn in the same screenshot). The torque the fits must hold is the result "Torque each
  shaft joint must hold": the safety factor times cold-high, or the operating pull-out with its allowance if larger
  (4.85 N·m there). An engineer reads the hover as the fit holding the coupling's requirement, short by a factor of
  about 3.6.
- Fix: name the result as its label does: "Each rotor's shrink fit must hold the torque each shaft joint must hold
  when loosest, stay elastic when tightest and be fitted within the allowed temperature; the key alone is a backup."
  Or bring back the definition: "… must hold the safety factor times cold-high, or the operating pull-out with
  allowance if larger, when loosest, …".
- Add a guard so that no shaft-joint text says "required torque" unqualified.

**MF-2. The torque-band note's Minimum magnet temperature step is false in the BL-067 corner (S24). Does not block.**

- What the step says: "a colder minimum raises cold-high and with it the shaft-joint and gearbox loads".
- Why it is false: since BL-067, the checks compare the larger of cold-high and the operating pull-out with its
  allowance. At operating −40 °C, minimum 20 °C and variation 0.1, one step colder (19.5 °C) raises cold-high, but
  both loads stay at the operating pull-out with its allowance (3.657 N·m at the output). BL-067 made the step
  false; before it, the loads were cold-high.
- Why it does not block: the corner is an operating temperature below the stated minimum, and the Problems there
  name the right torque.
- Fix: "a colder minimum raises cold-high and, where cold-high is the larger, the shaft-joint and gearbox loads; …".
  Or "… raises cold-high, which the shaft-joint and gearbox checks read; …".
- Hand it to fable's physics re-review (BL-130) with the notes' other BL-067 changes. The step at line 398 of
  `notes.rs` ("loads rise with cold-high") is still true there, since longer magnets raise both torques; it is only
  inexact.

### Should fix

**SF-1. At equal temperatures the name the guide gives the compared torque follows float noise (S12, S13, S14).**

- The cause: `operating_pullout_compared` is `checked_pullout > cold-high`. With the operating temperature equal to
  the minimum, the two are the same physical torque, but the engine forms them by different paths. On 180 of 605
  equal-temperature designs, the operating pull-out with its allowance comes out above cold-high in the last bits.
- What the page shows: at operating = minimum = 0 °C (variation 0.15, a plain setting), the gearbox Problem's
  sentence 1 reads "The operating pull-out with its allowance at the gearbox input is 0.7344 N·m". Sentence 2, which
  uses `min_temp_C <= op_temp_C`, then says "cold-high is 1.353 times hot-low here" of the same torque (screenshot
  15). The torque card also loses its "Gearbox input torque: cold-high" quantity there.
- Why it matters: nothing in it is false, but one Problem names one torque two ways. The name also suggests the
  magnets run below the minimum when they do not.
- Fix: name the operating pull-out only where it is larger beyond float noise. That means `checked > cold_high × (1 +
  1e-12)`, as `RATING_NOISE` does, or `op_temp_C < min_temp_C` with the engine's comparison unchanged.
- Test at operating = minimum = 0 °C with a failing rating.

**SF-2. Three unchanged texts still tie the loads to cold-high alone (S25).**

- The texts:
  - the gearbox guide's Why (drawn as "What it is", screenshot 4): "Cold-high torque can reach the gearbox output …;
    at the gearbox input it must stay within the rating";
  - the shaft joints guide's Why: "each fit must carry cold-high with margin";
  - the torque-band note's sentence 4: "must carry cold-high with margin …".
- What happens in the corner: each sits right above a Problem that names the operating pull-out with its allowance.
  None of them is false, but together they read as if cold-high were the torque checked.
- Fix: one clause each, for example "Cold-high, or the operating pull-out with allowance if larger, can reach …".
  Changing these texts needs its own G-3 round.

**SF-3. The Cold-high help and entry do not say that the checks it names can compare another torque (S18, S19).**

- What the page shows: in the corner, the Cold-high entry lists Shaft joints and Gearbox input torque, both red
  (screenshot 8), while neither fails on cold-high.
- Fix: add "or the operating pull-out with allowance, if larger" after "read it". Keep the help short.

**SF-4. One name for the other torque.**

- The strings call it five ways:
  - "the operating pull-out with allowance if larger" (S2, S3, S9, S10, S15);
  - "the operating pull-out with the production allowance if larger" (S4);
  - "the operating pull-out with its allowance" (S12, S13);
  - "a larger operating pull-out with allowance" (S11);
  - "this plus allowance" (S16).
- Its equation term `T_op` is labelled "Predicted candidate torque at operating temperature" (screenshot 12). That is
  the engine's label for `metal.torque_op_Nm`, which has no row in LABELS. The tile calls the same number "Pull-out
  torque (nominal, at −40 °C)".
- Fix: keep one form, "the operating pull-out with its allowance", and give `metal.torque_op_Nm` a display label in
  the tile's words, for example "Pull-out torque at operating temperature".

**SF-5. Three helps could be clearer.**

- S16 says "compares cold-high", where the old text said "cold-high at the gearbox input". The comparison is at the
  input, and the ripple is an input torque.
- S15's "or the operating pull-out with allowance if larger" hangs after "with it". Clearer: "The gearbox check
  compares with it the torque at the gearbox input: cold-high, or the operating pull-out with allowance if larger."
- S3's "this times cold-high, or the operating pull-out with allowance if larger" can be read as the factor applying
  to cold-high only. Clearer: "this times the larger of cold-high and the operating pull-out with allowance".

**SF-6. The older source line now reads as the whole rule (S7).**

- In each note's "Where this comes from", `GEARBOX_CHECK_SOURCE` ("the gearbox check compares cold-high at the
  gearbox input with the gearbox input torque rating; …") sits just above the new assumption line
  (screenshot 14).
- Fix: "… compares the torque at the gearbox input with the rating …", or put the assumption line first. The
  precedent for "owner" in a source line exists ("The owner's decisions of 2026-10-17"), so the new line's wording
  is consistent.

### Nits

- **N-1 (BL-083).** The tile reads "2.284 N·m … needs 2.285 N·m … short 1.000e-7 N·m (-0.0 %)" (screenshot 19).
  - The gap is the true one, and the pair is BL-068's rounding apart, as BL-083's entry records.
  - The percent still reads "-0.0 %". It is true at one decimal and the sign marks the side, so the change can stand.
  - A percent in significant digits would match the N·m.
  - The case is reachable only through a link or a file.
- **N-2.** Two openers read clipped. S11's "Loosest," reads better as "At its loosest" (if the sentence cap allows).
  S8's "unless running below the minimum" reads better as "unless it runs colder than the minimum".
- **N-3.** S17's "a hotter operating temperature … lowers it" and S18's "a colder minimum … raises it" do not hold at
  a Br coefficient of exactly zero, the top of the calibration slider, where the torque stays. Every grade's
  coefficient is below zero, and the older temperature helps make the same assumption.

## Not checked

- Fable's physics re-review (BL-130) is still owed.
- The clamp guide's Problem was not checked: the clamp section is paused and hidden.
- The exports (CSV, JSON and XLSX) were not opened.
- The Geometry tab was not rendered: it draws no wall line for the cap, as before.

## Fix wave

Applied on `magcoupling/r4-physics` on 2026-11-28, uncommitted at the time of writing. Every help keeps the 30-word cap,
so three wordings differ from the ones proposed above.

- **MF-1 (taken).** `shaft_joint.check` help: "Each shrink fit must hold the torque each shaft joint must hold when
  loosest, stay elastic when tightest and be fitted within the allowed temperature; the key is a backup." The
  result's label in full did not fit beside "Each rotor's" and "the key alone", so those words went.
  - Guard: `tests/shaft_joint.rs`
    `no_shaft_joint_text_calls_the_torque_a_fit_must_hold_the_required_torque`. No label or help of the check's inputs
    and results, and no text of the shaft-joint note, says "required torque". The check's help names the result by
    its label.
- **MF-2 (taken).** The torque-band note's minimum step: "a colder minimum raises cold-high and, where cold-high is the
  larger, the shaft-joint and gearbox loads; enter the coldest the magnets really get." `tests/explain.rs` now checks
  that cold-high is the larger at every probe design.
- **SF-1 (taken).** `operating_pullout_compared` (`gui/guidance.rs`) takes the operating pull-out with its allowance only
  where it is above cold-high × (1 + `RATING_NOISE`). That gives the name to the gearbox Problem, the shaft joints slip
  sentence and the torque card's gearbox term. The check's numbers (`checked_pullout_Nm`) are unchanged.
  - Red test first: `at_equal_temperatures_the_texts_name_cold_high_whatever_the_float_noise`. It covers the 605-design
    sweep, where the noise still shows on some designs, and operating = minimum = 0 °C at variation 0.15, where sentence
    1 now reads "Cold-high at the gearbox input is 0.7344 N·m" beside sentence 2.
  - The BL-067 corner still names the operating pull-out with its allowance.
- **SF-3 (taken, shortened).** Cold-high help: "The pull-out at the minimum magnet temperature plus the production
  variation allowance. The shaft joint and gearbox checks read it, or the operating pull-out with allowance if larger."
  - "a colder minimum or a larger allowance raises it" was cut: with the new clause the help would run to 37 words.
  - The claim test now pins hot-low's direction words only. Cold-high's directions are still stepped, and the
    torque-band note's steps state them.
- **SF-5 (taken).**
  - Rating: "The gearbox's input torque rating. The gearbox check compares with it the torque at the gearbox input:
    cold-high, or the operating pull-out with allowance if larger." (as proposed)
  - Ripple: "… divided by ratio and efficiency. The gearbox check compares input cold-high, or this plus allowance if
    larger." "Cold-high at the gearbox input" would take the help to 33 words.
  - Safety factor: "Each fit and each key, alone, must hold this times the larger of cold-high and the operating
    pull-out with allowance. While the coupling slips, the torque can reverse many times." `input_schema.json`
    follows.
- **Left to fable's re-review (BL-130):** SF-2, SF-4, SF-6 and the nits N-1 to N-3. The cut wordings above (MF-1's
  "Each shrink fit", SF-3's dropped direction clause, SF-5's "input cold-high") are for that review too.
