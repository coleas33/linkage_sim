# Magcoupler UX overhaul, slice A: text and physics review (BL-077, plan Task 6)

Date: 2026-11-17. Reviewed: branch `magcoupling/overhaul-a` at bd47b2f (plan Tasks 1 to 5) against `main` at 6af5948.
Plan: `docs/superpowers/plans/2026-11-17-magcoupler-overhaul-a-first-arrival.md`. Design: `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md`, sections 2.3, 7.1, 8.3 and 12.3.

**Who reviewed.** The plan asks for a reviewer with a fresh context. This review was done by the agent that applied the plan's blocks for Tasks 1 to 5, on the controller's instruction (no subagents). That agent did not write the strings or the claims (the plan did), and it checked each one against the code, the tests named below and its own probe of `compute_all`. It is still less independent than the plan intended. The owner's review of the diff stands behind it.

**How.** The design's sections above were read, then the branch's diff of `magcoupling-rs/src/gui/guidance.rs`, `gui/panel.rs`, `app.rs` and `README.md`. The named tests were run on this branch. The numbers marked "probe" come from a temporary test in `app.rs` that printed them from `compute_all` and from the app's first frames; it was removed afterwards and is not in any commit.

Probe results at `app::starting_design()`:

- Hot-low 1.448325 N·m: 0.098 N·m and 7.28 % over 1.35 N·m. The design passes at a Required torque of 1.44 and 1.448 N·m and fails one check at 1.449 and 1.45 N·m.
- Cold-high 2.423151 N·m; at the gearbox input 0.510137 N·m (2.423151 / (5 x 0.95)) against the 0.6 N·m rating. Pull-out 1.703912 N·m.
- Counted checks: 0 failing and 0 cautions of 9 (clamp section paused); no material warning; no drawing note with a level.
- Hub wall past the keyway 2.2499999999999991 mm. Cap skirt (42.8 - 41.7) / 2 = 0.55 mm.
- Given facts read back: gearbox input rating 0.6 N·m, gearbox ratio 5.0, maximum diameter 43.0 mm, cap outside diameter 42.8 mm, `B842SH` on both rings, no axial length override.
- The six on-screen labels of the changed inputs equal the README table's: Magnets per ring, Ring radius, Required torque, Magnet face gap, Minimum outer return ring wall, Cap thread nominal diameter.
- First view of the app at 1280 x 800 and at 1280 x 6000: the only text in the caution or the error colour is "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." Nothing is drawn in the error colour.
- Share link at the starting design: 2,692 characters.
- The other answer of register row A-17 (ring radius 9.26 mm, cap thread 42.2 mm): wall 2.51 mm, 0 failing and 0 cautions of 9, no material warning, no drawing note with a level, hot-low 1.462720 N·m, 0.515207 N·m at the gearbox input, skirt 0.30 mm; both values are on their slider steps. With the thread left at 41.7 mm the drawing adds a caution ("Design check: Cap thread nominal diameter 41.70 mm is below Steel cup body OD 42.18 mm"), so the 42.2 mm thread is needed. At 9.25 mm the wall is 2.4999999999999991 mm and the line still shows.

## The five strings

| # | String | Verdict | Evidence | Replacement |
|---|---|---|---|---|
| S1 | "Starting design" (`guidance::STARTING_DESIGN`) | PASS | Plain and true: it is the design the page starts on (`app::tests::the_app_opens_the_starting_design`). `status_line` uses it only while the inputs equal the baseline and are not the spreadsheet's values (`guidance::tests::the_status_line_names_the_design_against_the_baseline`). The text rules of 8.3 hold (`panel::tests::the_first_arrival_s_new_texts_keep_the_text_rules`). | |
| S2 | "A design that passes every check. It keeps the example design's gearbox rating, space and magnets, and asks for less torque." (`guidance::STARTING_DESIGN_HOVER`) | FAIL (one word) | "passes every check", "gearbox rating", "space" and "asks for less torque" are true (claims C1 and C2). "keeps ... magnets" is not exact: the magnet part is kept (`B842SH` on both rings at the stock length, which is what the six-inputs test asserts), but the number of magnets per ring changes from 10 to 8, and "Magnets per ring" is the first of the six rows that carry the changed dot after Open the example design. The input rows are named "Inner magnet part" and "Outer magnet part", so "magnet part" is the tool's own word. | "A design that passes every check. It keeps the example design's gearbox rating, space and magnet part, and asks for less torque." Also where it is quoted: `magcoupling-rs/README.md` and the design's section 7.1. |
| S3 | "The values from the original spreadsheet." (`guidance::EXAMPLE_DESIGN_HOVER`) | PASS | True: the example design is `DesignInputs::default`, which this slice does not touch (no file under `magcoupling-rs/src/engine` or `magcoupling-rs/tests` is in the diff). Dropping "starting" removes the clash with the new name. It reads correctly in both places it is drawn: the status line at the example design and the button's hover. | |
| S4 | "Open the example design" (`panel::OPEN_EXAMPLE`) | PASS | Plain and true: one undo step to the spreadsheet's values, the baseline stays (`panel::tests::open_the_example_design_is_one_undo_step_and_keeps_the_baseline`, `app::tests::the_example_design_opens_as_one_undo_step_from_the_starting_design`). The two names differ by their first word, and the header's button and the dashboard's status line each have a hover that says what the design is. | |
| S5 | "Example design opened" (`panel::EXAMPLE_OPENED`) | PASS | Plain and true at the click. Like every header status it stays until the next one: after an Undo the header still reads it while the status line reads "Starting design". "Default design restored" behaves the same way today; note 5. | |

No string breaks section 8.3: each passes the plain-words guard and the gearbox-protection guard, none has "sealed", "containment", "leak", "clamp" or a ring named by a colour, and none carries a digit (the test named under S1; it reads the constants, so it also holds the replacement of S2).

## The five claims

| # | Claim | Verdict | Evidence | Replacement |
|---|---|---|---|---|
| C1 | The starting design passes every counted check with no caution and no material warning. | PASS | `app::tests::the_starting_design_passes_every_check_with_no_caution_and_no_warning` passes: 0 failing and 0 cautions of 9 with the clamp section paused and of 10 with it shown, every counted guide Good, no warning line, no drawing note with a level. The probe agrees. | |
| C2 | It keeps the example design's gearbox rating, space claim and magnets, and asks for less torque. | PASS | `app::tests::the_starting_design_changes_six_inputs_on_their_slider_steps_and_no_assumption` passes: exactly six inputs differ, each on its slider's step; the gearbox rating, the three space limits, both magnet parts and the axial length override are equal; Required torque is lower; no assumption is changed at either design. "Magnets" here means the magnet part; the on-screen wording is S2. | |
| C3 | The numbers of the README paragraph and the doc comment: hot-low 1.448 N·m, 7.3 % over 1.35 N·m, passing up to 1.44 N·m, 0.510 N·m against the 0.60 N·m rating, the cap skirt of 0.55 mm. | PASS | The test of C1 asserts 1.448, 7.3 %, 0.510, the 0.6 rating, pull-out 1.704, cold-high 2.423 and the 1.44 and 1.45 cases. The probe gives 1.448325, 7.28 %, 0.510137 and 0.55. The README table's six rows match `STARTING_CHANGES` and the on-screen labels. "5:1", "43 mm" and "B842SH" are true by the probe. | |
| C4 | The split of 1.35 N·m into 0.35 N·m of bearing drag and 1.00 N·m for acceleration is presented as the owner's figures and not as something the tool computes. | PASS | The README and the doc comment both say "(the owner's figures; the tool takes the requirement as one number)". No on-screen string carries the split (S1 to S5 have no digit). The owner gave 0.35 N·m (O-8) and confirmed the total of 1.35 N·m (O-9), so 1.00 N·m follows from the two. Nothing in the repository measures the drag (design 12.3). | |
| C5 | At this design the Size tab draws "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm." beside a status line and a hover that say the design passes every check. Is the hover still honest, and is the line's presence stated wherever the design is described? | FAIL (one place) | Honest: yes. In the tool a check is one of the counted checks of the dashboard, the status line counts them ("passes all 9 checks"), and the wall guidance is not one of them. The line is not hidden: it is drawn in the caution colour on the opening tab, it is the only such text on the first view (probe), and `app::tests::the_starting_design_draws_one_advisory_on_the_opening_tab` pins it. Stated: in the README paragraph, the design's section 7.1, register row A-17 and the T3 commit message. Not stated: the doc comment of `app::starting_design`, which says the design is "sized inside them to pass every check" and stops there. | Add to the doc comment of `app::starting_design`, as its last paragraph: "One line of the opening tab is outside the counted checks: the hub wall past the keyway is 2.25 mm, under the Size tab's guidance of about 2.5 mm, and the tab says so in the caution colour (register row A-17 in `docs/analyses/2026-11-10-conservative-assumptions.md`; a test pins the line)." |

## Notes for the owner (no verdict depends on them)

1. **The first view shows one amber line under a status that says every check passes.** That is true to the tool's own terms (C5) and nothing is hidden, but it is the first thing a careful visitor may ask about. Row A-17 holds the choice: keep the approved values, or take ring radius 9.26 mm with a 42.2 mm cap thread, which clears the line and thins the cap skirt from 0.55 to 0.30 mm.
2. **The other answer's figures are pinned by no test.** 9.26 mm, 42.2 mm, 0.30 mm, 1.463 N·m and 0.515 N·m are quoted in the README, the design's section 7.1, row A-17 and a test's comment. They are true on this branch (probe). A physics fix that lands before the owner rules can move them without turning a test red.
3. **The plan's Task 3 Step 6 says more than the tests hold.** It says every number in the texts is asserted by two tests except three. "5:1", "43 mm" and "B842SH" are held only through "exactly six inputs differ" plus the parity tests that lock the spreadsheet's values, and the figures of note 2 are held by nothing. All are true by the probe, so no text was changed for this.
4. **The hover describes the app's design from inside the panel's code.** `STARTING_DESIGN_HOVER` is drawn for any baseline that is not the spreadsheet's design, and `MagcouplingPanel::with_baseline` is public. Only the standalone app calls it today, so the hover is true wherever it is shown. A second host with another baseline would show a hover that describes this design.
5. **Three names on today's screen.** After Reset all the header reads "Default design restored" while the status line reads "Starting design". The design keeps those words on purpose (7.1; slice B rewords the button). The header status also stays after an Undo, as it does today.
6. **Not verified here:** whether a visitor tells "Starting design" from "Example design" without help (design 12.3); the hover as a browser draws it; the address write in a browser (the smoke run of plan Task 8); Safari and Firefox.

CHANGES NEEDED: 2 (S2: one word of the hover; C5: one paragraph of a doc comment).

## Re-review

2026-11-17, the same reviewer, of the two changed texts only (commit 022efd9).

| # | Text as changed | Verdict | Evidence |
|---|---|---|---|
| S2 | "A design that passes every check. It keeps the example design's gearbox rating, space and magnet part, and asks for less torque." (`guidance::STARTING_DESIGN_HOVER`; quoted in `magcoupling-rs/README.md` and in the design's section 7.1, which also says what the review changed) | PASS | Each of the three things it says are kept is asserted equal by `app::tests::the_starting_design_changes_six_inputs_on_their_slider_steps_and_no_assumption` (the rating, the three space limits, both parts and the length override). It no longer speaks for the number of magnets. `panel::tests::the_first_arrival_s_new_texts_keep_the_text_rules`, `dashboard::tests::the_starting_design_s_status_line_says_on_hover_what_it_is` and the glyph guard read the constant and pass. |
| C5 | The doc comment of `app::starting_design` ends: "One line of the opening tab is outside the counted checks: the hub wall past the keyway is 2.25 mm, under the Size tab's guidance of about 2.5 mm, and the tab says so in the caution colour (register row A-17 in `docs/analyses/2026-11-10-conservative-assumptions.md`; a test pins the line)." | PASS | 2.25 mm and the line's words are asserted by `app::tests::the_starting_design_draws_one_advisory_on_the_opening_tab`; the colour is `warn_fg_color` in `gui/panel.rs`, `view_ui`. The line is now stated in every place that describes the design. |

Checks after the change: `cargo fmt --check` clean, `cargo test --features app` green (1,029 lib tests), clippy clean with and without the app feature and for the wasm32 web binary. The plan's own text was not edited, as its Task 6 says, so the plan still quotes the first wording of the hover.

APPROVED (one round; the two changes above are in their own commit, before this record's).

## Addendum: a second round (2026-11-17)

Found by the same reviewer while reading the README for the docs sweep (plan Task 7, Step 3), after the verdict above.

| # | Claim | Verdict | Evidence | Replacement |
|---|---|---|---|---|
| C6 | The texts that describe the starting design say what its "passes every check" rests on. | FAIL (three places) | Two of the nine checks read the stored 3D fields (`dashboard::STORED_3D_INPUTS`): the temperature check, and the inner hub's fitting temperature inside the shaft joints check. The fields were computed for the example design (10 magnets, ring radius 10.15 mm, face gap 1.40 mm) and stay fixed for every design; the starting design has the same 14 values (probe). The tool says so itself: the dashboard draws "Fixed 3D fields from the example design" under those checks, and its hover ends "for another design, enter its own 3D fields to check the limits". Until this slice the page opened on the design the fields belong to. Now it opens on another one, and the README paragraph, the doc comment of `app::starting_design` and the design's section 7.1 described that design without saying so. On screen nothing is hidden, so the five strings keep their verdicts; at 1280 x 800 the line is below the dashboard's first screen (it was not drawn in the probe's 800-point frame and was drawn in its 6000-point frame). | One sentence in the README's starting-design paragraph, one in the doc comment of `app::starting_design`, one bullet under "What the owner should know about the starting design" in the design's section 7.1, and register row A-18 for the owner's ruling. No on-screen string, check or value changes. |

Probe for C6, at `app::starting_design()` with the stored fields scaled (the tested points, not thresholds):

| Stored fields scaled | x 1.1 | x 1.25 | x 1.5 | x 1.75 | x 2 to x 4 |
|---|---|---|---|---|---|
| The four reverse fields | all 9 pass, no caution | Temperature limits: caution | the same | Temperature limits: caution; Shaft joints: fails | the same |
| The ten slip-loss fields | all 9 pass, no caution | the same | the same | the same | the same |

So the verdicts do not sit on an edge: they hold with reverse fields a tenth larger than the example design's. Whether the starting geometry's own reverse fields are larger or smaller than the example design's was not computed; the Rust tool has no 3D field solver.

**Re-review of round 2** (commit 20120ac). The three texts and row A-18 say what the table above says and no more: which two checks, whose fields, that every design shares them, where the dashboard says it, and the probe's four results. The row gives the answer taken for now (nothing changes) and the two other answers (compute and enter the fields for this geometry, or say it on the first view). Checks after the change: `cargo fmt --check` clean, `cargo test --features app` green (1,029 lib tests), clippy clean with and without the app feature. PASS.

Note 7 for the owner, added with this round: **the first view's "passes every check" is the tool's verdict with the example design's 3D fields.** Row A-18 holds the choice.

APPROVED after two rounds (three changes in all: S2, C5 and C6; commits 022efd9 and 20120ac).

## Addendum: the independent review (2026-11-17)

A reviewer with a fresh context read the branch at e258eb8, after the verdict above. It is the independent read the first paragraph of this record says the two rounds were not. It found the defects below; each was fixed with a failing test first, in its own commit.

| # | Finding | Evidence | Fix |
|---|---|---|---|
| I-1 | A held Ctrl+Z or Ctrl+Y wrote the address at every key repeat. A refused write was counted as written, so the address stayed on an older design with nothing to show it. | The panel's shortcuts act on key repeat (`gui/panel.rs`, `shortcuts`), and each undo or redo step is at once the history's settled design, so `MagcouplingApp::ui` wrote one address per repeat. Safari refuses more than about 100 `replaceState` calls in 10 seconds. `replace_address` dropped the call's result and `address_design` had already moved. Red before the fix: five undo steps held through 60 frames gave five writes, and a refused write left `address_design` on the edited design. The design's section 7.2 said writing only on settled edits "keeps far under" the limit, which a held undo disproved. | `MagcouplingApp::ui` writes only while no key is held, so a held run gives one write at the key's release. `replace_address` returns whether the page took the write, and `address_design` advances only then; a refused write is asked again on the next frame. Tests: `app::tests::a_held_undo_or_redo_writes_the_address_once_at_its_release`, `app::tests::a_refused_address_write_is_tried_again_on_the_next_frame`. The design's 7.2, the README and `docs/ai` say the new rule. |
| I-2 | Four texts called the spreadsheet's design "the default design". Since this slice, "default" on screen is the design the session started on (the changed dot, a row's reset, the tooltip's Default line, Reset all), so at the starting design each of these texts pointed at the wrong design. | `gui/guidance.rs`, the temperature Problem's two source clauses: "3D results for the default NdFeB design" (cold limit) and "slip fields from the default design" (hot magnet limit); the stored fields are the example design's, as the dashboard's own line says ("Fixed 3D fields from the example design"). `gui/corrections.rs`, `MARKER_SCOPE`, the last line of a marked value's hover: "Was and now are values at the default design"; the values are the workbook's and the corrected one at the spreadsheet's inputs. `engine/deviations.rs`, the `corrected_formula` of E20 ("so the default design does not move") and of E21 ("on the default design"): registry text that no screen or export shows, read in the source only. | Each says "example design" ("example NdFeB design" in the cold-limit clause, which keeps its qualifier). The comments beside them, the README's three sentences and the `docs/ai` lines that quote them follow. Tests, red before the change: the pinned Problem strings in `guidance::tests` (`the_temperature_problem_names_each_failing_term`, `a_quoted_magnet_limit_says_what_it_is`, `a_ring_limit_that_is_not_a_number_is_not_quoted_as_one`), a new pin of `MARKER_SCOPE` in `corrections::tests::the_headline_pull_out_carries_e3_with_its_workbook_value`, and the new `deviations::tests::no_entry_calls_the_spreadsheet_s_design_the_default_design`. |

Not changed under I-2, on purpose: "Default design restored" after Reset all (it restores the baseline, which is what "default" means on screen; note 5 above, slice B rewords it); the cold-limit clause's "the default reverse-field inputs" (the inputs' own defaults, the same at both designs); the spreadsheet export's "Changed from default" column (against the workbook by decision C-22); and the code comments, test names and older plans and records where "default design" means `DesignInputs::default()`.

Added with I-1, as coverage the review asked for (it passed before any change): `app::tests::a_link_cut_at_two_thousand_characters_is_refused_and_the_header_says_why`. A link is about 2,700 characters. Cut at 2,000, 1,999, 1,998 or 1,997 characters (one cut per remainder of a base64 group) it is refused, the design and the address stay, and the header shows the reader's reason: "share link refused: not URL-safe base64 (...)" or "share link refused: not a compressed design (...)". Neither reason says the link looks cut short; the words were not changed here.

Limits of the I-1 fix, for the owner:

- The retry has no back-off: while the browser keeps refusing, each frame encodes the link and asks again. A refusal by Safari's limit ends within its 10 seconds. A browser that refused every write would be asked on every frame.
- A click on the header's Undo or Redo button still writes once per click, since each click is its own settled step. More than about 100 clicks in 10 seconds would be needed to reach Safari's limit.
- The write, the limit and the retry were not run in a browser. The tests use a stand-in page that records each write and can refuse one. The smoke run exercises the call in Chromium only and does not reach a limit.
- Safari's figures (100 calls, 10 seconds) are the reviewer's; they were not checked against a Safari build.
