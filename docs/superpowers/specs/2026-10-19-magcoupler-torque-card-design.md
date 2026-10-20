# Magcoupler guidance, slice 3: the torque card — design

- Date: 2026-10-19. Branch `magcoupling/slice3-torque-card` from `main` at 69fa505 (BL-064 merged).
- Backlog: BL-073 (new, `dimension: gui`, `risk: physics`; section 9). Research (gitignored):
  `.superpowers/sdd/2026-10-19-magcoupler-slice3/`: `owner-decisions.md` (binding), `context-brief.md`,
  `context-critique.md`, `ctx-engine.md`, `ctx-gui.md`, `ctx-physics.md`, `ctx-history.md`.
- `M` = `magcoupling-rs`; code paths are under `M/src/` unless they say otherwise, and every `path:line` is at 69fa505.
  spec1 = `docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md`; spec2 =
  `docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md`.
- Numbers marked **(E)** come from `compute_all` at 69fa505. Numbers marked **(C)** come from this spec's prototype of the
  card, a scratch crate outside the repo: `<scratchpad>/slice3-spec/` (`src/main.rs`; `src/bin/bl075.rs` for BL-075's
  evidence), `magcoupling-rs` by path with feature `gui`, built into the repo's gitignored `M/target/slice3-probe` (no
  LNK1104). `<scratchpad>` is
  `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/c368a0e9-0dad-4f12-ad62-643ce3a7f894/scratchpad`.
  It reuses the probe crate's code (`slice3-probe/src/bin/sizetab.rs`, `engine-ctx/src/bin/critic.rs`, `critic2.rs`).
  Every number below was re-run there; none is copied from the research. Numbers marked **(H)** are hand calculations
  from a formula quoted beside them.
- **Revision 1** (after spec reviews 1, `.superpowers/sdd/2026-10-19-magcoupler-slice3/spec-review-physics-1.md` and
  `spec-review-code-1.md`, each with a "Writer's response"): the prototype applies the review's rules (the new
  `CAPTION`, `JUMP` and gearbox phrase, the decimals cap, `shows`, the two one-line forms) and was run again. Its output
  `out-run3.txt` is the source of every (C) number and drafted line below, its section 3b holds the review's checks
  (the cup cavity, the steel slice, the radius step, the temperature edge, the slider ends). `src/bin/layout.rs`
  (`layout-run3.txt`) measures the room at the card in the shipped panel, and `src/bin/forty.rs` (`forty-run3.txt`)
  finds a 40-magnet design that fits. Section 7.1's timing is the first run's (`out.txt`): the revision changes none of
  the work it times, and the rerun agrees (a: 0.991 / 1.019 ms against 1.011 / 1.036). Apart from the changed words, no
  number, flip, margin or side word moved between the runs.
- **Revision 2** (after spec reviews 2, `spec-review-physics-2.md` and `spec-review-code-2.md` in the same folder, each
  with a "Writer's response"): the prototype applies the round-2 rules (the one step slack `STEP_SLACK_MM`, the form
  `ShownNoHotLow`, a row's shared N·m decimals of T-14) and was run again. Its output `out-run4.txt` is now the source of
  every (C) number and drafted line; its sections 1 to 3b equal `out-run3.txt`'s line for line except the seven lines
  named next, so the citations of `out-run3.txt` below hold in it. Section 3c holds the round-2 checks and section 6 the
  rule-3 sweep in the shipped panel (`src/round2.rs`, included by `main.rs`). `layout.rs` adds the test harness's style
  and shorter windows (`layout-run4.txt`), and `src/bin/layoutdbg.rs` finds the panel's minimum height. Against run 3
  only these moved: the N·m text of the Outside diameter row at (e), (e2), (e2)+Auto length, (e2)+Auto count and (e)
  with the lowest Minimum OD (T-14), and variation 1.0's line and its section 3b check (`ShownNoHotLow`). The slack
  changes no line at any design of section 2 (C).
- **Revision 3** (after spec reviews 3, `spec-review-physics-3.md` and `spec-review-code-3.md` in the same folder, each
  with a "Writer's response", and the controller's rulings in `owner-decisions.md`, "Spec approval and the night run"
  and "Spec review round 3 rulings"): owner questions O-1 and O-2 are answered (section 1.3), so the spec states one
  rule for each, the widened jump rule (3.3 step 7) and no temperature margin (T-13). The prototype applies the round-3
  rules (the widened jump rule with one `JUMP` draft; test 0 on the readouts' own rule, with the reworded
  `SHOWN_NO_HOT_LOW`; the design "zero torque") and was run again. Its output `out-run5.txt` is now the source of every
  (C) number and drafted line; its sections 1 to 3c equal `out-run4.txt`'s except the lines these rules move (the
  `JUMP` text in the mixed pair's and the short cavity's length rows; the short cavity's `+1 mm` side, which now reads
  words; variation 1.0's line; the new design "zero torque"), so the citations of `out-run3.txt` and `out-run4.txt`
  below hold in it, and its sections 5 and 6 (the heights and the rule-3 sweep) are unchanged. Section 3d holds the
  round-3 checks (`src/round3.rs`: every design of section 2 under both jump rules, the shown-design routing, the
  smallest number side, the boss plateau, the odd count under Auto) and section 7 the round-3 layout checks in the
  shipped panel (the centre region over design changes, in one session and on a fresh panel; test 25's windows).
- **Revision 4** (after spec reviews 4, `spec-review-physics-4.md` and `spec-review-code-4.md` in the same folder, each
  with a "Writer's response"; physics review 4 found no must-fix item): code review 4's two must-fix items are applied
  with the reviewer's recommended option, as `owner-decisions.md` ("Spec review round 3 rulings", its last bullet)
  rules for later rounds: the card's header row wraps (6.1), and a count side whose calibration switch takes its
  torque to or below zero shows its number (Success 2, 3.3 step 8). Every should-fix item of both reviews is applied
  but the refresh of `plan-contract.md`, which is the controller's. The prototype adds the design "(c),
  `f_cal_original` -0.95" and the round-4 checks (`src/round4.rs`: section 3e, that design, step 6 through `predicts`,
  the cup cavity at 10.7 mm, the cap thread of R-13; section 8, the header row in three forms) and was run again. Its
  output `out-run6.txt` is now the source of every (C) number and drafted line; its sections 1 to 3d equal
  `out-run5.txt`'s except the new design's block and the count of designs section 3d scans (46 -> 47), so the
  citations of runs 3 to 5 below hold in it, and its sections 5 to 7 are unchanged.
- **Revision 5** (after spec reviews 5, `spec-review-physics-5.md` and `spec-review-code-5.md`, both PASS): the
  controller applied their exact should-fix texts: the grid's three part pairs (3.3 step 8), the calibration edge at
  zero (3.3 step 8, R-8, test 3), the key term's quoted pressure on a length step (3.6), the import
  `end_effect_in_range` and the two test helpers that gain `clamps_paused` (6.3), the Minimum OD above the diameter
  shown (2.1, 2.3 Edges, test 3, 10.1 step 2), `signed_percent` at -0.0 (section 5) and the heading's click (6.1).
  No number of the prototype's runs changed; the new values are the reviewers' reproduced ones.
- Torques are hot-low (`metal.torque_hot_low_Nm`) at the gearbox output unless a text says otherwise. "Step" means one
  of the four fixed steps of decision 3A. "Side" means one direction of a step (the `-` side or the `+` side).

## 1. Purpose

The owner went from 10 to 8 magnets, lowered the torque target and expected the coupling to shrink. It did not, and
nothing on screen said why (spec1:8-9). The audience is "mechanically literate engineers who are not magnetics
specialists" whose notes were "too dense" and did "not say what to change" (spec1:25-27). Decision G-1 put "a torque
sensitivity card" third in the order (spec1:34).

Slice 3 adds one card to the Size and torque tab, between the torque readouts and the drawing. For each of the four
Size controls it shows what one fixed step each way does to the hot-low torque, in N·m and %, with what the step
costs: the change in Outside diameter and Overall length, the dashboard checks whose level flips, and the change of
each failing check's own quantity. Every number is a difference of two `compute_all` runs. Where a side has no number
(the magnets do not fit, the model has no prediction, the slider ends, the overall-length mapping jumps) it says so in
words.

In the owner's case the card answers the question directly. At 8 magnets on the default ring (design (e)), the
Outside diameter row reads `-1 mm -> -0.006 N·m (-0.4 %)` and `+1 mm -> -0.005 N·m (-0.3 %)` while the Overall length
row reads `-1 mm -> -0.15 N·m (-9.1 %)`: one step of diameter either way moves hot-low by under 0.4 % there, and one
step of length by 9.1 % (C, section 2.3). A larger move of the diameter does cost torque: down to the 8-magnet fit
edge, Outside diameter 37.42 mm, hot-low falls 10.3 % (1.5009 against 1.6731 N·m; C, section 3c of `out-run4.txt`;
physics review 2 SF-2).

Success:
1. At design (e) the diameter row reads a change under 1 % with its true sign, never "+0 %" or "-0.00", its two sides
   at one precision (T-14).
2. Every number equals `compute_all` of a stated design (section 8 pins each row); no number shows for a design shown
   that does not fit, has no prediction, or whose pull-out or hot-low is not above zero, nor on a side whose stepped
   design does not fit or has no prediction (code review 4 MF-2: a side's number shows whatever its sign, 3.3 step 8).
3. No text says any trap of section 4.4; the G-3 review passes every card string.
4. At 1280 x 800 the card is open on the first view and the drawing keeps its 160 pt floor; at 930 x 800 and with the
   Equation panel open it starts closed (section 6).
5. A drag frame in a real browser stays under one 60 fps frame, measured before release (section 7).

### 1.1 Owner decisions (2026-10-19, rounds 1 to 5)

These are binding. They override the brief and this spec's own choices where they differ. The brief's question
numbers are in the second column; the meaning is the decision's own words from `owner-decisions.md`.

| # | Brief Q | Decision |
|---|---|---|
| 1A | Q1, approach 1 | The step card. Each Size control one step each way, change in hot-low N·m and %, plus what it costs; live while dragging. |
| 2A | Q3 (implies Q2 A) | Rows for the four Size controls only: Outside diameter, Overall length, Magnet face gap, Magnets per ring. Hot-low only. |
| 3A | Q1b | Fixed round steps: ±1 mm diameter, ±1 mm length, ±0.1 mm gap, ±2 magnets. |
| 4A | Q4 | Centre column between the readouts and the drawing, replacing the fixed label (`gui/panel.rs:936`); collapsible, open by default; starts closed while the Equation panel is open or the window is narrow (the drawing keeps its 160 pt floor). |
| 5A | Q5 | While Auto is on, the card describes the design as shown with Auto's variable held: the held control's row reads "set by Auto", and one line says Auto re-sizes after a change (the rule of `AUTO_RESETS_LENGTH` / `AUTO_RESIZES_RING`). |
| 6A | Q6 | Fixed row order, the same as the Size controls; each row states its step; no sorting, no ranking words. |
| 7A | Q7 | The N±2 neighbour lines under Magnets per ring move into the card, in hot-low; "Ring size stays the same..." and the smallest-OD fit line stay under the control. **This reverses spec1 section 3.3 for these lines**, which the owner approved explicitly. |
| 8A | Q8 | Each row shows its computed change in Outside diameter and Overall length, the dashboard checks whose level flips, and the change in margin of each counted check that already fails and that the step moves (label plus its own quantity, numbers from results, G-3-reviewed wording). |
| 9A | Q9 | Words on the failing side only ("magnets do not fit", "no prediction", "at the end of its range", the end being the slider's own range at the user's Minimum OD); the other side keeps its number; no numbers at all when the design shown does not fit or has no prediction. Validity: full `is_valid` for diameter, length and gap, D-1 for the count. A length step whose active length (`model.active_length_mm`) changes by more than the step shows words, not a number (widened by O-1: words whenever the active length changes by anything other than the step). |
| 10B | Q14, widened | One decimal of percent and N·m to the fewest decimals that keep the true sign and are not zero, on the card **and on the readout's short/margin line** (`torque_gap_text`): one percent rule and one formatter for the Size tab. **This changes the shipped readout's whole-percent format** (a number-format change, no wording change). |
| 11A | Q11 | Read-only rows, labels through `label_for`; no links, no apply. |
| 12A | Q12a + Q12b A | No new note. The card heading has a Why link to the existing `torque_from_shear` note, plus one reviewed caption line (the model's numbers for one step, everything else held). Ship with the model's numbers: BL-065 stays open, no extra caveat line. |
| 13A | Q10 | Recompute live on each changed design inside the existing `ConsequenceCache` (one key); measure a drag frame in a real browser before release; optimize (e.g. reuse the base design's geometry) only if a drag frame misses 60 fps; update `docs/ai/02-system.yaml`'s "under 1.6 ms" line to the measured figure. |
| 14A | Q15 | The full test set (section 8). The card-scoped ban list is strongest, biggest, best, most, lever, dominant, ranks (owner-confirmed). |
| 15A | Q13 | New item BL-073 (gui, `risk: physics`). Gates: GATE PASS with gate 12, G-3 physics review of every card string, whole-branch review, gui-smoke with a card step, a browser frame-time check; then STOP with a short sheet of every new text and number at the probe designs; merge and push only on the owner's OK; deploy authorization asked at that stop. |
| 16A | new (critique finding 7) | A step that changes the calibration factor (`select_calibration_factor` picks a different factor for the stepped design) shows the true `compute_all` number plus a short reviewed note on that side (e.g. "includes a calibration change"); never explained as physics. |
| 17A + 17B | new | File two backlog items in the slice's docs commit: the Overall-length slider jump with a shorter inner part (BL-074), and the BL-064 fold-in's Minor review notes with substance (BL-075). |

**The night run** (2026-10-19; `owner-decisions.md`, "Spec approval and the night run" and "Spec review round 3
rulings"). The owner approved the spec ("Approved. Run throughout the night to implement /ultracode as needed"). On that
approval the controller answered O-1 "widen" (about 22:00) and O-2 "quote none" (about 23:35), and ruled the round-3
reviews' must-fix items with each reviewer's recommended option: test 0 on the readouts' own rule (3.3), the `CardView`
keyed on the screen (6.2) and test 25's windows (8.5). The same rulings rule a later round's must-fix items with the
reviewer's recommended exact fix, and so settle code review 4's two: the card's header row wraps, so its Why link stays
inside a narrow centre region (6.1), and a count side whose calibration switch takes its torque to or below zero shows
its number with the calibration note (3.3 step 8). Each is on the STOP sheet for the owner to reverse (section 1.3,
10.1 step 6).

### 1.2 Technical decisions this spec takes (each for the spec review; none reverses an owner decision)

T-9, T-12, T-13 and T-14 narrow or read an owner decision (10B; 5A, and 9A through `ShownNoHotLow`; 8A; 10B), and
O-1's answer widens 9A's jump rule (words also when the active length changes by less than the step or the other way,
section 1.3); each is on the STOP sheet (10.1 step 6).

- **T-1. One pure module, `gui/torque_card.rs`**, public (`pub mod torque_card;` in `gui/mod.rs:9-36`), no egui
  (section 3). The panel draws it; the screen-dump tool prints it.
- **T-2. The neighbour logic moves, it is not copied.** `NeighbourOutcome`'s rule (`neighbour_outcome`,
  `gui/guidance.rs:2924-2944`), `neighbour_counts` (`:2902-2909`; as `count_step`, its rule for one side, 3.3) and the
  two shipped side words (`:2742`, `:2745`) move into `torque_card.rs` and serve all four rows. `prediction`
  (`:2913-2917`, whose only caller is `neighbours`, `:2888`) leaves with the neighbour lines: the card's test 0 and its
  step 6 read the readouts' own rule, `size_controls::predicts` (3.3; physics review 3 MF-2, code review 4 SF-2), so
  the readouts, the design shown and every side share one rule. `Consequences.neighbours` (`:2807`) becomes
  `Consequences.card`.
- **T-3. The "set by Auto" marking is applied outside the cache** (section 3.4).
- **T-4. Two word forms beyond 9A's three plus the jump**: "the keyway breaks through the hub" (full `is_valid` has a
  keyway clause; the text is the shipped one of `gui/panel.rs:926`, made a shared constant) and "not a value the
  slider takes" (an odd count from a design file). Section 3.3 gives the order.
- **T-5. The two size rows are left out where the Size tab draws no such slider** (`dimensions` is `None`: an odd or
  out-of-range count, or geometry the controls cannot define; `gui/size_controls.rs:243`, `:324-329`), as 6A's "the
  same as the Size controls" reads.
- **T-6. Margins come from one shared table of check terms**, lifted from the test-private `terms()`
  (`gui/guidance.rs:4271-4350`); the temperature check quotes none (section 3.6).
- **T-7. The card's open state**: decided when the view changes (the Equation panel opened or closed, the window
  resized), never on a design change; the body capped above the drawing's floor (section 6.2). The view is keyed on the
  screen's size, not the centre region's width, which a side panel widening to fit a design's content moves (code
  review 3 MF-2).
- **T-8. `AUTO_RESIZES_RING` and `AUTO_RESETS_LENGTH` leave the Magnets per ring lines** with the neighbour lines they
  describe ("the lines below keep ..."); the card's Auto line takes their rule (5A). `RING_SIZE_STAYS` stays under the
  control (7A) and needs one clause changed (section 4.3). So with the card closed (930 x 800, or the Equation panel
  open) nothing near Magnets per ring says Auto re-sizes after a count change: 7A names the two lines that stay under
  the control, and these are not among them. On the STOP sheet (code review 2 S15).
- **T-9. The decimals search stops at `from + 2`** (section 5): a change under half a unit of that decimal reads `from`
  decimals with its true sign ("+0.00", "-0.00"). This narrows 10B's "not zero" for float noise only: Auto's solve
  residual at (b) is 1e-10 N·m, and no card side here comes near the cap (the smallest, 0.0038 N·m, (a) at Overall
  length 22.10 mm, diameter `+1`, drawn "+0.004 N·m (+1.1 %)"; C, section 3d of `out-run5.txt`; physics review 3
  SF-3). On the STOP sheet (R-6).
- **T-10. 4A's "the window is narrow" is a width rule**: the card starts closed while the centre region is narrower
  than `CARD_OPEN_MIN_WIDTH` = 400 pt (between the 278 pt of a 930 x 800 window and the 628 pt of 1280 x 800,
  measured), as well as while the open card would take the drawing under its floor (section 6.2). Without the width
  rule a short card (the one-line forms, 64 pt at 278 pt wide) would start open in a 930 x 800 window.
- **T-11. The gearbox phrase names its torque**: "Gearbox input torque: cold-high +0.03 N·m" (physics review SF-1).
  Every other N·m on the card is hot-low at the output; this one is cold-high at the gearbox input, as the dashboard's
  Problem calls it ("Cold-high at the gearbox input", `gui/guidance.rs:2443`), in the form of the brief's Q8 A example
  the owner took ("Gearbox input torque: cold-high +0.34 N·m"). It names a third torque on the Size tab; G-3 decides.
- **T-12. The three one-line forms draw their line alone**: `ShownDoesNotFit`, `ShownNoPrediction` and
  `ShownNoHotLow` draw no caption (it talks about steps) and no Auto line (it talks about rows), Auto on or off.
- **T-13. No Temperature limits margin** (narrows 8A, as T-9 narrows 10B; owner question O-2, answered "quote none",
  section 1.3; code review 2 MF-2): the card quotes no quantity of the Temperature limits check, even when the check
  fails on both designs. Its terms read the stored 3D demagnetization and slip-heating fields of the example design,
  which do not follow the step, so a number would teach that they do (section 3.6, R-9). Its flips still show.
- **T-14. A row's two N·m share their decimals** (a reading of 10B per row, not per number; physics review 2 SF-1): the
  card draws both sides' N·m at the larger of their `change_decimals(.., 2)` (section 5). (e)'s diameter row reads
  `-0.006` and `-0.005` N·m (per number `-0.01` and `-0.005`: a 2:1 ratio on screen for a true 1.35:1, -0.00626 against
  -0.00465), and (e2)+Auto length's reads `-0.005` and `-0.005` (per number `-0.005` and `-0.01` for -0.00477 and
  -0.00502) (C, section 3c of `out-run4.txt`). Each number still keeps its true sign and a digit that is not zero, as
  10B asks; it is not always the fewest such decimals. Two sides can then read the same N·m, as their one-decimal
  percents already can; that is not BL-068's trap (a value and its own limit reading equal). Percents, size changes
  and margins keep the per-number rule (each is one quantity on one side). On the STOP sheet with (e)'s row both ways.

### 1.3 Owner questions (answered)

Revisions 1 and 2 put these two questions to the owner. Both are answered with the recommended option: the
controller's ruling on the owner's night-run approval ("Approved. Run throughout the night", 2026-10-19;
`owner-decisions.md`, "Spec approval and the night run" for O-1, "Spec review round 3 rulings" for O-2). Each answer
is on the STOP sheet for the owner to reverse (10.1 step 6). The options are kept as the record; the rest of the spec
states the answer as its one rule.

- **O-1. A length step whose overlapping magnet length changes by less than the step** (physics review MF-2).
  **Answered 2026-10-19 (about 22:00): widen.** 9A as written shows words when `model.active_length_mm` changes by
  **more** than the step. With matched parts, a blank override and the cup cavity input (`metal.cup_depth_mm`, slider
  3-40 mm) shorter than the outer ring's own length, the first step sets both rings to the cavity input ± 1 mm
  (section 2.3, "short cavity"). At 11.0 mm the `-1 mm` side moves the active length by -2.70 mm (words under either
  rule), but the `+1 mm` side moves it by -0.70 mm, and under 9A as written it printed **-0.14 N·m (-6.2 %) for +1 mm
  of Overall length**: a jump read as a slope, a trap of section 4.4 that "more than the step" lets through (C).
  - **The answer (the recommended option): the widened rule**, words when the active length changes by anything
    other than the step, `|Δactive − sign·step| > STEP_SLACK_MM` (3.3 step 7), with `JUMP` reading "not a step: the
    overlapping magnet length does not change by the step". It catches the mixed pair and every short-cavity side
    (11.0, 12.0 and 12.5 mm) and keeps the reverse pair, matched parts in the default cavity (15.5 mm) and every other
    design of section 2 as numbers (C, section 3d of `out-run5.txt`: the rule was run at every design). It adds no
    input and one comparison.
  - Not taken: keep 9A as written, `JUMP` reading "not a step: the overlapping magnet length changes by more than
    the step", and put the short-cavity `+1 mm` side on the STOP sheet as a known slope-like jump (BL-074 tracks the
    mapping).
  - Not taken: "each ring's own length changes by the step" (physics review SF-5) also catches the reverse pair's
    first step, which trims the longer inner ring 12.7 -> 5.35 / 7.35 mm (C). Its hot-low number is the model's
    honest value (the model has no overhang term), so the spec keeps it a number and BL-074 records the trim.
- **O-2. The Temperature limits margin** (code review 2 MF-2). **Answered 2026-10-19 (about 23:35): quote none**
  (carried as recommended; T-13). 8A asks for "the change in margin of each counted check that already fails and that
  the step moves", and the brief's Q8 A, which the owner took, names the temperature terms among those margins
  (`context-brief.md:420`). The card quotes no Temperature limits margin, even when the check is red on both designs.
  Its terms read stored 3D fields that do not follow the step, so a number would teach that they do (section 3.6;
  physics review 3 confirms it: the magnet limit does not move on any step, and the slip-fault peak margins move
  0.04-0.06 °C a step, partly through the stored heating field). Its flips still show.
  - **The answer (the recommended option): quote none** (section 3.6; T-13). A flip is the dashboard's own verdict on
    the stepped design; a margin in °C would put a stored field's number beside a step that does not move that field.
  - Not taken: quote the governing term's margin in °C, with the stored-3D label beside it (a new phrase for G-3,
    and a quoted quantity for the temperature terms in `check_terms`).

## 2. Numbers basis

### 2.1 The designs (exact inputs)

Every design is `DesignInputs::default()` with the edits listed. "Shown" is the design the panel shows: for Auto, the
inputs with the variable at the solved or best value through `gui::sizing::sized` (`gui/sizing.rs:107-114`), searched
over the range the panel searches (`gui/panel.rs:424-430`). The Minimum OD preference is `None` unless given.

| Key | Edits | Shown (E) |
|---|---|---|
| (a) | none | hot-low 2.2852 N·m; fails 4 of 9: Hot-low torque, Running clearance, Cup wall, Gearbox input torque |
| (b) | the gui-smoke share link (`.claude/workflows/gui-smoke.js:48`, decoded by `gui::session::decode_share_payload`): `metal.face_gap_mm` 1.5, Auto on magnet length | axial override 14.181318158 mm (Solved); hot-low 2.5000; fails 4 of 9: Running clearance, Cup wall, Gearbox input torque, Space claim |
| (c) | `coupling.backiron` 0 | hot-low 1.4428; calibration factor 1.0495 (the measured one); fails 3 of 9: Hot-low torque, Running clearance, Shaft joints |
| (e) | `coupling.npole` 8 | hot-low 1.6731; fails 3 of 9: Hot-low torque, Running clearance, Cup wall |
| (e2) | (e) and `metal.required_min_Nm` 1.5 | hot-low 1.6731; fails 2 of 9: Running clearance, Cup wall |
| mixed | `part_inner` B442, `part_outer` B842SH, axial override blank | hot-low 1.0099; active length 6.35 mm; fails 4 of 9 and 1 caution (Hot-low, Running clearance, Cup wall, Shaft joints; Temperature limits amber) |
| reverse | `part_inner` B842SH, `part_outer` B442 | hot-low 1.0099; fails 3 of 9 and 1 caution |
| (a)+Auto length | Auto on magnet length | override 13.769522 mm (Solved); hot-low 2.5000; fails 3 of 9 |
| (a)+Auto count | Auto on magnets per ring | not reachable: best 2.285205 N·m at 10 (shown = (a)) |
| (a)+Auto ring | Auto on ring size | ring 11.675697 mm, cap 46.0084 mm (Solved); OD 46.01; hot-low 2.5000; fails 4 of 9 (Space claim among them) |
| (e2)+Auto length | (e2), Auto on magnet length | override 11.556900 mm; hot-low 1.5000 |
| (e2)+Auto count | (e2), Auto on magnets per ring | 8 (Solved): shown = (e2) |
| (e2)+Auto ring | (e2), Auto on ring size | ring 9.000000 mm (the radius floor), cap 40.3105; hot-low 1.6360 |
| (e2)+Auto ring, low Min OD | (e2), Minimum OD 31.7596 mm (its lowest), Auto on ring size | ring 7.665128 mm, cap 37.4208; hot-low 1.5009 (the 8-magnet fit edge) |
| short magnets | `test_support::short_magnets()` (`gui/test_support.rs:8-16`): `c_end` 0.5, both parts blank, manual lengths 2.0 mm | no prediction |
| end effect at a step | `c_end` 0.5, axial override 5.0 mm | hot-low 0.1196 |
| odd count | `coupling.npole` 9 (a design file's count; the slider takes even counts only) | hot-low 2.0295; no size controls (`dimensions` is `None`) |
| (e), low Min OD | (e), Minimum OD 31.7596 | as (e) |
| (a), Min OD 42.8 | (a), Minimum OD 42.8 mm (the diameter shown) | as (a) |
| (a), Min OD 45 | (a), Minimum OD 45.0 mm (above the diameter shown: the Minimum OD slider runs to 80 mm "without changing your current size", `gui/size_controls.rs:252`, and the diameter slider clamps only edits, `SliderClamping::Edits`, `gui/input_ui.rs:72`); also at 43.3 mm | as (a) |
| shown does not fit | (a) at ring 9.5 mm through `with_ring_radius` (cap 41.4331) | fails 5 of 9, Magnets fit among them |
| keyway through | `coupling.bore_mm` 16, `coupling.keyway_depth_mm` 2.5 | hub wall past the keyway below 0; Shaft joints amber (no wall) |
| gap at 0.3 | `metal.face_gap_mm` 0.3 (its slider's bottom) | hot-low 3.2734 |
| npole 4 | `coupling.npole` 4 (its slider's bottom) | hot-low 0.4013; the cup sets the outside diameter |
| hot day | `test_support::hot_day` (ambient 80 °C, driving rise 60 °C) | Temperature limits amber |
| only the joint fails | `test_support::passing_design()` with `shaft_joint.hub_interference_min_um` 8 | fails 1 of 9: Shaft joints |
| passing | `test_support::passing_design()` | passes all 9 |
| short cavity | (a) with `metal.cup_depth_mm` 11.0: matched B842SH, blank override, the cup cavity input (slider 3-40 mm) shorter than the outer ring's 12.7 mm | as (a): hot-low 2.2852, fails 4 of 9; Overall length 27.30 mm (range 18.30-67.10) |
| temperature edge | (a) with `temperature.duty.hot_ambient_C` 82.08 (the temperature verdict's edge is at 82.0913 °C, C) | hot-low 2.2852; fails 4 of 9; Temperature limits green |
| npole 40 | `coupling.npole` 40, both parts blank, manual inner and outer widths 1.2 mm (`forty.rs`) | the blocks fit; hot-low 0.4857 |
| slider tops | (a) with `metal.face_gap_mm` 5.0; (a) at the Outside diameter's top (80 mm) through the slider mapping; (a) at the Overall length range's top (69.90 mm) and bottom (21.10 mm) through `resize_length` | as built |
| steps to an end | one slider step inside each end (code review 2 MF-1): (a) and (e) at Overall length 68.90 and 22.10 mm through `resize_length`; (a) with `metal.face_gap_mm` 4.90 and 0.40; (a) and (e) at Outside diameter 79.0 mm through the slider mapping; (e) with Minimum OD 41.80 mm (the diameter shown less 1 mm; at (a) the `-1 mm` side does not fit) | as built; `resize_length(.., 68.9)` shows 68.90000000000000568 mm |
| variation 1.0 | (a) with `metal.variation` 1.0 (a design file's value; `validate()` checks only that it is finite) | hot-low 0 while the model predicts (pull-out 2.6885 N·m, f_end 0.896): `ShownNoHotLow`; at 1.5, hot-low -1.3442 N·m, the same (C) |
| zero torque | (a) with both parts blank (`coupling.magnets.part_inner` and `part_outer` "") and `coupling.magnets.manual_inner_br_T` 0 (a design file's value: the slider stops at 0.2 T, and `set()` and `validate()` check only that it is finite, `check_value`, `engine/meta.rs:745-755`) | pull-out 0 and hot-low 0 N·m, both shown as numbers (f_end 0.896, in range); fails 2 of 9 and 2 cautions (Hot-low torque, Running clearance; Shaft joints and Gearbox input torque amber): `ShownNoHotLow` (C, section 3d of `out-run5.txt`; physics review 3 MF-2). An unmagnetized inner ring transmits no torque, so 0 N·m is the model's true prediction. `calibration.f_cal_original` 0 (slider 0.5-1.5) reads the same, and at -0.95 the pull-out is -2.6885 and hot-low -2.2852 N·m, both shown (C) |
| (c), `f_cal_original` -0.95 | (c) and `calibration.f_cal_original` -0.95 (a design file's value; `validate()` passes) | as (c): the design shown takes the measured factor, 1.0495 (`f_cal_updated` is `f_cal_original` × measured / model, and the model's torque carries `f_cal_original` too, so it cancels, `engine/calibration.rs:215`, `:229`, `:235`), hot-low 1.4428 and pull-out 1.6974 N·m, fails 3 of 9 as (c); its count `-2` design (8 poles) takes the coefficient itself (`select_calibration_factor`, `engine/model.rs:584-600`): hot-low -0.8505 and pull-out -1.0006 N·m, f_end 0.870 (C, section 3e of `out-run6.txt`; code review 4 MF-2). At `f_cal_original` 0 the measured factor is NaN and (c) reads `ShownNoPrediction` (C) |

### 2.2 How a number is made

- Hot-low change: `stepped.metal.torque_hot_low_Nm - shown.metal.torque_hot_low_Nm`, each from its own `compute_all`;
  percent of the shown hot-low.
- Size change: `dimensions(stepped)` less `dimensions(shown)` (`engine/size_controls.rs:249-251`): the Outside
  diameter and Overall length the sliders show; no value (`None`) when either design's `dimensions` is `None` (the odd
  count's gap sides, C). On the diameter row the stepped diameter is the shown one ± 1 mm, and on the length row the
  stepped length is ± 1 mm, to 7.1e-15 mm at every design here (C, section 3 of `out-run3.txt`), so neither row repeats
  its own step; the other dimension moves on those two rows by float noise only (largest 3.6e-15 mm, C), which `shows`
  (section 3.6) keeps off the screen.
- Flips: `level(guide, ..)` of each counted guide (`gui/guidance.rs:633-658`) on the shown and the stepped design.
- Margin changes: the change of a failing check's own quantity, read from the results (section 3.6).
- Calibration: `model.f_cal` (the factor `select_calibration_factor` returned, `engine/api.rs:243-251`, a result at
  `engine/model.rs:238`) of the two designs.
- In the tables, "words: ..." is a side with no number; "the number it would show" (jump rows only) is printed for the
  record, never on screen.

### 2.3 The card at every design (C; N·m to 4 decimals, % to 2; on screen they follow section 5)

The tables name a quoted quantity by its check ("Gearbox input torque +0.0281 N·m"); the drafted lines show the phrase
of section 3.6 ("Gearbox input torque: cold-high +0.03 N·m").

#### (a) default

| Row | − step | + step |
|---|---|---|
| Outside diameter ±1 mm | words: magnets do not fit | +0.0798 N·m (+3.49 %); flip: Space claim Good -> Bad; Running clearance +0.0125 mm; Cup wall, wall needed +0.0651 mm; Gearbox input torque +0.0281 N·m |
| Overall length ±1 mm | -0.2008 N·m (-8.79 %); Gearbox input torque -0.0707 N·m | +0.2008 N·m (+8.79 %); Gearbox input torque +0.0707 N·m |
| Magnet face gap ±0.1 mm | +0.0759 N·m (+3.32 %); Running clearance -0.1000 mm; wall needed +0.0182 mm; Gearbox input torque +0.0267 N·m | -0.0733 N·m (-3.21 %); Running clearance +0.1000 mm; wall needed -0.0177 mm; Gearbox input torque -0.0258 N·m |
| Magnets per ring ±2 | -0.6121 N·m (-26.79 %); flip: Gearbox input torque Bad -> Good; wall needed +0.4797 mm | words: magnets do not fit |

No size change on any row: the cap sets the Outside diameter (42.80 mm) at every stepped design; the count row's cup
grows to 42.44 mm under it. No calibration change. Hot-low fails on every side and is never repeated (section 3.6).

Drafted lines (C; the words are section 4's drafts):
```
One step of each size control   [Why: How the coupling makes torque]
The change in the model's hot-low torque after one step of one control, everything else held.
Outside diameter: -1 mm -> magnets do not fit; +1 mm -> +0.08 N·m (+3.5 %), Space claim: fails, Running clearance +0.01 mm, Cup wall: wall needed +0.07 mm, Gearbox input torque: cold-high +0.03 N·m
Overall length: -1 mm -> -0.20 N·m (-8.8 %), Gearbox input torque: cold-high -0.07 N·m; +1 mm -> +0.20 N·m (+8.8 %), Gearbox input torque: cold-high +0.07 N·m
Magnet face gap: -0.1 mm -> +0.08 N·m (+3.3 %), Running clearance -0.10 mm, Cup wall: wall needed +0.02 mm, Gearbox input torque: cold-high +0.03 N·m; +0.1 mm -> -0.07 N·m (-3.2 %), Running clearance +0.10 mm, Cup wall: wall needed -0.02 mm, Gearbox input torque: cold-high -0.03 N·m
Magnets per ring: -2 -> -0.61 N·m (-26.8 %), Gearbox input torque: passes, Cup wall: wall needed +0.48 mm; +2 -> magnets do not fit
```

#### (b) the smoke design, Auto on magnet length (the Overall length row reads "Set by Auto ...")

| Row | − step | + step |
|---|---|---|
| Outside diameter ±1 mm | words: magnets do not fit | +0.0892 N·m (+3.57 %); Running clearance +0.0125 mm; wall needed +0.0643 mm; Gearbox input torque +0.0314 N·m; Space claim, Rotating diameter +1.0000 mm (no flip: the check already fails on its large-diameter stack; the step puts a second axis over, so that axis is quoted, section 3.6) |
| Overall length ±1 mm (held; computed, not shown) | -0.1945 N·m (-7.78 %); flips: Hot-low torque Good -> Bad, Space claim Bad -> Good; Gearbox input torque -0.0685 N·m | +0.1945 N·m (+7.78 %); Gearbox input torque +0.0685 N·m; Space claim, Large-diameter stack +1.0000 mm |
| Magnet face gap ±0.1 mm | +0.0827 N·m (+3.31 %); Running clearance -0.1000 mm; wall needed +0.0177 mm; Gearbox input torque +0.0291 N·m | -0.0798 N·m (-3.19 %); flip: Hot-low torque Good -> Bad; Running clearance +0.1000 mm; wall needed -0.0173 mm; Gearbox input torque -0.0281 N·m |
| Magnets per ring ±2 | -0.6579 N·m (-26.32 %); flip: Hot-low torque Good -> Bad; wall needed +0.4753 mm; Gearbox input torque -0.2317 N·m | words: magnets do not fit |

Every step that lowers hot-low turns Hot-low torque red: Auto put hot-low at the requirement (2.5000 N·m), and the
row is computed before Auto re-sizes, as the Auto line says.

#### (c) no back iron (the calibration switch)

| Row | − step | + step |
|---|---|---|
| Outside diameter ±1 mm | words: magnets do not fit | +0.0271 N·m (+1.88 %); flip: Space claim Good -> Bad; Running clearance +0.0125 mm; Shaft joints, Inner hub smallest measured interference that holds -0.0916 µm; Cup smallest measured interference that holds +0.0878 µm |
| Overall length ±1 mm | -0.1268 N·m (-8.79 %); inner hub -0.0703 µm; cup -0.4107 µm | +0.1268 N·m (+8.79 %); inner hub +0.0602 µm; cup +0.4107 µm |
| Magnet face gap ±0.1 mm | +0.0498 N·m (+3.45 %); Running clearance -0.1000 mm; inner hub +0.2042 µm; cup +0.1612 µm | -0.0481 N·m (-3.33 %); Running clearance +0.1000 mm; inner hub -0.1974 µm; cup -0.1558 µm |
| Magnets per ring ±2 | -0.5923 N·m (-41.05 %); **calibration factor 1.0495 -> 0.9500 (includes a calibration change)**; inner hub -2.4300 µm; cup -1.9182 µm | words: magnets do not fit |

The count side's -41.05 % holds the factor's -9.48 %; without it the count alone is -34.88 % (critique, critic run;
(1 - 0.4105) / (0.9500 / 1.0495) - 1, H). The card prints the true -41.1 % with the note (16A). Both joints fail on
the aluminium hub and body at (c); the interference phrases quote the needed minimum (section 3.6).

#### (e) the owner's case: 8 magnets on the default ring

| Row | − step | + step |
|---|---|---|
| Outside diameter ±1 mm | **-0.0063 N·m (-0.37 %)**; Running clearance -0.0130 mm; wall needed -0.0790 mm | **-0.0046 N·m (-0.28 %)**; flip: Space claim Good -> Bad; Running clearance +0.0122 mm; wall needed +0.0790 mm |
| Overall length ±1 mm | -0.1514 N·m (-9.05 %) | +0.1514 N·m (+9.05 %); flip: Gearbox input torque Good -> Bad |
| Magnet face gap ±0.1 mm | +0.0510 N·m (+3.05 %); flip: Gearbox input torque Good -> Bad; Running clearance -0.1000 mm; wall needed +0.0227 mm | -0.0489 N·m (-2.92 %); Running clearance +0.1000 mm; wall needed -0.0221 mm |
| Magnets per ring ±2 | -0.8868 N·m (-53.00 %); Outside diameter +2.23 mm (42.80 -> 45.03); flip: Space claim Good -> Bad; wall needed +0.7995 mm | +0.6121 N·m (+36.58 %); flip: Gearbox input torque Good -> Bad; wall needed -0.4797 mm |

Drafted lines (C):
```
Outside diameter: -1 mm -> -0.006 N·m (-0.4 %), Running clearance -0.01 mm, Cup wall: wall needed -0.08 mm; +1 mm -> -0.005 N·m (-0.3 %), Space claim: fails, Running clearance +0.01 mm, Cup wall: wall needed +0.08 mm
Overall length: -1 mm -> -0.15 N·m (-9.1 %); +1 mm -> +0.15 N·m (+9.1 %), Gearbox input torque: fails
Magnet face gap: -0.1 mm -> +0.05 N·m (+3.0 %), Gearbox input torque: fails, Running clearance -0.10 mm, Cup wall: wall needed +0.02 mm; +0.1 mm -> -0.05 N·m (-2.9 %), Running clearance +0.10 mm, Cup wall: wall needed -0.02 mm
Magnets per ring: -2 -> -0.89 N·m (-53.0 %), Outside diameter +2.2 mm, Space claim: fails, Cup wall: wall needed +0.80 mm; +2 -> +0.61 N·m (+36.6 %), Gearbox input torque: fails, Cup wall: wall needed -0.48 mm
```
As shipped today `fixed` and `signed_percent` (`gui/format.rs:82-100`) would print the diameter row as
"-0.01 N·m (+0 %)" and "-0.00 N·m (+0 %)" (critique finding 1); section 5 removes both. Read per number (revision 1),
the row's N·m was "-0.01" against "-0.005", a 2:1 ratio for a true 1.35:1; T-14 draws both at the row's 3 decimals
(physics review 2 SF-1).

#### (e2) the owner's case with Required 1.5 N·m

Every row and side is the same as (e) (the requirement moves no torque), except the count row's `-2` side, which adds
the flip Hot-low torque Good -> Bad (0.7863 N·m against 1.5): -0.8868 N·m (-53.00 %); Outside diameter +2.23 mm;
flips Hot-low torque and Space claim; wall needed +0.7995 mm. At (e2) the gearbox check passes, so the length and gap
rows' gearbox flips match (e)'s.

#### The mixed-length pair

| Row | mixed (B442 inner, B842SH outer) | reverse (B842SH inner, B442 outer) |
|---|---|---|
| Overall length −1 mm | **words: jump** (active 6.35 -> 11.70 mm; the number it would show: +106.39 %) | -0.2008 N·m (-19.89 %) (active 6.35 -> 5.35) |
| Overall length +1 mm | **words: jump** (active 6.35 -> 13.70 mm; +146.16 %) | +0.2008 N·m (+19.89 %) |
| Outside diameter ±1 mm | magnets do not fit / +0.0300 N·m (+2.98 %; 0.03004991 N·m); flip: Space claim; Running clearance +0.0125 mm; wall needed +0.0651 mm | the same |
| Magnet face gap ±0.1 mm | +0.0341 N·m (+3.38 %) / -0.0329 N·m (-3.26 %); Running clearance ∓0.1000 mm; wall needed +0.0182 / -0.0177 mm | the same |
| Magnets per ring ±2 | -0.2984 N·m (-29.55 %); wall needed +0.4797 mm / magnets do not fit | the same |

`resize_length` (`engine/size_controls.rs:453-466`) sets both rings' override to `requested - length_range.min +
axial_range.min` (`:460`): from a blank override the length range follows the outer ring (the cup cavity's,
`:220-226`), so a step of the mixed pair sets both rings near the outer's 12.7 mm. The reverse pair moves the active
length by the step, but its first step trims the longer inner ring 12.7 -> 5.35 / 7.35 mm, the hub length following
it (13.0 -> 5.65 / 7.65 mm, C); the model has no overhang term, so the hot-low number is the model's honest value, and
BL-074 records the trim. So do matched parts move the active length by the step while the cup cavity input is at
least the outer ring's length. With it shorter (blank override), the first step sets both rings to the cavity input
± 1 mm, because the cavity `follow`s the outer ring as `max(cup_depth + L - own, L)` (`engine/housing.rs:106-108`).
At (a) with `metal.cup_depth_mm` 11.0 the -1 mm side moves the active length 12.70 -> 10.00 mm and the +1 mm side
12.70 -> 12.00 mm: both sides read jump words under the widened rule (O-1, 3.3 step 7). The `+1 mm` side, a number
under 9A as written, would show -6.15 % for +1 mm of Overall length (physics review 1, `housing.rs`; C, section 3b of
`out-run3.txt`, which also gives 12.5 mm: -1.20 / +0.80 mm, +7.03 %, and 12.0 mm: -1.70 / +0.30 mm, +2.64 %; each of
those `+1 mm` sides reads jump words too, C, section 3d of `out-run5.txt`). The shipped slider does the same; BL-074
records it (section 9).

#### Auto on each variable it can hold (the held row reads `SET_BY_AUTO`; its sides are computed, not shown)

| Design | Held row | The other rows (C) |
|---|---|---|
| (a)+Auto length | Overall length | Outside diameter: no fit / +0.0882 N·m (+3.53 %), Space claim flip, Running clearance +0.0125 mm, wall needed +0.0651 mm, Gearbox input torque +0.0311 N·m. Gap: +0.0830 N·m (+3.32 %) / -0.0801 N·m (-3.20 %) with the Hot-low flip on `+`. Count: -0.6649 N·m (-26.60 %), Hot-low flip, wall needed +0.4797 mm, Gearbox input torque -0.2342 N·m / no fit. |
| (a)+Auto count | Magnets per ring | Diameter, length and gap rows equal (a)'s (the shown design is (a)). |
| (a)+Auto ring | Outside diameter | Length: -0.2225 N·m (-8.90 %), Hot-low flip, Gearbox input torque -0.0784 N·m / +0.2225 N·m (+8.90 %), +0.0784 N·m. Gap: +0.0812 (+3.25 %) / -0.0781 (-3.12 %), Hot-low flip on `+`. Count: -0.8790 N·m (-35.16 %), flips Hot-low (fails) and Gearbox input torque (passes), wall needed +0.5319 mm / no fit. |
| (e2)+Auto length | Overall length | Diameter: **-0.0048 N·m (-0.32 %) and -0.0050 N·m (-0.33 %), both with the Hot-low flip** (hot-low sits at 1.5000; on screen both read "-0.005 N·m (-0.3 %)", T-14); `+1` also flips Space claim. Gap: +0.0458 (+3.05 %) / -0.0439 (-2.93 %), Hot-low flip on `+`. Count: -0.7993 N·m (-53.29 %), OD +2.23 mm, flips Hot-low and Space claim / +0.5556 N·m (+37.04 %), Gearbox flip. |
| (e2)+Auto count | Magnets per ring | Diameter, length and gap rows equal (e2)'s. |
| (e2)+Auto ring | Outside diameter | Length: -0.1463 N·m (-8.94 %), Hot-low flip / +0.1463 (+8.94 %), Gearbox flip. Gap: +0.0517 (+3.16 %) / -0.0497 (-3.04 %). Count: -0.7864 N·m (-48.07 %), OD +2.06 mm, Hot-low flip / no fit. |
| (e2)+Auto ring, low Min OD | Outside diameter | Length: -0.1323 N·m (-8.82 %), Hot-low flip / +0.1323 (+8.82 %). Gap: +0.0483 (+3.22 %) / -0.0466 (-3.11 %), Hot-low flip on `+`. Count: -0.5639 N·m (-37.57 %), OD +1.87 mm, Hot-low flip / no fit. |

#### Edges

| Design | What the card shows (C) |
|---|---|
| short magnets | No rows, no caption, no Auto line (T-12): "No step is compared while the design shown has no torque prediction." |
| shown does not fit | No rows, no caption, no Auto line: "No step is compared while the magnets of the design shown do not fit." |
| variation 1.0 | One line alone, no caption, no Auto line (T-12): "No step is compared while the design shown has a pull-out or hot-low torque that is not above zero." (`ShownNoHotLow`). The model predicts, and the readout above the card shows the pull-out, 2.69 N·m; hot-low, the percent's base, is 0 (code review 1 S1; physics review 2 MF-1, code review 2 S11: the round-1 draft routed it to `SHOWN_NO_PREDICTION`, false here). At 1.5, hot-low -1.3442 N·m: the same (C). |
| zero torque | The same one line alone (T-12; `ShownNoHotLow`). The readouts above the card show the pull-out and hot-low as numbers, 0 N·m each, so `SHOWN_NO_PREDICTION`, where revision 2's rule sent it (a pull-out not positive), would contradict them (physics review 3 MF-2). `calibration.f_cal_original` 0 and -0.95 read the same, and so does -0.95 with `metal.variation` 1.5 (pull-out -2.6885 N·m, hot-low +1.3442 N·m: the line stays true through the pull-out) (C, section 3d of `out-run5.txt`). |
| (c), `f_cal_original` -0.95 | Every side equals (c)'s, bit for bit, but the count `-2` side: **-2.2932 N·m (-158.95 %)**, to a hot-low of -0.8505 N·m (pull-out -1.0006 N·m, f_end 0.870: the model predicts), with the calibration change (factor 1.0495 -> -0.9500) and the flip Shaft joints Bad -> Good; no joint quantity is quoted, since the check passes on the stepped design. It is a number, as 3.3 step 8 says: 16A asks for the true `compute_all` number with its note, and a side's number shows whatever its sign (code review 4 MF-2). The row reads "Magnets per ring: -2 -> -2.29 N·m (-158.9 %), includes a calibration change, Shaft joints: passes; +2 -> magnets do not fit" (C, section 3e of `out-run6.txt`). |
| end effect at a step | Overall length `-1 mm`: **no prediction** (override 4.0 mm, f_end at or below 0); `+1 mm`: +0.2008 N·m (+167.93 %). Count `-2`: **no prediction** (pitch grows at 8 poles); `+2`: magnets do not fit. Diameter: no fit / -0.0265 N·m (-22.16 %) (the pitch grows with the ring), Space claim flip. Gap: +0.0072 (+6.00 %) / -0.0068 (-5.72 %). |
| odd count (9) | The Outside diameter and Overall length rows are left out (T-5: the Size tab shows "Size controls unavailable", `gui/size_controls.rs:324-329`). Gap: +0.0653 N·m (+3.22 %) / -0.0628 N·m (-3.10 %), with Running clearance ∓0.1000 mm, wall needed +0.0202 / -0.0197 mm, Gearbox input torque +0.0230 / -0.0221 N·m. Count: both sides **"not a value the slider takes"** (7 and 11 are odd; `neighbour_counts` keeps even counts only, `gui/guidance.rs:2902-2909`). |
| (e), low Min OD | Equal to (e): the slider's own minimum at the default preference (40.3105 mm) and the lowest (31.7596 mm) map −1 mm to the same ring, 9.688060 mm, and +1 mm to 10.611940 mm (C, section 3 of `out-run3.txt`). The two diverge only where −1 mm leaves the default range; `resize_diameter` and the slider mapping also agree at (a) and (e) at the default preference (C). |
| (a), Min OD 42.8 | Outside diameter `-1 mm`: **at the end of its range** (the user's Minimum OD is the diameter shown); every other side equals (a)'s. At the default preference the same side reads "magnets do not fit" (the range starts at 40.38 mm). |
| (a), Min OD 45 | Outside diameter `-1 mm` and `+1 mm`: both **at the end of its range** (41.80 and 43.80 mm both lie below the 45.00-80.00 mm range), while the slider's handle sits at its bottom end and moves up; every other side equals (a)'s. At Minimum OD 43.3 mm only the `-1 mm` side reads words; the `+1 mm` side equals (a)'s, +0.0798 N·m (+3.49 %) (code review 5 SF-3, reproduced). G-3 reads the `+1 mm` side's words here (10.1 step 2); other words would go beyond 9A, so any change goes to the owner at the STOP. |
| keyway through | Overall length and Magnet face gap: all four sides **"the keyway breaks through the hub"** (full `is_valid`; neither step moves the hub wall). Outside diameter `+1 mm`: +0.0798 N·m (+3.49 %) with flips Shaft joints Caution -> Bad (the wall returns, so the joint is judged and slips) and Space claim. Magnets per ring `-2`: -0.6121 N·m (-26.79 %) (D-1 skips the keyway). |
| gap at 0.3 | Gap `-0.1 mm`: **at the end of its range**; `+0.1 mm`: -0.1037 N·m (-3.17 %). The count row's `-2` flips Shaft joints Bad -> Good (cold-high falls) and Hot-low torque Good -> Bad. |
| npole 4 | Count `-2`: **at the end of its range**; `+2`: +0.3850 N·m (+95.93 %), Outside diameter -9.31 mm. The cup sets the diameter here, so the gap row shows Outside diameter ∓0.28 mm and Space claim, Rotating diameter ∓0.2828 mm; the diameter row shows Space claim, Rotating diameter ±1.0000 mm. |
| hot day | Equal to (a): Temperature limits is amber, not red, and the card quotes no temperature margin in any case (section 3.6). |
| only the joint fails | Every side quotes Shaft joints, Inner hub smallest measured interference that holds (e.g. length ∓1 mm: -0.0404 / +0.0346 µm; count -2: -0.8670 µm). The gap `-0.1 mm` side flips Running clearance Good -> Bad. |
| passing | Flips only: diameter `+1` Space claim; gap `-0.1` Running clearance; count `-2` Hot-low torque, Cup wall and Space claim (Outside diameter +0.90 mm). |
| short cavity | Overall length `-1 mm`: **jump** words (active 12.70 -> 10.00 mm; the number it would show: -23.73 %). `+1 mm`: **jump** words (active 12.70 -> 12.00 mm; the number it would show: -0.1406 N·m, -6.15 %, with Gearbox input torque -0.0495 N·m; Overall length 27.30 -> 28.30 mm), the one side of section 2 where the widened rule (O-1) and 9A as written differ (C, section 3d of `out-run5.txt`). The row reads "Overall length: -1 mm -> not a step: the overlapping magnet length does not change by the step; +1 mm -> not a step: the overlapping magnet length does not change by the step". Every other side equals (a)'s. |
| temperature edge | As (a), plus **Temperature limits: caution** on three sides: diameter `+1 mm`, length `+1 mm` and count `-2` (the slip-fault margins move by about 0.04-0.06 °C a step, C). At 82.12 °C (amber shown) the length `-1 mm` side flips it back ("passes"). The one probe design with a temperature flip (physics review SF-6); the G-3 sheet shows it rendered. |
| npole 40 | Count `+2`: **at the end of its range** (42 is past the slider); `-2`: +0.0110 N·m (+2.26 %) (C). |
| slider tops | Gap 5.0, `+0.1 mm`; Outside diameter 80 mm, `+1 mm`; Overall length 69.90 mm, `+1 mm`, and 21.10 mm, `-1 mm`: each **at the end of its range** (C). |
| steps to an end | (a) at Overall length 68.90 mm (a value on the slider's grid): `+1 mm` lands 1.42e-14 mm past the range's top (69.90000000000000568 against 69.89999999999999147). With the slack of 3.3 step 2 it reads **+0.2008 N·m (+2.06 %)**, equal bit for bit to `compute_all(resize_length(.., length_range.max))`; an exact range test read "at the end of its range" while the slider moves exactly 1 mm further (code review 2 MF-1; (e) the same, +0.1514 N·m). Numbers under either test (C): (a) at 22.10 mm `-1 mm` -0.2008 N·m (-59.57 %, the step lands exactly on the bottom), and its Outside diameter `+1 mm` +0.0038 N·m (+1.13 %), drawn "+0.004 N·m (+1.1 %)", the smallest number side of section 2 (T-9; C, section 3d of `out-run5.txt`); the gap at 4.90 `+0.1` -0.0213 N·m (-2.65 %) and at 0.40 `-0.1` +0.1037 N·m (+3.27 %); Outside diameter 79.0 mm `+1 mm`, +0.0198 N·m (+0.83 %) at (a) and +0.0157 N·m (+0.91 %) at (e); (e) with Minimum OD 41.80 mm, `-1 mm` -0.0063 N·m (-0.37 %), as (e)'s. Over the sliders' grids (the length's 0.1 mm at (a), the gap's 0.01 mm at (a), the diameter's 0.1 mm at (e)) the length at 68.90 mm is the one side the exact test gets wrong (C, section 3c of `out-run4.txt`). |
| count overflow | `coupling.npole` `i64::MAX` or `i64::MIN` (a file's value): `ShownDoesNotFit` (C), so no row is reached; the count step's own arithmetic is checked (`checked_add` / `checked_sub` -> `EndOfRange`) and pinned on the step function (test 3). |

## 3. The module `gui/torque_card.rs` (pure, no egui)

### 3.1 Types

```rust
/// A Size-tab control the card steps (owner 2A), in the controls' order (6A).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control { OutsideDiameter, OverallLength, FaceGap, MagnetsPerRing }

impl Control {
    pub const ALL: [Control; 4];
    /// The step of decision 3A: 1.0 mm, 1.0 mm, 0.1 mm, 2 magnets.
    pub const fn step(self) -> f64;
    /// The label drawn above the control's slider: size_controls::DIAMETER and LENGTH (no input carries
    /// those two names), label_for("metal.face_gap_mm") and label_for("coupling.npole") (owner 11A).
    pub fn label(self) -> &'static str;
    /// The Auto variable whose slider is this control: the inverse of size_controls::variable_slider
    /// (gui/size_controls.rs:61-67); None for the gap.
    pub fn held_by(self) -> Option<FreeVariable>;
}

/// Why a side shows words, not a number (owner 9A, 16A; T-4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Words { EndOfRange, NotOnTheSlider, DoesNotFit, KeywayBreaksThrough, NoPrediction, Jump }

/// A dashboard check whose level differs on the stepped design (owner 8A).
#[derive(Clone, Debug, PartialEq)]
pub struct Flip { pub check: &'static str /* the guide's label */, pub to: Level }

/// The change of a failing check's own quantity (owner 8A; section 3.6). `unit` is the result's own
/// (`result_info(path).meta.unit`: mm, µm, MPa, °C or N·m for the quoted paths), drawn through `with_unit`
/// as `result_text` does; no unit is typed.
#[derive(Clone, Debug, PartialEq)]
pub struct MarginChange { pub check: &'static str, pub name: Option<&'static str>, pub change: f64,
                          pub unit: &'static str, pub decimals: usize }

/// A side with a number.
#[derive(Clone, Debug, PartialEq)]
pub struct Change {
    pub hot_low_Nm: f64,                   // the stepped design's hot-low (tests pin it to compute_all)
    pub change_Nm: f64,                    // hot-low less the shown design's
    pub fraction: f64,                     // change_Nm over the shown hot-low
    pub calibration_changed: bool,         // 16A: model.f_cal differs
    pub diameter_change_mm: Option<f64>,   // dimensions(stepped) less dimensions(shown);
    pub length_change_mm: Option<f64>,     //   None when either `dimensions` is None
    pub flips: Vec<Flip>,                  // in CHECK_GUIDES order
    pub margins: Vec<MarginChange>,        // only those `shows` passes (section 3.6)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Side { Words(Words), Change(Box<Change>) }

/// One control's row: its `-` side, then its `+` side.
#[derive(Clone, Debug, PartialEq)]
pub struct Row { pub control: Control, pub fewer: Side, pub more: Side }

/// The card of a design. `ShownNoHotLow`: the model predicts (the readouts show both torques), but hot-low,
/// the percent's base, is not above zero (a design file's variation of 1 or more; physics review 2 MF-1), or
/// the pull-out is not above zero (a magnet Br or calibration coefficient of 0 or less; physics review 3 MF-2).
#[derive(Clone, Debug, PartialEq)]
pub enum TorqueCard { ShownDoesNotFit, ShownNoPrediction, ShownNoHotLow, Rows(Vec<Row>) }
```
`TorqueCard: Clone + PartialEq` (it sits inside `Consequences`, which derives both, `gui/guidance.rs:2803`), so every
type inside it derives `Clone, Debug, PartialEq`. No field holds a NaN for "no value" (`Option` does), so a card
equals itself and the cache and equality tests can compare cards.

### 3.2 Public functions

```rust
/// The card of `inputs`, the design shown, whose results are `results` (compute_all(inputs)), with the Size
/// tab's Minimum OD preference (`None`: unset), as the panel holds it (gui/panel.rs:309), and the shaft-clamp
/// pause (`clamps_paused`, the panel's field, gui/panel.rs:333): the flips and margins run over
/// counted_guides(clamps_paused).
pub fn card(inputs: &DesignInputs, results: &DesignResults, minimum_diameter_mm: Option<f64>,
            clamps_paused: bool) -> TorqueCard;

/// The held row's line, "{control.label()}: {SET_BY_AUTO}": `lines` draws it, and app.rs's smoke test asserts
/// it in the gui-smoke prompt (size_controls is a private module, so app.rs reaches the words only here).
pub fn held_line(control: Control) -> String;

impl TorqueCard {
    /// The lines under the heading, as drawn and as the screen dump prints them: for Rows the caption, the
    /// Auto line for `auto` (Some while Auto is on), then one line per row (the held row reads held_line;
    /// a row's two N·m at its shared decimals, T-14); for the three Shown* forms their one line alone (T-12).
    pub fn lines(&self, auto: Option<FreeVariable>) -> Vec<String>;
    /// The rows (empty for the three Shown* forms), for the tests.
    pub fn rows(&self) -> &[Row];
}
```
`Consequences::of` (`gui/guidance.rs:2825-2845`) gains `clamps_paused: bool`, calls `card(inputs, results,
minimum_diameter_mm, clamps_paused)` and stores it as `pub card: TorqueCard` in place of `neighbours` (`:2807`).
`ConsequenceCache::get(inputs, results, minimum_diameter_mm, clamps_paused)` keeps `clamps_paused` in
`CachedConsequences` and in its equality test: still one key, now in three parts, as `FixCache` keys on the pause
(`gui/panel.rs:616`). Both panel call sites pass `self.clamps_paused`. Reading the constant instead would show a paused
card on an unpaused test panel (`gui/panel.rs:1674`, `:4280`, `:4596` flip the field); reading the field without the
key would leave the card stale after a toggle. Unpaused, the clamp's guide counts and is labelled "Shaft clamp"
(`gui/guidance.rs:212`), so the card may then name it; it never does while paused (test 16 runs at the shipped pause).

`guidance::Consequences` holds a `torque_card::TorqueCard`, and `torque_card` calls `guidance`'s `check_terms`,
`level` and `counted_guides`: the two modules depend on each other, which Rust allows inside one crate. The cycle is
kept on purpose: moving `Consequences` and `ConsequenceCache` beside the controls would move shipped code and tests for
no behaviour change (code review S14).

### 3.3 How each control moves the design, and the order of the tests

The card reuses each slider's own mapping, so a side is the design the slider's mapping makes at exactly ± one step.
(A slider drag itself snaps to the slider's 0.1 mm step, `step_by`, `gui/input_ui.rs:80`: at an off-grid diameter
shown, such as Auto on ring size's 46.0084 mm, a 1 mm slider move lands on the grid while the card steps by exactly
1 mm.)

- **Outside diameter.** The slider's minimum is the expression at `gui/size_controls.rs:248-250`
  (`preference.unwrap_or(legacy.diameter_range.min.min(legacy.diameter_mm)).clamp(range)`). It moves into a
  `pub(crate) fn slider_minimum(legacy: &SizeControls, preference: Option<f64>) -> f64` in `gui/size_controls.rs`,
  called by `controls_ui` and the card (one expression). It takes `dimensions(inputs)`, which both callers already
  hold (`controls_ui` at `:243`; the card for T-5 and the length row), so it runs no geometry of its own (taking the
  inputs, it would run one more `geometry`, so one more `compute_all`, on every Size-tab frame, and the card would make
  three base geometries, not 7.1's two; code review 2 S5). The range is `dimensions_with_minimum_diameter(inputs,
  minimum)` (`engine/size_controls.rs:255-260`); where that is `None` while `dimensions` is `Some`, the card leaves the
  diameter row out, with no `unwrap` in `torque_card.rs` (the controls `.expect` it, `gui/size_controls.rs:272-273`).
  The requested value is its `diameter_mm ± 1.0` (through step 2's slack); the design is
  `resize_diameter_with_minimum(inputs, requested, minimum)` (`:405-416`), what the slider calls
  (`gui/size_controls.rs:303-304`). `resize_diameter` is not used: it ignores the preference
  (`engine/size_controls.rs:397-401`) and would put "at the end of its range" where the slider has none.
- **Overall length.** `dimensions(inputs)`'s `overall_length_mm ± 1.0` against its `length_range`, then
  `resize_length(inputs, requested)` (`engine/size_controls.rs:453-466`), what the slider calls
  (`gui/size_controls.rs:306`).
- **Magnet face gap.** `metal.face_gap_mm ± 0.1` against its slider (`InputCatalogue::get()`'s range, 0.3-5), set
  through `InputSet::set` as the control does (`gui/size_controls.rs:361`). A refused `set` reads `EndOfRange`, never
  an unwrap (`set` refuses only a value that is not finite, `engine/meta.rs:745`, which the range test already catches,
  so no design here reaches it).
- **Magnets per ring.** The count's step function in `torque_card.rs`, which test 3 calls directly:
  `pub(crate) fn count_step(npole: i64, more: bool) -> Result<i64, Words>`. It takes `checked_sub(2)` /
  `checked_add(2)`; an overflow, or a count outside the slider's 4-40, reads `EndOfRange`; an odd count inside it
  reads `NotOnTheSlider`; else the count.
  It is `neighbour_counts`' rule (even, inside 4-40, never overflowing; the shipped rule, `gui/guidance.rs:2902-2909`)
  for one side, so `neighbour_counts` moves into it rather than beside it (T-2). The design sets `coupling.npole`
  alone, as the neighbour lines did (`FreeVariable::MagnetsPerRing.apply`, `engine/sizing.rs:164-172`), keeping the
  ring radius.

**Order of the tests** (the first that holds decides):
0. **The design shown** (one line in place of the rows), by the readouts' own rule (physics review 3 MF-2): its
   blocks do not fit (`blocks_fit`, `engine/model.rs:579-581`) -> `ShownDoesNotFit`; else the end effect is out of
   range, or the pull-out or hot-low is not finite, so `size_controls::predicts` (`gui/size_controls.rs:207-209`) is
   false for one of them and its readout above the card shows `NO_PREDICTION` or `NO_TORQUE` (`torque_readout`,
   `gui/panel.rs:1544-1566`, called at `:879` and `:891`) -> `ShownNoPrediction`; else the pull-out **or** hot-low is
   **not above zero** -> `ShownNoHotLow`. In that last case
   the readouts show both torques as numbers, so "no torque prediction" would be false (physics review 2 MF-1); and
   hot-low is the percent's base. It is reached from a design file or link: `hot_low = torque_op × (1 − variation)`
   (`engine/metal_design.rs:413`), and `set()` and `validate()` check only that a number is finite (`check_value`,
   `engine/meta.rs:745-755`), so a variation of 1 or more (design "variation 1.0": hot-low 0; at 1.5, -1.3442 N·m)
   or a magnet Br or calibration coefficient of 0 or less (design "zero torque": pull-out and hot-low 0 N·m;
   `calibration.f_cal_original` -0.95: -2.6885 and -2.2852 N·m) reaches it (C). Revision 2 put a pull-out that is not
   above zero under "no prediction" (`prediction`'s rule), which contradicted the readouts at zero torque;
   `prediction` has no caller in the card and leaves with the neighbour lines (T-2).
1. **Which rows.** The two size rows only where `dimensions(inputs)` is `Some` (T-5); the gap and count rows always.
2. **On the slider.** The stepped value more than `STEP_SLACK_MM` past an end of its slider's range -> `EndOfRange`
   (the diameter's range is the slider's at the user's Minimum OD, owner 9A). A value within the slack past an end is
   taken as the end itself (`f64::clamp` into the range), and the mapping is called with that value: the mappings'
   range checks are exact (`check_request`, `engine/size_controls.rs:386`) and would refuse it. Without the slack, (a)
   at Overall length 68.90 mm, a value on the slider's grid, read "at the end of its range" on its `+1 mm` side, which
   lands 1.42e-14 mm past the range's top, while the slider moves exactly 1 mm further (code review 2 MF-1; section
   2.3 Edges, "steps to an end"). The count is an integer: `count_step` decides it (a count inside the range but odd
   -> `NotOnTheSlider`).
3. **Mapping refused.** A `resize_*` that returns `Err` for a value inside its range (a non-finite request), the
   gap's `InputSet::set` returning `Err`, or the count's checked arithmetic overflowing -> `EndOfRange`.
4. `compute_all(stepped)`; **blocks do not fit** -> `DoesNotFit`.
5. **Keyway** (diameter, length and gap only; D-1 for the count, `gui/guidance.rs:2919-2923`):
   `!(model.hub_wall_past_key_mm > 0.0)` (NaN fails) -> `KeywayBreaksThrough`. Tests 4, 5 and 6 are
   `sizing::is_valid`'s clauses in its order (`engine/sizing.rs:225-230`).
6. **No prediction**: by the readouts' own rule, the one test 0 reads (T-2; code review 4 SF-2),
   `!(predicts(results, pull-out) && predicts(results, hot-low))` (`size_controls::predicts`: the end effect in range
   and the torque finite) -> `NoPrediction`. It equals the shipped neighbour rule it replaces,
   `end_effect_in_range(f_end)` with both torques finite (`gui/guidance.rs:2933-2938`): `predicts` reads
   `end_effect_out_of_range` (`gui/dashboard.rs:619-621`), and `end_effect_check` is "OK" exactly when f_end > 0
   (`engine/model.rs:103-114`, `:1147`); the two agree at every stepped design of section 2 (313; C, section 3e of
   `out-run6.txt`).
7. **The jump** (length only): `|stepped active length - shown active length - sign × step| > STEP_SLACK_MM`
   (`model.active_length_mm`; `sign` -1 on the `-` side and +1 on the `+` side) -> `Jump`: words whenever the
   overlapping magnet length changes by anything other than the step (9A, widened by O-1, section 1.3).
   `STEP_SLACK_MM = 1e-6` mm is the one slack of steps 2 and 7, named once in `torque_card.rs` (code review 2 MF-1):
   far above the float noise seen (1.42e-14 mm past a range's end, 3.6e-15 mm in `resize_length`'s subtraction) and
   far below the sliders' 0.01 and 0.1 mm steps and any real change. For the record, 9A as written,
   `|stepped active length - shown active length| > step + STEP_SLACK_MM`, differs from this rule at every design of
   section 2 only on the short cavity's `+1 mm` side, which it showed as a number, -6.15 % for +1 mm (C, section 3d of
   `out-run5.txt`).
8. Otherwise `Change`, with the calibration rule: `stepped.model.f_cal != shown.model.f_cal` sets
   `calibration_changed` (16A). Among the four controls only the count can change it (the factor reads the back
   iron, the count and the parts, `engine/model.rs:584-600`); the test is general. A side's number shows whatever its
   sign: only the count's calibration switch can take a stepped torque to or below zero while the shown one is above
   it (a design file's `calibration.f_cal_original` below 0, or a measured torque of 0 or less), and 16A asks for the true
   `compute_all` number with `CALIBRATION_CHANGE` there (code review 4 MF-2; design "(c), `f_cal_original` -0.95",
   2.3 Edges: "-2 -> -2.29 N·m (-158.9 %), includes a calibration change"). Over a grid of 96 designs that read rows
   (4 to 12 poles, back iron 0 and 1, three part pairs (B842SH/B842SH, B442/B442, B842SH/B442), `f_cal_original` ±0.95, a measured torque of ±1.8 N·m,
   variation 0.15 and 0.5), 4 of 696 number sides have a hot-low not above zero, each a count side whose factor
   changes sign (C, section 3e of `out-run6.txt`).

Tests 4-6 are the one validity function, `side_words(design, results, keyway: bool) -> Option<Words>`, the moved
`neighbour_outcome` with D-1 as its parameter, its step 6 through `predicts`: one function for all four rows (T-2).

### 3.4 "Set by Auto": applied outside the cache

The cache key stays the design shown, the Minimum OD and the clamp pause (`ConsequenceCache`,
`gui/guidance.rs:2992-3028`, with the pause added, section 3.2); the Auto state is not in it. The card computes all
four rows whatever the Auto state; `lines(auto)` replaces the held row's text with `held_line` and adds the Auto line.

Why outside: the same design can be shown with Auto on and off (turning Auto off adopts the shown value, so the inputs
then equal the design shown, `gui/panel.rs:719-732`); a marking inside a cache keyed on the design alone would keep the
old marking after the toggle. Putting the sizing state in the key would fix that at the cost of a second key part and
a full recompute (about 1 ms native, section 7) on every Auto toggle, for two rows' worth of work saved. Outside, a
toggle costs one `lines()` call.

### 3.5 Where the panel asks the cache

- The lines under the controls keep asking as today (`gui/panel.rs:1141-1149`), with the design before this frame's
  edit, now passing `self.clamps_paused` as well.
- The card asks in the centre view with the design shown **after** this frame's edits: `self.shown_inputs()`, its own
  `compute_all` and `self.minimum_diameter_mm` as `inputs_ui` just set it (`:1163`), with `self.clamps_paused`. At the
  card `self.results` is already `compute_all(self.shown_inputs())`: the header (`:602-604`) and the inputs side run
  before `self.results = compute_all(&shown)` (`:613`), and the Required edit recomputes it (`:868`). The lag the
  comment at `:1142-1146` names (the header's assumptions reset edits the inputs after the last compute) is the inputs
  side's, not the centre's (code review 2 S6). The card still computes its own, as `inputs_ui` does: **defensive**, so
  the cache's key and the results it is handed come from one design whatever code a later change puts between `:613`
  and the card, for one `compute_all` a frame (17-35 µs, 7.1). The drawing below already uses `self.shown_inputs()`
  (`:939`).
- Per drag frame that is one miss, as today: frame N's centre computes design N; frame N+1's controls ask for design N
  and hit. Idle frames hit both times.

### 3.6 Margins and flips: every counted check

**Flips:** for each counted guide (`counted_guides(clamps_paused)`, `gui/guidance.rs:633-637`; 9 while paused), the
level on the shown design and on the stepped design (`level`, `:643-658`); a flip is a different level. Phrase
"{label}: fails" / "passes" / "caution" (section 4). Hot-low torque flips show: the row's number does not say whether
the step crosses the requirement.

**Margins:** the counted checks' terms move into one production function,
`pub(crate) fn check_terms(check: &str, inputs: &DesignInputs, results: &DesignResults) -> Vec<CheckTerm>` in
`gui/guidance.rs`, with `CheckTerm { name: String, margin: f64, passes: bool, quoted: Option<Quoted> }`. It is the
production half of the test-private `terms()` (`gui/guidance.rs:4271-4350`: margins larger-is-better, `passes` as the
verdict compares); the test's `terms()` calls it and adds the clamp's test margin (`clamp_margin`, `:4250-4266`), so the
fix tests and the card read one table (DRY). Two test-private items move with it, not beside it (code review 2 S9): the
test's `Term` (`:4240-4245`), which `CheckTerm` replaces (the fix tests read `CheckTerm`'s three shared fields), and
`joint_term` (`:4351-4359`), which names the shaft joints' terms ("inner hub slips", "cup key"). Each check's margin
is an arm of `check_terms`, as in `terms()` (each has one use, so no separate margin functions), and since
`check_terms` lives in `guidance.rs` it reads `axis_reserves` and `shaft_joints` there: neither changes visibility.
`Quoted { path: &'static str, name: Option<&'static str>, decimals: usize }` is the quantity the card quotes; its unit is the result's own (`result_info(path).meta.unit`, through
`with_unit`, as `result_text` does, `gui/guidance.rs:1767-1773`), never typed.

**When a change is drawn** (one named predicate in `torque_card.rs`, `fn shows(change: f64, from: usize) -> bool`: the
change is finite and `fixed(change, from + CHANGE_EXTRA_DECIMALS)` parses to a value that is not zero, the cap of
section 5, whose constant `CHANGE_EXTRA_DECIMALS = 2` lives in `gui/format.rs` beside `change_decimals`):
- A **size change** is drawn only on rows other than its own control's, only when it has a value, and only when
  `shows(change, 1)`. Float noise (3.6e-15 mm, section 2.2) is never drawn as "+0.0 mm".
- A **margin change** is drawn when all of these hold: the check is Bad on the shown **and** the stepped design, the
  term fails on the shown **or** the stepped design, and `shows(change of the quoted quantity, Quoted.decimals)`. This
  replaces an exact `!=` test (which would quote float noise as "+0.00 mm", and a NaN, since NaN != NaN) and the
  separate rule for an infinite change (`shows` is false for NaN and ±inf: a key with no bearing area, +inf against
  +inf, is not quoted).

Each quoted change reads "{label} {change} {unit}" when `name` is `None`, else "{label}: {name} {change} {unit}". "Or
the stepped design" catches a step that puts a second term of an already-failing check over its limit, which no flip
shows: at (b) the diameter `+1 mm` puts the Rotating diameter over the claim while the large-diameter stack already
fails (C; quoting only the shown design's failing terms printed nothing there, C first run). With `shows` in place of
`!=` every quoted change of section 2 is unchanged (C, `out-run3.txt` against `out.txt`).

| Counted check (guide label) | Margin, larger is better | Quoted quantity and unit | Phrase draft | Notes |
|---|---|---|---|---|
| Hot-low torque (`metal.hot_min_check`) | hot-low − required [N·m] | none | none | The row's own number is this quantity; it is never repeated. Its flip shows. |
| Running clearance (`metal.clearance_check`) | `metal.min_running_clearance_mm` − `metal.residual_target_mm` [mm] (`engine/metal_design.rs:471`) | `metal.min_running_clearance_mm`, mm, 2 decimals | "Running clearance -0.10 mm" | An arm of `check_terms`. The label names the quantity, as the Problem quotes it (`gui/guidance.rs:1972-1982`). |
| Cup wall (`materials.cup_wall_check`) | `materials.cup_wall_corner_mm` − `materials.backiron_thickness_needed_mm` [mm] (`engine/materials.rs:243-249`) | `materials.backiron_thickness_needed_mm`, mm, 2 | "Cup wall: wall needed +0.07 mm" | An arm of `check_terms`. The steps move the need, not the wall input. |
| Temperature limits (`temperature.summary.verdict`) | the eight `TemperatureTerm` margins (`gui/guidance.rs:1368-1382`) | none (T-13; owner question O-2, answered: quote none) | none | Its terms read the stored 3D demagnetization and slip-heating fields of the example design (the row is a `STORED_3D_ROWS` row, `gui/dashboard.rs:219-224`; `STORED_3D_NOTE`, `:230`). A step's change in them is partly a field that does not follow the step (C: the slip-fault peak margins move by -0.06 °C at (a)'s `+1 mm` diameter), so a number would teach that the fields follow the design (spec1 section 8). Its flips show; its dashboard row carries the stored-3D label. |
| Shaft joints (`shaft_joint.check`) | `JointTerms::margin(cause)` per joint and cause (`engine/shaft_joint.rs:570-577`) | per failing (cause, joint), the guide's result path (`JOINT_SIDES`, `gui/guidance.rs:299-356`): slips -> `{hub,cup}_needed_um` µm, 2; bore stress -> `{..}_bore_stress_MPa` MPa, 1; fitting -> `{..}_fit_temp_C` °C, 1; key -> `{..}_key_pressure_MPa` MPa, 1; `name` = `label_for(path)` | "Shaft joints: Inner hub smallest measured interference that holds -0.09 µm" | The existing margin, read as `terms()` reads it: `shaft_joint::joints` over `shaft_joint_links` for the margins, `shaft_joints(..).is_ok()` for judged (`:363-373`, unchanged). The quoted quantities read no stored field (only the allowed fitting temperature does, and no step moves it). The key term's margin is in mm of bearing length, and the hub follows the inner ring, so on a length step the quoted pressure can move the same way as that margin ((a) with `shaft_joint.key_contact_mm` 0.1, length -1 mm: inner hub key pressure 1529.32 -> 1549.91 MPa, margin -34.22 -> -31.34 mm; physics review 5 SF-3). No test pairs a key quote's sign with its margin. While the check cannot be made (`shaft_joints` is `Err`) every term passes, as `terms()` has it (`gui/guidance.rs:4317-4330`), so none is quoted; `check_terms` always returns the same eight terms in the same order, so two designs' lists pair up as the fix tests need (`:4268-4270`). |
| Gearbox input torque (`gearbox.cold_high_check`) | `coupling.gearbox_input_rating_Nm` − `metal.cold_high_input_Nm` [N·m at the gearbox input] | `metal.cold_high_input_Nm`, N·m, 2; `name` = `COLD_HIGH` ("cold-high", T-11) | "Gearbox input torque: cold-high +0.07 N·m" | An arm of `check_terms`. The label names the compared torque (BL-059's side rule: the input side), the name the torque (T-11). |
| Space claim (`housing.space_claim_check`) | `axis_reserves` (`gui/guidance.rs:482-489`, unchanged) per axis | the axis's dimension, named as the Problem names it: `metal.rotating_od_mm` "Rotating diameter", `metal.axial_stack_mm` "Overall length", `metal.large_dia_stack_mm` "Large-diameter stack"; mm, 2 | "Space claim: Large-diameter stack +1.00 mm" | One table of the axes' names and sizes (code review 2 S10): a module const `SPACE_CLAIM_AXIS_SIZES: [(&str, &str); 3]`, each axis's (name, dimension path) in `axis_reserves`' order, read by the Problem's axis table (`gui/guidance.rs:2601-2626`, which keeps its overshoot and claim paths) and by `check_terms`' `Quoted`, so no axis name or dimension path is typed twice. Test 9 holds `SPACE_CLAIM_AXES`, `axis_reserves` and it to one axis order. |
| End-effect model (`model.end_effect_check`) | `model.f_end` | none | none | A design shown that fails it has no numbers (test 0); a side that fails it reads "no prediction" (test 6). It can never flip on a number side either. |
| Magnets fit (`guidance.blocks_fit`) | the flat's slack per ring | none | none | As above with "magnets do not fit" (tests 0 and 4). |

`clamps.recommended` does not count while paused (`counts`, `gui/guidance.rs:627-629`), so the card never names it while
paused (`PAUSED = true`, `engine/clamps.rs:53`; the card takes the panel's `clamps_paused`, section 3.2). Unpaused, it
counts and its flip shows; `check_terms` returns no term for it (its one term is the test's `clamp_margin`, which
`terms()` adds), so no clamp quantity is quoted and no call panics. Un-pausing is outside this slice (section 10.2).

Order within a side: the number, the calibration note, the size changes (Outside diameter, then Overall length; the
row's own dimension is never repeated, section 2.2), the flips in `CHECK_GUIDES` order, the margins in `CHECK_GUIDES`
order and each check's term order.

**One change at two precisions** (physics review 2 SF-4; accepted here, G-3 decides, 10.1 step 2). A quoted Space claim
axis can state a change the same side also draws as a size change, or the row's own step:
- Where the cup sets the Outside diameter, a side draws the diameter's change at the size change's 1 decimal and again
  as the Space claim's Rotating diameter at its 2: npole 4's gap row reads "Outside diameter -0.3 mm, ..., Space claim:
  Rotating diameter -0.28 mm", and its count `+2` "Outside diameter -9.3 mm" beside "Rotating diameter -9.31 mm" (C).
- The diameter row's Space claim quote repeats the row's own step ("Space claim: Rotating diameter +1.00 mm" at (b)'s
  `+1 mm`, "-1.00 mm" / "+1.00 mm" at npole 4). "The row's own dimension is never repeated" (2.2) covers the size
  changes, not a failing check's quantity.

Each is true, and each says something the other does not: the size change says what the step does to the coupling's
size, the quote what it does to the margin of a check that already fails (8A). They are not always one number either:
the Rotating diameter is the larger of the cup and the cap (`engine/metal_design.rs:432`), while the Outside diameter
also counts the boss (`engine/size_controls.rs:62-64`); the Space claim's Overall length axis is the Overall length
itself (`metal.axial_stack_mm`, `engine/size_controls.rs:148`). So the card keeps both. If G-3 rules otherwise, the
change is local to `lines`: draw no size change whose quantity a quoted axis on the same side states, or no quote that
repeats the row's own step.

## 4. Text

Every new string is a named constant in `gui/torque_card.rs` with the draft below; the G-3 physics reviewer sets the
words (spec1:36; spec2:298-299). Numbers come from results; no literal number is typed in a text. Labels come from
`label_for` or the Size-tab label constants.

### 4.1 New strings (drafts)

| Constant | Draft | Where |
|---|---|---|
| `CARD_HEADING` | "One step of each size control" | the collapsing header (named apart from `size_controls::HEADING` and `panel::HEADING`) |
| `CAPTION` | "The change in the model's hot-low torque after one step of one control, everything else held." | first body line of the rows form (12A); every number on the card is a change (physics review MF-1) |
| `AUTO_HOLDS_LENGTH` | "Auto sets the magnet length again after a change; each row keeps the length shown." | Auto on magnet length (5A) |
| `AUTO_HOLDS_RING` | "Auto sizes the ring again after a change; each row keeps the ring radius shown." | Auto on ring size |
| `AUTO_HOLDS_COUNT` | "Auto sets the magnet count again after a change; each row keeps the count shown." | Auto on magnet count |
| `END_OF_RANGE` | "at the end of its range" | a side off its slider's range (9A's words) |
| `NOT_ON_THE_SLIDER` | "not a value the slider takes" | an odd count's sides (T-4) |
| `JUMP` | "not a step: the overlapping magnet length does not change by the step" | the length jump (3.3 step 7: 9A, widened by O-1). Cause-neutral: the old draft "both rings take one magnet length" implied the rings had two lengths before, false for the short cavity (physics review MF-2) |
| `COLD_HIGH` | "cold-high" | the gearbox term's quantity name (T-11) |
| `CALIBRATION_CHANGE` | "includes a calibration change" | after a number whose factor differs (16A's own example) |
| `SHOWN_DOES_NOT_FIT` | "No step is compared while the magnets of the design shown do not fit." | test 0; drawn alone (T-12) |
| `SHOWN_NO_PREDICTION` | "No step is compared while the design shown has no torque prediction." | test 0: the end effect out of range, or a pull-out or hot-low that is not finite (the readout shows `NO_PREDICTION` or `NO_TORQUE`); drawn alone (T-12) |
| `SHOWN_NO_HOT_LOW` | "No step is compared while the design shown has a pull-out or hot-low torque that is not above zero." | test 0: both torques shown, the pull-out or hot-low at or below zero (a design file's variation of 1 or more, or a magnet Br or calibration coefficient of 0 or less); drawn alone (T-12). Physics reviews 2 MF-1 and 3 MF-2: `SHOWN_NO_PREDICTION` would be false there, under readouts that show both torques. The draft is true in each case routed to it (revision 2's "has no hot-low torque above zero" was false where the pull-out is at or below zero while hot-low is above it, a variation above 1); G-3 sets the words |
| `FLIP_FAILS`, `FLIP_PASSES`, `FLIP_CAUTION` | "fails", "passes", "caution", drawn "{label}: {word}" | flips (8A). Not `Level::word()`'s "OK", "Check", "Fails" (`gui/dashboard.rs:79-85`): those are the badge's words, set apart on the dashboard; these are a phrase inside a sentence ("Space claim: fails") (code review 2 S14) |
| `WALL_NEEDED` | "wall needed" | the cup wall's quoted quantity |
| `KEYWAY_BREAKS_THROUGH` | "the keyway breaks through the hub" | T-4; the shipped text of `gui/panel.rs:926` ("Exploration only: the keyway breaks through the hub."), which is then built from it, `format!("{EXPLORATION_ONLY}{KEYWAY_BREAKS_THROUGH}.")`, with one private `EXPLORATION_ONLY` = "Exploration only: " in `panel.rs` that `:919` shares (4.2) |

Templates (no words of their own): a row "{label}: {fewer}; {more}"; a side "{step} -> {N·m} N·m ({percent})" (the N·m
at the row's shared decimals, T-14) then
", " and each cost; a step "{±step}{ mm}" from `Control::step` (`-1 mm`, `+0.1 mm`, `-2`); a size change "{Outside
diameter | Overall length} {change} mm" with `size_controls::DIAMETER` / `LENGTH`.

### 4.2 Reused strings (no new review needed for the words; G-3 reads them in their new place)

- `NEIGHBOUR_DOES_NOT_FIT` "magnets do not fit" and `NEIGHBOUR_NO_PREDICTION` "no prediction"
  (`gui/guidance.rs:2742`, `:2745`), moved into `torque_card.rs` as `SIDE_DOES_NOT_FIT` and `SIDE_NO_PREDICTION`
  (named apart from `size_controls::NO_PREDICTION`, "no prediction (magnets too short ...)", `:157`). The Size tab's
  exploration line at `gui/panel.rs:919` ("Exploration only: magnets do not fit at this diameter.", a literal today) is
  built from it, `format!("{EXPLORATION_ONLY}{SIDE_DOES_NOT_FIT} at this diameter.")`, as `:926` is from
  `KEYWAY_BREAKS_THROUGH` (code review 2 S14): no word changes, and the panel test that reads both lines,
  `smaller_exploration_shows_magnet_fit_and_keyway_breakthrough` (`gui/panel.rs:4896-4920`, which types both texts
  today), builds its expected texts from `EXPLORATION_ONLY`, `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` (code
  review 3 S9); a G-3 change to the side word then reaches the exploration line too (4.3).
- `SET_BY_AUTO` "Set by Auto; turn Auto off to edit" (`gui/size_controls.rs:54`) for the held row (5A's "set by
  Auto"; the row tells how to see the numbers), drawn through `held_line`.
- The Why link: `why_note_link(ui, SIZE_TAB_NOTE)` (`gui/dashboard.rs:651-653`; `SIZE_TAB_NOTE` =
  `torque_from_shear`, `gui/guidance.rs:608`): "Why: How the coupling makes torque". That note owns the area and lever
  equations; it already compares a millimetre of magnet length with a millimetre of **ring radius** on the example
  design ("on the example design each millimetre beats a millimetre of radius", `engine/explain/notes.rs:181`), and its
  watch-out covers the owner's case (e) ("T_2D grows as R_g² only at fixed σ: a larger ring with the same magnet count
  fills each pole less, so the pull-out can even fall.", `:198`). No new note (12A).
- What the note does not say (physics review SF-2; for the lessons slice, not a card text, since 12A rules out a new
  note): the card's Outside diameter step is not a millimetre of radius. The cup moves 2/cos(π/N) mm per mm of ring
  radius (2.10 at 10 magnets, 2.16 at 8), so +1 mm of Outside diameter is about 0.48 mm of radius (0.4755 mm at (a),
  0.4619 at (e), C). At (a), +1 mm of ring radius is +6.76 % (Outside diameter 42.80 -> 44.90 mm) while the card's
  +1 mm of diameter is +3.49 % and +1 mm of length +8.79 % (C). No on-screen text states the 2/cos(π/N) slope, so the
  card's 8.8 % against 3.5 % at (a) is not explained by the linked note.
- The guide labels (`CheckGuide.label`, `gui/guidance.rs:166-533`), the space-claim axis names
  (`:2603-2620`), `label_for` of the shaft-joint results.

### 4.3 Shipped strings that change words (on the STOP sheet; G-3 sets them)

- `RING_SIZE_STAYS` (`gui/guidance.rs:2717`): "...; the cup OD below can still change." points at the neighbour lines'
  "cup OD Y mm", which leave the control (7A). Draft: "The ring radius stays the same when you change the magnet
  count; the cup can still change size." It stays under the control with Auto off and with Auto on the magnet length
  (true there: that Auto moves no ring).
- `AUTO_RESIZES_RING` and `AUTO_RESETS_LENGTH` (`:2722`, `:2727`) are removed: both say "the lines below keep ...",
  and those lines move to the card. `magnet_count_line` (`:2732-2739`) returns `RING_SIZE_STAYS` for Auto off and Auto
  on magnet length, `None` for Auto on ring size or count. The card's three Auto lines take their rule (5A).
- No other shipped words change, with one condition: the Size tab's exploration lines at `gui/panel.rs:919` and `:926`
  are built from `SIDE_DOES_NOT_FIT` and `KEYWAY_BREAKS_THROUGH` (4.1, 4.2), so a G-3 change to either word changes
  those shipped lines too; the STOP sheet then lists them (code review 3 S9). The readout's gap changes its numbers
  only (section 5).

### 4.4 The card's ban list and the traps (spec1 section 8, spec2 section 8, the brief's section 6)

- **The card ban list**: `CARD_RANKING_WORDS = ["strongest", "biggest", "best", "most", "lever", "dominant",
  "ranks"]` in `engine/explain/plain.rs`, beside `TORQUE_ORDER_WORDS` (`:131-133`), where the guards live (spec2 R2).
  A word matches whole and case-insensitive, with an `s` added or taken off (so "levers" and "rank" match; "almost"
  does not). Enforced over every card line at every design of section 2 (test 16), with `TORQUE_ORDER_WORDS`,
  `plain_text_violations` (`:62-90`) and `mentions_gearbox_protection` (`:126-129`) over the same lines.

| Trap | Guarded by |
|---|---|
| "X is the strongest / biggest / best lever", any sorted claim | fixed row order (6A); the ban list over `CARD_HEADING`, `CAPTION`, the Auto lines and every row; no comparative text anywhere on the card |
| A small change as "+0 %" or "-0.00", or signs that disagree; two numbers that read equal (BL-068) | section 5's formatter on every side, size and margin number; tests 8.3 |
| Two sides of a row whose N·m misstate their ratio ("-0.01" against "-0.005" for a true 1.35:1) | a row's two N·m at one precision (T-14); test 2a |
| Extrapolating a step ("each mm adds X") | `CAPTION` ("the change ... after one step ... everything else held"); each side states its step; no per-mm text |
| A jump read as a slope | `JUMP` on every length side whose active length does not change by the step (3.3 step 7, widened by O-1), the short cavity's `+1 mm` included (under 9A as written it printed -6.2 % for +1 mm). Not guarded: the diameter row's boss plateau (R-13, on the STOP sheet) |
| "Torque grows as R_g²" without "at fixed shear stress"; "a bigger coupling gives more torque" | no physics sentence on the card; the Why link goes to the note that owns the equation; at (e) the numbers say otherwise |
| "Fewer poles reach across the gap better", "fewer poles keep the size" | the count row prints numbers and costs only (at (e): `-2` grows the Outside diameter +2.2 mm) |
| The calibration switch explained as physics | `CALIBRATION_CHANGE` names the change and no cause (16A) |
| `f_cal`, `c_end`, variation, μ0 or bench inputs as things to change; "the end factor is accurate"; "edit the calibration" | rows for the four controls only (2A); no text names an assumption |
| "Guaranteed" torque; ordering the torques | every N·m is hot-low but the gearbox phrase's, which names its own torque ("cold-high", T-11) and compares none; `TORQUE_ORDER_WORDS` over its lines |
| "More torque is safer"; "the coupling protects the gearbox" | the gearbox flip and margin print beside every torque-raising step that moves them; `mentions_gearbox_protection` over the lines |
| A row about 0 % called "no effect" or "free" | no such words; the true sign and digits; every cost listed (8A) |
| "A hotter-rated grade is free"; stored 3D fields that follow the design | no grade row (2A); no temperature margin (section 3.6) |
| An elasticity in °C | no temperature row |
| Under Auto, "the torque rises" when Auto would shorten the magnets | the Auto lines say Auto sets the size again after a change |
| "clamp" while the section is paused | no card text says it (test 16 scans for it, case-insensitive, at the shipped pause; test 7a shows the card names the clamp only when unpaused) |

## 5. Number format (decision 10B)

One formatter for the Size tab, in `gui/format.rs`:

- `fewest_decimals_apart` and `MOST_DECIMALS` move from `gui/guidance.rs:1880-1896` to `gui/format.rs`
  (`pub(crate)`), unchanged; `band_quotes` (`:1906-1926`), `window_quotes` (`:2362`) and `:2127` call them there.
- `pub(crate) const CHANGE_EXTRA_DECIMALS: usize = 2` and `pub(crate) fn change_decimals(x: f64, from: usize) ->
  usize`: the fewest decimals from `from` up to `from + CHANGE_EXTRA_DECIMALS` at which `fixed(x, d)` reads a number
  that is not zero, through `fewest_decimals_apart`; `from` when `x` is zero, not finite, or rounds to zero even at the
  cap (T-9). `fewest_decimals_apart` has no cap of its own (it searches up to `MOST_DECIMALS`, 15, and falls back to
  it, `gui/guidance.rs:1887-1896`), so `change_decimals` takes its answer and reads `from` when that answer is above
  `from + CHANGE_EXTRA_DECIMALS` (code review 2 S4); `MOST_DECIMALS` stays private to `format.rs`. The card's `shows`
  (section 3.6) reads the same constant.
- `pub(crate) fn signed_fixed(x: f64, decimals: usize) -> String`: `x` to exactly `decimals` places with its true sign
  (a zero reads "+", the house rule; a value that rounds to zero keeps its sign, "-0.00"; not finite: `non_finite_text`).
- `pub(crate) fn signed_change(x: f64, from: usize) -> String`: `signed_fixed(x, change_decimals(x, from))`.
- **A row's N·m (T-14)**: the card draws a row's two N·m at the row's decimals, the larger of the two sides'
  `change_decimals(change_Nm, 2)` (a side with words has none; one number side keeps its own), through `signed_fixed`.
  (e)'s diameter row reads "-0.006" and "-0.005" (per number "-0.01" and "-0.005"), and so does that row at (e2),
  (e2)+Auto count and (e) with the lowest Minimum OD; (e2)+Auto length's reads "-0.005" and "-0.005" (per number
  "-0.005" and "-0.01"); no other row of section 2 changes (C, section 3c of `out-run4.txt`). Percents, size changes
  and margins keep the per-number rule.
- `signed_percent` (`gui/format.rs:89-100`) becomes one decimal: `fraction × 100` with `{:+.1}` and " %", an exact
  zero of either sign reading "+0.0 %" (the house rule; `{:+.1}` alone prints "-0.0" for -0.0), and a value that
  rounds to zero keeping its true sign (code review 5 SF-4).

Where `from` is: N·m 2 (the Size tab's N·m, `fixed(x, 2)`); size changes in mm 1 (the sliders' 0.1 mm); margin
quantities by their `Quoted.decimals` (mm, µm and N·m 2; MPa and °C 1).

Every value in this table is the capped prototype's (C, section 3 of `out-run3.txt`):

| Case | `signed_change(x, 2)` | `signed_percent` |
|---|---|---|
| exact zero, either sign (`0.0`, `-0.0`) | "+0.00" (the house rule: a zero reads "+", `gui/format.rs:89-91`) | "+0.0 %" |
| a change that rounds to zero at 2 decimals | more decimals, up to 4: -0.0046 -> "-0.005"; -0.00012 -> "-0.0001" | — |
| a change that rounds to zero even at 4 decimals (-0.00004; float noise such as Auto's solve residual: (b)'s gap is 1e-10 N·m) | `from` decimals with the true sign (T-9): -0.00004 -> "-0.00", 1e-10 -> "+0.00" | -0.0004 -> "-0.0 %", +0.0004 -> "+0.0 %" (true sign kept) |
| negative zero | as exact zero: "+0.00" | "+0.0 %" |
| NaN, ±inf | `non_finite_text` ("NaN", "+inf", "-inf"; `gui/format.rs:69-77`) | "NaN %", "+inf %", "-inf %" |

The cap at `from + 2` keeps float noise off the screen: without it (b)'s gap would read "margin +0.0000000001 N·m"
(C, the first run, `out.txt`, which had no cap). It narrows 10B's "not zero" for such noise only (T-9, on the STOP
sheet). A card side cannot reach these edges with a number: a non-finite torque reads "no prediction".

**`torque_gap_text`** (`gui/size_controls.rs:179-201`): margin "margin {signed_change(gap, 2)} N·m
({signed_percent(gap / required)})"; short "short {fixed(|gap|, change_decimals(gap, 2))} N·m ({signed_percent(gap /
required)})", the percent's sign now the gap's own, so the `replacen('+', "-", 1)` re-signing (`:188-195`) goes; no
percent while the requirement is not positive (unchanged). Its doc comment (`:173-177`, "a whole percent ... (-12 %)",
"(+16 %)") changes with it to the new rule and examples.

| Design | Shipped | After (C, `out-run3.txt`) |
|---|---|---|
| (a) | short 0.21 N·m (-9 %) | short 0.21 N·m (-8.6 %) |
| (b) | margin +0.00 N·m (+0 %) | margin +0.00 N·m (+0.0 %) |
| (c) | short 1.06 N·m (-42 %) | short 1.06 N·m (-42.3 %) |
| (e) | short 0.83 N·m (-33 %) | short 0.83 N·m (-33.1 %) |
| (e2) | margin +0.17 N·m (+12 %) | margin +0.17 N·m (+11.5 %) |

Shipped tests that change, each to the value the new rule gives (C, section 3 of `out-run3.txt`):
- `the_gap_reads_short_or_a_margin_against_the_requirement` (`gui/size_controls.rs:418-437`): "short 0.29 N·m
  (-11.6 %)", "margin +0.40 N·m (+16.0 %)", "margin +0.00 N·m (+0.0 %)", "short 0.001 N·m (-0.0 %)" (was "short 0.00
  N·m (-0 %)": the new rule shows the shortfall's digit).
- `a_requirement_that_is_no_positive_torque_drops_the_percent` (`:440-455`): unchanged ("margin +2.00 N·m",
  "margin +3.00 N·m", "short 0.50 N·m").
- `a_change_reads_as_a_whole_signed_percentage_never_minus_zero` (`gui/format.rs:295-306`) is replaced by
  `a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign` (section 8).
- Panel tests that build the expected gap with `torque_gap_text` itself (`gui/panel.rs:4982`, `:5091`, `:5150`,
  `:5647`, `:5701`) pass unchanged.
- `signed_percent`'s other caller, `Neighbour::line` (`gui/guidance.rs:2777-2799`), leaves with the neighbour lines.

## 6. GUI

### 6.1 Placement

`view_ui`'s Size tab (`gui/panel.rs:914-944`): the readouts (`:915`), the exploration lines (`:916-935`), then the
card in place of the label at `:936` (removed; the dashboard and the drawing say what it said), then the drawing
(`:937-943`). A new `fn torque_card_ui(&mut self, ui, readouts)` draws it:

- An `egui::collapsing_header::CollapsingState` with id `"magcoupling_torque_card"`: its header row holds
  `CARD_HEADING` and the Why link (`why_note_link(ui, SIZE_TAB_NOTE)`; a click calls
  `readouts.open_note(SIZE_TAB_NOTE)`, as the Required row's link does, `:871-873`) in one `ui.horizontal_wrapped`
  row inside `show_header` (as the Required, Hot-low and Pull-out rows above it, `gui/panel.rs:846`, `:876`, `:889`),
  so in a narrow region the link wraps under the heading: 18 pt at 627.9 pt and 32 pt at 277.9 pt, the heights of 6.2
  (C, `measure_in`; code review 4 MF-1). `show_header` lays its row out with `ui.horizontal` (egui 0.32.3
  `containers/collapsing_header.rs:160`), where a label extends rather than wraps (`Ui::wrap_mode`, `ui.rs:686-692`):
  put straight in it, the link would end 396.0 pt (harness style) or 394.0 pt (app style) from the row's left, 118.1 /
  116.1 pt past the 277.9 pt region of a 930 x 800 window, where the card starts closed, and `confined` would clip it
  (`gui/panel.rs:1581`). Wrapped, the link's first row follows the heading and its second sits under it, ending at
  269.8 / 267.8 pt; at 425.7 pt and wider the row is one 18 pt line in both forms (C, section 8 of `out-run6.txt`).
- The heading toggles the card as the arrow does (code review 4 SF-6): it is a `Label::new(CARD_HEADING)
  .sense(Sense::click())` whose click calls the header response's `toggle()` before `.body(..)` (the closure records the
  heading's `clicked()` in a local, since `HeaderResponse` keeps the closure's return private until `.body(..)`;
  `card_choice` then takes `is_open()`; code review 5 SF-5). `CollapsingState`
  toggles only on its arrow, an untexted rect `ui.spacing().indent` wide left of the header's content
  (`collapsing_header.rs:114-135`), while every other collapsible of the calculator opens on its text: the
  `CollapsingHeader`s, which sense a click over the whole header (`collapsing_header.rs:557-573`; `gui/panel.rs:1176`,
  `:1331`, `:1365`, `:1374`, `gui/explorer.rs:888`, `gui/clamp_drawing.rs:840`), and the dashboard's caution header, a
  `CollapsingState` whose label toggles too ("a click on it or on its arrow opens or shuts the block",
  `caution_header_ui`, `gui/dashboard.rs:903-919`). The heading lays out the same with the click sense (C, section 8
  of `out-run6.txt`).
- The body: each of `card.lines(auto)` as one wrapped label (`egui::Label::new(..).wrap()`), weak like the lines under
  the controls, inside a vertical `ScrollArea` (6.2).
- `auto` is `Some(self.sizing.variable)` while `self.sizing.mode == SizingMode::TorqueToMagnets` (`:1134-1135`).
- `gui/size_controls.rs:330-371`: `magnet_lines` now gives `RING_SIZE_STAYS` (Auto off or on length) and the fit line
  (7A); `Consequences::magnet_lines` (`gui/guidance.rs:2851-2859`) drops the neighbour lines; "Ring size stays the
  same..." and the smallest-OD line stay under the control.

### 6.2 Open or closed

Panel fields (session view state, like `welcome_dismissed`, `gui/panel.rs:297-299`: in no file, link or undo step,
kept by Reset all and every load):
- `card_choice: Option<bool>`: the user's own open or close, set when the header's arrow or its heading is clicked
  (6.1); it wins for the session.
- `card_start: Option<(CardView, bool)>`: the default and the view it was decided for. `CardView { equation_open,
  screen_width, screen_height }`: whether the Equation panel is open, and the screen's width and height in whole
  points (`ctx.screen_rect()`) (code review 3 MF-2). No design change moves it.
  - Not the centre region's width. egui keeps a side panel at its widest drawn content for the rest of the session:
    `SidePanel` stores its drawn rect, overflowing content included, and the next frame starts from that width (egui
    0.32.3 `containers/panel.rs:250`, `:298`, `:343`). So a design whose dashboard or inputs content is wider narrows
    the region for good: in one session at 1280 x 800, (c) (no back iron) takes it from 627.9 to 625.7 pt, and back at
    (a) it stays 625.7 pt; at 1080 x 800, 427.9 -> 425.7 pt (428.0 -> 425.7 in the harness's style) (C, section 7 of
    `out-run5.txt`). A key on it would decide again on that design change, and at 1080 x 800 close an open card: rule
    3 at (c) leaves 59.1 pt less than the drawing's floor in the app's style, 28.1 in the harness's.
  - Not the region's height: the header above the region (`TopBottomPanel::top`, `:602-604`) gains or loses lines on
    a design change (the assumptions banner while an assumption differs from its default, `:1094-1113`; the error or
    the status line, `:1114-1118`), so a key on it would decide again after an assumption edit under Detailed inputs
    or a drag that clears an earlier error (code review 2 S1).
  - The header's lines, the exploration lines (`:916-935`: "Exploration only: magnets do not fit at this diameter.",
    the keyway line, the "Wall past keyway" warning), the end-effect banner (`:827`), a readout that wraps and a side
    panel's content move the card's place or the region, not the screen (code review M3).

Rule, while `card_choice` is `None`: when the `CardView` differs from `card_start`'s (first frame, the Equation panel
opened or closed, the window resized), the default is decided again. A side panel's resize does not decide again: a
panel also widens by itself to fit content, so its width cannot tell a user's drag from a design change. After a user
drags a panel's edge the card keeps its state; the body cap keeps the drawing's floor, and a header click still
decides. **Open** iff all three hold on the deciding frame:
1. the Equation panel is closed (`self.explorer.open`, `:792`);
2. the centre region is at least `CARD_OPEN_MIN_WIDTH` = 400 pt wide (T-10, 4A's "the window is narrow"), its width
   read at the top of `centre_ui` from `ui.available_rect_before_wrap()` before the Equation panel is added
   (`gui/panel.rs:788-793`), on the deciding frame only. On a fresh panel's first frame the width sits at most a side
   panel's growth above the settled one (628.0 then 627.9 pt at (a) and 628.0 then 625.7 pt at (c), 1280 x 800; 428.0
   then 425.7 pt at (c), 1080 x 745; C, section 7 of `out-run5.txt`), which can change rule 2's answer only within
   2.3 pt of 400 pt;
3. the open card fits above the drawing's floor: `available_height - open_height - item_spacing.y >= drawing_floor`,
   with `available_height` measured at the card on that frame (4A's "the drawing keeps its 160 pt floor").

A design change never changes the `CardView`, so it never re-decides: the card does not open or close under a drag,
even one that crosses the fit edge and adds the "Exploration only" line, or a design change whose content widens a
side panel (test 23).

- `open_height` is measured before drawing, by one free function in `gui/panel.rs`, `card_open_height(ui: &egui::Ui,
  lines: &[String]) -> f32`, which the panel calls on a deciding frame and test 25 calls in a scratch frame of the
  same width and style (code review 4 SF-5): the header row laid out as it is drawn (wrapped, 6.1: the link's first
  row starts after the heading, as a label in a wrapping row is laid out, egui 0.32.3 `widgets/label.rs:192-213`;
  code review 4 MF-1) plus each line laid out as a galley at the body's wrap width (the available width less
  `ui.spacing().indent`), with the item spacing. Test 25 pins it to the drawn height where the body is under its cap,
  at 930 x 1024, where the header wraps, and at 1280 x 800, where it does not, so it holds both layouts; and it pins a
  capped body to the cap.
- `drawing_floor` is a new `pub(crate) fn drawing_floor(ui: &egui::Ui) -> f32` in `gui/geometry_view.rs`:
  `MIN_DRAWING_HEIGHT + MIN_LIST_ROWS × row` (`:33`, `:36`), with `row` the expression `geometry_ui` uses (`:166`),
  lifted into one `fn list_row(ui)` that both read. It is the room `drawing_height` (`:221-227`) needs to keep the
  drawing at 160 pt with three list rows: 215.1 pt at the app's spacing (row 18.38 pt, C, `layout-run3.txt`).
- **The body is capped**: its `ScrollArea` has `max_height = max(available_height - header - 2 × item_spacing.y -
  drawing_floor, one body row)`. A card that grows after the start (a design with more costs, or opened by the user in
  a narrow window) scrolls inside itself, and the drawing keeps its floor at every design.
- Clicking the card's own Why link opens the Equation panel, so with no user choice the card closes (4A). The user's
  click on the heading or its arrow then decides. The STOP sheet says so (section 10.1).

Measured (C). The open and closed heights: egui 0.32 headless, the app's spacing 6 x 4, the header row wrapped as 6.1
draws it, at the centre region's widths as `layout.rs` measures them in the shipped panel (627.9 pt at 1280 x 800,
277.9 pt at 930 x 800; the first draft used 628 and ctx-gui's 290 pt):

| Width | (a) open | (b) open | (c) open | (e) open | one-line form (short magnets) | closed | the label it replaces |
|---|---|---|---|---|---|---|---|
| 627.9 pt | 178 pt | 196 pt | 262 pt | 178 pt | 36 pt | 18 pt | 28 pt |
| 277.9 pt | 360 pt | 406 pt | 542 pt | 360 pt | 64 pt | 32 pt | 42 pt |

The room at the card (`layout.rs`, the shipped panel at 69fa505, default design, Equation panel closed: the label's top
to the bottom of the centre view's region): 588 pt at 1280 x 800 with the welcome strip shown, 610 pt with it hidden;
468 / 504 pt at 930 x 800. Against the rule (code review M7, restating R-2):
- **1280 x 800, strip shown**: rule 3 allows an open card up to 588 − 4 − 215.1 = 368.9 pt. (a) and (e) start open with
  190.9 pt to spare, (b) with 172.9, (c) with 106.9. The drawing then takes what the callout list leaves
  (`drawing_height`): at (a), with its 8-row list (220.6 pt), about 588 − 178 − 4 − 220.6 = 185 pt, down from 335.5
  pt. The list keeps every row while the card is at most 588 − 4 − 160 − 220.6 = 203.4 pt: (a) has 25.4 pt to spare,
  and (b), with a list of the same 8 rows (not measured), 7.4 pt; a taller card holds the drawing at 160 pt and the
  list scrolls.
- **930 x 800**: the region is 277.9 pt wide, under `CARD_OPEN_MIN_WIDTH`, so the card starts closed at every design
  (rule 2). Rule 3 alone would close (a) (468 − 360 − 4 = 104 pt left, under 215.1) but open the one-line form (64 pt),
  which is why rule 2 exists. Closed, its 32 pt header is 10 pt shorter than the 42 pt label, so the drawing gains
  10 pt (201.5 -> 211.5 pt).
- **Equation panel open**: closed (rule 1); the drawing is already at its floor there (ctx-gui 4).
- **Rule 3 alone** (rules 1 and 2 passing): in the windows measured it closes the card only where the region is
  little over 400 pt wide and the card is tall. At 1080 x 745 (region 425.7 pt) (c)'s open card (360 pt) leaves
  88.1 pt less than the drawing's floor, while the one-line form (36 pt) leaves 171.9 pt over it; at 1280 x 800 even
  (c) (262 pt) starts open with 106.9 pt to spare (C, section 6 of `out-run4.txt`). A shorter window does not help:
  below about 745 pt of window the panel's content is taller than the window (pre-existing at 69fa505), the outer
  panel's rect stays 745 pt tall and the centre region ends at 729 pt (C, `layoutdbg.rs`), so the room at the card is
  533 pt at 1280 x 600 and at 1280 x 540 alike. Test 20a uses 1080 x 745.

The headless tests draw in egui's default style, not the app's: `Harness::of` makes an `egui::Context::default()`
(`gui/panel.rs:1673-1682`) and the app's spacing is `app::theme::apply`'s (`src/app/theme.rs:51-58`, behind the `app`
feature). Its figures differ (item spacing 8 x 3, button padding 4 x 1): at 1280 x 800 the room at the card is 608 pt
and the floor 212.1 pt (row 17.38 pt), at 930 x 800 the drawing at 69fa505 is 246.5 pt, not 201.5 (C,
`layout-run4.txt`). So the layout tests (section 8.5) assert outcomes chosen with margin in both styles and pin no
figure measured in the app's style (code review 2 S12, MF-5).

### 6.3 Shipped tests, strings, docs and smoke items that change

- **Tests that read the neighbour lines** (they move to `torque_card.rs` as the card's tests, rewritten to the card's
  types and hot-low, or change as listed):
  - `gui/guidance.rs`: `each_neighbour_is_the_design_with_only_the_magnet_count_changed` (`:8792`),
    `the_magnet_lines_name_each_neighbour_s_torque_and_cup_or_say_why_not` (`:8842`),
    `at_either_end_of_the_magnet_slider_only_the_neighbour_inside_it_is_shown` (`:8864`),
    `neighbours_are_even_counts_inside_the_slider_and_never_overflow` (`:8892`),
    `a_keyway_through_the_hub_changes_no_neighbour_s_reading` (`:8914`),
    `with_hub_wall_past_the_keyway_a_neighbour_reads_a_torque_exactly_when_sizing_counts_it` (`:8934`),
    `a_neighbour_without_a_prediction_reads_words_and_no_change_is_invented` (`:8958`),
    `with_magnets_too_short_for_the_model_every_neighbour_reads_words` (`:9004`) -> the count row's tests;
    `the_cache_computes_once_per_design_and_minimum_and_follows_an_edit` (`:9316`) reads `card` in place of
    `neighbours`; `without_a_breakdown_or_a_fit_there_is_no_line_for_it` (`:9230-9241`), which builds
    `Consequences { neighbours: Vec::new(), .. }`, takes `card: TorqueCard::Rows(Vec::new())` and keeps its assertions
    (code review 4 SF-3); `the_first_magnet_line_follows_the_auto_state` (`:9272`) and
    `no_guide_text_calls_the_ring_radius_ring_size` (`:6068`, which lists `AUTO_RESIZES_RING` and `AUTO_RESETS_LENGTH`)
    follow section 4.3; the line counts of
    `the_fit_line_names_a_diameter_that_fits_and_one_step_below_does_not` (`:9024`),
    `the_fit_line_s_edge_forms_name_the_ends_of_the_diameter_range` (`:9063`),
    `with_minimum_od_unset_the_fit_line_says_how_far_it_could_go_and_what_unlocks_it` (`:9118`),
    `at_8_magnets_the_ring_size_stays_and_the_cap_sets_the_diameter` (`:9199`),
    `every_consequence_line_has_its_glyphs_and_plain_words` (`:9244`) and the review dump
    `guidance_strings_for_review` (`:9570`, which gains the card's strings at the section 2 designs).
    `Consequences::of` and `ConsequenceCache::get` gain `clamps_paused` (3.2): the helper `consequences_of`
    (`:8754-8756`, 13 callers) passes `clamps::PAUSED`, the shipped state; the cache test (`:9316`) passes it to every
    `get` and `of` (`:9319-9355`) (code review 5 SF-2).
  - `gui/panel.rs`: `size_tab_draws_each_what_moves_what_line_under_its_control` (`:5258`),
    `size_tab_owner_case_at_8_magnets_says_the_ring_stays_and_the_cap_sets_the_diameter` (`:5412`),
    `size_tab_first_magnet_line_follows_the_auto_state` (`:5462`), `size_tab_what_moves_what_lines_follow_each_edit`
    (`:5508`), `size_tab_smallest_diameter_line_searches_with_the_minimum_od` (`:5547`),
    `size_tab_magnet_slider_ends_show_only_the_neighbour_in_range` (`:5582`),
    `size_tab_narrow_window_wraps_the_new_lines_inside_their_sides` (`:5606`) and
    `size_tab_new_texts_have_glyphs_and_plain_words` (`:5662`). The helper `shown_consequences` (`:5002-5008`, 11
    callers) passes `harness.panel.clamps_paused`, as the panel does (code review 5 SF-2).
- **Imports that go with the change** (code review 4 SF-3; the compiler and clippy's `-D warnings` catch all three): the
  test module's `use` of `AUTO_RESETS_LENGTH, AUTO_RESIZES_RING` (`gui/panel.rs:1604`), whose constants 4.3 removes,
  and `gui/guidance.rs:65`'s `signed_percent`, unused there once `Neighbour::line` (its one use, `:2786`) leaves, and
  `gui/guidance.rs:51`'s `end_effect_in_range`, whose only non-test uses, `prediction` (`:2915`) and `neighbour_outcome`
  (`:2933`), leave too (its test uses at `:8820`, `:8964` and `:9014` move with the neighbour tests; code review 5 SF-1).
- **Counts that move** (the lines under Magnets per ring): Auto off, at (a) and (e), 4 (first line, two neighbours,
  fit) -> 2; at a slider end 3 -> 2; Auto on magnet length 4 -> 2 (`RING_SIZE_STAYS` in place of
  `AUTO_RESETS_LENGTH`); Auto on ring size 4 -> 1 (the fit line); Auto on magnet count 3 -> 1.
- **Tests that count the Size tab's Why links** (code review 2 MF-3). The card's heading draws
  `why_note_link(ui, SIZE_TAB_NOTE)` in every form, open or closed (6.1), so the Size tab gains a third link to
  `torque_from_shear`:
  - `the_size_tab_s_why_links_open_their_notes` (`gui/panel.rs:8624`) asserts 2 links per note (`:8640`) and clicks
    each. It becomes a count per note: 3 to `SIZE_TAB_NOTE` (under Magnets per ring, under Outside diameter, the card's
    heading) and 2 to `TORQUE_BAND_NOTE` (the Required and Hot-low rows, unchanged), each clicked; the card's opens
    `torque_from_shear` on its own in the Equation panel (which closes the card: no user choice, 6.2).
  - `without_the_diameter_control_the_size_tab_still_links_from_magnets_per_ring` (`:8650`) asserts 1 link at
    `with_poles(7)` (`:8664`); it reads 2 (Magnets per ring and the card's heading, whose gap and count rows the odd
    count keeps, T-5).
- **The exploration-line test** (code review 3 S9): `smaller_exploration_shows_magnet_fit_and_keyway_breakthrough`
  (`gui/panel.rs:4896-4920`) builds its two expected texts from `EXPLORATION_ONLY`, `SIDE_DOES_NOT_FIT` and
  `KEYWAY_BREAKS_THROUGH` in place of the typed literals (4.2), as the plan contract asks tests to prefer constants.
- **Strings**: the label at `gui/panel.rs:936` (no test reads it); section 4.3; the gap numbers (section 5).
- **Docs**: section 9.
- **gui-smoke** (`.claude/workflows/gui-smoke.js`): `first_view` (`:91`) gains "between the torque readouts and the
  drawing, the heading 'One step of each size control' with its Why link, open"; a new card step, the frame-time step
  and schema fields (sections 7.2 and 8.6, test 26). The card step has a fixed place (code review 2 MF-4): on the page
  the frame-time step opens (7.2 step 1: the init script, the share link again, 8 s), which is the Size tab with the
  Equation panel closed and Auto on the magnet length solved; one screenshot there for the card step, then a fresh one
  to find the gap slider for the drag. It cannot run after the existing steps, and the prompt says why: the design file
  loaded there has no sizing (Auto off: no held row, no Auto line; `:52`, `:92-94`), the dashboard Why link opens the
  Equation panel (`:95`), which closes the card, and the tooltip step leaves the Geometry tab selected (`:96`), where no
  card is drawn. The existing checks end with "Collect the console messages at level "info" once more, then close the
  browser" (`:98`); that sentence moves after the card and frame-time steps, so the browser stays open for them and
  the last console collection covers the drag (code review 3 S8). `app.rs`'s `the_smoke_test_share_link_opens_its_design`
  (`app.rs:211-284`) asserts the prompt quotes the card's words whole (test 26: `CARD_HEADING`, `CAPTION`,
  `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)`; `torque_card` is a `pub` module), so the R14 rule
  (`app.rs:251-253`) covers them after G-3 sets the words. The meta description (`:3`) and the calculator step's
  comment (`:77-84`) gain the card step and the frame timing.
- **Screen dump**: section 8.6, test 27.

## 7. Performance (decision 13A)

### 7.1 Expected cost (C, native release, this dev box; min / median of five runs of a mean of 200)

| Work per changed design | (a) |
|---|---|
| `compute_all` | 0.017 ms |
| `dimensions` (one Size-tab geometry) | 0.058 / 0.059 ms |
| `Consequences::of` today (two neighbours and the fit search) | 0.728 / 0.731 ms |
| the card: 8 stepped designs (the base geometry twice, 4 resizes, 8 `compute_all`, 8 `dimensions`, levels and margins: 22 `compute_all` in all, since each `resize_*` and each `dimensions` runs `geometry`, which runs `compute_all` once, `engine/size_controls.rs:139`) | 1.011 / 1.036 ms |
| the card with each stepped size read from its own results (the held-back optimization) | 0.502 / 0.510 ms |

The card leaves `Consequences::of` its fit search and breakdowns and drops the two neighbour runs (about 0.04 ms). A
drag frame's work grows from today's measured 1.42 ms (ctx-gui 6) by about 1.0 ms: about 2.4 ms native. Later runs on
this machine slowed to about twice these figures (`compute_all` 0.034 ms; C at (b), (c), (e)), so the native figure is
1.8-3.5 ms. No browser figure exists: node V8 ran `compute_all` within 1.1x of native (ctx-engine 6).

### 7.2 The browser frame-time check (before release)

The gui-smoke calculator step, after its existing checks (the card step runs on this step's page, before its drag)
and before the console collection and the browser's close that end those checks today (`gui-smoke.js:98`, moved after
this step; code review 3 S8), on the local server and the share link (design (b), Auto on length) at 1280 x 800, with
the Playwright MCP tools the workflow already loads (`browser_run_code_unsafe`):
1. An init script times every `requestAnimationFrame` callback (eframe's web runner paints in one per frame):
   `page.addInitScript(() => { window.__frameMs = []; const raf = window.requestAnimationFrame.bind(window);
   window.requestAnimationFrame = cb => raf(t => { const s = performance.now(); cb(t); window.__frameMs.push(performance.now() - s); }); })`,
   then the page is opened again with the share link and waited for 8 s. This page is where the card step runs (test
   26; code review 2 MF-4): the Size tab, the Equation panel closed, Auto on the magnet length solved. One screenshot
   for the card step.
2. In a fresh screenshot, find the Magnet face gap slider's handle (left side, at 1.50 mm; not locked under Auto on
   length). Then **one** `browser_run_code_unsafe` call does the drag and the reading: it sets `window.__frameMs = []`
   just before `page.mouse.down` on the handle, moves the pointer 60 times, 50 ms apart, alternately to 3 CSS px right
   and 3 CSS px left of where it pressed, releases it (`page.mouse.up`), and reads `window.__frameMs` right after, so
   only the drag's frames are counted, not the 8 s load, the solve or the screenshots (code review S7). The rail is
   egui's default 100 pt (`slider_width`; the crate sets none) less the handle's radius each end (18 / 2.5 pt), about
   86 pt for 0.3-5.0 mm, about 0.055 mm a point: each move crosses 0.16 mm or more (0.33 mm after the first), many of
   the gap's 0.01 mm steps, and stays on the rail (the handle starts about 22 pt from its left end), so every move
   changes the design and the frame that draws it misses the cache. (Revision 1's 60 moves of 2 px one way reached
   5.0 mm after about 32 and changed nothing after; code review 2 S2.) Each edit also asks for one more frame
   (`gui/panel.rs:1157-1158`), which hits the cache, so about half the frames are misses and half are hits: the median
   mixes the two, and the p95 reads the misses. A screenshot after the release shows the gap moved off 1.50 mm (`drag_moved`: the drag held the handle).
   The notes give the number of `magcoupling sizing:` console lines logged during the drag (expected none: moves
   50 ms apart never let the 0.25 s debounce elapse; a frame that ran a solve is not a drag frame).
3. Fields: `drag_moved`, `drag_frames` (count), `drag_frame_ms_median`, `drag_frame_ms_p95`, `drag_frame_ms_max`, and
   `drag_frames_within_60fps` = `drag_moved && drag_frames >= 30 && p95 <= 16.7`. If fewer than 30 frames were
   captured, the hook missed eframe's loop: the field is false and the notes say so.
4. The fields are recorded on every run. `passed` requires `drag_frames_within_60fps` only when the workflow is run
   with `frame_check: true` (`ARGS.frame_check`, the release run of section 10.1 step 5), so a machine's speed never
   fails an ordinary smoke run or the post-deploy run on colesorkness.com (code review S6).

**Threshold:** the 95th percentile of the drag frames at or under 16.7 ms (one 60 fps frame). Missed: take the
base-geometry reuse first (each stepped design's Outside diameter as `max(model.cup_od_mm, metal.cap_od_mm,
metal.boss_od_mm)` and its length `metal.axial_stack_mm`, the expressions `geometry` uses,
`engine/size_controls.rs:140-148`, made one `pub fn` there; C: halves the card), measure again, and stop for the owner
if it still misses.

**Docs line:** `docs/ai/02-system.yaml:116`'s "(both together measured at under 1.6 ms per changed design, native
release, so a drag frame stays within budget)" becomes the measured figures: the native per-changed-design time of the
two caches with the card and the browser's drag-frame median and p95, with the date and the browser. `:106`'s "The
consequence lines recompute on every drag frame (about 30 compute_all calls)" becomes the new count: the card's 22
(section 7.1) plus the fit search's and the two breakdowns' (one `geometry` each), plus the panel's own `compute_all`
for the card (section 3.5).

## 8. Tests (decision 14A; red first: each fails at 69fa505 or before its change, and passes after)

Engine-side helpers get their tests where they live; the card's own tests sit in `gui/torque_card.rs`'s test module;
panel tests in `gui/panel.rs`. `Harness::on_screen` sizes as named (`SCREEN` is 1280 x 1024,
`gui/test_support.rs:203`). New shared test designs go in `gui/test_support.rs` (`mixed_lengths()`,
`reverse_lengths()` (both apart from the existing `mixed_rings`, which mixes grades, `:65`), `no_back_iron()`,
`end_effect_at_a_step()`, `odd_count()`, `keyway_through()`, `short_cavity()`, `temperature_edge()`,
`forty_narrow_magnets()`, `zero_torque()`, `with_variation(variation: f64)` (as `with_poles`; variation 1.0 and 1.5)
and `no_back_iron_negative_calibration()` (design "(c), `f_cal_original` -0.95"); code review 4 SF-4), with the
existing `short_magnets`, `with_poles`, `passing_design`, `only_the_joint_fails`, `only_the_clamp_fails`,
`clamp_at_its_limit`, `hot_day`. Each new design's doc comment states its section 2.1 fact, and test 1a pins it. Tests
15, 16 and 17 loop over one table in the card's test module, `card_designs()`: every design of section 2.1 with its
Minimum OD preference and Auto state (an Auto design solved as the panel solves it, through `gui::sizing::sized`), so a
design added once reaches all three (code review 4 SF-4). The screen dump (test 27), a separate tool that cannot read a
test module, keeps its own list of the same keys. The layout tests read the drawing's height through one new
probe, `test_support::drawing_rect(output)`: the painted rect with the extreme background colour and 4 pt corners
that `geometry_ui` fills (`gui/geometry_view.rs:182`), the shape ctx-gui's measuring tool and `layout.rs` read.
`view_ui` drops `geometry_ui`'s `GeometryLayout` (`gui/panel.rs:937-943`), and no production field is added for a
test (code review S11). The headless harness draws in egui's default style, not the app's (6.2): the layout tests pin
outcomes chosen with margin in both styles, never a height measured in the app's style.

### 8.1 Rows pinned to `compute_all`

1. `each_number_is_the_difference_of_two_compute_all_runs`: at (a), (b), (c), (e), (e2), the mixed and reverse pairs,
   every Auto design of 2.3, end effect at a step, the odd count, (e) with the lowest Minimum OD, (a) with Minimum OD
   42.8, the short cavity, the temperature edge, npole 40 and (c) with `f_cal_original` -0.95 (code review 4 MF-2):
   for every `Change` side, rebuild the stepped design by the slider's mapping (written in the test, not through the
   card), and assert `change_Nm == compute_all(stepped) - compute_all(shown)` bit for bit, `fraction`, the size changes
   against `dimensions` (`None` exactly when either is `None`), and `calibration_changed` against `model.f_cal`.
1a. `the_shared_card_designs_read_as_section_2_1_says` (code review 4 SF-4): each new design of section 8's list
   reads its 2.1 fact, so a changed default that moves one fails here first, not inside a card test: among them
   `zero_torque()` pull-out and hot-low 0 N·m with the end effect in range; `with_variation(1.0)` hot-low 0 with the
   pull-out 2.6885 N·m; `no_back_iron_negative_calibration()` the measured factor 1.0495 and `no_back_iron()`'s
   hot-low on the design shown, and the factor -0.95 on its count `-2` design; `short_cavity()` Overall length 27.30 mm;
   `temperature_edge()` Temperature limits green; `odd_count()` `dimensions` `None`.
2. `the_card_reads_the_spec_s_numbers`: the section 2.3 values to 1e-4 N·m, 0.01 % and 1e-4 mm: (a)'s four rows;
   (b)'s; (c)'s count side -0.5923 N·m (-41.05 %) with the calibration change; (e)'s diameter row -0.0063 / -0.0046
   N·m (-0.37 / -0.28 %) and the count row's Outside diameter +2.2307 mm (2.3's +2.23 mm; +2.230655, C, section 3e of
   `out-run6.txt`; physics review 4 SF-5); the mixed pair's two jumps, its diameter `+1` +0.0300 N·m, and the reverse
   pair's ±19.89 %. It pins no number for the short cavity's length row, whose two sides read jump words (test 3).
2a. `the_card_s_lines_read_as_drafted` (code review 2 MF-5): `lines(..)` equals whole expected lines, each built with
   `format!` from the constants, `Control::label`, the step texts and `signed_fixed` / `signed_percent` of the pinned
   values at the row's decimals (T-14), so a G-3 word change flows through: (a)'s four rows; (e)'s diameter row
   ("-0.006" and "-0.005", T-14) and count row ("Outside diameter +2.2 mm"); (c)'s count `-2` side, with
   `CALIBRATION_CHANGE` right after the number; the mixed pair's and the short cavity's length rows (`JUMP` both
   sides; code review 3 MF-1); the odd count's two
   rows (the gap's numbers; `NOT_ON_THE_SLIDER` both sides of the count); (b)'s `lines(Some(AxialLength))`: `CAPTION`,
   `AUTO_HOLDS_LENGTH`, then the held row in its place. It pins what tests 1-7 do not read (they assert the fields;
   test 15 parses only the N·m and the percent): the order within a side (3.6: the number, the calibration note, the
   size changes, the flips, the margins), the templates of 4.1 (`"{label}: {fewer}; {more}"`,
   `"{step} -> {N·m} N·m ({percent})"`, `", "` per cost, the step texts `-1 mm`, `+0.1 mm`, `-2`) and the flip and
   margin phrases. The drafted blocks of 2.3 are its expected values.
3. `the_words_follow_the_order_of_the_tests` (section 3.3), one design per word: (a) diameter `-1` and count `+2`
   DoesNotFit; (a) with Minimum OD 42.8, diameter `-1` EndOfRange (and DoesNotFit at the default preference); gap at
   0.3, gap `-0.1` EndOfRange; npole 4, count `-2` EndOfRange; npole 40, count `+2` EndOfRange; the slider tops (gap
   5.0 `+0.1`, Outside diameter 80 mm `+1`, Overall length 69.90 mm `+1` and 21.10 mm `-1`) EndOfRange; the odd
   count's two NotOnTheSlider; keyway through, length and gap all KeywayBreaksThrough with the count `-2` a number
   (D-1); end effect at a step, length `-1` and count `-2` NoPrediction; the mixed pair's Jump both ways and not the
   reverse pair's (its active length moves by exactly the step); the short cavity's `-1 mm` and `+1 mm` both Jump (red
   first: 9A as written reads the `+1 mm` side as a number), and so do the `+1 mm` sides of `short_cavity()` with
   `metal.cup_depth_mm` 12.0 and 12.5 (active length +0.30 / +0.80 mm), and so does the `+1 mm` side at 10.7 mm
   (active length -1.00 mm: an unsigned comparison of |Δactive| with the step, and 9A as written, read it as a number,
   -0.20 N·m, -8.79 % for +1 mm; C, section 3e of `out-run6.txt`; physics review 4 SF-1), while at 12.7 and 15.5 mm
   both sides stay numbers (exactly ±1 mm; C, section 3d of `out-run5.txt`). (c) with `f_cal_original` -0.95: its count
   `-2` side is a `Change` with `calibration_changed` and a hot-low below zero, equal to `compute_all` bit for bit
   (code review 4 MF-2). (c) with `coupling.npole` 8 and `calibration.measured_torque_Nm` 0: its count `+2` side is a
   `Change` with `calibration_changed` and a hot-low of exactly 0, drawn "-0.85 N·m (-100.0 %)" (physics review 5
   SF-2). (a) with Minimum OD 45.0 mm (above the diameter shown): Outside diameter `-1` and `+1` both EndOfRange;
   with Minimum OD 43.3 mm: `-1` EndOfRange and `+1` a `Change` equal to (a)'s (code review 5 SF-3). The count step function, `count_step`, called directly, reads
   EndOfRange at `i64::MAX` and `i64::MIN` (checked arithmetic; `card()` reads `ShownDoesNotFit` there, C), as the
   moved `neighbours_are_even_counts_inside_the_slider_and_never_overflow` keeps for `count_step`. **A step that
   lands on an end** (the slack of 3.3 step 2; code review 2 MF-1; designs "steps to an end"): (a) at Overall length
   68.90 mm, `+1 mm` is a `Change` equal to `compute_all(resize_length(.., length_range.max))` bit for bit (red first:
   the exact test reads EndOfRange there, C); (a) at 22.10 mm, `-1 mm` likewise at `length_range.min`; the gap at 4.90
   `+0.1` and at 0.40 `-0.1`: numbers; Outside diameter 79.0 mm `+1 mm` at (a) and (e): numbers; (e) with Minimum OD
   41.80 mm (the diameter shown less 1 mm): `-1 mm` a number, equal to (e)'s at the default preference.
4. `the_design_shown_decides_first`: short magnets -> `ShownNoPrediction`; variation 1.0 and 1.5 -> `ShownNoHotLow`
   (the model predicts: the pull-out is finite and positive, physics review 2 MF-1); zero torque -> `ShownNoHotLow`
   (red first: revision 2's rule reads `ShownNoPrediction`; physics review 3 MF-2), and so do (a) with
   `calibration.f_cal_original` -0.95 (both torques below zero) and with -0.95 and `metal.variation` 1.5 (the pull-out
   below zero, hot-low above it: the case a rule on hot-low alone would let through to the rows); shown does not fit ->
   `ShownDoesNotFit`; each draws its one line alone, with no caption and no Auto line, for `lines(None)` and for every
   `lines(Some(v))` (T-12), and no row.
5. `the_size_rows_follow_the_size_controls`: the odd count has no Outside diameter or Overall length row; every
   other design has four rows in `Control::ALL` order.
6. `the_diameter_row_steps_as_the_slider_does_at_the_user_s_minimum`: at (e) with the lowest Minimum OD and with
   Minimum OD 42.8 the stepped ring equals `resize_diameter_with_minimum(.., slider_minimum(&dimensions(..), ..))`'s;
   `slider_minimum` equals the controls' own minimum at every design (it is their expression).
7. `the_flips_and_the_margins_name_the_checks_that_move`: (a) diameter `+1` flips Space claim only and quotes Running
   clearance +0.0125 mm, wall needed +0.0651 mm and Gearbox input torque, cold-high, +0.0281 N·m (its line reads
   "Gearbox input torque: cold-high +0.03 N·m"); (a) gap `-0.1` quotes Running clearance -0.1000 mm (owner's 8A
   example); (b) length `+1` quotes Space claim, Large-diameter stack +1.0000 mm, and (b) diameter `+1` quotes Space
   claim, Rotating diameter +1.0000 mm (a term failing on the stepped design only); only-the-joint-fails quotes the
   inner hub's needed minimum; the temperature edge's diameter `+1`, length `+1` and count `-2` flip Temperature limits
   to caution and quote no temperature quantity; Hot-low torque is never quoted; Temperature limits is never quoted
   (hot day, the temperature edge, and a temperature-red design built in the test); a check passing before or after is
   never quoted; a term whose quoted change fails `shows` is not quoted; every quoted unit is the result's own through
   `with_unit`, and every quoted path's metadata unit is one of mm, µm, MPa, °C and N·m (`metal.cold_high_input_Nm`
   carries "N·m" itself, `engine/metal_design.rs:308`; only the sweep columns carry "Nm", `engine/sweeps.rs:73-77`;
   code review 2 S7).
7a. `the_card_names_the_clamp_only_while_unpaused` (code review M1): at `clamp_at_its_limit()` with `clamps_paused`
   false the Overall length `+1 mm` side flips Shaft clamp to "fails" (C, `clamp.rs`: Good -> Bad) and no clamp
   quantity is quoted; with `clamps_paused` true no side of it, and no line of `only_the_clamp_fails()`, names the
   clamp.
8. `the_held_row_reads_set_by_auto_and_the_card_is_the_same_design_s`: for each Auto variable at (a) and (e2),
   `lines(Some(v))` replaces exactly the row `Control::held_by` names with `held_line` and adds the matching Auto
   line; `lines(None)` of the same card shows that row's numbers; the card (the cached value) is equal either way. At
   the odd count, which has no size rows (T-5), `lines(Some(RingRadius))` and `lines(Some(AxialLength))` draw the Auto
   line and no held line, without a panic, and the gap and count rows equal `lines(None)`'s; `lines(Some(MagnetsPerRing))`
   holds the count row (C, section 3d of `out-run5.txt`; code review 3 S6).
9. `check_terms_keeps_the_fix_tests_margins`: `terms()` (test) returns `check_terms` plus the clamp's term
   (`check_terms` gives the clamp none); the fix tests that read it pass unchanged; the shaft joints' list has the
   same eight terms in the same order at a judged and a not-judged design; `SPACE_CLAIM_AXES`, `axis_reserves` and
   `SPACE_CLAIM_AXIS_SIZES` name the three axes in one order (code review 2 S10).
9a. `size_changes_and_margins_are_drawn_only_when_they_show` (code review M2): `shows` is false for 0.0, ±1e-15,
   3.6e-15, NaN and ±inf, true for 0.0046 (from 1 and 2), and 6e-5 shows from 2 but not from 1 (C); the odd count's gap
   sides have `diameter_change_mm == None` and `length_change_mm == None`, and `card()` of that design equals a second
   `card()` of it; no row at (a) draws a size text; a rendered side with a planted 1e-15 size change draws nothing for
   it.

### 8.2 The cache and idle frames

10. `idle_frames_compute_the_card_once_and_an_edit_again` (`gui/panel.rs`, mirroring `:8980-8993`): on the Size tab,
    five idle frames leave `consequences.computed` unchanged; one gap edit adds exactly one; toggling
    `harness.panel.clamps_paused` adds exactly one (M1). Auto (T-3; code review M5: turning Auto off adopts the solved
    value, `gui/panel.rs:723-729`, so a toggle adds none only when the design shown does not change):
    - (i) at (b), Auto on the magnet length, solved after `frame_after(DEBOUNCE_S + 0.01)`: turning Auto off adds none,
      and the Overall length row then shows numbers equal to the held row's computed sides;
    - (ii) at (a) with Auto on the magnet count (best at 10, the design shown, section 2.1): turning Auto on, letting it
      solve and turning it off adds none.
11. `idle_frames_change_no_input` (`:2348`) and `idle_frames_on_the_first_view_change_no_input` (`:2147`) run with the
    card open and closed, their card cases with `harness.panel.clamps_paused = true` (the shipped state; code review 2
    S3).

### 8.3 Number format

12. `a_change_reads_as_a_signed_percentage_to_a_tenth_with_its_true_sign` (`gui/format.rs`): +0.0349 -> "+3.5 %",
    -0.0037 -> "-0.4 %", -0.0028 -> "-0.3 %", -0.0004 -> "-0.0 %", +0.0004 -> "+0.0 %", 0.0 and -0.0 -> "+0.0 %",
    NaN -> "NaN %", +inf -> "+inf %", -inf -> "-inf %" (C).
13. `a_change_keeps_its_sign_and_shows_a_digit_that_is_not_zero`: `signed_change(.., 2)`: 0.2010 -> "+0.20",
    -0.0063 -> "-0.01", -0.0046 -> "-0.005", -0.00012 -> "-0.0001", 0.0 and -0.0 -> "+0.00", -0.00004 -> "-0.00"
    (past the cap, T-9), 1e-10 -> "+0.00", NaN -> "NaN", ±inf -> "+inf" / "-inf" (C); `change_decimals` at the cap
    (an answer of `fewest_decimals_apart` above `from + 2` reads `from`); `signed_fixed`: -0.0063 at 3 -> "-0.006",
    -0.0046 at 3 -> "-0.005", -0.00004 at 2 -> "-0.00", 0.0 and -0.0 at 3 -> "+0.000", NaN -> "NaN" (C); and a row's
    shared decimals (T-14): (e)'s diameter row 3, a row with one number side that side's own, a row of words none.
14. The gap tests of section 5, and `the_readout_texts_have_glyphs_and_plain_words` (`gui/size_controls.rs:475`) with
    the new texts.
15. `no_card_number_reads_zero_or_disagrees_with_its_percent`: at every design of section 2 (`card_designs()`, the
    table of section 8's intro), every side's N·m text is not zero after parsing unless the change is under the cap,
    and its sign equals the percent's (the count `-2` side of (c) with `f_cal_original` -0.95 included: "-2.29" and
    "-158.9 %"); the one-line forms at short magnets, shown does not fit, variation 1.0 and zero torque carry no
    number.

### 8.4 Words, glyphs and bans

16. `the_card_texts_never_rank_the_controls` (beside `the_shaft_joint_texts_never_order_the_torques`'s pattern,
    `gui/guidance.rs:8437`): every line of `lines(auto)` for each Auto state at every design of section 2
    (`card_designs()`), at the shipped pause (`clamps::PAUSED`), contains no word of `CARD_RANKING_WORDS` by the
    matching rule of 4.4, no `TORQUE_ORDER_WORDS`, no `mentions_gearbox_protection`, no "clamp" in any case, and passes `plain_text_violations`;
    the matching rule has its own positive and negative examples ("levers", "Rank" match; "almost", "bestow" do not).
17. `every_card_text_has_its_glyphs`: `assert_glyphs` (`gui/test_support.rs:453-462`) over every line at every design
    (`card_designs()`) and the heading.
18. `the_card_s_strings_are_constants_and_its_numbers_come_from_results`: two designs with different results give
    different lines at every number position (the pattern of `the_problems_quote_numbers_from_the_results`).

### 8.5 Layout (headless; the drawing's height through `drawing_rect`)

Every test here sets `harness.panel.clamps_paused = true`: `Harness::of` unpauses the clamp section
(`gui/panel.rs:1674`), while the heights of 6.2 were measured in the shipped, paused state (code review 2 S3). It sets
it before the first frame that draws the card, the frame that decides the default (code review 3 S3): each test
starts on the Geometry tab (`Harness::on_screen`, `:1656-1660`), sets the pause, the design and the Equation panel,
then clicks the Size tab, through one helper in `gui/panel.rs`'s tests. `Harness::of` draws its first frame unpaused
(`:1673-1682`) and `size_tab()` clicks the Size tab inside itself (`:4991-4998`), so setting the pause after either
would leave the default decided on an unpaused frame. Tests 21 to 25, which change the card's state, step frames past
the collapse animation (`CollapsingState` animates its openness over egui's animation time; `frame_after`) before
they measure (code review 2 S13; test 25 since code review 4 SF-5). A test opens or closes the card as a user does,
by a click on the heading's text (`text_center(output, CARD_HEADING)`; the heading toggles, 6.1); test 22 also
clicks the arrow, half an `ui.spacing().indent` left of the heading's text rect (code review 4 SF-6).

19. `the_card_sits_between_the_readouts_and_the_drawing` (1280 x 800, (a) and (b)): the heading's text rect is below
    the Pull-out readout and above the drawing; it starts open; every row's text is drawn; the drawing is at least
    160 pt; the label of `:936` is not drawn.
20. `the_card_starts_closed_in_a_narrow_window_and_the_drawing_gains` (930 x 800, Equation panel closed): closed at (a)
    and at short magnets (the one-line form, which rule 3 alone would open: T-10's width rule decides). The drawing at
    least as tall as at 69fa505 with the label: the closed card's drawn height is at most the replaced label's, laid
    out as a wrapped galley at the region's width in the test (the label's 69fa505 text kept in the test as the record
    of what the card replaced): 32 against 42 pt at 930 x 800, 18 against 28 pt at 1280 x 800, in both styles (C,
    section 6 of `out-run4.txt`). Not a pinned height: the 201.5 pt of 6.2 is the app's style, where the harness's
    drawing at 69fa505 is 246.5 pt (6.2; code review 2 S12). At 1280 x 800 short magnets starts open. At 930 x 800
    the card's Why link (of the three drawn links to `SIZE_TAB_NOTE`, the one whose text rect's left edge is at or
    right of `drawing_rect(output).left()`) lies inside the centre region: its text rect's right edge is at most
    `drawing_rect(output).right()` (the header row wraps, 6.1; code review 4 MF-1). A click on its last character
    (`last_characters`: in the wrapped row its text rect starts at the row's left, over the heading, which toggles the
    card) then opens `torque_from_shear`. Test 21 clicks it at 1280 x 800 only.
20a. `the_card_starts_closed_when_it_would_take_the_drawing_under_its_floor` (code review 2 MF-5): rule 3 deciding
    alone, at 1080 x 745 with the Equation panel closed and the welcome strip shown. The test first asserts, on the
    deciding frame, that the `CardView` it was decided for has the Equation panel closed and that the centre region is
    at least `CARD_OPEN_MIN_WIDTH` wide, read as `drawing_rect(output).width()`: the `CardView` holds no width (6.2;
    code review 3 MF-2), and the drawing fills the region's width, equal to it on every frame measured (425.7 to
    428.0 pt there; C, section 7 of `out-run5.txt`). So rules 1 and 2 pass and cannot be what closes it. Then (c)
    starts closed (its open card, 360 pt in the app's style and 355 in the harness's, leaves 88.1 / 83.1 pt less than
    the drawing's floor) and short magnets, the one-line form, starts open (171.9 / 153.9 pt to spare) (C, section 6
    of `out-run4.txt`). Not the review's 1280 x 600: below about 745 pt of window the panel overflows the window, so
    the room at the card stays 533 pt there and (a) starts open with 135.9 pt to spare (6.2).
21. `the_card_starts_closed_while_the_equation_panel_is_open` (1280 x 800): after a Why link opens the panel, the card
    closes (no user choice); `no_centre_view_paints_or_takes_clicks_in_the_equation_panel` (`gui/panel.rs:7292`)
    passes with the card open and closed.
22. `the_user_s_open_or_close_wins_for_the_session`: at 930 x 800 a click on the heading opens it; it stays open after
    a design edit, a resize to 1280 x 800 and the Equation panel opening; Reset all and a load keep the choice. A click
    on the arrow then closes it and one on the heading opens it again: both set `card_choice` (6.1, 6.2).
23. `the_card_never_opens_or_closes_under_a_drag`: twenty gap-slider drag frames at 1280 x 800 keep its state; then,
    from (a), the Outside diameter dragged down across the fit edge (the "Exploration only: magnets do not fit" line
    appears above the card), then Required dragged across hot-low, then an assumption edited under Detailed inputs so
    that the assumptions banner appears in the header (the region loses a line of height): the card's state holds
    through all three, and `card_start`'s `CardView` does not change (code review M3; code review 2 S1). Then a design
    change that widens a side panel (code review 3 MF-2): at 1080 x 800, from (a) with the card open (rule 3 leaves
    83.9 pt over the floor in the harness's style, 52.9 in the app's) and no user choice, `coupling.backiron` set to 0
    ((c)): the card stays open and `card_start`'s `CardView` is unchanged, while
    the centre region narrows 428.0 -> 425.7 pt in the harness's style (427.9 -> 425.7 in the app's) and rule 3 at (c)
    would close the card (-28.1 pt in the harness's style, -59.1 in the app's) (C, section 7 of `out-run5.txt`). Under a
    key on the region's width this case is red; the earlier edits widen no panel, so they cannot catch it.
24. `the_card_body_scrolls_above_the_drawing_s_floor`: opened by a click on the heading at 930 x 800, the drawing is
    still at least 160 pt.
25. `the_card_s_measured_height_is_the_drawn_height` (code review 3 MF-3): the pre-measure equals the drawn open height
    within 1 pt at (a) and (b), at 1280 x 800 and at 930 x 1024, where the card is opened by a click on the heading
    (rule 2 closes it there) and the body is under its cap (harness style: bodies 152 / 169 pt against a cap of
    371.9 pt at 1280 x 800, and 320 / 365 pt against 473.9 / 459.9 pt at 930 x 1024; app style: 156 / 174 against
    346.9, and 324 / 370 against 436.9; C, section 7 of `out-run5.txt`). Plus one assertion at 930 x 800, where the
    bodies are over the cap (320 / 365 pt against 249.9 / 235.9 in the harness's style; 324 / 370 against 212.9 in the
    app's), so the drawn card is not the pre-measured one: with the card opened by the user, the drawn body equals the
    cap, `max(room - header - 2 × item_spacing.y - drawing_floor, one body row)`, within 1 pt. The pre-measure and the
    cap each have a pin; test 24 checks only the drawing's floor. The probes (code review 4 SF-5):
    - the pre-measure: the panel computes `open_height` on a deciding frame only, and a click on the heading sets the
      user's choice with no deciding frame, so the test calls `card_open_height` (6.2) itself, with the card's
      `lines(auto)`, in a scratch frame of the centre region's width and the harness's style;
    - the drawn height: from the top of the heading's text rect (`text_rect(output, CARD_HEADING)`) down to
      `drawing_rect(output).top()`, less one item spacing;
    - `room`: from the heading's top to the bottom of its text shape's clip rect (`confined` clips to the centre
      view's region, `gui/panel.rs:1581`), as the prototype's `layout.rs` reads the room at the replaced label.

### 8.6 Browser, screen dump and release record

26. **gui-smoke** (`.claude/workflows/gui-smoke.js`): `MAGCOUPLING_SCHEMA` (`:21-39`) gains the required booleans
    `torque_card`, `drag_moved` and `drag_frames_within_60fps` and the numbers of 7.2; the fallback object (`:102`)
    gains the fields. The prompt (`:89-100`) gains the card step, quoting whole texts (code review M9), at a fixed
    place (code review 2 MF-4): after the existing checks, on the page the frame-time step opens (7.2 step 1: the init
    script, the share link again at 1280 x 800, 8 s), one screenshot: between the readouts and the drawing, the
    heading `CARD_HEADING` and its Why link, open; under it `CAPTION`, `AUTO_HOLDS_LENGTH`, and four rows: one starting
    "Outside diameter:", the held row `held_line(Control::OverallLength)` ("Overall length: Set by Auto; turn Auto off
    to edit"), one starting "Magnet face gap:" and one starting "Magnets per ring:"; the drawing still visible. Then
    the drag in a fresh screenshot (7.2 step 2), then the console collection and the browser's close that ended the
    existing checks (`:98`, moved after both steps; code review 3 S8). The prompt says why the card step is not run
    earlier: after the file load the design has no sizing (no held row, no Auto line), the dashboard Why link opens the
    Equation panel (which closes the card) and the tooltip step leaves the Geometry tab selected (no card).
    `passed` requires `torque_card`, and `drag_frames_within_60fps` only with `ARGS.frame_check` (7.2 step 4).
    `the_smoke_test_share_link_opens_its_design` (`app.rs:211-284`) asserts that `CARD_HEADING`, `CAPTION`,
    `AUTO_HOLDS_LENGTH` and `held_line(Control::OverallLength)` are each in the script, and that the three row starts
    are `"{label}:"` of `Control::OutsideDiameter`, `FaceGap` and `MagnetsPerRing`. The prompt is written after G-3
    sets the words (10.1 step 2 comes before step 5), and the assertions keep it from going stale (the R14 rule,
    `app.rs:251-253`).
27. **Screen dump**: the slice's copy of the BL-064 tool (`.superpowers/sdd/2026-10-19-magcoupler-slice3/screen-dump/`,
    gitignored; from `.superpowers/sdd/2026-10-17-bl064-shaft-joint/screen-dump/`) gains a `card` record kind:
    `card<TAB><design key> NN<TAB><line>` for the heading and every line at every design of section 2.1 with its Auto
    state, and a `gap` record of `torque_gap_text` at (a), (b), (c), (e), (e2). Both read public items only:
    `gui::torque_card::{card, CARD_HEADING}` and a `pub use size_controls::torque_gap_text` in `gui/mod.rs`. **Both
    sections** are behind the tool's own feature `card`, on in `after/Cargo.toml` only: main has neither the module nor
    a public `torque_gap_text` (`size_controls` is a private module, `gui/mod.rs:29`); the before values of the gap come
    from the shipped literals (`gui/size_controls.rs:418-455`) (code review S12). The slice's copy of `check_shipped.py`
    (`.superpowers/sdd/2026-10-19-magcoupler-slice3/check_shipped.py`, from
    `.superpowers/sdd/2026-10-17-bl064-shaft-joint/check_shipped.py`; code review 2 S14) gains `QUOTES` for every card
    line and gap text the docs and the STOP sheet quote (at least (a)'s four rows, (e)'s diameter row in both forms of
    T-14, (c)'s count row with `CALIBRATION_CHANGE`, the count row of (c) with `f_cal_original` -0.95 (a side below
    zero with `CALIBRATION_CHANGE`; code review 4 MF-2), the mixed pair's `JUMP`, the short cavity's length row with
    `JUMP` on both sides, the temperature edge's flips, variation 1.0's `SHOWN_NO_HOT_LOW`, (b)'s held row and Auto
    line, and the five gap texts).

Pinned counts that move: section 6.3's line counts, and the Size tab's Why links to `SIZE_TAB_NOTE` (2 -> 3 on the
default Size tab, 1 -> 2 at an odd count; 6.3, code review 2 MF-3). No count of checks, guides, notes, inputs, results,
dashboard rows or share-link length moves (no engine result or input is added).

## 9. Docs and backlog

### 9.1 `docs/ai/backlog.yaml` (three new items and one line in BL-065's notes, in the slice's docs commit)

```yaml
- id: BL-073
  title: "Magcoupler Size tab: a torque card that shows what one step of each size control does to hot-low torque, and what it costs"
  dimension: gui
  risk: physics
  evidence: "Guidance order G-1 (docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md:34): the torque sensitivity card after the note rewrite; the trigger case (spec1:8-9): 10 -> 8 magnets and a lower target did not shrink the coupling and nothing on screen said why. At 8 magnets on the default ring (compute_all, main 69fa505) Outside diameter -1 / +1 mm moves hot-low -0.37 / -0.28 % while Overall length moves it -9.05 / +9.05 %; nothing on screen says so. Owner decisions 1A-17A (2026-10-19), spec docs/superpowers/specs/2026-10-19-magcoupler-torque-card-design.md."
  acceptance: "gui/torque_card.rs (pure) and its card on the Size tab between the readouts and the drawing, per the spec: four rows (Outside diameter, Overall length, Magnet face gap, Magnets per ring) at ±1 mm, ±1 mm, ±0.1 mm, ±2 through each slider's own mapping; every number a difference of two compute_all runs; words where a side does not fit, has no prediction, leaves its slider, jumps or breaks the keyway; flips and failing-check quantities per side; the Auto marking; the N±2 lines moved off Magnets per ring; one Size-tab number format (one decimal of percent, N·m to the fewest decimals that keep the true sign, a row's two sides at one precision) on the card and the readout's gap; the tests of the spec's section 8 red first; G-3 physics review of every card string; whole-branch review; GATE PASS with gate 12 running; web build; gui-smoke with the card step and a browser drag-frame p95 at or under 16.7 ms; docs/ai 02-05 and the README; STOP with the sheet; the owner's OK of the diff."
  priority: 2
  status: open
  notes: "Owner decisions 1A-17A (2026-10-19, owner-decisions.md in the gitignored slice-3 research). Reverses spec1 section 3.3 for the neighbour lines (7A) and the readout's whole-percent format (10B). The controller's rulings on the owner's night-run approval, on the STOP sheet for the owner to reverse: the jump rule widened (owner question O-1: words whenever the overlapping magnet length changes by anything other than the step) and no temperature margin (O-2). BL-065 stays open (12A): the card ships the model's numbers. Deploy authorization is asked at the STOP."
- id: BL-074
  title: "Overall length from a blank override jumps when the housing does not hold the outer ring's own length or the inner part is shorter: the first slider move sets both rings to one length (B442 inner, B842SH outer: -1 mm reads +106 %; matched parts in an 11 mm cup cavity: +1 mm moves hot-low -6.2 %)"
  dimension: gui
  risk: physics
  evidence: "magcoupling-rs/src/engine/size_controls.rs:453-466: resize_length sets both rings' axial override to requested - length_range.min + axial_range.min (:460); from a blank override the length range follows the outer ring the cup cavity holds (:220-226), the cavity following the ring as max(cup_depth + L - own, L) (engine/housing.rs:106-108). With part_inner B442 (6.35 mm) and part_outer B842SH (12.7 mm), overall length 31.80 -> 30.80 mm gives override 11.7 mm, active length 6.35 -> 11.70 mm and hot-low 1.0099 -> 2.0844 N m (+106.39 %); +1 mm gives +146.16 % (compute_all, main 69fa505; slice-3 spec section 2.3). The reverse pair moves the active length by the step but trims the longer inner ring to the outer's length less or plus the step (12.7 -> 5.35 / 7.35 mm, the hub length following it, 13.0 -> 5.65 / 7.65 mm). Matched parts move it by the step only while the cup cavity input (metal.cup_depth_mm) is at least the outer ring's length: at 11.0 mm (input range 3-40) -1 mm sets both rings to 10.0 mm (active -2.70 mm) and +1 mm to 12.0 mm (active -0.70 mm, hot-low -6.15 %) (slice-3 spec physics review 1; spec section 2.3). No doc records it (grep of docs/ai and the README). The slice-3 torque card shows words on every first step that is not a step (its jump rule, widened by owner question O-1, answered 2026-10-19); the slider itself does it unannounced. A diameter step can jump the same way (slice-3 spec R-13): where the shaft boss sets the Outside diameter (metal.boss_od_mm, slider 12-40 mm, above both the cup and the cap, all under Detailed inputs), the slider's mapping lifts the cup from its own size to the boss plus the step in one step (requested_cup = requested - cap_offset.max(0), engine/size_controls.rs:431-432). At npole 8, ring 8.0 mm, cap 30 mm, cap thread 29 mm and boss 40 mm, +1 mm of Outside diameter lifts the cup +3.22 mm (ring 8.00 -> 9.49 mm, against 0.46 mm of ring for 1 mm elsewhere at 8 magnets) and hot-low +7.55 %, which the card shows as a number (it has no diameter jump rule); with the default boss (22 mm) and magnets the boss never sets the diameter (slice-3 spec, section 3d of its prototype's out-run5.txt)."
  acceptance: "owner decides whether the jump is intended, for a blank override whose housing does not hold the outer ring's own length, or a shorter inner part. If intended: the Overall length hover (LENGTH_HOVER, gui/size_controls.rs:46), the README Size-tab paragraph and docs/ai/02-system.yaml say that a first move from a blank override sets both rings to one length, with headless tests pinning the mixed pair's, the reverse pair's and the short cavity's first step. If not: red tests first at the mixed pair and the short cavity (the first step moves the active length by 1 mm, or is refused with a reason), then a mapping change in engine/size_controls.rs chosen with the owner (for example the override counted from the shorter ring, or a per-ring step), the card's jump rule kept as a guard, the README and docs/ai updated; G-3 review of any new text; GATE PASS with gate 12 running"
  priority: 3
  status: open
  notes: "Filed with slice 3 (owner decision 17A, 2026-10-19), from the slice-3 context critique's finding 6, widened by the slice-3 spec's physics review 1 (MF-2, SF-5: the short cavity and the reverse pair's trim). The torque card's jump rule (spec section 3.3 step 7, test 3; widened by owner question O-1) keeps the card honest either way on the length row; the diameter row's boss plateau has no such rule (spec R-13)."
- id: BL-075
  title: "BL-064 fold-in review notes with substance: a too-cold gearbox shaft hides a judged red joint, the joint links read the inputs not compute's copies, the fitting source quotes results not links, and test and DRY polish"
  dimension: quality
  risk: physics
  evidence: "Gitignored records .superpowers/sdd/2026-10-18-bl064-shaft-joint/lo-B1-review.md .. lo-B12-review.md and lo-final-review.md (Minor items; ruling B-6 in rulings.md left them unfiled). (1) lo-B1 M-2, lo-final M-1 O-2: CannotBeMade::GearboxShaftTooCold (magcoupling-rs/src/engine/shaft_joint.rs:912, returned at :960) stops every joint being judged though the gearbox shaft's temperature enters only the cup's fitting term: compute_all at 69fa505 with the hub minimum at 10 um: gearbox shaft at -20 C reads 'The fit slips (inner hub): holds 3.48 N·m against 7.65 N·m', at -21 C (a file or link value; the slider stops at -20 C, OUTPUT_SHAFT_COLDEST_C :175) 'Shaft joint check unknown: the gearbox shaft is below its slider's bottom' (amber; slice-3 spec scratch crate, src/bin/bl075.rs; the review's own example said 3.85 N·m). (2) lo-B2 M-1: joint_links (magcoupling-rs/src/engine/api.rs:177-200) reads the eight geometry values from inputs, so the_joints_rebuilt_from_the_results_are_the_engine_s no longer guards against compute overriding one in ci or md. (3) lo-B4 M-2: fitting_source (magcoupling-rs/src/gui/guidance.rs:2193) chooses its arm on the links but quotes the adhesive limit and the inner onset from results (onset_texts :2226); a planted fault read 'above the 49.93 C allowed: the magnets' single-ring onset, 91.35 C, less 10.00 C'. (4) Polish: one keyed rule written twice (lo-B1 M-5, shaft_joint.rs); 10f64.powi(-(decimals as i32)) typed twice (lo-B4 M-3, guidance.rs:1917, :2354); retyped test designs and lookups (lo-B8 M-2 and M-3, lo-B9 M-3 and M-4); unpinned edges (lo-B1 M-6, lo-B5 M-2, lo-B7 M-2, lo-B10 M-2); four inline paths in the joint guide (lo-final M-5)."
  acceptance: "owner rules item (1) (keep the amber with its reason recorded, or refuse only the cup's fitting term so the inner hub stays judged, red test first at hub minimum 10 um and gearbox shaft -21 C); (2) joint_links takes &CouplingInputs and &MetalDesignInputs, compute passes ci and md, the rebuilt-joints test regains its guard (a planted override fails it); (3) fitting_source quotes links.adhesive_limit_C and links.inner_onset_single_ring_C through the links shaft_joint_problem already built (pins unchanged; a planted fault in joint_links fails one); (4) the listed polish, each with a test that fails before it where it is behaviour; G-3 re-check of any on-screen text; GATE PASS with gate 12 running"
  priority: 3
  status: open
  notes: "Filed with slice 3 (owner decision 17B, 2026-10-19). Ruling B-6 (2026-10-19) kept the 70 Minor notes in the review records unless the owner asked to file them; these are the ones with substance. Item (1) sits against build ruling B-5 note 1's 'a real red stays red'."
```

One line added to BL-065's `notes` in the same commit (physics review SF-3): "The slice-3 torque card's percents are
a consumer: re-run its probe rows when BL-065 lands (in the model with c_end scaled to the steel slice, the length row
reads about 9.3 %/mm against 8.8 %, and (a)'s count -2 about -28.1 % against -26.8 %; the N·m of the length row is
unchanged)."

### 9.2 Other docs (with the code, each in the commit that makes it true)

- `docs/ai/02-system.yaml`: `:82`'s `key_files` gains `src/gui/torque_card.rs`; `:116`'s Size-tab clause (the card,
  its rows and words, the moved neighbour lines, the first-line rule of 4.3, the gap's number format, the cache's card
  and its one key in three parts: the design, the Minimum OD and the clamp pause) and its "under 1.6 ms" figure
  (section 7.2); `:106`'s drag-frame clause and its compute_all count (section 7.2); `:115`'s reachability clause,
  which lists where the Size tab's Why links sit ("under Magnets per ring and Outside diameter"): the card's heading
  too (code review 2 MF-3); a magcoupling status line for slice 3.
- `docs/ai/03-structure.yaml`: the `explain` line (`:64`): `plain.rs` gains `CARD_RANKING_WORDS` and its whole-word
  matcher, beside `TORQUE_ORDER_WORDS`. The `gui` line (`:68`): `torque_card.rs` with its types and functions
  (`ShownNoHotLow`, `count_step`, and `STEP_SLACK_MM`, the one slack of the range test and the jump rule);
  `format.rs`'s `signed_fixed`, `signed_change`, `change_decimals`, `CHANGE_EXTRA_DECIMALS`, `fewest_decimals_apart` and
  one-decimal `signed_percent`; `guidance.rs`'s `check_terms` (with `CheckTerm` and `joint_term`),
  `SPACE_CLAIM_AXIS_SIZES` and the cache's three-part key; `geometry_view.rs`'s `drawing_floor` and `list_row`;
  `size_controls.rs`'s `slider_minimum`; `panel.rs`'s `torque_card_ui` and `card_open_height` (6.1, 6.2); the
  `pub use size_controls::torque_gap_text` in `gui/mod.rs`; the
  `test_support.rs` clause gains the new test designs and the `drawing_rect` probe. Removed from the same line (code
  review 3 S5): `Neighbour` and `NeighbourOutcome`, "neighbours N-2 and N+2" under `Consequences::of`, and
  `AUTO_RESIZES_RING` and `AUTO_RESETS_LENGTH` under `magnet_count_line`. `manual_size_gui` (`:73`): the card in the
  centre and the lines under Magnets per ring.
- `docs/ai/04-memory.yaml`: `:13`'s NEXT (slice 3 done; then lessons); the open BL-065 dependency of the card's
  percents (section 10.3).
- `docs/ai/05-update-tracker.md`: a new top entry. It records that 10B closes slice 1's S19 (percent precision;
  deferred in 04-memory at 5a1126a, dropped from it in 47bdcd7, so no doc holds it at 69fa505), and the answers to O-1
  (widen) and O-2 (quote none, carried as recommended), the controller's rulings on the owner's night-run approval
  with the round-3 and round-4 ones (section 1.1, "The night run").
- `magcoupling-rs/README.md`: `:155`, the `src/gui/` module list, gains `torque_card.rs`, and `size_controls.rs`'s
  "consequence lines" no longer include the neighbour lines; `:203` (the gap's format, "-P.P %"); `:214-226` (the
  Magnets per ring lines and the card); `:362` (where the Size tab's Why links sit: the card's heading too; code
  review 2 MF-3); the test tables' rows for `gui/torque_card.rs` (new), `gui/format.rs`, `src/engine/explain/plain.rs`
  (`:684`: the card ban list and its matcher test), `gui/panel.rs` (`:686`, whose slice-2 T19 clause also names the Why
  links' places), `gui/geometry_view.rs` (`:687`: `drawing_floor`) and `gui/guidance.rs` (`:693`, the same). `:693`'s
  guidance row also loses its neighbour clauses ("the consequence numbers equal `compute_all` of the neighbour design
  ...; neighbours outside 4 to 40 magnets are not shown"), whose tests move to the `gui/torque_card.rs` row (code
  review 3 S5).
- Code doc comments and tags that state the old rule, the old key or the old places (code review 2 S8, MF-3):
  - `gui/size_controls.rs:173-177` (`torque_gap_text`: "a whole percent", section 5) and the module doc `:6-7` (the
    what-moves-what lines under each control);
  - `gui/guidance.rs:33-38`, the module doc (code review 3 S4): "two magnets fewer and two more ... words, not
    numbers, where the blocks do not fit or the model has no prediction" (the neighbour lines, now the card) and
    "[`ConsequenceCache`] keeps them until the inputs or the minimum diameter change" (now the clamp pause too);
  - `gui/guidance.rs:2711-2712` (the what-moves-what comment: "changes to whole percent"), `:2719-2727` (the docs of
    the two removed constants), `:2729-2731` (`magnet_count_line`), `:2988-2990` and `:3007-3008` (the cache recomputes
    "when the inputs or the minimum diameter change": now the clamp pause too);
  - `gui/guidance.rs:606-607` (`SIZE_TAB_NOTE`'s doc: "the Why links under Magnets per ring and Outside diameter"), the
    test comment at `:3926-3928`, and the review dump's place tag "size tab/under Magnets per ring and Outside
    diameter" (`:9731`), which gains a place of its own, "size tab/the torque card's heading";
  - `gui/panel.rs:310-311` (the `consequences` field: "recomputed only when either changes");
  - `gui/format.rs:89-91` (`signed_percent`: "a whole signed percentage").
- `.claude/workflows/gui-smoke.js`: the meta description (`:3`) and the calculator step's comment (`:77-84`) gain the
  card step and the frame timing (test 26).
- `docs/analyses/2026-10-14-magcoupler-help-review.md`: a new `## Slice 3 torque card` section with the G-3 review.
- Do not edit `docs/superpowers/plans/*` or earlier specs (history).

## 10. Release, out of scope, risks

### 10.1 Release (15A)

0. O-1 is answered (widen) and O-2 answered as recommended (quote none) (section 1.3); the plan starts from this text.
1. Red tests, implementation, both clippy runs, fmt, the touched suites. `risk: physics`: session-model implementers.
2. The G-3 review of every card string (section 4.1 and 4.3, the templates rendered at every design of section 2),
   fresh-context session model, recorded in the help review; must-fix applied and re-reviewed. The sheet it reviews
   includes the short cavity's length row and the temperature edge, the one probe design whose card shows a
   Temperature limits flip rendered (physics review SF-6: the card's flip carries no stored-3D marker, the dashboard
   row does). G-3 also confirms "everything else held" beside rows that list size changes (the count row's "Outside
   diameter +2.2 mm" at (e), the diameter row's cap following the cup: consequences, not held inputs; physics review
   SF-7) and beside the reverse pair's length row, whose steps trim the longer inner ring 12.7 -> 5.35 / 7.35 mm and
   the hub with it (13.0 -> 5.65 / 7.65 mm) with no flip and no quoted cost: the one place where the caption is not
   literally true and no cost line shows it (physics review 2 SF-5; the number is the model's honest value, and BL-074
   records the trim). G-3 decides T-11's "cold-high", the new `SHOWN_NO_HOT_LOW` (rendered at variation 1.0 and at
   zero torque), and the one change drawn at two precisions (3.6: npole 4's gap and count rows, the diameter row's
   "Space claim: Rotating diameter ±1.00 mm"; physics review 2 SF-4). It also reads the count row of (c) with
   `f_cal_original` -0.95, a side below zero with `CALIBRATION_CHANGE` and "Shaft joints: passes", the dashboard's own
   verdict on the stepped design (code review 4 MF-2). And it reads the diameter row at (a) with Minimum OD 45.0 mm,
   where both sides read "at the end of its range" below the slider's range (code review 5 SF-3).
3. A fresh-context session-model review of the whole branch.
4. `MAGCOUPLING_PYTHON=<venv python> bash linkage-sim-rs/scripts/gate.sh`: GATE PASS with gate 12 running (no SKIP).
5. The web build, then gui-smoke on the local server with `frame_check: true`: the card step and the frame-time check
   (section 7.2).
6. **STOP**: send the owner a short sheet of every new and changed on-screen text and number at the probe designs
   ((a), (b), (c), (e), (e2), the mixed pair, the short cavity, the temperature edge, (c) with `f_cal_original` -0.95,
   and the one-line forms at short magnets, shown does not fit, variation 1.0 and zero torque), from the screen dump and
   checked by `check_shipped.py`, with the browser frame times and the spec's own decisions T-1 to T-14. It lists the
   controller's rulings on the owner's night-run approval, each for the owner to reverse (section 1.1, "The night
   run"): O-1's answer (the widened jump rule: the short cavity's `+1 mm` side reads words, not -6.2 %), O-2's (no
   temperature margin), test 0 on the readouts' own rule (`SHOWN_NO_HOT_LOW` also for a pull-out not above zero, its
   draft reworded), the `CardView` keyed on the screen (a side panel's width, a user's drag of its edge included,
   never decides the default again), test 25's windows, the wrapped header row (code review 4 MF-1: at 930 x 800 the
   Why link continues under the heading) and a count side below zero shown as its number (code review 4 MF-2: (c) with
   a design file's `f_cal_original` -0.95 reads "Magnets per ring: -2 -> -2.29 N·m (-158.9 %), includes a calibration
   change, Shaft joints: passes; +2 -> magnets do not fit"). Among the spec's decisions, the narrowings and readings of
   owner decisions: T-9 (10B, float noise), T-12 (5A, the one-line forms; and 9A through `ShownNoHotLow`), T-13 (8A,
   no temperature margin) and T-14 (10B per row), with (e)'s diameter row both ways ("-0.006" and "-0.005" against per
   number "-0.01" and "-0.005") and one line on a row whose two sides read the same "-0.005 N·m (-0.3 %)" while only
   one flips Hot-low torque ((e2)+Auto length's design with Auto off and Required between its diameter sides' hot-low,
   1.494977 and 1.495232 N·m; C, section 3d of `out-run5.txt`; physics review 3). It names R-13, the boss plateau,
   where a diameter step that lifts the cup 3.22 mm reads as a number. It says that the first view's drawing shrinks
   from 335.5 to about 185 pt at (a) with the card open (section 6.2); that a click on the card's heading, as on its
   arrow, opens or closes it (6.1; code review 4 SF-6); that clicking the card's own Why link opens the Equation
   panel, which collapses the card when the user has made no choice (4A; code review S17); and that with the card
   closed (930 x 800, or the Equation panel open) nothing near Magnets per ring says Auto re-sizes after a count change
   (T-8; code review 2 S15). Ask for deploy authorization there. Merge (`--no-ff`) and push only on the owner's
   OK; then watch the deploy and run gui-smoke on https://colesorkness.com (without `frame_check`).

**The backlog's physics rule** (code review S15). `docs/ai/backlog.yaml`'s header asks a `risk: physics` item for the
`fbd-math-reviewer` pass and the user's explicit review of its diff before merge. The `fbd-math-reviewer` is
linkage-specific: magcoupling physics changes get a dedicated physics reviewer whose prompt names the relevant
derivations instead (`docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md:264-266`). For BL-073 that is
step 2's G-3 physics review (after this spec's physics review 1), and the user's review is the owner's OK of the diff
at step 6; BL-073 is not `fixed` before both.

### 10.2 Out of scope

- Guided lessons, plot captions and drawing titles; the path-to-note map (BL-056 SF-3); Key design additions. The
  card's diameter step in ring radius (2/cos(π/N), section 4.2) is for the lessons slice.
- "What it takes to meet Required" (approach 2), any re-solve under Auto, links or apply on the rows (11A).
- Any engine physics, input, result, file or link change; the BL-065 fix (12A); the `resize_length` mapping (BL-074).
- Rows for back iron, parts, grades, temperatures or assumptions (2A).
- The paused clamp section.

### 10.3 Risks and what the reviews confirmed

- **R-1, the BL-065 slice argument (brief Q12b). Confirmed by physics review 1.** `T_pull = T2D · f_end · f_cal`
  with `T2D ∝ L` and `f_end = 1 - c_end · τ_p / L` (`engine/model.rs:1001-1004`), so `T ∝ L - c_end · τ_p` and a
  millimetre of length is worth the same N·m whatever the end loss: +0.200833 N·m at both `c_end` 0.15 and 0.225
  (slices 1.3214 and 1.9820 mm, C, in-model). Against BL-065's own 3D figures the length row's N·m holds to within a
  few percent near 12.7 mm: in 3D it is the model's times (1 - ds/dL), s the 3D end loss, and BL-065's figures put s
  at about 1.58 mm at 2.9 mm, 1.74 at 3.66, 1.81 at 4.25 and 2.03 at 12.7, and 1.98 mm long (H: s = L - (L - 1.3214) /
  (1 + e), e the model's over-read there; +6.65 % at 12.7 mm gives 2.03 mm, C, section 3b of `out-run3.txt`). s rises
  about 0.026 mm per mm between 4.25 and 12.7 mm, then falls about 0.05 mm to its long value; its slope at 12.7 mm is
  not measured (10.4; physics review 2 SF-3, which corrects revision 1's "to about 0.5 %"). Its % does not hold
  (1 / (L - slice): 8.79 %/mm in the model, 9.33 %/mm with the steel slice, C). The slice scales with the pole pitch,
  so the diameter and count rows carry it too. With the steel slice scaling the same way, (a)'s count `-2` would read
  -28.08 % against the model's -26.79 % (C, in-model; the f_end ratio 0.954 against 0.971 between 8 and 10 poles), and
  the diameter `+1` about
  0.24 points less (+3.25 % against the model's +3.49 %: a larger ring has a longer pole pitch, so a larger slice;
  in-model with `c_end` scaled to 0.225, physics review 1, `steel.rs`; C); at (e) the diameter row would read
  -0.07 % / -0.58 %, still under 1 % both ways (C: -0.0709 / -0.5845 %). **Answer: no words on the card; BL-065's
  notes record it** (section 9.1). Against the steel slice, at (a) the model under-reads the length row both ways
  (±8.79 % against ±9.33 %) and the count `-2` loss (-26.79 % against -28.08 %), and over-reads the diameter `+1` gain
  (+3.49 % against +3.25 %): each shift under 1.3 points (the largest, the count's, 1.297). At (e) it over-reads the
  diameter `-1` loss (-0.37 % against -0.07 %) and under-reads the `+1` loss (-0.28 % against -0.58 %), 0.30 and 0.31
  points; its length row shifts 0.73 points (±9.05 % against ±9.78 %) and its count row more, 1.53 and 2.46 points
  (-53.00 % against -54.54 %, +36.58 % against +39.05 %; C, section 3b of `out-run3.txt`), a step of two poles in
  eight. The owner's-case message survives (physics review 2 SF-3, which corrects revision 1's "the same way as the
  readout's existing BL-065 optimism": the length `+1` gain is under-read). 12A decided "no extra caveat line", so a
  later finding that the card needs words goes back to the owner, not into the card (code review S13).
- **R-2, the card's height** (restated against the rule of section 6.2, code review M7). At 1280 x 800 with the strip
  shown, (a) and (e) start open with 190.9 pt to spare before the drawing's floor, (b) with 172.9, (c) with 106.9; the
  callout list keeps every row while the card is at most 203.4 pt with (a)'s 8-row list ((a) 25.4 pt to spare; (b)
  7.4 pt if its list has 8 rows too). Longer G-3 words, or a design with more costs, first scroll the list (the
  drawing held at 160 pt), then close the card at the start or scroll its body; tests 19 and 25 fail first if the
  first view would start closed. The first view's drawing is about 185 pt at (a) with the card open (335.5 pt today).
  The owner sees the first view on the STOP sheet.
- **R-3, browser cost unmeasured.** Section 7.2 measures it; the held-back optimization is ready.
- **R-4, words beyond 9A** (T-4): "the keyway breaks through the hub" and "not a value the slider takes" are this
  spec's; the owner sees them on the sheet.
- **R-5, RING_SIZE_STAYS's clause** (4.3) is a word change to a slice-1 string the owner did not name; on the sheet.
- **R-6, a too-small change** (T-9). A step whose hot-low change is under 0.00005 N·m reads "+0.00 N·m" or
  "-0.00 N·m" with its sign (section 5's cap). None of the designs here reaches it (the smallest, 0.0038 N·m, (a) at
  Overall length 22.10 mm, diameter `+1`, drawn "+0.004 N·m (+1.1 %)"; C, section 3d of `out-run5.txt`; physics
  review 3 SF-3).
- **R-7, the end-effect neighbour.** Near the end-effect edge a step's % is large (+167.93 % at "end effect at a
  step"): true by the model, and the side next to it reads "no prediction". Physics review 1 found no text on the
  card that calls it a slope.
- **R-8, the calibration switch. Confirmed by physics review 1.** 16A's note covers the count row at (c); the words
  explain no physics, and only the count can change the factor (`select_calibration_factor` reads the back iron, the
  count, the part names and the calibration's own two factors and pole count; `engine/api.rs:243-251` passes the part
  names, which the length override keeps). A design file's calibration coefficient below 0, or measured torque of 0 or less,
  can take a count side's torque to or below zero while the design shown's is above it; that side keeps its number and the note
  (3.3 step 8; 2.3 Edges, (c) with `f_cal_original` -0.95; code review 4 MF-2).
- **R-9, temperature. Confirmed by physics review 1.** The card quotes no temperature margin (section 3.6); its flips
  still show. A Temperature limits flip is the dashboard's own verdict on the stepped design, so it says nothing the
  dashboard would not say after the step. The temperature edge shows one rendered for G-3 (10.1 step 2). Quoting no
  margin narrows 8A, whose words cover every failing check the step moves: T-13, owner question O-2 (answered: quote
  none), on the STOP sheet (code review 2 MF-2).
- **R-10, parity.** No ported row changes; gate 12 must run and not SKIP.
- **R-11, the jump rule's gap** (O-1). Closed by the widened rule (O-1 answered: widen; 3.3 step 7). For the record:
  under 9A as written the short cavity's `+1 mm` side printed -6.2 % for +1 mm, a slope-like number for a jump smaller
  than the step.
- **R-12, a third torque on the Size tab** (T-11). "cold-high" in the gearbox phrase names the torque the check
  compares; G-3 decides whether it stays.
- **R-13, the boss plateau** (code review 3 S2). Where the shaft boss sets the Outside diameter (`metal.boss_od_mm`,
  slider 12-40 mm, raised above both the cup and the cap, all under Detailed inputs), the diameter slider's own mapping
  lifts the cup from its own size to the boss plus the step in one step (`requested_cup = requested -
  cap_offset.max(0)`, `engine/size_controls.rs:431-432`). At npole 8, ring 8.0 mm, cap 30 mm, cap thread 29 mm and
  boss 40 mm (the magnets fit; Outside diameter 40.00 mm, set by the boss; slider 40.00-85.41 mm, or 40.00-76.78 mm
  with the thread left at its default 41 mm, wider than the cap, for the same numbers: C, section 3e of `out-run6.txt`;
  physics review 4 SF-4, code review 4 SF-1), `-1 mm` reads "at the end of its range" and `+1 mm` lifts the cup
  +3.22 mm (37.78 -> 41.00 mm; ring 8.00 -> 9.49 mm, against 0.46 mm of ring for 1 mm of Outside diameter elsewhere
  at 8 magnets): the card reads "+1 mm -> +0.12 N·m (+7.5 %)" (+7.55 %); at ring
  8.5 mm, cup +2.14 mm and +3.98 % (C, section 3d of `out-run5.txt`; the same at the lowest Minimum OD). This is
  4.4's "jump read as a slope" on the diameter row, which the jump rule (length only) does not guard. With the default
  boss (22 mm) and magnets it does not arise: over every even count, at the slider's smallest diameter, the boss stays
  at least 9.76 mm under the larger of the cup and the cap (C). No diameter jump rule: that goes beyond 9A and needs
  the owner (`owner-decisions.md`, "Spec review round 3 rulings"). On the STOP sheet and in BL-074's evidence.

### 10.4 Not verified

- A real browser's frame time (section 7.2 measures it before release).
- The card's height in the shipped page: the open heights are measured in a stand-alone egui layout at the centre
  widths the shipped panel has (`layout.rs`); test 25 pins the pre-measure to the drawn height where the body is under
  its cap, and a capped body to the cap.
- The steel-slice figures of R-1 are in-model (`c_end` scaled to 0.225), not 3D. They rest on R-1's hypothesis that the
  steel end loss scales with the pole pitch, which BL-065 did not measure.
- The slope of the 3D end loss at 12.7 mm (R-1): BL-065's figures give its level at five lengths, not its slope there,
  so the length row's N·m in 3D is known to within a few percent near 12.7 mm, not better (physics review 2 SF-3).
- Why hot-low peaks near ring 10.2 mm at 8 magnets (the brief's open item; the card shows the effect, not the cause).
