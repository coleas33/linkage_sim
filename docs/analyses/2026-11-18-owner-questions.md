# Questions for the owner

Written 2026-11-18. Updated as answers and new questions arrive.

**Nothing here stops the work.** Every question has a default, written under it. The default stands until you say otherwise.

**To answer:** reply with the number and a letter, such as "1b, 3a, 7c", or a sentence. Skip any you do not care about.

**How "we will most likely keep things generic" was read:** the tools stay general-purpose. They are not fitted to your own coupling, shafts, gearbox or press. Questions that only matter for your own hardware are in part D, each with the generic default. Question 1 asks whether that reading is right.

**Where each question comes from:** the codes at the end of a question point to its source.

| Code | Source |
|---|---|
| A-n | The register of assumptions, `docs/analyses/2026-11-10-conservative-assumptions.md` |
| BL-n | The backlog, `docs/ai/backlog.yaml` |
| C-n, O-n, section numbers | The UX design, `docs/superpowers/specs/2026-11-09-magcoupler-ux-overhaul-design.md` |
| P-n, L-n | The plans for slice A and for slice B stage 1, in `docs/superpowers/plans/` |
| Three-letter codes (LGS-11, LFA-16) | The whole-project audit, `docs/analyses/2026-11-10-whole-project-audit.md` |

Contents: part A, six answers that change what is built next. Part B, eight things only you can try. Part C, 35 calls to make when convenient (questions 48, 49 and 50 were added there on 2026-11-22, questions 51, 53, 54 and 55 on 2026-11-23). Part D, five that matter only if the tools stop being generic.

Answered so far: 1, 2, 3, 4, 5 and 6 (2026-11-22). Each answer is written in bold under its question.

---

## A. Answers that change what is built next

### 1. What does "generic" mean for the calculator's wording?

**Answered 2026-11-22: (a).** The wording stays as it is: your machine is the worked example.

Your machine is the worked example today. The starting design is sized to its facts: a 5:1 gearbox rated 0.6 N·m and a 43 mm space. Labels on screen say "Wheel shaft temperature at fitting" and "Torque the coupling must carry for driving (at the wheel)". The layout stage rewrites the first screen's text, so now is the cheap time to change names.

- **If you do not answer:** nothing is reworded. Your machine stays the worked example and its figures stay editable defaults.
- (a) Keep my machine as the worked example, as now.
- (b) Keep the numbers, but use neutral names ("driving shaft", "driven shaft").
- (c) Neutral names, and the gearbox and shaft-joint checks start switched off once a later stage lets a user switch them off.
- (d) Not generic after all: fit it to my hardware. Part D then matters.

Source: your message of 2026-11-18; design O-8, C-19, section 10.2.

### 2. Two of the starting design's nine checks use magnetic field numbers computed for a different design. Is a note enough?

**Answered 2026-11-22: (b), "Compute for the starting design".** The reference engine's 3D routine was run once for the starting design and its numbers are stored with it, so "passes all 9 checks" rests on the starting design's own fields. The note stays, with other words: "Fixed 3D fields: starting or example design". The fields still do not follow an edit. Built as BL-080: the starting design now differs from the spreadsheet in its seven sized inputs and in fourteen stored 3D values (the four reverse fields and the ten slip-heating fields).

The check run's result (2026-11-22; a second run by another method agreed):

1. With fields computed for its own geometry the starting design still passes all 9 checks, with no caution.
2. The two checks that read the fields barely move: the magnet temperature limit goes from 93.1 to 94.9 °C against a 65 °C hot-day start (margin 28.1 to 29.9 °C), and the inner hub's fitting temperature keeps its 6.7 °C margin.
3. The field that sets the temperature limit would have to be about 23 % higher than computed before that check turns to a caution, and the field behind the hub fitting about 47 % higher before the shaft joints check fails.

As asked on 2026-11-18, before the answer (the stored numbers, the check run and the line named below are now as above):

The calculator cannot compute 3D fields itself. The stored numbers belong to the example design (10 magnets, 10.15 mm ring radius, 1.40 mm gap). The starting design is 8 magnets, 9.00 mm and 1.80 mm, and it says "passes all 9 checks". The two checks are the temperature check and the inner hub's fitting temperature inside the shaft joints check.

How much room there is: the verdicts hold if the true reverse fields are up to 10 % higher. At 25 % higher the temperature check turns to a caution. At 75 % higher the shaft joints check fails.

The Python reference engine can compute the fields for any design, and its 3D package is already installed here. A check run for the starting design is under way (it changes nothing in the tool). Its result will be added under this question.

- **If you do not answer:** the design and its checks stay. The layout stage puts the line "Fixed 3D fields from the example design" on the first screen; today it sits below the first screen at 1280 x 800. If the check run shows a verdict would change, the starting design is fixed first and you are told.
- (a) A note on the first screen is enough.
- (b) Store fields computed for the starting design. It then differs from the spreadsheet in four more stored values, and the note needs other words.
- (c) Build live 3D fields for any design. This is a project of its own.

Source: A-18.

### 3. The opening screen says "passes all 9 checks" and shows one amber line about the keyway wall. Keep it?

**Answered 2026-11-22: (c), "Standard 1.4 mm keyway".** Run through the nine checks first, as promised: every verdict and every margin is the same with 1.4 mm as with 1.7 mm. The wall goes from 2.25 to 2.55 mm and the amber line is gone. The standard (DIN 6885-1 and BS 4235-1, 3 x 3 mm key on a shaft over 8 up to 10 mm) gives 1.8 mm in the shaft and 1.4 mm in the hub, each +0.1 mm; the calculator's input is the depth in the hub. At the deepest the tolerance allows, 1.5 mm, the wall would be 2.45 mm. The example design keeps 1.7 mm. Built as BL-080.

As asked on 2026-11-18, before the answer (the run through the nine checks is now done, above):

The line reads "Wall past keyway: 2.25 mm; existing guidance is about 2.5 mm". No counted check judges it, so a first-time visitor sees a pass and an amber warning together. There are two ways to clear it.

- **Larger ring.** Ring radius 9.00 to 9.26 mm and cap thread 41.7 to 42.2 mm. It still passes all 9 checks (hot-low torque 1.448 to 1.463 N·m; gearbox input 0.510 to 0.515 N·m against 0.60). But the cap's skirt wall thins from 0.55 to 0.30 mm, which no check judges (question 4).
- **Standard keyway depth.** The spreadsheet's keyway depth is 1.7 mm, which matches no standard key. The standard depth for a 3 x 3 mm key on a 10 mm shaft is 1.4 mm. By hand arithmetic the wall goes from 2.25 to 2.55 mm, with the ring and the skirt untouched. This has not been run through the nine checks yet. The example design would keep 1.7 mm either way, so it still matches the spreadsheet.

- **If you do not answer:** the starting design stays and the amber line stays visible. A test holds it there.
- (a) Keep the design and the amber line.
- (b) Larger ring: 9.26 mm and a 42.2 mm cap thread (skirt 0.30 mm).
- (c) The starting design uses the standard 1.4 mm keyway depth.
- (d) Keep the design and hide the line on the opening tab.

Source: A-17, P-10, BL-064.

### 4. Should the calculator judge the cap's thin skirt wall, and below what thickness?

**Answered 2026-11-22: (b) in this list.** The owner's words were "4a,6a, go!". The letter is the controller's dialog's, which put its recommendation first: (a) there was an amber caution below 0.5 mm, which is (b) here, not this list's (a), "leave it unjudged". So: an amber caution below a minimum of 0.5 mm (the controller's offer), and the starting design's 0.55 mm skirt stays clear. Filed as BL-082 (risk: physics). Reading confirmed by the controller 2026-11-22: the owner's 4a was the dialog's amber caution. **Built 2026-11-22** on the branch `magcoupling/skirt-wall-caution` (not merged until its reviews, the gate and the smoke run pass): a skirt under 0.5 mm (`gui::geometry::MIN_CAP_SKIRT_MM`, register row A-22) draws one amber line on the Size tab, "Cap wall past its thread (skirt): 0.4900 mm; guide minimum is 0.5 mm." (at 0.49 mm), counted as no failed check; the starting design's 0.55 mm draws none and still reads "passes all 9 checks".

The skirt is the wall of the cap beyond its thread. No check looks at it: a design with a 0.15 mm skirt passes all 9 checks. The starting design's skirt is 0.55 mm, and nobody has confirmed that a wall that thin can be made.

A minimum between 0.55 and 0.70 mm forces a different starting design. A minimum of 0.70 mm or more cannot be met with stock-length magnets inside the 43 mm space (the design search found none). It would take custom-length magnets or a larger space.

- **If you do not answer:** the skirt wall stays unjudged and the starting design keeps its 0.55 mm skirt.
- (a) Leave it unjudged.
- (b) An amber caution below a minimum. Give the mm.
- (c) A failing check below a minimum. Give the mm.

Source: A-6; design section 12.4 item 1.

### 5. Do you want to see each stage of the redesign before it goes live, or after?

**Answered 2026-11-22: (a).** A change goes live once the full test gate, the browser smoke test and the independent reviews pass; the report comes after.

Every push to the main line publishes straight to colesorkness.com; a run takes 15 to 27 minutes. There is no preview step, so the first time you can open a stage in a browser it is already public.

The approved design says each stage stops for your OK. It also asks for your sign-off on a design addendum before each of the four later slices (the drawing, checks that act, the worked example, the lessons) and on each new lesson note. You then said to stop asking. Slice A (the starting design and the address bar) went live that way on 2026-11-17. The look (stage 1) and the layout (stage 2) are next.

- **If you do not answer:** each stage goes live once the full test gate, the browser smoke test and the independent reviews pass. You get a report with pictures afterwards. The design addenda and the lesson notes are checked by an independent reviewer, not by you.
- (a) As the default: live on passing tests and reviews, report after.
- (b) Send me pictures first; go live unless I object within a day.
- (c) Set up a private preview address, and wait for my OK on stages that change the layout or show new numbers.
- (d) Wait for my OK on every stage.

Source: design section 9.4; A-4; L-12.

### 6. Codex's October fixes for the linkage simulator were never merged. Bring them in, redo them, or drop them?

**Answered 2026-11-22: (a).** The branch stays parked and untouched. Its fixes are reused one item at a time, each after a fresh physics review; the two confirmed faults go first. Recorded in `docs/ai/04-memory.yaml`.

Codex's branch holds 78 changes, the last from 2026-10-11. It was built on the main line of 2026-10-06, and the main line has moved 528 changes since. Of the 33 linkage to-do items still open on the main line, that branch marks 24 fixed and 8 more in progress (the physics ones, each waiting for your review).

It holds fixes for two faults that are confirmed wrong on the live site:

- **A sliding joint's reaction.** A 10 N·m couple is read as a 10 N force. Four failing tests reproduce it. Slider-crank reaction numbers may be wrong.
- **The motion profile on a partial sweep.** On a sweep of, say, 28 to 65 degrees every sample lands in the acceleration ramp, and the torque it adds can have the wrong sign.

Neither the audit of 2026-11-10 nor this list had looked at the branch before today. Its own new items reuse numbers 53 to 71, which mean other things on the main line.

- **If you do not answer:** the branch stays parked and untouched. When a linkage item's turn comes, Codex's fix is read first and reused where a fresh physics review agrees with it. The two confirmed faults go first.
- (a) As the default.
- (b) Bring the whole branch over in reviewed batches now, before more calculator work.
- (c) Hand it back to Codex to bring up to date.
- (d) Drop the branch and fix everything fresh.

Source: branch `codex/physics-review-rework-2026-10-06`; LCS-06; LGE-03; BL-029.

---

## B. Things only you can try

### 7. Do the tools work in Firefox and Safari?

Every automated browser check has run in Chromium only. The calculator now rewrites the page address after each edit, and the address is about 2,700 characters long. Safari is said to refuse more than about 100 such writes in 10 seconds; that figure was never checked on a real Safari.

The try: open both tools, change one value in the calculator, reload, and see whether the design comes back and the pages look right.

- **If you do not answer:** Chrome and Edge are treated as the supported browsers. Nothing is tested anywhere else.
- (a) Chrome and Edge only is fine.
- (b) I tried Firefox: say what you saw.
- (c) I tried Firefox and Safari: say what you saw.
- (d) Add Firefox to the automated check.

Source: slice A review; design section 12.3; BL-013.

### 8. Do "Starting design" and "Example design" read as two clearly different designs?

On the live calculator, read the status line before and after pressing "Open the example design". It goes from "Starting design: passes all 9 checks" to "Example design: fails 4 of 9 checks". A rename touches about 17 reviewed sentences.

- **If you do not answer:** both names stay.
- (a) Clear enough.
- (b) Rename the spreadsheet's design, for example "Spreadsheet design".
- (c) Rename the opening design. Say to what.

Source: C-20; design section 12.4 item 8.

### 9. Once the new look is live: can the crimson brand mark be mistaken for the red of a failing check?

Crimson is the brand colour; red means a check fails. The design itself names "crimson beside red" as a risk and asks for your eyes on a failing design. When stage 1 is live (you will be told), open the example design, which fails 4 checks, and look.

- **If you do not answer:** the look stays as built.
- (a) Clearly different.
- (b) Too close: use less crimson.
- (c) Too close: change the brand colour.

Source: design sections 9.4 and 12.2; L-12.

### 10. On the linkage simulator's Custom 6-bar sample, do the actuator forces now agree with your hand check?

You reported values one to two orders of magnitude too small and a flat curve, and were checking by hand (input force x input travel = output force x output travel). No result was recorded. The same calculation matches your press spreadsheet to within 0.012 %, so it is probably sound, but the 6-bar was never signed off.

- **If you do not answer:** the force calculation stays and the 6-bar difference is not looked into.
- (a) It agrees; close it.
- (b) Still too small; investigate.
- (c) Not rechecked; drop it.

### 11. Does the spreadsheet you download from the calculator open cleanly in real Excel?

Only two code libraries have read the file so far. A repair prompt or a broken sheet would show only in Excel itself.

- **If you do not answer:** it is taken to open fine.
- (a) Opens fine.
- (b) Opens with a repair prompt or errors. Say which.

### 12. On the desktop linkage app, does dragging a saved mechanism file onto the window open it?

The code reads only the dropped file's bytes. The desktop build may be handed only a file path, in which case a drop does nothing. Nobody has tried it.

- **If you do not answer:** desktop drag-and-drop stays untested and unchanged. File > Open works.
- (a) It opens.
- (b) Nothing happens; fix it.
- (c) File > Open is enough.

Source: LGS-01.

### 13. Will you run the hands-on checklist for the payload weights feature?

The checklist is at the end of `docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`. It was meant to be the gate before release, but the feature is already live with no record that it was run.

- **If you do not answer:** the feature stays live on its automated tests alone.
- (a) Skip it.
- (b) I'll run it.

### 14. Should a dragged weight's gold arrow stay?

After a weight is dragged, its arrow is gold until the sweep is recomputed. The gold state is in the app and in the checklist, but the written spec names only green, red and grey.

- **If you do not answer:** the gold arrow stays, and the spec is left as it is.
- (a) Keep gold and add it to the spec.
- (b) Replace gold with grey.

---

## C. Your call, when convenient

### Magnetic coupling calculator

### 15. When should the torque lost at the magnet ends be checked against a 3D model with magnetic steel behind the magnets?

That is how the designs on screen are built. Two ideal-steel 3D checks say the model reads torque about 6.65 % high at a 12.7 mm magnet length, and about 20 % high for magnets of 2.9 to 4.25 mm. So short magnets look stronger than they are. The starting design passes its hot torque check by only 7.3 % (1.448 against 1.35 N·m).

- **If you do not answer:** it stays an unscheduled written caveat. The screen states no figure.
- (a) Leave it as a written caveat.
- (b) Schedule it after the redesign.
- (c) Do it next.

Source: BL-065.

### 16. Should the calculator check the back-and-forth torque swing during a slip against the gearbox's continuous rating?

Today only the single biggest peak is compared with the gearbox rating. Yet each 0.1 s slip reverses the gearbox torque about 33 times, which is 3.33e8 cycles over the assumed 2e7 slips per life. On the example design the swing is 0.566 N·m at the gearbox input, under the 0.6 N·m rating, but 0.651 N·m with the 15 % production allowance. So a design can pass on screen and still wear the gearbox.

- **If you do not answer:** no such check. A note on screen says repeated slips also load the gearbox in fatigue and nothing checks it.
- (a) Keep the written warning only.
- (b) Add the check, with the continuous rating as a new input.
- (c) Add the check against the existing rating box.

Source: BL-066.

### 17. Should the calculator check the two shaft joints for fretting under the repeated slip swing?

The shaft joint check is static only. A shrink fit micro-slips at its ends below its slip torque, and the keyway corner concentrates stress (roughly 2 to 3 times, unverified), over about 6.7e8 torque reversals per life at the defaults.

- **If you do not answer:** no check. The shaft joint note says nothing checks fretting.
- (a) Keep the written warning.
- (b) Add a caution only.
- (c) Add a counted check from a published criterion.

Source: BL-070.

### 18. Should the calculator take a separate shaft diameter for each side?

One 10 mm bore input drives both shaft joints' fit, key, joining clearance and the masses. A user whose two shafts differ gets one joint judged on the wrong size. This was waiting for your real shaft sizes; kept generic, it can be built with both defaults at 10 mm.

- **If you do not answer:** one shared bore, 10 mm, serves both joints.
- (a) Keep one shared bore.
- (b) Add a second bore input, default 10 mm.

Source: BL-071.

### 19. Should the hidden shaft-clamp section stay hidden, come back, or be removed?

You paused it on 2026-10-15 because your hubs use a keyed shrink fit. Other users may clamp. If it returns, its boss diameter (25 mm) disagrees with the drawn shaft boss (22 mm), and at 22 mm with the 10 mm clamp no screw size works, so that has to be fixed first.

- **If you do not answer:** it stays hidden, not deleted. Saved files and links keep its values.
- (a) Stay hidden.
- (b) Bring it back as a second way to hold the hubs, with the boss sizes linked.
- (c) Remove it for good.

Source: BL-060.

### 20. When a loaded file sets the gearbox shaft colder than -20 °C, should a joint that fails for another reason still show red?

Today the whole shaft joint check turns amber "unknown" and hides the failure. With a 10 µm hub fit it reads red "the fit slips: holds 3.48 against 7.65 N·m" at -20 °C, and amber at -21 °C. The cold shaft only matters to the other joint's fitting step.

- **If you do not answer:** the whole check goes amber below -20 °C.
- (a) Keep the whole check amber.
- (b) Keep the failing joint red; only the cold-shaft part reads unknown.

Source: BL-075 item 1.

### 21. Seven smaller problems found while designing the redesign are on no to-do list. File them?

They are:

1. A design note that can print two equal numbers.
2. The diameter control leaves the ring radius between slider steps.
3. One result text reads "Nominal only: hot test" on every design, passing ones included.
4. Another reads "Meets it nominally (no variation allowance)" on a design that passes all 9 checks.
5. The Size tab's Minimum OD setting is lost on reload.
6. No separate inputs for bearing drag and acceleration torque (the requirement is one number, 1.35 N·m on the starting design).
7. No short form of the share link (about 2,700 characters today).

- **If you do not answer:** all seven are filed in the backlog, unscheduled.
- (a) File all seven.
- (b) File some. Name the ones to strike.
- (c) Leave them unfiled.

Source: design section 12.4 items 2 to 7 and 9; C-24; C-27; P-13.

### 22. When the keyway wall prints as exactly "2.50 mm", should the amber "guidance is about 2.5 mm" line stop showing?

At a ring radius of 9.25 mm the wall computes as 2.4999999999999990 mm. The line shows and reads "Wall past keyway: 2.50 mm; existing guidance is about 2.5 mm", which looks like a mistake.

- **If you do not answer:** it is fixed by comparing the value as printed (rounded to 0.01 mm).
- (a) Compare the value as printed.
- (b) Compare with a small tolerance.
- (c) Leave it.

Source: BL-078.

### 48. Should the torque model be checked against a real-iron 3D solve before the starting design is called passing?

Added 2026-11-22 (it keeps the next free number, so the other questions keep theirs).

The 3D routine that gave the starting design its stored fields also returns a pull-out torque. For the starting design it is 1.34 N·m at 20 °C, against the calculator's 1.83 N·m: a ratio of 0.73. For the example design the ratio is 0.76 (2.17 against 2.85 N·m, both in the reference engine's README at Br 1.29 T; the Rust engine's 20 °C pull-out at Br 1.30 T is now 2.89 N·m, a ratio of about 0.75). The reference engine's README calls that 3D figure a lower bound for real iron, because the routine models the steel as mirror images only. No check reads the 3D torque. But the starting design passes its hot torque check by only 7 % (1.448 against 1.35 N·m), so "passes all 9 checks" depends far more on the torque model than on the stored fields.

- **If you do not answer:** nothing changes. The starting design keeps its name and its "passes all 9 checks", and the caveat stays in writing.
- (a) Leave it as a written caveat.
- (b) Check the torque model against a real-iron solve, together with the end-effect check of question 15, before the starting design is called passing.
- (c) Until that check is done, resize the starting design so its hot-low torque clears 1.35 N·m by a margin that covers the gap. Give the margin you want.

Source: BL-080 (a by-product of the 3D run for A-18); BL-065; question 15.

### 49. Should a check's row show a short form of its number, or only its state?

Added 2026-11-22 (the next free number). Since the layout stage each check is one row in the right column: a shape, its name and a word (OK, Check or Fails). The design asked for the check's own number in short form on the row too, such as "0.26 / 0.20 mm" on Running clearance; the numbers are one click away, in the check's Explain entry.

- **If you do not answer:** the row shows the shape, the name and the word.
- (a) Shape, name and word, as now.
- (b) Add a short form of the check's number to each row (nine new short texts, each with a physics review).

Source: UX design section 4.5; A-20; BL-081 plan, decision D-2.

### 50. Should the status line of a file or a link sit at the right end of the top bar?

Added 2026-11-22 (the next free number). The design puts the line that reports a saved or loaded file or a copied link ("Share link copied to the clipboard (2,690 characters)") at the right end of the top bar. Today it is a line under the bar, and part 2a keeps it there.

- **If you do not answer:** it stays under the bar.
- (a) Under the bar, as now.
- (b) At the right end of the bar (a long line wraps the bar onto a second row at 1280 points).

Source: UX design section 4.2; A-24; BL-085 plan, decision D-22.

### 51. Should "Set by Auto; turn Auto off to edit" stay under the slider Auto locks?

Added 2026-11-23 (the next free number). With Auto on, the slider of the size Auto sets cannot be dragged, and a line under it says so. Part 2b-i moves every other line under the Size controls into the Explain panel (the design's 4.3) and keeps this one under the slider.

- **If you do not answer:** it stays under the slider.
- (a) Under the slider, as now.
- (b) In the control's Explain entry only (a click on the locked slider shows it).

Source: UX design section 4.3; A-26; BL-090 plan, decision D-37.

### 52. Should the sentence on what the size controls move be cut where it repeats the control's own hover?

Added 2026-11-23 (the next free number). The sentence "Outside diameter moves the ring radius; the cap and its thread follow the cup. Overall length sets both rings' magnet length, and the housing follows. Gap and magnet count stay as you set them." was a footnote under the Size column; part 2b-i puts it in the Outside diameter and Overall length entries of the Explain panel. There its first two clauses repeat each control's hover word for word, and its last clause is false while Auto sets the magnet count, so it is left out then. Four hint lines (Minimum OD, Outside diameter, Magnet face gap, Magnets per ring) also say what their hover already says.

- **If you do not answer:** the whole sentence stays in those two entries (not while Auto sets the count), and the four hint lines stay.
- (a) As the default.
- (b) Cut the sentence to "Gap and magnet count stay as you set them." and drop the four hint lines.

Source: A-28; BL-090 review of 2026-11-23, findings PT-1 and PT-2.

### 53. What should a row's label hover say now that a click on the label shows the input's entry?

Added 2026-11-23 (the next free number). The hover over a Detailed inputs row's label ended "Click the label to highlight the results it drives that have an equation". Part 2b-ii-a makes the click select the input (the design's 5.2); the highlighting is the link "Highlight the results this drives" at the top of the input's entry. The old line would be false, so it goes.

- **If you do not answer:** the hover ends at its Default line (and its spreadsheet lines), with no line about the click.
- (a) As the default.
- (b) A last line such as "Click the label to show its entry".

Source: UX design section 5.2; A-31; BL-092 plan, decision D-53.

### 54. Should a teaching note's input links select the input, as a check's fix links now do?

Added 2026-11-23 (the next free number). In a check's entry a fix link ("Raise Magnet face gap") now selects the input, so the Explain panel shows the input's entry, and shows its row. A teaching note's "What to change" lines link to inputs too; part 2b-ii-a leaves them as they were: the row shows and the note stays in the Explain panel.

- **If you do not answer:** a note's input links show the row and keep the note.
- (a) As the default.
- (b) They select the input, as a fix link does (the note gives way to the input's entry; "Checks it touches" and Learn lead back).

Source: UX design section 5.2; A-32; BL-092 plan, decision D-55.

### 55. Should a trace started on the opening tab keep an off switch in view?

Added 2026-11-23 (the next free number). Part 2b-ii-a moves the trace from a row's label to the link "Highlight the results this drives" in the input's Explain entry; a second click on the link ends it. On the opening tab the trace draws its frames and no banner or Clear trace (as since part 2b-i). Once you select something else there, the trace stays on with no off switch in view: select the input again and click its link, or open another tab for Clear trace. An input whose results have no equation record (the drive torque) traces none, and on the opening tab nothing says so. Part 2b-ii-c, which keeps the left column the same on every tab, places the banner anyway.

- **If you do not answer:** as described, until part 2b-ii-c.
- (a) As the default.
- (b) While the selected input is the one traced, its entry also shows the banner and Clear trace under the link (no new text; a second copy of the banner).

Source: UX design section 5.2; A-30; BL-092 plan, decision D-52; plan review UXP-3.

### Linkage simulator

### 23. After the calculator's redesign, should the linkage simulator get a design pass of its own?

You ruled that the redesign covers the calculator only. Once the new look is live the site has three looks: the tools page in grey with a blue accent, the linkage simulator in its CAD dark theme, and the calculator in charcoal and crimson. The simulator's first view, wording and layout have never been reviewed. It also has a list of planned features (force margin, hydraulic cylinder sizing, duty cycle) with no order set.

- **If you do not answer:** the linkage simulator keeps its look and gets bug fixes only.
- (a) Leave it as it is.
- (b) The shared look only (colours and font).
- (c) A full pass like the calculator's: first view, wording, teaching.
- (d) Decide when the calculator's redesign is done.

Source: design O-1 and section 2.2; `docs/FEATURES.md`.

### 24. On the desktop linkage app, when should the automatic backup of unsaved work be offered back, and when deleted?

Today the backup file is never cleaned up, a backup beside a saved file is never offered, and File > New brings the "recover your work?" prompt back. This was found by reading the code, not by running it. It is the last open "can lose work" finding of the audit.

- **If you do not answer:** the backup is offered only if it is newer than the saved file, and deleted on save, on New and on a clean exit.
- (a) As the default.
- (b) Offer it at every start until dismissed.
- (c) Drop the desktop backup; keep it in the browser only.
- (d) Leave it as it is today.

Source: LGS-11.

### 25. Two linkage results are suspected wrong and wait for a physics review. Review them now, later, or never?

- **The motion simulation's drift number** is measured after the positions are corrected, so it can read smaller than the real drift.
- **The solve-for-target tool** says "SVD solve failed" when the target may simply be out of reach, and spots a jump to the other assembly by a rule of thumb.

(The sliding joint and the motion profile, which are confirmed wrong, are under question 6.)

- **If you do not answer:** both get a physics review after the calculator's layout stage, and are fixed if the review confirms them.
- (a) As the default.
- (b) Review and fix now.
- (c) Leave them.

Source: LCS-02; LCS-14.

### 26. What should a gas spring do when a sweep drives it to full compression?

The model has no end stop. The gas column is floored at 1e-10 m, so the force comes out about 2e9 times nominal for a 0.2 m stroke, and a sweep can print an absurd spike.

- **If you do not answer:** the sample is marked "out of stroke" and shows no number.
- (a) Mark it "out of stroke" and show no number.
- (b) Add a stiff end stop at a minimum length.
- (c) Cap the force at a set multiple of nominal.
- (d) Leave the spike.

Source: LFA-16.

### 27. When a motor is driven backwards, should its torque fade with speed as it does forwards?

The motor model gives full stall torque at any reverse speed. The motor sizing check derates with speed in both directions. The two disagree whenever a motor runs backwards.

- **If you do not answer:** both stay as they are and disagree in reverse.
- (a) Leave as is.
- (b) The model derates in both directions, like the sizing check.
- (c) The sizing check matches the model (full torque in reverse).
- (d) Give reverse its own curve.

Source: LFA-17.

### 28. When a link is selected, which joint should the "Joint Torques" button act on?

It pairs the selected link with an arbitrary neighbouring link, so the torque may land on a joint you did not mean.

- **If you do not answer:** the arbitrary pairing stays.
- (a) Leave as is.
- (b) Every joint on the selected link.
- (c) Ask which joint.
- (d) The joint nearest the click.

Source: LGC-17.

### 29. Should every length in the linkage app follow the one display unit you choose?

The property panel mixes three length conventions. Plot grid labels and hover tips ignore the chosen unit and lose sub-millimetre precision on the Y axis. So numbers do not match between panels.

- **If you do not answer:** one chosen display unit everywhere.
- (a) One chosen display unit everywhere.
- (b) Metres in data panels, the display unit in plots.
- (c) Leave the mix.

Source: LGC-11; LGC-20.

### 30. Should the linkage app's Health and Diagnostics sections be merged?

They show the same facts with different thresholds, colours and wording. A mechanism can read OK in one and warn in the other.

- **If you do not answer:** both stay, but share one set of thresholds and wording.
- (a) Keep both; share thresholds and wording.
- (b) Merge into one section.
- (c) Leave as is.

Source: LGC-14.

### 31. Should the "Parametric Study [WIP]" menu item and the grid "Spacing" setting be finished or removed?

The menu item is a permanently disabled checkbox. The Spacing setting is overwritten on the first frame. Both look like features and do nothing.

- **If you do not answer:** both are removed.
- (a) Remove both.
- (b) Finish the parametric study; remove Spacing.
- (c) Finish both.
- (d) Leave both.

Source: LGS-07; LGS-08.

### 32. When an imported DXF drawing holds shapes the app cannot draw, should it say how many it skipped?

Import skips unsupported shapes silently, and the overlay drops lines that cross the view edge. A drawing can look complete when it is not.

- **If you do not answer:** the app shows a count of skipped shapes.
- (a) Show a count of skipped shapes.
- (b) Refuse the file.
- (c) Keep silent.

Source: LGE-16.

### 33. A weight's mass cannot be typed above 1,000 kg. Raise the limit?

A heavier payload loads from a file but cannot be typed in the editor or the toolbar's mass box. The written spec names no upper bound.

- **If you do not answer:** the 1,000 kg limit stays.
- (a) Keep 1,000 kg and say so on hover.
- (b) Raise it to 100,000 kg.
- (c) No upper limit.

Source: BL-034.

### 34. Should the linkage app let you draw a link's shape by dragging on the canvas?

The drag tool was never built. Its two buttons were replaced by "Add rectangle" and "Add circle" on 2026-10-06, so nothing on screen reaches it, but its code is still there.

- **If you do not answer:** the leftover code is removed.
- (a) Remove the leftover code.
- (b) Build drag-to-draw.
- (c) Leave it.

Source: BL-044.

### The site

### 35. Do you still want the separate dark landing page in front of the calculator?

You asked for it on 2026-10-03. It was never started. The redesign gives the calculator its own look and a first view meant to explain the tool, which may cover the same ground.

- **If you do not answer:** it stays an unscheduled to-do behind the redesign.
- (a) Keep it as a to-do for after the redesign.
- (b) Drop it; the new first view is enough.
- (c) Show me a private preview first.
- (d) Build it now.

### 36. Should the tools page take the calculator's new look, and do you still plan a page of your own at colesorkness.com?

The tools page (colesorkness.com/tools/) is grey with a blue accent and reads "Engineering Tools: Two calculators that run in the browser". It has no name, contact or link to the source. The bare address only forwards to it, by a redirect kept temporary so you could put your own page there.

- **If you do not answer:** both stay as they are.
- (a) Leave both.
- (b) Give the tools page the new look.
- (c) The new look, plus a name and contact line.
- (d) I'll put my own page at the bare address.

### 37. Do you want a licence, a "check before you build" notice, or a feedback link?

The code is public on GitHub with no licence, so nobody else may legally reuse it. Both tools size real parts (shaft fits, magnet temperatures, actuator forces) for anyone who opens them. A stranger who finds a wrong number has no route to tell you.

- **If you do not answer:** no licence, no notice and no feedback link.
- (a) Leave as is.
- (b) Add a licence. Say which: MIT, Apache-2.0 or all rights reserved.
- (c) Add a short notice and a feedback link on the tools page.
- (d) Both (b) and (c).

### Housekeeping

### 38. Is the old Python version of the linkage simulator still wanted?

The Python tree holds unused force elements and functions, and 19 MB of generated GIFs that nothing uses. The README and the architecture documents still describe a Python design that was never built. Some Rust tests still compare against Python-made reference data.

- **If you do not answer:** Python is kept untouched and its documents stay out of date.
- (a) Keep everything.
- (b) Keep Python as a reference only; delete the GIFs; rewrite the documents for Rust.
- (c) Delete the Python tree and the GIFs; rewrite the documents for Rust.

Source: PY-01; PY-02; PY-10.

### 39. Should the audit and fix automation run from any folder, and its local helper files be committed?

Both automation scripts hard-code the main checkout's path, so they cannot run from a side checkout. Two agent definitions, the hooks and one skill exist only on your machine, so a fresh clone cannot follow the project rules that name them.

- **If you do not answer:** the path is fixed; the helper files stay local.
- (a) Fix the path only.
- (b) Fix the path and commit the helper files.
- (c) Leave as is.

Source: INF-06.

### 40. May the repository be tidied?

Three things, none of which changes the tools:

- Delete the 20 local branches that are already merged.
- One formatting-only commit of about 136 linkage files, with a format check added to the test gate. It touches nearly every linkage file, so it must land when no other branch is open, and it makes line history harder to read.
- Archive 17 old plan and spec files that nothing links to (1.76 MB).

Codex's two folders are not touched.

- **If you do not answer:** nothing is deleted, reformatted or archived.
- (a) Leave everything.
- (b) All three.
- (c) Branches and formatting only.
- (d) Branches only.

Source: audit section J; DMH-03.

### 41. May the leftovers that are not merged be cleared too?

They are five draft branches of the payload weights work (2026-09-29), one stash of a superseded fix attempt (2026-11-09), five old branches on GitHub (all merged, and visible on the public repository), and two spare checkout folders that hold merged branches and no uncommitted work. The finished payload feature and the finished fix are both in the main line, but the drafts' own changes are not. Deleting a draft or a stash cannot be undone.

- **If you do not answer:** all of them stay.
- (a) Leave them.
- (b) Delete all of them.
- (c) Delete everything but the stash and the drafts.
- (d) Delete the GitHub branches only.

### 42. The write-up of your own press linkage has sat uncommitted since May. Commit it or keep it out?

The file is `docs/analyses/2026-05-28-press-4bar-analysis.md`. The repository is public and the file holds your press's geometry and load. It shows as an uncommitted file in every session and is never touched.

- **If you do not answer:** it is left where it is, uncommitted and untouched.
- (a) Leave it as is.
- (b) Keep it local and add it to the ignore list.
- (c) Move it out of the repository.
- (d) Commit it.

---

## D. Only if the tools stop being generic

Each of these has a generic default that stands. Answer only if you want the tool to describe your own hardware.

### 43. Which line of your gearbox's datasheet is the 0.6 N·m rating?

Input or output side, and continuous, acceleration peak or emergency-stop peak. An output-side figure would make the gearbox check 5 % conservative. A continuous rating is the number a repeated-slip check (question 16) would compare against.

- **Generic default:** the rating is a number the user enters ("gearbox input torque rating", default 0.6 N·m). No text on screen calls it a peak or a continuous rating.
- (a) Leave it generic.
- (b) Input side, acceleration peak.
- (c) Input side, continuous.
- (d) An output-side figure (divide by the 5:1 ratio).

Source: BL-059.

### 44. Do you want the shaft joint check set up for your own parts?

The check runs on placeholders: both shafts 10 mm, fits of 13 to 18 µm (inner hub) and 15 to 22 µm (cup), annealed 4140 at 415 MPa yield, a gearbox shaft no colder than -20 °C, and a 0.05 mm bondline. Real figures would move each joint's holding torque and fitting temperature. A pre-hardened steel raises the allowed tightness.

- **Generic default:** the placeholders stay as editable defaults.
- (a) Keep the generic placeholders.
- (b) I'll send my figures.

Source: BL-064; BL-071.

### 45. Should the calculator model your UltraOx finish with a masked bore?

The materials figures allow 0.030 mm on the bore diameter for electroless nickel; a masked bore needs none. The finish's process temperature would also cap the bare cup's fitting temperature (300 °C default, 400 °C slider top). It would need the finish's layer thickness, whether it builds up or converts the surface, and its process temperature.

- **Generic default:** electroless nickel only; the shaft joint check uses the bare bore.
- (a) Keep nickel only.
- (b) Add a generic coating thickness with a masked-bore switch.
- (c) Add UltraOx as a named finish. Send its data.

Source: BL-069.

### 46. For your own press linkage: is the crank range you gave the real working stroke, where is the output link's centre of gravity, and should the load box be shrunk?

These affect only your press sizing figure, not the simulator. The actuator travel you quoted is shorter than the travel over the full sweep you described. A centre of gravity out toward the tool head adds gravity load at the weak end of the stroke. And in your press model the load box is much bigger than the tool, so the full load acts over the whole sweep. (A force zone already acts only where the body overlaps its box, so a smaller box would say where pressing starts and stops.)

- **Generic default:** the simulator is unaffected. The press figure stands on the crank range you gave, mid-link centres of gravity and the full load at every angle, which is the safe side for sizing an actuator.
- (a) Leave the press figure as it is.
- (b) I'll send the real stroke, the centre of gravity and where the tool presses.

### 47. Which motion controller, if any, should the linkage app's trajectory export target?

Each controller format (G-code, Aerotech, Beckhoff, Galil) is about 150 lines of work, and is only worth building once a real controller is chosen.

- **Generic default:** JSON only.
- (a) JSON only.
- (b) G-code.
- (c) A named controller. Say which.

---

## Not asked, and why

- **Approved on 2026-11-16:** the calculator's own theme, the staged release, the share link's long form and the other conservative answers of that day (register rows A-1 to A-16), and the fix diffs BL-058, BL-061, BL-062, BL-063 and BL-074.
- **Code structure choices** in the plans (P-1 to P-9, P-11, P-12, L-1 to L-14): they change nothing a user sees that the approved design did not already settle.
- **Plain bugs and dead code** in the audit: they are fixed as their turn comes and need no answer.

## What the sweep did not read

Six readers went through the register, the design, the plans, the backlog, the audit and the working notes; a second pass merged their 68 candidates and a third looked for what was missed. They did not read end to end: the two long plans (searched for decisions and owner mentions), `docs/ai/05-update-tracker.md` (searched by keyword), the audit's per-area detail files, and `magcoupling-rs/README.md` (searched by keyword). The older linkage backlog items BL-004 to BL-052 were skimmed by title; three physics ones (BL-029, BL-036, BL-050) may hide choices.
