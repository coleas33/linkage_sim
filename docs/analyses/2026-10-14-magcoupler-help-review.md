# Magcoupler guidance slice 2: help and hint review (G-3)

One fresh-context session-model physics review per batch of the on-screen help and hint lines
(`magcoupling-rs/src/gui/plain_text.rs`), as spec
`docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md` section 4 asks for help, labels and
hint lines (slice-1 decision G-3). Each section below is the reviewer's report as returned, after the
must-fix items of its rounds were applied. The sheet each review read is written by the ignored test
`help_for_review` (`HELP_REVIEW_GROUPS`, `HELP_REVIEW_OUT`). A batch applies its must-fix items only;
the should-fix items a report offers stay open unless its section says otherwise.

## Batch A (T7)

**Verdict:** PASS at round 2. Round 1 returned 8 must-fix items (4 helps, 4 hints). They were applied
verbatim, and round 2 found none.

**Round 1 must-fix items** (old -> new, and why):

1. `temperature.duty.hot_ambient_C`.
   - Old: "Air temperature around the coupling on the hottest day. With the driving rise it sets the
     hot-day magnet temperature, so higher leaves less temperature margin; hot-low torque ignores it."
   - New: "Air temperature around the coupling on the hottest day; with the driving rise it sets the
     hot-day magnet temperature. Higher lowers the hot-day pull-out and temperature margins, not
     hot-low torque."
   - Why: the old text implied no torque moves, but the hot-day pull-out check falls to "Below it" at
     about 68.7 °C ambient at the example design.
2. `coupling.npole`.
   - Old: "... On the same ring, fewer blocks leave more space between them and usually lose torque,
     while more may not fit."
   - New: "... At the same ring radius, fewer usually lose torque and, if flat, enlarge the cup; more may
     not fit."
   - Why: the slice-1 trap that fewer poles keep the same size. The cup grows 1.11 mm from 10 to 8
     magnets and 2.59 mm from 8 to 6.
3. `coupling.magnets.manual_inner_thickness_mm`.
   - Old: "Thicker raises torque with shrinking gains" -> New: "Thicker usually adds torque, with
     shrinking gains".
   - Why: at 8 manual magnets per ring the pull-out peaks near 7.4 mm and dips to 9 mm.
4. `coupling.bore_mm`.
   - New: "The shaft diameter under the hub and clamp. Larger lets each screw carry more torque but
     moves the screws outward, so fewer sizes may fit, and thins the hub wall."
   - Why: the old text named only the clamp benefit. At the example design the clamp reads "None" from
     10.6 mm.
5. Hint `metal.cup_wall_corner_mm` (seed S-13, accepted).
   - New: "Thicker passes the wall check more easily but grows the cup."
   - Why: the check uses the design flux density, not saturation.
6. Hint `coupling.backiron` (seed S-13's half replaced).
   - New: "Magnetic steel behind the magnets adds torque if thick enough."
   - Why: a 304 back iron is a steel that adds nothing, and the old text said "saturate".
7. Hint `coupling.magnets.axial_length_mm` (seed S-14, reworded).
   - New: "Blank keeps each ring's length; longer adds torque, length and clamp demand."
   - Why: at the example design lengthening turns the clamp red before the space claim. "Clamp load"
     reads as preload, and manual rings have no part.
8. Hint `size_tab.overall_length` (seed S-14, reworded).
   - New: "Sets both rings' magnet length; longer adds torque and clamp demand."
   - Why: one 0.1 mm step up at the example design turns the clamp to "None".

**Round 2 must-fix items:** none.

**What was checked:**

Round 1 checked all 50 helps and 13 hints against the engine at 18 designs: the five probe designs,
8 magnets at a 20 mm ring, a 4.4 mm gap, true arcs, 12 magnets, and nine manual-dimension designs.
- One slider step each way, diffing every result and check text.
- Full-range sweeps of the operating temperature, ambient, magnet count (4..40), face gap, bore, clamp
  length and the four manual block dimensions.

Round 2 re-checked the eight changed strings at 24 designs: those 18, plus 6 magnets, the Back iron
toggle off, and 1018, 416 and 6061 back irons.
- Full-slider sweeps of the ambient, magnet count, inner thickness, bore, cup wall, axial length and the
  Size tab's overall length.
- The Back iron toggle against all eight back-iron materials.
- The magnet count on 20, 25 and 30 mm rings at wide gaps.

Every claim held. The two hedges ("usually") cover the only exceptions found:
- the inner-thickness dip at 8 manual magnets;
- about 2 % more torque with one magnet fewer at the highest count that fits on 25 and 30 mm rings
  with a 4.4 mm gap.

The trap lists (slice-2 spec 8, slice-1 spec 8, T-A..T-P, and "more magnets make the coupling stiffer")
were read against every string. The round-1 hits were fixed:
- the slice-1 "same size" (item 2);
- T-K saturation wording (items 5 and 6);
- T-E, kept by naming the hot-day pull-out apart from hot-low (item 1).

No string says that:
- the coupling protects the gearbox;
- the clamp is sized to the drive;
- the end factor is accurate;
- torque grows as R_g²;
- the outside diameter grows with fewer magnets.

**Should-fix, offered to the implementer:**
- name the cup growth in the magnet-count hint;
- name the clamp demand in the face-gap help and hint;
- a clearer subject in the inner-thickness help;
- for T8, the slip heating of a thicker non-magnetic cup.

### Round 1 report (as returned)

**Verdict:** round 1: MUST_FIX (8 items). Final verdict pending the implementer's round-2 sheet.

**Round 1 must-fix items** (old -> new):

1. `temperature.duty.hot_ambient_C`: "... so higher leaves less temperature margin; hot-low torque
   ignores it." -> "Air temperature around the coupling on the hottest day; with the driving rise it sets
   the hot-day magnet temperature. Higher lowers the hot-day pull-out and temperature margins, not hot-low
   torque." Why: the hot-day pull-out check flips to "Below it" at about 68.7 °C ambient at the example
   design; the old text implied no torque moves.
2. `coupling.npole`: "On the same ring, fewer blocks leave more space between them and usually lose
   torque, while more may not fit." -> "... At the same ring radius, fewer usually lose torque and, if
   flat, enlarge the cup; more may not fit." Why: slice-1 trap (fewer poles keep the same size); cup
   +1.11 mm at 10 -> 8, +2.59 mm at 8 -> 6.
3. `coupling.magnets.manual_inner_thickness_mm`: "Thicker raises torque with shrinking gains" ->
   "Thicker usually adds torque, with shrinking gains". Why: at 8 manual magnets per ring the pull-out
   peaks near 7.4 mm and dips to 9 mm.
4. `coupling.bore_mm` -> "The shaft diameter under the hub and clamp. Larger lets each screw carry
   more torque but moves the screws outward, so fewer sizes may fit, and thins the hub wall." Why: at the
   example design the clamp reads "None" from 10.6 mm; the old text named only the clamp benefit.
5. Hint `metal.cup_wall_corner_mm` (seed S-13): -> "Thicker passes the wall check more easily but grows
   the cup." Why: the check uses the design flux density, not saturation.
6. Hint `coupling.backiron`: -> "Magnetic steel behind the magnets adds torque if thick enough." Why:
   304 is a steel and adds nothing; "saturate" dropped (T-K).
7. Hint `coupling.magnets.axial_length_mm` (seed S-14, reworded): -> "Blank keeps each ring's length;
   longer adds torque, length and clamp demand." Why: Auto's lengthening at the example design turns
   the clamp red (13.77 mm) with the space claim still inside; "clamp load" means preload; manual rings
   have no part.
8. Hint `size_tab.overall_length` (seed S-14, reworded): -> "Sets both rings' magnet length; longer adds
   torque and clamp demand."

**What was checked:** every one of the 50 helps and 13 hints against the engine at 18 designs. These
were the five probe designs; 8 magnets at a 20 mm ring; a 4.4 mm gap; true arcs; 12 magnets; and nine
manual-dimension designs, since the library-part probes hold the manual inputs. One slider step each
way diffed every result and check text; full-range sweeps covered the operating temperature, ambient,
magnet count 4..40, face gap, bore, clamp length and the four manual block dimensions. Results by group:

- Requirements: the requirement leaves hot-low untouched and moves the floor and the 20 °C requirement.
  The operating temperature lowers pull-out and hot-low monotonically from -40 to 150 °C at all 18
  designs, with cold-high unchanged. The drive torque and its factor move only the floor (the hot-low
  check ignores them). Ratio and efficiency lower the ripple and the cold-high torque at the gearbox
  input; the rating moves only the informational reference. Service life moves only the daily-cycle
  count. Daily swing, slip speed and fault trip move as stated.
- Magnets: part, width, Br and outer length behave as stated. The outer thickness has one peak (4.9 to
  8.8 mm) at every manual design. The ring radius raises the gap radius, the cup and the spacing; its
  helps claim no torque direction.
- Gap: monotone at all 18 designs, the outer blocks fit across the whole slider, and the clearance
  moves one for one. Bedding moves the clearance and never the torque. The six allowances subtract.
- Shaft and clamps: keyway, key width (no result moves; only the drawings read it), key contact, clamp
  length, slit, ligament, wall, grip, margin, relief (layout text only), two-piece factor, engagement
  and stripping factor behave as stated.
- Hints: part, operating temperature, conductance, measured drag, Minimum OD and Outside diameter pass.

The trap lists (slice-2 spec 8, slice-1 spec 8, T-A..T-P, "more magnets make the coupling stiffer") were
read against every string. The hits are must-fixes 2 (slice-1 "same size"), 5 and 6 (T-K), and 1
(T-E kept: the hot-day pull-out named apart from hot-low). No string says the coupling protects the
gearbox, the clamp is sized to the drive, the end factor is accurate, or torque grows as R_g².

**Should-fix, offered to the implementer:** face-gap clamp demand; "daily-cycle screen"; efficiency as
a power share; compare the gearbox rating with the cold-high input torque; the ring radius can raise or
lower torque; the space claim leaves out the shaft boss; "with Auto on"; the magnet rating at
80/60 °C parts; Br raises the wall needed; and "can fit" for the clamp length.

### Seeds and amendments (round 2, as returned)

- A-24 S-13, cup-wall half: ACCEPTED, applied verbatim. The check compares the wall with the rule at
  the design flux density, not saturation; probe-confirmed (item 5).
- A-24 S-13, back-iron half ("Steel behind the magnets adds torque if its wall is thick enough."):
  REJECTED as worded. "Steel" is false at the 304 probe, a steel that adds nothing. The applied
  replacement keeps the seed's thickness caveat and is confirmed (item 6).
- A-24 S-14 (both hints): ACCEPTED IN INTENT, REJECTED AS WORDED. The clamp is what goes red, so the
  intent stands, but "clamp load" means screw preload to this audience. "Each part's" is also wrong on
  manual rings. The applied "each ring's ... clamp demand" texts are confirmed (items 7 and 8).
- A-21, the only text amendment indexed for T7: still applied verbatim. The `clamps.preload_fraction`
  row equals A-21's text and appears once in the source and once in the sheet.
- A-19, A-20, A-22, A-23 are other tasks' texts. The changed T7 rows agree with them:
  - the npole help matches A-19's "fewer flat blocks grow the cup";
  - the axial hint matches A-23's "each ring's own" and its cold-high/clamp trade-off;
  - no changed row touches A-20's or A-22's claims.
- A-13: every changed drawn field has no audit, correction or spreadsheet id and no file or code name.
  Batch A has no note sources.

## Batch B (T8)

- Scope: the 52 empty helps of Calibration and model, Materials, Housing and retainers, and Thermal
  and demagnetization, and 31 overrides (tables A and B of the slice-2 plan, Task 8).
- Read: T7's review sheet for the four groups, T2's overlay dump for the six rows outside them,
  `batch_b_tests::CLAIMS` (172 direction claims, all holding), spec section 8 and slice-1 spec
  section 8, and the engine code behind each claim. Each round also probed the engine from a
  scratch crate outside the repository: round 1 stepped every batch-B input one slider step up and
  diffed every result, and re-ran the claims on 14 designs beyond the example (304 and 6061 back
  irons, 8 magnets per ring, Y30 grades with and without the coercivity inputs, SmCo grades, manual
  magnets with no grade, no back iron alone and at 8 magnets, a 1018 back iron, a 20 mm ring radius,
  the coercivity inputs, a measured drag, class 10.9, an axial length set, true arcs).
- Round 1: 4 must-fix items, text applied verbatim:
  - `calibration.mu0` (seed S-17): the unit identity "(H/m equals T·m/A)" dropped. New: "Magnetic
    constant for the prototype model; leave it unchanged. It sets the updated calibration
    coefficient, used only by no-back-iron designs with the prototype's magnet part and count."
  - `temperature.duty.driving_rise_C`: "where slip events start" -> "the hot-day starting magnet
    temperature". New: "Rise of the coupling above outside air while driving, from housing air, sun
    and gearbox heat; it raises the hot-day starting magnet temperature. The default is a guess."
    Why: the old phrase named no result and read as a slip threshold. The rise adds to the hot-day
    starting temperature, which lowers the hot-day pull-out and margin (+0.5 °C at the example
    design: 65.0 -> 65.5 °C, hot-day pull-out 2.589 -> 2.586 N·m). A claim on the hot-day start
    was added.
  - `metal.cap_od_mm` (T-P): "above the cup's, it sets the outside diameter" -> "when larger than
    the cup and boss, it sets the outside diameter". New: "Front cap diameter; when larger than the
    cup and boss, it sets the outside diameter. Size-tab diameter changes and Auto on the ring
    radius move it with the cup." Why: the Size tab's Outside diameter is the largest of cup, cap
    and boss; at a 7 mm ring radius with the cap at 36 mm and the boss at 38 mm, a cap step changes
    nothing. Two pins added: a cap step moves the Outside diameter when the cap is the largest part
    and leaves it at a larger boss.
  - `materials.steel.specific_heat_J_kgK`: "Picking another back-iron material replaces it" ->
    "A magnetic back-iron pick replaces it". New: "Specific heat of the steel parts, including keys,
    screws and lock tabs; higher slows slip heating, not the steady temperature. A magnetic
    back-iron pick replaces it." Why: a 304 or 6061 pick leaves this input on the keys, screws and
    lock tabs (heat capacity +0.006 J/K per step). A "304 back iron" claim design and its Up row
    were added. Round 2 corrected the last sentence again.

  Should-fix offered, not applied: S-a, `coupling.mu0`'s "every torque" leaves out the prototype
  and its demagnetization and slip-heat uses; S-b, name "magnetic cycles over life" in the slip
  event help; S-c, "Squared-field integral" for the cap; S-d, a ferrite ring's hot-side limit does
  not move with the knee fraction or the design margin (the cold limit does); S-e, for the
  controller, override `metal.cap_thread_dia_mm` ("M41 × 0.5 concept.", stale after a resize) and
  `materials.screws.proof_12_9_MPa` ("ISO 898-1."); S-f, for the code review, the cluster caveats
  typed in every row (resolved in T8's review fix round 1: each caveat is typed once, as a macro
  the rows are built from; no text changed).

  Recorded, no change asked: the single-ring field's "lowers the cure margin" is moot for ferrite
  (the margin is infinite on the cold side); with the Back iron toggle off and a non-magnetic pick,
  the hub, cup and boss take the pick's values, not the aluminium inputs; some engine helps in batch
  B's groups outside its 31 still read terse (`coupling.c_end`, `calibration.f_cal_original`,
  `temperature.thermal.conductance_W_K`, `calibration.fea_torque1_Nm`).
- Round 2: 1 must-fix item, text applied verbatim:
  - `materials.steel.specific_heat_J_kgK`: "... A magnetic back-iron pick replaces it." -> "Specific
    heat of the steel parts, including keys, screws and lock tabs; higher slows slip heating, not
    the steady temperature. Picking another magnetic back-iron material replaces it." Why: the
    picker's default, "4140 annealed", is a magnetic pick that keeps the input (heat capacity +0.12
    J/K per step at the example design); only the five other magnetic picks replace it. No `CLAIMS`
    change: the example, 1018 and 304 rows pin the new wording.
  - The other three round-1 replacements were confirmed as applied (`calibration.mu0`,
    `temperature.duty.driving_rise_C`, `metal.cap_od_mm`); their claims hold at every probed design.

  Should-fix offered, not applied: S-1, one name for the hot-day starting temperature in T7's
  ambient help; S-2, the cup wall adds slip heat with no back iron or a non-magnetic pick; S-3, a
  third cap pin with the cup as the largest part.
- Round 3: none. The round-2 replacement for `materials.steel.specific_heat_J_kgK` was confirmed as
  applied, and it is exact on all eight back-iron picks with the toggle on and off: the default 4140
  and the non-magnetic 304 and 6061 keep the input; 1018, 12L14, 416 and both 17-4PH picks move no
  result; the high-case steady temperature never moves.

  Should-fix offered, not applied: S-1, name the 4140 hub, cup and boss, so that a 304 user does not
  read the input as their hub's (at 304 only the keys, screws and lock tabs take it, +0.006 J/K per
  step); S-2, the aluminium specific heat and density do not reach the hub, cup and boss at the 304
  and 6061 picks (no change proposed).
- Seeds: A-24 S-17 (`calibration.mu0`) ACCEPTED, applied verbatim in round 1. Reason: the unit
  identity is notation, not physics the user acts on; the seed keeps the prototype caveat last,
  states no new direction and fits the cap (27 words).
- Amendments: A-7 applied (all eleven stored-3D helps end with "Stored 3D value for the example
  design; it does not follow yours."; the hub, cup and web fields keep "doubled by the steel", and
  the aligned and pull-out reverse fields "which no check uses");
  A-6 applied (Step 4c dropped; T2's all-rows test admits a result key); A-13 met (no audit,
  correction or spreadsheet id and no file or code name in any drawn field; batch B has no note
  sources). Ruling 5: no "Corrected note" toggle line; the three engine gaps (the space claim leaves
  out the boss, ferrite eddy heat uses the NdFeB conductivity, the daily-cycle screen reads the
  recommended bondline) go to T20 as BL-061..BL-063, and the helps state today's behaviour.
- Verdict: PASS.

## Engine wording (T9)

- Scope: not an overlay batch but the engine's own wording, reviewed under the same G-3 rule. The
  96 on-screen strings T9 rewrote:
  - every Rust-only input and result help, and three labels (two with cell ids, and the autofit
    label);
  - the help of the five ported results and the screw-length column whose correction already
    rewords it, and the sweeps' swept-variable column;
  - the harmonic set's default choice, `1, 3, 5 (default)`;
  - the stored-3D label, note and sentence (`gui::dashboard`).
- Read: the string dump (`git diff -U0 -- magcoupling-rs/src`) and a review sheet of the 96 strings
  (old, new, word and sentence counts), not `help_for_review`; T9's brief and drafter notes; the
  amendments (Global Constraints, A-7, A-13, A-16, A-19 to A-24); slice-2 spec 3.2, 3.5, 3.8 and 8,
  slice-1 spec 8 and `phys-new-notes.md` section 2; the engine code behind every changed string.
  Both rounds probed the engine from scratch crates outside the repository:
  - round 1: the five probe designs; grid A, 945 designs (4 to 40 magnets per ring, ring radius 9
    to 30 mm, face gap 0.3 to 5 mm, the library B842SH or six manual cross-sections), each with
    the back iron on and off (1890 runs); grid B, 18,000 designs over the same ranges;
  - round 2: the two changed fields at the five probe designs, eight grade states, and 68,688 runs
    at the bench prototype's magnet count and part.
- Round 1: 2 must-fix items, text applied verbatim:
  - `materials.parts.back_iron` help (seed S-15, accepted).
    - Old: "The hub, cup and boss material. 4140 uses the steel inputs; another steel brings its own
      library values, and 304 or aluminium removes the steel return path, so torque drops."
    - New: "The hub, cup and boss material. 4140 uses the steel inputs; other steels bring library
      values, some tripping the saturation caution; 304 or aluminium drops the steel path, losing
      torque."
    - Why: the old text set "another steel" (no warning) against the one pick that loses torque,
      which invites spec 8's clean-swap reading ("416, 17-4PH or 12L14 are clean swaps") and "steel
      gives the torque shown" without the caveat. 12L14, 416 and both 17-4PH tempers trip the
      low-saturation caution at every design with the back iron on (the rule reads no geometry);
      4140 and 1018 do not, so "some" is exact. 304 and 6061 lose torque on all 945 grid-A designs
      (pull-out ratio 0.138 to 0.9999).
  - `model.inner_alpha_br_per_C` help.
    - Old: "The picked grade's coefficient for manual dimensions with a grade; otherwise the shared
      Br temperature coefficient input, the same for both rings."
    - New: "The picked grade's coefficient for manual dimensions with a grade; otherwise the shared
      Br temperature coefficient input, which every ring without a picked grade uses."
    - Why: each ring takes its own grade's coefficient only in grade mode. With the library B842SH
      inside and Y30 on manual dimensions outside, the rows read −0.0012 and −0.002 /°C, beside a
      hover that said "the same for both rings". The new text agrees with A-22 and T8's
      `calibration.alpha_br_per_C` overlay, and closes spec 8's "each ring uses its own α" both ways.

  Should-fix offered, not applied (open for the controller):
  - S-a, `coupling.max_harmonic` help: "The torque model sums odd field harmonics up to this one,
    for your design and the bench prototype. Higher ones matter most for narrow, widely spaced
    blocks." (27 words). The current "Each higher one fades faster across the gap." is true at every
    design but names no case where the set matters: going from harmonic 1 to the 1, 3, 5 set changes
    the pull-out by +33 % at 4 magnets per ring and +31 % at a 30 mm ring radius.
  - S-b, `materials.cup_wall_suggested_mm` help: "The back-iron wall rule's thickness at the back
    iron's design flux density, rounded up to 0.1 mm. The cup wall check asks for it; n/a without
    back iron." (28 words). Why: T-I, the rule's error depends on the pole count (1.92 mm at the
    defaults against 2.17 to 2.27 mm from exact solutions; 2.40 mm at 8 magnets against 2.17 mm
    from the flat-top form), and the label already says "the rule asks for".
  - S-c, `STORED_3D_NOTE` (A-7's text, so the controller's call): "The temperature limits use" ->
    "The temperature checks use". Only the reverse fields feed the limits; the slip-heating fields
    feed the slip-fault peak the verdict compares with them.

  Recorded, no change asked:
  - the grade's "it sets their ... coercivity" holds at the default coercivity source (the
    coercivity-source help names the override); `demag_ring`'s "each with its own grade": with the
    source on the inputs, only the rating and Br come from each ring's grade;
  - the free-space slip-field helps dropped the old "not doubled" (T8's steel rows say "doubled by
    the steel"); parallel wording would be a nicety only.
- Round 2: none. Both round-1 replacements were confirmed as applied and hold at every probed design:
  - `materials.parts.back_iron`: at the bench prototype's magnet count and part the free-space
    circuit takes the bench-matched factor (about 1.05) and the steel circuit the original 0.95, so
    the shown torque was checked too: 304 or aluminium loses it on all 68,688 runs (highest shown
    ratio 0.679).
  - `model.inner_alpha_br_per_C` holds in eight grade states, among them a library ring on an
    edited input (−0.0009 /°C) beside a graded manual ring keeping its grade's −0.0012 /°C, the case
    the old text got wrong.

  Recorded, no change asked:
  - calibration edits can make 304 show more torque than 4140 (at the default design a measured
    bench torque above 2.85 N·m does); that is spec 8's "edit the calibration inputs to match your
    design", and the jump between circuits is S-10's watch-out (T17), not a 30-word input help;
  - with the Back iron toggle already off, the back-iron help's clauses presuppose the steel
    circuit; a user there sees no torque change and no caution, and is not led to a wrong pick;
  - the coefficient help does not cover a grade typed while a library part is picked (the ring then
    reads the shared input, which the text's relative clause still states); "every other ring"
    would read as "the outer ring" on the inner row.
- Seeds:
  - A-24 S-15 (`materials.parts.back_iron`): ACCEPTED, applied verbatim in round 1. Reason: it names
    the caution four of the five other steels trip, which closes the clean-swap trap within the
    30-word cap; "some" is exact (1018 does not trip it) and the torque loss holds at every probed
    design.
  - A-24 S-16 (`coupling.max_harmonic`): REJECTED. Reason: "Past the fifth they add little" is false
    at reachable designs; T12's test pins under 2 % only at the five probe designs. On grid A, 458 of
    1890 runs move the pull-out by more than 2 % from the 1, 3, 5 set to the set up to 11, 295 by
    more than 5 %; the worst is +284 % (4 magnets per ring, 30 mm ring radius, 1 mm-wide blocks,
    0.3 mm gap, no back iron). S-a offers a true replacement.
- Amendments: A-7 applied (`STORED_3D_SENTENCE`, `STORED_3D_LABEL` and `STORED_3D_NOTE` byte-equal
  to A-7; the three free-space engine helps end with the sentence; at 8 magnets per ring the stored
  fields keep their values, so "it does not follow yours" holds and no string promises a live 3D
  model); A-23 applied verbatim (the pull-out, the cold-high torque and the axial stack rise at
  every step of the magnet length from 2 to 50.8 mm, 8550 points); A-16 applied (`tests/schema.rs`
  uses `engine::explain::plain`'s counters); A-13 met (no audit, correction or spreadsheet id and
  no file or code name in any changed string; T9 adds no notes or sources). A-19 to A-22 are other
  tasks' texts; must-fix 2 brings the coefficient help into line with A-22.
- Drafter flags: "NdFeB's 7.5 g/cm³" kept (the engine constant every non-grade magnet gets, not an
  editable default); "1, 3, 5 (default)" accepted (spec 3.8 names it).
- Verdict: PASS (round 2).

### Report for the record, round 1 (as returned)

- G-3 round 1 (T9 engine wording, 96 strings): MUST_FIX, 2 items.
  - `materials.parts.back_iron`: seed S-15 accepted. Other steels are named as tripping the
    saturation caution, against the spec 8 clean-swap trap.
  - `model.inner_alpha_br_per_C`: "the same for both rings" -> "which every ring without a picked
    grade uses". On a mixed design the two coefficients differ, −0.0012 against −0.002 /°C.
  - Seed S-16 rejected: past the fifth harmonic the pull-out moves by more than 2 % on 458 of 1890
    grid runs, up to +284 %.
  - A-7, A-16 and A-23 were applied verbatim. A-23's direction holds at 8550 probe points.
  - Should-fix S-a (when the harmonic set matters), S-b (name the wall rule) and S-c (A-7 note:
    "checks", not "limits") offered.

### Report for the record, round 2 (as returned)

- G-3 round 2 (T9 engine wording, the two fields changed in round 1): PASS, no must-fix.
  - `materials.parts.back_iron` (seed S-15, verbatim):
    - the other steels' saturation caution trips for 12L14, 416 and both 17-4PH tempers, not for
      4140 or 1018;
    - the torque loss of 304 or aluminium holds on 68,688 runs at the prototype's magnet count and
      part, where the free-space circuit takes the bench factor (highest shown ratio 0.679).
  - `model.inner_alpha_br_per_C` ("which every ring without a picked grade uses") holds in eight
    grade states. One is a mixed design with an edited input: −0.0009 on the library ring against
    −0.0012 on the graded ring.
  - S-16 stays rejected: past the fifth harmonic the pull-out moves +284 % (4 magnets per ring,
    30 mm, narrow blocks).
  - A-7 and A-23 are byte-equal to the amendments.
  - Round 1's should-fix items S-a, S-b and S-c stay open for the controller.

### Task review fix round 1 (2026-10-15): S-a and S-c applied

- The task code review (`task-9-review.md`: I-1, and M-2 with the coverage note) asked for two of
  round 1's should-fix items, and the fix round applied both verbatim:
  - S-c: `STORED_3D_NOTE` now opens "The temperature checks use the demagnetization reverse fields
    and the slip-heating fields, ...". A new test pins the fact behind it
    (`dashboard::tests::the_stored_3d_note_says_the_slip_fields_feed_the_checks_not_the_limits`).
    At their slider ends, the slip-heating fields move only the temperature verdict among the
    headline rows. The reverse fields move the governing limit and the hot-day margin. The note's
    sentence that names the slip-heating fields must not say "limit".
  - S-a: the `coupling.max_harmonic` help is S-a's text, and `tests/data/input_schema.json` is
    re-blessed (one help line). A new engine test pins its claim
    (`model::tests::higher_harmonics_matter_most_for_narrow_widely_spaced_blocks`; steel circuit,
    1.4 mm face gap). Going from harmonic 1, or from the 1, 3, 5 set, to every harmonic up to 11
    moves the pull-out by (relative; with the corrections the app uses, E7 on; the test's
    corrections-off values in brackets):
    - 11.27 and 1.16 (2.16 and 1.70) for 1 mm blocks at 4 magnets per ring on a 30 mm ring;
    - 0.045 and 0.034 (0.32 and 0.039) for 1 mm blocks at the default pitch;
    - 5.78 and 0.44 (1.12 and 1.11) for the default blocks on the long pitch;
    - 0.0071 and 0.00001 (the same) for the default blocks at the default pitch.
- S-b stays open.
- The G-3 re-check of the two changed strings ran: PASS, no must-fix (the re-review below).

### Re-review: code-review fix round 1

- Reviewer: a fresh-context physics and wording reviewer (session model), 2026-10-15, on commit
  `5379ea5` (`task-9-textreview-1.md`).
- Scope: the two strings the task review's fix round 1 changed, `STORED_3D_NOTE` (S-c) and the
  `coupling.max_harmonic` help (S-a), and every on-screen sentence whose meaning depends on them.
- Verdict: PASS, no must-fix. Both strings are byte-equal to G-3 round 1's S-a and S-c texts, both
  are true at every probed design with the corrections the app uses, and neither writes a trap.
- Should-fix, both applied after the re-review (no on-screen text changed):
  1. The harmonic claim test pinned the help with the corrections off only. It now runs as the
     app shows it (every correction on) and with the corrections off, and names the set in its
     message. At both sets its values match the figures in "Task review fix round 1" above, to
     their rounding.
  2. Those figures now give the app's values, with the corrections-off values in brackets
     (the reviewer's replacement text, verbatim).

The reviewer's report for the record, as returned:

#### Currency

- Both strings are byte-equal to the round-1 texts. This was checked by script against `task-9-inreview-r1.md`:
  - S-a: `model.rs:187` and the re-blessed schema help (`input_schema.json:186`) are both equal to it;
  - S-c: `dashboard.rs:212` is equal to it.
- `git diff --numstat HEAD~1 HEAD -- magcoupling-rs/tests/data/input_schema.json` reads `1 1`, which is only the help line. Against `ad992e1` it reads `11 11`, as round 2 recorded, so the re-bless swapped only this one line.
- The repo was clean before and after this review: `git status --short` shows only the untracked owner file `docs/analyses/2026-05-28-press-4bar-analysis.md`.

#### Tests run (in the repo; they wrote only `magcoupling-rs/target/`)

All pass:
- `--features app --lib stored_3d` (5 tests), which includes the new `the_stored_3d_note_says_the_slip_fields_feed_the_checks_not_the_limits`;
- `--features app --lib` filtered on `higher_harmonics_matter_most`, `each_cluster_ends_with_one_shared_caveat` and `harmonic` (21 tests);
- `--features app --test schema` (11 tests), which includes `input_schema_json_is_current` and `engine_owned_help_is_plain_and_short` (it names `coupling.max_harmonic` and checks the guard, `provenance_words`, `code_names` and the 30-word, 2-sentence cap).

#### Probes

- Location: a scratch crate outside the repo, `<scratchpad>/t9-rr1-probe`, with `magcoupling-rs` by path, `default-features = false` and `workbook-parity` (only for the corrections-off comparison). Every other run uses `compute_all`, as the app does.
- Bins: `src/main.rs` (the harmonic grid and the five probe designs), `src/bin/corners.rs` (the test's corners, corrections off and on), `src/bin/note.rs` (what the stored fields move) and `src/bin/y30.rs`.

#### `STORED_3D_NOTE`

Text: "The temperature checks use the demagnetization reverse fields and the slip-heating fields, computed in 3D for the example design. Those fields stay fixed when you change the geometry, magnet count or grade; for another design, enter its own 3D fields to check the limits."

- **Caps and words:**
  - 44 words in 2 sentences (19 and 25), as A-7 set it; the only change is one word.
  - `plain_text_violations` is empty (the dashboard test).
  - No cell, id, file or code name, no typed number, no "live".
- **"The temperature checks use the reverse fields":** true.
  - The reverse fields set the four onsets. The like-pole onset minus the design margin is the magnet limit (`temperature.rs:827-835`).
  - The single-ring onset gives the cure margin (`:978`, `:1382`).
  - On the cold side the reverse fields give the cold limit (`:845-856`).
  - All three reach the verdict's `ok` (`:1383-1388`).
- **"... and the slip-heating fields":** true while no drag is measured.
  - The slip fields set the losses (`:1120-1196`), then the peak with the fault (`:1335`), which the verdict compares with both limits.
  - Probe, each of the 10 slip fields doubled, at all five probe designs:
    - the governing limit and the hot-day margin never move;
    - the peak moves for 7 of the 10 (the steel-circuit or the free-space variants, whichever circuit is on).
- **The fix is right, and "limits" was wrong in more than one way:**
  - Slip fields: they never set a limit (probe above).
  - Y30 probe (a positive coercivity coefficient, E20 cold side):
    - the magnet limit is the grade's rating (250 °C) and the adhesive governs (120 °C), so the like-pole field ×1.2 moves no limit row;
    - it does move the cold limit (335.7 -> 361.1 °C), which the verdict reads;
    - so "checks" holds there too.
- **"Those fields stay fixed when you change the geometry, magnet count or grade":** true.
  - They are inputs, and nothing scales them with the design.
  - The losses built from them do change with the geometry (radius, length, pitch), but the sentence says the fields stay fixed, not the losses.
- **"enter its own 3D fields to check the limits":** reads as "check against the limits". That covers both field sets: the reverse fields set the limits and the slip fields set what is compared with them.
- **Dependent texts, all consistent:**
  - `STORED_3D_LABEL` and `STORED_3D_SENTENCE` (unchanged).
  - T8's `metal.measured_drag_Nm` overlay: "Once entered, it replaces the estimated slip heat in every temperature check." This is the same "temperature check" vocabulary the note now uses.
  - The three rows the note is attached to (`STORED_3D_ROWS`): governing limit, hot-day margin and verdict.
  - `docs/ai/02-system.yaml` and `05-update-tracker.md` describe the change correctly.
- **Traps:** none written.
  - Slice-1 trap 6 and T-N: the note says the fields stay fixed and does not follow the design.
  - No live 3D model is promised.

Recorded, no change asked:
- **Once a measured drag is entered, the slip fields reach no check.**
  - Probe: with a 0.02 N·m drag, doubling any slip field moves nothing, at all five designs.
  - So "the slip-heating fields" in the note's first sentence then over-includes.
  - Not worth a change: the drag's own help says it replaces the slip estimate in every temperature check; the reverse fields still matter, so the note's advice stands; and the 44 words are A-7's.
- **The stored reverse fields are the NdFeB example design's.**
  - At the Y30 probe they give a cold limit of 335.7 °C, so the verdict reads CHECK.
  - The note's advice ("for another design, enter its own 3D fields") is what that user needs.
  - The engine behaviour is outside this review.
- **A naming gap ahead:**
  - A-7's planned T10 headings say "Slip-loss fields from 3D analysis (example design)", and the note says "slip-heating fields".
  - Both are A-7's texts and the meaning is clear; T10's reviewer may want one name.

#### `coupling.max_harmonic` help

Text: "The torque model sums odd field harmonics up to this one, for your design and the bench prototype. Higher ones matter most for narrow, widely spaced blocks."

- **Caps and words:**
  - 27 words in 2 sentences.
  - The guard, `provenance_words` and `code_names` are clean (schema test).
  - No default number (spec 3.5): the choice text carries "(default)".
- **"sums odd field harmonics up to this one":** exact.
  - The choices are 1, 3, 5 (default), 7, 9 and 11, each the highest odd harmonic summed (`model.rs:51-58`).
  - The model sums the first `harmonic_count` terms, each built from both rings' field-harmonic amplitudes (`model.rs:993-999`, `680-711`).
- **"for your design and the bench prototype":** exact.
  - `calibration::compute(cal_in, ci.max_harmonic, dev)` (`api.rs:183`) sums the same set for the prototype's model torque (`calibration.rs:193-213`).
  - The sweeps read it too (`api.rs:392`); the help need not name them.
- **"Higher ones matter most for narrow, widely spaced blocks":** true with the corrections the app uses.
  - Grid: 9 pole counts (4-40) × 4 ring radii (9-30 mm) × 4 face gaps (0.3-5 mm) × 9 widths (1-25.4 mm) × 5 thicknesses (0.5-10 mm) × both circuits. That is 12,960 designs, each run at sets 1, 5 and 11, and every one had a positive pull-out.
  - 5 -> 11, median (max) change by mean fill:

    | Fill | Median | Max |
    |---|---|---|
    | Below 0.2 | 5.07 % | 527 % |
    | 0.2 to 0.4 | 0.27 % | 51 % |
    | 0.4 to 0.6 | 0.21 % | 51 % |
    | 0.6 to 0.8 | 0.06 % | 72 % |
    | 0.8 to 1 | 0.02 % | 43 % |

  - 5 -> 11, median (max) change by pole pitch over the face gap:

    | Pitch / gap | Median | Max |
    |---|---|---|
    | Below 5 | 0.00 % | 9 % |
    | 5 to 10 | 0.39 % | 62 % |
    | 10 to 30 | 1.42 % | 263 % |
    | 30 to 100 | 4.56 % | 376 % |
    | 100 and above | 28.5 % | 527 % |

  - In the fill × pitch table, the cell with fill below 0.2 and pitch at least 100 gaps has the highest median (86.5 %) and the highest maximum.
    - All top-8 changes are fills of 0.02 to 0.06 on 24-50 mm pitches at a 0.3 mm gap.
    - With wide blocks on a short pitch (fill ≥ 0.6, pitch < 10 gaps) the largest change is 7.0 %.
  - From harmonic 1 the pattern is the same: by fill, the median falls from 142 % below 0.2 to 1.5-2.7 % above 0.6.
  - The five probe designs, 5 -> 11:

    | Probe design | Change |
    |---|---|
    | Default | +0.00 % |
    | 8 magnets per ring | +0.29 % |
    | 304 back iron | −0.04 % |
    | Y30 | +0.00 % |
    | 20 mm ring radius | +0.39 % |

  - Thickness is a weaker, secondary factor: by thickness, the medians are 0.17-0.46 % and the maxima fall from 527 % (0.5 mm) to 153 % (10 mm). The help may leave it out.
- **No direction is stated, and that is right:** on the grid the 5 -> 11 change runs from −14.4 % to +527 %.
- **Dependent texts:**
  - The "harmonics" assumption's rationale ("add little at the default gap and more at small gaps or low fill") agrees.
  - The harmonics note's "which is why the workbook keeps only 1, 3 and 5" does not contradict it.
  - S-16 ("past the fifth they add little") stays rejected. The probe reproduces round 1's reason (+527 %).
- **Traps:** none written.
  - T-A: no fill formula.
  - T-D: no pitch optimum.
  - Not "fewer poles reach across the gap better": the sentence says which designs need the higher set, not which design has more torque.
  - Nothing on stiffness or magnet count, so the Global Constraints trap is not touched.

Recorded, no change asked:
- **"Widely spaced" could, at a stretch, be read as a wide ring gap, which would be backwards** (a wider gap fades the higher harmonics: pitch/gap below 5 gives a 0.00 % median). The phrase attaches to "blocks", and the face gap has its own screen name, "Magnet face gap". The natural reading is spacing along the ring, which is right.
- **The calibration factor.** On a bench-matched design, raising the set also moves the prototype's model torque and so the bench-matched factor. That case is a free-space circuit with the library blocks, which are not narrow, so the shown change there stays small. No text claims otherwise.

#### Should-fix status from earlier rounds

- S-b (`materials.cup_wall_suggested_mm`) stays open, as `task-9-review.md` M-1 and the analysis record say. It is not part of this change set, so this review does not rule on it.

## GUI strings and rationales (T10)

- Scope: not an overlay batch. These are T10's on-screen strings outside the help overlay,
  reviewed under the same G-3 rule (the brief's Step 12 prompt):
  - the 8 assumption rationales rewritten in plain words (ruling R10, `engine/assumptions.rs`);
    the end-effect one is A-8's text;
  - the uncoated-steel warning text;
  - the cup wall's failing verdict as drawn (ruling R9, A-9, ruling D-4): the dashboard, the
    results table, the Equation panel's value line and, after round 1, the equation's formula in
    the Equation panel and in every readout's hover; the engine string and the exports keep the
    engine text;
  - the end-effect banner and the greyed rows' hover;
  - the cheap jargon rows (map-locks J1-J3, J7, J8, J10, J12, J13, J15; map-jargon C-4 to C-6,
    C-8, C-11, A-10): the no-cell provenance line, the assumption line, the reset strings, the
    order toggle, the stored-field headings (A-7), the trace hint and banners, the no-equation
    text and tag, the outside-the-slider-range hover, the geometry notes and the one-piece clamp
    note.
- Read: the string dump (`git diff -U0 -- magcoupling-rs/src`, written to
  `magcoupling-rs/target/g3/T10-strings.diff`) and a review sheet (`T10-rendered.txt`: every
  changed string as drawn, and the engine values behind the rationales' claims at the five probe
  designs), not `help_for_review`; T10's brief, drafter notes and open questions; the amendments
  (Global Constraints, the T10 row: A-7, A-8, A-9, A-25 T-3, A-27; A-13, A-19 to A-24, ruling D-4,
  RF-3); slice-2 spec 3.2, 3.4, 3.6, 3.8, 5 and 8, slice-1 spec 8 and `phys-new-notes.md` section
  2; the engine and GUI code behind every changed string. Both rounds probed the engine from
  scratch crates outside the repository:
  - round 1: the harmonic sets (1 / 1, 3 / 1, 3, 5 / up to 11) at the five probe designs and at a
    0.5 mm gap; the knee fraction at 0.8, 0.9 and 1.0; the cup wall at the five designs. The three
    must-fix items were applied in a mirror of the crate, whose suites and both clippy runs passed;
  - round 2: the guard and counters on the changed fields; the cup wall equation as the
    typesetter lays it out; the cup wall at the five designs; the knee fraction on rated,
    gradeless and Y30 magnets; the end effect out of range at the five designs.
- Round 1: 3 must-fix items, text applied verbatim:
  - The cup wall equation's verdict literal (R9 wherever it is drawn: A-9, D-4, RF-3).
    - Old (drawn in the Equation panel's formula and in every cup-wall readout's hover): "Too
      thin: raise Metal design C122 to at least "
    - New: "Too thin: raise Minimum outer return ring wall to at least ". The typesetter draws a
      text literal through `gui::plain_text::shown_literal`, which `verdict_display` also builds
      on. The record keeps the engine text, so it still evaluates to the engine verdict, and the
      spreadsheet export (`render::plain`) keeps it. New test:
      `typeset::tests::the_cup_wall_equation_names_the_wall_input_by_its_label`.
    - Why: with Show spreadsheet details off, the spreadsheet cell was still drawn in the Equation
      panel (Review Focus RF-3's failure) and in every cup-wall readout's hover.
  - `END_EFFECT_GREYED`, the end of every greyed row's hover.
    - Old: "computed from the pull-out, which is not valid while the end-effect factor is zero or
      below."
    - New: "the pull-out and every number computed from it are not valid while the end-effect
      factor is zero or below."
    - Why: the first greyed row is the pull-out itself, whose hover said it was computed from the
      pull-out.
  - The `knee_fraction` rationale.
    - Old: "The knee of the demagnetization curve as a fraction of Hcj: the reverse field at which
      irreversible loss starts. The onsets are calibrated to the magnet's rating. Other reasonable
      calibration forms move the example design's onset by up to 9.83 °C, nearly all of the
      default demagnetization margin, so little is left for model uncertainty."
    - New: "The knee of the demagnetization curve as a fraction of Hcj: the reverse field at which
      irreversible loss starts. For a rated magnet the onsets reached on heating are calibrated to
      its rating; a hard ferrite's cold onsets are not. Other reasonable calibration forms move the
      example design's skipping onset by up to 9.83 °C, nearly all of the default demagnetization
      margin, so little is left for model uncertainty."
    - Why: with a positive coercivity coefficient the calibration offset is 0 and the cold onsets
      are not calibrated. At the Y30 probe the knee fraction moves the cold limit uncalibrated:
      352.4 / 335.7 / 319.9 °C at 0.8 / 0.9 / 1.0. A gradeless magnet has no rating and is not
      calibrated either, which the word "rated" covers.

  Should-fix offered, not applied (open for the controller):
  - SF-1, `harmonics` rationale, sentence 1: "a square wave" is exact only at full fill, and "each
    contributing to the torque" reads as "each adds torque" (at the 20 mm ring-radius probe the
    third harmonic lowers the pull-out by 12.4 %). Suggested: "Each ring's alternating blocks make
    a stepped magnetization wave of odd harmonics 1, 3, 5, ..., each of which can raise or lower
    the torque."
  - SF-2, `end_effect` rationale (A-8's binding text, so the controller's call): "magnet length"
    is the overlapping length, which differs from a ring's length only when the rings' parts
    differ and the axial override is blank. Suggested: "... / overlapping magnet length" and "At
    or below an overlapping magnet length of coefficient × pole pitch ...".
  - SF-3, `preload_fraction` rationale (not in T10's diff, but now under the guard): for A4-70
    stainless the stress is the yield, as A-21 fixed in T7's help and T16's note. Suggested:
    "Screw preload as a share of the proof load (the yield load for stainless; bolted-joint
    practice), capped by thread stripping in the aluminium."
  - SF-4, the assumption Source lines (drawn only with Show spreadsheet details on): audit and
    plan-decision ids are not spreadsheet details; a later pass could rewrite them as A-13 does,
    keeping the cells.
  - SF-5, the wall-rule note prints the rule's wall to four significant figures ("at least 2.000
    mm") beside the verdict's "at least 2.0 mm". Cosmetic.
  - SF-6, the `coupling.c_end` and `calibration.c_end` hovers still show the engine help "f_end =
    1 − c_end · pole pitch / L." (T8's overlay table, not T10's diff). Suggested HELP row: "The
    magnet length lost at the ends, in pole pitches: end-effect factor = 1 − coefficient × pole
    pitch / magnet length. Change it only with 3D or bench evidence."

  Recorded, no change asked:
  - the implementer's departures, all accepted: `demag_margin` sentence 2 (a hard ferrite has no
    hot onset, so its rating is its hot limit; the drafted text was false at the Y30 probe); the
    Br rationale's "the inner Br times the outer Br" (a graded manual ring has its own
    coefficient); "example design" in the headings, the warning and the knee rationale (A-7); the
    end-effect rationale byte for byte A-8's; R9 through one `shown_text` helper for the results
    table and the value line; the dashboard's wrapped value-and-ids flow is a layout change,
    outside G-3, for the task code review;
  - typed numbers, all allowed: "1, 3 and 5" (the assumption's own default set), "10 magnets per
    ring and a 1.4 mm gap" (where the free-space check ran, A-8), "9.83 °C" (an audit finding),
    "20 to 150 °C" (the coefficient's fitted range), "0.10" (the oily-bore advice; the default is
    0.15).
- Round 2: PASS, no must-fix. It re-reviewed the three changed fields and every sentence whose
  meaning depends on them; round 1's findings on the unchanged fields stand.

  Should-fix:
  - SF-R2-1, the doc comment of `engine::materials::CUP_WALL_TOO_THIN` was stale after round 1's
    first item. Applied before the commit, the reviewer's text verbatim (a code comment, not
    on-screen text).
  - SF-R2-2, the brief's drafted commit message said the results table keeps the engine text,
    false after A-9 and round 1's first item. Applied: the commit message carries the reviewer's
    sentence.
  - SF-R2-3 (optional), `END_EFFECT_GREYED`: "every number" -> "every result", since two greyed
    rows are texts (the hot-minimum check and the recommended screw). Not applied: it would change
    reviewed on-screen text. Open for the controller.

  Recorded, no change asked: the typesetter draws `fmt`'s decimals as "(to 1 decimals)" beside the
  changed literal; pre-existing wording, not a T10 string.
- Seeds: the T10 index row names none of A-24's seeds, so there was no seed to accept or reject.
  For consistency only, the round-1 reviewer checked that S-4 (T13's end-effect sentence) uses the
  same facts as the end-effect rationale, and that S-16 agrees with the harmonics rationale at the
  default gap (S-16 itself stays rejected, as recorded under Engine wording (T9)).
- Amendments: A-7 applied (the headings read "Reverse fields from 3D analysis (example design)"
  and "Slip-loss fields from 3D analysis (example design)"); A-8 applied (T3's hover-text check
  un-ignored, the drafted inputs tooltip test and Step 10's contingency rows not added, the
  end-effect rationale byte for byte A-8's, checked by script); A-9 and D-4 complete after round 1
  (the dashboard, the results table, the Equation panel's value line and formula, and every
  readout's hover; the CSV, JSON and spreadsheet keep the engine text); A-25 T-3 present
  (`results_table::tests::the_cup_wall_verdict_is_drawn_with_the_wall_input_s_label_and_exported_as_the_engine_s`);
  A-27 present (the rationale-branch comment in the dashboard guide test). A-13: no drawn field
  carries a cell, correction, audit or decision id, or a file or code name; the assumption Source
  lines, drawn only with the details on, are SF-4's. A-19 to A-23 are other tasks' texts, and no
  T10 string contradicts them.
- Traps: none of spec section 8's, slice-1 spec section 8's, `phys-new-notes.md` T-A to T-P, or
  "more magnets make the coupling stiffer" is written in any T10 string.
- Verdict: PASS (round 2).

### Report for the record, round 1 (as returned)

#### Amendments for T10

- **A-7 (one name for the example design): applied.**
  - The headings read "Reverse fields from 3D analysis (example design)" and "Slip-loss fields from 3D analysis (example design)" (`inputs.rs:541, 552`).
  - The implementer carried the name into the warning text and the knee rationale. Accepted.
- **A-8: applied.**
  - The `#[ignore]` on T3's `every_hover_text_with_the_details_off_is_in_plain_words` is deleted.
  - T10's drafted inputs tooltip test is not added.
  - Step 10's contingency HELP rows are not added.
  - The end_effect rationale is A-8's text byte for byte (checked by script).
- **A-9 / D-4: partly applied.**
  - The dashboard, the results-table row (`row_ui` → `shown_text`) and the Equation panel's value line (`value_text` → `shown_text`) draw the plain verdict.
  - The CSV keeps the engine text (test `the_cup_wall_verdict_is_drawn_with_the_wall_input_s_label_and_exported_as_the_engine_s`).
  - The formula's literal is not yet converted: must-fix 1.
- **A-25 T-3: present** (A-9's results-table test).
- **A-27: present** (the rationale-branch comment in the dashboard guide test).
- **A-24 seeds:** the T10 index row names none. There is no seed to accept or reject.
  - For consistency only: S-4 (T13 end_effect sentence 4) uses the same facts as the end_effect rationale (10 magnets per ring, 1.4 mm, 2 %, 3.2 mm).
  - S-16 ("past the fifth they add little") agrees with the harmonics rationale and with the probe (1,3,5 → 11 changes the pull-out by under 0.4 % at the default gap at all five designs).
- **A-19 to A-23:** none is indexed for T10, so there is nothing to apply.
  - The T10 texts do not contradict them. A-22's "rings with a picked grade ignore it" agrees with the br rationale ("A ring on manual dimensions with a picked grade uses that grade's own coefficient instead").
  - A-21's stress basis is the point of SF-3.

#### Traps

None of spec section 8's, slice-1 section 8's, `phys-new-notes.md` T-A to T-P, or "more magnets make the coupling stiffer" is written in any T10 string.

**Spec section 8 traps:**
- "The end factor is accurate": not written. The rationale gives the checked band and says it reads optimistic at wider gaps.
- "Edit the calibration inputs to match your design": not written. The calibration rationale states the match rule (no steel, the prototype's magnets per ring, B842SH on both rings, `model.rs:584-601`, `api.rs:184-192`) and that size is not compared. No direction is stated, per brief open question 5.
- "Each ring uses its own α": not written. Only a manual ring with a picked grade does, which is exactly `ResolvedMagnet::alpha_br`. Every library part's grade is −0.0012 /°C (the default).
- "The clamp is sized to the drive": not written.
- "A stored 3D field follows the design": not written. The headings say "(example design)".
- "Raising Boss outside diameter enlarges the drawn boss": not written. The design check says only that the two bosses differ, naming each by its label and "in the clamp inputs" (ruling 3).
- "1.5 T is 4140's saturation": no saturation figure is printed.
- T-L: the verdict says nothing about torque.
- "The coupling protects the gearbox": not touched.

**Slice-1 section 8:** nothing written about radius, pole count, re-locking, 416, or the bench calibration beyond the prototype.

#### Strings checked against the engine, at the five designs where a condition is named

- **`calibration_factor`:** correct (f_cal 0.95, 0.95, 1.049 at 304, 0.95, 0.95).
- **`br_temperature_coefficient`:** correct, with the implementer's departure 2 accepted (`ring_pair_factor`; `shear_stress` multiplies bi·bo).
- **`hcj_temperature_coefficient`:** correct (`temperature.rs:787-792`: the grade's Hcj and β when E20 is on and the source is 1; otherwise the inputs for both rings).
- **`demag_margin`:** correct, with departure 1 accepted. NdFeB: limit = skipping onset − margin (103.06 → 93.06 °C). Y30: hot limit = rating 250 °C, cold limit = cold onset 325.71 + 10 = 335.71 °C.
- **`clamp_friction`:** correct (Tper = µ·min(Fb, Fs)·d·k at `clamps.rs:366`; 0.10 is the engine help's advice, not the 0.15 default; the joint friction is the separate input `clamps.joint_friction`).
- **`harmonics`:** correct apart from SF-1.
- **`knee_fraction`:** must-fix 3.
- **`uncoated_low_alloy_steel`:** correct. The default plating input is electroless nickel, 0.015 mm (`materials.rs:52-56`); the rule fires only at a plating thickness of zero.
- **Cup-wall verdict.** The dashboard, table and value line are correct at every design, failing or not. "Raise Minimum outer return ring wall to at least X mm" passes the check, because X ≥ the rule's unrounded wall and the wall input is the check's operand.
- **End-effect banner:** correct ("end-effect factor -1.202" with short magnets).
- **Greyed hover:** must-fix 2.
- **Geometry notes.** The numbers come from results and inputs, and the labels from `label_for`. The design-check wording is correct at all five designs.
- **The other strings:**
  - **Trace banners:** correct.
  - **The singular forms:** correct.
  - **TRACE_HINT:** correct.
  - **NO_EQUATION:** correct; the "name = value" line is drawn above it (`no_equation_body`).
  - **NOTHING_TRACED:** correct.
  - **ONE_PIECE_ONLY:** correct.
  - **The J7 reset strings:** correct.
  - **"By original sheet":** correct.
  - **ASSUMPTION_LINE:** correct; the view is named "Assumptions" (`panel.rs:172`).
- **OUTSIDE_RANGE_HINT: correct.** The differential tests run seeded cases across every slider range against the spreadsheet's formulas (the Python port, corrections off; `tests/differential.rs:1-10`). Outside the range nothing checks them.

#### Rulings on the implementer's departures

1. **demag_margin, sentence 2:** accepted. Sentence 1 alone was false for ferrite.
2. **br, "the inner Br times the outer Br":** accepted.
3. **knee, "example design's onset" and "nearly all of the default demagnetization margin":** accepted. Must-fix 3 adds "skipping" and the calibration scope.
4. **The warning's "example design":** accepted (A-7).
5. **The headings' "(example design)":** accepted (A-7).
6. **end_effect, A-8 verbatim:** confirmed.
7. **R9 through `shown_text` to the table and the value line:** accepted. The "known limit" (the formula's literal) is not accepted: must-fix 1.
8. **The dashboard's wrapped value-and-ids flow:** a layout change, not a string. It is outside G-3; the post-commit task code review owns it.

#### Typed numbers

All five are allowed.
- **"1, 3 and 5":** the assumption's own default set. The choice label itself reads "1, 3, 5 (default)".
- **"10 magnets per ring and a 1.4 mm gap":** A-8's text. These are where the free-space check ran, not the user's design (brief open question 4).
- **"9.83 °C":** an audit finding, not a default. It was measured against the workbook's 102.55 °C onset; the example design's is now 103.06 °C. That is close enough for "up to".
- **"20 to 150 °C":** the coefficient's fitted range, in the input's label.
- **"0.10":** the oily-bore advice, also in the engine help; the default is 0.15.

No editable default is typed anywhere else. The geometry and verdict numbers are computed.

### Report for the record, round 2 (as returned)

#### Round-1 must-fix items: applied and checked

**MF-1 (R9 wherever drawn; A-9, D-4, RF-3): applied verbatim.**
- `plain_text.rs:21, 41-57` adds `CUP_WALL_INPUT` and `shown_literal`.
- `typeset.rs:756-759` draws through it, as do both test oracles (`typeset.rs` about 1383 and 1555).
- `verdict_display` builds on it (`dashboard.rs:385-388`). The private `CUP_WALL_INPUT` in `dashboard.rs` is gone (DRY: one copy of the plain prefix).
- New test `typeset::tests::the_cup_wall_equation_names_the_wall_input_by_its_label`, byte for byte as specified.
- Physics:
  - `t_wall` in the drawn formula is the wall input (`symbols.rs:74`). The literal names that same input by its label.
  - The rule says "raise ... to at least t_wall,sug mm", where t_wall,sug = ceil(t_bi, 0.1) ≥ t_bi.
  - So the advice passes the check at every failing design (default, 8 magnets, 20 mm ring radius).
  - It says nothing about torque (T-L). It prints no saturation figure (T-K). It speaks of the rule only (T-I).
- Coverage:
  - Every on-screen equation goes through `layout_equation`: the explorer at `explorer.rs:841`, and readout hovers at `readouts.rs:135`.
  - Only the spreadsheet export uses `render::plain` (`spreadsheet.rs:528`). It keeps the engine text, as D-4 requires.
- The record still evaluates to the engine verdict, so `tests/explain.rs` is unaffected.

**MF-2 (`END_EFFECT_GREYED`, `dashboard.rs:246`): applied verbatim.**
- It is true on all seven greyed rows at all five designs: the pull-out row no longer says it is "computed from the pull-out".
- It agrees with:
  - the banner ("the pull-out and the numbers computed from it are greyed");
  - the end_effect rationale ("the factor is zero or negative, and the model gives no prediction");
  - A-11's "no prediction" for the explorer's live values.
- "Zero or below" is `end_effect_in_range`'s condition (f_end > 0 to be in range).
- It carries no "f_end", and the panel test holds it to the guard and the INTERNAL list.
- See SF-R2-3 for the optional word change.

**MF-3 (`knee_fraction` rationale, `assumptions.rs:94`): applied verbatim.**
- Sentence 2 holds at every case probed:
  - Rated magnets (library parts and picked grades, `model.rs:454, 464`) have calibrated heating onsets (`offset = t_ref − tmax`, `temperature.rs:804-807`).
  - Gradeless magnets ("n/a") are uncalibrated, which matches the "rated" scope.
  - A hard ferrite has no heating onset (+inf). Its cold onsets come from `cold_onset_C`, which takes no offset.
- Sentence 3 matches the audit:
  - It quotes the 9.83 °C as the strictest form against the engine's skipping onset, and "up to" covers it.
  - "Nearly all of the default demagnetization margin" types no 10 °C.
- Dependent sentences:
  - The demag_margin rationale ("A positive-beta magnet (hard ferrite) has no hot onset, so its rating is its hot limit; its cold limit is its cold onset plus this margin") agrees with it.
  - So does T16's demagnetization source in A-13 ("other calibration forms move the skipping onset by up to 9.83 °C").
- Two traps, checked:
  - "A hotter-rated grade is free": not written. The rationale describes the calibration and closes on how little margin is left for model uncertainty.
  - "Model ferrite by typing a positive β": not written. The rationale names hard ferrite and suggests no typed value.
- The word "calibration" here is the onset calibration, not the bench calibration factor. Nothing suggests the bench calibration reaches other designs (slice-1 section 8).

#### Amendments for T10

- **A-24 seeds.** The T10 index row names none (A-7, A-8, A-9, A-25 T-3, A-27 only). There is no seed to accept or reject.
- **A-19 to A-23.** None is indexed for T10, so there is nothing for T10 to apply. The changed T10 texts contradict none of them.
- **Text-changing amendments that are T10's own:**
  - **A-7 (headings): still applied.** "Reverse fields from 3D analysis (example design)" and "Slip-loss fields from 3D analysis (example design)".
  - **A-8 (end_effect rationale): still byte for byte A-8's text**, checked by script against `amendments.md:267`. T3's tooltip check is still un-ignored: `dashboard.rs` has no `#[ignore]`.
  - **A-9 / D-4: now complete.** The plain verdict is drawn on the dashboard, in the results table, on the Equation panel's value line, and (MF-1) in the formula in the Equation panel and every readout's hover. The CSV, JSON and spreadsheet export keep the engine text.
  - **A-25 T-3: present.** The test is `results_table::tests::the_cup_wall_verdict_is_drawn_with_the_wall_input_s_label_and_exported_as_the_engine_s`.
  - **A-27: present.** The rationale-branch comment is in the dashboard guide test.
- **A-13 (every drawn field checked like any other).**
  - The three changed fields and the drawn formula literal are plain words. None carries a cell, correction, audit or decision id, or a file or code name.
  - The assumption Source lines were not changed this round. They are drawn only with Show spreadsheet details on (spec 3.6). Round 1's SF-4 covers their audit ids (for example the knee's "M1 audit rulings (onset calibration) and M13"), and it stands.

#### Traps

None is written in any changed field:
- spec section 8;
- slice-1 spec section 8;
- `phys-new-notes.md` T-A to T-P;
- "more magnets make the coupling stiffer".

The relevant ones were checked above: T-I, T-K and T-L on the verdict; "a hotter-rated grade is free" and "model ferrite by typing a positive β" on the knee; "the bench calibration applies beyond the prototype" on the word "calibration".

#### Typed numbers

- The only typed number in a changed field is "9.83 °C" (knee). It is an audit finding, not a default, and round 1 already allowed it.
- The verdict's number and the banner's factor are computed.

#### Observation outside T10's diff (not a finding)

- The typesetter draws `fmt`'s decimals as "(to 1 decimals)" (`typeset.rs:662`, and the oracle at 1626). It sits beside the changed literal in the cup wall formula.
- It is pre-existing wording, not a T10 string. A later pass could singularize it for 1.

### Task review fix round 1 (2026-10-15): the wall-rule line

- The task code review (`task-10-review.md`, Important 1) found a wrong statement in the Geometry
  tab's wall-rule line. The line printed the failing verdict's suggestion,
  `materials.cup_wall_suggested_mm` (the rule's wall rounded up to 0.1 mm), as the rule. The
  check compares the wall with the unrounded rule (`engine/materials.rs`, `compute`), and the
  wall's slider steps by 0.05 mm. At the defaults with a 1.95 mm wall, the dashboard reads OK
  while the line asked for at least 2.000 mm. Round 1's SF-5 raised only the number's format.
- The line now prints the rule's own wall, `materials.backiron_thickness_needed_mm`. It is still
  drawn only while `materials.cup_wall_suggested_mm` is a number, so it drops without back iron.
  - Old, at a 1.95 mm wall: "Wall rule: at least 2.000 mm of cup wall at the pocket corners;
    Minimum outer return ring wall is 1.950 mm"
  - New, at a 1.95 mm wall: "Wall rule: at least 1.919 mm of cup wall at the pocket corners;
    Minimum outer return ring wall is 1.950 mm"
  - At the defaults (1.8 mm): "Wall rule: at least 1.919 mm of cup wall at the pocket corners;
    Minimum outer return ring wall is 1.800 mm"
- The line now quotes the same need as the cup wall guide's Problem line ("the flux it carries
  needs 1.919 mm at the 1.500 T design flux density"). Since the re-check below, the line rounds
  it up and the guide rounds it to the nearest digit, so the two differ in the last digit at
  about half the failing designs (the re-check's SF-2). The failing verdict still advises "at
  least 2.0 mm", the suggestion, which passes the check, so the two numbers differ on purpose:
  the rule's wall and its rounding. SF-5 (one number in two formats) no longer applies.
- New test: `geometry::tests::the_wall_rule_line_asks_for_the_wall_the_check_compares_with`. It
  steps the wall's slider up from the defaults to the first wall the check passes (1.95 mm),
  checks that the wall lies between the rule's wall and its rounding, and pins the line.
- The G-3 re-check of this line ran: MUST_FIX, one item, applied (the re-review below). The rule's
  wall is a minimum, so the line now rounds it up at its last digit (`geometry::mm_at_least`), so
  a wall shown equal to it passes. The line is unchanged at the defaults (1.919 mm).
  - Rounded to the nearest digit, 12 poles at a 1.23 T design flux density with a 1.95 mm wall
    (slider values; the rule's wall is 1.9501155 mm) read "Wall rule: at least 1.950 mm of cup
    wall at the pocket corners; Minimum outer return ring wall is 1.950 mm" beside the dashboard's
    "Too thin: raise Minimum outer return ring wall to at least 2.0 mm".
  - Now, at that design: "Wall rule: at least 1.951 mm of cup wall at the pocket corners; Minimum
    outer return ring wall is 1.950 mm".
  - Over the 379,620 slider designs (poles 2 to 40, design flux density 0.50 to 2.20 T, wall 0.50
    to 6.00 mm, the defaults otherwise), the nearest digit disagreed with the check 19 times,
    always at the wall just below the rule's; rounding up disagrees 0 times. The reviewer probed
    this, and the fix reproduced both counts.
  - New tests: `geometry::tests::a_wall_shown_equal_to_the_wall_rule_passes_the_check` (that
    design) and `geometry::tests::a_minimum_length_is_rounded_up_at_its_last_shown_digit` (the
    helper).
- Left for the whole-branch review, as the task review says: T9's engine label of
  `materials.cup_wall_suggested_mm`, "Cup wall the rule asks for at the pocket corners", has the
  same conflation.

### Re-review: code-review fix round 1

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `ba2627d`
  (`task-10-textreview-1.md`).
- Scope: the one string the task review's fix round 1 changed, the wall-rule line under the
  Geometry tab's views (`gui/geometry.rs`, `notes()`), and every on-screen text whose meaning
  depends on it. The notes dump does not cover it (it is a Geometry-tab line, not a guidance
  note), so the reviewer probed the line with `compute_all` designs.
- Verdict: MUST_FIX, one item. The claim is right: the line prints the wall the check compares
  with, names the right place, writes no trap and passes the plain-words guard at every probe.
- MF-1, applied verbatim (the review's option A): the line rounded the rule's wall, a minimum, to
  the nearest shown digit, while the check passes only at the unrounded wall. Now:
  - `mm_at_least` (`gui/geometry.rs`, after `mm`) rounds a length up at the last digit `mm`
    shows (tied to `format::SIGNIFICANT_DIGITS`); a number that is not finite or not positive is
    shown as it is;
  - the line prints `mm_at_least(materials.backiron_thickness_needed_mm)`, and its comment says
    why;
  - the tests' oracle, `expected_wall_rule`, uses `mm_at_least`;
  - the review's red-first test, `a_wall_shown_equal_to_the_wall_rule_passes_the_check`, and its
    optional helper test, `a_minimum_length_is_rounded_up_at_its_last_shown_digit` (the review's
    six values, plus both infinities, zero and a negative length).
  - The wording is unchanged (21 words); only the rounding of the number moved.
- Should-fix, not in this round's list, left open for the controller:
  - SF-1: T9's label and help of `materials.cup_wall_suggested_mm` ("Cup wall the rule asks for
    at the pocket corners"; "The cup wall check asks for it") now give the rule a second number
    at the defaults (2.0 mm beside this line's 1.919 mm). The review gives replacement text.
  - SF-2: the cup wall guide's Problem line (`gui/guidance.rs`) rounds the same need to the
    nearest digit, so at the MF-1 design it reads "needs 1.950 mm" beside this line's 1.951 mm.
    The review leans to passing it through `mm_at_least` too.
  - SF-3: two test comments (`gui/dashboard.rs`, `gui/panel.rs`) call the suggestion the rule's
    wall. Not on screen.

The reviewer's report for the record, as returned:

#### The claim against the engine

- **Same number as the check.** `materials::compute` stores `t_bi_req_mm` as
  `backiron_thickness_needed_mm`, then tests `wall_corner_mm >= t_bi_req_mm`. `api.rs` passes
  `m.backiron_needed_mm` and `md.cup_wall_corner_mm`, so the line's two numbers are exactly the
  check's two operands. `model.cup_ring_check` (`model.rs:1043-1051`) compares the same wall with
  the same unrounded `t_bi`, so the line agrees with both checks.
- **Gating.**
  - The line shows exactly when the suggestion is a number, which is when the circuit in effect
    has back iron (`parts.backiron`, E9).
  - It drops for "Back iron" off and for the 304 and 6061 picks (probed: no line, check "No
    back iron").
  - With E9 off and no back iron, the workbook's "Too thin" verdict and the line both appear.
    They are consistent with each other, and this is pre-existing.
- **Place.** On faceted rings the pole transitions fall at the polygon corners. That is where
  the return ring's circumferential flux peaks and the pocketed wall is thinnest, so "at the
  pocket corners" is the right place for the rule's wall. The input's help says the same
  (`plain_text.rs:717`).
- **Other texts.** The line now quotes the same need as the cup wall guide's Problem line
  ("needs 1.919 mm at the 1.500 T design flux density", pinned at `guidance.rs:3485`). The
  record subsection's claim is verified.

#### Probe

Default design except where the row says. The line is quoted after "Wall rule: at least".

| Design | Need (mm) | Wall (mm) | Check | Dashboard verdict | Line number |
|---|---|---|---|---|---|
| defaults | 1.918914 | 1.80 | Too thin | ... at least 2.0 mm | 1.919 mm |
| wall 1.90 | 1.918914 | 1.90 | Too thin | ... 2.0 mm | 1.919 mm |
| wall 1.95 | 1.918914 | 1.95 | OK | none | 1.919 mm |
| 8 poles | 2.398642 | 1.80 | Too thin | ... 2.4 mm | 2.399 mm |
| 20 mm ring radius | 3.267081 | 1.80 | Too thin | ... 3.3 mm | 3.267 mm |
| Y30 ferrite | 0.532555 | 1.80 | OK | none | 0.5326 mm |
| 1018 back iron (1.7 T) | 1.693159 | 1.80 | OK | none | 1.693 mm |
| 12L14, 416, 17-4PH picks (1.5 T input) | 1.918914 | 1.80 | Too thin | ... 2.0 mm | 1.919 mm |
| design flux density 0.5 T | 5.756741 | 1.80 | Too thin | ... 5.8 mm | 5.757 mm |
| 2 poles | 9.594568 | 1.80 | Too thin | ... 9.6 mm | 9.595 mm |
| 40 poles | 0.479728 | 1.80 | OK | none | 0.4797 mm |
| back iron off; 304; 6061 | n/a | 1.80 | No back iron | none | no line |

- Every line is 21 words, has level `None`, and is guard-clean.
- At the first passing slider wall (1.95 mm) the line no longer asks for more than the check.
  This is Important 1's case, and it is fixed.

#### Traps (none written)

- **Slice-2 section 8.** No claim about poles, ring radius, re-locking, the end factor,
  calibration, α, steel swaps, torque with steel, grades, mass, slipping, the clamp, the boss,
  ferrite β, the expansion caution, 4140 saturation or the gearbox. The 416, 17-4PH and 12L14
  picks show the rule's number only, with no adequacy claim.
- **Slice-1 section 8.** Nothing on radius, poles, pull-out, calibration, 416 or the stored
  fields.
- **T-I** (the rule reads 12 to 16 % thin). The number is attributed to "Wall rule", the
  calculator's rule, and claims no sufficiency.
  - The cup wall guide links `a5.low_saturation`, whose watch-out says the formula reads 12 to
    16 % thin and that walls near the limit deserve margin.
  - The line now shows the unrounded rule (1.919 mm) where it used to show 2.000 mm. That is
    the check's own number, so it states nothing new about adequacy.
- **T-J, T-K, T-L.** The line names no temperature, no saturation figure and no torque effect.
- **The added trap** (more magnets make the coupling stiffer): not touched.

#### Caps and plain words

- The wording is byte-equal to the reviewed text. Only the number's source changed, so the word
  count is unchanged at 21.
- The geometry notes have no row in the caps table. The line was reviewed at this length.
- The number is a computed result, not a typed default.
- The input's name comes from `label_for(CUP_WALL_INPUT)`.
- No cell, correction id, file or code name appears.
- `plain_text_violations` is empty at every probe design that draws the line (29 of 33).

#### Pre-existing, not new, for the record

- **Non-finite input from a design file.** A design file with a design flux density of 0 or NaN
  (the slider stops at 0.5 T) gives "Wall rule: at least +inf mm" or "at least NaN mm". The old
  line printed the same through `mm(ceiling(...))`. MF-1's helper keeps that behaviour (a
  non-finite number is shown as it is). The geometry module drops non-finite pieces with a note,
  but not this text. It is outside this round.
- **Float edge.** `(x / step).ceil()` can lift a need that sits exactly on a shown digit by
  one digit, through floating-point error, which asks 0.001 mm more than the check. A computed
  need landing exactly on a digit is measure-zero, and the error is conservative.
- **`docs/ai/02-system.yaml` does not parse as YAML.** The fix report says this predates this
  round (line 15, the linkage-solver list item). It is not this review's.

#### Docs checked against the code

- README geometry row: true.
- `02-system.yaml` sentence: true.
- Tracker bullet: true. "1.95 mm (on the slider's 0.05 mm grid)" was verified by the probe.
- BL-056: true.
- The record subsection, `docs/analyses/2026-10-14-magcoupler-help-review.md` "Task review fix
  round 1": true, including that SF-5 no longer applies.
- After MF-1, each of these gets the "rounded up at its last digit" clause.

### Re-review: code-review fix round 1 (re-check 2)

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `8a45797`
  (`task-10-textreview-2.md`): the re-review above's MF-1 as applied.
- Scope: the wall-rule line under the Geometry tab's views (`gui/geometry.rs`, `notes()`), whose
  number is now `mm_at_least(materials.backiron_thickness_needed_mm)`, and every on-screen text
  whose meaning depends on it. The wording is unchanged. The notes dump does not cover the line,
  so the reviewer probed it with a scratch crate on `compute_all` designs.
- Verdict: PASS, no must-fix.
  - MF-1 is applied verbatim: the import, the helper, the comment, the call, the oracle and both
    tests.
  - The line prints the wall the check compares with, rounded up at its last shown digit. Over
    the 379,620 slider designs, a wall shown at or above it passes and a wall shown below it
    fails: 0 disagree, against 19 for the nearest digit.
  - The line is 21 words, guard-clean at every design, and writes no trap.
- SF-3, applied: the garbled sentence in "Task review fix round 1" above now reads as the review
  gives it, in the form for SF-2 left open.
- Should-fix, left open for the controller:
  - SF-1: `mm_at_least` meets its guarantee only to within a few ulps, while its doc comment and
    the README's geometry row and geometry test row state it exactly.
    - A need 1 to 3 ulps above a digit is shown at that digit: 1.9000000000000001 shows
      "1.900 mm", and a 1.90 mm wall fails.
    - 40 of the 9,000 lengths on a digit from 0.1 to 10 mm are lifted one digit (4.001 shows
      "4.002 mm"), so the test row's "a length already on a digit stays" is false as a general
      statement.
    - No slider design (0 of 379,620) and no random need (0 of 2,000,000) reaches either case.
    - The review recommends option A: show the nearest number, and step one digit up only when
      that number reads back below the need. It gives the exact code and two red-first asserts.
      Option B, the fallback if no further code round is wanted, qualifies the three claims
      (docs only).
  - SF-2 (re-review 1's SF-2, new numbers): the cup wall guide's Problem line rounds the same
    need to the nearest digit.
    - It quotes a different last digit from this line at 48,563 of the 94,832 failing slider
      designs. At a 20 mm ring radius it reads "needs 3.267 mm" beside "at least 3.268 mm".
    - At the boundary designs it still reads "is 1.950 mm; ... needs 1.950 mm" beside a red
      badge.
    - The review leans to rounding the guide's need up too. That changes slice 1's G-3 contract
      (a Problem's numbers as the readouts show them), so it is the controller's ruling.
  - Carried from re-review 1, unchanged: SF-1 (the suggestion's label and help) and SF-3 (test
    comments). SF-3's list gains two code comments that call the suggestion the rule's wall:
    `engine/materials.rs:224` (T10's) and `engine/materials.rs:241` (pre-existing).
- Corrections to re-review 1's copied report above, which stays as returned:
  - Its probe table gives the line's numbers from before MF-1. Three rows now read one digit
    up: the 20 mm ring radius 3.268 mm (was 3.267), the 1018 back iron 1.694 mm (was 1.693) and
    40 poles 0.4798 mm (was 0.4797). The other rows are unchanged.
  - Its "Float edge" bullet names only the conservative direction, a need on a digit lifted one
    digit. SF-1 gives the other: a need a few ulps above a digit shown at that digit.

The reviewer's report for the record, as returned:

#### MF-1 as applied

The commit matches re-review 1's exact replacement:
- the import (`SIGNIFICANT_DIGITS`);
- `mm_at_least` after `mm`;
- the four-line comment in `notes()`;
- the call `mm_at_least(results.materials.backiron_thickness_needed_mm)`;
- the oracle `expected_wall_rule`;
- `a_wall_shown_equal_to_the_wall_rule_passes_the_check`, verbatim;
- `a_minimum_length_is_rounded_up_at_its_last_shown_digit`, with the review's six values plus
  both infinities, zero and a negative length.

`the_wall_rule_line_asks_for_the_wall_the_check_compares_with` still holds: at the defaults the
rounded-up need is 1.919 mm, not 2.000 mm.

#### The claim against the engine (unchanged by this round)

- The line's two numbers are the check's two operands: `t_bi_req_mm` and `wall_corner_mm`
  (`materials.rs:246`), and the model's `thick_check` compares the same pair (`model.rs:1043`).
- Rounding the need up keeps the line a true statement of a minimum. At 4 significant digits, a
  wall shown at or above the shown rule meets the unrounded rule. This holds for every wall on
  the 0.05 mm slider grid, which lies on the display's 1e-3 mm grid from 1 mm up and on its
  1e-4 mm grid below 1 mm.
- The gating is unchanged: the line shows exactly while the suggestion is a number, that is,
  while the circuit in effect has back iron.

#### Probe

Named designs, quoted as the app draws them:

| Design | Need (mm) | Check (shown) | Wall-rule line |
|---|---|---|---|
| defaults | 1.918913657 | Too thin: raise Minimum outer return ring wall to at least 2.0 mm | at least 1.919 mm ...; ... is 1.800 mm |
| 12 poles, 1.23 T, 1.95 mm | 1.950115505 | Too thin: ... 2.0 mm | at least 1.951 mm ...; ... is 1.950 mm |
| 12 poles, 1.23 T, 2.00 mm | 1.950115505 | OK (none) | at least 1.951 mm ...; ... is 2.000 mm |
| 8 poles, 2.18 T, 1.65 mm | 1.650441792 | Too thin: ... 1.7 mm | at least 1.651 mm ...; ... is 1.650 mm |
| 6 poles, 1.17 T, 4.10 mm | 4.100242856 | Too thin: ... 4.2 mm | at least 4.101 mm ...; ... is 4.100 mm |
| 2 poles, 0.50 T | 28.783704848 | Too thin: ... 28.8 mm | at least 28.79 mm ...; ... is 1.800 mm |
| 40 poles, 2.20 T | 0.327087555 | OK (none) | at least 0.3271 mm ...; ... is 1.800 mm |

- Each line is 21 words, has level `None`, and is guard-clean.
- Each failing design's line shows the wall below the rule.

The slider sweep, through the real notes: every pole count (2 to 40), every design flux density
(0.50 to 2.20 T, 0.01 T steps) and every wall (0.50 to 6.00 mm, 0.05 mm steps), with the default
design otherwise. That is 379,620 designs, every one with a line.

| Rounding | Shown met, check fails | Shown short, check passes |
|---|---|---|
| up (as shipped) | 0 | 0 |
| to the nearest (before MF-1) | 19 | 0 |

- Across the sweep, 0 lines tripped the guard, 0 lines were not 21 words, and 0 numbers failed
  to parse.
- `mm_at_least` was also checked against every slider wall for 2,000,000 random needs
  (log-uniform, 0.05 to 20 mm): it never showed a need too low and never disagreed with a wall.
- The commit message's and the record's "19 ... now none" is verified.

Display range and edge inputs:
- 4.321e-4 shows "4.321e-4 mm", 9.9999e-4 "0.001000 mm", 1.234567e6 "1.235e6 mm" and 999999.9
  "1.000e6 mm", so scientific display rounds up too.
- NaN, +inf, 0 and -1 show as `mm` shows them.

#### Traps (none written)

- **Slice-2 section 8** and **slice-1 section 8**: nothing in the line touches a trap. The change
  moves only the last digit of a number already reviewed.
- **T-I** (the rule reads 12 to 16 % thin): rounding up nudges the shown minimum the
  conservative way, by less than 1 µm. The line still attributes the number to "Wall rule" and
  claims no sufficiency. The cup wall guide's learn note, `a5.low_saturation`, still carries
  the margin caution.
- **T-J, T-K, T-L**: no temperature, saturation figure or torque effect.
- **The added trap** (more magnets make the coupling stiffer): not touched.

#### Caps and plain words

- The wording is byte-equal to the reviewed text: 21 words. Only the number's rounding changed.
- The geometry notes have no row in the caps table.
- The number is a computed result. The input's name comes from `label_for(CUP_WALL_INPUT)`.
- No cell, correction id, file or code name appears.
- `plain_text_violations` is empty on all 379,620 sweep lines and all 7 named designs.

#### Docs checked against the code

- README `README.md:418`, "rounded up at its last digit (`mm_at_least`), so a wall shown equal
  to it passes": true on every reachable design. It is exact to within the ulps of SF-1.
- README geometry test row: true for the tests it describes. "A length already on a digit
  stays" is false as a general statement for 40 of 9,000 on-digit lengths between 0.1 and 10 mm
  (SF-1).
- `02-system.yaml` sentence: true.
- Tracker bullet: true. "The rule's wall is 1.9501 mm, shown 1.950 mm, now 1.951 mm" was
  probed (1.950115505 mm).
- BL-056: true. Its should-fix list matches re-review 1's.
- The record:
  - "Re-review: code-review fix round 1": an accurate account of the round.
  - Its copied report: as returned, with the caveats under SF-3.
  - "Task review fix round 1": the sentence in SF-3 is garbled.

#### Pre-existing, for the record

- **Off-grid walls from a design file or share link.** Typed values snap to the step, but a
  file can carry any wall. Near the rule, the wall shown to the nearest digit can then read one
  digit below the rounded-up rule while the check passes. Probed: at 12 poles and 1.23 T (need
  1.9501155 mm), a 1.9503 mm wall passes, and the line reads "at least 1.951 mm ...; ... is
  1.950 mm". This is the conservative direction, inherent to showing a 4-digit wall, and not a
  false safety claim.
- **Non-finite input from a design file:** "at least +inf mm" or "at least NaN mm", as recorded
  by re-review 1. Unchanged by this round.
