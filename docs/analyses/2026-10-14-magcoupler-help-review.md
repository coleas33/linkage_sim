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

### Re-review: changed by the implementer after its review

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `9d68db5`
  (`task-12b-textreview-1.md`). It re-reviews the strings that T12b, the shaft-clamp pause
  (controller ruling 11), changed after their review, so that no shown text names the clamp while
  the section is paused (ruling 11, 3A).
- Scope: nine changed items.
  - From this batch: the `coupling.bore_mm` help (item 4 above), both manual Br helps, the
    magnet-length hint, the Size tab's overall-length hint (`LENGTH_HINT_KEY`), and the workflow
    group "Shaft and hub joint" (was "Shaft, key and clamps") with its first section.
  - From batch B: the `metal.boss_od_mm` help.
  - From the engine wording: the help of `materials.parts.cap_housing`.
  - From the explorer: the Start here hover (`start_here_hover(true)`).
  - The one note field, `torque_from_shear` change[0], is recorded under that note in
    `docs/analyses/2026-10-14-magcoupler-notes-review.md`.
- Verdict: MUST_FIX, one item (MF-1, the bore help). The other eight items pass.
- MF-1, applied verbatim (the `coupling.bore_mm` help):
  - Reviewed (round 1's must-fix item 4 above, passed in round 2): "The shaft diameter under the
    hub and clamp. Larger lets each screw carry more torque but moves the screws outward, so fewer
    sizes may fit, and thins the hub wall."
  - T12b's text: "Diameter of the shaft in the hub bore. Larger thins the hub wall left past the
    keyway, and too large can break through the hub."
  - Now: "Diameter of the shaft in the hub bore. Larger thins the hub wall: the hub check can
    fail, and the wall past the keyway can break through." (27 words, 2 sentences, no
    plain-words problem.)
  - Why: T12b's text named only the wall past the keyway. A larger bore also thins the hub wall
    under the magnets (`model.hub_wall_mm`), which the hub check (`model.hub_check`) reads. The
    results table badges that check and its failing filter lists it.
    - On the 8-magnet probe design the hub check reads "Too thin" from a 15.5 mm bore, keyed or
      keyless: the hub wall is 2.350 mm against the back-iron rule's 2.399 mm. The wall past the
      keyway is still 0.65 mm there, and 0.40 mm at the 16 mm slider top. The keyway breaks
      through at none of the five probe designs on the bore slider.
    - For a keyless joint (BL-064's likely shrink fit), no slider design breaks through: the wall
      is at least 0.95 mm, at the 9 mm ring and the 16 mm bore.
  - Test: `the_joint_neutral_texts_claim_what_the_engine_does` (`gui/plain_text.rs`) now also
    pins, from the engine:
    - the help names the hub check by its display label (`label_for("model.hub_check")`);
    - on the 8-magnet probe design, with the design's keyway (1.7 mm) and keyless (the keyway
      slider's minimum), the bore's slider top turns `dashboard::check_level(..,
      "model.hub_check")` from `Some(Level::Good)` at the design's bore to `Some(Level::Bad)`;
    - there, `model.hub_wall_mm` falls (5.1 mm to 2.1 mm) and `model.hub_wall_past_key_mm` stays
      above zero, so the check fails before any break-through.

    The test keeps its smallest-ring break-through assertion for "can break through".
- Should-fix, both optional, left open for the controller:
  - SF-1: the `metal.boss_od_mm` help no longer tells the two bosses apart once the section
    returns. The cheapest fix is a note on BL-060.
  - SF-2: the cap material help's "6061-T6 uses the inputs" covers the density and specific heat,
    not the conductivity.
  - The reviewer also notes, for the controller, the housing section "Optional adapter and its
    joint", whose three joint inputs are hidden while paused.
- Correction (text re-review 2): keyless, the input sliders leave at least 0.80 mm of hub wall (the
  9 mm ring, the 16 mm bore and the 0.2 mm bondline), not 0.95 mm; and the Size tab's lowest
  Minimum OD takes the ring radius down to the bore radius plus the bondline, where the wall is
  zero keyless too. The bore help's "can break through" holds either way.

The reviewer's report for the record, as returned:

T12b text re-review 1 (2026-10-15, fresh-context physics reviewer, session model). Commit
`9d68db5`. Nine changed items; one must-fix.

- **`torque_from_shear` change[0]: PASS.**
  - The text: "torque rises almost linearly; on the example design each millimetre beats a
    millimetre of radius. Costs: longer stack, higher shaft-joint and gearbox loads." (23 words.)
  - The new cost holds at every design. T_2D is proportional to L, and f_end = 1 − c_end·τ_p/L
    rises with L, so the pull-out rises with L. The cold-high torque (`metal.torque_cold_high_Nm`)
    and its gearbox-input value (`metal.cold_high_input_Nm`) scale with the pull-out.
  - `the_joint_neutral_texts_claim_what_the_engine_does` pins all three one slider step up at the
    example design. The dump reads "moves as it says" at all five probe designs.
  - The text names no joint method and does not say the coupling protects the gearbox (spec 8,
    T-G, ruling 8).
- **HELP `coupling.bore_mm`: MUST-FIX MF-1.**
  - The new text names only the wall past the keyway. A larger bore also thins the hub wall under
    the magnets, which the results table's Hub check (`model.hub_check`) reads.
  - At the 8-magnet probe design the hub check turns "Too thin" from a 15.5 mm bore, keyed or
    keyless, with 0.65 mm still past the keyway.
  - For a keyless joint (BL-064's likely shrink fit), break-through is unreachable on any slider:
    the wall is at least 0.95 mm.
  - Replacement: "Diameter of the shaft in the hub bore. Larger thins the hub wall: the hub check
    can fail, and the wall past the keyway can break through." (27 words.)
- **HELP `manual_inner_br_T` / `manual_outer_br_T`: PASS** (28 words each).
  - "Stronger raises torque; shaft-joint and gearbox loads rise with it." holds:
    - the pull-out rises with either ring's Br;
    - manual blocks leave the calibration factor at its default, since the bench factor needs
      B842SH on both rings (`model.rs:583-599`);
    - the cold-high torque and its gearbox-input value follow the pull-out.
  - The test pins both helps at the example design with the parts cleared.
- **HELP `metal.boss_od_mm`: PASS** (23 words).
  - The drawn boss is `metal.boss_od_mm` (`geometry.rs:663`). It sets `mass.boss_g` and the
    adapter's mass.
  - It sets the Size tab's outside diameter when larger than the cup and cap (T-P; pinned at
    `plain_text.rs:2005-2030`).
  - The section-8 boss trap concerns the hidden clamp input.
  - SF-1 (optional) records the lost two-boss distinction for un-pausing.
  - Not changed by T12b: "outside diameter" here is the Size tab's canonical one (spec 3.5).
    `metal.rotating_od_mm` leaves the boss out (BL-061).
- **HINT `coupling.magnets.axial_length_mm`: PASS** (12 words, at the cap).
  - "Blank keeps ring lengths; longer adds torque, length, shaft-joint and gearbox loads."
  - Pinned: the pull-out, the cold-high torque, its input-equivalent value and the overall length
    all rise one step up.
- **HINT `LENGTH_HINT_KEY`: PASS** (12 words). It covers the same claims as the axial-length hint.
- **help `materials.parts.cap_housing`: PASS** (23 words).
  - The adapter's mass uses the aluminium density input whatever the pick.
  - SF-2 (optional) covers the existing "uses the inputs" against the fixed 6061-T6 conductivity.
- **WORKFLOW "Shaft and hub joint": PASS.**
  - Paused, the group holds the bore and the keyway depth. Unpaused, it holds the key and the
    clamp sections too; the label fits both states and names no method.
  - No "Shaft, key and clamps" is left in `magcoupling-rs/src` or the README.
- **`start_here_hover(true)`: PASS.**
  - Its nine topics map onto the eleven steps of `start_here_for(true)`, in order. "Slipping and
    its heat" covers slip heating and the thermal time constant. "Demagnetization" covers the
    demagnetization and ferrite cold-side notes.
  - The unpaused form is the reviewed text, unchanged.
  - Neither form contains "clamp" when paused, and both pass `plain_text_violations`.
- **Traps:** no text writes a trap from slice-1 section 8, slice-2 section 8, T-A to T-P, or the
  stiffness trap.
- **Rules:** none says a joint method is chosen or sized, or that the coupling protects the
  gearbox. Every cap holds.
- **Outside the listed items, for the controller:** the housing section "Optional adapter and its
  joint" (`inputs.rs:311`) keeps its label while its three joint inputs (`clamps.joint_*`) are
  hidden. Paused, it shows only the six adapter inputs. The label is joint-neutral and has no clamp
  word; leave it, or show "Optional adapter" while paused.
- **Scratchpad mishap:** while setting up the probe I overwrote `scratchpad/probe/Cargo.toml` and
  `scratchpad/probe/src/main.rs`. Both belong to an earlier session agent's scratch crate, which
  has bins `p1` to `p4` and an egui `lib.rs`; its `lib.rs` no longer builds against the new
  manifest. My probes ran from `scratchpad/t12b-rr1-probe`. Nothing in the repository changed.

### Re-review: changed by the implementer after its review (re-check 2)

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `2c9a0d1`
  (`task-12b-textreview-2.md`): the re-review above's MF-1 as applied. This is the second text
  re-review of the strings T12b changed after their review.
- Scope: one changed item, the `coupling.bore_mm` help (item 4 above), "Diameter of the shaft in
  the hub bore. Larger thins the hub wall: the hub check can fail, and the wall past the keyway can
  break through." (27 words, 2 sentences), with the assertions it added to
  `the_joint_neutral_texts_claim_what_the_engine_does`.
- Verdict: PASS, no must-fix.
  - "Larger thins the hub wall": the wall falls one-for-one with half the bore, at all five probe
    designs with the design's keyway, keyless and the deepest keyway. Auto with the ring radius
    free keeps the ring at 11.676 mm for bores of 10 to 16 mm, so the wall still thins.
  - "the hub check can fail": on the 8-magnet probe design from a 15.5 mm bore, keyed or keyless,
    and at the smallest ring. It cannot fail where it does not apply ("No back iron", the 304 probe
    design).
  - "the wall past the keyway can break through": at the smallest ring with the design's keyway,
    and from a 12.2 mm bore with the deepest keyway at four of the five probe designs.
  - Not named, rightly: the key and clamp results (hidden while paused), and the lighter coupling
    and shorter thermal time constant, which change no check.
  - The test's new assertions each check one claim against the engine.
  - Caps, plain words and traps: 27 words, no violation, no trap written.
- Should-fix SF-1 (optional, record only), applied: the correction line in the re-review above.
  Keyless, the input sliders leave at least 0.80 mm of hub wall, not 0.95 mm, and the Size tab's
  lowest Minimum OD reaches a zero wall. The reviewer asked that the quoted report stay as
  returned, so the line is added and the quote is not edited. No code, test or on-screen text
  changed in this step.
- Left open for the controller (both outside the changed item):
  - At a keyless zero wall reached through the Size tab's Minimum OD, the Size tab's red line reads
    "Exploration only: the keyway breaks through the hub." (`gui/panel.rs`), though there is no
    keyway.
  - The input label "Keyed bore diameter" is the engine's Python-locked label. It will read oddly
    beside a keyless shrink fit (BL-064).

The reviewer's report for the record, as returned:

T12b text re-review 2 (2026-10-15, fresh-context physics reviewer, session model). Commit
`2c9a0d1`. One changed item. Verdict: PASS, no must-fix.

- **HELP `coupling.bore_mm`: PASS.**
  - The text: "Diameter of the shaft in the hub bore. Larger thins the hub wall: the hub check can
    fail, and the wall past the keyway can break through." (27 words, 2 sentences, no plain-words
    or glyph problem.) It is re-review 1's MF-1 replacement, verbatim.
  - **Engine:** `hub_wall = a_i − bond_inner − bore/2` (`model.rs:931`) feeds the hub check, which
    reads "Too thin" below t_bi (`:1043-1052`, `:1168`). The wall past the keyway is
    `hub_wall − keyway` (`:1084`). t_bi does not depend on the bore.
  - **Reproduced:**
    - At the 8-magnet probe the hub check turns "Too thin" from 15.5 mm, with the 1.7 mm keyway,
      keyless and the 4 mm keyway. The hub wall is 2.350 mm against 2.399 mm, with 0.650 mm past
      the 1.7 mm keyway.
    - At the 16 mm bore the hub wall is 2.100 mm (from 5.100 mm) and 0.400 mm is left past the
      keyway.
    - At the smallest ring and the 16 mm bore, keyed, both fail (0.950 mm against 1.762 mm;
      −0.750 mm past the keyway). Keyless, only the hub check fails.
    - The 4 mm keyway breaks through from 12.2 mm at four probe designs.
  - **Auto** (ring radius free, default target) keeps the ring at 11.676 mm for bores of 10 to
    16 mm, so the wall still thins. Auto never makes the bore its free variable.
  - **Not named, rightly:**
    - the key and clamp effects (hidden while paused);
    - the 16 % lighter coupling and its 16 % shorter thermal time constant (no check changes,
      +0.04 °C peak, no room in the cap, close to the slip-heating trap).
  - **Test:** `the_joint_neutral_texts_claim_what_the_engine_does` pins, from the engine:
    - the help names the hub check by its display label;
    - at the 8-magnet probe, keyed and keyless, the bore's slider top turns the hub check from Good
      to Bad, lowers the hub wall and leaves wall past the keyway;
    - the smallest-ring break-through.
- **Traps and rules:** no trap from slice-1 section 8, slice-2 section 8, T-A to T-P, or the
  stiffness trap. No joint method, sizing or gearbox claim. Every cap holds.
- **SF-1 (optional, record only):**
  - The help-review record's keyless floor "at least 0.95 mm" is 0.80 mm at the 0.2 mm bondline.
  - The Size tab's lowest Minimum OD reaches a zero hub wall keyless too (ring radius 8.05 mm).
  - An exact line to append is given above.
- **For the controller, outside the changed item:**
  - At a keyless zero wall reached through the Size tab's Minimum OD, the Size tab's red line
    reads "Exploration only: the keyway breaks through the hub." (`panel.rs:916`), though there is
    no keyway. It is an existing GUI string, reachable only at the Minimum OD floor with a keyless
    joint.
  - The input label "Keyed bore diameter" is the engine's Python-locked label, so it cannot be
    renamed in this slice. It will read oddly beside a keyless shrink fit (BL-064).
- **Probes:** a fresh scratch crate, `scratchpad/t12b-rr2-probe` (bins `main`, `moved`, `floor`),
  with its own target directory. No other scratch crate was touched, and nothing in the
  repository changed apart from this file.

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

### Re-review: whole-branch review fix wave 1

The slice-2 whole-branch review's fix wave 1 (2026-10-16; `final-review.md` MF-5) changed two
assumption rationales of this section (`engine/assumptions.rs`): `end_effect` (the reviewed
note's band, scope and steel case, in place of R10's literal "within 2 %", deviation D-5) and
`calibration_factor` ("no magnetic steel ... B842SH parts"). Their G-3 re-review ran in a fresh
context. The reviewer's report for the record follows unchanged (full answer:
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-21-textreview-1.md`).

- Reviewer: physics reviewer (session model), fresh context, 2026-10-16; G-3 review of the two
  assumption rationales changed by the T21 whole-branch review fix wave 1 (MF-5),
  `task-21-textreview-1.md`.
- Bar: plain words for engineers who are not magnetics specialists, with no cells, audit ids or
  code names; nothing false at the default design, the five probe designs or anywhere the inputs
  reach; agreement with the reviewed `end_effect` and `calibration` notes.
- **`calibration_factor`: PASS.**
  - "No magnetic steel behind the magnets, its magnets per ring, B842SH parts on both rings" is
    the engine's rule (`select_calibration_factor`, called with the resolved circuit; a
    non-ferromagnetic pick forces circuit 0).
  - The 304 probe design matches and takes f_cal = `calibration.f_cal_updated` = 1.0494.
  - "Parts" is exact: the rule compares the pick.
  - It agrees with the `calibration` note's sentences[2].
- **`end_effect`: MUST_FIX (MF-B).**
  - The new band and scope are true and agree with the note: within 3 % on the example design
    without its steel from 3.2 mm, likely high with magnetic steel.
  - But the steel clause has no scope. It reverses at reachable short magnets on steel designs,
    with the check "OK":
    - the 20 mm ring probe, −84.7 % at 2.36 mm;
    - the 8-magnet probe, −47.0 % at 2.0 mm;
    - the example design at the 2.0 mm floor, −1.3 to −4.0 %.
  - There the model understates the torque, which is the unsafe side for cold-high loads.
  - The note qualifies the same clause in its watch_out[1]; the rationale, drawn on its own, does
    not.
  - Replacement for the last sentence: "It is least reliable for short magnets, where it can read
    too high or too low; at or below a magnet length of coefficient × pole pitch the factor is
    zero or negative, and the model gives no prediction."
  - Test: the rationale names "short magnets" and "too high or too low".
- **Plain words:** both rationales pass `every_label_and_rationale_reads_in_plain_words`, and so
  does the replacement (nothing in `BANNED`).
- **Verdict:** MUST_FIX (MF-B, end_effect); `calibration_factor` passes.

- MF-B, applied verbatim by the implementer (session model), 2026-10-16:
  - `end_effect` rationale, last sentence: "At or below a magnet length of coefficient × pole
    pitch the factor is zero or negative, and the model gives no prediction." -> "It is least
    reliable for short magnets, where it can read too high or too low; at or below a magnet
    length of coefficient × pole pitch the factor is zero or negative, and the model gives no
    prediction."
  - Test: `tests/assumptions.rs`
    `the_end_effect_and_calibration_rationales_state_what_was_checked` also asserts that the
    rationale names "short magnets" and "too high or too low". The previous text fails it.
    `every_label_and_rationale_reads_in_plain_words` passes.
  - Docs: the README `tests/assumptions.rs` row, the update tracker's MF-5 sentence and the MF-5
    commit message.

### Re-review: whole-branch review fix wave 1 (re-check 2)

The re-review above ended on a MUST_FIX (MF-B), applied verbatim in b4ed1cf. The applied
`end_effect` rationale had its G-3 re-review in a fresh context at HEAD f4f9624. The reviewer's
report for the record follows unchanged (full answer:
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-21-textreview-2.md`).

- Reviewer: physics reviewer (session model), fresh context, 2026-10-16; G-3 re-review of the
  `end_effect` assumption rationale as applied for text re-review 1's MF-B,
  `task-21-textreview-2.md`.
- Bar: plain words for engineers who are not magnetics specialists, with no cells, audit ids or
  code names; nothing false at the default design, the five probe designs or anywhere the inputs
  reach; agreement with the reviewed `end_effect` note.
- **`end_effect`: PASS.**
  - The last sentence now reads "It is least reliable for short magnets, where it can read too
    high or too low; at or below a magnet length of coefficient × pole pitch the factor is zero
    or negative, and the model gives no prediction."
  - It is true at every circuit checked. The largest error lies at 0.16 to 0.42 τ_p, and both
    directions occur with the check "OK": +15.6 % at 2.5 mm on the example design with steel,
    −20.7 % on the 304 design at the 2.0 mm floor, −84.7 % at 2.36 mm on the 20 mm ring with
    steel.
  - The steel clause's unsafe short-magnet exception is now stated, as in the note's watch_out[1].
  - The band (within 3 % from 3.2 mm, the example design without its steel) is read from the note
    by the test. The steel case ("likely reads high", ideal-steel evidence at both permeability
    limits) and the wide-gap clause are unchanged and true.
  - The test asserts "short magnets" and "too high or too low".
- **Plain words:** `every_label_and_rationale_reads_in_plain_words` passes. Every draw site
  (Assumptions view, hover, Explain's nearest assumption, spreadsheet) shows the whole text.
- **Should-fix (for the controller):** S-1. In sentence 3, "it is within 3 %" becomes "the
  factor is within 3 %", because the nearest antecedent is "coefficient". The exact text is in
  `task-21-textreview-2.md`.
- **Verdict:** PASS. MF-B is closed.

- No must-fix, and no text changed in this round. S-1 ("it is within 3 %" -> "the factor is
  within 3 %" in sentence 3) stays open for the controller, with the `end_effect` note's carried
  should-fix items SF-A (extended), SF-1 and SF-T (texts in `task-21-textreview-1.md`, S-1 to
  S-3).

## Reachability strings (T19)

- Scope: not an overlay batch. These are T19's on-screen strings outside the help overlay,
  reviewed under the same G-3 rule (the brief's Step 13 prompt):
  - the Equation panel's own lines (`gui/explorer.rs`): the empty text (`EMPTY_TEXT`), the line
    under a value with nothing better to offer (`NO_NOTE`), the Start here hint
    (`START_HERE_HINT`) and the two leads (`NEAREST_NOTE`, `NEAREST_ASSUMPTION`);
  - the three Why-link texts outside the dashboard rows (`dashboard::why_note_text`): under
    Magnets per ring and Outside diameter on the Size tab (`guidance::SIZE_TAB_NOTE`), at the end
    of the Required and Hot-low rows (`TORQUE_BAND_NOTE`) and under each copy of the end-effect
    banner (`BANNER_NOTE`);
  - for each dashboard row shown without a note of its own, what Explain offers
    (`explorer::note_fallback`), judged as a sensible next read for that value whose lead does
    not claim the note explains the value itself (the prompt's item 3). The clamp row, hidden
    while the clamp section is paused, is not drawn and was skipped (A-28g).
- Read: the Step 12 dump (`magcoupling-rs/target/t19-strings.txt`, 237 strings; its 17 grep
  lines in `t19-strings-grep.txt`) and a review sheet (`t19-review-sheet.txt`: where each string
  is drawn, each row's offer and how the walk reaches it, the offers off the dashboard, and the
  grep verbatim); T19's brief, report and drafter notes; the amendments (Global Constraints, the
  T19 row: A-4, A-25 T-6, A-28g; A-13, A-19 to A-24, ruling 11 and A-28); slice-2 spec 3.4, 3.7,
  5, 6 and 8, slice-1 spec 8, `phys-new-notes.md` section 2 and synthesis decision 8; the code
  behind every string. Each round probed from a scratch crate outside the repository:
  - round 1: each dashboard row's offer, recomputed with `note_fallback` and an independent
    level-by-level walk; every reachable equation without a note of its own, by offer (closest
    note 36, assumption 10, row reason 3, `NO_NOTE` 82); the values a note discusses that still
    drew `NO_NOTE`; the guard, the word counts and "clamp" over every line; the three must-fix
    texts in a mirror of the crate (lib suite, release gates, dump);
  - round 2: every place `NO_NOTE` is drawn, the no-record arm included (594 values off the
    dashboard with no equation record); `explorer_ui` drawn headless with Explain on at three of
    them; what they are computed from (the engine and `notes::note_for`); the record arm checked
    sound (none of its 82 values reads a record-less term); the must-fix text in a mirror;
  - round 3: the form checks on `NO_NOTE`; its counts per arm (82 under an equation, 594 without
    a record, 0 on the dashboard); `explorer_ui` drawn headless under an equation, under
    record-less values at the five probe designs, and with the Start here bar up after three term
    clicks.
- Round 1: 3 must-fix items, text applied verbatim (all three constants in `gui/explorer.rs`):
  - `START_HERE_HINT`.
    - Old: "Start here walks the teaching notes in order."
    - New: "Start here walks the main teaching notes in a suggested order."
    - Why: Start here has 13 steps while 19 notes are shown; the six warning notes are never
      steps, and one of them is this task's offer for the rotating mass. "A suggested order" is
      the Start here button's own hover wording.
  - `EMPTY_TEXT`, the same clause.
    - Old: "Hover a value to see its equation; click it to open it here, then click a term to
      follow it. Start here walks the teaching notes in order; Explain shows the note under the
      open equation."
    - New: "Hover a value to see its equation; click it to open it here, then click a term to
      follow it. Start here walks the main teaching notes in a suggested order; Explain shows the
      note under the open equation."
  - `NO_NOTE`.
    - Old: "No teaching note covers this value."
    - New: "No teaching note is attached to this value or to anything it is computed from."
      (replaced again in round 2)
    - Why: the walk looks only upstream, so "covers" was false for values a note explains from
      downstream: slip-loss power, the operating temperature, the effective gap and Br at 20 °C.

  Should-fix offered, not applied in the round: SF-1, `NO_NOTE`'s doc comment follows the new
  text; SF-2, the Step 14 docs quote the post-review texts; SF-3, a backlog candidate: a short
  curated path-to-note map for values the upstream walk misses.

  Recorded, no change asked: the two leads, the three Why-link texts and all nine fallback offers
  pass. `mass.total_g`'s offer, "No magnetic steel behind the magnets" (the drafter's flagged
  risk), is accepted: it is the only reviewed note upstream (two steps, through the back iron's
  circuit), it is the coupling's main mass trade-off, and the lead says "closest", not
  "explains". "Closest teaching note" over a row's own Why note is accepted. The implementer's
  decisions 1-6 change no drawn string.
- Round 2: 1 must-fix item, applied verbatim:
  - `NO_NOTE`.
    - Old: "No teaching note is attached to this value or to anything it is computed from."
    - New: "No teaching note is shown here for this value."
    - Why: Explain also draws it under each of the 594 values off the dashboard with no equation
      record, where nothing is known of what the value is computed from. Some of them are
      computed from values that carry notes: the six warnings (each rule names its note), the
      adhesive's cold-high torque copy and its tangential force, the calibration's end-effect
      check and the model verdict. "here" anchors the claim to the Equation panel.

  Passed: `EMPTY_TEXT`, `START_HERE_HINT` and the four fallback lines that end in the hint.
  Should-fix: SF-1 restated for the new text; SF-2 extended to the review sheet; SF-3 extended
  (a warning's path could offer its rule's note; a curated map for the record-less copies).
- Round 3: PASS, no must-fix. It re-reviewed `NO_NOTE` and the lines drawn next to it: true under
  the 82 equations, under the 594 values with no record and with the Start here bar up, and still
  true if a note is ever sent back to Draft. Its should-fix items restate round 2's SF-1 to SF-3.

  Recorded, outside the review: where a warning does not fire, its value line reads
  "Non-ferromagnetic back iron = " with nothing after the "=". That is the value formatter's
  existing behaviour, not a T19 string.
- What happened to the should-fix items (after the PASS, before the commit):
  - SF-1 applied: `NO_NOTE`'s doc comment is the reviewers' text, verbatim. It is a code comment,
    not on-screen text, and it had been stale since round 1.
  - SF-2 applied: the README's Teaching notes bullet, `docs/ai/05-update-tracker.md` and the
    BL-056 notes quote the final texts; the sheet already did.
  - SF-3 not built: the BL-056 notes record the curated map as a slice-3 candidate, each entry to
    be checked against the engine first. Open for the controller.
- Seeds: the T19 index row names none of A-24's seeds, so there was no seed to accept or reject
  (all three rounds say so).
- Amendments: A-4 applied (`NoteLink::Note`, which `Explorer::follow_note_link` opens;
  `note_fallback_ui` returns `Option<NoteLink>`; `equation_body` keeps its 7 parameters, with no
  `asked_note`; the Size-tab links are anchored after `weak_line`); A-25 T-6 applied
  (`a_wrapped_change_entry_stays_inside_a_narrow_panel_and_its_label_links`); A-28g applied (the
  predicate is `clamps::hides`; the guidance test asserts `notes::shown`; both explorer tests and
  the dump skip what the paused section hides and assert that no offer is a note not shown; the
  panel tests are unchanged). A-13: no drawn field carries a cell, a correction or audit id, or a
  file or code name. A-19 to A-23 are other tasks' texts, and no T19 string contradicts them.
- Traps: none of spec section 8's, slice-1 spec section 8's, `phys-new-notes.md` T-A to T-P, "more
  magnets make the coupling stiffer", or A-28's (a shaft-joint method chosen or sized, the
  coupling protecting the gearbox) is written in any T19 string, and none names a clamp.
- Verdict: PASS (round 3).

### Report for the record, round 1 (as returned)

#### Per line (the 17 lines of Step 12's grep)

| # | Line | Verdict | Reason |
|---|---|---|---|
| 1 | explorer/empty text | MUST-FIX 2 | "walks the teaching notes": 13 of 19 shown. The first sentence and the Explain clause are true. |
| 2 | explorer/no note | MUST-FIX 3 | "covers" is false for values a note explains downstream (slip-loss power, the operating temperature, the effective gap, Br at 20 °C). |
| 3 | explorer/start here hint | MUST-FIX 1 | As line 1. |
| 4 | explorer/closest note lead | PASS | See [Line 4](#line-4-the-closest-note-lead). |
| 5 | explorer/assumption lead | PASS | See [Line 5](#line-5-the-assumption-lead). |
| 6 | why link/size tab, under Magnets per ring and Outside diameter | PASS | See [Line 6](#line-6-the-size-tab-link-to-torque_from_shear). |
| 7 | why link/size tab, end of the Required and Hot-low rows | PASS | See [Line 7](#line-7-the-required-and-hot-low-link). |
| 8 | why link/end-effect banner | PASS | See [Line 8](#line-8-the-end-effect-banner-link). |
| 9 | explain fallback/`model.cup_od_mm` | PASS | See [Line 9](#line-9-modelcup_od_mm). |
| 10 | explain fallback/`mass.total_g` | PASS | See [Line 10](#line-10-masstotal_g-the-drafters-flagged-risk). |
| 11, 12 | explain fallback/the two clearance rows | PASS | Nothing upstream (28 and 30 paths); "No note covers the running-clearance stack yet." is true. |
| 13-15 | explain fallback/the three temperature rows | PASS | See [Lines 13-15](#lines-13-15-the-three-temperature-rows). |
| 16 | explain fallback/`housing.space_claim_check` | PASS | No record; the reason is true. |
| 17 | explain fallback/`model.end_effect_check` | PASS | The check's own note (row link). The value has no record, so `no_equation_body`'s line sits above it. |

##### Line 4: the closest-note lead

- "Closest" claims nearness, not that the note explains the value (prompt item 3).
- Where the row's own link supplies the note (the temperature rows, the end-effect check), "closest" understates but does not mislead.

##### Line 5: the assumption lead

- It is offered only for an assumption actually upstream.
- No dashboard row gets it today; 10 results off the dashboard do, from four assumptions.
- Their rationales are plain words with no "clamp", and none of the paused clamp assumptions is reachable (probe).

##### Line 6: the Size-tab link to `torque_from_shear`

- The note covers the magnet count and the ring radius. A-19's sentence ("The magnet count N keeps the gap radius R_g ... fewer flat blocks grow the cup") is where the 10 -> 8 confusion is answered (synthesis 8-A).
- The "Why: " prefix keeps F11's single bare step-1 title.
- On an odd magnet count the Outside diameter control and its link are absent; a test pins this.

##### Line 7: the Required and Hot-low link

- It leads to the hot minimum check's own note (`HOT_MIN_GUIDE.learn`), which compares hot-low at the operating temperature with the requirement (T-E).
- Both rows lead to the same note.

##### Line 8: the end-effect banner link

- It leads to the end-effect check's note, which explains f_end = 1 − c_end·τ_p/L and why there is no prediction at or below c_end·τ_p.
- It is drawn under both copies of the banner, and only while the banner shows.

##### Line 9: `model.cup_od_mm`

- Its equation reads the pocket corner radius and the wall. The 20 upstream paths are geometry and inputs, none owned by a note, with no assumption.
- "No note covers the cup's outside diameter yet." is true. `torque_from_shear` says only that fewer blocks grow the cup.
- I reject making `torque_from_shear` this row's link: it would put a torque note on the dashboard's cup-OD Why line.

##### Line 10: `mass.total_g`, the drafter's flagged risk

**Ruling: accept.**

**Why it is the offer.** It is the only reviewed note anywhere upstream: `materials.circuit_backiron`, two steps up through the cup, hub and boss masses. Those masses take the steel density with a steel circuit and the body material's otherwise.

**Why it is a sensible next read.** It is the coupling's main mass trade-off:
- sentence 5: "The hub, cup and boss then become non-magnetic metal";
- the three change entries: "steel's mass", "adds mass", "coupling grows longer and heavier".

**Why the line is not false.** The lead says "closest", not "explains".
- At the default (4140, steel circuit) the title describes the other case. Drawn as a link after "Closest teaching note:", it is a note's name, not a status line.
- At the 304 probe design it matches the design.

**No better offer exists.** There is no mass assumption, and no other note is upstream. A `ROW_LEARN` change would also rewrite the dashboard row's own "No note covers the rotating mass yet."

##### Lines 13-15: the three temperature rows

**What the offer is.** These rows show the row's own Why note (`TEMPERATURE_GUIDE.learn`, reviewed). It is also the nearest note upstream:
- the governing limit reads `temperature.demag.magnet_limit_C` directly;
- the hot-day margin reads it at two steps;
- the verdict reads `margin_limit_C` at one step.

**The limit's other operand.** The adhesive design limit has no note, so the demagnetization note is the right read for these rows.

#### Traps

None of the 17 lines, and none of the three replacements, writes:
- a slice-1 spec section 8 trap;
- a slice-2 spec section 8 trap;
- a `phys-new-notes.md` T-A to T-P trap;
- the added "more magnets make the coupling stiffer";
- the A-28 trap: a joint method chosen or sized, or the coupling protecting the gearbox.

The lines are navigation text and note titles, and the titles were reviewed with their notes (T11, T13, T14, T16, T17).

#### Rulings on the implementer's concerns

- **`mass.total_g`'s offer:** accepted, as above; no `ROW_LEARN` change.
- **"Closest teaching note" over a row's own note:** accepted. A separate lead for a row's own link would add a constant and a branch for no gain in truth.
- **Decisions 1-6 of the report:** none changes a drawn string, so I have no wording ruling to make.
- **The planned one-line BL-056 entry:** fits the Global Constraints' "docs change with the code"; it should carry the post-review texts.

#### For round 2

Re-review only the three constants, and the dump lines that show them: lines 1-3, and the hint inside lines 9, 11, 12 and 16. The mirror run above shows the suites green with them.

### Report for the record, round 2 (as returned)

#### Per line (the changed lines of Step 12's grep)

| # | Line | Verdict | Reason |
|---|---|---|---|
| 1 | explorer/empty text | PASS | See [Lines 1 and 3](#lines-1-and-3-the-start-here-clause). |
| 2 | explorer/no note | MUST-FIX 1 | False under record-less values (594): the warnings, the cold-high copy and its tangential force, the calibration's end-effect check, the model verdict. |
| 3 | explorer/start here hint | PASS | As line 1. 11 words, within the 12-word hint cap. |
| 9, 11, 12, 16 | explain fallback / `model.cup_od_mm`, the two clearance rows, `housing.space_claim_check` | PASS | Each row's own reason (unchanged, passed in round 1), then the hint, which is true as line 1. |

The other ten lines (4-8, 10, 13-15, 17) are byte-for-byte as in round 1, which passed them.

##### Lines 1 and 3: the Start here clause

- **"main teaching notes" is true.** Start here walks 13 of the 19 notes shown. The other six are the warning notes, reached from their warnings, their rows and Explain.
- **"in a suggested order" is true.** These are the Start here button's own hover words (`START_HERE_HOVER`, "A suggested order: ...").
- **The rest of `EMPTY_TEXT` is unchanged.** Round 1 judged it true. The ";" still ends the Start here clause, so the hint is a whole string of its own and the exact-match counts are unaffected (mirror run).

##### Line 2: `NO_NOTE`

See MUST-FIX 1:
- the record arm is sound (probe 4);
- the no-record arm is false (probes 1-3);
- the replacement passed the mirror's suites (probe 6).

#### Traps

None of the changed lines, and not the replacement, writes:
- a slice-1 or slice-2 spec section 8 trap;
- a `phys-new-notes.md` T-A to T-P trap;
- "more magnets make the coupling stiffer";
- the A-28 trap: a joint method chosen or sized, or the coupling protecting the gearbox.

They name no clamp.

#### Tests

No new test is needed for MUST-FIX 1: no test reads the words, as in round 1. These already pin the behaviour the new text describes:
- `a_dashboard_row_s_own_link_comes_first_and_a_value_with_no_record_has_nothing_upstream`: a record-less value off the dashboard gets `NO_NOTE`;
- `the_offered_note_is_the_nearest_upstream_and_a_note_outranks_an_assumption`: the record arm;
- `explain_never_shows_no_teaching_note_on_a_dashboard_row`: never on a row.

#### For round 3

Re-review only `NO_NOTE`, which is grep line 2: the hint lines do not contain it. The mirror run above shows the suites green with the replacement.

### Report for the record, round 3 (as returned)

#### Per line (the changed line of Step 12's grep)

| # | Line | Verdict | Reason |
|---|---|---|---|
| 2 | explorer/no note: "No teaching note is shown here for this value." | PASS | True in both arms and with the Start here bar up (probes 2-3); plain words, 9 words, no clamp; see "Line 2, judged". |

Lines 1 and 3-17 are byte-for-byte as in round 2, which passed them. Their verdicts stand:
- **Line 1:** PASS (round 2).
- **Line 3:** PASS (round 2).
- **Lines 4-8, 10, 13-15 and 17:** PASS (round 1).
- **Lines 9, 11, 12 and 16:** PASS (round 2).

#### Sheet and dump

- **The dump and the grep.** Step 12's grep over `t19-strings.txt` equals `t19-strings-grep.txt` (17 lines), and section D of the sheet equals it too (checked with `diff`).
- **The changed line.** Line 224 of the dump is the tree's constant.
- **The counts.** The sheet's count lines (82; and round 2's 594, attributed) agree with my probe 2.

#### Tests

No new test is needed. No test reads the words, and these already pin the behaviour the text describes:
- `a_dashboard_row_s_own_link_comes_first_and_a_value_with_no_record_has_nothing_upstream`: the no-record arm gets `NO_NOTE`;
- `the_offered_note_is_the_nearest_upstream_and_a_note_outranks_an_assumption`: the arm under an equation;
- `explain_never_shows_no_teaching_note_on_a_dashboard_row`: never on a row;
- `the_empty_text_and_explain_s_own_lines_name_the_buttons_in_plain_words`: guard and glyphs.

The implementer's round-2 reruns are green (`task-19-inreview-r2-changes.md`), and round 2's mirror run showed the same text green.

#### Outside this review (for the record, not T19's text)

- **An empty value.** Where a warning does not fire, its value line reads "Non-ferromagnetic back iron = " with nothing after the "=" (probe 3b, four of the five designs).
- **Not from T19.** That is the value formatter's existing behaviour, not a T19 string. It may deserve a look in a later slice.

#### For the controller

The in-task review loop for T19 ends at round 3 with PASS. Next come Step 14 (docs, with SHOULD-FIX 2's texts) and the commit.

## BL-059 gearbox check

- Scope: not an overlay batch alone. These are BL-059's new and changed on-screen strings
  (spec `docs/superpowers/specs/2026-10-16-bl059-gearbox-check-design.md`, section 6), reviewed
  under the same G-3 rule:
  - the gearbox guide (`gui/guidance.rs`): its label (`GEARBOX_GUIDE.label`), its Why
    (`GEARBOX_GUIDE.why`), and its Problem: sentence 1 (`gearbox_problem`), the conflict sentence
    2 (`gearbox_conflict`) and the text when the check cannot be made (`GEARBOX_UNKNOWN`);
  - the engine (`engine/gearbox.rs`): the three verdict texts (`WITHIN_THE_GEARBOX_RATING`,
    `ABOVE_THE_GEARBOX_RATING` with its ": {q} N·m against {r} N·m" tail,
    `GEARBOX_CHECK_UNKNOWN`), and the label and help of `gearbox.cold_high_check`, the
    dashboard row's label;
  - the overlay (`gui/plain_text.rs`): the help of the gearbox input torque rating
    (`coupling.gearbox_input_rating_Nm`), and the label and help of the ripple row
    (`model.gearbox_input_ripple_Nm`);
  - read in context, judged only for how the new words read there: the trade-off and face-gap
    side-effect lines in the gearbox block at the defaults and the status line ("Example design:
    fails 4 of 8 checks").
- Read: the sheets `magcoupling-rs/target/g3/BL059-guidance.txt` (`guidance_strings_for_review`),
  `BL059-overlay.txt` (`overlay_strings_for_review`) and `BL059-strings.diff` (`git diff -U0
  main` of the source files); the spec, sections 1 to 6; the plan's trap list; the two notes'
  gearbox watch-outs as Task 6 shipped them, for consistency only; the engine and GUI code behind
  every string. Each round probed the engine from a scratch crate outside the repository:
  - round 1 (2026-10-17):
    - the defaults: cold-high at the gearbox input 0.804907 against 0.6; the verdict "Above the
      gearbox input rating: 0.80 N·m against 0.60 N·m"; the Problem quotes 0.8049, 0.6000, 1.673,
      2.500, 0.8806 (0.880564), 1.703 (1.703454) and 1.140, with sentence 2;
    - the five probe designs, cold-high at the gearbox input: the defaults 0.8049, above; 8
      magnets 0.5893, within; 304 back iron 0.5082, within; Y30 0.07117, within (band 1.9207);
      a 20 mm ring radius 0.7602, above, with sentence 2;
    - the rating: at 0.80, "0.81 N·m against 0.80 N·m" with sentence 2 (at most 2.271 N·m, band
      1.520); at 0.81, within; at 0.05, the second form (at most 0.1419 N·m, the requirement
      alone 0.5264); at 5.0, within;
    - the passing design: within, no Problem; the "gearbox only" design: 0.7070, above,
      R·B/(i·η) = 0.5636, so no sentence 2;
    - Auto on length: 13.7695 mm at the defaults (hot-low 2.5000, 0.8806 at the input, red, with
      sentence 2); 11.9023 mm with the minimum temperature at 20 °C and no variation allowance
      (0.5664, within, no Problem);
    - a NaN rating: "Gearbox check unknown: a torque is not a number", Caution, and the unknown
      Problem;
    - short magnets: hot-low -0.4829, at the input -0.1701, the engine text "Within the gearbox
      input rating", greyed, no Problem;
    - the operating temperature below the minimum (sentence 1 only): op -40 °C, min 20 °C,
      variation 0, cold-high 2.8930 below hot-low 3.3246 and at the input 0.6091 against a
      ripple of 0.6999; variation 0.05, 0.6395 against 0.6999; op 0 °C, 0.6091 against 0.6386;
      variation 0.15, 0.7004 against 0.6999;
    - ratio and efficiency: 7.647 at ratio 1 and efficiency 0.5 (ripple 5.377); 3.823 at ratio
      1 and efficiency 1; 0.7647 at ratio 5 and efficiency 1; 1.529 at ratio 5 and efficiency
      0.5; 0.1912 at ratio 20 and efficiency 1, within;
    - each band lever alone at the defaults: no variation allowance 0.6509, the minimum at 20 °C
      0.7663, SmCo 2:17 (Recoma 26) on both rings 0.7581; none passes;
    - the second form: at ratio 1, at most 0.3406 N·m and the requirement alone 2.632 N·m; at
      efficiency 0.5, 0.8965 and 1.000; at ratio 1 with the requirement at 0.3406 N·m no
      sentence 2 (0.59984), at 0.3407 N·m sentence 2 (0.6001);
    - "meeting the requirement puts": twelve poles (hot-low 2.5415) reads "is 0.8952 N·m" beside
      "puts 0.8806 N·m"; at an axial length of 20 mm, 1.321 against the same 0.8806; a sweep of
      6,720 designs shows sentence 2 in 2,342, of which 982 meet the requirement and put more
      than the quoted figure and none puts less than R·B/(i·η); Auto on length at 60
      requirements (1.80-2.39 N·m): in 27, sentence 1 reads one last digit below sentence 2's
      figure (0.6340 against 0.6341 at 1.800 N·m);
    - equal readouts on the slider grids: axial length 8.01 mm, face gap 1.00 mm and rating
      0.54 N·m read "Cold-high at the gearbox input is 0.5400 N·m, above its 0.5400 N·m input
      rating." (0.5400404 at the input; the badge "0.55 N·m against 0.54 N·m"); also 8.01 / 1.42
      / 0.47 and 8.02 / 1.49 / 0.46; a design file with a rating of 0.80487 reads "0.8049 ...
      0.8049";
    - the band over geometry (4-16 magnets, 4-40 mm length, 0.5-4 mm gap, 4140, 304 or 6061 back
      iron): largest relative change 2.2e-16;
    - the caps of every proposed text, with the crate's own `word_count`, `sentence_count`,
      `plain_text_violations` and `mentions_gearbox_protection`;
  - round 2 (2026-10-17):
    - at every design probed, cold-high at the gearbox input equals cold-high at the output over
      ratio × efficiency, and the ripple the nominal pull-out over ratio × efficiency, exactly;
    - round 1's probes again, with the same numbers, now reading the new sentence 2: at the
      defaults "so even just meeting the 2.500 N·m requirement at the gearbox output puts 0.8806
      N·m on the gearbox input"; at ratio 1, 4.403 N·m on the gearbox input, at most 0.3406 N·m,
      the requirement alone 2.632 N·m; at ratio 1 with the requirement at 0.3407 N·m, 0.6001 N·m,
      at most 0.3406 N·m or a band of at most 1.672 times; Auto on length at the defaults reads
      0.8806 in both sentences;
    - twelve poles: "is 0.8952 N·m" beside "even just meeting ... puts 0.8806 N·m"; at 20 mm
      axial length, 1.459 against the same 0.8806;
    - a sweep of 15,120 designs (rating 0.05-5.0, ratio and efficiency including both slider
      ends, five temperature pairs including the operating temperature below and equal to the
      minimum, variation 0, 0.15 and 0.5, requirement 0.1-10 N·m, four geometries): sentence 2 in
      5,499, each with the minimum at or below the operating temperature and each naming the
      "requirement at the gearbox output"; in all 2,301 that meet the requirement, sentence 1
      reads above sentence 2's quoted load and cold-high at the input is at least R·B/(i·η);
    - Auto on length at 60 requirements: the same 27 one-last-digit differences (0.6340 against
      0.6341 at 1.800 N·m, unrounded 0.634017);
    - the band reads 1.6731 at every NdFeB design probed (the defaults, 8 magnets, 304 back iron,
      12 poles, 12 poles at 20 mm, the 20 mm ring radius, Auto at 13.77 mm);
    - the counters on the changed texts: the engine help 29 words in 2 sentences, the ripple help
      30 words in 2 sentences, the ripple label 39 characters, the defaults' Problem 3 sentences;
      the guard empty and no protection wording; the crate's gearbox, conflict, problems-quote,
      canonical-names, help, plain-words and sentence tests pass.
- Round 1: 3 must-fix items, text applied verbatim (constants first, tests through the
  constants):
  - `gearbox_conflict` (`gui/guidance.rs`), the final `format!`: Problem sentence 2.
    - Old: "No magnet size, gap or count passes both this and the hot minimum: cold-high is
      {band_text} times hot-low here, so meeting the {} requirement puts {} on the gearbox input.
      {remedy}"
    - New: "No magnet size, gap or count passes both this and the hot minimum: cold-high is
      {band_text} times hot-low here, so even just meeting the {} requirement at the gearbox
      output puts {} on the gearbox input. {remedy}"
    - Why: the requirement is a torque at the gearbox output (hot-low must reach it, and the
      coupling sits on the output), and it was quoted without its side beside two torques on the
      gearbox input; naming it once covers the remedy's "a requirement of at most ..." in the next
      sentence. "Meeting the requirement puts X" stated one load, but X is what a design that just
      meets the requirement puts there, and one that meets it with room puts more: twelve poles
      reads "is 0.8952 N·m" beside "puts 0.8806 N·m", as do 982 of the 2,342 sweep designs that
      show sentence 2. Not "puts at least": X is quoted rounded up (Task 3 review I-2), and under
      Auto sentence 1 reads one last digit below it in 27 of 60 requirements (0.6340 against
      0.6341 at 1.800 N·m), where "at least 0.6341" would read false. "Even just meeting" is true
      at every design and makes that difference read as rounding.
  - The `HELP` row of `model.gearbox_input_ripple_Nm` (`gui/plain_text.rs`).
    - Old: "Torque swing while slipping: the pull-out at the operating temperature, without the
      production allowance, divided by the ratio and efficiency. The gearbox check compares
      cold-high at the gearbox input instead."
    - New: "Slip swing: the pull-out at the gearbox output at operating temperature, without the
      production allowance, divided by ratio and efficiency. The gearbox check compares cold-high
      at the gearbox input instead."
    - Why: the pull-out it divides is a torque at the gearbox output (`gearbox_input_ripple_Nm =
      T_pull / (gear_ratio × gear_efficiency)`, `engine/model.rs`), named without its side. The
      new text holds at the defaults (2.688 / 4.75 = 0.5660), with the operating temperature
      below the minimum (3.3246 / 4.75 = 0.6999) and at ratio 1 with efficiency 0.5 (5.377). It
      adds no "amplitude": the row is the forward-drive half's T/(i·η), and with an efficiency
      below one the back-driven half at the input is smaller.
  - The help of `gearbox.cold_high_check` (`engine/gearbox.rs`, its `out_rust_only` row).
    - Old: "Compares the cold-high torque at the gearbox input, cold-high divided by the ratio and
      efficiency, with the gearbox input torque rating. A peak check only."
    - New: "Compares cold-high at the gearbox input (cold-high at the gearbox output divided by
      ratio and efficiency) with the gearbox input torque rating. Fatigue from repeated slips is
      not checked."
    - Why: the cold-high it divides is the torque at the gearbox output (`cold_high_input_Nm =
      cold_high / (gear_ratio * gear_eff)`, `engine/metal_design.rs`), named without its side;
      since BL-059 a bare "cold-high" has two sides on screen (the dashboard's "Cold-high torque"
      row at the output, this check at the input). "A peak check only." was the one place in
      these texts where "peak" stood beside the rating: it can read as a check against a peak
      rating, which owner decision 2A leaves open, and the spreadsheet's "GAM 5:1 peak input
      rating." under the same input (with Show spreadsheet details on) invites that reading. What
      it meant (decision 8A) is now said plainly, as the slipping note's watch-out says it.

  Should-fix offered, not applied (open for the controller):
  - S-1, Problem sentence 1 (`gearbox_problem`): where the two readouts are equal, both numbers
    get as many more decimals as tell them apart, the way `band_quotes` separates the band from
    the widest band near one ("Cold-high at the gearbox input is 0.54004 N·m, above its 0.54000
    N·m input rating."); every other case is unchanged. Why: the badge never shows two equal
    numbers, but the Problem can, within 5e-5 N·m above a rating (three designs on the slider
    grids, and a rating of 0.80487 from a design file). Nothing is false, it is rare, and the
    badge beside it separates the numbers.
  - S-2, `SMALLER_VARIATION` (`gui/guidance.rs`): "a smaller measured variation" -> "a smaller
    variation allowance backed by measurement". Why: the input's row reads "Production variation
    allowance" and the hot minimum's Problem says "the 15 % variation allowance", so "measured
    variation" names no row; "backed by measurement" keeps the point that the allowance is lowered
    only on evidence. Not false today.
  - S-3, the `HELP` row of `coupling.gear_ratio` (`gui/plain_text.rs`): "... such as the slip
    ripple." -> "... such as the slip swing." Why: the dashboard row now reads "Nominal slip swing
    at the gearbox input", so "the slip ripple" names a row the dashboard no longer shows. Not
    false (decision 7A allows leaving it); round 2 found the case slightly stronger, since the
    ripple help itself now opens with "Slip swing:".

  Recorded, no change asked:
  - strings 1-3, 5-9, 11 and 12 pass as drafted or as Task 2 reworded them (the Why's "Cold-high
    torque can reach" is true at every probe, including the operating temperature below the
    minimum, where "up to" would be false);
  - "a warmer minimum temperature" with the minimum at its slider's top, and a largest
    requirement below the requirement slider's 0.1 N·m bottom, are true statements of what would
    have to change (the minimum temperature is an environment fact, not a slider move);
  - "swing" read as peak to peak needs no change: the help's definition says what the number is;
  - no overlay label for the check row: an engineer finds "the gearbox check" on the dashboard
    (the rating's help and the row's label use the same words, the verdict starts "Within" or
    "Above the gearbox input rating", the guide's block is "Gearbox input torque", and the results
    table's search matches "gearbox"); the engine label also names the verdict's first number,
    and the dashboard wraps it instead of truncating it;
  - at a rating of 0.80 the badge quotes 0.81 beside the Problem's 0.8049: spec 4's rule against
    two equal numbers;
  - under Auto, sentence 1 (rounded to the nearest) and sentence 2 (rounded up) can differ in the
    last digit: numbers, not text (Task 3 review I-2, ruling B-6);
  - the context lines are fine: "One input pulls two ways (Axial magnet length, both rings):
    Hot-low torque needs it raised and Gearbox input torque needs it lowered, so change another
    input to settle one of them."; "Raising Magnet face gap also lowers the pull-out, so Hot-low
    torque falls further." (true in the gearbox block at the defaults, where the hot minimum
    fails); "Example design: fails 4 of 8 checks" while the clamp section is paused ("4 of 9" on
    the sheet, unpaused).
- Round 2: PASS, no must-fix. It re-reviewed the changed strings and every string whose meaning
  depends on them; round 1's findings on the unchanged strings stand. The three changed strings
  pass: the requirement is named at the gearbox output and each load on the gearbox input, "even
  just meeting" holds at every design (none that meets the requirement puts less than R·B/(i·η),
  over 2,301 designs), the engine help names both sides and nothing calls the rating a peak or a
  nominal rating, and the ripple help names the pull-out at the gearbox output and claims no
  ordering with cold-high (the swing is above cold-high at the input in three of the four designs
  with the operating temperature below the minimum). No new should-fix; round 1's S-1 to S-3
  stand as offered.

  Recorded, no change asked: under Auto the design just meets the requirement, and sentence 1 can
  read one last digit below sentence 2's quoted load (27 of 60 sampled requirements; 0.6340
  against 0.6341, unrounded 0.634017). It is the deliberate rounding of Task 3 review I-2 (the
  load is rounded up, so that a gearbox rated for the quoted torque takes it), it changes numbers,
  not text, and "even just meeting" makes it no worse than "meeting"; no design that meets the
  requirement with room reads below the load (2,301 of 2,301).
- What happened to the should-fix items: S-1, S-2 and S-3 were left open, not applied. Each would
  change reviewed on-screen text, and the brief applies a should-fix only when the controller asks
  for it.
- Tests that followed the changes (each new assertion was seen failing on the old text first):
  - `gui::guidance` `assert_gearbox_problem_consistent`: wherever sentence 2 exists it names the
    "requirement at the gearbox output", and wherever hot-low is at or above the requirement,
    cold-high at the gearbox input is at least what just meeting the requirement puts there
    (requirement × band / (ratio × efficiency), less 1e-12 of float noise) (the first item);
  - `at_the_defaults_the_gearbox_problem_says_no_geometry_passes_both_checks`: the defaults' sentence
    2 contains "so even just meeting the 2.500 N·m requirement at the gearbox output puts 0.8806
    N·m on the gearbox input." (the first item);
  - `the_conflict_sentence_shows_exactly_when_no_geometry_passes_both_checks`: twelve poles shows
    sentence 2 while meeting the requirement with room, so the bound is checked where it is not an
    equality (the first item);
  - `gui::plain_text` `the_gearbox_texts_claim_what_the_engine_does`: the ripple help names the
    "pull-out at the gearbox output" (the second item); the check row's help, the engine's (the
    overlay has no row for it), names "gearbox output", "gearbox input" and "fatigue from repeated
    slips is not checked", and joins the ban on "no check", "peak" and "warm" (the third item).

  No test held an old text as a literal: every other test reads the strings through
  `gearbox_conflict`, the `HELP` table and the engine metadata. The doc lines that describe the
  changed texts follow them: `magcoupling-rs/README.md` (the `guidance.rs` and `plain_text.rs`
  test rows) and `docs/ai/02-system.yaml` (the ripple help).
- Traps: none of the plan's is written in any string: each torque names its side of the
  gearbox; efficiency raises the input torque; cold-high is an expectation; nothing says the
  coupling protects the gearbox; "peak" names the compared torque only; the ripple row is the
  nominal swing at the operating temperature; geometry does not narrow the band; no "hot-day";
  no path, id, "BL-059", "Rust-only" or "clamp"; glyph-safe. Round 1 found the first trap three
  times (the requirement in sentence 2, the pull-out in the ripple help and the divided cold-high
  in the engine help were named without their side of the gearbox) and "peak" beside the rating
  once (the engine help's "A peak check only."); its three must-fix items fixed them, and round 2
  found none.
- Verdict: PASS (round 2).

### Report for the record, round 1 (as returned)

**1.** Reviewer: physics reviewer (session model), fresh context, 2026-10-17; round 1.

**2. Read.**
- The brief (task-8-inreview-brief-r1.md), rulings.md (B-1 to B-10) and the plan's head up to its Tasks heading: Global Constraints, Review Focus, R-1 to R-13, plan review 1A-8A, PD-1 to PD-4. Spec sections 1-6.
- Code:
  - engine/gearbox.rs (whole);
  - engine/api.rs `gearbox::compute`;
  - engine/metal_design.rs `cold_high_input_Nm` (:509);
  - engine/model.rs `gearbox_input_ripple_Nm` and `gearbox_reference_Nm` (:349-352, :1156-1157);
  - gui/guidance.rs: GEARBOX_GUIDE, gearbox_problem, gearbox_conflict, band_levers, quoted_limit, quoted_load, band_quotes, level_of, fixes, helps, trade_offs, side_effects, and the sentence-2 test helpers (assert_conflict_quotes, assert_gearbox_problem_consistent, the B-6 pins, the lever-list test);
  - gui/dashboard.rs: DASHBOARD, END_EFFECT_ROWS, verdict_level, check_level, greyed_by_end_effect, and the row label `.wrap()` at :716;
  - gui/plain_text.rs: LABELS, the rating, ripple, gear ratio and efficiency HELP rows, display_label, the_gearbox_texts_claim_what_the_engine_does, and the protect ban;
  - gui/results_table.rs (the table draws the catalogue label, truncated);
  - gui/input_ui.rs (sliders round to their step);
  - engine/explain/plain.rs (the guard);
  - engine/explain/notes.rs: GEARBOX_CHECK_SOURCE and the two gearbox watch-outs, for consistency only.
- Sheets: BL059-guidance.txt, BL059-overlay.txt and BL059-strings.diff.

Probes: a scratch crate (bl059g3) in my scratchpad with its own CARGO_TARGET_DIR, depending on magcoupling-rs by path with `gui`. It calls `compute_all` and the gearbox guide's `(g.problem)`. Nothing in the repo was changed.
- (1) Defaults:
  - cold-high at the input is 0.804907 against 0.6;
  - the verdict reads "Above the gearbox input rating: 0.80 N·m against 0.60 N·m";
  - the Problem quotes 0.8049, 0.6000, 1.673, 2.500, 0.8806 (0.880564), 1.703 (1.703454) and 1.140, with sentence 2.
- (2) The probe designs, cold-high at the input:
  - default: 0.8049, above;
  - 8 magnets: 0.5893, within;
  - 304 back iron: 0.5082, within;
  - Y30: 0.07117, within (band 1.9207);
  - 20 mm ring radius: 0.7602, above, with sentence 2.
- (3) The rating:
  - at 0.80: "Above the gearbox input rating: 0.81 N·m against 0.80 N·m". Sentence 2 shows (0.8806 > 0.80; at most 2.271 N·m, band 1.520). The Problem reads "0.8049 N·m, above its 0.8000 N·m".
  - at 0.81: within.
  - at 0.05: "0.80 N·m against 0.05 N·m", with sentence 2's second form (at most 0.1419 N·m; the requirement alone 0.5264).
  - at 5.0: within.
- (4) The passing design is within, with no Problem. The gearbox-only design reads 0.7070, above; R·B/(i·η) = 0.5636, so no sentence 2.
- (5) Auto on length:
  - at the defaults it solves 13.7695 mm: hot-low 2.5000, 0.8806 at the input, red, with sentence 2;
  - with the minimum temperature at 20 °C and variation 0 it solves 11.9023 mm: 0.5664, within, no Problem.
- (6) A NaN rating gives "Gearbox check unknown: a torque is not a number", Caution, and Problem 5.
- (7) Short magnets:
  - hot-low is -0.4829 and cold-high at the input -0.1701;
  - the engine text reads "Within the gearbox input rating";
  - check_level is None (greyed), and there is no Problem.
- (8) The operating temperature below the minimum (sentence 1 only in every case):
  - op -40 °C, min 20 °C, v 0: cold-high 2.8930 below hot-low 3.3246, and at the input 0.6091 against a ripple of 0.6999;
  - v 0.05: 0.6395 against 0.6999;
  - op 0 °C, min 20 °C, v 0: 0.6091 against 0.6386;
  - op -40 °C, min 20 °C, v 0.15: 0.7004 against 0.6999.
- (9) Ratio and efficiency (efficiency raises the input torque):
  - ratio 1, efficiency 0.5: 7.647, ripple 5.377, second form;
  - ratio 1, efficiency 1: 3.823;
  - ratio 5, efficiency 1: 0.7647;
  - ratio 5, efficiency 0.5: 1.529;
  - ratio 20, efficiency 1: 0.1912, within.
- (10) Each lever alone at the defaults: variation 0 gives 0.6509 (and the lever list drops the variation), minimum 20 °C gives 0.7663, and SmCo 2:17 (Recoma 26) on both rings gives 0.7581. None passes.
- (11) The second form:
  - ratio 1 gives "...a requirement of at most 0.3406 N·m: the requirement alone puts 2.632 N·m on the gearbox input...";
  - efficiency 0.5 gives 0.8965 and 1.000;
  - ratio 1 with the requirement at 0.3406 shows no sentence 2 (0.59984); at 0.3407 sentence 2 shows (0.6001).
- (12) twelve_poles (`with_poles(12)`):
  - hot-low is 2.5415, so the design meets the 2.5 requirement with room ("Estimate covers hot min");
  - sentence 1 reads "is 0.8952 N·m", but sentence 2 says "meeting the 2.500 N·m requirement puts 0.8806 N·m";
  - axial length 20 mm: 1.321 against the same 0.8806.
  - A sweep of 6,720 designs (rating 0.05-5, ratio 1-20, efficiency 0.5-1, five temperature pairs including op < min, variation 0-0.5, requirement 0.1-10) shows sentence 2 in 2,342.
    - In 982 of those, the design meets the requirement and puts more than the figure quoted.
    - No design meeting the requirement puts less than R·B/(i·η).
  - Auto on length at 60 requirements (1.80-2.39 N·m) showing sentence 2: in 27, sentence 1 reads one last digit below sentence 2's figure (0.6340 against 0.6341 at 1.800 N·m). Sentence 1 rounds to the nearest; sentence 2 rounds up (Task 3 I-2).
- (a) Equal reading on the slider grids:
  - axial length 8.01 mm, face gap 1.00 mm, rating 0.54 N·m: cold-high at the input is 0.5400404, and the Problem reads "Cold-high at the gearbox input is 0.5400 N·m, above its 0.5400 N·m input rating." The badge beside it reads "0.55 N·m against 0.54 N·m";
  - also 8.01 / 1.42 / 0.47 and 8.02 / 1.49 / 0.46: the first three found scanning 8.00-12.00 mm by 0.90-1.50 mm;
  - a design file with a rating of 0.80487 at the defaults reads "0.8049 ... 0.8049".
- The band does not depend on geometry: over 4-16 magnets, 4-40 mm length, 0.5-4 mm gap and 4140, 304 or 6061 back iron, its largest relative change is 2.2e-16.
- Caps of every proposed text were checked with the crate's own word_count, sentence_count, plain_text_violations and mentions_gearbox_protection.

**3. Strings.**

| # | String | Verdict | Reason |
|---|---|---|---|
| 1 | Guide label "Gearbox input torque" | PASS | Distinct from the other guides' labels. It reads well in the trade-off line ("Gearbox input torque needs it lowered"). |
| 2 | Why | PASS | One sentence of 28 words. "Can reach" is true at every probe, including (8), where cold-high is below hot-low and "up to" would be false. Both sides are named. It makes no protection claim. |
| 3 | Problem sentence 1 | PASS (S-1 offered) | Live numbers, and true at every probe. Two equal readouts are possible in a window of 5e-5 N·m above the rating (routed point (a)); see S-1. |
| 4 | Problem sentence 2 (the conflict) | MUST-FIX (M-1) | The requirement, an output-side torque, is quoted without its side. "Meeting the requirement puts X" is contradicted by sentence 1 wherever the design meets the requirement with room (twelve_poles; 982 of 2,342 sweep designs). |
| 5 | Problem when the check cannot be made | PASS | True for a NaN rating (6); both values named with their side. |
| 6 | "Within the gearbox input rating" | PASS | Equality passes. Where the engine reads Within on a negative torque (7), the row greys with no badge and no Problem. |
| 7 | "Above the gearbox input rating" + ": {q} N·m against {r} N·m" | PASS | True at (1), (3) and (9). Never two equal numbers (0.81 against 0.80). The rule that inflates q is spec 4's, so it is recorded, not changed. |
| 8 | "Gearbox check unknown: a torque is not a number" | PASS | Unreachable from the UI (R-12). "A torque" quotes no value, so no side is needed. |
| 9 | Engine label "Cold-high torque at the gearbox input against its rating" | PASS | Names the quantity the verdict's first number is. See (13). |
| 10 | Engine help | MUST-FIX (M-3) | The cold-high it divides is named without its side. "A peak check only." can be read as a check against a peak rating (2A). |
| 11 | Rating help | PASS | True; no peak or nominal. The spreadsheet note shown under Show spreadsheet details reads as the spreadsheet's own labelled note (R-11). |
| 12 | Ripple label "Nominal slip swing at the gearbox input" | PASS | 39 characters, one quantity, side named, no "warm". |
| 13 | Ripple help | MUST-FIX (M-2) | The pull-out it divides, an output-side torque, is named without its side (routed point (d)). |

Routed points:
- (a): SHOULD-FIX S-1. Every number is the correct readout and "above" is true, so nothing is false; the case is rare, and the badge beside it shows two different numbers.
- (b), first point (the requirement's side): MUST-FIX, in M-1. Second point (the variation's name): SHOULD-FIX S-2. Third point ("puts"): probe (12), in M-1.
- (c): PASS, recorded.
  - "A warmer minimum temperature" with the minimum at its slider's top (20 °C), below the operating temperature, is true. Sentence 2 names what would have to change, and the minimum temperature is an environment fact (spec 5), not a slider move.
  - A largest requirement below the requirement slider's 0.1 N·m bottom is likewise a true statement of what the application would need.
- (d), the side: MUST-FIX, in M-2.
  - "Swing" read as peak to peak needs no change: the help's definition (the pull-out divided by ratio and efficiency) says what the number is.
  - "Amplitude" would be less exact. The row is the forward-drive half's T/(i·η), and with efficiency below one the back-driven half at the input is smaller.
- (e): SHOULD-FIX S-3. Decision 7A allows leaving it, since the help is not false.
- (f) and probe (13): no overlay label is asked for. An engineer finds "the gearbox check" on the dashboard:
  - The rating help says it "compares the cold-high torque at the gearbox input with it", and the row reads "Cold-high torque at the gearbox input against its rating": the same words.
  - Its verdict starts with "Within" or "Above the gearbox input rating", and the guide block is "Gearbox input torque".
  - The results table's search matches "gearbox".
  - The engine label also names the verdict's first number, which a "Gearbox check ..." overlay label would drop. On the dashboard the label wraps instead of truncating.

**4. MUST-FIX.**

**M-1.** Problem sentence 2 (string 4), `gui/guidance.rs` `gearbox_conflict`: the final `format!`.
- Old: `"No magnet size, gap or count passes both this and the hot minimum: cold-high is {band_text} times hot-low here, so meeting the {} requirement puts {} on the gearbox input. {remedy}"`
- New: `"No magnet size, gap or count passes both this and the hot minimum: cold-high is {band_text} times hot-low here, so even just meeting the {} requirement at the gearbox output puts {} on the gearbox input. {remedy}"`
- At the defaults it reads: "No magnet size, gap or count passes both this and the hot minimum: cold-high is 1.673 times hot-low here, so even just meeting the 2.500 N·m requirement at the gearbox output puts 0.8806 N·m on the gearbox input. It takes a gearbox rated for that, a requirement of at most 1.703 N·m, or a band of at most 1.140 times (a warmer minimum temperature, a smaller measured variation or SmCo magnets narrow it)."
- At ratio 1: "... so even just meeting the 2.500 N·m requirement at the gearbox output puts 4.403 N·m on the gearbox input. It takes a gearbox rated for that or a requirement of at most 0.3406 N·m: the requirement alone puts 2.632 N·m on the gearbox input, so no narrower band settles it."
- Still two sentences, as before. The numbers, their order and the remedy forms are unchanged, so the B-6 pins and `assert_conflict_quotes` hold.
- Why:
  - (i) Trap: the requirement is a torque at the gearbox output. It is the coupling's torque, hot-low must reach it, and the coupling sits on the output. The sentence quotes it without its side, right beside two input-side torques ("2.500 N·m requirement ... 0.8806 N·m on the gearbox input"). Naming the side once covers the remedy's "a requirement of at most 1.703 N·m", the same quantity, in the next sentence.
  - (ii) Probe (12): "meeting the requirement puts X" states one load, but X is the load of a design that just meets the requirement. One that meets it with room puts more.
    - twelve_poles, a test design (hot-low 2.5415 against 2.5), reads "Cold-high at the gearbox input is 0.8952 N·m ... meeting the 2.500 N·m requirement puts 0.8806 N·m on the gearbox input".
    - The sweep finds 982 such designs among 2,342.
  - Why not "puts at least": X is quoted rounded up (quoted_load, Task 3 I-2, so that a gearbox rated for X takes it). An Auto-sized design meets the requirement exactly, yet sentence 1 reads one last digit below X in 27 of 60 sampled requirements (0.6340 against 0.6341 at 1.800 N·m). "At least 0.6341" would read false beside "is 0.6340".
  - "Even just meeting" is true at every design, says that the quoted torque is the least a compliant design puts on the input, and makes that last-digit difference read as rounding.
- Test assertion:
  - in `at_the_defaults_the_gearbox_problem_says_no_geometry_passes_both_checks`: `conflict.contains("so even just meeting the 2.500 N·m requirement at the gearbox output puts 0.8806 N·m on the gearbox input.")`;
  - in `assert_gearbox_problem_consistent`, wherever sentence 2 shows: it contains `"requirement at the gearbox output"`. Where `metal.torque_hot_low_Nm >= required`, `metal.cold_high_input_Nm >= required * band / (ratio * efficiency) * (1.0 - 1e-12)`: a design meeting the requirement puts at least what just meeting it puts;
  - `with_poles(12)` is among the designs it runs on.

**M-2.** The ripple help (string 13), `gui/plain_text.rs` HELP row `"model.gearbox_input_ripple_Nm"`.
- Old: "Torque swing while slipping: the pull-out at the operating temperature, without the production allowance, divided by the ratio and efficiency. The gearbox check compares cold-high at the gearbox input instead."
- New: "Slip swing: the pull-out at the gearbox output at operating temperature, without the production allowance, divided by ratio and efficiency. The gearbox check compares cold-high at the gearbox input instead."
- 30 words in 2 sentences, with no braces and no "guarantee", "peak" or "warm". It keeps the four phrases `the_gearbox_texts_claim_what_the_engine_does` requires. It claims no inequality with cold-high (PD-2).
- Why:
  - Trap: the pull-out it divides is a torque at the gearbox output, named without its side. The engine computes `gearbox_input_ripple_Nm = T_pull / (gear_ratio × gear_efficiency)` (model.rs:1156), from the coupling's pull-out on the output.
  - The new text holds at every probe: (1) 2.688 / 4.75 = 0.5660; (8) 3.3246 / 4.75 = 0.6999, with the operating temperature below the minimum; (9) 2.688 / 0.5 = 5.377.
  - It does not add "amplitude", for the reason under routed point (d).
- Test assertion: in `the_gearbox_texts_claim_what_the_engine_does`, the RIPPLE `names` gain `"pull-out at the gearbox output"`.

**M-3.** The engine help (string 10), `engine/gearbox.rs`: the `out_rust_only` help of `cold_high_check`.
- Old: "Compares the cold-high torque at the gearbox input, cold-high divided by the ratio and efficiency, with the gearbox input torque rating. A peak check only."
- New: "Compares cold-high at the gearbox input (cold-high at the gearbox output divided by ratio and efficiency) with the gearbox input torque rating. Fatigue from repeated slips is not checked."
- 29 words in 2 sentences; the plain-words guard is empty.
- Why:
  - (i) Trap: the cold-high it divides is the output-side torque (`cold_high_input_Nm = cold_high / (gear_ratio * gear_eff)`, metal_design.rs:509), named without its side. Since BL-059, bare "cold-high" has two sides on screen: the dashboard's "Cold-high torque" row at the output, and this check at the input.
  - (ii) "A peak check only." is the only place in these texts where "peak" stands beside the rating. An engineer can read "a peak check" as a check against a peak rating, which decision 2A leaves open. The spreadsheet note "GAM 5:1 peak input rating." shows under the same input with spreadsheet details on, which invites that reading.
  - What the sentence means (decision 8A) is said plainly instead: fatigue from repeated slips is not checked, as the slipping note's watch-out also says.
- Test assertion: `help_for("gearbox.cold_high_check")` (the engine help, since the overlay has no row for it) is added to the loop in `the_gearbox_texts_claim_what_the_engine_does` that bans "no check", "peak" and "warm". It is also checked by `names` with `"gearbox output"`, `"gearbox input"` and `"fatigue from repeated slips is not checked"`.

**5. SHOULD-FIX.**

**S-1.** Routed point (a): Problem sentence 1 (string 3), `gui/guidance.rs` `gearbox_problem`.
- Old: `"Cold-high at the gearbox input is {}, above its {} input rating."`, from `result_text(cold_high_input)` and `input_text(rating)`. It reads "Cold-high at the gearbox input is 0.5400 N·m, above its 0.5400 N·m input rating." at an axial length of 8.01 mm, face gap 1.00 mm and rating 0.54 N·m.
- New: the same words. Where the two readouts are equal, both numbers get as many more decimals as tell them apart, the way band_quotes separates the band from the widest band near one. The example then reads: "Cold-high at the gearbox input is 0.54004 N·m, above its 0.54000 N·m input rating."
  - Every other case is unchanged: "0.8049 N·m, above its 0.6000 N·m" at the defaults, so the B-6 pins hold.
- Why: the badge beside it never shows two equal numbers (spec 4, Review Focus 1), but the Problem can. It happens within 5e-5 N·m above a rating: on the slider grids (three designs found) and from a design file (0.80487 at the defaults). It is a should, not a must: nothing is false, it is rare, and the badge separates the two numbers.
- Test assertion:
  - with the defaults and the rating at cold-high at the gearbox input minus 1e-6 N·m, sentence 1's two numbers parse to different values, the first larger;
  - at the defaults, sentence 1 still contains "0.8049 N·m, above its 0.6000 N·m".

**S-2.** Routed point (b), second point: `gui/guidance.rs` `SMALLER_VARIATION`.
- Old: "a smaller measured variation"
- New: "a smaller variation allowance backed by measurement"
- It reads at the defaults: "(a warmer minimum temperature, a smaller variation allowance backed by measurement or SmCo magnets narrow it)". Last in the list: "... or a smaller variation allowance backed by measurement narrows it".
- Why:
  - The input's row reads "Production variation allowance", and the hot minimum's Problem one block up says "the 15 % variation allowance". "Measured variation" names no row.
  - "Backed by measurement" keeps the point that the allowance is an assumption, lowered only on evidence (spec 5: not a fix).
  - Not false today.
- Test assertion: the lever-list test reads the constant, so it follows. Add the defaults' parenthetical above as a literal.

**S-3.** Routed point (e): `gui/plain_text.rs` HELP row `"coupling.gear_ratio"`.
- Old: "Reduction ratio of the gearbox; the coupling sits on its wheel side. A higher ratio lowers the coupling torques referred to the gearbox input, such as the slip ripple."
- New: "Reduction ratio of the gearbox; the coupling sits on its wheel side. A higher ratio lowers the coupling torques referred to the gearbox input, such as the slip swing."
- 29 words in 2 sentences.
- Why: LABELS gives each quantity one name on screen, and since BL-059 the dashboard row reads "Nominal slip swing at the gearbox input". "The slip ripple" now names a row the dashboard no longer shows. The results table and exports keep the catalogue label "Estimated gearbox input torque ripple amplitude", and the slipping note says "torque ripple"; neither is in scope here. Decision 7A allows leaving the help, since it is not false; the change is one word.
- Test assertion: `help_for("coupling.gear_ratio").contains("the slip swing")`.

**6. Traps checked.**
- Side of the gearbox: every torque has its side after M-1, M-2 and M-3. String 8's "a torque" quotes no value.
- Efficiency: no text says it reduces the load. "Divided by ... efficiency" raises the input torque: 7.647 at efficiency 0.5 against 3.823 at 1, both at ratio 1.
- Cold-high as an expectation: the Why says "can reach". "At most" bounds only the requirement and the band, never cold-high. Sentence 1 states the model's value, not what every unit reaches.
- Protect: mentions_gearbox_protection is false for all 13 strings and every new text.
- Peak: no text names the rating a peak or a nominal rating. M-3 removes the ambiguous "A peak check only.". The spreadsheet's "GAM 5:1 peak input rating." is shown as the spreadsheet's labelled note (R-11).
- Ripple row: it is the nominal swing at the gearbox input, with no "warm". The help says "at operating temperature", which holds with the operating temperature below the minimum (probe (8)).
- Band and geometry: no text says geometry narrows the band. Sentence 2 says no size, gap or count passes both checks, and the band is the same over geometry to 2.2e-16.
- Forbidden words: no "hot-day", internal path, id, cell, "BL-059", "Rust-only" or "clamp" in strings 1-13 or the new texts.
- Glyphs: no glyph beyond the "N·m" already drawn.
- plain_text_violations is empty for every string and every new text.
- Caps:
  - Why: 28 words, one sentence, no ". " inside;
  - M-2: 30 words, 2 sentences;
  - M-3: 29 words, 2 sentences;
  - S-3: 29 words;
  - overlay label: 39 characters;
  - each Problem sentence keeps its sentence count.
- Typed numbers: none in strings 1-13 or in the new texts. Every number is formatted from the results or inputs, and the verdict's 0.01 is a formatting rule. The "5:1" in the spreadsheet note is Python-locked and not ours.

**7. Context lines.**
- Trade-off line: fine. "One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Gearbox input torque needs it lowered, so change another input to settle one of them."
- Face-gap side effect: true in the gearbox block at the defaults, where the hot minimum fails. "Raising Magnet face gap also lowers the pull-out, so Hot-low torque falls further."
- Status line: "Example design: fails 4 of 8 checks" while the clamp section is paused, and "4 of 9" unpaused on the sheet. Fine.
- Recorded, no action:
  - At a rating of 0.80, the badge quotes 0.81 beside the Problem's 0.8049. This is spec 4's rule against two equal numbers.
  - Under Auto, sentence 1 (nearest) and sentence 2 (rounded up) can differ in the last digit. These are numbers, not text (Task 3 I-2, B-6), and M-1's wording makes the difference read as rounding.

**8. Verdict:** MUST_FIX (M-1, M-2, M-3; S-1 to S-3 offered).

### Report for the record, round 2 (as returned)

**1.** Reviewer: physics reviewer (session model), fresh context, 2026-10-17; round 2.

**2. Read.**
- The round-2 brief (task-8-inreview-brief-r2.md), round 1's answer (t8-g3-r1.md), rulings.md (B-1 to B-10), the plan's head up to its Tasks heading (Global Constraints, Review Focus, R-1 to R-13, plan review 1A-8A, PD-1 to PD-4) and spec sections 1-6.
- The round's source diff (`git diff` of the working tree: the three strings, the tests that pin them and the three doc lines) and the round-2 sheets: BL059-guidance.txt, BL059-overlay.txt and BL059-strings.diff.
- Code behind the changed strings and their dependents:
  - engine/gearbox.rs (whole);
  - engine/metal_design.rs `cold_high_input_Nm` (:509);
  - engine/model.rs `gearbox_input_ripple_Nm` and `gearbox_reference_Nm` (:349-352, :1156-1157);
  - gui/guidance.rs: gearbox_problem, gearbox_conflict, band_levers, quoted_value, quoted_limit, quoted_load, band_quotes, and the round's test changes (assert_gearbox_problem_consistent, the defaults' pin, the twelve_poles assertion);
  - gui/plain_text.rs: the LABELS and HELP rows of the ripple, the rating, the gear ratio, the efficiency and the requirement; label_for, help_for; the_gearbox_texts_claim_what_the_engine_does;
  - gui/dashboard.rs: DASHBOARD, END_EFFECT_ROWS, and result_tooltip, which draws the engine help under the engine label on the gearbox row's hover (the overlay has no row for it);
  - engine/explain/notes.rs: GEARBOX_CHECK_SOURCE and the two gearbox watch-outs, for consistency only.

Probes: a new scratch crate (bl059g3) in my own scratchpad folder with its own CARGO_TARGET_DIR there, depending on magcoupling-rs by path with `gui`. It calls `compute_all`, the gearbox guide's `(g.problem)`, `guidance::level`, `check_level` and `greyed_by_end_effect`, and runs the crate's own `word_count`, `sentence_count`, `plain_text_violations` and `mentions_gearbox_protection` on every changed text. Nothing in the repo was changed (`git status --short` is the same before and after).
- At every design below, cold-high at the gearbox input equals cold-high at the output over ratio x efficiency, and the ripple equals the nominal pull-out over ratio x efficiency, both exactly (difference 0).
- (1) Defaults:
  - cold-high at the input is 0.804907 against 0.6;
  - the verdict reads "Above the gearbox input rating: 0.80 N·m against 0.60 N·m";
  - the Problem reads: "Cold-high at the gearbox input is 0.8049 N·m, above its 0.6000 N·m input rating. No magnet size, gap or count passes both this and the hot minimum: cold-high is 1.673 times hot-low here, so even just meeting the 2.500 N·m requirement at the gearbox output puts 0.8806 N·m on the gearbox input. It takes a gearbox rated for that, a requirement of at most 1.703 N·m, or a band of at most 1.140 times (a warmer minimum temperature, a smaller measured variation or SmCo magnets narrow it)." (R·B/(i·η) = 0.880564.)
- (2) The probe designs, cold-high at the input:
  - default: 0.804907, above, with sentence 2;
  - 8 magnets: 0.589311, within;
  - 304 back iron: 0.508178, within;
  - Y30: 0.071172, within (band 1.9207);
  - 20 mm ring radius: 0.760233, above, with the defaults' sentence 2 (0.8806).
- (3) The rating:
  - at 0.80: "Above the gearbox input rating: 0.81 N·m against 0.80 N·m". Sentence 2 shows (0.8806 > 0.80; at most 2.271 N·m, band 1.520);
  - at 0.81: within;
  - at 0.05: "0.80 N·m against 0.05 N·m"; the second form, "a requirement of at most 0.1419 N·m: the requirement alone puts 0.5264 N·m on the gearbox input";
  - at 5.0: within.
- (4) The passing design: 0.706991, within, no Problem. The gearbox-only design: "Cold-high at the gearbox input is 0.7070 N·m, above its 0.6000 N·m input rating." only; R·B/(i·η) = 0.563561, so no sentence 2.
- (5) Auto on length:
  - at the defaults it solves 13.7695 mm: hot-low 2.5000, 0.880564 at the input, red; sentence 1 reads "is 0.8806 N·m" and sentence 2 "even just meeting ... puts 0.8806 N·m": the same number, as it should be;
  - with the minimum at 20 °C and variation 0 it solves 11.9023 mm: 0.566360, within, no Problem.
- (6) A NaN rating: "Gearbox check unknown: a torque is not a number", Caution, and Problem 5.
- (7) Short magnets: hot-low -0.4829, at the input -0.170091, the engine text "Within the gearbox input rating", greyed, check_level None, no Problem.
- (8) The operating temperature below the minimum (sentence 1 only in every case):
  - op -40 °C, min 20 °C, v 0: cold-high 2.8930 below hot-low 3.3246; at the input 0.609058 against a ripple of 0.6999 (= 3.3246 / 4.75);
  - v 0.05: 0.639511 against 0.6999;
  - op 0 °C, min 20 °C, v 0: 0.609058 against 0.6386 (= 3.0336 / 4.75);
  - op -40 °C, min 20 °C, v 0.15: 0.700417 against 0.6999.
- (9) Ratio and efficiency (dividing by the efficiency raises the input torque):
  - ratio 1, efficiency 0.5: 7.646621, ripple 5.3770;
  - ratio 1, efficiency 1: 3.823310, ripple 2.6885;
  - ratio 5, efficiency 0.5: 1.529324, ripple 1.0754;
  - ratio 20, efficiency 1: 0.191166, within.
- (10) Each lever alone at the defaults, R·B/(i·η): variation 0 gives 0.6509 (the lever list drops the variation); minimum 20 °C gives 0.7663; Recoma 26 on both rings (parts cleared) gives 0.7581. None reaches 0.6.
- (11) The second form:
  - ratio 1: "... so even just meeting the 2.500 N·m requirement at the gearbox output puts 4.403 N·m on the gearbox input. It takes a gearbox rated for that or a requirement of at most 0.3406 N·m: the requirement alone puts 2.632 N·m on the gearbox input, so no narrower band settles it.";
  - efficiency 0.5 (ratio 5): at most 0.8965 N·m, the requirement alone 1.000 N·m;
  - ratio 1 with the requirement at 0.3406: no sentence 2 (0.599840); at 0.3407 sentence 2 shows (0.600016): "even just meeting the 0.3407 N·m requirement at the gearbox output puts 0.6001 N·m on the gearbox input. It takes a gearbox rated for that, a requirement of at most 0.3406 N·m, or a band of at most 1.672 times (...)".
- (12) twelve_poles and "even just meeting":
  - twelve_poles: hot-low 2.5415 meets the 2.5 requirement with room; sentence 1 reads "is 0.8952 N·m" and sentence 2 "even just meeting ... puts 0.8806 N·m": the two now agree (a design with room puts more). At 20 mm axial length: 1.459 against the same 0.8806.
  - A sweep of 15,120 designs: rating 0.05-5.0, ratio and efficiency including both slider ends, five temperature pairs (including op below min and op equal to min), variation 0, 0.15 and 0.5, requirement 0.1-10 N·m, four geometries.
    - Sentence 2 shows in 5,499, each with the minimum at or below the operating temperature and each containing "requirement at the gearbox output".
    - 2,301 of those meet the requirement. In every one, sentence 1's readout is above sentence 2's quoted load, and cold-high at the input is at least R·B/(i·η). There are no exceptions.
  - Auto on length at 60 requirements (1.80-2.39 N·m), each with sentence 2: in 27 sentence 1 reads one last digit below sentence 2. At 1.800 N·m: "is 0.6340 N·m" against "even just meeting the 1.800 N·m requirement at the gearbox output puts 0.6341 N·m", where hot-low is 1.80000000003 and R·B/(i·η) is 0.634017. Sentence 1 rounds to the nearest; sentence 2 rounds up (Task 3 I-2). See section 7.
- Texts, by the crate's own counters, with the plain-words guard empty and no protection wording in each:
  - string 10: 29 words, 2 sentences;
  - string 13: 30 words, 2 sentences (at the cap);
  - string 12: 39 characters;
  - the defaults' Problem: 3 sentences, and sentence 2's first sentence is still one sentence (38 words).
- The crate's tests, run with `--locked` against my scratch target directory, all pass:
  - the 25 gearbox tests;
  - the conflict, problems-quote, canonical-names, help, plain-words ("word") and sentence groups.

**3. Strings.** Strings 4, 10 and 13 are re-judged as changed in this round. The other strings are judged beside them, for how they read; round 1's findings on them stand.

| # | String | Verdict | Reason |
|---|---|---|---|
| 1 | Guide label "Gearbox input torque" | PASS (round 1) | Unchanged. |
| 2 | Why | PASS (round 1) | Unchanged. It agrees with the new texts: "can reach" for cold-high, and each torque's side is named. |
| 3 | Problem sentence 1 | PASS | Unchanged. Beside the new sentence 2 it now agrees wherever a design meets the requirement with room: twelve_poles 0.8952 against "even just meeting ... 0.8806"; 2,301 sweep designs. The one-digit difference under Auto is recorded in section 7. Round 1's S-1 is still open. |
| 4 | Problem sentence 2 (the conflict) | PASS | M-1 applied verbatim. The requirement is named at the gearbox output and the load on the gearbox input. "Even just meeting" holds at every design: none that meets the requirement puts less than R·B/(i·η), over 2,301 designs. The remedy's "a requirement of at most {X}" is the same output-side quantity, named in the clause before. "It takes a gearbox rated for that" still refers to the torque on the gearbox input. Each sentence is still one sentence. |
| 5 | Problem when the check cannot be made | PASS (round 1) | Unchanged. |
| 6 | "Within the gearbox input rating" | PASS (round 1) | Unchanged. |
| 7 | "Above the gearbox input rating" + ": {q} N·m against {r} N·m" | PASS (round 1) | Unchanged. Re-probed at (1), (3) and (9). |
| 8 | "Gearbox check unknown: a torque is not a number" | PASS (round 1) | Unchanged. |
| 9 | Engine label "Cold-high torque at the gearbox input against its rating" | PASS | Unchanged. On the row's hover it sits above the new help, and the two say the same thing: "its rating" is "the gearbox input torque rating". |
| 10 | Engine help | PASS | M-3 applied verbatim. Both sides are named (metal_design.rs:509, exact at every probe). Nothing calls the rating a peak or a nominal rating (2A). "Fatigue from repeated slips is not checked." is true (8A, backlog BL-066) and agrees with the torque-band watch-out ("repeated slips also load the gearbox in fatigue, which nothing checks"). It makes no protection claim. 29 words. |
| 11 | Rating help | PASS | Unchanged. It agrees with the new engine help: the same comparison, the same two sides. The spreadsheet note "GAM 5:1 peak input rating." still reads as the spreadsheet's own labelled note (R-11). |
| 12 | Ripple label "Nominal slip swing at the gearbox input" | PASS | Unchanged. With the new help: "nominal" is "without the production allowance", and "at the gearbox input" is the output's pull-out divided by ratio and efficiency. |
| 13 | Ripple help | PASS | M-2 applied verbatim. The pull-out it divides is named at the gearbox output, and the row is exactly pull-out / (ratio x efficiency) at every probe (model.rs:1156). "At operating temperature" holds with the operating temperature below the minimum (8). It claims no ordering with cold-high (PD-2): the swing is above cold-high at the input in three of the four (8) designs. No "warm", "peak" or "guarantee". 30 words, 2 sentences. |

Routed points (a)-(f): round 1's judgments stand; no string they concern changed except as M-1 and M-2 settled them. (b)'s first point and (d)'s side point are now fixed in strings 4 and 13.

**4. MUST-FIX.** None.

**5. SHOULD-FIX.** No new one. Round 1's S-1 (sentence 1's equal readouts within 5e-5 N·m above the rating), S-2 ("a smaller measured variation" against the row "Production variation allowance") and S-3 (the gear ratio's help still says "the slip ripple", while the row and the new help now say "slip swing") stand as round 1 offered them, open for the controller. This round does not re-judge them. S-3's case is slightly stronger now: the ripple help itself opens with "Slip swing:".

**6. Traps checked.**
- Side of the gearbox: every torque in strings 4, 10 and 13 names its side.
  - String 4: the requirement at the output, and the loads on the input.
  - String 10: cold-high at the input and at the output, and the input rating.
  - String 13: the pull-out at the output, and cold-high at the input.
  - "Cold-high is 1.673 times hot-low" is a ratio on one side.
- Efficiency: "divided by ratio and efficiency" in strings 10 and 13 raises the input torque (ratio 1: 7.647 at efficiency 0.5 against 3.823 at 1). No text says efficiency reduces the load.
- Cold-high as an expectation:
  - No new text bounds cold-high with "at most"; "at most" bounds only the requirement and the band.
  - "Even just meeting ... puts {X}" states the model's cold-high at the gearbox input of a design that just meets the requirement. The clause before says "cold-high is 1.673 times hot-low here", and sentence 1 states the same model quantity for this design. It does not say every unit reaches it.
  - The word "puts" is the spec's owner-approved draft wording, and round 1 judged it (probe 12).
- Protect: mentions_gearbox_protection is false for strings 4, 10 and 13. The protection-ban tests pass.
- Peak: none of strings 4, 10, 11, 12 and 13 says "peak". The test bans it in the rating help, both ripple texts and the engine help. The rating is called only "the gearbox input torque rating" or "input rating" (2A).
- Ripple row: the nominal swing at the operating temperature, with no "warm".
- Band and geometry: sentence 2 still says no size, gap or count passes both checks. The band reads 1.6731 at every NdFeB design probed here: the defaults, 8 magnets, 304 back iron, 12 poles, 12 poles at 20 mm length, the 20 mm ring radius and Auto at 13.77 mm. Round 1 found it constant over geometry to 2.2e-16.
- Forbidden words: no "hot-day", path, id, cell, "BL-059", "Rust-only" or "clamp" in strings 4, 10 and 13. plain_text_violations is empty for each and for the full Problems at (1), (9) and (11).
- Glyphs: no new glyph; the parentheses in string 10 pass the guard.
- Caps:
  - string 10: 29 words, 2 sentences;
  - string 13: 30 words, 2 sentences, no braces or "guarantee";
  - string 12: 39 characters;
  - each Problem sentence keeps its sentence count;
  - the Why is unchanged (28 words, 1 sentence).
- Typed numbers: none in strings 4, 10 and 13. Every number in sentence 2 is formatted from the results and inputs.

**7. Context lines.**
- Trade-off line, face-gap side effect and status line: unchanged on the round-2 sheet. Round 1's "fine" stands:
  - "One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Gearbox input torque needs it lowered, so change another input to settle one of them.";
  - "Raising Magnet face gap also lowers the pull-out, so Hot-low torque falls further.";
  - "Example design: fails 4 of 8 checks" while the clamp section is paused ("4 of 9" on the sheet, unpaused).
- Recorded, no action. Under Auto, the design is the one that "just meets" the requirement, and sentence 1 can read one last digit below sentence 2's quoted load: 27 of 60 sampled requirements, for example 0.6340 against 0.6341, with the unrounded value 0.634017.
  - This is the deliberate rounding of Task 3 I-2: the load is rounded up, so that a gearbox rated for the quoted torque takes it.
  - It changes numbers, not text (B-6), and a 4-significant-digit reader takes it as rounding.
  - "Even just meeting" makes it no worse than the old "meeting".
  - No design that meets the requirement with room reads below the load (2,301 of 2,301).

**8. Verdict:** PASS (no must-fix; round 1's S-1 to S-3 remain offered).

### Re-review after the whole-branch review, fix wave 1 (2026-10-17)

The BL-059 whole-branch review's fix wave 1 (2026-10-17; `final-review.md` MF-1, commit 6cbc852) changed Problem
sentence 1 (`gearbox_problem`, `gui/guidance.rs`). Where cold-high at the gearbox input and the rating would read the
same to 4 significant digits, the torque is now quoted one last digit past the rating (`quoted_load`, rounded up), so
on the sliders a failing check never reads two equal numbers; everywhere else it reads as before, and the template is
unchanged.
Round 1's S-1 (more decimals) offered another answer to the same defect. The changed text had its G-3 re-review in a
fresh context at HEAD d2565e5. The reviewer's report for the record follows unchanged, its headings demoted one level
(full answer: `.superpowers/sdd/2026-10-16-bl059-gearbox-check/textreview-T10-the-whole-branch-review-fix-wave-1-1.md`).

- Reviewer: fresh context, session model, 2026-10-17; G-3 re-review of gearbox Problem sentence 1 as changed by MF-1
  (round 1).
- Verdict: PASS, no must-fix. Sentence 1 is true at the defaults, at the slider design that read "0.5400 ... 0.5400"
  (now "0.5401 ... 0.5400"), at every Auto tie solved, at the rating slider's ends, at the reachable rounding edges
  and with the operating temperature below the minimum; it breaks none of the spec section 6 traps.
- Should-fix offered, neither for this branch's on-screen text:
  - SF-1 (a follow-up item, not this branch): the hot minimum Problem, unchanged by BL-059, can read two equal numbers
    while its check fails (825 cases in a sweep of slider designs; for example "Hot-low torque 0.9300 N·m is below the
    required 0.9300 N·m"); file a backlog item to quote the hot-low torque rounded down past the requirement where
    both read equal, and to audit the other Problems for the same pattern.
  - SF-2 (docs precision, optional): the README's "The gearbox Problem's first sentence never reads two equal
    numbers" holds on the sliders only (with design-file values, a non-positive cold-high at the gearbox input beside
    a negative rating can read "-0.1000 against -0.1000"); the exact text offered is "The gearbox Problem's first
    sentence never reads two equal numbers on the sliders:".

**Report for the record (as returned).**

#### The changed text, as drawn (probe output at HEAD)

| Design | cold-high at input (unrounded) | Readout | Sentence 1 | Badge |
|---|---|---|---|---|
| defaults | 0.804907446 | 0.8049 N·m | "Cold-high at the gearbox input is 0.8049 N·m, above its 0.6000 N·m input rating." (unchanged; sentence 2 unchanged: 1.673, 0.8806, 1.703, 1.140) | 0.80 against 0.60 |
| P1: 8.01 mm, 1.00 mm, 0.54 N·m (each a slider step) | 0.540040364 | 0.5400 N·m | "... is 0.5401 N·m, above its 0.5400 N·m input rating." + sentence 2 (1.673, 0.8806; largest requirement 1.533 N·m; widest band 1.026) | 0.55 against 0.54 |
| Auto ties (24 solved: 6 from each of the 4 tie-test bases) | rating + 6e-14 to 5e-11 | equal to the rating | rating + one last digit, e.g. "0.08001 ... 0.08000", "0.2001 ... 0.2000", "0.9201 ... 0.9200", "1.841 ... 1.840", "3.241 ... 3.240"; no sentence 2 (a tie) | rating + 0.01 |
| defaults, rating 0.8049 or cold-high - 1e-6 (file values) | 0.804907446 | 0.8049 | "... is 0.8050 N·m, above its 0.8049 N·m input rating." + sentence 2 | 0.81 against 0.80 |
| corner (op -40 °C, min 20 °C), rating cold-high - 1e-7 (file) | 0.700416510 | 0.7004 | "... is 0.7005 N·m, above its 0.7004 N·m input rating." (sentence 1 only, PD-1) | 0.71 against 0.70 |

#### Sweep

The sweep covered the axial length at every slider step × face gap 0.5, 1.0, 1.5 and 2.0 mm (19,524 geometries with hot-low > 0) × every rating step from 0.05 to 5.00.
- Failing (geometry, rating) pairs: 3,611,518.
- Pairs in the new branch, where the two would read equal: 743.
  - 29 of them show sentence 2.
  - 714 meet the requirement.
- The quoted cold-high minus the true value lies between 0.50 and 1.00 of the rating's last digit. It never goes below the true value and never exceeds it by more than one last digit.
- Where the design meets the requirement, sentence 1 never reads below sentence 2's quoted load (0 of 714). Inside this window MF-1 also removes the earlier last-digit mismatch (nearest-rounded sentence 1 against rounded-up sentence 2), because both now use `quoted_load`, which is monotone. Outside the window, sentence 1 still rounds to the nearest, as G-3 round 2 recorded and accepted. That is unchanged.

#### Edge pairs

Each pair (true value against rating) produced:
- Decade carries: 0.99998 against 0.99996 gives "1.001 against 1.000"; 9.99996 against 9.99995 gives "10.01 against 10.00".
- Rating slider ends: 0.05 gives "0.05001 against 0.05000"; 5.0 gives "5.001 against 5.000".
- Scientific range: "1.001e-4 against 1.000e-4".
- Off-grid ratings: correct.
- File-only non-positive values: "-0.1000 against -0.1000" (SF-2).
- The sentence-1 parser in `sentence_1_numbers` reads every one of these forms.

#### Traps and rules (spec section 6, plan Global Constraints)

The sentence template is unchanged and was reviewed at G-3 round 1. It holds against each rule:
- the torque names its side ("at the gearbox input");
- the rating is "its ... input rating", never "peak";
- no efficiency wording;
- no "protects";
- cold-high is not used as a bound;
- no "hot-day", "warm", clamp, id or path;
- the number is in readout format (4 significant digits) through `number_text`, with the result's own unit, "N·m";
- `plain_text_violations` cannot change, since only digits differ.

"Above" is true wherever the Bad arm runs, because the engine is strict (`cold_high_input_Nm > rating`). The quoted torque is rounded up, toward the load side. This is the house rule for a torque on the gearbox input set against its rating (sentence 2's `quoted_load`, review I-2), and the badge's +0.01 rule (spec 4) goes further. So nothing understates the load.

#### Why `quoted_load`, not G-3 S-1 (more decimals)

S-1 reads exactly at P1 ("0.54004 against 0.54000"). At an Auto tie, though, the excess is 6e-14 to 5e-11 N·m. Telling the two numbers apart there takes 11 to 14 decimals ("0.50000000003 ... 0.50000000000"). That puts solve tolerance on screen, which a non-specialist engineer would read as noise or a bug. `quoted_load` reads one last digit past the rating at every tie and at P1. It is conservative, it uses the convention the badge and sentence 2 already use, and it never moves more than one last digit.

#### Recorded, no change asked

- In the window only, which is half a last digit above the rating (5e-5 N·m for ratings from 0.10 to 0.99, 5e-4 N·m from 1.00 to 5.00), sentence 1 can read one last digit above the result's own readout (0.5401 in the Problem, 0.5400 in the results table). This is the deliberate price of "never two equal numbers". It is smaller than the badge's own difference (0.55) beside it. Nothing reads false: the quoted figure is the load rounded up, and "above" is true.
- At an exact Auto tie the check fails by solve tolerance (about 1e-11 N·m), and sentence 1 now reads "0.5001 above 0.5000". The final review already recorded this as an engine-tolerance question for the owner, not this branch. The text is true under the engine's strict rule.
- The code comment at `gearbox_problem`, the test doc comments (`sentence_1_never_reads_two_equal_numbers`, `assert_gearbox_problem_consistent`, the MF-3 pointer) and the two README additions match the code's behaviour. The README examples (0.5001 against 0.5000; 0.5401 against 0.5400; the defaults' 0.8049 against 0.6000) are what the code draws.
- Tests: the `gui::guidance` module on the HEAD tree gives 93 passed, 0 failed, 3 ignored (two of the ignored are my scratch-only probes), run with `--features gui` on the scratch copy. The new assertion (sentence 1's parsed torque > rating wherever Bad) runs through every caller of `assert_gearbox_problem_consistent`, including the tie test.
- G-3 S-2 and S-3 (round 1) remain open as offered. They are not part of this changed item, and I re-reviewed neither.

- No must-fix, and no text changed in this round. SF-1 and SF-2 stay open for the controller. Round 1's S-1 is
  answered by MF-1 in another form: the reviewer prefers `quoted_load` to S-1's more decimals, which at an Auto tie
  would need 11 to 14 decimals. Round 1's S-2 and S-3 stay open as offered; this round did not re-review them.

- Correction (2026-10-17, BL-059 review leftovers; task 8 review m-1): the bullet "Tests that followed the changes (each new assertion was seen failing on the old text first)" in this section's record above overstates. The "requirement at the gearbox output" assertions were seen failing on the old text; the "even just meeting" bound in `assert_gearbox_problem_consistent` cannot fail: it takes the band from the same design's results, so wherever hot-low meets the requirement it holds by construction, and the twelve-pole case gives it no way to fail. The band's independence of geometry, which sentence 2 rests on, is tested by `the_band_is_the_same_at_every_magnet_size_gap_count_and_back_iron` (`magcoupling-rs/tests/gearbox.rs`, whole-branch review MF-3).

### Re-review of the review leftovers (2026-10-17)

The BL-059 review leftovers (owner decision 2A, 2026-10-17; commit a11832e) changed two on-screen strings, round 1's
S-3 and S-2: the gear ratio's help ("such as the slip swing") and `SMALLER_VARIATION`, the variation allowance in
gearbox Problem sentence 2's bracket ("a smaller variation allowance backed by measurement"). Both had a G-3 text
re-review in a fresh context. The reviewer's report for the record follows unchanged, its headings demoted one level
(full answer: `.superpowers/sdd/2026-10-16-bl059-gearbox-check/leftovers-g3-1.md`).

- Reviewer: fresh context, session model, 2026-10-17; G-3 re-review of the two changed strings (round 1).
- Verdict: MUST_FIX (one item, M-1). The gear ratio's help passes. The band clause is true at every reachable design,
  but at the defaults, the first gearbox Problem a user sees, "(a warmer minimum temperature, a smaller variation
  allowance backed by measurement or SmCo magnets narrow it)" lets "or SmCo magnets" attach to "by": a false reading,
  since SmCo magnets do not back a smaller variation allowance.
- M-1, applied as the reviewer's option A in the commit that adds this record: the allowance is named last, after SmCo
  magnets, so no "or" follows "backed by measurement"; the words are unchanged. At the defaults the bracket now reads
  "(a warmer minimum temperature, SmCo magnets or a smaller variation allowance backed by measurement narrows it)";
  the other four reachable forms are unchanged. New test `sentence_2_names_the_variation_allowance_last` (each of its
  three assertions seen failing on the old order first).
- No should-fix.

**Report for the record (as returned).**

#### Read
- The brief.
- The plan's head up to "## Tasks": Global Constraints with the physics traps, Review Focus, R-1 to R-13, plan review 1A-8A, PD-1 to PD-4.
- The earlier G-3 record under "## BL-059 gearbox check" in `docs/analyses/2026-10-14-magcoupler-help-review.md` (rounds 1 and 2, the MF-1 re-review, and the m-1 correction).
- Spec section 5 (the band clause's draft, and the variation "not offered as a fix").
- Commit a11832e: its message and its source diff.
- Code:
  - `gui/plain_text.rs`: LABELS (:64-78), HELP for the gear ratio, the efficiency, the rating and the ripple, `the_gearbox_texts_claim_what_the_engine_does` (:2080-2145);
  - `gui/guidance.rs`: `listed` (:1407), `gearbox_problem`, the constants, `band_levers`, `gearbox_conflict` (:1724-1894), `band_lever_designs`, `sentence_2_names_only_what_narrows_that_design_s_band`, `sentence_2_names_the_variation_by_its_input`;
  - `gui/results_table.rs` :143 (the table draws `display_label`);
  - `engine/model.rs` :349 and :1156-1157, `engine/metal_design.rs` :507-509, `engine/api.rs` :237-252;
  - `engine/grades.rs` (the SmCo coefficients);
  - `engine/explain/notes.rs`: the torque-band note's variation change and sources, and the slipping note's ripple sentence.

#### Probes
I built a scratch crate (`g3probe`) in my scratchpad with its own CARGO_TARGET_DIR. It depends on `magcoupling-rs` by path with `gui`, calls `compute_all` and the gearbox guide's `(g.problem)`, and runs the crate's own `word_count`, `sentence_count`, `plain_text_violations` and `mentions_gearbox_protection`.

| Design | Cold-high at input | The bracket, or what shows instead |
|---|---|---|
| Defaults | 0.804907 | "(a warmer minimum temperature, a smaller variation allowance backed by measurement or SmCo magnets narrow it)": M-1 |
| Rating 0.80 | 0.804907 | Same bracket; band at most 1.520 |
| Ring radius 20 mm | 0.760233 | Same bracket |
| Variation 0.5 | 1.049879 | Same bracket; band 3.710 |
| No variation | 0.699920 | "(a warmer minimum temperature or SmCo magnets narrow it)" |
| SmCo 2:17 (Recoma 26) rings, 25 mm length, rating 0.60 | 0.899057 | "(a warmer minimum temperature or a smaller variation allowance backed by measurement narrows it)" |
| Minimum at the operating temperature | 0.804907 | "(a smaller variation allowance backed by measurement narrows it)" |
| Br coefficient 0 | 0.700417 | "(a smaller variation allowance backed by measurement narrows it)" |
| Rating 0.05, ratio 1, efficiency 0.5 | 0.8049 / 4.025 / 1.529 | Second form, no bracket |
| Operating -40 °C, minimum 20 °C | 0.700417 | Sentence 1 only (PD-1) |
| 8 magnets; SmCo rings at the defaults | 0.589311; 0.432037 | Within the rating, so no Problem |

Every Problem probed:
- has an empty plain-words guard, no protection wording, and its expected sentence count (3, or 1 in the corner);
- matches the earlier G-3 record's numbers (1.673, 0.8806, 1.703, 1.140; 2.271 and 1.520; 0.1419 and 0.5264; 0.3406 and 2.632).

#### String 1: the gear ratio's help (PASS)

The help now reads: "Reduction ratio of the gearbox; the coupling sits on its wheel side. A higher ratio lowers the coupling torques referred to the gearbox input, such as the slip swing."

- **One name.** "The slip swing" matches the quantity's on-screen name everywhere it is drawn:
  - the row label "Nominal slip swing at the gearbox input", which the results table also draws through `display_label`;
  - the ripple help, which opens "Slip swing:".

  Dropping "Nominal" names no other quantity, since no other slip swing is on screen. "Slip ripple" no longer appears in any on-screen GUI text. The only "slip ripple" left is the workbook's Python-locked "Measure resonance and slip ripple; tune the traction loop." (`engine/metal_design.rs` :235), which this change leaves alone.
- **True at every reachable input.** The ripple is T_pull / (ratio × efficiency) (`model.rs` :1156), and the ratio enters the engine nowhere else that feeds T_pull: `api.rs` passes it only to `metal_design`, for the two input-side torques at :507-509. So a higher ratio lowers the swing, and cold-high at the input, at every design.
- **Traps.**
  - The swing is introduced as one of "the coupling torques referred to the gearbox input", so its side is named.
  - No "warm", "peak", "protect", efficiency claim, cell, id or path.
- **Caps.** 29 words in 2 sentences; guard empty.
- **Fit beside the neighbouring texts.** It agrees with the efficiency help ("refers the coupling torques to the gearbox input; a higher efficiency lowers them there") and with the ripple help.
- **The example stays the swing, not cold-high.** Spec 5 keeps the ratio out of the fixes ("not a real part"). Naming cold-high, the checked torque, in this help would read as an invitation to use the ratio as a lever. "Such as" is not exhaustive, so nothing is lost.

#### String 2: the band clause (true, but fails "one list": M-1)

- **One name.** "Variation allowance" matches:
  - the input's label, "Production variation allowance";
  - the hot-minimum Problem ("the 15 % variation allowance");
  - the clamp Problem.

  "Backed by measurement" agrees with the input's help ("Engineering allowance, not measured.") and with the torque-band note's change for the input ("only with measured production scatter").
- **Physics.**
  - The band is (1 + v)/(1 - v) times the temperature terms, so a smaller allowance narrows it wherever v > 0. That is exactly where it is named (`inputs.metal.variation > 0.0`), and `sentence_2_names_only_what_narrows_that_design_s_band` checks it one slider step away.
  - The new words are more exact than "a smaller measured variation": the band is computed from the allowance input, not from any measurement.
- **Traps.**
  - No geometry is said to narrow the band.
  - No torque is in the bracket, so no side is needed.
  - No "hot-day", "peak", "warm" or "protect".
  - The guard is empty on every Problem probed.
- **Reading.** One list in four of the five forms. The three-item form, which is the defaults' form, is M-1.

#### Recorded, no change asked
- **"Wheel side" against "gearbox output".** The gear ratio's help says the coupling "sits on its wheel side", while the ripple help and sentence 2 say "at the gearbox output". This is the same side in two plain wordings, not two names for one quantity. The help is pre-existing (slice 2, reviewed), and the S-3 change did not touch it.
- **The slipping note.** It still says "sends torque ripple to the gearbox: at its input, here {model.gearbox_input_ripple_Nm}" (`notes.rs` :733), and its source line says "the ripple at the gearbox input". The note is signed off and out of this change's scope. After S-3, every overlay text and the dashboard say "slip swing"; only the note keeps "ripple". It could be a follow-up item if the owner wants one name in the notes too.
- **Exports.** They keep the catalogue label "Estimated gearbox input torque ripple amplitude" (`results_table.rs` :352, :371). Exports are not overlay text.
- **The commit body.** a11832e's message says the new constant "keeps the bracket one plain sentence". It is still one sentence, but not one unambiguous list until M-1 lands. The commit is history, so it needs no change.

### Re-review of the review leftovers (2026-10-17), round 2

Round 2 of the G-3 text re-review of the leftovers' two on-screen strings, after commit 80d4d60 applied round 1's M-1
(the variation allowance named last in gearbox Problem sentence 2's bracket, after "SmCo magnets"). The reviewer worked
in a fresh context. Its report for the record follows unchanged, its headings demoted one level (full answer:
`.superpowers/sdd/2026-10-16-bl059-gearbox-check/leftovers-g3-2.md`).

- Reviewer: fresh context, session model, 2026-10-17; G-3 re-review of the two changed strings (round 2).
- Verdict: PASS. No must-fix, no should-fix. M-1 is fixed as option A asked: at the defaults the bracket reads "(a
  warmer minimum temperature, SmCo magnets or a smaller variation allowance backed by measurement narrows it)", and in
  the reviewer's sweep of 201,620 designs no "or" follows "backed by measurement". The gear ratio's help is unchanged
  since round 1 passed it.
- No text changed in this round. The four items recorded with no change asked (the remedy sentence's two comma styles,
  the `band_lever_designs` doc's wording and order, the historical quotes of the old forms in this record, commit
  80d4d60's type) stay as recorded.

**Report for the record (as returned).**

#### Scope: what changed since round 1

`git diff 741f02b HEAD -- magcoupling-rs/src` (round 1's base to HEAD) touches only `magcoupling-rs/src/gui/guidance.rs`:
- **Constants (:1756-1761).** Reordered to WARMER_MINIMUM, SMCO_MAGNETS, SMALLER_VARIATION. Their doc says why: "the variation allowance last, so that no "or" follows "backed by measurement"". The three strings are byte-identical to round 1's.
- **`band_levers` (:1792-1799).** The `names` array is reordered to match. The doc example (:1764-1766) shows the new three-item form. The verb `match` and its comment ("only SmCo magnets is plural") are unchanged and still accurate.
- **New test `sentence_2_names_the_variation_allowance_last` (:5418-5444).**

`plain_text.rs` is untouched, so the gear ratio's help ("... such as the slip swing.") is still the text round 1 passed. Per the brief, I did not re-judge it.

#### Read
- The brief.
- Round 1's answer (`leftovers-g3-1.md`).
- The plan's head up to "## Tasks": Global Constraints and physics traps, Review Focus, R-1 to R-13, 1A-8A, PD-1 to PD-4.
- Under "## BL-059 gearbox check" in `docs/analyses/2026-10-14-magcoupler-help-review.md`:
  - the earlier G-3 record (rounds 1 and 2, the MF-1 re-review);
  - the new "Re-review of the review leftovers (2026-10-17)" section.
- Commit 80d4d60: its message and its diff of the source, the README and `docs/ai/05-update-tracker.md`.
- Code:
  - `guidance.rs`: `listed` (:1407), `gearbox_problem`, the constants, `band_levers`, `gearbox_conflict` (:1724-1896), `band_lever_designs` (:2459-2510), and `sentence_2_names_only_what_narrows_that_design_s_band`, `sentence_2_names_the_variation_by_its_input` and `sentence_2_names_the_variation_allowance_last` (:5342-5444);
  - `engine/grades.rs`: every grade's family and Br coefficient;
  - `gui/pickers.rs`: the grade labels.

#### Probes

I used the scratch crate `g3probe` (depends on `magcoupling-rs` by path with `gui`; its own target directory). It draws the gearbox guide's `(g.problem)` text and runs the crate's own `plain_text_violations`, `mentions_gearbox_protection`, `sentence_count` and `word_count`.

**Named designs.** The full Problem was drawn at each. Every one has an empty guard, no protection wording, and 3 sentences, or 1 in the PD-1 corner.

| Design | Bracket, or what shows instead |
|---|---|
| Defaults (0.8049 at the input) | "(a warmer minimum temperature, SmCo magnets or a smaller variation allowance backed by measurement narrows it)" |
| Rating 0.80; ring radius 20 mm; variation 0.5; Br coefficient at its bottom (-0.0025) | Same three-item bracket |
| No variation allowance | "(a warmer minimum temperature or SmCo magnets narrow it)", unchanged |
| SmCo 2:17 rings, 25 mm length | "(a warmer minimum temperature or a smaller variation allowance backed by measurement narrows it)", unchanged |
| Minimum at the operating temperature; Br coefficient 0 | "(a smaller variation allowance backed by measurement narrows it)", unchanged |
| Rating 0.05, ratio 1, efficiency 0.5; SmCo rings at no variation and a rating of 0.30 | Second remedy form (PD-1), no bracket |
| Operating -40 °C, minimum 20 °C | Sentence 1 only |

The numbers match the earlier records: 1.673, 0.8806, 1.703 and 1.140 at the defaults; 2.271 and 1.520 at a rating of 0.80.

**Sweep.** 201,620 designs:
- every grade choice:
  - library parts;
  - each of the 17 grades on both rings;
  - an inner library part beside an outer ring of each grade;
- the Br coefficient every tenth slider step, plus the SmCo boundaries (-0.00045 and -0.00035, each and one step below) and the top two steps;
- allowances of 0, 0.01, 0.15 and 0.5;
- minimum temperatures every tenth slider step;
- operating temperatures -20, 20, 50, 100 and 150 °C;
- the defaults' rating.

Of these, 111,175 draw a bracket, in exactly five forms:

| Count | Form |
|---|---|
| 59,417 | "a warmer minimum temperature, SmCo magnets or a smaller variation allowance backed by measurement narrows it" |
| 32,643 | "a warmer minimum temperature or a smaller variation allowance backed by measurement narrows it" |
| 13,606 | "a warmer minimum temperature or SmCo magnets narrow it" |
| 3,310 | "a warmer minimum temperature narrows it" |
| 2,199 | "a smaller variation allowance backed by measurement narrows it" |

The sweep found no rule break:
- no "backed by measurement or" and no "backed by measurement,";
- the verb is "narrow" exactly where SmCo magnets is last, and "narrows" everywhere else;
- SmCo magnets is never first;
- every drawn Problem has an empty plain-words guard and no protection wording.

**Tests.** `cargo test --locked --features gui --lib -- sentence_2_ gearbox_problem` ran 11 tests and all passed, the new test among them.

#### The band clause, re-read (PASS)

**One list.** "A, B or C narrows it" with no serial comma is the house list form (`listed`). Each item is a noun phrase that cannot attach to its neighbour:
- "a warmer minimum temperature, SmCo magnets" is not an apposition;
- "backed by measurement" now ends the list, so no later item can be taken as a second object of "by".

The one other attachment a reader could try is "(SmCo magnets or a smaller variation allowance) backed by measurement". "Magnets backed by measurement" makes no sense, so the narrow reading wins. Even read the wide way, it would only add a needless condition; it would not make a false claim.

**Verb.** With "or", the verb agrees with the nearest item: "a smaller variation allowance ... narrows it". In the [warmer, SmCo] form, "SmCo magnets ... narrow it" is still right.

**Order.** The reorder changes only the three-item form. The brief's "2-item and 1-item forms unchanged" holds at every input, not just on the sliders:
- the one two-item form whose order would change, [SmCo, allowance], cannot be reached;
- SmCo is named only with every ring's coefficient below every SmCo grade's, and those coefficients (-0.00035, -0.00045) are const and below zero;
- so every ring's coefficient is below zero, and the warmer minimum (same `spread` condition) is named too.

A design file cannot reach [SmCo, allowance] either, because this is an implication on the same two values.

**One name per quantity.** The words are unchanged:
- "variation allowance" is the input's label ("Production variation allowance");
- "SmCo" is the family word the grade picker and the grade hover use;
- "minimum temperature" names the minimum-temperature input.

**Physics traps.** No torque is in the bracket, so no side is needed. No geometry is named as narrowing the band. There is no "peak", "warm" (for the ripple row), "hot-day" or "protect", and no cell, id or path. Ruling R-3 still holds: each change is named exactly where one step of it narrows the band (`sentence_2_names_only_what_narrows_that_design_s_band`, unchanged and passing).

#### The new test

`sentence_2_names_the_variation_allowance_last` makes three assertions:
- it pins the defaults' bracket as drawn;
- at all seven `band_lever_designs()`, `"{SMALLER_VARIATION} or "` is absent;
- wherever the allowance is named, `"{SMALLER_VARIATION} narrows it)"` is present.

The second and third together force the allowance to be last in any list:
- in the middle, the "or" check fails;
- first, the "narrows it)" check fails.

Every reachable form is among the seven designs. Each assertion fails on the old order, which the commit reports was seen red first:
- the old defaults' bracket lacks the new literal;
- it contains "... backed by measurement or ";
- it ends "narrow it)".

#### Docs

All three docs agree with the code:
- **`magcoupling-rs/README.md` :677.** The `guidance.rs` row names the new order ("a warmer minimum temperature, SmCo magnets or a smaller variation allowance (always last, so no "or" follows "backed by measurement")") and the new test.
- **`docs/ai/05-update-tracker.md` :17.** One clause, with the defaults' bracket quoted correctly.
- **`docs/analyses/2026-10-14-magcoupler-help-review.md` :2476-.** The round-1 record is appended, and M-1 is recorded as applied with the correct new reading.

No other doc in `docs/ai` or the README quotes the bracket.

#### Recorded, no change asked

- **Comma styles in one sentence.** The remedy sentence uses a serial comma in its outer list ("a gearbox rated for that, a requirement of at most 1.703 N·m, or a band ...") and none in the bracket.
  - The outer comma is needed: without it, "at most 1.703 N·m or a band" would attach.
  - The bracket follows `listed`, which the temperature Problem's literal-pinned lists share. Round 1 weighed and declined a serial comma there (its option C).
  - This is pre-existing and unchanged by this fix.
- **The test helper's doc (`band_lever_designs`, :2459-2463).** It still calls the lever "a smaller variation", and it lists the flags in the order [warmer, variation, SmCo].
  - That order is the order of the test's `changes` array, which it zips with, so the helper is consistent with itself.
  - It is not the on-screen order, and it is not on screen. An optional tidy only.
- **Historical quotes of the old forms in the help-review record.** These are :2162 and :2211 (the earlier rounds' as-returned reports) and :2517 (round 1's probe table, which marks the row as M-1). They record what was on screen at the time. No change.
- **Commit 80d4d60's type.** It is typed `docs(magcoupler)` but carries the on-screen fix and a new test. It is history, so it needs no change.

## BL-064 shaft joints

BL-064's G-3 review of every new on-screen text (plan Task 8) is recorded first in this section, with each round's
report as returned. The text re-reviews made while Tasks 1 to 7 were built follow it, in the order they ran.

- Scope: BL-064's new on-screen strings (spec
  `docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md`, sections 6.2 and 6.3), reviewed
  under the same G-3 rule:
  - the engine (`engine/shaft_joint.rs`): the 15 input labels and helps, the 24 result labels and
    helps (the check's label is the dashboard row's label), the eleven verdict texts
    (`SHAFT_JOINTS_PASS`, the `SHAFT_JOINT_UNKNOWN` forms with the safety factor's, `FIT_SLIPS`, `BORE_OVER_ITS_LIMIT`,
    `CANNOT_BE_FITTED`, `KEY_OVER_ITS_LIMIT`, `KEY_HAS_NO_BEARING_AREA`) with their tails
    (`JointTerms::text`) and the joint names;
  - the inner ring's single-ring onset (`engine/temperature.rs`): its label and help;
  - the four assumption rows (`engine/assumptions.rs`: `fit_friction`, `fit_smoothing`,
    `rotor_over_shaft`, `joining_clearance`), their labels and rationales;
  - the guide (`gui/guidance.rs`): its label and Why (`SHAFT_JOINT_GUIDE`) and every Problem
    sentence (`shaft_joint_problem`);
  - the headings: "Shrink fit" and "Fitting temperatures" (`gui/inputs.rs`, `WORKFLOW`), "Shaft
    joints" in `SECTION_LABELS`, `CHAIN_LABELS` and `PACKAGE_LABELS` (the last never drawn);
  - the stored-3D hover note's new sentence (`gui/dashboard.rs`, `STORED_3D_NOTE`);
  - read in context: the four overlay helps the spec's section 5.3 offered as should-fix
    candidates (the bore, the keyway depth, the boss length, the single-ring reverse field),
    the fix texts, the status line ("Example design: fails 4 of 9 checks"), the stored-3D label on
    the joint row and Explain's closest-note line; the note `shaft_joint` for consistency only
    (its own physics review is recorded under `## shaft_joint` of
    `docs/analyses/2026-10-14-magcoupler-notes-review.md`).
- Read: the sheets `magcoupling-rs/target/g3/BL064-inputs.txt` (`help_for_review`, group
  `shaft`), `BL064-guidance.txt` (`guidance_strings_for_review`), `BL064-overlay.txt`
  (`overlay_strings_for_review`), `BL064-note.txt` (`notes_for_review`) and `BL064-strings.diff`
  (`git diff -U0 main` of the source files); the spec, sections 1 to 6 and 11; the engine and
  GUI code behind every string. Each round probed the engine from a scratch crate outside the
  repository:
  - round 1 (2026-10-19):
    - the defaults: "Both shaft joints hold and can be fitted", no Problem, "Example design: fails 4 of 9 checks" with
      the clamp section paused; required 7.6466 N·m (2 × cold-high 3.8233); holding 9.4844 / 16.899 N·m; bore stress
      241.70 / 312.53 against 345.83 MPa; key 127.44 / 101.95 MPa; needed 12.082 / 11.310 µm; elastic 23.808 /
      23.863 µm; shrink 18.988 / 0 µm; fittable 18.828 / 24.440 µm; fitting 93.267 / 280.16 against 100 / 300 °C; the
      inner ring's and the governing single-ring onset both 143.356 °C;
    - the five probe designs: 8 magnets passes (5.598 N·m required); 304 reads "The fit slips (inner hub): holds 0.00
      N·m against 4.83 N·m" with both empty-window sentences; Y30 passes (inner onset +inf); the 20 mm ring passes
      (11.77 against 7.22 N·m);
    - the spec's section 3.6 designs: hub minimum 10 (3.48 against 7.65 N·m) and cup minimum 10 (4.36 N·m); hub
      maximum 26 / 28 (385.1 / 421.0 MPa) and cup maximum 26 / 28 (384.0 / 419.8 MPa); friction 0.10 (6.77 N·m);
      smoothing 0 (the cup 401.9 MPa); the wheel shaft at 20, -78.5 and 40 °C (needs 247.6, 163.5 and 266.3 °C); the
      gearbox shaft at -20 and 40 °C (243.1 and 298.9 °C); the bare cup allowed 250 °C; DP460 (93.3 against
      60.0 °C); adhesive code 0 (unknown); 2214 Hi-Temp with 150 °C typed and the hub maximum 26 µm (133.4 °C allowed:
      the onset 143.4 °C less 10.0 °C); the single-ring caps (N50M 88.80 °C, inner N42M with outer N35 81.35 °C,
      B842 in both rings 59.83 °C; inner B842SH with outer B842 passes at 100 °C); keyway 0.3 mm ("Key backup above
      its limit (inner hub): 509.8 MPa against 345.8 MPa") and 0; boss length 0 (the cup slips at 2.73 N·m) and hub
      length 3.0 mm (2.19 N·m); bore 16 mm with a 12 mm boss, and with a 2.1 mm keyway (no wall); the reversed hub
      band; typed: bore 0, key width 40 mm, contact height -0.5 mm; safety factor 0 and -1 (unknown, Caution, no
      fix); back-iron pick 99 and the aluminium body; an axial override of 40 mm (the cup slips at 16.90 against
      25.99 N·m); c_end 1.5 and the short magnets (greyed); the BL-067 corner (cold-high 3.1823 N·m, below the
      operating pull-out with its allowance, 3.657 N·m); the single-ring field at 1500 kA/m ("needs 93.3 °C against
      -12.4 °C") and 862 kA/m ("93.3 °C against 93.2 °C");
    - every shaft-joint input at both slider ends; which temperature end binds (416 holds 14.0443 N·m at a rotor
      allowance of 0, 10 and 20 °C); the slip reversals per event (16.7 cycles at the defaults, 0.833 at 100 rpm,
      0.033 at 100 rpm with 4 magnets and a 0.01 s event); one step of each C1 overlay help's input;
  - round 2 (2026-10-19), on round 1's changes, their five tests (`tests/shaft_joint.rs` 34, `tests/assumptions.rs`
    13 and `tests/schema.rs` 11 passed) and the sheets regenerated after them (only the two changed input helps
    moved):
    - MF-1: 16.667 cycles per event at the defaults, 0.8333 at 100 rpm, 0.0333 at 100 rpm with 4 poles and a 0.01 s
      event, 50 at 6000 rpm;
    - MF-2, the inner hub's holding at a rotor allowance of 0 / 10 / 20 / 60 °C: 4140 14.407 / 11.946 / 9.484 / 0 N·m
      (the hot end binds); 416 14.0443 at 0, 10, 20 and 21 °C and 6.625 at 60 (the cold end binds at the default);
      17-4PH H900 14.888 at 0 and 10, 12.845 at 20; 304 4.975 / 1.741 / 0 / 0; 6061 0 throughout;
    - MF-3: the aluminium cup (E 68.9 GPa, yield 241 MPa) needs 155.6 °C against the 300 °C default;
    - MF-4: the wheel shaft shrinks 18.988 µm at -196 °C, 10.354 at -78.5, 0 at 20 and -2.300 at 40; the gearbox
      shaft 4.553 µm at -20 °C; the rationales at their slider ends (friction 0.05 / 0.14 / 0.30: 3.387 / 9.484 /
      20.32 N·m; smoothing 0 / 15: 19.49 / 0 N·m and 331.4 / 62.4 MPa; allowance 0 / 60: 14.41 / 0 N·m; clearance
      0 / 30: 11.97 / 255.9 °C);
    - the defaults, the five probe designs and the section 3.6 designs again, each as round 1 has it, and the
      direction steps of the should-fix items.
- Rulings of the note's physics review handed to this review: none. Both rounds of the note's physics review (Task 7)
  returned CONFIRM on R-1 to R-4 and R-14 and no CHANGE (`.superpowers/sdd/2026-10-17-bl064-shaft-joint/t7-review-r1.md`
  and `t7-review-r2.md`, item 5). Its round 2 observed that the wheel shaft's help still says "the inner hub needs less
  heat"; this review judged that help against trap 14: PASS, with SF-4 offered.
- Round 1: 4 must-fix items (ten strings), text applied verbatim (constants first, tests through the
  constants):
  - MF-1, `shaft_joint.safety_factor`, input help (its `param_rust_only` row).
    - Old: "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. The torque
      reverses many times in each slip."
    - New: "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. A slip can
      reverse the torque many times."
    - Why: the swing runs at `metal.slip_freq_Hz` for `metal.slip_event_s`: 16.7 cycles, about 33 reversals, per
      event at the defaults (trap 4), but 0.833 cycles (1.67 reversals) at the slip speed's slider bottom, 100 rpm,
      and 0.033 cycles at 100 rpm with 4 magnets and a 0.01 s event, where the torque never reverses. "each slip ...
      many times" is false there; "can" is true.
  - MF-2, `shaft_joint.hub_holding_Nm` and `shaft_joint.cup_holding_Nm`, result helps (their `out_rust_only` rows).
    - Old (inner hub): "Torque the inner hub's fit holds without slipping at its loosest: minimum measured
      interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft."
    - New (inner hub): "Torque the inner hub's fit holds without slipping at its loosest: minimum measured
      interference less smoothing, cold, or hot with the rotor warmer than its shaft, whichever loosens it more."
    - Old (cup): "Torque the cup's fit holds without slipping at its loosest: minimum measured interference less
      smoothing, at the temperature end that loosens it more, rotor warmer than shaft."
    - New (cup): "Torque the cup's fit holds without slipping at its loosest: minimum measured interference less
      smoothing, cold, or hot with the rotor warmer than its shaft, whichever loosens it more."
    - Why: the loosest end is `py_min(mismatch(cold), mismatch(hot) − α·ΔT_rotor·d)`, and the allowance enters only
      the hot end. With 416 (back-iron pick 4) the cold end binds: the inner hub holds 14.0443 N·m at a rotor
      allowance of 0, 10 and 20 °C alike; with 17-4PH H900 (pick 6) it binds up to 10 °C (14.8876 N·m). There the
      old help said the allowance was in the number, and moving the allowance changes nothing.
  - MF-3, `shaft_joint.cup_fit_max_C`, input help (its `param_rust_only` row).
    - Old: "The cup is fitted before its magnets are bonded. Stay below the steel's and its finish's treatment
      temperatures, and keep the heat off the gearbox seal."
    - New: "The cup is fitted before its magnets are bonded. Stay below the treatment temperatures of its material and
      finish, and keep the heat off the gearbox seal."
    - Why: the cup is the part material in effect. At back-iron pick 8 (6061), and with `coupling.backiron` 0 at any
      pick, that is the library's 6061-T6 (E 68.9 GPa, α 23.6e-6 /°C, yield 241 MPa), reachable from the selectors.
      There the old help set the bare cup's limit from a steel's treatment temperatures (the cup needs 155.6 °C
      against the 300 °C default).
  - MF-4 (a), `shaft_joint.wheel_shaft_shrink_um`, result help (its `out_rust_only` row).
    - Old: "How much the wheel shaft's diameter shrinks from 20 °C to its fitting temperature, from published
      low-temperature data for a nickel steel; negative above 20 °C, where it grows."
    - New: "How much the wheel shaft's diameter shrinks from room temperature to its fitting temperature, from
      published low-temperature data for a nickel steel; negative above room temperature, where it grows."
  - MF-4 (b), `shaft_joint.output_shaft_shrink_um`, result help (its `out_rust_only` row).
    - Old: "How much the gearbox shaft's diameter shrinks from 20 °C to its fitting temperature, from published
      low-temperature data for a nickel steel; negative above 20 °C, where it grows."
    - New: "How much the gearbox shaft's diameter shrinks from room temperature to its fitting temperature, from
      published low-temperature data for a nickel steel; negative above room temperature, where it grows."
  - MF-4 (c), the `fit_friction` rationale (its `ASSUMPTIONS` row).
    - Old: "The friction between each rotor's bore and its shaft that turns the fit's grip into holding torque. 0.14 is
      for dry, degreased steel on steel; oil, a coating or a finish in the bore lowers it, and the holding torque
      falls in proportion."
    - New: "The friction between each rotor's bore and its shaft that turns the fit's grip into holding torque. The
      default is for dry, degreased steel on steel; oil, a coating or a finish in the bore lowers it, and the holding
      torque falls in proportion."
  - MF-4 (d), the `fit_smoothing` rationale (its `ASSUMPTIONS` row).
    - Old: "Interference lost as the surface peaks of bore and shaft flatten when the fit is made, taken off both ends
      of the measured band. 5 µm is conservative for holding with ground and honed surfaces; a finer finish loses
      less, which raises the grip and the bore stress together."
    - New: "Interference lost as the surface peaks of bore and shaft flatten when the fit is made, taken off both ends
      of the measured band. The default is conservative for holding with ground and honed surfaces; a finer finish
      loses less, which raises the grip and the bore stress together."
  - MF-4 (e), the `rotor_over_shaft` rationale (its `ASSUMPTIONS` row).
    - Old: "Slip heating warms the rotors from the gap, so each rotor can run warmer than the shaft it sits on. Its bore
      then grows more than the shaft, and the grip at the hot end falls. 20 °C is a placeholder until a measurement
      or a thermal model gives the difference."
    - New: "Slip heating warms the rotors from the gap, so each rotor can run warmer than the shaft it sits on. Its bore
      then grows more than the shaft, and the grip at the hot end falls. The default is a placeholder until a
      measurement or a thermal model gives the difference."
  - MF-4 (f), the `joining_clearance` rationale (its `ASSUMPTIONS` row).
    - Old: "The gap each part needs over its shaft, after warming and cooling, to slide to its stop before the gap
      closes; one value for both joints. Too little and the part can seize half-way. 10 µm is an assumed value for a
      10 mm bore; less needs a guided, single fast push to a stop."
    - New: "The gap each part needs over its shaft, after warming and cooling, to slide to its stop before the gap
      closes; one value for both joints. Too little and the part can seize half-way. The default is an assumed value
      for the example design's bore; less needs a guided, single fast push to a stop."
  - Why (MF-4, a to f): the brief allows a typed number only for the bath temperatures, the DIN 6885-1 row and the
    cure margin; the physics of each text holds. "20 °C" is `ROOM_C`, the engine's own name for the reference (the
    measured interference's temperature, and where a shaft's shrink is zero). A typed default goes stale when the
    default moves, and the value shows on the input's own row. The reviewer offered the controller the option of
    ruling `ROOM_C` an allowed number and keeping "20 °C"; the controller did not take it.

  Should-fix offered, not applied (open for the controller), as round 1 returned them (round 2 re-offered each):
  - SF-1: name the inner ring in the bonded inner hub's two limit helps: `shaft_joint.hub_fit_max_C` "Its magnets are
    bonded before fitting. The check caps it at the adhesive's design limit and the inner ring's single-ring onset
    less 10 °C; bond strain is not checked." and `shaft_joint.hub_fit_allowed_C` "The lowest of your limit for the
    bonded inner hub, the selected adhesive's design limit and the inner ring's single-ring onset less 10 °C; not a
    number if any is." Why: with mixed rings (inner B842SH, outer B842) the results table shows two onsets, the
    governing 69.83 °C and "Inner ring: single-ring onset" 143.36 °C, and the cap reads the second; trap 20's bond
    strain matters at this input.
  - SF-2: the inner onset's help, "The temperature at which the inner ring alone on its carrier starts to
    demagnetize, whichever ring governs. The bonded hub's allowed fitting temperature stays at least 10 °C below
    it." Why: the old help never says what an onset is, and "stays below it" hides the margin (the Task 1
    re-review's SF-1).
  - SF-3: `shaft_joint.stress_limit_MPa`'s help, "The part's yield strength with a margin. The bore stress at the
    tightest fit must stay within it, so the part stays elastic; so must the key backup's pressure." Why: the key's
    badge quotes this row ("509.8 MPa against 345.8 MPa" at a 0.3 mm keyway), but the help names only the bore.
  - SF-4: `shaft_joint.wheel_shaft_fit_C`'s help, "A cooled shaft shrinks, so the temperature the inner hub's fit
    would need falls. Dry ice is about -78 °C; liquid nitrogen is -196 °C." Why: the note's reviewed wording, and no
    "heat" (trap 14); the old text is not false.
  - SF-5: the Problem's empty-window clause, "at its allowed temperature its bore does not even open the
    {clearance} joining clearance on the {side.shaft} as set" for "the allowed temperatures do not even open its
    {clearance} joining clearance". Why: each joint has one allowed temperature.
  - SF-6: `shaft_joint.friction`'s help, "The default is for steel on steel, dry and degreased. ..." for "Steel on
    steel, dry and degreased. ..." Why: a 304 or aluminium part (pick 7 or 8, or no back iron) is not steel on
    steel.
  - SF-7 to SF-10, the four C1 overlay helps (`gui/plain_text.rs`, `HELP`), each with its direction pins:
    - the bore: "Diameter of both shafts in the inner hub's and cup's bores. Larger thins the walls: the hub check
      can fail, and the wall past the keyway can break through.";
    - the keyway depth: "Depth of the key slot in the inner hub's and the cup's bores; it caps the key's contact
      height. Deeper thins the wall past the key; too deep breaks through." (a new claim design, "shallow keyway");
    - the boss length: "Axial length of the shaft boss behind the rear wall: longer adds overall length and boss
      mass, not large-diameter length, and lets the cup's fit hold more.";
    - the single-ring reverse field: "Reverse field on a single ring; higher lowers the cure margin and can lower
      the hub's fitting limit." with the stored-3D caveat (a new claim design, "N42 parts", where the onset binds).

  Recorded, no change asked: the guidance sheet prints the status line with the clamp section shown ("fails 4 of
  10"); with it paused the line reads "fails 4 of 9". No item needs a change to the engine's model.
- Round 2: PASS, no must-fix. It re-reviewed the changed strings
  and every string whose meaning depends on them; round 1's findings on the unchanged strings
  stand. It re-offered SF-1 to SF-10, two of them reworded: SF-6 to the new rationale's own clause ("The default is
  for dry, degreased steel on steel. Holding torque scales with it; oil or a coating in the bore lowers it.") and
  SF-10 to name the inner hub ("Single-ring reverse field; higher lowers the cure margin and can lower the inner hub's
  fitting limit."). It offered two more:
  - SF-11: `shaft_joint.safety_factor`'s help, "The fit alone must hold cold-high times this, and the key alone is
    checked at the same torque. While the coupling slips, the torque can reverse many times." Why: in this group
    "slip" is the fit's word ("The fit slips"; "holds without slipping"), and the house names the coupling's slip
    in this form.
  - SF-12: `shaft_joint.rotor_over_shaft_C`'s help, "How much hotter each rotor may run than its shaft from slip
    heating. A warmer rotor's bore grows, so the grip at the hot end falls." Why: MF-2's fact seen from the input;
    with 416 the holding rows do not move with it (14.0443 / 22.530 N·m at 0, 20 and 21 °C).

  Recorded, no change asked: O-1, with adhesive code 0 (a NaN limit) the holding rows read the cold end alone
  (16.97 / 26.28 N·m), since `py_min` drops the NaN hot end. The check reads unknown and the Problem says it cannot
  be made, so nothing is judged on those numbers, and no slider or typed value the spec names reaches code 0 (the
  selector offers 1 to 4 and `validate` rejects 0). It concerns the engine's NaN handling, outside this review.
- What happened to the should-fix items: SF-1 to SF-12 are left open, each. The plan's rule keeps should-fix items
  offered unless the controller says so, and the controller adopted none; they go to Task 10's triage. Of the
  should-fix items the earlier text re-reviews below left for this review:
  - Task 2's SF-1 (the bare cup's "steel's") and SF-8 (the holding helps' warmer rotor at both ends) are fixed by
    MF-3 and MF-2;
  - Task 3's SF-6 (a test that ties the rationales' typed defaults to the inputs) has nothing left to tie: MF-4
    removed the typed defaults, and `the_shaft_joint_rationales_say_the_default_and_type_no_number` keeps them out;
  - Task 1's SF-1, Task 2's SF-2 and Task 2's SF-6 come back as SF-2, SF-1 and SF-3, open;
  - the others on on-screen strings concern strings this review passes in its tables, and stay open where they
    were left.
- Tests that followed the changes: none changed. No test pinned an old text by literal (`git grep -n -F` of each old
  text over `magcoupling-rs/src`, `magcoupling-rs/tests`, the README and `docs/ai` at 967c0ec found only the sources
  and the input schema); every other test reads the strings through their metadata. Five tests were added, each red on
  the old text and green on the new:
  - `tests/shaft_joint.rs`:
    - `the_safety_factor_help_says_a_slip_can_reverse_the_torque_not_that_each_does` (MF-1: 16.6667 cycles per event
      at the defaults, 0.83333 at `metal.slip_rpm` 100, its slider's bottom; no "each slip" in the help);
    - `the_holding_helps_put_the_rotor_allowance_at_the_hot_end_alone` (MF-2: 4140 holds 14.41 N·m on the inner hub
      with no allowance and less at 20 °C; 416 holds the same at 0 and 20 °C, bit for bit, on both joints, 14.0443 N·m
      on the inner hub, and less at the slider's top);
    - `the_bare_cup_s_limit_help_names_no_steel_as_the_cup_can_be_aluminium` (MF-3);
    - `the_shaft_shrink_helps_count_from_room_temperature_and_type_no_number` (MF-4 a, b);
  - `tests/assumptions.rs`: `the_shaft_joint_rationales_say_the_default_and_type_no_number` (MF-4 c to f; the four
    rows are one table, `BL064_ROWS`, shared with `the_shaft_joint_rows_are_rust_only_and_cite_the_bl064_spec`).

  `tests/data/input_schema.json`: re-blessed for the changed input helps, two lines (`shaft_joint.safety_factor`
  and `shaft_joint.cup_fit_max_C`); the result helps and the rationales are not in it. The README's
  `tests/shaft_joint.rs` and `tests/assumptions.rs` rows name the new pins.
- Traps: none of the spec's 22 is written in any string. Round 1 found trap 4 once, in the safety factor's help
  ("The torque reverses many times in each slip", fixed by MF-1), and three texts false or outside the brief's
  typed-number rule that no trap names (MF-2, MF-3 and MF-4); round 2 found none.
- Verdict: PASS (round 2).

### Report for the record, round 1 (as returned)

Verdict: MUST_FIX. Four must-fix items, MF-1 to MF-4. Ten should-fix items, SF-1 to SF-10; the four C1 overlay helps are SF-7 to SF-10. No item needs a change to the engine's model. Every replacement below is a text change within its caps.

**Report for the record (as returned).**

1. Reviewer: physics reviewer (session model), fresh context, 2026-10-19; round 1.

2. Read:
- **The brief and its binding papers:**
  - `task-8-inreview-brief-r1.md`;
  - `rulings.md` (B-1, B-2);
  - `owner-decisions.md` (1A-8A, Process, Hub temp, Cup temp, Shaft temps, R-15, R-15 ring);
  - the plan head to "## Tasks" (Global Constraints, Review Focus, C-1 to C-9, PD-1 to PD-3, plan review 1A-4A);
  - the spec's sections 1 to 6 and 11.
- **Code:**
  - `engine/shaft_joint.rs`, all of it: inputs, results, constants, `hub_fit_allowed`, `shaft_shrink_um`, `joint`, `joint_specs`, `failing`, `verdict`, `compute`, `JointTerms::text`, `quoted_above`, `quoted_below`;
  - `engine/api.rs` (`shaft_joint_links`, the `shaft_joint::compute` call);
  - `engine/temperature.rs`: the demagnetization results block, with `inner_onset_single_ring_C` and its value `inner.onsets[3]`;
  - `engine/assumptions.rs`: all rows, the four new ones included;
  - `gui/guidance.rs`: `SHAFT_JOINT_GUIDE`, `JOINT_SIDES`, `shaft_joints`, `shaft_joint_fixes`, `reverses_the_band`, `shaft_joint_problem`, `term_sentence`, `fitting_source`, `onset_texts`, `key_sentence`, `window_sentence`, `window_quotes`, `not_judged_sentences`, `apart_from`, `design_status`, and the test designs `breaking`, `n42_parts`, `cool_adhesive`, `shaft_joint_problem_designs`;
  - `gui/dashboard.rs`: `DASHBOARD`, `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `STORED_3D_NOTE`, `verdict_level`, `result_tooltip`;
  - `gui/inputs.rs`: `SECTION_LABELS` and the "shaft" workflow group;
  - `gui/result_groups.rs`: `CHAIN_LABELS`, `PACKAGE_LABELS`;
  - `gui/plain_text.rs`: the four C1 rows, the direction table and its claim designs, and the bore-help test that requires "hub check";
  - `gui/format.rs`: a non-finite number reads "+inf";
  - `gui/test_support.rs`: `passing_design`, `only_the_joint_fails`.
- **Sheets in `magcoupling-rs/target/g3/`:**
  - `BL064-inputs.txt`, `BL064-guidance.txt` and `BL064-overlay.txt`;
  - `BL064-note.txt`, for consistency;
  - `BL064-strings.diff`: the assumptions, temperature and dashboard hunks.

  `BL064-guidance.txt` prints the status line with the clamp section shown ("fails 4 of 10"). While the clamp is paused it reads "fails 4 of 9" (probe below).
- **Earlier reviews:**
  - the T7 physics reviews `t7-review-r1.md` and `t7-review-r2.md`, the items on R-1 and R-14;
  - this record's BL-064 section: the T1 text re-review, whose SF-1 is taken up as SF-2 here.
- **The probe crate.** It is outside the repo: `<scratchpad>/bl064g3`, built with its own `CARGO_TARGET_DIR`, `magcoupling-rs` by path, feature gui. Each design went through `compute_all`, the shaft joints guide's Problem, `guidance::fixes` and `guidance::level`. The results:
  - **Defaults:**
    - "Both shaft joints hold and can be fitted", no Problem; status "Example design: fails 4 of 9 checks" with the clamp paused;
    - required 7.6466 N·m (2 × cold-high 3.8233); E 205 GPa, α 1.23e-5, yield 415 MPa, limit 345.83 MPa;
    - holding 9.4844 / 16.899 N·m; bore stress 241.70 / 312.53 MPa; key 127.44 / 101.95 MPa;
    - needed 12.082 / 11.310 µm; elastic 23.808 / 23.863 µm; shrink 18.988 / 0 µm; fittable 18.828 / 24.440 µm;
    - fitting 93.267 / 280.16 °C against 100 / 300 °C;
    - the inner ring's onset and the governing onset are both 143.356 °C.
  - **The five probes:**
    - defaults: as above;
    - 8 magnets: needs 5.598 N·m; passes;
    - 304: "The fit slips (inner hub): holds 0.00 N·m against 4.83 N·m". Both empty-window sentences show: 16.31 against 11.72 µm and 15.80 against 11.74 µm;
    - Y30: inner onset +inf; needs 0.676 N·m; passes;
    - 20 mm ring: hub holds 11.77 N·m against 7.22 N·m; passes.
  - **The spec's section 3.6 designs:**
    - hub minimum 10: 3.48 against 7.65 N·m; window 12.09 to 18.82 µm. Cup minimum 10: 4.36 against 7.65 N·m;
    - hub maximum 26 / 28: 385.1 / 421.0 MPa, with fitting 158.3 / 174.6 °C. Cup maximum 26 / 28: 384.0 / 419.8 MPa, with fitting 312.7 / 328.9 °C;
    - friction 0.10: 6.77 N·m. Smoothing 0: the cup reads 401.9 MPa, window 6.310 to 18.86 µm;
    - wheel shaft 20 °C: needs 247.6 °C; the joining-clearance sentence shows. At -78.5 °C: 163.5 °C, fittable 10.19 µm. At 40 °C: 266.3 °C, shrink -2.300 µm;
    - gearbox shaft -20 °C: 243.1 °C, shrink 4.553 µm. At 40 °C: 298.9 °C;
    - bare cup allowed 250 °C: 280.2 against 250.0 °C;
    - DP460: needs 93.3 against 60.0 °C, "the selected adhesive's design limit". Adhesive code 0: unknown, allowed NaN;
    - 2214 Hi-Temp, hub input typed 150 °C, hub maximum 26 µm: "above the 133.4 °C allowed: the magnets' single-ring onset, 143.4 °C, less 10.0 °C";
    - single-ring onset caps: N50M on manual dimensions 88.80 °C (onset 98.80); inner N42M with outer N35 81.35 °C (inner 91.35, governing 59.93, outer governs); B842 in both rings 59.83 °C (onset 69.83);
    - inner B842SH with outer B842: passes, allowed 100 °C (inner onset 143.36, governing 69.83);
    - keyway 0.3: "Key backup above its limit (inner hub): 509.8 MPa against 345.8 MPa"; the cup reads 407.8 MPa. Keyway 0: passes, holding 10.49 / 18.68 N·m, key 0 / 0;
    - boss length 0: the cup slips at 2.73 N·m, key +inf, "its boss and web, 0 mm and 2.500 mm long". Hub length 3.0 mm: 2.19 N·m, key +inf;
    - bore 16, boss 12: the inner hub slips at 4.57 N·m and the cup's no-wall sentence shows. With the hub minimum at 15: "Shaft joint check unknown: no wall around the bore (cup)";
    - bore 16, keyway 2.1: no wall (inner hub). Typed bore 21: the "reaches the flats" sentence shows. Boss Ø12 at bore 10: "cuts through the boss's wall";
    - hub band 20 over 18: unknown (reversed). Add the cup minimum at 10: the cup's slip leads and the reversed-band sentence follows;
    - typed bore 0: not a number. Key width 40: 0 N·m on both joints, both no-length key sentences, no window sentence. Contact height -0.5 mm: "its contact height, -0.5000 mm, is not above zero";
    - safety factor 0 and -1: "Shaft joint check unknown: the safety factor is not above zero", Caution. Problems "The safety factor, 0, is not above zero: enter at least 1." and "... -1.000 ...", no fix;
    - back-iron pick 99: not a number. `coupling.backiron` 0 and pick 8 (6061): E 68.9 GPa, α 23.6e-6, yield 241 MPa; both slip; the cup needs 155.6 °C against 300 °C;
    - axial override 40: the cup slips at 16.90 against 25.99 N·m, and its key reads 346.6 MPa;
    - c_end 1.5: cold-high -0.1726 N·m, required -0.3451 N·m; "passes", greyed. The short magnets (c_end 0.5, both rings on manual dimensions, 2 mm magnets): "The key has no bearing area (inner hub)", greyed;
    - BL-067 corner: cold-high 3.1823 N·m, required 6.3647 N·m, hot-low 2.9922 N·m. The operating pull-out with its allowance is 2.9922 / 0.9 × 1.1 = 3.657 N·m. The design passes;
    - single-ring field 1500 kA/m: inner onset -2.45 °C, "needs 93.3 °C against -12.4 °C", fittable 4.996 µm. At 862 kA/m: "93.3 °C against 93.2 °C".
  - **Every shaft-joint input at both slider ends,** plus the bore, keyway, boss length, hub length and single-ring field, at the defaults and at the passing design. Among the results:
    - friction 0.05 / 0.30: hub 3.387 / 20.32 N·m, proportional to 9.484 × μ / 0.14;
    - smoothing 15: 0 / 0 N·m, stress 62.4 / 133.7 MPa;
    - rotor over shaft 0 / 60: 14.41 / 0 N·m on the hub;
    - joining clearance 0 / 30: hub 11.97 / 255.9 °C;
    - hub limit 20 °C: fittable 8.988 µm;
    - bore 4: hub 585.7 MPa, cup 763.2 MPa, fit 434.7 / 670.4 °C;
    - key width 6: passes at 8.48 / 15.11 N·m. Key contact height 0.5: passes at 305.9 / 244.7 MPa.
  - **Which temperature end binds:**
    - 416 (pick 4, α 10.08e-6): the hub holds 14.0443 N·m at rotor-over-shaft 0, 10 and 20 °C; the cold end binds;
    - 17-4PH H900 (pick 6): 12.8449 N·m at 20 °C; 14.8876 at 0 and at 10 °C.
  - **Slip reversals per event:**
    - defaults: 166.67 Hz × 0.1 s = 16.7 cycles (33.3 reversals);
    - slip speed 100 rpm (its slider's bottom): 8.333 Hz, 0.833 cycles (1.67 reversals);
    - 100 rpm, 4 magnets, 0.01 s: 0.033 cycles.
  - **Overlay direction steps:**
    - single-ring field +1 kA/m. Example: cure margin 121.356 -> 121.227 °C, inner onset 143.356 -> 143.227 °C, hub allowed 100 -> 100 °C. B842 in both rings: allowed 59.8326 -> 59.6334 °C;
    - boss length +0.5 mm: cup holding 16.899 -> 17.444 N·m; axial stack up; large-diameter stack the same;
    - keyway +0.05 mm. Example: wall past the key 3.40 -> 3.35 mm; key pressures the same. Keyway 1.0 mm: hub key 152.93 -> 145.65 MPa;
    - bore +0.1 mm: hub wall 5.10 -> 5.05 mm; wall past the key 3.40 -> 3.35 mm.

3. Strings:

| # | String (field) | Verdict | Reason |
|---|---|---|---|
| 1.1 | `hub_interference_min_um` label, help | PASS | Names measured (trap 5). Holding is computed from it, as the help says (9.484 N·m from 13 µm). |
| 1.2 | `hub_interference_max_um` label, help | PASS | Stress and fitting are computed at it (241.7 MPa and 93.27 °C at 18 µm; 636.2 MPa and 272.1 °C at 40 µm). |
| 1.3 | `cup_interference_min_um` label, help | PASS | As 1.1, for the cup (16.90 N·m). |
| 1.4 | `cup_interference_max_um` label, help | PASS | As 1.2 (312.5 MPa, 280.2 °C). |
| 1.5 | `safety_factor` label, help | MUST-FIX (MF-1) | "The torque reverses many times in each slip" is false at reachable inputs: 1.67 reversals per event at 100 rpm, 0.07 at 100 rpm with 4 magnets and 0.01 s. |
| 1.6 | `friction` label, help | PASS | Holding is proportional to friction (0.05 / 0.14 / 0.30 give 3.387 / 9.484 / 20.32 N·m). SF-6 says it is the default's basis. |
| 1.7 | `smoothing_um` label, help | PASS | Both grip and stress fall with it (trap 19). At 0: 19.49 / 29.44 N·m and 331.4 / 401.9 MPa. At 15: 0 / 0 N·m and 62.4 / 133.7 MPa. |
| 1.8 | `rotor_over_shaft_C` label, help | PASS | The hot end loses grip as it rises: 14.41 N·m at 0, 9.484 at 20, 0 at 60. |
| 1.9 | `key_width_mm` label, help | PASS | DIN 6885-1, 3 × 3 for Ø10 (trap 17). Without a keyway no slot is taken: 10.49 N·m. |
| 1.10 | `key_contact_mm` label, help | PASS | The DIN 3 × 3 row gives h − t1 = 3 − 1.8 = 1.2 mm. The cap at the keyway depth is real: at a 0.3 mm keyway the key bears 509.8 MPa. |
| 1.11 | `hub_fit_max_C` label, help | PASS | Bonded before fitting (trap 15); the cap is the engine's rule (150 °C typed with 2214 Hi-Temp gives 133.36 °C). SF-1 names the inner ring and trap 20's bond strain. |
| 1.12 | `cup_fit_max_C` label, help | MUST-FIX (MF-3) | "the steel's ... treatment temperatures" names a steel the part does not have where it is aluminium (pick 8 or `coupling.backiron` 0: E 68.9 GPa, yield 241 MPa). |
| 1.13 | `wheel_shaft_fit_C` label, help | PASS | The bath temperatures are stated as facts (trap 12). "needs less heat" does not break trap 14, since it says cool the shaft, not heat the hub. SF-4 aligns it with the note. |
| 1.14 | `output_shaft_fit_C` label, help | PASS | "never below the gearbox's rated range" (trap 11); -20 °C is the slider's bottom (243.1 °C, shrink 4.553 µm). |
| 1.15 | `joining_clearance_um` label, help | PASS | At 0 the hub needs 11.97 °C; at 30 neither part can be fitted (255.9 and 442.8 °C). True. |
| 2.1 | `required_Nm` label, help | PASS | Claims no order of torques (trap 2). 2 × 3.8233; at the BL-067 corner 2 × 3.1823. |
| 2.2 | `part_modulus_GPa` label, help | PASS | 205 / 193 (304) / 68.9 (6061, no back iron). Contact pressure rises with E. |
| 2.3 | `part_cte_per_C` label, help | PASS | Trap 7, both ways: 416 binds cold, 304 hot. |
| 2.4 | `part_yield_MPa` label, help | PASS | 415 (the library's annealed 4140), 220 (1018), 241 (6061). |
| 2.5 | `stress_limit_MPa` label, help | PASS | yield / 1.2 = 345.83 MPa. True, but it does not say the key is held to it too (SF-3). |
| 2.6-2.7 | `hub_holding_Nm`, `cup_holding_Nm` labels, helps | MUST-FIX (MF-2) | "at the temperature end that loosens it more, rotor warmer than shaft" says the allowance is applied where the cold end binds. It is not: 416 reads 14.0443 N·m at 0, 10 and 20 °C. |
| 2.8-2.9 | bore stress labels, helps | PASS | 241.7 / 312.5 MPa. The tightest end carries no allowance, as written. "The keyway corner is not included" covers trap 20. |
| 2.10-2.11 | key pressure labels, helps | PASS | 127.4 / 102.0 MPa. 0 with keyway 0; "+inf" is the readout's own text (key width 40, contact -0.5). Trap 1 holds. |
| 2.12-2.13 | needed labels, helps | PASS | 12.08 / 11.31 µm, measured, the window's lower end. NaN with no wall; +inf at key width 40. |
| 2.14-2.15 | elastic labels, helps | PASS | 23.81 / 23.86 µm. |
| 2.16-2.17 | shrink labels, helps | MUST-FIX (MF-4) | The physics is true (18.99 µm at -196 °C, -2.30 at 40 °C, 4.553 at -20 °C). "20 °C" is typed twice, and the brief's list does not allow it. |
| 2.18-2.19 | fittable labels, helps | PASS | 18.83 / 24.44 µm. 4.996 µm at a 1500 kA/m field (allowed -12.45 °C). |
| 2.20-2.21 | fit-temperature labels, helps | PASS | "would need" and "never warm it further" (trap 9). Bore 16 needs 7.90 °C, below room: the help still holds. |
| 2.22 | `hub_fit_allowed_C` label, help | PASS | It is the minimum of the three at every probe: 100, 120 (input 130), 59.83, 88.80, 81.35, 133.36, NaN, -12.45 °C. SF-1 names the inner ring. |
| 2.23 | `cup_fit_allowed_C` label, help | PASS | Trap 15. |
| 2.24 | `check` label "Shaft joints: keyed shrink fits", help | PASS | 30 words. The four asks in order, the key as a backup. |
| 3.1 | `SHAFT_JOINTS_PASS` | PASS | Shown at the defaults, the passing design, the five probes and the BL-067 corner. Greyed at c_end 1.5. |
| 3.2 | `SHAFT_JOINT_UNKNOWN` (prefix) | PASS | Caution for every unknown form. |
| 3.3 | `UNKNOWN_NOT_A_NUMBER` | PASS | Adhesive code 0, bore 0 (the holding works out NaN), pick 99. |
| 3.4 | `UNKNOWN_NO_WALL` | PASS | Cup at bore 16 with boss Ø12 and hub minimum 15; inner hub at bore 16 with keyway 2.1. |
| 3.5 | `UNKNOWN_BAND_REVERSED` | PASS | Hub at 20 over 18, and minimum 40 or maximum 0 at the slider ends. |
| 3.6 | `UNKNOWN_SAFETY_FACTOR` | PASS | Safety factor 0 and -1. |
| 3.7 | `FIT_SLIPS` | PASS | "holds X against Y": below the torque it must hold, it slips. |
| 3.8 | `BORE_OVER_ITS_LIMIT` | PASS | 385.1, 421.0, 384.0, 419.8, 401.9 MPa. |
| 3.9 | `CANNOT_BE_FITTED` | PASS | Values never read equal: 93.3 against 93.2 °C at 862 kA/m. 247.6 against 100.0 °C; 93.3 against -12.4 °C. |
| 3.10 | `KEY_OVER_ITS_LIMIT` | PASS | 509.8 against 345.8 MPa. |
| 3.11 | `KEY_HAS_NO_BEARING_AREA` | PASS | Boss length 0, hub length 3, contact -0.5 and 0, key width 40. |
| 4 | badge tails; "inner hub", "cup" | PASS | "{q} ... against {r} ..."; the numbers never read equal. Badges are exempt from trap 9. |
| 5 | `inner_onset_single_ring_C` label "Inner ring: single-ring onset", help | PASS | Mixed rings (inner B842SH, outer B842): it reads 143.36 °C against the governing 69.83 °C, "whichever ring governs", and the allowed 100 °C is below it. Below it at every probe (+inf with Y30, -2.45 °C at 1500 kA/m). SF-2. |
| 6 | guide label "Shaft joints"; Why | PASS | The label is distinct and not "Shaft clamp". The Why is one sentence of 29 words with no ". " inside. |
| 7.1 | Problem, not a number | PASS | "a value it reads or works out": bore 0 works one out. |
| 7.2 | Problem, safety factor | PASS | "The safety factor, 0, is not above zero: enter at least 1." (also "-1.000"); no fix. |
| 7.3 | Problem, no wall: cup Q ≥ 1; inner hub Q ≥ 1 | PASS | Bore 16 with boss 12; bore 21 typed. |
| 7.4 | Problem, keyway through the wall (inner hub; cup's boss) | PASS | Bore 16 with keyway 2.1; bore 10 with boss 12. |
| 7.5 | Problem, reversed band | PASS | |
| 7.6 | Problem, slips | PASS | "cold-high, X, times the safety factor of Y" claims no bound (trap 2). |
| 7.7 | Problem, stress | PASS | "(415.0 MPa yield with a margin)". |
| 7.8 | Problem, fitting: three hub sources and the cup's | PASS | Trap 9 holds in each form. The onset's quoted figures subtract to the allowed temperature (69.83 − 10.00 = 59.83; 143.4 − 10.0 = 133.4). |
| 7.9 | Problem, key: above the limit, no length (hub, cup), no contact | PASS | Contact 0 with boss 0 shows the contact sentences alone, as the spec rules. |
| 7.10 | Problem, window | PASS | Measured, quoted inward. "would hold and {edge}" names the edge that binds. |
| 7.11 | Problem, empty window: stress and fitting edges | PASS | 304, 6061, hub length 3, -78.5 °C. |
| 7.12 | Problem, empty window: "the allowed temperatures do not even open its ... joining clearance" | PASS | True wherever it shows (opening ≤ clearance). The plural is loose: SF-5. |
| 7.13 | Problem fallback | PASS | Not reached; plain. |
| 8 | the four assumption labels and rationales | MUST-FIX (MF-4) | The labels pass. Each rationale's physics holds at the slider ends (friction proportional; smoothing 0 / 15 raises or lowers grip and stress together; allowance 0 / 60; clearance 0 / 30). Each types its default (0.14; 5 µm; 20 °C; 10 µm for a 10 mm bore), which the brief's list does not allow. |
| 9 | section labels "Shrink fit", "Fitting temperatures" | PASS | ASCII, no "clamp". |
| 10 | catalogue heading "Shaft joints" | PASS | Distinct from "Shaft clamps". |
| 11 | package heading "Shaft joints" | PASS | Consistent, and never drawn. |
| 12 | chain heading "Shaft joints" | PASS | Drawn between Geometry and Mass. |
| 13 | the new sentence of `STORED_3D_NOTE` | PASS | Only the single-ring field moves the joint row: "needs 93.3 °C against -12.4 °C" at 1500 kA/m. The sentence that names the slip-heating fields says no "limit"; "stay fixed" and "example design" are kept. |

4. Must-fix:

- **MF-1. `shaft_joint.safety_factor`, engine help (`engine/shaft_joint.rs:61`).**
  - Old: "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. The torque reverses many times in each slip."
  - New: "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. A slip can reverse the torque many times." (26 words, 2 sentences)
  - Why: the swing runs at `metal.slip_freq_Hz` for `metal.slip_event_s`. At the defaults that is 16.7 cycles, about 33 reversals (trap 4). At the slip speed's slider bottom, 100 rpm, it is 0.833 cycles (1.67 reversals) per 0.1 s event. At 100 rpm with 4 magnets and a 0.01 s event it is 0.033 cycles: the torque never reverses. "Each slip ... many times" is false there; "can" is true.
  - Test: `assert!(!help.contains("each slip"))`. Pin the reason: at `metal.slip_rpm` 100, `slip_freq_Hz * slip_event_s` is 0.8333 (< 1), and at the defaults it is 16.667.
- **MF-2. `shaft_joint.hub_holding_Nm` and `shaft_joint.cup_holding_Nm`, engine helps (`engine/shaft_joint.rs:116`, `:118`).**
  - Old (hub): "Torque the inner hub's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft."
  - New (hub): "Torque the inner hub's fit holds without slipping at its loosest: minimum measured interference less smoothing, cold, or hot with the rotor warmer than its shaft, whichever loosens it more." (30 words, 1 sentence)
  - Old (cup): "Torque the cup's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft."
  - New (cup): "Torque the cup's fit holds without slipping at its loosest: minimum measured interference less smoothing, cold, or hot with the rotor warmer than its shaft, whichever loosens it more." (29 words, 1 sentence)
  - Why: the loosest end is `py_min(mismatch(cold), mismatch(hot) − α·ΔT_rotor·d)`, and the allowance enters only the hot end. With the library's 416 (pick 4, α 10.08e-6, below the shaft's 11.5e-6) the cold end binds: the hub holds 14.0443 N·m at rotor-over-shaft 0, 10 and 20 °C alike. With 17-4PH H900 (pick 6) the cold end binds up to 10 °C (14.8876 N·m). There the old help tells the reader the allowance is in the number, and moving the allowance changes nothing. The new text keeps the trap-18 form "holds without slipping".
  - Test: at `materials.parts.back_iron` 4, `hub_holding_Nm` is bit for bit equal at `rotor_over_shaft_C` 0 and 20, and both helps contain "whichever loosens it more".
- **MF-3. `shaft_joint.cup_fit_max_C`, engine help (`engine/shaft_joint.rs:85`).**
  - Old: "The cup is fitted before its magnets are bonded. Stay below the steel's and its finish's treatment temperatures, and keep the heat off the gearbox seal."
  - New: "The cup is fitted before its magnets are bonded. Stay below the treatment temperatures of its material and finish, and keep the heat off the gearbox seal." (27 words, 2 sentences; "gearbox" without "protect"; no "clamp")
  - Why: the cup is the part material in effect. At back-iron pick 8 (6061), and at `coupling.backiron` 0 with any pick, that is the library's 6061-T6: E 68.9 GPa, α 23.6e-6, yield 241 MPa. Those designs are reachable from the selectors, and there the help tells the user to set the bare cup's limit from a steel's treatment temperatures. The cup needs 155.6 °C against the 300 °C default there. An aluminium cup's own treatment temperature, not a steel's, is what the user's limit must stay below.
  - Test: at `coupling.backiron` 0, `part_modulus_GPa` is 68.9 and the help does not contain "steel".
- **MF-4. Typed numbers the brief's list does not allow:** "20 °C" in the two shrink helps, and the defaults typed in the four assumption rationales. The physics of each text is true. Only the typed number breaks the rule: "the bath temperatures, the DIN 6885-1 row and the cure margin are; no other is".
  - **(a) `shaft_joint.wheel_shaft_shrink_um`, help (`engine/shaft_joint.rs:136`).**
    - Old: "How much the wheel shaft's diameter shrinks from 20 °C to its fitting temperature, from published low-temperature data for a nickel steel; negative above 20 °C, where it grows."
    - New: "How much the wheel shaft's diameter shrinks from room temperature to its fitting temperature, from published low-temperature data for a nickel steel; negative above room temperature, where it grows." (29 words, 1 sentence)
  - **(b) `shaft_joint.output_shaft_shrink_um`, help (`:138`).**
    - Old: as (a), with "gearbox".
    - New: "How much the gearbox shaft's diameter shrinks from room temperature to its fitting temperature, from published low-temperature data for a nickel steel; negative above room temperature, where it grows." (29 words, 1 sentence)
    - Why (a, b): `ROOM_C` is the engine's own name for the 20 °C reference ("the measured interference's temperature, and where a shaft's shrink is zero").
    - Option for the controller: rule `ROOM_C` allowed, like the cure margin, and keep "20 °C" with a test that builds it from `shaft_joint::ROOM_C`.
  - **(c) `fit_friction` rationale.** "0.14 is for dry, degreased steel on steel; ..." becomes "The default is for dry, degreased steel on steel; ...". Full New: "The friction between each rotor's bore and its shaft that turns the fit's grip into holding torque. The default is for dry, degreased steel on steel; oil, a coating or a finish in the bore lowers it, and the holding torque falls in proportion."
  - **(d) `fit_smoothing` rationale.** "5 µm is conservative ..." becomes "The default is conservative ...". Full New: "Interference lost as the surface peaks of bore and shaft flatten when the fit is made, taken off both ends of the measured band. The default is conservative for holding with ground and honed surfaces; a finer finish loses less, which raises the grip and the bore stress together."
  - **(e) `rotor_over_shaft` rationale.** "20 °C is a placeholder ..." becomes "The default is a placeholder ...". Full New: "Slip heating warms the rotors from the gap, so each rotor can run warmer than the shaft it sits on. Its bore then grows more than the shaft, and the grip at the hot end falls. The default is a placeholder until a measurement or a thermal model gives the difference."
  - **(f) `joining_clearance` rationale.** "10 µm is an assumed value for a 10 mm bore; ..." becomes "The default is an assumed value for the example design's bore; ...". Full New: "The gap each part needs over its shaft, after warming and cooling, to slide to its stop before the gap closes; one value for both joints. Too little and the part can seize half-way. The default is an assumed value for the example design's bore; less needs a guided, single fast push to a stop."
  - Why (c-f): a typed default goes stale when the default moves, and the brief allows none of them. Each rationale is still true word for word after the change. The value itself shows on the input's own row. The T10 record allowed "the assumption's own default set"; this brief's list is narrower, and it binds this round.
  - Test: no ASCII digit in the four BL-064 rationales (`tests/assumptions.rs`) or in the two shrink helps (`engine/shaft_joint.rs` tests).

5. Should-fix offered:

- **SF-1. Name the inner ring in the bonded hub's limit helps.**
  - **`shaft_joint.hub_fit_max_C`, help (`engine/shaft_joint.rs:82`).**
    - Old: "Its magnets are bonded before fitting. The check caps it at the adhesive's design limit and at the magnets' single-ring onset less 10 °C."
    - New: "Its magnets are bonded before fitting. The check caps it at the adhesive's design limit and the inner ring's single-ring onset less 10 °C; bond strain is not checked." (29 words, 2 sentences)
  - **`shaft_joint.hub_fit_allowed_C`, help (`:148`).**
    - Old: "... and the magnets' single-ring onset less 10 °C; not a number if any is."
    - New: "The lowest of your limit for the bonded inner hub, the selected adhesive's design limit and the inner ring's single-ring onset less 10 °C; not a number if any is." (30 words, 1 sentence)
  - Why: the R-15 ring ruling makes "the magnets'" true. With mixed rings, though, the results table shows two onsets:
    - the governing "Onset, single ring during an adhesive cure": 69.83 °C with inner B842SH and outer B842;
    - "Inner ring: single-ring onset": 143.36 °C.

    The allowed temperature reads 100 °C. "The inner ring's" names the row the cap reads. "Bond strain is not checked" is trap 20's item that matters at this input.
  - Both keep "onset less 10 °C" for the cure-margin test. No label changes.
  - If taken, the Problem's MagnetOnset form ("the magnets' single-ring onset, {onset}, less {margin}") may follow. Its tests pin those words (`guidance.rs:3375`, `:7672-7713`), and its live number already shows which onset it is, so leaving it is fine.
- **SF-2. `temperature.demag.inner_onset_single_ring_C`, help (`engine/temperature.rs:511`).**
  - Old: "The inner ring's own onset on its carrier with no outer ring around it, whichever ring governs. The bonded inner hub's allowed fitting temperature stays below it."
  - New: "The temperature at which the inner ring alone on its carrier starts to demagnetize, whichever ring governs. The bonded hub's allowed fitting temperature stays at least 10 °C below it." (30 words, 2 sentences)
  - Why: the help never says what "onset" means, and its last sentence hides the margin (the T1 re-review's SF-1). "starts to demagnetize" is the house's wording, from the temperature Problem's cure sentence. "at least" holds because the allowed temperature is a minimum.
  - Test: build "at least 10 °C below" from `CURE_MARGIN_MIN_C`, beside the two "onset less 10 °C" pins.
- **SF-3. `shaft_joint.stress_limit_MPa`, help (`engine/shaft_joint.rs:114`).**
  - Old: "The part's yield strength with a margin. At the tightest fit the bore stress must not exceed it, so the part stays elastic and keeps its grip."
  - New: "The part's yield strength with a margin. The bore stress at the tightest fit must stay within it, so the part stays elastic; so must the key backup's pressure." (29 words, 2 sentences)
  - Why: "Key backup above its limit (inner hub): 509.8 MPa against 345.8 MPa" (keyway 0.3) quotes this row, but the row's label and help speak only of the bore. The label is kept.
- **SF-4. `shaft_joint.wheel_shaft_fit_C`, help (`engine/shaft_joint.rs:88`).**
  - Old: "A cooled shaft shrinks, so the inner hub needs less heat. Dry ice is about -78 °C; liquid nitrogen is -196 °C."
  - New: "A cooled shaft shrinks, so the temperature the inner hub's fit would need falls. Dry ice is about -78 °C; liquid nitrogen is -196 °C." (25 words, 2 sentences)
  - Why: it matches the note's reviewed change entry ("so the temperature the inner hub's fit would need rises") and avoids "heat" (trap 14). Not false today: at bore 16 the fit needs only 7.90 °C, and the old "needs less heat" stays true there.
  - Direction: Down, pinned by the note's own aim (`hub_fit_temp_C`, Up for a warmer shaft).
- **SF-5. The Problem's empty-window clause (`gui/guidance.rs:2183`).**
  - Old: "the allowed temperatures do not even open its {clearance} joining clearance"
  - New: "at its allowed temperature its bore does not even open the {clearance} joining clearance on the {side.shaft} as set"
  - It renders "... holding needs at least 12.09 µm, but at its allowed temperature its bore does not even open the 10.00 µm joining clearance on the wheel shaft as set." (wheel shaft at 20 °C).
  - Why: each joint has one allowed temperature, and the other temperature in play is its shaft's set one, not an allowed one. The clause is true wherever it shows (hub opening 9.84 µm against 10 µm at 20 °C; cup allowed 20 °C opens 0). The test at `guidance.rs:7656` follows.
- **SF-6. `shaft_joint.friction`, help (`engine/shaft_joint.rs:64`).**
  - Old: "Steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it."
  - New: "The default is for steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it." (24 words, 2 sentences)
  - Why: with a 304 or aluminium part (pick 7 or 8, or no back iron) the contact is not steel on steel, and the fragment reads as a statement about the design.
- **SF-7 (C1). `coupling.bore_mm`, overlay help (`gui/plain_text.rs:263`).**
  - Old: "Diameter of the shaft in the hub bore. Larger thins the hub wall: the hub check can fail, and the wall past the keyway can break through."
  - New: "Diameter of both shafts in the inner hub's and cup's bores. Larger thins the walls: the hub check can fail, and the wall past the keyway can break through." (29 words, 2 sentences)
  - Pins, on "example": `model.hub_wall_mm` Down (5.10 -> 5.05 mm) and `model.hub_wall_past_key_mm` Down (3.40 -> 3.35 mm). The bore test's "hub check" stays in the text.
- **SF-8 (C1). `coupling.keyway_depth_mm`, overlay help (`:267`).**
  - Old: "Depth of the key slot cut into the hub bore. Deeper leaves a thinner hub wall past the key, and too deep breaks through the hub."
  - New: "Depth of the key slot in the inner hub's and the cup's bores; it caps the key's contact height. Deeper thins the wall past the key; too deep breaks through." (30 words, 2 sentences)
  - Pins:
    - "example": `model.hub_wall_past_key_mm` Down (3.40 -> 3.35 mm); `shaft_joint.hub_key_pressure_MPa` Stays (127.44 MPa: the 1.2 mm contact is under the 1.7 mm keyway);
    - a new claim design, "shallow keyway" (keyway 1.0 mm): `shaft_joint.hub_key_pressure_MPa` Down (152.93 -> 145.65 MPa).
- **SF-9 (C1). `metal.boss_length_mm`, overlay help (`:475`).**
  - Old: "Axial length of the shaft boss behind the rear wall: it adds to the overall length and the boss mass, not to the large-diameter length."
  - New: "Axial length of the shaft boss behind the rear wall: longer adds overall length and boss mass, not large-diameter length, and lets the cup's fit hold more." (27 words, 1 sentence)
  - Pins: the three existing rows, and `("metal.boss_length_mm", "example", "shaft_joint.cup_holding_Nm", Up)` (16.899 -> 17.444 N·m at +0.5 mm).
- **SF-10 (C1). `temperature.demag.h_rev_single_ring_kA_m`, overlay help (`:662`).**
  - Old: "Worst reverse field on a single ring during adhesive cure; higher lowers the cure margin." + the stored-3D caveat.
  - New: "Reverse field on a single ring; higher lowers the cure margin and can lower the hub's fitting limit." + the stored-3D caveat (18 + 12 = 30 words, 2 sentences). "Worst" and "single ring on its carrier" stay in the label.
  - Pins:
    - the two existing rows;
    - `(…, "example", "shaft_joint.hub_fit_allowed_C", Stays)` (100 -> 100 °C: the onset does not bind at the defaults);
    - `(…, "N42 parts", "shaft_joint.hub_fit_allowed_C", Down)`, with a new claim design "N42 parts": library B842 in both rings (59.8326 -> 59.6334 °C at +1 kA/m).

    "can" covers the claim at both designs.

6. Traps, each checked against every string above:
   1. **Fit and key torques never added.** Holds: "the key alone"; "were the key to carry the torque alone".
   2. **Cold-high.** No string says "strongest", "most expected" or "at most" of it. The Problems' "only up to" is about interference. At the BL-067 corner 3.1823 < 3.657 N·m, and no text orders them.
   3. **No ratio, efficiency or `cold_high_input_Nm`.** Holds.
   4. **The swing frequency.** No Hz quoted. The safety factor's "many times in each slip" is false off the defaults: MF-1.
   5. **Measured against effective.** Named in each help: holding effective, assembly measured maximum (no smoothing).
   6. **No fit class, no "tighter is safer".** Holds.
   7. **Temperature cuts both ways.** Stated in the expansion help, true for 416 and 304.
   8. **No speed term.** None quoted.
   9. **Sources named.** Every fitting temperature in the Problem and helps names its source; the badge is exempt. No "safe". The needed temperature is quoted against the allowed one. SF-1 sharpens "the magnets'".
   10. **No 150 °C.** Not quoted.
   11. **Gearbox.** "never below the gearbox's rated range", "keep the heat off the gearbox seal", no "protect", no catalog named.
   12. **Bath temperatures.** Stated as facts. No text calls -196 °C routine. The chill of the bonded hub is in the note's watch-out 2.
   13. **No induction heating.** None named.
   14. **No "heat the hub" or "heat until".** None. "needs less heat" is not an instruction: SF-4.
   15. **Fitting order.** Holds: bare cup first, then magnets; inner hub bonded first.
   16. **No "lower the safety factor to fit".** None.
   17. **The DIN 6885-1 row.** Named in the key width and contact helps.
   18. **"holds without slipping".** Holds; no "permanent" or "no backlash".
   19. **Friction and smoothing are assumptions.** Friction: "scales with it"; smoothing: "lowers both".
   20. **What nothing checks.** The keyway corner is in the stress helps. The bond strain is not stated anywhere on screen: SF-1. The rest is in the note.
   21. **No "clamp".** None in any new text, label, section or heading.
   22. **Glyphs.** ASCII "-" and Ø as specified; no U+2192, U+2300 or U+2212 in a string (U+2212 appears only in code comments). No cell ids, "BL-064" or "Rust-only" on screen.

   **Typed numbers:**
   - **Allowed:**
     - bath temperatures: -78 °C and -196 °C (wheel shaft); -20 °C (cold spray at the gearbox shaft's slider bottom, the owner's Shaft temps row);
     - the DIN 6885-1 row: 3 mm, 1.2 mm, 10 mm;
     - the cure margin: 10 °C in the two hub helps, and in SF-2 if taken.
   - **Allowed by ruling C-4:** the "1" of "enter at least 1". The brief prescribes the text and the owner confirmed it (1A).
   - **Names, not numbers:** "4140", "DIN 6885-1", "3D".
   - **Not allowed:** "20 °C" in both shrink helps; 0.14, 5 µm, 20 °C, 10 µm and 10 mm in the four rationales. All go in MF-4.
   - **Sources:** the "BL-064" in the assumption sources is outside this review (section 6.3), and they are not drawn as on-screen text.

7. The context lines:
   - **C2: the fix texts.** Fine: every fix is "Raise/Lower <label>", and each moves its term. "Lower Gearbox shaft temperature at fitting" stops at the slider's -20 °C, and the input's help forbids going below the rated range. "Lower Wheel shaft temperature at fitting" follows the owner's process (a cooled wheel shaft) and names no bath.
   - **C3: the status line.** "Example design: fails 4 of 9 checks" with the clamp paused (10 with it shown, as the sheet prints).
   - **C3: the row's hover.** It ends "Fixed 3D fields from the example design: <STORED_3D_NOTE>", and the new sentence reads right under the row's label.
   - **C3: Explain.** "Closest teaching note: Shaft joints: a keyed shrink fit" is right.
   - **C4: the note.** Consistent with every string. SF-4 would align the wheel-shaft help with the note's wording ("the temperature the inner hub's fit would need").

8. Verdict: MUST_FIX (MF-1 to MF-4). No item needs a model change.

### Report for the record, round 2 (as returned)

Verdict: PASS. No must-fix item remains. Round 1's MF-1 to MF-4 are applied verbatim. Each new text is true at every design probed, stays within its cap and is pinned by a test that passes. Twelve should-fix items are offered: round 1's SF-1 to SF-10 (SF-6 and SF-10 reworded; the four C1 overlay helps are SF-7 to SF-10) and two new ones, SF-11 and SF-12. No item needs a change to the engine's model.

**Report for the record (as returned).**

1. Reviewer: physics reviewer (session model), fresh context, 2026-10-19; round 2.

2. Read:
- **The brief and its binding papers:**
  - `task-8-inreview-brief-r2.md`;
  - `rulings.md` (B-1, B-2);
  - `owner-decisions.md` (1A-8A, Process, Hub temp, Cup temp, Shaft temps, R-15, R-15 ring);
  - the plan head to its Tasks heading (Global Constraints, Review Focus, C-1 to C-9, PD-1 to PD-3, plan review 1A-4A).
- **Round 1:** `t8-g3-r1.md` in full (its table, MF-1 to MF-4, SF-1 to SF-10, its probes).
- **The round's changes:** the working-tree diff against 967c0ec:
  - `engine/shaft_joint.rs`: six strings;
  - `engine/assumptions.rs`: four rationales;
  - `tests/shaft_joint.rs`: four new tests, two helpers;
  - `tests/assumptions.rs`: one new test, `BL064_ROWS`;
  - `tests/data/input_schema.json`: two lines;
  - `README.md`: two lines.
- **Code:**
  - `engine/shaft_joint.rs`, all of it: module doc, inputs, results, `ROOM_C`, `hub_fit_allowed`, `shaft_shrink_um`, `joint`, `joint_specs`, `failing`, `verdict`, `compute`, `JointTerms::text`, `quoted_above`, `quoted_below`;
  - `engine/api.rs` (`shaft_joint_links`);
  - `engine/explain/records/shaft_joint.rs` (the holding, needed and bore stress records);
  - `engine/explain/notes.rs` (the note `shaft_joint`, C4);
  - `engine/assumptions.rs`: every rationale;
  - `engine/temperature.rs`: the adhesive table, `NO_ADHESIVE`, `stored_3d_sentence`;
  - `engine/model.rs` and `engine/metal_design.rs`: the slip inputs and their helps;
  - `gui/guidance.rs`:
    - `SHAFT_JOINT_GUIDE`, `shaft_joint_problem`, `term_sentence`, `fitting_source`, `onset_texts`, `key_sentence`, `window_sentence`, `not_judged_sentences`;
    - `level`, `fixes`, `design_status_text`;
    - the tests at `:3415` and `:7656`;
  - `gui/dashboard.rs`: `END_EFFECT_ROWS`, `STORED_3D_ROWS`, `STORED_3D_LABEL`, `STORED_3D_NOTE`, an assumption's Why-link hover;
  - `gui/panel.rs` (`assumptions_ui`): each rationale is drawn under its input's own row, which shows the value, a reset and the changed-from-default dot;
  - `gui/explorer.rs`: the rationale in Explain;
  - `gui/plain_text.rs`: the four C1 rows, `stored_3d_caveat`, the bore-help test;
  - `gui/test_support.rs`: `passing_design`, `only_the_joint_fails`, `short_magnets`.
- **Sheets in `magcoupling-rs/target/g3/`, each against its `.r1`:**
  - `BL064-inputs.txt` differs in two input helps alone: the safety factor's and the bare cup's limit's;
  - `BL064-guidance.txt`, `BL064-overlay.txt` and `BL064-note.txt` differ only in a test timing line.

  So no Problem sentence, overlay row or note sentence moved this round.
- **Tests run.** They ran from the scratchpad, with the target dir there, `--locked` and `MAGCOUPLING_BLESS` unset:
  - `tests/shaft_joint.rs`: 34 passed; `tests/assumptions.rs`: 13 passed; `tests/schema.rs`: 11 passed. The five new tests are among them.
  - Each new test is red on the old text by inspection. Each asserts a phrase only the new text has, or no ASCII digit where the old text typed "20 °C", "0.14", "5 µm" or "10 µm".
  - `git status` was unchanged afterwards.
- **The probe crate:** `<scratchpad>/bl064g3-r2`, its own `CARGO_TARGET_DIR`, `magcoupling-rs` by path, feature gui. Each design went through `compute_all`, the "Shaft joints" guide's Problem, `guidance::fixes` (clamp section paused), `guidance::level` and the status line.
  - **Defaults:**
    - "Both shaft joints hold and can be fitted", no Problem, "Example design: fails 4 of 9 checks";
    - required 7.6466 N·m (2 × cold-high 3.8233); E 205 GPa, yield 415 MPa, limit 345.83 MPa;
    - holding 9.4844 / 16.899 N·m; bore stress 241.70 / 312.53 MPa; key 127.44 / 101.95 MPa;
    - needed 12.082 / 11.310 µm; elastic 23.808 / 23.863 µm; shrink 18.988 / 0 µm; fittable 18.828 / 24.440 µm;
    - fitting 93.267 / 280.16 °C against 100 / 300 °C;
    - the inner and the governing single-ring onset are both 143.356 °C.
  - **The five probes:**
    - 8 magnets: passes, needs 5.598 N·m;
    - 304: "The fit slips (inner hub): holds 0.00 N·m against 4.83 N·m". Both empty-window sentences show: 16.31 against 11.72 µm and 15.80 against 11.74 µm;
    - Y30: inner onset +inf; passes;
    - 20 mm ring: passes, the hub holds 11.77 N·m.
  - **MF-1, slip cycles per event** (`metal.slip_freq_Hz` × `metal.slip_event_s`):
    - 16.667 at the defaults (166.67 Hz × 0.1 s, about 33 reversals);
    - 0.8333 at 100 rpm, the slider's bottom;
    - 0.0333 at 100 rpm with 4 poles and a 0.01 s event;
    - 50 at 6000 rpm.
  - **MF-2, which end binds.** Holding in N·m at rotor-over-shaft 0 / 10 / 20 / 60 °C:
    - 4140 (pick 1): hub 14.407 / 11.946 / 9.484 / 0, cup 23.067 / 19.983 / 16.899 / 4.563. The hot end binds throughout.
    - 416 (pick 4): hub 14.0443 at 0, 10, 20 and 21 °C, then 6.625 at 60. Cup 22.530, then 13.230 at 60. The cold end binds up to 20 °C.
    - 17-4PH H900 (pick 6): hub 14.888 at 0 and 10, 12.845 at 20, 4.360 at 60.
    - 304 (pick 7): hub 4.975 / 1.741 / 0 / 0.
    - 6061 (pick 8): 0 throughout.
    - With the minimum magnet temperature at 20 °C (no cold mismatch): 9.484 / 16.899.
  - **MF-3, the aluminium cup** (`coupling.backiron` 0, or pick 8): E 68.9 GPa, α 23.6e-6 /°C, yield 241 MPa.
    - The cup needs 155.6 °C against the 300 °C default.
    - With the user's limit at 150 °C: "Fitting the cup would need it at 155.6 °C ... above the 150.0 °C allowed: your limit for the bare cup".
  - **MF-4 (a, b), the shrinks:**
    - wheel shaft: 18.988 µm at -196 °C, 10.354 at -78.5, 0 at 20, -2.300 at 40;
    - gearbox shaft: 4.553 µm at -20 °C, 0 at 20, -2.300 at 40.
  - **MF-4 (c-f), the rationales at their slider ends:**
    - friction 0.05 / 0.14 / 0.30: hub 3.387 / 9.484 / 20.32 N·m, in proportion;
    - smoothing 0 / 15: hub holding 19.49 / 0 N·m, hub stress 331.4 / 62.4 MPa, cup stress 401.9 / 133.7 MPa;
    - rotor over shaft 0 / 60: hub 14.41 / 0 N·m;
    - joining clearance 0 / 30: hub needs 11.97 / 255.9 °C, cup 198.9 / 442.8 °C.
  - **The spec's section 3.6 designs** (each as round 1 has it):
    - hub minimum 10: 3.48 against 7.65 N·m, window 12.09 to 18.82 µm. Cup minimum 10: 4.36 N·m;
    - hub maximum 26 / 28: 385.1 / 421.0 MPa, fitting 158.3 / 174.6 °C. Cup maximum 26 / 28: 384.0 / 419.8 MPa, 312.7 / 328.9 °C;
    - friction 0.10: 6.77 N·m. Smoothing 0: cup 401.9 MPa, window 6.310 to 18.86 µm;
    - wheel shaft 20 °C: 247.6 °C, and the joining-clearance clause shows. -78.5 °C: 163.5 °C, fittable 10.19 µm. 40 °C: 266.3 °C;
    - gearbox shaft -20 °C: 243.1 °C. 40 °C: 298.9 °C. Bare cup allowed 250 °C: 280.2 against 250.0 °C;
    - DP460: 93.3 against 60.0 °C, "the selected adhesive's design limit, 60.00 °C". Adhesive code 0: unknown, allowed NaN;
    - 2214 Hi-Temp, the hub input typed 150 °C, hub maximum 26: "above the 133.4 °C allowed: the magnets' single-ring onset, 143.4 °C, less 10.0 °C";
    - the passing design's single-ring caps:
      - N50M on manual dimensions: 88.80 °C (onset 98.80);
      - inner N42M with outer N35: 81.35 °C (inner onset 91.35, governing 59.93);
      - B842 in both rings: 59.83 °C (onset 69.83);
      - inner B842SH with outer B842: passes at 100 °C (inner onset 143.36, governing 69.83);
    - keyway 0.3: "Key backup above its limit (inner hub): 509.8 MPa against 345.8 MPa"; the cup reads 407.8 MPa. Keyway 0: passes, 10.49 / 18.68 N·m, key 0 / 0;
    - boss length 0: the cup slips at 2.73 N·m, key +inf, "its boss and web, 0 mm and 2.500 mm long". Hub length 3: 2.19 N·m, key +inf;
    - bore 16 with boss 12: the inner hub slips at 4.57 N·m, and the cup's no-wall sentence shows. With the hub minimum at 15: "Shaft joint check unknown: no wall around the bore (cup)";
    - bore 16 with keyway 2.1: no wall (inner hub);
    - hub band 20 over 18: unknown (reversed). Add the cup minimum at 10: the cup's slip leads and the reversed-band sentence follows;
    - bore 0: not a number. Key width 40: 0 N·m on both joints, both no-length key sentences, no window sentence. Contact height -0.5: "its contact height, -0.5000 mm, is not above zero";
    - safety factor 0 / -1: "Shaft joint check unknown: the safety factor is not above zero", Caution. Problem: "The safety factor, 0, is not above zero: enter at least 1." (and "-1.000"), no fix;
    - back-iron pick 99: not a number. Axial override 40: the cup slips at 16.90 against 25.99 N·m, key 346.6 MPa;
    - c_end 1.5: cold-high -0.1726, required -0.3451 N·m. The row is greyed (`END_EFFECT_ROWS`);
    - the tests' short magnets (c_end 0.5, both rings on manual dimensions, blank parts, 2 mm magnets, no axial override): cold-high -0.8079, required -1.616 N·m. The row is greyed;
    - BL-067 corner: cold-high 3.1823, required 6.3647 N·m; passes. The operating pull-out is 3.3246 N·m, so 3.657 with the allowance;
    - single-ring field 1500 kA/m: inner onset -2.45 °C, "needs 93.3 °C against -12.4 °C", fittable 4.997 µm. At 862 kA/m: "93.3 °C against 93.2 °C".
  - **Direction steps** (for SF-4, SF-7 to SF-10 and SF-12):
    - bore +0.1 mm:
      - hub wall 5.10 -> 5.05 mm; wall past the key 3.40 -> 3.35 mm;
      - holding 9.484 -> 9.461 (hub) and 16.899 -> 16.914 N·m (cup);
      - bore stress 241.70 -> 239.49 and 312.53 -> 309.61 MPa;
    - keyway +0.05 mm: wall past the key 3.40 -> 3.35 mm; key pressures unchanged (127.44 / 101.95 MPa). From 1.0 mm: hub key 152.93 -> 145.65, cup key 122.35 -> 116.52 MPa;
    - boss length +0.5 mm:
      - cup holding 16.899 -> 17.444 N·m; cup key 101.95 -> 98.03 MPa;
      - axial stack 31.8 -> 32.3 mm; large-diameter stack 18.8 mm, unchanged;
      - boss mass 30.78 -> 31.96 g;
    - single-ring field +1 kA/m:
      - example: cure margin 121.356 -> 121.227 °C, inner onset 143.356 -> 143.227 °C, hub allowed 100 -> 100 °C;
      - B842 in both rings: cure margin 47.833 -> 47.633 °C, hub allowed 59.833 -> 59.633 °C;
    - wheel shaft +1 °C: shrink 18.988 -> 18.936 µm; the hub needs 93.267 -> 93.687 °C;
    - rotor over shaft +1 °C: 416 holds 14.0443 -> 14.0443 N·m (cup 22.530, unchanged); 4140 holds 9.484 -> 9.238 (cup 16.899 -> 16.591).
  - **Observation O-1** (no text change asked):
    - With adhesive code 0, the empty adhesive with a NaN limit, the holding rows read 16.97 / 26.28 N·m. That is the cold end alone: `py_min` drops the NaN hot end.
    - The check reads unknown and the Problem says it cannot be made, so nothing is judged on those numbers.
    - The selector offers codes 1 to 4 and `validate` rejects 0, so no slider or typed value the spec names reaches it.
    - This is recorded for the engine's NaN handling, outside this review.

3. Strings. Rows marked "changed" are this round's; every other row is round 1's finding, which stands.

| # | String (field) | Verdict | Reason |
|---|---|---|---|
| 1.1-1.4 | the four interference inputs, labels and helps | PASS | Unchanged; round 1 stands. |
| 1.5 | `safety_factor` label, help (changed, MF-1) | PASS | "A slip can reverse the torque many times" is true: 16.67 cycles (about 33 reversals) per event at the defaults and 50 at 6000 rpm. "can" covers 0.83 cycles at 100 rpm, and "each slip" is gone. 26 words, 2 sentences, no Hz, no bound on cold-high. SF-11: in this group "slip" is the fit's word. |
| 1.6 | `friction` label, help | PASS | Unchanged. SF-6 would now match the new rationale word for word. |
| 1.7 | `smoothing_um` label, help | PASS | Unchanged. |
| 1.8 | `rotor_over_shaft_C` label, help | PASS | Unchanged and true: a warmer rotor loses grip at the hot end (4140: 9.484 -> 9.238 N·m per +1 °C). It now reads looser than the holding helps and the rationale: at 416 the holding row does not move (14.0443 N·m at 0, 20 and 21 °C). SF-12. |
| 1.9-1.11 | key width, key contact, the bonded hub's limit | PASS | Unchanged; round 1 stands (SF-1). |
| 1.12 | `cup_fit_max_C` label, help (changed, MF-3) | PASS | Names no steel. "the treatment temperatures of its material and finish" fits 4140 with UltraOx and the 6061-T6 cup the selectors reach (E 68.9 GPa, yield 241 MPa, needs 155.6 °C). "gearbox seal" has no "protect", and there is no "clamp". 27 words, 2 sentences. |
| 1.13 | `wheel_shaft_fit_C` label, help | PASS | Unchanged (SF-4). It agrees with the new shrink help. |
| 1.14 | `output_shaft_fit_C` label, help | PASS | Unchanged. It agrees with the new shrink help: 4.553 µm at -20 °C. |
| 1.15 | `joining_clearance_um` label, help | PASS | Unchanged. It agrees with the new rationale; neither types a number. |
| 2.1-2.5 | required, modulus, expansion, yield, stress limit | PASS | Unchanged; round 1 stands (SF-3). |
| 2.6-2.7 | `hub_holding_Nm`, `cup_holding_Nm` helps (changed, MF-2) | PASS | Exactly the engine: `py_min(mismatch(cold), mismatch(hot) − α·ΔT·d)`, cold the minimum magnet temperature, hot the adhesive's design limit. True where the hot end binds (4140, 304, 6061) and where the cold end binds (416 at 0-20 °C, 17-4PH H900 at 0-10 °C). 30 / 29 words, 1 sentence. "holds without slipping" (trap 18); "measured ... less smoothing" (trap 5). |
| 2.8-2.9 | bore stress helps | PASS | Unchanged. The parallel "at the temperature end that tightens it more" stays exact: the tightest end takes no allowance. |
| 2.10-2.11 | key pressure helps | PASS | Unchanged. |
| 2.12-2.13 | needed helps | PASS | Unchanged. "with smoothing and temperature included" reads the same loosest end the holding helps now name: 12.08 -> 9.62 µm with no allowance. |
| 2.14-2.15 | elastic helps | PASS | Unchanged. |
| 2.16-2.17 | shrink helps (changed, MF-4 a, b) | PASS | Zero at `ROOM_C`, the measured interference's temperature. 18.99 µm at -196 °C, 10.35 at -78.5, 4.553 at -20, -2.300 at 40 °C ("negative above room temperature, where it grows"). Above 265 K the engine takes 11.5e-6 /°C, near the NIST fit's own slope there (11.3e-6 /K), so "from published low-temperature data" still describes the number. No digit; 29 words each. |
| 2.18-2.23 | fittable, fit temperatures, allowed | PASS | Unchanged. None types the reference temperature, so nothing contradicts "room temperature". |
| 2.24 | `check` label, help | PASS | Unchanged. |
| 3.1-3.11 | the eleven verdict constants | PASS | Unchanged; each reproduced, `UNKNOWN_SAFETY_FACTOR` (Caution) at 0 and -1 among them. |
| 4 | badge tails; "inner hub", "cup" | PASS | Unchanged; the numbers never read equal: "93.3 °C against 93.2 °C" at 862 kA/m. |
| 5 | `inner_onset_single_ring_C` label, help | PASS | Unchanged (SF-2). |
| 6 | guide label "Shaft joints"; Why | PASS | Unchanged; 29 words. |
| 7.1-7.13 | every Problem sentence form | PASS | Unchanged: the guidance sheet is round 1's but for a timing line. SF-5 stands. |
| 8 | the four assumption labels, rationales (changed, MF-4 c-f) | PASS | Each says "The default is" and types no number. The panel draws it under the input's own row (value, reset, changed dot), so the default shows beside it. The physics holds at the slider ends: friction is proportional; smoothing moves grip and stress together; the allowance acts at the hot end; the clearance sets the fitting temperature (numbers above). "the example design's bore" is the 10 mm both joints share. |
| 9 | section labels "Shrink fit", "Fitting temperatures" | PASS | Unchanged. |
| 10-12 | headings "Shaft joints" (catalogue, package, chain) | PASS | Unchanged. |
| 13 | the new sentence of `STORED_3D_NOTE` | PASS | Unchanged; 1500 kA/m reads "needs 93.3 °C against -12.4 °C". |

4. Must-fix: none.

5. Should-fix offered. An input help changed here needs its line re-blessed in `tests/data/input_schema.json`, which the Global Constraints allow for T8. No label changes, so nothing is renamed.

- **SF-1. Name the inner ring in the bonded hub's limit helps** (round 1, unchanged).
  - **`shaft_joint.hub_fit_max_C`, help (`engine/shaft_joint.rs:82`).**
    - Old: "Its magnets are bonded before fitting. The check caps it at the adhesive's design limit and at the magnets' single-ring onset less 10 °C."
    - New: "Its magnets are bonded before fitting. The check caps it at the adhesive's design limit and the inner ring's single-ring onset less 10 °C; bond strain is not checked." (29 words, 2 sentences)
  - **`shaft_joint.hub_fit_allowed_C`, help (`:148`).**
    - Old: "The lowest of your limit for the bonded inner hub, the selected adhesive's design limit and the magnets' single-ring onset less 10 °C; not a number if any is."
    - New: "The lowest of your limit for the bonded inner hub, the selected adhesive's design limit and the inner ring's single-ring onset less 10 °C; not a number if any is." (30 words, 1 sentence)
  - Why: with mixed rings (inner B842SH, outer B842) the results table shows two onsets: the governing one at 69.83 °C and "Inner ring: single-ring onset" at 143.36 °C. The allowed temperature reads 100 °C. "the inner ring's" names the row the cap reads. "bond strain is not checked" is trap 20's item at this input.
  - Test: both keep "onset less 10 °C" for the cure-margin pin; add `contains("the inner ring's single-ring onset")`.
- **SF-2. `temperature.demag.inner_onset_single_ring_C`, help (`engine/temperature.rs:511`)** (round 1, unchanged).
  - Old: "The inner ring's own onset on its carrier with no outer ring around it, whichever ring governs. The bonded inner hub's allowed fitting temperature stays below it."
  - New: "The temperature at which the inner ring alone on its carrier starts to demagnetize, whichever ring governs. The bonded hub's allowed fitting temperature stays at least 10 °C below it." (30 words, 2 sentences)
  - Why: the help never says what an onset is, and it hides the margin. The temperature is exactly 10 °C below the onset where the onset binds (-2.45 and -12.45 °C at 1500 kA/m) and further below elsewhere.
  - Test: build "at least 10 °C below" from `CURE_MARGIN_MIN_C`.
- **SF-3. `shaft_joint.stress_limit_MPa`, help (`engine/shaft_joint.rs:114`)** (round 1, unchanged).
  - Old: "The part's yield strength with a margin. At the tightest fit the bore stress must not exceed it, so the part stays elastic and keeps its grip."
  - New: "The part's yield strength with a margin. The bore stress at the tightest fit must stay within it, so the part stays elastic; so must the key backup's pressure." (29 words, 2 sentences)
  - Why: "Key backup above its limit (inner hub): 509.8 MPa against 345.8 MPa" (keyway 0.3) quotes this row, but the row speaks only of the bore.
  - Test: `contains("key backup")`.
- **SF-4. `shaft_joint.wheel_shaft_fit_C`, help (`engine/shaft_joint.rs:88`)** (round 1, unchanged).
  - Old: "A cooled shaft shrinks, so the inner hub needs less heat. Dry ice is about -78 °C; liquid nitrogen is -196 °C."
  - New: "A cooled shaft shrinks, so the temperature the inner hub's fit would need falls. Dry ice is about -78 °C; liquid nitrogen is -196 °C." (25 words, 2 sentences)
  - Why: it matches the note's reviewed change entry and avoids "heat" (trap 14). The old text is not false.
  - Direction: Down, the note's aim reversed (+1 °C: 93.267 -> 93.687 °C).
- **SF-5. The Problem's empty-window clause (`gui/guidance.rs:2183`)** (round 1, unchanged).
  - Old: "the allowed temperatures do not even open its {clearance} joining clearance"
  - New: "at its allowed temperature its bore does not even open the {clearance} joining clearance on the {side.shaft} as set"
  - Renders: "... holding needs at least 12.09 µm, but at its allowed temperature its bore does not even open the 10.00 µm joining clearance on the wheel shaft as set." (wheel shaft 20 °C). The cup at a 20 °C limit reads "... on the gearbox shaft as set".
  - Why: each joint has one allowed temperature, so the plural misleads.
  - Tests: `:3415` becomes "does not even open the", and `:7656` takes the rendered sentence.
- **SF-6. `shaft_joint.friction`, help (`engine/shaft_joint.rs:64`)** (round 1's item, reworded to match MF-4 c).
  - Old: "Steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it."
  - New: "The default is for dry, degreased steel on steel. Holding torque scales with it; oil or a coating in the bore lowers it." (23 words, 2 sentences)
  - Why: with a 304 or aluminium part (pick 7 or 8, or no back iron) the contact is not steel on steel, so the bare fragment reads as a statement about the design. Its first sentence is now the rationale's own clause.
  - Test: the help's first sentence is a substring of the `fit_friction` rationale.
- **SF-7 (C1). `coupling.bore_mm`, overlay help (`gui/plain_text.rs:263`)** (round 1, re-verified).
  - Old: "Diameter of the shaft in the hub bore. Larger thins the hub wall: the hub check can fail, and the wall past the keyway can break through."
  - New: "Diameter of both shafts in the inner hub's and cup's bores. Larger thins the walls: the hub check can fail, and the wall past the keyway can break through." (29 words, 2 sentences)
  - Pins on "example": `model.hub_wall_mm` Down (5.10 -> 5.05 mm); `model.hub_wall_past_key_mm` Down (3.40 -> 3.35 mm). It keeps "hub check", the label the bore test needs ("Hub check").
- **SF-8 (C1). `coupling.keyway_depth_mm`, overlay help (`:267`)** (round 1, re-verified).
  - Old: "Depth of the key slot cut into the hub bore. Deeper leaves a thinner hub wall past the key, and too deep breaks through the hub."
  - New: "Depth of the key slot in the inner hub's and the cup's bores; it caps the key's contact height. Deeper thins the wall past the key; too deep breaks through." (30 words, 2 sentences)
  - Pins:
    - "example": `model.hub_wall_past_key_mm` Down (3.40 -> 3.35 mm); `shaft_joint.hub_key_pressure_MPa` Stays (127.44 MPa);
    - a new claim design "shallow keyway" (keyway 1.0 mm): `shaft_joint.hub_key_pressure_MPa` Down (152.93 -> 145.65 MPa).
- **SF-9 (C1). `metal.boss_length_mm`, overlay help (`:475`)** (round 1, re-verified).
  - Old: "Axial length of the shaft boss behind the rear wall: it adds to the overall length and the boss mass, not to the large-diameter length."
  - New: "Axial length of the shaft boss behind the rear wall: longer adds overall length and boss mass, not large-diameter length, and lets the cup's fit hold more." (27 words, 1 sentence)
  - Pins: the three existing rows (31.8 -> 32.3 mm; 18.8 mm unchanged; 30.78 -> 31.96 g), and `("metal.boss_length_mm", "example", "shaft_joint.cup_holding_Nm", Up)` (16.899 -> 17.444 N·m).
- **SF-10 (C1). `temperature.demag.h_rev_single_ring_kA_m`, overlay help (`:662`)** (round 1's item, reworded to name the inner hub).
  - Old: "Worst reverse field on a single ring during adhesive cure; higher lowers the cure margin." + the stored-3D caveat.
  - New: "Single-ring reverse field; higher lowers the cure margin and can lower the inner hub's fitting limit." + the stored-3D caveat (16 + 12 = 28 words, 2 sentences).
  - The label keeps "worst" and "on its carrier". "fitting limit" is `STORED_3D_NOTE`'s own word.
  - Pins:
    - the two existing rows;
    - `(…, "example", "shaft_joint.hub_fit_allowed_C", Stays)`: 100 -> 100 °C, since the onset does not bind at the defaults;
    - `(…, "N42 parts", "shaft_joint.hub_fit_allowed_C", Down)`, with a new claim design "N42 parts" (library B842 in both rings): 59.833 -> 59.633 °C.

    "can" covers the claim at both designs.
- **SF-11 (new). `shaft_joint.safety_factor`, help (`engine/shaft_joint.rs:61`).**
  - Old: "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. A slip can reverse the torque many times."
  - New: "The fit alone must hold cold-high times this, and the key alone is checked at the same torque. While the coupling slips, the torque can reverse many times." (28 words, 2 sentences)
  - Why: in this group "slip" is the fit's word. The badge reads "The fit slips"; the holding helps say "holds without slipping". So "A slip" next to "The fit alone must hold" can be read as the fit slipping.
  - The house names the coupling's slip in exactly this form:
    - the gearbox Why: "... and in each swing while it slips";
    - "When the coupling slips" on the slip-heating texts.
  - The physics is MF-1's: 16.67 cycles per event at the defaults, 0.83 at 100 rpm, so "can" stays.
  - Test: in `the_safety_factor_help_says_a_slip_can_reverse_the_torque_not_that_each_does`, the asserted phrase becomes "While the coupling slips, the torque can reverse many times."; its `!contains("each slip")` and both cycle pins stay.
- **SF-12 (new). `shaft_joint.rotor_over_shaft_C`, help (`engine/shaft_joint.rs:72`).**
  - Old: "How much hotter each rotor may run than its shaft from slip heating. A warmer rotor's bore grows, so the grip falls."
  - New: "How much hotter each rotor may run than its shaft from slip heating. A warmer rotor's bore grows, so the grip at the hot end falls." (26 words, 2 sentences)
  - Why: this is MF-2's own fact, seen from the input. The allowance enters only the hot end.
    - With 416 (pick 4) the cold end binds. The holding rows read 14.0443 / 22.530 N·m at 0, 20 and 21 °C, so moving this input moves nothing, and "the grip falls" reads as a broken readout.
    - The rationale already says "the grip at the hot end falls", and the holding helps now say "hot with the rotor warmer than its shaft".
    - At 4140 the holding falls (9.484 -> 9.238 N·m per +1 °C), so the new claim is true wherever the old one was.
  - Test: `contains("the grip at the hot end falls")`. The round's `the_holding_helps_put_the_rotor_allowance_at_the_hot_end_alone` already pins 416 bit for bit at 0 and 20 °C.

6. Traps, checked against every changed string and its dependents; round 1's check of the unchanged strings stands.
   1. **Fit and key torques never added.** Holds; no changed string adds a torque.
   2. **Cold-high.** No "strongest", "at most" or "most expected"; MF-1 keeps "cold-high times this". At the BL-067 corner 3.1823 < 3.3246 × 1.1 = 3.657 N·m, and no text orders them.
   3. **No ratio, efficiency or `cold_high_input_Nm`.** Holds.
   4. **The swing frequency.** MF-1 quotes no Hz and claims reversals only as "can" (16.7 cycles, about 33 reversals, per 0.1 s event at the defaults).
   5. **Measured against effective.** Holds: the holding helps say "minimum measured interference less smoothing"; the shrink helps speak of the shaft alone.
   6. **No fit class, no "tighter is safer".** Holds.
   7. **Temperature cuts both ways.** MF-2 puts it in words: 416 binds cold; 4140, 304 and 6061 bind hot.
   8. **No speed term.** Holds.
   9. **Sources named.** No changed string quotes a fitting temperature. MF-3 says what the bare cup's limit must stay below. No "safe".
   10. **No 150 °C.** Not quoted.
   11. **Gearbox.** "keep the heat off the gearbox seal"; no "protect", no catalog, no cooling below the rated range.
   12. **Bath temperatures.** No changed string names a bath; "room temperature" is a reference, not advice.
   13. **No induction heating.** No method named.
   14. **No "heat the hub" or "heat until".** None in a changed string. The wheel shaft's unchanged "needs less heat" is SF-4.
   15. **Fitting order.** MF-3 keeps "fitted before its magnets are bonded".
   16. **No "lower the safety factor to fit".** Holds.
   17. **The DIN 6885-1 row.** Unchanged.
   18. **"holds without slipping".** Kept in MF-2.
   19. **Friction and smoothing are assumptions.** The rationales keep friction proportional and smoothing moving grip and stress together, and each value is now "The default".
   20. **What nothing checks.** Nothing changed claims the seal, bond strain or fretting fine; the stress helps keep "The keyway corner is not included" (SF-1 adds bond strain).
   21. **No "clamp".** None.
   22. **Glyphs.** ASCII "-" only; no U+2192, U+2300 or U+2212. No cell id, "BL-064" or "Rust-only" on screen (the README and test comments are not drawn).

   **Typed numbers:**
   - **In the changed strings:** none. The round's tests pin no ASCII digit in the two shrink helps and the four rationales.
   - **In the unchanged strings,** round 1's list stands:
     - allowed: the bath temperatures -78 °C and -196 °C, and -20 °C at the gearbox shaft's slider bottom (the owner's Shaft temps row); the DIN 6885-1 row (3 mm, 1.2 mm, 10 mm); the cure margin, 10 °C, in two helps;
     - "1" in "enter at least 1" (ruling C-4);
     - names, not numbers: "4140", "DIN 6885-1", "3D".
   - **In the offered items:** SF-1 and SF-2 type the cure margin (allowed, built from `CURE_MARGIN_MIN_C`); SF-4 the bath temperatures (allowed). No other number.

7. The context lines:
   - **C2, the fix texts:** fine. No label changed, so every fix still reads "Raise/Lower <label>": "Lower Inner hub interference, maximum", "Lower Gearbox shaft temperature at fitting", "Raise Shaft key contact height", "Raise Integral steel boss axial length".
   - **C3, the status line:** "Example design: fails 4 of 9 checks" with the clamp paused.
   - **C3, the row's hover:** `STORED_3D_LABEL` "Fixed 3D fields from the example design" with `STORED_3D_NOTE`. The joint row is in `STORED_3D_ROWS` and `END_EFFECT_ROWS`.
   - **C3, Explain:** "Closest teaching note: Shaft joints: a keyed shrink fit". An assumption's "Closest assumption: <label>: <rationale>" reads right with "The default is".
   - **C4, the note:** consistent.
     - Sentence 3 ("a rotor running warmer than its shaft loses more") agrees with MF-2.
     - Watch-out 1 ("fretting under the reversing swing") agrees with MF-1.
     - No note sentence types a reference temperature.

8. Verdict: PASS. No must-fix item; SF-1 to SF-12 offered; no item needs a model change.

### Re-review after a change by the implementer after its review (2026-10-18)

BL-064 Task 1 (commit 1c1d92a) added the Rust-only result `temperature.demag.inner_onset_single_ring_C`, the inner
ring's own single-ring onset (`engine/temperature.rs`), with its label "Inner ring: single-ring onset" and its help
"The inner ring's own onset on its carrier with no outer ring around it, whichever ring governs. The bonded inner hub's
allowed fitting temperature stays below it.", and the doc and README texts around it (the `Material::yield_MPa` and
`HubMaterial` docs in `engine/material_library.rs`, the README rows). The label and help stay drafts until Task 8's
G-3 review. They had a G-3 text re-review in a fresh context at HEAD 1c1d92a. The reviewer's report for the record
follows unchanged (full answer:
`.superpowers/sdd/2026-10-18-bl064-shaft-joint/textreview-T1-a-change-by-the-implementer-after-its-review-1.md`).

- Reviewer: fresh context, session model, 2026-10-18; G-3 text re-review of BL-064 Task 1's texts (round 1). There
  was no note text to review: the note `shaft_joint` arrives in Task 7.
- Verdict: PASS, no must-fix. Nothing in the changed text is false at the merge, at the defaults, at the five probe
  designs or at the mixed-ring pins; no trap is broken and no cap is exceeded (the help: 27 words in 2 sentences, the
  plain-words guard green).
- Should-fix offered, none applied in this round; they stay open for the controller:
  - SF-1 (for Task 8's G-3 review, item 5 of its list): the help's "stays below it" hides the 10 °C margin. Offered:
    "The inner ring's own onset on its carrier with no outer ring around it, whichever ring governs. The bonded inner
    hub's allowed fitting temperature is capped 10 °C below it." (30 words, 2 sentences).
  - SF-2 (code doc): the `Material::yield_MPa` doc should say that the engine reads only the hub, cup and boss
    material's yield, for the shaft joint check, through `PartProperties::hub`.
  - SF-3 (README): the `tests/grades.rs` row should say which onset is which: "its own 143.3565 °C against the
    governing 69.8326 °C" and "its own 112.8297 °C against the governing 191.8772 °C".
  - SF-4 (code doc): the `HubMaterial` field docs should read "when the circuit in effect is steel" and "a code
    outside the choices".
- No text changed in this round.

**Report for the record (as returned).**

**R-1. The field behind the "inner ring's own" onset (physics; no change asked).**
- The inner ring's onset uses the one shared single-ring field input, `temperature.demag.h_rev_single_ring_kA_m`.
  Its default, 569 kA/m, is the outer blocks' value. The input's own help says the inner ring alone sees 545 kA/m.
- At the defaults the result reads 143.36 °C. With 545 kA/m it would read 146.45 °C
  (`.superpowers/sdd/2026-10-17-bl064-shaft-joint/research-assembly.md:98-99`). So the cap is about 3 °C tighter
  than the inner-only field would give. That is the safe side.
- This is the convention of the whole block:
  - the workbook's own C59 checks the inner ring at 569;
  - `inner_magnet_limit_C` reads 863 kA/m, not the inner ring's 844.
- The owner's R-15 row accepted 143.36 / 133.36 °C. The help is therefore not false. Using a per-ring field would
  change the model, which is the owner's call (next to BL-072), not a wording fix.
- The G-3 reviewer may still judge whether "on its carrier with no outer ring around it" should say that it reads the
  single-ring field input.

**R-2. Forward references (true at merge).** Three T1 texts name things T2 adds:
- the help's "allowed fitting temperature" (`shaft_joint.hub_fit_allowed_C`);
- the yield doc's "shaft joint check";
- the README temperature row's "for the BL-064 shaft-joint check".

The commit message says only the merge ships, as the brief asks. At T1, `HubMaterial::yield_MPa` has no engine
reader yet; only the tests read it.

**R-3. What was verified.**
- **The `hub()` rule, branch by branch, against `resolve`:**
  - steel circuit: the pick's props (the Materials inputs for the default 4140) and the pick's yield;
  - non-magnetic pick: the free-space circuit, the body equal to the pick, and the pick's yield;
  - `coupling.backiron` 0 with a magnetic pick or a code outside the choices: the workbook aluminium body and
    6061-T6's 241 MPa;
  - a code outside the choices on the steel circuit: NaN props and NaN yield.
- **The `api.rs` hub expansion** is bit for bit the old `if parts.backiron == 1` choice.
- **The README pins:** edit (e)'s numbers and governing rings match `tests/grades.rs:424-468`, and edit (f) matches
  `tests/material_links.rs:437-500`.
- **README edit (g)** fits its paragraph: E20's per-ring results are `out_rust_only` in the ported temperature group.
- **The counts:** the scope 170 and the trace 393 of 1,088 match the tests. `TemperatureResults` stays 130 cells,
  since the new result is Rust-only with no cell.
- **The record** (`records/demagnetization.rs`) matches `ring_demag`'s fourth onset:
  - the same calibration offset;
  - +inf for a positive beta;
  - no design margin.

**R-4. Two 6061 yields are deliberate.** The library's 6061-T6 is 241 MPa (published minimum). The workbook's
`AL6061.yield_MPa` is 276 MPa (`materials.rs:88`), and no engine code reads it. Synthesis correction 6 chose one
aluminium with one yield, and the README's "library yield" keeps that right.

**R-5. The yield is the pick's, by the owner's decision.** On the steel circuit with the default pick, E and α come
from the Materials inputs, which the user can type. The yield stays the library's 4140 value, 415 MPa (annealed,
conservative). This follows the owner's "Material" row, so "the metal in effect" in the README means the pick as far
as the yield goes. No change.

**R-6. One stale count, already scheduled.** `docs/ai/04-memory.yaml:20` still says "392 explained results of 1087".
`task-9-brief.md:564` schedules it for T9 Step 6 (final 416 of 1,112). Not T1's to fix under the brief.

**R-7. The trace doc's wording.** `gui/trace.rs:7` says "(the A-3 scope: 393 of the 1088 results)". The 393 now
include one BL-064 record, and "A-3 scope" is the historic name. T2 and T3 rewrite this line again (to 393/1112, then
416/1112), so it can stay.

**R-8. The traps on the new label and help.**
- None of these appear: "clamp", "safe", "heat the hub", "BL-064", the grade's 150 °C, U+2192 or U+2300.
- "Inner ring: single-ring onset" matches the Problem's "the magnets' single-ring onset" (spec 5.2), so a reader can
  find the quoted onset in the results table.
- "on its carrier" matches the input label "single ring on its carrier".

### Re-review after a change by the implementer after its review (2026-10-18)

BL-064 Task 2 (commit 02277cd) added the shaft joint check (`engine/shaft_joint.rs`): the 15 Rust-only
`shaft_joint` inputs and the 24 results with their labels and helps, the verdict texts, the input group heading "Shaft
joints", the inputs panel's sections "Shrink fit" and "Fitting temperatures", the results heading "Shaft joints", and the doc
texts around them (the README rows, the crate doc in `src/lib.rs`, `docs/ai/02-system.yaml` and
`docs/ai/03-structure.yaml`). The labels, helps and verdict texts stay drafts until Task 8's G-3 review. They had a
G-3 text re-review in a fresh context at HEAD 02277cd. The reviewer's report for the record follows unchanged (full
answer: `.superpowers/sdd/2026-10-18-bl064-shaft-joint/textreview-T2-a-change-by-the-implementer-after-its-review-1.md`).

- Reviewer: fresh context, session model, 2026-10-18; G-3 text re-review of BL-064 Task 2's texts (round 1). There
  was no note text to review: the note `shaft_joint` arrives in Task 7 (the note dump names no note).
- Verdict: MUST_FIX, one item. Every on-screen draft is true at the defaults and the five probe designs, no trap is
  broken and no cap is exceeded (the longest helps are 30 words, the schema guard green). The one false text was a
  README number.
- Must-fix applied in this round:
  - MF-1 (`magcoupling-rs/README.md`, the `tests/shaft_joint.rs` row): the row called 6.3647 N·m cold-high, but that
    is the required torque, twice cold-high. The test asserts cold-high 3.1823 N·m and required 6.3647 N·m
    (`tests/shaft_joint.rs:407-409`). Old: "the BL-067 corner reads cold-high (6.3647 N·m) and passes;" New: "the BL-067 corner reads cold-high (3.1823 N·m, so 6.3647 N·m required) and passes;". Docs only: no test, count or on-screen text moves, and
    `input_schema.json` is not re-blessed.
- Should-fix offered, none applied in this round; they stay open for the controller (SF-1 to SF-8 for Task 8's G-3
  review, SF-9 for the Task 2 fix round or Task 9):
  - SF-1 (`shaft_joint.cup_fit_max_C` help): "the steel's" names the wrong material when the cup is 6061-T6.
    Offered (28 words, 2 sentences): "The cup is fitted before its magnets are bonded. Stay below the treatment temperatures of its material and its finish, and keep the heat off the gearbox seal."
  - SF-2 (`shaft_joint.hub_fit_allowed_C` help): "the magnets'" is ambiguous with mixed rings. Offered (30 words,
    1 sentence): "The lowest of your limit for the bonded inner hub, the selected adhesive's design limit and the inner ring's single-ring onset less 10 °C; not a number if any is." Optional, same count: "the magnets'" becomes "its magnets'" in the `hub_fit_max_C` input
    help (an input help change re-blesses `input_schema.json`, which Task 8 may do).
  - SF-3 (the two shaft shrink helps): the published low-temperature data hold only below -8 °C, and a constant
    coefficient applies above. Offered for the wheel shaft (30 words, 1 sentence), the gearbox shaft's the same with
    "gearbox" for "wheel": "The wheel shaft's diameter shrink from 20 °C to its fitting temperature: published data for a nickel steel below -8 °C, a constant expansion coefficient above; negative when it grows."
  - SF-4 (`shaft_joint.wheel_shaft_fit_C` help): at the default -196 °C the help does not say that a cooled shaft can
    chill the bonded inner hub below the minimum magnet temperature, and "needs less heat" becomes "needs less
    warming". Offered (30 words, 2 sentences): "A cooled shaft shrinks, so the inner hub needs less warming but may chill below the minimum magnet temperature. Liquid nitrogen is -196 °C; dry ice is about -78 °C."
  - SF-5 (`shaft_joint.output_shaft_fit_C` help, low priority): the slider's bottom, -20 °C, may be below the
    gearbox's lowest rated temperature. Offered (30 words, 2 sentences): "The gearbox output shaft cannot be immersed. Cold spray can take it to -20 °C, the slider's bottom, but never below the gearbox's lowest rated temperature, which may be warmer."
  - SF-6 (`shaft_joint.stress_limit_MPa` help): "with a margin" is vague and the key is not named. Offered (28 words,
    2 sentences): "The part's yield strength divided by 1.2. The bore stress at the tightest fit, and the key's pressure alone, must not exceed it, so the part stays elastic." If adopted, a test holds the quoted 1.2 to `YIELD_FACTOR`.
  - SF-7 (`shaft_joint.part_yield_MPa` help): "while the condition is open" is shorthand. Offered (26 words,
    1 sentence): "The library's yield strength of the inner hub and cup material in effect; for 4140, the annealed value, conservative while its heat treatment is not known."
  - SF-8 (the two holding helps): "rotor warmer than shaft" applies only at the hot end. Offered for the inner hub
    (29 words, 1 sentence), the cup's the same with "the cup's fit" (28 words): "Torque the inner hub's fit holds without slipping at its loosest: minimum measured interference less smoothing, cold or hot with the rotor warmer than its shaft, whichever loosens more."
  - SF-9 (`docs/ai/02-system.yaml`, the BL-067 citation; not Task 8's): BL-067 is the operating temperature below the
    minimum, not backlash, resonance or re-lock. Old: "it is not a bound on them (backlash, resonance and re-lock can exceed it on the gearbox side; BL-067)." New: "it is not a bound on them (backlash, resonance and re-lock can exceed it on the gearbox side; with the operating temperature below the minimum, so can the operating pull-out, BL-067)."
- Text changed in this round: the README row of MF-1 only.

**Report for the record (as returned).**

**R-1. What was verified true.**
- **Every label and help against the engine:**
  - the Lamé pressure and the bore von Mises;
  - the holding torque with the key arc floored;
  - the key alone: 0 with no keyway, +inf with no bearing area;
  - the needed, elastic and fittable window ends;
  - the needed fitting temperature;
  - `hub_fit_allowed` (NaN first, `CURE_MARGIN_MIN_C` = 10, `temperature.rs:748`);
  - the expansion claim: a higher α loosens hot, a lower one loosens cold;
  - the modulus claim: pressure rises with E;
  - the part material rule of `PartProperties::hub`.
- **The verdict texts and badge tails** against `verdict` and `text`. The cause order, NaN first, then the safety
  factor, then no wall, then the reversed band.
- **The traps.**
  - None of these appear in any T2 text: "clamp", "safe", "strongest", "at most", "up to", "most expected", "heat the
    hub", "protect", "permanent", induction, the grade's 150 °C, a fit class, a ratio, "BL-064", U+2192 or U+2300.
  - The bath temperatures use the house minus sign.
  - The fitting order is the owner's.
  - Each fitting-temperature help names its source.
  - The keyway corner is said to be left out.
  - The fit and the key are never added.
- **README (a), (c)-(k) and (m), the crate doc, `02-system.yaml` (a) and (c) and `03-structure.yaml` (a)-(d)** match
  the code and the tests:
  - the 15 links;
  - 393 of 1,112;
  - 2,643 characters, guard 2,700;
  - the four "gearbox" matches;
  - the old link and a file loading the joint inputs at their defaults;
  - the robustness cases.
  "The first top-level Rust-only input group" is true: every earlier group in `DesignInputs` is Python-ported.

**R-2. Forward references, true at merge** (as in the T1 review's R-2).
- README edit (l) describes the CSV and JSON export test of the 24 results and 15 inputs. That test is T4's
  `the_exports_carry_every_shaft_joint_input_and_result_with_no_cell` (`task-4-brief.md:381-470`, which matches every
  clause). At `02277cd` no test asserts it.
- BL-070 is cited in the module doc and in `02-system.yaml` (b). It is created in T9.

**R-3. Caps.**
- All 39 helps: at most 30 words and 1-2 sentences. At the cap of 30: `hub_bore_stress_MPa`, both `*_fit_temp_C`, and
  `check`.
- All 15 input defaults sit on their slider grids (tested).

**R-4. Minor wording, the G-3 reviewer's call.**
- **`key_contact_mm`: "DIN 6885-1 gives 1.2 mm for 10 mm".** 1.2 mm is the 3 × 3 row's h − t1 (3 − 1.8;
  `research-physics.md:295-297`), not a tabulated value. It is true as "gives".
- **The key pressure helps.** "were the key to carry the torque alone" means the required torque.
- **The section "Fitting temperatures"** holds the joining clearance, in µm (spec 5.3 places it there).
- **README edit (f)'s parenthetical** ("the bore and the shaft key") leaves out the keyway depth. That is incomplete,
  not false.
- **"inf" in the badge.** An infinite cold-high quotes "against inf N·m" (`tests/shaft_joint.rs:1481`), while the key
  helps write "+inf". It is reachable only through a non-finite engine value.

**R-5. The typed "10 °C".** It appears in two helps (`hub_fit_max_C`, `hub_fit_allowed_C`), and no test ties it to the
constant. `CURE_MARGIN_MIN_C` is the workbook's Temperature design C25 and is Python-locked, so in practice it cannot
drift. The note's copy is pinned in T7 (spec 7.11).

**R-6. No model question arises.** None of the texts asks to change the engine. R-14 (the hot-end form) and R-1 (the
stress side's smoothing) stay with T7's physics review, as planned.

### Re-review after a change by the implementer after its review (2026-10-18)

BL-064 Task 2's text re-review 1 (commit b732cdb) applied its MF-1 to the `tests/shaft_joint.rs` row of
`magcoupling-rs/README.md`: the BL-067 corner now reads cold-high 3.1823 N·m, so 6.3647 N·m required, where the row
had called 6.3647 N·m cold-high. The same commit appended the round-1 record above. The changed text had its G-3 text
re-review (round 2) in a fresh context at HEAD b732cdb. The reviewer's report for the record follows unchanged (full
answer: `.superpowers/sdd/2026-10-18-bl064-shaft-joint/textreview-T2-a-change-by-the-implementer-after-its-review-2.md`).

- Reviewer: fresh context, session model, 2026-10-18; G-3 text re-review of BL-064 Task 2's changed texts (round 2):
  the README row of MF-1 and the round-1 record above. There was no note text to review: the note `shaft_joint`
  arrives in Task 7 (the note dump names no note).
- Verdict: PASS, no must-fix. The new README clause is true: the test asserts cold-high 3.1823 N·m and required
  6.3647 N·m at the BL-067 corner, and the engine's required torque is the safety factor (default 2) times cold-high.
  The round-1 record is faithful: its R-1 to R-6 are the round-1 answer's, unchanged, and every quoted text is
  verbatim. No trap is broken, no cap applies (neither item is on-screen) and `input_schema.json` needs no re-bless.
- Should-fix offered, none applied in this round; they stay open for the controller:
  - SF-1 (`magcoupling-rs/README.md`, the `tests/shaft_joint.rs` row, optional): the row gives the right numbers but
    not what "reads cold-high" is set against, the larger operating pull-out with its allowance. Offered: "the BL-067 corner reads cold-high (3.1823 N·m, not the 3.657 N·m operating pull-out with its allowance, so 6.3647 N·m required) and passes;"
  - SF-2 (this doc): the round-1 heading above repeats the Task 1 record's word for word, so a reader scanning the
    headings cannot tell them apart. Offered: "### Re-review after a change by the implementer after its review (2026-10-18), Task 2".
    Not applied: this record keeps the same heading form, and the first paragraph of each record names its task and
    round.
  - SF-3 (process): when Task 2's lines are written into the build ledger, one of them carries round 1's SF-9 (the
    BL-067 citation in `docs/ai/02-system.yaml`) with its offered text, which the round-1 record above holds verbatim.
- No text changed in this round.

**Report for the record (as returned).**

**R-1. The MF-1 clause, verified true.**
- `tests/shaft_joint.rs:407` asserts `metal.torque_cold_high_Nm` = 3.1823 and `:409` asserts `required_Nm` = 6.3647,
  each within an absolute 5e-5 (`assert_near`, `:75-77`). The corner is op -40 °C, minimum 20 °C, variation 0.1
  (`:400-404`), as spec 3.6 and 7.1 give it.
- `engine/shaft_joint.rs:742`: `required_Nm = inputs.safety_factor * links.cold_high_Nm`, the default safety factor
  2.0 (`:60`), `cold_high_Nm` = `metal.torque_cold_high_Nm` (`api.rs:147`). So "so 6.3647 N·m required" follows from
  cold-high as the clause says, and agrees with the `required_Nm` help ("Cold-high ... times the safety factor").
- Rounding: the two pins and the exact factor 2 bound cold-high to [3.182325, 3.18235] and required to
  [6.36465, 6.3647], so 3.1823 and 6.3647 are each the engine value correctly rounded to four decimals. Doubling the
  rounded 3.1823 gives 6.3646, a last-digit rounding difference a careful reader may notice; spec 7.1 quotes the same
  pair ("required 6.3647 N·m (2 × cold-high 3.1823)"). Not false; no change asked.
- The rest of the row is unchanged by b732cdb (a one-clause diff). No other doc calls 6.3647 N·m cold-high
  (`git grep` of `docs`, the README and `src`; the plan's line 4659 keeps the T2 draft, and plan edits go to the
  ledger). No later task brief (T3-T10) edits this README row, so no later Old text breaks.

**R-2. The appended record, verified faithful.**
- R-1 to R-6 under "Report for the record (as returned)" are identical to the round-1 answer's lines 138-195 once
  CRLF is stripped (the working tree is CRLF, the index LF, as for the rest of the file).
- All 12 quoted texts of the round-1 answer (MF-1's Old and New, SF-1 to SF-8's offered texts, SF-9's Old and New)
  appear verbatim in the record. The word and sentence counts the record gives (SF-1 28/2, SF-2 30/1, SF-3 30/1,
  SF-4 30/2, SF-5 30/2, SF-6 28/2, SF-7 26/1, SF-8 29/1 and the cup's 28) are the answer's.
- The record's own sentences are true: the input group heading "Shaft joints" (`inputs.rs:62`), the sections
  "Shrink fit" and "Fitting temperatures" (`:349`, `:364`), the results heading "Shaft joints"
  (`result_groups.rs:86`); 15 inputs and 24 results (39 helps); SF-3's "below -8 °C" (`NIST_FIT_TOP_K` 265 K is
  -8.15 °C, the constant 11.5e-6 /°C from there up, `:321-325`); SF-8's loosest end (`py_min` at `:636`);
  "Text changed in this round: the README row of MF-1 only" (the diff touches that row and the record alone).
- No line of the appended text starts with "## "; the record sits under "## BL-064 shaft joints" after the Task 1
  record (`:2727`), as the section's intro (`:2724-2725`) says re-reviews do until Task 8's G-3 review.
- The full-answer path (`.superpowers/sdd/...-1.md`) is git-ignored (`.gitignore:56`), as every earlier full-answer
  path in this doc is (`:1434`, `:1486`, `:2395`, `:2482`, `:2579`, `:2736`): the house convention.
- Cosmetic only: the MF-1 bullet's second line (`:2837`) runs past the doc's 120-column wrap because it inlines
  Old and New; it renders as one paragraph. No change asked.

**R-3. Traps and caps.** Neither changed item is on-screen text, so the help caps and the plain-words guard do not
apply. Even so, the new clause holds the traps that could touch it: no "at most", "strongest", "up to" or
"most expected" (trap 2), no ratio or efficiency (trap 3), no "clamp" (trap 21). No input label or help moved, so
`input_schema.json` is correctly not re-blessed (plan Global Constraints, re-bless rule).

**R-4. Guards run green.** `tests/shaft_joint.rs` 30 passed; the README-reading guards
`the_docs_count_the_notes_and_the_steps` and `the_docs_count_the_dashboard_rows_and_the_checks` passed.

**R-5. No model question arises.** Both changes are docs; neither asks to change the engine.

### Re-review after a change by the implementer after its review (2026-10-18)

BL-064 Task 3 (commit 6e8d172) added the shaft joint check's explain layer: the 23 equation records
(`engine/explain/records/shaft_joint.rs`), the three `back_iron` table fields with their labels and symbols (`E_{BI}`,
`α_{BI}`, `σ_{y,BI}`), the 18 new input symbols, the scope chain `shaft_joint` and its results-table heading "Shaft
joints", and the four assumption rows `fit_friction`, `fit_smoothing`, `rotor_over_shaft` and `joining_clearance`
(their labels, rationales and sources, the spec 4.5 drafts), with the doc texts around them (the README,
`docs/ai/02-system.yaml` and `docs/ai/03-structure.yaml`). The assumption labels and rationales stay drafts until Task
8's G-3 review, and the records are Task 7's physics review's. They had a G-3 text re-review in a fresh context at HEAD
6e8d172. The reviewer's report for the record follows unchanged (full answer:
`.superpowers/sdd/2026-10-18-bl064-shaft-joint/textreview-T3-a-change-by-the-implementer-after-its-review-1.md`).

- Reviewer: fresh context, session model, 2026-10-18; G-3 text re-review of BL-064 Task 3's texts (round 1). There
  was no note text to review: the note `shaft_joint` arrives in Task 7 (the note dump names no note).
- Verdict: PASS, no must-fix. No changed text is false at the defaults, at the five probe designs or at any reachable
  input, and no trap or cap is broken. The 23 records transcribe `shaft_joint::compute` and `PartProperties::hub`
  exactly. The four assumption rows are the spec 4.5 drafts verbatim (label, path, rationale and source, character for
  character), and each claim holds in the engine. The three table labels, the chain heading and the 18 symbols are
  accurate, and no new symbol collides with the crate's 500.
- Should-fix offered, none applied in this round; they stay open for the controller (SF-1 and SF-2 for Task 8's G-3
  review, SF-3 to SF-5 for Task 7's physics review, which owns the records, and SF-6, a test, for Task 8):
  - SF-1 (`fit_friction` rationale): "a finish in the bore lowers it" claims more than is known: oil lowers the
    friction, but no source in the spec or the research gives the direction for a coating or a finish. Offered (44
    words, 2 sentences): "The friction between each rotor's bore and its shaft that turns the fit's grip into holding torque. 0.14 is for dry, degreased steel on steel; oil, a coating or a finish in the bore can lower it, and the holding torque falls in proportion."
    The `shaft_joint.friction` input help (Task 2's text) has the same "lowers it"; a change there goes to the G-3
    review with a re-bless of `input_schema.json`.
  - SF-2 (`fit_smoothing` rationale, trap 5): "taken off both ends of the measured band" is true for holding and the
    bore stress, not for fitting, which reads the measured maximum with no smoothing taken off (5 µm is about 41 °C on
    the 4140 hub). Offered (57 words, 2 sentences): "Interference lost as the surface peaks of bore and shaft flatten when the fit is made: taken off both ends of the measured band for holding and bore stress, not for fitting. 5 µm is conservative for holding with ground and honed surfaces; a finer finish loses less, which raises the grip and the bore stress together."
  - SF-3 (`records/shaft_joint.rs`, the eight Lamé records' no-wall condition; the implementer's concern 1): it
    typesets as `D ≤ d_{bore} or t_{kw} > 0 and (D − d_{bore})/2 ≤ t_{kw}`. The evaluator groups it as the engine does
    (`and` binds tighter), but a reader may group it left to right. Offered: split it into the engine's two no-wall
    kinds, one arm each, with the same values. For the inner hub's four records the first `cases` line becomes the two
    lines below; the cup's four take `{metal.boss_od_mm}` for `[D]`. It typesets as
    `{ undefined if D ≤ d_{bore}; undefined if t_{kw} > 0 and (D − d_{bore})/2 ≤ t_{kw}; ... otherwise }`.

    ```text
    cases([D] <= {coupling.bore_mm} => nan;
          {coupling.keyway_depth_mm} > 0 and frac([D] - {coupling.bore_mm}, 2) <= {coupling.keyway_depth_mm} => nan;
    ```
  - SF-4 (`hub_fit_allowed_C`'s local `[ϑ_{cap}]`): in this crate "cap" is a part (the end cap's `D_{cap}`,
    `t_{cap}`, `σ_{cap}` and the `cap_housing` table). Offered: `[ϑ_{lim,sr,i}]` at each of the six occurrences, with
    `where [ϑ_{lim,sr,i}] = {temperature.demag.inner_onset_single_ring_C} - 10`.
  - SF-5 (two record symbols, low): (a) the key-arc local `[b]` sits beside the bondline `b_i` in the inner hub's
    formula. Offered `[b_{arc}]` in `hub_holding_Nm`, `hub_needed_um`, `cup_holding_Nm` and `cup_needed_um`:
    `[b_{arc}] = cases({coupling.keyway_depth_mm} > 0 => {shaft_joint.key_width_mm}; else => 0)`, and inside `[T_1]`
    `max(0, π · {coupling.bore_mm} - [b_{arc}])`. (b) `δ_{fit,hub}` and `δ_{fit,cup}`, the fittable edge, read as "the
    fit's interference". Offered `δ_{join,hub}` and `δ_{join,cup}`. Either rename belongs before the note quotes
    symbols.
  - SF-6 (`tests/assumptions.rs`, for Task 8): the four rationales type their defaults (0.14, 5 µm, 20 °C, 10 µm), and
    no test ties them to the inputs. Offered: for each `(id, path, quoted, value)` in
    `("fit_friction", "shaft_joint.friction", "0.14 is", 0.14)`,
    `("fit_smoothing", "shaft_joint.smoothing_um", "5 µm is", 5.0)`,
    `("rotor_over_shaft", "shaft_joint.rotor_over_shaft_C", "20 °C is", 20.0)` and
    `("joining_clearance", "shaft_joint.joining_clearance_um", "10 µm is", 10.0)`, assert
    `rationale(id).contains(quoted)` and `DesignInputs::default().get(path) == Some(Value::Num(value))`.
- No text changed in this round.

**Report for the record (as returned).**

**R-1. What was verified true.**
- **The 23 records against the engine, term by term:**
  - the Lamé pressure: `1e-3 / (d·((k_θ + ν)/E_part + (1 − ν)/E_shaft))`, with E in MPa;
  - the holding torque, with the key arc floored at 0 and taken off only with a keyway;
  - the bore von Mises: `√(k_θ² + k_θ + 1)·p`, plane stress, σ_r = −p;
  - the cold and hot mismatch;
  - `δ_lo = min(Δ_c, Δ_h − α_part·Δϑ_rot·d·1000)` and `δ_hi = max(Δ_c, Δ_h)`, in `py_min` and `py_max` argument order;
  - the needed and elastic edges;
  - the key alone: three arms, matching `keyed` / `bearing` / else, a NaN bearing length giving +inf as the engine does;
  - the NIST shrink: linear from 265 K up, the Fe-2.25Ni fit below, anchored at 265 K, with a NaN shaft temperature
    falling to NaN in both;
  - the fittable edge and the fitting temperature, both on the measured maximum;
  - `hub_fit_allowed_C`: each of the four arms equals `py_min(py_min(yours, adhesive), onset − 10)` with ties to the
    earlier value, a NaN fails every arm, and an onset of +inf never binds;
  - the cup's allowed temperature: your limit alone.
- **The part-material cases against `PartProperties::hub` and `resolve`:**
  - steel with the default pick: the Materials inputs;
  - steel with another pick, or a non-magnetic pick: that pick's `engine` values;
  - otherwise the workbook's aluminium, E18's 68.9 GPa and 23.6e-6 /°C;
  - the yield: the pick's library value on steel or for a non-magnetic pick, else 6061-T6's 241 MPa;
  - a pick outside the choices: NaN on steel, aluminium off it.
- **The four assumption rationales at the engine.**
  - **Friction.** Holding is linear in μ, so "falls in proportion" holds.
  - **Smoothing.** It is taken off both the minimum and the maximum, so a finer finish raises the grip and the stress
    together (trap 19).
  - **Rotor over shaft.** The allowance enters only the hot end (`rotor_warmer_um` in `loosest_um`), as "the grip at
    the hot end falls" says. This is more precise than the input helps T2's SF-8 flagged.
  - **Joining clearance.** It is added to the measured maximum in the fitting temperature, and one value serves both
    joints (`JointCommon`).
- **The traps over all changed text.** None of these appear: "clamp", "safe", "strongest", "at most", "up to", "most
  expected", "heat the hub", "protect", "permanent", induction, the grade's 150 °C, a fit class, a ratio or efficiency,
  U+2192 or U+2300. "After warming and cooling" is trap 14's form. Neither path nor heading contains "clamp" or
  "gearbox". `plain_problems` is clean on the labels and rationales (tested).

**R-2. Table labels.**
- **Not drawn on screen.** The three new `TableField` labels are not drawn anywhere. Only `symbol` is read, by
  `render.rs:142` and `typeset.rs:797/1653`.
- **The yield label is consistent with the data file.** "(library: the published minimum where one exists)" matches
  `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json`:
  - guaranteed minima for 17-4PH (both), 304, 6061-T6, 316L and Ti-6Al-4V;
  - typical values where no minimum is published: 416 ("no minimum published"), and the 4140 Otai annealed row.
- **"(library)" means the entry's `engine` values.** On the modulus and expansion labels, "(library)" follows the
  existing σ, ρ and c labels, which also read the entry's `engine` values. For 4140 the engine's 12.3e-6 /°C differs
  from the sourced 12.2. The test pins the engine value, as the check uses it.

**R-3. Two "Shaft joints" headings, T3 to T4 only.**
- **Now.** The new chain heading and T2's package heading share the text. Until T5 puts `shaft_joint.check` on the
  dashboard (spec 5.1, index 15), the results table shows "Shaft joints" twice:
  - the chain, with 23 rows;
  - the package under "Other results", with the verdict alone.
- **After T5.** The package group empties and `groups_of` drops it (`retain(!rows.is_empty())`).
- **Impact.** The branch is not released before T10's stop, so no change is needed. `result_groups.rs`'s new test
  comment says the same.

**R-4. Sources (not G-3 text, spec 4.5 and 6.3; pinned by `tests/assumptions.rs`).**
- **Shown on screen.** The panel prints them ("Source: ...", `panel.rs:1268`). So "BL-064 shaft joint spec", the
  backticked file path and "§" appear on screen. This is by design: trap 22 covers labels and rationales, not sources,
  and the older sources show "Workbook ...!C..", "Addendum A3" and "M1 audit".
- **Not verified here, taken from the spec.**
  - "DIN 7190-1:2017 §8.2".
  - "smoothing 0.4·Rz per surface". This does not say radius or diameter. The research used 0.8·ΣRz on the diameter,
    from memory (`research-physics.md:168`). That figure agrees with the source only if 0.4·Rz is per surface on the
    radius.
  - Gear Technology Nov/Dec 2017 (0.145) and ANSI/AGMA 9003 (0.15).
- **Robustness of the rationale.** "5 µm is conservative for holding with ground and honed surfaces" holds under either
  reading: ground Rz ≈ 2.5 µm and honed Rz ≤ 3 µm give at most about 4.4 µm at 0.8·ΣRz.

**R-5. Symbols.**
- **No collisions.** The 18 new input symbols and 23 record symbols collide with none of the crate's 500.
- **Overlaps the module doc already tells apart:**
  - `σ_{BI}` (conductivity) beside `σ_{y,BI}` (yield) in one table;
  - `μ_{fit}` and `μ_{joint}`; `S_{fit}` beside the clamp's `S_{req}` and `S_{cl}`;
  - `p_{key,hub}` and `p_{key,cup}` beside the paused clamp's `p_{key,Al}`;
  - `ϑ_{lim,hub}` (your fitting limit) and `ϑ_{lim,i}` (the inner magnet limit).
- **Module doc claim checked.** "Elsewhere α is a Br temperature coefficient" is true: every α that is an expansion
  coefficient (`α_{part}`, `α_{st,in}`, `α_{BI}`) is new in this commit.

**R-6. The records module doc (a code comment).** It says the no-wall case is "a keyway as deep as the wall". The engine
tests `wall ≤ depth`: at least as deep. Not on screen.

**R-7. No model question arises.** No changed text asks to change the engine. R-1 (the stress side's smoothing) and R-14
(which body sits at the bond limit at the hot end) stay with T7's physics review, as planned. SF-2's wording is true
under either outcome of R-1.

### Re-review after a change by the implementer after its review (2026-10-19)

BL-064 Task 5 (commit 2d5c36a) put the shaft joint check on the dashboard (`gui/dashboard.rs`) with its guide
(`gui/guidance.rs`): the guide's label "Shaft joints", its Why, its Learn reason "No note covers the shaft joints yet.",
the Problem sentences `shaft_joint_problem` builds (the slip, stress, fitting and key sentences, the window and
empty-window sentences, and the not-judged sentences), the constants `SHAFT_JOINT_NOT_A_NUMBER` and
`SHAFT_JOINT_FALLBACK`, ruling C-4's safety-factor sentence, and `STORED_3D_NOTE`'s new sentence, with the doc texts
around them (the README, `docs/ai/02-system.yaml` and `docs/ai/03-structure.yaml`). These texts stay drafts until Task
8's G-3 review. They had a G-3 text re-review in a fresh context at HEAD 2d5c36a. The reviewer's report for the record
follows unchanged (full answer:
`.superpowers/sdd/2026-10-18-bl064-shaft-joint/textreview-T5-a-change-by-the-implementer-after-its-review-1.md`).

- Reviewer: fresh context, session model, 2026-10-19; G-3 text re-review of BL-064 Task 5's texts (round 1). There
  was no note text to review: the note `shaft_joint` arrives in Task 7 (the note dump names no note).
- Verdict: PASS, no must-fix. No reviewed text changed: none of these texts is reviewed text yet. No trap of spec 6.2
  is broken and both caps hold (the Why is one sentence of 29 words; the stored-3D note's new sentence has 18 words).
  Every number and every source a Problem quotes is the engine's at each of the 66 designs probed.
- Should-fix offered, none applied in this round; all five stay open for Task 8's G-3 review, which the answer asks to
  weigh SF-1 and SF-2 as must-fix candidates:
  - SF-1 (the cup's two no-wall sentences in `not_judged_sentences`, reachable on the sliders at a bore of 16 mm with
    a boss of Ø12): "is as wide as its boss" reads as equality beside two different numbers, where the engine's rule
    is "at least as wide"; and "it leaves no wall to grip the gearbox shaft" says the cup has nothing to grip with,
    while the web still surrounds the bore. Only the boss has no wall, and the check, which takes the boss for the
    whole length, cannot judge the joint. The cup's keyway sentence has the same issue. Offered: "The cup's bore, {coupling.bore_mm}, is at least as wide as its boss, {metal.boss_od_mm}, so the boss leaves no wall to grip the gearbox shaft; the check does not judge the web alone."
    and "The cup's keyway, {coupling.keyway_depth_mm} deep, cuts through the boss's wall around its {coupling.bore_mm} bore, so the boss has no closed ring to grip the gearbox shaft; the check does not judge the web alone."
    The `KeywayThroughWall` arm then splits by joint, and the inner hub's two sentences stay. With it comes an engine
    observation for Tasks 7 and 9, not a text fix: at a boss length of 0 with a boss narrower than the bore, the
    engine reads "no wall" while the web carries the fit (conservative: unknown, never a pass).
  - SF-2 (`SHAFT_JOINT_NOT_A_NUMBER`, at a typed bore of 0): the NaN is a term the check works out, not an input, so
    "a value it reads is not a number" sends the user looking for an input that is not a number. Offered: "The shaft joint check cannot be made: a value it reads or works out is not a number."
    That is 18 words; the answer's "15 words" is the count of today's text.
  - SF-3 (`window_sentence`'s empty-window edge clauses): "the bore stays within its limit" names the bore where the
    bore stress is meant, and "it can be fitted" can read as either subject before it. Offered:
    "its bore stress stays within the limit only up to {}" and "the {name} can be fitted only up to {}".
  - SF-4 (the Why): "it" takes "grip" as its antecedent, and a grip is not fitted. Offered (30 words, one sentence): "A shrink fit's grip holds each rotor on its shaft; each fit must carry cold-high with margin when loosest, stay elastic when tightest, and be fitted within its allowed temperature."
  - SF-5 (low; `STORED_3D_NOTE`'s new sentence): "fitting limit" is a third name for the allowed fitting
    temperature. Offered (19 words): "The bonded inner hub's allowed fitting temperature, on the shaft joints row, uses one of the reverse fields too."
    The note is now 3 sentences and 62 words; whether the Help cap binds this hover note is Task 8's call.
- No text changed in this round.

**Report for the record (as returned).**

**R-1. What was verified true, form by form, against `engine/shaft_joint.rs`.**
- **Slip.**
  - The holding is the effective minimum times the holding per µm, and the required torque is the safety factor times
    cold-high. "At its loosest" is the engine's loosest end.
  - A holding that reads equal to its need is quoted one last digit under it (`apart_from`, tested).
  - A floored effective interference reads "holds 0 N·m", the badge's 0.00.
- **Stress.** The bore's von Mises stress at the tightest is compared with yield / 1.2, quoted as "(415.0 MPa yield with
  a margin)" and, for 304, "(205.0 MPa yield with a margin)".
- **Fitting.**
  - The needed temperature is quoted against the allowed one, and the shaft is named with its set temperature
    (SF-12).
  - The source comes from the engine's own `hub_fit_allowed`, with the same arguments `joint_specs` passes: one rule,
    with its live number.
  - The onset and the margin subtract to the allowed temperature at every design probed:

    | Design | Onset | Less | Allowed |
    |---|---|---|---|
    | Library N42 parts | 69.83 | 10.00 | 59.83 |
    | N50M | 98.80 | 10.00 | 88.80 |
    | Mixed N42M/N35 | 91.35 | 10.00 | 81.35 |
    | 2214 Hi-Temp, hub limit 150 °C | 143.4 | 10.0 | 133.4 |
    | N52 | 81.77 | 10.00 | 71.77 |
    | Single-ring field 1500 | -2.45 | 10.00 | -12.45 |

    With mixed rings the quoted onset is the inner N42M's own; the outer N35's 59.93 °C is never quoted.
  - With ferrite magnets the onset is +inf and never binds.
- **Key.**
  - The three sentences match `joint`'s key arms and `key_has_bearing_area`. With no contact height, that sentence
    shows alone, as spec 5.2 asks (contact 0 with boss length 0).
  - The contact-height sentence quotes the typed input. That is the engine's capped contact wherever the key term
    applies, because a keyway depth of 0 or less has no key term.
- **Window.**
  - The needed, elastic and fittable edges are each independent of the band. A failing term always leaves its band
    outside the window, so the sentence never contradicts the band it quotes:
    - slip: holding < required ⟺ min < needed (required > 0);
    - stress: stress > limit ⟺ max > elastic;
    - fitting: needed temperature > allowed ⟺ max > fittable (α > 0).
  - Naming only the edge that binds is true, because a band below the smaller edge is below both.
  - On the sliders the needed edge is positive. The minimum magnet temperature is at most 20 °C, so the loosest end is
    at most 0, and no window can start at or below zero.
- **Empty window.** Its ends cannot read equal or crossed, because one is rounded up and the other rounded down, from
  numbers in strict order.
- **Not judged.**
  - The order follows spec 4.3: the determinate joint's causes and window come first, then the other joint's sentence.
    With both joints indeterminate, the inner hub comes first and no fix shows.
  - The inner hub's "reaches the flats" holds for any bore at or past the incircle, including the typed part diameter
    of 0 (PD-1).
- **Safety factor (ruling C-4).**
  - 0 reads "0" and -1 reads "-1.000", with no fix.
  - 0.5, typed, passes silently, as C-4 rules.
- **`STORED_3D_NOTE`.**
  - The inner onset follows `h_rev_single_ring_kA_m`.
  - The allowed temperature always reads that onset, so "uses one of the reverse fields" holds even where the onset
    does not bind.
- **The five probes.**
  - Four pass and show no Problem.
  - The 304 probe's Problem reads, in order: the two slips, the two stresses, then the two empty windows. It matches
    spec 3.6: 16.31 against 11.72 µm, and 15.80 against 11.74 µm.

**R-2. Traps.**
- **The word scan.** Over the 66 Problems, the label, the Why, the Learn reason and the note:
  - none of these words: "clamp", "heat" (outside the note's pre-existing "slip-heating"), "strongest", "at most",
    "most expected", "permanent", "backlash", "protect", "induction", a fit class, "ratio", "efficiency";
  - "safe" only inside "safety factor";
  - "up to" only for the interference edges ("only up to 11.72 µm"), never for cold-high. Trap 2 is about cold-high,
    and trap 18 endorses "up to" for holding.
- **Glyphs.** Only µ, ° and ·. The house minus sign is the ASCII hyphen. No U+2192 or U+2300.

**R-3. Length (no cap applies to a Problem).**
- The 304 probe's Problem runs to six sentences.
- Its two slip sentences each repeat "cold-high, 2.414 N·m, times the safety factor of 2.000".
- T8 may want a shorter second one, such as "At its loosest the cup holds 2.927 N·m, also less than that." It is not
  required.

**R-4. A trade-off line that names the label.**

> One input pulls two ways (Axial magnet length, both rings): Hot-low torque needs it raised and Shaft joints needs it lowered, so change another input to settle one of them.

- "Shaft joints needs" puts a plural label before a singular verb. The pre-existing "Temperature limits" label does the
  same.
- The label stands as a block's name, so this is left to T8 (C2 and C3).

**R-5. The inner hub's keyway sentence takes the flats' incircle.**
- That incircle is the engine's conservative wall. Where the keyway sits under a corner of the polygon, the real wall
  is thicker.
- The check reads unknown there, and the sentence states the engine's rule as fact.
- Leave it.

**R-6. Typed bores below the cases the spec names.**
- **Bore -5.** Both joints are judged and slip ("holds 0 N·m"). Never a pass.
- **Bore 0.5.** It reads:
  - slip;
  - a stress of 4624 MPa;
  - a needed fitting temperature of 4418 °C;
  - a key pressure of 2549 MPa.

  All are true to the model, and none is a pass.
- **Nothing to change.** Both are engine behaviour (T2), with no text change.

**R-7. No model question arises.**
- No changed text asks to change the engine.
- SF-1's observation is an engine observation, not a model change: the boss stands for the web, and the result is
  conservative.
- R-1 and R-14 stay with T7's physics review, as planned.

### Re-review after task-review fix round 1 (2026-10-19)

BL-064 Task 5's task-review fix round 1 (commit 6f3dbe0) applied the record above's SF-1 and SF-2 word for word, in
`gui/guidance.rs`. The cup's bore sentence in `not_judged_sentences` now reads "is at least as wide as its boss" and
"the boss leaves no wall". The `KeywayThroughWall` arm splits by joint: the cup gets its own keyway sentence, on the
boss's wall, and the inner hub's stays word for word. Both cup sentences end "the check does not judge the web alone".
`SHAFT_JOINT_NOT_A_NUMBER` reads "a value it reads or works out is not a number". The doc comments around them and the
README's `src/gui/guidance.rs` row changed with them. These texts stay drafts until Task 8's G-3 review. The changed
texts had a G-3 text re-review in a fresh context at HEAD 6f3dbe0. The reviewer's report for the record follows
unchanged (full answer:
`.superpowers/sdd/2026-10-18-bl064-shaft-joint/textreview-T5-task-review-fix-round-1-1.md`).

- Reviewer: fresh context, session model, 2026-10-19; G-3 text re-review of BL-064 Task 5's changed texts after its
  task-review fix round 1 (round 1). There was no note text to review: the note `shaft_joint` arrives in Task 7 (the
  note dump names no note).
- Verdict: PASS, no must-fix. Each of the three changed texts is true at every one of the 27 designs probed, and each
  says what the engine rule behind it decides. The inner hub's keyway sentence renders word for word as it did before
  the arm was split. No trap of spec 6.2 is broken, and no cap applies: the caps cover helps, Whys, note sentences and
  watch-outs, not Problem sentences. The changed string lines are ASCII only.
- Should-fix offered, none applied in this round; both stay open for Task 8's G-3 review:
  - SF-1 (the cup's two no-wall sentences): "the check does not judge the web alone" can also be read as "the check
    judges more than the web". Both readings are true, but the second hides the point, that the web's own grip is not
    judged. Offered: end both cup sentences with "the check does not judge the web's own grip." (8 words, as now):
    "The cup's bore, {coupling.bore_mm}, is at least as wide as its boss, {metal.boss_od_mm}, so the boss leaves no wall to grip the gearbox shaft; the check does not judge the web's own grip."
    and "The cup's keyway, {coupling.keyway_depth_mm} deep, cuts through the boss's wall around its {coupling.bore_mm} bore, so the boss has no closed ring to grip the gearbox shaft; the check does not judge the web's own grip."
    The whole-text pins at `guidance.rs:6917`, `:7071` and `:7076` move; the form words at `:3385` and `:3398` stay.
  - SF-2 (the cup's keyway sentence): it quotes the keyway depth and the bore but not the boss's diameter, the third
    number that decides the verdict, so a user cannot see why the wall is too thin. Offered, naming the wall by its two
    edges, with SF-1: "The cup's keyway, {coupling.keyway_depth_mm} deep, cuts through the boss's wall between its {coupling.bore_mm} bore and its {metal.boss_od_mm} outside diameter, so the boss has no closed ring to grip the gearbox shaft; the check does not judge the web's own grip."
    Without SF-1: "The cup's keyway, {coupling.keyway_depth_mm} deep, cuts through the boss's wall between its {coupling.bore_mm} bore and its {metal.boss_od_mm} outside diameter, so the boss has no closed ring to grip the gearbox shaft; the check does not judge the web alone."
    At boss 12 it renders 39 words. The arm adds `input_text(inputs, "metal.boss_od_mm")`, and the pin at `:7076`
    moves. The inner hub's keyway sentence cannot do the same: its wall runs to the flats' incircle, which is not an
    input (round 1's R-5).
- No text changed in this round.

**Report for the record (as returned).**

**R-1. What was verified true, claim by claim, against `engine/shaft_joint.rs`.**
- **`SHAFT_JOINT_NOT_A_NUMBER`: "a value it reads or works out is not a number".**
  - **When it shows.** Only when `shaft_joints` is `None` for `UNKNOWN_NOT_A_NUMBER`. `verdict` gives that when an
    input, a link, or a compared term of a judged joint is NaN (`:807-816`). "Reads" covers the first two cases and
    "works out" the third, so the sentence covers every case and claims nothing outside them.
  - **Bore 0.**
    - The pressure per µm is 1e-3/(0·…) = +inf.
    - The contact arc is max(0, π·0 − 3.0) = 0, or π·0 = 0 keyless.
    - So the holding per µm is f·inf·0·… = NaN.
    - Both parts keep a wall (boss 22 > 0, and the hub's incircle > 0), so both joints are judged and the NaN holding
      is a compared term. Keyed and keyless read alike.
  - **Probed:** bore 0 keyed and keyless, boss NaN, keyway NaN and adhesive 0. Each reads the sentence alone, with no
    fix, under the badge "Shaft joint check unknown: a value is not a number", which agrees with it.
  - **Length.** 18 words. No cap applies; round 1's record already corrected its "15 words".
- **The cup's bore sentence: "is at least as wide as its boss".**
  - **The rule.** It shows for `NoWall::BoreAsWideAsPart` on the cup, the engine's `!(boss_od_mm > bore_mm)`.
  - **No reversed numbers.** A NaN boss is a link, so it reads "not a number" first. Wherever this sentence shows, the
    bore is at least the boss numerically. The readout's rounding is monotonic, so the quoted bore never reads below
    the quoted boss. Probed:
    - 12.00 against 12.00;
    - 16.00 against 16.00 and against 12.00;
    - 10.00 against "0 mm" and against "-22.00 mm" (PD-1's typed bosses).
- **"so the boss leaves no wall to grip the gearbox shaft".** A bore at least as wide as the boss removes the boss's
  material all round, and a boss typed at or below zero has none.
- **"the check does not judge the web alone".**
  - `joint_specs` takes the boss's diameter over `boss_length_mm + web_mm`, so the check never judges the web by
    itself.
  - The web is the cup's rear wall and runs out to the cup's diameter (`model.rs:1256`), so it still surrounds the
    bore.
  - At boss length 0 (bore 16, boss 12) the web alone carries the fit, and the clause says exactly what is true.
- **The cup's keyway sentence.**
  - **The rule.** `(boss_od_mm − bore_mm)/2 <= keyway_depth_mm`, with a keyway (depth above 0), tested after the bore
    rule.
  - **The physics.** The depth is cut into the bore (its help), so the slot's floor reaches the bore's radius plus
    the depth, at or past the boss's radius. The keyway is through the boss's wall, and the boss is an open ring: "no
    closed ring" holds.
  - **The web.** Its wall reaches the cup's diameter and is far thicker than the keyway slider's top, 4.0 mm, so the
    web alone is still not judged here either.
  - **Where it shows:**
    - boss 12 (the slider's bottom) with the default 1.7 mm keyway: a 1.0 mm wall;
    - bore 16 with boss 17: 0.5 mm;
    - bore 16 with keyway 4.0: 3.0 mm;
    - a typed bore of 21: 0.5 mm.
  - **The rule's edge.** Boss 13.4 with bore 10 gives a wall of 1.7000000000000002 mm, above 1.7, and passes. Boss
    13.5 passes too.
- **The inner hub's keyway sentence.** At bore 16 with keyway 2.1, and with keyway 4.0, it renders "The inner hub's
  keyway, 2.100 mm deep, cuts through the wall around its 16.00 mm bore, so the fit has no closed ring to grip the
  wheel shaft." That is the old shared template with the inner hub's name and shaft.
- **Order and fixes, unchanged by the fix.**
  - The determinate joint's causes and window come first, then the inner hub's not-judged sentence, then the cup's.
  - With the cup's band also reversed, the cup's keyway sentence is followed by its band sentence.
  - With both joints indeterminate, or one joint indeterminate and the other passing, no fix shows.
  - Where the inner hub slips beside a cup with no wall, the inner hub's three slip fixes show (boss 12 with a hub
    minimum of 8; bore 16 with boss 12).

**R-2. Traps.**
- **Trap 20.** The clause names what is not judged and claims nothing about it is fine.
- **Trap 21.** No "clamp"; the texts say "grip", as the trap asks.
- **Trap 11.** The gearbox shaft is named only as the shaft gripped, with no advice about it.
- **Trap 22.** The texts are ASCII only.
  - "web" is the input's own label word ("Integral steel rear web thickness"), and the guide already says "its boss
    and web".
  - "boss" is the input's own label word ("Shaft boss outside diameter").
- **The other traps.** None applies: the texts quote no torque, temperature, fit class or interference.
- **The word scan.** None of these is in the three texts: clamp, heat, safe, protect, strongest, at most, most
  expected, permanent, backlash, induction, up to.

**R-3. Spec 5.2's drafts still read the old words (`2026-10-17-bl064-shaft-joint-design.md:707-714`).**
- **Not drift to fix in this round.**
  - The spec has not been edited since its approval (`0687034`).
  - Section 5.2 calls these texts drafts whose final words the G-3 review sets.
  - Round 1's "Spec 5.2's drafts follow" was not carried out, and it need not be.
- **For Task 8.** Its G-3 record should quote the words as shipped.
- **For Task 9.** Its sweep may note the difference from the spec.
- **Historical copies.** The plan's text (`:11395-11397`, `:11667-11685`), the T5 brief and round 1's record also
  keep the old words. All three are historical, so leave them.

**R-4. The badge is not changed text (T8 row 3).**
- `UNKNOWN_NO_WALL` reads "Shaft joint check unknown: no wall around the bore (cup)". It states the same model fact
  that round 1's SF-1(b) addressed in the Problem.
- After "unknown" it reads as the check's reason, and the Problem under it now says the web's grip is not judged, so
  the two read together.
- No change recommended.

**R-5. Round 1's R-5 stands for the inner hub's keyway sentence.** The sentence treats the flats' incircle as the
wall, which is the engine's conservative wall. Its words did not change in this fix. Leave it.

**R-6. A typed bore at or below zero is never named.**
- Bore 0 reads "not a number". Bore -5 is judged and slips, reading "holds 0 N·m" (round 1's R-6).
- Both are off the bore slider (4-16 mm), and neither reads a pass.
- A sentence that names the bore would be a new form. That is the owner's call, and not a fix for this round.

**R-7. Code checks in passing (not text).**
- The split keyway arm hard-codes "wheel shaft" and "gearbox shaft", as the bore arm already did.
- `JointSide::shaft` is still read by the fitting sentence (`guidance.rs:2039`), so the split leaves nothing orphaned.
- The doc comments at `:346-349`, `:1960-1961` and `:2233-2235`, and the README row, describe the new texts
  correctly.

**R-8. No model question arises.** No changed text asks to change the engine.
