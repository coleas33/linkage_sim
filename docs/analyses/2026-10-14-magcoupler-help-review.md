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
