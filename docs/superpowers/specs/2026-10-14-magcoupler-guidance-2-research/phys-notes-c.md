# Physics review, notes-c: the six A5 warning notes

Scope: `a5.non_ferromagnetic_back_iron`, `a5.ferromagnetic_sleeve_or_liner`,
`a5.high_conductivity_sleeve_or_liner`, `a5.low_saturation`, `a5.uncoated_low_alloy_steel`,
`a5.cte_mismatch_with_magnets` (`magcoupling-rs/src/engine/explain/notes.rs:403-538`).
I read the rules (`engine/warnings.rs`), how they are fed (`engine/api.rs:324-340`), the library
(`engine/material_library.rs`), the inputs and results (`materials.rs`, `metal_design.rs`,
`model.rs`, `temperature.rs`), the records (`explain/records/{torque,dashboard,slip_heating,temperature}.rs`),
the GUI links (`gui/guidance.rs:182-188`, `gui/dashboard.rs`) and the audit's findings TC-7, TC-9,
TC-10, TC-17, TC-21, TC-26, PA-1, PA-5, PA-6, PA-9, PA-10, PA-13, PA-15, CG-11 and the dropped TC-6.
Read-only: I ran no engine code. The numbers come from the audit's verified probes and from
`tests/data/reference_values.json` (the workbook basis).

Labels quoted below are the current metadata labels. If part C renames them, the notes must follow.
See D5.

---

## 0. Cross-cutting findings (all six notes)

**X1. Which picks fire which rule.** This comes from the library plus `warnings::compute`. The notes
should say it, and tests should pin it (D8).

| Rule | Fires for | Never fires for |
|---|---|---|
| non_ferromagnetic_back_iron (red) | Back iron material = 304 or 6061; or Back iron = "no back iron" with any pick | 4140, 1018, 12L14, 416, 17-4PH, with Back iron = steel circuit |
| ferromagnetic_sleeve_or_liner (red) | **nothing**: all four Sleeve and liner choices are non-magnetic (`material_library.rs:744-749`; `gui/dashboard.rs` test comment "no design fires that warning") | everything |
| high_conductivity_sleeve_or_liner (amber) | Sleeve and liner = 316L **and** "316L conductivity" typed above 1.35e6 S/m (the threshold equals that input's default) | Ti-6Al-4V (0.595e6), Inconel 625 (0.775e6), PEEK |
| low_saturation (amber) | steel circuit + 12L14, 416, 17-4PH H1150 or H900 (no design value of their own; 416 also has a sourced 1.6 T) | 4140 (its value is the input), 1018 (1.7 T library value) |
| uncoated_low_alloy_steel (amber) | steel circuit + 4140, 1018 or 12L14 + Plating thickness per surface = 0 | stainless picks; the default 0.015 mm |
| cte_mismatch_with_magnets (amber) | 304 hub (17.7), 6061 hub (24.4), Back iron = no back iron (aluminium body, 24.4); a typed steel Expansion coefficient above 14.2e-6 | every listed steel (largest is 4140 at 13.1) |

So with plating above zero, **4140 and 1018 are the only back irons that fire no warning**. 416, 17-4PH
and 12L14 trade the plating caution for the low-saturation caution. This feeds trap N8.

**X2. None of these effects changes the torque the calculator shows, except the circuit switch.**
The torque path reads only k, the magnet thicknesses and the gap (`model.rs:593-603`). Saturation,
wall thickness, shell permeability, conductivity and plating never enter it. Two notes say "the
torque falls" (low_saturation s3, ferromagnetic_sleeve s2-s3; PA-9 and the latent TC-6). Each must
also say that the calculator does not show that loss.

**X3. Sources line.** Users see it (explorer.rs:583-593). It carries audit ids and plan names in all
six notes: "M1 audit M5", "E15 to E17", "Plan A-3 Task 16 physics review", "M1 audit P1", "M1 audit E1",
"Addendum A decision 16 (E18)", "Workbook Materials C26" (TC-9). Keep the field for the gate, but
render it collapsed or only in the review sheet.

**X4. Glyph.** The superscript minus (U+207B) renders as "10¯⁶" (`gui/typeset.rs:271-278`).
Only the CTE note uses it, three times. Write "µm/m per °C" or "millionths per °C" instead.

**X5. Symbols.** A bare σ for conductivity collides with the shear stress σ_n of the torque records
(PA-15; high_conductivity s1). In prose, say "conductivity". The record already calls it σ_{sl}.

---

## 1. `a5.non_ferromagnetic_back_iron` (red warning; equations: `materials.circuit_backiron`)

**(1) What the engineer needs.** Keep magnetic steel behind the magnets (4140 or 1018) unless you
accept a large torque loss, a stray field and a bond-stress caution. If you go non-magnetic on
purpose, size the magnets on the no-back-iron torque.

**(2) Problem statements**
- s2 (74 words, TC-10): "the torque drops by more than 40 % (2.69 to 1.54 N·m)" holds only at the
  default design. The no-back-iron to steel ratio depends on the gap against the pole pitch and on
  the magnet thicknesses. Every number in the sentence is a result or an input: 2.69, 1.54 and 1.70
  are results; 0.95 is the editable assumption "Original calibration coefficient" (PA-10); 1.049
  derives from the bench input. The dashboard shows 2.69 -> 1.70 (-37 %), not "more than 40 %".
- s2's calibration clause is correct for the default, but it never states the rule. The measured
  factor applies only when the circuit is free space **and** the pole count equals the prototype's
  **and** both parts are B842SH (`records/torque.rs:84-87`). Without the rule, a reader can generalize
  it: spec section 8 trap "bench calibration applies beyond prototype-matching designs".
- Missing: the "Back iron" switch (Calculator selector) set to "no back iron" fires the same warning
  with a steel pick (`warnings.rs:145-147`, `material_library.rs:801-804`). The note names only
  304 and aluminium.
- Missing consequences: the hub, cup and boss become that material (mass, heat capacity, slip losses
  by E17); the cup-wall check reads "No back iron" (no wall rule); the expansion-mismatch caution
  always fires (X1).
- Missing T6 trap (TC-14, synthesis T6): without back iron, moving off the prototype's pole count or
  magnet part drops the factor from measured to assumed. That steps the shown torque by about 10 %,
  and the step is not physics.
- s3 "strong stray field": qualitatively right, but the calculator does not compute it. Say so.
- Terms used before definition: "relative permeability", "free-space circuit", "geometry factor",
  "sinh form" (lives in back_iron_factor; no link, TC-26), "bench calibration".
- s1 duplicates back_iron_factor s2 (TC-26, DRY).
- The audit's suggested fix "pick 4140, 1018 or 416" (TC-7, TC-10) is a trap: 416 trips
  low_saturation (spec section 8).
- The result label "Same layout with steel back iron (factor 0.95)" hard-codes an editable factor
  in its label. That is for part C, not the note.

**(3) Minimum facts a rewrite must keep**
- Steel behind each ring returns the flux pole to pole, so more of it crosses the gap and makes
  torque. Without it, the flux spreads behind the magnets.
- 304 and aluminium are non-magnetic (literal OK). Picking either, or Back iron = no back iron,
  selects the no-back-iron model.
- The torque loss is design-dependent. **From results**: `model.pullout_iron_Nm` ("Same layout with
  steel back iron") against `model.pullout_noiron_Nm` ("Raw no-back-iron prediction, same layout"),
  both computed for every design. Never quote a fixed share.
- Calibration switch, only for a prototype-matching design. **From results**: `model.f_cal`.
  Never a literal 0.95 or 1.049.
- Stray field outside the coupling attracts ferrous debris. Not computed.
- It also fires the expansion-mismatch caution, and the wall check stops applying.

**(4) What to change**

| Input (on-screen) | Direction | Effect | Trade-off |
|---|---|---|---|
| Back iron material (hub, cup and boss) | -> 4140 annealed or 1018 hot rolled | steel circuit back; full torque | heavier; needs plating (keep "Plating thickness per surface" above 0); steel slip losses. Not 416, 17-4PH or 12L14: they trip the low-saturation caution |
| Back iron (Calculator selector) | -> steel circuit | same as above when a steel is picked | as above |
| Overall length / "Axial magnet length, both rings" | raise (if you stay non-magnetic) | torque about proportional to active length | more magnet mass and cost, longer coupling |
| Outside diameter (Size tab) | raise (if you stay non-magnetic) | larger ring radius, more torque | larger envelope; cap and cup grow |

**(5) Draft rewrite**

Title: **No magnetic steel behind the magnets**
1. Magnetic steel behind each ring carries flux from one pole round to the next, so more of the flux crosses the gap and makes torque.
2. 304 stainless and aluminium are non-magnetic, like air: picking either as Back iron material, or setting Back iron to no back iron, selects the no-back-iron model.
3. That model loses a large share of the torque; the results show both for your layout: 'Same layout with steel back iron' and 'Raw no-back-iron prediction, same layout'.
4. Without steel the field also spreads outside the coupling and pulls in steel chips; the calculator does not compute this stray field.
5. To restore the torque, pick 4140 or 1018; to stay non-magnetic, raise Overall length or Outside diameter, and expect the expansion-mismatch caution.

Watch-out: Only a design matching the bench prototype (its pole count, B842SH on both rings) gets
the measured calibration; moving off it steps the shown torque by the ratio of the two factors.

Live-number variant of s3, if placeholders land (TC-8 A): "...: at your layout {model.pullout_iron_Nm}
with steel, {model.pullout_noiron_Nm} without." Caveat: after the switch, the second value already
carries the bench factor for a prototype-matching design. The watch-out covers that.

Keep `materials.circuit_backiron` and `Diagram::FluxPathBackIron`. Consider owning
`model.pullout_noiron_Nm`, which no note owns today (D4). Add see_also to `back_iron_factor` and
`calibration` if TC-26 lands.

---

## 2. `a5.ferromagnetic_sleeve_or_liner` (red warning; equations: none)

**(1) What the engineer needs.** Keep the sleeve and liner non-magnetic. Every listed choice is.

**(2) Problem statements**
- **Unreachable.** The rule can never fire (X1), and the note has no equations, so no user sees it.
  The audit dropped TC-6 for this reason. Rewriting it costs a re-review for no user benefit today.
  Do it only inside the batch re-review, and keep it short.
- s2-s3 imply the shown torque drops (warning text: "so torque drops"). The torque model has no
  sleeve permeability term (X2). This is latent while the rule cannot fire.
- s3 "B_sat · t per unit length": B_sat is undefined, and "per unit length" means per axial length.
  The physics is right. The cap compares with the half-pole flux B_gap τ_p / π, so the share lost
  also grows with **more poles**, not only with weaker magnets. That follows from the same
  formula, and the note omits it.
- s3 "the retainers are non-magnetic 316L, titanium, Inconel or PEEK" reads as a design fact. It
  means "the list offers only non-magnetic materials".
- Sources quote "Plan A-3 Task 16 physics review" (X3).

**(3) Minimum facts.** The shells sit in the gap that the torque flux must cross. A magnetic shell
lets flux short-cut pole to pole on the same ring. Saturation limits the short cut, so the loss
grows with shell thickness and is a larger share for weaker magnets or more poles. All listed
choices are non-magnetic. The torque model has no term for it. No numbers are needed.

**(4) What to change.** "Sleeve and liner material": keep a non-magnetic choice (all four are).
"Inner rotating retaining sleeve thickness" and "Outer rotating keeper liner thickness": thinner
would shunt less if the shell were magnetic. Trade-off: retention strength, which the calculator
does not check (metal_design.rs:208-210, "Retention qualification: Open").

**(5) Draft rewrite**

Title: **Magnetic sleeve or liner**
1. The sleeve and liner sit in the magnetic gap, which all torque-making flux must cross from ring to ring.
2. A magnetic shell lets flux short-cut from one pole to the next on the same ring, so that flux never reaches the other ring.
3. Saturation limits the short cut, so thicker shells, weaker magnets and more poles lose a larger share of torque.
4. All listed Sleeve and liner materials are non-magnetic, so this warning cannot fire; the torque model has no term for a magnetic shell.

Watch-out (needs a source; the Carpenter "magnetic properties of stainless steels" page already
cited at `material_library.rs:217,361` likely covers it): Cold working can make austenitic
stainless such as 316L slightly magnetic; specify annealed sleeve and liner stock.

Equations: none. No record exists.

---

## 3. `a5.high_conductivity_sleeve_or_liner` (amber; equations: `temperature.slip_loss.sleeve_W`, `liner_W`)

**(1) What the engineer needs.** Choose the sleeve and liner material and thickness with slip heating
in mind: lower conductivity and thinner shells heat less. The caution only flags a 316L
conductivity typed above its default.

**(2) Problem statements**
- s1 "sit in the strongest alternating field": that is true of the **stored** inputs "Opposite-ring
  field at the inner sleeve (fundamental)" (0.416 T) and "...at the outer liner" (0.419 T), against
  0.21 T at the hub and cup. These are 3D values for the default design. They never follow the gap,
  pole count, radius or grade (PA-1, synthesis T6, `temperature.rs:236-241`, `:1169-1170`). The note
  implies the field belongs to the user's design. This is the slip-loss twin of spec section 8
  bullet 6.
- s1 "times an end factor of 0.7": that is the editable input "End factor for thin shells and the
  cap" (PA-10).
- s1 uses a bare σ, "ω_s" and "per unit area" (X5, PA-15). The equation panel already shows the formula.
- s2 "(1.35 × 10⁶ S/m)": the threshold is a code constant (`SLEEVE_SIGMA_BASELINE_S_M`) equal to
  the **default** of the "316L conductivity" input, and it does not follow that input. So the caution
  fires on any upward edit of "316L conductivity", even a better measurement of real 316L.
- s3 "a typed conductivity can": name the input, "316L conductivity", which counts only while
  Sleeve and liner material = 316L (`api.rs:167`).
- Missing scale. At the workbook example the shells make about 9 % of the slip loss: 0.075 + 0.181 W
  of 2.75 W (`reference_values.json` C126-C130; the Rust total is from the PA-1 probe). The cup and
  cap dominate. Read this from results; do not quote the literal.
- Missing levers: the loss is proportional to thickness and to the square of slip speed.
- Result labels "Inner 316L sleeve" and "Outer 316L liner" keep "316L" with a Ti or PEEK pick. That
  is for part C.
- Physics check, OK: the resistance-limited thin-shell form holds over the whole input range. The
  shell reaction μ0·σ·t·v is about 5e-4 at the defaults and at most about 0.1 at σ = 1e7 S/m,
  t = 1 mm and 6000 rpm. No trap.
- Sources quote "M1 audit P1" (X3).

**(3) Minimum facts.** During slip the shells cut the other ring's field, and eddy currents heat
them. Heat is proportional to conductivity and thickness, and to the square of slip surface speed
and of the field. The caution compares the sleeve conductivity in effect with 316L's 1.35e6 S/m
(literal OK: a code constant). Ti, Inconel 625 and PEEK all conduct less (library literals OK).
More heat means a higher steady temperature and a shorter time to the limit. The field values are
stored for the workbook example. **From results**: `sleeve_W`, `liner_W`, `total_W`;
"Continuous slip: steady rise, estimate/high case"; "Continuous slip, time to the limit (high case)".

**(4) What to change**

| Input | Direction | Effect | Trade-off |
|---|---|---|---|
| Sleeve and liner material | -> Ti-6Al-4V, Inconel 625 or PEEK | lower conductivity, less shell heat (PEEK about zero) | cost and machinability (Ti, Inconel); PEEK strength and temperature limit; strength not checked here |
| Inner rotating retaining sleeve thickness / Outer rotating keeper liner thickness | lower | loss proportional to thickness; also opens running clearance (the clearance guide's first fix) | retention strength not checked ("Retention qualification: Open") |
| Relative slip speed / Slip duration per event | lower (if the duty allows) | shell loss goes as speed squared; less energy per event | set by the application, not the design |
| 316L conductivity | enter the real value | the caution fires above 1.35e6 | none; it is data, not a design lever |
| Opposite-ring field at the inner sleeve / outer liner | enter your own 3D values when the design differs from the example | loss goes as field squared | needs a field solution |

**(5) Draft rewrite**

Title: **Sleeve or liner that conducts better than 316L**
1. When the coupling slips, the sleeve and liner cut through the other ring's field, and eddy currents heat them.
2. That heat rises in proportion to their conductivity and thickness, and with the square of the slip speed and of the field they see.
3. This caution fires when the sleeve conductivity in effect exceeds 316L's 1.35 million S/m; today only a raised '316L conductivity' entry does that.
4. Titanium, Inconel 625 or PEEK, or a thinner shell, cuts the loss; the calculator does not check the shell's strength.
5. Compare 'Inner 316L sleeve' and 'Outer 316L liner' with 'Total estimated slip loss' to see whether the shells matter for your design.

Watch-out: The field at each shell is a stored 3D value for the workbook example design; it does not
follow your gap, magnets or pole count.

Equations: keep `sleeve_W` and `liner_W`. **Add `materials.sleeve_sigma_S_m`** (record σ_{sl},
`records/slip_heating.rs:47`), which no note owns and which is exactly what the rule compares (D3).

---

## 4. `a5.low_saturation` (amber; equations: `model.bsat_T`, `model.backiron_needed_mm`; also `CUP_WALL_GUIDE.learn`)

**(1) What the engineer needs.** If the back-iron steel has no design flux density of its own
(12L14, 416, 17-4PH) or saturates below 1.7 T, set a design flux density its data supports and
thicken the walls to match, or use 4140 or 1018.

**(2) Problem statements**
- **Framing (TC-17, CG-11).** This note is the "Why" of the red cup-wall check on the default design
  (`guidance.rs:188`). It owns `model.backiron_needed_mm`, so Explain shows it on every design under
  the title "Low saturation", although the warning does not fire. The wall rule belongs in its own
  note (one of the three candidate new notes). See D1 and D2.
- s1 "t = B_gap τ_p / (π B_des)": B_gap and τ_p are undefined. B_gap is taken at the "Operating
  magnet temperature" remanence (`model.rs:1032-1039`). For the wall note: the rule is not a
  worst case across temperature. Cold magnets push about 11 % more flux (-40 °C against 50 °C at the
  defaults). That case matters only for the cold torque, not for the hot-low check, so it is not a
  safety trap. Just do not call the rule "worst case".
- s2 (57 words) mixes three ideas and says "workbook". "1.5 T for annealed 4140" is the editable
  input "Design flux density for the back-iron check" (T6, PA-10). The reader asks why 4140 at 1.5 T
  is not flagged against 1.7 T. Answer: the rule compares a **sourced saturation** (4140 has none)
  or the **absence** of a design value (4140's value is the input). Do **not** write "4140
  saturates well above 1.7 T"; the library has no source for it (TC-17 corrected claim).
- The results row the note explains is labelled "Back-iron saturation flux density" (Calculator C36)
  but shows the design value (PA-5). Fix the label in part C, not in note text.
- s3 "flux leaks out of the circuit and the torque falls": the physics is true, but the calculator's
  torque assumes ideal steel (PA-9, X2). A thin or low-saturation wall shows only as a failed wall check.
- The watch-out "12 to 16 % thin ... (M1 audit M1)" quotes an audit id and holds at the default
  design only. The error grows with pole count (about 19 % thin at 14 poles) and turns conservative
  at 6 (PA-6 check). It belongs in the wall note.
- The note never names which steels fire (X1), so the user cannot predict it. That invites trap N8.
- Sources quote "Addendum A decision 20" and "M1 audit M1 and M2" (X3).

**(3) Minimum facts**
- The walls are sized at a design flux density, a working level below saturation. The needed wall
  is inversely proportional to it, which is exact from the formula. Literal OK as words.
- The caution fires for a sourced saturation below 1.7 T (literal OK: code constant
  `LOW_SATURATION_T`, 1018's design value) or no design value of its own. In the list: 12L14, 416,
  17-4PH H1150 and H900 (literal OK; pin with a test, D8).
- For those, the check uses the "Design flux density for the back-iron check" input: 1.5 T **by
  default**, or `{model.bsat_T}` live.
- 4140 and 1018 do not trip it.
- The torque model assumes ideal steel.

**(4) What to change**

| Input | Direction | Effect | Trade-off |
|---|---|---|---|
| Design flux density for the back-iron check | lower, to what the steel's data supports (17-4PH data reach only about 1.07 T at 140 Oe, `material_library.rs:306`) | the wall check becomes honest; the needed wall rises as 1/B_des | the check may then fail |
| Minimum outer return ring wall | raise until the cup-wall check passes | more steel to carry the flux | cup OD grows by twice the change (the cup-wall check's help); mass |
| Back iron material | -> 4140 annealed or 1018 hot rolled | clears the caution | needs plating (rust); 4140 and 1018 are not stainless |
| Number of poles per ring (Size tab: Magnets per ring) | more poles | shorter pole pitch, thinner wall needed (10 -> 8 poles needs +25 %, CG-11) | the rule's own error grows with pole count; pull-out changes (see the torque notes) |

**(5) Draft rewrite.** This assumes the wall rule moves to the new wall note, as recommended.

Title: **Back-iron steel with low or unknown saturation**
1. The wall check sizes the cup and hub walls for a design flux density, a working level set below the steel's saturation.
2. This caution fires for a steel that saturates below 1.7 T or has no design value of its own: in the list, 12L14, 416 and 17-4PH.
3. For those steels the check falls back on 'Design flux density for the back-iron check' (1.5 T by default), which the steel may not reach.
4. Lower that entry to what the steel's data supports; the needed wall rises as it falls, so then raise 'Minimum outer return ring wall'.
5. 4140 and 1018 carry their own design values and do not trip this caution.

Watch-out: The torque shown assumes ideal steel: a saturating wall shows up only as a failed wall
check, while the real torque would fall below the prediction.

**Fallback, if no separate wall note is made.** Retitle the note "Back-iron wall and saturation".
Replace s1 with two sentences: "The cup and hub walls carry each pole's flux round to its neighbour,
at a design flux density set below saturation." and "The wall needed grows with the gap flux density
and the pole pitch, so fewer, wider poles need more wall." Keep s2 to s4. Use two watch-out lines:
ideal steel, and "the wall rule reads thin at the workbook example (about 12-16 %), more so at
higher pole counts". That is 6 sentences, the cap.

Equations: recommended split. `model.bsat_T` stays here (the "what value fits my steel" question).
`model.backiron_needed_mm`, plus the unowned `materials.cup_wall_check` and
`materials.cup_wall_suggested_mm`, go to the wall note. Repoint `CUP_WALL_GUIDE.learn` (D1, D2).

---

## 5. `a5.uncoated_low_alloy_steel` (amber; equations: none)

**(1) What the engineer needs.** A carbon or low-alloy steel back iron needs a plating thickness
entered, and the machining allowances follow it. Switching to stainless avoids plating but trips
the low-saturation caution.

**(2) Problem statements**
- s1 "the stainless grades in the list (416, 17-4PH, 304) resist it": overstated for 416
  (free-machining, the least corrosion-resistant of them). 304 is not a usable back iron at all: it
  is non-magnetic and fires the red warning. Say they "need no plating here".
- s2 "The workbook therefore plans": jargon (part C). "adds about 0.015 mm per surface" is the
  editable default of "Plating thickness per surface" (T6). "Non-magnetic as plated" is true for
  high-P EN, but it does not matter to the user: on a steel surface a thin magnetic layer would
  simply join the steel. Drop it to save words.
- s3 "With no plating thickness entered": name the input and the condition, "Plating thickness per
  surface" set to 0 (TC-21). The rule tests `<= 0`.
- Missing: the plating thickness drives four results, "Machine the hub flats under by", "Machine the
  cup pockets over by", "Machine bores over (on diameter)" and "Machine outside diameters under (on
  diameter)" (`materials.rs:258-261`). That is the practical reason the number matters.
- Missing: the rule fires only with the steel circuit in effect and for 4140, 1018 or 12L14 (X1).
- Trap: offering 416 or 17-4PH as the plating-free fix without saying that they trip low_saturation
  (spec section 8, bullet 5).
- Sources quote "Workbook Materials C26" (X3).

**(3) Minimum facts.** 4140, 1018 and 12L14 rust, so they need a coating. The design plans
high-phosphorus electroless nickel, typically 0.013-0.025 mm per surface (literal OK: the input's
sourced help). The caution flags a zero thickness. The thickness sets the machining allowances.
The stainless picks need none but trip low_saturation, and 304 kills the steel circuit. No result
numbers are needed. Do not quote 0.015 except as "by default".

**(4) What to change**

| Input | Direction | Effect | Trade-off |
|---|---|---|---|
| Plating thickness per surface | set within 0.013-0.025 mm | clears the caution; sets the machining allowances | plating cost; tolerance stack on fits |
| Back iron material | -> 416 or 17-4PH | no plating needed | trips the low-saturation caution; lower or unsourced flux capability; 17-4PH much lower induction |

**(5) Draft rewrite**

Title: **Unplated carbon or low-alloy steel**
1. 4140, 1018 and 12L14 rust in humid air, so a back iron made of them needs a coating.
2. The design plans high-phosphorus electroless nickel, typically 0.013 to 0.025 mm per surface, and flags 'Plating thickness per surface' set to zero.
3. That thickness also sets the machining allowances in the results, such as 'Machine the hub flats under by'.
4. The stainless back irons, 416 and 17-4PH, need no plating here but trip the low-saturation caution; 304 removes the steel circuit altogether.

Watch-out: This caution checks only that some thickness is entered; it does not judge whether the
coating suits your environment.

Equations: none. The four machining-allowance results have no records. Adding them is optional and
outside notes.

---

## 6. `a5.cte_mismatch_with_magnets` (amber; equations: none)

**(1) What the engineer needs.** If the hub expands much more than the magnets (aluminium or 304),
check the bond's peak end shear and reduce it with a steel hub, a thicker bondline or a softer
adhesive.

**(2) Problem statements**
- s1 "barely changes across its magnetization": ambiguous (TC-21); it means in the bond plane.
  "-0.8 × 10⁻⁶ /°C" is the editable input "NdFeB expansion in the bond plane" (T6), and it renders as
  "10¯⁶" (X4).
- **PA-13 (must state).** That NdFeB value is used for **every** grade, because E22 has not landed
  (no E22 in `deviations.rs`). Ferrite and SmCo rings expand far more in that plane, so they get
  false cautions and an overstated shear. If E22 lands, this sentence must change in the same commit.
- s2 "glued to the hub": correct, and the screen covers only the inner blocks on the hub
  (`temperature.rs:1063-1085` uses the inner block's dimensions). The outer blocks, bonded in the cup
  pockets ("Outer magnet back bondline"), see the same mismatch and are **not screened**. Omitted.
- s3 "The larger the mismatch and the temperature swing, the higher that shear, so a hub ... is
  flagged": the "so" implies the flag accounts for the swing. It does not. The rule is only the
  expansion difference above 15e-6 (`warnings.rs:155`). The swing, bondline, adhesive stiffness and
  part stiffnesses enter only the peak-shear results.
- s3 "the Volkersen screen puts a number on it": names a method, not a row (TC-9, TC-21). The rows
  are "Peak end shear, current bondline", "Peak end shear, recommended bondline" and "Reading"
  (current bondline against "Selected adhesive lap shear at 22 °C (TDS)").
- Missing T6 trap: "Adhesive shear modulus" (0.107 GPa, AA 326's by E1) does **not** follow "Selected
  adhesive (code)". The adhesive table has no modulus (`temperature.rs:150-190`). Switching adhesive
  without re-entering the modulus leaves the shear computed for AA 326.
- Missing: which hubs fire (X1). In practice this means "an aluminium or 304 hub, or no back iron".
- Sources quote "M1 audit E1" and "Addendum A decision 16 (E18)" (X3).

**(3) Minimum facts**
- Heat grows a part by its expansion coefficient times ΔT. In the bond plane NdFeB barely expands
  (default input, quote as "barely"); steel is about 12 and aluminium about 24 µm/m per °C (literal OK).
- The glue line carries the difference as shear, peaking at the block ends.
- The caution fires above a 15 µm/m per °C difference (literal OK: code constant): an aluminium or
  304 hub, or no back iron. No listed steel fires it.
- The bond check is "Peak end shear, current bondline" against the lap shear ("Reading"), which also
  counts the swing and the glue. **From results**, never literal.
- The NdFeB value is used for every grade.

**(4) What to change** (directions from `volkersen_peak_shear_MPa`: peak shear ≈ Δα·ΔT·√(G/(η·S))·tanh(λL/2), with S = 1/(E_m t_m) + 1/(E_h t_h))

| Input | Direction | Effect | Trade-off |
|---|---|---|---|
| Back iron material / Back iron | -> a steel (4140 or 1018) | Δα 24.4 -> 13.1 µm/m per °C; the largest lever (workbook E18 row: 73.9 -> 46.1 MPa) | steel hub is heavier; needs plating |
| Inner magnet back bondline | raise (range 0.01-0.2 mm) | peak shear falls about as 1/√bondline | the hub wall thins (the apothem includes the bondline); more glue between magnet and steel (the torque model does not count it); bondline control in production |
| Adhesive shear modulus | lower (a softer adhesive), and **re-enter it whenever "Selected adhesive (code)" changes** | peak shear falls about as √G | softer adhesives are often weaker or rated cooler |
| Minimum magnet temperature / Selected adhesive (cure temperature) | narrow the swing from the cure temperature | shear proportional to swing | set by the duty and the adhesive |
| NdFeB expansion in the bond plane | for ferrite or SmCo rings, enter that grade's in-plane value (until E22) | removes false cautions | needs the grade's data |

**(5) Draft rewrite**

Title: **Hub expands much more than the magnets**
1. Across its magnetization, in the plane of the glue joint, NdFeB barely expands with heat; steel grows about 12 and aluminium about 24 µm/m per °C.
2. The glue line between each inner magnet and the hub takes that difference as shear, highest at the ends of each block.
3. This caution fires above a 15 µm/m per °C difference: an aluminium or 304 hub, or Back iron set to no back iron.
4. The caution ignores the temperature swing and the glue; 'Peak end shear, current bondline' and 'Reading' check the bond itself.
5. To lower that shear, use a steel hub, a thicker 'Inner magnet back bondline' or a softer adhesive ('Adhesive shear modulus').

Watch-out: The screen uses 'NdFeB expansion in the bond plane' for every grade, so ferrite and SmCo
rings, which expand more, can be flagged falsely.

Second watch-out candidates, if the parent allows one more line or uses s5's slot: "The outer
magnets, bonded in the cup, see the same mismatch but are not screened." Or: "'Adhesive shear
modulus' does not follow the selected adhesive; re-enter it when you change adhesive."

Equations: none today. There is no record for the peak-shear rows; only
`temperature.mismatch.cold_limit_C` exists, which is trivial. Adding a Volkersen record would let
this note own it. That is outside notes.

---

## 7. Traps

**From slice-1 spec section 8 that bite these notes**
- "The bench calibration applies beyond prototype-matching designs." Applies to non_ferromagnetic
  s2. The rewrite states the rule.
- "416 stainless is a safe back-iron fix." Applies to non_ferromagnetic (TC-7 and TC-10 suggestions)
  and uncoated_low_alloy_steel. Extend it to 12L14 and 17-4PH (N8).
- "Fewer poles keep the same size." Applies to the wall rule: fewer poles need a thicker wall (CG-11).
- "The stored 3D ... fields follow geometry or grade." The slip fields at the sleeve and liner are
  the same kind (N3).

**Proposed additions to slice-2 section 8 (must not be written)**
- N1. The calculator's torque falls when the back iron saturates, the wall is thin, or a shell is
  magnetic (PA-9, TC-6).
- N2. A non-magnetic back iron costs a fixed share such as "more than 40 %". The share depends on
  the design, and the dashboard shows 37 % at the example because of the calibration switch.
- N3. The sleeve, liner, hub, cup or magnet slip fields follow the user's design (PA-1).
- N4. "1.5 T is 4140's saturation", or "4140 saturates well above 1.7 T". Neither is sourced
  (PA-5, TC-17).
- N5. The expansion screen uses each grade's own expansion. False until E22 lands (PA-13).
- N6. The expansion caution accounts for the temperature swing, or clearing it means the bond is
  fine (`warnings.rs:155`).
- N7. "Raise Cup wall thickness at the pocket corners". That is a result; the input is "Minimum outer
  return ring wall" (TC-17 corrected claim).
- N8. 416, 17-4PH or 12L14 is a clean swap for 4140 or 1018. All trip the low-saturation caution.
- N9. A listed sleeve or liner can be magnetic, or the ferromagnetic-sleeve warning can fire.
- N10. "Adhesive shear modulus" follows "Selected adhesive (code)".
- N11. The wall rule is exact or a worst case. It reads 12-16 % thin at the example, about 19 % at
  14 poles, and is sized at the operating-temperature remanence.
- N12. The high-conductivity threshold follows the "316L conductivity" entry. It is a constant
  equal to that entry's default.

---

## 8. Dependencies for the parent

- D1. `CUP_WALL_GUIDE.learn = Learn::Note("a5.low_saturation")` (`guidance.rs:188`) must point to the
  new wall note if one is made. The test requires a reviewed note.
- D2. One note per equation. `model.backiron_needed_mm` moves to the wall note, which also picks up
  the unowned `materials.cup_wall_check` and `materials.cup_wall_suggested_mm`. `model.bsat_T` stays
  in `a5.low_saturation` (recommended) or moves too.
- D3. Add `materials.sleeve_sigma_S_m` to `a5.high_conductivity_sleeve_or_liner` (unowned).
- D4. `model.pullout_iron_Nm` and `model.pullout_noiron_Nm` are unowned. Candidate owners:
  non_ferromagnetic, back_iron_factor or the new torque note.
- D5. Every on-screen label quoted in a note should be checked by a test against the input and result
  metadata (DRY), so that a part-C rename cannot strand a note. These drafts quote about 20 labels.
- D6. All six notes are warning-linked. They must be `Reviewed` at merge (`release_notes_are_reviewed`),
  so the rewrite and its physics review land in one commit.
- D7. The note cap is 15-20 (`notes.rs:630-633`). Three new notes take it to exactly 20.
- D8. Tests that pin the literal lists in these drafts to the library: low_saturation fires for back
  iron codes {3,4,5,6} and not {1,2}; cte fires for {7,8} and for Back iron = 0, not {1..6}; no sleeve
  code fires ferromagnetic_sleeve; no sleeve code fires high_conductivity; uncoated fires for {1,2,3}
  at plating 0. The existing `each_rule_fires_exactly_on_its_condition` uses synthetic inputs, not
  library picks.
