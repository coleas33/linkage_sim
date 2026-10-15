# Magcoupler teaching notes: per-note physics reviews (guidance slice 2)

Branch `magcoupling/guidance-2`, crate `magcoupling-rs`. Spec
`docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md`, section 4 (decision S2-6).

Each note that guidance slice 2 adds or rewrites is reviewed by a fresh-context session-model
physics reviewer before it ships. The reviewer reads the note rendered at the five probe designs
(the default design, 8 magnets per ring, a 304 back iron, Y30 ferrite on custom dimensions, a
20 mm ring radius; `engine::explain::probes`), the engine code behind every claim and the
physics traps of the slice-1 and slice-2 specs (section 8 of each), reproduces every number from
`compute_all` and probes every "What to change" direction. Must-fix items are applied and
re-reviewed, up to three rounds (four from T12 on, by controller ruling 9; T12 ran a fifth under
ruling 12). A note becomes `Reviewed` in the same commit that appends its section here; its
`record` names the section as `<this file>#<note id>`.

## torque_from_shear

Task T11. New note "How the coupling makes torque", Start here step 1. Reviewer: physics
reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice plan fixes for the
note's `review` field; both rounds ran on 2026-10-15).

Rounds: 2. Verdict: PASS.

Round 1:

**What I read**
- The dump `magcoupling-rs/target/notes-review-T11.txt`: the note as Draft, rendered at the five probe designs.
- The literal `torque_from_shear` in `magcoupling-rs/src/engine/explain/notes.rs`, lines 150-198.
- `src/engine/model.rs`:
  - `compute`: L = min(L_i, L_o); r_pocket = (A_back + bond)/cos(π/N) for flat blocks; R_g = r_face,i + g/2; τ_p = 2π·R_g/N; the fills through `pitch_share` at each ring's magnet mid-radius, clamped at 1; the harmonic sum at the E7 pull-out angle; A_L = 2π·R_g²·L; T_2D = σ·A_L; f_end = 1 − c_end·τ_p/L; T_pull = T_2D·f_end·f_cal.
  - `pitch_share`, `geometry_factor`, `harmonic_amplitude` and `shear_stress`: k = n·(N/2)/R_g.
  - `select_calibration_factor`.
- `src/engine/explain/records/torque.rs`: the five owned records.
- `src/engine/explain/probes.rs`: the five designs and `step_input`.
- `src/engine/api.rs`: `OPTIONAL_SEEDS`.
- `src/gui/plain_text.rs`: the labels "Ring radius", "Magnets per ring" and "Pull-out torque (nominal)".
- The task-11 brief and its report; `amendments.md`.
- Spec 2026-10-14, sections 3.1-3.3, 3.5, R5, R6, R8 and 8.
- Slice-1 spec, section 8.
- `phys-new-notes.md`, section 2 (T-A to T-P) and section 3.1.
- The M1 math audit rows M3, M4 and M8, to check sources[1].

**How I reproduced the numbers.** I built a scratch probe crate outside the repo, depending on magcoupling-rs by path with default-features = false. It calls `compute_all`, `probes::step_input`, `model::blocks_fit` and `size_controls::dimensions`, and runs `plain::{word_count, plain_text_violations, audit_ids}` on every field and on both replacements.

**Check a: physical truth at all five designs as rendered.** OK, except the two must-fix items.
- Sentence 0 is true and plain.
- Sentence 1: σ is the summed shear stress at the true peak. E7 is on: the peak is 90° electrical at four designs and 52.96° at the 20 mm design, so "at its peak" holds at all five.
- Sentence 2: the formula is exact. A_L = 2π·R_g²·L and T_2D = σ·A_L, reproduced to 7 digits at every design. Only the term "endless-ring" fails (must-fix 1).
- Sentence 3: σ falls with the gap at all five designs (+0.01 mm and +1 mm). It falls with narrower blocks at all five (widths −0.01 to −2 mm on a manual copy of each design's geometry). It is higher with magnetic steel than with 304 at all five. But "steel" is false for 304 (must-fix 2). A thicker inner block lowers σ at the default (+2 mm: 201400 -> 200700 Pa) and at 8 magnets (+0.5 mm: 151800 -> 151100 Pa) while the pull-out rises, which is why the replacement says "magnet grades".
- Sentence 4 (A-19): R_g is identical at N = 4 to 40 at every design (14.02 mm; 23.87 mm at the 20 mm design). Two fewer magnets grow the cup at all five: 41.33 -> 42.44, 42.44 -> 45.03, 41.33 -> 42.44, 41.33 -> 42.44 and 62.04 -> 63.76 mm. The Outside diameter stays at 42.8 mm (the cap) at the example design, and the note says only "the cup" (T-P).
- watch_out[0] is true; the cause is decomposed below.
- watch_out[1] is true: T_pull = T_2D·f_end·f_cal at all five, and f_cal is the bench 1.0494 at the 304 design (10 magnets, B842SH on both rings, no magnetic steel). The text says only "a calibration factor", so it does not claim the bench factor applies elsewhere.

**Check b: change entries.** OK.
- change[0], `coupling.magnets.axial_length_mm` (the Size tab's Overall length sets it): the pull-out rises by one slider step, by +5 and +10 mm, and up to 50.8 mm at all five designs. The rise is exactly linear: 0.23627 N·m/mm at the default. Its costs are true: at the default, +5 mm raises the clamp's required torque 7.647 -> 11.007 N·m, the cold-high at the gearbox input 0.805 -> 1.159 N·m, and the stack length. Nothing says the coupling protects the gearbox.
- change[1], `coupling.inner_back_apothem_mm` (the Outside diameter control moves it): A_L rises by one step, +2, +5 and up to 30 mm at all five. The blocks spread apart: the inner flat slack goes 0.246 -> 1.546 mm at +2 mm on the default. The cup grows by 2.10 mm per mm. The pull-out is not claimed, and watch_out[0] says it can fall.
- change[2], `coupling.npole`: T_2D rises by one step (+2) and by +6 at all five. Default 3.159 -> 3.446 (+2) -> 3.310 (+6); 8 magnets 2.382 -> 3.159 -> 3.463; 304 1.805 -> 2.156 -> 2.425; Y30 0.2433 -> 0.2654 -> 0.2550; 20 mm 3.248 -> 5.131 -> 8.649 N·m. The blocks stop fitting at N = 12 on the four 14.02 mm designs and at N = 20 on the 20 mm design. T_2D rises everywhere the blocks fit, and peaks past the fit limit: at N = 14 (default, 8 magnets, Y30), N = 16 (304) and N = 22 (20 mm). Two fewer magnets lose pull-out at all five and grow the cup. At the 304 design the pull-out rises 1.697 -> 1.871 N·m: f_cal steps from 1.049 to 0.95, which is why the aim is T_2D.
- No entry offers a model assumption or the calibration prototype as a lever.

**Check c: the two scoped claims.** OK.
- Example design, per mm: magnet length +0.236274 N·m against ring radius +0.181726 N·m. It also holds at the other four designs: +0.178 vs −0.027, +0.149 vs +0.056, +0.0182 vs +0.0140, +0.243 vs +0.019.
- "The pull-out can even fall": at 8 magnets, +1 mm of ring radius gives −0.026723 N·m. That is R_g² ×1.1477, σ ×0.8687 and f_end 0.8699 -> 0.8607. The fills go 0.689/0.496 -> 0.635/0.467 (fundamental product ×0.907) and S_1 goes 0.2373 -> 0.2326. The fill does most of the damage, as the watch-out says. The pull-out also dips between 14 and 20 mm at the default and between 12 and 18 mm at 304.

**Check d: plain words and glosses.**
- Sentence 0 answers the question with no symbol.
- Glossed at first use: shear stress (sentence 1), gap radius (sentence 1), pole pitch as each pole's span (sentence 4), fill (in words: "blocks fill less of each pole").
- Not glossed: "endless ring" (must-fix 1).
- "Pole" comes one sentence before its span (should-fix 1); "pull-out" has no gloss (should-fix 3).
- `plain_text_violations` and `audit_ids` are empty for every field, sources included (A-13).
- No input label is typed: the dump draws "Axial magnet length, both rings", "Ring radius" and "Magnets per ring".

**Check e: traps.** None is written or implied.
- Ring radius is never called the strongest lever, and length beats it on the example design.
- Fewer magnets grow the cup; the note never says they keep it or the Outside diameter.
- R_g² growth is said "only at fixed σ".
- Nothing says fewer poles reach across the gap better, that fill = w/τ_p, or that end effects grow with more poles.
- "Thicker magnets raise σ" is not written, and must-fix 2 removes the "stronger magnets" reading.
- No guarantee, and no bench calibration for this design.
- None of the stiffness, re-lock or gearbox traps appear.

**Check f: caps.** OK, raw whitespace words.
- Sentences: 23, 23, 24, 24, 25 (5 sentences).
- Change texts: 23, 24, 25 (3 entries).
- Watch-outs: 26, 20 (2 entries).
- Both replacements: 25 each.
- The title "How the coupling makes torque" is plain, with no ϑ, ∝ or U+207B.

**Check g: sources.** OK and accurate.
- Line 0 names the five owned records.
- Line 1 matches M1 rows M3 (planar against cylindrical), M4 (exact 2D flat-block sections) and M8 (magpylib 3D rings).
- Line 2 matches the guidance audit's 10-against-8 and length-against-radius probes.

**Numbers reproduced** (`compute_all`; σ Pa, T_2D and pull-out N·m, R_g and τ_p mm, fills inner/outer):

| Design | σ | T_2D | Pull-out | f_end·f_cal | R_g | τ_p | Fills |
|---|---|---|---|---|---|---|---|
| default | 201380 | 3.1586 | 2.6885 | 0.8960 × 0.95 | 14.02 | 8.809 | 0.861/0.620 |
| 8 magnets per ring | 151848 | 2.3817 | 1.9684 | 0.8699 × 0.95 | 14.02 | 11.01 | 0.689/0.496 |
| 304 back iron | 115093 | 1.8052 | 1.6974 | 0.8960 × 1.0494 | 14.02 | 8.809 | 0.861/0.620 |
| Y30 ferrite, custom dimensions | 15511 | 0.24328 | 0.20707 | 0.8960 × 0.95 | 14.02 | 8.809 | 0.861/0.620 |
| 20 mm ring radius | 71445 | 3.2483 | 2.5393 | 0.8229 × 0.95 | 23.87 | 15.00 | 0.468/0.386 |

All match the dump and the drafter's numbers.

**Verdict:** MUST-FIX. Two items: the sentences[2] gloss and the sentences[3] magnetic-steel wording. Everything else passes.

Round 2:

Round 2. Scope: the two fields changed since round 1 (sentences[2] and sentences[3]), and the fields whose meaning depends on them: sentences[1], change[2], watch_out[0] and watch_out[1].

**What I read**
- The change list `task-11-inreview-r1-changes.md`, the round-1 answer `task-11-inreview-r1.md`, and the task-11 brief (Step 7 prompt).
- `amendments.md`: the Global Constraints, the T11 index row, and A-12, A-13, A-14, A-19 to A-25.
- The regenerated dump `magcoupling-rs/target/notes-review-T11.txt`: the note as Draft at the five probe designs. Both new sentences appear at all five, with no brace and no "no prediction".
- The literal `torque_from_shear` in `magcoupling-rs/src/engine/explain/notes.rs` (lines 150-198). Its text matches the change list.
- `src/engine/model.rs`:
  - `compute`, lines 915-1004: L = min(L_i, L_o); R_g; τ_p; fills; Br at the operating temperature per ring; σ at the pull-out angle; A_L; T_2D = σ·A_L; f_end = 1 − c_end·τ_p/L; T_pull = T_2D·f_end·f_cal.
  - `geometry_factor`, `harmonic_amplitude`, `shear_stress` and `Harmonic::s`: steel only when the circuit flag is 1.
- `src/engine/material_library.rs` `resolve`: a non-ferromagnetic back-iron pick forces the free-space circuit.
- `src/engine/explain/records/torque.rs` and the engine label of `model.torque_2d_Nm`, "2D pull-out torque (infinite length)".
- `src/engine/explain/probes.rs`.
- T12's draft `back_iron_factor` (task-12 brief), for where Start here gives the saturation caveat.
- Spec 2026-10-14 sections 2 (S2-4, R6, R8), 3.1, 3.2 and 8; slice-1 spec section 8; `phys-new-notes.md` section 2.

**How I reproduced the numbers.** I built a new scratch probe crate outside the repo, depending on magcoupling-rs by path with default-features = false. It calls:
- `compute_all`, `probes::{probe_designs, step_input}` and `grades::GRADES`;
- `plain::{word_count, sentence_count, plain_text_violations, audit_ids}`, on every field and on each should-fix text.

**Check a: physical truth of the changed fields at all five designs, as rendered.** OK.
- sentences[2], the formula: T_2D equals σ·2π·R_g²·L at every design, from the rendered σ and from R_g and L. The values are 3.15861, 2.38171, 1.80521, 0.24328 and 3.24831 N·m.
- sentences[2], "(magnet ends ignored)": T_2D carries no end term.
  - Setting c_end to 0, ×0.5 or ×1.5 leaves T_2D identical to 1e-12 at all five designs, while the pull-out moves. At the default it goes 3.0007, 2.8446 and 2.5324 against 2.6885 N·m.
  - With c_end = 0 the pull-out equals T_2D·f_cal exactly.
  - T_2D/L is constant to 8 digits for +1, +5 and +10 mm of length at all five designs, for example 0.24870929 N·m/mm at the default.
  - So "magnet ends" means the axial ends of the overlapping length L, the same ends watch_out[1] says the dashboard corrects for.
- sentences[2], "times lever arm R_g": σ·(2π·R_g·L)·R_g is the formula as the engine computes it.
- sentences[3], "stronger magnet grades": σ scales exactly with Br_i·Br_o at the operating temperature.
  - On a custom-dimension copy of each design, swapping both rings through all 17 grades keeps σ/(Br_i,op·Br_o,op) constant to 6e-16 relative. σ rises monotonically with that product, from Y30 to N52.
  - With one ring alone (inner N42SH -> N52, outer N42SH), σ goes 201380 -> 224616, 151848 -> 169369, 115093 -> 128373 and 71445 -> 79688 Pa. The Y30-design copy at N42SH is the default geometry, 201380 -> 224616.
  - "Grades" in place of "magnets" also keeps the trap reading "thicker magnets raise σ" out.
- sentences[3], "magnetic steel behind the magnets": true at all five designs.
  - The six magnetic steels (4140, 1018, 12L14, 416, 17-4PH H1150 and H900) give identical σ. The model treats steel as ideal (T-L).
  - 304, 6061 or "no back iron" give the free-space σ. Steel over free space: default 201380 vs 115093 Pa (×1.750); 8 magnets 151848 vs 77189 (×1.967); Y30 15511 vs 8865 (×1.750); 20 mm 71445 vs 33900 (×2.108).
  - At the 304 design, any magnetic steel pick raises σ 115093 -> 201380 Pa. Setting the circuit flag to 1 under a 304 pick keeps 115093, because the pick forces free space. So the sentence's word "magnetic" is what makes it true there (round-1 must-fix 2).
  - A scan of 24000 geometries found steel above free space in every case, with the lowest ratio 1.000031 (thick magnets on a short pitch). The scan covered N 4–40, gap 0.3–10 mm, thickness 1–10 mm, width 2–10 mm, ring radius 8–40 mm, and harmonics 1, 3, 5 and 11. This matches the analytic ratio (1 + e^(−k·t_i))(1 + e^(−k·t_o))/(1 − e^(−2k(t_i + t_o + g))) > 1.
- sentences[3], "falls as the gap widens": σ falls at one slider step (+0.01 mm) and at +1 mm at all five designs. Default 201380 -> 200589 -> 136883; 8 magnets 151848 -> 151296 -> 107213; 304 115093 -> 114626 -> 76998; Y30 15511 -> 15450 -> 10543; 20 mm 71445 -> 71232 -> 54243 Pa.
- sentences[3], "or blocks fill less of each pole": σ falls for blocks narrower by 0.01, 0.5, 1 and 2 mm on both rings at all five designs (custom copies at each design's grade). It also falls whenever the fill drops another way: through a 1 mm larger ring and through two fewer magnets, at all five designs.
  - Default: fills 0.861/0.620 -> 0.794/0.584, σ 201380 -> 188882 Pa.
  - 8 magnets: σ 151848 -> 131912 Pa.
  - 20 mm: σ 71445 -> 66910 Pa.
  - Two fewer magnets: σ 201380 -> 151848, 151848 -> 75105, 115093 -> 77189, 15511 -> 11696 and 71445 -> 52248 Pa.
- Dependent fields:
  - sentences[1] defines σ, and sentences[2] now opens with it. That reads correctly.
  - change[2] ("blocks fill more of each pole") is the mirror of sentences[3].
  - watch_out[0] still holds: at 8 magnets, +1 mm of ring radius lowers the pull-out 1.9684 -> 1.9416 N·m (σ ×0.869, fills 0.689/0.496 -> 0.635/0.467).
  - watch_out[1]'s "corrected for the magnet ends" now refers back to sentences[2]'s "(magnet ends ignored)", and T_pull = T_2D·f_end·f_cal holds at all five designs.

**Check b: change entries.** Not changed this round. Their meaning does not depend on the changed sentences, except change[2]'s fill wording, which agrees with sentences[3] (above). The dump shows every aim "moves as it says" at all five designs.

**Check c: scoped claims.** Not changed. watch_out[0]'s "the pull-out can even fall" is re-confirmed above at 8 magnets (−0.0268 N·m per +1 mm of ring radius).

**Check d: plain words and glosses.**
- "endless ring" is now glossed at its first use: "(magnet ends ignored)" (round-1 must-fix 1). The equation panel calls the same quantity "2D pull-out torque (infinite length)"; should-fix 2 is the optional wording match.
- "pole" is still first used in sentences[3] and glossed in sentences[4]; should-fix 1 is the swap.
- `plain_text_violations` and `audit_ids` are empty for every field, and each changed field is one sentence (`sentence_count` 1).
- No input label is typed.

**Check e: traps.** None is written or implied by the changed fields.
- No "steel gives the torque shown": the saturation caveat is at Start here step 3 in T12's draft.
- No "thicker magnets raise σ", fill = w/τ_p, end effects growing with poles, R_g² without fixed σ, or "more magnets make the coupling stiffer".

**Check f: caps.** OK, raw whitespace words.
- Sentences: 23, 23, 25, 25, 25 (5 sentences).
- Change texts: 23, 24, 25.
- Watch-outs: 26, 20.
- Should-fix texts: 25 each (sentences); 18 (sources[0]).
- Title unchanged.

**Check g: sources.** Unchanged and exactly A-13's lines, plain-words clean. Line 0's "endless-ring torque" now matches the glossed term in sentences[2].

**Numbers reproduced** (`compute_all`; σ in Pa, T_2D and pull-out in N·m, R_g and τ_p in mm, fills inner/outer):

| Design | σ | T_2D | Pull-out | f_end · f_cal | R_g | τ_p | Fills |
|---|---|---|---|---|---|---|---|
| default | 201379.9 | 3.15861 | 2.68848 | 0.8960 × 0.9500 | 14.020 | 8.809 | 0.861/0.620 |
| 8 magnets per ring | 151848.1 | 2.38171 | 1.96836 | 0.8699 × 0.9500 | 14.020 | 11.011 | 0.689/0.496 |
| 304 back iron | 115092.8 | 1.80521 | 1.69737 | 0.8960 × 1.0495 | 14.020 | 8.809 | 0.861/0.620 |
| Y30 ferrite, custom dimensions | 15510.8 | 0.24328 | 0.20707 | 0.8960 × 0.9500 | 14.020 | 8.809 | 0.861/0.620 |
| 20 mm ring radius | 71444.6 | 3.24831 | 2.53926 | 0.8229 × 0.9500 | 23.870 | 14.998 | 0.468/0.386 |

All match the dump as rendered (201400, 151800, 115100, 15510 and 71440 Pa; 3.159, 2.382, 1.805, 0.2433 and 3.248 N·m becoming 2.688, 1.968, 1.697, 0.2071 and 2.539 N·m) and round 1's table.

**Amendments:**
- A-13, A-14, A-19 and A-25 T-1 are in place.
- A-20 to A-23 do not apply to T11.
- There are no A-24 seeds for T11.

**Verdict:** PASS. There is no open must-fix. Two optional should-fixes remain: the sentence swap (round 1's, still recommended because it is free) and the "infinite-length" wording match, which needs a ruling on A-13 for sources[0].

Must-fix items applied:
- sentences[2] (round 1, must-fix 1: gloss "endless-ring" at its first use): "That stress times the cylinder's area 2π·R_g·L, times the lever arm R_g, is the endless-ring torque T_2D = σ·2π·R_g²·L, L the overlapping magnet length." -> "σ times the cylinder's area 2π·R_g·L, times lever arm R_g, is the endless-ring torque T_2D = σ·2π·R_g²·L (magnet ends ignored), L the overlapping magnet length."
- sentences[3] (round 1, must-fix 2: "steel" is false for 304; "stronger magnets" can read as thicker magnets): "σ grows with stronger magnets and with steel behind them, and falls as the gap widens or as blocks fill less of each pole." -> "σ grows with stronger magnet grades and magnetic steel behind the magnets; it falls as the gap widens or blocks fill less of each pole."

Seeds (A-24): none names T11, so neither round had a seed to accept or reject.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS):
- Round 1 S-1, repeated as round 2 S-1: swap sentences[3] and sentences[4], so that "pole" is
  glossed (sentence 4, "each pole spanning τ_p = 2π·R_g/N") before the fill sentence uses "each
  pole". No word changes. Round 2 keeps it a should-fix: the gloss comes one sentence later and
  the text is not false or misleading without the swap.
- Round 1 S-2: four more engine pins in `tests/explain.rs`
  `torque_from_shear_claims_hold_on_the_engine` at all five probe designs: two fewer magnets keep
  the gap radius and grow the cup; one gap step lowers σ; magnetic steel gives a higher σ than
  304; T_2D rises with +2 magnets while the blocks still fit.
- Round 1 S-3: gloss "pull-out" in watch_out[1] ("the most the rings pass before slipping") and
  order it before the R_g² watch-out.
- Round 2 S-2: "infinite-length" in place of "endless-ring" in sentences[2], to match the T_2D
  equation's label "2D pull-out torque (infinite length)". It needs a ruling on A-13, which fixes
  sources[0] verbatim with "endless-ring"; the reviewer advises changing both or neither.

### Re-review: changed by the implementer after its review

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `9d68db5`
  (`task-12b-textreview-1.md`). It re-reviews the texts that T12b, the shaft-clamp pause
  (controller ruling 11), changed after their review.
- Scope here: change[0].text, made joint-neutral by T12b (ruling 11, 3A): "... Costs: longer
  stack, higher clamp and gearbox loads." -> "... Costs: longer stack, higher shaft-joint and
  gearbox loads." The same re-review covered eight help, hint and GUI strings. Its whole report,
  with its one must-fix item (the bore help, applied), is recorded in
  `docs/analyses/2026-10-14-magcoupler-help-review.md`, Batch A (T7), "Re-review: changed by the
  implementer after its review".
- Verdict for this note: PASS, no must-fix. The dump (`NOTES_FOR_REVIEW=torque_from_shear`)
  reads change[0] "moves as it says" at all five probe designs. No text of this note changed.

The reviewer's report for the record, as returned (its opening line, this note's item, and the
traps and rules items, which cover every changed text):

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
- **Traps:** no text writes a trap from slice-1 section 8, slice-2 section 8, T-A to T-P, or the
  stiffness trap.
- **Rules:** none says a joint method is chosen or sized, or that the coupling protects the
  gearbox. Every cap holds.

## harmonics

Task T12. Rewrite of "Harmonic decomposition" as "Block fill and the magnet pattern (harmonics)",
Start here step 2; it now also owns the fill rows (`model.fill_inner`, `model.fill_outer`).
Reviewer: physics reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice
plan fixes for the note's `review` field; all five rounds ran on 2026-10-15).

Rounds: 5. Verdict: PASS.

Verdict by round: 1 MUST-FIX (2 items); 2 MUST-FIX (1); 3 MUST-FIX (1); 4 PASS; 5 PASS (no field
had changed since round 4, so the round-4 PASS stood). Rounds 4 and 5 ran under controller
rulings 9 and 12, which raised T12's round limit. Each round's full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-12-inreview-r<N>.md`; its opening
amendment checks, which the reports below call "above", hold the seed verdicts summarized under
"Seeds (A-24)" in each section.

Round 1:

**What I read**
- The dump `magcoupling-rs/target/notes-review-T12.txt`: harmonics, Draft, at the five probe designs.
- The literals in `notes.rs`: harmonics, plus torque_from_shear for context (lines 150-258).
- `diagrams.rs` `square_wave`, unchanged: fill 0.85, harmonics 1, 3 and 5.
- `model.rs`:
  - `pitch_share` (:509-514);
  - `harmonic_amplitude` (:645-650);
  - `shear_stress` (:676-712);
  - `peak_off_half_pitch`, `stationary_polynomial` and `at_pull_out` (:714-892);
  - `compute` (:900-1178), including the clamped fills at :973-974 and the summed σ at the E7 angle at :998-1000.
- `probes.rs`.
- The 2026-10-14 spec: 3.1, 3.2, 3.3, R5, R6, R8 and 8.
- The slice-1 spec, section 8.
- `phys-notes-a.md`, sections 1, 2 and 8.
- `phys-new-notes.md`, section 2 (T-A to T-P).
- `amendments.md`.

**How I reproduced the numbers.** I used a new scratch probe crate outside the repo (scratchpad `t12-inreview-r1-probe`, magcoupling-rs by path, default-features = false). It calls `compute_all`, `probes::{probe_designs, step_input}`, `model::blocks_fit`, and `plain::{word_count, plain_text_violations, audit_ids}`.

**Numbers reproduced** (fills inner/outer; σ_1 of σ, both at the pull-out angle; A3/A1; pull-out with the harmonic set to 1, 3, 5, 7, 9 and 11):

| Design | Fills | σ_1 of σ (Pa) | A3/A1 | Pull-out (N·m) |
|---|---|---|---|---|
| default | 0.8612 / 0.6198 | 199961 of 201380 | −0.0109 | 2.6695 (1), 2.6987 (3), 2.6885 (5), 2.6886 (7), 2.6884 (9), 2.6885 (11) |
| 8 magnets | 0.6890 / 0.4959 | 149136 of 151848 | −0.0085 | 1.9332 (1), 1.9497 (3), 1.9684 (5), 1.9742 (7, +0.295 %), 1.9737 (9), 1.9740 (11) |
| 304 back iron | 0.8612 / 0.6198 | 113808 of 115093 | −0.0179 | 1.6881 (1), 1.6974 (5); 7-11 within −0.05 % |
| Y30 custom | 0.8612 / 0.6198 | 15401.5 of 15510.8 | −0.0109 | 7-11 within ±0.005 % |
| 20 mm ring | 0.4682 / 0.3864 | 64874 of 71445 (amplitude A1 81274) | +0.2158 | 2.8886 (1, at 90°), 2.5303 (3, 0.9234 rad, −12.40 % against 1), 2.5393 (5, 0.9243 rad), 2.5535 (7, +0.560 %), 2.5513 (9), 2.5492 (11) |

The dump's rounded values (0.8612/0.6198 ... 200000 of 201400 Pa ... 64870 of 71440 Pa) match at all five designs.

**Check a: physical truth at all five designs.**
- Sentence 0: false ("small gaps"), must-fix 1.
- Sentence 1: true. No harmonic exceeds the fundamental, since |sin nx| ≤ n|sin x|. Alternating N/S blocks give odd harmonics only.
- Sentence 2: true. B_1 = Br·4/π·sin(πλ/2) (model.rs:649).
- Sentence 3: true (see check c).
- Sentence 4: true as rendered. Should-fix 1 clarifies where the split is taken.
- change[0]: true at all five designs, B862 (9.5 mm) outer against B842SH:
  - +15.0 % and fits at the default;
  - +28.7 % at 8 magnets;
  - +1.1 % at the 304 design, where the bench calibration lapses (1.0495 -> 0.95) but the torque still rises;
  - +50.2 % at the 20 mm design;
  - at the Y30 design the pick also brings an NdFeB grade.
  B882 (12.7 mm) does not fit at four designs (fill clamped at 1). Must-fix 2 is wording only.
- change[1] (manual outer width): held by the part pick at four designs, as the dump draws. At the Y30 design one step (6.35 -> 6.36 mm) moves the pull-out 0.207074 -> 0.207243 N·m. The sweep to 25.4 mm rises until the blocks stop fitting at about 9.7 mm, then stays flat (fill clamped at 1). "While the blocks still fit" names the limit.
- change[2] (magnets per ring, aim outer fill Up): one step holds at all five designs (0.6198 -> 0.7438 at the default, 0.4959 -> 0.6198 at 8 magnets, 0.3864 -> 0.4637 at 20 mm). The sweep to 40 never falls; the fill clamps at 1. At the default ring 12 magnets already fail the fit (inner flat 5.44 mm against a 6.35 mm block), which "they must still fit" names. At 20 mm the blocks fit up to 18.
- change[3]: true. It presents a model setting, not a design lever. "Under 2 % where checked" holds: at most +0.560 % from 7-11 against 5, at the 20 mm design.
- watch_out[0]: true. At the 20 mm design the third harmonic moves the pull-out by −12.4 % and its angle from 90° to 52.9°. At the default ring with 6 magnets the angle is 59.15°, and with 4 magnets 33.4°.

**Check b: change entries.** As above. Every Raise entry was stepped one slider step and swept to its slider end at each design where the input is free. Trade-offs: fit and magnet mass (change[0]); the fit (change[1] and change[2]).

**Check c: the physics of the note.**
- The fill is each ring's own `pitch_share` at its magnet mid-radius ("the room each pole has on that ring"), never w/τ_p. At the default w/τ_p is 0.721, against fills of 0.8612 and 0.6198.
- Higher pairs add little mainly through the amplitude product. At the default, B_i3·B_o3/(B_i1·B_o1) = −0.024, against S_3/S_1 = 0.456 with steel. "Being weaker" comes first, then "fading faster across the gap" (e^(−n·k·g)).
- In free space S_3/S_1 can exceed 1: 1.107 at 8 magnets and 1.762 at 20 mm, through the thickness terms. The sentence claims only the gap fade and says "usually", so it stays true.
- The setting matters at low fill (watch-out and change[3]).

**Check g: plain words and caps.**
- Word counts: sentences 24, 24, 22, 24 and 24; changes 21, 21, 23 and 23; watch-out 28.
- Sentence 0 has no symbol and no placeholder.
- The placeholders sit in an "On this design" clause.
- No input label is typed. The title is plain and unique.
- Replacements checked with `word_count`, `plain_text_violations` and `audit_ids`: 23 and 23 words (should-fix: 25 and 23), all clean.

**Check h: sources.** Correct and plain (A-13). The low-fill third harmonic (6 magnets, a 20 mm ring, narrow B822 blocks) is phys-notes-a section 2's probe set.

**Trap check.** No spec-8, slice-1-8, T-A to T-P or "more magnets stiffer" trap appears. DRY: change[2] overlaps torque_from_shear change[2] on the same input. Accepted: the aims differ (outer fill against T_2D), and each note is reached on its own.

**Verdict:** MUST-FIX (2).

Round 2:

**What I read**
- `task-12-inreview-r1-changes.md` and the round-1 answer `task-12-inreview-r1.md`.
- The regenerated dump `magcoupling-rs/target/notes-review-T12.txt` (harmonics, at the five designs).
- The harmonics literal in `notes.rs` (lines 203-258).
- `model.rs`:
  - `pitch_share` and `flat_fits` (:505-515);
  - `ring_fits` and `blocks_fit` (:518-582);
  - the outer flat and pocket geometry in `compute`: `A_o = r_corner_i + corner_gap` and `r_pocket` from `A_back = A_o + t_o`, so a wider outer block does not grow the cup.
- `library.rs`: all 15 parts with their width, thickness, length and grade.
- `diagrams.rs` `square_wave`: the label "blocks and gaps (fill 0.85)", for the dependent-wording check.
- Spec 8, slice-1 spec 8, phys-new-notes section 2 and phys-notes-a section 8.

**How I reproduced the numbers.** I built a new scratch crate outside the repo: scratchpad `t12-inreview-r2-own-probe`, with magcoupling-rs by path, default-features = false, and its own target directory. It calls:
- `compute_all`;
- `probes::probe_designs`;
- the result paths `model.fill_*`, `model.*_flat_check`, `model.pullout_Nm` and `mass.magnets_g`;
- `plain::{word_count, plain_text_violations, audit_ids}`.

**sentences[0] (new: "spaces between them").** It is true at all five designs: every ring has room between its blocks.

| Design | Fills (inner / outer) | Inner flat slack | Outer blocks apart at the faces |
|---|---|---|---|
| default | 0.8612 / 0.6198 | 0.25 mm | 3.22 mm |
| 8 magnets | 0.6890 / 0.4959 | 2.06 mm | 5.84 mm |
| 304 | 0.8612 / 0.6198 | 0.25 mm | 3.22 mm |
| Y30 | 0.8612 / 0.6198 | 0.25 mm | 3.22 mm |
| 20 mm | 0.4682 / 0.3864 | 6.65 mm | 9.62 mm |

- At the default inner ring the 0.25 mm slack is at the block backs. The faces along the gap fan apart (fill 0.8612, so 14 % of the pole is empty at the mid-radius).
- "Spaces" makes no size claim, so the round-1 falsehood ("small gaps") is gone.
- Sentence-0 rules hold: no symbol and no placeholder. It has 23 words.

**change[0] (new: A-24 S-2 text).** I tried every library part wider than the 6.35 mm outer block, at each design. Pull-out change; fit; magnet-mass change:

| Part (w x t mm, grade) | default | 8 magnets | 304 | Y30 | 20 mm |
|---|---|---|---|---|---|
| B862 (9.5 x 3.17, N42) | +14.96 %; fits; +9.51 g | +28.68 %; fits; +7.61 g | +1.14 %; fits; +9.51 g | +314.21 %; fits; +15.90 g | +50.24 %; fits; +9.51 g |
| B882 (12.7 x 3.17, N42) | +15.51 %; no fit | +37.25 %; no fit | +1.51 %; no fit | +316.21 %; no fit | +99.73 %; fits; +19.17 g |
| B882-N52 (12.7 x 3.17, N52) | +28.84 %; no fit | +53.09 %; no fit | +13.23 %; no fit | +364.24 %; no fit | +122.78 %; fits |
| BX082SH (12.7 x 3.17, 25.4 long) | no fit | no fit | no fit | no fit | +99.73 %; fits |
| B861 (9.5 x 1.59, N42) | **−12.76 %; fits; −4.79 g** | **−4.83 %; fits; −3.83 g** | **−36.47 %; fits; −4.79 g** | +214.34 %; fits; +1.61 g | +6.42 %; fits; −4.79 g |
| B881 (12.7 x 1.59, N42) | −12.71 %; no fit | −0.93 %; no fit | −36.45 %; no fit | +214.51 %; no fit | +43.20 %; fits; +0.06 g |
| M5026 (6.4 x 1.67, 15 long, N50) | **−12.10 %; fits; −7.15 g** | **−14.36 %; fits; −5.72 g** | **−33.06 %; fits; −7.15 g** | +216.73 %; fits; −0.76 g | **−18.48 %; fits; −7.15 g** |

- The rows in bold contradict the round-1 text: a wider block that fits, yet the pull-out and the magnet mass fall. That is must-fix 1.
- Every equal-thickness wider part that fits raises the outer fill, the pull-out and the mass. So "of equal thickness" makes the entry true at all five designs.
- The f_cal is unchanged in every outer pick (0.95, and the 304 bench factor ends exactly as round 1 found).
- A wider outer block does not move the pocket or the cup: `A_o` comes from the inner ring, and `r_pocket` from the outer thickness. So "it adds magnet mass" is the complete trade-off.

**Dependent fields.**
- change[1] (manual outer width) stays consistent: with custom dimensions the thickness is its own input.
- change[2] keeps "gaps" against sentences[0]'s "spaces" (should-fix 1). It is not false: the room between blocks does widen with fewer magnets.
- The square-wave label "blocks and gaps" is the same wording point (DIAGRAMS should-fix 1).
- change[2] does not say "fewer poles keep the same size" or "more magnets make the coupling stiffer".

**Check g.**
- Word counts: sentences[0] 23; change[0] 23 now, 25 after must-fix 1; should-fix 1 text 23. All are plain-words clean with no ids.
- No input label is typed.

**Verdict:** MUST-FIX (1).

Round 3:

**What I read**
- Inputs to this round:
  - the task-12 brief, Step 8 (the reviewer prompt) and Step 9;
  - `amendments.md`: the Global Constraints, the index row for T12, A-13, A-19 to A-24 and A-27;
  - `task-12-inreview-r2-changes.md` and the round-2 answer `task-12-inreview-r2.md`;
  - the regenerated dump `magcoupling-rs/target/notes-review-T12.txt` (all three notes, five designs).
- Code:
  - the harmonics literal (`notes.rs:203-258`) and the A-19 line (`notes.rs:165`);
  - `probes.rs` (`probe_designs`, `step_input`);
  - `library.rs`: all 15 parts with length, width, thickness, grade and Br;
  - `model.rs`: `pitch_share`, `RingFit`, `ring_fits`, `blocks_fit` and `select_calibration_factor` (:505-600);
  - the `SummaryResults` fields in `temperature.rs` (:426-449);
  - the three claim tests in `tests/explain.rs` (:1273-1356).
- Specs:
  - the 2026-10-14 spec, 3.1 (the change-entry rules), 4 and 8;
  - the slice-1 spec, section 8;
  - phys-new-notes.md, section 2 (T-A to T-P).

**How I reproduced the numbers.** I built a new scratch crate outside the repo: scratchpad `t12-inreview-r3-own-probe`, with magcoupling-rs by path, default-features = false. It calls:
- `compute_all`, `probes::probe_designs`, `model::blocks_fit` and `meta::result_rows`;
- the result paths `model.fill_outer`, `model.pullout_Nm`, `model.pullout_20C_Nm`, `metal.torque_hot_low_Nm`, `metal.torque_cold_high_Nm`, `mass.magnets_g`, `model.f_cal`, `model.pullout_angle_rad`, `temperature.summary.*` and `clamps.recommended`;
- `plain::{word_count, plain_text_violations, audit_ids}`.

For every library part wider than the current outer block, at each design, I compared every text result (checks, verdicts, notes) against the design as it stands.

**The outer block at each design.** It is 6.35 mm wide and 3.17 mm thick, with the part's own length of 12.7 mm (axial length left blank).
- At four designs the outer part is B842SH (N42SH, 1.29 T, rated 150 °C).
- At the Y30 design it is blank, with manual dimensions 6.35 x 3.17 and grade Y30.

**change[0], round-2 text: "a wider outer block of equal thickness raises the outer fill and torque while the blocks still fit side by side; it adds magnet mass."**

The library's wider parts of equal thickness (3.17 mm) are B862 (9.5 wide, N42), B882 (12.7, N42), B882-N52 (12.7, N52) and BX082SH (12.7 wide, 25.4 long, N42SH). Picked on the outer ring at each design (fill_outer; pull-out change; cold-high change; mass change; temperature verdict; clamp screw):

| Design | B862 | B882 | B882-N52 | BX082SH |
|---|---|---|---|---|
| default | fits; 0.6198 -> 0.9273; +14.96 %; +14.96 %; +9.51 g; **OK -> CHECK**; **M4 x 14 -> None** | no fit | no fit | no fit |
| 8 magnets | fits; 0.4959 -> 0.7418; +28.68 %; +28.68 %; +7.61 g; **OK -> CHECK**; unchanged | no fit | no fit | no fit |
| 304 | fits; 0.6198 -> 0.9273; +1.14 % (f_cal 1.0495 -> 0.95); +1.14 %; +9.51 g; **OK -> CHECK**; unchanged | no fit | no fit | no fit |
| Y30 | fits; 0.6198 -> 0.9273; +314.21 %; +286.59 %; +15.90 g; CHECK stays CHECK (governing limit 120 °C, adhesive -> −3.52 °C, magnets); unchanged | no fit | no fit | no fit |
| 20 mm | fits; 0.3864 -> 0.5781; +50.24 %; +50.24 %; +9.51 g; **OK -> CHECK**; **M4 x 14 -> None** | fits; -> 0.7728; +99.73 %; +19.17 g; **OK -> CHECK**; **-> None** | fits; -> 0.7728; +122.78 %; +19.17 g; **OK -> CHECK**; **-> None** | fits; -> 0.7728; +99.73 %; +57.52 g; OK stays OK; **-> None** |

- The round-2 text's three claims hold in all 8 fitting cases: the outer fill, the pull-out and the mass all rise.
- The 20 °C pull-out, hot-low and cold-high move by the same percentage as the pull-out, except at Y30, where the grade change makes them differ (+303.90 %, +314.21 %, +286.59 %).
- The qualifier is still needed. The wider but thinner parts that fit lower the pull-out:
  - B861 at the default, 8-magnet and 304 designs (−12.76 %, −4.83 %, −36.47 %);
  - M5026 at the same three and at 20 mm (−12.10 %, −14.36 %, −33.06 %, −18.48 %).
- What the round-2 text misses is spec 3.1's trade-off:
  - The temperature verdict turns red at the default, 8-magnet, 304 and 20 mm designs whenever the pick is N42 or N52. The magnet design limit (the skipping-case onset) falls from 93.06 °C to −3.52 °C for N42 (margin −68.52 °C) or 0.17 °C for N52 (B882-N52 at 20 mm, margin −64.83 °C). Both are below the 65 °C hot-day start.
  - The same block in N42 (B842, same size as B842SH) gives the same −3.52 °C. So the grade causes it, not the width.
  - BX082SH, the one equal-thickness wider part in N42SH, keeps 93.06 °C. Hence "may".
  - The clamp turns red at the default and 20 mm designs: the cold-high rise leaves no M4 screw length that engages its thread inside the boss ("None: enlarge the boss or the clamp length"). This holds even for BX082SH at 20 mm.
  - Base cold-high values (N·m): 3.823 (default), 2.799 (8 magnets), 2.414 (304), 0.338 (Y30), 3.611 (20 mm).
- Correction to the round-2 record. It said "it adds magnet mass" was the complete trade-off. That held for the cup and pocket geometry: `A_o` comes from the inner ring and `r_pocket` from the outer thickness. It did not hold for the picked part's grade or for cold-high.
- The B862 pick also changes things the replacement does not need to name, because they are improvements or not red:
  - the default design's hot-low turns from "Below hot minimum" to "Estimate covers hot min" (it does the same at 20 mm);
  - the adhesive fatigue margin narrows but stays OK.

**Replacement (must-fix 1): "a wider, equally thick outer block raises the outer fill and torque if the blocks fit; clamp loads rise; its grade may tolerate less heat."** Checked against the table:
- "raises the outer fill and torque if the blocks fit": true in all 8 fitting cases. The non-fitting picks are excluded by the condition.
- "clamp loads rise": cold-high rises in all 8 (by +1.14 % to +286.59 %). The clamp is sized to cold-high times its safety factor, and it turns red at two designs.
- "its grade may tolerate less heat": the picks are N42, N52 (80 °C) or N42SH (150 °C), against N42SH, or Y30 at the ferrite design.
  - Three of the four picks that fit at some design lower the magnet limit: B862 and B882 to −3.52 °C, B882-N52 to 0.17 °C.
  - BX082SH keeps 93.06 °C. Hence "may".
  - The wording matches pullout_angle's grade entry ("it often tolerates less heat").
- It drops "it adds magnet mass" (true, but no check) to fit the cap. The two checks that go red outrank it.
- It drops "side by side": "if the blocks fit" carries the same condition, as change[1] and change[2] word it ("while the blocks still fit", "they must still fit").
- It has no "its flat" (S-2's purpose) and no typed label, symbol or placeholder.
- Check g: 25 words by `word_count`; `plain_text_violations` [] and `audit_ids` [].

**Dependent fields.**
- sentences[2] ("The fundamental grows with sin(π·λ/2) ... the block's width over the room each pole has on that ring") is consistent: an equally thick, wider block raises λ at the same mid-radius.
- change[1] (manual outer width) stays true. With custom dimensions the thickness is its own input, so it needs no "equally thick", and the grade does not change. It now lacks the clamp trade-off that change[0] names (should-fix 1).
- change[2] keeps "gaps" against sentences[0]'s "spaces" (should-fix 2). It is not false.
- The watch-out (low fill, the third harmonic) does not depend on change[0].
- The square-wave label "blocks and gaps (fill 0.85)" makes the same wording point as change[2]. It is DIAGRAMS should-fix 1 of round 2, still open.

**Verdict:** MUST-FIX (1).

Round 4:

**What I read**
- `task-12-inreview-r3-changes.md` and the answers of rounds 1 to 3 (`task-12-inreview-r1.md` to `-r3.md`).
- The brief's Step 8 prompt and Step 9.
- `amendments.md`: the Global Constraints, the index row for T12, rulings 9 and 11, A-13, A-19 to A-24 and A-28 (its rules for every task, A-28j and A-28k).
- The regenerated dump `magcoupling-rs/target/notes-review-T12.txt`.
- The literals in `notes.rs:150-356` (torque_from_shear for context, then the three notes).
- `diagrams.rs` `square_wave` (:135-197) and the test's `wave_label` (:792).
- `tests/explain.rs:1269-1444` (the four T12 claim tests).
- `probes.rs` (`probe_designs`, `step_input`), `library.rs` (all 15 parts), the cold-high and gearbox-input lines in `metal_design.rs` (:455, :509), and `pickers.rs` (every library part is offered on either ring).
- The 2026-10-14 spec section 8, the slice-1 spec section 8 and phys-new-notes section 2.

**How I reproduced the numbers.** I built a new scratch crate outside the repo: scratchpad `t12-inreview-r4-own-probe`, with magcoupling-rs by path, default-features = false. It calls:
- `compute_all`, `probes::{probe_designs, step_input}`, `model::blocks_fit`, `meta::{input_rows, result_rows}` and `notes::NOTES`;
- `plain::{word_count, plain_text_violations, audit_ids}`.

For each design and each move, it compared every text result row before and after.

**change[0]** ("a wider, equally thick outer block raises outer fill and torque if blocks fit; shaft-joint and gearbox loads rise; its grade may tolerate less heat.", 25 words). It was tried with every library part wider than the 6.35 x 3.17 mm outer block, at each design. The wider parts of equal thickness that fit are (outer fill; pull-out, cold-high and gearbox input; magnet limit; temperature verdict):

| Design | Part | Fill | Pull-out = cold-high = input | Magnet limit (°C) | Verdict |
|---|---|---|---|---|---|
| default | B862 (N42) | 0.6198 -> 0.9273 | +14.96 % | 93.06 -> −3.52 | OK -> CHECK |
| 8 magnets | B862 | 0.4959 -> 0.7418 | +28.68 % | 93.06 -> −3.52 | OK -> CHECK |
| 304 | B862 | 0.6198 -> 0.9273 | +1.14 % | 93.06 -> −3.52 | OK -> CHECK |
| Y30 | B862 | 0.6198 -> 0.9273 | +314.21 % (cold-high and input +286.59 %) | 250.00 -> −3.52 | CHECK stays CHECK |
| 20 mm | B862 | 0.3864 -> 0.5781 | +50.24 % | 93.06 -> −3.52 | OK -> CHECK |
| 20 mm | B882 (N42) | -> 0.7728 | +99.73 % | −3.52 | OK -> CHECK |
| 20 mm | B882-N52 | -> 0.7728 | +122.78 % | 0.17 | OK -> CHECK |
| 20 mm | BX082SH (N42SH) | -> 0.7728 | +99.73 % | 93.06 (kept) | OK stays OK |

- All 8 fitting cases raise the outer fill, the pull-out, the cold-high torque and the gearbox input. So "raises outer fill and torque if blocks fit" and "shaft-joint and gearbox loads rise" hold at every design.
- "Its grade may tolerate less heat": seven of the eight picks lower the magnet limit and BX082SH keeps it. Hence "may".
- "Equally thick" is still needed:
  - B861 (9.5 x 1.59) fits and changes the pull-out by −12.76 %, −4.83 % and −36.47 % at the default, 8-magnet and 304 designs.
  - M5026 (6.4 x 1.67) fits and changes it by −12.10 %, −14.36 %, −33.06 % and −18.48 % at the default, 8-magnet, 304 and 20 mm designs.
  - The cold-high torque falls with the pull-out in each case.
- The non-fitting picks are excluded by "if blocks fit".
- The new test `harmonics_outer_part_entry_holds_on_the_example_design` pins the B862 and B861 cases at the default design from the engine, and it passes.

**change[1]** ("with custom dimensions, a wider outer block fills more of its pole and raises torque while the blocks still fit; shaft-joint and gearbox loads rise.", 25 words; aim `model.pullout_Nm` Up).
- It is held by the part pick at four designs (the dump names the pick).
- At the Y30 design, one step (6.35 -> 6.36 mm) moves:
  - the pull-out 0.207074 -> 0.207243 N·m;
  - cold-high 0.338067 -> 0.338343 N·m;
  - the gearbox input 0.071172 -> 0.071230 N·m;
  - the outer fill 0.6198 -> 0.6208.
- A sweep in 0.05 mm steps raises all three at every width up to 9.55 mm, where the blocks are 0.02 mm apart at the faces. Past that they no longer fit, which "while the blocks still fit" names.
- At 9.55 mm the pull-out, cold-high and input are each +15.03 %, the fill is 0.9322, the cup OD stays 41.326 mm, and the magnet mass goes from 25.564 to 32.006 g.
- The only text rows that change are the outer flat check (still OK), the adhesive fatigue screen (84x -> 73x, still OK) and the statuses in the sweep tables. So the named trade-off is the only one that matters at that design.
- The thickness is its own input with custom dimensions, so the entry needs no "equally thick". The grade does not change.

**change[2]** ("... fewer magnets leave wider spaces between blocks.", 23 words; aim `model.fill_outer` Up). One step up (+2 magnets) gives:
- outer fill 0.6198 -> 0.7438 (default, 304, Y30), 0.4959 -> 0.6198 (8 magnets) and 0.3864 -> 0.4637 (20 mm);
- the space between outer blocks (1 − fill) 0.3802 -> 0.2562, 0.5041 -> 0.3802 and 0.6136 -> 0.5363.

So fewer magnets leave wider spaces. The word now matches sentences[0] ("spaces between them") and the square-wave label, and "gap" means only the gap between the rings (sentences[0] and [3]). It does not say more magnets make the coupling stiffer.

**Dependent sentences.**
- sentences[2]: the fill is "the block's width over the room each pole has on that ring". An equally thick, wider block at the same mid-radius raises λ, as change[0] says.
- sentences[0] and [3]: their wording now agrees with change[2] and the label.
- The watch-out does not depend on the changed fields.

**Check g.** Word counts: change[0] 25, change[1] 25, change[2] 23. All three are plain-words clean, with no ids and no typed input label.

**Verdict:** PASS.

Round 5:

No harmonics field changed since round 4, and none depends on pullout_angle change[1]. harmonics change[0] ("a wider, equally thick outer block ...") now has a matching size qualifier in the inner-ring entry of the next note; its text is unchanged. The round-4 PASS stands. I re-ran `harmonics_outer_part_entry_holds_on_the_example_design` (it passes) and re-checked the S-2 numbers above.

**Verdict:** PASS.

Must-fix items applied:
- sentences[0] (round 1, must-fix 1: "small gaps" is false where the space between outer blocks is 38 % to 61 % of each pole): "Around each ring, blocks alternate north and south with small gaps between them, so the magnet pattern along the gap is squarish, not smooth." -> "Around each ring, blocks alternate north and south with spaces between them, so the magnet pattern along the gap is squarish, not smooth."
- change[0].text (round 1, must-fix 2, the A-24 S-2 text: "its flat" is undefined): "a wider outer block raises the outer fill and the torque, while it still fits its flat; it adds magnet mass." -> "a wider outer block raises the outer fill and the torque while the blocks still fit side by side; it adds magnet mass."
- change[0].text (round 2, must-fix 1: wider but thinner parts that fit, B861 and M5026, lower the pull-out and the magnet mass): "a wider outer block raises the outer fill and the torque while the blocks still fit side by side; it adds magnet mass." -> "a wider outer block of equal thickness raises the outer fill and torque while the blocks still fit side by side; it adds magnet mass."
- change[0].text (round 3, must-fix 1: spec 3.1, the entry named no check that the pick turns red; applied in controller ruling 9's wording, which ruling 11 made joint-neutral in place of the reviewer's "clamp loads rise"): "a wider outer block of equal thickness raises the outer fill and torque while the blocks still fit side by side; it adds magnet mass." -> "a wider, equally thick outer block raises outer fill and torque if blocks fit; shaft-joint and gearbox loads rise; its grade may tolerate less heat."

Should-fix items applied by controller ruling 9 (the ruling's text, verbatim):
- change[1].text (round-3 should-fix 1, made joint-neutral): "with custom dimensions, a wider outer block fills more of its pole and raises the torque, while the blocks still fit." -> "with custom dimensions, a wider outer block fills more of its pole and raises torque while the blocks still fit; shaft-joint and gearbox loads rise."
- change[2].text (should-fix 2 of round 1, repeated in rounds 2 and 3): "more blocks on the same ring fill more of each pole; they must still fit, and fewer magnets leave wider gaps between blocks." -> "more blocks on the same ring fill more of each pole; they must still fit, and fewer magnets leave wider spaces between blocks."
- `tests/explain.rs` `harmonics_outer_part_entry_holds_on_the_example_design` (round-3 should-fix 3; round-2 should-fix 2 asked for the same pin with the magnet mass, which the final text no longer claims): at the example design, B862 on the outer ring fits by `model::blocks_fit`, raises `model.fill_outer`, `model.pullout_Nm` and `metal.torque_cold_high_Nm`, and lowers `temperature.summary.magnet_limit_C`; B861 also fits but lowers `model.pullout_Nm`.

Seeds (A-24):
- S-2 (change[0]: "a wider outer block raises the outer fill and the torque while the blocks still fit side by side; it adds magnet mass."): accepted in round 1 for its purpose, removing the undefined "its flat", and applied as must-fix 2. Round 2 found its claim false for wider, thinner parts (must-fix 1). Rounds 3 to 5 reject the seed's exact text: B861 (9.5 x 1.59 mm) and M5026 (6.4 x 1.67 mm) are wider and fit at the example design, yet lower the pull-out by 12.76 % and 12.10 % and the magnet mass by 4.79 g and 7.15 g; and its only trade-off, magnet mass, is no check. The final change[0] keeps the seed's aim.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-5 PASS):
- Round 1 S-1: say where sentences[4] takes the split: "On this design the fills are {model.fill_inner} inner and {model.fill_outer} outer; at pull-out the fundamental gives σ_1 = {model.tau1_Pa} of the total σ = {model.tau_Pa}." (25 words). At the 20 mm design both values are read at the 52.96° pull-out angle, so "σ_1 = 64870 Pa of the total σ = 71440 Pa" can read as the harmonics adding 10 %, while the third harmonic lowers the pull-out by 12.4 % against the fundamental alone. (Superseded by the task review fix round 1 below, which adds "at pull-out" and drops "of the total".)

### Task review fix round 1 (2026-10-15): the harmonic-set entry and the σ line

- The task code review (`task-12-review.md`, Important I1 and I2) found two fields that hold at
  the five probe designs but not on designs the sliders reach with the example design's library
  parts. Its grid: the ring radius from 9 to 30 mm by 0.5 mm against 4 to 40 magnets per ring;
  353 of those designs fit. The fix round reproduced every number below with a scratch probe
  outside the repo.
- **change[3].text** (`coupling.max_harmonic`; input, step and aim unchanged):
  - Old: "a model setting, not a design lever: keep at least 1, 3, 5; adding more moved the
    pull-out under 2 % where checked."
  - New: "a model setting, not a design lever: keep at least 1, 3, 5; more for narrow, widely
    spaced blocks, where higher harmonics can move pull-out." (25 words)
  - Why: "where checked" meant the five probe designs, which the reader cannot see. Summing up
    to 11 in place of 1, 3, 5 moves the pull-out by more than 2 % at 36 of the 353 designs, by
    up to 43.89 % (4 magnets on a 30 mm ring), and every one of the 36 has both fills under
    0.224. T9's G-3 design (4 magnets, 30 mm, 1 mm custom blocks, 0.3 mm gap, no back iron)
    gives +284.43 %. T9's G-3 review had rejected seed S-16 ("Past the fifth they add little")
    for this reason.
  - The new text names the case in the words of the input's own help ("Higher ones matter most
    for narrow, widely spaced blocks"), which `model::tests::higher_harmonics_matter_most_for_narrow_widely_spaced_blocks`
    pins. Narrow blocks keep their higher harmonics strong; a long pole pitch, from few magnets
    on a large ring, lets them cross the gap. With custom block widths (a coarser grid, 634
    fitting designs) the largest change falls as the smaller fill rises:
    - up to 116 % below 0.1, and 49 % between 0.1 and 0.2;
    - 8 to 11 % between 0.2 and 0.5;
    - 5.65 % between 0.5 and 0.6, and 2.66 % between 0.6 and 0.7;
    - at most 1.09 % above 0.7.

    The few cases above 2 % at a fill over one half are 4 or 8 magnets on 18 to 30 mm rings:
    wide blocks, but widely spaced.
- **sentences[4]**:
  - Old: "On this design the fills are {model.fill_inner} inner and {model.fill_outer} outer, and
    the fundamental gives σ_1 = {model.tau1_Pa} of the total σ = {model.tau_Pa}."
  - New: "On this design the fills are {model.fill_inner} inner and {model.fill_outer} outer; at
    pull-out the fundamental gives σ_1 = {model.tau1_Pa}, the summed harmonics σ =
    {model.tau_Pa}." (25 words)
  - Why: both values are taken at the pull-out angle, and σ sums each harmonic there; a
    same-sign third harmonic subtracts at 90°. σ_1 exceeds σ at 114 of the 353 designs, from 6
    to 28 magnets per ring. For example:
    - 12 magnets on a 12 mm ring: 211331 against 210445 Pa;
    - 14 magnets on a 26.5 mm ring: 99669 against 87528 Pa;
    - the example design with only its ring moved to 17.5 mm on the Size tab: 103045 against
      92559 Pa.

    There "σ_1 ... of the total σ" put a part above its whole: a comparison whose truth depends
    on the values, which spec 3.2 forbids. The new line says where both are taken (round 1's
    S-1) and makes no share claim. Rendered at the five probe designs (default, 8 magnets, 304,
    Y30, 20 mm): σ_1 = 200000, 149100, 113800, 15400 and 64870 Pa; the summed harmonics σ =
    201400, 151800, 115100, 15510 and 71440 Pa.
- **Tests.** New: `tests/explain.rs` `harmonics_claims_hold_off_the_probe_designs`, computed
  from the engine:
  - at the slider's fewest magnets on the probes' 20 mm ring (Size tab), the blocks fit, both
    fills are under one half, and summing to 11 moves the pull-out by more than a twentieth
    (+8.58 %); the harmonic-set entry names "narrow, widely spaced blocks" and states no bound
    (no "%");
  - σ_1 lies below σ on the example design and above it on a 17.5 mm ring; the sentence that
    draws both says "at pull-out", and no " of ", "total", "share" or "part" stands between the
    two values. Round 1's S-1 text alone (with "of the total") fails it.

  `harmonics_claims_hold_at_the_probe_designs` keeps its under-2 % pin at the five probe
  designs, as the evidence for "keep at least 1, 3, 5" there.
- A text re-review of these two fields follows (controller ruling 7). Its report goes below as
  `### Re-review: task review fix round 1`.

### Re-review: task review fix round 1

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `5d489bf`
  (`task-12-textreview-1.md`). This is the text re-review of the code-review fix round 1 above
  (controller ruling 7).
- Scope: the two fields the fix round changed, change[3] and sentences[4], and every on-screen
  text whose meaning depends on them: sentences[1] and [3], the watch-out, `pullout_angle`
  sentences[3] ("The fundamental alone peaks at 90°") and the σ gloss in `torque_from_shear`.
  The dump (`NOTES_FOR_REVIEW=harmonics`) draws both fields at the five probe designs with no
  plain-words flag.
- Verdict: MUST_FIX, one item, in sentences[4]. change[3] passes.
- MF-1, applied verbatim (sentences[4]):
  - Old: "On this design the fills are {model.fill_inner} inner and {model.fill_outer} outer; at
    pull-out the fundamental gives σ_1 = {model.tau1_Pa}, the summed harmonics σ =
    {model.tau_Pa}." (25 words)
  - New: "On this design the fills are {model.fill_inner} inner and {model.fill_outer} outer; the
    fundamental alone peaks at {model.amp1_Pa}, the summed harmonics at σ = {model.tau_Pa}."
    (24 words)
  - Why: σ_1 = A_1·sin(φ_pull) is the fundamental's stress at the summed curve's pull-out
    angle. Set beside σ, it reads as what the harmonics do to the pull-out, which holds only
    while the pull-out angle stays at 90°.
    - On the 20 mm ring radius design (pull-out angle 52.96°) the old line read σ_1 = 64870 Pa
      against σ = 71440 Pa, which looks like the harmonics adding 10.1 %. In fact the harmonic
      set lowers the pull-out by 12.09 % against the fundamental alone (A_1 = 81273 Pa; pull-out
      2.8886 -> 2.5393 N·m), and the watch-out's "more than a tenth" then read with the wrong
      sign.
    - On the example design's library-part grid (4 to 40 magnets per ring, every ring radius the
      Size tab reaches from 9 to 30 mm by 0.5 mm, 289 fitting designs) the sign was wrong at 34
      designs, 32 of them with a real effect above 2 %. The worst, 10 magnets on a 23.5 mm ring,
      showed +45.04 % where the real effect is −1.25 %.
  - A_1 (`model.amp1_Pa`) is the pull-out stress of the fundamental alone: with the harmonic set
    at 1, σ equals A_1 exactly, at the five probe designs and on a 17.5 mm ring. So σ against
    A_1 is the set's real effect, with the right sign. On the same grid, σ/A_1 − 1 matches the
    pull-out's change from the set 1 to the design's set at all 289 designs (0 disagreements).
    `model.amp#_Pa` is already in this note's equations, so no equation ownership moves.
  - Rendered at the five probe designs (default, 8 magnets, 304, Y30, 20 mm): the fundamental
    alone peaks at 200000, 149100, 113800, 15400 and 81270 Pa; the summed harmonics at σ =
    201400, 151800, 115100, 15510 and 71440 Pa. On the 20 mm design the harmonics now visibly
    take stress away, as the watch-out says.
  - Tests: `tests/explain.rs` `harmonics_claims_hold_off_the_probe_designs` now finds the
    sentence that draws `{model.amp1_Pa}` and `{model.tau_Pa}`, asserts that it names "the
    fundamental alone", and keeps the check that no " of ", "total", "share" or "part" stands
    between the two values. Its "at pull-out" assert gave way to engine pins:
    - with the harmonic set at 1, `model.tau_Pa` equals the design's `model.amp1_Pa` (1e-12
      relative) at the five probe designs and on the 17.5 mm ring;
    - at the same designs, σ − A_1 has the sign of `model.torque_2d_Nm` with the design's set
      minus `model.torque_2d_Nm` with the set 1;
    - on the 20 mm design, σ_1 < σ < A_1: the case the old line misread.

    The σ_1-against-σ asserts on the example design and the 17.5 mm ring stay, as the record of
    why the line uses no share wording.
- Should-fix, not in this round's list, left open for the controller:
  - SF-1 (optional): change[3]'s "narrow" leaves out wide blocks on a long pitch. With custom
    widths (10,787 fitting designs) blocks wider than their spaces on 4 to 6 magnets and a large
    ring still move by up to about 5 % when summing to 11, inside the model's stated
    uncertainty. The review offers "a model setting, not a design lever: keep at least 1, 3, 5;
    more for widely spaced blocks, where higher harmonics can move pull-out." (24 words), with
    the test's phrase becoming "widely spaced blocks". Not applied; the current text stands as
    true.

The reviewer's report for the record, as returned:

**change[3]** ("a model setting, not a design lever: keep at least 1, 3, 5; more for narrow,
widely spaced blocks, where higher harmonics can move pull-out."): PASS.
- Physics:
  - Narrow blocks keep their higher-harmonic amplitudes close to the fundamental's:
    B_n ∝ sin(nπλ/2)/n, which is about πλ/2 for small nλ.
  - A long pole pitch against the gap and the magnet thickness lets those harmonics cross the
    gap: S_n grows with n while n·k·t is small, and fades as e^(−n·k·g) only when the pitch is
    short.
  - So the truncation after 5 matters exactly where the entry says.
- Library parts grid (example design, 4 to 40 magnets, every reachable ring radius by 0.5 mm,
  305 fitting designs):
  - 14 designs move by more than 2 %. All have 4 magnets on a 16.5 to 23 mm ring, with fills of
    at most 0.224 and 0.178, and all move upward, up to +16.78 % at 23 mm.
  - The task review's grid, with ring radii to 30 mm, found 36 of 353, up to 43.89 %.
- The entry states no bound and no direction. That is right, because the custom-width grid
  moves both ways.
- "Keep at least 1, 3, 5" is right:
  - at low fill the third harmonic moves the pull-out by more than a tenth (the watch-out; at
    the 20 mm design −12.09 % from the set 1 to the set 1, 3, 5);
  - more harmonics only bring the model closer to its own square-wave limit.
- "Not a design lever": the setting changes the model, not the hardware. The bench factor is
  recomputed with the same set (on the 304 design 1.0495 against 1.0555 at the set 1), so it
  stays like for like.
- It is consistent with sentences[3] ("higher pairs usually add little", hedged), with the
  watch-out (the third harmonic at low fill), and with the input's help.
- Caps and words: 25 words (cap 25). No input label is typed, the drawn label comes from
  `label_for`, and "1, 3, 5" names the choice "1, 3, 5 (default)". Plain words are clean at all
  five designs (dump). No trap is written.

**sentences[4]** (current text): literally true, misleading as a pair (MF-1).
- σ_1 and σ are both read at φ_pull (`model.rs:997-1000, 1106, 1143`), so "at pull-out" is
  accurate. σ_1 exceeds σ on the 17.5 mm ring (103045 against 92559 Pa), so dropping "of the
  total" was right.
- The remaining defect is the implied comparison, on the 20 mm probe design and at 34 of 289
  library-part grid designs (MF-1).
- Caps and words: 25 words (cap 25). Plain words are clean at all five designs. The placeholders
  sit in the "On this design" clause.

**Traps:** neither field writes any item of slice-2 spec section 8, slice-1 spec section 8,
`phys-new-notes.md` T-A to T-P, or the added trap "more magnets make the coupling stiffer". The
fill wording (T-A) is in sentences[2] and unchanged.

**Seeds:** none were assigned to this re-review.

**Repo state:** this review is read-only. It wrote only this file and the dump under
`magcoupling-rs/target/`. The probes ran from the session scratchpad.

### Re-review: code-review fix round 1 (re-check 2)

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `925350a`
  (`task-12-textreview-2.md`): the re-review above's MF-1 as applied. This is the second text
  re-review of the code-review fix round 1 (controller ruling 7).
- Scope: `harmonics` sentences[4], "On this design the fills are {model.fill_inner} inner and
  {model.fill_outer} outer; the fundamental alone peaks at {model.amp1_Pa}, the summed harmonics
  at σ = {model.tau_Pa}." (24 words), and every on-screen text whose meaning depends on it:
  sentences[1] to [3], change[3], the watch-out, the note's square-wave diagram, `pullout_angle`
  sentences[3] ("The fundamental alone peaks at 90°") and the σ gloss in `torque_from_shear`.
  The dump (`NOTES_FOR_REVIEW=harmonics`) draws the line at the five probe designs with no
  brace left and no plain-words flag.
- Verdict: PASS, no must-fix.
  - "The fundamental alone peaks at A_1" holds at every design the app reaches: A_1 > 0, and
    with the harmonic set at 1 σ equals A_1 bit for bit.
  - σ is the summed curve's maximum. The reviewer's grid: the example design, no back iron,
    304, arcs and Y30 ferrite; 4 to 40 magnets per ring; every ring radius the Size tab reaches
    from 9 to 30 mm by 0.5 mm; every harmonic set from 1 to 11. At all 22,200 design-sets (8,976
    of them fit), σ equals a brute-force maximum over 200,001 samples, with a worst relative gap
    of 8.2e-10.
  - Both values read the same circuit, the effective back iron after the material pick.
  - The line states no ordering, which is right: σ < A_1 at 132 of the 289 fitting library-part
    designs and σ > A_1 at 157. σ − A_1 is the harmonic set's effect on the stress and the 2D
    torque, with the right sign at every design.
  - Caps, plain words and traps: 24 words, clean at the five probe designs, no trap written.
- Should-fix, left open for the controller (both optional; no code, test or on-screen text
  changed in this step):
  - SF-1 (clarity): sentences[1] and [2] and the square-wave diagram call the magnet pattern's
    first harmonic, a field, "the fundamental". The diagram's n = 1 curve peaks above the summed
    one (1.24 against about 1.01 at the diagram's fill of 0.85). This line's "fundamental alone"
    is the pair's shear stress, A_1, which at the default design lies below σ (200000 against
    201400 Pa). The unit and σ keep the line correct, and `pullout_angle` uses the same phrase
    for the same stress curve, but a reader who sets the line beside the picture can think the
    two contradict. The review offers "Here the fills are {model.fill_inner} inner and
    {model.fill_outer} outer; the fundamental alone gives a peak shear stress of
    {model.amp1_Pa}, the summed harmonics σ = {model.tau_Pa}." (25 words), which keeps every
    check of `harmonics_claims_hold_off_the_probe_designs`. Not applied: it would change
    reviewed on-screen text, and the review says the current text stands as true.
  - SF-2 (test strength): pin (iii) of `harmonics_claims_hold_off_the_probe_designs`, the sign
    of σ − A_1 against the 2D torque's change from the set 1, follows from pin (i) and
    T_2D = σ·A_L, where A_L does not depend on the set, so it adds no protection. To pin the
    reader's step from the line to the watch-out, the review suggests the change in
    `model.pullout_Nm` at the probe designs instead. It agrees in sign at all five; on the 304
    design the bench factor is refitted with the set, giving +1.129 % in stress against
    +0.546 % in pull-out. Off the probe designs, bench-calibrated free-space designs with 10
    magnets can disagree (12 of 3,675 design-sets each for no back iron and 304, all off the
    default set, under 1.4 %), so such a pin belongs at the probe designs only.
  - Carried from the re-review above, unchanged: its SF-1 (change[3]'s "narrow").

The reviewer's report for the record, as returned:

**sentences[4]** (current text, 24 words): PASS.

**Physics.**
- "The fundamental alone peaks at A_1" is true at every design the app reaches:
  - The fundamental's stress against electrical angle is A_1·sin φ, with A_1 = B_i,1·B_o,1·S_1/(2μ0).
  - A_1 > 0 at every design, because B_n is proportional to sin(πλ/2) > 0 for fills in (0, 1], S_1 > 0, and Br at the operating temperature is positive. So its peak is A_1, at 90°.
  - With the harmonic set at 1 the engine's pull-out stress is A_1 exactly, so "alone" is literal. It is what the user sees on choosing "1" in the same note's harmonic-set entry.
- "The summed harmonics [peak] at σ" is true:
  - The app always runs E7 (`compute_all` uses `Deviations::ALL`).
  - `tau_Pa` is the summed curve's maximum, and it equals the brute-force maximum at all 22,200 design-sets.
- Same circuit: both values read `ci.backiron`, the effective circuit after the back-iron material pick. The 304 probe draws both in free space.
- No ordering and no share:
  - The line states no ordering, which is right. σ < A_1 at 132 of the 289 fitting library-part designs and σ > A_1 at 157.
  - σ − A_1 is the harmonic set's effect on the stress and the 2D torque, with the right sign at every design.
- Size of that effect at the default set (library parts, 289 fitting designs):
  - more than 2 % at 162 designs and more than 10 % at 99;
  - up to +271.6 % at 4 magnets on a 23 mm ring;
  - every design above 10 % has a smaller fill under 0.5;
  - at a smaller fill of 0.5 or more the most is 2.65 %.

  This agrees with sentences[3] (hedged: "usually add little") and with the watch-out ("At low fill ... the third harmonic can change the pull-out by more than a tenth"). The fills drawn in the same line let the reader connect the two.
- Bench-calibrated designs, for the record only (the line claims nothing about pull-out):
  - Free-space designs that keep the prototype's 10 B842SH magnets use the bench factor, which is refitted with the harmonic set.
  - Off the default set, on a ring other than the prototype's, the pull-out's change from the set 1 can have the opposite sign to σ − A_1. This happens at 12 of 3,675 design-sets for the no-back-iron variant and 12 for the 304 variant, all at 10 magnets with the sets 3, 7, 9 or 11.
  - Both changes stay under 1.4 % there. The worst is the set 3 on a 13.5 mm ring: +0.20 % stress, −1.16 % pull-out.
  - At the default set 5 there is no disagreement on any grid. On steel designs the two always agree, because f_cal,0 does not depend on the set.
  - Re-review 1's record ("0 disagreements" on the steel example grid; the same sign on the 304 probe) is accurate as written.

**Consistency.**
- `pullout_angle` sentences[3] uses "The fundamental alone peaks at 90°" for the same curve.
- `torque_from_shear` glosses σ as the shear stress "at its peak".
- The torque-angle diagram labels its dashed curve "fundamental alone".
- The square-wave diagram's fundamental is the magnet pattern (SF-1, optional).

**Caps and words.**
- 24 words, under the 25 cap, and the note still has 5 sentences.
- The placeholders sit in the "On this design" clause, with no unit typed after them.
- The glyphs are σ and =, with no ∝, ϑ or U+207B, and this is not sentence 0.
- No input label is typed.
- `plain_text_violations` and `audit_ids` return nothing at the five probe designs.

**Traps.** The line writes no item of slice-2 spec section 8, slice-1 spec section 8, `phys-new-notes.md` T-A to T-P, or the added "more magnets make the coupling stiffer" trap. The fill wording (T-A) is unchanged.

**Tests and docs.** `harmonics_claims_hold_off_the_probe_designs` pins the line as follows:
- the sentence draws `{model.amp1_Pa}` and `{model.tau_Pa}` and names "the fundamental alone";
- no share word stands between the two values;
- σ(set 1) = A_1 at the five probe designs and on a 17.5 mm ring;
- σ_1 < σ < A_1 on the 20 mm design.

The README row, `05-update-tracker.md`, BL-056 and the record's "Re-review: task review fix round 1" describe the line and the test accurately.

**Repo state.** This review is read-only. It wrote only this file and the dump under `magcoupling-rs/target/`. The probes ran from the session scratchpad (`hsweep2`).

## back_iron_factor

Task T12. Rewrite of "Back iron: the sinh factor against free space" as "Back iron: why steel
behind the magnets adds torque", Start here step 3; its flux diagram letters each block N or S
(TC-24). Reviewer: physics reviewer (session model), fresh context. Date: 2026-10-14 (the date
the slice plan fixes for the note's `review` field; all five rounds ran on 2026-10-15).

Rounds: 5. Verdict: PASS.

Verdict by round: 1 MUST-FIX (1 item); 2 PASS; 3 PASS (no field changed, not re-reviewed); 4 PASS
(sentences[2] changed under ruling 9); 5 PASS (no field changed, not re-reviewed).

Round 1:

**What I read**
- The dump (back_iron_factor at the five designs).
- The literal in `notes.rs` (lines 259-306).
- `diagrams.rs` `flux_panel` and `flux_paths`.
- `model.rs`: `geometry_factor` (:632-643), `Harmonic::s`/`amplitude` (:614-630), `shear_stress` (k = n·(N/2)/R_g, :695) and `compute` (the wall rule at :1033-1052).
- `material_library.rs`: `BACK_IRON_CHOICES` (:731-740) and `resolve` (:880-900).
- `warnings.rs`: `LOW_SATURATION_T` and the low-saturation rule.
- Spec R11 and section 8.
- phys-notes-a section 3.
- phys-new-notes T-D, T-K and T-L.

**Numbers reproduced:**

| Design | S_1 steel / free | Gain | e^(−π·g/τ_p) | Circuit |
|---|---|---|---|---|
| default | 0.2445 / 0.1392 | 1.757 | 0.607 (π·g/τ_p = 0.4993) | steel |
| 8 magnets | 0.2373 / 0.1188 | 1.998 | 0.671 | steel |
| 304 back iron | 0.2445 / 0.1392 | 1.757 | 0.607 | free space (circuit 0) |
| Y30 custom | 0.2445 / 0.1392 | 1.757 | 0.607 | steel |
| 20 mm ring | 0.2096 / 0.0878 | 2.388 | 0.746 | steel |

These match the dump (including "0.08780").

**Check a and check d: physical truth.**
- Sentence 0: false at the 304 design as written (must-fix 1).
- Sentence 1: true. S_iron/S_free = (1+e^(−k·t_i))(1+e^(−k·t_o))/(1−e^(−2k(t_i+t_o+g))) > 1 for every n, derived from `geometry_factor`. S_n is the factor that turns B_i,n·B_o,n/(2μ0) into shear stress, not "how much field reaches through".
- Sentence 2: true. The steel gain is 3.01 at 4 magnets (τ_p 22.0 mm), 1.757 at 10 and 1.14 at 24 (3.67 mm), so it is largest when the pole pitch is long against the magnet thickness.
- Sentence 3: true. The free factor carries e^(−k·g), k_1 = π/τ_p. With steel, d ln S/dg = −k·coth(k(t_i+t_o+g)), which is −1.008·k at the default.
- Sentence 4: true, and it is not the "fewer poles reach across better" trap. S_1 with steel at the default ring is:
  - 0.1630 (4 magnets), 0.2120 (6), 0.2373 (8), 0.2445 (10), 0.2397 (12), 0.2281 (14) and 0.1495 (24): best near 10 magnets, τ_p 8.8 mm;
  - at the 20 mm ring, best near 16-18 (0.2439 and 0.2441, τ_p 9.4 and 8.3 mm);
  - it tends to 0 for a long pitch and a short pitch (the gap is at least 0.3 mm on the slider).

**Check b: change entries.**
- change[0]: true on the library.
  - Codes 1 (4140) and 2 (1018) keep the steel circuit with no caution.
  - Codes 3-6 (12L14, 416, 17-4PH H1150 and H900) keep it but trip the low-saturation caution ("may saturate"; R11 {3,4,5,6}). None is offered as a clean swap.
  - Codes 7 (304) and 8 (6061-T6) give circuit 0 and 1.6974 against 2.6885 N·m (−36.9 % on screen, including the bench factor 1.0495).
- change[1]: true. With Back iron 0 every code gives circuit 0. At the 304 design the dump draws it as set by the material pick.
- change[2]: one step (1.40 -> 1.39 mm) raises the pull-out at all five designs:
  - default 2.688476 -> 2.697267;
  - 8 magnets 1.968360 -> 1.974247;
  - 304 1.697369 -> 1.703137;
  - Y30 0.207074 -> 0.207751;
  - 20 mm 2.539258 -> 2.545905.
  The sweep to 0.3 mm rises monotonically (default 3.851 N·m). The trade-off (running clearance, sleeve and liner room) is named.
- change[3] (A-20): held by the part pick at four designs. At the Y30 design:
  - one step (3.17 -> 3.18 mm) moves S_1 with steel 0.244488 -> 0.244683;
  - the sweep to 10 mm gives 0.2616 (4.54), 0.2680 (5.91), 0.2704 (7.27), 0.2713 (8.64) and 0.2716 (10.0): shrinking gains;
  - the outer fill falls 0.620 -> 0.512;
  - the pull-out peaks at about 5.2 mm (0.2173 N·m) and falls to 0.2053 at 10 mm.
  The cup grows (A_back = A_o + t_o feeds the pocket radius and the OD). Every clause holds.
- watch_out[0]: true, and T-L is respected. The wall check never changes the torque. The cup wall reads "Too thin" at the default (wants 2.0 mm), 8 magnets (2.4) and 20 mm (3.3), "OK" at Y30 and "No back iron" at 304.

**Check g.**
- Word counts: sentences 23, 21, 24, 23 and 24; changes 24, 18, 21 and 24; watch-out 22.
- Material names are plain designations, not input labels.
- "the cup wall check" names a result row, not a typed input label.
- The title is plain and unique.

**Check h: sources.** Correct and plain: M3 (planar against cylindrical), M5 (steel at the magnet backs), and TC-11 corrected (the best pole pitch) as reproduced above.

**Trap check.** "416, 17-4PH or 12L14 are clean swaps": not written. "Steel gives the torque shown" without the caveat: not written (watch-out). "Fewer poles reach across the gap better": not written. "1.5 T is 4140's saturation": not written.

**Verdict:** MUST-FIX (1).

Round 2:

**What I read**
- The dump (back_iron_factor at the five designs).
- The literal in `notes.rs` (lines 259-306).
- `diagrams.rs` flux labels ("with back iron", "free space").
- The 304 probe design's circuit flag.

**sentences[0] (new: "Magnetic steel behind each ring ...").** True at all five designs as a statement of what magnetic back iron does.
- The steel carries each magnet's back-side flux to the next pole, so the gap side gets the field that would otherwise loop behind the magnets.
- S_1 with steel against free space is 0.2445/0.13915 (default, 304, Y30), 0.2373/0.11881 (8 magnets) and 0.2096/0.08780 (20 mm), as the dump draws.
- At the 304 design the sentence no longer misdescribes the design's steel. 304 is non-magnetic: circuit 0, and the torque uses S_1 = 0.13915 (pull-out 1.69737 N·m).
- Sentence-0 rules hold (no symbol, no placeholder). It has 24 words, plain-words clean.
- Traps checked:
  - "416 is a safe back iron": not implied. Magnetic steels that trip the low-saturation caution stay named in change[0] as "may saturate".
  - "Steel gives the torque shown": not implied. The watch-out keeps the saturation caveat.

**Dependent fields.**
- sentences[1] ("is therefore always larger with steel") now refers back to "Magnetic steel" through "therefore": correct.
- sentences[2]: correct as the shorthand of sentences[0]; should-fix 1 makes it explicit at the 304 design.
- The title ("why steel behind the magnets adds torque") reads as the magnetic case, with "back iron" in the same title. Unchanged and not re-reviewed.
- The flux diagram's "with back iron" panel is the magnetic case. It matches.

**Verdict:** PASS.

Round 3:

No field of this note changed since round 2.
- The dump draws the round-2 text at all five designs, and change[3] keeps A-20's text and aim (`model.s1_iron` Up).
- The A-13 sources are unchanged.
- No changed harmonics field feeds a sentence of this note.
- Per Step 9 ("a note that returned PASS in an earlier round is not re-reviewed unless one of its fields changed"), the round-2 PASS stands.

**Verdict:** PASS (not re-reviewed).

Round 4:

**sentences[2]** ("On this design S_1 is {model.s1_iron} with magnetic steel and {model.s1_free} without; the gain is largest when the pole pitch is long against magnet thickness.", 25 words). The reproduced values match the dump:

| Design | S_1 with magnetic steel / without | Gain | Circuit used |
|---|---|---|---|
| default | 0.2445 / 0.13915 | 1.757 | steel |
| 8 magnets | 0.2373 / 0.11881 | 1.998 | steel |
| 304 | 0.2445 / 0.13915 | 1.757 | free space (304 is not magnetic) |
| Y30 | 0.2445 / 0.13915 | 1.757 | steel |
| 20 mm | 0.2096 / 0.08780 | 2.388 | steel |

- "With magnetic steel" removes the 304-design misreading. That design's back iron is a steel, yet its torque uses the free-space 0.1392, and the sentence no longer suggests otherwise.
- The gain claim is general, and true. On the default ring, with both blocks 3.17 mm thick:

| Magnets | Pole pitch (mm) | Gain |
|---|---|---|
| 4 | 22.02 | 3.008 |
| 6 | 14.68 | 2.358 |
| 8 | 11.01 | 1.998 |
| 10 | 8.81 | 1.757 |
| 12 | 7.34 | 1.583 |
| 16 | 5.51 | 1.355 |
| 24 | 3.67 | 1.137 |

- The placeholders sit in the "On this design" clause. The comparison does not depend on their values (spec 3.2).
- It is plain-words clean with no ids.
- Dependent sentences: sentences[0] ("Magnetic steel ...") and sentences[1] ("always larger with steel", referring back) read consistently with it. No harmonics change feeds this note.

**Verdict:** PASS.

Round 5:

No back_iron_factor field changed since round 4, and none depends on the changed pullout_angle entry. The round-4 PASS stands. All 14 of its fields are still plain-words clean with no ids and no "clamp".

**Verdict:** PASS.

Must-fix items applied:
- sentences[0] (round 1, must-fix 1: "Steel" is false for the 304 design's steel, which is not magnetic and runs on the free-space circuit): "Steel behind each ring gives the flux an easy return path, so more of it crosses the gap instead of leaking out behind." -> "Magnetic steel behind each ring gives the flux an easy return path, so more of it crosses the gap instead of leaking out behind."

Should-fix items applied by controller ruling 9 (the ruling's text, verbatim):
- sentences[2] (round-2 should-fix 1: at the 304 design, "with steel" reads as the design's own 304): "On this design S_1 is {model.s1_iron} with steel and {model.s1_free} without; the gain is largest when the pole pitch is long against magnet thickness." -> "On this design S_1 is {model.s1_iron} with magnetic steel and {model.s1_free} without; the gain is largest when the pole pitch is long against magnet thickness."

Amendment applied before round 1 (A-20, binding): change[3].text "a thicker outer block raises S_n, with diminishing returns past about a third of a pole pitch; the cup grows with it." -> "a thicker outer block raises S_n with shrinking gains, but it lowers the outer fill, so past a point torque falls; the cup grows." Round 1 confirmed every clause at the Y30 design, the only design where the outer thickness is free.

Seeds (A-24): none names back_iron_factor.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-5 PASS):
- Round 1 S-1: "sideways pull across the gap" in place of "pull across the gap" in sentences[1] (22 words), the name step 1 gives the shear stress, so that a reader does not take it for the radial attraction between the rings.

## pullout_angle

Task T12. Rewrite of "Pull-out torque against rotation angle" as "Pull-out: the most torque before
the rings slip", Start here step 4 and the dashboard Why link of the pull-out row (and of the
gearbox ripple row until T15); its torque-angle diagram draws the fundamental alone dashed (TC-22).
The end-fringing diagram's brackets (TC-24, shown by `end_effect`) were reviewed in this round.
Reviewer: physics reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice
plan fixes for the note's `review` field; all five rounds ran on 2026-10-15).

Rounds: 5. Verdict: PASS.

Verdict by round: 1 MUST-FIX (1 item); 2 PASS; 3 PASS (no field changed, not re-reviewed); 4
MUST-FIX (1); 5 PASS.

Round 1:

**What I read**
- The dump (pullout_angle at the five designs).
- The literal in `notes.rs` (lines 307-356).
- `diagrams.rs` `torque_angle` (the T12 version) and `t12-diagrams.png`.
- `model.rs`:
  - `peak_off_half_pitch` (:728-746);
  - `stationary_polynomial`, whose Q(0) = a1 − 9·a3 + 25·a5 gives the "about a ninth" rule (:752);
  - `at_pull_out` (:876-892);
  - `compute`: T_pull = T_2D·f_end·f_cal at :1002-1004, Br at `ci.op_temp_C` at :978-979, `gearbox_input_ripple_Nm` at :1156;
  - `select_calibration_factor` (:584-600).
- `api.rs` (f_cal selection; `metal.op_temp_C` is `ci.op_temp_C` passed through).
- `library.rs` (B842SH 150 °C, B842-N52 1.45 T at 80 °C).
- `temperature.slip_loss.drag_Nm`.
- phys-notes-a section 4.
- phys-new-notes T-E, T-H and T-M.
- Spec section 8.

**Numbers reproduced** (pull-out; angle electrical and twist; A3/A1; drag as a share of the pull-out):

| Design | Pull-out (N·m) | Angle / twist | A3/A1 | Drag |
|---|---|---|---|---|
| default | 2.6885 | 90° / 18.0° | −0.0109 | 0.0131 N·m, 0.49 % |
| 8 magnets | 1.9684 | 90° / 22.5° | −0.0085 | 0.70 % |
| 304 back iron | 1.6974 (f_cal 1.0495) | 90° / 18.0° | −0.0179 | 0.26 % |
| Y30 custom | 0.2071 | 90° / 18.0° | −0.0109 | 6.3 % (the stored slip fields do not follow the ferrite grade) |
| 20 mm ring | 2.5393 | 52.96° / 10.59° | +0.2158 | 1.84 % |

The operating temperature is 50 °C at all five, and `metal.op_temp_C` = `coupling.op_temp_C` at all five.

**Check a and check e: physical truth.**
- Sentence 0: true. It says nominal estimate, not guaranteed, and answers "what is the pull-out torque".
- Sentence 1: true. It names T_2D·f_end·f_cal at the operating magnet temperature. The "here:" placeholder renders 50.00 °C.
- Sentence 2: true. φ = N/2 × twist; 90° = 180°/N = 18° at 10 magnets.
- Sentence 3: true. Only the 20 mm design has a same-sign third above a ninth: −1 + 9·0.2158 + 25·0.0031 = +1.02 > 0, so the peak is at 52.96°.
  - At the 8-magnet design the inner fill (0.689) is above 2/3. The thirds are then of opposite sign and the peak stays at 90°.
  - The first peak lies below 90° (S-1 rejected).
- Sentence 4: true (T-M). The torque swings ±pull-out once per pole pair. The average is only the drag, at most 1.84 % of the pull-out on the NdFeB designs.
- watch_out[0]: true. The rings lock again only once the slip speed falls, never "below pull-out".

**Check b: change entries.**
- change[0] (axial length, blank seed): one step raises the pull-out at all five designs (2.688476 -> 2.690839 at the default). The sweep to 50.8 mm is linear: pull-out ∝ L − c_end·τ_p, 11.69 N·m at the default. The trade-offs are true: stack length, and clamp and gearbox loads through cold-high.
- change[1]: false at the 304 design (must-fix 1). The proportion holds exactly:
  - ×1.115385 at the default, 8-magnet and 20 mm designs (B842-N52 for B842SH, inner or outer);
  - ×4.019 at the Y30 design, which is the Br ratio at 50 °C (the grades' coefficients differ; at 20 °C the ratio is 3.919).
  The rating falls from 150 to 80 °C ("often tolerates less heat" holds).
- change[2] (operating temperature, Lower): one step raises the pull-out at all five designs (2.688476 -> 2.691824 at the default). The sweep to −40 °C is monotone (3.3246 N·m at the default). The "only if they really run cooler" trade-off is named.

**Check g.**
- Word counts: sentences 24, 23, 24, 24 and 24; changes 16, 23 and 21; watch-out 25. Must-fix 1's replacement is 25.
- Sentence 0 has no symbol and no placeholder.
- The title is plain and unique against the dashboard's "Pull-out torque ..." rows.
- R8 glosses: electrical angle, φ, T_2D·f_end·f_cal and Br ("built-in flux density") are all glossed. N and T_2D come from step 1.

**Check h: sources.** Correct and plain: E7, the one peak search, and PA-8 with TC-12. They omit the past-pull-out finding (should-fix 2).

**Trap check.**
- "Pull-out is guaranteed": not written.
- "The rings re-lock below pull-out": not written; the watch-out states the opposite.
- "The coupling protects the gearbox": not written.
- "A hotter-rated grade is free": not written.
- "More magnets make the coupling stiffer": not written (phys-notes-a's draft list had it, and the note rightly leaves it out).
- DRY: change[0] repeats torque_from_shear change[0] almost word for word. Accepted: the pull-out row's Why link reaches this note alone, and the axial length is its most direct lever.
- A-27: `when_slipping` (T15) must not repeat sentences[4] and the watch-out.

**Verdict:** MUST-FIX (1).

Round 2:

**What I read**
- The dump (pullout_angle at the five designs).
- The literal in `notes.rs` (lines 307-356).
- `select_calibration_factor` (`model.rs` :583-600). The bench factor applies only with no back iron, the prototype's magnets per ring, and B842SH on both rings.
- `library.rs` grades and Br: B842SH N42SH 1.29 T, 1.30 T with correction E3; B842 N42 1.30 T; B842-N52 N52 1.45 T.
- The calibration note's own wording ("measured on the bench") for terminology.
- `tests/explain.rs` `pullout_angle_claims_hold_on_the_engine` (lines 1333-1356).

**change[1] (new: "..., unless the bench-measured calibration then stops applying; it often tolerates less heat").** Inner part to B842-N52 at each design; pull-out ratio against the Br ratio at the operating temperature:

| Design | Pull-out (N·m) | Ratio | Br(op) ratio | f_cal |
|---|---|---|---|---|
| default | 2.68848 -> 2.99869 | ×1.115385 | ×1.115385 | 0.95 -> 0.95 |
| 8 magnets | 1.96836 -> 2.19548 | ×1.115385 | ×1.115385 | 0.95 -> 0.95 |
| 304 | 1.69737 -> 1.71381 | ×1.009686 | ×1.115385 | 1.0495 -> 0.95 |
| Y30 (parts blank) | 0.20707 -> 0.83222 | ×4.018976 | ×4.018976 (×3.918919 at 20 °C) | 0.95 -> 0.95 |
| 20 mm | 2.53926 -> 2.83225 | ×1.115385 | ×1.115385 | 0.95 -> 0.95 |

- The proportion is exact wherever the calibration stays. At the 304 design the clause names exactly why it fails.
- The proportion is to Br at the running temperature. Sentence[1] sets that context ("computed at the magnet temperature while running").
- An equal-Br pick (B842) at the 304 design drops the pull-out ×0.905236 for the same reason. B842 is not "a stronger grade", but the clause still warns the reader that a pick can end the bench factor.
- "It often tolerates less heat" holds: B842-N52 is rated 80 °C against 150 °C, as the existing test pins.
- The round-1 drop of "the outer part likewise" loses nothing that this input's entry needs.

**Traps.**
- "The bench calibration applies beyond prototype-matching designs": not written. The clause says it applies and then stops.
- "A hotter-rated grade is free": not written. The entry says a stronger grade often tolerates less heat.
- "More magnets make the coupling stiffer": not written.

**Dependent fields.** sentences[1] ("end and calibration factors", T_pull = T_2D·f_end·f_cal) is consistent with the clause. Nothing else in the note depends on the change.

**Check g.**
- 25 words (at the cap); plain-words clean, no ids.
- "Br" is glossed in the same entry ("built-in flux density").
- No input label is typed.

**Verdict:** PASS.

Round 3:

No field of this note changed since round 2.
- The dump draws the round-2 text at all five designs, and the A-13 sources are unchanged.
- The S-1 seed is rejected (see the amendment checks above). From my probe, `model.pullout_angle_rad` is 1.57080 (90.00°) at four designs and 0.92434 (52.96°) at the 20 mm design.
- The round-2 PASS stands.
- The clamp point in should-fix 1 comes from my probe of the B842-N52 inner pick at the five designs. It is recorded so that it is not lost, not as a must-fix in a round where this field did not change.

**Verdict:** PASS (not re-reviewed).

Round 4:

**What I read:** as for harmonics, plus `model.rs` (`br_inner_T_op` at :264, `pullout_angle_rad` at :1152) and `library.rs` (M5044, M5045 and M5026 are SuperMagnetMan arcs; B842-N52 is K&J's N52 block at B842SH's size).

**change[0]** ("pull-out grows almost linearly with the overlap; the stack lengthens, and shaft-joint and gearbox loads rise.", 16 words; aim `model.pullout_Nm` Up). One step (12.70 -> 12.71 mm; the input is blank and seeded from the overlap) raises the pull-out, cold-high and gearbox input at all five designs:

| Design | Pull-out (N·m) | Cold-high (N·m) | Gearbox input (N·m) |
|---|---|---|---|
| default | 2.688476 -> 2.690839 | 3.823310 -> 3.826670 | 0.804907 -> 0.805615 |
| 8 magnets | 1.968360 -> 1.970141 | 2.799225 -> 2.801759 | 0.589311 -> 0.589844 |
| 304 | 1.697369 -> 1.698861 | 2.413847 -> 2.415968 | 0.508178 -> 0.508625 |
| Y30 | 0.207074 -> 0.207256 | 0.338067 -> 0.338364 | 0.071172 -> 0.071235 |
| 20 mm | 2.539258 -> 2.541688 | 3.611106 -> 3.614561 | 0.760233 -> 0.760960 |

At 50.8 mm all three are higher still (default 11.6905 / 16.6252 / 3.5000 N·m). Every clause holds. Only the trade-off wording changed this round ("clamp and gearbox" became "shaft-joint and gearbox").

**change[1], current text** ("a stronger grade raises pull-out in proportion to its built-in flux density Br unless calibration changes; shaft-joint and gearbox loads rise; heat tolerance often drops.", 25 words). These are the inner-ring picks with a higher Br than the current block. Pull-out ratio; fit; cold-high change, which equals the gearbox-input change:

| Part (w x t x L mm, grade) | default | 8 magnets | 304 | Y30 | 20 mm |
|---|---|---|---|---|---|
| B842-N52 (6.35 x 3.17 x 12.7, N52, same size) | x1.115385; fits; +11.54 % | x1.115385; fits; +11.54 % | x1.009686 (f_cal 1.0495 -> 0.95); fits; +0.97 % | x4.018976; fits; +275.10 % | x1.115385; fits; +11.54 % |
| B882-N52 (12.7 x 3.17 x 12.7, N52) | x1.140467; no fit | x1.315905; no fit | x1.030827; no fit | x4.109354; no fit | **x2.213568; fits; +121.36 %** |
| M5044 (arc 5.5 x 1.31 x 13, N50) | **x0.614561; fits; −38.54 %** | **x0.604027; fits; −39.60 %** | **x0.477403; fits; −52.26 %** | x2.214398; fits; +106.67 % | **x0.615606; fits; −38.44 %** |
| M5045 (arc 5.6 x 1.12 x 6.56, N50M) | **x0.260125; fits; −73.99 %** | **x0.251279; fits; −74.87 %** | **x0.196984; fits; −80.30 %** | **x0.937288; fits; −12.52 %** | **x0.235937; fits; −76.41 %** |
| M5026 (arc 6.4 x 1.67 x 15, N50) | **x0.748577; fits; −25.14 %** | **x0.781540; fits; −21.85 %** | **x0.602630; fits; −39.74 %** | x2.697288; fits; +151.74 % | **x0.775969; fits; −22.40 %** |

- At Y30 every library part has a higher Br than Y30. Of the same-size ones, B842SH and B842 give x3.603220 each, equal to their Br ratio at 50 °C.
- The bold cells contradict the entry: the pull-out and the loads fall, or the rise is far from "in proportion".
- At four designs, three of the four stronger-grade parts that fit lower the pull-out.

**Must-fix 1's replacement, checked.** These are the same-size, stronger-grade picks (equal width, thickness and length) at each design. The pull-out ratio matches the Br ratio at the running temperature (`model.br_inner_T_op`) times the f_cal ratio to 2.2e-16 in every case:
- default, 8 magnets and 20 mm: B842-N52, x1.115385, cold-high and input +11.54 %, rated 150 -> 80 °C.
- 304: B842-N52, x1.115385 x 0.905236 = x1.009686 (the bench factor ends), cold-high and input +0.97 %, rated 150 -> 80 °C. This is "unless calibration changes".
- Y30: B842SH x3.603220 (+236.29 %, rated 250 -> 150 °C), B842 x3.603220 (+236.29 %, 250 -> 80 °C) and B842-N52 x4.018976 (+275.10 %, 250 -> 80 °C).

The calibration can also start. At the 304 design, a custom inner ring at B842SH's size in N35, changed to B842SH, gives pull-out x1.227427 against Br x1.111111 (f_cal 0.95 -> 1.0495) and cold-high +22.74 %. In N42 it gives x1.104684 against Br x1.000000 and cold-high +10.47 %. "Unless calibration changes" covers both directions, and the loads still rise.

So with "same-size":
- "raises pull-out proportionally to its ... Br unless calibration changes" holds at all five designs;
- "shaft-joint and gearbox loads rise" holds in every case (+0.97 % to +275.10 %);
- the rating falls in every library case.

"Often" stays, because a same-size stronger pick can carry a higher rating: a custom N35 block rated 80 °C, changed to B842SH, is rated 150 °C. "Its" refers to the block. Br keeps its gloss ("built-in flux density") at its first use in the note.

**Dependent sentences.**
- sentences[1] (T_pull = T_2D·f_end·f_cal, "end and calibration factors") is what "unless calibration changes" refers to. It is consistent.
- sentences[3] is unchanged; S-1 is rejected above.
- Nothing else in the note depends on the two changed entries.

**Tests.** `pullout_angle_claims_hold_on_the_engine` now pins three things, all computed from the engine, and it passes:
- the B842-N52 proportion, unchanged f_cal and higher cold-high at the default design;
- the lower rating;
- the 304 case, f_cal moving from `calibration.f_cal_updated` to `f_cal_original` and the pull-out ratio equal to Br x f_cal.

It does not pin the size qualifier (should-fix 1).

**Check g.**
- Word counts: change[0] 16, change[1] 25, must-fix replacement 25.
- All are plain-words clean, with no ids, no "clamp" and no typed input label.

**Verdict:** MUST-FIX (1).

Round 5:

**What I read**
- `task-12-inreview-r4-changes.md` and the answers of rounds 1 to 4 (`task-12-inreview-r1.md` to `-r4.md`).
- The brief's Step 8 prompt and Step 9.
- `amendments.md`: the Global Constraints, the index row for T12, rulings 9, 11 and 12, A-13, A-19 to A-25, A-27 and A-28 (its rules for every task, A-28j and A-28k).
- The regenerated dump `magcoupling-rs/target/notes-review-T12.txt`.
- The pullout_angle literal (`notes.rs:307-356`) and the test (`tests/explain.rs:1388-1493`).
- For the claims:
  - `model.rs`: `resolve_magnets` (:445-503), `select_calibration_factor` (:584-599), the Br at the running temperature (:977-979) and `blocks_fit` (:579);
  - `calibration.rs`: the bench factor `f_cal_updated = f_cal_original x measured / model` (:190-235), whose model sums the design's harmonic set;
  - `library.rs`: all 15 parts;
  - `grades.rs`: the grade table and its coefficients;
  - `engine/sizing.rs` and `gui/sizing.rs`: the header comments on Auto.
- The 2026-10-14 spec section 8, the slice-1 spec section 8 and phys-new-notes section 2.

**How I reproduced the numbers.** I built a new scratch crate outside the repo (scratchpad `t12-inreview-r5-own-probe`), with magcoupling-rs by path and default-features = false. It calls:
- `compute_all`, `probes::probe_designs`, `model::blocks_fit` and `model::MAX_HARMONIC_CHOICES`;
- `library::MAGNET_LIBRARY`, `grades::GRADES` and `notes::NOTES`;
- `plain::{word_count, plain_text_violations, audit_ids}`.

"Same size" means `model.inner_width_mm`, `model.inner_thickness_mm` and `model.inner_length_mm` are exactly equal after the pick. "Stronger grade" means a higher 20 °C Br (`model.inner_br_T`).

**The probe designs' inner ring.** B842SH (6.35 x 3.17 x 12.7 mm, Br 1.30 T at 20 °C, 1.2532 T at 50 °C, rated 150 °C) at the default, 8-magnet, 304 and 20 mm designs. At the Y30 design it is a custom ring of the same size in Y30 (0.37 T, 0.3478 T at 50 °C, α −0.002/°C, rated 250 °C). f_cal is 0.95 everywhere except the 304 design, which has the bench factor 1.049450.

**change[1], new text** ("a same-size, stronger-grade block raises pull-out proportionally to its built-in flux density Br unless calibration changes; shaft-joint and gearbox loads rise; heat tolerance often drops.", 25 words).

These are the same-size, stronger-grade library picks on the inner ring. Columns: pull-out ratio; Br ratio at 50 °C; f_cal before and after; cold-high change, which equals the gearbox-input change; rating:

| Design | Pick | Pull-out | Br (50 °C) | f_cal | Cold-high = input | Rated (°C) |
|---|---|---|---|---|---|---|
| default | B842-N52 | x1.115385 | x1.115385 | 0.95 kept | +11.54 % | 150 -> 80 |
| 8 magnets | B842-N52 | x1.115385 | x1.115385 | 0.95 kept | +11.54 % | 150 -> 80 |
| 304 | B842-N52 | x1.009686 | x1.115385 | 1.0495 -> 0.95 | +0.97 % | 150 -> 80 |
| Y30 | B842SH | x3.603220 | x3.603220 | 0.95 kept | +236.29 % | 250 -> 150 |
| Y30 | B842 | x3.603220 | x3.603220 | 0.95 kept | +236.29 % | 250 -> 80 |
| Y30 | B842-N52 | x4.018976 | x4.018976 | 0.95 kept | +275.10 % | 250 -> 80 |
| 20 mm | B842-N52 | x1.115385 | x1.115385 | 0.95 kept | +11.54 % | 150 -> 80 |

- In every row the pull-out ratio divided by (Br ratio at the running temperature x f_cal ratio) is 1.000000000000. So "raises pull-out proportionally to its ... Br unless calibration changes" holds at all five designs. The 304 row is the "unless" case.
- "Shaft-joint and gearbox loads rise" holds in every row (+0.97 % to +275.10 %).
- "Heat tolerance often drops": the rating drops in every row.
- The parts "same-size" now excludes are the ones round 4 found false. M5044, M5045 and M5026 (arcs, 1.12-1.67 mm thick) fit but cut the pull-out and the loads by 21.85 % to 80.30 % at four designs, and M5045 by 12.52 % at Y30. B882-N52 fits at the 20 mm design and gives x2.213568 against Br x1.115385. None of these is the same size as the inner block.

**Generality beyond the probe designs' own block.** I tried every library part as the base on the inner ring wherever it fits, and every same-size library part with a higher Br. That gives 11 pairs over the five designs: B842SH -> B842-N52, B842 -> B842-N52 and, at 20 mm, B882 -> B882-N52.
- All 11 keep pull-out = Br ratio x f_cal ratio to 1e-12, and all raise the cold-high torque and the gearbox input.
- The rating drops (150 -> 80) or stays (80 -> 80). Hence "often".

I also tried a custom ring at the inner block's size changed to a library part (N35, N42 or Y30 to B842SH, B842 or B842-N52). All 35 cases hold.
- At the 304 design, custom N35 -> B842SH starts the bench factor: x1.227427 against Br x1.111111 (f_cal 0.95 -> 1.0495), cold-high +22.74 %. The rating rises 80 -> 150 °C, which is why the entry says "often".

**The bench design across settings.** B842SH -> B842-N52 on the 304 design at each harmonic setting:

| Harmonics summed | Bench factor | Pull-out = cold-high = input |
|---|---|---|
| 1 | 1.055530 | +0.39 % |
| 1, 3 | 1.041240 | +1.77 % |
| 1, 3, 5 (default) | 1.049450 | +0.97 % |
| up to 7 | 1.048993 | +1.01 % |
| up to 9 | 1.049012 | +1.01 % |
| up to 11 | 1.048955 | +1.02 % |

- The break-even bench factor is 0.95 x 1.45 / 1.30 = 1.059615, so the loads rise at every setting.
- The operating temperature (−20, 20, 50, 80 °C) does not change the result (+0.97 % each), because both library parts share one Br coefficient.
- The margin is small, hence SHOULD-FIX 1.

**Edges checked, not findings.**
- **Br at the running temperature.** At the Y30 design the pull-out follows Br at the running temperature: x3.603220 for B842SH, against x3.513514 for the 20 °C datasheet values. The two differ by 2.55 %, and by up to 5.27 % for a SmCo grade, because a ferrite or SmCo ring uses its grade's own coefficient. Within NdFeB (every NdFeB grade and every library part at −0.0012/°C) the two ratios are equal. Sentence[1] says the pull-out is computed at the running magnet temperature, and the dashboard shows Br at that temperature (`model.br_inner_T_op`). Physically, torque at a temperature follows Br at that temperature. So "proportionally to its ... Br" is true as written.
- **The custom-grade route on the bench design.** Choosing manual dimensions in this picker and then a stronger NdFeB grade in the separate grade input, at B842SH's size on the 304 design, ends the bench factor:
  - N48: pull-out, cold-high and input −3.91 %;
  - N50 and N50M: −1.82 % each;
  - N52: +0.97 %.

  Elsewhere every such grade raises all three, and the pull-out stays exactly proportional to Br x f_cal.

  I do not count this against the entry, for three reasons:
  - it needs a second input (the grade under custom dimensions), which this entry does not offer;
  - the entry names the cause ("unless calibration changes");
  - the drop is the model switching from the measured bench factor to the assumed one, while the stronger magnet itself still pulls harder.

  The step on leaving the bench match belongs to the calibration note (T13, A-13's calibration source "the step on leaving a match"). I record it here for T13's reviewer.
- **Auto (Torque -> Magnets).** Auto re-solves its free variable after any change (`gui/sizing.rs` header). With Auto on, a stronger grade shortens the magnets to hold the hot-low target, rather than raising the pull-out. This applies equally to every torque-raising entry in every note and was there in all earlier rounds. It is outside this round's scope and is not a finding on this field.

**Plain words and form.**
- 25 words by `plain::word_count`, at the cap.
- `plain_text_violations` and `audit_ids` are empty, and there is no "clamp".
- No input label is typed as text.
- Br is glossed ("built-in flux density") at its first use in the note.
- The drawn line reads "Choose Inner magnet part: a same-size, stronger-grade block raises ...". "Its" refers to the block.

**Dependent sentences.**
- sentences[1] ("T_pull = T_2D·f_end·f_cal ... end and calibration factors") is what "unless calibration changes" refers to. It is consistent.
- change[0] and change[2] do not depend on change[1].
- harmonics change[0] now has a matching size qualifier, as noted above.

**The test.** `pullout_angle_claims_hold_on_the_engine` passes. I ran it with `--locked --test explain` in a scratch target, together with `harmonics_outer_part_entry_holds_on_the_example_design`: 2 passed. `--lib engine::explain::notes` gave 16 passed and 3 ignored, including `the_rewritten_notes_keep_to_the_slice_2_caps` and `the_rewritten_notes_read_in_plain_words`.
- What it asserts, every value from `compute_all`:
  - the example design's inner part is "B842SH";
  - B842-N52 has B842SH's width, thickness and length (exact equality);
  - B842-N52 keeps f_cal, its pull-out ratio equals the Br ratio to 1e-12, the cold-high torque rises and the rating drops;
  - M5026 fits by `blocks_fit`, has a higher `inner_br_T` (1.42 against 1.30), is thinner (1.67 against 3.17 mm), and lowers the pull-out (2.688476 -> 2.012532 N·m) and the cold-high torque (3.823310 -> 2.862042 N·m);
  - at the 304 design, B842-N52 moves f_cal from `f_cal_updated` to `f_cal_original`, and the pull-out ratio equals Br x f_cal.
- The new asserts bite. With B882-N52 as the same-size pick, the width assert fails (12.7 against 6.35 mm). With B842-N52 in place of M5026, the "thinner" assert fails (3.17 against 3.17 mm). Both values come from my probe, and they agree with the implementer's two mutation runs.
- The `with_inner` closure removes the duplicated clone-set-compute path; this is DRY and explicit.
- The doc comment says "a same-size, stronger-grade block" and says why "same-size" is needed. It is not drawn.
- What is missing is only the loads' sign on the bench design and the gearbox-input row (SHOULD-FIX 1).

**Verdict:** PASS.

Must-fix items applied:
- change[1].text (round 1, must-fix 1: false at the 304 design, where the pick ends the bench calibration): "a stronger grade raises pull-out in proportion to its built-in flux density Br, the outer part likewise; stronger grades often tolerate less heat." -> "a stronger grade raises pull-out in proportion to its built-in flux density Br, unless the bench-measured calibration then stops applying; it often tolerates less heat."
- change[1].text (round 4, must-fix 1, bound by controller ruling 12: stronger-grade parts of another size, M5044, M5045, M5026 and, at the 20 mm design, B882-N52, break both claims): "a stronger grade raises pull-out in proportion to its built-in flux density Br unless calibration changes; shaft-joint and gearbox loads rise; heat tolerance often drops." -> "a same-size, stronger-grade block raises pull-out proportionally to its built-in flux density Br unless calibration changes; shaft-joint and gearbox loads rise; heat tolerance often drops."

Changes by controller ruling (the rulings' text, verbatim):
- change[0].text (ruling 11, the shaft-clamp section paused: joint-neutral wording): "pull-out grows almost linearly with the overlap; the stack lengthens, and clamp and gearbox loads rise." -> "pull-out grows almost linearly with the overlap; the stack lengthens, and shaft-joint and gearbox loads rise."
- change[1].text (ruling 9: round-3 should-fix 1, the entry named no load trade-off; round-2 should-fix 2, "unless calibration changes" covers the bench factor starting as well as stopping; joint-neutral): "a stronger grade raises pull-out in proportion to its built-in flux density Br, unless the bench-measured calibration then stops applying; it often tolerates less heat." -> "a stronger grade raises pull-out in proportion to its built-in flux density Br unless calibration changes; shaft-joint and gearbox loads rise; heat tolerance often drops." Round 4 then replaced it (must-fix above).
- `tests/explain.rs` `pullout_angle_claims_hold_on_the_engine`, ruling 9 (should-fix 1 of rounds 1 and 2): at the "304 back iron" design, B842-N52 on the inner ring moves `model.f_cal` from the bench factor to the default one, and the pull-out ratio equals the inner Br ratio times the f_cal ratio; at the example design `model.f_cal` is unchanged and `metal.torque_cold_high_Nm` rises.
- The same test, ruling 12 (round-4 should-fix 1): at the example design, B842-N52 has B842SH's `model.inner_width_mm`, `model.inner_thickness_mm` and `model.inner_length_mm`; M5026 on the inner ring fits by `model::blocks_fit`, has a higher `model.inner_br_T` and a smaller `model.inner_thickness_mm`, and lowers `model.pullout_Nm` and `metal.torque_cold_high_Nm`.

Seeds (A-24):
- S-1 (sentences[3]: "The fundamental alone peaks at 90°; at low fill a same-sign third harmonic above about a ninth of it moves the peak off 90°."): rejected in every round, 1 to 5. The engine's pull-out is the first peak, the one a rising load reaches (`peak_off_half_pitch` returns x in [0, π/2]): 52.96° electrical at the 20 mm design and 90° at the other four. "Below 90°" says the rings slip at a smaller twist; "off 90°" drops that direction. The mirror peak at 180° − φ lies past slip and is never reached under a rising load. sentences[3] keeps the drafted text.

A-27 (prose duplicates): no `when_slipping` note existed during T12, so nothing was trimmed. Recorded for T15 (round 1): `when_slipping` must not repeat sentences[4] or the watch-out verbatim.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-5 PASS):
- Round 1 S-2: a fourth source line for the past-pull-out finding behind sentences[4] and the watch-out, "Guidance audit of 2026-10-13: what happens past pull-out; the rings re-lock only when the slip speed falls". A-13 fixes the three lines verbatim, so adding one needs a controller ruling.
- Round 5 S-1: pin the sign of the loads where the entry's margin is smallest. At the "304 back iron" design, assert that B842-N52 raises `metal.torque_cold_high_Nm` and `metal.cold_high_input_Nm` (+0.97 %: the Br gain x1.115385 beats the bench-factor drop x0.905236; the break-even bench factor is 1.059615); at the example design, add the `metal.cold_high_input_Nm` assert beside the cold-high one and add that row to the M5026 check.

Diagrams (torque angle, flux paths, end fringing):

Verdict by round: 1 PASS; 2 and 3 PASS (no diagram changed, not re-reviewed); 4 PASS (the
square-wave label); 5 PASS (no diagram changed).

Round 1:

**What I read**
- `diagrams.rs`, the T12 diff: the module doc, `FUNDAMENTAL_ALONE`, `STRONG_THIRD`, `END_LOSS_LABEL` and `END_LOSS`, `flux_panel`, `torque_angle` and `end_fringing`, and the three new tests plus `no_two_labels_of_a_diagram_overlap`.
- The rendered picture `magcoupling-rs/target/t12-diagrams.png`.

**Torque angle.**
- The dashed fundamental is the ink colour, its label "fundamental alone" above its top. It peaks at its middle sample, t = 0.5 (90°), where the 90° tick and its label sit.
- The solid sum is sin φ + 0.15·sin 3φ, a same-sign third above a ninth (−1 + 9·0.15 = +0.35 > 0). It dips to 0.85 at 90° and peaks at about 63° and 117°, at about 0.87 of the fundamental's peak.
- The engine agrees: at the 20 mm design the third harmonic lowers the pull-out by 12.4 % and moves it to 53°.
- The marker and "pull-out" sit on the first peak, left of 90°. That matches pullout_angle sentences[3] ("below 90°") and `peak_off_half_pitch`, which returns x in [0, π/2].
- The sum is labelled "strong third harmonic" in its own colour, so the unusual case is named.

**Flux paths.**
- Every block carries a white N (on red) or S (on blue), centred.
- Across the gap an N faces an S at zero twist, the attracting alignment.
- With steel, the loops close from one steel bar to the other. In free space they bulge far behind the blocks.
- This matches back_iron_factor sentence 0 (with must-fix 1 applied, the picture's "with back iron" is the magnetic case).

**End fringing.**
- Two level brackets, each END_LOSS = 0.08 of the block, sit under its ends in the label's colour: "magnet length L, c_end·τ_p/2 lost per end".
- f_end = 1 − c_end·τ_p/L (model.rs:1003) takes c_end·τ_p in total, half at each end. That matches the current end_effect text ("half at each end") and A-8's rationale.
- The share is schematic: at the default design 1 − f_end = 0.104, so 0.052 per end. The code comment says so.

**Contradictions.** No picture teaches anything the notes or the engine contradict.

**Verdict:** PASS.

Round 2:

No diagram changed since round 1: `diagrams.rs` was last written at 08:22, before the round-1 answer at 08:41. The round-1 DIAGRAMS PASS stands, and I did not re-review the torque-angle, flux or end-fringing pictures. The one dependent point is the square-wave label wording above (should-fix 1). It is a wording consistency, not a physics error: the picture's low level is the room between blocks.

**Verdict:** PASS (not re-reviewed; round-1 PASS stands).

Round 3:

No diagram changed since round 1: `diagrams.rs` was last written at 08:22, before the round-1 answer (08:41). Must-fix 1 changes only note text and draws no picture. The round-1 DIAGRAMS PASS stands, and I did not re-review the torque-angle, flux or end-fringing pictures.

**Verdict:** PASS (not re-reviewed; round-1 PASS stands).

Round 4:

**The change.** The square-wave label is now "blocks and spaces (fill 0.85)" (`diagrams.rs:179`). The test's `wave_label` (:792) expects it, and two doc comments (:135 and `notes.rs:23`) match. The picture is unchanged.
- The ink wave is +1 over north blocks, −1 over south blocks and 0 between them. Those zero stretches are the spaces, so the label is true.
- It now uses the same word as harmonics sentences[0] and change[2], so "gap" is left to the gap between the rings.

**Verification.** `cargo test --features app --lib gui::diagrams` (scratch target directory, `--locked`): 10 passed. This includes:
- `no_two_labels_of_a_diagram_overlap`: the label grew by two characters beside "n = 1" and "1 + 3 + 5";
- `the_square_wave_paints_the_sum_of_its_harmonics_at_its_fill`;
- `every_diagram_draws_inside_its_rect_with_finite_points_and_labels`.

**Not drawn.** The code comments at `diagrams.rs:139` ("0 in the gaps") and in the test at :797 ("its gaps are 0") still say "gaps". They are not drawn, so this is no finding.

**Earlier rounds.** The torque-angle, flux-path and end-fringing pictures did not change, and the round-1 PASS stands for them.

**Verdict:** PASS.

Round 5:

`diagrams.rs` was not changed since round 4 (file time 09:13; md5 unchanged across this review). No picture depends on the changed entry. The round-4 PASS stands for the square wave, and the round-1 PASS for the torque-angle, flux-path and end-fringing pictures.

**Verdict:** PASS.

Must-fix items applied: none.

Should-fix items applied by controller ruling 9:
- `square_wave` label (round-2 DIAGRAMS should-fix 1, so that "gap" means only the gap between the rings): "blocks and gaps (fill {FILL})" -> "blocks and spaces (fill {FILL})", drawn as "blocks and spaces (fill 0.85)"; the test's `wave_label` and the doc comments of `square_wave` and `Diagram::SquareWaveHarmonics` match.
