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

## end_effect

Task T13. Rewrite of "End effect" as "Magnet ends: the length lost to fringing", the note behind
the end factor (`model.f_end`, `calibration.f_end`) and the end-effect check's Why link. It keeps
the end-fringing diagram, whose brackets were reviewed with T12's diagrams (`## pullout_angle`).
Reviewer: physics reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice
plan fixes for the note's `review` field; all three rounds ran on 2026-10-15).

Rounds: 3. Verdict: PASS. Note: "Magnet ends: the length lost to fringing" (plan T13).

Verdict by round: 1 MUST-FIX (4 items); 2 MUST-FIX (1); 3 PASS. Each round's full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-13-inreview-r<N>.md`, and the
fields that rounds 1 and 2 changed are listed with their old and new text in
`task-13-inreview-r<N>-changes.md` beside it. Rounds 2 and 3 re-checked only the changed fields
and the fields whose meaning depends on them. Each round wrote its own free-space 3D check with no
code shared (round 1: magpylib's cuboid field and surface charges; round 2: its own closed-form
charged-rectangle field; round 3: equivalent surface currents). All three reproduce the math
audit's 1.777459 N·m and agree with one another to the digits shown.

The text reviewed in round 1 already carried A-2, A-13 and A-28a: the joint-neutral change[0] and
change[1] texts ("shaft-joint and gearbox loads" in place of the clamp load) and A-13's plain-words
sources. Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump `notes-review-T13.txt`; `notes.rs` end_effect literal;
  - `model.rs:94-114` (`END_EFFECT_OUT_OF_RANGE`, `end_effect_in_range` f_end > 0) and `:913-1004` (L = min of the ring lengths, R_g = r_face + g/2, τ_p = 2πR_g/N, T_2D = τ·2πR_g²·L, f_end = 1 − c_end·τ_p/L, pull-out = T_2D·f_end·f_cal);
  - `calibration.rs:155-235` (the prototype's own c_end and f_end); `metal_design.rs:455, 509` (cold-high and its gearbox-input equivalent);
  - `assumptions.rs:68-73`; `guidance.rs:270-279` (the end-effect guide offers only magnet length);
  - `diagrams.rs:349-398` (the brackets: c_end·τ_p/2 at each end, which matches "half at each end");
  - math audit rows M8, M9 and the results JSON (T3-RC1 pole and gap sweeps, T3-RC3 short-magnet table, all three lenses).
- Numbers reproduced from `compute_all`, in design order: default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius.
  - f_end: 0.8960, 0.8699, 0.8960, 0.8960, 0.8229.
  - τ_p: 8.809, 11.01, 8.809, 8.809, 15.00 mm.
  - L: 12.70 mm at all five.
  - Every rendered value matches the dump. Placeholders sit only in sentence 2's "here" clause, and no comparison depends on them.
- Sentence 3 holds exactly: pull-out at L, L+1 and L+2 mm steps by 0.236274, 0.178159, 0.149171, 0.018198 and 0.242984 N·m per mm. The second differences are at most 9e-16, and each step equals T_pull/(L − c_end·τ_p).
- Out-of-range edge (watch-out 2): 0.999·c_end·τ_p reads "End-effect model out of range" and 1.001·c_end·τ_p reads "OK" at all five. c_end·τ_p is 1.3214, 1.6517, 1.3214, 1.3214 and 2.2497 mm.
  - At 2.0 mm, f_end is 0.3393, 0.1742, 0.3393, 0.3393 and −0.1248. The 20 mm design is out of range there.
  - At 1.0 mm f_end is negative at all five, with a negative pull-out, flagged.
- Gap probe (model): f_end at 0.6, 1.4, 2.0, 3.0 and 4.4 mm is 0.8989, 0.8960, 0.8937, 0.8900 and 0.8848 on the example ring. The end loss length runs 1.284 to 1.463 mm. The gap enters only through τ_p at R_g (14.02 to 15.52 mm).
- Levers, one slider step (`step_input`) at all five designs:
  - Axial length +0.01 mm:
    - f_end rises (default 0.895956 to 0.896038);
    - pull-out, hot-low, cold-high and `metal.cold_high_input_Nm` rise (default 3.823310 to 3.826670 and 0.804907 to 0.805615 N·m);
    - `calibration.f_end` stays 0.898183.
  - c_end +0.005: f_end and every predicted torque fall by the same ratio (0.996129 at the default: the 20 predicted torque rows of the model, metal and temperature results, and the two hidden clamp rows), cold-high and its gearbox-input value included; `calibration.f_end` stays 0.898183.
  - Magnets per ring +2:
    - f_end rises in the model at all five (0.8960 to 0.9133 at the default);
    - 12 blocks do not fit the example ring (or the 304 and Y30 designs); they fit at 8 to 10 and at the 20 mm ring.
- Independent free-space 3D check, written for this review (magpylib cuboid field; surface-charge torque on the outer ring; half a pole pitch; Br 1.29 T):
  - It reproduces the audit's 1.777459 N·m at 10 magnets, 12.7 mm and 1.4 mm.
  - By length (example ring), model/3D: 2.0 mm −20.7 %; 2.94 mm (τ_p/3) −0.6 %; 3.17 +0.8 %; 3.6 +2.15 %; 4.0 +2.61 %; 4.5 +2.69 %; 5.0 +2.53 %; 5.5 +2.28 %; 6.35 +1.83 %; 8.0 +1.15 %; 12.7 +0.37 %; 25.4 +0.10 %.
  - By gap (12.7 mm): 0.6 mm −3.2 %; 1.4 +0.4 %; 2.0 +2.7 %; 3.0 +6.4 %; 4.4 +11.2 % (0.8848 against 0.7955).
  - By count, example ring: 6 magnets −11.75 %, 8 −3.42 %, 10 +0.37 % (3D f_end 0.937, 0.901, 0.893).
  - By count, 20 mm ring: 10 magnets −12.25 %, 12 −5.65 %, 14 −2.40 % (3D f_end 0.938, 0.903, 0.895).
- Traps checked:
  - "the end factor is accurate": not written; MF-2 scopes the 3D agreement to the example design, and watch-out 1 keeps wide-gap optimism.
  - "more poles help until end effects bite": not written; MF-1 removes "more magnets lose less at the ends", which is the model's direction but not 3D's at fixed blocks.
  - "more magnets make the coupling stiffer": not written.
  - Any torque called guaranteed: none.
  - "a shaft-joint method is chosen or sized here, or the coupling is said to protect the gearbox": not written.
  - "clamp" in any field, sources included: none.
- Form:
  - Sentences 24, 22, 18, 23 and 25 words (MF-2 25, MF-3 25).
  - Change texts 23 and 24 words after MF-1, each starting in lower case with no label typed.
  - Watch-outs 28 and 28 words.
  - Sentence 0 has no symbol and no placeholder. The title is plain and unique.
- Seeds:
  - S-3 accepted, amended (MF-3).
  - S-4 rejected as written: its point is in MF-2, and it kept the 2 % figure and a scope that the 20 mm ring meets while reading 12 % low.
- Must-fix items and how they were applied (round 1): MF-1 (change[1] deleted, tests and commit text follow), MF-2 (sentences[4]), MF-3 (sentences[1]), MF-4 (sources[1], BL-056 and commit text follow). Pending the round-2 re-check.
- Verdict, round 1: MUST_FIX.

Round 2:

- Round 2 (physics reviewer, session model, fresh context, 2026-10-15) re-checked the round-1 changes:
  - change[1] (magnets per ring) deleted;
  - sentences[4] (3D agreement);
  - sentences[1] (c_end glossed as an empirical coefficient);
  - sources[1] (this review's 3D sweeps).
- Independent free-space 3D check, written for round 2 with no code shared with round 1 or the audit:
  - Method: own closed-form field of a uniformly charged rectangle; torque on the outer ring from the surface charges on its radial faces; composite Gauss-Legendre (0.8 mm and 0.4 mm panels agree to 7 digits); half a pole pitch.
  - Validation: it reproduces the audit's 1.777459 N·m (10 magnets, 12.7 mm, 1.4 mm gap, Br 1.29 T) and its 3D end factors 0.5785, 0.7777, 0.8926 and 0.9470 at 3.17, 6.35, 12.7 and 25.4 mm. The long-length slope is 156.792 N·m/m.
- By length, example ring, model/3D: 2.0 mm −20.72 %; 2.94 −0.58 %; 3.17 +0.80 %; 3.2 +0.94 %; 3.4 +1.67 %; 3.6 +2.15 %; 3.8 +2.45 %; 4.0 +2.61 %; 4.25 +2.70 %; 4.5 +2.69 %; 4.75 +2.63 %; 5.0 +2.53 %; 5.5 +2.28 %; 6.35 +1.83 %; 8.0 +1.15 %; 12.7 +0.37 %; 25.4 +0.10 %; 50.8 +0.05 %.
  - So "within 3 % at 3.2 mm or longer" holds on the example design; the worst case is +2.70 % near 4.25 mm.
- By count, example ring, 12.7 mm: 6 magnets −11.75 %, 8 −3.42 %, 10 +0.37 %, 12 +2.64 %, 14 +3.92 % (12 and 14 do not fit with B842SH).
  - The 3D end deficit is 0.80, 1.26, 1.36, 1.40 and 1.39 mm, against the model's 2.20, 1.65, 1.32, 1.10 and 0.94 mm. More magnets do not cut the real end loss; this confirms the deletion of change[1].
- By ring size, 10 magnets, 12.7 mm: 12 mm apothem −1.44 %, 14 mm −3.52 %, 20 mm −12.25 %.
  - The 20 mm ring at 12, 14 and 16 magnets: −5.65 %, −2.40 %, −0.10 %.
  - The 20 mm ring at 3.2, 4.5 and 6.35 mm: −49.7 %, −30.0 %, −21.2 %.
  - 8 magnets on the example ring at 3.2, 4.5 and 6.35 mm: −17.61 %, −8.90 %, −5.85 %.
- By gap, 12.7 mm: 0.6 mm −3.18 %; 4.4 mm +11.23 % (0.8848 against 0.7955), as watch-out 0 says.
- Engine values reproduced with `compute_all` at the five designs (default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius):
  - f_end 0.895956, 0.869945, 0.895956, 0.895956, 0.822859;
  - τ_p 8.809, 11.01, 8.809, 8.809, 15.00 mm;
  - L 12.70 mm at all five. Each matches the dump.
- Levers, one `step_input` step at all five designs:
  - axial length up raises f_end, cold-high and cold-high at the gearbox input;
  - c_end up lowers f_end, hot-low, cold-high and the gearbox input;
  - `calibration.f_end` stays 0.898183.
- Form, after round 2's fix:
  - sentences 24, 25, 18, 23 and 25 words;
  - change texts 23 and 24 words;
  - watch-outs 28 and 28 words;
  - sources 18 and 30 words;
  - every field plain, with no audit id and no "clamp".
- Must-fix, round 2: R2-MF-1, sentences[4]. "low with fewer magnets or larger rings" could be read as 3D confirming the formula's lower f_end on sparser rings, which is false in 3D. It becomes "below 3D with fewer magnets or larger rings", with "at 3.2 mm or longer" to stay within 25 words. Pending the round-3 re-check.
- Seeds: S-3 accepted as amended (in place); S-4 rejected (2 % exceeded from about 3.5 to 6.0 mm; its scope admits the 20 mm ring at −12 %).
- Verdict, round 2: MUST_FIX (one item).

Round 3:

- Round 3 (physics reviewer, session model, fresh context, 2026-10-15) re-checked sentences[4] after R2-MF-1: "Steel-free 3D checks of the example design put f_end within 3 % at 3.2 mm or longer, below 3D with fewer magnets or larger rings." It also re-checked the fields that depend on it: sentences[1] to [3], both watch-outs and sources[1].
- **Independent free-space 3D check, third method.**
  - The outer ring's force comes from its equivalent surface currents on the four side faces of each block, in the closed-form field of the inner ring's charged pole faces. It shares no force formulation or code with rounds 1 and 2 or the audit.
  - Validation: 1.7774591 N·m against the audit's 1.777459 (10 magnets, 12.7 mm, 1.4 mm gap, Br 1.29 T), converged at 0.8, 0.4 and 0.2 mm panels.
  - Slope 156.790 N·m/m. Half a pole pitch is the 3D peak.
- **Example design by length** (model/3D): 3.2 mm +0.94 %; 3.4 +1.67; 3.6 +2.15; 3.8 +2.44; 4.0 +2.61; 4.25 +2.70; 4.5 +2.69; 4.75 +2.63; 5.0 +2.53; 5.5 +2.28; 6.35 +1.83; 8.0 +1.15; 12.7 +0.37; 25.4 +0.10. "Within 3 % at 3.2 mm or longer" holds, with the worst case +2.70 % at 4.25 mm.
- **Below 3D with fewer magnets or larger rings** (3.2, 4.25, 6.35, 12.7 and 25.4 mm):
  - 8 magnets −17.61, −9.79, −5.85, −3.42, −1.80 %;
  - 6 magnets −46.60, −29.86, −19.61, −11.81, −6.58 %;
  - 12 mm ring −8.78 to −0.78 %; 14 mm −19.17 to −1.81 %; 20 mm −49.74, −32.28, −21.23, −12.27, −6.51 %;
  - B822 at 8 magnets and at a 20 mm ring: also below.
  - Inside the 3 % band only: a 10.6 mm ring +1.14 % at 4.25 mm; the 20 mm ring at 16 magnets −3.27 to +0.42 %.
- **Gap:** 4.4 mm gives +11.23 % (0.8848 against 0.7955), as watch-out 0 says.
- **Engine values at the five designs:**
  - f_end 0.895956, 0.869945, 0.895956, 0.895956, 0.822859;
  - τ_p 8.8090, 11.0113, 8.8090, 8.8090, 14.9980 mm;
  - L 12.700 mm at all five. Each matches the dump.
- **Levers** (`step_input`, all five designs):
  - the axial step raises f_end, the pull-out, cold-high and its gearbox-input value;
  - the c_end step lowers f_end, hot-low, cold-high and the gearbox input;
  - `calibration.f_end` stays 0.898183.
- **Form:** sentences 24, 25, 18, 23 and 25 words; changes 23 and 24; watch-outs 28 and 28; sources 18 and 30. Every field is plain, with no audit id and no "clamp".
- **Seeds:**
  - S-3 accepted as amended (in place since round 1).
  - S-4 rejected: 2 % is exceeded from about 3.5 to 6.0 mm, and its scope admits the 20 mm ring (−12.27 %) and B822 blocks (+8.82 % at 4.25 mm).
- **Open for the controller, outside round-3 scope (SF-C):**
  - Narrower or more tightly packed blocks read high: B822 on the example ring at 10, 12, 16 and 20 per ring (all fit) gives +1.87, +3.29, +4.62 and +5.23 % at 12.7 mm, and up to +27 % at 3.2 mm. B842SH at 18 per ring on the 20 mm ring gives +5.99 % at 4.25 mm.
  - watch_out[1]'s unscoped "reads well low" is false for B822 below τ_p/3: +3.80 % at 2.5 mm with 10, +9.29 % at 2.0 mm with 12.
  - Candidate texts W1 and W0 are in the round-3 review file.
- **Verdict, round 3:** PASS. R2-MF-1 was applied verbatim and is true; no must-fix remains.

Must-fix items applied:
- change[1] (round 1, MF-1: at fixed blocks, more magnets per ring do not cut the real end loss in free-space 3D, so the entry offered a model artefact as a design move), deleted: input `coupling.npole`, Raise, aim `model.f_end` Up, "a shorter pole pitch loses less at the ends; more blocks must then fit around the same ring, so check the fit." The note keeps two entries. `the_end_effect_note_s_levers_move_the_end_factor_as_it_says` follows: two levers, aims Up and Down, and the magnet-count blocks deleted.
- sentences[4] (round 1, MF-2: "within 2 %" is exceeded from about 3.5 to 6.0 mm; the scope "10 poles and a 1.4 mm gap" admits the 20 mm ring at −12 %; "poles" is not the on-screen name; "it" can read as the torque): "A 3D field check without steel, at 10 poles and a 1.4 mm gap, put it within 2 % for magnets 3.2 mm and longer." -> "Steel-free 3D checks of the example design put f_end within 3 % for magnets 3.2 mm or longer, low with fewer magnets or larger rings."
- sentences[1] (round 1, MF-3, seed S-3 accepted with the reviewer's amendment: c_end was unglossed at its first use): "The model counts this as missing length: f_end = 1 − c_end·τ_p/L, with τ_p the pole pitch and L the magnet overlap." -> "The model counts this as missing length: f_end = 1 − c_end·τ_p/L, with c_end an empirical coefficient, τ_p the pole pitch, L the magnet overlap."
- sources[1] (round 1, MF-4: the line repeated, under the fold, the 2 % figure MF-2 corrects): "Guidance audit of 2026-10-13: within 2 % of free-space 3D for magnets 3.2 mm and longer, not 1 %" -> "Physics review of 2026-10-15: the end factor against free-space 3D by magnet length (within 3 % from 3.2 mm), by magnets per ring and at a 20 mm ring radius"
- sentences[4] (round 2, R2-MF-1: the bare "low" could be read as 3D confirming the formula's lower end factor on sparser rings, which is false in 3D): "Steel-free 3D checks of the example design put f_end within 3 % for magnets 3.2 mm or longer, low with fewer magnets or larger rings." -> "Steel-free 3D checks of the example design put f_end within 3 % at 3.2 mm or longer, below 3D with fewer magnets or larger rings."

Seeds (A-24):
- S-3 (sentences[1]: "The model counts this as missing length: f_end = 1 − c_end·τ_p/L, with c_end an end coefficient, τ_p the pole pitch, L the overlap."): accepted in round 1 with one change and applied as MF-3; rounds 2 and 3 confirm it in place. Reason: c_end was unglossed at its first use. "An empirical coefficient" says what kind of number it is (fitted to 3D, not derived), which change[1]'s "only with 3D or bench evidence" relies on; the seed's "an end coefficient" says less, and its "L the overlap" drops what overlaps.
- S-4 (sentences[4]: "A free-space 3D check at 10 magnets per ring and a 1.4 mm gap put it within 2 % for magnets 3.2 mm or longer."): rejected in rounds 1 to 3. Its "within 2 %" is false on the example design from about 3.5 to 6.0 mm (+2.70 % at 4.25 mm, the worst). Its scope (10 magnets per ring, a 1.4 mm gap) admits the 20 mm ring (−12.27 % at 12.7 mm) and B822 blocks on the example ring (+8.82 % at 4.25 mm). Its "it" can be taken for the torque. Its one valid point, naming the count in magnets rather than poles, is carried by the final sentences[4] ("fewer magnets").

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-3 PASS):
- SF-1 (rounds 1 to 3), watch_out[0] (28 words): "The model's end loss barely grows with the gap: at a 4.4 mm gap a 3D check found f_end 11 % high, so treat wide-gap torque as optimistic." Why: in "Its end loss ... it read 11 % high", "Its" can attach to the bench prototype named at the end of the entry above it, and "it" to "end loss", which inverts the direction. The 11 % is the end factor's (+11.23 %); the total free-space torque is +7.0 %.
- SF-2 (rounds 1 to 3), watch_out[1] (28 words): "Below about a third of a pole pitch of overlap it reads increasingly low; at or below c_end·τ_p it gives no prediction and the end-effect check flags it." Why: at τ_p/3 (2.94 mm) the model is only −0.58 % against 3D, and −20.7 % at 2.0 mm. Round 3's W1 (under SF-C) replaces it. (Superseded: W1 applied by the task review fix round 1 below.)
- SF-A (rounds 2 and 3), sources[1]: scope the figure as sentences[4] does (34 words): "Physics review of 2026-10-15: the end factor against free-space 3D by magnet length (within 3 % from 3.2 mm on the example design), by magnets per ring and at a 20 mm ring radius". Round 3 offers a 32-word form that also names block width: "Physics review of 2026-10-15: the end factor against free-space 3D by magnet length (within 3 % from 3.2 mm on the example design), by magnets per ring, block width and ring radius".
- SF-C (round 3, new; physics; outside round 3's scope): narrow or tightly packed blocks read HIGH against 3D.
  - B822 (3.17 mm wide) fits the example ring at 10, 12, 16 and 20 per ring, with the end-effect check "OK". The model's f_end reads above 3D there: +1.87, +3.29, +4.62 and +5.23 % at 12.7 mm, and up to +27 % at 3.2 mm. B842SH at 18 per ring on the 20 mm ring reads +5.99 % at 4.25 mm. The model's end loss c_end·τ_p tracks only the pole pitch, while the 3D loss stays about 1.2 to 1.6 mm at this gap until the pitch gets long.
  - sentences[4] stays true: it is scoped to the example design and to its two named levers.
  - watch_out[1]'s unscoped "reads well low" is false for B822 below τ_p/3 (+3.80 % at 2.5 mm with 10 blocks; +9.29 % at 2.0 mm with 12). It is true at the five probe designs and at the brief's extra probes, which is why rounds 1 and 2 passed it.
  - Candidate texts, checked by the reviewer for form: W1, watch_out[1] (30 words): "On the example design it reads low below a third of a pole pitch of overlap; at or below c_end·τ_p it gives no prediction and the end-effect check flags it." W0, watch_out[0] (29 words; it folds in SF-1 and needs its own per-note pass, because it drops "treat wide-gap torque as optimistic"): "The model's end loss ignores block width and barely grows with the gap; 3D read f_end 11 % high at a 4.4 mm gap and high with narrower blocks."
  - The reviewer recommends W1 before the flip if a 4th round is taken, and otherwise routing it with SF-4 to the final fix wave or a G-3 re-review. The controller closed the in-task loop at the round-3 PASS, so it is routed. (W1 applied by the task review fix round 1 below; W0 and the rest of SF-C stay open.)
- SF-4 (rounds 1 to 3; outside T13's fields, seen in the Assumptions view): the `end_effect` assumption rationale (`assumptions.rs:71`) still says "Checked at 10 magnets per ring and a 1.4 mm gap, it is within 2 % of free-space 3D for magnets 3.2 mm and longer", which the note now contradicts: 2 % is exceeded from about 3.5 to 6.0 mm, the scope admits the 20 mm ring at −12 %, and it is blind to block width. Route it to the final fix wave or a G-3 re-review. Also from round 1: phys-new-notes T-B ("end effects grow with fewer poles") is a statement about the model only. At fixed blocks the free-space 3D end loss grows with more magnets per ring, so no note may present "more magnets lose less at the ends" as physics.

### Task review fix round 1 (2026-10-15): the short-overlap watch-out scoped to the example design

- The task code review (`task-13-review.md`, Important I-1) found watch_out[1] false at designs
  the inputs reach, in the optimistic direction. That is round 3's SF-C, which had been routed to
  the final fix wave. The review recommended applying round 3's W1 now, with a per-note re-review
  (controller ruling 7).
- **watch_out[1]** (round 3's W1, verbatim; 30 words by `word_count`, at the cap):
  - Old: "Below about a third of a pole pitch of overlap it reads well low; at or below c_end·τ_p it
    gives no prediction and the end-effect check flags it."
  - New: "On the example design it reads low below a third of a pole pitch of overlap; at or below
    c_end·τ_p it gives no prediction and the end-effect check flags it."
  - Superseded by the text re-reviews' MF-1 items below. Re-check 1 made it "Below a third of a
    pole pitch of overlap f_end can read well high or low; at or below c_end·τ_p there is no
    prediction and the end-effect check flags it." Re-check 2 dropped the pole-pitch boundary: it
    is now "At short overlap f_end is least reliable and can read too high or too low; at or below
    c_end·τ_p there is no prediction and the end-effect check flags it." The 3D figures below are
    steel-free, but the example design has magnetic steel behind both rings, and with that steel
    the model reads high there.
  - Why: the unscoped claim is false for the narrower B822 block on the example ring with the
    end-effect check "OK". Free-space 3D reads the model high there: +3.80 % at 2.5 mm with 10
    blocks, and +9.29 % at 2.0 mm (the magnet length slider's floor) with 12. "Reads low" sounds
    conservative, but for those blocks the short-magnet torque is optimistic. "Well low" also
    overstated the case near τ_p/3 on the example design (−0.58 % at 2.94 mm; SF-2). W1 narrows the
    claim to the design the 3D data cover: −0.58, −5.75 and −20.72 % at 2.94, 2.5 and 2.0 mm.
    The no-prediction clause is unchanged.
- **Engine side, reproduced** with `compute_all` in a scratch crate outside the repo (B822 on both
  rings of the example ring; the example's c_end 0.15):
  - 10 per ring at 2.5 mm: the blocks fit (3.17 mm wide against B842SH's 6.35); τ_p 8.8090 mm, so
    τ_p/3 is 2.9363 mm and c_end·τ_p 1.3214 mm; f_end 0.471458; check "OK"; pull-out 0.086963 N·m.
  - 12 per ring at 2.0 mm: the blocks fit; τ_p 7.3409 mm, τ_p/3 2.4470 mm, c_end·τ_p 1.1011 mm;
    f_end 0.449436; check "OK"; pull-out 0.095710 N·m.
  - The model gives 10 B822 at 2.5 mm the same f_end as the example design at 2.5 mm (0.471458),
    because f_end does not depend on block width. 3D gives them opposite signs (+3.80 % against
    −5.75 %): this is SF-C's cause.
- **Test.** New: `tests/explain.rs`
  `the_end_effect_note_s_short_overlap_watch_out_is_scoped_to_the_example_design`, computed from
  the engine and the input metadata:
  - at both B822 designs above (12 per ring at the slider's floor, read from the metadata), the
    blocks fit, are narrower than the example's, sit below a third of the pole pitch, and read
    the end-effect check "OK";
  - the watch-out's claim (the clause before ";") says "reads low" on the example design.

  The old text fails it ("scopes \"reads low\" to the example design").

  The re-review's MF-1 below renamed it
  `the_end_effect_note_s_short_overlap_watch_out_names_both_directions`: the claim now names both
  directions and no one design.
- Not changed: W0 (watch_out[0]) and the rest of SF-C (the note does not say that narrower or more
  tightly packed blocks read high), SF-1, SF-A and SF-4. They stay open as listed above.
- Text re-reviews of this field follow (controller ruling 7). Their reports are below, as
  `### Re-review: code-review fix round 1` and
  `### Re-review: code-review fix round 1 (re-check 2)`.

### Re-review: code-review fix round 1

The reviewer's report for the record follows unchanged. Its "(method above)" is the 3D method in
the full answer, `.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-13-textreview-1.md`:
rings unrolled to a plane, two solvers (exact Fourier and finite difference) with no code shared
with rounds 1 to 3, ideal steel, and the 3D table by length for each circuit.

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; text re-review 1 of the T13 task review's fix round 1 (`bbec565`).
- Scope: end_effect watch_out[1], changed to round 3's W1: "On the example design it reads low below a third of a pole pitch of overlap; at or below c_end·τ_p it gives no prediction and the end-effect check flags it." Also the fields it depends on: sentences[1] to [4] and watch_out[0].
- Read:
  - the diff `HEAD~3..HEAD`, `task-13-review.md` I-1, `task-13-inreview-r3.md` SF-C, and the record's `## end_effect`;
  - the trap lists (slice-2 spec 8, slice-1 spec 8, phys-new-notes T-A to T-P, the Global Constraints extra);
  - the caps and plain-words rules;
  - `model.rs` 94-114, 165 and 970-1004;
  - `metal_design.rs` 99-104;
  - the audit's M8 to M10 rows and limits line.
- Dump `target/rereview-T13-1.txt`: the field renders identically at the five probe designs. The three end_effect tests in `tests/explain.rs` pass.
- **Engine side, confirmed.**
  - 10 B822 per ring at 2.5 mm: f_end = 1 − 1.3214/2.5 = 0.4715.
  - 12 B822 per ring at 2.0 mm: τ_p 7.3409 mm, f_end = 1 − 1.1011/2.0 = 0.4494, check "OK".
  - The example design at 2.0 mm: f_end 0.3393, check "OK".
  - At L <= c_end·τ_p, f_end <= 0 and the check reads "End-effect model out of range". The no-prediction clause holds.
- **Independent 3D check, with steel** (method above):
  - The example design has magnetic steel behind both rings (`coupling.backiron` 1), but every 3D figure in rounds 1 to 3 is steel-free.
  - With ideal steel roughly flush with the magnets, model/3D on the example ring is −4.0 % at 2.0 mm, +12.8 % at 2.5, +17.6 % at 2.9, +19.0 % at 3.2 and +6.2 % at 12.7 mm.
  - With overhang or infinite plates the result is the same in sign and larger.
  - Steel-free on the same planar method: −19.6, −4.6, +0.3, +2.1 and +1.1 %, within about 1 % of the rounds' ring results.
  - The long-overlap end loss is about 1.98 mm with steel, 1.44 mm without, and 1.32 mm in the model.
- **Verdict: MUST_FIX.** MF-1: "On the example design it reads low" is false for the example design as shipped (steel) below a third of a pole pitch, except near the 2 mm floor, and it errs in the optimistic direction. Replacement, 30 words: "Below a third of a pole pitch of overlap f_end can read well high or low; at or below c_end·τ_p there is no prediction and the end-effect check flags it."
  - It is true at every design checked: high with steel and with B822; low without steel at 2.0 mm.
  - It names f_end in place of "it".
  - The no-prediction clause is unchanged in meaning.
  - The test and the docs that quote the fix-round wording follow.
- **Should-fix, for the controller.**
  - SF-S (new): the steel circuit's end factor is unchecked by any 3D model with steel, and reads about 5 to 6 % high at 12.7 mm in the ideal-steel planar check. sentences[4] is scoped "Steel-free" and stays true, but nothing warns a steel-backed design. Route it with SF-4 and SF-A, and add a backlog item (risk: physics) for a permeable-steel 3D check.
  - Still open: SF-1, SF-A, W0 and the rest of SF-C, and SF-4.
- **Traps:** none written by the current text or by the replacement.
  - "The end factor is accurate": the replacement states that error.
  - T-B: no magnet-count lever.
  - "More magnets make the coupling stiffer": not written.
  - No torque is called guaranteed.
  - The coupling is not said to protect the gearbox, and no joint method is chosen or sized.
  - No "clamp".
- **Form.**
  - Current text: 30 words, 1 sentence.
  - Replacement: 30 words, 1 sentence.
  - Both pass `plain_text_violations` and `audit_ids`, with no placeholder.

- MF-1, applied verbatim by the implementer (session model), 2026-10-15:
  - watch_out[1]: "On the example design it reads low below a third of a pole pitch of overlap; at
    or below c_end·τ_p it gives no prediction and the end-effect check flags it." -> "Below a third
    of a pole pitch of overlap f_end can read well high or low; at or below c_end·τ_p there is no
    prediction and the end-effect check flags it." (30 words, 1 sentence; the caps and
    plain-words tests pass).
  - Test: `tests/explain.rs`
    `the_end_effect_note_s_short_overlap_watch_out_is_scoped_to_the_example_design` is now
    `the_end_effect_note_s_short_overlap_watch_out_names_both_directions`. Computed from the
    engine and the input metadata, it keeps the two B822 pins (10 per ring at 2.5 mm, 12 at the
    magnet length slider's floor: the blocks fit, are narrower than the example's, sit below a
    third of a pole pitch, have f_end above 0 and read the end-effect check "OK"). It adds the
    example design at the slider's floor, read from the metadata, with the same pins except the
    width. It asserts that the claim (the clause before ";") has the whole words "high" and
    "low" ("below" does not count) and does not contain "on the example design". The
    fix-round text fails it ("names both directions").
  - Docs that quoted the fix-round wording now give the new one: the README `tests/explain.rs`
    row, the fix-round subsection above (a superseded line and the test's new name), BL-056 and
    the update tracker.
  - Not applied in this round (the round's list names MF-1 only): SF-S, and the carried SF-1,
    SF-A, W0 and the rest of SF-C, and SF-4. They stay open for the controller.

### Re-review: code-review fix round 1 (re-check 2)

The reviewer's report for the record follows unchanged. Its "method in the full answer" is the
3D check in `.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-13-textreview-2.md`:
rings unrolled to a plane, the interaction energy of the magnets' surface charges in a 2D
Fourier transform, free space and ideal steel plates behind both rings, with no code shared with
rounds 1 to 3 or with text re-review 1; and its table of ten circuits.

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; text re-review 2 of the T13 task review's fix round 1 (`6de4ad3`, text re-review 1's MF-1 applied).
- Scope: end_effect watch_out[1], now "Below a third of a pole pitch of overlap f_end can read well high or low; at or below c_end·τ_p there is no prediction and the end-effect check flags it." Also the fields it depends on: sentences[1] to [4] and watch_out[0].
- Read:
  - the diff `HEAD~3..HEAD`;
  - `task-13-textreview-1.md`, `task-13-review.md` I-1, `task-13-inreview-r1.md` to `-r3.md`, and the record's `## end_effect`;
  - the trap lists (slice-2 spec 8, slice-1 spec 8, phys-new-notes T-A to T-P, the Global Constraints extra) and the caps and plain-words rules;
  - `model.rs` 94-114, 151-153, 182-184, 633-643 and 970-1004; `metal_design.rs` 99-104; `plain.rs`; `probes.rs`.
- Dump `target/rereview-T13-2.txt`: the field renders identically at the five probe designs. The three end_effect tests and the rewritten-notes caps and plain-words tests pass.
- **Engine side, confirmed** (`compute_all`, scratch crate):
  - the example design at 2.0, 2.5 and 2.9 mm: f_end 0.3393, 0.4715 and 0.5444, check "OK";
  - B822 ×10 at 2.5 mm: f_end 0.4715; B822 ×12 at 2.0 mm: f_end 0.4494 (τ_p 7.3409 mm), check "OK";
  - at L <= c_end·τ_p, f_end <= 0 and the check reads "End-effect model out of range". The no-prediction clause holds.
  - The hub (13.0 mm) and cup (15.5 mm) keep their lengths when the magnets are shortened, so the steel overhangs a short magnet.
- **Independent 3D check** (planar Fourier, free space and ideal steel plates; method in the full answer, `task-13-textreview-2.md`). It reproduces the rounds' free-space ring results within about 1 % and text re-review 1's infinite-plate results within 0.3 %.
  - Example design as shipped (steel), model/3D: +15.6 % at 2.5 mm; +20.5 % at 2.94 (τ_p/3); +21.4 % at 3.2; +21.6 % at 3.6; +20.3 % at 4.25; +18.2 % at 5.0; +14.6 % at 6.35; +11.4 % at 8.0; +6.65 % at 12.7. Its crossover from low to high is at 2.02 mm, and its largest error is +21.5 % at 3.66 mm.
  - The crossover sits at τ_p/3 only for the steel-free example ring (2.91 mm). Elsewhere: 8 magnets with steel 0.27 τ_p, 20 mm ring with steel 0.30 τ_p, B822 ×10 without steel 0.27 τ_p; none for 8 magnets or the 20 mm ring without steel (always low) or for B822 ×12 (always high).
  - The largest error lies at short overlap in all ten circuits checked, between 0.16 and 0.42 τ_p. The largest optimistic errors lie above τ_p/3: example with steel +21.5 % at 3.66 mm; B822 ×10 with steel +31.2 % at 3.24 mm; B822 ×12 without steel +16.0 % at 3.02 mm.
- **Verdict: MUST_FIX.** MF-1: the text is literally true, but its "below a third of a pole pitch" boundary reads as the edge of the hazard. On the example design as shipped, the largest error (+21.5 %, optimistic) lies on the safe side of that edge, so the field implies an accurate end factor where it errs most. Replacement, 29 words: "At short overlap f_end is least reliable and can read too high or too low; at or below c_end·τ_p there is no prediction and the end-effect check flags it."
  - It is true at every design checked: the largest error lies at short overlap; high with steel (+15.6 % at 2.5 mm); low without steel (−19.9 % at 2.0 mm).
  - It draws no boundary, names both directions and no one design, and names f_end as the subject.
  - The test (finder, comment, a no-"pole pitch" assert) and the docs that quote the wording follow.
- **Should-fix, for the controller.**
  - SF-S raised: about +20 % with steel at 2.9 to 4.25 mm on the default design, where sentences[4]'s steel-free "within 3 % at 3.2 mm or longer" invites trust.
  - SF-T (new): pin the 304 design at the floor as the test's low side.
  - SF-C / W0 with new evidence: B822 reads high above τ_p/3 without steel.
  - Still open: SF-1, SF-A and SF-4.
- **Traps:** none written by the replacement.
  - "The end factor is accurate": the current text implies it above τ_p/3, and the replacement does not.
  - T-B: no magnet-count lever.
  - "More magnets make the coupling stiffer": not written.
  - No torque is called guaranteed.
  - The coupling is not said to protect the gearbox, and no joint method is chosen or sized.
  - No "clamp".
- **Form.**
  - Current text: 30 words, 1 sentence.
  - Replacement: 29 words, 1 sentence.
  - Both are free of banned patterns and audit ids, with no placeholder.

- MF-1, applied verbatim by the implementer (session model), 2026-10-15:
  - watch_out[1]: "Below a third of a pole pitch of overlap f_end can read well high or low; at or below
    c_end·τ_p there is no prediction and the end-effect check flags it." -> "At short overlap
    f_end is least reliable and can read too high or too low; at or below c_end·τ_p there is no
    prediction and the end-effect check flags it." (29 words, 1 sentence; the caps and
    plain-words tests pass).
  - Test: `tests/explain.rs`
    `the_end_effect_note_s_short_overlap_watch_out_names_both_directions` finds the watch-out by
    "short overlap" (any case) or "c_end·τ_p", no longer by "a third of a pole pitch". It keeps
    its engine pins (the example design at the magnet length slider's floor, and 10 B822 per
    ring at 2.5 mm and 12 at the floor: the blocks fit, the B822 blocks are narrower than the
    example's, each overlap is below a third of a pole pitch, f_end is above 0 and the
    end-effect check reads "OK"). A third of a pole pitch is now the test's own measure of a
    short overlap, not the note's, and its comment says so. It keeps the whole-word "high" and
    "low" asserts and the no-"on the example design" assert, and adds that the claim (the
    clause before ";") does not contain "pole pitch" (any case). Re-check 1's text fails it
    ("draws no pole-pitch boundary"). Its doc comment gives the new wording and this review's
    figures, with the earlier methods' as ranges.
  - Docs that quoted re-check 1's wording now give the new one: the README `tests/explain.rs`
    row, the fix-round subsection above (its superseded line and its pointer to both
    re-reviews), BL-056 and the update tracker.
  - Not applied in this round (the round's list names MF-1 only): SF-S, SF-T, W0 and the rest
    of SF-C, SF-1, SF-A and SF-4. They stay open for the controller.

## calibration

Task T13. Rewrite of "The one-point calibration" as "Calibration: what scales the torque
estimate"; it now also owns `calibration.model_torque_Nm` (the prototype's modelled torque, the
measured factor's divisor), which had no owner. Reviewer: physics reviewer (session model), fresh
context. Date: 2026-10-14 (the date the slice plan fixes for the note's `review` field; all three
rounds ran on 2026-10-15).

Rounds: 3. Verdict: PASS. Note: "Calibration: what scales the torque estimate" (plan T13).

Verdict by round: 1 MUST-FIX (2 items); 2 PASS; 3 PASS (no field changed after round 2). Each
round's full answer is `task-13-inreview-r<N>.md`, as under `## end_effect`; the two notes were
reviewed together.

The text reviewed in round 1 already carried A-2 (the two bench entries are `Step::Enter` with no
aim, so the panel draws "Enter <input>" and never invites tuning them to a design; the assumed
factor is `Step::Lower` with no aim, under A-1's model-assumption exception) and A-13's
plain-words sources. Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump; `notes.rs` calibration literal;
  - `model.rs:583-599` (`select_calibration_factor`: circuit 0, npole = the prototype's poles per ring, B842SH on both rings);
  - `calibration.rs:1-235`: the prototype at `test_temp_C` through `alpha_br_per_C`, model = T_2D·f_end·f_cal_original, f_cal_updated = f_cal_original·measured/model, so the assumed factor cancels;
  - `material_library.rs:355-361, 731-740` (304 is "304 stainless (non-magnetic)", not ferromagnetic); `assumptions.rs:74-80`;
  - math audit M4 to M6 and the T3-RC5 lenses (raw prototype −3.8 %).
- Numbers reproduced from `compute_all`:
  - `model.f_cal`: 0.9500, 0.9500, 1.049 (the 304 back iron design, circuit 0, the only match), 0.9500, 0.9500. The renders match.
  - `calibration.f_cal_updated` 1.049450, `model_torque_Nm` 1.629425 N·m, measured/model 1.104684 and model error −9.48 % at all five.
- Change entries:
  - Original calibration coefficient −0.005 (0.95 to 0.945): outside the match, pull-out, hot-low and cold-high scale by exactly 0.994737, as do all 20 predicted torque rows of the model, metal and temperature results (probe section H). At the 304 match `model.f_cal` and the pull-out are unchanged (ratio 1.000000).
  - Measured pull-out +0.01 N·m: `f_cal_updated` 1.049450 to 1.055280 at all five; `model.f_cal` moves only at the match.
  - Test temperature +0.5 °C: the prototype model 1.629425 to 1.627470 N·m and `f_cal_updated` 1.049450 to 1.050711 at all five. At α = 0 nothing moves; this is an edge case and noted.
- Rule probes on the 304 design:
  - Resized (ring radius 20 mm via the Size tab and raw, gap 2.0 mm, axial 20 mm): it keeps 1.049450.
  - At 8 magnets per ring: 0.9500, a −9.5 % step in every torque.
  - At 8 with Total installed magnets 16: 1.567470 (+49 % on the bench factor), the watch-out's hazard.
  - The default with no back iron at all: a match at 1.049450.
  - The same with custom dimensions equal to B842SH: 0.9500, pull-out 1.5365 against 1.6974 N·m.
- Traps checked:
  - "edit the calibration inputs to match your design": the watch-out forbids it.
  - "the bench calibration applies beyond prototype-matching designs": not written; s3 states that the rule ignores size, and s4 that the step is "by rule, not physics".
  - "steel gives the torque shown": not written.
  - No torque is called guaranteed; nothing on joints or gearbox protection; no "clamp".
  - A-2's Enter entries keep "Raise <Measured pull-out torque>" from reading as an invitation to tune it.
- Form:
  - Sentences 22, 25 (MF-5), 25 (MF-6), 21 and 23 words.
  - Change texts 23, 23 and 17 words; the watch-out 26 words.
  - Sentence 0 is plain; the placeholder sits only in s4's "here" clause.
  - The sources are A-13's lines, plain, with no ids and no "clamp".
- Must-fix items and how they were applied (round 1): MF-5 (sentences[1]) and MF-6 (sentences[2]): "magnetic steel" in place of "steel", because the 304 design is steel-backed and gets the measured factor; "B842SH parts" in place of "blocks". Pending the round-2 re-check.
- Verdict, round 1: MUST_FIX.

Round 2:

- Round 2 (physics reviewer, session model, fresh context, 2026-10-15) re-checked:
  - sentences[1] ("a design backed by magnetic steel");
  - sentences[2] ("no magnetic steel behind the magnets ... B842SH parts on both rings").
- Rule probes (`model.f_cal`, `materials.circuit_backiron`):
  - back-iron picks at the default:
    - 4140, 1018, 12L14, 416, 17-4PH H1150 and H900: circuit 1, 0.9500;
    - 304 and 6061: circuit 0, 1.049450.
  - "Back iron" set to "no back iron" with the 4140 pick: circuit 0, 1.049450 (the engine then takes the hub as aluminium).
  - 304 with a B842-N52 outer part: 0.9500.
  - 304 with N52 typed in the grade fields on B842SH parts: ignored (Br 1.30 T, pull-out 1.697369 N·m unchanged), still 1.049450.
  - 304 at a 25.4 mm axial length: 1.049450.
  - Round 1's custom-dimensions probe (B842SH sizes, no part: 0.9500) stands.
- At the five designs `model.f_cal` is 0.9500, 0.9500, 1.049, 0.9500 and 0.9500.
  - The four magnetic-steel designs get the assumed factor ("a judgement").
  - The 304 design, which has steel but not magnetic steel, gets the measured one. Sentences 1, 2 and 4 now agree at every design.
- Dependent text: s3, s4, change[0] ("no steel bench test backs any value", true: no bench test had any steel) and the watch-out are unchanged and true.
- Form: sentences 22, 25, 25, 21 and 23 words; change texts 23, 23 and 17; watch-out 26; all fields plain, no audit id, no "clamp".
- Must-fix items of round 1 (MF-5, MF-6) are applied verbatim and pass.
- Verdict, round 2: PASS.

Round 3:

- Round 3 (physics reviewer, session model, fresh context, 2026-10-15): no calibration field changed after round 2.
- The re-check confirmed that every field still passes `plain_text_violations` and `audit_ids` and names no clamp. Sentences are 22, 25, 25, 21 and 23 words; changes 23, 23 and 17; the watch-out 26.
- A-2's Enter entries are in place.
- Verdict, round 3: PASS (unchanged since round 2).

Must-fix items applied:
- sentences[1] (round 1, MF-5: the rule tests the back-iron circuit, not steel; the 304 back iron design has non-magnetic stainless behind the magnets and gets the measured factor): "The only bench test it holds is of a prototype without steel, so for a steel-backed design the factor is a judgement." -> "The only bench test it holds is of a prototype without steel, so for a design backed by magnetic steel the factor is a judgement."
- sentences[2] (round 1, MF-6: the same circuit point, and the rule reads the picked part name, so custom dimensions equal to B842SH get the assumed factor): "A design gets the measured factor only with no steel behind the magnets, the prototype's magnets per ring and B842SH blocks on both rings." -> "A design gets the measured factor only with no magnetic steel behind the magnets, the prototype's magnets per ring and B842SH parts on both rings."

Seeds (A-24): none names `calibration` (S-3 and S-4 are `end_effect`'s).

Considered, no change (the reviewer's reasons):
- "Every torque" (sentences[4] and change[0]) leaves out the slip drag (`temperature.slip_loss.drag_Nm`) and the requirement rows (`metal.required_20C_Nm`, `metal.cold_for_hot_min_Nm`), which do not scale. Sentence 0 frames the subject as the torque model's predictions, and every one of those scales (round 1).
- change[2] at a coefficient α of 0, which the slider allows: the test temperature then moves nothing. This is an edge case, and the text is right at every real grade (round 1).
- change[0]'s "no steel bench test backs any value" is true as written: no bench test had any steel. A hyphenated "no magnetic-steel bench test" would only align the term (round 2).

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS):
- SF-3 (rounds 1 to 3; not on screen): the doc comment of `the_calibration_note_s_levers_and_rule_hold_at_every_probe_design` in `tests/explain.rs` still says "(the rule's own terms: no steel behind the magnets, ...)"; to follow MF-6 it becomes "no magnetic steel behind the magnets". The test's `circuit_backiron == 0` check is already right.
- SF-4 (rounds 1 to 3; outside T13's fields, seen in the Assumptions view): the `calibration_factor` assumption rationale (`assumptions.rs:78`) still says "no steel behind the magnets ... B842SH blocks", while a 304 design matches and the note now says "no magnetic steel" and "B842SH parts". Route it with `end_effect`'s SF-4 to the final fix wave or a G-3 re-review, so the Assumptions view and the notes agree.
