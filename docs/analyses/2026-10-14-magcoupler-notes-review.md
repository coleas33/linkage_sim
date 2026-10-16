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

### Re-review: whole-branch review fix wave 1

The slice-2 whole-branch review's fix wave 1 (2026-10-16; `final-review.md` MF-1 to MF-3,
controller ruling 10) applied three of the should-fix items left open above: round 1 S-1
(sentences[3] and [4] swapped, words unchanged), S-3 (the pull-out gloss, now watch_out[0],
moved first) and S-2 (the four engine pins in `tests/explain.rs`). Round 2 S-2
("infinite-length") is not taken: ruling 10 keeps "endless-ring". The changed texts were
re-reviewed in a fresh context. The reviewer's report for the record follows unchanged (full
answer: `.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-21-textreview-1.md`).

- Reviewer: physics reviewer (session model), fresh context, 2026-10-16; text re-review 1 of the
  T21 whole-branch review fix wave 1 (working tree on 47bdcd7), `task-21-textreview-1.md`.
- Scope:
  - sentences[3] and [4], swapped with the words unchanged (ruling 10 a);
  - watch_out[0], ruling 10 b's pull-out gloss, moved first;
  - watch_out[1], the R_g² line, moved second;
  - the fields that depend on them: sentences[1], [2] and change[2].
- Dump: `target/rereview-T21-1-torque_from_shear.txt`, the five probe designs; every aim reads
  "moves as it says".
- Tests: `notes.rs` 29 passed; the RF-1 panel test passes; `tests/explain.rs` 56 passed, MF-3's
  four engine pins among them, at all five designs.
- **sentences[3]: PASS.**
  - The engine has no N in R_g (`model.rs:918-971`). τ_p = 2π·R_g/N.
  - The cup grows with fewer magnets only for flat blocks (r_pocket = (A_back + bond)/cos(π/N)).
  - Pin (i) holds at all five designs.
  - "Each pole" is now first used where a pole's span is given (R8).
- **sentences[4]: PASS.** The words are unchanged and true (T11 round 2). Pins (ii) and (iii)
  hold at all five designs. "It" is σ.
- **watch_out[1]: PASS.** It is unchanged; the 8-magnet dip is pinned, and "pull-out" is now
  glossed before it.
- **watch_out[0]: MUST_FIX (MF-A).**
  - The appositive equates the dashboard's nominal number with the most the rings pass. Read as
    capacity, that is the slice-1 trap "pull-out is guaranteed"; read as a ceiling, it is T-O's
    "can never exceed".
  - That contradicts the result's own help ("not a guaranteed minimum"), its label "(nominal)"
    and `pullout_angle` sentences[0].
  - On the default design the end factor likely reads high (end_effect sentences[4]; +6.65 % at
    12.7 mm with ideal steel). When the magnets are cold the rings pass more (cold-high 3.823 N·m
    against 2.688 N·m).
  - Replacement, ruling 13's text verbatim (30 words): "The dashboard's pull-out torque, the most
    the rings are expected to pass before slipping, is T_2D corrected for the magnet ends and by
    a calibration factor (here: {model.torque_2d_Nm} becomes {model.pullout_Nm})."
  - T_pull = T_2D·f_end·f_cal reproduced at all five designs: 2.68848, 1.96835, 1.69737, 0.207075
    and 2.53926 N·m.
  - Test: MF-2's test also asserts "expected to pass".
  - Docs that quote the gloss follow: the tracker and the MF-2 commit message.
- **Traps:** none written once MF-A is applied.
  - R_g² is said only at fixed σ.
  - Fewer magnets grow the cup and do not keep it (T-P).
  - Nothing says fewer poles reach across the gap better, that fill is w/τ_p, or that more magnets
    make the coupling stiffer.
  - Nothing says the coupling protects the gearbox.
- **Form:** sentences 23, 23, 25, 25 and 25 words; watch-outs 30 (after MF-A) and 20. Plain
  words; no audit id; no input label typed.
- **Verdict:** MUST_FIX (MF-A). Everything else passes.

- MF-A, applied verbatim by the implementer (session model), 2026-10-16:
  - watch_out[0]: "The dashboard's pull-out torque, the most the rings pass before slipping, is
    T_2D corrected for the magnet ends and by a calibration factor (here: {model.torque_2d_Nm}
    becomes {model.pullout_Nm})." -> "The dashboard's pull-out torque, the most the rings are
    expected to pass before slipping, is T_2D corrected for the magnet ends and by a calibration
    factor (here: {model.torque_2d_Nm} becomes {model.pullout_Nm})." (controller ruling 13's
    text; 30 words, 1 sentence; the caps and plain-words tests pass).
  - Test: `notes.rs` `the_torque_note_glosses_pull_out_where_it_is_first_used` also asserts
    that watch_out[0] contains "expected to pass". The text without "expected" fails it.
  - Docs that quote the gloss now give the new one: the update tracker's fix-wave bullet and
    the MF-2 commit message.
  - No should-fix item of this round concerns this note.

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

### Re-review: whole-branch review fix wave 1

The slice-2 whole-branch review's fix wave 1 (2026-10-16; `final-review.md` MF-4) rewrote
sentences[4] to name the steel case (SF-S of the re-reviews above) and asked for a confirming
round on watch_out[1] as committed in 13b34f5, whose record above ends on a MUST_FIX. Both were
re-reviewed in a fresh context. The reviewer's report for the record follows unchanged (full
answer: `.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-21-textreview-1.md`).

- Reviewer: physics reviewer (session model), fresh context, 2026-10-16; text re-review 1 of the
  T21 whole-branch review fix wave 1, `task-21-textreview-1.md`.
- Scope:
  - (a) watch_out[1] as committed in 13b34f5 (its record ended on a MUST_FIX with no confirming
    round);
  - (b) sentences[4], now "Steel-free 3D puts the example design's f_end within 3 % from 3.2 mm;
    with magnetic steel behind the magnets, the default, f_end likely reads high.";
  - the fields that depend on them: sentences[1] to [3], watch_out[0] and sources[1].
- Dump: `target/rereview-T21-1-end_effect.txt`.
  - f_end 0.8960, 0.8699, 0.8960, 0.8960 and 0.8229; τ_p 8.809, 11.01, 8.809, 8.809 and
    15.00 mm; L 12.70 mm.
  - Both aims read "moves as it says" at all five designs.
- Tests: the four end-effect tests in `tests/explain.rs` pass, the new
  `the_end_effect_note_s_3d_band_names_the_steel_case` among them.
- **(b) sentences[4]: PASS.**
  - The model's f_end has no circuit term, so the sentence compares one number with two 3D
    references.
  - Steel-free, rings (T13 rounds 1 to 3): +0.94 % at 3.2 mm, the worst +2.70 % at 4.25 mm,
    +0.05 % at 50.8 mm. This is within 3 % on the example design.
  - With ideal steel (text re-reviews 1 and 2): high from 2.02 mm, +21.4 % at 3.2 mm and +6.65 % at
    12.7 mm.
  - The model reads high from 3.2 mm at both limits, so "likely reads high" holds whatever the
    steel's permeability or saturation.
  - "The default" is true: the default design is on circuit 1.
  - The other steel probe designs read high at their rendered 12.7 mm. Below their crossovers
    they read low, which watch_out[1] covers.
  - Dropping "below 3D with fewer magnets or larger rings" states nothing false.
  - The trap "the end factor is accurate" is removed. 25 words.
- **(a) watch_out[1]: PASS, confirming 13b34f5.**
  - All ten circuits of text re-review 2 have their largest error at short overlap (0.16 to
    0.42 τ_p): too high on the example design with steel (+15.6 % at 2.5 mm), too low on the 304
    design at the floor (−20.7 %).
  - At L <= c_end·τ_p, `end_effect_in_range` is false and the check reads "End-effect model out of
    range". 29 words.
- **Dependent fields:** sentences[1] to [3] are unchanged and read with the new sentences[4].
  watch_out[0] and sources[1] are unchanged; their should-fix items are carried.
- **Should-fix (carried, for the controller):**
  - SF-A extended: scope sources[1] to the example design, and add a source for the ideal-steel
    checks (texts in `task-21-textreview-1.md` S-1).
  - SF-1: watch_out[0]'s opening "Its" (text in S-2).
  - SF-T: the 304 design at the floor as the test's low side.
- **Traps:** none written.
  - "The end factor is accurate": the steel case now reads high.
  - T-B: no magnet-count lever.
  - Nothing says more magnets make the coupling stiffer.
  - No torque is called guaranteed.
  - Nothing says the coupling protects the gearbox.
  - No "clamp".
- **Verdict:** PASS. No must-fix in this note.
  - The steel clause's short-magnet exception is missing only from the assumption rationale.
    That is MF-B, recorded with the help review.

- No must-fix for this note, and no text of it changed in this round. The round's must-fix
  MF-B (the short-magnet qualifier in the end-effect assumption's rationale) is applied and
  recorded in `docs/analyses/2026-10-14-magcoupler-help-review.md`, `## GUI strings and
  rationales (T10)`. The should-fix items SF-A (extended), SF-1 and SF-T stay open for the
  controller (texts in the full answer, S-1 to S-3).

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

## hot_low_cold_high

Task T14. New note "Hot-low and cold-high: the torque band", Start here step 8 (after the
temperature note, in the full order and in the order shown while the shaft-clamp section is
paused). It owns the band's equations: hot-low, cold-high, the hot-minimum check and its margin,
the required 20 °C torque, the cold rows and cold-high at the gearbox input. The hot-minimum
guide and the hot-low, hot-minimum and cold-high dashboard rows now link it as their Why link
(they linked the production-variation assumption before). Reviewer: physics reviewer (session
model), fresh context. Date: 2026-10-14 (the date the slice plan fixes for the note's `review`
field; all three rounds ran on 2026-10-15).

Rounds: 3. Verdict: PASS. Note: "Hot-low and cold-high: the torque band" (new; plan T14).

Verdict by round: 1 PASS on physics (the round's verdict was MUST-FIX, for `br_temperature`,
reviewed in the same rounds); 2 PASS (no field changed, not re-reviewed); 3 PASS (no field
changed). Each round's full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-14-inreview-r<N>.md`, and the
fields that rounds 1 and 2 changed, all in `br_temperature`, are listed with their old and new
text in `task-14-inreview-r<N>-changes.md` beside it.

The text reviewed in round 1 already carried A-13's plain-words sources and A-28b's joint-neutral
texts (ruling 11, the shaft-clamp section paused): sentences[3] ("Whatever holds the hub on the
shaft must carry cold-high with margin, and the gearbox output carries it too, so added torque
loads both."), change[0] and change[3] ("the shaft-joint and gearbox loads" in place of the
clamp load) and sources[2] (the design rulings of 2026-10-14 and 2026-10-15 in place of the
clamp's sizing). The gearbox watch-out is the plan's text with live values only (ruling 8: the
coupling sits on the gearbox output, which carries its torque directly); the reviewer proposed
no rephrase. Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump `notes-review-T14.txt`; the `hot_low_cold_high` literal in `notes.rs`;
  - `metal_design.rs:404-466`: cold = T_20·th_i(T_min)·th_o(T_min); hot-low = T_op·(1 − v); cold-high = cold·(1 + v); the hot-minimum check `hot_low < required`; required at 20 °C = required/(th_i·th_o(T_op)·(1 − v)); the nominal margin T_op/required − 1;
  - `metal_design.rs:506-509` (cold-high at the gearbox input = cold-high/(ratio·efficiency));
  - `temperature.rs:918` (hot-day start = hot ambient + driving rise) and `:1336-1350` (hot-day torque = T_20·th_i·th_o(start), checked against the requirement without the allowance);
  - `temperature.rs:1030` (the adhesive load reads cold-high) and `:1066` (the mismatch swing reads the minimum);
  - `model.rs:978-1005` (Br at the operating temperature; T_pull20 = T_pull·Br_i·Br_o/(Br_i,op·Br_o,op));
  - `assumptions.rs:82-87` (production variation, symmetric, not measured);
  - `guidance.rs:158-167` (HOT_MIN_GUIDE, its why and its new link);
  - `dashboard.rs:154-184` (DASHBOARD) and `:259-290` (CHECKS);
  - spec 2 (S2-9), 3.1-3.3, 5 and 8; slice-1 spec 8; phys-new-notes 2 and 3.2; phys-notes-a 7 and 8; amendments ruling 8, ruling 11, A-13, A-24 and A-28/A-28b.
- Numbers reproduced from `compute_all`, in design order (default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius); every rendered value matches the dump:
  - hot-low: 2.285205, 1.673106, 1.442764, 0.176013, 2.158370 N·m.
  - cold-high: 3.823310, 2.799225, 2.413847, 0.338067, 3.611106 N·m.
  - required at 20 °C: 3.164951 at four designs and 3.328629 N·m at Y30 (zero-scatter 2.690209 and 2.829335).
  - cold-high at the gearbox input: 0.804907, 0.589311, 0.508178, 0.071172, 0.760233 N·m. At the default this is above the 0.6 N·m input rating.
  - hot-day start: 65.00 °C at all five. Hot-day torque: 2.589014 ("Meets it nominally (no variation allowance)") at the default, while hot-low reads "Below hot minimum" (T-E reproduced); 1.895539, 1.634574, 0.194067, 2.445316 ("Below it") at the others.
  - Nominal hot margin against the hot-low margin (T-F, never conflated in the text): +7.54 / −8.59 %, −21.27 / −33.08 %, −32.11 / −42.29 %, −91.72 / −92.96 %, +1.57 / −13.67 %.
- Identities, at all five designs:
  - hot-low = pull-out × (1 − v), exactly;
  - cold-high = the same design's pull-out run at the minimum temperature, × (1 + v), within 2.2e-16;
  - cold-high at the input × ratio × efficiency = cold-high, within 2.2e-16;
  - hot-low ≥ requirement exactly when the 20 °C pull-out ≥ the required 20 °C torque, with the requirement at 0.99 and 1.01 × hot-low;
  - the hot-minimum check flips with it.
- Placeholders sit only in "here" or "on this design" clauses (s1, s2, s4, w0, w1). None sits inside a comparison whose truth depends on its value. Sentence 0 has no symbol and no placeholder.
- Levers, one slider step (`step_input`) at all five designs:
  - Axial length +0.01 mm:
    - hot-low, cold-high and the gearbox-input value rise (default 2.285205 to 2.287213; 3.823310 to 3.826670; 0.804907 to 0.805615 N·m);
    - the axial stack grows 31.80 to 31.81 mm at all five (the cup depth follows the magnet length).
  - Allowance −0.005: hot-low rises, and cold-high falls (default 2.285205 to 2.298647, 3.823310 to 3.806687 N·m).
  - Operating temperature −0.5 °C:
    - hot-low rises (default to 2.288050);
    - cold-high and the hot-day torque stay to rounding;
    - the cup-wall rule asks slightly more (a reviewer note, T-J).
  - Minimum temperature −0.5 °C: cold-high and the gearbox-input value rise (default to 3.827591 and 0.805809 N·m), and hot-low is bit-identical.
  - Every aim line in the dump reads "moves as it says".
- Extra probes:
  - Required torque 2.0 N·m:
    - the check reads "Estimate covers hot min";
    - s4 renders 2.532 N·m;
    - the 20 °C pull-out (2.893) is above it, so "passes once ... reaches" reads right.
  - Operating temperature 0 °C:
    - hot-low is 2.579 N·m and passes;
    - s4 renders 2.805 N·m;
    - every sentence still reads true.
  - Operating temperature 120 °C: hot-low is 1.904 N·m and s4 renders 3.798 N·m.
  - Mixed design (inner ring custom dimensions with Y30, outer library):
    - hot-low 0.6342, cold-high 1.137, gearbox input 0.2393 N·m;
    - s4 renders 3.246 N·m.
  - Gearbox input rating 1.0 N·m: watch-out[1] stays true; nothing checks it (`nothing_checks_a_gearbox_torque_against_its_rating_yet`).
  - Axial length 1.0 mm (f_end −0.3214, "End-effect model out of range"):
    - hot-low, cold-high and the gearbox-input value render "no prediction";
    - the required 20 °C torque (3.165 N·m) and the hot-day start (65.00 °C) render, as they do not depend on f_end;
    - no negative torque is rendered, so nothing goes to the controller for T5.
- Traps checked:
  - "hot" is the operating-temperature input, not the hot-day start (w0 says so);
  - the nominal hot margin is never called the hot-low margin;
  - cold-high is "expected", with "margin" (T-O);
  - nothing says the coupling protects the gearbox; w1 says nothing checks it;
  - no shaft-joint method is chosen or sized;
  - no torque is called guaranteed ("a nominal estimate");
  - nothing says the stored 3D fields follow the design;
  - "more magnets make the coupling stiffer" is not written;
  - slice-1 traps none; T-H (gearbox ripple) not written;
  - "clamp" in any field, sources included: none.
- Form:
  - title 6 words, plain and unique against the drawn dashboard labels ("Hot-low torque", "Cold-high torque");
  - sentences 19, 19, 20, 24, 22 words raw (rendered 19, 20, 21, 24, 23);
  - changes 22, 16, 18, 20, each starting in lower case with no label typed;
  - watch-outs 22, 21 (rendered 23, 22);
  - sources 12, 15, 25, 12;
  - every field passes `plain_text_violations` and `audit_ids`;
  - "variation allowance" is glossed by sentence 0's "production scatter"; "hot-day start" is glossed at Start here step 7.
- Seeds: S-6 rejected ("made weaker by heat" double-counts heat against the dashboard pull-out and is false below 20 °C). S-7 rejected ("once ... reaches" states the threshold; "exactly" adds nothing a user acts on).
- Must-fix items: none in this note in round 1. Should-fix offered: SF-1 (change[2]) and SF-3 (sentences[2]).
- Verdict for this note, round 1: PASS on physics. The task's verdict is MUST_FIX because of br_temperature.

Round 2:

- Round 2 (2026-10-15): no field of this note changed in round 1 (`task-14-inreview-r1-changes.md`), and no changed `br_temperature` field changes its meaning. Round 1's PASS on physics stands. Its should-fix items SF-1 and SF-3 remain offered, not applied.

Round 3:

- Round 3 (2026-10-15): no field of this note changed in rounds 1 or 2, and the `br_temperature` field changed in round 2 changes the meaning of no sentence here. A-28b's texts (sentences[3], change[0], change[3], sources[2]) are verbatim. No field, sources included, contains "clamp" (0 of 16 fields), and the guards are empty. Round 1's PASS on physics stands. SF-1 and SF-3 remain offered, not applied. **Verdict for this note: PASS.** Rounds: 3.

Must-fix items applied: none. Round 1 passed this note on physics, and no field of it changed in
any round.

Seeds (A-24):
- S-6 (sentences[0]: "Hot-low and cold-high are the pull-out made weaker by heat and production scatter, and stronger by cold and scatter: the torque band."): rejected in rounds 1 to 3. Reason: "the pull-out made weaker by heat" double-counts heat. The dashboard's pull-out is already at the operating temperature, and hot-low is that value less the allowance only (2.688 × 0.85 = 2.285 N·m, an exact identity): the T-E confusion over which temperature hot-low uses. It is also false below 20 °C: at 0 °C hot-low is 2.579 N·m, above the 20 °C pull-out less the allowance (2.893 × 0.85 = 2.459 N·m). The shipped sentence 0 is true at every probe and, with sentences 1 and 2, answers the three linking rows. (Sentence 0 was replaced by the task review fix round 1 below, so that it names both edges itself; S-6's text was not used.)
- S-7 (sentences[4]: "On this design, hot-low passes exactly when the 20 °C pull-out reaches {metal.required_20C_Nm}, the requirement taken back to 20 °C with the allowance."): rejected in rounds 1 to 3. Reason: "passes once the 20 °C pull-out reaches X" already states the threshold the user acts on, and "exactly when" adds an iff that changes no action. At the boundary itself floating-point rounding makes the two comparisons disagree, so "exactly" claims more than the numbers show. The equivalence holds at all five designs with the requirement at 0.99 and 1.01 times hot-low, which `the_torque_band_note_s_sentences_hold_at_every_probe_design` pins.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-1 PASS until the task review fix round 1 below, which changed sentences[0] only):
- SF-1 (round 1; rounds 2 and 3 keep it offered), change[2] (`coupling.op_temp_C`, Lower; aim unchanged), 16 words: "only if the magnets really run cooler; hot-low rises, while cold-high and the hot-day torque stay." Why: "The hot-day torque is reported apart." repeats watch_out[0]'s words in the same note and does not say what the lever does to it; one step down leaves the hot-day torque unchanged to rounding at all five designs (it is the 20 °C pull-out times the factor at the hot-day start and never reads the operating temperature). Its test: in the levers test's entry 3, assert that the hot-day torque stays.
- SF-3 (round 1; rounds 2 and 3 keep it offered), sentences[2], 21 words raw: "Cold-high, here {metal.torque_cold_high_Nm}, is the pull-out at the minimum temperature plus that allowance: the most torque expected before the coupling slips." Why: in "before it slips", "it" follows "allowance" and "cold-high", and only the coupling slips. The physics is unchanged ("expected" keeps T-O).

Considered, no change (the reviewer's observations, round 1):
- A cooler operating temperature raises the gap flux density the cup-wall rule reads, since Br is taken at the operating temperature (`model.backiron_needed_mm` up about 0.06 % per step; at the 8-magnet design the suggested wall goes from 2.4 to 2.5 mm, a check already red there). phys-new-notes T-J keeps the wall rule's temperature dependence out of note text, and change[2] is an honest-input entry ("only if the magnets really run cooler"), so it stays unnamed.
- The magnet bonds also carry cold-high: the adhesive fatigue screen reads it (7x, 8x, 12x, 84x and 14x margin at the five designs, against a pass at 4x), and a colder minimum widens the bond's thermal-mismatch swing only once the cold side governs (below about −49 °C at the default; at −60 °C the peak shear rises from 11.68 to 13.48 MPa, still below the lap-shear strength). The joint-neutral "shaft-joint and gearbox loads" is controller-mandated and true, and no probe brings either bond check near red.
- `HOT_MIN_GUIDE.why` and sentences 0 and 1 say nearly the same thing without contradiction; the note is also reached from the cold-high row and Start here, so it must stand alone.

### Task review fix round 1 (2026-10-15): sentence 0 names both edges of the band

- Why: the task code review (`task-14-review.md`, M-9) found that sentence 0 named neither edge
  of the band. Spec 3.1 says sentence 0 answers the question of the row that links to the note,
  and three rows link here (the hot-low, hot-minimum and cold-high rows); only sentences 1 and 2
  named the edges. The rounds above judged sentences 0 to 2 together and rejected S-6 on its
  physics, not on this rule.
- sentences[0]:
  - Old: "Pull-out is a nominal estimate: production scatter makes real couplings weaker or
    stronger, and magnets weaken as they warm."
  - New: "Hot-low is the weakest coupling expected at the operating temperature, and cold-high
    the strongest at the minimum temperature, allowing for production scatter." (22 words; no
    symbol, placeholder, ϑ, U+207B or ∝; the caps and plain-words tests pass.)
- How the text was chosen (the implementer's wording; the re-review below judges it):
  - It does not take S-6's "made weaker by heat". Hot-low is the pull-out at the operating
    temperature less the allowance, with no further heat loss (2.688 × 0.85 = 2.285 N·m at the
    default), so the sentence names the temperature each edge uses and nothing more.
  - It keeps "production scatter", which glosses sentences[1]'s "variation allowance" at its
    first use (round 1's form check relied on it).
  - "Expected" keeps T-O (cold-high is an expectation, not a bound). "At the operating
    temperature" keeps T-E (not the hot-day start, which watch_out[0] spells out).
  - The task review's own example ("Hot-low and cold-high are the edges of the torque band: the
    weakest coupling expected at the operating temperature and the strongest at the minimum.")
    was not taken verbatim, because it drops "production scatter", the gloss above.
  - Dropped from the old sentence: "a nominal estimate" and "magnets weaken as they warm". The
    second is `br_temperature`'s subject, the Start here step before this one.
- Engine, at the five probe designs: hot-low lies below cold-high (2.285 < 3.823, 1.673 < 2.799,
  1.443 < 2.414, 0.1760 < 0.3381 and 2.158 < 3.611 N·m; round 1's numbers).
- Test: `the_torque_band_note_s_sentences_hold_at_every_probe_design` (`tests/explain.rs`) also
  asserts that sentence 0 names "hot-low", "cold-high" and "production scatter" (any case), and
  that hot-low lies below cold-high at every probe design. It failed on the old text ("sentence 0
  names hot-low") and passes on the new.
- The dump for the re-review is `magcoupling-rs/target/notes-review-T14-fix1.txt` (both notes,
  five designs; git-ignored).
- A text re-review of this field follows (controller ruling 7). Its report goes below as
  `### Re-review: code-review fix round 1`.

### Re-review: code-review fix round 1

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `332eda1`
  (`task-14-textreview-1.md`). This is the text re-review of the code-review fix round 1 above
  (controller ruling 7). It re-checked both changed fields together: this note's sentences[0]
  and `br_temperature` change[3].
- Scope here: sentences[0], and the fields whose meaning depends on it: sentences[1] and [2],
  watch_out[0], and `HOT_MIN_GUIDE.why`, which the hot-minimum row shows beside the link. The
  dump (`NOTES_FOR_REVIEW=hot_low_cold_high,br_temperature`, written to
  `magcoupling-rs/target/rereview-T14-1.txt`) draws the sentence identically at the five probe
  designs; it holds no placeholder.
- Verdict: PASS, no must-fix. Three optional should-fix items, all applied with no on-screen
  text change: SF-1 (a test of this field) below; SF-2 (a test) and SF-3 (record wording) under
  `## br_temperature`. The report's part on the tests of both notes is there too.

The reviewer's report for the record on this field follows unchanged.

- **Engine.** Hot-low = T_op·(1 − v) and cold-high = T_20·th2(T_min)·(1 + v) (`metal_design.rs:412-414`).
  - "Hot-low is the weakest coupling expected at the operating temperature" is that identity in words: the nominal at the operating input, less the allowance.
  - "cold-high the strongest at the minimum temperature" is the nominal at the minimum-temperature input, plus the allowance.
  - "Allowing for production scatter" names what the allowance covers. Its display label is "Production variation allowance"; sources[0] says "symmetric; an engineering allowance, not measured".
- **The five probe designs** (hot-low / cold-high, N·m): 2.2852 / 3.8233, 1.6731 / 2.7992, 1.4428 / 2.4138, 0.1760 / 0.3381, 2.1584 / 3.6111.
  - These agree with the dump and with the record's fix-round numbers.
  - The new assertion (hot-low below cold-high) passes.
- **Edge cases.**
  - Operating temperature below the minimum (op −40 °C, min 20 °C, both in range; default allowance): hot-low 2.826 against cold-high 3.327 N·m.
  - The sentence ties each edge to its own temperature, so it stays literally true even where the two temperatures swap. It never says "hot" or "cold" in absolute terms.
  - With v = 0 both edges are the nominal at their temperatures, still "expected".
- **Traps.**
  - **T-E** (hot-low means the operating input, not the hot-day start or the limit): the sentence says "at the operating temperature", and watch_out[0] spells out the hot-day contrast.
  - **T-O** (cold-high is an expectation, not a bound): "expected" governs both clauses through the ellipsis ("cold-high [is] the strongest [coupling expected]"), and sentence 2 repeats it.
  - **T-F** (the nominal hot margin): no margin is mentioned.
  - **Slice-1** ("pull-out torque is guaranteed"): "expected" carries no guarantee.
  - **Heat double count:** S-6's "made weaker by heat" is gone. The sentence names the temperature each edge uses and nothing more. Hot-low at the default is exactly 2.688 × 0.85 = 2.285 N·m.
  - **Global:** no clamp wording (ruling 11), and no claim about the gearbox (ruling 8).
  - No other trap applies.
- **Spec 3.1, sentence 0.**
  - No symbol, placeholder, ϑ, U+207B or ∝. "Hot-low" and "cold-high" are the rows' names, not symbols. The sentence is drawn as one label.
  - It answers the hot-low row and the cold-high row outright.
  - The hot-minimum row's question is whether hot-low reaches the requirement. Sentence 0 says what hot-low is, and sentence 1, right after it, says "it must reach the requirement". The check guide's own `why`, shown with the row, ends "and must reach the requirement".
  - Considered and not proposed: a 25-word variant that also names the requirement. Every form I tried either detaches "allowing for production scatter" from hot-low or shortens "the minimum temperature" to "the minimum". The second reads as the minimum torque, next to "requirement".
- **Form.** 22 words by `word_count`, one sentence. The caps and plain-words tests pass.
- **Considered, no change:**
  - **"Hot-low is the weakest coupling".** Hot-low is a torque, so this is metonymy. It reads naturally to an engineer: the row is labelled "Hot-low torque", and sentence 1 gives the formula. A 24-word "the pull-out of the weakest coupling ...; cold-high, of the strongest ..." is more literal but clumsier, so I do not propose it.
  - **Overlap with sentences 1 and 2.** Sentences 0, 1 and 2 now restate the two temperatures. Sentence 0 is the plain answer drawn as one label, and sentences 1 and 2 add the live values and the formulas. Spec 3.1 asks for this structure, and nothing contradicts.
  - **The Y30 probe.** Its cold check reads "Below the cold demagnetization limit" (cold limit 335.7 °C). That comes from the stored 3D reverse fields of the NdFeB example design (T-N). Every torque in the calculator is the reversible model's, and the temperature verdict flags this. `br_temperature` watch_out[1] and `ferrite_cold_demag` cover the irreversible loss. The field does not change this, and it is not this note's job.

- SF-1, applied by the implementer (session model), 2026-10-15:
  `the_torque_band_note_s_sentences_hold_at_every_probe_design` (`tests/explain.rs`) also
  asserts that sentence 0 names "operating temperature", "minimum temperature" and "expected"
  (any case). These are the words that keep it clear of T-E (hot-low reads the operating input,
  not the hot-day start) and T-O (cold-high is an expectation, not a bound). With "expected"
  dropped from sentence 0 the test fails ("sentence 0 names expected"); the shipped text passes.

## br_temperature

Task T14. Rewrite of "Remanence against temperature, and torque ∝ Br²" as "Heat weakens magnets:
torque against temperature", Start here step 7; it now also owns the three coefficient rows
(`model.alpha_br_per_C`, `model.inner_alpha_br_per_C`, `model.outer_alpha_br_per_C`). The old
watch-out "Each ring keeps its own coefficient", false for library parts, is gone: only a ring
with custom dimensions and a picked grade uses its grade's coefficient. Reviewer: physics
reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice plan fixes for the
note's `review` field; all three rounds ran on 2026-10-15).

Rounds: 3. Verdict: PASS. Note: "Heat weakens magnets: torque against temperature" (plan T14).

Verdict by round: 1 MUST-FIX (2 items); 2 MUST-FIX (1); 3 PASS. Each round's full answer is
`task-14-inreview-r<N>.md`, as under `## hot_low_cold_high`; the two notes were reviewed
together. Rounds 2 and 3 re-checked only the changed fields and the fields whose meaning depends
on them.

The text reviewed in round 1 already carried A-22 (change[1], the coefficient input: "a value
nearer zero changes torque less with temperature, hot and cold. Take it from the datasheet;
rings with a picked grade ignore it.", `Step::Raise` with no aim under A-1's model-assumption
exception) and A-13's plain-words sources. Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump; the `br_temperature` literal;
  - `model.rs:412-426` (`alpha_br`: the grade's α only in the grade mode, else the shared input) and `:445-475` (`resolve_magnets`: library parts are never in the grade mode; a manual ring with no grade uses the input);
  - `model.rs:663-674` (`br_factor`, `ring_pair_factor`), `:978-979` (Br at the operating temperature) and `:1005` (pull-out at 20 °C);
  - `temperature.rs:918, 1336-1350, 1380-1410` (hot-day start, hot-day torque, the margins);
  - `assumptions.rs:88-94`; `grades.rs` (all 17 grades: family, Br, α, rating).
- Numbers reproduced from `compute_all`, in design order:
  - pull-out at 50.00 °C / at 20 °C: 2.688 / 2.893, 1.968 / 2.118, 1.697 / 1.827, 0.2071 / 0.2344, 2.539 / 2.732 N·m (s3, as the dump shows).
  - α in use, inner / outer: −0.001200 / −0.001200 1/°C at the four library designs, −0.002000 / −0.002000 at Y30 (w0); the mixed design renders −0.002000 / −0.001200.
  - Per +1 °C from 50 °C: −0.231 % of the 20 °C pull-out (−0.249 % of the current one) against α_i + α_o = −0.240 %; at Y30, −0.376 % (−0.425 %) against −0.400 %.
  - Pull-out = 20 °C pull-out × th_i·th_o(T_op), within 2.2e-16 at all five.
- Levers:
  - Operating temperature +0.5 °C: the pull-out and hot-low fall at all five (default 2.688476 to 2.685131); the 20 °C pull-out stays to 1e-12. Also at 0 °C (3.033556 to 3.030002) and 120 °C (2.240358 to 2.237304).
  - Coefficient +1e-5 (nearer zero):
    - library designs: the 50 °C pull-out rises and the cold torque falls (default 2.688476 to 2.690150; 3.324618 to 3.320897);
    - Y30: nothing moves;
    - mixed design: only the outer ring's α moves;
    - at 0 °C both the 0 °C pull-out and the cold torque fall, so "less with temperature, hot and cold" holds;
    - `f_cal` is unchanged at the 304 design.
  - Grade (Choose), both rings on custom dimensions, pull-out at 20 / 50 / 120 °C:
    - N42SH 2.893 / 2.688 / 2.240;
    - SmCo_2_17_30 2.034 / 1.991 / 1.894 (kept 0.979 at 50 °C against 0.929);
    - SmCo_2_17_26 1.712 / 1.676 / 1.594; SmCo_1_5_20 1.237 / 1.204 / 1.128;
    - Y30 0.234 / 0.207 / 0.150; bonded NdFeB 0.723 / 0.664 / 0.535.
    - From the Y30 design, SmCo_2_17_30 on both rings raises the 20 °C pull-out from 0.2344 to 2.034 N·m and the kept share from 0.884 to 0.979. That is MF-1.
  - Hot-day ambient +0.5 °C, at all five:
    - the hot-day torque falls (default 2.589014 to 2.585731 N·m);
    - the hot-day, magnet-limit and adhesive margins each fall by 0.5 °C;
    - the pull-out at the operating temperature is unchanged.
- Extra probes:
  - Required torque 2.0 N·m: the note is unchanged and true.
  - Operating temperature 0 °C: s3 renders "3.034 N·m at 0 °C and 2.893 N·m at 20 °C", and every sentence holds.
  - Operating temperature 120 °C: s3 renders "2.240 N·m at 120.0 °C".
  - Mixed design: w0 renders "inner -0.002000 1/°C, outer -0.001200 1/°C", which matches the engine.
  - Gearbox rating 1.0 N·m: no effect.
  - Axial length 1.0 mm: s3 renders "no prediction" for both pull-outs; the temperature and the α values render; no negative torque.
- Traps checked:
  - "each ring uses its own α": not written; w0 says only a ring with custom dimensions and a picked grade does;
  - "a hotter-rated or lower-coefficient grade is free": the cost is named ("starts weaker"); MF-1 keeps it and makes it true design by design;
  - "the stored 3D fields follow the design": not written;
  - no torque is called guaranteed;
  - no shaft-joint method is chosen or sized, and nothing says the coupling protects the gearbox;
  - "clamp" in any field: none.
- Form:
  - title 6 words, plain, no ∝ (the old title's ∝ is gone);
  - sentences 16, 22, 21, 17 (rendered 20), 21; after MF-2 sentence 2 has 24;
  - changes 19, 24, 24, 22; after MF-1 change 2 has 24;
  - watch-outs 27 (rendered 29), 24;
  - sources 10, 20, 13;
  - all within the caps; guards empty; sentence 0 plain.
  - Br and α are glossed at first use (s1). "Hot-low" and "cold-high" are dashboard row names, glossed at step 8.
- Seeds: S-5 accepted as MF-2.
- Must-fix items, round 1:
  - MF-1, change[2]: "starts weaker" was false at the Y30 design; the replacement text and its claims test are above.
  - MF-2, sentences[2]: S-5 verbatim.
  - Pending the round-2 re-check.
- Should-fix offered: SF-2 (change[3], "the temperature margins"), SF-4 (an optional test of the per-degree share).
- Verdict for this note, round 1: MUST_FIX.

Round 2:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 2. The scope is the fields round 1 changed: change[2] (MF-1) and sentences[2] (MF-2, S-5), plus the sentences that depend on them.
- Read: the dump; the `br_temperature` literal (`notes.rs:464-523`); `model.rs:412-426, 445-475, 647-649, 663-674, 978-1005`; `grades.rs` (all 17 grades); the grade claims test (`tests/explain.rs:2237-2317`); the round-1 changes file; spec section 8, slice-1 spec section 8, phys-new-notes section 2; A-13, A-22, A-24, A-28/A-28b, ruling 11.
- Sentences[2] (S-5) is confirmed true.
  - Torque ∝ Br_i·Br_o is exact in the model: within 2.7e-16 across temperature, and within 6.7e-16 across the 17 grades at fixed geometry, at all five designs.
  - Per °C at the designs' 50 °C, as the share of the 20 °C pull-out / the share of the current pull-out against α_i + α_o:
    - NdFeB designs: −0.231 / −0.249 against −0.240 %/°C;
    - Y30: −0.376 / −0.425 against −0.400 %/°C;
    - mixed design: −0.306 / −0.337 against −0.320 %/°C.
  - α_i + α_o always lies between the two readings. The widest gap is Y30 at 120 °C: −0.32 / −0.50 against −0.40.
  - Verdict on this field: PASS.
- Change[2] (MF-1) is confirmed fixed on "starts weaker".
  - Every SmCo grade starts (at 20 °C) below every sintered NdFeB grade at all five designs, on one ring or both, at 0, 50 and 120 °C, and on the mixed design. The smallest ratio is 1.0459.
  - SmCo is the nearest to zero (≤ 0.045 against ≥ 0.12 %/°C).
  - From the Y30 design, SmCo starts stronger: 2.034 against 0.2344 N·m on both rings.
- New must-fix MF-3 (change[2]): "loses less when hot" changes to "loses a smaller share when hot", and ", per ring" is dropped to fit the cap.
  - Why: the nearer-zero grade loses more N·m in 30 to 32 of 69 grade pairs at every warm probe.
    - On the Y30 design, SmCo on the inner ring loses 0.0482 N·m against Y30's 0.0273 while keeping 0.930 against 0.884.
    - On both rings it loses 0.0425 N·m and keeps 0.979.
  - Read as a share, the claim holds in 69 of 69 pairs.
  - New text, 24 words: "a grade whose coefficient is nearer zero loses a smaller share when hot; SmCo, the nearest, starts weaker than sintered NdFeB. Custom dimensions only."
  - The test pins "smaller share" and the N·m counterexample (0.0425 against 0.0273 N·m).
  - Pending the round-3 re-check.
- Form of both fields: 24 words each; guards empty; no "clamp"; no new glyph; within the caps and plain-words tests.
- Seeds: S-5 accepted (applied); S-6 and S-7 rejected (fields unchanged, round-1 reasons stand).
- Verdict for this note, round 2: MUST_FIX (MF-3).

Round 3:

- **Reviewer:** physics reviewer (session model), fresh context, 2026-10-15; round 3.
- **Scope:** the field changed in round 2, change[2] (MF-3), and its claims test, plus the sentences that depend on it: sentences[1] and [2], change[1] and watch-out[0].
- **Read:**
  - the dump; the `br_temperature` literal (`notes.rs:464-523`);
  - `model.rs:416-426` (each ring's α), `:445-475` (the grade mode), `:663-674` (`br_factor`, `ring_pair_factor`) and `:1005` (the 20 °C pull-out);
  - `grades.rs` (all 17 grades: family, Br, α, rating);
  - `calibration.rs:65-68` (the coefficient input, −0.0025 to 0);
  - the grade claims test (`tests/explain.rs:2237-2335`);
  - the round-2 changes file;
  - A-13, A-22, A-24, A-28/A-28b and ruling 11.
- **Change[2] is confirmed true:** "a grade whose coefficient is nearer zero loses a smaller share when hot; SmCo, the nearest, starts weaker than sintered NdFeB. Custom dimensions only."
  - **"A smaller share when hot"** is an identity of the model. The kept share is (1 + α_i(T − 20))(1 + α_o(T − 20)), and every α is ≤ 0.
    - Probed: 69 grade pairs; 20.5 to 150 °C; one ring and both; five probe designs, the mixed design and an ungraded manual ring. That is 6762 comparisons and 0 failures.
    - The hot-day and peak torques, as shares, also gave 0 failures.
    - Against the shared input, from −0.0025 to 0, there were 0 failures.
  - **Read as N·m, the claim would fail** in 30 to 33 of 69 pairs at every design. On the Y30 design at 50 °C, SmCo_2_17_30 on the inner ring loses 0.0482 N·m against Y30's 0.0273, while keeping 0.930 against 0.884. On both rings it loses 0.0425 N·m and keeps 0.979.
  - **"SmCo, the nearest, starts weaker than sintered NdFeB"** is re-confirmed: 504 comparisons at 20 °C, 0 failures, with a smallest ratio of 1.0459. |α| is at most 0.00045 for SmCo, against at least 0.0012 for every other grade.
  - **"Custom dimensions only"** is true (`resolve_magnets`). The per-ring rule stays on screen in the entry's label and in watch-out[0].
- **Form:** 24 words, 2 sentences, lower-case start; guards empty; no "clamp" in any of the note's 15 fields; no typed label, no placeholder, no new glyph.
- **Test:** `a_grade_nearer_zero_loses_a_smaller_share_when_hot_and_smco_starts_below_sintered_ndfeb` pins "smaller share" and the N·m counterexample. It passes on the current tree, run with `--locked` and an outside target dir.
- **Seeds:** S-5 accepted (applied in round 1); S-6 and S-7 rejected (fields unchanged; the reasons of rounds 1 and 2 stand).
- **Should-fix offered:** SF-5 (the README wording at Step 16) and SF-6 (an optional loop of the kept-share assert over the five probe designs).
- **Must-fix items, all rounds:**
  - MF-1 (change[2], "starts weaker" tied to "nearer zero"): fixed in round 1, confirmed in round 2.
  - MF-2 (sentences[2], S-5): fixed in round 1, confirmed in round 2.
  - MF-3 (change[2], "loses less" made a share): fixed in round 2, confirmed in round 3.
- **Verdict for this note, round 3: PASS.** Rounds: 3.

Must-fix items applied:
- sentences[2] (round 1, MF-2, seed S-5 accepted verbatim: "about α_i + α_o" left the unit open beside the watch-out's α printed in 1/°C, so the change could be read as an absolute torque; "a share of" matches sentence 1's definition): "Torque is proportional to inner Br times outer Br, so each °C of warming changes it by about α_i + α_o." -> "Torque is proportional to inner Br times outer Br, so each °C of warming changes it by a share of about α_i + α_o."
- change[2] (round 1, MF-1; input `coupling.magnets.grade_inner`, `Step::Choose`, no aim, unchanged: "starts weaker" was tied to any grade nearer zero, which is false on the "Y30 ferrite, custom dimensions" design, the one probe where the entry is a live link, since there every other grade is nearer zero and starts stronger): "a grade whose coefficient is nearer zero, such as SmCo, loses less when hot but starts weaker; custom dimensions only, each ring its own." -> "a grade whose coefficient is nearer zero loses less when hot; SmCo, the nearest, starts weaker than sintered NdFeB. Custom dimensions only, per ring."
- change[2] (round 2, MF-3; input, step and aim unchanged: "loses less when hot" reads as fewer N·m, but the nearer-zero grade loses more N·m in 30 to 33 of the 69 grade pairs at every warm probe, though it always keeps a larger share; ", per ring" dropped for the 25-word cap, the entry's label and watch_out[0] carrying the per-ring rule): "a grade whose coefficient is nearer zero loses less when hot; SmCo, the nearest, starts weaker than sintered NdFeB. Custom dimensions only, per ring." -> "a grade whose coefficient is nearer zero loses a smaller share when hot; SmCo, the nearest, starts weaker than sintered NdFeB. Custom dimensions only."
- The claims test follows MF-1 and MF-3: `a_grade_nearer_zero_loses_a_smaller_share_when_hot_and_smco_starts_below_sintered_ndfeb` in `tests/explain.rs` (first `a_grade_nearer_zero_loses_less_when_hot_but_starts_weaker`) pins, computed and never typed: the grade table (every SmCo grade nearest zero, and below every sintered NdFeB grade's Br); SmCo_2_17_30 against N42SH on the default design (a larger kept share, a lower 20 °C pull-out); SmCo_2_17_30 against Y30 on the Y30 design (a larger kept share, more N·m lost, a higher 20 °C pull-out); and the entry's shape ("smaller share", "weaker than sintered NdFeB", no clause that ties "weaker" to "nearer zero").

Seeds (A-24):
- S-5 (sentences[2]: "Torque is proportional to inner Br times outer Br, so each °C of warming changes it by a share of about α_i + α_o."): accepted in round 1 and applied verbatim as MF-2; rounds 2 and 3 confirm it. Reason: "a share of" makes the unit explicit and matches sentence 1's definition; torque ∝ Br_i·Br_o is exact in the model, and α_i + α_o lies between the two readings of the share (of the 20 °C pull-out and of the current one) at every probe.

Should-fix offered (open for the controller; no on-screen text changed after the round-3 PASS
until the task review fix round 1 below, which applied SF-2):
- SF-2 (round 1; rounds 2 and 3 keep it offered), change[3] (`temperature.duty.hot_ambient_C`, Raise; aim unchanged), 21 words: "a hotter day lowers the hot-day torque and the temperature margins; the hot-day start is this ambient plus the driving rise." Why: one step lowers all three margins the temperature verdict reads (the hot-day, magnet-limit and adhesive margins, each by 0.5 °C at all five designs), not only the hot-day one; the shipped help for this input already says "temperature margins". Its test: also assert that `temperature.magnet_life.margin_limit_C` and `temperature.adhesive_life.margin_C` fall. (Applied verbatim by the task review fix round 1 below.)
- SF-4 (round 1, an optional test): at each design, one degree up from the operating temperature changes the pull-out, as a share of the 20 °C pull-out, by within 10 % of α_i + α_o (the worst case, 6.4 %, is at Y30).
- SF-5 (rounds 2 and 3; the README, not on-screen text): applied with this task's docs. The README's `tests/explain.rs` row says "a grade with a coefficient nearer zero loses a smaller share of its torque when hot (not always fewer N·m), and SmCo, the nearest, starts below sintered NdFeB (on the Y30 design it starts above Y30)", in place of the plan's "keeps more of its torque hot but starts weaker", which restated the claim MF-1 and MF-3 remove.
- SF-6 (round 3, an optional test): loop the kept-share assert over the five probe designs (SmCo_2_17_30 on both rings against each design as it is), so the claims test's "design by design" is tested at every design, not two.

Considered, no change (the reviewer's observations):
- "Custom" against "manual" dimensions: the grade input's label says "(manual dimensions)" and the note "custom dimensions"; the part picker's blank choice is named "Custom dimensions" on screen, so both point to what the user picks (round 1).
- The rendered α, "-0.001200 1/°C" (four significant digits from the dashboard formatter), is correct and readable to an engineer (round 1).
- A hotter operating temperature approaches the magnet temperature check; change[0] is an honest-input entry ("set it to the magnet temperature while running") and names its torque cost (round 1).
- Sentence 2's "about" far from 20 °C: for ferrite at 120 °C the two readings are −0.32 and −0.50 %/°C against −0.40 %/°C. The first-order rule is what an engineer needs, sentence 3 shows the real values, and the cap leaves no room for "near 20 °C" (round 2).
- Only the Y30 design draws change[2] as a live link; the other four draw "(set by the Inner magnet part pick)", which is correct because library parts are never in the grade mode (round 2).
- A grade nearer zero can carry a lower heat rating (from the Y30 design every sintered NdFeB grade is nearer zero, and N35 to N52 are rated 80 °C). The entry's claim concerns the reversible share, watch_out[1] names the magnet's limits, and the dashboard's magnet temperature check flags an operating temperature over the rating (round 3).

### Task review fix round 1 (2026-10-15): the hotter-day entry names the temperature margins

- Why: the task code review (`task-14-review.md`, M-8) applied spec 3.1's trade-off rule: change[3]
  named only the hot-day margin, while one step of the hot-day ambient also lowers the
  magnet-limit and adhesive margins, both read by the temperature verdict. SF-2 above is the fix.
- change[3] (`temperature.duty.hot_ambient_C`, `Step::Raise`, aim
  `temperature.magnet_life.torque_hot_day_Nm` down; all unchanged):
  - Old: "a hotter day lowers the hot-day torque and the hot-day temperature margin; the hot-day
    start is this ambient plus the driving rise."
  - New (SF-2 verbatim, 21 words): "a hotter day lowers the hot-day torque and the temperature
    margins; the hot-day start is this ambient plus the driving rise."
- Engine, one slider step (+0.5 °C) at the five probe designs, from `compute_all`:
  - the hot-day, magnet-limit and adhesive margins each fall by 0.5 °C (default 28.06 to 27.56,
    27.09 to 26.59 and 54.04 to 53.54 °C), since the hot-day start and the peak follow the
    ambient;
  - the skipping-onset margin falls by 0.5 °C at the four NdFeB designs and stays infinite at the
    Y30 design (ferrite has no hot onset; the readout draws "no prediction");
  - the service margin (the governing limit less the operating temperature) and the cure margin
    read no ambient and stay exactly (43.06 and 121.36 °C at the default; 70 °C and infinite at
    Y30).
  - So every temperature margin (in °C) that moves falls, and none rises. The adhesive's hot
    fatigue margin, a strength ratio and not a temperature margin, rises slightly (4.780 to
    4.786 at the default): the model holds the hot strength fraction fixed while the peak torque
    falls. (Text re-review 1's SF-3.)
- For the re-review: "the temperature margins" covers every margin that moves; two (service, cure)
  do not move. A narrower 24-word wording names the three the verdict reads: "a hotter day lowers
  the hot-day torque and the hot-day, magnet-limit and adhesive margins; the hot-day start is this
  ambient plus the driving rise." It was not applied: SF-2 is the reviewed text, and the input's
  shipped help already says "temperature margins".
- Test: `the_temperature_note_s_levers_hold_at_every_probe_design` (`tests/explain.rs`) asserts
  that change[3] says "the temperature margins", and at each probe design that the hot-day,
  magnet-limit and adhesive margins fall, the skipping-onset margin falls wherever it is finite
  (at one design or more) and stays infinite elsewhere, and the service and cure margins do not
  rise. It failed on the old text ("names the margins it lowers") and passes on the new.
- The dump for the re-review is `magcoupling-rs/target/notes-review-T14-fix1.txt`, as under
  `## hot_low_cold_high`.
- A text re-review of this field follows (controller ruling 7). Its report goes below as
  `### Re-review: code-review fix round 1`.

### Re-review: code-review fix round 1

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `332eda1`
  (`task-14-textreview-1.md`). This is the text re-review of the code-review fix round 1 above
  (controller ruling 7), run on this field and `hot_low_cold_high` sentences[0] together.
- Scope here: change[3] (input, step and aim unchanged), and the fields whose meaning depends on
  it: sentences[4] (the hot-day torque's temperature, the ambient plus the driving rise) and the
  input's shipped help ("Higher lowers the hot-day pull-out and temperature margins, not hot-low
  torque."). The dump, as under `## hot_low_cold_high`, draws the entry identically at the five
  probe designs, and the entry's aim moves as it says at all five (default 2.58901 to 2.58573
  N·m).
- Verdict: PASS, no must-fix. Three optional should-fix items, all applied with no on-screen
  text change: SF-2 (a test) and SF-3 (record wording) below; SF-1 (a test of
  `hot_low_cold_high` sentences[0]) under that note.

The reviewer's report for the record on this field, then its part on the tests of both notes,
follow unchanged.

- **Engine.**
  - T0 = hot ambient + driving rise (`temperature.rs:918`), exactly the second clause.
  - The hot-day torque is T_20·thf(T0), so it falls with T0 for any α < 0.
  - The fault temperature, and so the slip-fault peak, are T0 plus terms that do not read the ambient (`:1282`, `:1335`). The peak therefore follows the ambient degree for degree.
- **One slider step (+0.5 °C), default design:**

  | Margin (°C) | Before | After | Reads the ambient? |
  |---|---|---|---|
  | Hot-day (governing − T0), dashboard row and verdict | 28.056 | 27.556 | yes |
  | Magnet design limit − peak, verdict | 27.091 | 26.591 | yes |
  | Skipping onset − peak | 37.091 | 36.591 | yes |
  | Adhesive design limit − peak, verdict | 54.036 | 53.536 | yes |
  | Service (governing − operating) | 43.056 | 43.056 | no |
  | Cure (single-ring onset − cure), verdict | 121.356 | 121.356 | no |

  - The other four designs behave the same way: every margin that reads the ambient falls by exactly 0.5 °C, and the service and cure margins stay put.
  - At Y30 the skipping onset is infinite (ferrite has no hot onset), so its margin stays infinite. The hot-day, magnet-limit and adhesive margins fall from 55.000 to 54.500, 184.021 to 183.521 and 54.021 to 53.521 °C. The service and cure margins (70 °C and infinite) do not move.
  - So "the temperature margins" covers every margin that moves, and none rises. On the dashboard (`DASHBOARD`), the only temperature margin is the hot-day one, and it falls.
- **The trade-off rule (spec 3.1).**
  - A hotter day pushes these toward red: the hot-day torque's check against the requirement ("Meets it nominally" can become "Below it"), and the verdict's hot-day, magnet-limit and adhesive terms. Both are named.
  - Nothing else that has a verdict moves toward red:
    - the cold terms read only the minimum temperature;
    - the daily bond-shear screen reads the daily swing;
    - the hot fatigue screen moves away from red (SF-3).
  - The time to the limit and the critical drag also fall, but neither carries a verdict.
- **Wording.**
  - Keeping SF-2's "the temperature margins" over the record's narrower 24-word alternative ("the hot-day, magnet-limit and adhesive margins") is right.
    - The broad phrase matches the input's shipped help word for word ("temperature margins"), so one input is described one way.
    - It is true of every margin that moves.
    - The narrow form would set "the hot-day margin" beside "magnet-limit and adhesive margins", although the hot-day margin is itself measured against the magnet or adhesive limit. That invites a misreading.
  - "This ambient" reads cleanly after the drawn label ("Raise Hot-day ambient temperature: ..."). "The driving rise" is the short name sentence 4 of the same note already uses.
- **Traps.**
  - None applies: no grade, no α per ring, no steady-temperature or mass claim, no slip-duration claim, no clamp or gearbox wording.
  - **T-E:** the entry concerns the hot-day torque and leaves hot-low alone. One step leaves `model.pullout_Nm` exact, as the levers test pins, and watch-out 1's test pins hot-low exact.
- **Form.** 21 words by `word_count`, one sentence. The caps and plain-words tests pass. The entry's input, step and aim are unchanged, and the aim test passes at all five designs.
- **Considered, no change:**
  - **α = 0** (the coefficient slider's top end, library rings): one step leaves the hot-day torque unchanged (2.893025 N·m both ways), so "lowers the hot-day torque" degenerates there. This clause predates this round and was reviewed in rounds 1 to 3. At α = 0 nothing in the note depends on temperature. Every probe design has α < 0.
  - **"ambient plus the driving rise" appears twice** in the note (sentence 4 and this entry). This also predates the round. In the entry it tells the user what the input they are about to move feeds, so it earns its place. SF-2 adds the test that M-6 found missing.

**Tests** (both notes):

- The sentences test pins sentence 0's three names and hot-low below cold-high at the five designs.
- The levers test pins "the temperature margins" and checks:
  - the three verdict margins fall at every design;
  - the skipping-onset margin falls wherever it is finite (with at least one such design);
  - the service and cure margins do not rise.
- I re-derived every one of these directions from the engine code above and confirmed them numerically at the five designs. SF-1 and SF-2 would make the tests cover the trap-critical words and the hot-day-start identity.

- SF-2, applied by the implementer (session model), 2026-10-15:
  `the_temperature_note_s_levers_hold_at_every_probe_design` (`tests/explain.rs`), entry 4,
  also asserts at each probe design that `temperature.summary.hot_day_start_C` equals the hot-day
  ambient plus the driving rise, exactly (`assert_eq!`, as the engine sums them), the second
  clause of change[3]. With the engine's hot-day start offset by 1e-9 °C the test fails
  ("default: the hot-day start"); the shipped engine passes.
- SF-3, applied by the implementer (session model), 2026-10-15: in the "Task review fix round 1"
  subsection above, "So every temperature margin that moves falls, and none rises." now reads as
  the reviewer gives it, so "none rises" is not read as covering the adhesive's hot fatigue
  margin, a strength ratio. The figures were re-run from `compute_all`: one step up raises it at
  all five probe designs (4.7795 to 4.7856 at the default, 5.2230 to 5.2296, 7.5579 to 7.5675,
  63.881 to 64.022 and 9.3578 to 9.3698). The note says nothing about this margin, so the screen
  is unaffected.

## when_slipping

Task T15. New note "When the coupling slips", Start here step 9 (after the torque band and
before slip heating, in the full order and in the order shown while the shaft-clamp section is
paused). It owns the gearbox ripple (`model.gearbox_input_ripple_Nm`) and the slip frequency
(`metal.slip_freq_Hz`), and the dashboard's gearbox ripple row now links it as its Why link (it
linked the pull-out note before). Reviewer: physics reviewer (session model), fresh context.
Date: 2026-10-14 (the date the slice plan fixes for the note's `review` field; both rounds ran
on 2026-10-15).

Rounds: 2. Verdict: PASS. Note: "When the coupling slips" (new; plan T15).

Verdict by round: 1 MUST-FIX (3 items in this note, 6 across the task's three notes); 2 PASS.
The three notes were reviewed together. Each round's full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-15-inreview-r<N>.md`, and the six
fields that round 1 changed are listed with their old and new text in
`task-15-inreview-r1-changes.md` beside it. Round 2 re-checked only those fields and the fields
whose meaning depends on them.

The text reviewed in round 1 already carried A-13's plain-words sources, and its gearbox
watch-out used live values only, never a typed rating (ruling 8: the coupling sits on the
gearbox output, which carries its torque directly). None of the three notes names a clamp, so
A-28c (ruling 11, the shaft-clamp section paused) changed only the Start here mechanics. Every
round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump `notes-review-T15-r1.txt`; the `when_slipping` literal (`notes.rs:673-711`);
  - `records/torque.rs:21-23` (k = n N/(2 R_g): one electrical cycle per pole pair), `:65` (the pull-out angle is the peak of the summed harmonics) and `:73` (T_pull = T_2D·f_end·f_cal);
  - `model.rs:1155-1156` and `records/dashboard.rs:29-30` (ripple = T_pull/(ratio·efficiency));
  - `records/slip_heating.rs:279` (slip frequency = N/2 × rpm/60);
  - `records/torque.rs:209-210, 229-231` (cold-high and its gearbox-input value);
  - `records/slip_heating.rs:66-72, 137-147` (duty, total loss, drag = P/ω_s);
  - `temperature.rs:1199-1202` (a measured drag replaces the loss used and the high case);
  - `temperature.rs:1282` (fault temperature) and `:1335` (peak);
  - `temperature.rs:105-114` (the stored reverse fields; like poles facing "once per pole pass while skipping") and `:711-731, 817-848` (onsets from those fields);
  - `metal_design.rs:225, 230` (re-engagement and slip detection open; no re-lock model);
  - `notes.rs` START_HERE_ALL (step 9); `guidance.rs` ROW_LEARN (the ripple row);
  - spec 3.1-3.3 and 8; slice-1 spec 8; phys-new-notes 2 (T-G, T-H, T-M, T-N, T-O) and 3.3; phys-notes-b 6; amendments ruling 8, ruling 11, A-13, A-24 and A-28/A-28c.
- Numbers reproduced from `compute_all`, in design order (default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius); every rendered value matches the dump:
  - drag (`temperature.slip_loss.drag_Nm`): 0.01314, 0.01370, 0.004412, 0.01314, 0.04668 N·m. Each equals total loss / slip speed. As a share of the pull-out: 0.49, 0.70, 0.26, 6.34, 1.84 %.
  - ripple at the gearbox input (`model.gearbox_input_ripple_Nm`): 0.5660, 0.4144, 0.3573, 0.04359, 0.5346 N·m. This equals pull-out (2.688, 1.968, 1.697, 0.2071, 2.539 N·m) / (5 × 0.95), exactly.
  - slip frequency (`metal.slip_freq_Hz`): 166.7, 133.3, 166.7, 166.7, 166.7 Hz. This equals N/2 × 2000/60.
  - cold-high at the gearbox input (`metal.cold_high_input_Nm`): 0.8049, 0.5893, 0.5082, 0.07117, 0.7602 N·m (cold-high 3.823, 2.799, 2.414, 0.3381, 3.611 N·m / 4.75). It is above the ripple at all five.
  - onsets at the NdFeB designs: like poles facing 103.06 °C, pull-out 113.35 °C, aligned 170.16 °C. So like poles facing is the hardest case, and the magnet limit is 93.06 °C. At Y30 the onsets are infinite on the hot side and the adhesive governs at 120 °C.
- Change directions confirmed (one slider step, `step_input`, at all five designs; the dump's aim lines all read "moves as it says"):
  - Lower fault trip time: the fault temperature falls (default 65.2001 to 65.1901 °C). It also falls with a 140 °C hot-day start, and with a 60 s trip at 6000 rpm (99.152 to 99.101 °C).
  - Lower slip speed: the total loss falls (default 2.7513 to 2.7287 W; at 6000 rpm 17.365 to 17.315 W). The swing frequency falls with it.
- Extra probes:
  - Measured drag 0.02 N·m: the drag and the loss stay the estimate (0.01314 N·m, 2.751 W), so "an estimated" stays true.
  - A 60 s trip at 6000 rpm: 500 Hz (400 Hz at 8 magnets), every aim holds.
  - Axial length 1.0 mm (f_end −0.3214): the engine's ripple (−0.01598) and cold-high input (−0.02273 N·m) render "no prediction". `a_value_computed_from_an_out_of_range_pull_out_reads_no_prediction` passes. The drag (0.004066 N·m) and the frequency render; neither depends on f_end. No negative torque is drawn.
- Traps checked:
  - "the rings re-lock below pull-out": sentence 3 says the opposite; MF-1 removes the unmodelled "near zero";
  - "pull-out is guaranteed": not written;
  - "the coupling protects the gearbox": not written; watch-out[0] says nothing checks it, and MF-2 stops the input value reading as the gearbox's swing;
  - "the stored 3D fields follow the design": not written; MF-3 makes the fields, not the results, what stays fixed;
  - T-H: the ripple row is the nominal swing at the operating temperature, larger when cold (watch-out[0]);
  - T-M: one cycle per pole pair; the average is the drag; re-lock;
  - T-N: stored fields;
  - T-O: "can reach";
  - "more magnets make the coupling stiffer": not written;
  - no shaft-joint method is chosen or sized;
  - "clamp" in any field, sources included: none.
- Form:
  - title 4 words;
  - sentences 22, 23, 23, 24, 23 words raw; after MF-1 and MF-2, 22 and 24;
  - changes 21, 20; watch-outs 25, 26, with 27 after MF-3;
  - every field passes `plain_text_violations` and `audit_ids`;
  - sentence 0 has no symbol and no placeholder;
  - placeholders sit only in "here" or "on this design" clauses;
  - eddy current and drag are glossed at their first use (sentence 2); pole pair is used at step 4 first.
- Seeds and duplicates: S-8 accepted (MF-1). A-27: no trim; the note must stand alone as the ripple row's Why target.
- Must-fix items: MF-1 (sentences[3], S-8), MF-2 (sentences[4], "at its input"), MF-3 (watch_out[1], "those fields"), to apply in round 2. Should-fix offered: SF-1 (sentence 0 names the gearbox), SF-2 (the mounting fact in watch_out[0]).
- Verdict for this note, round 1: MUST_FIX (3 items).

Round 2:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; rounds 1 and 2 (PASS in round 2).
- Read:
  - the dumps `notes-review-T15-r1.txt` and `-r2.txt`; the `when_slipping` literal (`notes.rs:673-711`);
  - `records/torque.rs:21-23, 65, 73` (one cycle per pole pair; pull-out as the peak of the summed curve; T_pull);
  - `model.rs:1155-1156` and `records/dashboard.rs:29-30` (ripple = T_pull/(ratio·efficiency));
  - `records/slip_heating.rs:279` (slip frequency);
  - `records/torque.rs:209-210, 229-231` (cold-high at the gearbox input);
  - `temperature.rs:105-114` (the stored reverse fields; like poles facing "once per pole pass while skipping"), `:236-275` (the stored slip fields), `:817-848` (onsets from the reverse fields, unscaled) and `:1116-1219` (the losses from the stored slip fields; a measured drag replaces the loss used and the high case);
  - `metal_design.rs:225, 230` (re-engagement and slip detection open; no re-lock model);
  - spec 8; slice-1 spec 8; phys-new-notes 2; amendments ruling 8, ruling 11, A-13, A-24, A-27 and A-28/A-28c.
- Numbers reproduced from `compute_all`, in design order (default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius); every rendered value matches the dump:
  - drag (`temperature.slip_loss.drag_Nm`): 0.01314, 0.01370, 0.004412, 0.01314, 0.04668 N·m. Each is under a tenth of the pull-out (most at Y30, 6.3 %, from the example design's fields).
  - ripple at the gearbox input (`model.gearbox_input_ripple_Nm`): 0.5660, 0.4144, 0.3573, 0.04359, 0.5346 N·m. Each equals pull-out (2.688, 1.968, 1.697, 0.2071, 2.539 N·m) / (5 × 0.95) exactly.
  - slip frequency (`metal.slip_freq_Hz`): 166.7, 133.3, 166.7, 166.7, 166.7 Hz. Each equals the pole pairs × 2000 rpm / 60.
  - cold-high at the gearbox input (`metal.cold_high_input_Nm`): 0.8049, 0.5893, 0.5082, 0.07117, 0.7602 N·m. It is above the ripple at all five.
  - The stored slip fields and reverse fields are identical at all five designs, and at a 2.0 mm gap and an 8 mm magnet length. The drag and the magnet limit follow the design (93.06 °C NdFeB, 250 °C Y30).
- Change directions confirmed (the aim test passes; the dump's aim lines all read "moves as it says"):
  - Lower fault trip time: the fault temperature falls at all five designs (default 65.2001 to 65.1901 °C).
  - Lower slip speed: the total loss falls (default 2.7513 to 2.7287 W), and the swing frequency falls with it.
  - Past the end effect (a 1 mm magnet): the ripple and cold-high input render "no prediction" (GUI test).
- Traps checked:
  - "the rings re-lock below pull-out" (sentence 4 says the opposite; no unmodelled threshold);
  - "pull-out is guaranteed"; "the coupling protects the gearbox" (the input is named, and nothing checks it);
  - "the stored 3D fields follow the design" (the fields, not the results, stay fixed);
  - T-H, T-M, T-N, T-O; "more magnets make the coupling stiffer";
  - no shaft-joint method chosen or sized;
  - "clamp" in any field, sources included: none.
- Form:
  - sentences 22, 23, 23 (24 counted), 22, 24 (25 counted); changes 21, 20 (21 counted); watch-outs 25 (26 counted), 27;
  - every field passes `plain_text_violations` and `audit_ids`;
  - sentence 0 has no symbol and no placeholder; placeholders sit only in "here" or "on this design" clauses.
- Seeds: S-8 accepted and applied verbatim.
- Must-fix items and how they were applied (round 1, applied verbatim in round 2, re-reviewed true):
  - sentences[3] drops "near zero" (S-8);
  - sentences[4] names the value as the ripple "at its input";
  - watch_out[1] makes "those fields" the subject that does not follow the design.
- Verdict: PASS.

Must-fix items applied (all in round 1, applied verbatim before round 2, which confirmed each):
- sentences[3] (MF-1, seed S-8 accepted verbatim: "near zero" stated a re-lock threshold the engine does not model, and the re-lock slip speed depends on the inertias on both sides and on the load torque): "The rings re-lock only once the slip speed falls near zero, so slow the drive; a load dropping just below pull-out is not enough." -> "The rings re-lock only once the slip speed falls, so slow the drive; a load dropping just below pull-out is not enough."
- sentences[4] (MF-2: the quoted value is the ripple referred to the gearbox input, the pull-out / (ratio × efficiency), while the gearbox output, where the coupling sits, carries the full swing; unqualified, 0.566 N·m read as the gearbox's own swing beside its 3 N·m output rating, the "coupling protects the gearbox" misreading): "Each swing brings like poles face to face, the hardest case for the magnets, and sends torque ripple to the gearbox, here {model.gearbox_input_ripple_Nm}." -> "Each swing puts like poles face to face, the magnets' hardest case, and sends torque ripple to the gearbox: at its input, here {model.gearbox_input_ripple_Nm}."
- watch_out[1] (MF-3: "they" read most naturally as the drag, the heating and the magnets' limits, which do follow the geometry and the grade; only the stored fields stay fixed): "The drag, the heating and the magnets' limits use fields stored from the example design's 3D model; they do not follow your geometry or magnet grade." -> "The drag, the heating and the magnets' limits use fields stored from the example design's 3D model; those fields do not follow your geometry or magnet grade."

Seeds (A-24):
- S-8 (sentences[3]: "The rings re-lock only once the slip speed falls, so slow the drive; a load dropping just below pull-out is not enough."): accepted in round 1 and applied verbatim as MF-1; round 2 confirms it. Reason: the re-lock slip speed depends on the inertias on both sides and on the load torque, and the engine does not model re-locking (`metal_design.rs:225, 230`), so "near zero" is a heavy-load case, not a general threshold; "falls" carries the trap-safe claim (re-locking is set by the slip speed falling, not by the load dropping below pull-out) and agrees with the note's sources[4] and `pullout_angle`'s reviewed watch-out.

Prose duplicates (A-27): no trim, in rounds 1 and 2. Sentences[1] and [3] overlap
`pullout_angle` sentences[4] and its watch-out by one clause each, but this note is the ripple
row's Why target and Start here step 9: a reader who arrives from the dashboard has not read
step 4, so the note must stand alone. After MF-1 the two notes agree word for word on
re-locking ("only once the slip speed falls"). `pullout_angle` (reviewed in T12) is not wrong,
so no must-fix is named there.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS until the task review fix round 1 below, which applied SF-1):
- SF-1 (round 1; round 2 keeps it offered), sentences[0], 24 words: "Past pull-out the coupling slips: one ring turns against the other, and the torque between them, carried by the gearbox, swings back and forth." Why: the linking row is the gearbox ripple, and the shipped sentence 0 answers when and what but never names the gearbox; the replacement stays free of symbols and placeholders. (Applied verbatim by the task review fix round 1 below.)
- SF-2 (round 1; round 2 keeps it offered; ruling 8), watch_out[0], 29 words: "Nothing checks the gearbox against these swings: its output carries them in full; when cold they can reach the cold-high torque, here {metal.cold_high_input_Nm} at its input; check its rating." Why: the mounting fact tells an engineer which rating to compare, and "can reach" keeps T-O (cold-high is an expectation, not a bound). It renders 0.8049, 0.5893, 0.5082, 0.07117 and 0.7602 N·m, or "no prediction" past the end effect.
- SF-R2-2 (round 2; round 1's SF-7, second half; a test, shared with `slip_heating`): extend `the_slip_losses_read_fields_stored_from_the_example_design` (`tests/explain.rs`) with the four `temperature.demag.h_rev_*` inputs in `stored`, since watch_out[1]'s "the magnets' limits use fields stored ..." has no tripwire yet, and assert that the 20 mm design's `temperature.slip_loss.drag_Nm` differs from the default's (0.04668 against 0.01314 N·m) and the Y30 design's `temperature.summary.magnet_limit_C` from the default's (250 against 93.06 °C), so the test pins "the fields stay fixed while the results follow the design".

Considered, no change (the reviewer's observations):
- "Almost no torque on average" holds at all five designs: the drag is 0.49, 0.70, 0.26, 6.34 and 1.84 % of the pull-out. The 6.34 % at Y30 comes from the example design's stored NdFeB fields, which watch_out[1] discloses (round 1).
- Past the end effect (a 1 mm magnet, f_end −0.3214) the ripple and cold-high at the gearbox input render "no prediction"; the drag and the swing frequency render, since neither depends on f_end, and no negative torque is drawn (rounds 1 and 2).
- Ferrite's cold side with the stored NdFeB fields: at the Y30 design the cold limit is 335.7 °C, because the example's 863 kA/m like-pole field is applied to ferrite unscaled, so the cold check fails at any temperature. Watch_out[1] discloses the cause; the cold-side physics belongs to T16 (`ferrite_cold_demag`) and the backlog (round 2).
- The shaft joint sees the slip swings too: while slipping, the hub-to-shaft joint carries the same ±pull-out reversal at the slip frequency (166.7 Hz at the default), a fretting and fatigue case for a shrink fit or a key. It is input for BL-064's joint check; no note text is proposed while the shaft-clamp section is paused (round 2).

### Task review fix round 1 (2026-10-15): sentence 0 names the gearbox

- Why: the task code review (`task-15-review.md`, M-2) applied spec 3.1: sentence 0 answers the
  question of the row that links to the note. The dashboard's gearbox ripple row links here, and
  Start here step 9 opens the note on it, but sentence 0 never named the gearbox. SF-1 above is
  the fix. Rounds 1 and 2 passed the form check and left SF-1 offered.
- sentences[0]:
  - Old: "Past its pull-out torque the coupling slips: one ring turns against the other, and the
    torque between them swings back and forth."
  - New (SF-1 verbatim, 24 words; no symbol, placeholder, ϑ, U+207B or ∝; the caps and
    plain-words tests pass): "Past pull-out the coupling slips: one ring turns against the
    other, and the torque between them, carried by the gearbox, swings back and forth."
- Ruling 8: the coupling sits on the gearbox output, which carries the torque between the rings
  directly. The sentence says the gearbox carries the swing; nothing says the coupling protects
  the gearbox. "Past its pull-out torque" becomes "Past pull-out", the term sentence 1 uses ("the
  pull-out").
- Engine, at the five probe designs (round 1's numbers): the ripple at the gearbox input is the
  pull-out through the ratio and the efficiency (0.5660, 0.4144, 0.3573, 0.04359 and 0.5346 N·m),
  which `the_slip_note_s_claims_hold_at_every_probe_design` already pins.
- Tests:
  - `the_slip_note_s_claims_hold_at_every_probe_design` (`tests/explain.rs`) also asserts that
    sentence 0 names the gearbox. It failed on the old text ("sentence 0 names the gearbox") and
    passes on the new.
  - `the_slip_notes_keep_the_caps` (`notes.rs`, A-15's per-id form, added in this round for the
    task review's M-1) holds the three notes' caps and plain sentence 0, this one's 24 words
    among them.
- The dump for the re-review is `magcoupling-rs/target/notes-review-T15-fix1.txt` (the three
  notes, five designs; git-ignored).
- A text re-review of this field follows (controller ruling 7). Its report goes below as
  `### Re-review: code-review fix round 1`.

### Re-review: code-review fix round 1

- Reviewer: a fresh-context physics reviewer (session model), 2026-10-15, on commit `8cb78d4`
  (`task-15-textreview-1.md`). This is the text re-review of the code-review fix round 1 above
  (controller ruling 7). It re-checked the six fields that round changed in the three slip notes
  together; this note's is sentences[0].
- Scope here: sentences[0], and the fields whose meaning depends on it: sentences[1] ("The swing
  runs from plus to minus the pull-out ..."), sentences[4] (the ripple "at its input"),
  watch_out[0] (nothing checks the gearbox) and the dashboard's gearbox ripple row that links
  here (Start here step 9). The dump
  (`NOTES_FOR_REVIEW=when_slipping,slip_heating,thermal_time_constant`, written to
  `magcoupling-rs/target/rereview-T15-1.txt`, "3 notes at 5 designs") draws the six fields
  identically at the five probe designs. None holds a placeholder, `{`, `}` or "no prediction",
  and every aim line reads "moves as it says".
- Verdict: PASS, no must-fix, and no should-fix for this field. The re-review's three optional
  should-fix items are recorded under `## slip_heating` (SF-1 and SF-2, on-screen text, left open)
  and `## thermal_time_constant` (SF-3, a test, applied).

The reviewer's report for the record on this field follows unchanged.

- **Engine.**
  - The coupling sits on the gearbox's wheel (output) side (`plain_text.rs:143`; ruling 8), so the gearbox output carries the torque between the rings.
  - The ripple row is that torque's swing referred to the input: T_pull/(ratio·efficiency) (`model.rs:1156`).
  - Probe: 0.5660, 0.4144, 0.3573, 0.04359 and 0.5346 N·m at the five designs. Each equals pull-out (2.6885, 1.9684, 1.6974, 0.2071, 2.5393 N·m) / (5 × 0.95) exactly.
- **Spec 3.1, sentence 0.** It answers the gearbox ripple row (label "Estimated gearbox input torque ripple amplitude"; `ROW_LEARN`, `guidance.rs:318-321`) and Start here step 9. The swing is the gearbox's, and sentences[1] and [4] give its size and where it is quoted. No symbol, placeholder, ϑ, U+207B or ∝. 24 words by `word_count`.
- **Traps.**
  - "The coupling protects the gearbox" (spec 8, ruling 8): the sentence says the opposite. The gearbox carries the swing, and watch_out[0] says nothing checks it.
  - "Pull-out is guaranteed" and "the rings re-lock below pull-out" (slice-1 and slice-2 spec 8): neither is written. Sentence 3 covers re-locking.
  - T-H, T-M and T-O: unchanged by this field.
  - Ruling 11: no "clamp" and no shaft-joint method.
- **Dependent fields.** "The swing" in sentences[1] takes "swings back and forth" as its antecedent. "sends torque ripple to the gearbox: at its input" in sentences[4] stays consistent. The gearbox is now named twice (sentences 0 and 4), which is the intended Why-target structure.
- **Test.** `the_slip_note_s_claims_hold_at_every_probe_design` pins "gearbox" in sentence 0. `the_slip_notes_keep_the_caps` holds the caps and the sentence-0 rule.
- **Observation (engine scope, no text change).** The engine assumes the gearbox carries the full swing. Real shaft torque at the slip frequency depends on the drive ring's inertia and the shaft's stiffness, and can exceed the swing near a torsional resonance. This is input for the gearbox and joint backlog.
- Verdict for this field: PASS.

Considered, no change (the reviewer's observations on this field):
- **"Past pull-out" without "torque"** (`when_slipping` sentences[0]). The dashboard row is "Pull-out torque", and sentences[1] uses "the pull-out" as a noun. Keeping "its pull-out torque" with the gearbox clause would make 26 words.
- **"carried by the gearbox" and the drive train's inertia.** The engine (`model.rs:1156`) and ruling 8 take the gearbox to carry the full swing. The sentence matches them.
  - Physically, at the slip frequency (166.7 Hz at the default) the shaft torque is the magnetic swing minus what the drive ring's own inertia absorbs.
  - Near a torsional resonance of the drive train it can be larger. As the drive slows after a slip, the ripple frequency falls through the range, and nothing checks for that.
  - This is an engine-scope observation for the gearbox and joint backlog (BL-059, BL-064), not a text matter.

## slip_heating

Task T15. Rewrite of "Eddy-current slip loss and skin depth" as "Slip heating: where the heat
comes from", Start here step 10. It now also owns the field frequency, the pole pairs and the
slip speed in rad/s (`temperature.duty.field_freq_Hz`, `temperature.duty.pole_pairs`,
`temperature.duty.slip_rad_s`). Reviewer: physics reviewer (session model), fresh context. Date:
2026-10-14 (the date the slice plan fixes for the note's `review` field; both rounds ran on
2026-10-15).

Rounds: 2. Verdict: PASS. Note: "Slip heating: where the heat comes from" (plan T15).

Verdict by round: 1 MUST-FIX (1 item in this note); 2 PASS. Each round's full answer and the
round-1 changes file are as under `## when_slipping`; the three notes were reviewed together.

The text reviewed in round 1 already carried A-2 (change[0], the measured drag: `Step::Enter`, no
aim, "your bench value from a slipping test; it replaces the estimated loss and its high case in
the temperature results.") and A-13's plain-words sources. Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump; the `slip_heating` literal (`notes.rs:712-775`);
  - `records/slip_heating.rs:23-64` (the materials in effect), `:66-72` (duty) and `:85-147`:
    - skin depth √(2/(ω_e μ0 μ_r σ));
    - steel hub, cup and web σω_e²B²δ/(4k²): speed^1.5;
    - non-magnetic body (1 − e^(−2kd))/(2k): speed²;
    - shells σ t (ω_s r B)²/2 and the cap: speed²;
    - magnets σω_e²B²w²/24: speed²;
    - drag = P/ω_s;
  - `temperature.rs:1117, 1196-1219` (a measured drag replaces the loss used and the high case; the total and the drag stay the estimate);
  - `material_library.rs:744-749` (sleeve and liner choices: 316L, Ti6Al4V, IN625, PEEK);
  - `records/dashboard.rs:37-42` (running clearance);
  - spec 8; phys-notes-b 3 and 6; amendments A-2, A-13, A-27.
- Numbers reproduced from `compute_all`, in design order; every rendered value matches the dump:
  - total loss (`temperature.slip_loss.total_W`): 2.751, 2.870, 0.9240, 2.751, 9.777 W. Each equals drag × slip speed (209.44 rad/s).
  - drag: 0.01314, 0.01370, 0.004412, 0.01314, 0.04668 N·m.
  - high case (`temperature.slip_loss.high_W`): 8.254, 8.609, 2.772, 8.254, 29.33 W. This equals total × 3.
  - per part at the default: hub 0.2259, cup 1.3531, web 0.3656, sleeve 0.0747, liner 0.1805, cap 0.3236, magnets 0.2279 W. Skin depth 1.2995 mm (1.4529 mm at 8 magnets).
  - speed exponents (slip speed doubled): steel hub, cup and web 1.5000 at the four steel designs; non-magnetic hub, cup and web 2.0000 at the 304 design; sleeve, liner, cap and magnets 2.0000 at all five.
  - stored slip fields: identical at all five (b_hub 0.207, b_cup 0.214, b_sleeve 0.416, b_liner 0.419, b_magnet 0.19, b_hub_free 0.07832, b_cup_free 0.08764 T).
- Change directions confirmed:
  - Enter measured drag 0.02 N·m: the loss used and the high case become 4.189 W at all five. Steady high-case temperature: default 92.51 to 78.96 °C, 304 74.24 to 78.96 °C (the measurement replaces in both directions), 20 mm 162.8 to 78.96 °C. The total and the drag keep the estimate. "replaces", not "refines", holds.
  - Choose sleeve and liner material, codes 2, 3 and 4 (Ti6Al4V 0.595, IN625 0.775, PEEK ~0 MS/m, against 316L 1.35 MS/m):
    - sleeve plus liner loss falls at all five, from 0.0747 + 0.1805 W (default) to 0.0329 + 0.0796, 0.0429 + 0.1037 and 0 + 0 W;
    - the total, the steady temperature and the peak all fall, though the lighter shells lower C (default peak 65.964 to 65.918, 65.928, 65.882 °C).
  - Lower sleeve 0.01 mm: total 2.7513 to 2.7438 W; minimum running clearance −0.1032 to −0.0932 mm; peak falls; at all five.
  - Lower liner 0.01 mm: total 2.7513 to 2.7425 W; clearance the same; peak falls; at all five.
- Traps checked:
  - "steel loss follows speed^1.5 exactly": "about";
  - "a measured drag only refines the estimate": "replaces";
  - "the stored 3D fields follow the design": MF-4;
  - no lower-conductivity back iron and no pole count offered as a slip-loss lever (sentence 4 describes, offers no lever);
  - "the coupling can slip indefinitely": not written;
  - no shaft-joint method; "clamp" in any field, sources included: none.
- Form:
  - title 7 words;
  - sentences 22, 19, 23, 24, 23; changes 20, 21, 21, 21; watch-outs 27, 25, with 28 after MF-4;
  - sources pass the guards;
  - "skin" is glossed ("crowd into a thin surface skin"); "high case" is glossed in watch-out[1].
- Seeds and duplicates: no seed for this note. A-27: no trim on T15's side.
- Must-fix items: MF-4 (watch_out[0], "those fields"), to apply in round 2. Should-fix offered: SF-3 (watch_out[1]), SF-6 (sources[3]).
- Verdict for this note, round 1: MUST_FIX (1 item).

Round 2:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; rounds 1 and 2 (PASS in round 2).
- Read:
  - the dumps; the `slip_heating` literal (`notes.rs:712-775`);
  - `records/slip_heating.rs:23-147`: skin depth; the steel hub, cup and web, thin-skin at speed^1.5; the non-magnetic body at the whole thickness, speed²; the shells, the cap and the magnets at speed²; drag = P/ω_s;
  - `temperature.rs:236-275` and `:1116-1219` (the stored slip fields read as entered; a measured drag replaces the loss used and the high case, and the total and the drag stay the estimate);
  - `material_library.rs:744-749` (sleeve and liner choices); `records/dashboard.rs:37-43` (running clearance);
  - spec 8; phys-notes-b 3 and 6; amendments A-2, A-13 and A-27.
- Numbers reproduced from `compute_all`, in design order; every rendered value matches the dump:
  - total loss (`temperature.slip_loss.total_W`): 2.751, 2.870, 0.9240, 2.751, 9.777 W. Each equals drag × slip speed.
  - drag: 0.01314, 0.01370, 0.004412, 0.01314, 0.04668 N·m.
  - high case (`temperature.slip_loss.high_W`): 8.254, 8.609, 2.772, 8.254, 29.33 W, the estimate × 3.
  - The stored slip fields are identical at all five designs, at a 2.0 mm gap (total 2.915 W) and at an 8 mm magnet length (1.988 W). The losses follow the design; the fields do not.
- Change directions confirmed:
  - Enter measured drag 0.02 N·m: the loss used and the high case become 4.1888 W at all five designs, while the total and the drag keep the estimate. The steady high-case temperature becomes 78.963 °C, down from 92.51 °C at the default and up from 74.24 °C at 304 ("replaces").
  - Choose sleeve and liner material, codes 2, 3 and 4: the sleeve plus liner loss falls from 0.2552 W to 0.1125, 0.1466 and 0 W (20 mm: 1.2179 to 0.5370, 0.6993 and 0 W). The total and the peak fall at all five designs.
  - Lower sleeve and lower liner: the total falls at all five designs (aim test). The thinner shell widens the running clearance.
- Traps checked:
  - "steel loss follows speed^1.5 exactly" ("about");
  - "a measured drag only refines the estimate" ("replaces");
  - "the stored 3D fields follow the design" (watch_out[0]: those fields do not);
  - no lower-conductivity back iron and no pole count offered as a slip-loss lever;
  - "more magnets make the coupling stiffer": not written;
  - no shaft-joint method; "clamp" in any field, sources included: none.
- Form:
  - sentences 22, 19 (21 counted), 23, 24, 23; changes 20, 21, 21, 21; watch-outs 28, 25 (26 counted);
  - the guards are empty on every field.
- Must-fix items and how they were applied: watch_out[0] makes "those fields" the subject that does not follow the design (round 1, applied verbatim in round 2, re-reviewed true).
- Verdict: PASS.

Must-fix items applied (round 1, applied verbatim before round 2, which confirmed it):
- watch_out[0] (MF-4: after the colon, "they" read as the losses, which do follow the magnet count, the ring radius, the gap, the magnet size and the grade, 2.751 W at the default, 2.870 W at 8 magnets and 9.777 W at a 20 mm ring radius; only the stored fields stay fixed): "The losses use fields stored from a 3D model of the example design: they do not follow your magnet count, ring radius, gap, magnet size or grade." -> "The losses use fields stored from a 3D model of the example design; those fields do not follow your magnet count, ring radius, gap, magnet size or grade."

Seeds (A-24): none for this note.

Prose duplicates (A-27): no trim on this note's side, in rounds 1 and 2. Change[1] to change[3]
(the sleeve and liner material, the sleeve, the liner) overlap
`a5.high_conductivity_sleeve_or_liner`, but this is the general note, reached from the
total-loss row and Start here step 10, while the A5 note shows only when its warning fires.
T17, which rewrites the A5 note, should point at these levers rather than restate them.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS until the task review fix round 1 below, which changed sentences[2] and [4] and
applied SF-6):
- SF-3 (round 1), watch_out[1], 23 words: "The temperature check uses the high case, here {temperature.slip_loss.high_W}: until a drag is measured, the estimate times an allowance for the model's errors." Why: with a measured drag of 0.02 N·m the shipped text renders "Until a drag is measured, the high case, here 4.189 W, is the estimate times an allowance ...", while the value is then the measurement. The conditional keeps the shipped text logically true, but the value and its description disagree on screen; the replacement is true in both states.
- SF-6 (round 1; this note's half), sources[3]: "Design report: the low-Reynolds form for aluminium parts" becomes "Design report: the weak-field, whole-thickness loss of non-magnetic body parts". Why: "low-Reynolds" is magnetics jargon for this audience, and the rule applies to any non-magnetic body pick (304 included: `materials.body_sigma_S_m`), not to aluminium alone. (Applied verbatim by the task review fix round 1 below.)
- SF-R2-2 (round 2, a test): as under `## when_slipping`; its drag half pins this note's watch_out[0] ("those fields", not the losses, stay fixed).

Considered, no change (the reviewer's observations):
- The steel exponent: in the model the hub, cup and web losses grow as exactly the speed to the power 1.5 (1.5000 at twice the speed at the four steel designs), while the real exponent is about 1.54 to 1.64 (the source); "about ... 1.5" avoids the "exactly" trap. Without steel the same parts go as the speed squared (2.0000 at the 304 design), as do the shells, the cap and the magnets everywhere (round 1).
- "Heat through their whole thickness" without steel: the engine's depth (1 − e^(−2kd))/(2k) is set by the field's decay, not by a skin; "whole thickness" is the right plain contrast with "a thin surface skin" (round 1).
- Ferrite magnet eddy loss: at the Y30 design the magnet loss is 0.2279 W, the same as NdFeB's, because the engine uses the NdFeB conductivity (BL-062, "verify first"). Physically ferrite's eddy loss is negligible, so the estimate is conservative (about 8 % of the 2.751 W total), and the sentence on magnets heating through their thickness is true for conducting magnets. Left to BL-062 (round 1).
- Running clearance: at the default the minimum running clearance is −0.1032 mm, and one thinner step of the sleeve or the liner raises it to −0.0932 mm, so "widens the running clearance" is true; the trade-off (retention, unchecked) is named (round 1).
- With stored fields a wider gap raises the modelled loss (the radii grow: 2.751 to 2.915 W at a 2.0 mm gap), where real fields would weaken; watch_out[0]'s "gap" item tells the reader not to trust that trend (round 2).

### Task review fix round 1 (2026-10-15): "magnetic steel" in sentences 2 and 4; the non-magnetic body source

- Why: the task code review (`task-15-review.md`, I-1) found that sentences[2] and [4] said
  "steel" where only magnetic steel behaves that way.
  - The engine takes the skin-limited, speed^1.5 form for the hub, cup and web only with the
    magnetic-steel circuit (`materials.circuit_backiron` = 1, `records/slip_heating.rs:93-120`).
  - A 304 pick ("304 stainless (non-magnetic)", a probe design) makes the hub, cup and web 304
    steel with the whole-thickness, speed² form (2.0000 at twice the slip speed at the 304
    design; round 1).
  - On that design the old sentence 2 told the reader that their thick stainless parts were
    skin-limited, and sentence 4 ("Without steel back iron") read as not applying to them. Both
    were false there.
  - The project ruled on this word in T11 (round-1 must-fix) and T12 (must-fix), and the back-iron
    hint reads "Magnetic steel behind the magnets ...".
- sentences[2] (the task review's option A):
  - Old: "In thick steel the currents crowd into a thin surface skin, so its loss grows about as slip
    speed to the power 1.5."
  - New (24 words): "In thick magnetic steel the currents crowd into a thin surface skin, so its
    loss grows about as slip speed to the power 1.5."
- sentences[4] (the task review's option A):
  - Old: "Without steel back iron the hub, cup and web are non-magnetic metal: they see a weaker
    field and heat through their whole thickness."
  - New (24 words): "Without magnetic steel back iron the hub, cup and web are non-magnetic metal:
    they see a weaker field and heat through their whole thickness."
- sources[3] (SF-6 verbatim; the task review's option B, the same 304 conflation under the fold,
  and "low-Reynolds" is magnetics jargon):
  - Old: "Design report: the low-Reynolds form for aluminium parts"
  - New: "Design report: the weak-field, whole-thickness loss of non-magnetic body parts"
- Engine (`compute_all`; `material_library::resolve`): with a non-ferromagnetic back-iron pick
  the hub, cup and web take the pick's own values. At the 304 design the body's conductivity is
  304's 1.389 MS/m (the workbook's 6061: 25 MS/m) and its specific heat 304's 500 J/(kg·K)
  (aluminium: 900). With no back iron at all (the circuit input at 0) the body is the workbook's
  6061. Both are non-magnetic metal, so the new sentence 4 holds for either.
- Tests (`tests/explain.rs`):
  - `the_slip_heating_note_s_claims_hold_at_every_probe_design` also asserts that sentences[2]
    starts "In thick magnetic steel " and sentences[4] "Without magnetic steel back iron ", and
    that every "steel" in the sentences is " magnetic steel" (the space keeps "non-magnetic
    steel" out). Its exponents already key on the magnetic-steel circuit: 1.5 with it and 2
    without, at the five probe designs. It failed on the old sentences[2] and passes on the new.
  - New `the_non_magnetic_body_is_the_picked_metal_not_only_aluminium`: sources[3] here and
    `thermal_time_constant` sources[1] name "non-magnetic body parts", and no source of either
    note says "aluminium". At each probe design with a non-ferromagnetic back-iron pick, the
    circuit is not magnetic steel and the body's conductivity and specific heat are the pick's;
    one such design (304) is not aluminium. It failed on the old source and passes on the new.
- Not changed: sources[0] ("thin-skin steel losses") and sources[4] ("the steel loss's speed
  exponent") name the skin-limited case, which only magnetic steel has; the task review did not
  flag them. SF-3 (watch_out[1]) stays open; the task review lists it as M-10, outside this
  round.
- The dump for the re-review: as under `## when_slipping`.
- A text re-review of these three fields follows (controller ruling 7). Its report goes below as
  `### Re-review: code-review fix round 1`.

### Re-review: code-review fix round 1

- Reviewer: as under `## when_slipping` (`task-15-textreview-1.md`, on commit `8cb78d4`; ruling
  7).
- Scope here: sentences[2], sentences[4] and sources[3], and the fields whose meaning depends on
  them: sentences[3] (the shells' and magnets' speed-squared law), and sources[0] and [4] (the
  skin-limited steel loss). The dump: as under `## when_slipping`.
- Verdict: PASS, no must-fix. Two optional should-fix items, SF-1 (sources[3]) and SF-2
  (sentences[4]), are offered below the report and not applied: each would change reviewed
  on-screen text, which ruling 7 sends to another text re-review, so they stay open for the
  controller (as T12's re-review 2 left its optional items).

The reviewer's report for the record on these fields follows unchanged.

- **Engine.**
  - The hub, cup and web take the skin-limited form only with the magnetic-steel circuit (`materials.circuit_backiron` = 1, `records/slip_heating.rs:93-120`).
  - The circuit is 0 for a non-ferromagnetic back-iron pick (304, 6061) or with the back-iron switch off (`records/torque.rs:161-162`).
  - Magnetic steel is offered only as a back-iron pick. The sleeve, liner and cap choices are all non-magnetic (`material_library.rs:731-754`), so sentences[2] covers every part the engine prices with a skin.
- **Probe: speed exponents at twice the slip speed.**
  - Hub, cup and web: 1.5000 at the four magnetic-steel designs, 2.0000 at the 304 design.
  - Without magnetic steel the body is the pick's own metal: 304's 1.389 MS/m and 500 J/(kg·K) at its design. With the switch off and the 4140 pick, it is the workbook's 6061 (25 MS/m, 900 J/(kg·K)).
  - So "non-magnetic metal" in sentences[4] holds in every branch.
  - Body losses with the switch off: hub 0.1942, cup 1.5433, web 0.3737 W. At the 304 design: 0.0108, 0.0857, 0.0208 W.
- **sentences[2]** ("In thick magnetic steel ..."). Now true at the 304 design, where it does not apply to the reader's parts. The "about" keeps the model's exact 1.5 apart from the audit's 1.54 to 1.64. 24 words.
- **sentences[4]** ("Without magnetic steel back iron ..."). Now reads as applying to the 304 pick, whose hub, cup and web are non-magnetic. The weaker field holds: b_hub_free 0.07832 against 0.207 T, and b_cup_free 0.08764 against 0.214 T. 24 words.
  - The unchanged clause "heat through their whole thickness" is loose for a thick non-magnetic hub (d_eff / wall 0.197 at the default geometry, 0.133 at 20 mm) and close for the cup and web (0.631, 0.414), which carry most of the body loss. SF-2 offers a replacement; not required.
- **sources[3]** ("Design report: the weak-field, whole-thickness loss of non-magnetic body parts").
  - The report's T1 is linear in σ for any μr = 1 conductor in the low-Reynolds regime. 304 (σ 18 times lower than 6061) sits deeper in that regime, so citing it for every non-magnetic body is sound.
  - "Whole-thickness" describes the thin-sheet form T0 that the report replaces. SF-1 offers a replacement that keeps the tested phrase; not required.
- **Traps.**
  - "Steel gives the torque shown" and "1.5 T is 4140's saturation": not written. The 1.5 is a speed exponent.
  - No lower-conductivity back iron is offered as a slip-loss lever.
  - "The stored 3D fields follow the design": unchanged watch_out[0] covers it.
  - Ruling 11: no "clamp".
- **Tests.**
  - `the_slip_heating_note_s_claims_hold_at_every_probe_design` pins the two "magnetic steel" openings and allows no bare "steel" in any sentence. Its exponents key on the circuit.
  - `the_non_magnetic_body_is_the_picked_metal_not_only_aluminium` pins both sources' "non-magnetic body parts", and no "aluminium". It asserts the body takes the pick's σ and c at the non-ferromagnetic probe design, with at least one such design not aluminium.
- Verdict for these fields: PASS. Should-fix offered: SF-1 (sources[3]), SF-2 (sentences[4], optional).

Should-fix offered, not applied (open for the controller; no on-screen text changed after this
re-review):
- **SF-1 (`slip_heating` sources[3]; the cited report's form is not "whole-thickness").**
  - Replacement (15 words; guards empty; still contains "non-magnetic body parts" and no "aluminium", so `the_non_magnetic_body_is_the_picked_metal_not_only_aluminium` passes unchanged): "Design report: the eddy loss of non-magnetic body parts in the weaker field they see"
  - Why:
    - The report's T1 loss (section 5.5.3) is σω_e²B_free²/(2k²)·d_eff, with d_eff = (1 − e^(−2kd))/(2k). Its depth is set by how far the field reaches, which "saturates at 1/(2k) for thick parts".
    - What sets T1 apart is that it replaces the whole-thickness thin-sheet form T0, which the report shows overstates these parts' loss 1.6 to 5.1 times (table 5.5.2, "because kd is not small"). Labelling the report's form "whole-thickness" describes the form it rejects.
    - Probe, at the 304 design (the default geometry), d_eff / wall: hub 0.197 (5.10 mm wall, kd 2.525), cup 0.631, web 0.414. At the 20 mm design the hub's ratio is 0.133.
    - "Weak-field" is also ambiguous: it could mean the weaker free-space field or the low-Reynolds "weak reaction". The replacement says the plain thing that is true in both readings and in the report (section 5.5.4: the free-space fields are 0.38 to 0.41 of the steel-circuit ones), and it matches sentences[4].
- **SF-2 (`slip_heating` sentences[4], the unchanged clause "heat through their whole thickness"; lower priority).**
  - Replacement (25 words; starts "Without magnetic steel back iron "; one "steel", as " magnetic steel"; guards empty; the claims test passes unchanged): "Without magnetic steel back iron the non-magnetic hub, cup and web see a weaker field and form no skin: their loss grows as speed squared."
  - Why:
    - In-task round 1 accepted "whole thickness" as the plain contrast with "a thin surface skin", and the clause did not change here. A fresh look at the numbers shows it is loose for the hub.
    - At the 304 design, 80 % of the non-magnetic hub's heat lies within 1.60 mm of its 5.10 mm wall (the field at the back face is 8 % of the front). A magnetic-steel hub at the same speed puts 80 % within 1.05 mm. So for the hub, the contrast "thin skin" against "whole thickness" is not what separates the two cases.
    - What separates them is what sets the depth: speed for the skin (δ ∝ speed^−1/2, hence 1.5), and the field's spread, which does not depend on speed, for non-magnetic metal (hence 2; the probe's hub, cup and web exponents are 2.0000 at the 304 design).
    - The cup and web carry most of the body loss: 1.543 of 2.111 W with the back-iron switch off, 0.0857 of 0.1173 W at the 304 design. They are closer to whole-thickness (0.631 and 0.414), so the shipped text misleads little. That is why this is optional.
    - The replacement states the speed-squared law outright instead of leaving it to the parallel with sentences[3].

Considered, no change (the reviewer's observations on these fields; the "body parts" item covers
`thermal_time_constant` sources[1] too):
- **"body parts"** (both sources). The word "body" is new on screen. It ties to the σ_body, ρ_body and c_body symbols the Equation panel shows above the note, and `slip_heating` sentences[4] names the hub, cup and web just above the fold. Kept.
- **`slip_heating` sources[0] and [4]** ("thin-skin steel losses", "the steel loss's speed exponent"). Both name the skin-limited case, which only magnetic steel has, and they sit under sentences[2], which now says "magnetic steel". The fix round considered and declined this. I agree.
- **"thick"** (`slip_heating` sentences[2]). It is a general physics statement. The model applies the half-space form to every magnetic-steel wall (the default's 1.80 mm cup corner wall is 1.4 skin depths). In-task round 1 reviewed it; the fix only inserted "magnetic".

## thermal_time_constant

Task T15. Rewrite of "The thermal time constant" as "Slip temperature: how hot, and how fast",
Start here step 11. It writes τ_th and the hot-day start as the panel shows them (it wrote τ and
ϑ_start before: the open item M43-13, now resolved), names the peak the temperature check uses,
and now also owns the steady temperatures, the rise per slip event and the peak
(`temperature.thermal.steady_est_C`, `temperature.thermal.steady_high_C`,
`temperature.slip_life.rise_per_event_high_C`, `temperature.magnet_life.peak_C`). Its heating
diagram marks τ_th at 63 % of the steady rise and draws the limit and the time to reach it.
Reviewer: physics reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice
plan fixes for the note's `review` field; both rounds ran on 2026-10-15).

Rounds: 2. Verdict: PASS. Note: "Slip temperature: how hot, and how fast" (plan T15).

Verdict by round: 1 MUST-FIX (2 items in this note); 2 PASS. Each round's full answer and the
round-1 changes file are as under `## when_slipping`; the three notes were reviewed together.

The text reviewed in round 1 already carried A-2 (change[1], the measured drag: `Step::Enter`, no
aim, "your bench value from a slipping test; the loss and its high case then follow the
measurement.") and A-13's plain-words sources. Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump; the `thermal_time_constant` literal (`notes.rs:776-833`);
  - `records/slip_heating.rs:220-276`: C, τ_th = C/G, t95 = 3τ_th, steady rises P/G, steady temperatures, time to the limit (E12 first, then "never" when the steady temperature is at or below the limit), critical drag, fault temperature, event rise P_hi·t/C, slip duty, peak;
  - `temperature.rs:918` (hot-day start = hot ambient + driving rise), `:1221-1300` (thermal network), `:1316, 1335` (duty, peak) and `:1382-1423` (the verdict reads the hot-day margin, magnet limit − peak and adhesive limit − peak);
  - `gui/diagrams.rs` `heating_curve` and `LIMIT_SHARE`;
  - spec 8; phys-notes-b 4 and 6; amendments A-2, A-13, A-24.
- Numbers reproduced from `compute_all`, in design order; every rendered value matches the dump:
  - τ_th (`temperature.thermal.time_constant_s`): 274.0, 276.1, 289.0, 255.2, 553.1 s. This equals C/G with C = 82.19, 82.82, 86.71, 76.57, 165.94 J/K and G = 0.3 W/K.
  - peak (`temperature.magnet_life.peak_C`): 65.96, 66.00, 65.32, 65.98, 68.07 °C. The fault term wins at all five (default max(65.2001, 65.0100) + 0.02778 × 27.51 = 65.9644).
  - high-case steady temperature (`temperature.thermal.steady_high_C`): 92.51, 93.70, 74.24, 92.51, 162.8 °C.
  - governing limit (`temperature.summary.governing_limit_C`): 93.06, 93.06, 93.06, 120.0 (adhesive governs), 93.06 °C.
  - time to the limit, high case: never, 1049.6 s, never, never, 187.1 s. The estimate case: never at four designs, 1091.1 s at 20 mm. "never" holds exactly when the high-case steady temperature is at or below the limit.
  - hot-day start 65.00 °C and slip duty 0.02778 at all five.
- Change directions confirmed:
  - Raise conductance 0.001 W/K: steady high-case temperature 92.513 to 92.422 °C at the default, falling at all five. +10 % shortens τ_th (273.98 to 249.07 s). At a 60 s trip and 6000 rpm: 238.65 to 238.07 °C.
  - Enter measured drag 0.02 N·m: the loss and the high case follow the measurement (above). The peak moves: 65.964 to 65.489 °C at the default, 65.320 to 65.484 °C at 304.
  - Lower fault trip 0.1 s: the peak falls at every step down to the 0.1 s minimum at all five (default 65.9644 to 65.9544). With a 1 s slip event, trip 1.0 to 0.5 s leaves it unchanged at all five, hence MF-5. With a 140 °C start the peak stays above the limit at the shortest trip.
  - More mass (hardware +100 g): the steady temperature is unchanged exactly at all five, and τ_th lengthens (sentence 0).
- Diagram: τ_th at 63 % of the steady rise, a dashed limit at 0.93 of it, and a guide from the crossing down to the time axis at −ln(0.07)/5 of the width. It agrees with first-order heating. Its two tests pass.
- Traps checked:
  - "more mass lowers the steady temperature": sentence 0 says the opposite;
  - "the coupling can slip indefinitely": watch-out[0] ties "never" to a placeholder conductance;
  - "a measured drag only refines the estimate": not written;
  - symbols as the panel shows them (τ_th, C, G, P); no ϑ, and no σ for conductivity;
  - the hot-day start is glossed (sentence 1) and the peak is glossed (sentence 4);
  - no shaft-joint method; "clamp" in any field, sources included: none.
- Form:
  - title 7 words;
  - sentences 23, 25, 23, 21, 24; changes 23, 17, 21, with 20 after MF-5; watch-outs 27, 24;
  - the guards are empty on every field;
  - sentence 0 has no symbol and no placeholder.
- Seeds and duplicates: S-9 accepted (MF-5). No A-27 duplicate here.
- Must-fix items: MF-5 (change[2], S-9) and MF-6 (sources[0], no typed number), to apply in round 2. Should-fix offered: SF-4 (sentence 3 glosses), SF-5 (sentence 0 names the time constant), SF-6 (sources[1]).
- Verdict for this note, round 1: MUST_FIX (2 items).

Round 2:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; rounds 1 and 2 (PASS in round 2).
- Read:
  - the dumps; the `thermal_time_constant` literal (`notes.rs:776-833`);
  - `records/slip_heating.rs:220-276`;
  - `temperature.rs:918` (hot-day start = hot ambient + driving rise), `:1221-1300` (the thermal network; T_fault :1282) and `:1316-1335` (slip duty; peak = max(T_fault, start + one-event rise) + duty × P_hi/G);
  - `gui/diagrams.rs` `heating_curve`;
  - the math audit of 2026-09-29, "Placeholder inputs" (`docs/analyses/2026-09-29-magcoupling-math-audit.md:73-82`);
  - spec 8; phys-notes-b 4 and 6; amendments A-2, A-13 and A-24.
- Numbers reproduced from `compute_all`, in design order; every rendered value matches the dump:
  - τ_th (`temperature.thermal.time_constant_s`): 274.0, 276.1, 289.0, 255.2, 553.1 s.
  - peak (`temperature.magnet_life.peak_C`): 65.96, 66.00, 65.32, 65.98, 68.07 °C, recomputed exactly from the fault and event terms.
  - high-case steady temperature: 92.51, 93.70, 74.24, 92.51, 162.8 °C.
  - governing limit: 93.06 °C at four designs, and 120.0 °C at Y30 (the adhesive).
  - time to the limit, high case: never, 1049.6 s, never, never, 187.1 s.
- Change directions confirmed:
  - Raise conductance: the steady high-case temperature falls at all five designs (aim test).
  - Enter measured drag: the loss and its high case follow the measurement (above).
  - Lower fault trip:
    - one step lowers the peak at all five designs (aim test);
    - with a 1 s slip event, 1.0 s to 0.5 s leaves it unchanged at all five, so "can lower" is right;
    - over 107,820 trip steps (four event lengths, three starts, three slip speeds) none raised it;
    - with a 140 °C start the peak stays 140.26 to 142.73 °C at the 0.1 s minimum, above the 93.06 and 120 °C limits;
    - with the start set equal to the limit and the smallest measured drag, it stays at or above the limit.
- Diagram: τ_th at 63 %, the limit and the time to reach it; its two tests pass.
- Traps checked:
  - "more mass lowers the steady temperature" (sentence 0 says the opposite);
  - "the coupling can slip indefinitely" (watch_out[0] and sources[0]: a placeholder conductance);
  - "a measured drag only refines the estimate";
  - a typed default value in any field, sources included: none after the round-1 fix;
  - symbols as the panel shows them;
  - no shaft-joint method; "clamp" in any field, sources included: none.
- Form:
  - sentences 23, 25, 23 (24 counted), 21, 24; changes 23, 17, 20 (21 counted); watch-outs 27 (29 counted), 24;
  - the guards are empty on every field.
- Seeds: S-9 accepted and applied verbatim.
- Must-fix items and how they were applied (round 1, applied verbatim in round 2, re-reviewed true):
  - change[2] says a shorter unbroken slip "can lower" the peak, "but not under a limit the hot-day start already reaches" (S-9);
  - sources[0] drops the typed −2.48 °C and names the audit's placeholder-input table.
- Verdict: PASS.

Must-fix items applied (all in round 1, applied verbatim before round 2, which confirmed each):
- change[2] (MF-5, seed S-9 accepted verbatim; input `temperature.duty.fault_trip_s`, `Step::Lower`, aim `temperature.magnet_life.peak_C` down, all unchanged: once one slip event is hotter than the fault slip, a shorter trip leaves the peak unchanged, as with a 1 s slip event at all five designs): "a shorter unbroken slip lowers the peak, here {temperature.magnet_life.peak_C}, but cannot bring it under a limit the hot-day start already reaches." -> "a shorter unbroken slip can lower the peak, here {temperature.magnet_life.peak_C}, but not under a limit the hot-day start already reaches."
- sources[0] (MF-6: a typed default-design number on screen under the fold, which A-13 and the Global Constraints forbid, and which did not reproduce: +10 % conductance moves the high-case steady temperature by −2.501 °C at the default and by −0.840 to −8.888 °C at the other designs): "Math audit of 2026-09-29: the placeholder inputs (conductance: −2.48 °C per +10 %)" -> "Math audit of 2026-09-29: the placeholder inputs and how far each moves the results, the conductance among them"

Seeds (A-24):
- S-9 (change[2]: "a shorter unbroken slip can lower the peak, here {temperature.magnet_life.peak_C}, but not under a limit the hot-day start already reaches."): accepted in round 1 and applied verbatim as MF-5; round 2 confirms it. Reason: the peak is max(fault slip, one slip event) + duty × P_hi/G (`temperature.rs:1335`), so once one slip event outlasts the fault slip a shorter trip leaves the peak unchanged (probed at all five designs with a 1 s event), and "lowers" is false on reachable inputs. "Can lower" holds at every probe; round 2 found no shorter trip that raised the peak over 107,820 trip steps, and with a 140 °C hot-day start the peak stays above the limits at the 0.1 s minimum trip.

Prose duplicates (A-27): none named for this note.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS until the task review fix round 1 below, which applied SF-5 and SF-6):
- SF-4 (round 1), sentences[3], 24 words: "One short slip lasting t adds only P·t/C; slips over the life add an average rise, P/G times the share of life spent slipping." Why: t is not defined, and "slip duty" is not glossed along Start here; the physics is unchanged (the event rise is P_hi·t/C, and the life-average rise is (life events × event duration / life hours) × P_hi/G).
- SF-5 (round 1), sentences[0], 25 words: "Unbroken slip heats the coupling toward a steady rise set by loss and heat path; mass changes only how fast, which the time constant measures." Why: the note is opened from the thermal time constant (Start here step 11), and the shipped sentence 0 never names it. The trap guard stays: more heat capacity leaves the steady temperature exactly unchanged at all five designs and lengthens τ_th (273.98 to 431.65 s for 100 g more hardware at the default). (Applied verbatim by the task review fix round 1 below.)
- SF-6 (round 1; this note's half), sources[1]: "Design ruling: aluminium parts at their own specific heat" becomes "Design ruling: non-magnetic body parts at their own specific heat". Why: the rule applies to any non-magnetic body pick (304 included: `materials.body_c_J_kgK`), not to aluminium alone. (Applied verbatim by the task review fix round 1 below.)
- SF-R2-1 (round 2; round 1's SF-7, first half; a test): in `tests/explain.rs`, at each probe design with `metal.slip_event_s` = 1.0, a fault trip of 1.0 s and of 0.5 s give the same `temperature.magnet_life.peak_C` while `temperature.thermal.temp_at_fault_C` falls (probed: equal peaks at all five designs). Why: today the aim test pins only the "lowers" half of "can lower".

Considered, no change (the reviewer's observations):
- "Never": the screen text reads "never: steady state stays below the limit" (`temperature.rs:704`), while the rule and watch_out[0] say "at or below"; they differ only at exact equality (round 1).
- The heating curve draws the finite-time case (the limit at 0.93 of the steady rise, crossed after −ln 0.07 = 2.66 time constants); three probe designs read "never", which watch_out[0] covers, so no second picture is needed (round 1).
- The same stale figure outside T15: `magcoupling-rs/src/engine/assumptions.rs:128` (the conductance assumption's `source`, drawn only with the spreadsheet details on) still reads "−2.48 °C of steady high-case magnet temperature per +10 %", the audit's figure at the old defaults (−2.501 °C today). Its owner is whichever task owns the Assumptions view's texts (round 2).

### Task review fix round 1 (2026-10-15): sentence 0 names the time constant; the non-magnetic body source

- Why:
  - The task code review (`task-15-review.md`, M-2) applied spec 3.1: Start here step 11 opens
    this note on the thermal time constant (`temperature.thermal.time_constant_s`), but sentence 0
    never named it. SF-5 above is the fix. Rounds 1 and 2 passed the form check and left SF-5
    offered.
  - The same review's I-1, option B, takes SF-6's half here (sources[1]): the same 304
    conflation as `slip_heating` sources[3].
- sentences[0]:
  - Old: "Unbroken slip heats the coupling toward a steady rise set by the loss and the heat path;
    more mass changes only how fast."
  - New (SF-5 verbatim, 25 words, at the cap; no symbol, placeholder, ϑ, U+207B or ∝; the caps
    and plain-words tests pass): "Unbroken slip heats the coupling toward a steady rise set by
    loss and heat path; mass changes only how fast, which the time constant measures."
- sources[1]:
  - Old: "Design ruling: aluminium parts at their own specific heat"
  - New (SF-6 verbatim): "Design ruling: non-magnetic body parts at their own specific heat"
- Engine:
  - The trap guard holds. More heat capacity lengthens τ_th and leaves the high-case steady
    temperature exactly unchanged at all five probe designs;
    `the_slip_temperature_note_s_claims_hold_at_every_probe_design` pins both.
  - Without magnetic steel the heat capacity takes the body's own specific heat
    (`records/slip_heating.rs:223-225`): 304's 500 J/(kg·K) at its design.
- Tests:
  - The slip-temperature claims test also asserts that sentence 0 names "time constant". It
    failed on the old text ("sentence 0 names the time constant") and passes on the new.
  - Sources[1] is held by `the_non_magnetic_body_is_the_picked_metal_not_only_aluminium` (under
    `## slip_heating`).
  - `the_slip_notes_keep_the_caps` holds sentence 0's 25 words.
- Not changed: SF-4 (sentences[3]) and the test SF-R2-1 stay open; the task review lists the
  latter as M-6, outside this round.
- The dump for the re-review: as under `## when_slipping`.
- A text re-review of these two fields follows (controller ruling 7). Its report goes below as
  `### Re-review: code-review fix round 1`.

### Re-review: code-review fix round 1

- Reviewer: as under `## when_slipping` (`task-15-textreview-1.md`, on commit `8cb78d4`; ruling
  7).
- Scope here: sentences[0] and sources[1], and the fields whose meaning depends on them:
  sentences[2] (τ_th = C/G, 63 %), change[0] (the conductance shortens τ_th) and the Start here
  step that opens the note on the thermal time constant (`temperature.thermal.time_constant_s`).
  The dump: as under `## when_slipping`.
- Verdict: PASS, no must-fix. One optional should-fix item, SF-3 (a test), applied with no
  on-screen text change (below the report).

The reviewer's report for the record on these fields follows unchanged.

- **Engine.** The steady rise is P/G and τ_th = C/G (`records/slip_heating.rs:231-236`). The heat capacity is the only place mass enters.
- **Probe: +100 g of hardware.**
  - τ_th: 273.98 to 431.65 s (default), 276.07 to 433.74, 289.03 to 446.69, 255.23 to 412.90 and 553.14 to 710.81 s.
  - The high-case and estimate steady temperatures are bit-identical before and after at all five designs (default 92.513112 and 74.171037 °C).
  - So "steady rise set by loss and heat path; mass changes only how fast" holds exactly.
  - Mass still lowers the checked peak through "how fast" (default 65.9644 to 65.8914 °C). That is consistent with the sentence and with sentences[3] and [4].
- **Spec 3.1, sentence 0.** It names the time constant, the row Start here step 11 opens the note on ("Thermal time constant"). No symbol, placeholder or Greek letter: τ_th first appears in sentences[2], which defines it as C/G. 25 words, at the cap.
- **sources[1].**
  - Without magnetic steel the heat capacity prices the hub, cup and boss at the body's own specific heat (`records/slip_heating.rs:222-229`): 304's 500 J/(kg·K) at its design, or 6061's 900.
  - E15's ruling was written for the aluminium body. The engine applies the same rule to any non-magnetic pick, so the source is accurate.
  - With magnetic steel, the hub, cup, boss and hardware take the steel's own specific heat, so the source implies nothing false about steel.
- **Traps.**
  - "More mass lowers the steady temperature": the sentence says the opposite.
  - "The coupling can slip indefinitely": not written. watch_out[0] ties "never" to the placeholder conductance.
  - Ruling 11: no "clamp".
- **Test.** The claims test pins "time constant" in sentence 0 and the heavier design's unchanged steady rise. SF-3 offers pinning "mass changes only how fast" as well.
- Verdict for these fields: PASS. Should-fix offered: SF-3 (test).

- SF-3, applied by the implementer (session model), 2026-10-15:
  `the_slip_temperature_note_s_claims_hold_at_every_probe_design` (`tests/explain.rs`) also
  asserts that sentence 0 contains "mass changes only how fast", the words that keep it clear of
  the trap "more mass lowers the steady temperature"; before, only "time constant" was pinned.
  With "only" dropped from sentence 0 the test fails ("sentence 0 keeps the mass trap guard");
  the shipped text passes.

Considered, no change (the reviewer's observations on these fields; "body parts" in sources[1]:
as under `## slip_heating`):
- **"mass changes only how fast"** (`thermal_time_constant` sentences[0]). Mass still lowers the checked peak through "how fast": +100 g of hardware takes the default's peak from 65.9644 to 65.8914 °C (fault 65.2001 to 65.1272 °C), while the steady temperatures stay exact (92.513112 and 74.171037 °C). The sentence does not say mass cannot lower the peak, and sentences[3] and [4] show the finite-time terms. Consistent.
- **"which the time constant measures."** "Which" binds to "how fast". τ_th = C/G also depends on the heat path, and change[0] says so ("shortens τ_th"). No false implication.

## demagnetization

Task T16. Rewrite of "Demagnetization: knee, permeance and the onset temperatures" as
"Demagnetization: permanent loss from heat and slipping", Start here step 12. It now also owns
the two onset margins (`temperature.magnet_life.margin_onset_C`,
`temperature.magnet_life.margin_limit_C`), types no stored reverse field, knee fraction or design
margin as a fact about the design (`the_rewritten_notes_type_no_editable_default` holds it), and
keeps the DemagKnee diagram. Reviewer: physics reviewer (session model), fresh context. Date:
2026-10-14 (the date the slice plan fixes for the note's `review` field; all three rounds ran on
2026-10-15).

Rounds: 3. Verdict: PASS. Note: "Demagnetization: permanent loss from heat and slipping" (plan
T16).

Verdict by round: 1 MUST-FIX (3 items in this note, 4 across the task's two notes); 2 MUST-FIX (1
item, in this note); 3 PASS. The two notes were reviewed together. Each round's full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-16-inreview-r<N>.md`, and the fields
that rounds 1 and 2 changed are listed with their old and new text in
`task-16-inreview-r1-changes.md` and `task-16-inreview-r2-changes.md` beside it. Rounds 2 and 3
re-checked only those fields and the fields whose meaning depends on them.

The text reviewed in round 1 already carried A-2 (change[2], the like-poles reverse field:
`Step::Enter`, no aim, "enter it from a 3D field calculation of your design, which needs an
outside solver; the stored value belongs to the example design.") and A-13's plain-words
sources. Neither note names a clamp: the `clamp_preload` rewrite is dropped while the
shaft-clamp section is paused (A-28d, ruling 11), and that note keeps its slice-1 text and its
2026-10-01 review. Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump `notes-review-T16-r1.txt`; the `demagnetization` literal (`notes.rs:599-656`);
  - `temperature.rs:88-121` (the inputs: the four stored fields :105-116; "once per pole pass while skipping" :112; the coercivity source :117-120);
  - `temperature.rs:710-732` (`demag_onset_C`, `cold_onset_C`) and `:775-874` (`ring_demag`: grade or inputs; the cold-side branch; t_ref and the offset t_ref − rating; the onsets; the limit = skipping onset − margin, or the rating for a positive coefficient);
  - `records/demagnetization.rs` (each ring's limit; the governing ring; h_ref, t_ref, offset and onsets; the cold side; the governing limit with the adhesive);
  - `records/temperature.rs:42-57` (the summary and the verdict); `records/slip_heating.rs:295-296` (the two onset margins);
  - `grades.rs` (all 17 grades: ratings, Br, Hcj, β, ranges); `library.rs:41-46, 339-385` (the arcs' 60 °C vendor rating under E19); `assumptions.rs:106` (the 9.83 °C);
  - `guidance.rs:193-200` (the Temperature limits guide); `notes.rs` START_HERE_ALL (step 12, `temperature.demag.onset_skipping_C`); `diagrams.rs:404-476` (DemagKnee);
  - spec 3.1-3.3, 5 and 8; slice-1 spec 8; phys-new-notes 2 (T-N); phys-notes-b 1 and 6; amendments A-2, A-13, A-15, A-28 and A-28d, ruling 11.
- Numbers reproduced from `compute_all`, in design order (default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius):
  - {temperature.demag.onset_skipping_C}: 103.0557, 103.0557, 103.0557, +inf, 103.0557 °C. The dump reads "103.1 °C" four times and "no prediction" at Y30.
  - Behind the sentences, at the four NdFeB designs:
    - h_ref = 517.25 kA/m; t_ref = 159.92 °C; offset = +9.92 °C (rating 150 °C);
    - onsets: aligned 170.16, pull-out 113.35, single ring 143.36 °C. Slipping is the lowest of the four.
    - the magnet limit is 93.0557 °C, which is the skipping onset − 10. The inner and outer limits are equal, and the magnets govern.
  - At Y30: each ring's limit is 250 °C (its rating); offset 0; the adhesive governs at 120 °C.
  - The owned margins:
    - `margin_onset_C`: 37.09, 37.05, 37.74, +inf, 34.99 °C;
    - `margin_limit_C`: 27.09, 27.05, 27.74, 184.02, 24.99 °C;
    - the peak: 65.96, 66.00, 65.32, 65.98, 68.07 °C. Each margin is the onset or limit less the peak.
  - Watch-out 1, offsets on custom dimensions at the default (°C):
    - negative: N52 −9.69, N50 −6.14, N50M −7.54, N38UH −7.34, N35EH −3.57, Recoma 26 −60.21, Recoma 30 −0.45;
    - positive: N35 +22.58, N42 +12.68, N48 +6.46, N42M +9.97, N42H +10.82, N42SH +9.92, N33AH +30.75, Recoma 20 +217.33, BCN-19 +94.28;
    - zero: Y30 (cold side).
    - Library arcs (N50 at 60 °C): +12.98. Library N52 blocks: −9.69.
  - The default design's aligned onset (170.16 °C) lies past N42SH's tested range (20 to 150 °C).
- Change directions confirmed:
  - **change[0], Choose inner part.** Custom dimensions and the grade, at default / 8 magnets / 304 / 20 mm:
    - **both rings:** the limit rises with N38UH, N35EH, N33AH and Recoma 26 (131.90, 155.43, 173.43, 161.95 °C against 93.06 °C), and the adhesive governs (120 °C).
      - The pull-out falls by 11.9, 19.0, 23.1 and 37.7 % at the default, 8-magnet and 20 mm designs.
      - At 304 it falls by 20.3, 26.7, 30.4 and 43.6 %, as the bench factor goes too (1.0495 to 0.95).
    - **inner only:** the inner limit rises, the outer 93.06 °C still governs, and the pull-out falls (default: 2.5230, 2.4196, 2.3576, 2.1228 against 2.6885 N·m).
    - **Counterexamples, now MF-1:**
      - Recoma 30 (rated 250 °C): 46.27 °C, both rings or one;
      - the Y30 design: all four grades lower the ring's limit from 250 °C and raise the pull-out;
      - lower-rated NdFeB starts (N42 to N42SH: same Br).
    - Under MF-1's text, the limit rises in all 55 sintered-NdFeB rated-hotter pairs. The only exception is the 8 pairs into bonded BCN-19 ("usually"). The three NdFeB grades above N42SH are weaker (1.22, 1.17, 1.14 against 1.30 T).
  - **change[1], Choose outer part.** The mirror case: `a_hotter_rated_grade_raises_...` runs `Rings::Outer`, and the other ring governs. "The lower of the two limits governs" holds at all five designs (`magnet_limit_C = min`).
  - **change[2], Enter like-poles field.** No direction is claimed.
    - The four stored fields are the example's at all five designs.
    - The onsets do not move with magnet count, ring radius, gap or block size (the claims test), nor with the 304 back iron (103.0557 °C).
    - The limit reads only the like-poles field, so this is the input to enter.
- Traps checked:
  - **"the stored 3D fields follow geometry or grade":** watch-out 0 says the opposite.
  - **"thicker magnets, a smaller gap or fewer poles raise the calculator's demagnetization limit":** not written.
  - **"a hotter-rated grade is free":** torque and the adhesive are named; under MF-1, only for NdFeB grades, where it holds.
  - **"the rating shift is conservative":** watch-out 1 says some onsets are shifted upward; MF-3 makes the list exact.
  - **"pull-out is guaranteed":** not written.
  - **"the coupling protects the gearbox":** not written; the guard test passes.
  - **"a shaft-joint method is chosen or sized here":** not written.
  - **"more magnets make the coupling stiffer":** not written.
  - **slice-1 spec 8 (radius lever, fewer poles, re-lock, bench calibration, 416):** not written.
  - **T-N:** slipping is the largest stored field and sets the limit.
  - **No editable default is typed:** no 354, 791, 863 or 569; no "10 °C"; no 0.9. `the_rewritten_notes_type_no_editable_default` passes.
  - **"clamp" in any field, sources included:** none.
- Form:
  - title 7 words; sentences 23, 25, 24, 20, 25; changes 24, 24, 23; watch-outs 27, 27;
  - after MF-1, MF-2 and MF-3: change[0] 25, sentences[4] 25, watch_out[1] 30;
  - sentence 0 has no symbol and no placeholder, and it answers the Temperature limits row ("too hot, and the magnets can lose strength for good");
  - the one placeholder sits in a "here" clause with no comparison;
  - reverse field, knee and onset are glossed in sentences 1-2, before step 13 uses them;
  - every field passes `plain_text_violations` and `audit_ids`;
  - the DemagKnee diagram is kept, and its knee and "H (reverse field)" match the text (SF-5: the "load lines" label).
- Seeds: none indexed for T16.
- Must-fix items: MF-1 (change[0]), MF-2 (sentences[4]) and MF-3 (watch_out[1]), to apply in round 2. Should-fix offered: SF-2 (back iron in watch_out[0]); SF-5 (diagram label, a GUI pass); SF-6 and SF-7 (tests, record).
- Verdict for this note, round 1: MUST_FIX (3 items).

Round 2:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 2. The form's fixed date is 2026-10-14; the review ran on 2026-10-15, as in round 1.
- Read:
  - the dump `notes-review-T16-r2.txt`, the r1 dump and the r1 to r2 diff (four fields, 20 lines);
  - the `demagnetization` literal (`notes.rs:599-656`) and `task-16-inreview-r1-changes.md`;
  - `temperature.rs:710-733` (`demag_onset_C`, `cold_onset_C`) and `:775-874` (`ring_demag`: offset = t_ref − the ring's `tmax_lib_C`; limit = skipping onset − margin; the rating for a positive coefficient);
  - `api.rs:278, 319` (each ring's rating from `model.inner/outer_tmax_C`); `model.rs:400-425` (`ResolvedMagnet`: library or grade rating);
  - `library.rs:24-56, 339-385` (E19: the arcs' 60 °C vendor rating; M5045 as N50); `grades.rs` (all 17 grades);
  - `pickers.rs:155-157, 209` (the part's and the grade's ratings as the GUI shows them); `guidance.rs:887` ("rated hotter than the part");
  - `tests/explain.rs:3005-3093` (the T16 claims tests);
  - spec 8, slice-1 spec 8, phys-new-notes 2 (T-N), amendments A-2, A-13, A-15, A-28 and A-28d, ruling 11.
- Numbers reproduced from `compute_all`, in design order (default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius):
  - {temperature.demag.onset_skipping_C}: 103.0557, 103.0557, 103.0557, +inf, 103.0557 °C. The dump reads "103.1 °C" four times and "no prediction" at Y30.
  - At the four NdFeB designs:
    - h_ref 517.25 kA/m; t_ref 159.92 °C; offset +9.92 °C (rating 150 °C);
    - onsets: aligned 170.16, pull-out 113.35, single ring 143.36 °C;
    - ring limits 93.0557 °C both, and the magnets govern.
  - At Y30: each ring's limit is 250 °C (its rating), the offset is 0, and the adhesive governs at 120 °C.
  - The rating shift's target, `t_ref − offset`, equals the ring's own rating for every library part and every custom grade.
    - For the arcs M5044, M5045 and M5026 that is 60.00 °C, the vendor's rating. Their grade N50 is rated 80 °C, and their offset is +12.98 °C. This is R2-MF-1.
  - Offsets on custom dimensions, identical at all five designs (°C):
    - negative: N52 −9.69, N50 −6.14, N50M −7.54, N38UH −7.34, N35EH −3.57, Recoma 26 −60.21, Recoma 30 −0.45;
    - library: N52 parts −9.69, arcs +12.98, N42 parts +12.68, N42SH parts +9.92.
- Change directions confirmed:
  - **change[0], Choose inner part.**
    - From the probe designs' N42SH, custom dimensions with N38UH, N35EH or N33AH raise the ring's limit to 131.90, 155.43 or 173.43 °C.
    - The pull-out falls by 11.9, 19.0 and 23.1 % on both rings (20.3, 26.7 and 30.4 % at 304) and by 6.2, 10.0 and 12.3 % on the inner ring alone. The adhesive governs on both rings.
    - Over every start (15 library parts, 17 custom grades; both rings and the inner ring only; five designs; knee fraction 0.5 and 1.0; coercivity from the inputs), 1670 of 1670 moves to a sintered NdFeB grade rated hotter raise the ring's limit.
    - Bonded BCN-19 lowers it in 200 of 200 moves ("usually").
    - At Y30, no NdFeB grade is rated above 250 °C.
  - **change[1], Choose outer part.** The lower ring limit governs in 640 of 640 mixed-ring cases.
  - **change[2], Enter like-poles field.** Unchanged this round, no direction claimed.
- Traps checked:
  - "a hotter-rated grade is free": torque and the adhesive are named.
  - "the rating shift is conservative": watch_out[1] says some onsets read optimistic.
  - "the stored 3D fields follow geometry or grade": not written.
  - "thicker magnets, a smaller gap or fewer poles raise the demagnetization limit": not written.
  - "pull-out is guaranteed": not written.
  - "the coupling protects the gearbox": not written, and the guard passes.
  - "a shaft-joint method is chosen or sized here": not written.
  - "more magnets make the coupling stiffer": not written.
  - slice-1 spec 8 and T-N: not contradicted.
  - "clamp" in any field, sources included: none.
- Form:
  - change[0] 25 words, sentences[4] 25 words, watch_out[1] 30 words;
  - every field passes `plain_text_violations` and `audit_ids`;
  - no typed default; the caps test passes.
- Must-fix items and how they were applied:
  - Round 1's MF-1, MF-2 and MF-3 were applied verbatim and re-checked: they pass.
  - Round 2: R2-MF-1, sentences[4]'s "the grade's rated maximum temperature" becomes "the magnet's rated maximum temperature", to apply in round 3.
- Verdict for this note, round 2: MUST_FIX (1 item).

Round 3:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 3. The form's fixed date is 2026-10-14; rounds 1 to 3 ran on 2026-10-15.
- Read:
  - the dumps `notes-review-T16-r1.txt` to `-r3.txt` and the diffs between them;
  - the `demagnetization` literal (`notes.rs:599-656`);
  - `temperature.rs:88-121` (the inputs: the four stored reverse fields, the coercivity source);
  - `temperature.rs:710-733` (`demag_onset_C`, `cold_onset_C`) and `:775-874` (`ring_demag`):
    - offset = t_ref − the ring's own rating, or 0 for a ring with no rating or a positive coefficient;
    - limit = skipping onset − margin, or the rating for a positive coefficient;
  - `records/demagnetization.rs` (each ring's limit, the governing ring, h_ref, t_ref, the offset, the onsets, the cold side, the governing limit);
  - `model.rs:400-476` (`ResolvedMagnet`: the part's rating, the grade's, or none);
  - `library.rs:36-56, 339-385` (E19: the arcs' vendor rating of 60 °C; M5045 as N50);
  - `grades.rs` (all 17 grades); `pickers.rs:155-157, 209` (the part's and the grade's ratings as the GUI shows them);
  - `tests/explain.rs`, the T16 claims tests;
  - spec 3.1-3.3, 5 and 8; slice-1 spec 8; phys-new-notes 2 (T-N); phys-notes-b 1 and 6; amendments A-2, A-13, A-15, A-28 and A-28d; ruling 11.
- Numbers reproduced from compute_all, in design order (default, 8 magnets per ring, 304 back iron, Y30 ferrite custom dimensions, 20 mm ring radius):
  - {temperature.demag.onset_skipping_C}: 103.0557, 103.0557, 103.0557, +inf, 103.0557 °C. The note reads "103.1 °C" four times and "no prediction" at Y30.
  - At the four NdFeB designs:
    - h_ref 517.25 kA/m; t_ref 159.92 °C; offset +9.92 °C, so t_ref − offset = 150 °C, the N42SH rating;
    - onsets: aligned 170.16, pull-out 113.35, single ring 143.36 °C (slipping is the lowest);
    - each ring's limit is 93.0557 °C (the skipping onset − 10), and the magnets govern;
    - margin_onset_C 37.09, 37.05, 37.74, 34.99 °C and margin_limit_C 27.09, 27.05, 27.74, 24.99 °C.
  - At Y30:
    - each ring's limit is 250 °C (its rating), with offset 0;
    - the adhesive governs at 120 °C;
    - margin_onset_C is +inf and margin_limit_C is 184.02 °C.
  - The rating shift follows each ring's own rating. In 5940 ring evaluations (every library part, every custom grade and a blank magnet; both rings, the inner ring or the outer ring; five designs; the knee fraction at 0.5, 0.9 and 1.0; the coercivity from the inputs; the margin at 0, 10 and 40 °C):
    - the ring's rating fits the engine's limit in 5390 of 5390 rated NdFeB-side rings. Only the N50 arcs M5044, M5045 and M5026 (rated 60 °C against N50's 80 °C) tell the two apart: −22.50 °C, not −2.50 °C.
    - the 120 blank rings are unshifted;
    - the 430 positive-coefficient rings take their rating as the hot limit.
  - The offsets on custom dimensions (the same at all five designs) that are negative, which shifts the onsets upward: N52 −9.69, N50 −6.14, N50M −7.54, N38UH −7.34, N35EH −3.57, Recoma 26 −60.21 and Recoma 30 −0.45 °C.
    - Library: N52 parts −9.69, arcs +12.98, N42 parts +12.68 and N42SH parts +9.92 °C.
    - The watch-out's list matches, at the grade's own rating, in 145 of 145 cases.
- Change directions confirmed:
  - **Choose inner part:** custom dimensions with N38UH, N35EH or N33AH raise the ring's limit from 93.06 °C to 131.90, 155.43 or 173.43 °C.
    - **Both rings:** the pull-out falls 11.9, 19.0 and 23.1 % (20.3, 26.7 and 30.4 % at 304), and the adhesive governs.
    - **Inner ring only:** the pull-out falls 6.2, 10.0 and 12.3 % (15.0, 18.5 and 20.6 % at 304), and the outer ring still governs.
    - **Every start:** 1670 of 1670 moves to a sintered NdFeB grade rated hotter than the magnet raise the ring's limit. Bonded BCN-19 lowers it in 200 of 200 ("usually").
    - **At Y30:** no NdFeB grade is rated above 250 °C, so the entry makes no claim there.
    - **"those above N42SH are weaker":** N38UH 1.22, N35EH 1.17 and N33AH 1.14 T, against N42SH's 1.30 T.
  - **Choose outer part:** the lower ring limit governs in 640 of 640 mixed-ring cases. With one ring changed, the other ring's limit still governs ("may not help").
  - **Enter like-poles field:** no direction is claimed.
    - The four stored fields are the example's at every design.
    - The onsets do not move with magnet count, ring radius, gap, block size or the 304 back iron.
- Traps checked:
  - "a hotter-rated grade is free": torque and the adhesive are named.
  - "the rating shift is conservative": watch_out[1] says some onsets are shifted upward.
  - "the stored 3D fields follow geometry or grade": watch_out[0] says the opposite.
  - "thicker magnets, a smaller gap or fewer poles raise the demagnetization limit": not written.
  - "pull-out is guaranteed": not written.
  - "the coupling protects the gearbox": not written, and the guard test passes.
  - "a shaft-joint method is chosen or sized here": not written.
  - "more magnets make the coupling stiffer": not written.
  - Slice-1 spec 8 and T-N are not contradicted.
  - No editable default is typed.
  - No field names a clamp (any case), sources included.
  - Every field passes `plain_text_violations` and `audit_ids`.
  - Sentence 0 answers the Temperature limits row with no placeholder or symbol.
  - The DemagKnee diagram is kept and matches the text.
- Must-fix items and how they were applied:
  - **Round 1:**
    - MF-1, change[0]: NdFeB, "usually", and the weakness relative to N42SH;
    - MF-2, sentences[4]: "each NdFeB ring's limit";
    - MF-3, watch_out[1]: "At their grade's own rating", the exact grade list.
  - **Round 2:** R2-MF-1, sentences[4]: "the grade's rated maximum temperature" became "the magnet's rated maximum temperature", because the N50 arcs' 60 °C rating, not their grade's 80 °C, sets their shift.
  - All four were applied verbatim by the implementer and re-checked in the next round. All pass.
- Verdict: PASS.

Must-fix items applied (rounds 1 and 2, each applied verbatim before the next round, which
confirmed it):
- change[0] (MF-1, round 1: "a grade rated hotter raise this ring's limit; such grades are weaker" was false for Recoma 30, a 2:17 SmCo grade rated 250 °C that lowers the ring's limit to 46.27 °C against 93.06 °C; at the Y30 probe design, where the only grade rated hotter, Recoma 26, lowers the limit from 250 to 161.95 °C and raises the pull-out; and from lower-rated NdFeB grades, such as the library's N42 block to custom N42SH at the same 1.30 T): "custom dimensions and a grade rated hotter raise this ring's limit; such grades are weaker, so torque falls, and the adhesive may then govern." -> "custom dimensions and an NdFeB grade rated hotter usually raise this ring's limit; those above N42SH are weaker, cutting torque, and the adhesive may govern."
- sentences[4], changed twice:
  - MF-2 (round 1: at the Y30 probe design each ring's limit is its 250 °C rating, not its slipping onset less a margin, so the clause is restricted to NdFeB, as sentences 2 and 3 are): "Onsets are shifted to match the grade's rated maximum temperature, and each ring's limit is its slipping onset less a margin; the lower limit governs." -> "Onsets are shifted to match the grade's rated maximum temperature; each NdFeB ring's limit is its slipping onset less a margin; the lower limit governs."
  - R2-MF-1 (round 2: the library's N50 arcs M5044, M5045 and M5026 carry the vendor's 60 °C rating, not N50's 80 °C, and the engine shifts their onsets to match that 60 °C; with "the grade's" here, watch_out[1]'s "At their grade's own rating" read as always true and listed the arcs as shifted upward again): "Onsets are shifted to match the grade's rated maximum temperature; each NdFeB ring's limit is its slipping onset less a margin; the lower limit governs." -> "Onsets are shifted to match the magnet's rated maximum temperature; each NdFeB ring's limit is its slipping onset less a margin; the lower limit governs."
- watch_out[1] (MF-3, round 1: the library's N50 arcs carry the vendor's 60 °C rating, so their onsets are shifted down, by 12.98 °C, while the text listed every N50 magnet as shifted upward; on custom dimensions the list was already exact): "Onsets for N52, N50, N50M, N38UH, N35EH and 2:17 SmCo are shifted upward, so their limits may read optimistic; onsets past a grade's tested range are extrapolated." -> "At their grade's own rating, onsets for N52, N50, N50M, N38UH, N35EH and 2:17 SmCo are shifted upward: limits may read optimistic; onsets past a grade's tested range are extrapolated."

Seeds (A-24): none for this note. The amendment index row for T16 lists A-2, A-13, A-15 and
A-28d and no seed; each round confirmed there was nothing to accept or reject.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-3 PASS):
- SF-2 (round 1), watch_out[0], 29 words: "The reverse fields are stored from the example design's 3D model: they ignore magnet count, ring radius, gap, block size, back iron and grade, reading low for stronger grades." Why: the list reads as complete, and the onsets are also unchanged with the 304 back iron (103.0557 °C at both designs); pin it beside the four moves in `the_reverse_fields_are_the_example_design_s_whatever_the_design` (the 304 design's pull-out differs and its onsets are equal).
- SF-5 (round 1), the DemagKnee diagram (`diagrams.rs:404-476`): it still draws and labels three "load lines", which the rewritten text no longer mentions (the old sentence 3 did). It does not contradict the text: the knee and "H (reverse field)" match sentences 1 to 3. A later GUI pass may draw the knee moving with heat (TC-23) in place of the load lines, or drop the label; not this task's text.
- SF-6 (round 1; this note's MF-1 and MF-3 halves; tests in `tests/explain.rs`, computed from the engine): for MF-1, at the default design `SmCo_2_17_30` on both rings is rated above N42SH yet has a lower `magnet_limit_C` (46.27 °C), every `GradeFamily::NdFeB` grade rated above N42SH has a lower `br_T` than N42SH, and at the Y30 probe design no NdFeB grade is rated above Y30; the doc comment of `a_hotter_rated_grade_raises_its_ring_s_limit_costs_torque_and_alone_may_not_help` should say why the entry names NdFeB. For MF-3, on the example design M5044, M5045 and M5026 on both rings have `model.inner_tmax_C` below their grade's `tmax_C` and a positive `calibration_offset_C` (+12.98 °C), while B842-N52 has its grade's rating and a negative offset (−9.69 °C).
- SF-7 (round 1, for the record, no text change): watch_out[1]'s list holds at the default knee fraction (0.9) and coercivity source, both assumption or choice inputs. At a knee fraction of 0.95, N50 and Recoma 30 leave the list; at 0.8, N42, N48 and N42M join it. BL-057 (E21: the offset clamped at 0) removes the upward shift, and the note text changes in E21's commit, as BL-057's acceptance says. The offsets line in `docs/ai/04-memory.yaml` is stale on four grades (now Recoma 26 −60.21, Recoma 30 −0.45, BCN-19 +94.28 and Recoma 20 +217.33 °C); its recompute is T20's (ruling 5).
- SF-R2-1 (round 2; restated in round 3 as SF-R3-1, with a sharper assert), a test for R2-MF-1 beside the rating-shift assert of `the_demagnetization_note_s_claims_hold_at_every_probe_design`: with M5044, M5045 and M5026 on both rings of the example design, `model.inner_tmax_C` is 60 °C and differs from the N50 grade's 80 °C, `t_ref_model_C − calibration_offset_C` is close to 60 °C, the offset is positive (+12.98 °C) and `inner_magnet_limit_C` is close to −22.50 °C (the grade's rating would give −2.50 °C). It is the only case where "the magnet's" and "the grade's" differ, and it also pins MF-3's exclusion of the arcs.
- SF-R2-2 (round 2; round 1's SF-6 for MF-1, restated from the round-2 probe): pin that every move to a sintered `GradeFamily::NdFeB` grade rated hotter than its start raises that ring's limit (1670 of 1670 moves over every library part and custom grade as starts), and that bonded BCN-19 is the exception "usually" covers (it lowers the limit in 200 of 200).

Considered, no change (the reviewer's observations):
- "here no prediction" at the Y30 design: sentence 3 reads "its onset, here no prediction, sets an NdFeB ring's limit". There is no hot onset there (+inf), and the sentence names NdFeB, so it still reads correctly (spec 5) (round 1).
- "shifted to match ... rated maximum temperature": the onsets are shifted so the reference magnet (permeance coefficient 1, Br/(2μ0) = 517.25 kA/m) reaches its knee at the rating, 159.92 − 9.92 = 150 °C at the example design. The sentence does not say the onsets equal the rating (round 1).
- "reading low for stronger grades": the engine scales the stored fields by each ring's own (1 + αΔT), never by Br at 20 °C, so for a higher-Br grade (N52, 1.45 T, against the example's 1.30 T) the field reads low, which is optimistic, and for weaker grades it reads high (round 1).
- Both rings use the outer blocks' stored fields; the inner ring's lower values appear only in the input help. watch_out[0]'s "ignore ... grade" covers it, and the effect is conservative for the inner ring (round 1).
- The engine help of the Hcj coefficient still says a positive value "is typed in" (`temperature.rs:94`). The GUI overlay replaces it ("ferrite needs the Y30 grade instead", `plain_text.rs:709-711`), so the trap is not on screen with the details off (round 1).
- "cutting torque" compares with N42SH: 241 of the 780 moves to N38UH, N35EH or N33AH raise the pull-out instead, all from magnets weaker or smaller than an N42SH block (the smaller library blocks, the arcs, N35 or BCN-19). The field names N42SH as the comparison and is true at all five designs as rendered, and no 25-word wording fixes the comparison without dropping the torque trade-off that the "a hotter-rated grade is free" trap needs (round 2).
- The SmCo grades follow the same limit rule (Recoma 26 161.95, Recoma 30 46.27, Recoma 20 101.74 °C), but sentence 4 names NdFeB only, as sentence 3 does; that narrowing is not false (round 2).
- "the magnet's" on mixed rings reads per ring, because the next clause says "each NdFeB ring's limit"; the engine shifts each ring by its own rating (the governing ring's shift matched its own rating in 2885 of 2885 cases). "Each magnet's" would also fit the cap but is not needed (round 3).

## ferrite_cold_demag

Task T16. Rewrite of "Ferrite: the demagnetization risk is at cold" as "Ferrite magnets:
demagnetization risk at the cold end", Start here step 13, the last step shown while the
shaft-clamp section is paused. Its Start here bar now adds "(ferrite rings only)" after the step
count (`explorer::FERRITE_ONLY`), since the check it opens reads n/a on NdFeB rings, and the note
drops the NdFeB knee diagram, which shows the knee moving the wrong way for ferrite (R7).
Reviewer: physics reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice
plan fixes for the note's `review` field; all three rounds ran on 2026-10-15).

Rounds: 3. Verdict: PASS. Note: "Ferrite magnets: demagnetization risk at the cold end" (plan
T16).

Verdict by round: 1 MUST-FIX (1 item in this note); 2 PASS for this note (the task's other must-fix
item was in `demagnetization`); 3 PASS. Each round's full answer and the changes files are as
under `## demagnetization`; the two notes were reviewed together. Round 3 re-read this note's
sentence 3 against `demagnetization`'s new sentence 4.

The text reviewed in round 1 already carried A-2 (change[2], the like-poles reverse field:
`Step::Enter`, no aim, "enter your ferrite design's own value from a 3D field calculation; the
stored one is the NdFeB example's, and yours may still fail.") and A-13's plain-words sources,
sources[3] as A-28d item 7 rewords it ("a typed positive coefficient is held within the slider's
negative range", where A-13 had "clamped away"). Every round confirmed them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 1.
- Read:
  - the dump `notes-review-T16-r1.txt`; the `ferrite_cold_demag` literal (`notes.rs:657-710`); `explorer.rs:175-195` (`FERRITE_ONLY`, `start_here_text`);
  - `temperature.rs:729-732` (`cold_onset_C`) and `:793-857` (the positive-coefficient branch: no hot onset; the rating as the hot limit; the cold onsets; the cold limit = skipping cold onset + margin; the pass rule coldest ≥ limit);
  - `records/demagnetization.rs` (each ring's cold limit; the cold ring with the higher limit; `cold_check` passing only if both rings pass);
  - `grades.rs:576-604` (Y30: β +0.0035, α −0.002, Hcj 180 kA/m, rating 250 °C, coefficient range 20 to 120 °C at :587); `metal_design.rs:28-30` (coldest-temperature slider −60 to 20 °C);
  - `input_ui.rs:64-72` (edits clamped to the slider); `plain_text.rs:709-711` (the overlay help: ferrite needs the Y30 grade);
  - `notes.rs` START_HERE_ALL (step 13, `temperature.demag.cold_check`);
  - spec 3.3, 5 and 8; slice-1 spec 8; phys-notes-b 2 and 6; amendments A-2, A-13 and A-28d item 7, ruling 11.
- Numbers reproduced from `compute_all`:
  - The note has no placeholder.
  - At the Y30 design:
    - the hot onsets are all +inf, and each ring's hot limit is 250 °C (its rating);
    - the cold onsets are aligned 170.59, pull-out 312.69, slipping 325.71 and single ring 258.71 °C. Slipping is the highest.
    - the cold limit is 335.71 °C (325.71 + 10) on both rings, with the cold ring inner. `cold_check` = "Below the cold demagnetization limit" at −40 °C, and the verdict is CHECK.
  - At the four NdFeB designs: `cold_limit_C` = n/a and `cold_check` = "n/a (coercivity rises as the magnet cools)".
  - Slider extremes, Y30 on both rings, with the coldest temperature at 20 °C and a 0 °C margin: knee fractions 0.5, 0.7, 0.9 and 1.0 give cold limits of 398.7, 360.1, 325.7 and 309.9 °C. All fail.
  - With "Hcj and beta inputs" as the coercivity source: n/a, and a magnet limit of 157.64 °C (SF-3).
- Change directions confirmed:
  - **change[0], Choose inner part.** Y30 on the inner ring alone (example design):
    - the cold check applies on that ring (cold ring inner, 335.71 °C) and fails;
    - the NdFeB outer ring keeps the hot limit, 93.06 °C;
    - at the Y30 design, N42SH on the inner ring gives that ring the 93.06 °C hot limit, and the check still fails on the ferrite outer ring.
    - So the trade of the cold risk for the hot one holds ring by ring.
  - **change[1], Choose outer part.** Y30 on the outer ring alone: cold ring outer, same limit, fails. The check passes only if both rings pass (the record's two-ring rule).
  - **change[2], Enter like-poles field.** With Y30 on both rings, the entered field sets the cold limit:
    - 431.5 kA/m (half the stored field): 218.46 °C, fails;
    - 150 kA/m: 16.16 °C, fails;
    - 100 kA/m: −50.83 °C, passes;
    - the slider's lowest value passes.
    - So "yours may still fail" holds.
- Traps checked:
  - **"model ferrite by typing a positive β":** not written. Change[0] uses the Y30 grade, and the source says a typed positive value is held within the negative slider range (true: `input_ui.rs:64-72`).
  - **"the ferrite cold check means anything with the default reverse fields":** watch-out 0 says it always fails; MF-4 makes the reason exact.
  - **"your own fields will make it pass":** "yours may still fail".
  - **"the stored 3D fields follow geometry or grade":** not written ("the NdFeB example's").
  - **"a shaft-joint method is chosen or sized here", "the coupling protects the gearbox", "more magnets make the coupling stiffer":** not written.
  - **"clamp" in any field, sources included:** none (A-28d item 7's rewording removed "clamped away").
  - **No editable default is typed:** the test passes.
- Form:
  - title 8 words; sentences 23, 24, 23, 22, 14; changes 22, 15, 23; watch-outs 26, 20, and 29 after MF-4;
  - sentence 0 has no symbol and no placeholder, and it answers Start here step 13 (why a cold check exists);
  - the cold limit is glossed in sentence 2; reverse field, knee and onset are glossed at step 12;
  - no diagram (R7), pinned by `the_demagnetization_notes_own_the_rows_spec_3_3_gives_them`;
  - the Start here bar reads "Start here 13 of 13 (ferrite rings only): <title>" (`the_ferrite_step_says_it_is_for_ferrite_rings_only` passes);
  - every field passes `plain_text_violations` and `audit_ids`.
- Seeds: none indexed for T16.
- Must-fix items: MF-4 (watch_out[0]), to apply in round 2. Should-fix offered: SF-1 (watch_out[1], the extrapolated cold limit); SF-3 (change[0], the coercivity source); SF-4 (sentences[4], "no ferrite ring").
- Verdict for this note, round 1: MUST_FIX (1 item).

Round 2:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 2. The form's fixed date is 2026-10-14; the review ran on 2026-10-15, as in round 1.
- Read:
  - the dump `notes-review-T16-r2.txt` and the r1 to r2 diff;
  - the `ferrite_cold_demag` literal (`notes.rs:657-710`);
  - `temperature.rs:729-732` (`cold_onset_C`) and `:793-857` (the positive-coefficient branch, the cold limit and the pass rule);
  - `grades.rs:576-604` (Y30); `model.rs:417-425` (a graded ring's own Br coefficient);
  - the input schema ranges: coldest temperature −60 to 20 °C, margin 0 to 40 °C, knee fraction 0.5 to 1.0;
  - `tests/explain.rs:3198` (`the_ferrite_note_s_claims_hold`).
- Numbers reproduced from `compute_all`:
  - The note has no placeholder.
  - At the Y30 design:
    - cold onsets: aligned 170.59, pull-out 312.69, slipping 325.71, single ring 258.71 °C;
    - cold limit 335.71 °C on both rings;
    - `cold_check` reads "Below the cold demagnetization limit".
  - The slipping field exceeds the Y30 knee by at least 141.12 kA/m over the knee-fraction slider and from −60 to 250 °C.
  - The aligned field's cold onset ranges from 150.04 to 278.06 °C over the knee-fraction slider.
- Change directions confirmed:
  - Unchanged this round.
  - The cold check fails in all 720 combinations tried: Y30 on both rings, the inner ring or the outer ring; five designs; coldest −60, −40, 0 and 20 °C; margin 0, 10 and 40 °C; knee fraction 0.5, 0.75, 0.9 and 1.0.
  - The calculator's Br coefficient does not move a Y30 ring's cold limit (335.71 °C).
- Traps checked:
  - "the ferrite cold check means anything with the default reverse fields": watch_out[0] says it always fails, and why.
  - "your own fields will make it pass": change[2] says "yours may still fail".
  - "model ferrite by typing a positive beta": not written.
  - "the stored 3D fields follow geometry or grade": not written.
  - "a shaft-joint method is chosen or sized here", "the coupling protects the gearbox", "more magnets make the coupling stiffer": not written.
  - "clamp" in any field, sources included: none.
- Form: watch_out[0] has 29 words; it passes `plain_text_violations` and `audit_ids`, and types no default.
- Must-fix items and how they were applied: round 1's MF-4 (watch_out[0]) was applied verbatim and re-checked: it passes. Round 2: none.
- Verdict for this note, round 2: PASS.

Round 3:

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; round 3. The form's fixed date is 2026-10-14; rounds 1 to 3 ran on 2026-10-15.
- Read:
  - the dumps `notes-review-T16-r1.txt` to `-r3.txt`;
  - the `ferrite_cold_demag` literal (`notes.rs:657-710`); `explorer.rs` (`FERRITE_ONLY`, `start_here_text`);
  - `temperature.rs:729-732` (`cold_onset_C`) and `:793-857`:
    - for a positive coefficient there is no hot onset, and the rating is the hot limit;
    - cold limit = skipping cold onset + margin;
    - the check passes when the coldest temperature is at or above the limit;
  - `records/demagnetization.rs` (each ring's cold limit, the cold ring, the two-ring check);
  - `grades.rs:576-604` (Y30: β +0.0035 /°C, α −0.002 /°C, Hcj 180 kA/m, rated 250 °C, coefficient range 20 to 120 °C at :587);
  - the slider ranges: coldest −60 to 20 °C, margin 0 to 40 °C, knee fraction 0.5 to 1.0; `input_ui.rs:64-72` (edits held within the slider);
  - spec 3.3, 5 and 8; slice-1 spec 8; phys-notes-b 2 and 6; amendments A-2, A-13 and A-28d item 7; ruling 11.
- Numbers reproduced from compute_all:
  - The note has no placeholder.
  - At the Y30 design:
    - the hot onsets are +inf, and each ring's hot limit is 250 °C (its rating);
    - cold onsets: aligned 170.59, pull-out 312.69, slipping 325.71 and single ring 258.71 °C;
    - the cold limit is 335.71 °C on both rings;
    - `cold_check` reads "Below the cold demagnetization limit".
  - At the four NdFeB designs, the cold limit is n/a and the check reads "n/a (coercivity rises as the magnet cools)".
  - The stored slipping field exceeds the Y30 knee by at least 141.12 kA/m over the knee-fraction slider, from −60 to 250 °C.
  - The aligned field's cold onset is 150.04 to 278.06 °C, within the rating, which is why watch_out[0] names the slipping field only.
  - In all 430 positive-coefficient ring evaluations of the round-3 probe, the hot limit is the rating.
- Change directions confirmed:
  - **Choose inner part:** Y30 on the inner ring alone gives that ring the cold check, and it fails; the NdFeB outer ring keeps the 93.06 °C hot limit.
  - **Choose outer part:** the same; the check passes only if both rings pass.
  - **Enter like-poles field:** with Y30 on both rings, the entered field sets the cold limit.
    - 431.5 kA/m gives 218.46 °C, and the check fails; 150 kA/m gives 16.16 °C, and it fails.
    - 100 kA/m gives −50.83 °C, and it passes.
    - So "yours may still fail" holds.
  - With the stored field, the check fails in 720 of 720 combinations of ring placement, design, coldest temperature, margin and knee fraction.
- Traps checked:
  - "model ferrite by typing a positive beta": not written; a typed value is held within the negative slider range.
  - "the ferrite cold check means anything with the default reverse fields": watch_out[0] says it always fails, and why.
  - "your own fields will make it pass": "yours may still fail".
  - "the stored 3D fields follow geometry or grade": not written.
  - "a shaft-joint method is chosen or sized here", "the coupling protects the gearbox" and "more magnets make the coupling stiffer": not written.
  - No field names a clamp (any case), sources included.
  - No editable default is typed.
  - Every field passes `plain_text_violations` and `audit_ids`.
  - There is no diagram (R7), and the Start here bar reads "(ferrite rings only)".
  - Sentence 3's "its rated maximum temperature" agrees with the `demagnetization` note's "the magnet's rated maximum temperature".
- Must-fix items and how they were applied:
  - **Round 1:** MF-4, watch_out[0]: the slipping field alone exceeds a Y30 knee at every rated temperature. It was applied verbatim and re-checked in round 2: it passes.
  - **Rounds 2 and 3:** none.
- Verdict: PASS.

Must-fix items applied (round 1, applied verbatim before round 2, which confirmed it):
- watch_out[0] (MF-4: the rings-aligned default field, 354 kA/m, meets the Y30 knee at 170.59 °C and lies below it from there to the 250 °C rating, so "the default reverse fields ... exceed a Y30 knee at every rated temperature" was false for one of the four; the check uses only the slipping field, whose cold onset, 325.71 °C, lies above the rating): "The default reverse fields belong to the NdFeB example design and exceed a Y30 knee at every rated temperature, so with them the check always fails." -> "The default reverse field for slipping, from the NdFeB example design, exceeds a Y30 knee at every temperature Y30 is rated for, so the check always fails with it."

Seeds (A-24): none for this note.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-3 PASS until the task review fix round 1 below, which changed change[0]):
- SF-1 (round 1), watch_out[1], 28 words: "Y30's temperature coefficients are measured only from 20 to 120 °C; the cold check extrapolates them, both to the coldest temperature and to cold onsets outside that range." Why: the shipped text is true, but what the user sees on the Y30 design is a cold limit of 335.71 °C, from an onset of 325.71 °C extrapolated 200 °C above the tested range; the replacement covers both directions and keeps "from 20 to 120 °C", which `the_ferrite_note_s_claims_hold` reads.
- SF-3 (round 1), change[0], 23 words: "custom dimensions with the Y30 grade, coercivity from the grade, model a ferrite ring; a library NdFeB part trades cold risk for hot." Why: with "Hcj and beta inputs" chosen as the coercivity source, a Y30 ring takes the input coefficient, which is always negative; the cold check then reads n/a and the magnet limit is 157.64 °C, computed as if the ring were NdFeB. The shipped text holds at the default source (the magnet grade) only. If "coercivity" is unwanted on screen (it is not glossed along Start here), leave the text and pin the default source in the claims test instead.
- SF-4 (round 1), sentences[4], 13 words: "With no ferrite ring the cold check does not apply and reads n/a." Why: the check also reads n/a for SmCo and bonded NdFeB (every grade but Y30 has a negative coefficient, which `the_ferrite_note_s_claims_hold` pins); the shipped text is true but narrower than the rule.
- SF-6 (round 1; this note's MF-4 half, a test): at the Y30 design, `cold_onset_aligned_C` (170.59 °C) lies below Y30's `tmax_C` and `cold_onset_skipping_C` (325.71 °C) above it.

Considered, no change (the reviewer's observations):
- With "Hcj and beta inputs" chosen as the coercivity source the check reads n/a rather than failing, and there is then no Y30 knee; watch_out[0]'s subject is "a Y30 knee", so it is not contradicted (round 2). The same choice narrows sentence 3 ("a ferrite ring's hot limit is simply its rated maximum temperature") and sentence 4 ("With NdFeB on both rings ... n/a") to the default source, the magnet grade: a Y30 ring on the input coefficient is treated as NdFeB (hot limit 157.64 °C, cold check n/a). Round 1 offered SF-3 for it (round 3).
- The calculator's Br coefficient input does not reach a Y30 ring: its cold limit is 335.71 °C either way (round 2).
- No library part has a positive coefficient, so a ferrite ring is always Y30 on custom dimensions, and its rating is the grade's; sentence 3's "its rated maximum temperature" agrees with `demagnetization`'s "the magnet's rated maximum temperature" (round 3).
- The engine help of the Hcj coefficient: as under `## demagnetization` (round 1).

### Task review fix round 1 (2026-10-15): the Y30 entry names the torque it costs

- Why: the task code review (`task-16-review.md`, I-1) applied spec 3.1: "Every entry names its
  trade-off when the move pushes another check toward red" (ruling 9 enforced the same rule on
  harmonics change[0]). change[0] named only the trade of the reverse move, back to a library
  NdFeB part. A move to Y30 cuts the torque and pushes the hot-minimum check deep into red, and
  nothing in the note said so. change[1] ("the same for the outer ring") inherited the gap. The
  three in-task rounds passed the old text.
- change[0]:
  - Old: "custom dimensions with the Y30 grade model a ferrite ring; a library NdFeB part trades
    the cold risk for the hot one."
  - New (25 words, at the cap; no placeholder, symbol or typed default; the caps and plain-words
    tests pass): "custom dimensions with the Y30 grade model a ferrite ring, with far less torque; a
    library NdFeB part trades the cold risk for the hot."
  - This is the task review's option A wording ("..., with far less torque; ..."), with the
    reviewed second clause's "a library NdFeB part" kept. Only "one" is dropped, to stay within
    25 words.
  - Superseded by the text re-review's MF-1 below. It is now "custom dimensions with the Y30
    grade model a ferrite ring, with far less torque than same-size library NdFeB parts, which
    trade cold risk for hot." "Custom dimensions (manual)" keeps the manual fields' own size, so
    "far less torque" holds only against a library part of the same size.
- Engine (`compute_all`; each NdFeB probe design's library parts against custom dimensions with
  Y30; scratch probe outside the repo):
  - Pull-out with Y30 on one ring, inner or outer alike: 27.8 % of the NdFeB design's at the
    default, 8-magnet and 20 mm designs (default 2.6885 -> 0.7461 N·m), 25.1 % at the 304 design.
    On both rings: 7.7 % (default 0.2071 N·m), 7.0 % at the 304 design.
  - The hot-minimum check is already red at the example's requirement (hot-low 2.2852 N·m
    against 2.5 N·m), so the move pushes it further in (hot-low 0.6342 N·m with one Y30 ring,
    0.1760 N·m with two). With the requirement at each design's own hot-low, the NdFeB design
    passes and Y30 on either ring, or both, fails.
  - The coercivity source does not enter the torque, so the new words hold with "Hcj and beta
    inputs" chosen too.
- Tests: `the_ferrite_note_s_entries_hold` (`tests/explain.rs`) asserts that change[0] contains
  "far less torque". On the old text it failed ("the Y30 entry names the torque it costs"), and
  on the new text it passes. It also checks, at every NdFeB probe design (four; the Y30 design is
  skipped by its positive coefficient) and for the inner ring, the outer ring and both:
  - Y30 on custom dimensions leaves under half the NdFeB design's pull-out;
  - with the requirement at the NdFeB design's own hot-low, which passes, Y30's hot-low falls
    below it, and `metal.hot_min_check` differs from the NdFeB design's passing verdict.

  The re-review's MF-1 below adds the "same-size" pin, the same-size check for every library
  part and the M5045 check.
- Not changed: SF-3 (the coercivity source, the task review's M-1) stays open; it was not in
  this round's fix list, and "coercivity" is not glossed along Start here. SF-1, SF-4 and SF-6
  stay open as before.
- The dump for the re-review: `NOTES_FOR_REVIEW=ferrite_cold_demag
  NOTES_REVIEW_OUT=target/notes-review-T16-fix1.txt` ("1 notes at 5 designs"). The new entry
  renders identically at the five probe designs, with no `{`, `}`, "no prediction" or "clamp".
- A text re-review of this field follows (controller ruling 7). Its report goes below as
  `### Re-review: code-review fix round 1`.

### Re-review: code-review fix round 1

The reviewer's report for the record follows unchanged. The full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-16-textreview-1.md`. Its scope was
change[0], read together with the fields whose meaning depends on it: change[1] ("the same for
the outer ring; ..."), sentences[3] and [4], and watch_out[0].

- Reviewer: physics reviewer (session model), fresh context, 2026-10-15; text re-review 1 of the fix round's change[0].
- Read: the dump `rereview-T16-1.txt`; `git show 432c836`; `notes.rs:662-718`; `tests/explain.rs:1651` (`on_grade`) and `:3274-3330`; `model.rs:117-150`; `gui/pickers.rs:13-20`; `library.rs:157`; `grades.rs`; `temperature.rs:1062-1110`; `task-16-review.md` (I-1, M-1); the trap lists (both specs' section 8, phys-new-notes T-A to T-P, the added stiffness trap, rulings 8, 11 and 12); the caps and plain-words rules.
- Numbers reproduced from `compute_all` (scratch probe outside the repo):
  - Library parts to Y30 on custom dimensions, at the manual defaults (B842's size):
    - one ring: 27.8 % of the pull-out at the default, 8-magnet and 20 mm designs (default 2.6885 to 0.7461 N·m), 25.1 % at 304;
    - both rings: 7.7 %, and 7.0 % at 304.
    - The record's figures hold.
  - From smaller fitting library parts, the same move:
    - M5045 (6.56 x 5.6 x 1.12 mm): raises the pull-out by x1.067 to x1.275 on the inner ring at the four NdFeB designs; outer ring x0.910 to x1.154;
    - B442: x0.628 to x0.707;
    - B822 (8 magnets) and M5044 (304): x0.526 to x0.554.
  - Same-size Y30 (manual dimensions set to the part's own), across 15 parts, either ring and five designs (150 cases): at most 27.75 % of the part's pull-out.
  - Y30 at the slider's 10 mm thickness: 44.6 % on one ring, 13.8 % on both (default design).
  - Reverse move at the Y30 design (B842SH on the inner ring):
    - pull-out 0.2071 to 0.7461 N·m; cold-high 0.3381 to 1.1369 N·m;
    - the magnet limit 250 to 93.06 °C; the cold ring inner to outer, and the check still fails.
- Must-fix: MF-1, change[0], becomes "custom dimensions with the Y30 grade model a ferrite ring, with far less torque than same-size library NdFeB parts, which trade cold risk for hot." (25 words). Why: "far less torque" is false for a move from a smaller fitting library part (M5045 gives x0.910 to x1.275, a rise in 7 of 8 cases; B442 keeps 63-71 %), because "Custom dimensions (manual)" keeps the manual size. This is ruling 12's same-size defect class.
- Should-fix:
  - SF-A: the coercivity source, carried from round 1's SF-3 and M-1;
  - SF-B: the thermal-mismatch screen turns red with Y30 on both rings (the governing hot end rises to the adhesive's limit, swing 71.06 to 98.00 °C, and the screen uses NdFeB expansion and modulus). Engine scope, for a backlog line beside BL-062.
- Traps: none written; "a hotter-rated grade is free" is guarded by "far less torque".
- Form: 25 words, 1 sentence, plain words, no number, no glyph, no clamp, in both the shipped text and MF-1.
- Verdict: MUST_FIX (1 item).

- MF-1, applied verbatim by the implementer (session model), 2026-10-15:
  - change[0] (input, step and aim unchanged): "custom dimensions with the Y30 grade model a
    ferrite ring, with far less torque; a library NdFeB part trades the cold risk for the hot."
    -> "custom dimensions with the Y30 grade model a ferrite ring, with far less torque than
    same-size library NdFeB parts, which trade cold risk for hot." (25 words, 1 sentence; the
    caps and plain-words tests pass).
  - Test: `the_ferrite_note_s_entries_hold` (`tests/explain.rs`), computed from the engine:
    - change[0] contains "same-size". The fix-round text fails it ("the Y30 entry compares with
      a same-size library part").
    - At every probe design, for every library part on either ring (150 cases), Y30 on custom
      dimensions set to the part's own length, width and thickness: the model's block size on
      that ring equals the part's, and the pull-out is under half the part's.
    - At the example design, M5045 on the inner ring fits by `model::blocks_fit`, its block is
      smaller every way than the manual defaults, and Y30 at the manual defaults does not leave
      under half its pull-out.
    - The doc comment says "far less torque than a same-size library NdFeB part".
  - Docs that quoted the fix-round wording now give the new one: the fix-round subsection above
    (a superseded line, and a pointer to the new checks), the README `tests/explain.rs` row, the
    BL-056 notes and the update tracker.
  - Not applied in this round (the round's list names MF-1 only): SF-A and SF-B. They stay open
    for the controller.

### Re-review: code-review fix round 1 (re-check 2)

The reviewer's report for the record follows unchanged. The full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-16-textreview-2.md`. Its scope was
change[0] as re-review 1's MF-1 left it (`f93dbb6`), "custom dimensions with the Y30 grade model
a ferrite ring, with far less torque than same-size library NdFeB parts, which trade cold risk
for hot.", read together with the fields whose meaning depends on it: change[1] ("the same for
the outer ring; ..."), sentences[3] and [4], and watch_out[0].

- **Reviewer:** physics reviewer (session model), fresh context, 2026-10-15; text re-review 2 of change[0], after re-review 1's MF-1.
- **Read:**
  - the dump `rereview-T16-2.txt`; `git show f93dbb6` and `432c836`; `notes.rs:662-718`;
  - `tests/explain.rs:1651` (`on_grade`) and `:3287` (`the_ferrite_note_s_entries_hold`);
  - `model.rs:117-150`, `:445-503` (`resolve_magnets`), `:583-599` (`select_calibration_factor`) and `:901-1005` (the torque);
  - `grades.rs:576-590`; `calibration.rs:40-42` and `:235`; `api.rs:183-192`; `sizing.rs`;
  - `task-16-textreview-1.md`;
  - the trap lists: both specs' section 8, `phys-new-notes.md` T-A to T-P, the added stiffness trap, and rulings 8, 11 and 12;
  - the caps and plain-words rules.
- **Numbers reproduced from `compute_all`** (scratch probe outside the repo):
  - **Same-size Y30 against every library part, either ring, five probe designs (150 cases):** at most 27.75 % of the part's pull-out, 27.75 % of its hot-low and 29.74 % of its cold-high.
  - **A wider grid** (−40 to 150 °C, shared α −0.0025 to 0, all 8 back irons, 4 to 20 magnets, 9 to 30 mm rings; 80,256 cases): at most 31.88 %.
  - **The reverse trade at the Y30 design:** B842SH on either ring makes that ring's cold limit n/a and its hot limit 93.06 °C (from 250 °C); the other ring keeps 335.71 °C, and the check still fails.
  - **Bench torque at the 304 design** (SF-C), Y30 inner against B842SH:
    - x0.251 at the stored 1.8 N·m (f_cal_updated 1.0495);
    - x0.502 at 0.90 N·m, and x1.507 at 0.30 N·m.
    - The same class hits `demagnetization` change[0]: custom N38UH shows x1.699 at 0.90 N·m.
  - **Mismatch screen** (SF-B), Y30 on both rings at the default, 8-magnet and 20 mm designs:
    - the swing goes from 71.06 to 98.00 °C;
    - the reading goes from "Below" to "Above the lap-shear strength at the block ends";
    - one ring changes neither.
  - **Auto (axial length, 2.5 N·m) at the default design:**
    - NdFeB 13.77 mm; Y30 inner 46.18 mm at the same 2.9412 N·m pull-out;
    - Y30 on both rings not reachable (50.8 mm, 0.9004 N·m).
- **Must-fix:** none.
- **Should-fix:**
  - SF-A: the coercivity source (carried).
  - SF-B: the mismatch screen with Y30 on both rings (carried, re-verified). Engine scope, beside BL-062.
  - SF-C (new): an entered bench torque at about half the stored value or below reverses on-screen cross-part torque comparisons at prototype-matching designs, because the bench factor applies only to B842SH on both rings. This one, `demagnetization` change[0] and others are affected. For the record and a verify-first backlog line; optionally a test pin naming the dependency.
  - SF-D: assert hot-low and cold-high in the same-size loop as well.
- **Traps:** none written. "A hotter-rated grade is free" is guarded by "far less torque". Ruling 12's same-size class is now satisfied.
- **Form:**
  - 25 words (at the cap) and 1 sentence;
  - plain words: `plain_text_violations` and `audit_ids` empty;
  - no number apart from "Y30", no glyph, no placeholder, no typed default or label, and no clamp.
- **Tests:** `the_ferrite_note_s_entries_hold` and `the_ferrite_note_s_claims_hold` pass; the lib `notes` suite has 33 passed and 4 ignored.
- **Verdict:** PASS.

- Recorded by the implementer (session model), 2026-10-15. The verdict is PASS, so no code, test
  or on-screen text changed in this step; change[0] stands as re-review 1's MF-1 left it.
  - The full answer's SF-C suggests a record line under this note. Here it is: the "far less
    torque than same-size library NdFeB parts" comparison holds at the stored bench inputs. The
    bench factor applies only to B842SH on both rings, in a free-space circuit (no magnetic back
    iron, as on the 304 design) and at the prototype's magnet count, so an entered bench torque
    near half the stored 1.8 N·m or below can reverse the on-screen comparison at such a design
    (Y30 on the inner ring keeps x0.502 of B842SH's pull-out at 0.90 N·m on the 304 design).
  - Left open for the controller: SF-A, SF-B, SF-C's optional test pin and its verify-first
    backlog line (cross-part torque comparisons at prototype-matching designs depend on the
    entered bench torque; `demagnetization` change[0] is in the same class), and SF-D.

## a5.non_ferromagnetic_back_iron

Task T17. Rewrite of "Non-ferromagnetic back iron" as "No magnetic steel behind the magnets",
the teaching note of the open-circuit warning (the first of `WARNING_RULES`). It keeps its
equation, the circuit in effect (`materials.circuit_backiron`, which the panel's warning test
clicks), and the FluxPathBackIron diagram. Reviewer: physics reviewer (session model), fresh
context. Date: 2026-10-14 (the date the slice plan fixes for the note's `review` field; both
rounds ran on 2026-10-15).

Rounds: 2. Verdict: PASS. Note: "No magnetic steel behind the magnets" (plan T17).

Verdict by round: 1 MUST-FIX (4 items in this note, 13 across the task's six notes); 2 PASS. The
six warning notes were reviewed together. Each round's full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-17-inreview-r<N>.md`, and the
fields that round 1 changed are listed with their old and new text in
`task-17-inreview-r1-changes.md` beside it. Round 2 re-checked only those fields and the fields
whose meaning depends on them.

The text reviewed in round 1 already carried A-13's plain-words sources and A-28e item 1's
joint-neutral change[2] (the axial magnet length: "... and shaft-joint and gearbox loads rise
with the cold-high torque.", in place of "the cold-high torque and clamp demand rise"; no field
names a clamp while the shaft-clamp section is paused, ruling 11). Every round confirmed them
applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

Round 1, 2026-10-15, physics reviewer (session model).
Checked:
- The fields, at the five probe designs: the title; 5 sentences; 3 change entries; 1 watch-out; 4 sources.
- The engine behind them:
  - `warnings.rs:144-172`;
  - `api.rs:337-353`;
  - `material_library.rs:731-749` and `:868-909`;
  - `model.rs:584-599` and `:1029-1030`;
  - `materials.rs:243-249`;
  - the `model.f_cal` and pull-out records in `explain/records/torque.rs`;
  - the R11 tests.
- The documents: spec 3.1-3.3 and 8; slice-1 spec 8; phys-new-notes 2; phys-notes-c 1 and 7; A-13; A-24 S-10; A-28e.
- Sentence 0 is plain ASCII, 25 words, and answers the warning text (`warnings.rs:61`).
- The sources are A-13's, verbatim. No field names a clamp.
- Change[2] is A-28e's text, verbatim.
Numbers reproduced:
- {model.pullout_iron_Nm} = 2.688, 1.968, 2.688, 0.2071, 2.539 N·m.
- {model.pullout_noiron_Nm} = 1.537, 1.001, 1.697, 0.1183, 1.205 N·m.
- `model.f_cal` = 0.95, 0.95, 1.0495, 0.95, 0.95.
- No-iron/iron = 0.57, 0.51, 0.63, 0.57, 0.47. Over fitting designs at 8 to 20 mm ring radius and 4 to 16 magnets, it is 0.45 to 0.60, so "much less" holds.
Directions probed:
- **change[2], axial length.** The input is blank at all five designs. It is seeded at the inner length, 12.70 mm, and stepped to 12.71 mm.
  - Pull-out, up at all five: 2.6885 to 2.6908, 1.9684 to 1.9701, 1.6974 to 1.6989, 0.20707 to 0.20726, 2.5393 to 2.5417 N·m.
  - Cold-high at the gearbox input, up at all five: 0.8049 to 0.8056, 0.5893 to 0.5898, 0.5082 to 0.5086, 0.07117 to 0.07124, 0.7602 to 0.7610 N·m.
  - Mass up at all five.
- **change[0], from the 304 design.**
  - 4140: pull-out 2.688 N·m; cold-high input 0.508 to 0.805 N·m; the check "Too thin ... 2.0 mm"; slip loss 0.924 to 2.751 W; mass 176.48 to 173.79 g.
  - 1018: the check OK; slip loss 3.024 W; time to the limit (high case) "never" to 735.9 s.
  - From 6061: mass 96.81 to 173.79 g.
- **change[1], circuit 0 to 1 with 4140.**
  - Pull-out: 1.697 to 2.688 (default), 1.001 to 1.968 (8 magnets), 0.1183 to 0.2071 (Y30), 1.331 to 2.539 N·m (20 mm).
  - Mass 96.81 to 173.79 g at the default, and the check returns.
  - The entry is held at the 304 design.
Must-fix:
1. **sentences[2].** Current: "That model keeps much less torque: on this design the same layout gives {model.pullout_iron_Nm} with steel and {model.pullout_noiron_Nm} without." Replacement: "That model keeps much less torque: on this design the same layout gives {model.pullout_iron_Nm} with unsaturated steel and {model.pullout_noiron_Nm} without." Reason: spec 8, steel torque without the saturation caveat. `model.rs:1029` is ideal steel, and the wall check reads "Too thin" at three designs.
2. **change[0].** Current: "4140 or 1018 restore the steel circuit; both need plating, and 12L14, 416 or 17-4PH trip the low-saturation caution." Replacement: "4140 or 1018 restore steel torque but bring back the cup-wall check, plating, steel's mass and slip heat, and higher shaft-joint and gearbox loads." Reason: spec 3.1. From 304, cold-high input goes from 0.508 to 0.805 N·m, past the 0.6 rating; the wall check returns as "Too thin"; slip loss triples. From 6061 the mass nearly doubles.
3. **change[1].** Current: "the steel circuit, once a magnetic steel is picked; the hub, cup and boss become steel, so the coupling gets heavier." Replacement: "the steel circuit, once a magnetic steel is picked, restores torque but adds mass, brings back the cup-wall check and raises shaft-joint and gearbox loads." Reason: the loads and the returning check went unnamed (`api.rs:347-351`, `materials.rs:243-249`).
4. **watch_out[0]** (seed S-10, amended). Current: "Only designs with the bench prototype's pole count and B842SH on both rings get its measured calibration; elsewhere the assumed factor applies, so the torque shown jumps." Replacement: "Only no-back-iron designs with the bench prototype's magnets per ring and B842SH on both rings get its measured calibration; elsewhere the assumed factor applies, so the torque shown jumps." Reason: false at the default design (f_cal 0.95; `model.rs:584-599`), which is the slice-1 bench-calibration trap.
Should (not blocking): sentences[4], "the expansion-mismatch caution usually fires". A 304 hub stops firing with the NdFeB expansion above about +1.9e-6 (ruling 5).
Verdict: MUST-FIX

Round 2:

Round 2, 2026-10-15, physics reviewer (session model).
Checked:
- The fields changed by round 1, at the five probe designs: sentences[2] (MF-1), change[0] (MF-2), change[1] (MF-3), watch_out[0] (MF-4); and sentence 4, which their "bring back the cup-wall check" answers. The title, sentences 0, 1, 3 and 4, change[2] (A-28e, verbatim) and the four sources are unchanged from round 1 and stand.
- The engine behind them: `model.rs:584-599` (the calibration rule) and `:1027-1030` (the steel and no-back-iron torques); `material_library.rs:880-909` (circuit, design flux density, body material); `api.rs:337-353` (the rules' inputs); `materials.rs` and `records/dashboard.rs:64-68` (the cup-wall check and its "No back iron"); `temperature.rs:1064-1111` (the bond screen); `records/torque.rs:76-87`; the R11 tests.
- The traps: spec 8 (steel torque without the saturation caveat; 416, 17-4PH or 12L14 clean swaps; the coupling protects the gearbox; a shaft-joint method chosen or sized here); slice-1 spec 8 (the bench calibration beyond prototype-matching designs); T-G, T-L, T-O; N1, N2. None is written.
- Form: every field within its cap (sentences 25, 24, 20, 22, 20 words; changes 24, 25, 25; watch-out 29); `plain_text_violations` and `audit_ids` empty for every field, sources included; no field names a clamp.
Numbers reproduced:
- {model.pullout_iron_Nm} = 2.688, 1.968, 2.688, 0.2071, 2.539 N·m.
- {model.pullout_noiron_Nm} = 1.537, 1.001, 1.697, 0.1183, 1.205 N·m. No-iron/iron = 0.57, 0.51, 0.63, 0.57, 0.47.
- The steel torque does not read the wall: with the wall at 0.5 mm, or one step lower in wall or design flux density, the pull-out is unchanged at all five designs (2.688476, 1.968360, 1.697369, 0.207074, 2.539258 N·m).
- `model.f_cal` = 0.95, 0.95, 1.0495, 0.95, 0.95. In the no-back-iron states (304, 6061, circuit off) it is 1.0495 at the default, 304 and 20 mm designs, and 0.95 at 8 magnets and at Y30 (parts cleared). The 304 default at 12 magnets gives 0.95 (pull-out 1.697 to 1.871 N·m).
Directions probed:
- **change[0], from 304, pick 4140** (1018 where it differs):
  - default and 304 designs: pull-out 1.697 to 2.688; cold-high at the gearbox input 0.508 to 0.805 N·m; check "No back iron" to "Too thin ... 2.0 mm" (1018 "OK"); slip loss 0.924 to 2.751 W (1018 3.024 W, time to the limit "never" to 735.9 s); mass 176.48 to 173.79 g (1018 174.09 g); peak end shear 15.76 to 11.68 MPa.
  - 8 magnets: pull-out 1.001 to 1.968; input 0.300 to 0.589 N·m; check to "Too thin ... 2.4 mm" (1018 2.2 mm); slip 0.826 to 2.870 W; time to the limit "never" to 1049.6 s (1018 607.9 s); mass 177.47 to 174.59 g.
  - Y30: pull-out 0.1183 to 0.2071; input 0.041 to 0.071 N·m; check to "OK"; slip 0.924 to 2.751 W; mass 163.70 to 161.01 g.
  - 20 mm: pull-out 1.331 to 2.539; input 0.399 to 0.760 N·m; check to "Too thin ... 3.3 mm" (1018 2.9 mm); slip 2.407 to 9.777 W; time to the limit "never" to 187.1 s; mass 354.95 to 348.64 g.
  - From 6061 the same pull-out, input and check; mass 96.81 to 173.79, 92.05 to 174.59, 84.03 to 161.01, 168.21 to 348.64 g; slip 2.918 to 2.751, 3.050 to 2.870, 2.918 to 2.751, 13.24 to 9.78 W.
  - At zero plating, 4140 and 1018 fire the plating caution at all five designs; 304 and 6061 do not.
- **change[1], circuit 0 to 1** with 4140, 1018, 12L14, 416 or 17-4PH H900 picked: pull-out restored (1.697 to 2.688, 1.001 to 1.968, 0.1183 to 0.2071, 1.331 to 2.539 N·m), cold-high at the input up, the check back, mass up at every design (+74 to +180 g). Held at the 304 design (set by the pick).
- **change[2], axial length** (unchanged, A-28e): the dump shows the pull-out up at all five (2.6885 to 2.6908 at the default).
Must-fix: none.
Should (not blocking): change[1] becomes "the steel circuit, with magnetic steel picked, restores torque but adds mass over aluminium, brings back the cup-wall check and raises shaft-joint and gearbox loads." (25 words): at the 304 design, where the entry is held, the path from 304 to steel lowers the mass 1.4 to 1.8 %. sentences[4]: "fires" becomes "usually fires" (ruling 5).
Verdict: PASS

Earlier rounds: 4 must-fix item(s) in round 1, applied as given (the old and new texts are in
round 1's Must-fix above; round 2 confirmed each):
- sentences[2]: "That model keeps much less torque: on this design the same layout gives {model.pullout_iron_Nm} with unsaturated steel ..."
- change[0] (`materials.parts.back_iron`, Choose): "4140 or 1018 restore steel torque but bring back the cup-wall check ..."
- change[1] (`coupling.backiron`, Choose): "the steel circuit, once a magnetic steel is picked, restores torque but adds mass ..."
- watch_out[0] (seed S-10, amended): "Only no-back-iron designs with the bench prototype's magnets per ring ..."

Seeds (A-24):
- S-10 (watch_out[0]: "Only designs with the bench prototype's magnets per ring and B842SH on both rings get its measured calibration; elsewhere the assumed factor applies, so the torque shown jumps."): accepted in round 1 with one change and applied as MF-4; round 2 confirms it in place. Reason: the seed's "magnets per ring" is right (the on-screen count, where "pole count" is ambiguous). But as seeded, and as drafted, the line still told the default design, which has 10 magnets per ring, B842SH on both rings and steel, that it gets the measured calibration. Only no-back-iron designs do: `model.f_cal` is the assumed 0.95 at the default and the bench 1.0495 only at the 304 design (`model.rs:584-599`). That is the slice-1 trap "the bench calibration applies beyond prototype-matching designs". The applied text adds "no-back-iron" and is true at all five designs.

Controller ruling 5 (T17 Q4: the reviewer may soften sentences[4]'s "the expansion-mismatch
caution fires" to "usually"): offered as SF-1 in rounds 1 and 2 and not applied, because the
apply step takes must-fix items only. It stays open for the controller below.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS):
- SF-1 (rounds 1 and 2; ruling 5), sentences[4], 21 words: "The hub, cup and boss then become non-magnetic metal, the cup-wall check no longer applies, and the expansion-mismatch caution usually fires." Why: with "NdFeB expansion in the bond plane" above about +1.9e-6 /°C (its slider runs to +6e-6), a 304 hub no longer fires the caution; 6061, and the circuit switched off, fire it everywhere on the slider.
- SF-8 (round 2), change[1], 25 words: "the steel circuit, with magnetic steel picked, restores torque but adds mass over aluminium, brings back the cup-wall check and raises shaft-joint and gearbox loads." Why: at the 304 design the entry is held ("set by the Back iron material (hub, cup and boss) pick"). Its only free move, the circuit from 0 to 1 with a magnetic steel picked, starts from the aluminium body (`material_library.rs:905-909`) and adds mass at all five designs (+74 to +180 g), so the shipped entry is true for the move it offers, which is why this is not blocking. But a reader at the 304 design who reads it as the whole path from 304 meets the opposite: 176.48 to 173.79 g (4140) or 174.09 g (1018) at the default, and a 1.4 to 1.8 % fall at every design, because 304 is the denser metal. Round 1 made change[0] neutral ("steel's mass") for the same reason (MF-2).

Considered, no change (the reviewer's observations):
- "The torque shown jumps" (watch_out[0]) is vague but true; sizing the step would need a typed factor ratio (round 1).
- At the 304 design the no-iron torque carries the bench factor while the steel torque carries the assumed one; watch_out[0] covers this (round 1).
- Identical tails: change[0] and `a5.cte_mismatch_with_magnets` change[0] share their trade-off clauses word for word, and so do the two notes' change[1]. The two warnings fire together at every state probed (304, 6061, the circuit off). Each note opens alone from its own warning and the claims are the same physics, so the repetition is harmless; trimming one would leave that note's entry without its trade-offs (spec 3.1) (round 2).
- "Restore steel torque" without the circuit: with the circuit switched off, a pick alone restores nothing (4140 or 1018 at circuit 0 leave every result unchanged); change[1] covers that state (rounds 1 and 2).
- A magnetic pick other than 4140 or 1018 with the circuit switched on (change[1]'s "a magnetic steel") also fires the low-saturation caution for 12L14, 416 and 17-4PH. The entry calls no steel a clean swap, the returning cup-wall check it names is where that caution belongs, and the low-saturation and plating notes name those picks (round 2).
- The bench factor at another size: at the 20 mm design a 304 pick gets the bench factor (1.0495), since the rule checks only the circuit, the count and the part. watch_out[0] states the rule exactly and claims no validity, so the slice-1 calibration trap is not written; the calibration note owns the caveat (round 2).

## a5.ferromagnetic_sleeve_or_liner

Task T17. Rewrite of "Ferromagnetic sleeve or liner" as "Magnetic sleeve or liner", the
teaching note of the magnetic-shell warning. It got the shortest rewrite (R13): every listed
sleeve and liner material is non-magnetic, so the warning cannot fire, and the note has no change
entry, watch-out, equation or diagram. Reviewer: physics reviewer (session model), fresh context.
Date: 2026-10-14 (the date the slice plan fixes for the note's `review` field; both rounds ran on
2026-10-15).

Rounds: 2. Verdict: PASS. Note: "Magnetic sleeve or liner" (plan T17).

Verdict by round: 1 PASS; 2 PASS (no field of this note changed). The six warning notes were
reviewed together; each round's full answer is as under `## a5.non_ferromagnetic_back_iron`.
Round 2 re-checked this note only for dependence on the fields round 1 changed in the other notes
(none) and for the form checks (all clean), and kept its round-1 section as the final-round record.

The text reviewed in round 1 already carried A-13's plain-words sources. Every round confirmed
them applied.

The reviewer's report for the record, as returned (round 1; round 2 reproduced it verbatim):

Round 1, 2026-10-15, physics reviewer (session model).
Checked:
- The fields, at the five probe designs: the title; 3 sentences; 0 change entries (R13); 0 watch-outs; 3 sources.
- The engine behind them:
  - `warnings.rs:148`;
  - `api.rs:339`;
  - the four sleeve choices (`material_library.rs:744-749`), all with `ferromagnetic: false`;
  - the R11 sleeve test.
- Traps N1 and N9 are not written. Sentence 2 says the warning cannot fire and the torque model has no term for a magnetic shell. Both are true: the rule reads only the library flag.
- Sentence 0 is plain, 18 words, and sets up the warning text's gap-flux short circuit, which sentence 1 completes.
- The sources are A-13's, verbatim. The third source's B_sat · t cap is right physics. No field names a clamp.
Numbers reproduced: none (no placeholders). No rule fires for any sleeve choice at any of the five designs.
Directions probed: none (no entries).
Must-fix: none.
Should (not blocking): none.
Verdict: PASS

Seeds (A-24): none for this note.

Should-fix offered: none.

Considered, no change (the reviewer's observation): sentence 0 sets the scene and sentence 1
answers it. This is the R13 shortest form (round 1).

## a5.high_conductivity_sleeve_or_liner

Task T17. Rewrite of "High-conductivity sleeve or liner" as "Sleeve or liner that conducts better
than 316L", the teaching note of the conductivity caution. Beside the sleeve and liner slip losses
(`temperature.slip_loss.sleeve_W`, `temperature.slip_loss.liner_W`) it now also owns the sleeve
conductivity in effect, which the caution compares (`materials.sleeve_sigma_S_m`; spec 3.3,
`the_wall_and_sleeve_notes_own_the_rows_slice_2_gives_them`). Reviewer: physics reviewer (session
model), fresh context. Date: 2026-10-14 (the date the slice plan fixes for the note's `review`
field; both rounds ran on 2026-10-15).

Rounds: 2. Verdict: PASS. Note: "Sleeve or liner that conducts better than 316L" (plan T17).

Verdict by round: 1 PASS; 2 PASS (no field of this note changed). The six warning notes were
reviewed together; each round's full answer is as under `## a5.non_ferromagnetic_back_iron`.
Round 2 re-checked this note only for dependence on the fields round 1 changed in the other notes
(none) and for the form checks (all clean), and kept its round-1 section as the final-round record.

The text reviewed in round 1 already carried A-13's plain-words sources. Every round confirmed
them applied. A-25 T-7 (`the_sleeve_entry_at_the_bottom_of_its_slider_is_plain_text_with_the_reason`,
`src/gui/explorer.rs`) draws the sleeve entry with the sleeve at its slider minimum as plain text,
" (at the bottom of its range)", with no link, while the liner entry beside it stays a link; it
passed in both rounds.

The reviewer's report for the record, as returned (round 1; round 2 reproduced it verbatim):

Round 1, 2026-10-15, physics reviewer (session model).
Checked:
- The fields, at the five probe designs: the title; 5 sentences; 3 change entries; 1 watch-out; 3 sources.
- The engine behind them:
  - `warnings.rs:39` and `:149` (a constant 1.35e6 S/m);
  - `api.rs:180` (the shells take the pick's conductivity);
  - `temperature.rs:1174-1180` (the shell loss: end factor × conductivity × t × (ω r)² × B² / 2 × 2π r L), with ω the slip speed (`temperature.rs:914`);
  - the R11 sleeve test: 316L typed above the threshold fires; another pick does not.
- Sentence 0 glosses the eddy currents ("cut through the other ring's field ... the eddy currents this drives").
- Traps N3 and N12 are not written.
- The sources are A-13's, verbatim. No field names a clamp.
Numbers reproduced:
- {temperature.slip_loss.sleeve_W} = 0.07470, 0.07470, 0.07470, 0.07470, 0.3696 W.
- {temperature.slip_loss.liner_W} = 0.1805, 0.1805, 0.1805, 0.1805, 0.8483 W.
- {temperature.slip_loss.total_W} = 2.751, 2.870, 0.9240, 2.751, 9.777 W.
- The shells make 9.3, 8.9, 27.6, 9.3 and 12.5 % of the total.
- The conductivity in effect is 1.35e6 S/m at all five designs.
Directions probed:
- **change[1], sleeve 0.10 to 0.09 mm.**
  - sleeve_W: 0.07470 to 0.06716 W at four designs; 0.3696 to 0.3324 W at 20 mm. Down at all five; the ratio 0.899 against 0.9 is "in proportion".
  - Running clearance +0.010 mm at all five; mass down.
- **change[2], liner 0.20 to 0.19 mm.**
  - liner_W: 0.1805 to 0.1717 W at four designs; 0.8483 to 0.8063 W at 20 mm. Down at all five; the ratio is 0.951 against 0.95.
  - Running clearance +0.010 mm at all five.
- **change[0], material, at the default.**
  - Ti, IN625, PEEK: sleeve 0.0329, 0.0429, 0 W; liner 0.0796, 0.1037, 0 W; total 2.609, 2.643, 2.496 W.
  - Mass 169.41, 174.33, 165.61 g.
  - No check moves toward red at any design.
- At the slider minimum the sleeve entry draws "(at the bottom of its range)" (A-25 T-7 passes).
Must-fix: none.
Should (not blocking):
- watch_out[0] becomes: "The field at each shell is an entered value from the example design's 3D model; it does not follow your gap, ring radius, magnet grade, size or count." The stored fields do not follow the ring radius either.
- A-27: the entries that duplicate `slip_heating` stay. Each note opens alone, the claims agree, and the note needs 1 to 5 entries.
Verdict: PASS

Seeds (A-24): none for this note.

Prose duplicates (A-27): no trim, in rounds 1 and 2. `slip_heating`'s sleeve-and-liner material,
sleeve and liner entries (reviewed in T15) say what this note's three entries say, in other words.
Both stay: each note opens alone from its own links (the slip-loss rows and Start here step 10;
the conductivity caution), the claims agree (lower conductivity and a thinner shell heat less, and
strength is not checked), and removing them would leave this note with no entries (spec 3.1 needs
1 to 5). T15's record (under `## slip_heating`) suggested that this note point at those levers
rather than restate them; the reviewer kept both for these reasons. `slip_heating` did not change.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-1 PASS):
- SF-5 (rounds 1 and 2), watch_out[0], 28 words: "The field at each shell is an entered value from the example design's 3D model; it does not follow your gap, ring radius, magnet grade, size or count." Why: the stored shell fields do not follow the ring radius either (at the 20 mm design the shell loss moves only through radius and speed: 0.3696 and 0.8483 W), and "count" replaces "pole count", for the on-screen magnets per ring.

Considered, no change (the reviewer's observation): Inconel 625 adds 0.54 g (174.33 against
173.79 g at the default). That is negligible (round 1).

## a5.low_saturation

Task T17. Rewrite of "Low saturation" as "Back-iron wall and saturation", the teaching note of the
low-saturation caution and now also the Why of the dashboard's Cup wall check row
(`CUP_WALL_GUIDE`). It owns the wall rule's rows (spec 3.3): the design flux density in effect
(`model.bsat_T`), the gap flux density (`model.gap_flux_density_T`), the wall the rule asks for
(`model.backiron_needed_mm`), the suggested wall (`materials.cup_wall_suggested_mm`) and the check
(`materials.cup_wall_check`). No note owns the cup's outside diameter (`model.cup_od_mm`).
Reviewer: physics reviewer (session model), fresh context. Date: 2026-10-14 (the date the slice
plan fixes for the note's `review` field; both rounds ran on 2026-10-15).

Rounds: 2. Verdict: PASS. Note: "Back-iron wall and saturation" (plan T17).

Verdict by round: 1 MUST-FIX (5 items in this note, 13 across the task's six notes); 2 PASS. The
six warning notes were reviewed together; each round's full answer and the round-1 changes file
are as under `## a5.non_ferromagnetic_back_iron`. Round 2 re-checked only the changed fields and
the fields whose meaning depends on them.

The text reviewed in round 1 already carried A-13's plain-words sources. Every round confirmed
them applied. A-25 T-4 (`the_wall_note_s_design_flux_entry_names_the_pick_that_sets_it`,
`src/gui/explorer.rs`, Review Focus RF-4): with the 1018 back iron, which brings its own design
flux density, the entry on "Design flux density for the back-iron check" is plain text, "set by
the Back iron material (hub, cup and boss) pick"; with 12L14 it is a link. It passed in both
rounds, after MF-8's deletion too.

The reviewer's report for the record, round by round, as returned:

Round 1:

Round 1, 2026-10-15, physics reviewer (session model).
Checked:
- The fields, at the five probe designs: the title; 5 sentences; 4 change entries; 2 watch-outs; 4 sources; 5 owned equations (the ownership test passes).
- The engine behind them:
  - `warnings.rs:150-153`;
  - `api.rs:342-345`;
  - `material_library.rs:879-886` (the design flux density in effect);
  - `model.rs:964` (cup OD) and `:1034-1040` (B_gap, the wall rule);
  - `materials.rs:243-249` (the check and its "No back iron");
  - the wall records in `explain/records/dashboard.rs`;
  - `guidance.rs:184-190` (the Cup wall guide) and `:274-275` (12 magnets do not fit the default ring).
- The documents: the math audit M1 (`2026-09-29-magcoupling-math-audit.md:57`); phys-new-notes T-I, T-K and T-L; phys-notes-c section 4 and N4, N7, N8 and N11.
- R11: on the steel circuit the caution fires for 12L14, 416, 17-4PH H1150 and 17-4PH H900 only.
- Sentence 0 is plain ASCII, 25 words, and answers the Cup wall check row and the caution.
- The sources are A-13's, verbatim. No field names a clamp.
Numbers reproduced:
- {model.backiron_needed_mm} = 1.919, 2.399, 1.919, 0.5326, 3.267 mm.
- {model.bsat_T} = 1.500 T at all five designs.
- B_gap = 1.0265 T, except 0.2849 T at Y30.
- Pole pitch = 8.809, 11.011, 8.809, 8.809, 14.998 mm.
- The check reads "Too thin ... 2.0 mm", "Too thin ... 2.4 mm", "No back iron", "OK", "Too thin ... 3.3 mm". The suggested wall is 2.0, 2.4, n/a, 0.6, 3.3 mm.
Directions probed:
- **change[0], wall 1.80 to 1.85 mm.** Cup OD +0.100 mm at all five: 41.326 to 41.426 at the default, 304 and Y30 designs; 42.436 to 42.536 at 8 magnets; 62.040 to 62.140 at 20 mm. Mass up.
- **change[1], design flux density 1.50 to 1.49 T.** The wall needed rises at all five: 1.9189 to 1.9318, 2.3986 to 2.4147 (the suggestion goes 2.4 to 2.5), 1.9189 to 1.9318, 0.5326 to 0.5361, 3.2671 to 3.2890 mm. With 1018 the entry is plain text, "set by the ... pick" (A-25 T-4 passes).
- **change[2], magnets +2.** The wall needed falls at all five: 1.919 to 1.599, 2.399 to 1.919, 1.919 to 1.599, 0.533 to 0.444, 3.267 to 2.723 mm.
  - 12 magnets do not fit at the default, 304 and Y30 designs.
  - The check flips from "Too thin" to "OK" at the default.
  - The pull-out rises 11 to 64 %, and cold-high at the gearbox input rises at all five.
- **change[3].** 4140 and 1018 clear the caution (R11).
Must-fix:
1. **sentences[1].** Current: "The needed wall grows with the gap flux density and the pole pitch, and shrinks as the design flux density, set below saturation, rises." Replacement: "The calculator's rule sizes the wall for half a pole's flux at the design flux density, a working level set below the steel's saturation." Reason: trap T-I. The rule (`model.rs:1040`) scales with the pole pitch, while the real flux follows block width (M1). Check g: the terms are unglossed.
2. **sentences[2].** Current: "Wider poles, from fewer magnets or a larger ring, and stronger magnets need more wall; here a steel cup would need {model.backiron_needed_mm}." Replacement: "So the rule asks for more wall with stronger magnets or wider poles, from fewer magnets or a larger ring; here it gives {model.backiron_needed_mm}." Reason: T-I names this claim as the rule's; M1's flat-top need with the same blocks is 2.16 mm at any count.
3. **sentences[4]** (seed S-11, amended). Current: "Here the cup-wall check uses {model.bsat_T}; for those steels that is the entered value, 4140's by default, which their data may not support." Replacement: "Here the rule uses {model.bsat_T}; for those steels that is the entered design flux density, which their own data may not support." Reason: false at the 304 design, where the check reads "No back iron" (`materials.rs:243-249`). "4140's" sits beside trap N4.
4. **change[2]** (`coupling.npole`, Raise). Current: "narrower poles need a thinner wall; the pull-out torque changes too, and the blocks must still fit around the ring." Replacement: delete the entry. Reason: T-I. At the default, 12 magnets do not fit, yet the check turns OK while the real need (about 2.0 to 2.2 mm) stays above the 1.8 mm wall. The rise in pull-out and in the gearbox load is unnamed.
5. **watch_out[0].** Current: "The torque shown assumes ideal steel: a thin or saturating wall shows only in the cup-wall check, while the real torque would fall below the prediction." Replacement: "With steel, the torque shown assumes it never saturates: a thin or saturating wall shows only in the cup-wall check, while the real torque would fall below the prediction." Reason: false at the 304 design, where the torque shown has no steel (`model.rs:1030`).
Should (not blocking):
- change[3] becomes: "4140 or 1018 clear the low-saturation caution but heat more when slipping than 416 or 17-4PH; both rust, so keep plating above zero." With 1018 the time to the limit goes from "never" to 735.9 s.
- watch_out[1]: "exact 2D sections" becomes "exact 2D field solutions".
Verdict: MUST-FIX

Round 2:

Round 2, 2026-10-15, physics reviewer (session model).
Checked:
- The fields changed by round 1, at the five probe designs: sentences[1] (MF-5), sentences[2] (MF-6), sentences[4] (MF-7), the deleted `coupling.npole` entry (MF-8), watch_out[0] (MF-9); and the fields that lean on them: sentence 3 (uses "design flux density" after sentence 1 glosses it), change[1] (the "Design flux density for the back-iron check" input), watch_out[1] ("the rule is approximate"). The title, sentence 0, change[0], change[2] (was change[3]), watch_out[1] and the four sources are unchanged from round 1 and stand.
- The engine behind them: `model.rs:1033-1040` (B_gap and the wall rule); the record `records/dashboard.rs:53-68` ("Half a pole's flux, B_gap τ_p / π per unit length, carried by the wall at B_des"); `material_library.rs:882-886` (the design flux density in effect) and the library (12L14, 416, 17-4PH H1150 and H900 have no design flux density of their own; 1018 has 1.7 T; 4140's saturation is not sourced); `warnings.rs:150-153`; the torque path (`model.rs:1027-1030`), which never reads the wall.
- The traps: T-I (the rule's dependence on pole pitch is the rule's, now written "the rule asks for"), T-K and N4 (no 4140 saturation; the design flux density is "a working level set below the steel's saturation"), T-L and N1 (the shown torque never falls; the real one would), N11 (watch_out[1] keeps "approximate"), T-P (no outside-diameter claim), slice-1 "fewer poles keep the same size" (sentence 2 says the rule asks for more wall with fewer magnets), and the Global Constraints trap "more magnets make the coupling stiffer" (no stiffness claim). None is written.
- Form: sentences 25, 24, 24, 25, 22 words; changes 19, 21, 16 (three entries: within 1 to 5); watch-outs 29, 27; every field clean of banned patterns, audit ids and "clamp", sources included. Each placeholder sits in a "here" clause.
- The ownership test `the_wall_and_sleeve_notes_own_the_rows_slice_2_gives_them` and A-25 T-4 pass after MF-8.
Numbers reproduced:
- {model.backiron_needed_mm} = 1.919, 2.399, 1.919, 0.5326, 3.267 mm; equal to B_gap·τ_p/(π·{model.bsat_T}) to 0 at all five (B_gap 1.0265 T, 0.2849 T at Y30; τ_p 8.809, 11.011, 8.809, 8.809, 14.998 mm).
- {model.bsat_T} = 1.500 T at all five designs. With 1.3 T entered it reads 1.3 T for 4140, 12L14, 416, both 17-4PH conditions, 304 and 6061, and stays 1.7 T for 1018.
- The check reads "Too thin ... 2.0 mm", "Too thin ... 2.4 mm", "No back iron", "OK", "Too thin ... 3.3 mm".
Directions probed:
- **Sentence 2's levers, in the rule:** magnets per ring 4, 6, 8, 10, 12, 14, 16, 20 give 4.797, 3.198, 2.399, 1.919, 1.599, 1.371, 1.199, 0.960 mm (falling); ring radius from 8 mm in 0.5 mm steps across the Size tab's reach gives 1.625 mm at 8, 3.267 at 20, 4.088 at 26 (rising, no reversal); grades N35, N42, N48, N52 give 1.727, 1.919, 2.037, 2.140 mm (rising with B_gap 0.924, 1.027, 1.090, 1.145 T).
- **Watch-out 0:** the wall at 0.5 mm or one step thinner, or the design flux density one step lower, leaves the pull-out unchanged at all five designs while the check reads "Too thin".
- **change[0], wall +0.05 mm:** cup OD +0.100 mm at all five (dump). **change[1], design flux density −0.01 T:** the wall needed rises at all five (dump: 1.9189 to 1.9318 mm at the default).
Must-fix: none.
Should (not blocking): change[2] becomes "4140 or 1018 clear the low-saturation caution but heat more when slipping than 416 or 17-4PH; both rust, so keep plating above zero." (slip loss 2.751 and 3.024 W against 2.021 and 1.832 W at the default). watch_out[1]: "exact 2D sections" becomes "exact 2D field solutions".
Verdict: PASS

Earlier rounds: 5 must-fix item(s) in round 1, applied as given (the old and new texts are in
round 1's Must-fix above; round 2 confirmed each):
- sentences[1]: "The calculator's rule sizes the wall for half a pole's flux at the design flux density ..."
- sentences[2]: "So the rule asks for more wall with stronger magnets or wider poles ..."
- sentences[4] (seed S-11, amended): "Here the rule uses {model.bsat_T}; for those steels that is the entered design flux density ..."
- change[2] (`coupling.npole`, Raise): deleted. The note keeps three entries, and the back-iron pick moved from change[3] to change[2], its text unchanged.
- watch_out[0]: "With steel, the torque shown assumes it never saturates ..."

Seeds (A-24):
- S-11 (sentences[4]: "Here the cup-wall check uses {model.bsat_T}; for those steels that is the entered design value, which their own data may not support."): accepted in round 1 with two changes and applied as MF-7; round 2 confirms it in place. Reason: dropping "4140's" is right, because it keeps trap N4 ("1.5 T is 4140's saturation"; 4140's saturation is not sourced, `material_library.rs:117`) away. But "the cup-wall check uses" is false at the 304 probe design, where the check reads "No back iron" and compares nothing (`materials.rs:243-249`) while the rule still uses the value (`model.rs:1040`), so it becomes "the rule uses"; and "design value" becomes "design flux density", the term sentences[1] glosses. The applied text is true at all five designs.

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS):
- SF-2 (rounds 1 and 2), change[2] (change[3] before MF-8), 23 words: "4140 or 1018 clear the low-saturation caution but heat more when slipping than 416 or 17-4PH; both rust, so keep plating above zero." Why: at the default the slip loss goes from 2.021 W (416) or 1.832 W (17-4PH H900) to 2.751 W (4140) or 3.024 W (1018); with 1018 the time to the limit (high case) goes from "never" to 735.9 s. The dashboard's temperature verdict does not change at any probe design, so this is not blocking.
- SF-3 (rounds 1 and 2), watch_out[1], 28 words: "The rule is approximate: against exact 2D field solutions it asks for about 12 to 16 % too little wall at the example design, more with more poles." Why: "2D sections" is jargon.

Considered, no change (the reviewer's observations):
- `model.bsat_T`'s locked label reads "Back-iron saturation flux density" (`tests/data/python_schema.json:2081`), while sentences[1] glosses the value as a design flux density set below saturation. The note is right (T-K); the label is Python-locked, so no change here (round 2).
- "Half a pole's flux": the rule treats the flat-circuit gap flux density as a sine's peak (`records/dashboard.rs:56-58`). sentences[1] states what the rule sizes for, and watch_out[1] says the rule is approximate and asks for too little wall, so trap N11 ("the wall rule is exact") is not written (round 2).

## a5.uncoated_low_alloy_steel

Task T17. Rewrite of "Uncoated low-alloy steel" as "Unplated carbon or low-alloy steel", the
teaching note of the plating caution. Reviewer: physics reviewer (session model), fresh context.
Date: 2026-10-14 (the date the slice plan fixes for the note's `review` field; both rounds ran on
2026-10-15).

Rounds: 2. Verdict: PASS. Note: "Unplated carbon or low-alloy steel" (plan T17).

Verdict by round: 1 PASS; 2 PASS (no field of this note changed). The six warning notes were
reviewed together; each round's full answer is as under `## a5.non_ferromagnetic_back_iron`.
Round 2 re-checked this note only for dependence on the fields round 1 changed in the other notes
(none) and for the form checks (all clean), and kept its round-1 section as the final-round record.

The text reviewed in round 1 already carried A-13's plain-words sources. Every round confirmed
them applied.

The reviewer's report for the record, as returned (round 1; round 2 reproduced it verbatim):

Round 1, 2026-10-15, physics reviewer (session model).
Checked:
- The fields, at the five probe designs: the title; 3 sentences; 2 change entries; 1 watch-out; 2 sources.
- The engine behind them:
  - `warnings.rs:154` (steel circuit, known material, needs plating, plating <= 0);
  - `materials.rs:258-261` (the machining allowances t, t, 2t, 2t);
  - the plating input's help: high-phosphorus EN, typically 0.013 to 0.025 mm;
  - the library's `needs_plating` flags;
  - the R11 tests: 4140, 1018 and 12L14 fire at zero plating only; with plating, only 4140 and 1018 fire nothing.
- The slice-1 trap "416 is a safe fix" and N8 are not written: change[1] names the low-saturation caution.
- Sentence 0 is plain, 18 words, and answers the caution.
- The sources are A-13's, verbatim. No field names a clamp.
Numbers reproduced: none (no placeholders). At the default plating the allowances are 0.015, 0.015, 0.030 and 0.030 mm.
Directions probed:
- **change[0], plating 0.015 to 0.016 mm.** "Machine the hub flats under by" goes 0.015 to 0.016 mm at all five designs (up).
- **change[1].** 416 and 17-4PH need no plating, and they fire the low-saturation caution at all five designs. 304 selects the no-back-iron circuit (R11).
Must-fix: none.
Should (not blocking): none.
Verdict: PASS

Seeds (A-24): none for this note.

Should-fix offered: none.

Considered, no change (the reviewer's observation): "typically 0.013 to 0.025 mm" (change[0]) is
the sourced range in the plating input's help, not a typed default (round 1).

### Task review fix round 1 (2026-10-15): the 304 option names what it costs

- Why: the task code review (`task-17-review.md`, I-1) applied spec 3.1: "Every entry names its
  trade-off when the move pushes another check toward red". change[1]'s 416 and 17-4PH half named
  its cost (the low-saturation caution); its 304 half named only the mechanism ("removes the
  steel circuit altogether"). From an unplated steel the 304 pick lowers the pull-out, fires the
  open-circuit and expansion cautions and, at four of the five probe designs, turns the bond
  check from "Below the lap-shear strength" to "Above the lap-shear strength at the block ends".
  Round 1 checked this entry for truth only, not for its trade-off.
- change[1] (input, step and aim unchanged):
  - Old: "416 or 17-4PH need no plating here but trip the low-saturation caution; 304 removes the
    steel circuit altogether."
  - New (23 words by `word_count`; no placeholder, symbol or typed default; the caps and
    plain-words tests pass): "416 or 17-4PH need no plating but trip the low-saturation caution;
    304 needs none but loses steel torque and fires the expansion caution."
  - This is the task review's suggested wording, verbatim. "Steel torque" is the open-circuit
    note's own phrase (its change[0], "4140 or 1018 restore steel torque ...").
  - Superseded by the text re-review's MF-1 below. It is now "416 or 17-4PH need no plating
    here but trip the low-saturation caution; 304 needs none but loses steel torque and fires the
    expansion caution." The suggested wording had also dropped "here" from the 416 and 17-4PH
    half, which rounds 1 and 2 had passed with it.
- Engine (`compute_all`; at each probe design, every back-iron pick on the steel circuit with zero
  plating; a temporary probe test, deleted before the commit). Pull-out in N·m; the bond check's
  peak end shear against the 15 MPa lap-shear strength:

  | Probe design | Pull-out, 4140 / 1018 / 12L14 | Pull-out, 304 | Peak end shear, 4140 / 1018 / 12L14 -> 304 (MPa) |
  |---|---|---|---|
  | default | 2.6885 | 1.6974 (63.1 %) | 11.68 / 11.41 / 10.96 -> 15.76 |
  | 8 magnets per ring | 1.9684 | 1.0006 (50.8 %) | 11.68 / 11.41 / 10.96 -> 15.76 |
  | 304 back iron (the pick set to each steel) | 2.6885 | 1.6974 (63.1 %) | 11.68 / 11.41 / 10.96 -> 15.76 |
  | Y30 ferrite, custom dimensions | 0.2071 | 0.1183 (57.2 %) | 16.11 / 15.74 / 15.12 -> 21.74 (already above) |
  | 20 mm ring radius | 2.5393 | 1.3310 (52.4 %) | 11.87 / 11.60 / 11.15 -> 16.04 |

  - At all five designs 304 fires the open-circuit and expansion cautions and not the plating
    caution; 416 and both 17-4PH conditions fire the low-saturation caution and not the plating
    caution.
- Tests: `the_plating_note_s_back_iron_entry_names_the_cost_of_each_pick_it_offers`
  (`src/engine/warnings.rs`) asserts that change[1] contains "trip the low-saturation caution",
  "loses steel torque" and "fires the expansion caution". On the old text it failed ("the entry
  names \"loses steel torque\""); on the new text it passes. It also checks, at every probe
  design, from each unplated steel that fires the plating caution (4140, 1018 and 12L14):
  - 416 and both 17-4PH conditions need no plating and fire the low-saturation caution;
  - 304 needs no plating, fires the expansion caution and has a lower pull-out than each of them.
- Left for the re-reviewer (not changed here):
  - The entry names neither the bond check nor the open-circuit caution. The expansion caution's
    own note names the bond check, and the open-circuit caution opens its own note. The entry
    has 2 words left under the cap.
  - "Fires the expansion caution" carries the open SF-1 edge (ruling 5; task review M-2): a 304
    hub stops firing it once "NdFeB expansion in the bond plane" passes about +1.9e-6 /°C, inside
    its slider. It holds at the five probe designs. The open-circuit note's sentences[4] says the
    same without "usually", so the two notes agree; ruling 5 lets the reviewer soften both.
- The dump for the re-review: `NOTES_FOR_REVIEW=a5.uncoated_low_alloy_steel
  NOTES_REVIEW_OUT=target/notes-review-T17-fix1.txt` ("1 notes at 5 designs"). The new entry
  renders identically at the five probe designs, with no `{`, `}`, "no prediction" or "clamp".
- A text re-review of this field follows (controller ruling 7). Its report goes below as
  `### Re-review: code-review fix round 1`.

### Re-review: code-review fix round 1

The reviewer's report for the record follows unchanged. The full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-17-textreview-1.md`. Its scope was
change[1], read together with the fields whose meaning depends on it: sentences[0], watch_out[0],
and the sibling entries that use the same terms (`a5.non_ferromagnetic_back_iron` sentences[4]
and change[0], `a5.cte_mismatch_with_magnets` sentences[2]).

- **Reviewer:** physics reviewer (session model), fresh context, 2026-10-15. Text re-review 1 of change[1], after task-review fix round 1 (I-1).
- **Read:**
  - the dump `rereview-T17-1.txt`; `git show 84ede89`; `notes.rs:1099-1135` and the sibling a5 notes;
  - `warnings.rs:41-156` and its tests; `api.rs:335-353`; `material_library.rs:864-932` and the back-iron records; `temperature.rs:209-211` and `:1098-1112`; `dashboard.rs:920-943`;
  - `task-17-review.md`; `phys-notes-c.md` section 5;
  - the trap lists: both specs' section 8, `phys-new-notes.md` T-A to T-P, the added stiffness trap, and rulings 5, 8 and 11;
  - the caps and plain-words rules.
- **Numbers reproduced from `compute_all`** (scratch probe outside the repo; each back-iron pick on the steel circuit, plating 0):

  | Probe design | Pull-out, 4140 / 1018 / 12L14 (N·m) | Pull-out, 304 (share) | Peak end shear, steels -> 304 (MPa, lap-shear 15.00) |
  |---|---|---|---|
  | default | 2.6885 | 1.6974 (63.1 %) | 11.68 / 11.41 / 10.96 -> 15.76 |
  | 8 magnets per ring | 1.9684 | 1.0006 (50.8 %) | 11.68 / 11.41 / 10.96 -> 15.76 |
  | 304 back iron | 2.6885 | 1.6974 (63.1 %) | 11.68 / 11.41 / 10.96 -> 15.76 |
  | Y30 ferrite, custom dimensions | 0.2071 | 0.1183 (57.2 %; 57.152 % at full precision) | 16.11 / 15.74 / 15.12 -> 21.74 |
  | 20 mm ring radius | 2.5393 | 1.3310 (52.4 %) | 11.87 / 11.60 / 11.15 -> 16.04 |

  - At every design, 304 fires `non_ferromagnetic_back_iron` and `cte_mismatch_with_magnets`, and not the plating caution.
  - 416 and both 17-4PH conditions fire `low_saturation` only, keep the steel pull-out and lower the peak end shear.
  - These match the fix round's table.
- **The expansion edge (SF-1):** 304 stops firing the expansion caution once the NdFeB in-plane expansion reaches 2.0e-6 /°C (it still fires at 1.8e-6). This holds at all five designs, and the slider reaches 6e-6.
- **Form:**
  - the current text has 23 words; MF-1's has 24 and SF-1's 25;
  - 1 sentence; the guard is clean; no clamp, placeholder or typed default.
- **Must-fix:** MF-1. Restore "here" ("416 or 17-4PH need no plating here but ..."). Fix round 1 dropped it without saying so, from a half that rounds 1 and 2 had passed. The research prescribed it because 416's rust resistance is overstated otherwise (`phys-notes-c.md` section 5(2)). The 304 half passes as written. I-1 is closed.
- **Should-fix:**
  - SF-1: "usually fires", together with the open-circuit note's sentences[4] (ruling 5).
  - SF-2: pin "need no plating here" in the cost test.
- **Verdict:** MUST_FIX.

- MF-1, applied verbatim by the implementer (session model), 2026-10-15:
  - change[1] (input, step and aim unchanged): "416 or 17-4PH need no plating but trip the
    low-saturation caution; 304 needs none but loses steel torque and fires the expansion
    caution." -> "416 or 17-4PH need no plating here but trip the low-saturation caution; 304
    needs none but loses steel torque and fires the expansion caution." (24 words, 1 sentence;
    the caps and plain-words tests pass). It is rounds 1 and 2's 416 and 17-4PH half with the fix
    round's 304 half.
  - Test: `the_plating_note_s_back_iron_entry_names_the_cost_of_each_pick_it_offers`
    (`src/engine/warnings.rs`) also asserts, beside the three cost phrases and not among them,
    that change[1] contains "need no plating here". Its comment gives the reason: 416 is the
    least corrosion-resistant stainless offered (`phys-notes-c.md` section 5(2)). The fix-round
    text fails it ("the entry scopes the stainless picks' plating to this caution"); the restored
    text passes. This is the pin SF-2 suggests, added as MF-1's failing test.
  - Docs: the fix-round subsection above (a superseded line under its New text), the update
    tracker (its full quote of change[1], and one sentence on this re-review) and the README
    `tests/material_links.rs` row (the new pin). The BL-056 notes quote only the 304 half and
    are unchanged.
  - Not applied in this round (the round's list names MF-1 only): SF-1 (ruling 5's "usually",
    to be taken only together with `a5.non_ferromagnetic_back_iron` sentences[4]). It stays open
    for the controller. The reviewer's ledger observation also stays open: the doc comment of
    `CTE_MISMATCH_LIMIT_PER_C` (`warnings.rs`) gives 304 stainless as 17.7e-6 /°C, while the
    library record is 16.9e-6; no on-screen text depends on it.

### Re-review: code-review fix round 1 (re-check 2)

The reviewer's report for the record follows unchanged. The full answer is
`.superpowers/sdd/2026-10-14-magcoupler-guidance-slice-2/task-17-textreview-2.md`. Its scope was
change[1] as re-review 1's MF-1 left it (`c8eb83b`), "416 or 17-4PH need no plating here but trip
the low-saturation caution; 304 needs none but loses steel torque and fires the expansion
caution.", read together with the fields whose meaning depends on it: sentences[0], watch_out[0],
and the sibling entries that use the same terms (`a5.non_ferromagnetic_back_iron` sentences[4]
and change[0], `a5.cte_mismatch_with_magnets` sentences[2]).

- **Reviewer:** physics reviewer (session model), fresh context, 2026-10-15. Text re-review 2 of change[1], after MF-1 restored "here" (`c8eb83b`).
- **Read:**
  - the dump `rereview-T17-2.txt`; `git show c8eb83b` and `84ede89`; `notes.rs:1099-1135` and the sibling a5 notes;
  - `warnings.rs:41-171` and its tests; `api.rs:335-353`; `material_library.rs` (back-iron records, `resolve`);
  - `model.rs` (`select_calibration_factor`, `geometry_factor`, pull-out); `temperature.rs:1355-1380`; `dashboard.rs:265-350`;
  - `task-17-textreview-1.md`; `phys-notes-c.md` sections 5 and 7;
  - the trap lists: both specs' section 8, T-A to T-P, the added stiffness trap, and rulings 5, 8 and 11;
  - the caps and plain-words rules.
- **Checks:**
  - The text equals MF-1's replacement verbatim, and the test pin "need no plating here" is in place.
  - Tests pass: `warnings::tests` 12; `explain::notes` 27 (3 ignored); `--test explain` 49 (3 ignored).
  - The dump renders identically at all five designs, with no leaks.
- **Numbers from `compute_all`** (scratch probe outside the repo):
  - **304's pull-out against 4140's**, both unplated on the steel circuit, is at most 0.637. That is over magnets per ring 4-40, ring radius 9-30 mm and face gap 0.3-5 mm (3707 layouts that fit), with the maximum where the bench factor (1.049 against 0.95) applies. Over every harmonic-count choice it is at most 0.679. "Loses steel torque" holds throughout.
  - **From 4140, 1018 or 12L14 unplated, at all five designs, 304 also turns the "Daily-cycle screen"** from "Below the fatigue endurance" to "Above the fatigue endurance: qualify by thermal cycling" (15 of 15). This is covered by the expansion caution's "higher bond stress over temperature swings". It corrects re-review 1's "No other check turns".
  - **304 turns the hot-day torque rows and `model.verdict` red** at the designs where they were green: the hot-day rows at the default and 304 designs, `model.verdict` also at the 20 mm design. These are covered by "loses steel torque".
  - **416 and both 17-4PH conditions** fire only `low_saturation` (and not the plating caution). From 1018 they turn the cup-wall check to "Too thin" at the default and 304 designs, which the caution's text covers.
- **Form:** 24 words; 1 sentence; the guard is clean; no clamp, placeholder or typed default.
- **Must-fix:** none.
- **Should-fix:**
  - SF-1, carried: "usually fires", together with the open-circuit note's sentences[4] (ruling 5).
  - SF-3: the record correction above, with an optional daily-screen assert in the cost test.
  - SF-4: close re-review 1's ledger item on the `CTE_MISMATCH_LIMIT_PER_C` doc comment. Its 17.7e-6 is 304's mismatch, 16.9e-6 − (−0.8e-6), not its expansion coefficient, so the comment is correct.
- **Verdict:** PASS.

- Recorded by the implementer (session model), 2026-10-15. The verdict is PASS, so no code, test
  or on-screen text changed in this step; change[1] stands as re-review 1's MF-1 left it.
  - SF-3, the record correction, made here: re-review 1's full answer said under "Other loads"
    that no other check turns. That is incomplete. From each unplated steel (4140, 1018 and
    12L14), at all five probe designs, the 304 pick also turns the dashboard's "Daily-cycle
    screen" from "Below the fatigue endurance" to "Above the fatigue endurance: qualify by
    thermal cycling" (15 of 15). The screen scales the expansion-mismatch peak end shear (at the
    recommended bondline) by the daily swing over the worst swing (`temperature.rs`,
    `s2 * al.daily_swing_C / dT`). That is the bond check's mechanism, so "fires the expansion
    caution" covers it and no text changes.
  - SF-4, closed with no change: re-review 1's ledger observation, left open above, misread the
    doc comment of `CTE_MISMATCH_LIMIT_PER_C` (`warnings.rs`). The comment gives mismatches, not
    expansion coefficients: its 17.7e-6 is 304's mismatch with the magnets, 16.9e-6 - (-0.8e-6),
    as its 4140 figure is 12.3e-6 - (-0.8e-6) = 13.1e-6. The comment is correct.
  - Left open for the controller: SF-1 (ruling 5's "usually", to be taken only together with
    `a5.non_ferromagnetic_back_iron` sentences[4]), SF-3's optional daily-screen assert in
    `the_plating_note_s_back_iron_entry_names_the_cost_of_each_pick_it_offers`, and SF-4's
    optional rewrite of the comment as "(16.9e-6 - (-0.8e-6) = 17.7e-6)".

## a5.cte_mismatch_with_magnets

Task T17. Rewrite of "Expansion mismatch with the magnets" as "Hub expands much more than the
magnets", the teaching note of the expansion-mismatch caution. Reviewer: physics reviewer (session
model), fresh context. Date: 2026-10-14 (the date the slice plan fixes for the note's `review`
field; both rounds ran on 2026-10-15).

Rounds: 2. Verdict: PASS. Note: "Hub expands much more than the magnets" (plan T17).

Verdict by round: 1 MUST-FIX (4 items in this note, 13 across the task's six notes); 2 PASS. The
six warning notes were reviewed together; each round's full answer and the round-1 changes file
are as under `## a5.non_ferromagnetic_back_iron`. Round 2 re-checked only the changed fields and
the fields whose meaning depends on them.

The text reviewed in round 1 already carried A-13's plain-words sources. Every round confirmed
them applied.

The reviewer's report for the record, round by round, as returned:

Round 1:

Round 1, 2026-10-15, physics reviewer (session model).
Checked:
- The fields, at the five probe designs: the title; 4 sentences; 4 change entries; 2 watch-outs; 3 sources.
- The engine behind them:
  - `warnings.rs:155` (|hub − magnet| > 15e-6);
  - `api.rs:347-352` (the hub in effect; one NdFeB input for every grade);
  - `temperature.rs:1070-1101` (the screen: the hub only, the swing, the bondline, the modulus);
  - `temperature.rs:209-215` (the NdFeB input and its range; the adhesive modulus, which does not follow the pick).
- The library expansion coefficients, in µm/m per °C: 4140 12.3, 1018 12.0, 12L14 11.5, 416 10.08, 17-4PH 11.9 and 10.8, 304 16.9, 6061 23.6.
- R11: the caution fires for 304, 6061, or the circuit switched off.
- Traps N5, N6 and N10 are not written.
- Sentence 0 is 24 words, with no placeholder; its only glyphs are µ and °.
- The sources are A-13's, verbatim. No field names a clamp.
Numbers reproduced:
- {temperature.mismatch.peak_shear_current_MPa} = 11.68, 11.68, 15.76, 16.11, 11.87 MPa.
- {temperature.adhesive.lap_shear_MPa} = 15.00 MPa at all five designs.
Directions probed:
- **change[2], bondline 0.050 to 0.055 mm.** Peak shear falls at all five: 11.681 to 10.692, 11.681 to 10.692, 15.759 to 14.427, 16.111 to 14.747, 11.874 to 10.855 MPa. The mass falls slightly: the hub is cut.
- **change[3], adhesive modulus 0.107 to 0.106 GPa.** Peak shear falls at all five: 11.681 to 11.580, 11.681 to 11.580, 15.759 to 15.623, 16.111 to 15.972, 11.874 to 11.770 MPa.
- **change[0].**
  - From 304 to 4140: 15.76 to 11.68 MPa; mass 176.48 to 173.79 g.
  - From 6061 to 4140: 20.76 to 11.68 MPa; mass 96.81 to 173.79 g.
- **change[1], circuit 0 to 1 with 4140.** 20.76 to 11.68 MPa.
- **NdFeB edges.** 304 stops firing above about +1.9e-6. 4140 fires below about −2.7e-6.
Must-fix:
1. **sentences[2]** (seed S-12, amended). Current: "The caution fires above a 15 µm/m per °C difference: a 304 or 6061 aluminium hub, or no back iron; no listed steel does." Replacement: "The caution fires above a 15 µm/m per °C difference: 304 stainless or 6061 aluminium hubs, or no back iron; no listed magnetic steel does." Reason: 304 stainless is a listed steel that fires the caution, and "304 or 6061 aluminium" misreads. "listed" is needed because a typed 4140 expansion coefficient can fire it (`api.rs:347-351`, `warnings.rs:155`).
2. **sentences[3].** Current: "... against {temperature.adhesive.lap_shear_MPa} lap shear." Replacement: "The caution ignores the temperature swing and the glue; the bond check counts both, here {temperature.mismatch.peak_shear_current_MPa} of peak end shear against {temperature.adhesive.lap_shear_MPa} lap-shear strength." Reason: check g, "lap shear" glossed as the adhesive's strength.
3. **change[0].** Current: "a steel hub such as 4140 or 1018 cuts the mismatch most; it is heavier and needs plating." Replacement: "4140 or 1018 hubs cut the mismatch but bring back the cup-wall check, plating, steel's mass and slip heat, and higher shaft-joint and gearbox loads." Reason: "heavier" is false from 304 (176.48 to 173.79 g). The rise in cold-high input from 0.508 to 0.805 N·m, the returning "Too thin" check and the tripled slip loss went unnamed.
4. **change[1].** Current: "the steel circuit, with a magnetic steel picked, bonds the magnets to steel instead of aluminium." Replacement: "the steel circuit, with magnetic steel picked, bonds magnets to steel but adds mass, brings back the cup-wall check and raises shaft-joint and gearbox loads." Reason: no trade-off was named. The mass goes from 96.81 to 173.79 g and the pull-out from 1.697 to 2.688 N·m, and the check returns.
Should (not blocking):
- sentences[0]: "steel grows about 12" becomes "magnetic steel grows about 12".
- The expansion list holds at the default NdFeB input only (see the edges above).
Verdict: MUST-FIX

Round 2:

Round 2, 2026-10-15, physics reviewer (session model).
Checked:
- The fields changed by round 1, at the five probe designs: sentences[2] (MF-10), sentences[3] (MF-11), change[0] (MF-12), change[1] (MF-13); and sentence 0, whose "steel grows about 12" sentence 2 now qualifies. The title, sentences 0 and 1, change[2], change[3], both watch-outs and the three sources are unchanged from round 1 and stand.
- The engine behind them: `warnings.rs:155` (|hub − magnet| > 15e-6 /°C); `api.rs:347-352` (the hub in effect: steel on the steel circuit, else the body); `material_library.rs:895-909` (a non-magnetic pick is the body; otherwise the body is aluminium); `temperature.rs:1064-1111` (the screen: the worst swing from the cure, the bondline, the modulus; the reading against the adhesive's lap-shear strength).
- The traps: spec 8 (the expansion caution counts the swing; the coupling protects the gearbox; a shaft-joint method chosen or sized here); N5, N6, N10. None is written.
- Form: sentences 24, 24, 25, 24 words; changes 25, 25, 19, 16; watch-outs 26, 19; every field clean of banned patterns, audit ids and "clamp", sources included. Sentence 3's placeholders sit in its "here" clause beside the neutral "against".
Numbers reproduced:
- {temperature.mismatch.peak_shear_current_MPa} = 11.68, 11.68, 15.76, 16.11, 11.87 MPa.
- {temperature.adhesive.lap_shear_MPa} = 15.00 MPa at all five. The reading is "Below the lap-shear strength" at the default, 8-magnet and 20 mm designs and "Above the lap-shear strength at the block ends" at the 304 and Y30 designs.
- The caution, by pick with the circuit on: fires for 304 and 6061 only; with the circuit off: fires for every pick; the same at all five designs (at the default NdFeB expansion input).
Directions probed:
- **change[0], from 304 to 4140 (1018):** peak end shear 15.76 to 11.68 (11.41) MPa at the default, 8-magnet and 304 designs; 21.74 to 16.11 (15.74) at Y30; 16.04 to 11.87 (11.60) at 20 mm. From 6061: 20.76 to 11.68 (11.41), 28.63 to 16.11 at Y30, 21.75 to 11.87 at 20 mm. The cup-wall check returns, cold-high at the gearbox input rises (0.508 to 0.805 N·m at the default) and the plating caution fires at zero plating, at all five. The mass and slip loss move both ways, as in `a5.non_ferromagnetic_back_iron` (from 304 the mass falls 1.4 to 1.8 % and the slip loss rises; from 6061 the mass nearly doubles and the slip loss falls).
- **change[1], circuit 0 to 1 with 4140:** peak shear 20.76 to 11.68 MPa (Y30 28.63 to 16.11; 20 mm 21.75 to 11.87); mass 96.81 to 173.79 g (92.05 to 174.59, 84.03 to 161.01, 168.21 to 348.64 g); the check returns; cold-high at the input rises. Held at the 304 design.
- **change[2], bondline +0.005 mm, and change[3], adhesive modulus −0.001 GPa** (unchanged): peak shear down at all five (dump).
Must-fix: none.
Should (not blocking): sentences[3] ends "against {temperature.adhesive.lap_shear_MPa} room-temperature lap-shear strength." (25 words): the strength is the 22 °C data-sheet value, the shear arises at the temperature extremes. change[1] becomes "the steel circuit, with magnetic steel picked, bonds magnets to steel, adds mass over aluminium, brings back the cup-wall check, raises shaft-joint and gearbox loads." (25 words; the path from 304 lowers the mass). sentences[0]: "steel grows about 12" becomes "magnetic steel grows about 12". The list in sentence 2 holds at the default NdFeB expansion input only (4140 fires below about −2.7e-6 /°C).
Verdict: PASS

Earlier rounds: 4 must-fix item(s) in round 1, applied as given (the old and new texts are in
round 1's Must-fix above; round 2 confirmed each):
- sentences[2] (seed S-12, amended): "The caution fires above a 15 µm/m per °C difference: 304 stainless or 6061 aluminium hubs ..."
- sentences[3]: "The caution ignores the temperature swing and the glue; ..." (now ending "... lap-shear strength.")
- change[0] (`materials.parts.back_iron`, Choose): "4140 or 1018 hubs cut the mismatch but bring back the cup-wall check ..."
- change[1] (`coupling.backiron`, Choose): "the steel circuit, with magnetic steel picked, bonds magnets to steel but adds mass ..."

Seeds (A-24):
- S-12 (sentences[2]: "The caution fires above a 15 µm/m per °C difference: a 304 stainless or 6061 aluminium hub, or no back iron; no magnetic steel does."): accepted in round 1 with one change and applied as MF-10; round 2 confirms it in place. Reason: "304 stainless" removes the misreading of the draft's "a 304 or 6061 aluminium hub" as two aluminium alloys, and the draft's "no listed steel does" was false: 304 stainless is a listed back-iron steel, and it fires the caution (mismatch 17.7 against 15; the R11 test `the_open_circuit_and_expansion_rules_fire_for_304_and_6061_or_the_circuit_switched_off`). The seed's "no magnetic steel does" becomes "no listed magnetic steel does": the rule reads the steel expansion coefficient in effect (`api.rs:347-351`, `warnings.rs:155`), and 4140's is a typed input whose slider runs to 25e-6, so a typed 4140 coefficient above about 14.2e-6 /°C fires the caution; "listed" keeps the claim to the picks the R11 tests pin. The applied text is true at all five designs (at the default NdFeB expansion input: SF-6 below).

Should-fix offered, not applied (open for the controller; no on-screen text changed after the
round-2 PASS):
- SF-4 (rounds 1 and 2), sentences[0], 25 words: "In the plane of the glue joint NdFeB barely expands with heat, while magnetic steel grows about 12 and aluminium about 24 µm/m per °C." Why: 304 stainless is also a steel, at 16.9 µm/m per °C, and sentences[2] now names it as a hub that fires; "steel grows about 12" leaves that unexplained.
- SF-6 (rounds 1 and 2; no text change, for the record): the expansion list holds at the default "NdFeB expansion in the bond plane" input only. 4140 fires with that input below about −2.7e-6 /°C (slider minimum −3e-6), and 304 stops firing above about +1.9e-6 /°C. So "no listed magnetic steel does" is true at the five designs but not at the slider's end. `a5.non_ferromagnetic_back_iron`'s SF-1 covers the other note.
- SF-8 (round 2), change[1], 25 words: "the steel circuit, with magnetic steel picked, bonds magnets to steel, adds mass over aluminium, brings back the cup-wall check, raises shaft-joint and gearbox loads." Why: as under `## a5.non_ferromagnetic_back_iron` (SF-8 there): the entry is true for the move it offers, but the path from 304 lowers the mass.
- SF-9 (round 2), sentences[3], 25 words: "The caution ignores the temperature swing and the glue; the bond check counts both, here {temperature.mismatch.peak_shear_current_MPa} of peak end shear against {temperature.adhesive.lap_shear_MPa} room-temperature lap-shear strength." Why: the bond check compares the peak end shear, which arises at the worst temperature swing from the cure (`temperature.rs:1066`), with the adhesive's lap-shear strength at 22 °C from its data sheet ("Selected adhesive lap shear at 22 °C (TDS)"). A reader may take "lap-shear strength" as the strength at temperature, which is lower when hot. The placeholders stay in the "here" clause.

Considered, no change (the reviewer's observation): the identical tails shared with
`a5.non_ferromagnetic_back_iron`'s change[0] and change[1] are harmless, as under that note
(round 2).
