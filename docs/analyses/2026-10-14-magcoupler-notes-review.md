# Magcoupler teaching notes: per-note physics reviews (guidance slice 2)

Branch `magcoupling/guidance-2`, crate `magcoupling-rs`. Spec
`docs/superpowers/specs/2026-10-14-magcoupler-guidance-2-design.md`, section 4 (decision S2-6).

Each note that guidance slice 2 adds or rewrites is reviewed by a fresh-context session-model
physics reviewer before it ships. The reviewer reads the note rendered at the five probe designs
(the default design, 8 magnets per ring, a 304 back iron, Y30 ferrite on custom dimensions, a
20 mm ring radius; `engine::explain::probes`), the engine code behind every claim and the
physics traps of the slice-1 and slice-2 specs (section 8 of each), reproduces every number from
`compute_all` and probes every "What to change" direction. Must-fix items are applied and
re-reviewed, up to three rounds. A note becomes `Reviewed` in the same commit that appends its
section here; its `record` names the section as `<this file>#<note id>`.

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
