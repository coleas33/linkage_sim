# Physics review, slice 2, notes-a

The six notes covered here are `harmonics`, `back_iron_factor`, `pullout_angle`, `end_effect`, `calibration` and `br_temperature`, all in `magcoupling-rs/src/engine/explain/notes.rs`.

Reviewer: session model, physics lens. The repo was read only. Every number below comes from `compute_all` at `DesignInputs::default()` with all corrections on, unless a line says otherwise. The probe crate lives outside the repo, at `scratchpad/s2/physa_probe` (`src/main.rs`, `src/bin/more.rs`, `src/bin/harm.rs`). Validation figures for the 3D and 2D checks come from `docs/analyses/2026-09-29-magcoupling-math-audit.md` and `...-audit-results.json`.

Notation used in the drafts:
- `{path}` means a live value of that result or input.
- `{= expr}` means a value derived live from results.
- Anything else is a literal, either allowed or flagged as such.

If the placeholder mechanism (TC-8) does not land, drop the live number and keep the words. Never substitute the default value.

---

## 0. Default-design numbers the reviews rely on (probe output)

| Quantity | Value |
|---|---|
| Pull-out at 50 °C / at 20 °C / ratio | 2.688 / 2.893 N·m / 0.9293 |
| T_2D, f_end, f_cal | 3.159 N·m, 0.8960, 0.950 |
| σ, σ_1 (share) | 201 380 Pa, 199 961 Pa (99.3 %) |
| R_g, τ_p, g, fill inner/outer | 14.02 mm, 8.809 mm, 1.400 mm, 0.861 / 0.620 |
| k_1, k_1·t | 356.6 /m, 1.131 |
| S_1 steel / free (ratio) | 0.2445 / 0.1392 (1.757); the ratio is 1.068 for S_3 and 1.007 for S_5 |
| e^(−k_n g), n = 1, 3, 5 | 0.607, 0.224, 0.082 (π g/τ_p = 0.499) |
| Harmonic setting 1 / 1,3 / 1,3,5 / to 11 | 2.670 / 2.699 / 2.688 / 2.689 N·m |
| Pull-out angle | 90° electrical = 18° mechanical |
| Gap +0.1 mm | −3.2 % at 10 poles, −2.9 % at 8, −2.4 % at 6 |
| dT/dL (magnet length) | 0.2363 N·m/mm = T_pull/(L − c_end·τ_p), exactly |
| c_end·τ_p | 1.321 mm (10 poles), 1.652 mm (8 poles) |
| f_cal bench (prototype match) | 1.0495; measured/model 1.1047; prototype model error −9.5 % |
| Cold (−40 °C), cold-high | 3.325 (×1.149), 3.823 N·m |
| Hot-day start 65 °C, torque at the demag limit (93.06 °C) | 2.589, 2.408 N·m |

---

## 1. Cross-cutting findings (all six notes)

**X1. Definitions come late or never.** Six terms are used before any note defines them: shear stress, pole pitch, fill, electrical angle, remanence and "2D/3D". If the new "how the coupling makes torque" note goes first in Start here, it should define:
- shear stress σ;
- gap radius R_g;
- pole pitch τ_p;
- the 2D (endless-ring) model.

Fill is best defined in the rewritten harmonics note (see X5). Remanence belongs in `br_temperature`, which comes later in the walk. Every earlier mention therefore needs a three-word gloss, such as "Br (the magnet's built-in flux density)", or must avoid the term.

**X2. Audit ids in the text.** The text quotes "correction E7", "correction E3", "M1 audit M4/M5/M9/P2" and "M1 skeptic", plus "workbook" (at notes.rs 77, 131, 152, 155, 175, 181). All of them come out. Sources move behind a fold.

**X3. Editable defaults stated as constants (T6).** The calibration note states 0.95, 1.8 N·m, the prototype's pole count and 20 °C as fixed. The br_temperature note states α = −0.12 %/°C, the 30 °C rise and the 10 °C margin as fixed. The end_effect note states 1.3 mm and 3 mm as fixed. Each must be either:
- a live value;
- "by default" plus the on-screen input label; or
- dropped.

**X4. Live versus literal.**
- *Must be live:* any value of the user's design. That covers fills, pole pitch, c_end·τ_p, f_end, f_cal (in effect, measured, assumed), the S_1 ratio, the gap fade, the temperature ratios, α in use, the pull-out angle and the mechanical twist.
- *Fine as literals:*
  - math constants and thresholds: 4/π, sin(πλ/2), 90°, 1/9, 2/3, 0.4, "a third of a pole pitch";
  - "B842SH" (hard-coded in the calibration rule);
  - typical material coefficients when worded "about", such as ferrite −0.2 %/°C or SmCo −0.035 %/°C;
  - results of the 2026 validation studies, when scoped to their conditions ("free-space 3D at 10 poles, 1.4 mm gap").

**X5. Equation ownership, to coordinate with the three new notes.** The rule is at most one note per equation (`tests/explain.rs:1174`).
- **Fill** (`model.fill_inner`, `model.fill_outer`). TC-1 proposes the new torque note. I recommend `harmonics` instead, because the fill is what that note is really about once rewritten. The torque note keeps `area_lever`, `torque_2d`, `gap_radius` and `pole_pitch`.
- **`metal.torque_cold_Nm`** is in `br_temperature` today. If the new hot-low/cold-high note takes the cold side, move it there and keep the temperature-only rows in `br_temperature`.
- **`model.gearbox_input_ripple_Nm`.** `ROW_LEARN` (`gui/guidance.rs:318`) already points the dashboard ripple row to `pullout_angle`, but the note does not list the equation. Add it (TC-12).
- **The α records** (`model.alpha_br_per_C`, `model.inner_alpha_br_per_C`, `model.outer_alpha_br_per_C`) are owned by no note. Add them to `br_temperature`; they carry the T6 wiring fact.

**X6. Gate mechanics that bite these six notes.**
- All six are in START_HERE (steps 1 to 6), so each rewrite must land `Reviewed` in the same commit (`gui/explorer.rs:1085-1091`, `tests/explain.rs:1038`).
- The title "Harmonic decomposition" is pinned at `gui/explorer.rs:1076-1084`.
- Each note needs 2 to 6 entries.
- Glyphs:
  - avoid ∝ (it renders "~");
  - avoid superscript minus;
  - "e^(−...)" with U+2212 is already in use, so it is OK.

**X7. Dashboard entry points.**
- `ROW_LEARN` maps "Pull-out torque at operating temperature" and the gearbox ripple row to `pullout_angle`.
- It maps "Pull-out torque at 20 °C" to `br_temperature`.
- The end-effect check guide's Learn link goes to `end_effect`.

These are now the main way users arrive, so the first sentence of each must answer that row's question.

---

## 2. harmonics (Start here step 1; opens `model.tau_Pa`)

### (1) What the engineer needs
How much block fill costs or gains, and whether the "Highest odd harmonic summed" setting matters. The fundamental carries nearly all the torque and grows with sin(π·fill/2). Wider blocks therefore help while they fit, especially on the outer ring, and fewer blocks hurt. The harmonic setting barely matters at normal fill, but it matters a lot at low fill.

### (2) Statement-by-statement
- **Title "Harmonic decomposition"**: jargon (TC-31); pinned by a test.
- **s1**: fine. It is an idealization (radial square wave, fill taken at the magnet mid-thickness), which the model accepts.
- **s2 (85 words)**:
  - It is mathematically correct.
  - It uses "remanence" before it is defined (that happens in step 6).
  - Its fill definition omits that fill is measured per ring at the magnet mid-thickness. The outer fill is therefore lower with the same blocks (0.62 against 0.86).
  - The negative-sign remark ("sin(3π/2) = −1") is sign-convention trivia with no design use. The sign matters only against the fundamental, which belongs to the pull-out note.
- **s3**: correct (orthogonality). "Shear stress" is undefined (X1).
- **s4: MISLEADING.**
  - The note says higher harmonics matter little "above all [because] their fields fade across the gap much faster". At the default design the larger cause is amplitude: the two rings' harmonic amplitudes multiply.
    - B_i3·B_o3 / B_i1·B_o1 = 0.024 and B_i5·B_o5 / B_i1·B_o1 = 0.023.
    - The gap-and-thickness factor only gives S_3/S_1 = 0.46 and S_5/S_1 = 0.17.
    - Gap fade dominates only at a larger g/τ_p than the default's 0.5.
  - The parenthesis "the fill can still make a higher one larger than a lower one" is true only between higher harmonics. No harmonic exceeds the fundamental, since |sin nx| ≤ n|sin x|. As written it confuses.
  - "the workbook keeps only 1, 3 and 5" is jargon. The choice label "1, 3, 5 (workbook)" (`model.rs:54`) is on-screen jargon too (map W-20).
- **Missing, decision-relevant**:
  - The setting is not always cosmetic.
    - At the default, harmonics 3 to 11 move the pull-out by under 1 %.
    - At low fill, the third harmonic moves it by 12 to 33 %, in either direction, and shifts the pull-out angle. Examples:
      - 6 poles: −12.6 %, angle 59°;
      - 4 poles: +32.6 %, angle 33°;
      - a 20 mm ring: −12.1 %, angle 53°;
      - narrow B822 blocks: −13.1 %, angle 62°.
    - Harmonics 7 to 11 add at most about 1.6 % on top of 1, 3, 5 in every probe.
  - Fill levers:
    - fewer magnets on the same ring lower the fill (10 → 8: 0.86/0.62 → 0.69/0.50);
    - a wider outer block raises it: B862 outer gives +15 % and still fits, with 0.07 mm between blocks; manual 8 mm gives +10.5 %; B882 outer does not fit.
  - Fill is clamped at 1. Above that the blocks overlap, and the fit check fails.
- **Watch-out (fill 0.4 removes the fifth)**: correct, but trivia. It supports no decision.

### (3) Minimum facts a rewrite must keep
1. Alternating blocks with gaps make a squarish pattern. It equals a fundamental (one wave per pair of poles) plus weaker odd harmonics.
2. The fundamental amplitude is proportional to sin(π·λ/2). λ is the fill: block width over the space per pole at the magnet mid-thickness, on each ring separately.
3. Harmonic n pairs only with harmonic n of the other ring.
4. Higher pairs are small, because both amplitudes are small and they fade faster across the gap.
5. At low fill the third harmonic is large and can change the pull-out and its angle.

- Live: σ_1 share `{= model.tau1_Pa / model.tau_Pa}`, `{model.fill_inner}`, `{model.fill_outer}`, `{coupling.max_harmonic}`.
- Literals OK: 4/π, sin(πλ/2), "two-thirds", "0.4".
- Not as literals: 99 %, 0.86/0.62, and "under 1 %" unless it says "at the default design".

### (4) What to change
- **Outer magnet part** (or **Manual outer width (tangential)**): wider → higher outer fill → more torque. Trade-off: it must still fit (Outer flat check / Magnets fit). It adds magnet mass and cost.
- **Inner magnet part**: little room at the default (0.25 mm slack on the flat). Wider inner blocks need a larger **Ring radius**.
- **Magnets per ring**: more magnets at the same ring raise fill and shorten the pitch. Trade-offs:
  - the blocks may stop fitting (12 do not fit at the default ring);
  - the cup OD changes;
  - S_1 with steel peaks near the default (see back iron).
  Fewer magnets lower fill (10 → 8: −27 % pull-out).
- **Ring radius / Outside diameter**: a larger ring at the same count and blocks lowers fill and can bring a large third harmonic (pull-out angle 53° at a 20 mm ring). The lever arm grows at the same time.
- **Highest odd harmonic summed** (Assumptions, "Harmonics included"): this is a model-detail setting, not a design lever. Keep at least "1, 3, 5". Raising it matters only at low fill, and then only by about 1 %.

### (5) Draft rewrite
**Title:** "Block fill and the magnet pattern (harmonics)". The explorer test that pins the old title needs updating.
1. Around each ring, blocks alternate north and south with small gaps between them, so the magnet pattern along the gap is squarish, not smooth. (24 words)
2. That pattern equals a sine wave, the fundamental (one wave per pair of poles), plus weaker odd harmonics with 3, 5, 7 times the waves. (25)
3. The fundamental grows with sin(π λ/2), where the fill λ is the block width over the space each pole has on that ring. (22)
4. Each harmonic pulls only on the same harmonic of the other ring; the higher pairs add little, being weaker and fading faster across the gap. (25)
5. Here the fundamental carries `{= σ_1/σ}` of the torque; fills are `{λ_i}` inner, `{λ_o}` outer, and a wider block raises torque while it still fits. (25)

**Watch-out:** "Fewer magnets on the same ring leave wider gaps between blocks, so the fill falls and the fundamental weakens; the ring does not shrink." (23)

**Optional 6th entry:** "At low fill, from few magnets, a large ring or narrow blocks, the third harmonic can change the pull-out by over 10 %." (21)

**Keep linking:**
- `model.b_i#`, `model.b_o#`, `model.amp#_Pa`, `model.tau_Pa`, `calibration.amp#_Pa`;
- add `model.fill_inner` and `model.fill_outer` (X5).

Diagram SquareWaveHarmonics: keep.

---

## 3. back_iron_factor (Start here step 5; opens `model.s1_iron`)

### (1) What the engineer needs
Whether to keep steel behind the magnets, and how gap, magnet thickness and pole pitch trade off.
- Steel raises the fundamental's factor by `{S1_iron/S1_free}` (1.76 at the default; 3.0 at 4 poles; 1.22 at 20).
- The gap costs torque in proportion to gap over pole pitch.
- Thicker magnets give diminishing returns in S.
- The shown torque assumes the steel does not saturate.

### (2) Statement-by-statement
- **Title "the sinh factor against free space"**: math jargon.
- **s1: MISLEADING.** S_n is not "how much of harmonic n's field from one ring reaches through the other ring". It is the dimensionless factor that turns B_i,n·B_o,n/(2μ0) into shear stress. It packs together both magnet thicknesses, the gap and the circuit.
- **s2**: correct.
- **s3**: correct, but boundary-value language with no design use. Drop it.
- **s4**: correct and general. S_iron/S_free = (1+e^(−kt_i))(1+e^(−kt_o))/(1−e^(−2k(t_i+t_o+g))), which is > 1 for every harmonic. Keep it.
- **s5**:
  - k is never defined in words: k_n = n·π/τ_p.
  - "a small gap matters most for the high harmonics" is true in relative terms but irrelevant, since the high harmonics carry about 1 %.
  - It misses the design rule: the fundamental fades about as e^(−π g/τ_p) (0.61 here), so the gap counts against the pole pitch. Gap +0.1 mm costs 3.2 % of the pull-out at 10 poles, 2.9 % at 8 and 2.4 % at 6.
- **Missing magnitude**: no number for the steel gain.
- **Missing thickness**: thickness enters S through k·t, and the note never mentions it.
- **Missing model limit (PA-9)**: the torque assumes ideal, unsaturated steel pressed on the magnet backs. The default design fails the cup-wall check, so the shown torque is optimistic there. The note says nothing.
- **Trap (TC-11 corrected)**: do not write "fewer poles reach across the gap better".
  - The gap factor improves, but sinh(k·t) falls, so S_1 with steel peaks near the default pitch: 0.2373 at 8 poles, 0.2445 at 10, 0.2397 at 12.
  - The net pull-out falls from 10 to 8 poles.
  - The true general statement: S_1 has a best pole pitch. It tends to 0 as k → 0 (k·t_i·t_o/(t_i+t_o+g)) and as k → ∞ (e^(−kg)/2).

### (3) Minimum facts
1. Steel gives the flux a return path.
2. S with steel is always larger than without.
3. The gain is largest for the fundamental, and largest when the pole pitch is long against the magnet thickness. It is about nil for the high harmonics.
4. The fundamental fades as about e^(−π g/τ_p).
5. S_1 has a best pole pitch.
6. Thicker magnets help with diminishing returns once thickness approaches about τ_p/3 (k·t ≈ 1).
7. The steel is assumed unsaturated.

- Live: `{= model.s1_iron / model.s1_free}`, `{= exp(-model.k1 * model.face_gap_mm/1000)}`, `{model.pole_pitch_mm}`, `{model.face_gap_mm}`.
- Literals OK: π, "a third of the pole pitch" (it is math: k·t = 1 → t = τ_p/π).

### (4) What to change
- **Back iron** / **Back iron material (hub, cup and boss)**: keep a ferromagnetic steel.
  - Without steel, the default pull-out drops to 1.537 N·m at the same calibration. The screen shows 1.697 with the bench factor.
  - Trade-offs: steel adds mass and slip-heating loss.
  - Do not offer 416 as the safe fix: it trips the low-saturation caution (spec 8).
- **Magnet face gap** (Size tab) = **Candidate flat-face magnetic gap**: smaller gives more torque, about 3 % per 0.1 mm here. Trade-off: running clearance, which already fails at the default, and the sleeve and liner thickness.
- **Magnet thickness** (magnet part, or **Manual inner/outer thickness (radial)**): thicker raises S. In this tool it also pushes the gap radius out:
  - 3.17 → 4.0 mm: pull-out +18 %, cup OD +3.5 mm;
  - 3.17 → 6.35 mm: +53 %, cup OD +13 mm.
  Diminishing in S only; never write "thickness does not help".
- **Magnets per ring**: changes the pole pitch.
  - Fewer: steel matters more, and each 0.1 mm of gap costs less.
  - But fill falls and S_1 with steel is near its peak.
  - It is not a lever through this factor alone (spec 8 trap).
- **Minimum outer return ring wall**: raise it until the cup-wall check passes. Until then, the steel benefit shown is optimistic.

### (5) Draft rewrite
**Title:** "Back iron: why steel behind the magnets adds torque"
1. Steel behind each ring gives the flux an easy return path, so more of it crosses the gap instead of leaking into the air behind. (25)
2. The geometry factor S_n, which turns both rings' magnet strengths into pull across the gap, is therefore always larger with steel. (21)
3. Here steel multiplies the fundamental's S_1 by `{= S1_iron/S1_free}`, most when the pole pitch is long compared with the magnet thickness. (21)
4. Across the gap the fundamental fades as about e^(−π g/τ_p), `{= exp(−k_1 g)}` here, so the gap counts against the pole pitch. (20)
5. S_1 with steel has a best pole pitch: too long and the magnets pull weakly sideways, too short and the field dies across the gap. (25)

**Watch-out:** "The torque assumes the steel never saturates: if the cup wall check reads too thin, the real torque will be below the number shown." (24)

**Optional 6th:** "Thicker magnets raise S_n with diminishing returns past about a third of a pole pitch, and they enlarge the gap radius and the cup." (23)

**Keep linking:** `model.s#_iron`, `model.s#_free`, `model.k#`. Diagram FluxPathBackIron: add N/S labels (TC-24).

---

## 4. pullout_angle (Start here step 2; opens `model.pullout_Nm`; dashboard Why for pull-out and gearbox ripple)

### (1) What the engineer needs
- What the pull-out number is: the peak of torque against twist, the most the coupling passes before slipping. It is a nominal estimate at the operating magnet temperature.
- How far the rings twist before they slip.
- What happens once they slip: torque swings, the average falls to the eddy-current drag, the parts heat, and the coupling may not re-lock until the slip slows.

### (2) Statement-by-statement
- **Title**: acceptable. A plainer version is offered below.
- **s1**: correct and good. It should add "nominal estimate". The result help already says "not a guaranteed minimum", and spec 8 forbids "guaranteed".
- **s2**: "A_n" is never defined (the record symbol is A_{#} = `model.amp#_Pa`). "Electrical angle" is never defined or tied to the mechanical twist that the Torque vs rotation tab plots (φ = N/2 × θ; 90° = 18° at 10 poles; TC-12).
- **s3: INCOMPLETE (PA-8).**
  - Only a third harmonic of the same sign as the fundamental, larger than about a ninth of it, turns 90° into a dip and moves the peak lower. An opposite-sign third harmonic sharpens the 90° peak, as at the default (A_3/A_1 = −0.011).
  - A same-sign A_3 needs both rings' fill under 2/3. That is necessary, not sufficient.
  - Probes: 6 poles 59°, a 20 mm ring 53°, B822 62°.
  - "(correction E7)" is an audit id.
- **Missing: the equation the dashboard row opens.** T_pull = T_2D·f_end·f_cal at the Operating magnet temperature. The note never says it, yet Start here step 2 and the Pull-out Why both land on it.
- **Missing: what happens past pull-out (CG-13, corrected).**
  - The rings slip.
  - The torque swings ±T_pull once per pole pair passing; the ripple frequency is N/2 × slip rpm/60 (`model.rs:1154`).
  - The average transmitted torque is only the eddy-current drag, and the parts heat.
  - Re-locking needs the slip speed to come down. **It is not automatic once the load falls below pull-out (spec 8 trap).**
  - The gearbox ripple row is that swing at the gearbox input: pull-out/(ratio·efficiency).
- **No watch-out**: the after-slip behaviour is the natural one.

### (3) Minimum facts
1. Torque rises with twist, peaks and falls. The peak is the pull-out, a nominal estimate.
2. T_pull = T_2D × f_end × f_cal, at the Operating magnet temperature.
3. Electrical angle φ = N/2 × mechanical twist. With the fundamental alone the peak is at 90° (half a pole pitch), which is a twist of 180°/N.
4. A same-sign third harmonic above about A_1/9 (from low fill) moves the peak lower. The calculator always takes the true peak.
5. Past pull-out the torque swings ±T_pull once per pole pair, averaging only the eddy-current drag. The coupling heats and may not re-lock until the slip slows.

- Live: `{model.pullout_Nm}`, `{coupling.op_temp_C}`, `{= deg(model.pullout_angle_rad)}`, `{= deg(model.pullout_angle_rad)/(coupling.npole/2)}`, `{model.gearbox_input_ripple_Nm}`.
- Literals OK: 90°, "a ninth", "two-thirds".

### (4) What to change
- To raise pull-out, use the levers of the new torque note:
  - **Overall length** (magnet length), the strongest per mm at the default (+0.236 N·m/mm);
  - **Magnet face gap** (smaller);
  - **Outside diameter / Ring radius** (larger; weaker per mm than length here, and fill falls);
  - wider blocks, while they fit;
  - steel back iron.
  Never call radius the strongest lever (spec 8).
- **Magnets per ring**: more magnets give a smaller twist at pull-out (180°/N), so a stiffer coupling. Trade-offs: fit, fill, S_1.
- **Operating magnet temperature**: set it to the real running magnet temperature, since the pull-out is reported there.
- Gearbox ripple: it follows the pull-out. A coupling much stronger than needed passes larger torque swings to the gearbox when it slips. The trade is against the hot-low requirement.

### (5) Draft rewrite
**Title:** "Pull-out: the most torque before the rings slip"
1. Twist one ring against the other and the torque rises from zero, peaks and falls; the peak is the pull-out torque, a nominal estimate. (24)
2. The calculator gives it at the Operating magnet temperature: the endless-ring torque T_2D times the end factor f_end and the calibration factor f_cal. (23)
3. Angles here are electrical, φ = N/2 times the mechanical twist: this peak is at φ = `{φ_pull}`°, a twist of `{θ_pull}`° (Torque vs rotation tab). (26)
4. With the fundamental alone the peak is at 90°; a same-sign third harmonic above about a ninth of it, from low fill, moves the peak lower. (26)
5. If the load exceeds it the rings slip: the torque swings between plus and minus pull-out once per pole pair, averaging only the eddy-current drag. (25)

**Watch-out:** "A slipping coupling may keep slipping after the load drops below pull-out: it re-locks only once the slip speed falls, and meanwhile it heats." (24)

**Optional 6th:** "The gearbox input ripple row is that swing seen through the gearbox: pull-out divided by ratio times efficiency." (18)

**Keep linking:**
- `model.tau#_Pa`, `model.pullout_angle_rad`, `model.iron_circuit_angle_rad`, `model.free_circuit_angle_rad`, `model.pullout_Nm`, `calibration.pullout_angle_rad`, `calibration.tau#_Pa`;
- add `model.gearbox_input_ripple_Nm`.

Diagram TorqueAngle: it now shows the rare double hump (TORQUE_THIRD 0.15 > 1/9; TC-22). Draw the fundamental-only curve dashed, peaking at 90°, and keep the summed curve for the E7 test.

**Pre-existing collision**: the help of `pullout_angle_rad` (`model.rs:340`) says θ and τ_n where the panel shows φ, σ_n and A_n (PA-15). Fix it in item C.

---

## 5. end_effect (Start here step 3; opens `model.f_end`; end-effect check guide's Learn)

### (1) What the engineer needs
How much torque the magnet ends cost, and how short the magnets can be.
- The model subtracts a fixed length c_end·τ_p, so torque grows in step with L − c_end·τ_p: `{dT/dL}` N·m per mm.
- More magnets per ring shrink the deduction.
- Trust the factor less at wide gaps, where it is optimistic, and at very short magnets.

### (2) Statement-by-statement
- **s1**: "2D model" is undefined (X1). Otherwise correct.
- **s2**:
  - "takes off a total of about c_end pole pitches" is awkward but correct: it removes the length c_end·τ_p, which is 0.15 of a pole pitch by default.
  - c_end is the editable **End-effect coefficient** (an assumption). The note gives it as a symbol, which is fine.
  - "active length L" means the shorter ring's length. Overhang is not credited (M10): +12.6 % in 3D for 25.4 over 12.7 mm.
- **s3: OVERSTATED (PA-14 corrected).**
  - "within about 1 % of 3D for magnets longer than about 3 mm" is wrong. The free-space 3D figures are +0.4 % at 12.7 mm, +1.8 % at the 6.35 mm library length and +0.8 % at 3.17 mm. "Within 2 % for 3.2 mm and longer" is right.
  - "at the default design" implies steel, but the check was magpylib free space at 10 poles and a 1.4 mm gap.
  - "one M1 skeptic found it 11 % high at a 4.4 mm gap" is audit jargon. The fact is decision-critical, though: the factor is **optimistic** at wide gaps, and the slice-1 clearance fix tells users to raise the gap.
  - It omits that the factor reads **low** with few poles: −3.4 % at 8 poles, −11.8 % at 6.
  - "3D" is undefined.
- **Watch-out**:
  - "(1.3 mm at the default pole pitch)" is a default literal. It must be live `{= c_end·τ_p}`: 1.32 mm at 10 poles, 1.65 mm at 8.
  - "about 3 mm" is a default literal. Use "about a third of a pole pitch" `{= τ_p/3}`.
  - "M1 audit M9" is an id.
  - Content is correct: f_end(2 mm) = 0.339 against 0.428 in 3D, so −21 %, and the check reads OK.
  - Missing direction: between c_end·τ_p and τ_p/3 the factor reads **low**, which is conservative. Only at or below c_end·τ_p is there no prediction.
- **Missing lever**: torque = σ·2πR_g²·(L − c_end·τ_p)·f_cal is exactly linear in L. Each mm adds `{= T_pull/(L − c_end·τ_p)}` (0.236 N·m/mm here), and the Overall length slider moves L one to one.
- **Missing independence (T6)**: the calibration prototype has its own **Original end-effect coefficient**. Editing End-effect coefficient does not change `calibration.f_end` (0.8982 stays when c_end = 0.3).

### (3) Minimum facts
1. Fringing at the ends; the model counts it as a lost length c_end·τ_p.
2. Torque grows with L − c_end·τ_p.
3. A shorter pole pitch (more magnets) loses less.
4. Validated against free-space 3D at 10 poles and a 1.4 mm gap: within 2 % for L ≥ 3.2 mm. These literals are fine when scoped.
5. No gap term: optimistic at wide gaps (+11 % at 4.4 mm), low with few poles.
6. Reads low below about τ_p/3, and gives no prediction at or below c_end·τ_p.

- Live: `{model.f_end}`, `{= coupling.c_end*model.pole_pitch_mm}`, `{= model.pole_pitch_mm/3}`, `{= model.pullout_Nm/(model.active_length_mm − coupling.c_end*model.pole_pitch_mm)}`.

### (4) What to change
- **Overall length** (Size tab) = **Axial magnet length, both rings**: longer brings f_end toward 1 and adds `{dT/dL}` N·m per mm. Trade-offs: axial space and housing; magnet mass; cold-high loads on the clamp, bond and gearbox rise with it.
- **Magnets per ring**: more magnets shorten τ_p, so less is lost at the ends. Trade-offs: fit, fill, S_1. This is not a free lever.
- **End-effect coefficient** (Assumptions): change it only with 3D or bench evidence. Higher is more conservative. It does not touch the calibration prototype's own coefficient.
- **Magnet face gap**: the factor ignores it. At gaps well above the 1.4 mm it was checked at, treat the torque as optimistic.

### (5) Draft rewrite
**Title:** "Magnet ends: the length lost to fringing"
1. The torque model works per unit length as if the rings were endless; at magnet ends some flux fringes outward instead of crossing the gap. (25)
2. It counts that as lost length: f_end = 1 − c_end·τ_p/L removes c_end·τ_p, `{= c_end·τ_p}` mm here, from the magnet length L. (21)
3. So torque grows with L − c_end·τ_p: each extra millimetre of magnet adds `{= T_pull/(L − c_end·τ_p)}` N·m here; a shorter τ_p loses less. (24)
4. A full 3D check without steel, at 10 poles and a 1.4 mm gap, found it within 2 % for magnets 3.2 mm or longer. (25)
5. Shorter than about a third of a pole pitch (`{= τ_p/3}` mm here) it reads well low, and at or below c_end·τ_p it gives no prediction. (25)

**Watch-out:** "It ignores the gap: in that check it read 11 % high at a 4.4 mm gap, so treat wide-gap torque as optimistic." (21)

**Keep linking:** `model.f_end`, `calibration.f_end`. Diagram EndFringing: add the "½ c_end·τ_p lost" brackets (TC-24).

**Consistency**: the `end_effect` assumption rationale (`assumptions.rs:59`) repeats "Within 1 % of 3D for magnets longer than about 3 mm". Fix it to the same wording.

---

## 6. calibration (Start here step 4; opens `model.f_cal`)

### (1) What the engineer needs
Whether the torque on screen is scaled by an assumed allowance or by the one bench test.
- Assumed allowance: every steel-backed design, the default included.
- Bench test: only no-steel designs that match the prototype.
- Which input makes the estimate more or less conservative.
- What not to touch.

### (2) Statement-by-statement
- **Title "The one-point calibration"**: acceptable but jargony.
- **s1 (54 words)**:
  - Audit ids (M4, M5) and "workbook".
  - "an assumed f_cal,0 (0.95 ...)" states an editable default as a constant (T6). It is **Original calibration coefficient**, an assumption.
  - "2D harmonic model" is undefined.
  - The physics is correct: about 4 % low in free space, smaller with steel. 0.95 is conservatism, not a bias correction.
- **s2**:
  - "the measured prototype's ... pole count" is editable: **Total installed magnets** / 2.
  - "the 1.8 N·m measurement" is editable: **Measured pull-out torque**.
  - "no back iron" should also cover a non-ferromagnetic **Back iron material**; the circuit record is `materials.circuit_backiron`.
  - "B842SH" is hard-coded, so it is fine as a literal.
  - T_model in the text is T_model^cal in the record.
- **s3**: correct (the cancellation), but it is a reviewer proof. Replace it with the consequence: changing Original calibration coefficient does not change prototype-matching designs.
- **s4**:
  - "keeps the assumed 0.95" is T6.
  - "does not compare gaps or radii" is correct but incomplete: magnet length is not compared either (PA-18; cut-to-length blocks keep their part name). Not comparing temperature is by design.
  - **Missing step (TC-14)**: in a no-steel design, leaving the prototype's magnet count switches f_cal from 1.0495 to 0.95. That is a −9.5 % step in every torque, from the rule rather than physics. It hits the owner's 10 → 8 case.
- **Watch-out**:
  - An audit trail ("correction E3", "M1 audit P2"), not actionable.
  - The one actionable part is buried: the 20 °C test temperature is the editable **Assumed test magnet temperature** ("ASSUMED; replace with the recorded value"). Entering 10 °C changes the bench factor to 1.0247 (−2.4 %).
- **Missing hazard (new, probe-verified)**: the prototype inputs describe the bench test. A user who "updates" them to match a new design (for example 8 poles with Total installed magnets 16) gets a matched factor of **1.57**: a meaningless +65 %.
- **Missing context**: the calculator holds no bench test of a steel-backed coupling. For the default design the factor is a judgement.

### (3) Minimum facts
1. Torque = model × f_cal.
2. Every design but a prototype match uses Original calibration coefficient (an assumption). Below 1 adds margin; there is no steel bench data behind it.
3. A match means: no steel behind the magnets, the prototype's magnets per ring (Total installed magnets / 2), and B842SH on both rings.
4. A match gets measured/uncalibrated-model, which reproduces the measured torque on the prototype.
5. The rule ignores ring radius, gap and magnet length.
6. Moving in or out of a match steps every torque by the ratio of the two factors.
7. The prototype inputs are bench facts, not design inputs.

- Live: `{model.f_cal}`, `{calibration.f_cal_updated}`, `{calibration.f_cal_original}`, `{calibration.measured_torque_Nm}`, `{calibration.poles_per_ring}`, `{calibration.model_error}`, `{calibration.test_temp_C}`.
- Literal OK: "B842SH".

### (4) What to change
- **Original calibration coefficient** (Calibration inputs; Assumptions "Calibration factor"): lower is more conservative for every non-matching design, the default steel design included; higher is less conservative. It has no effect on matching designs, because it cancels. Steel 0.95 → 1.0 gives pull-out 2.688 → 2.830. It is unvalidated either way.
- **Measured pull-out torque** and **Assumed test magnet temperature**: enter the recorded bench values. They affect matching designs only: 2.0 N·m gives f_cal 1.166; 10 °C gives 1.025.
- **Total installed magnets, Prototype inner hub apothem, Prototype magnet length/width/thickness, Reported spacing, Gap definition, Prototype remanence**: change these only to describe a real bench prototype you tested. Never use them to "match" the current design.
- **Back iron, Back iron material, Magnets per ring, Inner/Outer magnet part**: leaving the prototype match switches to the assumed coefficient. Expect a step, `{f_cal_updated}` → `{f_cal_original}`.

### (5) Draft rewrite
**Title:** "Calibration: what scales the torque estimate"
1. The calculator multiplies its torque model by a calibration factor f_cal; for most designs it is an assumed allowance, Original calibration coefficient (`{f_cal_original}`). (23)
2. The only bench test the calculator holds is a prototype without steel, so for steel-backed designs the factor is a judgement. (21)
3. A design gets the measured factor (`{f_cal_updated}`) only if it has no steel, `{N_cal}` magnets per ring and B842SH blocks on both rings, like that prototype. (26)
4. That factor makes the prototype's modelled torque equal its measured `{T_meas}` N·m; the rule ignores ring radius, gap and magnet length. (21)
5. Leaving that match, say another magnet count without steel, switches f_cal from `{f_cal_updated}` to `{f_cal_original}`: a step in every torque from the rule, not physics. (25)

**Watch-out:** "The prototype inputs describe that bench test; enter a real test's values, never edit them to match a new design, which gives a meaningless factor." (25)

**Keep linking:**
- `model.f_cal`, `calibration.f_cal_updated`, `calibration.measured_over_model`, `calibration.model_error`;
- optionally `calibration.model_torque_Nm`, which no note owns.

**Same theme in on-screen text, for item C:**
- The `pullout_iron_Nm` label "Same layout with steel back iron (factor 0.95)" (`model.rs:336`) uses the editable f_cal,0, so the "0.95" goes stale.
- The `f_cal` help "0.95 for the steel-backed candidate." (`model.rs:235`) has the same problem.
- The `f_cal_updated` help "Applies to the no-iron prototype only." (`calibration.rs:97`) is wrong: resized matching designs get it too.
- The calibration_factor rationale "the prototype's own rings" (`assumptions.rs:66`) overstates the rule (PA-18).

---

## 7. br_temperature (Start here step 6; opens `model.pullout_20C_Nm`; dashboard Why for "Pull-out torque at 20 °C")

### (1) What the engineer needs
Set the operating and minimum magnet temperatures honestly.
- Torque falls reversibly by about α_i + α_o per °C (0.24 %/°C for the NdFeB default).
- The hot case limits the torque, and the cold case sets the loads.
- Know which temperature each torque row uses, and which α a ring actually uses.

### (2) Statement-by-statement
- **Title**: "torque ∝ Br²" renders as "torque ~ Br²" (TC-18). "Remanence" is jargon in link text.
- **s1**:
  - "α about −0.12 %/°C for sintered NdFeB" is fine as a typical value.
  - But the α used is an editable input (**Reversible Br temperature coefficient**, an assumption) for library parts, and the grade's own value for grade-mode rings. Neither is said (T6).
  - The definition of remanence is acceptable.
- **s2**: correct. "the square of Br when both rings are one grade" is better read as "when both rings share one coefficient". Two library parts of different grades still share the one α input.
- **s3**: DEFAULT-ONLY. "a 30 °C rise costs about 7 %" is the default α at the default 50 °C. Make it live: `{= model.pullout_Nm/model.pullout_20C_Nm}`, which is exactly the gap between the dashboard's two pull-out rows (TC-18).
- **s4 (53 words)**:
  - It front-loads demagnetization ("onset", "reverse field", "knee", all undefined here).
  - "10 °C" is the editable **Design margin below the onset** (T6).
  - "the lowest onset, the one while the coupling skips" holds only for the stored 3D reverse fields, which never follow the design (spec 8 trap). The limit uses the skipping onset by construction.
  - "cool the magnet and the torque comes back, unless on the way it passed a demagnetization onset" assumes heat. For ferrite the risk is cold.
- **Watch-out "Each ring keeps its own coefficient"**: MISLEADING as general. Only a ring in grade mode (manual dimensions plus a picked grade) keeps its own. Library parts all use the one Reversible Br temperature coefficient. Probe: setting it to −0.002 moves an outer library ring but not an inner grade-mode N42SH ring. The ferrite figure (Y30 −0.2 %/°C) is right, and ferrite is only reachable in grade mode.
- **Missing (PA-4)**: which temperature each owned row uses.
  - Pull-out and hot-low: **Operating magnet temperature**.
  - Cold and cold-high: **Minimum magnet temperature**.
  - Hot-day torque: **Hot-day ambient temperature** + **Coupling rise above ambient while driving** (65 °C by default).
  - Peak torque: after a slip event.
  - Torque at the limit: the demagnetization limit, which itself uses stored reverse fields.
- **Missing lever / trade**:
  - Low-coefficient grades (SmCo, about −0.035 %/°C) lose less with heat but start weaker. Both rings SmCo_2_17_30 give 1.89 N·m at 120 °C against N42SH's 2.24.
  - Cold makes the coupling stronger: cold-high, not nominal, sizes the clamp, bond and gearbox.

### (3) Minimum facts
1. Br(ϑ) = Br20·(1 + α(ϑ − 20 °C)): linear and reversible.
2. Torque is proportional to Br_i·Br_o, so it changes by about α_i + α_o per °C.
3. The ratio at the operating temperature is the gap between the two dashboard pull-out rows.
4. Each row has its own temperature (list above).
5. Reversible unless the magnet starts to demagnetize: too hot for NdFeB, too cold for ferrite.
6. Library parts share the one α input; grade-mode rings use the grade's.

- Live: `{model.inner_alpha_br_per_C}`, `{model.outer_alpha_br_per_C}`, `{coupling.op_temp_C}`, `{= model.pullout_Nm/model.pullout_20C_Nm}`, `{metal.min_temp_C}`, `{temperature.duty.hot_day_start_C}`.
- Literals OK: 20 °C (the reference temperature is fixed in code), "about −0.12 %/°C typical sintered NdFeB", "about −0.2 %/°C ferrite".

### (4) What to change
- **Operating magnet temperature**: set it to the magnet temperature while running. Higher lowers the pull-out and hot-low, by about `{α_i+α_o}` per °C. Trade-off: an honest margin against a wish. The hot-day check is separate and uses ambient + driving rise.
- **Minimum magnet temperature**: set it to the coldest real case. Lower raises the cold and cold-high torque, so the clamp, bond and gearbox loads rise.
- **Reversible Br temperature coefficient** (Assumptions, "Br temperature coefficient"): take it from the magnet datasheet. Only library parts use it.
- **Inner/Outer magnet grade (manual dimensions)**: a lower-coefficient grade (SmCo) holds torque better with heat but has lower Br. A higher-rated NdFeB grade keeps α here and raises the demagnetization limit instead.
- **Hot-day ambient temperature**, **Coupling rise above ambient while driving**: these set the hot-day torque row.

### (5) Draft rewrite
**Title:** "Heat weakens magnets: torque against temperature"
1. A magnet's remanence Br, its built-in flux density, falls reversibly as it warms: Br = Br20·(1 + α(ϑ − 20 °C)), α negative. (23)
2. Torque is proportional to inner Br times outer Br, so each degree of warming changes it by about α_i + α_o (`{= 100(α_i+α_o)}` %/°C here). (25)
3. At the Operating magnet temperature (`{ϑ_op}` °C) that leaves `{= T_pull/T_pull,20}` of the 20 °C pull-out: the gap between the dashboard's two pull-out rows. (23)
4. Each torque row uses its own temperature: hot-low the operating one, cold-high the Minimum magnet temperature, the hot-day row ambient plus the driving rise. (24)
5. The change is reversible unless the magnet starts to demagnetize (too hot for NdFeB, too cold for ferrite), which costs strength for good. (23)

**Watch-out:** "Library magnets all use one Reversible Br temperature coefficient input; only a ring with manual dimensions and a picked grade uses its grade's own coefficient." (25)

**Keep linking:**
- `model.br_inner_T_op`, `model.br_outer_T_op`, `model.pullout_20C_Nm`, `temperature.magnet_life.torque_hot_day_Nm`, `temperature.magnet_life.torque_peak_Nm`, `temperature.demag.torque_at_limit_Nm`;
- `metal.torque_cold_Nm`, unless the new hot-low/cold-high note takes it (X5);
- add `model.alpha_br_per_C`, `model.inner_alpha_br_per_C`, `model.outer_alpha_br_per_C`.

**Same theme in help, for item C:**
- The `coupling.op_temp_C` help "torque ~ Br²" has the glyph problem.
- The `metal.required_min_Nm` help "At the 50 °C magnet temperature." is stale when op_temp changes (PA-4).

---

## 8. Physics traps

### Spec section 8 traps, checked against these six notes and the drafts

| Trap | Where it threatens | Draft status |
|---|---|---|
| Radius is the strongest lever at a fixed pole count | pullout_angle "what to change" | Avoided: length is listed first, with its live dT/dL |
| Fewer poles keep the same size, or always improve torque per mass | harmonics watch-out; back_iron s5 | Avoided: "the ring does not shrink" (not "same size"); "S_1 has a best pitch" |
| Pull-out is guaranteed | pullout_angle s1 | Avoided: "a nominal estimate" |
| The rings re-lock as soon as the load falls below pull-out | pullout_angle s5 and watch-out | Avoided: "may keep slipping ... re-locks only once the slip speed falls" |
| Bench calibration beyond prototype-matching designs | calibration s3 | Avoided: "only if ... like that prototype" |
| 416 is a safe back-iron fix | back_iron "what to change" | Flagged: do not list 416 |
| Stored 3D demag fields follow geometry or grade | br_temperature s4 (old) | Avoided: the draft no longer mentions onsets or fields |

### New traps found in this review (add to the reviewer's list)

1. **"Higher harmonics matter little above all because they fade across the gap."** At the default the amplitude product (≈1/40) outweighs the gap fade (≈1/2 to 1/6).
2. **"The harmonic setting does not matter."** At low fill (few poles, a large ring, narrow blocks) the third harmonic moves the pull-out by 12 to 33 %, in either direction.
3. **"Fewer poles reach across the gap better"** (TC-11). Per mm of gap they lose less, but S_1 with steel and the fill fall, so the net torque falls at the default.
4. **"Thicker magnets give diminishing returns"** must not be written without noting that the tool also grows the gap radius and the cup. Pull-out rises +53 % from 3.17 to 6.35 mm.
5. **"The end factor is accurate/conservative."**
   - It is optimistic at wide gaps (+11 % at 4.4 mm), exactly where the slice-1 clearance fix ("Raise Magnet face gap") sends users.
   - It is conservative, not optimistic, for short magnets above c_end·τ_p.
6. **"Edit the Calibration inputs to match your design."** That gives meaningless factors (1.57 at 8 poles with Total installed magnets 16).
7. **"Each ring uses its own α."** Only grade-mode rings do; library parts share one input.
8. **"Steel back iron gives the torque shown."** Only if the wall does not saturate, and the default fails the cup-wall check.
9. **"Re-lock is impossible until the load is below the drag"** is also too strong. Inertia and drive slowdown matter. Say "once the slip speed falls".
