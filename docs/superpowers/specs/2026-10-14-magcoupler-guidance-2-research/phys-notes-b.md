# Physics review, slice 2, notes-b: demagnetization, ferrite_cold_demag, slip_heating, thermal_time_constant, clamp_preload

Reviewer: session-model physics pass (read-only; nothing in the repo was changed; no subagents).
Read: `src/engine/explain/notes.rs` (the five notes), `engine/temperature.rs` (inputs :74-110, :215-291; `demag_onset_C` :698, `cold_onset_C` :717, `ring_demag` :763-862, `compute` :895-1423), `engine/clamps.rs`, `engine/metal_design.rs`, `engine/grades.rs`, `engine/library.rs`, `explain/records/{demagnetization,slip_heating,clamps}.rs`, `explain/symbols.rs`, `gui/guidance.rs` (slice-1 guide text, reviewed vocabulary), `gui/inputs.rs` (on-screen groups), `gui/diagrams.rs`, the audit (`audit.json`: TC-5, TC-7, TC-8, TC-15, TC-16, TC-19, TC-20, TC-23, TC-24, TC-31, TC-32, TC-33, PA-1, PA-2, PA-10, PA-15, PA-16, PA-17, PA-19, CG-5, CG-13, CG-21), the M1 math audit rows E5, M11-M14, P1, and the slice-1 spec section 8.

## 0. Numbers this review checked (default design unless stated)

Hand-computed from the engine formulas and defaults; the starred ones are pinned by a test or an earlier probe.

| Quantity | Value | Source |
|---|---|---|
| Governing temperature limit (magnets govern) | 93.056 °C* | tests/deviations.rs:1967 |
| Hot-day start (55 °C ambient + 10 °C driving rise) | 65 °C | temperature.rs:906 |
| High-case steady temperature, continuous slip | 92.513 °C* | tests/deviations.rs:1992 |
| Gap between the two | 0.54 °C | difference |
| Conductance at which "time to limit: never" flips to a finite time | 0.2942 W/K (1.9 % below the 0.3 default) | P_hi / (limit - start) |
| Critical drag vs high-case drag | 0.0402 vs 0.0394 N·m | critical_drag_Nm vs 3 × drag |
| Total slip loss, mean drag | 2.751 W*, 0.0131 N·m (0.5 % of the 2.69 N·m pull-out) | PA-1 probe, M1 E5 row |
| Steel skin depth / 316L skin depth at 166.7 Hz | 1.2995 mm* / 33.6 mm | probe out.txt / hand |
| Thermal time constant | about 270 s (C about 80 J/K) | M1 placeholder table (273.98 s, workbook) |
| Rise per default 0.1 s event, high case / 2 s fault | about 0.01 °C / about 0.2 °C | P·t/C, rise_h(1 - e^(-t/τ)) |
| N42SH reference knee temperature t_ref, offset | 159.9 °C, +9.9 °C | demag_onset_C with h_ref = 517 kA/m |
| N42SH skipping onset, limit | 103.1 °C, 93.06 °C | matches the pinned 93.056 |
| N38UH limit (custom dimensions) | 131.9 °C (124.6 °C once E21 clamps its -7.3 °C offset) | hand |
| N35EH limit | 155.4 °C (151.9 °C with E21) | hand |
| N52 limit | 0.17 °C (offset -9.69 °C) | PA-16 probe, hand |
| Y30 skipping cold onset, cold limit | 325.7 °C, 335.7 °C* | PA-17 probe, hand |
| Y30 aligned-field crossing | 170.6 °C (so 354 kA/m also exceeds the knee up to there) | hand |
| Clamp: one M4 12.9 per-screw torque vs requirement | 7.665 vs 7.647 N·m (SF on coupling 2.005) | clamps.rs compute |
| Same clamp at μ = 0.10 | 5.11 N·m per screw: needs 2 M4, 1 fits, no size works ("None: enlarge the boss or the clamp length") | hand |
| Clamp boss 22 mm (= Shaft boss outside diameter) with the 10 mm clamp | no size works | clamps.rs help, hand |

## 1. demagnetization ("Demagnetization: knee, permeance and the onset temperatures", notes.rs:228-265)

### 1.1 Decision it supports
Is each magnet grade safe at the hottest temperature it can reach while the coupling slips, and if not, which grade (or cooling) to move to, at what cost in torque.

### 1.2 Problems in the current text
- **s3, T6, misleading.** "like poles of the other ring facing it while the coupling skips push it highest, 863 kA/m here": 863 kA/m is the stored input "3D worst reverse field, like poles facing" (temperature.rs:99), a 3D result for the default design (10 poles, 10.15 mm apothem, 1.4 mm gap, B842SH at 1.29 T). It does not follow pole count, ring size, gap, magnet size or grade; "here" reads as "your design" (TC-5, TC-8, PA-1). The engine only applies the ring's own (1 + αΔT) to it; it never rescales with Br at 20 °C, so it reads low for a higher-Br grade (N52: non-conservative) and high for a lower-Br grade (UH/EH: conservative). Spec section 8 trap 6 sits exactly here.
- **s3, true only with recoil permeability 1.** "a reference magnet on the load line B = -μ0 H (permeance coefficient 1) sees Br/(2 μ0)": that is the engine's choice (temperature.rs:785), but the grades carry μ_rec = 1.05 (grades.rs:377), which gives Br/(2.05 μ0), 2.4 % lower. Fine to drop from a plain note; do not restate as physics.
- **s4, unvalidated model choice stated as fact (PA-16).** "The onsets are shifted so the reference magnet reaches its knee exactly at its rated temperature": correct description of the code, but it assumes the vendor rating is a Pc = 1 knee rating (never shown for any grade). For N52, N50, N50M, N38UH, N35EH and 2:17 SmCo the offset is negative, so the shift *raises* the onsets (N38UH +7.3 °C, N52 +9.7 °C). E21 (clamp the offset at 0) is user-approved but not landed (no E21 in deviations.rs; 04-memory.yaml:46). A rewrite must not call the shift conservative.
- **s2, s4, PA-10.** "taken as H_k = 0.9 · Hcj" and "keeps a 10 °C margin" state editable assumptions as constants (Knee field as a fraction of Hcj; Design margin below the onset). "β about -0.5 %/°C" is N42SH's; the NdFeB grades span -0.375 to -0.675 %/°C.
- **s3, "the skipping onset sets the limit" because it is highest.** It sets the limit by code (h_rev_likepole always), not by comparison; true for NdFeB only (a ferrite ring's hot limit is its rating).
- **s5, audit id.** "With correction E20". Also: both rings use the same stored fields (the outer blocks' values; the inner ring's own 341/760/844/545 kA/m in the help text are not used), so "each ring with its own grade" is right but the field is shared (conservative for the inner ring). Not worth prose; keep as a fact.
- **Watch-out, audit id and default-only.** "(M1 audit M13)"; "the default N42SH ... moves even the default limit by up to about 3 °C" is a default-design number (true: 89.6-94.4 °C). The extrapolation point itself should stay user-visible in short form (TC-5 verifier): higher-rated grades push onsets past their coefficient ranges.
- **Terms before definition.** intrinsic curve, Hcj, knee (only as 0.9·Hcj), permeance, load line, permeance coefficient, μ0, "skips", rated temperature. Sentences of 58 and 53 words.
- **Missing.** No lever (which input, which way, what it costs). No statement that the limit is what the hot-day start and the slip peak are compared with.
- **Title** names the math (TC-31).
- **Sources line** shows "M1 audit M13", "Addendum A decision 19 (E20 ...)" on screen (CG-21).

### 1.3 Facts a plain rewrite must keep
1. A reverse field (one opposing the magnet's magnetization) does no lasting harm until it passes the knee; past it, part of the magnetization is lost for good (not recovered on cooling, unlike the reversible Br loss of br_temperature).
2. NdFeB's knee falls as it heats, so each reverse field has an onset temperature. (The reverse field also falls a little with Br; the equation shows both coefficients. Optional in prose.)
3. The largest reverse field is while slipping, like poles face to face; the calculator always sets an NdFeB ring's limit from that case, because the coupling is meant to slip under overload.
4. Onsets are shifted to agree with the grade's rated maximum temperature; a magnet with no rating ("n/a") gets no shift.
5. Limit = slipping onset less the design margin; each ring with its own grade; the lower limit governs.
6. The four reverse fields are stored 3D results for the default design (watch-out).
7. Onsets beyond a grade's tested coefficient range are extrapolations (optional second watch-out line).

Numbers: from results only: the onsets, the limit, the offset, h_ref, the governing ring. From inputs (live, not literals): the design margin, the knee fraction, the reverse fields. Fine as literals: none needed; if a typical β is quoted, "about -0.5 %/°C for N42SH" is acceptable as a grade fact. Provenance of the stored fields (default design, 10 poles, N42SH) is fine as a literal because it is fixed data, not the user's design.

### 1.4 What to change
| On-screen input | Direction | Effect | Trade-off |
|---|---|---|---|
| Inner magnet part / Outer magnet part -> "Custom dimensions", then Inner/Outer magnet grade (manual dimensions) | a grade rated hotter: N38UH (180 °C), N35EH (200 °C), N33AH (220 °C), SmCo | raises that ring's limit (N38UH about +39 °C, N35EH about +62 °C here) | lower Br (1.22, 1.17, 1.14 T vs 1.30): torque about -12 %, -19 %, -23 % with both rings changed; cost; the manual block dimensions must be typed (the B842 library family stops at SH) |
| Design margin below the onset | lower | raises the limit 1:1 | less safety; it is an assumption |
| Knee field as a fraction of Hcj | higher (toward 1) | raises onsets | less conservative; an assumption |
| Coercivity for the demagnetization check -> "Hcj and beta inputs" + Intrinsic coercivity Hcj at 20 °C / Hcj temperature coefficient | to a specific vendor datasheet | uses that part's real Hcj and β for both rings | overrides both grades; the β slider clamps to negative values |
| The four "3D worst reverse field ..." inputs (group "3D results: stored reverse fields") | enter values from a 3D calculation of your design | makes the limit belong to your geometry and grade | needs an outside 3D/FEA run; no in-app solver yet |
| Hot-day ambient, Coupling rise above ambient while driving, Thermal conductance, Slip fault trip time | cooler / shorter | moves the temperature the limit is compared with, not the limit | see thermal note |

Do not list as levers in this build: thicker magnets, a smaller gap, fewer poles, a weaker opposite ring. Physically they change the reverse field; the calculator does not show it (fields are stored).

### 1.5 Draft rewrite
**Title:** Demagnetization: when heat and slipping weaken magnets for good

1. A field pushing against a magnet's direction, a reverse field, does no lasting harm until it passes the knee, where the magnet stops springing back.
2. Heat lowers an NdFeB magnet's knee, so each reverse field has an onset temperature; above it, part of the magnetization is lost for good.
3. The reverse field is strongest when the coupling slips and like poles pass face to face, so that case sets an NdFeB ring's magnet limit.
4. Onsets are shifted to match the grade's rated maximum temperature, and the limit is the slipping onset less the design margin.
5. Each ring is checked with its own grade and the lower limit governs; a grade rated hotter raises it, usually at the cost of Br and so torque.

**Watch out:** The reverse fields are stored 3D results for the default design: they do not follow pole count, ring size, gap, magnet size or grade, and read low for a higher-Br grade. Enter fields from a 3D calculation of your design. (Optional second line: Onsets above a grade's tested temperature range are extrapolated.)

**Equations:** keep all 11 (`temperature.demag.h_ref_kA_m`, `t_ref_model_C`, `calibration_offset_C`, the four onsets, `magnet_limit_C`, `inner_magnet_limit_C`, `outer_magnet_limit_C`, `demag_ring`; START_HERE opens `onset_skipping_C`). Add (unowned today): `temperature.demag.inner_hcj20_kA_m`, `outer_hcj20_kA_m`, `inner_beta_per_C`, `outer_beta_per_C`, `hcj20_used_kA_m`, `beta_used_per_C`, `temperature.summary.onset_skipping_C`, `temperature.summary.magnet_limit_C`, `temperature.magnet_life.margin_onset_C`, `temperature.magnet_life.margin_limit_C`.

**Diagram:** DemagKnee (TC-23): two intrinsic curves (20 °C and hot) with the knee moving toward the slipping operating point. If any load line is labelled "permeance coefficient 1", draw it in the J-H plane as J = -2 μ0 H ((Pc + 1) μ0), not -μ0 H; the note's own B = -μ0 H is the B-plane form.

**Gate note:** the dashboard's "Temperature limits" Why link lands on this note (guidance.rs:197), yet the check also judges the adhesive limit, the hot-day start, the slip peak, the cure margin and the cold side. The rewrite's sentence 3-4 cover the magnet part only; the thermal note covers the peak. Consider linking the guide's Learn to the thermal note when only a peak term fails (code-only).

## 2. ferrite_cold_demag ("Ferrite: the demagnetization risk is at cold", notes.rs:266-299)

### 2.1 Decision it supports
If a ring is ferrite, does it survive the coldest magnet temperature while slipping, and what must be supplied or changed for that answer to mean anything.

### 2.2 Problems in the current text
- **Watch-out, reads as a bug (TC-19) and default-only.** "put a Y30 ring's cold limit at 336 °C": correct (335.7 °C) but a "cold" limit above room temperature reads as a typo; it means the ring is past its knee at every temperature it is rated for (the skipping crossing is 325.7 °C, above Y30's 250 °C rating). Say that in words; take the number from results if quoted.
- **Watch-out, T6.** "the defaults (354 to 863 kA/m) come from the NdFeB default design": right, but the values are live inputs a user may already have changed; phrase as "the default reverse fields". Do not promise that a ferrite design's own fields will pass: a rough Br scaling of 863 kA/m (×0.37/1.29, about 250 kA/m) still exceeds Y30's 162 kA/m knee at 20 °C, so thin ferrite blocks are genuinely at risk while slipping.
- **Watch-out, default-only.** "still fail at -40 °C": -40 °C is the default of "Minimum magnet temperature".
- **s3, PA-10.** "plus the 10 °C margin" is the editable Design margin below the onset.
- **Missing (PA-17).** Y30's coefficients are sourced for 20-120 °C (grades.rs:587); the cold check extrapolates them to the minimum temperature, and with stored NdFeB fields above 120 °C too. The engine never reads `coefficient_range_C`.
- **Missing (PA-2 trap).** How to model ferrite: the Hcj coefficient help says to type a positive β, but GUI edits clamp to [-0.008, -0.001], so the cold check never switches on that way. The working route is a Y30 grade on Custom dimensions with Coercivity for the demagnetization check = "magnet grade". The note must not repeat the typed-β route.
- **Missing (TC-32).** START_HERE step 8 opens `cold_check`, which reads "n/a (coercivity rises as the magnet cools)" on the default NdFeB design; the note never says the check is ferrite-only.
- **Terms.** "coercivity", "skipping cold onset", formula symbols in prose (ϑ = 20 + (H_k - H)/(H α - H_k β)) for a non-specialist; fine in the equation panel.
- Physics of s1-s3 is right: β > 0 so the knee falls on cooling (temperature.rs:717-720, 784-845), Br rises on cooling, the hot limit is the rating (temperature.rs:815-819), the cold side uses the skipping field plus the margin (838-841), and it passes only if both rings pass (965).

### 2.3 Facts a plain rewrite must keep
1. Ferrite's knee falls as it cools (opposite to NdFeB) while its field, and so the reverse field, grows.
2. Below a cold onset the ring loses magnetization for good; the slipping case sets it.
3. Cold limit = skipping cold onset + design margin; Minimum magnet temperature must not be below it; both rings must pass.
4. A ferrite ring's hot limit is its rated maximum temperature.
5. NdFeB on both rings: the check does not apply ("n/a").
6. The default reverse fields are the NdFeB design's and fail Y30 at any rated temperature; enter the ferrite design's own.
7. Y30 coefficients are tested 20-120 °C only.

Numbers: from results: cold onset, cold limit, the cold ring. From inputs: Minimum magnet temperature, design margin, reverse fields. Fine as literals: Y30 β about +0.35 %/°C (grade data), 20-120 °C range (grade data).

### 2.4 What to change
| On-screen input | Direction | Effect | Trade-off |
|---|---|---|---|
| Inner/Outer magnet part -> "Custom dimensions" + Inner/Outer magnet grade (manual dimensions) = Y30 | to model ferrite at all | switches on the cold check for that ring | the only ferrite grade in the table |
| The four "3D worst reverse field ..." inputs | values from a 3D calculation of the ferrite design | makes the cold limit mean something | needs an outside 3D run; may still fail |
| Minimum magnet temperature | raise only if the application never gets that cold | passes a cold check that is close | it is a requirement, not a design choice |
| Design margin below the onset | lower | lowers the cold limit 1:1 | less safety |
| Ring grade back to NdFeB | change | removes the cold risk | hot-side risk, cost, much higher torque and stray field |

Do not list: typing a positive Hcj temperature coefficient (clamped away); thicker ferrite blocks as a calculator lever (physically they help; the stored fields do not move).

### 2.5 Draft rewrite
**Title:** Ferrite magnets: the demagnetization risk is at the cold end

1. Ferrite works the opposite way to NdFeB: its knee falls as it cools, while its own field, and so the reverse field, grows.
2. Below some temperature the reverse field passes the knee, and the ring loses part of its magnetization for good.
3. The calculator finds that cold onset for the slipping case and adds the design margin: the Minimum magnet temperature must stay above that cold limit.
4. Heat moves a ferrite knee away from the reverse field, so a ferrite ring's hot limit is simply its rated maximum temperature.
5. With NdFeB on both rings the cold check does not apply and reads n/a.

**Watch out:** The default reverse fields come from the NdFeB design and exceed a Y30 knee at every rated temperature, so the check fails however warm the ring runs: enter your ferrite design's own. Y30's coefficients are tested only from 20 to 120 °C.

**Equations:** keep all 9 (`temperature.demag.inner_cold_limit_C`, `outer_cold_limit_C`, `cold_ring`, `cold_limit_C`, `cold_check`, the four cold onsets). START_HERE opens `cold_check`: prefix its bar with "(ferrite rings only)" (code-only, TC-32).

**Diagram:** a cold variant of DemagKnee (20 °C and cold curves, knee moving toward the operating point on cooling), or no diagram; reusing the NdFeB picture teaches the wrong direction.

## 3. slip_heating ("Eddy-current slip loss and skin depth", notes.rs:300-335)

### 3.1 Decision it supports
How much heat a slip makes, which parts make it, and when to replace the estimate with a measured drag (the thermal note then decides whether that heat is a problem).

### 3.2 Problems in the current text
- **Missing, T6 (PA-1).** The field inputs (b_hub_T, b_cup_T, b_sleeve_T, b_liner_T, cap and web integrals, b_magnet_T, and the three free-space fields) are stored 3D results for the default design. Only the radii, the length, the pole pairs and the speeds update (temperature.rs:1104-1184); at a 15 mm apothem the loss moves 2.751 -> 5.397 W with every field held fixed. The note says nothing; this is the slip-loss twin of trap 6.
- **s3, model presented as physics (PA-19).** "the steel loss grows as speed to the power 1.5 rather than 2": the engine uses the thin-skin limit σω²B²δ/(4k²), valid for kδ << 1; at the defaults kδ = 0.36-0.64, it reads 6-13 % high and the true exponent is 1.54-1.64 (M1 audit M11). Say "about" or "in this model".
- **s3, default-only.** "about 1.3 mm at the default slip": a result (`skin_depth_mm`), and it moves with pole count, slip speed and back-iron material (8 poles: 1.45 mm).
- **s3, symbol clash and undefined terms (TC-15, PA-15).** Bare σ is conductivity here but shear stress in step 1; μr (the steel's *incremental* relative permeability, an input) is undefined; the panel writes σ_st, σ_sl, μ_r, f_thin.
- **s4, default-only and PA-10.** "about 34 mm in 316L" is at the default 166.7 Hz (33.6 mm); the qualitative claim (shells far thinner than their skin depth) holds over every slider value (at 1000 Hz, 316L is still 13.7 mm against a 1 mm maximum shell), so state it qualitatively. "the 0.7 end factor" is the editable End factor for thin shells and the cap. The shell formula and its speed-squared law are right; the shell's magnetic Reynolds number is about 5e-4 at the defaults, so neglecting the reaction field is sound.
- **s1, 65 words; omits the rear web** although `web_W` is linked.
- **Watch-out, jargon and a small inaccuracy.** "resistance-limited, not skin-limited", "(correction E17)". "Aluminium parts (no back iron)": with a 304 back-iron pick the body parts are 304, not aluminium; say "non-magnetic metal". "factor-3 high case" is the editable High-case multiplier on the estimate. The E17 point itself must survive in plain words (TC-15 verifier): without back iron the hub, cup and web see the weaker free-space field and heat through their thickness.
- **Missing, what to change.** No lever named. A measured drag replaces both the estimate and its high case (temperature.rs:1187-1190), the single most useful action.
- **Missing, consequence.** The drag is the average torque the coupling still passes while slipping (0.013 N·m against a 2.69 N·m pull-out at the defaults): the bridge to the "after the coupling slips" note.

### 3.3 Facts a plain rewrite must keep
1. Slip sweeps each ring's poles past the other ring's metal; eddy currents heat the cup, hub, rear web, sleeve, liner, cap and magnets.
2. Power = drag torque × slip speed; the drag is the average torque passed while slipping.
3. Thick steel heats in a surface skin: loss about speed^1.5 in this model; thin parts and magnets: speed squared.
4. Without back iron the body parts are non-magnetic metal, see a weaker field and heat through their thickness.
5. The fields are stored 3D results for the default design (watch-out).
6. A high case = estimate × High-case multiplier; a measured drag replaces both.

Numbers: from results: skin depth, per-part losses, total, drag, used and high powers. From inputs: end factor, high-case multiplier, slip speed, materials. Fine as literals: the exponents 1.5 and 2 (as model statements), "far thinner than their skin depth" (qualitative).

### 3.4 What to change
| On-screen input | Direction | Effect | Trade-off |
|---|---|---|---|
| Measured mean slip drag torque | enter the bench value | replaces the estimate and the high case | needs a slipping bench test |
| High-case multiplier on the estimate | 1 once measured; leave at its default until then | scales the high case | the multiplier is the only allowance for the model's errors |
| Sleeve and liner material | Ti-6Al-4V (σ 0.60 MS/m) or Inconel 625 (0.78) instead of 316L (1.35); PEEK about 0 | cuts the sleeve and liner share in proportion to conductivity | strength, stiffness, cost; PEEK's temperature and creep limits; the share is small at the defaults (about a tenth of the total by hand estimate; read the per-part results) |
| Inner rotating retaining sleeve thickness / Outer rotating keeper liner thickness | thinner | cuts their loss in proportion, and opens running clearance | retention strength |
| Relative slip speed | lower | loss falls as speed^1.5-2 | an operating condition, not usually a design choice |
| End factor for thin shells and the cap | only to a better-founded value (geometry gives 0.65-0.68 for the shells, M1 P1) | small | an assumption |

Do not list: a different back-iron steel to cut loss (416 trips the low-saturation caution, trap 5; the loss goes as the square root of σ/μr and the stored fields do not change); fewer or more poles (with fixed fields the steel loss goes as p^-0.5 and the real fields change too; traps 1 and 2).

### 3.5 Draft rewrite
**Title:** Slip heating: eddy currents when the coupling slips

1. When the coupling slips, each ring's poles sweep past the other ring's metal parts, and the changing field drives eddy currents that heat them.
2. The heat power equals the drag torque times the slip speed; that drag is the average torque the coupling still passes while it slips.
3. Loss grows with slip speed: about to the power 1.5 in thick steel, which heats only in a thin surface layer, and squared in thin parts and magnets.
4. Without back iron the hub, cup and web are non-magnetic metal: they see a weaker field and heat through their whole thickness.
5. The per-part losses show what dominates; a less conductive or thinner sleeve and liner cut their share, and a measured drag replaces the whole estimate.

**Watch out:** The estimate uses field values stored from a 3D run of the default design: they ignore pole count, ring size, gap, magnet size and grade. Until you enter a measured drag, the high case multiplies it by the High-case multiplier.

**Equations:** keep all 9 (`temperature.duty.field_omega_rad_s`, `temperature.slip_loss.skin_depth_mm`, `hub_W`, `cup_W`, `web_W`, `cap_W`, `magnets_W`, `total_W`, `drag_Nm`; START_HERE opens `total_W`). Add (unowned): `temperature.slip_loss.used_W`, `temperature.slip_loss.high_W`, `temperature.duty.field_freq_Hz`, `temperature.duty.pole_pairs`, `temperature.duty.slip_rad_s`. (`sleeve_W` and `liner_W` stay with `a5.high_conductivity_sleeve_or_liner`.)

## 4. thermal_time_constant ("The thermal time constant", notes.rs:336-370)

### 4.1 Decision it supports
Can the coupling survive unbroken slip (a fault or a stall), how long until it reaches its limit, and which lever (measure G, measure the drag, trip sooner, cool the start) changes the answer.

### 4.2 Problems in the current text
- **s2, s4, symbol mismatch (TC-20, PA-15, tracked as M43-13).** τ and ϑ_start where the panel shows τ_th and ϑ_hot (records/slip_heating.rs:231; demagnetization.rs:221).
- **s4, undefined and audit id.** "the limit temperature" is the governing limit, the lower of the magnet design limit and the adhesive design limit (records: `temperature.summary.governing_limit_C`); "(correction E12)".
- **Watch-out, default-only and audit id.** "a 10 % higher G lowers the high-case steady temperature by about 2.5 °C (M1 audit placeholder table)": true at the defaults (27.5 -> 25.0 °C rise) but scales with the loss. The general fact is "steady rise ∝ 1/G". It also misses the sharper default-design fact: the high-case steady temperature is 0.54 °C under the limit, so "never" flips to a finite time if G is 1.9 % lower than the 0.3 W/K placeholder (or the critical drag 0.0402 N·m is compared with the 0.0394 N·m high-case drag). That gap must come from results, not a literal.
- **Watch-out, input not named (TC-20).** G is "Thermal conductance to the housing and shafts" (Key design).
- **Missing.** The peak the verdict checks (`temperature.magnet_life.peak_C`: hot-day start + fault-limited slip or one event, whichever is hotter, + the life-average slip heating) is never explained; the HeatingCurve diagram has no limit line (TC-24).
- **Missing caveat (model limit, PLAUSIBLE).** One lump means both rotors and the cap at one temperature. The loss is deposited in the cup's steel skin, the sleeve, liner and magnets, next to the magnets and the bond; for events much shorter than τ_th (the 2 s fault) the parts near the gap run hotter than the lump average. At the defaults the effect is tenths of a degree, so it is a watch-out candidate, not a correction (the M1 audit lists the single-lump network as "not independently validated").
- s1-s3 physics is right: τ_th = C/G; 63 % at τ, 95 % at 3τ; the steady rise P/G is independent of C; one short event adds P·t/C (temperature.rs:1234-1296). The time-to-limit cases (never; 0 at or above the limit) match the code.

### 4.3 Facts a plain rewrite must keep
1. One lumped heat capacity C and one conductance G.
2. Unbroken slip: rise toward P/G above the hot-day start, 63 % at τ_th = C/G.
3. G sets the steady temperature (for a given loss); C only sets the speed; one short slip adds P·t/C.
4. The check uses the peak (hot-day start + fault + life-average slip heating).
5. Time to the limit: to the lower of the magnet and adhesive limits; "never" if steady stays at or below it (0 if the start is already there: optional).
6. G is a placeholder; the steady rise scales as 1/G (watch-out).

Numbers: from results: τ_th, t95, steady temperatures, the governing limit, time to limit, peak, critical drag. From inputs: G, hot-day ambient, driving rise, fault trip time. Fine as literals: 63 %, 95 %, 3τ.

### 4.4 What to change
| On-screen input | Direction | Effect | Trade-off |
|---|---|---|---|
| Thermal conductance to the housing and shafts | measure it; raise it by better heat paths (shaft size, housing contact) | steady rise ∝ 1/G, τ_th ∝ 1/G | it is a placeholder: no design should rest on 0.3 W/K |
| Measured mean slip drag torque | enter the bench value | replaces the estimate and the high case: usually the largest change | needs a test |
| High-case multiplier on the estimate | 1 once measured | high-case rise ∝ multiplier | removes the model-error allowance |
| Slip fault trip time (unbroken slip) | shorter | lowers the fault peak | only while the hot-day start is below the limit (slice-1 N1); nuisance trips |
| Hot-day ambient temperature / Coupling rise above ambient while driving (no slip) | lower (if the application allows) | lowers the start 1:1 | operating conditions, not design |
| Relative slip speed, Slip duration per event, Life events | lower | less heat per event and over life | duty, not design |
| Magnet grade (demagnetization note) / Selected adhesive (code) | hotter-rated | raises the governing limit | torque, cost, cure process |
| Mass | more | slows heating only; no change to the steady temperature | weight, inertia |

### 4.5 Draft rewrite
**Title:** Slip and temperature: how hot, and how fast

1. The calculator treats all rotating parts as one block at one temperature that stores heat C (J/K) and loses it through one conductance G (W/K).
2. In unbroken slip it heats from the hot-day start toward a steady rise P/G, reaching 63 % of it after one time constant, τ_th = C/G.
3. For a given loss, G alone sets the steady temperature; more mass only slows the heating, and one short slip adds just P·t/C.
4. The temperature check uses the peak: the hot-day start, plus a slip fault lasting the trip time, plus the average heating from slips over life.
5. Time to the limit is how long unbroken slip takes to reach the lower of the magnet and adhesive limits; "never" means it levels off below.

**Watch out:** Thermal conductance is a placeholder, not a measurement, and the steady rise scales as 1/G: compare the high-case steady temperature with the limit before trusting "never", and measure G. (If live numbers land: "here {temperature.thermal.steady_high_C} against {temperature.summary.governing_limit_C}".)

**Equations:** keep all 9 (`temperature.thermal.heat_capacity_J_K`, `time_constant_s`, `t95_s`, `steady_rise_est_C`, `steady_rise_high_C`, `time_to_limit_high`, `time_to_limit_est`, `temp_at_fault_C`, `critical_drag_Nm`; START_HERE opens `time_constant_s`). Add (unowned): `temperature.thermal.steady_est_C`, `steady_high_C`, `rise_per_event_C`, `rotations_to_limit_high`, `temperature.magnet_life.peak_C`, `temperature.summary.governing_limit_C`, `temperature.slip_life.rise_per_event_high_C`, `temperature.slip_life.slip_duty`, `temperature.duty.hot_day_start_C`.

**Diagram:** HeatingCurve: relabel "τ: 63 %" as "τ_th: 63 %"; add a dashed "limit" line and a "time to the limit" marker and a temperature axis label (TC-24).

## 5. clamp_preload ("Clamp preload and friction", notes.rs:371-402)

### 5.1 Decision it supports
Does the shaft clamp hold more than the coupling can pass (so the coupling, not the clamp, slips), and if no screw size works, what to enlarge.

### 5.2 Problems in the current text
- **Missing WHY (TC-16).** The coupling is the torque limiter; the clamp is sized to the cold-high torque (`clamps.max_torque_Nm = metal.torque_cold_high_Nm`: pull-out at the Minimum magnet temperature with +variation) so the coupling slips first. "cold-high torque" is undefined.
- **s3, PA-10 and a small inaccuracy.** "75 % of the screw's proof load", "safety factor of 1.5": editable defaults (Preload as a share of proof load; Safety factor on thread stripping). For class A4-70 the engine uses the yield stress, not a proof stress (records/clamps.rs comment; materials.rs:105). "proof load" undefined.
- **s4, PA-10.** "a safety factor of 2" is the editable Safety factor, clamp alone (help: "Reversing torque and vibration; the key is extra").
- **s4, incomplete.** The count sets the screws needed, but the size is the smallest of M2.5-M6 that passes geometry (wall, head seat, grip, thread) and fits that many screws along the clamp length (clamps.rs compute); this is what fails as "None: enlarge the boss or the clamp length".
- **Watch-out, PA-10.** μ = 0.15 is the editable Friction coefficient, shaft to bore. The arithmetic is right (holding torque ∝ μ: 0.10/0.15 cuts a third). At the defaults that third matters: one M4 holds 7.665 N·m against a 7.647 N·m requirement (0.2 % over), and at μ = 0.10 no size works.
- **Missing trap (deferred from slice 1).** The clamp is sized on its own "Boss outside diameter" (Shaft clamps, 25 mm), not the "Shaft boss outside diameter" (Metal design, 22 mm) that the Outside diameter, the drawing and the mass use (size_controls.rs:81). At 22 mm with the 10 mm clamp no size works. A note or fix that says "raise Boss outside diameter" must say the drawn boss does not follow.
- **Sources line.** "Addendum A3 assumptions", "M1 audit M14", "M1 audit E2" (CG-21).
- s1-s2 physics is right: friction on both jaws at the shaft radius gives T ≈ μ F d k per screw (2 μ F k d/2), k the share of screw force pressing the shaft (0.8 one-piece, 1.0 two-piece); the model ignores the wrap-around pressure bonus, which is conservative.

### 5.3 Facts a plain rewrite must keep
1. The coupling limits torque by slipping; the clamp must hold more than the coupling's highest expected pull-out.
2. That is the cold-high torque (cold, plus the variation allowance), times the clamp safety factor.
3. Preload = a share of the screw's proof load (yield for A4-70), capped by thread stripping in the aluminium.
4. Per screw: μ × preload × shaft diameter × clamp factor; the smallest size that fits enough screws is picked.
5. Levers when nothing fits: the clamp's boss diameter, the clamp length, a two-piece clamp.
6. Friction assumption, and the clamp boss is not the drawn boss (watch-out).

Numbers: from results: cold-high torque, required torque, capacity, screws, recommended size, tightening torque, safety factor on the coupling torque. From inputs: μ, safety factor, preload share, stripping factor, clamp factor. Fine as literals: "about 0.10 for an oily bore" (a typical value, not a default), "a third" (ratio 0.10/0.15, valid only against the 0.15 default, so phrase "(0.15 by default)").

### 5.4 What to change
| On-screen input | Direction | Effect | Trade-off |
|---|---|---|---|
| Boss outside diameter (Shaft clamps) | raise | larger screws fit (with the default ligament, wall and grip minimums, M4 needs about 24.5 mm; M5 about 27.2 mm and a clamp at least 11 mm long) | it is not the drawn Shaft boss outside diameter; keep the two consistent by hand |
| Clamp length, free end to relief cut | raise | more screws fit along it | axial length |
| Clamp type | two-piece | clamp factor 1.0 vs 0.8: about 25 % more per screw | more parts and machining |
| Screw class | keep 12.9 (strongest); 10.9 and A4-70 hold less | preload ∝ proof (yield) stress | corrosion (A4-70) |
| Thread engagement in aluminium (× screw diameter) / Aluminium (7075-T6 vs 6061-T6) | longer / 7075 | raises the stripping cap (not governing at the defaults) | needs thread length in the far jaw |
| Friction coefficient, shaft to bore | set to the real assembly (degreased 0.15, oily 0.10) | holding torque ∝ μ | not a lever to raise freely |
| Safety factor, clamp alone | lower only with a reason | fewer screws | the factor covers reversing torque and vibration |
| Upstream: Minimum magnet temperature, Symmetric torque variation allowance, magnet length | colder / larger / longer raise the cold-high torque | raises the clamp demand | the clamp follows the coupling, not the drive |

### 5.5 Draft rewrite
**Title:** Shaft clamp: it must out-hold the coupling

1. The coupling limits torque by slipping, so the shaft clamp must hold more than the coupling's highest pull-out, or the clamp slips first.
2. That highest pull-out is the cold-high torque: pull-out at the Minimum magnet temperature plus the variation allowance; the clamp must hold it times a safety factor.
3. Each screw's preload is a set share of its proof load, the most it carries without permanent stretch, or less if the aluminium thread would strip.
4. Friction on both jaws gives each screw about μ × preload × shaft diameter × clamp factor; the smallest screw size that fits enough screws is picked.
5. If none fits, raise the clamp's Boss outside diameter or Clamp length, or choose a two-piece clamp, which presses the shaft harder.

**Watch out:** Friction (0.15 by default) assumes a degreased bore; an oily bore (about 0.10) cuts the holding torque by a third. The clamp's boss is a separate input from the drawn Shaft boss outside diameter.

**Equations:** keep all 6 (`clamps.table[#].preload_strength_N`, `preload_strip_N`, `preload_N`, `torque_per_screw_Nm`, `screws_needed`, `clamps.capacity_Nm`; START_HERE opens `capacity_Nm`). Add (unowned): `clamps.max_torque_Nm`, `clamps.required_Nm`, `clamps.clamp_factor`, `clamps.screw_proof_MPa`, `clamps.sf_coupling`, `clamps.tightening_Nm`, `clamps.table[#].screws_fit`, `clamps.table[#].works`, `clamps.table[#].clamp_torque_Nm`. `metal.torque_cold_high_Nm` only if no new hot-low/cold-high note takes it (that note is its better owner).

## 6. Spec section 8 traps that touch these notes, and traps to add for slice 2

Existing traps, where they bite:
- **Trap 6 (stored demag fields follow geometry or grade).** Demagnetization s3 today ("863 kA/m here"); ferrite watch-out. Extend the trap to the stored slip-loss fields (slip heating has the same defect and no disclosure).
- **Trap 3 (pull-out guaranteed; re-lock as soon as the load falls).** Clamp note: write "highest expected pull-out" with the variation allowance, never "can never exceed". Slip note: the drag sentence must not say or imply re-locking.
- **Trap 5 (416 a safe back-iron fix).** Do not suggest a lower-conductivity back iron to cut slip loss.
- **Traps 1-2 (pole-count levers).** Do not suggest pole count to cut slip loss or raise the demag limit.

Proposed additions for the slice-2 physics review's must-not-write list:
1. Thicker magnets, a smaller gap, fewer poles or a weaker opposite ring raise the calculator's demagnetization limit (the stored fields do not move).
2. A hotter-rated grade costs nothing (lower Br and torque; needs Custom dimensions; until E21 lands its onsets may be raised by a negative calibration shift).
3. The calibration shift is conservative (it raises onsets for N52, N50, N50M, N38UH, N35EH and 2:17 SmCo).
4. More mass or heat capacity lowers the steady temperature (it only slows heating).
5. "Never" means the coupling can slip indefinitely (it rests on a placeholder G and an estimated loss; at the defaults the margin is 0.54 °C).
6. Steel slip loss follows speed^1.5 exactly (thin-skin model form; real exponent 1.54-1.64).
7. A measured drag only refines the estimate (it replaces the estimate and the high case).
8. The clamp is sized to the drive or required torque (it is sized to the cold-high pull-out × safety factor).
9. Raising the clamp's Boss outside diameter enlarges the drawn boss or the Outside diameter.
10. Model ferrite by typing a positive Hcj temperature coefficient (clamped away; use a Y30 grade).
11. The ferrite cold check means anything with the default reverse fields, or that entering ferrite fields will make it pass.
12. The coupling still passes useful torque while slipping (the average is the eddy drag, 0.5 % of pull-out at the defaults).

## 7. Cross-note points for the slice-2 writer

- **Symbols** (all five): match the panel or avoid symbols in prose: σ_st and σ_sl for conductivities (bare σ is shear stress from step 1), μ_r, f_thin, k_hi, τ_th, ϑ_hot, k_knee, Δϑ_m, μ, k_cl.
- **Glossary terms these notes need** (TC-33): reverse field, knee, coercivity (Hcj), onset, rated maximum temperature, slipping/skipping, eddy current, skin depth, conductance, time constant, preload, proof load, cold-high torque.
- **Sources** render on screen as audit ids (CG-21); keep plain sources or collapse them.
- **Gate:** each rewrite sends a START_HERE note to Draft; all five are in START_HERE and two are dashboard Learn targets (guidance.rs:197, 209), so they must be re-reviewed in the same change.
- **New-note links:** the clamp note leans on a "hot-low and cold-high" note; slip heating and demagnetization lean on "what happens after the coupling slips" (the average torque while slipping is the drag; like poles pass once per pole pass; re-locking needs the slip speed brought near zero, CG-13 verifier).

## 8. Engine observations outside the notes (for the owner; not to be fixed in note text)

- **E21 not landed.** Negative calibration offsets still raise onsets: N38UH +7.3 °C, N35EH +3.6 °C, N52 +9.7 °C (hand, matches PA-16's N52 probe).
- **Clamp boss vs drawn boss** (deferred from slice 1): two inputs, 25 mm and 22 mm; at 22 mm no size works with the 10 mm clamp.
- **Hot-onset degenerate case** (low severity, arithmetic-verified, reachable only through extreme slider settings): when the stored field already exceeds the knee at 20 °C and H|α| > H_k|β| (e.g. Coercivity source = inputs, β = -0.001/°C, knee fraction 0.5), `demag_onset_C` returns an onset above 20 °C that reads as "safe below it", the wrong direction. With a rating the calibration offset happens to drive the limit very negative; a magnet with no rating ("n/a", offset 0) would pass.
- **Both rings share the outer blocks' stored fields** (the inner ring's lower 341/760/844/545 kA/m from the help text are unused): conservative for the inner ring, worth one line in the reverse-field help.
- **Y30 coefficient range** is data the engine never reads (PA-17); nothing flags out-of-range onsets for any grade.
