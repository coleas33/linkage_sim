# Slice 2: the new teaching notes (physics scoping)

Read-only pass. Repo paths are relative to `magcoupling-rs/` unless they start with `docs/`.
Default-design numbers come from the verified probe output in the session scratchpad
(`probe/out.txt`, `compute_all` at defaults and at 8 poles), from the audit's verified checks
(`audit.json`), or are derived here by hand from the engine's own formulas (marked "derived").
Thermal numbers come from README E5 and the M1 audit; they were not re-probed.

## 0. Recommendation in one screen

- **The cap is verified:** `notes.rs:630-633` asserts `(15..=20).contains(&NOTES.len())`.
  There are 17 notes now, so there are exactly 3 free slots. A 4th note fails that test.
  Raising the cap is a one-line change, plus the doc strings listed in section 6.
- **Value ranking for the owner's audience:**
  1. (a) How the coupling makes torque.
  2. (b) Hot-low and cold-high.
  3. (d) When the coupling slips.
  4. (c) The back-iron wall.
- **Add three new notes: (a), (b) and (d).**
- **Deliver (c) by rewriting `a5.low_saturation`.** Slice 2 rewrites that note anyway, and the
  cup-wall guide already links to it. Retitle it "Back-iron wall and saturation" and give it
  the wall-rule records. A draft is in 3.5.
- **Why (d) gets a slot and (c) does not:**
  - (c) has a natural home. The low-saturation caution's whole consequence is "the wall must
    be thicker", and the wiring needs no change.
  - (d) has no home. Folding it into the rewrites splits it across `pullout_angle`,
    `slip_heating` and `clamp_preload`. That leaves the audit's CG-13 complaint unfixed: slip
    "is never explained in one place". The coupling exists to slip (20 million slip events in
    the life input), so this is a core topic.
- **Alternative R2:** add (a), (b) and (c) as notes, and spread (d) across those three rewrites
  (section 5).
- **Alternative R3:** raise the cap to 21 and add all four.

## 1. Constraints any new note must meet (verified in code)

| Constraint | Where | Consequence |
|---|---|---|
| 15 to 20 notes | `notes.rs:630-633` | 3 slots. |
| 2 to 6 `sentences`, non-empty `sources` | `notes.rs:635-644` | No word limit is tested. The drafts below keep each sentence at 25 words or fewer. |
| Each equation has at most one note; every entry must name a default-design result that has a record | `tests/explain.rs:1173-1218` | (c) must *move* `model.backiron_needed_mm` and `model.bsat_T` out of `a5.low_saturation`, or share nothing with it. |
| A dashboard row with no check, covered by a reviewed note, must link that note | `gui/guidance.rs:2381-2400` | (b) claiming `metal.torque_cold_high_Nm` forces its `ROW_LEARN` entry to `Learn::Note(b)`. (d) claiming `model.gearbox_input_ripple_Nm` forces that row's entry to `Learn::Note(d)` (today it is `pullout_angle`). `mass.total_g` must stay unowned (`panel.rs:7006-7010`). |
| The step-1 title is drawn exactly once while Start here step 1 is open | `panel.rs:7027` | If (a) becomes `START_HERE[0]`, no dashboard row may use (a) as its Why link. So (a) must not claim `model.cup_od_mm`. The pinned title at `explorer.rs:1084` ("Harmonic decomposition") must change. |
| A note title must not equal other drawn dashboard text | `dashboard.rs:1719-1735` | Titles must not be "Hot-low torque", "Cup wall" or any other guide label. The four titles below are clear of every label. |
| `sentences[0]` of a linked note is counted raw | `panel.rs:6998, 7265, 7505` | No ϑ, superscript minus or ∝ in `sentences[0]`. All drafts comply. |
| Glyphs | `test_support.rs:183-192` | Drafts use only σ π τ ² · and ASCII, all already used by existing notes. |
| Every linked note must be `Reviewed` in the merge commit | about 10 GUI tests; `tests/explain.rs:1033` | Draft, then a fresh-context session-model physics review, then `Reviewed`, all before the gate. |

## 2. Physics traps (slice-1 spec section 8, plus those found here)

The spec's six (`docs/superpowers/specs/2026-10-13-magcoupler-guidance-design.md:197-204`)
apply to every draft:
- that ring radius is the strongest lever;
- that fewer poles keep the same size;
- that pull-out is guaranteed, or that the rings re-lock as soon as the load is below pull-out;
- that the bench calibration applies beyond the prototype;
- that 416 is a safe back iron;
- that the stored 3D fields follow the design.

New traps found in this pass:

- **T-A. Fill is not w/τ_p.**
  - τ_p is taken at the gap mid radius R_g (`model.rs`: `tau_p = 2πR_g/N`).
  - The fill is taken at each ring's magnet mid-radius, clamped to 1 (`pitch_share`,
    `model.rs:507-509`). Inner and outer fills differ: 0.861 and 0.620 at the defaults.
  - Write "each block's share of its pole", never "λ = w/τ_p".
- **T-B. End effects grow with *fewer* poles, not more.**
  - f_end = 1 − c_end·τ_p/L. A shorter pitch loses less at the ends (0.896 at 10 poles, 0.870
    at 8).
  - "More poles help until end effects bite" is backwards. With more poles it is the gap,
    relative to the pitch, that bites.
- **T-C. T ∝ R_g²·L holds only at fixed σ.**
  - At a fixed count and fixed blocks, a larger ring lowers the fill and raises the end loss.
  - On the default design the torque rises to about a 14 mm ring radius, falls to about 20 mm,
    then rises again (`engine/sizing.rs:10-11`). The CG-4 probe gives 3.048, 2.539 and
    2.949 N·m at 14, 20 and 30 mm.
- **T-D. Pole-pitch optimum.**
  - At fixed fill the steel-circuit S₁ = sinh(kt)²/sinh(k(2t+g)), with k = π/τ_p.
  - Derived: S₁ peaks near 10 to 12 poles at the default ring, gap and thickness, where
    k(2t+g) ≈ 2.7 to 3.3.
  - With no gap, S₁ rises with k all the way, so a fixed "best pitch ≈ gap" rule is not
    general.
  - Note text should state the trade-off only: a short pitch fades across the gap faster.
  - At the defaults, 10 → 8 lowers S₁_iron only slightly (0.2445 to 0.2373). The fill does the
    damage: B₁ product −23 %, S −3 %, f_end −3 %, total −27 % (derived from the probe values).
- **T-E. "Hot" in hot-low means the operating-magnet-temperature input** (`coupling.op_temp_C`,
  50 °C). It is not the hot-day start (55 + 10 = 65 °C) and not the governing limit
  (93.06 °C).
  - The hot-day torque, 2.589 N·m with no allowance, reads "Meets it nominally". At the same
    time hot-low reads "Below hot minimum" (2.285 against 2.5) (PA-4).
  - The required-minimum help, "At the 50 °C magnet temperature.", is workbook-locked and goes
    stale when the input changes.
- **T-F. `metal.hot_margin` is a nominal margin.**
  - It is T_op/required − 1 = +7.5 % at the defaults, with no allowance, while hot-low fails
    by 8.6 % (`metal_design.rs`).
  - Never describe it as the hot-low margin.
  - `model.verdict` reuses the phrase "Below hot minimum" for a different comparison: the
    nominal pull-out against the required floor (CG-5).
- **T-G. Nothing checks cold-high against the gearbox rating.**
  - Derived: cold-high reflected to the gearbox input (`metal.cold_high_input_Nm`) is
    3.823/(5 × 0.95) = 0.805 N·m. The input rating is 0.6 N·m ("GAM 5:1 peak input rating").
  - Even with no allowance, a coupling that just meets the hot minimum gives 0.651 N·m cold at
    the input (`metal.cold_for_hot_min_input_Nm`).
  - The rating feeds only a legacy reference (CG-18).
  - No note may say the coupling protects the gearbox at its rating. Section 7 has the owner
    question.
- **T-H. The gearbox ripple row is the nominal swing at the operating temperature**
  (`T_pull/(ratio·eff)`, 0.566 N·m at the input). When cold, the swing reaches the cold
  pull-out, and up to cold-high with the allowance (0.805 N·m).
- **T-I. The wall rule's error depends on the pole count.**
  - t = B_gap·τ_p/(π·B_des) scales with τ_p. M1 says the real flux "follows block width", so
    "the error grows with pole count".
  - At the defaults the rule gives 1.92 mm, against about 2.17 to 2.27 mm from exact
    solutions.
  - Derived at 8 poles: the rule gives 2.40 mm. The flat-top form gives 2.17 mm (it does not
    change with N). The fundamental-peak form gives 2.40 mm.
  - So "fewer, wider poles carry more flux, so need a thicker wall" (CG-11's text) is a claim
    about the *rule*, not established physics with the same blocks. Write "in the calculator's
    rule".
- **T-J. The wall rule uses Br at the operating temperature** (B_gap from `br_*_T_op`). Derived:
  at −40 °C the magnets drive about 11 % more flux (1.072/0.964). The rule does not look at the
  cold case. This is a reviewer note; keep it out of the note text.
- **T-K. Saturation is not the design flux density.**
  - B_des is a design limit: 1.5 T is the input default for 4140, and the library gives 1018
    1.7 T.
  - 4140's saturation is `NOT_SOURCED` (`material_library.rs:117`). Never print a 4140
    saturation figure.
  - 416 is the only sourced low-saturation steel (1.6 T) and has no design value of its own,
    so its wall check uses the 1.5 T input.
- **T-L. The torque model assumes ideal steel.**
  - A thin or weak wall changes only the check text, never the torque (`model.rs:1000-1012`;
    PA-9).
  - Do not say "the shown torque falls". Do say the real coupling can fall below the
    prediction.
- **T-M. Slip mechanics.**
  - One torque cycle is one pole pair, not "a pole at a time".
  - Over a cycle at steady slip the magnet torque averages to zero, leaving only the eddy drag:
    0.0131 N·m estimate, 0.039 N·m high case, about 0.5 % and 1.5 % of the pull-out (README E5:
    2.751 W at 209.4 rad/s).
  - Re-locking needs the slip speed brought near zero. A load between the drag and the
    pull-out cannot re-lock the rings (CG-13 check).
  - The engine does not model re-locking.
- **T-N. Stored fields.**
  - The reverse fields (354/791/863/569 kA/m) and the slip fields (`b_hub_T`, ...) are stored
    3D values from the default design (`temperature.rs` inputs; PA-1).
  - The skipping (like poles facing) case is the largest reverse field and sets the magnet
    limit: onset − 10 °C = 93.06 °C.
  - Its 863 kA/m does not follow geometry or grade.
- **T-O. Cold-high is an expectation, not a bound.** Model error is non-conservative for the
  cold-high loads: E7 off-default, and no credit for overhang (M10). Write "expected to pass",
  not "can never exceed".
- **T-P. Outside diameter is max(cup, cap, boss)** (`engine/size_controls.rs:81`). Fewer
  magnets grow the *cup*; at the defaults the cap still sets the Outside diameter. Never write
  "the outside diameter grows" or "stays the same" in general.

## 3. The candidates

### 3.1 (a) How the coupling makes torque — RECOMMENDED (new; `START_HERE[0]`)

**Verified against the engine** (`model.rs` compute; `records/torque.rs:66-104`):

| Quantity | Formula | Default value |
|---|---|---|
| Area × lever arm | A_L = 2π·R_g²·L | — |
| 2D torque | T_2D = σ·A_L | 3.159 N·m |
| Shear stress | σ | 201.4 kPa ("PM-PM couplings typically 100–250 kPa" is the workbook help on `model.tau_Pa`) |
| Pull-out | T_pull = T_2D·f_end·f_cal | 2.688 N·m |
| Gap mid radius | R_g = r_face,i + g/2 | 14.02 mm |
| Pole pitch | τ_p = 2πR_g/N | 8.81 mm |
| Fills | λ_i, λ_o | 0.861 and 0.620 |
| Wave number | k_n = nN/(2R_g) = nπ/τ_p | k₁ = 356.6 /m (TC-11 check) |

- Outside diameter moves `coupling.inner_back_apothem_mm` (`engine/size_controls.rs:3`).
  Magnets per ring moves only N.
- **10 → 8 at the default ring** (probe):
  - pull-out 2.688 → 1.968 N·m (−27 %);
  - fills 0.861/0.620 → 0.689/0.496;
  - τ_p 8.81 → 11.01 mm;
  - f_end 0.896 → 0.870;
  - cup OD 41.33 → 42.44 mm, because r_pocket = (A_back + bond)/cos(π/N) (`model.rs:919-924`;
    derived 17.94/0.9511 → 17.94/0.9239);
  - Outside diameter stays at 42.8 mm, set by the cap.
- **Per-mm levers at the defaults** (CG-3 check probe):
  - +1 mm magnet length gives +0.236 N·m, which matches T ∝ L − c_end·τ_p (derived +8.8 %);
  - +1 mm ring radius gives +0.18 N·m;
  - +1 mm Outside diameter gives +0.094 N·m.

**Link to:**
- `model.torque_2d_Nm`, `model.area_lever_m3`, `model.gap_radius_mm`, `model.pole_pitch_mm`,
  `model.fill_inner`, `model.fill_outer`; optionally `model.active_length_mm`. All are free,
  all have records, all are default results.
- `START_HERE[0] = ("torque_from_shear", "model.torque_2d_Nm")`. Change the pinned title
  (`explorer.rs:1084`) and the tooltip group words (`explorer.rs:486`).
- No dashboard Why link (`panel.rs:7027`).
- A Size-tab link (for example under "Ring size stays the same...") is GUI wiring for later.
  Its link text must not be the bare title, or the step-1 count becomes 2.

**Traps:** spec 1 and 2, T-A, T-B, T-C, T-D, T-P. Do not use "λ" undefined. Do not mention
the bench calibration.

**Draft** (id `torque_from_shear`):
- Title: **How the coupling makes torque**
- Sentences (word counts):
  1. Twist one ring against the other and the magnets pull each other sideways: a shear stress σ spread over a cylinder in the gap. (24)
  2. Torque is σ times that cylinder's area and radius: T_2D = σ·2π·R_g²·L, with R_g the gap's mid radius and L the magnet overlap. (23)
  3. σ comes from magnet grade and thickness, each block's share of its pole, and the gap against the pole pitch τ_p = 2π·R_g/N. (23)
  4. The ring radius is its own input, moved by Outside diameter; the magnet count N only sets how many blocks share that ring. (23)
  5. So on the same ring, fewer magnets leave wider gaps between blocks: torque usually falls, and a polygon cup grows as its corners move out. (24)
  6. To add torque, lengthen the magnets first: on the example design a millimetre of length adds more than a millimetre of ring radius. (23)
- Watch out: T_2D grows as R_g² only while σ holds; at a fixed magnet count a larger ring spreads the blocks, so add magnets as it grows.
- Diagram: none for now. A "shear on the gap cylinder" kind is a later addition in three
  places: the enum, `diagram_shapes` and `KINDS`.
- Sources (plain wording, pending the slice-2 jargon decision):
  - the torque model's own equations (gap radius, pole pitch, fill, shear stress, 2D torque);
  - math audit 2026-09-29, rows M3, M4 and M8 (the planar model against exact 2D and 3D);
  - guidance audit 2026-10-13, verified probes (10 to 8 magnets; length against ring radius).
- Reviewer flags:
  - Is S1's qualitative shear picture acceptable?
  - S3 lumps grade (Br) with thickness, and leaves the temperature dependence to
    `br_temperature`.
  - S5 says "usually", because fewer poles can win at large gaps (T-D).
  - "Outside diameter" must use whatever name slice 2 (C) settles for this radius (it has four
    names today).

### 3.2 (b) Hot-low and cold-high: the torque band — RECOMMENDED (new)

**Verified** (`metal_design.rs:405-500`; `records/torque.rs:195-228`; `assumptions.rs:73`).
What each torque uses:

| Quantity (path) | Temperature | Allowance | Compared with / feeds | Default |
|---|---|---|---|---|
| Hot-low `metal.torque_hot_low_Nm` | operating magnet temperature input (50 °C) | × (1 − v) | `metal.required_min_Nm` → `metal.hot_min_check` (dashboard, red); the sizing solver's target (slice 1 G-2) | 2.285 against 2.5: short 0.215 N·m (−8.6 %) |
| Nominal hot margin `metal.hot_margin` | operating | none | T_op/req − 1 | +7.5 % (T-F) |
| `model.verdict` | operating | none | required floor = max(drive × SF = 0.91, 2.5) | "Nominal only: hot test" (T-F) |
| Hot-day torque `temperature.magnet_life.torque_hot_day_Nm` | hot-day start, 55 + 10 = 65 °C | none | required min → `torque_hot_day_check` | 2.589, "Meets it nominally (no variation allowance)" |
| Cold-high `metal.torque_cold_high_Nm` | minimum magnet temperature input (−40 °C) | × (1 + v) | `clamps.max_torque_Nm`; × clamp SF 2 = `clamps.required_Nm` | 3.823; clamp needs 7.65 N·m |
| Cold-high at the gearbox input `metal.cold_high_input_Nm` | −40 °C | × (1 + v) | shown, not checked | 0.805 against a 0.6 rating (T-G) |
| Adhesive peak load `temperature.adhesive_life.torque_peak_var_Nm` | peak temperature | × (1 + v) | adhesive life | — |
| Torque vs temperature tab | curve | band edges | its low and high edges equal hot-low and cold-high bit for bit (`plots.rs` test) | — |

- Band ratio, derived: cold-high/hot-low = (1.072/0.964)² × 1.15/0.85 = 1.673 at the
  defaults.
- To pass the hot minimum the nominal pull-out at 50 °C must reach 2.5/0.85 = 2.941 N·m
  (+9.4 %; `metal.required_20C_Nm` = 3.165 N·m at 20 °C).
- v defaults to 0.15: "Engineering allowance, not measured" (`metal_design.rs:31-34`). It is
  symmetric (assumption `production_variation`).

**Link to:**
- Equations: `metal.torque_hot_low_Nm`, `metal.torque_cold_high_Nm`, `metal.hot_min_check`,
  `metal.hot_margin`, `metal.required_20C_Nm`, `metal.required_20C_zero_scatter_Nm`,
  `metal.torque_cold_zero_var_Nm`, `metal.cold_for_hot_min_Nm`,
  `metal.cold_for_hot_min_input_Nm`, `metal.cold_high_Nm`, `metal.cold_high_input_Nm`,
  `metal.torque_op_Nm`, `metal.torque_20C_Nm`, `clamps.max_torque_Nm`. All are free.
  `metal.torque_cold_Nm` stays with `br_temperature`.
- Optional: `temperature.magnet_life.torque_hot_day_check` and
  `temperature.summary.torque_hot_day_note`, so Explain on the hot-day check explains the two
  "hot" torques.
- Wiring:
  - `HOT_MIN_GUIDE.learn` changes from `Learn::Assumption("production_variation")` to
    `Learn::Note("hot_low_cold_high")`, and the hot-low and hot-min rows follow it.
  - `ROW_LEARN` for `metal.torque_cold_high_Nm` must become `Learn::Note(..)` (the test
    forces it).
  - Optional `START_HERE` step `("hot_low_cold_high", "metal.torque_hot_low_Nm")`, right after
    `br_temperature`.

**Traps:** T-E, T-F, T-G, T-O, T-H. Spec 3: pull-out is not guaranteed; the engine help on
`model.pullout_Nm` says "Analytical estimate, not a guaranteed minimum."

**Draft** (id `hot_low_cold_high`):
- Title: **Hot-low and cold-high: the torque band**
- Sentences:
  1. Pull-out is a nominal estimate at one temperature; magnets weaken as they warm, and production scatter moves real couplings either way. (21)
  2. Hot-low is the pull-out at the operating magnet temperature less the variation allowance, 15 % by default: the torque to count on. (22)
  3. Cold-high is the pull-out at the minimum magnet temperature plus that allowance: the most the coupling is expected to pass before slipping. (22)
  4. The Torque vs temperature tab draws this band; hot-low must reach the required minimum, and the shortfall shows how far it must rise. (23)
  5. The shaft clamp is sized to cold-high times a safety factor, and the gearbox sees it too, so added torque loads both. (22)
  6. To pass hot-low, add torque, longer magnets first; lower the operating temperature or the allowance only on measured evidence. (19)
- Watch out: Hot-low uses the operating temperature you enter, not the hot-day start; the temperature results check the hot-day torque separately, without the allowance.
- Diagram: none. S4 points to the live plot (TC-25).
- Sources:
  - the production variation assumption (rationale and source);
  - the Metal design hot and cold torque rows (hot-low, cold-high, the hot-minimum check);
  - shaft-clamp sizing (cold-high times the clamp safety factor);
  - guidance audit 2026-10-13, findings PA-4 and CG-5.
- Reviewer flags:
  - Is "the torque to count on" too strong for an unmeasured allowance?
  - "longer magnets first" matches the slice-1 hot-min fix. It also lengthens the stack and
    raises cold-high, which S5 says.
  - "15 % by default" is the only literal.

### 3.3 (d) When the coupling slips — RECOMMENDED (new)

**Verified:**
- Torque-angle curve: Σ A_n sin(nφ), φ = (N/2)θ (`records/torque.rs`; `plots.rs`).
- The peak is the pull-out (E7 search). One cycle is one pole pair.
- Ripple: amplitude `model.gearbox_input_ripple_Nm` = T_pull/(ratio·eff) = 0.566 N·m. Frequency
  N/2 · slip rpm/60 = 166.7 Hz (`model.ripple_freq_Hz`, no record; `metal.slip_freq_Hz` and
  `temperature.duty.field_freq_Hz` have records).
- Mean drag while slipping: 0.0131 N·m estimate and 0.039 N·m high case
  (`temperature.slip_loss.drag_Nm` = P/ω_s). Critical drag 0.0402 N·m. High-case steady
  temperature 92.51 °C against a 93.06 °C limit, so "never" (README E5; M1 audit).
- `temperature.thermal.temp_at_fault_C` uses the slip fault trip time (2 s).
- Like poles facing is the largest stored reverse field (863 kA/m; help: "once per pole pass
  while skipping"). The magnet limit is the skipping onset − 10 °C, unless the adhesive
  governs (`temperature.summary.governing_note`).
- The engine has no re-lock model. The validation items list "re-engagement" and "detect
  sustained clutch slip" as open (`metal_design.rs` `VALIDATION_ITEMS`).

**Link to:**
- Equations: `model.gearbox_input_ripple_Nm` (free; a dashboard row with no check, so
  `ROW_LEARN` must change from `pullout_angle` to this note), `metal.slip_freq_Hz`,
  `temperature.duty.field_freq_Hz`, `temperature.slip_life.rise_per_event_high_C`,
  `temperature.magnet_life.peak_C`, `temperature.magnet_life.margin_onset_C`. All are free.
- Leave the summary thermal copies (`temperature.summary.time_to_limit_high`,
  `.critical_drag_Nm`, `temperature.thermal.steady_*_C`) for the `thermal_time_constant`
  rewrite, so one idea has one note.
- Leave the demagnetization onsets with `demagnetization`.
- Optional `START_HERE` step before `slip_heating`, opening `model.gearbox_input_ripple_Nm`.
- `pullout_angle` keeps the `model.pullout_Nm` row, so `panel.rs:6972-7015` still finds its
  title on the dashboard.

**Traps:**
- T-M, T-N, T-H.
- Spec 3: neither "guaranteed" nor "re-locks below pull-out".
- Do not quote the drag as measured; it is an estimate with a factor-3 high case.

**Draft** (id `when_slipping`):
- Title: **When the coupling slips**
- Sentences:
  1. Past the pull-out torque the rings slip, and the torque swings between plus and minus the pull-out once per pole pair passed. (22)
  2. Averaged over a swing that torque is near zero, so while slipping the load gets only a small eddy-current drag. (20)
  3. The rings lock again only when the slip speed falls near zero: the drive must slow, or the load must drop below that drag. (24)
  4. Meanwhile eddy currents heat the steel, sleeve and magnets, and each swing reaches the gearbox as ripple at the pole-pair passing frequency. (22)
  5. Each swing also puts like poles face to face, the strongest reverse field the magnets meet, so this case sets their temperature limit. (23)
  6. Detect slip and slow the drive within the slip fault trip time; the calculator bounds the heating with that time but does not model re-locking. (25)
- Watch out: The slip and reverse fields are stored from the example design's 3D model; they do not follow your geometry or magnet grade.
- Diagram: reuse `TorqueAngle` after TC-22 fixes its main picture, or none.
- Sources:
  - the torque-angle model (pull-out as the peak of the summed harmonics);
  - the slip-loss and thermal model (drag, heating, slip fault trip time);
  - the demagnetization model (the like-poles-facing field while skipping);
  - guidance audit 2026-10-13, CG-13 and its re-lock correction.
- Reviewer flags:
  - S2 ignores hysteresis loss, which is not modelled.
  - S5: "sets their temperature limit" is the magnets' limit; the adhesive can govern instead.
  - S4: the ripple row shows the swing at the operating temperature; when cold it is larger
    (T-H). Consider saying so in the `pullout_angle` or (b) rewrite.

### 3.4 (c) How thick the cup wall must be — standalone draft (used only under R2 or R3)

**Verified:**
- `model.backiron_needed_mm` = B_gap·τ_p/(π·B_des) (`records/dashboard.rs:52-60`).
- B_gap = (Br_i·t_i + Br_o·t_o)/(t_i + t_o + g) at the operating temperature (E10).
  Derived 1.0265 T, giving 1.919 mm.
- The check advises "at least 2.0 mm"; the input default is 1.8 mm, and the check is red.
- `model.cup_od_mm` = 2(r_pocket + wall input), so the cup grows by twice the wall change. The
  `materials.cup_wall_check` help says so.
- Derived: the cup reaches the 42.8 mm cap at a wall of about 2.53 mm, so a 2.0 mm wall leaves
  the Outside diameter unchanged.
- 8 poles: 2.40 mm.
- M1: the rule reads 12 to 16 % thin at the defaults (exact 2.17 to 2.27 mm), and the error
  grows with pole count (`docs/analyses/2026-09-29-magcoupling-math-audit.md:57`).
- B_des: the 1.5 T input, or a library steel's own value (1018 1.7 T; `records/dashboard.rs:44`).

**Link to:**
- Equations: `materials.cup_wall_check`, `materials.cup_wall_suggested_mm`,
  `model.gap_flux_density_T` (all free); `model.backiron_needed_mm` and `model.bsat_T` (moved
  from `a5.low_saturation`, both notes in one commit). Optionally `model.cup_od_mm`, which
  forces its `ROW_LEARN` entry.
- `CUP_WALL_GUIDE.learn` changes to this note. `a5.low_saturation` stays the warning note,
  with no equations (legal).

**Traps:** T-I, T-J, T-K, T-L, and spec 5 (416).

**Draft** (id `cup_wall`):
- Title: **How thick the cup wall must be**
- Sentences:
  1. The steel cup behind the outer magnets carries each pole's flux around to its neighbours; too thin a wall saturates and leaks flux. (23)
  2. The calculator sizes the wall as t = B_gap·τ_p / (π·B_des): half a pole's flux, carried at a design flux density below saturation. (23)
  3. Stronger or thicker magnets, a smaller gap, a larger ring or fewer magnets raise the need, because B_gap or the pole pitch τ_p grows. (24)
  4. To fix a thin wall, raise Minimum outer return ring wall to the suggested value; the cup diameter grows by twice the change. (23)
  5. The torque shown assumes the steel carries all the flux, so a thin wall can cost real torque that the prediction does not show. (24)
  6. B_des is an input, 1.5 T by default for annealed 4140; a library steel with its own design value, such as 1018, replaces it. (24)
- Watch out: On the example design the rule reads 12 to 16 % thin against detailed field solutions, and its error changes with pole count: leave margin.
- Diagram: reuse `FluxPathBackIron`.
- Sources:
  - math audit 2026-09-29, rows M1 and M2;
  - the back-iron design flux density assumption and decision 20;
  - the wall check and its suggestion (decision 27).

### 3.5 (c) folded into `a5.low_saturation` — the recommended delivery of (c) under R1

- Keep the id, which the warning rule forces (`warnings.rs:215`).
- Retitle it **Back-iron wall and saturation**.
- Keep `model.bsat_T` and `model.backiron_needed_mm`. Add `materials.cup_wall_check`,
  `materials.cup_wall_suggested_mm` and `model.gap_flux_density_T`.
- No guide or `ROW_LEARN` change: `CUP_WALL_GUIDE.learn` already points here.
- Sentences 1 to 4: as in 3.4.
- Sentence 5: The low-saturation caution flags a steel that saturates below 1.7 T or has no design value; its wall check uses the design flux density input. (25)
- Sentence 6: For such a steel, set that input safely below its saturation, so the wall rule asks for enough steel. (19)
- Watch out: The torque shown assumes ideal steel, and the rule reads 12 to 16 % thin on the example design: leave margin.
- This text fixes TC-17 (the saturation/design-value conflation and the wrong title on the
  wall rule), PA-5 (the note side) and PA-9.
- Reviewer flag on S6: it is engineering advice. 416's 1.6 T saturation against the 1.5 T
  default leaves about 6 % margin, and its wall check passes on that. S6 must not read as
  "416 is fine".

## 4. Ranking and reasons

1. **(a) Torque from shear stress, ring radius against magnet count.**
   - It is the owner's own 10 → 8 confusion (TC-1, CG-1, CG-3, CG-4, VZ-10, PA-7).
   - It is the natural Start here step 1, replacing the Fourier opener (TC-3, TC-4).
   - It owns six unexplained torque-chain records.
   - It says what to change: length first, with magnets added when the ring grows.
2. **(b) Hot-low and cold-high.**
   - The example design's first red check has no note today, only an assumption rationale.
   - It answers "which torque do I design to" (CG-5, PA-4).
   - It ties the clamp and the gearbox to cold-high (TC-16, CG-18).
   - Three dashboard rows gain a real Why link.
3. **(d) When the coupling slips.**
   - The coupling's job is to slip, and slip is the design case for heat, demagnetization,
     ripple and clamp loads.
   - Two of the six spec traps are slip misconceptions.
   - Its pieces sit in five notes in specialist terms, and no rewrite can gather them.
   - It needs care because re-locking is not modelled; the draft says so.
4. **(c) The back-iron wall.**
   - It is a real gap: the default cup wall is red, and its Why link says "Low saturation".
   - The slice-1 guide already gives Problem, Why and Fix for it.
   - It fits the low-saturation note the slice must rewrite anyway, with no wiring change.

## 5. Options for the three slots (for the owner's AskUserQuestion)

| Option | New notes | (c) / (d) handling | Effort | Risk | Maintenance |
|---|---|---|---|---|---|
| **R1 (recommended)** | (a) (b) (d) | (c) as the rewrite of `a5.low_saturation` (3.5) | 3 notes, 4 wiring edits (guide learn, two `ROW_LEARN` rows, `START_HERE`), 1 rewrite already planned | Medium: 4 notes into physics review; (d) states unmodelled behaviour qualitatively | Each idea in one place; one note serves the caution and the wall check |
| R2 | (a) (b) (c) | (d) spread: a `pullout_angle` watch-out (TC-12 text with the CG-13 re-lock correction), one `slip_heating` sentence (mean drag), one `clamp_preload` sentence (TC-16) | About the same | Medium; `pullout_angle` is Start here step 2 and grows | Slip stays scattered (CG-13 unfixed) |
| R3 | all four | none | R1 + R2 notes; cap 15..=20 → 15..=21, plus the doc strings (`notes.rs:59-61`, README.md:29/662, `docs/ai/03-structure.yaml:61`) | Low extra; a spec A4 departure ("about 15 to 20") | Cleanest content; one more note to review |

## 6. Wiring and docs checklist for whichever option

- `notes.rs`: append the notes as `Draft`, then make them `Reviewed` after the fresh-context
  session-model review, in the same commit as any link to them.
- `START_HERE`: (a) first. Update `explorer.rs:1084` and the tooltip at `explorer.rs:486`.
- `guidance.rs`:
  - `HOT_MIN_GUIDE.learn` → (b);
  - `ROW_LEARN`: `metal.torque_cold_high_Nm` → (b), `model.gearbox_input_ripple_Nm` → (d);
  - (R2/R3) `CUP_WALL_GUIDE.learn` → (c).
  - Remove the matching `Learn::None` reasons only where a note now covers the row (none of
    the three covers clearance, space claim or block fit).
- Docs with the code: README.md (the note count "17 of the 17"), `docs/ai/02-system.yaml`,
  `03-structure.yaml:61`, `04-memory.yaml`, `05-update-tracker.md`, `backlog.yaml`.
- Tests to add: one per new link, in `guidance.rs` and `dashboard.rs` (the existing tests
  already enforce the coverage), and a plain-language test over note text that reuses
  `plain_text_violations` (TC-34).

## 7. Owner-level findings from this pass (outside the notes)

1. **Cold-high exceeds the gearbox input rating, and nothing checks it** (T-G).
   - At the defaults, cold-high at the gearbox input is 0.805 N·m against the 0.6 N·m rating.
   - A coupling that just meets the 2.5 N·m hot minimum already reaches 0.651 N·m cold with
     no allowance.
   - The workbook's validation item "Verify minimum 2.5 N·m and gearbox-safe torque at service
     temperature limits" names exactly this conflict.
   - Question: should a gearbox check be added (CG-18)? Or is the 0.6 N·m rating not the right
     limit?
2. **The operating temperature (50 °C) is below the hot-day start (65 °C) on the example
   design**, although Metal design C15 labels it "Operating / maximum magnet temperature".
   - So hot-low is not the hottest service case.
   - Question: is the 2.5 N·m requirement defined at 50 °C on purpose?
3. **"Nominal hot margin" (+7.5 %) sits beside a failing hot-low (−8.6 %)**, and
   `model.verdict` reuses "Below hot minimum" for a nominal comparison (T-F). These are worth
   a GUI label fix in slice 2 (C).
4. **The wall rule uses operating-temperature Br**, so at −40 °C the wall carries about 11 %
   more flux than it is sized for (T-J). This is a physics-review item, not note text.
