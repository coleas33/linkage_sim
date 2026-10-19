# BL-064: a check of the keyed shrink fits that hold the rotors on their shafts — design

- Date: 2026-10-17. Branch `magcoupling/bl064-shaft-joint` from `main` at fb79095 (BL-059 merged).
- Backlog: BL-064 (`risk: physics`, priority 2). Research (gitignored): `.superpowers/sdd/2026-10-17-bl064-shaft-joint/`:
  `owner-decisions.md` (binding), `synthesis.md`, `research-architecture.md`, `research-physics.md`,
  `research-surfaces.md`, `research-assembly.md`.
- `M` = `magcoupling-rs`. Torques are in N·m at the gearbox output (both joints are on the output side). Interference is
  diametral, in µm. "Measured" interference is shaft diameter less bore, both measured; "effective" is what grips after
  smoothing and temperature.
- Numbers marked **(E)** come from `compute_all` at fb79095; numbers marked **(C)** come from this spec's model of the check,
  implemented in a scratch crate outside the repo: `<scratchpad>/bl064/spec/` (`src/main.rs`, output `out.txt`, its own
  `CARGO_TARGET_DIR`, `magcoupling-rs` by path with feature `gui`). `<scratchpad>` is
  `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/c368a0e9-0dad-4f12-ad62-643ce3a7f894/scratchpad`.
  Every number in this spec was re-run there at the owner's defaults; none is copied from the research.
- Revised after spec review 1 (`.superpowers/sdd/2026-10-17-bl064-shaft-joint/spec-review-1.md`, must-fix MF-1 to MF-6
  and the should-fix items accepted there under "Writer's response"); the numbers the revision added or changed are in
  section 9 of the scratch crate's `out.txt`.
- Revised again after spec review 2 (`spec-review-2.md` in the same folder, must-fix MF-1 to MF-3 and the should-fix
  items accepted there under "Writer's response"); the numbers that revision added or changed are in section 10 of
  `out.txt` (function `review2()`).
- Revised again after spec review 3 (`spec-review-3.md`, its should-fix items) and the owner's decision of R-15 (the
  last row of `owner-decisions.md`, section 2 below): the bonded inner hub's allowed fitting temperature is also capped
  at the magnets' single-ring onset less 10 °C. The scratch crate's `check` reads the onset (field `r15` of its
  `Settings`; `false` gives the round-3 rule); the numbers that revision added or changed are in section 11 of
  `out.txt` (functions `r15()` and `r15b()`). Sections 1 to 10 of `out.txt` re-ran unchanged except where the onset
  now binds (10.2: `n42_parts` and the seven manual grades) and the allowed temperature printed for adhesive code 0,
  NaN by the rule of section 4.4 (the verdict there is unknown either way).
- Revised again after spec review 4 (`spec-review-4.md`, must-fix MF-1 and MF-2 and the should-fix items accepted
  there under "Writer's response"): R-15's which-ring question went to the owner (answered: the next bullet), and
  note sentence 4 keeps the wheel shaft. The numbers that revision added are in section 12 of `out.txt` (function
  `review4()`; `Settings.onset` reads the inner ring's own onset for option (b)); sections 1 to 11 re-ran unchanged.
- Revised again after the owner's ring ruling (the "R-15 ring" row of `owner-decisions.md`: option (b), section 2)
  and spec review 5 (`spec-review-5.md`, must-fix MF-1 and MF-2 and should-fix SF-1 to SF-4, listed under its
  "Writer's response (inner-ring onset and round 5)"). The bonded inner hub's cap reads the **inner ring's own**
  single-ring onset, a new Rust-only result `temperature.demag.inner_onset_single_ring_C` (sections 4.1, 11); note
  sentences 3 and 4 name only number results. The scratch crate's `check` now reads the inner ring's own onset
  (`Settings.governing` false; true gives the governing ring's, option (a), for comparison), and every library part
  it names is checked by `library::lookup` (`part()`). Sections 1 to 12 of `out.txt` re-ran unchanged except at mixed
  rings: 11.2 now changes the inner part to the library's N42SH parts, 11.5's `breaking(custom_hot_day)` prints the
  inner ring's own onset, 11.9 reads option (b), 11.12b prints both options, and 12.3 is re-run over
  `MAGNET_LIBRARY`. The numbers this revision added are in section 13 (function `review5()`).
- Revised again after spec review 6 (`spec-review-6.md`, should-fix SF-1 to SF-4, listed under its "Writer's
  response"). Note sentence 3 names the joint whose bore stress it quotes, the cup. The mixed-ring pins of 7.1 name
  the grade id they set and assert the ring that governs. Section 8 gains two doc lines. Sentence 4's render on a
  40 °C wheel shaft is listed. The numbers this revision added are in section 14 of `out.txt` (function `review6()`);
  sections 1 to 13 re-ran byte for byte.

## 1. Purpose

Nothing in the calculator checks how the two rotors are held on their shafts. The owner's ruling of 2026-10-17: "Hub is
shrink fit with key only" (no clamp screws, no adapter joint). The drive line is motor -> GAM 5:1 gearbox -> coupling ->
wheel. The coupling has two rotor-to-shaft joints, and both carry the coupling's torque:

- the **inner hub** (ten flats, inner magnets bonded on them) on the **wheel shaft**;
- the **cup** (outer magnets in its pocket, rear web and integral boss) on the **gearbox output shaft**.

Each is a keyed shrink fit on a Ø10 shaft (`coupling.bore_mm`, one input for both). Cold-high
(`metal.torque_cold_high_Nm`, the pull-out at the minimum magnet temperature with the production allowance, 3.8233 N·m
at the defaults, E) can reach each joint before the coupling slips. It is not a bound on them: backlash, resonance
crossings and re-lock ring-down can exceed it on the gearbox side, and with the operating temperature below the minimum
the operating pull-out with its allowance is above it (BL-067: 3.657 against 3.1823 N·m in the corner of section 7.1).
While the coupling slips each joint carries a fully reversing swing of about that amplitude, more on the gearbox side
with backlash, at 166.7 full cycles a second (E, `metal.slip_freq_Hz`). This paragraph claims no order of the torques,
and neither may any text built from it (review 2, MF-1).

BL-064 adds one counted dashboard check, `shaft_joint.check`, with its slice-1 guide (Problem, Why, Fix) and a new
reviewed teaching note, `shaft_joint`. For each joint the check asks four things, in this order:

1. the fit alone holds the safety factor times cold-high at its loosest;
2. the part stays within its stress limit at its tightest;
3. the part can be fitted within its allowed temperature, with its shaft at its fitting temperature;
4. the key alone, as a backup, carries the same torque within the stress limit.

Success: an engineer sees on the dashboard whether both joints hold, stay elastic and can be built; when one cannot, the
guide names the joint and the cause, quotes the live numbers, gives the window of measured interference that would work
(or says in numbers that there is none), and offers the inputs that move the failing term.

The paused shaft-clamp section stays paused and unchanged (owner decision 6A). The check is outside its `SECTION`, so it
counts while the clamp is paused.

## 2. Owner decisions (2026-10-17)

These are binding and override the synthesis where they differ.

| # | Decision |
|---|---|
| 1A | **Both joints, one counted dashboard row.** One pure joint function evaluated twice (inner hub, cup); the verdict names the failing joint. |
| 2A | **The fit holds; the key is a backup.** The fit alone must hold safety factor × cold-high at its loosest; the part must stay elastic at its tightest (yield ÷ 1.2); the key is checked alone at the same load as a backup; fit and key capacities are never added. |
| 3A | **Minimum and maximum measured diametral interference as inputs (µm).** |
| Bands | **One band per joint**, each with its own assembly inputs. Defaults (placeholders until the real fits are set): **inner hub 13-18 µm; cup 15-22 µm**. |
| 4A | **New key inputs, DIN 6885-1 3 × 3 on a 10 mm shaft:** key width 3.0 mm, key contact height 1.2 mm, shared by both joints. The paused clamp's key inputs stay with the paused section. The drawing shows the new key. |
| 5A | **Safety factor input, default 2.0**, on slip and on the key backup. The stress limit is yield ÷ 1.2, a named engine constant the physics review confirms. |
| 6A | **The paused clamp section stays paused and unchanged;** comments and docs that say it returns with BL-064 are reworded. |
| 7A | **Assembly feasibility is a counted cause ("cannot be fitted"), per joint.** The part's needed temperature, 20 °C + (maximum interference + joining clearance − shaft shrink at its fitting temperature) ÷ (α_part · d), must be at or below the part's allowed fitting temperature. |
| Process | **The bare cup is shrunk onto the gearbox shaft first, then its magnets are bonded; the inner hub is bonded first, then shrunk onto the cooled wheel shaft.** |
| Hub temp | **Allowed (bonded) inner hub fitting temperature: an input, default 100 °C,** never above the selected adhesive's design limit (120 °C for AA 326). R-15 below adds the inner magnets' onset. |
| Cup temp | **Allowed bare-cup fitting temperature: an input, default 300 °C** (below the 4140 and UltraOx treatment temperatures). The note says to keep the heat off the gearbox seal and fit quickly; the physics review checks the seal side. |
| Shaft temps | **Shaft fitting temperatures as inputs: wheel shaft −196 °C (liquid nitrogen), gearbox shaft 20 °C** (it cannot be immersed; its slider bottom is −20 °C, cold spray). The shrink below 20 °C uses integrated NIST expansion data, not a constant coefficient. |
| 8A | **A new reviewed teaching note `shaft_joint`; the note cap rises from 20 to 21.** Its Start here step goes just before `clamp_preload`. |
| Material | **Inner hub and cup: 4140 with an UltraOx finish, the bore masked.** The condition is not given: the check uses the library's 4140 yield (415 MPa, the annealed value, conservative). The engine's plating inputs (electroless nickel) are not BL-064's: recorded as a new backlog item (section 8). |
| Shafts | **Both shafts Ø10 for now** (one bore input, `coupling.bore_mm`). |
| Release | As BL-059: the physics re-review of the note and the G-3 review of every new text, the whole-branch review, GATE PASS with gate 12, the web build and gui-smoke, then **STOP for the owner's review of the diff**; merge and push only on the owner's OK. |
| R-15 | **(owner, 2026-10-17, after spec review 3) The bonded inner hub's allowed fitting temperature is the lowest of the owner's input (100 °C), the selected adhesive's design limit (120 °C) and the magnets' single-ring onset less `CURE_MARGIN_MIN_C` (10 °C),** the margin the temperature verdict already keeps between the adhesive cure and the single-ring onset. Which ring's onset: the next row. No change at the defaults (N42SH, onset 143.36 °C, cap 133.36 °C); lower grades read "cannot be fitted"; ferrite (onset +∞) is unaffected. The row reads a stored 3D field, so it joins `STORED_3D_ROWS` and carries the "Fixed 3D fields from the example design" label. Its consequences are in section 11, R-15. |
| R-15 ring | **(owner, 2026-10-17, after spec review 5) The cap reads the INNER ring's own single-ring onset:** one new Rust-only result, `temperature.demag.inner_onset_single_ring_C`, beside `temperature.demag.inner_magnet_limit_C`, from the inner ring's own onsets (`temperature.rs:931`, `:1015`), with its record, its scope path and the counts it moves (sections 4.1, 4.5, 7.17). The hub's fitting heat reaches only the inner magnets: the cup's are bonded after the cup is fitted. No change at the defaults or the probes. The temperature verdict's own cure-margin gap (it reads the governing ring's onset) stays BL-072's (section 8). The record of this ruling is in section 11, R-15. |

**Technical decisions taken as recommended** (synthesis 9-23), with the corrections of `research-assembly.md` and
`owner-decisions.md`:
- 9: a new Rust-only top-level **input** group and **result** group, both `shaft_joint`, in a new engine module
  `M/src/engine/shaft_joint.rs`, computed in `api.rs` after `gearbox`. No path contains "clamp" or "gearbox", or starts
  with `clamps.`.
- 10: `DASHBOARD` index 15, after the hidden clamp row and before the gearbox row; `SHAFT_JOINT_GUIDE` between
  `CLAMP_GUIDE` and `GEARBOX_GUIDE`.
- 11: the row greys with the end-effect rows (`END_EFFECT_ROWS` 8 -> 9).
- 12: one verdict text, causes in a fixed order, numbers never equal; the window in the Problem.
- 13: the temperature model of section 3.3 (hot end = the selected adhesive's design limit, which reads no stored 3D
  field). The row joins `STORED_3D_ROWS` all the same, through R-15's onset term on the hub's allowed fitting
  temperature (section 5.1), not through the hot end. The hub-over-shaft allowance applies to both joints, so its path
  is `rotor_over_shaft_C`.
- 14: one hub-material helper, read by the warnings and the joint (DRY); yield from the library.
- 15: geometry of section 3.2; key bearing length L − b; a non-positive bearing length fails.
- 16: friction, smoothing, the rotor-over-shaft allowance and the joining clearance are assumption inputs, with
  `ASSUMPTIONS` rows (four, not three: research-assembly made the joining clearance an assumption).
- 17: `Fixes::PerTerm`, extended to the fitting cause and per joint (section 5.2).
- 18: `passing_design()` unchanged (it passes the joint); a new `only_the_joint_fails()`.
- 19: the share-link guard raised, with the measured length.
- 20: the key drawn in both states from the new width.
- 21: Auto unchanged.
- 22: µm units.
- 23: no sign-off sheet regeneration; `CLAMPS_PAUSED_LINE` reworded.

**Where this spec departs from the synthesis** (each checked in the scratch crate):
- **Fifteen inputs, not eight:** the owner's per-joint bands (4), the four fitting temperatures and the joining
  clearance (5) join the six shared fit and key inputs.
- **The key flank is checked against the stress limit (yield ÷ 1.2), not against yield.** One named limit for the part,
  conservative, and close to the yield of the C45+C key steel DIN keys are drawn from (about 340 MPa, from memory). It
  binds nowhere at the five probes. The physics review confirms or changes it (section 11).
- **Joint names on screen are "inner hub" and "cup"** (the owner's words), not "cup boss". F1 is answered by owner
  decision 1A, so texts may name the shafts: the wheel shaft and the gearbox shaft.
- **The hub material in effect is three result rows** (modulus, expansion, yield), so the equation records and the
  note read them instead of repeating the material rule.
- **Synthesis decision 7A (an informational fitting temperature, no badge)** is replaced by owner decision 7A (a
  counted cause).
- **The row reads the stored 3D fields** (owner R-15): the hub's allowed fitting temperature is capped at the inner
  ring's own single-ring onset less 10 °C, so the row joins `STORED_3D_ROWS`, which synthesis 13 kept it out of.
- **One temperature result is added** (owner R-15 ring): `temperature.demag.inner_onset_single_ring_C`, the inner
  ring's own single-ring onset, so the check never reads the outer ring's onset for the inner hub (section 4.1).

**One joining clearance for both joints (a deliberate technical choice).** The owner's "Bands" row gives each joint its
own assembly inputs: here its allowed fitting temperature and its shaft's fitting temperature (four inputs). The joining
clearance stays one shared assumption input, as the owner's decisions name it ("joining clearance 10 µm as an
assumption input", from research-assembly 3): both joints sit on the one Ø10 bore, and the rule of thumb behind the
10 µm (0.001·d, from memory) depends on the bore alone. The clearance closes fastest on the liquid-nitrogen side
(research-assembly 3), so the shared value is the one that side needs; its help says less needs a guided, single fast
push. Splitting it into a hub and a cup clearance later is one more input and a share-link re-measure (section 5.4).

**Facts still open** (recorded in BL-064's notes; none blocks the build):
- the gearbox output shaft's real diameter, tolerance and key; the wheel shaft's diameter, tolerance and material;
- the real interference bands (13-18 and 15-22 are placeholders);
- the 4140 condition (annealed 415 MPa or pre-hardened) and the UltraOx process and its temperature;
- the key length, and the real keyway depth: the Python-locked 1.7 mm default (`coupling.keyway_depth_mm`) matches no
  DIN 6885-1 row (3 × 3 has a hub depth t2 of 1.4 mm); the key contact input (1.2 mm) is separate;
- the bondline thickness: at 0.05 mm the engine's thermal-mismatch screen reaches the AA 326 lap shear at about 113 °C
  (research-assembly 1.2, C from E); at 0.10 mm at about 198 °C. The 100 °C hub default rests on the 0.05 mm figure.

## 3. Physics basis (numbers from `compute_all` at fb79095 and the scratch crate)

### 3.1 Formulas (per joint; plane-stress Lamé thick cylinders, solid shaft)

Units: lengths d, D, L, b, k, l in mm; interferences δ, smoothing, the clearance s_join and the shrink S in µm (×10⁻³
for mm); moduli E and stresses in MPa (N/mm²); torques in N·m (×1000 for N·mm); expansion coefficients α in 1/°C.

- Contact pressure per µm of effective diametral interference, with Q = d / D (bore over the part's outer diameter):
  p = 10⁻³ / ( d · [ (1/E_h)·((1 + Q²)/(1 − Q²) + ν) + (1/E_s)·(1 − ν) ] ) [MPa per µm].
- Holding torque on the contact, the key's arc removed: T = μ · p · max(0, π·d − b) · L · d / 2 [N·mm; ÷ 1000 for
  N·m], which is μ·p·π·d²·L/2 × (1 − b/(π·d)). With no keyway (`coupling.keyway_depth_mm` = 0) b = 0. The arc is
  floored at zero (review 2, SF-4): a key width typed above π·d (31.4 mm on the Ø10 bore; 40 mm in the examples below)
  would otherwise give a negative holding (−2.87 N·m at the inner hub) and a negative window end (−4.39 µm); floored,
  the holding is 0 and the needed minimum +∞ (C, `out.txt` 10.5). The key-width slider (1-6 mm) never reaches it.
- Bore von Mises stress: σ_θ = p·(1 + Q²)/(1 − Q²), σ_r = −p, σ_vM = √(σ_θ² + σ_θ·p + p²) [MPa].
- Key flank pressure, the key alone at the same load: p_key = 2·T_req·1000 / (d · k · l) [MPa, T_req in N·m], with
  k = min(key contact height, keyway depth) and l = L − b (form A). l ≤ 0 or k ≤ 0 means the key has no bearing area:
  p_key = +∞ and the key fails (a typed contact of −0.5 mm would otherwise give −305.9 MPa, a pass; C). With no keyway
  the key term does not apply (p_key reads 0).
- The key's bearing length needed at the limit (with a keyway): l_needed = 2·T_req·1000 / (d · k · R_e/1.2) [mm], +∞
  only when k ≤ 0. The key passes exactly when l ≥ l_needed (for T_req > 0), so l − l_needed is the key term's margin
  (section 5.2): finite at every slider value, rising with the boss or hub length and with the contact height. With no
  keyway, k = min(contact, 0) = 0 would make it −∞ on a term that does not apply, so the keyless key term passes with
  margin +∞ (review 2, SF-5c).
- Required torque: T_req = safety factor × cold-high.
- Stress limit: R_e ÷ 1.2 (`YIELD_FACTOR` = 1.2).
- Needed fitting temperature: T_fit = 20 °C + (δ_max + s_join − S(T_shaft))·10⁻³ / (α_part · d) [°C], with S the
  shaft's diametral shrink from 20 °C to its fitting temperature [µm], positive when colder, negative above 20 °C:
  S = (strain from 20 °C) × d × 1000.
- Allowed fitting temperature (owner R-15): inner hub T_allow = min(T_input, T_adhesive, ϑ_sr,i − 10 °C), with T_input
  the owner's input, T_adhesive the selected adhesive's design limit, ϑ_sr,i the inner ring's own single-ring onset
  (`temperature.demag.inner_onset_single_ring_C`, section 4.1; owner R-15 ring: the ring the bonded hub carries,
  whichever ring governs) and 10 °C `CURE_MARGIN_MIN_C`; NaN when any of the three is NaN (section 4.4). Cup
  T_allow = its input (the cup is
  fitted bare). The fittable edge of the window, δ_fit = α_part·(T_allow − 20 °C)·d·1000 + S − s_join [µm], reads
  T_allow, so it moves with the onset where the onset binds.
- **No wall.** A joint has no wall when Q ≥ 1 (the bore is as wide as the part), or, with a keyway, when the wall
  (D − d)/2 is no thicker than the keyway depth (the keyway cuts through it). Its Lamé terms then mean nothing: its
  holding, bore stress, needed and elastic results read NaN, and the joint is not judged (section 4.3).

### 3.2 Geometry and material (E, defaults)

| quantity | path | inner hub | cup |
|---|---|---|---|
| bore d | `coupling.bore_mm` | 10.0 mm | 10.0 mm |
| outer diameter D | hub: 2·(`coupling.inner_back_apothem_mm` − `metal.bond_inner_mm`), the flats' incircle (conservative); cup: `metal.boss_od_mm` (conservative for the web) | 20.20 mm | 22.00 mm |
| engaged length L | hub: `housing.hub_length_mm` (follows the magnets under the axial override); cup: `metal.boss_length_mm` + `metal.web_mm` (the bore runs through both, `gui/geometry.rs:662-663`) | 13.00 mm | 15.50 mm |
| key bearing length l = L − b | — | 10.0 mm | 12.5 mm |
| part material in effect | `parts.steel` when the circuit is steel, else `parts.body` (the rule of `api.rs:349-353`) | 4140: E 205 GPa, α 12.3e-6 /°C, R_e 415 MPa (library) | same |
| cold-high | `metal.torque_cold_high_Nm` | 3.8233 N·m | same |
| minimum magnet temperature | `metal.min_temp_C` | −40 °C | same |
| hot end (bond limit in service) | `temperature.adhesive.design_limit_C` | 120 °C (AA 326) | same |
| inner ring's own single-ring onset (R-15 and its ring ruling) | `temperature.demag.inner_onset_single_ring_C` (new, section 4.1; a stored 3D field feeds it; equal to the governing ring's `temperature.demag.onset_single_ring_C` at the defaults, E) | 143.36 °C (N42SH), cap 133.36 °C | not read (the cup is fitted bare) |

Constants (named in the engine, each documented; all need the physics review):
- shaft E 205 GPa, ν 0.30 (shaft and part), shaft expansion 11.5e-6 /°C above −8.15 °C (265 K);
- room temperature 20 °C; `YIELD_FACTOR` 1.2;
- the onset margin is not a new constant: the check reads `temperature::CURE_MARGIN_MIN_C` (10 °C,
  `temperature.rs:746`), the temperature verdict's cure margin, so the two rules cannot drift apart (DRY);
- the shaft shrink below 265 K from NIST's linear-expansion fit for nickel steel Fe-2.25Ni:
  y = (L − L293)/L293 × 10⁵ = −210.4 − 0.05699·T + 5.072e-3·T² − 1.381e-5·T³ + 1.897e-8·T⁴ (T in K, equation range
  4-265 K, 1.5 % fit error). Coefficients re-checked this session on the NIST page
  (`trc.nist.gov/cryogenics/materials/NickelSteel/NickelSteel_Fe2.25Ni)_rev.htm`). The fit is anchored at 265 K and
  the linear 11.5e-6 added from 265 K to 20 °C. It is the nearest ferritic steel NIST lists (no 4140 or 1045 page);
  that other ferritic steels contract within about 5-10 % of it is from memory.

### 3.3 Temperature terms (synthesis decision 13)

Δ(T) = (α_shaft − α_part)·(T − 20 °C)·d·1000 [µm] is the change of interference at a uniform temperature T.
- Loosest: δ_eff,min = max(0, δ_min − smoothing + min[Δ(T_min), Δ(T_hot) − α_part·ΔT_rotor·d·1000]).
- Tightest: δ_eff,max = max(0, δ_max − smoothing + max[Δ(T_min), Δ(T_hot)]).
- T_min = `metal.min_temp_C`; T_hot = the selected adhesive's design limit; ΔT_rotor = the rotor-over-shaft allowance
  (hot end only). No speed term: the centrifugal loss is 0.0017 µm at 2000 rpm (synthesis, C).
- The hot-end term puts the shaft at T_hot and the rotor ΔT_rotor above it. Which body sits at the bond limit is R-14
  (section 11) for the physics review; the 7.1 test "the 416 part's minimum is taken cold" pins this form.

At the defaults (4140 on steel): Δ(−40 °C) = +0.48 µm; Δ(120 °C) = −0.80 µm; the 20 K allowance takes 2.46 µm; so
the loosest end loses 3.26 µm and the tightest gains 0.48 µm (C).

### 3.4 Per µm of effective interference (C, defaults)

| | inner hub (Ø20.2, L 13) | cup (Ø22, L 15.5) |
|---|---|---|
| contact pressure | 7.738 MPa | 8.132 MPa |
| holding torque (μ 0.14, 3 mm key arc removed) | 2.0009 N·m | 2.5073 N·m |
| bore von Mises | 17.930 MPa | 17.879 MPa |

### 3.5 The defaults pass (C)

| | inner hub | cup |
|---|---|---|
| measured band | 13-18 µm | 15-22 µm |
| effective at the loosest / tightest | 4.74 / 13.48 µm | 6.74 / 17.48 µm |
| holding at the minimum, against 2 × 3.8233 = 7.6466 N·m | **9.484 N·m** (1.24 × the need, 2.48 × cold-high) | **16.899 N·m** (2.21 ×, 4.42 ×) |
| bore stress at the maximum, against 415 ÷ 1.2 = 345.83 MPa | **241.7 MPa** (70 % of the limit) | **312.5 MPa** (90 %) |
| key alone (3 × 1.2 mm), against 345.83 MPa | **127.4 MPa** (l = 10 mm) | **102.0 MPa** (l = 12.5 mm) |
| shaft at fitting; its shrink | wheel shaft −196 °C; 18.99 µm | gearbox shaft 20 °C; 0 µm |
| needed fitting temperature, against allowed | **93.3 °C against 100 °C** (the lowest of your 100, the adhesive's 120 and the onset 143.36 less 10, 133.36: your limit binds) | **280.2 °C against 300 °C** |
| window: smallest minimum that holds - largest maximum that stays elastic and fits | **12.08 - 18.83 µm** (the fitting edge binds; elastic edge 23.81) | **11.31 - 23.86 µm** (the stress edge binds; fitting edge 24.44) |
| verdict | holds, elastic, fits, key OK | holds, elastic, fits, key OK |

**Verdict at the defaults: "Both shaft joints hold and can be fitted".** The hub band sits 0.92 µm above its needed
minimum and 0.83 µm under its fitting edge (one 0.5 µm step up on the maximum, 18.5 µm, needs 97.3 °C and still passes;
19 µm needs 101.4 °C and fails). The cup band has 3.69 µm and 1.86 µm to spare. The window row gives the results
(12.0815, 18.8281, 11.3098, 23.8626 µm, C); a Problem quotes a window inward (section 5.2): 12.09 - 18.82 and
11.31 - 23.86 µm. With the clamp paused, the status reads **"Example design: fails 4 of 9 checks"** (hot-low, running
clearance, cup wall and gearbox input torque fail, E via `design_status` at fb79095: "fails 4 of 8" today).

What the hub's needed minimum (12.08 µm) is made of: torque 3.82, smoothing 5.00, rotor over shaft 2.46, the expansion
mismatch at 120 °C 0.80 µm (C). The cup's 11.31: 3.05 + 5.00 + 2.46 + 0.80.

### 3.6 Probe designs and edges (C; holding against the need, stress against the limit, window, key, fitting)

"h / c" is inner hub / cup. The window is "needed - min(elastic, fittable)"; "empty" when needed exceeds it. Every row
uses the owner's defaults except the edit named.

| design | holding h / c vs need (N·m) | bore stress h / c vs limit (MPa) | window h / c (µm) | key h / c (MPa) | fitting h / c, needed vs allowed (°C) | verdict (badge) |
|---|---|---|---|---|---|---|
| default | 9.48 / 16.90 vs 7.65 | 241.7 / 312.5 vs 345.8 | 12.08-18.83 / 11.31-23.86 | 127.4 / 102.0 | 93.3 vs 100 / 280.2 vs 300 | passes |
| 8 magnets per ring | 9.48 / 16.90 vs 5.60 | 241.7 / 312.5 vs 345.8 | 11.06-18.83 / 10.49-23.86 | 93.3 / 74.6 | 93.3 vs 100 / 280.2 vs 300 | passes |
| 304 back iron (E 193, α 16.9e-6, R_e 205) | 0.00 / 2.93 vs 4.83 | 278.5 / 346.3 vs 170.8 | empty: needs 16.30, elastic to 11.72 / empty: 15.79 vs 11.74 | 80.5 / 64.4 | 73.3 vs 100 / 209.3 vs 300 | slips (inner hub); also slips (cup), both overstressed |
| Y30 ferrite, custom dimensions | 9.48 / 16.90 vs 0.68 | 241.7 / 312.5 vs 345.8 | 8.60-18.83 / 8.53-23.86 | 11.3 / 9.0 | 93.3 vs 100 / 280.2 vs 300 | passes |
| 20 mm ring radius (hub Ø39.9) | 11.77 / 16.90 vs 7.22 | 239.5 / 312.5 vs 345.8 | 11.17-18.83 / 11.14-23.86 | 120.4 / 96.3 | 93.3 vs 100 / 280.2 vs 300 | passes |
| `passing_design()` | 9.48 / 16.90 vs 6.72 | 241.7 / 312.5 vs 345.8 | 11.62-18.83 / 10.94-23.86 | 111.9 / 89.6 | 93.3 vs 100 / 280.2 vs 300 | passes |
| `passing_design()`, hub minimum 8 (`only_the_joint_fails()`) | 0.00 / 16.90 vs 6.72 | as above | as above | 111.9 / 89.6 | as above | slips (inner hub) |
| default, hub minimum 10 | 3.48 / 16.90 vs 7.65 | 241.7 / 312.5 | 12.08-18.83 / 11.31-23.86 | 127.4 / 102.0 | 93.3 / 280.2 | slips (inner hub) |
| default, cup minimum 10 | 9.48 / 4.36 vs 7.65 | 241.7 / 312.5 | as default | 127.4 / 102.0 | 93.3 / 280.2 | slips (cup) |
| default, hub maximum 26 / 28 | 9.48 / 16.90 | 385.1 / 421.0 (hub) vs 345.8 | as default | 127.4 / 102.0 | 158.3 / 174.6 vs 100 (hub) | bore stress (inner hub); also cannot be fitted |
| default, cup maximum 26 / 28 | 9.48 / 16.90 | 384.0 / 419.8 (cup) vs 345.8 | as default | 127.4 / 102.0 | 312.7 / 328.9 vs 300 (cup) | bore stress (cup); also cannot be fitted |
| default, friction 0.10 | 6.77 / 12.07 vs 7.65 | 241.7 / 312.5 | 13.61-18.83 / 12.53-23.86 | 127.4 / 102.0 | 93.3 / 280.2 | slips (inner hub) |
| default, safety factor 1.0 | 9.48 / 16.90 vs 3.82 | 241.7 / 312.5 | 10.17-18.83 / 9.78-23.86 | 63.7 / 51.0 | 93.3 / 280.2 | passes |
| default, smoothing 0 | 19.49 / 29.44 vs 7.65 | 331.4 / **401.9** vs 345.8 | 7.08-18.81 / 6.31-18.86 | 127.4 / 102.0 | 93.3 / 280.2 | bore stress (cup) |
| default, rotor over shaft 0 K | 14.41 / 23.07 vs 7.65 | 241.7 / 312.5 | 9.62-18.83 / 8.85-23.86 | 127.4 / 102.0 | 93.3 / 280.2 | passes |
| default, joining clearance 0 / 30 | as default | as default | 0: 12.08-23.81 / 11.31-23.86; 30: empty both | 127.4 / 102.0 | 0: 12.0 / 198.9; 30: 255.9 vs 100 / 442.8 vs 300 | 0: passes; 30: cannot be fitted (inner hub) |
| keyless (keyway depth 0) | 10.49 / 18.68 vs 7.65 | 241.7 / 312.5 | 11.72-18.83 / 11.02-23.86 | 0 / 0 (no key) | 93.3 / 280.2 | passes |
| keyway depth 1.0 (contact capped at 1.0) | 9.48 / 16.90 | 241.7 / 312.5 | as default | 152.9 / 122.3 | 93.3 / 280.2 | passes |
| 1018 back iron (R_e 220) | 10.20 / 17.80 vs 7.65 | 238.5 / 309.3 vs 183.3 | 11.72-14.92 / 10.95-14.95 | 127.4 / 102.0 | 95.1 / 286.7 | bore stress (inner hub) |
| 12L14 (R_e 415, α 11.5e-6) | 11.20 / 18.96 | 228.9 / 298.6 vs 345.8 | 11.19-18.19 / 10.40-22.20 | 127.4 / 102.0 | 98.4 vs 100 / 298.3 vs 300 | passes (narrowly on fitting) |
| 416 (R_e 276, α 10.08e-6) | 14.04 / 22.53 | 253.9 / 323.5 vs 230.0 | 9.74-16.64 / 8.96-16.68 | 127.4 / 102.0 | 109.4 vs 100 / 337.5 vs 300 | bore stress (inner hub); also cannot be fitted |
| 17-4PH H1150 (R_e 725) | 10.25 / 17.77 | 233.0 / 302.7 vs 604.2 | 11.67-18.51 / 10.89-23.32 | 127.4 / 102.0 | 95.7 / 288.9 | passes |
| 17-4PH H900 (R_e 1170, α 10.8e-6) | 12.84 / 21.02 | 241.1 / 310.7 vs 975.0 | 10.35-17.63 / 9.57-20.24 | 127.4 / 102.0 | 103.4 vs 100 / 316.3 vs 300 | cannot be fitted (inner hub) |
| 6061 pick, or `coupling.backiron` 0 (E 68.9, α 23.6e-6, R_e 241), with any pick (one outside the choices too) | 0.00 / 0.00 vs 4.83 | 148.1 / 178.7 vs 200.8 | empty: 27.74 vs 25.22 / 26.49 vs 25.00 | 80.5 / 64.4 | 58.2 / 155.6 | slips (inner hub) |
| back-iron pick outside the choices (99), steel circuit | NaN | NaN | — | — | — | unknown: a value is not a number (the steel values and the yield are NaN) |
| axial override 13.77 mm (Auto on length) | 10.27 / 16.90 vs 8.37 | 241.7 / 312.5 | 12.12-18.83 / 11.60-23.86 | 126.0 / 111.5 | 93.3 / 280.2 | passes |
| axial override 40 mm (hub 40.3 mm long) | 29.40 / 16.90 vs 25.99 | 241.7 / 312.5 | 12.45-18.83 / 18.63-23.86 | 116.1 / 346.6 | 93.3 / 280.2 | slips (cup); also key (cup) |
| `breaking(long_magnets)` (50.8 mm) | 37.28 / 16.90 vs 29.24 | 241.7 / 312.5 | 11.98-18.83 / 19.92-23.86 | 101.3 / 389.9 | 93.3 / 280.2 | slips (cup); also key (cup) |
| `breaking(no_back_iron)` | 0.00 / 0.00 vs 4.22 | 148.1 / 178.7 vs 200.8 | empty / empty | 70.3 / 56.2 | 58.2 / 155.6 | slips (inner hub) |
| `metal.boss_length_mm` 0 (cup L 2.5 < b 3) | 9.48 / 2.73 vs 7.65 | 241.7 / 312.5 | 12.08-18.83 / empty (27.17 vs 23.86) | 127.4 / +∞ | 93.3 / 280.2 | slips (cup); also the key has no bearing area (cup) |
| `passing_design()`, boss length 0 (key margin: l −0.50 − l_needed 3.24 = −3.74 mm) | 9.48 / 2.73 vs 6.72 | 241.7 / 312.5 | as `passing_design()` / empty | 111.9 / +∞ | 93.3 / 280.2 | slips (cup); also the key has no bearing area (cup); every other check green |
| `metal.hub_length_mm` 3.0, its slider's bottom (hub L 3 = b) | 2.19 / 16.90 vs 7.65 | 241.7 / 312.5 | empty (24.82 vs 18.83) / as default | +∞ / 102.0 | 93.3 / 280.2 | slips (inner hub); also the key has no bearing area (inner hub) |
| `passing_design()`, hub length 3.0 (key margin −3.24 mm; slip margin −9.81 µm) | 2.19 / 16.90 vs 6.72 | 241.7 / 312.5 | empty (22.81 vs 18.83) / as `passing_design()` | +∞ / 89.6 | 93.3 / 280.2 | slips (inner hub); also the key (inner hub); every other check green; the key passes from 6.3 mm, the slip from 9.3 mm |
| `breaking(cool_adhesive)` (`passing_design()`, DP460) | 10.44 / 18.10 vs 6.72 | 241.7 / 312.5 | 11.14-13.91 / 10.46-23.86 | 111.9 / 89.6 | 93.3 vs **60** / 280.2 vs 300 | cannot be fitted (inner hub), the only failing term (the temperature check is amber there) |
| `breaking(n42_parts)` (`passing_design()`, library N42 blocks B842 in both rings; onset 69.83 °C; R-15, `out.txt` 11.2) | 9.48 / 16.90 vs 6.72 | 241.7 / 312.5 | 11.62-13.89 / 10.94-23.86 | 111.9 / 89.6 | 93.3 vs **59.8** (the onset less 10 °C) / 280.2 vs 300 | cannot be fitted (inner hub), the only failing term (the temperature check is amber there); round 3 read "passes" |
| `passing_design()`, N50M on manual dimensions (onset 98.80 °C, above the need; less 10 °C, 88.80 °C, below it; 11.14) | 9.48 / 16.90 vs 7.90 | 241.7 / 312.5 | 12.21-17.45 / 11.41-23.86 | 131.7 / 105.3 | 93.3 vs **88.8** / 280.2 vs 300 | cannot be fitted (inner hub); the cup wall fails too and the temperature check is amber |
| default, N52 on manual dimensions (onset 81.77 °C) | 9.48 / 16.90 vs 9.51 | 241.7 / 312.5 | 13.01-15.36 / 12.05-23.86 | 158.6 / 126.8 | 93.3 vs 71.8 / 280.2 vs 300 | slips (inner hub); also cannot be fitted (inner hub) |
| default, `temperature.demag.h_rev_single_ring_kA_m` 1500 (its slider's top; onset −2.45 °C) | as default | as default | hub empty: needs 12.08, fits to 5.00 | as default | 93.3 vs **−12.4** / 280.2 vs 300 | cannot be fitted (inner hub): the row moves with a stored 3D input (section 5.1; 11.6) |
| default, 2214 Hi-Temp, hub input 150 °C typed (above the 130 °C slider top) | 8.57 / 15.76 vs 7.65 | 241.7 / 312.5 | 12.54-22.93 / 11.77-23.86 | 127.4 / 102.0 | 93.3 vs 133.4 (the onset less 10 °C binds) / 280.2 vs 300 | passes |
| `coupling.c_end` 1.5 (f_end < 0) | 9.48 / 16.90 vs −0.35 | 241.7 / 312.5 | — | −5.8 / −4.6 | 93.3 / 280.2 | "passes", meaningless: the row greys |
| BL-067 corner (op −40, min 20, variation 0.1) | 9.48 / 16.90 vs 6.36 | 233.1 / 303.9 | 11.44-18.83 / 10.80-24.34 | 106.1 / 84.9 | 93.3 / 280.2 | passes (reads cold-high 3.18, not the 3.66 operating pull-out: BL-067) |
| default, hub minimum 20 > maximum 18 | 23.49 / 16.90 | 241.7 / 312.5 | — | — | — | unknown: the band is reversed (inner hub); the terms are still computed |
| default, hub minimum 20 > maximum 18, cup minimum 10 | 23.49 / 4.36 vs 7.65 | 241.7 / 312.5 | — | — | — | slips (cup): the determinate cup's red cause outranks the hub's reversed band (section 4.3) |
| default, wheel shaft 20 °C (not cooled) | as default | as default | hub empty: needs 12.08, fits to −0.16 | as default | **247.6 vs 100** / 280.2 vs 300 | cannot be fitted (inner hub) |
| default, wheel shaft −78.5 °C (dry ice) | as default | as default | hub empty: 12.08 vs 10.19 | as default | 163.5 vs 100 / 280.2 | cannot be fitted (inner hub) |
| default, gearbox shaft 40 °C / −20 °C | as default | as default | cup 11.31-22.14 / 11.31-23.86 | as default | 93.3 / 298.9 vs 300; 93.3 / 243.1 vs 300 | passes |
| default, bare cup allowed 250 °C | as default | as default | cup 11.31-18.29 | as default | 93.3 / 280.2 vs 250 | cannot be fitted (cup) |
| default, adhesive DP460 (limit 60 °C) | 10.44 / 18.10 | 241.7 / 312.5 | 11.60-13.91 / 10.83-23.86 | 127.4 / 102.0 | 93.3 vs **60** / 280.2 vs 300 | cannot be fitted (inner hub) |
| default, adhesive code 0 (no adhesive: NaN limit) | — | — | — | — | — | unknown: a value is not a number |
| default, bore 16 mm | 4.57 / 11.85 vs 7.65 | 162.5 / 206.1 | 14.87-33.54 / 13.30-34.04 | 79.7 / 63.7 | 7.9 / 182.6 | slips (inner hub) |
| bore 16, boss Ø12 (cup Q ≥ 1: no wall) | 4.57 / NaN vs 7.65 | 162.5 / NaN | 14.87-33.54 / — | 79.7 / 63.7 | 7.9 / 182.6 | slips (inner hub); the Problem adds the cup's no-wall sentence (without SF-9's NaN the cup would read −19.56 N·m, −282.5 MPa, elastic edge −17.52 µm) |
| bore 16, boss Ø12, hub minimum 15 | 7.86 / NaN vs 7.65 | 162.5 / NaN | 14.87-33.54 / — | 79.7 / 63.7 | 7.9 / 182.6 | unknown: no wall around the bore (cup) |
| bore 16, keyway 2.1 (hub wall 2.10 mm, no thicker than the keyway) | NaN / 11.85 vs 7.65 | NaN / 206.1 | — / 13.30-34.04 | 79.7 / 63.7 | 7.9 / 182.6 | unknown: no wall around the bore (inner hub); the cup passes |
| bore 16, keyway 2.05 (hub wall 2.10 mm, 0.05 mm under the keyway) | 4.57 / 11.85 vs 7.65 | 162.5 / 206.1 | 14.87-33.54 / 13.30-34.04 | 79.7 / 63.7 | 7.9 / 182.6 | slips (inner hub) |

Notes on the table:
- **Smoothing cuts both ways.** The same smoothing is taken off the maximum, so the 5 µm assumption, conservative for
  holding, under-states the bore stress when the real finish smooths less: at smoothing 0 the cup's 15-22 band is
  over its limit (401.9 MPa). This is synthesis decision 13 as recommended; the physics review decides whether the
  stress side should take a smaller smoothing (section 11, R-1).
- **NaN is not carried by min or max.** `py_min(a, b)` is `b` if `b < a`, else `a` (`compat.rs:25-27`), so it
  carries a NaN only in its first argument: `py_min(100, NaN)` is 100, `py_min(NaN, 120)` is NaN, and
  `py_min(py_min(100, 120), NaN)` is 100 (C, `out.txt` 11.8). A prototype that took min(hub limit, adhesive limit)
  passed the "no adhesive" design until the unknown test read every value directly (section 4.4). `f64::min` drops a
  NaN in either argument: `f64::min(NaN, 120)` is 120, so a NaN hub input would show an allowed 120 °C and a fittable
  21.29 µm (C, `out.txt` 10.4). So the allowed temperature is NaN when any of its three values is (section 4.4,
  review 2 SF-3, R-15), whatever the order of the arguments.
- The 304 and aluminium hubs have no working interference at Ø10; so says the Problem's empty-window sentence.
- **The onset (R-15) moves no probe.** The defaults and all five probes read as before, the Y30 ferrite probe and
  `breaking(ferrite)` included (onset +∞, so the cap is +∞ and drops out of the min; C, `out.txt` 11.1, 11.5). At
  each of them the inner ring governs, so its own onset and the governing ring's are equal bit for bit (13.1). Of the
  numeric inputs, four move the inner ring's onset at their slider ends at the defaults: the single-ring reverse
  field (a stored 3D input), the knee fraction and the Br coefficient (assumptions) and `coupling.mu0` (13.6, as
  11.7 found for the governing onset); the inner magnet part and grade move it too. No geometry input moves it, at the
  defaults or on manual N42SH dimensions (11.7, 11.15).
- The cup starts to slip at a magnet length of 26.5 mm (cold-high 8.46 N·m): the hub grows with the magnets, the
  boss does not.

### 3.7 The shaft's shrink at fitting (C, Ø10, diametral)

| shaft at | NIST-based | a constant 11.5e-6 would give |
|---|---|---|
| +40 °C | −2.30 µm (it grows) | −2.30 |
| −20 °C (cold spray, the gearbox slider's bottom) | 4.55 µm | 4.60 |
| −40 °C | 6.66 µm | 6.90 |
| −78.5 °C (dry ice) | 10.35 µm | 11.33 |
| −196 °C (liquid nitrogen) | **18.99 µm** | 24.84 (+31 %) |

The shrink falls strictly as the shaft warms at every 1 °C step from −196 to 40 °C, and is continuous at 265 K (C).

## 4. Engine

### 4.1 New module `M/src/engine/shaft_joint.rs` (Rust-only, like `gearbox.rs`)

A module doc in the voice of `gearbox.rs` states the rule (decision 2A), the two joints and their shafts, the order of
causes, the temperature model, the fitting order (7A, Process) and what is not checked (section 10). It says what
section 1 says of cold-high and no more (review 2, MF-1): cold-high "can reach" each joint; never "strongest", "at
most" or "up to" said of it, though `gearbox.rs`'s own doc says "the strongest pull-out expected" under its
temperature condition.

**Inputs**, `ShaftJointInputs`, every one `param_rust_only` (no cell), in this schema order:

| path | label (draft) | unit | default | slider (min, max, step) | assumption | help (draft, ≤ 30 words, plain) |
|---|---|---|---|---|---|---|
| `shaft_joint.hub_interference_min_um` | Inner hub interference, minimum | µm | 13 | 0, 40, 0.5 | no | The smallest measured diametral interference of the inner hub on the wheel shaft: shaft diameter less bore, both measured. Holding is checked at it. |
| `shaft_joint.hub_interference_max_um` | Inner hub interference, maximum | µm | 18 | 0, 40, 0.5 | no | The largest measured diametral interference of the inner hub. Its bore stress and fitting temperature are checked at it. |
| `shaft_joint.cup_interference_min_um` | Cup interference, minimum | µm | 15 | 0, 40, 0.5 | no | The smallest measured diametral interference of the cup's boss on the gearbox shaft. Holding is checked at it. |
| `shaft_joint.cup_interference_max_um` | Cup interference, maximum | µm | 22 | 0, 40, 0.5 | no | The largest measured diametral interference of the cup. Its bore stress and fitting temperature are checked at it. |
| `shaft_joint.safety_factor` | Shaft joint safety factor | - | 2.0 | 1.0, 4.0, 0.1 | no | The fit alone must hold cold-high times this, and the key alone is checked at the same torque. The torque reverses many times in each slip. |
| `shaft_joint.friction` | Shrink-fit friction coefficient | - | 0.14 | 0.05, 0.30, 0.01 | **yes** | Steel on steel, dry and degreased. Holding torque scales with it; oil or a coating in the bore lowers it. |
| `shaft_joint.smoothing_um` | Surface smoothing of the fit | µm | 5.0 | 0, 15, 0.1 | **yes** | Interference lost as surface peaks flatten when fitted. It lowers both the grip and the bore stress; a ground shaft and honed bore lose less. |
| `shaft_joint.rotor_over_shaft_C` | Rotor hotter than its shaft, running | °C | 20 | 0, 60, 1 | **yes** | How much hotter each rotor may run than its shaft from slip heating. A warmer rotor's bore grows, so the grip falls. |
| `shaft_joint.key_width_mm` | Shaft key width | mm | 3.0 | 1.0, 6.0, 0.5 | no | Width of the parallel key in both joints (DIN 6885-1 gives 3 mm for a 10 mm shaft). Its slot takes this much of the fit's contact. |
| `shaft_joint.key_contact_mm` | Shaft key contact height | mm | 1.2 | 0.5, 3.0, 0.1 | no | Height of the key face bearing on the inner hub and the cup, capped at the keyway depth (DIN 6885-1 gives 1.2 mm for 10 mm). |
| `shaft_joint.hub_fit_max_C` | Hottest the bonded inner hub may be for fitting | °C | 100 | 20, **130**, 1 | no | Its magnets are bonded before fitting. The check caps it at the adhesive's design limit and at the magnets' single-ring onset less 10 °C. (24 words; R-15; "the magnets'" are its own, the inner ring's, under the ring ruling) |
| `shaft_joint.cup_fit_max_C` | Hottest the bare cup may be for fitting | °C | 300 | 20, 400, 5 | no | The cup is fitted before its magnets are bonded. Stay below the steel's and its finish's treatment temperatures, and keep the heat off the gearbox seal. |
| `shaft_joint.wheel_shaft_fit_C` | Wheel shaft temperature at fitting | °C | −196 | −196, 40, 1 | no | A cooled shaft shrinks, so the inner hub needs less heat. Dry ice is about -78 °C; liquid nitrogen is -196 °C. |
| `shaft_joint.output_shaft_fit_C` | Gearbox shaft temperature at fitting | °C | 20 | −20, 40, 1 | no | The gearbox output shaft cannot be immersed. Cold spray can take it to about -20 °C, the slider's bottom; never below the gearbox's rated range. |
| `shaft_joint.joining_clearance_um` | Joining clearance | µm | 10 | 0, 30, 0.5 | **yes** | Clearance each part needs over its shaft to slide on before it grips; one value for both joints. Less needs a guided, single fast push to a stop. |

Notes on the inputs:
- Defaults sit on their slider grids (13, 18, 15, 22 on 0.5; −196 on 1 from −196; 300 on 5 from 20).
- The bonded hub's slider stops at 130 °C (review SF-3): below the inner magnets' single-ring onset less its 10 °C
  margin at the defaults (143.36 − 10 = 133.36 °C, `temperature.demag.inner_onset_single_ring_C`, equal there to the
  governing ring's `temperature.demag.onset_single_ring_C`, E; the margin is the engine's cure
  margin, `CURE_MARGIN_MIN_C`). The check now reads the onset (owner R-15), so the slider top is a convenience, not
  the guard: a typed value above it is accepted with the house's outside-range note (`input_ui.rs:134`) and the onset
  term caps it. With 2214 Hi-Temp (177 °C) the slider top binds at 130 °C, and a typed 150 °C is capped at the onset
  less 10 °C, 133.36 °C (C, `out.txt` 11.10). Reading the onset puts the row in `STORED_3D_ROWS` (section 5.1).
- The key labels say "Shaft key" so they never collide with the paused clamp's "Key width" (no C-3 suffix needed).
- The gearbox shaft's label contains "gearbox" (explicit wins); its path does not. The input filter's "gearbox" tests move
  (section 7.17). The helps quote no number from memory except the bath temperatures (facts) and the DIN 6885-1 row
  (verified in the physics research); the joining-clearance help keeps "about 0.001 times the bore" out until the
  physics review confirms it.
- `.assumption()` on friction, smoothing, the rotor-over-shaft allowance and the joining clearance.

**Results**, `ShaftJointResults`, every one `out_rust_only` (no cell), labels and helps plain (≤ 30 words; drafts, the
G-3 review sets the words; no help says "safe"):

| path | label (draft) | unit | value at the defaults (C) | help (draft) |
|---|---|---|---|---|
| `shaft_joint.required_Nm` | Torque each shaft joint must hold | N·m | 7.6466 | Cold-high (the pull-out at the minimum magnet temperature, with the production allowance) times the safety factor. Each fit alone must hold it; the key alone is checked at it. (29 words; review 2, MF-1: it claims no order of the torques) |
| `shaft_joint.part_modulus_GPa` | Inner hub and cup elastic modulus | GPa | 205 | Of the inner hub and cup material in effect: the back-iron steel with a steel circuit, else the hub body's material. The fit's contact pressure rises with it. |
| `shaft_joint.part_cte_per_C` | Inner hub and cup expansion coefficient | 1/°C | 12.3e-6 | Of the inner hub and cup material in effect. Against the shafts' steel, a higher value loosens the fit hot and a lower one loosens it cold. |
| `shaft_joint.part_yield_MPa` | Inner hub and cup yield strength | MPa | 415 | The library's yield strength of the inner hub and cup material in effect; for 4140, the annealed value, conservative while the condition is open. |
| `shaft_joint.stress_limit_MPa` | Bore stress limit | MPa | 345.83 | The part's yield strength with a margin. At the tightest fit the bore stress must not exceed it, so the part stays elastic and keeps its grip. |
| `shaft_joint.hub_holding_Nm` | Inner hub holding torque at its loosest | N·m | 9.4844 | Torque the inner hub's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft. |
| `shaft_joint.cup_holding_Nm` | Cup holding torque at its loosest | N·m | 16.899 | Torque the cup's fit holds without slipping at its loosest: minimum measured interference less smoothing, at the temperature end that loosens it more, rotor warmer than shaft. |
| `shaft_joint.hub_bore_stress_MPa` | Inner hub bore stress at its tightest | MPa | 241.70 | Von Mises stress at the inner hub's bore at its tightest: maximum measured interference less smoothing, at the temperature end that tightens it more. The keyway corner is not included. |
| `shaft_joint.cup_bore_stress_MPa` | Cup bore stress at its tightest | MPa | 312.53 | Von Mises stress at the cup's bore at its tightest: maximum measured interference less smoothing, at the temperature end that tightens it more. The keyway corner is not included. |
| `shaft_joint.hub_key_pressure_MPa` | Key pressure on the inner hub, key alone | MPa | 127.44 | Pressure on the key's face in the inner hub were the key to carry the torque alone (a backup check). 0 with no keyway; +inf with no bearing area. |
| `shaft_joint.cup_key_pressure_MPa` | Key pressure on the cup, key alone | MPa | 101.95 | Pressure on the key's face in the cup were the key to carry the torque alone (a backup check). 0 with no keyway; +inf with no bearing area. |
| `shaft_joint.hub_needed_um` | Inner hub smallest measured interference that holds | µm | 12.082 | The lowest the band's minimum may be for the inner hub to hold the torque it must: the window's lower end, with smoothing and temperature included. |
| `shaft_joint.cup_needed_um` | Cup smallest measured interference that holds | µm | 11.310 | The lowest the band's minimum may be for the cup to hold the torque it must: the window's lower end, with smoothing and temperature included. |
| `shaft_joint.hub_elastic_max_um` | Inner hub largest measured interference within the stress limit | µm | 23.808 | The highest the band's maximum may be for the inner hub's bore stress to stay within its limit, with smoothing and temperature included. One upper end of the window. |
| `shaft_joint.cup_elastic_max_um` | Cup largest measured interference within the stress limit | µm | 23.863 | The highest the band's maximum may be for the cup's bore stress to stay within its limit, with smoothing and temperature included. One upper end of the window. |
| `shaft_joint.wheel_shaft_shrink_um` | Wheel shaft shrink at its fitting temperature | µm | 18.988 | How much the wheel shaft's diameter shrinks from 20 °C to its fitting temperature, from published low-temperature data for a nickel steel; negative above 20 °C, where it grows. |
| `shaft_joint.output_shaft_shrink_um` | Gearbox shaft shrink at its fitting temperature | µm | 0 | How much the gearbox shaft's diameter shrinks from 20 °C to its fitting temperature, from published low-temperature data for a nickel steel; negative above 20 °C, where it grows. |
| `shaft_joint.hub_fittable_max_um` | Inner hub largest measured interference that can be fitted | µm | 18.828 | The highest the band's maximum may be for the inner hub to be fitted: what its allowed temperature and the wheel shaft's shrink open, less the joining clearance. |
| `shaft_joint.cup_fittable_max_um` | Cup largest measured interference that can be fitted | µm | 24.440 | The highest the band's maximum may be for the cup to be fitted: what its allowed temperature and the gearbox shaft's shrink open, less the joining clearance. |
| `shaft_joint.hub_fit_temp_C` | Inner hub temperature the fit would need | °C | 93.267 | What the inner hub would need on its wheel shaft as set, the joining clearance included. Above the allowed temperature it cannot be fitted this way: never warm it further. |
| `shaft_joint.cup_fit_temp_C` | Cup temperature the fit would need | °C | 280.16 | What the bare cup would need on the gearbox shaft as set, the joining clearance included. Above the allowed temperature it cannot be fitted this way: never warm it further. |
| `shaft_joint.hub_fit_allowed_C` | Inner hub fitting temperature allowed | °C | 100 | The lowest of your limit for the bonded inner hub, the selected adhesive's design limit and the magnets' single-ring onset less 10 °C; not a number if any is. (29 words; R-15) |
| `shaft_joint.cup_fit_allowed_C` | Cup fitting temperature allowed | °C | 300 | Your limit for the bare cup, which is fitted before its magnets are bonded. |
| `shaft_joint.check` | Shaft joints: keyed shrink fits | (text) | "Both shaft joints hold and can be fitted" | Each rotor's shrink fit must hold cold-high times the safety factor when loosest, stay elastic when tightest and be fitted within the allowed temperature; the key alone is a backup. (30 words, the cap) |

For a joint with no wall (section 3.1) its holding, bore stress, needed and elastic rows read NaN; its key pressure,
fittable and fitting-temperature rows keep their values (they do not rest on the wall). The cup's key pressure at the
defaults is 101.95 MPa (C, and review 1); the first draft's 101.96 was a rounding slip.

**Named constants** in `shaft_joint.rs`: `SHAFT_MODULUS_GPA` 205.0, `POISSON` 0.30, `SHAFT_CTE_PER_C` 11.5e-6,
`ROOM_C` 20.0, `YIELD_FACTOR` 1.2, `NIST_FE_2_25NI` (the five coefficients), `NIST_FIT_TOP_K` 265.0, and the joint
names `INNER_HUB` = "inner hub", `CUP` = "cup".

**A pure function per joint**, `joint(…) -> JointTerms`, evaluated twice by `compute(inputs, links) ->
ShaftJointResults`. `links` carries the engine values the check reads (a struct, as `TemperatureLinks` is):
cold-high, bore, keyway depth, hub length in effect, hub apothem and bondline, boss length, web, boss diameter, minimum
magnet temperature, the selected adhesive's design limit, the inner ring's own single-ring onset
(`temperature.demag.inner_onset_single_ring_C`, owner R-15 and its ring ruling; below), and the part material in
effect (modulus, expansion, yield).

**The allowed hub temperature (owner R-15)** comes from one pure helper, `hub_fit_allowed(input_C, adhesive_C,
onset_C) -> (f64, Option<HubFitLimit>)`, with `enum HubFitLimit { Yours, Adhesive, MagnetOnset }`:
- it reads the input `shaft_joint.hub_fit_max_C`, the link values `temperature.adhesive.design_limit_C` and
  `temperature.demag.inner_onset_single_ring_C`, and the constant `temperature::CURE_MARGIN_MIN_C`;
- NaN and `None` when any of the three values is NaN, tested first: `py_min` carries a NaN only in its first argument
  and `f64::min` in neither (section 3.6), so no order of the arguments makes a bare min safe;
- otherwise `py_min(py_min(input_C, adhesive_C), onset_C − CURE_MARGIN_MIN_C)` and the limit that binds: the first of
  `Yours`, `Adhesive`, `MagnetOnset` whose value equals the result (`py_min` returns one of its arguments bit for bit,
  so the equality is exact; ties name the earlier: an input of 120 °C with AA 326 names your limit, C 11.10);
- an onset of +∞ (ferrite) gives a cap of +∞, which never binds; an onset below 30 °C gives an allowed temperature
  below 20 °C, which the fitting formula takes as it is (C: the single-ring field's slider top gives −12.45 °C).
`compute` writes the value to `shaft_joint.hub_fit_allowed_C`; the guide calls the same helper, with the same
three values, to name the limit (section 5.2), so the number and its name come from one place (DRY).

**`M/src/engine/temperature.rs`: the inner ring's own single-ring onset (owner R-15 ring).** The governing ring's
`temperature.demag.onset_single_ring_C` is the ring with the lower magnet limit (the inner on a tie; E20,
`temperature.rs:969-972`), not necessarily the inner ring the bonded hub carries, so the check reads a new result:
- `inner_onset_single_ring_C: f64 => out_rust_only("°C", "Inner ring: single-ring onset", "The inner ring's own
  onset on its carrier with no outer ring around it, whichever ring governs. The bonded inner hub's allowed fitting
  temperature stays below it.")` in `DemagResults`, right after `inner_magnet_limit_C` (label and help drafts, 4 and
  27 words, no plain-words violation, C `out.txt` 13.8; the G-3 review sets the words).
- Its value is `inner.onsets[3]`: `ring_demag` builds a ring's four onsets (`temperature.rs:816`), `compute` builds the
  inner ring's block at `:931`, and the per-ring results read that block at `:1012-1016` (`inner_magnet_limit_C` at
  `:1015`). A ring's block reads only its own Br, Br coefficient, rating and grade and
  the shared demagnetization inputs, so the value is the inner ring's whatever the outer ring is. It equals
  `onset_single_ring_C` bit for bit wherever the inner ring governs (`hot` is `inner` there) and always with E20 off
  (the workbook checks the inner ring only); with E20, an inner ring with a positive beta (hard ferrite) reads +∞, as
  its onsets do.
- Not Python-ported, no cell; no outer twin (nothing reads one; a per-ring onset for the cure margin is BL-072's).

**`M/src/engine/material_library.rs`:** a helper `PartProperties::hub(&self) -> HubMaterial { props: EngineProps,
yield_MPa: f64 }`: the steel values when the circuit in effect is steel (`backiron == 1`), else the body's; the yield of
the back-iron pick's library record when the circuit is steel or the pick is non-magnetic, else the library's 6061-T6
(241 MPa, the workbook's aluminium hub; one aluminium, one yield: synthesis correction 6). The yield follows the
material in effect, not the pick (review SF-16): a pick outside the choices on the steel circuit gives NaN (as
`resolve` already gives NaN steel values, `material_library.rs:864-910`), so the check reads unknown; with no back iron
(`coupling.backiron` 0) `resolve` gives the workbook's aluminium body whatever the pick, so the helper gives 6061-T6's
241 MPa there too, one outside the choices included (C: E 68.9 GPa, α 23.6e-6, "slips (inner hub)", as the aluminium
row of section 3.6). `api.rs:349-353` (the warnings' hub expansion) reads it too (DRY). The `yield_MPa` doc at `:91`
("not read by the engine") becomes "read by the shaft joint check".

**`M/src/engine/api.rs`:** `shaft_joint: ShaftJointInputs` after `clamps` in `DesignInputs` (the first top-level
Rust-only input group); `shaft_joint: ShaftJointResults` after `gearbox` in `DesignResults`; computed after
`gearbox::compute` (`:359-360`), when `mdr`, `axial`, `temp` and `parts` all exist; the struct init. `M/src/engine/mod.rs`:
`pub mod shaft_joint;`.

### 4.2 Verdict texts (named constants, plain, glyph-safe, no internal paths, no "clamp")

| constant | text | level |
|---|---|---|
| `SHAFT_JOINTS_PASS` | "Both shaft joints hold and can be fitted" | Good |
| `SHAFT_JOINT_UNKNOWN` (prefix) | "Shaft joint check unknown" | Caution, by prefix |
| `UNKNOWN_NOT_A_NUMBER` | `SHAFT_JOINT_UNKNOWN` + ": a value is not a number" | Caution |
| `UNKNOWN_NO_WALL` (prefix) | `SHAFT_JOINT_UNKNOWN` + ": no wall around the bore" + " ({joint})" (Q ≥ 1, or a keyway through the wall) | Caution |
| `UNKNOWN_BAND_REVERSED` (prefix) | `SHAFT_JOINT_UNKNOWN` + ": the minimum interference is above the maximum" + " ({joint})" | Caution |
| `FIT_SLIPS` | "The fit slips" + " ({joint}): holds {q} N·m against {r} N·m" | Bad, by prefix |
| `BORE_OVER_ITS_LIMIT` | "Bore stress above its limit" + " ({joint}): {q} MPa against {r} MPa" | Bad, by prefix |
| `CANNOT_BE_FITTED` | "Cannot be fitted at the allowed temperature" + " ({joint}): needs {q} °C against {r} °C" | Bad, by prefix |
| `KEY_OVER_ITS_LIMIT` | "Key backup above its limit" + " ({joint}): {q} MPa against {r} MPa" | Bad, by prefix |
| `KEY_HAS_NO_BEARING_AREA` | "The key has no bearing area" + " ({joint})" | Bad, by prefix |

**Never-equal numbers (the BL-059 rounding, `gearbox.rs:49-58`).** The limit is quoted half-up to its decimals with
`fmt_fixed`; the failing value is pushed one last digit past the limit as quoted:
- above a limit (stress and key to 0.1 MPa, temperature to 0.1 °C): `fmt_fixed(py_max(x, r_quoted + step), d)`;
- below the need (holding, to 0.01 N·m): `fmt_fixed(py_max(0, py_min(x, r_quoted − 0.01)), 2)`; where the need itself
  quotes as 0.00 N·m (a need under 0.005 N·m, reachable only near the end-effect limit), the holding is quoted 0.00 and
  the need 0.01, so the pair still differs.
- One shared private helper pair (`quoted_above`, `quoted_below`) serves all four texts (DRY).
- Examples (C): "The fit slips (inner hub): holds 3.48 N·m against 7.65 N·m" (hub minimum 10); "Bore stress above its
  limit (inner hub): 385.1 MPa against 345.8 MPa" (hub maximum 26); "Cannot be fitted at the allowed temperature (inner
  hub): needs 247.6 °C against 100.0 °C" (wheel shaft 20 °C); "Cannot be fitted at the allowed temperature (inner hub):
  needs 93.3 °C against 93.0 °C" (hub allowed 93 °C); "Cannot be fitted at the allowed temperature (inner hub): needs
  93.3 °C against 59.8 °C" (`breaking(n42_parts)`: the onset less 10 °C binds, R-15); "Key backup above its limit
  (inner hub): 509.8 MPa against
  345.8 MPa" (keyway depth 0.3 mm, on its slider's 0.05 mm grid: the contact height is capped at 0.3 mm; the cup's key
  reads 407.8 MPa, both fits hold, stress and fitting pass, so the key is the badge's first cause; review 2, SF-9; C,
  `out.txt` 10.8). At an axial override of 40 mm the cup's key (346.6 MPa) only follows the cup's slip in the Problem.

### 4.3 Order of causes

A joint is **determinate** when it has a wall (section 3.1) and its band is not reversed (minimum ≤ maximum). Only a
determinate joint is judged: an indeterminate joint's terms turn nothing red and offer no fix, whatever they read. The
badge shows the first that applies (review SF-8: a determinate joint's red cause outranks the other joint's unknown):
1. unknown: a value is not a number (section 4.4), including a determinate joint whose compared term is NaN from finite
   inputs. Caution. It covers both joints.
2. the fit slips: holding < required (strict; equality passes).
   Every comparison of causes 2 to 5 runs after cause 1, so none is reached with a NaN term; each is also written so
   that a NaN fails it (`!(holding >= required)`, `!(stress <= limit)`), never passed by a false comparison (review 2,
   SF-4). Bore typed 0: the pressure is +∞ and the holding ∞·0 = NaN, which a bare "holding < required" would pass;
   it reads "unknown: a value is not a number" (C, `out.txt` 10.5).
3. the bore stress is above its limit: stress > limit (strict).
4. it cannot be fitted: needed fitting temperature > allowed (strict).
5. the key backup: no bearing area (l ≤ 0 or k ≤ 0); else key pressure > limit (strict). Not with a keyway depth of 0.
6. unknown, per joint: no wall around the bore (Q ≥ 1, reachable on the sliders: bore 16 mm, boss Ø12 mm; or a keyway
   through the wall: bore 16 mm with a keyway of 2.1 mm or more leaves the inner hub's 2.10 mm wall cut through); then a
   reversed band. Caution, shown only when no determinate joint fails.

Within a cause the badge names the first failing joint in the order inner hub, cup. +∞ fails where it is compared (a
cold-high of +∞ slips). The terms are computed even when the band is reversed or the check is unknown, so the fix
tests see a monotone margin (synthesis 12).

**What each unknown shows (Problem and fixes).**
- Not a number: the Problem is the not-a-number sentence alone; no fix.
- No wall or a reversed band on one joint: the Problem lists the determinate joint's failing causes first (cause order,
  each with its window sentence) and then the indeterminate joint's own sentence; the fixes are the determinate joint's
  failing terms' fixes only. With both joints indeterminate, both sentences, inner hub first, and no fix.
- So the status line counts a red joint as failing even while the other joint is unknown (C: hub minimum 20 over its
  maximum 18 with the cup minimum at 10 reads "The fit slips (cup): holds 4.36 N·m against 7.65 N·m").

### 4.4 The unknown test

The check reads unknown when **any** of its 15 inputs, or any of the 15 engine values it reads (cold-high, bore,
keyway depth, hub length, hub apothem, bondline, boss length, web, boss diameter, minimum temperature, the adhesive's
design limit, the inner ring's own single-ring onset (R-15), the part's modulus, expansion and yield), is NaN. The
test reads the values themselves, before any min or max: `py_min`/`py_max` (and `f64::min`/`max`) drop a NaN in one
argument, so a NaN adhesive limit (adhesive code 0, `NO_ADHESIVE`) would otherwise give the hub an allowed temperature
of 100 °C and a pass (shown in the scratch crate). The allowed hub temperature (`shaft_joint.hub_fit_allowed_C`, the
helper of section 4.1) is `NaN` when **any** of the hub input, the adhesive limit and the onset is NaN, and the
lowest of the three otherwise (the third being the onset less `CURE_MARGIN_MIN_C`; review 5, SF-3). Review 2, SF-3:
`f64::min(NaN, 120)` is 120, so a NaN hub input would otherwise show 120 °C allowed and a fittable 21.29 µm while the
verdict reads unknown. A NaN onset is reached through the struct (7.2): a single-ring reverse field set to NaN gives
a NaN inner onset, a NaN allowed temperature and "Shaft joint check unknown: a value is not a number" (C, `out.txt`
11.8, 13.1). A design file or share link cannot carry a
NaN number (`session.rs:219-227` reads a number input only from a JSON number; review 4, SF-1). The unknown test reads
the value all the same. An onset of +∞ (ferrite) is not NaN and reads as a cap that never binds.

The test also covers the terms a **determinate** joint compares (review 2, SF-4): its holding, the required torque,
its bore stress, the stress limit, its needed and allowed fitting temperatures and its key pressure. Any of them NaN
reads "unknown: a value is not a number", even with every input and engine value finite (bore typed 0). An
indeterminate joint's terms are left out of this test: a joint with no wall reads NaN on its Lamé rows by design
(section 3.1) and gives its own unknown (cause 6).

### 4.5 Records, assumptions and scope

- `M/src/engine/explain/records/shaft_joint.rs` (new): one evaluable record per numeric result (23), transcribed from
  `shaft_joint::compute` with every correction on: the part-material rows as `cases` over the back-iron pick and the
  circuit (`tables.rs` gains `modulus_GPa`, `cte_per_C` and `yield_MPa` for the `back_iron` table, and the default
  pick reads the Materials inputs, as the engine does); the shrink as `cases` on the 265 K boundary with the NIST
  polynomial; the four Lamé rows (holding, bore stress, needed, elastic) with a no-wall case that gives NaN; the
  allowed hub temperature as `min(min({shaft_joint.hub_fit_max_C}, {temperature.adhesive.design_limit_C}),
  {temperature.demag.inner_onset_single_ring_C} - 10)` (R-15; the 10 is `CURE_MARGIN_MIN_C`, named in the record's
  comment), with a NaN arm if the markup's `min` drops a NaN where the engine gives one (the implementer checks
  `min`'s NaN rule in the evaluator first). `records/mod.rs` registers it (as the house does, one file per batch, its chain flipped to `Explained`);
  `symbols.rs` gives the 15 inputs symbols. The drift guard then holds every record to the engine.
- **The inner ring's onset record** (owner R-15 ring), in `M/src/engine/explain/records/demagnetization.rs` right
  after `temperature.demag.inner_magnet_limit_C`'s, symbol `ϑ_{sr,i}`, transcribed from `ring_demag`:
  `cases({temperature.demag.inner_beta_per_C} > 0 => inf; else => 20 + frac([H_k] -
  {temperature.demag.h_rev_single_ring_kA_m}, [H_k] · abs({temperature.demag.inner_beta_per_C}) -
  {temperature.demag.h_rev_single_ring_kA_m} · abs({model.inner_alpha_br_per_C})) - [Δϑ_{cal,i}])`, with the `where`
  terms `[H_k]`, `[H_{ref,i}]` and `[Δϑ_{cal,i}]` of `inner_magnet_limit_C`'s record, `.corrected(&[E20])`. It is
  that record with the single-ring field in place of the skipping field, no design margin and +∞ (not the rating)
  for a positive beta. The `where` terms repeat `inner_magnet_limit_C`'s, as the inner and outer magnet-limit records
  already repeat each other (a per-ring calibration-offset result would let them share it: one more result and an
  edit of a reviewed record; not done here). Its terms are those of `inner_magnet_limit_C`'s record, so it drills down
  to inputs, and the knee fraction and the Br coefficient move it at the defaults (C 13.6): no `CANCELLATIONS` entry.
  The `shaft_joint` note owns it (section 6.1).
- **The scope chains** (review MF-3; owner R-15 ring). `scope.rs` gains an explained chain `shaft_joint` (the 23
  numeric paths) **after `geometry` and before `dashboard`** (the scope's "start here" order, then the dashboard
  paths), and the `demagnetization` chain gains `temperature.demag.inner_onset_single_ring_C` right after
  `temperature.demag.onset_single_ring_C` (18 -> 19 paths): the union grows 169 -> 193 (C 13.9;
  `tests/explain.rs:1526`; the module doc's "169 paths", `scope.rs:12`, and its chain count). The results
  table follows (section 5.4): `CHAIN_LABELS` gains the chain's heading, since
  `the_headline_comes_first_then_the_chains_then_the_other_results` and
  `every_chain_path_names_results_and_every_package_has_a_heading` (`result_groups.rs:347-357`, `:330-345`) hold every
  scope chain but `dashboard` to a `CHAIN_LABELS` entry in the scope's order. `shaft_joint.check` is a dashboard row,
  so it sits in the headline group; no `CHAIN_PREFIXES` entry is needed (every numeric result is a chain path).
- `M/src/engine/assumptions.rs`: four rows (drafts; the G-3 review sets the words). `tests/assumptions.rs:127-131`
  requires a Rust-only assumption's source to contain "Addendum A3". The rule widens (review 2, MF-3) to accept a
  Rust-only source that contains "Addendum A3" **or** "BL-064 shaft joint spec", the words all four drafted sources
  below carry, instead of a false citation; the test's comment (`:127-128`) says so. The first row (`fit_friction`)
  keeps the spec's file path as well. The G-3 review does not cover sources (section 6.3), so the four sources go into
  code as drafted here, and each must keep those words.

  | id | label | paths | rationale (draft) | source (draft) |
  |---|---|---|---|---|
  | `fit_friction` | Shrink-fit friction coefficient | `shaft_joint.friction` | The friction between each rotor's bore and its shaft that turns the fit's grip into holding torque. 0.14 is for dry, degreased steel on steel; oil, a coating or a finish in the bore lowers it, and the holding torque falls in proportion. | BL-064 shaft joint spec (`docs/superpowers/specs/2026-10-17-bl064-shaft-joint-design.md`); dry steel on steel: Gear Technology, Nov/Dec 2017 (0.145) and ANSI/AGMA 9003 (0.15). |
  | `fit_smoothing` | Surface smoothing of the fit | `shaft_joint.smoothing_um` | Interference lost as the surface peaks of bore and shaft flatten when the fit is made, taken off both ends of the measured band. 5 µm is conservative for holding with ground and honed surfaces; a finer finish loses less, which raises the grip and the bore stress together. | BL-064 shaft joint spec; DIN 7190-1:2017 (smoothing 0.4·Rz per surface). |
  | `rotor_over_shaft` | Rotor hotter than its shaft, running | `shaft_joint.rotor_over_shaft_C` | Slip heating warms the rotors from the gap, so each rotor can run warmer than the shaft it sits on. Its bore then grows more than the shaft, and the grip at the hot end falls. 20 °C is a placeholder until a measurement or a thermal model gives the difference. | BL-064 shaft joint spec (the hot end of the loosest fit, section 3.3). |
  | `joining_clearance` | Joining clearance | `shaft_joint.joining_clearance_um` | The gap each part needs over its shaft, after warming and cooling, to slide to its stop before the gap closes; one value for both joints. Too little and the part can seize half-way. 10 µm is an assumed value for a 10 mm bore; less needs a guided, single fast push to a stop. | BL-064 shaft joint spec; DIN 7190-1:2017 §8.2 (the joining clearance enters the joining temperature). |
- No change to any Python-ported input, result, label, help, default, range or cell; to `tests/data/python_schema.json`
  or the differential files; or to `reference/magcoupling-py`. `tests/data/input_schema.json` is re-blessed
  (`MAGCOUPLING_BLESS=1 cargo test --test schema`).
- `M/src/engine/clamps.rs:38-48`: the `PAUSED` doc no longer says the keyway drawing is hidden (the key is drawn in
  both states) and records that un-pausing would show the clamp beside the shrink-fit check (two joint methods).

## 5. GUI

### 5.1 Dashboard (`M/src/gui/dashboard.rs`)

- `DASHBOARD` 18 -> 19: `("shaft_joint.check", Some("shaft_joint.check"))` at **index 15**, after the hidden clamp row
  and before the gearbox row, so `paths[15..]` = [shaft joint, gearbox, space claim, end effect]. The GUI-only Magnets
  fit row stays last.
- `CHECKS` 31 -> 32, `"shaft_joint.check"` appended last (schema order: the group follows `gearbox`).
- `verdict_level`: Good for `SHAFT_JOINTS_PASS`; Caution for the `SHAFT_JOINT_UNKNOWN` prefix; Bad for the five failing
  prefixes. Every text the engine produces is classified (an unclassified text reads green,
  `check_level(..).unwrap_or(Good)`).
- `END_EFFECT_ROWS` 8 -> 9: the check reads cold-high, which is −0.1726 N·m at `c_end` 1.5, where the verdict would
  read a pass. The row greys, uncounted, with no badge in the results table and out of the failing list.
  The whole row greys, so a stress or fitting failure, which does not read cold-high, is hidden there too (review 2,
  SF-13). That trade is deliberate: one row cannot grey in part, and at every such design the end-effect check fails
  and the banner "End-effect model out of range" says the model does not hold there.
- **`STORED_3D_ROWS` 3 -> 4** (owner R-15): `"shaft_joint.check"` is appended. The hot end still reads no stored 3D
  field (the adhesive's design limit); the onset term does: `temperature.demag.inner_onset_single_ring_C` (the inner
  ring's own onset, owner R-15 ring) follows the stored single-ring reverse field
  `temperature.demag.h_rev_single_ring_kA_m`, one of `STORED_3D_INPUTS`, as every ring's onsets do.
  - **The row carries the label.** Its hover gains "Fixed 3D fields from the example design: …" (`result_tooltip`
    with `stored_3d`), and the XLSX Summary's note column reads `STORED_3D_LABEL` for it (`GREYED_NOTE` while greyed:
    greying wins, `spreadsheet.rs:341-347`).
  - **The drawn label moves.** `dashboard_ui` draws `STORED_3D_LABEL` once, after the last stored-3D row
    (`rposition`, `dashboard.rs:710`). That is now the shaft-joint row, so the label moves from under the temperature
    verdict's guide block to under the shaft-joint row's. While the clamp is paused (the default) the hidden clamp row
    leaves the temperature rows and the joint row adjacent, so the label closes one run of stored-3D rows. Unpaused, the
    clamp row sits inside that run without reading the fields. This spec keeps the one label (count 1 in both panel
    tests); each row's hover says exactly which rows read the fields. The doc comment on `dashboard_ui` ("the
    stored-3D label after the last temperature row") becomes "after the last stored-3D row".
  - **`STORED_3D_NOTE` gains a sentence**, since the joint row's hover shows it and it names only the temperature checks
    (draft; the G-3 review sets the words): "The temperature checks use the demagnetization reverse fields and the
    slip-heating fields, computed in 3D for the example design. The bonded inner hub's fitting limit on the shaft
    joints row uses one of the reverse fields too. Those fields stay fixed when you change the geometry, magnet count
    or grade; for another design, enter its own 3D fields to check the limits." It keeps what the tests pin
    (`dashboard.rs:1517-1574`): plain words, "example design", no "live", "stay fixed", and the one sentence that names
    the slip-heating fields says no "limit" (the new sentence names the limit, not the slip-heating fields).
  - The doc comments that say "the temperature rows" or "the headline rows" of the stored fields (`dashboard.rs:14-17`,
    `:205-207`, and the `STORED_3D_INPUTS` doc at `:220-223`, "[`STORED_3D_ROWS`] are the headline rows that move with
    them"; review 4, SF-4) name the shaft-joint row too.
  - The membership test holds (C, `out.txt` 11.6, re-run with the inner ring's own onset in 13.6). `dashboard_moved_by`
    over the 14 stored 3D inputs at their slider ends at back iron 1 and 0 moves the joint row at back iron 1 with the
    single-ring field at its slider's top, 1500 kA/m (inner onset −2.45 °C: "needs 93.3 °C against −12.4 °C"). At back
    iron 0 the aluminium hub slips at both ends,
    so the row does not move there. No other stored input moves it, the slip-heating ones included, so
    `the_stored_3d_note_says_the_slip_fields_feed_the_checks_not_the_limits` (its `assert_eq` on the slip fields) passes
    unchanged. At the defaults the joint first fails at a single-ring field of 862 kA/m (onset 103.20 °C).

### 5.2 Guide (`M/src/gui/guidance.rs`)

`const SHAFT_JOINT: &str = "shaft_joint.check"`; `SHAFT_JOINT_GUIDE` in `CHECK_GUIDES` (9 -> 10) between `CLAMP_GUIDE`
and `GEARBOX_GUIDE`; counted while paused (not under `SECTION`); its `ROW_LEARN` entry; `Learn::Note("shaft_joint")`.
Numbers in the house readout format (`result_text`, `input_text`, 4 significant digits).

- **Label:** "Shaft joints" (distinct; never "Shaft clamp").
- **Problem** (`shaft_joint_problem`; drafts, final words set by the G-3 review; one sentence each, in the badge's order
  of section 4.3, inner hub first; `Some` exactly when the level is Bad or Caution; what each unknown shows is in
  section 4.3):
  - unknown, not a number: "The shaft joint check cannot be made: a value it reads is not a number."
  - unknown, no wall, Q ≥ 1: cup "The cup's bore, {coupling.bore_mm}, is as wide as its boss, {metal.boss_od_mm}, so
    it leaves no wall to grip the gearbox shaft." Inner hub (only a typed value reaches it: at the apothem slider's
    bottom, 9 mm, the flats stay outside the 16 mm bore slider top) "The inner hub's bore, {coupling.bore_mm}, reaches
    the flats its magnets are bonded on, so it leaves no wall to grip the wheel shaft."
  - unknown, no wall, a keyway through the wall (either joint): "The {joint}'s keyway, {coupling.keyway_depth_mm} deep,
    cuts through the wall around its {coupling.bore_mm} bore, so the fit has no closed ring to grip the {wheel |
    gearbox} shaft."
  - unknown, reversed: "The {joint}'s minimum interference, {min}, is above its maximum, {max}: enter the band's
    smaller value as the minimum."
  - slips: "At its loosest the {joint} holds {holding}, less than the {shaft_joint.required_Nm} it must hold: cold-high,
    {metal.torque_cold_high_Nm}, times the safety factor of {shaft_joint.safety_factor}."
  - stress: "At its tightest the {joint}'s bore stress is {stress}, above the {shaft_joint.stress_limit_MPa} limit
    ({shaft_joint.part_yield_MPa} yield with a margin)." The words "yield divided by 1.2" wait for the physics review.
  - fitting, naming the shaft (review SF-12): "Fitting the inner hub would need it at {shaft_joint.hub_fit_temp_C} with
    the wheel shaft at {shaft_joint.wheel_shaft_fit_C}, above the {shaft_joint.hub_fit_allowed_C} allowed: {source}."
    and "Fitting the cup would need it at {shaft_joint.cup_fit_temp_C} with the gearbox shaft at
    {shaft_joint.output_shaft_fit_C}, above the {shaft_joint.cup_fit_allowed_C} allowed: your limit for the bare cup."
    The hub's `{source}` names the limit that binds (owner R-15), as the helper of section 4.1 returns it, each with its
    live number:
    - `Yours`: "your limit for the bonded inner hub, {shaft_joint.hub_fit_max_C}" (review 2, SF-6b: the joint's name,
      as the `hub_fit_allowed_C` help has it);
    - `Adhesive`: "the selected adhesive's design limit, {temperature.adhesive.design_limit_C}";
    - `MagnetOnset`: "the magnets' single-ring onset, {temperature.demag.inner_onset_single_ring_C}, less
      {CURE_MARGIN_MIN_C}", the margin written from the constant. The placeholder is the inner ring's own onset, the
      one the cap reads (owner R-15 ring; review 5, SF-1), so in a sentence about the inner hub "the magnets'" are the
      hub's own, and with mixed rings the quoted onset and margin still subtract to the allowed temperature: inner
      N42M with outer N35 reads "above the 81.35 °C allowed: the magnets' single-ring onset, 91.35 °C, less 10.00 °C",
      where the governing ring's placeholder would have quoted the outer N35's 59.93 °C (C, `out.txt` 13.5). The onset
      and the margin are each quoted with as
      many decimals as the allowed temperature shows, so the three numbers subtract (the mirror of
      `cure_margin_texts`, `guidance.rs:1373-1405`, which writes the margin with the onset's decimals,
      `fixed(CURE_MARGIN_MIN_C, …)` at `:1399`; review 4, SF-5). Where the allowed temperature's text has an exponent
      or is not finite, the onset reads as any result does and the margin with `number_text` ("10.00 °C"), as
      `cure_margin_texts` falls back. At 59.83 and 88.80 °C allowed the margin reads "10.00 °C"; with 2214 Hi-Temp,
      the hub input typed 150 °C and a 26 µm hub maximum the source reads "above the 133.4 °C allowed: the magnets'
      single-ring onset, 143.4 °C, less 10.0 °C" (C, `out.txt` 12.6; the draft before review 4 read "less 10.00 °C" there).
    The first two quote a number equal to the allowed one; that is deliberate: each form says which limit it is and
    reads its own path, and a test can hold each to its source. "Would need" (review MF-6): the number is what the fit
    would take, never a temperature to use. It never says "heat" alone: a needed temperature below 20 °C reads the
    same sentence (it passes on the sliders).
  - key: "Alone, the key would bear {key pressure} on the {joint}, above the {limit} limit." No bearing length, inner
    hub: "The key has no bearing area in the inner hub: the hub, {housing.hub_length_mm} long, is no longer than the
    key is wide, {shaft_joint.key_width_mm}." Cup: "The key has no bearing area in the cup: its boss and web,
    {metal.boss_length_mm} and {metal.web_mm} long, together are no longer than the key is wide,
    {shaft_joint.key_width_mm}." No contact height (k ≤ 0, a typed value, zero or negative): "The key has no bearing
    area in the {joint}: its contact height, {shaft_joint.key_contact_mm}, is not above zero." (review 2, SF-5a). Where
    a joint has neither (l ≤ 0 and k ≤ 0), the contact-height sentence shows alone (review 2, SF-5b): no length gives a
    key bearing area without contact, and the contact fix is the one that moves the key margin there (section 5.2,
    "Three fixes held back by rule"); once the contact is above zero the length sentence follows if l ≤ 0.
  - window, once per joint failing a slip, stress or fitting term, when it is not empty: "On the {joint}, a measured
    interference from {needed} to {upper} would hold and {stay within the stress limit | fit at the allowed
    temperature} (its band is {min} to {max})." `upper` is the smaller of the elastic and fittable edges, computed
    from the two results (`number_text`), and the clause names the edge that binds. Neither window sentence shows where
    the needed value is not finite (a typed key width of π·d or more leaves no contact arc, section 3.1): the slip
    sentence's "holds 0 N·m" says it, and "at least inf µm" would not.
  - empty window (BL-059 decision-4 style), in place of the window sentence: "No measured interference works on the
    {joint} here: holding needs at least {needed}, but {the bore stays within its limit only up to {elastic}}{ and }{it
    can be fitted only up to {fittable}}." Each upper edge below the needed value is named; a fittable edge at or below
    zero reads "the allowed temperatures do not even open its {shaft_joint.joining_clearance_um} joining clearance".
  - **Window ends quoted inward** (review SF-11, as BL-059's band is quoted): in both window sentences the needed value
    is rounded up at its last shown digit (`quoted_load`'s rounding) and each upper edge rounded down (`quoted_limit`'s),
    so a band typed at the quoted ends passes. Where a window is narrower than the readouts' last digit, so that the
    two rounded ends would read equal or crossed, both ends take more decimals until they read apart in order, as
    `band_quotes` does (BL-059 NB-1).
  - Never-equal quoting in the Problem (BL-059 MF-1): a holding read equal to the need is quoted with `quoted_limit`;
    a stress, temperature or key pressure read equal to its limit with `quoted_load`.
  - Examples (C). Every number is written as the readouts write it (`format_value`: four significant digits, trailing
    zeros kept, so an input of 100 reads "100.0 °C", 18 reads "18.00 µm" and 2 reads "2.000"; the house's own Problem
    tests read so, e.g. "less the 10.00 °C design margin", `guidance.rs:5453`). Hub minimum 10: "At its loosest the
    inner hub holds 3.482 N·m, less than the 7.647 N·m it must hold: cold-high, 3.823 N·m, times the safety factor of
    2.000. On the inner hub, a measured interference from 12.09 µm to 18.82 µm would hold and fit at the allowed
    temperature (its band is 10.00 µm to 18.00 µm)." (results 12.0815 and 18.8281). The 304 probe: "No measured
    interference works on the inner hub here: holding needs at least 16.31 µm, but the bore stays within its limit only
    up to 11.72 µm." (and the cup's: 15.80 against 11.74; results 16.3031, 11.7235, 15.7920, 11.7439). Wheel shaft
    20 °C: "Fitting the inner hub would need it at 247.6 °C with the wheel shaft at 20.00 °C, above the 100.0 °C
    allowed: your limit for the bonded inner hub, 100.0 °C. No measured interference works on the inner hub here:
    holding needs at least 12.09 µm, but the allowed temperatures do not even open its 10.00 µm joining clearance."
    `breaking(cool_adhesive)`: "Fitting the inner hub would need it at 93.27 °C with the wheel shaft at -196.0 °C,
    above the 60.00 °C allowed: the selected adhesive's design limit, 60.00 °C. On the inner hub, a measured
    interference from 11.14 µm to 13.90 µm would hold and fit at the allowed temperature (its band is 13.00 µm to
    18.00 µm)." (results 11.1367 and 13.9081). `breaking(n42_parts)` (R-15): "Fitting the inner hub would need it at
    93.27 °C with the wheel shaft at -196.0 °C, above the 59.83 °C allowed: the magnets' single-ring onset, 69.83 °C,
    less 10.00 °C. On the inner hub, a measured interference from 11.62 µm to 13.88 µm would hold and fit at the
    allowed temperature (its band is 13.00 µm to 18.00 µm)." (results 93.2674, 59.8326, 69.8326, 11.6167 and 13.8875;
    C, `out.txt` 11.2). `passing_design()` with N50M on manual dimensions: "… above the 88.80 °C allowed: the magnets'
    single-ring onset, 98.80 °C, less 10.00 °C. … from 12.21 µm to 17.45 µm …" (results 88.8029, 98.8029, 12.2087,
    17.4509; 11.14).
- **Why** (one sentence, 29 words): "A shrink fit's grip holds each rotor on its shaft; it must carry cold-high with margin
  when loosest, stay elastic when tightest, and be fitted within the allowed temperatures."
- **Fixes** (`Fixes::PerTerm(shaft_joint_fixes)`): for each failing term in cause order, each of its fixes whose one
  step helps (`helps` against the term's margin), each fix once:

  | term | fixes |
  |---|---|
  | inner hub slips | Raise `shaft_joint.hub_interference_min_um`; Raise `metal.hub_length_mm` (review SF-2: a longer hub grips over more length); Raise `metal.face_gap_mm` (lowers cold-high: the need falls) |
  | cup slips | Raise `shaft_joint.cup_interference_min_um`; Raise `metal.boss_length_mm`; Lower `coupling.magnets.axial_length_mm` (the cup does not lengthen with the magnets); Raise `metal.face_gap_mm` |
  | inner hub over its stress limit | Lower `shaft_joint.hub_interference_max_um` |
  | cup over its stress limit | Lower `shaft_joint.cup_interference_max_um` |
  | inner hub cannot be fitted | Lower `shaft_joint.hub_interference_max_um`; Lower `shaft_joint.wheel_shaft_fit_C` |
  | cup cannot be fitted | Lower `shaft_joint.cup_interference_max_um`; Lower `shaft_joint.output_shaft_fit_C` (it stops at −20 °C) |
  | inner hub key | Raise `shaft_joint.key_contact_mm`; Raise `metal.hub_length_mm` |
  | cup key | Raise `shaft_joint.key_contact_mm`; Raise `metal.boss_length_mm` |

  **Margins** (each finite wherever the term can fail on the sliders, so a fix that helps is seen to help; a typed key
  width of π·d or more leaves the slip margin −∞, and that block shows no fix, review 3 SF-3):
  - slip: δ_min − `{joint}_needed_um` [µm] (review SF-1), the band's minimum less the needed minimum. Holding is
    `max(0, …)·T_per_µm`, so holding − required would be flat wherever the effective interference is floored at zero
    (a hub minimum of 8.26 µm or less at the defaults; the 304 and aluminium hubs at their defaults), and `helps()`
    would hide the main fix there. This margin is strictly monotone in the minimum (C: 0 to 40 µm, no flat step; the
    304 hub −3.30 -> −2.80 µm for one step), and its sign matches holding ≥ required whenever the need is positive
    (C: every 0.5 µm step from 0 to 40). A need at or below zero greys the row (`END_EFFECT_ROWS`).
  - stress: limit − stress [MPa]. Fitting: allowed − needed temperature [°C].
  - key: l − l_needed [mm] (review MF-2; section 3.1), not limit − key pressure: that is −∞ at every slider value
    with no bearing length (boss length 0 to 0.5 mm, hub length 3.0 mm), so no one step would show any fix moving it,
    and `each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red` would fail on the "cup key, no
    bearing area" design (section 7.9). C: l_needed 3.685 mm at the defaults (margins 6.315 / 8.815 mm); at
    `passing_design()` 3.237 mm, and with boss length 0 the cup's margin is −3.737 mm, rising 0.1 mm per boss step;
    its sign matches the key verdict over boss lengths 0 to 40 mm at contact heights 0.5, 0.8, 1.2 and 1.7 mm. The
    result rows keep +inf. With no keyway the key term passes with margin +∞ (section 3.1; review 2, SF-5c), never the
    −∞ that k = 0 would give: `Term.passes` alone would keep today's tests safe, but `helps()` reads the margin.
  - Each margin is monotone over its fix's slider (C: every row above swept over the whole slider, including the hub
    minimum past the maximum, the wheel shaft from −196 to 40 °C, the hub length from 3 to 40 mm and the boss length
    from 0 to 40 mm). Two are only non-decreasing: the key margin against the contact height is flat past the keyway
    depth (13 steps at 1.7 mm), and the hub's slip margin against the hub length is flat where the axial override
    sets the hub's length (97 steps at an override of 40 mm, `housing.rs:132`); `helps()` hides a fix where its step
    is flat.

  **Three fixes held back by rule** (explicit, in `shaft_joint_fixes`):
  - "Raise {joint} minimum" is not offered where one step would put the minimum above the maximum (review SF-10): the
    band would then read reversed and the row amber, not green, which `helps()` (a margin that rises) would count as
    helping. The window sentence already gives the band to enter. "Raise {joint} maximum" is not offered in its place:
    it does not move the slip margin, and it moves the stress and fitting terms the wrong way.
  - "Lower {joint} maximum" is not offered, for the stress or the fitting term, where one step would put the maximum
    below the minimum (review 2, SF-2, the mirror of the rule above), for the same reason. C (`out.txt` 10.3):
    `passing_design()` with DP460 and the hub band 18-18 fails on fitting alone (93.3 against 60.0 °C); one step on the
    maximum (17.5) reads "Shaft joint check unknown: the minimum interference is above the maximum (inner hub)",
    amber, though the fitting margin rises (−33.27 -> −29.20 °C); the wheel shaft is at its slider's bottom. So no fix
    shows (`NO_FIX`), and the window sentence names the band: "On the inner hub, a measured interference from
    11.14 µm to 13.90 µm would hold and fit at the allowed temperature (its band is 18.00 µm to 18.00 µm)."
  - "Raise key contact height" is not offered for a joint whose key has no bearing length (l ≤ 0) and a positive
    contact height: there no contact height gives the key any bearing area, though the margin rises with it. Only the
    length fix (hub or boss) shows. With a contact height not above zero (typed) the contact fix is the one that moves
    it (the margin is −∞ until the contact is positive).

  The unknown causes give no fix (the Problem asks for the number, the wall or the band); with one joint indeterminate
  the other's failing terms keep theirs (section 4.3).
  **Not offered:** friction, smoothing, the rotor-over-shaft allowance and the joining clearance (assumptions are not
  fixes, `guidance.rs:2884-2915`); the safety factor (not a design lever); the allowed fitting temperatures (limits of
  the bond, the magnets, the steel and the gearbox: offering them invites overheating; the inner hub's is also capped by
  the onset, R-15); the back iron (a choice that
  turns other checks red); the inner hub's magnet length (monotone but almost useless, since the hub grows with the
  magnets); a fit class (it does not fix the interference of a bought shaft); the keyway depth
  (`coupling.keyway_depth_mm`, review 2 SF-10), though it caps the contact height, so "Raise keyway depth" would move
  the key margin where the contact is capped (keyway 0.3 mm, section 4.2): it is the real keyway's number, not a free
  choice; a deeper keyway thins the wall past it (the no-wall rule of section 3.1); and it enters Auto's validity rule
  (`model.hub_wall_past_key_mm` > 0, `sizing.rs:222-227`). Raise key contact height is offered there and `helps()`
  hides it while the contact is capped. Nor the magnets' part or grade (`coupling.magnets.part_*`, `grade_*`), though
  a hotter inner one raises the inner ring's onset that caps the hub (R-15 and its ring ruling); see "Where the onset
  binds" below.
  Raise `metal.hub_length_mm` (C, `out.txt` 9.3): one step from `passing_design()` and from the defaults turns no other
  check red anywhere from 3 to 40 mm; at `passing_design()` with the hub length at 3.0 mm the key passes from 6.3 mm and
  the slip from 9.3 mm.

  **Where the onset binds (owner R-15).** The inner hub's fitting term keeps its two fixes, and they are the ones that
  help (C, `out.txt` 11.2, 11.11):
  - Lower `shaft_joint.hub_interference_max_um` is shown. At `breaking(n42_parts)` one step (18 -> 17.5 µm) raises the
    fitting margin from −33.43 to −29.37 °C and turns no other check red. The window sentence gives the band that
    fits: 11.62 to 13.88 µm, so a maximum of 13.5 µm or less. At `passing_design()` with N50M one step takes the
    margin from −4.46 to −0.40 °C.
  - Lower `shaft_joint.wheel_shaft_fit_C` is not shown there: the wheel shaft is at its slider's bottom, −196 °C.
  - The "Lower maximum" rule above still holds: with a band whose maximum is one step from its minimum, the block
    shows `NO_FIX` and the window sentence names the band.
  The magnets' part or grade is not offered for the joint's fitting term, though the inner one moves the cap (the cap
  reads the inner ring's own onset, owner R-15 ring: `breaking(n42_parts)` with the inner part changed to B842SH
  passes, allowed 100 °C, C `out.txt` 13.5), for three reasons:
  1. It is the magnets' choice, made for torque and service temperature. A grade or part sets Br and so cold-high,
     which moves both joints' slip and key terms and the hot-low, cold-high and gearbox checks. Offered from a fitting
     limit, it would trade the magnetics for an assembly step.
  2. The onset it would raise rests on the stored 3D single-ring field of the example design (the row's "Fixed 3D
     fields" label), so the gain it promises is not this design's own.
  3. Where the onset binds, the temperature check is amber at every design probed (11.4: each grade on manual
     dimensions whose onset less 10 °C is under the 100 °C input reads Caution on temperature; over the 289 grade pairs
     on manual dimensions at `passing_design()`, the inner ring's own onset binds in 136 and the temperature check is
     Caution in all 136, C 13.2), and its guide already offers the hotter part or grade (`breaking(n42_parts)`: Change
     inner part, Change outer part; the manual grades: both rings' part and grade; 11.2, 11.3). Over the 136 pairs it
     offers the inner part or grade in 135; the exception, inner N50M with outer SmCo 2:17-26 (`SmCo_2_17_26`), shows
     no temperature fix at all (13.2).
- `terms()` (test module) gains the eight terms with these margins (an indeterminate joint's terms pass there, as they
  are not judged; with no keyway both key terms pass with margin +∞, so the count stays eight). The doc comments' counts move ("the nine dashboard checks", `guidance.rs:309`, `:1502-1505`).

### 5.3 Inputs panel (`M/src/gui/inputs.rs`)

- `SECTION_LABELS` 18 -> 19: `("shaft_joint", "Shaft joints")`, after `clamps` (package order).
- Workflow group "Shaft and hub joint" (`:327-345`):
  - section `shaft` gains `shaft_joint.key_width_mm` and `shaft_joint.key_contact_mm` after the clamp's key rows
    (hidden while paused), so it shows bore, keyway depth, shaft key width, shaft key contact height;
  - a new section `shaft.fit`, label "Shrink fit", not advanced: the four interference inputs, the safety factor,
    friction, smoothing, the rotor-over-shaft allowance;
  - a new section `shaft.fitting`, label "Fitting temperatures", not advanced: the two allowed temperatures, the two
    shaft temperatures, the joining clearance.
  - **Section order** (review SF-17): `shaft`, then `shaft.fit`, then `shaft.fitting`, then the paused clamp's
    `shaft.clamp` and `shaft.clamp_geometry`, then the advanced `shaft.screw_model` last. The joint in force comes
    before the paused clamp, so the order shown is the same while paused and not, and no ordinary section follows the
    advanced one (`the_workflow_groups_run_in_design_order_with_the_advanced_rows_last`).
  - Section labels are ASCII and contain no "clamp"; ids start with `shaft.`. No new advanced row
    (`rows(true) == 27` holds).
- **Overlays (`M/src/gui/plain_text.rs`):** nothing is required (Rust-only help is written in the engine and never
  overlaid, `docs/ai/02-system.yaml:109-111`). Three shown helps become incomplete, not false, and go to the G-3
  reviewer as should-fix candidates: the bore's ("the shaft in the hub bore": it is now both shafts and both fits), the
  keyway depth's (its Python-locked label is "Keyway depth in the hub", but it now also sets the cup's keyway, caps the
  key's contact height and enters the no-wall rule of both joints; review 2, SF-6d) and the boss length's (it now sets
  the cup's fit length). A fourth follows from R-15: the overlay help of `temperature.demag.h_rev_single_ring_kA_m`
  (`plain_text.rs:662-665`, "higher lowers the cure margin") does not say that a higher field now also lowers the
  bonded inner hub's allowed fitting temperature. The governing onset's own label ("Onset, single ring during an
  adhesive cure", `temperature.rs:483`) is Python-ported and stays; the check reads the new inner-ring result, whose
  label and help are Rust-only and written in the engine (section 4.1), and the `hub_fit_allowed_C` help names the
  onset.

### 5.4 Result groups, drawing, exports, session, Auto

- `M/src/gui/result_groups.rs` (review MF-3):
  - `CHAIN_LABELS` 8 -> 9: `("shaft_joint", "Shaft joints")` after `("geometry", "Geometry")` and before
    `("mass", "Mass")`, so the scope's chains keep the scope's order and the mass stays last. The results table shows
    the 23 numeric results under the chain group "Shaft joints", between Geometry and Mass; `shaft_joint.check` is in
    the headline group (a `DASHBOARD` row, taken first).
  - `PACKAGE_LABELS` 13 -> 14, `("shaft_joint", "Shaft joints")`: the heading test requires every result's package to
    have a heading (`every_chain_path_names_results_and_every_package_has_a_heading`), though the package group stays
    empty and is dropped (a chain and a package may share an id). No `CHAIN_PREFIXES` entry.
  - `the_headline_comes_first_then_the_chains_then_the_other_results` (`:347-357`): `ids[..9]` -> `ids[..10]` with
    `"shaft_joint"` between `"geometry"` and `"mass"`, `groups[..10]` not other, `groups[10..]` other; it gains
    `group_of("shaft_joint.hub_holding_Nm") == "shaft_joint"` and `group_of("shaft_joint.check") == "headline"`.
  - The inner ring's onset (owner R-15 ring) is a `demagnetization` chain path (section 4.5), so it sits among that
    chain's own rows, after the governing single-ring onset; the test gains
    `group_of("temperature.demag.inner_onset_single_ring_C") == "demagnetization"` (beside the
    `inner_magnet_limit_C` assertion at `:306-309`, which the `temperature.demag.` prefix places).
  - The module doc's chain list (`:1-8`, "torque, temperature, demagnetization, slip heating, clamps, geometry") gains
    the shaft joints.
- `M/src/gui/geometry.rs:448-457`: the key is drawn in both states from `shaft_joint.key_width_mm` (depth from the
  keyway depth); the comment at `:448-450` ("the key's width (Shaft clamps C58). The key is the shaft-clamp section's:
  not drawn while it is paused") reworded to the shaft joints' key, drawn in both states, and the test
  `the_key_and_the_clamp_boss_check_are_drawn_only_while_the_clamp_section_shows` (`:981`) renamed, for example
  `the_key_is_drawn_in_both_states_and_the_clamp_boss_check_only_while_the_clamp_section_shows` (review 2, SF-11). The
  boss design check stays paused (BL-060). The interference has nothing to draw. The key outline test (`geometry.rs:975-977`) moves with the 3 mm key: `[6.7, -2.0]` and √21 become
  `[6.7, -1.5]` and √22.75 ("its 3 mm sides").
- Exports pick the new inputs and results up with no code: CSV, JSON, the XLSX Inputs and Results sheets; the XLSX
  Summary lists 19 dashboard rows with the joint badge.
- `M/src/gui/session.rs:578-583`: the default share link grows from 2,462 to **2,643 characters** with the 15 inputs at
  their defaults (C, the house encoder: deflate best, URL-safe base64, checked equal to `encode_share_payload` at
  HEAD). The guard `< 2_500` becomes `< 2_700`, in the same commit as the inputs, its comment quoting both lengths and
  the reason (the whole design is in the link). A design file or link from before BL-064 loads with the new inputs at
  their defaults (no `DESIGN_VERSION` bump, `session.rs:18-19`, `:59-61`).
- `M/src/gui/test_support.rs`: `passing_design()` unchanged (it passes the joint: 9.48 against 6.72 N·m, 93.3 °C
  against 100 °C); its doc says "all 10 guided checks". New `only_the_joint_fails()`: `passing_design()` with
  `shaft_joint.hub_interference_min_um` 8 (effective 0: the inner hub slips, nothing else fails).
- Auto (`sizing.rs`, the Size tab): unchanged (decision 21). At the defaults Auto on length solves 13.770 mm and both
  joints hold there (inner hub 10.27, cup 16.90 against 8.37 N·m, C).

## 6. Notes and texts

### 6.1 The new note `shaft_joint` (decision 8A; drafts, the physics reviewer sets the words)

- `id` `shaft_joint`, title "Shaft joints: a keyed shrink fit". It owns the 23 numeric records and the inner ring's
  single-ring onset it reads (`temperature.demag.inner_onset_single_ring_C`, owner R-15 ring): 24 equations. The
  onset's Equation panel then links the note that says why the hub's cap reads it; the reviewed `demagnetization`
  note, which owns the governing onset and the per-ring magnet limits, is left unchanged (a record has at most one
  note, `tests/explain.rs:1531-1564`). Its Start here step is
  `("shaft_joint", "shaft_joint.hub_holding_Nm")`, inserted just before `("clamp_preload", "clamps.capacity_Nm")`, which
  stays last.
- **What it teaches** (3-5 sentences of at most 25 words; sentence 0 plain, no live value):
  0. "A shrink fit holds each rotor on its shaft: the bore is made smaller than the shaft, and friction from the grip
     carries the torque." (25 words)
  1. Holding at the loosest against the need, live: "At its loosest the inner hub holds {shaft_joint.hub_holding_Nm}
     here and the cup {shaft_joint.cup_holding_Nm}; each must hold {shaft_joint.required_Nm}."
  2. "Measured interference is not what grips: smoothing takes micrometres off at fitting, and a rotor running warmer
     than its shaft loses more." (22 words; review SF-12: the warm-rotor loss is in running, not before gripping)
  3. "Tighter is not safer here: the cup's bore stress, {shaft_joint.cup_bore_stress_MPa} against
     {shaft_joint.stress_limit_MPa}, follows interference over diameter, hardly the wall." (19 words, no plain-words
     violation, C `out.txt` 14.3; review 6, SF-1: it names the joint and quotes the one nearer its limit, the cup at
     90 % against the inner hub's 70 % (section 3.5), whose window the stress edge binds at the defaults; unnamed, the
     inner hub's 70 % read as the coupling's bore-stress use. Review 5, SF-4: the earlier draft's `{coupling.bore_mm}`
     is an input, and a note placeholder must name a number result, so it would fail the placeholder test and read "no
     prediction" on screen; the only result that carries the bore, `clamps.shaft_mm`, is the paused clamp's and is not
     used. It keeps "here". Von Mises at the bore runs from about 0.87 to 1.0 × E·δ/d from a thick to a thin wall,
     17.75 to 20.5 MPa per µm at Ø10.) It renders "Tighter is not safer here: the cup's bore stress, 312.5 MPa against
     345.8 MPa, …" at the defaults (C 14.3).
  4. "Here the inner hub would need {shaft_joint.hub_fit_temp_C} on the wheel shaft shrunk
     {shaft_joint.wheel_shaft_shrink_um}; allowed: {shaft_joint.hub_fit_allowed_C}, lowest of your limit, adhesive's,
     single-ring onset less 10 °C." (25 words, the cap, counted as `plain::word_count` counts, a placeholder as one; no
     plain-words violation, C 13.8; owner R-15; review MF-6: the need against the allowed temperature, with its
     sources, never a temperature to use; review 4, MF-2; review 5, MF-2). It renders "Here the inner hub would need
     93.27 °C on the wheel shaft shrunk 18.99 µm; allowed: 100.0 °C, …" at the defaults, "allowed: 59.83 °C" at
     `breaking(n42_parts)` and "would need 247.6 °C on the wheel shaft shrunk 0 µm" on a 20 °C wheel shaft (C 13.8),
     and "Here the inner hub would need 266.3 °C on the wheel shaft shrunk -2.300 µm; allowed: 100.0 °C, …" on a
     40 °C wheel shaft, its slider's top (C 14.4; review 6, SF-4). That last render is not false (the shrink's help
     says "negative above 20 °C, where it grows"), but "shrunk -2.300 µm" reads oddly; a rewording that avoids it
     ("with a wheel shaft shrink of {…}") costs one word over the 25-word cap. The physics reviewer, who sets the
     words, sees it and decides.
     `{shaft_joint.hub_fit_allowed_C}` is the lowest of the three; the sentence cannot name which of the three binds,
     since a note placeholder names a number result and a note's words do not switch with the design
     (`note_text.rs:1-6`); the Problem names it (section 5.2). The list form names the three limits and reads the
     "less 10 °C" with the onset alone. It keeps the wheel shaft's state beside the need, as its live shrink (a note
     placeholder names a number result, `tests/explain.rs:4818-4826`, so an input cannot stand there), and it holds a
     "here" clause (`:4727-4740`). The physics reviewer may reword it within those rules. (Review 1, MF-6: the need
     holds only on the cooled shaft; an uncooled shaft reads a shrink of 0, and on a 20 °C shaft the inner hub would
     need 247.6 °C, and a hub warmed to about 93 °C seizes half-way on it, trap 14.) "Single-ring onset" is the inner
     ring's own, the hub's own magnets' (owner R-15 ring). The "10 °C" is typed in the note; a test holds it to
     `CURE_MARGIN_MIN_C` (section 7.11). The bare cup's order is in watch-out 1.
- **Change entries** (1-5, each with an aim that holds at the five probe designs; the 304 probe fails the joint and its
  inner hub's holding is floored at zero, so no entry aims at the hub's holding):
  - Raise `shaft_joint.cup_interference_min_um`, aim (`shaft_joint.cup_holding_Nm`, Up): "the cup grips harder at its
    loosest; its maximum must still stay within the stress limit and fit."
  - Lower `shaft_joint.hub_interference_max_um`, aim (`shaft_joint.hub_bore_stress_MPa`, Down).
  - Raise `shaft_joint.wheel_shaft_fit_C`, aim (`shaft_joint.hub_fit_temp_C`, Up): "a warmer wheel shaft shrinks less,
    so the inner hub needs more heat."
  - Raise `metal.boss_length_mm`, aim (`shaft_joint.cup_holding_Nm`, Up): "a longer boss grips over more length; the
    overall length grows."
  - Lower `shaft_joint.smoothing_um`, aim (`shaft_joint.cup_holding_Nm`, Up): "only with a measured finish: the grip
    rises, and so does the bore stress."
  Each aim holds one slider step from each of the five probes in the scratch crate's model (C, `out.txt` section 6b);
  the implementer re-checks with `probes::step_input`.
- **Watch-outs** (0-2, at most 30 words; both fixed, review MF-1; the physics reviewer sets the words, not the topics):
  1. "Fit the bare cup quickly and keep its heat off the gearbox seal; nothing here checks the gearbox's shaft, bearings
     or seal, or fretting under the reversing swing." (28 words: the owner's "Cup temp" decision, which says the note
     says to keep the heat off the gearbox seal and fit quickly)
  2. "A wheel shaft cooled in liquid nitrogen can chill the bonded inner hub below the minimum magnet temperature,
     widening the bond's cold swing; ferrite magnets lose coercivity cold." (28 words: the default wheel shaft is
     −196 °C, so traps 12 and 20 apply at the default design. Review 2, SF-6a: it names the shaft and the joint; the
     bond's wider swing is research-assembly 2.3's: a hub that settles near −55 °C gives the bond a 77 K cold swing
     against the 62 K the engine's mismatch screen covers, C rough)
  The draft smoothing watch-out ("The smoothing assumed is taken off the maximum too: a finer finish than assumed leaves
  more grip and more bore stress.") moves to R-1, which decides the stress side's smoothing.
- **Sources** (plain, no audit ids, no "BL-064"): the owner's decisions of 2026-10-17 (a keyed shrink fit on both
  rotors, the bands, the fitting order and temperatures, and the bonded inner hub held 10 °C below its magnets'
  single-ring onset, the margin the adhesive cure keeps); Lamé thick-cylinder fit pressure and bore stress (Shigley,
  Roark); DIN 7190-1:2017 interference fits (the 0.4·Rz smoothing rule that makes 5 µm conservative for holding, the
  joining clearance); DIN 6885-1 parallel keys (3 × 3 for shafts over 8 up to 10 mm); NIST cryogenic material
  properties, nickel steel Fe-2.25Ni linear expansion; friction of dry steel on steel (Gear Technology, Nov/Dec 2017:
  0.145; ANSI/AGMA 9003: 0.15).
- **Review:** `Review::Reviewed { reviewer: "physics reviewer (session model)", date, record:
  "docs/analyses/2026-10-14-magcoupler-notes-review.md#shaft_joint" }` after a fresh-context session-model review that
  reads the note at the five probe designs, reproduces every number and probes every direction; a new `## shaft_joint`
  section with "Date: <date>".
- `NOTES` 20 -> 21; the cap at `notes.rs:1407` becomes `(15..=21)` (the owner raised it); `START_HERE_ALL` 14 -> 15;
  `SLICE_2_ORDER`; the hover topics macro (`explorer.rs:112-133`) gains "shaft joints"; `CLAMPS_PAUSED_LINE`
  (`explorer.rs:2764`) no longer says the clamp returns with BL-064 (draft: "The shaft-clamp section is paused (owner,
  2026-10-15): its note, clamp_preload, keeps its earlier text, is not shown on screen and is not part of this
  sign-off; the shafts' keyed shrink fits are checked instead (owner, 2026-10-17).").
- **The sign-off sheet** (review SF-14; `explorer.rs:2754-2932`). A post-slice list `AFTER_THE_SLICE: [&str; 1] =
  ["shaft_joint"]` sits beside `NEW_NOTES`, which keeps meaning slice 2's three notes. `signoff_sheet` prints, in place
  of the Before text of a note in `AFTER_THE_SLICE`, a new constant `ADDED_AFTER_THE_SLICE` = "(added after slice 2;
  its own physics review is its record)" instead of `NEW_IN_SLICE` ("(new in slice 2)", which would be false for it).
  The test `the_signoff_sheet_pairs_every_note_with_its_text_before_the_slice`: the snapshot holds
  `NOTES.len() − NEW_NOTES.len() − AFTER_THE_SLICE.len()` notes (17); the notes without a Before text are
  `NEW_NOTES` and `AFTER_THE_SLICE` together; in both switch states the sheet holds `NEW_IN_SLICE` 3 times and
  `ADDED_AFTER_THE_SLICE` once (the note is shown while paused). The sheet file is not regenerated (decision 23): the
  owner's review of the BL-064 diff is the new note's sign-off.
- The torque-band note's sentence 3 and source (`notes.rs:557`, `:594`) stay true; left as they are (its caps are full).
  The 14 "shaft-joint and gearbox loads rise" change entries stay true.

### 6.2 Traps every new text is checked against

For every text: the engine labels and helps (inputs, results, the check), the verdict constants, the guide's label,
Problem and Why, the note, and the four assumption labels and rationales.
1. **Never add the fit's and the key's torques.** The key carries nothing until the fit slips.
2. **Cold-high can reach the joints; it is not a bound on them** (backlash, resonance crossings and re-lock ring-down can
   exceed it on the gearbox side), and with the operating temperature below the minimum it is not the strongest torque
   expected either (BL-067). Write "can reach"; never "at most", "the strongest" or "the most expected" (review 2,
   MF-1; section 7.8 pins it).
3. **No ratio or efficiency.** Both joints are on the output side; never quote `metal.cold_high_input_Nm`.
4. **166.7 Hz is the swing frequency at the defaults:** 16.7 full cycles, about 33 reversals, per 0.1 s slip event
   (review 2, MF-2; E: `metal.slip_freq_Hz` 166.67 Hz × `metal.slip_event_s` 0.1 s; one cycle is two reversals).
5. **Measured is not effective.** Name which one a number is. Assembly uses the measured maximum (no smoothing taken
   off); holding uses the effective minimum.
6. **"Tighter is safer" is false at Ø10.** Never print a fit class as a fix ("H7/s6"); say "set the interference by
   measuring the shaft".
7. **Temperature cuts both ways.** A part above the shaft's expansion (304, aluminium) loses grip hot; below it (416,
   17-4PH H900), cold. Never "temperature does not affect the fit".
8. **Speed does not loosen this fit** (0.0017 µm at 2000 rpm); do not cite it.
9. **Name the limit's source with every fitting temperature:** "your limit for the bonded inner hub", "the adhesive's
   design limit", "the magnets' single-ring onset less 10 °C" (owner R-15; the hub's own magnets' onset, the inner
   ring's, by its ring ruling), "your limit for the bare cup". "Rated to
   120 °C" is not "safe at 120 °C": at 0.05 mm bondline the engine's own mismatch screen passes the lap shear near
   113 °C. Never write "safe". A needed temperature is what the fit
   would take, never a temperature to use: quote it against the allowed one. Trap 9 applies to the Problem, the note,
   the labels and the helps; the badge ("Cannot be fitted at the allowed temperature (inner hub): needs 93.3 °C
   against 93.0 °C") is exempt from naming the source, since it says "allowed" and the Problem under it names the
   source (review SF-12).
10. **Do not quote the grade's 150 °C** as the magnets' limit for fitting: the rotor alone sees its single-ring field
    (onset 143.4 °C, E); below the onset the flux lost hot returns on cooling ("reversible", never "no loss").
11. **The gearbox:** never suggest cooling its shaft below its rated range, or with dry ice or liquid nitrogen; never
    suggest pressing the cup on against its bearings; name the catalog only as the source of a default (the GAM model
    is unconfirmed); never "protect".
12. **Cryogenic temperatures:** helps may name the bath temperatures as facts, never as advice; no text makes −196 °C
    sound routine; a cryogenic shaft can chill the bonded inner hub below the minimum magnet temperature.
13. **No induction heating** of a magnetized rotor; if a method is named, "oven" or "hot plate".
14. **"Shrink fit" is not "heat the hub":** write "warm the part or cool the shaft"; never "heat until it slides on"
    (seizing half-way is the failure; a stuck rotor cannot be tapped home).
15. **The fitting order is the owner's:** the bare cup first, then its magnets bonded; the inner hub bonded, then
    fitted on a cooled shaft. Never "fit the bare hub first".
16. **The torque is not why fitting is hard;** do not write "lower the safety factor to make it fit" (SF 1.5 saves
    about 1 µm). The finish and the band width decide.
17. **The key row:** a Ø10 shaft takes DIN 6885-1 3 × 3; name the row; the 1.7 mm keyway default matches no row.
18. **"Holds without slipping up to …", never "permanent" or "no backlash":** micro-slip at the part's ends happens
    below the slip torque.
19. **Friction and smoothing are assumptions:** capacity scales with friction; smoothing lowers grip and stress
    together.
20. **Nothing checks** the gearbox's shaft, bearings or seal, the keyway corner's stress concentration, the bond strain
    from fitting a bonded hub, how the liner and band are held when a rotor is heated, or fretting. Say so where it
    matters; claim none of them is fine.
21. **No "clamp"** anywhere, not "clamping pressure": write "contact pressure" or "grip" (the panel guard is a
    case-insensitive substring test over texts, paths and cells under details).
22. **Plain words and glyphs:** no cell ids, no "BL-064", no "Rust-only"; "->", "Ø", "-20 °C" (the house minus sign);
    no U+2192 or U+2300.

### 6.3 Reviews

- The note: a fresh-context session-model physics review (section 6.1).
- One G-3 review, recorded under a new `## BL-064 shaft joints` section of
  `docs/analyses/2026-10-14-magcoupler-help-review.md`: the 15 input labels and helps, the 24 result labels and helps,
  the verdict texts, the guide's label, Problem sentences and Why, the four assumption labels and rationales, the two
  section labels and the package label, `STORED_3D_NOTE`'s new sentence (R-15, section 5.1), and the label and help of
  the new temperature result `temperature.demag.inner_onset_single_ring_C` (owner R-15 ring, section 4.1); plain words for
  engineers who are not magnet specialists; nothing false at the defaults, the five probes or any reachable input (the
  designs of section 3.6 are the reviewer's starting set).

## 7. Tests (red first: each fails at HEAD and passes after)

1. **Engine, `M/tests/shaft_joint.rs`** (modelled on `tests/gearbox.rs`), end to end through `compute_all` and on the
   pure `joint` function:
   - per µm: 7.738 MPa, 2.0009 N·m and 17.930 MPa (inner hub, 3 mm key); 8.132, 2.5073, 17.879 (cup);
   - the defaults pass with the numbers of section 3.5 (9.4844 / 16.899 N·m; 241.70 / 312.53 MPa; 127.44 / 101.95 MPa;
     93.267 / 280.16 °C; windows 12.082-18.828 and 11.310-23.863 µm);
   - **the BL-067 corner** (review MF-4; `coupling.op_temp_C` −40, `metal.min_temp_C` 20, `metal.variation` 0.1) reads
     required 6.3647 N·m (2 × cold-high 3.1823), holding 9.484 / 16.899 N·m, bore stress 233.09 / 303.95 MPa, key
     106.08 / 84.86 MPa, and passes. A comment names BL-067: the check reads cold-high, not the 3.657 N·m operating
     pull-out with its allowance (pull-out 3.3246 N·m × 1.1, C), so BL-067's fix moves this pin on purpose;
   - the inner hub at `metal.hub_length_mm` 3.0, its slider's bottom (review SF-2): L = b = 3 mm, holding 2.189 N·m
     against 7.647, the key +∞ with no bearing area, both causes in the Problem, the slip in the badge;
   - each of the five probes reads the verdict of section 3.6, in `probe_designs()` order;
   - hub minimum 10 slips at the inner hub (3.48 against 7.65); cup minimum 10 slips at the cup (4.36);
   - hub maximum 26 and 28 read the stress text (385.1, 421.0); the cup's 26 and 28 too (384.0, 419.8); the fitting
     cause follows in the Problem, not the badge;
   - the 304 and aluminium windows are empty (16.30 > 11.72; 27.74 > 25.22);
   - the 416 part's minimum is taken cold, the 304's hot (correction 1 of the synthesis; this pins the hot-end form of
     section 3.3, which R-14 puts to the physics review: the other form takes 416's minimum hot);
   - wheel shaft 20 °C and −78.5 °C cannot be fitted (247.6, 163.5 against 100); the bare cup at 250 °C cannot (280.2);
   - DP460 caps the hub's allowed temperature at 60 °C; a hub input of 150 °C (typed, above the 130 °C slider top) is
     capped at the AA 326's 120 °C;
   - the hub fitting slider's top, 130 °C, is below the inner ring's single-ring onset less the cure margin at the
     defaults (`temperature.demag.inner_onset_single_ring_C` − `CURE_MARGIN_MIN_C` = 133.36 °C; review SF-3);
   - **the allowed hub temperature is the lowest of three (owner R-15; C, `out.txt` 11.1-11.10):** 100 °C at the
     defaults (your limit; the onset 143.3565 °C less 10 is 133.3565 °C); 59.8326 °C at `breaking(n42_parts)`, which
     reads "Cannot be fitted at the allowed temperature (inner hub): needs 93.3 °C against 59.8 °C" (it read a pass
     before R-15); 88.8029 °C at `passing_design()` with N50M on manual dimensions in both rings (onset 98.8029 °C,
     above the 93.267 °C need; only the onset less 10 °C is below it), "needs 93.3 °C against 88.8 °C"; 100 °C at the
     Y30 ferrite probe and `breaking(ferrite)` (onset +∞); 133.3565 °C with 2214 Hi-Temp and a typed hub input of
     150 °C (the onset binds, not the 177 °C adhesive limit). Each test names R-15 in a comment and reads
     `CURE_MARGIN_MIN_C`, never a typed 10;
   - **the cap reads the inner ring's own onset with mixed rings (owner R-15 ring; C, `out.txt` 13.5, 13.3, 14.2):**
     - `passing_design()` with inner B842SH and outer B842 passes (allowed 100 °C, your limit; window 11.62-18.83 µm)
       though the outer B842 governs with 69.83 °C, and asserts `temperature.demag.demag_ring` = "outer" with
       `temperature.demag.onset_single_ring_C` = 69.8326 °C beside the inner onset of 143.3565 °C;
     - inner B842 with outer B842SH reads "needs 93.3 °C against 59.8 °C" (the inner B842's own 69.83 °C less 10);
     - with 2214 Hi-Temp, a 130 °C hub input and a 22 µm hub maximum, inner N42H with outer `SmCo_2_17_30` (the grade
       id the test sets) on manual dimensions reads "Cannot be fitted at the allowed temperature (inner hub): needs
       125.8 °C against 102.8 °C", and asserts `temperature.demag.demag_ring` = "outer" with
       `temperature.demag.onset_single_ring_C` = 191.8772 °C beside the inner onset of 112.8297 °C (the same at
       `passing_design()` and at the defaults, C 14.2);
     - each pin then fails if the cap read the governing ring's onset or if a grade or part name were mistyped (review
       6, SF-2). A name that is not a grade id, such as this spec's prose name "SmCo 2:17-30" (the grade's display
       name is "Recoma 30 (Sm2Co17 grade 30)"), is taken by `DesignInputs::set` and by a direct field write alike. The outer ring then reads the Hcj inputs, the inner N42H
       ring governs with 112.8297 °C, and the badge reads "needs 125.8 °C against 102.8 °C" under either onset
       (C 14.2). Each test names the ring ruling in a comment;
   - **the inner ring's onset result** (owner R-15 ring; in `temperature.rs`'s tests): `each_ring_shows_its_own_block`
     (`:2128`) gains `mixed.inner_onset_single_ring_C == ndfeb.onset_single_ring_C` (the NdFeB inner ring's own, the
     NdFeB-only run's onset); on the swapped rings (a ferrite inner ring, the NdFeB outer governing) the inner onset is
     +∞ while `onset_single_ring_C` is the NdFeB ring's; through `compute_all`, `passing_design()` with inner B842SH
     and outer B842 reads 143.3565 °C against the governing 69.8326 °C (C 12.3). Which pin tells the rules apart
     (review 6, SF-2): in `each_ring_shows_its_own_block` the NdFeB inner ring governs (`temperature.rs:2164-2166`),
     so its assertion would also hold were the result written from the governing block, `hot.onsets[3]`. The swapped
     rings (the inner onset +∞ while the governing onset is the NdFeB ring's; built as
     `e20_checks_both_rings_each_side_from_the_weaker` builds them, `:2101`) are the pin that distinguishes. Both are
     kept, and the swapped one is required;
   - the five probes and the defaults read as before R-15 (11.1), and as with the governing ring's onset (13.1);
   - an axial override of 40 mm names the **cup**; the cup starts to slip near 26.5 mm.
2. **Edges:**
   - exactly at the need, the limit, the allowed temperature and the key's limit (p_key = limit, key margin 0; review
     SF-2) passes; the next float fails (on `joint`);
   - NaN in each of the 15 inputs and in each engine value read reads unknown; adhesive code 0 reads unknown
     (`tests/robustness.rs` `an_invalid_adhesive_code_selects_no_adhesive` gains the assertion);
   - a NaN hub input (`shaft_joint.hub_fit_max_C`) makes `shaft_joint.hub_fit_allowed_C` NaN, not the 120 °C that
     `f64::min` gives (with a fittable 21.29 µm), and the check reads unknown (review 2, SF-3);
   - the allowed-temperature helper (owner R-15, on `hub_fit_allowed` directly): a NaN in any of its three arguments,
     in each position, gives NaN and no limit, though `py_min` alone would drop a NaN in its second argument; ties name
     the earlier limit (an input of 120 °C with AA 326 names your limit; an onset of exactly input + 10 °C names your
     limit); an onset of +∞ never binds; through `compute_all`, a single-ring reverse field written as NaN on the
     struct gives a NaN inner onset, a NaN allowed temperature and "Shaft joint check unknown: a value is not a
     number" (11.8, 13.1), and the field at its slider's top, 1500 kA/m, gives an inner onset of −2.449 °C and "needs
     93.3 °C against −12.4 °C" (a negative limit quoted by the same never-equal rule; 11.6, 13.1);
   - degenerate typed inputs (review 2, SF-4; in `tests/robustness.rs`): bore typed 0 gives a NaN holding from finite
     inputs and reads "Shaft joint check unknown: a value is not a number", never a pass; a key width typed 40 mm (above
     π·d) gives holding 0 N·m and a needed minimum of +∞ (never −2.87 N·m and a −4.39 µm window end), "The fit slips
     (inner hub): holds 0.00 N·m against 7.65 N·m" in the badge and no window sentence in the Problem;
   - key edges (review 2, SF-5): a contact height typed −0.5 mm gives the key no bearing area (+∞, never −305.9 MPa)
     and the sentence "… its contact height, -0.5 mm, is not above zero."; with boss length 0 and a contact typed 0
     (shared, so both joints' keys have no contact) each joint's key sentence is the contact-height one, and the cup's
     length sentence does not show beside it; with no keyway (keyway depth 0) both key terms pass with margin +∞ in
     `terms()`;
   - the key as the badge's first cause (review 2, SF-9): keyway depth 0.3 mm reads "Key backup above its limit (inner
     hub): 509.8 MPa against 345.8 MPa", with the cup's key (407.8 MPa) following in the Problem and both fits, both
     stresses and both fitting temperatures passing;
   - a back-iron pick outside the choices (review SF-16): on the steel circuit the part's modulus, expansion and yield
     read NaN and the check reads unknown, not a number; with `coupling.backiron` 0 they read 68.9 GPa, 23.6e-6 and
     241 MPa and the verdict is the aluminium row's ("slips (inner hub)");
   - a reversed band reads unknown and names the joint, and the terms are still computed; with the hub's band reversed
     (20 over 18) and the cup minimum at 10 the badge is the cup's slip (4.36 against 7.65), red (review SF-8), and the
     Problem holds the cup's slip, its window and the hub's reversed-band sentence;
   - no wall (review SF-8, SF-9): bore 16 mm with a 12 mm boss reads "The fit slips (inner hub): holds 4.57 N·m
     against 7.65 N·m" (the determinate hub's red outranks the cup's no wall), the cup's holding, bore stress, needed
     and elastic rows read NaN (its key, fittable and fitting-temperature rows are numbers), and the Problem ends with
     the cup's no-wall sentence; with the hub minimum also at 15 it reads unknown, "no wall around the bore (cup)";
     bore 16 mm with a keyway of 2.1 mm (the hub's wall 2.10 mm) reads unknown naming the inner hub, while 2.05 mm
     reads the hub's slip;
   - keyway depth 0: no key term, no arc (10.49 / 18.68 N·m), the key rows read 0;
   - boss length 0 (and 0.5, L = b exactly): the key has no bearing area, never a negative pressure that passes;
   - the key margin (review MF-2) is finite at boss lengths 0 to 40 mm and its sign matches the key verdict there at
     contact heights 0.5, 0.8, 1.2 and 1.7 mm (C: no disagreement); the slip margin's sign matches holding ≥ required
     over the hub minimum 0 to 40 µm (review SF-1);
   - the contact height is capped at the keyway depth (keyway 1.0: 152.9 / 122.3 MPa);
   - effective interference floored at zero (304 inner hub: 0 N·m);
   - a shaft above 20 °C (gearbox shaft 40 °C) gives a negative shrink (−2.30 µm) and passes at 298.9 °C;
   - the shrink is continuous at 265 K and strictly falls as the shaft warms over the whole slider;
   - the badge texts never show two equal numbers: the BL-059-style cases on `joint` (a stress 0.04 MPa over the limit;
     a fitting temperature 0.04 °C over; a holding 0.004 N·m under; a need under 0.005 N·m) and the slider-step case
     "needs 93.3 °C against 93.0 °C";
   - +∞ cold-high slips; `tests/robustness.rs` keeps running over the extremes.
3. **Rust-only and parity:** every new input and result has `meta.rust_only` and no cell; `python_schema.rs`,
   `parity.rs`, `differential.rs` and `deviations.rs` pass unchanged; labels and helps pass `tests/schema.rs`; a guard
   test in the spirit of `clamps.rs:596`: each `shaft_joint.*` input at both slider ends, at the five probe designs,
   moves no result outside `shaft_joint.*`.
4. **Dashboard row:** `the_shaft_joint_check_is_a_dashboard_row_with_its_guide`: one row, at index 15; in `CHECKS`; a
   guide whose learn is `Note("shaft_joint")`, equal to its `ROW_LEARN`; the gearbox row test moves to index 16 and
   `at(SHAFT_JOINT) == at(CLAMP) + 1`, `at(GEARBOX) == at(SHAFT_JOINT) + 1`.
5. **`verdict_level`:** Good, Caution for each unknown form, Bad for each of the five failing prefixes; the
   classification tests (`dashboard.rs:1043`, `:1132`) extended so every text `compute` can produce is classified. The
   key text is reached as the badge at keyway depth 0.3 mm (review 2, SF-9: "Key backup above its limit (inner hub):
   509.8 MPa against 345.8 MPa", Bad), so no prefix is classified only from a constant.
6. **Greying and the stored 3D fields** (owner R-15 flips the second half of this item):
   - at `short_magnets()` (f_end < 0) the row greys, `check_level` is `None`, the check leaves the failing list and the
     table badge; `END_EFFECT_ROWS` holds as a set (`dashboard_moved_by`);
   - the row **is** in `STORED_3D_ROWS` (4 rows). `the_stored_3d_rows_are_the_headline_rows_that_move_with_the_stored_3d_inputs`
     (`dashboard.rs:1506-1514`) is renamed `the_stored_3d_rows_are_the_dashboard_rows_that_move_with_the_stored_3d_inputs`
     (the joint row is not a headline row) and passes with the four rows: the joint row moves at back iron 1 with the
     single-ring field at its slider's top, through the inner ring's own onset (C, `out.txt` 11.6, 13.6);
   - `the_default_dashboard_has_badges_markers_and_the_3d_rows` counts 4 stored-3D lines (`dashboard.rs:1628`, 3 -> 4),
     and its loop holds the joint row's hover to `STORED_3D_LABEL`;
   - `the_stored_3d_note_says_the_slip_fields_feed_the_checks_not_the_limits` passes unchanged (the slip-heating
     fields move only the temperature verdict; the reverse fields' set gains the joint row, which its `contains`
     checks allow) and `the_stored_3d_label_and_note_say_the_fields_are_fixed_in_plain_words` passes with the new
     `STORED_3D_NOTE` sentence;
   - the panel's `the_dashboard_shows_the_stored_3d_label_and_the_corrected_markers` (`panel.rs:2842-2863`) still
     counts one drawn label; a new assertion there, or in a dashboard test, holds it after the joint row: the last
     stored-3D line is `shaft_joint.check`;
   - at `short_magnets()` the greyed joint row keeps `stored_3d`: its hover carries both notes, and the XLSX Summary
     note reads `GREYED_NOTE` (item 15).
7. **Status:** the defaults read "Example design: fails 4 of 9 checks" while paused and "fails 4 of 10 checks"
   unpaused; `passing_design()` "passes all 9" / "all 10"; `only_the_joint_fails()` "fails 1 of 9"; with the clamp also
   failing, the clamp still moves no count while paused.
8. **Problem:** `each_problem_is_some_exactly_when_its_check_fails` covers the joint; `the_problems_quote_numbers_from_the_results`
   gains a SHAFT_JOINT case that recomputes the window's upper edge from the two results; at hub minimum 10 the
   Problem carries 3.482, 7.647, 3.823, 12.09 and 18.82 (the window quoted inward, review SF-11: results 12.0815 and
   18.8281); at the 304 probe the empty-window sentence (16.31, 11.72) and none on a design with a window; at wheel
   shaft 20 °C the fitting sentence ("would need it at 247.6 °C with the wheel shaft at 20.00 °C, above the 100.0 °C
   allowed: your limit for the bonded inner hub, 100.0 °C", review MF-6 and SF-12, review 2 SF-6b, R-15) and "do not
   even open its 10.00 µm joining clearance";
   at `breaking(cool_adhesive)` the source names "the selected adhesive's design limit, 60.00 °C" and the window reads
   11.14 to 13.90 µm; at `breaking(n42_parts)` (owner R-15) the source names "the magnets' single-ring onset,
   69.83 °C, less 10.00 °C" after "above the 59.83 °C allowed", and the window reads 11.62 to 13.88 µm, the fitting
   edge binding; at `passing_design()` with N50M on manual dimensions, "above the 88.80 °C allowed: the magnets'
   single-ring onset, 98.80 °C, less 10.00 °C" and 12.21 to 17.45 µm (the onset and the margin quoted with the allowed
   temperature's decimals, section 5.2); with 2214 Hi-Temp, the hub input typed 150 °C and a 26 µm hub maximum, "above
   the 133.4 °C allowed: the magnets' single-ring onset, 143.4 °C, less 10.0 °C" (review 4, SF-5: the margin takes the
   allowed temperature's one decimal); so each of the three source forms is pinned; with mixed rings (owner R-15
   ring; review 5, SF-1), `passing_design()` with inner N42M and outer N35 on manual dimensions reads "above the
   81.35 °C allowed: the magnets' single-ring onset, 91.35 °C, less 10.00 °C", the inner ring's own onset, never the
   outer N35's 59.93 °C (C, `out.txt` 13.5); a window typed at its quoted ends passes (the band 12.09-18.82 at the defaults); on `joint`, a window
   narrower than the last shown digit quotes its ends with more decimals, in order; the no-wall sentences (Q ≥ 1 and a
   keyway through the wall) and the reversed-band sentence follow a determinate joint's causes (section 4.3); with a
   value that is not a number the Problem is that sentence alone; the plain-words, glyph, clamp and protect guards run
   over it.
   **No order of the torques** (review 2, MF-1), a pin like BL-059's `the_gearbox_why_never_bounds_the_torque_by_cold_high`
   (`guidance.rs:2848-2880`): `the_shaft_joint_texts_never_order_the_torques` sets the BL-067 corner on its sliders
   (`coupling.op_temp_C` −40, `metal.min_temp_C` 20, `metal.variation` 0.1), asserts there that the operating pull-out
   with its allowance (`model.pullout_Nm` × 1.1 = 3.657 N·m) is above cold-high (3.1823 N·m), and asserts that no
   shaft-joint text contains "strongest", "most expected" or "most torque expected" (case-insensitive): the 15 input
   labels and helps, the 24 result labels and helps, the verdict constants, the guide's label and Why, its Problem at
   the corner and at every design of this item and of 7.9, and the note's sentences, change entries and watch-outs.
9. **Fixes:** each PerTerm fix moves its term toward passing over its whole slider (`every_fix_moves_its_check_toward_passing_over_its_slider`),
   the hub minimum past the band's maximum included; `every_fix_is_a_design_input_with_a_slider_or_choices` passes with
   no assumption among the fixes; `each_failing_term_has_a_fix_that_moves_it_and_no_fix_turns_another_check_red` runs
   over `designs()` plus "joint slips" (`only_the_joint_fails()`), "joint overstressed" (`passing_design()`, hub maximum
   26: stress and fitting terms), "joint cannot be fitted" (`passing_design()`, wheel shaft 20 °C), "cup key, no
   bearing area" (`passing_design()`, boss length 0: the key term moves with Raise boss length, review MF-2) and
   "inner hub key, no bearing area" (`passing_design()`, `metal.hub_length_mm` 3.0: the slip moves with Raise hub
   minimum, Raise hub length and Raise face gap, the key with Raise hub length; review SF-2), each keeping every other
   check green (C, `out.txt` 9.3); on those two, "Raise key contact height" is not shown (section 5.2). The re-probed
   `designs()` take their new expectations:
   - `breaking(no_back_iron)`: both joints slip;
   - `breaking(long_magnets)`: the cup slips, key too;
   - `breaking(cool_adhesive)` (review MF-5): the inner hub's fitting term fails, and only that term (93.3 °C against
     the DP460's 60 °C; window 11.14-13.91 µm, the fitting edge binding); "Lower `shaft_joint.hub_interference_max_um`"
     is shown (one step: 93.27 -> 89.20 °C); "Lower `shaft_joint.wheel_shaft_fit_C`" is not, at its slider's bottom.
   - `breaking(n42_parts)` (owner R-15; this pin flips from round 3's pass): the inner hub's fitting term fails, and
     only that term, "Cannot be fitted at the allowed temperature (inner hub): needs 93.3 °C against 59.8 °C"
     (`temperature.demag.inner_onset_single_ring_C` 69.83 °C less 10 °C; window 11.62-13.89 µm, the fitting edge binding;
     the temperature check stays Caution; C, `out.txt` 11.2). "Lower `shaft_joint.hub_interference_max_um`" is shown
     (one step: −33.43 -> −29.37 °C); "Lower `shaft_joint.wheel_shaft_fit_C`" is not, at its slider's bottom; no other
     check turns red. The test pins the three numbers (93.27, 69.83, 59.83 °C) with a comment naming R-15.
   - Only `n42_parts` moves (C, `out.txt` 11.5): every other design of `designs()` reads the joint as before R-15.
     `breaking(hot_cure)`'s stored single-ring field of 700 kA/m puts its onset at 125.95 °C, a cap of 115.95 °C above
     the 100 °C input; `breaking(ferrite)`'s onset is +∞. `breaking(custom_hot_day)` (the inner part cleared: a
     gradeless manual inner ring, the outer B842SH governing) is the one `designs()` entry where the inner ring's own
     onset differs from the governing one, 153.28 against 143.36 °C; both caps are above the 100 °C input, so its
     joint reads the same (C 13.1).
   - The tests that use `breaking(n42_parts)` read the temperature guide alone and pass unchanged:
     `shown(breaking(n42_parts)) == [part_inner, part_outer]` (`guidance.rs:3562`), the level table (`:4280`,
     temperature Caution), the cooling rule (`:6686`) and `a_quoted_magnet_limit_says_what_it_is` (`:6741`, `:6763`).
     The joint is red there already, so no temperature fix can redden it.
   - **A manual grade** (owner R-15, a dedicated test like the near-limit one): `passing_design()` with N50M on manual
     dimensions in both rings reads "Cannot be fitted at the allowed temperature (inner hub): needs 93.3 °C against
     88.8 °C". The comment says why this grade: its onset (98.80 °C) is above the 93.27 °C need, and only its onset less
     10 °C (88.80 °C) is below it. "Lower hub maximum" is shown (−4.46 -> −0.40 °C). The cup wall fails there too, so
     the test asserts the joint's verdict, Problem and fixes, not the status (C, 11.3, 11.11, 11.14).
   The five tests that use `breaking(cool_adhesive)` pass unchanged (each reads the temperature guide alone, and the
   joint, red there already, is never reddened by its Change adhesive fix): `shown(breaking(cool_adhesive)) ==
   [adhesive]` (`guidance.rs:3564`), the level table (`:4282`, temperature Caution), the Problem text (`:5472`), the
   cooling rule (`:6686`) and the quiet-limit loop (`:6774`). **Fix held back** (review SF-10): `passing_design()` with
   the hub band 10-10 slips at the inner hub, and "Raise hub minimum" is not shown (one step reverses the band); Raise
   hub length and Raise face gap are. Its mirror (review 2, SF-2): `passing_design()` with DP460 (adhesive 4) and the
   hub band 18-18 fails on fitting alone; "Lower hub maximum" is not shown (one step, 17.5, reverses the band) and
   "Lower wheel shaft temperature" is at its slider's bottom, so the joint's block shows `NO_FIX`, and the Problem's
   window sentence names the band ("from 11.14 µm to 13.90 µm … (its band is 18.00 µm to 18.00 µm)"). **Near the
   joint limit** is a dedicated test, not a `designs()` entry (as BL-059's): the defaults with
   the cup minimum at `cup_needed_um` + 0.001 µm; one length step from the blank (12.71 mm) makes the cup slip (C: holds
   7.649 against 7.653 N·m, badge "holds 7.64 N·m against 7.65 N·m"), so `helps()` holds back the hot minimum's Raise
   length fix: the hot-low block shows `NO_FIX` and the joint stays green.
10. **Assumptions:** four new rows with explained results downstream (`tests/explain.rs:4460-4481`: 15 -> 19 paths, 18
    rows); `every_row_has_a_label_rationale_source_and_one_unit` with its widened source rule (a Rust-only source
    contains "Addendum A3" or "BL-064 shaft joint spec", review 2 MF-3; all four new sources pass it as drafted in
    section 4.5); none hidden while paused (`tests/assumptions.rs:324`).
11. **Note:** caps, sentence 0, plain words; each change entry's aim at the five probes; `release_notes_are_reviewed`;
    `every_note_names_its_committed_review_section`; Start here with the step before `clamp_preload`
    (`notes.rs:1656`, `tests/explain.rs:1212` FULL 15, `FULL[..14]`); the cap at 21;
    `notes_link_to_records_and_each_equation_has_at_most_one`; the docs say "21 notes, 20 shown while the clamp
    section is paused". The owner's "Cup temp" decision (review MF-1): the note's watch-outs contain "gearbox seal" and
    "quickly", and "liquid nitrogen", "wheel shaft" and "inner hub" (review 2, SF-6a). Sentence 4 (review MF-6, owner
    R-15) quotes `{shaft_joint.hub_fit_allowed_C}` beside `{shaft_joint.hub_fit_temp_C}` and names the three sources
    ("your limit", "adhesive's", "single-ring onset"); it contains `{shaft_joint.wheel_shaft_shrink_um}` (not
    `{shaft_joint.wheel_shaft_fit_C}`, an input), "wheel shaft" and "Here", and renders "on the wheel shaft shrunk
    18.99 µm" at the defaults (review 4, MF-2; review 5, MF-2); sentence 3 contains "here", "cup's" and no
    `{coupling.bore_mm}` (review 5, SF-4; review 6, SF-1); sentence 4's "10 °C" is
    `CURE_MARGIN_MIN_C` formatted (the test builds the words from the constant, so a change to the margin fails it);
    it renders "allowed: 59.83 °C" at `breaking(n42_parts)`, and "would need 266.3 °C on the wheel shaft shrunk
    -2.300 µm" on a 40 °C wheel shaft, the slider's top, the render the physics reviewer decides on (section 6.1;
    review 6, SF-4; C 14.4). `every_note_placeholder_names_a_number_result_and_renders_at_every_probe_design`
    (`tests/explain.rs:4840`, with `live_value_outside_a_here_clause`, `:4727-4740`) and the explorer's ship test
    (`explorer.rs:2910-2925`, `live_value(&results, path).is_some()`) pass for `shaft_joint`, so a later draft cannot
    bring back an input placeholder or drop the "here" clause unnoticed (review 5, MF-2). The same margin words, built from the constant the same
    way, are asserted in the `shaft_joint.hub_fit_max_C` help, the `shaft_joint.hub_fit_allowed_C` help and the note's
    Sources ("held 10 °C below"), which type them too, so a change to the margin fails all four (review 4, SF-3).
12. **Records:** the drift guard holds the 23 records and the inner ring's onset record (owner R-15 ring, section 4.5)
    to the engine at every probe and at a no-wall design (bore 16 mm, boss Ø12: the cup's four Lamé rows NaN), and the
    inner onset record also at the mixed rings of 7.1 (inner B842SH with outer B842, where it differs from the
    governing onset); the new chain is explained, and `every_explained_chain_drills_down_to_inputs` holds the
    `demagnetization` chain with its new path. The traceability checks run over the new
    records too (review 2, SF-12): `check_traceability` (`tests/explain.rs:4262`) and
    `each_assumption_moves_what_depends_on_it_and_nothing_else` (`:4461`), with `CANCELLATIONS` at `:4432`. The floors
    (`max(0, …)` on the effective interferences and the contact arc), the min and max over the two temperature ends, the
    keyway-depth case on b and the contact cap at the keyway depth can each leave an input on a record's active path
    that a nudge never moves (for example the rotor-over-shaft allowance where the cold end binds, or the contact height
    above the keyway depth). The implementer probes each record for these first and adds the `CANCELLATIONS` entries or
    record `cases` they need, before the gate finds them. R-15 adds one more such min: the allowed hub temperature
    reads the inner ring's onset, so that onset's inputs (the single-ring field, `coupling.mu0`, and the assumptions
    knee fraction and Br coefficient) lie on its record's path but move it only where the onset binds. At the defaults
    the input binds; at the trace point "grade, 11 harmonics, 6 poles, test at 35 °C" (`tests/explain.rs:4142-4150`,
    inner N52 on manual dimensions, the inner ring governing, so its own onset is the governing one, C 13.1) the onset
    binds (allowed 71.77 °C, C `out.txt` 11.13). The inner onset's own record moves with the knee fraction and the Br
    coefficient at the defaults (13.6), so it needs no entry. The knee fraction moves the allowed temperature there
    (71.75 / 71.80 °C for one step), as do the single-ring field (71.55 / 72.00 °C) and `coupling.mu0` (71.774 /
    71.776 °C); the Br coefficient does not (71.7748 °C at both steps), since the inner ring's grade sets its own
    coefficient (C, `out.txt` 12.5; review 4, SF-2). Unless another trace point has the onset binding on a library
    part, `calibration.alpha_br_per_C` -> `shaft_joint.hub_fit_allowed_C` and -> `shaft_joint.hub_fittable_max_um`
    need `CANCELLATIONS` entries with that reason. The implementer confirms that before adding them.
13. **Drawing:** the key is drawn in both states (`geometry.rs:981`: keys (1, 1), in the renamed test of section 5.4,
    review 2 SF-11); the key outline test
    (`geometry.rs:975-977`) reads `[6.7, -1.5]` and √22.75 for the 3 mm key (was `[6.7, -2.0]` and √21 for the clamp's
    4 mm key).
14. **Clamp guard:** `every_text_the_panel_shows_has_glyphs_and_none_names_the_paused_clamp` passes with the new
    sections, inputs, results and guide (automatic); `ALLOWED` stays empty.
15. **Exports:** CSV and JSON carry every `shaft_joint.*` input and result; the XLSX Summary lists 19 rows with the
    joint badge (`spreadsheet.rs:1195`, renamed for nineteen), the joint row's note column reading `STORED_3D_LABEL`
    at the defaults and on a green variant, and `GREYED_NOTE` at `short_magnets()` (owner R-15; greying wins,
    `spreadsheet.rs:341-347`); the results table shows the 23 numeric results in the
    chain group "Shaft joints" (between Geometry and Mass) and the check in the headline group (review MF-3).
16. **Session:** a pre-BL-064 design file and share link load with the new inputs at their defaults;
    `a_share_link_of_the_default_design_stays_short` with `< 2_700`.
17. **Pinned counts that move** (each confirmed by running; the BL-059 table at
    `docs/superpowers/plans/2026-10-16-bl059-gearbox-check.md:2684-2718` lists the same sites):
    - status texts "4 of 8" -> "4 of 9", "all 8" -> "all 9", "1 of 8" -> "1 of 9" while paused, every "of 9" ->
      "of 10" unpaused: `guidance.rs:5887-5941` (`the_status_text_reads_each_form` keeps its literal total), `:5967-5980`,
      `:5996-6023`, `dashboard.rs:2080`, `:2322`; `panel.rs` reads them live;
    - arrays: `DASHBOARD` 19, `ROW_LEARN` 19, `CHECKS` 32, `CHECK_GUIDES` 10, `END_EFFECT_ROWS` 9, `STORED_3D_ROWS` 4
      (owner R-15), `PACKAGE_LABELS` 14, `SECTION_LABELS` 19, `CHAIN_LABELS` 9 (`result_groups.rs:30`);
    - the stored-3D tests (owner R-15, item 6): the renamed membership test (`dashboard.rs:1506`), the stored-3D line
      count (`dashboard.rs:1628`, 3 -> 4), and the label's place after the joint row; the doc comments at
      `dashboard.rs:14-17`, `:205-207`, `:220-223` (the `STORED_3D_INPUTS` doc; review 4, SF-4) and on
      `dashboard_ui` (section 5.1);
    - the scope (review MF-3; owner R-15 ring): `tests/explain.rs:1526` (169 -> 193: the 23 shaft-joint paths and the
      inner ring's onset in the `demagnetization` chain, 18 -> 19 paths; C 13.9), the `scope.rs:12` doc ("169 paths"),
      the result groups test (`result_groups.rs:347-357`, `ids[..10]`; the inner onset's group, section 5.4), README's
      "the eight chains" (module table `:153`, test table `:674`) -> nine;
    - positions: `the_dashboard_starts_with_the_headline_in_order` (`paths[15..]`), `guidance.rs:3033-3065`;
    - the greyed list (`dashboard.rs:1612`) and `panel.rs:3151` (`checks.len()`, re-derived);
    - `spreadsheet.rs:1195-1222` (19 rows);
    - `trace.rs:194`: `(392, 1087)` -> `(416, 1112)` (23 shaft-joint records and the inner ring's onset record; 24
      shaft-joint results and the inner ring's onset; C 13.9 reads 392 and 1,087 at fb79095); README `:294` and the
      test table's "392 of the 1087" (`:674`);
    - inputs 172 -> 187, results 1,087 -> 1,112;
    - `temperature.rs`'s `each_ring_shows_its_own_block` (`:2128`) and the swapped rings (item 1, owner R-15 ring);
    - `inputs.rs:1021-1077` (the group names gain `shaft_joint`; the shown `shaft` rows gain the two key rows) and
      `:1086-1097`; `panel.rs:4500-4540`;
    - the "gearbox" input filter: `inputs.rs:1646-1676` (3 -> 4 matches, `shaft_joint.output_shaft_fit_C` last; the
      workflow runs 1 -> 2) and `panel.rs` `the_filter_box_shows_the_matching_inputs_alone_and_edits_no_design`
      ("3 of N inputs" -> "4 of N");
    - the assumption count (`tests/explain.rs:4469`) and the doc counts (`tests/explain.rs:1323-1413`);
    - Start here (`notes.rs:1605-1671`, `tests/explain.rs:1206-1258`), the hover test (`explorer.rs:1460-1471`), the
      sign-off test and sheet (`explorer.rs:2754-2932`: `AFTER_THE_SLICE`, `ADDED_AFTER_THE_SLICE`,
      `CLAMPS_PAUSED_LINE`; section 6.1), the note cap (`notes.rs:1407`);
    - the key outline (`geometry.rs:975-977`, review SF-14);
    - the session guard (`session.rs:578-583`);
    - stale texts the key move leaves (review 2, SF-11; no count moves, the words do): `INERT_INPUTS`' reason for
      `clamps.key_width_mm` (`tests/explain.rs:4340-4341`, "only the drawing reads it (Python drawing.py)") is false
      once the drawing reads `shaft_joint.key_width_mm`; the input stays inert, so only the reason is reworded (draft:
      "the paused shaft-clamp section's key width: nothing reads it while the shrink-fit check's own key width sets the
      drawn key; the key pressure divides by the contact height"); `geometry.rs`'s comment at `:448-450` and the test
      name at `:981` (section 5.4).
18. **Gates:** `MAGCOUPLING_PYTHON=<venv python> bash linkage-sim-rs/scripts/gate.sh` ends GATE PASS with gate 12 run
    (no SKIP); `cargo test --features app`; both clippy runs and fmt; `build_web.sh`; the gui-smoke calculator step
    (Magnets fit still the last row at 1280×800; a Why link opens a note; the input tooltip shows no spreadsheet
    details).

## 8. Docs and backlog

- `M/README.md`: the dashboard rows (19) and the status ("N of 9 checks" while paused, "of 10" unpaused; "the status
  line counts 9 guides"); "every one of the 32 checks"; the pause paragraph (`:459`) and `:46`, `:505` (the clamp no
  longer returns with BL-064; the shrink-fit check replaces it on screen); the note and step counts (21 notes, 20 shown
  while paused; its 14 steps, the full order 15); the trace totals (416 of 1,112, `:294` and the test table's
  "392 of the 1087"); the Python-parity paragraph (a Rust-only input group, and one Rust-only temperature result, the
  inner ring's single-ring onset); the `temperature.rs` module row (`:141`) adds the inner ring's own single-ring
  onset (`inner_onset_single_ring_C`, read by the shaft-joint check) (review 6, SF-3); the test table rows. Named rows (review SF-14): a module-table row for
  `src/engine/shaft_joint.rs` (as `:147` has for `gearbox.rs`: the inputs, the results, the rule of decision 2A, the
  order of causes, the temperature model) and one for `src/engine/explain/records/shaft_joint.rs`; a test-table row for
  `tests/shaft_joint.rs` (as `:661` has for `tests/gearbox.rs`); "the eight chains" -> "the nine chains" in the
  module table (`:153`, `result_groups.rs`) and the test table (`:674`); the scope's 193 paths where the README counts
  them. The stored-3D sentence (`:311`, owner R-15): "the temperature rows that read the stored 3D fields carry …"
  becomes the temperature rows and the shaft-joints row (through the inner magnets' single-ring onset); the test table's
  `STORED_3D_ROWS` line (`:674`, "exactly those that move with the 14 stored 3D inputs") stays true.
- `docs/ai/02-system.yaml`: the guidance-layer line (`:107`, nine dashboard checks, ten guides), the pause line (`:108`,
  the joint check is outside `SECTION`), the stored-3D line (`:109`: the shaft-joints row carries the label too, owner
  R-15), the Rust-only lines (`:89-90`: the first top-level Rust-only input group), the magcoupling status entry at
  `:486`; the invariants that the shrink-fit check reads cold-high (it greys with the end-effect rows) and the inner
  ring's own single-ring onset, `temperature.demag.inner_onset_single_ring_C`, never the governing ring's (it is a
  stored-3D row; the bonded inner hub's allowed fitting temperature is the lowest of the input, the adhesive's design
  limit and that onset less `CURE_MARGIN_MIN_C`).
- `docs/ai/03-structure.yaml`: a module entry for `shaft_joint.rs` modelled on `gearbox`; `addendum_only` +=
  `shaft_joint.rs`; `records/shaft_joint.rs`; CHECKS (32 design checks); CHECK_GUIDES (the nine dashboard checks and
  FIT_CHECK); "9 while the clamp section is paused"; NOTES (21 notes); the `tests:` line (`:74`) gains
  `shaft_joint.rs` (the BL-064 shaft-joint check end to end) beside `gearbox.rs` (review SF-14); `:67`, CHAIN_LABELS
  ("the scope's six chains, adhesive, mass" -> the scope's seven chains, the shaft joints among them); the dashboard
  entry's STORED_3D_ROWS (four rows: the three temperature rows and the shaft joints, owner R-15); the `temperature`
  entry (`:57`) adds the inner ring's own single-ring onset to its Rust-only demag results (review 6, SF-3).
- `docs/ai/04-memory.yaml` (`:7`, `:12`, `:15`: BL-064 done, the clamp paused for good); `docs/ai/05-update-tracker.md`:
  a new top entry (history entries are not rewritten).
- `docs/ai/backlog.yaml`:
  - **BL-064:** the acceptance amended to the owner's ruling (no adapter joint; the clamp stays paused; both joints
    checked; the keyway depth stays the input it is, recorded as open); notes: the owner decisions of section 2 (R-15
    among them: the bonded hub's fitting temperature capped at the single-ring onset less 10 °C, so lower grades
    read "cannot be fitted" and the row is a stored-3D row; and its ring ruling: the onset is the inner ring's own, a
    new temperature result; section 11), the open facts of section 2, "un-pausing the clamp would
    show two joint methods", and the default verdict; status `fixed` on merge.
  - **BL-060** (`:52`): "comes back with BL-064" -> paused with the clamp section for good.
  - **BL-066:** its acceptance's input-filter count ("move to four") becomes "move to five" (the joint adds a
    "gearbox"-labelled input).
  - **BL-067** (`:107-115`): the shrink-fit check reads cold-high too, and its texts claim no order of the torques
    (review 2, MF-1: the required-torque help defines cold-high without ranking it, and a pin keeps "strongest" and
    "most expected" out of every shaft-joint text); one fix of which pull-out to compare serves both checks.
  - **BL-056**'s notes (`:16`): reword the "BL-064" mention that says the clamp returns.
  - **New BL-069** (`risk: physics`): the plating model assumes electroless nickel (`materials.nickel.thickness_mm`
    0.015 mm, `materials.bores_over_dia_mm`, the 4140 record's `needs_plating` and the uncoated-low-alloy warning)
    while the owner finishes the 4140 hub and cup with UltraOx and masks the bore; decide how the calculator models the
    finish (thickness, its effect on the bore, the warning's words).
  - **New BL-070** (`risk: physics`): fretting and fatigue of both joints under the reversing slip swing (3.33e8
    cycles per life at the defaults (`metal.magnetic_cycles`), about 6.7e8 reversals; review 2, MF-2), beside BL-066;
    micro-slip at the part ends below the slip torque.
  - **New BL-071** (`risk: physics`): separate shaft diameters for the gearbox output and the wheel shaft once their
    sizes are known (a second bore input; the boss and web masses read `coupling.bore_mm`, `model.rs:1256`, `:1271`);
    the key row follows each bore.
  - **New BL-072** (`risk: physics`; the owner's ring ruling keeps it, section 11, R-15): the temperature verdict's
    cure margin reads the governing ring's single-ring onset (`temperature.demag.onset_single_ring_C`, the ring with the
    lower magnet limit), while each ring's own cure heats that ring's bond on its own carrier (the bare cup's magnets
    are bonded on the fitted cup, the inner hub's before fitting). Where the two rings differ, the onset read can be
    above a ring's own, so the cure margin passes a cure that reaches that ring's onset: over the 289 grade pairs on
    manual dimensions at `passing_design()`, the read is above the inner ring's own in 10 and above the lower of the
    two rings' own in 20, by up to 79.05 °C (inner N42H with outer SmCo 2:17-30 (`SmCo_2_17_30`); with 2214 Hi-Temp,
    cure 121 °C, the cure margin passes at 70.88 °C while against the inner ring's own onset it is −8.17 °C; C,
    `out.txt` 13.7). The fix, a cure margin against each ring's own single-ring onset (BL-064's
    `inner_onset_single_ring_C` is the inner half), is a temperature-module change, for the owner. The bonded hub's
    fitting cap is not BL-072's: it reads the inner ring's own onset already (owner R-15 ring).
- Code texts made stale by the key move (review 2, SF-11; tests and comments, not docs, listed here so the docs pass
  does not miss them): `INERT_INPUTS`' reason for `clamps.key_width_mm` (`tests/explain.rs:4340-4341`), the key comment
  at `geometry.rs:448-450` and the test name at `geometry.rs:981` (sections 5.4 and 7.17). R-15 adds the doc comments
  of section 5.1 (`dashboard.rs:14-17`, `:205-207`, `:220-223`, `dashboard_ui`).
- Do not edit `docs/superpowers/plans/*` or earlier specs (history).

## 9. Release

1. Red tests, implementation, both clippy runs, fmt, the touched suites. `risk: physics`: session-model implementers.
2. The physics review of the `shaft_joint` note and the G-3 review of every new text (section 6.3); must-fix items
   applied and re-reviewed.
3. A fresh-context session-model review of the whole branch.
4. `MAGCOUPLING_PYTHON=<venv python> bash linkage-sim-rs/scripts/gate.sh`: GATE PASS with gate 12 running (no SKIP).
5. `build_web.sh`, then gui-smoke on the local server (Magnets fit still the last dashboard row; a Why link opens a note).
6. **STOP:** send the owner a short summary of every changed on-screen text and number and the pinned counts that moved.
   Merge (`--no-ff`) and push (which deploys) only on the owner's OK; then watch the deploy and run gui-smoke on
   https://colesorkness.com.

## 10. Out of scope

- The paused clamp section, its key inputs, its note `clamp_preload`, BL-060 and the adapter variant (owner 6A; the
  adapter's Metal design inputs stay shown as an informational variant).
- Fretting and fatigue of the joints (BL-070), the gearbox's own shaft, bearings, seal and heat soak, the keyway corner's
  stress concentration (Kt about 2-3, from memory), the bond strain from fitting a bonded hub, how the liner and band
  are held when a rotor is heated, the rotor chill from a cryogenic shaft, frost on a cold shaft, and the key's shear
  (the flank binds first while the contact height is under about half the key width: τ = p·k/b against 0.577·R_e;
  DIN 6885-1 rows give 0.4).
- Curing the cup's magnets on the gearbox shaft (review SF-13). The owner's order fits the bare cup first and bonds its
  magnets after, so with a heat-cured adhesive the cure heats the cup, and through it the gearbox output, to the cure
  temperature: EA 9514 cures at 120 °C (60 min), 2214 Hi-Temp at 121 °C (`temperature.rs:148-163`), above the
  catalog's 90 °C (research-assembly 1.3). The default AA 326 cures at 22 °C. Not checked; R-4.
- The magnets' single-ring onset while the bonded hub is warmed for fitting is **in scope** since the owner decided
  R-15: the check caps the hub's allowed temperature at the inner ring's own onset less 10 °C, whichever ring
  governs (sections 3.1, 4.1; the ring ruling). Out of scope stay:
  - the temperature verdict's cure margin, which reads the governing ring's onset and can read above a ring's own
    (BL-072, section 8);
  - the time at temperature (the check compares temperatures, not a soak);
  - how far the onset's stored single-ring field is from this design's own (the row carries the stored-3D label).
- A press-fit route for the cup (research-assembly 5c), a fit-class picker, and any change to Auto.
- Separate shaft diameters (BL-071), the UltraOx finish model (BL-069), and which pull-out the checks compare when the
  operating temperature is below the minimum (BL-067).

## 11. Risks and what needs the physics review

- **R-1, smoothing on the stress side.** The maximum takes off the same smoothing as the minimum (synthesis 13). With a
  finer finish than assumed the real bore stress is higher: at smoothing 0 the cup's 15-22 µm band is at 401.9 MPa
  against 345.8. The physics review decides: keep, or use no smoothing (or a smaller one) on the stress side, which
  would make the cup's default band fail its stress limit (the cup's elastic edge falls to 18.86 µm). The note's two
  watch-outs are fixed (section 6.1, review MF-1), so the draft smoothing watch-out moved here: "The smoothing assumed
  is taken off the maximum too: a finer finish than assumed leaves more grip and more bore stress." (21 words). If the
  review keeps the shared smoothing, it says where that warning lives: the smoothing input's help ("It lowers both the
  grip and the bore stress"), the `fit_smoothing` rationale (section 4.5), or a watch-out in place of one of the two.
- **R-2, from memory, to confirm before any text quotes it:** the yield factor 1.2 (DIN 7190 practice, 1.1-1.2); SF 2
  for reversing torque (1.5 static); friction 0.14 (DIN 7190 / Roloff-Matek; the measured 0.145 and AGMA's 0.15 are
  verified); the 5 µm smoothing (0.8·ΣRz under DIN 7190:2001 is from memory; DIN 7190-1:2017's 0.4·Rz is a cited
  reference); the 0.001·d joining clearance; shaft E, ν and expansion (a generic alloy steel; the shaft materials are
  unknown); the NIST Fe-2.25Ni curve standing in for the shafts' steel (its coefficients are verified, the 5-10 %
  spread among ferritic steels is from memory); the key flank allowable (this spec's choice, section 2); the UltraOx
  process temperature (unknown; nitrocarburize-and-oxidize finishes run near 500-600 °C, from memory) and 4140's
  tempering temperature, which the 300 °C cup default and its 400 °C slider top must stay below.
- **R-3, constant expansion when the cup is warmed to 300 °C.** The part's expansion is the room-temperature value
  (12.3e-6); 4140's mean coefficient from 20 to 300 °C is higher (about 13e-6, from memory), so the needed temperature
  is over-stated: conservative. The physics review confirms.
- **R-4, the seal side.** The cup at 300 °C on the gearbox shaft puts heat toward the output seal and grease (the GAM
  catalog's range ends at 90 °C, research-assembly 1.3). The check does not model it; the note's first watch-out says
  to fit the bare cup quickly and keep its heat off the gearbox seal (section 6.1); the physics review checks the
  wording. The same side takes a second heat load the owner's order brings (review SF-13, section 10): with a
  heat-cured adhesive (EA 9514 at 120 °C, 2214 at 121 °C) the cup's magnets cure with the cup already on the gearbox
  shaft. The physics review says whether the note or the adhesive help should name it.
- **R-5, the first top-level Rust-only input group.** Probe the catalogue, `SECTION_LABELS`,
  `every_input_is_in_exactly_one_section_in_schema_order`, the session reader and the XLSX Inputs sheet with an empty
  group before writing the guide.
- **R-6, guidance designs that newly fail the joint:** `breaking(no_back_iron)` (both slip; touches
  `a_check_that_does_not_apply_gives_no_problem`), `breaking(long_magnets)` (the cup slips; touches
  `shown(breaking(long_magnets))` and its neighbours) and `breaking(cool_adhesive)` (review MF-5: DP460's 60 °C caps
  the bonded hub's allowed temperature, so the inner hub cannot be fitted, 93.3 against 60.0 °C, window 11.14-13.91 µm;
  its five tests read the temperature guide alone and pass unchanged, section 7.9) and, since the owner decided R-15,
  `breaking(n42_parts)` (the onset less 10 °C caps the hub at 59.83 °C, 93.3 against 59.8 °C; its tests read the
  temperature guide alone, section 7.9). Re-probe all of `designs()` and
  `two_failing_checks_that_pull_one_input_apart_say_so` (the cup's Lower length opposes the hot minimum's Raise length)
  and `raising_the_face_gap_says_it_lowers_the_failing_hot_low_torque` (the joint offers the face gap where it fails
  with hot-low).
- **R-7, fix hiding:** near the joint limit the hot minimum's length fix is held back; covered by the dedicated test.
- **R-8, NaN handling:** the unknown test must read the values, not only the terms (section 4.4), and it reads a
  determinate joint's compared terms too, so a NaN term from finite inputs (bore typed 0) is never passed by a false
  comparison (review 2, SF-4).
- **R-9, the default hub band is tight:** 0.83 µm under its fitting edge. Two 0.5 µm steps on the hub maximum (19 µm
  needs 101.4 °C), 17-4PH H900 (103.4 °C) or a DP460 adhesive (60 °C allowed) turn the row red on fitting; a 12L14 hub
  passes with 1.6 °C to spare (98.4 against 100 °C; review 2, SF-8; C, `out.txt` 10.7). That is the physics, not noise;
  the owner's real bands replace the placeholders.
- **R-10, the BL-067 corner:** the check reads cold-high, as the gearbox check does (BL-067).
- **R-11, parity:** gate 12 must run and not SKIP; no ported row is edited.
- **R-12, dashboard height:** one more row; Magnets fit stays last, so the smoke's scroll check holds.
- **R-13, un-pausing later** would show both the clamp and the shrink-fit check as joint methods (recorded in the
  `PAUSED` doc and in BL-064).
- **R-14, which body sits at the hot end** (review SF-4). Section 3.3's loosest term, Δ(T_hot) − α_part·ΔT_rotor·d,
  models the shaft at T_hot and the rotor ΔT_rotor above it, which puts the bonded rotor above its own bond limit. The
  bond-limited case puts the rotor at T_hot and the shaft at T_hot − ΔT_rotor: Δ(T_hot) − α_shaft·ΔT_rotor·d. C, Ø10,
  20 K, hot end 120 °C:

  | part | cold end Δ(−40 °C) | hot, this spec's form | hot, bond-limited form | loosest, this spec / bond-limited |
  |---|---|---|---|---|
  | 4140 (α 12.3e-6) | +0.48 | −3.26 | −3.10 | −3.26 / −3.10 µm (this spec 0.16 µm conservative) |
  | 416 (α 10.08e-6) | −0.852 | −0.596 | −0.880 | −0.852 (cold) / −0.880 (hot) µm (this spec 0.028 µm unconservative) |
  | 17-4PH H900 (α 10.8e-6) | −0.42 | −1.46 | −1.60 | −1.46 / −1.60 µm (0.14 µm unconservative) |
  | 304 (α 16.9e-6) | +3.24 | −8.78 | −7.70 | −8.78 / −7.70 µm |
  | 6061 (α 23.6e-6) | +7.26 | −16.82 | −14.40 | −16.82 / −14.40 µm |

  The rotor-over-shaft allowance alone takes 2.46 µm in this spec's form against 2.30 µm in the other for 4140 (2.02
  against 2.30 for 416). The physics review chooses; the bond-limited form moves 416's binding end from cold to hot, so
  the 7.1 pin "the 416 part's minimum is taken cold" and every hot-end number of sections 3.3 to 3.6 move with it.
- **R-15, the bonded hub's fitting temperature against the magnets' single-ring onset: decided by the owner in two
  rulings (2026-10-17, after spec reviews 3 and 5), before the plan.**
  - **The decision.** The bonded inner hub's allowed fitting temperature is the lowest of three: the owner's input
    (`shaft_joint.hub_fit_max_C`, 100 °C), the selected adhesive's design limit (120 °C for AA 326) and the inner
    ring's own single-ring onset (`temperature.demag.inner_onset_single_ring_C`, the ring ruling below) less
    `CURE_MARGIN_MIN_C` (10 °C). It is NaN when any of the three is NaN (sections 3.1, 4.1, 4.4). This was the second
    of round 3's options. The bare cup is unaffected: it is fitted before its magnets are bonded.
  - **Why, and its precedent.** The trap was "nothing suggests an unsafe assembly temperature" at reachable designs.
    The engine already keeps this margin for the adhesive cure: the temperature verdict requires the cure temperature
    to be at least `CURE_MARGIN_MIN_C` below the single-ring onset (`temperature.rs:1381-1387`; the result's label is
    "Onset, single ring during an adhesive cure", `:483`). Warming the bonded inner hub for fitting is the same heat
    soak of a single bonded ring, so the check keeps the same margin (DRY: the constant is read, not retyped), against
    the onset of the ring the fitting heats.
  - **What it corrects in round 3's record** (review 3, MF-1). The seven manual-dimension onsets (both rings, the
    defaults' geometry; E) are N35 59.93 °C, N42 69.83, N48 76.05, N50 78.22, N52 81.77, N42M 91.35 and N50M 98.80.
    Six are below the 93.27 °C need; N50M's 98.80 °C is above it, and only its onset less 10 °C (88.80 °C) is below
    it. N42M's 91.35 °C is 1.9 °C below. So the help's rule was broken at all seven, and the old check passed each but
    N52. Round 3 also costed the old help-only rule as "no risk". Its cost was the trap: the badge passed, note
    sentence 4 quoted "allowed: 100 °C", and the amber temperature check there was amber for a service reason (the
    magnet limit below the hot-day start; the cure margin at `n42_parts`, 47.83 °C, passes). Nothing on screen said
    the fitting would heat the hub's magnets past their onset.
  - **Consequences** (C, `out.txt` section 11, re-run with the inner ring's own onset and unchanged; 13.1):
    - At the defaults nothing changes (N42SH, onset 143.36 °C, cap 133.36 °C above the 100 °C input), nor at the five
      probes (11.1). Ferrite's onset is +∞, so the cap never binds there (the Y30 probe, `breaking(ferrite)`).
    - `breaking(n42_parts)` reads "Cannot be fitted at the allowed temperature (inner hub): needs 93.3 °C against
      59.8 °C" (11.2). `passing_design()` with each of N35, N42, N48, N50, N52, N42M and N50M on manual dimensions
      reads "cannot be fitted" with "needs 93.3 °C against" 49.9, 59.8, 66.1, 68.2, 71.8, 81.3 and 88.8 °C (11.3).
      At the defaults' geometry the same grades read the same, but N52's badge is its slip (9.48 against 9.51 N·m),
      with the fitting cause after it in the Problem.
    - Of the grade table on manual dimensions at `passing_design()`, every grade whose onset less 10 °C is under the
      100 °C input fails on fitting, and each of those already reads Caution on temperature (11.4: the seven above
      and the bonded NdFeB grade, onset 34.26 °C).
    - Engine: one link value, one helper and one NaN rule (section 4.1); the unknown test reads 15 engine values
      (section 4.4); the record gains the term (section 4.5); with the ring ruling, one temperature result with its
      record and scope path (sections 4.1, 4.5).
    - Dashboard: the row joins `STORED_3D_ROWS` (3 -> 4), carries the label, and the drawn label moves under it;
      `STORED_3D_NOTE` gains a sentence (section 5.1).
    - Texts: the `hub_fit_max_C` and `hub_fit_allowed_C` helps, the Problem's fitting source (three forms, each with
      its live number), note sentence 4 and trap 9 (sections 4.1, 5.2, 6.1, 6.2), and the new result's label and
      help. The fixes stay the band and the shaft temperature; the grade is not offered (section 5.2, "Where the
      onset binds").
    - Tests: 7.6 and 7.9 flip; 7.1, 7.2, 7.8, 7.11, 7.12, 7.15 and 7.17 gain the pins of section 7.
  - **The ring (owner, 2026-10-17, after spec review 5).** Review 4 found that `temperature.demag.onset_single_ring_C`
    is the governing ring's, the ring with the lower magnet limit (inner on a tie; E20, `temperature.rs:969-972`), not
    necessarily the inner ring the bonded hub carries. The hub is warmed for fitting alone, and the cup's magnets are
    bonded after the cup is fitted (owner, Process), so the hub's fitting heat reaches only the inner magnets. The
    owner chose to read the inner ring's own onset.
    - **What it adds** (section 4.1): one Rust-only result, `temperature.demag.inner_onset_single_ring_C`, beside
      `temperature.demag.inner_magnet_limit_C`, valued `inner.onsets[3]` (`temperature.rs:931` builds the inner block,
      `:1012-1016` write its per-ring results); its record in `records/demagnetization.rs` and its path in the
      `demagnetization` scope chain (section 4.5); the counts it moves, one each over BL-064's own: results
      1,087 -> 1,112, explained records 392 -> 416, the scope 169 -> 193 (C 13.9; section 7.17).
    - **What reads it** (review 5, SF-1): the link, the unknown test's value, the record term, the guide's call of the
      helper and the Problem's MagnetOnset placeholder (sections 4.1, 4.4, 4.5, 5.2), as do 3.1's ϑ_sr,i, 3.2's onset
      row, 5.1's membership reasoning and the path in 7.9's pin. The words of the texts stay as drafted and are true
      under this ruling: the helps' and the Problem's "the magnets'", the Sources' "its magnets'" and note sentence 4's
      "single-ring onset" name the bonded hub's own magnets.
    - **No pinned number moves** (C, `out.txt` 13.1). At the defaults, `passing_design()`, `only_the_joint_fails()`,
      the five probes, every `designs()` entry, the seven grades and N50M on manual dimensions, 2214 Hi-Temp with a
      typed 150 °C, the single-ring field at 0, 862 and 1500 kA/m and NaN, the trace point "grade, 11 harmonics, 6
      poles, test at 35 °C" and an uncooled wheel shaft (53 designs), the joint's verdict, causes, allowed temperature
      and fittable edge are the same with either onset. In 52 the inner ring governs or both rings are alike, so the
      two onsets are equal bit for bit. The one that differs is `breaking(custom_hot_day)`'s (a gradeless manual inner
      ring, the outer B842SH governing): 153.28 °C against 143.36 °C, both caps above the 100 °C input.
    - **What the governing ring's onset would have read, and the inner ring's own does not** (C, `out.txt` 12.1-12.3
      for the governing ring's, 13.2-13.5 for both; over the 289 grade pairs on manual dimensions at
      `passing_design()` the outer ring governs in 136, the onset read below the inner ring's own in 126 and above it
      in 10; each library part named is checked by `library::lookup`, review 5 MF-1):
      - **False reds, gone.** At the owner's defaults 72 of the 289 grade pairs read "Cannot be fitted … (inner hub)"
        only through the outer ring's onset, e.g. inner N42H with outer N42, "needs 93.3 °C against 59.8 °C" (the
        outer N42's 69.83 °C less 10). With the inner ring's own onset all 72 pass (inner N42H: 112.83 °C, cap
        102.83 °C, allowed 100 °C, your limit; 13.2). So did 36 of the 225 library pairs: an N42SH part inside
        (B842SH, the default, BX042SH or BX082SH) with any of the twelve other library parts outside, the onset read
        69.83 °C with an N42 outer, 81.77 °C with B842-N52 or B882-N52 and 59.11 °C with M5044, M5045 or M5026,
        where the inner part's own is 143.36 °C (13.4). All 36 now pass: `passing_design()` with inner B842SH and
        outer B842 reads "Both shaft joints hold and can be fitted", window 11.62-18.83 µm (13.5).
      - **Red either way, now with the inner ring's numbers.** In 28 more grade pairs and 41 library pairs the joint
        is red either way. The governing ring's onset made the Problem quote the outer ring's: inner N42M with outer
        N35 read "above the 49.93 °C allowed: the magnets' single-ring onset, 59.93 °C, less 10.00 °C". It now reads
        "above the 81.35 °C allowed: the magnets' single-ring onset, 91.35 °C, less 10.00 °C"; inner N50M with outer
        N42 reads 88.80 °C allowed (onset 98.80 °C) in place of 59.83 °C (13.2, 13.5); inner B842-N52 with outer
        B842, 71.8 °C in place of 59.8 °C (13.4).
      - **The false pass, gone.** With 2214 Hi-Temp, the hub input at its 130 °C slider top and a 22 µm hub maximum,
        the governing ring's onset passed inner N42H with outer SmCo 2:17-30 (`SmCo_2_17_30`) needing 125.79 °C,
        13 °C above the inner magnets' own onset of 112.83 °C (it read the outer SmCo's 191.88 °C). It now reads
        "Cannot be fitted at the allowed temperature (inner hub): needs 125.8 °C against 102.8 °C", source "the
        magnets' single-ring onset, 112.8 °C, less 10.0 °C" (13.3, 13.5), and the 71 pairs red there only by the
        onset read pass.
      - The joint now equals the joint read with the inner grade's own onset (both rings on it) at all 289 grade
        pairs in both settings and at all 225 library pairs (asserted in the crate, 13.2-13.4).
    - **The fixes.** Section 5.2's fourth reason (one ring's choice may not move the cap) goes: the inner part now
      moves the cap (`breaking(n42_parts)` with the inner part changed to B842SH passes, 13.5). The part or grade is
      still not offered, for 5.2's three reasons; where the inner ring's own onset binds (136 of the 289 grade pairs)
      the temperature check is Caution in all 136 and its guide offers the inner part or grade in 135 (13.2).
    - **What stays BL-072's** (section 8). The temperature verdict's own cure margin reads the governing ring's onset,
      the same gap from the cure's side: the read is above the inner ring's own in 10 of the 289 grade pairs and above
      the lower of the two rings' own in 20, by up to 79.05 °C (inner N42H with outer SmCo 2:17-30 (`SmCo_2_17_30`),
      where the cure margin passes at 70.88 °C with 2214 Hi-Temp while against the inner ring's own onset it is
      −8.17 °C; 13.7).
