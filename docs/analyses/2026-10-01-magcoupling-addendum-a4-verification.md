# Magcoupling Addendum A-4 verification: E21 (calibration offset clamped at 0) and per-grade magnet thermal properties (E22, E23)

Date: 2026-10-01. Basis: `main` at 05ec40a. Status: numbers for the user's approval; nothing is applied. On 2026-10-01 the user approved both corrections in principle (`docs/ai/04-memory.yaml`): "the E20 rating-calibration offset is clamped at 0", and "ferrite, SmCo and bonded grades get sourced per-grade magnet CTE and specific heat, registered as corrections that move only non-NdFeB designs". Machine-readable twin: `docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json`.

## Summary

- **E21 is ready for approval.** Its numbers reproduce, and a skeptic checked them independently on two bases (2,045 items agreed). The clamp is the conservative choice for the demagnetization check. The demag outputs C58, C60, C12 and C15 never rise in any configuration, pair or differential case. The default design, every grade or part with an offset of 0 or more, and every pair in which neither ring's offset is negative are bit-identical.
- **One E21 finding should change the landing plan.** Under E20 the cure margin C24 belongs to the governing ring. In 66 ordered pairs on the registry basis and 38 on the user basis, the clamp moves the governing ring to the one with the higher single-ring onset. The shown C24 then overstates the real minimum, by up to 62.4 °C (N38UH beside Recoma 26). The same E20 defect already exists without E21, up to 79.1 °C (Recoma 30 beside N42H). Decision 2 recommends landing C24 = MIN(both rings' single-ring onsets) − cure together with E21, as its own correction.
- **The property split works as named.** E22 is the inner ring's grade CTE in the bond plane. It changes C104, C105, C106, C201 and C202, plus warning rule 6. E23 is each ring's grade specific heat. It changes C141 and the 25 thermal cells downstream of it. The two cell sets do not overlap. Neither correction moves the default design, any sintered NdFeB grade, any library part or any of the 3,391 differential cases.
- **E22 matters and E23 barely does.** E22 changes the bond-screen readings. On an aluminium hub (user basis), the SmCo and Y30 rings clear the CTE warning. C106 goes from "Above" to "Below" in 9 of the 10 same-grade and inner-only configurations of the five non-NdFeB grades; Y30 on both rings stays "Above" (15.96 against 15 MPa). E23 moves the heat capacity C141 by −5.9 % (Recoma 26 and Recoma 30) to +11.6 % (Y30, both on an aluminium hub), and the peak temperature by at most 0.031 °C. Neither correction changes the verdict C25 anywhere.
- **Property values.** Six of the ten engine-consumed values are resolved: Y30's CTE and specific heat, Recoma 20's CTE and specific heat, and the specific heat of Recoma 26 and Recoma 30. Four are open, in three decisions. The first is the Sm2Co17 perpendicular CTE (Recoma 26 and Recoma 30): Arnold prints 13e-6, and four independent sources print 11 to 12e-6. No reading, warning or verdict differs between 13 and 11. The second is the bonded CTE, where the proxy sources are 4x apart. I recommend leaving it at C95's −0.8e-6, which is the most conservative value. The third is the bonded specific heat, which has a single source; I recommend 420, the conservative side of 440.
- **Ten decisions** are in section 5. Each lists the recommended option first.

## 1. Purpose and method

**What this report verifies.**
- E21: the numbers in the E21 verification (`a4-verify/e21.md`), with the six disagreements raised by the E21 skeptic.
- The per-grade CTE and specific heat for Y30, Recoma 20, Recoma 26, Recoma 30 and bonded NdFeB (BCN-19): the sourced table (`a4-verify/props.md` and `props.json`), with the six disagreements raised by the property skeptic.
- What E22 and E23 would change. No input gave these numbers, so I computed them (section 4).

**Inputs.**

| Input | What it holds | Skeptic result |
|---|---|---|
| `a4-verify/e21.md` + `e21.patch` | E21 patch, harness, 95 configurations × 2 bases, 964 ordered pairs, 3,391 differential cases, knee sweep | 2,045 items agreed; 6 disagreements (section 2.7) |
| `a4-verify/props.md` + `props.json` | the sourced property table: CTE ⊥, CTE ∥ and cp per grade, with URLs, tiers and flags TH1 to TH8 | 6 of 14 value items agreed on independent confirmation; 6 disagreement entries cover the other 8 (section 3.4) |

**Computation copy (`a4-verify/crate-props`).**
- It is the repository's `magcoupling-rs` at 05ec40a with `e21.patch` applied. The src and tests directories are identical to the E21 crate (`a4-verify/crate`). The repository was not modified.
- I added scratch E22 and E23 on top (`a4-verify/e22_e23_scratch.patch`, not for landing):
  - `DeviationId::E22` and `E23` in `ALL`, with no registry entries.
  - `grades::magnet_thermal`, a table of the proposed values for the five non-NdFeB grades, with a per-thread override for the alternatives. A sintered NdFeB grade returns none.
  - E22: `temperature::bond_plane_cte_per_C` returns the inner ring's grade CTE when E22 is on and the inner grade is non-NdFeB, else C95. Both consumers read it: Δα at `temperature.rs` (the Volkersen screen) and `magnet_cte_per_C` at `api.rs` (warning rule 6).
  - E23: the magnets' heat-capacity term is m_inner · cp_inner + m_outer · cp_outer, each ring at its grade's cp when non-NdFeB, else C138. It is used in both E15 branches. If both rings have the same cp, the term stays C110 · cp, bit for bit (the A2-7 density pattern). The per-ring masses N · V · ρ are passed through two new link fields.
- Harness `tests/a4_props.rs`:
  - Configurations: the default design; each of the 17 grades on both rings, inner only (beside B842SH) and outer only; the 15 library parts on both rings; and all 272 ordered grade pairs.
  - Each configuration runs on two hubs (steel circuit `coupling.backiron = 1`, the default; no back iron `= 0`, an aluminium hub) and two bases:
    - **registry**: NONE → NONE.with(E22), and NONE → NONE.with(E23);
    - **user**: ALL without E22 and E23 → ALL without E23 (E22 alone), ALL without E22 (E23 alone), and ALL (both).
  - Every result row is compared bit for bit, Rust-only fields and warnings included. Cell names come from `cell_values_for`.
- Assertions, all passing:
  - nothing moves unless a ring is in the move set (E22: a non-NdFeB inner ring; E23: a non-NdFeB ring on either side);
  - everything in the move set moves;
  - wiring: E22 with a non-NdFeB inner ring equals E22 off with C95 set to the grade value, and E23 with one grade on both rings equals E23 off with C138 set to the grade value, on every result row, both hubs and both bases;
  - the 3,391 differential cases are bit-identical: NONE against only(E22) and only(E23), and the user basis without both against ALL.

**Gates on crate-props.**
- `cargo test` (debug): parity.rs 4 passed, differential.rs 19 passed, every other suite passed, and the harness passed 7 of 7.
- Two registry tests fail, as expected, because E22 and E23 have no registry entries: `registry_is_in_report_order_with_one_entry_per_id` (21 entries for 23 ids) and `each_deviation_alone_changes_exactly_its_registered_cells` (index out of range).
- clippy and fmt were not run on the scratch code.

**Conventions.** These follow `e21.md`. "a -> b" is the value without the correction, then with it; a single value is bit-identical. Tables use 4 significant figures. Probe values are the shortest round-trip `f64`. Both bases are labelled throughout.

## 2. E21: the calibration offset clamped at 0

### 2.1 The physics

- The model puts each demagnetization onset where the reverse field H(T) = H20 (1 + α ΔT) meets the knee k · Hcj20 (1 + β ΔT).
- E20's calibration then works in two steps:
  1. It finds the model's onset t_ref (C49) for a reference magnet at permeance coefficient 1 (H_ref = Br / 2μ0, C48).
  2. It shifts every onset by the offset C50 = C49 − C47, where C47 is the vendor's rating.
- A positive offset lowers the onsets. That is conservative, however rough the rating is.
- A negative offset raises every onset to match the rating. That needs the rating to be accurate and on the model's basis: Pc = 1, the same knee criterion and the same Hcj minimum. This has not been shown for any grade.
- The NdFeB ratings are class figures by suffix: N = 80 °C, M = 100, H = 120, SH = 150, UH = 180, EH = 200, AH = 220. In the three large NdFeB cases, an independent rating sides with the model:
  - N52: Arnold 60 °C, Eclipse 70 °C;
  - N50: SuperMagnetMan 60 °C;
  - N50M: Eclipse 90 °C.
- Recoma 26's sheet itself says its 350 °C "may be considerably lower at low load line".

**What E21 does.**
- It sets C50 = MAX(C49 − C47, 0) for each ring, before E20 picks the weaker ring.
- It is gated on E20 and E21 together.
- `< 0.0` keeps a NaN offset as NaN, and keeps −0.0 and every offset of 0 or more bit for bit.
- The cold side (positive β) and unrated magnets already have an offset of 0.
- Clamping the offset is bitwise identical to taking, for each field, the lower of the calibrated and the uncalibrated onset (the skeptic checked this on every clamped configuration).

**Corrected wording (skeptic).** E21 computes the lower of two unvalidated estimates. It is conservative for the demagnetization check (C58, C60, C12 and C15 never rise), but not uniformly downstream:
- The mismatch screen relaxes where the magnets come to govern (section 2.9).
- The shown C24 can rise under a governing-ring flip (section 2.8).
- The uncalibrated arm uses the stored reverse fields C52 to C55, which do not scale with Br. For a 1.45 T N52 ring they are likely low, so the uncalibrated onset can itself be optimistic.

### 2.2 Where it binds

The offset before the clamp, for each ring in the grade mode on both rings, with every correction but E21 on. Only the negative rows move.

| Grade | Br (T) | α(Br) (/°C) | C47 rating (°C) | C49 t_ref (°C) | C50 offset (°C) | A-1 basis (α −0.0012) |
|---|---|---|---|---|---|---|
| N52 | 1.45 | -0.0012 | 80 | 70.31 | **-9.689** | -9.689 |
| N50 | 1.41 | -0.0012 | 80 | 73.86 | **-6.138** | -6.138 |
| N50M | 1.41 | -0.0012 | 100 | 92.46 | **-7.535** | -7.535 |
| N38UH | 1.22 | -0.0012 | 180 | 172.7 | **-7.343** | -7.343 |
| N35EH | 1.17 | -0.0012 | 200 | 196.4 | **-3.574** | -3.574 |
| Recoma 26 | 1.00 | -0.00035 | 350 | 289.8 | **-60.21** | -18.55 |
| Recoma 30 | 1.09 | -0.00035 | 250 | 249.6 | **-0.4500** | +46.06 |
| N35, N42, N48, N42M, N42H, N42SH, N33AH | | -0.0012 | | | +6.459 to +30.75 | unchanged |
| Recoma 20 | 0.85 | -0.00045 | 250 | 467.3 | +217.3 | +255.0 |
| BCN-19 | 0.65 | -0.0014 | 140 | 234.3 | +94.28 | +89.92 |
| Y30 | 0.37 | -0.0020 | 250 | +inf (cold side) | 0 | n/a |

- Decision A2-7 gave each grade-mode ring its own α(Br), and that moved the non-NdFeB offsets after A-1: Recoma 26 went from −18.55 to −60.21 °C, and Recoma 30 changed sign. The `04-memory.yaml` item that quotes the A-1 numbers needs refreshing when E21 lands.
- Library parts that move: B842-N52 and B882-N52 (N52) on both bases. The arcs M5044, M5026 (N50, rating 80) and M5045 (N50M, rating 100) move on the registry basis only; with E19's 60 °C rating their offset is +12.98 °C.

### 2.3 Changed cells at report precision

Configurations that E21 changes; every other configuration is bit-identical. The grade mode gives the same values on both bases. "Cells" is the number of workbook cells that change, registry / user. On the user basis E12 already holds C19, C23 and C150 to C153 at 0 for a start above the limit.

| Configuration (both rings) | C50 offset | C56 aligned | C58 skipping | C59 single ring | C60 magnet limit | C12 governing | C15 hot-day margin | C24 cure margin | C25 | Cells |
|---|---|---|---|---|---|---|---|---|---|---|
| default design B842SH (registry / user) | 10.43 / 9.921 | 169.7 / 170.2 | 102.6 / 103.1 | 142.9 / 143.4 | 92.55 / 93.06 | 92.55 / 93.06 | 27.55 / 28.06 | 120.9 / 121.4 | OK | 0 / 0 |
| N52 grade; B842-N52; B882-N52 | -9.689 -> 0 | 127.0 -> 117.3 | 10.17 -> 0.4787 | 81.77 -> 72.09 | 0.1679 -> -9.521 | 0.1679 -> -9.521 | -64.83 -> -74.52 | 59.77 -> 50.09 | CHECK | 23 / 17 |
| N50 grade | -6.138 -> 0 | 123.4 -> 117.3 | 6.617 -> 0.4787 | 78.22 -> 72.09 | -3.383 -> -9.521 | -3.383 -> -9.521 | -68.38 -> -74.52 | 56.22 -> 50.09 | CHECK | 23 / 17 |
| N50M grade | -7.535 -> 0 | 129.8 -> 122.3 | 51.90 -> 44.37 | 98.80 -> 91.27 | 41.90 -> 34.37 | 41.90 -> 34.37 | -23.10 -> -30.63 | 76.80 -> 69.27 | CHECK | 23 / 17 |
| N38UH grade | -7.343 -> 0 | 192.3 -> 185.0 | 141.9 -> 134.6 | 171.9 -> 164.6 | 131.9 -> 124.6 | 120.0 | 55.00 | 149.9 -> 142.6 | OK | 14 / 14 |
| N35EH grade | -3.574 -> 0 | 209.1 -> 205.5 | 165.4 -> 161.9 | 191.3 -> 187.8 | 155.4 -> 151.9 | 120.0 | 55.00 | 169.3 -> 165.8 | OK | 14 / 14 |
| Recoma 26 grade | -60.21 -> 0 | 365.6 -> 305.4 | 171.9 -> 111.7 | 287.2 -> 227.0 | 161.9 -> 101.7 | 120.0 -> 101.7 | 55.00 -> 36.73 | 265.2 -> 205.0 | OK | 24 / 24 |
| Recoma 30 grade | -0.4500 -> 0 | 283.1 -> 282.6 | 56.27 -> 55.82 | 191.9 -> 191.4 | 46.27 -> 45.82 | 46.27 -> 45.82 | -18.73 -> -19.18 | 169.9 -> 169.4 | CHECK | 23 / 17 |
| M5044, M5026 (registry only) | -7.023 -> 0 | 124.3 -> 117.3 | 7.502 -> 0.4787 | 79.11 -> 72.09 | -2.498 -> -9.521 | -2.498 -> -9.521 | -67.50 -> -74.52 | 57.11 -> 50.09 | CHECK | 23 / 0 |
| M5045 (registry only) | -8.132 -> 0 | 130.4 -> 122.3 | 52.50 -> 44.37 | 99.40 -> 91.27 | 42.50 -> 34.37 | 42.50 -> 34.37 | -22.50 -> -30.63 | 77.40 -> 69.27 | CHECK | 23 / 0 |

- **Beside B842SH, both orders.** No flip. Where the clamped ring is the weaker one, both orders show the same block, with the same C50, C58, C60 and C12 as above; only C61 differs with ring order. N38UH, N35EH and Recoma 26 beside B842SH change nothing, because B842SH's limit (92.55 °C registry, 93.06 °C user) still governs.
- **Every ordered pair** (272 grade pairs plus 210 part pairs per basis):

| Basis | Pairs | Moved | Governing ring flips | Verdict changes |
|---|---|---|---|---|
| registry | 482 | 214 | 71 | 0 |
| user | 482 | 150 | 39 | 0 |

- **Ties.** Only N52 beside N50 is an exact tie (C58 = 0.4787157208430415 for both after the clamp), and it resolves to the inner ring. Recoma 26 (101.73342351672972 °C) beside Recoma 20 (101.74043173174721 °C) is not a tie: with Recoma 20 inner, the block flips from inner to outer.
- **Knee fraction C46.** Below a threshold even the default part's offset turns negative, and E21 binds. For N42SH the threshold is 0.7834925165904798, so C46 = 0.7835 is not clamped (offset +0.000742 °C). With the slider step of 0.01, the first clamped setting is 0.78. At C46 = 0.70 the default design's verdict goes from OK to CHECK (C12 74.81 -> 65.42 °C, user basis). Y30 never binds, so it is absent from the threshold table.

### 2.4 Changed-cell list (corrected)

Cells that E21 changes in any configuration or pair, with the number of the 95 configurations per basis (registry / user).

| Cell | Result | Configs | When it changes |
|---|---|---|---|
| C50 | calibration offset | 30 / 21 | always (the clamp): negative -> 0 |
| C56 to C59, C7 to C9 | onsets and their summary copies | 30 / 21 | always: each falls by the clamped amount |
| C60, C10 | magnet design limit | 30 / 21 | always |
| C61 | pull-out torque at the limit | 30 / 21 | always (rises: the limit is colder) |
| C24 | cure margin | 30 / 21 | always; under a flip it can rise (section 2.8) |
| C181, C182 | margins to the skipping onset and the magnet limit | 30 / 21 | always |
| C12, C13, C15 | governing limit and its margins | 28 / 19 | where the magnets govern or come to govern |
| F12 | which limit governs | 1 / 1 | Recoma 26: "Adhesive governs." -> "Magnets govern (skipping case)." |
| C19, C150, C151, C152 | unbroken slip time and rotations | 27 / 0 | registry only (E12 zeroes them on the user basis) |
| C23, C153 | critical drag torque | 28 / 1 | as C12; on the user basis only Recoma 26 |
| C101, C104, C105 | worst swing, peak end shears | 1 / 1 | Recoma 26 (swing hot end 98.0 -> 79.73 °C) |
| C106 | mismatch-screen reading | 0 / 1 | Recoma 26, user basis: "Above" -> "Below" (section 2.9) |
| C201 | daily peak shear | 1 (1 ulp) / 0 | rounding only (C105 · C200 / C101 cancels C101) |
| C42, **C43**, **C47**, C48, C49 | the shown ring's Br, **α(Br)**, **rating**, h_ref, t_ref | 0 / 0 | only on a governing-ring flip. C43 and C47 change in 6 ordered pairs per basis: N38UH or N35EH beside Recoma 26, and Recoma 26 beside Recoma 20, both orders (C47 180, 200 or 250 -> 350 °C; C43 -0.0012 or -0.00045 -> -0.00035) |

- Rust-only identity fields `demag_ring`, `hcj20_used_kA_m` and `beta_used_per_C` also change under a flip.
- Never changed: C44 and C45 (inputs), C11, C14, C16, C25, the cold-side results, and every cell outside the Temperature design sheet.
- **Record correction.** The PROVISIONAL registry entry lists 31 cells. It omits C43 and C47, so `addendum_entries_name_every_cell_their_probes_change` would fail on a probe such as N38UH beside Recoma 26. Add both before landing, for 33 cells.

### 2.5 Probes (full precision, registry basis `NONE.with(E20)` -> `.with(E21)`)

| Probe | Cell | E20 | E20 + E21 |
|---|---|---|---|
| 1. B842-N52 on both rings (library path) | C50 | -9.689222777921628 | 0.0 |
| | C58 | 10.16793849876467 | 0.4787157208430415 |
| | C12 | 0.16793849876466993 | -9.521284279156959 |
| 2. Recoma 26 (`SmCo_2_17_26`), manual dimensions, both rings | C50 | -60.21319132809839 | 0.0 |
| | C12 | 120.0 | 101.73342351672972 |
| | F12 | "Adhesive governs." | "Magnets govern (skipping case)." |
| 3. B842 inner (N42), B842-N52 outer: ring flip | C42 | 1.3 | 1.45 |
| | C50 | 12.681126305465526 | 0.0 |
| | C12 | -3.5174216150834887 | -9.521284279156959 |
| | C24 | 47.83256962065032 | 50.08556444987687 |
| 4 (proposed). N38UH inner, Recoma 26 outer: flip that names C43 and C47 | C43 | -0.0012 | -0.00035 |
| | C47 | 180.0 | 350.0 |
| | C12 | 120.0 | 101.73342351672972 |
| | C24 | 149.91967217479288 | 205.01249772124206 (142.57635668906815 under decision 2 A) |

**Source-0 check (E21 report section 1, recomputed in crate-props).** N42's Hcj 954.9 kA/m and β −0.0062 /°C are typed into C44 and C45 on B842SH, with the coercivity source set to 0.

| Basis | C50 | C58 | C12 | C15 | C24 | C25 |
|---|---|---|---|---|---|---|
| registry | -56.55 -> 0 | 75.71 -> 19.16 | 65.71 -> 9.164 | 0.7119 -> -55.84 | 117.1 -> 60.51 | CHECK |
| user | -57.32 -> 0 | 76.48 -> 19.16 | 66.48 -> 9.164 | 1.483 -> -55.84 | 117.8 -> 60.51 | **OK -> CHECK** |

On the user basis this input also flips the verdict. The E21 report did not mention it. It supports E21: without the clamp, the typed N42 coercivity would read OK on a 150 °C rating.

### 2.6 Defaults unchanged

- The default design (B842SH on both rings) is bit-identical on every result field, Rust-only fields included, on both bases. Its offset is +10.43 °C (workbook Br) or +9.921 °C (E3).
- Every grade and part with an offset of 0 or more is bit-identical on both rings, beside B842SH in both orders, and in every pair in which neither ring is negative. That is 139 of the 190 configurations.
- `Deviations::only(E21)` is bit-identical to NONE at the defaults and on all 3,391 differential cases (double gate).
- On the E21 crate: `cargo test` 325 passed and 0 failed; clippy `-D warnings` and fmt `--check` are clean. The skeptic re-ran these on its own copy.

### 2.7 Skeptic verdict and dispositions

The skeptic reimplemented the clamp independently from `ring_demag`, with its own harness. It reproduced every number the E21 report quotes:
- all 1,514 values in section 5 and all 170 in section 4;
- all 276 probe values at full precision;
- the pair counts 482/214/71/0 and 482/150/39/0;
- the differential counts 637 and 349.

It agrees that option A is right for the demagnetization check. Its six disagreements:

| # | Item | Disposition |
|---|---|---|
| 1 | C43 and C47 missing from the cells list | Accepted. Add both, for 33 cells; proposed probe 4 (section 2.5) pins them. |
| 2 | C24 under a ring flip: the report deferred it to a follow-up | Accepted. The defect is wider than reported (section 2.8). Decision 2 recommends landing the fix with E21. |
| 3 | "A conservative bound" | Accepted. The wording becomes "the lower of two unvalidated estimates; conservative for the demag check, not uniformly downstream" (section 2.1). The C106 relaxation is decision 3. |
| 4 | "Two ties resolve to the inner ring" | Accepted. Only N52/N50 is a tie; Recoma 26/Recoma 20 is a flip (section 2.3). Wording only. |
| 5 | "At C46 ≤ 0.7835 the default design is clamped" | Accepted. The threshold is 0.78349, so the first clamped slider setting is 0.78. Wording only. |
| 6 | Option B2 (cap each onset at the rating) "touches in practice only C56" | Accepted. For N52, h_ref 576.9 kA/m is above the single-ring field (569 kA/m), and C59 = 81.77 °C is above the 80 °C rating, so B2 would also cap C59 and, through it, C24. B2 still never touches C58 (863 kA/m is above every h_ref), so the rejection stands. |

### 2.8 Cure margin C24 under E20

- **Why it happens.** Under E20 the shown block, C24 included, is the governing ring's: the ring with the lower magnet limit C60. C24 = C59 − cure, where C59 is the single-ring onset. The governing ring does not always have the lower C59. Because C59 is set by a ring's own field, Hcj, β, α and offset, it does not depend on the partner ring. So the minimum over both rings can be tabulated from the both-rings configurations. Cure is 22 °C (AA 326).

| Pair (either order unless stated) | Shown C24 without E21 | Shown C24 with E21 | MIN(both C59) − cure, with E21 | Overstatement with E21 |
|---|---|---|---|---|
| N38UH / Recoma 26 | 149.9 | 205.0 | 142.6 | **62.44** (created by E21) |
| N35EH / Recoma 26 | 169.3 | 205.0 | 165.8 | 39.25 (created) |
| Recoma 26 / Recoma 20 | 169.7 | 205.0 | 169.7 | 35.27 (created) |
| Recoma 30 / N38UH | 169.9 | 169.4 | 142.6 | 26.85 (was 19.96) |
| Recoma 30 / N35EH | 169.9 | 169.4 | 165.8 | 3.66 (was 0.54) |
| B842 / B842-N52 (probe 3), and the other N42 parts beside the N52 parts or arcs | 47.83 | 50.09 | 47.83 | 2.25 (created) |
| Recoma 30 / N42H (pre-existing E20) | 169.9 | 169.4 | 90.83 | 78.60 (was 79.05) |
| Recoma 30 / N42SH (pre-existing E20) | 169.9 | 169.4 | 121.4 | 48.07 |

- **Counts.** E21 creates an overstatement in 66 ordered pairs (registry) and 38 (user); all of them are pairs whose C24 rises. It worsens 4 more. Among the pairs E21 moves, the same E20 defect already exists in 16, up to 79.05 °C.
- **The fix.** C24 = MIN(C59_inner, C59_outer) − cure.
  - At default inputs it would flip no verdict. The lowest single-ring C24 of any grade or part is 12.26 °C (BCN-19), above the 10 °C threshold.
  - It would change C24 under E20 even where E21 does not bind. So it should be its own correction (proposed id E24, gated on E20), landed in the same change as E21. That keeps E21's bit-identity claim exact.
  - The display onsets C7 and C8 stay the governing ring's block; the label already says so.

### 2.9 Mismatch-screen relaxation (C106) and its interplay with E22

- **Mechanism.** The worst swing is C101 = MAX(cure − cold limit, C12 − cure). When the clamp makes the magnets govern, C12 falls, the swing shrinks, and so do C104, C105 and C106.
- **Extent.** On the user basis with a steel hub, C106 goes from "Above the lap-shear strength at the block ends" to "Below" in 9 cases. The shear is 16.11 -> 13.11 MPa against AA 326's 15 MPa in every case:
  - Recoma 26 on both rings;
  - Recoma 26 beside N38UH, N35EH, N33AH or Y30, in both orders.

  On an aluminium hub, none flip.
- **With E22 on.** 3 of the 9 remain: N38UH, N35EH or N33AH inner with Recoma 26 outer. E22 does not touch an NdFeB inner ring. A SmCo inner ring already reads "Below" under E22, because Δα is negative (Recoma 26 on a steel hub: C104 −0.7004 MPa). On an aluminium hub, E22 adds one E21-driven flip: Y30 inner beside Recoma 26 outer, 15.96 -> 12.98 MPa.
- **Reading.** It follows from the workbook's definition: the design is rated to C12, so the bond is screened over the temperature range the design allows. E21 exposes this; it did not create it. See decision 3.

### 2.10 Recommendation

Approve E21 as specified (option A: C50 = MAX(C49 − C47, 0) per ring, gated on E20; C50 shows the applied offset, with the help text "E21: clamped at 0; the rating never raises the onsets"), with these landing corrections:
- add C43 and C47 to the cells list, and probe 4;
- use the corrected wording in sections 2.1 and 2.3;
- land the C24 fix in the same change (decision 2);
- refresh the A-1 offsets in `04-memory.yaml`;
- give E21 an approval record of its own: this report's approval line, and either an `Approval` variant or an Addendum section that `every_entry_is_approved_in_its_report` checks;
- the remaining E21 report section 9 items: the A-3 explain records, porting `e21_without_e20_changes_nothing`, the README and the tracker.

## 3. Per-grade magnet thermal properties

### 3.1 What the engine consumes

- **CTE, perpendicular to the magnetization (the bond plane).**
  - The engine magnetizes radially through the block thickness (`r_mid = inner_back_apothem + thickness/2`), and the bond area is length × width. So the bond plane is perpendicular to M, and C95 is labelled "Across the magnetization".
  - Only the inner ring's grade is consumed. The hub bonds the inner ring, and the outer ring's bond to the cup is not CTE-screened.
  - There are two consumers: the E18 Volkersen screen (Δα = hub CTE − magnet CTE; hub 4140 at 12.3e-6, or the aluminium body at 23.6e-6 under E18) and warning rule 6 (|hub − magnet| > 15e-6).
  - The two consumers gate the hub differently. The warning takes the body CTE whenever C6 ≠ 1; the screen does so only when E18 is on. On the registry basis with an aluminium hub, the screen therefore uses steel while the warning uses aluminium.
- **Specific heat.** A scalar in J/(kg·K), with no temperature dependence. C141 = (C110 · C138 + the other parts) / 1000. C110 covers both rings, each at its own density (decision A2-7). A per-grade cp must therefore split the same way, m_i · cp_i + m_o · cp_o.

### 3.2 The values, with citations

Units: CTE in 1e-6/K (equal to 1e-6/°C and µm/(m·°C)); cp in J/(kg·K). Tier 1 = manufacturer datasheet (a proxy grade or vendor where marked); 2 = distributor or materials service; 3 = aggregator read through a summary. "Engine" marks the values the engine reads.

| Grade | Quantity | Value | Range printed | Primary source (tier) | Other sources | Status |
|---|---|---|---|---|---|---|
| Y30 (`Y30`) | CTE ⊥ (engine) | **10** | none | Eclipse ferrite datasheet p3, C⊥ 10 (2; the sheet that names Y30/C5); MS-Schramberg HF 26/24 and HF 30/26, p.p.d. 10-11 (1, proxy grade) | thyssenkrupp ferrite factsheet p3, "normal to mag. direction" 9.2-10 (2) | resolved: the only value inside all three ranges |
| | CTE ∥ (not read) | **null** (range 12-15) | none | MS-Schramberg i.p.d. 12-13 (1, proxy) | Maruwa (ferrite manufacturer) 14-15; Eclipse C// 15; Vega 14; thyssenkrupp 9.2-13.3 | recorded only: two manufacturers disagree; not consumed |
| | cp (engine) | **700** | none | MS-Schramberg HF 26/24 and HF 30/26, "approx. 700" (1, proxy) | Maruwa 630-840; thyssenkrupp 500-800; Eclipse 795-855 | resolved: primary, bracketed by 2 of 3 others, and the conservative side (heats faster) |
| Recoma 20 (`SmCo_1_5_20`) | CTE ⊥ (engine) | **14** | 20-200 °C | Arnold Recoma 20 sheet, Recoma-Combined-160301 p4, column "C ⊥" (1) | Eclipse 13; sourcemagnets.com 15 (3) | resolved (skeptic agreed) |
| | CTE ∥ (not read) | **7** | 20-200 °C | Arnold p4 (1) | Eclipse 6 | resolved (skeptic agreed) |
| | cp (engine) | **370** | 20-150 °C | Arnold p4 (1) | Eclipse 334.9; Allstar 370 | resolved (skeptic agreed) |
| Recoma 26 (`SmCo_2_17_26`) and Recoma 30 (`SmCo_2_17_30`) | CTE ⊥ (engine) | **13** | 20-200 °C | Arnold Recoma 26 p8, 26HE p9, 30 p12 and 30HE p13, column "C ⊥" (1) | thyssenkrupp SmCo factsheet p3: 11; Eclipse C⊥ 11; Dexter 2:17 11.0-12.0; magnet-materials.com 11; Allstar (20-100 °C) and Vega 11.8, generic SmCo | **open: decision 6** (13 by rule 3; 11 is the conservative alternative) |
| | CTE ∥ (not read) | **11** | 20-200 °C | Arnold (1) | thyssenkrupp 8; Eclipse 8-10; Dexter 9.0-10.0; magnet-materials.com 8 | recorded: Arnold kept by rule 3, flagged; not consumed |
| | cp (engine) | **350** | 20-150 °C | Arnold p8, p12 (1); 26HE p9 the same | Eclipse 376.8 | resolved (skeptic agreed) |
| Bonded NdFeB (`Bonded_NdFeB_BCN19`) | CTE (engine) | proposed 4.8; **recommended none** (keep C95 −0.8) | 25-200 °C (EAM) | EAM compression-bonded Nd physical properties p1, 4.8 µm/(m·°C), all five grades, no direction (1, proxy vendor) | HGT bonded NdFeB Table III: 1-2 (1); Vega bonded Nd: about 20 (2). Alliance BCN-19 prints none | **open: decision 7** |
| | cp (engine) | **420** | "room temperature", no number | EAM 0.42 W·s/(g·°C) × 1000 (1, proxy vendor) | no second source. Plausibility: Arnold sintered N35, N52, N42SH print 0.11 cal/(g·°C) = 460.5 | **open: decision 8** (single source) |

Engine literals (written as typed, never converted at load time):

| Grade | Bond-plane CTE | cp |
|---|---|---|
| Y30 | `10.0e-6` | `700.0` |
| Recoma 20 | `14.0e-6` | `370.0` |
| Recoma 26, Recoma 30 | `13.0e-6` (decision 6 B: `11.0e-6`) | `350.0` |
| BCN-19 | none: C95 applies (decision 7 B: `4.8e-6`) | `420.0` (decision 8 B: C138 applies) |

### 3.3 Direction and temperature range

- **Direction.**
  - Every source that prints a direction labels it against the magnetization:
    - Arnold "C //" and "C ⊥", where C is the easy axis;
    - Eclipse C// and C⊥;
    - MS-Schramberg p.p.d. and i.p.d.;
    - thyssenkrupp "in magnetizing direction" and "normal to mag. direction".
  - The order is consistent: perpendicular above parallel for Sm2Co17 and SmCo5, and parallel above perpendicular for ferrite. No source inverts it.
  - EAM prints one undirected value for an isotropic powder.
  - Maurer's undirected 5.6 (SmCo 2:17) and 8.5 (ferrite) are low-tier outliers and are not used.
- **Range.** The engine treats both properties as constants. The swing they act over runs from the cold limit or the cure temperature to the governing limit: 22 °C to 101.7 or 120 °C for SmCo, and 22 to 120 °C for Y30.
  - Arnold's CTE is a 20-200 °C mean, and its cp a 20-150 °C mean. CTE rises with temperature, so a 20-200 °C mean is likely higher than the 20-120 °C mean the engine needs. Arnold 13 against Allstar 11.8 (20-100 °C) fits that, which is the case for decision 6 B.
  - No ferrite source prints a range (flag TH8), and none is borrowed.

### 3.4 Skeptic disagreements: resolved or open

The skeptic rendered 15 cited documents and confirmed every cited number, footnote, direction column and unit conversion. Six of 14 value items agreed on independent confirmation:
- Y30 CTE ⊥ 10;
- Recoma 20 CTE ⊥ 14, CTE ∥ 7 and cp 370;
- Recoma 26 and 30 cp 350.

The other eight items fall under six entries:

| # | Item | Table | Skeptic found | Resolution |
|---|---|---|---|---|
| 1 | Sm2Co17 CTE ⊥ (engine) | 13 (Arnold, 20-200 °C) | 11-12 from four independent sources (thyssenkrupp, Eclipse, Dexter, magnet-materials); 15.4 % below 13 | **Open: decision 6.** Rule 3 (primary over secondary) gives 13: Arnold is the only grade-specific, range-stated source. 11 is the conservative alternative: it gives a larger \|Δα\| on both hubs. No reading, warning or verdict differs between them (section 4.6). |
| 2 | Sm2Co17 CTE ∥ (not read) | 11 (Arnold) | 8-10 | Resolved by rule 3: Arnold 11 recorded, flagged. Not consumed. |
| 3 | Y30 CTE ∥ (not read) | 12.5 (midpoint of MS-Schramberg 12-13) | 14-15 (Eclipse, Maruwa, Vega) | Resolved: no single value. Two manufacturers disagree (MS 12-13, Maruwa 14-15), so the range 12-15 is recorded and the midpoint dropped. Not consumed. |
| 4 | Y30 cp (engine) | 700 (MS-Schramberg, proxy grade) | Eclipse 795-855 (14-22 % above); Maruwa 630-840 and thyssenkrupp 500-800 bracket 700 | Resolved: 700. It is the primary value, it is supported by 3 of 4 sources, and it is the conservative end. The Eclipse alternative is in decision 5 B (sensitivity: section 4.6). |
| 5 | Bonded CTE (engine) | 4.8 (EAM proxy) | HGT 1-2, EAM 4.8, Vega about 20: three proxies 4x apart, none BCN-19. HGT's rejection as a "sintered template" is only partly supported (its density row is bonded-scale). At 20e-6 the aluminium-hub warning would not fire | **Open: decision 7.** By rule 4 the conservative option is the lowest CTE (largest Δα): C95's −0.8e-6, close to HGT's 1-2. Recommended. |
| 6 | Bonded cp (engine) | 420 (EAM proxy) | confirmed as printed; no second source exists; Arnold sintered NdFeB 460.5 is 8.8 % above | **Open: decision 8.** 420 is single-source but on the conservative side of C138's 440. Recommended, flagged. |

Not accessible to either pass: MatWeb (Cloudflare), ScienceDirect and ResearchGate (403), and TDK FB (403). No search-tool summary was used as a source.

### 3.5 NdFeB keeps its values

The 12 sintered NdFeB grades (N35, N42, N48, N52, N50, N42M, N50M, N42H, N42SH, N38UH, N35EH, N33AH) and every library part keep C95 = −0.8e-6 and C138 = 440. The harness asserts this bit for bit (section 4.2). For comparison only, Arnold's sintered sheets print C ⊥ −1 and C // 7 (20-200 °C) and cp 460.5 (20-140 °C). No NdFeB value is proposed.

## 4. What E22 and E23 would change

### 4.1 Scope and reach

| Correction | What it changes | Reach (who moves) | Cells, registry basis | Cells, user basis |
|---|---|---|---|---|
| E22 | magnet CTE in the bond plane: the inner ring's grade value for a non-NdFeB grade, else C95 | a non-NdFeB **inner** ring only; an outer-only non-NdFeB ring moves nothing | C104, C105, C106, C201, C202, plus warning rule 6 (Rust-only) | the same 5, plus rule 6 |
| E23 | each ring's specific heat: its grade value for a non-NdFeB grade, else C138 | a non-NdFeB ring on **either** side | 27: C19, C20, C141, C143, C145, C150 to C152, C154 to C161, C171, C172, C180 to C182, C186, C189, C190, C192, C193, C196 | 26 (C152 does not move; E12 handles a start above the limit) |

- The two cell sets do not overlap, so each correction can be probed and approved on its own.
- Some cells move only in some configurations. C150 to C152 and C19 move only where the time to the limit is a number. C181 does not move for Y30, whose skipping onset is +inf.
- **Pairs.** Of the 272 ordered grade pairs, E22 moves 80 (the 5 non-NdFeB inner grades × 16 partners) and E23 moves 140 (every pair with at least one non-NdFeB ring). The counts are the same on both hubs and both bases, and every pair in the move set moves.
- **These counts use the proposed table** (bonded CTE 4.8e-6 and cp 420). Under the recommended decision 7 A, bonded leaves E22's move set: E22 then reaches 4 grades and 64 pairs, and the bonded E22 rows in section 4.3 become no-change (section 4.6, row −0.8). Under decision 8 B, which is not recommended, E23 would drop to 116 pairs.
- **The verdict C25 never changes.**
  - E22 cannot change it by construction, because the verdict has no shear term.
  - E23 could in principle, through the peak temperature C180 in the margins C182 and C190. It changes no verdict in any configuration or pair, because C180 moves by at most 0.031 °C (Y30 on both rings, aluminium hub).
  - No other text result changes under E23 (C185, C197 and F12 are untouched).

### 4.2 Defaults unchanged

All of these were asserted by the harness and passed:
- The default design is bit-identical on every result field, both hubs, both bases and every correction set.
- Every sintered NdFeB grade on both rings, inner only and outer only, and every library part on both rings, is bit-identical.
- Every pair with no non-NdFeB ring is bit-identical (E23), and every pair with an NdFeB inner ring is bit-identical (E22).
- None of the 3,391 differential cases moves: NONE against only(E22) and only(E23), and the user basis without both against ALL. No case selects a grade.
- parity.rs (4 tests) and differential.rs (19 tests) pass on crate-props.

### 4.3 E22: the bond screen per grade (both rings, proposed values)

AA 326 lap shear is 15 MPa. The daily screen threshold is 15 × 0.2 = 3 MPa. On the registry basis the screen uses steel on either hub (E18 off) and the workbook adhesive modulus (E1 off), which gives the large NdFeB baseline of 64.07 MPa. Inner-only configurations (beside B842SH) show the same flips except where noted below; their shears are lower wherever B842SH governs.

| Grade (inner ring) | CTE ⊥ (1e-6/K) | Basis | Hub | C101 swing (°C) | C104 peak shear, current (MPa) | C105 recommended (MPa) | C106 reading | C201 daily (MPa) | C202 daily screen | Warning rule 6 |
|---|---|---|---|---|---|---|---|---|---|---|
| Y30 | 10 | registry | steel | 98.00 | 64.07 -> 11.25 | 37.13 -> 6.519 | **Above -> Below** | 11.37 -> 1.995 | **Above -> Below** | silent |
| Y30 | 10 | registry | aluminium | 98.00 | 64.07 -> 11.25 | 37.13 -> 6.519 | **Above -> Below** | 11.37 -> 1.995 | **Above -> Below** | **fires -> silent** |
| Y30 | 10 | user | steel | 98.00 | 16.11 -> 2.829 | 8.373 -> 1.470 | **Above -> Below** | 2.563 -> 0.4500 | Below | silent |
| Y30 | 10 | user | aluminium | 98.00 | 28.63 -> 15.96 | 15.21 -> 8.476 | Above | 4.655 -> 2.595 | **Above -> Below** | **fires -> silent** |
| Recoma 20 | 14 | registry | steel | 98.00 | 64.07 -> -8.315 | 37.13 -> -4.818 | **Above -> Below** | 11.37 -> -1.475 | **Above -> Below** | silent |
| Recoma 20 | 14 | registry | aluminium | 98.00 | 64.07 -> -8.315 | 37.13 -> -4.818 | **Above -> Below** | 11.37 -> -1.475 | **Above -> Below** | **fires -> silent** |
| Recoma 20 | 14 | user | steel | 79.74 | 13.11 -> -1.701 | 6.813 -> -0.8841 | Below | 2.563 -> -0.3326 | Below | silent |
| Recoma 20 | 14 | user | aluminium | 79.74 | 23.30 -> 9.166 | 12.37 -> 4.869 | **Above -> Below** | 4.655 -> 1.832 | **Above -> Below** | **fires -> silent** |
| Recoma 26 | 13 | registry | steel | 98.00 | 64.07 -> -3.424 | 37.13 -> -1.984 | **Above -> Below** | 11.37 -> -0.6073 | **Above -> Below** | silent |
| Recoma 26 | 13 | registry | aluminium | 98.00 | 64.07 -> -3.424 | 37.13 -> -1.984 | **Above -> Below** | 11.37 -> -0.6073 | **Above -> Below** | **fires -> silent** |
| Recoma 26 | 13 | user | steel | 79.73 | 13.11 -> -0.7004 | 6.812 -> -0.3640 | Below | 2.563 -> -0.1370 | Below | silent |
| Recoma 26 | 13 | user | aluminium | 79.73 | 23.30 -> 10.12 | 12.37 -> 5.375 | **Above -> Below** | 4.655 -> 2.022 | **Above -> Below** | **fires -> silent** |
| Recoma 30 | 13 | registry | steel | 98.00 | 64.07 -> -3.424 | 37.13 -> -1.984 | **Above -> Below** | 11.37 -> -0.6073 | **Above -> Below** | silent |
| Recoma 30 | 13 | registry | aluminium | 98.00 | 64.07 -> -3.424 | 37.13 -> -1.984 | **Above -> Below** | 11.37 -> -0.6073 | **Above -> Below** | **fires -> silent** |
| Recoma 30 | 13 | user | steel | 62.00 | 10.19 -> -0.5446 | 5.297 -> -0.2831 | Below | 2.563 -> -0.1370 | Below | silent |
| Recoma 30 | 13 | user | aluminium | 62.00 | 18.11 -> 7.869 | 9.621 -> 4.180 | **Above -> Below** | 4.655 -> 2.022 | **Above -> Below** | **fires -> silent** |
| Bonded (BCN-19) | 4.8 (proposed) | registry | steel | 62.00 | 40.54 -> 23.21 | 23.49 -> 13.45 | Above | 11.37 -> 6.507 | Above | silent |
| Bonded (BCN-19) | 4.8 (proposed) | registry | aluminium | 62.00 | 40.54 -> 23.21 | 23.49 -> 13.45 | Above | 11.37 -> 6.507 | Above | fires |
| Bonded (BCN-19) | 4.8 (proposed) | user | steel | 62.00 | 10.19 -> 5.835 | 5.297 -> 3.033 | Below | 2.563 -> 1.467 | Below | silent |
| Bonded (BCN-19) | 4.8 (proposed) | user | aluminium | 62.00 | 18.11 -> 13.96 | 9.621 -> 7.413 | **Above -> Below** | 4.655 -> 3.587 | Above | fires |

Exceptions and counts:
- Inner-only differs from both rings in two places. Y30 inner beside B842SH on the user basis with an aluminium hub reads 20.76 -> 11.57 MPa, so C106 goes Above -> Below; on a steel hub it reads 11.68 -> 2.051 MPa, already Below. The SmCo inner-only shears are lower, because B842SH governs (swing 71.06 °C), but the flips are the same.
- Discrete flips over the 272 pairs:
  - registry: C106 and C202 flip in 64 pairs on each hub, and warning rule 6 in 64 pairs on the aluminium hub;
  - user, steel hub: C106 flips in 3 pairs;
  - user, aluminium hub: C106 in 77, C202 in 64 and warning rule 6 in 64.

### 4.4 E23: the thermal network per grade (proposed values, user basis)

Steel hub unless stated. The rings use the default manual dimensions, so inner-only and outer-only give the same masses and the same values ("one ring"). A value shown as "a -> a" differs below 4 significant figures.

| Grade | cp (J/(kg·K)) | Rings | C141 heat capacity (J/K) | C143 time constant (s) | C180 peak (°C) | C182 margin to magnet limit (°C) | C190 adhesive margin (°C) | C150 time to limit, high (s), aluminium hub | C25 |
|---|---|---|---|---|---|---|---|---|---|
| Y30 | 700 | both rings | 76.57 -> 83.22 | 255.2 -> 277.4 | 65.98 -> 65.96 | 184.0 -> 184.0 | 54.02 -> 54.04 | never | CHECK |
| Y30 | 700 | one ring | 79.38 -> 82.71 | 264.6 -> 275.7 | 65.97 -> 65.96 | 27.08 -> 27.09 | 54.03 -> 54.04 | 653.6 -> 689.6 | CHECK |
| Recoma 20 | 370 | both rings | 84.22 -> 81.21 | 280.7 -> 270.7 | 65.96 -> 65.97 | 35.78 -> 35.77 | 54.04 -> 54.03 | never | OK |
| Recoma 20 | 370 | one ring | 83.21 -> 81.70 | 277.4 -> 272.3 | 65.96 -> 65.97 | 27.09 -> 27.09 | 54.04 -> 54.03 | 695.1 -> 678.8 | OK |
| Recoma 26 | 350 | both rings | 83.99 -> 80.17 | 280.0 -> 267.2 | 65.96 -> 65.97 | 35.77 -> 35.76 | 54.04 -> 54.03 | never | OK |
| Recoma 26 | 350 | one ring | 83.09 -> 81.18 | 277.0 -> 270.6 | 65.96 -> 65.97 | 27.09 -> 27.09 | 54.04 -> 54.03 | 693.9 -> 673.1 | OK |
| Recoma 30 | 350 | both rings | 83.99 -> 80.17 | 280.0 -> 267.2 | 65.96 -> 65.97 | -20.14 -> -20.15 | 54.04 -> 54.03 | 0 | CHECK |
| Recoma 30 | 350 | one ring | 83.09 -> 81.18 | 277.0 -> 270.6 | 65.96 -> 65.97 | -20.14 -> -20.15 | 54.04 -> 54.03 | 0 | CHECK |
| Bonded (BCN-19) | 420 | both rings | 78.37 -> 77.78 | 261.2 -> 259.3 | 65.97 -> 65.98 | -193.5 -> -193.5 | 54.03 -> 54.02 | 0 | CHECK |
| Bonded (BCN-19) | 420 | one ring | 80.28 -> 79.99 | 267.6 -> 266.6 | 65.97 -> 65.97 | -193.5 -> -193.5 | 54.03 -> 54.03 | 0 | CHECK |

Check: Y30 on both rings gives 25.565 g × (700 − 440) / 1000 = +6.647 J/K, and Recoma 26 on one ring gives 21.22 g × (350 − 440) / 1000 = −1.910 J/K. Both match C141.

### 4.5 Ring placement

- E22 reads only the inner ring, so a ferrite or SmCo outer ring beside an NdFeB inner ring leaves the bond screen and the warning at the NdFeB value. This follows from the workbook: the hub bonds the inner ring, and the cup bond is not screened.
- E23 reads both rings.

### 4.6 Alternatives for the open values

Both rings. Registry and user bases; E22 alone on the registry basis.

| Grade | CTE ⊥ (1e-6/K) | Registry, steel hub: C104 / C106 / C202 | User, steel hub: C104 / C106 | User, aluminium hub: C104 / C106 / C202 / warning 6 |
|---|---|---|---|---|
| Recoma 26 | 13 (Arnold) | 64.07 -> -3.424 / **Above -> Below** / **Above -> Below** | 13.11 -> -0.7004 / Below | 23.30 -> 10.12 / **Above -> Below** / **Above -> Below** / **fires -> silent** |
| Recoma 26 | 12 | 64.07 -> 1.467 / **Above -> Below** / **Above -> Below** | 13.11 -> 0.3002 / Below | 23.30 -> 11.07 / **Above -> Below** / **Above -> Below** / **fires -> silent** |
| Recoma 26 | 11 (cluster) | 64.07 -> 6.358 / **Above -> Below** / **Above -> Below** | 13.11 -> 1.301 / Below | 23.30 -> 12.03 / **Above -> Below** / **Above -> Below** / **fires -> silent** |
| Recoma 30 | 13 (Arnold) | 64.07 -> -3.424 / **Above -> Below** / **Above -> Below** | 10.19 -> -0.5446 / Below | 18.11 -> 7.869 / **Above -> Below** / **Above -> Below** / **fires -> silent** |
| Recoma 30 | 11 (cluster) | 64.07 -> 6.358 / **Above -> Below** / **Above -> Below** | 10.19 -> 1.011 / Below | 18.11 -> 9.354 / **Above -> Below** / **Above -> Below** / **fires -> silent** |
| Bonded (BCN-19) | -0.8 (C95, no change) | 40.54 / Above / Above | 10.19 / Below | 18.11 / Above / Above / fires |
| Bonded (BCN-19) | 1.5 (HGT 1-2) | 40.54 -> 33.42 / Above / Above | 10.19 -> 8.403 / Below | 18.11 -> 16.41 / Above / Above / fires |
| Bonded (BCN-19) | 4.8 (EAM) | 40.54 -> 23.21 / Above / Above | 10.19 -> 5.835 / Below | 18.11 -> 13.96 / **Above -> Below** / Above / fires |
| Bonded (BCN-19) | 20 (Vega) | 40.54 -> -23.83 / **Above -> Below** / **Above -> Below** | 10.19 -> -5.991 / Below | 18.11 -> 2.673 / **Above -> Below** / **Above -> Below** / **fires -> silent** |

- **Sm2Co17 at 13, 12 or 11.** The readings, the daily screen and the warning are the same in every case. The aluminium-hub Δα is 10.6, 11.6 or 12.6e-6, all under the 15e-6 limit. The decision is low-stakes: the largest difference is 1.9 MPa against a 15 MPa strength.
- **Bonded.** The choice flips C106 on an aluminium hub (4.8 and 20 clear it) and, at 20, the warning and C202 as well. At 20e-6 the steel-hub registry shear is −23.83 MPa: larger in magnitude than the 15 MPa strength, yet it reads "Below" (section 4.7, IN1).
- **Specific heat** (steel hub, both rings; aluminium hub, one ring for C150):

| Grade | cp (J/(kg·K)) | C141 / C143 / C180 | C150 time to limit (s) |
|---|---|---|---|
| Y30 | 700 (MS-Schramberg) | 76.57 -> 83.22 / 255.2 -> 277.4 / 65.98 -> 65.96 | 653.6 -> 689.6 |
| Y30 | 825 (Eclipse midpoint) | 76.57 -> 86.41 / 255.2 -> 288.0 / 65.98 -> 65.95 | 653.6 -> 707.0 |
| Bonded (BCN-19) | 420 (EAM) | 78.37 -> 77.78 / 261.2 -> 259.3 / 65.97 -> 65.98 | 0 |
| Bonded (BCN-19) | 440 (C138, no change) | 78.37 / 261.2 / 65.97 | 0 |

### 4.7 Implementation notes for landing

- **IN1, signed comparison.**
  - C106 tests `s1 > lap_shear`, and C202 tests `daily < lap_shear × endurance`, both with signed values.
  - Without E22, s1 is positive at every slider setting. C95 is at most 6e-6, below the lowest library hub CTE of 9.0e-6. The swing is at least 2 °C, because the cold limit is at most 20 °C, below the lowest cure temperature of 22 °C.
  - E22 makes Δα negative for every SmCo grade on a steel hub, and for Y30 on a 9.0e-6 hub. A negative shear of any size then reads "Below".
  - With the proposed values no reading is wrong: wherever the shear is negative, |C104| ≤ 8.31 MPa and |C201| ≤ 1.48 MPa. At the Vega alternative it would be wrong (−23.83 MPa). Decision 9 covers this.
- **IN2, inputs bypassed.** C95 (range −3e-6 to 6e-6) and C138 (300 to 600) cannot carry 10e-6 to 14e-6 or 700, and E22 and E23 bypass them for non-NdFeB rings. Decision 10 covers the display.
- **IN3, modulus.** The same screen reads C97 = 160 GPa for every grade. The sheets print 140 GPa (Recoma) and 150 GPa (MS-Schramberg ferrite). With E22 on, the shear at the sheet modulus differs by less than 1 % and is lower, so 160 is conservative; examples:
  - aluminium hub, Recoma 26: 10.12 -> 10.05 MPa;
  - Y30: 15.96 -> 15.91 MPa.

  No reading changes. Recorded as a carry-over; no decision needed now.
- **IN4, constants.** Each property is one constant per grade, while the sources print means over 20-200 °C, 20-150 °C or 25-200 °C, or no range at all.
- **Registry and tests.**
  - Each correction needs a registry entry, with cells as in section 4.1, the probes below, and an approval record citing this report.
  - Add tests that NdFeB and library parts are untouched, that the wiring matches the inputs (E22 = C95 at the grade value; E23 = C138 for same-grade rings), and that outer-only rings are inert under E22.
  - Add the per-grade values to the `Grade` record (two `Option<f64>` fields), checked against the data file in `tests/grades.rs`.
  - Update the README, `04-memory.yaml` and the tracker.

**Probe candidates** (registry basis, steel hub, full precision):

| Probe | Cell | Without | With |
|---|---|---|---|
| E22-1. Recoma 26 on both rings, NONE -> NONE.with(E22) | C104 | 64.07367112080888 | -3.423783953020314 |
| | C106 | "Above the lap-shear strength at the block ends" | "Below the lap-shear strength" |
| | C201 | 11.365659614702166 | -0.6073253229230151 |
| E22-2. Y30 on both rings | C104 | 64.07367112080888 | 11.249575845638201 |
| | C202 | "Above the fatigue endurance: qualify by thermal cycling" | "Below the fatigue endurance" |
| E23-1. Y30 on both rings, NONE -> NONE.with(E23) | C141 | 76.57010154344385 | 83.21686244344386 |
| | C143 | 255.23367181147952 | 277.3895414781462 |
| | C180 | 65.88143223919256 | 65.86604472256508 |
| E23-2. Recoma 26 inner, B842SH outer (the per-ring split) | C141 | 83.09415301144385 | 81.18448747594385 |
| | C180 | 65.86630657629901 | 65.87048331422399 |

## 5. Decisions for the user

**Approved (user, 2026-10-01): option A on all 10 decisions.**

Each decision lists the recommended option first.

1. **E21 numbers.**
   - **A.** Approve E21 as specified, with the record corrections of section 2.10 (C43 and C47 in the cells list, probe 4, the corrected wording). *Why: every demag output moves only down, and every case with an offset of 0 or more is bit-identical, default design included.*
   - B. Do nothing: N52 parts keep C12 0.17 °C, and Recoma 26 keeps a 161.9 °C magnet limit that its own model does not support.
   - C. Drop the calibration altogether: this makes the default design 9.9 °C less conservative.
2. **Cure margin C24 under E20.**
   - **A.** Land C24 = MIN(C59_inner, C59_outer) − cure as its own correction (proposed E24, gated on E20) in the same change as E21. *Why: E21 makes the shown C24 overstate the real minimum by up to 62.4 °C, the same E20 defect already reaches 79.1 °C, and the fix flips no verdict at default inputs (the lowest single-ring C24 is 12.26 °C).*
   - B. Land E21 now and the C24 fix as a later follow-up.
   - C. Do nothing.
3. **Mismatch-screen relaxation under E21** (the swing's hot end is the governing limit C12).
   - **A.** Accept, and document it in the C101 help text. *Why: the design is rated to C12, so screening the bond over the range the design allows is consistent; E21 exposes the coupling but did not create it, and with E22 on only 3 of the 9 user-basis flips remain.*
   - B. Add a correction that takes the swing's hot end from the adhesive limit C11, so the bond screen does not depend on the magnets.
4. **How the property correction is split.**
   - **A.** Two corrections: E22 (the inner ring's grade CTE in the bond plane: C104 to C106, C201, C202 and warning rule 6) and E23 (each ring's grade specific heat: C141 and the 25 thermal cells downstream). *Why: the cell sets do not overlap, so each can be probed and approved alone, and neither moves the default design, any NdFeB grade, any part or any differential case.*
   - B. One combined correction.
5. **The resolved values:** Y30 CTE ⊥ 10.0e-6 and cp 700.0; Recoma 20 CTE ⊥ 14.0e-6 and cp 370.0; Recoma 26 and Recoma 30 cp 350.0.
   - **A.** Approve. *Why: each is the primary or primary-proxy value, independently confirmed or bracketed, and Y30's 700 is also the conservative end.*
   - B. Approve, but with Eclipse's 825 for Y30 cp. This is less conservative: C141 is 3.8 % higher and the time to the limit 2.5 % longer.
6. **Recoma 26 and Recoma 30 CTE perpendicular to the magnetization.**
   - **A.** 13.0e-6 (Arnold). *Why: rule 3 (primary over secondary): it is the only grade-specific, range-stated source; and no reading, warning or verdict differs from 11.*
   - B. 11.0e-6, the independent cluster (thyssenkrupp, Eclipse, Dexter, magnet-materials). This is conservative: it gives a larger |Δα| on both hubs, and the 20-200 °C range likely makes Arnold's 13 high for the engine's 20-120 °C swing.
   - C. 12.0e-6, the midpoint.
7. **Bonded NdFeB (BCN-19) CTE.**
   - **A.** No per-grade CTE: the bonded ring keeps C95's −0.8e-6, and bonded leaves E22's move set. *Why: the three proxies are 4x apart (1-2, 4.8, about 20), none is BCN-19, and −0.8 is the most conservative value on both hubs (rule 4).*
   - B. 4.8e-6 (EAM proxy, as proposed). On an aluminium hub it clears C106 (18.11 -> 13.96 MPa), and the warning still fires.
   - C. 20e-6 (Vega). On an aluminium hub it clears C106, C202 and the warning, and it makes Δα negative on steel.
8. **Bonded NdFeB (BCN-19) specific heat.**
   - **A.** 420.0 (EAM proxy), flagged as single-source. *Why: it is the only source, and it is on the conservative side of 440 (heats faster); the effect is C141 −0.76 % and peak +0.002 °C (steel hub, both rings).*
   - B. Keep C138's 440, and bonded leaves E23's move set.
9. **Signed shear comparison (IN1).**
   - **A.** Under E22, compare |C104| with the lap shear (C106) and |C201| with the endurance limit (C202), keeping the signed values on display. *Why: E22 makes a negative mismatch reachable for the first time, and a signed test reads any negative shear as "Below"; with the proposed values no reading changes, but at the Vega alternative −23.83 MPa reads "Below" a 15 MPa strength.*
   - B. A separate correction (E25) for the absolute value.
   - C. Do nothing.
10. **Display of the inputs that E22 and E23 bypass (IN2).**
    - **A.** Keep C95 and C138 as the sintered-NdFeB inputs, with labels and ranges unchanged (python_schema). Add help text, plus Rust-only result fields for the CTE in effect (inner ring) and each ring's specific heat in effect. *Why: explicit, with no hidden input mutation, mirroring E21's C50 treatment.*
    - B. Widen the slider ranges and write the grade values into C95 and C138 when a grade is picked. The input state would then depend on the grade pick, and the workbook ranges would break.
    - C. Do nothing: the engine silently ignores C95 and C138 for non-NdFeB rings.

## Files

- This report: `docs/analyses/2026-10-01-magcoupling-addendum-a4-verification.md`.
- Data: `docs/analyses/2026-10-01-magcoupling-addendum-a4-data.json`. It holds the per-grade values with sources and every alternative the decisions offer, E21's offsets, probes and clamped configurations at full precision, the E22 and E23 probe candidates, and the decision list.
- Scratch, not in the repository (`a4-verify/` in the session scratchpad):
  - `crate-props/`: the repository plus `e21.patch` plus scratch E22 and E23, with the harness `tests/a4_props.rs`. Run it with `cargo test --test a4_props -- --nocapture --test-threads=1`.
  - `e22_e23_scratch.patch`: the scratch diff on top of E21.
  - `props_harness.out`: the raw harness output.
  - `c24_pairs.txt`: the C24 min-of-both computation over the skeptic's pair output, `crate-check/xcheck_out.json`.
