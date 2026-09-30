# Magnetic coupling calculator: Addendum A verification (A5, A6, E15–E17, defaults, A2)

**Date:** 2026-09-30. **Spec:** `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`, Addendum A (A1–A6).
**Status:** evidence complete. Nothing is applied until the user answers the decisions in section 8.
**Companion data file:** `docs/analyses/2026-09-30-magcoupling-addendum-a-data.json` has the final grade table, the part mapping and the materials table, each value with its source URL, for the Addendum A engine plan to consume. Pending items carry every option there. The plain `value` is the recommended option.

This report is self-contained. The working files it was built from were temporary, so every number a plan needs is here or in the JSON.

---

## 1. Purpose and method

### 1.1 Purpose

Before the Addendum A engine plan is written, five things have to be settled:

1. the A6 magnet grade table and the mapping of the 15 library parts to grades;
2. the A5 materials library;
3. the numbers of corrections E15–E17, which are the E9 residuals at back iron = 0 (user decision of 2026-09-30);
4. whether the defaults can stay workbook-exact, as Addendum A promises;
5. the size of the A2 equation explorer.

### 1.2 Inputs

| Deliverable | Content |
|---|---|
| A6 grade table | 17 grades × 9 fields, 15 part mappings, 7 K&J N42SH stock blocks, 10 discrepancies (X1–X10), open gaps. Built from 27 fetched manufacturer and distributor documents. No value was filled from memory, and null means no source was found. |
| A5 materials table | 14 entries (13 materials, with 17-4PH in two conditions), 95 sourced numeric values, 17 nulls with reasons, the workbook constants, and inputs for the warning rules. Built from 37 fetched documents, with source tiers 1–3. |
| E15–E17 report | Patched copies of the Python engine, rerun and diffed over the whole flattened result schema. It includes an exact 1D layered eddy-current solution, magpylib 3D fields, and a Rust cross-check against `Deviations::NONE`, `only(E9)` and `ALL`. |
| Defaults study | Every A5/A6/A1/A3 value checked against the workbook defaults, in NONE mode (Python, patched in memory) and ALL mode (a scratch copy of the crate), with cells-moved counts. |
| A2 scope count | Fields counted by running the Rust engine (`result_rows`, `input_rows`, `headline`) and cross-checked against the declaration counts in the source. |

### 1.3 Skeptic pass

An independent skeptic checked each data deliverable and the corrections. Each one re-fetched or re-derived everything on its own.

| Skeptic | Items checked | Agreed | Disagreements | How |
|---|---|---|---|---|
| Grades | 152 non-null selected cells (17 grades × 9 fields, less the null BCN-19 μrec) | 129 | 23 cells excluded, across 17 listed entries, 5 of which are note-only and do not change the count (section 2.4) | Re-fetched all 27 cited URLs: 24 with curl plus pdftotext/PyMuPDF, and the 3 SuperMagnetMan pages with WebFetch after curl got HTTP 429. Cross-checked against Eclipse, HGT and Intemag sheets. A cell counts as confirmed when its cited URL contains the number and any second source agrees within tolerance: 3 % for Br, density and (BH)max; 10 % for coercivity and coefficients. |
| Materials | 95 sourced values | 71 | 24 flagged: 8 minimum-versus-typical yields, 15 numeric or variant disagreements, 1 provenance mismatch. This report counts from the skeptic's list; the skeptic's own summary says 10 + 13 + 1. | Re-fetched all 37 cited URLs (MatWeb returned HTTP 403, as the table says). Used about 18 second-source sheets from producers and distributors. Checked every unit conversion. |
| E15–E17 | 613 items | all reproduce at report precision | 8. None is a cell-value error in the main (workbook + E9) column. One is a physics error in the standalone column, and one is a probe-basis hazard (section 5.6). | Own harness that imports the vendored engine read-only. Own derivation of the layered solution, with Poynting flux equal to ∫σ·abs(E)²/2 to 4e-9. Rebuilt the Rust dump from a separate target directory, byte-identical. |

Both data skeptics found no fabricated or misquoted number and no unit or sign error. Every value is in its cited document. The disagreements concern which source to prefer, the minimum-versus-typical basis, variants, and labelling.

### 1.4 The resolution rule

Disagreements were resolved in this order. A step settles a disagreement only when it separates the candidates.

1. **Condition match.** The entry's stated condition and temperature decide. A value measured at another condition is an alternate, not a competitor. Examples: hardened-and-tempered 4140, and PEEK specific heat at 50 °C.
2. **Vendor minimum** for guaranteed properties (yield strength; Br, Hcj, Hcb, (BH)max). The published minimum becomes the value and the typical is recorded as `typ`. A typical stays only where no minimum is published, and it is flagged.
3. **Primary over secondary.** A manufacturer or producer beats a distributor or aggregator. A polymer's resin producer beats a stock-shape converter. A grade-specific manufacturer sheet beats a family average. A part vendor's own statement governs its own parts.
4. **Otherwise open.** When the sources have equal standing, the item becomes a decision in section 8, and the recommended option is the conservative value for the check that the field feeds.

A standing user decision overrides step 3 for NdFeB. D1 (2026-09-29) set N42SH Br to 1.30 T, the K&J minimum, and the grade table extends that K&J basis to every NdFeB cell. Decision 1 asks the user to confirm the extension.

**Library value and workbook default are separate.** The library carries the sourced value as its reference. Where that value differs from today's engine number, the engine's default profile keeps the workbook number, so the NONE-mode parity and differential tests do not move (section 6). A sourced value reaches the engine only through a registered deviation.

---

## 2. A6 grade table

### 2.1 Selection rule as applied

- **NdFeB (12 grades):**
  - Br, Hcj, Hcb and (BH)max are the K&J published minimum, which is the low end of K&J's printed range. K&J does not call it a guaranteed minimum (see R17).
  - Tmax, μrec and density are K&J class values.
  - α(Br) and β(Hcj), with their measured temperature ranges, come from the Arnold per-grade sheets, because K&J publishes no coefficients.
  - K&J publishes no typical values. Arnold's min / nominal / max are in 2.3.
- **SmCo (3 grades):** the Arnold Recoma sheet minimum, with Arnold's nominal as `typ`.
- **Hard ferrite Y30:** the lower bound of the Y30 window, where Eclipse and Hyab agree. `typ` is Eclipse's C5 row. No manufacturer datasheet could be retrieved: the Arnold ceramic sheets were empty and TDK returned HTTP 403.
- **Bonded NdFeB:** the Alliance BCN-19 minimum, with Br min 650 mT for "about 0.65 T". Alliance publishes ranges only.
- **Conversions**, computed rather than typed: 1 kOe = 79.5775 kA/m, 1 kG = 0.1 T, 1 MGOe = 7.9577 kJ/m³.

### 2.2 Selected values

Legend:
- Plain number: the selected minimum.
- (typ …; max …): printed typical and maximum, where they exist.
- `c`: class-level value (one value per family, not per grade).
- `≈`: approximation, flagged in the JSON.
- `b`: temperature range borrowed from Arnold TN 0303 (Ferrite 8).
- `d`: derived (the midpoint of a printed range).
- R#: resolved in 2.4.
- D#: pending decision in section 8.

| Grade | Br (T) | Hcj (kA/m) | Hcb (kA/m) | (BH)max (kJ/m³) | α(Br) %/°C [°C] | β(Hcj) %/°C [°C] | μrec | Tmax (°C) | ρ (g/cm³) | Sources |
|---|---|---|---|---|---|---|---|---|---|---|
| N35 | 1.17 | 954.9 | 875.4 | 262.6 | −0.12 [20–80] | −0.62 [20–80] | 1.05 c | 80 | 7.50 | K&J[^kj]; α, β Arnold[^arn35] |
| N42 | 1.30 | 954.9 | 875.4 | 318.3 | −0.12 [20–80] | −0.62 [20–80] | 1.05 c | 80 | 7.50 | K&J[^kj]; α, β Arnold[^arn42] |
| N48 | 1.38 | 954.9 (R17) | 875.4 | 358.1 | −0.12 [20–80] | −0.62 [20–80] | 1.05 c | 80 | 7.50 | K&J[^kj]; α, β Arnold[^arn48] |
| N52 | 1.45 | 875.4 | **836** (R3) | 393.9 | −0.12 [20–60] | −0.62 [20–60] | 1.05 c | 80 (R4) | 7.50 | K&J[^kj]; Hcb, α, β Arnold[^arn52] |
| N42M | 1.30 | 1114.1 | 907.2 | 318.3 | −0.12 [20–100] | −0.60 [20–100] | 1.05 c | 100 | 7.50 | K&J[^kj]; α, β Arnold[^arn42m] |
| N42H | 1.30 | 1352.8 | 907.2 | 318.3 | −0.12 [20–120] | −0.57 [20–120] | 1.05 c | 120 | 7.50 | K&J[^kj]; α, β Arnold[^arn42h] |
| N42SH | 1.30 | 1592 (D17; K&J 1591.5) | 907.2 | 318.3 | −0.12 [20–150] | −0.55 [20–150] (engine default −0.50, D18) | 1.05 c | 150 | 7.50 | K&J[^kj]; Hcj, α, β Arnold[^arn42sh] |
| N38UH | 1.22 | 1989.4 | 907.2 | 286.5 | −0.12 [20–180] | −0.51 [20–180] | 1.05 c | 180 | 7.50 | K&J[^kj]; α, β Arnold[^arn38uh] |
| N35EH | 1.17 | 2387.3 | 859.4 | 262.6 | −0.12 [20–200] | −0.47 [20–200] | 1.05 c | 200 | 7.50 | K&J[^kj]; α, β Arnold[^arn35eh] |
| N33AH | 1.14 | 2705.6 | 811.7 | 246.7 (R2) | −0.12 [20–220] | −0.375 [20–220] | 1.05 c | 220 | 7.50 | K&J[^kj]; α, β Arnold[^arn33ah] |
| N50 (library only) | 1.41 | 875.4 | 875.4 | 382.0 | −0.12 [20–80] | −0.62 [20–80] | 1.05 c | 80 (R5) | 7.50 | K&J[^kj]; α, β Arnold[^arn50] |
| N50M (library only) | 1.41 | 1114.1 | 907.2 (R1) | 382.0 | −0.12 [20–100] | −0.675 [20–100] | 1.05 c | 100 (R6) | 7.50 | K&J[^kj]; α, β Arnold[^arn50m] |
| Recoma 26 (Sm2Co17 "26") | 1.00 (typ 1.04) | 1200 (typ 2000) (D4) | 680 (typ 765) | 185 (typ 205) | −0.035 [20–150] | −0.247 [20–150] | 1.05 c≈ | 350 (R11) | 8.30 | Arnold Recoma p8[^recoma]; μrec[^arn2006] |
| Recoma 30 (Sm2Co17 "30") | 1.09 (typ 1.12) | 1040 (typ 1600) (D4) | 700 (typ 820) | 215 (typ 230) | −0.035 [20–150] | −0.25 [20–150] | 1.05 c≈ | 250 | 8.30 | Arnold Recoma p12[^recoma]; μrec[^arn2006] |
| Recoma 20 (SmCo5 "20") | 0.85 (typ 0.90) | 2000 (typ 2400) | 640 (typ 700) | 140 (typ 160) | −0.045 [20–150] | −0.19 [20–150] (R8) | 1.05 c≈ | 250 | 8.40 | Arnold Recoma p4[^recoma]; μrec[^arn2006] |
| Ferrite Y30 | 0.37 (typ 0.38; max 0.40) | 180 (typ 199; max 220) | 175 (typ 191; max 210) | 26 (typ 27; max 30) | −0.20 [20–120 b] | **+0.35** (D3; the table's pick is +0.27) [20–120 b] | 1.05 c≈ | 250 | 5.00 d | Eclipse[^eclfer], Hyab[^hyab]; β option A Alliance C-5[^allc5]; range[^tn0303]; μrec[^arn2006] |
| Bonded NdFeB (Alliance BCN-19) | 0.65 (max 0.69) | 880 (max 1080) | 416 (max 496) | 72 (max 80) | −0.14 ≈ [none printed] | −0.36 ≈ [none printed] | null | 140 | 5.80 d | Alliance[^allbcn] |

Notes on the table:
- N50 and N50M are not in the spec's A6 list. They are included because parts M5044, M5026 and M5045 need them (decision 30 fixes the spec text).
- Ferrite β is positive by design: coercivity falls as the magnet gets colder.
- Bonded μrec is null. Arnold gives only a range of 1.1–1.7 for isotropic bonded Neo.
- Recoma 20 was chosen over Recoma 22 because its typical (BH)max of 20.1 MGOe is the literal "grade 20", and it is the lower-energy, conservative choice. For reference, Recoma 22[^recoma] (p5) is: Br min 0.90 T (nominal 0.94), HcB 680 / 730 kA/m, HcJ min 2000 / nominal 2400 kA/m, (BH)max 155 / 175 kJ/m³, α −0.045, β −0.25 %/°C, Tmax 250 °C, density 8.4.
- The alternates for BCN-19 in the same Alliance table are BCN-17 (Br 600–650 mT, Hci 640–800 kA/m, β −0.43 %/°C, 120 °C) and BCN-29 (Br 640–680 mT, Hci 880–1080 kA/m, 140 °C).

### 2.3 Arnold per-grade sheets (min / nominal / max), NdFeB

These are the manufacturer's guaranteed minima and nominal values. They are the basis of decision 1 option B.

| Grade | Br (T) | HcB (kA/m) | HcJ min (kA/m) | (BH)max (kJ/m³) | α, β %/°C, range | ρ | Sheet |
|---|---|---|---|---|---|---|---|
| N35 | 1.17 / 1.210 / 1.25 | 860 / 907 / 955 | 955 | 263 / 283 / 302 | −0.12, −0.62, 20–80 | 7.6 | [^arn35] |
| N42 | 1.28 / 1.315 / 1.35 | 860 / 943 / 1027 | 955 | 318 / 334 / 350 | −0.12, −0.62, 20–80 | 7.6 | [^arn42] |
| N48 | 1.37 / 1.400 / 1.43 | 836 / 963 / 1090 | 875 | 358 / 374 / 390 | −0.12, −0.62, 20–80 | 7.6 | [^arn48] |
| N52 | 1.42 / 1.450 / 1.48 | 836 / 979 / 1122 | 875 | 390 / 406 / 422 | −0.12, −0.62, 20–60 | 7.6 | [^arn52] |
| N50 | 1.39 / 1.425 / 1.46 | 836 / 975 / 1114 | 875 | 374 / 390 / 406 | −0.12, −0.62, 20–80 | 7.6 | [^arn50] |
| N42M | 1.28 / 1.315 / 1.35 | 955 / 991 / 1027 | 1114 | 318 / 338 / 358 | −0.12, −0.60, 20–100 | 7.6 | [^arn42m] |
| N50M | 1.39 / 1.415 / 1.44 | 1035 / 1066 / 1098 | 1114 | 374 / 390 / 406 | −0.12, −0.675, 20–100 | 7.5 | [^arn50m] |
| N42H | 1.28 / 1.300 / 1.32 | 955 / 979 / 1003 | 1353 | 318 / 330 / 342 | −0.12, −0.57, 20–120 | 7.6 | [^arn42h] |
| N42SH | 1.28 / 1.310 / 1.34 | 955 / 987 / 1019 | 1592 | 310 / 330 / 350 | −0.12, −0.55, 20–150 | 7.6 | [^arn42sh] |
| N38UH | 1.22 / 1.260 / 1.30 | 876 / 931 / 987 | 1990 | 287 / 307 / 326 | −0.12, −0.51, 20–180 | 7.6 | [^arn38uh] |
| N35EH | 1.17 / 1.200 / 1.23 | 836 / 887 / 939 | 2388 | 263 / 279 / 295 | −0.12, −0.47, 20–200 | 7.6 | [^arn35eh] |
| N33AH | 1.11 / 1.140 / 1.17 | 812 / 852 / 891 | 2706 | 215 / 231 / 247 | −0.12, −0.375, 20–220 | 7.5 | [^arn33ah] |

The Arnold 2021 catalog[^arncat] gives Tw max of 80 / 100 / 120 / 150 / 180 / 200 °C for the same classes, with one exception: **N52, 60 °C**. N50M and N33AH are not in the catalog.

### 2.4 Skeptic disagreements and how each was resolved

| # | Cell(s) | Table value | Other evidence | Resolution (rule in 1.4) | Status |
|---|---|---|---|---|---|
| R1 | N50M Hcb | 907.2 (K&J Hc > 11.4 kOe) | Arnold 1035 min / 1066 nominal[^arn50m]; Eclipse 1030[^eclnd]; HGT 1035[^hgt] | Keep 907.2. It is the K&J minimum on the D1 basis, and the lowest of the four sources, so it is conservative. The engine does not read Hcb. The vendor spread is recorded as alternates. | resolved |
| R2 | N33AH (BH)max | 246.7 (K&J 31 MGOe) | Arnold 215 / 231 / 247 (27 / 29 / 31 MGOe)[^arn33ah]; HGT 31–34 MGOe[^hgt] | Keep 246.7 on the D1 basis. K&J and HGT agree, and the value is consistent with K&J's own Br of 1.14 T. Rule 3 alone would prefer Arnold, the manufacturer; that is decision 1 option B. The engine does not read (BH)max. | resolved on the D1 basis (decision 1) |
| R3 | N52 Hcb against Hcj | Hcb 891.3 > Hcj 875.4 (both from K&J's table) | Arnold HcB min 836, HcJ min 875[^arn52]; Eclipse Hcb 830, Hcj 875[^eclnd] | Hcb above Hcj is physically impossible. Three sources agree on Hcj = 875, so Hcb is the defective cell. **Hcb = 836**, the Arnold (manufacturer) minimum, by rules 2 and 3. K&J's 891.3 is recorded as a table defect. The engine does not read Hcb. | resolved (value changed) |
| R4 | N52 Tmax | 80 (K&J class) | Arnold catalog Tw 60[^arncat]; Eclipse 70[^eclnd] | Keep 80. K&J sells both N52 library parts (B842-N52 and B882-N52), and a part vendor's rating governs its own parts (rule 3). Decision 2 applies the same rule to SuperMagnetMan. Arnold and Eclipse are shown as vendor spread. Decision 1 option B would take Arnold's 60. | resolved on the D1 basis (decision 1) |
| R5 | N50 Tmax | 80 | SuperMagnetMan 60 for M5044 and M5026[^smm44][^smm26]; Eclipse 70[^eclnd]; K&J and Arnold catalog 80 | The grade keeps 80 (K&J class; the Arnold catalog agrees). The part vendor's 60 °C is a question about the parts. | grade resolved; parts in decision 2 |
| R6 | N50M Tmax | 100 (K&J class NM) | Eclipse 90[^eclnd]; SuperMagnetMan 60 for M5045[^smm45]; the Arnold sheet prints none | The grade keeps 100. M5045 is a question about the part. | grade resolved; part in decision 2 |
| R7 | Y30 β(Hcj) | +0.27 | Alliance C-5 +0.35[^allc5]; Eclipse +0.27[^eclfer] and TN 0303 Ferrite 8 +0.27[^tn0303]. These two may be the same figure. | The sources have equal standing: a distributor and a manufacturer's representative, both labelling the grade C5/Y30. They disagree by 30 %, so rule 4 applies. | open, decision 3 |
| R8 | Recoma 20 β | −0.19 [20–150] | Eclipse −0.30 [20–100][^eclsmco]; TN 0303 SmCo5 −0.40 (family average over about 20–120)[^tn0303]; HGT and Intemag −0.2 to −0.3 | Keep −0.19. A grade-specific manufacturer sheet with a stated range beats family averages (rule 3). Flagged: it is 37–52 % smaller in magnitude than the family values, the non-conservative side. Hcj min 2000 kA/m keeps the stakes low. | resolved, flagged |
| R9 | Recoma 26 and 30 β | −0.247, −0.25 | Eclipse −0.20[^eclsmco]; TN 0303 −0.20; Intemag −0.17 to −0.27[^intemag] | Keep. These are grade-specific manufacturer values, and the larger magnitude is the conservative side. | resolved |
| R10 | SmCo Hcj | 1200 / 1040 / 2000 (Arnold minima) | Eclipse SmCo26 > 19 kOe = 1512 (its own kA/m column prints > 1433); SmCo30 > 21 kOe = 1671; SmCo20 min 1194[^eclsmco]; Arnold Recoma 26HE 1500 | Every cell matches its Recoma page. The spread is a sub-grade naming question: Recoma 26 and 30 are the low-Hcj variants. | open, decision 4 |
| R11 | Recoma 26 Tmax | 350 | Arnold prints 350, with footnote 3 "may be considerably lower at low load line"[^recoma]. Eclipse reserves 350 °C for H variants (L 250, standard 300)[^eclsmco]. | Keep 350, the manufacturer's grade sheet (rule 3). The footnote is carried as a flag. The value follows decision 4. | resolved, flagged |
| R12 | Y30 and BCN-19 density | 5.0, 5.8 | Eclipse 4.9–5.1; Alliance C-5 4.8; Alliance BCN 5.6–6.0 | Keep both values and relabel them as "derived (range midpoint)". Density is not a guaranteed property. | resolved (label) |
| R13 | BCN-19 α, β | −0.14, −0.36 | Alliance's column headings are "Bd" and "Hd" (operating-point coefficients). Arnold TN 0303 bonded MQP-C gives −0.07 / −0.40, but for a different, cobalt-bearing powder. | Keep, flagged as an approximation: operating-point coefficients stand in for α(Br) and β(Hcj), and no temperature range is printed. | resolved (flag) |
| R14 | μrec, SmCo ×3 and Y30 | 1.05 | One source: Arnold 2006, slide 23, "about 1.05"[^arn2006] | Keep, flagged as class-level, approximate and single-source. The engine does not read μrec. The null for BCN-19 stays. | resolved (flag) |
| R15 | Y30 coefficient range | [20, 120] | TN 0303 says this of its own table, whose ferrite row is Ferrite 8, not Y30 | Keep it as a borrowed range, marked in the field (`temp_range_basis`). | resolved (flag) |
| R16 | NdFeB density basis text | 7.5, "Arnold's newer sheets (7.5)" | Arnold prints 7.6 on 10 of its 12 sheets (7.5 only on N50M and N33AH); K&J 7.4–7.5; Eclipse 7.5; HGT 7.4–7.7 | Keep 7.5. It is within 1.3 % of every source and bit-equal to the engine's 0.0075 g/mm³. The basis text is corrected. | resolved (text) |
| R17 | NdFeB "minimum" meaning | the low end of K&J's range | K&J prints ranges and never calls the low end guaranteed. Arnold's guaranteed Br minimum is 0.7–2.7 % lower (N42 1.28, N52 1.42, N50 1.39, N42SH 1.28, N33AH 1.11). N48 Hcj is 954.9 at K&J and 875 at Arnold (+8.4 %; Eclipse's 955 agrees with K&J). | Everything is inside tolerance. This is the D1 basis itself. | decision 1 |

The grade skeptic also confirmed these points:
- K&J's table is transcribed correctly. Every Br, Hc, Hci and (BH)max range converts correctly at 79.5775 kA/m per kOe and 7.9577 kJ/m³ per MGOe.
- The Arnold values match all 12 per-grade sheets, and the Recoma values match pages 2, 4, 8 and 12.
- Y30 matches Eclipse, Hyab and Courage Magnet, and BCN-19 matches Alliance.
- Signs are correct: ferrite β is positive and every other β is negative.
- No Wikipedia or remembered values were used.

### 2.5 What the engine does with a grade

- **Fields the engine reads.** Today it reads only `br_T` and `tmax_C` per part, plus the grade text for the E3 gate (`library.rs:55-61`). After the A6 fix (decision 19) it also reads Hcj and β. α(Br) = −0.12 %/°C and density 7.5 g/cm³ are bit-equal to the engine constants −0.0012 /°C and 0.0075 g/mm³, as long as they are written as literals. Hcb, (BH)max and μrec are for display only.
- **Conversion hazard.** −0.55/100 evaluates to −0.0055000000000000005, not −0.0055. The JSON therefore stores engine-unit literals (`beta_hcj_per_C: -0.0055`), and nothing should be converted at load time.
- **Positive ferrite β.** `demag_onset_C` takes `beta.abs()` (`temperature.rs:609`, and the same in `temperature.py`), so a positive ferrite β is silently negated today. Whichever option decision 3 takes, the Y30 cold case needs a formula branch in which the knee falls as temperature falls, not only new grade data.
- **Coefficient ranges are short**, from N52's 20–60 °C to N33AH's 20–220 °C. Beyond them a linear coefficient is extrapolation, and TN 0303 says the change is non-linear. No source here covers below 20 °C, which is the side the ferrite cold case needs (TN 0303 Fig. 2 shows the knee forming at cold for Ceramic 5).

---

## 3. Part-to-grade mapping

What the engine does today:
- `library.py` / `library.rs` have **no Hcj column**. `MagnetSpec` holds Br and tmax.
- The demag check uses one input for every part: `hcj20_kA_m` = 1592 kA/m, which is the N42SH minimum, with `beta_hcj_per_C` = −0.005 /°C (`temperature.py:56-57`, `temperature.rs:68-71`).
- The demag block reads only the **inner** ring (`api.rs:217-219`).

In the table below, "Br engine" is the corrected engine value under E3, and "Hcj ratio" is today's 1592 divided by the grade's Hcj.

| Part | Vendor | Shape | Dimensions (mm) | Grade | Br stored → engine (E3) | Grade Br min (Arnold nominal) | Hcj ratio | Tmax stored / grade | Grade β %/°C | Flags |
|---|---|---|---|---|---|---|---|---|---|---|
| B842SH (default, both rings) | K&J | block | 12.7 × 6.35 × 3.17 | N42SH | 1.29 → 1.30 | 1.30 (1.31) | 1.00 | 150 / 150 | −0.55 | E3 (D1) |
| B842 | K&J | block | 12.7 × 6.35 × 3.17 | N42 | 1.30 | 1.30 (1.315) | 1.667 | 80 / 80 | −0.62 | |
| B842-N52 | K&J | block | 12.7 × 6.35 × 3.17 | N52 | 1.45 | 1.45 (1.45) | 1.819 | 80 / 80 | −0.62 | R4 |
| B822 | K&J | block | 12.7 × 3.17 × 3.17 | N42 | 1.30 | 1.30 (1.315) | 1.667 | 80 / 80 | −0.62 | |
| B862 | K&J | block | 12.7 × 9.5 × 3.17 | N42 | 1.30 | 1.30 (1.315) | 1.667 | 80 / 80 | −0.62 | |
| B882 | K&J | block | 12.7 × 12.7 × 3.17 | N42 | 1.30 | 1.30 (1.315) | 1.667 | 80 / 80 | −0.62 | |
| B882-N52 | K&J | block | 12.7 × 12.7 × 3.17 | N52 | 1.45 | 1.45 (1.45) | 1.819 | 80 / 80 | −0.62 | R4 |
| B861 | K&J | block | 12.7 × 9.5 × 1.59 | N42 | 1.30 | 1.30 (1.315) | 1.667 | 80 / 80 | −0.62 | |
| B881 | K&J | block | 12.7 × 12.7 × 1.59 | N42 | 1.30 | 1.30 (1.315) | 1.667 | 80 / 80 | −0.62 | |
| B442 | K&J | block | 6.35 × 6.35 × 3.17 | N42 | 1.30 | 1.30 (1.315) | 1.667 | 80 / 80 | −0.62 | |
| BX042SH | K&J | block | 25.4 × 6.35 × 3.17 | N42SH | 1.29 → 1.30 | 1.30 (1.31) | 1.00 | 150 / 150 | −0.55 | E3 |
| BX082SH | K&J | block | 25.4 × 12.7 × 3.17 | N42SH | 1.29 → 1.30 | 1.30 (1.31) | 1.00 | 150 / 150 | −0.55 | E3 |
| M5044 | SuperMagnetMan | arc | 13 × 5.5 × 1.31 | N50 | 1.42 | 1.41 (1.425) | 1.819 | 80 / 80 | −0.62 | decision 2 |
| M5045 | SuperMagnetMan | arc | 6.56 × 5.6 × 1.12 | N50M (grid says N50) | 1.42 | 1.41 (1.415) | 1.429 | 100 / 100 | −0.675 | decision 2 |
| M5026 | SuperMagnetMan | arc | 15 × 6.4 × 1.67 | N50 | 1.42 | 1.41 (1.425) | 1.819 | 80 / 80 | −0.62 | decision 2 |

Sources: Br min, Hcj min and Tmax come from K&J[^kj]. Arnold nominal Br and β come from the mapped grade's Arnold sheet (2.3). Stored values are from `library.py`, `library.rs` and `temperature.py` in this repository.

**What the mapping shows:**
- **Tmax, α and density** reproduce every part exactly.
- **Br** differs only for:
  - the three N42SH parts, which are already E3;
  - the three SuperMagnetMan arcs, which store 1.42 T against the K&J N50 and N50M minimum of 1.41 T (X4, defaults mismatch 4, decision 2).
- **Hcj and β** differ for every part except the three N42SH parts. For the other 12 this difference *is* the A6 correctness fix. For example, B842 (N42) sees 1.667 times its grade's Hcj today (decision 19).
- **SuperMagnetMan pages.** The vendor grid lists "Max Working Temp 60 °C" on all three arc pages[^smm44][^smm45][^smm26], where the library stores 80 / 100 / 80. M5045's grid also contradicts its own title: the grid says "Neodymium 50" and 6.50 mm, the title says N50M and 6.56 mm. SuperMagnetMan publishes no Br or Hcj, so the grade values for these parts are K&J and Arnold proxies (decision 2).
- **Coverage.** 12 of the 17 grades have no library part (13 if decision 2 option A maps M5045 to N50): N35, N48, N42M, N42H, N38UH, N35EH, N33AH, the three SmCo grades, Y30 and BCN-19. They are reachable only through the new "any grade with manual dimensions" mode. So decisions 3 and 4, and every value in those grades, move nothing at the defaults or in the 3,391 differential cases. The ferrite cold-case test must run through the custom-dimension mode.
- **Missing part fields.** A6's part table also lists coating and magnetization direction. Neither was collected in this verification. The only one recorded is that B842SH is magnetized through its 3.17 mm thickness. The engine plan must fill both from the vendor pages.

**K&J stock blocks in the added grades.** Source: the K&J High Temperature listing, 37 results over 2 pages, read on 2026-09-30[^kjht1][^kjht2]. The only grades in stock are N42SH, N52SH and N35AH, and blocks exist **only in N42SH**. There are no K&J stock blocks in N42M, N42H, N38UH, N35EH or N33AH; N52SH and N35AH come only as discs and cylinders. Read literally, the spec's "stock block sizes in the added grades where they exist" adds nothing. The table lists four N42SH blocks that are not yet in the library (decision 1, option C):

| Part | Grade | Size (in) | mm | In library |
|---|---|---|---|---|
| B421SH | N42SH | 1/4 × 1/8 × 1/16 | 6.35 × 3.17 × 1.59 | no |
| B842SH | N42SH | 1/2 × 1/4 × 1/8 | 12.7 × 6.35 × 3.17 | yes |
| BX042SH | N42SH | 1 × 1/4 × 1/8 | 25.4 × 6.35 × 3.17 | yes |
| BX082SH | N42SH | 1 × 1/2 × 1/8 | 25.4 × 12.7 × 3.17 | yes |
| BX088SH | N42SH | 1 × 1/2 × 1/2 | 25.4 × 12.7 × 12.7 | no |
| BY042SH | N42SH | 2 × 1/4 × 1/8 | 50.8 × 6.35 × 3.17 | no |
| BY0X02SH | N42SH | 2 × 1 × 1/8 | 50.8 × 25.4 × 3.17 | no |

---

## 4. A5 materials table

Legend:
- `†`: model approximation. Every μr and Bsat carries it.
- `min`: published guaranteed minimum.
- `lo`: lower bound of a printed range.
- `max`: printed upper limit.
- `typ*`: typical, where no minimum was found.
- `c`: read from a chart.
- `d`: derived by interpolation.
- `p`: proxy.
- `≤`: upper bound.
- `n/a`: not applicable (non-ferromagnetic).
- `null`: no source; the reason is in 4.4.
- M#: resolved in 4.2.
- D#: pending decision.

Conditions: 4140 annealed; 1018 hot rolled (resistivity, CTE and cp are for annealed material, as the sources state); 12L14 cold drawn; 416 annealed; 17-4PH H1150 (default) and H900; 304 and 316L annealed; 6061 and 7075 in T6/T651; Ti-6Al-4V annealed; IN625 annealed; PEEK unfilled; acetal POM homopolymer (Delrin 150).

| Material | Role | Ferro | μr | Bsat (T) | σ (S/m) | ρ (g/cm³) | CTE (1e-6/K) | E (GPa) | Yield (MPa) | cp (J/(kg·K)) | Sources |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `4140_annealed` | back iron (default) | yes | 363† (secant at 1.5 T) | null | 4.33e6 (M13) | 7.85 | 12.2 | 205 | 415 | 461 | [^smag][^lucefin][^azom4140][^otai4140] |
| `1018_hot_rolled` | back iron | yes | null | null | 5.85e6 d (D7) | 7.87 | 12.0 | 205 | 220 | 486 | [^azom1018][^twm1018] |
| `12L14_cold_drawn` | back iron | yes (from composition) | null | null | 5.75e6 (D7) | 7.87 | 11.5 | 200 | 415 | 472 | [^twm12l14][^azom12l14] |
| `416_annealed` | back iron | yes | 110† | 1.60† p (D6) | 1.75e6 | 7.64 | 10.5 (D6; to be re-sourced) | 200 (D6) | 276 typ* (D6) | 460.5 | [^carp416][^smith416][^smag][^zapp416] |
| `17-4PH_H1150` | back iron (17-4PH default) | yes | 76† c | null | 1.25e6 p (M24) | 7.82 | 11.9 | 199.9 | **725 min** (M1) | 460 p | [^armco174][^rolled174] |
| `17-4PH_H900` | back iron | yes | 96† c | null | 1.25e6 | 7.80 | 10.8 | 199.9 | **1170 min** (M2) | 460 | [^armco174][^rolled174] |
| `304_annealed` | back iron (demo) | no | 1.02† max | n/a | 1.39e6 | 8.03 | 16.9 | 193 (M20) | **205 min** (M3) | 500 | [^ak304][^sandm304] |
| `6061_T6` | cap/housing (default), back iron (demo) | no | 1† p | n/a | 2.49e7 | 2.70 | 23.6 | 68.3 | **241 min** (M5) | 896 | [^kaiser6061][^qq250-11][^wikiperm] |
| `7075_T6` | cap/housing | no | 1† p | n/a | 1.91e7 | 2.80 | 23.4 | 71.0 | **462 min** (M6) | 960 | [^kaiser7075][^qq250-12][^wikiperm] |
| `316L_annealed` | sleeve/liner (default) | no | 1.02† max | n/a | 1.35e6 | 7.99 | 16.0 | 193 (M21) | **172 min** (M4) | 500 | [^ak316][^sandm316] |
| `Ti6Al4V_annealed` | sleeve/liner | no | 1† | n/a | 5.95e5 | 4.42 | 9.0 | 113.8 (M22) | **828 min** (M7) | 580 (M23) | [^timet][^asmti64] |
| `IN625_annealed` | sleeve/liner | no | 1.001† | n/a | 7.75e5 | 8.44 | 12.8 | 207.5 | 414 lo | 410 | [^sm625] |
| `PEEK_unfilled` | sleeve/liner | no | null | n/a | ≤ 1e-13 | 1.31 | 50 (M17) | **4.0** (M16) | **98** (M15) | 1100 (M14) | [^tecapeek][^victrex] |
| `POM_H_acetal` | cap/housing | no | null | n/a | ≤ 1e-13 | 1.41 | 122.4 (M10) | **3.1** (M9) | 75.84 | 1465 | [^ensdelrin][^dupont150][^delringuide] |

Values in **bold** changed as a result of the skeptic pass. PEEK and acetal conductivity is an upper bound taken from the printed volume resistivity; for slip-loss purposes it is zero.

**Design flux density** is a separate field. It is the number the wall-thickness check actually uses (`model.rs:644`, `model.py:280`). Only two workbook values exist: 4140 = 1.5 T (Materials!C13), and 1018 ≈ 1.7 T, which appears only in a workbook comment. "Use about 1.4 T for pre-hardened stock" is also a comment. See decision 20.

**CTE mismatch with the magnets** is the material's CTE minus the workbook's NdFeB bond-plane CTE of −0.8e-6 /K (`temperature.rs:184`), with the same sign convention as `d_alpha`. Values in 1e-6/K: 4140 13.0; 1018 12.8; 12L14 12.3; 416 11.3; 17-4 H1150 12.7; H900 11.6; 304 17.7; 6061 24.4; 7075 24.2; 316L 16.8; Ti-6Al-4V 9.8; IN625 13.6; PEEK 50.8; POM 123.2.

### 4.1 Per-value sources (the cells that are not obvious from the source column)

- **4140:**
  - σ: 0.231 Ω·mm²/m at 20 °C (Lucefin, DIN SEW 310)[^lucefin].
  - ρ 7.85 and CTE 12.2 over 0–100 °C: AZoM[^azom4140].
  - E 205 and yield 415 annealed at about 197 HB: Otai (tier 3)[^otai4140]. AZoM gives the same 415. The skeptic confirms 415 is a real minimum.
  - cp 461 at 20 °C (479 at 100 °C): Lucefin.
  - μr 363 is SMAG p.45[^smag], the secant value at 1.5 T. It is not the incremental μr the engine uses (workbook 200, decision 22).
  - Bsat is null. K&J's generic 2.2 T[^kjshield] exceeds SMAG's pure-iron limit of 2.158 T, so it is not used.
- **1018:**
  - σ is derived by linear interpolation to 20 °C from AZoM's annealed values: 15.9 µΩ·cm at 0 °C and 21.9 at 100 °C[^azom1018].
  - ρ 7.87 from AZoM. CTE 12.0 (0–100 °C, annealed), yield 220 (hot-rolled bar, 20–32 mm) and cp 486 (50–100 °C) from The World Material[^twm1018].
  - E 205 is AZoM's "typical for steel". Alternates are 186 (TWM) and 190 (MakeItFrom[^mif1018]).
  - No producer sheet was reachable.
- **12L14:** σ 0.174 µΩ·m, temperature not stated (TWM[^twm12l14]). ρ, CTE 11.5 ("typical steel") and yield 415 cold drawn from AZoM[^azom12l14]. E 200 and cp 472 from TWM. Physical data here come from aggregators only. The alloy contains lead, which is a RoHS/REACH concern.
- **416:**
  - From Carpenter[^carp416]: σ 343.0 Ω·cmil/ft = 57.02 µΩ·cm at 70 °F; ρ 7.64; yield 276 as the annealed typical at 70 °F; cp 0.1100 Btu/lb/°F (0–100 °C).
  - From Zapp 1.4005 IA, the magnetic-quality variant with C ≤ 0.02 %[^zapp416]: CTE 10.5 over 20–100 °C, E 215, and Bsat, from Js ≥ 1.70 T.
  - μr 110 is the secant value at 1.5 T from SMAG Table 4[^smag].
  - Carpenter's own blog[^carpblog] gives the maximum μr: 750 annealed and 95 hardened.
  - The entry therefore mixes two variants, which is decision 6.
  - The table above shows option A of decision 6 (standard martensitic 416): E 200 GPa (Smiths[^smith416]) and Bsat 1.60 T, a proxy from SMAG Table 4 (410 at 0.15 % C; the 416 row is blank). The Zapp CTE of 10.5 remains as a placeholder until a standard-416 range is confirmed.
- **17-4PH:**
  - ARMCO[^armco174] Table 17 gives ρ, CTE over 21–93 °C (H1150 11.9, H900 10.8) and cp 460 (Condition A and H900; H1150's column is "–").
  - μr 76 and 96 are read from ARMCO Fig. 3 at 140 Oe. B is about 1.07 T for H1150 and 1.35 T for H900, ±0.05 T. These are lower bounds on Bsat, not Bsat.
  - E 199.9 is the dynamic modulus from Rolled Alloys[^rolled174] (29.0 Msi). ARMCO Table 7 gives 201 for H1150 and 200 for H900, which agrees.
  - Yield minima are from ARMCO Table 3. AMS 5643[^ams5643] gives 724 and 1172.
- **304 and 316L:**
  - AK Steel[^ak304][^ak316] gives σ (72 and 74 µΩ·cm at 20 °C), ρ, CTE over 0–100 °C, E 193 in tension, cp 500, and μr "1.02 max" at 200 Oe annealed. Carpenter reports 1.003–1.005 for well-annealed material.
  - The yield minima are ASTM A240 (30 and 25 ksi), via Sandmeyer[^sandm304][^sandm316].
  - The typicals of 290 (AK) are kept as `typ`.
- **6061 and 7075:**
  - Kaiser[^kaiser6061][^kaiser7075] gives 43 % and 33 % IACS × 58e6, ρ, CTE over 20–100 °C, E, and cp at 100 °C.
  - The yield minima are QQ-A-250/11 (6061-T6, 35 ksi)[^qq250-11] and QQ-A-250/12 (7075-T6, 462–476 MPa by thickness; the lowest band is taken)[^qq250-12].
  - The typicals of 276 and 503 (Kaiser, equal to the workbook) are kept as `typ`.
  - μr uses the pure-aluminium Wikipedia row[^wikiperm] as a flagged tier-3 proxy.
- **Ti-6Al-4V:**
  - From TIMET[^timet]: μr 1.00005 at 20 Oe; σ 1.68 µΩ·m at 0 °C (the 20 °C value is not tabulated); ρ 4.42; CTE over 0–100 °C; cp 0.580 J/(g·K); yield minimum 120 ksi from Table 3.
  - E 113.8 is ASM's value[^asmti64], inside TIMET's 107–122 range.
- **IN625:** everything from Special Metals Tables 2–5[^sm625]. Yield is the lower bound of the composite range 414–655, which the datasheet says is not for specification.
- **PEEK:** Ensinger TECAPEEK stock shape[^tecapeek] for σ (10^15 Ω·cm), ρ, CTE over 23–100 °C and cp at 23 °C. Victrex 450G[^victrex] for E (4000 MPa, ISO 527-1) and yield (98.0 MPa, ISO 527-2).
- **Acetal:**
  - Ensinger TECAFORM AD / Delrin 150[^ensdelrin] gives σ (> 10^15 Ω·cm), ρ, CTE 6.8e-5 /°F and yield 11,000 psi.
  - E is 3100 MPa from the DuPont Delrin 150 NC010 datasheet (ISO 527)[^dupont150].
  - cp is 0.35 Btu/lb/°F. The DuPont design guide[^delringuide] confirms it as the average over −18 to 100 °C.

### 4.2 Skeptic disagreements and how each was resolved

| # | Cell | Table value | Other evidence | Resolution (rule in 1.4) | Status |
|---|---|---|---|---|---|
| M1 | 17-4 H1150 yield | 862 (Rolled Alloys, "representative") | ARMCO Table 3 minimum 725 (Table 2 typical 862)[^armco174]; AMS 5643 724[^ams5643] | Rule 2: **725 min** (ARMCO, the producer, by rule 3); 862 kept as `typ`. | resolved |
| M2 | 17-4 H900 yield | 1276 | ARMCO minimum 1170; AMS 1172 | **1170 min**; 1276 as `typ`. | resolved |
| M3 | 304 yield | 290 (AK typical) | ASTM A240 minimum 205 (30 ksi)[^sandm304] | **205 min**; 290 as `typ`. | resolved |
| M4 | 316L yield | 290 (AK typical) | ASTM A240 minimum 172 (25 ksi)[^sandm316] | **172 min**; 290 as `typ`. | resolved |
| M5 | 6061-T6 yield | 276 (Kaiser typical) | QQ-A-250/11 minimum 241 (35 ksi)[^qq250-11] | **241 min** as the library value. The workbook's 276 (Materials!C39) stays the engine default. Neither engine reads aluminium `yield_MPa`: grepping `magcoupling-rs/src` and the Python package finds only the definitions (`materials.rs:67,77,87`), and the alloy rows are constants, not inputs. So zero cells move. | resolved |
| M6 | 7075-T6 yield | 503 (Kaiser typical) | QQ-A-250/12 minimum 462–476 by thickness[^qq250-12]; SE Metal ≥ 430[^semetal7075] | **462 min**, the lowest band on the cited sheet. Confirm the product form when the part's stock is known. SE Metal's 430 (secondary, form not stated) is recorded. The workbook's 503 (Materials!C34) stays the default. It is not read, so zero cells move. | resolved |
| M7 | Ti-6Al-4V yield | 880 (ASM typical) | TIMET Table 3 minimum 828 (120 ksi)[^timet] | **828 min**; 880 as `typ`. | resolved |
| M8 | 416 yield | 276 (Carpenter typical) | Zapp ≥ 230, but for the ferritic 1.4005 IA variant[^zapp416]; Rolled Alloys typical 40–50 ksi[^rolled416] | No minimum was found for standard martensitic 416 annealed bar. The value depends on which 416 the entry represents. | open, decision 6 |
| M9 | Acetal E | 2.413 (Ensinger stock shape, 350 kpsi) | DuPont Delrin 150 3.1 GPa (ISO 527)[^dupont150]; Alro 450 kpsi = 3.10[^alro]; Laminated Plastics 3.10[^lamdelrin] | Rule 3: the resin producer is primary, and two distributors agree with it. **3.1 GPa.** 2.413 is recorded. Acetal E feeds nothing in the engine. | resolved |
| M10 | Acetal CTE | 122.4 (Ensinger, range not stated) | Alro and Laminated Plastics 84.6 (4.7e-5 /°F); DuPont guide 104–135 over −40 to 94 °C[^delringuide] | Rule 3: DuPont is primary. Ensinger's 122.4 lies inside DuPont's range, and the distributors' 84.6 lies outside it. Keep 122.4. The higher value is also the conservative one for the mismatch warning. | resolved |
| M11 | 12L14 σ | 5.747e6 (TWM, 0.174 µΩ·m) | 4.12e6 (7.1 % IACS: MakeItFrom[^mif12l14], Titanium Industries[^ti12l14]) | Every source is an aggregator or distributor, and the spread is 40 % (rule 4). | open, decision 7 |
| M12 | 1018 σ | 5.848e6 (interpolated from AZoM, annealed) | 4.06e6 (7.0 % IACS, hot rolled, MakeItFrom[^mif1018]). AZoM, TWM and Elgin agree with the table value. | Secondary sources only; the conditions are mixed (hot-rolled yield, annealed σ); spread 44 % (rule 4). | open, decision 7 |
| M13 | 4140 σ | 4.33e6 (Lucefin, 20 °C) | 5.26e6 (Böhler, 0.19 Ω·mm²/m)[^bohler4140], probably a hardened-and-tempered 4140-class grade | Rule 1: keep Lucefin, whose data card is not heat-treatment-specific and fits the annealed entry. Böhler is recorded at its own condition. The engine default stays at the workbook's 4.5e6, which lies between them (decision 21). | resolved |
| M14 | PEEK cp | 1100 (Ensinger, 23 °C) | 1560 at 50 °C and 2150 at 200 °C (Syensqo KetaSpire KT-820, DSC)[^kt820] | Rule 1: 1100 is the only value at 23 °C. Keep it; the lower cp is also conservative for heating (a faster rise). KT-820 is recorded at its temperatures. | resolved, flagged |
| M15 | PEEK yield | 116 (Ensinger stock shape) | Victrex 450G 98 (ISO 527-2)[^victrex]; KT-820 96 ISO / 95 ASTM | Rule 3: the two resin producers agree. **98 MPa** (Victrex 450G, the table's cross-check grade). 116 is recorded. | resolved |
| M16 | PEEK E | 4.2 (Ensinger) | Victrex 4.0; KT-820 3.83 ISO / 3.5 ASTM | Rule 3: **4.0 GPa** (Victrex 450G). 4.2 is recorded. | resolved |
| M17 | PEEK CTE | 50 (Ensinger, 23–100 °C) | Victrex 55 (average below Tg, 143 °C); KT-820 43 (flow direction, −50 to 50 °C) | Rule 1: 23–100 °C is the range used for every other material, so keep 50. The others are recorded at their ranges. Low severity. | resolved |
| M18 | 416 E | 215 (Zapp 1.4005 IA) | 200 (Smiths, standard 416)[^smith416] | The variant question. | open, decision 6 |
| M19 | 416 Bsat | 1.7 (Zapp, Js ≥ 1.70 T, C ≤ 0.02 %) | SMAG Table 4: 1.60 T for 410 at 0.15 % C, 1.70 T only for 13-FM at 0.03 % C; 416 is left blank[^smag] | The variant question. Standard 416 (C up to 0.15 %) is nearer 1.6 T. | open, decision 6 |
| M20 | 304 E | 193 (AK) | 200 (Outokumpu Core[^okcore]; Sandmeyer) | Two producers' typical values, 3.5 % apart. Keep AK, the producer for the rest of the 304 record; 200 is recorded. It matters only through the bond-stress screen, when 304 is picked as a demonstration back iron. | resolved |
| M21 | 316L E | 193 (AK) | 200 (Outokumpu Supra[^oksupra]; Sandmeyer) | As M20. 316L E is not read by the engine. | resolved |
| M22 | Ti-6Al-4V E | 113.8 (ASM) | Carpenter 104.8[^carpti64]; TIMET 107–122 | Rule 3: the primary producer's range (TIMET) brackets 113.8. Keep it. The spread comes from texture. | resolved |
| M23 | Ti-6Al-4V cp | 580 (TIMET, 20 °C) | ASM 526.3 | Rule 3: keep the primary producer, TIMET. | resolved |
| M24 | 17-4 H1150 σ | 1.299e6 (Rolled Alloys, 463 Ω·cmil/ft, "condition not stated") | Carpenter Custom 630 prints exactly 463 Ω·cmil/ft under Condition H900[^carp630]. ARMCO leaves H1150 blank. | Provenance: this is an H900 number. Rule 3: use ARMCO's H900 figure, 80 µΩ·cm = **1.25e6**, flagged as a proxy because no source gives an H1150 resistivity. The value changes by 4 %. | resolved (value and label) |

**Watch list**. These items are below the skeptic's thresholds and are not counted as disagreements; they are recorded in the JSON:
- 4140 CTE: Industeel and Böhler give 11.1 over 20–100 °C for hardened-and-tempered plate, against Lucefin's 12.1 and the table's 12.2.
- 17-4 H900 cp: 460 (ARMCO) against Carpenter's 419.
- 416 CTE: 10.5 against Smiths' 9.9 and Rolled Alloys' 10.1.
- Ti σ: taken at 0 °C, which is 4–6 % high against room-temperature sources.
- 1018 and 12L14: aggregator data only.
- Single-source magnetics: 4140 and 416 μr (SMAG), the 17-4 chart reads, the "1.02 max" for 304 and 316L, and the aluminium Wikipedia proxies.
- The Aero Metals Alliance densities (2.63 and 2.71) look like typos and were ignored.

### 4.3 What the engine reads

- **Steel μr** (`materials.steel.mu_r_incremental`, Materials!C15 = 200) enters only the skin depth (`temperature.rs:790`). The torque circuit treats iron as infinitely permeable: it uses the sinh factor for steel or the exponential factor for free space, chosen by `backiron`. So a ferromagnetic yes/no selects the circuit, and μr cannot change torque.
- **Steel density lives in two places.** Materials!C19 is not consumed anywhere. Metal design!C132 (0.00785 g/mm³) is the one the mass model reads. A back-iron picker must drive a single field.
- **The sleeve density** (Metal design!C44) also sets the endplate mass, and A5 names no endplate material.
- **The cap conductivity is hard-wired** to 6061-T6 (`api.rs:261`, `api.py:85`). A 7075 or acetal cap must change that.
- **The clamp alloy table** (`clamps.alloy`, 7075 by default) is separate. Its allowables (shear, head pressure, key bearing) are not A5 fields, so the cap/housing picker must not drive it.
- **Yield** is not read anywhere in the engine today.

### 4.4 Nulls, and why

| Item | Why |
|---|---|
| Bsat for 4140, 1018, 12L14, 17-4PH | No manufacturer or handbook value was found. K&J's generic 2.2 T is above the pure-iron Js of 2.158 T (SMAG), so it is not used. For 17-4 there is only B at 140 Oe, a lower bound. |
| μr for 1018 and 12L14 | Only neighbouring grades were found (1010: 585–1260; 1020: about 500 at 1.5 T). 12L14 is treated as 1018-class. |
| Incremental μr at the magnet bias, for every ferromagnetic entry | No source gives it. The values present are secant, maximum or chart-read. The engine keeps the workbook's 200. |
| μr for PEEK and POM | No datasheet states one. They are non-magnetic, so the engine should key on ferromagnetic = false. |
| Bsat for the eight non-ferromagnetic entries | Not applicable. |
| Copolymer acetal | Not researched; the homopolymer was chosen. |

MatWeb (HTTP 403 on every URL), Metal Suppliers Online (503), the Curbell TECAPEEK sheet (bot check) and the ATI 17-4 host could not be read, so none of them is cited.

---

## 5. E15–E17 audit rows

### 5.1 Basis

- **Engine:** magcoupling 1.0.0, the workbook port. Read-only from `reference/magcoupling-py/magcoupling`.
- **Rust cross-check:** `magcoupling-rs` was used read-only from a scratch Cargo project (feature `workbook-parity`). It dumped every celled value for `Deviations::NONE`, `only(E9)` and `ALL`, at back iron 0 and 1.
- **Method:** the M1 audit's method. Each correction is written as a patched copy of the engine function and rerun. Changed and unchanged cells come from diffing the whole flattened result schema. Values are given to 4 significant figures.

**Back iron defaults to 1** (`model.py:52`, `model.rs:82-84`), so **nothing changes at the defaults.** At back iron = 1, 70 runs were made: every subset of {E15, E16, E17}, times the five E17 variants, times E9 on or off. Every result path was bit-identical to the workbook engine. So was a non-default steel design (12 poles, 3000 rpm, 4 mm web) with all three corrections on. The skeptic added 192 switch combinations at back iron = 1, all bit-identical. That confirms the stated gates, but it does not test an implementation: the gates make the result identical by construction.

**Three baselines at back iron = 0:**
1. **Workbook + E9**, the main column. E9 is approved, and E15–E17 are its residuals.
2. **M2**, the state that ships, with E1–E14 on. The Python emulation of E1, E3, E5, E9, E11 and E12 matches the Rust `ALL` dump on every Temperature design cell to 1e-12. Only Calculator!C63 (E6) and Shaft clamps!C48 (E2) differ, as expected.
3. **Standalone**, `only(Ek)` with E9 off. The cup and boss are then still steel, so only the hub, which the workbook already makes aluminium, can move.

**Parity checks:**
- With no switch set, the patched copies reproduce the vendored engine bit for bit at both back-iron settings.
- Python matches Rust `NONE` and `only(E9)` to 1e-12 on all 330 celled results at both settings.
- The E9 registry probe literals are reproduced exactly: C111 20.896171609459955, C113 10.585910605536167, C114 96.81172961300027, C141 45.78201892981087, Materials!C22 "No back iron".

### 5.2 Headline effect at back iron = 0

| Headline cell | Workbook + E9 | + E15 | + E16 | + E17 | + E15 + E16 + E17 | M2 | M2 + E15 + E16 + E17 |
|---|---|---|---|---|---|---|---|
| Pull-out C93 (N·m) | 1.697 | 1.697 | 1.697 | 1.697 | 1.697 | 1.697 | 1.697 |
| Hot minimum Metal design!C11 | "Below hot minimum" | same | same | same | same | same | same |
| Governing limit C12 (°C) | 92.55 | 92.55 | 92.55 | 92.55 | 92.55 | 93.06 | 93.06 |
| Steady high-case temperature C18 (°C) | 89.77 | 89.77 | 89.77 | **94.18** | **94.18** | 92.51 | **94.18** |
| Time to limit, high case C19 = C150 (s) | "never" | "never" | "never" | **440.3** | **606.1** | "never" | **684.1** |
| Rotations to limit C151 (rev) | "never" | "never" | "never" | 1.468e4 | 2.020e4 | "never" | 2.280e4 |
| Time to limit, estimate C152 | "never" | "never" | "never" | "never" | "never" | "never" | "never" |
| Peak with fault C20 (°C) | 66.01 | 65.92 | 66.01 | 66.19 | 66.09 | 66.12 | 66.09 |
| Critical drag C23 (N·m) | 0.03946 | same | same | same | same | 0.04019 | 0.04019 |
| Verdict C25 | "OK on temperature…" | same | same | same | same | same | same |

Headline effects:
- **E17 changes a headline number. The change sits inside the model's uncertainty** (5.5.5).
- **E15** lengthens the time scales: heat capacity rises by 37.6 %.
- **E16** changes only the masses of the adapter variant.
- **No correction** changes the torque, the governing limit, any mass cell or the verdict C25. C25 never reads C19.

### 5.3 Audit rows

The columns follow the M1 audit, with a skeptic column added.

| # | Group | Cells | Formula (workbook) | Issue and evidence | Effect (C6 = 0) | Proposed correction | Skeptic verdict | Decision |
|---|---|---|---|---|---|---|---|---|
| E15 | Heat capacity prices the aluminium cup, boss and hub at steel specific heat (E9 residual 1) | Temperature design!C141 → C143, C145, C154–C161, C171, C172, C180–C182, C186, C189, C190, C192, C193, C196, C20 | C141 = [m_mag·c_NdFeB + (C111 + C112 + C113 + hardware)·Materials!C16 + (retainers + endplates)·c_316 + cap·C140] / 1000 | With C6 = 0, E9 prices the cup and boss at aluminium density, and the workbook already prices the hub that way (C112). But C141 still multiplies all three by 4140's specific heat, 473 J/(kg·K), where aluminium has 900 (C140, already an input; 6061-T6 sheet 896). Breakdown with E9, steel c → aluminium c: cup 20.90 g, 9.884 → 18.81 J/K; hub 8.877 g, 4.199 → 7.989 J/K; boss 10.59 g, 5.007 → 9.527 J/K. The 6 g of steel keys and screws stays at 2.838 J/K. The error understates C by 27 % and so overstates the rise per event, which is the conservative direction. | **C141 45.78 → 63.02 J/K**; τ C143 152.6 → 210.1 s; C145 0.005411 → 0.003931 °C; C20 66.01 → 65.92 °C; 19 more cells (5.4). No verdict changes. M2: the same C141, and C20 66.12 → 66.02. Standalone: hub term only, C141 74.19 → 77.98. | C141 = [m_mag·c_NdFeB + (Calculator!C111 + Calculator!C113)·c_cup + Calculator!C112·c_hub + Metal design!C128·Materials!C16 + (retainers + endplates)·Temperature design!C139 + cap·Temperature design!C140] / 1000. c_cup = C140 when the cup is aluminium (E9's gate: C6 ≠ 1 and E9 on), otherwise Materials!C16. c_hub = C140 when C6 ≠ 1, otherwise Materials!C16. The gate must read the same flag the density reads, not a second test on C6. | Agreed. All 23 cells reproduce in all three columns. | pending (decision 8) |
| E16 | The disc removed from the web in the aluminium-adapter variant is priced at steel density (E9 residual 2) | Metal design!C189 → C191, C148, C149 (C147 label) | C189 = π/4·(C185² − Calculator!C39²)·C125·C132 (steel density). C191 = C47 − Calculator!C113 − C189 + C188 + C190 | With C6 = 0 the web is part of the E9 aluminium cup, but the disc bored out for the adapter pilot is still priced at 7.85 g/cm³, so the hybrid mass comes out 2.265 g too low. The labels C147 "Selected one-piece steel-cup mass" and C189 "Steel removed…" say "steel" at C6 = 0. The values of C147 and C188 are right. | **C189 3.453 → 1.188 g; C191 = C148 101.5 → 103.8 g; C149 −4.689 → −6.954 g.** The negative "mass saved" is real: at C6 = 0 the aluminium adapter replaces a boss that is already aluminium, so the variant adds 6.954 g. M2: identical. Standalone: no change. | C189 = π/4·(C185² − Calculator!C39²)·C125·ρ_cup, where ρ_cup is the density Calculator!C111 uses for the web (C42 when aluminium under E9, otherwise C132), passed from the mass model as one source of truth. Keep the labels (schema parity) and reword the help (decision 14). | Agreed. The disc lies in the E9 aluminium web, so it is priced at 2.7 g/cm³ (C189 3.453 → 1.188 g; C191 101.5 → 103.8 g). | pending (decision 9) |
| E17 | The cup, web and hub eddy losses use the steel skin-depth model, steel constants and steel-circuit fields for aluminium parts (E9 residual 3; hub included, decision 13) | Temperature design!C123, C124, C125 → C130–C134, C145–C157, C161, C169–C177, C180–C182, C186–C196, C17–C22 (41 cells); 3 new inputs | C123/C124 = σ_st·ω_e²·B²·δ_st/(4k²)·2πrL; C125 = σ_st·ω_e²·δ_st/4·(r_mid/p)²·C121; δ_st = √(2/(ω_e μ0 μr σ_st)), σ_st = Materials!C14 = 4.5e6 S/m, μr = Materials!C15 = 200, B = C116 / C117 / C121 (3D, doubled at a steel surface, including the opposite member's steel image) | **This is not a constant swap.** For 6061-T6, σ = 2.5e7 S/m (Materials!C43, confirmed by two datasheets[^all6061][^ndtal]) and μr = 1. At the 166.7 Hz slip frequency δ_Al = 7.797 mm, thicker than every aluminium part (d/δ = 0.23–0.65). The magnetic Reynolds number is 0.13–0.42 and kd = 0.50–2.5. These parts are resistance-limited and the field decays across the wall, which is not the skin-limited half-space the formula assumes. Swapping only σ and μr gives a cup loss of 45.10 W instead of 1.543 W. The stored fields are also wrong for aluminium, because they include the image doubling and the opposite member's steel image. | **Hub C123 0.2259 → 0.1942 W; cup C124 1.353 → 1.543 W; web C125 0.0914 → 0.3737 W; total C130 2.477 → 2.918 W; C18 89.77 → 94.18 °C, above the 92.55 °C limit, so C19/C150 "never" → 440.3 s** and C151 → 1.468e4 rev. C152 stays "never", and no text verdict changes. M2: C125 0.3656 → 0.3737, total 2.751 → 2.918, C19 → 497.0 s (684.1 s with E15). **Standalone (E9 off), amended:** C123 0.2259 → 0.3392 W, C130 2.477 → 2.590 W, C18 89.77 → 90.90 °C. | At C6 = 0, price each aluminium part with the low-Reynolds closed form T1: P = f_end·σ_Al·ω_e²·B_free²/(2k²)·d_eff·A, with d_eff = (1 − e^(−2kd))/(2k), k = p/r, σ_Al = Materials!C43, f_end = C114 and A = 2πrL (L = Calculator!C33). Hub: r = Calculator!C8 − Metal design!C120, d = Calculator!C38. Cup: r = Calculator!C60 + Metal design!C121, d = Metal design!C122, B_free = 0.08764 T. Web: (r_mid/p)² replaces 1/k², with r_mid = Calculator!C8 + Calculator!C20/2 and d = Metal design!C125, and ∫B_free² dA = 6.837e-6 T²·m² replaces A·B². **Hub field, amended:** B_free = 0.07832 T when the cup is aluminium (E9 on). It is C116/2 = 0.1035 T when the cup is steel (E9 off), because the outer ring then has its first-order image in the steel cup, not doubled. The material gate is the same as E15's. | Agreed on every main-column and M2 cell. **Disagreed on the standalone column**, which used the free-space hub field where the cup is steel. The arithmetic was right and the field basis was wrong, so the proposal is amended. Other items: a probe-basis hazard and five text corrections (5.6). | pending (decisions 10–15) |
| E18 (candidate) | The adhesive thermal-mismatch screen uses 4140's CTE and modulus for a hub that the mass model makes aluminium | C94, C98 → C104, C105, C201; verdict text C106, C202 | Uses Materials!C17 (12.3e-6 /°C) and C18 (205 GPa) whatever the hub's material is | Found during verification; outside the approved scope. At C6 = 0 the hub is aluminium: 23.6e-6 /°C and 68.9 GPa (Alliance 6061-T6[^all6061]). | M2 basis (E1 on): C104 11.68 → 20.76 MPa; C105 6.071 → 11.03 MPa; C201 2.563 → 4.655 MPa. **Verdict text changes:** C106 "Below the lap-shear strength" → "Above the lap-shear strength at the block ends", and C202 "Below the fatigue endurance" → "Above the fatigue endurance: qualify by thermal cycling". On the workbook basis (C96 = 0.55) both already read "Above", and C104 goes 46.13 → 73.87 MPa. Nothing changes at the defaults. | Register it with E15's gate: the hub material follows the hub density rule. | Reproduced, including both verdict flips. | pending (decision 16) |

### 5.4 Changed cells at back iron = 0

Some cells change only below 4 significant figures (for example C186 and C192), so they look unchanged at report precision.

**E15**

| Cell | Label | Workbook + E9 → corrected | M2 → + E15 | Standalone (E9 off) |
|---|---|---|---|---|
| Temperature design!C20 | Peak magnet and bond temperature with the slip fault | 66.01 → 65.92 | 66.12 → 66.02 | 65.89 → 65.88 |
| Temperature design!C141 | Heat capacity of the rotating coupling | 45.78 → 63.02 | 45.78 → 63.02 | 74.19 → 77.98 |
| Temperature design!C143 | Thermal time constant | 152.6 → 210.1 | 152.6 → 210.1 | 247.3 → 259.9 |
| Temperature design!C145 | Temperature rise per slip event | 0.005411 → 0.003931 | 0.00601 → 0.004366 | 0.003339 → 0.003177 |
| Temperature design!C154 | Initial heating rate, estimate | 0.05411 → 0.03931 | 0.0601 → 0.04366 | 0.03339 → 0.03177 |
| Temperature design!C155 | Initial heating rate, high case | 0.1623 → 0.1179 | 0.1803 → 0.131 | 0.1002 → 0.0953 |
| Temperature design!C156 | Relative rotations per °C at the start, estimate | 616.1 → 848 | 554.7 → 763.5 | 998.3 → 1049 |
| Temperature design!C157 | Relative rotations per °C at the start, high case | 205.4 → 282.7 | 184.9 → 254.5 | 332.8 → 349.8 |
| Temperature design!C158 | Relative rotations per thermal time constant | 5087 → 7002 | 5087 → 7002 | 8243 → 8664 |
| Temperature design!C159 | Time to 95 % of the steady rise | 457.8 → 630.2 | 457.8 → 630.2 | 741.9 → 779.8 |
| Temperature design!C160 | Relative rotations to 95 % of the steady rise | 1.526e4 → 2.101e4 | 1.526e4 → 2.101e4 | 2.473e4 → 2.599e4 |
| Temperature design!C161 | Magnet temperature at the fault trip time, high case | 65.32 → 65.23 | 65.36 → 65.26 | 65.2 → 65.19 |
| Temperature design!C171 | Temperature rise per event, estimate | 0.005411 → 0.003931 | 0.00601 → 0.004366 | 0.003339 → 0.003177 |
| Temperature design!C172 | Temperature rise per event, high case | 0.01623 → 0.01179 | 0.01803 → 0.0131 | 0.01002 → 0.00953 |
| Temperature design!C180 | Peak magnet temperature: hot day plus fault-limited slip | 66.01 → 65.92 | 66.12 → 66.02 | 65.89 → 65.88 |
| Temperature design!C181 | Margin to the skipping onset | 36.54 → 36.63 | 36.93 → 37.03 | 36.66 → 36.67 |
| Temperature design!C182 | Margin to the magnet design limit | 26.54 → 26.63 | 26.93 → 27.03 | 26.66 → 26.67 |
| Temperature design!C186 | Pull-out torque at the peak temperature (reversible) | 1.63 → 1.631 | 1.63 → 1.63 | 1.631 → 1.631 |
| Temperature design!C189 | Peak bond temperature | 66.01 → 65.92 | 66.12 → 66.02 | 65.89 → 65.88 |
| Temperature design!C190 | Margin to the adhesive design limit | 53.99 → 54.08 | 53.88 → 53.98 | 54.11 → 54.12 |
| Temperature design!C192 | Pull-out torque at the peak temperature, with +variation | 1.875 → 1.875 | 1.874 → 1.875 | 1.876 → 1.876 |
| Temperature design!C193 | Shear stress amplitude per reversal, hot | 0.1981 → 0.1982 | 0.1981 → 0.1981 | 0.1982 → 0.1982 |
| Temperature design!C196 | Hot fatigue margin | 7.571 → 7.569 | 7.573 → 7.571 | 7.569 → 7.569 |

C19 and C150 stay "never" in every column. C137, the display of the steel specific heat, does not change.

**E16**

| Cell | Label | Workbook + E9 → corrected | M2 → + E16 | Standalone |
|---|---|---|---|---|
| Metal design!C148 | Optional aluminium-adapter variant mass | 101.5 → 103.8 | 101.5 → 103.8 | unchanged |
| Metal design!C149 | Mass saved by optional aluminium adapter | −4.689 → −6.954 | −4.689 → −6.954 | unchanged |
| Metal design!C189 | Steel removed for optional larger pilot bore | 3.453 → 1.188 | 3.453 → 1.188 | unchanged |
| Metal design!C191 | Optional hybrid gross mass | 101.5 → 103.8 | 101.5 → 103.8 | unchanged |

C147 (96.81 g), C188 (16.73 g), C47, C114 and C192 do not change. Full precision: C189 3.4526103262951824 → 1.1875220230569419; C191 101.5003046059424 → 103.76539290918065; C149 −4.6885749929421365 → −6.95366329618038.

**E17** (recommended form: T1, end factor C114, free-space fields)

The standalone column of the original report used the wrong hub field and is not reproduced here. With the amended hub field (C116/2 = 0.1035 T while the cup is steel), the skeptic computed C123 0.2259 → 0.3392 W, C130 2.477 → 2.590 W and C18 89.77 → 90.90 °C. The other 33 standalone cells must be regenerated with the amended field before any standalone probe is written.

| Cell | Label | Workbook + E9 → corrected | M2 → + E17 |
|---|---|---|---|
| Temperature design!C17 | Continuous slip, steady magnet temp, estimate | 73.26 → 74.73 | 74.17 → 74.73 |
| Temperature design!C18 | Continuous slip, steady magnet temp, high case | 89.77 → 94.18 | 92.51 → 94.18 |
| Temperature design!C19 | Unbroken slip time to the limit, high case | "never" → 440.3 | "never" → 497 |
| Temperature design!C20 | Peak magnet and bond temperature with the slip fault | 66.01 → 66.19 | 66.12 → 66.19 |
| Temperature design!C22 | Average slip heating over life, high case | 0.6881 → 0.8105 | 0.7643 → 0.8105 |
| Temperature design!C123 | Hub surface (solid steel) | 0.2259 → 0.1942 | 0.2259 → 0.1942 |
| Temperature design!C124 | Cup surface (solid steel) | 1.353 → 1.543 | 1.353 → 1.543 |
| Temperature design!C125 | Rear web (solid steel) | 0.0914 → 0.3737 | 0.3656 → 0.3737 |
| Temperature design!C130 | Total estimated slip loss | 2.477 → 2.918 | 2.751 → 2.918 |
| Temperature design!C131 | Equivalent mean drag torque | 0.01183 → 0.01393 | 0.01314 → 0.01393 |
| Temperature design!C132 | Loss used below | 2.477 → 2.918 | 2.751 → 2.918 |
| Temperature design!C134 | Loss, high case | 7.431 → 8.754 | 8.254 → 8.754 |
| Temperature design!C145 | Temperature rise per slip event | 0.005411 → 0.006374 | 0.00601 → 0.006374 |
| Temperature design!C146 | Continuous slip: steady rise, estimate | 8.257 → 9.726 | 9.171 → 9.726 |
| Temperature design!C147 | Continuous slip: steady rise, high case | 24.77 → 29.18 | 27.51 → 29.18 |
| Temperature design!C148 | Continuous slip, steady magnet temp, estimate | 73.26 → 74.73 | 74.17 → 74.73 |
| Temperature design!C149 | Continuous slip, steady magnet temp, high case | 89.77 → 94.18 | 92.51 → 94.18 |
| Temperature design!C150 | Continuous slip, time to the limit (high case) | "never" → 440.3 | "never" → 497 |
| Temperature design!C151 | Continuous slip, rotations to the limit (high case) | "never" → 1.468e4 | "never" → 1.657e4 |
| Temperature design!C154 | Initial heating rate, estimate | 0.05411 → 0.06374 | 0.0601 → 0.06374 |
| Temperature design!C155 | Initial heating rate, high case | 0.1623 → 0.1912 | 0.1803 → 0.1912 |
| Temperature design!C156 | Relative rotations per °C at the start, estimate | 616.1 → 523 | 554.7 → 523 |
| Temperature design!C157 | Relative rotations per °C at the start, high case | 205.4 → 174.3 | 184.9 → 174.3 |
| Temperature design!C161 | Magnet temperature at the fault trip time, high case | 65.32 → 65.38 | 65.36 → 65.38 |
| Temperature design!C169 | Heat per event, estimate | 0.2477 → 0.2918 | 0.2751 → 0.2918 |
| Temperature design!C170 | Heat per event, high case | 0.7431 → 0.8754 | 0.8254 → 0.8754 |
| Temperature design!C171 | Temperature rise per event, estimate | 0.005411 → 0.006374 | 0.00601 → 0.006374 |
| Temperature design!C172 | Temperature rise per event, high case | 0.01623 → 0.01912 | 0.01803 → 0.01912 |
| Temperature design!C173 | Total slip heat over life, high case | 14.86 → 17.51 | 16.51 → 17.51 |
| Temperature design!C175 | Average temperature rise from slip, estimate | 0.2294 → 0.2702 | 0.2548 → 0.2702 |
| Temperature design!C176 | Average temperature rise from slip, high case | 0.6881 → 0.8105 | 0.7643 → 0.8105 |
| Temperature design!C177 | Average rise per 1 % of time slipping, high case | 0.2477 → 0.2918 | 0.2751 → 0.2918 |
| Temperature design!C180 | Peak magnet temperature: hot day plus fault-limited slip | 66.01 → 66.19 | 66.12 → 66.19 |
| Temperature design!C181 | Margin to the skipping onset | 36.54 → 36.36 | 36.93 → 36.87 |
| Temperature design!C182 | Margin to the magnet design limit | 26.54 → 26.36 | 26.93 → 26.87 |
| Temperature design!C186 | Pull-out torque at the peak temperature (reversible) | 1.63 → 1.63 | 1.63 → 1.63 |
| Temperature design!C189 | Peak bond temperature | 66.01 → 66.19 | 66.12 → 66.19 |
| Temperature design!C190 | Margin to the adhesive design limit | 53.99 → 53.81 | 53.88 → 53.81 |
| Temperature design!C192 | Pull-out torque at the peak temperature, with +variation | 1.875 → 1.874 | 1.874 → 1.874 |
| Temperature design!C193 | Shear stress amplitude per reversal, hot | 0.1981 → 0.198 | 0.1981 → 0.198 |
| Temperature design!C196 | Hot fatigue margin | 7.571 → 7.575 | 7.573 → 7.575 |

Full precision on the workbook + E9 basis, with the 4-s.f. field literals pinned (decision 12):
- C123 0.22590854790913922 → 0.1942432063711941
- C124 1.3530776316850626 → 1.5432858065214725
- C125 0.09140096902640166 → 0.3736960055840505
- C130 2.4771082543421903 → 2.917946124198304
- C18 89.7710825434219 → 94.17946124198303
- C19 "never…" → 440.3079945062734

**Cells verified unchanged at back iron = 0**, for each correction:
- torque and geometry: Calculator!C62, C93, C94; Metal design!C9, C11;
- every mass: Calculator!C110–C114; Metal design!C47, C147, C188;
- the temperature limits and demagnetization onsets: Temperature design!C10–C12, C14, C56–C59;
- the steel display cells and the other loss terms: C109, C110, C112, C115, C126, C127, C128, C129;
- the adhesive and mismatch screens: C86, C91, C104, C106;
- Materials!C20 and C22.

### 5.5 Validity reasoning for E17

#### 5.5.1 Exact 1D layered solution (the reference)

Take x as the slip direction, y as the normal into the part, and the axial direction as infinite. The opposite ring's fundamental is a travelling wave e^{j(ω_e t − kx)}, with k = p/r and ω_e = p·Ω_slip. Its free-space normal field at the part's surface is B_inc. The part is a layer 0 ≤ y ≤ d with conductivity σ and permeability μr, and free space behind it.

- In free space A″ = k²A; in the conductor A″ = γ²A, with γ² = k² + jω_e μ0 μr σ.
- A and (1/μ)∂A/∂y are continuous at y = 0 and y = d. With β = γ/μr and T = tanh(γd), the layer seen from y = 0 is Λ = −β(k + βT)/(β + kT).
- A(0) = 2k·a/(k − Λ), with a = B_inc/k. The loss per area is the Poynting flux, P/A = −(ω_e·abs(A(0))²/(2μ0))·Im Λ.

Checks:
- The Poynting flux equals ∫σ·abs(E)²/2 dy (to 1e-7 in the report; 4e-9 in the skeptic's independent derivation).
- As μr → ∞ with kδ → 0 the solution reproduces the workbook's steel formula (0.9994–0.9996), with a surface field magnitude of B_inc × 2, which is the image doubling.
- As σ → 0 the surface field equals B_inc.

#### 5.5.2 Regime of each aluminium part

The operating point is 10 poles and 2000 rpm, so f = 166.7 Hz, δ_Al = 7.797 mm and δ_steel = 1.299 mm.

| Part | d (mm) | k (1/m) | d/δ_Al | k·d | Rm = ω_e μ0 σ/k² | B_surface/B_inc (magnitude) | exact / T1 | exact / thin sheet T0 | steel formula (B = 2B_inc) / exact |
|---|---|---|---|---|---|---|---|---|---|
| Cup, corner wall C122 | 1.800 | 278.7 | 0.2309 | 0.5017 | 0.4235 | 0.9963 | 0.9920 | 0.6262 | 0.4150 |
| Cup, flat wall C63 (E6) | 2.723 | 278.7 | 0.3493 | 0.7590 | 0.4235 | 0.9935 | 0.9849 | 0.5066 | 0.3391 |
| Rear web C125 (k at r_mid) | 2.500 | 426.1 | 0.3206 | 1.065 | 0.1812 | 0.9982 | 0.9956 | 0.4118 | 0.4544 |
| Hub, C38 (flat to bore) | 5.100 | 495.0 | 0.6541 | 2.525 | 0.1342 | 0.9984 | 0.9950 | 0.1958 | 0.4685 |

**The steel formula's regime assumption does not hold for aluminium.** The steel formula assumes a skin-limited half-space: the eddy currents live in a skin much thinner than the part, and a high-μ surface doubles B. Aluminium differs on all three counts. Its skin depth (7.8 mm) exceeds every part, it has no magnetic image, and Rm < 1, so the eddy currents barely disturb the incident field, which penetrates with its free-space decay e^{−ky}. Two simpler forms also fail:
- The thin flat-sheet form that the workbook uses for the 316L shells (T0 = σ d ω_e² B²/(2k²)) overstates the loss 1.6–5.1 times, because kd is not small.
- The constant swap gives cup 45.10 W, hub 7.530 W and web 3.047 W. The cup figure is **29.2 times** the recommended 1.543 W: 5.96 from the doubled steel-circuit field, times 3.43 from δ/(2d_eff), divided by the 0.7 end factor. The original report said 20.5, which is the ratio against T1 without the end factor (2.205 W).

#### 5.5.3 T1: the closed form proposed

In the unscreened (low-Rm) limit the conductor sees the incident field a·e^{−ky}. So

  P/A = σ ω_e² B_inc² / (2k²) · d_eff,  with d_eff = (1 − e^{−2kd}) / (2k).

This is the exact solution's first term in Rm. It reduces to the thin-shell form as kd → 0, saturates at 1/(2k) for thick parts, and is linear in σ.

| Check | Result |
|---|---|
| At the operating point | exact/T1 = 0.985–0.996 for all four parts |
| Across the input box at default radii (poles 4–40, slip 100–6000 rpm) | within 1.4 % at ≤ 2000 rpm for every pole count; 0.887–0.976 at 6000 rpm (worst: hub, 4 poles) |
| Error crossings at 10 poles, by bisection | 5 %: 3,708 rpm (flat wall) to 6,889 rpm (web). 10 %: 5,396–10,039 rpm. The original report gave grid points 0.35–1.5 % past these. |
| Far corner of the slider box, inner apothem 30 mm (r_hub 29.95 mm, solid hub wall 24.95 mm, r_cup 37.79 mm) | 2000 rpm, 10 poles: cup 0.959, web 0.951, **hub 0.738**. 6000 rpm, 4–10 poles: cup 0.703–0.723, web 0.647–0.684, **hub 0.130–0.338**, so T1 overstates the hub loss by up to **7.7 times**. The original report quoted only the cup ("30 % screened"). Screening reaches 87 % for the hub. |
| Never non-conservative | Over kd = 0.01–10 and Rm = 1e-4–1e3, the largest exact/T1 is 1 − 3e-13. T1 can only overstate the loss. |
| Against the real cross-sections (magpylib cylindrical decay) | Cup: section/T1 = 1.012 with d = C122, 0.821 with d = C63, 0.834 with the area-mean wall, so C122 is the right thickness for the closed form. Hub: 1.008–1.009 with d = C38. |
| Rear web (corrected by the skeptic) | Converged over 51 z-points, the planar d_eff overstates the depth by **28 %** (numeric/closed 0.7832; the original 0.8075 came from a 6-point trapezoid). Weighting and depth interact, so the true section/closed ratio is **1.141**. The closed form is therefore **12.4 % low on the web: +0.053 W, about 1.8 % of the total**, on the non-conservative side. The original said 8.5 % and 1.2 %. |
| Conductivity temperature | σ is kept at its 20 °C value, as for every other conductor in the workbook. 6061 loses roughly 15–20 % of its σ by 70–90 °C, and loss is proportional to σ in this regime, so the 20 °C value is conservative. |

#### 5.5.4 Fields

`fields3d.run` doubles both b_hub and b_cup (the `2 *` factor; C117's "3D." label does not say so). At back iron = 1 it also adds the opposite member's first-order image ring, and E5's ×4 on the web is the same image effect. None of that applies with μr = 1. The helpers reproduce the stored values first (0.2069 → C116 0.207; 0.2138 → C117 0.214; 1.0346e-5 → C121 1.035e-5). Then, at back iron = 0 with no images and no doubling:

| New input | Value (4 s.f., pinned) | Stored steel-circuit value it replaces at C6 = 0 | Ratio |
|---|---|---|---|
| b_hub_free_T (cup aluminium) | 0.07832 T | C116 0.207 T | 0.3783 |
| b_cup_free_T | 0.08764 T | C117 0.214 T | 0.4095 |
| web_integral_free_T2m2 | 6.837e-6 T²·m² | C121 1.035e-5 (workbook), 4.14e-5 (E5) | 0.6606 / 0.1652 |
| hub field, cup steel (E9 off), amended | C116/2 = 0.1035 T (no new input) | C116 0.207 T | 0.5 |

Running T1 with the stored steel-circuit fields gives a total of 11.93 W and C19 = 40.07 s, 4.1 times the correct loss. Unrounded, the magpylib fields are 0.0783182, 0.0876388 and 6.83722e-6. The consequence for the probes is in 5.6.

#### 5.5.5 End factor, and how firm the headline is

- **The workbook's convention.** It applies C114 = 0.7 to every thin conductor (the 316L sleeve and liner, and the aluminium cap), and none to the skin-limited steel. The aluminium parts at back iron = 0 are thin conductors in the same regime.
- **The physical bracket.** The Russell–Norsworthy end-effect factor for the cup wall (k = p/r, magnet length 12.7 mm) is 0.467 with no overhang, 0.605 with 1.4 mm each side (the actual 15.5 mm cup with the magnets centred), 0.709 with 5 mm, and 0.726 with infinite overhang. C114 = 0.7 lies inside that bracket, not at its top as the original report said. The rear web closes the cup like an end ring and could push the true factor above the bracket.
- **For the hub** (ka ≈ 3.1) the bracket is about 0.68–0.84. The hub term is only 0.19 W.

**The headline flip is inside the model's uncertainty.** C19 is finite only when C130 exceeds (governing limit − T0)·G/3 = 2.755 W on the workbook + E9 basis (2.806 W on M2). The recommended total of 2.918 W is only 6 % above that threshold. Aluminium-part loss is proportional to f_end, so C19 is finite only for **f_end ≥ 0.646** (M2: 0.663). Plausible inputs land on both sides:

| Input choice | Total C130 | C19 |
|---|---|---|
| T1, f_end = C114 = 0.7 (recommended) | 2.918 W | 440.3 s |
| Exact layered, f_end = 0.7 | 2.903 W | 454.2 s |
| T1, f_end = 0.7, plus the converged web section correction (+0.053 W) | 2.971 W | finite, shorter |
| T1, Russell–Norsworthy for the actual cup (0.605) | 2.631 W | "never" |
| T1, 6061 σ at 70–90 °C (×0.80–0.85) | 2.50–2.60 W | "never" |
| T1, no end factor (f = 1) | 3.823 W | 194.6 s (M2: 202.0 s) |
| Exact layered, no end factor | 3.801 W | 196.9 s |
| T1, f = 0.7, stored steel-circuit fields (wrong) | 11.93 W | 40.07 s |

C19 is the high case: C134 = 3 × C132 already carries the 3× loss multiplier. The verdict C25 does not read C19.

### 5.6 Skeptic verdict on E15–E17

- **Reproduction:**
  - Every E15, E16 and E17 cell in the main and M2 columns reproduces at report precision (613 items).
  - With no switches, the skeptic's harness is bit-identical to `compute_all` on all 995 result paths.
  - The skeptic's M2 emulation differs from Rust `ALL` only at C63 (E6) and C48 (E2).
  - The diff of the full schema gives exactly the report's changed-cell lists: E15 23 cells, E16 4, E17 41.
- **E15:** the formula was re-derived. C141 is the only reader of the specific heat, and the new value is 45.78 → 63.02 J/K.
- **E16:** the disc lies in the E9 aluminium web, so it is priced at 2.7 g/cm³.
- **Candidate E18** reproduces in full, including both verdict flips. Decision 13 option B's total (2.950 W) and decision 11 option B's factor (1.31 times) also check.
- **Corrections carried into this report:**
  1. **Standalone hub field** (physics). With E9 off the cup is steel, so the aluminium hub sees 0.1035 T (C116/2), not 0.07832 T. The hub loss is 1.745 times higher, and the standalone total rises (2.477 → 2.590 W) instead of falling. **E17 is amended** (5.3, decision 12). Under decision 15 option A, `only(E17)` is never probed, but the engine must still be right whenever E9 is off.
  2. **Probe basis.** Every full-precision E17 value reproduces only with the 4-s.f. literals 0.07832, 0.08764 and 6.837e-6. With the unrounded fields, C124 becomes 1.5432443 and C19 440.3417 (M2: 497.1), and C145, C154 and C171 move in the fourth significant figure: probes drift by 1e-5 to 8e-5 relative. **The registry pins the 4-s.f. literals**, the same precision as the other stored 3D inputs, and M3 re-blesses the probes when it computes the fields live (decision 12).
  3. Constant-swap factor: 29.2 times, not 20.5 (5.5.2).
  4. Far corner: T1 overstates the hub loss by up to 7.7 times, not the 30 % seen on the cup (5.5.3).
  5. 0.959 at the 30 mm apothem is for the cup only; the hub is 0.738 (5.5.3).
  6. Rear web: 28 % depth overstatement and 12.4 % below the section integral, not 24 % and 8.5 % (5.5.3).
  7. The 5 % and 10 % crossing speeds come from bisection (5.5.3).
  - Also: the headline flip hangs on the end-factor choice, and 0.7 is not the "top" of the bracket (5.5.5).

### 5.7 Other observations (no change proposed)

- The three new free-space inputs, like every stored 3D value, were computed at the workbook's Br = 1.29 T. M3 must apply E3 (1.30 T) to them, as it must for the steel-circuit values.
- The match for d = C122 (section/T1 = 1.012) was validated at the default geometry only.
- The new inputs are Rust-only. `tests/python_schema.rs` must tolerate them, as it tolerates `clamps.length_note` (decision D6).
- The other 3D inputs are also steel-circuit values and are still used at C6 = 0. These are the demagnetization fields C52–C55, which set C12, and the sleeve, liner, cap and magnet fields C118–C120 and C122. Recomputing them for back iron = 0 is M3's live 3D work.
- The E5 help text in Rust calls 1.035e-5 "the free-space field". It is actually the field including the steel hub's first-order image; the inner ring alone gives 6.837e-6. E5's ×4 remains right for the steel web.
- At back iron = 1 the workbook's semi-infinite steel formula is 5–24 % conservative against an exact slab: exact/W = 0.941 for a half-space, 0.948 at the 2.72 mm flat wall and 0.808 at the 1.8 mm corner wall. This is outside the present scope and is noted for the engine plan.
- **E9 probe interaction.** In ALL mode, E15 moves C141, which is a registered E9 cell (45.78 J/K). Under decision 15 option A, E9's own probe (NONE against `only(E9)`) keeps 45.78. The ALL-mode registry test still passes because C141 is registered under both E9 and E15.

---

## 6. Defaults exactness

### 6.1 What "workbook-exact" is enforced by

Addendum A promises that "the M2 parity and differential tests are unchanged". Those tests are:
1. **parity:** every celled value at the defaults, in `Deviations::NONE`;
2. **differential:** 3,391 seeded input sets in `tests/data/differential/*.json`, in NONE mode. They use every library part (plus `''` and the manual `b842sh`) and vary every input independently, including all 17 housing inputs, `hcj20_kA_m`, `beta_hcj_per_C`, `bsat_T` and `conductivity_S_m` (101 distinct values each in `full.json`);
3. **the registry tests:** `each_deviation_alone_changes_exactly_its_registered_cells`, `all_deviations_together_change_only_registered_cells`, `each_probe_shows_its_correction` and `e7_leaves_every_default_cell_bit_for_bit`.

**Classification rule:** a change can ship only as a registered deviation, with probes, if it moves any value in NONE mode (at the defaults or in any differential case) or any unregistered cell in ALL mode. If the library number must equal today's number, the library entry is marked "workbook".

**Short answer:** as specified, the defaults cannot stay workbook-exact. Thirteen mismatches need a decision (6.6). Three structural findings apply to every item:

1. **Preset, not replacement.** If the engine read grade, material or autofit values *instead of* the existing inputs, the differential tests would break even where every default is equal. The pickers should be GUI presets that write the existing inputs, with the engine left as ported. Alternatively, each input stays as an override that wins when set.
2. **New inputs need a Rust-only rule.** A harmonic-set, material or grade input changes `input_schema.json`, and the differential generator (`tools/gen_differential.py`) would pass that path to Python.
3. **Library values are stored in engine units as literals** (section 2.5).

### 6.2 The default part, field by field (B842SH on both rings, grade N42SH)

| Field | Engine today | Grade N42SH (library) | NONE: cells moved | ALL: cells moved | Effect |
|---|---|---|---|---|---|
| Br | 1.29 T (workbook row); 1.30 T with E3 | 1.30 T | 233 (this is E3, already registered) | 0 | none in ALL |
| Tmax | 150 °C | 150 °C | 0 | 0 | none |
| Hcj | 1592 kA/m (input C44) | 1591.5 (K&J "> 20 kOe"; exactly 20 kOe is 1591.55) or 1592 (Arnold sheet, in kA/m) | 23 for 1591.5; 0 for 1592 | 23; 0 | C12 92.550 → 92.531 °C (NONE); 93.056 → 93.037 °C (ALL) |
| β(Hcj) | −0.50 %/°C (input C45, "effective 20–150 °C") | −0.55 %/°C (Arnold, 20–150 °C) | 23 (+1 bit-only) | 23 | C12 92.55 → 96.67 °C (NONE), 93.06 → 97.13 °C (ALL); C50 offset 10.43 → −3.42 °C; C104 46.13 → 48.82 MPa (NONE) |
| Hcj + β together | | 1591.5, −0.55 | 23 | 23 | C12 → 96.65 (NONE), 97.12 (ALL) |
| α(Br) | −0.0012 /°C | −0.12 %/°C, written as −0.0012 | 0 | 0 | none |
| density | 0.0075 g/mm³ | 7.5 g/cm³, written as 0.0075 | 0 | 0 | none |
| Hcb, (BH)max, μrec | not read | 907.2, 318.3, 1.05 | 0 | 0 | none |

The 23 cells are Temperature design C7–C10, C12, C13, C15, C23, C24, C49, C50, C56–C61, C101, C104, C105, C153, C181 and C182. Note the sign: a steeper β *raises* every onset, because the calibration offset absorbs more than the knee loses (a Pc = 1 magnet is pinned to the 150 °C rating). Adopting Arnold's β makes the default design look 4 °C safer.

### 6.3 All 15 parts (NONE mode, part on both rings)

| Part | Grade | Br lib → grade (T) | Hcj 1592 → grade | β −0.50 → grade | Cells moved: Br / Hcj / β / all | C93 today → with grade Br (N·m) | C12 today → with grade Hcj + β (°C) |
|---|---|---|---|---|---|---|---|
| B842SH | N42SH | 1.29 → 1.30 (E3) | → 1591.5 | → −0.55 | 233 / 23 / 23 / 233 | 2.6473 → 2.6885 | 92.55 → 96.65 |
| B842 | N42 | 1.30 → 1.30 | → 954.9 | → −0.62 | 0 / 24 / 24 / 24 | 2.6885 → 2.6885 | 23.06 → −3.52 |
| B842-N52 | N52 | 1.45 → 1.45 | → 875.4 | → −0.62 | 0 / 24 / 24 / 24 | 3.3447 → 3.3447 | 30.73 → 0.17 |
| B822 | N42 | 1.30 → 1.30 | → 954.9 | → −0.62 | 0 / 24 / 24 / 24 | 0.8126 → 0.8126 | 23.06 → −3.52 |
| B862 | N42 | 1.30 → 1.30 | → 954.9 | → −0.62 | 0 / 24 / 24 / 24 | 3.1458 → 3.1458 | 23.06 → −3.52 |
| B882 | N42 | 1.30 → 1.30 | → 954.9 | → −0.62 | 0 / 24 / 24 / 24 | 3.1612 → 3.1612 | 23.06 → −3.52 |
| B882-N52 | N52 | 1.45 → 1.45 | → 875.4 | → −0.62 | 0 / 24 / 24 / 24 | 3.9327 → 3.9327 | 30.73 → 0.17 |
| B861 | N42 | 1.30 → 1.30 | → 954.9 | → −0.62 | 0 / 24 / 24 / 24 | 1.5473 → 1.5473 | 23.06 → −3.52 |
| B881 | N42 | 1.30 → 1.30 | → 954.9 | → −0.62 | 0 / 24 / 24 / 24 | 1.5473 → 1.5473 | 23.06 → −3.52 |
| B442 | N42 | 1.30 → 1.30 | → 954.9 | → −0.62 | 0 / 24 / 24 / 24 | 1.1881 → 1.1881 | 23.06 → −3.52 |
| BX042SH | N42SH | 1.29 → 1.30 (E3) | → 1591.5 | → −0.55 | 233 / 26 / 26 / 233 | 5.6020 → 5.6892 | 92.55 → 96.65 |
| BX082SH | N42SH | 1.29 → 1.30 (E3) | → 1591.5 | → −0.55 | 233 / 26 / 26 / 233 | 6.5869 → 6.6894 | 92.55 → 96.65 |
| M5044 | N50 | **1.42 → 1.41** | → 875.4 | → −0.62 | 233 / 24 / 24 / 233 | 1.3108 → 1.2924 | 29.18 → −2.50 |
| M5045 | N50M | **1.42 → 1.41** | → 1114.1 | → −0.675 | 233 / 24 / 24 / 233 | 0.5081 → 0.5010 | 49.18 → 42.50 |
| M5026 | N50 | **1.42 → 1.41** | → 875.4 | → −0.62 | 232 / 24 / 24 / 232 | 2.1883 → 2.1576 | 29.18 → −2.50 |

Findings:
- Tmax, α and density reproduce every part.
- Hcj and β differ for every part, and for the 12 non-N42SH parts that difference is the A6 fix. For example, the N42 parts' governing limit falls from 23.06 to −3.52 °C, below the 50 °C operating point. By construction the fix cannot be workbook-exact.
- Today a blank or unknown part gives `tmax = "n/a"`, an uncalibrated demag offset (0) and "unknown" temperature checks, and the differential sets include `''` and `b842sh`. "Any grade with manual dimensions" must therefore be a **new** mode. The existing manual mode, which has no grade, and the manual Br inputs (Calculator!C17 and C27) must stay as they are.
- The demag block uses the inner ring's Br and Tmax against the outer blocks' reverse fields. With a grade per ring that choice becomes visible. At the defaults both rings carry the same part, so nothing moves.
- The Tmax cell counts for decision 2 (60 °C on the arcs) were not measured. They must be counted when the deviation is registered.

### 6.4 Default materials against the workbook constants (NONE mode)

Every input-settable row was also run in ALL mode, where it moved the same number of cells. The 6061 conductivity was measured in Python only, because the Rust constant is hard-wired (`api.rs:261`).

| Constant (cell) | Engine | Library | Bit-equal | Cells moved | Main effect |
|---|---|---|---|---|---|
| Back-iron design flux density `steel.bsat_T` (Materials!C13) | 1.5 T (a design value, not saturation) | 4140 Bsat null; `design_flux_density_T` = 1.5 (workbook) | n/a | 0 with a design-flux field of 1.5. An illustrative 2.0 T moves 5. | C36, C104, C105, Materials!C20, C22; at 2.0 T "Too thin … 2.0 mm" becomes "OK" |
| 4140 σ (Materials!C14) | 4.5e6 S/m | 4.33e6 (Lucefin) | no | 40 | C130 2.477 → 2.445 W; C17, C18; skin depth C115 |
| 4140 μr incremental (Materials!C15) | 200 | 363 (secant at 1.5 T) | no | 40 | C130 2.477 → 2.047 W; C18 89.77 → 85.47 °C |
| 4140 cp (Materials!C16) | 473 | 461 (Lucefin) | no | 24 | C141 82.19 → 80.71 J/K; C143 274.0 → 269.0 s |
| 4140 CTE (Materials!C17) | 12.3e-6 | 12.2e-6 (AZoM); 12.3 is a tier-3 alternate | no | 4 | C94, C104 (46.13 → 45.77 MPa NONE; 11.68 → 11.59 ALL), C105, C201 |
| 4140 E (Materials!C18) | 205 | 205 | yes | 0 | none |
| 4140 density, Materials!C19 | 7.85 (not consumed) | 7.85 | yes | 0 | none |
| Steel density, Metal design!C132 | 0.00785 g/mm³ | 7.85/1000 is bit-equal | yes | 0 | none |
| 316L σ (Temperature design!C111) | 1.35e6 | 1.351e6 (AK, converted) | no | 37 | sleeve and liner loss; C130 +7.6e-5 relative |
| 316L density (Metal design!C44) | 0.008 g/mm³ (sleeve, liner and endplates) | 7.99 (AK); Sandmeyer 7.90 | no | 32 | C46, C181, C114 173.793 → 173.781 g, C115, C141 |
| 316L cp (Temperature design!C139) | 500 | 500 | yes | 0 | none |
| 6061 σ (Materials!C43, hard-wired `api.rs:261`) | 2.5e7 | 2.494e7 (Kaiser 43 % IACS) | no | 37 | cap-face loss C128; C130 −3.1e-4 relative |
| Aluminium density (Metal design!C42) | 0.0027 g/mm³ | 2.7/1000 is bit-equal | yes | 0 | none |
| Aluminium cp (Temperature design!C140) | 900 | 896 (Kaiser, **at 100 °C**) | no | 23 | C141 82.194 → 82.185 J/K |
| 6061 and 7075 yield, 7075 row | 276, 503 (not read) | 241 and 462 min (this report) | n/a | 0 | none (4.2 M5, M6) |
| All sourced values of the three default materials at once (μr kept at 200, bsat at 1.5) | | | | 63 | C130 2.477 → 2.445 W; C141 82.19 → 80.70 J/K; C114; C104 |

Findings beyond the numbers:
- A5 says "saturation feeds the wall-thickness check", but the engine feeds a design flux density into t_bi = B_gap·τp/(π·B) (decision 20).
- Steel density is stored twice, and the sleeve density also sets the endplates (4.3).
- The clamp alloy table is separate (4.3).

### 6.5 Housing autofit (A1) and the harmonic set (A3/D7)

Classes: **D** = already derived today; **R** = an input with an existing engine rule; **N** = an input with no rule anywhere in the engine or the README.

| Dimension (cell) | Default | Class | Rule | Reproduces the default? | Cells moved |
|---|---|---|---|---|---|
| Cup OD (Calculator!C62); pocket corner radius C61; wall at flats C63; sleeve ID/OD; liner OD/ID; endplate OD | 41.326 mm etc. | D | from the block apothem, bondline and N | yes | 0 |
| **Cup wall at the corners** (C122) | **1.8 mm** | R | Materials check: wall ≥ t_bi, advice rounded up to 0.1 mm (`materials.rs:163-172`) | **no**: the rule gives 2.0 mm (t_bi = 1.904 NONE, 1.919 ALL) | 57 (ALL): Materials!C22 "Too thin …" → "OK"; C105 "Too thin" → "Thickness OK"; C62 41.33 → 41.73; C111, C114, C115; C141 82.19 → 83.94; gap-sweep G and AA |
| Sleeve and liner thickness (C25, C26) | 0.1 / 0.2 mm | R (a check only) | running clearance C35 ≥ 0.2 mm (C36) | the rule cannot be met: the minimum running clearance is −0.103 mm, and even with a sleeve and liner of 0 it is 0.197 mm | n/a |
| Inner back apothem (Calculator!C8) | 10.15 mm | R (pole sweep only) | max(w/(2 tan(π/N)) + 0.05, bore/2 + keyway + 2.5) | no (9.822 mm). A1's "ring radius" variable must start from 10.15. | not in autofit scope |
| Hub wall past the keyway (C53) | 3.40 mm | R (help text: "keep ≥ 2.5 mm") | a check | yes | 0 |
| Cup cavity depth (C124) | 15.5 mm | N | none. Two different identities both give exactly 15.5 (L + thread engagement + cap axial; retainer span + rear endplate), so neither is evidence of intent. | only with an invented rule | n/a |
| Retainer span C172, hub length C123, rear web C125, boss length C126, endplates C170, C171, C182 | 14.5, 13.0, 2.5, 13.0, 0.5, 1.0, 4.5 | N | none | n/a | n/a |
| Boss OD (Metal design!C127) | **22 mm** | N | none. The clamp calculator has its own boss OD, Shaft clamps!C35 = **25 mm**. | no (25 if taken from the clamp) | mass, hybrid mass, C141 |
| Cap OD, thread, engagement, axial (C166, C169, C168, C133) | 42.8, M41 × 0.5, 2.0, 0.8 | N | none | n/a | any rule moves C135 and C136 |

The space claim at the defaults, in both modes:
- rotating OD C135 = 42.8 mm, set by the cap input, not by the cup (41.33); reserve C136 = 0.20 mm of 43;
- overall stack C134 = 31.8 mm; reserve C139 = 3.2 mm of 35;
- large-diameter stack C137 = 18.8 mm; reserve C138 = 1.2 mm of a **20 mm** bay (C129).

The envelope is stepped, while A1 describes a plain 43 × 35 cylinder (decision 30). Two inconsistencies already exist: the M41 cap thread is smaller than the 41.33 mm cup body OD, and the boss OD is 22 mm here but 25 mm in the clamp sheet. The differential data set all 17 housing inputs, so autofit must keep every one of them settable.

**Harmonic set (A3/D7).** A prototype, `peak_off_half_pitch_general(ns, a)`, was tested. It computes dT/dx = cos x · Q(u) with u = cos²x and a stable recurrence for cos(nx)/cos x. It scans for sign changes on 1,024 cells and then bisects.

| Check | Result |
|---|---|
| Default outputs, NONE and ALL | bit-identical; `e7_leaves_every_default_cell_bit_for_bit` passes |
| E7 probes | pass (Calibration!C44 differs by 1 ulp, inside parity) |
| The three cancellation tests; 3,000 triples against the closed form | pass; 0 None/Some disagreements; largest dx 5.0e-16 rad |
| 11 harmonics against brute force (3,000 spectra) | 0 misses |
| All 3,391 differential sets with every correction on | 431 differ in at least one bit; largest relative difference 7.4e-13; 0 parity failures |
| `peak_angle_is_the_e7_gate` | **fails** on exact equality (1 ulp); it needs a tolerance |
| Cost | 3.35 µs per call against 0.054 µs; about 80 µs per frame |
| Failure mode | A root pair inside one scan cell is missed. With a1 > 0 it flipped the result twice in 100,000 cases, with a shortfall ≤ 1.4e-11. A Sturm or Q′ guard closes the gap. |

Also: Calibration's own harmonic formula (Calibration C40–C42) moves by 2.3e-16 to 3.5e-16 if it is routed through `shear_stress` (parity holds, bit-identity does not). The schema has fixed cells for n = 1, 3 and 5 only. And `static_data.rs` pins `HARMONICS` to Python's list.

**A3 assumptions.** Every A3 item except the harmonic set is already an input with a workbook cell, and 14 fields carry `.assumption()`. The addendum's example "the knee fraction is hard-coded" is wrong: it is Temperature design!C46. The end-effect coefficient is two inputs (`coupling.c_end` and `calibration.c_end`) that the differential data vary independently, so one assumption row must map to both. The clamp friction assumption must name which input it is: `clamps.friction`, flagged, or `clamps.joint_friction` (C66), not flagged.

### 6.6 The mismatches, and where each is decided

| # (defaults study) | Mismatch | Cells moved | Decision |
|---|---|---|---|
| 1 | N42SH Hcj 1592 against 1591.5 | 23 | 17 |
| 2 | N42SH β −0.50 against −0.55 %/°C | 23 | 18 |
| 3 | Per-grade Hcj and β for the 12 non-N42SH parts (the A6 fix), plus the positive-ferrite-β branch | 24–26 per part when selected | 19 |
| 4 | Arc parts' Br 1.42 against 1.41 T | 232–233 when selected | 2 (combined with the vendor's 60 °C) |
| 5 | Back-iron "saturation" against the 1.5 T design flux density | 0 or 5 | 20 |
| 6 | 4140 σ 4.5e6 against 4.33e6 | 40 | 21 |
| 7 | 4140 μr 200 against 363 | 40 | 22 |
| 8 | 4140 cp 473 against 461 | 24 | 23 |
| 9 | 4140 CTE 12.3 against 12.2 | 4 | 24 |
| 10 | 316L σ 1.35e6 against 1.351e6; density 8.0 against 7.99 | 37; 32 | 25 |
| 11 | 6061 σ 2.5e7 against 2.494e7; aluminium cp 900 against 896 | 37; 23 | 26 |
| 12 | Cup-wall autofit: 1.8 against 2.0 mm | 57 | 27 |
| 13 | Housing dimensions with no rule | n/a | 28 |
| (choice) | Harmonic peak search | 0 at the defaults | 29 |

---

## 7. A2 scope

### 7.1 Counts (from running the engine)

The spec gives no count of equations; its scope is "each explained result". It asks for five deliverables: an explanation layer (one equation record per result, with 7 attributes), a drift guard, hover, an equation panel and a typesetter. A3 and A4 also need a **term graph**, not just isolated formulas.

| Module | Result fields | Values at defaults |
|---|---|---|
| calibration.rs | 23 | 23 |
| model.rs (ModelResults 73 + MassResults 6) | 79 | 79 |
| metal_design.rs (Retainer 9 + MetalDesign 49) | 58 | 58 |
| materials.rs | 7 | 7 |
| temperature.rs (10 groups; 130 celled + 1 uncelled) | 131 | 131 |
| clamps.rs (34 scalars + 34 `ScrewRow` columns) | 68 | 204 |
| sweeps.rs (26 `SweepRow` columns, used by both sweeps) | 52 | 494 |
| **Total** | **418** (332 scalar + 86 table-column paths) | **996** |

- There are 392 distinct declarations, because `SweepRow` is declared once.
- 989 of the 996 values have a workbook cell. The README's "330 result cells" counts the celled scalars.
- By type: 357 `f64`, 8 `i64`, 29 text and 24 number-or-text fields. So **53 fields are verdicts or sentinels**, which are comparisons against thresholds rather than formulas.
- Inputs: **161**, of which **14 are flagged `assumption`**. 160 have a workbook value at the defaults; the exception is `metal.measured_drag_Nm`, which defaults to None.
- **Headline (dashboard): 15 paths.** They split into model 4, mass 1, metal 5, materials 1, temperature 3 and clamps 1, and 5 of the 15 are verdict text.

| Chain | Fields | Sub-topics |
|---|---|---|
| Torque | 54 | shear stress 7; harmonics 15; pull-out 9; end effect 2; calibration 6; hot/cold 15 |
| Temperature | 45 | Br(T) 10; torque against T 13; limits 22 |
| Demagnetization | 18 | knee and reference 3; onsets 8; torque at limits 2; margins 5 |
| Slip heating | 30 | eddy losses 18; heat capacity 2; time constant 4; time to limit 6 |
| Clamps | 19 (11 scalar + 8 table columns) | preload 7; friction torque 12 |
| **Union of the five chains** | **152** (36 % of 418) | 14 paths sit in two chains |

Seven headline paths lie outside the chains: `model.gearbox_input_ripple_Nm`, `model.cup_od_mm`, `mass.total_g`, `metal.min_running_clearance_mm`, `metal.clearance_check`, `materials.cup_wall_check` and `clamps.recommended`. **Chains plus headline = 159 paths.** The A1 geometry callouts come on top of that. They are not yet enumerated and should be counted when the A1 view is specified.

### 7.2 What the engine does and does not expose

- **There is no permeance-coefficient result.** The operating-point reverse fields are stored inputs (M3 replaces them), so A4's "knee and permeance" note has no permeance result to attach to.
- **The knee is an input** (`knee_fraction`). The product hk = knee × Hcj20 is an intermediate that no result field exposes.
- Other intermediates that would need a term id: the E7 peak angle, the `backiron` choice between s_n_iron and s_n_free, and the Volkersen internals behind `temperature.mismatch.peak_shear_*`.
- Sweep columns recompute the torque chain for each row. They would either reuse the model's records with a row-scoped term set, or show their cell only.
- **Drift-guard scale:** the 3,391 differential cases (300–410 per module file, plus 101 in `full.json`) plus the defaults. For the 53 text fields the guard compares strings or sentinels.

### 7.3 Scope options

- **Every result field:** 418 records. The drift guard would run on all 996 values, and the 53 verdict fields would need records that are comparisons.
- **Dashboard, geometry callouts and the five chains:** 159 paths plus the callouts. Every other field shows its workbook cell and label, which the registry already carries. Decision 31.

---

## 8. Decisions for the user

**Approved (user, 2026-09-30): option A on all 31 decisions.** The spec text fixes of decision 30 are applied in the same commit.

The recommended option is always A, with a one-line reason. Decisions 1–7 concern the data tables, 8–16 the corrections, 17–28 the defaults mismatches (one each; defaults mismatch 4 is inside decision 2), 29 the harmonic peak search, 30 the spec text, and 31 the A2 scope.

1. **A6 grade table and part mapping (NdFeB basis).**
   - **A (recommended):** approve the table with the resolutions R1–R17: the K&J basis for every NdFeB cell (extending D1), the 15 parts mapped as in section 3, and the four extra N42SH blocks recorded but not added. Why: every cell matches its source, and the basis is the one D1 already chose for the default grade.
   - B: switch NdFeB to Arnold's guaranteed minima: Br 0.7–2.7 % lower, N48 Hcj 875, N52 Tmax 60, N33AH (BH)max 215, N50M Hcb 1035. This reverses D1 (N42SH Br 1.28 T).
   - C: A, plus add the four extra K&J N42SH blocks (B421SH, BX088SH, BY042SH, BY0X02SH) as Rust-only parts.
2. **SuperMagnetMan arc parts M5044, M5045, M5026** (includes defaults mismatch 4).
   - **A (recommended):** register a deviation that applies the vendor's 60 °C Tmax to all three parts and maps M5045 to N50, as its specification grid says. Br stays at the workbook's 1.42 T. Why: the vendor's page is the only part-specific source, and where the page contradicts itself (title N50M, grid N50), the safe reading is taken.
   - B: register 60 °C for all three, and keep M5045 as N50M per its title.
   - C: keep the workbook values (80 / 100 / 80 °C, N50M, 1.42 T) as "workbook" entries, with a visible part warning quoting the vendor page. No cells move.
3. **Ferrite Y30 β(Hcj).**
   - **A (recommended):** +0.35 %/°C (Alliance C-5). Why: it is the conservative value for the cold-demagnetization check that the ferrite branch exists for, and the two +0.27 sources may be one figure.
   - B: +0.27 %/°C (Eclipse; Arnold TN 0303 Ferrite 8).
   - Either option needs the positive-β formula branch (`demag_onset_C` uses `beta.abs()`), and a cold-case test run through the custom-dimension mode.
4. **SmCo sub-grades.**
   - **A (recommended):** keep Arnold Recoma 20, 26 and 30 as they are, named by sub-grade (Hcj min 2000 / 1200 / 1040 kA/m), with the flags R8 and R11. Why: every cell matches one manufacturer grade sheet, and the low-Hcj variants are the conservative reading of "grade 26" and "grade 30".
   - B: represent the high-coercivity variants (Recoma 26HE; 30HE or 30S; Hcj min 1500 / 1500–1750). This needs their full sheets sourced first.
5. **A5 materials table.**
   - **A (recommended):** approve the table with the resolutions M1–M24: published minimum yields as the library value with typicals as `typ`, resin-producer values for the disputed polymer cells, and the 17-4 H1150 σ relabelled as an H900 proxy. Workbook defaults stay unchanged. Why: it applies one rule everywhere, and zero engine cells move.
   - B: approve the original table (typical yields, converter polymer values, the Rolled Alloys σ).
6. **What the `416_annealed` entry represents.**
   - **A (recommended):** standard martensitic 416 throughout. Carpenter for σ, ρ, cp and yield (276, typical, flagged because no minimum is published); E 200 GPa (Smiths); Bsat 1.60 T, flagged as a proxy (SMAG Table 4, 410 at 0.15 % C); CTE re-sourced from a standard-416 sheet (Smiths 9.9 or Rolled Alloys 10.1, range to be confirmed). Why: "416 stainless" bar is martensitic, and the lower Bsat is the safe side for the low-saturation warning.
   - B: the magnetic-quality Zapp 1.4005 IA variant throughout: yield ≥ 230 min, E 215, CTE 10.5, Js ≥ 1.70 T, σ ≤ 1.81e6, ρ 7.7, cp 460, maximum μr ≥ 2000.
   - C: keep the present mix, with a caveat on every field that comes from Zapp.
7. **1018 and 12L14 conductivity** (aggregator sources only; about 1.4 times apart).
   - **A (recommended):** keep the higher values (1018: 5.85e6 derived; 12L14: 5.75e6), flagged ±30 %, with the 7 % IACS values as alternates. Why: steel slip loss rises with √σ, so the higher value is the conservative one for heating.
   - B: follow the majority of the secondary sources (1018 keeps 5.85e6; 12L14 takes 4.12e6).
   - C: set both to null until a producer sheet is found.
8. **E15, heat capacity at the part's own specific heat.**
   - **A (recommended):** approve as proposed, with c_cup on E9's gate, c_hub on C6 ≠ 1, and both reading the same flag as the density. Why: it removes a 27 % understatement of C, and every cell reproduced.
   - B: document only. C stays understated, which is conservative.
9. **E16, the removed web disc priced at the cup's density.**
   - **A (recommended):** approve, with ρ_cup passed from the mass model (one source of truth) and the labels handled by decision 14. Why: the hybrid mass is 2.265 g low at C6 = 0, and the fix has no effect elsewhere.
   - B: document only.
10. **E17 loss formula.**
    - **A (recommended):** the closed form T1. Why: it is never non-conservative, it is within 1.5 % of the exact solution at the operating point and within 1.4 % up to 2000 rpm at the default radii, and it takes a few lines of real arithmetic. At the far corner of the slider box it overstates the hub loss by up to 7.7 times.
    - B: the exact layered formula in the engine: hand-written complex sqrt and tanh, about 30 lines plus tests. It is right across the whole slider box, including the thick hub at large radius.
    - C: swap the constants only. Rejected: 29 times off against the recommended form.
11. **E17 end factor.**
    - **A (recommended):** apply C114 = 0.7. C19 is reported as inside the model's uncertainty: it is finite only for f_end ≥ 0.646 (M2: 0.663). Why: this is the workbook's convention for every thin conductor, 0.7 lies inside the physical bracket 0.467–0.726, and the high case already multiplies the loss by 3.
    - B: none (f = 1). Aluminium-part losses × 1.43; total 3.823 W; C19 194.6 s (M2: 202.0 s).
    - C: the Russell–Norsworthy factor for the actual cup (0.605). Total 2.631 W; C19 "never".
12. **E17 field inputs, amended.**
    - **A (recommended):** three stored free-space inputs, pinned at 4 s.f. as the registry probe basis: hub 0.07832 T (cup aluminium), cup 0.08764 T, web 6.837e-6 T²·m². The hub field is C116/2 = 0.1035 T when the cup is steel (E9 off). M3 computes them live, drops the doubling and E5's ×4 for aluminium parts, and re-blesses the probes. Why: the physics is right for both cup materials, and the probes cannot drift until M3.
    - B: scale the stored steel values by fixed ratios (0.378, 0.410, 0.661). Rejected: hidden, geometry-specific constants.
    - C: keep the steel-circuit fields. Rejected: 4.1 times the correct loss.
13. **E17 hub scope.**
    - **A (recommended):** include the hub (C123). Why: it uses the same physics and gate, and the workbook's mass model already makes the hub aluminium at C6 = 0.
    - B: cup and web only. C123 stays steel-priced (0.2259 W), and the total becomes 2.950 W.
14. **Labels that say "steel"** (C147, C189, C123–C125).
    - **A (recommended):** keep the labels and reword the help through the registry's `workbook_help` (for example "4140 at back iron = 1; 6061 aluminium when there is no back iron (E9)"). Why: `tests/python_schema.rs` compares labels with Python, and this needs no new mechanism.
    - B: relabel them material-neutral, which needs a new `workbook_label` escape hatch in the schema test.
15. **Registry probes for corrections that depend on E9.**
    - **A (recommended):** add a `Deviations::with(id)` / `without(id)` combinator. E15–E17 record their dependency on E9, and their back-iron-0 probes run on top of E9. Why: `only(E16)` changes nothing and `only(E15)` / `only(E17)` move only the hub, so NONE-against-`only` probes would test little.
    - B: probe `ALL without Ek` against `ALL` (the M2 column).
    - C: define E16 independently of E9, pricing the disc at aluminium whenever C6 = 0. Rejected: it disagrees with a steel C111 when E9 is off.
16. **Candidate E18: the adhesive mismatch screen uses 4140's CTE and E for an aluminium hub.**
    - **A (recommended):** register it in A5 with E15's gate (the hub material follows the hub density rule). Why: on the M2 basis at C6 = 0 it flips C106 and C202 to "Above", and it reproduced exactly.
    - B: document only.
17. **(Defaults 1) N42SH Hcj.**
    - **A (recommended):** 1592 kA/m (the Arnold N42SH sheet). Why: it is a sourced grade minimum and moves zero cells.
    - B: register K&J's 1591.5 (23 cells; C12 −0.02 °C).
18. **(Defaults 2) N42SH β(Hcj).**
    - **A (recommended):** keep −0.0050 /°C as the workbook default, with Arnold's −0.0055 as the library reference, and send any change to M1 physics review first. Why: the steeper β makes the default design look 4 °C safer, which is not a conservative correction.
    - B: register −0.0055 (23 cells; C12 92.55 → 96.67 °C).
19. **(Defaults 3) Per-grade Hcj and β, the A6 correctness fix.**
    - **A (recommended):** register it as a deviation. It is gated so that NONE keeps the single input, with probes on B842 at both rings (C12 23.06 → −3.52 °C) and a ferrite cold case through the custom-dimension mode. The positive-β branch is included, and `hcj20_kA_m` / `beta_hcj_per_C` stay as overrides that win when set. Why: it is the correction the spec asks for, and only a registered deviation keeps the parity and differential layers exact. It is default-neutral only together with 17A and 18A.
    - B: every grade carries the workbook's 1592 / −0.005. This defeats the fix.
20. **(Defaults 5) Back-iron saturation against the design flux density.**
    - **A (recommended):** a separate library field `design_flux_density_T` (4140 = 1.5 T, workbook) feeds the wall check. Bsat is informational and drives only the low-saturation warning. Why: the check is a design limit by intent, and the spec's default facts list the "Too thin" verdict.
    - B: register a saturation-based wall check. Any saturation-class value flips "Too thin" to "OK".
21. **(Defaults 6) 4140 σ.**
    - **A (recommended):** workbook 4.5e6 as the default, Lucefin 4.33e6 as the reference. Why: 3.9 % is inside the spread of a placeholder-level loss model that uses a high multiplier of 3.
    - B: register 4.33e6 (40 cells; slip loss 2.477 → 2.445 W).
22. **(Defaults 7) 4140 μr.**
    - **A (recommended):** workbook 200. Why: 363 is a secant value at 1.5 T, not the incremental value the formula needs.
    - B: register 363 (40 cells; loss 2.477 → 2.047 W).
23. **(Defaults 8) 4140 cp.**
    - **A (recommended):** workbook 473. Why: 2.5 %, and 473 lies between Lucefin's 20 °C and 100 °C values (461, 479).
    - B: register 461 (24 cells).
24. **(Defaults 9) 4140 CTE.**
    - **A (recommended):** workbook 12.3e-6 /°C. Why: 0.8 %, and a tier-3 source prints 12.3.
    - B: register 12.2e-6 (4 cells).
25. **(Defaults 10) 316L σ and density.**
    - **A (recommended):** workbook 1.35e6 S/m and 8.0 g/cm³. Why: both are below 1e-4 relative, and the density also moves the endplates, which A5 does not list.
    - B: register 1.351e6 and 7.99 (37 and 32 cells).
26. **(Defaults 11) 6061 σ and aluminium cp.**
    - **A (recommended):** workbook 2.5e7 S/m and 900 J/(kg·K). Why: both are below 3.1e-4 relative, and 896 is quoted at 100 °C.
    - B: register 2.494e7 and 896 (37 and 23 cells).
27. **(Defaults 12) Cup-wall autofit.**
    - **A (recommended):** keep `metal.cup_wall_corner_mm` as an input at 1.8 mm, with autofit showing the rule's 2.0 mm as a suggestion. Why: it keeps the spec's default fact "Too thin", and the differential data need the input.
    - B: register "autofit sizes the wall to the ceiling of t_bi at 0.1 mm" (57 cells, two verdict flips, cup OD 41.33 → 41.73 mm).
28. **(Defaults 13) Housing dimensions with no rule.**
    - **A (recommended):** keep them as inputs, so that autofit covers only classes D and R, and raise two M4 design questions: the M41 cap thread below the 41.33 mm cup OD, and the boss OD of 22 mm against 25 mm. Why: two different identities reproduce the cup depth, so neither shows the workbook's intent.
    - B: define new rules and register them.
29. **Harmonic peak search (A3/D7).**
    - **A (recommended):** one general search with a Sturm or Q′ root-pair guard, and `peak_angle_is_the_e7_gate` changed to compare within 1e-12. Why: it is DRY, the default cells stay bit-identical, and the E7-active results are within 7.4e-13.
    - B: keep the closed form for {1, 3, 5} and use the general scan only for larger sets. This is bit-identical everywhere but has two code paths.
30. **Spec text fixes.**
    - **A (recommended):** add N50 and N50M to the A6 grade list; add the 20 mm large-diameter bay to the A1 envelope; reword A5's "saturation feeds the wall-thickness check" to "the design flux density feeds …". Why: the spec's own test "every library part resolves to a grade" fails on its present list.
    - B: leave the spec as it is.
31. **A2 scope for v1.**
    - **A (recommended):** the dashboard, the geometry callouts and the five chains (159 paths plus the callouts). Every other field shows its workbook cell. Why: it covers every number on screen and every idea the A4 notes teach, without 266 records for geometry, sweeps and verdict text.
    - B: every result field (418).

Once the user has answered, these need entries: `docs/ai/04-memory.yaml` (open questions resolved) and `docs/ai/05-update-tracker.md`.

---

## Sources

Grades:

[^kj]: [K&J Magnetics, Neodymium Magnet Specifications & Tolerances](https://www.kjmagnetics.com/neodymium-magnet-specifications.asp) (magnetic, thermal and physical tables).
[^kjht1]: [K&J Magnetics, High Temperature Neodymium Magnets, page 1](https://www.kjmagnetics.com/products/high-temp-magnets).
[^kjht2]: [K&J Magnetics, High Temperature Neodymium Magnets, page 2](https://www.kjmagnetics.com/products/high-temp-magnets?pg=2).
[^arncat]: [Arnold Magnetic Technologies, NdFeB Magnet Catalog, Rev. 210607](https://www.arnoldmagnetics.com/wp-content/uploads/2019/06/Arnold-Neo-Catalog.pdf) (summary tables, Tw max).
[^arn35]: [Arnold N35 datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35-151021.pdf).
[^arn42]: [Arnold N42 datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf).
[^arn48]: [Arnold N48 datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N48-151021.pdf).
[^arn52]: [Arnold N52 datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf).
[^arn50]: [Arnold N50 datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50-151021.pdf).
[^arn42m]: [Arnold N42M datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42M-151021.pdf).
[^arn50m]: [Arnold N50M datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N50M-151021.pdf).
[^arn42h]: [Arnold N42H datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42H-151021.pdf).
[^arn42sh]: [Arnold N42SH datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf).
[^arn38uh]: [Arnold N38UH datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38UH-151021.pdf).
[^arn35eh]: [Arnold N35EH datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N35EH-151021.pdf).
[^arn33ah]: [Arnold N33AH datasheet](https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N33AH-151021.pdf).
[^recoma]: [Arnold Recoma sintered SmCo, combined datasheet 160301](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf) (p2 summary; p4 Recoma 20; p5 Recoma 22; p8 Recoma 26; p12 Recoma 30).
[^tn0303]: [Arnold TECHNotes TN 0303, Temperature Effects on Magnetic Output](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/TN_0303_rev_150715.pdf) (family coefficients, averages over about 20–120 °C).
[^arn2006]: [Arnold (Constantinides), APEEM 2006, bonded and sintered magnets](https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Manufacturing-and-performance-comparison-between-bonded-and-sintered-permanent-magnets-Constantinides-APEEM-2006-psn-hi-res.pdf) (slide 23: recoil permeability).
[^eclfer]: [Eclipse Magnetics, Ferrite/Ceramic Magnets Datasheet](https://www.eclipsemagnetics.com/site/assets/files/19602/ferrite_ceramic_datasheet.pdf).
[^hyab]: [Hyab, ferrite material specification](https://hyab.com/Ferrit-materialspecifikation.php?lang=en).
[^allc5]: [Alliance LLC, Ferrite C-5](https://allianceorg.com/magnetic-materials/ceramic-magnets/ferrite-c-5/).
[^allbcn]: [Alliance LLC, Compression Bonded Neo](https://allianceorg.com/magnetic-materials/bonded-magnets/compression-bonded-neo/).
[^eclsmco]: [Eclipse Magnetics, Samarium Cobalt Technical Data Sheet](https://www.eclipsemagnetics.com/site/assets/files/19544/eclipsemagnetics_smco_datasheet.pdf).
[^eclnd]: [Eclipse Magnetics, neodymium grades data](https://www.eclipsemagnetics.com/site/assets/files/23180/neodymium_grades_data.pdf) (skeptic cross-check).
[^hgt]: [HGT Advanced Magnets, sintered NdFeB specifications](https://www.advancedmagnets.com/wp-content/uploads/2020/07/Sintered-Neodymium-Iron-Boron-NdFeB-Magnets-Specifications.pdf) (skeptic cross-check).
[^intemag]: [Intemag, SmCo magnetic material properties](https://www.intemag.com/smco-magnetic-materials-properties-data) (skeptic cross-check).
[^smm44]: [SuperMagnetMan M5044](https://supermagnetman.com/products/m5044) (specification grid).
[^smm45]: [SuperMagnetMan M5045](https://supermagnetman.com/products/m5045) (specification grid).
[^smm26]: [SuperMagnetMan M5026](https://supermagnetman.com/products/m5026) (specification grid).

Materials:

[^carp416]: [Carpenter, CarTech 416 Stainless datasheet](https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/416.pdf).
[^zapp416]: [Zapp, Ergste 1.4005 IA (AISI 416 solenoid grade)](https://www.zapp.com/fileadmin/_documents/Downloads/materials/stainless_steel/PW/en/AISI-416-1.4005-data-sheet.pdf).
[^carpblog]: [Carpenter, Magnetic Properties of Stainless Steels](https://www.carpentertechnology.com/blog/magnetic-properties-of-stainless-steels).
[^smag]: [MagWeb, SMAG Handbook v7](https://www.magweb.us/wp-content/uploads/2021/08/SMAG-Handook-Version-7.pdf) (p.39 Table 4; p.41 Js correlation; p.44–45 steels).
[^lucefin]: [Lucefin, 42CrMo4 technical card](https://www.lucefin.com/wp-content/files_mf/152353604042CrMo4.pdf).
[^azom4140]: [AZoM, AISI 4140](https://www.azom.com/article.aspx?ArticleID=6769).
[^otai4140]: [Otai Special Steel, AISI 4140](https://otaisteel.com/aisi-4140-material-data-sheet/) (tier 3).
[^kjshield]: [K&J Magnetics blog, Magnetic Shielding Materials](https://www.kjmagnetics.com/blog/magnetic-shielding-materials) (tier 3).
[^azom1018]: [AZoM, AISI 1018](https://www.azom.com/article.aspx?ArticleID=6115).
[^twm1018]: [The World Material, AISI 1018](https://www.theworldmaterial.com/astm-sae-aisi-1018-carbon-steel/).
[^mif1018]: [MakeItFrom, hot rolled 1018](https://www.makeitfrom.com/material-properties/Hot-Rolled-1018-Carbon-Steel/).
[^azom12l14]: [AZoM, AISI 12L14](https://www.azom.com/article.aspx?ArticleID=6604).
[^twm12l14]: [The World Material, 12L14](https://www.theworldmaterial.com/12l14-steel/).
[^mif12l14]: [MakeItFrom, 12L14](https://www.makeitfrom.com/material-properties/SAE-AISI-12L14-G12144-Carbon-Steel).
[^ti12l14]: [Titanium Industries, AISI 12L14](https://titanium.com/alloys/carbon-steels/carbon-steel-aisi-12l14/) (skeptic cross-check).
[^armco174]: [Cleveland-Cliffs / AK Steel, ARMCO 17-4 PH bulletin](https://www.aksteel.nl/files/downloads/clf_datasheet_armco_17-4_ph_pdb_euro_102022_89.pdf) (Tables 2, 3, 7, 17; Fig. 3).
[^rolled174]: [Rolled Alloys, 17-4 data sheet](https://www.rolledalloys.com/wp-content/uploads/17-4_Data-sheet-rolled-alloys.pdf).
[^ams5643]: [AMS 5643 (17-4 PH) minima, Space Materials Database](https://www.spacematdb.com/spacemat/manudatasheets/17-4_SMC.pdf).
[^carp630]: [Carpenter, Custom 630 (17-4 PH)](<https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/Custom%20630%20(17-4%20PH).pdf>) (skeptic cross-check).
[^ak304]: [AK Steel 304/304L data sheet](https://www.spacematdb.com/spacemat/manudatasheets/304_304L_Data_Sheet.pdf).
[^ak316]: [AK Steel 316/316L data sheet](https://www.spacematdb.com/spacemat/manudatasheets/316_316L_Data_Sheet.pdf).
[^sandm304]: [Sandmeyer, Alloy 304/304L](https://www.sandmeyersteel.com/wp-content/uploads/Alloy304-304L-APR2013.pdf) (ASTM A240 minima; skeptic).
[^sandm316]: [Sandmeyer, Alloy 316/316L/317L](https://www.sandmeyersteel.com/wp-content/uploads/316-316l-317l-spec-sheet.pdf).
[^okcore]: [Outokumpu Core range datasheet](https://www.outokumpu.com/-/media/files/products/core/outokumpu-core-range-datasheet.pdf) (skeptic).
[^oksupra]: [Outokumpu Supra range datasheet](https://www.notzgroup.com/media/wysiwyg/PDF/NME/13_Outokumpu_Supra_range_Datasheet_May_2015.pdf) (skeptic).
[^kaiser6061]: [Kaiser Aluminum, 6061 sheet, coil and plate](https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1015/Kaiser_Aluminum_6061_Sheet_Coil_and_Plate.pdf).
[^kaiser7075]: [Kaiser Aluminum, 7075 rod and bar](https://online.kaiseraluminum.com/depot/PublicProductInformation/Document/1028/Kaiser_Aluminum_7075_Rod_and_Bar.pdf).
[^qq250-11]: [Aero Metals Alliance, QQ-A-250/11 6061-T6 sheet](https://www.aerometalsalliance.com/resources/data-sheets/view/Aluminium-Alloy-QQ-A-25011-T6-Sheet_200) (minima; skeptic).
[^qq250-12]: [Aero Metals Alliance, QQ-A-250/12 7075-T6 sheet](https://www.aerometalsalliance.com/resources/data-sheets/view/Aluminium-Alloy-QQ-A-25012-T6-Sheet_203) (minima; skeptic).
[^semetal7075]: [SE Metal, 7075 specification](https://semetalgroup.com/wp-content/uploads/Aluminum-7075-specification-datasheet.pdf) (skeptic).
[^wikiperm]: [Wikipedia, Permeability (electromagnetism)](https://en.wikipedia.org/wiki/Permeability_(electromagnetism)) (tier 3, flagged proxy).
[^timet]: [TIMET, TIMETAL 6-4](https://www.timet.com/assets/local/documents/datasheets/alphaandbetaalloys/6-4.pdf).
[^asmti64]: [ASM Aerospace Specification Metals, Ti-6Al-4V annealed](https://www.aerospacemetals.com/wp-content/uploads/2023/07/Titanium-Ti-6Al-4V-Grade-5-Annealed.pdf).
[^carpti64]: [Carpenter, Ti 6Al-4V](https://www.carpentertechnology.com/hubfs/7407324/Material%20Saftey%20Data%20Sheets/Ti%206Al-4V.pdf) (skeptic).
[^sm625]: [Special Metals, INCONEL alloy 625](https://www.specialmetals.com/documents/technical-bulletins/inconel/inconel-alloy-625.pdf).
[^tecapeek]: [Ensinger, TECAPEEK natural](https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=694&FL=7&FILENAME=HZ_TECAPEEK_GB_DE_201402.pdf&ZOOM=1.2).
[^victrex]: [Victrex PEEK 450G](https://www.victrex.com/-/media/downloads/datasheets/victrex_tds_450g.pdf).
[^kt820]: [Syensqo KetaSpire KT-820](https://drakeplastics.com/wp-content/uploads/2025/05/KT-820-PEEK-Syensqo-Datasheet.pdf) (skeptic).
[^ensdelrin]: [Ensinger, TECAFORM AD natural (Delrin 150)](https://www.ensinger-online.com/modules/public/sheet/createsheet.php?SID=1937&FL=14&FILENAME=DELRIN_150_Series_Nat_Acetal_Homopolymer_14.PDF&ZOOM=1.0).
[^dupont150]: [DuPont Delrin 150 NC010 datasheet](https://cdn.thomasnet.com/ccp/00072207/74788.pdf) (skeptic).
[^delringuide]: [Delrin Technical Guide](https://www.delrin.com/wp-content/uploads/2023/01/Delrin-Design-Guide-NA-FNL.pdf).
[^lamdelrin]: [Laminated Plastics, Delrin](https://laminatedplastics.com/delrin.pdf).
[^alro]: [Alro, acetals](https://www.alro.com/Resources/WebResources/AlroCom/PlasticsReferanceCatalog/PDFs/002-Acetals.pdf) (skeptic).
[^smith416]: [Smiths Metal Centres, 416 stainless](https://www.smithmetal.com/pdf/stainless/416-stainless.pdf) (skeptic).
[^rolled416]: [Rolled Alloys, 416 data sheet](https://www.rolledalloys.com/wp-content/uploads/416_stainless-steel-data-sheet-rolled-alloys.pdf) (skeptic).
[^bohler4140]: [Böhler-Uddeholm 4140-class sheet](https://www.flamehardening.com.au/wp-content/uploads/2016/05/4140.pdf) (skeptic).

Corrections:

[^all6061]: [Alliance, aluminium 6061-T6 datasheet](https://www.allianceorg.com/pdfs/alumext/6061t6.pdf) (resistivity 3.99e-6 Ω·cm, so σ = 2.506e7 S/m; cp 0.896; CTE 23.6; E 68.9 GPa).
[^ndtal]: [NDT Education, Conductivity and Resistivity Values for Aluminum & Alloys](https://content.ndtsupply.com/media/Conductivity_Al%20Reference%20Chart.pdf) (6061-T6 43.0 % IACS; only the %IACS column is cited).
