# BL-062 verify-first: magnet conductivity per grade family

Date 2026-11-09. Read-only research; nothing in the repo was changed. Every number below was read on the
page named (rendered PDF page or the page's HTML table) unless the row says otherwise.
Downloads, page renders and the hand check (`bl062/downloads/`, `bl062/renders/`, `bl062/tools/calc.py`) were kept in the research session's scratch folder and are not in the repository; the source of each number is named in section 4.

## 1. Answer in short

| Family | Library grades | Recommended sigma (S/m) | Literal | Printed basis | magnets_W in the example (now 0.2278 W) |
|---|---|---|---|---|---|
| Sintered NdFeB | N35, N42, N48, N52, N50, N42M, N50M, N42H, N42SH, N38UH, N35EH, N33AH | 7.69e5 | `7.7e5` | Arnold "150 // 130 ⊥" µΩ·cm, perpendicular value | 0.2619 W (+15 %) |
| Sm2Co17 | Recoma 26, Recoma 30 | 1.11e6 | `1.11e6` | Arnold Recoma 26 and 30: "90" µΩ·cm | 0.3775 W (+66 %) |
| SmCo5 | Recoma 20 | 1.82e6 | `1.82e6` | Arnold Recoma 20: "55" µΩ·cm | 0.6189 W (+172 %) |
| Sintered ferrite | Y30 (= C5) | 1.0e-2 | `1.0e-2` | Eclipse: "1x10^10" µΩ·cm | 3.4e-9 W (zero in effect) |
| Bonded NdFeB | Bonded_NdFeB_BCN19 | 4.0e3 | `4.0e3` | Alliance BCN-19 row: "0.025" Ohm-cm | 0.0014 W (-99.4 %) |

- The engine's 6.7e5 S/m is a real sourced number, but for the wrong direction: it is 1/(150 µΩ·cm), the value
  PARALLEL to the magnetization (Arnold) or with no direction given (Eclipse prints the pair "150" µΩ·cm and
  "0.667" x10^6 S/m). The loss model needs the PERPENDICULAR value, 130 µΩ·cm, 7.7e5 S/m.
- The backlog's guess is confirmed: ferrite falls by 8 orders of magnitude, SmCo rises (x1.66 and x2.7).
  Today's single value is therefore NOT conservative for the three SmCo grades: it understates their magnet
  eddy heat by a factor of 1.7 (2:17) and 2.7 (1:5).

## 2. Direction: which resistivity the model needs

Loss as coded (`magcoupling-rs/src/engine/temperature.rs` lines 1230-1237):
`p_mag = sigma * we^2 * b_magnet_T^2 * w^2 / 24 * (block volume) * 2 * npole`, with `w` the block's tangential
width. There is no skin-depth term on this line, so p_mag is exactly linear in sigma.

- The blocks are magnetized through the radial thickness (B842SH: 12.7 axial x 6.35 tangential x 3.17 radial mm).
- `b_magnet_T` is "Alternating radial field inside the blocks": the modelled field is along the magnetization.
- Eddy currents circle the field, so they flow in the axial-tangential face, everywhere PERPENDICULAR to the
  magnetization. The perpendicular resistivity applies. Confidence: high for the model as coded.
- Caveat: the real slipping field also has a tangential part (not modelled). Its loops would carry some current
  along the magnetization (150 instead of 130 µΩ·cm), which lowers the loss a little. Ignoring it is conservative.
- Only two sources print both directions (Arnold N-grade sheets, Vacuumschmelze). No SmCo, ferrite or bonded
  source opened prints a direction. Bonded NdFeB is isotropic (Bunting sheet: "Plastic Bonded magnets are
  isotropic"), so the question does not arise there.

## 3. Regime: is "higher sigma = more heat" safe?

Yes for this design. Example: 10 poles, 2000 rpm slip, 166.67 Hz, we = 1047.2 rad/s, mu_r 1.05.
Skin depth: NdFeB 7.7e5 S/m 43 mm; Sm2Co17 1.11e6 36 mm; SmCo5 1.82e6 28 mm (24 mm at 2.5e6); block width 6.35 mm.
The width is 0.15 to 0.26 skin depths, so the loss is resistance-limited and rises in proportion to sigma.
The trend would only turn (more sigma, less loss) near width = 2 skin depths: about 31 kHz for NdFeB and
9.6 kHz for SmCo5 at 2.5e6, that is 57 to 186 times the example's pole count x slip speed. Not reachable here.
Conservative therefore means the HIGHER sigma (the LOWER printed resistivity) in every family.

Temperature: all values are room-temperature typicals (VAC states 20 °C, Alliance ferrite "@ 20 C", EAM
"typical at room temperature"; the Arnold, K&J, Eclipse, thyssenkrupp, Allstar, Bunting and Alliance bonded rows
state none). Eclipse prints a temperature coefficient of resistivity for NdFeB of "2" x10^-4/°C: +2 % resistivity
per 100 K, so the room-temperature sigma is slightly on the hot magnet's conservative side. No coefficient was
found for the other families.

## 4. Sources, as printed

Conversions: 1 µΩ·cm = 1e-8 Ω·m; 1 Ω·mm²/m = 1e-6 Ω·m; 1 Ω·cm = 1e-2 Ω·m; sigma = 1/rho.
Tier follows `meta.resolution_rule` 3 of the A-4 data file: manufacturer grade sheet > manufacturer proxy grade >
distributor > aggregator. Rule 2 (vendor minimum) does not apply: these are typical values, not guaranteed ones.

### 4.1 Sintered NdFeB

| Vendor (tier) | URL | Locator | As printed | sigma (S/m) |
|---|---|---|---|---|
| Arnold N42SH (manufacturer grade sheet) | https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42SH-151021.pdf | p1, Rev. 020821, Other Properties, "Electrical Resistivity, ρ", unit µΩ • cm | "150 // 130 ⊥" | par 6.67e5, perp 7.69e5 |
| Arnold N42 (manufacturer grade sheet) | https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N42-151021.pdf | p1, Rev. 210607, same row | "150 // 130 ⊥" | same |
| Arnold N52 (manufacturer grade sheet) | https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N52-151021.pdf | p1, Rev. 210802, same row | "150 // 130 ⊥" | same |
| Arnold N38EH (manufacturer grade sheet, not a library grade) | https://www.arnoldmagnetics.com/wp-content/uploads/2017/11/N38EH-151021.pdf | p1, Rev. 210607, same row | "150 // 130 ⊥" | same |
| Vacuumschmelze VACODYM (manufacturer, family table; copy hosted by CERN, edition 2015) | https://indico.cern.ch/event/682107/attachments/1560598/2465492/VACODYM-VACOMAX-PD002_2015_en.pdf | p18, Table 3 "at room temperature (20 °C)", column "Specific electrical resistance", Ωmm²/m | "1.4-1.6 (II c)" and "1.2-1.4 (⊥ c)" | par 6.25e5 to 7.14e5, perp 7.14e5 to 8.33e5 |
| K&J Magnetics (distributor; the engine's NdFeB source) | https://www.kjmagnetics.com/neodymium-magnet-specifications.asp | table "Neodymium Magnet Physical and Mechanical Specifications & Tolerances", row "Electrical Resistivity (ρ)" | "160 µ-ohm-cm" (no direction) | 6.25e5 |
| Eclipse Magnetics (distributor) | https://www.eclipsemagnetics.com/site/assets/files/19485/ndfeb_neodymium_iron_boron-standard_ndfeb_range_datasheet_rev1.pdf | p3, Physical Characteristics (Typical) | "Electrical Resistivity ρ µ Ω.cm 150"; "Temperature coefficient of resistivity α 10^-4/°C 2"; "Electrical Conductivity σ 10^6 S/m 0.667" (no direction) | 6.67e5 |

Reading: perpendicular 130 µΩ·cm (Arnold, four sheets) sits in the middle of VAC's perpendicular range 120 to 140.
Range of printed values: 6.25e5 (K&J, no direction) to 8.33e5 S/m (VAC perpendicular, low-resistivity end).
The Arnold sheets for N35, N42UH, N45SH and N48H were downloaded and carry the same row label, but their value
was not read on a rendered page (the text layer drops it): not counted.

### 4.2 Sm2Co17 (Recoma 26, Recoma 30)

| Vendor (tier) | URL | Locator | As printed | sigma (S/m) |
|---|---|---|---|---|
| Arnold Recoma 26 (manufacturer grade sheet; the engine's source) | https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf | p8, sheet Rev. 131025, "Electrical Resistivity, ρ", µΩ • cm | "90" | 1.11e6 |
| Arnold Recoma 30 (manufacturer grade sheet; the engine's source) | same file | p12, same row | "90" | 1.11e6 |
| Vacuumschmelze VACOMAX Sm2Co17 (manufacturer, family table, CERN copy) | https://indico.cern.ch/event/682107/attachments/1560598/2465492/VACODYM-VACOMAX-PD002_2015_en.pdf | p18, Table 3 (20 °C) | "0.65-0.95" Ωmm²/m | 1.05e6 to 1.54e6 |
| thyssenkrupp Magnettechnik (distributor) | https://ucpcdn.thyssenkrupp.com/_binary/UCPthyssenkruppBAMXSchulteMicrositeMagnettechnik/en/downloads/link-thyssenkruppMagnettechnik-Factsheet-samarium-cobalt-magnets.pdf | p3, Physical properties, Sm2Co17 180/120, electrical resistivity Ω mm²/m | "0.85" | 1.18e6 |
| Eclipse Magnetics (distributor) | https://www.eclipsemagnetics.com/site/assets/files/19544/eclipsemagnetics_smco_datasheet.pdf | p1, Physical Characteristics (excluding bonded variants) | "Electrical Resistivity Sm2Co17 ρ µ Ω.cm 80-90"; "Electrical Conductivity Sm2Co17 σ 10^6 S/m 1.11-1.25" | 1.11e6 to 1.25e6 |
| Allstar Magnetics (distributor, generic SmCo) | https://allstarmagnetics.com/wp-content/uploads/2024/09/Samarium-Cobalt-Technical-Data-ASM-09-23-2023.pdf | p2 | "Resistivity 0.8 x 10-6 Ωm" | 1.25e6 |

### 4.3 SmCo5 (Recoma 20)

| Vendor (tier) | URL | Locator | As printed | sigma (S/m) |
|---|---|---|---|---|
| Arnold Recoma 20 (manufacturer grade sheet; the engine's source) | https://www.arnoldmagnetics.com/wp-content/uploads/2017/10/Recoma-Combined-160301.pdf | p4, sheet Rev. 131025, "Electrical Resistivity, ρ", µΩ • cm | "55" | 1.82e6 |
| Vacuumschmelze VACOMAX SmCo5 (manufacturer, family table, CERN copy) | https://indico.cern.ch/event/682107/attachments/1560598/2465492/VACODYM-VACOMAX-PD002_2015_en.pdf | p18, Table 3 (20 °C) | "0.4-0.7" Ωmm²/m | 1.43e6 to 2.50e6 |
| thyssenkrupp Magnettechnik (distributor) | (SmCo factsheet above) | p3, Sm1Co5 150/120 | "0.55" Ω mm²/m | 1.82e6 |
| Eclipse Magnetics (distributor) NOT USED | (SmCo sheet above) | p1 | "Electrical Resistivity Sm1Co5 ρ µ Ω.cm 5-6"; "Electrical Conductivity Sm1Co5 σ 10^6 S/m 16.6-20" | 1.66e7 to 2.0e7 as printed |

The Eclipse SmCo5 pair is self-consistent but ten times off the three other sources (5-6 against 40 to 70 µΩ·cm);
it reads like a slipped decimal (50-60 would give 1.67e6 to 2.0e6). Recorded as printed, not used.

### 4.4 Sintered ferrite (Y30 = C5)

| Vendor (tier) | URL | Locator | As printed | sigma (S/m) |
|---|---|---|---|---|
| Eclipse Magnetics (distributor; the engine's main Y30 source) | https://www.eclipsemagnetics.com/site/assets/files/19602/ferrite_ceramic_datasheet.pdf | p3, Physical Characteristics (Typical) | "Electrical Resistivity ρ µ Ω.cm 1x10^10" | 1.0e-2 |
| Alliance LLC ceramic chart (distributor; the engine's C5 source vendor) | https://allianceorg.com/magnetic-materials/ceramic-magnets/ | second table, row C5, column "Electrical Resistivity Ω-cm @ 20 C" | "100,000" | 1.0e-3 |
| MS-Schramberg HF 26/24 (manufacturer proxy grade) | https://magnete.de/fileadmin/user_upload/Magnetische_Kenndaten/Gesinterte_Hartferritmagnete/HF_26-24_Sr_E.pdf | p1, Mechanical values (08/2022), "spec. elec. resistance approx. 10^-6 Ωm" | ">10^4" | below 1.0e2 (a bound, not a value) |
| thyssenkrupp Magnettechnik (distributor) | https://ucpcdn.thyssenkrupp.com/_binary/UCPthyssenkruppBAMXSchulteMicrositeMagnettechnik/en/downloads/link-thyssenkruppMagnettechnik-Factsheet-ferrite-magnets.pdf | p3, electrical resistivity (Ω mm²/m) | "∞" | 0 |
| Alliance C-5 grade page | https://allianceorg.com/magnetic-materials/ceramic-magnets/ferrite-c-5/ | whole page | no resistivity row | n/a |

Order of magnitude: resistivity 1e2 to 1e3 Ω·m, sigma 1e-3 to 1e-2 S/m, about 8 to 9 decades below NdFeB. The one
manufacturer sheet gives only a ceiling (sigma below 100 S/m). At 1e-2 S/m the loss is 3.4e-9 W; even at the
100 S/m ceiling it is 3.4e-5 W (0.015 % of the NdFeB figure). Both print as 0.0000 W.

### 4.5 Bonded NdFeB (BCN-19)

| Vendor (tier) | URL | Locator | As printed | sigma (S/m) |
|---|---|---|---|---|
| Alliance LLC, BCN-19 itself (grade vendor page; the engine's source) | https://allianceorg.com/magnetic-materials/bonded-magnets/compression-bonded-neo/ | Mechanical Properties table, row BCN-19 (5.6-6.0 / 80-120 / 27-33 / 28,000 / 0.025), column "Electrical Resistivity", unit "Ohm-cm" | "0.025" (same on every BCN row) | 4.0e3 |
| EAM compression-bonded Neo (manufacturer proxy grade, density 6.0-6.1 g/cm³) | https://eamagnetics.com/wp-content/uploads/2021/08/EAM-ComprBond-Neo-PhysProp-1.pdf | p1, Physical Properties, all five grades, "typical at room temperature" | "Electrical Resistivity 14000 µohm-cm" | 7.14e3 |
| Bunting / E-Magnets UK BREMAG (distributor; compression and injection moulded, density 6.0 g/cc) | https://e-magnetsuk.com/wp-content/uploads/2020/06/E-Magnets-UK-Bonded-Neodymium-Data-Sheet.pdf | p2, Issue 1 January 2017, Typical Value table, row "Resistivity", unit Ωcm | "4 x 10^-3" | 2.5e4 |
| MCE BN0966 (distributor) | https://mceproducts.com/materials/neodymium-iron-boron-bonded/bn0966/ | row "Electrical Resistivity (μΩ.cm)" | "N/A" | n/a |

No proxy is needed: the Alliance page that already supplies BCN-19's magnetic data prints its resistivity. The
nearest proxy, were one wanted, is EAM compression-bonded Neo (same process, density 6.0-6.1 against 5.6-6.0).
The three printed values span a factor of 6 (binder and loading); every one is more than 26 times below 6.7e5.

### 4.6 Opened without a usable value, or not opened

- Arnold APEEM 2006 slides (the engine's mu_rec source), 54 pages: text layer searched, no resistivity number
  found; the slide images were not each viewed.
- Vacuumschmelze current "Product Information" brochure (vacuumschmelze.com, 36 pages): opened, text searched, no
  resistivity table. The table used above is from the 2015 edition as hosted by CERN, not from VAC's own site.
- Maruwa ferrite page (https://www.maruwa-g.com/e/products/ceramic/000321.html): opened, no resistivity row.
- NOT OPENED: thyssenkrupp NdFeB factsheet (HTTP 403 on every URL tried), TDK FB ferrite catalogue (HTTP 403),
  Magnequench and Proterial sheets (not tried).

## 5. Options for the owner (recommended first)

Effect figures are for the example design, all else equal (same blocks, B 0.19 T, 2000 rpm slip).

1. Sintered NdFeB
   - 1A (recommended) `7.7e5`: Arnold perpendicular, the direction the model needs (rule 1), manufacturer grade
     sheet (rule 3), inside VAC's range. magnets_W 0.2278 to 0.2619 W. Cost: it moves the default example and the
     workbook value at Temperature design!C113, so it needs its own registered deviation.
   - 1B keep `6.7e5`: no parity change. Understates the perpendicular case by 13 %; the thin-strip formula's own
     margin (audit row M12: engine +46 % over the rectangular-section solution) still covers that.
   - 1C `8.3e5`: VAC's low-resistivity end, the most conservative printed value (+24 %).
2. Sm2Co17
   - 2A (recommended) `1.11e6`: Arnold Recoma 26 and 30 grade sheets, the engine's own source (rule 3). 0.3775 W.
   - 2B `1.25e6`: the 80 µΩ·cm end (Eclipse, Allstar; distributors). 0.4251 W.
   - 2C `1.54e6`: VAC's low end for its own 2:17 alloys. 0.5232 W.
3. SmCo5
   - 3A (recommended) `1.82e6`: Arnold Recoma 20 grade sheet; thyssenkrupp agrees. 0.6189 W.
   - 3B `2.5e6`: VAC's low end. 0.8502 W. Note it lies above the input's present upper limit of 2e6.
4. Ferrite
   - 4A (recommended) carry `1.0e-2`: the Eclipse sheet the grade already rests on, and the higher of the two
     printed values (rule 4). 3.4e-9 W. No special case in the code: the formula is linear in sigma and never
     divides by it.
   - 4B carry `1.0e2`: MS-Schramberg's ceiling used as if it were the value. 3.4e-5 W. Most conservative, but it
     is a bound, not a measurement.
   - 4C a hard zero for the family: same displayed result, one more branch to test.
5. Bonded NdFeB
   - 5A (recommended) `4.0e3`: BCN-19's own row on the Alliance page (rule 3). 0.0014 W.
   - 5B `2.5e4`: Bunting BREMAG, the most conservative printed value. 0.0085 W.
   - 5C `7.1e3`: EAM proxy. 0.0024 W.

## 6. Notes for whoever writes the fix

- `sigma_ndfeb_S_m` is a user input with range 1e5 to 2e6 (log). The ferrite and bonded values, and option 3B,
  lie outside it, so the per-family number has to live in the grade data (as the backlog entry already says)
  and must not pass through that input's range check.
- The acceptance test's two directions are both supported: ferrite falls far below NdFeB, SmCo rises.
- The help text at `magcoupling-rs/src/gui/plain_text.rs` about line 402 says the single value "overstates the
  heat in ferrite". It also UNDERSTATES the heat in SmCo, which is the non-conservative side.
- Audit row M12 says a 0.10 W change in magnets_W moves the total slip loss by at most 4 %, so the example's
  total is roughly 2.5 W or more; SmCo5 would add about 0.39 W to it. That is an inference from M12, not an
  engine run (cargo was not run: the repo was read-only).

## Skeptic check

Date 2026-11-09. Independent pass, aim: refute. Nothing above this heading was edited. Every source was
downloaded again into its own folder (`bl062/skeptic-downloads/<name>/`) and read on a rendered page or in the
page HTML (`bl062/skeptic-renders/`); the researcher's downloads and renders were not reused. The engine was read,
not run.

### S1. Result in short

| Family | Researcher's value (S/m) | Verdict | Why |
|---|---|---|---|
| Sintered NdFeB | 7.7e5 | agreed | "150 // 130 perp" micro-ohm-cm seen on all four Arnold sheets; VAC p18 perpendicular range 1.2-1.4 seen. |
| Sm2Co17 | 1.11e6 | agreed, with a caveat | "90" seen on Recoma 26 (p8) and 30 (p12). It is the LOWEST sigma any source prints for 2:17, bar one range end. |
| SmCo5 | 1.82e6 | agreed | "55" seen on Recoma 20 (p4); thyssenkrupp "0.55" seen; independent sources give 50-60 and 53. |
| Sintered ferrite | 1.0e-2 | agreed (order of magnitude only) | "1x10^10" micro-ohm-cm seen. Printed values now span 1e2 to 1e4 ohm-m; all give zero heat. |
| Bonded NdFeB | 4.0e3 | DISPUTED (the choice, not the transcription) | "0.025" Ohm-cm seen, but it is the least conservative of the printed values, which span a factor of 25. |

- Transcription: every cited number that could be opened is on the page at the stated place, with the stated
  unit and direction. No slip found in 21 sources. Every rho-to-sigma conversion and every effect figure in
  sections 1 and 5 was recomputed and holds (baseline 0.2278 W reproduced by hand from the coded formula).
- Not openable by me either: thyssenkrupp NdFeB factsheet (HTTP 403 by curl and by WebFetch), TDK FB catalogue
  (HTTP 403). The researcher said the same and did not rely on them.

### S2. Direction and regime: holds

- Code read: `temperature.rs` 1230-1237 is linear in sigma with no skin term, as stated. `model.rs` has the inner
  block as length axial, width tangential, thickness radial. `reference/magcoupling-py/magcoupling/fields3d.py`
  builds each block with its polarization along the thickness (radial) and takes `b_mag` as the field projected
  on that same direction. So the modelled field lies along the magnetization, the eddy loops lie in the
  axial-tangential face, and the current runs perpendicular to the magnetization everywhere. The perpendicular
  resistivity is the right one. 6.7e5 is the parallel (or direction-free) value. Agreed.
- Regime: skin depth 43.4 mm at 7.7e5, mu_r 1.05, 166.67 Hz (recomputed); width / skin depth 0.146. The classical
  slab loss turns near width = 2.25 skin depths, so "about 2" and the 31 kHz and 9.6 kHz figures are right.
  Higher sigma is the conservative side in every family, at every slip speed this calculator can reach.
- Where p_mag goes: into p_tot, then temperature rise and time to limit. It does not touch the critical drag.
  If a measured drag is entered, p_mag is not used at all.
- Published support, read at abstract level only: Ruoho et al., IEEE Trans. Magn. 46(1) 2010, "Temperature
  Dependence of Resistivity of Sintered Rare-Earth Permanent-Magnet Materials", says the anisotropy makes a
  larger resistivity difference than temperature does from -40 to 150 deg C. The numbers in that paper were NOT
  seen (the core.ac.uk copy of the lead author's dissertation gives HTTP 403; the lib.tkk.fi copy fails its
  certificate check and was not forced).

### S3. Independent sources (none of these is cited above)

| Source | Locator | As printed | sigma (S/m) | Against the researcher |
|---|---|---|---|---|
| MMPA Standard 0100-00 (trade standard; copy at https://www.duramag.com/wp-content/uploads/MMPA_0100-00.pdf) | pdf p20, Table IV-3, "Electrical resistivity Ohm-cm x 10^-6 (at 20 deg C)" | 1-5 Alloys "53"; 2-17 Alloys "86"; Nd-Fe-B "160" (no direction) | 1.89e6; 1.16e6; 6.25e5 | SmCo5 +3.7 % and Sm2Co17 +4.7 % above the recommended values. NdFeB matches the direction-free 150 to 160 group. |
| MMPA Standard 0100-00 | pdf p15, Table III-4, ceramic magnets, "Electrical resistivity" | "10^6 ohm-cm" | 1.0e-4 | Ferrite: 100 times below the recommended 1.0e-2. Immaterial. |
| Dexter Magnetic Technologies Sm-Co table (https://www.dextermag.com/wp-content/uploads/2016/10/Dexter_Sm-Co_Curve__Properties_Table.pdf) | pdf p2, V101, 2010 | Type 2:17 "80 - 90" micro-ohm-cm; Type 1:5 "50 - 60" micro-ohm-cm | 1.11e6 to 1.25e6; 1.67e6 to 2.0e6 | Agrees with both. Also supports reading the Eclipse "5-6" as a lost digit: same row layout, 50 - 60 here. |
| Dexter Nd-Fe-B table (https://www.dextermag.com/wp-content/uploads/2016/10/Dexter_Nd-Fe-B__Curves_Properties_Table.pdf) | pdf p3 | "120 - 160" micro-ohm-cm (no direction) | 6.25e5 to 8.33e5 | Brackets 130 and 150; the same span as VAC perpendicular plus parallel. |
| Advanced Magnets (https://www.advancedmagnets.com/?p=33) | HTML table, row "Resistivity" | "1.4-1.6" micro-ohm-m (no direction) | 6.25e5 to 7.14e5 | The parallel band. Does not speak to the perpendicular value. |
| Alliance LLC, "Physical Properties of Permanent Magnet Materials" (https://www.allianceorg.com/pdfs/PhysicalPropertiesofMagnets.pdf; same vendor as BCN-19, a different document) | 1 page, row "Electrical Resistivity", unit micro-ohm-m | SmCo 1:5 "0.5 - 0.6"; SmCo 2:17 "0.75 - 0.85"; sintered NdFeB "1.2 - 1.6"; bonded (compression) NdFeB "10 - 30"; bonded (injection) NdFeB "40 - 70"; sintered ferrite blank | 1.67e6 to 2.0e6; 1.18e6 to 1.33e6; 6.25e5 to 8.33e5; 3.3e4 to 1.0e5; 1.4e4 to 2.5e4 | Sintered families agree. Compression-bonded is 8 to 25 times MORE conductive than on the BCN page of the same vendor. |
| FEMM material notes (https://femm.info/wiki/BondedNeoProperties; aggregator tier) | page text | "A typical value of electrical conductivity is 0.01 MS/m." | 1.0e4 | 2.5 times the recommended bonded value. |
| Brooks et al., J. Magn. Magn. Mater. 610 (2024) 172526; abstract on the University of Birmingham page, read through WebFetch only | abstract | 2:17 types have "lower resistivity in the plane perpendicular to the c-axis"; the 1:5 type has lower resistivity "in the parallel plane" | none printed | Direction matters for SmCo too. For 2:17 the direction this model needs is the LOWER resistivity. |

### S4. Points the owner should weigh

1. Bonded NdFeB (disputed). The 0.025 Ohm-cm is printed identically on every BCN row, so it is a family figure,
   not a BCN-19 measurement, and Alliance is a supplier, not the maker: rule 3 does not lift it clearly above the
   others. Printed values, low to high sigma: 4.0e3 (Alliance BCN page), 7.1e3 (EAM), 1.0e4 (FEMM typical),
   2.5e4 (Bunting), 3.3e4 to 1.0e5 (Alliance overview table, compression). The recommended value is the lowest,
   which is the NON-conservative end. The "factor of 6" in section 4.5 becomes a factor of 25, and option 5B is
   no longer "the most conservative printed value".
   Caution on the Alliance overview table: its bonded-ferrite cell in the same row reads "0.01" micro-ohm-m,
   which is impossible, so that row is not reliable enough to set a value. It is enough to deny confidence in
   4.0e3. Rule 4 (conservative option) points to 5B `2.5e4`, the highest value printed with a clear unit.
   Effect, all else equal: 0.0014 W (5A), 0.0085 W (5B), 0.034 W at 1.0e5. All three are far below the NdFeB
   figure, so the direction the acceptance test needs holds for any of them.
2. Sm2Co17 (agreed, caveat). 90 micro-ohm-cm is the grade sheet value and rule 3 picks it. But it gives the
   lowest sigma of all the single printed values (MMPA 86, thyssenkrupp 85, Allstar 80; ranges: Dexter and
   Eclipse 80-90, Alliance 75-85, VAC 65-95). This is the family where the present engine value is NOT
   conservative, no source states a direction, and the one published abstract says the perpendicular resistivity
   is the lower one for 2:17. Option 2B `1.25e6` (+13 % on 2A) would cover that; it is a judgment for the owner.
3. The effect figures for SmCo, ferrite and bonded are "all else equal": they hold `b_magnet_T` at 0.19 T, which
   is a stored 3D result for the default NdFeB ring. A ring of lower remanence carries a lower field and the
   loss goes with its square, so +66 % and +172 % are upper-side figures for a like-for-like grade swap, and the
   bonded and ferrite figures are overstated further. The sigma ratios themselves (x1.66, x2.7) are unaffected.
4. Sintered NdFeB. The perpendicular 130 rests on two makers (Arnold on four sheets, VAC as a range). No
   independent source prints a direction; they bracket it (120 to 160). The Arnold rows state no temperature.
5. Ferrite. "Resistivity 1e2 to 1e3 ohm-m" in section 4.4 should read 1e2 to 1e4 with the MMPA value. 1.0e-2 stays
   the highest printed sigma, so it is still the conservative carry. The only maker sheet gives a bound.
6. The Eclipse temperature coefficient of resistivity (2e-4 per deg C) is a distributor figure, for NdFeB only.

### S5. Not checked

- Arnold APEEM 2006 slide images (text layer only, as the researcher); the MS-Schramberg sheet date "08/2022".
- The measured anisotropy numbers in Ruoho 2010 and Brooks 2024 (abstracts only).
- No engine run and no cargo test: the repo was read-only for this pass.
