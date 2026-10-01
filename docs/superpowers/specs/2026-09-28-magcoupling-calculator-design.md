# Magnetic Coupling Calculator — Design

**Date:** 2026-09-28
**Status:** Approved by user in interactive brainstorm (all four design sections)
**Scope:** new crate `magcoupling-rs/`, vendored Python reference `reference/magcoupling-py/`,
host integration in `linkage-sim-rs/`, web deploy (`.github/workflows/deploy-web.yml`,
`linkage-sim-rs/scripts/build_web.sh`, `linkage-sim-rs/web/`)

## Purpose

Bring the rail robot's **magnetic slip coupling** design calculator into this
tool. The coupling is a coaxial synchronous permanent-magnet coupling between a
5:1 traction gearbox (0.6 N·m input rating, 3,000 rpm limit) and the wheel. It
carries driving torque and slips without contact when the rail overdrives the
wheel or friction spikes the torque at touchdown.

The source is a Python package (`magcoupling` 1.0.0, uploaded 2026-09-28),
itself a port of `magnetic_coupling_torque_calculator.xlsx`: ~2,400 lines of
pure-Python engine, a workbook snapshot (`tests/reference_values.json`, a
`"Sheet!Cell" → value` map), and a parity suite. Verified on 2026-09-28:
`pytest` → 1,158 passed, 1 skipped; all 935 workbook formula cells match to 1e-9.
The `.xlsx` itself was not provided; the snapshot is the workbook's record.

**What the user asked for, in order:**

1. **The math must be right** — first and above all. Agreed meaning: the Rust
   port must reproduce the workbook exactly, **and** the workbook's equations
   must be independently verified against physics.
2. **A GUI with nice sliders** controlling the calculator inputs and geometry.
3. **Standalone and integrated:** usable on its own and inside the linkage tool.

**Decisions from the brainstorm:**

| Question | Decision |
|---|---|
| Meaning of "math is right" | Port fidelity **and** independent physics verification |
| Where it lives | Separate Rust crate; standalone egui app (native + web) and a window in the linkage app |
| 3D field model | Port to Rust, run live so temperature limits follow geometry |
| When physics check finds a workbook error | Correct it as a **documented deviation**; everything else stays workbook-exact |

**Default-design facts the tool must surface (from the Python run):** pull-out
2.647 N·m at 50 °C; hot low with 15 % allowance 2.25 N·m vs 2.5 N·m requirement
("Below hot minimum"); running clearance −0.103 mm ("Below target"); cup wall
"Too thin: raise … to at least 2.0 mm"; governing temperature limit 92.55 °C;
recommended clamp screw ISO 4762 M4 × 12, class 12.9.

## Structure

- **`magcoupling-rs/`** — new crate, sibling of `linkage-sim-rs/`:
  - **engine** (`src/engine/`): pure functions and data, no GUI dependencies,
    one Rust module per Python module (`model`, `calibration`, `metal_design`,
    `materials`, `temperature`, `clamps`, `sweeps`, `library`, `fields3d`,
    `api`);
  - **panel** (`src/gui/`, behind feature `gui`): the egui panel, hostable by
    any egui app;
  - **binary** `magcoupling-app` (feature `app`): standalone native + WASM entry.
- **`linkage-sim-rs`** depends on it by path with feature `gui` and opens the
  panel from a **Tools** menu.
- **No Cargo workspace conversion.** A workspace would move the target directory
  and break `deploy-web.yml` and `build_web.sh` for no functional gain.
- **`reference/magcoupling-py/`** — the Python package vendored unchanged
  (engine, tests, snapshot, tools). It is the **oracle** for parity and
  differential tests and the workbench for M1. Its own `pytest` stays green.
- **Gate:** `linkage-sim-rs/scripts/gate.sh` is extended to run
  `cargo test`, `cargo clippy --all-targets`, and the WASM check for
  `magcoupling-rs`, plus the vendored Python parity suite.
- **Deploy:** the standalone page ships at
  `linkage.colesorkness.com/magcoupling/` as a separate, smaller WASM bundle
  built by the same deploy job (`web/magcoupling/`).

## Phases

Each phase ends with a testable deliverable. **User review gates** follow M1
and M4.

| Phase | Deliverable | Gate |
|---|---|---|
| **M1** Math verification | Findings report in `docs/analyses/` | **User approves which corrections to apply** |
| **M2** Engine port | Rust engine, workbook parity + differential tests + deviation registry | Tests + review |
| **M3** 3D field port | Rust block-field solver + geometry builder feeding demag limits | Tests vs magpylib + review |
| **M4** Standalone GUI | Sliders, geometry view, dashboard, plots at `/magcoupling/` | **User hands-on checklist** |
| **M5** Embed | Tools-menu window in the linkage app, shared build and deploy | Tests + smoke + user check |

**Interleave with the payload-weights feature**
(`docs/superpowers/specs/2026-09-28-payload-weights-gravity-assist-design.md`):
payload Track 1 fixes → magcoupling M1 → payload Track 2 alongside M2–M5. The
features touch disjoint code; each merges through its own review.

## M1 — Math verification

**Scope — every formula family:**

- **Torque model** (`model.py`, `calibration.py`): harmonic amplitudes
  `B_n = Br·4/(nπ)·sin(n·fill·π/2)` and fill factor; back-iron factor
  `sinh(k t_i) sinh(k t_o)/sinh(k(t_i+t_o+g))` and free-space factor
  `(1−e^{−k t_i})(1−e^{−k t_o}) e^{−k g}/2`; wave number `k = n·(p/2)/R_gap`;
  shear stress `τ_n = B_in B_on/(2µ0)·S_n·sin(nπ/2)`; torque
  `Σ τ_n · 2π R_gap² L`; end-effect factor `1 − c_end·pitch/L`; calibration
  factor selection and the one-point prototype correction; temperature scaling
  `Br(T)` linear at −0.12 %/°C with torque ∝ Br²; block/polygon geometry
  (corner gap and radii from the flat-face gap); mass and inertia.
- **Metal design and materials** (`metal_design.py`, `materials.py`): hot/cold
  torque with variation, radial clearance stack, retainer geometry and masses,
  axial envelope, slip duty, back-iron thickness at 1.5 T, cup-wall check,
  plating offsets.
- **Temperature** (`temperature.py`): demagnetization knee
  `Hk(T) = 0.9·Hcj20·(1 − 0.5 %/°C·(T−20))` and onsets with field scaling by
  `Br(T)` and the permeance-coefficient calibration; adhesive limits and bond
  loads; Volkersen thermal-mismatch shear; slip losses (travelling field on
  permeable conductor ∝ speed^1.5, thin-shell ∝ speed², thin-strip magnets);
  thermal network (heat capacity, conductance, time constant, per-event rise,
  steady rise, time to limit, critical drag torque); life totals and life checks.
- **Clamps** (`clamps.py`): per-size geometry, preload at 75 % proof capped by
  aluminium thread stripping, capacity `µ·F·d·factor`, screw fit, recommendation,
  key backup, adapter joint.
- **Sweeps and constants** (`sweeps.py`, `constants.py`; e.g. rounded
  `MU0 = 1.256637e-6`).

**Methods — each formula gets at least one; the torque model gets all four:**

1. **Re-derivation** from first principles, compared term by term.
2. **Independent numerical reference.** Torque: a separate 2D field solution
   written from scratch, plus real magpylib 3D, sweeping torque vs relative
   rotation angle. Other modules: an independent re-implementation in test code.
3. **Limits and scaling laws** (e.g. torque ∝ L and ∝ Br², → 0 as gap → ∞,
   correct thick-magnet limits, symmetry), plus dimensional consistency of every
   formula.
4. **Literature cross-check:** standard coaxial PM coupling results, eddy-current
   loss in travelling fields, bolted-joint preload practice, Volkersen shear lag,
   NdFeB knee behaviour.

**Known candidate to examine (illustrative):** every harmonic is evaluated at the
fundamental's pull-out angle (quarter pole pitch), hence the `sin(nπ/2)` sign
(third harmonic negative). The true pull-out is the peak of the combined
torque-angle curve, which can sit at a different angle.

**Adversarial verification:** each suspected problem goes to independent skeptics
prompted to refute it (three lenses for physics claims, as in
`.claude/workflows/audit-campaign.js`). Only surviving findings are reported.

**Output — `docs/analyses/<date>-magcoupling-math-audit.md`** (dated the day M1
completes), one entry per
finding: workbook cells and formula; the issue with derivation and numbers;
effect at the default design (N·m, °C, …) and whether any verdict changes; the
proposed correction. Three classes:

- **Error** — wrong physics, sign, unit, or transcription → proposed correction.
- **Model approximation** — e.g. empirical end-effect factor, no saturation or
  eddy solution → documented, not "corrected".
- **Placeholder input** — e.g. `conductance_W_K = 0.3`,
  `driving_rise_C = 10` → flagged for measurement.

The report also lists every formula **confirmed correct** and the method used,
so coverage is explicit. **Only corrections the user approves enter M2.**

## M2 — Engine port

- **Module-by-module port** of the engine, including the magnet, adhesive, and
  screw libraries as static data.
- **Inputs** are Rust structs; every field carries path, label, unit, help,
  workbook cell, default, kind, and choices (selectors), **plus a slider range**
  (min, max, step, logarithmic flag) defined by hand from physical bounds — the
  Python has no ranges.
- **One pure entry point** `compute_all(&DesignInputs) -> Results`: no I/O, no
  global state, milliseconds per call so the GUI recomputes every frame during a
  drag. Text verdicts are reproduced character for character (they drive status
  badges). Selector integer codes match the workbook (`backiron`: 1 steel, 0 none,
  etc.). Units as the package: mm, N·m, °C, T, kA/m, MPa, W, J, rpm, g.
- **Result schema** mirrors `result_schema()`: every value with label, unit, cell.

**Testing — three layers:**

1. **Workbook parity.** The snapshot is copied into `magcoupling-rs/tests/data/`.
   Every result field with a cell reference, every sweep cell (494), every
   screw-table cell (165) and every default input (160) must match: numbers to
   1e-9 relative (1e-12 absolute), text exactly — the same rule as
   `test_parity.py`.
2. **Differential testing against Python.** The snapshot covers only default
   inputs. A script in `reference/magcoupling-py/tools/` runs the Python engine on
   a few hundred seeded random input sets spanning each slider range and every
   selector branch, writing inputs and outputs to JSON; the Rust port must match
   on all of them.
3. **Deviation registry.** Approved corrections live in a static registry: cells,
   workbook value at defaults, corrected formula, evidence link (the M1 report
   entry). A **test-only** switch disables all deviations so layers 1–2 still
   compare exactly against the workbook and Python; with deviations on, only
   registered fields may differ, and a test asserts each registered field takes
   its corrected value. The switch is not exposed to users.

## M3 — 3D field port

- **Solver:** closed-form B/H field of a uniformly magnetized cuboid (the formula
  magpylib uses), with explicit handling of observers on edges and faces.
- **Geometry builder** mirroring `fields3d._Geom`: rings of blocks on polygon
  faces, phase offsets, the four demagnetization cases (aligned, pull-out, like
  poles facing, single ring on its carrier), first-order steel images, the
  per-block evaluation grid, and the fundamental-Br disk integration.
- **Outputs** replace the stored 3D inputs of the temperature design live, via
  the equivalent of `fields3d.apply_to_inputs`.
- **Validation:** point fields vs real magpylib at a few thousand observer points
  including near edges (1e-9 relative away from singular sets); full run vs the
  Python `fields3d.run` outputs (which match the workbook's 3D inputs within 1 %).
- **Scheduling:** estimated a few hundred ms native, longer on WASM. Runs on
  slider **release**, not per frame: background thread natively; time-sliced
  across frames on the web (no threads). Until complete, temperature limits carry
  a **"3D updating"** badge instead of presenting stale numbers as current.

## M4 — Standalone GUI

**Layout:**

- **Left — inputs**, generated from metadata, grouped as the package groups them
  (`coupling`, `metal`, `calibration`, `materials`, `temperature`, `clamps`).
  A **Key design** group on top: face gap, pole count, magnet part, axial length,
  operating temperature, back iron, cup wall, conductance, measured drag (about a
  dozen).
- **Center — geometry view**, to scale: end view of both rings with blocks on
  polygon faces, cup, sleeve, liner, shaft and key; side view of the axial stack
  against the 20 mm bay and 35 mm overall length. Redraws live during drags.
  Dimension callouts for face gap, corner gap, running clearance; violations
  (negative clearance, envelope exceeded) draw red.
- **Right — dashboard**: headline numbers with green/amber/red badges derived
  from check verdicts; corrected values carry a "corrected vs workbook" marker
  with the deviation in its tooltip.
- **Bottom — plot tabs**: torque vs temperature (hot/cold allowance band,
  requirement line); gap sweep; pole sweep; slip heating over time vs limit;
  torque vs rotation angle (pull-out point); clamp table plus the clamp drawing
  (end and top views, egui painter port of `drawing.py`).
- **Results table**: every computed value with label, unit, cell; searchable;
  CSV and JSON export.

**Sliders:** live recompute while dragging; value box for typed entry; arrow-key
nudges; step snapping (e.g. even-only pole counts); logarithmic scale for wide
ranges; per-field reset-to-default; a dot when changed from default; tooltip with
help text and workbook cell; selectors as drop-downs. 3D recompute on release
(M3).

**Session:** undo/redo of input changes; reset all; save/load design JSON; share
link (as the linkage tool's `?m=`). No unit switching in v1. Theme matches the
linkage app.

## M5 — Embed in the linkage tool

**Tools → Magnetic coupling** opens the same panel in an `egui::Window`, with
state independent of the linkage model. The standalone page stays at
`/magcoupling/`. `build_web.sh` and `deploy-web.yml` build and ship both bundles.

## Testing and validation summary

- M1: adversarially verified findings report; coverage list of confirmed formulas.
- M2: workbook parity (1e-9 / exact text), differential vs Python on seeded random
  inputs, deviation-registry assertions.
- M3: point fields vs magpylib; full 3D run vs Python.
- M4/M5 headless egui tests: slider change updates results; reset restores the
  default; selector switches branch; undo reverses a change; 3D badge lifecycle.
- `gui-smoke` extended to `/magcoupling/`; `scripts/gate.sh` green on every
  change; fresh-context review per change; physics changes additionally get a
  dedicated physics reviewer agent (the `fbd-math-reviewer` is linkage-specific,
  so the magcoupling review prompt names the relevant M1 derivations).
- User gates: M1 findings approval; M4 hands-on checklist.

## Out of scope (v1)

- Unit switching in the magcoupling GUI.
- Editing the `.xlsx` (not provided); corrections are carried as documented
  deviations to transfer back by hand.
- More harmonics than the workbook's (1, 3, 5) as a default — the engine keeps
  the parameter, parity tests keep the default.
- A finite-element solver; saturation and eddy-current field solutions remain
  model approximations (documented in M1).
- Any coupling between the magcoupling panel and the linkage model's data.

## Addendum A — GUI v1 additions (approved 2026-09-29)

Requested by the user after the spec was approved; design approved in chat the
same day. These extend M2 (engine) and M4/M5 (GUI). None affects M1. At
default inputs and default assumptions every result stays workbook-exact, so
the M2 parity and differential tests are unchanged.

### A1. Self-sizing geometry, two directions

- **Mode switch** in the Key design group: **Magnets → Torque** (today's
  forward calculation) and **Torque → Magnets** (inverse sizing).
- **Inverse sizing** takes a target torque and makes the hot-low torque with
  production variation (`metal.torque_hot_low_Nm`) meet it, by adjusting ONE
  free variable the user picks: axial magnet length (default), magnets per
  ring (discrete; poles stay even), or ring radius. Every other input stays
  fixed. The engine solves with a bracketed 1-D search: bisection for the
  continuous variables, stepping for the discrete one. It returns the
  smallest value that meets the target, or "not reachable" with the best
  value achieved inside the variable's slider range. It fails loudly and
  never extrapolates past the range. A multi-variable optimizer is out of
  scope for v1.
- **Housing autofit.** Cup, sleeve, liner, cap and the axial stack are always
  derived from the magnet layout and the calculator's existing clearance and
  wall rules, in both modes, and the geometry view redraws them live.
- **Space claim.** The envelope (43 mm diameter × 35 mm overall length, and
  the 20 mm large-diameter bay, from the metal-design inputs) is drawn as a dashed outline. Exceeding it shows a
  red callout on the view naming the overshoot in mm per axis, plus a red
  dashboard badge.

### A2. Equation explorer

- **Explanation layer.** Each explained result registers an equation record:
  target id, display symbol, a display formula in a small markup (fractions,
  sub/superscripts, Σ, √), its term ids, unit and workbook cell. The formula
  markup is itself what the drift guard evaluates over the term values; a
  Rust closure only where the markup cannot state it, counted and reviewed
  by hand (plan A-3: none). The engine code stays as ported (parity), and
  the explanation layer is separate.
- **Drift guard (test).** For every equation record, its evaluation over the engine's
  term values must reproduce the engine's result (1e-9 relative) at defaults
  and at the M2 differential-test input sets. The equation shown is provably
  the one that produced the number.
- **Hover.** Any displayed value (dashboard, results table, geometry
  callouts) shows a tooltip with its equation. Each term is coloured, and the
  same colour marks that term wherever its value appears on screen.
- **Equation panel** (docked, toggleable). It shows the open equation large,
  its terms with values and units, a breadcrumb trail, and a "used by" list.
  Clicking a term drills into that term's own equation. Leaf terms (inputs)
  highlight their slider. One equation at a time: never a page of every
  formula.
- **Rendering.** A small egui typesetter for the markup: inline fractions,
  sub/superscripts, Σ and √. No LaTeX dependency.

### A3. Assumptions panel

- **Toggle panel** that separates model **assumptions** from design inputs.
  Each assumption shows its value, unit, rationale and source.
- **The v1 set:** harmonics included (workbook 1, 3, 5; selectable up to 11),
  end-effect coefficient, calibration factor, production variation (±15 %),
  Br and Hcj temperature coefficients, demag knee fraction, demag margin,
  back-iron design flux density (1.5 T), thermal conductance, driving rise,
  slip-event duration, clamp friction coefficient, preload fraction of proof
  load.
- **Engine support.** Assumptions that are hard-coded in the Python engine
  today (for example the harmonic list and the knee fraction) become engine
  parameters in M2, with defaults equal to the workbook's.
- **Traceability.** A changed assumption flows through every dependent
  equation. In the equation panel the term is styled as an assumption, with a
  changed-from-default dot. An "assumptions modified" banner shows
  whenever any assumption differs from its workbook default, beside a
  "reset to workbook defaults" button.

### A4. Teaching notes

- An optional **Explain** section per equation, hidden by default and
  toggled in the equation panel. It always shows the note for the equation
  currently open, so it follows what the user investigates.
- **Content:** 2–6 plain-language sentences at Physics 2 level, an optional
  "watch out" line, and an optional small diagram (for example the
  square-wave magnetization and its harmonics, and the flux path with and
  without back iron).
- **Scope:** notes for about 15–20 genuinely tricky ideas, not every
  equation. They cover: harmonic decomposition; the back-iron (sinh) vs
  free-space factor; pull-out torque vs rotation angle; end effect; Br(T)
  and torque ∝ Br²; demagnetization, knee and permeance; eddy-current slip
  loss and skin depth; the thermal time constant; clamp preload and
  friction; and the physics behind each A5 material warning.
- **Start here:** a short suggested order (torque chain → back iron →
  temperature → demagnetization → slip heating → clamps) opens the matching
  equations in turn.
- **Accuracy gate:** notes are drafted from the M1 derivations, and each one
  is checked by a physics reviewer before release.

### A5. Materials per part, with consequence warnings

- **Per-part material pickers** backed by a small library. Each entry
  carries: relative permeability (incremental), saturation flux density,
  electrical conductivity, density, CTE, Young's modulus, yield strength,
  specific heat, and ferromagnetic yes/no.
- **Choices:**
  - back iron (hub and cup): 4140 (default), 1018, 12L14, 416 stainless,
    17-4PH, plus non-magnetic 304 and 6061 for demonstration;
  - sleeve and liner: 316L (default), titanium grade 5, Inconel 625, PEEK;
  - cap and housing: 6061-T6 (default), 7075-T6, acetal.
- **Physics links:**
  - a ferromagnetic back-iron choice selects the steel circuit, and a
    non-ferromagnetic one selects the free-space circuit (replacing the bare
    `backiron` selector, which stays available as an override);
  - the design flux density feeds the wall-thickness check (saturation flux
    density is informational and drives the low-saturation warning);
  - conductivity feeds slip losses;
  - density feeds mass and inertia;
  - CTE and modulus feed the bond-stress screen.
- **Warning rules** (plain language, colour-coded, linked to their teaching
  note):
  - non-ferromagnetic back iron: an open magnetic circuit, so torque drops,
    a strong stray field extends outside, and the part collects ferrous chips
    and debris;
  - ferromagnetic sleeve or liner: short-circuits the gap flux, so torque
    collapses;
  - high-conductivity sleeve or liner: higher slip heating;
  - low saturation: back-iron walls need to be thicker;
  - uncoated low-alloy steel: needs plating (corrosion);
  - large CTE mismatch with the magnets: higher bond stress.

### A6. Magnet library: grades and parts

- **Two tables** replace the single part list (DRY: grade data lives once).
  - **Grade:** Br, Hcj, Hcb, (BH)max, α(Br) and β(Hcj) temperature
    coefficients, recoil permeability, maximum operating temperature,
    density.
  - **Part:** vendor, part number, shape (block or arc), dimensions, grade,
    coating, magnetization direction.
- **Custom dimensions** stay available: pick any grade with manual
  dimensions.
- **Grades in v1:**
  - NdFeB N35, N42, N48, N50, N52 (80 °C), and N50M (100 °C);
  - NdFeB N42M (100 °C), N42H (120 °C), N42SH (150 °C), N38UH (180 °C),
    N35EH (200 °C), N33AH (220 °C);
  - SmCo Sm2Co17 grade 26 and grade 30, and SmCo5;
  - hard ferrite Y30;
  - bonded NdFeB (about 0.65 T).
- **Parts:** the existing 15 library parts, mapped to grades, plus K&J stock
  block sizes in the added grades where they exist.
- **Correctness fix in M2** (and reported in the M1 audit if its checks confirm it). Today the demagnetization check uses one
  N42SH curve (a single Hcj input) for every magnet. With grade data, each
  magnet's own Hcj(T) is used. Ferrite's opposite-sign β (coercivity falls
  as it gets colder, so the demag risk is at cold) must be handled, and it
  gets a teaching note. Sources for grade values: supplier datasheets
  (K&J/Arnold/Electron Energy style), cited per grade in the library.

### Addendum testing

- **Inverse sizing:**
  - for each free variable, solving for the forward result's torque returns
    the original variable value (round trip, 1e-6);
  - "not reachable" is reported when the target exceeds what the range
    allows;
  - poles stay even.
- **Housing autofit:** the envelope-exceeded callout triggers exactly when a
  derived dimension exceeds the space claim, per axis.
- **Equation drift guard:** runs over every equation record (A2).
- **Assumptions:**
  - at defaults, results are bit-identical to workbook parity;
  - changing each assumption changes no result outside its dependents in the
    equation registry's dependency graph, and every numeric result on its
    active path at some design point, above rounding (algebraic
    cancellations listed with their reasons).
- **Materials:**
  - each warning rule fires exactly on its condition;
  - a non-ferromagnetic back iron switches the circuit factor.
- **Grades:**
  - every library part resolves to a grade;
  - the demag check uses the part's own Hcj(T), including a ferrite
    cold-case test.
- **GUI (headless egui):**
  - hover shows the equation;
  - clicking a term drills in, and the breadcrumb returns;
  - the Explain toggle shows the current equation's note;
  - the assumptions banner appears on change and clears on reset.
