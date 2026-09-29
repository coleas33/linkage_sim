# magcoupling

Design calculator for the rail robot's magnetic slip coupling: a coaxial, synchronous permanent-magnet coupling that sits between the traction gearbox output and the wheel. It carries the driving torque and slips, without contact or wear, when the rail forces the wheel faster than the gearbox can safely go.

This package is a Python port of `magnetic_coupling_torque_calculator.xlsx`. It reproduces every formula in the workbook: all 935 formula cells are checked against the workbook's own values by the test suite. It is written to be dropped into another repository and driven from a GUI.

## Purpose

The coupling protects a small 5:1 gearbox (0.6 N·m input rating, 3,000 rpm limit) from two things the rail can do during a wheel swing:

- **Overspeed while gripping.** The rail drives the wheel faster than the gearbox allows.
- **Torque spikes while skidding.** Friction at touchdown puts a sudden torque through the drivetrain.

A magnetic coupling limits torque without touching, so repeated slip events (a 20 million event life target) cause no wear.

The calculator answers the design questions:

- **Torque:** how much pull-out torque a given magnet layout gives, hot and cold, with a production allowance.
- **Fit:** whether it fits the 43 mm × 35 mm envelope, with enough running clearance and back-iron thickness.
- **Temperature limits:** how hot it can run before the magnets demagnetize or the adhesive gives up.
- **Slip heating:** how much heat slipping makes, and how long it can slip.
- **Shaft clamps:** which cap screw clamps the aluminium adapter to a 10 mm shaft.

## Intent

- **Same numbers as the workbook.** Each result field names the workbook cell it reproduces, for example `Calculator!C93`. The tests compare against a snapshot of the workbook, so any drift shows up immediately.
- **Built for a GUI.**
  - Inputs are plain dataclasses with a default, unit, label, help text and source cell on every field.
  - `input_schema()` and `result_schema()` return that metadata as lists, so forms and result tables can be generated instead of hand-built.
- **Pure functions, no I/O, no global state.** `compute_all(inputs)` returns a results object and takes a few milliseconds, fast enough to recompute on every keystroke.
- **No required dependencies.** The core is standard-library Python. Two optional modules need extra packages:
  - `magpylib` for the 3D field check
  - `matplotlib` for the clamp drawing

## Quick start

```bash
pip install -e .                 # core
pip install -e ".[fields3d,drawing,dev]"   # optional extras and tests
python -m magcoupling            # headline results with workbook defaults
python -m magcoupling --set coupling.npole=12 --set metal.face_gap_mm=1.2
python -m magcoupling --json results.json
pytest                           # parity with the workbook + API tests
```

```python
from magcoupling import DesignInputs, compute_all, headline, set_input

res = compute_all(DesignInputs())                 # workbook defaults
res.model.pullout_Nm                              # 2.647 N·m at 50 °C
res.temperature.summary.governing_limit_C         # 92.55 °C (magnets, skipping case)
res.clamps.recommended                            # 'ISO 4762 M4 x 12, class 12.9'

hot = compute_all(set_input(DesignInputs(), "coupling.op_temp_C", 80))
print(headline(hot))
```

## Using it from a GUI

```python
from magcoupling import DesignInputs, compute_all, input_schema, result_schema, set_input, headline

inp = DesignInputs()
form = input_schema(inp)       # [{path, label, unit, help, cell, choices, value, kind}, ...]
inp = set_input(inp, "metal.face_gap_mm", 1.2)   # returns a new object; the old one is untouched
res = compute_all(inp)
dash = headline(res)           # the handful of numbers to show first
table = result_schema(res)     # every computed value with label, unit and source cell
```

- **Form building.** Group fields by the first part of `path`: `coupling`, `metal`, `calibration`, `materials`, `temperature`, `clamps`. Selector inputs such as `coupling.backiron` or `clamps.screw_class` carry a `choices` mapping for drop-downs.
- **Result types.** Most results are numbers. Checks and verdicts are text, such as `"Below target"` or `"OK on temperature…"`, and are good for colour-coded status badges. A few fields can be either, such as `"never: steady state stays below the limit"` or `"not measured"`, so handle both.
- **Tables.** `res.gap_sweep`, `res.pole_sweep` and `res.clamps.table` are lists of row dataclasses, ready for tables or plots.
- **Slow jobs.** The optional 3D check (`fields3d.run`) takes about 2 s; run it in a worker thread. The clamp drawing (`drawing.clamp_layout`) returns a matplotlib `Figure` that can be embedded with the Qt or Tk canvas.
- **Examples.** `examples/gui_integration.py` shows the pattern without a GUI framework. `examples/tk_demo.py` is a minimal working Tkinter front end.

## Package map

| Module | Workbook sheet | What it does |
|---|---|---|
| `model.py` | Calculator | Block geometry, 2D harmonic torque model, temperature scaling, fit/back-iron/temperature checks, mass and inertia |
| `calibration.py` | Calibration | Reproduces the model for the 1.8 N·m no-iron prototype and derives the one-point correction |
| `metal_design.py` | Metal design | Hot/cold torque with variation, radial clearance stack, sleeve/liner/cap/endplate geometry and mass, axial envelope, optional aluminium adapter, slip duty |
| `materials.py` | Materials | 4140 + electroless nickel, 7075/6061 aluminium, screw classes; cup-wall check and pre-plate machining offsets |
| `temperature.py` | Temperature design | Demagnetization onsets, adhesive selection and bond loads, thermal-mismatch screen, slip-loss estimate, thermal time constant, unbroken-slip timing, life totals, magnet and adhesive life checks |
| `clamps.py` | Shaft clamps, Clamp screw sizes | One-piece slotted clamp sizing: per-size screw table, recommendation, key backup, adapter joint, cut-layout dimensions |
| `sweeps.py` | Gap sweep, Pole sweep | Pull-out vs corner gap and vs pole count, with fit/status flags |
| `library.py` | Magnet library | Stock magnet dimensions, Br and ratings; looked up by exact part text |
| `api.py` | – | `DesignInputs`, `compute_all`, schema and serialization helpers |
| `fields3d.py` | – (optional) | magpylib 3D check that regenerates the Temperature design's 3D inputs for a new geometry |
| `drawing.py` | – (optional) | Dimensioned end and top views of the recommended clamp |

Calculation order, which is the same as the workbook's dependencies:

```
Calibration ─► Calculator (model) ─► Metal design retainers ─► Calculator mass
            ─► Metal design ─► Materials ─► Temperature design ─► Shaft clamps ─► sweeps
```

## Approach

### Torque model (`model.py`)

Flat blocks sit on polygon faces: ten per ring by default, B842SH (12.7 × 6.35 × 3.17 mm, N42SH). The inner blocks' corners stick out, so the outer blocks' faces are set back from the corners. The flat-face gap is the input. The corner gap and all radii are derived from it.

1. **Harmonic field.** Each ring's alternating magnetization is a square wave around the gap. Its odd harmonics n = 1, 3, 5 have amplitudes B_n = Br · 4/(nπ) · sin(n · fill · π/2), where fill is the block width divided by the pole pitch.
2. **Shear stress at pull-out.** τ_n = B_in · B_on / (2µ0) · S_n · sin(nπ/2), with wave number k = n · (poles/2) / R_gap. The factor S_n depends on the circuit:
   - With steel back iron: S_n = sinh(k·t_i) · sinh(k·t_o) / sinh(k·(t_i + t_o + g)).
   - With free-space rings: S_n = (1 − e^−k·t_i)(1 − e^−k·t_o) · e^−k·g / 2.
3. **Torque.** T = Σ τ_n · 2π·R_gap² · L, multiplied by two factors:
   - an end-effect factor, 1 − c_end · pole pitch / L;
   - a calibration factor: 0.95 for the steel-backed candidate, or the measured correction for the no-iron prototype only.
4. **Temperature.** Br varies linearly with temperature at −0.12 %/°C, and torque varies with Br².

This is an analytical estimate, not a guaranteed minimum. Finite permeability, saturation, the solid cup web and eddy currents are not solved. A 3D check with first-order steel images gives 2.17 N·m at 20 °C against the model's 2.85 N·m. Images are a lower bound for real iron, but it is one more reason the hot bench test matters.

### Metal design (`metal_design.py`)

- **Torque range.**
  - The hot low value is the pull-out at 50 °C × (1 − 15 %).
  - The cold high value is the pull-out at −40 °C × (1 + 15 %).
  - The cold high value also sets the clamp and bond loads.
- **Running clearance.** The radial gap between the rotating 316L sleeve (0.10 mm) and liner (0.20 mm), minus the allowances for shaft float, runout, deflection, thermal growth, sleeve form and magnet position.
- **Masses and envelope.** Gross solids for mass (holes and threads are not subtracted). The axial stack is checked against the 20 mm large-diameter bay and the 35 mm overall length.

### Materials (`materials.py`)

- **Magnetic parts.** Annealed 4140 at a design flux density of 1.5 T. That feeds the back-iron thickness check: the cup corners need about 1.9 mm.
- **Plating.** High-phosphorus electroless nickel is non-magnetic, so it doesn't change the gap. The module reports the pre-plate machining offsets.
- **Other parts.** Aluminium goes everywhere outside the magnetic path: 7075-T6 for clamps, 6061-T6 for the cap and housing.

### Temperature design (`temperature.py`)

- **Demagnetization.**
  - The worst reverse field inside a block comes from the 3D check, for four cases: aligned, pull-out, like poles facing, and a single ring on its carrier.
  - Each is compared with the knee of the N42SH curve, Hk(T) = 0.9 · Hcj20 · (1 − 0.5 %/°C · (T − 20)), with the fields themselves scaling with Br(T).
  - A magnet at permeance coefficient 1 calibrates the method to the 150 °C rating, which shifts every onset down by 10.4 °C.
  - Skipping is the lowest onset (102.6 °C). A 10 °C margin gives the 92.6 °C magnet limit.
- **Adhesive.**
  - The design limit is the TDS service maximum or Tg − 20 °C, whichever is lower. The recommended Loctite AA 326 + SF 7649 is rated to 120 °C, so it doesn't govern.
  - Bond shear from torque is about 0.4 MPa, and the magnetics press the blocks onto the steel.
  - A Volkersen shear-lag screen shows thermal-mismatch shear at the block ends is the real bond load. NdFeB barely expands across its magnetization while steel does.
- **Slip heating (estimate).** The bench drag torque replaces this estimate as soon as it's entered.
  - Solid steel surfaces use the loss of a travelling field on a permeable conductor, which grows with speed^1.5.
  - The 316L sleeve and liner and the aluminium cap use the thin-shell low-speed limit with an end factor, which grows with speed².
  - The magnets use the thin-strip formula.
- **Thermal network.** The heat capacity comes from the part masses (about 82 J/K). One conductance to the surroundings (0.3 W/K placeholder) gives a time constant of about 4.6 minutes. From that come the per-event rise, the steady continuous-slip temperature, the time and rotations to reach the limit, and a critical drag torque that tells the bench test what to look for.
- **Life checks.**
  - Life totals: 20 million events × 0.1 s at 2,000 rpm is 67 million slip rotations.
  - Magnet and adhesive checks are made at the hot-day peak, which is the ambient, plus the driving rise, plus a fault-limited slip.

### Shaft clamps (`clamps.py`)

The recommended clamp is a one-piece slotted 7075-T6 clamp on a keyed shaft. It is sized so the clamp alone holds twice the coupling's cold high torque; the key is the backup.

For each size from M2.5 to M6 the module works out three things:

- **Geometry:** the screw offset, the wall outside the hole, the head seat, the grip and the thread length.
- **Preload:** 75 % of proof load, capped by thread stripping in the aluminium.
- **Capacity:** µ · preload · shaft diameter · clamp factor.

It also checks how many screws fit along the clamp. The first size that passes everything is recommended.

## Validation

- **Test coverage.** `tests/test_parity.py` checks every one of the workbook's 935 formula cells, plus the constant columns of the sweep and screw tables: 330 mapped result fields, 494 sweep cells and 165 screw-table cells. It also checks the 160 default inputs.
- **Tolerance.** Numbers must agree to 1e-9 relative, and text must match exactly.
- **Updating the snapshot.** `tests/reference_values.json` is the workbook snapshot. After editing and recalculating the workbook, regenerate it:
  ```bash
  python tools/extract_reference.py path/to/magnetic_coupling_torque_calculator.xlsx
  pytest tests/test_parity.py
  ```
- **Optional modules.** `tests/test_optional.py` confirms the 3D module reproduces the workbook's 3D inputs to within 1 % and that the drawing renders. Both are skipped if the extras aren't installed.

## Inputs that are placeholders — replace them with measurements

| Input | Default | Why it matters |
|---|---|---|
| `metal.measured_drag_Nm` | None | Replaces the slip-loss estimate. If drag at 2,000 rpm exceeds the reported `critical_drag_Nm` (about 0.04 N·m), the slip fault needs a time limit |
| `temperature.duty.driving_rise_C` | 10 °C | Sets the hot-day starting temperature (the most influential thermal input) |
| `temperature.thermal.conductance_W_K` | 0.3 W/K | Sets steady rise and time constant |
| `metal.slip_event_s` | 0.1 s | Life rotations, heat per event |
| `temperature.duty.life_hours` | 20,000 h | Average slip duty |
| `temperature.adhesive_life.hot_strength_retained` | 0.5 | Hot fatigue margin (not published for AA 326) |
| `calibration.test_temp_C` | 20 °C | Assumed, not recorded |
| clearance allowances in `metal` | various | Marked UNCONFIRMED in the workbook |

## What the default design currently shows

These come straight from the defaults. They are open design items, not code issues:

- **Hot torque.** The hot low torque with the 15 % allowance is 2.25 N·m, against the 2.5 N·m requirement. The nominal value is 2.65 N·m.
- **Running clearance.** It's −0.10 mm after allowances, below the 0.2 mm target.
- **Cup wall.** With 4140 at 1.5 T the cup corners need 1.9 mm; the design has 1.8 mm. Raise `metal.cup_wall_corner_mm` to 2.0 mm and recheck the cap thread.
- **Clamp boss.** The clamp calculator uses a 25 mm boss, which takes one M4. The Metal design sheet still has 22 mm. At 22 mm only M3 fits, and two of them need a 14.5 mm clamp.

## Differences from the workbook

- **Magnet not in the library.** The workbook's Temperature design sheet returns an error. Here the demagnetization onsets are left uncalibrated (offset 0) instead of failing.
- **22 mm boss note.** The workbook's Shaft clamps note says two M3 screws need a 14 mm clamp. The calculation needs 14.5 mm, and the help text here says so.
- **Everything else** is formula-for-formula, including the text of every check.

## Units and conventions

| Quantity | Unit |
|---|---|
| Lengths | mm |
| Torque | N·m |
| Temperature | °C |
| Flux density | T |
| Reverse field | kA/m |
| Stress | MPa |
| Power and energy | W, J |
| Speed | rpm, rad/s |
| Mass | g |

- **Densities** use the workbook's g/mm³ values.
- **Selectors** use the workbook's integer codes, for example `backiron`: 1 = steel, 0 = none; `clamp_type`: 1 = one-piece, 2 = two-piece.
- **Field names** end in their unit where it isn't obvious: `_mm`, `_Nm`, `_C`, `_W`.

## Extending

- **New magnet:** add a `MagnetSpec` row in `library.py`. For a new geometry, rerun `fields3d.run`, then `fields3d.apply_to_inputs`, so the temperature limits follow.
- **New adhesive:** add an `AdhesiveCandidate` to `temperature.default_adhesives()` and select it by its 1-based index.
- **New screw size:** add a `ScrewSize` to `clamps.SCREW_SIZES`. The parity test covers only the five workbook columns.
- **More harmonics:** change `model.HARMONICS`. This breaks parity with the workbook on purpose, so keep the tests on the default.
