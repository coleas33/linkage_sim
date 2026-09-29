# magcoupling-rs

Magnetic slip coupling design calculator: the Rust port of the `magcoupling`
1.0.0 Python package, itself a port of `magnetic_coupling_torque_calculator.xlsx`.
Spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (M2).

The vendored Python package, `reference/magcoupling-py/`, is the **oracle**.
Every result must match the workbook snapshot and the Python engine, except
where an approved correction from the M1 math audit is registered in the
deviation registry.

**Status (M2, in progress):** engine infrastructure (metadata model, Python and
Excel semantics helpers, deviation registry), the **Calibration** sheet, the
magnet library, the **Calculator model** (`model`), the **Metal design
retainers** (`metal_design::retainers`), the **mass estimate**
(`model::mass_estimate`) and the **Metal design** sheet (`metal_design::compute`,
with the validation checklist `VALIDATION_ITEMS`). The Materials sheet
contributes its inputs only so far.
Deviations E1 to E14 are all registered as `Planned`; none is applied yet.

```rust
use magcoupling::{DesignInputs, compute_all};
let res = compute_all(&DesignInputs::default());
println!("{}", res.calibration.f_cal_updated); // 1.0658
```

## Layout

| Path | Contents |
|---|---|
| `src/engine/meta.rs` | Field metadata: `inputs!`/`results!`, `param`/`out` builders, `Value`, get/set/visit by dotted path |
| `src/engine/compat.rs` | Python and Excel semantics the port reproduces (see the translation rules below) |
| `src/engine/deviations.rs` | Registry of approved workbook corrections, and the `Deviations` switch |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all` |
| `src/engine/<module>.rs` | One module per Python module: `constants`, `calibration`, `library`, `model` (with the mass estimate), `metal_design` so far (`materials` holds inputs only) |
| `tests/` | Parity, differential, metadata and registry tests (below) |
| `tests/data/` | Workbook snapshot copy, exported schemas, differential data |

Not a Cargo workspace member: `linkage-sim-rs` will depend on it by path.
Features: `gui` and `app` are declared for M4 and empty. `workbook-parity` is
**test-only**: it reaches the tests through a self dev-dependency, so shipped
builds cannot switch corrections off.

## Tests

```bash
cargo test                      # from magcoupling-rs/
bash linkage-sim-rs/scripts/gate.sh   # everything, both crates and the Python oracle
```

| Test | Checks |
|---|---|
| `tests/parity.rs` | Every result with a workbook cell and every default input equals `tests/data/reference_values.json` (numbers 1e-9 relative, 1e-12 absolute; text exact), deviations off. Per-group cell counts are a ratchet (`PORTED_INPUTS`, `PORTED_RESULTS` in `tests/common/mod.rs`). |
| `tests/differential.rs` | Every result of every seeded case equals the Python engine (`tests/data/differential/<group>.json`), deviations off. `every_branch_is_reached` checks the `BRANCHES` table (every branch of every text result is hit) and `every_varied_input_takes_two_values` checks that each varied input changes; the helpers corpus checks `compat` against Python exactly. |
| `tests/python_schema.rs` | Every ported field carries the Python label, unit, help, cell, choices and default; no Python field of a ported group is missing. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`). |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells; exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells. |

### Regenerating test data

The slider ranges are defined once, in Rust. The data flows one way:

1. Rust metadata to `tests/data/input_schema.json`:
   `MAGCOUPLING_BLESS=1 cargo test --test schema`
2. `input_schema.json` to the Python generator, which writes
   `tests/data/python_schema.json`, `tests/data/static_data.json` and
   `tests/data/differential/*.json`:
   `cd reference/magcoupling-py && ./.venv/Scripts/python tools/gen_differential.py`
   (use `.venv/bin/python` on POSIX; `--check` only verifies).
3. `cargo test` compares.

`MODULES` in `gen_differential.py` maps each ported **result group** to the
**input groups** each case varies: every group the results read, directly or
through upstream sheets (`TEXT_CHOICES` lists the values a text input may take,
since it has no slider). `differential/<group>.json` is columnar: the header
lists `input_groups`, `input_paths` and `result_paths` once, and each case is
one line `{"id", "tag", "inputs": [...], "results": [...]}` with values in the
header's path order. A module's cases are the fixed ones first (defaults, each
input at each range end and each choice, the `PROBES` on branch boundaries),
then random ones: as many as it takes to reach `CASES_PER_MODULE` and never
fewer than `RANDOM_MIN` (100). A data file over `MAX_FILE_BYTES` (4 MB) fails
the generator: vary fewer groups or cut cases.

`BRANCHES` in `tests/differential.rs` lists, per text-producing result path
(`[*]` matches any table row), the numbers and texts (`Number`, `Text`,
`Prefix`) that the module's cases must reach, so each branch of the Python
source is compared at least once. Add its rows when a module lands.

The gate runs `gen_differential.py --check`, so stale data fails it. The
snapshot copy must equal `reference/magcoupling-py/tests/reference_values.json`
(a test checks).

## Porting a module

1. Read the Python module whole. List its dataclasses and cells, helpers,
   f-strings, text sentinels, branches and the places Python can raise. Read
   the audit report rows that touch it (the planned deviations).
2. Inputs: one `inputs!` struct per Python input dataclass. Transcribe each
   `name: float = param(default, unit, label, help, cell, choices)` as
   `name: f64 = default => param(unit, label, help, cell).choices(..)`, byte for
   byte (the metadata-parity test catches typos). Add `.range(min, max, step)`
   from physical bounds (and `.log()` for wide ranges) and `.assumption()` for
   the Addendum A3 assumptions.
3. Results: one `results!` struct per Python result dataclass, `out()` calls
   transcribed. Type each field by the values Python actually produces (the
   annotations are not reliable): `f64`, `NumOrText` for a number or a fixed
   text sentinel, `String` for built text, `i64` for an index.
4. Compute: `pub fn compute(.., dev: Deviations) -> XResults`, line by line in
   Python's order with Python's local names (`#[allow(non_snake_case)]` on the
   function where Python uses capitals). Follow the translation rules.
5. Wire it into `api.rs` in the Python `compute_all` order; add the group to
   `PORTED_RESULTS` (and its input groups to `PORTED_INPUTS`) in
   `tests/common/mod.rs` and to `MODULES` (with the input groups it varies)
   and `PROBES` (a case on each side of every branch boundary) in
   `gen_differential.py`.
6. Bless the schema, regenerate the data, run `cargo test`, add the module's
   text results to `BRANCHES` in `tests/differential.rs`, and run the gate.
7. Apply deviations afterwards, one commit each, never in the port commit.

## Translation rules (Python to Rust)

| Python | Rust |
|---|---|
| `float` arithmetic | `f64`, same operand order: never reorder, factor or simplify a formula |
| `x ** 2`, `x ** 3` | `x.powi(2)`, `x.powi(3)`; `x ** y` (float y) is `x.powf(y)` |
| `math.sqrt/exp/log/sin/...`, `math.pi` | `f64` methods, `std::f64::consts::PI` |
| `min(a, b)`, `max(a, b)` | `compat::py_min`, `compat::py_max` (not `f64::min`/`max`) |
| `sum(...)` | a left fold from `0.0` in the same order |
| `lo <= x <= hi` | `(lo..=hi).contains(&x)` |
| `int / int` | `a as f64 / b as f64` (Python `/` is always float division) |
| `math.ceil`, `math.floor` | `.ceil()`, `.floor()`, but Python returns an int: never `-0.0` |
| `_fields.ceiling`, `_fields.floor_` | `compat::ceiling`, `compat::floor_` |
| `f"{x:.2f}"` | `compat::fmt_fixed(x, 2)` (ties to even; `"nan"`) |
| `repr(x)`, `str(x)` of a float | `compat::py_repr` (exact 17-digit ties go to the even digit) |
| `clamps._fmt_num`, `temperature._text0` | `compat::fmt_num`, `compat::text0` |
| `round(x)`, `int(x)` | `x.round_ties_even()`, `x.trunc()` |
| `None` input | `Option<f64>`; a text result sentinel is `NumOrText::Text` |
| `isinstance(x, (int, float))` | a match on `Option` or `NumOrText` |
| `if code == 1 ... else ...` on a selector | the same, catch-all `else` included |
| `{1: a, 2: b}[code]` (KeyError) | a `match` whose fallback yields NaN or an Excel-style error text; never panic |
| an exception (`ZeroDivisionError`, math domain) | Rust yields inf or NaN; slider ranges keep Python in its domain, and the generator fails loudly if Python raises |
| rounded workbook constants | the same literal (`MU0 = 1.256637e-06`, not `4π·1e-7`) |

## Deviations

A correction is applied in its own commit after its module is workbook-exact:
branch at the formula with `if dev.is_on(DeviationId::Ek) { corrected } else
{ workbook }` (keep the workbook form beside it); for a corrected default,
declare the corrected default and record the workbook value in the entry's
`workbook_input_defaults`; set the entry to `Applied` and list every cell that
changes at defaults in `changes_at_defaults`. `tests/deviations.rs` then
proves the correction changes exactly those cells, to those values. Physics
changes get a dedicated physics reviewer (spec, testing summary).
