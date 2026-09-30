# magcoupling-rs

Magnetic slip coupling design calculator: the Rust port of the `magcoupling`
1.0.0 Python package, itself a port of `magnetic_coupling_torque_calculator.xlsx`.
Spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (M2).

The vendored Python package, `reference/magcoupling-py/`, is the **oracle**.
Every result must match the workbook snapshot and the Python engine, except
where an approved correction from the M1 math audit is registered in the
deviation registry.

**Status:** M2 complete: every Python module except `fields3d` (M3) is ported;
workbook parity covers all 1,149 checks; E1 to E14 applied (decisions D1 to D7
as recorded).

The 1,149 checks are 330 result cells, 659 table cells (494 sweep cells and 165
screw-table cells) and 160 default inputs. The corrections are listed under
[Differences from the workbook](#differences-from-the-workbook); E14 rewords help
text and README only, no number changes. Decisions D1 to D7 are recorded in
`docs/ai/04-memory.yaml` and `docs/ai/05-update-tracker.md`. Next: M3 (`fields3d`,
which must also apply E3 and E5), then the M4 GUI.

```rust
use magcoupling::{DesignInputs, compute_all, headline};
let res = compute_all(&DesignInputs::default());
for (key, value) in headline(&res) {
    println!("{key}: {value:?}"); // the dashboard numbers, in Python's order
}
```

## Invalid inputs

`compute_all` never panics, whatever the inputs hold. `InputSet::set` refuses a
wrong type, NaN, an infinity and a selector code outside its choices, but a struct
literal, a design file or a share link can hold one anyway. Then the results are
meaningless, not fatal. A selector code outside its choices gives NaN numbers or
the Excel-style text `"#N/A"` where Python would raise or silently pick another
row (the adhesive, the screw class), and the workbook's own IF fall-through where
a two-way IF decides. A typed value far outside its slider gives results that may
be inf or NaN. Call `DesignInputs::validate()` where inputs enter: it returns
every offending path, in schema order, with the reason `set` would give.

## Differences from the workbook

Every correction below is approved in the M1 math audit report
(`docs/analyses/2026-09-29-magcoupling-math-audit.md`, the row with the same
id) and registered in `src/engine/deviations.rs` with the cells it changes and
their workbook and corrected values (for E3, E4 and E5, which change 246, 47 and
37 cells, in the reviewed golden files `tests/data/deviations/E3.json`, `E4.json` and `E5.json`). The corrections are always on for users
(`compute_all`); only the parity and differential tests switch them off,
through the test-only `workbook-parity` feature, to compare against the workbook
and the Python engine exactly.

| Id | Cells | Workbook | This port | Report |
|---|---|---|---|---|
| E1 | Temperature design!C96 (feeds C104-C106, C201, C202) | 0.55 GPa (EA 9514's modulus) | 0.107 GPa (AA 326 TDS); C106 and C202 now read "Below ..." | E1 |
| E2 | Clamp screw sizes!C34:G35, Shaft clamps!C48 | length = CEILING(grip + 2d, 2); fits = length <= grip + thread | the slit (0.8 mm) is added to both; M4 x 12 becomes M4 x 14, which protrudes 0.34 mm: clamps.length_note says so | E2 |
| E3 | Magnet library N42SH rows; Calculator!C17, C21, C27, C31; Calibration!C21 | 1.29 T | 1.30 T (vendor minimum); pull-out 2.688 N·m, limit 93.06 °C, C91 "OK: 7x margin"; fields3d rerun pending (M3) | E3 |
| E4 | Pole sweep!C6:C11 | a_i = MAX(w/(2 tan(pi/N)) + 0.05, bore/2 + key + 2.5) | + inner bondline in the wall term; the 6-pole row reads "outside OD envelope" | E4 |
| E5 | Temperature design!C121 (feeds C125, C130 and the thermal rows) | 1.035e-5 T²·m² (free-space field) | 4.14e-5 T²·m² (doubled at the steel surface); total slip loss 2.751 W; fields3d part is M3 | E5 |
| E6 | Calculator!C63 | OD/2 − block-back apothem | OD/2 − (block back + outer bondline): 2.723 mm | E6 |
| E7 | Calculator!C76, C82, C88 → C89-C96; sweeps N, Q, T, U; Calibration!C40-C42 | every harmonic at half a pole pitch | the maximum of the harmonic torque-angle curve (closed form); same at defaults; 6 poles 0.861 → 0.911 N·m | E7 |
| E8 | Calculator!C9, C57, C111; Metal design!C175 | flat-block corner geometry in arc mode | the corner radius C55 (face radius for arcs), round pocket; arc-mode pull-out 2.99 → 2.65 N·m | E8 |
| E9 | Calculator!C111, C113; Materials!C22 | steel cup and boss, back-iron wall advice even with no back iron | aluminium cup and boss, "No back iron"; total mass 156.9 → 96.8 g at backiron = 0 | E9 |
| E10 | Calculator!C103 → C104-C106; Materials!C20-C22 | (Br_i + Br_o)/2 · (t_i + t_o)/(t_i + t_o + g) | (Br_i t_i + Br_o t_o)/(t_i + t_o + g); same for identical rings | E10 |
| E11 | Temperature design!C91 | 0.2 typed into the formula | the fatigue-endurance input C195 | E11 |
| E12 | Temperature design!C19, C23, C150-C153 | negative times and drag when the start is above the limit | 0 s, 0 rev, 0 N·m | E12 |
| E13 | Temperature design!C156, C157 | #DIV/0! (Python: the whole calculation stops) | +inf; the other results are computed. JSON export must encode inf (M4) | E13 |
| E14 | Shaft clamps!C35 help; README | "At 22 mm only M3 fits; two of them need a 14.5 mm clamp." | "At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3." | E14 (documentation) |

Results can be +inf (E13); exporters must handle it.

## Layout

| Path | Contents |
|---|---|
| `src/lib.rs` | Crate docs and re-exports: `compute_all`, `headline`, `DesignInputs`, `DesignResults` |
| `src/engine/meta.rs` | Field metadata: `inputs!`/`results!`, `param`/`out` builders, `Value`, get/set/visit by dotted path; table rows (`rows!`, `TableLayout`, `col`/`at_row` builders) with synthesized workbook cells |
| `src/engine/compat.rs` | Python and Excel semantics the port reproduces (see the translation rules below) |
| `src/engine/deviations.rs` | Registry of the approved workbook corrections E1 to E14 (all applied), probes, and the `Deviations` switch |
| `src/engine/constants.rs` | Physical constants (`MU0 = 1.256637e-06`, the workbook's rounded value) |
| `src/engine/library.rs` | `Magnet library` sheet: the stock magnet rows and the exact-text lookup; `br_T` applies E3 |
| `src/engine/calibration.rs` | `Calibration` sheet: prototype measurement and model calibration (23 result cells) |
| `src/engine/model.rs` | `Calculator` sheet: coupling and magnet inputs, `ModelResults` (73 cells), the fixed harmonic set `HARMONICS` (1, 3, 5), and the mass estimate `MassResults` (6 cells) |
| `src/engine/metal_design.rs` | `Metal design` sheet: 48 inputs, retainers (9 cells), `MetalDesignResults` (49 cells), the validation checklist `VALIDATION_ITEMS` |
| `src/engine/materials.rs` | `Materials` sheet: steel, nickel and screw-class inputs, the aluminium alloys, `ScrewClasses::proof`, `MaterialsResults` (7 cells) |
| `src/engine/temperature.rs` | `Temperature design` sheet: 7 input groups (37 cells), the adhesive table `ADHESIVES` with `selected_adhesive`, `TemperatureLinks`, `TemperatureResults` (130 cells) |
| `src/engine/clamps.rs` | `Shaft clamps` and `Clamp screw sizes` sheets: 25 input cells, `SCREW_SIZES`, `MACHINING_STEPS`, `ClampResults` (33 cells), the 165-cell screw table, `screw_class_name` |
| `src/engine/sweeps.rs` | `Gap sweep` (338 table cells) and `Pole sweep` (156) sheets |
| `src/engine/api.rs` | `DesignInputs`, `DesignResults`, `compute_all`, `headline` (with `HEADLINE`), `DesignInputs::validate` |
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
| `tests/parity.rs` | Every result with a workbook cell (table values too: their cells are synthesized from the table layout) and every default input equals `tests/data/reference_values.json` (numbers 1e-9 relative, 1e-12 absolute; text exact), deviations off. Per-group cell counts are a ratchet (`PORTED_INPUTS`, `PORTED_RESULTS` with `cells` and `table_cells` in `tests/common/mod.rs`); `the_port_checks_every_cell_test_parity_checks` pins their totals to `test_parity.py`'s 1,149. |
| `tests/differential.rs` | Every result of every seeded case equals the Python engine (`tests/data/differential/<group>.json`), deviations off. The full run (`differential/full.json`, `full_run_matches_python_on_every_case`) varies all 160 inputs at once and compares every result, groups and tables, so it also catches a `MODULES` entry that forgot an input group; `every_selector_pair_is_covered_in_the_full_run` checks that every pair of selector choices across groups occurs in it, and `every_selector_choice_appears_in_every_module_file` that each module file sets every selector it varies to every choice. `every_branch_is_reached` checks the `BRANCHES` table (every branch of every text result is hit), `every_text_result_has_a_branches_entry` that no text-producing result lacks a `BRANCHES` row (a new branch cannot land unchecked), and `every_varied_input_takes_two_values` that each varied input changes; the helpers corpus checks `compat` against Python exactly. Rust-only results (`ResultMeta::rust_only`, e.g. `clamps.length_note`) have no Python counterpart and are skipped. |
| `tests/python_schema.rs` | Every ported field carries the Python label, unit, help, cell, choices and default; no Python field of a ported group is missing; each ported table has the Python field order, row count and cell of every value (`tables_match_the_python_layout`); `headline` has Python's keys, order and values at the defaults; inputs and scalar results are listed in the Python order (the GUI's tables and CSV export follow it); Rust-only results are skipped. |
| `tests/static_data.rs` | Static tables equal the Python engine's, value for value (`tests/data/static_data.json`): the magnet library and its exact-text lookup, the harmonic set, the validation checklist, the aluminium alloys, the adhesives, the screw sizes and machining steps, and the sweep variables. |
| `tests/robustness.rs` | Inputs no parity or differential case holds (the plan's Review Focus): a selector code outside its choices set directly on the struct (adhesive, screw class, and every selector at once with `validate()` naming each), a measured drag of exactly zero, every numeric input at 0, -1, a tenth of its minimum, ten times its maximum and 1e300 (`compute_all_never_panics_on_extreme_inputs`), and a debug-build time bound per `compute_all` call. The engine must never panic. |
| `tests/schema.rs` | Slider ranges, selectors, labels, unique well-formed paths and cells (a Rust-only result has none); each table column's label, unit and note equal the workbook headers (`table_columns_match_the_workbook_headers`; where a correction rewords a column's note, the recorded workbook text equals the note and the port's differs); exports `tests/data/input_schema.json`. |
| `tests/deviations.rs` | Registry cells exist in the snapshot, entries are approved rows of the audit report, and one correction switched on changes exactly its registered cells (hand-listed, or in the correction's golden file under `tests/data/deviations/`); a correction that changes more than 15 cells uses a golden file and one that changes 15 or fewer lists them (`broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`); each applied correction's figures match the report (`e1_adhesive_shear_modulus_matches_the_report`, `e2_clamp_screw_length_matches_the_report`, `e3_library_remanence_matches_the_report`), and help a correction rewords belongs to a real input and differs from the workbook's. A correction that changes nothing at defaults (E7 to E13) carries registry probes, off-default inputs from the report whose cells are checked with the correction off and on (`each_probe_shows_its_correction`); `every_applied_engine_correction_is_visible_somewhere` requires every applied engine correction to show at defaults or in a probe. A probe's workbook value can be a workbook error (`Literal::Error`, e.g. `#DIV/0!` where Python raises): that side still runs but is not compared. E7, E8, E9, E10, E11, E12 and E13 must leave every default cell bit for bit (`e7_leaves_every_default_cell_bit_for_bit`, `e8_leaves_flat_blocks_bit_for_bit`, `e9_leaves_every_default_cell_bit_for_bit`, `e10_leaves_every_default_cell_bit_for_bit`, `e11_leaves_every_default_cell_bit_for_bit`, `e12_leaves_every_default_cell_bit_for_bit`, `e13_leaves_every_default_cell_bit_for_bit`). `all_corrections_together_give_the_reviewed_headline` pins what users see: the headline with every correction on at the default design. |

### Regenerating test data

The slider ranges are defined once, in Rust. Three generators write the data
files; `cargo test` then compares:

| # | Generator | Writes |
|---|---|---|
| 1 | `MAGCOUPLING_BLESS=1 cargo test --test schema` | `tests/data/input_schema.json`: every input's metadata and slider range, read by generator 2 |
| 2 | `cd reference/magcoupling-py && ./.venv/Scripts/python tools/gen_differential.py` (`.venv/bin/python` on POSIX; `--check` only verifies, and is what the gate runs) | `tests/data/python_schema.json` (the Python metadata, with the table layouts), `tests/data/static_data.json` (the static tables) and `tests/data/differential/*.json` (one file per result group, `helpers.json` for `compat`, and `full.json`) |
| 3 | `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` | `tests/data/deviations/E<k>.json`, the golden files of the broad corrections (E3, E4, E5); see [Deviations](#deviations). It writes what this engine computes with the correction alone, so review the diff |

Run 1, then 2, when an input, a range, a choice or a table layout changes; run 3
when a broad correction changes cells.

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

`differential/full.json` (`FULL`) is not a module: its cases vary every input
group at once (`full_cases`: the workbook defaults, `FULL_RANDOM` = 100 random
sets, then one case for each pair of selector choices across groups that the
random sets missed) and compare every result group and table. A new module needs
no line for it: the full run picks up its inputs and results from the schema.

`BRANCHES` in `tests/differential.rs` lists, per text-producing result path
(`[*]` matches any table row), the numbers and texts (`Number`, `Text`,
`Prefix`) that the module's cases must reach, so each branch of the Python
source is compared at least once. Add its rows when a module lands:
`every_text_result_has_a_branches_entry` fails while a `Text` or `NumOrText`
result has none.

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
   text sentinel, `String` for built text, `i64` for an index. Tables (the screw
   table, the sweeps): one `rows!` struct per Python row dataclass, each field
   with `col(unit, label, help, "B")` (rows run down the sheet) or
   `at_row(unit, label, help, 6)` (rows run across it), `uncelled_col` for a
   field without a cell; list it in the `results!` struct as
   `tables { name: Row => TableLayout::RowsDown { sheet, first_row } }` (or
   `ColumnsAcross { sheet, columns }`). Its values get the paths `name[i].field`
   and synthesized cells, so the parity test checks them and
   `table_columns_match_the_workbook_headers` checks each column's label and
   unit against the workbook; add the table's cells to `table_cells` in
   `tests/common/mod.rs`, and its layout to `table_layouts()` in
   `gen_differential.py` (`tables_match_the_python_layout` compares).
   `result_rows` lists a group's tables after its scalars, so the screw table
   comes after `clamps.layout_relief` (Python declares it after
   `boss_radius_mm`); the scalar order is Python's.
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
8. Add one equality-edge unit test per module with threshold comparisons: put
   each comparison at exact equality (inputs chosen so both sides are the same
   `f64`), assert the equality first, so a drifted input fails loudly instead of
   testing the wrong side, then assert the branch Python takes there. The module
   tests of `model`, `metal_design`, `materials`, `temperature`, `clamps` and
   `sweeps` are the examples (`checks_take_the_python_branch_at_exact_equality`,
   `wall_check_passes_at_equality`, `hot_day_torque_meets_the_minimum_at_equality`,
   `status_checks_are_strict_at_exact_equality`).

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
| `{1: a, 2: b}[code]` (KeyError, or a negative index that wraps around) | a `match` whose fallback yields NaN or an Excel-style error text; never panic, never another row. Three uses: `ScrewClasses::proof` (NaN), `selected_adhesive` (`NO_ADHESIVE`: name `"#N/A"`, NaN numbers) and `screw_class_name` (`"#N/A"`) |
| an `int` from an input or a derived count (`coupling.npole`, `calibration.total_magnets`, `clamps.joint_screws`) | `i64`, but never `+`, `-` or `*` on it: Python ints cannot overflow and a debug build panics. Convert with `n as f64` for arithmetic; `x as i64` saturates (NaN gives 0) and `saturating_add` replaces Python's `+` on an int; comparisons against selector codes stay integer |
| an exception (`ZeroDivisionError`, math domain) | Rust yields inf or NaN; slider ranges keep Python in its domain, and the generator fails loudly if Python raises |
| rounded workbook constants | the same literal (`MU0 = 1.256637e-06`, not `4π·1e-7`) |

## Deviations

A correction is applied in its own commit after its module is workbook-exact:
branch at the formula with `if dev.is_on(DeviationId::Ek) { corrected } else
{ workbook }` (keep the workbook form beside it); for a corrected default,
declare the corrected default and record the workbook value in the entry's
`workbook_input_defaults`; set the entry to `Applied` and list every cell that
changes at defaults in `changes_at_defaults`. Where the correction rewords a
help text, record the workbook's text in `workbook_help` (an input path, or a
table column as `group.table[*].field`): `tests/python_schema.rs` compares
Python's help against it, `tests/schema.rs` the workbook note of a table
column, and both `tests/deviations.rs` and `tests/schema.rs` require the port's
help to differ. A result the Python engine does not have (E2's
`clamps.length_note`) is declared with `out_rust_only`: it has no cell, and the
metadata-parity, order and differential tests skip it. `tests/deviations.rs`
then proves the correction changes exactly those cells, to those values; add a
test of the report's figures (`assert_report`, `assert_workbook`) and a row to
[Differences from the workbook](#differences-from-the-workbook). Physics
changes get a dedicated physics reviewer (spec, testing summary).

A broad correction, one that changes more than 15 cells at defaults (decision
D4; E3 changes 246 through Br), is not hand-listed: its entry leaves
`changes_at_defaults` empty and names a golden file in `changes_file`,
`tests/data/deviations/E<k>.json`, which maps every changed cell to
`[workbook, corrected]`. Write it from a run with the correction alone:

```bash
MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells
```

(run only this test: the rest of the `deviations` binary may run first and fail
on a missing file). Then review the file as a diff: every changed cell should
follow from the correction (for E3, Br-linear values move by 1.30/1.29, torques
by its square, safety factors by the inverse), and the report's figures stay
pinned by hand in the correction's `e<k>_..._matches_the_report` test, so a
blessed file cannot quietly move them. Without `MAGCOUPLING_BLESS` the test
compares against the file.
