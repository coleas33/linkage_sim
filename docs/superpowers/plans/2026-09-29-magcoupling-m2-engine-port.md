# Magcoupling M2 Engine Port (Rust) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the M2 engine port of the magnetic slip coupling calculator: every remaining Python module ported to `magcoupling-rs` with workbook parity (1,149 checks), differential tests against the Python engine on seeded inputs covering every slider range and selector branch, and the approved corrections E1 to E14 applied as registered deviations.

**Architecture:** The tracer slice on branch `magcoupling/m2` already built the crate, the metadata model (`inputs!`/`results!`), the Python/Excel semantics helpers (`compat.rs`), the deviation registry and test-only switch, the Calibration port, and the three test layers. This plan ports the remaining modules one at a time in dependency order, each against the unchanged Python oracle, then applies each approved correction in its own commit behind `Deviations`, so the parity and differential tests keep comparing the workbook itself (`Deviations::NONE`) while users always get `Deviations::ALL`.

**Tech Stack:** Rust 2024 edition, std only for the library (wasm32-unknown-unknown clean); `serde_json` (feature `float_roundtrip`) as the only dev-dependency; the vendored Python 3.12 oracle `reference/magcoupling-py/` run with the venv at `C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe`; Git Bash for every command.

**Spec:** `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (section "M2 — Engine port"; Addendum A engine items are OUT of scope: a later plan builds them on this engine). Approved corrections: `docs/analyses/2026-09-29-magcoupling-math-audit.md`, rows E1 to E14 (E14 is documentation only). Architecture: `C:/Users/Cole/AppData/Local/Temp/claude/C--Users-Cole-source-repos-linkage-simulation/3165685f-bf49-42bb-a036-671c0e065429/scratchpad/m2-plan/architecture.md` (sections 4 to 9 are binding: naming, cell mapping, deviation mechanics, the porting pattern, the translation rules, module notes). In-repo short form of the architecture: `magcoupling-rs/README.md`.

**Execution worktree:** `C:/Users/Cole/source/repos/lsim-mag-m2`, branch `magcoupling/m2`. Every command below uses absolute paths into it. Nothing is pushed.

## Decisions (confirmed by the user, 2026-09-29)

**Confirmed:** the user chose the recommended option for D1, D2, D3 and D7 on 2026-09-29; D4, D5 and D6 take the recommended option as announced defaults. Every task below implements the recommended column; no task stops on a decision.

The controller asked these once, before Task 1, in one message. Each has a recommended option that this plan implements. A task that depends on a decision says so in its first line; if the user picked the alternative and this plan gives no concrete variant for it, the task stops and escalates instead of improvising.

| Id | Question | Recommended (implemented here) | Alternative | Needed by |
|---|---|---|---|---|
| D1 | E3: Br of the N42SH library rows | **1.30 T** (vendor grade minimum, like the other K&J rows) | about 1.315 T (nominal); Task 16 gives its expected numbers too | Task 16 |
| D2 | E1: form of the correction | **Constant C96 = 0.107 GPa** now; the per-adhesive modulus goes with Addendum A5 | Per-adhesive modulus now (needs TDS moduli for 2214 and DP460 that nobody has supplied): Task 14 stops and escalates | Task 14 |
| D3 | A selector code outside its choices set directly on a struct (bypassing `set()`) | **Never panic**: NaN or an Excel-style `"#N/A"` text where Python would raise or silently pick another row; plus `validate()` for input boundaries. `compute_all(&DesignInputs) -> DesignResults` keeps the spec's signature | `compute_all` returns `Result` (spec change): stop and escalate | Tasks 7, 8, 10, 12 |
| D4 | Registry form for corrections that change more than 15 cells at defaults (E3 246 cells, E4 47, E5 37) | **Golden file** `tests/data/deviations/E<k>.json`, blessed from `only(E<k>)` and reviewed as a diff, plus hand-entered report-precision checks | Hand-list every cell in `deviations.rs` | Task 16 |
| D5 | `cargo clippy -D warnings` for `magcoupling-rs` only | **Yes** (already gate 5; `linkage-sim-rs` keeps warnings non-fatal) | Drop `-D warnings` from gate 5 | all |
| D6 | E2 "report when no 2 mm step meets both rules": where | **A Rust-only, uncelled result `clamps.length_note`** (empty when the recommended screw fits); C48 changes exactly as the report states | Append the warning to the C48 text (departs from the report's stated C48 text) | Task 15 |
| D7 | The harmonic set as an engine parameter (spec Out of scope: "the engine keeps the parameter"; Addendum A3: selectable up to 11) | **Deferred to the Addendum A engine plan.** M2 keeps the workbook's fixed set: `HARMONICS: [u32; 3] = [1, 3, 5]`, `[Harmonic; 3]`, and E7's closed form (a quadratic in cos² x, valid for {1, 3, 5} only). The Addendum A plan turns the set into a parameter and replaces the closed form by a sign-change search of dT/dx on (0, π/2] that keeps the "`None` when half a pitch is the maximum" rule, so default outputs stay bit-identical | Generalize now (slice-based `shear_stress`, numeric peak search in Task 20): this plan has no concrete variant for it, so Tasks 3 and 20 stop and escalate | Tasks 3, 20 |

Not implemented (optional parts of approved corrections, YAGNI): E7 "optionally report the pull-out angle"; E13 "optionally reject a negative drag" (the measured-drag slider starts above 0 and `set()` accepts what Python accepts); E2 "optionally compute the stripping preload from the engagement achieved".

## Global Constraints

Every task's requirements implicitly include this section.

- New crate `magcoupling-rs/` is a sibling of `linkage-sim-rs/`: "No Cargo workspace conversion."
- Engine is "pure functions and data, no GUI dependencies, one Rust module per Python module (`model`, `calibration`, `metal_design`, `materials`, `temperature`, `clamps`, `sweeps`, `library`, `fields3d`, `api`)". `fields3d` is M3, not this plan.
- "One pure entry point `compute_all(&DesignInputs) -> Results`: no I/O, no global state, milliseconds per call so the GUI recomputes every frame during a drag."
- "Text verdicts are reproduced character for character (they drive status badges)."
- "Selector integer codes match the workbook (`backiron`: 1 steel, 0 none, etc.)."
- "Units as the package: mm, N·m, °C, T, kA/m, MPa, W, J, rpm, g."
- Inputs: "every field carries path, label, unit, help, workbook cell, default, kind, and choices (selectors), plus a slider range (min, max, step, logarithmic flag) defined by hand from physical bounds — the Python has no ranges."
- "Result schema mirrors `result_schema()`: every value with label, unit, cell."
- Parity: "Every result field with a cell reference, every sweep cell (494), every screw-table cell (165) and every default input (160) must match: numbers to 1e-9 relative (1e-12 absolute), text exactly — the same rule as `test_parity.py`." Full-port totals: 330 result cells + 494 + 165 + 160 = 1,149 checks.
- Differential: seeded random input sets "spanning each slider range and every selector branch ... the Rust port must match on all of them."
- Deviation registry: "cells, workbook value at defaults, corrected formula, evidence link (the M1 report entry). A test-only switch disables all deviations ... with deviations on, only registered fields may differ, and a test asserts each registered field takes its corrected value. The switch is not exposed to users."
- "More harmonics than the workbook's (1, 3, 5)" are out of scope; the default stays 1, 3, 5.
- `reference/magcoupling-py/` is the vendored oracle: its engine (`magcoupling/`) is never edited; its `pytest` stays green. Tools we own live in `reference/magcoupling-py/tools/`. Its `README.md` takes exactly one approved sentence (E14, Task 27); nothing else outside `tools/` changes.
- The library is pure std with no `[dependencies]`; `cargo check --target wasm32-unknown-unknown --lib` stays clean. The `workbook-parity` feature is reachable only through the self dev-dependency (never enable it elsewhere).
- Porting rules: architecture.md sections 7 and 8 (operand order never changed; `py_min`/`py_max`; `fmt_fixed`; `py_repr`; `fmt_num`; `text0`; catch-all `else` kept; Python names verbatim; `#[allow(non_snake_case)]` where Python uses capitals). One addition this plan makes binding: **never do `+`, `-` or `*` on an `i64` that came from an input or a derived count** (Python ints cannot overflow; Rust debug builds panic). Convert to `f64` (`n as f64`) for arithmetic, use saturating casts (`x as i64`) and `saturating_add` where Python adds to an int. Integer comparisons against selector codes stay integer.
- `bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh` must end with `GATE PASS` before every commit. Every gate run is prefixed with `MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe`: without it, gate 7 prints `SKIP gate 7/7` and still ends `GATE PASS` when it finds no oracle Python (the only venv lives in the `linkage_simulation-m1` worktree, which may be removed now that `magcoupling/m1` is merged); with it, a missing interpreter fails the gate. Every gate step's Expected therefore names the line before `GATE PASS` (`differential data is current (...)`) and a `SKIP gate 7/7` line is a failure. Gate 5 runs `cargo clippy --all-targets -- -D warnings` on `magcoupling-rs` (D5). Running the gate rewrites `docs/chebyshev_lambda/*.png` (known quirk of the `linkage-sim-rs` tests): restore them with `git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda` and never commit them.
- Run `cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml` before every commit (it breaks lines, never reorders operands).
- Every commit message ends with exactly these two lines:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
  ```
- Every code change updates its docs in the same commit (repo rule): at least the status line of `magcoupling-rs/README.md` and the `ported`/`remaining` lists of `docs/ai/03-structure.yaml`; Task 28 does the full pass. Read `docs/ai/*.yaml` before starting (Task 0).
- Physics changes (Tasks 14 to 26) additionally get a dedicated physics reviewer whose prompt names the audit report row (spec, "Testing and validation summary").
- Model tiers (CLAUDE.md section 5): each task's `**Model:**` line names its implementer model. Tasks 0 to 13, 27 and 28 give exact code or transcription rules and run on `sonnet` (set `model: 'sonnet'` explicitly when dispatching); Tasks 14 to 26 are `risk: physics` and run on the session model (omit `model` on purpose and say why in a comment), as do their physics reviews. A `sonnet` attempt that ends `blocked`, leaves the gate red or is rejected in review is retried and reworked on the session model; so is a task that turns out mid-way to need judgment. Final whole-branch review: session model.
- Equality edges (architecture.md section 7 step 8, binding): every module with a threshold comparison gets one unit test that puts each comparison at exact equality and asserts the branch Python takes there (Tasks 3, 6, 7, 8, 10, 11). Each test first asserts the equality itself, so a drifted input fails loudly instead of testing the wrong side.
- Commit subjects: `feat(magcoupling-rs): port <module>`, `test(magcoupling): ...` for harness and data, `fix(magcoupling-rs): apply E<k> <short title>` for corrections, `docs(magcoupling): ...` for docs.

## Review Focus

Five input classes the spec implies but no parity or differential case exercises (the differential data only ever holds values inside slider ranges and valid codes). Each line names the test that pins it and the task that owns it.

1. **A selector code outside its choices set directly on the struct** (a struct literal, a stale design file), e.g. `temperature.adhesive.selected = 0`: Python silently takes the LAST adhesive (`candidates[-1]`, DP460). Expected: no panic, never another adhesive; `"#N/A"` name and NaN numbers, and `validate()` names the path. Tests: `an_invalid_adhesive_code_selects_no_adhesive` (Task 8), `an_invalid_screw_class_is_nan_not_a_panic` (Task 10), `compute_all_never_panics_on_selector_codes_outside_the_choices` (Task 12).
2. **Magnet part text that almost matches the library** (`"b842sh"`, `"B842SH "`, `""`). Expected: exact-text lookup like the workbook's INDEX/MATCH, so manual dimensions, `inner_tmax_C = "n/a"`, temperature check `"unknown"`, and the measured calibration factor only for exactly `"B842SH"`. Test: `part_lookup_is_exact_text` (Task 3).
3. **Integer inputs typed far outside their slider** (`coupling.npole` = 0, 2, 3, `i64::MAX`; `clamps.joint_screws = i64::MAX`; `calibration.total_magnets = 0`). Expected: no overflow panic; results may be inf or NaN. Test: `compute_all_never_panics_on_extreme_inputs` (Task 12).
4. **Measured drag typed as exactly 0** (the GUI passes through 0 while the user types 0.05; Python raises ZeroDivisionError and aborts every sheet). Expected: no panic with corrections off or on; Temperature design C156 and C157 are +inf, every other result finite. A typed −0.0 gives −inf with the corrections off (IEEE sign of zero); E13 makes it +inf as well. Tests: `zero_measured_drag_does_not_panic` (Task 8), the two E13 probes, +0.0 and −0.0 (Task 26).
5. **Dimensions typed at or below zero, or far beyond physical sense** (bond line 0 divides the Volkersen screen by zero; a boss smaller than the bore; a magnet shorter than `c_end · pole pitch` gives a negative end factor, audit M9, not corrected). Expected: no panic, a clamp recommendation of `"None: enlarge the boss or the clamp length"` where nothing fits. Test: `compute_all_never_panics_on_extreme_inputs` (Task 12), which sets every numeric input to 0, −1, min/10, 10·max, ±1e300 in turn.

## File Structure

Created (all under `C:/Users/Cole/source/repos/lsim-mag-m2/`):

| File | Responsibility | Task |
|---|---|---|
| `magcoupling-rs/src/engine/library.rs` | Magnet library rows (static data) and exact-text lookup | 1 |
| `magcoupling-rs/src/engine/model.rs` | Calculator sheet: coupling and magnet inputs, `ModelResults`, harmonics, `MassResults` | 3, 5 |
| `magcoupling-rs/src/engine/metal_design.rs` | Metal design sheet: inputs, retainers, `MetalDesignResults`, `VALIDATION_ITEMS` | 3, 4, 6 |
| `magcoupling-rs/src/engine/materials.rs` | Materials sheet: steel, nickel, screw classes, aluminium alloys, `MaterialsResults` | 3, 7 |
| `magcoupling-rs/src/engine/temperature.rs` | Temperature design sheet: 7 input groups, 10 result groups, links, adhesives | 8 |
| `magcoupling-rs/src/engine/clamps.rs` | Shaft clamps and Clamp screw sizes sheets | 10 |
| `magcoupling-rs/src/engine/sweeps.rs` | Gap sweep and Pole sweep sheets | 11 |
| `magcoupling-rs/tests/static_data.rs` | Static tables equal the Python engine's | 1 |
| `magcoupling-rs/tests/robustness.rs` | Review Focus inputs: out-of-choice codes, zero drag, extreme values; never panics | 8 (Tasks 10, 12 add tests) |
| `magcoupling-rs/tests/data/static_data.json` | Python static tables (generated) | 1 |
| `magcoupling-rs/tests/data/differential/<group>.json`, `full.json` | Python results per result group (generated) | 3 to 13 |
| `magcoupling-rs/tests/data/deviations/E3.json`, `E4.json`, `E5.json` | Golden cell changes of broad corrections (blessed) | 16 to 18 |

Modified: `src/engine/{mod,meta,api,deviations,calibration}.rs`, `src/lib.rs`, `tests/{common/mod,parity,differential,python_schema,schema,deviations}.rs`, `tests/data/{input_schema,python_schema}.json`, `reference/magcoupling-py/tools/gen_differential.py`, `reference/magcoupling-py/README.md` (one sentence, E14, Task 27), `magcoupling-rs/README.md`, `docs/ai/{02-system,03-structure,04-memory}.yaml`, `docs/ai/05-update-tracker.md`.

Python module -> owner (every module of `reference/magcoupling-py/magcoupling/` has exactly one owner; a module split across tasks is split by function, each function owned once):

| Python module | Rust owner | Task |
|---|---|---|
| `library.py` | `library.rs` | 1 |
| `model.py` | `model.rs`: inputs, `ModelResults`, helpers, `compute` (3); `MassResults`, `mass_estimate` (5) | 3, 5 |
| `metal_design.py` | `metal_design.rs`: `MetalDesignInputs` (3); `RetainerResults`, `retainers` (4); `MetalDesignResults`, `compute`, `VALIDATION_ITEMS` (6) | 3, 4, 6 |
| `materials.py` | `materials.rs`: input structs (3); alloys, `ScrewClasses::proof`, `MaterialsResults`, `compute` (7) | 3, 7 |
| `temperature.py` | `temperature.rs` | 8 |
| `clamps.py` | `clamps.rs` | 10 |
| `sweeps.py` | `sweeps.rs` | 11 |
| `api.py` | `api.rs`: `DesignInputs`/`DesignResults` and `compute_all` grow with each port (3 to 11); `headline`, `validate` (12). `input_schema`/`result_schema`/`set_input` are the slice's `input_rows`/`result_rows`/`InputSet::set`. `to_dict` is not ported: JSON export is M4 | 3 to 12 |
| `_fields.py`, `constants.py`, `calibration.py` | `meta.rs`, `compat.rs`, `constants.rs`, `calibration.rs` (the slice; Task 0 verifies) | slice |
| `__init__.py` | `lib.rs` re-exports | 12 |
| `fields3d.py` | `fields3d.rs` | M3 (out of scope) |
| `drawing.py` | the clamp drawing as an egui painter port | M4 (out of scope) |
| `__main__.py` | none: the Python CLI has no Rust counterpart | out of scope |

Snapshot cells, each owned once: result cells calibration 23 (slice), model 73 (3), mass 6 (5), retainers 9 (4), metal 49 (6), materials 7 (7), temperature 130 (8), clamps 33 (10) = 330; default inputs 98 (3) + 37 (8) + 25 (10) = 160; screw table 165 (10); sweeps 338 + 156 (11). Uncelled results (differential only): `temperature.adhesive.selected_name` (8), `clamps.table[*].size` (10); Rust-only: `clamps.length_note` (15).

Order note: the computed task list put `library` near the end, but `model.resolve_magnets` calls `library.lookup`, so dependency order puts it first (Task 1). Two infrastructure tasks sit where they are first needed: Task 2 (multi-group modules in the harness, before `model`) and Task 9 (table rows in the metadata model, before `clamps`).

---

## Engine port

### Task 0: The slice exists

**Model:** `sonnet` for Steps 1-5 (running commands and reporting output); Step 6 is the controller's.

Verification only; nothing to write.

**Files:** none.

**Interfaces:**
- Consumes: branch `magcoupling/m2` at `86674a6` (or later commits of this plan).
- Produces: a confirmed green baseline and the D1 to D7 answers.

- [ ] **Step 1: Read the project context.** Read `C:/Users/Cole/source/repos/lsim-mag-m2/docs/ai/02-system.yaml`, `03-structure.yaml`, `04-memory.yaml`, the architecture document (path in the header), `C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/README.md`, and the spec's M2 section.

- [ ] **Step 2: Check the branch and a clean tree**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 rev-parse --abbrev-ref HEAD
git -C C:/Users/Cole/source/repos/lsim-mag-m2 log --oneline -1
git -C C:/Users/Cole/source/repos/lsim-mag-m2 status --short
```
Expected: `magcoupling/m2`; `86674a6 docs(magcoupling): M2 slice guide and project context`; no status lines.

- [ ] **Step 3: Run the crate's tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "Running|test result"
```
Expected, in order: unit tests `35 passed`; `tests\deviations.rs` 6 passed; `tests\differential.rs` 3 passed; `tests\parity.rs` 3 passed; `tests\python_schema.rs` 3 passed; `tests\schema.rs` 6 passed; doc-tests `1 passed; 0 failed; 1 ignored`.

- [ ] **Step 4: Check the oracle Python and the data freshness**

```bash
ls C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py --check
```
Expected: `ls` prints the path; then `differential data is current (calibration + helpers + python schema)`. If `ls` fails (the `linkage_simulation-m1` worktree was removed after the M1 merge), stop and escalate: every Python command and every gate run of this plan uses that interpreter, and recreating the venv (a network `pip install`) needs the user's go-ahead.

- [ ] **Step 5: Run the gate**

```bash
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task0.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task0.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last three lines are the oracle's `1159 passed ...`, `differential data is current (calibration + helpers + python schema)` and `GATE PASS`; no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 6: Decisions.** The controller asks the user D1 to D7 (table above) in one message and records the answers in this plan's execution notes. Unanswered decisions default to the recommended option up to the first task that needs them; that task waits.

---

### Task 1: Magnet library (`library.py`)

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Create: `magcoupling-rs/src/engine/library.rs`
- Create: `magcoupling-rs/tests/static_data.rs`
- Create (generated): `magcoupling-rs/tests/data/static_data.json`
- Modify: `magcoupling-rs/src/engine/mod.rs` (add `pub mod library;`)
- Modify: `reference/magcoupling-py/tools/gen_differential.py` (new output `static_data.json`)
- Modify: `magcoupling-rs/README.md` (status line, layout row), `docs/ai/03-structure.yaml` (`ported`, `remaining`)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `pub struct MagnetSpec { pub part: &'static str, pub vendor: &'static str, pub shape: &'static str, pub length_mm: f64, pub width_mm: f64, pub thickness_mm: f64, pub grade: &'static str, pub br_T: f64, pub tmax_C: f64, pub notes: &'static str }`
  - `pub const MAGNET_LIBRARY: [MagnetSpec; 15]` (workbook values, Python row order)
  - `pub fn lookup(part: &str) -> Option<&'static MagnetSpec>` (exact text; `""` gives `None`)
  - `tests/static_data.rs` with a helper `fn python_static() -> serde_json::Value` reading `tests/data/static_data.json`; later tasks add keys and tests to this file.
  - Generator: `static_data()` writing `static_data.json` with key `"magnet_library"`.

The "Magnet library" sheet is not in the workbook snapshot, so this module has no parity cells. Its parity check is `tests/static_data.rs` against the Python rows.

- [ ] **Step 1: Write the failing static-data test**

Create `magcoupling-rs/tests/static_data.rs`:

```rust
//! Static engine tables (magnet library, and later adhesives, alloys, screw
//! sizes, checklists) equal the Python engine's, value for value.
//! `tests/data/static_data.json` is written by
//! `reference/magcoupling-py/tools/gen_differential.py`.

mod common;

use common::{data_path, read_json};
use magcoupling::engine::library::{MAGNET_LIBRARY, lookup};

fn python_static() -> serde_json::Value {
    read_json(&data_path("static_data.json"))
}

fn text(row: &serde_json::Value, key: &str) -> String {
    row[key].as_str().unwrap_or_else(|| panic!("{key} is text")).to_owned()
}

fn number(row: &serde_json::Value, key: &str) -> f64 {
    row[key].as_f64().unwrap_or_else(|| panic!("{key} is a number"))
}

#[test]
fn magnet_library_equals_the_python_rows() {
    let doc = python_static();
    let rows = doc["magnet_library"].as_array().expect("a magnet_library array");
    assert_eq!(rows.len(), MAGNET_LIBRARY.len(), "row count");
    for (py, rs) in rows.iter().zip(MAGNET_LIBRARY.iter()) {
        let p = text(py, "part");
        assert_eq!(rs.part, p);
        assert_eq!(rs.vendor, text(py, "vendor"), "{p}");
        assert_eq!(rs.shape, text(py, "shape"), "{p}");
        assert_eq!(rs.grade, text(py, "grade"), "{p}");
        assert_eq!(rs.notes, text(py, "notes"), "{p}");
        for (key, value) in [
            ("length_mm", rs.length_mm),
            ("width_mm", rs.width_mm),
            ("thickness_mm", rs.thickness_mm),
            ("br_T", rs.br_T),
            ("tmax_C", rs.tmax_C),
        ] {
            // Static data: no arithmetic, so equality is exact.
            assert_eq!(value, number(py, key), "{p}.{key}");
        }
    }
}

#[test]
fn every_part_resolves_to_its_own_row_by_exact_text() {
    // By value: a const table has no guaranteed address, and parts are unique.
    for spec in &MAGNET_LIBRARY {
        assert_eq!(lookup(spec.part), Some(spec), "{}", spec.part);
    }
    let parts: std::collections::BTreeSet<&str> = MAGNET_LIBRARY.iter().map(|m| m.part).collect();
    assert_eq!(parts.len(), MAGNET_LIBRARY.len(), "part names are unique");
    for near_miss in ["", "b842sh", "B842SH ", " B842SH", "B842", "B842SH\n"] {
        let found = lookup(near_miss).map(|s| s.part);
        let expected = (near_miss == "B842").then_some("B842");
        assert_eq!(found, expected, "{near_miss:?}");
    }
}
```

- [ ] **Step 2: Run it to see it fail**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test static_data 2>&1 | tail -5
```
Expected: compile error `unresolved import magcoupling::engine::library`.

- [ ] **Step 3: Port `library.py`**

Python spec: `reference/magcoupling-py/magcoupling/library.py` lines 13-24 (`MagnetSpec`), 27-44 (`_ROWS`), 46 (`MAGNET_LIBRARY`), 49-53 (`lookup`).

Create `magcoupling-rs/src/engine/library.rs`:

```rust
//! Stock magnet library ('Magnet library' sheet).
//!
//! Port of `reference/magcoupling-py/magcoupling/library.py`. The calculator
//! looks magnets up by exact part text, like the workbook's INDEX/MATCH.
//! Dimensions in mm, Br is the 20 °C remanence in tesla, tmax the supplier
//! rating in °C. The rows hold the WORKBOOK values; the approved correction E3
//! (N42SH remanence) is applied where the Calculator resolves a part
//! (`model::resolve_magnets`), not here.

/// One library row (Python `MagnetSpec`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct MagnetSpec {
    pub part: &'static str,
    pub vendor: &'static str,
    pub shape: &'static str,
    /// Axial length [mm].
    pub length_mm: f64,
    /// Tangential width [mm].
    pub width_mm: f64,
    /// Radial thickness, the magnetized direction [mm].
    pub thickness_mm: f64,
    pub grade: &'static str,
    /// Remanence at 20 °C [T].
    pub br_T: f64,
    /// Supplier maximum operating temperature [°C].
    pub tmax_C: f64,
    pub notes: &'static str,
}

const fn row(
    part: &'static str,
    vendor: &'static str,
    shape: &'static str,
    dims: [f64; 3],
    grade: &'static str,
    br_T: f64,
    tmax_C: f64,
    notes: &'static str,
) -> MagnetSpec {
    MagnetSpec {
        part,
        vendor,
        shape,
        length_mm: dims[0],
        width_mm: dims[1],
        thickness_mm: dims[2],
        grade,
        br_T,
        tmax_C,
        notes,
    }
}

/// The library, in the Python row order (`_ROWS`).
pub const MAGNET_LIBRARY: [MagnetSpec; 15] = [
    row("B842SH", "K&J", "block", [12.7, 6.35, 3.17], "N42SH", 1.29, 150.0, "1/2 x 1/4 x 1/8 in, magnetized through 1/8 in"),
    row("B842", "K&J", "block", [12.7, 6.35, 3.17], "N42", 1.30, 80.0, ""),
    row("B842-N52", "K&J", "block", [12.7, 6.35, 3.17], "N52", 1.45, 80.0, ""),
    row("B822", "K&J", "block", [12.7, 3.17, 3.17], "N42", 1.30, 80.0, "1/2 x 1/8 x 1/8 in"),
    row("B862", "K&J", "block", [12.7, 9.5, 3.17], "N42", 1.30, 80.0, "1/2 x 3/8 x 1/8 in"),
    row("B882", "K&J", "block", [12.7, 12.7, 3.17], "N42", 1.30, 80.0, "1/2 x 1/2 x 1/8 in"),
    row("B882-N52", "K&J", "block", [12.7, 12.7, 3.17], "N52", 1.45, 80.0, ""),
    row("B861", "K&J", "block", [12.7, 9.5, 1.59], "N42", 1.30, 80.0, "1/2 x 3/8 x 1/16 in"),
    row("B881", "K&J", "block", [12.7, 12.7, 1.59], "N42", 1.30, 80.0, "1/2 x 1/2 x 1/16 in (check stock)"),
    row("B442", "K&J", "block", [6.35, 6.35, 3.17], "N42", 1.30, 80.0, "1/4 x 1/4 x 1/8 in"),
    row("BX042SH", "K&J", "block", [25.4, 6.35, 3.17], "N42SH", 1.29, 150.0, "1 x 1/4 x 1/8 in"),
    row("BX082SH", "K&J", "block", [25.4, 12.7, 3.17], "N42SH", 1.29, 150.0, "1 x 1/2 x 1/8 in"),
    row("M5044", "SuperMagnetMan", "arc", [13.0, 5.5, 1.31], "N50", 1.42, 80.0,
        "22.60 OD x 19.97 ID x 13, 29.8 deg, 12 pcs; width = mean arc length"),
    row("M5045", "SuperMagnetMan", "arc", [6.56, 5.6, 1.12], "N50M", 1.42, 100.0, "22.70 OD x 20.47 ID x 6.56, 12 pcs"),
    row("M5026", "SuperMagnetMan", "arc", [15.0, 6.4, 1.67], "N50", 1.42, 80.0, "26.60 OD x 23.26 ID x 15, 12 pcs"),
];

/// Exact-text lookup (Python `lookup`): `None` for an empty or unknown part,
/// and the model then uses the manual dimensions.
pub fn lookup(part: &str) -> Option<&'static MagnetSpec> {
    if part.is_empty() {
        return None;
    }
    MAGNET_LIBRARY.iter().find(|m| m.part == part)
}
```

`row` has 8 parameters, two with Python's capitals: put `#[allow(clippy::too_many_arguments, non_snake_case)]` above it. Add `pub mod library;` to `src/engine/mod.rs` (alphabetical, after `deviations`).

- [ ] **Step 4: Export the Python rows from the generator**

In `reference/magcoupling-py/tools/gen_differential.py`: import `from magcoupling.library import _ROWS` next to the other imports; add below the helpers corpus section:

```python
# --------------------------------------------------------------------------- static data
def static_data() -> str:
    """Static engine tables, for magcoupling-rs/tests/static_data.rs."""
    doc = {
        "about": "Static tables of the magcoupling engine (plain data, no metadata). Written by "
                 "reference/magcoupling-py/tools/gen_differential.py. Do not edit by hand.",
        "engine": f"magcoupling {magcoupling.__version__}",
        "magnet_library": [dataclasses.asdict(m) for m in _ROWS],
    }
    return dumps(doc) + "\n"
```

Add `import dataclasses` at the top, add `DATA / "static_data.json": static_data(),` to the `files` dict in `outputs()`, list the file in the module docstring's "Writes:" block, and change the `--check` success message to `f"differential data is current ({', '.join(MODULES)} + helpers + python schema + static data)"`.

Regenerate:

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote magcoupling-rs/tests/data/static_data.json (...)` among the lines; `git -C C:/Users/Cole/source/repos/lsim-mag-m2 status --short` shows only `static_data.json` as new data (the other generated files are unchanged).

- [ ] **Step 5: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test static_data 2>&1 | tail -4
```
Expected: `test result: ok. 2 passed; 0 failed`.

- [ ] **Step 6: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task1.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task1.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 7: Docs and commit.** README status line: add "magnet library"; layout table row for `src/engine/<module>.rs` lists `constants`, `calibration`, `library`; tests table gains `tests/static_data.rs` ("static tables equal the Python engine's"). `docs/ai/03-structure.yaml`: `ported: [constants.rs, calibration.rs, library.rs]`, remove `library` from `remaining`.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src/engine/library.rs magcoupling-rs/src/engine/mod.rs magcoupling-rs/tests/static_data.rs magcoupling-rs/tests/data/static_data.json reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the magnet library

Static rows and exact-text lookup (library.py), checked value for value
against the Python rows through the new static_data.json export.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 2: Harness for multi-group modules

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

From `model` on, a result group depends on several input groups, some input groups have no result group (`coupling`), and `metal` names both an input group and a result group that land in different tasks. This task reshapes the harness once, with `calibration` as the only module, so every later port only adds lines.

**Files:**
- Modify: `magcoupling-rs/tests/common/mod.rs`
- Modify: `magcoupling-rs/tests/parity.rs`, `tests/python_schema.rs`, `tests/differential.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: `magcoupling-rs/tests/data/differential/calibration.json` (new file format)

**Interfaces:**
- Consumes: the slice's harness.
- Produces (used by every later port):
  - `tests/common/mod.rs`: `pub struct PortedInputs { pub group: &'static str, pub cells: usize }`, `pub struct PortedResults { pub group: &'static str, pub cells: usize }`, `pub const PORTED_INPUTS: &[PortedInputs]`, `pub const PORTED_RESULTS: &[PortedResults]`, `pub fn is_ported_input(path: &str) -> bool`, `pub fn is_ported_result(path: &str) -> bool` (replacing `Ported`, `PORTED`, `is_ported`).
  - Generator: `MODULES: dict[str, list[str]]` (result group to the input groups each case varies), `TEXT_CHOICES: dict[str, list[str]]`, `PROBES` keyed by result group, `RANDOM_MIN = 100`, `MAX_FILE_BYTES = 4_000_000`, `group_of(path)`.
  - Data format of `differential/<group>.json`: header keys `module`, `seed`, `input_groups`, `input_paths`, `result_paths`; each case `{"id", "tag", "inputs": [values in input_paths order], "results": [values in result_paths order]}`.
  - `tests/differential.rs`: `fn load_cases(module: &str) -> Vec<Case>`, `fn check_module(module: &str) -> Vec<Case>`, `enum Reach { Text(&'static str), Prefix(&'static str), Number }`, `const BRANCHES: &[(&str, &[Reach])]` (patterns may use `[*]` for a table index), tests `every_branch_is_reached` and `every_varied_input_takes_two_values`.

- [ ] **Step 1: Split the ratchet (tests fail to compile until Step 2)**

Replace `Ported`, `PORTED` and `is_ported` in `magcoupling-rs/tests/common/mod.rs` with:

```rust
/// A ported input group of `DesignInputs` and its workbook input cells
/// (inputs whose default is `None` are not counted, as in `test_parity.py`).
pub struct PortedInputs {
    pub group: &'static str,
    pub cells: usize,
}

/// A ported result group of `DesignResults` and its workbook result cells.
pub struct PortedResults {
    pub group: &'static str,
    pub cells: usize,
}

/// Ported input groups, in Python `DesignInputs` order. The counts are a
/// ratchet against silently dropped fields; `tests/python_schema.rs` checks
/// them against the Python engine's own schema.
pub const PORTED_INPUTS: &[PortedInputs] = &[PortedInputs {
    group: "calibration",
    cells: 16,
}];

/// Ported result groups, in Python `DesignResults` order.
pub const PORTED_RESULTS: &[PortedResults] = &[PortedResults {
    group: "calibration",
    cells: 23,
}];

/// Whether an input path belongs to a ported input group.
pub fn is_ported_input(path: &str) -> bool {
    PORTED_INPUTS.iter().any(|p| p.group == group_of(path))
}

/// Whether a result path belongs to a ported result group.
pub fn is_ported_result(path: &str) -> bool {
    PORTED_RESULTS.iter().any(|p| p.group == group_of(path))
}
```

- [ ] **Step 2: Use the split ratchet in the tests**

`tests/parity.rs`: build the two `expected` maps from `PORTED_RESULTS` (`p.cells`) and `PORTED_INPUTS` (`p.cells`) instead of `PORTED`; import `{PORTED_INPUTS, PORTED_RESULTS, ...}`.

`tests/python_schema.rs`, replace the body of `every_python_field_of_a_ported_group_is_ported` with:

```rust
    let python = python_rows();
    let inputs = input_rows(&DesignInputs::default());
    let results = result_rows(&compute_all(&DesignInputs::default()));
    let rust_paths: BTreeSet<&str> = inputs
        .iter()
        .map(|r| r.path.as_str())
        .chain(results.iter().map(|r| r.path.as_str()))
        .collect();
    let missing: Vec<&String> = python
        .iter()
        .filter(|(path, row)| match row.kind.as_str() {
            "input" => is_ported_input(path),
            "result" => is_ported_result(path),
            _ => false,
        })
        .filter(|(path, _)| !rust_paths.contains(path.as_str()))
        .map(|(path, _)| path)
        .collect();
    assert!(missing.is_empty(), "Python fields not ported: {missing:?}");

    // The ratchet counts equal the Python engine's, counted as test_parity.py
    // does (cells only; inputs whose default is None skipped).
    let count = |group: &str, kind: &str| {
        python
            .iter()
            .filter(|(path, row)| {
                common::group_of(path) == group
                    && row.kind == kind
                    && row.cell.is_some()
                    && row.default != Some(serde_json::Value::Null)
            })
            .count()
    };
    for p in PORTED_INPUTS {
        assert_eq!(count(p.group, "input"), p.cells, "{}: input cells", p.group);
    }
    for p in PORTED_RESULTS {
        assert_eq!(count(p.group, "result"), p.cells, "{}: result cells", p.group);
    }
```
(add `use std::collections::BTreeSet;`, import `is_ported_input`, `is_ported_result`, `PORTED_INPUTS`, `PORTED_RESULTS`).

`tests/differential.rs`: `every_ported_module_has_differential_data` iterates `PORTED_RESULTS`.

- [ ] **Step 3: Reshape the generator**

In `reference/magcoupling-py/tools/gen_differential.py`:

1. `import re`; replace the `CASES_PER_MODULE`/`MODULES` block with:

```python
CASES_PER_MODULE = 300
RANDOM_MIN = 100            # at least this many random cases, however many fixed cases a module has
MAX_FILE_BYTES = 4_000_000  # a data file above this means: vary fewer groups or cut cases
# Ported result groups -> the input groups each case varies (every group the
# results read, directly or through upstream sheets). Add a line when a
# module's Rust port lands, and list its groups in magcoupling-rs/tests/common/mod.rs.
MODULES = {
    "calibration": ["calibration"],
}
# Text inputs have no slider: the values each case may take (sampled like a selector).
TEXT_CHOICES: dict[str, list[str]] = {}


def group_of(path: str) -> str:
    """Top-level group of a dotted path; table rows ('gap_sweep[3].x') count as their table."""
    return re.split(r"[.\[]", path, maxsplit=1)[0]
```

2. `sample()`: before `rng_ = field["range"]`, add
```python
    if field["path"] in TEXT_CHOICES:
        return rng.choice(TEXT_CHOICES[field["path"]])
```

3. `module_cases()`: after the choices/range loop body, add a branch `elif f["path"] in TEXT_CHOICES:` that appends one case per text (`f"{f['path']} = {text!r}"`); change the probe check message to `f"probe {tag!r} names inputs outside {module}'s groups: {sorted(unknown)}"`; replace the random loop with
```python
    for _ in range(max(CASES_PER_MODULE - len(cases), RANDOM_MIN)):
        cases.append(("random", {f["path"]: sample(f, rng) for f in fields}))
```
and delete the `len(cases) > CASES_PER_MODULE` error (the fixed cases now always come first, then at least `RANDOM_MIN` random ones).

4. `run_case()`: collect table rows too (`kind` is `""` for them) and filter by group:
```python
    return {r["path"]: plain(r["value"], f"{tag}: {r['path']}") for r in result_schema(res)
            if r["kind"] in ("result", "") and group_of(r["path"]) == module}
```

5. `module_file(module, groups, schema)` writes the columnar format:
```python
def module_file(module: str, groups: list, schema: list) -> str:
    fields = [f for f in schema if group_of(f["path"]) in groups]
    missing = sorted(set(groups) - {group_of(f["path"]) for f in fields})
    if missing:
        raise KeyError(f"input_schema.json has no inputs under {missing}")
    input_paths = [f["path"] for f in fields]
    rng = random.Random(f"{SEED}:{module}")
    result_paths, cases = None, []
    for i, (tag, inputs) in enumerate(module_cases(module, fields, rng)):
        results = run_case(module, tag, inputs)
        if result_paths is None:
            result_paths = list(results)
        elif list(results) != result_paths:
            raise ValueError(f"{module} case {tag!r}: result paths differ from case 0")
        cases.append({"id": i, "tag": tag,
                      "inputs": [plain(inputs[p], f"{tag}: {p}") for p in input_paths],
                      "results": list(results.values())})
    header = {"about": f"Python engine results for seeded inputs, result group {module}; compared by "
                       "magcoupling-rs/tests/differential.rs. Paths are listed once; each case holds "
                       "values in that order. Written by reference/magcoupling-py/tools/"
                       "gen_differential.py. Do not edit by hand.",
              "engine": f"magcoupling {magcoupling.__version__}", "module": module, "seed": SEED,
              "input_groups": groups, "input_paths": input_paths, "result_paths": result_paths}
    head = dumps(header, compact=True)
    body = ",\n".join(dumps(c, compact=True) for c in cases)
    text = head[:-1] + ',"cases":[\n' + body + "\n]}\n"
    if len(text.encode("utf-8")) > MAX_FILE_BYTES:
        raise ValueError(f"differential/{module}.json is {len(text.encode('utf-8'))} bytes, over {MAX_FILE_BYTES}")
    return text
```

6. `outputs()`: `for module, groups in MODULES.items(): files[...] = module_file(module, groups, schema)`.

Regenerate:
```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote magcoupling-rs/tests/data/differential/calibration.json (302 lines)` (37 fixed cases + 263 random = 300 cases, one per line, plus the header and closing lines; the file shrinks from about 528 KB because paths are no longer repeated per case), and `python_schema.json`, `helpers.json`, `static_data.json` rewritten byte-identical (`git status` lists only `calibration.json` among the data files).

- [ ] **Step 4: Read the new format in `tests/differential.rs`**

Replace `scalar_map` and `load_cases` with:

```rust
fn load_cases(module: &str) -> Vec<Case> {
    let doc = read_json(&data_path(&format!("differential/{module}.json")));
    assert_eq!(doc["module"], module);
    let paths = |key: &str| -> Vec<String> {
        doc[key]
            .as_array()
            .unwrap_or_else(|| panic!("{module}: {key} is an array"))
            .iter()
            .map(|p| p.as_str().expect("a path").to_owned())
            .collect()
    };
    let (input_paths, result_paths) = (paths("input_paths"), paths("result_paths"));
    doc["cases"]
        .as_array()
        .expect("a cases array")
        .iter()
        .map(|c| {
            let id = c["id"].as_u64().expect("a case id");
            let zip = |paths: &[String], key: &str| -> BTreeMap<String, Value> {
                let values = c[key].as_array().unwrap_or_else(|| panic!("case {id}: {key}"));
                assert_eq!(values.len(), paths.len(), "{module} case {id}: {key} length");
                paths.iter().cloned().zip(values.iter().map(json_to_value)).collect()
            };
            Case {
                id,
                tag: c["tag"].as_str().expect("a case tag").to_owned(),
                inputs: zip(&input_paths, "inputs"),
                results: zip(&result_paths, "results"),
            }
        })
        .collect()
}
```

Add the branch coverage table and the two generic tests (append below `check_module`):

```rust
/// What a text-producing result must reach across its module's cases.
#[derive(Clone, Copy, Debug)]
enum Reach {
    /// Exactly this text.
    Text(&'static str),
    /// A text starting with this.
    Prefix(&'static str),
    /// A number (the numeric side of a number-or-text result).
    Number,
}
use Reach::{Number, Prefix, Text};

/// Every branch of every ported module's text results, so each branch of the
/// Python source is compared at least once. `[*]` matches any table row.
/// Add the rows of a module when it lands.
const BRANCHES: &[(&str, &[Reach])] = &[
    ("calibration.fea_interp_Nm", &[Number, Text("outside range")]),
    ("calibration.fea_interp_error", &[Number, Text("n.a.")]),
];

fn reached(value: &Value, reach: Reach) -> bool {
    match (reach, value) {
        (Text(t), Value::Text(v)) => v == t,
        (Prefix(p), Value::Text(v)) => v.starts_with(p),
        (Number, Value::Num(_) | Value::Int(_)) => true,
        _ => false,
    }
}

/// `gap_sweep[*].status` matches `gap_sweep[3].status`; other patterns match exactly.
fn matches_pattern(pattern: &str, path: &str) -> bool {
    match pattern.split_once("[*]") {
        None => pattern == path,
        Some((head, tail)) => path
            .strip_prefix(head)
            .and_then(|rest| rest.strip_prefix('['))
            .and_then(|rest| rest.split_once(']'))
            .is_some_and(|(index, rest)| {
                !index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) && rest == tail
            }),
    }
}

#[test]
fn every_branch_is_reached() {
    let mut cache: BTreeMap<&str, Vec<Case>> = BTreeMap::new();
    let mut failures = Vec::new();
    for &(pattern, reaches) in BRANCHES {
        let module = group_of(pattern);
        let cases = cache.entry(module).or_insert_with(|| load_cases(module));
        let values: Vec<&Value> = cases
            .iter()
            .flat_map(|c| c.results.iter())
            .filter(|(path, _)| matches_pattern(pattern, path))
            .map(|(_, v)| v)
            .collect();
        if values.is_empty() {
            failures.push(format!("{pattern}: no such result in differential/{module}.json"));
            continue;
        }
        for &reach in reaches {
            if !values.iter().any(|v| reached(v, reach)) {
                failures.push(format!("{pattern}: never reaches {reach:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_varied_input_takes_two_values() {
    let mut failures = Vec::new();
    for p in PORTED_RESULTS {
        let cases = load_cases(p.group);
        let paths: BTreeSet<&String> = cases.iter().flat_map(|c| c.inputs.keys()).collect();
        for path in paths {
            let distinct: BTreeSet<String> =
                cases.iter().map(|c| format!("{:?}", c.inputs[path])).collect();
            if distinct.len() < 2 {
                failures.push(format!("{}: {path} never varies", p.group));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn branch_patterns_match_table_rows_only_by_index() {
    assert!(matches_pattern("gap_sweep[*].status", "gap_sweep[12].status"));
    assert!(!matches_pattern("gap_sweep[*].status", "gap_sweep[].status"));
    assert!(!matches_pattern("gap_sweep[*].status", "gap_sweep[1].status_x"));
    assert!(!matches_pattern("gap_sweep[*].status", "pole_sweep[1].status"));
    assert!(matches_pattern("model.verdict", "model.verdict"));
}
```

In `calibration_matches_python_on_every_case`, delete the final "every input actually varied" loop (now `every_varied_input_takes_two_values`); keep the gap-definition, span-side and inclusive-end assertions.

- [ ] **Step 5: Run the harness**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity --test python_schema --test differential 2>&1 | grep -E "^test |test result"
```
Expected: every listed test `ok`, including `every_branch_is_reached`, `every_varied_input_takes_two_values`, `branch_patterns_match_table_rows_only_by_index`, `calibration_matches_python_on_every_case`; three `test result: ok.` lines, 0 failed.

Mutation spot check: in `gen_differential.py` temporarily change `MODULES = {"calibration": ["calibration"]}` to `{"calibration": ["coupling"]}` and run the generator: expected `KeyError: "input_schema.json has no inputs under ['coupling']"`. Revert.

- [ ] **Step 6: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task2.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task2.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 7: Docs and commit.** README "Regenerating test data": describe `MODULES` as result group to input groups, the columnar data format, `RANDOM_MIN` and the 4 MB budget, and `BRANCHES`. `docs/ai/03-structure.yaml` `tests:` line: `common/mod.rs holds the PORTED_INPUTS / PORTED_RESULTS ratchets`.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/tests reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
test(magcoupling): harness for modules that read several input groups

Split the ratchet into input and result groups, let the generator vary
several input groups per result group, sample text inputs, keep a random
floor per module, write paths once per file (columnar cases, 4 MB budget),
and check branch coverage with one table-driven test.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---
### Task 3: Calculator model (`model.py`), plus the Metal design and Materials input structs

First line: depends on D7 (the fixed harmonic set [1, 3, 5]). If D7 is the alternative, stop and escalate.

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Create: `magcoupling-rs/src/engine/model.rs`
- Create: `magcoupling-rs/src/engine/metal_design.rs` (inputs only in this task)
- Create: `magcoupling-rs/src/engine/materials.rs` (inputs only in this task)
- Modify: `magcoupling-rs/src/engine/mod.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`, `tests/static_data.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: `tests/data/input_schema.json`, `tests/data/static_data.json`; new `tests/data/differential/model.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`

**Interfaces:**
- Consumes: `library::lookup(part: &str) -> Option<&'static MagnetSpec>` (Task 1); `calibration::compute`, `CalibrationResults { poles_per_ring, f_cal_updated, .. }`, `CalibrationInputs { alpha_br_per_C, f_cal_original, .. }` (slice); `compat::{py_min, py_max, fmt_fixed}`; `constants::MU0`.
- Produces:
  - `model.rs`: `pub const HARMONICS: [u32; 3] = [1, 3, 5];` `pub const NOT_IN_LIBRARY: &str = "n/a";` `inputs! MagnetInputs` (10 fields), `inputs! CouplingInputs` (14 fields, group `magnets: MagnetInputs`), `results! ModelResults` (73 fields), `pub struct ResolvedMagnet { pub length_mm: f64, pub width_mm: f64, pub thickness_mm: f64, pub br_T: f64, pub tmax_C: NumOrText }`, `pub fn resolve_magnets(m: &MagnetInputs, dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet)`, `pub fn select_calibration_factor(backiron: i64, npole: i64, part_i: &str, part_o: &str, prototype_poles: f64, f_cal_updated: f64, f_cal_original: f64) -> f64`, `pub fn geometry_factor(k: f64, t_i_mm: f64, t_o_mm: f64, g_mm: f64, backiron: i64) -> f64`, `pub fn harmonic_amplitude(br_T: f64, n: u32, fill: f64) -> f64`, `pub struct Harmonic { pub n: u32, pub k: f64, pub bi: f64, pub bo: f64, pub s_iron: f64, pub s_free: f64, pub tau: f64 }` with `impl Harmonic { pub fn s(&self, backiron: i64) -> f64 }` (the one place that picks `s_iron` or `s_free` by circuit; used by `shear_stress`, the sweeps and E7), `pub fn shear_stress(br_i: f64, br_o: f64, fill_i: f64, fill_o: f64, npole: i64, r_g_mm: f64, t_i_mm: f64, t_o_mm: f64, g_mm: f64, backiron: i64, mu0: f64) -> [Harmonic; 3]`, `pub fn compute(ci: &CouplingInputs, face_gap_mm: f64, bond_inner_mm: f64, bond_outer_mm: f64, cup_wall_corner_mm: f64, alpha_br: f64, bsat_T: f64, f_cal: f64, f_cal_original: f64, slip_rpm: f64, required_min_Nm: f64, dev: Deviations) -> ModelResults`.
  - `metal_design.rs`: `inputs! MetalDesignInputs` (48 fields; `measured_drag_Nm: Option<f64>`).
  - `materials.rs`: `inputs! Steel4140` (7), `inputs! ElectrolessNickel` (1), `inputs! ScrewClasses` (3), `inputs! MaterialsInputs { fields {} groups { steel: Steel4140, nickel: ElectrolessNickel, screws: ScrewClasses } }`.
  - `api.rs`: `DesignInputs` groups in Python order `coupling, metal, calibration, materials`; `DesignResults` groups `calibration, model`.

- [ ] **Step 1: Enable the cells (failing)**

In `magcoupling-rs/tests/common/mod.rs` set:

```rust
pub const PORTED_INPUTS: &[PortedInputs] = &[
    PortedInputs { group: "coupling", cells: 24 },
    PortedInputs { group: "metal", cells: 47 }, // metal.measured_drag_Nm defaults to None: not counted
    PortedInputs { group: "calibration", cells: 16 },
    PortedInputs { group: "materials", cells: 11 },
];

pub const PORTED_RESULTS: &[PortedResults] = &[
    PortedResults { group: "calibration", cells: 23 },
    PortedResults { group: "model", cells: 73 },
];
```

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "left|right|test result"
```
Expected: FAIL. `every_result_cell_matches_the_workbook` reports `left: {"calibration": 23}` / `right: {"calibration": 23, "model": 73}`, and `every_default_input_matches_the_workbook` reports `left: {"calibration": 16}` against the four input groups.

- [ ] **Step 2: Port**

Python spec, `reference/magcoupling-py/magcoupling/`: `metal_design.py` lines 25-80 (`MetalDesignInputs`); `materials.py` lines 18-33 (`Steel4140`, `ElectrolessNickel`), 60-64 (`ScrewClasses` fields; `proof()` comes in Task 7), 71-82 (`MaterialsInputs`); `model.py` lines 31 (`HARMONICS`), 35-71 (inputs), 75-155 (`ModelResults`), 170-226 (helpers), 230-312 (`compute`). The Python is the spec of the "how"; follow architecture.md section 8.

- [ ] **2a. Input structs.** Transcribe every Python `param(...)` byte for byte as `name: <type> = <default> => param(unit, label, help, cell)` (the metadata-parity test compares label, unit, help, cell, choices and default with Python). Types: Python `float` is `f64` with a float literal default (`op_temp_C: f64 = 50.0`); `int` is `i64`; `str` is `String`; `float | None` is `Option<f64>` with default `None`. Selectors get `.choices(...)` in the Python dict order: `backiron` `&[(1, "steel circuit"), (0, "no back iron")]`, `faceted` `&[(1, "flat blocks"), (0, "arcs")]`. `CouplingInputs` declares `groups { magnets: MagnetInputs }` after its fields (Python puts `magnets` last). `MaterialsInputs` has `fields {}` and `groups { steel: Steel4140, nickel: ElectrolessNickel, screws: ScrewClasses }`: the Python `aluminium` member has no metadata, so it is not an input (static data in Task 7). Example lines:

```rust
inputs! {
    /// Magnet parts and the manual fallbacks (Calculator!C11:C27).
    pub struct MagnetInputs {
        fields {
            part_inner: String = "B842SH" => param("-", "Inner magnet part",
                "Looked up in the library by exact text; blank = manual.", "Calculator!C11"),
            part_outer: String = "B842SH" => param("-", "Outer magnet part", "", "Calculator!C12"),
            manual_inner_length_mm: f64 = 12.7 => param("mm", "Manual inner length (axial)",
                "Used only if the part is not in the library.", "Calculator!C14")
                .range(2.0, 50.8, 0.01),
            // ... the other seven manual fields, same pattern (model.py lines 40-46)
        }
    }
}
```

Slider ranges (defined here once; the generator reads them from `input_schema.json`). A3 marks `.assumption()`. The constraints column says which Python raise point a bound keeps the engine away from.

| Path | Default | min | max | step | log | A3 | Constraint |
|---|---|---|---|---|---|---|---|
| `coupling.npole` (i64) | 10 | 4 | 40 | 2 | | | even; `tan(π/N)` blows up at 2 |
| `coupling.inner_back_apothem_mm` | 10.15 | 9.0 | 30.0 | 0.01 | | | hub steel `a_i − bond_inner − bore/2 > 0` for every bond and bore in range (Volkersen divides by it) |
| `coupling.op_temp_C` | 50 | -40.0 | 150.0 | 0.5 | | | |
| `coupling.bore_mm` | 10 | 4.0 | 16.0 | 0.1 | | | |
| `coupling.keyway_depth_mm` | 1.7 | 0.0 | 4.0 | 0.05 | | | |
| `coupling.c_end` | 0.15 | 0.0 | 0.5 | 0.005 | | yes | |
| `coupling.mu0` | MU0 | 1.2566e-6 | 1.2567e-6 | 1e-11 | | | same as `calibration.mu0` |
| `coupling.gear_ratio` | 5 | 1.0 | 20.0 | 0.1 | | | |
| `coupling.gear_efficiency` | 0.95 | 0.5 | 1.0 | 0.01 | | | |
| `coupling.gearbox_input_rating_Nm` | 0.6 | 0.05 | 5.0 | 0.01 | | | |
| `coupling.drive_torque_Nm` | 0.7 | 0.0 | 5.0 | 0.01 | | | |
| `coupling.drive_safety_factor` | 1.3 | 1.0 | 3.0 | 0.05 | | | |
| `coupling.magnets.manual_{inner,outer}_length_mm` | 12.7 | 2.0 | 50.8 | 0.01 | | | |
| `coupling.magnets.manual_{inner,outer}_width_mm` | 6.35 | 1.0 | 25.4 | 0.01 | | | |
| `coupling.magnets.manual_{inner,outer}_thickness_mm` | 3.17 | 0.5 | 10.0 | 0.01 | | | |
| `coupling.magnets.manual_{inner,outer}_br_T` | 1.29 | 0.2 | 1.5 | 0.001 | | | |
| `metal.required_min_Nm` | 2.5 | 0.1 | 10.0 | 0.01 | | | divides `hot_margin` |
| `metal.min_temp_C` | -40 | -60.0 | 20.0 | 0.5 | | | |
| `metal.variation` | 0.15 | 0.0 | 0.5 | 0.005 | | yes | |
| `metal.sleeve_mm` | 0.1 | 0.0 | 1.0 | 0.01 | | | |
| `metal.liner_mm` | 0.2 | 0.0 | 1.0 | 0.01 | | | |
| `metal.shaft_displacement_mm` | 0.4 | 0.0 | 2.0 | 0.01 | | | |
| `metal.runout_mm`, `metal.deflection_mm`, `metal.sleeve_form_mm` | 0.05 | 0.0 | 0.5 | 0.005 | | | |
| `metal.thermal_mm` | 0.03 | 0.0 | 0.3 | 0.005 | | | |
| `metal.magnet_position_mm`, `metal.residual_target_mm` | 0.2 | 0.0 | 1.0 | 0.01 | | | |
| `metal.al_density_g_mm3` | 0.0027 | 0.0025 | 0.003 | 0.00001 | | | |
| `metal.sleeve_density_g_mm3` | 0.008 | 0.004 | 0.009 | 0.0001 | | | |
| `metal.steel_density_g_mm3` | 0.00785 | 0.0075 | 0.0081 | 0.00001 | | | |
| `metal.slip_rpm` | 2000 | 100.0 | 6000.0 | 10.0 | | | > 0: skin depth `sqrt(2/(ω…))`, divisions by ω |
| `metal.slip_event_s` | 0.1 | 0.01 | 10.0 | 0.01 | yes | yes | |
| `metal.life_events` | 2e7 | 1e4 | 1e9 | 1000.0 | yes | | |
| `metal.measured_drag_Nm` (`Option<f64>`) | None | 0.001 | 1.0 | 0.0001 | yes | | > 0: exactly 0 raises in Python (E13, Review Focus 4) |
| `metal.face_gap_mm` | 1.4 | 0.3 | 5.0 | 0.01 | | | |
| `metal.bond_inner_mm` | 0.05 | 0.01 | 0.2 | 0.005 | | | > 0: Volkersen divides by the bondline |
| `metal.bond_outer_mm` | 0.05 | 0.0 | 0.2 | 0.005 | | | |
| `metal.cup_wall_corner_mm` | 1.8 | 0.5 | 6.0 | 0.05 | | | |
| `metal.hub_length_mm`, `metal.cup_depth_mm` | 13, 15.5 | 3.0 | 40.0 | 0.1 | | | |
| `metal.web_mm` | 2.5 | 0.5 | 8.0 | 0.1 | | | |
| `metal.boss_length_mm` | 13 | 0.0 | 40.0 | 0.1 | | | |
| `metal.boss_od_mm` | 22 | 12.0 | 40.0 | 0.1 | | | |
| `metal.hardware_g` | 6 | 0.0 | 30.0 | 0.5 | | | |
| `metal.max_large_dia_axial_mm` | 20 | 5.0 | 60.0 | 0.5 | | | |
| `metal.max_overall_axial_mm` | 35 | 10.0 | 100.0 | 0.5 | | | |
| `metal.max_diameter_mm` | 43 | 20.0 | 80.0 | 0.5 | | | |
| `metal.cap_axial_mm` | 0.8 | 0.2 | 5.0 | 0.1 | | | |
| `metal.cap_od_mm`, `metal.cap_thread_dia_mm` | 42.8, 41 | 20.0 | 80.0 | 0.1 | | | |
| `metal.cap_thread_engagement_mm` | 2 | 0.5 | 8.0 | 0.1 | | | |
| `metal.front_endplate_mm`, `metal.rear_endplate_mm` | 0.5, 1 | 0.1 | 3.0 | 0.05 | | | |
| `metal.retainer_span_mm` | 14.5 | 3.0 | 50.0 | 0.1 | | | |
| `metal.sleeve_bedding_mm`, `metal.liner_bedding_mm` | 0.025 | 0.0 | 0.2 | 0.005 | | | |
| `metal.rear_endplate_hole_mm` | 4.5 | 0.0 | 12.0 | 0.1 | | | |
| `metal.adapter_flange_dia_mm` | 30 | 10.0 | 60.0 | 0.5 | | | |
| `metal.adapter_flange_mm` | 4.5 | 0.5 | 15.0 | 0.1 | | | |
| `metal.adapter_pilot_dia_mm` | 18 | 5.0 | 40.0 | 0.1 | | | |
| `metal.adapter_pilot_mm` | 2 | 0.0 | 10.0 | 0.1 | | | |
| `metal.adapter_boss_mm` | 10 | 0.0 | 30.0 | 0.1 | | | |
| `metal.adapter_hardware_g` | 2 | 0.0 | 20.0 | 0.5 | | | |
| `materials.steel.bsat_T` | 1.5 | 0.5 | 2.2 | 0.01 | | yes | |
| `materials.steel.conductivity_S_m` | 4.5e6 | 1e6 | 1e7 | 1e4 | yes | | |
| `materials.steel.mu_r_incremental` | 200 | 1.0 | 2000.0 | 1.0 | yes | | |
| `materials.steel.specific_heat_J_kgK` | 473 | 300.0 | 1000.0 | 1.0 | | | |
| `materials.steel.cte_per_C` | 12.3e-6 | 5e-6 | 25e-6 | 1e-7 | | | |
| `materials.steel.modulus_GPa` | 205 | 50.0 | 250.0 | 1.0 | | | |
| `materials.steel.density_g_cm3` | 7.85 | 7.0 | 8.2 | 0.01 | | | |
| `materials.nickel.thickness_mm` | 0.015 | 0.0 | 0.05 | 0.001 | | | |
| `materials.screws.proof_12_9_MPa` | 970 | 500.0 | 1200.0 | 5.0 | | | |
| `materials.screws.proof_10_9_MPa` | 830 | 400.0 | 1100.0 | 5.0 | | | |
| `materials.screws.yield_A4_70_MPa` | 450 | 200.0 | 800.0 | 5.0 | | | |

If the generator (2f) reports that Python raised or returned a non-finite number on some case, narrow the offending range (never beyond physical sense), add a `// keeps ... > 0 (Python raises ...)` comment on the `.range(..)` line, and rerun.

- [ ] **2b. `ModelResults` and the helpers.** Transcribe `out(...)` lines 78-155 as `name: <type> => out(unit, label, help, cell)` (Python's `cell=` keyword form has help `""`). Types by the value Python produces: `inner_tmax_C`, `outer_tmax_C` are `NumOrText` (library rating or `"n/a"`); `inner_flat_check`, `outer_flat_check`, `verdict`, `cup_ring_check`, `hub_check`, `inner_temp_check`, `outer_temp_check` are `String`; everything else `f64`. Helpers:

```rust
/// One magnet ring as the Calculator uses it (Python `ResolvedMagnet`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct ResolvedMagnet {
    pub length_mm: f64,
    pub width_mm: f64,
    pub thickness_mm: f64,
    pub br_T: f64,
    /// Library rating, or `NOT_IN_LIBRARY` for manual magnets.
    pub tmax_C: NumOrText,
}

/// Library values when the part is found, else the manual values (workbook IFERROR/INDEX/MATCH).
#[allow(non_snake_case)]
pub fn resolve_magnets(m: &MagnetInputs, _dev: Deviations) -> (ResolvedMagnet, ResolvedMagnet) {
    let resolve = |part: &str, length_mm: f64, width_mm: f64, thickness_mm: f64, br_T: f64| match lookup(part) {
        Some(spec) => ResolvedMagnet {
            length_mm: spec.length_mm,
            width_mm: spec.width_mm,
            thickness_mm: spec.thickness_mm,
            br_T: spec.br_T,
            tmax_C: NumOrText::Num(spec.tmax_C),
        },
        None => ResolvedMagnet { length_mm, width_mm, thickness_mm, br_T, tmax_C: NumOrText::Text(NOT_IN_LIBRARY) },
    };
    (
        resolve(&m.part_inner, m.manual_inner_length_mm, m.manual_inner_width_mm, m.manual_inner_thickness_mm, m.manual_inner_br_T),
        resolve(&m.part_outer, m.manual_outer_length_mm, m.manual_outer_width_mm, m.manual_outer_thickness_mm, m.manual_outer_br_T),
    )
}

/// Measured correction only for the prototype's circuit (no iron, same poles, B842SH both rings).
pub fn select_calibration_factor(
    backiron: i64, npole: i64, part_i: &str, part_o: &str,
    prototype_poles: f64, f_cal_updated: f64, f_cal_original: f64,
) -> f64 {
    // Python compares the int npole with the float poles_per_ring (10 == 10.0).
    if backiron == 0 && npole as f64 == prototype_poles && part_i == "B842SH" && part_o == "B842SH" {
        f_cal_updated
    } else {
        f_cal_original
    }
}

/// One harmonic's terms (an entry of the Python `shear_stress` dict).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Harmonic {
    pub n: u32,
    pub k: f64,
    pub bi: f64,
    pub bo: f64,
    pub s_iron: f64,
    pub s_free: f64,
    pub tau: f64,
}

impl Harmonic {
    /// The geometry factor of the circuit `backiron` selects: `s_iron` for 1 (steel), else `s_free`.
    pub fn s(&self, backiron: i64) -> f64 {
        if backiron == 1 { self.s_iron } else { self.s_free }
    }
}

/// S_n: sinh ratio for a steel-backed circuit, exponential form for free-space rings.
pub fn geometry_factor(k: f64, t_i_mm: f64, t_o_mm: f64, g_mm: f64, backiron: i64) -> f64 {
    if backiron == 1 {
        (k * t_i_mm / 1000.0).sinh() * (k * t_o_mm / 1000.0).sinh() / (k * (t_i_mm + t_o_mm + g_mm) / 1000.0).sinh()
    } else {
        (1.0 - (-k * t_i_mm / 1000.0).exp()) * (1.0 - (-k * t_o_mm / 1000.0).exp()) * (-k * g_mm / 1000.0).exp() / 2.0
    }
}

/// Odd-harmonic amplitude of a square-wave magnetization with the given fill factor.
#[allow(non_snake_case)]
pub fn harmonic_amplitude(br_T: f64, n: u32, fill: f64) -> f64 {
    let n = f64::from(n);
    br_T * (4.0 / (n * PI)) * (n * fill * PI / 2.0).sin()
}

/// Per-harmonic pull-out shear stress [Pa] and its parts, for HARMONICS.
#[allow(clippy::too_many_arguments)] // Python signature
pub fn shear_stress(
    br_i: f64, br_o: f64, fill_i: f64, fill_o: f64, npole: i64, r_g_mm: f64,
    t_i_mm: f64, t_o_mm: f64, g_mm: f64, backiron: i64, mu0: f64,
) -> [Harmonic; 3] {
    HARMONICS.map(|n| {
        let nf = f64::from(n);
        let k = nf * (npole as f64 / 2.0) / (r_g_mm / 1000.0);
        let bi = harmonic_amplitude(br_i, n, fill_i);
        let bo = harmonic_amplitude(br_o, n, fill_o);
        let s_iron = geometry_factor(k, t_i_mm, t_o_mm, g_mm, 1);
        let s_free = geometry_factor(k, t_i_mm, t_o_mm, g_mm, 0);
        let mut hn = Harmonic { n, k, bi, bo, s_iron, s_free, tau: 0.0 };
        hn.tau = bi * bo / (2.0 * mu0) * hn.s(backiron) * (nf * PI / 2.0).sin();
        hn
    })
}
```

- [ ] **2c. `compute`.** Line by line `model.py` 230-312, Python local names. The whole body:

```rust
/// Calculator sheet. Linked values come from Metal design, Calibration and Materials (see `api`).
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn compute(
    ci: &CouplingInputs, face_gap_mm: f64, bond_inner_mm: f64, bond_outer_mm: f64,
    cup_wall_corner_mm: f64, alpha_br: f64, bsat_T: f64, f_cal: f64, f_cal_original: f64,
    slip_rpm: f64, required_min_Nm: f64, dev: Deviations,
) -> ModelResults {
    let (mi, mo) = resolve_magnets(&ci.magnets, dev);
    let (N, a_i) = (ci.npole as f64, ci.inner_back_apothem_mm);
    let L = py_min(mi.length_mm, mo.length_mm);
    // corner gap from the flat-face gap (formula always uses the corner geometry)
    let corner_gap = face_gap_mm
        - (((a_i + mi.thickness_mm).powi(2) + (mi.width_mm / 2.0).powi(2)).sqrt() - (a_i + mi.thickness_mm));
    let hub_wall = a_i - bond_inner_mm - ci.bore_mm / 2.0;

    let flat_i = 2.0 * a_i * (PI / N).tan();
    let chk_i = if ci.faceted == 1 {
        if flat_i >= mi.width_mm {
            format!("OK, {} mm slack", fmt_fixed(flat_i - mi.width_mm, 2))
        } else {
            "TOO NARROW: increase apothem or reduce poles".to_owned()
        }
    } else {
        "n/a (arcs)".to_owned()
    };
    let r_face_i = a_i + mi.thickness_mm;
    let r_corner_i = if ci.faceted == 1 { (r_face_i.powi(2) + (mi.width_mm / 2.0).powi(2)).sqrt() } else { r_face_i };
    let A_o = r_corner_i + corner_gap;
    let g_m = A_o - r_face_i;
    let flat_o = 2.0 * A_o * (PI / N).tan();
    let chk_o = if ci.faceted == 1 {
        if flat_o >= mo.width_mm {
            format!("OK, blocks {} mm apart at the faces", fmt_fixed(flat_o - mo.width_mm, 2))
        } else {
            "TOO NARROW: increase gap/apothem or reduce poles".to_owned()
        }
    } else {
        "n/a (arcs)".to_owned()
    };
    let A_back = A_o + mo.thickness_mm;
    let r_pocket = if ci.faceted == 1 { (A_back + bond_outer_mm) / (PI / N).cos() } else { A_back + bond_outer_mm };
    let OD = 2.0 * (r_pocket + cup_wall_corner_mm);
    let wall_f = OD / 2.0 - A_back;
    let R_g = r_face_i + g_m / 2.0;
    let tau_p = 2.0 * PI * R_g / N;
    let al_i = py_min(1.0, mi.width_mm / (2.0 * PI * (a_i + mi.thickness_mm / 2.0) / N));
    let al_o = py_min(1.0, mo.width_mm / (2.0 * PI * (A_o + mo.thickness_mm / 2.0) / N));

    let bri = mi.br_T * (1.0 + alpha_br * (ci.op_temp_C - 20.0));
    let bro = mo.br_T * (1.0 + alpha_br * (ci.op_temp_C - 20.0));
    let h = shear_stress(bri, bro, al_i, al_o, ci.npole, R_g, mi.thickness_mm, mo.thickness_mm, g_m, ci.backiron, ci.mu0);
    let tau = h.iter().fold(0.0, |acc, x| acc + x.tau); // Python sum(): left fold from 0
    let AL = 2.0 * PI * (R_g / 1000.0).powi(2) * (L / 1000.0);
    let T2D = tau * AL;
    let f_end = 1.0 - ci.c_end * tau_p / L;
    let T_pull = T2D * f_end * f_cal;
    let T_pull20 = T_pull * (mi.br_T * mo.br_T) / (bri * bro);
    // sum(bi * bo * S * sin(n pi/2) for n) / (2 mu0) * ...: note S inside the product, /(2 mu0) after the sum
    let circuit = |s: fn(&Harmonic) -> f64| {
        h.iter().fold(0.0, |acc, x| acc + x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin())
    };
    let T_iron = circuit(|x| x.s_iron) / (2.0 * ci.mu0) * AL * f_end * f_cal_original;
    let T_noiron = circuit(|x| x.s_free) / (2.0 * ci.mu0) * AL * f_end * f_cal;

    let floor_ = py_max(ci.drive_torque_Nm * ci.drive_safety_factor, required_min_Nm);
    let B_gap = (bri + bro) / 2.0 * (mi.thickness_mm + mo.thickness_mm) / (mi.thickness_mm + mo.thickness_mm + g_m);
    let t_bi = B_gap * tau_p / (PI * bsat_T);

    // (A block may not start with `if ... {}.to_owned()`: bind the &str first.)
    let thick_check = |wall: f64| -> String {
        let text = if ci.backiron == 0 {
            "No back iron"
        } else if wall >= t_bi {
            "Thickness OK"
        } else {
            "Too thin"
        };
        text.to_owned()
    };
    let temp_check = |tmax: NumOrText| -> String {
        let text = match tmax {
            NumOrText::Text(_) => "unknown",
            NumOrText::Num(t) if ci.op_temp_C <= t => "OK",
            NumOrText::Num(_) => "OVER the magnet rating",
        };
        text.to_owned()
    };
    let [h1, h3, h5] = h;

    ModelResults {
        inner_length_mm: mi.length_mm, inner_width_mm: mi.width_mm, inner_thickness_mm: mi.thickness_mm,
        inner_br_T: mi.br_T, inner_tmax_C: mi.tmax_C, outer_length_mm: mo.length_mm, outer_width_mm: mo.width_mm,
        outer_thickness_mm: mo.thickness_mm, outer_br_T: mo.br_T, outer_tmax_C: mo.tmax_C, active_length_mm: L,
        corner_gap_mm: corner_gap, alpha_br_per_C: alpha_br, bsat_T, cup_wall_corner_mm, hub_wall_mm: hub_wall,
        f_cal, inner_flat_width_mm: flat_i, inner_flat_check: chk_i, hub_wall_past_key_mm: hub_wall - ci.keyway_depth_mm,
        inner_face_radius_mm: r_face_i, inner_corner_radius_mm: r_corner_i, outer_face_apothem_mm: A_o,
        face_gap_mm: g_m, outer_flat_width_mm: flat_o, outer_flat_check: chk_o, outer_back_apothem_mm: A_back,
        pocket_corner_radius_mm: r_pocket, cup_od_mm: OD, cup_wall_flat_mm: wall_f, gap_radius_mm: R_g,
        pole_pitch_mm: tau_p, fill_inner: al_i, fill_outer: al_o, br_inner_T_op: bri, br_outer_T_op: bro,
        k1: h1.k, b_i1: h1.bi, b_o1: h1.bo, s1_iron: h1.s_iron, s1_free: h1.s_free, tau1_Pa: h1.tau,
        k3: h3.k, b_i3: h3.bi, b_o3: h3.bo, s3_iron: h3.s_iron, s3_free: h3.s_free, tau3_Pa: h3.tau,
        k5: h5.k, b_i5: h5.bi, b_o5: h5.bo, s5_iron: h5.s_iron, s5_free: h5.s_free, tau5_Pa: h5.tau,
        tau_Pa: tau, area_lever_m3: AL, torque_2d_Nm: T2D, f_end, pullout_Nm: T_pull, pullout_20C_Nm: T_pull20,
        pullout_iron_Nm: T_iron, pullout_noiron_Nm: T_noiron, ripple_freq_Hz: N / 2.0 * slip_rpm / 60.0,
        gearbox_input_ripple_Nm: T_pull / (ci.gear_ratio * ci.gear_efficiency),
        gearbox_reference_Nm: ci.gearbox_input_rating_Nm * ci.gear_ratio * ci.gear_efficiency,
        required_floor_Nm: floor_,
        verdict: if T_pull < floor_ { "Below hot minimum" } else { "Nominal only: hot test" }.to_owned(),
        gap_flux_density_T: B_gap, backiron_needed_mm: t_bi, cup_ring_check: thick_check(cup_wall_corner_mm),
        hub_check: thick_check(hub_wall), inner_temp_check: temp_check(mi.tmax_C), outer_temp_check: temp_check(mo.tmax_C),
    }
}
```

Imports of `model.rs`: `use std::f64::consts::PI; use super::compat::{fmt_fixed, py_max, py_min}; use super::deviations::Deviations; use super::library::lookup; use super::meta::{NumOrText, inputs, out, param, results};` (`metal_design.rs` and `materials.rs` need only `use super::meta::{inputs, param};` in this task). Module doc comment: say it ports `model.py`, and list the planned deviations here: E3 (via `resolve_magnets`), E6 (C63), E7 (pull-out over angle), E8 (C9, C111), E9 (masses), E10 (C103).

- [ ] **2d. Wire into `api.rs`.** `mod.rs`: add `pub mod materials;`, `pub mod metal_design;`, `pub mod model;` (alphabetical). `api.rs`:

```rust
use super::calibration::{self, CalibrationInputs, CalibrationResults};
use super::materials::MaterialsInputs;
use super::metal_design::MetalDesignInputs;
use super::model::{self, CouplingInputs, ModelResults};

inputs! {
    /// Every editable input, grouped as the Python `DesignInputs` (same order).
    pub struct DesignInputs {
        fields {}
        groups {
            coupling: CouplingInputs,
            metal: MetalDesignInputs,
            calibration: CalibrationInputs,
            materials: MaterialsInputs,
        }
    }
}

results! {
    /// Every computed value, grouped as the Python `DesignResults` (same order).
    pub struct DesignResults {
        fields {}
        groups {
            calibration: CalibrationResults,
            model: ModelResults,
        }
    }
}

// Python api.compute_all lines 52-99; Python local names.
fn compute(inputs: &DesignInputs, dev: Deviations) -> DesignResults {
    let (ci, md, cal_in, mat_in) = (&inputs.coupling, &inputs.metal, &inputs.calibration, &inputs.materials);
    let cal = calibration::compute(cal_in, dev);
    let f_cal = model::select_calibration_factor(
        ci.backiron, ci.npole, &ci.magnets.part_inner, &ci.magnets.part_outer,
        cal.poles_per_ring, cal.f_cal_updated, cal_in.f_cal_original,
    );
    let m = model::compute(
        ci, md.face_gap_mm, md.bond_inner_mm, md.bond_outer_mm, md.cup_wall_corner_mm,
        cal_in.alpha_br_per_C, mat_in.steel.bsat_T, f_cal, cal_in.f_cal_original, md.slip_rpm,
        md.required_min_Nm, dev,
    );
    DesignResults { calibration: cal, model: m }
}
```
Update the module doc "Ported so far" line. The `api::tests::paths_match_the_python_api` assertion that every input path starts with `calibration.` no longer holds: change it to assert the first input path is `coupling.npole` and that `calibration.br_T` still reads 1.29.

- [ ] **2e. Unit tests in `model.rs`** (workbook values from `tests/data/reference_values.json`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The Calculator at the workbook defaults, links as `api::compute` passes them.
    fn at(ci: &CouplingInputs) -> ModelResults {
        compute(ci, 1.4, 0.05, 0.05, 1.8, -0.0012, 1.5, 0.95, 0.95, 2000.0, 2.5, Deviations::NONE)
    }

    fn close(got: f64, want: f64) -> bool {
        (got - want).abs() <= 1e-9 * want.abs()
    }

    #[test]
    fn default_design_matches_the_workbook() {
        let r = at(&CouplingInputs::default());
        assert!(close(r.pullout_Nm, 2.6472742027215)); // Calculator!C93
        assert!(close(r.corner_gap_mm, 1.02682560543393)); // C9
        assert_eq!(r.inner_flat_check, "OK, 0.25 mm slack"); // C52
        assert_eq!(r.outer_flat_check, "OK, blocks 3.22 mm apart at the faces"); // C59
        assert_eq!(r.verdict, "Nominal only: hot test"); // C102
        assert_eq!((r.cup_ring_check.as_str(), r.hub_check.as_str()), ("Too thin", "Thickness OK")); // C105, C106
        assert_eq!(r.inner_tmax_C, NumOrText::Num(150.0)); // C22
    }

    #[test]
    fn part_lookup_is_exact_text() {
        // Review Focus 2: near misses fall back to the manual magnet, like the workbook.
        for part in ["b842sh", "B842SH ", ""] {
            let mut ci = CouplingInputs::default();
            ci.magnets.part_inner = part.to_owned();
            let r = at(&ci);
            assert_eq!(r.inner_tmax_C, NumOrText::Text(NOT_IN_LIBRARY), "{part:?}");
            assert_eq!(r.inner_temp_check, "unknown", "{part:?}");
            assert_eq!(select_calibration_factor(0, 10, part, "B842SH", 10.0, 1.0658, 0.95), 0.95, "{part:?}");
        }
        assert_eq!(select_calibration_factor(0, 10, "B842SH", "B842SH", 10.0, 1.0658, 0.95), 1.0658);
    }

    #[test]
    fn calibration_factor_needs_every_prototype_condition() {
        let f = |backiron, npole, poles| select_calibration_factor(backiron, npole, "B842SH", "B842SH", poles, 1.0658, 0.95);
        assert_eq!(f(0, 10, 10.0), 1.0658);
        assert_eq!(f(1, 10, 10.0), 0.95, "steel circuit");
        assert_eq!(f(0, 12, 10.0), 0.95, "other pole count");
        assert_eq!(f(0, 12, 12.0), 1.0658, "a 24-magnet prototype matches 12 poles");
    }

    #[test]
    fn temperature_check_is_inclusive_at_the_rating() {
        let mut ci = CouplingInputs { op_temp_C: 150.0, ..CouplingInputs::default() }; // B842SH rating
        assert_eq!(at(&ci).inner_temp_check, "OK");
        ci.op_temp_C = 150.5;
        assert_eq!(at(&ci).inner_temp_check, "OVER the magnet rating");
    }

    #[test]
    fn checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8. Each right-hand side is a value its left-hand side
        // does not read, so one run supplies the left side and a second run puts the
        // comparison at exact equality (no tolerance involved).
        let base = at(&CouplingInputs::default());

        // Verdict: `T_pull < floor_` is strict, so a pull-out equal to the floor is not "Below".
        let ci = CouplingInputs::default();
        let r = compute(&ci, 1.4, 0.05, 0.05, 1.8, -0.0012, 1.5, 0.95, 0.95, 2000.0, base.pullout_Nm, Deviations::NONE);
        assert_eq!(r.required_floor_Nm, r.pullout_Nm);
        assert_eq!(r.verdict, "Nominal only: hot test");

        // Thickness check `wall >= t_bi` (one closure serves the cup ring and the hub).
        let r = compute(&ci, 1.4, 0.05, 0.05, base.backiron_needed_mm, -0.0012, 1.5, 0.95, 0.95, 2000.0, 2.5, Deviations::NONE);
        assert_eq!(r.backiron_needed_mm, r.cup_wall_corner_mm);
        assert_eq!(r.cup_ring_check, "Thickness OK");

        // Flat checks `flat >= width`. Manual magnets make the widths inputs; the inner width
        // moves the outer apothem, so the outer width is copied after the inner one is set.
        let mut ci = CouplingInputs::default();
        ci.magnets.part_inner = String::new();
        ci.magnets.part_outer = String::new();
        ci.magnets.manual_inner_width_mm = base.inner_flat_width_mm;
        ci.magnets.manual_outer_width_mm = at(&ci).outer_flat_width_mm;
        let r = at(&ci);
        assert_eq!((r.inner_width_mm, r.outer_width_mm), (r.inner_flat_width_mm, r.outer_flat_width_mm));
        assert_eq!(r.inner_flat_check, "OK, 0.00 mm slack");
        assert_eq!(r.outer_flat_check, "OK, blocks 0.00 mm apart at the faces");
    }
}
```
(Checked on a scratch build of this plan's code: the test passes, and swapping any of its four operators, `<` to `<=` or `>=` to `>`, makes it fail.)

- [ ] **2f. Generator and data.** In `gen_differential.py`:

```python
from magcoupling.library import MAGNET_LIBRARY  # noqa: E402  (next to the other engine imports)

MODULES = {
    "calibration": ["calibration"],
    "model": ["coupling", "metal", "materials", "calibration"],
}
# every library part, blank (manual magnet) and a near miss of the default part
PART_CHOICES = list(MAGNET_LIBRARY) + ["", "b842sh"]
TEXT_CHOICES: dict[str, list[str]] = {
    "coupling.magnets.part_inner": PART_CHOICES,
    "coupling.magnets.part_outer": PART_CHOICES,
}
```
and add to `PROBES`:

```python
    # The measured calibration factor applies only to the prototype's circuit
    # (model.select_calibration_factor); random sampling almost never hits all four conditions.
    "model": [
        ("prototype circuit: measured calibration factor",
         {"coupling.backiron": 0, "coupling.npole": 10, "calibration.total_magnets": 20,
          "coupling.magnets.part_inner": "B842SH", "coupling.magnets.part_outer": "B842SH"}),
        ("prototype circuit but 12 poles", {"coupling.backiron": 0, "coupling.npole": 12}),
        ("prototype circuit, 12 poles against a 24-magnet prototype",
         {"coupling.backiron": 0, "coupling.npole": 12, "calibration.total_magnets": 24}),
        ("prototype circuit but an N52 inner part",
         {"coupling.backiron": 0, "coupling.magnets.part_inner": "B842-N52"}),
        ("operating temperature at the 150 C rating", {"coupling.op_temp_C": 150.0}),
        ("operating temperature above the 80 C rating of the outer part",
         {"coupling.op_temp_C": 80.5, "coupling.magnets.part_outer": "B842"}),
        ("manual magnets in both rings", {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""}),
    ],
```
Add `"harmonics": list(model.HARMONICS)` to `static_data()` (import `from magcoupling import model as py_model` and use `py_model.HARMONICS` to avoid shadowing). In `tests/static_data.rs` add:

```rust
#[test]
fn harmonics_equal_the_python_list() {
    let doc = python_static();
    let python: Vec<u64> = doc["harmonics"].as_array().expect("harmonics").iter().map(|n| n.as_u64().expect("an int")).collect();
    let rust: Vec<u64> = magcoupling::engine::model::HARMONICS.iter().map(|&n| u64::from(n)).collect();
    assert_eq!(rust, python);
}
```

Bless the schema and regenerate:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test schema
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: the generator prints `wrote .../differential/model.json (...)` with about 336 case lines (236 fixed + 100 random) and no traceback. A `RuntimeError: model case ... raised ...` or `ValueError: ...: non-finite value` means a range lets Python leave its domain: fix the range (2a), bless, rerun.

- [ ] **2g. Differential test.** In `tests/differential.rs` add

```rust
#[test]
fn model_matches_python_on_every_case() {
    let cases = check_module("model");
    let prototype = cases
        .iter()
        .find(|c| c.tag == "prototype circuit: measured calibration factor")
        .expect("the prototype probe");
    assert_ne!(
        prototype.results["model.f_cal"], prototype.inputs["calibration.f_cal_original"],
        "the prototype probe must select the measured factor"
    );
}
```
and these `BRANCHES` rows:

```rust
    ("model.inner_tmax_C", &[Number, Text("n/a")]),
    ("model.outer_tmax_C", &[Number, Text("n/a")]),
    ("model.inner_flat_check", &[Prefix("OK, "), Text("TOO NARROW: increase apothem or reduce poles"), Text("n/a (arcs)")]),
    ("model.outer_flat_check", &[Prefix("OK, blocks "), Text("TOO NARROW: increase gap/apothem or reduce poles"), Text("n/a (arcs)")]),
    ("model.verdict", &[Text("Below hot minimum"), Text("Nominal only: hot test")]),
    ("model.cup_ring_check", &[Text("No back iron"), Text("Thickness OK"), Text("Too thin")]),
    ("model.hub_check", &[Text("No back iron"), Text("Thickness OK"), Text("Too thin")]),
    ("model.inner_temp_check", &[Text("unknown"), Text("OK"), Text("OVER the magnet rating")]),
    ("model.outer_temp_check", &[Text("unknown"), Text("OK"), Text("OVER the magnet rating")]),
```

- [ ] **Step 3: Run parity and differential for the module**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "^test .*(FAILED|ok)$|test result"
```
Expected: every `test result: ok.`, 0 failed. Parity now checks 96 result cells (calibration 23 + model 73) and 98 default inputs (coupling 24, metal 47, calibration 16, materials 11); `model_matches_python_on_every_case`, `every_branch_is_reached`, `every_varied_input_takes_two_values`, `ported_inputs_carry_the_python_metadata_and_defaults`, `every_python_field_of_a_ported_group_is_ported`, `harmonics_equal_the_python_list`, and the five new `model::tests` pass (including `checks_take_the_python_branch_at_exact_equality`).

Mutation spot check: change `(ci.op_temp_C - 20.0)` in `bri` to `(ci.op_temp_C - 21.0)`, run `--test parity`: expected FAIL naming `model.br_inner_T_op (Calculator!C69)` among others. Then restore it and change `"No back iron"` to `"No backiron"`: parity stays green (default back iron is steel), `--test differential` FAILS on `model.cup_ring_check` for `backiron = 0` cases. Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task3.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task3.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status: "Calibration, magnet library, Calculator model"; `docs/ai/03-structure.yaml`: `ported: [constants.rs, calibration.rs, library.rs, model.rs]`, note `metal_design.rs` and `materials.rs` hold inputs only so far.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the Calculator model

model.py (73 result cells) with the Metal design and Materials input
structs it reads (58 input cells), wired into compute_all in the Python
order; slider ranges for every new input; differential data for the model
group (library parts, blank and near-miss part text, prototype-circuit
probes).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 4: Retainers (`metal_design.retainers`)

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Modify: `magcoupling-rs/src/engine/metal_design.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: new `tests/data/differential/retainers.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`

**Interfaces:**
- Consumes: `MetalDesignInputs` (Task 3); `ModelResults { inner_thickness_mm, inner_width_mm, outer_face_apothem_mm, .. }`, `CouplingInputs { inner_back_apothem_mm, bore_mm, .. }` (Task 3).
- Produces: `results! RetainerResults` (9 fields: `retainer_span_mm, retainers_g, sleeve_id_mm, sleeve_od_mm, liner_od_mm, liner_id_mm, endplate_od_mm, cap_g, endplates_g`), `pub fn retainers(md: &MetalDesignInputs, inner_back_apothem_mm: f64, inner_thickness_mm: f64, inner_width_mm: f64, outer_face_apothem_mm: f64, bore_mm: f64, dev: Deviations) -> RetainerResults`; `DesignResults` groups `calibration, model, retainers` (Task 5 inserts `mass` before `retainers`).

- [ ] **Step 1: Enable the cells (failing).** Append `PortedResults { group: "retainers", cells: 9 }` to `PORTED_RESULTS`.

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "right|test result"
```
Expected: FAIL, `right: {"calibration": 23, "model": 73, "retainers": 9}`.

- [ ] **Step 2: Port.** Python spec: `metal_design.py` lines 100-110 (`RetainerResults`, all `f64`), 166-181 (`retainers`).

```rust
/// Sleeve, liner, endplate and cap geometry and mass (Metal design rows 45-46, 175-181).
#[allow(clippy::too_many_arguments)] // Python signature
pub fn retainers(
    md: &MetalDesignInputs, inner_back_apothem_mm: f64, inner_thickness_mm: f64, inner_width_mm: f64,
    outer_face_apothem_mm: f64, bore_mm: f64, _dev: Deviations,
) -> RetainerResults {
    let sleeve_id = 2.0
        * (((inner_back_apothem_mm + inner_thickness_mm).powi(2) + (inner_width_mm / 2.0).powi(2)).sqrt()
            + md.sleeve_bedding_mm);
    let sleeve_od = sleeve_id + 2.0 * md.sleeve_mm;
    let liner_od = 2.0 * (outer_face_apothem_mm - md.liner_bedding_mm);
    let liner_id = liner_od - 2.0 * md.liner_mm;
    let span = md.retainer_span_mm;
    let m_ret = PI / 4.0 * (sleeve_od.powi(2) - sleeve_id.powi(2) + liner_od.powi(2) - liner_id.powi(2))
        * span * md.sleeve_density_g_mm3;
    let cap = (PI / 4.0 * (md.cap_od_mm.powi(2) - liner_id.powi(2)) * md.cap_axial_mm
        + PI / 4.0 * (md.cap_od_mm.powi(2) - md.cap_thread_dia_mm.powi(2)) * md.cap_thread_engagement_mm)
        * md.al_density_g_mm3;
    let endplate_od = sleeve_id;
    let endplates = PI / 4.0
        * ((endplate_od.powi(2) - bore_mm.powi(2)) * md.front_endplate_mm
            + (endplate_od.powi(2) - md.rear_endplate_hole_mm.powi(2)) * md.rear_endplate_mm)
        * md.sleeve_density_g_mm3;
    RetainerResults {
        retainer_span_mm: span, retainers_g: m_ret, sleeve_id_mm: sleeve_id, sleeve_od_mm: sleeve_od,
        liner_od_mm: liner_od, liner_id_mm: liner_id, endplate_od_mm: endplate_od, cap_g: cap, endplates_g: endplates,
    }
}
```

`metal_design.rs` now also imports `std::f64::consts::PI`, `super::deviations::Deviations` and `super::meta::{out, results}`. `api.rs`: add `retainers: RetainerResults` after `model` in `DesignResults`, and after `let m = ...`:

```rust
    let ret = metal_design::retainers(
        md, ci.inner_back_apothem_mm, m.inner_thickness_mm, m.inner_width_mm, m.outer_face_apothem_mm, ci.bore_mm, dev,
    );
```

Generator: `MODULES["retainers"] = ["coupling", "metal"]`. `tests/differential.rs`: `#[test] fn retainers_matches_python_on_every_case() { check_module("retainers"); }` (no text results, no `BRANCHES` rows). Regenerate (no schema change, so no bless):

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote .../differential/retainers.json (...)`, no traceback.

- [ ] **Step 3: Run parity and differential**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: every `test result: ok.`, no `FAILED`. Parity checks 105 result cells (23 + 73 + 9); Metal design!C175 equals 27.4363487891322 (snapshot).

Mutation spot check: change `md.sleeve_bedding_mm` to `md.liner_bedding_mm` in `sleeve_id`: parity stays green (both defaults 0.025), differential FAILS on `retainers.sleeve_id_mm`. Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task4.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task4.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status adds "retainers"; `docs/ai/03-structure.yaml` notes `metal_design.rs` has inputs and retainers.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the Metal design retainers

metal_design.retainers (9 result cells): sleeve, liner, endplates and cap.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 5: Mass estimate (`model.mass_estimate`)

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Modify: `magcoupling-rs/src/engine/model.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: new `tests/data/differential/mass.json`
- Modify: `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `ModelResults`, `CouplingInputs` (Task 3); `RetainerResults { retainers_g, cap_g, endplates_g, .. }` (Task 4); `constants::NDFEB_DENSITY_G_MM3`.
- Produces: `results! MassResults` (6 fields: `magnets_g, cup_g, hub_g, boss_g, total_g, added_inertia_kgm2`), `pub fn mass_estimate(ci: &CouplingInputs, r: &ModelResults, bond_inner_mm: f64, bond_outer_mm: f64, cup_depth_mm: f64, web_mm: f64, hub_length_mm: f64, boss_length_mm: f64, boss_od_mm: f64, steel_density_g_mm3: f64, al_density_g_mm3: f64, retainers_g: f64, hardware_g: f64, cap_g: f64, endplates_g: f64, dev: Deviations) -> MassResults`; `DesignResults` groups `calibration, model, mass, retainers`.

- [ ] **Step 1: Enable the cells (failing).** Insert `PortedResults { group: "mass", cells: 6 }` between `model` and `retainers` (Python order).

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "right|test result"
```
Expected: FAIL, `right` includes `"mass": 6`.

- [ ] **Step 2: Port.** Python spec: `model.py` lines 158-166 (`MassResults`, all `f64`), 315-330 (`mass_estimate`).

```rust
/// Calculator rows 110-115. Gross solids: no holes, slots or threads subtracted.
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn mass_estimate(
    ci: &CouplingInputs, r: &ModelResults, bond_inner_mm: f64, bond_outer_mm: f64, cup_depth_mm: f64,
    web_mm: f64, hub_length_mm: f64, boss_length_mm: f64, boss_od_mm: f64, steel_density_g_mm3: f64,
    al_density_g_mm3: f64, retainers_g: f64, hardware_g: f64, cap_g: f64, endplates_g: f64, _dev: Deviations,
) -> MassResults {
    let N = ci.npole as f64;
    let m_mag = N
        * (r.inner_length_mm * r.inner_width_mm * r.inner_thickness_mm
            + r.outer_length_mm * r.outer_width_mm * r.outer_thickness_mm)
        * NDFEB_DENSITY_G_MM3;
    let m_ring = ((PI * (r.cup_od_mm / 2.0).powi(2)
        - N * (r.outer_back_apothem_mm + bond_outer_mm).powi(2) * (PI / N).tan())
        * cup_depth_mm
        + PI * ((r.cup_od_mm / 2.0).powi(2) - (ci.bore_mm / 2.0).powi(2)) * web_mm)
        * steel_density_g_mm3;
    let hub_area = if ci.faceted == 1 {
        N * (ci.inner_back_apothem_mm - bond_inner_mm).powi(2) * (PI / N).tan()
    } else {
        PI * (ci.inner_back_apothem_mm - bond_inner_mm).powi(2)
    };
    let m_hub = (hub_area - PI * (ci.bore_mm / 2.0).powi(2))
        * hub_length_mm
        * (if ci.backiron == 1 { steel_density_g_mm3 } else { al_density_g_mm3 });
    let m_boss = PI * ((boss_od_mm / 2.0).powi(2) - (ci.bore_mm / 2.0).powi(2)) * boss_length_mm * steel_density_g_mm3;
    let total = m_mag + m_ring + m_hub + m_boss + retainers_g + hardware_g + cap_g + endplates_g;
    MassResults {
        magnets_g: m_mag, cup_g: m_ring, hub_g: m_hub, boss_g: m_boss, total_g: total,
        added_inertia_kgm2: total / 1000.0 * 0.25_f64.powi(2),
    }
}
```

`model.rs` imports `super::constants::NDFEB_DENSITY_G_MM3`. `api.rs`: insert `mass: MassResults` between `model` and `retainers` in `DesignResults`; after `let ret = ...`:

```rust
    let mass = model::mass_estimate(
        ci, &m, md.bond_inner_mm, md.bond_outer_mm, md.cup_depth_mm, md.web_mm, md.hub_length_mm,
        md.boss_length_mm, md.boss_od_mm, md.steel_density_g_mm3, md.al_density_g_mm3, ret.retainers_g,
        md.hardware_g, ret.cap_g, ret.endplates_g, dev,
    );
```
and build `DesignResults { calibration: cal, model: m, mass, retainers: ret }`.

Generator: `MODULES["mass"] = ["coupling", "metal"]`. `tests/differential.rs`: `#[test] fn mass_matches_python_on_every_case() { check_module("mass"); }`. Regenerate:

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote .../differential/mass.json (...)`.

- [ ] **Step 3: Run parity and differential**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. Parity: 111 result cells; Calculator!C114 = 173.793262199962.

Mutation spot check: replace `al_density_g_mm3` with `steel_density_g_mm3` in `m_hub`: parity green (default back iron is steel), differential FAILS on `mass.hub_g` for `backiron = 0` cases. Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task5.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task5.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status adds "mass estimate".

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the Calculator mass estimate

model.mass_estimate (6 result cells), after the retainers as in Python.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 6: Metal design (`metal_design.compute`)

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Modify: `magcoupling-rs/src/engine/metal_design.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`, `tests/static_data.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: new `tests/data/differential/metal.json`, `tests/data/static_data.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`

**Interfaces:**
- Consumes: `ModelResults`, `MassResults { total_g, boss_g, .. }`, `RetainerResults`, `CalibrationInputs { alpha_br_per_C, measured_torque_Nm, test_temp_C, .. }`, `compat::py_max`.
- Produces: `results! MetalDesignResults` (49 fields), `pub const NOT_MEASURED: &str = "not measured";`, `pub const VALIDATION_ITEMS: [(&str, &str, &str); 12]` (label, status, what to do), `pub fn compute(md: &MetalDesignInputs, torque_op_Nm: f64, torque_20C_Nm: f64, op_temp_C: f64, alpha_br: f64, corner_gap_mm: f64, face_gap_mm: f64, cup_od_mm: f64, npole: i64, bore_mm: f64, gear_ratio: f64, gear_eff: f64, mass_total_g: f64, boss_mass_g: f64, ret: &RetainerResults, proto_measured_Nm: f64, proto_test_temp_C: f64, dev: Deviations) -> MetalDesignResults`; `DesignResults` groups `calibration, model, mass, retainers, metal`.

- [ ] **Step 1: Enable the cells (failing).** Append `PortedResults { group: "metal", cells: 49 }`.

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "right|test result"
```
Expected: FAIL, `right` includes `"metal": 49`.

- [ ] **Step 2: Port.** Python spec: `metal_design.py` lines 84-97 (`VALIDATION_ITEMS`), 113-163 (`MetalDesignResults`), 184-236 (`compute`). Types: `slip_loss_W`, `slip_energy_J` are `NumOrText` (`NOT_MEASURED` until a drag is entered); `hot_min_check`, `clearance_check` are `String`; `installed_magnets` is `f64` (Python `2 * npole`, compared numerically). Put `#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature` on `compute`: it has 18 parameters (`clippy::too_many_arguments` under gate 5's `-D warnings`) and the closure `th = |T: f64|` keeps Python's capital (`non_snake_case`). The non-obvious lines:

```rust
    let th = |T: f64| 1.0 + alpha_br * (T - 20.0); // Python lambda th
    let cold = torque_20C_Nm * th(md.min_temp_C).powi(2);
    // ... hot_low, cold_high, clearance, adverse (six terms, Python order), min_run as lines 190-195
    let slip_f = npole as f64 / 2.0 * md.slip_rpm / 60.0;
    let (loss, energy) = match md.measured_drag_Nm {
        Some(drag) => {
            let loss = drag * 2.0 * PI * md.slip_rpm / 60.0;
            (NumOrText::Num(loss), NumOrText::Num(loss * md.slip_event_s))
        }
        None => (NumOrText::Text(NOT_MEASURED), NumOrText::Text(NOT_MEASURED)),
    };
    let rot_od = py_max(cup_od_mm, md.cap_od_mm);
    // ... stack, large, adapter, removed, hybrid as lines 202-209
    let cold_for_min = md.required_min_Nm * (th(md.min_temp_C) / th(op_temp_C)).powi(2);
```
and in the struct literal: `installed_magnets: 2.0 * npole as f64` (never `2 * npole`: integer overflow rule), `hot_min_check: if hot_low < md.required_min_Nm { "Below hot minimum" } else { "Estimate covers hot min" }.to_owned()`, `clearance_check: if min_run < md.residual_target_mm { "Below target" } else { "Meets assumed target" }.to_owned()`. All other fields transcribe lines 211-236 in order.

```rust
/// Open validation items from the sheet: (label, status, what to do). A GUI checklist.
pub const VALIDATION_ITEMS: [(&str, &str, &str); 12] = [
    ("Prototype metrology", "Open", "Record corner gap, flat gap, radii, overlap, magnet orientation and magnet temperature."),
    // ... the other eleven, lines 86-96, same order and text
];
```

Unit test in `metal_design.rs` (the equality edges of its two checks):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8. Neither left-hand side (hot_low, min_run) reads its
        // right-hand side, so one run supplies it and a second run puts the check at equality.
        // The default design's retainers, rounded:
        let ret = RetainerResults {
            retainer_span_mm: 14.5, retainers_g: 3.131, sleeve_id_mm: 27.436, sleeve_od_mm: 27.636,
            liner_od_mm: 29.39, liner_id_mm: 28.99, endplate_od_mm: 27.436, cap_g: 2.322, endplates_g: 6.653,
        };
        let run = |md: &MetalDesignInputs| {
            compute(md, 2.6473, 2.8487, 50.0, -0.0012, 1.0268, 1.4, 41.2, 10, 10.0, 5.0, 0.95, 173.79, 30.778, &ret,
                0.9, 20.0, Deviations::NONE)
        };
        let base = run(&MetalDesignInputs::default());
        let md = MetalDesignInputs {
            required_min_Nm: base.torque_hot_low_Nm,
            residual_target_mm: base.min_running_clearance_mm,
            ..MetalDesignInputs::default()
        };
        let r = run(&md);
        assert_eq!((r.torque_hot_low_Nm, r.min_running_clearance_mm), (md.required_min_Nm, md.residual_target_mm));
        assert_eq!(r.hot_min_check, "Estimate covers hot min"); // `hot_low < required` is strict
        assert_eq!(r.clearance_check, "Meets assumed target"); // `min_run < target` is strict
    }
}
```
(The retainers are a literal, not a `retainers(...)` call, so Task 21's new `retainers` parameter does not touch this test.)

`metal_design.rs` imports `super::compat::py_max` and `super::meta::NumOrText`. `api.rs`: add `metal: MetalDesignResults` after `retainers`; after `let mass = ...`:

```rust
    let mdr = metal_design::compute(
        md, m.pullout_Nm, m.pullout_20C_Nm, ci.op_temp_C, cal_in.alpha_br_per_C, m.corner_gap_mm, m.face_gap_mm,
        m.cup_od_mm, ci.npole, ci.bore_mm, ci.gear_ratio, ci.gear_efficiency, mass.total_g, mass.boss_g, &ret,
        cal_in.measured_torque_Nm, cal_in.test_temp_C, dev,
    );
```

Generator: `MODULES["metal"] = ["coupling", "metal", "calibration"]`; `PROBES["metal"]`:

```python
    # A bench drag replaces 'not measured' (random cases leave it None 20 % of the time).
    "metal": [
        ("measured drag at the slider minimum", {"metal.measured_drag_Nm": 0.001}),
        ("measured drag 0.05 N m", {"metal.measured_drag_Nm": 0.05}),
    ],
```
`static_data()`: `"validation_items": [[label, status, action] for label, (status, action) in py_metal_design.VALIDATION_ITEMS.items()]` (import `from magcoupling import metal_design as py_metal_design`). `tests/static_data.rs`:

```rust
#[test]
fn validation_items_equal_the_python_checklist() {
    let doc = python_static();
    let python: Vec<[String; 3]> = doc["validation_items"]
        .as_array()
        .expect("validation_items")
        .iter()
        .map(|row| [0, 1, 2].map(|i| row[i].as_str().expect("text").to_owned()))
        .collect();
    let rust: Vec<[String; 3]> = magcoupling::engine::metal_design::VALIDATION_ITEMS
        .iter()
        .map(|&(a, b, c)| [a.to_owned(), b.to_owned(), c.to_owned()])
        .collect();
    assert_eq!(rust, python);
}
```
`tests/differential.rs`: `#[test] fn metal_matches_python_on_every_case() { check_module("metal"); }` and `BRANCHES` rows:

```rust
    ("metal.hot_min_check", &[Text("Below hot minimum"), Text("Estimate covers hot min")]),
    ("metal.clearance_check", &[Text("Below target"), Text("Meets assumed target")]),
    ("metal.slip_loss_W", &[Number, Text("not measured")]),
    ("metal.slip_energy_J", &[Number, Text("not measured")]),
```
Regenerate:

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote .../differential/metal.json (...)` and `static_data.json` rewritten.

- [ ] **Step 3: Run parity and differential**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. Parity: 160 result cells; Metal design!C9 = 2.25018307231327, C11 = "Below hot minimum", C35 = −0.103174394566074, C37 = "Below target", C91 = "not measured"; `metal_design::tests::checks_take_the_python_branch_at_exact_equality` passes.

Mutation spot check: swap the two `hot_min_check` texts: parity FAILS on Metal design!C11. Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task6.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task6.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status adds "Metal design"; `docs/ai/03-structure.yaml` `ported` adds `metal_design.rs`.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the Metal design sheet

metal_design.compute (49 result cells) and the validation checklist.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 7: Materials (`materials.compute`)

First line: depends on D3 (the `proof` fallback). If D3 is the `Result` alternative, stop and escalate.

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Modify: `magcoupling-rs/src/engine/materials.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`, `tests/static_data.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: new `tests/data/differential/materials.json`, `tests/data/static_data.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`

**Interfaces:**
- Consumes: `MaterialsInputs` (Task 3), `ModelResults { backiron_needed_mm, .. }`, `compat::{ceiling, fmt_fixed}`.
- Produces: `pub struct AluminiumAlloy { pub name: &'static str, pub yield_MPa: f64, pub shear_MPa: f64, pub head_pressure_limit_MPa: f64, pub key_bearing_allow_MPa: f64, pub conductivity_S_m: f64 }`, `pub const AL7075: AluminiumAlloy`, `pub const AL6061: AluminiumAlloy`, `impl ScrewClasses { pub fn proof(&self, code: i64) -> f64 }` (NaN outside 1..=3, decision D3), `results! MaterialsResults` (7 fields), `pub fn compute(mat: &MaterialsInputs, t_bi_req_mm: f64, wall_corner_mm: f64, dev: Deviations) -> MaterialsResults`; `DesignResults` groups `..., metal, materials`.

- [ ] **Step 1: Enable the cells (failing).** Append `PortedResults { group: "materials", cells: 7 }`.

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "right|test result"
```
Expected: FAIL, `right` includes `"materials": 7`.

- [ ] **Step 2: Port.** Python spec: `materials.py` lines 36-57 (`AluminiumAlloy`, `Aluminium`: plain data, Materials!C34:C43), 66-68 (`ScrewClasses.proof`), 85-95 (`MaterialsResults`; `cup_wall_check` is `String`), 98-113 (`compute`).

```rust
/// An aluminium alloy's properties (Materials!C34:C43). Plain data in Python
/// (no metadata), so static consts here, not inputs.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct AluminiumAlloy {
    pub name: &'static str,
    pub yield_MPa: f64,
    pub shear_MPa: f64,
    pub head_pressure_limit_MPa: f64,
    pub key_bearing_allow_MPa: f64,
    pub conductivity_S_m: f64,
}

/// 7075-T6 (Materials!C34:C38): clamp collars and adapters.
pub const AL7075: AluminiumAlloy = AluminiumAlloy {
    name: "7075-T6", yield_MPa: 503.0, shear_MPa: 331.0, head_pressure_limit_MPa: 400.0,
    key_bearing_allow_MPa: 100.0, conductivity_S_m: 1.9e7,
};

/// 6061-T6 (Materials!C39:C43): cap, housing, brackets.
pub const AL6061: AluminiumAlloy = AluminiumAlloy {
    name: "6061-T6", yield_MPa: 276.0, shear_MPa: 207.0, head_pressure_limit_MPa: 250.0,
    key_bearing_allow_MPa: 60.0, conductivity_S_m: 2.5e7,
};

impl ScrewClasses {
    /// Proof stress by class code: 1 = 12.9, 2 = 10.9, 3 = A4-70 (workbook CHOOSE order).
    /// Python raises KeyError for any other code; here it is NaN (decision D3): no
    /// panic, and `DesignInputs::validate` reports the code.
    pub fn proof(&self, code: i64) -> f64 {
        match code {
            1 => self.proof_12_9_MPa,
            2 => self.proof_10_9_MPa,
            3 => self.yield_A4_70_MPa,
            _ => f64::NAN,
        }
    }
}

/// Cup wall check against the back-iron need, and electroless-nickel pre-plate offsets.
pub fn compute(mat: &MaterialsInputs, t_bi_req_mm: f64, wall_corner_mm: f64, _dev: Deviations) -> MaterialsResults {
    let check = if wall_corner_mm >= t_bi_req_mm {
        "OK".to_owned()
    } else {
        format!("Too thin: raise Metal design C122 to at least {} mm", fmt_fixed(ceiling(t_bi_req_mm, 0.1), 1))
    };
    let t = mat.nickel.thickness_mm;
    MaterialsResults {
        backiron_thickness_needed_mm: t_bi_req_mm, cup_wall_corner_mm: wall_corner_mm, cup_wall_check: check,
        hub_flats_under_mm: t, cup_pockets_over_mm: t, bores_over_dia_mm: 2.0 * t, ods_under_dia_mm: 2.0 * t,
    }
}
```

Unit tests in `materials.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wall_check_passes_at_equality() {
        let mat = MaterialsInputs::default();
        assert_eq!(compute(&mat, 1.8, 1.8, Deviations::NONE).cup_wall_check, "OK");
        assert_eq!(
            compute(&mat, 1.90415278222222, 1.8, Deviations::NONE).cup_wall_check,
            "Too thin: raise Metal design C122 to at least 2.0 mm" // Materials!C22
        );
    }

    #[test]
    fn proof_follows_the_class_code_and_is_nan_outside_it() {
        let s = ScrewClasses::default();
        assert_eq!([s.proof(1), s.proof(2), s.proof(3)], [970.0, 830.0, 450.0]);
        for bad in [0, 4, -1, i64::MIN, i64::MAX] {
            assert!(s.proof(bad).is_nan(), "{bad}");
        }
    }
}
```

`materials.rs` imports `super::compat::{ceiling, fmt_fixed}`, `super::deviations::Deviations` and `super::meta::{out, results}`. `api.rs`: add `materials: MaterialsResults` after `metal`; after `let mdr = ...`: `let matr = materials::compute(mat_in, m.backiron_needed_mm, md.cup_wall_corner_mm, dev);`.

Generator: `MODULES["materials"] = ["coupling", "metal", "materials", "calibration"]`; `static_data()`: `"aluminium": [dataclasses.asdict(a) for a in (Aluminium().al7075, Aluminium().al6061)]` (import `from magcoupling.materials import Aluminium`). `tests/static_data.rs`:

```rust
#[test]
fn aluminium_alloys_equal_the_python_data() {
    use magcoupling::engine::materials::{AL6061, AL7075};
    let doc = python_static();
    let rows = doc["aluminium"].as_array().expect("aluminium");
    assert_eq!(rows.len(), 2);
    for (py, rs) in rows.iter().zip([AL7075, AL6061]) {
        assert_eq!(rs.name, text(py, "name"));
        for (key, value) in [
            ("yield_MPa", rs.yield_MPa),
            ("shear_MPa", rs.shear_MPa),
            ("head_pressure_limit_MPa", rs.head_pressure_limit_MPa),
            ("key_bearing_allow_MPa", rs.key_bearing_allow_MPa),
            ("conductivity_S_m", rs.conductivity_S_m),
        ] {
            assert_eq!(value, number(py, key), "{}.{key}", rs.name);
        }
    }
}
```
`tests/differential.rs`: `#[test] fn materials_matches_python_on_every_case() { check_module("materials"); }` and `BRANCHES` row `("materials.cup_wall_check", &[Text("OK"), Prefix("Too thin: raise Metal design C122 to at least ")]),`. Regenerate:

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote .../differential/materials.json (...)`.

- [ ] **Step 3: Run parity and differential**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. Parity: 167 result cells; the new unit tests and `aluminium_alloys_equal_the_python_data` pass.

Mutation spot check: `fmt_fixed(..., 1)` to `fmt_fixed(..., 2)`: parity FAILS on Materials!C22 (`2.00 mm`). Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task7.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task7.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status adds "Materials"; translation-rules table row for a dict lookup by code now cites `ScrewClasses::proof` as the example; `docs/ai/03-structure.yaml` `ported` adds `materials.rs`.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the Materials sheet

materials.compute (7 result cells), the aluminium alloys as static data,
and ScrewClasses::proof (NaN for a code outside 1-3 instead of a panic).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---
### Task 8: Temperature design (`temperature.py`)

First line: depends on D3 (the adhesive fallback). If D3 is the `Result` alternative, stop and escalate.

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Create: `magcoupling-rs/src/engine/temperature.rs`
- Create: `magcoupling-rs/tests/robustness.rs`
- Modify: `magcoupling-rs/src/engine/mod.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`, `tests/static_data.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: `tests/data/input_schema.json`, `tests/data/static_data.json`; new `tests/data/differential/temperature.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`

**Interfaces:**
- Consumes: `ModelResults`, `MassResults`, `RetainerResults`, `MetalDesignResults { torque_cold_high_Nm, magnetic_cycles, .. }`, `MetalDesignInputs`, `MaterialsInputs { steel, .. }`, `materials::AL6061`, `CalibrationInputs { alpha_br_per_C, .. }`, `CouplingInputs { op_temp_C, npole, mu0, inner_back_apothem_mm, .. }`, `compat::{py_min, py_max, text0}`.
- Produces:
  - Input structs `DutyInputs` (5), `DemagInputs` (8), `AdhesiveInputs` (1: `selected`), `MismatchInputs` (4), `SlipLossInputs` (11), `ThermalInputs` (4), `AdhesiveLifeInputs` (4), `TemperatureInputs { fields {} groups { duty, demag, adhesive, mismatch, slip_loss, thermal, adhesive_life } }`.
  - `pub struct AdhesiveCandidate { pub name: &'static str, pub design_limit_C: f64, pub cure_C: f64, pub lap_shear_MPa: f64, pub role: &'static str, pub note: &'static str }`, `pub const ADHESIVES: [AdhesiveCandidate; 4]`, `pub const NO_ADHESIVE: AdhesiveCandidate`, `pub fn selected_adhesive(code: i64) -> &'static AdhesiveCandidate`.
  - `pub struct TemperatureLinks` (47 fields, below).
  - Result structs `SummaryResults` (22), `DutyResults` (7), `DemagResults` (13), `AdhesiveResults` (14 incl. uncelled `selected_name`), `MismatchResults` (9), `SlipLossResults` (15), `ThermalResults` (21), `SlipLifeResults` (14), `MagnetLifeResults` (6), `AdhesiveLifeResults` (10), `TemperatureResults` (groups in that order).
  - `pub const NEVER: &str = "never: steady state stays below the limit";`, `pub const NEVER_SHORT: &str = "never";`
  - `pub fn demag_onset_C(h_rev_kA_m: f64, hcj20: f64, beta: f64, knee: f64, alpha_br: f64, offset_C: f64) -> f64`, `pub fn volkersen_peak_shear_MPa(G_GPa: f64, d_alpha: f64, dT: f64, bondline_mm: f64, magnet_E_GPa: f64, magnet_t_mm: f64, steel_E_GPa: f64, steel_t_mm: f64, bond_length_mm: f64) -> f64`, `pub fn compute(ti: &TemperatureInputs, k: &TemperatureLinks, dev: Deviations) -> TemperatureResults`.
  - `DesignInputs` groups `coupling, metal, calibration, materials, temperature`; `DesignResults` groups `..., materials, temperature`.
  - `tests/robustness.rs` (Review Focus tests; Tasks 10 and 12 add to it).

- [ ] **Step 1: Enable the cells (failing).** `PORTED_INPUTS`: append `PortedInputs { group: "temperature", cells: 37 }`. `PORTED_RESULTS`: append `PortedResults { group: "temperature", cells: 130 }` (the 131st result, `temperature.adhesive.selected_name`, has no cell: differential only).

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "right|test result"
```
Expected: FAIL; both `right` maps include `"temperature"`.

- [ ] **Step 2: Port.** Imports of `temperature.rs`: `use std::f64::consts::PI; use super::compat::{py_max, py_min, text0}; use super::deviations::Deviations; use super::meta::{NumOrText, inputs, out, out_uncelled, param, results};`. Python spec: `temperature.py` lines 37-39 (`_text0` = `compat::text0`), 43-157 (inputs, adhesive candidates), 160-209 (`TemperatureLinks`), 213-395 (results), 399-410 (helpers), 413-547 (`compute`). One `compute` with the Python section comments; the file stays one file (about 700 lines, under the architecture's 800-line split threshold).

- [ ] **2a. Inputs.** Transcribe lines 44-137 byte for byte as `name: <type> = <default> => param(unit, label, help, cell)` (Python `float` is `f64` with a float literal default, `int` is `i64`; the metadata-parity test compares every string and default with Python). The `candidates` list field of `AdhesiveInputs` has metadata but no cell and holds plain rows (Python's `input_schema()` lists none of them): it becomes the static `ADHESIVES` table, not an input. `selected: i64 = 1 => param("-", "Selected adhesive (code)", "1 = AA 326, 2 = EA 9514, 3 = 2214 Hi-Temp, 4 = DP460.", "Temperature design!C75").choices(&[(1, "AA 326"), (2, "EA 9514"), (3, "2214 Hi-Temp"), (4, "DP460")])`. Ranges:

| Path (`temperature.` omitted) | Default | min | max | step | log | A3 | Constraint |
|---|---|---|---|---|---|---|---|
| `duty.wheel_rotor_rpm` | 2000 | 100.0 | 6000.0 | 10.0 | | | |
| `duty.hot_ambient_C` | 55 | -20.0 | 80.0 | 0.5 | | | |
| `duty.driving_rise_C` | 10 | 0.0 | 60.0 | 0.5 | | yes | spans the E12 boundary (37.55 °C at defaults) |
| `duty.fault_trip_s` | 2 | 0.1 | 60.0 | 0.1 | yes | | |
| `duty.life_hours` | 20000 | 1000.0 | 100000.0 | 100.0 | | | > 0: divides the slip duty |
| `demag.hcj20_kA_m` | 1592 | 800.0 | 3000.0 | 1.0 | | | |
| `demag.beta_hcj_per_C` | -0.005 | -0.008 | -0.001 | 0.0001 | | yes | |
| `demag.knee_fraction` | 0.9 | 0.5 | 1.0 | 0.01 | | yes | |
| `demag.design_margin_C` | 10 | 0.0 | 40.0 | 0.5 | | yes | |
| `demag.h_rev_{aligned,pullout,likepole,single_ring}_kA_m` | 354, 791, 863, 569 | 0.0 | 1500.0 | 1.0 | | | |
| `mismatch.ndfeb_cte_per_C` | -0.8e-6 | -3e-6 | 6e-6 | 1e-8 | | | |
| `mismatch.adhesive_shear_modulus_GPa` | 0.55 | 0.01 | 3.0 | 0.001 | yes | | > 0: square root in the Volkersen λ |
| `mismatch.ndfeb_modulus_GPa` | 160 | 100.0 | 200.0 | 1.0 | | | |
| `mismatch.recommended_bondline_mm` | 0.1 | 0.01 | 0.5 | 0.005 | | | > 0: divides |
| `slip_loss.sigma_316_S_m` | 1.35e6 | 1e5 | 1e7 | 1000.0 | yes | | |
| `slip_loss.sigma_ndfeb_S_m` | 6.7e5 | 1e5 | 2e6 | 1000.0 | yes | | |
| `slip_loss.end_factor` | 0.7 | 0.0 | 1.0 | 0.01 | | | |
| `slip_loss.b_{hub,cup,sleeve,liner,magnet}_T` | 0.207, 0.214, 0.416, 0.419, 0.19 | 0.0 | 1.0 | 0.001 | | | |
| `slip_loss.cap_integral_T2m4` | 5.27e-10 | 1e-12 | 1e-8 | 1e-13 | yes | | |
| `slip_loss.web_integral_T2m2` | 1.035e-5 | 1e-7 | 1e-3 | 1e-8 | yes | | E5 default 4.14e-5 inside |
| `slip_loss.high_multiplier` | 3 | 1.0 | 10.0 | 0.1 | | | |
| `thermal.c_ndfeb` | 440 | 300.0 | 600.0 | 1.0 | | | |
| `thermal.c_316` | 500 | 300.0 | 700.0 | 1.0 | | | |
| `thermal.c_aluminium` | 900 | 700.0 | 1000.0 | 1.0 | | | |
| `thermal.conductance_W_K` | 0.3 | 0.01 | 5.0 | 0.001 | yes | yes | > 0: divides |
| `adhesive_life.hot_strength_retained` | 0.5 | 0.05 | 1.0 | 0.01 | | | |
| `adhesive_life.fatigue_endurance` | 0.2 | 0.02 | 0.6 | 0.005 | | | spans the E11 flip (0.106) |
| `adhesive_life.service_years` | 10 | 1.0 | 40.0 | 0.5 | | | |
| `adhesive_life.daily_swing_C` | 30 | 0.0 | 100.0 | 0.5 | | | |

Adhesive table (lines 78-88, same order and text) and the fallback:

```rust
/// One adhesive of the Temperature design!C65:C78 table (Python `AdhesiveCandidate`).
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct AdhesiveCandidate {
    pub name: &'static str,
    /// TDS service maximum or Tg − 20 °C [°C].
    pub design_limit_C: f64,
    /// Stress-free (cure) temperature [°C].
    pub cure_C: f64,
    /// TDS lap shear at about 22 °C [MPa].
    pub lap_shear_MPa: f64,
    pub role: &'static str,
    pub note: &'static str,
}

pub const ADHESIVES: [AdhesiveCandidate; 4] = [
    AdhesiveCandidate { name: "Loctite AA 326 + SF 7649", design_limit_C: 120.0, cure_C: 22.0, lap_shear_MPa: 15.0,
        role: "Recommended",
        note: "No-mix acrylic for magnet bonding; room-temperature cure; 0.10 mm bondline; service to 120 °C." },
    AdhesiveCandidate { name: "Loctite EA 9514", design_limit_C: 113.0, cure_C: 120.0, lap_shear_MPa: 45.0,
        role: "Alternative: more hot strength",
        note: "One-part toughened heat-cure epoxy; Tg 133 °C; cure 60 min at 120 °C (not 150 °C)." },
    AdhesiveCandidate { name: "3M Scotch-Weld 2214 Hi-Temp", design_limit_C: 177.0, cure_C: 121.0, lap_shear_MPa: 17.0,
        role: "Not recommended", note: "Rated to 177 °C but brittle (1 % elongation, 9 N/cm T-peel)." },
    AdhesiveCandidate { name: "3M Scotch-Weld DP460", design_limit_C: 60.0, cure_C: 23.0, lap_shear_MPa: 19.0,
        role: "Not recommended", note: "Room-temperature epoxy; T-peel collapses by 82 °C." },
];

/// What a selector code outside 1..=4 selects (decision D3). Python indexes
/// `candidates[selected - 1]`, so 0 silently picks the LAST adhesive; the port
/// never picks another adhesive: the results are NaN and the name says so.
pub const NO_ADHESIVE: AdhesiveCandidate = AdhesiveCandidate {
    name: "#N/A", design_limit_C: f64::NAN, cure_C: f64::NAN, lap_shear_MPa: f64::NAN, role: "", note: "",
};

/// The adhesive a selector code selects.
pub fn selected_adhesive(code: i64) -> &'static AdhesiveCandidate {
    match code {
        1..=4 => &ADHESIVES[(code - 1) as usize],
        _ => &NO_ADHESIVE,
    }
}
```

- [ ] **2b. Links.** Python lines 160-209, the same 47 fields in the same order:

```rust
/// Values the Temperature design sheet reads from other sheets (Python
/// `TemperatureLinks`), filled by `api::compute` exactly as Python's `compute_all`.
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct TemperatureLinks {
    pub op_temp_C: f64,             // Calculator C10
    pub npole: i64,                 // Calculator C5
    pub br20_T: f64,                // Calculator C21
    pub alpha_br: f64,              // Calibration C22
    pub tmax_lib_C: NumOrText,      // Calculator C22 ("n/a" for manual magnets)
    pub mu0: f64,                   // Calculator C43
    pub pullout_op_Nm: f64,         // Calculator C93
    pub pullout_20C_Nm: f64,        // Calculator C94
    pub inner_back_apothem_mm: f64, // Calculator C8
    pub inner_length_mm: f64,       // Calculator C18
    pub inner_width_mm: f64,        // Calculator C19
    pub inner_thickness_mm: f64,    // Calculator C20
    pub hub_wall_mm: f64,           // Calculator C38
    pub active_length_mm: f64,      // Calculator C33
    pub outer_back_apothem_mm: f64, // Calculator C60
    pub mass_magnets_g: f64,        // Calculator C110
    pub mass_cup_g: f64,            // Calculator C111
    pub mass_hub_g: f64,            // Calculator C112
    pub mass_boss_g: f64,           // Calculator C113
    pub slip_rpm: f64,              // Metal design C85
    pub slip_event_s: f64,          // Metal design C87
    pub life_events: f64,           // Metal design C88
    pub measured_drag_Nm: Option<f64>, // Metal design C90
    pub cold_high_Nm: f64,          // Metal design C10
    pub required_min_Nm: f64,       // Metal design C7
    pub variation: f64,             // Metal design C18
    pub min_temp_C: f64,            // Metal design C16
    pub magnetic_cycles: f64,       // Metal design C89
    pub bond_inner_mm: f64,         // Metal design C120
    pub bond_outer_mm: f64,         // Metal design C121
    pub sleeve_mm: f64,             // Metal design C25
    pub liner_mm: f64,              // Metal design C26
    pub sleeve_id_mm: f64,          // Metal design C175
    pub sleeve_od_mm: f64,          // Metal design C176
    pub liner_od_mm: f64,           // Metal design C177
    pub liner_id_mm: f64,           // Metal design C178
    pub cap_face_mm: f64,           // Metal design C167
    pub hardware_g: f64,            // Metal design C128
    pub retainers_g: f64,           // Metal design C46
    pub cap_g: f64,                 // Metal design C180
    pub endplates_g: f64,           // Metal design C181
    pub steel_sigma_S_m: f64,       // Materials C14
    pub steel_mu_r: f64,            // Materials C15
    pub steel_c: f64,               // Materials C16
    pub steel_cte: f64,             // Materials C17
    pub steel_E_GPa: f64,           // Materials C18
    pub al6061_sigma_S_m: f64,      // Materials C43
}
```

- [ ] **2c. Results.** Transcribe lines 213-395 as `results!` structs. Non-`f64` types: `SummaryResults.governing_note`, `torque_hot_day_note`, `verdict`: `String`; `SummaryResults.time_to_limit_high`: `NumOrText`; `DemagResults.tmax_lib_C`: `NumOrText` (annotated `float`, but receives `"n/a"`); `AdhesiveResults.selected_name`: `String` via `out_uncelled("", "Selected adhesive", "")`; `AdhesiveResults.fatigue_screen`, `MismatchResults.reading`, `MagnetLifeResults.torque_hot_day_check`, `AdhesiveLifeResults.hot_fatigue_screen`, `daily_screen`: `String`; `ThermalResults.time_to_limit_high`, `rotations_to_limit_high`, `time_to_limit_est`: `NumOrText`. `TemperatureResults` has `fields {}` and the ten groups in Python order.

- [ ] **2d. Helpers and `compute`.**

```rust
/// Temperature where the reverse field (scaling with Br) reaches the knee of the Hcj curve.
#[allow(non_snake_case)]
pub fn demag_onset_C(h_rev_kA_m: f64, hcj20: f64, beta: f64, knee: f64, alpha_br: f64, offset_C: f64) -> f64 {
    let hk = knee * hcj20;
    20.0 + (hk - h_rev_kA_m) / (hk * beta.abs() - h_rev_kA_m * alpha_br.abs()) - offset_C
}

/// Linear-elastic shear-lag peak shear at the ends of a bonded block from thermal mismatch.
#[allow(non_snake_case, clippy::too_many_arguments)] // Python names and signature
pub fn volkersen_peak_shear_MPa(
    G_GPa: f64, d_alpha: f64, dT: f64, bondline_mm: f64, magnet_E_GPa: f64, magnet_t_mm: f64,
    steel_E_GPa: f64, steel_t_mm: f64, bond_length_mm: f64,
) -> f64 {
    let G = G_GPa * 1e9;
    let lam = (G / (bondline_mm / 1000.0)
        * (1.0 / (magnet_E_GPa * 1e9 * magnet_t_mm / 1000.0) + 1.0 / (steel_E_GPa * 1e9 * steel_t_mm / 1000.0)))
        .sqrt();
    G * d_alpha * dT * (lam * bond_length_mm / 2000.0).tanh() / (lam * bondline_mm / 1000.0) / 1e6
}

#[allow(non_snake_case)] // Python names (T0, Ft, Fc, dT, L, C, G, Te, Th, Ee, Eh, ...)
pub fn compute(ti: &TemperatureInputs, k: &TemperatureLinks, _dev: Deviations) -> TemperatureResults {
    let npole = k.npole as f64; // integer overflow rule: arithmetic in f64
    // ---- duty
    let omega = k.slip_rpm * 2.0 * PI / 60.0;
    let pp = npole / 2.0;
    let f = pp * k.slip_rpm / 60.0;
    let we = 2.0 * PI * f;
    let T0 = ti.duty.hot_ambient_C + ti.duty.driving_rise_C;
    let duty = DutyResults {
        slip_rpm: k.slip_rpm, slip_rad_s: omega, pole_pairs: pp, field_freq_Hz: f, field_omega_rad_s: we,
        slip_event_s: k.slip_event_s, hot_day_start_C: T0,
    };

    // ---- demagnetization
    let d = &ti.demag;
    let h_ref = k.br20_T / (2.0 * k.mu0) / 1000.0;
    let t_ref = demag_onset_C(h_ref, d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, k.alpha_br, 0.0);
    // the workbook errors out when the magnet is not in the library; Python leaves the onset uncalibrated
    let offset = match k.tmax_lib_C {
        NumOrText::Num(tmax) => t_ref - tmax,
        NumOrText::Text(_) => 0.0,
    };
    let on = |h: f64| demag_onset_C(h, d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, k.alpha_br, offset);
    let (on_al, on_po, on_lp, on_cu) =
        (on(d.h_rev_aligned_kA_m), on(d.h_rev_pullout_kA_m), on(d.h_rev_likepole_kA_m), on(d.h_rev_single_ring_kA_m));
    let mag_lim = on_lp - d.design_margin_C;
    let thf = |T: f64| (1.0 + k.alpha_br * (T - 20.0)).powi(2);
    let demag = DemagResults {
        br20_T: k.br20_T, alpha_br: k.alpha_br, tmax_lib_C: k.tmax_lib_C, h_ref_kA_m: h_ref, t_ref_model_C: t_ref,
        calibration_offset_C: offset, onset_aligned_C: on_al, onset_pullout_C: on_po, onset_skipping_C: on_lp,
        onset_single_ring_C: on_cu, magnet_limit_C: mag_lim, torque_at_limit_Nm: k.pullout_20C_Nm * thf(mag_lim),
        torque_at_service_Nm: k.pullout_op_Nm,
    };

    // ---- adhesive selection and loads
    let sel = selected_adhesive(ti.adhesive.selected);
    let area = k.inner_length_mm * k.inner_width_mm;
    let m_block = k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 0.0075; // literal, as Python
    let r_mid = k.inner_back_apothem_mm + k.inner_thickness_mm / 2.0;
    let Ft = k.cold_high_Nm / (npole * r_mid / 1000.0);
    let tau_b = Ft / area;
    let Fc = m_block / 1000.0 * (ti.duty.wheel_rotor_rpm * 2.0 * PI / 60.0).powi(2) * r_mid / 1000.0;
    let fat = 0.2 * sel.lap_shear_MPa / tau_b;
    let adh = AdhesiveResults {
        selected_name: sel.name.to_owned(), design_limit_C: sel.design_limit_C, cure_C: sel.cure_C,
        lap_shear_MPa: sel.lap_shear_MPa, bond_area_mm2: area, block_mass_g: m_block, cold_high_torque_Nm: k.cold_high_Nm,
        inner_mid_radius_mm: r_mid, tangential_force_N: Ft, bond_shear_MPa: tau_b, centrifugal_force_N: Fc,
        static_ratio: sel.lap_shear_MPa / tau_b, shear_reversals: k.magnetic_cycles,
        fatigue_screen: if fat >= 4.0 { format!("OK: {}x margin", text0(fat)) } else { "CHECK".to_owned() },
    };

    let gov = py_min(mag_lim, sel.design_limit_C);

    // ---- thermal mismatch screen
    let mm = &ti.mismatch;
    let dT = py_max(sel.cure_C - k.min_temp_C, gov - sel.cure_C);
    let d_alpha = k.steel_cte - mm.ndfeb_cte_per_C;
    let s1 = volkersen_peak_shear_MPa(mm.adhesive_shear_modulus_GPa, d_alpha, dT, k.bond_inner_mm, mm.ndfeb_modulus_GPa,
        k.inner_thickness_mm, k.steel_E_GPa, k.hub_wall_mm, k.inner_length_mm);
    let s2 = volkersen_peak_shear_MPa(mm.adhesive_shear_modulus_GPa, d_alpha, dT, mm.recommended_bondline_mm,
        mm.ndfeb_modulus_GPa, k.inner_thickness_mm, k.steel_E_GPa, k.hub_wall_mm, k.inner_length_mm);
    let mis = MismatchResults {
        steel_cte: k.steel_cte, steel_E_GPa: k.steel_E_GPa, steel_thickness_mm: k.hub_wall_mm, cold_limit_C: k.min_temp_C,
        worst_swing_C: dT, current_bondline_mm: k.bond_inner_mm, peak_shear_current_MPa: s1, peak_shear_recommended_MPa: s2,
        reading: if s1 > sel.lap_shear_MPa { "Above the lap-shear strength at the block ends" } else { "Below the lap-shear strength" }.to_owned(),
    };

    // ---- slip losses (estimates)
    let sl = &ti.slip_loss;
    let delta = (2.0 / (we * k.mu0 * k.steel_mu_r * k.steel_sigma_S_m)).sqrt() * 1000.0; // mm
    let L = k.active_length_mm / 1000.0;
    let r_hub = (k.inner_back_apothem_mm - k.bond_inner_mm) / 1000.0;
    let r_cup = (k.outer_back_apothem_mm + k.bond_outer_mm) / 1000.0;
    let surface = |B: f64, r: f64| {
        k.steel_sigma_S_m * we.powi(2) * B.powi(2) * (delta / 1000.0) / (4.0 * (pp / r).powi(2)) * 2.0 * PI * r * L
    };
    let p_hub = surface(sl.b_hub_T, r_hub);
    let p_cup = surface(sl.b_cup_T, r_cup);
    let p_web = k.steel_sigma_S_m * we.powi(2) * (delta / 1000.0) / 4.0 * ((r_mid / 1000.0) / pp).powi(2) * sl.web_integral_T2m2;
    let r_s = (k.sleeve_id_mm + k.sleeve_od_mm) / 4.0 / 1000.0;
    let r_l = (k.liner_od_mm + k.liner_id_mm) / 4.0 / 1000.0;
    let shell = |t_mm: f64, r: f64, B: f64| {
        sl.end_factor * sl.sigma_316_S_m * (t_mm / 1000.0) * (omega * r).powi(2) * B.powi(2) / 2.0 * 2.0 * PI * r * L
    };
    let p_slv = shell(k.sleeve_mm, r_s, sl.b_sleeve_T);
    let p_lin = shell(k.liner_mm, r_l, sl.b_liner_T);
    let p_cap = sl.end_factor * k.al6061_sigma_S_m * (k.cap_face_mm / 1000.0) * omega.powi(2) * sl.cap_integral_T2m4;
    let p_mag = sl.sigma_ndfeb_S_m * we.powi(2) * sl.b_magnet_T.powi(2) * (k.inner_width_mm / 1000.0).powi(2) / 24.0
        * (k.inner_length_mm * k.inner_width_mm * k.inner_thickness_mm * 1e-9) * 2.0 * npole;
    let p_tot = p_hub + p_cup + p_web + p_slv + p_lin + p_cap + p_mag;
    // Python: measured = isinstance(drag, (int, float)); p_use = drag * omega if measured else p_tot;
    // p_hi = p_use if measured else p_use * high_multiplier
    let (p_use, p_hi) = match k.measured_drag_Nm {
        Some(drag) => (drag * omega, drag * omega),
        None => (p_tot, p_tot * sl.high_multiplier),
    };
    let loss = SlipLossResults {
        steel_sigma_S_m: k.steel_sigma_S_m, steel_mu_r: k.steel_mu_r, cap_sigma_S_m: k.al6061_sigma_S_m,
        skin_depth_mm: delta, hub_W: p_hub, cup_W: p_cup, web_W: p_web, sleeve_W: p_slv, liner_W: p_lin, cap_W: p_cap,
        magnets_W: p_mag, total_W: p_tot, drag_Nm: p_tot / omega, used_W: p_use, high_W: p_hi,
    };

    // ---- thermal network
    let th = &ti.thermal;
    let C = (k.mass_magnets_g * th.c_ndfeb
        + (k.mass_cup_g + k.mass_hub_g + k.mass_boss_g + k.hardware_g) * k.steel_c
        + (k.retainers_g + k.endplates_g) * th.c_316
        + k.cap_g * th.c_aluminium)
        / 1000.0;
    let G = th.conductance_W_K;
    let tau_th = C / G;
    let (rise_e, rise_h) = (p_use / G, p_hi / G);
    let (Te, Th) = (T0 + rise_e, T0 + rise_h);
    let t_lim_h = if Th <= gov { NumOrText::Text(NEVER) } else { NumOrText::Num(-tau_th * (1.0 - (gov - T0) / rise_h).ln()) };
    let t_lim_e = if Te <= gov { NumOrText::Text(NEVER) } else { NumOrText::Num(-tau_th * (1.0 - (gov - T0) / rise_e).ln()) };
    let rev_s = k.slip_rpm / 60.0;
    let (ke, kh) = (p_use / C, p_hi / C);
    let T_fault = T0 + rise_h * (1.0 - (-ti.duty.fault_trip_s / tau_th).exp());
    let thermal = ThermalResults {
        steel_c: k.steel_c, heat_capacity_J_K: C, time_constant_s: tau_th, start_C: T0,
        rise_per_event_C: p_use * k.slip_event_s / C, steady_rise_est_C: rise_e, steady_rise_high_C: rise_h,
        steady_est_C: Te, steady_high_C: Th, time_to_limit_high: t_lim_h,
        rotations_to_limit_high: match t_lim_h {
            NumOrText::Num(t) => NumOrText::Num(t * rev_s),
            NumOrText::Text(_) => NumOrText::Text(NEVER_SHORT),
        },
        time_to_limit_est: t_lim_e, critical_drag_Nm: (gov - T0) * G / omega, heating_rate_est_C_s: ke,
        heating_rate_high_C_s: kh, rev_per_C_est: rev_s / ke, rev_per_C_high: rev_s / kh, rev_per_tau: tau_th * rev_s,
        t95_s: 3.0 * tau_th, rev95: 3.0 * tau_th * rev_s, temp_at_fault_C: T_fault,
    };

    // ---- slip life
    let rpe = rev_s * k.slip_event_s;
    let rot = k.life_events * rpe;
    let hrs = k.life_events * k.slip_event_s / 3600.0;
    let (Ee, Eh) = (p_use * k.slip_event_s, p_hi * k.slip_event_s);
    let (dTe, dTh) = (Ee / C, Eh / C);
    let duty_frac = hrs / ti.duty.life_hours;
    let life = SlipLifeResults {
        events: k.life_events, rev_per_event: rpe, rotations: rot, slip_hours: hrs, like_pole_passes: rot * pp,
        heat_per_event_est_J: Ee, heat_per_event_high_J: Eh, rise_per_event_est_C: dTe, rise_per_event_high_C: dTh,
        life_heat_high_MJ: k.life_events * Eh / 1e6, slip_duty: duty_frac, avg_rise_est_C: duty_frac * rise_e,
        avg_rise_high_C: duty_frac * rise_h, rise_per_pct_duty_C: 0.01 * rise_h,
    };

    // ---- magnet life
    let peak = py_max(T_fault, T0 + dTh) + duty_frac * rise_h;
    let tq_hot = k.pullout_20C_Nm * thf(T0);
    let tq_peak = k.pullout_20C_Nm * thf(peak);
    let mlife = MagnetLifeResults {
        peak_C: peak, margin_onset_C: on_lp - peak, margin_limit_C: mag_lim - peak, torque_hot_day_Nm: tq_hot,
        torque_hot_day_check: if tq_hot >= k.required_min_Nm { "Meets it nominally (no variation allowance)" } else { "Below it" }.to_owned(),
        torque_peak_Nm: tq_peak,
    };

    // ---- adhesive life
    let al = &ti.adhesive_life;
    let tq_var = tq_peak * (1.0 + k.variation);
    let amp = tq_var / (npole * r_mid / 1000.0) / area;
    let hot_fm = sel.lap_shear_MPa * al.hot_strength_retained * al.fatigue_endurance / amp;
    let daily = s2 * al.daily_swing_C / dT;
    let alife = AdhesiveLifeResults {
        peak_C: peak, margin_C: sel.design_limit_C - peak, reversals: rot * pp, torque_peak_var_Nm: tq_var,
        shear_amplitude_MPa: amp, hot_fatigue_margin: hot_fm,
        hot_fatigue_screen: if hot_fm >= 4.0 { "OK" } else { "CHECK: get hot fatigue data" }.to_owned(),
        daily_cycles: al.service_years * 365.0, daily_peak_shear_MPa: daily,
        daily_screen: if daily < sel.lap_shear_MPa * al.fatigue_endurance { "Below the fatigue endurance" } else { "Above the fatigue endurance: qualify by thermal cycling" }.to_owned(),
    };

    // ---- summary
    let cure_margin = on_cu - sel.cure_C;
    let margin_hot = gov - T0;
    let ok = margin_hot > 0.0 && (mag_lim - peak) > 0.0 && (sel.design_limit_C - peak) > 0.0 && cure_margin >= 10.0;
    let summary = SummaryResults {
        service_max_C: k.op_temp_C, onset_aligned_C: on_al, onset_pullout_C: on_po, onset_skipping_C: on_lp,
        magnet_limit_C: mag_lim, adhesive_limit_C: sel.design_limit_C, governing_limit_C: gov,
        governing_note: if mag_lim <= sel.design_limit_C { "Magnets govern (skipping case)." } else { "Adhesive governs." }.to_owned(),
        margin_service_C: gov - k.op_temp_C, hot_day_start_C: T0, margin_hot_day_C: margin_hot, torque_hot_day_Nm: tq_hot,
        torque_hot_day_note: mlife.torque_hot_day_check.clone(), steady_estimate_C: Te, steady_high_C: Th,
        time_to_limit_high: t_lim_h, peak_with_fault_C: peak, life_rotations: rot, avg_slip_heating_high_C: duty_frac * rise_h,
        critical_drag_Nm: (gov - T0) * G / omega, cure_margin_C: cure_margin,
        verdict: if ok { "OK on temperature. Confirm drag torque and thermal cycling by test." } else { "CHECK: see the rows above." }.to_owned(),
    };
    TemperatureResults {
        summary, duty, demag, adhesive: adh, mismatch: mis, slip_loss: loss, thermal, slip_life: life,
        magnet_life: mlife, adhesive_life: alife,
    }
}
```

Unit tests in `temperature.rs`: the adhesive fallback, and the equality edges of every threshold comparison (architecture section 7 step 8). Four edges have a constant or a table value on the right (`fat >= 4`, `hot_fm >= 4`, `s1 > lap`, `cure_margin >= 10`): their inputs are chosen so every step is exact, or were found by a search on a scratch build of this code (`8.83111640173109e-6`, `666.3586306326292`); the other edges copy the left side into an input it does not read. Every test asserts the equality before the branch, and every comparison was checked by mutation (swapping `>=`/`>` or `<=`/`<` makes its test fail). Inputs that a correction changes by default (E1: the adhesive shear modulus) are set explicitly.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_adhesive_covers_the_four_codes_only() {
        for (code, name) in [(1, "Loctite AA 326 + SF 7649"), (4, "3M Scotch-Weld DP460")] {
            assert_eq!(selected_adhesive(code).name, name);
        }
        for bad in [0, 5, -1, i64::MIN, i64::MAX] {
            let none = selected_adhesive(bad);
            assert_eq!(none.name, "#N/A", "{bad}");
            assert!(none.lap_shear_MPa.is_nan() && none.design_limit_C.is_nan(), "{bad}");
        }
    }

    /// The default design's links (`api::compute` at the workbook defaults, long decimals
    /// rounded). Each equality test below changes only the fields it names.
    fn links() -> TemperatureLinks {
        TemperatureLinks {
            op_temp_C: 50.0, npole: 10, br20_T: 1.29, alpha_br: -0.0012, tmax_lib_C: NumOrText::Num(150.0),
            mu0: 1.256637e-6, pullout_op_Nm: 2.6473, pullout_20C_Nm: 2.8487, inner_back_apothem_mm: 10.15,
            inner_length_mm: 12.7, inner_width_mm: 6.35, inner_thickness_mm: 3.17, hub_wall_mm: 5.1,
            active_length_mm: 12.7, outer_back_apothem_mm: 17.89, mass_magnets_g: 38.347, mass_cup_g: 60.754,
            mass_hub_g: 25.81, mass_boss_g: 30.778, slip_rpm: 2000.0, slip_event_s: 0.1, life_events: 2e7,
            measured_drag_Nm: None, cold_high_Nm: 3.7647, required_min_Nm: 2.5, variation: 0.15, min_temp_C: -40.0,
            magnetic_cycles: 3.3333e8, bond_inner_mm: 0.05, bond_outer_mm: 0.05, sleeve_mm: 0.1, liner_mm: 0.2,
            sleeve_id_mm: 27.436, sleeve_od_mm: 27.636, liner_od_mm: 29.39, liner_id_mm: 28.99, cap_face_mm: 0.8,
            hardware_g: 6.0, retainers_g: 3.131, cap_g: 2.322, endplates_g: 6.653, steel_sigma_S_m: 4.5e6,
            steel_mu_r: 200.0, steel_c: 473.0, steel_cte: 12.3e-6, steel_E_GPa: 205.0, al6061_sigma_S_m: 2.5e7,
        }
    }

    fn run(ti: &TemperatureInputs, k: &TemperatureLinks) -> TemperatureResults {
        compute(ti, k, Deviations::NONE)
    }

    #[test]
    fn fatigue_screens_pass_at_a_margin_of_exactly_four() {
        // `fat >= 4` and `hot_fm >= 4`, with values that make every step exact:
        // npole * r_mid / 1000 = 10 * 100 / 1000 = 1 and a 1 mm² bond area.
        let mut k = links();
        k.inner_back_apothem_mm = 99.0;
        k.inner_thickness_mm = 2.0; // r_mid = 99 + 2 / 2 = 100 mm
        k.inner_length_mm = 1.0;
        k.inner_width_mm = 1.0;
        k.cold_high_Nm = 0.75; // tau_b = 0.75 MPa: fat = 0.2 * 15 / 0.75 = 4 (AA 326, 15 MPa)
        k.alpha_br = 0.0; // every temperature factor is exactly 1: amp = pullout_20C * (1 + variation)
        k.variation = 0.0;
        k.pullout_20C_Nm = 0.9375;
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        ti.adhesive_life.hot_strength_retained = 0.5;
        ti.adhesive_life.fatigue_endurance = 0.5; // hot_fm = 15 * 0.5 * 0.5 / 0.9375 = 4
        let r = run(&ti, &k);
        assert_eq!((r.adhesive.bond_shear_MPa, r.adhesive_life.hot_fatigue_margin), (0.75, 4.0));
        assert_eq!(r.adhesive.fatigue_screen, "OK: 4x margin");
        assert_eq!(r.adhesive_life.hot_fatigue_screen, "OK");
        k.cold_high_Nm = 0.7500001; // both margins just under 4
        k.pullout_20C_Nm = 0.9375001;
        let r = run(&ti, &k);
        assert_eq!(r.adhesive.fatigue_screen, "CHECK");
        assert_eq!(r.adhesive_life.hot_fatigue_screen, "CHECK: get hot fatigue data");
    }

    #[test]
    fn mismatch_reading_is_below_at_equal_shear() {
        // `s1 > lap` is strict. A 1 m bond length saturates tanh to exactly 1.0 (no
        // libm-dependent digits); the NdFeB CTE was found by search so that the peak
        // shear is exactly AA 326's 15 MPa lap shear (the first assert proves it).
        let mut k = links();
        k.inner_length_mm = 1000.0;
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        ti.mismatch.adhesive_shear_modulus_GPa = 0.55; // explicit: correction E1 changes the default
        ti.mismatch.ndfeb_cte_per_C = 8.83111640173109e-6;
        let r = run(&ti, &k);
        assert_eq!(r.mismatch.peak_shear_current_MPa, 15.0);
        assert_eq!(r.mismatch.reading, "Below the lap-shear strength");
    }

    #[test]
    fn limits_are_never_reached_when_the_steady_state_equals_the_limit() {
        // `Th <= gov` and `Te <= gov`. A bench drag of 0 puts both steady states at the
        // hot-day start (50 + 10 = 60 °C); DP460's 60 °C limit governs.
        let mut k = links();
        k.measured_drag_Nm = Some(0.0);
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 4;
        ti.duty.hot_ambient_C = 50.0;
        ti.duty.driving_rise_C = 10.0;
        let r = run(&ti, &k);
        assert_eq!((r.thermal.steady_est_C, r.thermal.steady_high_C, r.summary.governing_limit_C), (60.0, 60.0, 60.0));
        assert_eq!(r.thermal.time_to_limit_high, NumOrText::Text(NEVER));
        assert_eq!(r.thermal.time_to_limit_est, NumOrText::Text(NEVER));
        assert_eq!(r.thermal.rotations_to_limit_high, NumOrText::Text(NEVER_SHORT));
    }

    #[test]
    fn hot_day_torque_meets_the_minimum_at_equality() {
        // `tq_hot >= required_min`; the hot-day torque does not read the required minimum.
        let ti = TemperatureInputs::default();
        let mut k = links();
        k.required_min_Nm = run(&ti, &k).magnet_life.torque_hot_day_Nm;
        let r = run(&ti, &k);
        assert_eq!(r.magnet_life.torque_hot_day_Nm, k.required_min_Nm);
        assert_eq!(r.magnet_life.torque_hot_day_check, "Meets it nominally (no variation allowance)");
        assert_eq!(r.summary.torque_hot_day_note, "Meets it nominally (no variation allowance)");
    }

    #[test]
    fn magnets_govern_when_the_limits_are_equal() {
        // `mag_lim <= design_limit`. mag_lim = onset - margin; with margin = onset - 120 both
        // subtractions are exact (the onset, about 102.6 °C, lies within a factor 2 of 120),
        // so the magnet limit is exactly AA 326's 120 °C (the first assert proves it).
        let k = links();
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 1;
        ti.demag.design_margin_C = run(&ti, &k).demag.onset_skipping_C - 120.0;
        let r = run(&ti, &k);
        assert_eq!(r.summary.magnet_limit_C, 120.0);
        assert_eq!(r.summary.governing_note, "Magnets govern (skipping case).");
    }

    #[test]
    fn verdict_accepts_a_cure_margin_of_exactly_ten() {
        // `cure_margin >= 10`, the other three margins positive. EA 9514 cures at 120 °C; the
        // single-ring field was found by search so that its onset is exactly 130 °C (the first
        // assert proves it).
        let k = links();
        let mut ti = TemperatureInputs::default();
        ti.adhesive.selected = 2;
        ti.demag.h_rev_single_ring_kA_m = 666.3586306326292;
        let r = run(&ti, &k);
        assert_eq!(r.summary.cure_margin_C, 10.0);
        assert_eq!(r.summary.verdict, "OK on temperature. Confirm drag torque and thermal cycling by test.");
        ti.demag.h_rev_single_ring_kA_m = 667.0; // a lower onset: the margin drops under 10
        assert_eq!(run(&ti, &k).summary.verdict, "CHECK: see the rows above.");
    }
}
```
If a searched value no longer gives exact equality (its first assert fails), the operation order of `volkersen_peak_shear_MPa` or `demag_onset_C` drifted from Python's: fix the port, never the constant.

- [ ] **2e. Wire into `api.rs`.** `mod.rs`: `pub mod temperature;`. Add `temperature: TemperatureInputs` after `materials` in `DesignInputs`, `temperature: TemperatureResults` after `materials` in `DesignResults`. After `let matr = ...` (Python lines 194-209):

```rust
    let links = temperature::TemperatureLinks {
        op_temp_C: ci.op_temp_C, npole: ci.npole, br20_T: m.inner_br_T, alpha_br: cal_in.alpha_br_per_C,
        tmax_lib_C: m.inner_tmax_C, mu0: ci.mu0, pullout_op_Nm: m.pullout_Nm, pullout_20C_Nm: m.pullout_20C_Nm,
        inner_back_apothem_mm: ci.inner_back_apothem_mm, inner_length_mm: m.inner_length_mm,
        inner_width_mm: m.inner_width_mm, inner_thickness_mm: m.inner_thickness_mm, hub_wall_mm: m.hub_wall_mm,
        active_length_mm: m.active_length_mm, outer_back_apothem_mm: m.outer_back_apothem_mm,
        mass_magnets_g: mass.magnets_g, mass_cup_g: mass.cup_g, mass_hub_g: mass.hub_g, mass_boss_g: mass.boss_g,
        slip_rpm: md.slip_rpm, slip_event_s: md.slip_event_s, life_events: md.life_events,
        measured_drag_Nm: md.measured_drag_Nm, cold_high_Nm: mdr.torque_cold_high_Nm, required_min_Nm: md.required_min_Nm,
        variation: md.variation, min_temp_C: md.min_temp_C, magnetic_cycles: mdr.magnetic_cycles,
        bond_inner_mm: md.bond_inner_mm, bond_outer_mm: md.bond_outer_mm, sleeve_mm: md.sleeve_mm, liner_mm: md.liner_mm,
        sleeve_id_mm: ret.sleeve_id_mm, sleeve_od_mm: ret.sleeve_od_mm, liner_od_mm: ret.liner_od_mm,
        liner_id_mm: ret.liner_id_mm, cap_face_mm: md.cap_axial_mm, hardware_g: md.hardware_g,
        retainers_g: ret.retainers_g, cap_g: ret.cap_g, endplates_g: ret.endplates_g,
        steel_sigma_S_m: mat_in.steel.conductivity_S_m, steel_mu_r: mat_in.steel.mu_r_incremental,
        steel_c: mat_in.steel.specific_heat_J_kgK, steel_cte: mat_in.steel.cte_per_C,
        steel_E_GPa: mat_in.steel.modulus_GPa, al6061_sigma_S_m: materials::AL6061.conductivity_S_m,
    };
    let temp = temperature::compute(&inputs.temperature, &links, dev);
```

- [ ] **2f. Review Focus tests.** Create `magcoupling-rs/tests/robustness.rs`:

```rust
//! Inputs no parity or differential case holds (the plan's Review Focus): a
//! selector code outside its choices set on the struct, a measured drag of
//! exactly zero, extreme typed values. The engine must never panic.

use magcoupling::engine::api::{DesignInputs, compute_all_with};
use magcoupling::engine::deviations::Deviations;
use magcoupling::engine::meta::NumOrText;

#[test]
fn an_invalid_adhesive_code_selects_no_adhesive() {
    for code in [0, 5, -1, i64::MIN, i64::MAX] {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.temperature.adhesive.selected = code; // bypasses set(), which refuses it
        let t = compute_all_with(&inputs, Deviations::NONE).temperature;
        // Python's candidates[selected - 1] would silently take DP460 for code 0.
        assert_eq!(t.adhesive.selected_name, "#N/A", "{code}");
        assert!(t.adhesive.lap_shear_MPa.is_nan() && t.summary.adhesive_limit_C.is_nan(), "{code}");
    }
}

#[test]
fn zero_measured_drag_does_not_panic() {
    // Python raises ZeroDivisionError (rotations per degree, C156/C157); the limit is +inf.
    for dev in [Deviations::NONE, Deviations::ALL] {
        let mut inputs = DesignInputs::defaults_with(dev);
        inputs.metal.measured_drag_Nm = Some(0.0);
        let t = compute_all_with(&inputs, dev).temperature;
        assert_eq!(t.thermal.rev_per_C_est, f64::INFINITY);
        assert_eq!(t.thermal.rev_per_C_high, f64::INFINITY);
        assert!(t.thermal.heat_capacity_J_K.is_finite() && t.summary.governing_limit_C.is_finite());
        assert_eq!(t.thermal.time_to_limit_high, NumOrText::Text("never: steady state stays below the limit"));
    }
}
```

- [ ] **2g. Generator, static data, differential test.** `MODULES["temperature"] = ["coupling", "metal", "calibration", "materials", "temperature"]`; `PROBES["temperature"]`:

```python
    "temperature": [
        ("hot-day start above the limit", {"temperature.duty.driving_rise_C": 40.0}),
        ("bench drag entered", {"metal.measured_drag_Nm": 0.05}),
        ("bench drag at the slider minimum", {"metal.measured_drag_Nm": 0.001}),
        ("manual inner magnet: uncalibrated onset", {"coupling.magnets.part_inner": ""}),
        ("low conductance: steady state above the limit", {"temperature.thermal.conductance_W_K": 0.02}),
    ],
```
`static_data()`: `"adhesives": [dataclasses.asdict(a) for a in py_temperature.default_adhesives()]` (import `from magcoupling import temperature as py_temperature`). `tests/static_data.rs`:

```rust
#[test]
fn adhesives_equal_the_python_candidates() {
    use magcoupling::engine::temperature::ADHESIVES;
    let doc = python_static();
    let rows = doc["adhesives"].as_array().expect("adhesives");
    assert_eq!(rows.len(), ADHESIVES.len());
    for (py, rs) in rows.iter().zip(ADHESIVES.iter()) {
        assert_eq!(rs.name, text(py, "name"));
        assert_eq!((rs.role, rs.note), (text(py, "role").as_str(), text(py, "note").as_str()), "{}", rs.name);
        for (key, value) in [("design_limit_C", rs.design_limit_C), ("cure_C", rs.cure_C), ("lap_shear_MPa", rs.lap_shear_MPa)] {
            assert_eq!(value, number(py, key), "{}.{key}", rs.name);
        }
    }
}
```
`tests/differential.rs`: `#[test] fn temperature_matches_python_on_every_case() { check_module("temperature"); }` and `BRANCHES`:

```rust
    ("temperature.summary.governing_note", &[Text("Magnets govern (skipping case)."), Text("Adhesive governs.")]),
    ("temperature.summary.torque_hot_day_note", &[Text("Meets it nominally (no variation allowance)"), Text("Below it")]),
    ("temperature.summary.time_to_limit_high", &[Number, Text("never: steady state stays below the limit")]),
    ("temperature.summary.verdict", &[Text("OK on temperature. Confirm drag torque and thermal cycling by test."), Text("CHECK: see the rows above.")]),
    ("temperature.demag.tmax_lib_C", &[Number, Text("n/a")]),
    ("temperature.adhesive.selected_name", &[Text("Loctite AA 326 + SF 7649"), Text("Loctite EA 9514"), Text("3M Scotch-Weld 2214 Hi-Temp"), Text("3M Scotch-Weld DP460")]),
    ("temperature.adhesive.fatigue_screen", &[Prefix("OK: "), Text("CHECK")]),
    ("temperature.mismatch.reading", &[Text("Above the lap-shear strength at the block ends"), Text("Below the lap-shear strength")]),
    ("temperature.thermal.time_to_limit_high", &[Number, Text("never: steady state stays below the limit")]),
    ("temperature.thermal.rotations_to_limit_high", &[Number, Text("never")]),
    ("temperature.thermal.time_to_limit_est", &[Number, Text("never: steady state stays below the limit")]),
    ("temperature.magnet_life.torque_hot_day_check", &[Text("Meets it nominally (no variation allowance)"), Text("Below it")]),
    ("temperature.adhesive_life.hot_fatigue_screen", &[Text("OK"), Text("CHECK: get hot fatigue data")]),
    ("temperature.adhesive_life.daily_screen", &[Text("Below the fatigue endurance"), Text("Above the fatigue endurance: qualify by thermal cycling")]),
```
If `every_branch_is_reached` names a branch no case reaches, find an input set that reaches it with the Python engine (`set_input` + `compute_all` in the oracle venv), add it to `PROBES["temperature"]` with a tag saying which branch it pins, and regenerate. Bless and regenerate:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test schema
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote .../differential/temperature.json (...)` (about 410 cases), no traceback, file under 4 MB.

- [ ] **Step 3: Run parity and differential**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. Parity: 297 result cells (330 minus clamps' 33) and 135 default inputs (160 minus clamps' 25); Temperature design!C12 = 92.5500498645358, C19 = "never: steady state stays below the limit", C25 = "OK on temperature. Confirm drag torque and thermal cycling by test.", F12 = "Magnets govern (skipping case)."; `tests/robustness.rs` 2 passed; the seven `temperature::tests` pass (the adhesive fallback and six equality-edge tests).

Mutation spot check: replace `py_min(mag_lim, sel.design_limit_C)` with `py_max(...)`: parity FAILS on Temperature design!C12 and many downstream cells. Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task8.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task8.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status adds "Temperature design"; tests table gains `tests/robustness.rs`; `docs/ai/03-structure.yaml` `ported` adds `temperature.rs`, `tests` lists `robustness.rs`.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the Temperature design sheet

temperature.py: 7 input groups (37 cells), 10 result groups (130 cells and
the uncelled adhesive name), the links from the other sheets, the adhesive
table as static data. A selector code outside 1-4 selects no adhesive
(NaN, "#N/A") instead of Python's silent last row; a zero bench drag gives
+inf rotations per degree instead of an exception.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 9: Table rows in the metadata model

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

The screw table and the two sweeps are lists of rows whose cells follow a layout (test_parity.py: `TABLE_COLUMNS`/`TABLE_ROWS`, `SWEEP_COLUMNS`). This task adds them to the metadata model and the harness with a toy table, so Tasks 10 and 11 only declare rows.

**Files:**
- Modify: `magcoupling-rs/src/engine/meta.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/parity.rs`, `tests/deviations.rs`, `tests/schema.rs`, `tests/python_schema.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py` (table layouts in `python_schema.json`)
- Regenerate: `tests/data/python_schema.json`
- Modify: `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `ResultMeta`, `ResultSet`, `results!` (slice).
- Produces:
  - `pub enum CellAxis { Column(&'static str), Row(u32), None }`, `pub struct ColumnMeta { pub meta: ResultMeta, pub axis: CellAxis }`, builders `pub const fn col(unit, label, help, column: &'static str) -> ColumnMeta`, `pub const fn at_row(unit, label, help, row: u32) -> ColumnMeta`, `pub const fn uncelled_col(unit, label, help) -> ColumnMeta`.
  - `pub enum TableLayout { RowsDown { sheet: &'static str, first_row: u32 }, ColumnsAcross { sheet: &'static str, columns: &'static [&'static str] } }` with `pub fn cell(&self, axis: CellAxis, index: usize) -> Option<String>`.
  - `pub type ResultVisitor<'a> = dyn FnMut(&str, &'static ResultMeta, Option<&str>, Value) + 'a;` (the visit callback; a named type because clippy's `type_complexity` rejects the spelled-out form under `-D warnings`).
  - `pub trait RowSet { const COLUMNS: &'static [ColumnMeta]; fn visit_row(&self, prefix: &str, layout: &TableLayout, index: usize, f: &mut ResultVisitor<'_>); }` and macro `rows!`.
  - `results!` accepts an optional `tables { name: Row => layout_expr, }` section after `groups`; the field is `Vec<Row>`; paths `name[i].field`.
  - Changed: `ResultSet::visit(&self, prefix: &str, f: &mut ResultVisitor<'_>)` (the cell is now an argument); `ResultRow { path, meta, cell: Option<String>, value }`.
  - Harness: `PortedResults { group, cells, table_cells }`; `pub fn is_table_path(path: &str) -> bool`; `cell_values` in `tests/deviations.rs` returns `BTreeMap<String, Value>`; new tests `table_columns_match_the_workbook_headers` (schema.rs) and `tables_match_the_python_layout` (python_schema.rs).

- [ ] **Step 1: Write the failing unit tests.** Append inside `#[cfg(test)] mod tests` of `meta.rs`:

```rust
    rows! {
        /// A toy screw-table row: one sheet column per row, fields name their sheet row.
        pub struct ToyRow {
            size: String => uncelled_col("", "Size", ""),
            d_mm: f64 => at_row("mm", "Diameter", "", 6),
            ok: i64 => at_row("-", "Fits", "", 7),
        }
    }

    rows! {
        /// A toy sweep row: one sheet row per row, fields name their column.
        pub struct SweepToy {
            x: f64 => col("mm", "X", "", "B"),
            status: String => col("", "Status", "", "AA"),
        }
    }

    const TOY_COLUMNS: [&str; 2] = ["C", "D"];

    results! {
        pub struct ToyOut {
            fields {
                n: f64 => out("-", "N", "", "S!C1"),
            }
            tables {
                table: ToyRow => TableLayout::ColumnsAcross { sheet: "Toy", columns: &TOY_COLUMNS },
                sweep: SweepToy => TableLayout::RowsDown { sheet: "Sweep", first_row: 6 },
            }
        }
    }

    #[test]
    fn table_rows_get_synthesized_cells() {
        let out = ToyOut {
            n: 1.0,
            table: vec![
                ToyRow { size: "M3".into(), d_mm: 3.0, ok: 1 },
                ToyRow { size: "M4".into(), d_mm: 4.0, ok: 0 },
            ],
            sweep: vec![SweepToy { x: 0.5, status: "a".into() }, SweepToy { x: 0.75, status: "b".into() }],
        };
        let rows: Vec<(String, Option<String>, Value)> =
            result_rows(&out).into_iter().map(|r| (r.path, r.cell, r.value)).collect();
        let s = |x: &str| x.to_owned();
        assert_eq!(
            rows,
            vec![
                (s("n"), Some(s("S!C1")), Value::Num(1.0)),
                (s("table[0].size"), None, Value::Text(s("M3"))),
                (s("table[0].d_mm"), Some(s("Toy!C6")), Value::Num(3.0)),
                (s("table[0].ok"), Some(s("Toy!C7")), Value::Int(1)),
                (s("table[1].size"), None, Value::Text(s("M4"))),
                (s("table[1].d_mm"), Some(s("Toy!D6")), Value::Num(4.0)),
                (s("table[1].ok"), Some(s("Toy!D7")), Value::Int(0)),
                (s("sweep[0].x"), Some(s("Sweep!B6")), Value::Num(0.5)),
                (s("sweep[0].status"), Some(s("Sweep!AA6")), Value::Text(s("a"))),
                (s("sweep[1].x"), Some(s("Sweep!B7")), Value::Num(0.75)),
                (s("sweep[1].status"), Some(s("Sweep!AA7")), Value::Text(s("b"))),
            ]
        );
        assert_eq!(ToyRow::COLUMNS[1].meta.name, "d_mm");
        assert_eq!(ToyRow::COLUMNS[2].meta.ty, FieldType::I64);
    }

    #[test]
    fn a_layout_gives_no_cell_to_a_mismatched_axis_or_a_row_past_its_columns() {
        let across = TableLayout::ColumnsAcross { sheet: "T", columns: &TOY_COLUMNS };
        assert_eq!(across.cell(CellAxis::Row(6), 1), Some("T!D6".to_owned()));
        assert_eq!(across.cell(CellAxis::Column("B"), 0), None);
        assert_eq!(across.cell(CellAxis::Row(6), 2), None, "only two columns");
        let down = TableLayout::RowsDown { sheet: "S", first_row: 6 };
        assert_eq!(down.cell(CellAxis::Column("AA"), 12), Some("S!AA18".to_owned()));
        assert_eq!(down.cell(CellAxis::Row(6), 0), None);
        assert_eq!(down.cell(CellAxis::None, 0), None);
    }
```

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --lib meta 2>&1 | tail -5
```
Expected: compile error (`cannot find macro rows`).

- [ ] **Step 2: Implement.** In `meta.rs`, after `ResultMeta`'s impl:

```rust
/// The callback of [`ResultSet::visit`] and [`RowSet::visit_row`]:
/// `f(path, meta, cell, value)`, where `cell` is the workbook cell (`meta.cell`
/// for a scalar, synthesized for a table row).
pub type ResultVisitor<'a> = dyn FnMut(&str, &'static ResultMeta, Option<&str>, Value) + 'a;

/// Where a table field sits on its sheet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellAxis {
    /// A column letter: rows run down the sheet (the sweeps).
    Column(&'static str),
    /// A row number: rows run across the sheet (the screw table).
    Row(u32),
    /// Not a workbook cell (e.g. the screw size name, a header).
    None,
}

/// Metadata of one table field: the result metadata plus its place on the sheet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColumnMeta {
    pub meta: ResultMeta,
    pub axis: CellAxis,
}

const fn column_meta(unit: &'static str, label: &'static str, help: &'static str, axis: CellAxis) -> ColumnMeta {
    ColumnMeta { meta: ResultMeta { name: "", ty: FieldType::F64, unit, label, help, cell: None }, axis }
}

/// A table field that sits in a sheet column (the sweeps: `B` .. `AA`).
pub const fn col(unit: &'static str, label: &'static str, help: &'static str, column: &'static str) -> ColumnMeta {
    column_meta(unit, label, help, CellAxis::Column(column))
}

/// A table field that sits in a sheet row (the screw table: rows 6 .. 38).
pub const fn at_row(unit: &'static str, label: &'static str, help: &'static str, row: u32) -> ColumnMeta {
    column_meta(unit, label, help, CellAxis::Row(row))
}

/// A table field with no workbook cell.
pub const fn uncelled_col(unit: &'static str, label: &'static str, help: &'static str) -> ColumnMeta {
    column_meta(unit, label, help, CellAxis::None)
}

impl ColumnMeta {
    /// Completes the metadata with the field's name and type (used by `rows!`).
    #[doc(hidden)]
    pub const fn bind(self, name: &'static str, ty: FieldType) -> Self {
        Self { meta: self.meta.bind(name, ty), ..self }
    }
}

/// How the rows of a table map to workbook cells.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TableLayout {
    /// Row `i` of the table is sheet row `first_row + i`; fields name their column.
    RowsDown { sheet: &'static str, first_row: u32 },
    /// Row `i` of the table is sheet column `columns[i]`; fields name their row.
    ColumnsAcross { sheet: &'static str, columns: &'static [&'static str] },
}

impl TableLayout {
    /// The cell of field `axis` in table row `index`; `None` when the axis does
    /// not fit the layout or the row has no column.
    pub fn cell(&self, axis: CellAxis, index: usize) -> Option<String> {
        match (*self, axis) {
            (TableLayout::RowsDown { sheet, first_row }, CellAxis::Column(column)) => {
                Some(format!("{sheet}!{column}{}", first_row as usize + index))
            }
            (TableLayout::ColumnsAcross { sheet, columns }, CellAxis::Row(row)) => {
                columns.get(index).map(|column| format!("{sheet}!{column}{row}"))
            }
            _ => None,
        }
    }
}

/// A table row declared with [`rows!`].
pub trait RowSet {
    /// Field metadata, in declaration order.
    const COLUMNS: &'static [ColumnMeta];

    /// Calls `f(path, meta, cell, value)` for every field of this row, row
    /// `index` of a table laid out by `layout`. `prefix` ends with `[index].`.
    fn visit_row(&self, prefix: &str, layout: &TableLayout, index: usize, f: &mut ResultVisitor<'_>);
}
```

Change `ResultSet::visit` to take the cell (`fn visit(&self, prefix: &str, f: &mut ResultVisitor<'_>);`, doc: "`cell` is the field's workbook cell: `meta.cell` for a scalar, synthesized for a table row"); add `pub cell: Option<String>` to `ResultRow` (doc: "the workbook cell, synthesized for table rows"); in `result_rows` push `cell: cell.map(str::to_owned)`.

Replace the `results!` macro:

```rust
macro_rules! results {
    (
        $(#[$sattr:meta])*
        pub struct $Name:ident {
            fields {
                $(
                    $(#[$fattr:meta])*
                    $field:ident : $ty:ty => $meta:expr
                ),* $(,)?
            }
            $(
                groups {
                    $( $(#[$gattr:meta])* $group:ident : $Group:ty ),* $(,)?
                }
            )?
            $(
                tables {
                    $( $(#[$tattr:meta])* $table:ident : $Row:ty => $layout:expr ),* $(,)?
                }
            )?
        }
    ) => {
        $(#[$sattr])*
        #[derive(Clone, Debug, PartialEq)]
        #[allow(non_snake_case)]
        pub struct $Name {
            $( $(#[$fattr])* pub $field: $ty, )*
            $( $( $(#[$gattr])* pub $group: $Group, )* )?
            $( $( $(#[$tattr])* pub $table: ::std::vec::Vec<$Row>, )* )?
        }

        impl $crate::engine::meta::ResultSet for $Name {
            const FIELDS: &'static [$crate::engine::meta::ResultMeta] = &[
                $( ($meta).bind(
                    stringify!($field),
                    <$ty as $crate::engine::meta::FieldValue>::TYPE,
                ), )*
            ];
            const GROUPS: &'static [&'static str] = &[ $( $( stringify!($group), )* )? ];

            fn visit(
                &self,
                prefix: &str,
                f: &mut $crate::engine::meta::ResultVisitor<'_>,
            ) {
                let values = [ $( $crate::engine::meta::FieldValue::to_value(&self.$field), )* ];
                for (meta, value) in <Self as $crate::engine::meta::ResultSet>::FIELDS.iter().zip(values) {
                    f(&::std::format!("{prefix}{}", meta.name), meta, meta.cell, value);
                }
                $( $(
                    $crate::engine::meta::ResultSet::visit(
                        &self.$group,
                        &::std::format!("{prefix}{}.", stringify!($group)),
                        f,
                    );
                )* )?
                $( $(
                    for (index, row) in self.$table.iter().enumerate() {
                        $crate::engine::meta::RowSet::visit_row(
                            row,
                            &::std::format!("{prefix}{}[{index}].", stringify!($table)),
                            &$layout,
                            index,
                            f,
                        );
                    }
                )* )?
            }
        }
    };
}
pub(crate) use results;

/// Declares a table row struct: fields with `col`/`at_row`/`uncelled_col` metadata.
/// Tables are declared in a `results!` struct's `tables { .. }` section, which
/// names the layout; the row names only each field's column or row.
#[allow(unused_macros)] // first used by clamps (Task 10), which removes this allow
macro_rules! rows {
    (
        $(#[$sattr:meta])*
        pub struct $Name:ident {
            $(
                $(#[$fattr:meta])*
                $field:ident : $ty:ty => $meta:expr
            ),* $(,)?
        }
    ) => {
        $(#[$sattr])*
        #[derive(Clone, Debug, PartialEq)]
        #[allow(non_snake_case)]
        pub struct $Name {
            $( $(#[$fattr])* pub $field: $ty, )*
        }

        impl $crate::engine::meta::RowSet for $Name {
            const COLUMNS: &'static [$crate::engine::meta::ColumnMeta] = &[
                $( ($meta).bind(
                    stringify!($field),
                    <$ty as $crate::engine::meta::FieldValue>::TYPE,
                ), )*
            ];

            fn visit_row(
                &self,
                prefix: &str,
                layout: &$crate::engine::meta::TableLayout,
                index: usize,
                f: &mut $crate::engine::meta::ResultVisitor<'_>,
            ) {
                let values = [ $( $crate::engine::meta::FieldValue::to_value(&self.$field), )* ];
                for (column, value) in <Self as $crate::engine::meta::RowSet>::COLUMNS.iter().zip(values) {
                    let cell = layout.cell(column.axis, index);
                    f(&::std::format!("{prefix}{}", column.meta.name), &column.meta, cell.as_deref(), value);
                }
            }
        }
    };
}
#[allow(unused_imports)] // first used by clamps (Task 10), which removes this allow
pub(crate) use rows;
```

Update the module doc of `meta.rs` with one paragraph on tables (`rows!`, `tables { }`, synthesized cells).

- [ ] **Step 3: Harness on synthesized cells.**
  - `tests/common/mod.rs`: `PortedResults` gains `pub table_cells: usize` (doc: "cells of the group's tables: the screw table, the sweeps"); every existing entry gets `table_cells: 0`; add `pub fn is_table_path(path: &str) -> bool { path.contains('[') }`.
  - `tests/parity.rs`: `let Some(cell) = row.cell.as_deref() else { continue };` for results; the expected result map is `p.cells + p.table_cells`.
  - `tests/deviations.rs`: `cell_values` returns `BTreeMap<String, Value>` (inputs keyed by `meta.cell`, results by `row.cell`); `changed_cells` returns `BTreeSet<String>`; the registered sets become `BTreeSet<String>` (`c.cell.to_owned()`); lookups use `corrected[change.cell]` via `&corrected[&change.cell.to_owned()]`.
  - `tests/schema.rs`: in `every_field_has_a_label_and_well_formed_unique_path_and_cell` take result cells from `r.cell.as_deref()` (collect owned `String`s for the uniqueness set). Add:

```rust
/// The workbook cells holding a table column's label, unit and note, from one
/// of its value cells (`Clamp screw sizes!C21` gives B21, H21, I21;
/// `Gap sweep!N6` gives N4, N5 and no note).
fn header_cells(cell: &str) -> Option<(String, String, Option<String>)> {
    let (sheet, address) = cell.split_once('!')?;
    let letters: String = address.chars().take_while(|c| c.is_ascii_uppercase()).collect();
    let row = &address[letters.len()..];
    match sheet {
        "Clamp screw sizes" => Some((format!("{sheet}!B{row}"), format!("{sheet}!H{row}"), Some(format!("{sheet}!I{row}")))),
        "Gap sweep" | "Pole sweep" => Some((format!("{sheet}!{letters}4"), format!("{sheet}!{letters}5"), None)),
        _ => None,
    }
}

#[test]
fn table_columns_match_the_workbook_headers() {
    let snapshot = snapshot();
    // An empty workbook cell is absent from the snapshot: it reads as "".
    let text_at = |cell: &str| match snapshot.get(cell) {
        Some(Value::Text(t)) => t.clone(),
        _ => String::new(),
    };
    let mut failures = Vec::new();
    for row in result_rows(&compute_all(&DesignInputs::default())) {
        // Headers belong to columns: check each once, at its first data row.
        let (Some(cell), true) = (row.cell.as_deref(), row.path.contains("[0].")) else { continue };
        // Column B of the sweeps: "Corner gap (mm)" on one sheet, "Poles" on the other.
        if row.path.ends_with(".variable") {
            continue;
        }
        let Some((label, unit, help)) = header_cells(cell) else {
            failures.push(format!("{}: no header rule for {cell}", row.path));
            continue;
        };
        let m = row.meta;
        for (what, rust, cell) in [("label", m.label, Some(label)), ("unit", m.unit, Some(unit)), ("help", m.help, help)] {
            if let Some(cell) = cell
                && rust != text_at(&cell)
            {
                failures.push(format!("{}: {what} {rust:?}, workbook {cell} {:?}", row.path, text_at(&cell)));
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}
```
  - `tests/python_schema.rs`: in `ported_results_carry_the_python_metadata` skip `is_table_path(&row.path)` (Python lists no metadata for table rows). Add:

```rust
#[test]
fn tables_match_the_python_layout() {
    let doc = read_json(&data_path("python_schema.json"));
    let results = result_rows(&compute_all_with(&DesignInputs::defaults_with(Deviations::NONE), Deviations::NONE));
    let mut failures = Vec::new();
    for (table, layout) in doc["tables"].as_object().expect("a tables object") {
        if !is_ported_result(table) {
            continue;
        }
        let prefix = format!("{table}[");
        let rows: Vec<&ResultRow> = results.iter().filter(|r| r.path.starts_with(&prefix)).collect();
        let fields: Vec<&str> = layout["fields"].as_array().expect("fields").iter().map(|f| f.as_str().expect("a name")).collect();
        let n_rows = layout["rows"].as_u64().expect("rows") as usize;
        if rows.len() != n_rows * fields.len() {
            failures.push(format!("{table}: {} values, Python has {n_rows} rows of {} fields", rows.len(), fields.len()));
            continue;
        }
        for (i, chunk) in rows.chunks(fields.len()).enumerate() {
            for (row, name) in chunk.iter().zip(&fields) {
                let path = format!("{table}[{i}].{name}");
                if row.path != path {
                    failures.push(format!("{}: expected {path}", row.path));
                    continue;
                }
                let cell = layout["cells"].get(*name).map(|cells| cells[i].as_str().expect("a cell").to_owned());
                if row.cell != cell {
                    failures.push(format!("{path}: cell rust={:?} python={cell:?}", row.cell));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}
```
(import `compute_all_with`, `ResultRow`, `is_ported_result`, `is_table_path`, `data_path`, `read_json`).

  - Generator: add (imports `from dataclasses import fields as dc_fields`, `from magcoupling.clamps import SCREW_SIZES, TABLE_COLUMNS, TABLE_ROWS, ScrewRow`, `from magcoupling.sweeps import GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES, SWEEP_COLUMNS, SweepRow`):

```python
def table_layouts() -> dict:
    """Each table's field order, row count and the cell of every (field, row): test_parity.py's mapping."""
    def sweep(sheet: str, n: int) -> dict:
        return {"fields": [f.name for f in dc_fields(SweepRow)], "rows": n,
                "cells": {name: [f"{sheet}!{col}{6 + i}" for i in range(n)] for name, col in SWEEP_COLUMNS.items()}}
    screw = {"fields": [f.name for f in dc_fields(ScrewRow)], "rows": len(SCREW_SIZES),
             "cells": {name: [f"Clamp screw sizes!{col}{row}" for col in TABLE_COLUMNS] for name, row in TABLE_ROWS.items()}}
    return {"clamps.table": screw, "gap_sweep": sweep("Gap sweep", len(GAP_SWEEP_CORNER_GAPS_MM)),
            "pole_sweep": sweep("Pole sweep", len(POLE_SWEEP_POLES))}
```
and put `"tables": table_layouts()` in the `python_schema()` document (update its `about` text: "table layouts under 'tables'").

Regenerate:

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: only `python_schema.json` changes among the data files.

- [ ] **Step 4: Run all tests, then gate**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok; `table_rows_get_synthesized_cells`, `a_layout_gives_no_cell_to_a_mismatched_axis_or_a_row_past_its_columns`, `table_columns_match_the_workbook_headers` and `tables_match_the_python_layout` pass (the last two check nothing yet: no ported table).

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task9.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task9.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README "Porting a module" step 3: tables (`rows!`, `tables { }`, synthesized cells, header test); layout table row for `meta.rs` mentions tables.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src/engine/meta.rs magcoupling-rs/tests reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): table rows in the metadata model

rows! declares a table row; results! gains a tables section that names the
layout (rows down the sheet or columns across it), so every table value
gets a synthesized workbook cell. The harness reads cells from the result
rows, checks table labels and units against the workbook headers, and
checks each table's fields, row count and cells against the Python layout.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 10: Shaft clamps (`clamps.py`)

First line: depends on D3 (the class-name fallback). If D3 is the `Result` alternative, stop and escalate.

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Create: `magcoupling-rs/src/engine/clamps.rs`
- Modify: `magcoupling-rs/src/engine/mod.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`, `tests/static_data.rs`, `tests/robustness.rs`, `tests/schema.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: `tests/data/input_schema.json`, `tests/data/static_data.json`; new `tests/data/differential/clamps.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`

**Interfaces:**
- Consumes: `rows!`, `TableLayout`, `results!` tables (Task 9); `materials::{AluminiumAlloy, AL7075, AL6061}`, `ScrewClasses::proof` (Task 7); `MetalDesignResults { torque_cold_high_Nm, .. }`; `compat::{ceiling, floor_, fmt_num, fmt_fixed, py_min}`.
- Produces: `pub struct ScrewSize { pub name: &'static str, pub d_mm: f64, pub pitch_mm: f64, pub As_mm2: f64, pub hole_mm: f64, pub head_mm: f64, pub head_h_mm: f64, pub hex_mm: f64 }`, `pub const SCREW_SIZES: [ScrewSize; 5]`, `pub const TABLE_COLUMNS: [&str; 5]`, `inputs! ClampInputs` (25), `rows! ScrewRow` (34 fields), `results! ClampResults` (33 scalar fields + `tables { table: ScrewRow }`), `pub const MACHINING_STEPS: [&str; 4]`, `pub const NONE_FITS: &str = "None: enlarge the boss or the clamp length";`, `pub fn screw_class_name(code: i64) -> &'static str`, `pub fn compute(ci: &ClampInputs, shaft_mm: f64, max_torque_Nm: f64, al_props: &AluminiumAlloy, screw_proof_MPa: f64, dev: Deviations) -> ClampResults`; `DesignInputs` groups `..., temperature, clamps`; `DesignResults` groups `..., temperature, clamps`.

- [ ] **Step 1: Enable the cells (failing).** `PORTED_INPUTS`: append `PortedInputs { group: "clamps", cells: 25 }`; `PORTED_RESULTS`: append `PortedResults { group: "clamps", cells: 33, table_cells: 165 }`.

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "right|test result"
```
Expected: FAIL; `right` includes `"clamps": 198` (results) and `"clamps": 25` (inputs).

- [ ] **Step 2: Port.** Imports of `clamps.rs`: `use std::f64::consts::PI; use super::compat::{ceiling, floor_, fmt_fixed, fmt_num, py_min}; use super::deviations::Deviations; use super::materials::AluminiumAlloy; use super::meta::{NumOrText, TableLayout, at_row, inputs, out, param, results, rows, uncelled_col};`. Python spec: `clamps.py` lines 28-47 (`ScrewSize`, `SCREW_SIZES`, `TABLE_COLUMNS`), 50-76 (`ClampInputs`), 79-119 (`ScrewRow`, `TABLE_ROWS`), 122-157 (`ClampResults`), 160-166 (`MACHINING_STEPS`), 169-171 (`_fmt_num` = `compat::fmt_num`), 174-237 (`compute`).

- [ ] **2a. Static data and inputs.** `ScrewSize` is a plain struct, so the allow `rows!` emits does not cover it: declare it with `#[derive(Clone, Copy, Debug, PartialEq)]` and `#[allow(non_snake_case)] // Python's As_mm2` (gate 5 runs clippy with `-D warnings`). `SCREW_SIZES` rows exactly as lines 41-45 (float literals: `ScrewSize { name: "M4", d_mm: 4.0, pitch_mm: 0.7, As_mm2: 8.78, hole_mm: 4.5, head_mm: 7.0, head_h_mm: 4.0, hex_mm: 3.0 }`, ...); `pub const TABLE_COLUMNS: [&str; 5] = ["C", "D", "E", "F", "G"];`; `MACHINING_STEPS` as lines 161-165 (step 2 is one string: join the two Python literals). Inputs: transcribe lines 52-76; selectors `clamp_type` `&[(1, "one-piece slotted"), (2, "two-piece")]`, `alloy` `&[(1, "7075-T6"), (2, "6061-T6")]`, `screw_class` `&[(1, "12.9"), (2, "10.9"), (3, "A4-70")]`. Ranges (`clamps.` omitted):

| Path | Default | min | max | step | A3 | Constraint |
|---|---|---|---|---|---|---|
| `safety_factor` | 2.0 | 1.0 | 5.0 | 0.1 | | |
| `friction` | 0.15 | 0.05 | 0.4 | 0.005 | yes | > 0: divides the screws needed |
| `factor_one_piece` | 0.8 | 0.3 | 1.0 | 0.01 | | > 0 |
| `factor_two_piece` | 1.0 | 0.3 | 1.2 | 0.01 | | > 0 |
| `preload_fraction` | 0.75 | 0.3 | 0.9 | 0.01 | yes | |
| `nut_factor` | 0.2 | 0.1 | 0.35 | 0.01 | | |
| `engagement_x_d` | 2.0 | 0.5 | 3.0 | 0.05 | | |
| `strip_sf` | 1.5 | 1.0 | 4.0 | 0.1 | | |
| `boss_od_mm` | 25 | 12.0 | 60.0 | 0.1 | | |
| `clamp_length_mm` | 10 | 3.0 | 40.0 | 0.1 | | > 0: divides the key pressure |
| `slit_mm` | 0.8 | 0.2 | 3.0 | 0.1 | | |
| `ligament_mm`, `wall_out_mm`, `relief_mm` | 1.0 | 0.3 | 5.0 | 0.1 | | |
| `grip_min_mm` | 3.0 | 0.5 | 10.0 | 0.1 | | |
| `axial_margin_mm` | 1.0 | 0.0 | 5.0 | 0.1 | | |
| `hinge_mm` | 8.0 | 1.0 | 30.0 | 0.1 | | |
| `key_width_mm` | 4 | 1.0 | 10.0 | 0.5 | | |
| `key_contact_mm` | 1.5 | 0.5 | 5.0 | 0.1 | | > 0: divides |
| `joint_screws` (i64) | 3 | 1 | 8 | 1 | | |
| `joint_bolt_circle_mm` | 26 | 10.0 | 60.0 | 0.5 | | |
| `joint_friction` | 0.15 | 0.05 | 0.4 | 0.005 | | |

- [ ] **2b. The table row and the results.** First delete the two allow lines Task 9 put on `rows!` in `meta.rs` (`#[allow(unused_macros)]` and `#[allow(unused_imports)]`): the macro is used from here on, and a leftover allow would hide a future unused macro. Table metadata comes from the workbook's own header cells (Python has none): label = column B, unit = column H, help = column I of the field's row ("" where the workbook cell is empty). `table_columns_match_the_workbook_headers` enforces it.

```rust
rows! {
    /// One column of 'Clamp screw sizes' (rows 6-38), one per screw size.
    pub struct ScrewRow {
        size: String => uncelled_col("", "Screw size", ""),
        d_mm: f64 => at_row("mm", "Nominal diameter", "Standard data.", 6),
        pitch_mm: f64 => at_row("mm", "Thread pitch (coarse)", "", 7),
        As_mm2: f64 => at_row("mm²", "Tensile stress area", "", 8),
        hole_mm: f64 => at_row("mm", "Clearance hole (medium)", "ISO 273.", 9),
        head_mm: f64 => at_row("mm", "Head diameter", "ISO 4762 maximum.", 10),
        head_h_mm: f64 => at_row("mm", "Head height", "", 11),
        hex_mm: f64 => at_row("mm", "Hex key", "", 12),
        tap_drill_mm: f64 => at_row("mm", "Tap drill", "Nominal minus pitch.", 13),
        offset_mm: f64 => at_row("mm", "Screw offset from the shaft axis",
            "Half the bore, plus the ligament, plus half the clearance hole.", 14),
        wall_out_mm: f64 => at_row("mm", "Wall outside the screw hole", "", 15),
        head_fits: i64 => at_row("-", "Head seat fits inside the boss (1 = yes)", "", 16),
        grip_mm: f64 => at_row("mm", "Head-side jaw (grip)", "From the head seat to the slit.", 17),
        thread_avail_mm: f64 => at_row("mm", "Thread length available in the far jaw", "", 18),
        engagement_req_mm: f64 => at_row("mm", "Thread engagement required", "", 19),
        geometry_ok: i64 => at_row("-", "Geometry fits (1 = yes)", "Wall, head seat, grip and engagement all pass.", 20),
        preload_strength_N: f64 => at_row("N", "Preload from screw strength", "Share of proof load times stress area.", 21),
        preload_strip_N: f64 => at_row("N", "Preload limit from thread stripping", "Shear area about 0.6·π·d·engagement.", 22),
        preload_N: f64 => at_row("N", "Allowable preload", "The smaller of the two.", 23),
        head_pressure_MPa: f64 => at_row("MPa", "Pressure under the head", "", 24),
        head_check: String => at_row("", "Head pressure check", "", 25),
        torque_per_screw_Nm: f64 => at_row("N·m", "Clamp torque per screw",
            "Friction x screw force x shaft diameter x clamp factor.", 26),
        screws_needed: i64 => at_row("-", "Screws needed", "", 27),
        pitch_axial_mm: f64 => at_row("mm", "Axial screw spacing", "Head diameter plus 1 mm.", 28),
        screws_fit: i64 => at_row("-", "Screws that fit along the clamp", "", 29),
        works: i64 => at_row("-", "Size works (1 = yes)", "Geometry fits and enough screws fit.", 30),
        clamp_torque_Nm: f64 => at_row("N·m", "Clamp torque with the screws needed", "", 31),
        sf_coupling: f64 => at_row("-", "Safety factor on the coupling torque", "", 32),
        tightening_Nm: f64 => at_row("N·m", "Tightening torque", "Nut factor x preload x diameter.", 33),
        length_mm: f64 => at_row("mm", "Screw length", "Grip plus required engagement, rounded up to an even length.", 34),
        length_ok: i64 => at_row("-", "Length stays inside the far jaw (1 = yes)", "", 35),
        cbore_dia_mm: f64 => at_row("mm", "Counterbore diameter", "Head plus 0.5 mm.", 36),
        cbore_depth_mm: f64 => at_row("mm", "Counterbore depth at the screw axis", "From the OD to the head seat.", 37),
        vent_port_ok: i64 => at_row("-", "Key passes the M6 vent port (1 = yes)", "4 mm key or smaller.", 38),
    }
}
```
(`As_mm2` needs `#[allow(non_snake_case)]`, which `rows!` already emits.) `ClampResults`: transcribe lines 124-157 without `table`; types `index: i64`; `recommended`, `head_check`, `vent_port`, `layout_slit`, `layout_relief`: `String`; `screws`, `tightening_Nm`, `hex_mm`, `capacity_Nm`, `sf_coupling` and the eight `layout_*_mm`: `NumOrText` (a number, or `""` when no size works); the rest `f64`. Then `tables { table: ScrewRow => TableLayout::ColumnsAcross { sheet: "Clamp screw sizes", columns: &TABLE_COLUMNS }, }`.

Accepted order difference: Python's `ClampResults` declares `table` between `boss_radius_mm` and `index`, while `results!` emits tables after all fields and groups, so `result_rows` lists the 170 screw-table values after `layout_relief`, not after `boss_radius_mm`. The scalar order still equals Python's (`schemas_list_fields_in_the_python_order`, Task 12, filters table paths), and the parity and differential tests key by path, so nothing compares positions across the table. The GUI results table (M4) therefore shows the screw table after the clamp scalars; record this in the README's clamps note (Step 5).

- [ ] **2c. `compute`.**

```rust
/// Screw class text for Shaft clamps!C48 (workbook CHOOSE order). Python raises
/// KeyError outside 1-3; here `"#N/A"` (decision D3).
pub fn screw_class_name(code: i64) -> &'static str {
    match code {
        1 => "12.9",
        2 => "10.9",
        3 => "A4-70",
        _ => "#N/A",
    }
}

/// Shaft clamps and Clamp screw sizes. `al_props`: the selected alloy.
#[allow(non_snake_case)] // Python names (R, T_req, Fb, Fs, F, Tper, T)
pub fn compute(
    ci: &ClampInputs, shaft_mm: f64, max_torque_Nm: f64, al_props: &AluminiumAlloy, screw_proof_MPa: f64, _dev: Deviations,
) -> ClampResults {
    let R = ci.boss_od_mm / 2.0;
    let T_req = max_torque_Nm * ci.safety_factor;
    let k = if ci.clamp_type == 1 { ci.factor_one_piece } else { ci.factor_two_piece };
    let rows: Vec<ScrewRow> = SCREW_SIZES
        .iter()
        .map(|s| {
            let e = shaft_mm / 2.0 + ci.ligament_mm + s.hole_mm / 2.0;
            let wall = R - e - s.hole_mm / 2.0;
            let fits: i64 = if e + s.head_mm / 2.0 <= R { 1 } else { 0 };
            let grip = if fits == 1 { (R.powi(2) - (e + s.head_mm / 2.0).powi(2)).sqrt() - ci.slit_mm / 2.0 } else { 0.0 };
            let avail = if e < R { (R.powi(2) - e.powi(2)).sqrt() - ci.slit_mm / 2.0 } else { 0.0 };
            let ereq = ci.engagement_x_d * s.d_mm;
            let geo: i64 = if wall >= ci.wall_out_mm && fits == 1 && grip >= ci.grip_min_mm && avail >= ereq { 1 } else { 0 };
            let Fb = ci.preload_fraction * screw_proof_MPa * s.As_mm2;
            let Fs = 0.6 * PI * s.d_mm * ereq * al_props.shear_MPa / ci.strip_sf;
            let F = py_min(Fb, Fs);
            let p = F / (PI / 4.0 * (s.head_mm.powi(2) - s.hole_mm.powi(2)));
            let Tper = ci.friction * F * shaft_mm / 1000.0 * k;
            // Python int(ceiling(..)): the saturating cast never panics (NaN gives 0).
            let need = ceiling(T_req / Tper, 1.0) as i64;
            let pitch = s.head_mm + 1.0;
            let span = ci.clamp_length_mm - 2.0 * ci.axial_margin_mm - (s.head_mm + 0.5);
            let fit = if span >= 0.0 { (floor_(span / pitch, 1.0) as i64).saturating_add(1) } else { 0 };
            let works: i64 = if geo == 1 && need <= fit { 1 } else { 0 };
            let length = if fits == 1 { ceiling(grip + ereq, 2.0) } else { 0.0 };
            ScrewRow {
                size: s.name.to_owned(), d_mm: s.d_mm, pitch_mm: s.pitch_mm, As_mm2: s.As_mm2, hole_mm: s.hole_mm,
                head_mm: s.head_mm, head_h_mm: s.head_h_mm, hex_mm: s.hex_mm, tap_drill_mm: s.d_mm - s.pitch_mm,
                offset_mm: e, wall_out_mm: wall, head_fits: fits, grip_mm: grip, thread_avail_mm: avail,
                engagement_req_mm: ereq, geometry_ok: geo, preload_strength_N: Fb, preload_strip_N: Fs, preload_N: F,
                head_pressure_MPa: p,
                head_check: if p <= al_props.head_pressure_limit_MPa { "OK" } else { "Use a hardened washer" }.to_owned(),
                torque_per_screw_Nm: Tper, screws_needed: need, pitch_axial_mm: pitch, screws_fit: fit, works,
                clamp_torque_Nm: need as f64 * Tper, sf_coupling: need as f64 * Tper / max_torque_Nm,
                tightening_Nm: ci.nut_factor * F * s.d_mm / 1000.0, length_mm: length,
                length_ok: if fits == 1 && length <= grip + avail { 1 } else { 0 }, cbore_dia_mm: s.head_mm + 0.5,
                cbore_depth_mm: if fits == 1 {
                    (R.powi(2) - e.powi(2)).sqrt() - (R.powi(2) - (e + s.head_mm / 2.0).powi(2)).sqrt()
                } else {
                    0.0
                },
                vent_port_ok: if s.hex_mm <= 4.0 { 1 } else { 0 },
            }
        })
        .collect();

    let index = rows.iter().position(|r| r.works == 1).map_or(0, |i| i as i64 + 1);
    let chosen = rows.iter().find(|r| r.works == 1);
    let recommended = match chosen {
        Some(r) => format!("ISO 4762 {} x {}, class {}", r.size, fmt_num(r.length_mm), screw_class_name(ci.screw_class)),
        None => NONE_FITS.to_owned(),
    };
    let blank = NumOrText::Text("");
    let pick = |value: fn(&ScrewRow) -> f64| chosen.map_or(blank, |r| NumOrText::Num(value(r)));
    let key_p = 2.0 * T_req / ((shaft_mm / 1000.0) * (ci.key_contact_mm / 1000.0) * (ci.clamp_length_mm / 1000.0)) / 1e6;
    let m3 = &rows[1]; // Python rows[1]: the M3 row
    let joint_T = ci.joint_friction * ci.joint_screws as f64 * m3.preload_N * ci.joint_bolt_circle_mm / 2000.0;
    ClampResults {
        shaft_mm, max_torque_Nm, required_Nm: T_req, clamp_factor: k, al_shear_MPa: al_props.shear_MPa,
        al_head_limit_MPa: al_props.head_pressure_limit_MPa, al_key_allow_MPa: al_props.key_bearing_allow_MPa,
        screw_proof_MPa, boss_radius_mm: R, index, recommended,
        screws: pick(|r| r.screws_needed as f64),
        tightening_Nm: pick(|r| r.tightening_Nm),
        hex_mm: pick(|r| r.hex_mm),
        capacity_Nm: pick(|r| r.clamp_torque_Nm),
        sf_coupling: pick(|r| r.sf_coupling),
        head_check: chosen.map_or(String::new(), |r| r.head_check.clone()),
        vent_port: chosen.map_or(String::new(), |r| {
            let text = if r.hex_mm <= 4.0 { "Yes: the key fits the 4 mm limit" } else { "No: key too large" };
            text.to_owned()
        }),
        key_pressure_MPa: key_p, key_sf: al_props.key_bearing_allow_MPa / key_p, joint_preload_N: m3.preload_N,
        joint_torque_Nm: joint_T, joint_sf: joint_T / T_req,
        layout_offset_mm: pick(|r| r.offset_mm),
        layout_pitch_mm: pick(|r| r.pitch_axial_mm),
        layout_first_mm: chosen.map_or(blank, |r| {
            NumOrText::Num((ci.clamp_length_mm - (r.screws_needed as f64 - 1.0) * r.pitch_axial_mm) / 2.0)
        }),
        layout_cbore_dia_mm: pick(|r| r.cbore_dia_mm),
        layout_cbore_depth_mm: pick(|r| r.cbore_depth_mm),
        layout_grip_mm: pick(|r| r.grip_mm),
        layout_tap_drill_mm: pick(|r| r.tap_drill_mm),
        layout_thread_avail_mm: pick(|r| r.thread_avail_mm),
        layout_slit: format!("{} mm wide, bore to OD on one side, free end to the relief cut", fmt_fixed(ci.slit_mm, 1)),
        layout_relief: format!(
            "{} mm wide at {} mm from the free end, {} mm deep from the slit side",
            fmt_fixed(ci.relief_mm, 1), fmt_fixed(ci.clamp_length_mm, 1), fmt_fixed(ci.boss_od_mm - ci.hinge_mm, 1)
        ),
        table: rows,
    }
}
```
If the borrow checker refuses `table: rows` after borrows of `rows` (it should accept it: the last borrow ends before the move), bind every derived value to a local first and build the struct afterwards.

- [ ] **2d. Wire into `api.rs`** (Python lines 211-212). `mod.rs`: `pub mod clamps;`. Add `clamps: ClampInputs` after `temperature` in `DesignInputs`, `clamps: ClampResults` after `temperature` in `DesignResults`. After `let temp = ...`:

```rust
    let alloy = if inputs.clamps.alloy == 1 { &materials::AL7075 } else { &materials::AL6061 };
    let clr = clamps::compute(
        &inputs.clamps, ci.bore_mm, mdr.torque_cold_high_Nm, alloy, mat_in.screws.proof(inputs.clamps.screw_class), dev,
    );
```

- [ ] **2e. Tests.** Unit test in `clamps.rs` (the equality edges of the per-size checks, on the M4 column):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::materials::AL7075;

    #[test]
    fn size_checks_take_the_python_branch_at_exact_equality() {
        // Architecture section 7 step 8, on the M4 column (it works at the defaults). Each
        // right-hand side is an input its left-hand side does not read: one run supplies it.
        let run = |ci: &ClampInputs, alloy: &AluminiumAlloy| compute(ci, 10.0, 3.7647, alloy, 970.0, Deviations::NONE);
        let m4 = run(&ClampInputs::default(), &AL7075).table[2].clone();
        assert_eq!(m4.size, "M4");
        let ci = ClampInputs {
            wall_out_mm: m4.wall_out_mm, // wall >= wall_out
            grip_min_mm: m4.grip_mm,     // grip >= grip_min
            // avail >= ereq, ereq = engagement_x_d * d with d = 4 mm: dividing and multiplying by 4 is exact
            engagement_x_d: m4.thread_avail_mm / 4.0,
            // span >= 0: 9.5 - 2 * 1.0 - (7.0 + 0.5) = 0 exactly (axial margin 1 mm, M4 head 7 mm)
            clamp_length_mm: 9.5,
            ..ClampInputs::default()
        };
        // p <= head limit: the head pressure reads the engagement (stripping preload), so take it from this ci.
        let p = run(&ci, &AL7075).table[2].head_pressure_MPa;
        let alloy = AluminiumAlloy { head_pressure_limit_MPa: p, ..AL7075 };
        let r = run(&ci, &alloy).table[2].clone();
        assert_eq!(
            (r.wall_out_mm, r.grip_mm, r.engagement_req_mm, r.head_pressure_MPa),
            (ci.wall_out_mm, ci.grip_min_mm, r.thread_avail_mm, p)
        );
        assert_eq!(r.geometry_ok, 1);
        assert_eq!(r.screws_fit, 1, "a span of exactly 0 fits one screw");
        assert_eq!(r.head_check, "OK");
    }
}
```
(Checked on a scratch build: passes, and swapping any of the five operators, `p <=`, `wall >=`, `grip >=`, `avail >=`, `span >=`, makes it fail.)

`tests/robustness.rs`:

```rust
#[test]
fn an_invalid_screw_class_is_nan_not_a_panic() {
    for code in [0, 4, i64::MIN] {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.clamps.screw_class = code; // bypasses set()
        let c = compute_all_with(&inputs, Deviations::NONE).clamps;
        assert!(c.screw_proof_MPa.is_nan(), "{code}");
        assert!(c.table.iter().all(|r| r.preload_N.is_nan()), "{code}");
        assert!(c.recommended.ends_with("class #N/A") || c.recommended.starts_with("None"), "{code}: {}", c.recommended);
    }
}
```
`tests/schema.rs`, add to `table_columns_match_the_workbook_headers` after the loop (the size field is uncelled, so the loop skips it):

```rust
    let size = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .find(|r| r.path == "clamps.table[0].size")
        .expect("the screw table");
    assert_eq!(size.meta.label, text_at("Clamp screw sizes!B5"));
```
`tests/static_data.rs`:

```rust
#[test]
fn screw_sizes_and_machining_steps_equal_the_python_data() {
    use magcoupling::engine::clamps::{MACHINING_STEPS, SCREW_SIZES, TABLE_COLUMNS};
    let doc = python_static();
    let rows = doc["screw_sizes"].as_array().expect("screw_sizes");
    assert_eq!(rows.len(), SCREW_SIZES.len());
    for (py, rs) in rows.iter().zip(SCREW_SIZES.iter()) {
        assert_eq!(rs.name, text(py, "name"));
        for (key, value) in [("d_mm", rs.d_mm), ("pitch_mm", rs.pitch_mm), ("As_mm2", rs.As_mm2), ("hole_mm", rs.hole_mm),
                             ("head_mm", rs.head_mm), ("head_h_mm", rs.head_h_mm), ("hex_mm", rs.hex_mm)] {
            assert_eq!(value, number(py, key), "{}.{key}", rs.name);
        }
    }
    let steps: Vec<&str> = doc["machining_steps"].as_array().expect("steps").iter().map(|s| s.as_str().expect("text")).collect();
    assert_eq!(MACHINING_STEPS.as_slice(), steps.as_slice());
    let columns: Vec<&str> = doc["table_columns"].as_array().expect("columns").iter().map(|s| s.as_str().expect("text")).collect();
    assert_eq!(TABLE_COLUMNS.as_slice(), columns.as_slice());
}
```
Generator `static_data()`: `"screw_sizes": [dataclasses.asdict(s) for s in py_clamps.SCREW_SIZES]`, `"machining_steps": list(py_clamps.MACHINING_STEPS)`, `"table_columns": list(py_clamps.TABLE_COLUMNS)` (import `from magcoupling import clamps as py_clamps`).

`tests/differential.rs`: extend `Reach` with a fourth variant

```rust
    /// Any text (a result with a single formatted form, e.g. clamps.layout_slit).
    AnyText,
```
(`use Reach::{AnyText, Number, Prefix, Text};`, and `(AnyText, Value::Text(_)) => true` in `reached`); `#[test] fn clamps_matches_python_on_every_case() { check_module("clamps"); }`; `BRANCHES`:

```rust
    ("clamps.recommended", &[Prefix("ISO 4762 "), Text("None: enlarge the boss or the clamp length")]),
    ("clamps.screws", &[Number, Text("")]),
    ("clamps.tightening_Nm", &[Number, Text("")]),
    ("clamps.hex_mm", &[Number, Text("")]),
    ("clamps.capacity_Nm", &[Number, Text("")]),
    ("clamps.sf_coupling", &[Number, Text("")]),
    ("clamps.head_check", &[Text("OK"), Text("Use a hardened washer"), Text("")]),
    ("clamps.vent_port", &[Text("Yes: the key fits the 4 mm limit"), Text("No: key too large"), Text("")]),
    ("clamps.layout_offset_mm", &[Number, Text("")]),
    ("clamps.layout_pitch_mm", &[Number, Text("")]),
    ("clamps.layout_first_mm", &[Number, Text("")]),
    ("clamps.layout_cbore_dia_mm", &[Number, Text("")]),
    ("clamps.layout_cbore_depth_mm", &[Number, Text("")]),
    ("clamps.layout_grip_mm", &[Number, Text("")]),
    ("clamps.layout_tap_drill_mm", &[Number, Text("")]),
    ("clamps.layout_thread_avail_mm", &[Number, Text("")]),
    ("clamps.layout_slit", &[AnyText]),
    ("clamps.layout_relief", &[AnyText]),
    ("clamps.table[*].size", &[Text("M2.5"), Text("M3"), Text("M4"), Text("M5"), Text("M6")]),
    ("clamps.table[*].head_check", &[Text("OK"), Text("Use a hardened washer")]),
```
Generator: `MODULES["clamps"] = ["coupling", "metal", "calibration", "materials", "clamps"]`; `PROBES["clamps"]`:

```python
    "clamps": [
        ("22 mm boss, 10 mm clamp: nothing fits", {"clamps.boss_od_mm": 22.0, "clamps.clamp_length_mm": 10.0}),
        ("22 mm boss, 14.5 mm clamp: two M3", {"clamps.boss_od_mm": 22.0, "clamps.clamp_length_mm": 14.5}),
        ("22 mm boss, 18 mm clamp: three M2.5", {"clamps.boss_od_mm": 22.0, "clamps.clamp_length_mm": 18.0}),
        ("stripping governs: 6061 with 1 x d engagement", {"clamps.alloy": 2, "clamps.engagement_x_d": 1.0}),
        # No fixed case reaches "No: key too large" (M6 must be the first size that works; about 0.5 %
        # of random clamp samples do). Checked in the oracle: recommended "ISO 4762 M6 x 26, class 12.9".
        ("M6 first size that works: key too large for the vent port",
         {"clamps.safety_factor": 3.0, "clamps.friction": 0.1, "clamps.boss_od_mm": 40.0, "clamps.clamp_length_mm": 13.0}),
    ],
```
If `every_branch_is_reached` still names an unreached branch, find an input set in the Python oracle that reaches it (`api.set_input` returns a new `DesignInputs`: rebind it), add it to `PROBES["clamps"]` with a tag naming the branch, and regenerate. Bless and regenerate:

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test schema
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote .../differential/clamps.json (...)` (about 390 cases), under 4 MB.

- [ ] **Step 3: Run parity and differential**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. Parity: 330 scalar result cells + 165 screw-table cells, and all 160 default inputs; Shaft clamps!C47 = 3, C48 = "ISO 4762 M4 x 12, class 12.9", Clamp screw sizes!E34 = 12, E35 = 1. `table_columns_match_the_workbook_headers` and `tables_match_the_python_layout` now check the screw table (33 celled fields x 5 sizes); `an_invalid_screw_class_is_nan_not_a_panic`, `clamps::tests::size_checks_take_the_python_branch_at_exact_equality` and `every_branch_is_reached` (including `"No: key too large"` from the M6 probe) pass.

Mutation spot check: `ceiling(grip + ereq, 2.0)` to `ceiling(grip + ereq, 1.0)`: parity FAILS on Clamp screw sizes!C34..E34 and Shaft clamps!C48. Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task10.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task10.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status adds "Shaft clamps", plus one sentence under the metadata model: "`result_rows` lists a group's tables after its scalars, so the screw table comes after `clamps.layout_relief` (Python declares it after `boss_radius_mm`); the scalar order is Python's."; `docs/ai/03-structure.yaml` `ported` adds `clamps.rs`.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the Shaft clamps and Clamp screw sizes sheets

clamps.py: 25 input cells, 33 result cells and the 165-cell screw table
(labels, units and notes from the workbook headers), screw sizes and the
machining steps as static data. Integer counts use saturating casts; a
screw class outside 1-3 gives NaN and "class #N/A" instead of a panic.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 11: Gap and pole sweeps (`sweeps.py`)

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Create: `magcoupling-rs/src/engine/sweeps.rs`
- Modify: `magcoupling-rs/src/engine/mod.rs`, `src/engine/api.rs`
- Modify: `magcoupling-rs/tests/common/mod.rs`, `tests/differential.rs`, `tests/static_data.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Regenerate: `tests/data/static_data.json`; new `tests/data/differential/gap_sweep.json`, `pole_sweep.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml`

**Interfaces:**
- Consumes: `model::{shear_stress, Harmonic}` and `Harmonic::s` (Task 3); `rows!`, `TableLayout` (Task 9); `ModelResults`, `CouplingInputs`, `MetalDesignInputs`, `f_cal` and `CalibrationInputs::f_cal_original` in `api::compute`; `compat::{py_min, py_max}`.
- Produces: `pub const GAP_SWEEP_CORNER_GAPS_MM: [f64; 13]`, `pub const POLE_SWEEP_POLES: [i64; 6]`, `rows! SweepRow` (26 fields), `pub struct SweepContext` (19 fields), `pub fn gap_sweep(ctx: &SweepContext, npole: i64, a_i: f64, f_cal: f64, dev: Deviations) -> Vec<SweepRow>`, `pub fn pole_sweep(ctx: &SweepContext, corner_gap_mm: f64, bore_mm: f64, keyway_depth_mm: f64, f_cal_original: f64, dev: Deviations) -> Vec<SweepRow>`; `DesignResults` gains `tables { gap_sweep: SweepRow, pole_sweep: SweepRow }`: the complete Python `DesignResults`.

- [ ] **Step 1: Enable the cells (failing).** Append `PortedResults { group: "gap_sweep", cells: 0, table_cells: 338 }` and `PortedResults { group: "pole_sweep", cells: 0, table_cells: 156 }`.

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test parity 2>&1 | grep -E "right|test result"
```
Expected: FAIL; `right` includes `"gap_sweep": 338, "pole_sweep": 156`.

- [ ] **Step 2: Port.** Imports of `sweeps.rs`: `use std::f64::consts::PI; use super::compat::{py_max, py_min}; use super::deviations::Deviations; use super::meta::{col, rows}; use super::model::shear_stress;` (`api.rs` adds `use super::meta::TableLayout;` and `use super::sweeps::{self, SweepRow};`). Python spec: `sweeps.py` lines 21-22 (constants), 25-60 (`SweepRow`, `SWEEP_COLUMNS`), 63-84 (`SweepContext`), 87-117 (`_row`), 120-133 (`gap_sweep`, `pole_sweep`).

```rust
/// Corner gaps of the gap sweep [mm] (rows 6-18).
pub const GAP_SWEEP_CORNER_GAPS_MM: [f64; 13] = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0];
/// Pole counts of the pole sweep (rows 6-11).
pub const POLE_SWEEP_POLES: [i64; 6] = [6, 8, 10, 12, 14, 16];

rows! {
    /// One row of the Gap sweep or Pole sweep sheet (columns B-AA). Labels and
    /// units are the workbook headers (rows 4 and 5).
    pub struct SweepRow {
        variable: f64 => col("", "Swept variable",
            "Corner gap [mm] in the gap sweep (workbook 'Corner gap (mm)'); poles per ring in the pole sweep ('Poles').", "B"),
        inner_apothem_mm: f64 => col("mm", "Inner apothem", "", "C"),
        corner_gap_mm: f64 => col("mm", "Corner gap", "", "D"),
        centre_gap_mm: f64 => col("mm", "Centre gap", "", "E"),
        outer_face_apothem_mm: f64 => col("mm", "Outer face apothem", "", "F"),
        cup_od_mm: f64 => col("mm", "Cup OD", "", "G"),
        gap_radius_mm: f64 => col("mm", "R_g", "", "H"),
        pole_pitch_mm: f64 => col("mm", "Pole pitch", "", "I"),
        fill_inner: f64 => col("-", "Fill in", "", "J"),
        fill_outer: f64 => col("-", "Fill out", "", "K"),
        k1: f64 => col("1/m", "k1", "", "L"),
        s1: f64 => col("-", "S1", "", "M"),
        tau1_Pa: f64 => col("Pa", "tau1", "", "N"),
        k3: f64 => col("1/m", "k3", "", "O"),
        s3: f64 => col("-", "S3", "", "P"),
        tau3_Pa: f64 => col("Pa", "tau3", "", "Q"),
        k5: f64 => col("1/m", "k5", "", "R"),
        s5: f64 => col("-", "S5", "", "S"),
        tau5_Pa: f64 => col("Pa", "tau5", "", "T"),
        tau_Pa: f64 => col("Pa", "tau", "", "U"),
        torque_2d_Nm: f64 => col("Nm", "T2D", "", "V"),
        f_end: f64 => col("-", "f_end", "", "W"),
        pullout_op_Nm: f64 => col("Nm", "Pull-out at T_op", "", "X"),
        pullout_20C_Nm: f64 => col("Nm", "Pull-out at 20 C", "", "Y"),
        gearbox_input_Nm: f64 => col("Nm", "Gearbox input at slip", "", "Z"),
        status: String => col("", "Hot nominal / fit", "", "AA"),
    }
}

/// Values the sweeps borrow from the Calculator and Metal design (Python `SweepContext`).
#[derive(Clone, Debug, PartialEq)]
#[allow(non_snake_case)]
pub struct SweepContext {
    pub faceted: i64,
    pub backiron: i64,
    pub t_i: f64,
    pub w_i: f64,
    pub t_o: f64,
    pub w_o: f64,
    pub L: f64,
    pub br_i20: f64,
    pub br_o20: f64,
    pub br_i_op: f64,
    pub br_o_op: f64,
    pub bond_outer: f64,
    pub cup_wall_corner: f64,
    pub c_end: f64,
    pub mu0: f64,
    pub gear_ratio: f64,
    pub gear_eff: f64,
    pub required_floor_Nm: f64,
    pub max_diameter_mm: f64,
}

/// One sweep row (Python `_row`), columns C-AA with the workbook's status priority.
#[allow(non_snake_case)] // Python names (F, E, G, H, I, J, K, U, V, W, X, Y, Z)
fn row(ctx: &SweepContext, variable: f64, npole: i64, a_i: f64, corner_gap: f64, factor: f64, _dev: Deviations) -> SweepRow {
    let n = npole as f64;
    let r_face = a_i + ctx.t_i;
    let F = (if ctx.faceted == 1 { (r_face.powi(2) + (ctx.w_i / 2.0).powi(2)).sqrt() } else { r_face }) + corner_gap;
    let E = F - r_face;
    let back = F + ctx.t_o + ctx.bond_outer;
    let G = 2.0 * ((if ctx.faceted == 1 { back / (PI / n).cos() } else { back }) + ctx.cup_wall_corner);
    let H = r_face + E / 2.0;
    let I = 2.0 * PI * H / n;
    let J = py_min(1.0, ctx.w_i / (2.0 * PI * (a_i + ctx.t_i / 2.0) / n));
    let K = py_min(1.0, ctx.w_o / (2.0 * PI * (F + ctx.t_o / 2.0) / n));
    let h = shear_stress(ctx.br_i_op, ctx.br_o_op, J, K, npole, H, ctx.t_i, ctx.t_o, E, ctx.backiron, ctx.mu0);
    let s = h.map(|x| x.s(ctx.backiron));
    let U = h.iter().fold(0.0, |acc, x| acc + x.tau);
    let V = U * 2.0 * PI * (H / 1000.0).powi(2) * (ctx.L / 1000.0);
    let W = 1.0 - ctx.c_end * I / ctx.L;
    let X = V * W * factor;
    let Y = X * (ctx.br_i20 * ctx.br_o20) / (ctx.br_i_op * ctx.br_o_op);
    let Z = X / (ctx.gear_ratio * ctx.gear_eff);
    let status = if 2.0 * a_i * (PI / n).tan() < ctx.w_i {
        "inner flat too narrow"
    } else if 2.0 * F * (PI / n).tan() < ctx.w_o {
        "outer flat too narrow"
    } else if G > ctx.max_diameter_mm {
        "outside OD envelope"
    } else if X < ctx.required_floor_Nm {
        "below hot minimum"
    } else {
        "nominal: test needed"
    };
    let [h1, h3, h5] = h;
    SweepRow {
        variable, inner_apothem_mm: a_i, corner_gap_mm: corner_gap, centre_gap_mm: E, outer_face_apothem_mm: F,
        cup_od_mm: G, gap_radius_mm: H, pole_pitch_mm: I, fill_inner: J, fill_outer: K, k1: h1.k, s1: s[0],
        tau1_Pa: h1.tau, k3: h3.k, s3: s[1], tau3_Pa: h3.tau, k5: h5.k, s5: s[2], tau5_Pa: h5.tau, tau_Pa: U,
        torque_2d_Nm: V, f_end: W, pullout_op_Nm: X, pullout_20C_Nm: Y, gearbox_input_Nm: Z, status: status.to_owned(),
    }
}

/// Pull-out vs corner gap for the current layout (Calculator calibration factor).
pub fn gap_sweep(ctx: &SweepContext, npole: i64, a_i: f64, f_cal: f64, dev: Deviations) -> Vec<SweepRow> {
    GAP_SWEEP_CORNER_GAPS_MM.iter().map(|&g| row(ctx, g, npole, a_i, g, f_cal, dev)).collect()
}

/// Pull-out vs poles; smallest apothem that fits the block (+0.05 mm) and the keyed-bore wall (2.5 mm).
pub fn pole_sweep(
    ctx: &SweepContext, corner_gap_mm: f64, bore_mm: f64, keyway_depth_mm: f64, f_cal_original: f64, dev: Deviations,
) -> Vec<SweepRow> {
    POLE_SWEEP_POLES
        .iter()
        .map(|&n| {
            let a_i = py_max(ctx.w_i / (2.0 * (PI / n as f64).tan()) + 0.05, bore_mm / 2.0 + keyway_depth_mm + 2.5);
            row(ctx, n as f64, n, a_i, corner_gap_mm, f_cal_original, dev)
        })
        .collect()
}
```

Unit test in `sweeps.rs` (the equality edges of the status priority; `row` is private, the test module sees it):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// The default design's sweep context (`api::compute` at the workbook defaults).
    fn ctx() -> SweepContext {
        SweepContext {
            faceted: 1, backiron: 1, t_i: 3.17, w_i: 6.35, t_o: 3.17, w_o: 6.35, L: 12.7, br_i20: 1.29, br_o20: 1.29,
            br_i_op: 1.24356, br_o_op: 1.24356, bond_outer: 0.05, cup_wall_corner: 1.8, c_end: 0.15, mu0: 1.256637e-6,
            gear_ratio: 5.0, gear_eff: 0.95, required_floor_Nm: 2.5, max_diameter_mm: 43.0,
        }
    }

    #[test]
    fn status_checks_are_strict_at_exact_equality() {
        // Architecture section 7 step 8. All four status comparisons at exact equality: the
        // strict `<` and `>` of sweeps.py fall through every check to "nominal: test needed".
        // Each right-hand side is a context value its left-hand side does not read.
        let (npole, a_i, gap) = (10, 10.15, 1.25);
        let n = npole as f64;
        let run = |c: &SweepContext| row(c, gap, npole, a_i, gap, 0.95, Deviations::NONE);
        let mut c = ctx();
        c.w_i = 2.0 * a_i * (PI / n).tan(); // the inner-flat expression of `row`, bit for bit
        c.w_o = 2.0 * run(&c).outer_face_apothem_mm * (PI / n).tan(); // F reads w_i, not w_o
        let second = run(&c);
        c.max_diameter_mm = second.cup_od_mm; // G reads neither limit
        c.required_floor_Nm = second.pullout_op_Nm; // X reads w_o (outer fill), already final
        let r = run(&c);
        assert_eq!((r.cup_od_mm, r.pullout_op_Nm), (c.max_diameter_mm, c.required_floor_Nm));
        assert_eq!(r.status, "nominal: test needed");
    }
}
```
(Checked on a scratch build: passes, and turning any of the four comparisons non-strict makes it fail.)

`api.rs` (Python lines 214-222). `mod.rs`: `pub mod sweeps;`. `DesignResults` gains after its groups:

```rust
        tables {
            gap_sweep: SweepRow => TableLayout::RowsDown { sheet: "Gap sweep", first_row: 6 },
            pole_sweep: SweepRow => TableLayout::RowsDown { sheet: "Pole sweep", first_row: 6 },
        }
```
and after `let clr = ...`:

```rust
    let ctx = sweeps::SweepContext {
        faceted: ci.faceted, backiron: ci.backiron, t_i: m.inner_thickness_mm, w_i: m.inner_width_mm,
        t_o: m.outer_thickness_mm, w_o: m.outer_width_mm, L: m.active_length_mm, br_i20: m.inner_br_T,
        br_o20: m.outer_br_T, br_i_op: m.br_inner_T_op, br_o_op: m.br_outer_T_op, bond_outer: md.bond_outer_mm,
        cup_wall_corner: md.cup_wall_corner_mm, c_end: ci.c_end, mu0: ci.mu0, gear_ratio: ci.gear_ratio,
        gear_eff: ci.gear_efficiency, required_floor_Nm: m.required_floor_Nm, max_diameter_mm: md.max_diameter_mm,
    };
    let gap = sweeps::gap_sweep(&ctx, ci.npole, ci.inner_back_apothem_mm, f_cal, dev);
    let pole = sweeps::pole_sweep(&ctx, m.corner_gap_mm, ci.bore_mm, ci.keyway_depth_mm, cal_in.f_cal_original, dev);
    DesignResults {
        calibration: cal, model: m, mass, retainers: ret, metal: mdr, materials: matr, temperature: temp, clamps: clr,
        gap_sweep: gap, pole_sweep: pole,
    }
```
(`SweepContext` has a field `L`: add `#[allow(non_snake_case)]` on `api::compute`.)

Generator: `MODULES["gap_sweep"] = ["coupling", "metal", "calibration"]`, `MODULES["pole_sweep"] = ["coupling", "metal", "calibration"]`; probes:

```python
    # Each status of the workbook's priority order (sweeps.py docstring).
    "gap_sweep": [
        ("small envelope: outside OD envelope", {"metal.max_diameter_mm": 20.0}),
        ("low requirement: nominal everywhere", {"metal.required_min_Nm": 0.1, "coupling.drive_torque_Nm": 0.0}),
        ("wide manual inner block: inner flat too narrow",
         {"coupling.magnets.part_inner": "", "coupling.magnets.manual_inner_width_mm": 25.4}),
        ("wide manual outer block: outer flat too narrow",
         {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 25.4}),
    ],
    "pole_sweep": [
        ("small envelope: outside OD envelope", {"metal.max_diameter_mm": 20.0}),
        ("low requirement: nominal everywhere", {"metal.required_min_Nm": 0.1, "coupling.drive_torque_Nm": 0.0}),
        ("wide manual outer block: outer flat too narrow",
         {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 25.4}),
    ],
```
`static_data()`: `"sweeps": {"gap_sweep_corner_gaps_mm": list(py_sweeps.GAP_SWEEP_CORNER_GAPS_MM), "pole_sweep_poles": list(py_sweeps.POLE_SWEEP_POLES)}` (import `from magcoupling import sweeps as py_sweeps`). `tests/static_data.rs`:

```rust
#[test]
fn sweep_variables_equal_the_python_lists() {
    use magcoupling::engine::sweeps::{GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES};
    let doc = python_static();
    let gaps: Vec<f64> = doc["sweeps"]["gap_sweep_corner_gaps_mm"].as_array().expect("gaps").iter().map(|g| g.as_f64().expect("a number")).collect();
    let poles: Vec<i64> = doc["sweeps"]["pole_sweep_poles"].as_array().expect("poles").iter().map(|p| p.as_i64().expect("an int")).collect();
    assert_eq!(GAP_SWEEP_CORNER_GAPS_MM.as_slice(), gaps.as_slice());
    assert_eq!(POLE_SWEEP_POLES.as_slice(), poles.as_slice());
}
```
`tests/differential.rs`: `#[test] fn gap_sweep_matches_python_on_every_case() { check_module("gap_sweep"); }`, `#[test] fn pole_sweep_matches_python_on_every_case() { check_module("pole_sweep"); }`, and

```rust
    ("gap_sweep[*].status", &[Text("inner flat too narrow"), Text("outer flat too narrow"), Text("outside OD envelope"), Text("below hot minimum"), Text("nominal: test needed")]),
    // "inner flat too narrow" cannot occur in the pole sweep: its apothem is chosen so that
    // 2 a_i tan(pi/N) >= w_i + 0.1 tan(pi/N) > w_i.
    ("pole_sweep[*].status", &[Text("outer flat too narrow"), Text("outside OD envelope"), Text("below hot minimum"), Text("nominal: test needed")]),
```
Regenerate (no schema change):

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote .../differential/gap_sweep.json (...)` and `pole_sweep.json`, each under 4 MB (gap_sweep about 2.7 MB). If gap_sweep exceeds the budget, the generator stops with the size: report it rather than raising the budget.

- [ ] **Step 3: Run parity and differential**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. Parity now covers the whole workbook: 330 scalar result cells, 659 table cells (165 + 338 + 156) and 160 default inputs = 1,149 checks; Gap sweep!AA6 = "nominal: test needed", AA9 = "below hot minimum", X9 = 2.4619000968698; Pole sweep!AA6 = "below hot minimum". Header and layout tests now also check both sweeps; `sweeps::tests::status_checks_are_strict_at_exact_equality` passes.

Mutation spot check: change `first_row: 6` of `pole_sweep` to `first_row: 7`: parity FAILS (every pole-sweep value lands one row low, and row 12 is not in the snapshot) and `tables_match_the_python_layout` FAILS. Revert.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task11.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task11.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README status: "every module except fields3d (M3) ported; workbook parity complete (1,149 checks)"; `docs/ai/03-structure.yaml` `ported` adds `sweeps.rs`, `remaining: [fields3d (M3)]`.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): port the gap and pole sweeps

sweeps.py: 338 Gap sweep and 156 Pole sweep cells, every status of the
workbook's priority order probed. Workbook parity now covers all 1,149
checks of test_parity.py.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---
### Task 12: API and schema completion (`api.py`)

First line: `validate()` depends on D3. If D3 is the `Result` alternative, stop and escalate.

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Modify: `magcoupling-rs/src/engine/meta.rs` (shared value check, `validate`)
- Modify: `magcoupling-rs/src/engine/api.rs` (`HEADLINE`, `headline`, `DesignInputs::validate`, module doc)
- Modify: `magcoupling-rs/src/lib.rs` (re-exports, doc example)
- Modify: `magcoupling-rs/tests/robustness.rs`, `tests/parity.rs`, `tests/python_schema.rs`
- Modify: `reference/magcoupling-py/tools/gen_differential.py` (headline export)
- Regenerate: `tests/data/python_schema.json`
- Modify: `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: the complete `DesignInputs`/`DesignResults` (Task 11); `InputSet`, `input_rows`, `result_rows`, `SetError`, `SetErrorKind` (slice).
- Produces:
  - `meta.rs`: `fn check_value(meta: &InputMeta, value: &Value) -> Option<SetErrorKind>` (private; used by `convert_input` and `validate`), `pub fn validate<T: InputSet>(inputs: &T) -> Result<(), Vec<SetError>>`.
  - `api.rs`: `pub const HEADLINE: [(&str, &str); 15]` (Python key, result path), `pub fn headline(results: &DesignResults) -> Vec<(&'static str, Value)>`, `impl DesignInputs { pub fn validate(&self) -> Result<(), Vec<SetError>> }`.
  - `lib.rs`: `pub use engine::api::{DesignInputs, DesignResults, compute_all, headline};`
  - Tests: `compute_all_never_panics_on_extreme_inputs`, `compute_all_never_panics_on_selector_codes_outside_the_choices`, `compute_all_is_cheap_enough_to_run_every_frame` (robustness.rs); `the_port_checks_every_cell_test_parity_checks` (parity.rs); `headline_matches_the_python_api`, `schemas_list_fields_in_the_python_order` (python_schema.rs).

- [ ] **Step 1: Write the failing tests.**

`tests/parity.rs`:

```rust
#[test]
fn the_port_checks_every_cell_test_parity_checks() {
    // test_parity.py: 330 result cells, 494 sweep cells, 165 screw-table cells, 160 default inputs = 1,149.
    let cells: usize = PORTED_RESULTS.iter().map(|p| p.cells).sum();
    let table_cells: usize = PORTED_RESULTS.iter().map(|p| p.table_cells).sum();
    let inputs: usize = PORTED_INPUTS.iter().map(|p| p.cells).sum();
    assert_eq!((cells, table_cells, inputs), (330, 659, 160));
    assert_eq!(cells + table_cells + inputs, 1149);
}
```

`tests/python_schema.rs`:

```rust
#[test]
fn headline_matches_the_python_api() {
    let doc = read_json(&data_path("python_schema.json"));
    let python: Vec<(String, Value)> = doc["headline"]
        .as_array()
        .expect("a headline array")
        .iter()
        .map(|pair| (pair[0].as_str().expect("a key").to_owned(), json_to_value(&pair[1])))
        .collect();
    let rust = headline(&compute_all_with(&DesignInputs::defaults_with(Deviations::NONE), Deviations::NONE));
    let keys: Vec<&str> = rust.iter().map(|(k, _)| *k).collect();
    let py_keys: Vec<&str> = python.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, py_keys, "keys and order");
    for ((key, r), (_, p)) in rust.iter().zip(&python) {
        assert!(parity_close(r, p), "{key}: rust={r:?} python={p:?}");
    }
}

#[test]
fn schemas_list_fields_in_the_python_order() {
    // The GUI's results table and CSV export follow this order. Table paths are filtered:
    // results! emits tables after the scalars (Python declares clamps.table mid-struct).
    let doc = read_json(&data_path("python_schema.json"));
    let python = |kind: &str| -> Vec<String> {
        doc["rows"]
            .as_array()
            .expect("rows")
            .iter()
            .filter(|r| r["kind"] == kind)
            .map(|r| r["path"].as_str().expect("a path").to_owned())
            .collect()
    };
    let inputs: Vec<String> = input_rows(&DesignInputs::default()).into_iter().map(|r| r.path).collect();
    let results: Vec<String> = result_rows(&compute_all(&DesignInputs::default()))
        .into_iter()
        .filter(|r| !is_table_path(&r.path))
        .map(|r| r.path)
        .collect();
    assert_eq!(inputs, python("input"));
    assert_eq!(results, python("result"));
}
```

`tests/robustness.rs` (add imports `magcoupling::{compute_all, engine::meta::{FieldType, InputSet, Value, input_rows}}`):

```rust
#[test]
fn compute_all_never_panics_on_extreme_inputs() {
    // Review Focus 3 and 5: typed values far outside the sliders (set() accepts them,
    // as Python does). Results may be inf or NaN; nothing may panic (no integer overflow).
    let base = DesignInputs::default();
    let mut tried = 0;
    for row in input_rows(&base) {
        let values: Vec<Value> = match (row.meta.ty, row.meta.choices.is_empty()) {
            (FieldType::F64 | FieldType::OptF64, _) => {
                let r = row.meta.range.expect("every numeric input has a range (tests/schema.rs)");
                [0.0, -1.0, r.min / 10.0, r.max * 10.0, 1e300, -1e300].map(Value::Num).to_vec()
            }
            // 2: Review Focus 3's `coupling.npole = 2` (tan(pi/2) is huge but finite).
            (FieldType::I64, true) => [0, 2, -2, 3, i64::MAX, i64::MIN].map(Value::Int).to_vec(),
            _ => continue, // selectors: next test; text: any text is valid (manual magnet)
        };
        for value in values {
            let mut inputs = base.clone();
            inputs.set(&row.path, value.clone()).unwrap_or_else(|e| panic!("{e}"));
            let _ = compute_all(&inputs);
            tried += 1;
        }
    }
    assert!(tried > 800, "only {tried} extreme cases");
}

#[test]
fn compute_all_never_panics_on_selector_codes_outside_the_choices() {
    // Review Focus 1: codes set on the struct, bypassing set(). validate() names every one.
    let mut inputs = DesignInputs::default();
    inputs.coupling.backiron = 7;
    inputs.coupling.faceted = -1;
    inputs.calibration.gap_definition = 9;
    inputs.temperature.adhesive.selected = 0;
    inputs.clamps.clamp_type = 3;
    inputs.clamps.alloy = 0;
    inputs.clamps.screw_class = i64::MIN;
    let res = compute_all(&inputs);
    assert_eq!(res.temperature.adhesive.selected_name, "#N/A");
    let errors = inputs.validate().expect_err("seven invalid codes");
    let paths: Vec<&str> = errors.iter().map(|e| e.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "coupling.backiron",
            "coupling.faceted",
            "calibration.gap_definition",
            "temperature.adhesive.selected",
            "clamps.clamp_type",
            "clamps.alloy",
            "clamps.screw_class",
        ]
    );
    assert!(DesignInputs::default().validate().is_ok());
    let mut nan = DesignInputs::default();
    nan.metal.face_gap_mm = f64::NAN;
    assert_eq!(nan.validate().expect_err("NaN")[0].path, "metal.face_gap_mm");
}

#[test]
fn compute_all_is_cheap_enough_to_run_every_frame() {
    // Spec: "milliseconds per call". Debug build, generous bound (a smoke check, not a benchmark).
    let inputs = DesignInputs::default();
    let start = std::time::Instant::now();
    for _ in 0..200 {
        std::hint::black_box(compute_all(std::hint::black_box(&inputs)));
    }
    let per_call = start.elapsed() / 200;
    assert!(per_call < std::time::Duration::from_millis(5), "{per_call:?} per call");
}
```

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test robustness --test python_schema --test parity 2>&1 | grep -E "error|FAILED|test result" | head
```
Expected: compile errors (`headline` and `validate` not found).

- [ ] **Step 2: Implement.**

`meta.rs` (replace the choices check in `convert_input` by the shared check; add `validate`):

```rust
/// Why `value` is not acceptable for the input `meta` describes, if it is not:
/// a selector code outside the choices, or NaN or an infinity. What `set()` and
/// `validate()` both check before the type conversion.
fn check_value(meta: &InputMeta, value: &Value) -> Option<SetErrorKind> {
    match *value {
        Value::Int(code) if !meta.choices.is_empty() && !meta.choices.iter().any(|&(c, _)| c == code) => {
            Some(SetErrorKind::NotAChoice { code })
        }
        Value::Num(x) if !x.is_finite() => Some(SetErrorKind::NotFinite),
        _ => None,
    }
}

/// Converts and validates a value for the input described by `meta` (used by `inputs!`).
#[doc(hidden)]
pub fn convert_input<T: InputValue>(meta: &InputMeta, value: Value) -> Result<T, SetError> {
    let error = |kind| SetError { path: meta.name.to_owned(), kind };
    if let Some(kind) = check_value(meta, &value) {
        return Err(error(kind));
    }
    T::from_value(value).map_err(error)
}

/// Checks every input the way [`InputSet::set`] checks one: selector codes among
/// their choices, numbers finite. For inputs built or edited without `set`
/// (struct literals, design files, share links). `compute_all` never panics on
/// invalid inputs, but its results are then meaningless: call this at input
/// boundaries. Errors name full paths, in schema order.
pub fn validate<T: InputSet>(inputs: &T) -> Result<(), Vec<SetError>> {
    let errors: Vec<SetError> = input_rows(inputs)
        .into_iter()
        .filter_map(|row| check_value(row.meta, &row.value).map(|kind| SetError { path: row.path, kind }))
        .collect();
    if errors.is_empty() { Ok(()) } else { Err(errors) }
}
```
Unit test in `meta.rs`:

```rust
    #[test]
    fn validate_reports_what_set_would_refuse() {
        let mut t = Top::default();
        assert_eq!(validate(&t), Ok(()));
        t.leaf.code = 5;
        t.leaf.drag_Nm = Some(f64::INFINITY);
        let errors = validate(&t).unwrap_err();
        assert_eq!(
            errors.iter().map(|e| (e.path.as_str(), e.kind.clone())).collect::<Vec<_>>(),
            [("leaf.code", SetErrorKind::NotAChoice { code: 5 }), ("leaf.drag_Nm", SetErrorKind::NotFinite)]
        );
    }
```

`api.rs`:

```rust
/// The numbers a dashboard shows first: (Python key, result path), in the order
/// of Python's `headline()` (`api.py` lines 131-150).
pub const HEADLINE: [(&str, &str); 15] = [
    ("pullout_at_op_temp_Nm", "model.pullout_Nm"),
    ("pullout_at_20C_Nm", "model.pullout_20C_Nm"),
    ("hot_low_with_variation_Nm", "metal.torque_hot_low_Nm"),
    ("hot_min_check", "metal.hot_min_check"),
    ("cold_high_with_variation_Nm", "metal.torque_cold_high_Nm"),
    ("gearbox_input_ripple_Nm", "model.gearbox_input_ripple_Nm"),
    ("cup_od_mm", "model.cup_od_mm"),
    ("rotating_mass_g", "mass.total_g"),
    ("running_clearance_mm", "metal.min_running_clearance_mm"),
    ("clearance_check", "metal.clearance_check"),
    ("cup_wall_check", "materials.cup_wall_check"),
    ("governing_temp_limit_C", "temperature.summary.governing_limit_C"),
    ("hot_day_margin_C", "temperature.summary.margin_hot_day_C"),
    ("temperature_verdict", "temperature.summary.verdict"),
    ("clamp_screw", "clamps.recommended"),
];

/// Python `headline(res)`: the dashboard numbers, keyed and ordered as in Python.
pub fn headline(results: &DesignResults) -> Vec<(&'static str, Value)> {
    let rows = result_rows(results);
    HEADLINE
        .iter()
        .map(|&(key, path)| (key, rows.iter().find(|r| r.path == path).map_or(Value::None, |r| r.value.clone())))
        .collect()
}

impl DesignInputs {
    /// Every selector code among its choices and every number finite (see
    /// [`crate::engine::meta::validate`]). Decision D3: `compute_all` never
    /// panics on invalid inputs; call this where inputs enter (design file, share link).
    pub fn validate(&self) -> Result<(), Vec<SetError>> {
        validate(self)
    }
}
```
Unit test in `api.rs`: `headline_names_existing_results` asserting `headline(&compute_all(&DesignInputs::default())).iter().all(|(_, v)| *v != Value::None)`. Update the module doc ("Ported: every module except fields3d (M3)"; add `headline`, `validate` to the Python API mapping).

`lib.rs`: re-export `headline`; replace the doc example (its `f_cal_updated ≈ 1.0658` changes when E3 is applied) with one that holds with and without corrections:

```rust
//! ```
//! use magcoupling::{DesignInputs, compute_all, headline};
//! let res = compute_all(&DesignInputs::default());
//! // The production-variation allowance lowers the hot-side torque below the nominal pull-out.
//! assert!(res.metal.torque_hot_low_Nm < res.model.pullout_Nm);
//! assert_eq!(headline(&res)[0].0, "pullout_at_op_temp_Nm");
//! ```
```

Generator `python_schema()`: add `"headline": [[key, plain(value, key)] for key, value in headline(res).items()]` (import `headline` from `magcoupling`). Regenerate:

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: only `python_schema.json` changes.

- [ ] **Step 3: Run all tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok, including the doc-test, `the_port_checks_every_cell_test_parity_checks`, `headline_matches_the_python_api`, `schemas_list_fields_in_the_python_order`, the three new robustness tests, `validate_reports_what_set_would_refuse`, `headline_names_existing_results`.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task12.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task12.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README: the example shows `headline`; a paragraph "Invalid inputs" (never panics; `validate()` at boundaries; out-of-choice codes give NaN or `#N/A`).

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
feat(magcoupling-rs): complete the API: headline, validate, totals

headline() in the Python order, DesignInputs::validate() for inputs that
bypass set(), a check that parity covers all 1,149 workbook checks, the
Python field order for both schemas, and never-panic tests over every
input at extreme values and every selector outside its choices.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 13: Differential coverage: every selector branch and the full run

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

Every module file already varies its upstream input groups, sets each selector to each choice, and reaches every branch listed in `BRANCHES`. This task closes the remaining gaps: (1) a `full` file that varies all 160 inputs at once and compares every result (all groups and tables), which also catches a `MODULES` entry that forgot an input group; (2) every pair of selector choices across groups; (3) a test that every text-producing result has a `BRANCHES` entry, so a new branch cannot land unchecked.

**Files:**
- Modify: `reference/magcoupling-py/tools/gen_differential.py`
- Modify: `magcoupling-rs/tests/differential.rs`
- Regenerate: new `tests/data/differential/full.json`
- Modify: `magcoupling-rs/README.md`

**Interfaces:**
- Consumes: `MODULES`, `module_file`, `run_case`, `sample` (Task 2); `check_module`, `BRANCHES`, `matches_pattern` (Task 2).
- Produces: generator constants `FULL = "full"`, `FULL_RANDOM = 100`, function `full_cases(fields, rng)`; data file `differential/full.json` (all input groups, all results); tests `full_run_matches_python_on_every_case`, `every_selector_pair_is_covered_in_the_full_run`, `every_selector_choice_appears_in_every_module_file`, `every_text_result_has_a_branches_entry`.

- [ ] **Step 1: Write the failing tests** in `tests/differential.rs`:

```rust
/// The data file whose cases vary every input group and compare every result.
const FULL: &str = "full";

#[test]
fn full_run_matches_python_on_every_case() {
    check_module(FULL);
}

/// Selector inputs and their codes, from the Rust metadata.
fn selectors() -> Vec<(String, Vec<i64>)> {
    input_rows(&DesignInputs::default())
        .into_iter()
        .filter(|r| !r.meta.choices.is_empty())
        .map(|r| (r.path, r.meta.choices.iter().map(|&(c, _)| c).collect()))
        .collect()
}

#[test]
fn every_selector_pair_is_covered_in_the_full_run() {
    let cases = load_cases(FULL);
    let selectors = selectors();
    let mut missing = Vec::new();
    for (i, (a, codes_a)) in selectors.iter().enumerate() {
        for (b, codes_b) in &selectors[i + 1..] {
            for &ca in codes_a {
                for &cb in codes_b {
                    let hit = cases.iter().any(|c| c.inputs[a] == Value::Int(ca) && c.inputs[b] == Value::Int(cb));
                    if !hit {
                        missing.push(format!("{a} = {ca} with {b} = {cb}"));
                    }
                }
            }
        }
    }
    assert!(missing.is_empty(), "{}", report(&missing));
}

#[test]
fn every_selector_choice_appears_in_every_module_file() {
    let mut missing = Vec::new();
    for p in PORTED_RESULTS {
        let cases = load_cases(p.group);
        for (path, codes) in selectors() {
            if !cases[0].inputs.contains_key(&path) {
                continue; // this module does not vary that group
            }
            for code in codes {
                if !cases.iter().any(|c| c.inputs[&path] == Value::Int(code)) {
                    missing.push(format!("{}: {path} never takes {code}", p.group));
                }
            }
        }
    }
    assert!(missing.is_empty(), "{}", report(&missing));
}

#[test]
fn every_text_result_has_a_branches_entry() {
    // A result that can be text (a verdict, a sentinel, a built message) must list
    // its branches in BRANCHES, so every branch is compared with Python at least once.
    let results = result_rows(&compute_all(&DesignInputs::default()));
    let missing: Vec<String> = results
        .iter()
        .filter(|r| matches!(r.meta.ty, FieldType::Text | FieldType::NumOrText))
        .filter(|r| !BRANCHES.iter().any(|(pattern, _)| matches_pattern(pattern, &r.path)))
        .map(|r| r.path.clone())
        .collect();
    assert!(missing.is_empty(), "text results without BRANCHES rows: {missing:?}");
}
```
In `check_module`, allow the smaller full file and compare all groups for it:

```rust
    let minimum = if module == FULL { 100 } else { 200 };
    assert!(cases.len() >= minimum, "{module}: only {} cases", cases.len());
```
and in `rust_results` filter with `module == FULL || group_of(&r.path) == module`. Imports: `compute_all`, `FieldType`, `input_rows`.

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test differential 2>&1 | grep -E "panicked|FAILED|test result"
```
Expected: FAIL: `full_run_matches_python_on_every_case` and `every_selector_pair_is_covered_in_the_full_run` panic reading the missing `differential/full.json`; `every_text_result_has_a_branches_entry` passes if every module task added its rows (if it names paths, add their rows now).

- [ ] **Step 2: Generate the full run.** In `gen_differential.py`:

```python
FULL = "full"       # every input group varied, every result compared
FULL_RANDOM = 100   # random cases of the full run (its fixed cases are defaults and selector pairs)


def full_cases(fields: list, rng: random.Random) -> list:
    """Defaults, random sets varying all inputs, then a case for every pair of
    selector choices (across all groups) the random sets missed."""
    defaults = {f["path"]: f["default"] for f in fields}
    cases = [("workbook defaults", dict(defaults))]
    for _ in range(FULL_RANDOM):
        cases.append(("random", {f["path"]: sample(f, rng) for f in fields}))
    selectors = [f for f in fields if f["choices"]]
    for i, a in enumerate(selectors):
        for b in selectors[i + 1:]:
            for ca, _ in a["choices"]:
                for cb, _ in b["choices"]:
                    if not any(c[a["path"]] == ca and c[b["path"]] == cb for _, c in cases):
                        cases.append((f"pair {a['path']} = {ca}, {b['path']} = {cb}",
                                      {**defaults, a["path"]: ca, b["path"]: cb}))
    return cases
```
In `module_file`, build the cases with `full_cases(fields, rng) if module == FULL else module_cases(module, fields, rng)`. In `run_case`, keep a row when `module == FULL or group_of(r["path"]) == module`. In `outputs()`, after the per-module loop:

```python
    all_groups = list(dict.fromkeys(group_of(f["path"]) for f in schema))  # schema order
    files[DATA / "differential" / f"{FULL}.json"] = module_file(FULL, all_groups, schema)
```
Update the module docstring (the full run) and the `--check` message (`+ full`).

```bash
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py
```
Expected: `wrote magcoupling-rs/tests/data/differential/full.json (...)` with 101 to about 125 cases, under 4 MB; no other data file changes.

- [ ] **Step 3: Run the full differential test**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test differential 2>&1 | grep -E "^test |test result"
```
Expected: every test `ok`: `calibration_`, `model_`, `retainers_`, `mass_`, `metal_`, `materials_`, `temperature_`, `clamps_`, `gap_sweep_`, `pole_sweep_matches_python_on_every_case`, `full_run_matches_python_on_every_case`, `every_branch_is_reached`, `every_varied_input_takes_two_values`, `every_selector_pair_is_covered_in_the_full_run`, `every_selector_choice_appears_in_every_module_file`, `every_text_result_has_a_branches_entry`, `helpers_match_python_on_the_corpus`, `every_ported_module_has_differential_data`, `branch_patterns_match_table_rows_only_by_index`. All cases match.

Spot checks: (1) temporarily delete the `model.verdict` row from `BRANCHES`: `every_text_result_has_a_branches_entry` FAILS naming `model.verdict`; restore. (2) In `api.rs` temporarily pass `&materials::AL6061` to `clamps::compute` whatever `clamps.alloy` says: `clamps_matches_python_on_every_case` and `full_run_matches_python_on_every_case` both FAIL on alloy-1 cases; restore.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task13.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task13.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: Docs and commit.** README tests table: the differential row mentions the full run, selector pairs and the BRANCHES completeness test.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add reference/magcoupling-py/tools/gen_differential.py magcoupling-rs/tests magcoupling-rs/README.md
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
test(magcoupling): full differential run and selector-pair coverage

A full run varies all 160 inputs at once and compares every result
(groups and tables) with Python; every pair of selector choices across
groups is covered; every text-producing result must list its branches.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---
## Approved corrections E1 to E14

Every task below follows the same shape (architecture.md section 6): a failing test first that asserts the corrected value at defaults at the audit report's precision (numbers from the report's "Effect at defaults" column; for E7 to E13, which change nothing at defaults, the report's off-default example as a registry probe); the correction behind `dev.is_on(DeviationId::Ek)` with the workbook form kept beside it; the registry entry set to `Applied` with its exact changes; a proof that `Deviations::NONE` still gives the snapshot value; the gate; one commit. Parity and differential tests stay on `Deviations::NONE` and must not change. Each task also adds its row to the "Differences from the workbook" table of `magcoupling-rs/README.md` (created in Task 14). Physics review: the controller dispatches a physics reviewer with the report row after each of Tasks 14 to 26.

A module that branches on a correction imports `super::deviations::{DeviationId, Deviations}` and renames its `_dev` parameter to `dev`; the snippets below assume both.

Exact values quoted below come from rerunning the Python oracle with the correction patched in memory (plan-writing aid); the Rust run must agree with them to 1e-9 relative, and the tests assert them at report precision.

### Task 14: E1, adhesive shear modulus

First line: depends on D2. If D2 is "per-adhesive modulus now", stop and escalate (needs TDS data and the A5 materials model).

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/deviations.rs` (new field `workbook_help`; E1 entry)
- Modify: `magcoupling-rs/src/engine/temperature.rs` (`MismatchInputs::adhesive_shear_modulus_GPa` default and help)
- Modify: `magcoupling-rs/src/engine/api.rs` (unit test)
- Modify: `magcoupling-rs/tests/deviations.rs`, `tests/python_schema.rs`
- Regenerate: `tests/data/input_schema.json` (help text)
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: `REGISTRY`, `Deviation`, `CellChange`, `Literal`, `restore_workbook_defaults`, `DesignInputs::defaults_with` (slice); `cell_values(dev) -> BTreeMap<String, Value>` (Task 9).
- Produces:
  - `Deviation.workbook_help: &'static [(&'static str, &'static str)]`: help texts the correction rewords, as (input path or table column path `group.table[*].field`, workbook text).
  - Test helpers in `tests/deviations.rs` used by every later correction task: `fn cell_values_for(inputs: &DesignInputs, dev: Deviations) -> BTreeMap<String, Value>` (`cell_values(dev)` becomes `cell_values_for(&DesignInputs::defaults_with(dev), dev)`), `fn at(cell: &str, dev: Deviations) -> Value`, `fn num(value: &Value) -> f64`, `fn assert_report(cell: &str, got: &Value, want: f64, half_step: f64)`, `fn assert_workbook(cell: &str)`.
  - `tests/python_schema.rs`: `fn reworded_help() -> BTreeMap<&'static str, &'static str>` (applied entries' `workbook_help`).

- [ ] **Step 1: Write the failing test.** In `tests/deviations.rs`, refactor `cell_values` and add the helpers and the E1 test:

```rust
/// Every value with a workbook cell (inputs and results) for `inputs` under `dev`.
fn cell_values_for(inputs: &DesignInputs, dev: Deviations) -> BTreeMap<String, Value> {
    let results = compute_all_with(inputs, dev);
    let mut cells = BTreeMap::new();
    for row in input_rows(inputs) {
        if let Some(cell) = row.meta.cell {
            cells.insert(cell.to_owned(), row.value);
        }
    }
    for row in result_rows(&results) {
        if let Some(cell) = row.cell {
            cells.insert(cell, row.value);
        }
    }
    cells
}

/// Every value with a workbook cell at the defaults `dev` implies.
fn cell_values(dev: Deviations) -> BTreeMap<String, Value> {
    cell_values_for(&DesignInputs::defaults_with(dev), dev)
}

/// The value at one workbook cell at the defaults `dev` implies.
fn at(cell: &str, dev: Deviations) -> Value {
    cell_values(dev).remove(cell).unwrap_or_else(|| panic!("{cell} is not a cell of the port"))
}

fn num(value: &Value) -> f64 {
    match value {
        Value::Num(x) => *x,
        Value::Int(i) => *i as f64,
        other => panic!("expected a number, got {other:?}"),
    }
}

/// `got` equals the audit report's figure `want`, which the report states to
/// within `half_step` (half a unit of its last digit).
fn assert_report(cell: &str, got: &Value, want: f64, half_step: f64) {
    let got = num(got);
    assert!((got - want).abs() <= half_step, "{cell}: {got} is not the report's {want} (± {half_step})");
}

/// With every correction off, the cell still holds the workbook snapshot value.
fn assert_workbook(cell: &str) {
    let snapshot = snapshot();
    assert!(
        parity_close(&at(cell, Deviations::NONE), &snapshot[cell]),
        "{cell}: the workbook-exact switch lost the snapshot value"
    );
}

#[test]
fn e1_adhesive_shear_modulus_matches_the_report() {
    let e1 = Deviations::only(DeviationId::E1);
    assert_eq!(at("Temperature design!C96", e1), Value::Num(0.107));
    assert_report("Temperature design!C104", &at("Temperature design!C104", e1), 11.6, 0.05); // was 46.1
    assert_report("Temperature design!C105", &at("Temperature design!C105", e1), 6.0, 0.05); // was 26.7
    assert_report("Temperature design!C201", &at("Temperature design!C201", e1), 2.6, 0.05); // was 11.4
    assert_eq!(at("Temperature design!C106", e1), Value::Text("Below the lap-shear strength".into()));
    assert_eq!(at("Temperature design!C202", e1), Value::Text("Below the fatigue endurance".into()));
    for cell in [
        "Temperature design!C96",
        "Temperature design!C104",
        "Temperature design!C105",
        "Temperature design!C106",
        "Temperature design!C201",
        "Temperature design!C202",
    ] {
        assert_workbook(cell);
    }
}
```

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations e1_ 2>&1 | grep -E "panicked|left|right|test result"
```
Expected: FAIL at the first assertion (`left: Num(0.55)`, `right: Num(0.107)`).

- [ ] **Step 2: Implement.**
  - `temperature.rs`: `adhesive_shear_modulus_GPa: f64 = 0.107 => param("GPa", "Adhesive shear modulus", "Loctite AA 326 + SF 7649 (Henkel TDS, Aug-2020): tensile modulus 0.300 GPa, so G = E/(2(1+ν)) ≈ 0.107 GPa at ν = 0.4. Correction E1: the workbook's 0.55 GPa is the modulus of EA 9514.", "Temperature design!C96")` with its existing range and `.log()`.
  - `deviations.rs`: add to `Deviation`

```rust
    /// Help texts this correction rewords, as (input path, or table column path
    /// `group.table[*].field`, and the workbook's text). The metadata tests
    /// compare the recorded text with Python and the workbook, and require the
    /// port's help to differ.
    pub workbook_help: &'static [(&'static str, &'static str)],
```
    and `workbook_help: &[],` to every entry and to the unit test's `FAKE`. Extend the unit test `planned_entries_change_nothing_yet` with `assert!(d.workbook_help.is_empty(), "{} is Planned but rewords help", d.id);`. The E1 entry becomes:

```rust
    Deviation {
        id: DeviationId::E1,
        title: "Adhesive shear modulus does not match the selected adhesive",
        class: DeviationClass::Engine,
        status: DeviationStatus::Applied,
        cells: &[ /* unchanged six cells */ ],
        corrected_formula: "Temperature design!C96 = 0.107 GPa (Loctite AA 326: E = 0.300 GPa, nu = 0.4). \
            The per-adhesive shear modulus the report also suggests is deferred to Addendum A5 (decision D2).",
        workbook_input_defaults: &[("temperature.mismatch.adhesive_shear_modulus_GPa", Literal::Num(0.55))],
        workbook_help: &[("temperature.mismatch.adhesive_shear_modulus_GPa", "")],
        changes_at_defaults: &[
            CellChange { cell: "Temperature design!C96", workbook: Literal::Num(0.55), corrected: Literal::Num(0.107) },
            CellChange { cell: "Temperature design!C104", workbook: Literal::Num(46.12653767935644), corrected: Literal::Num(11.59827175128882) },
            CellChange { cell: "Temperature design!C105", workbook: Literal::Num(26.728261752019268), corrected: Literal::Num(6.027792167235714) },
            CellChange {
                cell: "Temperature design!C106",
                workbook: Literal::Text("Above the lap-shear strength at the block ends"),
                corrected: Literal::Text("Below the lap-shear strength"),
            },
            CellChange { cell: "Temperature design!C201", workbook: Literal::Num(11.365659614702167), corrected: Literal::Num(2.563198259452589) },
            CellChange {
                cell: "Temperature design!C202",
                workbook: Literal::Text("Above the fatigue endurance: qualify by thermal cycling"),
                corrected: Literal::Text("Below the fatigue endurance"),
            },
        ],
    },
```
  - `tests/deviations.rs`: add

```rust
#[test]
fn reworded_help_is_recorded_for_real_fields() {
    let inputs = input_rows(&DesignInputs::default());
    for d in REGISTRY.iter().filter(|d| d.status == DeviationStatus::Applied) {
        for &(path, workbook) in d.workbook_help {
            if path.contains("[*]") {
                continue; // table columns: checked against the workbook headers in tests/schema.rs
            }
            let row = inputs.iter().find(|r| r.path == path).unwrap_or_else(|| panic!("{}: {path} is not an input", d.id));
            assert_ne!(row.meta.help, workbook, "{}: {path} help is not reworded", d.id);
        }
    }
}
```
  - `tests/python_schema.rs`: add

```rust
/// Help texts an applied correction rewords: path -> the workbook (and Python) text.
fn reworded_help() -> BTreeMap<&'static str, &'static str> {
    REGISTRY
        .iter()
        .filter(|d| d.status == DeviationStatus::Applied)
        .flat_map(|d| d.workbook_help.iter().copied())
        .collect()
}
```
    and in `ported_inputs_carry_the_python_metadata_and_defaults` pass the recorded workbook help to `compare` instead of the Rust help when the path is in `reworded_help()` (Python still has the workbook text).
  - `api.rs` unit test: replace `workbook_defaults_equal_the_defaults_while_no_default_is_corrected` with

```rust
    #[test]
    fn workbook_defaults_put_back_every_corrected_default() {
        assert_eq!(DesignInputs::defaults_with(Deviations::ALL), DesignInputs::default());
        let workbook = DesignInputs::defaults_with(Deviations::NONE);
        assert_eq!(workbook.temperature.mismatch.adhesive_shear_modulus_GPa, 0.55);
        assert_eq!(DesignInputs::default().temperature.mismatch.adhesive_shear_modulus_GPa, 0.107);
    }
```
  - Bless the schema (the help text changed): `MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test schema`. The generator reads only types, defaults, choices and ranges, so `gen_differential.py --check` stays current.

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py --check
```
Expected: all ok (including `e1_adhesive_shear_modulus_matches_the_report`, `each_deviation_alone_changes_exactly_its_registered_cells`, `all_deviations_together_change_only_registered_cells`, `registered_workbook_values_equal_the_snapshot`, `reworded_help_is_recorded_for_real_fields`, parity unchanged); the generator prints `differential data is current (...)`.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task14.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task14.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Add a section "Differences from the workbook" to `magcoupling-rs/README.md`: one paragraph (every correction is approved in the M1 report, registered in `deviations.rs`, on for users, off only in the parity tests) and a table `| Id | Cells | Workbook | This port | Report |` with the E1 row: `Temperature design!C96 (feeds C104-C106, C201, C202)` | `0.55 GPa (EA 9514's modulus)` | `0.107 GPa (AA 326 TDS); C106 and C202 now read "Below ..."` | `E1`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1 Applied, E2-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E1 adhesive shear modulus

Temperature design!C96 defaults to 0.107 GPa, the shear modulus of the
selected Loctite AA 326 (TDS tensile modulus 0.300 GPa, nu 0.4), instead
of 0.55 GPa (EA 9514). Peak end shear falls from 46.1 to 11.6 MPa and the
C106 and C202 screens read "Below ...". Registry records the reworded help.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 15: E2, clamp screw length includes the slit

First line: depends on D6 (Rust-only `clamps.length_note`). If D6 is the alternative, stop and escalate.

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/meta.rs` (`rust_only` results)
- Modify: `magcoupling-rs/src/engine/clamps.rs`
- Modify: `magcoupling-rs/src/engine/deviations.rs` (E2 entry)
- Modify: `magcoupling-rs/tests/deviations.rs`, `tests/python_schema.rs`, `tests/schema.rs`, `tests/differential.rs`
- Regenerate: none (`input_schema.json` has no results)
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers (`at`, `num`, `assert_report`, `assert_workbook`), `Deviation.workbook_help`.
- Produces:
  - `ResultMeta.rust_only: bool` and `pub const fn out_rust_only(unit: &'static str, label: &'static str, help: &'static str) -> ResultMeta` (no cell; the metadata-parity, order and differential tests skip it).
  - `ClampResults.length_note: String` (Rust-only, after `recommended`).

- [ ] **Step 1: Write the failing test** in `tests/deviations.rs`:

```rust
#[test]
fn e2_clamp_screw_length_matches_the_report() {
    let e2 = Deviations::only(DeviationId::E2);
    assert_eq!(at("Shaft clamps!C48", e2), Value::Text("ISO 4762 M4 x 14, class 12.9".into()));
    assert_eq!(num(&at("Clamp screw sizes!E34", e2)), 14.0);
    assert_eq!(num(&at("Clamp screw sizes!E35", e2)), 0.0, "the 14 mm screw protrudes from the 25 mm boss");
    // Screw strength limits the preload, so capacity and safety factor do not change.
    assert_report("Shaft clamps!C52", &at("Shaft clamps!C52", e2), 7.665, 0.0005);
    assert_report("Shaft clamps!C53", &at("Shaft clamps!C53", e2), 2.04, 0.005);
    let note = |dev| compute_all_with(&DesignInputs::defaults_with(dev), dev).clamps.length_note;
    assert_eq!(
        note(e2),
        "No 2 mm length step of M4 both engages 8 mm of thread and stays inside the boss; M4 x 14 protrudes 0.34 mm"
    );
    assert_eq!(note(Deviations::NONE), "");
    for cell in ["Shaft clamps!C48", "Clamp screw sizes!E34", "Clamp screw sizes!E35"] {
        assert_workbook(cell);
    }
}
```

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations e2_ 2>&1 | grep -E "error|panicked|test result" | head -3
```
Expected: compile error (`no field length_note`).

- [ ] **Step 2: Implement.**
  - `meta.rs`: add `pub rust_only: bool` to `ResultMeta` (doc: "A result the Python engine does not have: no workbook cell; metadata-parity, order and differential tests skip it."); `out`, `out_uncelled` and `column_meta` set `rust_only: false`; add

```rust
/// Declares a result the Python engine does not have (no cell), e.g. a
/// correction's explanatory note.
pub const fn out_rust_only(unit: &'static str, label: &'static str, help: &'static str) -> ResultMeta {
    ResultMeta { name: "", ty: FieldType::F64, unit, label, help, cell: None, rust_only: true }
}
```
    plus a unit test asserting `out_rust_only("", "L", "").rust_only && out("", "L", "", "S!C1").rust_only == false`.
  - `clamps.rs`: `length_note: String => out_rust_only("", "Screw length note", "Set when no 2 mm length step of the recommended size both engages the required thread and stays inside the boss (correction E2); empty otherwise.")` after `recommended`. The `length_mm` column help becomes `"Grip plus slit plus required engagement, rounded up to an even length (correction E2)."`. In `compute`, inside the per-size closure:

```rust
            // E2: the screw crosses the open slit before it reaches the far jaw.
            let e2 = dev.is_on(DeviationId::E2);
            let length = if fits == 1 {
                if e2 { ceiling(grip + ci.slit_mm + ereq, 2.0) } else { ceiling(grip + ereq, 2.0) }
            } else {
                0.0
            };
            let inside = if e2 { grip + ci.slit_mm + avail } else { grip + avail };
```
    with `length_ok: if fits == 1 && length <= inside { 1 } else { 0 }`; rename `_dev` to `dev`; and after `recommended`:

```rust
    let length_note = match chosen {
        Some(r) if dev.is_on(DeviationId::E2) && r.length_ok == 0 => format!(
            "No 2 mm length step of {} both engages {} mm of thread and stays inside the boss; {} x {} protrudes {} mm",
            r.size,
            fmt_num(r.engagement_req_mm),
            r.size,
            fmt_num(r.length_mm),
            fmt_fixed(r.length_mm - (r.grip_mm + ci.slit_mm + r.thread_avail_mm), 2),
        ),
        _ => String::new(),
    };
```
  - `deviations.rs` E2 entry: `status: Applied`; `corrected_formula` adds "; the note is the Rust-only result clamps.length_note (decision D6)"; `workbook_help: &[("clamps.table[*].length_mm", "Grip plus required engagement, rounded up to an even length.")]`; `changes_at_defaults`:

```rust
        changes_at_defaults: &[
            CellChange { cell: "Clamp screw sizes!E34", workbook: Literal::Num(12.0), corrected: Literal::Num(14.0) },
            CellChange { cell: "Clamp screw sizes!E35", workbook: Literal::Int(1), corrected: Literal::Int(0) },
            CellChange {
                cell: "Shaft clamps!C48",
                workbook: Literal::Text("ISO 4762 M4 x 12, class 12.9"),
                corrected: Literal::Text("ISO 4762 M4 x 14, class 12.9"),
            },
        ],
```
  - Skip Rust-only results: `tests/python_schema.rs` (`ported_results_carry_the_python_metadata`, `schemas_list_fields_in_the_python_order`: filter `!r.meta.rust_only`); `tests/differential.rs::rust_results` (filter `!r.meta.rust_only`); `tests/differential.rs::every_text_result_has_a_branches_entry` (Task 13): insert `.filter(|r| !r.meta.rust_only)` before the `BRANCHES` filter. `length_note` is a `String` (`FieldType::Text`) with no Python counterpart, so without the filter that test reports it, and a `BRANCHES` row for it would fail `every_branch_is_reached` ("no such result in differential/clamps.json"). The test body becomes:

```rust
    let missing: Vec<String> = results
        .iter()
        .filter(|r| matches!(r.meta.ty, FieldType::Text | FieldType::NumOrText))
        .filter(|r| !r.meta.rust_only) // no Python counterpart, so no differential data to reach
        .filter(|r| !BRANCHES.iter().any(|(pattern, _)| matches_pattern(pattern, &r.path)))
        .map(|r| r.path.clone())
        .collect();
```
    `tests/schema.rs`: add `assert!(r.meta.cell.is_none())` for every `rust_only` result in `every_field_has_a_label_and_well_formed_unique_path_and_cell`.
  - `tests/schema.rs::table_columns_match_the_workbook_headers`: build `reworded` = applied entries' `workbook_help`; for the help of a column whose pattern (`row.path.replacen("[0]", "[*]", 1)`) is in `reworded`, require the recorded text to equal the workbook I-cell and the Rust help to differ, instead of comparing the Rust help.

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok, including `e2_clamp_screw_length_matches_the_report`, `every_text_result_has_a_branches_entry` (it skips the Rust-only `clamps.length_note`) and `every_branch_is_reached`; parity and differential unchanged (they run with `Deviations::NONE`, where `length_note` is `""` and lengths are the workbook's).

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task15.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task15.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences table row E2: `Clamp screw sizes!C34:G35, Shaft clamps!C48` | `length = CEILING(grip + 2d, 2); fits = length <= grip + thread` | `the slit (0.8 mm) is added to both; M4 x 12 becomes M4 x 14, which protrudes 0.34 mm: clamps.length_note says so` | `E2`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E2 Applied, E3-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E2 clamp screw length includes the slit

Screw length and the fits-inside check add the open clamp slit. The
recommendation becomes ISO 4762 M4 x 14, class 12.9; no 2 mm step both
engages 2 x d and stays inside the 25 mm boss, which the new Rust-only
result clamps.length_note reports.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 16: E3, N42SH library remanence

First line: depends on D1 (value) and D4 (golden file). This task gives both D1 variants; D4's alternative (hand-listing about 250 cells) stops the task for escalation.

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/library.rs` (`N42SH_BR_CORRECTED_T`, `br_T`)
- Modify: `magcoupling-rs/src/engine/model.rs` (`resolve_magnets`; manual Br defaults)
- Modify: `magcoupling-rs/src/engine/calibration.rs` (`br_T` default; unit test)
- Modify: `magcoupling-rs/src/engine/api.rs` (unit test)
- Modify: `magcoupling-rs/src/engine/deviations.rs` (new field `changes_file`; E3 entry)
- Modify: `magcoupling-rs/tests/deviations.rs`
- Create (blessed): `magcoupling-rs/tests/data/deviations/E3.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; `lookup`, `MagnetSpec` (Task 1).
- Produces:
  - `library.rs`: `pub const N42SH_BR_CORRECTED_T: f64` (1.30 for D1 = 1.30 T, 1.315 for the nominal variant), `pub fn br_T(spec: &MagnetSpec, dev: Deviations) -> f64`.
  - `Deviation.changes_file: Option<&'static str>`: path (relative to `magcoupling-rs/`) of the golden file listing every cell a broad correction changes at defaults, `{"changes": {"cell": [workbook, corrected]}}`.
  - `tests/deviations.rs`: `fn registered_changes(d: &Deviation) -> BTreeMap<String, (Value, Value)>`, bless mode for golden files (`MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells`).

- [ ] **Step 1: Write the failing test** in `tests/deviations.rs` (D1 = 1.30 T; the nominal variant's numbers follow):

```rust
#[test]
fn e3_library_remanence_matches_the_report() {
    let e3 = Deviations::only(DeviationId::E3);
    assert_eq!(at("Calculator!C21", e3), Value::Num(1.30));
    assert_report("Calculator!C93", &at("Calculator!C93", e3), 2.688, 0.0005); // pull-out, was 2.647
    assert_report("Metal design!C9", &at("Metal design!C9", e3), 2.285, 0.0005); // hot low, was 2.250
    assert_eq!(at("Metal design!C11", e3), Value::Text("Below hot minimum".into()));
    assert_report("Metal design!C10", &at("Metal design!C10", e3), 3.823, 0.0005); // cold high, was 3.765
    assert_report("Temperature design!C12", &at("Temperature design!C12", e3), 93.06, 0.005); // limit, was 92.55
    assert_eq!(at("Temperature design!C91", e3), Value::Text("OK: 7x margin".into()));
    assert_eq!(at("Gap sweep!AA9", e3), Value::Text("nominal: test needed".into())); // 1.25 mm row
    assert_report("Gap sweep!X9", &at("Gap sweep!X9", e3), 2.500, 0.0005); // was 2.462
    assert_eq!(at("Shaft clamps!C48", e3), Value::Text("ISO 4762 M4 x 12, class 12.9".into()), "below 1.3025 T the clamp still fits");
    for cell in ["Calculator!C21", "Calculator!C93", "Metal design!C9", "Temperature design!C12", "Temperature design!C91", "Gap sweep!AA9"] {
        assert_workbook(cell);
    }
}
```
D1 = about 1.315 T variant: `C21 = 1.315`; C93 2.751, Metal design!C9 2.338, C10 3.912 (each ± 0.0005); Temperature design!C12 93.82 (± 0.005); C91 `"OK: 7x margin"`; Gap sweep!AA9 `"nominal: test needed"`, X9 2.558 (± 0.0005); Shaft clamps!C48 `"None: enlarge the boss or the clamp length"` (the report: from 1.3025 T up).

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations e3_ 2>&1 | grep -E "panicked|left|right|test result"
```
Expected: FAIL (`left: Num(1.29)`).

- [ ] **Step 2: Implement.**
  - `library.rs`:

```rust
use super::deviations::{DeviationId, Deviations};

/// E3: the N42SH rows' remanence with the approved correction: the vendor's
/// grade minimum (K&J 13.0 kG), like the library's other K&J rows (decision D1).
pub const N42SH_BR_CORRECTED_T: f64 = 1.30;

/// The 20 °C remanence the Calculator uses for a library part: the row's
/// workbook value, or with E3 on the corrected N42SH value.
#[allow(non_snake_case)]
pub fn br_T(spec: &MagnetSpec, dev: Deviations) -> f64 {
    if dev.is_on(DeviationId::E3) && spec.grade == "N42SH" { N42SH_BR_CORRECTED_T } else { spec.br_T }
}
```
    Update the module doc: rows keep the workbook value; `br_T` applies E3; fields3d must be rerun with the new Br (M3).
  - `model.rs`: `resolve_magnets` uses `br_T: library::br_T(spec, dev)` (rename `_dev` to `dev`); `manual_inner_br_T` and `manual_outer_br_T` default to `1.30` (Calculator C17, C27 follow the library per the report).
  - `calibration.rs`: `br_T` defaults to `1.30` (Calibration!C21); the unit test `default_design_reproduces_the_bench_correction` builds `CalibrationInputs { br_T: 1.29, ..CalibrationInputs::default() }` and runs with `Deviations::NONE` (its numbers are the workbook's).
  - `api.rs` unit test `paths_match_the_python_api`: read `calibration.br_T` from `DesignInputs::defaults_with(Deviations::NONE)` (1.29) and assert `DesignInputs::default()` gives 1.30.
  - `deviations.rs`: add

```rust
    /// For a broad correction (decision D4: more than 15 cells change at
    /// defaults): the golden file, relative to `magcoupling-rs/`, that lists
    /// every changed cell as `{"cell": [workbook, corrected]}`. Rewritten by
    /// `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` and reviewed as a diff.
    /// `changes_at_defaults` stays empty for such entries.
    pub changes_file: Option<&'static str>,
```
    with `changes_file: None,` on every entry and on `FAKE`; `planned_entries_change_nothing_yet` also asserts `d.changes_file.is_none()`. E3 entry: `status: Applied`; `corrected_formula: "Br of library rows B842SH, BX042SH and BX082SH = 1.30 T (vendor grade minimum; decision D1), with Calculator!C17, C27 and Calibration!C21 to match. Rerunning fields3d with the new Br is M3 scope."`; `workbook_input_defaults: &[("coupling.magnets.manual_inner_br_T", Literal::Num(1.29)), ("coupling.magnets.manual_outer_br_T", Literal::Num(1.29)), ("calibration.br_T", Literal::Num(1.29))]`; `changes_file: Some("tests/data/deviations/E3.json")`.
  - `tests/deviations.rs`: golden-file support (imports `std::fs`, `std::path::Path`, `common::{json_to_value, read_json, value_to_json}`, `magcoupling::engine::deviations::Deviation`):

```rust
const BLESS_VAR: &str = "MAGCOUPLING_BLESS";

fn golden_path(file: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(file)
}

/// The cells a correction changes at defaults, with their workbook and corrected
/// values: hand-listed in the registry, or read from the correction's golden file.
fn registered_changes(d: &Deviation) -> BTreeMap<String, (Value, Value)> {
    match d.changes_file {
        None => d
            .changes_at_defaults
            .iter()
            .map(|c| (c.cell.to_owned(), (c.workbook.to_value(), c.corrected.to_value())))
            .collect(),
        Some(file) if !golden_path(file).exists() => BTreeMap::new(),
        Some(file) => read_json(&golden_path(file))["changes"]
            .as_object()
            .expect("a changes object")
            .iter()
            .map(|(cell, pair)| (cell.clone(), (json_to_value(&pair[0]), json_to_value(&pair[1]))))
            .collect(),
    }
}

/// Bless mode: writes the golden file of a broad correction from this run.
fn bless_changes(d: &Deviation, workbook: &BTreeMap<String, Value>, corrected: &BTreeMap<String, Value>, changed: &BTreeSet<String>) {
    let file = d.changes_file.expect("only golden-file entries are blessed");
    let changes: serde_json::Map<String, serde_json::Value> = changed
        .iter()
        .map(|cell| (cell.clone(), serde_json::json!([value_to_json(&workbook[cell]), value_to_json(&corrected[cell])])))
        .collect();
    let doc = serde_json::json!({
        "about": format!("Every workbook cell correction {} changes at default inputs, as [workbook, corrected]. \
                          Written by `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells`; review the diff.", d.id),
        "id": d.id.to_string(),
        "changes": changes,
    });
    let path = golden_path(file);
    fs::create_dir_all(path.parent().expect("a parent directory")).expect("create tests/data/deviations");
    fs::write(&path, serde_json::to_string_pretty(&doc).expect("serializes") + "\n").expect("write the golden file");
}
```
    Rewrite `each_deviation_alone_changes_exactly_its_registered_cells` on top of them:

```rust
#[test]
fn each_deviation_alone_changes_exactly_its_registered_cells() {
    let workbook = cell_values(Deviations::NONE);
    let mut failures = Vec::new();
    for id in DeviationId::ALL {
        let d = &REGISTRY[id.index()];
        let corrected = cell_values(Deviations::only(id));
        let changed = changed_cells(&workbook, &corrected);
        if d.changes_file.is_some() && std::env::var_os(BLESS_VAR).is_some() {
            bless_changes(d, &workbook, &corrected, &changed);
        }
        let registered = registered_changes(d);
        let keys: BTreeSet<String> = registered.keys().cloned().collect();
        if changed != keys {
            failures.push(format!("{id}: changed {changed:?}, registered {keys:?}"));
        }
        for (cell, (_, want)) in &registered {
            match corrected.get(cell) {
                Some(got) if parity_close(got, want) => {}
                got => failures.push(format!("{id}: {cell} = {got:?}, registered {want:?}")),
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}
```
    Use `registered_changes` in `registered_workbook_values_equal_the_snapshot` (every workbook value equals the snapshot) and `all_deviations_together_change_only_registered_cells` (the union of their keys); in `applied_engine_entries_list_their_changes_at_defaults_or_are_default_neutral` keep the `cells` check for hand-listed entries only. Add:

```rust
#[test]
fn broad_corrections_use_golden_files_and_narrow_ones_list_their_cells() {
    // Decision D4: more than 15 changed cells go to a golden file.
    for d in REGISTRY.iter().filter(|d| d.status == DeviationStatus::Applied) {
        match d.changes_file {
            Some(file) => {
                assert!(golden_path(file).exists(), "{}: {file} missing: bless it", d.id);
                assert!(d.changes_at_defaults.is_empty(), "{}: golden file and hand list", d.id);
                assert!(registered_changes(d).len() > 15, "{}: a golden file for 15 cells or fewer", d.id);
            }
            None => assert!(d.changes_at_defaults.len() <= 15, "{}: more than 15 hand-listed cells", d.id),
        }
    }
}
```

- [ ] **Step 3: Bless the golden file, review it, run the tests**

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations each_deviation_alone_changes_exactly_its_registered_cells
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe -c "import json; d=json.load(open('C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/tests/data/deviations/E3.json', encoding='utf-8')); print(len(d['changes']))"
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
The bless run is filtered to the test that writes the golden files: the whole `deviations` binary would also run `broad_corrections_use_golden_files_and_narrow_ones_list_their_cells`, which can run before the writer and fail on a missing golden file, muddying the bless output (Tasks 17 and 18 bless the same way).
Expected: the bless run prints `1 passed`; the golden file lists **246** cells (D1 = 1.30 T; **265** for about 1.315 T), including the inputs `Calculator!C17`, `C27`, `Calibration!C21`, the resolved `Calculator!C21`, `C31`, and `Calculator!C93` as `[2.64727420272149..., 2.68847629505397...]` (workbook from the NONE run, corrected from only(E3)); the physics reviewer reads the diff. Then every test ok; parity and differential unchanged.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task16.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task16.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E3: `Magnet library N42SH rows; Calculator!C17, C21, C27, C31; Calibration!C21` | `1.29 T` | `1.30 T (vendor minimum); pull-out 2.688 N·m, limit 93.06 °C, C91 "OK: 7x margin"; fields3d rerun pending (M3)` | `E3`. README "Deviations" section: golden files and how to bless and review them. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E3 Applied, E4-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E3 N42SH library remanence

B842SH, BX042SH and BX082SH resolve to Br = 1.30 T, the vendor's grade
minimum, and the manual and calibration Br defaults follow. Pull-out rises
to 2.688 N m and the governing limit to 93.06 C. The 246 changed cells are
a reviewed golden file (tests/data/deviations/E3.json).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```
(For D1 = 1.315 T, write 1.315 T, 2.751 N m, 93.82 C and 265 cells.)

---

### Task 17: E4, pole-sweep hub wall excludes the bondline

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/sweeps.rs` (`SweepContext.bond_inner`, `pole_sweep`, the unit test's `ctx()` fixture)
- Modify: `magcoupling-rs/src/engine/api.rs` (fills `bond_inner`)
- Modify: `magcoupling-rs/src/engine/deviations.rs` (E4 entry)
- Modify: `magcoupling-rs/tests/deviations.rs`
- Create (blessed): `magcoupling-rs/tests/data/deviations/E4.json`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; golden-file support (Task 16).
- Produces: `SweepContext.bond_inner: f64` (Metal design C120; read only by E4).

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e4_pole_sweep_hub_wall_matches_the_report() {
    let e4 = Deviations::only(DeviationId::E4);
    for cell in ["Pole sweep!C6", "Pole sweep!C7"] {
        assert_report(cell, &at(cell, e4), 9.250, 0.0005); // 6 and 8 poles, was 9.200
    }
    assert_eq!(at("Pole sweep!AA6", e4), Value::Text("outside OD envelope".into())); // was "below hot minimum"
    assert_report("Pole sweep!G6", &at("Pole sweep!G6", e4), 43.01, 0.005); // cup OD, was 42.90
    assert_report("Pole sweep!X6", &at("Pole sweep!X6", e4), 0.954, 0.0005); // pull-out, was 0.958
    assert_eq!(at("Pole sweep!AA7", e4), at("Pole sweep!AA7", Deviations::NONE), "the 8-pole status does not change");
    for cell in ["Pole sweep!C6", "Pole sweep!C7", "Pole sweep!AA6", "Pole sweep!G6", "Pole sweep!X6"] {
        assert_workbook(cell);
    }
}
```
Run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations e4_`: expected FAIL (C6 = 9.2).

- [ ] **Step 2: Implement.** `SweepContext` gains `pub bond_inner: f64` (doc: "Inner magnet bondline [mm] (Metal design C120), read only by correction E4"); `api.rs` fills `bond_inner: md.bond_inner_mm`, and the `ctx()` fixture of the Task 11 unit test in `sweeps.rs` gains `bond_inner: 0.05`. In `pole_sweep` (rename `_dev`):

```rust
            // E4: the 2.5 mm keyed-bore wall is steel; the inner bondline sits on top of it.
            let keyed_wall = if dev.is_on(DeviationId::E4) {
                bore_mm / 2.0 + keyway_depth_mm + 2.5 + ctx.bond_inner
            } else {
                bore_mm / 2.0 + keyway_depth_mm + 2.5
            };
            let a_i = py_max(ctx.w_i / (2.0 * (PI / n as f64).tan()) + 0.05, keyed_wall);
```
E4 entry: `status: Applied`, `changes_file: Some("tests/data/deviations/E4.json")`, `corrected_formula` unchanged.

- [ ] **Step 3: Bless, review, run**

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations each_deviation_alone_changes_exactly_its_registered_cells
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: `E4.json` lists **47** cells, all in Pole sweep rows 6 and 7 (columns C, E-Z, and AA6); every test ok.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task17.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task17.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E4: `Pole sweep!C6:C11` | `a_i = MAX(w/(2 tan(pi/N)) + 0.05, bore/2 + key + 2.5)` | `+ inner bondline in the wall term; the 6-pole row reads "outside OD envelope"` | `E4`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E4 Applied, E5-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E4 pole-sweep hub wall excludes the bondline

The pole sweep's keyed-bore wall term adds the inner bondline, as the
Calculator's hub wall does. The 6- and 8-pole apothems become 9.25 mm and
the 6-pole row reads "outside OD envelope" (golden file, 47 cells).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 18: E5, rear-web eddy loss uses the field at the steel surface

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/temperature.rs` (`SlipLossInputs::web_integral_T2m2` default and help)
- Modify: `magcoupling-rs/src/engine/deviations.rs` (E5 entry)
- Modify: `magcoupling-rs/tests/deviations.rs`
- Create (blessed): `magcoupling-rs/tests/data/deviations/E5.json`
- Regenerate: `tests/data/input_schema.json` (help text)
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers, `workbook_input_defaults`, `workbook_help`; golden-file support (Task 16).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e5_web_eddy_loss_matches_the_report() {
    let e5 = Deviations::only(DeviationId::E5);
    assert_eq!(at("Temperature design!C121", e5), Value::Num(4.14e-5));
    assert_report("Temperature design!C125", &at("Temperature design!C125", e5), 0.365, 0.0005); // web loss, was 0.091 W
    assert_report("Temperature design!C130", &at("Temperature design!C130", e5), 2.751, 0.0005); // total, was 2.477 W
    assert_report("Temperature design!C131", &at("Temperature design!C131", e5), 0.0131, 0.00005); // drag, was 0.0118
    assert_report("Temperature design!C148", &at("Temperature design!C148", e5), 74.17, 0.005); // was 73.26 C
    assert_report("Temperature design!C149", &at("Temperature design!C149", e5), 92.51, 0.005); // was 89.77 C
    // No text changes: the high case stays 0.04 C under the 92.55 C limit.
    assert_eq!(at("Temperature design!C19", e5), Value::Text("never: steady state stays below the limit".into()));
    for cell in ["Temperature design!C121", "Temperature design!C125", "Temperature design!C130", "Temperature design!C149"] {
        assert_workbook(cell);
    }
}
```
Run `--test deviations e5_`: expected FAIL (C121 = 1.035e-5).

- [ ] **Step 2: Implement.** `temperature.rs`: `web_integral_T2m2: f64 = 4.14e-5 => param("T²·m²", "Rear-web end field, ∫B² dA", "3D, doubled at the steel surface (correction E5: the workbook's 1.035e-5 T²·m² is the free-space field, which made the web loss 4 times too low).", "Temperature design!C121")` with its existing range and `.log()`. E5 entry: `status: Applied`; `corrected_formula` adds "M2 applies the default; the factor 4 inside fields3d.run is M3 scope."; `workbook_input_defaults: &[("temperature.slip_loss.web_integral_T2m2", Literal::Num(1.035e-5))]`; `workbook_help: &[("temperature.slip_loss.web_integral_T2m2", "3D.")]`; `changes_file: Some("tests/data/deviations/E5.json")`. Extend `api::tests::workbook_defaults_put_back_every_corrected_default` with the web integral (1.035e-5 vs 4.14e-5). Bless the schema: `MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test schema`.

- [ ] **Step 3: Bless, review, run**

```bash
MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations each_deviation_alone_changes_exactly_its_registered_cells
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py --check
```
Expected: `E5.json` lists **37** cells, all on Temperature design: C17, C18, C20, C22, C121, C125, C130-C132, C134, C145-C149, C154-C157, C161, C169-C173, C175-C177, C180-C182, C186, C189, C190, C192, C193, C196 (no text cell changes); all tests ok; data current.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task18.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task18.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E5: `Temperature design!C121 (feeds C125, C130 and the thermal rows)` | `1.035e-5 T²·m² (free-space field)` | `4.14e-5 T²·m² (doubled at the steel surface); total slip loss 2.751 W; fields3d part is M3` | `E5`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E5 Applied, E6-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E5 rear-web eddy loss at the steel surface

The rear-web field integral defaults to 4.14e-5 T^2 m^2 (the field at the
steel surface, like the hub and cup inputs). Web loss 0.365 W, total slip
loss 2.751 W, high-case steady temperature 92.51 C (golden file, 37 cells).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 19: E6, cup wall at the flats excludes the outer bondline

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/model.rs`, `src/engine/deviations.rs`, `tests/deviations.rs`, `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e6_cup_wall_at_the_flats_matches_the_report() {
    let e6 = Deviations::only(DeviationId::E6);
    assert_report("Calculator!C63", &at("Calculator!C63", e6), 2.723, 0.0005); // was 2.773
    assert_workbook("Calculator!C63");
}
```
Run `--test deviations e6_`: expected FAIL.

- [ ] **Step 2: Implement.** In `model::compute`:

```rust
    // E6: the pocket flat sits at the block back plus the outer bondline (as C61, C62 place it).
    let wall_f = if dev.is_on(DeviationId::E6) { OD / 2.0 - (A_back + bond_outer_mm) } else { OD / 2.0 - A_back };
```
E6 entry: `status: Applied`, `changes_at_defaults: &[CellChange { cell: "Calculator!C63", workbook: Literal::Num(2.773232302834515), corrected: Literal::Num(2.7232323028345142) }]`.

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task19.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task19.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E6: `Calculator!C63` | `OD/2 − block-back apothem` | `OD/2 − (block back + outer bondline): 2.723 mm` | `E6`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E6 Applied, E7-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E6 cup wall at the flats excludes the bondline

Calculator!C63 subtracts the outer bondline, like C61, C62 and the hub
counterpart C38: 2.723 mm instead of 2.773 mm. Display only.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 20: E7, pull-out at the true maximum over angle

First line: depends on D7 (the closed form holds for the harmonic set [1, 3, 5] only). If D7 is the alternative, stop and escalate.

**Model:** session model (`risk: physics`, new math: the closed-form peak), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/model.rs` (`peak_off_half_pitch`; E7 in `compute`)
- Modify: `magcoupling-rs/src/engine/sweeps.rs`, `src/engine/calibration.rs` (E7)
- Modify: `magcoupling-rs/src/engine/deviations.rs` (`Probe`, field `probes`; E7 entry)
- Modify: `magcoupling-rs/tests/deviations.rs`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers (`cell_values_for`, `num`, `assert_report`); `Harmonic`, `Harmonic::s`, `shear_stress` (Task 3).
- Produces:
  - `pub fn peak_off_half_pitch(a: [f64; 3]) -> Option<f64>` in `model.rs`.
  - `pub struct Probe { pub label: &'static str, pub inputs: &'static [(&'static str, Literal)], pub expect: &'static [CellChange] }` and `Deviation.probes: &'static [Probe]`.
  - `tests/deviations.rs`: `fn at_probe(cell: &str, probe: &Probe, dev: Deviations) -> Value`, tests `each_probe_shows_its_correction`, `every_applied_engine_correction_is_visible_somewhere`.

- [ ] **Step 1: Write the failing tests.** `model.rs` unit tests:

```rust
    #[test]
    fn half_pitch_stays_the_peak_when_the_third_harmonic_is_small() {
        assert_eq!(peak_off_half_pitch([1.0, -0.011, 0.001]), None); // default design: A3/A1 = -0.011
        assert_eq!(peak_off_half_pitch([1.0, 0.0, 0.0]), None);
    }

    #[test]
    fn a_large_third_harmonic_moves_the_peak_and_raises_it() {
        let a = [1.0, 0.2, 0.0]; // A1 < 9 A3: half a pitch is a local minimum
        let x = peak_off_half_pitch(a).expect("the peak moves");
        let t = |x: f64| a[0] * x.sin() + a[1] * (3.0 * x).sin() + a[2] * (5.0 * x).sin();
        assert!(x > 0.0 && x < std::f64::consts::FRAC_PI_2);
        assert!(t(x) > t(std::f64::consts::FRAC_PI_2));
        // stationary: dT/dx = 0 at the returned angle
        let slope = a[0] * x.cos() + 3.0 * a[1] * (3.0 * x).cos() + 5.0 * a[2] * (5.0 * x).cos();
        assert!(slope.abs() < 1e-9, "{slope}");
    }
```
`tests/deviations.rs`:

```rust
/// The value at one cell for a probe's inputs, applied to the defaults `dev` implies.
fn at_probe(cell: &str, probe: &Probe, dev: Deviations) -> Value {
    let mut inputs = DesignInputs::defaults_with(dev);
    for &(path, value) in probe.inputs {
        inputs.set(path, value.to_value()).unwrap_or_else(|e| panic!("probe {:?}: {e}", probe.label));
    }
    cell_values_for(&inputs, dev).remove(cell).unwrap_or_else(|| panic!("{cell} is not a cell of the port"))
}

#[test]
fn each_probe_shows_its_correction() {
    let mut failures = Vec::new();
    for d in REGISTRY.iter().filter(|d| d.status == DeviationStatus::Applied) {
        for probe in d.probes {
            for change in probe.expect {
                let workbook = at_probe(change.cell, probe, Deviations::NONE);
                if !parity_close(&workbook, &change.workbook.to_value()) {
                    failures.push(format!("{} {:?}: {} workbook {workbook:?}, registered {:?}", d.id, probe.label, change.cell, change.workbook));
                }
                let corrected = at_probe(change.cell, probe, Deviations::only(d.id));
                if !parity_close(&corrected, &change.corrected.to_value()) {
                    failures.push(format!("{} {:?}: {} corrected {corrected:?}, registered {:?}", d.id, probe.label, change.cell, change.corrected));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", report(&failures));
}

#[test]
fn every_applied_engine_correction_is_visible_somewhere() {
    // A correction that changes nothing at defaults must carry a probe, so no applied correction goes untested.
    for d in REGISTRY.iter().filter(|d| d.status == DeviationStatus::Applied && d.class == DeviationClass::Engine) {
        let shows_at_defaults = !d.changes_at_defaults.is_empty() || d.changes_file.is_some();
        assert!(shows_at_defaults || !d.probes.is_empty(), "{} changes nothing at defaults and has no probe", d.id);
    }
}

#[test]
fn e7_pull_out_over_angle_matches_the_report() {
    let e7 = &REGISTRY[DeviationId::E7.index()];
    assert_eq!(e7.status, DeviationStatus::Applied);
    let six_poles = &e7.probes[0];
    let got = at_probe("Calculator!C93", six_poles, Deviations::only(DeviationId::E7));
    assert_report("Calculator!C93", &got, 0.911, 0.0005); // 6 poles, steel: was 0.861 N m
    let workbook = at_probe("Calculator!C93", six_poles, Deviations::NONE);
    assert_report("Calculator!C93", &workbook, 0.861, 0.0005);
}
```

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations e7_ 2>&1 | grep -E "error|panicked|test result" | head -3
```
Expected: compile error (`Probe` not found).

- [ ] **Step 2: Implement.**
  - `deviations.rs`:

```rust
/// Inputs on which a correction that is neutral at defaults shows (the report's
/// off-default example), and the cells it changes there.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Probe {
    /// What the inputs represent (the report's example).
    pub label: &'static str,
    /// Input overrides by path, applied to the defaults.
    pub inputs: &'static [(&'static str, Literal)],
    /// Cells with their workbook value and corrected value for these inputs.
    pub expect: &'static [CellChange],
}
```
    `Deviation` gains `pub probes: &'static [Probe],` (doc: "Off-default checks for corrections neutral at defaults (E7 to E13)"), `probes: &[]` on every entry and `FAKE`; `planned_entries_change_nothing_yet` asserts `d.probes.is_empty()`. E7 entry: `status: Applied`; `probes`:

```rust
        probes: &[
            Probe {
                label: "6 poles on the default hub, steel circuit (report: 0.861 -> 0.911 N m)",
                inputs: &[("coupling.npole", Literal::Int(6))],
                expect: &[
                    CellChange { cell: "Calculator!C89", workbook: Literal::Num(69917.71639509343), corrected: Literal::Num(73953.7345241203) },
                    CellChange { cell: "Calculator!C93", workbook: Literal::Num(0.8611577121363674), corrected: Literal::Num(0.9108682621562417) },
                    CellChange { cell: "Gap sweep!X6", workbook: Literal::Num(0.94859431918547), corrected: Literal::Num(1.038687728764382) },
                ],
            },
            Probe {
                label: "12-magnet no-iron prototype (calibration)",
                inputs: &[("calibration.total_magnets", Literal::Int(12))],
                expect: &[CellChange { cell: "Calibration!C44", workbook: Literal::Num(0.3518120782396669), corrected: Literal::Num(0.4420296708028627) }],
            },
        ],
```
  - `model.rs`:

```rust
/// E7: the electrical angle of the true pull-out, when half a pole pitch is not it.
///
/// Torque against electrical angle x is T(x) = a1 sin x + a3 sin 3x + a5 sin 5x,
/// `a` the per-harmonic amplitudes. Half a pitch (x = π/2) is always a stationary
/// point, and the workbook evaluates every harmonic there. The other stationary
/// points solve dT/dx = 0; with c = cos x,
/// dT/dx = c·[(a1 − 9 a3 + 25 a5) + (12 a3 − 100 a5) c² + 80 a5 c⁴],
/// a quadratic in u = c². T is symmetric about π/2 (odd harmonics), so
/// x in [0, π/2] suffices. Returns `None` when half a pitch is the maximum, so
/// callers keep the workbook expression and default outputs stay bit-identical;
/// otherwise the angle whose torque exceeds the half-pitch torque by more than
/// 1e-12 relative. Valid for the harmonic set [1, 3, 5] only (`HARMONICS`); the
/// Addendum A plan generalizes it with the harmonic parameter (decision D7).
pub fn peak_off_half_pitch(a: [f64; 3]) -> Option<f64> {
    let [a1, a3, a5] = a;
    let torque = |x: f64| a1 * x.sin() + a3 * (3.0 * x).sin() + a5 * (5.0 * x).sin();
    let half_pitch = a1 - a3 + a5; // sin(π/2) = 1, sin(3π/2) = −1, sin(5π/2) = 1
    let (qa, qb, qc) = (80.0 * a5, 12.0 * a3 - 100.0 * a5, a1 - 9.0 * a3 + 25.0 * a5);
    let roots: Vec<f64> = if qa != 0.0 {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < 0.0 {
            Vec::new()
        } else {
            vec![(-qb + disc.sqrt()) / (2.0 * qa), (-qb - disc.sqrt()) / (2.0 * qa)]
        }
    } else if qb != 0.0 {
        vec![-qc / qb]
    } else {
        Vec::new()
    };
    roots
        .into_iter()
        .filter(|u| (0.0..=1.0).contains(u))
        .map(|u| u.sqrt().acos())
        .map(|x| (x, torque(x)))
        .filter(|&(_, t)| t - half_pitch > 1e-12 * half_pitch.abs())
        .max_by(|p, q| p.1.total_cmp(&q.1))
        .map(|(x, _)| x)
}
```
    In `model::compute`, after `let h = shear_stress(...)` (keep `h` for the per-circuit sums):

```rust
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum.
    let amplitude = |x: &Harmonic| x.bi * x.bo / (2.0 * ci.mu0) * x.s(ci.backiron);
    let mut h_pull = h;
    if dev.is_on(DeviationId::E7)
        && let Some(x) = peak_off_half_pitch(h.map(|hn| amplitude(&hn)))
    {
        for hn in h_pull.iter_mut() {
            hn.tau = amplitude(hn) * (f64::from(hn.n) * x).sin();
        }
    }
```
    and use `h_pull` for `tau` and the `tau1_Pa`/`tau3_Pa`/`tau5_Pa` fields (`let [h1, h3, h5] = h_pull;`). (A let chain, not `if dev.is_on(..) { if let .. }`: clippy's `collapsible_if` rejects the nested form under `-D warnings` on this toolchain.) The two circuit sums become:

```rust
    let circuit = |s: fn(&Harmonic) -> f64| {
        let coefficients = h.map(|x| x.bi * x.bo * s(&x));
        let peak = if dev.is_on(DeviationId::E7) { peak_off_half_pitch(coefficients) } else { None };
        match peak {
            Some(x) => h.iter().zip(coefficients).fold(0.0, |acc, (hn, c)| acc + c * (f64::from(hn.n) * x).sin()),
            None => h.iter().fold(0.0, |acc, x| acc + x.bi * x.bo * s(x) * (f64::from(x.n) * PI / 2.0).sin()),
        }
    };
```
  - `sweeps.rs::row` (rename `_dev` to `dev`). Imports become `use super::deviations::{DeviationId, Deviations};` and `use super::model::{Harmonic, peak_off_half_pitch, shear_stress};`. After `let h = shear_stress(...)`, replace the lines from `let s = ...` through `let U = ...` with:

```rust
    let s = h.map(|x| x.s(ctx.backiron));
    // E7: every harmonic at the true pull-out angle when half a pitch is not the maximum (as the model).
    let amplitude = |x: &Harmonic| x.bi * x.bo / (2.0 * ctx.mu0) * x.s(ctx.backiron);
    let mut h_pull = h;
    if dev.is_on(DeviationId::E7)
        && let Some(x) = peak_off_half_pitch(h.map(|hn| amplitude(&hn)))
    {
        for hn in h_pull.iter_mut() {
            hn.tau = amplitude(hn) * (f64::from(hn.n) * x).sin();
        }
    }
    let U = h_pull.iter().fold(0.0, |acc, x| acc + x.tau);
```
    and change `let [h1, h3, h5] = h;` to `let [h1, h3, h5] = h_pull;` (the `s1`/`s3`/`s5` columns keep `s`, which E7 does not change). Checked on a scratch build of this code, with `only(E7)`: at `coupling.npole = 6`, C89 = 73953.7345241203, C93 = 0.9108682621562417 and Gap sweep!X6 = 1.038687728764382; at `calibration.total_magnets = 12`, Calibration!C44 = 0.44202967080286276 (the registry probe values below); and at the defaults E7 leaves the model, both sweeps and the calibration bit-identical.
  - `calibration.rs::compute`: split the `tau_n` closure into `amp_n` (everything before the final `* (n * PI / 2.0).sin()`) and

```rust
    let amps = [amp_n(1), amp_n(3), amp_n(5)];
    let peak = if dev.is_on(DeviationId::E7) { peak_off_half_pitch(amps) } else { None };
    let tau_at = |a: f64, n: f64| match peak {
        Some(x) => a * (n * x).sin(),
        None => a * (n * PI / 2.0).sin(), // the workbook expression, bit for bit
    };
    let (t1, t3, t5) = (tau_at(amps[0], 1.0), tau_at(amps[1], 3.0), tau_at(amps[2], 5.0));
```
    (rename `_dev`; import `super::model::peak_off_half_pitch`).

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. `each_deviation_alone_changes_exactly_its_registered_cells` confirms E7 changes no cell at defaults (every default row peaks at half a pitch); `each_probe_shows_its_correction` checks the four probe cells both ways; parity and differential unchanged.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task20.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task20.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E7: `Calculator!C76, C82, C88 → C89-C96; sweeps N, Q, T, U; Calibration!C40-C42` | `every harmonic at half a pole pitch` | `the maximum of the harmonic torque-angle curve (closed form); same at defaults; 6 poles 0.861 → 0.911 N·m` | `E7`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E7 Applied, E8-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E7 pull-out at the maximum over angle

Pull-out is the maximum of the harmonic torque-angle curve, found from the
roots of dT/dx (a quadratic in cos^2 x), in both circuits, the sweeps and
the calibration. Unchanged at defaults; 6 poles on the default hub rise
from 0.861 to 0.911 N m. Registry probes pin the off-default values.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---
### Task 21: E8, arc mode uses arc geometry

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/model.rs` (C9 in `compute`, C111 in `mass_estimate`)
- Modify: `magcoupling-rs/src/engine/metal_design.rs` (C175 in `retainers`: new parameter)
- Modify: `magcoupling-rs/src/engine/api.rs`, `src/engine/deviations.rs`, `tests/deviations.rs`, `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; `Probe`, `at_probe` (Task 20).
- Produces (changed signature): `pub fn retainers(md: &MetalDesignInputs, inner_back_apothem_mm: f64, inner_thickness_mm: f64, inner_width_mm: f64, outer_face_apothem_mm: f64, bore_mm: f64, inner_corner_radius_mm: f64, dev: Deviations) -> RetainerResults`.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e8_arc_mode_matches_the_report() {
    let e8 = &REGISTRY[DeviationId::E8.index()];
    assert_eq!(e8.status, DeviationStatus::Applied);
    let arcs = &e8.probes[0];
    let (workbook, corrected) = (Deviations::NONE, Deviations::only(DeviationId::E8));
    assert_report("Calculator!C93", &at_probe("Calculator!C93", arcs, workbook), 2.99, 0.005);
    assert_report("Calculator!C93", &at_probe("Calculator!C93", arcs, corrected), 2.65, 0.005);
    assert_report("Metal design!C9", &at_probe("Metal design!C9", arcs, corrected), 2.25, 0.005); // hot low, was 2.54
    assert_report("Metal design!C35", &at_probe("Metal design!C35", arcs, corrected), 0.270, 0.0005); // was -0.476
    assert_report("Metal design!C175", &at_probe("Metal design!C175", arcs, corrected), 26.69, 0.005); // sleeve ID, was 27.44
    assert_report("Calculator!C111", &at_probe("Calculator!C111", arcs, corrected), 48.41, 0.005); // cup mass, was 42.96
    assert_eq!(at_probe("Shaft clamps!C48", arcs, corrected), Value::Text("ISO 4762 M4 x 12, class 12.9".into()));
}
```
Run `cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations e8_`: expected FAIL (`E8` is Planned).

- [ ] **Step 2: Implement.**
  - `model::compute`: move the `r_face_i` and `r_corner_i` lines above `corner_gap` (they do not depend on it) and:

```rust
    // Corner gap from the flat-face gap. Workbook: always the flat-block corner
    // geometry. E8: the inner corner radius C55, which is the face radius for arcs.
    let corner_gap = if dev.is_on(DeviationId::E8) {
        face_gap_mm - (r_corner_i - r_face_i)
    } else {
        face_gap_mm
            - (((a_i + mi.thickness_mm).powi(2) + (mi.width_mm / 2.0).powi(2)).sqrt() - (a_i + mi.thickness_mm))
    };
```
    (bit-identical for flat blocks: `r_corner_i` is the same expression).
  - `model::mass_estimate` (rename `_dev`):

```rust
    let pocket = r.outer_back_apothem_mm + bond_outer_mm;
    // E8: arcs sit in a round pocket, not a polygon.
    let cavity = if dev.is_on(DeviationId::E8) && ci.faceted != 1 {
        PI * pocket.powi(2)
    } else {
        N * pocket.powi(2) * (PI / N).tan()
    };
    let m_ring = ((PI * (r.cup_od_mm / 2.0).powi(2) - cavity) * cup_depth_mm
        + PI * ((r.cup_od_mm / 2.0).powi(2) - (ci.bore_mm / 2.0).powi(2)) * web_mm)
        * steel_density_g_mm3;
```
  - `metal_design::retainers`: add `inner_corner_radius_mm: f64` before `dev` and

```rust
    // E8: the sleeve clears the inner corner radius C55 (the face radius for arcs).
    let sleeve_id = if dev.is_on(DeviationId::E8) {
        2.0 * (inner_corner_radius_mm + md.sleeve_bedding_mm)
    } else {
        2.0 * (((inner_back_apothem_mm + inner_thickness_mm).powi(2) + (inner_width_mm / 2.0).powi(2)).sqrt()
            + md.sleeve_bedding_mm)
    };
```
    `api.rs` passes `m.inner_corner_radius_mm`.
  - E8 entry: `status: Applied`, `probes`:

```rust
        probes: &[Probe {
            label: "arc magnets (coupling.faceted = 0), the report's arc-mode rerun",
            inputs: &[("coupling.faceted", Literal::Int(0))],
            expect: &[
                CellChange { cell: "Calculator!C9", workbook: Literal::Num(1.0268256054339253), corrected: Literal::Num(1.4) },
                CellChange { cell: "Calculator!C93", workbook: Literal::Num(2.9915853633580882), corrected: Literal::Num(2.6472742027214955) },
                CellChange { cell: "Calculator!C111", workbook: Literal::Num(42.955452974403656), corrected: Literal::Num(48.40907404648711) },
                CellChange { cell: "Metal design!C11", workbook: Literal::Text("Estimate covers hot min"), corrected: Literal::Text("Below hot minimum") },
                CellChange { cell: "Metal design!C35", workbook: Literal::Num(-0.4763487891321485), corrected: Literal::Num(0.2700000000000007) },
                CellChange { cell: "Metal design!C37", workbook: Literal::Text("Below target"), corrected: Literal::Text("Meets assumed target") },
                CellChange { cell: "Metal design!C175", workbook: Literal::Num(27.43634878913215), corrected: Literal::Num(26.69) },
                CellChange {
                    cell: "Shaft clamps!C48",
                    workbook: Literal::Text("None: enlarge the boss or the clamp length"),
                    corrected: Literal::Text("ISO 4762 M4 x 12, class 12.9"),
                },
            ],
        }],
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok; E8 changes no cell at defaults (flat blocks); the probe matches both ways.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task21.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task21.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E8: `Calculator!C9, C57, C111; Metal design!C175` | `flat-block corner geometry in arc mode` | `the corner radius C55 (face radius for arcs), round pocket; arc-mode pull-out 2.99 → 2.65 N·m` | `E8`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E8 Applied, E9-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E8 arc mode uses arc geometry

With arc magnets the corner gap, the sleeve bore and the cup cavity use
the arc geometry (corner radius C55 = face radius, round pocket). Flat
blocks are bit-identical; the arc-mode pull-out falls from 2.99 to 2.65 N m
and the clearance becomes +0.270 mm (registry probe).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 22: E9, "no back iron" also applies to the cup

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/model.rs` (`mass_estimate`), `src/engine/materials.rs` (`compute` gains `backiron`), `src/engine/api.rs`, `src/engine/deviations.rs`, `tests/deviations.rs`, `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; `Probe`, `at_probe`.
- Produces (changed signature): `pub fn compute(mat: &MaterialsInputs, t_bi_req_mm: f64, wall_corner_mm: f64, backiron: i64, dev: Deviations) -> MaterialsResults` in `materials.rs`.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e9_no_back_iron_matches_the_report() {
    let e9 = &REGISTRY[DeviationId::E9.index()];
    assert_eq!(e9.status, DeviationStatus::Applied);
    let no_iron = &e9.probes[0];
    let corrected = Deviations::only(DeviationId::E9);
    assert_report("Calculator!C111", &at_probe("Calculator!C111", no_iron, corrected), 20.90, 0.005); // cup, was 60.75 g
    assert_report("Calculator!C113", &at_probe("Calculator!C113", no_iron, corrected), 10.59, 0.005); // boss, was 30.78 g
    assert_report("Calculator!C114", &at_probe("Calculator!C114", no_iron, corrected), 96.8, 0.05); // total, was 156.9 g
    assert_report("Temperature design!C141", &at_probe("Temperature design!C141", no_iron, corrected), 45.8, 0.05); // was 74.2 J/K
    assert_eq!(at_probe("Materials!C22", no_iron, corrected), Value::Text("No back iron".into()));
}
```
Run `--test deviations e9_`: expected FAIL.

- [ ] **Step 2: Implement.**
  - `model::mass_estimate`:

```rust
    // E9: with no intentional back iron the cup and boss are aluminium, as the hub already is.
    let cup_boss_density = if dev.is_on(DeviationId::E9) && ci.backiron != 1 { al_density_g_mm3 } else { steel_density_g_mm3 };
```
    used in place of `steel_density_g_mm3` in `m_ring` and `m_boss` (the heat capacity follows through the links).
  - `materials::compute(mat, t_bi_req_mm, wall_corner_mm, backiron, dev)`:

```rust
    let check = if dev.is_on(DeviationId::E9) && backiron == 0 {
        "No back iron".to_owned() // as the Calculator's C105/C106 read
    } else if wall_corner_mm >= t_bi_req_mm {
        "OK".to_owned()
    } else {
        format!("Too thin: raise Metal design C122 to at least {} mm", fmt_fixed(ceiling(t_bi_req_mm, 0.1), 1))
    };
```
    `api.rs` passes `ci.backiron`; the Task 7 unit tests pass `1` for `backiron`.
  - E9 entry: `status: Applied`, `probes`:

```rust
        probes: &[Probe {
            label: "no intentional back iron (coupling.backiron = 0)",
            inputs: &[("coupling.backiron", Literal::Int(0))],
            expect: &[
                CellChange { cell: "Calculator!C111", workbook: Literal::Num(60.753684123800234), corrected: Literal::Num(20.896171609459955) },
                CellChange { cell: "Calculator!C113", workbook: Literal::Num(30.777554908688483), corrected: Literal::Num(10.585910605536167) },
                CellChange { cell: "Calculator!C114", workbook: Literal::Num(156.86088643049283), corrected: Literal::Num(96.81172961300027) },
                CellChange { cell: "Temperature design!C141", workbook: Literal::Num(74.18527010448489), corrected: Literal::Num(45.78201892981087) },
                CellChange {
                    cell: "Materials!C22",
                    workbook: Literal::Text("Too thin: raise Metal design C122 to at least 2.0 mm"),
                    corrected: Literal::Text("No back iron"),
                },
            ],
        }],
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok; no change at defaults (steel circuit).

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task22.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task22.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E9: `Calculator!C111, C113; Materials!C22` | `steel cup and boss, back-iron wall advice even with no back iron` | `aluminium cup and boss, "No back iron"; total mass 156.9 → 96.8 g at backiron = 0` | `E9`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E9 Applied, E10-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E9 no back iron also applies to the cup

With backiron = 0 the cup and boss take the aluminium density like the
hub, the heat capacity follows, and Materials!C22 reads "No back iron".
Unchanged at defaults (registry probe).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 23: E10, gap flux density sums each magnet's MMF

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/model.rs`, `src/engine/deviations.rs`, `tests/deviations.rs`, `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; `Probe`, `at_probe`.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e10_gap_flux_density_matches_the_report() {
    let e10 = &REGISTRY[DeviationId::E10.index()];
    assert_eq!(e10.status, DeviationStatus::Applied);
    let mixed = &e10.probes[0];
    let corrected = Deviations::only(DeviationId::E10);
    assert_report("Calculator!C103", &at_probe("Calculator!C103", mixed, Deviations::NONE), 1.024, 0.0005);
    assert_report("Calculator!C103", &at_probe("Calculator!C103", mixed, corrected), 1.043, 0.0005);
    assert_report("Calculator!C104", &at_probe("Calculator!C104", mixed, corrected), 1.949, 0.0005); // was 1.915
}
```
Run `--test deviations e10_`: expected FAIL.

- [ ] **Step 2: Implement.** In `model::compute`:

```rust
    // E10: in the series circuit each magnet contributes its own MMF (Br·t), not the mean Br.
    let B_gap = if dev.is_on(DeviationId::E10) {
        (bri * mi.thickness_mm + bro * mo.thickness_mm) / (mi.thickness_mm + mo.thickness_mm + g_m)
    } else {
        (bri + bro) / 2.0 * (mi.thickness_mm + mo.thickness_mm) / (mi.thickness_mm + mo.thickness_mm + g_m)
    };
```
E10 entry: `status: Applied`, `probes`:

```rust
        probes: &[Probe {
            label: "N52 inner (3.17 mm) with a B861 outer (1.59 mm)",
            inputs: &[
                ("coupling.magnets.part_inner", Literal::Text("B842-N52")),
                ("coupling.magnets.part_outer", Literal::Text("B861")),
            ],
            expect: &[
                CellChange { cell: "Calculator!C103", workbook: Literal::Num(1.0242499999999999), corrected: Literal::Num(1.0427944805194804) },
                CellChange { cell: "Calculator!C104", workbook: Literal::Num(1.9146646666666662), corrected: Literal::Num(1.9493304822510817) },
                CellChange { cell: "Materials!C20", workbook: Literal::Num(1.9146646666666662), corrected: Literal::Num(1.9493304822510817) },
            ],
        }],
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok. At defaults the rings are identical, so C103 may differ from the workbook form by an ULP but not by the parity rule: `each_deviation_alone_changes_exactly_its_registered_cells` sees no change.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task23.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task23.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E10: `Calculator!C103 → C104-C106; Materials!C20-C22` | `(Br_i + Br_o)/2 · (t_i + t_o)/(t_i + t_o + g)` | `(Br_i t_i + Br_o t_o)/(t_i + t_o + g); same for identical rings` | `E10`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E10 Applied, E11-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E10 gap flux density sums each magnet's MMF

B_gap = (Br_i t_i + Br_o t_o) / (t_i + t_o + g), the ideal-iron series
circuit. Identical rings are unchanged; an N52/B861 pair rises from 1.024
to 1.043 T (registry probe).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 24: E11, 22 °C fatigue screen uses the endurance input

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/temperature.rs`, `src/engine/deviations.rs`, `tests/deviations.rs`, `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; `Probe`, `at_probe`.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e11_fatigue_screen_matches_the_report() {
    let e11 = &REGISTRY[DeviationId::E11.index()];
    assert_eq!(e11.status, DeviationStatus::Applied);
    let low = &e11.probes[0];
    assert_eq!(at_probe("Temperature design!C91", low, Deviations::NONE), Value::Text("OK: 8x margin".into()));
    assert_eq!(at_probe("Temperature design!C91", low, Deviations::only(DeviationId::E11)), Value::Text("CHECK".into())); // margin 3.77
}
```
Run `--test deviations e11_`: expected FAIL.

- [ ] **Step 2: Implement.** In `temperature::compute` (rename `_dev`):

```rust
    // E11: the 22 °C screen reads the fatigue-endurance input (C195) like C196, C197 and C202.
    let endurance = if dev.is_on(DeviationId::E11) { ti.adhesive_life.fatigue_endurance } else { 0.2 };
    let fat = endurance * sel.lap_shear_MPa / tau_b;
```
(bit-identical at the default 0.2). E11 entry: `status: Applied`, `probes: &[Probe { label: "fatigue endurance 0.1 (report: margin 3.77, flips below 0.106)", inputs: &[("temperature.adhesive_life.fatigue_endurance", Literal::Num(0.1))], expect: &[CellChange { cell: "Temperature design!C91", workbook: Literal::Text("OK: 8x margin"), corrected: Literal::Text("CHECK") }] }]`.

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task24.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task24.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E11: `Temperature design!C91` | `0.2 typed into the formula` | `the fatigue-endurance input C195` | `E11`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E11 Applied, E12-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E11 fatigue screen uses the endurance input

The 22 C fatigue screen (C91) reads the fatigue-endurance input instead of
a typed-in 0.2. Unchanged at defaults; at 0.1 it reads CHECK (probe).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 25: E12, a hot-day start above the limit

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/temperature.rs`, `src/engine/deviations.rs`, `tests/deviations.rs`, `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; `Probe`, `at_probe`.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test:**

```rust
#[test]
fn e12_start_above_the_limit_matches_the_report() {
    let e12 = &REGISTRY[DeviationId::E12.index()];
    assert_eq!(e12.status, DeviationStatus::Applied);
    let hot = &e12.probes[0];
    let (workbook, corrected) = (Deviations::NONE, Deviations::only(DeviationId::E12));
    assert_report("Temperature design!C150", &at_probe("Temperature design!C150", hot, workbook), -25.8, 0.05);
    assert_report("Temperature design!C152", &at_probe("Temperature design!C152", hot, workbook), -71.2, 0.05);
    assert_report("Temperature design!C151", &at_probe("Temperature design!C151", hot, workbook), -861.0, 0.5);
    assert_report("Temperature design!C153", &at_probe("Temperature design!C153", hot, workbook), -0.0035, 0.00005);
    for cell in ["Temperature design!C19", "Temperature design!C23", "Temperature design!C150", "Temperature design!C151",
                 "Temperature design!C152", "Temperature design!C153"] {
        assert_eq!(num(&at_probe(cell, hot, corrected)), 0.0, "{cell}");
    }
}
```
Run `--test deviations e12_`: expected FAIL.

- [ ] **Step 2: Implement.** In `temperature::compute`:

```rust
    // E12: a start already at or above the limit reaches it at once; tested before "never".
    let start_above = dev.is_on(DeviationId::E12) && T0 >= gov;
    let t_lim_h = if start_above {
        NumOrText::Num(0.0)
    } else if Th <= gov {
        NumOrText::Text(NEVER)
    } else {
        NumOrText::Num(-tau_th * (1.0 - (gov - T0) / rise_h).ln())
    };
    let t_lim_e = if start_above {
        NumOrText::Num(0.0)
    } else if Te <= gov {
        NumOrText::Text(NEVER)
    } else {
        NumOrText::Num(-tau_th * (1.0 - (gov - T0) / rise_e).ln())
    };
    // E12: slip never cools the magnets, so the critical drag is not negative.
    let critical_drag = if dev.is_on(DeviationId::E12) { py_max(0.0, gov - T0) * G / omega } else { (gov - T0) * G / omega };
```
and use `critical_drag` for both `ThermalResults::critical_drag_Nm` and `SummaryResults::critical_drag_Nm`. E12 entry: `status: Applied`, `probes`:

```rust
        probes: &[Probe {
            label: "driving rise 40 C: hot-day start 95 C above the 92.55 C limit",
            inputs: &[("temperature.duty.driving_rise_C", Literal::Num(40.0))],
            expect: &[
                CellChange { cell: "Temperature design!C19", workbook: Literal::Num(-25.83996179152416), corrected: Literal::Num(0.0) },
                CellChange { cell: "Temperature design!C23", workbook: Literal::Num(-0.003509295069490145), corrected: Literal::Num(0.0) },
                CellChange { cell: "Temperature design!C150", workbook: Literal::Num(-25.83996179152416), corrected: Literal::Num(0.0) },
                CellChange { cell: "Temperature design!C151", workbook: Literal::Num(-861.3320597174721), corrected: Literal::Num(0.0) },
                CellChange { cell: "Temperature design!C152", workbook: Literal::Num(-71.1887399460417), corrected: Literal::Num(0.0) },
                CellChange { cell: "Temperature design!C153", workbook: Literal::Num(-0.003509295069490145), corrected: Literal::Num(0.0) },
            ],
        }],
```

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task25.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task25.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E12: `Temperature design!C19, C23, C150-C153` | `negative times and drag when the start is above the limit` | `0 s, 0 rev, 0 N·m` | `E12`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E12 Applied, E13-E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E12 hot-day start above the limit

When the hot-day start is already at or above the governing limit, the
times and rotations to the limit are 0 (tested before "never") and the
critical drag is max(0, T_lim - T0) G / omega. Unchanged at defaults.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 26: E13, a measured drag of zero; all corrections together

**Model:** session model (`risk: physics`), omit `model` on purpose; the physics reviewer runs on the session model too.

**Files:**
- Modify: `magcoupling-rs/src/engine/temperature.rs`, `src/engine/deviations.rs` (`Literal::Error`; E13 entry)
- Modify: `magcoupling-rs/tests/deviations.rs`
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)

**Interfaces:**
- Consumes: Task 14 helpers; `Probe`, `at_probe`, `each_probe_shows_its_correction`; `headline` (Task 12).
- Produces: `Literal::Error(&'static str)`: a workbook error value (e.g. `"#DIV/0!"`) where Python raises; `each_probe_shows_its_correction` skips the workbook-side comparison for it.

- [ ] **Step 1: Write the failing tests:**

```rust
#[test]
fn e13_zero_drag_matches_the_report() {
    let e13 = &REGISTRY[DeviationId::E13.index()];
    assert_eq!(e13.status, DeviationStatus::Applied);
    let (zero, neg_zero) = (&e13.probes[0], &e13.probes[1]);
    for cell in ["Temperature design!C156", "Temperature design!C157"] {
        assert_eq!(at_probe(cell, zero, Deviations::only(DeviationId::E13)), Value::Num(f64::INFINITY), "{cell}");
        // +0.0 already gives +inf without the correction; -0.0 proves the guard does something:
        // without it p_use = ke = -0.0 and rev_s / -0.0 = -inf; the guard (-0.0 == 0.0) gives +inf.
        assert_eq!(at_probe(cell, neg_zero, Deviations::NONE), Value::Num(f64::NEG_INFINITY), "{cell}: E13 off");
        assert_eq!(at_probe(cell, neg_zero, Deviations::only(DeviationId::E13)), Value::Num(f64::INFINITY), "{cell}: E13 on");
    }
    // compute_all runs to the end (Python aborts every sheet): the rest is finite.
    let mut inputs = DesignInputs::default();
    inputs.metal.measured_drag_Nm = Some(0.0);
    let t = compute_all(&inputs).temperature;
    assert!(t.summary.governing_limit_C.is_finite() && t.thermal.heat_capacity_J_K.is_finite());
}

#[test]
fn all_corrections_together_give_the_reviewed_headline() {
    // What users see (compute_all, every correction on) at the default design, D1 = 1.30 T.
    // Values from one Python rerun with E1, E2, E3 and E5 patched in together (E4, E6-E13 are
    // neutral for these cells at defaults).
    let res = compute_all(&DesignInputs::default());
    let h: BTreeMap<&str, Value> = headline(&res).into_iter().collect();
    let close = |key: &str, want: f64| {
        let got = num(&h[key]);
        assert!((got - want).abs() <= 1e-9 * want.abs(), "{key}: {got} vs {want}");
    };
    close("pullout_at_op_temp_Nm", 2.6884762950539796);
    close("hot_low_with_variation_Nm", 2.2852048507958824);
    close("cold_high_with_variation_Nm", 3.823310370488638);
    close("governing_temp_limit_C", 93.05566428111358);
    close("running_clearance_mm", -0.10317439456607391);
    assert_eq!(h["hot_min_check"], Value::Text("Below hot minimum".into()));
    assert_eq!(h["clearance_check"], Value::Text("Below target".into()));
    assert_eq!(h["cup_wall_check"], Value::Text("Too thin: raise Metal design C122 to at least 2.0 mm".into()));
    assert_eq!(h["temperature_verdict"], Value::Text("OK on temperature. Confirm drag torque and thermal cycling by test.".into()));
    assert_eq!(h["clamp_screw"], Value::Text("ISO 4762 M4 x 14, class 12.9".into()));
    assert_eq!(res.temperature.mismatch.reading, "Below the lap-shear strength");
    assert_eq!(res.temperature.adhesive_life.daily_screen, "Below the fatigue endurance");
    assert_eq!(res.temperature.adhesive.fatigue_screen, "OK: 7x margin");
    assert!((res.temperature.thermal.steady_high_C - 92.51311161421395).abs() <= 1e-9 * 92.5);
    assert_eq!(res.temperature.thermal.time_to_limit_high, NumOrText::Text("never: steady state stays below the limit"));
    assert_eq!(
        res.clamps.length_note,
        "No 2 mm length step of M4 both engages 8 mm of thread and stays inside the boss; M4 x 14 protrudes 0.34 mm"
    );
}
```
For D1 = about 1.315 T use: pull-out 2.75087598894362, hot low 2.338244590602077, cold high 3.9120496304190624, limit 93.81547097429792, clearance −0.10317439456607391, clamp `"None: enlarge the boss or the clamp length"`, `length_note` `""`; the texts and the 92.51311161421395 steady temperature are the same.

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test deviations e13_ all_corrections 2>&1 | grep -E "panicked|test result"
```
Expected: `e13_zero_drag_matches_the_report` FAILS (E13 is Planned); `all_corrections_together_give_the_reviewed_headline` already passes (E1 to E12 are applied); keep it as the user-facing contract.

- [ ] **Step 2: Implement.**
  - `deviations.rs`: `Literal` gains `Error(&'static str)` (doc: "A workbook error value such as `#DIV/0!`, where the Python engine raises"); `to_value` maps it to `Value::Text`; extend the unit test `literal_converts_to_value`. In `tests/deviations.rs::each_probe_shows_its_correction`, compare the workbook side only when `!matches!(change.workbook, Literal::Error(_))` (still run it, which proves no panic).
  - `temperature::compute`:

```rust
    // E13: no heating power means no heating: rotations per degree are +inf
    // (Python divides by zero and aborts every sheet).
    let per_degree = |power: f64, rate: f64| {
        if dev.is_on(DeviationId::E13) && power == 0.0 { f64::INFINITY } else { rev_s / rate }
    };
```
    with `rev_per_C_est: per_degree(p_use, ke)`, `rev_per_C_high: per_degree(p_hi, kh)`.
  - E13 entry: `status: Applied`, `corrected_formula` adds "With the correction off, Rust arithmetic gives +inf for a drag of +0.0 but -inf for -0.0 (IEEE sign of zero); the guard makes both +inf. Negative drags are not rejected (optional part, not implemented).", `probes`:

```rust
        probes: &[
            Probe {
                label: "measured drag exactly 0 (Python raises ZeroDivisionError; the workbook shows #DIV/0!)",
                inputs: &[("metal.measured_drag_Nm", Literal::Num(0.0))],
                expect: &[
                    CellChange { cell: "Temperature design!C156", workbook: Literal::Error("#DIV/0!"), corrected: Literal::Num(f64::INFINITY) },
                    CellChange { cell: "Temperature design!C157", workbook: Literal::Error("#DIV/0!"), corrected: Literal::Num(f64::INFINITY) },
                ],
            },
            Probe {
                label: "measured drag typed as -0.0 (Python raises ZeroDivisionError too; without E13 Rust gives -inf)",
                inputs: &[("metal.measured_drag_Nm", Literal::Num(-0.0))],
                expect: &[
                    CellChange { cell: "Temperature design!C156", workbook: Literal::Error("#DIV/0!"), corrected: Literal::Num(f64::INFINITY) },
                    CellChange { cell: "Temperature design!C157", workbook: Literal::Error("#DIV/0!"), corrected: Literal::Num(f64::INFINITY) },
                ],
            },
        ],
```
(Checked in the oracle: `compute_all` raises `ZeroDivisionError` for both 0.0 and -0.0. `set()` accepts -0.0, a finite number. `parity_close` never equates -inf with +inf, so `each_probe_shows_its_correction` would catch a guard that let -0.0 through.)

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
```
Expected: all ok, including both new tests (the -0.0 probe gives -inf with E13 off and +inf with it on), `each_probe_shows_its_correction` (both E13 probes) and `zero_measured_drag_does_not_panic` (robustness).

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task26.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task26.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E13: `Temperature design!C156, C157` | `#DIV/0! (Python: the whole calculation stops)` | `+inf; the other results are computed. JSON export must encode inf (M4)` | `E13`. Add a README note: "Results can be +inf (E13); exporters must handle it." `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E13 Applied, E14 Planned`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
fix(magcoupling-rs): apply E13 zero measured drag gives +inf

With a bench drag of exactly 0 the rotations per degree (C156, C157) are
+inf and every other sheet still computes. A test pins the user-facing
headline with all approved corrections applied together.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 27: E14, the 22 mm boss statement (documentation)

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

README and help text only; no number changes.

**Files:**
- Modify: `magcoupling-rs/src/engine/clamps.rs` (help of `boss_od_mm`)
- Modify: `magcoupling-rs/src/engine/deviations.rs` (E14 entry)
- Modify: `magcoupling-rs/tests/deviations.rs`
- Regenerate: `tests/data/input_schema.json` (help text)
- Modify: `magcoupling-rs/README.md`, `docs/ai/03-structure.yaml` (deviations status)
- Modify: `reference/magcoupling-py/README.md` (line 195, the approved sentence; outside the read-only engine `magcoupling/`)

**Interfaces:**
- Consumes: `Deviation.workbook_help`, `reworded_help` (Task 14).
- Produces: nothing new.

- [ ] **Step 1: Write the failing test** (the reworded statement must be what the engine computes):

```rust
#[test]
fn e14_the_22mm_boss_statement_is_true() {
    let e14 = &REGISTRY[DeviationId::E14.index()];
    assert_eq!(e14.status, DeviationStatus::Applied);
    let clamp = |length: f64| {
        let mut inputs = DesignInputs::defaults_with(Deviations::NONE);
        inputs.clamps.boss_od_mm = 22.0;
        inputs.clamps.clamp_length_mm = length;
        compute_all_with(&inputs, Deviations::NONE).clamps
    };
    let c = clamp(14.0);
    assert_eq!(c.index, 0, "below 14.5 mm nothing fits");
    assert_eq!(c.table[2].works, 0, "M4 no longer fits at 22 mm");
    let c = clamp(14.5);
    assert!(c.recommended.starts_with("ISO 4762 M3 x "), "{}", c.recommended);
    assert_eq!(c.screws, NumOrText::Num(2.0), "two M3 need a 14.5 mm clamp");
    for length in [18.0, 25.0] {
        let c = clamp(length);
        assert!(c.recommended.starts_with("ISO 4762 M2.5 x "), "{length}: {}", c.recommended);
        assert_eq!(c.screws, NumOrText::Num(3.0), "from 18 mm up: three M2.5");
    }
}
```
Run `--test deviations e14_`: expected FAIL at the status assertion (Planned); the rest already holds.

- [ ] **Step 2: Implement.** `clamps.rs`: the `boss_od_mm` help becomes exactly `"At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3."`. E14 entry: `status: Applied`, `workbook_help: &[("clamps.boss_od_mm", "At 22 mm only M3 fits; two of them need a 14.5 mm clamp.")]`. Bless the schema: `MAGCOUPLING_BLESS=1 cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml --test schema`. The approved correction says "Reword both places": the help text above and the README section "What the default design currently shows", which is `reference/magcoupling-py/README.md` line 195. That README is not the engine (the read-only rule covers `reference/magcoupling-py/magcoupling/`), and no test reads it (the audit test `test_readme_22mm_only_m3_fits` hard-codes the claim; gate 7 runs only `tests/`). In line 195, replace the sentence `At 22 mm only M3 fits, and two of them need a 14.5 mm clamp.` with `At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3.`; keep the rest of the line. The engine's own help text in `magcoupling/clamps.py` stays (read-only; `workbook_help` records it). Then search every place that states the claim as current (not `deviations.rs`'s `workbook_help` or `tests/data/python_schema.json`, which hold the workbook text on purpose, nor the Differences row Step 5 adds, which quotes it):

```bash
grep -rn "only M3" C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/README.md C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/src/engine/clamps.rs C:/Users/Cole/source/repos/lsim-mag-m2/docs/ai C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/README.md
grep -n "At 22 mm M4 no longer fits" C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/README.md
```
Expected after the edit: the first command prints nothing; the second prints line 195.

- [ ] **Step 3: Run the tests**

```bash
cargo test --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml 2>&1 | grep -E "FAILED|test result"
C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe C:/Users/Cole/source/repos/lsim-mag-m2/reference/magcoupling-py/tools/gen_differential.py --check
```
Expected: all ok (`documentation_entries_change_no_number`, `reworded_help_is_recorded_for_real_fields`, the metadata-parity test comparing the recorded workbook help with Python); data current.

- [ ] **Step 4: Gate**

```bash
cargo fmt --manifest-path C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/Cargo.toml
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task27.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task27.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)` (result groups in `MODULES` order); no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 5: README and commit.** Differences row E14: `Shaft clamps!C35 help; README` | `"At 22 mm only M3 fits; two of them need a 14.5 mm clamp."` | `"At 22 mm M4 no longer fits. Two M3 need a 14.5 mm clamp; three M2.5 need 18 mm, and from 18 mm up the calculator recommends M2.5 x 3."` | `E14 (documentation)`. `docs/ai/03-structure.yaml`, line `engine_files.deviations`: the status part (`all Planned` before Task 14, the previous task's status after it) becomes `E1-E14 Applied (all fourteen)`; the rest of the line is unchanged.

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/src magcoupling-rs/tests magcoupling-rs/README.md reference/magcoupling-py/README.md docs/ai/03-structure.yaml
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
docs(magcoupling): apply E14 the 22 mm boss statement

The boss help text and the vendored README's default-design section say
what the calculator computes at a 22 mm boss: M4 no longer fits, two M3
need 14.5 mm, three M2.5 from 18 mm. A test checks the statement against
the engine.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

### Task 28: Docs

**Model:** `sonnet` (CLAUDE.md section 5: the plan gives the code or the transcription rule). A `blocked` end, a red gate or a rejected review escalates every retry and rework to the session model.

**Files:**
- Modify: `magcoupling-rs/README.md`
- Modify: `docs/ai/03-structure.yaml`, `docs/ai/02-system.yaml`, `docs/ai/04-memory.yaml`, `docs/ai/05-update-tracker.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the project record of M2.

- [ ] **Step 1: README full pass** (`magcoupling-rs/README.md`):
  - Status: "M2 complete: every Python module except `fields3d` (M3) is ported; workbook parity covers all 1,149 checks; E1 to E14 applied (decisions D1 to D7 as recorded)."
  - Layout table: one row per engine file (`meta`, `compat`, `deviations`, `constants`, `library`, `calibration`, `model`, `metal_design`, `materials`, `temperature`, `clamps`, `sweeps`, `api`).
  - Tests table: `parity.rs` (1,149 checks and the totals test), `differential.rs` (per-group files, full run, selector pairs, BRANCHES), `python_schema.rs` (metadata, order, headline, table layouts), `schema.rs` (ranges, selectors, cells, table headers), `deviations.rs` (registry, hand-listed and golden changes, probes, all-corrections headline), `static_data.rs`, `robustness.rs`.
  - Regenerating data: the three generators (`MAGCOUPLING_BLESS=1 cargo test --test schema`, `gen_differential.py`, `MAGCOUPLING_BLESS=1 cargo test --test deviations each_deviation_alone_changes_exactly_its_registered_cells` for golden files) and what each writes.
  - Translation rules: add the integer rule (no `+ - *` on input-derived `i64`; saturating casts) and the dict-by-code rule with its three uses (`ScrewClasses::proof`, `selected_adhesive`, `screw_class_name`).
  - Porting a module, step 8: one equality-edge unit test per module (each threshold comparison at exact equality, the equality asserted first; the module tests of `model`, `metal_design`, `materials`, `temperature`, `clamps` and `sweeps` are the examples).
  - Differences from the workbook: confirm 14 rows, E1 to E14 in order:

```bash
grep -c "^| E[0-9]" C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/README.md
```
Expected: `14`.

- [ ] **Step 2: `docs/ai/03-structure.yaml`** `magcoupling_rs` entry: `role` says M2 complete; `engine_files` lists every module with one line each (as the `meta`/`compat` lines do); `ported` lists all twelve engine files; `remaining: [fields3d (M3)]`; `tests` lists the seven integration tests; `test_data` adds `static_data.json`, `differential/<group>.json` and `full.json`, `deviations/E3.json`, `E4.json`, `E5.json`; `deviations` line: all fourteen Applied.

- [ ] **Step 3: `docs/ai/02-system.yaml`** `major_subsystems.magcoupling`: responsibility "Engine complete (M2): every module but fields3d"; add `invariants:`:

```yaml
    invariants:
      - With Deviations::NONE every celled result, table cell and default input equals the workbook snapshot (1,149 checks; 1e-9 relative, 1e-12 absolute, text exact).
      - Every departure from the workbook is an approved, registered deviation (E1-E14 in deviations.rs). compute_all always applies Deviations::ALL; the switch exists only behind the test-only workbook-parity feature.
      - Differential data comes from the unchanged Python oracle; slider ranges are defined once in Rust (input_schema.json) and read by gen_differential.py; gate 7 fails on stale data.
      - A correction that changes more than 15 cells at defaults is a reviewed golden file (tests/data/deviations); one that changes none carries a registry probe.
      - The engine never panics on inputs; out-of-choice selector codes give NaN or "#N/A"; DesignInputs::validate() at input boundaries; no i64 arithmetic on input-derived values.
      - Rust-only results (rust_only metadata) have no cell and are skipped by the parity, metadata and differential tests.
```
and `current_statuses.magcoupling`: "M2 complete: engine ported except fields3d (M3); parity 1,149 checks; differential 10 group files + full run; E1-E14 applied; next: M3 fields3d, then M4 GUI". Keep the file's pre-existing YAML error at line 14 untouched (out of scope).

- [ ] **Step 4: `docs/ai/04-memory.yaml`**: replace the open question "Magcoupling M2 decisions needed ..." with one line recording D1 to D7 as decided (value per decision), and add three open items: "Addendum A lives on main (7a9d2e8): merge main into magcoupling/m2 before the Addendum A engine plan"; "M3 must apply E3 and E5 inside fields3d: rerun it with the corrected N42SH Br (E3) and multiply the web integral by 4 in fields3d.run (E5); see each entry's corrected_formula in magcoupling-rs/src/engine/deviations.rs"; "Addendum A engine plan: make the harmonic set a parameter and generalize peak_off_half_pitch beyond [1, 3, 5] (decision D7)".

- [ ] **Step 5: `docs/ai/05-update-tracker.md`**: new top entry `## <date> — Magcoupling M2: engine port complete` with bullets: modules ported and their cell counts; harness additions (multi-group modules, columnar data, tables, full run, BRANCHES, robustness); corrections E1 to E14 with the headline changes (pull-out 2.688 N·m, limit 93.06 °C, clamp M4 x 14, C106/C202 "Below ..."); decisions D1 to D7; equality-edge unit tests per module.

- [ ] **Step 6: Check the YAML files and run the gate**

```bash
C:/Users/Cole/AppData/Local/Programs/Python/Python312/python.exe -c "import yaml; [yaml.safe_load(open(f, encoding='utf-8')) for f in ('C:/Users/Cole/source/repos/lsim-mag-m2/docs/ai/03-structure.yaml', 'C:/Users/Cole/source/repos/lsim-mag-m2/docs/ai/04-memory.yaml')]; print('03 and 04 parse')"
C:/Users/Cole/AppData/Local/Programs/Python/Python312/python.exe -c "import yaml
try:
    yaml.safe_load(open('C:/Users/Cole/source/repos/lsim-mag-m2/docs/ai/02-system.yaml', encoding='utf-8'))
except yaml.YAMLError as e:
    print(str(e).splitlines()[1])"
MAGCOUPLING_PYTHON=C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py/.venv/Scripts/python.exe bash C:/Users/Cole/source/repos/lsim-mag-m2/linkage-sim-rs/scripts/gate.sh > C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task28.log 2>&1; echo "exit=$?"; tail -3 C:/Users/Cole/source/repos/lsim-mag-m2/magcoupling-rs/target/gate-task28.log
git -C C:/Users/Cole/source/repos/lsim-mag-m2 checkout -- docs/chebyshev_lambda
```
Expected: `03 and 04 parse`; the 02 error is still `  in "...02-system.yaml", line 14, column 5` (pre-existing, unchanged: no new error introduced); `exit=0`; the last line is `GATE PASS` and the line before it `differential data is current (calibration, model, retainers, mass, metal, materials, temperature, clamps, gap_sweep, pole_sweep + helpers + python schema + static data + full)`; no `SKIP gate 7/7` line (with `MAGCOUPLING_PYTHON` set, a missing interpreter fails the gate instead).

- [ ] **Step 7: Commit**

```bash
git -C C:/Users/Cole/source/repos/lsim-mag-m2 add magcoupling-rs/README.md docs/ai/02-system.yaml docs/ai/03-structure.yaml docs/ai/04-memory.yaml docs/ai/05-update-tracker.md
git -C C:/Users/Cole/source/repos/lsim-mag-m2 commit -F - <<'EOF'
docs(magcoupling): M2 engine port complete

Crate guide, project structure, system invariants for parity and the
deviation registry, the decisions taken, and the update tracker.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

---

## Self-review record

- **Spec coverage (M2 section):** module-by-module port with libraries as static data (Tasks 1, 3-8, 10, 11; magnet, adhesive and screw libraries plus alloys, checklists and sweep constants checked in `static_data.rs`); every input with path, label, unit, help, cell, default, kind, choices and a hand-defined slider range (range tables in Tasks 3, 8, 10; `schema.rs` enforces); one pure `compute_all` (Tasks 3-11; purity is structural, speed smoke-tested in Task 12); verdict texts character for character (parity and differential compare text exactly; `BRANCHES` forces every branch); workbook selector codes (choices copied from Python, metadata parity); result schema with label, unit, cell (scalar metadata from Python, table metadata from the workbook headers, Task 9); parity including 494 sweep, 165 screw-table and 160 input cells (Tasks 10-12 totals test); differential across every slider range and selector branch (per-module files, Task 13 full run and pairs); deviation registry with cells, workbook values, corrected formula, evidence, a test-only switch, exact registered changes (Tasks 14-27). Addendum A engine items and fields3d (M3) are out of scope by instruction.
- **Placeholders:** none; the decision-dependent spots give both variants (D1) or stop with escalation (D2, D3, D4, D6, D7 alternatives).
- **Type consistency:** `retainers` gains `inner_corner_radius_mm` in Task 21 and `materials::compute` gains `backiron` in Task 22, both stated in those tasks' Interfaces; `SweepContext.bond_inner` in Task 17; `ResultMeta.rust_only` in Task 15 (with `column_meta` updated); `Deviation` gains `workbook_help` (14), `changes_file` (16), `probes` (20) and `Literal::Error` (26), each added to every registry entry and to the unit test's `FAKE`.
- **Review Focus:** each of the five lines has its test in the owning task (Tasks 3, 8, 10, 12, 26).
- **Revision after the plan critique:** Task 15 filters Rust-only results out of `every_text_result_has_a_branches_entry` (blocking); every gate run sets `MAGCOUPLING_PYTHON` and its Expected names the check line; equality-edge unit tests in Tasks 3, 6, 8, 10 and 11 (verified on a scratch build of this plan's code: all pass, and each of the 23 operator swaps makes its test fail); the E13 -0.0 probe; the vendored README sentence (E14, Task 27); `Harmonic::s` replaces four copies of the circuit choice; the E7 sweeps block is exact code with let chains (clippy-clean) instead of `.then(..).flatten()`; D7 records the deferred harmonic parameter; the M6 clamps probe pins "No: key too large" (checked in the oracle); bless runs filter to the writer test; docs/ai/03-structure.yaml's deviations status moves with every correction; a model tier per task; the Python module -> owner table.
