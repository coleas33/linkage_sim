# Magnetic Coupling M1 — Math Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Independently verify every formula family of the magcoupling calculator (Python port of `magnetic_coupling_torque_calculator.xlsx`) and produce an adversarially verified findings report that the user reviews before any correction enters the Rust port.

**Architecture:** The Python package is vendored unchanged under `reference/magcoupling-py/` as a read-only oracle. A separate `audit/` package holds independent references (re-derivations, a 2D field model, a magpylib 3D model, literature formulas) and pytest checks comparing engine outputs to them; a failing check is a candidate finding. Task 8 runs skeptic agents over every candidate and writes the report.

**Tech Stack:** Python 3.12 (package floor 3.9), pytest, numpy, scipy, magpylib 5.x, Git Bash on Windows.

**Spec:** `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (section "M1 — Math verification").

## Global Constraints

- The engine under `reference/magcoupling-py/magcoupling/` is **read-only** in M1: no edits. Audit code lives only in `reference/magcoupling-py/audit/`.
- The vendored parity suite must stay green: `1159 passed` (with the optional extras installed).
- Every audit check carries `@pytest.mark.family(<torque|geometry|metal|materials|temperature|thermal|clamps|sweeps|constants>)` and builds its failure message with `audit.common.mismatch(...)`, naming the workbook cells.
- A check's reference must be **independent**: never produced by calling `magcoupling` internals. The engine is only ever the thing under test.
- Tolerances: `TOL_ALGEBRA = 1e-9` for same-algebra re-derivations; `TOL_MODEL = 0.02` for every engine-model-vs-reference comparison (defined once in `audit/common.py`); other numerical references state and justify their tolerance in the docstring.
- Never tune a check to pass; a failure is a candidate finding for Task 8.
- magpylib pinned `>=5,<6` (verified with 5.2.3).
- No changes to `linkage-sim-rs/` in M1.
- Work happens on branch `magcoupling/m1` in a git worktree outside the main checkout (the fix loop uses the main checkout, and its `git stash -u` must never see this work).
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
  ```

## Review Focus

Inputs the spec implies but default-only checks would never exercise, most likely to bite first. Each is pinned by a test in the owning task:

1. **Selector branches** (`coupling.backiron` 1 vs 0; clamp alloy 7075 vs 6061) — torque checks run in both circuits (Tasks 2, 3); clamp checks run for both alloys (Task 7).
2. **Non-default pole counts** (e.g. 8, 12, 14) — geometry and torque checks parametrized over pole count (Tasks 3, 4).
3. **Temperature range ends** (−40 °C and ≥ 80 °C operating) — Br(T) scaling and demagnetization checks at both ends (Tasks 3, 5).
4. **Small and large face gaps** (0.5 mm and 3.0 mm; 0.3 mm is not buildable because the inner block corners stand 0.373 mm proud of the face radius), where the planar harmonic approximation is most strained — the 2D reference is compared at 0.5, 1.4 and 3.0 mm in both circuits (Task 2).
5. **Measured drag entered** (`metal.measured_drag_Nm` set instead of `None`), which replaces the slip-loss estimate — thermal checks cover both branches (Task 6).

---

### Task 1: Vendor the Python package and the audit harness

**Files:**
- Create: `reference/magcoupling-py/` (the uploaded package, unchanged: `README.md`, `pyproject.toml`, `magcoupling/`, `tests/`, `tools/`, `examples/`)
- Create: `reference/magcoupling-py/.gitignore`
- Create: `reference/magcoupling-py/requirements-audit.txt`
- Create: `reference/README.md`
- Create: `reference/magcoupling-py/audit/__init__.py`, `audit/common.py`, `audit/references/__init__.py`, `audit/tests/__init__.py`, `audit/tests/conftest.py`, `audit/tests/test_harness_smoke.py`

**Interfaces:**
- Consumes: the uploaded archive `C:\Users\Cole\.claude\uploads\3165685f-bf49-42bb-a036-671c0e065429\ca3845b2-magcoupling.zip` (top-level folder `repo/`).
- Produces (used by Tasks 2–8): `audit.common.defaults() -> DesignInputs`, `run(inp=None) -> DesignResults`, `vary(inp, changes: dict[str, Any]) -> DesignInputs`, `rel_err(value, reference) -> float`, `mismatch(what, cells, engine, reference, tol) -> str`, constants `MU0_EXACT`, `TOL_ALGEBRA` and `TOL_MODEL`, plus the multi-check helpers defined in `common.py` below (the only definition; later tasks never edit `common.py`); the `family` marker; `audit/out/results.json`, written after every audit run as a list of `{id, family, outcome, message, doc}`.

- [ ] **Step 1: Create the worktree and branch**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation worktree add /c/Users/Cole/source/repos/linkage_simulation-m1 -b magcoupling/m1 main
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 rev-parse --abbrev-ref HEAD
```

Expected: `magcoupling/m1`. **Every command in this plan uses absolute paths into this worktree, and every `File:` path is relative to the worktree root `/c/Users/Cole/source/repos/linkage_simulation-m1`. Never write into the main checkout.**

- [ ] **Step 2: Vendor the package unchanged**

```bash
WT=/c/Users/Cole/source/repos/linkage_simulation-m1
[ "$(git -C "$WT" rev-parse --abbrev-ref HEAD)" = magcoupling/m1 ] || { echo "STOP: worktree missing or on the wrong branch"; exit 1; }
mkdir -p "$WT/reference/magcoupling-py"
TMP=$(mktemp -d)
unzip -q "/c/Users/Cole/.claude/uploads/3165685f-bf49-42bb-a036-671c0e065429/ca3845b2-magcoupling.zip" -d "$TMP"
cp -r "$TMP"/repo/. "$WT/reference/magcoupling-py/"
find "$WT/reference/magcoupling-py" -name __pycache__ -type d -prune -exec rm -rf {} +
ls "$WT/reference/magcoupling-py"
```

Expected: `README.md  examples  magcoupling  pyproject.toml  tests  tools`

- [ ] **Step 3: Add ignore rules, audit requirements, and the reference README**

File: `reference/magcoupling-py/.gitignore` (create)
```gitignore
__pycache__/
*.egg-info/
.pytest_cache/
.venv/
audit/out/
```

File: `reference/magcoupling-py/requirements-audit.txt` (create)
```text
magpylib>=5,<6
numpy
scipy
pytest
matplotlib
openpyxl
```

File: `reference/README.md` (create)
```markdown
# reference/

Upstream code kept verbatim as an **oracle** for tests.

- `magcoupling-py/`: the magnetic coupling calculator (Python port of
  `magnetic_coupling_torque_calculator.xlsx`, version 1.0.0). **Do not edit
  `magcoupling-py/magcoupling/`**: parity and differential tests for the Rust
  port (`magcoupling-rs/`) compare against it unchanged. Independent physics
  checks live in `magcoupling-py/audit/`
  (spec: `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md`).
```

- [ ] **Step 4: Create the audit virtual environment and run the vendored parity suite**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py
python -m venv .venv
./.venv/Scripts/python -m pip install -q -e . -r requirements-audit.txt
./.venv/Scripts/python -m pytest -q tests
```

Expected: `1159 passed` (the optional 3D and drawing tests run because the extras are installed).

- [ ] **Step 5: Write the audit harness**

File: `reference/magcoupling-py/audit/__init__.py` (create)
```python
"""Independent verification of the magcoupling engine (M1 of the magcoupling spec).

The vendored engine under ``magcoupling/`` is a read-only oracle here: nothing in
``audit/`` modifies it. Each check compares an engine value against an
independent reference (re-derivation, separate numerical model, limit/scaling
law, or literature formula). A failing check is a *candidate finding*.
"""
```

File: `reference/magcoupling-py/audit/common.py` (create)
```python
"""Shared helpers for every audit check. Keep this file small and dependency-free."""
from __future__ import annotations

import math
from typing import Any

from magcoupling import DesignInputs, compute_all, set_input
from magcoupling.api import DesignResults

#: Exact vacuum permeability (the workbook uses the rounded 1.256637e-6).
MU0_EXACT = 4e-7 * math.pi

#: Tolerance for checks that re-derive the same closed-form algebra.
TOL_ALGEBRA = 1e-9

# Engine analytical model vs an exact or independent reference. 2 %: the default design misses the hot-torque
# requirement by ~10 %, so a systematic model bias of a few percent is material and must surface as a candidate
# finding.
TOL_MODEL = 0.02

#: Peak-angle check: torque at the engine's pull-out angle (a quarter pole-pair pitch) vs the true maximum of the
#: torque-angle curve. The 3D curve (test_torque_laws.py) is converged to ~1e-7 and its maximum is located to 1e-4 of
#: a pole pitch; the exact 2D curve (test_torque_field2d.py) is exact to < 1e-3 (BEM) and its maximum is located to
#: 1e-7 rad. So 1e-3 is a resolution bound.
TOL_PEAK = 1e-3


def defaults() -> DesignInputs:
    """Workbook-default inputs (a fresh object every call)."""
    return DesignInputs()


def run(inp: DesignInputs | None = None) -> DesignResults:
    """Run the engine; defaults when ``inp`` is None."""
    return compute_all(inp or defaults())


def vary(inp: DesignInputs, changes: dict[str, Any]) -> DesignInputs:
    """Return a copy of ``inp`` with dotted-path inputs changed, e.g.
    ``vary(defaults(), {"coupling.npole": 12, "metal.face_gap_mm": 1.2})``."""
    out = inp
    for path, value in changes.items():
        out = set_input(out, path, value)
    return out


def rel_err(value: float, reference: float) -> float:
    """|value - reference| / |reference| (absolute error when reference is 0)."""
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def mismatch(what: str, cells: str, engine: float, reference: float, tol: float) -> str:
    """Assertion message every failing check uses, so findings are uniform:
    what was compared, which workbook cells, both values, the relative error, and the tolerance."""
    return (f"{what} [{cells}]: engine={engine!r} reference={reference!r} "
            f"rel_err={rel_err(engine, reference):.3e} tol={tol:.1e}")


# --------------------------------------------------------------------------- multi-cell checks
#: assert_all prints at most this many failing comparisons in full, then only counts the rest.
MAX_REPORTED = 12


def assert_all(items) -> None:
    """Fail once, listing every failing comparison.

    ``items`` is an iterable of (what, cells, engine, reference, tol) tuples. Each passes when
    rel_err(engine, reference) <= tol; NaN never passes. The message starts with the failure count and
    formats each failure with mismatch(), so one check reports all of its wrong cells."""
    items = list(items)
    bad = [mismatch(*it) for it in items if not rel_err(it[2], it[3]) <= it[4]]
    shown = " || ".join(bad[:MAX_REPORTED])
    more = f" || ... and {len(bad) - MAX_REPORTED} more" if len(bad) > MAX_REPORTED else ""
    assert not bad, f"{len(bad)} of {len(items)} comparisons failed: {shown}{more}"


def flag_item(what: str, cells: str, holds: bool) -> tuple:
    """A yes/no condition as an assert_all item (engine 1.0 when it holds, reference 1.0).
    Put the numbers behind the condition in ``what``."""
    return (f"{what} (1 = holds)", cells, 1.0 if holds else 0.0, 1.0, 0.0)


def text_item(what: str, cells: str, engine_text: str, expected_text: str) -> tuple:
    """A text comparison as an assert_all item, with both texts in the description."""
    return flag_item(f"{what}: engine {engine_text!r}, expected {expected_text!r}", cells, engine_text == expected_text)


class ScenarioCache:
    """``cache[name]`` -> (inputs, results) for vary(defaults(), scenarios[name]); each scenario runs once."""

    def __init__(self, scenarios: dict[str, dict[str, Any]]):
        self.scenarios = scenarios
        self._done: dict[str, tuple[DesignInputs, DesignResults]] = {}

    def __getitem__(self, name: str) -> tuple[DesignInputs, DesignResults]:
        if name not in self._done:
            inp = vary(defaults(), self.scenarios[name])
            self._done[name] = (inp, run(inp))
        return self._done[name]
```

File: `reference/magcoupling-py/audit/references/__init__.py` (create)
```python
```

File: `reference/magcoupling-py/audit/tests/__init__.py` (create)
```python
```

File: `reference/magcoupling-py/audit/tests/conftest.py` (create)
```python
"""Audit test plumbing: the ``family`` marker and a machine-readable results file.

Every check is marked ``@pytest.mark.family("<name>")`` with one of FAMILIES.
After a run, ``audit/out/results.json`` lists every check with its family,
outcome and failure message; the findings report is built from that file.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FAMILIES = ("torque", "geometry", "metal", "materials", "temperature", "thermal", "clamps", "sweeps", "constants")
_RESULTS: list[dict] = []


def pytest_configure(config):
    config.addinivalue_line("markers", "family(name): formula family this audit check belongs to")


def pytest_collection_modifyitems(items):
    for item in items:
        m = item.get_closest_marker("family")
        if m is None or not m.args or m.args[0] not in FAMILIES:
            raise pytest.UsageError(f"{item.nodeid}: every audit check needs @pytest.mark.family(<one of {FAMILIES}>)")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if rep.when == "call" or (rep.when == "setup" and rep.outcome != "passed"):
        _RESULTS.append({
            "id": item.nodeid,
            "family": item.get_closest_marker("family").args[0],
            "outcome": rep.outcome,
            "message": (str(rep.longrepr.reprcrash.message) if rep.failed and hasattr(rep.longrepr, "reprcrash") else ""),
            "doc": (item.function.__doc__ or "").strip(),
        })


def pytest_sessionfinish(session, exitstatus):
    out = Path(__file__).resolve().parents[1] / "out"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(_RESULTS, indent=1), encoding="utf-8")
```

File: `reference/magcoupling-py/audit/tests/test_harness_smoke.py` (create)
```python
import pytest

from audit.common import defaults, mismatch, rel_err, run, vary


@pytest.mark.family("constants")
def test_harness_runs_engine_defaults():
    """Harness sanity: defaults run and reproduce the README headline pull-out (2.647 N·m at 50 °C)."""
    res = run()
    assert abs(res.model.pullout_Nm - 2.647) < 5e-4, mismatch("pull-out at op temp", "Calculator", res.model.pullout_Nm, 2.647, 5e-4)


@pytest.mark.family("constants")
def test_vary_changes_only_named_input():
    """Harness sanity: vary() returns a changed copy and leaves the original untouched."""
    base = defaults()
    changed = vary(base, {"coupling.npole": 12})
    assert changed.coupling.npole == 12 and base.coupling.npole != 12
    assert rel_err(2.0, 1.0) == 1.0
```

- [ ] **Step 6: Run the harness smoke tests**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py
./.venv/Scripts/python -m pytest -q audit/tests
ls audit/out
```

Expected: `2 passed`; `results.json` exists.

- [ ] **Step 7: Commit**

```bash
WT=/c/Users/Cole/source/repos/linkage_simulation-m1
git -C "$WT" add reference/README.md reference/magcoupling-py
git -C "$WT" diff --cached --name-only | grep -E "\.venv/|__pycache__|audit/out/" && echo "STOP: ignored paths staged" && exit 1
git -C "$WT" commit -m "chore(reference): vendor magcoupling 1.0.0 as oracle; add audit harness

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
```

### Task 2: Torque model against independent references (term-by-term chain, exact 2D flat-block field with polygonal steel, cylindrical harmonics)

**Files:**
- Create: `reference/magcoupling-py/audit/references/planar.py` (the module's owner portion; Task 3 later appends its planar torque-chain helpers to it)
- Create: `reference/magcoupling-py/audit/references/field2d.py`
- Create: `reference/magcoupling-py/audit/tests/test_torque2d_reference_sanity.py`
- Create: `reference/magcoupling-py/audit/tests/test_torque_field2d.py`
- Not modified: `reference/magcoupling-py/audit/common.py`. Task 1 owns it completely, including the shared `TOL_MODEL`, `TOL_PEAK` and `ScenarioCache`.

**Interfaces:**
- Consumes: `audit.common` helpers `defaults`, `run`, `vary`, `rel_err`, `mismatch`, `ScenarioCache` (one engine run per named scenario, cached), `MU0_EXACT`, `TOL_ALGEBRA`, `TOL_MODEL` (2 %) and `TOL_PEAK` (1e-3), all defined by Task 1. This task never edits `audit/common.py` and defines neither `TOL_MODEL` nor `TOL_PEAK`.
- Engine result fields used (dotted paths, workbook cells):
  - `model.inner_face_radius_mm` (Calculator!C54), `inner_thickness_mm` (C20), `inner_width_mm` (C19), `inner_br_T` (C21)
  - `model.outer_face_apothem_mm` (C56), `outer_thickness_mm` (C30), `outer_width_mm` (C29), `outer_br_T` (C31), `outer_back_apothem_mm` (C60)
  - `model.active_length_mm` (C33), `alpha_br_per_C` (C35), `f_cal` (C42), `face_gap_mm` (C57), `gap_radius_mm` (C64)
  - `model.fill_inner`, `fill_outer` (C66, C67), `br_inner_T_op`, `br_outer_T_op` (C69, C70)
  - `model.k1`, `b_i1`, `b_o1`, `s1_iron`, `s1_free`, `tau1_Pa` (C71 to C76); the same for n = 3 (C77 to C82) and n = 5 (C83 to C88)
  - `model.tau_Pa` (C89), `area_lever_m3` (C90), `torque_2d_Nm` (C91), `f_end` (C92), `pullout_Nm` (C93), `pullout_20C_Nm` (C94)
  - `calibration.poles_per_ring` (Calibration!C13), `face_radius_mm` (C28), `outer_face_apothem_mm` (C31), `br_test_T` (C36), `torque_2d_Nm` (C43)
- Engine inputs read or varied: `coupling.npole` (Calculator!C5), `coupling.backiron` (C6), `coupling.inner_back_apothem_mm` (C8), `coupling.op_temp_C` (C10), `coupling.c_end` (C41), `coupling.mu0` (C43), `coupling.magnets.part_inner` (C11), `coupling.magnets.part_outer` and `manual_outer_width_mm` / `manual_outer_thickness_mm` / `manual_outer_br_T` (C12, C25 to C27), `metal.face_gap_mm` (Metal design!C119), `metal.bond_inner_mm` (C120), `metal.bond_outer_mm` (C121), `calibration.magnet_length_mm` / `magnet_width_mm` / `magnet_thickness_mm` (Calibration!C18 to C20).
- Produces (reusable by Tasks 3 to 8):
  - `audit.references.planar` (SI units): `square_wave_harmonic(br_T, n, fill) -> float` and `planar_s_factor(k, t_i, t_o, g, backed) -> float`. These are the only copies of the planar formulas; later tasks import them. Task 3 appends its own helpers to the same file (`br_at`, `wave_number`, `iron_backed_factor` and `free_space_factor`, which delegate to `planar_s_factor`, `harmonic_shear_stress`, `planar_shear_stress`, `torque_on_cylinder`, `end_factor`) and does not redefine either function. Task 4's `slab_field2d` imports `square_wave_harmonic` and Task 4's `remanence` re-exports Task 3's `br_at`, so `planar.py` (with Task 3's append) must be integrated before them.
  - `audit.references.field2d`. Lengths are in mm for the dataclasses and SI elsewhere; points are complex `x + iy` in metres; fields are complex `B_x + i B_y`.
    - `MM = 1e-3`
    - `Magnet(vertices: tuple, J: complex)`, `Sheets(z1, z2, k)`
    - `current_sheets(magnets) -> Sheets`, `charge_sheets(magnets) -> Sheets`
    - `b_from_currents(z, sh, scale=1.0)`, `b_from_inverted_currents(z, sh, rho)`, `b_from_charges(z, sh)`
    - `IronCircles(r_inner_mm, r_outer_mm, generations=8)`, `image_maps(iron)`
    - `IronPolygons(hub_apothem_mm, cup_apothem_mm, n_sides, panels_per_side=40)`
    - `polygon_iron_charges(magnets, iron, cup_phase, antiperiod=None) -> Sheets`
    - `b_field(z, magnets, iron=None, cup_phase=0.0, antiperiod=None) -> ndarray`
    - `maxwell_torque_per_length(magnets, r_mm, n_poles, iron=None, cup_phase=0.0, n_per_pitch=256) -> float` (N·m/m on the inner rotor)
    - `rect_block(back_apothem_m, thickness_m, width_m, angle, br_signed) -> Magnet`
    - `CouplingSection(n_poles, inner_back_mm, inner_thickness_mm, inner_width_mm, outer_face_mm, outer_thickness_mm, outer_width_mm, br_inner_T, br_outer_T)`, with `.inner()`, `.outer(delta_mech)`, `.inner_corner_mm()`, `.outer_back_corner_mm()` and `.stress_radius_mm()`
    - `torque_per_length(sec, theta_e, iron=None, r_mm=None) -> float`
    - `Pullout(torque_per_m, theta_e_peak, at_quarter, harmonics)`, `pullout(sec, iron=None, n_samples=48) -> Pullout`
    - `cylindrical_s_factor(m, r1a, r1b, r2a, r2b, r_gap, iron=None) -> float`
- Cell ownership: this task is the sole same-algebra owner of the torque chain Calculator!C66:C94 (`test_calculator_torque_chain_rederived`, TOL_ALGEBRA). Task 3's copy of that check (`test_torque_chain_term_by_term`, C71 to C94) was deleted; its free-space asymmetric scenario now runs here as the chain case `free-unequal-rings`. Task 3's 3D, law and temperature-scaling checks test the model behind C69 to C94 by other methods and do not re-derive those cells, and Task 4's `test_geometry_mass.py` also compares the fill factors C66 and C67 with its block-geometry reference.

**Design decisions:**
- **Polygonal steel.** Steel is modelled with a BEM on the real polygonal hub flats and cup pockets, not with circle images, because flat blocks cannot sit against a circle. The outer back corners (18.17 mm) lie beyond C60 (17.89 mm), and a circle clear of the corners leaves an air crescent behind each block, which reads 5 % low. The image series is still used where it is exact: arcs (the S_n checks) and the sliced-ring sanity test. It also cross-checks the BEM to 2.3e-4 on 200-gons.
- **The steel verdict uses the as-built steel:** hub flats at C8 minus the bondline (Metal design!C120), pockets at C60 plus the bondline (C121). The engine's own mass and pocket rows describe this steel. A second check puts the steel on the magnet backs, the idealisation inside S_iron, so the formula's error can be told apart from the omitted bondline. That check is diagnostic only.
- **One-pitch BEM.** The sources flip sign every pole pitch and the steel is N-fold symmetric, so the BEM solves one pitch (`antiperiod`). Its gap field matches the full-boundary solution to 4e-14, and it is about 12 times faster.
- **Face gaps 0.5, 1.4 and 3.0 mm, in both circuits.** 0.3 mm cannot be modelled: the inner corners stand 0.373 mm proud of the face radius, so at 0.3 mm the blocks overlap (C9 = −0.073 mm).
- **Tolerances:**
  - `TOL_MODEL` (shared, from `audit.common`) for every model-vs-exact comparison. Harmonics are measured against the fundamental: |a_n engine − a_n reference| ≤ TOL_MODEL · a_1.
  - `TOL_ALGEBRA` (shared, from `audit.common`) for the term-by-term chain.
  - `TOL_PEAK` (shared, from `audit.common`) for the quarter-pitch peak check; its 1e-3 resolution bound is justified where Task 1 defines it.
  - `TOL_PLANAR_LIMIT = 1e-6`, the only tolerance this task defines, in `test_torque_field2d.py` (only this task uses it), justified there.

- [ ] **Step 1: Write the reference modules**

File: `reference/magcoupling-py/audit/references/planar.py` (create)

```python
"""Planar (unrolled) harmonic model of two facing magnet arrays: the textbook building blocks (M1).

Shared by every torque task so the planar formulas exist once. Nothing here uses the engine.

* ``square_wave_harmonic``: amplitude of harmonic n of an alternating square-wave magnetization
  +/-B that fills the fraction ``fill`` of each pole pitch. The Fourier series of that pulse train
  has only odd harmonics, B_n = 4 B / (n pi) sin(n pi fill / 2); even harmonics vanish by half-wave
  symmetry (any Fourier-series text, e.g. Kreyszig, Advanced Engineering Mathematics, ch. 11).
* ``planar_s_factor``: geometry factor S of one harmonic (wave number k) between two planar arrays
  of thickness t_i and t_o separated by the gap g, normalised so the shear-stress amplitude is
  B_i B_o / (2 mu0) * S. Free space: (1 - e^-k t_i)(1 - e^-k t_o) e^-k g / 2 (a magnetic charge
  sheet s cos kx gives H = s/2 e^-k|y| on both sides). Infinitely permeable planes at both array
  backs: sinh(k t_i) sinh(k t_o) / sinh(k (t_i + t_o + g)) (Green's function between two
  equipotential planes). ``test_torque2d_reference_sanity.py`` checks both against the exact
  cylindrical solution in ``field2d`` as R_gap grows, and against the thick-magnet limit e^-k g / 2.

SI units: k in 1/m, lengths in m.
"""
from __future__ import annotations

import math


def square_wave_harmonic(br_T: float, n: int, fill: float) -> float:
    """Amplitude [T] of harmonic n (n >= 1) of an alternating +/-br_T square wave with fill fraction ``fill``."""
    if n < 1:
        raise ValueError(f"harmonic order must be >= 1, got {n}")
    if n % 2 == 0:
        return 0.0
    return 4 * br_T / (n * math.pi) * math.sin(n * math.pi * fill / 2)


def planar_s_factor(k: float, t_i: float, t_o: float, g: float, backed: bool) -> float:
    """Planar geometry factor S for wave number k [1/m], thicknesses t_i, t_o and gap g [m].
    ``backed`` puts infinitely permeable planes at both array backs; otherwise free space."""
    if backed:
        return math.sinh(k * t_i) * math.sinh(k * t_o) / math.sinh(k * (t_i + t_o + g))
    return (1 - math.exp(-k * t_i)) * (1 - math.exp(-k * t_o)) * math.exp(-k * g) / 2
```

File: `reference/magcoupling-py/audit/references/field2d.py` (create)

```python
"""Independent 2D magnetostatic reference for the coupling cross-section (M1, Task 2).

Nothing here uses the engine (only MU0_EXACT comes from audit.common). The coupling is
treated as infinitely long (2D).

Numerical-exact flat-block model (``b_field``, ``torque_per_length``, ``pullout``):
  * Each block is a uniformly magnetized rectangle, represented by its bound
    surface current mu0 K = (J x n)_z on the edges. The field of a straight sheet
    is integrated in closed form, so the free-space field is exact to rounding.
  * Steel, option 1, ``IronCircles``: infinitely permeable circular surfaces by
    the method of images (a line current I at z has the image I, same sign, at
    R^2 / conj z), a converged series of reflections between the two circles.
  * Steel, option 2, ``IronPolygons``: infinitely permeable polygonal surfaces
    (the real flat hub faces and cup pockets) by a boundary-element method: the
    magnetic scalar potential is constant on each steel body and each body's net
    induced charge is zero. Needed because flat blocks cannot touch a circle.
  * Torque per unit length from the Maxwell stress on a circle in the air gap.

Exact cylindrical harmonic solution (``cylindrical_s_factor``): sinusoidal RADIAL
magnetization M cos(m theta) in two concentric annuli, free space or between
infinitely permeable circles; torque from the force on the outer ring's magnetic
charges. This is the curved counterpart of the engine's planar S_n factors.

Conventions: SI inside; a point is a complex number z = x + iy; a field is the
complex number B_x + i B_y. ``CouplingSection``, ``IronCircles`` and
``IronPolygons`` take lengths in mm, like the engine's fields.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from audit.common import MU0_EXACT

MM = 1e-3


# --------------------------------------------------------------------------- sources
@dataclass(frozen=True)
class Magnet:
    """Uniformly magnetized infinite prism: counter-clockwise polygon vertices
    (complex, metres) and polarization J = mu0 * M (complex, tesla)."""
    vertices: tuple
    J: complex


@dataclass(frozen=True)
class Sheets:
    """Straight sheets z1 -> z2 with a uniform strength (tesla): mu0 K for currents
    (K along +z), mu0 sigma for magnetic charge."""
    z1: np.ndarray
    z2: np.ndarray
    k: np.ndarray


def _edge_sheets(magnets: list[Magnet], part: str) -> Sheets:
    """For each edge, with outward normal n: conj(J) n = J.n + i (J x n)_z. The real
    part is the surface charge mu0 sigma = J.n, the imaginary part the bound current."""
    z1, z2, k = [], [], []
    for mag in magnets:
        v = mag.vertices
        for a, b in zip(v, v[1:] + v[:1]):
            n = -1j * (b - a) / abs(b - a)          # outward normal of a CCW polygon edge
            c = np.conj(mag.J) * n
            s = c.real if part == "charge" else c.imag
            if abs(s) > 1e-12 * abs(mag.J):
                z1.append(a), z2.append(b), k.append(s)
    return Sheets(np.array(z1, dtype=complex), np.array(z2, dtype=complex), np.array(k, dtype=float))


def current_sheets(magnets: list[Magnet]) -> Sheets:
    return _edge_sheets(magnets, "current")


def charge_sheets(magnets: list[Magnet]) -> Sheets:
    return _edge_sheets(magnets, "charge")


def _kernel(z: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Integral over the straight segment z1 -> z2 of ds / (z - zeta), shape (points, sheets):
    conj(d) Log((z - z1)/(z - z2)), d the unit direction. z - zeta runs along a straight
    line that misses 0 when z is off the segment, so the principal Log is exact."""
    zz = z[:, None]
    d = (z2 - z1) / np.abs(z2 - z1)
    return np.conj(d)[None, :] * np.log((zz - z1[None, :]) / (zz - z2[None, :]))


def b_from_currents(z: np.ndarray, sh: Sheets, scale: float = 1.0) -> np.ndarray:
    """B_x + i B_y of current sheets, optionally scaled about the origin by ``scale``.
    A line current I at z' gives conj(B) = -i mu0 I / (2 pi (z - z')). Scaling keeps each
    element's current, so the scaled sheet carries density K / scale."""
    cb = -1j / (2 * math.pi) * (_kernel(z, sh.z1 * scale, sh.z2 * scale) @ (sh.k / scale))
    return np.conj(cb)


def b_from_inverted_currents(z: np.ndarray, sh: Sheets, rho: float) -> np.ndarray:
    """B_x + i B_y of the inversion images (radius rho, same sign) of current sheets.
    Element K dt at z'(t) = z1 + t d becomes a line current K dt at w = rho^2 / conj z'(t).
    With e = conj d and u(t) = z conj z'(t) - rho^2 (linear in t):
    integral over [0, l] of dt / (z - w) = l / z + rho^2 / (z^2 e) Log(u(l) / u(0))."""
    zz = z[:, None]
    length = np.abs(sh.z2 - sh.z1)
    e = np.conj((sh.z2 - sh.z1) / length)
    integral = (length[None, :] / zz + rho ** 2 / (zz ** 2 * e[None, :])
                * np.log((zz * np.conj(sh.z2)[None, :] - rho ** 2) / (zz * np.conj(sh.z1)[None, :] - rho ** 2)))
    return np.conj(-1j / (2 * math.pi) * (integral @ sh.k))


def b_from_charges(z: np.ndarray, sh: Sheets) -> np.ndarray:
    """B_x + i B_y (= mu0 H, valid in air) of charge sheets: conj(B) = mu0 sigma / (2 pi) * kernel."""
    return np.conj(_kernel(z, sh.z1, sh.z2) @ sh.k / (2 * math.pi))


def _log_potential(z: np.ndarray, z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """Integral over the segment of ln|z - zeta| ds, shape (points, sheets). In segment
    coordinates (u along, v across, segment on 0 <= s <= l):
    F(s) = x ln sqrt(x^2 + v^2) - x + v atan(x / v), x = s - u; result F(l) - F(0)."""
    length = np.abs(z2 - z1)
    d = (z2 - z1) / length
    w = (z[:, None] - z1[None, :]) / d[None, :]
    u, v = w.real, w.imag

    def f(s):
        x = s - u
        r2 = x * x + v * v
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(r2 > 0, 0.5 * x * np.log(np.where(r2 > 0, r2, 1.0)), 0.0)
            t3 = np.where(v != 0, v * np.arctan(x / np.where(v != 0, v, 1.0)), 0.0)
        return t1 - x + t3

    return f(length[None, :]) - f(0.0)


# --------------------------------------------------------------------------- steel
@dataclass(frozen=True)
class IronCircles:
    """Infinitely permeable steel filling r <= r_inner_mm (hub) and r >= r_outer_mm (cup).

    The image series reflects the sources alternately in the two circles,
    ``generations`` reflections per chain. Two generations move the images by
    q = (r_outer / r_inner)^2 in radius; an alternating ring of N poles has only angular
    harmonics of order >= N/2, so each generation lowers the omitted field by about
    (r_inner / r_outer)^(N/2) (0.055 for the default design; 8 generations leave
    < 1e-10, which ``test_sanity_image_series_converged`` confirms)."""
    r_inner_mm: float
    r_outer_mm: float
    generations: int = 8


def image_maps(iron: IronCircles) -> list[tuple[str, float]]:
    """Transformations taking the real sources to each image: ("scale", s) is z -> s z,
    ("invert", rho) is z -> rho^2 / conj z. Reflection in a circle of radius R maps
    ("scale", s) to ("invert", R / sqrt s) and ("invert", rho) to ("scale", (R / rho)^2)."""
    r1, r2 = iron.r_inner_mm * MM, iron.r_outer_mm * MM
    maps = []
    for first, second in ((r1, r2), (r2, r1)):
        kind, value = "scale", 1.0
        for g in range(iron.generations):
            radius = first if g % 2 == 0 else second
            kind, value = ("invert", radius / math.sqrt(value)) if kind == "scale" else ("scale", (radius / value) ** 2)
            maps.append((kind, value))
    return maps


@dataclass(frozen=True)
class IronPolygons:
    """Infinitely permeable steel inside a regular ``n_sides``-gon of apothem hub_apothem_mm
    (flats centred at angles 2 pi i / n_sides) and outside one of apothem cup_apothem_mm
    (flats centred at cup_phase + 2 pi i / n_sides, cup_phase passed to ``b_field``: the cup
    turns with the outer ring). Solved by collocation with piecewise-constant charge panels,
    graded towards the vertices; ``test_sanity_bem_converged`` bounds the discretisation error."""
    hub_apothem_mm: float
    cup_apothem_mm: float
    n_sides: int
    panels_per_side: int = 40


def _polygon_panels(apothem: float, n_sides: int, per_side: int, phase: float) -> tuple[np.ndarray, np.ndarray]:
    rv = apothem / math.cos(math.pi / n_sides)
    t = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, per_side + 1))
    a, b = [], []
    for i in range(n_sides):
        c = phase + 2 * math.pi * i / n_sides
        v0, v1 = rv * np.exp(1j * (c - math.pi / n_sides)), rv * np.exp(1j * (c + math.pi / n_sides))
        pts = v0 + (v1 - v0) * t
        a.append(pts[:-1]), b.append(pts[1:])
    return np.concatenate(a), np.concatenate(b)


def _polygon_support(z: np.ndarray, n_sides: int, phase: float) -> np.ndarray:
    """Largest projection of each point on the flat normals of a regular polygon (flats centred at
    phase + 2 pi i / n_sides): a point lies outside the polygon of apothem a when it exceeds a."""
    normals = np.exp(1j * (phase + 2 * math.pi * np.arange(n_sides) / n_sides))
    return np.max((z[:, None] * np.conj(normals)[None, :]).real, axis=1)


def polygon_iron_charges(magnets: list[Magnet], iron: IronPolygons, cup_phase: float,
                         antiperiod: int | None = None) -> Sheets:
    """Induced charge on the steel surfaces: the scalar potential psi (B = -grad psi) is a
    constant C_hub on the hub surface and C_cup on the cup surface, and each body's net
    charge is zero (div B = 0 inside the steel, H = 0 there). The charge model assumes the
    magnets sit in the air, so a magnet vertex inside the hub polygon or outside the cup polygon
    is refused (touching the steel, as a block on its flat does, is allowed).

    ``antiperiod`` = N (even) declares that turning the sources by 2 pi / N flips their sign, as in
    any coupling section with N alternating poles per ring. The steel is also N-fold symmetric, so
    the induced charge flips sign from one pole pitch to the next and both constants vanish (each
    equals its own negative). Then only the panels of the first pitch are unknowns, with the kernel
    summed over the N pitches with alternating signs: the same discrete solution (the panels of each
    pitch are exact rotated copies), N times fewer collocation points and an N^2 smaller system.
    ``test_sanity_bem_antiperiodic_matches_full`` checks the two solutions agree. None solves the
    full boundary, valid for any sources."""
    verts = np.array([v for mag in magnets for v in mag.vertices], dtype=complex)
    hub, cup = iron.hub_apothem_mm * MM, iron.cup_apothem_mm * MM
    if (_polygon_support(verts, iron.n_sides, 0.0).min() < hub * (1 - 1e-9)
            or _polygon_support(verts, iron.n_sides, cup_phase).max() > cup * (1 + 1e-9)):
        raise ValueError(f"a magnet enters the steel polygons (hub apothem {iron.hub_apothem_mm} mm, "
                         f"cup apothem {iron.cup_apothem_mm} mm)")
    h1, h2 = _polygon_panels(hub, iron.n_sides, iron.panels_per_side, 0.0)
    c1, c2 = _polygon_panels(cup, iron.n_sides, iron.panels_per_side, cup_phase)
    z1, z2 = np.concatenate([h1, c1]), np.concatenate([h2, c2])
    p, nh = len(z1), len(h1)
    mid, length = (z1 + z2) / 2, np.abs(z2 - z1)
    src = charge_sheets(magnets)
    if antiperiod is None:
        a = np.zeros((p + 2, p + 2))
        a[:p, :p] = -_log_potential(mid, z1, z2) / (2 * math.pi)
        a[:nh, p] = -1.0
        a[nh:p, p + 1] = -1.0
        a[p, :nh], a[p + 1, nh:p] = length[:nh], length[nh:]
        rhs = np.zeros(p + 2)
        rhs[:p] = _log_potential(mid, src.z1, src.z2) @ src.k / (2 * math.pi)
        return Sheets(z1, z2, np.linalg.solve(a, rhs)[:p])
    if antiperiod % 2 or iron.n_sides % antiperiod:
        raise ValueError(f"anti-period {antiperiod} needs an even pole count dividing the {iron.n_sides} steel sides")
    per = nh // antiperiod                                          # panels per pitch on each body
    signs = (-1.0) ** np.arange(antiperiod)
    colloc = np.concatenate([mid[:per], mid[nh:nh + per]])
    g = -_log_potential(colloc, z1, z2) / (2 * math.pi)             # (2 per, p): pitch-major panel order

    def fold(block: np.ndarray) -> np.ndarray:                      # sum the N pitches with signs +, -, +, ...
        return (block.reshape(2 * per, antiperiod, per) * signs[None, :, None]).sum(axis=1)

    a = np.concatenate([fold(g[:, :nh]), fold(g[:, nh:])], axis=1)
    rhs = _log_potential(colloc, src.z1, src.z2) @ src.k / (2 * math.pi)
    sol = np.linalg.solve(a, rhs)
    hub_k, cup_k = (np.outer(signs, part).ravel() for part in (sol[:per], sol[per:]))   # back to all pitches
    return Sheets(z1, z2, np.concatenate([hub_k, cup_k]))


# --------------------------------------------------------------------------- field and torque
def b_field(z, magnets: list[Magnet], iron: IronCircles | IronPolygons | None = None,
            cup_phase: float = 0.0, antiperiod: int | None = None) -> np.ndarray:
    """B_x + i B_y [T] at points z (complex, metres), in air, for the magnets and optional steel.

    Each magnet's bound currents sum to zero, so the centre images an infinitely
    permeable circle also needs cancel and are omitted. Images are only valid for sources
    in the air between the circles, so a current outside that annulus is refused.
    ``antiperiod`` is passed to ``polygon_iron_charges`` (polygonal steel only)."""
    z = np.atleast_1d(np.asarray(z, dtype=complex))
    sh = current_sheets(magnets)
    b = b_from_currents(z, sh)
    if isinstance(iron, IronCircles):
        radii = np.abs(np.concatenate([sh.z1, sh.z2]))
        if radii.min() < iron.r_inner_mm * MM * (1 - 1e-9) or radii.max() > iron.r_outer_mm * MM * (1 + 1e-9):
            raise ValueError(f"bound currents span r = {radii.min() / MM:.4f}..{radii.max() / MM:.4f} mm, outside the "
                             f"air between the steel circles ({iron.r_inner_mm}..{iron.r_outer_mm} mm)")
        for kind, value in image_maps(iron):
            b = b + (b_from_currents(z, sh, value) if kind == "scale" else b_from_inverted_currents(z, sh, value))
    elif isinstance(iron, IronPolygons):
        b = b + b_from_charges(z, polygon_iron_charges(magnets, iron, cup_phase, antiperiod))
    return b


def maxwell_torque_per_length(magnets: list[Magnet], r_mm: float, n_poles: int,
                              iron: IronCircles | IronPolygons | None = None, cup_phase: float = 0.0,
                              n_per_pitch: int = 256) -> float:
    """Torque per unit length [N·m/m] on everything inside the circle of radius r_mm (the
    inner rotor), from the Maxwell stress: T/L = (r^2 / mu0) * integral of B_r B_theta dtheta.

    The integrand repeats every pole pitch (turning the whole section by one pitch
    flips every source, so B -> -B), so one pitch is sampled with the midpoint rule,
    which is spectrally accurate for a smooth periodic integrand. That needs magnets that
    flip sign under a turn of 2 pi / n_poles and steel with n_poles-fold symmetry; the same
    property lets polygonal steel be solved over one pole pitch (``antiperiod``)."""
    if isinstance(iron, IronPolygons) and iron.n_sides % n_poles:
        raise ValueError(f"steel polygon with {iron.n_sides} sides lacks the {n_poles}-fold symmetry the stress integral uses")
    r = r_mm * MM
    theta = 2 * math.pi / n_poles * (np.arange(n_per_pitch) + 0.5) / n_per_pitch
    b = b_field(r * np.exp(1j * theta), magnets, iron, cup_phase, antiperiod=n_poles) * np.exp(-1j * theta)
    return float(2 * math.pi * r ** 2 / MU0_EXACT * np.mean(b.real * b.imag))


def rect_block(back_apothem_m: float, thickness_m: float, width_m: float, angle: float, br_signed: float) -> Magnet:
    """Flat block with its back face on the polygon face at ``angle``, magnetized along the
    face normal (outward for br_signed > 0)."""
    rot = complex(math.cos(angle), math.sin(angle))
    a, t, h = back_apothem_m, thickness_m, width_m / 2
    local = (complex(a, -h), complex(a + t, -h), complex(a + t, h), complex(a, h))
    return Magnet(tuple(p * rot for p in local), br_signed * rot)


@dataclass(frozen=True)
class CouplingSection:
    """2D section: n_poles flat blocks per ring on polygon faces, alternating magnetization,
    block 0 of each ring magnetized outward (aligned rings attract). Lengths in mm, Br in T."""
    n_poles: int
    inner_back_mm: float
    inner_thickness_mm: float
    inner_width_mm: float
    outer_face_mm: float
    outer_thickness_mm: float
    outer_width_mm: float
    br_inner_T: float
    br_outer_T: float

    def _ring(self, back_mm: float, thickness_mm: float, width_mm: float, br: float, phase: float) -> list[Magnet]:
        return [rect_block(back_mm * MM, thickness_mm * MM, width_mm * MM, phase + 2 * math.pi * i / self.n_poles,
                           br if i % 2 == 0 else -br) for i in range(self.n_poles)]

    def inner(self) -> list[Magnet]:
        return self._ring(self.inner_back_mm, self.inner_thickness_mm, self.inner_width_mm, self.br_inner_T, 0.0)

    def outer(self, delta_mech: float) -> list[Magnet]:
        return self._ring(self.outer_face_mm, self.outer_thickness_mm, self.outer_width_mm, self.br_outer_T, delta_mech)

    def inner_corner_mm(self) -> float:
        """Largest radius of the inner blocks (front corners)."""
        return math.hypot(self.inner_back_mm + self.inner_thickness_mm, self.inner_width_mm / 2)

    def outer_back_corner_mm(self) -> float:
        """Largest radius of the outer blocks (back corners)."""
        return math.hypot(self.outer_face_mm + self.outer_thickness_mm, self.outer_width_mm / 2)

    def stress_radius_mm(self) -> float:
        """Middle of the clear air annulus between the inner front corners and the outer faces."""
        if self.inner_corner_mm() >= self.outer_face_mm:
            raise ValueError("inner block corners reach the outer block faces: no clear gap for the stress circle")
        return 0.5 * (self.inner_corner_mm() + self.outer_face_mm)


def torque_per_length(sec: CouplingSection, theta_e: float, iron: IronCircles | IronPolygons | None = None,
                      r_mm: float | None = None) -> float:
    """Torque per unit length [N·m/m] on the inner rotor with the outer ring (and cup) turned
    by the electrical angle theta_e (mechanical theta_e / (N/2)). Positive = restoring."""
    r = sec.stress_radius_mm() if r_mm is None else r_mm
    if not sec.inner_corner_mm() < r < sec.outer_face_mm:
        raise ValueError(f"stress circle r = {r} mm is not in the clear gap "
                         f"({sec.inner_corner_mm():.4f}..{sec.outer_face_mm} mm)")
    delta = theta_e / (sec.n_poles / 2)
    return maxwell_torque_per_length(sec.inner() + sec.outer(delta), r, sec.n_poles, iron, cup_phase=delta)


@dataclass(frozen=True)
class Pullout:
    torque_per_m: float        # true pull-out: max over angle [N·m/m]
    theta_e_peak: float        # electrical angle of that max [rad]
    at_quarter: float          # torque at theta_e = pi/2, where the engine evaluates [N·m/m]
    harmonics: dict            # {n: a_n}, T(theta_e) = sum over n >= 1 of a_n sin(n theta_e) [N·m/m]


def pullout(sec: CouplingSection, iron: IronCircles | IronPolygons | None = None, n_samples: int = 48) -> Pullout:
    """Torque-angle curve over one pole pair, its sine harmonics, and the true pull-out.

    Every configuration here is mirror-symmetric, so T(-theta) = -T(theta) and the curve is
    a pure sine series; midpoint samples on (0, pi) give a_n for n < n_samples exactly
    (discrete sine transform), apart from aliasing of harmonics that fall off like
    exp(-n k1 g). Circles or free space leave only odd n; polygonal steel adds cogging
    between each ring and the other rotor's flats, which appears as even n. The max is
    refined with a bounded scalar search between the neighbours of the best sample."""
    th = math.pi * (np.arange(n_samples) + 0.5) / n_samples
    t = np.array([torque_per_length(sec, x, iron) for x in th])
    harm = {n: float(2 / n_samples * np.sum(t * np.sin(n * th))) for n in range(1, n_samples)}
    i = int(np.argmax(t))
    lo, hi = th[max(i - 1, 0)], th[min(i + 1, n_samples - 1)]
    best = minimize_scalar(lambda x: -torque_per_length(sec, x, iron), bounds=(lo, hi), method="bounded",
                           options={"xatol": 1e-7})
    return Pullout(float(-best.fun), float(best.x), torque_per_length(sec, math.pi / 2, iron), harm)


# --------------------------------------------------------------------------- cylindrical harmonics
def cylindrical_s_factor(m: int, r1a: float, r1b: float, r2a: float, r2b: float, r_gap: float,
                         iron: tuple[float, float] | None = None) -> float:
    """Exact 2D geometry factor for order-m sinusoidal RADIAL magnetization in the annuli
    r1a..r1b (inner ring) and r2a..r2b (outer ring), normalised like the engine:
    torque amplitude per length = B_i B_o / (2 mu0) * S * 2 pi r_gap^2. Radii in one unit.

    Free space (``iron=None``): a ring charge s cos(m theta) has potential
    (s / 2m) (r</r>)^m cos(m theta) (the free-space subdomain solution for radial
    magnetization, Zhu & Howe, IEEE Trans. Magn. 29(1), 1993). Integrating both rings'
    surface (M.n) and volume (-div M = -M cos(m theta) / r) charges gives, with x = r / r_gap,
        S = m^2 / (2 (m^2 - 1)) * (x1b^(m+1) - x1a^(m+1)) * (x2a^(1-m) - x2b^(1-m)).
    Steel (``iron=(rho1, rho2)``: infinitely permeable for r <= rho1 and r >= rho2): the
    potential is constant on both circles, the order-m Green's function is
    u1(r<) u2(r>) / W, u1 = (r/rho1)^m - (rho1/r)^m, u2 = (rho2/r)^m - (r/rho2)^m,
    W = 2m (Q - 1/Q), Q = (rho2/rho1)^m, and S = m p_i p_o / W with
    p = [r u(r)] across the ring minus the integral of u across the ring."""
    if m < 2:
        raise ValueError("order m must be >= 2 (m = 1 needs the logarithmic solution)")
    x1a, x1b, x2a, x2b = (r / r_gap for r in (r1a, r1b, r2a, r2b))
    if not x1a < x1b < x2a < x2b:
        raise ValueError("need r1a < r1b < r2a < r2b")
    if iron is None:
        return m * m / (2 * (m * m - 1)) * (x1b ** (m + 1) - x1a ** (m + 1)) * (x2a ** (1 - m) - x2b ** (1 - m))
    rho1, rho2 = iron[0] / r_gap, iron[1] / r_gap
    if rho1 > x1a * (1 + 1e-12) or rho2 < x2b * (1 - 1e-12):
        raise ValueError("steel must lie outside the magnet annuli: rho1 <= r1a and rho2 >= r2b")

    def u1(s):
        return (s / rho1) ** m - (rho1 / s) ** m

    def u1_int(s):
        return s ** (m + 1) / ((m + 1) * rho1 ** m) + rho1 ** m * s ** (1 - m) / (m - 1)

    def u2(r):
        return (rho2 / r) ** m - (r / rho2) ** m

    def u2_int(r):
        return rho2 ** m * r ** (1 - m) / (1 - m) - r ** (m + 1) / ((m + 1) * rho2 ** m)

    p_i = x1b * u1(x1b) - x1a * u1(x1a) - (u1_int(x1b) - u1_int(x1a))
    p_o = x2b * u2(x2b) - x2a * u2(x2a) - (u2_int(x2b) - u2_int(x2a))
    q = (rho2 / rho1) ** m
    return abs(m * p_i * p_o / (2 * m * (q - 1 / q)))
```

- [ ] **Step 2: Write the reference's own sanity test**

This file never runs the engine.

File: `reference/magcoupling-py/audit/tests/test_torque2d_reference_sanity.py` (create)

```python
"""M1 Task 2: sanity checks of the 2D field reference (``field2d``) and the planar formulas (``planar``).

These tests never run the engine. Each one checks a reference against a textbook result with a
known answer, against a second independent method, or against its own convergence behaviour, so
the engine checks in ``test_torque_field2d.py`` compare the engine with a trustworthy reference.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from audit.common import MU0_EXACT, mismatch
from audit.references.field2d import (
    MM, CouplingSection, IronCircles, IronPolygons, Magnet, Sheets, b_field, b_from_charges, b_from_currents,
    b_from_inverted_currents, charge_sheets, current_sheets, cylindrical_s_factor, maxwell_torque_per_length,
    polygon_iron_charges, torque_per_length,
)
from audit.references.planar import planar_s_factor, square_wave_harmonic

#: The default section written out (10 poles, B842SH 6.35 x 3.17 mm, inner backs at 10.15 mm,
#: outer faces at 14.72 mm, outer backs at 17.89 mm), so these tests need no engine.
SANITY_SECTION = CouplingSection(10, 10.15, 3.17, 6.35, 14.72, 3.17, 6.35, 1.29, 1.29)
SANITY_CUP_MM = 17.89
#: The same section at a 0.5 mm face gap (outer faces at 13.82 mm): the inner block corners
#: (13.693 mm) clear the outer faces by only 0.127 mm, the hardest case for the stress quadrature.
SMALL_GAP_SECTION = CouplingSection(10, 10.15, 3.17, 6.35, 13.82, 3.17, 6.35, 1.29, 1.29)


def sliced_radial_ring(r_a_mm: float, r_b_mm: float, m: int, b_T: float, phase: float, slices: int) -> list[Magnet]:
    """Annulus of radially magnetized polygon slices, J = b_T cos(m theta - phase) along each
    slice's centre radius: a polygon stand-in for sinusoidal radial magnetization."""
    out = []
    for j in range(slices):
        c, h = 2 * math.pi * (j + 0.5) / slices, math.pi / slices
        lo, hi = np.exp(1j * (c - h)), np.exp(1j * (c + h))
        verts = (r_a_mm * MM * lo, r_b_mm * MM * lo, r_b_mm * MM * hi, r_a_mm * MM * hi)
        out.append(Magnet(tuple(complex(v) for v in verts), b_T * math.cos(m * c - phase) * np.exp(1j * c)))
    return out


# --------------------------------------------------------------------------- field2d: sources
@pytest.mark.family("torque")
def test_sanity_sheet_fields_match_quadrature():
    """Closed-form sheet fields (direct current, inverted-image current, charge) equal direct
    numerical integration of the line-source field over the sheet (scipy quad, 1e-13)."""
    sh = Sheets(np.array([0.001 + 0.002j]), np.array([0.004 + 0.0035j]), np.array([0.7]))
    pts = np.array([0.003 + 0.001j, -0.002 + 0.005j, 0.0025 + 0.0027j])
    rho = 0.0025

    def integrate(source_at, line_field):
        d, out = sh.z2[0] - sh.z1[0], []
        for p in pts:
            def f(t, part):
                v = line_field(p - source_at(sh.z1[0] + t * d)) * abs(d) * sh.k[0]
                return v.real if part == 0 else v.imag
            out.append(quad(f, 0, 1, args=(0,), epsabs=0, epsrel=1e-13, limit=200)[0]
                       + 1j * quad(f, 0, 1, args=(1,), epsabs=0, epsrel=1e-13, limit=200)[0])
        return np.array(out)

    current = lambda r: 1j / (2 * math.pi) * r / abs(r) ** 2    # line current, mu0 I = 1 along +z
    charge = lambda r: 1 / (2 * math.pi) * r / abs(r) ** 2      # line charge, mu0 q = 1
    for name, closed, numeric in (
            ("direct current", b_from_currents(pts, sh), integrate(lambda z: z, current)),
            ("inverted current", b_from_inverted_currents(pts, sh, rho), integrate(lambda z: rho ** 2 / np.conj(z), current)),
            ("charge", b_from_charges(pts, sh), integrate(lambda z: z, charge))):
        err = float(np.max(np.abs(closed - numeric)) / np.max(np.abs(numeric)))
        assert err < 1e-10, mismatch(f"{name} sheet field vs quadrature", "reference only", err, 0.0, 1e-10)


@pytest.mark.family("torque")
def test_sanity_bar_on_axis_textbook():
    """Infinitely long bar, width w, height t, magnetized across its height: on the axis above
    it B_y = (Br/pi) [atan(w / 2(y - t)) - atan(w / 2y)], B_x = 0 (the field of two
    uniformly charged strips, sigma = +/-Br/mu0; e.g. Furlani, Permanent Magnet and
    Electromechanical Devices, 2001, ch. 4)."""
    br, w, t = 1.3, 6e-3, 3e-3
    bar = Magnet((complex(-w / 2, 0), complex(w / 2, 0), complex(w / 2, t), complex(-w / 2, t)), 1j * br)
    ys = np.array([3.5e-3, 5e-3, 1e-2, 3e-2])
    b = b_field(1j * ys, [bar])
    expect = br / math.pi * (np.arctan(w / (2 * (ys - t))) - np.arctan(w / (2 * ys)))
    for got, ref in zip(b, expect):
        assert abs(got.imag - ref) <= 1e-12 * abs(ref), mismatch("bar on-axis B_y", "reference only", got.imag, ref, 1e-12)
        assert abs(got.real) <= 1e-12 * abs(ref), mismatch("bar on-axis B_x", "reference only", got.real, 0.0, 1e-12)


@pytest.mark.family("torque")
def test_sanity_polygon_centre_field_is_half_polarization():
    """At the centre of a uniformly magnetized regular polygon (n >= 3) the 2D demagnetizing
    tensor is isotropic with trace 1, so H = -M/2 and B = J/2 exactly: the transverse
    demagnetizing factor 1/2 of an infinite cylinder (Osborn, Phys. Rev. 67, 1945)."""
    j = 1.2 * np.exp(0.7j)
    for n in (3, 6, 11):
        poly = Magnet(tuple(complex(2e-3 * np.exp(2j * math.pi * k / n)) for k in range(n)), j)
        got = b_field(np.array([0j]), [poly])[0]
        assert abs(got - j / 2) <= 1e-12, mismatch(f"centre field of a {n}-gon", "reference only", abs(got), abs(j / 2), 1e-12)


@pytest.mark.family("torque")
def test_sanity_charge_and_current_models_agree():
    """In air the magnet field from surface charges (used by the BEM) equals the field from
    bound surface currents (used everywhere else), at 64 points around the default gap."""
    sec = SANITY_SECTION
    mags = sec.inner() + sec.outer(0.1)
    z = sec.stress_radius_mm() * MM * np.exp(2j * math.pi * (np.arange(64) + 0.3) / 64)
    bc, bq = b_from_currents(z, current_sheets(mags)), b_from_charges(z, charge_sheets(mags))
    err = float(np.max(np.abs(bc - bq)) / np.max(np.abs(bc)))
    assert err < 1e-12, mismatch("charge vs current field in the gap", "reference only", err, 0.0, 1e-12)


# --------------------------------------------------------------------------- field2d: steel
@pytest.mark.family("torque")
def test_sanity_images_make_field_normal_to_steel():
    """Infinitely permeable steel: B has no tangential part on its surface. With 8 image
    generations |B_t| / max|B| < 1e-9 on both circles; without images it is order 1."""
    sec = SANITY_SECTION
    mags = sec.inner() + sec.outer(0.1)
    iron = IronCircles(10.0, 18.5, 8)
    theta = 2 * math.pi * (np.arange(97) + 0.37) / 97
    for r_mm in (iron.r_inner_mm, iron.r_outer_mm):
        z = r_mm * MM * np.exp(1j * theta)
        for images, bound in ((iron, 1e-9), (None, None)):
            b = b_field(z, mags, images) * np.exp(-1j * theta)
            ratio = float(np.max(np.abs(b.imag)) / np.max(np.abs(b)))
            if bound is None:
                assert ratio > 0.1, mismatch(f"tangential B without images at r={r_mm}", "reference only", ratio, 0.1, 0.1)
            else:
                assert ratio < bound, mismatch(f"tangential B on steel at r={r_mm}", "reference only", ratio, 0.0, bound)


@pytest.mark.family("torque")
def test_sanity_image_series_converged():
    """Truncating the image series at 8 generations changes the torque by < 1e-10 relative
    compared with 12: each generation contributes about (r_inner / r_outer)^(N/2) = 0.055,
    and 0.055^8 is about 8e-11.

    Bound history: the first draft asserted 1e-12, which contradicts this estimate; the bound
    was corrected to match it. The reference output did not change (observed 3.3e-11)."""
    sec = SANITY_SECTION
    r1, r2 = sec.inner_back_mm, sec.outer_back_corner_mm()
    t8 = torque_per_length(sec, math.pi / 2, IronCircles(r1, r2, 8))
    t12 = torque_per_length(sec, math.pi / 2, IronCircles(r1, r2, 12))
    assert abs(t8 / t12 - 1) < 1e-10, mismatch("image series 8 vs 12 generations", "reference only", t8, t12, 1e-10)


@pytest.mark.family("torque")
def test_sanity_bem_matches_images_on_circular_steel():
    """Two independent steel methods agree: the BEM on 200-gons (hub inscribed in r1, cup
    circumscribed about r2) against the exact image series on the circles. The polygons hold
    slightly less steel; that error is O((pi/200)^2), about 2e-4, so the bound is 1e-3."""
    sec = SANITY_SECTION
    r1, r2 = sec.inner_back_mm, sec.outer_back_corner_mm()
    t_img = torque_per_length(sec, math.pi / 2, IronCircles(r1, r2))
    t_bem = torque_per_length(sec, math.pi / 2, IronPolygons(r1 * math.cos(math.pi / 200), r2, 200, 4))
    assert abs(t_bem / t_img - 1) < 1e-3, mismatch("BEM 200-gon vs image circles", "reference only", t_bem, t_img, 1e-3)


@pytest.mark.family("torque")
def test_sanity_bem_antiperiodic_matches_full():
    """The one-pitch BEM (``antiperiod`` = N, used by the stress integral) and the full-boundary BEM
    give the same gap field, for 10-gon steel at the magnet backs and with bondlines and for 200-gon
    steel, at two angles: 1e-10 relative (the discrete systems are equivalent; only rounding differs)."""
    sec = SANITY_SECTION
    z = sec.stress_radius_mm() * MM * np.exp(2j * math.pi * (np.arange(64) + 0.3) / 64)
    for iron in (IronPolygons(10.15, SANITY_CUP_MM, 10), IronPolygons(10.10, 17.94, 10),
                 IronPolygons(10.15 * math.cos(math.pi / 200), sec.outer_back_corner_mm(), 200, 4)):
        for delta in (0.06, 0.22):
            mags = sec.inner() + sec.outer(delta)
            full = b_from_charges(z, polygon_iron_charges(mags, iron, delta))
            reduced = b_from_charges(z, polygon_iron_charges(mags, iron, delta, antiperiod=sec.n_poles))
            err = float(np.max(np.abs(reduced - full)) / np.max(np.abs(full)))
            assert err < 1e-10, mismatch(f"one-pitch vs full BEM, {iron.n_sides}-gon, delta={delta}",
                                         "reference only", err, 0.0, 1e-10)


@pytest.mark.parametrize("hub_mm, cup_mm", [(10.15, 17.89), (10.10, 17.94)], ids=["at-magnet-backs", "with-bondlines"])
@pytest.mark.family("torque")
def test_sanity_bem_converged(hub_mm, cup_mm):
    """BEM discretisation error on the real flat-face steel, for both steel placements the engine
    checks use (steel touching the magnet backs, and 0.05 mm bondlines): 40 vs 80 panels per side
    agree to < 1e-3 relative, the error bound quoted for the steel reference."""
    sec = SANITY_SECTION
    t40 = torque_per_length(sec, math.pi / 2, IronPolygons(hub_mm, cup_mm, 10, 40))
    t80 = torque_per_length(sec, math.pi / 2, IronPolygons(hub_mm, cup_mm, 10, 80))
    assert abs(t40 / t80 - 1) < 1e-3, mismatch("BEM 40 vs 80 panels per side", "reference only", t40, t80, 1e-3)


# --------------------------------------------------------------------------- field2d: stress integral
@pytest.mark.family("torque")
def test_sanity_maxwell_stress_independent_of_radius():
    """The Maxwell stress is divergence-free in source-free air, so the torque is the same on
    any circle in the clear gap (free space, circular steel and polygonal steel, 1e-9)."""
    sec = SANITY_SECTION
    lo, hi = sec.inner_corner_mm(), sec.outer_face_mm
    for iron in (None, IronCircles(sec.inner_back_mm, sec.outer_back_corner_mm()),
                 IronPolygons(sec.inner_back_mm, SANITY_CUP_MM, 10)):
        ta = torque_per_length(sec, 1.1, iron, r_mm=lo + 0.25 * (hi - lo))
        tb = torque_per_length(sec, 1.1, iron, r_mm=lo + 0.75 * (hi - lo))
        assert abs(ta / tb - 1) < 1e-9, mismatch(f"Maxwell torque at two gap radii, {type(iron).__name__}",
                                                 "reference only", ta, tb, 1e-9)


@pytest.mark.family("torque")
def test_sanity_stress_quadrature_converged_at_small_gap():
    """At the 0.5 mm face gap the stress circle passes 0.064 mm from the inner block corners, where
    the field varies fastest. The midpoint rule is spectrally accurate for this periodic integrand,
    with an error of at most about exp(-2 pi d / h) (d the corner distance, h the sample spacing):
    that estimate is 1e-5 for 256 samples per pitch, so 256 against 2048 must agree to 1e-4."""
    sec = SMALL_GAP_SECTION
    mags = sec.inner() + sec.outer(math.pi / 2 / (sec.n_poles / 2))
    t256 = maxwell_torque_per_length(mags, sec.stress_radius_mm(), sec.n_poles, n_per_pitch=256)
    t2048 = maxwell_torque_per_length(mags, sec.stress_radius_mm(), sec.n_poles, n_per_pitch=2048)
    assert abs(t256 / t2048 - 1) < 1e-4, mismatch("stress quadrature 256 vs 2048 samples per pitch, 0.5 mm gap",
                                                  "reference only", t256, t2048, 1e-4)


# --------------------------------------------------------------------------- cylindrical harmonics and planar formulas
@pytest.mark.family("torque")
def test_sanity_square_wave_harmonic_matches_numeric_fourier():
    """square_wave_harmonic equals the Fourier coefficient of the alternating pulse train computed
    by quadrature over one pole pair (+B for |x| < fill tau/2, -B for |x - tau| < fill tau/2):
    a_n = (1/tau) * integral of f(x) cos(n pi x / tau) dx, for n = 1..6 (even n vanish), 1e-12."""
    tau, b = 1.0, 1.3
    for fill in (0.3, 0.72, 1.0):
        h = fill * tau / 2
        for n in range(1, 7):
            c = lambda x: math.cos(n * math.pi * x / tau)
            numeric = b / tau * (quad(c, -h, h, epsabs=1e-14)[0] - quad(c, tau - h, tau + h, epsabs=1e-14)[0])
            closed = square_wave_harmonic(b, n, fill)
            assert abs(closed - numeric) <= 1e-12, mismatch(f"square-wave harmonic n={n}, fill={fill}",
                                                            "reference only", closed, numeric, 1e-12)
    with pytest.raises(ValueError, match="must be >= 1"):
        square_wave_harmonic(1.0, 0, 0.5)


@pytest.mark.family("torque")
def test_sanity_planar_s_factor_thick_magnet_limit():
    """Thick magnets make the backing irrelevant: for k t -> infinity both planar factors tend to
    e^-k g / 2 (one semi-infinite array faces another). At k t = 20 the remaining difference is
    of order e^-2kt, so 1e-8 relative."""
    k, g = 357.0, 1.4e-3
    t = 20 / k
    limit = math.exp(-k * g) / 2
    for backed in (False, True):
        s = planar_s_factor(k, t, t, g, backed)
        assert abs(s / limit - 1) < 1e-8, mismatch(f"thick-magnet limit, backed={backed}", "reference only", s, limit, 1e-8)


@pytest.mark.family("torque")
def test_sanity_cylindrical_s_factor_planar_limit():
    """As R_gap grows at fixed k (m = k R_gap) the exact cylindrical factor converges to the planar
    ones in ``planar_s_factor`` (free space incl. the /2, and the steel sinh form). With t_i != t_o
    the residual is a first-order curvature term: it halves when R_gap doubles (ratio 2 +/- 1 %)
    and stays below dr / R_gap, dr = t_i + g + t_o. This one test validates both references.

    Bound history: the first draft asserted a fixed residual bound whose error order was wrong
    (it assumed second-order convergence). It now asserts the first-order rate, which is
    stronger; error x R_gap is constant from 5 m to 80 m. The references did not change."""
    k, ti, to, g = 357.0, 3.17e-3, 2.5e-3, 1.4e-3
    planar = {"free": planar_s_factor(k, ti, to, g, backed=False), "steel": planar_s_factor(k, ti, to, g, backed=True)}

    def residuals(m: int) -> dict:
        rg = m / k
        radii = (rg - g / 2 - ti, rg - g / 2, rg + g / 2, rg + g / 2 + to)
        return {"free": cylindrical_s_factor(m, *radii, rg) / planar["free"] - 1,
                "steel": cylindrical_s_factor(m, *radii, rg, iron=(radii[0], radii[3])) / planar["steel"] - 1}

    e20, e40 = residuals(7140), residuals(14280)           # R_gap = 20 m and 40 m
    bound = (ti + g + to) / (14280 / k)
    for name in planar:
        assert abs(e20[name] / e40[name] - 2) < 0.02, mismatch(
            f"cylindrical S planar residual 20 m / 40 m, {name}", "reference only", e20[name] / e40[name], 2.0, 0.01)
        assert abs(e40[name]) < bound, mismatch(
            f"cylindrical S planar residual at 40 m, {name}", "reference only", e40[name], 0.0, bound)


@pytest.mark.parametrize("steel", [False, True], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_sanity_cylindrical_s_factor_matches_sliced_rings(steel):
    """Two independent solutions of the curved problem agree: the analytic cylindrical harmonic
    factor against Maxwell-stress torque on rings of K radially magnetized polygon slices
    (m = 5, unequal ring thicknesses so an inner/outer mix-up would show, steel as image
    circles at the ring backs). The slicing error is second order in 1/K, so going from
    K = 720 to 360 must quadruple it (ratio 4 +/- 2 %), which shows the analytic value is the
    exact limit; at K = 720 the error must also be below 1e-3.

    Bound history: the first draft asserted only a fixed bound tighter than the K = 720 slicing
    error. It now asserts the second-order rate, which is stronger; error x K^2 is -92.1 from
    K = 180 to 1440. The references did not change."""
    m_ord = 5
    r1a, r1b, r2a, r2b, rg = 10.15, 13.32, 14.72, 17.0, 14.02
    iron = IronCircles(r1a, r2b) if steel else None
    s = cylindrical_s_factor(m_ord, r1a, r1b, r2a, r2b, rg, iron=(r1a, r2b) if steel else None)
    t_analytic = 1.0 / (2 * MU0_EXACT) * s * 2 * math.pi * (rg * MM) ** 2

    def error(slices: int) -> float:
        mags = (sliced_radial_ring(r1a, r1b, m_ord, 1.0, 0.0, slices)
                + sliced_radial_ring(r2a, r2b, m_ord, 1.0, math.pi / 2, slices))
        return maxwell_torque_per_length(mags, 0.5 * (r1b + r2a), 2 * m_ord, iron) / t_analytic - 1

    e360, e720 = error(360), error(720)
    assert abs(e720) < 1e-3, mismatch("sliced rings (K=720) vs cylindrical harmonic torque", "reference only",
                                      e720, 0.0, 1e-3)
    assert abs(e360 / e720 / 4 - 1) < 0.02, mismatch("sliced-ring error ratio K=360 / K=720", "reference only",
                                                     e360 / e720, 4.0, 0.02)


# --------------------------------------------------------------------------- error paths
@pytest.mark.family("torque")
def test_sanity_reference_refuses_invalid_geometry():
    """The reference refuses set-ups where its method is invalid instead of returning a number:
    a steel circle through the outer blocks' back faces (their corners at 18.17 mm lie beyond
    17.89 mm), polygon steel that cuts into a magnet (hub above the block backs, or cup inside
    the outer backs), steel without the pole symmetry, a one-pitch BEM whose anti-period is odd
    or does not divide the steel sides, a stress circle outside the clear gap, inner corners that
    reach the outer faces (0.3 mm face gap), and cylindrical factors with m < 2, unordered radii or
    steel inside a magnet ring."""
    sec = SANITY_SECTION
    with pytest.raises(ValueError, match="outside the air between the steel circles"):
        torque_per_length(sec, math.pi / 2, IronCircles(sec.inner_back_mm, SANITY_CUP_MM))
    with pytest.raises(ValueError, match="enters the steel polygons"):
        torque_per_length(sec, math.pi / 2, IronPolygons(sec.inner_back_mm + 0.05, SANITY_CUP_MM, 10))
    with pytest.raises(ValueError, match="enters the steel polygons"):
        torque_per_length(sec, math.pi / 2, IronPolygons(sec.inner_back_mm, SANITY_CUP_MM - 0.05, 10))
    with pytest.raises(ValueError, match="symmetry"):
        torque_per_length(sec, math.pi / 2, IronPolygons(sec.inner_back_mm, SANITY_CUP_MM, 15))
    for bad in (4, 5):
        with pytest.raises(ValueError, match="anti-period"):
            polygon_iron_charges(sec.inner() + sec.outer(0.0), IronPolygons(sec.inner_back_mm, SANITY_CUP_MM, 10),
                                 0.0, antiperiod=bad)
    with pytest.raises(ValueError, match="clear gap"):
        torque_per_length(sec, math.pi / 2, None, r_mm=13.5)
    with pytest.raises(ValueError, match="no clear gap"):
        CouplingSection(10, 10.15, 3.17, 6.35, 13.62, 3.17, 6.35, 1.29, 1.29).stress_radius_mm()
    with pytest.raises(ValueError, match="m must be >= 2"):
        cylindrical_s_factor(1, 10.0, 13.0, 14.0, 17.0, 13.5)
    with pytest.raises(ValueError, match="r1a < r1b < r2a < r2b"):
        cylindrical_s_factor(5, 10.0, 14.5, 14.0, 17.0, 13.5)
    with pytest.raises(ValueError, match="outside the magnet annuli"):
        cylindrical_s_factor(5, 10.0, 13.0, 14.0, 17.0, 13.5, iron=(10.5, 17.0))
```

- [ ] **Step 3: Run the sanity test**

Run:
```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_torque2d_reference_sanity.py -k sanity
```
Expected: PASS, `18 passed` in about 3 s.

- [ ] **Step 4: Write the engine checks**

The engine checks import `TOL_MODEL` (2 %, the materiality threshold for every engine-model-vs-exact comparison), `TOL_PEAK` (1e-3, the peak-angle resolution bound) and `ScenarioCache` (the cached engine runs behind `engine_case`) from `audit/common.py`, where Task 1 defines them together with the other shared helpers. Do not edit `common.py` in this task, and do not define either tolerance in the test file.

File: `reference/magcoupling-py/audit/tests/test_torque_field2d.py` (create)

```python
"""M1 Task 2: the Calculator torque model against independent references.

Groups:
  * chain  - Calculator!C66:C94 re-derived term by term from the textbook planar formulas
             (``audit.references.planar``), same algebra, TOL_ALGEBRA; the sole same-algebra owner
             of those cells.
  * 2D     - the engine's 2D pull-out, torque-angle harmonics and quarter-pitch evaluation against
             the exact 2D field of the real flat-block section (``audit.references.field2d``), in
             free space and with polygonal steel, across face gaps 0.5, 1.4 and 3.0 mm, TOL_MODEL.
  * S_n    - the planar (unrolled) geometry factors against the exact cylindrical solution, and
             their planar limit.

The reference sanity tests live in ``test_torque2d_reference_sanity.py``. A failing check is a
candidate finding for the report (Task 8); the engine is never edited.
"""
from __future__ import annotations

import functools
import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from audit.common import TOL_ALGEBRA, TOL_MODEL, TOL_PEAK, ScenarioCache, defaults, mismatch, rel_err, run, vary
from audit.references.field2d import MM, CouplingSection, IronPolygons, cylindrical_s_factor, pullout
from audit.references.planar import planar_s_factor, square_wave_harmonic

#: Planar limit at R_gap = 28 m with t_i = t_o and R_gap mid-gap: the first-order curvature
#: terms cancel between the rings, so the residual is O((dr / R_gap)^2) = (4.3 mm / 28 m)^2 ~ 2e-8.
TOL_PLANAR_LIMIT = 1e-6

CIRCUITS = {0: "free space", 1: "steel"}
#: Face gaps (Metal design!C119). The inner block corners stand 0.373 mm proud of the face radius, so a
#: flat-block section exists only above that (0.3 mm would overlap the blocks: corner gap C9 = -0.07 mm).
#: 0.5 mm leaves a 0.127 mm corner gap, the smallest buildable case; 1.4 mm is the default; 3.0 mm is wide.
FACE_GAPS_MM = (0.5, 1.4, 3.0)
DEFAULT_GAP_MM = 1.4
#: Steel placement that decides the steel verdicts (see ``steel_from``); ignored in free space.
VERDICT_STEEL = "as-built"
#: Calculator rows per harmonic: k_n, B_in, B_on, S_n steel, S_n free, tau_n.
HARMONIC_ROWS = {1: ("C71", "C72", "C73", "C74", "C75", "C76"),
                 3: ("C77", "C78", "C79", "C80", "C81", "C82"),
                 5: ("C83", "C84", "C85", "C86", "C87", "C88")}
#: Term-by-term cases: both circuits at defaults, plus branches defaults never reach: another pole
#: count at the cold end, the fill clamp min(1, ...) (20 poles: blocks wider than the pitch), and
#: unequal inner/outer magnets (outer thinner, narrower and weaker) at the hot end with steel and at
#: 50 °C in free space with manual magnets in both rings, so an inner/outer mix-up would show in
#: either circuit's S_n branch.
CHAIN_CASES = {
    "steel": {},
    "free": {"coupling.backiron": 0},
    "steel-12-poles-cold": {"coupling.npole": 12, "coupling.op_temp_C": -40},
    "free-20-poles-fill-clamped": {"coupling.backiron": 0, "coupling.npole": 20},
    "steel-unequal-rings-hot": {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_thickness_mm": 2.5,
                                "coupling.magnets.manual_outer_width_mm": 5.0,
                                "coupling.magnets.manual_outer_br_T": 1.2, "coupling.op_temp_C": 100},
    "free-unequal-rings": {"coupling.backiron": 0, "coupling.magnets.part_inner": "",
                           "coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 5.0,
                           "coupling.magnets.manual_outer_thickness_mm": 2.5,
                           "coupling.magnets.manual_outer_br_T": 1.2},
}


# --------------------------------------------------------------------------- helpers
#: One engine run per (circuit, face gap) pair the 2D and S_n checks use.
ENGINE_CASES = ScenarioCache({f"{CIRCUITS[b]}, face gap {g} mm": {"coupling.backiron": b, "metal.face_gap_mm": g}
                              for b in CIRCUITS for g in FACE_GAPS_MM})


def engine_case(backiron: int, face_gap_mm: float):
    """(inputs, Calculator results) at defaults with the back-iron circuit (Calculator!C6) and face gap
    (Metal design!C119) changed. Only the pairs in ENGINE_CASES exist (KeyError otherwise)."""
    inp, res = ENGINE_CASES[f"{CIRCUITS[backiron]}, face gap {face_gap_mm} mm"]
    return inp, res.model


def section_from(m, n_poles: int) -> CouplingSection:
    """Reference geometry from the engine's resolved fields, remanence at 20 °C."""
    return CouplingSection(
        n_poles=n_poles,
        inner_back_mm=m.inner_face_radius_mm - m.inner_thickness_mm,                  # C54 - C20
        inner_thickness_mm=m.inner_thickness_mm, inner_width_mm=m.inner_width_mm,     # C20, C19
        outer_face_mm=m.outer_face_apothem_mm,                                        # C56
        outer_thickness_mm=m.outer_thickness_mm, outer_width_mm=m.outer_width_mm,     # C30, C29
        br_inner_T=m.inner_br_T, br_outer_T=m.outer_br_T)                             # C21, C31


def steel_from(inp, m, placement: str) -> IronPolygons:
    """Polygonal steel, one flat per pole, turning with its ring.
    "as-built" (decides the verdict): the machined hub flats at C8 minus the inner bondline
    (Metal design!C120) and the cup pocket flats at C60 plus the outer bondline (C121), the steel the
    engine's own geometry and mass rows describe. "at-backs": steel touching the magnet backs
    (hub C54 - C20, cup C60), the idealisation inside the engine's S_iron (t_i + t_o + g, no bondline)."""
    if placement == "as-built":
        hub = inp.coupling.inner_back_apothem_mm - inp.metal.bond_inner_mm
        cup = m.outer_back_apothem_mm + inp.metal.bond_outer_mm
    elif placement == "at-backs":
        hub, cup = m.inner_face_radius_mm - m.inner_thickness_mm, m.outer_back_apothem_mm
    else:
        raise ValueError(f"unknown steel placement {placement!r}")
    return IronPolygons(hub, cup, inp.coupling.npole)


@functools.lru_cache(maxsize=None)
def reference_pullout(backiron: int, face_gap_mm: float, placement: str):
    """Exact 2D pull-out of the engine's flat-block section (cached; a polygonal-steel case takes about 0.6 s).
    Always pass ``placement`` positionally so every caller shares one cache entry per case."""
    inp, m = engine_case(backiron, face_gap_mm)
    iron = steel_from(inp, m, placement) if backiron else None
    return pullout(section_from(m, inp.coupling.npole), iron)


def reference_2d_nm(backiron: int, face_gap_mm: float, placement: str) -> float:
    """Reference pull-out per metre times the active length C33 [N·m]."""
    _, m = engine_case(backiron, face_gap_mm)
    return reference_pullout(backiron, face_gap_mm, placement).torque_per_m * m.active_length_mm * MM


def engine_2d_torque_20c(m) -> float:
    """Engine pull-out at 20 °C before the end-effect and calibration factors: C94 / (C92 * C42)."""
    return m.pullout_20C_Nm / (m.f_end * m.f_cal)


def engine_harmonic_torque_20c(m, n: int) -> float:
    """Engine torque-angle amplitude of harmonic n at 20 °C: tau_n / sin(n pi/2) * C90, times
    (C21 * C31) / (C69 * C70) to take Br from the operating temperature back to 20 °C."""
    tau = {1: m.tau1_Pa, 3: m.tau3_Pa, 5: m.tau5_Pa}[n]
    return (tau / math.sin(n * math.pi / 2) * m.area_lever_m3
            * (m.inner_br_T * m.outer_br_T) / (m.br_inner_T_op * m.br_outer_T_op))


@functools.lru_cache(maxsize=None)
def prototype_ratio() -> float:
    """Engine / exact 2D torque of the Calibration sheet's no-iron prototype: Calibration!C43 against the
    flat-block section (20 x B842SH on a 9.85 mm apothem, flat gap C32, Br at the test temperature C36)
    times the magnet length C18."""
    inp = defaults()
    c, ci = run(inp).calibration, inp.calibration
    sec = CouplingSection(int(c.poles_per_ring), c.face_radius_mm - ci.magnet_thickness_mm,       # C13, C28 - C20
                          ci.magnet_thickness_mm, ci.magnet_width_mm, c.outer_face_apothem_mm,   # C20, C19, C31
                          ci.magnet_thickness_mm, ci.magnet_width_mm, c.br_test_T, c.br_test_T)  # C36
    return c.torque_2d_Nm / (pullout(sec).torque_per_m * ci.magnet_length_mm * MM)


def s_factor_pair(m, n: int, n_poles: int, backiron: int) -> tuple[float, float, str]:
    """(engine S_n, exact cylindrical S for arcs with the engine's radii, workbook cell).
    Arcs: inner C54 - C20 .. C54, outer C56 .. C60; steel circles at C54 - C20 and C60."""
    r1a = m.inner_face_radius_mm - m.inner_thickness_mm
    radii = (r1a, m.inner_face_radius_mm, m.outer_face_apothem_mm, m.outer_back_apothem_mm)
    kind = "iron" if backiron else "free"
    cyl = cylindrical_s_factor(n * n_poles // 2, *radii, m.gap_radius_mm, iron=(radii[0], radii[3]) if backiron else None)
    cell = HARMONIC_ROWS[n][3] if backiron else HARMONIC_ROWS[n][4]
    return getattr(m, f"s{n}_{kind}"), cyl, cell


def torque_chain_reference(inp, m) -> dict:
    """Calculator!C66:C94 re-derived from first principles: {cell: (engine field, reference value)}.

    Inputs: C5, C6, C8, C10, C41, C43 (inputs); C19-C21, C29-C31, C33, C35, C42 (resolved values);
    and the geometry rows C56, C57, C64, owned by the geometry checks. Fill = block width / pole pitch
    at the block's mid-thickness radius, clamped at 1; Br(T) = Br20 (1 + alpha (T - 20)); pole pitch
    tau_p = 2 pi R_g / N and k_n = n pi / tau_p; B_n from ``square_wave_harmonic``; S_n from
    ``planar_s_factor``; tau_n = B_in B_on / (2 mu0) S_n sin(n pi/2) (the engine's quarter-pitch
    evaluation, examined separately by the peak checks); torque = sum tau_n * 2 pi R_g^2 L;
    f_end = 1 - c_end tau_p / L; pull-out = T2D f_end f_cal, and at 20 °C times Br20^2 / Br(T)^2."""
    c = inp.coupling
    n_poles, backed = c.npole, c.backiron == 1
    t_i, w_i, t_o, w_o = m.inner_thickness_mm, m.inner_width_mm, m.outer_thickness_mm, m.outer_width_mm
    fill_i = min(1.0, w_i * n_poles / (2 * math.pi * (c.inner_back_apothem_mm + t_i / 2)))
    fill_o = min(1.0, w_o * n_poles / (2 * math.pi * (m.outer_face_apothem_mm + t_o / 2)))
    temp_factor = 1 + m.alpha_br_per_C * (c.op_temp_C - 20)
    br_i, br_o = m.inner_br_T * temp_factor, m.outer_br_T * temp_factor
    pitch_mm = 2 * math.pi * m.gap_radius_mm / n_poles
    ref = {"C66": ("fill_inner", fill_i), "C67": ("fill_outer", fill_o),
           "C69": ("br_inner_T_op", br_i), "C70": ("br_outer_T_op", br_o)}
    tau = 0.0
    for n, (c_k, c_bi, c_bo, c_si, c_sf, c_tau) in HARMONIC_ROWS.items():
        k = n * math.pi / (pitch_mm * MM)
        b_i, b_o = square_wave_harmonic(br_i, n, fill_i), square_wave_harmonic(br_o, n, fill_o)
        s_iron = planar_s_factor(k, t_i * MM, t_o * MM, m.face_gap_mm * MM, backed=True)
        s_free = planar_s_factor(k, t_i * MM, t_o * MM, m.face_gap_mm * MM, backed=False)
        tau_n = b_i * b_o / (2 * c.mu0) * (s_iron if backed else s_free) * math.sin(n * math.pi / 2)
        tau += tau_n
        ref.update({c_k: (f"k{n}", k), c_bi: (f"b_i{n}", b_i), c_bo: (f"b_o{n}", b_o),
                    c_si: (f"s{n}_iron", s_iron), c_sf: (f"s{n}_free", s_free), c_tau: (f"tau{n}_Pa", tau_n)})
    area_lever = 2 * math.pi * (m.gap_radius_mm * MM) ** 2 * (m.active_length_mm * MM)
    t2d = tau * area_lever
    f_end = 1 - c.c_end * pitch_mm / m.active_length_mm
    pull = t2d * f_end * m.f_cal
    ref.update({"C89": ("tau_Pa", tau), "C90": ("area_lever_m3", area_lever), "C91": ("torque_2d_Nm", t2d),
                "C92": ("f_end", f_end), "C93": ("pullout_Nm", pull),
                "C94": ("pullout_20C_Nm", pull * (m.inner_br_T * m.outer_br_T) / (br_i * br_o))})
    return ref


# --------------------------------------------------------------------------- chain: term by term
@pytest.mark.parametrize("case", list(CHAIN_CASES))
@pytest.mark.family("torque")
def test_calculator_torque_chain_rederived(case):
    """Every Calculator row C66:C94 (fill, Br(T), k_n, B_n, S_n for both circuits, tau_n, sum, area x
    lever, 2D torque, end factor, pull-out at T and at 20 °C) equals its first-principles value
    (``torque_chain_reference``). Same algebra: TOL_ALGEBRA. One check per case lists every failing row."""
    inp = vary(defaults(), CHAIN_CASES[case])
    m = run(inp).model
    failures = [mismatch(f"{field}, {case}", f"Calculator!{cell}", getattr(m, field), value, TOL_ALGEBRA)
                for cell, (field, value) in torque_chain_reference(inp, m).items()
                if rel_err(getattr(m, field), value) > TOL_ALGEBRA]
    assert not failures, " | ".join(failures)


# --------------------------------------------------------------------------- 2D: pull-out
@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_pullout_2d_matches_flat_block_reference(backiron, face_gap_mm):
    """2D pull-out at 20 °C before end effect and calibration, C94 / (C92 * C42), against the true
    pull-out (max over angle) of the exact 2D flat-block section times C33. Steel: as-built polygonal
    steel with the bondlines (``steel_from``), which decides the steel verdict."""
    _, m = engine_case(backiron, face_gap_mm)
    engine, reference = engine_2d_torque_20c(m), reference_2d_nm(backiron, face_gap_mm, VERDICT_STEEL)
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        f"2D pull-out at 20 °C, {CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        "Calculator!C94/(C92*C42) vs 2D field x C33", engine, reference, TOL_MODEL)


@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.family("torque")
def test_pullout_2d_steel_at_magnet_backs(face_gap_mm):
    """Diagnostic companion of the steel verdict above: the same comparison with the steel moved onto
    the magnet backs, the idealisation inside the engine's S_iron. The as-built check decides the
    verdict; comparing the two separates the formula's own error from the bondline it omits. TOL_MODEL."""
    _, m = engine_case(1, face_gap_mm)
    engine, reference = engine_2d_torque_20c(m), reference_2d_nm(1, face_gap_mm, "at-backs")
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        f"2D pull-out at 20 °C, steel at the magnet backs (no bondline), face gap {face_gap_mm} mm",
        "Calculator!C94/(C92*C42) vs 2D field x C33", engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_calibration_prototype_2d_matches_flat_block_reference():
    """The Calibration sheet's model of the no-iron prototype (20 x B842SH on a 9.85 mm apothem, 1.4 mm
    flat-face gap, 20 °C): its 2D torque before end effect and calibration (Calibration!C43) against
    the exact 2D flat-block section times the magnet length (C18). This 2D value sets the one-point
    correction Calibration!C8 and C9. TOL_MODEL."""
    ratio = prototype_ratio()
    assert abs(ratio - 1) <= TOL_MODEL, mismatch(
        "prototype 2D torque at the test temperature, free space, engine / exact 2D",
        "Calibration!C43 vs 2D field x C18", ratio, 1.0, TOL_MODEL)


@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.family("torque")
def test_free_space_2d_error_transfers_from_prototype(face_gap_mm):
    """The free-space design takes the prototype's one-point correction (C42 = Calibration!C9), which
    cancels the 2D model error only if that error is the same at the design as at the prototype. So
    (engine / exact 2D at the design) / (engine / exact 2D at the prototype) must be 1 within TOL_MODEL:
    this is the 2D part of the calibrated free-space prediction's error."""
    _, m = engine_case(0, face_gap_mm)
    design = engine_2d_torque_20c(m) / reference_2d_nm(0, face_gap_mm, VERDICT_STEEL)
    assert abs(design / prototype_ratio() - 1) <= TOL_MODEL, mismatch(
        f"free-space 2D error at face gap {face_gap_mm} mm relative to the prototype's",
        "Calculator!C94/(C92*C42) and Calibration!C43 vs 2D field", design, prototype_ratio(), TOL_MODEL)


# --------------------------------------------------------------------------- 2D: angle and harmonics
@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_quarter_pitch_evaluation_is_true_peak(backiron, face_gap_mm):
    """The engine evaluates every harmonic at the fundamental's pull-out angle (theta_e = pi/2, hence
    sin(n pi/2)). In the exact 2D section the torque there must be within TOL_PEAK of the true max over
    angle; the observed peak angle is in the message."""
    ref = reference_pullout(backiron, face_gap_mm, VERDICT_STEEL)
    assert abs(ref.at_quarter / ref.torque_per_m - 1) <= TOL_PEAK, mismatch(
        f"2D torque at theta_e=90° vs true peak at {math.degrees(ref.theta_e_peak):.3f}° electrical, "
        f"{CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        "Calculator!C76,C82,C88 sin(n·pi/2)", ref.at_quarter, ref.torque_per_m, TOL_PEAK)


@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_engine_harmonic_series_peaks_at_quarter_pitch(backiron, face_gap_mm):
    """The engine's own curve tau(theta) = sum of tau_n / sin(n pi/2) * sin(n theta) must peak where it is
    evaluated, or C89 is not the pull-out of its own model (a positive third-harmonic amplitude above
    a_1 / 9 would make the top double-humped). Its max over angle (dense scan + bounded search) must
    equal C89. TOL_ALGEBRA."""
    _, m = engine_case(backiron, face_gap_mm)
    amp = {n: tau / math.sin(n * math.pi / 2) for n, tau in ((1, m.tau1_Pa), (3, m.tau3_Pa), (5, m.tau5_Pa))}
    curve = lambda th: sum(a * math.sin(n * th) for n, a in amp.items())
    grid = np.linspace(0.0, math.pi, 2001)
    i = int(np.argmax([curve(x) for x in grid]))
    best = minimize_scalar(lambda x: -curve(x), bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 2000)]),
                           method="bounded", options={"xatol": 1e-12})
    assert rel_err(m.tau_Pa, -best.fun) <= TOL_ALGEBRA, mismatch(
        f"engine shear at 90° vs max of its own harmonic curve (at {math.degrees(best.x):.4f}°), "
        f"{CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        "Calculator!C89 vs C76,C82,C88", m.tau_Pa, -best.fun, TOL_ALGEBRA)


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_harmonic_torque_amplitudes(backiron, n):
    """Per-harmonic torque-angle amplitude at 20 °C (engine tau_n / sin(n pi/2) * C90 scaled to 20 °C)
    against the sine coefficient a_n of the exact 2D torque-angle curve times C33 (default gap, as-built
    steel). The difference is measured against the fundamental a_1: harmonics 3 and 5 are about 1 % of
    a_1, so a relative bound on them would flag differences that cannot move the pull-out.
    |a_n engine - a_n reference| <= TOL_MODEL * a_1 reference."""
    _, m = engine_case(backiron, DEFAULT_GAP_MM)
    scale = m.active_length_mm * MM
    harmonics = reference_pullout(backiron, DEFAULT_GAP_MM, VERDICT_STEEL).harmonics
    engine, reference, a1 = engine_harmonic_torque_20c(m, n), harmonics[n] * scale, harmonics[1] * scale
    cell = HARMONIC_ROWS[n][5]
    assert abs(engine - reference) <= TOL_MODEL * abs(a1), mismatch(
        f"harmonic {n}, {CIRCUITS[backiron]}: (engine a_n - reference a_n) / reference a_1; "
        f"engine a_n = {engine:.6g} N·m, reference a_n = {reference:.6g} N·m, reference a_1 = {a1:.6g} N·m",
        f"Calculator!{cell}*C90*(C21*C31)/(C69*C70) vs 2D field x C33", (engine - reference) / a1, 0.0, TOL_MODEL)


@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_omitted_harmonics_negligible(backiron):
    """The engine keeps harmonics 1, 3, 5 only (C89 = C76 + C82 + C88). The reference torque at 90° minus
    its own n <= 5 terms is the part the engine drops (n >= 7, plus the even cogging terms, which vanish
    at 90°); it must stay within TOL_MODEL of the true pull-out (default gap, as-built steel)."""
    ref = reference_pullout(backiron, DEFAULT_GAP_MM, VERDICT_STEEL)
    kept = sum(ref.harmonics[n] * math.sin(n * math.pi / 2) for n in range(1, 6))
    tail = (ref.at_quarter - kept) / ref.torque_per_m
    assert abs(tail) <= TOL_MODEL, mismatch(
        f"reference harmonic tail n >= 7 at 90° as a fraction of the true pull-out, {CIRCUITS[backiron]}",
        "Calculator!C89 (harmonics 1, 3, 5 only)", tail, 0.0, TOL_MODEL)


# --------------------------------------------------------------------------- S_n: curvature and planar limit
@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("face_gap_mm", FACE_GAPS_MM, ids=lambda g: f"{g}mm")
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_planar_s_factor_matches_cylindrical(backiron, face_gap_mm, n):
    """Planar (unrolled, k = n (N/2) / R_gap) S_n against the exact cylindrical factor for arcs with the
    engine's radii (inner C54 - C20 .. C54, outer C56 .. C60, steel at C54 - C20 and C60), same
    normalisation: torque = B_i B_o / (2 mu0) * S * 2 pi C64^2. TOL_MODEL."""
    inp, m = engine_case(backiron, face_gap_mm)
    engine, reference, cell = s_factor_pair(m, n, inp.coupling.npole, backiron)
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        f"S_{n} planar vs cylindrical, {CIRCUITS[backiron]}, face gap {face_gap_mm} mm",
        f"Calculator!{cell} (C71, C54, C56, C60, C64)", engine, reference, TOL_MODEL)


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("backiron", [0, 1], ids=["free", "steel"])
@pytest.mark.family("torque")
def test_planar_s_factor_is_exact_planar_limit(backiron, n):
    """At a planar-like design (20000 poles on a 28 m hub, R_gap 28.004 m, k1 = 357 /m like the default)
    the engine's S_n, including the free-space factor's /2 and the steel sinh form, must equal the exact
    cylindrical factor to O((dr / R_gap)^2). TOL_PLANAR_LIMIT."""
    m = run(vary(defaults(), {"coupling.npole": 20000, "coupling.inner_back_apothem_mm": 28000.0,
                              "coupling.backiron": backiron})).model
    engine, reference, cell = s_factor_pair(m, n, 20000, backiron)
    assert rel_err(engine, reference) <= TOL_PLANAR_LIMIT, mismatch(
        f"S_{n} planar limit (R_gap 28 m), {CIRCUITS[backiron]}", f"Calculator!{cell}", engine, reference,
        TOL_PLANAR_LIMIT)
```

- [ ] **Step 5: Run the checks**

Run:
```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_torque2d_reference_sanity.py audit/tests/test_torque_field2d.py
```

Expected: `15 failed, 66 passed`, about 7 s in total (81 checks, all `torque`: 18 sanity checks, all passed, and 63 engine checks, 48 passed and 15 failed). The command exits non-zero because of the FAILs. The 63 engine checks are: term-by-term chain 6, 2D pull-out 6, steel at the magnet backs 3, prototype 1, prototype-to-design transfer 3, quarter-pitch evaluation 6, engine harmonic series peak 6, harmonic amplitudes 6, omitted harmonics 2, planar S_n against exact arcs 18, planar limit 6. The 15 FAILs are **candidate findings for Task 8**, not bugs to fix now. Do not edit `magcoupling/`, and do not change a tolerance or a check to make one pass. All 15 fail against `TOL_MODEL` (2 %, defined in `audit/common.py`); every `TOL_ALGEBRA`, `TOL_PEAK` and `TOL_PLANAR_LIMIT` check passes.

FAIL (observed). Errors are engine / reference − 1; the assertion messages in `audit/out/results.json` carry the same numbers:
1. `test_pullout_2d_matches_flat_block_reference`, free space only (3 FAILs; the 3 steel as-built cases pass). Engine C94/(C92·C42) against the exact 2D pull-out × C33, at 20 °C:
   - `[free-0.5mm]`: 2.573874 vs 2.676652 N·m (−3.84 %).
   - `[free-1.4mm]`: 1.912786 vs 1.991230 N·m (−3.94 %).
   - `[free-3.0mm]`: 1.131771 vs 1.181119 N·m (−4.18 %).
2. `test_pullout_2d_steel_at_magnet_backs` (2 FAILs; diagnostic only, the as-built steel verdict `test_pullout_2d_matches_flat_block_reference[steel-*]` passes at all three gaps):
   - `[0.5mm]`: 4.477681 vs 4.597124 N·m (−2.60 %).
   - `[1.4mm]`: 3.346836 vs 3.424449 N·m (−2.27 %).
3. `test_calibration_prototype_2d_matches_flat_block_reference`: Calibration!C43 = 1.880350 vs exact 2D 1.967081 N·m, engine/ref 0.9559 (−4.41 %).
4. `test_harmonic_torque_amplitudes[free-1]`: free-space fundamental a_1 1.891436 (engine) vs 1.973783 N·m (reference), difference −0.082348 N·m = −4.17 % of a_1 (bound 2 %). The other 5 harmonic cases pass.
5. `test_planar_s_factor_matches_cylindrical` (8 FAILs of 18): planar S_n against the exact cylindrical factor for arcs with the engine's radii.
   - Free space, 3.0 mm gap: S_1 0.078396 vs 0.076332 (+2.70 %); S_3 0.022102 vs 0.021368 (+3.43 %); S_5 0.0031404 vs 0.0030426 (+3.21 %).
   - Steel, 0.5 mm gap: S_1 0.341533 vs 0.334806 (+2.01 %, a marginal FAIL).
   - Steel, 1.4 mm gap: S_1 0.244488 vs 0.237718 (+2.85 %).
   - Steel, 3.0 mm gap: S_1 0.141696 vs 0.135059 (+4.91 %, the largest); S_3 0.023924 vs 0.023304 (+2.66 %); S_5 0.0031703 vs 0.0030853 (+2.76 %).

Observations for Task 8, from the numbers above (whether FAILs 1, 3 and 4 share a root cause is Task 8's call):
- FAILs 1, 3 and 4 are the same size: the engine's free-space 2D torque sits 3.8 to 4.4 % below the exact 2D field at the design, at the prototype and in the fundamental.
- Moving the steel from the magnet backs onto the as-built flats (with the bondlines) lowers the exact torque by 1.6 to 1.7 % at every gap; that is why the as-built steel pull-out passes while FAIL 2 does not.

The 66 passing checks still quantify model approximations, and Task 8 must list the numbers under "Model approximation". A passing check writes no numbers to `results.json`. Observed values, at 20 °C, before end effect and calibration:

2D pull-out, engine C94/(C92·C42) against the exact 2D flat-block section × C33:

| Circuit | Face gap | Engine (N·m) | Exact 2D (N·m) | Engine/ref | Peak angle (electrical) | T(90°)/max − 1 | vs 2 % |
|---|---|---|---|---|---|---|---|
| Free space | 0.5 mm | 2.573874 | 2.676652 | 0.9616 | 90.000° | 2e-16 | FAIL (1) |
| Free space | 1.4 mm | 1.912786 | 1.991230 | 0.9606 | 90.000° | −8e-16 | FAIL (1) |
| Free space | 3.0 mm | 1.131771 | 1.181119 | 0.9582 | 90.000° | 7e-16 | FAIL (1) |
| Steel, as-built (verdict) | 0.5 mm | 4.477681 | 4.517555 | 0.9912 | 89.702° | −1.17e-5 | pass (−0.88 %) |
| Steel, as-built (verdict) | 1.4 mm | 3.346836 | 3.367425 | 0.9939 | 89.734° | −1.11e-5 | pass (−0.61 %) |
| Steel, as-built (verdict) | 3.0 mm | 2.034135 | 2.031545 | 1.0013 | 89.754° | −9.8e-6 | pass (+0.13 %) |
| Steel at magnet backs (diagnostic) | 0.5 mm | 4.477681 | 4.597124 | 0.9740 | | | FAIL (2) |
| Steel at magnet backs (diagnostic) | 1.4 mm | 3.346836 | 3.424449 | 0.9773 | | | FAIL (2) |
| Steel at magnet backs (diagnostic) | 3.0 mm | 2.034135 | 2.065518 | 0.9848 | | | pass (−1.52 %) |

Other checks:
- **Prototype.** Calibration!C43 = 1.880350 against 1.967081 N·m: engine/ref 0.9559. FAIL (3).
- **Transfer to the design.** (Engine/ref at the design) / (engine/ref at the prototype) is 1.0060 at 0.5 mm, 1.0049 at 1.4 mm and 1.0024 at 3.0 mm. All 3 pass: the one-point correction cancels the free-space 2D error to within 0.6 %.
- **Harmonics at the default gap.** Amplitudes as engine / reference, in N·m. The difference is shown as a fraction of a_1. Only free-space n = 1 fails (FAIL 4); the other 5 pass.

  | Circuit | n = 1 | n = 3 | n = 5 | Difference / a_1 (n = 1, 3, 5) |
  |---|---|---|---|---|
  | Free space | 1.891436 / 1.973783 | −0.033934 / −0.015764 | −0.012584 / +0.006007 | −4.2 % (FAIL), −0.92 %, −0.94 % |
  | Steel | 3.323251 / 3.344838 | −0.036257 / −0.020844 | −0.012672 / +0.006027 | −0.65 %, −0.46 %, −0.56 % |

  - The engine's n = 5 has the opposite sign to the exact curve.
  - The terms the engine drops (n ≥ 7) are −0.22 % (free space) and −0.13 % (steel) of the pull-out. Both omitted-harmonics checks pass.
  - The steel flats add cogging terms a_2 = 0.0086 N·m and a_4 = 0.00025 N·m. These move the steel peak by −0.27° electrical. All 6 quarter-pitch checks pass (`TOL_PEAK`).
- **The engine's own 3-harmonic curve** peaks at exactly 90° in all 6 cases, and C89 equals its maximum. All 6 pass.
- **Term-by-term chain.** 6 cases × 28 rows (the sixth, `free-unequal-rings`, is Task 3's free-space asymmetric scenario: manual unequal magnets in both rings, 50 °C). The worst row error is 3.6e-15 (`steel-12-poles-cold`, C79); the new case's is 1.9e-15. All 6 pass.
- **Planar S_n against exact arcs.** The error grows with the gap. 8 FAILs (5) and 10 passes: free space at 0.5 and 1.4 mm passes for all n (+0.09 to +1.34 %), and steel S_3 and S_5 at 0.5 and 1.4 mm pass (−0.07 to +0.62 %). The largest error is steel S_1 at 3.0 mm: 0.141696 against 0.135059, +4.9 %.
- **Planar limit.** At 28 m, all 6 S_n factors agree with the exact cylindrical factors to 7e-9 or better. All 6 pass.

- [ ] **Step 6: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 add reference/magcoupling-py/audit/references/planar.py reference/magcoupling-py/audit/references/field2d.py reference/magcoupling-py/audit/tests/test_torque2d_reference_sanity.py reference/magcoupling-py/audit/tests/test_torque_field2d.py
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 commit -m "test(magcoupling-audit): torque independent checks" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
```

### Task 3: Torque model - real 3D cross-check, scaling laws, temperature, calibration

**Files:**
- Modify: `reference/magcoupling-py/audit/references/planar.py` (Task 2 creates it and owns its top part, `square_wave_harmonic` and `planar_s_factor`; this task appends the shared torque-chain and Br(T) formulas, reused by Tasks 4-7)
- Create: `reference/magcoupling-py/audit/references/torque_ref.py`
- Create: `reference/magcoupling-py/audit/tests/test_torque_reference_sanity.py`
- Create: `reference/magcoupling-py/audit/tests/test_torque_laws.py`

**Interfaces:**
- Consumes:
  - Task 1's `audit.common` (this task does not edit it): `defaults`, `run`, `vary`, `rel_err`, `mismatch`, `MU0_EXACT`, `TOL_ALGEBRA`. Also `TOL_MODEL` (0.02, the 2 % model-accuracy bar) and `TOL_PEAK` (1e-3, the peak-angle resolution bound), both imported from `audit.common` (Task 1) and defined nowhere else. Nothing else from the harness.
  - Task 2's `audit.references.planar` (SI units): `square_wave_harmonic(br_T, n, fill) -> float` and `planar_s_factor(k, t_i, t_o, g, backed) -> float`. The functions this task appends call them and do not redefine them. `square_wave_harmonic` is defined for odd n only: it returns 0 for even n and raises `ValueError` for n < 1.
  - Task 2: `audit.references.field2d.CouplingSection(n_poles, inner_back_mm, inner_thickness_mm, inner_width_mm, outer_face_mm, outer_thickness_mm, outer_width_mm, br_inner_T, br_outer_T)` and `field2d.torque_per_length(sec, theta_e)`. These are used by exactly one sanity test, which cross-validates the two independent references and imports them inside the test.
  - Third party: `magpylib>=5` (closed-form cuboid field), `numpy`, `scipy` (bounded Brent search, `Rotation`, `quad`).
  - Engine result fields read:
    - `model.{inner_face_radius_mm, inner_thickness_mm, inner_width_mm, inner_length_mm, outer_face_apothem_mm, outer_thickness_mm, outer_width_mm, outer_length_mm, inner_br_T, outer_br_T, br_inner_T_op, br_outer_T_op, active_length_mm, pole_pitch_mm, gap_radius_mm, face_gap_mm, corner_gap_mm, fill_inner, fill_outer, k1, s1_iron, s1_free, k3, s3_iron, s3_free, k5, s5_iron, s5_free, torque_2d_Nm, f_end, f_cal, pullout_Nm, pullout_20C_Nm, pullout_iron_Nm, pullout_noiron_Nm}`
    - `calibration.{poles_per_ring, face_radius_mm, corner_radius_mm, corner_gap_mm, outer_face_apothem_mm, flat_gap_mm, gap_radius_mm, fill_inner, fill_outer, br_test_T, pole_pitch_mm, f_end, tau1_Pa, tau3_Pa, tau5_Pa, torque_2d_Nm, model_torque_Nm, original_model_Nm, model_error, measured_over_model, f_cal_updated, fea_interp_Nm, fea_interp_error}`
    - Workbook cells come from each result field's `cell` metadata (`dataclasses.fields`), where the test names a single field.
  - Inputs varied with `vary()`:
    - `coupling.{op_temp_C, backiron, npole, faceted, inner_back_apothem_mm, mu0}`
    - `coupling.magnets.{part_inner, part_outer, manual_inner_/manual_outer_ length_mm, width_mm, thickness_mm, br_T}`
    - `metal.face_gap_mm`
    - `calibration.{gap_definition, spacing_mm, test_temp_C, measured_torque_Nm, total_magnets, mu0, fea_torque1_Nm, fea_torque2_Nm}`
  - Inputs read only: `coupling.c_end` and the rest of `calibration.*`.
- Checked by other tasks (one owner per cell, so this task does not repeat them):
  - The term-by-term re-derivation of the torque chain C66-C94, which includes the free-space unequal-rings case this task once carried, is Task 2's `test_torque_field2d.py::test_calculator_torque_chain_rederived`.
  - The gearbox cells C99 and C100, including the ratio 7 / efficiency 0.8 / free-space scenario, are Task 4's `test_metal_design.py::test_gearbox_equivalents`.
  - The magnet rating verdicts C107 and C108 (with C22 and C32), including the mixed inner 150 °C / outer 80 °C case at 100 °C, are Task 5's `test_temperature_demag_adhesive.py::test_magnet_temperature_checks_follow_kj_rating`.
  - This task keeps C95, C96, the Calibration sheet, and the 3D, scaling-law, temperature-scaling and constants checks of the model behind C69-C94.
- Produces:
  - `audit/references/planar.py`, appended to Task 2's file. Together with Task 2's `square_wave_harmonic` and `planar_s_factor`, it is the single copy of the planar and material formulas; Tasks 4, 5 and 7 should import it instead of keeping their own. This task appends:
    - `br_at(br20_T: float, alpha_per_C: float, temp_C: float) -> float`
    - `wave_number(n: int, pole_pitch_m: float) -> float`
    - `iron_backed_factor(k: float, t_i: float, t_o: float, g: float) -> float` (Task 2's `planar_s_factor(..., backed=True)` under the name the checks use)
    - `free_space_factor(k: float, t_i: float, t_o: float, g: float) -> float` (`planar_s_factor(..., backed=False)`)
    - `harmonic_shear_stress(b_i_n: float, b_o_n: float, s_n: float, k: float, delta_m: float, mu0: float) -> float`
    - `planar_shear_stress(br_i, br_o, fill_i, fill_o, pole_pitch_m, t_i_m, t_o_m, g_m, delta_m, mu0, backed: bool, harmonics: Iterable[int]) -> float`
    - `torque_on_cylinder(tau_Pa: float, r_m: float, length_m: float) -> float`
    - `end_factor(c_end: float, pole_pitch_mm: float, length_mm: float) -> float`
  - `audit/references/torque_ref.py`
    - `MM = 1e-3`
    - `FieldFn = Callable[[np.ndarray], np.ndarray]`
    - `Block(x_m, y_m, phi, a_m, b_m, c_m, j_T)`: frozen dataclass, polarized along local x. `Block.to_magpylib() -> magpy.magnet.Cuboid`.
    - `magpylib_field(sources: Iterable[Block]) -> FieldFn`
    - `torque_on_blocks(targets: Iterable[Block], field: FieldFn, panel_m: float = 1.0 * MM, order: int = 4, mu0: float = MU0_EXACT) -> tuple[np.ndarray, np.ndarray]`
    - `RingPair(npole, r_i_m, t_i_m, w_i_m, l_i_m, j_i_T, r_o_m, t_o_m, w_o_m, l_o_m, j_o_T)`: frozen. Property `RingPair.half_pitch`; method `RingPair.with_length(length_m) -> RingPair`.
    - `ring_pair_from_engine(inp, res) -> RingPair`: raises `ValueError` for `faceted != 1` or an odd pole count.
    - `ring_blocks(npole, r_m, t_m, w_m, l_m, j_T, phase) -> list[Block]`
    - `ring_torque_3d(pair, theta, panel_m=1.0 * MM, order=4, all_blocks=False) -> float`: restoring torque on the outer ring.
    - `ring_pullout_3d(pair, panel_m=1.0 * MM, order=4, n_scan=9) -> tuple[float, float]`
    - `torque_per_length_3d(pair, l1_m, l2_m, theta, panel_m=1.0 * MM, order=4) -> float`
    - `end_factor_3d(pair, length_m, per_length, theta, panel_m=1.0 * MM, order=4) -> float`
    - Prototype helpers: `PrototypeGeometry`; `prototype_geometry(c)`; `prototype_br(c) -> float`; `prototype_calibration(c, harmonics=(1, 3, 5)) -> dict`; `ring_pair_from_prototype(c) -> RingPair`. Here `c` is a `CalibrationInputs`-like object.

- [ ] **Step 1: Write the reference modules**

Task 2 has already created `reference/magcoupling-py/audit/references/planar.py` with `square_wave_harmonic` and `planar_s_factor`. Append the section below to it and do not redefine those two functions. The appended `from typing import Iterable` import sits below Task 2's module header on purpose, because that header belongs to Task 2.

File: `reference/magcoupling-py/audit/references/planar.py` (append)

```python



# =========================================================================== Task 3 additions
# The planar (2D slab) torque chain and Br(T), appended by Task 3 on top of Task 2's building blocks
# above. ``square_wave_harmonic`` and ``planar_s_factor`` are Task 2's and are reused here, not
# redefined. The names and signatures below match the copies drafted in Task 7 (sweep_ref: ``br_at``,
# ``square_wave_harmonic``, ``iron_backed_factor``, ``free_space_factor``, ``torque_on_cylinder``,
# ``end_factor``), so later tasks import them instead of keeping their own copies.
#
# * ``wave_number``: k_n = n pi / pole pitch (harmonic n repeats every 2 * pitch / n).
# * ``free_space_factor`` / ``iron_backed_factor``: the two branches of Task 2's ``planar_s_factor``
#   under the names the torque-chain checks use (free space; infinitely permeable backings behind both
#   arrays). Sources: Furlani, *Permanent Magnet and Electromechanical Devices*, Academic Press, 2001
#   (2D fields of magnetized arrays and couplings); Hague, *The Principles of Electromagnetism Applied
#   to Electrical Machines*, 1929 (image solutions between permeable boundaries). The Task 3 sanity
#   tests check the free-space factor against 3D arrays, and both factors against their thick-array
#   and image-series limits.
# * ``harmonic_shear_stress`` / ``planar_shear_stress``: the shear stress between the two arrays
#   displaced by delta, tau_n = B_in B_on / (2 mu0) * S_n * sin(k_n delta). It is positive when it
#   opposes the displacement.
# * ``torque_on_cylinder``: a uniform shear stress tau on a cylinder of radius r and length L gives a
#   force tau 2 pi r L acting at lever arm r.
# * ``end_factor``: the engine's documented EMPIRICAL end-effect form 1 - c_end * pitch / L (README,
#   model step 4). It restates the same algebra and is not physics; the physical end effect is
#   ``torque_ref.end_factor_3d``.
# * ``br_at``: linear reversible remanence Br(T) = Br20 (1 + alpha (T - 20)).
#
# Units are SI (m, T, Pa, N*m) unless a name ends in _mm.
from typing import Iterable  # noqa: E402  (appended section; the module header belongs to Task 2)


def br_at(br20_T: float, alpha_per_C: float, temp_C: float) -> float:
    """Linear reversible remanence [T]: Br(T) = Br20 * (1 + alpha * (T - 20))."""
    return br20_T * (1.0 + alpha_per_C * (temp_C - 20.0))


def wave_number(n: int, pole_pitch_m: float) -> float:
    """Wave number [1/m] of harmonic n for a pole pitch in metres: k_n = n * pi / pitch."""
    return n * math.pi / pole_pitch_m


def iron_backed_factor(k: float, t_i: float, t_o: float, g: float) -> float:
    """S_n with infinitely permeable backings behind both arrays (k in 1/m, lengths in m): Task 2's steel form."""
    return planar_s_factor(k, t_i, t_o, g, backed=True)


def free_space_factor(k: float, t_i: float, t_o: float, g: float) -> float:
    """S_n for two arrays in free space (k in 1/m, lengths in m): Task 2's free-space form."""
    return planar_s_factor(k, t_i, t_o, g, backed=False)


def harmonic_shear_stress(b_i_n: float, b_o_n: float, s_n: float, k: float, delta_m: float, mu0: float) -> float:
    """Shear stress [Pa] of one harmonic: B_in * B_on / (2 mu0) * S_n * sin(k * delta)."""
    return b_i_n * b_o_n / (2.0 * mu0) * s_n * math.sin(k * delta_m)


def planar_shear_stress(br_i: float, br_o: float, fill_i: float, fill_o: float, pole_pitch_m: float,
                        t_i_m: float, t_o_m: float, g_m: float, delta_m: float, mu0: float, backed: bool,
                        harmonics: Iterable[int]) -> float:
    """Shear stress [Pa] between two planar alternating arrays displaced by ``delta_m``, summed over ``harmonics``.

    ``backed`` selects infinitely permeable backings behind both arrays; otherwise free space.
    """
    tau = 0.0
    for n in harmonics:
        k = wave_number(n, pole_pitch_m)
        if backed:
            s = iron_backed_factor(k, t_i_m, t_o_m, g_m)
        else:
            s = free_space_factor(k, t_i_m, t_o_m, g_m)
        tau += harmonic_shear_stress(square_wave_harmonic(br_i, n, fill_i), square_wave_harmonic(br_o, n, fill_o),
                                     s, k, delta_m, mu0)
    return tau


def torque_on_cylinder(tau_Pa: float, r_m: float, length_m: float) -> float:
    """Torque [N*m] of a uniform shear stress on a cylinder: tau * 2 pi r L * r."""
    return tau_Pa * 2.0 * math.pi * r_m ** 2 * length_m


def end_factor(c_end: float, pole_pitch_mm: float, length_mm: float) -> float:
    """The engine's documented empirical end-effect form, 1 - c_end * pitch / L (dimensionless)."""
    return 1.0 - c_end * pole_pitch_mm / length_mm
```

File: `reference/magcoupling-py/audit/references/torque_ref.py` (create)

```python
"""Independent torque references for the magcoupling torque model (M1, Task 3).

Nothing here imports or calls ``magcoupling``. Two references:

1. **3D magnetostatics** (``torque_on_blocks``, ``ring_torque_3d``, ``ring_pullout_3d``,
   ``torque_per_length_3d``, ``end_factor_3d``). Each block is a rigid, uniformly magnetized
   cuboid with relative permeability 1. That is magpylib's model, and the engine's harmonic
   model assumes the same. The source ring's field comes from magpylib's closed-form cuboid
   solution. The force on a target block uses the magnetic surface-charge model: a uniformly
   magnetized body carries charge sigma = M.n = (J.n)/mu0 on its surfaces and no volume
   charge, so in the external field B_ext it feels

       F = sum_faces sigma * integral(B_ext dA),   T = sum_faces sigma * integral(r x B_ext dA).

   This is exact for rigid uniform magnets. With M uniform and B_ext curl-free and
   divergence-free inside the target, the divergence theorem turns the dipole-density force
   integral(M.grad)B_ext dV and torque integral(M x B_ext + r x (M.grad)B_ext) dV into these
   surface integrals. Only the two faces normal to the magnetization carry charge. The face
   integrals use composite Gauss-Legendre quadrature; the integrand is smooth because the
   target faces never touch the sources.

2. **Prototype calibration** (``prototype_geometry``, ``prototype_calibration``). This
   re-derives the Calibration sheet's one-point correction from the prototype description:
   two equal rings of flat blocks on a polygonal hub, in free space, with a bench pull-out
   torque. The planar algebra comes from ``audit.references.planar``.

INDEPENDENCE NOTE: ``ring_pair_from_engine`` reads the ring geometry from the engine's
Calculator geometry fields (C18-C20, C28-C30, C54, C56, C69, C70). The 3D checks built on it
therefore inherit any error in those cells. That is acceptable because Task 4
(test_geometry_mass.py) verifies the faceted geometry cells independently. The prototype
rings (``ring_pair_from_prototype``) are built from the Calibration inputs alone.

Units: SI (m, T, N, N*m) in the 3D code. The calibration helpers take the engine's mm inputs
and convert them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Iterable

import magpylib as magpy
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from audit.common import MU0_EXACT
from audit.references import planar

MM = 1e-3
FieldFn = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- 3D: blocks and charge-model force
@dataclass(frozen=True)
class Block:
    """Uniformly magnetized cuboid centred in the z = 0 plane, polarized along its local x axis.

    ``phi`` rotates the block about +z, so local x maps to (cos phi, sin phi, 0).
    ``a_m`` is the size along local x (the magnetized direction), ``b_m`` along local y and
    ``c_m`` along z. ``j_T`` is the signed polarization J = mu0 M along local x.
    """
    x_m: float
    y_m: float
    phi: float
    a_m: float
    b_m: float
    c_m: float
    j_T: float

    def to_magpylib(self) -> magpy.magnet.Cuboid:
        return magpy.magnet.Cuboid(dimension=(self.a_m, self.b_m, self.c_m), polarization=(self.j_T, 0.0, 0.0),
                                   position=(self.x_m, self.y_m, 0.0), orientation=Rotation.from_euler("z", self.phi))


def magpylib_field(sources: Iterable[Block]) -> FieldFn:
    """B field [T] of ``sources`` at an (n, 3) array of points [m], from magpylib's analytical cuboid solution."""
    mags = [b.to_magpylib() for b in sources]

    def field(points: np.ndarray) -> np.ndarray:
        return np.asarray(magpy.getB(mags, points, sumup=True), dtype=float).reshape(-1, 3)

    return field


def _composite_gauss(length: float, panel: float, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights of composite Gauss-Legendre quadrature on [-length/2, length/2]."""
    n_panels = max(1, math.ceil(length / panel - 1e-9))
    x, w = np.polynomial.legendre.leggauss(order)
    edges = np.linspace(-length / 2, length / 2, n_panels + 1)
    mid, half = (edges[:-1] + edges[1:]) / 2, (edges[1:] - edges[:-1]) / 2
    return (mid[:, None] + half[:, None] * x[None, :]).ravel(), (half[:, None] * w[None, :]).ravel()


def torque_on_blocks(targets: Iterable[Block], field: FieldFn, panel_m: float = 1.0 * MM,
                     order: int = 4, mu0: float = MU0_EXACT) -> tuple[np.ndarray, np.ndarray]:
    """Net force [N] and torque about the origin [N*m] on ``targets`` in the external ``field``.

    Surface-charge model (see the module docstring): charge sigma = +-J/mu0 on the two faces
    normal to each block's magnetization. The faces are integrated with composite
    Gauss-Legendre quadrature, ``order`` points per panel, panels no longer than ``panel_m``.
    """
    pts, dq = [], []                     # quadrature points and charge elements sigma*dA [A*m]
    for blk in targets:
        y, wy = _composite_gauss(blk.b_m, panel_m, order)
        z, wz = _composite_gauss(blk.c_m, panel_m, order)
        Y, Z = np.meshgrid(y, z, indexing="ij")
        W = np.outer(wy, wz).ravel()
        c, s = math.cos(blk.phi), math.sin(blk.phi)
        for side in (1.0, -1.0):         # +x face carries +J/mu0, -x face carries -J/mu0
            lx = side * blk.a_m / 2
            gx = blk.x_m + c * lx - s * Y.ravel()
            gy = blk.y_m + s * lx + c * Y.ravel()
            pts.append(np.c_[gx, gy, Z.ravel()])
            dq.append(side * blk.j_T / mu0 * W)
    P, Q = np.vstack(pts), np.concatenate(dq)
    dF = Q[:, None] * field(P)
    return dF.sum(axis=0), np.cross(P, dF).sum(axis=0)


# --------------------------------------------------------------------------- 3D: coaxial rings of flat blocks
@dataclass(frozen=True)
class RingPair:
    """Two coaxial rings of flat blocks with alternating radial polarization (block i at 2*pi*i/N, sign (-1)^i).

    Radii are block-centre radii. The flat back face is at r - t/2 and the magnet face is at
    r + t/2 for the inner ring, r - t/2 for the outer ring. Both rings are centred on z = 0.
    """
    npole: int
    r_i_m: float
    t_i_m: float
    w_i_m: float
    l_i_m: float
    j_i_T: float
    r_o_m: float
    t_o_m: float
    w_o_m: float
    l_o_m: float
    j_o_T: float

    @property
    def half_pitch(self) -> float:
        """Half a pole pitch [rad]: where the engine evaluates pull-out (sin(n pi/2) in every tau_n)."""
        return math.pi / self.npole

    def with_length(self, length_m: float) -> "RingPair":
        return replace(self, l_i_m=length_m, l_o_m=length_m)


def ring_pair_from_engine(inp, res) -> RingPair:
    """Build the 3D ring pair exactly as the engine's geometry fields describe it.

    Inner block: radial span [inner_face_radius - t_i, inner_face_radius] (Calculator!C54, C20).
    Outer block: radial span [outer_face_apothem, outer_face_apothem + t_o] (Calculator!C56, C30).
    Width, axial length and Br at the operating temperature come from Calculator!C18-C20,
    C28-C30, C69 and C70. See the module's independence note.
    """
    if inp.coupling.faceted != 1:
        raise ValueError("the 3D reference models flat blocks only (coupling.faceted = 1)")
    n = inp.coupling.npole
    if n < 2 or n % 2:
        raise ValueError(f"alternating rings need an even pole count >= 2, got {n}")
    m = res.model
    return RingPair(npole=n,
                    r_i_m=(m.inner_face_radius_mm - m.inner_thickness_mm / 2) * MM, t_i_m=m.inner_thickness_mm * MM,
                    w_i_m=m.inner_width_mm * MM, l_i_m=m.inner_length_mm * MM, j_i_T=m.br_inner_T_op,
                    r_o_m=(m.outer_face_apothem_mm + m.outer_thickness_mm / 2) * MM, t_o_m=m.outer_thickness_mm * MM,
                    w_o_m=m.outer_width_mm * MM, l_o_m=m.outer_length_mm * MM, j_o_T=m.br_outer_T_op)


def ring_blocks(npole: int, r_m: float, t_m: float, w_m: float, l_m: float, j_T: float, phase: float) -> list[Block]:
    """One ring: N flat blocks, block i centred at angle phase + 2*pi*i/N, polarization (-1)^i * j_T radially."""
    out = []
    for i in range(npole):
        ang = phase + 2 * math.pi * i / npole
        out.append(Block(r_m * math.cos(ang), r_m * math.sin(ang), ang, t_m, w_m, l_m, j_T * (-1) ** i))
    return out


def ring_torque_3d(pair: RingPair, theta: float, panel_m: float = 1.0 * MM, order: int = 4,
                   all_blocks: bool = False) -> float:
    """Restoring torque [N*m] on the outer ring when it is rotated by ``theta`` [rad] from the aligned position.

    The value is positive for 0 < theta < 2*pi/N, where the torque pulls the outer ring back
    to theta = 0. With ``all_blocks=False`` only outer block 0 is integrated and the result is
    multiplied by N. Rotating the whole assembly by 2*pi/N maps inner block i to i+1 and outer
    block i to i+1 with both polarities flipped, so every outer block feels the same torque.
    The sanity tests check this against ``all_blocks=True``.
    """
    inner = ring_blocks(pair.npole, pair.r_i_m, pair.t_i_m, pair.w_i_m, pair.l_i_m, pair.j_i_T, 0.0)
    outer = ring_blocks(pair.npole, pair.r_o_m, pair.t_o_m, pair.w_o_m, pair.l_o_m, pair.j_o_T, theta)
    targets = outer if all_blocks else outer[:1]
    _, t = torque_on_blocks(targets, magpylib_field(inner), panel_m, order)
    return -float(t[2]) * (1 if all_blocks else pair.npole)


def ring_pullout_3d(pair: RingPair, panel_m: float = 1.0 * MM, order: int = 4, n_scan: int = 9) -> tuple[float, float]:
    """Pull-out torque [N*m] and the angle where it occurs [rad]: the max of ``ring_torque_3d`` over one pole pitch.

    A coarse scan of ``n_scan`` interior angles is followed by a bounded Brent refinement to
    1e-4 of a pole pitch.
    """
    pitch = 2 * math.pi / pair.npole
    grid = np.linspace(0.0, pitch, n_scan + 2)
    vals = [ring_torque_3d(pair, th, panel_m, order) for th in grid[1:-1]]
    i = int(np.argmax(vals)) + 1
    opt = minimize_scalar(lambda th: -ring_torque_3d(pair, th, panel_m, order), bounds=(grid[i - 1], grid[i + 1]),
                          method="bounded", options={"xatol": 1e-4 * pitch})
    return -float(opt.fun), float(opt.x)


def torque_per_length_3d(pair: RingPair, l1_m: float, l2_m: float, theta: float, panel_m: float = 1.0 * MM,
                         order: int = 4) -> float:
    """Long-length (2D) torque per unit length [N*m/m] of the ring pair's cross-section at angle ``theta``.

    For lengths well beyond the end-fringing zone the torque is T(L) = a*L - b, where b is the
    constant end deficit. The slope (T(l2) - T(l1)) / (l2 - l1) is therefore the 2D value a,
    with the end deficit removed exactly.
    """
    t1 = ring_torque_3d(pair.with_length(l1_m), theta, panel_m, order)
    t2 = ring_torque_3d(pair.with_length(l2_m), theta, panel_m, order)
    return (t2 - t1) / (l2_m - l1_m)


def end_factor_3d(pair: RingPair, length_m: float, per_length: float, theta: float, panel_m: float = 1.0 * MM,
                  order: int = 4) -> float:
    """3D end-effect factor at ``length_m``: torque at that length / (2D torque per length * length), same angle."""
    return ring_torque_3d(pair.with_length(length_m), theta, panel_m, order) / (per_length * length_m)


# --------------------------------------------------------------------------- prototype calibration
@dataclass(frozen=True)
class PrototypeGeometry:
    """Prototype ring geometry [mm] derived from the Calibration sheet's description."""
    poles: float
    face_radius_mm: float        # inner block magnet face (flat centre)
    corner_radius_mm: float      # inner block outer corner
    corner_gap_mm: float         # inner block corner to the outer block face
    flat_gap_mm: float           # flat centre of the inner face to the flat centre of the outer face
    outer_face_apothem_mm: float
    gap_radius_mm: float
    pole_pitch_mm: float
    fill_inner: float            # block width / pole pitch at the inner block's mid-thickness radius
    fill_outer: float            # the same at the outer block's mid-thickness radius


def prototype_geometry(c) -> PrototypeGeometry:
    """Geometry from a CalibrationInputs-like object (two equal rings of identical flat blocks).

    The inner block's corner sits at hypot(face radius, width/2). The reported spacing is
    measured either at the flat centres (gap_definition 1) or at the inner corners (0). The
    flat-centre gap follows by adding the corner overhang.
    """
    poles = c.total_magnets / 2
    r_face = c.apothem_mm + c.magnet_thickness_mm
    r_corner = math.hypot(r_face, c.magnet_width_mm / 2)
    g = c.spacing_mm if c.gap_definition == 1 else c.spacing_mm + (r_corner - r_face)
    a_o = r_face + g
    r_g = r_face + g / 2
    fill_i = min(1.0, c.magnet_width_mm * poles / (2 * math.pi * (c.apothem_mm + c.magnet_thickness_mm / 2)))
    fill_o = min(1.0, c.magnet_width_mm * poles / (2 * math.pi * (a_o + c.magnet_thickness_mm / 2)))
    return PrototypeGeometry(poles, r_face, r_corner, g - (r_corner - r_face), g, a_o, r_g, 2 * math.pi * r_g / poles,
                             fill_i, fill_o)


def prototype_br(c) -> float:
    """Prototype remanence [T] at the assumed test temperature (Calibration!C16, C21, C22)."""
    return planar.br_at(c.br_T, c.alpha_br_per_C, c.test_temp_C)


def prototype_calibration(c, harmonics: Iterable[int] = (1, 3, 5)) -> dict:
    """One-point correction from a CalibrationInputs-like object.

    Model: the free-space planar shear stress at a displacement of half a pole pitch, times the
    gap area and lever arm (2 pi R_g^2 L), the end factor 1 - c_end * pitch / L, and the
    original calibration factor. Correction: measured / model, applied on top of the original
    factor. Returns the intermediate values, keyed by the engine's result-field names where one
    exists (``tau_n_Pa`` maps each harmonic n to its shear stress).
    """
    geo = prototype_geometry(c)
    br = prototype_br(c)
    t_m, pitch_m = c.magnet_thickness_mm * MM, geo.pole_pitch_mm * MM
    tau_n = {n: planar.planar_shear_stress(br, br, geo.fill_inner, geo.fill_outer, pitch_m, t_m, t_m,
                                           geo.flat_gap_mm * MM, pitch_m / 2, c.mu0, False, (n,))
             for n in harmonics}
    tau = sum(tau_n.values())
    t2d = planar.torque_on_cylinder(tau, geo.gap_radius_mm * MM, c.magnet_length_mm * MM)
    f_end = planar.end_factor(c.c_end, geo.pole_pitch_mm, c.magnet_length_mm)
    model = t2d * f_end * c.f_cal_original
    ratio = c.measured_torque_Nm / model
    return {"geometry": geo, "br_test_T": br, "tau_n_Pa": tau_n, "tau_Pa": tau, "torque_2d_Nm": t2d, "f_end": f_end,
            "model_torque_Nm": model, "model_error": model / c.measured_torque_Nm - 1,
            "measured_over_model": ratio, "f_cal_updated": c.f_cal_original * ratio}


def ring_pair_from_prototype(c) -> RingPair:
    """3D ring pair of the free-space prototype described by a CalibrationInputs-like object."""
    geo = prototype_geometry(c)
    if geo.poles != int(geo.poles) or int(geo.poles) % 2:
        raise ValueError(f"prototype needs an even whole number of poles per ring, got {geo.poles}")
    br = prototype_br(c)
    t, w, length = c.magnet_thickness_mm * MM, c.magnet_width_mm * MM, c.magnet_length_mm * MM
    return RingPair(npole=int(geo.poles), r_i_m=(c.apothem_mm + c.magnet_thickness_mm / 2) * MM, t_i_m=t, w_i_m=w,
                    l_i_m=length, j_i_T=br, r_o_m=(geo.outer_face_apothem_mm + c.magnet_thickness_mm / 2) * MM,
                    t_o_m=t, w_o_m=w, l_o_m=length, j_o_T=br)
```

- [ ] **Step 2: Write the references' own sanity tests**

`reference/magcoupling-py/audit/tests/test_torque_reference_sanity.py` tests both reference modules against cases with known answers, and does not test the engine:
- planar.py (Task 2's `square_wave_harmonic` and the functions Step 1 appended): the square-wave harmonics by quadrature, both S_n factors in the thick-array limit and in the thin-array image-series limit, and hand values.
- The 3D charge-model integrator: a uniform field (T = m x B), the dipole-dipole force, and long planar arrays against `planar_shear_stress`.
- The rings: Newton's third law, the symmetries, quadrature convergence at the default gap and at the tightest gap, and length-independence of the long-length slope.
- Cross-validation of the 3D slope against Task 2's exact 2D solution.
- The prototype geometry, and the ring builder's field mapping and input rejection.

Every test name contains `sanity`, so Task 8's coverage table can exclude these tests from the engine confirmations.

File: `reference/magcoupling-py/audit/tests/test_torque_reference_sanity.py` (create)

```python
"""Sanity tests for audit/references/planar.py and audit/references/torque_ref.py (M1, Task 3).

Each reference is checked against a case with a known answer. No engine value is under test
here, so a failure means the reference is wrong, not the engine. Every test name contains
"sanity", so Task 8's coverage table can leave these tests out of the engine confirmations.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from audit.common import MU0_EXACT, defaults, mismatch, run, vary
from audit.references import planar
from audit.references import torque_ref as tr

MM = tr.MM
REF = "reference only"

#: The workbook-default section written out by hand, so the 3D sanity tests need no engine: 10 poles, B842SH
#: blocks 3.17 mm (radial) x 6.35 mm x 12.7 mm, Br 1.29 T (20 °C), inner block backs on the 10.15 mm hub apothem
#: (faces at 13.32 mm), outer block faces 1.4 mm further out at 14.72 mm.
DEFAULT_PAIR = tr.RingPair(npole=10, r_i_m=(10.15 + 3.17 / 2) * MM, t_i_m=3.17 * MM, w_i_m=6.35 * MM, l_i_m=12.7 * MM,
                           j_i_T=1.29, r_o_m=(14.72 + 3.17 / 2) * MM, t_o_m=3.17 * MM, w_o_m=6.35 * MM, l_o_m=12.7 * MM,
                           j_o_T=1.29)
#: The same section at a 0.6 mm face gap (outer faces at 13.92 mm, corner gap 0.23 mm): the tightest 3D engine case.
TIGHT_PAIR = replace(DEFAULT_PAIR, r_o_m=(13.92 + 3.17 / 2) * MM)


# --------------------------------------------------------------------------- planar.py
@pytest.mark.family("torque")
@pytest.mark.parametrize("n, fill", [(1, 0.8), (3, 0.8), (5, 0.52), (3, 0.37)])
def test_sanity_square_wave_harmonic_by_quadrature(n, fill):
    """square_wave_harmonic vs the Fourier integral b_n = (1/pi) * integral of m(theta) cos(n theta) over one period,
    with m = +1 within fill*pi/2 of 0 and -1 within fill*pi/2 of pi. The integral uses adaptive quadrature with the
    window edges as break points; tolerance 1e-10."""
    half = fill * math.pi / 2

    def m(th: float) -> float:
        if abs(th) < half:
            return 1.0
        if abs(th - math.pi) < half:
            return -1.0
        return 0.0

    val, _ = quad(lambda th: m(th) * math.cos(n * th), -math.pi / 2, 3 * math.pi / 2,
                  points=[-half, half, math.pi - half, math.pi + half], limit=200, epsabs=1e-13)
    expected = val / math.pi
    got = planar.square_wave_harmonic(1.0, n, fill)
    assert abs(got - expected) <= 1e-10, mismatch(f"square-wave harmonic n={n}, fill={fill}", REF, got, expected, 1e-10)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backed", [False, True])
def test_sanity_planar_factors_thick_array_limit(backed):
    """Both circuits reduce to semi-infinite arrays when k t -> infinity: S_1 -> e^{-k g}/2, so at delta = pitch/2
    tau_1 = B_1^2 e^{-k g} / (4 mu0). Here k t = 20 pi, which leaves e^{-20 pi} ~ 5e-28; tolerance 1e-12."""
    p, g, br = 6.0 * MM, 1.0 * MM, 1.3
    tau = planar.planar_shear_stress(br, br, 1.0, 1.0, p, 20 * p, 20 * p, g, p / 2, MU0_EXACT, backed, (1,))
    b1 = 4 * br / math.pi
    expected = b1 * b1 * math.exp(-math.pi * g / p) / (4 * MU0_EXACT)
    assert abs(tau / expected - 1) <= 1e-12, mismatch(f"thick-array shear, backed={backed}", REF, tau, expected, 1e-12)


@pytest.mark.family("torque")
@pytest.mark.parametrize("kg", [0.2, 0.52, 1.5])
def test_sanity_iron_backed_thin_array_limit_is_image_series(kg):
    """Method of images for two thin arrays, each lying on its own permeable backing, a gap g apart. Each array's image
    in its own backing doubles its field (2 x 2 = 4). Reflections between the two backings add the geometric series
    1 / (1 - e^{-2 k g}). So S(backed) / S(free) -> 4 / (1 - e^{-2 k g}) as t -> 0. Evaluated at k t = 1e-7, where
    the first-order correction k t (2 coth(k g) - 1) is at most 9.2e-7 (observed 9.1e-7 at k g = 0.2); tolerance
    1e-5."""
    k, t = 1000.0, 1e-10
    g = kg / k
    ratio = planar.iron_backed_factor(k, t, t, g) / planar.free_space_factor(k, t, t, g)
    expected = 4 / (1 - math.exp(-2 * kg))
    assert abs(ratio / expected - 1) <= 1e-5, mismatch(f"thin-array iron gain at k g = {kg}", REF, ratio, expected, 1e-5)


@pytest.mark.family("torque")
def test_sanity_planar_helper_hand_values():
    """Hand values: Br(50 °C) = 1.29 (1 - 0.0012 * 30) = 1.24356 T; k_3 at a 6 mm pitch = 3 pi / 0.006 m;
    100 kPa on a cylinder of radius 13.7 mm and length 12.7 mm gives 1e5 * 2 pi * 0.0137^2 * 0.0127 N*m;
    end factor 1 - 0.15 * 8.62 / 12.7 = 0.898189. The first two are exact, the rest are rounding-level: 1e-12."""
    cases = [
        ("Br at 50 °C", planar.br_at(1.29, -0.0012, 50.0), 1.24356),
        ("wave number k_3", planar.wave_number(3, 0.006), 1570.7963267948965),
        ("torque on cylinder", planar.torque_on_cylinder(1e5, 0.0137, 0.0127), 1e5 * 2 * math.pi * 0.0137 ** 2 * 0.0127),
        ("end factor", planar.end_factor(0.15, 8.62, 12.7), 1 - 0.15 * 8.62 / 12.7),
    ]
    for name, got, expected in cases:
        assert abs(got / expected - 1) <= 1e-12, mismatch(name, REF, got, expected, 1e-12)


# --------------------------------------------------------------------------- torque_ref.py: charge-model integrator
@pytest.mark.family("torque")
def test_sanity_uniform_field_force_zero_and_torque_m_cross_b():
    """Charge-model integrator in a uniform field: F = 0 and T = m x B with m = J V / mu0 (textbook dipole torque).

    The integrand is constant on each face, so Gauss-Legendre is exact: tolerance 1e-12 of the scale (one face's
    force for F, |m x B| for T).
    """
    blk = tr.Block(1.0 * MM, 2.0 * MM, 0.3, 3.0 * MM, 6.0 * MM, 12.0 * MM, 1.2)
    b0 = np.array([0.05, 0.5, 0.2])
    force, torque = tr.torque_on_blocks([blk], lambda p: np.tile(b0, (len(p), 1)))
    m = blk.j_T * blk.a_m * blk.b_m * blk.c_m / MU0_EXACT * np.array([math.cos(blk.phi), math.sin(blk.phi), 0.0])
    expected = np.cross(m, b0)
    face_force = blk.j_T / MU0_EXACT * blk.b_m * blk.c_m * float(np.linalg.norm(b0))
    torque_scale = float(np.linalg.norm(expected))
    for i in range(3):
        assert abs(force[i]) <= 1e-12 * face_force, mismatch(
            f"uniform-field net force component {i} [N] (scale {face_force:.3e} N)", REF, force[i], 0.0, 1e-12)
        assert abs(torque[i] - expected[i]) <= 1e-12 * torque_scale, mismatch(
            f"uniform-field torque component {i} [N*m]", REF, torque[i], expected[i], 1e-12)


@pytest.mark.family("torque")
def test_sanity_coaxial_cubes_approach_dipole_force():
    """Two coaxial cubes (a = 2 mm, J = 1.3 T) at d = 10 a: F = -3 mu0 m^2 / (2 pi d^4) (textbook dipole-dipole).

    A uniformly magnetized cube has no quadrupole-order correction (cubic symmetry), so the finite-size error is
    O((a/d)^4) = 1e-4 (observed 1.0e-4); tolerance 1e-3. The off-axis components vanish by symmetry: 1e-9 of F.
    """
    a, d, j = 2.0 * MM, 20.0 * MM, 1.3
    src, tgt = tr.Block(0.0, 0.0, 0.0, a, a, a, j), tr.Block(d, 0.0, 0.0, a, a, a, j)
    force, _ = tr.torque_on_blocks([tgt], tr.magpylib_field([src]), panel_m=a / 4, order=6)
    m = j * a ** 3 / MU0_EXACT
    expected = -3 * MU0_EXACT * m * m / (2 * math.pi * d ** 4)
    assert abs(force[0] / expected - 1) <= 1e-3, mismatch("coaxial cube axial force [N]", REF, force[0], expected, 1e-3)
    for i in (1, 2):
        assert abs(force[i]) <= 1e-9 * abs(expected), mismatch(
            f"coaxial cube off-axis force component {i} [N] (scale {abs(expected):.3e} N)", REF, force[i], 0.0, 1e-9)


@pytest.mark.family("torque")
@pytest.mark.parametrize("delta_over_pitch", [0.5, 0.25])
def test_sanity_3d_planar_arrays_reproduce_planar_shear_stress(delta_over_pitch):
    """3D charge model on long planar alternating arrays vs planar.planar_shear_stress (all odd harmonics to 3999).

    Pitch 6 mm, fill 0.8, t = 3 mm, g = 1 mm, Br = 1.3 T, 41 source blocks. The 2D value is the slope of force vs
    length between 30 and 60 mm. Observed agreement 2.4e-6 (array truncation and slope residual); tolerance 1e-4.
    The target's own-array neighbours exert no tangential force on it by symmetry, so only the lower array is a source.
    """
    p, w, t, g, br = 6.0 * MM, 4.8 * MM, 3.0 * MM, 1.0 * MM, 1.3
    delta = delta_over_pitch * p

    def force_x(length):
        lower = [tr.Block(k * p, -t / 2, math.pi / 2, t, w, length, br * (-1) ** k) for k in range(-20, 21)]
        target = tr.Block(delta, g + t / 2, math.pi / 2, t, w, length, br)
        f, _ = tr.torque_on_blocks([target], tr.magpylib_field(lower))
        return f[0]

    tau_3d = -(force_x(60 * MM) - force_x(30 * MM)) / (30 * MM) / p
    tau_ref = planar.planar_shear_stress(br, br, w / p, w / p, p, t, t, g, delta, MU0_EXACT, False, range(1, 4001, 2))
    assert abs(tau_3d / tau_ref - 1) <= 1e-4, mismatch(
        f"planar-array shear stress at delta = {delta_over_pitch} pitch [Pa]", REF, tau_3d, tau_ref, 1e-4)


# --------------------------------------------------------------------------- torque_ref.py: rings
@pytest.mark.family("torque")
def test_sanity_ring_newton_third_law():
    """Torque on the outer ring from the inner ring = -(torque on the inner ring from the outer ring), all blocks.

    The two integrals use different faces and fields, so agreement tests the integrator; tolerance 1e-6.
    """
    pair = DEFAULT_PAIR
    theta = 0.3 * 2 * math.pi / pair.npole
    inner = tr.ring_blocks(pair.npole, pair.r_i_m, pair.t_i_m, pair.w_i_m, pair.l_i_m, pair.j_i_T, 0.0)
    outer = tr.ring_blocks(pair.npole, pair.r_o_m, pair.t_o_m, pair.w_o_m, pair.l_o_m, pair.j_o_T, theta)
    _, t_on_outer = tr.torque_on_blocks(outer, tr.magpylib_field(inner))
    _, t_on_inner = tr.torque_on_blocks(inner, tr.magpylib_field(outer))
    assert abs(t_on_outer[2] + t_on_inner[2]) <= 1e-6 * abs(t_on_outer[2]), \
        mismatch("action-reaction torque [N*m]", REF, t_on_outer[2], -t_on_inner[2], 1e-6)


@pytest.mark.family("torque")
def test_sanity_ring_symmetries():
    """Exact ring symmetries: zero torque when aligned and when like poles face, T(pitch - th) = T(th),
    T(th + pitch) = -T(th), and block 0 x N equals the sum over all outer blocks. Tolerance 1e-9 of the peak."""
    pair = DEFAULT_PAIR
    pitch = 2 * math.pi / pair.npole
    th = 0.37 * pitch
    peak = tr.ring_torque_3d(pair, pair.half_pitch)
    t_th = tr.ring_torque_3d(pair, th)
    cases = [
        ("torque when aligned (theta = 0)", tr.ring_torque_3d(pair, 0.0), 0.0),
        ("torque when like poles face (theta = pitch)", tr.ring_torque_3d(pair, pitch), 0.0),
        ("mirror symmetry T(pitch - th) vs T(th)", tr.ring_torque_3d(pair, pitch - th), t_th),
        ("antisymmetry T(th + pitch) vs -T(th)", tr.ring_torque_3d(pair, th + pitch), -t_th),
        ("block 0 x N vs all outer blocks", t_th, tr.ring_torque_3d(pair, th, all_blocks=True)),
    ]
    for name, got, expected in cases:
        assert abs(got - expected) <= 1e-9 * abs(peak), mismatch(
            f"{name} [N*m] (scale: peak {peak:.4f} N*m)", REF, got, expected, 1e-9)


@pytest.mark.family("torque")
def test_sanity_ring_quadrature_converged_default_gap():
    """Default quadrature (1 mm panels, 4 points) vs 0.25 mm panels, 6 points, at half a pole pitch and the 1.4 mm
    face gap. Observed difference 1e-7; tolerance 1e-5, far below the 2 % engine comparison bar."""
    coarse = tr.ring_torque_3d(DEFAULT_PAIR, DEFAULT_PAIR.half_pitch)
    fine = tr.ring_torque_3d(DEFAULT_PAIR, DEFAULT_PAIR.half_pitch, panel_m=0.25 * MM, order=6)
    assert abs(coarse / fine - 1) <= 1e-5, mismatch("quadrature convergence, 1.4 mm face gap [N*m]", REF, coarse, fine, 1e-5)


@pytest.mark.family("torque")
def test_sanity_ring_quadrature_converged_tightest_gap():
    """At the tightest engine case (0.6 mm face gap, 0.23 mm corner clearance) 1 mm panels are 4.8e-4 off, so the gap
    sweep uses 0.25 mm / 6-point panels. Those agree with 0.1 mm / 6-point panels to 1e-10; tolerance 1e-6."""
    fine = tr.ring_torque_3d(TIGHT_PAIR, TIGHT_PAIR.half_pitch, panel_m=0.25 * MM, order=6)
    finer = tr.ring_torque_3d(TIGHT_PAIR, TIGHT_PAIR.half_pitch, panel_m=0.1 * MM, order=6)
    assert abs(fine / finer - 1) <= 1e-6, mismatch("quadrature convergence, 0.6 mm face gap [N*m]", REF, fine, finer, 1e-6)


@pytest.mark.family("torque")
def test_sanity_long_length_slope_is_length_independent():
    """The 2D (per-length) torque from lengths 30/60 mm equals that from 60/120 mm, because the end deficit is
    constant. Observed difference 1.1e-5; tolerance 1e-4."""
    th = DEFAULT_PAIR.half_pitch
    a_short = tr.torque_per_length_3d(DEFAULT_PAIR, 30 * MM, 60 * MM, th)
    a_long = tr.torque_per_length_3d(DEFAULT_PAIR, 60 * MM, 120 * MM, th)
    assert abs(a_short / a_long - 1) <= 1e-4, mismatch("per-length torque [N*m/m]", REF, a_short, a_long, 1e-4)


@pytest.mark.family("torque")
def test_sanity_3d_long_length_slope_matches_2d_field_reference():
    """Cross-validation of two independent references on the same free-space section at half a pole pitch: the 3D
    long-length slope (magpylib fields, surface-charge force) vs Task 2's exact 2D flat-block solution
    (audit.references.field2d: bound-current sheets, Maxwell stress). The 3D slope residual is 1.1e-5 and the 2D
    free-space solution is exact to about 1e-9, so the tolerance is 1e-4.

    field2d is imported inside the test so that only this test depends on Task 2's module. It uses
    CouplingSection(n_poles, inner_back_mm, inner_thickness_mm, inner_width_mm, outer_face_mm, outer_thickness_mm,
    outer_width_mm, br_inner_T, br_outer_T) and torque_per_length(sec, theta_e), whose positive sign is restoring.
    """
    from audit.references import field2d

    sec = field2d.CouplingSection(10, 10.15, 3.17, 6.35, 14.72, 3.17, 6.35, 1.29, 1.29)
    ref_2d = field2d.torque_per_length(sec, math.pi / 2)
    per_length_3d = tr.torque_per_length_3d(DEFAULT_PAIR, 30 * MM, 60 * MM, DEFAULT_PAIR.half_pitch)
    assert abs(per_length_3d / ref_2d - 1) <= 1e-4, mismatch(
        "3D long-length slope vs Task 2 exact 2D torque per length [N*m/m]", REF, per_length_3d, ref_2d, 1e-4)


# --------------------------------------------------------------------------- torque_ref.py: prototype and builders
@pytest.mark.family("torque")
def test_sanity_prototype_geometry_hand_values():
    """Prototype geometry by hand: face radius 9.85 + 3.17 = 13.02 mm, corner radius hypot(13.02, 3.175), flat gap
    1.4 mm, corner gap 1.4 mm minus the corner overhang, outer face 14.42 mm, gap radius 13.72 mm, pitch
    2 pi 13.72 / 10, fills w N / (2 pi r_mid). Rounding-level agreement: 1e-12."""
    geo = tr.prototype_geometry(defaults().calibration)
    corner = math.sqrt(13.02 ** 2 + 3.175 ** 2)
    cases = [
        ("poles per ring", geo.poles, 10.0),
        ("face radius [mm]", geo.face_radius_mm, 13.02),
        ("corner radius [mm]", geo.corner_radius_mm, corner),
        ("flat gap [mm]", geo.flat_gap_mm, 1.4),
        ("corner gap [mm]", geo.corner_gap_mm, 1.4 - (corner - 13.02)),
        ("outer face apothem [mm]", geo.outer_face_apothem_mm, 14.42),
        ("gap radius [mm]", geo.gap_radius_mm, 13.72),
        ("pole pitch [mm]", geo.pole_pitch_mm, 2 * math.pi * 13.72 / 10),
        ("inner fill", geo.fill_inner, 6.35 * 10 / (2 * math.pi * (9.85 + 3.17 / 2))),
        ("outer fill", geo.fill_outer, 6.35 * 10 / (2 * math.pi * (14.42 + 3.17 / 2))),
    ]
    for name, got, expected in cases:
        assert abs(got - expected) <= 1e-12 * max(1.0, abs(expected)), mismatch(
            f"prototype {name}", REF, got, expected, 1e-12)


@pytest.mark.family("torque")
def test_sanity_prototype_gap_definitions_agree():
    """The same prototype described by its corner gap (definition 0) gives the same calibration as by its flat gap."""
    c = defaults().calibration
    geo = tr.prototype_geometry(c)
    as_corner = replace(c, gap_definition=0, spacing_mm=c.spacing_mm - (geo.corner_radius_mm - geo.face_radius_mm))
    a, b = tr.prototype_calibration(c), tr.prototype_calibration(as_corner)
    assert abs(a["f_cal_updated"] / b["f_cal_updated"] - 1) <= 1e-12, \
        mismatch("corner vs flat gap definition", REF, b["f_cal_updated"], a["f_cal_updated"], 1e-12)


@pytest.mark.family("torque")
def test_sanity_ring_builder_matches_hand_geometry():
    """ring_pair_from_engine at the free-space 20 °C defaults reproduces DEFAULT_PAIR (written out by hand), so the
    builder reads the right engine fields: block centre radius = face radius -/+ t/2, widths, lengths, Br."""
    inp = vary(defaults(), {"coupling.op_temp_C": 20, "coupling.backiron": 0})
    built = tr.ring_pair_from_engine(inp, run(inp))
    for name in ("r_i_m", "t_i_m", "w_i_m", "l_i_m", "j_i_T", "r_o_m", "t_o_m", "w_o_m", "l_o_m", "j_o_T"):
        got, expected = getattr(built, name), getattr(DEFAULT_PAIR, name)
        assert abs(got / expected - 1) <= 1e-12, mismatch(f"ring builder {name}", REF, got, expected, 1e-12)
    assert built.npole == DEFAULT_PAIR.npole, mismatch("ring builder npole", REF, built.npole, DEFAULT_PAIR.npole, 0.0)


@pytest.mark.family("torque")
def test_sanity_ring_builder_rejects_unsupported_geometry():
    """The 3D builder refuses arc magnets (faceted = 0) and odd pole counts instead of silently modelling them."""
    inp_arc = vary(defaults(), {"coupling.faceted": 0})
    with pytest.raises(ValueError):
        tr.ring_pair_from_engine(inp_arc, run(inp_arc))
    inp_odd = vary(defaults(), {"coupling.npole": 9})
    with pytest.raises(ValueError):
        tr.ring_pair_from_engine(inp_odd, run(inp_odd))
```

- [ ] **Step 3: Run the sanity tests**

Run: `cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_torque_reference_sanity.py -k sanity`

Expected: PASS, `24 passed`, in about 10-16 s. Tasks 1 and 2 must already be in place: Task 2's `audit/references/field2d.py` (one test cross-validates against it) and Task 2's `audit/references/planar.py`, which Step 1 appended to.

Observed agreement:
- Square-wave harmonics vs quadrature: ≤ 1e-16.
- Thick-array limit of both circuits: < 1e-12.
- Thin-array iron gain vs the image series: 9.1e-7 / 3.2e-7 / 1.2e-7 (k g = 0.2 / 0.52 / 1.5).
- Uniform-field torque: exact.
- Dipole-dipole force: 1.0e-4 at d = 10a, scaling as (a/d)^4.
- 3D planar arrays vs `planar_shear_stress`: 2.4e-6 (delta = p/2) and 8.5e-7 (p/4).
- Newton's third law: 3.8e-10. Symmetries: ≤ 3.8e-15 of the peak.
- Quadrature, 1 mm/4-point vs 0.25 mm/6-point at the 1.4 mm gap: 1.0e-7. At the 0.6 mm gap, 0.25 mm vs 0.1 mm panels: 8.5e-11 (1 mm panels would be 4.8e-4 off).
- Long-length slope, 30/60 mm vs 60/120 mm: 1.1e-5.
- 3D slope 156.7915 N·m/m vs Task 2's 2D solution 156.7898 N·m/m: 1.13e-5.
- Prototype geometry and ring builder: ≤ 1e-12.

- [ ] **Step 4: Write the engine checks**

File: `reference/magcoupling-py/audit/tests/test_torque_laws.py` (create)

```python
"""Torque model: independent checks of the engine (M1, Task 3).

Groups:
  * 3D: real magpylib 3D free-space rings vs the engine's free-space pull-out (Calculator!C93/C42), its 2D torque
    (C91), its end factor (C92), and the Calibration sheet's prototype model, correction and supplied 3D results.
  * laws: scaling and limit laws on the engine via vary(): length, Br, gap, fill, thick and thin magnets.
  * temperature: Br(T) linear at alpha_Br, pull-out ~ Br^2, the 20 °C pull-out.
  * calibration: the one-point prototype correction re-derived, C95/C96, the factor-selection rule.
  * constants: rounded mu0 vs exact.

A failing check is a CANDIDATE FINDING for Task 8, not a bug to fix here. Checks that can fail for the same reason
carry the same "Root-cause group" line in their docstrings (T3-RC1 to T3-RC6), so Task 8 can review each root cause
once. Sweeps that share one root cause are a single check that lists every point in its message.

Ownership (one owner per cell): this task owns Calculator C95, C96 and the Calibration sheet. The term-by-term
re-derivation of the torque chain C66-C94 is Task 2's (test_torque_field2d.py::test_calculator_torque_chain_rederived);
the gearbox rows C99/C100 are Task 4's (test_metal_design.py::test_gearbox_equivalents); the magnet temperature checks
C22/C32/C107/C108 are Task 5's (test_temperature_demag_adhesive.py::test_magnet_temperature_checks_follow_kj_rating);
Task 4 also owns the geometry rows C64/C65 and Metal design C8-C10 (hot/cold torque band). The 3D, law and
temperature-scaling checks here test the model behind C69-C94 by other methods (3D fields, limits, scaling across
temperature); they do not re-derive those cells.
"""
from __future__ import annotations

import dataclasses
import math
from functools import lru_cache

import numpy as np
import pytest

from audit.common import MU0_EXACT, TOL_ALGEBRA, TOL_MODEL, TOL_PEAK, defaults, mismatch, rel_err, run, vary
from audit.references import planar
from audit.references import torque_ref as tr

MM = tr.MM

#: Free-space comparison state: 20 °C (Br = 1.29 T) and no back iron (magpylib has no permeable material; the
#: restriction comes from magpylib, not from the tolerance). Relative model error does not depend on temperature,
#: because pull-out ~ Br_i*Br_o exactly (test_pullout_scales_as_br_squared_with_temperature), so comparing at 20 °C is
#: equivalent to 50 °C. The 3D reference is converged to <= 1e-5 (test_torque_reference_sanity.py), so the whole
#: deviation from it belongs to the engine and is judged against audit.common.TOL_MODEL (checks here) and TOL_PEAK.
FREE_20C = {"coupling.op_temp_C": 20, "coupling.backiron": 0}
#: Blank part numbers make the engine use the manual magnet dimensions (their defaults equal B842SH).
MANUAL = {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""}
#: Lengths for the 3D long-length (2D) slope. The end deficit is constant beyond ~20 mm (sanity test).
L_SLOPE = (30 * MM, 60 * MM)
#: Face gaps of the 3D gap-sweep check [mm]: corner gaps 0.23-4.03 mm, i.e. the engine's gap sweep (corner
#: 0.5-4 mm) plus one point below it.
GAP_SWEEP_FACE_GAPS_MM = (0.6, 0.9, 1.4, 2.4, 3.4, 4.4)
#: Face gaps beyond the sweep for the decay law [mm].
DECAY_FACE_GAPS_MM = (6.0, 10.0, 30.0, 100.0)
#: The pole sweep's pole counts (sweeps.POLE_SWEEP_POLES).
POLE_SWEEP = (6, 8, 10, 12, 14, 16)
#: Quadrature for the gap sweep: 0.25 mm / 6-point panels are converged to 1e-10 at the 0.23 mm corner clearance.
PANEL_FINE_M, ORDER_FINE = 0.25 * MM, 6
#: Corner gaps of the two supplied 3D results, from the labels of their input cells: Calibration!C47 "3D result at
#: 1.0 mm corner gap" and C48 "3D result at 1.5 mm corner gap". Those cells hold only the torques. The dataclass
#: attributes fea_gap1_mm and fea_gap2_mm are not inputs (no param() metadata) and the engine does not read them.
FEA_LABEL_GAPS_MM = (1.0, 1.5)


# --------------------------------------------------------------------------- helpers
def _assert_all_within(what: str, cells: str, rows: list[tuple[str, float, float]], tol: float) -> None:
    """One check over a sweep (one root cause, one candidate): every (label, engine, reference) row must agree within
    ``tol``. The message is mismatch() for the worst row, followed by every row's signed error."""
    errors = [rel_err(eng, ref) for _, eng, ref in rows]
    worst = max(range(len(rows)), key=errors.__getitem__)
    label, eng, ref = rows[worst]
    table = "; ".join(f"{lab}: {(e - r) / abs(r):+.3%}" if r != 0 else f"{lab}: {e:+.3e} (reference 0)"
                      for lab, e, r in rows)
    assert errors[worst] <= tol, mismatch(f"{what}; worst at {label}; all rows (engine/reference - 1): {table}",
                                          cells, eng, ref, tol)


def _cell(results_obj, field: str) -> str:
    """Workbook cell of a result field, from its field metadata."""
    for f in dataclasses.fields(results_obj):
        if f.name == field:
            return f.metadata["cell"]
    raise KeyError(f"{type(results_obj).__name__} has no field {field!r}")


@lru_cache(maxsize=None)
def _free_case(changes: tuple = ()) -> tuple:
    """(inputs, results, 3D ring pair) at FREE_20C plus ``changes``, a tuple of (path, value) pairs.

    Deliberately not an audit.common.ScenarioCache: the keys are open-ended change tuples (the pole sweep's hub
    apothem, _hub_that_fits, comes from an engine run at call time, so no fixed named table exists at import) and
    the value also carries the 3D ring pair, which _t3d and the sweeps reuse under the same key."""
    inp = vary(defaults(), {**FREE_20C, **dict(changes)})
    res = run(inp)
    return inp, res, tr.ring_pair_from_engine(inp, res)


def _engine_raw(res) -> float:
    """Engine free-space pull-out without its calibration factor: 2D torque x end factor = C93 / C42."""
    return res.model.pullout_Nm / res.model.f_cal


@lru_cache(maxsize=None)
def _t3d(changes: tuple = ()) -> float:
    """3D torque [N*m] at the engine's pull-out angle (half a pole pitch) for _free_case(changes), default quadrature."""
    _, _, pair = _free_case(changes)
    return tr.ring_torque_3d(pair, pair.half_pitch)


@lru_cache(maxsize=None)
def _default_per_length() -> float:
    """3D long-length (2D) torque per metre of the default cross-section at 20 °C, half a pole pitch."""
    _, _, pair = _free_case()
    return tr.torque_per_length_3d(pair, *L_SLOPE, pair.half_pitch)


@lru_cache(maxsize=None)
def _prototype_3d() -> float:
    """3D pull-out (max over angle) of the free-space prototype, from the Calibration inputs alone."""
    t3d, _ = tr.ring_pullout_3d(tr.ring_pair_from_prototype(defaults().calibration))
    return t3d


def _hub_that_fits(npole: int) -> float:
    """The default hub apothem where the inner blocks fit its flats. Otherwise, the smallest apothem whose flats hold
    the block plus 0.05 mm (the pole sweep's fit rule). This keeps every 3D pole case physically buildable."""
    width = run().model.inner_width_mm                                  # Calculator!C19
    return max(defaults().coupling.inner_back_apothem_mm, width / (2 * math.tan(math.pi / npole)) + 0.05)


def _pole_changes(npole: int) -> tuple:
    return (("coupling.npole", npole), ("coupling.inner_back_apothem_mm", _hub_that_fits(npole)))


# =========================================================================== 3D cross-checks
@pytest.mark.family("torque")
def test_3d_free_space_amplitude_across_gap_sweep():
    """Engine free-space pull-out without its calibration factor (C93/C42 = C91*C92) vs real 3D magpylib rings at the
    same angle (half a pole pitch), face gaps 0.6-4.4 mm (corner gaps 0.23-4.03 mm), 20 °C. Because the angle is
    matched, this measures the harmonic model's amplitude only; the angle is the peak check. The panels are
    0.25 mm / 6-point, converged to 1e-10 at the smallest gap. One check for the whole sweep. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (planar harmonic model amplitude, free space)."""
    rows = []
    for g in GAP_SWEEP_FACE_GAPS_MM:
        _, res, pair = _free_case((("metal.face_gap_mm", g),))
        ref = tr.ring_torque_3d(pair, pair.half_pitch, panel_m=PANEL_FINE_M, order=ORDER_FINE)
        rows.append((f"face gap {g} mm (corner {res.model.corner_gap_mm:.2f} mm)", _engine_raw(res), ref))
    _assert_all_within("free-space pull-out vs 3D at half a pole pitch, 20 °C, gap sweep",
                       "Calculator!C93/C42 = C91*C92 (backiron 0), Metal design!C119", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_free_space_decay_beyond_gap_sweep():
    """Limit law g -> infinity: pull-out must vanish at the physical rate. Engine ratio pull-out(g) / pull-out(1.4 mm)
    vs the same 3D ratio at face gaps 6, 10, 30 and 100 mm (free space, 20 °C, half a pole pitch). Ratios cancel the
    engine's error at the design gap. This replaces a loose "ratio < 1e-3" bound. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (the planar approximation's error grows with gap / radius)."""
    _, base_res, _ = _free_case()
    e0, t0 = _engine_raw(base_res), _t3d()
    rows = []
    for g in DECAY_FACE_GAPS_MM:
        changes = (("metal.face_gap_mm", g),)
        _, res, _ = _free_case(changes)
        rows.append((f"face gap {g} mm", _engine_raw(res) / e0, _t3d(changes) / t0))
    _assert_all_within("free-space pull-out ratio T(g)/T(1.4 mm) vs 3D ratio, 20 °C",
                       "Calculator!C93/C42 (backiron 0), Metal design!C119", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_free_space_amplitude_across_pole_sweep():
    """Engine free-space pull-out without its calibration factor vs 3D rings at half a pole pitch, for the pole sweep's
    pole counts 6-16. The hub is the default apothem where the blocks fit, otherwise the pole sweep's smallest fitting
    apothem. Face gap 1.4 mm, 20 °C, angle-matched. One check. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (planar harmonic model amplitude, free space)."""
    rows = []
    for n in POLE_SWEEP:
        _, res, _ = _free_case(_pole_changes(n))
        rows.append((f"{n} poles (hub {_hub_that_fits(n):.3f} mm)", _engine_raw(res), _t3d(_pole_changes(n))))
    _assert_all_within("free-space pull-out vs 3D at half a pole pitch, 20 °C, pole sweep",
                       "Calculator!C93/C42 = C91*C92 (backiron 0), C5, C8", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_peak_at_half_pole_pitch_across_pole_sweep():
    """The engine evaluates every harmonic at half a pole pitch (sin(n pi/2) in each tau_n); the spec names this as a
    known candidate. Half a pitch is always a stationary point by symmetry. This checks that it is the maximum: 3D
    torque at half a pitch vs the 3D peak of the torque-angle curve, poles 6-16. One check. Tolerance TOL_PEAK.
    Root-cause group: T3-RC2 (pull-out evaluated at half a pole pitch)."""
    rows = []
    for n in POLE_SWEEP:
        _, _, pair = _free_case(_pole_changes(n))
        peak, theta = tr.ring_pullout_3d(pair)
        rows.append((f"{n} poles (3D peak at {math.degrees(theta):.2f} deg, engine angle {180 / n:.2f} deg)",
                     _t3d(_pole_changes(n)), peak))
    _assert_all_within("3D torque at the engine's pull-out angle vs 3D peak (engine value = 3D at half a pitch)",
                       "Calculator!C76, C82, C88 (sin(n pi/2)) -> C93", rows, TOL_PEAK)


@pytest.mark.family("torque")
def test_3d_long_length_limit_matches_engine_torque_2d():
    """Engine 2D (infinite-length) free-space torque vs the 3D per-length torque of the same cross-section x L.

    The reference is the slope of 3D torque vs length between 30 and 60 mm, which removes the end deficit exactly
    (it agrees with Task 2's exact 2D solution to 1.1e-5, sanity test). This isolates the 2D harmonic model from the
    end factor. Tolerance TOL_MODEL.
    Root-cause group: T3-RC1 (planar harmonic model amplitude, free space).
    """
    _, res, _ = _free_case()
    reference = _default_per_length() * res.model.active_length_mm * MM
    engine = res.model.torque_2d_Nm
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        "2D free-space torque at 20 °C vs 3D long-length slope x L", "Calculator!C91 (backiron 0), C33",
        engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_end_effect_factor_across_lengths():
    """Engine end factor 1 - c_end * pitch / L vs the 3D factor T3D(L) / (2D torque per length x L), default cross-
    section, both rings of length L (manual magnets), L = 3.17-50.8 mm, at half a pole pitch. One check. Tolerance
    TOL_MODEL. Root-cause group: none expected (the empirical form's magnitude)."""
    _, _, pair = _free_case()
    rows = []
    for length_mm in (3.17, 6.35, 12.7, 25.4, 50.8):
        inp = vary(defaults(), {**FREE_20C, **MANUAL, "coupling.magnets.manual_inner_length_mm": length_mm,
                                "coupling.magnets.manual_outer_length_mm": length_mm})
        engine = run(inp).model.f_end
        reference = tr.end_factor_3d(pair, length_mm * MM, _default_per_length(), pair.half_pitch)
        rows.append((f"L = {length_mm} mm", engine, reference))
    _assert_all_within("end-effect factor vs 3D", "Calculator!C92, C41, C65, C33", rows, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_end_effect_factor_short_magnet():
    """Edge case L = 1 mm (below c_end * pitch = 1.32 mm): a physical end factor lies in (0, 1] for every L > 0.
    Engine f_end vs the 3D factor at half a pole pitch. Tolerance TOL_MODEL.
    Root-cause group: T3-RC3 (the linear end-factor form turns negative for short magnets)."""
    inp = vary(defaults(), {**FREE_20C, **MANUAL, "coupling.magnets.manual_inner_length_mm": 1.0,
                            "coupling.magnets.manual_outer_length_mm": 1.0})
    m = run(inp).model
    _, _, pair = _free_case()
    reference = tr.end_factor_3d(pair, 1.0 * MM, _default_per_length(), pair.half_pitch)
    assert rel_err(m.f_end, reference) <= TOL_MODEL, mismatch(
        f"end-effect factor at L = 1 mm (engine pull-out {m.pullout_Nm:.4f} N·m)", "Calculator!C92, C93",
        m.f_end, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_axial_overhang_effect():
    """Outer ring BX042SH (25.4 mm long) over inner B842SH (12.7 mm). The engine uses the active length min(L_i, L_o)
    (Calculator!C33), so its pull-out ratio to the equal-length case is 1. The reference is the same ratio in 3D, at
    half a pole pitch. Ratios cancel the 2D model error. Tolerance TOL_MODEL.
    Root-cause group: T3-RC4 (active length = min(L_i, L_o) ignores the overhang)."""
    changes = (("coupling.magnets.part_outer", "BX042SH"),)
    _, base_res, _ = _free_case()
    _, over_res, _ = _free_case(changes)
    engine = _engine_raw(over_res) / _engine_raw(base_res)
    reference = _t3d(changes) / _t3d()
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        "pull-out ratio, 25.4 mm outer over 12.7 mm inner vs equal lengths", "Calculator!C33, C93/C42",
        engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_prototype_raw_model():
    """Calibration sheet's original model before the 0.95 factor (2D torque x end factor) vs the 3D pull-out of the
    free-space prototype (20 x B842SH, hub apothem 9.85 mm, 1.4 mm flat gap, 20 °C), built from the Calibration inputs
    alone. Tolerance TOL_MODEL. Root-cause group: T3-RC1 (planar harmonic model amplitude, free space)."""
    cal = run().calibration
    engine = cal.torque_2d_Nm * cal.f_end
    reference = _prototype_3d()
    assert rel_err(engine, reference) <= TOL_MODEL, mismatch(
        "prototype model before the calibration factor vs 3D", "Calibration!C43*C38", engine, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_prototype_measured_correction():
    """The one-point correction C9 = 0.95 x measured / (raw model x 0.95) = measured / raw model is meant to absorb
    model error only. If it did only that, it would equal the physics-implied correction 3D pull-out / raw model.
    The difference is bench vs physics: measurement, the ASSUMED 20 °C test temperature (C16), the actual Br vs the
    nominal 1.29 T. Tolerance TOL_MODEL. Root-cause group: T3-RC5 (bench measurement vs physics; calibration premise)."""
    cal = run().calibration
    reference = _prototype_3d() / (cal.torque_2d_Nm * cal.f_end)
    assert rel_err(cal.f_cal_updated, reference) <= TOL_MODEL, mismatch(
        f"updated calibration factor vs 3D / raw model (measured {defaults().calibration.measured_torque_Nm} N·m, "
        f"3D {_prototype_3d():.4f} N·m)", "Calibration!C9, C5, C43, C38", cal.f_cal_updated, reference, TOL_MODEL)


@pytest.mark.family("torque")
def test_3d_prototype_vs_supplied_3d_results():
    """The Calibration sheet's two supplied 3D results (C47: 2.06 N·m at a 1.0 mm corner gap; C48: 1.7 N·m at 1.5 mm;
    "supplied context, not rerun") feed the interpolation C49/C50. They are compared with the 3D pull-out (max over
    angle) of the same free-space prototype at those corner gaps, built from the Calibration inputs alone. One check.
    Tolerance TOL_MODEL. Root-cause group: T3-RC6 (supplied 3D inputs vs free-space physics)."""
    c = defaults().calibration
    rows = []
    for gap_mm, supplied in zip(FEA_LABEL_GAPS_MM, (c.fea_torque1_Nm, c.fea_torque2_Nm)):
        proto = dataclasses.replace(c, gap_definition=0, spacing_mm=gap_mm)
        t3d, _ = tr.ring_pullout_3d(tr.ring_pair_from_prototype(proto))
        rows.append((f"corner gap {gap_mm} mm", supplied, t3d))
    _assert_all_within("supplied 3D prototype results (engine column) vs 3D pull-out", "Calibration!C47, C48", rows,
                       TOL_MODEL)


# =========================================================================== scaling and limit laws (engine only)
@pytest.mark.family("torque")
@pytest.mark.parametrize("length_mm", [1.0, 6.35, 25.4, 50.8])
def test_pullout_affine_in_active_length(length_mm):
    """T2D ~ L and f_end = 1 - c_end*pitch/L give T(L) = a*(L - c_end*pitch), so pull-out(L) / pull-out(12.7 mm) =
    (L - c_end*pitch) / (12.7 - c_end*pitch). Same algebra: TOL_ALGEBRA."""
    base = run(vary(defaults(), MANUAL)).model
    new = run(vary(defaults(), {**MANUAL, "coupling.magnets.manual_inner_length_mm": length_mm,
                                "coupling.magnets.manual_outer_length_mm": length_mm})).model
    c_end, pitch = defaults().coupling.c_end, base.pole_pitch_mm
    expected = base.pullout_Nm * (length_mm - c_end * pitch) / (base.active_length_mm - c_end * pitch)
    assert rel_err(new.pullout_Nm, expected) <= TOL_ALGEBRA, mismatch(
        f"pull-out at L = {length_mm} mm", "Calculator!C33, C65, C93", new.pullout_Nm, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("length_mm", [1.0, 6.35, 25.4, 50.8])
def test_torque_2d_proportional_to_active_length(length_mm):
    """2D torque = tau * 2 pi R_g^2 L is exactly proportional to L at a fixed cross-section. TOL_ALGEBRA."""
    base = run(vary(defaults(), MANUAL)).model
    new = run(vary(defaults(), {**MANUAL, "coupling.magnets.manual_inner_length_mm": length_mm,
                                "coupling.magnets.manual_outer_length_mm": length_mm})).model
    expected = base.torque_2d_Nm * length_mm / base.active_length_mm
    assert rel_err(new.torque_2d_Nm, expected) <= TOL_ALGEBRA, mismatch(
        f"2D torque at L = {length_mm} mm", "Calculator!C33, C91", new.torque_2d_Nm, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("changes", [
    {**MANUAL, "coupling.magnets.manual_inner_br_T": 1.1, "coupling.magnets.manual_outer_br_T": 1.1},
    {**MANUAL, "coupling.magnets.manual_inner_br_T": 1.1},
    {"coupling.magnets.part_inner": "B842-N52", "coupling.magnets.part_outer": "B842-N52"},
], ids=["both-rings-1.1T", "inner-only-1.1T", "library-N52"])
def test_pullout_proportional_to_br_product(changes):
    """Linear magnetostatics: every field harmonic is proportional to its ring's Br, so pull-out ~ Br_i * Br_o
    (~ Br^2 for equal rings). Steel circuit, f_cal 0.95 in every case. TOL_ALGEBRA."""
    base = run().model
    new = run(vary(defaults(), changes)).model
    expected = base.pullout_Nm * (new.inner_br_T * new.outer_br_T) / (base.inner_br_T * base.outer_br_T)
    assert rel_err(new.pullout_Nm, expected) <= TOL_ALGEBRA, mismatch(
        "pull-out vs Br_i*Br_o", "Calculator!C21, C31, C93", new.pullout_Nm, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backiron", [1, 0])
def test_pullout_decreases_monotonically_with_face_gap(backiron):
    """Field harmonics decay with distance (e^{-k g}), so pull-out must fall strictly as the face gap grows. The rate
    is checked against 3D in test_3d_free_space_decay_beyond_gap_sweep (free space only)."""
    gaps = [0.5, 0.75, 1.0, 1.4, 2.0, 3.0, 4.0, 6.0, 10.0, 30.0, 100.0]
    vals = [run(vary(defaults(), {"coupling.backiron": backiron, "metal.face_gap_mm": g})).model.pullout_Nm for g in gaps]
    for (g0, v0), (g1, v1) in zip(zip(gaps, vals), zip(gaps[1:], vals[1:])):
        assert v1 < v0, mismatch(f"pull-out at face gap {g1} mm vs {g0} mm (must be lower), backiron {backiron}",
                                 "Metal design!C119 -> Calculator!C93", v1, v0, 0.0)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backiron", [1, 0])
def test_pullout_increases_monotonically_with_fill(backiron):
    """At half a pitch the other ring's fundamental tangential field has one sign across each whole pole, so widening
    the blocks (fill 0.27-0.95, below the cap of 1) must raise pull-out. Width changes nothing else, because the outer
    face apothem is the inner face radius plus the face gap."""
    widths = [2.0, 3.0, 4.0, 5.0, 6.0, 6.35, 7.0]
    rows = [run(vary(defaults(), {**MANUAL, "coupling.backiron": backiron, "coupling.magnets.manual_inner_width_mm": w,
                                  "coupling.magnets.manual_outer_width_mm": w})).model for w in widths]
    max_fill = max(max(r.fill_inner, r.fill_outer) for r in rows)
    assert max_fill < 1, mismatch("precondition: every fill below the cap of 1 (the law holds only there)",
                                  "Calculator!C66, C67", max_fill, 1.0, 0.0)
    for (w0, r0), (w1, r1) in zip(zip(widths, rows), zip(widths[1:], rows[1:])):
        assert r1.pullout_Nm > r0.pullout_Nm, mismatch(
            f"pull-out at width {w1} mm (fill {r1.fill_inner:.3f}/{r1.fill_outer:.3f}) vs {w0} mm (must be higher), "
            f"backiron {backiron}", "Calculator!C66, C67, C93", r1.pullout_Nm, r0.pullout_Nm, 0.0)


@pytest.mark.family("torque")
@pytest.mark.parametrize("npole, n", [(60, 5), (200, 1)])
@pytest.mark.parametrize("circuit", ["iron", "free"])
def test_geometry_factor_thick_magnet_limit(npole, n, circuit):
    """Thick-magnet limit k*t -> infinity: two semi-infinite arrays give S_n = e^{-k g} / 2 with or without back iron.
    S_n depends on thickness only through k_n*t, so raising the pole count reaches k_n*t > 22 at the default
    thickness and gap. The remaining relative deviation, about 2 e^{-k t} < 6e-10, is below TOL_ALGEBRA."""
    m = run(vary(defaults(), {"coupling.npole": npole})).model
    k = getattr(m, f"k{n}")
    kt = k * min(m.inner_thickness_mm, m.outer_thickness_mm) * MM
    field = f"s{n}_{circuit}"
    cells = f"{_cell(m, field)}, {_cell(m, f'k{n}')}, Calculator!C57"
    assert kt > 22, mismatch(f"precondition: k{n}*t in the thick-magnet regime, {npole} poles", cells, kt, 22.0, 0.0)
    s = getattr(m, field)
    expected = math.exp(-k * m.face_gap_mm * MM) / 2
    assert rel_err(s, expected) <= TOL_ALGEBRA, mismatch(
        f"S{n} ({circuit}) thick-magnet limit, {npole} poles, k*t = {kt:.1f}", cells, s, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("n", [1, 3, 5])
def test_geometry_factor_thin_magnet_iron_gain(n):
    """Thin-magnet limit (t = 1e-4 mm): back iron multiplies the free-space interaction by 4 / (1 - e^{-2 k g}).
    By the method of images, each thin ring's image in its own backing doubles its field (2 x 2). Repeated
    reflections between the two iron surfaces, a distance g apart, sum to 1 / (1 - e^{-2 k g}). The first-order
    correction k(t_i + t_o)|coth(k g) - 1/2| is below 3e-4, so the tolerance is 1e-3."""
    m = run(vary(defaults(), {**MANUAL, "coupling.magnets.manual_inner_thickness_mm": 1e-4,
                              "coupling.magnets.manual_outer_thickness_mm": 1e-4})).model
    k, g = getattr(m, f"k{n}"), m.face_gap_mm * MM
    ratio = getattr(m, f"s{n}_iron") / getattr(m, f"s{n}_free")
    expected = 4 / (1 - math.exp(-2 * k * g))
    assert rel_err(ratio, expected) <= 1e-3, mismatch(
        f"S{n} iron/free ratio, thin magnets", f"{_cell(m, f's{n}_iron')}/{_cell(m, f's{n}_free')}",
        ratio, expected, 1e-3)


# =========================================================================== temperature scaling
@pytest.mark.family("torque")
@pytest.mark.parametrize("temp_C", [-40, 50, 80, 150])
@pytest.mark.parametrize("ring", ["inner", "outer"])
def test_br_linear_in_temperature(temp_C, ring):
    """Br(T) = Br20 * (1 + alpha_Br * (T - 20)), the linear reversible coefficient (planar.br_at). TOL_ALGEBRA."""
    m = run(vary(defaults(), {"coupling.op_temp_C": temp_C})).model
    alpha = defaults().calibration.alpha_br_per_C
    br_t = getattr(m, f"br_{ring}_T_op")
    expected = planar.br_at(getattr(m, f"{ring}_br_T"), alpha, temp_C)
    assert rel_err(br_t, expected) <= TOL_ALGEBRA, mismatch(
        f"{ring} Br at {temp_C} °C", f"{_cell(m, f'br_{ring}_T_op')}, Calibration!C22", br_t, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
def test_alpha_br_within_sintered_ndfeb_band():
    """Literature: sintered NdFeB datasheets give a reversible Br coefficient of about -0.09 to -0.13 %/°C
    (N42SH is typically quoted at -0.12 %/°C, e.g. in the Arnold Magnetic Technologies and K&J Magnetics grade
    tables). Check: the default alpha lies inside that band (centre -0.11 %/°C, half-width 0.02 %/°C)."""
    alpha = defaults().calibration.alpha_br_per_C
    centre, half = -0.0011, 0.0002
    assert rel_err(alpha, centre) <= half / abs(centre), mismatch(
        "alpha_Br vs sintered NdFeB band", "Calibration!C22", alpha, centre, half / abs(centre))


@pytest.mark.family("torque")
@pytest.mark.parametrize("temp_C", [-40, 80, 150])
def test_pullout_scales_as_br_squared_with_temperature(temp_C):
    """Both rings share alpha_Br, so pull-out(T) = pull-out(20 °C) * (1 + alpha (T - 20))^2. TOL_ALGEBRA.
    (This task owns the Br^2 temperature law on C93.)"""
    alpha = defaults().calibration.alpha_br_per_C
    t20 = run(vary(defaults(), {"coupling.op_temp_C": 20})).model.pullout_Nm
    t_t = run(vary(defaults(), {"coupling.op_temp_C": temp_C})).model.pullout_Nm
    expected = t20 * (planar.br_at(1.0, alpha, temp_C)) ** 2
    assert rel_err(t_t, expected) <= TOL_ALGEBRA, mismatch(
        f"pull-out at {temp_C} °C", "Calculator!C93, Calibration!C22", t_t, expected, TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("temp_C", [-40, 50, 80, 150])
def test_pullout_20C_field_equals_pullout_at_20C(temp_C):
    """The 20 °C pull-out reported at any operating temperature (rescaled by Br20^2 / Br(T)^2) equals the pull-out
    obtained by operating at 20 °C. TOL_ALGEBRA."""
    reported = run(vary(defaults(), {"coupling.op_temp_C": temp_C})).model.pullout_20C_Nm
    expected = run(vary(defaults(), {"coupling.op_temp_C": 20})).model.pullout_Nm
    assert rel_err(reported, expected) <= TOL_ALGEBRA, mismatch(
        f"20 °C pull-out reported at {temp_C} °C", "Calculator!C94", reported, expected, TOL_ALGEBRA)


# =========================================================================== calibration
_CAL_SCENARIOS = {
    "default": {},
    "corner-gap-1.0mm-35C-measured-2.0": {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.0,
                                          "calibration.test_temp_C": 35, "calibration.measured_torque_Nm": 2.0},
}
_CAL_FIELDS = [  # (engine CalibrationResults field, value from the re-derivation)
    ("poles_per_ring", lambda d: d["geometry"].poles),
    ("face_radius_mm", lambda d: d["geometry"].face_radius_mm),
    ("corner_radius_mm", lambda d: d["geometry"].corner_radius_mm),
    ("corner_gap_mm", lambda d: d["geometry"].corner_gap_mm),
    ("outer_face_apothem_mm", lambda d: d["geometry"].outer_face_apothem_mm),
    ("flat_gap_mm", lambda d: d["geometry"].flat_gap_mm),
    ("gap_radius_mm", lambda d: d["geometry"].gap_radius_mm),
    ("fill_inner", lambda d: d["geometry"].fill_inner),
    ("fill_outer", lambda d: d["geometry"].fill_outer),
    ("br_test_T", lambda d: d["br_test_T"]),
    ("pole_pitch_mm", lambda d: d["geometry"].pole_pitch_mm),
    ("f_end", lambda d: d["f_end"]),
    ("tau1_Pa", lambda d: d["tau_n_Pa"][1]),
    ("tau3_Pa", lambda d: d["tau_n_Pa"][3]),
    ("tau5_Pa", lambda d: d["tau_n_Pa"][5]),
    ("torque_2d_Nm", lambda d: d["torque_2d_Nm"]),
    ("model_torque_Nm", lambda d: d["model_torque_Nm"]),
    ("original_model_Nm", lambda d: d["model_torque_Nm"]),
    ("model_error", lambda d: d["model_error"]),
    ("measured_over_model", lambda d: d["measured_over_model"]),
    ("f_cal_updated", lambda d: d["f_cal_updated"]),
]


@pytest.mark.family("torque")
@pytest.mark.parametrize("scenario", list(_CAL_SCENARIOS))
@pytest.mark.parametrize("field, pick", _CAL_FIELDS, ids=[f[0] for f in _CAL_FIELDS])
def test_prototype_calibration_rederived(scenario, field, pick):
    """One-point prototype correction re-derived from the Calibration inputs (reference prototype_calibration:
    polygon geometry, free-space planar shear at half a pitch, 2 pi R_g^2 L, end factor, 0.95, measured/model).
    Same algebra: TOL_ALGEBRA."""
    inp = vary(defaults(), _CAL_SCENARIOS[scenario])
    cal = run(inp).calibration
    engine = getattr(cal, field)
    reference = pick(tr.prototype_calibration(inp.calibration))
    assert rel_err(engine, reference) <= TOL_ALGEBRA, mismatch(
        f"calibration {field} ({scenario})", _cell(cal, field), engine, reference, TOL_ALGEBRA)


@pytest.mark.family("torque")
def test_calculator_reproduces_prototype_model():
    """The Calculator run on the prototype layout (no iron, hub 9.85 mm, 1.4 mm gap, 20 °C) with the original 0.95
    factor gives the re-derived prototype model torque, so the two sheets agree. TOL_ALGEBRA."""
    inp = vary(defaults(), {"coupling.backiron": 0, "coupling.inner_back_apothem_mm": 9.85,
                            "metal.face_gap_mm": 1.4, "coupling.op_temp_C": 20})
    m = run(inp).model
    engine = m.pullout_Nm / m.f_cal * inp.calibration.f_cal_original
    reference = tr.prototype_calibration(inp.calibration)["model_torque_Nm"]
    assert rel_err(engine, reference) <= TOL_ALGEBRA, mismatch(
        "Calculator on the prototype layout x 0.95", "Calculator!C93/C42 vs Calibration!C6", engine, reference,
        TOL_ALGEBRA)


@pytest.mark.family("torque")
@pytest.mark.parametrize("changes", [
    {},
    {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.3},
    {"calibration.gap_definition": 0, "calibration.spacing_mm": 1.5},
    {"calibration.gap_definition": 0, "calibration.spacing_mm": 0.8},
    {"calibration.fea_torque1_Nm": 2.3, "calibration.fea_torque2_Nm": 1.5},
], ids=["default", "corner-1.3mm", "corner-1.5mm-span-end", "corner-0.8mm-outside", "other-3D-torques"])
def test_fea_interpolation(changes):
    """Linear interpolation of the two supplied 3D results at the prototype corner gap, and its error vs the
    measurement. Off the 1.0-1.5 mm span (ends included) the engine reports 'outside range' / 'n.a.'. The reference is
    numpy.interp over FEA_LABEL_GAPS_MM with the fea_torque inputs. TOL_ALGEBRA."""
    inp = vary(defaults(), changes)
    c, cal = inp.calibration, run(inp).calibration
    cg = tr.prototype_geometry(c).corner_gap_mm
    if FEA_LABEL_GAPS_MM[0] <= cg <= FEA_LABEL_GAPS_MM[1]:
        ref = float(np.interp(cg, FEA_LABEL_GAPS_MM, [c.fea_torque1_Nm, c.fea_torque2_Nm]))
        assert rel_err(cal.fea_interp_Nm, ref) <= TOL_ALGEBRA, mismatch(
            f"3D interpolation at corner gap {cg:.4f} mm", "Calibration!C49, C47, C48", cal.fea_interp_Nm, ref,
            TOL_ALGEBRA)
        ref_err = ref / c.measured_torque_Nm - 1
        assert rel_err(cal.fea_interp_error, ref_err) <= TOL_ALGEBRA, mismatch(
            "3D interpolation error vs measured", "Calibration!C50, C5", cal.fea_interp_error, ref_err, TOL_ALGEBRA)
    else:
        ok = (cal.fea_interp_Nm, cal.fea_interp_error) == ("outside range", "n.a.")
        assert ok, mismatch(
            f"3D interpolation off the span at corner gap {cg:.4f} mm: engine {cal.fea_interp_Nm!r} / "
            f"{cal.fea_interp_error!r}, expected 'outside range' / 'n.a.' (1 = match)", "Calibration!C49, C50",
            float(ok), 1.0, 0.0)


@pytest.mark.family("torque")
@pytest.mark.parametrize("backiron", [1, 0])
@pytest.mark.parametrize("field, backed", [("pullout_iron_Nm", True), ("pullout_noiron_Nm", False)],
                         ids=["steel-C95", "free-C96"])
def test_circuit_variant_pullouts_rederived(backiron, field, backed):
    """C95 (same layout, steel circuit, original 0.95 factor) and C96 (same layout, free space, the selected factor
    C42), re-derived end to end from the engine's geometry fields with planar.planar_shear_stress: shear at half a
    pitch (harmonics 1, 3, 5) x 2 pi R_g^2 L x f_end x factor. Same algebra: TOL_ALGEBRA."""
    inp = vary(defaults(), {"coupling.backiron": backiron})
    m = run(inp).model
    tau = planar.planar_shear_stress(m.br_inner_T_op, m.br_outer_T_op, m.fill_inner, m.fill_outer, m.pole_pitch_mm * MM,
                                     m.inner_thickness_mm * MM, m.outer_thickness_mm * MM, m.face_gap_mm * MM,
                                     m.pole_pitch_mm * MM / 2, inp.coupling.mu0, backed, (1, 3, 5))
    factor = inp.calibration.f_cal_original if backed else m.f_cal
    expected = (planar.torque_on_cylinder(tau, m.gap_radius_mm * MM, m.active_length_mm * MM)
                * planar.end_factor(inp.coupling.c_end, m.pole_pitch_mm, m.active_length_mm) * factor)
    engine = getattr(m, field)
    assert rel_err(engine, expected) <= TOL_ALGEBRA, mismatch(
        f"{field}, backiron {backiron}", f"{_cell(m, field)}, Calculator!C42, C64-C70, C92", engine, expected,
        TOL_ALGEBRA)


_RULE_CASES = [  # (label, changes, measured correction expected?)
    ("steel-circuit", {}, False),
    ("prototype-circuit", {"coupling.backiron": 0}, True),
    ("no-iron-8-poles", {"coupling.backiron": 0, "coupling.npole": 8}, False),
    ("no-iron-inner-B842", {"coupling.backiron": 0, "coupling.magnets.part_inner": "B842"}, False),
    ("no-iron-outer-B842", {"coupling.backiron": 0, "coupling.magnets.part_outer": "B842"}, False),
    ("no-iron-8-poles-8-pole-prototype", {"coupling.backiron": 0, "coupling.npole": 8, "calibration.total_magnets": 16},
     True),
]


@pytest.mark.family("torque")
@pytest.mark.parametrize("label, changes, measured", _RULE_CASES, ids=[c[0] for c in _RULE_CASES])
def test_calibration_factor_selection_rule(label, changes, measured):
    """README / model docstring rule: the measured correction applies only to the prototype's circuit (no back iron,
    the same pole count as the prototype, B842SH in both rings); otherwise the original 0.95 applies. The expected
    value comes from the re-derived correction. TOL_ALGEBRA."""
    inp = vary(defaults(), changes)
    engine = run(inp).model.f_cal
    expected = tr.prototype_calibration(inp.calibration)["f_cal_updated"] if measured else inp.calibration.f_cal_original
    assert rel_err(engine, expected) <= TOL_ALGEBRA, mismatch(
        f"calibration factor, {label}", "Calculator!C42, Calibration!C9, C24", engine, expected, TOL_ALGEBRA)


# =========================================================================== constants
@pytest.mark.family("constants")
def test_pullout_inversely_proportional_to_mu0():
    """tau = B_i B_o / (2 mu0) * ..., so pull-out(exact mu0) / pull-out(rounded) = MU0_rounded / MU0_exact.
    TOL_ALGEBRA."""
    rounded = run().model.pullout_Nm
    exact = run(vary(defaults(), {"coupling.mu0": MU0_EXACT})).model.pullout_Nm
    expected = rounded * defaults().coupling.mu0 / MU0_EXACT
    assert rel_err(exact, expected) <= TOL_ALGEBRA, mismatch(
        "pull-out with exact mu0", "Calculator!C43, C93", exact, expected, TOL_ALGEBRA)


@pytest.mark.family("constants")
def test_rounded_mu0_effect_on_pullout_is_negligible():
    """Rounded MU0 = 1.256637e-6 vs 4 pi 1e-7: the effect on pull-out must be negligible, i.e. below 1e-6. For
    comparison, the Calculator shows 4 significant figures (5e-4) and the 3D comparison bar is 2e-2."""
    rounded = run().model.pullout_Nm
    exact = run(vary(defaults(), {"coupling.mu0": MU0_EXACT})).model.pullout_Nm
    assert rel_err(rounded, exact) < 1e-6, mismatch("pull-out, rounded vs exact mu0", "Calculator!C43, C93",
                                                    rounded, exact, 1e-6)


@pytest.mark.family("constants")
def test_calibrated_noiron_pullout_independent_of_mu0():
    """With the measured correction in use (prototype circuit), the correction absorbs mu0: changing mu0 in both
    sheets leaves the calibrated pull-out unchanged. TOL_ALGEBRA."""
    base = run(vary(defaults(), {"coupling.backiron": 0})).model.pullout_Nm
    exact = run(vary(defaults(), {"coupling.backiron": 0, "coupling.mu0": MU0_EXACT,
                                  "calibration.mu0": MU0_EXACT})).model.pullout_Nm
    assert rel_err(exact, base) <= TOL_ALGEBRA, mismatch(
        "calibrated no-iron pull-out, exact vs rounded mu0 in both sheets", "Calculator!C43, Calibration!C25, C93",
        exact, base, TOL_ALGEBRA)
```

- [ ] **Step 5: Run the checks**

Run: `cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_torque_laws.py`

Expected: `10 failed, 100 passed`, in about 9 s (110 checks: 107 `torque`, 3 `constants`; all 10 FAILs are `torque`). The FAILs are **candidate findings for Task 8**, not bugs to fix now. Do not change the engine and do not loosen a tolerance. Each FAIL is listed with its root-cause group; the 10 FAILs reduce to 6 root causes: T3-RC1 (FAILs 1, 2, 3, 5, 8), T3-RC2 (4), T3-RC3 (6), T3-RC4 (7), T3-RC5 (9) and T3-RC6 (10).

FAIL (observed; each one is a candidate finding for Task 8). Nine are judged against `audit.common.TOL_MODEL` (2 %), FAIL 4 against `audit.common.TOL_PEAK` (1e-3):
1. `test_3d_free_space_amplitude_across_gap_sweep` [T3-RC1]:
   - Worst row: face gap 4.4 mm, engine 0.65419 vs 3D 0.61134 N·m (rel_err 7.01e-2).
   - All rows: 0.6 mm -6.89 %; 0.9 mm -5.54 %; 1.4 mm -3.58 %; 2.4 mm -0.10 %; 3.4 mm +3.33 %; 4.4 mm +7.01 %.
2. `test_3d_free_space_decay_beyond_gap_sweep` [T3-RC1]:
   - Engine vs 3D ratio T(g)/T(1.4 mm): 6 mm +17.9 %; 10 mm +40.2 %; 30 mm +299 %; 100 mm +7101 % (engine 1.303e-4 vs 3D 1.810e-6).
3. `test_3d_free_space_amplitude_across_pole_sweep` [T3-RC1]:
   - Worst row: 8 poles, engine 1.11600 vs 3D 1.16356 N·m (-4.09 %).
   - Other rows: 6 +3.04 %; 10 -3.58 %; 12 -2.34 %; 14 -1.24 %; 16 -0.39 %.
4. `test_3d_peak_at_half_pole_pitch_across_pole_sweep` [T3-RC2, the spec's known candidate]:
   - 6 poles: 3D torque at the engine's angle (30°) is 0.32855 N·m vs the 3D peak of 0.53678 N·m at 15.74° (-38.8 %).
   - 8-16 poles: 0.000 %.
5. `test_3d_long_length_limit_matches_engine_torque_2d` [T3-RC1]: C91 1.91279 vs 3D 1.99125 N·m (-3.94 %).
6. `test_3d_end_effect_factor_short_magnet` [T3-RC3]: at L = 1 mm, f_end -0.32135 vs 3D 0.24061. The engine pull-out is -0.0460 N·m.
7. `test_3d_axial_overhang_effect` [T3-RC4]: engine ratio 1.000 vs 3D 1.12576 (rel_err 1.12e-1).
8. `test_3d_prototype_raw_model` [T3-RC1]: Calibration C43·C38 1.68890 vs 3D 1.75538 N·m (-3.79 %).
9. `test_3d_prototype_measured_correction` [T3-RC5]: C9 1.06578 vs the physics-implied 3D/raw 1.03936 (+2.54 %). Equivalently, the bench value of 1.8 N·m is 2.54 % above the 3D value of 1.7554 N·m.
10. `test_3d_prototype_vs_supplied_3d_results` [T3-RC6]: the supplied 3D results are +16.5 % and +16.1 % above free-space 3D. C47 is 2.06 vs 3D 1.76766 N·m at a 1.0 mm corner gap, and C48 is 1.7 vs 3D 1.46488 N·m at 1.5 mm.

PASS (observed numbers):
- `test_3d_end_effect_factor_across_lengths`: C92 vs 3D at L = 3.17 / 6.35 / 12.7 / 25.4 / 50.8 mm is +0.80 % / +1.83 % / +0.37 % / +0.10 % / +0.05 %. The c_end implied by 3D is 0.152-0.160, against the input 0.15.
- `test_prototype_calibration_rederived`: 21 fields × 2 scenarios, ≤ 3.3e-15. f_cal_updated is 1.0657837 (default) and 1.2203296 (the corner-gap scenario).
- Exact to ≤ 5e-15:
  - Scaling laws: pull-out affine in L; T2D ∝ L; pull-out ∝ Br_i·Br_o (3 cases).
  - Temperature: Br(T) linear (8 items); pull-out ∝ Br² with temperature (3); C94 equals a direct 20 °C run (4).
  - Calibration cross-checks: the Calculator on the prototype layout equals Calibration!C6; C95/C96 (4); the calibration-factor selection rule (6).
  - Constants: pull-out ∝ 1/µ0; the calibrated no-iron pull-out is independent of µ0.
- 3D interpolation (5 cases): exact, e.g. 2.0467021 N·m at the default corner gap of 1.0185 mm, and 'outside range' / 'n.a.' at 0.8 mm.
- Thick-magnet limit: ≤ 3.7e-15, except S1 free at 200 poles, which is 3.0e-10.
- Thin-magnet iron gain: -1.2e-4 / -1.5e-4 / -2.3e-4 (n = 1 / 3 / 5).
- Monotonic laws: pull-out falls monotonically with gap (0.5-100 mm) and rises monotonically with fill. alpha_Br lies inside the NdFeB band.
- Rounded µ0 changes the pull-out by 4.89e-8, which is negligible.

- [ ] **Step 6: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 add reference/magcoupling-py/audit/references/planar.py reference/magcoupling-py/audit/references/torque_ref.py reference/magcoupling-py/audit/tests/test_torque_reference_sanity.py reference/magcoupling-py/audit/tests/test_torque_laws.py
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 commit -F - <<'EOF'
test(magcoupling-audit): torque independent checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 4: Geometry, mass/inertia, metal design and materials: independent checks

**Files:**
- Create: `reference/magcoupling-py/audit/references/block_geometry.py`
- Create: `reference/magcoupling-py/audit/references/slab_field2d.py` (shared two-plate slab field model; Task 5 appends its force and energy functions)
- Create: `reference/magcoupling-py/audit/references/backiron_flux.py`
- Create: `reference/magcoupling-py/audit/references/remanence.py` (the shared torque ∝ Br² law, `br_ratio` and `torque_at`; it re-exports `br_at` from `audit.references.planar`; Tasks 5 and 6 import it)
- Create: `reference/magcoupling-py/audit/references/metal_stack.py`
- Create: `reference/magcoupling-py/audit/tests/test_geometry_metal_reference_sanity.py` (sanity tests for all five references)
- Create: `reference/magcoupling-py/audit/tests/test_geometry_mass.py` (family `geometry`)
- Create: `reference/magcoupling-py/audit/tests/test_metal_design.py` (families `metal`, `torque`)
- Create: `reference/magcoupling-py/audit/tests/test_materials.py` (family `materials`)

The scope covers three workbook sheets, so this task has five reference modules and three check files. Steps 1, 2 and 4 show every file in full. Dependencies beyond the harness: `numpy` and `scipy` (both in `requirements-audit.txt`).

**Shared modules.** `slab_field2d.py` holds the field half of the slab model first drafted for Task 5 (`MagnetLayer`, `_sheets`, `_ratio`, `harmonic_coefficients`, verbatim except that the square-wave coefficient is factored into `_square_wave`, which evaluates `planar.square_wave_harmonic` order by order instead of repeating the closed form; the values are bitwise identical), plus the new `iron_face_coefficients`. Task 5 appends `_MU0`, `strip_stress_N_per_m`, `layer_block_forces_N` and `field_energy_J_per_m` to this file (its `(append)` block) instead of creating it; the `(create)` block below is only this task's part. `remanence.py` does not define Br(T): the audit's single implementation is `br_at` in `audit/references/planar.py` (Task 3's append to Task 2's file), and `remanence.py` re-exports it with `from audit.references.planar import br_at` next to `br_ratio` and `torque_at`. Likewise `slab_field2d.py` does not define the square-wave harmonic amplitude: `_square_wave` calls `square_wave_harmonic` in `audit/references/planar.py` (Task 2's part of that file). Both modules therefore need Tasks 2 and 3 integrated first, which the task order guarantees.

**Cell ownership.** One task checks each cell:
- This task owns Metal design C8–C10, C112 and C157 (torque band), C86/C89/C91/C92 and Calculator C97 (slip duty), Calculator C110–C115 (mass), Metal design C141/C143 (clearance), and Calculator C99 and C100 (gearbox equivalents; handed over by Task 3, whose ratio 7 / efficiency 0.8 / free-space scenario now runs here as the second `GEAR_SCENARIOS` entry, so `test_gearbox_equivalents` is the sole owner of both cells).
- It leaves the Br² temperature law on Calculator C93 to Task 3, and the Metal design C11/C37 verdict thresholds to Task 7. The values those verdicts compare (C9, C12, C35) are checked here.
- It does not check the magnet temperature ratings, Calculator C22, C32, C107 and C108: Task 5 owns them (`test_temperature_demag_adhesive.py::test_magnet_temperature_checks_follow_kj_rating`), which absorbed this task's manual-magnet `n/a` case and its N42-outer-ring-at-100 °C case.

**Interfaces:**
- Consumes: the `audit.common` helpers `defaults`, `run`, `vary`, `rel_err`, `mismatch` and the tolerances `TOL_ALGEBRA` and `TOL_MODEL`, all supplied by Task 1 (this task neither creates nor edits `audit/common.py`). `MU0_EXACT` is not used: no formula in this scope involves µ0, and the slab model works in T and mm. `test_materials.py` imports `TOL_MODEL` (2 %) from `audit.common` and defines no tolerance constant of its own. The only other task's module used is `audit.references.planar` (Task 2's file plus Task 3's append): `br_at` (Task 3's append), reached through `remanence.py`'s re-export, and `square_wave_harmonic` (Task 2's part), called by `slab_field2d._square_wave`. Engine result fields used:
  - `model.`: `inner_length_mm`, `inner_width_mm`, `inner_thickness_mm`, `inner_br_T`, `outer_length_mm`, `outer_width_mm`, `outer_thickness_mm`, `outer_br_T` (C18–C21, C28–C31, data only), `active_length_mm` (C33), `corner_gap_mm` (C9), `alpha_br_per_C` (C35), `bsat_T` (C36), `cup_wall_corner_mm` (C37), `hub_wall_mm` (C38), `inner_flat_width_mm` (C51), `inner_flat_check` (C52), `hub_wall_past_key_mm` (C53), `inner_face_radius_mm` (C54), `inner_corner_radius_mm` (C55), `outer_face_apothem_mm` (C56), `face_gap_mm` (C57), `outer_flat_width_mm` (C58), `outer_flat_check` (C59), `outer_back_apothem_mm` (C60), `pocket_corner_radius_mm` (C61), `cup_od_mm` (C62), `cup_wall_flat_mm` (C63), `gap_radius_mm` (C64), `pole_pitch_mm` (C65), `fill_inner` (C66), `fill_outer` (C67), `pullout_Nm` (C93), `pullout_20C_Nm` (C94), `pullout_iron_Nm` (C95, message only), `pullout_noiron_Nm` (C96), `ripple_freq_Hz` (C97), `gearbox_input_ripple_Nm` (C99), `gearbox_reference_Nm` (C100), `gap_flux_density_T` (C103), `backiron_needed_mm` (C104), `cup_ring_check` (C105), `hub_check` (C106).
  - `mass.`: `magnets_g`, `cup_g`, `hub_g`, `boss_g`, `total_g`, `added_inertia_kgm2` (C110–C115).
  - `retainers.*`: every field (Metal design C45, C46, C175–C181).
  - `metal.*`: every field of `MetalDesignResults` except `hot_min_check` (C11) and `clearance_check` (C37), which Task 7 owns.
  - `materials.*`: every field of `MaterialsResults` (Materials C20–C22, C27–C30).
  - Inputs read: `coupling.*` (including `coupling.magnets.part_inner` and `part_outer`), `metal.*`, `calibration.alpha_br_per_C`, `calibration.measured_torque_Nm`, `calibration.test_temp_C`, `materials.steel.*`, `materials.nickel.thickness_mm`, `materials.aluminium.*`, `materials.screws.*`.
- Produces (reusable; none of them imports `magcoupling`):
  - `audit.references.block_geometry`:
    - `Point = tuple[float, float]`
    - `NDFEB_DENSITY_G_MM3 = 7.5e-3`
    - `regular_polygon(apothem: float, n: int, phase: float = 0.0) -> list[Point]`
    - `polygon_area(pts: list[Point]) -> float`
    - `polygon_polar_moment(pts: list[Point]) -> float`
    - `side_length(pts: list[Point]) -> float`
    - `point_segment_distance(p: Point, a: Point, b: Point) -> float`
    - `convex_polygon_distance(P: list[Point], Q: list[Point]) -> float`
    - `axial_overlap(length_a: float, length_b: float, offset: float = 0.0) -> float`
    - `max_radius(P: list[Point]) -> float`
    - `min_radius(P: list[Point]) -> float`
    - `block_rectangle(near_radius: float, thickness: float, width: float, angle: float) -> list[Point]`
    - `ring_min_clearance(npole: int, inner_back: float, t_i: float, w_i: float, outer_face: float, t_o: float, w_o: float, scan: int = 360) -> float`
    - `CouplingGeometry` (frozen dataclass)
    - `coupling_geometry(npole, inner_back_apothem, t_i, w_i, t_o, w_o, face_gap, bond_inner, bond_outer, cup_wall_corner, bore, keyway_depth, faceted=True) -> CouplingGeometry`
    - `Part(mass_g: float, polar_inertia_g_mm2: float)`, which supports `+`
    - `prism_part(area, polar_moment, length, rho) -> Part`
    - `polygon_part(pts, length, rho) -> Part`
    - `disk_section(diameter) -> tuple[float, float]`
    - `annulus_part(od, id_, length, rho) -> Part`
    - `rotating_parts(npole, inner_back_apothem, t_i, w_i, L_i, outer_face_apothem, t_o, w_o, L_o, pocket_apothem, hub_apothem, cup_od, bore, cup_depth, web, hub_length, boss_length, boss_od, cup_density, hub_density, faceted=True, magnet_density=NDFEB_DENSITY_G_MM3) -> dict[str, Part]`
    - `RetainerGeometry` (frozen dataclass)
    - `retainer_geometry(inner_ring_max_radius, outer_ring_min_radius, sleeve, liner, sleeve_bedding, liner_bedding) -> RetainerGeometry`
    - `retainer_parts(g, span, retainer_density, cap_od, cap_face, cap_thread_dia, cap_thread_engagement, al_density, bore, front_endplate, rear_endplate, rear_hole) -> dict[str, Part]`
    - Adapters: `geometry_for_design(inp, res) -> CouplingGeometry`, `rotating_parts_for_design(inp, res, cup_density: float | None = None) -> dict[str, Part]`, `retainer_geometry_for_design(inp, res) -> RetainerGeometry`, `retainer_parts_for_design(inp, res) -> dict[str, Part]`
  - `audit.references.slab_field2d` (shared):
    - `MagnetLayer(y_bottom_mm: float, thickness_mm: float, br_T: float, fill: float, shift_mm: float = 0.0)` (frozen dataclass)
    - `harmonic_coefficients(y_mm: float, layers: list[MagnetLayer], h_mm: float, pitch_mm: float, n_max: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]`, returning `(n, A, B, C, D)`
    - `iron_face_coefficients(surface: str, layers: list[MagnetLayer], h_mm: float, pitch_mm: float, n_max: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray]`, returning `(n, A, B)`
  - `audit.references.backiron_flux`:
    - `flat_circuit_flux_density(br_i: float, br_o: float, t_i: float, t_o: float, gap: float) -> float`
    - `sinusoidal_backiron_thickness(b_peak: float, pole_pitch: float, b_design: float) -> float`
    - `coupling_slab(br_i: float, br_o: float, fill_i: float, fill_o: float, t_i: float, t_o: float, gap: float) -> tuple[list[MagnetLayer], float]`
    - `backiron_flux_per_depth(surface: str, layers: list[MagnetLayer], h_mm: float, pitch_mm: float, n_max: int = 4001) -> float`
  - `audit.references.remanence` (shared):
    - `br_at(br20_T: float, alpha_per_C: float, temp_C: float) -> float`, re-exported from `audit.references.planar` (the single implementation)
    - `br_ratio(alpha_per_C: float, temp_C: float, ref_C: float = 20.0) -> float`
    - `torque_at(torque_ref_Nm: float, alpha_per_C: float, ref_C: float, temp_C: float) -> float`
  - `audit.references.metal_stack`:
    - `torque_low(torque_Nm: float, variation: float) -> float`
    - `torque_high(torque_Nm: float, variation: float) -> float`
    - `gearbox_input_torque(output_Nm: float, ratio: float, efficiency: float) -> float`
    - `gearbox_output_torque(input_Nm: float, ratio: float, efficiency: float) -> float`
    - `sleeve_liner_clearance(corner_gap: float, sleeve: float, liner: float, sleeve_bedding: float, liner_bedding: float) -> float`
    - `linear_stack(*items: float) -> float`
    - `pole_pairs(npole: int) -> float`
    - `pole_pair_frequency_Hz(npole: int, rpm: float) -> float`
    - `omega_rad_s(rpm: float) -> float`
    - `shaft_power_W(torque_Nm: float, rpm: float) -> float`
    - `plated_size(size: float, thickness: float, surfaces: int, external: bool) -> float`
    - `preplate_size(finished: float, thickness: float, surfaces: int, external: bool) -> float`

- [ ] **Step 1: Write the reference modules**

File: `reference/magcoupling-py/audit/references/block_geometry.py` (create)

```python
"""Independent flat-block coupling geometry, solid masses and polar inertia (M1 audit, Task 4).

Written from first principles; imports nothing from ``magcoupling``. Shapes are
built explicitly (block rectangles, pocket polygons from intersecting side
lines) and measured with generic tools (vertex radii, point-to-segment
distances, shoelace area and polar moment). The only shared content with the
workbook is the *definition* of each quantity, so a transcription or algebra
slip in the workbook shows up as a mismatch.

Conventions: lengths mm, areas mm², densities g/mm³, masses g. ``Part`` holds a
mass and its polar moment of inertia about the coupling axis in g·mm²
(1 g·mm² = 1e-9 kg·m²). Block 0 of each ring and side 0 of each polygon are
centred on the +x axis.

The ``*_for_design`` adapters at the bottom only read input/result attributes by
name (duck typing) so tests can build a reference from a ``DesignInputs`` object;
they contain no formulas beyond the calls into the functions above them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import minimize_scalar

Point = tuple[float, float]

#: Sintered NdFeB density, 7.5 g/cm³ (datasheets quote 7.4–7.7 g/cm³, e.g. Arnold
#: Magnetic Technologies N42SH, K&J Magnetics "Magnet specifications").
NDFEB_DENSITY_G_MM3 = 7.5e-3


# --------------------------------------------------------------------------- 2D primitives
def regular_polygon(apothem: float, n: int, phase: float = 0.0) -> list[Point]:
    """Vertices (counter-clockwise) of a regular n-gon whose side k lies on the line
    x·cos(θk) + y·sin(θk) = apothem, θk = phase + 2πk/n. Vertex k is the intersection
    of sides k and k+1, solved by Cramer's rule (no circumradius formula used)."""
    pts: list[Point] = []
    for k in range(n):
        t1 = phase + 2 * math.pi * k / n
        t2 = phase + 2 * math.pi * (k + 1) / n
        det = math.cos(t1) * math.sin(t2) - math.sin(t1) * math.cos(t2)
        x = apothem * (math.sin(t2) - math.sin(t1)) / det
        y = apothem * (math.cos(t1) - math.cos(t2)) / det
        pts.append((x, y))
    return pts


def polygon_area(pts: list[Point]) -> float:
    """Shoelace area (absolute value) of a simple polygon."""
    s = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def polygon_polar_moment(pts: list[Point]) -> float:
    """Polar second moment of area about the origin, J_O = ∬(x² + y²) dA, from the
    standard vertex formula Σ c_i·(x_i² + x_i·x_j + x_j² + y_i² + y_i·y_j + y_j²)/12 with
    c_i = x_i·y_j − x_j·y_i (absolute value, so vertex order does not matter)."""
    s = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        c = x1 * y2 - x2 * y1
        s += c * (x1 * x1 + x1 * x2 + x2 * x2 + y1 * y1 + y1 * y2 + y2 * y2)
    return abs(s) / 12


def side_length(pts: list[Point]) -> float:
    """Length of the first side (vertex 0 to vertex 1) of a polygon."""
    return math.dist(pts[0], pts[1])


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from point p to the closed segment ab."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    s = 0.0 if L2 == 0.0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + s * dx), p[1] - (ay + s * dy))


def convex_polygon_distance(P: list[Point], Q: list[Point]) -> float:
    """Minimum distance between two disjoint convex polygons: the smallest
    vertex-to-edge distance taken both ways (exact for non-intersecting convex sets)."""
    best = math.inf
    for A, B in ((P, Q), (Q, P)):
        for v in A:
            for i in range(len(B)):
                best = min(best, point_segment_distance(v, B[i], B[(i + 1) % len(B)]))
    return best


def axial_overlap(length_a: float, length_b: float, offset: float = 0.0) -> float:
    """Overlap of two axial intervals: [−a/2, a/2] and [offset − b/2, offset + b/2]."""
    lo = max(-length_a / 2, offset - length_b / 2)
    hi = min(length_a / 2, offset + length_b / 2)
    return max(0.0, hi - lo)


def max_radius(P: list[Point]) -> float:
    """Largest distance from the axis to the polygon (attained at a vertex)."""
    return max(math.hypot(x, y) for x, y in P)


def min_radius(P: list[Point]) -> float:
    """Smallest distance from the axis to the polygon boundary (polygon not containing the axis)."""
    return min(point_segment_distance((0.0, 0.0), P[i], P[(i + 1) % len(P)]) for i in range(len(P)))


def block_rectangle(near_radius: float, thickness: float, width: float, angle: float) -> list[Point]:
    """Flat block centred on the ray at polar angle ``angle``: its near face is the chord at
    distance ``near_radius`` from the axis, it extends ``thickness`` radially outward and
    ``width`` tangentially. Vertices counter-clockwise."""
    er = (math.cos(angle), math.sin(angle))
    et = (-math.sin(angle), math.cos(angle))
    local = ((near_radius, -width / 2), (near_radius + thickness, -width / 2),
             (near_radius + thickness, width / 2), (near_radius, width / 2))
    return [(r * er[0] + u * et[0], r * er[1] + u * et[1]) for r, u in local]


def ring_min_clearance(npole: int, inner_back: float, t_i: float, w_i: float,
                       outer_face: float, t_o: float, w_o: float, scan: int = 360) -> float:
    """Smallest 2D distance between the inner and outer block rings over every relative rotation.

    Inner block 0 sits at angle 0 (all inner blocks are equivalent by symmetry); outer block k
    sits at φ + 2πk/N. φ is scanned over one pole pitch, then refined with a bounded Brent
    search (xatol 1e-12 rad) around the best sample."""
    inner = block_rectangle(inner_back, t_i, w_i, 0.0)
    pitch = 2 * math.pi / npole

    def clearance(phi: float) -> float:
        return min(convex_polygon_distance(inner, block_rectangle(outer_face, t_o, w_o, phi + pitch * k))
                   for k in range(npole))

    phis = [pitch * j / scan for j in range(scan)]
    vals = [clearance(p) for p in phis]
    j = min(range(scan), key=vals.__getitem__)
    step = pitch / scan
    res = minimize_scalar(clearance, bounds=(phis[j] - step, phis[j] + step), method="bounded",
                          options={"xatol": 1e-12})
    return min(float(res.fun), vals[j])


# --------------------------------------------------------------------------- coupling geometry
@dataclass(frozen=True)
class CouplingGeometry:
    inner_face_radius: float        # apothem of the inner blocks' outer faces
    inner_corner_radius: float      # largest radius of the inner ring (block corners, or arc OD)
    outer_face_apothem: float       # apothem of the outer blocks' inner faces
    face_gap: float                 # magnetic gap at the flat centres
    corner_gap: float               # inner ring's largest radius to the outer faces
    outer_back_apothem: float       # outer block backs (without bondline)
    pocket_apothem: float           # cup pocket flats (block back + bondline)
    pocket_corner_radius: float     # pocket polygon vertex radius (or pocket radius for arcs)
    cup_od: float
    cup_wall_flat: float            # steel from the pocket flat to the cup OD
    gap_radius: float
    pole_pitch: float               # at the gap radius
    fill_inner: float               # block width / pole arc at the block's mid-thickness radius (planar unrolling)
    fill_outer: float
    inner_flat_width: float         # side of the polygon through the inner block backs
    outer_flat_width: float         # side of the polygon through the outer block faces
    hub_apothem: float              # machined hub flats (block back minus bondline)
    hub_wall: float                 # hub flat to bore
    hub_wall_past_key: float


def coupling_geometry(npole: int, inner_back_apothem: float, t_i: float, w_i: float, t_o: float, w_o: float,
                      face_gap: float, bond_inner: float, bond_outer: float, cup_wall_corner: float,
                      bore: float, keyway_depth: float, faceted: bool = True) -> CouplingGeometry:
    """Every block/polygon dimension from the inputs, by construction.

    Definitions used: the flat-face gap is the distance between the facing block faces at the
    flat centres, for flat blocks and for arcs alike; the cup wall input is the minimum steel
    at the pocket corners; the hub apothem is the block-back apothem minus the bondline; the
    keyway depth is measured radially from the bore."""
    pitch_angle = 2 * math.pi / npole
    inner_block = block_rectangle(inner_back_apothem, t_i, w_i, 0.0)
    r_face = inner_back_apothem + t_i
    r_corner = max_radius(inner_block) if faceted else r_face
    a_o = r_face + face_gap
    a_back = a_o + t_o
    pocket_ap = a_back + bond_outer
    r_pocket = max_radius(regular_polygon(pocket_ap, npole)) if faceted else pocket_ap
    od = 2 * (r_pocket + cup_wall_corner)
    r_gap = r_face + face_gap / 2
    r_mid_i = inner_back_apothem + t_i / 2
    r_mid_o = a_o + t_o / 2
    hub_ap = inner_back_apothem - bond_inner
    return CouplingGeometry(
        inner_face_radius=r_face,
        inner_corner_radius=r_corner,
        outer_face_apothem=a_o,
        face_gap=a_o - r_face,
        corner_gap=a_o - r_corner,
        outer_back_apothem=a_back,
        pocket_apothem=pocket_ap,
        pocket_corner_radius=r_pocket,
        cup_od=od,
        cup_wall_flat=od / 2 - pocket_ap,
        gap_radius=r_gap,
        pole_pitch=r_gap * pitch_angle,
        fill_inner=min(1.0, w_i / (r_mid_i * pitch_angle)),
        fill_outer=min(1.0, w_o / (r_mid_o * pitch_angle)),
        inner_flat_width=side_length(regular_polygon(inner_back_apothem, npole)),
        outer_flat_width=side_length(regular_polygon(a_o, npole)),
        hub_apothem=hub_ap,
        hub_wall=hub_ap - bore / 2,
        hub_wall_past_key=hub_ap - bore / 2 - keyway_depth,
    )


# --------------------------------------------------------------------------- solids
@dataclass(frozen=True)
class Part:
    mass_g: float
    polar_inertia_g_mm2: float      # about the coupling axis

    def __add__(self, other: "Part") -> "Part":
        return Part(self.mass_g + other.mass_g, self.polar_inertia_g_mm2 + other.polar_inertia_g_mm2)


def prism_part(area: float, polar_moment: float, length: float, rho: float) -> Part:
    """Straight prism along the axis: mass = ρ·A·L, J_mass = ρ·L·J_area (J_area about the axis)."""
    return Part(rho * area * length, rho * polar_moment * length)


def polygon_part(pts: list[Point], length: float, rho: float) -> Part:
    return prism_part(polygon_area(pts), polygon_polar_moment(pts), length, rho)


def disk_section(diameter: float) -> tuple[float, float]:
    """(area, polar moment about the centre) of a full disk: πD²/4 and πD⁴/32."""
    return math.pi * diameter ** 2 / 4, math.pi * diameter ** 4 / 32


def annulus_part(od: float, id_: float, length: float, rho: float) -> Part:
    """Tube: disk(od) minus disk(id)."""
    a_o, j_o = disk_section(od)
    a_i, j_i = disk_section(id_)
    return prism_part(a_o - a_i, j_o - j_i, length, rho)


def rotating_parts(npole: int, inner_back_apothem: float, t_i: float, w_i: float, L_i: float,
                   outer_face_apothem: float, t_o: float, w_o: float, L_o: float,
                   pocket_apothem: float, hub_apothem: float, cup_od: float, bore: float,
                   cup_depth: float, web: float, hub_length: float, boss_length: float, boss_od: float,
                   cup_density: float, hub_density: float, faceted: bool = True,
                   magnet_density: float = NDFEB_DENSITY_G_MM3) -> dict[str, Part]:
    """Gross solids (no holes, slots, keyways or threads subtracted), matching the workbook's
    documented simplification.

    magnets: 2N explicit block rectangles × length. cup: disk(OD) minus the pocket cavity
    (regular N-gon for flat blocks, circle for arcs) over the cavity depth, plus the rear web
    disk(OD) minus the bore. hub: hub polygon (circle for arcs) minus the bore. boss: tube.
    The web and boss are one piece with the cup, so they take ``cup_density``."""
    pitch = 2 * math.pi / npole
    magnets = Part(0.0, 0.0)
    for k in range(npole):
        magnets = magnets + polygon_part(block_rectangle(inner_back_apothem, t_i, w_i, pitch * k), L_i, magnet_density)
        magnets = magnets + polygon_part(block_rectangle(outer_face_apothem, t_o, w_o, pitch * k), L_o, magnet_density)
    a_od, j_od = disk_section(cup_od)
    a_bore, j_bore = disk_section(bore)
    if faceted:
        pocket = regular_polygon(pocket_apothem, npole)
        a_cav, j_cav = polygon_area(pocket), polygon_polar_moment(pocket)
        hub_poly = regular_polygon(hub_apothem, npole)
        a_hub, j_hub = polygon_area(hub_poly), polygon_polar_moment(hub_poly)
    else:
        a_cav, j_cav = disk_section(2 * pocket_apothem)
        a_hub, j_hub = disk_section(2 * hub_apothem)
    cup = (prism_part(a_od - a_cav, j_od - j_cav, cup_depth, cup_density)
           + prism_part(a_od - a_bore, j_od - j_bore, web, cup_density))
    hub = prism_part(a_hub - a_bore, j_hub - j_bore, hub_length, hub_density)
    boss = annulus_part(boss_od, bore, boss_length, cup_density)
    return {"magnets": magnets, "cup": cup, "hub": hub, "boss": boss}


# --------------------------------------------------------------------------- retainers
@dataclass(frozen=True)
class RetainerGeometry:
    sleeve_id: float
    sleeve_od: float
    liner_od: float
    liner_id: float
    endplate_od: float


def retainer_geometry(inner_ring_max_radius: float, outer_ring_min_radius: float, sleeve: float, liner: float,
                      sleeve_bedding: float, liner_bedding: float) -> RetainerGeometry:
    """Round sleeve over the inner ring (clears its largest radius by the bedding clearance) and
    round liner inside the outer ring (clears its smallest radius). Endplates cap the inner
    magnets up to the sleeve bore."""
    sleeve_id = 2 * (inner_ring_max_radius + sleeve_bedding)
    liner_od = 2 * (outer_ring_min_radius - liner_bedding)
    return RetainerGeometry(sleeve_id=sleeve_id, sleeve_od=sleeve_id + 2 * sleeve, liner_od=liner_od,
                            liner_id=liner_od - 2 * liner, endplate_od=sleeve_id)


def retainer_parts(g: RetainerGeometry, span: float, retainer_density: float, cap_od: float, cap_face: float,
                   cap_thread_dia: float, cap_thread_engagement: float, al_density: float, bore: float,
                   front_endplate: float, rear_endplate: float, rear_hole: float) -> dict[str, Part]:
    """Sleeve and liner tubes over the retainer span; the aluminium cap as a face plate
    (cap OD to the liner ID opening) plus the threaded skirt (cap OD to thread diameter);
    the two 316L endplates (front: bore opening, rear: screw-clearance opening)."""
    return {
        "sleeve": annulus_part(g.sleeve_od, g.sleeve_id, span, retainer_density),
        "liner": annulus_part(g.liner_od, g.liner_id, span, retainer_density),
        "cap": (annulus_part(cap_od, g.liner_id, cap_face, al_density)
                + annulus_part(cap_od, cap_thread_dia, cap_thread_engagement, al_density)),
        "endplates": (annulus_part(g.endplate_od, bore, front_endplate, retainer_density)
                      + annulus_part(g.endplate_od, rear_hole, rear_endplate, retainer_density)),
    }


# --------------------------------------------------------------------------- adapters for DesignInputs
def geometry_for_design(inp, res) -> CouplingGeometry:
    """Reference geometry from the design inputs and the resolved magnet data (Calculator C18–C31)."""
    c, md, m = inp.coupling, inp.metal, res.model
    return coupling_geometry(npole=c.npole, inner_back_apothem=c.inner_back_apothem_mm, t_i=m.inner_thickness_mm,
                             w_i=m.inner_width_mm, t_o=m.outer_thickness_mm, w_o=m.outer_width_mm,
                             face_gap=md.face_gap_mm, bond_inner=md.bond_inner_mm, bond_outer=md.bond_outer_mm,
                             cup_wall_corner=md.cup_wall_corner_mm, bore=c.bore_mm, keyway_depth=c.keyway_depth_mm,
                             faceted=c.faceted == 1)


def rotating_parts_for_design(inp, res, cup_density: float | None = None) -> dict[str, Part]:
    """Solids built on the engine's own ring placement (Calculator C56, C60, C62), so a mass check
    isolates the mass formula; the placement itself is verified by the geometry checks.
    ``cup_density`` defaults to the steel density; the hub takes steel with back iron and the
    aluminium carrier density without, as the workbook states."""
    c, md, m = inp.coupling, inp.metal, res.model
    steel, al = md.steel_density_g_mm3, md.al_density_g_mm3
    return rotating_parts(npole=c.npole, inner_back_apothem=c.inner_back_apothem_mm, t_i=m.inner_thickness_mm,
                          w_i=m.inner_width_mm, L_i=m.inner_length_mm, outer_face_apothem=m.outer_face_apothem_mm,
                          t_o=m.outer_thickness_mm, w_o=m.outer_width_mm, L_o=m.outer_length_mm,
                          pocket_apothem=m.outer_back_apothem_mm + md.bond_outer_mm,
                          hub_apothem=c.inner_back_apothem_mm - md.bond_inner_mm, cup_od=m.cup_od_mm, bore=c.bore_mm,
                          cup_depth=md.cup_depth_mm, web=md.web_mm, hub_length=md.hub_length_mm,
                          boss_length=md.boss_length_mm, boss_od=md.boss_od_mm,
                          cup_density=steel if cup_density is None else cup_density,
                          hub_density=steel if c.backiron == 1 else al, faceted=c.faceted == 1)


def retainer_geometry_for_design(inp, res) -> RetainerGeometry:
    """Retainers sized on the reference ring extents: the inner ring's largest radius (block
    corners, or the arc OD) and the outer ring's smallest radius (the outer face centres)."""
    c, md, m = inp.coupling, inp.metal, res.model
    g = geometry_for_design(inp, res)
    outer_block = block_rectangle(g.outer_face_apothem, m.outer_thickness_mm, m.outer_width_mm, 0.0)
    outer_min = min_radius(outer_block) if c.faceted == 1 else g.outer_face_apothem
    return retainer_geometry(g.inner_corner_radius, outer_min, md.sleeve_mm, md.liner_mm,
                             md.sleeve_bedding_mm, md.liner_bedding_mm)


def retainer_parts_for_design(inp, res) -> dict[str, Part]:
    c, md = inp.coupling, inp.metal
    return retainer_parts(retainer_geometry_for_design(inp, res), span=md.retainer_span_mm,
                          retainer_density=md.sleeve_density_g_mm3, cap_od=md.cap_od_mm, cap_face=md.cap_axial_mm,
                          cap_thread_dia=md.cap_thread_dia_mm, cap_thread_engagement=md.cap_thread_engagement_mm,
                          al_density=md.al_density_g_mm3, bore=c.bore_mm, front_endplate=md.front_endplate_mm,
                          rear_endplate=md.rear_endplate_mm, rear_hole=md.rear_endplate_hole_mm)
```

File: `reference/magcoupling-py/audit/references/slab_field2d.py` (create)

```python
"""2D magnetostatic reference: alternating magnet layers between two infinitely permeable plates.

Shared slab model of the audit (created in Task 4; Task 5 appends its force and energy functions).
Written from scratch; nothing here imports ``magcoupling``. The square-wave harmonic amplitudes come from
``audit.references.planar.square_wave_harmonic``, the audit's single implementation.

Geometry (unrolled coupling, x tangential, y radial, units mm and T)
    Steel plates (mu -> infinity) at y = 0 and y = h. Each magnet layer occupies
    y in [y_bottom, y_bottom + thickness]; its blocks have width fill * pitch, alternate in
    polarity every pole pitch, and the block centred at x = shift is magnetized +y (Br > 0).
    Magnets have recoil permeability 1, so the whole slab is one linear medium.

Method (magnetic scalar potential with surface charges)
    A layer magnetized M_y(x) carries surface charge sigma = -M_y on its bottom face and
    +M_y on its top face. With mu0*sigma_n cos(k (x - s)) on a sheet at y = ys, the potential
    that vanishes on both plates (H tangential = 0 at mu -> infinity) is
        mu0 phi = a_n cos(k(x - s)) sinh(k y<) sinh(k (h - y>)) / (k sinh(k h)),
    where y< = min(y, ys), y> = max(y, ys), k = n pi / pitch and, for the alternating square wave,
    a_n = Br * 4/(n pi) * sin(n pi fill / 2) (odd n). B = -mu0 grad(phi) in air.

Flux into the plates (``iron_face_coefficients``)
    On a plate face B = mu0 (H + M). The H part is the limit of the sheet potentials at the face;
    a sheet lying on the face itself contributes nothing there (its potential vanishes identically,
    because sinh(k * 0) = 0), and the M part is the square-wave magnetization of a layer that
    touches the plate. Evaluating ``harmonic_coefficients`` does not give this: it is defined for
    air points off the magnet faces, and inside a magnet it returns mu0 H, not B.

This is the standard slotless-machine / linear-coupling field solution; see
Furlani, *Permanent Magnet and Electromechanical Devices* (2001), ch. 4 and 8, and
Zhu & Howe, IEEE Trans. Magn. 29 (1993) 124-135 for the same separation-of-variables approach.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from audit.references.planar import square_wave_harmonic


@dataclass(frozen=True)
class MagnetLayer:
    y_bottom_mm: float
    thickness_mm: float
    br_T: float          # signed: > 0 means the block centred at x = shift points +y
    fill: float          # block width / pole pitch, 0 < fill <= 1
    shift_mm: float = 0.0


def _sheets(layers: list[MagnetLayer]):
    for lay in layers:
        yield lay.y_bottom_mm, -lay.br_T, lay.fill, lay.shift_mm
        yield lay.y_bottom_mm + lay.thickness_mm, lay.br_T, lay.fill, lay.shift_mm


def _square_wave(br_T: float, n: np.ndarray, fill: float) -> np.ndarray:
    """a_n = Br * 4/(n pi) * sin(n pi fill / 2) for each odd order in ``n``: the audit's single implementation,
    ``planar.square_wave_harmonic``, evaluated order by order."""
    return np.array([square_wave_harmonic(br_T, int(order), fill) for order in n])


def _ratio(k: np.ndarray, p: float, q: float, h: float, cosh_q: bool) -> np.ndarray:
    """sinh(k p) * (cosh or sinh)(k q) / sinh(k h), overflow-free for 0 <= p, q and p + q <= h."""
    sign = 1.0 if cosh_q else -1.0
    return (np.exp(k * (p + q - h)) * (1.0 - np.exp(-2.0 * k * p)) * (1.0 + sign * np.exp(-2.0 * k * q))
            / (2.0 * (1.0 - np.exp(-2.0 * k * h))))


def harmonic_coefficients(y_mm: float, layers: list[MagnetLayer], h_mm: float, pitch_mm: float,
                          n_max: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(n, A, B, C, D) with B_y(x) = sum A cos(kx) + B sin(kx) and B_x(x) = sum C sin(kx) + D cos(kx) at height y.

    ``y_mm`` must not coincide with a charge sheet (a magnet face).
    """
    n = np.arange(1, n_max + 1, 2, dtype=float)
    k = n * math.pi / pitch_mm
    A = np.zeros_like(n); B = np.zeros_like(n); C = np.zeros_like(n); D = np.zeros_like(n)
    for ys, br, fill, shift in _sheets(layers):
        if abs(y_mm - ys) < 1e-12:
            raise ValueError(f"y = {y_mm} mm lies on a magnet face")
        a = _square_wave(br, n, fill)
        if y_mm < ys:
            p = -a * _ratio(k, h_mm - ys, y_mm, h_mm, True)     # B_y amplitude
            qx = a * _ratio(k, y_mm, h_mm - ys, h_mm, False)    # B_x amplitude
        else:
            p = a * _ratio(k, ys, h_mm - y_mm, h_mm, True)
            qx = a * _ratio(k, ys, h_mm - y_mm, h_mm, False)
        c, s = np.cos(k * shift), np.sin(k * shift)
        A += p * c; B += p * s; C += qx * c; D -= qx * s
    return n, A, B, C, D


def iron_face_coefficients(surface: str, layers: list[MagnetLayer], h_mm: float, pitch_mm: float,
                           n_max: int = 2001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, A, B) with B_y(x) = sum A cos(kx) + B sin(kx) [T] on the face of the plate at y = 0 (``surface``
    'inner') or y = h ('outer'): the flux density that enters the steel (see the module docstring)."""
    if surface not in ("inner", "outer"):
        raise ValueError(f"surface must be 'inner' or 'outer', not {surface!r}")
    y_face = 0.0 if surface == "inner" else h_mm
    n = np.arange(1, n_max + 1, 2, dtype=float)
    k = n * math.pi / pitch_mm
    A = np.zeros_like(n); B = np.zeros_like(n)
    for ys, br, fill, shift in _sheets(layers):
        if abs(ys - y_face) < 1e-12:
            continue                                            # a sheet on this plate adds no field here
        a = _square_wave(br, n, fill)
        if surface == "outer":                                  # face above the sheet: y = h
            p = a * _ratio(k, ys, 0.0, h_mm, True)
        else:                                                   # face below the sheet: y = 0
            p = -a * _ratio(k, h_mm - ys, 0.0, h_mm, True)
        A += p * np.cos(k * shift); B += p * np.sin(k * shift)
    for lay in layers:                                          # mu0 M of a layer touching the plate
        touching = lay.y_bottom_mm if surface == "inner" else lay.y_bottom_mm + lay.thickness_mm
        if abs(touching - y_face) < 1e-12:
            a = _square_wave(lay.br_T, n, lay.fill)
            A += a * np.cos(k * lay.shift_mm); B += a * np.sin(k * lay.shift_mm)
    return n, A, B
```

File: `reference/magcoupling-py/audit/references/backiron_flux.py` (create)

```python
"""Back-iron flux by flux conservation (M1 audit, Task 4). Imports nothing from ``magcoupling``.

Two independent routes to the flux the return steel carries between poles:

1. Magnetic-circuit route. With ideal iron and mu_r = 1 magnets, the series circuit of two
   magnets and a gap gives B_g = (Br_i t_i + Br_o t_o) / (t_i + t_o + g)
   [E. P. Furlani, *Permanent Magnet and Electromechanical Devices*, Academic Press 2001,
   sec. 3.3; D. Hanselman, *Brushless Permanent Magnet Motor Design*, 2nd ed. 2006, ch. 2].
   For a sinusoidal gap field of peak B over a pole pitch tau, one pole carries
   integral_0^tau B sin(pi x / tau) dx = 2 B tau / pi per unit axial length and half of it turns
   each way in the back iron, so the back iron needs t = B tau / (pi B_design) [Hanselman ch. 4].

2. Field route (``backiron_flux_per_depth``): the exact 2D field of the workbook's steel-backed
   slab from the shared ``slab_field2d`` model (ideal iron at both faces, square-wave
   magnetization, all odd harmonics). The back-iron section at the inter-pole line carries the
   flux that enters the plate between the pole centre, where the back-iron flux is zero by
   symmetry, and that line.
"""
from __future__ import annotations

import math

import numpy as np

from audit.references.slab_field2d import MagnetLayer, iron_face_coefficients


def flat_circuit_flux_density(br_i: float, br_o: float, t_i: float, t_o: float, gap: float) -> float:
    """Gap flux density [T] of the ideal-iron series circuit of two magnets and a gap (mu_r = 1).
    Ampere's law around a current-free loop, H_i t_i + H_g g + H_o t_o = 0, with B continuous
    and B = mu0 (H + M) in each magnet, gives B = (Br_i t_i + Br_o t_o) / (t_i + t_o + gap)."""
    return (br_i * t_i + br_o * t_o) / (t_i + t_o + gap)


def sinusoidal_backiron_thickness(b_peak: float, pole_pitch: float, b_design: float) -> float:
    """Back-iron thickness [unit of pole_pitch] carrying half the flux of one pole of a sinusoidal
    gap field of peak ``b_peak``: (2 B tau / pi) / 2 / B_design."""
    flux_per_pole = 2 * b_peak * pole_pitch / math.pi
    return flux_per_pole / 2 / b_design


def coupling_slab(br_i: float, br_o: float, fill_i: float, fill_o: float, t_i: float, t_o: float,
                  gap: float) -> tuple[list[MagnetLayer], float]:
    """The steel-backed coupling slab: inner magnet on the inner plate (y = 0), the gap, outer magnet
    on the outer plate (y = h). Both are magnetized +y at x = 0: aligned rings, the no-load and
    maximum-flux position. Returns (layers, h)."""
    layers = [MagnetLayer(0.0, t_i, br_i, fill_i), MagnetLayer(t_i + gap, t_o, br_o, fill_o)]
    return layers, t_i + gap + t_o


def backiron_flux_per_depth(surface: str, layers: list[MagnetLayer], h_mm: float, pitch_mm: float,
                            n_max: int = 4001) -> float:
    """Flux per unit axial length [T*mm] that the 'inner' or 'outer' plate carries across the
    inter-pole line x = tau/2, for aligned layers (every shift zero, so x = 0 is a pole centre):
    integral_0^(tau/2) sum A cos(kx) dx = sum A sin(k tau/2) / k. Terms fall as 1/n^2, so
    n_max = 4001 truncates below 2e-4 relative."""
    if any(lay.shift_mm != 0.0 for lay in layers):
        raise ValueError("backiron_flux_per_depth needs aligned layers (every shift_mm == 0)")
    n, A, _ = iron_face_coefficients(surface, layers, h_mm, pitch_mm, n_max)
    k = n * math.pi / pitch_mm
    return float(np.sum(A * np.sin(k * pitch_mm / 2) / k))
```

File: `reference/magcoupling-py/audit/references/remanence.py` (create)

```python
"""Remanence versus temperature and the torque law that follows from it (M1 audit; shared by Tasks 4-7).

Imports nothing from ``magcoupling``.

- Br(T) = Br20 * (1 + alpha * (T - 20 degC)), with alpha the reversible temperature coefficient of Br
  in 1/degC referenced to 20 degC, the datasheet convention for sintered NdFeB (IEC 60404-8-1; e.g.
  Arnold Magnetic Technologies N42SH: alpha(Br) = -0.12 %/degC over 20-150 degC).
- The pull-out torque of a PM-PM coupling is bilinear in the two rings' magnetizations (each
  harmonic's shear stress is B_in * B_on * S_n / (2 mu0)), so with one alpha for both rings the
  torque scales as Br(T)^2.
"""
from __future__ import annotations

# One implementation of Br(T): planar.br_at. Re-exported here so ``audit.references.remanence.br_at`` stays
# importable for its consumers.
from audit.references.planar import br_at


def br_ratio(alpha_per_C: float, temp_C: float, ref_C: float = 20.0) -> float:
    """Br(temp_C) / Br(ref_C)."""
    return br_at(1.0, alpha_per_C, temp_C) / br_at(1.0, alpha_per_C, ref_C)


def torque_at(torque_ref_Nm: float, alpha_per_C: float, ref_C: float, temp_C: float) -> float:
    """Torque moved from ``ref_C`` to ``temp_C`` with torque proportional to Br_inner * Br_outer = Br^2."""
    return torque_ref_Nm * br_ratio(alpha_per_C, temp_C, ref_C) ** 2
```

File: `reference/magcoupling-py/audit/references/metal_stack.py` (create)

```python
"""Independent re-derivations for the Metal design sheet and the plating offsets (M1 audit, Task 4).

Imports nothing from ``magcoupling``. Temperature scaling lives in ``audit.references.remanence``.
Each function states the physical rule it encodes:

- A symmetric production allowance v gives a low value T (1 - v) and a high value T (1 + v).
- A gearbox of ratio i and efficiency eta, motor driving: speeds w_in = i w_out and power
  P_out = eta P_in, so T_out = i eta T_in and T_in = T_out / (i eta).
- Radial clearances stack linearly (worst case); axial stacks add.
- A coupling with N poles per ring has N/2 pole pairs; slipping at n rpm passes (N/2) n / 60
  pole pairs per second, and each pole-pair pass is one full torque/field cycle.
- n rpm is an angular speed w = 2 pi n / 60 rad/s; mechanical power is torque times angular
  speed, P = T w.
- Plating grows every coated surface by its thickness along the surface's outward normal
  (out of the material).
"""
from __future__ import annotations

import math


def torque_low(torque_Nm: float, variation: float) -> float:
    """Low end of a symmetric allowance."""
    return torque_Nm * (1 - variation)


def torque_high(torque_Nm: float, variation: float) -> float:
    """High end of a symmetric allowance."""
    return torque_Nm * (1 + variation)


def gearbox_input_torque(output_Nm: float, ratio: float, efficiency: float) -> float:
    """Gearbox input torque for an output torque, motor driving: T_in = T_out / (i eta).
    (Back-driven by the wheel the input sees T_out eta / i, which is smaller; the driving form
    is the conservative one.)"""
    return output_Nm / (ratio * efficiency)


def gearbox_output_torque(input_Nm: float, ratio: float, efficiency: float) -> float:
    """Gearbox output torque for an input torque, motor driving: T_out = i eta T_in."""
    return input_Nm * ratio * efficiency


def sleeve_liner_clearance(corner_gap: float, sleeve: float, liner: float,
                           sleeve_bedding: float, liner_bedding: float) -> float:
    """Radial space between the round sleeve (over the inner corners) and the round liner
    (inside the outer faces): the corner gap minus both retainer walls and both beddings."""
    return corner_gap - sleeve - liner - sleeve_bedding - liner_bedding


def linear_stack(*items: float) -> float:
    """Worst-case (arithmetic) stack of allowances or lengths."""
    return math.fsum(items)


def pole_pairs(npole: int) -> float:
    """Pole pairs per ring of a coupling with ``npole`` poles per ring: N/2."""
    return npole / 2


def pole_pair_frequency_Hz(npole: int, rpm: float) -> float:
    """Pole pairs passing per second at a relative speed of ``rpm``."""
    return pole_pairs(npole) * rpm / 60


def omega_rad_s(rpm: float) -> float:
    """Angular speed [rad/s] of ``rpm`` revolutions per minute: w = 2 pi rpm / 60."""
    return 2 * math.pi * rpm / 60


def shaft_power_W(torque_Nm: float, rpm: float) -> float:
    """P = T w with w = omega_rad_s(rpm)."""
    return torque_Nm * omega_rad_s(rpm)


def plated_size(size: float, thickness: float, surfaces: int, external: bool) -> float:
    """Size after plating. An external feature (hub flat apothem, outside diameter) grows by
    ``thickness`` per plated surface it spans; an internal feature (pocket apothem, bore)
    shrinks by the same amount, because the coating grows out of the material."""
    return size + (1 if external else -1) * surfaces * thickness


def preplate_size(finished: float, thickness: float, surfaces: int, external: bool) -> float:
    """Machined (pre-plate) size that plates to ``finished``. Plating shifts a size by a
    constant, so the machined size is the finished size minus that shift."""
    growth = plated_size(0.0, thickness, surfaces, external)
    return finished - growth
```

- [ ] **Step 2: Write the references' own sanity tests**

File: `reference/magcoupling-py/audit/tests/test_geometry_metal_reference_sanity.py` (create)

```python
"""Sanity tests for the Task 4 references against textbook cases with known answers.

These never touch the engine: they prove the references are trustworthy before the engine
is compared against them. Every test name contains 'sanity' so coverage tools can tell them
apart from engine checks.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from audit.common import TOL_ALGEBRA, mismatch, rel_err
from audit.references.backiron_flux import (backiron_flux_per_depth, coupling_slab, flat_circuit_flux_density,
                                            sinusoidal_backiron_thickness)
from audit.references.block_geometry import (annulus_part, axial_overlap, block_rectangle, convex_polygon_distance,
                                             max_radius, min_radius, point_segment_distance, polygon_area,
                                             polygon_polar_moment, regular_polygon, ring_min_clearance, side_length)
from audit.references.metal_stack import (gearbox_input_torque, gearbox_output_torque, linear_stack, omega_rad_s,
                                          plated_size, pole_pair_frequency_Hz, pole_pairs, preplate_size, shaft_power_W,
                                          sleeve_liner_clearance)
from audit.references.remanence import br_at, br_ratio, torque_at
from audit.references.slab_field2d import MagnetLayer, harmonic_coefficients, iron_face_coefficients


# --------------------------------------------------------------------------- block_geometry
@pytest.mark.family("geometry")
def test_sanity_regular_polygon_square_and_hexagon():
    """Square of apothem 1: vertices at (±1, ±1), side 2. Hexagon of apothem √3/2: circumradius 1, side 1."""
    sq = regular_polygon(1.0, 4, phase=0.0)
    for x, y in sq:
        assert abs(abs(x) - 1) < 1e-12 and abs(abs(y) - 1) < 1e-12, \
            mismatch(f"square vertex ({x}, {y})", "reference", max(abs(x), abs(y)), 1.0, 1e-12)
    assert abs(side_length(sq) - 2) < 1e-12, mismatch("square side", "reference", side_length(sq), 2.0, 1e-12)
    hx = regular_polygon(math.sqrt(3) / 2, 6)
    assert rel_err(max_radius(hx), 1.0) < 1e-12, mismatch("hexagon circumradius", "reference", max_radius(hx), 1.0, 1e-12)
    assert rel_err(side_length(hx), 1.0) < 1e-12, mismatch("hexagon side", "reference", side_length(hx), 1.0, 1e-12)
    assert rel_err(min_radius(hx), math.sqrt(3) / 2) < 1e-12, \
        mismatch("hexagon apothem", "reference", min_radius(hx), math.sqrt(3) / 2, 1e-12)


@pytest.mark.family("geometry")
def test_sanity_area_and_polar_moment():
    """Unit square about its centre: A = 1, J = 1/6. A 2×1 rectangle centred at (3, 0): J_O = J_c + A·d²
    = 2·(4 + 1)/12 + 2·9 (parallel-axis theorem). A 20000-gon of apothem 1 approaches the unit disk:
    A → π, J → π/2 (polygon error ~ (π/n)²/3 ≈ 8e-9)."""
    sq = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    assert rel_err(polygon_area(sq), 1.0) < 1e-12, mismatch("square area", "reference", polygon_area(sq), 1.0, 1e-12)
    assert rel_err(polygon_polar_moment(sq), 1 / 6) < 1e-12, \
        mismatch("square J", "reference", polygon_polar_moment(sq), 1 / 6, 1e-12)
    rect = [(2.0, -0.5), (4.0, -0.5), (4.0, 0.5), (2.0, 0.5)]
    want = 2 * (4 + 1) / 12 + 2 * 9
    assert rel_err(polygon_polar_moment(rect), want) < 1e-12, \
        mismatch("offset rectangle J", "reference", polygon_polar_moment(rect), want, 1e-12)
    disk = regular_polygon(1.0, 20000)
    assert rel_err(polygon_area(disk), math.pi) < 1e-7, mismatch("disk area", "reference", polygon_area(disk), math.pi, 1e-7)
    assert rel_err(polygon_polar_moment(disk), math.pi / 2) < 1e-7, \
        mismatch("disk J", "reference", polygon_polar_moment(disk), math.pi / 2, 1e-7)


@pytest.mark.family("geometry")
def test_sanity_annulus_part():
    """Tube OD 4, ID 2, length 3, ρ 1: m = π(4 − 1)·3 = 9π; J = m·(R² + r²)/2 = 9π·(4 + 1)/2."""
    p = annulus_part(4.0, 2.0, 3.0, 1.0)
    assert rel_err(p.mass_g, 9 * math.pi) < 1e-12, mismatch("tube mass", "reference", p.mass_g, 9 * math.pi, 1e-12)
    want = 9 * math.pi * 5 / 2
    assert rel_err(p.polar_inertia_g_mm2, want) < 1e-12, mismatch("tube J", "reference", p.polar_inertia_g_mm2, want, 1e-12)


@pytest.mark.family("geometry")
def test_sanity_distances():
    """Point (0, 2) to segment (−1, 0)–(1, 0): 2; to (3, 0)–(5, 0): hypot(3, 2). Unit squares at x ∈ [0, 1] and
    [3, 4]: 2. A square rotated 45° with its corner at (1, 0) against a square whose face is at x = 1.5: 0.5."""
    mid = point_segment_distance((0, 2), (-1, 0), (1, 0))
    assert abs(mid - 2) < 1e-15, mismatch("point-segment interior", "reference", mid, 2.0, 1e-15)
    d = point_segment_distance((0, 2), (3, 0), (5, 0))
    assert rel_err(d, math.hypot(3, 2)) < 1e-15, mismatch("point-segment end", "reference", d, math.hypot(3, 2), 1e-15)
    a = [(0, 0), (1, 0), (1, 1), (0, 1)]
    b = [(3, 0), (4, 0), (4, 1), (3, 1)]
    assert abs(convex_polygon_distance(a, b) - 2) < 1e-15, mismatch("square gap", "reference", convex_polygon_distance(a, b), 2, 1e-15)
    diamond = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    box = [(1.5, -2), (3, -2), (3, 2), (1.5, 2)]
    got = convex_polygon_distance(diamond, box)
    assert abs(got - 0.5) < 1e-15, mismatch("corner to face", "reference", got, 0.5, 1e-15)


@pytest.mark.family("geometry")
def test_sanity_axial_overlap():
    """Centred 12.7 and 25.4 mm blocks overlap by 12.7 mm; two 10 mm blocks offset by 4 mm overlap by 6 mm;
    offset by 20 mm they do not overlap."""
    for args, want in (((12.7, 25.4), 12.7), ((10.0, 10.0, 4.0), 6.0), ((10.0, 10.0, 20.0), 0.0)):
        got = axial_overlap(*args)
        assert abs(got - want) < 1e-12, mismatch(f"axial overlap {args}", "reference", got, want, 1e-12)


@pytest.mark.family("geometry")
def test_sanity_block_rectangle_and_ring_clearance():
    """A block at angle 0 with near face at 1, thickness 1, width 2 has corners at radius √(2² + 1²) = √5.
    Four such blocks against outer blocks whose faces sit at apothem 3: the inner corner can face an outer
    face centre, so the smallest clearance over rotation is 3 − √5 (hand geometry). In the thin-block
    limit (width 1e-6) the clearance is the face-to-face gap, 1.0."""
    blk = block_rectangle(1.0, 1.0, 2.0, 0.0)
    assert rel_err(max_radius(blk), math.sqrt(5)) < 1e-15, mismatch("block corner radius", "reference", max_radius(blk), math.sqrt(5), 1e-15)
    got = ring_min_clearance(4, 1.0, 1.0, 2.0, 3.0, 1.0, 2.0)
    assert rel_err(got, 3 - math.sqrt(5)) < 1e-9, mismatch("ring clearance, N=4", "reference", got, 3 - math.sqrt(5), 1e-9)
    thin = ring_min_clearance(10, 10.0, 3.0, 1e-6, 14.0, 3.0, 1e-6)
    assert rel_err(thin, 1.0) < 1e-9, mismatch("thin-block ring clearance", "reference", thin, 1.0, 1e-9)


# --------------------------------------------------------------------------- backiron_flux and slab_field2d
@pytest.mark.family("materials")
def test_sanity_flat_circuit_textbook():
    """Furlani 2001 §3.3: ideal iron, μ_r = 1, Br = 1.2 T, magnet length 4 mm, gap 1 mm → B = 1.2·4/5 = 0.96 T.
    Split between two 2 mm magnets it is the same; with no gap B = Br."""
    got = flat_circuit_flux_density(1.2, 1.2, 2.0, 2.0, 1.0)
    assert rel_err(got, 0.96) < TOL_ALGEBRA, mismatch("flat circuit", "reference", got, 0.96, TOL_ALGEBRA)
    got = flat_circuit_flux_density(1.2, 1.2, 4.0, 0.0, 0.0)
    assert rel_err(got, 1.2) < TOL_ALGEBRA, mismatch("flat circuit, no gap", "reference", got, 1.2, TOL_ALGEBRA)


@pytest.mark.family("materials")
def test_sanity_sinusoidal_backiron_thickness():
    """Half the flux of one pole of B̂·sin(πx/τ), integrated numerically (scipy quad), divided by B_design:
    B̂ = 1 T, τ = 10 mm, B_design = 1.5 T → (2·10/π)/2/1.5 = 2.1221 mm."""
    flux, _ = quad(lambda x: math.sin(math.pi * x / 10.0), 0.0, 10.0)
    want = flux / 2 / 1.5
    got = sinusoidal_backiron_thickness(1.0, 10.0, 1.5)
    assert rel_err(got, want) < 1e-12, mismatch("sinusoidal back iron", "reference", got, want, 1e-12)


@pytest.mark.family("materials")
@pytest.mark.parametrize("surface", ["outer", "inner"])
def test_sanity_slab_magnet_fills_space(surface):
    """One magnet filling the whole space between the plates: H = 0 and B = μ0·M exactly, so the flux entering
    either plate over half a pole is Br·fill·τ/2 (square wave). Exercises both terms of iron_face_coefficients
    (the sheet on the plate adds nothing; μ0M of the touching layer is added).
    Tolerance 2e-4: harmonic truncation at n = 4001 (terms fall as 1/n²)."""
    br, fill, tau = 1.3, 0.8, 9.0
    got = backiron_flux_per_depth(surface, [MagnetLayer(0.0, 3.0, br, fill)], 3.0, tau)
    want = br * fill * tau / 2
    assert rel_err(got, want) < 2e-4, mismatch(f"slab, magnet fills space ({surface} plate)", "reference", got, want, 2e-4)


@pytest.mark.family("materials")
def test_sanity_slab_long_pitch_is_flat_circuit():
    """When the pole pitch is 10⁴ times the circuit height the field is the flat circuit's under the magnets and
    zero between them, so the half-pole flux is B_flat·fill·τ/2. Tolerance 1e-3: fringing is O(h/τ)."""
    br, fill, ti, to, g = 1.24, 0.7, 3.17, 3.17, 1.4
    layers, h = coupling_slab(br, br, fill, fill, ti, to, g)
    tau = 1e4 * h
    got = backiron_flux_per_depth("outer", layers, h, tau, n_max=40001)
    want = flat_circuit_flux_density(br, br, ti, to, g) * fill * tau / 2
    assert rel_err(got, want) < 1e-3, mismatch("slab, long pitch", "reference", got, want, 1e-3)


@pytest.mark.family("materials")
def test_sanity_slab_mirror_symmetry():
    """Identical rings: the inner and outer plates carry the same flux (mirror symmetry y → h − y)."""
    layers, h = coupling_slab(1.24, 1.24, 0.75, 0.75, 3.0, 3.0, 1.2)
    outer = backiron_flux_per_depth("outer", layers, h, 8.8)
    inner = backiron_flux_per_depth("inner", layers, h, 8.8)
    assert rel_err(inner, outer) < 1e-12, mismatch("slab symmetry", "reference", inner, outer, 1e-12)


@pytest.mark.family("materials")
@pytest.mark.parametrize("surface", ["outer", "inner"])
def test_sanity_iron_face_is_limit_of_general_solution(surface):
    """iron_face_coefficients against the module's general solution harmonic_coefficients: with air between the
    last magnet face and a plate, B_y at distance d from that plate is B_face·cosh(k d) (the Green's function's
    y-dependence there), harmonic by harmonic. Unequal, opposed and shifted layers exercise both cos and sin
    terms. Tolerance 1e-12 of the largest harmonic: rounding only."""
    layers = [MagnetLayer(0.6, 2.0, 1.2, 0.8, 0.7), MagnetLayer(3.7, 1.1, -1.1, 0.6, -0.4)]
    h, pitch, d = 5.3, 7.0, 0.05
    n, a_face, b_face = iron_face_coefficients(surface, layers, h, pitch, n_max=201)
    _, a_y, b_y, _, _ = harmonic_coefficients(h - d if surface == "outer" else d, layers, h, pitch, n_max=201)
    growth = np.cosh(n * math.pi / pitch * d)
    scale = float(np.max(np.abs(np.concatenate([a_y, b_y]))))
    err = float(np.max(np.abs(np.concatenate([a_y - a_face * growth, b_y - b_face * growth])))) / scale
    assert err < 1e-12, mismatch(f"iron-face coefficients vs general solution ({surface})", "reference", err, 0.0, 1e-12)


@pytest.mark.family("materials")
def test_sanity_backiron_flux_needs_aligned_layers():
    """The pole-centre symmetry argument only holds for aligned layers; a shifted layer is refused."""
    with pytest.raises(ValueError):
        backiron_flux_per_depth("outer", [MagnetLayer(0.0, 3.0, 1.3, 0.8, 0.5)], 3.0, 9.0)


# --------------------------------------------------------------------------- remanence
@pytest.mark.family("metal")
def test_sanity_br_temperature_scaling():
    """α = −0.12 %/°C: Br(50 °C) = 1.29·(1 − 0.0012·30) = 1.24356 T; Br(100 °C)/Br(20 °C) = 0.904; torque scales
    by 0.904²; moving 20 → 100 → 20 °C returns the start value."""
    assert rel_err(br_at(1.29, -0.0012, 50.0), 1.24356) < TOL_ALGEBRA, \
        mismatch("Br at 50 °C", "reference", br_at(1.29, -0.0012, 50.0), 1.24356, TOL_ALGEBRA)
    assert rel_err(br_ratio(-0.0012, 100.0), 0.904) < TOL_ALGEBRA, \
        mismatch("Br ratio", "reference", br_ratio(-0.0012, 100.0), 0.904, TOL_ALGEBRA)
    t100 = torque_at(2.0, -0.0012, 20.0, 100.0)
    assert rel_err(t100, 2.0 * 0.904 ** 2) < TOL_ALGEBRA, mismatch("torque at 100 °C", "reference", t100, 2.0 * 0.904 ** 2, TOL_ALGEBRA)
    back = torque_at(t100, -0.0012, 100.0, 20.0)
    assert rel_err(back, 2.0) < TOL_ALGEBRA, mismatch("round trip", "reference", back, 2.0, TOL_ALGEBRA)


# --------------------------------------------------------------------------- metal_stack
@pytest.mark.family("metal")
def test_sanity_duty_and_power():
    """20 poles are 10 pole pairs. 2 poles (one pole pair) at 60 rpm pass 1 pole pair per second. 60/(2π) rpm is
    1 rad/s, and 1 N·m at that speed is 1 W."""
    assert rel_err(pole_pairs(20), 10.0) < TOL_ALGEBRA, mismatch("pole pairs", "reference", pole_pairs(20), 10.0, TOL_ALGEBRA)
    assert rel_err(pole_pair_frequency_Hz(2, 60.0), 1.0) < TOL_ALGEBRA, \
        mismatch("pole-pair frequency", "reference", pole_pair_frequency_Hz(2, 60.0), 1.0, TOL_ALGEBRA)
    w = omega_rad_s(60 / (2 * math.pi))
    assert rel_err(w, 1.0) < TOL_ALGEBRA, mismatch("angular speed", "reference", w, 1.0, TOL_ALGEBRA)
    p = shaft_power_W(1.0, 60 / (2 * math.pi))
    assert rel_err(p, 1.0) < TOL_ALGEBRA, mismatch("shaft power", "reference", p, 1.0, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_sanity_gearbox_power_balance():
    """0.6 N·m in at 3000 rpm through 5:1 at η = 0.95: the output turns at 600 rpm and carries 95 % of the input
    power, so T_out = 0.95·P_in/ω_out = 2.85 N·m. The input form inverts the output form."""
    want = 0.95 * shaft_power_W(0.6, 3000.0) / (2 * math.pi * 600.0 / 60)
    got = gearbox_output_torque(0.6, 5.0, 0.95)
    assert rel_err(got, want) < TOL_ALGEBRA, mismatch("gearbox output torque", "reference", got, want, TOL_ALGEBRA)
    back = gearbox_input_torque(got, 5.0, 0.95)
    assert rel_err(back, 0.6) < TOL_ALGEBRA, mismatch("gearbox round trip", "reference", back, 0.6, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_sanity_plating_and_stacks():
    """A 10 mm pin plated 0.015 mm becomes 10.030 mm; a 10 mm bore becomes 9.970 mm; a pin machined with
    preplate_size plates back to 10 mm. A 1.0 mm corner gap minus 0.1 + 0.2 walls and 2 × 0.025 bedding
    leaves 0.65 mm; 0.4 + 0.05 + 0.05 stacks to 0.5."""
    assert rel_err(plated_size(10.0, 0.015, 2, True), 10.03) < TOL_ALGEBRA, \
        mismatch("plated pin", "reference", plated_size(10.0, 0.015, 2, True), 10.03, TOL_ALGEBRA)
    assert rel_err(plated_size(10.0, 0.015, 2, False), 9.97) < TOL_ALGEBRA, \
        mismatch("plated bore", "reference", plated_size(10.0, 0.015, 2, False), 9.97, TOL_ALGEBRA)
    pre = preplate_size(10.0, 0.015, 2, True)
    assert rel_err(plated_size(pre, 0.015, 2, True), 10.0) < TOL_ALGEBRA, \
        mismatch("pre-plate round trip", "reference", plated_size(pre, 0.015, 2, True), 10.0, TOL_ALGEBRA)
    c = sleeve_liner_clearance(1.0, 0.1, 0.2, 0.025, 0.025)
    assert rel_err(c, 0.65) < TOL_ALGEBRA, mismatch("clearance stack", "reference", c, 0.65, TOL_ALGEBRA)
    assert rel_err(linear_stack(0.4, 0.05, 0.05), 0.5) < TOL_ALGEBRA, \
        mismatch("linear stack", "reference", linear_stack(0.4, 0.05, 0.05), 0.5, TOL_ALGEBRA)
```

- [ ] **Step 3: Run the sanity test**

Run: `cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_geometry_metal_reference_sanity.py -k sanity`
Expected: PASS (`19 passed`, about 0.5 s).

- [ ] **Step 4: Write the engine checks**

File: `reference/magcoupling-py/audit/tests/test_geometry_mass.py` (create)

```python
"""Geometry, mass and inertia checks (Calculator rows 9, 33, 38, 51–67, 110–115) against the
independent block-geometry reference. A failing check is a candidate finding for Task 8."""
from __future__ import annotations

import math
import re

import pytest

from audit.common import TOL_ALGEBRA, defaults, mismatch, rel_err, run, vary
from audit.references.block_geometry import (axial_overlap, geometry_for_design, retainer_geometry_for_design,
                                             retainer_parts_for_design, ring_min_clearance, rotating_parts_for_design)

#: Physically valid layouts (blocks fit on both polygons) that exercise N, apothem and gap.
CONFIGS = {
    "defaults": {},
    "8 poles": {"coupling.npole": 8},
    "12 poles, 12.5 mm apothem": {"coupling.npole": 12, "coupling.inner_back_apothem_mm": 12.5},
    "1.0 mm gap": {"metal.face_gap_mm": 1.0},
    "2.0 mm gap": {"metal.face_gap_mm": 2.0},
}

#: (engine field on res.model, workbook cell, reference attribute on CouplingGeometry)
GEOMETRY_FIELDS = [
    ("corner_gap_mm", "Calculator!C9", "corner_gap"),
    ("hub_wall_mm", "Calculator!C38", "hub_wall"),
    ("inner_flat_width_mm", "Calculator!C51", "inner_flat_width"),
    ("hub_wall_past_key_mm", "Calculator!C53", "hub_wall_past_key"),
    ("inner_face_radius_mm", "Calculator!C54", "inner_face_radius"),
    ("inner_corner_radius_mm", "Calculator!C55", "inner_corner_radius"),
    ("outer_face_apothem_mm", "Calculator!C56", "outer_face_apothem"),
    ("face_gap_mm", "Calculator!C57", "face_gap"),
    ("outer_flat_width_mm", "Calculator!C58", "outer_flat_width"),
    ("outer_back_apothem_mm", "Calculator!C60", "outer_back_apothem"),
    ("pocket_corner_radius_mm", "Calculator!C61", "pocket_corner_radius"),
    ("cup_od_mm", "Calculator!C62", "cup_od"),
    ("gap_radius_mm", "Calculator!C64", "gap_radius"),
    ("pole_pitch_mm", "Calculator!C65", "pole_pitch"),
    ("fill_inner", "Calculator!C66", "fill_inner"),
    ("fill_outer", "Calculator!C67", "fill_outer"),
]

#: Swing-arm radius of the added-inertia estimate (Calculator C115 label: "at 0.25 m from the swing axis").
SWING_RADIUS_M = 0.25


@pytest.mark.family("geometry")
@pytest.mark.parametrize("name", list(CONFIGS))
def test_corner_gap_is_minimum_ring_clearance(name):
    """Calculator!C9 equals the smallest 2D distance between the inner and outer block rings over every
    relative rotation, measured numerically on explicit block rectangles placed from the inputs (outer
    faces at inner back apothem + thickness + flat-face gap). Tolerance 1e-8: the bounded Brent search
    locates a smooth quadratic minimum to 1e-12 rad."""
    inp = vary(defaults(), CONFIGS[name])
    m = run(inp).model
    outer_face = inp.coupling.inner_back_apothem_mm + m.inner_thickness_mm + inp.metal.face_gap_mm
    ref = ring_min_clearance(inp.coupling.npole, inp.coupling.inner_back_apothem_mm, m.inner_thickness_mm,
                             m.inner_width_mm, outer_face, m.outer_thickness_mm, m.outer_width_mm)
    assert rel_err(m.corner_gap_mm, ref) < 1e-8, mismatch(f"corner gap vs 2D ring clearance ({name})", "Calculator!C9",
                                                           m.corner_gap_mm, ref, 1e-8)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("field,cell,attr", GEOMETRY_FIELDS, ids=[f[1] for f in GEOMETRY_FIELDS])
def test_block_geometry_rederived(name, field, cell, attr):
    """Every polygon/block dimension re-derived by construction (block rectangles, pocket polygon from
    intersecting side lines, vertex radii, side lengths) from the flat-face gap and block data. The fill
    factors are block width over the pole arc at the block's mid-thickness radius, the planar unrolling the
    workbook's slab model uses."""
    inp = vary(defaults(), CONFIGS[name])
    res = run(inp)
    got, ref = getattr(res.model, field), getattr(geometry_for_design(inp, res), attr)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"{field} ({name})", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("changes", [{}, {"coupling.magnets.part_inner": "BX042SH"}], ids=["defaults", "25.4 mm inner"])
def test_active_length_is_axial_overlap(changes):
    """Calculator!C33: the active length is the axial overlap of the two rings' blocks, both centred on the
    same mid-plane (interval intersection)."""
    res = run(vary(defaults(), changes))
    m = res.model
    ref = axial_overlap(m.inner_length_mm, m.outer_length_mm)
    assert rel_err(m.active_length_mm, ref) < TOL_ALGEBRA, mismatch(f"active length ({changes or 'defaults'})", "Calculator!C33",
                                                                     m.active_length_mm, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_cup_wall_at_flats_is_steel_only():
    """Calculator!C63 'Ring wall at the flats' should be the steel between the pocket flat and the OD. The pocket
    flat sits at the outer block back plus the outer bondline (the engine's own pocket apothem, used for C61),
    so the wall is OD/2 − (C60 + bond_outer), as the inner hub wall C38 excludes the inner bondline. The engine
    subtracts C60 only, counting the bondline as steel."""
    inp = defaults()
    res = run(inp)
    ref = geometry_for_design(inp, res).cup_wall_flat
    got = res.model.cup_wall_flat_mm
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("cup wall at the flats", "Calculator!C63, Calculator!C60, Metal design!C121",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_arc_mode_uses_arc_geometry():
    """One root cause, three symptoms. With arcs (coupling.faceted = 0) the inner ring has no block corners: its
    largest radius is the arc OD a + t, and the pocket is round. Reference, by construction:
    (1) the effective gap Calculator!C57 equals the flat-face gap input Metal design!C119, because the arc gap is
        uniform; the engine subtracts the flat-block corner protrusion √((a+t)² + (w/2)²) − (a+t) (C9 'formula
        always uses the corner geometry');
    (2) the sleeve bore Metal design!C175 clears the arc OD, 2·(a + t + bedding), not flat-block corners;
    (3) the cup cavity in Calculator!C111 is the round pocket the engine itself uses for C61 and the OD
        (radius C60 + bond, no 1/cos(π/N)), not the N-gon of the same apothem. Built on the engine's
        placement C60/C62, so this item isolates the cavity shape.
    All three are listed in the failure message, failing or not."""
    inp = vary(defaults(), {"coupling.faceted": 0})
    res = run(inp)
    items = [
        ("(1) effective gap vs the input gap", "Calculator!C57, Calculator!C9, Metal design!C119",
         res.model.face_gap_mm, geometry_for_design(inp, res).face_gap),
        ("(2) sleeve bore over the arc OD", "Metal design!C175",
         res.retainers.sleeve_id_mm, retainer_geometry_for_design(inp, res).sleeve_id),
        ("(3) cup mass with a round pocket", "Calculator!C111, Calculator!C61",
         res.mass.cup_g, rotating_parts_for_design(inp, res)["cup"].mass_g),
    ]
    all_ok = all(rel_err(got, ref) < TOL_ALGEBRA for _, _, got, ref in items)
    assert all_ok, "arc mode (faceted = 0) reuses flat-block geometry: " + " || ".join(
        mismatch(what, cells, got, ref, TOL_ALGEBRA) for what, cells, got, ref in items)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("changes,inner_want,outer_want", [
    ({}, "OK", "OK"),
    ({"coupling.npole": 12}, "TOO NARROW", "OK"),
    ({"coupling.faceted": 0}, "n/a", "n/a"),
])
def test_flat_fit_checks(changes, inner_want, outer_want):
    """Calculator!C52/C59: a block fits its polygon when the polygon side at the block's tight plane (inner: the
    block back; outer: the block face) is at least the block width; the reported slack is side − width.
    Arcs report n/a."""
    inp = vary(defaults(), changes)
    res = run(inp)
    m, g = res.model, geometry_for_design(inp, res)
    for got, want, side, width, cell in ((m.inner_flat_check, inner_want, g.inner_flat_width, m.inner_width_mm, "Calculator!C52"),
                                         (m.outer_flat_check, outer_want, g.outer_flat_width, m.outer_width_mm, "Calculator!C59")):
        fits = side >= width
        assert got.startswith(want), mismatch(f"fit verdict {got!r}, reference {want!r} (fits={fits})", cell,
                                              float(got.startswith(want)), 1.0, 0.0)
        if want == "OK":
            shown = float(re.search(r"(-?\d+\.\d+) mm", got).group(1))
            assert abs(shown - (side - width)) <= 0.005 + 1e-12, \
                mismatch(f"slack shown in {got!r} (2-decimal text, abs tol 0.005 mm)", cell, shown, side - width, 5e-3)


@pytest.mark.family("geometry")
@pytest.mark.parametrize("name", ["defaults", "8 poles", "12 poles, 12.5 mm apothem"])
@pytest.mark.parametrize("part,field,cell", [("magnets", "magnets_g", "Calculator!C110"), ("cup", "cup_g", "Calculator!C111"),
                                             ("hub", "hub_g", "Calculator!C112"), ("boss", "boss_g", "Calculator!C113")],
                         ids=["C110", "C111", "C112", "C113"])
def test_component_masses(name, part, field, cell):
    """Gross solid masses from explicit shapes: 2N block rectangles (7.5 g/cm³ NdFeB), cup = disk(OD) minus the
    shoelace area of the pocket polygon over the cavity depth plus the web disk minus the bore, hub polygon
    minus bore, boss tube. Built on the engine's ring placement (C56, C60, C62) so only the mass formula is
    tested. Holes, keyways and threads are not subtracted (the workbook's documented simplification)."""
    inp = vary(defaults(), CONFIGS[name])
    res = run(inp)
    ref = rotating_parts_for_design(inp, res)[part].mass_g
    got = getattr(res.mass, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"{part} mass ({name})", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_total_mass():
    """Calculator!C114 = magnets + cup + hub + boss + sleeve and liner + hardware allowance + cap + endplates,
    every solid from the independent references."""
    inp = defaults()
    res = run(inp)
    parts = rotating_parts_for_design(inp, res) | retainer_parts_for_design(inp, res)
    ref = math.fsum(p.mass_g for p in parts.values()) + inp.metal.hardware_g
    assert rel_err(res.mass.total_g, ref) < TOL_ALGEBRA, mismatch("total rotating mass", "Calculator!C114", res.mass.total_g, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_no_back_iron_hub_is_aluminium():
    """Calculator!C112 with coupling.backiron = 0 ('no intentional back iron'): the hub is the aluminium carrier
    (Metal design!C42), as the workbook states. (Whether the steel cup is consistent with this mode is
    test_materials.py::test_no_back_iron_mode_treats_the_cup_consistently.)"""
    inp = vary(defaults(), {"coupling.backiron": 0})
    res = run(inp)
    ref = rotating_parts_for_design(inp, res)["hub"].mass_g
    got = res.mass.hub_g
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("hub mass without back iron", "Calculator!C112, Calculator!C6",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_arc_mode_hub_is_round():
    """Calculator!C112 with arcs (coupling.faceted = 0): the hub is round, π·r² with r the hub apothem."""
    inp = vary(defaults(), {"coupling.faceted": 0})
    res = run(inp)
    ref = rotating_parts_for_design(inp, res)["hub"].mass_g
    got = res.mass.hub_g
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("arc-mode hub mass", "Calculator!C112", got, ref, TOL_ALGEBRA)


@pytest.mark.family("geometry")
def test_added_inertia_point_mass():
    """Calculator!C115 is a point-mass estimate, m·d² at d = 0.25 m. Reference: parallel-axis theorem,
    I = m·d² + J_own, with J_own the coupling's polar inertia about its own axis (assumed parallel to the
    swing axis) summed over every solid, and the 6 g hardware allowance put at the cup OD radius (upper
    bound). Tolerance 1 %: the check confirms the neglected own-axis term is below 1 % of the result."""
    inp = defaults()
    res = run(inp)
    parts = rotating_parts_for_design(inp, res) | retainer_parts_for_design(inp, res)
    mass_g = math.fsum(p.mass_g for p in parts.values()) + inp.metal.hardware_g
    j_own_g_mm2 = (math.fsum(p.polar_inertia_g_mm2 for p in parts.values())
                   + inp.metal.hardware_g * (res.model.cup_od_mm / 2) ** 2)
    ref = mass_g / 1000 * SWING_RADIUS_M ** 2 + j_own_g_mm2 * 1e-9
    got = res.mass.added_inertia_kgm2
    assert rel_err(got, ref) < 1e-2, mismatch(f"added inertia (J_own = {j_own_g_mm2 * 1e-9:.3e} kg·m²)", "Calculator!C115",
                                              got, ref, 1e-2)
```

File: `reference/magcoupling-py/audit/tests/test_metal_design.py` (create)

```python
"""Metal design checks (torque band, requirements, radial clearance stack, retainers, axial envelope,
slip duty, adapter variant) and the Calculator gearbox rows C99/C100, against independent
re-derivations. A failing check is a candidate finding for Task 8.

Owned elsewhere (not repeated here): the Br² temperature law on Calculator C93 (Task 3) and the
verdict thresholds Metal design C11 and C37 (Task 7). The values those verdicts compare,
C9, C12 and C35, are checked here."""
from __future__ import annotations

import math

import pytest

from audit.common import TOL_ALGEBRA, defaults, mismatch, rel_err, run, vary
from audit.references.block_geometry import (annulus_part, geometry_for_design, retainer_geometry_for_design,
                                             retainer_parts_for_design)
from audit.references.metal_stack import (gearbox_input_torque, gearbox_output_torque, linear_stack,
                                          pole_pair_frequency_Hz, shaft_power_W, sleeve_liner_clearance, torque_high,
                                          torque_low)
from audit.references.remanence import br_ratio, torque_at

#: 7075-T6 density, 2.81 g/cm³ (Aluminum Association, *Aluminum Standards and Data*; ASM Handbook Vol. 2).
AL7075_DENSITY_G_MM3 = 2.81e-3


# --------------------------------------------------------------------------- torque band
@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("torque_cold_Nm", "Metal design!C8"), ("torque_hot_low_Nm", "Metal design!C9"),
                                        ("torque_cold_high_Nm", "Metal design!C10"),
                                        ("torque_cold_zero_var_Nm", "Metal design!C112"),
                                        ("cold_high_Nm", "Metal design!C157")], ids=["C8", "C9", "C10", "C112", "C157"])
def test_torque_band(field, cell):
    """Hot low = pull-out at the hot (operating) temperature × (1 − v); cold = pull-out moved to the minimum
    temperature with torque ∝ Br(T)²; cold high = cold × (1 + v). The reference starts from the 50 °C
    pull-out (C93) while the engine starts from the 20 °C value (C94), so the C94 scaling is cross-checked."""
    inp = defaults()
    res = run(inp)
    md, a = inp.metal, inp.calibration.alpha_br_per_C
    cold = torque_at(res.model.pullout_Nm, a, inp.coupling.op_temp_C, md.min_temp_C)
    ref = {"torque_cold_Nm": cold, "torque_hot_low_Nm": torque_low(res.model.pullout_Nm, md.variation),
           "torque_cold_high_Nm": torque_high(cold, md.variation), "torque_cold_zero_var_Nm": cold,
           "cold_high_Nm": torque_high(cold, md.variation)}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_cold_torque_equals_model_run_cold():
    """Metal design!C8 (cold torque by Br² scaling) equals the Calculator pull-out computed directly at the
    minimum temperature (an independent path through the engine's own torque model)."""
    inp = defaults()
    got = run(inp).metal.torque_cold_Nm
    ref = run(vary(inp, {"coupling.op_temp_C": inp.metal.min_temp_C})).model.pullout_Nm
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("cold torque vs model run at min temp", "Metal design!C8, Calculator!C93",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("required_20C_Nm", "Metal design!C19"), ("required_20C_zero_scatter_Nm", "Metal design!C111"),
                                        ("hot_margin", "Metal design!C20"), ("cold_for_hot_min_Nm", "Metal design!C155"),
                                        ("cold_for_hot_min_input_Nm", "Metal design!C156"),
                                        ("cold_high_input_Nm", "Metal design!C158"),
                                        ("noiron_baseline_hot_Nm", "Metal design!C151")],
                         ids=["C19", "C111", "C20", "C155", "C156", "C158", "C151"])
def test_requirement_derivations(field, cell):
    """Requirements by inverting the torque band: the 20 °C nominal that leaves hot low = required
    (T20·(Br(hot)/Br20)²·(1 − v) = T_req), the same with v = 0, the nominal hot margin, the cold torque of a
    design whose hot nominal equals the requirement, gearbox-input equivalents T/(i·η) (driving), and the
    no-iron prototype moved from its test temperature to the operating temperature."""
    inp = defaults()
    res = run(inp)
    md, c, cal = inp.metal, inp.coupling, inp.calibration
    a, hot, req = cal.alpha_br_per_C, c.op_temp_C, md.required_min_Nm
    cold_for_min = torque_at(req, a, hot, md.min_temp_C)
    cold_high = torque_high(torque_at(res.model.pullout_Nm, a, hot, md.min_temp_C), md.variation)
    ref = {
        "required_20C_Nm": req / (br_ratio(a, hot) ** 2 * (1 - md.variation)),
        "required_20C_zero_scatter_Nm": torque_at(req, a, hot, 20.0),
        "hot_margin": res.model.pullout_Nm / req - 1,
        "cold_for_hot_min_Nm": cold_for_min,
        "cold_for_hot_min_input_Nm": gearbox_input_torque(cold_for_min, c.gear_ratio, c.gear_efficiency),
        "cold_high_input_Nm": gearbox_input_torque(cold_high, c.gear_ratio, c.gear_efficiency),
        "noiron_baseline_hot_Nm": torque_at(cal.measured_torque_Nm, a, cal.test_temp_C, hot),
    }[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


#: Gearbox scenarios: defaults, and another ratio and efficiency in the free-space circuit (a different C93).
GEAR_SCENARIOS = {"defaults": {},
                  "ratio 7, efficiency 0.8, free space": {"coupling.gear_ratio": 7, "coupling.gear_efficiency": 0.8,
                                                          "coupling.backiron": 0}}


@pytest.mark.family("torque")
@pytest.mark.parametrize("cell", ["Calculator!C99", "Calculator!C100"])
def test_gearbox_equivalents(cell):
    """Calculator!C99: the gearbox input torque while the coupling carries its pull-out torque C93, T/(i·η)
    (motor driving, the conservative form also used for Metal design C156/C158). Calculator!C100: the
    output-side torque that loads the gearbox input to its rating C47, i·η·T_rating (power balance, same
    convention), so C99 ≤ C47 exactly when C93 ≤ C100. Sole owner of C99 and C100; one check per cell covers
    every GEAR_SCENARIOS entry and lists each failing scenario."""
    failures = []
    for name, changes in GEAR_SCENARIOS.items():
        inp = vary(defaults(), changes)
        res = run(inp)
        c = inp.coupling
        if cell == "Calculator!C99":
            got, ref = res.model.gearbox_input_ripple_Nm, gearbox_input_torque(res.model.pullout_Nm, c.gear_ratio, c.gear_efficiency)
        else:
            got, ref = res.model.gearbox_reference_Nm, gearbox_output_torque(c.gearbox_input_rating_Nm, c.gear_ratio, c.gear_efficiency)
        if not rel_err(got, ref) < TOL_ALGEBRA:
            failures.append(mismatch(f"gearbox equivalent ({name})", f"{cell}, Calculator!C45, C46, C47", got, ref, TOL_ALGEBRA))
    assert not failures, " | ".join(failures)


# --------------------------------------------------------------------------- radial clearance stack
@pytest.mark.family("metal")
@pytest.mark.parametrize("changes", [{}, {"metal.shaft_displacement_mm": 0.1}, {"metal.face_gap_mm": 2.0}], ids=["defaults", "shaft 0.1", "gap 2.0"])
@pytest.mark.parametrize("field,cell", [("sleeve_liner_clearance_mm", "Metal design!C27"), ("nominal_sleeve_liner_mm", "Metal design!C143"),
                                        ("adverse_movement_mm", "Metal design!C34"), ("min_running_clearance_mm", "Metal design!C35"),
                                        ("running_clearance_mm", "Metal design!C12"), ("allowed_radial_disp_mm", "Metal design!C144")],
                         ids=["C27", "C143", "C34", "C35", "C12", "C144"])
def test_radial_clearance_stack(changes, field, cell):
    """Nominal sleeve-to-liner clearance = reference corner gap − sleeve (0.10) − liner (0.20) − both beddings;
    adverse movement = linear sum of shaft float, runout, deflection, thermal, sleeve form and magnet
    position; running clearance = nominal − adverse; allowed shaft displacement = nominal − the other five
    allowances − the residual target."""
    inp = vary(defaults(), changes)
    res = run(inp)
    md = inp.metal
    nominal = sleeve_liner_clearance(geometry_for_design(inp, res).corner_gap, md.sleeve_mm, md.liner_mm,
                                     md.sleeve_bedding_mm, md.liner_bedding_mm)
    others = [md.runout_mm, md.deflection_mm, md.thermal_mm, md.sleeve_form_mm, md.magnet_position_mm]
    adverse = linear_stack(md.shaft_displacement_mm, *others)
    ref = {"sleeve_liner_clearance_mm": nominal, "nominal_sleeve_liner_mm": nominal, "adverse_movement_mm": adverse,
           "min_running_clearance_mm": nominal - adverse, "running_clearance_mm": nominal - adverse,
           "allowed_radial_disp_mm": nominal - linear_stack(*others) - md.residual_target_mm}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"{field} ({changes or 'defaults'})", cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- retainers
@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell,attr", [("sleeve_id_mm", "Metal design!C175", "sleeve_id"), ("sleeve_od_mm", "Metal design!C176", "sleeve_od"),
                                             ("liner_od_mm", "Metal design!C177", "liner_od"), ("liner_id_mm", "Metal design!C178", "liner_id"),
                                             ("endplate_od_mm", "Metal design!C179", "endplate_od")],
                         ids=["C175", "C176", "C177", "C178", "C179"])
def test_retainer_geometry(field, cell, attr):
    """Sleeve bore clears the inner ring's largest radius (block corners, from explicit rectangles) by the
    bedding clearance; liner OD clears the outer ring's smallest radius (point-to-segment distance from the
    axis to the outer block faces); walls add inward/outward; endplates cap the magnets to the sleeve bore.
    (Arc mode is test_geometry_mass.py::test_arc_mode_uses_arc_geometry.)"""
    inp = defaults()
    res = run(inp)
    ref = getattr(retainer_geometry_for_design(inp, res), attr)
    got = getattr(res.retainers, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell,parts", [("retainers_g", "Metal design!C46", ("sleeve", "liner")),
                                              ("cap_g", "Metal design!C180", ("cap",)),
                                              ("endplates_g", "Metal design!C181", ("endplates",))],
                         ids=["C46", "C180", "C181"])
def test_retainer_masses(field, cell, parts):
    """316L sleeve and liner tubes over the retainer span; aluminium cap face plate (cap OD to the liner-ID
    opening) plus threaded skirt (cap OD to thread diameter over the engagement); front endplate with the
    bore opening and rear endplate with the screw-clearance opening. Metal design!C45 span and C150 link
    are checked with the other links."""
    inp = defaults()
    res = run(inp)
    rp = retainer_parts_for_design(inp, res)
    ref = math.fsum(rp[p].mass_g for p in parts)
    got = getattr(res.retainers, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- axial envelope
@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("axial_stack_mm", "Metal design!C134"), ("rotating_od_mm", "Metal design!C135"),
                                        ("diameter_reserve_mm", "Metal design!C136"), ("large_dia_stack_mm", "Metal design!C137"),
                                        ("large_dia_reserve_mm", "Metal design!C138"), ("axial_reserve_mm", "Metal design!C139"),
                                        ("hybrid_length_mm", "Metal design!C192")],
                         ids=["C134", "C135", "C136", "C137", "C138", "C139", "C192"])
def test_axial_envelope(field, cell):
    """Axial stacks: the large-diameter region is cap face + cup cavity + rear web (the cap skirt overlaps the
    cup body); overall adds the boss; the hybrid replaces the boss with adapter flange + boss extension.
    Reserves are envelope minus stack: 20 mm bay, 35 mm overall, 43 mm diameter; the rotating OD is the
    larger of the cup body and the cap."""
    inp = defaults()
    res = run(inp)
    md = inp.metal
    large = linear_stack(md.cap_axial_mm, md.cup_depth_mm, md.web_mm)
    overall = linear_stack(large, md.boss_length_mm)
    rot_od = max(geometry_for_design(inp, res).cup_od, md.cap_od_mm)
    ref = {"axial_stack_mm": overall, "rotating_od_mm": rot_od, "diameter_reserve_mm": md.max_diameter_mm - rot_od,
           "large_dia_stack_mm": large, "large_dia_reserve_mm": md.max_large_dia_axial_mm - large,
           "axial_reserve_mm": md.max_overall_axial_mm - overall,
           "hybrid_length_mm": linear_stack(large, md.adapter_flange_mm, md.adapter_boss_mm)}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- slip duty
@pytest.mark.family("metal")
def test_slip_duty():
    """Pole-pair frequency (N/2)·rpm/60 (Metal design!C86, also the ripple frequency Calculator!C97) and
    magnetic cycles over life = frequency × event duration × events (C89)."""
    inp = defaults()
    res = run(inp)
    md = inp.metal
    f = pole_pair_frequency_Hz(inp.coupling.npole, md.slip_rpm)
    for got, ref, cell in ((res.metal.slip_freq_Hz, f, "Metal design!C86"), (res.model.ripple_freq_Hz, f, "Calculator!C97"),
                           (res.metal.magnetic_cycles, f * md.slip_event_s * md.life_events, "Metal design!C89")):
        assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("slip duty", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
@pytest.mark.parametrize("drag", [None, 0.04, 0.25])
def test_slip_loss_from_measured_drag(drag):
    """Slip loss P = T_drag·2π·rpm/60 (Metal design!C91) and energy per event P·t (C92); 'not measured' until a
    bench drag torque is entered."""
    inp = vary(defaults(), {"metal.measured_drag_Nm": drag})
    m = run(inp).metal
    if drag is None:
        for got, cell in ((m.slip_loss_W, "Metal design!C91"), (m.slip_energy_J, "Metal design!C92")):
            assert got == "not measured", mismatch(f"slip loss placeholder {got!r}", cell, float(got == "not measured"), 1.0, 0.0)
        return
    p = shaft_power_W(drag, inp.metal.slip_rpm)
    for got, ref, cell in ((m.slip_loss_W, p, "Metal design!C91"), (m.slip_energy_J, p * inp.metal.slip_event_s, "Metal design!C92")):
        assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"slip loss at {drag} N·m", cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- adapter variant
def _adapter_mass(md, bore: float, density: float) -> float:
    """Optional aluminium adapter as three tubes on the bore: flange, boss extension (boss OD) and pilot."""
    return (annulus_part(md.adapter_flange_dia_mm, bore, md.adapter_flange_mm, density).mass_g
            + annulus_part(md.boss_od_mm, bore, md.adapter_boss_mm, density).mass_g
            + annulus_part(md.adapter_pilot_dia_mm, bore, md.adapter_pilot_mm, density).mass_g)


@pytest.mark.family("metal")
@pytest.mark.parametrize("field,cell", [("adapter_g", "Metal design!C188"), ("adapter_steel_removed_g", "Metal design!C189"),
                                        ("hybrid_mass_g", "Metal design!C191"), ("adapter_variant_mass_g", "Metal design!C148"),
                                        ("adapter_mass_saved_g", "Metal design!C149")],
                         ids=["C188", "C189", "C191", "C148", "C149"])
def test_adapter_variant(field, cell):
    """Hybrid = steel-cup total − integral boss − web ring opened to the adapter pilot bore + adapter + its
    hardware, with the adapter at the workbook's aluminium density (Metal design!C42)."""
    inp = defaults()
    res = run(inp)
    md, bore = inp.metal, inp.coupling.bore_mm
    adapter = _adapter_mass(md, bore, md.al_density_g_mm3)
    removed = annulus_part(md.adapter_pilot_dia_mm, bore, md.web_mm, md.steel_density_g_mm3).mass_g
    hybrid = res.mass.total_g - res.mass.boss_g - removed + adapter + md.adapter_hardware_g
    ref = {"adapter_g": adapter, "adapter_steel_removed_g": removed, "hybrid_mass_g": hybrid,
           "adapter_variant_mass_g": hybrid, "adapter_mass_saved_g": res.mass.total_g - hybrid}[field]
    got = getattr(res.metal, field)
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(field, cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("metal")
def test_adapter_mass_with_7075_density():
    """The Materials plan (materials.py docstring) puts the clamp collars and adapters in 7075-T6 (2.81 g/cm³)
    and the cap in 6061-T6 (2.70 g/cm³), but Metal design!C42 gives the adapter the 6061 density.
    Reference: adapter at 7075 density."""
    inp = defaults()
    ref = _adapter_mass(inp.metal, inp.coupling.bore_mm, AL7075_DENSITY_G_MM3)
    got = run(inp).metal.adapter_g
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("adapter mass at 7075 density", "Metal design!C188, Metal design!C42",
                                                      got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- links
@pytest.mark.family("metal")
@pytest.mark.parametrize("cell", ["Calculator!C35", "Metal design!C5", "Metal design!C6", "Metal design!C15", "Metal design!C17", "Metal design!C24",
                                  "Metal design!C45", "Metal design!C47", "Metal design!C140", "Metal design!C141",
                                  "Metal design!C142", "Metal design!C147", "Metal design!C150", "Metal design!C165",
                                  "Metal design!C167"])
def test_linked_cells(cell):
    """Cells that link another sheet or an input: each must carry exactly its source value."""
    inp = defaults()
    res = run(inp)
    m, md, g = res.metal, inp.metal, geometry_for_design(inp, res)
    got, ref = {
        "Calculator!C35": (res.model.alpha_br_per_C, inp.calibration.alpha_br_per_C),
        "Metal design!C5": (m.torque_op_Nm, res.model.pullout_Nm),
        "Metal design!C6": (m.torque_20C_Nm, res.model.pullout_20C_Nm),
        "Metal design!C15": (m.op_temp_C, inp.coupling.op_temp_C),
        "Metal design!C17": (m.alpha_br_per_C, inp.calibration.alpha_br_per_C),
        "Metal design!C24": (m.corner_gap_mm, g.corner_gap),
        "Metal design!C45": (res.retainers.retainer_span_mm, md.retainer_span_mm),
        "Metal design!C47": (m.rotating_mass_g, res.mass.total_g),
        "Metal design!C140": (m.installed_magnets, 2 * inp.coupling.npole),
        "Metal design!C141": (m.assembled_face_gap_mm, md.face_gap_mm),
        "Metal design!C142": (m.corner_clearance_mm, g.corner_gap),
        "Metal design!C147": (m.steel_cup_mass_g, res.mass.total_g),
        "Metal design!C150": (m.retainers_mass_g, res.retainers.retainers_g),
        "Metal design!C165": (m.cup_body_od_mm, g.cup_od),
        "Metal design!C167": (m.cap_face_mm, md.cap_axial_mm),
    }[cell]
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("linked value", cell, got, ref, TOL_ALGEBRA)
```

File: `reference/magcoupling-py/audit/tests/test_materials.py` (create)

```python
"""Materials checks: back-iron thickness by flux conservation (Calculator C103–C106, Materials C20–C22),
the no-back-iron mode, electroless-nickel pre-plate offsets (Materials C27–C30) and material constants
against literature. The magnet temperature ratings (Calculator C22, C32, C107, C108) are Task 5's
(test_temperature_demag_adhesive.py::test_magnet_temperature_checks_follow_kj_rating).
A failing check is a candidate finding for Task 8."""
from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

import pytest

from audit.common import TOL_ALGEBRA, TOL_MODEL, defaults, mismatch, rel_err, run, vary
from audit.references.backiron_flux import (backiron_flux_per_depth, coupling_slab, flat_circuit_flux_density,
                                            sinusoidal_backiron_thickness)
from audit.references.block_geometry import geometry_for_design, rotating_parts_for_design
from audit.references.metal_stack import preplate_size
from audit.references.remanence import br_at

#: 100 % IACS = 58.0 MS/m (International Annealed Copper Standard, IEC 60028).
IACS_S_M = 58.0e6


def _br_op(inp, res) -> tuple[float, float]:
    """Inner and outer remanence at the operating temperature, from the 20 °C values (Calculator C21, C31)."""
    a, t = inp.calibration.alpha_br_per_C, inp.coupling.op_temp_C
    return br_at(res.model.inner_br_T, a, t), br_at(res.model.outer_br_T, a, t)


def _flat_circuit_b(inp, res) -> float:
    br_i, br_o = _br_op(inp, res)
    return flat_circuit_flux_density(br_i, br_o, res.model.inner_thickness_mm, res.model.outer_thickness_mm,
                                     geometry_for_design(inp, res).face_gap)


def _required_thickness_sinusoidal(inp, res) -> float:
    return sinusoidal_backiron_thickness(_flat_circuit_b(inp, res), geometry_for_design(inp, res).pole_pitch,
                                         inp.materials.steel.bsat_T)


def _required_thickness_2d(inp, res, surface: str) -> float:
    """Back-iron thickness from the exact 2D slab field. Fills: the arc-length fills of the reference geometry
    (block width over the pole arc at mid-thickness, the planar unrolling of the rings), the definition the
    geometry checks confirm for C66/C67. The reference and C104 then share the geometry and differ only in the
    flux model."""
    br_i, br_o = _br_op(inp, res)
    g = geometry_for_design(inp, res)
    layers, h = coupling_slab(br_i, br_o, g.fill_inner, g.fill_outer, res.model.inner_thickness_mm,
                              res.model.outer_thickness_mm, g.face_gap)
    return backiron_flux_per_depth(surface, layers, h, g.pole_pitch) / inp.materials.steel.bsat_T


def _ceiling_tenth(x: float) -> float:
    """Round up to the next 0.1 mm in decimal arithmetic."""
    return float(Decimal(repr(x)).quantize(Decimal("0.1"), rounding=ROUND_CEILING))


# --------------------------------------------------------------------------- back iron
@pytest.mark.family("materials")
@pytest.mark.parametrize("changes", [{}, {"coupling.magnets.part_inner": "B842-N52", "coupling.magnets.part_outer": "B861"}],
                         ids=["defaults", "N52 inner, 1.59 mm outer"])
def test_gap_flux_density_flat_circuit(changes):
    """Calculator!C103 against the ideal-iron series circuit B = (Br_i·t_i + Br_o·t_o)/(t_i + t_o + g) at the
    operating temperature (Furlani 2001 §3.3). The engine uses the mean Br times (t_i + t_o), which equals
    the MMF sum only when the rings have equal Br or equal thickness."""
    inp = vary(defaults(), changes)
    res = run(inp)
    ref = _flat_circuit_b(inp, res)
    got = res.model.gap_flux_density_T
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"flat-circuit gap flux density ({changes or 'defaults'})", "Calculator!C103",
                                                      got, ref, TOL_ALGEBRA)


@pytest.mark.family("materials")
@pytest.mark.parametrize("field,cell", [("model", "Calculator!C104"), ("materials", "Materials!C20")])
def test_backiron_requirement_sinusoidal(field, cell):
    """Calculator!C104 / Materials!C20 against flux conservation for a sinusoidal gap field: one pole carries
    2·B̂·τ/π per unit length, half turns each way in the back iron, so t = B̂·τ/(π·B_design), with B̂ the
    flat-circuit value and τ the reference pole pitch at the gap radius."""
    inp = defaults()
    res = run(inp)
    ref = _required_thickness_sinusoidal(inp, res)
    got = res.model.backiron_needed_mm if field == "model" else res.materials.backiron_thickness_needed_mm
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("back-iron thickness, sinusoidal flux", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("materials")
def test_cup_backiron_requirement_vs_2d_slab():
    """Calculator!C104 (compared with the cup corner wall) against the flux entering the cup steel between a
    pole centre and the inter-pole line in the exact 2D field of the steel-backed slab (ideal iron, square-wave
    magnetization, 4001 harmonics, aligned rings = maximum flux), divided by B_design. The inter-pole line is
    at the pocket corners, where the wall is thinnest. Fills as in _required_thickness_2d (arc-length, the
    engine's own definition), so only the flux model differs. Tolerance TOL_MODEL."""
    inp = defaults()
    res = run(inp)
    ref = _required_thickness_2d(inp, res, "outer")
    got = res.model.backiron_needed_mm
    assert rel_err(got, ref) < TOL_MODEL, mismatch("cup back-iron thickness vs 2D slab flux", "Calculator!C104, Materials!C20",
                                                    got, ref, TOL_MODEL)


@pytest.mark.family("materials")
def test_hub_backiron_requirement_vs_2d_slab():
    """The hub check (Calculator!C106) reuses the single requirement C104. Reference: the flux entering the hub
    steel in the same 2D slab solution (same fills). The inner ring's larger fill (C66) puts more flux,
    including the inner magnets' own inter-pole leakage, into the hub than into the cup. Tolerance TOL_MODEL."""
    inp = defaults()
    res = run(inp)
    ref = _required_thickness_2d(inp, res, "inner")
    got = res.model.backiron_needed_mm
    assert rel_err(got, ref) < TOL_MODEL, mismatch("hub back-iron thickness vs 2D slab flux", "Calculator!C104, Calculator!C106",
                                                    got, ref, TOL_MODEL)


@pytest.mark.family("materials")
@pytest.mark.parametrize("wall", [1.8, 2.0, "required", 1.0])
def test_cup_wall_check_logic(wall):
    """Materials!C22: 'OK' when the corner wall ≥ the requirement (boundary included), otherwise ask for the
    requirement rounded up to 0.1 mm. Materials!C21 must carry the wall input. Logic only: the threshold is
    the engine's own requirement C104, whose value test_backiron_requirement_sinusoidal verifies."""
    inp = defaults()
    t_req = run(inp).model.backiron_needed_mm
    wall_mm = t_req if wall == "required" else wall
    res = run(vary(inp, {"metal.cup_wall_corner_mm": wall_mm}))
    want = "OK" if wall_mm >= t_req else f"Too thin: raise Metal design C122 to at least {_ceiling_tenth(t_req):.1f} mm"
    got = res.materials.cup_wall_check
    assert got == want, mismatch(f"cup-wall verdict {got!r}, reference {want!r}", "Materials!C22", float(got == want), 1.0, 0.0)
    assert rel_err(res.materials.cup_wall_corner_mm, wall_mm) < TOL_ALGEBRA, \
        mismatch("cup wall link", "Materials!C21", res.materials.cup_wall_corner_mm, wall_mm, TOL_ALGEBRA)


@pytest.mark.family("materials")
@pytest.mark.parametrize("changes", [{}, {"coupling.backiron": 0}, {"metal.cup_wall_corner_mm": 2.0}],
                         ids=["defaults", "no back iron", "2.0 mm wall"])
def test_calculator_thickness_checks(changes):
    """Calculator!C105 (cup, corner wall) and C106 (hub, wall under the flats): 'No back iron' without steel,
    otherwise 'Thickness OK' when the wall ≥ the requirement, else 'Too thin'. Logic only: the threshold is
    the engine's C104; the hub wall is the reference hub wall."""
    inp = vary(defaults(), changes)
    res = run(inp)
    t_req = res.model.backiron_needed_mm
    g = geometry_for_design(inp, res)
    for got, wall, cell in ((res.model.cup_ring_check, inp.metal.cup_wall_corner_mm, "Calculator!C105"),
                            (res.model.hub_check, g.hub_wall, "Calculator!C106")):
        want = "No back iron" if inp.coupling.backiron == 0 else ("Thickness OK" if wall >= t_req else "Too thin")
        assert got == want, mismatch(f"thickness verdict {got!r}, reference {want!r}", cell, float(got == want), 1.0, 0.0)


@pytest.mark.family("materials")
def test_no_back_iron_mode_treats_the_cup_consistently():
    """coupling.backiron = 0 selects 'no intentional back iron' (Calculator!C6). The torque model then uses the
    free-space factor S_n (Calculator!C93 equals its free-space variant C96) and Calculator!C105 reports 'No back
    iron'. Free-space S_n is valid only when no ferromagnetic material sits behind either ring, yet in the same
    run two outputs keep a flux-carrying steel cup directly behind the outer magnets:
    - Calculator!C111 prices the one-piece cup as steel: C111 divided by the independent cup volume is the
      steel density Metal design!C132, unchanged from the backiron = 1 run (the hub, by contrast, becomes
      aluminium, C112);
    - Materials!C22 sizes that cup wall as back iron, with the same verdict as the backiron = 1 run.
    Reference: with the selector at 0, no output treats the cup as back iron (count 0). Which side needs the
    correction (a non-magnetic cup, or a torque model with steel behind the outer ring) is for the finding to
    decide. The message also gives the pull-out of the same run in both circuits, free space (C96) and steel
    (C95): a steel ring behind only the outer magnets puts the true value between them."""
    base = run(defaults())
    inp = vary(defaults(), {"coupling.backiron": 0})
    res = run(inp)
    cup_volume_mm3 = rotating_parts_for_design(inp, res, cup_density=1.0)["cup"].mass_g
    implied_density = res.mass.cup_g / cup_volume_mm3
    steel = inp.metal.steel_density_g_mm3
    as_back_iron = {
        f"C111 cup at {implied_density * 1000:.3f} g/cm³ (steel {steel * 1000:.3f}), unchanged from backiron = 1":
            rel_err(implied_density, steel) < TOL_ALGEBRA and rel_err(res.mass.cup_g, base.mass.cup_g) < TOL_ALGEBRA,
        f"Materials C22 {res.materials.cup_wall_check!r}, same as backiron = 1":
            res.materials.cup_wall_check == base.materials.cup_wall_check,
    }
    as_absent = {
        "C93 = C96 (free-space S_n)": rel_err(res.model.pullout_Nm, res.model.pullout_noiron_Nm) < TOL_ALGEBRA,
        f"C105 {res.model.cup_ring_check!r}": res.model.cup_ring_check == "No back iron",
    }
    n_iron = sum(as_back_iron.values())
    what = (f"backiron = 0: cup treated as back iron by [{'; '.join(k for k, v in as_back_iron.items() if v)}] "
            f"and as absent by [{'; '.join(k for k, v in as_absent.items() if v)}]; pull-out C96 = "
            f"{res.model.pullout_noiron_Nm:.4f} N·m (free space), C95 = {res.model.pullout_iron_Nm:.4f} N·m (steel)")
    assert n_iron == 0, mismatch(what, "Calculator!C6, C93, C95, C96, C105, C111; Materials!C22; Metal design!C132",
                                 float(n_iron), 0.0, 0.0)


# --------------------------------------------------------------------------- plating
@pytest.mark.family("materials")
@pytest.mark.parametrize("thickness", [0.015, 0.025])
def test_preplate_offsets(thickness):
    """Materials!C27–C30: each machined size is the one that plates to the finished size, with the coating
    growing out of the material: hub flat apothem (external, one surface) under by t, pocket apothem
    (internal, one surface) over by t, bores (internal, two surfaces) over by 2t on diameter, outside
    diameters (external, two surfaces) under by 2t."""
    inp = vary(defaults(), {"materials.nickel.thickness_mm": thickness})
    res = run(inp)
    g, bore = geometry_for_design(inp, res), inp.coupling.bore_mm
    cases = ((res.materials.hub_flats_under_mm, g.hub_apothem - preplate_size(g.hub_apothem, thickness, 1, True), "Materials!C27"),
             (res.materials.cup_pockets_over_mm, preplate_size(g.pocket_apothem, thickness, 1, False) - g.pocket_apothem, "Materials!C28"),
             (res.materials.bores_over_dia_mm, preplate_size(bore, thickness, 2, False) - bore, "Materials!C29"),
             (res.materials.ods_under_dia_mm, g.cup_od - preplate_size(g.cup_od, thickness, 2, True), "Materials!C30"))
    for got, ref, cell in cases:
        assert rel_err(got, ref) < TOL_ALGEBRA, mismatch(f"pre-plate offset at {thickness} mm", cell, got, ref, TOL_ALGEBRA)


# --------------------------------------------------------------------------- constants
@pytest.mark.family("materials")
@pytest.mark.parametrize("cell", ["Calculator!C36", "Calculator!C37"])
def test_calculator_material_links(cell):
    """Calculator!C36 carries the back-iron design flux density (Materials!C13) and C37 the corner wall input
    (Metal design!C122)."""
    inp = defaults()
    m = run(inp).model
    got, ref = {"Calculator!C36": (m.bsat_T, inp.materials.steel.bsat_T),
                "Calculator!C37": (m.cup_wall_corner_mm, inp.metal.cup_wall_corner_mm)}[cell]
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("linked value", cell, got, ref, TOL_ALGEBRA)


@pytest.mark.family("materials")
def test_steel_density_units_consistent():
    """Materials!C19 (g/cm³) and the mass model's Metal design!C132 (g/mm³) must be the same density."""
    inp = defaults()
    got, ref = inp.metal.steel_density_g_mm3, inp.materials.steel.density_g_cm3 / 1000
    assert rel_err(got, ref) < TOL_ALGEBRA, mismatch("steel density units", "Metal design!C132, Materials!C19", got, ref, TOL_ALGEBRA)


LITERATURE = [
    # (label, cell, getter, literature value, tolerance, source)
    ("4140 density g/cm3", "Materials!C19", lambda i: i.materials.steel.density_g_cm3, 7.85, 1e-2, "ASM Handbook Vol. 1, AISI 4140"),
    ("4140 conductivity S/m", "Materials!C14", lambda i: i.materials.steel.conductivity_S_m, 1 / 0.22e-6, 3e-2,
     "AISI 4140 annealed resistivity 0.22 µΩ·m (MatWeb/ASM), two significant figures"),
    ("4140 specific heat J/kgK", "Materials!C16", lambda i: i.materials.steel.specific_heat_J_kgK, 473, 2e-2, "MatWeb AISI 4140: 0.473 J/g·°C"),
    ("4140 CTE 1/C", "Materials!C17", lambda i: i.materials.steel.cte_per_C, 12.3e-6, 2e-2, "ASM: 12.2–12.3 µm/m·°C, 20–100 °C"),
    ("4140 modulus GPa", "Materials!C18", lambda i: i.materials.steel.modulus_GPa, 205, 3e-2, "MatWeb AISI 4140: 190–210 GPa, typical 205"),
    ("7075-T6 yield MPa", "Materials!C34", lambda i: i.materials.aluminium.al7075.yield_MPa, 503, 1e-2, "ASM Handbook Vol. 2, 7075-T6"),
    ("7075-T6 shear MPa", "Materials!C35", lambda i: i.materials.aluminium.al7075.shear_MPa, 331, 1e-2, "ASM Handbook Vol. 2, 7075-T6"),
    ("7075-T6 conductivity S/m", "Materials!C38", lambda i: i.materials.aluminium.al7075.conductivity_S_m, 0.33 * IACS_S_M, 2e-2,
     "7075-T6: 33 % IACS"),
    ("6061-T6 yield MPa", "Materials!C39", lambda i: i.materials.aluminium.al6061.yield_MPa, 276, 1e-2, "ASM Handbook Vol. 2, 6061-T6"),
    ("6061-T6 shear MPa", "Materials!C40", lambda i: i.materials.aluminium.al6061.shear_MPa, 207, 1e-2, "ASM Handbook Vol. 2, 6061-T6"),
    ("6061-T6 conductivity S/m", "Materials!C43", lambda i: i.materials.aluminium.al6061.conductivity_S_m, 0.43 * IACS_S_M, 2e-2,
     "6061-T6: 43 % IACS"),
    ("class 12.9 proof MPa", "Materials!C48", lambda i: i.materials.screws.proof_12_9_MPa, 970, TOL_ALGEBRA, "ISO 898-1:2013 Table 3"),
    ("class 10.9 proof MPa", "Materials!C49", lambda i: i.materials.screws.proof_10_9_MPa, 830, TOL_ALGEBRA, "ISO 898-1:2013 Table 3"),
    ("A4-70 yield MPa", "Materials!C50", lambda i: i.materials.screws.yield_A4_70_MPa, 450, TOL_ALGEBRA, "ISO 3506-1:2020, Rp0.2 min"),
    ("6061 cap density g/cm3", "Metal design!C42", lambda i: i.metal.al_density_g_mm3 * 1000, 2.70, 1e-2, "Aluminum Association, 6061"),
    ("316L density g/cm3", "Metal design!C44", lambda i: i.metal.sleeve_density_g_mm3 * 1000, 8.0, 1e-2, "ASM Handbook Vol. 1, 316L: 7.99–8.0"),
]


@pytest.mark.family("materials")
@pytest.mark.parametrize("label,cell,getter,value,tol,source", LITERATURE, ids=[row[0] for row in LITERATURE])
def test_material_constants_vs_literature(label, cell, getter, value, tol, source):
    """Material inputs against handbook/standard values; tolerance is the spread or rounding of the
    published value (see ``source``)."""
    got = getter(defaults())
    assert rel_err(got, value) < tol, mismatch(f"{label} vs {source}", cell, got, value, tol)
```

- [ ] **Step 5: Run the checks**

Run: `cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_geometry_metal_reference_sanity.py audit/tests/test_geometry_mass.py audit/tests/test_metal_design.py audit/tests/test_materials.py`

Expected: `7 failed, 228 passed` in about 1 s. The command exits non-zero because of the 7 FAILs, and that is expected. `audit/out/results.json` lists every check.

The FAILs are **candidate findings for Task 8**, not bugs to fix now. Do not edit `magcoupling/`, and do not tune any tolerance. Each FAIL is one root cause.

**PASS (228):**
- 19 sanity tests.
- Corner gap C9 equals the numerically found minimum ring clearance in 5 layouts. At defaults, C9 = 1.0268256054339253 and the reference is 1.0268256054339244 (rel 8.6e-16).
- 16 block/polygon dimensions (C9, C38, C51, C53–C58, C60–C62, C64–C67) × 5 layouts, 80 cases: max rel 4.3e-16.
- C33 active length: 2 cases.
- Flat-fit verdicts C52/C59: 3 cases.
- Masses C110–C113 in 3 layouts: max rel 4.2e-16. Defaults: 38.3467 / 60.7537 / 25.8095 / 30.7776 g.
- C114 total mass: 173.79326219996213 against 173.79326219996216.
- C112 hub mass: aluminium without back iron (8.8772 g) and round in arc mode (24.6894 g).
- C115 added inertia: 0.010862 against m·d² + J_own = 0.010898 kg·m² (rel 3.3e-3, tol 1e-2; J_own = 3.58e-5 kg·m²).
- Torque band C8/C9/C10/C112/C157: 3.273666 / 2.250183 / 3.764716 N·m. C8 equals a model run at −40 °C.
- Requirement cells C19, C111, C20, C155, C156, C158, C151.
- Gearbox rows, one check per cell over both `GEAR_SCENARIOS` entries: at defaults C99 = 0.5573209 N·m = C93/(i·η) and C100 = 2.85 N·m = i·η·C47; at ratio 7, η 0.8 and free space (C93 = 1.697369 N·m) C99 = 0.3031017 N·m and C100 = 3.36 N·m.
- Clearance stack C27, C143, C34, C35, C12, C144 in 3 variants. Defaults: 0.676826, 0.78, −0.103174 and 0.096826 mm.
- Retainer geometry C175–C179 (sleeve ID 27.436349, liner OD 29.39) and masses C46/C180/C181 (3.131004, 2.321639, 6.653138 g).
- Axial envelope C134–C139 and C192 (31.8, 42.8, 0.2, 18.8, 1.2, 3.2, 33.3).
- Slip duty C86/C97/C89 (166.667 Hz, 3.333e8 cycles) and C91/C92 at None, 0.04 and 0.25 N·m.
- Adapter variant C188/C189/C191/C148/C149 at the workbook density.
- 15 linked cells.
- C103 at defaults (1.0186267 T).
- C104 and Materials C20 against the sinusoidal flux-conservation formula (1.9041528 mm).
- Verdict logic for Materials C22 (4 walls) and Calculator C105/C106 (3 variants).
- Pre-plate offsets C27–C30 at 0.015 and 0.025 mm.
- Links C36/C37 and the steel-density unit consistency.
- 16 material constants within their literature bands.

**FAIL (7), each a candidate finding for Task 8** (numbers as recorded in `audit/out/results.json`; tolerance `TOL_ALGEBRA = 1e-9` except items 3 and 4, which use `TOL_MODEL = 2e-2` from `audit.common` (Task 1), and item 2, which counts inconsistent outputs against an expected 0):
1. `test_geometry_mass.py::test_arc_mode_uses_arc_geometry` (family `geometry`; faceted = 0; one root cause, three symptoms):
   - (1) C57 effective gap: 1.026826 mm against the 1.4 mm input (rel 2.67e-1).
   - (2) C175 sleeve bore: 27.436349 mm against 26.69 mm over the arc OD (rel 2.80e-2).
   - (3) C111 cup mass: 42.9555 g with an N-gon cavity against 46.9958 g with the round pocket (rel 8.60e-2).
2. `test_materials.py::test_no_back_iron_mode_treats_the_cup_consistently` (family `materials`; backiron = 0): two outputs treat the cup as back iron, and the reference is zero. C111 prices the cup at 7.850 g/cm³ (steel), unchanged from backiron = 1, and Materials C22 still returns 'Too thin: raise Metal design C122 to at least 2.0 mm'. Meanwhile C93 = C96 (free-space S_n) and C105 = 'No back iron'. In the same run, pull-out is C96 = 1.6974 N·m (free space) and C95 = 2.6473 N·m (steel).
3. `test_materials.py::test_cup_backiron_requirement_vs_2d_slab` (family `materials`; C104): 1.904153 mm against 2.014512 mm, rel 5.48e-2 (TOL_MODEL 2e-2).
4. `test_materials.py::test_hub_backiron_requirement_vs_2d_slab` (family `materials`; C104 as used in C106): 1.904153 mm against 2.665781 mm, rel 2.86e-1 (TOL_MODEL 2e-2).
5. `test_geometry_mass.py::test_cup_wall_at_flats_is_steel_only` (family `geometry`; C63): 2.773232 mm against 2.723232 mm, rel 1.84e-2. The engine counts the 0.05 mm outer bondline as steel.
6. `test_materials.py::test_gap_flux_density_flat_circuit[N52 inner, 1.59 mm outer]` (family `materials`; C103): 1.02425 T against 1.042794 T, rel 1.78e-2.
7. `test_metal_design.py::test_adapter_mass_with_7075_density` (family `metal`; C188): 16.7271 g against 17.4086 g, rel 3.91e-2.

- [ ] **Step 6: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 add \
    reference/magcoupling-py/audit/references/block_geometry.py \
    reference/magcoupling-py/audit/references/slab_field2d.py \
    reference/magcoupling-py/audit/references/backiron_flux.py \
    reference/magcoupling-py/audit/references/remanence.py \
    reference/magcoupling-py/audit/references/metal_stack.py \
    reference/magcoupling-py/audit/tests/test_geometry_metal_reference_sanity.py \
    reference/magcoupling-py/audit/tests/test_geometry_mass.py \
    reference/magcoupling-py/audit/tests/test_metal_design.py \
    reference/magcoupling-py/audit/tests/test_materials.py
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 commit -F - <<'EOF'
test(magcoupling-audit): geometry, metal and materials independent checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 5: Temperature design part A — demagnetization onsets and magnet ratings, adhesive limits and bond loads, Volkersen screen

**Files:**
- Create: `reference/magcoupling-py/audit/references/demag_adhesive.py`
- Modify: `reference/magcoupling-py/audit/references/slab_field2d.py` (Task 4's shared two-plate slab module). Append the Maxwell-stress force and field-energy functions; nothing above them changes.
- Create: `reference/magcoupling-py/audit/tests/test_temperature_demag_adhesive_reference_sanity.py`
- Create: `reference/magcoupling-py/audit/tests/test_temperature_demag_adhesive.py`

**Interfaces:**
- Consumes:
  - `audit.common` (all from Task 1, which owns `audit/common.py` in full; this task never edits it): `defaults`, `run`, `vary`, `rel_err`, `mismatch`, `assert_all`, `text_item`, `MU0_EXACT`, `TOL_ALGEBRA`. The engine-check file defines no helpers of its own for multi-comparison checks: it imports `assert_all` and `text_item` from `audit.common`.
  - Task 4 references:
    - `audit.references.slab_field2d.{MagnetLayer, harmonic_coefficients}`. The appended code also uses that module's private helpers `_sheets`, `_square_wave` and `_ratio`, so if Task 4 renames them, this step fails loudly.
    - `audit.references.remanence.{br_ratio, torque_at}` and `audit.references.metal_stack.{pole_pair_frequency_Hz, omega_rad_s}`. These are the audit's single definitions of Br(T), the Br² torque law, pole-pair passes and rpm to rad/s (`demag_adhesive.centrifugal_force_N` takes its omega from `omega_rad_s`).
    - `audit.references.planar` (Task 2's module with Task 3's append), indirectly: `remanence.br_at` is a re-export of `planar.br_at`, and `slab_field2d._square_wave` evaluates `planar.square_wave_harmonic`, so `planar.py` must be in place before this task's modules import.
  - Engine result fields under test:
    - `temperature.demag.{br20_T, alpha_br, tmax_lib_C, h_ref_kA_m, t_ref_model_C, calibration_offset_C, onset_aligned_C, onset_pullout_C, onset_skipping_C, onset_single_ring_C, magnet_limit_C, torque_at_limit_Nm, torque_at_service_Nm}`
    - `temperature.summary.{onset_aligned_C, onset_pullout_C, onset_skipping_C, magnet_limit_C, adhesive_limit_C, governing_limit_C, governing_note, margin_service_C, margin_hot_day_C, torque_hot_day_note, cure_margin_C}`
    - `temperature.adhesive.{selected_name, design_limit_C, cure_C, lap_shear_MPa, bond_area_mm2, cold_high_torque_Nm, inner_mid_radius_mm, tangential_force_N, bond_shear_MPa, centrifugal_force_N, static_ratio, shear_reversals, fatigue_screen}`
    - `temperature.mismatch.{steel_cte, steel_E_GPa, steel_thickness_mm, cold_limit_C, worst_swing_C, current_bondline_mm, peak_shear_current_MPa, peak_shear_recommended_MPa, reading}`
    - `model.{inner_tmax_C, outer_tmax_C, inner_temp_check, outer_temp_check}`
    - the engine function `magcoupling.temperature.volkersen_peak_shear_MPa`, called only as the thing under test;
    - the library rows `magcoupling.library.MAGNET_LIBRARY`, as data under test.
  - Engine values owned by other tasks, consumed here as inputs:
    - `model.{inner_br_T (C21), pullout_Nm (C93), pullout_20C_Nm (C94), inner_length_mm, inner_width_mm, inner_thickness_mm, outer_thickness_mm, outer_br_T, hub_wall_mm, face_gap_mm, pole_pitch_mm, fill_inner, fill_outer, active_length_mm}`
    - `metal.torque_cold_high_Nm` (Metal design C10)
    - `temperature.adhesive.block_mass_g` (C82, owned by Task 7)
- Produces (reusable by other tasks):
  - `audit.references.demag_adhesive`:
    - Literature data: `ARNOLD_N42SH: dict`, `KJ_GRADE_MAX_OPERATING_C: dict[str, float]`, `KJ_BR_RANGE_T: dict[str, tuple[float, float]]`, `KJ_PART_GRADE: dict[str, str]`.
    - Adhesive data sheets: `AdhesiveTds(name, service_max_C, tg_C, tensile_modulus_GPa, source)` and `ADHESIVE_TDS: tuple[AdhesiveTds, ...]`.
    - Ratings:
      - `grade_max_operating_C(grade: str) -> float`
      - `part_rating_C(part: str) -> float | None`
    - Demagnetization:
      - `reverse_field_kA_m(t_C, h_rev20_kA_m, alpha_br_per_C) -> float`
      - `knee_field_kA_m(t_C, hcj20_kA_m, beta_hcj_per_C, knee_fraction) -> float`
      - `load_line_reverse_field_kA_m(br_T, permeance_coefficient, mu0, recoil_permeability=1.0) -> float`
      - `knee_crossing_C(h_rev20_kA_m, hcj20_kA_m, beta_hcj_per_C, knee_fraction, alpha_br_per_C, t_lo_C=-273.15, t_hi_C=1273.15) -> float`
      - `calibration_offset_C(h_ref_kA_m, rating_C, hcj20_kA_m, beta_hcj_per_C, knee_fraction, alpha_br_per_C) -> float`
    - Adhesive:
      - `adhesive_design_limit_C(service_max_C, tg_C, tg_margin_C=20.0) -> float`
      - `bond_shear_MPa(torque_Nm, n_blocks, lever_arm_mm, bond_area_mm2) -> float`
      - `centrifugal_force_N(mass_g, rpm, radius_mm) -> float`
      - `shear_modulus_GPa(tensile_modulus_GPa, poisson) -> float`
    - Volkersen:
      - `volkersen_lambda_per_m(G_Pa, eta_m, E1_Pa, t1_m, E2_Pa, t2_m) -> float`
      - `volkersen_thermal_peak_shear_Pa(G_Pa, eta_m, E1_Pa, t1_m, E2_Pa, t2_m, d_alpha_per_C, dT_C, overlap_m) -> float`
      - `volkersen_thermal_fd(G_Pa, eta_m, E1_Pa, t1_m, E2_Pa, t2_m, d_alpha_per_C, dT_C, overlap_m, n_nodes=20001) -> tuple[np.ndarray, np.ndarray, np.ndarray]`
  - `audit.references.slab_field2d` (appended to Task 4's module, which stays the audit's one two-plate slab model):
    - `strip_stress_N_per_m(y_mm, layers, h_mm, pitch_mm, n_max=2001) -> (∫T_yy dx, ∫T_xy dx)`
    - `layer_block_forces_N(layers, h_mm, pitch_mm, depth_mm, n_max=2001) -> list[(Fx, Fy)]`
    - `field_energy_J_per_m(layers, h_mm, pitch_mm, n_max=2001) -> float`
- Cell ownership (one owner per cell, so a single engine change yields a single candidate).
  - Task 5 owns:
    - Temperature design:
      - C7–C13, F12, C15, F16 and C24;
      - C42, C43, C47–C51 and C56–C62;
      - C75–C78, C81 and C83–C91;
      - C94–C106.
    - Calculator C22, C32, C107 and C108.
    - Magnet library ratings and Br.
  - Cells this task owns that other drafts also checked; the other checks are deleted, so each yields one candidate:
    - Calculator C22, C32, C107 and C108: Task 4's `test_materials.py::test_magnet_temperature_rating` and Task 3's `test_torque_laws.py::test_magnet_temperature_rating_check` no longer exist. Their extra cases moved into `test_magnet_temperature_checks_follow_kj_rating` here (the outer part `B842` at 100 °C, both parts blank, and the exact rating text `'n/a'` for a blank part).
    - Temperature design C13, C15 and F16: Task 6's `test_slip_thermal.py::test_row_rederivation` no longer covers them (its margin rows are removed and its hot-day screen keeps only C185). `test_summary_margins_and_hot_day_note` here is the only check, and it exercises both F16 branches.
  - Consumed but owned elsewhere:
    - C82 (Task 7);
    - C14, C16, C17–C23, C168 and C191 (Task 6);
    - Calculator C93/C94: Task 2 owns the same-algebra re-derivation of C66–C94, Task 3 owns the Br² temperature law on C93, and Task 4's cold-torque check only cross-checks the C94 scaling;
    - Metal design C10 and C89 (Task 4);
    - the 3D reverse-field inputs C52–C55 (M3).

- [ ] **Step 1: Write the reference modules**

File: `reference/magcoupling-py/audit/references/demag_adhesive.py` (create)

```python
"""Independent references for the Temperature design demagnetization and adhesive sections (M1 audit, Task 5).

Nothing here imports ``magcoupling``. Every function is written from first principles or from a
cited source. Br(T) comes from Task 4's ``remanence.br_ratio`` and rpm -> rad/s from Task 4's
``metal_stack.omega_rad_s`` (one definition each for the audit).

Demagnetization (knee crossing)
    Reverse fields scale with Br:       H(T)  = H20 * Br(T) / Br20
    (for a recoil permeability of ~1 every field in a fixed geometry, both the
    self-demagnetizing field and the field of the other ring, is proportional to Br).
    Knee of the intrinsic curve:        Hk(T) = knee * Hcj20 * (1 + beta * (T - 20))
    Onset: the temperature where |H(T)| = Hk(T). Here it is found by bracketing
    root-finding (Brent), which is a different method from the engine's closed form.

Load line (permeance coefficient)
    A linear magnet with recoil permeability mu_rec on load line B = -Pc * mu0 * H:
    Br + mu_rec * mu0 * H = -Pc * mu0 * H  =>  |H| = Br / (mu0 * (mu_rec + Pc)).
    Textbook form, e.g. Campbell, *Permanent Magnet Materials and their Application*
    (Cambridge, 1994), ch. 5; Furlani, *Permanent Magnet and Electromechanical Devices*
    (Academic Press, 2001), sec. 3.4.

Supplier ratings (K&J Magnetics, the workbook's magnet vendor)
    Maximum operating temperature by grade suffix and Br ranges by grade, from the K&J
    "Neodymium Magnet Specifications" page (kjmagnetics.com/specs.asp, read 2026-09-28).
    K&J states the rating depends on magnet shape (permeance coefficient) and is a guideline.

Adhesive
    Design limit: the lower of the TDS service maximum and Tg - 20 degC.
    Bond shear: torque / (blocks * lever arm) spread over the bonded back face.
    Centrifugal force: m * omega^2 * r.

Volkersen shear lag for thermal mismatch (derived here)
    Adherend 1 (magnet, E1, t1) and adherend 2 (steel, E2, t2) per unit width, joined by
    an adhesive layer of shear modulus G and thickness eta over an overlap of length L.
    Free ends, no external load, so the axial forces satisfy N2 = -N1.
        strains:      eps1 = N1/(E1 t1) + a1 dT,  eps2 = -N1/(E2 t2) + a2 dT
        adhesive:     tau = G (u2 - u1) / eta,    equilibrium: dN1/dx = -tau
        =>            N1'' - lam^2 N1 = -(G/eta) (a2 - a1) dT,  lam^2 = (G/eta)(1/(E1 t1) + 1/(E2 t2))
        with N1(+-L/2) = 0:
                      N1(x)  = (G da dT / (eta lam^2)) (1 - cosh(lam x)/cosh(lam L/2))
                      tau(x) = (G da dT / (eta lam)) sinh(lam x)/cosh(lam L/2)
        peak at the ends:  tau_max = G da dT tanh(lam L/2) / (eta lam)
    Sources: O. Volkersen, Luftfahrtforschung 15 (1938) 41-47 (shear lag);
    W.T. Chen and C.W. Nelson, "Thermal stress in bonded joints", IBM J. Res. Dev. 23 (1979)
    179-188 (the thermal-mismatch form); L.F.M. da Silva et al., Int. J. Adhes. Adhes. 29
    (2009) 319-330 (review).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from audit.references.metal_stack import omega_rad_s
from audit.references.remanence import br_ratio

# --------------------------------------------------------------------------- literature data
#: Arnold Magnetic Technologies, N42SH datasheet (Rev. 020821): reversible temperature
#: coefficients measured between 20 and 150 degC; CTE between 20 and 200 degC.
ARNOLD_N42SH = {
    "alpha_br_per_C": -0.0012,         # -0.12 %/degC
    "beta_hcj_per_C": -0.0055,         # -0.55 %/degC
    "hcj20_min_kA_m": 1592.0,          # 20,000 Oe minimum
    "br_nominal_T": 1.31,              # 13,100 G nominal
    "hcb_nominal_kA_m": 987.0,         # 12,400 Oe nominal
    "coefficient_range_C": (20.0, 150.0),
    "cte_perpendicular_per_C": -1.0e-6,
    "cte_parallel_per_C": 7.0e-6,
    "density_g_cm3": 7.6,
}

#: K&J maximum operating temperature by grade suffix ("" = plain N grade): N 176 F (80 C), M 212 F (100 C),
#: H 248 F (120 C), SH 302 F (150 C), UH 356 F (180 C), EH 392 F (200 C), AH 428 F (220 C).
KJ_GRADE_MAX_OPERATING_C = {"": 80.0, "M": 100.0, "H": 120.0, "SH": 150.0, "UH": 180.0, "EH": 200.0, "AH": 220.0}

#: K&J Br range by grade [T]: N42 13.0-13.2 kG, N42SH 13.0-13.3 kG, N52 14.5-14.8 kG.
KJ_BR_RANGE_T = {"N42": (1.30, 1.32), "N42SH": (1.30, 1.33), "N52": (1.45, 1.48)}

#: Grade of the K&J parts the checks use: the B842SH product page ("1/2 x 1/4 x 1/8 Inch ... N42SH", max operating
#: temperature 302 F (150 C), Br max 13,200 G) lists the same block in the alternative grades N42 (B842) and N52 (B842-N52).
KJ_PART_GRADE = {"B842SH": "N42SH", "B842": "N42", "B842-N52": "N52"}


@dataclass(frozen=True)
class AdhesiveTds:
    """Numbers read from a manufacturer technical data sheet (TDS)."""
    name: str
    service_max_C: float | None
    tg_C: float | None
    tensile_modulus_GPa: float
    source: str


#: TDS data for the candidates whose sheets were read for this audit.
ADHESIVE_TDS = (
    AdhesiveTds("Loctite AA 326 + SF 7649", 120.0, None, 0.300,
                "Henkel TDS LOCTITE AA 326 (Aug-2020): heat-ageing data at 100 and 120 degC (no higher), "
                "Tg not published, tensile modulus ISO 527-2 = 300 N/mm^2, elongation 135 %, "
                "lap shear 15.2 N/mm^2 with Activator 7649 on one side, recommended bondline 0.1 mm."),
    AdhesiveTds("Loctite EA 9514", 200.0, 133.0, 1.460,
                "Henkel TDS LOCTITE EA 9514 (Oct-2014): Tg 133 degC (ASTM E1640), heat-ageing data to "
                "200 degC, tensile modulus ISO 527-3 = 1,460 N/mm^2, lap shear 45 N/mm^2."),
)


# --------------------------------------------------------------------------- supplier ratings
def grade_max_operating_C(grade: str) -> float:
    """K&J maximum operating temperature for an NdFeB grade such as 'N42', 'N42SH' or 'N50M'."""
    m = re.fullmatch(r"N(\d+)([A-Z]*)", grade)
    if m is None or m.group(2) not in KJ_GRADE_MAX_OPERATING_C:
        raise ValueError(f"not an NdFeB grade with a K&J rating: {grade!r}")
    return KJ_GRADE_MAX_OPERATING_C[m.group(2)]


def part_rating_C(part: str) -> float | None:
    """K&J rating of a part the checks use; None for a blank part (the engine then runs uncalibrated).

    Raises KeyError for a part this module has no grade for, so a check can never silently fall back.
    """
    if not part:
        return None
    return grade_max_operating_C(KJ_PART_GRADE[part])


# --------------------------------------------------------------------------- demagnetization
def reverse_field_kA_m(t_C: float, h_rev20_kA_m: float, alpha_br_per_C: float) -> float:
    """Magnitude of a reverse field that scales with Br(T) (fixed geometry, recoil permeability ~1)."""
    return h_rev20_kA_m * br_ratio(alpha_br_per_C, t_C)


def knee_field_kA_m(t_C: float, hcj20_kA_m: float, beta_hcj_per_C: float, knee_fraction: float) -> float:
    """Knee of the intrinsic curve: knee * Hcj(T), Hcj linear in T with the signed coefficient beta."""
    return knee_fraction * hcj20_kA_m * (1.0 + beta_hcj_per_C * (t_C - 20.0))


def load_line_reverse_field_kA_m(br_T: float, permeance_coefficient: float, mu0: float,
                                 recoil_permeability: float = 1.0) -> float:
    """|H| at the intersection of the linear demagnetization line and the load line B = -Pc*mu0*H."""
    return br_T / (mu0 * (recoil_permeability + permeance_coefficient)) / 1000.0


def knee_crossing_C(h_rev20_kA_m: float, hcj20_kA_m: float, beta_hcj_per_C: float, knee_fraction: float,
                    alpha_br_per_C: float, t_lo_C: float = -273.15, t_hi_C: float = 1273.15) -> float:
    """Temperature where the Br-scaled reverse field equals the knee field, by Brent root-finding.

    Raises ValueError when the two lines do not cross inside [t_lo_C, t_hi_C].
    """
    def margin(t_C: float) -> float:
        return (knee_field_kA_m(t_C, hcj20_kA_m, beta_hcj_per_C, knee_fraction)
                - reverse_field_kA_m(t_C, h_rev20_kA_m, alpha_br_per_C))

    lo, hi = margin(t_lo_C), margin(t_hi_C)
    if lo * hi > 0:
        raise ValueError(f"no knee crossing between {t_lo_C} and {t_hi_C} degC (margins {lo:.3g}, {hi:.3g} kA/m)")
    return brentq(margin, t_lo_C, t_hi_C, xtol=1e-12, rtol=4 * np.finfo(float).eps, maxiter=500)


def calibration_offset_C(h_ref_kA_m: float, rating_C: float | None, hcj20_kA_m: float, beta_hcj_per_C: float,
                         knee_fraction: float, alpha_br_per_C: float) -> float:
    """Model onset of the reference magnet minus its supplier rating (0 when there is no rating)."""
    if rating_C is None:
        return 0.0
    return knee_crossing_C(h_ref_kA_m, hcj20_kA_m, beta_hcj_per_C, knee_fraction, alpha_br_per_C) - rating_C


# --------------------------------------------------------------------------- adhesive
def adhesive_design_limit_C(service_max_C: float | None, tg_C: float | None, tg_margin_C: float = 20.0) -> float:
    """Lower of the TDS service maximum and Tg - margin; a missing value simply does not constrain."""
    limits = [v for v in (service_max_C, None if tg_C is None else tg_C - tg_margin_C) if v is not None]
    if not limits:
        raise ValueError("need a service maximum or a Tg")
    return min(limits)


def bond_shear_MPa(torque_Nm: float, n_blocks: int, lever_arm_mm: float, bond_area_mm2: float) -> float:
    """Average bond shear: tangential force per block (torque / (blocks * lever arm)) over the bond area."""
    force_N = torque_Nm / (n_blocks * lever_arm_mm / 1000.0)
    return force_N / bond_area_mm2


def centrifugal_force_N(mass_g: float, rpm: float, radius_mm: float) -> float:
    """m * omega^2 * r, omega = omega_rad_s(rpm)."""
    omega = omega_rad_s(rpm)
    return mass_g / 1000.0 * omega ** 2 * radius_mm / 1000.0


def shear_modulus_GPa(tensile_modulus_GPa: float, poisson: float) -> float:
    """Isotropic G = E / (2 (1 + nu))."""
    return tensile_modulus_GPa / (2.0 * (1.0 + poisson))


# --------------------------------------------------------------------------- Volkersen thermal shear lag
def volkersen_lambda_per_m(G_Pa: float, eta_m: float, E1_Pa: float, t1_m: float, E2_Pa: float, t2_m: float) -> float:
    """Shear-lag parameter lam = sqrt((G/eta) (1/(E1 t1) + 1/(E2 t2)))."""
    return math.sqrt(G_Pa / eta_m * (1.0 / (E1_Pa * t1_m) + 1.0 / (E2_Pa * t2_m)))


def volkersen_thermal_peak_shear_Pa(G_Pa: float, eta_m: float, E1_Pa: float, t1_m: float, E2_Pa: float, t2_m: float,
                                    d_alpha_per_C: float, dT_C: float, overlap_m: float) -> float:
    """Peak (end) adhesive shear from the closed form derived in the module docstring."""
    lam = volkersen_lambda_per_m(G_Pa, eta_m, E1_Pa, t1_m, E2_Pa, t2_m)
    return G_Pa * d_alpha_per_C * dT_C * math.tanh(lam * overlap_m / 2.0) / (eta_m * lam)


def volkersen_thermal_fd(G_Pa: float, eta_m: float, E1_Pa: float, t1_m: float, E2_Pa: float, t2_m: float,
                         d_alpha_per_C: float, dT_C: float, overlap_m: float,
                         n_nodes: int = 20001) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference solution of N1'' - lam^2 N1 = -(G/eta) da dT with N1(+-L/2) = 0.

    Returns (x [m], N1 [N/m], tau [Pa]); tau = -dN1/dx by second-order differences
    (central inside, one-sided at the ends). Discretization error is O((lam dx)^2).
    """
    lam = volkersen_lambda_per_m(G_Pa, eta_m, E1_Pa, t1_m, E2_Pa, t2_m)
    q = G_Pa / eta_m * d_alpha_per_C * dT_C
    x = np.linspace(-overlap_m / 2.0, overlap_m / 2.0, n_nodes)
    dx = x[1] - x[0]
    m = n_nodes - 2                                     # interior unknowns
    bands = np.zeros((3, m))
    bands[0, 1:] = 1.0 / dx ** 2                        # super-diagonal
    bands[1, :] = -2.0 / dx ** 2 - lam ** 2             # diagonal
    bands[2, :-1] = 1.0 / dx ** 2                       # sub-diagonal
    n_interior = solve_banded((1, 1), bands, np.full(m, -q))
    n1 = np.concatenate(([0.0], n_interior, [0.0]))
    dn = np.empty_like(n1)
    dn[1:-1] = (n1[2:] - n1[:-2]) / (2.0 * dx)
    dn[0] = (-3.0 * n1[0] + 4.0 * n1[1] - n1[2]) / (2.0 * dx)
    dn[-1] = (3.0 * n1[-1] - 4.0 * n1[-2] + n1[-3]) / (2.0 * dx)
    return x, n1, -dn
```

Append the next block to the end of Task 4's `slab_field2d.py` (Task 4 creates that file; this step never creates it). The module already imports `math`, `dataclass`, `numpy` and `planar.square_wave_harmonic`, and it defines `MagnetLayer`, `_sheets`, `_square_wave` (which calls `square_wave_harmonic`; its name and signature are what the block relies on), `_ratio` and `harmonic_coefficients`, which the block uses. The block begins with one empty line, so that appended after the file's single separating blank line it leaves the new section two blank lines below the existing code, as in the integrated file.

File: `reference/magcoupling-py/audit/references/slab_field2d.py` (append)

```python

# =========================================================================== forces and energy (Task 5)
# Maxwell stress in air: T_yy = (B_y^2 - B_x^2)/(2 mu0), T_xy = B_x B_y / mu0. The field is antiperiodic over one
# pitch, so the stress is pitch-periodic: the sides of a pitch-wide strip cancel, and the force on one block of a
# layer is the difference of the strip integrals of T_yy (T_xy) on planes in the air just above and just below it.
# Odd harmonics are orthogonal over one pitch, so each strip integral is an exact sum over harmonics (no quadrature).
# The field energy (1/2) * integral of sigma * phi over the charge sheets gives the same forces by virtual work,
# which the Task 5 reference sanity tests use to check the stress sums.

_MU0 = 4e-7 * math.pi


def strip_stress_N_per_m(y_mm: float, layers: list[MagnetLayer], h_mm: float, pitch_mm: float,
                         n_max: int = 2001) -> tuple[float, float]:
    """(integral of T_yy dx, integral of T_xy dx) over one pole pitch at a height y in air [N per metre of depth]."""
    _, A, B, C, D = harmonic_coefficients(y_mm, layers, h_mm, pitch_mm, n_max)
    half_pitch_m = pitch_mm / 2.0 / 1000.0
    t_yy = half_pitch_m / (2.0 * _MU0) * float(np.sum(A ** 2 + B ** 2 - C ** 2 - D ** 2))
    t_xy = half_pitch_m / _MU0 * float(np.sum(A * D + B * C))
    return t_yy, t_xy


def layer_block_forces_N(layers: list[MagnetLayer], h_mm: float, pitch_mm: float, depth_mm: float,
                         n_max: int = 2001) -> list[tuple[float, float]]:
    """(F_x, F_y) on one block of each layer [N], for blocks ``depth_mm`` long (2D, no end effects).

    Layers must be listed bottom to top with air between them and between the outer layers and the plates
    (a bond gap): the cutting planes are the two plate faces and the mid-planes of the air between layers.
    """
    planes = [0.0]
    for lower, upper in zip(layers, layers[1:]):
        top = lower.y_bottom_mm + lower.thickness_mm
        if upper.y_bottom_mm <= top:
            raise ValueError("layers overlap or touch; list them bottom to top with air between")
        planes.append((top + upper.y_bottom_mm) / 2.0)
    planes.append(h_mm)
    stress = [strip_stress_N_per_m(y, layers, h_mm, pitch_mm, n_max) for y in planes]
    depth_m = depth_mm / 1000.0
    return [((stress[i + 1][1] - stress[i][1]) * depth_m, (stress[i + 1][0] - stress[i][0]) * depth_m)
            for i in range(len(layers))]


def field_energy_J_per_m(layers: list[MagnetLayer], h_mm: float, pitch_mm: float, n_max: int = 2001) -> float:
    """Magnetostatic energy (1/2) * integral of sigma * phi over the charge sheets, per pitch, per metre of depth.

    For rigid magnetization the force on a movable part is minus the derivative of this energy.
    """
    n = np.arange(1, n_max + 1, 2, dtype=float)
    k_per_m = n * math.pi / (pitch_mm / 1000.0)
    h_m = h_mm / 1000.0
    sheets = [(ys, _square_wave(br, n, fill), shift) for ys, br, fill, shift in _sheets(layers)]
    total = 0.0
    for yi, ai, si in sheets:
        for yj, aj, sj in sheets:
            lo, hi = min(yi, yj) / 1000.0, max(yi, yj) / 1000.0
            green = _ratio(k_per_m, lo, h_m - hi, h_m, False) / k_per_m        # [m]
            phase = np.cos(k_per_m * (si - sj) / 1000.0)
            total += float(np.sum(ai * aj * green * phase))
    return 0.5 / _MU0 * total * (pitch_mm / 2.0 / 1000.0)
```

- [ ] **Step 2: Write the references' own sanity tests**

File: `reference/magcoupling-py/audit/tests/test_temperature_demag_adhesive_reference_sanity.py` (create)

```python
"""Reference sanity for Task 5 (not engine checks): the demagnetization, rating, adhesive and Volkersen references in
``audit.references.demag_adhesive`` and the force and energy functions Task 5 appends to Task 4's
``audit.references.slab_field2d``, each against a known answer. Task 4's sanity file covers the slab field itself
(long-pitch circuit limit, mirror symmetry, iron-face limit). Nothing here calls the engine.
"""
from __future__ import annotations

import math

import pytest
from scipy.integrate import trapezoid

from audit.common import MU0_EXACT, TOL_ALGEBRA, mismatch, rel_err
from audit.references import demag_adhesive as ref
from audit.references.slab_field2d import MagnetLayer, field_energy_J_per_m, layer_block_forces_N, strip_stress_N_per_m


@pytest.mark.family("temperature")
def test_sanity_knee_crossing_hand_cases():
    """Sanity: knee-crossing root-find reproduces hand-solved linear intersections; parallel lines raise."""
    # alpha = 0: 1000 (1 - 0.005 dT) = 500  ->  dT = 100
    t = ref.knee_crossing_C(500.0, 1000.0, -0.005, 1.0, 0.0)
    assert rel_err(t, 120.0) < TOL_ALGEBRA, mismatch("hand case alpha=0", "reference", t, 120.0, TOL_ALGEBRA)
    # 500 (1 - 0.001 dT) = 1000 (1 - 0.005 dT)  ->  4.5 dT = 500
    t = ref.knee_crossing_C(500.0, 1000.0, -0.005, 1.0, -0.001)
    expect = 20.0 + 500.0 / 4.5
    assert rel_err(t, expect) < TOL_ALGEBRA, mismatch("hand case alpha=-0.001", "reference", t, expect, TOL_ALGEBRA)
    # zero reverse field: the knee itself reaches zero at 20 + 1/0.005
    t = ref.knee_crossing_C(0.0, 1592.0, -0.005, 0.9, -0.0012)
    assert rel_err(t, 220.0) < TOL_ALGEBRA, mismatch("hand case H=0", "reference", t, 220.0, TOL_ALGEBRA)
    with pytest.raises(ValueError):
        ref.knee_crossing_C(500.0, 1000.0, -0.0025, 1.0, -0.005)   # equal slopes (1000*0.0025 = 500*0.005): never cross


@pytest.mark.family("temperature")
def test_sanity_load_line_textbook():
    """Sanity: Pc = 1, mu_rec = 1 puts the operating point at H = -Br/(2 mu0), B = Br/2; Pc -> inf gives H -> 0."""
    h = ref.load_line_reverse_field_kA_m(1.2, 1.0, MU0_EXACT)
    expect = 1.2 / (2 * MU0_EXACT) / 1000                       # 477.4648 kA/m
    assert rel_err(h, expect) < TOL_ALGEBRA, mismatch("H at Pc=1", "reference", h, expect, TOL_ALGEBRA)
    b_on_load_line = 1.0 * MU0_EXACT * h * 1000                 # |B| = Pc mu0 |H|
    assert rel_err(b_on_load_line, 0.6) < TOL_ALGEBRA, mismatch("B at Pc=1", "reference", b_on_load_line, 0.6, TOL_ALGEBRA)
    h_big = ref.load_line_reverse_field_kA_m(1.2, 1e9, MU0_EXACT)
    assert h_big < 1e-6, mismatch("H as Pc -> inf", "reference", h_big, 0.0, 1e-6)


@pytest.mark.family("temperature")
def test_sanity_grade_rating_table():
    """Sanity: grade parsing gives the K&J ratings quoted on its pages (N42 and N52 80 C, N42SH 150 C, N35AH 220 C)."""
    for grade, rating in (("N42", 80.0), ("N52", 80.0), ("N42SH", 150.0), ("N50M", 100.0), ("N35AH", 220.0)):
        got = ref.grade_max_operating_C(grade)
        assert got == rating, mismatch(f"K&J rating of {grade}", "reference", got, rating, 0.0)
    assert ref.part_rating_C("B842SH") == 150.0, mismatch("B842SH rating", "reference", ref.part_rating_C("B842SH"), 150.0, 0.0)
    assert ref.part_rating_C("") is None, mismatch("blank part has no rating", "reference", 1.0, 0.0, 0.0)
    for bad in ("N42XX", "SmCo26", "42SH"):
        with pytest.raises(ValueError):
            ref.grade_max_operating_C(bad)
    with pytest.raises(KeyError):
        ref.part_rating_C("B999")


@pytest.mark.family("temperature")
def test_sanity_adhesive_limit_and_bond_formulas():
    """Sanity: min(service, Tg - 20) with missing data, unit bond shear, unit centrifugal force, G = E/(2(1+nu))."""
    assert ref.adhesive_design_limit_C(120.0, None) == 120.0, mismatch("service only", "reference", 120.0, 120.0, 0.0)
    assert ref.adhesive_design_limit_C(200.0, 133.0) == 113.0, mismatch("Tg governs", "reference", 113.0, 113.0, 0.0)
    assert ref.adhesive_design_limit_C(100.0, 133.0) == 100.0, mismatch("service governs", "reference", 100.0, 100.0, 0.0)
    with pytest.raises(ValueError):
        ref.adhesive_design_limit_C(None, None)
    tau = ref.bond_shear_MPa(1.0, 1, 1000.0, 1.0)               # 1 N over 1 mm^2
    assert rel_err(tau, 1.0) < TOL_ALGEBRA, mismatch("unit bond shear", "reference", tau, 1.0, TOL_ALGEBRA)
    f = ref.centrifugal_force_N(1000.0, 60.0 / (2 * math.pi), 1000.0)   # 1 kg, 1 rad/s, 1 m
    assert rel_err(f, 1.0) < TOL_ALGEBRA, mismatch("unit centrifugal", "reference", f, 1.0, TOL_ALGEBRA)
    g = ref.shear_modulus_GPa(2.6, 0.3)
    assert rel_err(g, 1.0) < TOL_ALGEBRA, mismatch("G from E, nu", "reference", g, 1.0, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_sanity_volkersen_limits_and_numerical_solution():
    """Sanity: Volkersen closed form hits its short- and long-overlap limits and matches the finite-difference BVP.

    Short overlap (lam L -> 0): rigid adherends, the adhesive takes the free mismatch u = da dT L/2 at each end,
    tau = G da dT L / (2 eta). Long overlap (lam L -> inf): tau = G da dT / (eta lam). Finite differences on
    20,001 nodes carry O((lam dx)^2) ~ 1e-8 error, so 1e-6 is a safe tolerance. The force balance
    integral(0..L/2) tau dx = N1(0) is also checked on the numerical profile.
    """
    base = dict(G_Pa=0.5e9, eta_m=1e-4, E1_Pa=160e9, t1_m=3e-3, E2_Pa=205e9, t2_m=5e-3, d_alpha_per_C=13e-6, dT_C=70.0)
    lam = ref.volkersen_lambda_per_m(base["G_Pa"], base["eta_m"], base["E1_Pa"], base["t1_m"], base["E2_Pa"], base["t2_m"])
    short = 1e-3 / lam
    got = ref.volkersen_thermal_peak_shear_Pa(**base, overlap_m=short)
    expect = base["G_Pa"] * base["d_alpha_per_C"] * base["dT_C"] * short / (2 * base["eta_m"])
    assert rel_err(got, expect) < 1e-6, mismatch("short-overlap limit", "reference", got, expect, 1e-6)
    long_ = 60.0 / lam
    got = ref.volkersen_thermal_peak_shear_Pa(**base, overlap_m=long_)
    expect = base["G_Pa"] * base["d_alpha_per_C"] * base["dT_C"] / (base["eta_m"] * lam)
    assert rel_err(got, expect) < 1e-12, mismatch("long-overlap limit", "reference", got, expect, 1e-12)
    L = 12.7e-3
    closed = ref.volkersen_thermal_peak_shear_Pa(**base, overlap_m=L)
    x, n1, tau = ref.volkersen_thermal_fd(**base, overlap_m=L)
    assert rel_err(tau[-1], closed) < 1e-6, mismatch("FD end shear", "reference", tau[-1], closed, 1e-6)
    assert rel_err(-tau[0], closed) < 1e-6, mismatch("FD antisymmetry", "reference", -tau[0], closed, 1e-6)
    mid = len(x) // 2
    balance = trapezoid(tau[mid:], x[mid:])
    assert rel_err(balance, n1[mid]) < 1e-6, mismatch("FD force balance", "reference", balance, n1[mid], 1e-6)


@pytest.mark.family("temperature")
def test_sanity_plate_pressure_is_b_squared_over_two_mu0():
    """Sanity: a fully filled layer between the plates with a very long period presses on each plate with the textbook
    magnetic pressure B^2/(2 mu0), B = Br t / h (magnetic circuit), and the strip stress is the same in the air gap.

    Each pitch holds one polarity transition, which lowers the strip-averaged pressure by a term linear in h/pitch
    (4.5e-4 at pitch 20 m); a two-pitch Richardson extrapolation (20 m and 40 m) removes it. The remainder is below
    1e-12 in practice, so the tolerance is 1e-8.
    """
    layer = [MagnetLayer(0.05, 3.17, 1.29, 1.0, 0.0)]
    h = 7.84
    p20 = strip_stress_N_per_m(h, layer, h, 20000.0, 200001)[0] / 20.0          # N/m per m of pitch = Pa
    p40 = strip_stress_N_per_m(h, layer, h, 40000.0, 200001)[0] / 40.0
    got = 2.0 * p40 - p20
    expect = (1.29 * 3.17 / h) ** 2 / (2.0 * MU0_EXACT)
    assert rel_err(got, expect) < 1e-8, mismatch("uniform-limit plate pressure", "reference", got, expect, 1e-8)
    in_gap = strip_stress_N_per_m(3.5, layer, h, 20000.0, 200001)[0] / 20.0
    assert rel_err(in_gap, p20) < 1e-12, mismatch("pressure at the plate vs in the gap", "reference", in_gap, p20, 1e-12)


@pytest.mark.family("temperature")
def test_sanity_slab_stress_is_divergence_free_and_matches_virtual_work():
    """Sanity: strip stress is the same at any height in air, and the plate force equals -dU/dh (virtual work).

    Maxwell stress is divergence-free in source-free air, so the strip integrals at two heights in the gap agree to
    round-off. The top-plate force from stress must equal minus the derivative of the field energy with respect to
    the plate position (central difference, step 1e-5 mm, truncation ~1e-10); tolerance 1e-8. Likewise the
    tangential force on the outer block must equal minus the derivative of the energy with respect to its shift.
    """
    layers = [MagnetLayer(0.05, 3.17, 1.29, 0.86, 0.0), MagnetLayer(4.62, 3.17, 1.29, 0.62, 2.0)]
    h, pitch = 7.84, 8.809
    a = strip_stress_N_per_m(3.5, layers, h, pitch)
    b = strip_stress_N_per_m(4.4, layers, h, pitch)
    for i, what in enumerate(("T_yy", "T_xy")):
        assert abs(a[i] - b[i]) < 1e-9 * max(abs(a[0]), 1.0), mismatch(f"{what} height invariance", "reference", a[i], b[i], 1e-9)
    dh = 1e-5
    f_energy = -(field_energy_J_per_m(layers, h + dh, pitch, 4001)
                 - field_energy_J_per_m(layers, h - dh, pitch, 4001)) / (2 * dh / 1000)
    f_stress = -strip_stress_N_per_m(h, layers, h, pitch, 4001)[0]
    assert rel_err(f_energy, f_stress) < 1e-8, mismatch("plate force: energy vs stress", "reference", f_energy, f_stress, 1e-8)
    forces = layer_block_forces_N(layers, h, pitch, 1000.0, 4001)
    ds = 1e-4

    def with_outer_shift(shift_mm: float) -> list[MagnetLayer]:
        return [layers[0], MagnetLayer(4.62, 3.17, 1.29, 0.62, shift_mm)]

    fx_energy = -(field_energy_J_per_m(with_outer_shift(2.0 + ds), h, pitch, 4001)
                  - field_energy_J_per_m(with_outer_shift(2.0 - ds), h, pitch, 4001)) / (2 * ds / 1000)
    assert rel_err(fx_energy, forces[1][0]) < 1e-8, mismatch("tangential force: energy vs stress", "reference",
                                                             fx_energy, forces[1][0], 1e-8)
    with pytest.raises(ValueError):
        layer_block_forces_N([MagnetLayer(0.0, 5.0, 1.29, 0.86), MagnetLayer(4.0, 3.0, 1.29, 0.62)], 8.0, pitch, 1.0)
```

- [ ] **Step 3: Run the sanity tests, and Task 4's slab tests against the extended module**

Run:
```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py
./.venv/Scripts/python -m pytest -q audit/tests/test_temperature_demag_adhesive_reference_sanity.py -k sanity
./.venv/Scripts/python -m pytest -q $(grep -l -e slab_field2d -e backiron_flux audit/tests/*.py | grep -v demag_adhesive)
```

Expected:
- First command: PASS (7 passed). The seven tests cover:
  - knee-crossing hand cases: 120 °C, 131.11 °C and 220 °C; parallel lines raise;
  - the Pc = 1 load line: 477.4648 kA/m and B = Br/2;
  - the K&J grade table: N42 and N52 at 80 °C, N42SH at 150, N50M at 100, N35AH at 220; bad grades and unknown parts raise;
  - unit cases for the adhesive limit, bond shear, centrifugal force and G = E/(2(1+ν));
  - Volkersen: short- and long-overlap limits, finite differences against the closed form (end shear 6.4e-9, force balance 1.5e-9);
  - plate pressure: the textbook B²/(2µ0) with B = Br·t/h, after a two-pitch Richardson extrapolation (4.4e-13), and the same pressure in the gap (2.7e-16);
  - Maxwell stress: invariant with height (2.8e-16) and equal to -dU/dh for the plate force (8.7e-11) and the tangential force (1.9e-10); overlapping layers raise.

  Task 4's sanity file already covers the slab field itself (long-pitch circuit limit, mirror symmetry, iron-face limit), so this file does not repeat it.
- Second command: the same pass/fail counts as before the edit, because the block only adds names. In the integrated suite the command selects `test_geometry_metal_reference_sanity.py` (19 passed) and `test_materials.py` (31 passed, 4 failed) and gives `4 failed, 50 passed` both before and after the append (the failures are Task 4's own candidates). `test_materials.py` no longer holds the magnet temperature rating check (deleted; this task owns C22/C32/C107/C108), which is why it has 35 tests.

- [ ] **Step 4: Write the engine checks**

File: `reference/magcoupling-py/audit/tests/test_temperature_demag_adhesive.py` (create)

```python
"""Temperature design, part A: demagnetization onsets and ratings, adhesive limits and bond loads, Volkersen screen.

Under test: ``magcoupling.temperature`` (``volkersen_peak_shear_MPa`` and the demag, adhesive, mismatch and
summary-margin parts of ``compute()``), the Calculator magnet temperature checks (C22, C32, C107, C108) and
the magnet library's ratings and Br. This file is the sole owner of Calculator C22, C32, C107, C108 and of the
Temperature design summary margins C13, C15 and the hot-day note F16. References:
``audit.references.demag_adhesive``, the slab forces Task 5 appends to Task 4's ``audit.references.slab_field2d``,
and Task 4's ``remanence`` and ``metal_stack``; none of them imports the engine. Engine values owned by other
tasks (model geometry, Calculator C21 Br20, C93/C94 pull-out, Metal design C10 cold-high torque, Temperature design
C82 block mass) are consumed as inputs and verified there. The reference self-tests live in
``test_temperature_demag_adhesive_reference_sanity.py``.

A failing check is a candidate finding for Task 8, not a bug to fix here.
"""
from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest
from scipy.optimize import brentq

from audit.common import MU0_EXACT, TOL_ALGEBRA, assert_all, defaults, mismatch, rel_err, run, text_item, vary
from audit.references import demag_adhesive as ref
from audit.references.metal_stack import pole_pair_frequency_Hz
from audit.references.remanence import br_ratio, torque_at
from audit.references.slab_field2d import MagnetLayer, layer_block_forces_N
from magcoupling.library import MAGNET_LIBRARY
from magcoupling.temperature import volkersen_peak_shear_MPa

#: Reverse-field case -> engine result field (DemagResults / SummaryResults).
ONSET_FIELDS = {"aligned": "onset_aligned_C", "pullout": "onset_pullout_C",
                "likepole": "onset_skipping_C", "single_ring": "onset_single_ring_C"}

#: Poisson's ratios used only for the biaxial-stiffness sensitivity (typical handbook values).
NU_NDFEB, NU_STEEL = 0.24, 0.29

HOT_DAY_NOTE_MEETS, HOT_DAY_NOTE_BELOW = "Meets it nominally (no variation allowance)", "Below it"
READING_ABOVE, READING_BELOW = "Above the lap-shear strength at the block ends", "Below the lap-shear strength"


# =========================================================================== helpers
def _cell(result_obj, name: str) -> str:
    """Workbook cell recorded in a result field's metadata."""
    return next(f.metadata["cell"] for f in fields(result_obj) if f.name == name)


def _cells(result_obj, *names: str) -> str:
    return ", ".join(_cell(result_obj, n) for n in names)


def _excel_text0(x: float) -> str:
    """Excel TEXT(x, "0") for x >= 0: round half away from zero."""
    return str(int(math.floor(x + 0.5)))


def _is_number(x) -> bool:
    return isinstance(x, (int, float))


def _demag_reference(inp, res) -> dict:
    """Reference demagnetization chain from engine inputs, Calculator C21 (Br20) and the K&J part rating."""
    d = inp.temperature.demag
    knee_args = (d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, inp.calibration.alpha_br_per_C)
    h_ref = ref.load_line_reverse_field_kA_m(res.model.inner_br_T, 1.0, inp.coupling.mu0)
    rating = ref.part_rating_C(inp.coupling.magnets.part_inner)
    offset = ref.calibration_offset_C(h_ref, rating, *knee_args)
    out = {"h_ref": h_ref, "t_ref": ref.knee_crossing_C(h_ref, *knee_args), "offset": offset, "rating": rating}
    for case in ONSET_FIELDS:
        out[case] = ref.knee_crossing_C(getattr(d, f"h_rev_{case}_kA_m"), *knee_args) - offset
    out["magnet_limit"] = out["likepole"] - d.design_margin_C
    return out


def _selected_candidate(inp):
    return inp.temperature.adhesive.candidates[inp.temperature.adhesive.selected - 1]


def _governing_reference(inp, res) -> float:
    return min(_demag_reference(inp, res)["magnet_limit"], _selected_candidate(inp).design_limit_C)


def _hot_day_start_C(inp) -> float:
    return inp.temperature.duty.hot_ambient_C + inp.temperature.duty.driving_rise_C


def _swing_reference(inp, res) -> float:
    cure = _selected_candidate(inp).cure_C
    return max(cure - inp.metal.min_temp_C, _governing_reference(inp, res) - cure)


def _volkersen_inputs_SI(inp, res, bondline_mm: float, biaxial: bool = False, ndfeb_cte: float | None = None) -> dict:
    """Keyword arguments for the reference Volkersen functions, in SI, from engine inputs."""
    mm = inp.temperature.mismatch
    e1, e2 = mm.ndfeb_modulus_GPa * 1e9, inp.materials.steel.modulus_GPa * 1e9
    if biaxial:
        e1, e2 = e1 / (1 - NU_NDFEB), e2 / (1 - NU_STEEL)
    cte = mm.ndfeb_cte_per_C if ndfeb_cte is None else ndfeb_cte
    return dict(G_Pa=mm.adhesive_shear_modulus_GPa * 1e9, eta_m=bondline_mm / 1000,
                E1_Pa=e1, t1_m=res.model.inner_thickness_mm / 1000, E2_Pa=e2, t2_m=res.model.hub_wall_mm / 1000,
                d_alpha_per_C=inp.materials.steel.cte_per_C - cte, dT_C=_swing_reference(inp, res),
                overlap_m=res.model.inner_length_mm / 1000)


def _reading_reference(peak_MPa: float, lap_MPa: float) -> str:
    return READING_ABOVE if peak_MPa > lap_MPa else READING_BELOW


def _inner_mid_radius_mm(inp, res) -> float:
    return inp.coupling.inner_back_apothem_mm + res.model.inner_thickness_mm / 2


def _bond_area_mm2(res) -> float:
    return res.model.inner_length_mm * res.model.inner_width_mm


def _bond_shear_reference(inp, res, lever_arm_mm: float | None = None) -> float:
    arm = _inner_mid_radius_mm(inp, res) if lever_arm_mm is None else lever_arm_mm
    return ref.bond_shear_MPa(res.metal.torque_cold_high_Nm, inp.coupling.npole, arm, _bond_area_mm2(res))


def _fatigue_screen_reference(fatigue_margin: float) -> str:
    return f"OK: {_excel_text0(fatigue_margin)}x margin" if fatigue_margin >= 4 else "CHECK"


# =========================================================================== demagnetization
@pytest.mark.family("temperature")
def test_h_ref_matches_permeance_coefficient_one_load_line():
    """C48 = |H| of a recoil-permeability-1 magnet on a Pc = 1 load line, Br20/(2 mu0) (workbook mu0 input)."""
    inp, res = defaults(), run()
    expect = _demag_reference(inp, res)["h_ref"]
    got = res.temperature.demag.h_ref_kA_m
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch("reference-magnet reverse field", _cell(res.temperature.demag, "h_ref_kA_m"),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_calibration_reference_onset_and_offset():
    """C49 (uncalibrated model onset of the Pc = 1 magnet) and C50 (offset = C49 - K&J 150 degC rating) by root-finding."""
    inp, res = defaults(), run()
    r = _demag_reference(inp, res)
    dm = res.temperature.demag
    assert_all([("reference-magnet model onset", _cell(dm, "t_ref_model_C"), dm.t_ref_model_C, r["t_ref"], TOL_ALGEBRA),
                ("calibration offset", _cell(dm, "calibration_offset_C"), dm.calibration_offset_C, r["offset"], TOL_ALGEBRA)])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("case", list(ONSET_FIELDS))
def test_demag_onset_matches_root_find(case):
    """C56-C59: calibrated onset = knee crossing of the Br-scaled 3D reverse field minus the calibration offset."""
    inp, res = defaults(), run()
    expect = _demag_reference(inp, res)[case]
    got = getattr(res.temperature.demag, ONSET_FIELDS[case])
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch(f"demag onset ({case})", _cell(res.temperature.demag, ONSET_FIELDS[case]),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_magnet_limit():
    """C60 = skipping onset - 10 degC design margin (C51) at defaults."""
    inp, res = defaults(), run()
    expect = _demag_reference(inp, res)["magnet_limit"]
    got = res.temperature.demag.magnet_limit_C
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch("magnet design limit", _cell(res.temperature.demag, "magnet_limit_C"),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [
    {"temperature.demag.hcj20_kA_m": 1400.0},
    {"temperature.demag.beta_hcj_per_C": -0.006},
    {"temperature.demag.knee_fraction": 0.95},
    {"calibration.alpha_br_per_C": -0.001},
    {"temperature.demag.h_rev_likepole_kA_m": 700.0, "temperature.demag.design_margin_C": 15.0},
    {"temperature.demag.h_rev_likepole_kA_m": 1500.0},     # reverse field above the 20 degC knee: onset below 20 degC
    {"coupling.mu0": MU0_EXACT},
    {"coupling.magnets.part_inner": "B842-N52"},            # Br 1.45 T, K&J N52 rating 80 degC
    {"coupling.magnets.part_inner": ""},                    # not in the library: uncalibrated (offset 0)
], ids=["hcj20", "beta", "knee", "alpha_br", "likepole_margin", "h_above_knee", "mu0_exact", "n52_part", "no_library"])
def test_demag_chain_tracks_root_find_under_varied_inputs(changes):
    """C48-C60 follow the independent chain when inputs change (including the uncalibrated not-in-library branch)."""
    inp = vary(defaults(), changes)
    res = run(inp)
    r = _demag_reference(inp, res)
    dm = res.temperature.demag
    pairs = [("h_ref_kA_m", "h_ref"), ("calibration_offset_C", "offset"), ("magnet_limit_C", "magnet_limit")]
    pairs += [(ONSET_FIELDS[c], c) for c in ONSET_FIELDS]
    assert_all([(f"{field_name} with {changes}", _cell(dm, field_name), getattr(dm, field_name), r[key], TOL_ALGEBRA)
                for field_name, key in pairs])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [{}, {"coupling.magnets.part_inner": "B842-N52"}, {"coupling.magnets.part_inner": ""}],
                         ids=["defaults", "n52_part", "no_library"])
def test_demag_inputs_link_to_calculator_and_kj_rating(changes):
    """C42 = Calculator C21 (Br20 used), C43 = Calibration C22 (alpha_Br), C47 = K&J rating of the inner part
    (= Calculator C22); a part outside the library has no numeric rating."""
    inp = vary(defaults(), changes)
    res = run(inp)
    dm = res.temperature.demag
    rating = ref.part_rating_C(inp.coupling.magnets.part_inner)
    items = [("Br20 used by demag vs Calculator C21", _cell(dm, "br20_T") + ", Calculator!C21", dm.br20_T, res.model.inner_br_T, 0.0),
             ("alpha_Br vs Calibration C22 input", _cell(dm, "alpha_br") + ", Calibration!C22", dm.alpha_br,
              inp.calibration.alpha_br_per_C, 0.0)]
    if rating is None:
        items.append((f"library rating must be non-numeric for a blank part (C47 {dm.tmax_lib_C!r})", _cell(dm, "tmax_lib_C"),
                      0.0 if _is_number(dm.tmax_lib_C) else 1.0, 1.0, 0.0))
    else:
        items.append(("library rating vs K&J grade rating", _cell(dm, "tmax_lib_C"),
                      dm.tmax_lib_C if _is_number(dm.tmax_lib_C) else math.nan, rating, 0.0))
    assert_all(items)


@pytest.mark.family("temperature")
def test_doc_consistency_readme_demag_numbers():
    """Documentation consistency, not an independent check: README 'Temperature design' figures (10.4 degC shift,
    102.6 degC skipping onset, 92.6 / 92.55 degC limit) and 'skipping is the lowest onset' match the engine."""
    dm = run().temperature.demag
    items = [(f"README {what}", _cell(dm, field_name), getattr(dm, field_name), printed, half_digit / printed)
             for what, field_name, printed, half_digit in [("calibration shift", "calibration_offset_C", 10.4, 0.05),
                                                           ("skipping onset", "onset_skipping_C", 102.6, 0.05),
                                                           ("magnet limit", "magnet_limit_C", 92.6, 0.05),
                                                           ("magnet limit (quick start)", "magnet_limit_C", 92.55, 0.005)]]
    lowest = min(getattr(dm, f) for f in ONSET_FIELDS.values())
    items.append(("README: skipping is the lowest onset", _cell(dm, "onset_skipping_C"), dm.onset_skipping_C, lowest, 0.0))
    assert_all(items)


@pytest.mark.family("temperature")
def test_torque_at_magnet_limit_scales_with_br_squared():
    """C61 = pull-out(20 degC, Calculator C94) * (Br(T_limit)/Br20)^2; C62 repeats the service pull-out (Calculator C93).
    The C93/C94 Br^2 law itself is owned by the torque tasks."""
    inp, res = defaults(), run()
    lim = _demag_reference(inp, res)["magnet_limit"]
    dm = res.temperature.demag
    assert_all([("pull-out at magnet limit", _cell(dm, "torque_at_limit_Nm"), dm.torque_at_limit_Nm,
                 torque_at(res.model.pullout_20C_Nm, inp.calibration.alpha_br_per_C, 20.0, lim), TOL_ALGEBRA),
                ("pull-out at service maximum vs Calculator C93", _cell(dm, "torque_at_service_Nm") + ", Calculator!C93",
                 dm.torque_at_service_Nm, res.model.pullout_Nm, 0.0)])


@pytest.mark.family("temperature")
def test_summary_repeats_demag_and_adhesive_values():
    """C7-C11 and C24 (cure margin = single-ring onset - cure temperature) equal the independent values."""
    inp, res = defaults(), run()
    r = _demag_reference(inp, res)
    s = res.temperature.summary
    expected = {"onset_aligned_C": r["aligned"], "onset_pullout_C": r["pullout"], "onset_skipping_C": r["likepole"],
                "magnet_limit_C": r["magnet_limit"], "adhesive_limit_C": _selected_candidate(inp).design_limit_C,
                "cure_margin_C": r["single_ring"] - _selected_candidate(inp).cure_C}
    assert_all([(f"summary {name}", _cell(s, name), getattr(s, name), expect, TOL_ALGEBRA) for name, expect in expected.items()])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [{}, {"coupling.op_temp_C": 80.0}, {"metal.required_min_Nm": 2.7},
                                     {"temperature.adhesive.selected": 4}, {"temperature.duty.hot_ambient_C": 30.0}],
                         ids=["defaults", "op_80C", "requirement_2p7", "dp460_governs", "mild_day"])
def test_summary_margins_and_hot_day_note(changes):
    """C13 = governing limit - service maximum (Calculator C10); C15 = governing limit - hot-day start (C35 + C36);
    F16 = 'Meets it nominally...' when pull-out at the hot-day start (Br^2 from Calculator C94) >= required (Metal C7).
    Sole owner of C13, C15 and F16; both branches of F16 are exercised (defaults meets, requirement_2p7 is below)."""
    inp = vary(defaults(), changes)
    res = run(inp)
    s = res.temperature.summary
    gov, t0 = _governing_reference(inp, res), _hot_day_start_C(inp)
    torque_hot = torque_at(res.model.pullout_20C_Nm, inp.calibration.alpha_br_per_C, 20.0, t0)
    note = HOT_DAY_NOTE_MEETS if torque_hot >= inp.metal.required_min_Nm else HOT_DAY_NOTE_BELOW
    assert_all([(f"margin above service maximum with {changes}", _cell(s, "margin_service_C"), s.margin_service_C,
                 gov - inp.coupling.op_temp_C, TOL_ALGEBRA),
                (f"margin above hot-day start with {changes}", _cell(s, "margin_hot_day_C"), s.margin_hot_day_C, gov - t0, TOL_ALGEBRA),
                text_item(f"hot-day torque note (reference {torque_hot:.6f} vs required {inp.metal.required_min_Nm} N.m)",
                          _cell(s, "torque_hot_day_note"), s.torque_hot_day_note, note)])


def _alternative_skipping_onset(form: str, inp, res) -> float:
    """Skipping onset under another reasonable way of calibrating the knee model to the supplier rating."""
    d = inp.temperature.demag
    alpha, br20, mu0 = inp.calibration.alpha_br_per_C, res.model.inner_br_T, inp.coupling.mu0
    rating = ref.part_rating_C(inp.coupling.magnets.part_inner)
    n42sh = ref.ARNOLD_N42SH
    mu_rec = n42sh["br_nominal_T"] / (MU0_EXACT * n42sh["hcb_nominal_kA_m"] * 1000)       # 1.056 (Arnold N42SH)
    h_ref = ref.load_line_reverse_field_kA_m(br20, 1.0, mu0, mu_rec if "recoil_permeability" in form else 1.0)
    h_lp = d.h_rev_likepole_kA_m
    if form.startswith("knee_fraction"):
        knee = brentq(lambda k: ref.knee_crossing_C(h_ref, d.hcj20_kA_m, d.beta_hcj_per_C, k, alpha) - rating, 0.3, 1.2)
        return ref.knee_crossing_C(h_lp, d.hcj20_kA_m, d.beta_hcj_per_C, knee, alpha)
    if form == "beta_hcj":
        beta = brentq(lambda b: ref.knee_crossing_C(h_ref, d.hcj20_kA_m, b, d.knee_fraction, alpha) - rating, -0.02, -0.001)
        return ref.knee_crossing_C(h_lp, d.hcj20_kA_m, beta, d.knee_fraction, alpha)
    args = (d.hcj20_kA_m, d.beta_hcj_per_C, d.knee_fraction, alpha)
    return ref.knee_crossing_C(h_lp, *args) - ref.calibration_offset_C(h_ref, rating, *args)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("form", ["knee_fraction", "beta_hcj", "recoil_permeability", "knee_fraction+recoil_permeability"])
def test_onset_calibration_form_sensitivity_within_design_margin(form):
    """Model form: other reasonable ways to calibrate to the 150 degC rating move the skipping onset by < the 10 degC margin.

    The engine shifts every onset by a constant temperature. Alternatives: scale the knee fraction, or scale the Hcj
    coefficient, so the Pc = 1 magnet reaches its knee at exactly 150 degC; keep the offset but place the Pc = 1
    operating point with the datasheet recoil permeability mu_rec = Br/(mu0 HcB) = 1.056 (Arnold N42SH); or both the
    knee-fraction calibration and the datasheet recoil permeability. Tolerance: the design margin (C51), whose job is to
    cover model-form uncertainty.
    """
    inp, res = defaults(), run()
    alt = _alternative_skipping_onset(form, inp, res)
    got = res.temperature.demag.onset_skipping_C
    margin = inp.temperature.demag.design_margin_C
    assert abs(got - alt) <= margin, mismatch(f"skipping onset vs {form} calibration (tol = {margin} degC margin)",
                                              _cells(res.temperature.demag, "onset_skipping_C", "calibration_offset_C")
                                              + ", Temperature design!C51", got, alt, margin / abs(alt))


@pytest.mark.family("temperature")
def test_knee_model_coefficients_vs_n42sh_datasheet():
    """Literature: alpha_Br and Hcj20 match the Arnold N42SH datasheet; with its beta_Hcj = -0.55 %/degC (engine -0.50)
    the calibrated skipping onset is not lower than the engine's (engine conservative) and within the 10 degC margin."""
    inp, res = defaults(), run()
    d, n42sh = inp.temperature.demag, ref.ARNOLD_N42SH
    dm = res.temperature.demag
    lit = _demag_reference(vary(inp, {"temperature.demag.beta_hcj_per_C": n42sh["beta_hcj_per_C"]}), res)["likepole"]
    got = dm.onset_skipping_C
    assert_all([("alpha_Br vs datasheet -0.12 %/degC", _cell(dm, "alpha_br"), inp.calibration.alpha_br_per_C,
                 n42sh["alpha_br_per_C"], TOL_ALGEBRA),
                ("Hcj20 vs datasheet minimum", "Temperature design!C44", d.hcj20_kA_m, n42sh["hcj20_min_kA_m"], TOL_ALGEBRA),
                ("skipping onset vs datasheet beta_Hcj (engine must be <= datasheet onset, within margin)",
                 _cell(dm, "onset_skipping_C") + ", Temperature design!C45", got,
                 min(max(got, lit - d.design_margin_C), lit), 0.0)])


@pytest.mark.family("temperature")
def test_knee_evaluated_inside_coefficient_validity_range():
    """Literature: the linear Hcj(T) and Br(T) coefficients are measured over 20-150 degC (Arnold N42SH note 1).

    The knee is evaluated at the uncalibrated model temperature (C49 for the Pc = 1 magnet, reported onset + C50
    offset for the four cases). One check for the one root cause; the message lists every case outside the range.
    """
    inp, res = defaults(), run()
    r = _demag_reference(inp, res)
    lo, hi = ref.ARNOLD_N42SH["coefficient_range_C"]
    dm = res.temperature.demag
    cases = [("reference magnet (Pc = 1)", r["t_ref"], _cell(dm, "t_ref_model_C"))]
    cases += [(case, r[case] + r["offset"], _cells(dm, ONSET_FIELDS[case], "calibration_offset_C")) for case in ONSET_FIELDS]
    assert_all([(f"knee evaluation temperature ({case}) outside {lo:.0f}-{hi:.0f} degC", cells, t_model,
                 min(max(t_model, lo), hi), 0.0) for case, t_model, cells in cases])


# =========================================================================== magnet ratings and library
@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [
    {},
    {"coupling.op_temp_C": 150.0},                                          # exactly at the rating: OK (<=)
    {"coupling.op_temp_C": math.nextafter(150.0, math.inf)},                # one ulp above: over
    {"coupling.magnets.part_inner": "B842-N52", "coupling.op_temp_C": 90.0},
    {"coupling.magnets.part_outer": "B842", "coupling.op_temp_C": 100.0},  # N42 (80 degC) outer: catches swapped rings
    {"coupling.magnets.part_inner": ""},
    {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""},
], ids=["defaults", "at_rating", "above_rating", "n52_inner_90C", "n42_outer_100C", "no_library_inner",
        "no_library_both"])
def test_magnet_temperature_checks_follow_kj_rating(changes):
    """Calculator C22/C32 carry the K&J rating of each ring's part ('n/a' for a blank part); C107/C108 read 'OK' when
    the operating temperature (C10) is at or below it, 'OVER the magnet rating' above it, and 'unknown' for a part
    outside the library. Sole owner of C22, C32, C107 and C108."""
    inp = vary(defaults(), changes)
    m = run(inp).model
    items = []
    for ring, part, tmax, check, c_rating, c_check in (
            ("inner", inp.coupling.magnets.part_inner, m.inner_tmax_C, m.inner_temp_check, "Calculator!C22", "Calculator!C107"),
            ("outer", inp.coupling.magnets.part_outer, m.outer_tmax_C, m.outer_temp_check, "Calculator!C32", "Calculator!C108")):
        rating = ref.part_rating_C(part)
        if rating is None:
            items.append(text_item(f"{ring} rating of a blank part (no library row)", c_rating, tmax, "n/a"))
            expected = "unknown"
        else:
            items.append((f"{ring} rating of {part} vs K&J", c_rating, tmax if _is_number(tmax) else math.nan, rating, 0.0))
            expected = "OK" if inp.coupling.op_temp_C <= rating else "OVER the magnet rating"
        items.append(text_item(f"{ring} temperature check at {inp.coupling.op_temp_C!r} degC", c_check, check, expected))
    assert_all(items)


@pytest.mark.family("temperature")
def test_library_ratings_follow_kj_grade_table():
    """Every magnet-library row's max operating temperature equals the K&J rating of its grade suffix."""
    assert_all([(f"library {p.part} ({p.grade}) max operating temperature", "Magnet library", p.tmax_C,
                 ref.grade_max_operating_C(p.grade), 0.0) for p in MAGNET_LIBRARY.values()])


@pytest.mark.family("materials")
def test_library_br_within_kj_grade_range():
    """Literature: library Br20 (Calculator C21/C31 source) lies inside K&J's published Br range for its grade.

    Rows whose grade K&J's specification page does not list (N50, N50M) are not checked.
    """
    items = []
    for p in MAGNET_LIBRARY.values():
        if p.grade in ref.KJ_BR_RANGE_T:
            lo, hi = ref.KJ_BR_RANGE_T[p.grade]
            items.append((f"library {p.part} ({p.grade}) Br20 vs K&J {lo}-{hi} T", "Magnet library / Calculator!C21",
                          p.br_T, min(max(p.br_T, lo), hi), 0.0))
    assert_all(items)


# =========================================================================== adhesive
@pytest.mark.family("temperature")
@pytest.mark.parametrize("selected", [1, 2, 3, 4])
def test_adhesive_selection_and_governing_limit(selected):
    """C75-C78 copy the selected candidate row; C12 = min(magnet limit, adhesive limit); F12 names the governing limit."""
    inp = vary(defaults(), {"temperature.adhesive.selected": selected})
    res = run(inp)
    cand = _selected_candidate(inp)
    a, s = res.temperature.adhesive, res.temperature.summary
    mag_lim = _demag_reference(inp, res)["magnet_limit"]
    note = "Magnets govern (skipping case)." if mag_lim <= cand.design_limit_C else "Adhesive governs."
    items = [(f"selected adhesive {name} ({cand.name})", _cell(a, name), getattr(a, name), expect, 0.0)
             for name, expect in (("design_limit_C", cand.design_limit_C), ("cure_C", cand.cure_C), ("lap_shear_MPa", cand.lap_shear_MPa))]
    items += [text_item("selected adhesive name", "Temperature design!C75", a.selected_name, cand.name),
              (f"governing limit ({cand.name})", _cell(s, "governing_limit_C"), s.governing_limit_C,
               min(mag_lim, cand.design_limit_C), TOL_ALGEBRA),
              text_item(f"governing note (magnet {mag_lim:.4f} vs adhesive {cand.design_limit_C} degC)",
                        _cell(s, "governing_note"), s.governing_note, note)]
    assert_all(items)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("tds", ref.ADHESIVE_TDS, ids=[t.name for t in ref.ADHESIVE_TDS])
def test_adhesive_design_limit_follows_tds(tds):
    """Literature: candidate design limit (C65:C74 data) = min(TDS service maximum, Tg - 20 degC) for the TDSs read."""
    cand = next(c for c in defaults().temperature.adhesive.candidates if c.name == tds.name)
    expect = ref.adhesive_design_limit_C(tds.service_max_C, tds.tg_C)
    assert cand.design_limit_C == expect, mismatch(f"design limit of {tds.name} ({tds.source})", "Temperature design!C65:C74",
                                                   cand.design_limit_C, expect, 0.0)


@pytest.mark.family("temperature")
def test_bond_shear_from_cold_high_torque():
    """C81 area = L*W, C83 = Metal C10, C84 r_mid = back apothem + t/2, C85 F = T_cold_high/(N r_mid), C86 tau = F/area."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    torque = res.metal.torque_cold_high_Nm
    r_mid = _inner_mid_radius_mm(inp, res)
    expected = {"cold_high_torque_Nm": torque, "bond_area_mm2": _bond_area_mm2(res), "inner_mid_radius_mm": r_mid,
                "tangential_force_N": torque / (inp.coupling.npole * r_mid / 1000), "bond_shear_MPa": _bond_shear_reference(inp, res)}
    assert_all([(name, _cell(a, name), getattr(a, name), expect, TOL_ALGEBRA) for name, expect in expected.items()])


@pytest.mark.family("temperature")
def test_centrifugal_force_from_block_mass():
    """C88 = m omega^2 r_mid at the wheel-rotor speed (C34). The block mass C82 is owned by Task 7's density checks
    and consumed here as an input."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    force = ref.centrifugal_force_N(a.block_mass_g, inp.temperature.duty.wheel_rotor_rpm, _inner_mid_radius_mm(inp, res))
    assert rel_err(a.centrifugal_force_N, force) < TOL_ALGEBRA, mismatch(
        "centrifugal force per block", _cell(a, "centrifugal_force_N"), a.centrifugal_force_N, force, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_static_ratio_and_fatigue_screen_at_defaults():
    """C89 = lap shear / bond shear; C91 = 'OK: Nx margin' when fatigue endurance * lap shear / bond shear >= 4."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    tau = _bond_shear_reference(inp, res)
    lap = _selected_candidate(inp).lap_shear_MPa
    fat = inp.temperature.adhesive_life.fatigue_endurance * lap / tau
    assert_all([("static strength ratio", _cell(a, "static_ratio"), a.static_ratio, lap / tau, TOL_ALGEBRA),
                text_item(f"fatigue screen (reference margin {fat:.4f})", _cell(a, "fatigue_screen"), a.fatigue_screen,
                          _fatigue_screen_reference(fat))])


@pytest.mark.family("temperature")
def test_fatigue_screen_uses_fatigue_endurance_input():
    """C91 should use the fatigue-endurance input (C195) that C196 uses; vary it to 0.1 and compare the screen."""
    inp = vary(defaults(), {"temperature.adhesive_life.fatigue_endurance": 0.1})
    res = run(inp)
    a = res.temperature.adhesive
    fat_ref = inp.temperature.adhesive_life.fatigue_endurance * _selected_candidate(inp).lap_shear_MPa / _bond_shear_reference(inp, res)
    assert_all([text_item(f"fatigue screen with endurance 0.1 (reference margin {fat_ref:.4f})",
                          _cell(a, "fatigue_screen") + ", Temperature design!C195", a.fatigue_screen,
                          _fatigue_screen_reference(fat_ref))])


@pytest.mark.family("temperature")
def test_shear_reversals_equal_pole_pair_passes():
    """C90 = pole-pair passes over life while slipping: (N/2) * slip rpm / 60 * event duration * events (one full shear
    reversal per pole-pair pass). Task 5 owns C90; Task 6 owns the separately computed C168 and C191."""
    inp, res = defaults(), run()
    md = inp.metal
    expect = pole_pair_frequency_Hz(inp.coupling.npole, md.slip_rpm) * md.slip_event_s * md.life_events
    got = res.temperature.adhesive.shear_reversals
    assert rel_err(got, expect) < TOL_ALGEBRA, mismatch("shear reversals", _cell(res.temperature.adhesive, "shear_reversals"),
                                                        got, expect, TOL_ALGEBRA)


@pytest.mark.family("temperature")
def test_fatigue_screen_robust_to_bond_plane_lever_arm():
    """Model form: taking the lever arm at the bond plane (back apothem) instead of the block mid radius raises bond
    shear 16 %; the fatigue screen verdict (C91, OK or CHECK) must not change."""
    inp, res = defaults(), run()
    a = res.temperature.adhesive
    tau_back = _bond_shear_reference(inp, res, lever_arm_mm=inp.coupling.inner_back_apothem_mm)
    fat_back = inp.temperature.adhesive_life.fatigue_endurance * _selected_candidate(inp).lap_shear_MPa / tau_back
    verdict = "OK" if fat_back >= 4 else "CHECK"
    assert_all([text_item(f"fatigue verdict with the bond-plane lever arm (tau {tau_back:.4f} MPa, margin {fat_back:.3f})",
                          _cells(a, "fatigue_screen", "inner_mid_radius_mm"), a.fatigue_screen.split(":")[0], verdict)])


def _press_on_forces(inp, res):
    """(inner F_y list, outer F_y list) over one pole pair at Br(governing limit), 2D multi-layer slab model."""
    m = res.model
    br = m.inner_br_T * br_ratio(inp.calibration.alpha_br_per_C, _governing_reference(inp, res))
    y_outer = inp.metal.bond_inner_mm + m.inner_thickness_mm + m.face_gap_mm
    h = y_outer + m.outer_thickness_mm + inp.metal.bond_outer_mm
    inner_fy, outer_fy = [], []
    for shift in np.linspace(0.0, 2 * m.pole_pitch_mm, 33):
        layers = [MagnetLayer(inp.metal.bond_inner_mm, m.inner_thickness_mm, br, m.fill_inner, 0.0),
                  MagnetLayer(y_outer, m.outer_thickness_mm, br * m.outer_br_T / m.inner_br_T, m.fill_outer, shift)]
        (_, fy_in), (_, fy_out) = layer_block_forces_N(layers, h, m.pole_pitch_mm, m.active_length_mm)
        inner_fy.append(fy_in)
        outer_fy.append(fy_out)
    return inner_fy, outer_fy


@pytest.mark.family("temperature")
def test_magnetics_press_inner_blocks_onto_hub():
    """README claim 'the magnetics press the blocks onto the steel', inner ring: the net magnetic radial force on an
    inner block points into the hub at every relative angle over a pole pair and exceeds the centrifugal force at
    wheel speed (C88, verified above). 2D two-plate model at Br(governing limit), the weakest case. Sign check: the
    planar model omits curvature, end leakage and finite steel permeability, so only a margin of the whole
    centrifugal force counts."""
    inp, res = defaults(), run()
    inner_fy, _ = _press_on_forces(inp, res)
    fc = res.temperature.adhesive.centrifugal_force_N
    weakest = max(inner_fy)                                       # least negative = weakest press-on
    assert weakest < -fc, mismatch("weakest inward magnetic force on an inner block vs centrifugal (must be < -Fc)",
                                   _cell(res.temperature.adhesive, "centrifugal_force_N"), weakest, -fc, 0.0)


@pytest.mark.family("temperature")
def test_magnetics_press_outer_blocks_onto_cup():
    """Same README claim, outer ring: the net magnetic radial force on an outer block points into the cup at every angle."""
    inp, res = defaults(), run()
    _, outer_fy = _press_on_forces(inp, res)
    weakest = min(outer_fy)
    assert weakest > 0.0, mismatch("weakest outward magnetic force on an outer block (must be > 0)",
                                   "README Temperature design claim; no workbook cell", weakest, 0.0, 0.0)


# =========================================================================== thermal mismatch (Volkersen)
@pytest.mark.family("temperature")
def test_mismatch_inputs_and_worst_swing():
    """C94, C98-C102: steel CTE and modulus, hub wall, cold limit, worst swing max(cure - cold, governing - cure), bondline."""
    inp, res = defaults(), run()
    mm = res.temperature.mismatch
    expected = {"steel_cte": inp.materials.steel.cte_per_C, "steel_E_GPa": inp.materials.steel.modulus_GPa,
                "steel_thickness_mm": res.model.hub_wall_mm, "cold_limit_C": inp.metal.min_temp_C,
                "worst_swing_C": _swing_reference(inp, res), "current_bondline_mm": inp.metal.bond_inner_mm}
    assert_all([(name, _cell(mm, name), getattr(mm, name), expect, TOL_ALGEBRA) for name, expect in expected.items()])


@pytest.mark.family("temperature")
@pytest.mark.parametrize("which", ["current", "recommended"])
def test_volkersen_peak_shear_matches_derivation(which):
    """C104/C105 and volkersen_peak_shear_MPa equal the independently derived closed form (TOL_ALGEBRA)."""
    inp, res = defaults(), run()
    bond = inp.metal.bond_inner_mm if which == "current" else inp.temperature.mismatch.recommended_bondline_mm
    k = _volkersen_inputs_SI(inp, res, bond)
    expect = ref.volkersen_thermal_peak_shear_Pa(**k) / 1e6
    fn = volkersen_peak_shear_MPa(k["G_Pa"] / 1e9, k["d_alpha_per_C"], k["dT_C"], bond, inp.temperature.mismatch.ndfeb_modulus_GPa,
                                  res.model.inner_thickness_mm, inp.materials.steel.modulus_GPa, res.model.hub_wall_mm,
                                  res.model.inner_length_mm)
    field_name = f"peak_shear_{which}_MPa"
    cell = _cell(res.temperature.mismatch, field_name)
    assert_all([("volkersen_peak_shear_MPa at engine inputs", cell, fn, expect, TOL_ALGEBRA),
                (f"peak end shear ({which} bondline)", cell, getattr(res.temperature.mismatch, field_name), expect, TOL_ALGEBRA)])


@pytest.mark.family("temperature")
def test_volkersen_peak_shear_matches_finite_difference():
    """C104 against the finite-difference shear-lag solution (20,001 nodes, O((lam dx)^2) ~ 1e-8; tolerance 1e-6)."""
    inp, res = defaults(), run()
    _, _, tau = ref.volkersen_thermal_fd(**_volkersen_inputs_SI(inp, res, inp.metal.bond_inner_mm))
    expect = float(np.max(np.abs(tau))) / 1e6
    got = res.temperature.mismatch.peak_shear_current_MPa
    assert rel_err(got, expect) < 1e-6, mismatch("peak end shear vs FD", _cell(res.temperature.mismatch, "peak_shear_current_MPa"),
                                                 got, expect, 1e-6)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [
    {"temperature.adhesive.selected": 2},                   # heat cure at 120 degC: cold side governs the swing
    {"temperature.adhesive.selected": 3},
    {"metal.bond_inner_mm": 0.2},
    {"temperature.mismatch.adhesive_shear_modulus_GPa": 1.2, "temperature.mismatch.recommended_bondline_mm": 0.25},
    {"metal.min_temp_C": -20.0, "temperature.mismatch.ndfeb_cte_per_C": -1.5e-6},
], ids=["ea9514", "2214", "bond_0p2", "stiff_glue", "mild_cold"])
def test_volkersen_tracks_derivation_under_varied_inputs(changes):
    """C101, C104, C105 follow the independent chain when adhesive, bondline, modulus, cold limit or CTE change."""
    inp = vary(defaults(), changes)
    res = run(inp)
    mm = res.temperature.mismatch
    items = [(f"worst swing with {changes}", _cell(mm, "worst_swing_C"), mm.worst_swing_C, _swing_reference(inp, res), TOL_ALGEBRA)]
    for which, bond in (("current", inp.metal.bond_inner_mm), ("recommended", inp.temperature.mismatch.recommended_bondline_mm)):
        expect = ref.volkersen_thermal_peak_shear_Pa(**_volkersen_inputs_SI(inp, res, bond)) / 1e6
        items.append((f"peak shear ({which}) with {changes}", _cell(mm, f"peak_shear_{which}_MPa"),
                      getattr(mm, f"peak_shear_{which}_MPa"), expect, TOL_ALGEBRA))
    assert_all(items)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("changes", [{}, {"temperature.mismatch.adhesive_shear_modulus_GPa": 0.107}], ids=["defaults", "soft_glue"])
def test_mismatch_reading_threshold(changes):
    """C106 reads 'Above the lap-shear strength at the block ends' exactly when C104 exceeds the selected lap shear."""
    inp = vary(defaults(), changes)
    res = run(inp)
    s1 = ref.volkersen_thermal_peak_shear_Pa(**_volkersen_inputs_SI(inp, res, inp.metal.bond_inner_mm)) / 1e6
    lap = _selected_candidate(inp).lap_shear_MPa
    assert_all([text_item(f"mismatch reading (reference peak {s1:.4f} vs lap shear {lap} MPa)",
                          _cells(res.temperature.mismatch, "reading", "peak_shear_current_MPa"),
                          res.temperature.mismatch.reading, _reading_reference(s1, lap))])


@pytest.mark.family("temperature")
def test_adhesive_shear_modulus_matches_default_adhesive_tds():
    """Literature: the adhesive shear modulus (C96) should match the selected adhesive's TDS tensile modulus through
    G = E/(2(1+nu)); any nu in 0.3-0.5 is accepted, so the tolerance is the half-width of that G band. C96 is one
    input shared by every candidate (not a per-candidate column), so only the default selection (AA 326) is tested."""
    inp = defaults()
    cand = _selected_candidate(inp)
    tds = next(t for t in ref.ADHESIVE_TDS if t.name == cand.name)
    g_lo, g_hi = ref.shear_modulus_GPa(tds.tensile_modulus_GPa, 0.5), ref.shear_modulus_GPa(tds.tensile_modulus_GPa, 0.3)
    g_mid, half = (g_lo + g_hi) / 2, (g_hi - g_lo) / 2
    got = inp.temperature.mismatch.adhesive_shear_modulus_GPa
    assert g_lo <= got <= g_hi, mismatch(f"adhesive shear modulus vs {tds.name} TDS ({tds.source})", "Temperature design!C96",
                                         got, g_mid, half / g_mid)


@pytest.mark.family("temperature")
@pytest.mark.parametrize("variant", ["biaxial_stiffness", "datasheet_ndfeb_cte"])
def test_mismatch_reading_robust_to_model_details(variant):
    """Model form: plane-stress biaxial adherend stiffness E/(1-nu) (nu 0.24 NdFeB, 0.29 steel), or the Arnold NdFeB
    CTE perpendicular to magnetization (-1.0e-6/degC instead of -0.8e-6), must not flip the C106 reading."""
    inp, res = defaults(), run()
    kwargs = {"biaxial": True} if variant == "biaxial_stiffness" else {"ndfeb_cte": ref.ARNOLD_N42SH["cte_perpendicular_per_C"]}
    s1_alt = ref.volkersen_thermal_peak_shear_Pa(**_volkersen_inputs_SI(inp, res, inp.metal.bond_inner_mm, **kwargs)) / 1e6
    lap = _selected_candidate(inp).lap_shear_MPa
    mm = res.temperature.mismatch
    assert_all([text_item(f"mismatch reading with {variant} (peak {s1_alt:.4f} vs engine {mm.peak_shear_current_MPa:.4f} MPa)",
                          _cells(mm, "reading", "peak_shear_current_MPa"), mm.reading, _reading_reference(s1_alt, lap))])
```

- [ ] **Step 5: Run the checks**

Run:
```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py
./.venv/Scripts/python -m pytest -q audit/tests/test_temperature_demag_adhesive.py -rA
```

Expected: **65 passed, 5 failed** (70 checks from 33 test functions). The 5 FAILs are **candidate findings for Task 8, not bugs to fix now**. Each is a separate root cause. Do not tune a check or edit `magcoupling/`.

PASS (values at workbook defaults unless noted):

Demagnetization
- `test_h_ref_matches_permeance_coefficient_one_load_line`: C48 = 513.2747165649268 kA/m.
- `test_calibration_reference_onset_and_offset`: C49 = 160.4269098639415 °C and C50 = 10.42690986394149 °C. The Brent root agrees with the engine's closed form to 3.5e-16 (C49) and 5.5e-15 (C50), using the K&J rating of 150 °C.
- `test_demag_onset_matches_root_find[aligned|pullout|likepole|single_ring]`: C56 = 169.6514, C57 = 112.8427, C58 = 102.5500, C59 = 142.8509 °C.
- `test_magnet_limit`: C60 = 92.55004986453575 °C.
- `test_demag_chain_tracks_root_find_under_varied_inputs`, 9 cases:
  - Hcj20;
  - β;
  - knee;
  - α_Br;
  - like-pole field plus margin;
  - H above the knee, giving an onset of -2.95 °C;
  - exact µ0;
  - the N52 part, which gives offset 72.25 °C and a skipping onset of 40.73 °C;
  - no library part, which gives offset 0 and a skipping onset of 112.98 °C.
- `test_demag_inputs_link_to_calculator_and_kj_rating[defaults|n52_part|no_library]`:
  - C42 = Calculator C21 (1.29 T, or 1.45 T for N52);
  - C43 = -0.0012 /°C;
  - C47 = 150 °C, 80 °C, or non-numeric 'n/a' for the blank part.
- `test_doc_consistency_readme_demag_numbers`: 10.43, 102.55 and 92.55 °C; skipping is the lowest onset. This is a **documentation-consistency check, not a confirmation**, and Task 8's coverage table should not count it as one.
- `test_torque_at_magnet_limit_scales_with_br_squared`: C61 = 2.374265 N·m; C62 = Calculator C93 = 2.647274 N·m.
- `test_summary_repeats_demag_and_adhesive_values`: C7–C11, and C24 = 120.8509 °C.
- `test_summary_margins_and_hot_day_note`, 5 cases (this task is the sole owner of C13, C15 and F16; both F16 branches appear):

  | Case | C13 (°C) | C15 (°C) | F16 |
  |---|---|---|---|
  | defaults | 42.55 | 27.55 | "Meets it nominally (no variation allowance)" (2.5493 ≥ 2.5 N·m) |
  | op_80C | 12.55 | 27.55 | "Meets it nominally (no variation allowance)" |
  | requirement_2p7 | 42.55 | 27.55 | "Below it" |
  | dp460_governs | 10 | −5 | "Meets it nominally (no variation allowance)" |
  | mild_day | 42.55 | 52.55 | "Meets it nominally (no variation allowance)" (2.7136 N·m) |

- `test_onset_calibration_form_sensitivity_within_design_margin[knee_fraction|beta_hcj|recoil_permeability]`: 92.72 °C (−9.83), 105.64 °C (+3.09) and 100.77 °C (−1.78), against 102.55 °C.
- `test_knee_model_coefficients_vs_n42sh_datasheet`: α_Br and Hcj20 match Arnold. With the datasheet β the onset is 106.67 °C, so the engine is 4.1 °C conservative.

Magnet ratings and library
- `test_magnet_temperature_checks_follow_kj_rating`, 7 cases (this task is the sole owner of C22, C32, C107 and C108):
  - defaults: 150/150, OK/OK;
  - at_rating (150 °C): OK/OK;
  - above_rating (150 °C plus one ulp): OVER/OVER;
  - n52_inner_90C: C22 = 80 and OVER, C32 = 150 and OK;
  - n42_outer_100C: C22 = 150 and OK, C32 = 80 and OVER (catches swapped rings);
  - no_library_inner: C22 exactly `'n/a'` and C107 'unknown', outer 150 and OK;
  - no_library_both: C22 and C32 both exactly `'n/a'`, C107 and C108 both 'unknown'.
- `test_library_ratings_follow_kj_grade_table`: all 15 rows match their grade suffix (N, M, SH).

Adhesive
- `test_adhesive_selection_and_governing_limit[1..4]`: for DP460 the governing limit is 60 °C and the note reads "Adhesive governs.".
- `test_adhesive_design_limit_follows_tds[AA 326 | EA 9514]`: 120 = 120, and 113 = min(200, 133 − 20).
- `test_bond_shear_from_cold_high_torque`: area 80.645 mm², r_mid 11.735 mm, F = 32.0811 N, τ = 0.397806 MPa.
- `test_centrifugal_force_from_block_mass`: C88 = 0.9869571 N, with C82 = 1.917335 g as input.
- `test_static_ratio_and_fatigue_screen_at_defaults`: 37.707; "OK: 8x margin" (margin 7.54).
- `test_shear_reversals_equal_pole_pair_passes`: C90 = 3.3333e8.
- `test_fatigue_screen_robust_to_bond_plane_lever_arm`: τ = 0.4599 MPa and margin 6.52, so the verdict stays OK.
- `test_magnetics_press_inner_blocks_onto_hub`: 14.47–34.30 N inward at Br(92.55 °C), against 0.987 N centrifugal.
- `test_magnetics_press_outer_blocks_onto_cup`: 7.51–27.33 N outward.

Volkersen mismatch
- `test_mismatch_inputs_and_worst_swing`: C101 = 70.55 °C.
- `test_volkersen_peak_shear_matches_derivation[current|recommended]`: C104 = 46.12654 MPa and C105 = 26.72826 MPa; the engine cells and `volkersen_peak_shear_MPa` agree with the closed form to 8e-16 or better.
- `test_volkersen_peak_shear_matches_finite_difference`: 2.4e-9.
- `test_volkersen_tracks_derivation_under_varied_inputs`, 5 cases:
  - EA 9514: swing 160 °C, C104 = 104.61 MPa;
  - 2214: swing 161 °C, C104 = 105.26 MPa;
  - bondline 0.2 mm;
  - stiff glue;
  - mild cold.
- `test_mismatch_reading_threshold[defaults|soft_glue]`: "Above the lap-shear strength at the block ends" at 46.13 MPa; "Below the lap-shear strength" at 11.60 MPa.
- `test_mismatch_reading_robust_to_model_details[biaxial_stiffness|datasheet_ndfeb_cte]`: 49.55 MPa (+7.4 %) and 46.83 MPa (+1.5 %); the reading does not change.

FAIL (5 candidate findings for Task 8, one per root cause):
1. `test_onset_calibration_form_sensitivity_within_design_margin[knee_fraction+recoil_permeability]` (model-form uncertainty larger than the design margin). Calibrating the knee fraction with the datasheet recoil permeability of 1.056 gives a skipping onset of 88.40 °C, against the engine's 102.55 °C. The difference is −14.15 °C; the tolerance is the 10 °C margin (C51).
2. `test_knee_evaluated_inside_coefficient_validity_range` (model approximation). The linear coefficients are used past the 20–150 °C range they were measured over:
   - Pc = 1 reference magnet: 160.43 °C (C49);
   - aligned: 180.08 °C;
   - single ring: 153.28 °C.

   The governing skipping case, at 112.98 °C, is inside the range.
3. `test_library_br_within_kj_grade_range` (input data, conservative). B842SH, BX042SH and BX082SH (N42SH) carry Br20 = 1.29 T, below K&J's published N42SH range of 1.30–1.33 T; the B842SH product page gives Br max 13,200 G. Both default rings are B842SH, so the torque (Br² law) is about 1.5 % lower than at the bottom of the range (1.30 T) and about 3.8 % lower than at its middle (1.315 T), i.e. conservative.
4. `test_fatigue_screen_uses_fatigue_endurance_input` (error/consistency). C91 hard-codes 0.2 instead of using the C195 input. At an endurance of 0.1 the engine still reports "OK: 8x margin", while the reference gives "CHECK" (margin 3.77). No effect at defaults.
5. `test_adhesive_shear_modulus_matches_default_adhesive_tds` (input data). C96 = 0.55 GPa, but the AA 326 TDS modulus of 300 MPa gives G = 0.100–0.115 GPa (relative error 4.1); C96 is also one value shared by every candidate. With G = 0.107 GPa:
   - C104 falls from 46.13 to 11.60 MPa and C105 from 26.73 to 6.03 MPa;
   - C106 flips to "Below the lap-shear strength";
   - in part B, C201 falls from 11.37 to 2.56 MPa and C202 flips from "Above the fatigue endurance: qualify by thermal cycling" to "Below the fatigue endurance".

- [ ] **Step 6: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 add reference/magcoupling-py/audit/references/demag_adhesive.py reference/magcoupling-py/audit/references/slab_field2d.py reference/magcoupling-py/audit/tests/test_temperature_demag_adhesive_reference_sanity.py reference/magcoupling-py/audit/tests/test_temperature_demag_adhesive.py
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 commit -F - <<'MSG'
test(magcoupling-audit): temperature independent checks

Demagnetization onsets (root-find, load line, K&J ratings, calibration-form and
datasheet cross-checks), Calculator magnet temperature checks and library data,
adhesive limits and bond loads (TDS data, press-on force from the slab forces
appended to slab_field2d), and the Volkersen thermal-mismatch screen
(derivation and FD solution).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
MSG
```

### Task 6: Slip losses, slip thermal network, life and part-B summary: independent checks

**Files:**
- Create: `reference/magcoupling-py/audit/references/slip_thermal.py`
- Create: `reference/magcoupling-py/audit/tests/test_slip_thermal_reference_sanity.py`
- Create: `reference/magcoupling-py/audit/tests/test_slip_thermal.py`
- Create: `reference/magcoupling-py/audit/tools/__init__.py` (empty)
- Create: `reference/magcoupling-py/audit/tools/placeholder_sensitivity.py`

**Interfaces:**
- Consumes:
  - `audit.common` helpers, all from Task 1, which owns `audit/common.py` completely (this task neither creates nor edits it, and adds no helper module of its own): `defaults`, `run`, `vary`, `rel_err`, `mismatch`, `TOL_ALGEBRA`, `TOL_MODEL` (2 %; imported, never defined in this task's files), `MU0_EXACT` (sanity tests only), and the multi-cell helpers:
    - `assert_all(items) -> None`, where items is an iterable of `(what, cells, engine, reference, tol)`
    - `text_item(what, cells, engine_text, expected_text) -> tuple`
    - `ScenarioCache(scenarios)`, where `cache[name]` returns `(inputs, results)`
  - Task 4's `audit.references.metal_stack`:
    - `pole_pairs(npole) -> float` (N / 2)
    - `omega_rad_s(rpm) -> float` (2 pi rpm / 60)
    - `pole_pair_frequency_Hz(npole, rpm) -> float`
    - `shaft_power_W(torque_Nm, rpm) -> float`
  - Task 4's `audit.references.remanence`:
    - `torque_at(torque_ref_Nm, alpha_per_C, ref_C, temp_C) -> float`
  - `numpy`, and `scipy` (`scipy.integrate.solve_ivp`, `scipy.linalg.solve_banded`), from `requirements-audit.txt`.
  - Engine result fields:
    - `temperature.slip_loss.{steel_sigma_S_m, steel_mu_r, cap_sigma_S_m, skin_depth_mm, hub_W, cup_W, web_W, sleeve_W, liner_W, cap_W, magnets_W, total_W, drag_Nm, used_W, high_W}`
    - `temperature.duty.{slip_rpm, slip_rad_s, pole_pairs, field_freq_Hz, field_omega_rad_s, slip_event_s, hot_day_start_C}`
    - `temperature.thermal.{steel_c, heat_capacity_J_K, time_constant_s, start_C, rise_per_event_C, steady_rise_est_C, steady_rise_high_C, steady_est_C, steady_high_C, time_to_limit_high, rotations_to_limit_high, time_to_limit_est, critical_drag_Nm, heating_rate_est_C_s, heating_rate_high_C_s, rev_per_C_est, rev_per_C_high, rev_per_tau, t95_s, rev95, temp_at_fault_C}`
    - `temperature.slip_life.{events, rev_per_event, rotations, slip_hours, like_pole_passes, heat_per_event_est_J, heat_per_event_high_J, rise_per_event_est_C, rise_per_event_high_C, life_heat_high_MJ, slip_duty, avg_rise_est_C, avg_rise_high_C, rise_per_pct_duty_C}`
    - `temperature.magnet_life.{peak_C, margin_onset_C, margin_limit_C, torque_hot_day_Nm, torque_hot_day_check, torque_peak_Nm}`
    - `temperature.adhesive_life.{peak_C, margin_C, reversals, torque_peak_var_Nm, shear_amplitude_MPa, hot_fatigue_margin, hot_fatigue_screen, daily_cycles, daily_peak_shear_MPa, daily_screen}`
    - `temperature.summary.{service_max_C, governing_limit_C, hot_day_start_C, margin_hot_day_C, torque_hot_day_Nm, steady_estimate_C, steady_high_C, time_to_limit_high, peak_with_fault_C, life_rotations, avg_slip_heating_high_C, critical_drag_Nm, cure_margin_C, verdict}`
    - `temperature.demag.{onset_skipping_C, magnet_limit_C}`
    - `temperature.mismatch.{peak_shear_recommended_MPa, worst_swing_C}`
  - Engine rows verified by other tasks, used here as inputs:
    - `model.{active_length_mm, outer_back_apothem_mm, inner_thickness_mm, inner_width_mm, inner_length_mm, pullout_20C_Nm}`
    - `mass.{magnets_g, cup_g, hub_g, boss_g, total_g}`
    - `retainers.{sleeve_id_mm, sleeve_od_mm, liner_id_mm, liner_od_mm, retainers_g, endplates_g, cap_g}`
- Produces:
  - `audit.references.slip_thermal`:
    - `skin_depth_m(omega, mu_r, sigma, mu0) -> float`
    - `halfspace_gamma(omega, mu_r, sigma, k, mu0) -> complex`
    - `halfspace_surface_bn(b_image, omega, mu_r, sigma, k, mu0) -> float`
    - `halfspace_loss_per_area(bn, omega, mu_r, sigma, k, mu0) -> float`
    - `halfspace_loss_thin_skin(bn, omega, mu_r, sigma, k, mu0) -> float`
    - `halfspace_poynting_per_area(bn, omega, mu_r, sigma, k, mu0) -> float`
    - `thin_sheet_loss_low_speed(bn, v, sigma, t) -> float`
    - `thin_disk_loss_low_speed(omega, sigma, t, bz2_r2_integral) -> float`
    - `thin_sheet_reaction_factor(v, sigma, t, mu0) -> float`
    - `sheet_end_factor(k, field_length, overhang=0.0) -> float`
    - `sheet_end_factor_numeric(k, field_length, overhang=0.0, cells=4000) -> float`
    - `strip_loss_per_volume(b, omega, sigma, d) -> float`
    - `rect_section_factor(aspect, terms=400) -> float`
    - `loglog_slope(x1, y1, x2, y2) -> float`
    - `central_difference(f, x0, rel_step=1e-4) -> float`
    - `heat_capacity_J_K(parts: Iterable[tuple[float, float]]) -> float`
    - `first_order_rise(power_W, conductance_W_K, capacity_J_K, t_s) -> float`
    - `time_to_rise(target_rise, power_W, conductance_W_K, capacity_J_K) -> float` (0 when already reached, inf when never reached)
    - `ode_rise(power_W, conductance_W_K, capacity_J_K, t_s) -> float`
    - `ode_time_to_rise(target_rise, power_W, conductance_W_K, capacity_J_K, t_max_s) -> float`
    - `event_train_mean_rise(power_W, conductance_W_K, capacity_J_K, t_on_s, period_s) -> float`
  - `audit.tools.placeholder_sensitivity`:
    - `PLACEHOLDERS`
    - `input_value(inp, path) -> float`
    - `sensitivity(path, getter, rel_step=1e-4) -> float`
    - `main()`, which prints the findings report's "Placeholder inputs" table

- [ ] **Step 1: Write the reference module**

The multi-cell helpers the checks use (`assert_all`, `text_item`, `ScenarioCache`) already exist in Task 1's `audit/common.py`, so this step writes only the reference module.

File: `reference/magcoupling-py/audit/references/slip_thermal.py` (create)
```python
"""Independent references for slip losses, the slip thermal network and life totals.

Written from the cited sources; nothing here imports ``magcoupling``. The Task 6
checks (``audit/tests/test_slip_thermal.py``) compare the engine's Temperature
design slip-loss, thermal, slip-life, magnet-life, adhesive-life and summary rows
against these functions; ``audit/tools/placeholder_sensitivity.py`` uses
``central_difference`` for the report's placeholder table.

Units: SI throughout (m, s, T, S/m, W, J, K), except masses in grams where named.

Sources
  [J]  J. D. Jackson, Classical Electrodynamics, 3rd ed. (Wiley, 1999), sec. 8.1:
       skin depth delta = sqrt(2/(mu sigma omega)); mean loss per unit area of a
       good conductor mu_c omega delta |H_par|^2 / 4.
  [HB] W. H. Hayt, J. A. Buck, Engineering Electromagnetics (McGraw-Hill):
       copper (sigma = 5.8e7 S/m) has delta = 66.1/sqrt(f) mm.
  [S]  R. L. Stoll, The Analysis of Eddy Currents (Clarendon Press, 1974):
       travelling field over a conducting, permeable half-space,
       A = A0 exp(-gamma y) exp(j(omega t - k x)), gamma^2 = k^2 + j omega mu sigma.
  [RN] R. L. Russell, K. H. Norsworthy, "Eddy currents and wall losses in
       screened-rotor induction motors", Proc. IEE 105A (1958) 163-175:
       end factor of a finite-length thin screen, 1 - tanh(a)/a, a = k L / 2.
  [R]  J. R. Reitz, "Forces on moving magnets due to eddy currents",
       J. Appl. Phys. 41 (1970) 2067: thin-sheet characteristic speed
       w = 2/(mu0 sigma t); the eddy drag peaks at v = w.
  [B]  G. Bertotti, Hysteresis in Magnetism (Academic Press, 1998): classical
       eddy loss of a lamination, P/V = pi^2 sigma d^2 f^2 B^2 / 6.
  [TG] S. P. Timoshenko, J. N. Goodier, Theory of Elasticity, 3rd ed.
       (McGraw-Hill, 1970): torsion of a rectangular bar, J_t = beta a^3 b with
       beta = 0.141, 0.229, 0.263, 0.312, 1/3 for b/a = 1, 2, 3, 10, infinity.
       The low-frequency eddy stream function in a prism solves the same
       Poisson problem (Prandtl membrane analogy).
  [I]  F. P. Incropera, D. P. DeWitt et al., Fundamentals of Heat and Mass
       Transfer, 6th ed. (Wiley, 2007), ch. 5: lumped capacitance,
       theta(t) = theta_ss (1 - exp(-t/tau)), tau = C/G.
"""
from __future__ import annotations

import cmath
import math
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded


# ============================================================ eddy currents: steel
def skin_depth_m(omega: float, mu_r: float, sigma: float, mu0: float) -> float:
    """[J] Skin depth delta = sqrt(2 / (omega mu0 mu_r sigma)), in metres."""
    return math.sqrt(2.0 / (omega * mu0 * mu_r * sigma))


def halfspace_gamma(omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> complex:
    """[S] Decay constant inside the half-space, gamma = sqrt(k^2 + j omega mu0 mu_r sigma), Re gamma > 0."""
    return cmath.sqrt(k * k + 1j * omega * mu0 * mu_r * sigma)


def halfspace_surface_bn(b_image: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """[S] Normal flux-density amplitude at the surface of a conducting, permeable half-space.

    ``b_image`` is the source's normal field at the surface doubled: the value an
    infinitely permeable, non-conducting surface carries (image method). With
    A = S e^{-ky} + R e^{ky} in air and T e^{-gamma y} inside, continuity of A and of
    H_t gives T = 2S / (1 + gamma/(mu_r k)), so Bn = b_image / |1 + gamma/(mu_r k)|.
    Limits: mu_r -> inf gives b_image; sigma = 0 gives b_image mu_r/(mu_r + 1), the
    static image coefficient (mu_r - 1)/(mu_r + 1).
    """
    g = halfspace_gamma(omega, mu_r, sigma, k, mu0)
    return abs(b_image / (1.0 + g / (mu_r * k)))


def halfspace_loss_per_area(bn: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """[S] Mean loss per unit area for a surface normal amplitude ``bn``, valid for any k*delta.

    Bn = k |A0| and J = -j omega sigma A0 e^{-gamma y}, so
    P/A = int_0^inf |J|^2 / (2 sigma) dy = sigma omega^2 bn^2 / (4 k^2 Re gamma).
    """
    g = halfspace_gamma(omega, mu_r, sigma, k, mu0)
    return sigma * omega ** 2 * bn ** 2 / (4.0 * k * k * g.real)


def halfspace_loss_thin_skin(bn: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """k*delta -> 0 limit of ``halfspace_loss_per_area`` (Re gamma -> 1/delta): sigma omega^2 bn^2 delta / (4 k^2)."""
    return sigma * omega ** 2 * bn ** 2 * skin_depth_m(omega, mu_r, sigma, mu0) / (4.0 * k * k)


def halfspace_poynting_per_area(bn: float, omega: float, mu_r: float, sigma: float, k: float, mu0: float) -> float:
    """[J] Mean Poynting flux into the surface, Re(E_z H_x*)/2, with E_z = -j omega A and H_x = -gamma A/(mu0 mu_r).

    Energy conservation makes it equal ``halfspace_loss_per_area``; the sanity test uses that."""
    g = halfspace_gamma(omega, mu_r, sigma, k, mu0)
    a = bn / k
    e_z = -1j * omega * a
    h_x = -g * a / (mu0 * mu_r)
    return 0.5 * (e_z * h_x.conjugate()).real


# ============================================================ eddy currents: thin sheets
def thin_sheet_loss_low_speed(bn: float, v: float, sigma: float, t: float) -> float:
    """Low-speed loss per unit area of a thin non-magnetic sheet swept at speed v by a normal field of amplitude bn.

    Infinitely long sheet, so the currents run straight across it: E = v x B,
    J = sigma v bn cos(kx - wt), P/A = t <J^2> / sigma = sigma t v^2 bn^2 / 2."""
    return sigma * t * v * v * bn * bn / 2.0


def thin_disk_loss_low_speed(omega: float, sigma: float, t: float, bz2_r2_integral: float) -> float:
    """Low-speed loss of a thin non-magnetic disk turning at omega relative to an axial field.

    v = omega r and E = v x B is radial, so an unbounded sheet has loss density sigma (omega r)^2 <Bz^2>
    (<.> = mean over angle, no return-path correction). Over the disk volume:
    sigma t omega^2 * int <Bz^2> r^2 dA, where ``bz2_r2_integral`` = int <Bz^2> r^2 dA in T^2 m^4."""
    return sigma * t * omega ** 2 * bz2_r2_integral


def thin_sheet_reaction_factor(v: float, sigma: float, t: float, mu0: float) -> float:
    """[R] Loss reduction from the sheet's own field: 1 / (1 + eps^2), eps = v / w = mu0 sigma t v / 2.

    One spatial harmonic k, sheet in free space: the sheet current K = -j omega sigma t A_s adds
    mu0 K / (2k) to the potential, so A_s = A_0 / (1 + j eps) with eps = omega sigma t mu0 / (2k)."""
    eps = mu0 * sigma * t * v / 2.0
    return 1.0 / (1.0 + eps * eps)


def sheet_end_factor(k: float, field_length: float, overhang: float = 0.0) -> float:
    """[RN] Low-speed loss of a finite thin sheet relative to the infinitely long one.

    Normal field bn cos(kx - wt), uniform over ``field_length`` and zero beyond; the sheet
    extends ``overhang`` past each end. With phi = f(z) cos(kx), J = sigma(-grad phi + v B z_hat)
    closes through the sheet (div J = 0, J_z = 0 at its edges): f = A sinh(kz) in the field,
    D cosh(k(L/2 + h - |z|)) in the overhang, f continuous and f' jumping by v bn at |z| = L/2.
    The loss (work of the motional field) relative to the long sheet is
        1 - tanh(a) / (a (1 + tanh(a) tanh(k h))),   a = k L / 2.
    ``overhang = 0`` is Russell and Norsworthy's 1 - tanh(a)/a."""
    a = k * field_length / 2.0
    return 1.0 - math.tanh(a) / (a * (1.0 + math.tanh(a) * math.tanh(k * overhang)))


def sheet_end_factor_numeric(k: float, field_length: float, overhang: float = 0.0, cells: int = 4000) -> float:
    """Finite-volume solution of the ``sheet_end_factor`` problem, loss integrated as int |J|^2 / sigma.

    Independent of the closed form: no jump conditions are imposed by hand and the loss is summed
    from the current density, not from the work of the motional field. sigma = v bn = 1.
    Unknown f_i: cos(kx) amplitude of phi per cell along the sheet. Face current
    J = (f_i - f_{i+1} + e_i d_i + e_{i+1} d_{i+1}) / (d_i + d_{i+1}) (e = 1 in the field,
    d = half cell width), J = 0 at the sheet edges; cell balance J_right - J_left + k^2 w_i f_i = 0,
    the last term being the divergence of J_x = k f sin(kx)."""
    n_over = max(1, round(cells * overhang / (field_length + 2 * overhang))) if overhang > 0 else 0
    n_field = cells - 2 * n_over
    over = np.full(n_over, overhang / n_over) if n_over else np.empty(0)
    widths = np.concatenate([over, np.full(n_field, field_length / n_field), over])
    emf = np.concatenate([np.zeros(n_over), np.ones(n_field), np.zeros(n_over)])
    half = widths / 2
    dist = half[:-1] + half[1:]                                   # centre-to-centre across each interior face
    face_emf = (emf[:-1] * half[:-1] + emf[1:] * half[1:]) / dist
    diag = k * k * widths
    diag[:-1] += 1 / dist
    diag[1:] += 1 / dist
    rhs = np.zeros(widths.size)
    rhs[:-1] -= face_emf
    rhs[1:] += face_emf
    bands = np.zeros((3, widths.size))
    bands[0, 1:] = -1 / dist
    bands[1] = diag
    bands[2, :-1] = -1 / dist
    f = solve_banded((1, 1), bands, rhs)
    jz = np.concatenate([[0.0], (f[:-1] - f[1:]) / dist + face_emf, [0.0]])
    faces = np.concatenate([[0.0], np.cumsum(widths)])
    int_jz2 = float(np.sum((jz[:-1] ** 2 + jz[1:] ** 2) / 2 * np.diff(faces)))   # trapezoid, kink on a face
    int_jx2 = float(np.sum((k * f) ** 2 * widths))                               # midpoint
    return (int_jz2 + int_jx2) / field_length


# ============================================================ eddy currents: magnets
def strip_loss_per_volume(b: float, omega: float, sigma: float, d: float) -> float:
    """[B] Low-frequency eddy loss per unit volume of a thin strip of thickness d in a uniform
    alternating field of amplitude b along its faces: sigma omega^2 b^2 d^2 / 24 (= pi^2 sigma d^2 f^2 b^2 / 6)."""
    return sigma * omega ** 2 * b ** 2 * d ** 2 / 24.0


def rect_section_factor(aspect: float, terms: int = 400) -> float:
    """[TG] Low-frequency eddy loss of a long rectangular prism relative to ``strip_loss_per_volume`` (d = short side).

    Section a x b with aspect = b/a >= 1, alternating field uniform and along the prism. The stream
    function psi (J = curl(psi z_hat)) solves laplacian psi = -sigma dB/dt with psi = 0 on the boundary:
    Prandtl's torsion problem. The loss per length is sigma (dB/dt)^2 J_t / 4 with J_t = beta a^3 b, and a
    thin strip has beta = 1/3, so the ratio is
        3 beta = 1 - (192 / (pi^5 aspect)) sum_{n odd} tanh(n pi aspect / 2) / n^5."""
    if aspect < 1:
        raise ValueError("aspect is the long side over the short side (>= 1)")
    s = sum(math.tanh(n * math.pi * aspect / 2) / n ** 5 for n in range(1, 2 * terms, 2))
    return 1.0 - 192.0 / (math.pi ** 5 * aspect) * s


def loglog_slope(x1: float, y1: float, x2: float, y2: float) -> float:
    """Exponent n of y = c x^n through (x1, y1) and (x2, y2)."""
    return math.log(y2 / y1) / math.log(x2 / x1)


def central_difference(f: Callable[[float], float], x0: float, rel_step: float = 1e-4) -> float:
    """df/dx at x0 by the central difference (f(x0 + h) - f(x0 - h)) / 2h with h = rel_step |x0|.

    Truncation error is f'''(x0) h^2 / 6, second order in h; x0 must be non-zero."""
    if x0 == 0.0:
        raise ValueError("central_difference needs a non-zero x0 (the step is relative)")
    h = rel_step * abs(x0)
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


# ============================================================ lumped thermal network
def heat_capacity_J_K(parts: Iterable[tuple[float, float]]) -> float:
    """[I] Lumped heat capacity sum(m c) for parts given as (mass in g, specific heat in J/(kg K))."""
    return sum(m_g * c for m_g, c in parts) / 1000.0


def first_order_rise(power_W: float, conductance_W_K: float, capacity_J_K: float, t_s: float) -> float:
    """[I] Rise after heating at constant power for t_s from equilibrium: (P/G) (1 - exp(-t G / C))."""
    return power_W / conductance_W_K * (1.0 - math.exp(-t_s * conductance_W_K / capacity_J_K))


def time_to_rise(target_rise: float, power_W: float, conductance_W_K: float, capacity_J_K: float) -> float:
    """[I] Time for ``first_order_rise`` to reach ``target_rise``.

    0 when target_rise <= 0 (the limit is already reached at the start); inf when the steady
    rise P/G does not exceed it (it is approached only asymptotically)."""
    if target_rise <= 0.0:
        return 0.0
    steady = power_W / conductance_W_K
    if steady <= target_rise:
        return math.inf
    return -capacity_J_K / conductance_W_K * math.log(1.0 - target_rise / steady)


def _network(power_W: float, conductance_W_K: float, capacity_J_K: float):
    return lambda t, y: [(power_W - conductance_W_K * y[0]) / capacity_J_K]


def ode_rise(power_W: float, conductance_W_K: float, capacity_J_K: float, t_s: float) -> float:
    """Numerical reference: integrate C dtheta/dt = P - G theta from theta = 0 to t_s (DOP853, rtol 1e-12)."""
    sol = solve_ivp(_network(power_W, conductance_W_K, capacity_J_K), (0.0, t_s), [0.0],
                    method="DOP853", rtol=1e-12, atol=1e-14)
    return float(sol.y[0, -1])


def ode_time_to_rise(target_rise: float, power_W: float, conductance_W_K: float, capacity_J_K: float,
                     t_max_s: float) -> float:
    """Numerical reference: first time the integrated rise reaches ``target_rise``.

    0 when target_rise <= 0 (reached at the start); inf when not reached by t_max_s."""
    if target_rise <= 0.0:
        return 0.0

    def hit(t, y):
        return y[0] - target_rise

    hit.terminal = True
    hit.direction = 1
    sol = solve_ivp(_network(power_W, conductance_W_K, capacity_J_K), (0.0, t_max_s), [0.0],
                    method="DOP853", events=hit, rtol=1e-12, atol=1e-14)
    return float(sol.t_events[0][0]) if sol.t_events[0].size else math.inf


def event_train_mean_rise(power_W: float, conductance_W_K: float, capacity_J_K: float,
                          t_on_s: float, period_s: float) -> float:
    """Mean rise of the periodic steady state when power flows for t_on_s in every period_s.

    A separate model from 'duty x steady rise': exact piecewise exponentials of the first-order
    network, the periodic condition solved in closed form, then averaged over one period."""
    tau = capacity_J_K / conductance_W_K
    steady = power_W / conductance_W_K
    e_on, e_off = math.exp(-t_on_s / tau), math.exp(-(period_s - t_on_s) / tau)
    top = steady * (1.0 - e_on) / (1.0 - e_on * e_off)      # rise at the end of each heating pulse
    bottom = top * e_off                                     # rise at the start of each heating pulse
    area_on = steady * t_on_s + (bottom - steady) * tau * (1.0 - e_on)
    area_off = top * tau * (1.0 - e_off)
    return (area_on + area_off) / period_s
```

- [ ] **Step 2: Write the reference's own sanity test**

The sanity test covers the reference created in Step 1 and pins the three shared multi-cell helpers the engine checks use (`assert_all`, `text_item`, `ScenarioCache`, all from Task 1's `audit/common.py`):

File: `reference/magcoupling-py/audit/tests/test_slip_thermal_reference_sanity.py` (create)
```python
"""Sanity tests for what Task 6 adds besides engine checks: audit/references/slip_thermal.py and the
shared multi-cell helpers from audit/common.py (Task 1) that the Task 6 checks rely on.

These test the references and helpers, not the engine: every function the Task 6 checks rely on is
pinned to a published value, a known limit, or an independent numerical solution. Every test name
contains 'sanity' so the coverage table can leave them out of the engine confirmations.
"""
import math

import pytest
from scipy.integrate import solve_ivp

from audit.common import MU0_EXACT, ScenarioCache, assert_all, mismatch, rel_err, text_item
from audit.references import slip_thermal as ref


@pytest.mark.family("thermal")
def test_sanity_skin_depth_copper():
    """[HB] Copper, sigma = 5.8e7 S/m, at 60 Hz: delta = 66.1/sqrt(f) mm = 8.53 mm. Tolerance 1e-3 (the constant has 3 figures)."""
    got = ref.skin_depth_m(2 * math.pi * 60, 1.0, 5.8e7, MU0_EXACT)
    want = 66.1e-3 / math.sqrt(60)
    assert rel_err(got, want) <= 1e-3, mismatch("copper skin depth at 60 Hz", "reference", got, want, 1e-3)


@pytest.mark.family("thermal")
def test_sanity_halfspace_thin_skin_limit():
    """[S] At k*delta = 1e-3 the exact half-space loss equals the thin-skin form; the gap is (k delta)^2/4 = 2.5e-7, tolerance 1e-6."""
    omega, mu_r, sigma = 1000.0, 200.0, 4.5e6
    delta = ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    k = 1e-3 / delta
    got = ref.halfspace_loss_per_area(0.2, omega, mu_r, sigma, k, MU0_EXACT)
    want = ref.halfspace_loss_thin_skin(0.2, omega, mu_r, sigma, k, MU0_EXACT)
    assert rel_err(got, want) <= 1e-6, mismatch("half-space loss, k*delta -> 0", "reference", got, want, 1e-6)


@pytest.mark.family("thermal")
def test_sanity_halfspace_resistance_limited_limit():
    """[S] At k*delta = 1e3 (field decays as e^{-ky}, currents resistance limited) P/A -> sigma omega^2 bn^2 / (4 k^3); tolerance 1e-9."""
    omega, mu_r, sigma, bn = 1000.0, 200.0, 4.5e6, 0.2
    k = 1e3 / ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    got = ref.halfspace_loss_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    want = sigma * omega ** 2 * bn ** 2 / (4 * k ** 3)
    assert rel_err(got, want) <= 1e-9, mismatch("half-space loss, k*delta -> inf", "reference", got, want, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_halfspace_matches_jackson_surface_loss():
    """[J] Thin-skin limit (k*delta = 1e-3): loss = mu omega delta |H_t|^2 / 4 with |H_t| = |gamma| bn / (k mu); tolerance 1e-6."""
    omega, mu_r, sigma, bn = 1000.0, 200.0, 4.5e6, 0.2
    mu = MU0_EXACT * mu_r
    delta = ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    k = 1e-3 / delta
    h_t = abs(ref.halfspace_gamma(omega, mu_r, sigma, k, MU0_EXACT)) * bn / (k * mu)
    want = mu * omega * delta * h_t ** 2 / 4
    got = ref.halfspace_loss_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    assert rel_err(got, want) <= 1e-6, mismatch("half-space loss vs Jackson surface loss", "reference", got, want, 1e-6)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("k_delta", [0.1, 0.643, 3.0])
def test_sanity_halfspace_poynting_equals_ohmic_loss(k_delta):
    """[J] Energy conservation at finite k*delta: Poynting flux into the surface = integral of |J|^2/(2 sigma). Tolerance 1e-12."""
    omega, mu_r, sigma, bn = 1047.0, 200.0, 4.5e6, 0.207
    k = k_delta / ref.skin_depth_m(omega, mu_r, sigma, MU0_EXACT)
    got = ref.halfspace_loss_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    want = ref.halfspace_poynting_per_area(bn, omega, mu_r, sigma, k, MU0_EXACT)
    assert rel_err(got, want) <= 1e-12, mismatch(f"ohmic loss vs Poynting flux, k*delta={k_delta}", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_surface_bn_infinite_permeability_is_the_image():
    """[S] mu_r -> inf: the surface field is the image-method (doubled) field. The correction |gamma|/(mu_r k)
    falls only as mu_r^-1/2 (1.4e-7 at mu_r = 1e12), so mu_r = 1e24 is used (1.4e-13); tolerance 1e-9."""
    got = ref.halfspace_surface_bn(0.4, 1047.0, 1e24, 4.5e6, 400.0, MU0_EXACT)
    assert rel_err(got, 0.4) <= 1e-9, mismatch("surface Bn, mu_r -> inf", "reference", got, 0.4, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_surface_bn_static_image_coefficient():
    """Non-conducting permeable half-space: image coefficient (mu_r - 1)/(mu_r + 1), so Bn = 2S mu_r/(mu_r + 1); tolerance 1e-12."""
    mu_r, source = 200.0, 0.2
    got = ref.halfspace_surface_bn(2 * source, 1047.0, mu_r, 0.0, 400.0, MU0_EXACT)
    want = source * (1 + (mu_r - 1) / (mu_r + 1))
    assert rel_err(got, want) <= 1e-12, mismatch("surface Bn, sigma = 0", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_thin_sheet_drag_peaks_at_reitz_speed():
    """[R] Thin-sheet eddy drag F = P/v peaks at v = w = 2/(mu0 sigma t); grid search, tolerance one grid step (1e-3)."""
    sigma, t = 2.5e7, 1e-3
    w = 2 / (MU0_EXACT * sigma * t)
    speeds = [w * (0.5 + 1e-3 * i) for i in range(1001)]
    drag = [ref.thin_sheet_loss_low_speed(0.3, v, sigma, t) * ref.thin_sheet_reaction_factor(v, sigma, t, MU0_EXACT) / v
            for v in speeds]
    v_peak = speeds[drag.index(max(drag))]
    assert rel_err(v_peak, w) <= 1e-3, mismatch("thin-sheet drag peak speed", "reference", v_peak, w, 1e-3)


@pytest.mark.family("thermal")
def test_sanity_thin_disk_matches_sheet_integrated_over_annulus():
    """A uniform field amplitude bn over an annulus: the disk loss (with <Bz^2> = bn^2/2, int r^2 dA = pi (r1^4 - r0^4)/2)
    equals the thin-sheet loss at v = omega r integrated over the annulus (midpoint rule, 20000 rings); tolerance 1e-8."""
    omega, sigma, t, bn, r0, r1 = 209.4, 2.5e7, 8e-4, 0.05, 0.0145, 0.0214
    n = 20000
    dr = (r1 - r0) / n
    rings = [r0 + (i + 0.5) * dr for i in range(n)]
    want = sum(ref.thin_sheet_loss_low_speed(bn, omega * r, sigma, t) * 2 * math.pi * r * dr for r in rings)
    got = ref.thin_disk_loss_low_speed(omega, sigma, t, bn ** 2 / 2 * math.pi * (r1 ** 4 - r0 ** 4) / 2)
    assert rel_err(got, want) <= 1e-8, mismatch("thin disk vs integrated thin sheet", "reference", got, want, 1e-8)


@pytest.mark.family("thermal")
def test_sanity_russell_norsworthy_value():
    """[RN] No overhang, a = kL/2 = 1: factor 1 - tanh(1) = 0.238406; tolerance 1e-12."""
    got = ref.sheet_end_factor(2.0, 1.0, 0.0)
    want = 1 - math.tanh(1.0)
    assert rel_err(got, want) <= 1e-12, mismatch("Russell-Norsworthy factor at a = 1", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_end_factor_long_sheet_limit():
    """A very long sheet (a = 1e6) has no end loss reduction: factor -> 1; tolerance 1e-5."""
    got = ref.sheet_end_factor(2.0, 1e6, 0.0)
    assert rel_err(got, 1.0) <= 1e-5, mismatch("end factor, L -> inf", "reference", got, 1.0, 1e-5)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("k,length,overhang", [(363.2, 0.0127, 0.0009), (200.0, 0.01, 0.0), (1.0, 1.0, 2.0)])
def test_sanity_end_factor_matches_finite_volume(k, length, overhang):
    """The closed-form end factor (own derivation, overhang included) matches a finite-volume solution that
    integrates |J|^2 directly. The FV error is second order: ~3e-7 at 4000 cells; tolerance 1e-5."""
    got = ref.sheet_end_factor(k, length, overhang)
    want = ref.sheet_end_factor_numeric(k, length, overhang, cells=4000)
    assert rel_err(got, want) <= 1e-5, mismatch(f"end factor vs finite volume (k={k}, L={length}, h={overhang})",
                                                "reference", got, want, 1e-5)


@pytest.mark.family("thermal")
def test_sanity_strip_loss_is_bertotti_classical_loss():
    """[B] sigma omega^2 B^2 d^2 / 24 equals pi^2 sigma d^2 f^2 B^2 / 6; tolerance 1e-12."""
    sigma, f, b, d = 2e6, 50.0, 1.5, 0.35e-3
    got = ref.strip_loss_per_volume(b, 2 * math.pi * f, sigma, d)
    want = math.pi ** 2 * sigma * d ** 2 * f ** 2 * b ** 2 / 6
    assert rel_err(got, want) <= 1e-12, mismatch("thin-strip loss vs Bertotti", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("aspect,beta", [(1.0, 0.141), (2.0, 0.229), (3.0, 0.263), (10.0, 0.312), (math.inf, 1 / 3)])
def test_sanity_rect_section_factor_matches_timoshenko(aspect, beta):
    """[TG] rect_section_factor = 3 beta with Timoshenko's torsion coefficients; tolerance half a unit in
    the table's third figure (0.0005/beta)."""
    got = ref.rect_section_factor(aspect)
    want = 3 * beta
    tol = 0.0005 / beta
    assert rel_err(got, want) <= tol, mismatch(f"rectangular-section factor at b/a={aspect}", "reference", got, want, tol)


@pytest.mark.family("thermal")
def test_sanity_loglog_slope():
    """y = 3 x^1.5 through x = 1000 and 2000 has slope 1.5; tolerance 1e-12."""
    got = ref.loglog_slope(1000.0, 3 * 1000.0 ** 1.5, 2000.0, 3 * 2000.0 ** 1.5)
    assert rel_err(got, 1.5) <= 1e-12, mismatch("log-log slope", "reference", got, 1.5, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_heat_capacity_water_and_steel():
    """[I] 1 kg water (4186 J/(kg K)) plus 500 g steel (473 J/(kg K)) = 4422.5 J/K; tolerance 1e-12."""
    got = ref.heat_capacity_J_K([(1000.0, 4186.0), (500.0, 473.0)])
    assert rel_err(got, 4422.5) <= 1e-12, mismatch("heat capacity sum", "reference", got, 4422.5, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_first_order_rise_at_one_time_constant():
    """[I] After one time constant the rise is (1 - 1/e) of the steady rise; tolerance 1e-12."""
    p, g, c = 7.0, 0.3, 82.0
    got = ref.first_order_rise(p, g, c, c / g)
    want = (1 - math.exp(-1)) * p / g
    assert rel_err(got, want) <= 1e-12, mismatch("first-order rise at t = tau", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
def test_sanity_time_to_95_percent_is_tau_ln20():
    """[I] Time to 95 % of the steady rise = tau ln 20 = 2.9957 tau; tolerance 1e-12."""
    p, g, c = 7.0, 0.3, 82.0
    got = ref.time_to_rise(0.95 * p / g, p, g, c)
    want = c / g * math.log(20)
    assert rel_err(got, want) <= 1e-12, mismatch("time to 95 %", "reference", got, want, 1e-12)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("target,want", [(0.0, 0.0), (-5.0, 0.0), (7.0 / 0.3, math.inf), (30.0, math.inf)])
def test_sanity_time_to_rise_edge_cases(target, want):
    """Target already reached (<= 0) takes 0 s; a target at or above the steady rise is never reached (inf)."""
    got = ref.time_to_rise(target, 7.0, 0.3, 82.0)
    assert got == want, mismatch(f"time to rise, target {target}", "reference", got, want, 0.0)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("t_s", [0.1, 2.0, 273.0, 1000.0])
def test_sanity_ode_rise_matches_closed_form(t_s):
    """Numerical integration agrees with the first-order closed form; DOP853 at rtol 1e-12, tolerance 1e-9."""
    got = ref.ode_rise(7.0, 0.3, 82.0, t_s)
    want = ref.first_order_rise(7.0, 0.3, 82.0, t_s)
    assert rel_err(got, want) <= 1e-9, mismatch(f"ODE rise at {t_s} s", "reference", got, want, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_ode_time_to_rise_matches_closed_form():
    """Event location in the numerical integration agrees with -tau ln(1 - target/steady); tolerance 1e-9."""
    got = ref.ode_time_to_rise(20.0, 7.0, 0.3, 82.0, 1e5)
    want = ref.time_to_rise(20.0, 7.0, 0.3, 82.0)
    assert rel_err(got, want) <= 1e-9, mismatch("ODE time to rise", "reference", got, want, 1e-9)


@pytest.mark.family("thermal")
def test_sanity_event_train_matches_brute_force_integration():
    """The periodic-steady-state mean from event_train_mean_rise matches brute-force integration of 80 periods
    (tau = 10 s, pulse 1 s every 4 s; the start-up transient has decayed by e^-32); tolerance 1e-8."""
    p, g, c, t_on, period = 5.0, 0.5, 5.0, 1.0, 4.0
    theta = 0.0
    for _ in range(80):
        on = solve_ivp(lambda t, y: [(p - g * y[0]) / c, y[0]], (0, t_on), [theta, 0.0], method="DOP853", rtol=1e-12, atol=1e-14)
        off = solve_ivp(lambda t, y: [(-g * y[0]) / c, y[0]], (0, period - t_on), [on.y[0, -1], 0.0], method="DOP853",
                        rtol=1e-12, atol=1e-14)
        theta = off.y[0, -1]
    got = (on.y[1, -1] + off.y[1, -1]) / period
    want = ref.event_train_mean_rise(p, g, c, t_on, period)
    assert rel_err(got, want) <= 1e-8, mismatch("event-train mean rise vs brute force", "reference", got, want, 1e-8)


@pytest.mark.family("thermal")
def test_sanity_event_train_continuous_limit():
    """A pulse that fills the whole period is continuous heating: mean rise = P/G; tolerance 1e-12."""
    got = ref.event_train_mean_rise(5.0, 0.5, 5.0, 4.0, 4.0)
    assert rel_err(got, 10.0) <= 1e-12, mismatch("event-train mean rise, duty 1", "reference", got, 10.0, 1e-12)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("x0", [0.3, -2.0, 20000.0])
def test_sanity_central_difference_matches_analytic_derivative(x0):
    """d/dx (sin(x) + x^3) = cos(x) + 3 x^2. With the relative step 1e-4 the truncation (h^2/6) f'''/f' is at most
    4e-9 and the round-off eps |f| / h at most 2e-12 of f' at these points; tolerance 1e-6."""
    got = ref.central_difference(lambda x: math.sin(x) + x ** 3, x0)
    want = math.cos(x0) + 3 * x0 ** 2
    assert rel_err(got, want) <= 1e-6, mismatch(f"central difference at x0={x0}", "reference", got, want, 1e-6)


@pytest.mark.family("thermal")
def test_sanity_central_difference_rejects_zero_point():
    """A relative step has no size at x0 = 0: the function refuses instead of dividing by zero."""
    with pytest.raises(ValueError):
        ref.central_difference(math.sin, 0.0)


@pytest.mark.family("constants")
def test_sanity_assert_all_passes_within_tolerance():
    """assert_all accepts items whose relative error is within tolerance, including an exact zero reference."""
    assert_all([("a", "reference", 1.0 + 1e-12, 1.0, 1e-9), ("b", "reference", 0.0, 0.0, 0.0)])


@pytest.mark.family("constants")
def test_sanity_assert_all_lists_every_failure():
    """assert_all fails once and names every failing item (each formatted by mismatch); passing items are omitted."""
    with pytest.raises(AssertionError) as err:
        assert_all([("first", "X!C1", 2.0, 1.0, 0.1), ("fine", "X!C2", 1.0, 1.0, 0.1), ("second", "X!C3", 5.0, 1.0, 0.1)])
    msg = str(err.value)
    got = 1.0 if ("first [X!C1]" in msg and "second [X!C3]" in msg and "fine" not in msg) else 0.0
    assert got == 1.0, mismatch(f"assert_all message {msg!r}", "reference", got, 1.0, 0.0)


@pytest.mark.family("constants")
@pytest.mark.parametrize("engine,reference", [(math.nan, 1.0), (math.inf, math.inf)])
def test_sanity_assert_all_never_passes_nan(engine, reference):
    """NaN never passes, and inf against inf gives rel_err NaN, so infinite values must be compared as flags."""
    with pytest.raises(AssertionError):
        assert_all([("nan", "reference", engine, reference, 1.0)])


@pytest.mark.family("constants")
@pytest.mark.parametrize("engine_text,expected_text,want", [("OK", "OK", 0.0), ("CHECK", "OK", 1.0)])
def test_sanity_text_item(engine_text, expected_text, want):
    """text_item passes equal texts and fails different ones (rel_err 0 or 1 against the flag 1.0)."""
    what, cells, engine, reference, tol = text_item("verdict", "X!C1", engine_text, expected_text)
    got = rel_err(engine, reference)
    assert got == want, mismatch(f"text_item {engine_text!r} vs {expected_text!r}", cells, got, want, tol)


@pytest.mark.family("constants")
def test_sanity_scenario_cache_runs_each_scenario_once():
    """ScenarioCache applies the named changes and returns the same objects on the second lookup (one engine run)."""
    cache = ScenarioCache({"poles12": {"coupling.npole": 12}})
    first, second = cache["poles12"], cache["poles12"]
    same = 1.0 if (first[0].coupling.npole == 12 and first is second) else 0.0
    assert same == 1.0, mismatch("scenario cache identity and change", "reference", same, 1.0, 0.0)
```

- [ ] **Step 3: Run the sanity test**

Run:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_slip_thermal_reference_sanity.py -k sanity
```

Expected: PASS (`48 passed`). Observed values:
- Copper skin depth: 8.5316 mm against 8.5335 mm (2.2e-4).
- Finite-volume end factor against the closed form: 1.1e-7, 8.1e-8 and 2.8e-7.
- 3 beta = 0.42173, 0.68605, 0.78995, 0.93698 and 1.0, against Timoshenko's 0.423, 0.687, 0.789, 0.936 and 1.
- Event train against brute-force integration: 1.3e-14.
- ODE time to rise: 5.0e-13.
- Central difference: 6.2e-10, 3.7e-9 and 3.0e-9.

- [ ] **Step 4: Write the engine checks and the placeholder report tool**

File: `reference/magcoupling-py/audit/tests/test_slip_thermal.py` (create)
```python
"""Task 6 engine checks: slip losses, the slip thermal network, slip life, magnet and adhesive life, and the
part-B summary rows of the Temperature design sheet.

References: audit/references/slip_thermal.py (this task), Task 4's audit/references/metal_stack.py
(omega_rad_s, pole_pairs, pole_pair_frequency_Hz, shaft_power_W) and audit/references/remanence.py (torque_at).
Multi-cell checks use the shared helpers in audit/common.py (assert_all, text_item, ScenarioCache).

One test per root cause: cells that one engine formula or one missing guard would get wrong together are
checked in one test (assert_all lists every failing cell), so Task 8 receives one candidate per root cause.

Tolerances
  TOL_ALGEBRA  the engine's closed form re-derived here (same algebra).
  TOL_MODEL    (audit.common) engine closed form vs the exact solution of the idealized problem it approximates:
               2 %. A failure is a model-approximation candidate.
  TOL_NUMERIC  engine closed form vs numerical integration (DOP853, rtol 1e-12): 1e-7.
  TOL_LABEL    a label that states '95 %' is met when the reached fraction rounds to 95 %.
Re-derivations use the engine's mu0 input (Calculator!C43, rounded 1.256637e-6), so a same-algebra check isolates
transcription; the rounding itself belongs to the constants family.

Inputs taken from engine rows that other tasks verify: part masses and the rotating mass (Task 4), the governing
limit, magnet limit, onsets, cure margin and the Volkersen worst-swing shear (Task 5), pull-out at 20 C (Tasks 2-4).
Ownership: this file owns the Temperature design result rows C6, C14, C16-C23, C25, C28-C33, C37 and C109-C202
(C13, C15 and F16 belong to Task 5's test_summary_margins_and_hot_day_note; C24 and C90 to Task 5). Metal design
C86, C89, C91 and C92 (slip duty) belong to Task 4 (test_slip_duty, test_slip_loss_from_measured_drag). Task 7's
test_thermal_chain_units and test_thermal_scaling_with_conductance check units and scaling on C141-C153, a
different method.

Not independently checkable here (listed for the report): the cap end factor (0.7 in C128; the radial profile of the
cap end field is collapsed into cap_integral_T2m4), the web's r^2 weighting ((r_mid/pp)^2 * int B^2 dA instead of
int B^2 (r/pp)^2 dA; profile collapsed into web_integral_T2m2), the seven 3D field inputs C116-C122 (fields3d, M3
scope), harmonics above the fundamental, the finite cup wall (1.8 mm at the corners against delta 1.30 mm) versus the
half-space, the single-lump network (the split of the 0.3 W/K between the two shafts is not an input), and the
judgement inputs high_multiplier (C133) and end_factor (C114), whose sensitivities audit/tools/placeholder_sensitivity.py
prints for the report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from audit.common import (TOL_ALGEBRA, TOL_MODEL, ScenarioCache, assert_all, defaults, mismatch, rel_err, run,
                          text_item, vary)
from audit.references import slip_thermal as ref
from audit.references.metal_stack import omega_rad_s, pole_pair_frequency_Hz, pole_pairs, shaft_power_W
from audit.references.remanence import torque_at

TOL_NUMERIC = 1e-7
TOL_LABEL = 0.5 / 95

CASES = ScenarioCache({
    "defaults": {},
    "half_speed": {"metal.slip_rpm": 1000},
    "measured_drag": {"metal.measured_drag_Nm": 0.02},
    "low_conductance": {"temperature.thermal.conductance_W_K": 0.08},
    "start_above_limit": {"temperature.duty.driving_rise_C": 40},
    "high_requirement": {"metal.required_min_Nm": 2.6},
    "low_hot_strength": {"temperature.adhesive_life.hot_strength_retained": 0.4},
    "small_daily_swing": {"temperature.adhesive_life.daily_swing_C": 3},
})


@dataclass(frozen=True)
class Slip:
    """Slip kinematics and geometry, from inputs and from geometry rows verified by other tasks."""
    omega: float          # mechanical slip speed, rad/s
    rev_s: float          # relative revolutions per second
    pp: float             # pole pairs per ring
    omega_e: float        # field angular frequency seen by the opposite ring, rad/s
    L: float              # active length, m
    r_hub: float          # hub steel surface radius, m
    r_cup: float          # cup steel surface radius, m
    r_mid: float          # inner block mid radius, m
    r_sleeve: float       # mean sleeve radius, m
    r_liner: float        # mean liner radius, m
    r_cap_out: float      # cap outer radius, m
    overhang: float       # retainer length past the active length at each end, m
    block_short: float    # short in-plane block side (circumferential width), m
    block_long: float     # long in-plane block side (axial length), m
    block_volume: float   # m^3


@dataclass(frozen=True)
class Heat:
    """Thermal network quantities re-derived from inputs, the re-derived slip loss and engine part masses."""
    p_estimate: float     # sum of the re-derived part losses, W
    p_use: float          # bench drag x Omega when entered, else p_estimate, W
    p_high: float         # W
    capacity: float       # J/K
    conductance: float    # W/K
    tau: float            # s
    t0: float             # hot-day start, C
    t_limit: float        # governing limit, C
    rise_limit: float     # t_limit - t0, C (negative when the start is above the limit)
    steady_est: float     # continuous-slip steady temperature, estimate, C
    steady_high: float    # continuous-slip steady temperature, high case, C
    time_to_limit_est: float   # s (inf: never; 0: already there)
    time_to_limit_high: float  # s
    critical_drag: float  # drag whose steady state equals the limit, N·m (0 when the start is above the limit)
    fault_rise: float     # high-case rise at the fault trip time, C
    event_rise: float     # high-case adiabatic rise per event, C
    duty: float           # share of life spent slipping
    avg_rise_high: float  # life-average slip heating, high case, C
    rotations: float      # relative slip rotations over life
    peak: float           # hot-day peak magnet / bond temperature, C


def slip_ctx(inp, res) -> Slip:
    ci, md, m, ret = inp.coupling, inp.metal, res.model, res.retainers
    w, l = m.inner_width_mm / 1000, m.inner_length_mm / 1000
    return Slip(omega=omega_rad_s(md.slip_rpm), rev_s=md.slip_rpm / 60, pp=pole_pairs(ci.npole),
                omega_e=2 * math.pi * pole_pair_frequency_Hz(ci.npole, md.slip_rpm), L=m.active_length_mm / 1000,
                r_hub=(ci.inner_back_apothem_mm - md.bond_inner_mm) / 1000,
                r_cup=(m.outer_back_apothem_mm + md.bond_outer_mm) / 1000,
                r_mid=(ci.inner_back_apothem_mm + m.inner_thickness_mm / 2) / 1000,
                r_sleeve=(ret.sleeve_id_mm + ret.sleeve_od_mm) / 4 / 1000,
                r_liner=(ret.liner_id_mm + ret.liner_od_mm) / 4 / 1000,
                r_cap_out=md.cap_od_mm / 2 / 1000,
                overhang=(md.retainer_span_mm - m.active_length_mm) / 2 / 1000,
                block_short=min(w, l), block_long=max(w, l),
                block_volume=m.inner_length_mm * m.inner_width_mm * m.inner_thickness_mm * 1e-9)


def steel_losses(inp, res, exact: bool, web_image: float) -> dict[str, float]:
    """Hub, cup and web loss of a travelling field over steel.

    exact=False: the thin-skin limit (k*delta -> 0, the form the engine uses); exact=True: Stoll's half-space with
    finite k*delta and the finite-permeability surface reflection. b_hub_T and b_cup_T are already doubled at the
    steel surface (fields3d.run doubles both; the b_hub_T help text, C116, says so). web_image multiplies the web
    end-field amplitude taken from web_integral_T2m2 (C121): 1 uses it as given, 2 doubles it at the steel web as hub
    and cup are doubled. The web wave number is taken at the block mid radius, as the engine takes it."""
    c, sl = slip_ctx(inp, res), inp.temperature.slip_loss
    st = inp.materials.steel
    steel = dict(omega=c.omega_e, mu_r=st.mu_r_incremental, sigma=st.conductivity_S_m, mu0=inp.coupling.mu0)

    def per_area(b_image, k):
        if exact:
            return ref.halfspace_loss_per_area(ref.halfspace_surface_bn(b_image, k=k, **steel), k=k, **steel)
        return ref.halfspace_loss_thin_skin(b_image, k=k, **steel)

    return {"hub": per_area(sl.b_hub_T, c.pp / c.r_hub) * 2 * math.pi * c.r_hub * c.L,
            "cup": per_area(sl.b_cup_T, c.pp / c.r_cup) * 2 * math.pi * c.r_cup * c.L,
            # per-area loss for the unit end-field amplitude (times web_image) times the end-field integral int B^2 dA
            "web": per_area(web_image, c.pp / c.r_mid) * sl.web_integral_T2m2}


def rederived_losses(inp, res) -> dict[str, float]:
    """The engine's slip-loss closed forms re-derived as limits of the reference models: thin-skin travelling field
    (hub, cup, web as given), long thin sheet at low speed x the end-factor input (sleeve, liner, cap), thin strip
    (magnets)."""
    c, sl = slip_ctx(inp, res), inp.temperature.slip_loss
    al = inp.materials.aluminium.al6061.conductivity_S_m

    def shell(b, r, t_mm):
        return sl.end_factor * ref.thin_sheet_loss_low_speed(b, c.omega * r, sl.sigma_316_S_m, t_mm / 1000) * 2 * math.pi * r * c.L

    return {
        **steel_losses(inp, res, exact=False, web_image=1.0),
        "sleeve": shell(sl.b_sleeve_T, c.r_sleeve, inp.metal.sleeve_mm),
        "liner": shell(sl.b_liner_T, c.r_liner, inp.metal.liner_mm),
        "cap": sl.end_factor * ref.thin_disk_loss_low_speed(c.omega, al, inp.metal.cap_axial_mm / 1000, sl.cap_integral_T2m4),
        # the engine takes the block width as the strip thickness whichever side is shorter
        "magnets": ref.strip_loss_per_volume(sl.b_magnet_T, c.omega_e, sl.sigma_ndfeb_S_m, res.model.inner_width_mm / 1000)
                   * c.block_volume * 2 * inp.coupling.npole,
    }


def literature_losses(inp, res) -> dict[str, float]:
    """Exact solutions of the idealized problems the engine's closed forms approximate: Stoll's half-space (hub, cup,
    web with its end field doubled at the steel web), Russell-Norsworthy end factor with the retainer overhang times
    Reitz's sheet reaction (sleeve, liner), sheet reaction only (cap), rectangular-section factor (magnets)."""
    c, sl, mu0 = slip_ctx(inp, res), inp.temperature.slip_loss, inp.coupling.mu0
    al = inp.materials.aluminium.al6061.conductivity_S_m
    t_cap = inp.metal.cap_axial_mm / 1000

    def shell(b, r, t_mm):
        t, v = t_mm / 1000, c.omega * r
        return (ref.sheet_end_factor(c.pp / r, c.L, c.overhang) * ref.thin_sheet_loss_low_speed(b, v, sl.sigma_316_S_m, t)
                * ref.thin_sheet_reaction_factor(v, sl.sigma_316_S_m, t, mu0) * 2 * math.pi * r * c.L)

    return {
        **steel_losses(inp, res, exact=True, web_image=2.0),
        "sleeve": shell(sl.b_sleeve_T, c.r_sleeve, inp.metal.sleeve_mm),
        "liner": shell(sl.b_liner_T, c.r_liner, inp.metal.liner_mm),
        # the cap end factor has no independent reference (radial end-field profile unknown): the engine's is kept
        # and only the sheet reaction is added, taken at the cap outer radius where it is largest
        "cap": (sl.end_factor * ref.thin_disk_loss_low_speed(c.omega, al, t_cap, sl.cap_integral_T2m4)
                * ref.thin_sheet_reaction_factor(c.omega * c.r_cap_out, al, t_cap, mu0)),
        "magnets": (ref.strip_loss_per_volume(sl.b_magnet_T, c.omega_e, sl.sigma_ndfeb_S_m, c.block_short)
                    * ref.rect_section_factor(c.block_long / c.block_short) * c.block_volume * 2 * inp.coupling.npole),
    }


def heat_ctx(inp, res) -> Heat:
    c, md, tt = slip_ctx(inp, res), inp.metal, inp.temperature
    th, mass, ret = tt.thermal, res.mass, res.retainers
    steel_c = inp.materials.steel.specific_heat_J_kgK
    measured = md.measured_drag_Nm is not None
    p_estimate = sum(rederived_losses(inp, res).values())
    p_use = shaft_power_W(md.measured_drag_Nm, md.slip_rpm) if measured else p_estimate
    p_high = p_use if measured else p_use * tt.slip_loss.high_multiplier
    capacity = ref.heat_capacity_J_K([
        (mass.magnets_g, th.c_ndfeb), (mass.cup_g, steel_c), (mass.hub_g, steel_c), (mass.boss_g, steel_c),
        (md.hardware_g, steel_c), (ret.retainers_g, th.c_316), (ret.endplates_g, th.c_316), (ret.cap_g, th.c_aluminium)])
    g = th.conductance_W_K
    t0 = tt.duty.hot_ambient_C + tt.duty.driving_rise_C
    t_limit = res.temperature.summary.governing_limit_C
    fault_rise = ref.first_order_rise(p_high, g, capacity, tt.duty.fault_trip_s)
    event_rise = p_high * md.slip_event_s / capacity
    duty = md.life_events * md.slip_event_s / 3600 / tt.duty.life_hours
    return Heat(p_estimate=p_estimate, p_use=p_use, p_high=p_high, capacity=capacity, conductance=g, tau=capacity / g,
                t0=t0, t_limit=t_limit, rise_limit=t_limit - t0,
                steady_est=t0 + p_use / g, steady_high=t0 + p_high / g,
                time_to_limit_est=ref.time_to_rise(t_limit - t0, p_use, g, capacity),
                time_to_limit_high=ref.time_to_rise(t_limit - t0, p_high, g, capacity),
                critical_drag=max(0.0, t_limit - t0) * g / c.omega,
                fault_rise=fault_rise, event_rise=event_rise, duty=duty, avg_rise_high=duty * p_high / g,
                rotations=md.life_events * c.rev_s * md.slip_event_s,
                peak=t0 + max(fault_rise, event_rise) + duty * p_high / g)


def case(name: str):
    """(inputs, results, Slip, Heat) for a named input case; CASES runs the engine once per case."""
    inp, res = CASES[name]
    return inp, res, slip_ctx(inp, res), heat_ctx(inp, res)


def pullout_at(inp, res, temp_C: float) -> float:
    """Pull-out torque at temp_C from the 20 C value (torque ~ Br^2, Br linear in temperature)."""
    return torque_at(res.model.pullout_20C_Nm, inp.calibration.alpha_br_per_C, 20.0, temp_C)


def shear_amplitude(inp, res, h: Heat) -> float:
    """Bond shear per reversal at the peak temperature with +variation: F = T / (N r_mid) per block over L x w (MPa)."""
    m = res.model
    r_mid_mm = inp.coupling.inner_back_apothem_mm + m.inner_thickness_mm / 2
    torque = pullout_at(inp, res, h.peak) * (1 + inp.metal.variation)
    return torque / (inp.coupling.npole * r_mid_mm / 1000) / (m.inner_length_mm * m.inner_width_mm)


def selected_adhesive(inp):
    return inp.temperature.adhesive.candidates[inp.temperature.adhesive.selected - 1]


def hot_fatigue_margin(inp, res, h: Heat) -> float:
    """Hot fatigue strength (lap shear x share retained hot x fatigue endurance) over the shear amplitude."""
    al = inp.temperature.adhesive_life
    return selected_adhesive(inp).lap_shear_MPa * al.hot_strength_retained * al.fatigue_endurance / shear_amplitude(inp, res, h)


def daily_peak_shear(inp, res) -> float:
    """Volkersen shear is linear in the temperature swing, so the daily-cycle shear scales the worst-swing value (MPa)."""
    mm = res.temperature.mismatch
    return mm.peak_shear_recommended_MPa * inp.temperature.adhesive_life.daily_swing_C / mm.worst_swing_C


def as_seconds(value) -> float:
    """Engine time rows are text ('never: ...') when the limit is not reached; that means infinite time."""
    return math.inf if isinstance(value, str) else value


def never_item(what: str, cell: str, engine_value, reference_s: float) -> tuple:
    """A 'limit never reached' comparison as an assert_all item: 1.0 when the engine gives its 'never' text, against
    1.0 when the reference time is infinite (inf against inf has no relative error)."""
    return (f"{what}: engine {engine_value!r}", cell, 1.0 if isinstance(engine_value, str) else 0.0,
            1.0 if math.isinf(reference_s) else 0.0, 0.0)


# ============================================================ slip losses vs first principles and literature
PARTS = {
    "hub": ("hub_W", "Temperature design!C123"),
    "cup": ("cup_W", "Temperature design!C124"),
    "web": ("web_W", "Temperature design!C125"),
    "sleeve": ("sleeve_W", "Temperature design!C126"),
    "liner": ("liner_W", "Temperature design!C127"),
    "cap": ("cap_W", "Temperature design!C128"),
    "magnets": ("magnets_W", "Temperature design!C129"),
}


def model_items(parts, reference, extra_cells: str = "") -> list[tuple]:
    """assert_all items comparing each part's engine loss and its 1000 -> 2000 rpm log-log slope with
    reference(inp, res)[part], at TOL_MODEL. extra_cells names a shared input cell behind the parts."""
    inp, res, _, _ = case("defaults")
    inp_h, res_h, _, _ = case("half_speed")
    want, want_h = reference(inp, res), reference(inp_h, res_h)
    items = []
    for part in parts:
        field, cell = PARTS[part]
        cell += extra_cells
        eng, eng_h = getattr(res.temperature.slip_loss, field), getattr(res_h.temperature.slip_loss, field)
        items.append((f"{part} loss", cell, eng, want[part], TOL_MODEL))
        items.append((f"{part} speed exponent 1000-2000 rpm", cell, ref.loglog_slope(1000.0, eng_h, 2000.0, eng),
                      ref.loglog_slope(1000.0, want_h[part], 2000.0, want[part]), TOL_MODEL))
    return items


@pytest.mark.family("thermal")
def test_steel_skin_depth():
    """Steel skin depth at the field frequency, delta = sqrt(2/(omega_e mu0 mu_r sigma)) [Jackson], same algebra."""
    inp, res, c, _ = case("defaults")
    st = inp.materials.steel
    want = ref.skin_depth_m(c.omega_e, st.mu_r_incremental, st.conductivity_S_m, inp.coupling.mu0) * 1000
    eng = res.temperature.slip_loss.skin_depth_mm
    assert rel_err(eng, want) <= TOL_ALGEBRA, mismatch("steel skin depth", "Temperature design!C115", eng, want, TOL_ALGEBRA)


@pytest.mark.family("thermal")
@pytest.mark.parametrize("part", list(PARTS))
def test_slip_loss_rederivation(part):
    """Each slip-loss row equals its closed form re-derived from first principles: travelling field over a permeable
    conductor in the thin-skin limit (hub, cup, web), long thin sheet at low speed times the end factor (sleeve,
    liner, cap), thin strip (magnets). Same algebra: TOL_ALGEBRA."""
    inp, res, _, _ = case("defaults")
    field, cell = PARTS[part]
    eng = getattr(res.temperature.slip_loss, field)
    want = rederived_losses(inp, res)[part]
    assert rel_err(eng, want) <= TOL_ALGEBRA, mismatch(f"{part} slip loss, re-derived closed form", cell, eng, want, TOL_ALGEBRA)


@pytest.mark.family("thermal")
def test_steel_surface_losses_vs_exact_halfspace():
    """Hub, cup and web losses and their speed exponents against Stoll's exact half-space for the same surface field
    (TOL_MODEL). One root cause: the thin-skin form (loss ~ speed^1.5) needs k*delta << 1, but k*delta is 0.64 (hub),
    0.36 (cup) and 0.55 (web) at defaults, and the finite-permeability reflection |1 + gamma/(mu_r k)|^-2 is left out.
    The web's missing steel image is a separate root cause (test_web_end_field_doubled_at_steel_web)."""
    assert_all(model_items(("hub", "cup", "web"), lambda i, r: steel_losses(i, r, exact=True, web_image=1.0)))


@pytest.mark.family("thermal")
def test_web_end_field_doubled_at_steel_web():
    """Web loss (C125) against the same thin-skin model with the end field doubled at the steel web (TOL_MODEL).

    Why doubled: at a highly permeable surface the normal field is twice the incident field (image method), and the
    engine doubles hub and cup for that reason: fields3d.run sets b_hub = 2 * _fundamental_br(g, g.outer(0), g.hub_surf)
    and b_cup = 2 * _fundamental_br(g, g.inner(0), g.cup_surf), and the b_hub_T help text reads "3D, doubled at the steel
    surface" (Temperature design!C116). The web integral is web_int = disk(9.0, 18.0, g.L / 2 + web_gap_mm, True), the
    fundamental of Bz from g.inner(0) alone (the inner ring and its hub images) at the web plane, with no factor 2 and
    no mirror across that plane; its help text reads only "3D." (C121). fields3d.run at defaults returns 1.0346e-5,
    the default input 1.035e-5, so the input is this free-space value. temperature.py then applies the same per-area
    steel formula to the web as to hub and cup. The web is steel in the engine (steel sigma and mu_r in C125).
    Consistent treatment multiplies the web loss by 4; with the exact half-space as well it is 0.332 W."""
    inp, res, _, _ = case("defaults")
    want = steel_losses(inp, res, exact=False, web_image=2.0)["web"]
    eng = res.temperature.slip_loss.web_W
    assert rel_err(eng, want) <= TOL_MODEL, mismatch("web loss with the end field doubled at the steel web",
                                                     "Temperature design!C125, C121", eng, want, TOL_MODEL)


@pytest.mark.family("thermal")
def test_shell_losses_vs_russell_norsworthy():
    """Sleeve and liner losses and their speed exponents against a finite thin shell (TOL_MODEL): Russell-Norsworthy end
    factor extended to the retainer overhang, times Reitz's sheet reaction. One root cause: the end-factor input 0.7
    (Temperature design!C114) shared by both shells."""
    assert_all(model_items(("sleeve", "liner"), literature_losses, extra_cells=", C114"))


@pytest.mark.family("thermal")
def test_cap_loss_vs_sheet_reaction():
    """Cap loss and its speed exponent with Reitz's sheet reaction at the cap outer radius (TOL_MODEL); the cap end
    factor itself is not independently checkable (module docstring)."""
    assert_all(model_items(("cap",), literature_losses))


@pytest.mark.family("thermal")
def test_magnet_loss_vs_rectangular_section():
    """Magnet eddy loss and its speed exponent against the exact low-frequency loss of a rectangular prism (Prandtl
    torsion analogy, Timoshenko-Goodier) instead of the thin-strip formula on the 2:1 block face (TOL_MODEL)."""
    assert_all(model_items(("magnets",), literature_losses))


@pytest.mark.family("thermal")
def test_total_slip_loss_vs_literature():
    """Total estimated slip loss against the sum of the literature part losses (TOL_MODEL)."""
    inp, res, _, _ = case("defaults")
    eng = res.temperature.slip_loss.total_W
    want = sum(literature_losses(inp, res).values())
    assert rel_err(eng, want) <= TOL_MODEL, mismatch("total slip loss vs literature", "Temperature design!C130", eng, want, TOL_MODEL)


# ============================================================ thermal network vs numerical and exact references
@pytest.mark.family("thermal")
def test_consistency_heat_capacity_mass_closure():
    """Internal consistency, not an independent check (both sides are engine rows): the parts in the lumped heat
    capacity (C141) add up to the modelled rotating mass (Calculator!C114, which Task 4 checks independently), so
    every rotating part is counted once."""
    inp, res, _, _ = case("defaults")
    m, ret = res.mass, res.retainers
    parts = m.magnets_g + m.cup_g + m.hub_g + m.boss_g + inp.metal.hardware_g + ret.retainers_g + ret.endplates_g + ret.cap_g
    eng = m.total_g
    assert rel_err(eng, parts) <= TOL_ALGEBRA, mismatch("rotating mass vs parts in the heat capacity",
                                                        "Calculator!C114 / Temperature design!C141", eng, parts, TOL_ALGEBRA)


@pytest.mark.family("thermal")
def test_time_to_limit_vs_ode():
    """Unbroken-slip time to the limit, estimate and high case, against numerical integration of
    C dT/dt = P - G (T - T0) with an event at the limit. Conductance 0.08 W/K so both cases reach it. TOL_NUMERIC."""
    inp, res, _, h = case("low_conductance")
    th = res.temperature.thermal
    assert_all([
        (f"time to limit ({which}) vs ODE", cell, as_seconds(eng),
         ref.ode_time_to_rise(h.rise_limit, p, h.conductance, h.capacity, 50 * h.tau), TOL_NUMERIC)
        for which, cell, eng, p in (("estimate", "Temperature design!C152", th.time_to_limit_est, h.p_use),
                                    ("high", "Temperature design!C150", th.time_to_limit_high, h.p_high))])


@pytest.mark.family("thermal")
def test_time_to_limit_never_branch():
    """At defaults the steady rise stays below the limit, so the limit is never reached: infinite time, which the
    engine reports as text (C150, C151, C152 and the summary copy C19)."""
    inp, res, c, h = case("defaults")
    th, s = res.temperature.thermal, res.temperature.summary
    assert_all([never_item("time to limit, high", "Temperature design!C150", th.time_to_limit_high, h.time_to_limit_high),
                never_item("rotations to limit, high", "Temperature design!C151", th.rotations_to_limit_high,
                           h.time_to_limit_high * c.rev_s),
                never_item("time to limit, estimate", "Temperature design!C152", th.time_to_limit_est, h.time_to_limit_est),
                never_item("summary time to limit", "Temperature design!C19", s.time_to_limit_high, h.time_to_limit_high)])


@pytest.mark.family("thermal")
def test_temperature_at_fault_trip_vs_ode():
    """Magnet temperature at the fault trip time (high case) against numerical integration. TOL_NUMERIC."""
    inp, res, _, h = case("defaults")
    want = h.t0 + ref.ode_rise(h.p_high, h.conductance, h.capacity, inp.temperature.duty.fault_trip_s)
    eng = res.temperature.thermal.temp_at_fault_C
    assert rel_err(eng, want) <= TOL_NUMERIC, mismatch("temperature at fault trip vs ODE", "Temperature design!C161", eng, want, TOL_NUMERIC)


@pytest.mark.family("thermal")
def test_rise_per_event_vs_first_order_response():
    """The per-event rise is adiabatic (P t / C); against the exact first-order step response the error is t/(2 tau)
    = 1.8e-4 at defaults. TOL_MODEL."""
    inp, res, _, h = case("defaults")
    t = inp.metal.slip_event_s
    est = ref.first_order_rise(h.p_use, h.conductance, h.capacity, t)
    high = ref.first_order_rise(h.p_high, h.conductance, h.capacity, t)
    assert_all([("rise per event (estimate)", "Temperature design!C145", res.temperature.thermal.rise_per_event_C, est, TOL_MODEL),
                ("rise per event (estimate)", "Temperature design!C171", res.temperature.slip_life.rise_per_event_est_C, est, TOL_MODEL),
                ("rise per event (high)", "Temperature design!C172", res.temperature.slip_life.rise_per_event_high_C, high, TOL_MODEL)])


@pytest.mark.family("thermal")
def test_average_slip_rise_vs_event_train():
    """Average slip heating over life (duty x steady rise) against the periodic steady state of an explicit train of
    slip events (one event every life_hours x 3600 / events seconds). TOL_NUMERIC."""
    inp, res, _, h = case("defaults")
    period = inp.temperature.duty.life_hours * 3600 / inp.metal.life_events
    t_on = inp.metal.slip_event_s
    sl = res.temperature.slip_life
    assert_all([
        (f"average slip rise ({which}) vs event train", cell, eng,
         ref.event_train_mean_rise(p, h.conductance, h.capacity, t_on, period), TOL_NUMERIC)
        for which, cell, eng, p in (("estimate", "Temperature design!C175", sl.avg_rise_est_C, h.p_use),
                                    ("high", "Temperature design!C176", sl.avg_rise_high_C, h.p_high))])


@pytest.mark.family("thermal")
def test_time_to_95_percent_label():
    """'Time / rotations to 95 % of the steady rise' use 3 tau; the fraction reached there must round to 95 %
    (TOL_LABEL). The exact time is tau ln 20 = 2.996 tau."""
    inp, res, c, h = case("defaults")
    th = res.temperature.thermal
    assert_all([(f"share of the steady rise reached at {what}", cell, 1 - math.exp(-t / h.tau), 0.95, TOL_LABEL)
                for what, cell, t in (("t95_s", "Temperature design!C159", th.t95_s),
                                      ("rev95", "Temperature design!C160", th.rev95 / c.rev_s))])


@pytest.mark.family("thermal")
def test_critical_drag_closure():
    """Critical drag is the drag whose steady state equals the limit: entered as the bench drag it must give a steady
    high-case temperature equal to the governing limit (Temperature design!C149 vs C12)."""
    inp, res, _, h = case("defaults")
    res_d = run(vary(inp, {"metal.measured_drag_Nm": res.temperature.thermal.critical_drag_Nm}))
    eng = res_d.temperature.thermal.steady_high_C
    assert rel_err(eng, h.t_limit) <= TOL_ALGEBRA, mismatch("steady temperature at the critical drag", "Temperature design!C149, C153",
                                                          eng, h.t_limit, TOL_ALGEBRA)


# ============================================================ edge cases
@pytest.mark.family("thermal")
def test_start_above_limit_gives_zero_time_and_drag():
    """Edge case, one root cause (no guard for a hot-day start above the governing limit): driving rise 40 C puts the
    start (95 C) above the 92.55 C limit. The limit is reached at t = 0 (0 s, 0 rev), and any drag exceeds it, so the
    critical drag is 0 N·m. Negative times and a negative drag (slip that cools) are unphysical."""
    inp, res, c, h = case("start_above_limit")
    th, s = res.temperature.thermal, res.temperature.summary
    assert_all([
        ("time to limit, high", "Temperature design!C150", as_seconds(th.time_to_limit_high), h.time_to_limit_high, TOL_ALGEBRA),
        ("rotations to limit, high", "Temperature design!C151", as_seconds(th.rotations_to_limit_high),
         h.time_to_limit_high * c.rev_s, TOL_ALGEBRA),
        ("time to limit, estimate", "Temperature design!C152", as_seconds(th.time_to_limit_est), h.time_to_limit_est, TOL_ALGEBRA),
        ("summary time to limit", "Temperature design!C19", as_seconds(s.time_to_limit_high), h.time_to_limit_high, TOL_ALGEBRA),
        ("critical drag", "Temperature design!C153", th.critical_drag_Nm, h.critical_drag, TOL_ALGEBRA),
        ("summary critical drag", "Temperature design!C23", s.critical_drag_Nm, h.critical_drag, TOL_ALGEBRA)])


@pytest.mark.family("thermal")
def test_zero_measured_drag_runs():
    """Edge case: a bench drag of 0 N·m means no slip heating; rotations per degree are unbounded. The engine must
    return a value (inf) rather than raise."""
    try:
        eng = run(vary(defaults(), {"metal.measured_drag_Nm": 0.0})).temperature.thermal.rev_per_C_est
    except ZeroDivisionError:
        eng = math.nan
    assert eng == math.inf, mismatch("rotations per degree at zero measured drag", "Temperature design!C156", eng, math.inf, 0.0)


# ============================================================ row re-derivations (same algebra)
def _row(what, family, engines, reference, case_name="defaults"):
    """engines: [(cell, getter), ...]; a section row and its copies (summary, link cells) share one reference."""
    return pytest.param(what, engines, reference, case_name, marks=pytest.mark.family(family), id=what)


ROWS = [
    # links to inputs
    _row("service_max", "temperature", [("Temperature design!C6", lambda r: r.temperature.summary.service_max_C)],
         lambda i, r, c, h: i.coupling.op_temp_C),
    _row("slip_rpm", "thermal", [("Temperature design!C28", lambda r: r.temperature.duty.slip_rpm)], lambda i, r, c, h: i.metal.slip_rpm),
    _row("slip_event_s", "thermal", [("Temperature design!C33", lambda r: r.temperature.duty.slip_event_s)],
         lambda i, r, c, h: i.metal.slip_event_s),
    _row("steel_conductivity", "thermal", [("Temperature design!C109", lambda r: r.temperature.slip_loss.steel_sigma_S_m)],
         lambda i, r, c, h: i.materials.steel.conductivity_S_m),
    _row("steel_mu_r", "thermal", [("Temperature design!C110", lambda r: r.temperature.slip_loss.steel_mu_r)],
         lambda i, r, c, h: i.materials.steel.mu_r_incremental),
    _row("cap_conductivity", "thermal", [("Temperature design!C112", lambda r: r.temperature.slip_loss.cap_sigma_S_m)],
         lambda i, r, c, h: i.materials.aluminium.al6061.conductivity_S_m),
    _row("steel_specific_heat", "thermal", [("Temperature design!C137", lambda r: r.temperature.thermal.steel_c)],
         lambda i, r, c, h: i.materials.steel.specific_heat_J_kgK),
    # duty and kinematics
    _row("slip_rad_s", "thermal", [("Temperature design!C29", lambda r: r.temperature.duty.slip_rad_s)], lambda i, r, c, h: c.omega),
    _row("pole_pairs", "thermal", [("Temperature design!C30", lambda r: r.temperature.duty.pole_pairs)], lambda i, r, c, h: c.pp),
    _row("field_freq_Hz", "thermal", [("Temperature design!C31", lambda r: r.temperature.duty.field_freq_Hz)],
         lambda i, r, c, h: pole_pair_frequency_Hz(i.coupling.npole, i.metal.slip_rpm)),
    _row("field_omega", "thermal", [("Temperature design!C32", lambda r: r.temperature.duty.field_omega_rad_s)], lambda i, r, c, h: c.omega_e),
    _row("hot_day_start", "thermal", [("Temperature design!C37", lambda r: r.temperature.duty.hot_day_start_C),
                                      ("Temperature design!C14", lambda r: r.temperature.summary.hot_day_start_C),
                                      ("Temperature design!C144", lambda r: r.temperature.thermal.start_C)], lambda i, r, c, h: h.t0),
    # the temperature margins C13 and C15 are Task 5's (test_summary_margins_and_hot_day_note)
    # slip-loss bookkeeping
    _row("total_W", "thermal", [("Temperature design!C130", lambda r: r.temperature.slip_loss.total_W)], lambda i, r, c, h: h.p_estimate),
    _row("drag_from_total", "thermal", [("Temperature design!C131", lambda r: r.temperature.slip_loss.drag_Nm)],
         lambda i, r, c, h: h.p_estimate / c.omega),
    _row("used_W_estimate", "thermal", [("Temperature design!C132", lambda r: r.temperature.slip_loss.used_W)], lambda i, r, c, h: h.p_use),
    _row("high_W_estimate", "thermal", [("Temperature design!C134", lambda r: r.temperature.slip_loss.high_W)], lambda i, r, c, h: h.p_high),
    _row("used_and_high_W_measured", "thermal", [("Temperature design!C132", lambda r: r.temperature.slip_loss.used_W),
                                                 ("Temperature design!C134", lambda r: r.temperature.slip_loss.high_W)],
         lambda i, r, c, h: shaft_power_W(0.02, i.metal.slip_rpm), "measured_drag"),
    # thermal network
    _row("heat_capacity", "thermal", [("Temperature design!C141", lambda r: r.temperature.thermal.heat_capacity_J_K)],
         lambda i, r, c, h: h.capacity),
    _row("time_constant", "thermal", [("Temperature design!C143", lambda r: r.temperature.thermal.time_constant_s)], lambda i, r, c, h: h.tau),
    _row("steady_rise_est", "thermal", [("Temperature design!C146", lambda r: r.temperature.thermal.steady_rise_est_C)],
         lambda i, r, c, h: h.p_use / h.conductance),
    _row("steady_rise_high", "thermal", [("Temperature design!C147", lambda r: r.temperature.thermal.steady_rise_high_C)],
         lambda i, r, c, h: h.p_high / h.conductance),
    _row("steady_est", "thermal", [("Temperature design!C148", lambda r: r.temperature.thermal.steady_est_C),
                                   ("Temperature design!C17", lambda r: r.temperature.summary.steady_estimate_C)],
         lambda i, r, c, h: h.steady_est),
    _row("steady_high", "thermal", [("Temperature design!C149", lambda r: r.temperature.thermal.steady_high_C),
                                    ("Temperature design!C18", lambda r: r.temperature.summary.steady_high_C)],
         lambda i, r, c, h: h.steady_high),
    _row("time_to_limit_high", "thermal", [("Temperature design!C150", lambda r: as_seconds(r.temperature.thermal.time_to_limit_high)),
                                           ("Temperature design!C19", lambda r: as_seconds(r.temperature.summary.time_to_limit_high))],
         lambda i, r, c, h: h.time_to_limit_high, "low_conductance"),
    _row("rotations_to_limit_high", "thermal",
         [("Temperature design!C151", lambda r: as_seconds(r.temperature.thermal.rotations_to_limit_high))],
         lambda i, r, c, h: h.time_to_limit_high * c.rev_s, "low_conductance"),
    _row("time_to_limit_est", "thermal", [("Temperature design!C152", lambda r: as_seconds(r.temperature.thermal.time_to_limit_est))],
         lambda i, r, c, h: h.time_to_limit_est, "low_conductance"),
    _row("critical_drag", "thermal", [("Temperature design!C153", lambda r: r.temperature.thermal.critical_drag_Nm),
                                      ("Temperature design!C23", lambda r: r.temperature.summary.critical_drag_Nm)],
         lambda i, r, c, h: h.critical_drag),
    _row("heating_rate_est", "thermal", [("Temperature design!C154", lambda r: r.temperature.thermal.heating_rate_est_C_s)],
         lambda i, r, c, h: h.p_use / h.capacity),
    _row("heating_rate_high", "thermal", [("Temperature design!C155", lambda r: r.temperature.thermal.heating_rate_high_C_s)],
         lambda i, r, c, h: h.p_high / h.capacity),
    _row("rev_per_C_est", "thermal", [("Temperature design!C156", lambda r: r.temperature.thermal.rev_per_C_est)],
         lambda i, r, c, h: c.rev_s * h.capacity / h.p_use),
    _row("rev_per_C_high", "thermal", [("Temperature design!C157", lambda r: r.temperature.thermal.rev_per_C_high)],
         lambda i, r, c, h: c.rev_s * h.capacity / h.p_high),
    _row("rev_per_tau", "thermal", [("Temperature design!C158", lambda r: r.temperature.thermal.rev_per_tau)],
         lambda i, r, c, h: h.tau * c.rev_s),
    _row("temp_at_fault", "thermal", [("Temperature design!C161", lambda r: r.temperature.thermal.temp_at_fault_C)],
         lambda i, r, c, h: h.t0 + h.fault_rise),
    # slip life
    _row("life_events", "temperature", [("Temperature design!C164", lambda r: r.temperature.slip_life.events)],
         lambda i, r, c, h: i.metal.life_events),
    _row("rev_per_event", "temperature", [("Temperature design!C165", lambda r: r.temperature.slip_life.rev_per_event)],
         lambda i, r, c, h: c.rev_s * i.metal.slip_event_s),
    _row("life_rotations", "temperature", [("Temperature design!C166", lambda r: r.temperature.slip_life.rotations),
                                           ("Temperature design!C21", lambda r: r.temperature.summary.life_rotations)],
         lambda i, r, c, h: h.rotations),
    _row("slip_hours", "temperature", [("Temperature design!C167", lambda r: r.temperature.slip_life.slip_hours)],
         lambda i, r, c, h: i.metal.life_events * i.metal.slip_event_s / 3600),
    # one full field and shear reversal per pole-pair pass
    _row("pole_pair_passes", "temperature", [("Temperature design!C168", lambda r: r.temperature.slip_life.like_pole_passes),
                                             ("Temperature design!C191", lambda r: r.temperature.adhesive_life.reversals)],
         lambda i, r, c, h: h.rotations * c.pp),
    _row("heat_per_event_est", "temperature", [("Temperature design!C169", lambda r: r.temperature.slip_life.heat_per_event_est_J)],
         lambda i, r, c, h: h.p_use * i.metal.slip_event_s),
    _row("heat_per_event_high", "temperature", [("Temperature design!C170", lambda r: r.temperature.slip_life.heat_per_event_high_J)],
         lambda i, r, c, h: h.p_high * i.metal.slip_event_s),
    _row("life_heat_high_MJ", "temperature", [("Temperature design!C173", lambda r: r.temperature.slip_life.life_heat_high_MJ)],
         lambda i, r, c, h: i.metal.life_events * h.p_high * i.metal.slip_event_s / 1e6),
    _row("slip_duty", "temperature", [("Temperature design!C174", lambda r: r.temperature.slip_life.slip_duty)], lambda i, r, c, h: h.duty),
    _row("rise_per_pct_duty", "temperature", [("Temperature design!C177", lambda r: r.temperature.slip_life.rise_per_pct_duty_C)],
         lambda i, r, c, h: 0.01 * h.p_high / h.conductance),
    _row("summary_avg_slip_heating", "temperature",
         [("Temperature design!C22", lambda r: r.temperature.summary.avg_slip_heating_high_C)], lambda i, r, c, h: h.avg_rise_high),
    # magnet life
    _row("peak", "temperature", [("Temperature design!C180", lambda r: r.temperature.magnet_life.peak_C),
                                 ("Temperature design!C20", lambda r: r.temperature.summary.peak_with_fault_C),
                                 ("Temperature design!C189", lambda r: r.temperature.adhesive_life.peak_C)], lambda i, r, c, h: h.peak),
    _row("margin_to_skipping_onset", "temperature", [("Temperature design!C181", lambda r: r.temperature.magnet_life.margin_onset_C)],
         lambda i, r, c, h: r.temperature.demag.onset_skipping_C - h.peak),
    _row("margin_to_magnet_limit", "temperature", [("Temperature design!C182", lambda r: r.temperature.magnet_life.margin_limit_C)],
         lambda i, r, c, h: r.temperature.demag.magnet_limit_C - h.peak),
    _row("torque_hot_day", "temperature", [("Temperature design!C184", lambda r: r.temperature.magnet_life.torque_hot_day_Nm),
                                           ("Temperature design!C16", lambda r: r.temperature.summary.torque_hot_day_Nm)],
         lambda i, r, c, h: pullout_at(i, r, h.t0)),
    _row("torque_at_peak", "temperature", [("Temperature design!C186", lambda r: r.temperature.magnet_life.torque_peak_Nm)],
         lambda i, r, c, h: pullout_at(i, r, h.peak)),
    # adhesive life
    _row("bond_margin", "temperature", [("Temperature design!C190", lambda r: r.temperature.adhesive_life.margin_C)],
         lambda i, r, c, h: selected_adhesive(i).design_limit_C - h.peak),
    _row("torque_peak_with_variation", "temperature",
         [("Temperature design!C192", lambda r: r.temperature.adhesive_life.torque_peak_var_Nm)],
         lambda i, r, c, h: pullout_at(i, r, h.peak) * (1 + i.metal.variation)),
    _row("shear_amplitude", "temperature", [("Temperature design!C193", lambda r: r.temperature.adhesive_life.shear_amplitude_MPa)],
         lambda i, r, c, h: shear_amplitude(i, r, h)),
    _row("hot_fatigue_margin", "temperature", [("Temperature design!C196", lambda r: r.temperature.adhesive_life.hot_fatigue_margin)],
         lambda i, r, c, h: hot_fatigue_margin(i, r, h)),
    _row("daily_cycles", "temperature", [("Temperature design!C200", lambda r: r.temperature.adhesive_life.daily_cycles)],
         lambda i, r, c, h: i.temperature.adhesive_life.service_years * 365),
    _row("daily_peak_shear", "temperature", [("Temperature design!C201", lambda r: r.temperature.adhesive_life.daily_peak_shear_MPa)],
         lambda i, r, c, h: daily_peak_shear(i, r)),
]


@pytest.mark.parametrize("what,engines,reference,case_name", ROWS)
def test_row_rederivation(what, engines, reference, case_name):
    """Engine row (and its summary or link copies) equals the closed form re-derived here from first principles
    (first-order RC network, P = T Omega, life totals from events x duration x speed, torque ~ Br^2 with Br linear in
    T). TOL_ALGEBRA."""
    inp, res, c, h = case(case_name)
    want = reference(inp, res, c, h)
    assert_all([(what, cell, getter(res), want, TOL_ALGEBRA) for cell, getter in engines])


# ============================================================ screens and verdict (text rows)
def _torque_meets_requirement(inp, res, c, h) -> bool:
    return pullout_at(inp, res, h.t0) >= inp.metal.required_min_Nm


def _hot_fatigue_ok(inp, res, c, h) -> bool:
    return hot_fatigue_margin(inp, res, h) >= 4


def _daily_shear_below_endurance(inp, res, c, h) -> bool:
    endurance = selected_adhesive(inp).lap_shear_MPa * inp.temperature.adhesive_life.fatigue_endurance
    return daily_peak_shear(inp, res) < endurance


def _temperature_ok(inp, res, c, h) -> bool:
    """Positive hot-day margin, magnet and bond margins at the peak, and cure margin of at least 10 C (C24, Task 5)."""
    return (h.rise_limit > 0 and res.temperature.demag.magnet_limit_C - h.peak > 0
            and selected_adhesive(inp).design_limit_C - h.peak > 0 and res.temperature.summary.cure_margin_C >= 10)


SCREENS = {  # rule: ([(cell, engine text getter), ...], passing text, other text, re-derived pass criterion)
    # C185 only: its summary copy, the hot-day note F16, is Task 5's (test_summary_margins_and_hot_day_note)
    "hot_day_torque": ([("Temperature design!C185", lambda r: r.temperature.magnet_life.torque_hot_day_check)],
                       "Meets it nominally (no variation allowance)", "Below it", _torque_meets_requirement),
    "hot_fatigue": ([("Temperature design!C197", lambda r: r.temperature.adhesive_life.hot_fatigue_screen)],
                    "OK", "CHECK: get hot fatigue data", _hot_fatigue_ok),
    "daily_cycle": ([("Temperature design!C202", lambda r: r.temperature.adhesive_life.daily_screen)],
                    "Below the fatigue endurance", "Above the fatigue endurance: qualify by thermal cycling",
                    _daily_shear_below_endurance),
    "verdict": ([("Temperature design!C25", lambda r: r.temperature.summary.verdict)],
                "OK on temperature. Confirm drag torque and thermal cycling by test.", "CHECK: see the rows above.",
                _temperature_ok),
}
SCREEN_CASES = [("hot_day_torque", "defaults"), ("hot_day_torque", "high_requirement"), ("hot_fatigue", "defaults"),
                ("hot_fatigue", "low_hot_strength"), ("daily_cycle", "defaults"), ("daily_cycle", "small_daily_swing"),
                ("verdict", "defaults"), ("verdict", "start_above_limit")]


@pytest.mark.family("temperature")
@pytest.mark.parametrize("rule,case_name", SCREEN_CASES, ids=[f"{rule}-{name}" for rule, name in SCREEN_CASES])
def test_screen_text(rule, case_name):
    """Life screens, the hot-day torque check (C185) and the temperature verdict pass exactly when the re-derived
    quantities meet the stated criterion; each rule is exercised on both branches."""
    engines, pass_text, other_text, passes = SCREENS[rule]
    inp, res, c, h = case(case_name)
    expected = pass_text if passes(inp, res, c, h) else other_text
    assert_all([text_item(f"{rule} ({case_name})", cell, getter(res), expected) for cell, getter in engines])
```

`reference/magcoupling-py/audit/tools/__init__.py` is an empty file. It makes `audit.tools` an explicit package, so the tool below runs with `-m`.

File: `reference/magcoupling-py/audit/tools/__init__.py` (create)
```python
```

`audit/tools/placeholder_sensitivity.py` produces report data, not checks:

File: `reference/magcoupling-py/audit/tools/placeholder_sensitivity.py` (create)
```python
"""Placeholder-input sensitivities at the default design, for the findings report's 'Placeholder inputs' table.

Report data, not a check: nothing here passes or fails. Task 6 re-derives every engine row listed below exactly
(audit/tests/test_slip_thermal.py), so comparing derivatives would add no check; what the report needs is how far
each placeholder moves a result. Each derivative is a central difference on the engine
(audit.references.slip_thermal.central_difference, relative step 1e-4, sanity-tested), and the last column is the
linearized change of the result for a +10 % change of the input.

Run from reference/magcoupling-py:  ./.venv/Scripts/python -m audit.tools.placeholder_sensitivity
"""
from __future__ import annotations

from audit.common import defaults, run, vary
from audit.references.slip_thermal import central_difference

STEADY_HIGH = ("steady magnet temperature, high case (C)", "Temperature design!C149", lambda r: r.temperature.thermal.steady_high_C)
PEAK = ("hot-day peak magnet temperature (C)", "Temperature design!C180", lambda r: r.temperature.magnet_life.peak_C)
CRITICAL_DRAG = ("critical drag (N m)", "Temperature design!C153", lambda r: r.temperature.thermal.critical_drag_Nm)
SLIP_DUTY = ("slip duty (-)", "Temperature design!C174", lambda r: r.temperature.slip_life.slip_duty)

PLACEHOLDERS = [  # (input path, why it is a placeholder, [(result, cell, getter), ...])
    ("temperature.thermal.conductance_W_K", "conductance to ambient not measured", [
        STEADY_HIGH,
        ("thermal time constant (s)", "Temperature design!C143", lambda r: r.temperature.thermal.time_constant_s),
        CRITICAL_DRAG, PEAK]),
    ("temperature.duty.driving_rise_C", "housing air, sun and gearbox heat not measured", [
        PEAK, CRITICAL_DRAG,
        ("pull-out at the hot-day start (N m)", "Temperature design!C184", lambda r: r.temperature.magnet_life.torque_hot_day_Nm),
        ("margin above the hot-day start (C)", "Temperature design!C15", lambda r: r.temperature.summary.margin_hot_day_C)]),
    ("metal.slip_event_s", "slip event length assumed", [
        ("life slip rotations (rev)", "Temperature design!C166", lambda r: r.temperature.slip_life.rotations),
        SLIP_DUTY, PEAK]),
    ("temperature.duty.life_hours", "operating hours over life assumed", [SLIP_DUTY, PEAK]),
    ("temperature.adhesive_life.hot_strength_retained", "hot adhesive strength not measured", [
        ("hot fatigue margin (-)", "Temperature design!C196", lambda r: r.temperature.adhesive_life.hot_fatigue_margin)]),
    ("temperature.slip_loss.high_multiplier", "judgement band on the slip-loss estimate", [STEADY_HIGH, PEAK]),
    ("temperature.slip_loss.end_factor", "judgement end factor for the shells and the cap", [
        ("total slip loss, estimate (W)", "Temperature design!C130", lambda r: r.temperature.slip_loss.total_W), STEADY_HIGH]),
]


def input_value(inp, path: str) -> float:
    """Value of a dotted-path input, e.g. 'temperature.duty.driving_rise_C'."""
    obj = inp
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def sensitivity(path: str, getter, rel_step: float = 1e-4) -> float:
    """d(result)/d(input) at the default design, by a central difference on the engine."""
    return central_difference(lambda x: getter(run(vary(defaults(), {path: x}))), input_value(defaults(), path), rel_step)


def main() -> None:
    base = run()
    print("| Input | Default | Why a placeholder | Result | Cell | Value at defaults | d result / d input | Change for +10 % input |")
    print("|---|---|---|---|---|---|---|---|")
    for path, why, results in PLACEHOLDERS:
        x0 = input_value(defaults(), path)
        for label, cell, getter in results:
            d = sensitivity(path, getter)
            print(f"| `{path}` | {x0:g} | {why} | {label} | {cell} | {getter(base):.6g} | {d:.6g} | {0.1 * x0 * d:+.4g} |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the checks**

Run:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_slip_thermal.py
```

Expected: `6 failed, 81 passed`. There is one failing test per root cause. The 6 FAILs are **candidate findings for Task 8, not bugs to fix now**: do not edit `magcoupling/*.py` and do not loosen any tolerance. The model-approximation tolerance is `TOL_MODEL` = 2 % from `audit.common` (Task 1), so the four `TOL_MODEL` FAILs below use `tol=2.0e-02`. Failing multi-cell checks open with `N of M comparisons failed:` (Task 1's `assert_all`). Observed messages, abridged to the failing items:
- `test_steel_surface_losses_vs_exact_halfspace` (6 of 6 comparisons failed, tol 2.0e-02). Candidate finding for Task 8:
  - `hub loss [Temperature design!C123]: engine=0.22590854790913922 reference=0.20037496168764454 rel_err=1.274e-01 tol=2.0e-02`
  - `hub speed exponent 1000-2000 rpm [Temperature design!C123]: engine=1.5000000000000002 reference=1.6369303940996314 rel_err=8.365e-02`
  - `cup loss [Temperature design!C124]: engine=1.3530776316850626 reference=1.272660423120791 rel_err=6.319e-02`
  - `cup speed exponent 1000-2000 rpm [Temperature design!C124]: engine=1.5000000000000002 reference=1.5359878999001724 rel_err=2.343e-02`
  - `web loss [Temperature design!C125]: engine=0.09140096902640166 reference=0.08305007996759096 rel_err=1.006e-01`
  - `web speed exponent 1000-2000 rpm [Temperature design!C125]: engine=1.5000000000000002 reference=1.6009780172375878 rel_err=6.307e-02`
- `test_web_end_field_doubled_at_steel_web`. Candidate finding for Task 8: `web loss with the end field doubled at the steel web [Temperature design!C125, C121]: engine=0.09140096902640166 reference=0.3656038761056067 rel_err=7.500e-01 tol=2.0e-02`
- `test_shell_losses_vs_russell_norsworthy` (2 of 4 comparisons failed, tol 2.0e-02). Candidate finding for Task 8:
  - `sleeve loss [Temperature design!C126, C114]: engine=0.07469961538641434 reference=0.07206950797751382 rel_err=3.649e-02 tol=2.0e-02`
  - `liner loss [Temperature design!C127, C114]: engine=0.1805398049833353 reference=0.16843815828706182 rel_err=7.185e-02 tol=2.0e-02`
  - Both exponents pass (2 against 2.0000).
- `test_magnet_loss_vs_rectangular_section` (1 of 2 comparisons failed, tol 2.0e-02). Candidate finding for Task 8: `magnets loss [Temperature design!C129]: engine=0.22784639081300484 reference=0.1563128843302991 rel_err=4.576e-01 tol=2.0e-02`. The exponent passes (2 against 2).
- `test_start_above_limit_gives_zero_time_and_drag` (driving rise 40 C; 6 of 6 comparisons failed, tol 1.0e-09). Candidate finding for Task 8; all against reference 0:
  - C150: -25.840 s
  - C151: -861.33 rev
  - C152: -71.189 s
  - C19: -25.840 s
  - C153: -0.0035093 N·m
  - C23: -0.0035093 N·m
- `test_zero_measured_drag_runs`. Candidate finding for Task 8: `rotations per degree at zero measured drag [Temperature design!C156]: engine=nan reference=inf rel_err=nan tol=0.0e+00`. `compute_all` raises ZeroDivisionError at C156 and C157 when `metal.measured_drag_Nm = 0`.

PASS (81), with observed numbers:
- Skin depth: 1.2994947 mm, exact.
- All 7 part losses re-derived to at most 3.7e-16.
- Cap: 0.32364 against 0.32261 W (3.2e-3), exponent 2 against 1.9966.
- Total: 2.4771 against 2.5247 W (1.9e-2).
- Heat-capacity mass closure: 173.7933 g. This is labelled internal consistency.
- Time to limit at G = 0.08 W/K against the ODE: 2265.4733 s (2e-12) and 361.4548 s (3e-14).
- Never-branch at defaults: all 4 cells give the 'never' text.
- Temperature at the fault trip: 65.18017 C.
- Rise per event: within 1.8e-4 of the exact response (3 cells).
- Average rise against the event train: 0.22936 and 0.68809 C (1.5e-15).
- The 3 tau '95 %' label reaches 95.02 %.
- Critical-drag closure: 92.55005 C = limit.
- All 55 row re-derivations pass. The margins C13 and C15 are not among them: with the hot-day note F16 they belong to Task 5's `test_summary_margins_and_hot_day_note`. Examples:
  - C16/C184: 2.54934 N·m
  - C141: 82.194 J/K
  - C143: 273.98 s
  - C149/C18: 89.771 C
  - C153/C23: 0.039463 N·m
  - C166/C21: 6.6667e7 rev
  - C168/C191: 3.3333e8
  - C180/C20/C189: 65.868 C
  - C193: 0.30911 MPa
  - C196: 4.8527
  - C201: 11.3657 MPa
- All 8 screen checks pass: C185, C197, C202 and C25, each on both branches. The hot-day note F16 is Task 5's (`test_summary_margins_and_hot_day_note`), not checked here.

Then print the placeholder table for the report:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m audit.tools.placeholder_sensitivity
```

Expected output: exactly this table, header plus 18 rows.

```text
| Input | Default | Why a placeholder | Result | Cell | Value at defaults | d result / d input | Change for +10 % input |
|---|---|---|---|---|---|---|---|
| `temperature.thermal.conductance_W_K` | 0.3 | conductance to ambient not measured | steady magnet temperature, high case (C) | Temperature design!C149 | 89.7711 | -82.5703 | -2.477 |
| `temperature.thermal.conductance_W_K` | 0.3 | conductance to ambient not measured | thermal time constant (s) | Temperature design!C143 | 273.981 | -913.27 | -27.4 |
| `temperature.thermal.conductance_W_K` | 0.3 | conductance to ambient not measured | critical drag (N m) | Temperature design!C153 | 0.0394625 | 0.131542 | +0.003946 |
| `temperature.thermal.conductance_W_K` | 0.3 | conductance to ambient not measured | hot-day peak magnet temperature (C) | Temperature design!C180 | 65.8683 | -2.29581 | -0.06887 |
| `temperature.duty.driving_rise_C` | 10 | housing air, sun and gearbox heat not measured | hot-day peak magnet temperature (C) | Temperature design!C180 | 65.8683 | 1 | +1 |
| `temperature.duty.driving_rise_C` | 10 | housing air, sun and gearbox heat not measured | critical drag (N m) | Temperature design!C153 | 0.0394625 | -0.00143239 | -0.001432 |
| `temperature.duty.driving_rise_C` | 10 | housing air, sun and gearbox heat not measured | pull-out at the hot-day start (N m) | Temperature design!C184 | 2.54934 | -0.00646766 | -0.006468 |
| `temperature.duty.driving_rise_C` | 10 | housing air, sun and gearbox heat not measured | margin above the hot-day start (C) | Temperature design!C15 | 27.55 | -1 | -1 |
| `metal.slip_event_s` | 0.1 | slip event length assumed | life slip rotations (rev) | Temperature design!C166 | 6.66667e+07 | 6.66667e+08 | +6.667e+06 |
| `metal.slip_event_s` | 0.1 | slip event length assumed | slip duty (-) | Temperature design!C174 | 0.0277778 | 0.277778 | +0.002778 |
| `metal.slip_event_s` | 0.1 | slip event length assumed | hot-day peak magnet temperature (C) | Temperature design!C180 | 65.8683 | 6.88086 | +0.06881 |
| `temperature.duty.life_hours` | 20000 | operating hours over life assumed | slip duty (-) | Temperature design!C174 | 0.0277778 | -1.38889e-06 | -0.002778 |
| `temperature.duty.life_hours` | 20000 | operating hours over life assumed | hot-day peak magnet temperature (C) | Temperature design!C180 | 65.8683 | -3.44043e-05 | -0.06881 |
| `temperature.adhesive_life.hot_strength_retained` | 0.5 | hot adhesive strength not measured | hot fatigue margin (-) | Temperature design!C196 | 4.85271 | 9.70541 | +0.4853 |
| `temperature.slip_loss.high_multiplier` | 3 | judgement band on the slip-loss estimate | steady magnet temperature, high case (C) | Temperature design!C149 | 89.7711 | 8.25703 | +2.477 |
| `temperature.slip_loss.high_multiplier` | 3 | judgement band on the slip-loss estimate | hot-day peak magnet temperature (C) | Temperature design!C180 | 65.8683 | 0.289417 | +0.08683 |
| `temperature.slip_loss.end_factor` | 0.7 | judgement end factor for the shells and the cap | total slip loss, estimate (W) | Temperature design!C130 | 2.47711 | 0.826964 | +0.05789 |
| `temperature.slip_loss.end_factor` | 0.7 | judgement end factor for the shells and the cap | steady magnet temperature, high case (C) | Temperature design!C149 | 89.7711 | 8.26964 | +0.5789 |
```

Finally, run the whole audit directory to confirm there is no interference:

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests -p no:cacheprovider
```

Expected: the Task 6 files add exactly 6 failed and 129 passed (81 + 48). These are the six failures listed above. The totals are the sum of the per-task counts of the tasks committed so far, and no other file's counts change. With Tasks 1-7 integrated, the run gave `47 failed, 710 passed` (757 checks, of which Task 6's two files are 135: 6 failed, 129 passed).

- [ ] **Step 6: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 add reference/magcoupling-py/audit/references/slip_thermal.py reference/magcoupling-py/audit/tests/test_slip_thermal_reference_sanity.py reference/magcoupling-py/audit/tests/test_slip_thermal.py reference/magcoupling-py/audit/tools/__init__.py reference/magcoupling-py/audit/tools/placeholder_sensitivity.py
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 commit -F - <<'EOF'
test(magcoupling-audit): thermal independent checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9
EOF
```

### Task 7: Shaft clamps, sweeps and constants/units: independent checks

**Files:**
- Create: `reference/magcoupling-py/audit/references/clamp_ref.py`
- Create: `reference/magcoupling-py/audit/references/sweep_ref.py`
- Create: `reference/magcoupling-py/audit/references/units_ref.py`
- Create: `reference/magcoupling-py/audit/tests/test_clamps_sweeps_units_reference_sanity.py`
- Create: `reference/magcoupling-py/audit/tests/test_clamps.py`
- Create: `reference/magcoupling-py/audit/tests/test_sweeps.py`
- Create: `reference/magcoupling-py/audit/tests/test_units_verdicts.py`

**Interfaces:**
- Consumes:
  - Task 1 `audit.common` (this task does not edit it): `defaults`, `run`, `vary`, `rel_err`, `mismatch`, `MU0_EXACT`, `TOL_ALGEBRA`, and the shared multi-cell helpers `MAX_REPORTED` = 12, `assert_all(items)` (items are `(what, cells, engine, reference, tol)`; the check fails once and lists every failing comparison, at most `MAX_REPORTED` in full), `flag_item(what, cells, holds) -> tuple`, `text_item(what, cells, engine_text, expected_text) -> tuple`, `ScenarioCache(scenarios)[name] -> (inputs, results)`.
  - `audit.references.planar` (Task 2 creates the module and owns `square_wave_harmonic`; Task 3 appends the other functions): `br_at(br20_T, alpha_per_C, temp_C)`, `square_wave_harmonic(br_T, n, fill)`, `wave_number(n, pole_pitch_m)`, `iron_backed_factor(k, t_i, t_o, g)`, `free_space_factor(k, t_i, t_o, g)`, `harmonic_shear_stress(b_i_n, b_o_n, s_n, k, delta_m, mu0)`, `torque_on_cylinder(tau_Pa, r_m, length_m)`, `end_factor(c_end, pole_pitch_mm, length_mm)`.
  - Task 4 `audit.references.block_geometry`: `coupling_geometry(npole, inner_back_apothem, t_i, w_i, t_o, w_o, face_gap, bond_inner, bond_outer, cup_wall_corner, bore, keyway_depth, faceted)` and the `CouplingGeometry` fields `inner_face_radius, inner_corner_radius, outer_face_apothem, face_gap, corner_gap, cup_od, gap_radius, pole_pitch, fill_inner, fill_outer, inner_flat_width, outer_flat_width, hub_wall_past_key`.
  - Task 4 `audit.references.metal_stack`: `gearbox_input_torque(output_Nm, ratio, efficiency)`.
  - Engine result fields:
    - `res.clamps.table[i].{d_mm, pitch_mm, As_mm2, hole_mm, head_mm, head_h_mm, hex_mm, tap_drill_mm, offset_mm, wall_out_mm, head_fits, grip_mm, thread_avail_mm, engagement_req_mm, geometry_ok, preload_strength_N, preload_strip_N, preload_N, head_pressure_MPa, head_check, torque_per_screw_Nm, screws_needed, pitch_axial_mm, screws_fit, works, clamp_torque_Nm, sf_coupling, tightening_Nm, length_mm, length_ok, cbore_dia_mm, cbore_depth_mm, vent_port_ok}`
    - `res.clamps.{max_torque_Nm, required_Nm, clamp_factor, al_shear_MPa, al_head_limit_MPa, al_key_allow_MPa, screw_proof_MPa, boss_radius_mm, index, recommended, screws, tightening_Nm, hex_mm, capacity_Nm, sf_coupling, head_check, vent_port, key_pressure_MPa, key_sf, joint_preload_N, joint_torque_Nm, joint_sf, layout_offset_mm, layout_pitch_mm, layout_first_mm, layout_cbore_dia_mm, layout_cbore_depth_mm, layout_grip_mm, layout_tap_drill_mm, layout_thread_avail_mm, layout_relief}`
    - `res.gap_sweep[j].*` and `res.pole_sweep[j].*` (every SweepRow field)
    - `res.model.{inner_length_mm, inner_width_mm, inner_thickness_mm, inner_br_T, outer_length_mm, outer_width_mm, outer_thickness_mm, outer_br_T, f_cal, corner_gap_mm, pullout_Nm, required_floor_Nm, verdict, k1, gap_radius_mm, active_length_mm, area_lever_m3, torque_2d_Nm, tau_Pa, f_end, cup_od_mm}`
    - `res.metal.{torque_cold_high_Nm, torque_hot_low_Nm, hot_min_check, min_running_clearance_mm, clearance_check}`
    - `res.temperature.adhesive.block_mass_g`
    - `res.temperature.thermal.{heat_capacity_J_K, time_constant_s, steady_rise_est_C, steady_rise_high_C, rise_per_event_C, critical_drag_Nm}`
  - Engine metadata, used only to label failures and pick the rows to check: `magcoupling.clamps.TABLE_COLUMNS/TABLE_ROWS`, `magcoupling.sweeps.SWEEP_COLUMNS/GAP_SWEEP_CORNER_GAPS_MM/POLE_SWEEP_POLES`. Values under test: `magcoupling.constants.MU0/NDFEB_DENSITY_G_MM3`.
- Produces:
  - `clamp_ref`:
    - Thread geometry: `triangle_height(pitch)`, `pitch_diameter(d, pitch)`, `minor_diameter_external(d, pitch)`, `minor_diameter_internal(d, pitch)`, `stress_area(d, pitch)`, `iso_stress_area(d, pitch)`, `round_sig(x, digits)`, `round_up(x, step)`.
    - Stripping: `internal_tooth_width(x, en_max, pitch)`, `internal_thread_shear_area(ds_min, en_max, pitch, engagement)`, `min_material_shear_area(screw, engagement)`, `stripping_capacity(shear_MPa, area_mm2, safety_factor)`.
    - Loads and fasteners: `preload_proof_share(fraction, proof_MPa, area_mm2)`, `friction_torque_coefficient(pressure_shape, n=20000)`, `clamp_torque_per_screw(mu, preload_N, shaft_mm, clamp_factor)`, `tightening_torque(nut_factor, preload_N, d_mm)`, `head_bearing_pressure(preload_N, head_dk, hole)`, `screws_needed(required_Nm, per_screw_Nm)`, `screws_fit(clamp_length, axial_margin, cbore_dia, spacing)`, `hex_key_passes_port(hex_s, port_d=6.0, port_pitch=1.0)`, `key_bearing_pressure_MPa(torque_Nm, shaft_mm, contact_mm, length_mm)`, `flange_friction_torque_Nm(mu, n_screws, preload_N, bolt_circle_mm)`.
    - Data: `IsoScrew`, `ISO_SCREWS`, `ThreadLimits`, `THREAD_LIMITS_6G_6H`, `PROOF_STRESS_MPA`, `AL_SHEAR_MPA`, `ClampSetup`.
    - Sizing: `clamp_geometry(s, screw) -> dict`, `geometry_ok(s, geo)`, `engagement_achieved(s, geo, length_mm)`, `max_length_inside(s, geo)`, `screw_tip_inside(s, geo, length_mm)`, `valid_screw_lengths(s, geo) -> list[float]`, `size_row(s, screw) -> dict`, `recommend(s) -> (index, rows)`.
  - `sweep_ref`: `SweepSetup`, `required_floor(drive_torque_Nm, drive_safety_factor, required_min_Nm)`, `geometry_at_corner_gap(s, npole, a_i_mm, corner_gap_mm) -> CouplingGeometry`, `sweep_status(s, geo, pullout_op_Nm) -> str`, `sweep_row(s, npole, a_i_mm, corner_gap_mm, f_cal) -> dict` (keys are the SweepRow field names), `stated_min_inner_apothem(npole, w_i_mm, bore_mm, keyway_mm)`, the `STATUS_*` texts, `FLAT_MARGIN_MM`, `KEYED_WALL_MM`.
  - `units_ref`: `Quantity(value, dims)` with `*`, `/`, `**int`, unary `-`, dimension-checked `+`/`-`, `.to(unit)` and `.dimensionless()`; `DimensionError`; units `ONE, M, KG, S, A, K, MM, GRAM, N, NM, PA, MPA, J, W, TESLA, H_PER_M, RAD_PER_S, RPM, J_PER_KG_K, J_PER_K, W_PER_K, A_PER_M, S_PER_M, PER_K, KG_M2`.

Design notes:
- **One check per root cause.** Every engine check loops over all of its scenarios and rows and asserts once with `assert_all`, so one wrong formula gives Task 8 one candidate that lists all of its wrong cells.
- **Stage-wise checks.** A check is fed the engine's own upstream cell wherever its formula takes one. The only end-to-end checks are the recommendation and the README claims.
- **Reuse.** The sweep reference reuses the Task 3 and Task 4 references and adds no formula of its own that they already state.

- [ ] **Step 1: Write the reference modules**

`audit/common.py` is not edited by this task: Task 1 owns it and already provides `assert_all`, `flag_item`, `text_item` and `ScenarioCache` (with `MAX_REPORTED = 12`), which the reference sanity test and the three check files import from `audit.common`.

File: `reference/magcoupling-py/audit/references/clamp_ref.py` (create)
```python
"""Independent reference for the shaft-clamp sizing (engine: clamps.py; sheets 'Shaft clamps', 'Clamp screw sizes').

Written from standards and first principles. Nothing here imports the engine.

Sources
- ISO 68-1 basic metric profile: fundamental triangle height H = (sqrt(3)/2)·P; pitch diameter d2 = d - (3/4)·H;
  external minor diameter d3 = d - (17/12)·H; internal minor diameter D1 = d - (5/4)·H. The internal thread has a
  crest flat of P/4 at D1 and a root gap of P/8 at D.
- ISO 898-1: tensile stress area As = (pi/4)·((d2 + d3)/2)^2, tabulated to 3 significant figures (proof loads are
  built from the tabulated As); proof stress 970 MPa (class 12.9), 830 MPa (class 10.9).
- ISO 3506-1: A4-70 stress at 0.2 % permanent strain, 450 MPa (the usual "proof" stress for stainless screws).
- ISO 965-2 limits of size, medium tolerance class 6g (screw) / 6H (tapped hole), as printed in Bossard, "Metric
  ISO threads" (technical section, 01-2025), p. 97, "Limits for metric (standard) coarse threads according to
  ISO 965".
- ISO 261 coarse pitches, ISO 273 medium clearance holes, ISO 4762 head diameter d_k max, head height k max and hex
  socket size s, DIN 336 / ISO 2306 tap-drill sizes.
- Internal-thread stripping area, FED-STD-H28/2B (also Machinery's Handbook, "Strength of screw threads"):
  A_n = pi·n·Le·Ds,min·[1/(2n) + 0.57735·(Ds,min - En,max)], with n = 1/P, Ds,min the minimum major diameter of
  the external thread and En,max the maximum pitch diameter of the internal thread. It is the internal tooth
  width at the screw's major diameter (P/2 at En,max, widening by tan30° per unit of diameter), times pi·Ds,min·Le/P.
  With the 6g/6H minimum-material limits it is the smallest area an in-tolerance thread pair can have; at basic
  size (Ds = d, En = D2) the tooth width is 7P/8 and A_n = 0.875·pi·d·Le.
- Ultimate shear strength (MMPDS / ASM): 7075-T6 331 MPa, 6061-T6 207 MPa.
- Preload of 75 % of proof load for reusable joints: Shigley, Mechanical Engineering Design, Eq. 8-31.
- Tightening torque T = K·F·d: Shigley Eq. 8-27.
- Clamp friction torque (derived in friction_torque_coefficient): T = C·mu·F·d with
  C = ∫p dθ / ∫p·cosθ dθ over one jaw. C >= 1 for any non-negative bore pressure (cosθ <= 1), with C = 1 for line
  contact, 4/pi for a cosine distribution and pi/2 for uniform pressure (uniform pressure reproduces Shigley's
  press-fit torque (pi/2)·f·p·l·d^2).
- Parallel-key bearing pressure p = 2T/(d·h·L): the force at the shaft surface T/(d/2) over the contact area h·L
  (Shigley, keys and pins).
- Bolted-flange friction torque T = mu·n·F·(D_bc/2).

Clamp geometry (cross-section normal to the shaft axis). The slit plane contains the shaft axis; each tangential
screw axis is normal to the slit plane at offset e from the shaft axis, i.e. a chord of the boss circle (radius R).
Along the screw axis, x = 0 is the middle of the slit, the slit faces are at x = ±slit/2 and the boss OD is at
x = ±sqrt(R^2 - e^2). The flat head seat is a disc of diameter d_k centred on the screw axis in the plane x = x_seat;
the disc lies inside the boss cylinder x^2 + y^2 <= R^2 exactly when x_seat^2 + (e + d_k/2)^2 <= R^2. A screw of
under-head length L seated at x_seat crosses the head-side jaw (grip = x_seat - slit/2) and the open slit, so it
engages L - grip - slit of thread; it stays inside the far jaw while L <= grip + slit + thread available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

SQRT3 = math.sqrt(3.0)
TAN30 = 1.0 / SQRT3
COS30 = SQRT3 / 2.0

#: Float slack for "fits"/"rounds up" decisions on quantities that come out of mm arithmetic.
LENGTH_TOL_MM = 1e-9


@dataclass(frozen=True)
class IsoScrew:
    """ISO metric cap screw data (all mm)."""
    name: str
    d: float           # nominal diameter
    pitch: float       # ISO 261 coarse pitch
    hole: float        # ISO 273 medium clearance hole
    head_dk: float     # ISO 4762 head diameter, max
    head_k: float      # ISO 4762 head height, max (= d)
    hex_s: float       # ISO 4762 hex socket size
    tap_drill: float   # DIN 336 / ISO 2306 tap drill


ISO_SCREWS = (
    IsoScrew("M2.5", 2.5, 0.45, 2.9, 4.5, 2.5, 2.0, 2.05),
    IsoScrew("M3", 3.0, 0.5, 3.4, 5.5, 3.0, 2.5, 2.5),
    IsoScrew("M4", 4.0, 0.7, 4.5, 7.0, 4.0, 3.0, 3.3),
    IsoScrew("M5", 5.0, 0.8, 5.5, 8.5, 5.0, 4.0, 4.2),
    IsoScrew("M6", 6.0, 1.0, 6.6, 10.0, 6.0, 5.0, 5.0),
)


@dataclass(frozen=True)
class ThreadLimits:
    """ISO 965-2 limits of size for a 6g screw and a 6H tapped hole (mm), as printed."""
    d_max: float    # 6g major diameter
    d_min: float
    d2_max: float   # 6g pitch diameter
    d2_min: float
    D2_max: float   # 6H pitch diameter
    D2_min: float
    D1_max: float   # 6H minor diameter
    D1_min: float


#: Bossard "Metric ISO threads" p. 97 (ISO 965-2). M10 is included for the sanity test against ISO 724.
THREAD_LIMITS_6G_6H = {
    "M2.5": ThreadLimits(2.480, 2.380, 2.188, 2.117, 2.303, 2.208, 2.138, 2.013),
    "M3": ThreadLimits(2.980, 2.874, 2.655, 2.580, 2.775, 2.675, 2.599, 2.459),
    "M4": ThreadLimits(3.978, 3.838, 3.523, 3.433, 3.663, 3.545, 3.422, 3.242),
    "M5": ThreadLimits(4.976, 4.826, 4.456, 4.361, 4.605, 4.480, 4.334, 4.134),
    "M6": ThreadLimits(5.974, 5.794, 5.324, 5.212, 5.500, 5.350, 5.153, 4.917),
    "M10": ThreadLimits(9.968, 9.732, 8.994, 8.862, 9.206, 9.026, 8.676, 8.376),
}

#: ISO 898-1 proof stress (12.9, 10.9) and ISO 3506-1 Rp0.2 (A4-70), MPa.
PROOF_STRESS_MPA = {"12.9": 970.0, "10.9": 830.0, "A4-70": 450.0}
#: MMPDS / ASM ultimate shear strength, MPa.
AL_SHEAR_MPA = {"7075-T6": 331.0, "6061-T6": 207.0}


# --------------------------------------------------------------------------- ISO 68-1 / ISO 898-1 thread geometry
def triangle_height(pitch: float) -> float:
    """ISO 68-1 fundamental triangle height H = (sqrt(3)/2)·P."""
    return SQRT3 / 2.0 * pitch


def pitch_diameter(d: float, pitch: float) -> float:
    """ISO 68-1 basic pitch diameter d2 = D2 = d - (3/4)·H."""
    return d - 0.75 * triangle_height(pitch)


def minor_diameter_external(d: float, pitch: float) -> float:
    """ISO 68-1 external-thread minor diameter d3 = d - (17/12)·H (root radius H/6 below D1)."""
    return d - 17.0 / 12.0 * triangle_height(pitch)


def minor_diameter_internal(d: float, pitch: float) -> float:
    """ISO 68-1 internal-thread minor diameter D1 = d - (5/4)·H."""
    return d - 1.25 * triangle_height(pitch)


def stress_area(d: float, pitch: float) -> float:
    """ISO 898-1 tensile stress area As = (pi/4)·((d2 + d3)/2)^2 [mm^2], unrounded."""
    return math.pi / 4.0 * ((pitch_diameter(d, pitch) + minor_diameter_external(d, pitch)) / 2.0) ** 2


def round_sig(x: float, digits: int) -> float:
    """Round to `digits` significant figures."""
    if x == 0.0:
        return 0.0
    return round(x, digits - 1 - math.floor(math.log10(abs(x))))


def iso_stress_area(d: float, pitch: float) -> float:
    """As as ISO 898-1 tabulates it (3 significant figures) [mm^2]."""
    return round_sig(stress_area(d, pitch), 3)


def round_up(x: float, step: float) -> float:
    """Smallest multiple of `step` that is >= x (with LENGTH_TOL_MM slack for float noise)."""
    return math.ceil((x - LENGTH_TOL_MM) / step) * step


# --------------------------------------------------------------------------- thread stripping (FED-STD-H28/2B)
def internal_tooth_width(x: float, en_max: float, pitch: float) -> float:
    """Axial width of the internal-thread tooth at diameter x: P/2 at the internal pitch diameter en_max, widening
    by tan30° per unit of diameter (each 60° flank moves (x - en_max)/2·tan30°)."""
    return pitch / 2.0 + (x - en_max) * TAN30


def internal_thread_shear_area(ds_min: float, en_max: float, pitch: float, engagement: float) -> float:
    """FED-STD-H28/2B internal-thread shear area, sheared on the cylinder at the screw major diameter ds_min:
    pi·ds_min·Le·w(ds_min)/P = pi·n·Le·Ds,min·[1/(2n) + tan30·(Ds,min - En,max)] [mm^2]."""
    return math.pi * ds_min * engagement * internal_tooth_width(ds_min, en_max, pitch) / pitch


def min_material_shear_area(screw: IsoScrew, engagement: float) -> float:
    """Internal-thread shear area at the ISO 965-2 6g/6H minimum-material limits (smallest screw major diameter,
    largest tapped pitch diameter) [mm^2]."""
    lim = THREAD_LIMITS_6G_6H[screw.name]
    return internal_thread_shear_area(lim.d_min, lim.D2_max, screw.pitch, engagement)


def stripping_capacity(shear_MPa: float, area_mm2: float, safety_factor: float) -> float:
    """Allowable screw force before the tapped thread strips, tau·A_n/SF [N]."""
    return shear_MPa * area_mm2 / safety_factor


# --------------------------------------------------------------------------- strength, friction, fasteners
def preload_proof_share(fraction: float, proof_MPa: float, area_mm2: float) -> float:
    """Preload as a share of proof load, F = fraction·Sp·As [N] (MPa·mm^2 = N)."""
    return fraction * proof_MPa * area_mm2


def friction_torque_coefficient(pressure_shape: Callable[[float], float], n: int = 20000) -> float:
    """C = T/(mu·F·d) for a two-jaw clamp on a shaft of diameter d = 2r and length l.

    theta is measured from the screw-force direction; each jaw presses the bore over |theta| <= pi/2 with pressure
    p(theta) = p0·shape(theta). Force balance on one jaw: F = ∫ p cos(theta) r l dtheta. Friction torque from both
    jaws: T = 2·mu·∫ p r^2 l dtheta. Hence C = T/(mu·F·2r) = ∫p dtheta / ∫p cos(theta) dtheta
    (midpoint rule, n panels).
    """
    h = math.pi / n
    num = den = 0.0
    for i in range(n):
        theta = -math.pi / 2.0 + (i + 0.5) * h
        p = pressure_shape(theta)
        num += p * h
        den += p * math.cos(theta) * h
    return num / den


#: Line contact on the screw axis (the conservative lower bound of friction_torque_coefficient).
LINE_CONTACT_COEFFICIENT = 1.0


def clamp_torque_per_screw(mu: float, preload_N: float, shaft_mm: float, clamp_factor: float) -> float:
    """Friction torque one screw's preload holds: C_line·mu·F·d·(clamp factor) [N·m]."""
    return LINE_CONTACT_COEFFICIENT * mu * preload_N * (shaft_mm / 1000.0) * clamp_factor


def tightening_torque(nut_factor: float, preload_N: float, d_mm: float) -> float:
    """Shigley Eq. 8-27, T = K·F·d [N·m]."""
    return nut_factor * preload_N * d_mm / 1000.0


def head_bearing_pressure(preload_N: float, head_dk: float, hole: float) -> float:
    """Pressure under the head on the annulus between the head diameter and the clearance hole [MPa]."""
    return preload_N / (math.pi / 4.0 * (head_dk ** 2 - hole ** 2))


def screws_needed(required_Nm: float, per_screw_Nm: float) -> int:
    """Smallest n >= 1 with n·T_per >= T_req (relative slack 1e-12 for float noise)."""
    if per_screw_Nm <= 0.0:
        raise ValueError("per-screw torque must be positive")
    n = 1
    while n * per_screw_Nm < required_Nm * (1.0 - 1e-12):
        n += 1
    return n


def screws_fit(clamp_length: float, axial_margin: float, cbore_dia: float, spacing: float) -> int:
    """Count screws placed one by one from the free end: every counterbore keeps `axial_margin` to both clamp ends
    and neighbouring screws sit `spacing` apart (greedy placement)."""
    if spacing <= 0.0:
        raise ValueError("screw spacing must be positive")
    first = axial_margin + cbore_dia / 2.0
    last_allowed = clamp_length - axial_margin - cbore_dia / 2.0
    count = 0
    centre = first
    while centre <= last_allowed + LENGTH_TOL_MM:
        count += 1
        centre = first + count * spacing
    return count


def hex_key_passes_port(hex_s: float, port_d: float = 6.0, port_pitch: float = 1.0) -> bool:
    """A hex key of size s (across flats) passes a tapped port when its across-corners size s/cos30° is below the
    port's internal minor diameter D1 (M6 x 1: D1 = 4.917 mm)."""
    return hex_s / COS30 < minor_diameter_internal(port_d, port_pitch)


def key_bearing_pressure_MPa(torque_Nm: float, shaft_mm: float, contact_mm: float, length_mm: float) -> float:
    """p = 2T/(d·h·L), everything in SI, returned in MPa."""
    return 2.0 * torque_Nm / ((shaft_mm / 1000.0) * (contact_mm / 1000.0) * (length_mm / 1000.0)) / 1e6


def flange_friction_torque_Nm(mu: float, n_screws: int, preload_N: float, bolt_circle_mm: float) -> float:
    """T = mu·n·F·(D_bc/2) [N·m]."""
    return mu * n_screws * preload_N * (bolt_circle_mm / 2.0) / 1000.0


# --------------------------------------------------------------------------- clamp sizing
@dataclass(frozen=True)
class ClampSetup:
    """Everything the sizing needs, in the units of the workbook inputs (mm, N·m, MPa)."""
    shaft_mm: float
    boss_od_mm: float
    slit_mm: float
    ligament_mm: float
    wall_min_mm: float
    grip_min_mm: float
    axial_margin_mm: float
    clamp_length_mm: float
    engagement_x_d: float
    preload_fraction: float
    strip_sf: float
    friction: float
    clamp_factor: float
    nut_factor: float
    screw_class: str          # "12.9", "10.9" or "A4-70"
    alloy: str                # "7075-T6" or "6061-T6"
    max_torque_Nm: float      # highest torque through the coupling
    safety_factor: float
    cbore_allowance_mm: float  # counterbore diameter = head + this (workbook design rule)
    head_gap_mm: float         # screw spacing = head + this (workbook design rule)
    length_step_mm: float      # screw lengths come in multiples of this (workbook design rule)


def clamp_geometry(s: ClampSetup, screw: IsoScrew) -> dict:
    """Screw offset, wall outside the hole, head seat, grip, thread available, counterbore depth.
    Quantities that do not exist when the head seat does not fit are None."""
    R = s.boss_od_mm / 2.0
    e = s.shaft_mm / 2.0 + s.ligament_mm + screw.hole / 2.0
    wall = R - (e + screw.hole / 2.0)
    head_edge = e + screw.head_dk / 2.0
    fits = head_edge <= R
    x_od = math.sqrt(R ** 2 - e ** 2) if e < R else None
    x_seat = math.sqrt(R ** 2 - head_edge ** 2) if fits else None
    engagement = s.engagement_x_d * screw.d
    return {
        "offset_mm": e,
        "wall_out_mm": wall,
        "head_fits": 1 if fits else 0,
        "grip_mm": (x_seat - s.slit_mm / 2.0) if fits else None,
        "thread_avail_mm": (x_od - s.slit_mm / 2.0) if x_od is not None else None,
        "engagement_req_mm": engagement,
        "cbore_dia_mm": screw.head_dk + s.cbore_allowance_mm,
        "cbore_depth_mm": (x_od - x_seat) if fits else None,
    }


def geometry_ok(s: ClampSetup, geo: dict) -> int:
    """All four geometric rules: wall, head seat, grip, thread length available."""
    ok = (geo["wall_out_mm"] >= s.wall_min_mm and geo["head_fits"] == 1
          and geo["grip_mm"] >= s.grip_min_mm
          and geo["thread_avail_mm"] is not None and geo["thread_avail_mm"] >= geo["engagement_req_mm"])
    return 1 if ok else 0


def engagement_achieved(s: ClampSetup, geo: dict, length_mm: float) -> float:
    """Thread a seated screw of this under-head length engages past the head-side jaw and the open slit [mm]."""
    return length_mm - geo["grip_mm"] - s.slit_mm


def max_length_inside(s: ClampSetup, geo: dict) -> float:
    """Longest under-head length that ends inside the far jaw: the seat-to-far-OD chord, grip + slit + avail [mm]."""
    return geo["grip_mm"] + s.slit_mm + geo["thread_avail_mm"]


def screw_tip_inside(s: ClampSetup, geo: dict, length_mm: float) -> int:
    """1 when a screw of this under-head length ends inside the far jaw."""
    return 1 if length_mm <= max_length_inside(s, geo) + LENGTH_TOL_MM else 0


def valid_screw_lengths(s: ClampSetup, geo: dict) -> list[float]:
    """Every length-step multiple that engages the required thread and ends inside the far jaw
    (empty when the head seat does not fit, or when no step multiple meets both rules)."""
    if geo["head_fits"] != 1:
        return []
    shortest = round_up(geo["grip_mm"] + s.slit_mm + geo["engagement_req_mm"], s.length_step_mm)
    out = []
    length = shortest
    while length <= max_length_inside(s, geo) + LENGTH_TOL_MM:
        out.append(length)
        length += s.length_step_mm
    return out


def size_row(s: ClampSetup, screw: IsoScrew) -> dict:
    """The complete independent evaluation of one screw size (lengths are the valid_screw_lengths list)."""
    geo = clamp_geometry(s, screw)
    As = iso_stress_area(screw.d, screw.pitch)
    F_strength = preload_proof_share(s.preload_fraction, PROOF_STRESS_MPA[s.screw_class], As)
    F_strip = stripping_capacity(AL_SHEAR_MPA[s.alloy], min_material_shear_area(screw, geo["engagement_req_mm"]),
                                 s.strip_sf)
    F = min(F_strength, F_strip)
    t_per = clamp_torque_per_screw(s.friction, F, s.shaft_mm, s.clamp_factor)
    need = screws_needed(s.max_torque_Nm * s.safety_factor, t_per)
    spacing = screw.head_dk + s.head_gap_mm
    fit = screws_fit(s.clamp_length_mm, s.axial_margin_mm, geo["cbore_dia_mm"], spacing)
    geo_ok = geometry_ok(s, geo)
    return {
        **geo,
        "As_mm2": As,
        "geometry_ok": geo_ok,
        "preload_strength_N": F_strength,
        "preload_strip_N": F_strip,
        "preload_N": F,
        "head_pressure_MPa": head_bearing_pressure(F, screw.head_dk, screw.hole),
        "torque_per_screw_Nm": t_per,
        "screws_needed": need,
        "pitch_axial_mm": spacing,
        "screws_fit": fit,
        "works": 1 if (geo_ok == 1 and need <= fit) else 0,
        "clamp_torque_Nm": need * t_per,
        "sf_coupling": need * t_per / s.max_torque_Nm,
        "tightening_Nm": tightening_torque(s.nut_factor, F, screw.d),
        "valid_lengths_mm": valid_screw_lengths(s, geo),
        "vent_port_ok": 1 if hex_key_passes_port(screw.hex_s) else 0,
    }


def recommend(s: ClampSetup) -> tuple[int, list[dict]]:
    """(1-based index of the first size that works, 0 when none; all rows)."""
    rows = [size_row(s, screw) for screw in ISO_SCREWS]
    for i, row in enumerate(rows):
        if row["works"] == 1:
            return i + 1, rows
    return 0, rows
```

File: `reference/magcoupling-py/audit/references/sweep_ref.py` (create)
```python
"""Independent single-point evaluation of one sweep row (engine: sweeps.py; sheets 'Gap sweep' and 'Pole sweep').

Nothing here imports the engine. The row is assembled from the earlier tasks' references, so each formula lives in
one place in the audit:
- planar model (Task 3, audit.references.planar): Br(T), harmonic amplitudes B_n = Br·4/(n·pi)·sin(n·pi·fill/2),
  wave number k_n = n·pi/pole pitch (= n·(poles/2)/R_gap, the README form), the steel-backed and free-space
  geometry factors S_n, the harmonic shear stress B_in·B_on/(2·mu0)·S_n·sin(k·delta), the torque of a uniform shear
  on a cylinder and the empirical end factor 1 - c_end·pitch/L;
- geometry (Task 4, block_geometry.coupling_geometry): face radius, inner corner radius (block rectangle, or the
  arc radius), outer face apothem, flat-face gap, pocket polygon vertex radius and cup OD, gap radius, pole pitch,
  fill = block width / pole pitch at the block's mid-thickness radius, flat widths;
- gearbox input torque (Task 4, metal_stack.gearbox_input_torque).

What this module adds: the row assembly (README 'Torque model' steps 1-4 at one corner gap and pole count, with
the pull-out evaluated at half a pole pitch, where sin(k·delta) = sin(n·pi/2)), the status priority and the pole
sweep's apothem rule as stated.

A row is parameterized by the corner gap (inner block corner to outer block face). coupling_geometry takes the
flat-face gap, which is the corner gap plus the inner block's corner overhang (corner radius - face radius; zero
for arc magnets), so geometry_at_corner_gap builds it in two explicit steps.

This reproduces the model as the README states it. Whether that model is right (for example the sin(n·pi/2)
evaluation of every harmonic at the fundamental's pull-out angle) is audited by the torque family, not here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from audit.references.block_geometry import CouplingGeometry, coupling_geometry
from audit.references.metal_stack import gearbox_input_torque
from audit.references.planar import (br_at, end_factor, free_space_factor, harmonic_shear_stress, iron_backed_factor,
                                     square_wave_harmonic, torque_on_cylinder, wave_number)

HARMONICS = (1, 3, 5)

STATUS_INNER_NARROW = "inner flat too narrow"
STATUS_OUTER_NARROW = "outer flat too narrow"
STATUS_OD = "outside OD envelope"
STATUS_BELOW_MIN = "below hot minimum"
STATUS_NOMINAL = "nominal: test needed"

#: Pole-sweep apothem rule as the sheet note states it (engine sweeps.py docstrings, ported from the workbook):
#: "the smallest inner apothem that fits the block width (+0.05 mm) and the keyed bore wall (2.5 mm)".
FLAT_MARGIN_MM = 0.05
KEYED_WALL_MM = 2.5


@dataclass(frozen=True)
class SweepSetup:
    """The values a sweep row needs (workbook units: mm, T, °C)."""
    faceted: int
    backiron: int
    t_i_mm: float
    w_i_mm: float
    t_o_mm: float
    w_o_mm: float
    length_mm: float
    br_i20_T: float
    br_o20_T: float
    alpha_per_C: float
    op_temp_C: float
    bond_inner_mm: float
    bond_outer_mm: float
    cup_wall_corner_mm: float
    bore_mm: float
    keyway_mm: float
    c_end: float
    mu0: float
    gear_ratio: float
    gear_eff: float
    required_floor_Nm: float
    max_diameter_mm: float


def required_floor(drive_torque_Nm: float, drive_safety_factor: float, required_min_Nm: float) -> float:
    """Larger of the traction need (drive torque × safety factor) and the service minimum."""
    return max(drive_torque_Nm * drive_safety_factor, required_min_Nm)


def geometry_at_corner_gap(s: SweepSetup, npole: int, a_i_mm: float, corner_gap_mm: float) -> CouplingGeometry:
    """Task 4's coupling geometry for this row: first the inner block's corner overhang, then the geometry at the
    flat-face gap corner_gap + overhang."""
    kw = dict(npole=npole, inner_back_apothem=a_i_mm, t_i=s.t_i_mm, w_i=s.w_i_mm, t_o=s.t_o_mm, w_o=s.w_o_mm,
              bond_inner=s.bond_inner_mm, bond_outer=s.bond_outer_mm, cup_wall_corner=s.cup_wall_corner_mm,
              bore=s.bore_mm, keyway_depth=s.keyway_mm, faceted=s.faceted == 1)
    probe = coupling_geometry(face_gap=0.0, **kw)
    overhang = probe.inner_corner_radius - probe.inner_face_radius
    return coupling_geometry(face_gap=corner_gap_mm + overhang, **kw)


def sweep_status(s: SweepSetup, geo: CouplingGeometry, pullout_op_Nm: float) -> str:
    """First failing rule in the workbook's priority order (sweeps.py docstring): the hub flat under the inner block
    is narrower than the block, the outer face flat is narrower than the outer block, the cup OD exceeds the
    envelope, the nominal pull-out is below the required floor."""
    if geo.inner_flat_width < s.w_i_mm:
        return STATUS_INNER_NARROW
    if geo.outer_flat_width < s.w_o_mm:
        return STATUS_OUTER_NARROW
    if geo.cup_od > s.max_diameter_mm:
        return STATUS_OD
    if pullout_op_Nm < s.required_floor_Nm:
        return STATUS_BELOW_MIN
    return STATUS_NOMINAL


def sweep_row(s: SweepSetup, npole: int, a_i_mm: float, corner_gap_mm: float, f_cal: float) -> dict:
    """Independent evaluation of one sweep row; keys are the engine's SweepRow field names (minus 'variable')."""
    geo = geometry_at_corner_gap(s, npole, a_i_mm, corner_gap_mm)
    br_i = br_at(s.br_i20_T, s.alpha_per_C, s.op_temp_C)
    br_o = br_at(s.br_o20_T, s.alpha_per_C, s.op_temp_C)
    r_gap_m = geo.gap_radius / 1000.0
    pitch_m = geo.pole_pitch / 1000.0
    t_i_m, t_o_m, g_m = s.t_i_mm / 1000.0, s.t_o_mm / 1000.0, geo.face_gap / 1000.0
    out = {"inner_apothem_mm": a_i_mm, "corner_gap_mm": geo.corner_gap, "centre_gap_mm": geo.face_gap,
           "outer_face_apothem_mm": geo.outer_face_apothem, "cup_od_mm": geo.cup_od, "gap_radius_mm": geo.gap_radius,
           "pole_pitch_mm": geo.pole_pitch, "fill_inner": geo.fill_inner, "fill_outer": geo.fill_outer}
    tau = 0.0
    for n in HARMONICS:
        k = wave_number(n, pitch_m)
        s_n = iron_backed_factor(k, t_i_m, t_o_m, g_m) if s.backiron == 1 else free_space_factor(k, t_i_m, t_o_m, g_m)
        tau_n = harmonic_shear_stress(square_wave_harmonic(br_i, n, geo.fill_inner),
                                      square_wave_harmonic(br_o, n, geo.fill_outer), s_n, k, pitch_m / 2.0, s.mu0)
        out.update({f"k{n}": k, f"s{n}": s_n, f"tau{n}_Pa": tau_n})
        tau += tau_n
    torque_2d = torque_on_cylinder(tau, r_gap_m, s.length_mm / 1000.0)
    f_end = end_factor(s.c_end, geo.pole_pitch, s.length_mm)
    pull_op = torque_2d * f_end * f_cal
    out.update({"tau_Pa": tau, "torque_2d_Nm": torque_2d, "f_end": f_end, "pullout_op_Nm": pull_op,
                # torque is bilinear in the two rings' remanence (README step 4)
                "pullout_20C_Nm": pull_op * (s.br_i20_T * s.br_o20_T) / (br_i * br_o),
                "gearbox_input_Nm": gearbox_input_torque(pull_op, s.gear_ratio, s.gear_eff),
                "status": sweep_status(s, geo, pull_op)})
    return out


def stated_min_inner_apothem(npole: int, w_i_mm: float, bore_mm: float, keyway_mm: float) -> float:
    """The pole sweep's apothem rule as stated: the hub flat holds the block, 2·a·tan(pi/N) >= w_i, plus 0.05 mm on
    the apothem; and a 2.5 mm keyed-bore wall measured, as the note leaves it, from the block-back apothem:
    a >= bore/2 + keyway + 2.5."""
    return max(w_i_mm / (2.0 * math.tan(math.pi / npole)) + FLAT_MARGIN_MM, bore_mm / 2.0 + keyway_mm + KEYED_WALL_MM)
```

File: `reference/magcoupling-py/audit/references/units_ref.py` (create)
```python
"""Minimal quantity algebra for dimensional checks (SI base dimensions m, kg, s, A, K).

A Quantity stores its value in SI base units plus integer dimension exponents. Multiplying and dividing combine
both; adding or subtracting quantities of different dimensions raises DimensionError; `to(unit)` converts to a
display unit and raises when the dimensions differ. Units are Quantities of value "one of that unit", so
`3.2 * MM` is 0.0032 m. Nothing here imports the engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_BASE = ("m", "kg", "s", "A", "K")


class DimensionError(ValueError):
    """Raised when an operation mixes incompatible dimensions."""


@dataclass(frozen=True)
class Quantity:
    value: float
    dims: tuple = (0, 0, 0, 0, 0)

    # -- arithmetic
    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, tuple(a + b for a, b in zip(self.dims, other.dims)))
        return Quantity(self.value * other, self.dims)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, tuple(a - b for a, b in zip(self.dims, other.dims)))
        return Quantity(self.value / other, self.dims)

    def __rtruediv__(self, other):
        return Quantity(other / self.value, tuple(-a for a in self.dims))

    def __pow__(self, n: int):
        if not isinstance(n, int):
            raise TypeError("integer powers only")
        return Quantity(self.value ** n, tuple(a * n for a in self.dims))

    def __neg__(self):
        return Quantity(-self.value, self.dims)

    def _same(self, other, op: str):
        if not isinstance(other, Quantity) or other.dims != self.dims:
            raise DimensionError(f"cannot {op} {self.describe()} and {other!r}")

    def __add__(self, other):
        self._same(other, "add")
        return Quantity(self.value + other.value, self.dims)

    def __sub__(self, other):
        self._same(other, "subtract")
        return Quantity(self.value - other.value, self.dims)

    # -- inspection
    def is_dimensionless(self) -> bool:
        return all(a == 0 for a in self.dims)

    def dimensionless(self) -> float:
        """The plain number, for exp/sinh arguments; raises unless dimensionless."""
        if not self.is_dimensionless():
            raise DimensionError(f"expected a dimensionless quantity, got {self.describe()}")
        return self.value

    def to(self, unit: "Quantity") -> float:
        """Value expressed in `unit`; raises when the dimensions differ."""
        if unit.dims != self.dims:
            raise DimensionError(f"cannot express {self.describe()} in {unit.describe()}")
        return self.value / unit.value

    def describe(self) -> str:
        parts = [f"{b}^{e}" for b, e in zip(_BASE, self.dims) if e]
        return f"{self.value:g} " + ("·".join(parts) if parts else "(dimensionless)")


def _base(i: int) -> Quantity:
    dims = [0, 0, 0, 0, 0]
    dims[i] = 1
    return Quantity(1.0, tuple(dims))


ONE = Quantity(1.0)
M, KG, S, A, K = (_base(i) for i in range(5))

MM = 1e-3 * M
GRAM = 1e-3 * KG
N = KG * M / S ** 2
NM = N * M
PA = N / M ** 2
MPA = 1e6 * PA
J = N * M
W = J / S
TESLA = KG / (S ** 2 * A)
H_PER_M = TESLA * M / A            # the unit of mu0
RAD_PER_S = ONE / S                # radians are dimensionless
RPM = (2.0 * math.pi / 60.0) * RAD_PER_S   # rev/min as an angular speed
J_PER_KG_K = J / (KG * K)
J_PER_K = J / K
W_PER_K = W / K
A_PER_M = A / M                    # magnetic field strength H
S_PER_M = A ** 2 * S ** 3 / (KG * M ** 3)   # electrical conductivity (siemens per metre)
PER_K = ONE / K                    # thermal expansion coefficient
KG_M2 = KG * M ** 2                # mass moment of inertia
```

- [ ] **Step 2: Write the reference's own sanity test**

File: `reference/magcoupling-py/audit/tests/test_clamps_sweeps_units_reference_sanity.py` (create)
```python
"""Sanity tests: the Task 7 reference modules against textbook cases with known answers (no engine involved).

Every test name contains 'sanity' so reference self-tests can be told apart from engine checks.
"""
import math
from types import SimpleNamespace

import pytest

from audit.common import MU0_EXACT, TOL_ALGEBRA, assert_all, flag_item, text_item
from audit.references import clamp_ref as cr
from audit.references import sweep_ref as sr
from audit.references import units_ref as u
from audit.references.planar import free_space_factor, harmonic_shear_stress, iron_backed_factor, wave_number

#: ISO 724 / ISO 965-2 print diameters to 0.001 mm, so a printed value is within 0.0005 mm of the exact one.
PRINT_TOL_MM = 0.0005


# --------------------------------------------------------------------------- clamp_ref: thread data and geometry
@pytest.mark.family("clamps")
def test_sanity_iso68_m10_diameters():
    """ISO 724 tabulates M10 x 1.5 as d2 = 9.026, d3 = 8.160, D1 = 8.376 mm (printed to 0.001 mm)."""
    assert_all([(f"M10x1.5 {what}", "ISO 724 table", got, table, PRINT_TOL_MM / table)
                for what, got, table in (("d2", cr.pitch_diameter(10, 1.5), 9.026),
                                         ("d3", cr.minor_diameter_external(10, 1.5), 8.160),
                                         ("D1", cr.minor_diameter_internal(10, 1.5), 8.376))])


@pytest.mark.family("clamps")
def test_sanity_iso898_stress_area():
    """ISO 898-1 tabulated stress areas (3 significant figures) are reproduced exactly: M8 36.6, M10 58.0,
    M12 84.3, M16 157 mm^2."""
    assert_all([(f"As M{d}", "ISO 898-1 table", cr.iso_stress_area(d, pitch), table, TOL_ALGEBRA)
                for d, pitch, table in ((8, 1.25, 36.6), (10, 1.5, 58.0), (12, 1.75, 84.3), (16, 2.0, 157.0))])


@pytest.mark.family("clamps")
def test_sanity_iso898_proof_load_m10():
    """ISO 898-1 proof loads for M10: 56 300 N (12.9), 48 100 N (10.9). The standard builds them as the tabulated
    As times Sp, printed to 3 significant figures; the same construction must match exactly."""
    as_table = cr.iso_stress_area(10, 1.5)
    assert_all([(f"proof load M10 {cls}", "ISO 898-1 table",
                 cr.round_sig(cr.preload_proof_share(1.0, cr.PROOF_STRESS_MPA[cls], as_table), 3), table, TOL_ALGEBRA)
                for cls, table in (("12.9", 56300.0), ("10.9", 48100.0))])


@pytest.mark.family("clamps")
def test_sanity_iso965_limits_transcription():
    """The transcribed 6g/6H rows are consistent with the ISO 68-1 basic profile: 6H has zero fundamental deviation,
    so its minimum pitch and minor diameters are the basic D2 and D1; 6g shifts major and pitch diameter by the same
    deviation es, so d - d_max = D2 - d2_max; every band is positive. Printed to 0.001 mm, so a difference of two
    printed values is good to 0.001 mm."""
    pitches = {s.name: s.pitch for s in cr.ISO_SCREWS} | {"M10": 1.5}
    items = []
    for name, lim in cr.THREAD_LIMITS_6G_6H.items():
        d, pitch = float(name[1:]), pitches[name]
        d2, d1 = cr.pitch_diameter(d, pitch), cr.minor_diameter_internal(d, pitch)
        items += [(f"{name} 6H D2 min = basic D2", "ISO 965-2 / ISO 68-1", lim.D2_min, d2, PRINT_TOL_MM / d2),
                  (f"{name} 6H D1 min = basic D1", "ISO 965-2 / ISO 68-1", lim.D1_min, d1, PRINT_TOL_MM / d1),
                  flag_item(f"{name} 6g deviation on major {d - lim.d_max:.3f} = on pitch "
                            f"{lim.D2_min - lim.d2_max:.3f}", "ISO 965-2",
                            abs((d - lim.d_max) - (lim.D2_min - lim.d2_max)) <= 2 * PRINT_TOL_MM),
                  flag_item(f"{name} every tolerance band positive", "ISO 965-2",
                            lim.d_min < lim.d_max <= d and lim.d2_min < lim.d2_max < d2
                            and lim.D2_min < lim.D2_max and lim.D1_min < lim.D1_max)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_sanity_internal_thread_shear_area():
    """ISO 68-1 basic profile: the internal tooth is P/2 wide at the pitch diameter, P/4 (crest flat) at D1 and
    leaves a P/8 root gap at d, so the basic-size area is 0.875·pi·d·Le. The FED-STD-H28/2B expression with its
    printed constant 0.57735 (tan30 to 5 digits, so tol 1e-5) gives the same area. Minimum-material areas are
    below the basic-size areas for every size."""
    items = []
    for s in cr.ISO_SCREWS:
        d, p = s.d, s.pitch
        d2 = cr.pitch_diameter(d, p)
        for what, got, ref in (("width at D2", cr.internal_tooth_width(d2, d2, p), p / 2),
                               ("width at D1", cr.internal_tooth_width(cr.minor_diameter_internal(d, p), d2, p), p / 4),
                               ("width at d", cr.internal_tooth_width(d, d2, p), 7 * p / 8),
                               ("basic area / (pi·d·Le)",
                                cr.internal_thread_shear_area(d, d2, p, 1.0) / (math.pi * d), 0.875)):
            items.append((f"{s.name} {what}", "ISO 68-1 profile", got, ref, TOL_ALGEBRA))
        lim = cr.THREAD_LIMITS_6G_6H[s.name]
        n = 1.0 / p
        fed = math.pi * n * 1.0 * lim.d_min * (1.0 / (2.0 * n) + 0.57735 * (lim.d_min - lim.D2_max))
        a_min, a_basic = cr.min_material_shear_area(s, 1.0), cr.internal_thread_shear_area(d, d2, p, 1.0)
        items.append((f"{s.name} FED-STD-H28 expression", "FED-STD-H28/2B", a_min, fed, 1e-5))
        items.append(flag_item(f"{s.name} min-material area {a_min:.4f} < basic {a_basic:.4f} mm^2 per mm of Le",
                               "ISO 965-2", a_min < a_basic))
    assert_all(items)


@pytest.mark.family("clamps")
def test_sanity_clamp_friction_coefficient():
    """C = ∫p/∫p·cos: uniform pressure -> pi/2, cosine -> 4/pi, a sharp cos^2000 peak -> 1 (bounded by the Wallis
    ratio, 1 <= C <= 1 + 1/m). Uniform pressure also reproduces Shigley's press-fit torque T = (pi/2)·f·p·l·d^2
    (per jaw F = ∫p·cos·r·l = p·d·l). Midpoint rule with 20 000 panels: error ~ (pi/20000)^2 ~ 2.5e-8,
    tol 1e-7."""
    tol = 1e-7
    c_uni = cr.friction_torque_coefficient(lambda t: 1.0)
    c_cos = cr.friction_torque_coefficient(math.cos)
    c_line = cr.friction_torque_coefficient(lambda t: math.cos(t) ** 2000)
    p, d, l, mu = 10e6, 0.010, 0.010, 0.15
    assert_all([("uniform pressure C", "derivation", c_uni, math.pi / 2, tol),
                ("cosine pressure C", "derivation", c_cos, 4 / math.pi, tol),
                flag_item(f"near-line-contact C = {c_line:.6f} within [1, 1 + 1/2000]", "Wallis bound",
                          1.0 <= c_line <= 1.0 + 1.0 / 2000),
                ("press-fit torque", "Shigley (pi/2)·f·p·l·d^2", c_uni * mu * (p * d * l) * d,
                 math.pi / 2 * mu * p * l * d ** 2, tol)])


@pytest.mark.family("clamps")
def test_sanity_screws_fit_hand_cases():
    """Greedy placement, hand-worked: centres at margin + cbore/2 = 4 mm, then every 6.5 mm while the counterbore
    keeps the margin at the far end (last centre <= L - 1 - 3)."""
    assert_all([(f"screws fit in {L} mm", "hand count", cr.screws_fit(L, 1.0, 6.0, 6.5), expected, 0.0)
                for L, expected in ((20.0, 2), (14.5, 2), (14.4, 1), (7.0, 0), (26.0, 3))])


@pytest.mark.family("clamps")
def test_sanity_hex_key_through_m6_port():
    """M6 x 1 internal minor diameter D1 = 4.917 mm (ISO 965-2 6H minimum); a 4 mm key (4.619 across corners)
    passes, a 5 mm key (5.774) does not."""
    d1 = cr.minor_diameter_internal(6.0, 1.0)
    table = cr.THREAD_LIMITS_6G_6H["M6"].D1_min
    assert_all([("M6 D1", "ISO 965-2 table", d1, table, PRINT_TOL_MM / table),
                flag_item("4 mm key passes the M6 port", "s/cos30 < D1", cr.hex_key_passes_port(4.0)),
                flag_item("5 mm key is refused by the M6 port", "s/cos30 < D1", not cr.hex_key_passes_port(5.0))])


@pytest.mark.family("clamps")
def test_sanity_invalid_inputs_raise():
    """Non-positive spacing or per-screw torque would loop forever or divide by zero; both are rejected."""
    with pytest.raises(ValueError):
        cr.screws_fit(10.0, 1.0, 5.0, 0.0)
    with pytest.raises(ValueError):
        cr.screws_needed(7.5, 0.0)


@pytest.mark.family("clamps")
def test_sanity_clamp_geometry_345_triangle():
    """Boss R = 5, screw offset e = 3 (shaft 2 + ligament 1 + hole 2), head 2, no slit: half-chord 4, seat at 3
    (3-4-5 triangles), wall 1, counterbore depth 1. With 2 mm of required engagement the shortest length is
    grip + slit + 2 = 5 mm and the seat-to-far-OD chord is 3 + 0 + 4 = 7 mm, so 5, 6 and 7 mm screws are valid
    in 1 mm steps; the head of a larger screw (d_k = 6) does not fit and leaves no valid length."""
    screw = cr.IsoScrew("test", d=1.0, pitch=0.25, hole=2.0, head_dk=2.0, head_k=1.0, hex_s=1.0, tap_drill=0.75)
    setup = cr.ClampSetup(shaft_mm=2.0, boss_od_mm=10.0, slit_mm=0.0, ligament_mm=1.0, wall_min_mm=0.5, grip_min_mm=1.0,
                          axial_margin_mm=0.0, clamp_length_mm=10.0, engagement_x_d=2.0, preload_fraction=0.75,
                          strip_sf=1.0, friction=0.15, clamp_factor=1.0, nut_factor=0.2, screw_class="12.9",
                          alloy="7075-T6", max_torque_Nm=1.0, safety_factor=1.0, cbore_allowance_mm=0.5,
                          head_gap_mm=1.0, length_step_mm=1.0)
    geo = cr.clamp_geometry(setup, screw)
    big = cr.clamp_geometry(setup, cr.IsoScrew("big", 1.0, 0.25, 2.0, 6.0, 1.0, 1.0, 0.75))
    items = [(key, "3-4-5 triangle", geo[key], ref, TOL_ALGEBRA)
             for key, ref in (("offset_mm", 3.0), ("wall_out_mm", 1.0), ("head_fits", 1), ("grip_mm", 3.0),
                              ("thread_avail_mm", 4.0), ("cbore_depth_mm", 1.0), ("engagement_req_mm", 2.0))]
    items += [("engagement of a 5 mm screw", "hand sum", cr.engagement_achieved(setup, geo, 5.0), 2.0, TOL_ALGEBRA),
              ("longest length inside", "hand sum", cr.max_length_inside(setup, geo), 7.0, TOL_ALGEBRA),
              text_item("valid lengths", "hand list", str(cr.valid_screw_lengths(setup, geo)), str([5.0, 6.0, 7.0])),
              text_item("valid lengths when the head does not fit", "hand list",
                        str(cr.valid_screw_lengths(setup, big)), str([]))]
    assert_all(items)


# --------------------------------------------------------------------------- sweep_ref
def _setup(**over) -> sr.SweepSetup:
    base = dict(faceted=1, backiron=1, t_i_mm=3.0, w_i_mm=6.0, t_o_mm=3.0, w_o_mm=6.0, length_mm=12.0, br_i20_T=1.3,
                br_o20_T=1.3, alpha_per_C=0.0, op_temp_C=20.0, bond_inner_mm=0.05, bond_outer_mm=0.0,
                cup_wall_corner_mm=2.0, bore_mm=10.0, keyway_mm=1.7, c_end=0.0, mu0=MU0_EXACT, gear_ratio=1.0,
                gear_eff=1.0, required_floor_Nm=2.0, max_diameter_mm=40.0)
    base.update(over)
    return sr.SweepSetup(**base)


@pytest.mark.family("sweeps")
def test_sanity_geometry_at_corner_gap():
    """Faceted: inner face radius 10 + 3 = 13, corner radius hypot(13, 3) = 13.3417 mm, so a 1 mm corner gap is a
    1.3417 mm flat-face gap and the corner gap comes back unchanged. Arcs: no overhang, face gap = corner gap.
    The README wave number n·(poles/2)/R_gap equals planar.wave_number(n, 2·pi·R_gap/poles)."""
    s = _setup()
    geo = sr.geometry_at_corner_gap(s, 10, 10.0, 1.0)
    arc = sr.geometry_at_corner_gap(_setup(faceted=0), 10, 10.0, 1.0)
    face_gap = math.hypot(13.0, 3.0) - 13.0 + 1.0
    assert_all([("corner gap round trip", "construction", geo.corner_gap, 1.0, TOL_ALGEBRA),
                ("flat-face gap", "hypot(13, 3) - 13 + 1", geo.face_gap, face_gap, TOL_ALGEBRA),
                ("gap radius", "13 + face gap / 2", geo.gap_radius, 13.0 + face_gap / 2, TOL_ALGEBRA),
                ("arc face gap", "no overhang", arc.face_gap, 1.0, TOL_ALGEBRA),
                ("wave number forms agree", "README k", wave_number(3, 2 * math.pi * 0.0135 / 10),
                 3 * (10 / 2) / 0.0135, TOL_ALGEBRA)])


@pytest.mark.family("sweeps")
def test_sanity_geometry_factor_limits():
    """Thick magnets (k·t = 40): both factors -> e^(-k·g)/2. Thin magnets (k·t = 1e-4): free space -> (k·t)^2/2 and
    iron-backed -> k·t_i·t_o/(t_i + t_o + g), the flat magnetic-circuit field ratio. Thin limits carry O(k·t)
    Taylor error, tol 1e-3. A harmonic at pitch/2 displacement carries sin(n·pi/2): +1, -1, +1 for n = 1, 3, 5."""
    k, t, g = 1000.0, 0.040, 0.0014
    items = [(f"thick-magnet limit ({name})", "e^-kg/2", f(k, t, t, g), math.exp(-k * g) / 2, 1e-12)
             for name, f in (("iron", iron_backed_factor), ("free", free_space_factor))]
    k, t = 1.0, 1e-4
    items += [("thin-magnet limit (free)", "(kt)^2/2", free_space_factor(k, t, t, 0.0), (k * t) ** 2 / 2, 1e-3),
              ("thin-magnet limit (iron)", "k·ti·to/(ti+to+g)", iron_backed_factor(k, t, t, 0.5 * t),
               k * t * t / (2.5 * t), 1e-3)]
    pitch = 0.004
    items += [(f"sign of harmonic {n} at half a pitch", "sin(n·pi/2)",
               harmonic_shear_stress(1.0, 1.0, 1.0, wave_number(n, pitch), pitch / 2, 0.5), (-1.0) ** ((n - 1) // 2),
               1e-12) for n in (1, 3, 5)]
    assert_all(items)


@pytest.mark.family("sweeps")
def test_sanity_stated_min_inner_apothem():
    """10 poles, 6.35 mm block: 6.35/(2·tan 18°) + 0.05 = 9.8216 mm; 6 poles: wall rule 5 + 1.7 + 2.5 = 9.2 mm."""
    assert_all([(f"stated min apothem N={npole}", "hand value", sr.stated_min_inner_apothem(npole, 6.35, 10.0, 1.7),
                 ref, TOL_ALGEBRA)
                for npole, ref in ((10, 6.35 / (2 * math.tan(math.pi / 10)) + 0.05), (6, 9.2))])


@pytest.mark.family("sweeps")
def test_sanity_sweep_status_priority():
    """Hand-built cases hit each status once, in priority order; a pull-out equal to the floor is not below it."""
    s = _setup()
    cases = ((5.2, 20.0, 30.0, 5.0, sr.STATUS_INNER_NARROW),
             (6.5, 5.85, 30.0, 5.0, sr.STATUS_OUTER_NARROW),
             (6.5, 7.8, 41.0, 5.0, sr.STATUS_OD),
             (6.5, 7.8, 39.0, 1.9, sr.STATUS_BELOW_MIN),
             (6.5, 7.8, 39.0, 2.0, sr.STATUS_NOMINAL))
    assert_all([text_item(f"status for flats {fi}/{fo} mm, OD {od} mm, pull-out {x} N·m", "priority rule",
                          sr.sweep_status(s, SimpleNamespace(inner_flat_width=fi, outer_flat_width=fo, cup_od=od), x),
                          expected)
                for fi, fo, od, x, expected in cases])


# --------------------------------------------------------------------------- units_ref
@pytest.mark.family("constants")
def test_sanity_magnetic_pressure_one_tesla():
    """B^2/(2·mu0) at 1 T is 397 887 Pa (textbook magnetic pressure) and carries the dimensions of Pa."""
    q = (1.0 * u.TESLA) ** 2 / (2 * MU0_EXACT * u.H_PER_M)
    assert_all([("magnetic pressure at 1 T", "B^2/2mu0", q.to(u.PA), 397887.3577, 1e-9)])


@pytest.mark.family("constants")
def test_sanity_copper_skin_depth():
    """Copper, sigma = 5.8e7 S/m, at 50 Hz: delta = sqrt(2/(omega·mu0·sigma)) = 66.1/sqrt(f) mm = 9.348 mm
    (Hayt & Buck, Engineering Electromagnetics). Checks the S/m and H/m units; the 66.1 constant has 3 figures."""
    omega = 2 * math.pi * 50 * u.RAD_PER_S
    delta_squared = 2 / (omega * (MU0_EXACT * u.H_PER_M) * (5.8e7 * u.S_PER_M))
    delta_mm = math.sqrt(delta_squared.to(u.M ** 2)) * 1000
    assert_all([("copper skin depth at 50 Hz", "66.1/sqrt(f) mm", delta_mm, 66.1 / math.sqrt(50), 1e-3)])


@pytest.mark.family("constants")
def test_sanity_unit_identities():
    """1 MPa·mm^2 = 1 N; 1 N·m = 1 J; 1 W·s = 1 J; 1 rpm = 2·pi/60 rad/s; 1 g·J/(kg·K) = 1e-3 J/K;
    1 T·A/m = 1 Pa (B·H is an energy density)."""
    assert_all([(what, "SI definition", got, ref, TOL_ALGEBRA)
                for what, got, ref in (("MPa·mm^2 in N", (u.MPA * u.MM ** 2).to(u.N), 1.0),
                                       ("N·m in J", u.NM.to(u.J), 1.0),
                                       ("W·s in J", (u.W * u.S).to(u.J), 1.0),
                                       ("rpm in rad/s", u.RPM.to(u.RAD_PER_S), 2 * math.pi / 60),
                                       ("g·J/(kg·K) in J/K", (u.GRAM * u.J_PER_KG_K).to(u.J_PER_K), 1e-3),
                                       ("T·A/m in Pa", (u.TESLA * u.A_PER_M).to(u.PA), 1.0))])


@pytest.mark.family("constants")
def test_sanity_dimension_errors_raise():
    """Mixing dimensions is rejected: m + s, expressing a torque in pascals, and a dimensional sinh argument."""
    with pytest.raises(u.DimensionError):
        _ = u.M + u.S
    with pytest.raises(u.DimensionError):
        u.NM.to(u.PA)
    with pytest.raises(u.DimensionError):
        (3.0 * u.MM).dimensionless()
```

- [ ] **Step 3: Run the sanity test**

Run: `cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_clamps_sweeps_units_reference_sanity.py -k sanity`
Expected: PASS, `18 passed`.

- [ ] **Step 4: Write the engine checks**

File: `reference/magcoupling-py/audit/tests/test_clamps.py` (create)
```python
"""Task 7: independent checks of the shaft-clamp sizing (engine clamps.py) against audit.references.clamp_ref.

One check per formula (one per root cause), each looping over every scenario and screw size with assert_all, so a
wrong formula yields one candidate that lists all of its wrong cells. Stage-wise where a formula takes an upstream
value (it is fed the engine's own upstream cell), so one wrong stage fails one check; end-to-end only for the
recommendation and the README claims.

Workbook layout rules that are design choices, not physics, are taken as given (not independently checkable):
counterbore = head + 0.5 mm, screw spacing = head + 1 mm, screw lengths in 2 mm steps, the one-/two-piece clamp
factors 0.8/1.0, and the aluminium allowables (head pressure, key bearing).
"""
from __future__ import annotations

import math
from typing import NamedTuple

import pytest

from audit.common import TOL_ALGEBRA, ScenarioCache, assert_all, defaults, flag_item, run, text_item, vary
from audit.references import clamp_ref as ref
# Workbook cell addresses only, used to label failures; no engine arithmetic is reused.
from magcoupling.clamps import TABLE_COLUMNS, TABLE_ROWS

CBORE_ALLOWANCE_MM = 0.5
HEAD_GAP_MM = 1.0
LENGTH_STEP_MM = 2.0

SCENARIOS = {
    "defaults": {},
    "boss22_len14.5": {"clamps.boss_od_mm": 22, "clamps.clamp_length_mm": 14.5},
    "al6061_cl10.9": {"clamps.alloy": 2, "clamps.screw_class": 2},
    "two_piece_oily_A4-70": {"clamps.clamp_type": 2, "clamps.friction": 0.10, "clamps.screw_class": 3},
    "strip_governs": {"clamps.alloy": 2, "clamps.engagement_x_d": 1.0},
}
CASES = ScenarioCache(SCENARIOS)
CLASS_TEXT = {1: "12.9", 2: "10.9", 3: "A4-70"}   # ClampInputs.screw_class selector codes
ALLOY_TEXT = {1: "7075-T6", 2: "6061-T6"}          # ClampInputs.alloy selector codes
SELECTION_CELLS = {
    "screws": "C49", "tightening_Nm": "C50", "hex_mm": "C51", "capacity_Nm": "C52", "sf_coupling": "C53",
    "head_check": "C54", "vent_port": "C55", "layout_offset_mm": "C72", "layout_pitch_mm": "C73",
    "layout_first_mm": "C74", "layout_cbore_dia_mm": "C75", "layout_cbore_depth_mm": "C76", "layout_grip_mm": "C77",
    "layout_tap_drill_mm": "C78", "layout_thread_avail_mm": "C79",
}


def _setup(inp, res) -> ref.ClampSetup:
    """Reference inputs built from the workbook inputs; the only upstream value is the cold-high torque."""
    c = inp.clamps
    return ref.ClampSetup(
        shaft_mm=inp.coupling.bore_mm, boss_od_mm=c.boss_od_mm, slit_mm=c.slit_mm, ligament_mm=c.ligament_mm,
        wall_min_mm=c.wall_out_mm, grip_min_mm=c.grip_min_mm, axial_margin_mm=c.axial_margin_mm,
        clamp_length_mm=c.clamp_length_mm, engagement_x_d=c.engagement_x_d, preload_fraction=c.preload_fraction,
        strip_sf=c.strip_sf, friction=c.friction,
        clamp_factor=c.factor_one_piece if c.clamp_type == 1 else c.factor_two_piece,
        nut_factor=c.nut_factor, screw_class=CLASS_TEXT[c.screw_class], alloy=ALLOY_TEXT[c.alloy],
        max_torque_Nm=res.metal.torque_cold_high_Nm, safety_factor=c.safety_factor,
        cbore_allowance_mm=CBORE_ALLOWANCE_MM, head_gap_mm=HEAD_GAP_MM, length_step_mm=LENGTH_STEP_MM)


def _cell(field: str, i: int) -> str:
    return f"Clamp screw sizes!{TABLE_COLUMNS[i]}{TABLE_ROWS[field]}"


def _na0(x):
    """The workbook prints 0 for a quantity that does not exist (e.g. grip when the head seat does not fit)."""
    return 0.0 if x is None else x


class Case(NamedTuple):
    tag: str            # "<scenario> <size>", for failure labels
    i: int              # size index 0..4 (table column C..G)
    inp: object
    res: object
    setup: ref.ClampSetup
    screw: ref.IsoScrew
    row: object         # engine table row
    geo: dict           # reference geometry


def _cases():
    """Every (scenario, screw size) pair with its engine row and reference geometry."""
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        setup = _setup(inp, res)
        for i, screw in enumerate(ref.ISO_SCREWS):
            yield Case(f"{sc} {screw.name}", i, inp, res, setup, screw, res.clamps.table[i],
                       ref.clamp_geometry(setup, screw))


def _pick_name(index: int):
    return ref.ISO_SCREWS[index - 1].name if index else None


# --------------------------------------------------------------------------- screw data (literature)
@pytest.mark.family("clamps")
def test_screw_catalogue_matches_iso():
    """d, coarse pitch (ISO 261), medium clearance hole (ISO 273), head diameter/height and hex key (ISO 4762),
    tap drill (DIN 336) equal the standard values for M2.5..M6."""
    table = CASES["defaults"][1].clamps.table
    assert_all([(f"{s.name} {f}", _cell(f, i), getattr(table[i], f), v, TOL_ALGEBRA)
                for i, s in enumerate(ref.ISO_SCREWS)
                for f, v in (("d_mm", s.d), ("pitch_mm", s.pitch), ("hole_mm", s.hole), ("head_mm", s.head_dk),
                             ("head_h_mm", s.head_k), ("hex_mm", s.hex_s), ("tap_drill_mm", s.tap_drill))])


@pytest.mark.family("clamps")
def test_stress_area_is_iso898():
    """As equals the ISO 898-1 formula (pi/4)·((d2 + d3)/2)^2 with d2, d3 from the ISO 68-1 profile and the ISO 261
    pitch, printed to 3 significant figures as ISO 898-1 tabulates it."""
    table = CASES["defaults"][1].clamps.table
    assert_all([(f"{s.name} stress area", _cell("As_mm2", i), table[i].As_mm2, ref.iso_stress_area(s.d, s.pitch),
                 TOL_ALGEBRA) for i, s in enumerate(ref.ISO_SCREWS)])


# --------------------------------------------------------------------------- geometry
@pytest.mark.family("clamps")
def test_clamp_geometry():
    """Screw offset, wall outside the hole, head seat, grip, thread available, engagement, counterbore and the
    geometry verdict, re-derived from the chord geometry of the boss (clamp_ref module docstring), for every
    scenario and size."""
    items = []
    for c in _cases():
        items += [(f"{c.tag} {f}", _cell(f, c.i), getattr(c.row, f), _na0(c.geo[f]), TOL_ALGEBRA)
                  for f in ("offset_mm", "wall_out_mm", "head_fits", "grip_mm", "thread_avail_mm", "engagement_req_mm",
                            "cbore_dia_mm", "cbore_depth_mm")]
        items.append((f"{c.tag} geometry_ok", _cell("geometry_ok", c.i), c.row.geometry_ok,
                      ref.geometry_ok(c.setup, c.geo), 0.0))
    assert_all(items)


@pytest.mark.family("clamps")
def test_screw_length_meets_engagement_and_stays_inside():
    """Row 34 (screw length), property check for every scenario and size whose head seat fits: the engine's
    under-head length must engage at least the required thread (engagement_x_d·d) past the head-side jaw and the
    open slit (engaged = length - grip - slit), and must end inside the far jaw (length <= grip + slit + thread
    available, the seat-to-far-OD chord). Each failure also lists which 2 mm-step lengths meet both rules ('none'
    means no standard length satisfies the engagement rule without protruding). Where the head does not fit the
    workbook prints 0. One check for the matrix: every case uses the same row-34 formula."""
    items = []
    for c in _cases():
        cell = _cell("length_mm", c.i)
        if not c.geo["head_fits"]:
            items.append((f"{c.tag} length when the head seat does not fit", cell, c.row.length_mm, 0.0, 0.0))
            continue
        L = c.row.length_mm
        engaged = ref.engagement_achieved(c.setup, c.geo, L)
        need = c.geo["engagement_req_mm"]
        longest = ref.max_length_inside(c.setup, c.geo)
        valid = ref.valid_screw_lengths(c.setup, c.geo)
        options = f"valid 2 mm-step lengths: {', '.join(f'{v:g}' for v in valid) or 'none'}"
        items.append(flag_item(f"{c.tag}: the {L:g} mm screw engages {engaged:.3f} mm >= required {need:.3f} mm "
                               f"({options})", cell, engaged >= need - ref.LENGTH_TOL_MM))
        items.append(flag_item(f"{c.tag}: the {L:g} mm screw ends inside the far jaw (<= {longest:.3f} mm)", cell,
                               L <= longest + ref.LENGTH_TOL_MM))
    assert_all(items)


@pytest.mark.family("clamps")
def test_length_ok_flag():
    """Row 35 (length OK): 1 exactly when the engine's own length ends inside the far jaw (stage-wise: fed the
    engine's length, so a wrong length does not also fail here). 0 where the head seat does not fit."""
    items = []
    for c in _cases():
        expected = ref.screw_tip_inside(c.setup, c.geo, c.row.length_mm) if c.geo["head_fits"] else 0
        items.append((f"{c.tag} length_ok for the {c.row.length_mm:g} mm screw", _cell("length_ok", c.i),
                      c.row.length_ok, expected, 0.0))
    assert_all(items)


# --------------------------------------------------------------------------- strength
@pytest.mark.family("clamps")
def test_stripping_capacity_not_above_min_material():
    """Row 22 (stripping-limited preload). The allowable force before the tapped aluminium strips must not exceed
    tau·A_n/SF with the FED-STD-H28/2B internal-thread shear area at the ISO 965-2 minimum-material limits (6g screw
    major diameter min, 6H pitch diameter max), the smallest area any in-tolerance 6H/6g pair has. One-sided: a
    smaller engine value is conservative and passes. The engine uses a constant 0.6·pi·d·Le; the minimum-material
    area is 0.570 (M2.5) to 0.647 (M6)·pi·d·Le. Evaluated at defaults: the ratio does not depend on the alloy or the
    engagement factor (both scale engine and reference alike; test_stripping_capacity_scales_with_shear_and_engagement
    checks how the engine uses them)."""
    inp, res = CASES["defaults"]
    setup = _setup(inp, res)
    items = []
    for i, s in enumerate(ref.ISO_SCREWS):
        le = setup.engagement_x_d * s.d
        tau = ref.AL_SHEAR_MPA[setup.alloy]
        area = ref.min_material_shear_area(s, le)
        cap = ref.stripping_capacity(tau, area, setup.strip_sf)
        eng = res.clamps.table[i].preload_strip_N
        eng_factor = eng * setup.strip_sf / (tau * math.pi * s.d * le)      # engine area / (pi·d·Le)
        items.append(flag_item(f"{s.name} stripping capacity {eng:.1f} N <= minimum-material {cap:.1f} N "
                               f"(ratio {eng / cap:.4f}; area / (pi·d·Le): engine {eng_factor:.4f}, "
                               f"minimum-material {area / (math.pi * s.d * le):.4f})",
                               _cell("preload_strip_N", i), eng <= cap * (1.0 + TOL_ALGEBRA)))
    assert_all(items)


@pytest.mark.family("clamps")
def test_stripping_capacity_scales_with_shear_and_engagement():
    """Row 22 across scenarios, scaling law: the stripping capacity is proportional to the alloy's shear strength
    (MMPDS 331/207 MPa) and to the engagement length, and inversely to the stripping SF, so
    capacity(scenario)/capacity(defaults) = (tau·Le/SF)(scenario)/(tau·Le/SF)(defaults), whatever the area factor."""
    d_inp, d_res = CASES["defaults"]
    d_setup = _setup(d_inp, d_res)
    items = []
    for c in _cases():
        ratio_ref = ((ref.AL_SHEAR_MPA[c.setup.alloy] * c.setup.engagement_x_d / c.setup.strip_sf)
                     / (ref.AL_SHEAR_MPA[d_setup.alloy] * d_setup.engagement_x_d / d_setup.strip_sf))
        items.append((f"{c.tag} stripping capacity over defaults", _cell("preload_strip_N", c.i),
                      c.row.preload_strip_N / d_res.clamps.table[c.i].preload_strip_N, ratio_ref, TOL_ALGEBRA))
    assert_all(items)


@pytest.mark.family("clamps")
def test_preload():
    """Rows 21 and 23: preload at the proof share = fraction·Sp·As with the ISO 898-1 tabulated As and the ISO 898-1 /
    ISO 3506-1 proof stress (Shigley Eq. 8-31: 75 % of proof for reusable joints); preload used = the smaller of
    that and the stripping capacity (stage-wise: the engine's row-22 value)."""
    items = []
    for c in _cases():
        strength = ref.preload_proof_share(c.setup.preload_fraction, ref.PROOF_STRESS_MPA[c.setup.screw_class],
                                           ref.iso_stress_area(c.screw.d, c.screw.pitch))
        items += [(f"{c.tag} preload at proof share", _cell("preload_strength_N", c.i), c.row.preload_strength_N,
                   strength, TOL_ALGEBRA),
                  (f"{c.tag} preload used", _cell("preload_N", c.i), c.row.preload_N,
                   min(strength, c.row.preload_strip_N), TOL_ALGEBRA)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_capacity_chain_given_preload():
    """Rows 24-27 and 31-33, fed the engine's preload: head pressure on the annulus and its verdict, torque per
    screw (line-contact bound mu·F·d times the clamp factor), screws needed, clamp torque, safety factor on the
    coupling torque and tightening torque K·F·d."""
    items = []
    for c in _cases():
        F = c.row.preload_N
        p = ref.head_bearing_pressure(F, c.screw.head_dk, c.screw.hole)
        t_per = ref.clamp_torque_per_screw(c.setup.friction, F, c.setup.shaft_mm, c.setup.clamp_factor)
        need = ref.screws_needed(c.setup.max_torque_Nm * c.setup.safety_factor, c.row.torque_per_screw_Nm)
        head_text = "OK" if p <= c.res.clamps.al_head_limit_MPa else "Use a hardened washer"
        items += [
            (f"{c.tag} head pressure", _cell("head_pressure_MPa", c.i), c.row.head_pressure_MPa, p, TOL_ALGEBRA),
            text_item(f"{c.tag} head check", _cell("head_check", c.i), c.row.head_check, head_text),
            (f"{c.tag} torque per screw", _cell("torque_per_screw_Nm", c.i), c.row.torque_per_screw_Nm, t_per,
             TOL_ALGEBRA),
            (f"{c.tag} screws needed", _cell("screws_needed", c.i), c.row.screws_needed, need, 0.0),
            (f"{c.tag} clamp torque", _cell("clamp_torque_Nm", c.i), c.row.clamp_torque_Nm, c.row.screws_needed * t_per,
             TOL_ALGEBRA),
            (f"{c.tag} SF on coupling torque", _cell("sf_coupling", c.i), c.row.sf_coupling,
             c.row.screws_needed * t_per / c.setup.max_torque_Nm, TOL_ALGEBRA),
            (f"{c.tag} tightening torque", _cell("tightening_Nm", c.i), c.row.tightening_Nm,
             ref.tightening_torque(c.setup.nut_factor, F, c.screw.d), TOL_ALGEBRA),
        ]
    assert_all(items)


# --------------------------------------------------------------------------- fit, verdicts, recommendation
@pytest.mark.family("clamps")
def test_screw_fit_and_works():
    """Rows 28-30 and 38: spacing, screws that fit (greedy placement with the axial margin at both ends), 'works'
    (geometry OK and needed <= fit, fed the engine's needed count) and the vent-port flag (hex key across corners
    below the M6 port's minor diameter)."""
    items = []
    for c in _cases():
        spacing = c.screw.head_dk + HEAD_GAP_MM
        fit = ref.screws_fit(c.setup.clamp_length_mm, c.setup.axial_margin_mm, c.geo["cbore_dia_mm"], spacing)
        works = 1 if (ref.geometry_ok(c.setup, c.geo) == 1 and c.row.screws_needed <= fit) else 0
        items += [
            (f"{c.tag} screw spacing", _cell("pitch_axial_mm", c.i), c.row.pitch_axial_mm, spacing, TOL_ALGEBRA),
            (f"{c.tag} screws that fit", _cell("screws_fit", c.i), c.row.screws_fit, fit, 0.0),
            (f"{c.tag} works", _cell("works", c.i), c.row.works, works, 0.0),
            (f"{c.tag} vent port ok", _cell("vent_port_ok", c.i), c.row.vent_port_ok,
             1 if ref.hex_key_passes_port(c.screw.hex_s) else 0, 0.0),
        ]
    assert_all(items)


@pytest.mark.family("clamps")
def test_recommendation():
    """Shaft clamps C47-C49, end to end from the reference (reference preload, count, fit): the recommended size is
    the first that works, with the same screw count and class. The screw length inside the text is the subject of
    test_screw_length_meets_engagement_and_stays_inside, so the text is built with the engine's own length."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c = res.clamps
        idx, rows = ref.recommend(_setup(inp, res))
        items.append(text_item(f"{sc} recommended size", "Shaft clamps!C47", str(_pick_name(c.index)),
                               str(_pick_name(idx))))
        if idx and c.index == idx:
            length = c.table[idx - 1].length_mm
            text = f"ISO 4762 {ref.ISO_SCREWS[idx - 1].name} x {length:g}, class {CLASS_TEXT[inp.clamps.screw_class]}"
            items += [text_item(f"{sc} recommendation text", "Shaft clamps!C48", c.recommended, text),
                      (f"{sc} screws per clamp", "Shaft clamps!C49", c.screws, rows[idx - 1]["screws_needed"], 0.0)]
        elif not idx:
            items.append(text_item(f"{sc} recommendation text", "Shaft clamps!C48", c.recommended,
                                   "None: enlarge the boss or the clamp length"))
    assert_all(items)


@pytest.mark.family("clamps")
def test_selected_size_summary_and_layout():
    """Shaft clamps C50-C55 and C72-C81 for the recommended size: tightening torque, capacity and SF (fed the
    engine's preload), head check, hex key, vent-port text and cut layout (offset, spacing, first screw centred in
    the clamp length with its counterbore keeping the axial margin, counterbore, grip, tap drill, thread available).
    With no recommendation every selection and layout cell must be blank. The relief-cut depth is boss OD - hinge."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, ci = res.clamps, inp.clamps
        relief = (f"{ci.relief_mm:.1f} mm wide at {ci.clamp_length_mm:.1f} mm from the free end, "
                  f"{ci.boss_od_mm - ci.hinge_mm:.1f} mm deep from the slit side")
        items.append(text_item(f"{sc} relief cut", "Shaft clamps!C81", c.layout_relief, relief))
        if not c.index:
            items += [text_item(f"{sc} blank {f} when nothing fits", f"Shaft clamps!{cell}", str(getattr(c, f)), "")
                      for f, cell in SELECTION_CELLS.items()]
            continue
        i = c.index - 1
        s, row = ref.ISO_SCREWS[i], c.table[i]
        setup = _setup(inp, res)
        geo = ref.clamp_geometry(setup, s)
        n = row.screws_needed
        t_per = ref.clamp_torque_per_screw(setup.friction, row.preload_N, setup.shaft_mm, setup.clamp_factor)
        spacing = s.head_dk + HEAD_GAP_MM
        first = (ci.clamp_length_mm - (n - 1) * spacing) / 2.0
        vent = "Yes: the key fits the 4 mm limit" if ref.hex_key_passes_port(s.hex_s) else "No: key too large"
        head = ("OK" if ref.head_bearing_pressure(row.preload_N, s.head_dk, s.hole) <= c.al_head_limit_MPa
                else "Use a hardened washer")
        items += [
            (f"{sc} tightening torque", "Shaft clamps!C50", c.tightening_Nm,
             ref.tightening_torque(setup.nut_factor, row.preload_N, s.d), TOL_ALGEBRA),
            (f"{sc} clamp capacity", "Shaft clamps!C52", c.capacity_Nm, n * t_per, TOL_ALGEBRA),
            (f"{sc} SF on coupling torque", "Shaft clamps!C53", c.sf_coupling, n * t_per / setup.max_torque_Nm,
             TOL_ALGEBRA),
            text_item(f"{sc} head check", "Shaft clamps!C54", c.head_check, head),
            (f"{sc} hex key", "Shaft clamps!C51", c.hex_mm, s.hex_s, 0.0),
            text_item(f"{sc} vent port", "Shaft clamps!C55", c.vent_port, vent),
            (f"{sc} layout offset", "Shaft clamps!C72", c.layout_offset_mm, geo["offset_mm"], TOL_ALGEBRA),
            (f"{sc} layout spacing", "Shaft clamps!C73", c.layout_pitch_mm, spacing, TOL_ALGEBRA),
            (f"{sc} layout first screw", "Shaft clamps!C74", c.layout_first_mm, first, TOL_ALGEBRA),
            flag_item(f"{sc} first counterbore keeps the {ci.axial_margin_mm:g} mm axial margin "
                      f"(edge at {c.layout_first_mm - geo['cbore_dia_mm'] / 2:.3f} mm)", "Shaft clamps!C74",
                      c.layout_first_mm - geo["cbore_dia_mm"] / 2 >= ci.axial_margin_mm - ref.LENGTH_TOL_MM),
            (f"{sc} layout counterbore dia", "Shaft clamps!C75", c.layout_cbore_dia_mm, geo["cbore_dia_mm"],
             TOL_ALGEBRA),
            (f"{sc} layout counterbore depth", "Shaft clamps!C76", c.layout_cbore_depth_mm, geo["cbore_depth_mm"],
             TOL_ALGEBRA),
            (f"{sc} layout grip", "Shaft clamps!C77", c.layout_grip_mm, geo["grip_mm"], TOL_ALGEBRA),
            (f"{sc} layout tap drill", "Shaft clamps!C78", c.layout_tap_drill_mm, s.tap_drill, TOL_ALGEBRA),
            (f"{sc} layout thread available", "Shaft clamps!C79", c.layout_thread_avail_mm, geo["thread_avail_mm"],
             TOL_ALGEBRA),
        ]
    assert_all(items)


@pytest.mark.family("clamps")
def test_head_check_threshold():
    """Row 25 verdict: 'OK' exactly when the head pressure <= the aluminium limit (M4 row): OK at equality,
    'Use a hardened washer' when the limit is one ulp lower."""
    p = run().clamps.table[2].head_pressure_MPa
    path = "materials.aluminium.al7075.head_pressure_limit_MPa"
    at = run(vary(defaults(), {path: p}))
    below = run(vary(defaults(), {path: math.nextafter(p, -math.inf)}))
    assert_all([
        text_item("limit = M4 head pressure", _cell("head_check", 2), at.clamps.table[2].head_check, "OK"),
        text_item("limit one ulp below the M4 head pressure", _cell("head_check", 2), below.clamps.table[2].head_check,
                  "Use a hardened washer"),
    ])


# --------------------------------------------------------------------------- loads, key, joint
@pytest.mark.family("clamps")
def test_design_torque_factor_and_material_constants():
    """Shaft clamps C15, C17, C22, C24, C28, C44: clamp design torque = safety factor × the cold-high coupling
    torque (README: the clamp alone holds twice it); clamp factor by type; boss radius; aluminium shear (MMPDS) and
    screw proof stress (ISO 898-1 / 3506-1)."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, setup = res.clamps, _setup(inp, res)
        items += [
            (f"{sc} max torque = cold high with +variation", "Shaft clamps!C15 / Metal design!C10", c.max_torque_Nm,
             res.metal.torque_cold_high_Nm, TOL_ALGEBRA),
            (f"{sc} torque the clamp must hold", "Shaft clamps!C17", c.required_Nm,
             setup.safety_factor * setup.max_torque_Nm, TOL_ALGEBRA),
            (f"{sc} clamp factor used", "Shaft clamps!C22", c.clamp_factor, setup.clamp_factor, TOL_ALGEBRA),
            (f"{sc} boss radius", "Shaft clamps!C44", c.boss_radius_mm, setup.boss_od_mm / 2, TOL_ALGEBRA),
            (f"{sc} {setup.alloy} shear strength", "Shaft clamps!C24", c.al_shear_MPa, ref.AL_SHEAR_MPA[setup.alloy],
             TOL_ALGEBRA),
            (f"{sc} class {setup.screw_class} proof stress", "Shaft clamps!C28", c.screw_proof_MPa,
             ref.PROOF_STRESS_MPA[setup.screw_class], TOL_ALGEBRA),
        ]
    assert_all(items)


@pytest.mark.family("clamps")
def test_key_backup():
    """Shaft clamps C60-C61: key bearing pressure at the clamp design torque, p = 2T/(d·h·L) in SI, and
    allowable over actual."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, ci = res.clamps, inp.clamps
        p = ref.key_bearing_pressure_MPa(c.required_Nm, inp.coupling.bore_mm, ci.key_contact_mm, ci.clamp_length_mm)
        items += [(f"{sc} key bearing pressure", "Shaft clamps!C60", c.key_pressure_MPa, p, TOL_ALGEBRA),
                  (f"{sc} key allowable over actual", "Shaft clamps!C61", c.key_sf, c.al_key_allow_MPa / p,
                   TOL_ALGEBRA)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_adapter_joint():
    """Shaft clamps C67-C69, stage-wise: the allowable preload per M3 is the M3 row's preload used (Clamp screw
    sizes!D23, itself checked by test_preload); slip torque mu·n·F·D_bc/2 and its ratio to the clamp design
    torque."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        c, ci = res.clamps, inp.clamps
        t_joint = ref.flange_friction_torque_Nm(ci.joint_friction, ci.joint_screws, c.joint_preload_N,
                                                ci.joint_bolt_circle_mm)
        items += [(f"{sc} allowable preload per M3", "Shaft clamps!C67", c.joint_preload_N, c.table[1].preload_N,
                   TOL_ALGEBRA),
                  (f"{sc} joint slip torque", "Shaft clamps!C68", c.joint_torque_Nm, t_joint, TOL_ALGEBRA),
                  (f"{sc} joint torque over clamp design torque", "Shaft clamps!C69", c.joint_sf,
                   t_joint / c.required_Nm, TOL_ALGEBRA)]
    assert_all(items)


# --------------------------------------------------------------------------- README claims
def _pick_at(boss_mm: float, length_mm: float):
    """(reference pick, reference screws, engine pick, engine screws, geometry-OK sizes by reference and engine)."""
    inp = vary(defaults(), {"clamps.boss_od_mm": boss_mm, "clamps.clamp_length_mm": length_mm})
    res = run(inp)
    setup = _setup(inp, res)
    idx, rows = ref.recommend(setup)
    ref_fit = [s.name for s, r in zip(ref.ISO_SCREWS, rows) if r["geometry_ok"]]
    eng_fit = [s.name for s, r in zip(ref.ISO_SCREWS, res.clamps.table) if r.geometry_ok]
    return (_pick_name(idx), rows[idx - 1]["screws_needed"] if idx else 0, _pick_name(res.clamps.index),
            res.clamps.screws if res.clamps.index else 0, ref_fit, eng_fit)


@pytest.mark.family("clamps")
def test_readme_22mm_two_m3_need_14_5mm():
    """README 'What the default design currently shows' and the Shaft clamps!C35 help: at a 22 mm boss two M3
    screws need a 14.5 mm clamp. Reference and engine agree on M3 x 2 at 14.5 mm and on 'none' at 14.4 mm."""
    items = []
    for length, size, screws in ((14.5, "M3", 2), (14.4, None, 0)):
        ref_pick, ref_n, eng_pick, eng_n, _, _ = _pick_at(22, length)
        items += [flag_item(f"22 mm boss, {length} mm clamp: reference pick {ref_pick} x {ref_n}, "
                            f"README {size} x {screws}", "README / Shaft clamps!C35",
                            ref_pick == size and ref_n == screws),
                  text_item(f"22 mm boss, {length} mm clamp: engine pick", "Shaft clamps!C48", str(eng_pick),
                            str(size)),
                  (f"22 mm boss, {length} mm clamp: engine screws", "Shaft clamps!C49", eng_n, screws, 0.0)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_readme_22mm_only_m3_fits():
    """README ('At 22 mm only M3 fits') and the Shaft clamps!C35 help say only M3 fits a 22 mm boss. Checked two
    ways: the sizes whose geometry passes at 22 mm (wall, head seat, grip, thread), and the first pick at a clamp
    long enough for three screws (18 mm), which the claim implies is M3. The engine is compared with the reference
    in the same check, so a disagreement between them would show here too."""
    _, _, _, _, ref_fit, eng_fit = _pick_at(22, 10)
    ref_pick, ref_n, eng_pick, eng_n, _, _ = _pick_at(22, 18)
    assert_all([
        flag_item(f"sizes whose geometry passes at a 22 mm boss: reference {ref_fit}, README ['M3']",
                  "README / Shaft clamps!C35", ref_fit == ["M3"]),
        text_item("sizes whose geometry passes at a 22 mm boss: engine vs reference",
                  f"Clamp screw sizes!C{TABLE_ROWS['geometry_ok']}:G{TABLE_ROWS['geometry_ok']}", str(eng_fit),
                  str(ref_fit)),
        flag_item(f"first pick at a 22 mm boss, 18 mm clamp: reference {ref_pick} x {ref_n}, README implies M3",
                  "README / Shaft clamps!C35", ref_pick == "M3"),
        text_item(f"first pick at a 22 mm boss, 18 mm clamp: engine ({eng_pick} x {eng_n}) vs reference",
                  "Shaft clamps!C48:C49", f"{eng_pick} x {eng_n}", f"{ref_pick} x {ref_n}"),
    ])
```

File: `reference/magcoupling-py/audit/tests/test_sweeps.py` (create)
```python
"""Task 7: every gap-sweep and pole-sweep row against an independent single evaluation of the README torque model
(audit.references.sweep_ref: Task 3's planar formulas and Task 4's coupling geometry, SI units, no engine
arithmetic), including the fit/status flags.

Upstream values taken from the engine as data, not re-derived here: the magnet dimensions and Br20 resolved from the
library (res.model.*), the Calculator's calibration factor (res.model.f_cal) and the Calculator's corner gap
(res.model.corner_gap_mm; Task 4 checks both of the latter's geometry).
"""
from __future__ import annotations

import math

import pytest

from audit.common import TOL_ALGEBRA, ScenarioCache, assert_all, defaults, flag_item, run, text_item, vary
from audit.references import sweep_ref as ref
# Sweep grids (the rows to check) and workbook column letters (failure labels); no engine arithmetic is reused.
from magcoupling.sweeps import GAP_SWEEP_CORNER_GAPS_MM, POLE_SWEEP_POLES, SWEEP_COLUMNS

SCENARIOS = {
    "defaults": {},
    "no_iron": {"coupling.backiron": 0},                 # free-space S_n, measured calibration factor in the gap sweep
    "arcs": {"coupling.faceted": 0},
    "npole16": {"coupling.npole": 16},                   # inner flats too narrow at the current hub apothem
    "wide_outer": {"coupling.magnets.part_outer": "", "coupling.magnets.manual_outer_width_mm": 12.0},
    "hot80": {"coupling.op_temp_C": 80},
}
CASES = ScenarioCache(SCENARIOS)
NUMERIC_FIELDS = [f for f in SWEEP_COLUMNS if f not in ("variable", "status")]
FIRST_ROW = 6   # workbook row of the first sweep row


def _setup(inp, res) -> ref.SweepSetup:
    ci, md, m = inp.coupling, inp.metal, res.model
    return ref.SweepSetup(
        faceted=ci.faceted, backiron=ci.backiron, t_i_mm=m.inner_thickness_mm, w_i_mm=m.inner_width_mm,
        t_o_mm=m.outer_thickness_mm, w_o_mm=m.outer_width_mm, length_mm=min(m.inner_length_mm, m.outer_length_mm),
        br_i20_T=m.inner_br_T, br_o20_T=m.outer_br_T, alpha_per_C=inp.calibration.alpha_br_per_C,
        op_temp_C=ci.op_temp_C,
        bond_inner_mm=md.bond_inner_mm, bond_outer_mm=md.bond_outer_mm, cup_wall_corner_mm=md.cup_wall_corner_mm,
        bore_mm=ci.bore_mm, keyway_mm=ci.keyway_depth_mm, c_end=ci.c_end, mu0=ci.mu0, gear_ratio=ci.gear_ratio,
        gear_eff=ci.gear_efficiency,
        required_floor_Nm=ref.required_floor(ci.drive_torque_Nm, ci.drive_safety_factor, md.required_min_Nm),
        max_diameter_mm=md.max_diameter_mm)


def _row_items(sheet: str, scenario: str, j: int, row, variable: float, expected: dict) -> list:
    """The row's variable, every numeric column at TOL_ALGEBRA and the status text."""
    r = FIRST_ROW + j
    items = [(f"{scenario} row {j} variable", f"{sheet}!B{r}", row.variable, variable, 0.0)]
    items += [(f"{scenario} row {j} {f}", f"{sheet}!{SWEEP_COLUMNS[f]}{r}", getattr(row, f), expected[f], TOL_ALGEBRA)
              for f in NUMERIC_FIELDS]
    items.append(text_item(f"{scenario} row {j} status", f"{sheet}!AA{r}", row.status, expected["status"]))
    return items


@pytest.mark.family("sweeps")
def test_gap_sweep_rows():
    """Every gap-sweep row, in every scenario, equals a single evaluation at that corner gap with the current poles
    and hub apothem and the Calculator's calibration factor: all 25 columns and the status."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        setup = _setup(inp, res)
        for j, gap in enumerate(GAP_SWEEP_CORNER_GAPS_MM):
            expected = ref.sweep_row(setup, inp.coupling.npole, inp.coupling.inner_back_apothem_mm, gap,
                                     res.model.f_cal)
            items += _row_items("Gap sweep", sc, j, res.gap_sweep[j], gap, expected)
    assert_all(items)


@pytest.mark.family("sweeps")
def test_pole_sweep_rows_given_apothem():
    """Every pole-sweep row, in every scenario, equals a single evaluation at that pole count and the row's own inner
    apothem, at the Calculator's corner gap and the original calibration factor (the measured correction does not
    transfer). The apothem rule itself is checked by the two apothem tests below."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        setup = _setup(inp, res)
        for j, npole in enumerate(POLE_SWEEP_POLES):
            row = res.pole_sweep[j]
            expected = ref.sweep_row(setup, npole, row.inner_apothem_mm, res.model.corner_gap_mm,
                                     inp.calibration.f_cal_original)
            items += _row_items("Pole sweep", sc, j, row, npole, expected)
    assert_all(items)


@pytest.mark.family("sweeps")
def test_pole_sweep_apothem_rule_as_stated():
    """Pole sweep column C against the rule as the sheet note states it (engine sweeps.py module and pole_sweep
    docstrings, ported from the workbook): 'the smallest inner apothem that fits the block width (+0.05 mm) and the
    keyed bore wall (2.5 mm)', i.e. max(w_i/(2·tan(pi/N)) + 0.05, bore/2 + keyway + 2.5), with the wall measured from
    the block-back apothem because the note does not mention the bondline. Whether that wall matches the
    Calculator's own wall definition is test_pole_sweep_apothem_keeps_calculator_hub_wall."""
    items = []
    for sc in SCENARIOS:
        inp, res = CASES[sc]
        ci = inp.coupling
        for j, npole in enumerate(POLE_SWEEP_POLES):
            items.append((f"{sc} N={npole} inner apothem", f"Pole sweep!C{FIRST_ROW + j}",
                          res.pole_sweep[j].inner_apothem_mm,
                          ref.stated_min_inner_apothem(npole, res.model.inner_width_mm, ci.bore_mm, ci.keyway_depth_mm),
                          TOL_ALGEBRA))
    assert_all(items)


@pytest.mark.family("sweeps")
def test_pole_sweep_apothem_keeps_calculator_hub_wall():
    """Definition consistency. The Calculator defines the hub wall past the keyway as apothem - inner bondline -
    bore/2 - keyway (Calculator!C53, engine field hub_wall_past_key_mm, help text 'Keep ≥ ~2.5 mm'; Task 4 verifies
    that definition, and Task 4's coupling_geometry computes it here). The pole sweep's rule uses the same 2.5 mm
    but measures it from the block-back apothem, without the bondline. This check asks whether each pole-sweep
    apothem keeps the Calculator's C53 wall at 2.5 mm; a shortfall equal to the bondline means the two sheets use
    different wall definitions, not an arithmetic slip. Every pole count in one check (one root cause)."""
    inp, res = CASES["defaults"]
    setup = _setup(inp, res)
    items = []
    for j, npole in enumerate(POLE_SWEEP_POLES):
        a = res.pole_sweep[j].inner_apothem_mm
        wall = ref.geometry_at_corner_gap(setup, npole, a, res.model.corner_gap_mm).hub_wall_past_key
        items.append(flag_item(f"N={npole}: apothem {a:.3f} mm leaves {wall:.3f} mm of hub wall past the keyway "
                               f"(Calculator C53 definition, bondline {setup.bond_inner_mm:g} mm) >= "
                               f"{ref.KEYED_WALL_MM:g} mm", f"Pole sweep!C{FIRST_ROW + j} / Calculator!C53",
                               wall >= ref.KEYED_WALL_MM - 1e-9))
    assert_all(items)


@pytest.mark.family("sweeps")
def test_sweep_status_covers_every_branch():
    """Across the scenarios the sweeps reach all five status texts, so every status branch was compared above."""
    seen = {r.status for sc in SCENARIOS for r in CASES[sc][1].gap_sweep + CASES[sc][1].pole_sweep}
    missing = sorted({ref.STATUS_INNER_NARROW, ref.STATUS_OUTER_NARROW, ref.STATUS_OD, ref.STATUS_BELOW_MIN,
                      ref.STATUS_NOMINAL} - seen)
    assert_all([flag_item(f"every status branch reached (missing: {missing})", "Gap sweep!AA, Pole sweep!AA",
                          not missing)])


@pytest.mark.family("sweeps")
def test_sweep_below_hot_minimum_threshold():
    """Sweep status 'below hot minimum' exactly when the row's nominal pull-out < the required floor: pinned on the
    1.0 mm corner-gap row (fits, inside the envelope) with the floor equal to the row's pull-out and one ulp above."""
    j = GAP_SWEEP_CORNER_GAPS_MM.index(1)
    x = run().gap_sweep[j].pullout_op_Nm
    at = run(vary(defaults(), {"metal.required_min_Nm": x}))
    above = run(vary(defaults(), {"metal.required_min_Nm": math.nextafter(x, math.inf)}))
    cell = f"Gap sweep!AA{FIRST_ROW + j}"
    assert_all([text_item("floor = row pull-out", cell, at.gap_sweep[j].status, ref.STATUS_NOMINAL),
                text_item("floor one ulp above the row pull-out", cell, above.gap_sweep[j].status,
                          ref.STATUS_BELOW_MIN)])
```

File: `reference/magcoupling-py/audit/tests/test_units_verdicts.py` (create)
```python
"""Task 7: constants, dimensional consistency, scaling laws and text-verdict thresholds.

Dimensional checks use audit.references.units_ref, a quantity algebra that rejects mixed dimensions:
- formula level: each formula in FORMULAS, written as the README / spec / literature states it, has the dimensions
  of its result (test_formula_is_dimensionally_consistent; independent of the engine code). Covered (FORMULAS groups):
  torque chain (wave-number argument k·t, harmonic amplitude B_n, Br(T), S_n steel and free, shear tau_n, torque
  sum, end factor, calibration correction, sweep-row torque); Metal design (hot/cold torque band, gearbox input
  T/(i·eta), clearance stack); Materials (back-iron thickness, plating offsets); clamps (preload, stripping, torque
  per screw, tightening torque, head pressure, key pressure, joint torque); thermal network (heat capacity, time
  constant, steady rise, event rise, time to limit, critical drag, slip power); slip losses (skin depth, half-space,
  thin shell, thin strip); demagnetization (knee, load line, calibration offset); adhesive (Volkersen lambda and
  shear, bond shear, centrifugal force); masses (density, inertia). Formulas not in FORMULAS are not
  dimension-checked here;
- engine level: the numeric unit factors the engine applies in the torque chain equal the SI conversions
  (test_torque_chain_unit_factors); elsewhere the owning task's SI re-derivation at TOL_ALGEBRA verifies them;
- scaling laws that follow from dimensions: geometric similarity of the Calculator chain and the thermal network's
  response to its conductance.
Verdict checks pin each comparison operator at its exact boundary with math.nextafter.

Ownership, as the other tasks' drafts assign it: Task 7 owns the verdict thresholds Metal design C11 and C37 and the
Calculator verdict C102 (the values they compare are checked by Tasks 3 and 4), the NdFeB density use in
Temperature design C82 (Task 5 consumes it), and units/scaling on Temperature design C141-C153 (Task 6 owns
their values). Calculator C110 (magnet mass) belongs to Task 4.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from audit.common import MU0_EXACT, TOL_ALGEBRA, assert_all, defaults, flag_item, run, text_item, vary
from audit.references import units_ref as u
# The constants under test.
from magcoupling.constants import MU0, NDFEB_DENSITY_G_MM3


# --------------------------------------------------------------------------- constants
@pytest.mark.family("constants")
def test_mu0_rounded_constant():
    """The workbook's 1.256637e-6 is 4·pi·1e-7 to 7 significant figures: allowed error is half a unit in the 7th
    figure, 0.5e-12/1.256637e-6 = 3.98e-7 relative. (Torque scales with 1/mu0, so the rounding moves it by the same
    relative amount; re-derivations at TOL_ALGEBRA must use the engine's mu0 input, not MU0_EXACT.)"""
    tol = 0.5e-12 / 1.256637e-6
    d = defaults()
    assert_all([("MU0 constant", "constants.MU0", MU0, MU0_EXACT, tol),
                ("vacuum permeability input", "Calculator!C43", d.coupling.mu0, MU0_EXACT, tol),
                ("vacuum permeability input", "Calibration!C25", d.calibration.mu0, MU0_EXACT, tol)])


@pytest.mark.family("constants")
def test_ndfeb_density_constant():
    """0.0075 g/mm^3 is 7.5 g/cm^3, the nominal density of sintered NdFeB (supplier datasheets span about
    7.4-7.6 g/cm^3, so tol = 0.1/7.5)."""
    got = (NDFEB_DENSITY_G_MM3 * u.GRAM / u.MM ** 3).to(u.GRAM / (10 * u.MM) ** 3)
    assert_all([("NdFeB density in g/cm^3", "constants.NDFEB_DENSITY_G_MM3", got, 7.5, 0.1 / 7.5)])


@pytest.mark.family("constants")
def test_block_mass_uses_ndfeb_density():
    """Temperature design C82 (block mass, which temperature.py computes with a hard-coded 0.0075 rather than the
    shared constant) = inner block volume × 7.5 g/cm^3, computed with units."""
    res = run()
    m = res.model
    rho = 7.5 * u.GRAM / (10 * u.MM) ** 3
    volume = (m.inner_length_mm * u.MM) * (m.inner_width_mm * u.MM) * (m.inner_thickness_mm * u.MM)
    assert_all([("block mass", "Temperature design!C82", res.temperature.adhesive.block_mass_g,
                 (volume * rho).to(u.GRAM), TOL_ALGEBRA)])


# --------------------------------------------------------------------------- formula-level dimensions
#: One of each unit, named by physical quantity; only the dimensions matter.
Q = SimpleNamespace(
    B=1.0 * u.TESLA, mu0=1.0 * u.H_PER_M, H=1.0 * u.A_PER_M, k=1.0 / u.M, length=1.0 * u.M, area=1.0 * u.M ** 2,
    F=1.0 * u.N, T=1.0 * u.NM, stress=1.0 * u.PA, P=1.0 * u.W, time=1.0 * u.S, dT=1.0 * u.K, alpha=1.0 * u.PER_K,
    omega=1.0 * u.RAD_PER_S, rpm=1.0 * u.RPM, v=1.0 * u.M / u.S, f=1.0 / u.S, sigma=1.0 * u.S_PER_M,
    mass=1.0 * u.KG, rho=1.0 * u.KG / u.M ** 3, c=1.0 * u.J_PER_KG_K, C=1.0 * u.J_PER_K, G=1.0 * u.W_PER_K)


def _f(fn, arg: u.Quantity) -> u.Quantity:
    """fn (sin, sinh, exp, tanh, log) of an argument that must be dimensionless; raises DimensionError otherwise."""
    return fn(arg.dimensionless()) * u.ONE


# (id, formula as stated, source, builder of the result from Q, unit of the result)
FORMULAS = [
    # torque model (README 'Torque model')
    ("torque-wave-number", "k·t with k = n·(poles/2)/R_gap", "README step 2",
     lambda q: (3 * (10 / 2) / q.length) * q.length, u.ONE),
    ("torque-harmonic-amplitude", "B_n = 4·Br/(n·pi)·sin(n·pi·fill/2), fill = block width/pole pitch",
     "README step 1",
     lambda q: 4 * q.B / (3 * math.pi) * _f(math.sin, 3 * math.pi * (q.length / q.length) / 2), u.TESLA),
    ("torque-br-temperature", "Br(T) = Br20·(1 + alpha·(T - 20)), alpha [1/K]", "README step 4",
     lambda q: q.B * (u.ONE + q.alpha * (q.dT - 20 * q.dT)), u.TESLA),
    ("torque-s-steel", "S_n = sinh(k·t_i)·sinh(k·t_o)/sinh(k·(t_i + t_o + g))", "README step 2",
     lambda q: _f(math.sinh, q.k * q.length) ** 2 / _f(math.sinh, q.k * (q.length + q.length + q.length)), u.ONE),
    ("torque-s-free", "S_n = (1 - e^(-k·t_i))·(1 - e^(-k·t_o))·e^(-k·g)/2", "README step 2",
     lambda q: (u.ONE - _f(math.exp, -q.k * q.length)) ** 2 * _f(math.exp, -q.k * q.length) / 2, u.ONE),
    ("torque-shear", "tau_n = B_in·B_on/(2·mu0)·S_n·sin(n·pi/2)", "README step 2",
     lambda q: q.B * q.B / (2 * q.mu0) * math.sin(math.pi / 2), u.PA),
    ("torque-sum", "T = sum(tau_n)·2·pi·R_gap^2·L", "README step 3",
     lambda q: q.stress * 2 * math.pi * q.length ** 2 * q.length, u.NM),
    ("torque-end-factor", "1 - c_end·pole pitch/L", "README step 3",
     lambda q: u.ONE - 0.7 * q.length / q.length, u.ONE),
    ("torque-calibration-correction", "f_cal = 0.95·T_measured/T_model", "README step 3 (Calibration!C9)",
     lambda q: 0.95 * q.T / q.T, u.ONE),
    ("sweep-row-torque", "T_row = sum(tau_n)·2·pi·R_gap^2·L·f_end·f_cal",
     "Gap sweep / Pole sweep rows (README steps 1-3)",
     lambda q: q.stress * 2 * math.pi * q.length ** 2 * q.length * (u.ONE - 0.7 * q.length / q.length) * 0.95, u.NM),
    # metal design and materials (README 'Metal design', 'Materials')
    ("metal-torque-band", "T_band = T·(1 ± v)·(Br(T_2)/Br(T_1))^2 (same dimensions for either sign)",
     "README 'Metal design', torque range",
     lambda q: q.T * (u.ONE - 0.15 * u.ONE) * (q.B / q.B) ** 2, u.NM),
    ("metal-gearbox-input", "T_in = T_out/(i·eta)", "power balance (Calculator!C99; Metal design!C156, C158)",
     lambda q: q.T / (5 * 0.95), u.NM),
    ("metal-clearance-stack", "c = corner gap - sleeve - liner - beddings - allowances",
     "README 'Metal design', running clearance",
     lambda q: q.length - 0.1 * q.length - 0.2 * q.length - 2 * (0.025 * q.length) - 0.05 * q.length, u.M),
    ("materials-backiron-thickness", "t = B·tau_p/(pi·B_sat)", "Hanselman ch. 4 (Calculator!C104)",
     lambda q: q.B * q.length / (math.pi * q.B), u.M),
    ("materials-plating-offset", "machined = finished ± surfaces·t_plate",
     "README 'Materials', plating (Materials!C27-C30)",
     lambda q: 10 * q.length - 2 * (0.015 * q.length), u.M),
    # clamps (README 'Shaft clamps'; clamp_ref sources)
    ("clamp-preload", "F = fraction·Sp·As", "ISO 898-1",
     lambda q: 0.75 * q.stress * q.area, u.N),
    ("clamp-stripping", "F = tau·A_n/SF, A_n = pi·n·Le·Ds·(1/(2n) + tan30·(Ds - En)), n = 1/P", "FED-STD-H28/2B",
     lambda q: q.stress * (math.pi * q.k * q.length * q.length * (1 / (2 * q.k) + 0.57735 * (q.length - q.length))),
     u.N),
    ("clamp-torque-per-screw", "T = mu·F·d·clamp factor", "README",
     lambda q: 0.15 * q.F * q.length * 0.8, u.NM),
    ("clamp-tightening", "T = K·F·d", "Shigley Eq. 8-27",
     lambda q: 0.2 * q.F * q.length, u.NM),
    ("clamp-head-pressure", "p = F/((pi/4)·(d_k^2 - d_h^2))", "bearing annulus",
     lambda q: q.F / (math.pi / 4 * (q.area - q.area / 4)), u.PA),
    ("clamp-key-pressure", "p = 2T/(d·h·L)", "Shigley, keys",
     lambda q: 2 * q.T / (q.length * q.length * q.length), u.PA),
    ("clamp-joint-torque", "T = mu·n·F·D_bc/2", "flange friction",
     lambda q: 0.15 * 4 * q.F * q.length / 2, u.NM),
    # thermal network (README 'Thermal network'; Incropera ch. 5)
    ("thermal-heat-capacity", "C = sum(m·c)", "README",
     lambda q: q.mass * q.c, u.J_PER_K),
    ("thermal-time-constant", "tau = C/G", "Incropera",
     lambda q: q.C / q.G, u.S),
    ("thermal-steady-rise", "dT = P/G", "Incropera",
     lambda q: q.P / q.G, u.K),
    ("thermal-event-rise", "dT = P·t_event/C", "README",
     lambda q: q.P * q.time / q.C, u.K),
    ("thermal-time-to-limit", "t = -tau·ln(1 - dT_allow/dT_steady)", "Incropera",
     lambda q: -(q.C / q.G) * _f(math.log, u.ONE - 0.5 * q.dT / q.dT), u.S),
    ("thermal-critical-drag", "T = (T_limit - T_start)·G/omega", "README",
     lambda q: q.dT * q.G / q.omega, u.NM),
    ("thermal-slip-power", "P = T·omega (omega from rpm)", "README",
     lambda q: q.T * q.rpm, u.W),
    # slip losses (README 'Slip heating'; Jackson 8.1, Stoll, Reitz, Bertotti)
    ("slip-skin-depth", "delta^2 = 2/(omega·mu0·mu_r·sigma)", "Jackson 8.1",
     lambda q: 2 / (q.omega * q.mu0 * 300 * q.sigma), u.M ** 2),
    ("slip-halfspace", "P/A = sigma·omega^2·B^2·delta/(4·k^2) (speed^1.5 law)", "Stoll",
     lambda q: q.sigma * q.omega ** 2 * q.B ** 2 * q.length / (4 * q.k ** 2), u.W / u.M ** 2),
    ("slip-thin-shell", "P/A = sigma·t·v^2·B^2/2 (speed^2 law)", "Reitz / Stoll",
     lambda q: q.sigma * q.length * q.v ** 2 * q.B ** 2 / 2, u.W / u.M ** 2),
    ("slip-thin-strip", "P/V = pi^2·sigma·d^2·f^2·B^2/6", "Bertotti",
     lambda q: math.pi ** 2 * q.sigma * q.length ** 2 * q.f ** 2 * q.B ** 2 / 6, u.W / u.M ** 3),
    # temperature design (README 'Demagnetization', 'Adhesive'; Volkersen)
    ("demag-knee", "Hk = 0.9·Hcj20·(1 - beta·(T - 20))", "README",
     lambda q: 0.9 * q.H * (u.ONE - 0.005 * q.alpha * q.dT), u.A_PER_M),
    ("demag-load-line", "H = Br/mu0·1/(1 + Pc)", "permeance-coefficient load line",
     lambda q: q.B / q.mu0 / (1 + 1.0), u.A_PER_M),
    ("demag-calibration-offset", "offset = T_ref - T_rating, T_ref = 20 + (Hk - H_ref)/(Hk·|beta| - H_ref·|alpha|)",
     "README 'Demagnetization' (Temperature design!C49, C50)",
     lambda q: 20 * q.dT + (0.9 * q.H - q.H) / (0.9 * q.H * (0.005 * q.alpha) - q.H * (0.0012 * q.alpha)) - 150 * q.dT,
     u.K),
    ("adhesive-volkersen-lambda", "lambda^2 = G/eta·(1/(E1·t1) + 1/(E2·t2))", "Volkersen",
     lambda q: q.stress / q.length * (1 / (q.stress * q.length) + 1 / (q.stress * q.length)), 1 / u.M ** 2),
    ("adhesive-volkersen-shear", "tau = G·d_alpha·dT·tanh(lambda·L/2)/(eta·lambda)", "Volkersen",
     lambda q: q.stress * q.alpha * q.dT * _f(math.tanh, q.k * q.length / 2) / (q.length * q.k), u.PA),
    ("adhesive-bond-shear", "tau = T/(n·r·A)", "README",
     lambda q: q.T / (10 * q.length * q.area), u.PA),
    ("adhesive-centrifugal", "F = m·omega^2·r", "Newton",
     lambda q: q.mass * q.omega ** 2 * q.length, u.N),
    # masses and inertia (README 'Masses and envelope')
    ("mass-density", "m = rho·V", "README",
     lambda q: q.rho * q.length ** 3, u.KG),
    ("mass-inertia", "J = sum(m·r^2)", "README",
     lambda q: q.mass * q.length ** 2, u.KG_M2),
]


@pytest.mark.family("constants")
@pytest.mark.parametrize("key, formula, source, build, unit", FORMULAS, ids=[f[0] for f in FORMULAS])
def test_formula_is_dimensionally_consistent(key, formula, source, build, unit):
    """Formula-level check: the formula as stated has the dimensions of its result (units_ref raises if a sum or a
    transcendental argument mixes dimensions). Independent of the engine code; the engine's numeric unit factors
    for the same formula are verified by the SI re-derivation of the owning check (torque chain:
    test_torque_chain_unit_factors)."""
    q = build(Q)
    assert_all([flag_item(f"{key}: {formula} gives {q.describe()}, expected {unit.describe()}", source,
                          q.dims == unit.dims)])


# --------------------------------------------------------------------------- engine-level unit factors
@pytest.mark.family("constants")
def test_torque_chain_unit_factors():
    """The engine's numeric unit factors in the torque chain, observed from its outputs, equal the SI conversions:
    k = n·(p/2)/R needs mm -> m (x1000), area_lever = 2·pi·R^2·L needs mm^3 -> m^3 (1e-9), and
    torque = tau·area_lever needs Pa·m^3 = N·m (1)."""
    m, ci = run().model, defaults().coupling
    r, L = m.gap_radius_mm, m.active_length_mm
    assert_all([
        ("k1 unit factor (1/mm -> 1/m)", "Calculator!C71", m.k1 * r / (ci.npole / 2), (1 / u.MM).to(1 / u.M),
         TOL_ALGEBRA),
        ("area x lever unit factor (mm^3 -> m^3)", "Calculator!C90", m.area_lever_m3 / (2 * math.pi * r ** 2 * L),
         (u.MM ** 3).to(u.M ** 3), TOL_ALGEBRA),
        ("2D torque unit factor (Pa·m^3 -> N·m)", "Calculator!C91", m.torque_2d_Nm / (m.tau_Pa * m.area_lever_m3),
         (u.PA * u.M ** 3).to(u.NM), TOL_ALGEBRA),
    ])


SIMILARITY_LENGTHS = (
    "coupling.inner_back_apothem_mm", "coupling.bore_mm", "coupling.keyway_depth_mm",
    "coupling.magnets.manual_inner_length_mm", "coupling.magnets.manual_inner_width_mm",
    "coupling.magnets.manual_inner_thickness_mm", "coupling.magnets.manual_outer_length_mm",
    "coupling.magnets.manual_outer_width_mm", "coupling.magnets.manual_outer_thickness_mm",
    "metal.face_gap_mm", "metal.bond_inner_mm", "metal.bond_outer_mm", "metal.cup_wall_corner_mm",
)


def _scaled(lam: float):
    """Defaults with manual magnets (same values as the library B842SH) and every length multiplied by lam."""
    base = vary(defaults(), {"coupling.magnets.part_inner": "", "coupling.magnets.part_outer": ""})
    changes = {}
    for path in SIMILARITY_LENGTHS:
        obj = base
        for p in path.split("."):
            obj = getattr(obj, p)
        changes[path] = obj * lam
    return run(vary(base, changes)).model


@pytest.mark.family("constants")
def test_geometric_similarity_scaling():
    """Scaling every length by lam = 1.7 (Br fixed) leaves every k·t, k·g and pitch/L unchanged, so shear stress is
    invariant and torque grows as lam^3; gap radius and cup OD grow as lam; wave number shrinks as 1/lam. Catches
    mixed-unit sums, hard-coded lengths and wrong exponents anywhere in the Calculator chain."""
    lam = 1.7
    a, b = _scaled(1.0), _scaled(lam)
    assert_all([
        ("shear stress invariant", "Calculator!C89", b.tau_Pa, a.tau_Pa, TOL_ALGEBRA),
        ("2D torque ~ lam^3", "Calculator!C91", b.torque_2d_Nm, lam ** 3 * a.torque_2d_Nm, TOL_ALGEBRA),
        ("end factor invariant", "Calculator!C92", b.f_end, a.f_end, TOL_ALGEBRA),
        ("pull-out ~ lam^3", "Calculator!C93", b.pullout_Nm, lam ** 3 * a.pullout_Nm, TOL_ALGEBRA),
        ("gap radius ~ lam", "Calculator!C64", b.gap_radius_mm, lam * a.gap_radius_mm, TOL_ALGEBRA),
        ("cup OD ~ lam", "Calculator!C62", b.cup_od_mm, lam * a.cup_od_mm, TOL_ALGEBRA),
        ("wave number ~ 1/lam", "Calculator!C71", b.k1, a.k1 / lam, TOL_ALGEBRA),
    ])


@pytest.mark.family("thermal")
def test_thermal_scaling_with_conductance():
    """Dimensions fix how the lumped network responds to its conductance G (Incropera ch. 5): doubling G halves
    tau = C/G and both steady rises P/G, doubles the critical drag (limit - start)·G/omega, and leaves the heat
    capacity and the per-event rise P·t/C unchanged."""
    a = run().temperature.thermal
    g2 = 2 * defaults().temperature.thermal.conductance_W_K
    b = run(vary(defaults(), {"temperature.thermal.conductance_W_K": g2})).temperature.thermal
    assert_all([
        ("heat capacity unchanged", "Temperature design!C141", b.heat_capacity_J_K, a.heat_capacity_J_K, TOL_ALGEBRA),
        ("time constant halves", "Temperature design!C143", b.time_constant_s, a.time_constant_s / 2, TOL_ALGEBRA),
        ("steady rise (estimate) halves", "Temperature design!C146", b.steady_rise_est_C, a.steady_rise_est_C / 2,
         TOL_ALGEBRA),
        ("steady rise (high) halves", "Temperature design!C147", b.steady_rise_high_C, a.steady_rise_high_C / 2,
         TOL_ALGEBRA),
        ("critical drag doubles", "Temperature design!C153", b.critical_drag_Nm, 2 * a.critical_drag_Nm, TOL_ALGEBRA),
        ("rise per event unchanged", "Temperature design!C145", b.rise_per_event_C, a.rise_per_event_C, TOL_ALGEBRA),
    ])


# --------------------------------------------------------------------------- verdict thresholds
@pytest.mark.family("metal")
def test_hot_min_check_threshold():
    """Metal design C11: 'Below hot minimum' exactly when the hot-low torque (C9: pull-out at the operating
    temperature × (1 - variation)) < the required minimum: at defaults (2.25 < 2.5 N·m, as the README states),
    'Estimate covers hot min' at equality, 'Below hot minimum' one ulp above."""
    base = run()
    hot_low = base.metal.torque_hot_low_Nm
    at = run(vary(defaults(), {"metal.required_min_Nm": hot_low}))
    above = run(vary(defaults(), {"metal.required_min_Nm": math.nextafter(hot_low, math.inf)}))
    assert_all([
        text_item(f"defaults (hot low {hot_low:.4f} N·m)", "Metal design!C11", base.metal.hot_min_check,
                  "Below hot minimum" if hot_low < defaults().metal.required_min_Nm else "Estimate covers hot min"),
        text_item("required = hot low", "Metal design!C11", at.metal.hot_min_check, "Estimate covers hot min"),
        text_item("required one ulp above hot low", "Metal design!C11", above.metal.hot_min_check,
                  "Below hot minimum"),
    ])


@pytest.mark.family("metal")
def test_clearance_check_threshold():
    """Metal design C37: 'Below target' exactly when the minimum running clearance (C35) < the residual target: at
    defaults (-0.10 < 0.2 mm, as the README states), 'Meets assumed target' at equality, 'Below target' one ulp
    above."""
    base = run()
    c = base.metal.min_running_clearance_mm
    at = run(vary(defaults(), {"metal.residual_target_mm": c}))
    above = run(vary(defaults(), {"metal.residual_target_mm": math.nextafter(c, math.inf)}))
    assert_all([
        text_item(f"defaults (clearance {c:.4f} mm)", "Metal design!C37", base.metal.clearance_check,
                  "Below target" if c < defaults().metal.residual_target_mm else "Meets assumed target"),
        text_item("target = clearance", "Metal design!C37", at.metal.clearance_check, "Meets assumed target"),
        text_item("target one ulp above clearance", "Metal design!C37", above.metal.clearance_check, "Below target"),
    ])


@pytest.mark.family("torque")
def test_calculator_verdict_threshold():
    """Calculator verdict compares the NOMINAL pull-out (no variation) with the floor max(drive·SF, required min):
    'Nominal only: hot test' at defaults (2.647 >= 2.5) and at equality, 'Below hot minimum' one ulp above."""
    base = run()
    ci, md = defaults().coupling, defaults().metal
    floor = max(ci.drive_torque_Nm * ci.drive_safety_factor, md.required_min_Nm)
    at = run(vary(defaults(), {"metal.required_min_Nm": base.model.pullout_Nm}))
    above = run(vary(defaults(), {"metal.required_min_Nm": math.nextafter(base.model.pullout_Nm, math.inf)}))
    assert_all([
        ("required floor", "Calculator!C101", base.model.required_floor_Nm, floor, TOL_ALGEBRA),
        text_item("defaults", "Calculator!C102", base.model.verdict,
                  "Below hot minimum" if base.model.pullout_Nm < floor else "Nominal only: hot test"),
        text_item("floor = pull-out", "Calculator!C102", at.model.verdict, "Nominal only: hot test"),
        text_item("floor one ulp above pull-out", "Calculator!C102", above.model.verdict, "Below hot minimum"),
    ])
```

- [ ] **Step 5: Run the checks**

Run: `cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py && ./.venv/Scripts/python -m pytest -q audit/tests/test_clamps.py audit/tests/test_sweeps.py audit/tests/test_units_verdicts.py`
Expected: `4 failed, 71 passed` in about 0.5 s (`test_clamps.py`: 3 failed, 15 passed; `test_sweeps.py`: 1 failed, 5 passed; `test_units_verdicts.py`: 51 passed), covering 3,925 comparisons (843 clamps, 3,009 sweeps, 73 constants and units). The 4 FAILs are 4 separate root causes. They are candidate findings for Task 8, not bugs to fix now. Leave the engine untouched and do not adjust tolerances.

1. **`test_screw_length_meets_engagement_and_stays_inside`**: 7 of 39 comparisons fail, all in the Clamp screw sizes row-34 length formula. The engine's length is CEILING(grip + engagement, 2). It leaves out the 0.8 mm slit, so the chosen screw engages less than the required 2·d (1·d in `strip_governs`):
   - `defaults` M4 (E34) is 12 mm and engages 7.335 of 8.000 mm. The valid 2 mm-step lengths are: none. A 14 mm screw would engage fully but pass the 13.656 mm seat-to-far-OD chord, so it would protrude.
   - `al6061_cl10.9` M4 and `two_piece_oily_A4-70` M4 give the same numbers.
   - `boss22_len14.5` M2.5 (C34) is 10 mm and engages 4.413 of 5.000 mm; 12 mm would be valid.
   - `strip_governs` M2.5 is 10 mm, 1.716 of 2.500 mm (valid: 12, 14, 16).
   - `strip_governs` M3 is 10 mm, 2.741 of 3.000 mm (valid: 12, 14, 16).
   - `strip_governs` M4 is 8 mm, 3.335 of 4.000 mm (valid: 10, 12).
2. **`test_stripping_capacity_not_above_min_material`**: 2 of 5 comparisons fail (row 22). The engine's area is 0.6000·pi·d·Le. The FED-STD-H28/2B area at ISO 965-2 minimum-material limits is 0.5700 (M2.5), 0.5885 (M3), 0.6182 (M4), 0.6365 (M5) and 0.6467 (M6)·pi·d·Le.
   - M2.5 (C22): 5199.3 N against 4939.8 N, ratio 1.0525.
   - M3 (D22): 7487.0 N against 7343.7 N, ratio 1.0195.
   - M4 to M6 are conservative, with ratios 0.9705, 0.9426 and 0.9277.
   - At defaults the strength limit governs every size (e.g. M2.5 2466 N against 4940 N), so no default number or verdict changes.
3. **`test_readme_22mm_only_m3_fits`**: 2 of 4 comparisons fail. The README and the Shaft clamps!C35 help say "At 22 mm only M3 fits".
   - At a 22 mm boss the reference passes the geometry for ['M2.5', 'M3'], and the engine agrees.
   - At a 22 mm boss with an 18 mm clamp, the reference and the engine both pick M2.5 x 3, not M3.
   - The companion claim "two M3 need a 14.5 mm clamp" passes (`test_readme_22mm_two_m3_need_14_5mm`).
4. **`test_pole_sweep_apothem_keeps_calculator_hub_wall`**: 2 of 6 comparisons fail, at N = 6 (Pole sweep!C6) and N = 8 (C7). The engine's apothem is 9.200 mm, which leaves 2.450 mm of hub wall past the keyway by the Calculator!C53 definition, against 2.5 mm. The shortfall equals the 0.05 mm inner bondline, so this is a definition mismatch between the sheets.
   - The rule as the sheet note states it passes (`test_pole_sweep_apothem_rule_as_stated`).
   - At 9.25 mm, the 6-pole row's pull-out changes by -0.49 % (0.95837 to 0.95369 N·m). Its cup OD becomes 43.013 mm, over the 43 mm envelope, so the status changes from 'below hot minimum' to 'outside OD envelope'.
   - The 8-pole row changes by +0.21 %, with no status change.

Everything else passes:
- **Clamps:**
  - ISO catalogue and stress areas; geometry (225 comparisons); length_ok flag; stripping scaling with shear and engagement.
  - Preload (e.g. M4 6387.45 N) and the capacity chain (M4: 7.66494 N·m per screw, 1 screw, SF 2.036, tightening 5.110 N·m, head 282.9 MPa 'OK').
  - Fit/works, recommendation ('ISO 4762 M4 x 12, class 12.9' at defaults; M3 with 2 screws, 'ISO 4762 M3 x 10, class 12.9', at boss 22 / 14.5 mm; 'None: enlarge the boss or the clamp length' in the other three scenarios `al6061_cl10.9`, `two_piece_oily_A4-70` and `strip_governs`, where reference and engine agree), summary/layout, head-check boundary, design torque (3.76472 to 7.52943 N·m), key (100.39 MPa, SF 0.9961), adapter joint (21.407 N·m, SF 2.843).
- **Sweeps:** all 78 gap rows and 36 pole rows in 6 scenarios (2,964 comparisons), all five status branches, the stated apothem rule, and the sweep threshold boundary.
- **Constants and units:**
  - MU0 (rel_err 4.9e-8 against the 3.98e-7 rounding bound) and NdFeB density.
  - C82 block mass (1.917335 g); 42 formula-level dimension checks (the torque chain, the metal-design band, gearbox input and clearance stack, the materials back-iron and plating formulas, the clamp, thermal, slip, demagnetization, adhesive and mass formulas).
  - Torque-chain unit factors (x1000, 1e-9, 1); lambda^3 similarity (pull-out ratio 4.913000000000004).
  - Conductance scaling, and the C11, C37 and C102 boundaries.

`audit/out/results.json` lists every check with its message.

- [ ] **Step 6: Commit**

```bash
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 add reference/magcoupling-py/audit/references/clamp_ref.py reference/magcoupling-py/audit/references/sweep_ref.py reference/magcoupling-py/audit/references/units_ref.py reference/magcoupling-py/audit/tests/test_clamps_sweeps_units_reference_sanity.py reference/magcoupling-py/audit/tests/test_clamps.py reference/magcoupling-py/audit/tests/test_sweeps.py reference/magcoupling-py/audit/tests/test_units_verdicts.py
git -C /c/Users/Cole/source/repos/linkage_simulation-m1 commit -m "test(magcoupling-audit): clamps, sweeps and constants independent checks" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
```
Do not add `audit/out/` (generated).

### Task 8: Adversarially verify candidate findings and write the report (controller-led)

This task is run by the **controller session**, not a single implementer subagent, because Step 3 launches a Workflow (subagents cannot). It ends at a **user review gate**. All commands use absolute paths into the worktree `/c/Users/Cole/source/repos/linkage_simulation-m1`.

**Files:**
- Create: `reference/magcoupling-py/audit/tools/group_candidates.py`
- Create: `reference/magcoupling-py/audit/tools/coverage_table.py`
- Create: `docs/analyses/<YYYY-MM-DD>-magcoupling-math-audit.md` (dated the day this task completes)
- Create: `docs/analyses/<YYYY-MM-DD>-magcoupling-audit-results.json` (raw check results plus skeptic verdicts)
- Modify: `docs/ai/05-update-tracker.md`, `docs/ai/04-memory.yaml`
- Modify (only when a skeptic majority says the *check* is wrong): the owning `reference/magcoupling-py/audit/tests/*.py` or `audit/references/*.py`

**Interfaces:**
- Consumes: `audit/out/results.json` (written by the Task 1 harness after every run); `audit/tools/__init__.py` and `audit/tools/placeholder_sensitivity.py` (Task 6); root-cause tags `Root-cause group: <tag>` in check docstrings (Task 3 uses T3-RC1..RC6); the engine README.
- Produces: `audit/out/candidate_groups.json`; the findings report the user reviews. Its **Decision** column is filled in after the user's review, and approved rows become the deviation registry of M2.

- [ ] **Step 1: Run the whole audit**

```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py
./.venv/Scripts/python -m pytest -q audit/tests -p no:cacheprovider || true
```

Expected: every check from Tasks 1–7 runs. The per-task "Expected" lists in Tasks 2–7 name each failing check; those failures are the candidate findings.

- [ ] **Step 2: Group candidates by root cause**

File: `reference/magcoupling-py/audit/tools/group_candidates.py` (create)
```python
"""Group failing audit checks by root cause so each root cause is verified once (Task 8).

A check's group is its declared tag (a docstring line 'Root-cause group: <tag>') or, failing that,
its test function name without pytest parameters, so all parametrized cases of one check form one
group. Writes audit/out/candidate_groups.json and prints one line per group.

Run from reference/magcoupling-py:  ./.venv/Scripts/python -m audit.tools.group_candidates
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out"
TAG = re.compile(r"Root-cause group:\s*([A-Za-z0-9_-]+)")


def group_key(result: dict) -> str:
    tag = TAG.search(result.get("doc") or "")
    if tag:
        return tag.group(1)
    name = result["id"].split("::")[-1]
    return name.split("[", 1)[0]


def main() -> None:
    results = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for r in results:
        if r["outcome"] == "passed":
            continue
        groups.setdefault(group_key(r), []).append(
            {"id": r["id"], "family": r["family"], "message": r["message"], "doc": r["doc"]})
    payload = [{"group": k, "members": v} for k, v in groups.items()]
    (OUT / "candidate_groups.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    total = sum(len(v) for v in groups.values())
    print(f"{total} candidates in {len(groups)} root-cause groups")
    for k, v in groups.items():
        print(f"  {k}: {len(v)} ({v[0]['family']})")


if __name__ == "__main__":
    main()
```

Run:
```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py
./.venv/Scripts/python -m audit.tools.group_candidates
```

Expected: one line per group, with fewer groups than candidates (all parametrized cases of a check, and every check tagged with the same `Root-cause group`, collapse into one group).

- [ ] **Step 3: Controller runs the skeptic workflow over the groups**

Pass the contents of `audit/out/candidate_groups.json` as `args.groups`. Script:

```javascript
export const meta = {
  name: 'magcoupling-m1-verify',
  description: 'Three skeptic lenses per root-cause group of candidate findings; majority verdict and class',
  phases: [{ title: 'Verify' }],
}
const ARGS = typeof args === 'string' ? JSON.parse(args || '{}') : (args || {})
const RP = 'C:/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py'
const VERDICT = {
  type: 'object', required: ['verdict', 'reason'],
  properties: {
    verdict: { enum: ['engine_error', 'model_approximation', 'placeholder_input', 'check_bug', 'not_reproduced'] },
    reason: { type: 'string' },
    proposed_correction: { type: 'string' },
    impact_at_defaults: { type: 'string' },
  },
}
const LENSES = [
  'Reproduce it from scratch: independently re-derive or re-compute both the engine value and the correct value; state what you ran.',
  'Attack the CHECK: look for a bug in the audit reference or test (units, wrong engine field, wrong cell, sign convention, geometry mismatch, unconverged numerics, tolerance). Default to check_bug if the check is unsound.',
  'Look for a benign explanation: is this a documented model approximation or a placeholder input in the engine README (sections "Approach", "Inputs that are placeholders", "Differences from the workbook")? Default to model_approximation or placeholder_input when the README already declares it.',
]
const groups = ARGS.groups || []
if (!groups.length) return { error: 'pass args.groups (audit/out/candidate_groups.json)' }
const results = await parallel(groups.map(g => () =>
  parallel(LENSES.map((lens, i) => () => agent( // session model: magcoupling math audit skeptics (CLAUDE.md section 5)
    `Skeptic ${i + 1} of 3 for one root-cause group of candidate findings from an independent audit of a magnetic-coupling calculator. Engine (read-only oracle): ${RP}/magcoupling ; audit code: ${RP}/audit ; engine README: ${RP}/README.md ; Python: ${RP}/.venv/Scripts/python . Do NOT modify any file.\nGroup "${g.group}" (${g.members.length} failing checks that share one suspected cause): ${JSON.stringify(g.members)}\nLens: ${lens}\nJudge the group as a whole; if members clearly have different causes, say so in reason. For engine_error, give the corrected formula and its effect at default inputs (numbers, and whether any verdict text such as 'Below hot minimum' changes).`,
    { label: `verify:${g.group.slice(0, 40)}:${i + 1}`, phase: 'Verify', schema: VERDICT }))))
    .then(vs => {
      const got = vs.filter(Boolean)
      const tally = {}
      for (const v of got) tally[v.verdict] = (tally[v.verdict] || 0) + 1
      const top = Object.entries(tally).sort((a, b) => b[1] - a[1])
      const majority = top.length && top[0][1] >= 2 ? top[0][0] : 'split'
      return { group: g.group, members: g.members.map(m => m.id), majority, verdicts: got }
    })
))
return { results }
```

Expected: one entry per group with `majority` in `engine_error | model_approximation | placeholder_input | check_bug | not_reproduced | split`.

- [ ] **Step 4: Resolve `check_bug` and `split` results**

For each `check_bug` group: fix the audit code test-first (first make the reference's sanity test expose the bug, then fix it), re-run that file, and re-verify only those groups with Step 3's script (at most two rounds). For each `split`: the controller reads all three verdicts and the evidence and records a ruling with its reason in the report. Commit audit-code fixes separately:

```bash
WT=/c/Users/Cole/source/repos/linkage_simulation-m1
git -C "$WT" add reference/magcoupling-py/audit
git -C "$WT" commit -m "test(magcoupling-audit): fix check bugs found in verification

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
```

- [ ] **Step 5: Build the coverage and placeholder tables**

File: `reference/magcoupling-py/audit/tools/coverage_table.py` (create)
```python
"""Print the 'confirmed correct' coverage table for the findings report (Task 8).

Only passing checks of the ENGINE count as confirmations. Excluded and counted separately:
reference self-tests (test functions named test_sanity*), the harness smoke tests, and checks whose
docstring says 'not an independent check'.

Run from reference/magcoupling-py:  ./.venv/Scripts/python -m audit.tools.coverage_table
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "out" / "results.json"


def classify(result: dict) -> str:
    module, _, name = result["id"].rpartition("::")
    if name.split("[", 1)[0].startswith("test_sanity"):
        return "reference-self-test"
    if module.endswith("test_harness_smoke.py"):
        return "harness"
    if "not an independent check" in (result.get("doc") or "").lower():
        return "consistency-only"
    return "engine"


def main() -> None:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    by_family: dict[str, list[tuple[str, str]]] = defaultdict(list)
    excluded: dict[str, int] = defaultdict(int)
    for r in results:
        if r["outcome"] != "passed":
            continue
        kind = classify(r)
        if kind != "engine":
            excluded[kind] += 1
            continue
        first_line = ((r.get("doc") or "").splitlines() or [""])[0]
        by_family[r["family"]].append((r["id"].split("::")[-1], first_line))
    print("| Family | Check | What it confirms |")
    print("|---|---|---|")
    for fam in sorted(by_family):
        for name, doc in sorted(by_family[fam]):
            print(f"| {fam} | `{name}` | {doc.replace('|', '/')} |")
    total = sum(len(v) for v in by_family.values())
    print(f"\n{total} engine checks passed across {len(by_family)} families.")
    print("Excluded from engine coverage: " + ", ".join(f"{n} {k}" for k, n in sorted(excluded.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
```

Run:
```bash
cd /c/Users/Cole/source/repos/linkage_simulation-m1/reference/magcoupling-py
./.venv/Scripts/python -m audit.tools.coverage_table > audit/out/coverage.md
./.venv/Scripts/python -m audit.tools.placeholder_sensitivity > audit/out/placeholders.md
```

Expected: `coverage.md` is a Markdown table with one row per passing engine check, and stderr reports how many reference self-tests, harness checks and consistency-only checks were excluded. `placeholders.md` holds the placeholder sensitivity table from Task 6's tool.

- [ ] **Step 6: Write the findings report**

`docs/analyses/<YYYY-MM-DD>-magcoupling-math-audit.md` (relative to the worktree root), with exactly these sections, filled from Steps 3–5:

```markdown
# Magnetic coupling calculator: math audit (M1)

**Date:** <YYYY-MM-DD> · **Engine:** magcoupling 1.0.0 (workbook port), vendored at `reference/magcoupling-py/`
**Spec:** `docs/superpowers/specs/2026-09-28-magcoupling-calculator-design.md` (M1)
**Method:** independent references per formula family (re-derivation, 2D field model, magpylib 3D, limits and scaling, literature); every root-cause group checked by three skeptic lenses. Model-vs-reference tolerance: 2 % (`TOL_MODEL`).

## Summary

| Class | Root-cause groups | Checks |
|---|---|---|
| Engine errors (corrections proposed) | N | N |
| Model approximations (documented) | N | N |
| Placeholder inputs (need measurement) | N | N |
| Engine checks confirmed correct | — | N |

Headline effect at the default design: <one sentence per verdict that changes, for example the hot-torque check>.

## Engine errors (for user decision)

| # | Group | Cells | Formula (workbook) | Issue and evidence | Effect at defaults | Proposed correction | Decision |
|---|---|---|---|---|---|---|---|
| E1 | ... | ... | ... | ... | ... | ... | _pending_ |

## Model approximations

| # | Group | Cells | Approximation | Evidence of size | Recommendation |
|---|---|---|---|---|---|

## Placeholder inputs

<paste audit/out/placeholders.md, then add any README "Inputs that are placeholders" row it does not cover>

## Split verdicts and controller rulings

| Group | Verdicts | Ruling and reason |
|---|---|---|

## Confirmed correct (coverage)

<paste audit/out/coverage.md and the excluded-count line>

## Not independently checkable

- Steel-circuit end-effect factor and 3D effects of the steel parts: magpylib models no permeable material, so the 3D cross-check covers the free-space circuit only (Task 3).
- Temperature-design inputs derived from the 3D field model (Temperature design C116–C122) and the demagnetization field inputs C52–C55: produced by `fields3d`, whose port and validation are M3 scope.
- Slip-loss modelling constants: cap end factor 0.7, the web r² weighting, `high_multiplier`, `end_factor` (Task 6; their sensitivities are in the placeholder table).
- The single-lump thermal network: a modelling choice, re-derived exactly but not independently validated (Task 6).
- Every further item a task marks "not independently checkable" or "listed for the report": search with `grep -rn "not independently checkable\|listed for the report" audit/tests audit/references` and add each with its reason.
```

Also write `docs/analyses/<YYYY-MM-DD>-magcoupling-audit-results.json`: the Step 1 results merged with the Step 3 verdicts (one object per check: `id, family, outcome, message, group, majority, verdicts`).

- [ ] **Step 7: Update the coordination docs**

`docs/ai/05-update-tracker.md`: a new top entry "YYYY-MM-DD — Magcoupling M1: math audit" with the counts and the report path. `docs/ai/04-memory.yaml`: add an `open_questions` item "Which magcoupling M1 corrections (E1..En in the audit report) to apply in the M2 port?".

- [ ] **Step 8: Commit**

```bash
WT=/c/Users/Cole/source/repos/linkage_simulation-m1
git -C "$WT" add docs/analyses docs/ai/05-update-tracker.md docs/ai/04-memory.yaml reference/magcoupling-py/audit/tools/group_candidates.py reference/magcoupling-py/audit/tools/coverage_table.py
git -C "$WT" commit -m "docs(analyses): magcoupling M1 math audit report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017VvLj1MRiP6cyevwwaXay9"
```

- [ ] **Step 9: User review gate — STOP**

Present the Summary table and each engine error in plain language, and ask the user to mark every row of "Engine errors" approve or reject. Record the decisions in the Decision column, commit, and only then begin planning M2. Nothing from M1 is merged to `main` or pushed without the user's choice.
