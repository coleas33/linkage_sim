# Multi-Coupler Trace & Arbitrary Trace Points — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support tracing multiple coupler points across different bodies with visual connection lines, plus a `Mechanism.add_trace_point()` API for arbitrary runtime trace points on any link.

**Architecture:** New `CouplerTrace` dataclass replaces the single-trace fields in `SweepData`. `_find_coupler_body_and_point` becomes `_find_all_coupler_points` returning a list. A new `Mechanism.add_trace_point(name, body_id, x, y)` method lets users attach arbitrary trace points to any link at build time (stored as coupler points on the body). Drawing helpers render a connection line from each coupler/trace point to its body's centroid. All viewer scripts and `export_gifs.py` are updated.

**Tech Stack:** Python 3.12, numpy, matplotlib, pytest, mypy strict.

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/linkage_sim/core/mechanism.py` | Add `add_trace_point()` method |
| Modify | `src/linkage_sim/core/bodies.py` | (no change needed — `add_coupler_point` already works) |
| Modify | `src/linkage_sim/viz/interactive_viewer.py` | `CouplerTrace` dataclass, multi-trace sweep, drawing with connection lines |
| Modify | `scripts/export_gifs.py` | Multi-trace drawing with connection lines |
| Modify | `scripts/view_sixbar.py` | Add second trace point on a different body |
| Modify | `scripts/view_sixbar_A1.py` | Verify/fix coupler point position |
| Modify | `scripts/view_sixbar_A2.py` | Verify/fix coupler point position |
| Modify | `scripts/view_sixbar_B2.py` | Verify/fix coupler point position |
| Modify | `scripts/view_sixbar_B3.py` | Verify/fix coupler point position |
| Create | `tests/test_coupler_trace.py` | Tests for multi-trace and add_trace_point |
| Modify | `RUST_MIGRATION.md` | Document coupler/trace point API for Rust port |
| Modify | `ROADMAP_IMPLEMENTATION.md` | Document the feature |

---

### Task 1: `CouplerTrace` dataclass and multi-trace sweep

**Files:**
- Modify: `src/linkage_sim/viz/interactive_viewer.py`
- Create: `tests/test_coupler_trace.py`

- [ ] **Step 1: Write test for multi-trace data collection**

```python
# tests/test_coupler_trace.py
"""Tests for multi-coupler trace support."""
from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.viz.interactive_viewer import CouplerTrace, precompute_sweep


def _build_fourbar_two_coupler_points() -> tuple[Mechanism, list]:
    """Build a 4-bar with coupler points on TWO different bodies."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
    crank.add_coupler_point("CP_crank", 1.0, 0.3)
    coupler = make_bar("coupler", "B", "C", length=4.0, mass=1.0)
    coupler.add_coupler_point("CP_mid", 2.0, 0.0)
    rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.8)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()
    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    return mech, [gravity]


class TestMultiCouplerTrace:
    """precompute_sweep collects ALL coupler points across all bodies."""

    def test_finds_two_traces(self) -> None:
        mech, forces = _build_fourbar_two_coupler_points()
        sweep = precompute_sweep(mech, forces, n_steps=36)
        assert len(sweep.coupler_traces) == 2

    def test_trace_has_correct_body_ids(self) -> None:
        mech, forces = _build_fourbar_two_coupler_points()
        sweep = precompute_sweep(mech, forces, n_steps=36)
        body_ids = {t.body_id for t in sweep.coupler_traces}
        assert body_ids == {"crank", "coupler"}

    def test_trace_arrays_correct_length(self) -> None:
        mech, forces = _build_fourbar_two_coupler_points()
        sweep = precompute_sweep(mech, forces, n_steps=36)
        for trace in sweep.coupler_traces:
            assert len(trace.x) == 36
            assert len(trace.y) == 36

    def test_single_coupler_point_still_works(self) -> None:
        """Backward compat: mechanism with one coupler point."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
        coupler = make_bar("coupler", "B", "C", length=4.0, mass=1.0)
        coupler.add_coupler_point("P", 2.0, 0.0)
        rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.8)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        sweep = precompute_sweep(mech, [], n_steps=36)
        assert len(sweep.coupler_traces) == 1
        assert sweep.coupler_traces[0].body_id == "coupler"

    def test_no_coupler_points_empty_list(self) -> None:
        """Mechanism with no coupler points gives empty traces list."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
        coupler = make_bar("coupler", "B", "C", length=4.0, mass=1.0)
        rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.8)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        sweep = precompute_sweep(mech, [], n_steps=36)
        assert len(sweep.coupler_traces) == 0
```

- [ ] **Step 2: Run tests — expect FAIL (CouplerTrace doesn't exist)**

Run: `pytest tests/test_coupler_trace.py -v`

- [ ] **Step 3: Implement CouplerTrace and multi-trace in precompute_sweep**

In `interactive_viewer.py`:

1. Add `CouplerTrace` dataclass above `SweepData`:

```python
@dataclass
class CouplerTrace:
    """Trace data for a single coupler point across a sweep.

    Attributes:
        body_id: Which body this point is attached to.
        point_name: Name of the coupler point on that body.
        x: X coordinates per sweep step (NaN where solve failed).
        y: Y coordinates per sweep step (NaN where solve failed).
    """
    body_id: str
    point_name: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
```

2. Replace `SweepData` fields:
```python
# OLD:
coupler_trace_x: NDArray[np.float64] | None = None
coupler_trace_y: NDArray[np.float64] | None = None

# NEW:
coupler_traces: list[CouplerTrace] = field(default_factory=list)
```

3. Replace `_find_coupler_body_and_point` with `_find_all_coupler_points`:
```python
def _find_all_coupler_points(
    mechanism: Mechanism,
) -> list[tuple[str, str, NDArray[np.float64]]]:
    """Find all coupler points across all bodies.

    Returns list of (body_id, point_name, point_local) tuples.
    """
    result = []
    for body_id, body in mechanism.bodies.items():
        if body_id == GROUND_ID:
            continue
        for cp_name, cp_local in body.coupler_points.items():
            result.append((body_id, cp_name, cp_local))
    return result
```

4. Replace the coupler trace section in `precompute_sweep` (the block starting at
   `# --- Coupler trace ---`) with:
```python
    # --- Coupler traces (all coupler points across all bodies) ---
    all_cp = _find_all_coupler_points(mechanism)
    coupler_traces: list[CouplerTrace] = []
    for cp_body_id, cp_name, cp_local in all_cp:
        xs = np.full(n_steps, np.nan)
        ys = np.full(n_steps, np.nan)
        for i, q in enumerate(solutions):
            if q is not None:
                pt_g = mechanism.state.body_point_global(cp_body_id, cp_local, q)
                xs[i] = float(pt_g[0])
                ys[i] = float(pt_g[1])
        coupler_traces.append(CouplerTrace(
            body_id=cp_body_id,
            point_name=cp_name,
            x=xs,
            y=ys,
        ))
```

5. Update `SweepData` construction at the bottom of `precompute_sweep`:
```python
    return SweepData(
        ...
        coupler_traces=coupler_traces,
    )
```

6. Remove the `coupler_body_id` and `coupler_point_name` parameters from
   `precompute_sweep` (they are no longer needed — all coupler points are
   auto-discovered). Keep them on `launch_interactive` for backward compat
   but ignore them internally (the sweep now collects everything).

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_coupler_trace.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/linkage_sim/viz/interactive_viewer.py tests/test_coupler_trace.py
git commit -m "feat: multi-coupler trace support with CouplerTrace dataclass"
```

---

### Task 2: Update drawing code for multi-trace + connection lines

**Files:**
- Modify: `src/linkage_sim/viz/interactive_viewer.py` (drawing in `launch_interactive`)
- Modify: `scripts/export_gifs.py` (drawing in `export_gif`)

The color palette for multiple traces:
```python
TRACE_COLORS = ["green", "orange", "purple", "cyan", "magenta", "olive"]
```

- [ ] **Step 1: Update `launch_interactive` drawing code**

In the `update()` function inside `launch_interactive`, replace the single-coupler
drawing block (the `if visibility["Coupler"] and sweep_data.coupler_trace_x...` block)
with:

```python
        if visibility["Coupler"] and sweep_data.coupler_traces:
            for ti, trace in enumerate(sweep_data.coupler_traces):
                color = TRACE_COLORS[ti % len(TRACE_COLORS)]
                # Draw trace path up to current step
                tx = trace.x[:step_idx + 1]
                ty = trace.y[:step_idx + 1]
                valid = ~np.isnan(tx)
                if np.any(valid):
                    ax_mech.plot(
                        tx[valid], ty[valid],
                        "-", color=color, linewidth=1.5, alpha=0.5, zorder=1,
                    )
                # Draw current coupler point + connection line to body
                cx, cy = trace.x[step_idx], trace.y[step_idx]
                if not np.isnan(cx):
                    ax_mech.plot(cx, cy, "o", color=color, markersize=5, zorder=5)
                    # Connection line to body centroid
                    body = mechanism.bodies[trace.body_id]
                    cg_g = mechanism.state.body_point_global(
                        trace.body_id, body.cg_local, q,
                    )
                    ax_mech.plot(
                        [cg_g[0], cx], [cg_g[1], cy],
                        "--", color=color, linewidth=0.8, alpha=0.4, zorder=1,
                    )
```

- [ ] **Step 2: Update `export_gifs.py` drawing code**

In the `update()` function inside `export_gif`, replace the single-coupler drawing
block with the identical multi-trace code above. Use the same `TRACE_COLORS` list
(define at module level).

- [ ] **Step 3: Fix any remaining references to old `coupler_trace_x/y` fields**

Search for `coupler_trace_x` and `coupler_trace_y` in both files. Replace any
remaining references with the new `coupler_traces` list. Also remove the now-unused
`_find_coupler_body_and_point` function.

The `launch_interactive` function signature keeps `coupler_body_id` and
`coupler_point_name` params for backward compat but they are no longer passed
to `precompute_sweep`.

- [ ] **Step 4: Verify interactive viewer and GIF export work**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import warnings; warnings.filterwarnings('ignore')
import numpy as np
from linkage_sim.viz.interactive_viewer import precompute_sweep
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity

mech = Mechanism()
ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
crank = make_bar('crank', 'A', 'B', length=2.0, mass=0.5)
crank.add_coupler_point('CP1', 1.0, 0.3)
coupler = make_bar('coupler', 'B', 'C', length=4.0, mass=1.0)
coupler.add_coupler_point('CP2', 2.0, 0.0)
rocker = make_bar('rocker', 'D', 'C', length=3.0, mass=0.8)
mech.add_body(ground)
mech.add_body(crank)
mech.add_body(coupler)
mech.add_body(rocker)
mech.add_revolute_joint('J1', 'ground', 'O2', 'crank', 'A')
mech.add_revolute_joint('J2', 'crank', 'B', 'coupler', 'B')
mech.add_revolute_joint('J3', 'coupler', 'C', 'rocker', 'C')
mech.add_revolute_joint('J4', 'ground', 'O4', 'rocker', 'D')
mech.add_revolute_driver('D1', 'ground', 'crank',
    f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0)
mech.build()
gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
sweep = precompute_sweep(mech, [gravity], n_steps=72)
print(f'Traces: {len(sweep.coupler_traces)}')
for t in sweep.coupler_traces:
    print(f'  {t.body_id}/{t.point_name}: valid={int(np.sum(~np.isnan(t.x)))}/{len(t.x)}')
"
```

Then export one GIF to verify visual:
```bash
python scripts/export_gifs.py chebyshev
```

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`

- [ ] **Step 6: Commit**

```bash
git add src/linkage_sim/viz/interactive_viewer.py scripts/export_gifs.py
git commit -m "feat: multi-trace drawing with connection lines to body centroid"
```

---

### Task 3: `Mechanism.add_trace_point()` API

**Files:**
- Modify: `src/linkage_sim/core/mechanism.py`
- Modify: `tests/test_coupler_trace.py`

This provides a convenience method on `Mechanism` to add a trace point to any body
at any position, without the user needing to access the `Body` object directly.

- [ ] **Step 1: Write tests**

Add to `tests/test_coupler_trace.py`:

```python
class TestAddTracePoint:
    """Mechanism.add_trace_point() attaches arbitrary trace points to bodies."""

    def test_adds_trace_point_to_body(self) -> None:
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_trace_point("TP1", "crank", 1.0, 0.5)
        assert "TP1" in mech.bodies["crank"].coupler_points
        pt = mech.bodies["crank"].coupler_points["TP1"]
        assert pt[0] == pytest.approx(1.0)
        assert pt[1] == pytest.approx(0.5)

    def test_trace_point_on_ground_raises(self) -> None:
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0))
        mech.add_body(ground)
        with pytest.raises(ValueError, match="ground"):
            mech.add_trace_point("TP1", "ground", 0.0, 0.0)

    def test_trace_point_unknown_body_raises(self) -> None:
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0))
        mech.add_body(ground)
        with pytest.raises(KeyError):
            mech.add_trace_point("TP1", "nonexistent", 0.0, 0.0)

    def test_trace_point_after_build_raises(self) -> None:
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.build()
        with pytest.raises(RuntimeError):
            mech.add_trace_point("TP1", "crank", 1.0, 0.0)

    def test_trace_points_appear_in_sweep(self) -> None:
        """Trace points added via add_trace_point show up in sweep traces."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0, mass=0.5)
        coupler = make_bar("coupler", "B", "C", length=4.0, mass=1.0)
        rocker = make_bar("rocker", "D", "C", length=3.0, mass=0.8)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech.add_trace_point("T1", "crank", 1.5, 0.2)
        mech.add_trace_point("T2", "rocker", 1.0, -0.3)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        sweep = precompute_sweep(mech, [], n_steps=36)
        assert len(sweep.coupler_traces) == 2
        body_ids = {t.body_id for t in sweep.coupler_traces}
        assert body_ids == {"crank", "rocker"}
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_coupler_trace.py::TestAddTracePoint -v`

- [ ] **Step 3: Implement `add_trace_point` in Mechanism**

Add to `src/linkage_sim/core/mechanism.py`:

```python
def add_trace_point(
    self,
    name: str,
    body_id: str,
    x: float,
    y: float,
) -> None:
    """Add a trace point to a body for path tracking during sweeps.

    Trace points are rigidly attached to the specified body at the given
    local coordinates. During a sweep, the viewer traces their global
    path. Multiple trace points can be added to different bodies.

    This is a convenience wrapper around Body.add_coupler_point().
    The user can also call body.add_coupler_point() directly.

    Args:
        name: Unique name for the trace point.
        body_id: ID of the body to attach to (must not be ground).
        x: X coordinate in body-local frame.
        y: Y coordinate in body-local frame.

    Raises:
        RuntimeError: If called after build().
        ValueError: If body_id is ground.
        KeyError: If body_id not found.
    """
    if self._built:
        raise RuntimeError("Cannot add trace points after build().")

    if body_id == GROUND_ID:
        raise ValueError(
            "Cannot add trace points to ground. "
            "Trace points must be on moving bodies."
        )

    body = self._get_body(body_id)
    body.add_coupler_point(name, x, y)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_coupler_trace.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/linkage_sim/core/mechanism.py tests/test_coupler_trace.py
git commit -m "feat: add Mechanism.add_trace_point() for arbitrary trace points on any link"
```

---

### Task 4: Update 6-bar viewer scripts

**Files:**
- Modify: `scripts/view_sixbar.py`
- Modify: `scripts/view_sixbar_A1.py`
- Modify: `scripts/view_sixbar_A2.py`
- Modify: `scripts/view_sixbar_B2.py`
- Modify: `scripts/view_sixbar_B3.py`

For each 6-bar script, add a second trace point on a different body to demonstrate
multi-trace. Also verify the existing coupler point coordinates make visual sense
(the point should be ON or near the body's outline, not floating in space).

- [ ] **Step 1: Update `view_sixbar.py`**

Add a second trace point on a different body (e.g., `output6` or `link5`):
```python
mech.add_trace_point("TP_output", "output6", 1.0, 0.0)
```

Remove the explicit `coupler_body_id="ternary"` and `coupler_point_name="CP"`
from the `launch_interactive` call (let auto-discovery find all trace points).

- [ ] **Step 2: Update remaining 4 sixbar scripts similarly**

For each, add a trace point on a body that doesn't already have one, and remove
explicit coupler_body_id/coupler_point_name from launch_interactive calls.

- [ ] **Step 3: Re-export all GIFs to verify**

```bash
python scripts/export_gifs.py
```

Visually inspect: each 6-bar should show multiple trace curves in different colors,
with dashed connection lines to each trace point's body centroid.

- [ ] **Step 4: Commit**

```bash
git add scripts/view_sixbar*.py
git commit -m "feat: add multi-trace demonstration to 6-bar viewer scripts"
```

---

### Task 5: Documentation and Rust port notes

**Files:**
- Modify: `RUST_MIGRATION.md`
- Modify: `ROADMAP_IMPLEMENTATION.md`

- [ ] **Step 1: Update RUST_MIGRATION.md**

Add a section under "Python Coding Conventions for Portability" or as a new section:

```markdown
### Coupler / Trace Points

Coupler points (also called trace points) are arbitrary points rigidly attached
to any moving body. They are tracked during kinematic sweeps for path tracing,
velocity/acceleration analysis, and visualization.

Python API:
- `body.add_coupler_point(name, x, y)` — low-level, on the Body object
- `mechanism.add_trace_point(name, body_id, x, y)` — convenience, on Mechanism

Rust mapping:
```rust
// On Body struct:
pub coupler_points: HashMap<String, Vector2<f64>>,

// On Mechanism:
pub fn add_trace_point(&mut self, name: &str, body_id: &str, x: f64, y: f64) {
    let body = self.bodies.get_mut(body_id).expect("body not found");
    body.coupler_points.insert(name.to_string(), Vector2::new(x, y));
}
```

The sweep/animation system collects ALL coupler points across ALL bodies and
traces each with a distinct color. The data structure per trace is:
```rust
struct CouplerTrace {
    body_id: String,
    point_name: String,
    x: Vec<f64>,  // per sweep step, NaN where failed
    y: Vec<f64>,
}
```
```

- [ ] **Step 2: Update ROADMAP_IMPLEMENTATION.md**

Add a row to the feature table documenting multi-coupler trace support and
`add_trace_point()`.

- [ ] **Step 3: Run full test suite + mypy**

```bash
pytest tests/ -x -q
mypy src/linkage_sim/ --strict
```

- [ ] **Step 4: Commit**

```bash
git add RUST_MIGRATION.md ROADMAP_IMPLEMENTATION.md
git commit -m "docs: document coupler/trace point API for multi-trace and Rust port"
```

- [ ] **Step 5: Push**

```bash
git push
```
