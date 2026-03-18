# Crank Selection Analysis & Build-Time Warning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically recommend the optimal driven link for maximum rotation, warn at build time if the choice is suboptimal, and fix all viewer scripts to work 0–360 degrees.

**Architecture:** New `analysis/crank_selection.py` provides Grashof-based 4-bar analysis and numerical probing for general mechanisms. `Mechanism.build()` emits a warning for 4-bars when the driven link limits rotation. `precompute_sweep` warns when >10% of steps fail. All viewer scripts are updated with correct crank choices, geometric initial guesses, and resized 6-bar dimensions.

**Tech Stack:** Python 3.12, numpy, pytest, warnings module. Follows existing frozen-dataclass + enum pattern from `analysis/grashof.py`.

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/linkage_sim/analysis/crank_selection.py` | Crank recommendation: Grashof-based for 4-bars, numerical probing for general mechanisms |
| Create | `tests/test_crank_selection.py` | Tests for all crank selection functions |
| Modify | `src/linkage_sim/core/mechanism.py` | Add `_check_crank_selection()` warning in `build()` |
| Modify | `src/linkage_sim/viz/interactive_viewer.py` | Add sweep failure warning in `precompute_sweep`; refactor `_detect_fourbar_link_lengths` to use new module |
| Modify | `scripts/view_crank_rocker.py` | Add geometric q0 |
| Modify | `scripts/view_double_crank.py` | Add geometric q0 |
| Modify | `scripts/view_sixbar.py` | Resize for full rotation |
| Modify | `scripts/view_sixbar_A1.py` | Resize for full rotation |
| Modify | `scripts/view_sixbar_A2.py` | Resize for full rotation |
| Modify | `scripts/view_sixbar_B2.py` | Resize for full rotation |
| Modify | `scripts/view_sixbar_B3.py` | Resize for full rotation |
| Modify | `ROADMAP_IMPLEMENTATION.md` | Document crank selection feature |

---

### Task 1: Core crank selection module — 4-bar Grashof recommendation

**Files:**
- Create: `src/linkage_sim/analysis/crank_selection.py`
- Create: `tests/test_crank_selection.py`

- [ ] **Step 1: Write tests for `detect_fourbar_topology`**

```python
# tests/test_crank_selection.py
"""Tests for crank selection analysis."""
from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.crank_selection import (
    CrankRecommendation,
    detect_fourbar_topology,
    recommend_crank_fourbar,
)
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism


def _build_fourbar(d: float, a: float, b: float, c: float) -> Mechanism:
    """Build a 4-bar with ground=d, crank=a, coupler=b, rocker=c.

    Does NOT add a driver — caller decides which link to drive.
    """
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(d, 0.0))
    crank = make_bar("crank", "A", "B", length=a, mass=0.5)
    coupler = make_bar("coupler", "B", "C", length=b, mass=0.5)
    rocker = make_bar("rocker", "D", "C", length=c, mass=0.5)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    return mech


class TestDetectFourbarTopology:
    """Detect whether a mechanism is a standard 4-bar."""

    def test_standard_fourbar_detected(self) -> None:
        mech = _build_fourbar(4.0, 2.0, 4.0, 3.0)
        result = detect_fourbar_topology(mech)
        assert result is not None
        assert result["ground_length"] == pytest.approx(4.0)

    def test_fourbar_with_driver_detected(self) -> None:
        mech = _build_fourbar(4.0, 2.0, 4.0, 3.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        result = detect_fourbar_topology(mech)
        assert result is not None

    def test_sixbar_not_detected(self) -> None:
        """A 6-bar should return None (not a 4-bar)."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 0.0), O6=(6.0, 0.0))
        mech.add_body(ground)
        for name in ["b1", "b2", "b3", "b4", "b5"]:
            mech.add_body(make_bar(name, "A", "B", length=2.0))
        # Not a 4-bar topology
        result = detect_fourbar_topology(mech)
        assert result is None

    def test_driver_on_rocker_detected(self) -> None:
        """Driver placed on rocker — driven_body_id should be rocker."""
        mech = _build_fourbar(4.0, 2.0, 4.0, 3.0)
        mech.add_revolute_driver(
            "D1", "ground", "rocker",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        result = detect_fourbar_topology(mech)
        assert result is not None
        assert result["driven_body_id"] == "rocker"

    def test_underjointed_not_detected(self) -> None:
        """3 moving bodies but only 3 revolute joints — not a 4-bar."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", length=2.0)
        coupler = make_bar("coupler", "B", "C", length=4.0)
        rocker = make_bar("rocker", "D", "C", length=3.0)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        # Missing J4: ground-rocker — only 3 joints
        result = detect_fourbar_topology(mech)
        assert result is None
```

- [ ] **Step 2: Run tests — expect FAIL (module doesn't exist)**

Run: `pytest tests/test_crank_selection.py::TestDetectFourbarTopology -v`

- [ ] **Step 3: Implement `detect_fourbar_topology` and `CrankRecommendation`**

```python
# src/linkage_sim/analysis/crank_selection.py
"""Crank selection analysis: recommend which link to drive for maximum rotation.

For 4-bar mechanisms, uses Grashof classification to determine the optimal
crank analytically. For general mechanisms (6-bar, etc.), uses numerical
probing to estimate the valid angular range of each candidate driven link.

The user can always override the recommendation by specifying their own
revolute driver — these functions are advisory, not prescriptive.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.bodies import Body
from linkage_sim.core.constraints import RevoluteJoint
from linkage_sim.core.drivers import RevoluteDriver
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.core.state import GROUND_ID


@dataclass(frozen=True)
class CrankRecommendation:
    """Recommendation for which link to drive.

    Attributes:
        body_id: Body ID of the recommended driven link.
        estimated_range_deg: Estimated angular range in degrees.
        full_rotation: True if ~360 degrees is achievable.
        reason: Human-readable explanation.
    """

    body_id: str
    estimated_range_deg: float
    full_rotation: bool
    reason: str


def detect_fourbar_topology(
    mechanism: Mechanism,
) -> dict[str, Any] | None:
    """Detect if a mechanism is a standard 4-bar and extract link data.

    A standard 4-bar has exactly 3 moving bodies connected by 4 revolute
    joints (plus optionally 1 revolute driver), with topology:
    ground—link_a—link_b—link_c—ground.

    Returns:
        Dict with keys: ground_length, ground_adjacent (dict of body_id
        to link_length for the two ground-connected bodies), coupler_id,
        coupler_length, driven_body_id (or None if no driver).
        Returns None if not a 4-bar.
    """
    moving_ids = [bid for bid in mechanism.bodies if bid != GROUND_ID]
    if len(moving_ids) != 3:
        return None

    # Separate revolute joints from drivers
    rev_joints = [j for j in mechanism.joints if isinstance(j, RevoluteJoint)]
    drivers = [j for j in mechanism.joints if isinstance(j, RevoluteDriver)]
    if len(rev_joints) != 4:
        return None

    # Find ground-connected bodies
    ground_adj: dict[str, str] = {}  # body_id -> joint_id
    for j in rev_joints:
        if j.body_i_id == GROUND_ID:
            ground_adj[j.body_j_id] = j.id
        elif j.body_j_id == GROUND_ID:
            ground_adj[j.body_i_id] = j.id
    if len(ground_adj) != 2:
        return None

    # Coupler is the remaining body
    coupler_ids = [bid for bid in moving_ids if bid not in ground_adj]
    if len(coupler_ids) != 1:
        return None
    coupler_id = coupler_ids[0]

    # Compute link lengths from attachment points
    def _link_length(body: Body) -> float:
        pts = list(body.attachment_points.values())
        if len(pts) < 2:
            return 0.0
        return float(np.linalg.norm(pts[1] - pts[0]))

    ground = mechanism.bodies[GROUND_ID]
    ground_pts = list(ground.attachment_points.values())
    if len(ground_pts) < 2:
        return None
    ground_length = float(np.linalg.norm(ground_pts[1] - ground_pts[0]))

    ground_adj_lengths = {
        bid: _link_length(mechanism.bodies[bid]) for bid in ground_adj
    }
    coupler_length = _link_length(mechanism.bodies[coupler_id])

    # Identify which body is driven (if any)
    driven_body_id: str | None = None
    if len(drivers) == 1:
        d = drivers[0]
        if d.body_i_id == GROUND_ID:
            driven_body_id = d.body_j_id
        elif d.body_j_id == GROUND_ID:
            driven_body_id = d.body_i_id

    return {
        "ground_length": ground_length,
        "ground_adjacent": ground_adj_lengths,
        "coupler_id": coupler_id,
        "coupler_length": coupler_length,
        "driven_body_id": driven_body_id,
    }
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_crank_selection.py::TestDetectFourbarTopology -v`

- [ ] **Step 5: Write tests for `recommend_crank_fourbar`**

Add to `tests/test_crank_selection.py`:

```python
class TestRecommendCrankFourbar:
    """Grashof-based crank recommendation for 4-bars."""

    def test_crank_rocker_recommends_shortest(self) -> None:
        """d=4, a=2, b=4, c=3. S=2 (crank). Should recommend crank."""
        mech = _build_fourbar(4.0, 2.0, 4.0, 3.0)
        recs = recommend_crank_fourbar(mech)
        assert recs[0].body_id == "crank"
        assert recs[0].full_rotation is True
        assert recs[0].estimated_range_deg == pytest.approx(360.0)

    def test_double_crank_either_works(self) -> None:
        """d=2 (shortest=ground). Both grounded links rotate 360."""
        mech = _build_fourbar(2.0, 4.0, 3.5, 3.0)
        recs = recommend_crank_fourbar(mech)
        assert recs[0].full_rotation is True
        assert recs[1].full_rotation is True

    def test_grashof_double_rocker_recommends_best(self) -> None:
        """S=coupler. Neither grounded link fully rotates. Recommend wider range."""
        mech = _build_fourbar(4.0, 5.0, 2.0, 5.0)
        recs = recommend_crank_fourbar(mech)
        assert recs[0].full_rotation is False
        assert recs[0].estimated_range_deg > 0

    def test_non_grashof_no_full_rotation(self) -> None:
        """S+L > P+Q. No link rotates fully."""
        mech = _build_fourbar(5.0, 3.0, 4.0, 7.0)
        recs = recommend_crank_fourbar(mech)
        assert recs[0].full_rotation is False

    def test_chebyshev_recommends_short_link(self) -> None:
        """Chebyshev: 4,5,2,5. Crank=5 is wrong; should recommend the 2-link."""
        # Build with the 2-link as "crank" name
        mech = _build_fourbar(4.0, 2.0, 5.0, 5.0)
        recs = recommend_crank_fourbar(mech)
        assert recs[0].body_id == "crank"  # length 2, shortest grounded
        assert recs[0].full_rotation is True

    def test_not_a_fourbar_returns_empty(self) -> None:
        """Non-4-bar mechanism returns empty list."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0))
        mech.add_body(ground)
        mech.add_body(make_bar("b1", "A", "B", length=2.0))
        recs = recommend_crank_fourbar(mech)
        assert recs == []
```

- [ ] **Step 6: Implement `recommend_crank_fourbar`**

Add to `src/linkage_sim/analysis/crank_selection.py`:

```python
from linkage_sim.analysis.grashof import GrashofType, check_grashof


def recommend_crank_fourbar(
    mechanism: Mechanism,
) -> list[CrankRecommendation]:
    """Rank ground-adjacent links by rotation capability for a 4-bar.

    Uses Grashof classification to determine which grounded link allows
    the widest angular sweep when used as the driven crank.

    For Grashof crank-rocker: the shortest grounded link gets 360 degrees.
    For Grashof double-crank: both grounded links get 360 degrees.
    For Grashof double-rocker: neither gets 360; estimates limited range.
    For non-Grashof: neither gets 360; estimates limited range.

    Returns:
        Ranked list of CrankRecommendation, best first. Empty if not a 4-bar.
    """
    topo = detect_fourbar_topology(mechanism)
    if topo is None:
        return []

    ground_len = topo["ground_length"]
    coupler_len = topo["coupler_length"]
    adj: dict[str, float] = topo["ground_adjacent"]
    body_ids = list(adj.keys())

    recommendations: list[CrankRecommendation] = []

    for body_id in body_ids:
        crank_len = adj[body_id]
        rocker_id = [bid for bid in body_ids if bid != body_id][0]
        rocker_len = adj[rocker_id]

        result = check_grashof(ground_len, crank_len, coupler_len, rocker_len)

        if result.classification == GrashofType.GRASHOF_CRANK_ROCKER:
            if result.shortest_is == "crank":
                recommendations.append(CrankRecommendation(
                    body_id=body_id,
                    estimated_range_deg=360.0,
                    full_rotation=True,
                    reason=(
                        f"Grashof crank-rocker: {body_id} (length={crank_len}) "
                        f"is the shortest grounded link"
                    ),
                ))
            else:
                # This body is the rocker, estimate its range
                est = _estimate_driven_range_analytical(
                    ground_len, crank_len, coupler_len, rocker_len,
                )
                recommendations.append(CrankRecommendation(
                    body_id=body_id,
                    estimated_range_deg=est,
                    full_rotation=False,
                    reason=(
                        f"{body_id} (length={crank_len}) is not the shortest "
                        f"grounded link — limited oscillation range"
                    ),
                ))
        elif result.classification == GrashofType.GRASHOF_DOUBLE_CRANK:
            recommendations.append(CrankRecommendation(
                body_id=body_id,
                estimated_range_deg=360.0,
                full_rotation=True,
                reason=(
                    f"Grashof double-crank: ground is shortest, "
                    f"both {body_id} and {rocker_id} rotate 360 deg"
                ),
            ))
        elif result.classification == GrashofType.CHANGE_POINT:
            recommendations.append(CrankRecommendation(
                body_id=body_id,
                estimated_range_deg=360.0,
                full_rotation=True,
                reason=(
                    f"Change-point mechanism: {body_id} can rotate 360 deg "
                    f"(singular at collinear configurations)"
                ),
            ))
        else:
            # Non-Grashof or double-rocker: estimate range
            est = _estimate_driven_range_analytical(
                ground_len, crank_len, coupler_len, rocker_len,
            )
            recommendations.append(CrankRecommendation(
                body_id=body_id,
                estimated_range_deg=est,
                full_rotation=False,
                reason=(
                    f"{result.classification.name}: {body_id} "
                    f"(length={crank_len}) has limited range ~{est:.0f} deg"
                ),
            ))

    # Sort: full rotation first, then by range descending
    recommendations.sort(
        key=lambda r: (-int(r.full_rotation), -r.estimated_range_deg),
    )
    return recommendations


def _estimate_driven_range_analytical(
    ground_len: float,
    crank_len: float,
    coupler_len: float,
    rocker_len: float,
) -> float:
    """Estimate the angular sweep of a non-full-rotation grounded link.

    Uses the closure condition: |coupler - rocker| <= d_tip <= coupler + rocker
    where d_tip = distance from crank tip to the other ground pivot.
    Scans 3600 angles and counts how many satisfy closure.
    """
    angles = np.linspace(0, 2 * np.pi, 3600, endpoint=False)
    # d_tip^2 = crank^2 + ground^2 - 2*crank*ground*cos(theta)
    d_tip_sq = (
        crank_len**2 + ground_len**2
        - 2 * crank_len * ground_len * np.cos(angles)
    )
    d_tip = np.sqrt(np.maximum(d_tip_sq, 0.0))

    lo = abs(coupler_len - rocker_len)
    hi = coupler_len + rocker_len
    valid = (d_tip >= lo) & (d_tip <= hi)

    return float(np.sum(valid)) / 3600.0 * 360.0
```

- [ ] **Step 7: Run tests — expect PASS**

Run: `pytest tests/test_crank_selection.py -v`

- [ ] **Step 8: Commit**

```bash
git add src/linkage_sim/analysis/crank_selection.py tests/test_crank_selection.py
git commit -m "feat: add crank selection analysis with Grashof-based 4-bar recommendation"
```

---

### Task 2: Numerical probing for general mechanisms

**Files:**
- Modify: `src/linkage_sim/analysis/crank_selection.py`
- Modify: `tests/test_crank_selection.py`

- [ ] **Step 1: Write test for `estimate_driven_range`**

Add to `tests/test_crank_selection.py`:

```python
from linkage_sim.analysis.crank_selection import estimate_driven_range


class TestEstimateDrivenRange:
    """Numerical probing of driven link range."""

    def test_crank_rocker_full_range(self) -> None:
        """Grashof crank-rocker: driven link achieves ~360 deg."""
        mech = _build_fourbar(4.0, 2.0, 4.0, 3.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        q0 = mech.state.make_q()
        est = estimate_driven_range(mech, q0, n_probes=72)
        assert est >= 355.0  # ~360 with probing tolerance

    def test_non_grashof_limited_range(self) -> None:
        """Non-Grashof: driven link has limited range."""
        mech = _build_fourbar(5.0, 3.0, 4.0, 7.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        q0 = mech.state.make_q()
        est = estimate_driven_range(mech, q0, n_probes=72)
        assert est < 360.0
        assert est > 0.0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_crank_selection.py::TestEstimateDrivenRange -v`

- [ ] **Step 3: Implement `estimate_driven_range`**

Add to `src/linkage_sim/analysis/crank_selection.py`:

```python
from linkage_sim.solvers.kinematics import solve_position


def estimate_driven_range(
    mechanism: Mechanism,
    q0: NDArray[np.float64],
    n_probes: int = 72,
) -> float:
    """Estimate valid angular range of the current driver by probing.

    Attempts to solve the mechanism at n_probes evenly spaced angles
    (0 to 360 degrees). Returns the estimated range in degrees based
    on how many probes converge.

    This works for any mechanism topology (4-bar, 6-bar, etc.).

    Assumes the driver uses f(t) = t (identity), so t maps directly
    to the driven angle in radians. This is the convention used by
    all viewer scripts.

    Args:
        mechanism: A built Mechanism with an identity revolute driver.
        q0: Initial guess for the solver.
        n_probes: Number of angles to test (default 72 = every 5 degrees).

    Returns:
        Estimated valid range in degrees (0 to 360).
    """
    angles = np.linspace(0, 2 * np.pi, n_probes, endpoint=False)
    n_converged = 0
    q_current = q0.copy()

    for angle in angles:
        result = solve_position(mechanism, q_current, t=float(angle))
        if result.converged:
            n_converged += 1
            q_current = result.q.copy()

    return n_converged / n_probes * 360.0
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_crank_selection.py::TestEstimateDrivenRange -v`

- [ ] **Step 5: Commit**

```bash
git add src/linkage_sim/analysis/crank_selection.py tests/test_crank_selection.py
git commit -m "feat: add numerical probing for driven link range estimation"
```

---

### Task 3: Build-time warning in `Mechanism.build()`

**Files:**
- Modify: `src/linkage_sim/core/mechanism.py`
- Modify: `tests/test_crank_selection.py`

- [ ] **Step 1: Write test for build warning**

Add to `tests/test_crank_selection.py`:

```python
import warnings


class TestBuildWarning:
    """Mechanism.build() warns when driven link limits rotation."""

    def test_warns_on_suboptimal_crank(self) -> None:
        """d=4, a=3, b=4, c=2. Rocker(2) is shortest grounded link.
        Driving crank(3) instead of rocker(2) should warn."""
        mech = _build_fourbar(4.0, 3.0, 4.0, 2.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mech.build()
            crank_warnings = [
                x for x in w if "crank selection" in str(x.message).lower()
            ]
            assert len(crank_warnings) == 1

    def test_no_warning_on_optimal_crank(self) -> None:
        """Driving the shortest link should not warn."""
        mech = _build_fourbar(4.0, 2.0, 4.0, 3.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mech.build()
            crank_warnings = [
                x for x in w if "crank selection" in str(x.message).lower()
            ]
            assert len(crank_warnings) == 0

    def test_no_warning_on_non_fourbar(self) -> None:
        """Non-4-bar mechanisms should not trigger the warning."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0))
        mech.add_body(ground)
        mech.add_body(make_bar("b1", "A", "B", length=2.0))
        mech.add_revolute_joint("J1", "ground", "O2", "b1", "A")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mech.build()
            crank_warnings = [
                x for x in w if "crank selection" in str(x.message).lower()
            ]
            assert len(crank_warnings) == 0
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `pytest tests/test_crank_selection.py::TestBuildWarning -v`

- [ ] **Step 3: Implement build-time warning**

Modify `src/linkage_sim/core/mechanism.py` — add to `build()` method after `self._built = True`:

```python
import warnings

# In build(), after self._built = True:
self._check_crank_selection()
```

Add private method to `Mechanism`:

```python
def _check_crank_selection(self) -> None:
    """Warn if a 4-bar's driven link is suboptimal for rotation.

    Only fires for 4-bar mechanisms with a revolute driver.
    The user can override by specifying their own driver — this
    warning is advisory, not blocking.
    """
    from linkage_sim.analysis.crank_selection import (
        detect_fourbar_topology,
        recommend_crank_fourbar,
    )

    topo = detect_fourbar_topology(self)
    if topo is None:
        return  # Not a 4-bar

    driven_id = topo["driven_body_id"]
    if driven_id is None:
        return  # No driver

    recs = recommend_crank_fourbar(self)
    if not recs:
        return

    best = recs[0]
    if best.body_id != driven_id and best.full_rotation:
        # Current driver is not optimal
        driven_len = topo["ground_adjacent"].get(driven_id, "?")
        warnings.warn(
            f"Crank selection: driven link '{driven_id}' "
            f"(length={driven_len}) cannot make full rotation. "
            f"For full 360-degree rotation, drive '{best.body_id}' "
            f"instead ({best.reason}). "
            f"Override by specifying your own driver if limited "
            f"range is intentional.",
            stacklevel=2,
        )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_crank_selection.py::TestBuildWarning -v`

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `pytest tests/ -x -q`
Expected: All existing tests still pass (some may now emit warnings — that's OK).

- [ ] **Step 6: Commit**

```bash
git add src/linkage_sim/core/mechanism.py tests/test_crank_selection.py
git commit -m "feat: add build-time warning when driven link limits rotation"
```

---

### Task 4: Sweep failure warning + DRY refactor in `interactive_viewer.py`

**Files:**
- Modify: `src/linkage_sim/viz/interactive_viewer.py`
- Modify: `tests/test_crank_selection.py`

- [ ] **Step 1: Write test for sweep failure warning**

Add to `tests/test_crank_selection.py`:

```python
class TestSweepWarning:
    """precompute_sweep warns when many steps fail."""

    def test_warns_on_high_failure_rate(self) -> None:
        """Non-Grashof 4-bar: >10% failures should trigger warning."""
        mech = _build_fourbar(5.0, 3.0, 4.0, 7.0)
        mech.add_revolute_driver(
            "D1", "ground", "crank",
            f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
        )
        mech.build()
        from linkage_sim.forces.gravity import Gravity
        from linkage_sim.viz.interactive_viewer import precompute_sweep
        gravity = Gravity(
            g_vector=np.array([0.0, -9.81]),
            bodies=mech.bodies,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            precompute_sweep(mech, [gravity], n_steps=72)
            sweep_warnings = [
                x for x in w if "sweep steps failed" in str(x.message).lower()
            ]
            assert len(sweep_warnings) == 1
```

- [ ] **Step 2: Add warning after sweep computation**

In `precompute_sweep()`, after `solutions = sweep.solutions` and before the
velocity/statics loop, add:

```python
import warnings as _warnings

n_failed = sum(1 for s in solutions if s is None)
if n_failed > n_steps * 0.1:  # >10% failure rate
    _warnings.warn(
        f"{n_failed} of {n_steps} sweep steps failed to converge. "
        f"The driven link may not be able to complete full rotation "
        f"with this mechanism's dimensions. "
        f"Use analysis.crank_selection.recommend_crank_fourbar() "
        f"(for 4-bars) or analysis.crank_selection.estimate_driven_range() "
        f"(for general mechanisms) to diagnose.",
        stacklevel=2,
    )
```

- [ ] **Step 3: Refactor `_detect_fourbar_link_lengths` to use `detect_fourbar_topology`**

Replace the existing `_detect_fourbar_link_lengths` function in `interactive_viewer.py`
with a thin wrapper that delegates to the new module to eliminate DRY violation:

```python
def _detect_fourbar_link_lengths(
    mechanism: Mechanism,
) -> tuple[float, float, float, float] | None:
    from linkage_sim.analysis.crank_selection import detect_fourbar_topology
    topo = detect_fourbar_topology(mechanism)
    if topo is None:
        return None
    adj = topo["ground_adjacent"]
    adj_ids = list(adj.keys())
    # Return (a, b, c, d) = (crank, coupler, rocker, ground)
    return (adj[adj_ids[0]], topo["coupler_length"], adj[adj_ids[1]], topo["ground_length"])
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_crank_selection.py::TestSweepWarning -v`

- [ ] **Step 5: Commit**

```bash
git add src/linkage_sim/viz/interactive_viewer.py tests/test_crank_selection.py
git commit -m "feat: sweep failure warning + refactor 4-bar detection to use crank_selection module"
```

---

### Task 5: Fix 4-bar viewer scripts with geometric q0

**Files:**
- Modify: `scripts/view_crank_rocker.py`
- Modify: `scripts/view_double_crank.py`

Both scripts fail at theta=0 because the default all-zeros initial guess doesn't
converge. Add a geometric q0 computation (same pattern as the Chebyshev fix).

- [ ] **Step 1: Fix `view_crank_rocker.py`**

Change `build_crank_rocker_with_gravity` return type to `tuple[Mechanism, list, np.ndarray]`
and add geometric q0 computation:

```python
def build_crank_rocker_with_gravity() -> tuple[Mechanism, list, np.ndarray]:
    # ... existing mechanism build code unchanged ...

    # Geometric initial guess at crank angle 0
    # Crank tip B at (a, 0). Find C on coupler-rocker circles.
    bx, by = a, 0.0
    dist_bo4 = np.hypot(bx - d, by)
    cos_beta = (b**2 + dist_bo4**2 - c**2) / (2 * b * dist_bo4)
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    angle_b_to_o4 = np.arctan2(-by, d - bx)
    theta_coupler = angle_b_to_o4 + beta
    cx = bx + b * np.cos(theta_coupler)
    cy = by + b * np.sin(theta_coupler)
    theta_rocker = np.arctan2(cy, cx - d)

    q0 = mech.state.make_q()
    mech.state.set_pose("crank", q0, 0.0, 0.0, 0.0)
    mech.state.set_pose("coupler", q0, bx, by, float(theta_coupler))
    mech.state.set_pose("rocker", q0, d, 0.0, float(theta_rocker))

    return mech, [gravity], q0
```

Update `main()` to unpack q0 and pass to `launch_interactive(..., q0=q0)`.

- [ ] **Step 2: Fix `view_double_crank.py`**

Same pattern — add geometric q0 at crank angle 0.

- [ ] **Step 3: Verify both scripts achieve 360/360**

```bash
python -c "
import sys; sys.path.insert(0,'.')
exec(open('scripts/view_crank_rocker.py').read().replace('if __name__','if False'))
import numpy as np
from linkage_sim.viz.interactive_viewer import precompute_sweep
mech, forces, q0 = build_crank_rocker_with_gravity()
s = precompute_sweep(mech, forces, n_steps=360, q0=q0)
ok = sum(1 for x in s.solutions if x is not None)
print(f'crank_rocker: {ok}/360')
"
```

Expected: `crank_rocker: 360/360` and `double_crank: 360/360`

- [ ] **Step 4: Commit**

```bash
git add scripts/view_crank_rocker.py scripts/view_double_crank.py
git commit -m "fix: add geometric initial guess to crank_rocker and double_crank viewers"
```

---

### Task 6: Resize 6-bar mechanisms for full rotation

**Files:**
- Modify: `scripts/view_sixbar.py`
- Modify: `scripts/view_sixbar_A1.py`
- Modify: `scripts/view_sixbar_A2.py`
- Modify: `scripts/view_sixbar_B2.py`
- Modify: `scripts/view_sixbar_B3.py`

The 6-bar mechanisms fail because their link dimensions don't allow the driven
link to complete full rotation. Each has two kinematic loops — BOTH must satisfy
closure at all crank angles.

**Strategy for each script:**
1. Identify the two 4-bar sub-loops
2. Check Grashof condition for each sub-loop
3. Resize links so the crank is the shortest link in BOTH loops
4. Use `estimate_driven_range` to verify 360/360

**Sizing rule of thumb:** In each sub-loop containing the crank, ensure
`crank_arm + longest_other <= sum(remaining_two)`. The crank arm is typically
the distance from the ground pivot to the nearest ternary attachment point.

- [ ] **Step 1: Resize `view_sixbar.py` (B1 — ternary ground, non-adjacent ternaries)**

Sub-loops:
- Loop 1: O2—crank(1.5)—ternary(P1-P2=3.0)—rocker4(2.5)—O6(6,0)
  Ground dist = 6.0. Links: 6, 1.5, 3, 2.5. S+L=7.5 > P+Q=5.5 → **Non-Grashof!**
- Loop 2: O4(3,1)—output6(2.0)—link5(2.0)—ternary(P3, offset 1.5,1 from P1)

Fix loop 1: Move O6 to (3.5, 0). New ground dist = 3.5.
Links: 3.5, 1.5, 3, 2.5 → S+L=5.0, P+Q=5.5 → **Grashof crank-rocker!**
Fix loop 2: May need to adjust O4 position and/or Link5/Output6 lengths
to ensure P3's trajectory is always reachable. Increase Link5 and Output6
to 3.0 if needed for margin.

- [ ] **Step 2: Resize `view_sixbar_A1.py` (Chain A, binary ground, adjacent ternaries)**

Sub-loops (T1 driven at O2):
- Loop 1: O2—T1(P1-P2=1.5)—B2(2.0)—B3(2.0)—T2(Q2)—...complex
  Since T1 and T2 are adjacent via J3: T1(P3)—T2(Q1), the two loops share
  the T1-T2 edge. Need T1's arms (1.5 and 1.0) short enough relative to
  connected links. Reduce ground dist O2-O4 if needed, or increase B2/B3.

- Loop 2: O4—B4(2.0)—T2(Q3)—...
  Ensure B4 + reachable T2 range covers full rotation.

Strategy: Reduce O2-O4 distance from 3.0 to 2.0, keep link lengths the same.
Verify with `estimate_driven_range`.

- [ ] **Step 3: Resize `view_sixbar_A2.py` (Chain A, ternary ground)**

Sub-loops (B1 driven at O2, length=1.0):
- Loop 1: O2—B1(1.0)—B4(2.5)—T2(Q3, arm ~1.2)—O6(5,0)
  Ground dist O2-O6 = 5. Links: 5, 1.0, 2.5, ~1.8. Check Grashof.
- Loop 2: O4(3,0)—B2(1.5)—B3(2.5)—T2(Q2, arm 2.5)—reachable from T2 origin

Likely issue: O2-O6=5 is too large for crank=1.0.
Fix: Reduce O6 to (3.5, 0) or increase B4 and T2 arm lengths.

- [ ] **Step 4: Resize `view_sixbar_B2.py` (Chain B, non-adjacent ternaries)**

Sub-loops (T1 driven at O2):
- Loop 1: O2—T1(P1-P2=1.2)—B2(1.5)—T2(Q2=1.2)—O4(3,0)
  Ground dist = 3. Links: 3, 1.2, 1.5, 1.2. S+L=1.2+3=4.2, P+Q=1.5+1.2=2.7 → **Non-Grashof!**
- Loop 2: T1(P3, arm 0.78)—B3(2.0)—B4(2.0)—T2(Q3, arm 0.78)

Fix loop 1: Reduce O4 to (2.0, 0). New: 2, 1.2, 1.5, 1.2.
S+L=1.2+2=3.2, P+Q=1.5+1.2=2.7 → still fails. Need larger B2.
Try B2=2.5: 2, 1.2, 2.5, 1.2. S+L=1.2+2.5=3.7, P+Q=2+1.2=3.2 → still fails.
Try: O4=(2.5,0), T1 P2=(1.5,0), B2=2.5, T2 Q2=(1.5,0).
Links: 2.5, 1.5, 2.5, 1.5. S+L=1.5+2.5=4, P+Q=2.5+1.5=4 → change-point. OK.

- [ ] **Step 5: Resize `view_sixbar_B3.py` (Chain B, non-adjacent ternaries)**

Sub-loops (T1 driven at O2):
- Loop 1: O2—T1(P1-P2=2.5)—B1(2.5)—T2(Q1)—...
  Ground dist O2-O4 = 6. T2 floats (not grounded directly in loop 1).
  This loop has: O2—T1(arm 2.5)—B1(2.5)—T2—B2(2.5)—T1(P3, arm ~1.56)
  Complex coupling. Both T1 arms + B1 + B2 must reach T2.
- Loop 2: O4—B4(2.5)—T2(Q3, arm ~1.56)

Fix: Reduce O4 from (6,0) to (4,0). This shortens the ground link for
loop 2, giving B4 more reach relative to T2's trajectory.

- [ ] **Step 6: Verify all 6-bar scripts achieve 360/360**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import numpy as np
from linkage_sim.viz.interactive_viewer import precompute_sweep

for name in ['view_sixbar','view_sixbar_A1','view_sixbar_A2','view_sixbar_B2','view_sixbar_B3']:
    ns = {}
    exec(open(f'scripts/{name}.py').read().replace('if __name__','if False'), ns)
    build = [f for n,f in ns.items() if callable(f) and n.startswith('build_')][0]
    r = build()
    mech, forces, q0 = r[0], r[1], r[2] if len(r)==3 else None
    s = precompute_sweep(mech, forces, n_steps=360, q0=q0)
    ok = sum(1 for x in s.solutions if x is not None)
    print(f'{name}: {ok}/360')
"
```

Expected: All `360/360`.

- [ ] **Step 7: Commit**

```bash
git add scripts/view_sixbar*.py
git commit -m "fix: resize 6-bar mechanisms for full 360-degree crank rotation"
```

---

### Task 7: Documentation and final verification

**Files:**
- Modify: `ROADMAP_IMPLEMENTATION.md`

- [ ] **Step 1: Add crank selection docs to ROADMAP_IMPLEMENTATION.md**

Add a section documenting:
- The `recommend_crank_fourbar()` and `estimate_driven_range()` API
- The build-time warning behavior
- How to override the recommendation (just specify your own driver)
- The Grashof classification used for 4-bars

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 3: Run mypy**

Run: `mypy src/linkage_sim/ --strict`
Expected: Clean.

- [ ] **Step 4: Verify all viewer scripts**

Run the verification script from Task 6 Step 6, including the 4-bar scripts.
All should show 360/360 (except `view_double_rocker` which is intentionally limited).

- [ ] **Step 5: Commit docs**

```bash
git add ROADMAP_IMPLEMENTATION.md
git commit -m "docs: document crank selection analysis and override pattern"
```
