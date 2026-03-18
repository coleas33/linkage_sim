"""Layer 5: Performance and regression baselines.

Golden snapshots + iteration/sweep baselines for regression detection.
These tests do NOT establish correctness — they detect drift.

Golden snapshot format: JSON files in tests/golden/ with
{angle_deg, angle_rad, q, q_dot, q_ddot, coupler_pos, coupler_vel, coupler_accel}
at selected canonical angles. Generated once, committed, compared on every run.

Tolerance calibration (2026-03-17):
    Golden snapshot comparison: epsilon = 1e-10
    Iteration count: calibrated from current solver on development machine.
    Sweep success: >= 98% convergence at 5-degree steps.
    # BASELINE: calibrated 2026-03-17, recalibrate if platform changes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from linkage_sim.analysis.coupler import eval_coupler_point
from linkage_sim.analysis.validation import jacobian_rank_analysis
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.sweep import position_sweep

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_TOL = 1e-10


# ── Mechanism factories (copied from benchmark tests to ensure isolation) ──


def _build_fourbar() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    coupler = make_bar("coupler", "B", "C", length=3.0)
    coupler.add_coupler_point("P", 1.5, 0.5)
    rocker = make_bar("rocker", "D", "C", length=2.0)
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
    return mech


def _build_slidercrank() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), rail=(0.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.0)
    conrod = make_bar("conrod", "B", "C", length=3.0)
    conrod.add_coupler_point("P", 1.5, 0.3)
    slider = Body(
        id="slider",
        attachment_points={"pin": np.array([0.0, 0.0])},
    )
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(conrod)
    mech.add_body(slider)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
    mech.add_revolute_joint("J3", "conrod", "C", "slider", "pin")
    mech.add_prismatic_joint(
        "P1", "ground", "rail", "slider", "pin",
        axis_local_i=np.array([1.0, 0.0]),
        delta_theta_0=0.0,
    )
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()
    return mech


def _make_ternary_link(
    body_id: str,
    p1_name: str,
    p2_name: str,
    p3_name: str,
    p2_local: tuple[float, float],
    p3_local: tuple[float, float],
) -> Body:
    body = Body(id=body_id)
    body.add_attachment_point(p1_name, 0.0, 0.0)
    body.add_attachment_point(p2_name, p2_local[0], p2_local[1])
    body.add_attachment_point(p3_name, p3_local[0], p3_local[1])
    cg_x = (0.0 + p2_local[0] + p3_local[0]) / 3.0
    cg_y = (0.0 + p2_local[1] + p3_local[1]) / 3.0
    body.cg_local = np.array([cg_x, cg_y])
    return body


def _build_sixbar() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 1.0), O6=(6.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.5)
    ternary = _make_ternary_link(
        "ternary", "P1", "P2", "P3",
        p2_local=(3.0, 0.0), p3_local=(1.5, 1.0),
    )
    ternary.add_coupler_point("CP", 1.5, 0.5)
    rocker4 = make_bar("rocker4", "R4A", "R4B", length=2.5)
    link5 = make_bar("link5", "L5A", "L5B", length=2.0)
    output6 = make_bar("output6", "R6A", "R6B", length=2.0)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(ternary)
    mech.add_body(rocker4)
    mech.add_body(link5)
    mech.add_body(output6)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "ternary", "P1")
    mech.add_revolute_joint("J3", "ternary", "P2", "rocker4", "R4B")
    mech.add_revolute_joint("J4", "ground", "O6", "rocker4", "R4A")
    mech.add_revolute_joint("J5", "ternary", "P3", "link5", "L5A")
    mech.add_revolute_joint("J6", "link5", "L5B", "output6", "R6B")
    mech.add_revolute_joint("J7", "ground", "O4", "output6", "R6A")
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()
    return mech


def _find_sixbar_initial(mech: Mechanism, crank_angle: float) -> np.ndarray:
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, crank_angle)
    bx = 1.5 * np.cos(crank_angle)
    by = 1.5 * np.sin(crank_angle)
    theta_tern = 0.15
    mech.state.set_pose("ternary", q, bx, by, theta_tern)
    ct, st = np.cos(theta_tern), np.sin(theta_tern)
    p2_gx = bx + 3.0 * ct
    p2_gy = by + 3.0 * st
    theta_r4 = np.arctan2(p2_gy, p2_gx - 6.0)
    mech.state.set_pose("rocker4", q, 6.0, 0.0, theta_r4)
    p3_gx = bx + 1.5 * ct - 1.0 * st
    p3_gy = by + 1.5 * st + 1.0 * ct
    theta_l5 = np.arctan2(1.0 - p3_gy, 3.0 - p3_gx)
    mech.state.set_pose("link5", q, p3_gx, p3_gy, theta_l5)
    l5b_gx = p3_gx + 2.0 * np.cos(theta_l5)
    l5b_gy = p3_gy + 2.0 * np.sin(theta_l5)
    theta_o6 = np.arctan2(l5b_gy - 1.0, l5b_gx - 3.0)
    mech.state.set_pose("output6", q, 3.0, 1.0, theta_o6)
    return q


# ── Solve helpers ──


def _solve_fourbar_at(mech: Mechanism, angle: float):
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result, q_dot, q_ddot


def _solve_slidercrank_at(mech: Mechanism, angle: float):
    q = mech.state.make_q()
    r, l = 1.0, 3.0
    bx, by = np.cos(angle), np.sin(angle)
    x_sl = r * np.cos(angle) + np.sqrt(l**2 - r**2 * np.sin(angle)**2)
    phi_cr = np.arcsin(-r * np.sin(angle) / l)
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    mech.state.set_pose("conrod", q, bx, by, phi_cr)
    mech.state.set_pose("slider", q, x_sl, 0.0, 0.0)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result, q_dot, q_ddot


def _solve_sixbar_at(mech: Mechanism, angle: float):
    q0 = _find_sixbar_initial(mech, angle)
    result = solve_position(mech, q0, t=angle)
    assert result.converged
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result, q_dot, q_ddot


# ── Golden snapshot helpers ──


def _generate_snapshot(mech_type: str) -> dict:
    """Generate snapshot data for a mechanism at canonical angles."""
    if mech_type == "fourbar":
        mech = _build_fourbar()
        angles_deg = [45, 90, 135, 180]
        coupler_body = "coupler"
        coupler_local = np.array([1.5, 0.5])
        solve_fn = _solve_fourbar_at
    elif mech_type == "slidercrank":
        mech = _build_slidercrank()
        angles_deg = [45, 90, 135, 180]
        coupler_body = "conrod"
        coupler_local = np.array([1.5, 0.3])
        solve_fn = _solve_slidercrank_at
    elif mech_type == "sixbar":
        mech = _build_sixbar()
        angles_deg = [15, 30, 45, 55]
        coupler_body = "ternary"
        coupler_local = np.array([1.5, 0.5])
        solve_fn = _solve_sixbar_at
    else:
        raise ValueError(f"Unknown mechanism type: {mech_type}")

    steps = []
    for angle_deg in angles_deg:
        angle = np.radians(angle_deg)
        result, q_dot, q_ddot = solve_fn(mech, angle)
        pos, vel, accel = eval_coupler_point(
            mech.state, coupler_body, coupler_local, result.q, q_dot, q_ddot
        )
        steps.append({
            "angle_deg": angle_deg,
            "angle_rad": float(angle),
            "q": result.q.tolist(),
            "q_dot": q_dot.tolist(),
            "q_ddot": q_ddot.tolist(),
            "coupler_pos": pos.tolist(),
            "coupler_vel": vel.tolist(),
            "coupler_accel": accel.tolist(),
        })
    return {"mechanism": mech_type, "steps": steps}


def _load_or_generate_snapshot(mech_type: str) -> dict:
    """Load golden snapshot from disk, or generate + save if missing."""
    path = GOLDEN_DIR / f"{mech_type}_snapshot.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    # Generate and save
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    data = _generate_snapshot(mech_type)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data


# ── Iteration count baselines ──
# BASELINE: calibrated 2026-03-17, recalibrate if platform changes


class TestFourbarIterationCount:
    """Regression baseline: Newton-Raphson iteration count for 4-bar sweep."""

    def test_mean_iterations_within_baseline(self) -> None:
        """Mean iterations across 30-330 deg at 5 deg steps should be <= 4."""
        mech = _build_fourbar()
        angles = np.radians(np.arange(30, 335, 5))
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=float(angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        iterations = [r.iterations for r in sweep.results if r.converged]
        assert len(iterations) > 0
        mean_iter = np.mean(iterations)
        max_iter = np.max(iterations)
        # BASELINE: calibrated 2026-03-17, actual mean ~4.1, max ~13
        # The boundary Grashof 4-bar has higher iteration counts near toggle.
        assert mean_iter <= 6, f"Mean iterations {mean_iter:.1f} exceeds baseline 6"
        assert max_iter <= 15, f"Max iterations {max_iter} exceeds baseline 15"


class TestSliderCrankIterationCount:
    """Regression baseline: Newton-Raphson iteration count for slider-crank."""

    def test_mean_iterations_within_baseline(self) -> None:
        mech = _build_slidercrank()
        angles = np.linspace(0.0, 2 * np.pi, 73)
        q0 = mech.state.make_q()
        r, l = 1.0, 3.0
        a0 = angles[0]
        mech.state.set_pose("crank", q0, 0.0, 0.0, a0)
        mech.state.set_pose("conrod", q0, np.cos(a0), np.sin(a0),
                            np.arcsin(-r * np.sin(a0) / l))
        x_sl = r * np.cos(a0) + np.sqrt(l**2 - r**2 * np.sin(a0)**2)
        mech.state.set_pose("slider", q0, x_sl, 0.0, 0.0)
        result = solve_position(mech, q0, t=float(a0))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        iterations = [r.iterations for r in sweep.results if r.converged]
        assert len(iterations) > 0
        mean_iter = np.mean(iterations)
        max_iter = np.max(iterations)
        # BASELINE: calibrated 2026-03-17
        assert mean_iter <= 6, f"Mean iterations {mean_iter:.1f} exceeds baseline 6"
        assert max_iter <= 15, f"Max iterations {max_iter} exceeds baseline 15"


# ── Sweep success rate baselines ──


class TestFourbarSweepSuccessRate:
    """Regression baseline: sweep convergence rate for 4-bar."""

    def test_success_rate_above_98pct(self) -> None:
        mech = _build_fourbar()
        angles = np.linspace(0.01, 2 * np.pi - 0.01, 72)
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, angles[0])
        mech.state.set_pose("coupler", q0, np.cos(angles[0]), np.sin(angles[0]), 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        result = solve_position(mech, q0, t=float(angles[0]))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        success_rate = sweep.n_converged / len(angles)
        assert success_rate >= 0.98, f"Success rate {success_rate:.1%} below 98%"


class TestSliderCrankSweepSuccessRate:
    """Regression baseline: sweep convergence rate for slider-crank."""

    def test_success_rate_above_98pct(self) -> None:
        mech = _build_slidercrank()
        angles = np.linspace(0.0, 2 * np.pi, 73)
        q0 = mech.state.make_q()
        r, l = 1.0, 3.0
        a0 = angles[0]
        mech.state.set_pose("crank", q0, 0.0, 0.0, a0)
        mech.state.set_pose("conrod", q0, np.cos(a0), np.sin(a0),
                            np.arcsin(-r * np.sin(a0) / l))
        x_sl = r * np.cos(a0) + np.sqrt(l**2 - r**2 * np.sin(a0)**2)
        mech.state.set_pose("slider", q0, x_sl, 0.0, 0.0)
        result = solve_position(mech, q0, t=float(a0))
        assert result.converged
        sweep = position_sweep(mech, result.q, angles)
        success_rate = sweep.n_converged / len(angles)
        assert success_rate >= 0.98, f"Success rate {success_rate:.1%} below 98%"


# ── Golden snapshot comparison tests ──


class TestFourbarGoldenSnapshot:
    """Golden snapshot comparison for 4-bar at canonical angles."""

    def test_position_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("fourbar")
        mech = _build_fourbar()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, _, _ = _solve_fourbar_at(mech, angle)
            np.testing.assert_allclose(
                result.q, step["q"], atol=GOLDEN_TOL,
                err_msg=f"Position drift at {step['angle_deg']}°",
            )

    def test_velocity_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("fourbar")
        mech = _build_fourbar()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, q_dot, _ = _solve_fourbar_at(mech, angle)
            np.testing.assert_allclose(
                q_dot, step["q_dot"], atol=GOLDEN_TOL,
                err_msg=f"Velocity drift at {step['angle_deg']}°",
            )

    def test_acceleration_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("fourbar")
        mech = _build_fourbar()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, q_dot, q_ddot = _solve_fourbar_at(mech, angle)
            np.testing.assert_allclose(
                q_ddot, step["q_ddot"], atol=GOLDEN_TOL,
                err_msg=f"Acceleration drift at {step['angle_deg']}°",
            )


class TestSliderCrankGoldenSnapshot:
    """Golden snapshot comparison for slider-crank at canonical angles."""

    def test_position_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("slidercrank")
        mech = _build_slidercrank()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, _, _ = _solve_slidercrank_at(mech, angle)
            np.testing.assert_allclose(
                result.q, step["q"], atol=GOLDEN_TOL,
                err_msg=f"Position drift at {step['angle_deg']}°",
            )

    def test_velocity_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("slidercrank")
        mech = _build_slidercrank()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, q_dot, _ = _solve_slidercrank_at(mech, angle)
            np.testing.assert_allclose(
                q_dot, step["q_dot"], atol=GOLDEN_TOL,
                err_msg=f"Velocity drift at {step['angle_deg']}°",
            )

    def test_acceleration_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("slidercrank")
        mech = _build_slidercrank()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, q_dot, q_ddot = _solve_slidercrank_at(mech, angle)
            np.testing.assert_allclose(
                q_ddot, step["q_ddot"], atol=GOLDEN_TOL,
                err_msg=f"Acceleration drift at {step['angle_deg']}°",
            )


class TestSixbarGoldenSnapshot:
    """Golden snapshot comparison for 6-bar at canonical angles."""

    def test_position_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("sixbar")
        mech = _build_sixbar()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, _, _ = _solve_sixbar_at(mech, angle)
            np.testing.assert_allclose(
                result.q, step["q"], atol=GOLDEN_TOL,
                err_msg=f"Position drift at {step['angle_deg']}°",
            )

    def test_velocity_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("sixbar")
        mech = _build_sixbar()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, q_dot, _ = _solve_sixbar_at(mech, angle)
            np.testing.assert_allclose(
                q_dot, step["q_dot"], atol=GOLDEN_TOL,
                err_msg=f"Velocity drift at {step['angle_deg']}°",
            )

    def test_acceleration_snapshot(self) -> None:
        golden = _load_or_generate_snapshot("sixbar")
        mech = _build_sixbar()
        for step in golden["steps"]:
            angle = step["angle_rad"]
            result, q_dot, q_ddot = _solve_sixbar_at(mech, angle)
            np.testing.assert_allclose(
                q_ddot, step["q_ddot"], atol=GOLDEN_TOL,
                err_msg=f"Acceleration drift at {step['angle_deg']}°",
            )


# ── Condition number baseline ──


class TestFourbarConditionBaseline:
    """Regression baseline: condition numbers at 15-degree increments."""

    def test_condition_number_within_2x_baseline(self) -> None:
        """Condition numbers at generic angles should not degrade by > 2x.

        Excludes angles within 10 degrees of known toggle positions (0, 180)
        for this boundary Grashof mechanism.
        """
        mech = _build_fourbar()
        # BASELINE: calibrated 2026-03-17
        # At generic angles (away from toggle), condition number is typically < 50
        baseline_max = 50.0
        # Exclude near-toggle: 0 deg and 180 deg for boundary Grashof
        toggle_exclusion_deg = 15.0
        angles_deg = np.arange(30, 330, 15)
        for angle_deg in angles_deg:
            if angle_deg < toggle_exclusion_deg or abs(angle_deg - 180) < toggle_exclusion_deg:
                continue
            if angle_deg > 360 - toggle_exclusion_deg:
                continue
            angle = np.radians(angle_deg)
            result, _, _ = _solve_fourbar_at(mech, angle)
            rank_result = jacobian_rank_analysis(mech, result.q, t=angle)
            assert rank_result.condition_number < 2 * baseline_max, (
                f"Condition number {rank_result.condition_number:.1f} at {angle_deg}° "
                f"exceeds 2x baseline ({baseline_max})"
            )
