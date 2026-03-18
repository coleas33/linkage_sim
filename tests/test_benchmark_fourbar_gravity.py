"""Benchmark: 4-bar linkage with gravity — Phase 2 integration test.

Validates the complete static force analysis pipeline:
position solve → static solve → reaction extraction → virtual work cross-check
→ envelope computation across a full crank sweep with gravity loading.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.envelopes import compute_envelope
from linkage_sim.analysis.grashof import GrashofType, check_grashof
from linkage_sim.analysis.reactions import extract_reactions, get_driver_reactions
from linkage_sim.analysis.toggle import detect_toggle
from linkage_sim.analysis.transmission import transmission_angle_fourbar
from linkage_sim.analysis.virtual_work import virtual_work_input_torque
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.solvers.kinematics import solve_position
from linkage_sim.solvers.statics import solve_statics


# --- Mechanism: crank=1, coupler=3, rocker=2, ground=4, all 2 kg ---


def build_gravity_fourbar() -> tuple[Mechanism, Gravity]:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=2.0, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=2.0, Izz_cg=0.05)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=2.0, Izz_cg=0.02)

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
    return mech, gravity


def solve_at(mech: Mechanism, angle: float) -> np.ndarray:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    return result.q


class TestGrashofClassification:
    def test_benchmark_is_change_point(self) -> None:
        result = check_grashof(4.0, 1.0, 3.0, 2.0)
        assert result.is_change_point or result.is_grashof


class TestStaticTorqueSweep:
    """Full sweep: static torque, reactions, virtual work."""

    def test_sweep_all_steps_converge(self) -> None:
        mech, gravity = build_gravity_fourbar()
        angles = np.linspace(np.radians(30), np.radians(150), 25)
        for angle in angles:
            q = solve_at(mech, angle)
            result = solve_statics(mech, q, [gravity], t=angle)
            assert result.residual_norm < 1e-10

    def test_driver_torque_nonzero_with_gravity(self) -> None:
        mech, gravity = build_gravity_fourbar()
        q = solve_at(mech, np.pi / 4)
        result = solve_statics(mech, q, [gravity], t=np.pi / 4)
        reactions = extract_reactions(mech, result, q)
        drivers = get_driver_reactions(reactions)
        assert abs(drivers[0].effort) > 0.1

    def test_virtual_work_matches_statics_across_sweep(self) -> None:
        """Core cross-check: virtual work ≈ Lagrange multiplier at every angle."""
        mech, gravity = build_gravity_fourbar()
        angles = np.linspace(np.radians(30), np.radians(150), 13)
        for angle in angles:
            q = solve_at(mech, angle)
            static_result = solve_statics(mech, q, [gravity], t=angle)
            reactions = extract_reactions(mech, static_result, q)
            lambda_torque = get_driver_reactions(reactions)[0].effort

            vw = virtual_work_input_torque(mech, q, [gravity], "crank", t=angle)
            assert vw.input_torque == pytest.approx(lambda_torque, rel=1e-6), (
                f"Mismatch at {np.degrees(angle):.0f}°"
            )

    def test_reactions_at_ground_joints(self) -> None:
        """Ground joint reactions are nonzero with gravity."""
        mech, gravity = build_gravity_fourbar()
        q = solve_at(mech, np.pi / 3)
        result = solve_statics(mech, q, [gravity], t=np.pi / 3)
        reactions = extract_reactions(mech, result, q)

        j1 = reactions[0]  # ground-crank
        j4 = reactions[3]  # ground-rocker
        assert j1.resultant > 1.0
        assert j4.resultant > 1.0


class TestEnvelopes:
    """Result envelope computation across sweep."""

    def test_torque_envelope(self) -> None:
        mech, gravity = build_gravity_fourbar()
        angles = np.linspace(np.radians(30), np.radians(150), 25)
        torques = []
        for angle in angles:
            q = solve_at(mech, angle)
            result = solve_statics(mech, q, [gravity], t=angle)
            reactions = extract_reactions(mech, result, q)
            torques.append(get_driver_reactions(reactions)[0].effort)

        env = compute_envelope(np.array(torques), angles)
        assert env.peak > 0
        assert env.rms > 0
        assert env.range > 0
        assert env.minimum <= env.maximum

    def test_reaction_envelope(self) -> None:
        mech, gravity = build_gravity_fourbar()
        angles = np.linspace(np.radians(30), np.radians(150), 13)
        j1_forces = []
        for angle in angles:
            q = solve_at(mech, angle)
            result = solve_statics(mech, q, [gravity], t=angle)
            reactions = extract_reactions(mech, result, q)
            j1_forces.append(reactions[0].resultant)

        env = compute_envelope(np.array(j1_forces), angles)
        assert env.peak > 0


class TestTransmissionAngleSweep:
    def test_transmission_angle_within_bounds(self) -> None:
        angles = np.linspace(np.radians(30), np.radians(150), 25)
        for theta in angles:
            result = transmission_angle_fourbar(1.0, 3.0, 2.0, 4.0, theta)
            assert 0 <= result.angle_deg <= 180

    def test_no_toggle_in_sweep_range(self) -> None:
        mech, _ = build_gravity_fourbar()
        angles = np.linspace(np.radians(30), np.radians(150), 25)
        for angle in angles:
            q = solve_at(mech, angle)
            result = detect_toggle(mech, q, t=angle)
            assert not result.is_near_toggle
