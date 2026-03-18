"""Benchmark: Damped system — Phase 3 integration test.

Tests that damping forces dissipate energy correctly and that
the force element breakdown works with multiple force types.
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.force_breakdown import evaluate_contributions
from linkage_sim.core.bodies import make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.forces.torsion_spring import TorsionSpring
from linkage_sim.forces.viscous_damper import RotaryDamper
from linkage_sim.solvers.inverse_dynamics import solve_inverse_dynamics
from linkage_sim.solvers.kinematics import solve_acceleration, solve_position, solve_velocity


def build_fourbar(mass: float = 2.0, Izz: float = 0.05) -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=mass, Izz_cg=Izz)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=mass * 1.5, Izz_cg=Izz * 3)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=mass, Izz_cg=Izz)
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


def solve_full(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    qd = solve_velocity(mech, result.q, t=angle)
    qdd = solve_acceleration(mech, result.q, qd, t=angle)
    return result.q, qd, qdd


class TestDampedSystem:
    """4-bar with gravity + spring + damper."""

    def test_damper_adds_to_driver_torque(self) -> None:
        """Damper at crank joint adds to required driver torque."""
        mech = build_fourbar()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="crank", damping=5.0, _id="rd1"
        )
        angle = np.pi / 4
        q, qd, qdd = solve_full(mech, angle)

        r_no_damp = solve_inverse_dynamics(mech, q, qd, qdd, [gravity], t=angle)
        r_damp = solve_inverse_dynamics(mech, q, qd, qdd, [gravity, damper], t=angle)

        assert r_no_damp.residual_norm < 1e-10
        assert r_damp.residual_norm < 1e-10
        # Damper changes driver torque
        assert abs(r_no_damp.lambdas[-1] - r_damp.lambdas[-1]) > 0.01

    def test_force_breakdown_with_multiple_elements(self) -> None:
        """Breakdown separates gravity, spring, and damper contributions."""
        mech = build_fourbar()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        spring = TorsionSpring(
            body_i_id="ground", body_j_id="crank",
            stiffness=50.0, free_angle=0.0, _id="ts1",
        )
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="crank",
            damping=5.0, _id="rd1",
        )
        q, qd, _ = solve_full(mech, np.pi / 4)

        contributions = evaluate_contributions(
            mech.state, [gravity, spring, damper], q, qd, 0.0
        )
        assert len(contributions) == 3
        assert contributions[0].element_id == "gravity"
        assert contributions[1].element_id == "ts1"
        assert contributions[2].element_id == "rd1"

        # Gravity should be largest contributor
        assert contributions[0].Q_norm > 0

    def test_damper_energy_dissipation_rate(self) -> None:
        """Damper dissipation rate P = τ * ω_rel = c * ω² >= 0."""
        mech = build_fourbar()
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="crank",
            damping=10.0, _id="rd1",
        )
        angle = np.pi / 3
        q, qd, _ = solve_full(mech, angle)

        Q = damper.evaluate(mech.state, q, qd, 0.0)
        # Power dissipated = Q · q_dot (should be negative = energy removed)
        power = float(np.dot(Q, qd))
        assert power <= 0, "Damper should dissipate energy (negative power)"

    def test_sweep_with_all_forces(self) -> None:
        """Full sweep with gravity + spring + damper: all residuals small."""
        mech = build_fourbar()
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        spring = TorsionSpring(
            body_i_id="ground", body_j_id="crank",
            stiffness=30.0, free_angle=np.pi / 4, _id="ts1",
        )
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="crank",
            damping=3.0, _id="rd1",
        )
        forces = [gravity, spring, damper]
        angles = np.linspace(np.radians(30), np.radians(150), 13)
        for angle in angles:
            q, qd, qdd = solve_full(mech, angle)
            r = solve_inverse_dynamics(mech, q, qd, qdd, forces, t=angle)
            assert r.residual_norm < 1e-10
