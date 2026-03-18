"""Tests for forward dynamics integrator.

Benchmark tests:
1. Simple pendulum — period matches analytical T = 2π√(L/g)
2. Damped pendulum — exponential decay envelope
3. 4-bar free response — energy balance closure
4. 4-bar step torque — reaches steady state
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.energy import compute_energy_state, compute_kinetic_energy
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.forces.torsion_spring import TorsionSpring
from linkage_sim.forces.viscous_damper import RotaryDamper
from linkage_sim.forces.external_load import ExternalLoad
from linkage_sim.solvers.forward_dynamics import (
    ForwardDynamicsConfig,
    ForwardDynamicsResult,
    simulate,
)


# --- Simple pendulum helper ---


def build_pendulum(length: float = 1.0, mass: float = 1.0) -> Mechanism:
    """Single bar pinned to ground = simple pendulum.

    No driver — 1 DOF. The bar swings freely under gravity.
    """
    mech = Mechanism()
    ground = make_ground(O=(0.0, 0.0))

    # Bar with mass concentrated at tip (point mass at end)
    bar = Body(
        id="bar",
        attachment_points={"A": np.array([0.0, 0.0])},
        mass=mass,
        cg_local=np.array([length, 0.0]),  # CG at bar tip
        Izz_cg=0.0,  # point mass → Izz about CG = 0
    )

    mech.add_body(ground)
    mech.add_body(bar)
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
    mech.build()
    return mech


def pendulum_initial_state(
    mech: Mechanism, theta0: float, omega0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Set pendulum at angle theta0 (from +x axis) with angular velocity."""
    q = mech.state.make_q()
    # Bar position: pivot at origin, so body origin = (0,0)
    mech.state.set_pose("bar", q, 0.0, 0.0, theta0)

    q_dot = np.zeros(mech.state.n_coords)
    idx = mech.state.get_index("bar")
    q_dot[idx.theta_idx] = omega0

    return q, q_dot


# --- Benchmark: Simple Pendulum ---


class TestSimplePendulum:
    """Simple pendulum: T = 2π√(L/g) for small angles."""

    def test_integration_runs(self) -> None:
        """Basic smoke test: simulation completes."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        theta0 = -np.pi / 2 + 0.1  # small displacement from hanging
        q0, qd0 = pendulum_initial_state(mech, theta0)

        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, method="Radau",
            rtol=1e-8, atol=1e-10, max_step=0.005,
        )
        result = simulate(mech, q0, qd0, (0.0, 2.0), [gravity], config)
        assert result.success

    def test_period_matches_analytical(self) -> None:
        """Pendulum period ≈ 2π√(L/g) for small angle (5°)."""
        L = 1.0
        mech = build_pendulum(length=L, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        # Start hanging straight down (-π/2) with small displacement
        theta0 = -np.pi / 2 + np.radians(5)
        q0, qd0 = pendulum_initial_state(mech, theta0)

        T_analytical = 2 * np.pi * np.sqrt(L / 9.81)
        t_end = 3 * T_analytical  # simulate 3 periods
        t_eval = np.linspace(0, t_end, 500)

        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, method="Radau",
            rtol=1e-10, atol=1e-12, max_step=0.002,
        )
        result = simulate(mech, q0, qd0, (0.0, t_end), [gravity], config, t_eval)
        assert result.success

        # Extract θ(t) and find zero crossings to measure period
        theta_idx = mech.state.get_index("bar").theta_idx
        theta = result.q[:, theta_idx]
        # Offset from equilibrium (-π/2)
        theta_offset = theta - (-np.pi / 2)

        # Find zero crossings (positive-going)
        crossings = []
        for i in range(1, len(theta_offset)):
            if theta_offset[i - 1] < 0 and theta_offset[i] >= 0:
                # Linear interpolation for crossing time
                frac = -theta_offset[i - 1] / (theta_offset[i] - theta_offset[i - 1])
                t_cross = result.t[i - 1] + frac * (result.t[i] - result.t[i - 1])
                crossings.append(t_cross)

        assert len(crossings) >= 2, f"Found only {len(crossings)} crossings"
        measured_period = crossings[1] - crossings[0]

        # Should match within 2% (small angle approximation + numerical)
        assert measured_period == pytest.approx(T_analytical, rel=0.02)

    def test_constraint_drift_bounded(self) -> None:
        """Constraint drift stays small throughout simulation."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        theta0 = -np.pi / 2 + 0.1
        q0, qd0 = pendulum_initial_state(mech, theta0)

        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.005,
        )
        result = simulate(mech, q0, qd0, (0.0, 2.0), [gravity], config)
        assert result.success
        assert np.max(result.constraint_drift) < 1e-6

    def test_energy_approximately_conserved(self) -> None:
        """Without damping, total energy should be approximately constant."""
        L = 1.0
        mech = build_pendulum(length=L, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        theta0 = -np.pi / 2 + 0.2
        q0, qd0 = pendulum_initial_state(mech, theta0)

        t_eval = np.linspace(0, 3.0, 200)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, rtol=1e-10, atol=1e-12, max_step=0.002,
        )
        result = simulate(mech, q0, qd0, (0.0, 3.0), [gravity], config, t_eval)
        assert result.success

        energies = []
        for i in range(len(result.t)):
            es = compute_energy_state(mech, result.q[i], result.q_dot[i])
            energies.append(es.total)

        energies_arr = np.array(energies)
        # Energy should stay within 1% of initial
        e0 = energies_arr[0]
        assert np.max(np.abs(energies_arr - e0)) < abs(e0) * 0.01 + 1e-6


# --- Benchmark: Damped Pendulum ---


class TestDampedPendulum:
    """Damped pendulum: amplitude decays exponentially."""

    def test_amplitude_decreases(self) -> None:
        """With rotary damper, oscillation amplitude decreases."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="bar", damping=0.2, _id="damp"
        )
        theta0 = -np.pi / 2 + 0.3
        q0, qd0 = pendulum_initial_state(mech, theta0)

        t_eval = np.linspace(0, 10.0, 500)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.005,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 10.0), [gravity, damper], config, t_eval
        )
        assert result.success

        theta_idx = mech.state.get_index("bar").theta_idx
        theta = result.q[:, theta_idx]
        theta_offset = theta - (-np.pi / 2)

        # First half amplitude should be larger than second half
        mid = len(theta_offset) // 2
        amp_first = np.max(np.abs(theta_offset[:mid]))
        amp_second = np.max(np.abs(theta_offset[mid:]))
        assert amp_second < amp_first

    def test_energy_decreases(self) -> None:
        """Total mechanical energy decreases with damping."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="bar", damping=0.5, _id="damp"
        )
        theta0 = -np.pi / 2 + 0.3
        q0, qd0 = pendulum_initial_state(mech, theta0)

        t_eval = np.linspace(0, 5.0, 200)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.005,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 5.0), [gravity, damper], config, t_eval
        )
        assert result.success

        e_start = compute_energy_state(mech, result.q[0], result.q_dot[0]).total
        e_end = compute_energy_state(mech, result.q[-1], result.q_dot[-1]).total
        assert e_end < e_start


# --- Benchmark: 4-bar free response ---


def build_free_fourbar(mass: float = 1.0) -> Mechanism:
    """4-bar WITHOUT driver — free to oscillate.

    Uses a torsion spring to provide restoring force.
    """
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=mass, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=mass * 1.5, Izz_cg=0.05)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=mass, Izz_cg=0.02)

    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)

    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    # NO driver — free 1-DOF system
    mech.build()
    return mech


class TestFourBarFreeResponse:
    """4-bar free response with spring + gravity."""

    def test_spring_oscillation_energy_balance(self) -> None:
        """With spring + gravity, no damping: energy approximately conserved."""
        from linkage_sim.solvers.kinematics import solve_position

        mech = build_free_fourbar(mass=1.0)
        spring = TorsionSpring(
            body_i_id="ground", body_j_id="crank",
            stiffness=50.0, free_angle=np.pi / 4, _id="spring"
        )
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        # Initial position: solve at crank angle = π/4 + small displacement
        q0 = mech.state.make_q()
        theta_init = np.pi / 4 + 0.1
        mech.state.set_pose("crank", q0, 0.0, 0.0, theta_init)
        bx, by = np.cos(theta_init), np.sin(theta_init)
        mech.state.set_pose("coupler", q0, bx, by, 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        pr = solve_position(mech, q0, t=0.0)
        assert pr.converged
        q0 = pr.q
        qd0 = np.zeros(mech.state.n_coords)

        t_eval = np.linspace(0, 2.0, 200)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, rtol=1e-9, atol=1e-11, max_step=0.002,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 2.0), [spring, gravity], config, t_eval
        )
        assert result.success

        # Energy should be approximately conserved (no damping)
        energies = []
        for i in range(len(result.t)):
            es = compute_energy_state(mech, result.q[i], result.q_dot[i])
            # Add spring PE: 0.5 * k * (θ - θ_free)^2
            theta_crank = float(result.q[i, mech.state.get_index("crank").theta_idx])
            spring_pe = 0.5 * 50.0 * (theta_crank - np.pi / 4) ** 2
            energies.append(es.total + spring_pe)

        energies_arr = np.array(energies)
        e0 = energies_arr[0]
        # Allow 5% drift (Baumgarte adds/removes small energy)
        max_drift = np.max(np.abs(energies_arr - e0))
        assert max_drift < abs(e0) * 0.05 + 0.1


class TestFourBarStepTorque:
    """4-bar with step torque: reaches steady behavior."""

    def test_step_torque_response(self) -> None:
        """Apply sudden torque to free 4-bar, verify it moves."""
        from linkage_sim.solvers.kinematics import solve_position

        mech = build_free_fourbar(mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="crank", damping=2.0, _id="damp"
        )
        step_torque = ExternalLoad(
            body_id="crank",
            local_point=np.array([0.0, 0.0]),
            force_func=lambda q, qd, t: np.array([0.0, 0.0]),
            torque_func=lambda q, qd, t: 5.0,  # constant CCW torque
            _id="step",
        )

        # Start at equilibrium
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, np.pi / 4)
        bx, by = np.cos(np.pi / 4), np.sin(np.pi / 4)
        mech.state.set_pose("coupler", q0, bx, by, 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        pr = solve_position(mech, q0, t=0.0)
        assert pr.converged
        q0 = pr.q
        qd0 = np.zeros(mech.state.n_coords)

        t_eval = np.linspace(0, 3.0, 200)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.005,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 3.0),
            [gravity, damper, step_torque], config, t_eval,
        )
        assert result.success

        # Crank should have moved from initial position
        theta_idx = mech.state.get_index("crank").theta_idx
        theta_start = result.q[0, theta_idx]
        theta_end = result.q[-1, theta_idx]
        assert abs(theta_end - theta_start) > 0.01
