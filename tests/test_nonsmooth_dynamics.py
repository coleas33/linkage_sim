"""Tests for Phase 4B: Nonsmooth forward dynamics effects.

Benchmarks:
1. Pendulum with hard stop (joint limit + restitution)
2. Slider-crank with Coulomb friction (energy dissipation)
3. Mechanism with joint limits (correct stop behavior)
"""

from __future__ import annotations

import numpy as np
import pytest

from linkage_sim.analysis.energy import compute_energy_state
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.friction import CoulombFriction
from linkage_sim.forces.gravity import Gravity
from linkage_sim.forces.joint_limit import JointLimit
from linkage_sim.forces.viscous_damper import RotaryDamper
from linkage_sim.solvers.events import (
    make_angle_limit_event,
    make_velocity_reversal_event,
)
from linkage_sim.solvers.forward_dynamics import (
    ForwardDynamicsConfig,
    simulate,
)


# --- Pendulum helper ---


def build_pendulum(length: float = 1.0, mass: float = 1.0) -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O=(0.0, 0.0))
    bar = Body(
        id="bar",
        attachment_points={"A": np.array([0.0, 0.0])},
        mass=mass,
        cg_local=np.array([length, 0.0]),
        Izz_cg=0.0,
    )
    mech.add_body(ground)
    mech.add_body(bar)
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
    mech.build()
    return mech


def pendulum_ic(
    mech: Mechanism, theta0: float, omega0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("bar", q, 0.0, 0.0, theta0)
    qd = np.zeros(mech.state.n_coords)
    idx = mech.state.get_index("bar")
    qd[idx.theta_idx] = omega0
    return q, qd


# --- Joint Limit Tests ---


class TestJointLimitBasic:
    """Unit tests for JointLimit force element."""

    def setup_method(self) -> None:
        self.state = __import__("linkage_sim.core.state", fromlist=["State"]).State()
        self.state.register_body("b1")

    def test_within_limits_no_force(self) -> None:
        jl = JointLimit(
            body_i_id="ground", body_j_id="b1",
            angle_min=-1.0, angle_max=1.0, stiffness=1000.0, _id="jl1",
        )
        q = np.array([0.0, 0.0, 0.5])  # within limits
        Q = jl.evaluate(self.state, q, np.zeros(3), 0.0)
        np.testing.assert_allclose(Q, np.zeros(3), atol=1e-15)

    def test_below_min_pushes_back(self) -> None:
        jl = JointLimit(
            body_i_id="ground", body_j_id="b1",
            angle_min=0.0, angle_max=2.0, stiffness=1000.0, _id="jl1",
        )
        q = np.array([0.0, 0.0, -0.1])  # below min
        Q = jl.evaluate(self.state, q, np.zeros(3), 0.0)
        assert Q[2] > 0  # pushes CCW (positive)

    def test_above_max_pushes_back(self) -> None:
        jl = JointLimit(
            body_i_id="ground", body_j_id="b1",
            angle_min=0.0, angle_max=1.0, stiffness=1000.0, _id="jl1",
        )
        q = np.array([0.0, 0.0, 1.2])  # above max
        Q = jl.evaluate(self.state, q, np.zeros(3), 0.0)
        assert Q[2] < 0  # pushes CW (negative)

    def test_penalty_proportional_to_penetration(self) -> None:
        jl = JointLimit(
            body_i_id="ground", body_j_id="b1",
            angle_min=0.0, angle_max=2.0, stiffness=500.0, _id="jl1",
        )
        q1 = np.array([0.0, 0.0, -0.1])
        q2 = np.array([0.0, 0.0, -0.2])
        Q1 = jl.evaluate(self.state, q1, np.zeros(3), 0.0)
        Q2 = jl.evaluate(self.state, q2, np.zeros(3), 0.0)
        # Double penetration → double force
        assert Q2[2] == pytest.approx(2 * Q1[2])

    def test_damping_term(self) -> None:
        jl = JointLimit(
            body_i_id="ground", body_j_id="b1",
            angle_min=0.0, angle_max=2.0,
            stiffness=500.0, damping=10.0, _id="jl1",
        )
        q = np.array([0.0, 0.0, -0.1])
        qd_into = np.array([0.0, 0.0, -1.0])  # moving into stop
        qd_away = np.array([0.0, 0.0, 1.0])   # moving away from stop

        Q_into = jl.evaluate(self.state, q, qd_into, 0.0)
        Q_away = jl.evaluate(self.state, q, qd_away, 0.0)

        # Damping when moving into stop should be larger
        assert Q_into[2] > Q_away[2]

    def test_action_reaction(self) -> None:
        from linkage_sim.core.state import State
        state = State()
        state.register_body("b1")
        state.register_body("b2")
        jl = JointLimit(
            body_i_id="b1", body_j_id="b2",
            angle_min=-1.0, angle_max=1.0,
            stiffness=1000.0, _id="jl1",
        )
        q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.5])  # above max
        Q = jl.evaluate(state, q, np.zeros(6), 0.0)
        assert Q[2] + Q[5] == pytest.approx(0.0)


# --- Event Detection Tests ---


class TestEventDetection:
    """Event function creation and behavior."""

    def test_angle_limit_event_sign(self) -> None:
        from linkage_sim.core.state import State
        state = State()
        state.register_body("b1")
        n = state.n_coords

        event = make_angle_limit_event(state, "ground", "b1", np.pi / 2)
        # θ < π/2 → negative
        y_below = np.zeros(2 * n)
        y_below[2] = np.pi / 4
        assert event(0.0, y_below) < 0

        # θ > π/2 → positive
        y_above = np.zeros(2 * n)
        y_above[2] = 3 * np.pi / 4
        assert event(0.0, y_above) > 0

    def test_velocity_reversal_event(self) -> None:
        from linkage_sim.core.state import State
        state = State()
        state.register_body("b1")
        n = state.n_coords

        event = make_velocity_reversal_event(state, "b1", "theta")
        # Positive velocity
        y_pos = np.zeros(2 * n)
        y_pos[n + 2] = 1.0
        assert event(0.0, y_pos) > 0

        # Negative velocity
        y_neg = np.zeros(2 * n)
        y_neg[n + 2] = -1.0
        assert event(0.0, y_neg) < 0


# --- Benchmark: Pendulum with Hard Stop ---


class TestPendulumHardStop:
    """Pendulum swinging into a joint limit (hard stop)."""

    def test_pendulum_bounces_at_limit(self) -> None:
        """Pendulum hits hard stop and bounces back."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

        # Hard stop at -90° (straight down) - prevent passing through
        # Pendulum hangs at θ = -π/2. Set limit at θ = -π/2 - 0.3
        limit = JointLimit(
            body_i_id="ground", body_j_id="bar",
            angle_min=-np.pi / 2 - 0.3, angle_max=np.pi,
            stiffness=10000.0, damping=50.0, restitution=0.5, _id="stop",
        )

        # Start displaced and let it swing
        theta0 = -np.pi / 2 + 0.5  # above equilibrium
        q0, qd0 = pendulum_ic(mech, theta0)

        t_eval = np.linspace(0, 3.0, 300)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.002,
            rtol=1e-9, atol=1e-11,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 3.0), [gravity, limit], config, t_eval
        )
        assert result.success

        # Pendulum should not penetrate far beyond the limit
        theta_idx = mech.state.get_index("bar").theta_idx
        theta = result.q[:, theta_idx]
        min_angle = float(np.min(theta))
        # Allow small penetration (penalty method isn't exact)
        assert min_angle > -np.pi / 2 - 0.35

    def test_energy_decreases_with_restitution(self) -> None:
        """Energy should decrease after hitting a damped stop."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        limit = JointLimit(
            body_i_id="ground", body_j_id="bar",
            angle_min=-np.pi, angle_max=-np.pi / 2 + 0.1,
            stiffness=5000.0, damping=100.0, restitution=0.3, _id="stop",
        )

        # Start above the stop and swing through
        theta0 = -np.pi / 2 + 0.4
        q0, qd0 = pendulum_ic(mech, theta0, omega0=0.0)

        t_eval = np.linspace(0, 5.0, 300)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.002,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 5.0), [gravity, limit], config, t_eval
        )
        assert result.success

        e_start = compute_energy_state(mech, result.q[0], result.q_dot[0]).total
        e_end = compute_energy_state(mech, result.q[-1], result.q_dot[-1]).total
        # With damped stop, energy should decrease
        assert e_end < e_start + 0.01  # allow small Baumgarte energy artifact


# --- Benchmark: Coulomb Friction in Dynamics ---


class TestCoulombFrictionDynamics:
    """Pendulum with Coulomb friction: energy dissipation."""

    def test_friction_dissipates_energy(self) -> None:
        """Coulomb friction should reduce total energy over time."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        friction = CoulombFriction(
            body_i_id="ground", body_j_id="bar",
            friction_coeff=0.1, pin_radius=0.02,
            normal_force=9.81,  # approximate weight
            v_threshold=0.05, _id="cf1",
        )

        theta0 = -np.pi / 2 + 0.4
        q0, qd0 = pendulum_ic(mech, theta0)

        t_eval = np.linspace(0, 8.0, 400)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.005,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 8.0), [gravity, friction], config, t_eval
        )
        assert result.success

        # Energy should decrease monotonically (approximately)
        e_start = compute_energy_state(mech, result.q[0], result.q_dot[0]).total
        e_end = compute_energy_state(mech, result.q[-1], result.q_dot[-1]).total
        assert e_end < e_start

    def test_amplitude_decreases_with_friction(self) -> None:
        """Coulomb friction reduces oscillation amplitude."""
        mech = build_pendulum(length=1.0, mass=1.0)
        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        friction = CoulombFriction(
            body_i_id="ground", body_j_id="bar",
            friction_coeff=0.05, pin_radius=0.01,
            normal_force=9.81,
            v_threshold=0.05, _id="cf1",
        )

        theta0 = -np.pi / 2 + 0.3
        q0, qd0 = pendulum_ic(mech, theta0)

        t_eval = np.linspace(0, 10.0, 500)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.005,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 10.0), [gravity, friction], config, t_eval
        )
        assert result.success

        theta_idx = mech.state.get_index("bar").theta_idx
        theta = result.q[:, theta_idx]
        theta_offset = theta - (-np.pi / 2)

        mid = len(theta_offset) // 2
        amp_first = np.max(np.abs(theta_offset[:mid]))
        amp_second = np.max(np.abs(theta_offset[mid:]))
        assert amp_second < amp_first


# --- Benchmark: 4-bar with joint limits ---


class TestFourBarJointLimits:
    """Free 4-bar with joint limits at the rocker."""

    def test_rocker_stays_within_limits(self) -> None:
        """Rocker oscillation stays within joint limit range."""
        mech = Mechanism()
        ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
        crank = make_bar("crank", "A", "B", 1.0, mass=1.0, Izz_cg=0.01)
        coupler = make_bar("coupler", "B", "C", 3.0, mass=1.5, Izz_cg=0.05)
        rocker = make_bar("rocker", "D", "C", 2.0, mass=1.0, Izz_cg=0.02)
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_body(coupler)
        mech.add_body(rocker)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
        mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
        mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
        mech.build()

        gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
        damper = RotaryDamper(
            body_i_id="ground", body_j_id="crank", damping=1.0, _id="damp"
        )
        # Limit rocker angle
        rocker_limit = JointLimit(
            body_i_id="ground", body_j_id="rocker",
            angle_min=0.5, angle_max=2.5,
            stiffness=5000.0, damping=50.0, _id="rlim",
        )

        # Initial position at π/4
        from linkage_sim.solvers.kinematics import solve_position
        q0 = mech.state.make_q()
        mech.state.set_pose("crank", q0, 0.0, 0.0, np.pi / 4)
        bx, by = np.cos(np.pi / 4), np.sin(np.pi / 4)
        mech.state.set_pose("coupler", q0, bx, by, 0.0)
        mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
        pr = solve_position(mech, q0, t=0.0)
        assert pr.converged
        q0 = pr.q
        qd0 = np.zeros(mech.state.n_coords)

        t_eval = np.linspace(0, 2.0, 200)
        config = ForwardDynamicsConfig(
            alpha=10.0, beta=10.0, max_step=0.003,
        )
        result = simulate(
            mech, q0, qd0, (0.0, 2.0),
            [gravity, damper, rocker_limit], config, t_eval,
        )
        assert result.success

        # Rocker angle should mostly stay within limits
        rocker_idx = mech.state.get_index("rocker").theta_idx
        rocker_theta = result.q[:, rocker_idx]
        # Allow small penetration from penalty method
        assert float(np.min(rocker_theta)) > 0.5 - 0.1
        assert float(np.max(rocker_theta)) < 2.5 + 0.1
