"""Export golden test fixtures for the Rust port.

Generates JSON files containing reference outputs for all benchmark
mechanisms across all analysis modes (kinematics, statics, inverse
dynamics, forward dynamics).

Usage:
    python scripts/export_golden.py

Output: data/benchmarks/golden/*.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from linkage_sim.analysis.coupler import eval_coupler_point
from linkage_sim.analysis.energy import compute_energy_state
from linkage_sim.analysis.reactions import extract_reactions, get_driver_reactions
from linkage_sim.core.bodies import Body, make_bar, make_ground
from linkage_sim.core.mechanism import Mechanism
from linkage_sim.forces.gravity import Gravity
from linkage_sim.solvers.forward_dynamics import ForwardDynamicsConfig, simulate
from linkage_sim.solvers.inverse_dynamics import solve_inverse_dynamics
from linkage_sim.solvers.kinematics import (
    solve_acceleration,
    solve_position,
    solve_velocity,
)
from linkage_sim.solvers.statics import solve_statics

GOLDEN_DIR = Path(__file__).parent.parent / "data" / "benchmarks" / "golden"


def to_list(arr: np.ndarray) -> list[float]:  # type: ignore[type-arg]
    return [float(x) for x in arr]


# --- 4-bar benchmark ---


def build_fourbar_driven() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=2.0, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=3.0, Izz_cg=0.05)
    coupler.add_coupler_point("P", 1.5, 0.5)
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
    return mech


def solve_fourbar_at(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q = mech.state.make_q()
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    bx, by = np.cos(angle), np.sin(angle)
    mech.state.set_pose("coupler", q, bx, by, 0.0)
    mech.state.set_pose("rocker", q, 4.0, 0.0, np.pi / 2)
    result = solve_position(mech, q, t=angle)
    assert result.converged, f"Failed at {np.degrees(angle):.1f}°"
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


def export_fourbar_kinematics() -> None:
    print("Exporting 4-bar kinematics...")
    mech = build_fourbar_driven()
    angles = np.linspace(np.radians(30), np.radians(330), 61)

    data = {
        "mechanism": "fourbar",
        "description": "4-bar crank-rocker: ground=4, crank=1, coupler=3, rocker=2",
        "link_lengths": {"ground": 4.0, "crank": 1.0, "coupler": 3.0, "rocker": 2.0},
        "masses": {"crank": 2.0, "coupler": 3.0, "rocker": 2.0},
        "steps": [],
    }

    for angle in angles:
        q, q_dot, q_ddot = solve_fourbar_at(mech, angle)
        coupler_pos, coupler_vel, coupler_acc = eval_coupler_point(
            mech.state, "coupler", np.array([1.5, 0.5]), q, q_dot, q_ddot,
        )
        data["steps"].append({
            "input_angle_rad": float(angle),
            "input_angle_deg": float(np.degrees(angle)),
            "q": to_list(q),
            "q_dot": to_list(q_dot),
            "q_ddot": to_list(q_ddot),
            "coupler_point_P": {
                "position": to_list(coupler_pos),
                "velocity": to_list(coupler_vel),
                "acceleration": to_list(coupler_acc),
            },
        })

    write_json("fourbar_kinematics.json", data)


def export_fourbar_statics() -> None:
    print("Exporting 4-bar statics with gravity...")
    mech = build_fourbar_driven()
    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    angles = np.linspace(np.radians(30), np.radians(330), 61)

    data = {
        "mechanism": "fourbar",
        "analysis": "statics",
        "gravity": [0.0, -9.81],
        "steps": [],
    }

    for angle in angles:
        q, _, _ = solve_fourbar_at(mech, angle)
        result = solve_statics(mech, q, [gravity], t=angle)
        reactions = extract_reactions(mech, result, q)
        drivers = get_driver_reactions(reactions)

        data["steps"].append({
            "input_angle_rad": float(angle),
            "lambdas": to_list(result.lambdas),
            "Q": to_list(result.Q),
            "residual_norm": result.residual_norm,
            "driver_torque": float(drivers[0].effort),
            "joint_reactions": [
                {
                    "joint_id": r.joint_id,
                    "force_global": to_list(r.force_global),
                    "resultant": r.resultant,
                }
                for r in reactions if r.n_equations > 1
            ],
        })

    write_json("fourbar_statics.json", data)


def export_fourbar_inverse_dynamics() -> None:
    print("Exporting 4-bar inverse dynamics...")
    mech = build_fourbar_driven()
    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    angles = np.linspace(np.radians(30), np.radians(330), 61)

    data = {
        "mechanism": "fourbar",
        "analysis": "inverse_dynamics",
        "gravity": [0.0, -9.81],
        "steps": [],
    }

    for angle in angles:
        q, q_dot, q_ddot = solve_fourbar_at(mech, angle)
        result = solve_inverse_dynamics(mech, q, q_dot, q_ddot, [gravity], t=angle)

        data["steps"].append({
            "input_angle_rad": float(angle),
            "lambdas": to_list(result.lambdas),
            "Q": to_list(result.Q),
            "M_q_ddot": to_list(result.M_q_ddot),
            "residual_norm": result.residual_norm,
            "driver_torque": float(result.lambdas[-1]),
        })

    write_json("fourbar_inverse_dynamics.json", data)


def build_fourbar_undriven() -> Mechanism:
    """4-bar without driver for forward dynamics (DOF=1, free motion)."""
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(4.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=2.0, Izz_cg=0.01)
    coupler = make_bar("coupler", "B", "C", 3.0, mass=3.0, Izz_cg=0.05)
    rocker = make_bar("rocker", "D", "C", 2.0, mass=2.0, Izz_cg=0.02)
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(coupler)
    mech.add_body(rocker)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "coupler", "B")
    mech.add_revolute_joint("J3", "coupler", "C", "rocker", "C")
    mech.add_revolute_joint("J4", "ground", "O4", "rocker", "D")
    mech.build()
    return mech


def export_fourbar_dynamics() -> None:
    """Export 4-bar forward dynamics with gravity (free motion, no driver)."""
    print("Exporting 4-bar forward dynamics...")
    mech = build_fourbar_undriven()
    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)

    # Initial config: crank at 60 deg with initial angular velocity
    init_angle = np.radians(60)
    q0 = mech.state.make_q()
    mech.state.set_pose("crank", q0, 0.0, 0.0, init_angle)
    bx, by = np.cos(init_angle), np.sin(init_angle)
    mech.state.set_pose("coupler", q0, bx, by, 0.0)
    mech.state.set_pose("rocker", q0, 4.0, 0.0, np.pi / 2)
    pos_result = solve_position(mech, q0, t=0.0)
    assert pos_result.converged, "4-bar FD initial position solve failed"
    q0 = pos_result.q

    # Give the crank an initial angular velocity of 5 rad/s
    qd0 = np.zeros(mech.state.n_coords)
    crank_theta_idx = mech.state.get_index("crank").theta_idx
    qd0[crank_theta_idx] = 5.0

    t_end = 3.0
    t_eval = np.linspace(0, t_end, 150)
    config = ForwardDynamicsConfig(
        alpha=10.0, beta=10.0, rtol=1e-10, atol=1e-12, max_step=0.005,
    )
    result = simulate(mech, q0, qd0, (0.0, t_end), [gravity], config, t_eval)
    assert result.success, f"4-bar FD simulation failed: {result.message}"

    data = {
        "mechanism": "fourbar",
        "analysis": "forward_dynamics",
        "description": "4-bar crank-rocker free motion under gravity, crank0=60deg, omega0=5rad/s",
        "initial_angle_rad": float(init_angle),
        "initial_omega": 5.0,
        "gravity": [0.0, -9.81],
        "steps": [],
    }

    for i in range(len(result.t)):
        es = compute_energy_state(mech, result.q[i], result.q_dot[i])
        data["steps"].append({
            "t": float(result.t[i]),
            "q": to_list(result.q[i]),
            "q_dot": to_list(result.q_dot[i]),
            "constraint_drift": float(result.constraint_drift[i]),
            "kinetic_energy": es.kinetic,
            "potential_energy": es.potential_gravity,
            "total_energy": es.total,
        })

    write_json("fourbar_dynamics.json", data)


# --- Slider-crank benchmark ---


def build_slidercrank_driven() -> Mechanism:
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), rail=(3.0, 0.0))
    crank = make_bar("crank", "A", "B", 1.0, mass=1.0, Izz_cg=0.01)
    conrod = make_bar("conrod", "B", "C", 3.0, mass=2.0, Izz_cg=0.1)
    slider = Body(
        id="slider",
        attachment_points={"C": np.array([0.0, 0.0])},
        mass=0.5, cg_local=np.array([0.0, 0.0]), Izz_cg=0.0,
    )
    mech.add_body(ground)
    mech.add_body(crank)
    mech.add_body(conrod)
    mech.add_body(slider)
    mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
    mech.add_revolute_joint("J2", "crank", "B", "conrod", "B")
    mech.add_revolute_joint("J3", "conrod", "C", "slider", "C")
    mech.add_prismatic_joint(
        "P1", "ground", "rail", "slider", "C",
        axis_local_i=np.array([1.0, 0.0]),
    )
    mech.add_revolute_driver(
        "D1", "ground", "crank",
        f=lambda t: t, f_dot=lambda t: 1.0, f_ddot=lambda t: 0.0,
    )
    mech.build()
    return mech


def solve_slidercrank_at(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q = mech.state.make_q()
    bx, by = np.cos(angle), np.sin(angle)
    phi = np.arcsin(-by / 3.0)
    cx = bx + 3.0 * np.cos(phi)
    mech.state.set_pose("crank", q, 0.0, 0.0, angle)
    mech.state.set_pose("conrod", q, bx, by, phi)
    mech.state.set_pose("slider", q, cx, 0.0, 0.0)
    result = solve_position(mech, q, t=angle)
    assert result.converged
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


def export_slidercrank_kinematics() -> None:
    print("Exporting slider-crank kinematics...")
    mech = build_slidercrank_driven()
    angles = np.linspace(np.radians(15), np.radians(345), 67)

    data = {
        "mechanism": "slidercrank",
        "description": "Slider-crank: crank=1, conrod=3, horizontal rail",
        "steps": [],
    }

    for angle in angles:
        q, q_dot, q_ddot = solve_slidercrank_at(mech, angle)
        data["steps"].append({
            "input_angle_rad": float(angle),
            "q": to_list(q),
            "q_dot": to_list(q_dot),
            "q_ddot": to_list(q_ddot),
        })

    write_json("slidercrank_kinematics.json", data)


def export_slidercrank_statics() -> None:
    print("Exporting slider-crank statics with gravity...")
    mech = build_slidercrank_driven()
    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    angles = np.linspace(np.radians(15), np.radians(345), 67)

    data = {
        "mechanism": "slidercrank",
        "analysis": "statics",
        "gravity": [0.0, -9.81],
        "steps": [],
    }

    for angle in angles:
        q, _, _ = solve_slidercrank_at(mech, angle)
        result = solve_statics(mech, q, [gravity], t=angle)
        reactions = extract_reactions(mech, result, q)
        drivers = get_driver_reactions(reactions)

        data["steps"].append({
            "input_angle_rad": float(angle),
            "lambdas": to_list(result.lambdas),
            "Q": to_list(result.Q),
            "residual_norm": result.residual_norm,
            "driver_torque": float(drivers[0].effort),
        })

    write_json("slidercrank_statics.json", data)


def export_slidercrank_inverse_dynamics() -> None:
    print("Exporting slider-crank inverse dynamics...")
    mech = build_slidercrank_driven()
    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    angles = np.linspace(np.radians(15), np.radians(345), 67)

    data = {
        "mechanism": "slidercrank",
        "analysis": "inverse_dynamics",
        "gravity": [0.0, -9.81],
        "steps": [],
    }

    for angle in angles:
        q, q_dot, q_ddot = solve_slidercrank_at(mech, angle)
        result = solve_inverse_dynamics(mech, q, q_dot, q_ddot, [gravity], t=angle)

        data["steps"].append({
            "input_angle_rad": float(angle),
            "lambdas": to_list(result.lambdas),
            "Q": to_list(result.Q),
            "M_q_ddot": to_list(result.M_q_ddot),
            "residual_norm": result.residual_norm,
            "driver_torque": float(result.lambdas[-1]),
        })

    write_json("slidercrank_inverse_dynamics.json", data)


# --- 6-bar benchmark ---


def make_ternary_link(
    body_id: str,
    p1_name: str,
    p2_name: str,
    p3_name: str,
    p2_local: tuple[float, float],
    p3_local: tuple[float, float],
) -> Body:
    """Create a ternary body with 3 attachment points. P1 at local origin."""
    body = Body(id=body_id)
    body.add_attachment_point(p1_name, 0.0, 0.0)
    body.add_attachment_point(p2_name, p2_local[0], p2_local[1])
    body.add_attachment_point(p3_name, p3_local[0], p3_local[1])
    cg_x = (0.0 + p2_local[0] + p3_local[0]) / 3.0
    cg_y = (0.0 + p2_local[1] + p3_local[1]) / 3.0
    body.cg_local = np.array([cg_x, cg_y])
    return body


def build_sixbar_driven() -> Mechanism:
    """Build a Watt I 6-bar with ternary coupler (same as test_sixbar_ternary).

    Topology:
        Loop 1: Ground(O2)—Crank—Ternary(P1,P2)—Rocker4—Ground(O6)
        Loop 2: Ternary(P3)—Link5—Output6—Ground(O4)
    """
    mech = Mechanism()
    ground = make_ground(O2=(0.0, 0.0), O4=(3.0, 1.0), O6=(6.0, 0.0))
    crank = make_bar("crank", "A", "B", length=1.5, mass=0.5, Izz_cg=0.01)
    ternary = make_ternary_link(
        "ternary", "P1", "P2", "P3",
        p2_local=(3.0, 0.0),
        p3_local=(1.5, 1.0),
    )
    ternary.mass = 1.0
    ternary.Izz_cg = 0.05
    rocker4 = make_bar("rocker4", "R4A", "R4B", length=2.5, mass=0.5, Izz_cg=0.02)
    link5 = make_bar("link5", "L5A", "L5B", length=2.0, mass=0.5, Izz_cg=0.02)
    output6 = make_bar("output6", "R6A", "R6B", length=2.0, mass=0.5, Izz_cg=0.02)

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


def find_sixbar_initial(mech: Mechanism, crank_angle: float) -> np.ndarray:  # type: ignore[type-arg]
    """Geometric initial guess for the 6-bar NR solver."""
    q = mech.state.make_q()

    # Crank
    mech.state.set_pose("crank", q, 0.0, 0.0, crank_angle)

    # Crank tip B
    bx = 1.5 * np.cos(crank_angle)
    by = 1.5 * np.sin(crank_angle)

    # Ternary: P1 at B, initial theta ~ 0.15
    theta_tern = 0.15
    mech.state.set_pose("ternary", q, bx, by, theta_tern)

    # Ternary P2 global (approximate)
    ct, st = np.cos(theta_tern), np.sin(theta_tern)
    p2_gx = bx + 3.0 * ct
    p2_gy = by + 3.0 * st

    # Rocker4: R4A at O6=(6,0), R4B should be near P2
    dx = p2_gx - 6.0
    dy = p2_gy
    theta_r4 = np.arctan2(dy, dx)
    mech.state.set_pose("rocker4", q, 6.0, 0.0, theta_r4)

    # Ternary P3 global (approximate)
    p3_gx = bx + 1.5 * ct - 1.0 * st
    p3_gy = by + 1.5 * st + 1.0 * ct

    # Link5: L5A at P3, pointing toward O4 region
    dx5 = 3.0 - p3_gx
    dy5 = 1.0 - p3_gy
    theta_l5 = np.arctan2(dy5, dx5)
    mech.state.set_pose("link5", q, p3_gx, p3_gy, theta_l5)

    # Output6: R6A at O4=(3,1), R6B should be near link5 L5B
    l5b_gx = p3_gx + 2.0 * np.cos(theta_l5)
    l5b_gy = p3_gy + 2.0 * np.sin(theta_l5)
    dx6 = l5b_gx - 3.0
    dy6 = l5b_gy - 1.0
    theta_o6 = np.arctan2(dy6, dx6)
    mech.state.set_pose("output6", q, 3.0, 1.0, theta_o6)

    return q


def solve_sixbar_at(mech: Mechanism, angle: float) -> tuple:  # type: ignore[type-arg]
    q0 = find_sixbar_initial(mech, angle)
    result = solve_position(mech, q0, t=angle)
    assert result.converged, f"6-bar solve failed at {np.degrees(angle):.1f}°"
    q_dot = solve_velocity(mech, result.q, t=angle)
    q_ddot = solve_acceleration(mech, result.q, q_dot, t=angle)
    return result.q, q_dot, q_ddot


def export_sixbar_kinematics() -> None:
    print("Exporting 6-bar kinematics...")
    mech = build_sixbar_driven()
    angles = np.linspace(np.radians(10), np.radians(60), 25)

    data = {
        "mechanism": "sixbar_watt_i",
        "description": "Watt I 6-bar: Ground O2=(0,0) O4=(3,1) O6=(6,0), "
                       "crank=1.5, ternary P2=(3,0) P3=(1.5,1), rocker4=2.5, "
                       "link5=2.0, output6=2.0",
        "link_lengths": {
            "crank": 1.5, "rocker4": 2.5, "link5": 2.0, "output6": 2.0,
        },
        "steps": [],
    }

    for angle in angles:
        q, q_dot, q_ddot = solve_sixbar_at(mech, angle)
        data["steps"].append({
            "input_angle_rad": float(angle),
            "input_angle_deg": float(np.degrees(angle)),
            "q": to_list(q),
            "q_dot": to_list(q_dot),
            "q_ddot": to_list(q_ddot),
        })

    write_json("sixbar_kinematics.json", data)


# --- Pendulum forward dynamics ---


def export_pendulum_dynamics() -> None:
    print("Exporting pendulum forward dynamics...")
    mech = Mechanism()
    ground = make_ground(O=(0.0, 0.0))
    bar = Body(
        id="bar",
        attachment_points={"A": np.array([0.0, 0.0])},
        mass=1.0,
        cg_local=np.array([1.0, 0.0]),
        Izz_cg=0.0,
    )
    mech.add_body(ground)
    mech.add_body(bar)
    mech.add_revolute_joint("J1", "ground", "O", "bar", "A")
    mech.build()

    gravity = Gravity(g_vector=np.array([0.0, -9.81]), bodies=mech.bodies)
    theta0 = -np.pi / 2 + 0.2
    q0 = mech.state.make_q()
    mech.state.set_pose("bar", q0, 0.0, 0.0, theta0)
    qd0 = np.zeros(mech.state.n_coords)

    t_eval = np.linspace(0, 5.0, 250)
    config = ForwardDynamicsConfig(
        alpha=10.0, beta=10.0, rtol=1e-10, atol=1e-12, max_step=0.002,
    )
    result = simulate(mech, q0, qd0, (0.0, 5.0), [gravity], config, t_eval)
    assert result.success

    data = {
        "mechanism": "pendulum",
        "description": "Simple pendulum L=1, m=1, theta0=-pi/2+0.2",
        "initial_angle_rad": float(theta0),
        "gravity": [0.0, -9.81],
        "steps": [],
    }

    for i in range(len(result.t)):
        es = compute_energy_state(mech, result.q[i], result.q_dot[i])
        data["steps"].append({
            "t": float(result.t[i]),
            "q": to_list(result.q[i]),
            "q_dot": to_list(result.q_dot[i]),
            "constraint_drift": float(result.constraint_drift[i]),
            "kinetic_energy": es.kinetic,
            "potential_energy": es.potential_gravity,
            "total_energy": es.total,
        })

    write_json("pendulum_dynamics.json", data)


# --- Utility ---


def write_json(filename: str, data: dict) -> None:  # type: ignore[type-arg]
    path = GOLDEN_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path}")


def main() -> None:
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    print(f"Exporting golden fixtures to {GOLDEN_DIR}\n")

    export_fourbar_kinematics()
    export_fourbar_statics()
    export_fourbar_inverse_dynamics()
    export_fourbar_dynamics()
    export_slidercrank_kinematics()
    export_slidercrank_statics()
    export_slidercrank_inverse_dynamics()
    export_sixbar_kinematics()
    export_pendulum_dynamics()

    print(f"\nDone. {len(list(GOLDEN_DIR.glob('*.json')))} golden fixture files exported.")


if __name__ == "__main__":
    main()
