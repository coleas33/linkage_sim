"""Mechanism: the top-level assembly of bodies, joints, and state.

A Mechanism owns the bodies, joints, and state vector mapping.
It provides the interface for building mechanisms and is the input
to all solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from typing import Callable

from linkage_sim.core.bodies import Body
from linkage_sim.core.constraints import (
    Constraint,
    FixedJoint,
    PrismaticJoint,
    RevoluteJoint,
    make_fixed_joint,
    make_prismatic_joint,
    make_revolute_joint,
)
from linkage_sim.core.drivers import (
    RevoluteDriver,
    constant_speed_driver,
    make_revolute_driver,
)
from linkage_sim.core.state import GROUND_ID, State


@dataclass
class Mechanism:
    """A planar mechanism: bodies connected by joint constraints.

    Usage:
        mech = Mechanism()
        mech.add_body(ground)
        mech.add_body(crank)
        mech.add_revolute_joint("J1", "ground", "O2", "crank", "A")
        mech.build()  # finalizes state vector, ready for solvers
    """

    _bodies: dict[str, Body] = field(default_factory=dict)
    _joints: list[Constraint] = field(default_factory=list)
    _state: State = field(default_factory=State)
    _built: bool = False

    @property
    def state(self) -> State:
        return self._state

    @property
    def bodies(self) -> dict[str, Body]:
        return self._bodies

    @property
    def joints(self) -> list[Constraint]:
        return self._joints

    @property
    def n_constraints(self) -> int:
        """Total number of constraint equations across all joints."""
        return sum(j.n_equations for j in self._joints)

    def add_body(self, body: Body) -> None:
        """Add a body to the mechanism.

        Must be called before build(). Ground and moving bodies are both added
        here; ground is recognized by its ID and excluded from the state vector.
        """
        if self._built:
            raise RuntimeError("Cannot add bodies after build().")
        if body.id in self._bodies:
            raise ValueError(f"Body '{body.id}' already exists in the mechanism.")
        self._bodies[body.id] = body

    def add_revolute_joint(
        self,
        joint_id: str,
        body_i_id: str,
        point_i_name: str,
        body_j_id: str,
        point_j_name: str,
    ) -> None:
        """Add a revolute joint between two bodies at named attachment points.

        Both bodies must already be added to the mechanism.
        """
        if self._built:
            raise RuntimeError("Cannot add joints after build().")

        body_i = self._get_body(body_i_id)
        body_j = self._get_body(body_j_id)
        pt_i = body_i.get_attachment_point(point_i_name)
        pt_j = body_j.get_attachment_point(point_j_name)

        joint = make_revolute_joint(
            joint_id=joint_id,
            body_i_id=body_i_id,
            point_i_name=point_i_name,
            point_i_local=pt_i,
            body_j_id=body_j_id,
            point_j_name=point_j_name,
            point_j_local=pt_j,
        )
        self._joints.append(joint)

    def add_fixed_joint(
        self,
        joint_id: str,
        body_i_id: str,
        point_i_name: str,
        body_j_id: str,
        point_j_name: str,
        delta_theta_0: float = 0.0,
    ) -> None:
        """Add a fixed joint that locks all relative motion."""
        if self._built:
            raise RuntimeError("Cannot add joints after build().")

        body_i = self._get_body(body_i_id)
        body_j = self._get_body(body_j_id)
        pt_i = body_i.get_attachment_point(point_i_name)
        pt_j = body_j.get_attachment_point(point_j_name)

        joint = make_fixed_joint(
            joint_id=joint_id,
            body_i_id=body_i_id,
            point_i_name=point_i_name,
            point_i_local=pt_i,
            body_j_id=body_j_id,
            point_j_name=point_j_name,
            point_j_local=pt_j,
            delta_theta_0=delta_theta_0,
        )
        self._joints.append(joint)

    def add_prismatic_joint(
        self,
        joint_id: str,
        body_i_id: str,
        point_i_name: str,
        body_j_id: str,
        point_j_name: str,
        axis_local_i: NDArray[np.float64],
        delta_theta_0: float = 0.0,
    ) -> None:
        """Add a prismatic joint that allows sliding along one axis.

        Body i owns the slide axis. Body j slides along it.
        Both bodies must already be added to the mechanism.

        Args:
            joint_id: Unique identifier for this joint.
            body_i_id: ID of the body that owns the slide axis.
            point_i_name: Name of the attachment point on body_i.
            body_j_id: ID of the sliding body.
            point_j_name: Name of the attachment point on body_j.
            axis_local_i: Unit vector along the slide axis in body_i's local frame.
            delta_theta_0: Initial relative angle θⱼ - θᵢ to maintain.
        """
        if self._built:
            raise RuntimeError("Cannot add joints after build().")

        body_i = self._get_body(body_i_id)
        body_j = self._get_body(body_j_id)
        pt_i = body_i.get_attachment_point(point_i_name)
        pt_j = body_j.get_attachment_point(point_j_name)

        joint = make_prismatic_joint(
            joint_id=joint_id,
            body_i_id=body_i_id,
            point_i_name=point_i_name,
            point_i_local=pt_i,
            body_j_id=body_j_id,
            point_j_name=point_j_name,
            point_j_local=pt_j,
            axis_local_i=axis_local_i,
            delta_theta_0=delta_theta_0,
        )
        self._joints.append(joint)

    def add_revolute_driver(
        self,
        driver_id: str,
        body_i_id: str,
        body_j_id: str,
        f: Callable[[float], float],
        f_dot: Callable[[float], float],
        f_ddot: Callable[[float], float],
    ) -> None:
        """Add a revolute driver prescribing relative angle vs. time.

        Adds one constraint equation: θⱼ - θᵢ - f(t) = 0.
        Both bodies must already be added to the mechanism.
        """
        if self._built:
            raise RuntimeError("Cannot add drivers after build().")

        self._get_body(body_i_id)
        self._get_body(body_j_id)

        driver = make_revolute_driver(
            driver_id=driver_id,
            body_i_id=body_i_id,
            body_j_id=body_j_id,
            f=f,
            f_dot=f_dot,
            f_ddot=f_ddot,
        )
        self._joints.append(driver)

    def add_constant_speed_driver(
        self,
        driver_id: str,
        body_i_id: str,
        body_j_id: str,
        omega: float,
        theta_0: float = 0.0,
    ) -> None:
        """Add a constant-speed revolute driver.

        f(t) = theta_0 + omega * t, f'(t) = omega, f''(t) = 0.
        """
        if self._built:
            raise RuntimeError("Cannot add drivers after build().")

        self._get_body(body_i_id)
        self._get_body(body_j_id)

        driver = constant_speed_driver(
            driver_id=driver_id,
            body_i_id=body_i_id,
            body_j_id=body_j_id,
            omega=omega,
            theta_0=theta_0,
        )
        self._joints.append(driver)

    def build(self) -> None:
        """Finalize the mechanism: register moving bodies in the state vector.

        After build(), no more bodies or joints can be added.
        The state vector size is fixed and solvers can operate.
        """
        if self._built:
            raise RuntimeError("Mechanism already built.")

        # Register all non-ground bodies in the state vector
        for body_id in sorted(self._bodies.keys()):
            if body_id != GROUND_ID:
                self._state.register_body(body_id)

        self._built = True

    def _get_body(self, body_id: str) -> Body:
        if body_id not in self._bodies:
            raise KeyError(
                f"Body '{body_id}' not found. "
                f"Available: {list(self._bodies.keys())}"
            )
        return self._bodies[body_id]
