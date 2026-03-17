"""Mechanism: the top-level assembly of bodies, joints, and state.

A Mechanism owns the bodies, joints, and state vector mapping.
It provides the interface for building mechanisms and is the input
to all solvers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.bodies import Body
from linkage_sim.core.constraints import (
    Constraint,
    FixedJoint,
    RevoluteJoint,
    make_fixed_joint,
    make_revolute_joint,
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
