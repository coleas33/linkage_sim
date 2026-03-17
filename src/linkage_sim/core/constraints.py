"""Joint constraint equations, Jacobians, and gamma (acceleration RHS).

Each constraint type provides:
  - constraint(q, t) -> residual vector Φ
  - jacobian(q, t)   -> Jacobian matrix rows ∂Φ/∂q
  - gamma(q, q_dot, t) -> acceleration RHS contribution
  - n_equations       -> number of constraint rows

Constraints connect two bodies at specified attachment points.
When one body is ground, its terms become constants (no q entries).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.state import State


class Constraint(Protocol):
    """Interface that all joint constraints must implement."""

    @property
    def id(self) -> str: ...

    @property
    def n_equations(self) -> int: ...

    @property
    def body_i_id(self) -> str: ...

    @property
    def dof_removed(self) -> int: ...

    @property
    def body_j_id(self) -> str: ...

    def constraint(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]: ...

    def jacobian(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]: ...

    def gamma(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]: ...


@dataclass
class RevoluteJoint:
    """Revolute joint: constrains two attachment points to be coincident.

    Removes 2 translational DOF. Allows relative rotation.

    Constraint (2 equations):
        Φ = rᵢ + Aᵢ * sᵢ - rⱼ - Aⱼ * sⱼ = 0

    Where:
        rᵢ, rⱼ = body origin positions in global frame
        Aᵢ, Aⱼ = body rotation matrices
        sᵢ, sⱼ = attachment point locations in body-local frames
    """

    _id: str
    _body_i_id: str
    _point_i_name: str
    _body_j_id: str
    _point_j_name: str
    _point_i_local: NDArray[np.float64]
    _point_j_local: NDArray[np.float64]

    @property
    def id(self) -> str:
        return self._id

    @property
    def n_equations(self) -> int:
        return 2

    @property
    def dof_removed(self) -> int:
        return 2

    @property
    def body_i_id(self) -> str:
        return self._body_i_id

    @property
    def body_j_id(self) -> str:
        return self._body_j_id

    def constraint(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """Φ = rᵢ + Aᵢ * sᵢ - rⱼ - Aⱼ * sⱼ"""
        global_i = state.body_point_global(self._body_i_id, self._point_i_local, q)
        global_j = state.body_point_global(self._body_j_id, self._point_j_local, q)
        return global_i - global_j

    def jacobian(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """∂Φ/∂q — 2 rows × n_coords columns.

        For body i (if not ground):
            ∂Φ/∂xᵢ = [1, 0]ᵀ
            ∂Φ/∂yᵢ = [0, 1]ᵀ
            ∂Φ/∂θᵢ = Bᵢ * sᵢ

        For body j (if not ground):
            ∂Φ/∂xⱼ = [-1, 0]ᵀ
            ∂Φ/∂yⱼ = [0, -1]ᵀ
            ∂Φ/∂θⱼ = -Bⱼ * sⱼ
        """
        n = state.n_coords
        jac = np.zeros((2, n))

        if not state.is_ground(self._body_i_id):
            idx_i = state.get_index(self._body_i_id)
            jac[0, idx_i.x_idx] = 1.0
            jac[1, idx_i.y_idx] = 1.0
            B_si = state.body_point_global_derivative(
                self._body_i_id, self._point_i_local, q
            )
            jac[0, idx_i.theta_idx] = B_si[0]
            jac[1, idx_i.theta_idx] = B_si[1]

        if not state.is_ground(self._body_j_id):
            idx_j = state.get_index(self._body_j_id)
            jac[0, idx_j.x_idx] = -1.0
            jac[1, idx_j.y_idx] = -1.0
            B_sj = state.body_point_global_derivative(
                self._body_j_id, self._point_j_local, q
            )
            jac[0, idx_j.theta_idx] = -B_sj[0]
            jac[1, idx_j.theta_idx] = -B_sj[1]

        return jac

    def gamma(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Acceleration RHS: γ = -(∂²(A*s)/∂θ²) * θ̇² for each body.

        For body i (if not ground):
            γ contribution = Aᵢ * sᵢ * θ̇ᵢ²

        For body j (if not ground):
            γ contribution = -Aⱼ * sⱼ * θ̇ⱼ²

        The sign follows from: Φ_q * q̈ = γ where
            γ = -(Φ_q * q̇)_q * q̇ = Σ Aᵢ * sᵢ * θ̇ᵢ² (centripetal terms)
        """
        result = np.zeros(2)

        if not state.is_ground(self._body_i_id):
            theta_i = state.get_angle(self._body_i_id, q)
            idx_i = state.get_index(self._body_i_id)
            theta_dot_i = q_dot[idx_i.theta_idx]
            A_i = state.rotation_matrix(theta_i)
            result += (A_i @ self._point_i_local) * theta_dot_i**2

        if not state.is_ground(self._body_j_id):
            theta_j = state.get_angle(self._body_j_id, q)
            idx_j = state.get_index(self._body_j_id)
            theta_dot_j = q_dot[idx_j.theta_idx]
            A_j = state.rotation_matrix(theta_j)
            result -= (A_j @ self._point_j_local) * theta_dot_j**2

        return result


@dataclass
class FixedJoint:
    """Fixed joint: constrains all relative motion to zero.

    Removes 3 DOF (2 translational + 1 rotational).

    Constraint (3 equations):
        Φ[0:2] = rᵢ + Aᵢ * sᵢ - rⱼ - Aⱼ * sⱼ = 0   (coincident points)
        Φ[2]   = θⱼ - θᵢ - Δθ₀ = 0                     (locked relative angle)

    Δθ₀ is the initial relative angle, captured at construction time.
    """

    _id: str
    _body_i_id: str
    _point_i_name: str
    _body_j_id: str
    _point_j_name: str
    _point_i_local: NDArray[np.float64]
    _point_j_local: NDArray[np.float64]
    _delta_theta_0: float

    @property
    def id(self) -> str:
        return self._id

    @property
    def n_equations(self) -> int:
        return 3

    @property
    def dof_removed(self) -> int:
        return 3

    @property
    def body_i_id(self) -> str:
        return self._body_i_id

    @property
    def body_j_id(self) -> str:
        return self._body_j_id

    def constraint(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """Φ[0:2] = position coincidence, Φ[2] = rotation lock."""
        global_i = state.body_point_global(self._body_i_id, self._point_i_local, q)
        global_j = state.body_point_global(self._body_j_id, self._point_j_local, q)

        theta_i = state.get_angle(self._body_i_id, q)
        theta_j = state.get_angle(self._body_j_id, q)

        phi = np.zeros(3)
        phi[0:2] = global_i - global_j
        phi[2] = theta_j - theta_i - self._delta_theta_0
        return phi

    def jacobian(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """∂Φ/∂q — 3 rows × n_coords columns.

        First 2 rows: same as revolute Jacobian.
        Third row: ∂Φ[2]/∂θⱼ = 1, ∂Φ[2]/∂θᵢ = -1, all position partials = 0.
        """
        n = state.n_coords
        jac = np.zeros((3, n))

        # Position constraint rows (same as revolute)
        if not state.is_ground(self._body_i_id):
            idx_i = state.get_index(self._body_i_id)
            jac[0, idx_i.x_idx] = 1.0
            jac[1, idx_i.y_idx] = 1.0
            B_si = state.body_point_global_derivative(
                self._body_i_id, self._point_i_local, q
            )
            jac[0, idx_i.theta_idx] = B_si[0]
            jac[1, idx_i.theta_idx] = B_si[1]
            # Rotation constraint: ∂Φ[2]/∂θᵢ = -1
            jac[2, idx_i.theta_idx] = -1.0

        if not state.is_ground(self._body_j_id):
            idx_j = state.get_index(self._body_j_id)
            jac[0, idx_j.x_idx] = -1.0
            jac[1, idx_j.y_idx] = -1.0
            B_sj = state.body_point_global_derivative(
                self._body_j_id, self._point_j_local, q
            )
            jac[0, idx_j.theta_idx] = -B_sj[0]
            jac[1, idx_j.theta_idx] = -B_sj[1]
            # Rotation constraint: ∂Φ[2]/∂θⱼ = 1
            jac[2, idx_j.theta_idx] += 1.0

        return jac

    def gamma(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Acceleration RHS.

        First 2 rows: same centripetal terms as revolute.
        Third row: γ[2] = 0 (rotation constraint is linear in θ).
        """
        result = np.zeros(3)

        if not state.is_ground(self._body_i_id):
            theta_i = state.get_angle(self._body_i_id, q)
            idx_i = state.get_index(self._body_i_id)
            theta_dot_i = q_dot[idx_i.theta_idx]
            A_i = state.rotation_matrix(theta_i)
            result[0:2] += (A_i @ self._point_i_local) * theta_dot_i**2

        if not state.is_ground(self._body_j_id):
            theta_j = state.get_angle(self._body_j_id, q)
            idx_j = state.get_index(self._body_j_id)
            theta_dot_j = q_dot[idx_j.theta_idx]
            A_j = state.rotation_matrix(theta_j)
            result[0:2] -= (A_j @ self._point_j_local) * theta_dot_j**2

        # γ[2] = 0 (no velocity-dependent terms in the rotation constraint)
        return result


def make_fixed_joint(
    joint_id: str,
    body_i_id: str,
    point_i_name: str,
    point_i_local: NDArray[np.float64],
    body_j_id: str,
    point_j_name: str,
    point_j_local: NDArray[np.float64],
    delta_theta_0: float = 0.0,
) -> FixedJoint:
    """Create a fixed joint that locks all relative motion.

    Args:
        joint_id: Unique identifier for this joint.
        body_i_id: ID of the first body (can be "ground").
        point_i_name: Name of the attachment point on body_i.
        point_i_local: Local coordinates of the attachment point on body_i.
        body_j_id: ID of the second body (can be "ground").
        point_j_name: Name of the attachment point on body_j.
        point_j_local: Local coordinates of the attachment point on body_j.
        delta_theta_0: Initial relative angle θⱼ - θᵢ to maintain.
            Default 0 means the bodies keep their initial angular alignment.
    """
    return FixedJoint(
        _id=joint_id,
        _body_i_id=body_i_id,
        _point_i_name=point_i_name,
        _body_j_id=body_j_id,
        _point_j_name=point_j_name,
        _point_i_local=point_i_local.copy(),
        _point_j_local=point_j_local.copy(),
        _delta_theta_0=delta_theta_0,
    )


def make_revolute_joint(
    joint_id: str,
    body_i_id: str,
    point_i_name: str,
    point_i_local: NDArray[np.float64],
    body_j_id: str,
    point_j_name: str,
    point_j_local: NDArray[np.float64],
) -> RevoluteJoint:
    """Create a revolute joint between two bodies at specified attachment points.

    Args:
        joint_id: Unique identifier for this joint.
        body_i_id: ID of the first body (can be "ground").
        point_i_name: Name of the attachment point on body_i.
        point_i_local: Local coordinates of the attachment point on body_i.
        body_j_id: ID of the second body (can be "ground").
        point_j_name: Name of the attachment point on body_j.
        point_j_local: Local coordinates of the attachment point on body_j.
    """
    return RevoluteJoint(
        _id=joint_id,
        _body_i_id=body_i_id,
        _point_i_name=point_i_name,
        _body_j_id=body_j_id,
        _point_j_name=point_j_name,
        _point_i_local=point_i_local.copy(),
        _point_j_local=point_j_local.copy(),
    )
