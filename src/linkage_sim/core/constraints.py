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

    def phi_t(
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

    def phi_t(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """∂Φ/∂t = 0 for geometric joints."""
        return np.zeros(2)

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

    def phi_t(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """∂Φ/∂t = 0 for geometric joints."""
        return np.zeros(3)

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


@dataclass
class PrismaticJoint:
    """Prismatic (slider) joint: allows translation along one axis, locks rotation.

    Removes 2 DOF (1 perpendicular translation + 1 rotation).

    Constraint (2 equations):
        Φ[0] = n̂ᵢ_global · d = 0        // no displacement perpendicular to slide axis
        Φ[1] = θⱼ - θᵢ - Δθ₀ = 0        // no relative rotation

    Where:
        d = rⱼ + Aⱼ * sⱼ - rᵢ - Aᵢ * sᵢ   (global vector from point_i to point_j)
        n̂ᵢ_global = Aᵢ * n̂ᵢ                (perpendicular to slide axis, rotates with body i)
        êᵢ = slide axis in body i's local frame
        n̂ᵢ = perpendicular to êᵢ (rotate êᵢ by 90° CCW)

    The free coordinate (slide displacement) is: s_slide = êᵢ_global · d
    """

    _id: str
    _body_i_id: str
    _point_i_name: str
    _body_j_id: str
    _point_j_name: str
    _point_i_local: NDArray[np.float64]
    _point_j_local: NDArray[np.float64]
    _axis_local_i: NDArray[np.float64]
    _n_hat_local_i: NDArray[np.float64]
    _delta_theta_0: float

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
        """Φ[0] = n̂ᵢ_global · d, Φ[1] = θⱼ - θᵢ - Δθ₀"""
        # Perpendicular constraint
        theta_i = state.get_angle(self._body_i_id, q)
        A_i = state.rotation_matrix(theta_i)
        n_hat_g = A_i @ self._n_hat_local_i

        pt_i_g = state.body_point_global(self._body_i_id, self._point_i_local, q)
        pt_j_g = state.body_point_global(self._body_j_id, self._point_j_local, q)
        d = pt_j_g - pt_i_g

        # Rotation constraint
        theta_j = state.get_angle(self._body_j_id, q)

        phi = np.zeros(2)
        phi[0] = float(np.dot(n_hat_g, d))
        phi[1] = theta_j - theta_i - self._delta_theta_0
        return phi

    def phi_t(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """∂Φ/∂t = 0 for geometric joints."""
        return np.zeros(2)

    def jacobian(
        self, state: State, q: NDArray[np.float64], t: float
    ) -> NDArray[np.float64]:
        """∂Φ/∂q — 2 rows × n_coords columns.

        Row 0 (perpendicular constraint Φ[0] = n̂_g · d):
            ∂Φ[0]/∂xᵢ = -n̂_g[0]
            ∂Φ[0]/∂yᵢ = -n̂_g[1]
            ∂Φ[0]/∂θᵢ = (Bᵢ @ n̂ᵢ) · d - n̂_g · (Bᵢ @ sᵢ)
            ∂Φ[0]/∂xⱼ = n̂_g[0]
            ∂Φ[0]/∂yⱼ = n̂_g[1]
            ∂Φ[0]/∂θⱼ = n̂_g · (Bⱼ @ sⱼ)

        Row 1 (rotation constraint Φ[1] = θⱼ - θᵢ - Δθ₀):
            ∂Φ[1]/∂θᵢ = -1, ∂Φ[1]/∂θⱼ = 1
        """
        n = state.n_coords
        jac = np.zeros((2, n))

        theta_i = state.get_angle(self._body_i_id, q)
        A_i = state.rotation_matrix(theta_i)
        B_i = state.rotation_matrix_derivative(theta_i)
        n_hat_g = A_i @ self._n_hat_local_i
        B_n = B_i @ self._n_hat_local_i

        pt_i_g = state.body_point_global(self._body_i_id, self._point_i_local, q)
        pt_j_g = state.body_point_global(self._body_j_id, self._point_j_local, q)
        d = pt_j_g - pt_i_g

        if not state.is_ground(self._body_i_id):
            idx_i = state.get_index(self._body_i_id)
            # Row 0: perpendicular constraint
            jac[0, idx_i.x_idx] = -n_hat_g[0]
            jac[0, idx_i.y_idx] = -n_hat_g[1]
            B_si = B_i @ self._point_i_local
            jac[0, idx_i.theta_idx] = float(np.dot(B_n, d) - np.dot(n_hat_g, B_si))
            # Row 1: rotation constraint
            jac[1, idx_i.theta_idx] = -1.0

        if not state.is_ground(self._body_j_id):
            idx_j = state.get_index(self._body_j_id)
            theta_j = state.get_angle(self._body_j_id, q)
            B_j = state.rotation_matrix_derivative(theta_j)
            # Row 0: perpendicular constraint
            jac[0, idx_j.x_idx] = n_hat_g[0]
            jac[0, idx_j.y_idx] = n_hat_g[1]
            B_sj = B_j @ self._point_j_local
            jac[0, idx_j.theta_idx] = float(np.dot(n_hat_g, B_sj))
            # Row 1: rotation constraint
            jac[1, idx_j.theta_idx] += 1.0

        return jac

    def gamma(
        self,
        state: State,
        q: NDArray[np.float64],
        q_dot: NDArray[np.float64],
        t: float,
    ) -> NDArray[np.float64]:
        """Acceleration RHS for the prismatic joint.

        γ[0] collects the velocity-quadratic terms from Φ̈[0] = d²(n̂_g · d)/dt²:
            γ[0] = n̂_g · d · θ̇ᵢ²
                   - 2 · θ̇ᵢ · (Bᵢ @ n̂ᵢ) · ḋ
                   + n̂_g · (Aⱼ @ sⱼ) · θ̇ⱼ²
                   - n̂_g · (Aᵢ @ sᵢ) · θ̇ᵢ²

        γ[1] = 0 (rotation constraint is linear in θ).
        """
        result = np.zeros(2)

        theta_i = state.get_angle(self._body_i_id, q)
        theta_j = state.get_angle(self._body_j_id, q)
        A_i = state.rotation_matrix(theta_i)
        B_i = state.rotation_matrix_derivative(theta_i)
        A_j = state.rotation_matrix(theta_j)
        B_j = state.rotation_matrix_derivative(theta_j)

        n_hat_g = A_i @ self._n_hat_local_i
        B_n = B_i @ self._n_hat_local_i

        pt_i_g = state.body_point_global(self._body_i_id, self._point_i_local, q)
        pt_j_g = state.body_point_global(self._body_j_id, self._point_j_local, q)
        d = pt_j_g - pt_i_g

        # Velocities (zero for ground bodies)
        theta_dot_i = 0.0
        r_dot_i = np.zeros(2)
        if not state.is_ground(self._body_i_id):
            idx_i = state.get_index(self._body_i_id)
            theta_dot_i = float(q_dot[idx_i.theta_idx])
            r_dot_i = np.array([q_dot[idx_i.x_idx], q_dot[idx_i.y_idx]])

        theta_dot_j = 0.0
        r_dot_j = np.zeros(2)
        if not state.is_ground(self._body_j_id):
            idx_j = state.get_index(self._body_j_id)
            theta_dot_j = float(q_dot[idx_j.theta_idx])
            r_dot_j = np.array([q_dot[idx_j.x_idx], q_dot[idx_j.y_idx]])

        # d_dot = velocity of d vector
        d_dot = (
            (r_dot_j + B_j @ self._point_j_local * theta_dot_j)
            - (r_dot_i + B_i @ self._point_i_local * theta_dot_i)
        )

        # γ[0]: velocity-quadratic terms from Φ̈[0] = d²(n̂_g · d)/dt²
        result[0] = (
            float(np.dot(n_hat_g, d)) * theta_dot_i**2
            - 2.0 * theta_dot_i * float(np.dot(B_n, d_dot))
            + float(np.dot(n_hat_g, A_j @ self._point_j_local)) * theta_dot_j**2
            - float(np.dot(n_hat_g, A_i @ self._point_i_local)) * theta_dot_i**2
        )

        # γ[1] = 0 (rotation constraint is linear in θ)
        return result


def make_prismatic_joint(
    joint_id: str,
    body_i_id: str,
    point_i_name: str,
    point_i_local: NDArray[np.float64],
    body_j_id: str,
    point_j_name: str,
    point_j_local: NDArray[np.float64],
    axis_local_i: NDArray[np.float64],
    delta_theta_0: float = 0.0,
) -> PrismaticJoint:
    """Create a prismatic joint that allows sliding along one axis.

    Args:
        joint_id: Unique identifier for this joint.
        body_i_id: ID of the body that owns the slide axis (can be "ground").
        point_i_name: Name of the attachment point on body_i.
        point_i_local: Local coordinates of the attachment point on body_i.
        body_j_id: ID of the sliding body (can be "ground").
        point_j_name: Name of the attachment point on body_j.
        point_j_local: Local coordinates of the attachment point on body_j.
        axis_local_i: Unit vector along the slide axis in body_i's local frame.
        delta_theta_0: Initial relative angle θⱼ - θᵢ to maintain.
    """
    axis = axis_local_i.copy().astype(np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        raise ValueError("axis_local_i must be non-zero.")
    axis /= norm

    # Perpendicular: rotate axis by 90° CCW → n̂ = (-e_y, e_x)
    n_hat = np.array([-axis[1], axis[0]], dtype=np.float64)

    return PrismaticJoint(
        _id=joint_id,
        _body_i_id=body_i_id,
        _point_i_name=point_i_name,
        _body_j_id=body_j_id,
        _point_j_name=point_j_name,
        _point_i_local=point_i_local.copy(),
        _point_j_local=point_j_local.copy(),
        _axis_local_i=axis,
        _n_hat_local_i=n_hat,
        _delta_theta_0=delta_theta_0,
    )


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
