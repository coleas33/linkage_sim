"""PointMass element and composite mass recomputation.

A PointMass adds concentrated mass at a body-local position. After adding
point masses, recompute the body's composite mass properties (total mass,
CG location, moment of inertia) using the parallel axis theorem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from linkage_sim.core.bodies import Body


@dataclass(frozen=True)
class PointMass:
    """A concentrated mass at a body-local position.

    Attributes:
        label: Descriptive name.
        mass: Mass in kg.
        local_position: Position in body-local coordinates (m).
    """

    label: str
    mass: float
    local_position: NDArray[np.float64]


def add_point_mass(body: Body, pm: PointMass) -> None:
    """Add a point mass and recompute composite mass properties.

    Updates body.mass, body.cg_local, and body.Izz_cg using the
    parallel axis theorem.

    The new composite CG is the mass-weighted average of the old CG
    and the point mass location. The new Izz_cg accounts for both
    the old Izz shifted to the new CG and the point mass contribution.

    Args:
        body: Body to modify (mutated in place).
        pm: Point mass to add.
    """
    m_old = body.mass
    m_new = pm.mass
    m_total = m_old + m_new

    if m_total <= 0.0:
        return

    # New composite CG
    cg_old = body.cg_local
    cg_new_pos = (m_old * cg_old + m_new * pm.local_position) / m_total

    # Shift old Izz to new CG via parallel axis theorem
    d_old = cg_old - cg_new_pos
    Izz_old_at_new_cg = body.Izz_cg + m_old * float(np.dot(d_old, d_old))

    # Point mass contribution at new CG
    d_pm = pm.local_position - cg_new_pos
    Izz_pm_at_new_cg = m_new * float(np.dot(d_pm, d_pm))

    # Update body
    body.mass = m_total
    body.cg_local = cg_new_pos
    body.Izz_cg = Izz_old_at_new_cg + Izz_pm_at_new_cg
