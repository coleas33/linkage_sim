"""Independent re-derivations for the Metal design sheet and the plating offsets (M1 audit, Task 4).

Imports nothing from ``magcoupling``. Temperature scaling lives in ``audit.references.remanence``.
Each function states the physical rule it encodes:

- A symmetric production allowance v gives a low value T (1 - v) and a high value T (1 + v).
- A gearbox of ratio i and efficiency eta, motor driving: speeds w_in = i w_out and power
  P_out = eta P_in, so T_out = i eta T_in and T_in = T_out / (i eta).
- Radial clearances stack linearly (worst case); axial stacks add.
- A coupling with N poles per ring has N/2 pole pairs; slipping at n rpm passes (N/2) n / 60
  pole pairs per second, and each pole-pair pass is one full torque/field cycle.
- n rpm is an angular speed w = 2 pi n / 60 rad/s; mechanical power is torque times angular
  speed, P = T w.
- Plating grows every coated surface by its thickness along the surface's outward normal
  (out of the material).
"""
from __future__ import annotations

import math


def torque_low(torque_Nm: float, variation: float) -> float:
    """Low end of a symmetric allowance."""
    return torque_Nm * (1 - variation)


def torque_high(torque_Nm: float, variation: float) -> float:
    """High end of a symmetric allowance."""
    return torque_Nm * (1 + variation)


def gearbox_input_torque(output_Nm: float, ratio: float, efficiency: float) -> float:
    """Gearbox input torque for an output torque, motor driving: T_in = T_out / (i eta).
    (Back-driven by the wheel the input sees T_out eta / i, which is smaller; the driving form
    is the conservative one.)"""
    return output_Nm / (ratio * efficiency)


def gearbox_output_torque(input_Nm: float, ratio: float, efficiency: float) -> float:
    """Gearbox output torque for an input torque, motor driving: T_out = i eta T_in."""
    return input_Nm * ratio * efficiency


def sleeve_liner_clearance(corner_gap: float, sleeve: float, liner: float,
                           sleeve_bedding: float, liner_bedding: float) -> float:
    """Radial space between the round sleeve (over the inner corners) and the round liner
    (inside the outer faces): the corner gap minus both retainer walls and both beddings."""
    return corner_gap - sleeve - liner - sleeve_bedding - liner_bedding


def linear_stack(*items: float) -> float:
    """Worst-case (arithmetic) stack of allowances or lengths."""
    return math.fsum(items)


def pole_pairs(npole: int) -> float:
    """Pole pairs per ring of a coupling with ``npole`` poles per ring: N/2."""
    return npole / 2


def pole_pair_frequency_Hz(npole: int, rpm: float) -> float:
    """Pole pairs passing per second at a relative speed of ``rpm``."""
    return pole_pairs(npole) * rpm / 60


def omega_rad_s(rpm: float) -> float:
    """Angular speed [rad/s] of ``rpm`` revolutions per minute: w = 2 pi rpm / 60."""
    return 2 * math.pi * rpm / 60


def shaft_power_W(torque_Nm: float, rpm: float) -> float:
    """P = T w with w = omega_rad_s(rpm)."""
    return torque_Nm * omega_rad_s(rpm)


def plated_size(size: float, thickness: float, surfaces: int, external: bool) -> float:
    """Size after plating. An external feature (hub flat apothem, outside diameter) grows by
    ``thickness`` per plated surface it spans; an internal feature (pocket apothem, bore)
    shrinks by the same amount, because the coating grows out of the material."""
    return size + (1 if external else -1) * surfaces * thickness


def preplate_size(finished: float, thickness: float, surfaces: int, external: bool) -> float:
    """Machined (pre-plate) size that plates to ``finished``. Plating shifts a size by a
    constant, so the machined size is the finished size minus that shift."""
    growth = plated_size(0.0, thickness, surfaces, external)
    return finished - growth
