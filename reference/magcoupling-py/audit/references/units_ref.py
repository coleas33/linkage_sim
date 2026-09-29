"""Minimal quantity algebra for dimensional checks (SI base dimensions m, kg, s, A, K).

A Quantity stores its value in SI base units plus integer dimension exponents. Multiplying and dividing combine
both; adding or subtracting quantities of different dimensions raises DimensionError; `to(unit)` converts to a
display unit and raises when the dimensions differ. Units are Quantities of value "one of that unit", so
`3.2 * MM` is 0.0032 m. Nothing here imports the engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_BASE = ("m", "kg", "s", "A", "K")


class DimensionError(ValueError):
    """Raised when an operation mixes incompatible dimensions."""


@dataclass(frozen=True)
class Quantity:
    value: float
    dims: tuple = (0, 0, 0, 0, 0)

    # -- arithmetic
    def __mul__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, tuple(a + b for a, b in zip(self.dims, other.dims)))
        return Quantity(self.value * other, self.dims)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, tuple(a - b for a, b in zip(self.dims, other.dims)))
        return Quantity(self.value / other, self.dims)

    def __rtruediv__(self, other):
        return Quantity(other / self.value, tuple(-a for a in self.dims))

    def __pow__(self, n: int):
        if not isinstance(n, int):
            raise TypeError("integer powers only")
        return Quantity(self.value ** n, tuple(a * n for a in self.dims))

    def __neg__(self):
        return Quantity(-self.value, self.dims)

    def _same(self, other, op: str):
        if not isinstance(other, Quantity) or other.dims != self.dims:
            raise DimensionError(f"cannot {op} {self.describe()} and {other!r}")

    def __add__(self, other):
        self._same(other, "add")
        return Quantity(self.value + other.value, self.dims)

    def __sub__(self, other):
        self._same(other, "subtract")
        return Quantity(self.value - other.value, self.dims)

    # -- inspection
    def is_dimensionless(self) -> bool:
        return all(a == 0 for a in self.dims)

    def dimensionless(self) -> float:
        """The plain number, for exp/sinh arguments; raises unless dimensionless."""
        if not self.is_dimensionless():
            raise DimensionError(f"expected a dimensionless quantity, got {self.describe()}")
        return self.value

    def to(self, unit: "Quantity") -> float:
        """Value expressed in `unit`; raises when the dimensions differ."""
        if unit.dims != self.dims:
            raise DimensionError(f"cannot express {self.describe()} in {unit.describe()}")
        return self.value / unit.value

    def describe(self) -> str:
        parts = [f"{b}^{e}" for b, e in zip(_BASE, self.dims) if e]
        return f"{self.value:g} " + ("·".join(parts) if parts else "(dimensionless)")


def _base(i: int) -> Quantity:
    dims = [0, 0, 0, 0, 0]
    dims[i] = 1
    return Quantity(1.0, tuple(dims))


ONE = Quantity(1.0)
M, KG, S, A, K = (_base(i) for i in range(5))

MM = 1e-3 * M
GRAM = 1e-3 * KG
N = KG * M / S ** 2
NM = N * M
PA = N / M ** 2
MPA = 1e6 * PA
J = N * M
W = J / S
TESLA = KG / (S ** 2 * A)
H_PER_M = TESLA * M / A            # the unit of mu0
RAD_PER_S = ONE / S                # radians are dimensionless
RPM = (2.0 * math.pi / 60.0) * RAD_PER_S   # rev/min as an angular speed
J_PER_KG_K = J / (KG * K)
J_PER_K = J / K
W_PER_K = W / K
A_PER_M = A / M                    # magnetic field strength H
S_PER_M = A ** 2 * S ** 3 / (KG * M ** 3)   # electrical conductivity (siemens per metre)
PER_K = ONE / K                    # thermal expansion coefficient
KG_M2 = KG * M ** 2                # mass moment of inertia
