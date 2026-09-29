"""Stock magnet library ('Magnet library' sheet).

The calculator looks magnets up by exact part text. Dimensions are mm,
Br is the 20 °C remanence in tesla, max temp is the supplier rating in °C.
Add rows freely; `lookup()` returns None for unknown parts, and the model then
falls back to the manual dimensions in `MagnetInputs`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MagnetSpec:
    part: str
    vendor: str
    shape: str
    length_mm: float      # axial
    width_mm: float       # tangential
    thickness_mm: float   # radial, magnetized direction
    grade: str
    br_T: float           # remanence at 20 °C
    tmax_C: float         # supplier max operating temperature
    notes: str = ""


_ROWS = [
    MagnetSpec("B842SH", "K&J", "block", 12.7, 6.35, 3.17, "N42SH", 1.29, 150, "1/2 x 1/4 x 1/8 in, magnetized through 1/8 in"),
    MagnetSpec("B842", "K&J", "block", 12.7, 6.35, 3.17, "N42", 1.30, 80),
    MagnetSpec("B842-N52", "K&J", "block", 12.7, 6.35, 3.17, "N52", 1.45, 80),
    MagnetSpec("B822", "K&J", "block", 12.7, 3.17, 3.17, "N42", 1.30, 80, "1/2 x 1/8 x 1/8 in"),
    MagnetSpec("B862", "K&J", "block", 12.7, 9.5, 3.17, "N42", 1.30, 80, "1/2 x 3/8 x 1/8 in"),
    MagnetSpec("B882", "K&J", "block", 12.7, 12.7, 3.17, "N42", 1.30, 80, "1/2 x 1/2 x 1/8 in"),
    MagnetSpec("B882-N52", "K&J", "block", 12.7, 12.7, 3.17, "N52", 1.45, 80),
    MagnetSpec("B861", "K&J", "block", 12.7, 9.5, 1.59, "N42", 1.30, 80, "1/2 x 3/8 x 1/16 in"),
    MagnetSpec("B881", "K&J", "block", 12.7, 12.7, 1.59, "N42", 1.30, 80, "1/2 x 1/2 x 1/16 in (check stock)"),
    MagnetSpec("B442", "K&J", "block", 6.35, 6.35, 3.17, "N42", 1.30, 80, "1/4 x 1/4 x 1/8 in"),
    MagnetSpec("BX042SH", "K&J", "block", 25.4, 6.35, 3.17, "N42SH", 1.29, 150, "1 x 1/4 x 1/8 in"),
    MagnetSpec("BX082SH", "K&J", "block", 25.4, 12.7, 3.17, "N42SH", 1.29, 150, "1 x 1/2 x 1/8 in"),
    MagnetSpec("M5044", "SuperMagnetMan", "arc", 13, 5.5, 1.31, "N50", 1.42, 80,
               "22.60 OD x 19.97 ID x 13, 29.8 deg, 12 pcs; width = mean arc length"),
    MagnetSpec("M5045", "SuperMagnetMan", "arc", 6.56, 5.6, 1.12, "N50M", 1.42, 100, "22.70 OD x 20.47 ID x 6.56, 12 pcs"),
    MagnetSpec("M5026", "SuperMagnetMan", "arc", 15, 6.4, 1.67, "N50", 1.42, 80, "26.60 OD x 23.26 ID x 15, 12 pcs"),
]

MAGNET_LIBRARY: dict[str, MagnetSpec] = {m.part: m for m in _ROWS}


def lookup(part: str | None) -> MagnetSpec | None:
    """Exact-text lookup, like the workbook's INDEX/MATCH. Returns None if not found."""
    if not part:
        return None
    return MAGNET_LIBRARY.get(part)
