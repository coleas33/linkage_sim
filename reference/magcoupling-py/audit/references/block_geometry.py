"""Independent flat-block coupling geometry, solid masses and polar inertia (M1 audit, Task 4).

Written from first principles; imports nothing from ``magcoupling``. Shapes are
built explicitly (block rectangles, pocket polygons from intersecting side
lines) and measured with generic tools (vertex radii, point-to-segment
distances, shoelace area and polar moment). The only shared content with the
workbook is the *definition* of each quantity, so a transcription or algebra
slip in the workbook shows up as a mismatch.

Conventions: lengths mm, areas mm², densities g/mm³, masses g. ``Part`` holds a
mass and its polar moment of inertia about the coupling axis in g·mm²
(1 g·mm² = 1e-9 kg·m²). Block 0 of each ring and side 0 of each polygon are
centred on the +x axis.

The ``*_for_design`` adapters at the bottom only read input/result attributes by
name (duck typing) so tests can build a reference from a ``DesignInputs`` object;
they contain no formulas beyond the calls into the functions above them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import minimize_scalar

Point = tuple[float, float]

#: Sintered NdFeB density, 7.5 g/cm³ (datasheets quote 7.4–7.7 g/cm³, e.g. Arnold
#: Magnetic Technologies N42SH, K&J Magnetics "Magnet specifications").
NDFEB_DENSITY_G_MM3 = 7.5e-3


# --------------------------------------------------------------------------- 2D primitives
def regular_polygon(apothem: float, n: int, phase: float = 0.0) -> list[Point]:
    """Vertices (counter-clockwise) of a regular n-gon whose side k lies on the line
    x·cos(θk) + y·sin(θk) = apothem, θk = phase + 2πk/n. Vertex k is the intersection
    of sides k and k+1, solved by Cramer's rule (no circumradius formula used)."""
    pts: list[Point] = []
    for k in range(n):
        t1 = phase + 2 * math.pi * k / n
        t2 = phase + 2 * math.pi * (k + 1) / n
        det = math.cos(t1) * math.sin(t2) - math.sin(t1) * math.cos(t2)
        x = apothem * (math.sin(t2) - math.sin(t1)) / det
        y = apothem * (math.cos(t1) - math.cos(t2)) / det
        pts.append((x, y))
    return pts


def polygon_area(pts: list[Point]) -> float:
    """Shoelace area (absolute value) of a simple polygon."""
    s = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def polygon_polar_moment(pts: list[Point]) -> float:
    """Polar second moment of area about the origin, J_O = ∬(x² + y²) dA, from the
    standard vertex formula Σ c_i·(x_i² + x_i·x_j + x_j² + y_i² + y_i·y_j + y_j²)/12 with
    c_i = x_i·y_j − x_j·y_i (absolute value, so vertex order does not matter)."""
    s = 0.0
    for i, (x1, y1) in enumerate(pts):
        x2, y2 = pts[(i + 1) % len(pts)]
        c = x1 * y2 - x2 * y1
        s += c * (x1 * x1 + x1 * x2 + x2 * x2 + y1 * y1 + y1 * y2 + y2 * y2)
    return abs(s) / 12


def side_length(pts: list[Point]) -> float:
    """Length of the first side (vertex 0 to vertex 1) of a polygon."""
    return math.dist(pts[0], pts[1])


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    """Euclidean distance from point p to the closed segment ab."""
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    L2 = dx * dx + dy * dy
    s = 0.0 if L2 == 0.0 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / L2))
    return math.hypot(p[0] - (ax + s * dx), p[1] - (ay + s * dy))


def convex_polygon_distance(P: list[Point], Q: list[Point]) -> float:
    """Minimum distance between two disjoint convex polygons: the smallest
    vertex-to-edge distance taken both ways (exact for non-intersecting convex sets)."""
    best = math.inf
    for A, B in ((P, Q), (Q, P)):
        for v in A:
            for i in range(len(B)):
                best = min(best, point_segment_distance(v, B[i], B[(i + 1) % len(B)]))
    return best


def axial_overlap(length_a: float, length_b: float, offset: float = 0.0) -> float:
    """Overlap of two axial intervals: [−a/2, a/2] and [offset − b/2, offset + b/2]."""
    lo = max(-length_a / 2, offset - length_b / 2)
    hi = min(length_a / 2, offset + length_b / 2)
    return max(0.0, hi - lo)


def max_radius(P: list[Point]) -> float:
    """Largest distance from the axis to the polygon (attained at a vertex)."""
    return max(math.hypot(x, y) for x, y in P)


def min_radius(P: list[Point]) -> float:
    """Smallest distance from the axis to the polygon boundary (polygon not containing the axis)."""
    return min(point_segment_distance((0.0, 0.0), P[i], P[(i + 1) % len(P)]) for i in range(len(P)))


def block_rectangle(near_radius: float, thickness: float, width: float, angle: float) -> list[Point]:
    """Flat block centred on the ray at polar angle ``angle``: its near face is the chord at
    distance ``near_radius`` from the axis, it extends ``thickness`` radially outward and
    ``width`` tangentially. Vertices counter-clockwise."""
    er = (math.cos(angle), math.sin(angle))
    et = (-math.sin(angle), math.cos(angle))
    local = ((near_radius, -width / 2), (near_radius + thickness, -width / 2),
             (near_radius + thickness, width / 2), (near_radius, width / 2))
    return [(r * er[0] + u * et[0], r * er[1] + u * et[1]) for r, u in local]


def ring_min_clearance(npole: int, inner_back: float, t_i: float, w_i: float,
                       outer_face: float, t_o: float, w_o: float, scan: int = 360) -> float:
    """Smallest 2D distance between the inner and outer block rings over every relative rotation.

    Inner block 0 sits at angle 0 (all inner blocks are equivalent by symmetry); outer block k
    sits at φ + 2πk/N. φ is scanned over one pole pitch, then refined with a bounded Brent
    search (xatol 1e-12 rad) around the best sample."""
    inner = block_rectangle(inner_back, t_i, w_i, 0.0)
    pitch = 2 * math.pi / npole

    def clearance(phi: float) -> float:
        return min(convex_polygon_distance(inner, block_rectangle(outer_face, t_o, w_o, phi + pitch * k))
                   for k in range(npole))

    phis = [pitch * j / scan for j in range(scan)]
    vals = [clearance(p) for p in phis]
    j = min(range(scan), key=vals.__getitem__)
    step = pitch / scan
    res = minimize_scalar(clearance, bounds=(phis[j] - step, phis[j] + step), method="bounded",
                          options={"xatol": 1e-12})
    return min(float(res.fun), vals[j])


# --------------------------------------------------------------------------- coupling geometry
@dataclass(frozen=True)
class CouplingGeometry:
    inner_face_radius: float        # apothem of the inner blocks' outer faces
    inner_corner_radius: float      # largest radius of the inner ring (block corners, or arc OD)
    outer_face_apothem: float       # apothem of the outer blocks' inner faces
    face_gap: float                 # magnetic gap at the flat centres
    corner_gap: float               # inner ring's largest radius to the outer faces
    outer_back_apothem: float       # outer block backs (without bondline)
    pocket_apothem: float           # cup pocket flats (block back + bondline)
    pocket_corner_radius: float     # pocket polygon vertex radius (or pocket radius for arcs)
    cup_od: float
    cup_wall_flat: float            # steel from the pocket flat to the cup OD
    gap_radius: float
    pole_pitch: float               # at the gap radius
    fill_inner: float               # block width / pole arc at the block's mid-thickness radius (planar unrolling)
    fill_outer: float
    inner_flat_width: float         # side of the polygon through the inner block backs
    outer_flat_width: float         # side of the polygon through the outer block faces
    hub_apothem: float              # machined hub flats (block back minus bondline)
    hub_wall: float                 # hub flat to bore
    hub_wall_past_key: float


def coupling_geometry(npole: int, inner_back_apothem: float, t_i: float, w_i: float, t_o: float, w_o: float,
                      face_gap: float, bond_inner: float, bond_outer: float, cup_wall_corner: float,
                      bore: float, keyway_depth: float, faceted: bool = True) -> CouplingGeometry:
    """Every block/polygon dimension from the inputs, by construction.

    Definitions used: the flat-face gap is the distance between the facing block faces at the
    flat centres, for flat blocks and for arcs alike; the cup wall input is the minimum steel
    at the pocket corners; the hub apothem is the block-back apothem minus the bondline; the
    keyway depth is measured radially from the bore."""
    pitch_angle = 2 * math.pi / npole
    inner_block = block_rectangle(inner_back_apothem, t_i, w_i, 0.0)
    r_face = inner_back_apothem + t_i
    r_corner = max_radius(inner_block) if faceted else r_face
    a_o = r_face + face_gap
    a_back = a_o + t_o
    pocket_ap = a_back + bond_outer
    r_pocket = max_radius(regular_polygon(pocket_ap, npole)) if faceted else pocket_ap
    od = 2 * (r_pocket + cup_wall_corner)
    r_gap = r_face + face_gap / 2
    r_mid_i = inner_back_apothem + t_i / 2
    r_mid_o = a_o + t_o / 2
    hub_ap = inner_back_apothem - bond_inner
    return CouplingGeometry(
        inner_face_radius=r_face,
        inner_corner_radius=r_corner,
        outer_face_apothem=a_o,
        face_gap=a_o - r_face,
        corner_gap=a_o - r_corner,
        outer_back_apothem=a_back,
        pocket_apothem=pocket_ap,
        pocket_corner_radius=r_pocket,
        cup_od=od,
        cup_wall_flat=od / 2 - pocket_ap,
        gap_radius=r_gap,
        pole_pitch=r_gap * pitch_angle,
        fill_inner=min(1.0, w_i / (r_mid_i * pitch_angle)),
        fill_outer=min(1.0, w_o / (r_mid_o * pitch_angle)),
        inner_flat_width=side_length(regular_polygon(inner_back_apothem, npole)),
        outer_flat_width=side_length(regular_polygon(a_o, npole)),
        hub_apothem=hub_ap,
        hub_wall=hub_ap - bore / 2,
        hub_wall_past_key=hub_ap - bore / 2 - keyway_depth,
    )


# --------------------------------------------------------------------------- solids
@dataclass(frozen=True)
class Part:
    mass_g: float
    polar_inertia_g_mm2: float      # about the coupling axis

    def __add__(self, other: "Part") -> "Part":
        return Part(self.mass_g + other.mass_g, self.polar_inertia_g_mm2 + other.polar_inertia_g_mm2)


def prism_part(area: float, polar_moment: float, length: float, rho: float) -> Part:
    """Straight prism along the axis: mass = ρ·A·L, J_mass = ρ·L·J_area (J_area about the axis)."""
    return Part(rho * area * length, rho * polar_moment * length)


def polygon_part(pts: list[Point], length: float, rho: float) -> Part:
    return prism_part(polygon_area(pts), polygon_polar_moment(pts), length, rho)


def disk_section(diameter: float) -> tuple[float, float]:
    """(area, polar moment about the centre) of a full disk: πD²/4 and πD⁴/32."""
    return math.pi * diameter ** 2 / 4, math.pi * diameter ** 4 / 32


def annulus_part(od: float, id_: float, length: float, rho: float) -> Part:
    """Tube: disk(od) minus disk(id)."""
    a_o, j_o = disk_section(od)
    a_i, j_i = disk_section(id_)
    return prism_part(a_o - a_i, j_o - j_i, length, rho)


def rotating_parts(npole: int, inner_back_apothem: float, t_i: float, w_i: float, L_i: float,
                   outer_face_apothem: float, t_o: float, w_o: float, L_o: float,
                   pocket_apothem: float, hub_apothem: float, cup_od: float, bore: float,
                   cup_depth: float, web: float, hub_length: float, boss_length: float, boss_od: float,
                   cup_density: float, hub_density: float, faceted: bool = True,
                   magnet_density: float = NDFEB_DENSITY_G_MM3) -> dict[str, Part]:
    """Gross solids (no holes, slots, keyways or threads subtracted), matching the workbook's
    documented simplification.

    magnets: 2N explicit block rectangles × length. cup: disk(OD) minus the pocket cavity
    (regular N-gon for flat blocks, circle for arcs) over the cavity depth, plus the rear web
    disk(OD) minus the bore. hub: hub polygon (circle for arcs) minus the bore. boss: tube.
    The web and boss are one piece with the cup, so they take ``cup_density``."""
    pitch = 2 * math.pi / npole
    magnets = Part(0.0, 0.0)
    for k in range(npole):
        magnets = magnets + polygon_part(block_rectangle(inner_back_apothem, t_i, w_i, pitch * k), L_i, magnet_density)
        magnets = magnets + polygon_part(block_rectangle(outer_face_apothem, t_o, w_o, pitch * k), L_o, magnet_density)
    a_od, j_od = disk_section(cup_od)
    a_bore, j_bore = disk_section(bore)
    if faceted:
        pocket = regular_polygon(pocket_apothem, npole)
        a_cav, j_cav = polygon_area(pocket), polygon_polar_moment(pocket)
        hub_poly = regular_polygon(hub_apothem, npole)
        a_hub, j_hub = polygon_area(hub_poly), polygon_polar_moment(hub_poly)
    else:
        a_cav, j_cav = disk_section(2 * pocket_apothem)
        a_hub, j_hub = disk_section(2 * hub_apothem)
    cup = (prism_part(a_od - a_cav, j_od - j_cav, cup_depth, cup_density)
           + prism_part(a_od - a_bore, j_od - j_bore, web, cup_density))
    hub = prism_part(a_hub - a_bore, j_hub - j_bore, hub_length, hub_density)
    boss = annulus_part(boss_od, bore, boss_length, cup_density)
    return {"magnets": magnets, "cup": cup, "hub": hub, "boss": boss}


# --------------------------------------------------------------------------- retainers
@dataclass(frozen=True)
class RetainerGeometry:
    sleeve_id: float
    sleeve_od: float
    liner_od: float
    liner_id: float
    endplate_od: float


def retainer_geometry(inner_ring_max_radius: float, outer_ring_min_radius: float, sleeve: float, liner: float,
                      sleeve_bedding: float, liner_bedding: float) -> RetainerGeometry:
    """Round sleeve over the inner ring (clears its largest radius by the bedding clearance) and
    round liner inside the outer ring (clears its smallest radius). Endplates cap the inner
    magnets up to the sleeve bore."""
    sleeve_id = 2 * (inner_ring_max_radius + sleeve_bedding)
    liner_od = 2 * (outer_ring_min_radius - liner_bedding)
    return RetainerGeometry(sleeve_id=sleeve_id, sleeve_od=sleeve_id + 2 * sleeve, liner_od=liner_od,
                            liner_id=liner_od - 2 * liner, endplate_od=sleeve_id)


def retainer_parts(g: RetainerGeometry, span: float, retainer_density: float, cap_od: float, cap_face: float,
                   cap_thread_dia: float, cap_thread_engagement: float, al_density: float, bore: float,
                   front_endplate: float, rear_endplate: float, rear_hole: float) -> dict[str, Part]:
    """Sleeve and liner tubes over the retainer span; the aluminium cap as a face plate
    (cap OD to the liner ID opening) plus the threaded skirt (cap OD to thread diameter);
    the two 316L endplates (front: bore opening, rear: screw-clearance opening)."""
    return {
        "sleeve": annulus_part(g.sleeve_od, g.sleeve_id, span, retainer_density),
        "liner": annulus_part(g.liner_od, g.liner_id, span, retainer_density),
        "cap": (annulus_part(cap_od, g.liner_id, cap_face, al_density)
                + annulus_part(cap_od, cap_thread_dia, cap_thread_engagement, al_density)),
        "endplates": (annulus_part(g.endplate_od, bore, front_endplate, retainer_density)
                      + annulus_part(g.endplate_od, rear_hole, rear_endplate, retainer_density)),
    }


# --------------------------------------------------------------------------- adapters for DesignInputs
def geometry_for_design(inp, res) -> CouplingGeometry:
    """Reference geometry from the design inputs and the resolved magnet data (Calculator C18–C31)."""
    c, md, m = inp.coupling, inp.metal, res.model
    return coupling_geometry(npole=c.npole, inner_back_apothem=c.inner_back_apothem_mm, t_i=m.inner_thickness_mm,
                             w_i=m.inner_width_mm, t_o=m.outer_thickness_mm, w_o=m.outer_width_mm,
                             face_gap=md.face_gap_mm, bond_inner=md.bond_inner_mm, bond_outer=md.bond_outer_mm,
                             cup_wall_corner=md.cup_wall_corner_mm, bore=c.bore_mm, keyway_depth=c.keyway_depth_mm,
                             faceted=c.faceted == 1)


def rotating_parts_for_design(inp, res, cup_density: float | None = None) -> dict[str, Part]:
    """Solids built on the engine's own ring placement (Calculator C56, C60, C62), so a mass check
    isolates the mass formula; the placement itself is verified by the geometry checks.
    ``cup_density`` defaults to the steel density; the hub takes steel with back iron and the
    aluminium carrier density without, as the workbook states."""
    c, md, m = inp.coupling, inp.metal, res.model
    steel, al = md.steel_density_g_mm3, md.al_density_g_mm3
    return rotating_parts(npole=c.npole, inner_back_apothem=c.inner_back_apothem_mm, t_i=m.inner_thickness_mm,
                          w_i=m.inner_width_mm, L_i=m.inner_length_mm, outer_face_apothem=m.outer_face_apothem_mm,
                          t_o=m.outer_thickness_mm, w_o=m.outer_width_mm, L_o=m.outer_length_mm,
                          pocket_apothem=m.outer_back_apothem_mm + md.bond_outer_mm,
                          hub_apothem=c.inner_back_apothem_mm - md.bond_inner_mm, cup_od=m.cup_od_mm, bore=c.bore_mm,
                          cup_depth=md.cup_depth_mm, web=md.web_mm, hub_length=md.hub_length_mm,
                          boss_length=md.boss_length_mm, boss_od=md.boss_od_mm,
                          cup_density=steel if cup_density is None else cup_density,
                          hub_density=steel if c.backiron == 1 else al, faceted=c.faceted == 1)


def retainer_geometry_for_design(inp, res) -> RetainerGeometry:
    """Retainers sized on the reference ring extents: the inner ring's largest radius (block
    corners, or the arc OD) and the outer ring's smallest radius (the outer face centres)."""
    c, md, m = inp.coupling, inp.metal, res.model
    g = geometry_for_design(inp, res)
    outer_block = block_rectangle(g.outer_face_apothem, m.outer_thickness_mm, m.outer_width_mm, 0.0)
    outer_min = min_radius(outer_block) if c.faceted == 1 else g.outer_face_apothem
    return retainer_geometry(g.inner_corner_radius, outer_min, md.sleeve_mm, md.liner_mm,
                             md.sleeve_bedding_mm, md.liner_bedding_mm)


def retainer_parts_for_design(inp, res) -> dict[str, Part]:
    c, md = inp.coupling, inp.metal
    return retainer_parts(retainer_geometry_for_design(inp, res), span=md.retainer_span_mm,
                          retainer_density=md.sleeve_density_g_mm3, cap_od=md.cap_od_mm, cap_face=md.cap_axial_mm,
                          cap_thread_dia=md.cap_thread_dia_mm, cap_thread_engagement=md.cap_thread_engagement_mm,
                          al_density=md.al_density_g_mm3, bore=c.bore_mm, front_endplate=md.front_endplate_mm,
                          rear_endplate=md.rear_endplate_mm, rear_hole=md.rear_endplate_hole_mm)
