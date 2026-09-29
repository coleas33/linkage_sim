"""Sanity tests: the Task 7 reference modules against textbook cases with known answers (no engine involved).

Every test name contains 'sanity' so reference self-tests can be told apart from engine checks.
"""
import math
from types import SimpleNamespace

import pytest

from audit.common import MU0_EXACT, TOL_ALGEBRA, assert_all, flag_item, text_item
from audit.references import clamp_ref as cr
from audit.references import sweep_ref as sr
from audit.references import units_ref as u
from audit.references.planar import free_space_factor, harmonic_shear_stress, iron_backed_factor, wave_number

#: ISO 724 / ISO 965-2 print diameters to 0.001 mm, so a printed value is within 0.0005 mm of the exact one.
PRINT_TOL_MM = 0.0005


# --------------------------------------------------------------------------- clamp_ref: thread data and geometry
@pytest.mark.family("clamps")
def test_sanity_iso68_m10_diameters():
    """ISO 724 tabulates M10 x 1.5 as d2 = 9.026, d3 = 8.160, D1 = 8.376 mm (printed to 0.001 mm)."""
    assert_all([(f"M10x1.5 {what}", "ISO 724 table", got, table, PRINT_TOL_MM / table)
                for what, got, table in (("d2", cr.pitch_diameter(10, 1.5), 9.026),
                                         ("d3", cr.minor_diameter_external(10, 1.5), 8.160),
                                         ("D1", cr.minor_diameter_internal(10, 1.5), 8.376))])


@pytest.mark.family("clamps")
def test_sanity_iso898_stress_area():
    """ISO 898-1 tabulated stress areas (3 significant figures) are reproduced exactly: M8 36.6, M10 58.0,
    M12 84.3, M16 157 mm^2."""
    assert_all([(f"As M{d}", "ISO 898-1 table", cr.iso_stress_area(d, pitch), table, TOL_ALGEBRA)
                for d, pitch, table in ((8, 1.25, 36.6), (10, 1.5, 58.0), (12, 1.75, 84.3), (16, 2.0, 157.0))])


@pytest.mark.family("clamps")
def test_sanity_iso898_proof_load_m10():
    """ISO 898-1 proof loads for M10: 56 300 N (12.9), 48 100 N (10.9). The standard builds them as the tabulated
    As times Sp, printed to 3 significant figures; the same construction must match exactly."""
    as_table = cr.iso_stress_area(10, 1.5)
    assert_all([(f"proof load M10 {cls}", "ISO 898-1 table",
                 cr.round_sig(cr.preload_proof_share(1.0, cr.PROOF_STRESS_MPA[cls], as_table), 3), table, TOL_ALGEBRA)
                for cls, table in (("12.9", 56300.0), ("10.9", 48100.0))])


@pytest.mark.family("clamps")
def test_sanity_iso965_limits_transcription():
    """The transcribed 6g/6H rows are consistent with the ISO 68-1 basic profile: 6H has zero fundamental deviation,
    so its minimum pitch and minor diameters are the basic D2 and D1; 6g shifts major and pitch diameter by the same
    deviation es, so d - d_max = D2 - d2_max; every band is positive. Printed to 0.001 mm, so a difference of two
    printed values is good to 0.001 mm."""
    pitches = {s.name: s.pitch for s in cr.ISO_SCREWS} | {"M10": 1.5}
    items = []
    for name, lim in cr.THREAD_LIMITS_6G_6H.items():
        d, pitch = float(name[1:]), pitches[name]
        d2, d1 = cr.pitch_diameter(d, pitch), cr.minor_diameter_internal(d, pitch)
        items += [(f"{name} 6H D2 min = basic D2", "ISO 965-2 / ISO 68-1", lim.D2_min, d2, PRINT_TOL_MM / d2),
                  (f"{name} 6H D1 min = basic D1", "ISO 965-2 / ISO 68-1", lim.D1_min, d1, PRINT_TOL_MM / d1),
                  flag_item(f"{name} 6g deviation on major {d - lim.d_max:.3f} = on pitch "
                            f"{lim.D2_min - lim.d2_max:.3f}", "ISO 965-2",
                            abs((d - lim.d_max) - (lim.D2_min - lim.d2_max)) <= 2 * PRINT_TOL_MM),
                  flag_item(f"{name} every tolerance band positive", "ISO 965-2",
                            lim.d_min < lim.d_max <= d and lim.d2_min < lim.d2_max < d2
                            and lim.D2_min < lim.D2_max and lim.D1_min < lim.D1_max)]
    assert_all(items)


@pytest.mark.family("clamps")
def test_sanity_internal_thread_shear_area():
    """ISO 68-1 basic profile: the internal tooth is P/2 wide at the pitch diameter, P/4 (crest flat) at D1 and
    leaves a P/8 root gap at d, so the basic-size area is 0.875·pi·d·Le. The FED-STD-H28/2B expression with its
    printed constant 0.57735 (tan30 to 5 digits, so tol 1e-5) gives the same area. Minimum-material areas are
    below the basic-size areas for every size."""
    items = []
    for s in cr.ISO_SCREWS:
        d, p = s.d, s.pitch
        d2 = cr.pitch_diameter(d, p)
        for what, got, ref in (("width at D2", cr.internal_tooth_width(d2, d2, p), p / 2),
                               ("width at D1", cr.internal_tooth_width(cr.minor_diameter_internal(d, p), d2, p), p / 4),
                               ("width at d", cr.internal_tooth_width(d, d2, p), 7 * p / 8),
                               ("basic area / (pi·d·Le)",
                                cr.internal_thread_shear_area(d, d2, p, 1.0) / (math.pi * d), 0.875)):
            items.append((f"{s.name} {what}", "ISO 68-1 profile", got, ref, TOL_ALGEBRA))
        lim = cr.THREAD_LIMITS_6G_6H[s.name]
        n = 1.0 / p
        fed = math.pi * n * 1.0 * lim.d_min * (1.0 / (2.0 * n) + 0.57735 * (lim.d_min - lim.D2_max))
        a_min, a_basic = cr.min_material_shear_area(s, 1.0), cr.internal_thread_shear_area(d, d2, p, 1.0)
        items.append((f"{s.name} FED-STD-H28 expression", "FED-STD-H28/2B", a_min, fed, 1e-5))
        items.append(flag_item(f"{s.name} min-material area {a_min:.4f} < basic {a_basic:.4f} mm^2 per mm of Le",
                               "ISO 965-2", a_min < a_basic))
    assert_all(items)


@pytest.mark.family("clamps")
def test_sanity_clamp_friction_coefficient():
    """C = ∫p/∫p·cos: uniform pressure -> pi/2, cosine -> 4/pi, a sharp cos^2000 peak -> 1 (bounded by the Wallis
    ratio, 1 <= C <= 1 + 1/m). Uniform pressure also reproduces Shigley's press-fit torque T = (pi/2)·f·p·l·d^2
    (per jaw F = ∫p·cos·r·l = p·d·l). Midpoint rule with 20 000 panels: error ~ (pi/20000)^2 ~ 2.5e-8,
    tol 1e-7."""
    tol = 1e-7
    c_uni = cr.friction_torque_coefficient(lambda t: 1.0)
    c_cos = cr.friction_torque_coefficient(math.cos)
    c_line = cr.friction_torque_coefficient(lambda t: math.cos(t) ** 2000)
    p, d, l, mu = 10e6, 0.010, 0.010, 0.15
    assert_all([("uniform pressure C", "derivation", c_uni, math.pi / 2, tol),
                ("cosine pressure C", "derivation", c_cos, 4 / math.pi, tol),
                flag_item(f"near-line-contact C = {c_line:.6f} within [1, 1 + 1/2000]", "Wallis bound",
                          1.0 <= c_line <= 1.0 + 1.0 / 2000),
                ("press-fit torque", "Shigley (pi/2)·f·p·l·d^2", c_uni * mu * (p * d * l) * d,
                 math.pi / 2 * mu * p * l * d ** 2, tol)])


@pytest.mark.family("clamps")
def test_sanity_screws_fit_hand_cases():
    """Greedy placement, hand-worked: centres at margin + cbore/2 = 4 mm, then every 6.5 mm while the counterbore
    keeps the margin at the far end (last centre <= L - 1 - 3)."""
    assert_all([(f"screws fit in {L} mm", "hand count", cr.screws_fit(L, 1.0, 6.0, 6.5), expected, 0.0)
                for L, expected in ((20.0, 2), (14.5, 2), (14.4, 1), (7.0, 0), (26.0, 3))])


@pytest.mark.family("clamps")
def test_sanity_hex_key_through_m6_port():
    """M6 x 1 internal minor diameter D1 = 4.917 mm (ISO 965-2 6H minimum); a 4 mm key (4.619 across corners)
    passes, a 5 mm key (5.774) does not."""
    d1 = cr.minor_diameter_internal(6.0, 1.0)
    table = cr.THREAD_LIMITS_6G_6H["M6"].D1_min
    assert_all([("M6 D1", "ISO 965-2 table", d1, table, PRINT_TOL_MM / table),
                flag_item("4 mm key passes the M6 port", "s/cos30 < D1", cr.hex_key_passes_port(4.0)),
                flag_item("5 mm key is refused by the M6 port", "s/cos30 < D1", not cr.hex_key_passes_port(5.0))])


@pytest.mark.family("clamps")
def test_sanity_invalid_inputs_raise():
    """Non-positive spacing or per-screw torque would loop forever or divide by zero; both are rejected."""
    with pytest.raises(ValueError):
        cr.screws_fit(10.0, 1.0, 5.0, 0.0)
    with pytest.raises(ValueError):
        cr.screws_needed(7.5, 0.0)


@pytest.mark.family("clamps")
def test_sanity_clamp_geometry_345_triangle():
    """Boss R = 5, screw offset e = 3 (shaft 2 + ligament 1 + hole 2), head 2, no slit: half-chord 4, seat at 3
    (3-4-5 triangles), wall 1, counterbore depth 1. With 2 mm of required engagement the shortest length is
    grip + slit + 2 = 5 mm and the seat-to-far-OD chord is 3 + 0 + 4 = 7 mm, so 5, 6 and 7 mm screws are valid
    in 1 mm steps; the head of a larger screw (d_k = 6) does not fit and leaves no valid length."""
    screw = cr.IsoScrew("test", d=1.0, pitch=0.25, hole=2.0, head_dk=2.0, head_k=1.0, hex_s=1.0, tap_drill=0.75)
    setup = cr.ClampSetup(shaft_mm=2.0, boss_od_mm=10.0, slit_mm=0.0, ligament_mm=1.0, wall_min_mm=0.5, grip_min_mm=1.0,
                          axial_margin_mm=0.0, clamp_length_mm=10.0, engagement_x_d=2.0, preload_fraction=0.75,
                          strip_sf=1.0, friction=0.15, clamp_factor=1.0, nut_factor=0.2, screw_class="12.9",
                          alloy="7075-T6", max_torque_Nm=1.0, safety_factor=1.0, cbore_allowance_mm=0.5,
                          head_gap_mm=1.0, length_step_mm=1.0)
    geo = cr.clamp_geometry(setup, screw)
    big = cr.clamp_geometry(setup, cr.IsoScrew("big", 1.0, 0.25, 2.0, 6.0, 1.0, 1.0, 0.75))
    items = [(key, "3-4-5 triangle", geo[key], ref, TOL_ALGEBRA)
             for key, ref in (("offset_mm", 3.0), ("wall_out_mm", 1.0), ("head_fits", 1), ("grip_mm", 3.0),
                              ("thread_avail_mm", 4.0), ("cbore_depth_mm", 1.0), ("engagement_req_mm", 2.0))]
    items += [("engagement of a 5 mm screw", "hand sum", cr.engagement_achieved(setup, geo, 5.0), 2.0, TOL_ALGEBRA),
              ("longest length inside", "hand sum", cr.max_length_inside(setup, geo), 7.0, TOL_ALGEBRA),
              text_item("valid lengths", "hand list", str(cr.valid_screw_lengths(setup, geo)), str([5.0, 6.0, 7.0])),
              text_item("valid lengths when the head does not fit", "hand list",
                        str(cr.valid_screw_lengths(setup, big)), str([]))]
    assert_all(items)


# --------------------------------------------------------------------------- sweep_ref
def _setup(**over) -> sr.SweepSetup:
    base = dict(faceted=1, backiron=1, t_i_mm=3.0, w_i_mm=6.0, t_o_mm=3.0, w_o_mm=6.0, length_mm=12.0, br_i20_T=1.3,
                br_o20_T=1.3, alpha_per_C=0.0, op_temp_C=20.0, bond_inner_mm=0.05, bond_outer_mm=0.0,
                cup_wall_corner_mm=2.0, bore_mm=10.0, keyway_mm=1.7, c_end=0.0, mu0=MU0_EXACT, gear_ratio=1.0,
                gear_eff=1.0, required_floor_Nm=2.0, max_diameter_mm=40.0)
    base.update(over)
    return sr.SweepSetup(**base)


@pytest.mark.family("sweeps")
def test_sanity_geometry_at_corner_gap():
    """Faceted: inner face radius 10 + 3 = 13, corner radius hypot(13, 3) = 13.3417 mm, so a 1 mm corner gap is a
    1.3417 mm flat-face gap and the corner gap comes back unchanged. Arcs: no overhang, face gap = corner gap.
    The README wave number n·(poles/2)/R_gap equals planar.wave_number(n, 2·pi·R_gap/poles)."""
    s = _setup()
    geo = sr.geometry_at_corner_gap(s, 10, 10.0, 1.0)
    arc = sr.geometry_at_corner_gap(_setup(faceted=0), 10, 10.0, 1.0)
    face_gap = math.hypot(13.0, 3.0) - 13.0 + 1.0
    assert_all([("corner gap round trip", "construction", geo.corner_gap, 1.0, TOL_ALGEBRA),
                ("flat-face gap", "hypot(13, 3) - 13 + 1", geo.face_gap, face_gap, TOL_ALGEBRA),
                ("gap radius", "13 + face gap / 2", geo.gap_radius, 13.0 + face_gap / 2, TOL_ALGEBRA),
                ("arc face gap", "no overhang", arc.face_gap, 1.0, TOL_ALGEBRA),
                ("wave number forms agree", "README k", wave_number(3, 2 * math.pi * 0.0135 / 10),
                 3 * (10 / 2) / 0.0135, TOL_ALGEBRA)])


@pytest.mark.family("sweeps")
def test_sanity_geometry_factor_limits():
    """Thick magnets (k·t = 40): both factors -> e^(-k·g)/2. Thin magnets (k·t = 1e-4): free space -> (k·t)^2/2 and
    iron-backed -> k·t_i·t_o/(t_i + t_o + g), the flat magnetic-circuit field ratio. Thin limits carry O(k·t)
    Taylor error, tol 1e-3. A harmonic at pitch/2 displacement carries sin(n·pi/2): +1, -1, +1 for n = 1, 3, 5."""
    k, t, g = 1000.0, 0.040, 0.0014
    items = [(f"thick-magnet limit ({name})", "e^-kg/2", f(k, t, t, g), math.exp(-k * g) / 2, 1e-12)
             for name, f in (("iron", iron_backed_factor), ("free", free_space_factor))]
    k, t = 1.0, 1e-4
    items += [("thin-magnet limit (free)", "(kt)^2/2", free_space_factor(k, t, t, 0.0), (k * t) ** 2 / 2, 1e-3),
              ("thin-magnet limit (iron)", "k·ti·to/(ti+to+g)", iron_backed_factor(k, t, t, 0.5 * t),
               k * t * t / (2.5 * t), 1e-3)]
    pitch = 0.004
    items += [(f"sign of harmonic {n} at half a pitch", "sin(n·pi/2)",
               harmonic_shear_stress(1.0, 1.0, 1.0, wave_number(n, pitch), pitch / 2, 0.5), (-1.0) ** ((n - 1) // 2),
               1e-12) for n in (1, 3, 5)]
    assert_all(items)


@pytest.mark.family("sweeps")
def test_sanity_stated_min_inner_apothem():
    """10 poles, 6.35 mm block: 6.35/(2·tan 18°) + 0.05 = 9.8216 mm; 6 poles: wall rule 5 + 1.7 + 2.5 = 9.2 mm."""
    assert_all([(f"stated min apothem N={npole}", "hand value", sr.stated_min_inner_apothem(npole, 6.35, 10.0, 1.7),
                 ref, TOL_ALGEBRA)
                for npole, ref in ((10, 6.35 / (2 * math.tan(math.pi / 10)) + 0.05), (6, 9.2))])


@pytest.mark.family("sweeps")
def test_sanity_sweep_status_priority():
    """Hand-built cases hit each status once, in priority order; a pull-out equal to the floor is not below it."""
    s = _setup()
    cases = ((5.2, 20.0, 30.0, 5.0, sr.STATUS_INNER_NARROW),
             (6.5, 5.85, 30.0, 5.0, sr.STATUS_OUTER_NARROW),
             (6.5, 7.8, 41.0, 5.0, sr.STATUS_OD),
             (6.5, 7.8, 39.0, 1.9, sr.STATUS_BELOW_MIN),
             (6.5, 7.8, 39.0, 2.0, sr.STATUS_NOMINAL))
    assert_all([text_item(f"status for flats {fi}/{fo} mm, OD {od} mm, pull-out {x} N·m", "priority rule",
                          sr.sweep_status(s, SimpleNamespace(inner_flat_width=fi, outer_flat_width=fo, cup_od=od), x),
                          expected)
                for fi, fo, od, x, expected in cases])


# --------------------------------------------------------------------------- units_ref
@pytest.mark.family("constants")
def test_sanity_magnetic_pressure_one_tesla():
    """B^2/(2·mu0) at 1 T is 397 887 Pa (textbook magnetic pressure) and carries the dimensions of Pa."""
    q = (1.0 * u.TESLA) ** 2 / (2 * MU0_EXACT * u.H_PER_M)
    assert_all([("magnetic pressure at 1 T", "B^2/2mu0", q.to(u.PA), 397887.3577, 1e-9)])


@pytest.mark.family("constants")
def test_sanity_copper_skin_depth():
    """Copper, sigma = 5.8e7 S/m, at 50 Hz: delta = sqrt(2/(omega·mu0·sigma)) = 66.1/sqrt(f) mm = 9.348 mm
    (Hayt & Buck, Engineering Electromagnetics). Checks the S/m and H/m units; the 66.1 constant has 3 figures."""
    omega = 2 * math.pi * 50 * u.RAD_PER_S
    delta_squared = 2 / (omega * (MU0_EXACT * u.H_PER_M) * (5.8e7 * u.S_PER_M))
    delta_mm = math.sqrt(delta_squared.to(u.M ** 2)) * 1000
    assert_all([("copper skin depth at 50 Hz", "66.1/sqrt(f) mm", delta_mm, 66.1 / math.sqrt(50), 1e-3)])


@pytest.mark.family("constants")
def test_sanity_unit_identities():
    """1 MPa·mm^2 = 1 N; 1 N·m = 1 J; 1 W·s = 1 J; 1 rpm = 2·pi/60 rad/s; 1 g·J/(kg·K) = 1e-3 J/K;
    1 T·A/m = 1 Pa (B·H is an energy density)."""
    assert_all([(what, "SI definition", got, ref, TOL_ALGEBRA)
                for what, got, ref in (("MPa·mm^2 in N", (u.MPA * u.MM ** 2).to(u.N), 1.0),
                                       ("N·m in J", u.NM.to(u.J), 1.0),
                                       ("W·s in J", (u.W * u.S).to(u.J), 1.0),
                                       ("rpm in rad/s", u.RPM.to(u.RAD_PER_S), 2 * math.pi / 60),
                                       ("g·J/(kg·K) in J/K", (u.GRAM * u.J_PER_KG_K).to(u.J_PER_K), 1e-3),
                                       ("T·A/m in Pa", (u.TESLA * u.A_PER_M).to(u.PA), 1.0))])


@pytest.mark.family("constants")
def test_sanity_dimension_errors_raise():
    """Mixing dimensions is rejected: m + s, expressing a torque in pascals, and a dimensional sinh argument."""
    with pytest.raises(u.DimensionError):
        _ = u.M + u.S
    with pytest.raises(u.DimensionError):
        u.NM.to(u.PA)
    with pytest.raises(u.DimensionError):
        (3.0 * u.MM).dimensionless()
