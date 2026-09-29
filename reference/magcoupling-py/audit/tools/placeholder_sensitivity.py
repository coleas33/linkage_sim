"""Placeholder-input sensitivities at the default design, for the findings report's 'Placeholder inputs' table.

Report data, not a check: nothing here passes or fails. Task 6 re-derives every engine row listed below exactly
(audit/tests/test_slip_thermal.py), so comparing derivatives would add no check; what the report needs is how far
each placeholder moves a result. Each derivative is a central difference on the engine
(audit.references.slip_thermal.central_difference, relative step 1e-4, sanity-tested), and the last column is the
linearized change of the result for a +10 % change of the input.

Run from reference/magcoupling-py:  ./.venv/Scripts/python -m audit.tools.placeholder_sensitivity
"""
from __future__ import annotations

from audit.common import defaults, run, vary
from audit.references.slip_thermal import central_difference

STEADY_HIGH = ("steady magnet temperature, high case (C)", "Temperature design!C149", lambda r: r.temperature.thermal.steady_high_C)
PEAK = ("hot-day peak magnet temperature (C)", "Temperature design!C180", lambda r: r.temperature.magnet_life.peak_C)
CRITICAL_DRAG = ("critical drag (N m)", "Temperature design!C153", lambda r: r.temperature.thermal.critical_drag_Nm)
SLIP_DUTY = ("slip duty (-)", "Temperature design!C174", lambda r: r.temperature.slip_life.slip_duty)

PLACEHOLDERS = [  # (input path, why it is a placeholder, [(result, cell, getter), ...])
    ("temperature.thermal.conductance_W_K", "conductance to ambient not measured", [
        STEADY_HIGH,
        ("thermal time constant (s)", "Temperature design!C143", lambda r: r.temperature.thermal.time_constant_s),
        CRITICAL_DRAG, PEAK]),
    ("temperature.duty.driving_rise_C", "housing air, sun and gearbox heat not measured", [
        PEAK, CRITICAL_DRAG,
        ("pull-out at the hot-day start (N m)", "Temperature design!C184", lambda r: r.temperature.magnet_life.torque_hot_day_Nm),
        ("margin above the hot-day start (C)", "Temperature design!C15", lambda r: r.temperature.summary.margin_hot_day_C)]),
    ("metal.slip_event_s", "slip event length assumed", [
        ("life slip rotations (rev)", "Temperature design!C166", lambda r: r.temperature.slip_life.rotations),
        SLIP_DUTY, PEAK]),
    ("temperature.duty.life_hours", "operating hours over life assumed", [SLIP_DUTY, PEAK]),
    ("temperature.adhesive_life.hot_strength_retained", "hot adhesive strength not measured", [
        ("hot fatigue margin (-)", "Temperature design!C196", lambda r: r.temperature.adhesive_life.hot_fatigue_margin)]),
    ("temperature.slip_loss.high_multiplier", "judgement band on the slip-loss estimate", [STEADY_HIGH, PEAK]),
    ("temperature.slip_loss.end_factor", "judgement end factor for the shells and the cap", [
        ("total slip loss, estimate (W)", "Temperature design!C130", lambda r: r.temperature.slip_loss.total_W), STEADY_HIGH]),
]


def input_value(inp, path: str) -> float:
    """Value of a dotted-path input, e.g. 'temperature.duty.driving_rise_C'."""
    obj = inp
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def sensitivity(path: str, getter, rel_step: float = 1e-4) -> float:
    """d(result)/d(input) at the default design, by a central difference on the engine."""
    return central_difference(lambda x: getter(run(vary(defaults(), {path: x}))), input_value(defaults(), path), rel_step)


def main() -> None:
    base = run()
    print("| Input | Default | Why a placeholder | Result | Cell | Value at defaults | d result / d input | Change for +10 % input |")
    print("|---|---|---|---|---|---|---|---|")
    for path, why, results in PLACEHOLDERS:
        x0 = input_value(defaults(), path)
        for label, cell, getter in results:
            d = sensitivity(path, getter)
            print(f"| `{path}` | {x0:g} | {why} | {label} | {cell} | {getter(base):.6g} | {d:.6g} | {0.1 * x0 * d:+.4g} |")


if __name__ == "__main__":
    main()
