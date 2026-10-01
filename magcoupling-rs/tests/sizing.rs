//! Addendum A1: the axial length override (Task 5), inverse sizing (Task 6) and the space
//! claim (Task 7), end to end through `compute_all`.

use magcoupling::compute_all;
use magcoupling::engine::api::DesignInputs;

#[test]
fn a_length_override_keeps_the_measured_calibration_factor_and_moves_the_mass() {
    // Decision A2-3: the measured correction keys on the part names, as the workbook's C42
    // does, so a length override keeps it for the prototype's rings with no back iron (no
    // step in torque as a sized length passes the prototype's 12.7 mm). The magnets' mass
    // follows the length; the housing inputs do not (decision 28: no rule sizes them).
    let mut inputs = DesignInputs::default();
    inputs.coupling.backiron = 0;
    let at_part = compute_all(&inputs);
    inputs.coupling.magnets.axial_length_mm = Some(20.0);
    let long = compute_all(&inputs);
    assert_eq!(at_part.model.f_cal, at_part.calibration.f_cal_updated);
    assert_eq!(long.model.f_cal, at_part.model.f_cal);
    assert!((long.mass.magnets_g / at_part.mass.magnets_g - 20.0 / 12.7).abs() < 1e-12);
    assert_eq!(long.metal.axial_stack_mm, at_part.metal.axial_stack_mm);
    assert_eq!(
        long.retainers.retainer_span_mm,
        at_part.retainers.retainer_span_mm
    );
}
