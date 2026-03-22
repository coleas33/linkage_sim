//! Integration test: ParallelogramActuator sample builds, solves, and
//! round-trips through JSON with compound force expansion.

use linkage_sim_rs::forces::elements::ForceElement;
use linkage_sim_rs::gui::samples::{build_sample, SampleMechanism};
use linkage_sim_rs::io::serialization::{load_mechanism_unbuilt, save_mechanism};

#[test]
fn parallelogram_actuator_builds_and_solves() {
    let (mech, q0) = build_sample(SampleMechanism::ParallelogramActuator);
    assert!(mech.is_built(), "sample should build successfully");
    assert!(!q0.is_empty(), "initial state vector should be non-empty");
}

#[test]
fn parallelogram_actuator_has_actuator_force() {
    let (mech, _q0) = build_sample(SampleMechanism::ParallelogramActuator);
    let actuators: Vec<_> = mech
        .forces()
        .iter()
        .filter(|f| matches!(f, ForceElement::LinearActuator(_)))
        .collect();
    assert_eq!(actuators.len(), 1, "should have exactly one linear actuator");
}

#[test]
fn parallelogram_actuator_has_crank_mount_point() {
    let (mech, _q0) = build_sample(SampleMechanism::ParallelogramActuator);
    let crank = &mech.bodies()["crank"];
    assert!(
        crank.mount_points.contains_key("M"),
        "crank should have mount point 'M' at midpoint"
    );
}

#[test]
fn parallelogram_actuator_json_round_trip_expands_compound() {
    let (mech, _q0) = build_sample(SampleMechanism::ParallelogramActuator);
    let json = save_mechanism(&mech).expect("save should succeed");
    let reloaded = load_mechanism_unbuilt(&json).expect("reload should succeed");

    // Compound expansion should create cylinder + rod bodies.
    assert!(
        reloaded.bodies().contains_key("force_0_cyl"),
        "compound cylinder should be created on reload"
    );
    assert!(
        reloaded.bodies().contains_key("force_0_rod"),
        "compound rod should be created on reload"
    );
}
