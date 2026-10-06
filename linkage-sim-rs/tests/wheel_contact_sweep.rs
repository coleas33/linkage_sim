//! A wheel's contact point loads the linkage exactly like the same vertical
//! force locked at the wheel's hub, at every sample of a sweep (decision R-7:
//! a sample linkage stands in for the user's press, which the public repo does
//! not carry).

use linkage_sim_rs::core::body::BodyGeometry;
use linkage_sim_rs::forces::elements::{force_zone_application, ForceElement, ZoneAppMode};
use linkage_sim_rs::gui::samples::SampleMechanism;
use linkage_sim_rs::gui::AppState;

/// The zone's force at the wheel's contact point (`contact`) or locked at its
/// hub, then rebuild and sweep. The blueprint is the same each time, so the
/// built mechanism orders its bodies the same way and the sweeps compare
/// sample for sample (two separately loaded states need not: each body map
/// iterates in its own order, and the solver may land on another 2 pi turn).
fn sweep_driver_torques(state: &mut AppState, contact: bool) -> Vec<f64> {
    for force in &mut state.blueprint.as_mut().expect("blueprint").forces {
        if let ForceElement::ForceZone(zone) = force {
            zone.at_contact_point = contact;
        }
    }
    state.rebuild();
    state.compute_sweep();
    state.sweep_data.as_ref().expect("a sweep").driver_torques.clone().expect("driver torques")
}

#[test]
fn a_wheel_s_contact_point_loads_the_linkage_like_its_hub() {
    // The Parallelogram Press with its coupler's rectangle swapped for a 40 mm
    // wheel centred where the rectangle was, the zone grown over the whole
    // sweep, and a locked point at the hub (kept while the contact mode is on).
    let mut state = AppState::default();
    state.load_sample(SampleMechanism::ParallelogramPress);
    let bp = state.blueprint.as_mut().expect("blueprint");
    let coupler = bp.bodies.get_mut("coupler").expect("coupler");
    let hub = coupler.geometry.as_ref().expect("the sample's geometry").offset;
    coupler.geometry = Some(BodyGeometry::circle(0.04, hub).unwrap());
    for force in &mut bp.forces {
        if let ForceElement::ForceZone(zone) = force {
            zone.zone_min = [-10.0, -10.0];
            zone.zone_max = [10.0, 10.0];
            zone.body_local_app_point = Some([hub.x, hub.y]);
        }
    }
    let hub_torques = sweep_driver_torques(&mut state, false);
    let contact_torques = sweep_driver_torques(&mut state, true);
    // The rebuilt zone really is in contact mode, its point the top of the
    // wheel (against the press's downward force), 20 mm from the hub.
    let mech = state.mechanism.as_ref().expect("mechanism");
    let zone = mech
        .forces()
        .iter()
        .find_map(|f| if let ForceElement::ForceZone(z) = f { Some(z.clone()) } else { None })
        .expect("the zone");
    assert_eq!(zone.app_mode(), ZoneAppMode::Contact);
    let geo = mech.bodies()["coupler"].geometry.clone().expect("the wheel");
    let pose = mech.state().get_pose("coupler", &state.q);
    let hub = geo.centre_world(pose.0, pose.1, pose.2);
    let point = force_zone_application(&zone, &geo, pose).point.expect("a contact point");
    assert!(
        (point.x - hub.x).abs() < 1e-12 && (point.y - (hub.y + 0.02)).abs() < 1e-12,
        "{point:?} vs hub {hub:?}"
    );
    assert_eq!(hub_torques.len(), contact_torques.len());
    let mut compared = 0;
    for (h, c) in hub_torques.iter().zip(&contact_torques) {
        assert_eq!(h.is_finite(), c.is_finite(), "hub {h} vs contact {c}");
        if h.is_finite() {
            assert!((h - c).abs() <= 1e-9 * h.abs().max(1.0), "hub {h} vs contact {c}");
            compared += 1;
        }
    }
    assert!(compared > 300, "only {compared} samples solved");
}
